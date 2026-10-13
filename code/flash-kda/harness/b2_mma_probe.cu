// P7-46: the *second* slice of the K2-shaped CUDA port of the fused B2/B3 -- the
// B2 (`wy_dqkg`) V-reduction GEMM core, this time **keeping `mma.sync` tensor core**
// as §47.5/§48.4 require, and contrasting P7-44's dead-end SIMT version.
//
// P7-44 (`b2_core_probe.cu`) tested reading (b) of §46.6's "把 dq/dk/dw 累加器按 warp
// 分片常驻寄存器": plain SIMT fp32 FMA.  It lost 3-5x to the identical-op Triton
// kernel (which is itself only 1.25x above its DRAM floor), pinning the answer to
// reading (a): *register-sharded accumulators **alongside** tensor core* (B1 §19-24
// recipe).  P7-45 (§48) then showed the fused kernel's ~1.9x headroom is not bytes
// and not occupancy but **"big operands (h/dh) kept register-resident -> spill/issue
// starvation"**, and named the first action:
//
//     "把大操作数（尤其 h/dh）从寄存器保活里拿掉、只留 warp 分片的小累加器".
//
// This prototype is that action on the isolated dominant GEMMs of B2:
//   dq[t,k] = sum_v do[t,v]*h[k,v]     (do  @ b_h^T)
//   dk[t,k] = sum_v v_new[t,v]*dh[k,v] (v_new @ b_dh^T)
//   dw[t,k] = sum_v dv[t,v]*h[k,v]     (dv  @ b_h^T)
//   dgk[k]  = sum_v h[k,v]*dh[k,v]
//
// Structure (the K2 shape, per chunk `(i_t, i_bh)` with grid `(NT, B*H)`):
//   * block = 8 warps; warp `w` owns the **N (K) output columns** [w*16, w*16+16).
//   * `h/dh/do/vn/dv` are **streamed through smem in V-tiles** (`BVK`) via `cp.async`
//     double buffering -- **no `[V,K]` operand is kept register-resident**; each warp
//     only carries its small `[16,16]` fp32 accumulators (`dq/dk/dw`) + one `dgk` scalar.
//   * the contraction is done with raw `mma.sync.m16n8k8.tf32` (matches `tl.dot`'s
//     default tf32 on fp32 inputs), fed by direct smem reads of the A/B fragments.
//
// This file implements *only* the GEMM core (like P7-44) so it is an apples-to-apples
// head-to-head vs the identical-op Triton kernel; the B2 finalize (`dkgb=A@dw`,
// `dA=dv@v^T`), `dv2`/`db` and the intra fold are not here (see docs/07 §47.6).
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b2_mma_probe.cu -o /tmp/b2_mma.so
// Host entry `extern "C" b2mma_launch(...)`, called from check_bwd_b2_mma.py via ctypes.
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cuda.h>
#include <cstdint>
#include <cstdio>

#define DEV __device__ __forceinline__

#if B2M_NOASYNC
DEV void cp16(void* dst, const void* src) {
    *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
}
#else
DEV void cp16(void* dst, const void* src) { __pipeline_memcpy_async(dst, src, 16); }
#endif

constexpr int KD = 128;   // K = D (output N)
constexpr int VD = 128;   // V = D (contraction K')
constexpr int BT = 16;    // chunk rows (M)
constexpr int NW = 8;     // warps
constexpr int BW = 16;    // output columns per warp = KD / NW

#ifndef B2M_BVK
#define B2M_BVK 32
#endif
#ifndef B2M_NS
#define B2M_NS 2
#endif
// P7-47 diagnostics:
//   B2M_NODGK : skip the `dgk` column reduction entirely (dgk stored as 0) -- isolates
//               whether the 8.81M smem bank conflicts live in the dgk loop or in the
//               direct mma-fragment reads (§49.4.3).
//   B2M_DGKCF : conflict-free `dgk` reduction -- 8 kcols per warp instruction (see below).
#ifndef B2M_NODGK
#define B2M_NODGK 0
#endif
#ifndef B2M_DGKCF
#define B2M_DGKCF 0
#endif
// P7-48 diagnostics: localize §49.4.3's "L2 bytes +35% vs the identical-op Triton core"
// (3.03 vs 2.24 GB), which §50.5 left as the open lead.
//   B2M_NOHDH : skip the h/dh global cp.async (zero-fill smem) -> bytes side of h/dh.
//   B2M_NODOV : skip the do/vn/dv global cp.async (zero-fill smem).
//   B2M_VECST : write the three outputs with 64-bit (`float2`) stores instead of the
//               default two scalar 32-bit `STG.E` (candidate fix for half-sector stores).
#ifndef B2M_NOHDH
#define B2M_NOHDH 0
#endif
#ifndef B2M_NODOV
#define B2M_NODOV 0
#endif
#ifndef B2M_VECST
#define B2M_VECST 0
#endif
//   B2M_NOASYNC : replace the 16B `cp.async` (`LDGSTS.E.BYPASS.128`) with a synchronous
//                 16B `float4` load+store -> same bytes, normal (L1-allocating) ld path.
//                 Isolates whether the 1.5x L2-read-sector amplification is specific to
//                 the cp.async/bypass path or to the access pattern itself.
#ifndef B2M_NOASYNC
#define B2M_NOASYNC 0
#endif
//   B2M_TMA (P7-49): stage the operands with `cp.async.bulk.tensor` (TMA) instead of
//                 `cp.async`.  §51.4 pinned the read-side +0.64 GB to the `cp.async`
//                 `.BYPASS` path (L1-bypass loses cross-lane 128 B coalescing) and named
//                 "读侧下一杠杆 = TMA 输入".  TMA issues per-tile descriptor copies off
//                 the LSU, into a SWIZZLE_128B smem layout (128 B = 32 fp32 inner rows),
//                 using a per-stage `mbarrier` (`complete_tx::bytes`) handshake.
//                 P7-50 tried BVK==64 with SWIZZLE_NONE (dense 256 B rows) -- bit-correct
//                 but 2.6-2.8x slower (dense 256 B stride = 64 words aliases every smem
//                 bank).  P7-51 (this revision) instead splits each operand tile into
//                 `NAT = BVK/32` independent **128 B swizzle atoms** (each a separate TMA
//                 box of inner width 32), so BVK==64 keeps the bank-conflict-free swizzle
//                 while still doing the best (BVK=64) tiling -- the two wins §52.6/§53.5
//                 said could not coexist.
#ifdef B2M_TMA
#include <cuda.h>
#define B2M_TMA_ON 1
#else
#define B2M_TMA_ON 0
#endif
// P7-52: the *first non-GEMM segment* of B2 in CUDA -- the output branch
// (`i_k==0`):   dvb[o_A,v] = sum_{o_t} A[o_A,o_t] * dv[o_t,v]   (== A @ dv)
//               dv2[o_A,v] = beta[o_A] * dvb[o_A,v]             (the complete B2 output `dv2`)
// `dv2` is a *complete* B2 output (never touched by the gate chain afterwards, see
// check_bwd_dqkg_kernel.py:294/297), so it is verifiable standalone.  `dvb` is stored too so
// the A@dv GEMM can be checked directly.  The operand roles differ from the GEMM core: here
// the *contraction* is over chunk rows `o_t` and the *output row* is also a chunk row
// `o_A`, so A is read from `Akk` with a plain index remap (no smem transpose needed) and the
// already-staged `dv` tile serves as the B operand.  See docs/07 §55.
#ifndef B2M_OUT
#define B2M_OUT 0
#endif
constexpr int BVK = B2M_BVK;   // V-tile (contraction) depth staged per pipeline stage
constexpr int NS  = B2M_NS;    // pipeline stages
#if B2M_TMA_ON
// Each operand is staged as NAT atom-major `[NAT][rows][32]` sub-buffers, each written by
// its own 128 B-swizzled TMA box (inner width 32).  BVK==32 -> NAT=1 (identical to P7-49).
constexpr int NAT = BVK / 32;
static_assert(BVK == 32 || BVK == 64, "TMA supports BVK==32 or BVK==64 (128 B atoms)");
#else
constexpr int BVP = BVK + 4;   // padded row stride (16B keeps 4-float cp.async aligned)
#endif

static_assert(VD % BVK == 0, "BVK must divide V=128");

struct alignas(128) Stage {
#if B2M_TMA_ON
    float do_[NAT][BT][32];   // do  [t, v], atom-major (each atom = 128 B-swizzled [BT][32])
    float vn [NAT][BT][32];   // v_new[t, v]
    float dv [NAT][BT][32];   // dv  [t, v]
    float h  [NAT][KD][32];   // h  [k, v]  (rows = output N index, cols = contraction v)
    float dh [NAT][KD][32];   // dh [k, v]
#else
    float do_[BT][BVP];   // do  [t, v]
    float vn [BT][BVP];   // v_new[t, v]
    float dv [BT][BVP];   // dv  [t, v]
    float h  [KD][BVP];   // h  [k, v]  (rows = output N index, cols = contraction v)
    float dh [KD][BVP];   // dh [k, v]
#endif
};
struct alignas(128) Shared {
    Stage st[NS];
#if B2M_TMA_ON
    uint64_t full[NS];    // per-stage TMA-completion mbarrier (arrive count 1)
#endif
};

// logical element read from a stage tile.  Non-TMA: padded row-major.  TMA: the operand is
// `[NAT][rows][32]` atom-major; pick the atom by `col>>5`, then apply the SWIZZLE_128B
// decode (Swizzle<3,4,3>: byte bits [4:6] ^= bits [7:9]) *within that atom*.  Since the
// atom stride (`rows*128` bytes) is a multiple of 8 rows, the decode reduces to the same
// `row & 7` XOR as the single-atom BVK==32 case.
DEV float rd(const float* buf, int row, int col, int rows) {
#if B2M_TMA_ON
    const int atom = col >> 5, cc = col & 31;
    int b = ((atom * rows + row) * 32 + cc) * 4;
    b = b ^ (((b >> 7) & 7) << 4);
    return *(const float*)((const char*)buf + b);
#else
    (void)rows;
    return buf[row * BVP + col];
#endif
}

#if B2M_TMA_ON
// ---- mbarrier / TMA (same primitives as b1_tma_probe.cu) --------------------
DEV uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
DEV void mbar_init(uint64_t* b, uint32_t cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(b)), "r"(cnt));
}
DEV void mbar_arrive_tx(uint64_t* b, uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
                 ::"r"(saddr(b)), "r"(bytes) : "memory");
}
DEV void mbar_wait(uint64_t* b, int ph) {
    asm volatile(
        "{\n\t.reg .pred P;\n"
        "LAB_%=:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n\t"
        "@!P bra LAB_%=;\n}\n" ::"r"(saddr(b)), "r"(ph) : "memory");
}
DEV void tma4d_load(const CUtensorMap* m, void* sm, int c0, int c1, int c2, int c3,
                    uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4,%5}], [%6];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(saddr(bar))
        : "memory");
}
DEV void tma5d_load(const CUtensorMap* m, void* sm, int c0, int c1, int c2, int c3, int c4,
                    uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2,%3,%4,%5,%6}], [%7];\n"
        ::"r"(saddr(sm)), "l"(m), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
          "r"(saddr(bar))
        : "memory");
}
#endif

DEV uint32_t to_tf32(float x) {
    uint32_t r;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(r) : "f"(x));
    return r;
}
// D = A*B + C, m16n8k8, A row-major, B col-major, tf32 x tf32 -> fp32
DEV void mma8(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---------------------------------------------------------------- stage prefetch
#if !B2M_TMA_ON
DEV void prefetch(Stage* st, const float* __restrict__ v, const float* __restrict__ vn,
                  const float* __restrict__ h, const float* __restrict__ dh,
                  const float* __restrict__ do_, const float* __restrict__ dv,
                  int n, int hh, int T, int H, int NT, int c, int v0) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int t0 = c * BT;
    // do/vn/dv [BT, BVK]  (token rows; zero the pad rows for a non-divisible tail)
    for (int idx = tid; idx < BT * (BVK / 4); idx += nthr) {
        int row = idx / (BVK / 4), cg = idx % (BVK / 4), t = t0 + row;
        float* pd = &st->do_[row][cg * 4];
        float* pv = &st->vn [row][cg * 4];
        float* pw = &st->dv [row][cg * 4];
#if B2M_NODOV
        for (int j = 0; j < 4; ++j) { pd[j] = 0.f; pv[j] = 0.f; pw[j] = 0.f; }
#else
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + hh) * VD + v0 + cg * 4;
            cp16(pd, do_ + off);
            cp16(pv, vn  + off);
            cp16(pw, dv  + off);
        } else {
            for (int j = 0; j < 4; ++j) { pd[j] = 0.f; pv[j] = 0.f; pw[j] = 0.f; }
        }
#endif
    }
    // h/dh [KD, BVK]  (chunk-granular, always full)
    const int64_t hbase = ((int64_t)(n * NT + c) * H + hh) * KD * VD;
    for (int idx = tid; idx < KD * (BVK / 4); idx += nthr) {
        int row = idx / (BVK / 4), cg = idx % (BVK / 4);
        const int64_t off = hbase + (int64_t)row * VD + v0 + cg * 4;
#if B2M_NOHDH
        for (int j = 0; j < 4; ++j) { st->h[row][cg * 4 + j] = 0.f; st->dh[row][cg * 4 + j] = 0.f; }
#else
        cp16(&st->h [row][cg * 4], h  + off);
        cp16(&st->dh[row][cg * 4], dh + off);
#endif
    }
}
#endif  // !B2M_TMA_ON

#if B2M_TMA_ON
// Single-thread TMA issue for one V-tile stage: do/vn/dv [BT,BVK] + h/dh [KD,BVK].  OOB
// token rows (non-divisible T) are zero-filled by TMA automatically.  `bar` is the stage's
// `full` mbarrier; `expect_tx` is the *full box* byte count (TMA counts zero-filled bytes).
// Each operand is issued as `NAT` 128 B (32 fp32) inner-width boxes into its own atom.
DEV void issue_tma(Stage* st, const CUtensorMap* dom, const CUtensorMap* vnm,
                   const CUtensorMap* dvm, const CUtensorMap* hm, const CUtensorMap* dhm,
                   int n, int hh, int c, int v0, uint64_t* bar) {
    const int t0 = c * BT;
    mbar_arrive_tx(bar, (uint32_t)((3 * BT + 2 * KD) * BVK * 4));
    #pragma unroll
    for (int a = 0; a < NAT; ++a) {
        const int va = v0 + a * 32;
        tma4d_load(dom, &st->do_[a][0][0], va, t0, hh, n, bar);
        tma4d_load(vnm, &st->vn [a][0][0], va, t0, hh, n, bar);
        tma4d_load(dvm, &st->dv [a][0][0], va, t0, hh, n, bar);
        tma5d_load(hm,  &st->h  [a][0][0], va, 0, hh, c, n, bar);
        tma5d_load(dhm, &st->dh [a][0][0], va, 0, hh, c, n, bar);
    }
}
#endif

// ---------------------------------------------------------------------- kernel
__global__ void __launch_bounds__(NW * 32)
b2mma_kernel(
#if B2M_TMA_ON
             const __grid_constant__ CUtensorMap dom, const __grid_constant__ CUtensorMap vnm,
             const __grid_constant__ CUtensorMap dvm, const __grid_constant__ CUtensorMap hm,
             const __grid_constant__ CUtensorMap dhm,
#endif
             const float* __restrict__ v, const float* __restrict__ vn,
             const float* __restrict__ h, const float* __restrict__ dh,
             const float* __restrict__ do_, const float* __restrict__ dv,
             float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dw,
             float* __restrict__ dgk,
#if B2M_OUT
             const float* __restrict__ A4, const float* __restrict__ beta,
             float* __restrict__ dv2, float* __restrict__ dvb,
#endif
             int B, int T, int H, int NT) {
    extern __shared__ char smem_raw[];
    Shared* sh = reinterpret_cast<Shared*>(smem_raw);

    const int i_t = blockIdx.x, i_bh = blockIdx.y;
    const int n = i_bh / H, hh = i_bh % H;
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = l >> 2, t = l & 3;   // groupID, threadID-in-group

    float accq[2][4], acck[2][4], accw[2][4];
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) { accq[nt][x] = 0.f; acck[nt][x] = 0.f; accw[nt][x] = 0.f; }
    float gpart = 0.f;
    float gpart1 = 0.f;   // B2M_DGKCF second kcol half (kcol = warp*BW + 8 + c)
#if B2M_OUT
    float acc2[2][4];     // dvb[o_A, v] output tile (m16 x 2*n8), warp-private V columns
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) acc2[nt][x] = 0.f;
    const int64_t a_base = ((int64_t)(n * T + i_t * BT) * H + hh) * (int64_t)BT;
#endif

    const int nvt = VD / BVK;
#if B2M_TMA_ON
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int s = 0; s < NS; ++s) mbar_init(&sh->full[s], 1);
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int s = 0; s < NS - 1; ++s)
            if (s < nvt)
                issue_tma(&sh->st[s], &dom, &vnm, &dvm, &hm, &dhm, n, hh, i_t, s * BVK,
                          &sh->full[s]);
    }
    for (int it = 0; it < nvt; ++it) {
        if (threadIdx.x == 0) {
            const int snext = it + NS - 1;
            if (snext < nvt)
                issue_tma(&sh->st[snext % NS], &dom, &vnm, &dvm, &hm, &dhm, n, hh, i_t,
                          snext * BVK, &sh->full[snext % NS]);
        }
        mbar_wait(&sh->full[it % NS], (it / NS) & 1);
        __syncthreads();

        const Stage* st = &sh->st[it % NS];
#else
    for (int s = 0; s < NS - 1; ++s) {
        if (s < nvt) prefetch(&sh->st[s], v, vn, h, dh, do_, dv, n, hh, T, H, NT, i_t, s * BVK);
        __pipeline_commit();
    }

    for (int it = 0; it < nvt; ++it) {
        const int snext = it + NS - 1;
        if (snext < nvt)
            prefetch(&sh->st[snext % NS], v, vn, h, dh, do_, dv, n, hh, T, H, NT, i_t, snext * BVK);
        __pipeline_commit();
        __pipeline_wait_prior(NS - 1);
        __syncthreads();

        const Stage* st = &sh->st[it % NS];
#endif
        #pragma unroll
        for (int kk = 0; kk < BVK / 8; ++kk) {
            const int kc = kk * 8;
            // A fragments (m16 x k8) from do/vn/dv: A[row=g/g+8][col=t/t+4]
            uint32_t aq[4], ak[4], aw[4];
            aq[0] = to_tf32(rd((const float*)st->do_, g    , kc + t, BT));
            aq[1] = to_tf32(rd((const float*)st->do_, g + 8, kc + t, BT));
            aq[2] = to_tf32(rd((const float*)st->do_, g    , kc + t + 4, BT));
            aq[3] = to_tf32(rd((const float*)st->do_, g + 8, kc + t + 4, BT));
            ak[0] = to_tf32(rd((const float*)st->vn, g    , kc + t, BT));
            ak[1] = to_tf32(rd((const float*)st->vn, g + 8, kc + t, BT));
            ak[2] = to_tf32(rd((const float*)st->vn, g    , kc + t + 4, BT));
            ak[3] = to_tf32(rd((const float*)st->vn, g + 8, kc + t + 4, BT));
            aw[0] = to_tf32(rd((const float*)st->dv, g    , kc + t, BT));
            aw[1] = to_tf32(rd((const float*)st->dv, g + 8, kc + t, BT));
            aw[2] = to_tf32(rd((const float*)st->dv, g    , kc + t + 4, BT));
            aw[3] = to_tf32(rd((const float*)st->dv, g + 8, kc + t + 4, BT));
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int krow = warp * BW + nt * 8 + g;   // output column = k = n index
                uint32_t bh[2], bd[2];
                bh[0] = to_tf32(rd((const float*)st->h , krow, kc + t, KD));
                bh[1] = to_tf32(rd((const float*)st->h , krow, kc + t + 4, KD));
                bd[0] = to_tf32(rd((const float*)st->dh, krow, kc + t, KD));
                bd[1] = to_tf32(rd((const float*)st->dh, krow, kc + t + 4, KD));
                mma8(accq[nt], aq, bh, accq[nt]);
                mma8(acck[nt], ak, bd, acck[nt]);
                mma8(accw[nt], aw, bh, accw[nt]);
            }
        }
#if !B2M_NODGK
  #if B2M_DGKCF
        // Conflict-free `dgk` column reduction.  smem word addr = kcol*BVP + vv with
        // BVP=36 -> bank = (4*kcol + vv) & 31; 16 consecutive kcols therefore alias as
        // (kcol, kcol+8).  Use only 8 kcols per instruction: c = l&7 (kcol low bits),
        // s = l>>3 (the vv subclass), so banks = 4*c + s cover 0..31 exactly.
        {
            const int c = l & 7, s = l >> 3;
            #pragma unroll
            for (int ph = 0; ph < 2; ++ph) {
                const int kcol = warp * BW + ph * 8 + c;
                float p = 0.f;
                for (int vv = s; vv < BVK; vv += 4)
                    p += rd((const float*)st->h, kcol, vv, KD) * rd((const float*)st->dh, kcol, vv, KD);
                if (ph == 0) gpart += p; else gpart1 += p;
            }
        }
  #else
        // baseline dgk column reduction: lane l handles column warp*BW + (l&15), split v by l>>4
        {
            const int kcol = warp * BW + (l & 15);
            float p = 0.f;
            for (int vv = (l >> 4); vv < BVK; vv += 2)
                p += rd((const float*)st->h, kcol, vv, KD) * rd((const float*)st->dh, kcol, vv, KD);
            gpart += p;
        }
  #endif
#endif
#if B2M_OUT
        // ---- output branch (P7-52): dvb = A @ dv, contraction over chunk rows o_t ----
        // Warp `w` owns V columns [w*BW, w*BW+BW); only the stage whose V-tile contains
        // them contributes.  A[o_A,o_t] = Akk[token=i_t*BT+o_t, o_A] (index remap, no
        // transpose); the B operand is the already-staged `dv` tile.
        {
            const int local0 = warp * BW - it * BVK;
            if (local0 >= 0 && local0 + BW <= BVK) {
                #pragma unroll
                for (int kk = 0; kk < BT / 8; ++kk) {
                    const int kc = kk * 8;
                    const int ot0 = kc + t, ot1 = kc + t + 4;
                    const float a00 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g]     : 0.f;
                    const float a10 = (i_t * BT + ot0) < T ? A4[a_base + (int64_t)ot0 * (H * BT) + g + 8] : 0.f;
                    const float a01 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g]     : 0.f;
                    const float a11 = (i_t * BT + ot1) < T ? A4[a_base + (int64_t)ot1 * (H * BT) + g + 8] : 0.f;
                    const uint32_t af[4] = {to_tf32(a00), to_tf32(a10), to_tf32(a01), to_tf32(a11)};
                    #pragma unroll
                    for (int nt = 0; nt < 2; ++nt) {
                        const int lc = local0 + nt * 8 + g;
                        uint32_t bf[2];
                        bf[0] = to_tf32(rd((const float*)st->dv, ot0, lc, BT));
                        bf[1] = to_tf32(rd((const float*)st->dv, ot1, lc, BT));
                        mma8(acc2[nt], af, bf, acc2[nt]);
                    }
                }
            }
        }
#endif
        __syncthreads();
    }

    // ---- store ----
    const int t0 = i_t * BT;
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = warp * BW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * KD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * KD + col;
        if (tr0 < T) {
#if B2M_VECST
            *reinterpret_cast<float2*>(&dq[o0]) = make_float2(accq[nt][0], accq[nt][1]);
            *reinterpret_cast<float2*>(&dk[o0]) = make_float2(acck[nt][0], acck[nt][1]);
            *reinterpret_cast<float2*>(&dw[o0]) = make_float2(accw[nt][0], accw[nt][1]);
#else
            dq[o0] = accq[nt][0]; dq[o0 + 1] = accq[nt][1];
            dk[o0] = acck[nt][0]; dk[o0 + 1] = acck[nt][1];
            dw[o0] = accw[nt][0]; dw[o0 + 1] = accw[nt][1];
#endif
        }
        if (tr1 < T) {
#if B2M_VECST
            *reinterpret_cast<float2*>(&dq[o1]) = make_float2(accq[nt][2], accq[nt][3]);
            *reinterpret_cast<float2*>(&dk[o1]) = make_float2(acck[nt][2], acck[nt][3]);
            *reinterpret_cast<float2*>(&dw[o1]) = make_float2(accw[nt][2], accw[nt][3]);
#else
            dq[o1] = accq[nt][2]; dq[o1 + 1] = accq[nt][3];
            dk[o1] = acck[nt][2]; dk[o1 + 1] = acck[nt][3];
            dw[o1] = accw[nt][2]; dw[o1 + 1] = accw[nt][3];
#endif
        }
    }
  #if B2M_DGKCF
    // reduce over s in {0,1,2,3} = lanes {l, l^8, l^16, l^24}; lane l<8 then holds both
    // halves' sums for kcol = warp*BW + l and warp*BW + 8 + l.
    gpart  += __shfl_xor_sync(0xffffffffu, gpart,  8);
    gpart1 += __shfl_xor_sync(0xffffffffu, gpart1, 8);
    gpart  += __shfl_xor_sync(0xffffffffu, gpart,  16);
    gpart1 += __shfl_xor_sync(0xffffffffu, gpart1, 16);
    if (l < 8) {
        const int64_t base = ((int64_t)(n * NT + i_t) * H + hh) * KD + warp * BW + l;
        dgk[base] = gpart;
        dgk[base + 8] = gpart1;
    }
  #else
    gpart += __shfl_xor_sync(0xffffffffu, gpart, 16);
    if (l < 16) {
        const int kcol = warp * BW + l;
        dgk[((int64_t)(n * NT + i_t) * H + hh) * KD + kcol] = gpart;
    }
  #endif
#if B2M_OUT
    // ---- store `dv2` (complete B2 output) and `dvb` = A@dv ----
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = warp * BW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * VD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * VD + col;
        if (tr0 < T) {
            const float bt0 = beta[((int64_t)(n * T + tr0) * H + hh)];
            dvb[o0] = acc2[nt][0]; dvb[o0 + 1] = acc2[nt][1];
            dv2[o0] = bt0 * acc2[nt][0]; dv2[o0 + 1] = bt0 * acc2[nt][1];
        }
        if (tr1 < T) {
            const float bt1 = beta[((int64_t)(n * T + tr1) * H + hh)];
            dvb[o1] = acc2[nt][2]; dvb[o1 + 1] = acc2[nt][3];
            dv2[o1] = bt1 * acc2[nt][2]; dv2[o1 + 1] = bt1 * acc2[nt][3];
        }
    }
#endif
}

#if B2M_TMA_ON
// do/vn/dv are [B, T, H, V] fp32 contiguous.  dims {V, T, H, B}, box {32, BT, 1, 1},
// SWIZZLE_128B (inner 32*4 == 128 B).  One box per 128 B atom (NAT of them per operand).
// OOB token rows of a non-divisible T zero-fill.
static void make_map_dov(CUtensorMap* m, void* ptr, int H, int T, int B) {
    cuuint64_t gdim[4] = {(cuuint64_t)VD, (cuuint64_t)T, (cuuint64_t)H, (cuuint64_t)B};
    cuuint64_t gstr[3] = {(cuuint64_t)H * VD * 4, (cuuint64_t)VD * 4,
                          (cuuint64_t)T * H * VD * 4};
    cuuint32_t box[4] = {32, (cuuint32_t)BT, 1, 1};
    cuuint32_t estr[4] = {1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(dov) failed\n");
}
// h/dh are [B, NT, H, KD, VD] fp32 contiguous.  dims {VD, KD, H, NT, B},
// box {32, KD, 1, 1, 1}, SWIZZLE_128B (128 B rows; swizzle repeats every 8 rows).
static void make_map_hdh(CUtensorMap* m, void* ptr, int H, int NT, int B) {
    cuuint64_t gdim[5] = {(cuuint64_t)VD, (cuuint64_t)KD, (cuuint64_t)H, (cuuint64_t)NT,
                          (cuuint64_t)B};
    cuuint64_t gstr[4] = {(cuuint64_t)VD * 4, (cuuint64_t)KD * VD * 4,
                          (cuuint64_t)H * KD * VD * 4, (cuuint64_t)NT * H * KD * VD * 4};
    cuuint32_t box[5] = {32, (cuuint32_t)KD, 1, 1, 1};
    cuuint32_t estr[5] = {1, 1, 1, 1, 1};
    if (cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 5, ptr, gdim, gstr, box,
                               estr, CU_TENSOR_MAP_INTERLEAVE_NONE,
                               CU_TENSOR_MAP_SWIZZLE_128B,
                               CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                               CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS)
        fprintf(stderr, "cuTensorMapEncodeTiled(hdh) failed\n");
}
#endif

static int b2mma_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv,
                             void* dq, void* dk, void* dw, void* dgk,
#if B2M_OUT
                             const void* A4, const void* beta, void* dv2, void* dvb,
#endif
                             int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    int smem = (int)sizeof(Shared);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b2mma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
#if B2M_TMA_ON
    CUtensorMap dom, vnm, dvm, hm, dhm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
#if B2M_OUT
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
#endif
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
#if B2M_OUT
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
#endif
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2mma_launch(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv,
                            void* dq, void* dk, void* dw, void* dgk,
                            int B, int T, int H, int NT, void* stream) {
#if B2M_OUT
    // Only meaningful when compiled without -DB2M_OUT; the output-branch build must use
    // b2out_launch (this path would feed null A/beta/dv2/dvb into the kernel).
    return -4;
#else
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dq, dk, dw, dgk, B, T, H, NT, stream);
#endif
}

#if B2M_OUT
extern "C" int b2out_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* A4, const void* beta,
                           void* dq, void* dk, void* dw, void* dgk, void* dv2, void* dvb,
                           int B, int T, int H, int NT, void* stream) {
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dq, dk, dw, dgk,
                             A4, beta, dv2, dvb, B, T, H, NT, stream);
}
#endif

extern "C" int b2mma_smem_bytes() { return (int)sizeof(Shared); }
