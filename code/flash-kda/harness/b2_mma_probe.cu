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
// This file implements the GEMM core (like P7-44) so it is an apples-to-apples head-to-head
// vs the identical-op Triton kernel; the B2 non-GEMM pieces are added as separate opt-in
// segments: P7-52 `-DB2M_OUT` (output branch `dv2=beta*(A@dv)`) and P7-53 `-DB2M_DKGB`
// (`dkgb=A@(-dw)`).  The remaining finalize (`dA` post-ops, `db`, gate chain) and the intra
// fold are still not here (see docs/07 §49.7/§55.6/§56).
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
// P7-53: the *second non-GEMM segment* of B2 in CUDA -- the `i_k==0` finalize term
//               dkgb[o_A, k] = sum_{o_t} A[o_A, o_t] * (-dw[o_t, k])   (== A @ (-dw))
// (FLA / check_bwd_dqkg_kernel.py:305-307 negate `dw` then `b_dkgb = A @ b_dw`).  Like the
// output branch (P7-52) the *contraction* is over chunk rows `o_t` and the output row is a
// chunk row `o_A`, so `Akk` feeds the mma A-operand by plain index remap (no transpose) and
// the freshly-staged `dw` tile is the B-operand (negated on the fly).  `dkgb` feeds `db`,
// `dg`, `dk` and `dA` downstream, so it is exposed as its own output to check the GEMM
// standalone (mirrors P7-52 exposing `dvb`).
#ifndef B2M_DKGB
#define B2M_DKGB 0
#endif
// P7-54: the *third non-GEMM segment* of B2 in CUDA -- the two GEMM contributions to `dA`
// (the per-chunk within-chunk attention-matrix adjoint) *before* the final
// `-A^T diag(beta) A^T` sandwich (which is the remaining step):
//   dApre[o_t,o_u] = sum_v dv[o_t,v]*v[o_u,v]                 (base, FLA `dA += dv @ v^T`)
//                  - sum_k dw[o_t,k]*kg[o_u,k]                 (FLA `b_dw = -b_dw; dA += b_dw @ kg^T`)
// with `kg = k * exp2(g)` and `dw` the GEMM core's `dw = dv @ h` output.  NOTE the sign:
// FLA negates `dw` (`chunk_bwd.py:280-281`), so the `dw@kg^T` term enters `dApre` *negated*.  Both contractions
// live *here* (no cross-chunk coupling), so this segment is verifiable standalone like P7-52's
// `dvb` / P7-53's `dkgb`.  The base contracts over V (warp `w` owns V columns [w*16,w*16+16),
// exactly the warp-per-V-column split of the output branch), the `dw@kg^T` contracts over k
// (warp `w` owns output k columns, same split as the GEMM core); each warp produces a private
// `[BT,BT]` partial, the 8 partials are summed through smem.
#ifndef B2M_DA
#define B2M_DA 0
#endif
// P7-55: the *fourth non-GEMM segment* of B2 in CUDA -- the final `dA` sandwich (the
// remaining `dA` post-op §55.6/§56.6/§57.6 left open), i.e. the *complete* B2 `dA` output:
//   A where m_A=(o_t>o_u) & valid:   dAm[o_t,o_u] = dApre[o_t,o_u] * beta[o_u]
//   dA = -A^T diag(beta) A^T (dApre)  ==  where(m_A, -(b_A @ (dAm @ b_A)), 0)
// where `b_A[p,j] = A[o_t=j, o_A=p]` (FLA/Triton `b_A` orientation, `check_bwd_dqkg_kernel.py:240`)
// and `dApre` is P7-54's pre-sandwich `dv@v^T - dw@kg^T`.  This consumes `dApre` from the
// in-kernel reduction (no re-read) and only adds two [BT,BT]@[BT,BT] fp32 matmuls + a `beta`
// scale, so it is verifiable standalone against a torch sandwich on the *same* `dApre`
// (bit-level, fp32 scalar) and against FLA's final `dA`.  Requires `B2M_DA`.
#ifndef B2M_SAND
#define B2M_SAND 0
#endif
#if B2M_SAND && !B2M_DA
#error "B2M_SAND requires B2M_DA"
#endif
// P7-56: the *fifth non-GEMM segment* of B2 in CUDA -- the per-token write-gate gradient
//   db[o_A] = sum_v dvb[o_A,v] * v[o_A,v]  +  sum_k dkgb[o_A,k] * kg[o_A,k]
// (FLA `chunk_bwd.py`, `check_bwd_dqkg_kernel.py:180-210`: `b_db = sum_V(b_dvb*b_v)` then
// `b_db += sum_k(b_dkgb*b_kg)`).  Both operands are already computed in the same kernel -- `dvb`
// is the P7-52 output branch (`=A@dv`) and `dkgb` is the P7-53 finalize term (`=A@(-dw)`), each
// owned by a per-warp column slice -- so `db` is exactly a **cross-warp reduction** of two
// warp-private column sums.  It is a *complete* B2 output (never touched by the gate chain), so it
// is verifiable standalone.  Requires the output and dkgb branches (`B2M_OUT` + `B2M_DKGB`), plus
// the raw `v` staging and the chunk `kg=k*exp2(g)` buffer (shared with `B2M_DA`).
#ifndef B2M_DB
#define B2M_DB 0
#endif
#if B2M_DB && !(B2M_OUT && B2M_DKGB)
#error "B2M_DB requires B2M_OUT and B2M_DKGB"
#endif
//   B2M_NODB : keep the `B2M_DB` build/signature/smem but skip the `db` reduction+store
//              (baseline for the isolated `db` segment timing).
#ifndef B2M_NODB
#define B2M_NODB 0
#endif
// raw `v` staging (`Stage::vv`) and the `kg` buffer are needed by `B2M_DA` (base dA) and/or `B2M_DB`.
#define B2M_VV_KG_ON (B2M_DA || B2M_DB)
constexpr int BVK = B2M_BVK;   // V-tile (contraction) depth staged per pipeline stage
constexpr int NS  = B2M_NS;    // pipeline stages
#if B2M_TMA_ON
// Each operand is staged as NAT atom-major `[NAT][rows][32]` sub-buffers, each written by
// its own 128 B-swizzled TMA box (inner width 32).  BVK==32 -> NAT=1 (identical to P7-49).
constexpr int NAT = BVK / 32;
static_assert(BVK == 32 || BVK == 64, "TMA supports BVK==32 or BVK==64 (128 B atoms)");
#else
#endif
// padded row stride for the *non-TMA-staged* scratch buffers (skg/swall/dapart/da16), used by
// P7-54 regardless of the stage layout.
constexpr int BVP = BVK + 4;   // padded row stride (16B keeps 4-float cp.async aligned)

static_assert(VD % BVK == 0, "BVK must divide V=128");

struct alignas(128) Stage {
#if B2M_TMA_ON
    float do_[NAT][BT][32];   // do  [t, v], atom-major (each atom = 128 B-swizzled [BT][32])
    float vn [NAT][BT][32];   // v_new[t, v]
    float dv [NAT][BT][32];   // dv  [t, v]
    float h  [NAT][KD][32];   // h  [k, v]  (rows = output N index, cols = contraction v)
    float dh [NAT][KD][32];   // dh [k, v]
#if B2M_VV_KG_ON
    float vv [NAT][BT][32];   // raw v [t, v]  (base dA / `db` operand)
#endif
#if B2M_DKGB
    float dwm[NAT][BT][32];   // dw [t, k]  (P7-53: B-operand of `A @ -dw`, k-tiled)
#endif
#else
    float do_[BT][BVP];   // do  [t, v]
    float vn [BT][BVP];   // v_new[t, v]
    float dv [BT][BVP];   // dv  [t, v]
    float h  [KD][BVP];   // h  [k, v]  (rows = output N index, cols = contraction v)
    float dh [KD][BVP];   // dh [k, v]
#if B2M_VV_KG_ON
    float vv [BT][BVP];   // raw v [t, v]
#endif
#if B2M_DKGB
    float dwm[BT][BVP];   // dw [t, k]
#endif
#endif
};
struct alignas(128) Shared {
    Stage st[NS];
#if B2M_TMA_ON
    uint64_t full[NS];    // per-stage TMA-completion mbarrier (arrive count 1)
#endif
#if B2M_VV_KG_ON
    // per-chunk `kg = k*exp2(g)` [BT, KD] (loaded once per chunk, padded row stride), shared
    // by the `B2M_DA` base dA and the `B2M_DB` `sum_k dkgb*kg` term.
    float skg[BT][KD];
#endif
#if B2M_DA
    // the per-warp `dw` mirror `swall[NW][BT][BW]`, the 8 per-warp partials `dapart`, and
    // the reduced tile `da16`.
    float swall[NW][BT][BW];
    float dapart[NW][BT][BT];
    float da16[BT][BT];
#endif
#if B2M_DB
    // `db` cross-warp reduction: per-warp per-row partials (`dbrow[NW][BT]`) plus the raw-v
    // mirror of each warp's owned V columns (`dbv[NW][BT][BW]`), since the staged `vv` tile is
    // overwritten before the output branch's `acc2` is complete.
    float dbrow[NW][BT];
    float dbv[NW][BT][BW];
#endif
#if B2M_SAND
    // b_A[p][j] = A[o_t=j, o_A=p]  (P7-55 sandwich operand; loaded once per chunk)
    float sA[BT][BT];
    float sbeta[BT];     // beta[o_t] (rows t<T)
    float daB[BT][BT];   // scratch: masked dApre, then the second matmul result
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

// P7-54: padded (never TMA-swizzled) row-major read for the `skg`/scratch buffers.
DEV float rdp(const float* buf, int row, int col) { return buf[row * BVP + col]; }

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
#if B2M_DKGB
                  const float* __restrict__ dw,
#endif
                  int n, int hh, int T, int H, int NT, int c, int v0) {
    const int tid = threadIdx.x, nthr = blockDim.x;
    const int t0 = c * BT;
    // do/vn/dv [BT, BVK]  (token rows; zero the pad rows for a non-divisible tail)
    for (int idx = tid; idx < BT * (BVK / 4); idx += nthr) {
        int row = idx / (BVK / 4), cg = idx % (BVK / 4), t = t0 + row;
        float* pd = &st->do_[row][cg * 4];
        float* pv = &st->vn [row][cg * 4];
        float* pw = &st->dv [row][cg * 4];
#if B2M_VV_KG_ON
        float* pvv = &st->vv[row][cg * 4];
#endif
#if B2M_NODOV
        for (int j = 0; j < 4; ++j) { pd[j] = 0.f; pv[j] = 0.f; pw[j] = 0.f; }
  #if B2M_VV_KG_ON
        for (int j = 0; j < 4; ++j) pvv[j] = 0.f;
  #endif
#else
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + hh) * VD + v0 + cg * 4;
            cp16(pd, do_ + off);
            cp16(pv, vn  + off);
            cp16(pw, dv  + off);
  #if B2M_VV_KG_ON
            cp16(pvv, v + off);
  #endif
        } else {
            for (int j = 0; j < 4; ++j) { pd[j] = 0.f; pv[j] = 0.f; pw[j] = 0.f; }
  #if B2M_VV_KG_ON
            for (int j = 0; j < 4; ++j) pvv[j] = 0.f;
  #endif
        }
#endif
#if B2M_DKGB
        // dw [BT, BVK] (token rows; k-tiled).  Same [B,T,H,K] layout as do/vn/dv, KD==VD.
        float* pn = &st->dwm[row][cg * 4];
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + hh) * KD + v0 + cg * 4;
            cp16(pn, dw + off);
        } else {
            for (int j = 0; j < 4; ++j) pn[j] = 0.f;
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
#if B2M_DKGB
                   const CUtensorMap* dwm,
#endif
#if B2M_VV_KG_ON
                   const CUtensorMap* vvm,
#endif
                   int n, int hh, int c, int v0, uint64_t* bar) {
    const int t0 = c * BT;
    mbar_arrive_tx(bar, (uint32_t)(((3
#if B2M_VV_KG_ON
                                    + 1
#endif
#if B2M_DKGB
                                    + 1
#endif
                                   ) * BT + 2 * KD) * BVK * 4));
    #pragma unroll
    for (int a = 0; a < NAT; ++a) {
        const int va = v0 + a * 32;
        tma4d_load(dom, &st->do_[a][0][0], va, t0, hh, n, bar);
        tma4d_load(vnm, &st->vn [a][0][0], va, t0, hh, n, bar);
        tma4d_load(dvm, &st->dv [a][0][0], va, t0, hh, n, bar);
#if B2M_VV_KG_ON
        tma4d_load(vvm, &st->vv [a][0][0], va, t0, hh, n, bar);
#endif
        tma5d_load(hm,  &st->h  [a][0][0], va, 0, hh, c, n, bar);
        tma5d_load(dhm, &st->dh [a][0][0], va, 0, hh, c, n, bar);
#if B2M_DKGB
        tma4d_load(dwm, &st->dwm[a][0][0], va, t0, hh, n, bar);
#endif
    }
}
#endif

// ---------------------------------------------------------------------- kernel
#if B2M_DKGB
#define TMA_DWM_ARG , &dwm
#define PRE_DW_ARG  , dw
#else
#define TMA_DWM_ARG
#define PRE_DW_ARG
#endif
#if B2M_VV_KG_ON
#define TMA_VV_ARG , &vvm
#else
#define TMA_VV_ARG
#endif
__global__ void __launch_bounds__(NW * 32)
b2mma_kernel(
#if B2M_TMA_ON
             const __grid_constant__ CUtensorMap dom, const __grid_constant__ CUtensorMap vnm,
             const __grid_constant__ CUtensorMap dvm, const __grid_constant__ CUtensorMap hm,
             const __grid_constant__ CUtensorMap dhm,
#if B2M_DKGB
             const __grid_constant__ CUtensorMap dwm,
#endif
#if B2M_VV_KG_ON
             const __grid_constant__ CUtensorMap vvm,
#endif
#endif
             const float* __restrict__ v, const float* __restrict__ vn,
             const float* __restrict__ h, const float* __restrict__ dh,
             const float* __restrict__ do_, const float* __restrict__ dv,
#if B2M_VV_KG_ON
             const float* __restrict__ kptr, const float* __restrict__ gptr,
#endif
#if B2M_DKGB
             const float* __restrict__ dw,
#endif
             float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dw_out,
             float* __restrict__ dgk,
#if B2M_OUT || B2M_DKGB
             const float* __restrict__ A4,
#endif
#if B2M_OUT
             const float* __restrict__ beta,
             float* __restrict__ dv2, float* __restrict__ dvb,
#endif
#if B2M_DKGB
             float* __restrict__ dkgb,
#endif
#if B2M_DB
             float* __restrict__ db_out,
#endif
#if B2M_DA
             float* __restrict__ dApre,
#endif
#if B2M_SAND
             const float* __restrict__ A4s, const float* __restrict__ beta,
             float* __restrict__ dAout,
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
#endif
#if B2M_DKGB
    float acc3[2][4];     // dkgb[o_A, k] output tile (m16 x 2*n8), warp-private k columns
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt)
        #pragma unroll
        for (int x = 0; x < 4; ++x) acc3[nt][x] = 0.f;
#endif
#if B2M_DA
    float dbpart[8], dkpart[8];   // per-lane partials: 8 [r,c] entries each (r=l>>1, c=(l&1)*8+i)
    #pragma unroll
    for (int i = 0; i < 8; ++i) { dbpart[i] = 0.f; dkpart[i] = 0.f; }
#endif
#if B2M_VV_KG_ON
    // `kg = k*exp2(g)` for this chunk (rows t<T; pad rows zeroed).  One load per chunk (not per
    // V-tile); read back post-loop by `rdp` (P7-54 `dw@kg^T`) and by the P7-56 `db` term.
    {
        for (int idx = threadIdx.x; idx < BT * KD; idx += NW * 32) {
            const int row = idx / KD, col = idx % KD, tt = i_t * BT + row;
            float kv = 0.f, gv = 0.f;
            if (tt < T) {
                const int64_t off = ((int64_t)(n * T + tt) * H + hh) * KD + col;
                kv = kptr[off]; gv = gptr[off];
            }
            sh->skg[row][col] = kv * exp2f(gv);
        }
    }
#endif
#if B2M_SAND
    // b_A[p][j] = A[b, t=i_t*BT+j, h, c=p]  (only rows t<T valid); beta[o_t] (rows t<T).
    for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
        const int j = idx / BT, p = idx % BT;
        float av = 0.f;
        if (i_t * BT + j < T)
            av = A4s[((int64_t)(n * T + i_t * BT + j) * H + hh) * BT + p];
        sh->sA[p][j] = av;
    }
    if (threadIdx.x < BT) {
        const int r = threadIdx.x;
        sh->sbeta[r] = (i_t * BT + r < T) ? beta[((int64_t)(n * T + i_t * BT + r) * H + hh)] : 0.f;
    }
#endif
#if B2M_DA || B2M_VV_KG_ON || B2M_SAND
    __syncthreads();
#endif
#if B2M_OUT || B2M_DKGB
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
                issue_tma(&sh->st[s], &dom, &vnm, &dvm, &hm, &dhm TMA_DWM_ARG TMA_VV_ARG, n, hh, i_t, s * BVK,
                          &sh->full[s]);
    }
    for (int it = 0; it < nvt; ++it) {
        if (threadIdx.x == 0) {
            const int snext = it + NS - 1;
            if (snext < nvt)
                issue_tma(&sh->st[snext % NS], &dom, &vnm, &dvm, &hm, &dhm TMA_DWM_ARG TMA_VV_ARG, n, hh, i_t,
                          snext * BVK, &sh->full[snext % NS]);
        }
        mbar_wait(&sh->full[it % NS], (it / NS) & 1);
        __syncthreads();

        const Stage* st = &sh->st[it % NS];
#else
    for (int s = 0; s < NS - 1; ++s) {
        if (s < nvt) prefetch(&sh->st[s], v, vn, h, dh, do_, dv PRE_DW_ARG, n, hh, T, H, NT, i_t, s * BVK);
        __pipeline_commit();
    }

    for (int it = 0; it < nvt; ++it) {
        const int snext = it + NS - 1;
        if (snext < nvt)
            prefetch(&sh->st[snext % NS], v, vn, h, dh, do_, dv PRE_DW_ARG, n, hh, T, H, NT, i_t, snext * BVK);
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
#if B2M_DKGB
        // ---- second non-GEMM segment (P7-53): dkgb = A @ (-dw), contraction over o_t ----
        // Structurally identical to the output branch above (A-operand index remap, no
        // transpose), except the B-operand is the freshly-staged `dw` tile (k-tiled, warp
        // `w` owns k columns [w*BW,w*BW+BW)) and its sign is flipped (`b_dw = -b_dw`, FLA
        // chunk_bwd.py:305 / check_bwd_dqkg_kernel.py:305).
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
                        bf[0] = to_tf32(-rd((const float*)st->dwm, ot0, lc, BT));
                        bf[1] = to_tf32(-rd((const float*)st->dwm, ot1, lc, BT));
                        mma8(acc3[nt], af, bf, acc3[nt]);
                    }
                }
            }
        }
#endif
#if B2M_DA
        // ---- P7-54 base dA: dApre[o_t,o_u] += sum_{v in warp V cols} dv[o_t,v]*v[o_u,v] ----
        // Warp `w` owns V columns [w*BW, w*BW+BW) (only its matching stage contributes).
        // Lane `l` owns the 8 entries (r = l>>1, c = (l&1)*8 + i, i=0..7) of the [BT,BT] partial.
        {
            const int local0 = warp * BW - it * BVK;
            if (local0 >= 0 && local0 + BW <= BVK) {
                const int r = l >> 1, c0 = (l & 1) * 8;
                #pragma unroll
                for (int j = 0; j < BW; ++j) {
                    const float dvr = rd((const float*)st->dv, r, local0 + j, BT);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i)
                        dbpart[i] += dvr * rd((const float*)st->vv, c0 + i, local0 + j, BT);
                }
            }
        }
#endif
#if B2M_DB && !B2M_NODB
        // ---- P7-56: mirror the raw `v` of this warp's owned V columns into `dbv` (the staged
        // `vv` tile is overwritten before `acc2` is complete, so `db`'s `dvb*v` term needs a copy).
        {
            const int local0 = warp * BW - it * BVK;
            if (local0 >= 0 && local0 + BW <= BVK) {
                for (int idx = l; idx < BT * BW; idx += 32) {
                    const int r = idx / BW, j = idx % BW;
                    sh->dbv[warp][r][j] = rd((const float*)st->vv, r, local0 + j, BT);
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
            *reinterpret_cast<float2*>(&dw_out[o0]) = make_float2(accw[nt][0], accw[nt][1]);
#else
            dq[o0] = accq[nt][0]; dq[o0 + 1] = accq[nt][1];
            dk[o0] = acck[nt][0]; dk[o0 + 1] = acck[nt][1];
            dw_out[o0] = accw[nt][0]; dw_out[o0 + 1] = accw[nt][1];
#endif
        }
        if (tr1 < T) {
#if B2M_VECST
            *reinterpret_cast<float2*>(&dq[o1]) = make_float2(accq[nt][2], accq[nt][3]);
            *reinterpret_cast<float2*>(&dk[o1]) = make_float2(acck[nt][2], acck[nt][3]);
            *reinterpret_cast<float2*>(&dw_out[o1]) = make_float2(accw[nt][2], accw[nt][3]);
#else
            dq[o1] = accq[nt][2]; dq[o1 + 1] = accq[nt][3];
            dk[o1] = acck[nt][2]; dk[o1 + 1] = acck[nt][3];
            dw_out[o1] = accw[nt][2]; dw_out[o1 + 1] = accw[nt][3];
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
#if B2M_DKGB
    // ---- store `dkgb` = A @ (-dw) (row = o_A token axis, col = output k) ----
    #pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
        const int col = warp * BW + nt * 8 + 2 * t;
        const int tr0 = t0 + g, tr1 = t0 + g + 8;
        const int64_t o0 = ((int64_t)(n * T + tr0) * H + hh) * KD + col;
        const int64_t o1 = ((int64_t)(n * T + tr1) * H + hh) * KD + col;
        if (tr0 < T) { dkgb[o0] = acc3[nt][0]; dkgb[o0 + 1] = acc3[nt][1]; }
        if (tr1 < T) { dkgb[o1] = acc3[nt][2]; dkgb[o1 + 1] = acc3[nt][3]; }
    }
#endif
#if B2M_DB && !B2M_NODB
    // ---- P7-56: db = sum_v dvb*v + sum_k dkgb*kg (cross-warp reduction of column sums) ----
    // acc2/acc3 fragment rows are g/g+8, cols nt*8+2t,nt*8+2t+1 (within the warp's owned slice).
    // `skg` is persistent (ensures the `dkgb*kg` term can be read post-loop); `dbv` mirrors the
    // warp's V columns of raw `v` for the `dvb*v` term.
    {
        float p0 = 0.f, p1 = 0.f;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int cl = nt * 8 + 2 * t;          // column within the warp's owned 16-wide slice
            const int ck = warp * BW + cl;          // absolute k column (== V column index)
            p0 += acc2[nt][0] * sh->dbv[warp][g][cl]     + acc2[nt][1] * sh->dbv[warp][g][cl + 1];
            p0 += acc3[nt][0] * sh->skg[g][ck]           + acc3[nt][1] * sh->skg[g][ck + 1];
            p1 += acc2[nt][2] * sh->dbv[warp][g + 8][cl] + acc2[nt][3] * sh->dbv[warp][g + 8][cl + 1];
            p1 += acc3[nt][2] * sh->skg[g + 8][ck]       + acc3[nt][3] * sh->skg[g + 8][ck + 1];
        }
        p0 += __shfl_xor_sync(0xffffffffu, p0, 1); p0 += __shfl_xor_sync(0xffffffffu, p0, 2);
        p1 += __shfl_xor_sync(0xffffffffu, p1, 1); p1 += __shfl_xor_sync(0xffffffffu, p1, 2);
        if (t == 0) { sh->dbrow[warp][g] = p0; sh->dbrow[warp][g + 8] = p1; }
    }
    __syncthreads();
    {
        const int idx = threadIdx.x;
        if (idx < BT) {
            float s = 0.f;
            #pragma unroll
            for (int w = 0; w < NW; ++w) s += sh->dbrow[w][idx];
            if (i_t * BT + idx < T)
                db_out[((int64_t)(n * T + i_t * BT + idx) * H + hh)] = s;
        }
    }
#endif
#if B2M_DA
    // ---- P7-54 post-loop: `dw @ kg^T` partial, reduce the 8 warps, store `dApre` ----
    {
        const int r = l >> 1, c0 = (l & 1) * 8;
        // Mirror this warp's `dw` (=accw, mma-fragment layout: rows g/g+8, cols 2t/2t+1) into
        // `swall[warp]` so any lane can read it in row-major order.
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            const int col = nt * 8 + 2 * t;
            sh->swall[warp][g][col]         = accw[nt][0];
            sh->swall[warp][g][col + 1]     = accw[nt][1];
            sh->swall[warp][g + 8][col]     = accw[nt][2];
            sh->swall[warp][g + 8][col + 1] = accw[nt][3];
        }
        __syncwarp();
        // dw@kg^T partial over this warp's output k columns [warp*BW, warp*BW+BW).  FLA negates
        // `dw`, so the term is *subtracted* from the base below (matches `chunk_bwd.py:280-281`).
        #pragma unroll
        for (int kk = 0; kk < BW; ++kk) {
            const float w0 = sh->swall[warp][r][kk];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
                dkpart[i] += w0 * sh->skg[c0 + i][warp * BW + kk];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            sh->dapart[warp][r][c0 + i] = dbpart[i] - dkpart[i];
    }
    __syncthreads();
    {   // reduce the 8 warp partials (one thread per [BT,BT] entry)
        const int idx = threadIdx.x, r = idx / BT, c = idx % BT;
        float s = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) s += sh->dapart[w][r][c];
        sh->da16[r][c] = s;
    }
    __syncthreads();
    {   // store `dApre` (row = o_t, col = o_u; rows o_t<T)
        const int idx = threadIdx.x;
        if (idx < BT * BT) {
            const int r = idx / BT, c = idx % BT, tr = i_t * BT + r;
            if (tr < T)
                dApre[((int64_t)(n * T + tr) * H + hh) * BT + c] = sh->da16[r][c];
        }
    }
#endif
#if B2M_SAND
    // ---- P7-55 final dA sandwich: dA = where(m_A, -(b_A @ ((dApre*beta_col) @ b_A)), 0) ----
    // `sh->da16` still holds `dApre` [o_t,o_u]; all ops are fp32 scalar (no tf32), so the
    // result is bit-comparable to a torch sandwich on the *same* `dApre`.
    __syncthreads();
    {   // dAm[r][c] = (r>c & valid) ? dApre[r][c]*beta[c] : 0
        const int idx = threadIdx.x;
        if (idx < BT * BT) {
            const int r = idx / BT, c = idx % BT;
            const bool v = (r > c) && (i_t * BT + r < T) && (i_t * BT + c < T);
            sh->daB[r][c] = v ? sh->da16[r][c] * sh->sbeta[c] : 0.f;
        }
    }
    __syncthreads();
    {   // X1[i][j] = sum_p dAm[i][p] * b_A[p][j]   -> da16
        for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
            const int i = idx / BT, jj = idx % BT;
            float s = 0.f;
            #pragma unroll
            for (int p = 0; p < BT; ++p) s += sh->daB[i][p] * sh->sA[p][jj];
            sh->da16[i][jj] = s;
        }
    }
    __syncthreads();
    {   // X2[i][j] = sum_p b_A[i][p] * X1[p][j]     -> daB
        for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
            const int i = idx / BT, jj = idx % BT;
            float s = 0.f;
            #pragma unroll
            for (int p = 0; p < BT; ++p) s += sh->sA[i][p] * sh->da16[p][jj];
            sh->daB[i][jj] = s;
        }
    }
    __syncthreads();
    {   // store final `dA` = where(m_A, -X2, 0).  X2 first axis maps to the stored token row
        // (golden/Triton use the same-index mapping, `check_bwd_dqkg_kernel.py:203`), so token
        // row = i and column index = jj, with the mask (i>jj) applied on X2's own axes.
        for (int idx = threadIdx.x; idx < BT * BT; idx += NW * 32) {
            const int i = idx / BT, jj = idx % BT;
            const bool v = (i > jj) && (i_t * BT + i < T) && (i_t * BT + jj < T);
            if (i_t * BT + i < T)
                dAout[((int64_t)(n * T + i_t * BT + i) * H + hh) * BT + jj] = v ? -sh->daB[i][jj] : 0.f;
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

#if !B2M_DA && !B2M_DB
static int b2mma_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv,
#if B2M_DKGB
                             const void* dwin,
#endif
                             void* dq, void* dk, void* dw, void* dgk,
#if B2M_OUT || B2M_DKGB
                             const void* A4,
#endif
#if B2M_OUT
                             const void* beta, void* dv2, void* dvb,
#endif
#if B2M_DKGB
                             void* dkgb,
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
#if B2M_DKGB
    CUtensorMap dwm;
    make_map_dov(&dwm, (void*)dwin, H, T, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, dwm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (float*)dkgb,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
#if B2M_OUT
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
#endif
        B, T, H, NT);
#endif
#else
#if B2M_DKGB
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (float*)dkgb,
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
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2mma_launch(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv,
                            void* dq, void* dk, void* dw, void* dgk,
                            int B, int T, int H, int NT, void* stream) {
#if B2M_OUT || B2M_DKGB
    // Only meaningful when compiled without -DB2M_OUT/-DB2M_DKGB; those builds must use
    // their own entries (b2out_launch / b2dkgb_launch), else null operands reach the kernel.
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

#if B2M_DKGB
extern "C" int b2dkgb_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv, const void* dwin,
                             const void* A4,
                             void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                             int B, int T, int H, int NT, void* stream) {
    return b2mma_launch_impl(v, vn, h, dh, dov, dvv, dwin, dq, dk, dw, dgk,
                             A4, dkgb, B, T, H, NT, stream);
}
#endif
#endif  // !B2M_DA && !B2M_DB

#if B2M_DA && !B2M_SAND
// P7-54 dedicated launcher (`b2da_launch`): only valid when compiled with `-DB2M_DA` and
// neither `-DB2M_OUT` nor `-DB2M_DKGB` (it assumes the bare `b2mma_kernel` signature).
static int b2da_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv, const void* k, const void* g,
                            void* dq, void* dk, void* dw, void* dgk, void* dApre,
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
    CUtensorMap dom, vnm, dvm, hm, dhm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2da_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* k, const void* g,
                           void* dq, void* dk, void* dw, void* dgk, void* dApre,
                           int B, int T, int H, int NT, void* stream) {
    return b2da_launch_impl(v, vn, h, dh, dov, dvv, k, g, dq, dk, dw, dgk, dApre,
                            B, T, H, NT, stream);
}
#endif  // B2M_DA && !B2M_SAND

#if B2M_SAND
// P7-55 dedicated launcher (`b2sand_launch`): compiled with `-DB2M_DA -DB2M_SAND`
// (plus the usual stage options).  Outputs both `dApre` (P7-54) and the final `dA`.
static int b2sand_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                              const void* dov, const void* dvv, const void* k, const void* g,
                              const void* A4, const void* beta,
                              void* dq, void* dk, void* dw, void* dgk, void* dApre, void* dA,
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
    CUtensorMap dom, vnm, dvm, hm, dhm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        (const float*)A4, (const float*)beta, (float*)dA,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, (float*)dApre,
        (const float*)A4, (const float*)beta, (float*)dA,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2sand_launch(const void* v, const void* vn, const void* h, const void* dh,
                             const void* dov, const void* dvv, const void* k, const void* g,
                             const void* A4, const void* beta,
                             void* dq, void* dk, void* dw, void* dgk, void* dApre, void* dA,
                             int B, int T, int H, int NT, void* stream) {
    return b2sand_launch_impl(v, vn, h, dh, dov, dvv, k, g, A4, beta,
                              dq, dk, dw, dgk, dApre, dA, B, T, H, NT, stream);
}
#endif  // B2M_SAND


#if B2M_DB
// P7-56 dedicated launcher (`b2db_launch`): compiled with `-DB2M_OUT -DB2M_DKGB -DB2M_DB`
// (plus the usual stage options).  Emits `dvb` (P7-52), `dkgb` (P7-53) and the complete `db`.
static int b2db_launch_impl(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv, const void* dwin,
                            const void* k, const void* g, const void* A4, const void* beta,
                            void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                            void* dv2, void* dvb, void* db_out,
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
    CUtensorMap dom, vnm, dvm, hm, dhm, dwm, vvm;
    make_map_dov(&dom, (void*)dov, H, T, B);
    make_map_dov(&vnm, (void*)vn,  H, T, B);
    make_map_dov(&dvm, (void*)dvv, H, T, B);
    make_map_dov(&dwm, (void*)dwin, H, T, B);
    make_map_dov(&vvm, (void*)v,   H, T, B);
    make_map_hdh(&hm,  (void*)h,  H, NT, B);
    make_map_hdh(&dhm, (void*)dh, H, NT, B);
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        dom, vnm, dvm, hm, dhm, dwm, vvm,
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb, (float*)db_out,
        B, T, H, NT);
#else
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv, (const float*)k, (const float*)g,
        (const float*)dwin,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        (const float*)A4, (const float*)beta, (float*)dv2, (float*)dvb,
        (float*)dkgb, (float*)db_out,
        B, T, H, NT);
#endif
    return (int)cudaGetLastError();
}

extern "C" int b2db_launch(const void* v, const void* vn, const void* h, const void* dh,
                           const void* dov, const void* dvv, const void* dwin,
                           const void* k, const void* g, const void* A4, const void* beta,
                           void* dq, void* dk, void* dw, void* dgk, void* dkgb,
                           void* dv2, void* dvb, void* db_out,
                           int B, int T, int H, int NT, void* stream) {
    return b2db_launch_impl(v, vn, h, dh, dov, dvv, dwin, k, g, A4, beta,
                            dq, dk, dw, dgk, dkgb, dv2, dvb, db_out, B, T, H, NT, stream);
}
#endif  // B2M_DB


extern "C" int b2mma_smem_bytes() { return (int)sizeof(Shared); }
