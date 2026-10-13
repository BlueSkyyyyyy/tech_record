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
#include <cstdint>

#define DEV __device__ __forceinline__

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
constexpr int BVK = B2M_BVK;   // V-tile (contraction) depth staged per pipeline stage
constexpr int NS  = B2M_NS;    // pipeline stages
constexpr int BVP = BVK + 4;   // padded row stride (16B keeps 4-float cp.async aligned)

static_assert(VD % BVK == 0, "BVK must divide V=128");

struct alignas(128) Stage {
    float do_[BT][BVP];   // do  [t, v]
    float vn [BT][BVP];   // v_new[t, v]
    float dv [BT][BVP];   // dv  [t, v]
    float h  [KD][BVP];   // h  [k, v]  (rows = output N index, cols = contraction v)
    float dh [KD][BVP];   // dh [k, v]
};
struct alignas(128) Shared { Stage st[NS]; };

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
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + hh) * VD + v0 + cg * 4;
            __pipeline_memcpy_async(pd, do_ + off, 16);
            __pipeline_memcpy_async(pv, vn  + off, 16);
            __pipeline_memcpy_async(pw, dv  + off, 16);
        } else {
            for (int j = 0; j < 4; ++j) { pd[j] = 0.f; pv[j] = 0.f; pw[j] = 0.f; }
        }
    }
    // h/dh [KD, BVK]  (chunk-granular, always full)
    const int64_t hbase = ((int64_t)(n * NT + c) * H + hh) * KD * VD;
    for (int idx = tid; idx < KD * (BVK / 4); idx += nthr) {
        int row = idx / (BVK / 4), cg = idx % (BVK / 4);
        const int64_t off = hbase + (int64_t)row * VD + v0 + cg * 4;
        __pipeline_memcpy_async(&st->h [row][cg * 4], h  + off, 16);
        __pipeline_memcpy_async(&st->dh[row][cg * 4], dh + off, 16);
    }
}

// ---------------------------------------------------------------------- kernel
__global__ void __launch_bounds__(NW * 32)
b2mma_kernel(const float* __restrict__ v, const float* __restrict__ vn,
             const float* __restrict__ h, const float* __restrict__ dh,
             const float* __restrict__ do_, const float* __restrict__ dv,
             float* __restrict__ dq, float* __restrict__ dk, float* __restrict__ dw,
             float* __restrict__ dgk, int B, int T, int H, int NT) {
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

    const int nvt = VD / BVK;
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
        #pragma unroll
        for (int kk = 0; kk < BVK / 8; ++kk) {
            const int kc = kk * 8;
            // A fragments (m16 x k8) from do/vn/dv: A[row=g/g+8][col=t/t+4]
            uint32_t aq[4], ak[4], aw[4];
            aq[0] = to_tf32(st->do_[g    ][kc + t]);
            aq[1] = to_tf32(st->do_[g + 8][kc + t]);
            aq[2] = to_tf32(st->do_[g    ][kc + t + 4]);
            aq[3] = to_tf32(st->do_[g + 8][kc + t + 4]);
            ak[0] = to_tf32(st->vn[g    ][kc + t]);
            ak[1] = to_tf32(st->vn[g + 8][kc + t]);
            ak[2] = to_tf32(st->vn[g    ][kc + t + 4]);
            ak[3] = to_tf32(st->vn[g + 8][kc + t + 4]);
            aw[0] = to_tf32(st->dv[g    ][kc + t]);
            aw[1] = to_tf32(st->dv[g + 8][kc + t]);
            aw[2] = to_tf32(st->dv[g    ][kc + t + 4]);
            aw[3] = to_tf32(st->dv[g + 8][kc + t + 4]);
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const int krow = warp * BW + nt * 8 + g;   // output column = k = n index
                uint32_t bh[2], bd[2];
                bh[0] = to_tf32(st->h [krow][kc + t]);
                bh[1] = to_tf32(st->h [krow][kc + t + 4]);
                bd[0] = to_tf32(st->dh[krow][kc + t]);
                bd[1] = to_tf32(st->dh[krow][kc + t + 4]);
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
                    p += st->h[kcol][vv] * st->dh[kcol][vv];
                if (ph == 0) gpart += p; else gpart1 += p;
            }
        }
  #else
        // baseline dgk column reduction: lane l handles column warp*BW + (l&15), split v by l>>4
        {
            const int kcol = warp * BW + (l & 15);
            float p = 0.f;
            for (int vv = (l >> 4); vv < BVK; vv += 2)
                p += st->h[kcol][vv] * st->dh[kcol][vv];
            gpart += p;
        }
  #endif
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
            dq[o0] = accq[nt][0]; dq[o0 + 1] = accq[nt][1];
            dk[o0] = acck[nt][0]; dk[o0 + 1] = acck[nt][1];
            dw[o0] = accw[nt][0]; dw[o0 + 1] = accw[nt][1];
        }
        if (tr1 < T) {
            dq[o1] = accq[nt][2]; dq[o1 + 1] = accq[nt][3];
            dk[o1] = acck[nt][2]; dk[o1 + 1] = acck[nt][3];
            dw[o1] = accw[nt][2]; dw[o1 + 1] = accw[nt][3];
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
}

extern "C" int b2mma_launch(const void* v, const void* vn, const void* h, const void* dh,
                            const void* dov, const void* dvv,
                            void* dq, void* dk, void* dw, void* dgk,
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
    b2mma_kernel<<<grid, NW * 32, smem, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk, B, T, H, NT);
    return (int)cudaGetLastError();
}

extern "C" int b2mma_smem_bytes() { return (int)sizeof(Shared); }
