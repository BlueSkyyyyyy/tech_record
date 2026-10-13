// P7-44: hand-written *plain CUDA* prototype of the B2 (`wy_dqkg`) V-reduction
// GEMM core -- the "first action" that §46.6 (P7-43) names for the K2-shaped CUDA
// port of the fused B2/B3:
//
//     "把 dq/dk/dw [BT,K] 累加器按 warp 分片常驻寄存器（先），TMA 输入流水（次）"
//
// §46 measured the deployed Triton fused B2/B3 at `B8 T2048 H8`: 255 regs + 114 KB
// smem -> **1 block/SM**, `Eligible Warps/Scheduler 0.18`, issue 0.14/cyc, `long_sb`
// 41.7% -> latency/issue-starved, and a same-traffic no-`tl.dot` kernel reaches
// **509 us** (3 blocks/SM, ~3.0 TB/s) vs the fused kernel's 988 us (1.94x).  The
// claim to test here is structural: can a *plain SIMT* CUDA kernel that keeps the
// `dq/dk/dw [BT,K]` accumulators register-resident and runs at high occupancy beat
// the Triton program shape on the *identical* dominant GEMMs?
//
// This file implements only the three V-reduction GEMMs of B2 plus the `dgk` column
// reduction -- the FLOP-dominant part of B2 -- not the finalize / `dA` / `dv2` / `db`
// / intra fold (those are left to a later increment; see docs/07 §47.6).  The point
// of an isolated core is an *apples-to-apples* comparison: the Python side races it
// against a Triton kernel computing exactly the same four tensors from the same
// operands.
//
// Per chunk (deployed BK=K=128, BV=V=128, BT=16):
//     dq[t,k] = sum_v do[t,v]   * h [k,v]        (= do   @ b_h)
//     dk[t,k] = sum_v v_new[t,v]* dh[k,v]        (= v_new@ b_dh)
//     dw[t,k] = sum_v dv[t,v]   * h [k,v]        (= dv   @ b_h)
//     dgk[k]  = sum_v h[k,v] * dh[k,v]
// (FLA/Triton load `b_h[v,k] = h[k,v]`, hence the `@ h^T` in the reference.)
//
// Thread map (block = 256 = 8 warps): thread `tid` owns column `kk = tid % K` and
// the 8 rows `tbase..tbase+7` where `tbase = (tid / K) * 8`.  So `dq/dk/dw` are 24
// fp32 registers per thread (8 rows x 3), `dgk` is 1; warp lanes own 32 consecutive
// `kk` (coalesced `h/dh` smem) while `do/v_new/dv` reads broadcast within a warp.
//
// Build (inside kernel_lab), same as the other `*_probe.cu`:
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b2_core_probe.cu -o /tmp/b2_core.so
// Host entry: `extern "C" b2_core_launch(...)`, called from
// `harness/check_bwd_b2_core.py` via ctypes with raw device pointers.
#include <cuda_runtime.h>
#include <cstdint>

#define DEV __device__ __forceinline__

constexpr int KD = 128;   // K = D
constexpr int VD = 128;   // V = D
constexpr int BTV = 16;   // BT (deployed chunk)
constexpr int BVT = 32;   // V tile staged in smem
constexpr int NTHR = 256; // 8 warps

struct Shared {
    float h [BVT][KD];    // h [k, v] staged (v = vt+r)
    float dh[BVT][KD];    // dh[k, v]
    float dov[BTV][BVT];  // do  [t, v] tile
    float vn [BTV][BVT];  // v_new[t, v] tile
    float dv [BTV][BVT];  // dv  [t, v] tile
};

__global__ void __launch_bounds__(NTHR)
b2_core_kernel(const float* __restrict__ v,
               const float* __restrict__ vn,
               const float* __restrict__ h,
               const float* __restrict__ dh,
               const float* __restrict__ dov,   // do
               const float* __restrict__ dvv,   // dv
               float* __restrict__ dq,
               float* __restrict__ dk,
               float* __restrict__ dw,
               float* __restrict__ dgk,
               int B, int T, int H, int NT) {
    __shared__ Shared sh;
    const int i_t  = blockIdx.x;
    const int i_bh = blockIdx.y;
    const int b    = i_bh / H;
    const int hh   = i_bh % H;
    const int tid  = threadIdx.x;
    const int kk   = tid % KD;
    const int tbase = (tid / KD) * 8;
    const int t0   = i_t * BTV;

    float aq[8], ak[8], aw[8];
#pragma unroll
    for (int x = 0; x < 8; ++x) { aq[x] = 0.f; ak[x] = 0.f; aw[x] = 0.f; }
    float ag = 0.f;

    const int64_t hbase = ((int64_t)(b * NT + i_t) * H + hh) * KD * VD;

    for (int vt = 0; vt < VD; vt += BVT) {
        // stage h/dh [BVT, KD] (k-major rows: sh_h[r][k] = h[k, vt+r])
        for (int idx = tid; idx < BVT * KD; idx += NTHR) {
            int r = idx / KD, c = idx % KD;
            sh.h [r][c] = h [hbase + (int64_t)c * VD + (vt + r)];
            sh.dh[r][c] = dh[hbase + (int64_t)c * VD + (vt + r)];
        }
        // stage do/v_new/dv [BT, BVT] tiles
        for (int idx = tid; idx < BTV * BVT; idx += NTHR) {
            int t = idx / BVT, vv = idx % BVT;
            int tg = t0 + t, vg = vt + vv;
            if (tg < T) {
                int64_t off = ((int64_t)(b * T + tg) * H + hh) * VD + vg;
                sh.dov[t][vv] = dov[off];
                sh.vn [t][vv] = vn [off];
                sh.dv [t][vv] = dvv[off];
            } else {
                sh.dov[t][vv] = 0.f; sh.vn[t][vv] = 0.f; sh.dv[t][vv] = 0.f;
            }
        }
        __syncthreads();

#pragma unroll 4
        for (int vv = 0; vv < BVT; ++vv) {
            const float hk  = sh.h [vv][kk];
            const float dhk = sh.dh[vv][kk];
            ag += hk * dhk;
#pragma unroll
            for (int x = 0; x < 8; ++x) {
                aq[x] += sh.dov[tbase + x][vv] * hk;
                ak[x] += sh.vn [tbase + x][vv] * dhk;
                aw[x] += sh.dv [tbase + x][vv] * hk;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int x = 0; x < 8; ++x) {
        int tg = t0 + tbase + x;
        if (tg < T) {
            int64_t off = ((int64_t)(b * T + tg) * H + hh) * KD + kk;
            dq[off] = aq[x];
            dk[off] = ak[x];
            dw[off] = aw[x];
        }
    }
    // one thread per (k) column writes the V-reduced gate term (tbase==0 duplicate is
    // harmless: every thread owning column kk computes the identical `ag`).
    if (tbase == 0)
        dgk[((int64_t)(b * NT + i_t) * H + hh) * KD + kk] = ag;
}

extern "C" int b2_core_launch(const void* v, const void* vn, const void* h, const void* dh,
                              const void* dov, const void* dvv,
                              void* dq, void* dk, void* dw, void* dgk,
                              int B, int T, int H, int NT, void* stream) {
    dim3 grid(NT, B * H);
    b2_core_kernel<<<grid, NTHR, 0, (cudaStream_t)stream>>>(
        (const float*)v, (const float*)vn, (const float*)h, (const float*)dh,
        (const float*)dov, (const float*)dvv,
        (float*)dq, (float*)dk, (float*)dw, (float*)dgk,
        B, T, H, NT);
    return (int)cudaGetLastError();
}

extern "C" int b2_core_smem_bytes() { return (int)sizeof(Shared); }
