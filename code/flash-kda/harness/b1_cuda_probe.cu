// P7-15: hand-written *plain CUDA* B1 (state adjoint / `dhu`) prototype, to answer the
// open boundary left by P7-14 (docs/07 §17): the two §16.5 design inputs were measured on
// a **Triton** B1 prototype, and Triton's automatic software pipeliner concluded that
//    (1) pipelining the reversed-chunk loads (mirror K2's TMA input pipeline) is the whole
//        win -> B1 1.96x -> 1.00x/0.85x FLA `dhu`;
//    (2) splitting the D=128 state into 2 accumulator chains is redundant / negative.
// The remaining question was whether an explicit hand-written CUDA kernel (where the
// prefetch and the accumulator are under our control, not the compiler's) still needs the
// manual split chain.  This file rebuilds B1 in CUDA and toggles only the load pipeline.
//
// Recurrence, exactly mirroring `check_bwd_dhu_kernel.py` / `check_bwd_b1_ilp.py` and
// fla/ops/common/chunk_delta_h.py:576-587 (reversed chunks, per (n,h), per V-tile):
//
//   for c = NT-1 .. 0:
//       dh_store[c] = bf16(S)                         # state entering chunk c
//       du = k @ S_bf16 + dv                          # [BT, BV]
//       dv2[c] = bf16(du)
//       S  = S*e[:,None] + scale * (q^T @ do) - (w^T @ du)
//   dh0 = S
//
// `e` (= E^tot, [K]) is the chunk's total decay.  Operands are bf16, the running state S is
// fp32 (mma accumulates fp32), mma is bf16 (m16n16k16 wmma) so it hits the same tensor-core
// path as FLA.  The ONLY structural knob here is NSTAGE: the number of `cp.async` in-flight
// chunk stages (NSTAGE=1 == issue-then-wait immediately, no overlap; NSTAGE=2/3 == prefetch
// the next chunk(s) while computing the current one).
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b1_cuda_probe.cu -o /tmp/b1_cuda.so
// The host launcher is `extern "C" b1_launch(...)` and is called from
// `harness/check_bwd_b1_cuda.py` via ctypes with raw device pointers.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define DEV __device__ __forceinline__
using bf16 = __nv_bfloat16;

constexpr int KD = 128;   // K = D
constexpr int VD = 128;   // V = D
#ifndef B1_BV
#define B1_BV 32
#endif
#ifndef B1_NW
#define B1_NW 2
#endif
constexpr int BV = B1_BV;    // columns handled per CTA
constexpr int NW = B1_NW;    // warps per CTA (blockDim = NW*32)
constexpr int BW = BV / NW;  // columns per warp == wmma n (must be 16)

__device__ __forceinline__ float f2f(bf16 x) { return __bfloat162float(x); }
__device__ __forceinline__ bf16 f2b(float x) { return __float2bfloat16(x); }

// B1_SYNC: replace cp.async with plain shared stores (synchronous-load baseline)
__device__ __forceinline__ void load16(void* dst, const void* src) {
#ifdef B1_SYNC
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
#else
    __pipeline_memcpy_async(dst, src, 16);
#endif
}

template <int BT>
struct Stage {
    bf16 k[BT][KD];
    bf16 q[BT][KD];
    bf16 w[BT][KD];
    bf16 dO[BT][BV];   // do
    bf16 dV[BT][BV];   // dv
    float e[KD];
};

template <int BT, int NSTAGE>
struct alignas(128) Shared {
    Stage<BT> st[NSTAGE];
    float    dh[NW][KD][BW];     // running state S (fp32)
    bf16     sdh[NW][KD][BW];    // rounded S, mma B operand of op1
    bf16     sdu[NW][BT][BW];    // du (mma B operand of op3 / dv2 source)
    float    sduf[NW][BT][BW];   // du fp32 staging
    float    sacc[NW][KD][BW];   // (q^T do * scale - w^T du) staging
};

// --------------------------------------------------------------------------- prefetch
template <int BT>
__device__ __forceinline__ void prefetch(Stage<BT>* st,
                                         const bf16* __restrict__ q, const bf16* __restrict__ k,
                                         const bf16* __restrict__ w,
                                         const bf16* __restrict__ do_, const bf16* __restrict__ dv,
                                         const float* __restrict__ etot,
                                         int n, int h, int T, int H, int NT, int c, int v0) {
    const int t0 = c * BT;
    const int tid = threadIdx.x;
    const int nthr = blockDim.x;

    // k,q,w : [BT, KD] bf16, 8 bf16 == 16 B per cp.async
    for (int idx = tid; idx < BT * (KD / 8); idx += nthr) {
        int row = idx / (KD / 8), cg = idx % (KD / 8);
        int t = t0 + row;
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + h) * KD + cg * 8;
            load16(&st->k[row][cg * 8], k + off);
            load16(&st->q[row][cg * 8], q + off);
            load16(&st->w[row][cg * 8], w + off);
        } else {
            for (int j = 0; j < 8; ++j) { st->k[row][cg * 8 + j] = f2b(0.f);
                                          st->q[row][cg * 8 + j] = f2b(0.f);
                                          st->w[row][cg * 8 + j] = f2b(0.f); }
        }
    }
    // do,dv : [BT, BV]
    for (int idx = tid; idx < BT * (BV / 8); idx += nthr) {
        int row = idx / (BV / 8), cg = idx % (BV / 8);
        int t = t0 + row;
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + h) * VD + v0 + cg * 8;
            load16(&st->dO[row][cg * 8], do_ + off);
            load16(&st->dV[row][cg * 8], dv + off);
        } else {
            for (int j = 0; j < 8; ++j) { st->dO[row][cg * 8 + j] = f2b(0.f);
                                          st->dV[row][cg * 8 + j] = f2b(0.f); }
        }
    }
    // e : [KD] fp32, 4 floats == 16 B
    for (int idx = tid; idx < KD / 4; idx += nthr) {
        const int64_t off = ((int64_t)(n * NT + c) * H + h) * KD + idx * 4;
        load16(&st->e[idx * 4], etot + off);
    }
}

// ----------------------------------------------------------------------------- kernel
template <int BT, int NSTAGE>
__global__ void __launch_bounds__(NW * 32)
b1_dhu_cuda(const bf16* __restrict__ q, const bf16* __restrict__ k,
            const bf16* __restrict__ w, const bf16* __restrict__ do_,
            const bf16* __restrict__ dv, const float* __restrict__ etot,
            bf16* __restrict__ dh_g, float* __restrict__ dh0_g, bf16* __restrict__ dv2_g,
            int T, int NT, int H, float scale) {
    extern __shared__ char smem_raw[];
    Shared<BT, NSTAGE>* sh = reinterpret_cast<Shared<BT, NSTAGE>*>(smem_raw);

    const int v0 = blockIdx.x * BV;
    const int nh = blockIdx.y;
    const int n = nh / H;
    const int h = nh % H;
    const int lane = threadIdx.x;
    const int warp = lane >> 5;
    const int wl = lane & 31;

    const int vw = v0 + warp * BW;   // this warp's V column base

    float (*S)[BW]   = sh->dh[warp];
    bf16  (*Sb)[BW]  = sh->sdh[warp];
    bf16  (*du)[BW]  = sh->sdu[warp];
    float (*duf)[BW] = sh->sduf[warp];
    float (*acc)[BW] = sh->sacc[warp];

    // zero the persistent state
    for (int idx = wl; idx < KD * BW; idx += 32) {
        ((float*)S)[idx] = 0.f;
    }
    __syncthreads();

    constexpr int MT1 = BT / 16;   // op1 m-tiles
    constexpr int MT2 = KD / 16;   // op2/op3 m-tiles (m = K = 128)
    constexpr int KT1 = KD / 16;   // op1 k-steps
    constexpr int KT2 = BT / 16;   // op2/op3 k-steps

    using FA  = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major>;
    using FAT = wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::col_major>;
    using FB  = wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major>;
    using FC  = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    // prologue: issue the first NSTAGE-1 chunks (chunk NT-1 -> stage 0, ...)
    for (int j = 0; j < NSTAGE - 1; ++j) {
        if (j < NT)
            prefetch<BT>(&sh->st[j], q, k, w, do_, dv, etot, n, h, T, H, NT, NT - 1 - j, v0);
        __pipeline_commit();
    }

    for (int i = 0; i < NT; ++i) {
        const int c = NT - 1 - i;
        const Stage<BT>* st = &sh->st[i % NSTAGE];

        // issue the load for the chunk (NSTAGE-1) ahead into its own ring slot
        const int cload = c - (NSTAGE - 1);
        if (cload >= 0)
            prefetch<BT>(&sh->st[(i + NSTAGE - 1) % NSTAGE], q, k, w, do_, dv, etot,
                         n, h, T, H, NT, cload, v0);
        __pipeline_commit();   // commit even if empty keeps the group count aligned
        // keep at most NSTAGE-1 groups pending -> the current chunk's copy is done
        __pipeline_wait_prior(NSTAGE - 1);
        __syncthreads();

        // ---- step 1: publish S to global dh[c] and round it for op1's B operand ----
        {
            const int64_t base = ((int64_t)(n * NT + c) * H + h) * KD * VD;
            for (int idx = wl; idx < KD * BW; idx += 32) {
                int kk = idx / BW, jj = idx % BW;
                float v = ((float*)S)[idx];
                Sb[kk][jj] = f2b(v);
                dh_g[base + (int64_t)kk * VD + vw + jj] = f2b(v);
            }
        }
        __syncthreads();

        // ---- op1: du[BT,BW] = k[BT,KD] @ S[KD,BW] ----
        {
            FC cdu[MT1];
            for (int m = 0; m < MT1; ++m) wmma::fill_fragment(cdu[m], 0.f);
            for (int m = 0; m < MT1; ++m)
                for (int kk = 0; kk < KT1; ++kk) {
                    FA a; wmma::load_matrix_sync(a, &st->k[m * 16][kk * 16], KD);
                    FB b; wmma::load_matrix_sync(b, &Sb[kk * 16][0], BW);
                    wmma::mma_sync(cdu[m], a, b, cdu[m]);
                }
            for (int m = 0; m < MT1; ++m)
                wmma::store_matrix_sync(&duf[m * 16][0], cdu[m], BW, wmma::mem_row_major);
        }
        __syncthreads();

        // ---- step 2: du += dv; publish dv2 and du (bf16) for op3 ----
        {
            for (int idx = wl; idx < BT * BW; idx += 32) {
                int row = idx / BW, jj = idx % BW;
                int t = c * BT + row;
                float v = duf[row][jj];
                if (t < T) v += f2f(st->dV[row][warp * BW + jj]);
                du[row][jj] = f2b(v);
                if (t < T)
                    dv2_g[((int64_t)(n * T + t) * H + h) * VD + vw + jj] = f2b(v);
            }
        }
        __syncthreads();

        // ---- op2/op3: acc = scale*(q^T @ do) - (w^T @ du) ----
        {
            for (int m = 0; m < MT2; ++m) {
                FC cq, cw;
                wmma::fill_fragment(cq, 0.f);
                wmma::fill_fragment(cw, 0.f);
                for (int kk = 0; kk < KT2; ++kk) {
                    FAT aq; wmma::load_matrix_sync(aq, &st->q[kk * 16][m * 16], KD);
                    FB  bd; wmma::load_matrix_sync(bd, &st->dO[kk * 16][warp * BW], BV);
                    wmma::mma_sync(cq, aq, bd, cq);

                    FAT aw; wmma::load_matrix_sync(aw, &st->w[kk * 16][m * 16], KD);
                    FB  bu; wmma::load_matrix_sync(bu, &du[kk * 16][0], BW);
                    wmma::mma_sync(cw, aw, bu, cw);
                }
                for (int j = 0; j < FC::num_elements; ++j)
                    cq.x[j] = cq.x[j] * scale - cw.x[j];
                wmma::store_matrix_sync(&acc[m * 16][0], cq, BW, wmma::mem_row_major);
            }
        }
        __syncthreads();

        // ---- step 3: S = S * e + acc ----
        for (int idx = wl; idx < KD * BW; idx += 32) {
            int kk = idx / BW;
            ((float*)S)[idx] = ((float*)S)[idx] * st->e[kk] + ((float*)acc)[idx];
        }
        __syncthreads();
    }

    // final: dh0 = S
    {
        const int64_t base = ((int64_t)nh) * KD * VD;
        for (int idx = wl; idx < KD * BW; idx += 32) {
            int kk = idx / BW, jj = idx % BW;
            dh0_g[base + (int64_t)kk * VD + vw + jj] = ((float*)S)[idx];
        }
    }
}

// ------------------------------------------------------------------------- launcher
template <int BT, int NSTAGE>
static int launch_impl(const void* q, const void* k, const void* w, const void* do_,
                       const void* dv, const void* etot, void* dh, void* dh0, void* dv2,
                       int B, int T, int H, int V, int NT, float scale, cudaStream_t s) {
    dim3 grid(V / BV, B * H);
    int smem = (int)sizeof(Shared<BT, NSTAGE>);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b1_dhu_cuda<BT, NSTAGE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return -3;   // smem exceeds the device limit
        configured = 1;
    }
    b1_dhu_cuda<BT, NSTAGE><<<grid, NW * 32, smem, s>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)w, (const bf16*)do_,
        (const bf16*)dv, (const float*)etot, (bf16*)dh, (float*)dh0, (bf16*)dv2,
        T, NT, H, scale);
    return (int)cudaGetLastError();
}

extern "C" int b1_launch(int BT, int NSTAGE,
                         const void* q, const void* k, const void* w, const void* do_,
                         const void* dv, const void* etot, void* dh, void* dh0, void* dv2,
                         int B, int T, int H, int V, int NT, float scale, void* stream) {
    if (V % BV != 0) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    if (BT == 64) {
        if (NSTAGE == 1) return launch_impl<64, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<64, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 3) return launch_impl<64, 3>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
    } else if (BT == 32) {
        if (NSTAGE == 1) return launch_impl<32, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<32, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 3) return launch_impl<32, 3>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
    } else if (BT == 16) {
        if (NSTAGE == 1) return launch_impl<16, 1>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 2) return launch_impl<16, 2>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
        if (NSTAGE == 3) return launch_impl<16, 3>(q, k, w, do_, dv, etot, dh, dh0, dv2, B, T, H, V, NT, scale, s);
    }
    return -1;
}

extern "C" int b1_smem_bytes(int BT, int NSTAGE) {
    if (BT == 64) { if (NSTAGE == 1) return sizeof(Shared<64,1>); if (NSTAGE == 2) return sizeof(Shared<64,2>); if (NSTAGE == 3) return sizeof(Shared<64,3>); }
    if (BT == 32) { if (NSTAGE == 1) return sizeof(Shared<32,1>); if (NSTAGE == 2) return sizeof(Shared<32,2>); if (NSTAGE == 3) return sizeof(Shared<32,3>); }
    if (BT == 16) { if (NSTAGE == 1) return sizeof(Shared<16,1>); if (NSTAGE == 2) return sizeof(Shared<16,2>); if (NSTAGE == 3) return sizeof(Shared<16,3>); }
    return -1;
}
