// P7-19: hand-written CUDA B1 (state adjoint / `dhu`) in the *K2 shape* flagged by
// P7-16..P7-18.  Every prior controlled experiment (P7-15..P7-18) kept the
// `wmma`-from-smem structure and only moved cheap knobs (padding, loop order, CTA
// shape); the residual to FLA `dhu` stayed 4.9-8x, and §21.6 pinned the *magnitude*
// term as "每 warp 的有效 mma 密度" (tensor 2.3% vs FLA 14%) rather than warp count /
// instruction-level ILP.
//
// This prototype attacks exactly that term: the running state S is kept **in
// registers** as mma accumulator fragments (no fp32 smem master, no fp32 staging
// round-trips), and the mma is issued with raw `mma.sync.m16n8k16` fed by
// `ldmatrix` from XOR-padded smem (conflict-free without the +29% smem of §20's
// padding on the *operand* arrays).  The recurrence is unchanged and mirrors
// check_bwd_dhu_kernel.py / fla/ops/common/chunk_delta_h.py:576-587:
//
//   for c = NT-1 .. 0:
//       dh[c] = bf16(S)
//       du = k @ S + dv ;  dv2[c] = bf16(du)
//       S  = S*e + scale*(q^T @ do) - (w^T @ du)
//   dh0 = S
//
// register state per warp: S as (K/16)*2 m16n8 accumulators (8*2*4 = 64 fp32).
// Each warp owns BW=16 V columns; BV = NW*BW columns per CTA.
//
// Build (inside kernel_lab):
//   nvcc -std=c++17 -O3 -arch=sm_90a -Xcompiler -fPIC -shared b1_reg_probe.cu -o /tmp/b1_reg.so
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cstdint>

#define DEV __device__ __forceinline__
using bf16 = __nv_bfloat16;

constexpr int KD = 128;   // K = D
constexpr int VD = 128;   // V = D

#ifndef B1R_BV
#define B1R_BV 32
#endif
#ifndef B1R_NW
#define B1R_NW 2
#endif
constexpr int BV = B1R_BV;    // V columns per CTA
constexpr int NW = B1R_NW;    // warps per CTA
constexpr int BW = BV / NW;   // V columns per warp (== 16)

// XOR-padding (a degenerate swizzle) making every ldmatrix 8x8 conflict-free:
// row stride in 32-bit words must be odd-ish so the 8 rows land on distinct banks.
constexpr int KDS   = KD + 8;    // 136 bf16 = 272 B -> stride 68 words ~ +4 banks/row
constexpr int BWS   = BW + 8;    // 24  bf16 = 48  B -> stride 12 words
constexpr int BVS   = BV + 8;    // per-CTA do/dv row stride

DEV float b2f(bf16 x) { return __bfloat162float(x); }
DEV bf16  f2b(float x) { return __float2bfloat16(x); }
DEV uint32_t saddr(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

// ---- ldmatrix helpers -------------------------------------------------------
// x4, row-major 16x16 A tile (rows/cols selected by (rowbase,colbase)).
DEV void ldsm_x4(uint32_t o[4], const bf16* base, int rowbase, int colbase,
                 int stride, int l) {
    int mq = l >> 3, r = l & 7;
    int rr = rowbase + ((mq & 1) ? 8 : 0);
    int cc = colbase + ((mq & 2) ? 8 : 0);
    uint32_t a = saddr(base) + (uint32_t)((rr + r) * stride + cc) * 2u;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(a));
}
// x4 .trans, for A = (row-major [K,M] memory) ^ T -> [M,K] fragment.  For .trans the
// four output registers a0..a3 correspond to memory sub-tiles (rowblk,colblk) =
// (0,0),(0,1),(1,0),(1,1) (rowblk = K/token, colblk = M), unlike the non-trans order.
DEV void ldsm_x4_t(uint32_t o[4], const bf16* base, int rowbase, int colbase,
                   int stride, int l) {
    int mq = l >> 3, r = l & 7;
    int rr = rowbase + ((mq >= 2) ? 8 : 0);
    int cc = colbase + ((mq & 1) ? 8 : 0);
    uint32_t a = saddr(base) + (uint32_t)((rr + r) * stride + cc) * 2u;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(o[0]), "=r"(o[1]), "=r"(o[2]), "=r"(o[3]) : "r"(a));
}
// x2 .trans, for B = row-major [K,N] memory -> [K,N] mma B fragment (b0,b1).
DEV void ldsm_x2_t(uint32_t o[2], const bf16* base, int rowbase, int colbase,
                   int stride, int l) {
    int r = l & 15;
    uint32_t a = saddr(base) + (uint32_t)((rowbase + r) * stride + colbase) * 2u;
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(o[0]), "=r"(o[1]) : "r"(a));
}
// m16n8k16 bf16 -> fp32, row.col  (D = A*B + C)
DEV void mma16816(float d[4], const uint32_t a[4], const uint32_t b[2], const float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <int BT>
struct Stage {
    bf16 k[BT][KDS];
    bf16 q[BT][KDS];
    bf16 w[BT][KDS];
    bf16 dO[BT][BVS];
    bf16 dV[BT][BVS];
    float e[KD];
};

template <int BT, int NSTAGE>
struct alignas(128) Shared {
    Stage<BT> st[NSTAGE];
    bf16 sdh[NW][KD][BWS];   // per-warp bf16 mirror of S (op1 B operand)
    bf16 sdu[NW][BT][BWS];   // per-warp bf16 du   (op3 B operand)
};

// --------------------------------------------------------------------------- load
template <int BT>
DEV void prefetch(Stage<BT>* st,
                  const bf16* __restrict__ q, const bf16* __restrict__ k,
                  const bf16* __restrict__ w,
                  const bf16* __restrict__ do_, const bf16* __restrict__ dv,
                  const float* __restrict__ etot,
                  int n, int h, int T, int H, int NT, int c, int v0) {
    const int t0 = c * BT, tid = threadIdx.x, nthr = blockDim.x;
    for (int idx = tid; idx < BT * (KD / 8); idx += nthr) {
        int row = idx / (KD / 8), cg = idx % (KD / 8), t = t0 + row;
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + h) * KD + cg * 8;
            __pipeline_memcpy_async(&st->k[row][cg * 8], k + off, 16);
            __pipeline_memcpy_async(&st->q[row][cg * 8], q + off, 16);
            __pipeline_memcpy_async(&st->w[row][cg * 8], w + off, 16);
        } else {
            for (int j = 0; j < 8; ++j) { st->k[row][cg * 8 + j] = f2b(0.f);
                                          st->q[row][cg * 8 + j] = f2b(0.f);
                                          st->w[row][cg * 8 + j] = f2b(0.f); }
        }
    }
    for (int idx = tid; idx < BT * (BV / 8); idx += nthr) {
        int row = idx / (BV / 8), cg = idx % (BV / 8), t = t0 + row;
        if (t < T) {
            const int64_t off = ((int64_t)(n * T + t) * H + h) * VD + v0 + cg * 8;
            __pipeline_memcpy_async(&st->dO[row][cg * 8], do_ + off, 16);
            __pipeline_memcpy_async(&st->dV[row][cg * 8], dv + off, 16);
        } else {
            for (int j = 0; j < 8; ++j) { st->dO[row][cg * 8 + j] = f2b(0.f);
                                          st->dV[row][cg * 8 + j] = f2b(0.f); }
        }
    }
    for (int idx = tid; idx < KD / 4; idx += nthr) {
        const int64_t off = ((int64_t)(n * NT + c) * H + h) * KD + idx * 4;
        __pipeline_memcpy_async(&st->e[idx * 4], etot + off, 16);
    }
}

// ------------------------------------------------------------------------ kernel
template <int BT, int NSTAGE>
__global__ void __launch_bounds__(NW * 32)
b1_reg_cuda(const bf16* __restrict__ q, const bf16* __restrict__ k,
            const bf16* __restrict__ w, const bf16* __restrict__ do_,
            const bf16* __restrict__ dv, const float* __restrict__ etot,
            bf16* __restrict__ dh_g, float* __restrict__ dh0_g, bf16* __restrict__ dv2_g,
            int T, int NT, int H, float scale) {
    extern __shared__ char smem_raw[];
    Shared<BT, NSTAGE>* sh = reinterpret_cast<Shared<BT, NSTAGE>*>(smem_raw);

    const int v0 = blockIdx.x * BV;
    const int nh = blockIdx.y;
    const int n = nh / H, h = nh % H;
    const int l = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int g = l >> 2, t = l & 3;

    bf16 (*sdh)[BWS] = sh->sdh[warp];
    bf16 (*sdu)[BWS] = sh->sdu[warp];

    // running state S: [K,BW] as MTK=K/16 m-tiles x 2 n-tiles, m16n8 fp32 accum
    constexpr int MTK = KD / 16;         // 8
    constexpr int MTKt = BT / 16;        // op1 m-tiles
    constexpr int KT1 = KD / 16;         // op1 k-steps = 8
    constexpr int KT2 = BT / 16;         // op2/op3 k-steps

    float S[MTK][2][4];
    #pragma unroll
    for (int m = 0; m < MTK; ++m)
        #pragma unroll
        for (int nn = 0; nn < 2; ++nn)
            #pragma unroll
            for (int i = 0; i < 4; ++i) S[m][nn][i] = 0.f;

    const __nv_bfloat162 scale2 = __floats2bfloat162_rn(scale, scale);

    for (int j = 0; j < NSTAGE - 1; ++j) {
        if (j < NT) prefetch<BT>(&sh->st[j], q, k, w, do_, dv, etot, n, h, T, H, NT, NT - 1 - j, v0);
        __pipeline_commit();
    }

    for (int i = 0; i < NT; ++i) {
        const int c = NT - 1 - i;
        const Stage<BT>* st = &sh->st[i % NSTAGE];
        const int cload = c - (NSTAGE - 1);
        if (cload >= 0)
            prefetch<BT>(&sh->st[(i + NSTAGE - 1) % NSTAGE], q, k, w, do_, dv, etot,
                         n, h, T, H, NT, cload, v0);
        __pipeline_commit();
        __pipeline_wait_prior(NSTAGE - 1);
        __syncthreads();

        // ---- publish S: global dh[c] + per-warp bf16 mirror ----
        {
            const int64_t base = ((int64_t)(n * NT + c) * H + h) * KD * VD;
            #pragma unroll
            for (int m = 0; m < MTK; ++m)
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    int r0 = m * 16 + g, r1 = r0 + 8;
                    int cc = nn * 8 + t * 2;   // local V col 0..15
                    bf16 v0v = f2b(S[m][nn][0]), v1v = f2b(S[m][nn][1]);
                    bf16 v2v = f2b(S[m][nn][2]), v3v = f2b(S[m][nn][3]);
                    sdh[r0][cc] = v0v; sdh[r0][cc + 1] = v1v;
                    sdh[r1][cc] = v2v; sdh[r1][cc + 1] = v3v;
                    int gc = v0 + warp * BW + cc;
                    dh_g[base + (int64_t)r0 * VD + gc] = v0v;
                    dh_g[base + (int64_t)r0 * VD + gc + 1] = v1v;
                    dh_g[base + (int64_t)r1 * VD + gc] = v2v;
                    dh_g[base + (int64_t)r1 * VD + gc + 1] = v3v;
                }
        }
        __syncwarp();

        // ---- op1: cdu[BT,BW] = k[BT,KD] @ S[KD,BW] ----
        float cdu[MTKt][2][4];
        #pragma unroll
        for (int m = 0; m < MTKt; ++m)
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn)
                #pragma unroll
                for (int x = 0; x < 4; ++x) cdu[m][nn][x] = 0.f;

        #pragma unroll
        for (int m = 0; m < MTKt; ++m)
            #pragma unroll
            for (int kk = 0; kk < KT1; ++kk) {
                uint32_t a[4];
                ldsm_x4(a, &st->k[0][0], m * 16, kk * 16, KDS, l);
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    uint32_t b[2];
                    ldsm_x2_t(b, &sdh[0][0], kk * 16, nn * 8, BWS, l);
                    mma16816(cdu[m][nn], a, b, cdu[m][nn]);
                }
            }
        __syncwarp();

        // ---- du = cdu + dv ; write dv2 and per-warp bf16 du (op3 B) ----
        {
            const int vbase = v0 + warp * BW;
            #pragma unroll
            for (int m = 0; m < MTKt; ++m)
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    int r0 = m * 16 + g, r1 = r0 + 8;
                    int cc = nn * 8 + t * 2;
                    int tr0 = c * BT + r0, tr1 = c * BT + r1;
                    float d0 = cdu[m][nn][0], d1 = cdu[m][nn][1];
                    float d2 = cdu[m][nn][2], d3 = cdu[m][nn][3];
                    if (tr0 < T) {
                        d0 += b2f(st->dV[r0][warp * BW + cc]);
                        d1 += b2f(st->dV[r0][warp * BW + cc + 1]);
                    }
                    if (tr1 < T) {
                        d2 += b2f(st->dV[r1][warp * BW + cc]);
                        d3 += b2f(st->dV[r1][warp * BW + cc + 1]);
                    }
                    bf16 b0 = f2b(d0), b1 = f2b(d1), b2 = f2b(d2), b3 = f2b(d3);
                    sdu[r0][cc] = b0; sdu[r0][cc + 1] = b1;
                    sdu[r1][cc] = b2; sdu[r1][cc + 1] = b3;
                    if (tr0 < T) {
                        int64_t o0 = ((int64_t)(n * T + tr0) * H + h) * VD + vbase + cc;
                        dv2_g[o0] = b0; dv2_g[o0 + 1] = b1;
                    }
                    if (tr1 < T) {
                        int64_t o1 = ((int64_t)(n * T + tr1) * H + h) * VD + vbase + cc;
                        dv2_g[o1] = b2; dv2_g[o1 + 1] = b3;
                    }
                }
        }
        __syncwarp();

        // ---- op2/op3: S = S*e + scale*(q^T @ do) - (w^T @ du) ----
        #pragma unroll
        for (int m = 0; m < MTK; ++m) {
            float e0 = st->e[m * 16 + g], e1 = st->e[m * 16 + g + 8];
            #pragma unroll
            for (int kk = 0; kk < KT2; ++kk) {
                uint32_t aq[4], aw[4];
                ldsm_x4_t(aq, &st->q[0][0], kk * 16, m * 16, KDS, l);
                ldsm_x4_t(aw, &st->w[0][0], kk * 16, m * 16, KDS, l);
                #pragma unroll
                for (int x = 0; x < 4; ++x) aw[x] = aw[x] ^ 0x80008000u;  // -w^T
                #pragma unroll
                for (int nn = 0; nn < 2; ++nn) {
                    if (kk == 0) {
                        S[m][nn][0] *= e0; S[m][nn][1] *= e0;
                        S[m][nn][2] *= e1; S[m][nn][3] *= e1;
                    }
                    uint32_t bd[2], bu[2];
                    ldsm_x2_t(bd, &st->dO[0][0], kk * 16, warp * BW + nn * 8, BVS, l);
                    union U { uint32_t u; __nv_bfloat162 b2; } x0, x1;
                    x0.u = bd[0]; x0.b2 = __hmul2(x0.b2, scale2); bd[0] = x0.u;
                    x1.u = bd[1]; x1.b2 = __hmul2(x1.b2, scale2); bd[1] = x1.u;
                    mma16816(S[m][nn], aq, bd, S[m][nn]);
                    ldsm_x2_t(bu, &sdu[0][0], kk * 16, nn * 8, BWS, l);
                    mma16816(S[m][nn], aw, bu, S[m][nn]);
                }
            }
        }
        __syncthreads();
    }

    // final: dh0 = S
    {
        const int64_t base = ((int64_t)nh) * KD * VD;
        #pragma unroll
        for (int m = 0; m < MTK; ++m)
            #pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                int r0 = m * 16 + g, r1 = r0 + 8;
                int gc = v0 + warp * BW + nn * 8 + t * 2;
                dh0_g[base + (int64_t)r0 * VD + gc] = S[m][nn][0];
                dh0_g[base + (int64_t)r0 * VD + gc + 1] = S[m][nn][1];
                dh0_g[base + (int64_t)r1 * VD + gc] = S[m][nn][2];
                dh0_g[base + (int64_t)r1 * VD + gc + 1] = S[m][nn][3];
            }
    }
}

// ------------------------------------------------------------------------- launch
template <int BT, int NSTAGE>
static int launch_impl(const void* q, const void* k, const void* w, const void* do_,
                       const void* dv, const void* etot, void* dh, void* dh0, void* dv2,
                       int B, int T, int H, int V, int NT, float scale, cudaStream_t s) {
    dim3 grid(V / BV, B * H);
    int smem = (int)sizeof(Shared<BT, NSTAGE>);
    static int configured = 0;
    if (!configured) {
        if (cudaFuncSetAttribute(b1_reg_cuda<BT, NSTAGE>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem) != cudaSuccess)
            return -3;
        configured = 1;
    }
    b1_reg_cuda<BT, NSTAGE><<<grid, NW * 32, smem, s>>>(
        (const bf16*)q, (const bf16*)k, (const bf16*)w, (const bf16*)do_,
        (const bf16*)dv, (const float*)etot, (bf16*)dh, (float*)dh0, (bf16*)dv2,
        T, NT, H, scale);
    return (int)cudaGetLastError();
}

extern "C" int b1r_launch(int BT, int NSTAGE,
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

extern "C" int b1r_smem_bytes(int BT, int NSTAGE) {
    if (BT == 64) { if (NSTAGE == 1) return sizeof(Shared<64,1>); if (NSTAGE == 2) return sizeof(Shared<64,2>); if (NSTAGE == 3) return sizeof(Shared<64,3>); }
    if (BT == 32) { if (NSTAGE == 1) return sizeof(Shared<32,1>); if (NSTAGE == 2) return sizeof(Shared<32,2>); if (NSTAGE == 3) return sizeof(Shared<32,3>); }
    if (BT == 16) { if (NSTAGE == 1) return sizeof(Shared<16,1>); if (NSTAGE == 2) return sizeof(Shared<16,2>); if (NSTAGE == 3) return sizeof(Shared<16,3>); }
    return -1;
}
