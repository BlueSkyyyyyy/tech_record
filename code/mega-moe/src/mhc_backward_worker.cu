// P122-1 实验：Mega mHC（HyperConnection）**反向**的 **per-token 融合 worker**——
// 把第 121 篇的 K1（系数反向）+ G1a（dz=dgrad）+ K2（混合反向）融进一个 block 处理一个 token，
// 省掉 `dAbase[T,D]` 与 `dz[T,D]` 两张全局中间张量的往返。这是前向 epilogue 的 per-token
// worker（`epilogue/sm100_mega_mhc.cuh:141` 的 `run_normal_worker`）在反向侧的对偶。
// 其余六段代码与 host double 参考全部沿用自 `src/mhc_backward_fused.cu`（P121-1），便于对拍。
//
// 背景（见 docs/130/131）：mHC 主 kernel（sm100_mega_mhc.cuh）是 SM100 专属、且**只有前向**，
// 它的反向按语义（tests/test_mega_mhc.py:123-210 的 mhc_reference）推导。第 130 篇验证了
// 「手写 VJP vs autograd」与「单片 device(fp32) vs double」；本篇换一个视角——把这条反向
// 拆成真正可以并行执行的算子流水，看它到底「重」在哪。
//
// 前向（投影 fn 是**全 batch 共享**的权重）：
//   A  = post ⊙ x + comb @ R                          [hc,H]   （K0，元素级）
//   u0 = A.flatten @ fnᵀ                               [Nout]   （G0，窄输出长 K：m=T,n=24,k=D）
//   ...  q/g、sigmoid/仿射、Sinkhorn、RMSNorm ...
//
// 反向（给定上游 dS/dPost/dM/dY[/dShift]）分成 6 段：
//   K0 重算 A
//   G0  u0 = A @ fnᵀ                                    （dgrad 型）
//   K1  系数反向（RMSNorm bwd + mix bwd + sigmoid/仿射 bwd + Sinkhorn VJP + q/g bwd）
//       → du0[T,24]、dAbase[T,D]、dgamma/dscales/dbases/dshift
//   G1a dz  = du0 @ fn                                  （dgrad：m=T,n=D,k=24）
//   G1b dfn = du0ᵀ @ A                                   （wgrad：m=24,n=D,k=T）
//   K2  残差多流混合反向（dA = dAbase + dz → dx/dR/dpost/dcomb）
//
// 本文件负责**全部正确性**（device 六段 vs host 单片 double 参考）与 K0/K1/K2 的真实设备计时；
// 三段 GEMM 的算术强度与真实形状计时见 harness/exp_mhc_backward_fused.py。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <type_traits>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1);} } while (0)

static constexpr int HC = 4;
static constexpr int NOUT = HC * (HC + 2);   // 24
static constexpr int MAXH = 64;
static constexpr int MAXIT = 8;
static constexpr int MAXNS = 2 * MAXIT + 2;

struct Params {
    double hc_norm_eps, hc_pre_eps, hc_post_scale, rmsnorm_eps, rmsnorm_scale, sinkhorn_eps;
    int iters;
    double scales[3];
};

template <typename T> __host__ __device__ inline T sigmoid(T z) {
    if constexpr (std::is_same<T, float>::value) return (T)(1.0f / (1.0f + expf(-z)));
    else return (T)(1.0 / (1.0 + exp(-(double)z)));
}

// ---------------------------------------------------------------------------
// Sinkhorn（test :114-120 / epilogue:63-86）前向 + 反向（逐行取自 docs/130 的 src）。
// ---------------------------------------------------------------------------
template <typename T>
static __host__ __device__ void sinkhorn_forward(const T logits[HC][HC], const Params& p,
                             T out[HC][HC], T seq[MAXNS][HC][HC], int dims[MAXNS], int& ns) {
    T psm[HC][HC];
    for (int j = 0; j < HC; ++j) {
        T mx = logits[j][0];
        for (int i = 1; i < HC; ++i) mx = logits[j][i] > mx ? logits[j][i] : mx;
        T s = 0;
        for (int i = 0; i < HC; ++i) { psm[j][i] = (T)exp((double)(logits[j][i] - mx)); s += psm[j][i]; }
        for (int i = 0; i < HC; ++i) psm[j][i] /= s;
    }
    T cur[HC][HC];
    for (int j = 0; j < HC; ++j)
        for (int i = 0; i < HC; ++i) cur[j][i] = psm[j][i] + (T)p.sinkhorn_eps;
    ns = 0;
    auto record = [&](const T x[HC][HC], int dim) {
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) seq[ns][j][i] = x[j][i];
        dims[ns] = dim; ++ns;
    };
    record(psm, 100);   // softmax 步（100 为标记）
    record(cur, -2);    // 第一步列归一化
    T S[HC] = {0,0,0,0};
    for (int i = 0; i < HC; ++i) for (int j = 0; j < HC; ++j) S[i] += cur[j][i];
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) cur[j][i] /= (S[i] + (T)p.sinkhorn_eps);
    for (int it = 1; it < p.iters; ++it) {
        record(cur, -1);
        T Sr[HC] = {0,0,0,0};
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) Sr[j] += cur[j][i];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) cur[j][i] /= (Sr[j] + (T)p.sinkhorn_eps);
        record(cur, -2);
        T Sc[HC] = {0,0,0,0};
        for (int i = 0; i < HC; ++i) for (int j = 0; j < HC; ++j) Sc[i] += cur[j][i];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) cur[j][i] /= (Sc[i] + (T)p.sinkhorn_eps);
    }
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) out[j][i] = cur[j][i];
}

template <typename T>
static __host__ __device__ void sinkhorn_backward(const T dM[HC][HC], const T seq[MAXNS][HC][HC],
                             const int dims[MAXNS], int ns, const Params& p, T dlogits[HC][HC]) {
    T d[HC][HC];
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) d[j][i] = dM[j][i];
    for (int s = ns - 1; s >= 1; --s) {
        const T (*X)[HC] = seq[s];
        int dim = dims[s];
        T S[HC] = {0,0,0,0};
        if (dim == -1) { for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) S[j] += X[j][i]; }
        else           { for (int i = 0; i < HC; ++i) for (int j = 0; j < HC; ++j) S[i] += X[j][i]; }
        T dX[HC][HC];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) {
            T den = S[dim == -1 ? j : i] + (T)p.sinkhorn_eps;
            dX[j][i] = d[j][i] / den;
        }
        T dS[HC] = {0,0,0,0};
        if (dim == -1) { for (int j = 0; j < HC; ++j) { T acc = 0; for (int i = 0; i < HC; ++i) acc += d[j][i] * X[j][i]; dS[j] = -acc / ((S[j] + (T)p.sinkhorn_eps) * (S[j] + (T)p.sinkhorn_eps)); } }
        else           { for (int i = 0; i < HC; ++i) { T acc = 0; for (int j = 0; j < HC; ++j) acc += d[j][i] * X[j][i]; dS[i] = -acc / ((S[i] + (T)p.sinkhorn_eps) * (S[i] + (T)p.sinkhorn_eps)); } }
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) d[j][i] = dX[j][i] + dS[dim == -1 ? j : i];
    }
    {
        const T (*psm)[HC] = seq[0];
        for (int j = 0; j < HC; ++j) {
            T dot = 0;
            for (int i = 0; i < HC; ++i) dot += d[j][i] * psm[j][i];
            for (int i = 0; i < HC; ++i) dlogits[j][i] = psm[j][i] * (d[j][i] - dot);
        }
    }
}

// ---------------------------------------------------------------------------
// host 单片 double 参考（第 130 篇同款语义）：一个 token 的全部梯度。
// ---------------------------------------------------------------------------
template <typename T>
struct RefIn {
    T x[MAXH];
    T R[HC][MAXH];
    T post[HC];
    T comb[HC][HC];
    T fn[NOUT][HC * MAXH];
    T scales[3];
    T bases[NOUT];
    T gamma[MAXH];
    T spm[HC];
    T dS[HC][MAXH], dPost[HC], dM[HC][HC], dY[MAXH], dShift[HC];
    int H; bool shifted;
};

template <typename T>
static void ref_token(const RefIn<T>& in, const Params& p,
                      T* dx, T* dR, T* dpost, T* dcomb, T* dfn, T* dscales, T* dbases,
                      T* dgamma, T* dshift) {
    const int H = in.H, D = HC * H;
    T A[HC][MAXH];
    for (int j = 0; j < HC; ++j)
        for (int h = 0; h < H; ++h) {
            T v = in.x[h] * in.post[j];
            for (int i = 0; i < HC; ++i) v += in.comb[j][i] * in.R[i][h];
            A[j][h] = v;
        }
    T q = 0;
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) q += A[j][h] * A[j][h];
    T gq = (T)(1.0 / std::sqrt((double)(q / (T)D + (T)p.hc_norm_eps)));
    T u0[NOUT], u[NOUT];
    for (int k = 0; k < NOUT; ++k) {
        T acc = 0;
        for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) acc += A[j][h] * in.fn[k][j * H + h];
        u0[k] = acc;
        int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
        u[k] = acc * gq * in.scales[seg] + in.bases[k];
    }
    T pre[HC], logits[HC][HC];
    for (int j = 0; j < HC; ++j) pre[j] = sigmoid(u[j]) + (T)p.hc_pre_eps;
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) logits[j][i] = u[2 * HC + j * HC + i];
    T M[HC][HC], seq[MAXNS][HC][HC]; int dims[MAXNS], ns = 0;
    sinkhorn_forward<T>(logits, p, M, seq, dims, ns);
    T mix[HC];
    for (int j = 0; j < HC; ++j) mix[j] = in.shifted ? in.spm[j] : pre[j];
    T n[MAXH];
    for (int h = 0; h < H; ++h) { T acc = 0; for (int j = 0; j < HC; ++j) acc += mix[j] * A[j][h]; n[h] = acc; }
    T m = 0; for (int h = 0; h < H; ++h) m += n[h] * n[h]; m /= (T)H;
    T rg = (T)(1.0 / std::sqrt((double)(m + (T)p.rmsnorm_eps)));
    // ---- 反向 ----
    T dn[MAXH], dLdrg = 0;
    for (int h = 0; h < H; ++h) {
        dn[h] = in.dY[h] * rg * in.gamma[h];
        dgamma[h] = in.dY[h] * n[h] * rg * (T)p.rmsnorm_scale;
        dLdrg += in.dY[h] * n[h] * in.gamma[h];
    }
    T dm = dLdrg * (T)(-0.5) * rg * rg * rg;
    for (int h = 0; h < H; ++h) dn[h] += dm * (T)(2.0 / H) * n[h];
    T dmix[HC], dA[HC][MAXH];
    for (int j = 0; j < HC; ++j) { dmix[j] = 0; for (int h = 0; h < H; ++h) dmix[j] += dn[h] * A[j][h]; }
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dA[j][h] = mix[j] * dn[h] + in.dS[j][h];
    T du[NOUT];
    for (int k = 0; k < NOUT; ++k) du[k] = 0;
    for (int j = 0; j < HC; ++j) {
        T dpre_in = in.shifted ? in.dShift[j] : dmix[j];
        T s = sigmoid(u[j]);
        du[j] = dpre_in * s * (1 - s);
    }
    for (int j = 0; j < HC; ++j) {
        T s = sigmoid(u[HC + j]);
        du[HC + j] += in.dPost[j] * (T)p.hc_post_scale * s * (1 - s);
    }
    {
        T dlogits[HC][HC];
        sinkhorn_backward<T>(in.dM, seq, dims, ns, p, dlogits);
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) du[2 * HC + j * HC + i] += dlogits[j][i];
    }
    T du0[NOUT], dgq = 0, dsc[3] = {0,0,0};
    for (int k = 0; k < NOUT; ++k) {
        int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
        du0[k] = du[k] * gq * in.scales[seg];
        dgq += du[k] * u0[k] * in.scales[seg];
        dsc[seg] += du[k] * u0[k] * gq;
        dbases[k] = du[k];
    }
    T dq = dgq * (T)(-0.5) * gq * gq * gq / (T)D;
    T dz[HC][MAXH];
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dz[j][h] = dq * 2 * A[j][h];
    for (int k = 0; k < NOUT; ++k) for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h)
        dz[j][h] += du0[k] * in.fn[k][j * H + h];
    for (int k = 0; k < NOUT; ++k) for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h)
        dfn[k * D + j * H + h] = du0[k] * A[j][h];
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dA[j][h] += dz[j][h];
    for (int h = 0; h < H; ++h) { dx[h] = 0; for (int j = 0; j < HC; ++j) dx[h] += dA[j][h] * in.post[j]; }
    for (int j = 0; j < HC; ++j) { dpost[j] = 0; for (int h = 0; h < H; ++h) dpost[j] += dA[j][h] * in.x[h]; }
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) { T acc = 0; for (int h = 0; h < H; ++h) acc += dA[j][h] * in.R[i][h]; dcomb[j * HC + i] = acc; }
    for (int i = 0; i < HC; ++i) for (int h = 0; h < H; ++h) { T acc = 0; for (int j = 0; j < HC; ++j) acc += dA[j][h] * in.comb[j][i]; dR[i * MAXH + h] = acc; }
    for (int s = 0; s < 3; ++s) dscales[s] = dsc[s];
    for (int j = 0; j < HC; ++j) dshift[j] = dmix[j];
}

// ===========================================================================
// Device：分阶段 kernel（SoA 扁平布局；fn 为全 batch 共享 [NOUT, D]）。
// ===========================================================================
// 一个 block 处理一个 token，线程 = h（跨 H 连续 → 访存合并）。
__global__ void k_fwd_mix(const float* __restrict__ x, const float* __restrict__ R,
                          const float* __restrict__ post, const float* __restrict__ comb,
                          float* __restrict__ A, int T, int H) {
    int t = blockIdx.x, h = threadIdx.x;
    if (h >= H) return;
    const int D = HC * H;
    const float* xt = x + (size_t)t * H;
    const float* Rt = R + (size_t)t * HC * H;
    const float* pt = post + t * HC;
    const float* ct = comb + t * HC * HC;
    float* At = A + (size_t)t * D;
    for (int j = 0; j < HC; ++j) {
        float v = xt[h] * pt[j];
        for (int i = 0; i < HC; ++i) v += ct[j * HC + i] * Rt[i * H + h];
        At[j * H + h] = v;
    }
}

__device__ inline float block_sum(float v, float* sred, int H) {
    int h = threadIdx.x;
    sred[h] = (h < H) ? v : 0.0f;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (h < s) sred[h] += sred[h + s];
        __syncthreads();
    }
    float r = sred[0];
    __syncthreads();
    return r;
}

__global__ void k_u0(const float* __restrict__ A, const float* __restrict__ fn,
                     float* __restrict__ u0, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // t*NOUT + k
    if (idx >= T * NOUT) return;
    int t = idx / NOUT, k = idx % NOUT;
    const float* At = A + (size_t)t * D;
    const float* fk = fn + (size_t)k * D;
    float acc = 0;
    for (int d = 0; d < D; ++d) acc += At[d] * fk[d];
    u0[idx] = acc;
}

// K1：一个 block 处理一个 token，线程 = h。跨线程归约用 shared。
__global__ void k_coeff(const float* __restrict__ A, const float* __restrict__ u0,
                        const float* __restrict__ gamma, const float* __restrict__ bases,
                        const float* __restrict__ dS, const float* __restrict__ dPost,
                        const float* __restrict__ dM, const float* __restrict__ dY,
                        const float* __restrict__ spm, const float* __restrict__ dShift,
                        float* __restrict__ du0, float* __restrict__ dAbase,
                        float* __restrict__ dgamma_p, float* __restrict__ dscales_p,
                        float* __restrict__ dbases_p, float* __restrict__ dshift,
                        int T, int H, bool shifted, int skip_sinkhorn, Params p) {
    int t = blockIdx.x, h = threadIdx.x;
    const int D = HC * H;
    const float* At = A + (size_t)t * D;
    const float* dSt = dS + (size_t)t * D;
    const float* dYt = dY + (size_t)t * H;
    const float* u0t = u0 + (size_t)t * NOUT;
    float* du0t = du0 + (size_t)t * NOUT;
    float* dAbt = dAbase + (size_t)t * D;

    __shared__ float sA[HC][MAXH];
    __shared__ float sdn[MAXH];
    __shared__ float su[NOUT], sdu[NOUT], sdu0[NOUT], ssc[NOUT], smix[HC], sdmix[HC];
    __shared__ float sred[128];
    __shared__ float sgq, sdq, srg;

    float dSreg[HC];
    if (h < H) {
        for (int j = 0; j < HC; ++j) { sA[j][h] = At[j * H + h]; dSreg[j] = dSt[j * H + h]; }
    }
    __syncthreads();

    // q
    float pq = 0;
    if (h < H) for (int j = 0; j < HC; ++j) pq += sA[j][h] * sA[j][h];
    float q = block_sum(pq, sred, H);
    if (h == 0) sgq = 1.0f / sqrtf(q / (float)D + (float)p.hc_norm_eps);
    __syncthreads();
    float gq = sgq;

    // 系数：u、sigmoid、Sinkhorn（thread 0 完成 24 维小计算）
    if (h == 0) {
        for (int k = 0; k < NOUT; ++k) {
            int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
            ssc[k] = (float)p.scales[seg];
            su[k] = u0t[k] * gq * ssc[k] + bases[k];
            sdu[k] = 0.0f;
        }
        float pre[HC], logits[HC][HC];
        for (int j = 0; j < HC; ++j) {
            pre[j] = sigmoid(su[j]) + (float)p.hc_pre_eps;
            smix[j] = shifted ? spm[t * HC + j] : pre[j];
            sdu[HC + j] = dPost[t * HC + j] * (float)p.hc_post_scale
                          * sigmoid(su[HC + j]) * (1.0f - sigmoid(su[HC + j]));
        }
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) logits[j][i] = su[2 * HC + j * HC + i];
        float M[HC][HC], seq[MAXNS][HC][HC]; int dims[MAXNS], ns = 0;
        float dMin[HC][HC], dlogits[HC][HC];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) dMin[j][i] = dM[(t * HC + j) * HC + i];
        if (skip_sinkhorn) {              // 仅用于计时归因：把 Sinkhorn 换成逐元素恒等
            for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) dlogits[j][i] = dMin[j][i];
        } else {
            sinkhorn_forward<float>(logits, p, M, seq, dims, ns);
            sinkhorn_backward<float>(dMin, seq, dims, ns, p, dlogits);
        }
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) sdu[2 * HC + j * HC + i] += dlogits[j][i];
    }
    __syncthreads();

    // RMSNorm 反向：n、m、dmix（跨 h 归约）
    float nh = 0;
    if (h < H) for (int j = 0; j < HC; ++j) nh += smix[j] * sA[j][h];
    float m = block_sum((h < H) ? nh * nh : 0.0f, sred, H) / (float)H;
    if (h == 0) srg = 1.0f / sqrtf(m + (float)p.rmsnorm_eps);
    __syncthreads();
    float rg = srg;
    float dnh = 0, dLdrg = 0;
    if (h < H) {
        dnh = dYt[h] * rg * gamma[h];
        dgamma_p[(size_t)t * H + h] = dYt[h] * nh * rg * (float)p.rmsnorm_scale;
        dLdrg = dYt[h] * nh * gamma[h];
    }
    float dm = block_sum(dLdrg, sred, H) * (-0.5f) * rg * rg * rg;
    if (h < H) { sdn[h] = dnh + dm * (2.0f / H) * nh; }
    __syncthreads();
    for (int j = 0; j < HC; ++j) {
        float pj = (h < H) ? sdn[h] * sA[j][h] : 0.0f;
        sdmix[j] = block_sum(pj, sred, H);
    }

    // 仿射反向（thread 0 完成）
    if (h == 0) {
        for (int j = 0; j < HC; ++j) {
            float dpre_in = shifted ? dShift[t * HC + j] : sdmix[j];
            sdu[j] = dpre_in * sigmoid(su[j]) * (1.0f - sigmoid(su[j]));
        }
        float dgq = 0, dsc[3] = {0, 0, 0};
        for (int k = 0; k < NOUT; ++k) {
            int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
            sdu0[k] = sdu[k] * gq * ssc[k];
            dgq += sdu[k] * u0t[k] * ssc[k];
            dsc[seg] += sdu[k] * u0t[k] * gq;
            dbases_p[(size_t)t * NOUT + k] = sdu[k];
        }
        for (int s = 0; s < 3; ++s) dscales_p[(size_t)t * 3 + s] = dsc[s];
        sdq = dgq * (-0.5f) * gq * gq * gq / (float)D;
        if (shifted) for (int j = 0; j < HC; ++j) dshift[t * HC + j] = sdmix[j];
    }
    __syncthreads();

    if (h < H) {
        for (int j = 0; j < HC; ++j)
            dAbt[j * H + h] = smix[j] * sdn[h] + dSreg[j] + sdq * 2.0f * sA[j][h];
    }
    if (h < NOUT) du0t[h] = sdu0[h];
}

// 每个 block 负责一段 token，线程 o 负责一个标量输出（H + NOUT + 3 个），块内局部和后 atomicAdd。
__global__ void k_reduce(const float* __restrict__ dgamma_p, const float* __restrict__ dbases_p,
                         const float* __restrict__ dscales_p, float* __restrict__ dgamma,
                         float* __restrict__ dbases, float* __restrict__ dscales, int T, int H) {
    int o = threadIdx.x;
    int ntot = H + NOUT + 3;
    if (o >= ntot) return;
    int per = (T + gridDim.x - 1) / gridDim.x;
    int t0 = blockIdx.x * per, t1 = t0 + per < T ? t0 + per : T;
    float a = 0;
    for (int t = t0; t < t1; ++t) {
        if (o < H) a += dgamma_p[(size_t)t * H + o];
        else if (o < H + NOUT) a += dbases_p[(size_t)t * NOUT + (o - H)];
        else a += dscales_p[(size_t)t * 3 + (o - H - NOUT)];
    }
    if (o < H) atomicAdd(&dgamma[o], a);
    else if (o < H + NOUT) atomicAdd(&dbases[o - H], a);
    else atomicAdd(&dscales[o - H - NOUT], a);
}

__global__ void k_dz(const float* __restrict__ du0, const float* __restrict__ fn,
                     float* __restrict__ dz, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // t*D + d
    if (idx >= T * D) return;
    int t = idx / D, d = idx % D;
    const float* du0t = du0 + (size_t)t * NOUT;
    float acc = 0;
    for (int k = 0; k < NOUT; ++k) acc += du0t[k] * fn[(size_t)k * D + d];
    dz[idx] = acc;
}

__global__ void k_dfn(const float* __restrict__ du0, const float* __restrict__ A,
                      float* __restrict__ dfn, int T, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // k*D + d
    if (idx >= NOUT * D) return;
    int k = idx / D, d = idx % D;
    float acc = 0;
    for (int t = 0; t < T; ++t) acc += du0[(size_t)t * NOUT + k] * A[(size_t)t * D + d];
    dfn[idx] = acc;
}

__global__ void k_mix_bwd(const float* __restrict__ dAbase, const float* __restrict__ dz,
                          const float* __restrict__ x, const float* __restrict__ R,
                          const float* __restrict__ post, const float* __restrict__ comb,
                          float* __restrict__ dx, float* __restrict__ dR,
                          float* __restrict__ dpost, float* __restrict__ dcomb, int T, int H) {
    int t = blockIdx.x, h = threadIdx.x;
    const int D = HC * H;
    const float* xt = x + (size_t)t * H;
    const float* Rt = R + (size_t)t * HC * H;
    const float* pt = post + t * HC;
    const float* ct = comb + t * HC * HC;
    const float* dAbt = dAbase + (size_t)t * D;
    const float* dzt = dz + (size_t)t * D;
    __shared__ float sred[128];
    __shared__ float sA[HC][MAXH], sdpost[HC], sdcomb[HC * HC];
    if (h < H) for (int j = 0; j < HC; ++j) sA[j][h] = dAbt[j * H + h] + dzt[j * H + h];
    __syncthreads();
    if (h < H) {
        float dacc = 0;
        for (int j = 0; j < HC; ++j) dacc += sA[j][h] * pt[j];
        dx[(size_t)t * H + h] = dacc;
    }
    for (int j = 0; j < HC; ++j) {
        float pj = (h < H) ? sA[j][h] * xt[h] : 0.0f;
        sdpost[j] = block_sum(pj, sred, H);
    }
    for (int j = 0; j < HC; ++j)
        for (int i = 0; i < HC; ++i) {
            float p = (h < H) ? sA[j][h] * Rt[i * H + h] : 0.0f;
            sdcomb[j * HC + i] = block_sum(p, sred, H);
        }
    if (h < HC) dpost[t * HC + h] = sdpost[h];
    if (h < HC * HC) dcomb[t * HC * HC + h] = sdcomb[h];
    // dR[i][h] = Σ_j dA[j][h]*comb[j][i]（每 h 局部，无需归约）
    if (h < H) {
        for (int i = 0; i < HC; ++i) {
            float acc = 0;
            for (int j = 0; j < HC; ++j) acc += sA[j][h] * ct[j * HC + i];
            dR[((size_t)t * HC + i) * H + h] = acc;
        }
    }
}

// ===========================================================================
// P122-1：**per-token 融合 worker**——把 K1（系数反向）+ G1a（dz=dgrad）+ K2（混合反向）
// 融进一个 block。相比第 121 篇的分段路径，**省掉 `dAbase[T,D]` 与 `dz[T,D]` 两张
// 全局中间张量的往返**（写 + 再读），也就是前向 epilogue 的 per-token worker
// （`epilogue/sm100_mega_mhc.cuh:141` 的 `run_normal_worker`）在反向侧的对偶：
// 一个 token 的元素级反向全在片上完成，只有必须跨 token 归约的 `dfn`（G1b wgrad）
// 与必须喂 GEMM 的 `du0[A]` 才落全局。G0/G1b 仍是真实 GEMM，不在这里融合。
// ===========================================================================
__global__ void k_fused_k1k2(const float* __restrict__ A, const float* __restrict__ u0,
                             const float* __restrict__ gamma, const float* __restrict__ bases,
                             const float* __restrict__ dS, const float* __restrict__ dPost,
                             const float* __restrict__ dM, const float* __restrict__ dY,
                             const float* __restrict__ spm, const float* __restrict__ dShift,
                             const float* __restrict__ fn, const float* __restrict__ x,
                             const float* __restrict__ R, const float* __restrict__ post,
                             const float* __restrict__ comb,
                             float* __restrict__ du0, float* __restrict__ dx, float* __restrict__ dR,
                             float* __restrict__ dpost, float* __restrict__ dcomb,
                             float* __restrict__ dgamma_p, float* __restrict__ dscales_p,
                             float* __restrict__ dbases_p, float* __restrict__ dshift,
                             int T, int H, bool shifted, Params p) {
    int t = blockIdx.x, h = threadIdx.x;
    const int D = HC * H;
    const float* At = A + (size_t)t * D;
    const float* dSt = dS + (size_t)t * D;
    const float* dYt = dY + (size_t)t * H;
    const float* u0t = u0 + (size_t)t * NOUT;
    const float* xt = x + (size_t)t * H;
    const float* Rt = R + (size_t)t * HC * H;
    const float* pt = post + t * HC;
    const float* ct = comb + t * HC * HC;
    float* du0t = du0 + (size_t)t * NOUT;

    __shared__ float sA[HC][MAXH];
    __shared__ float sdn[MAXH];
    __shared__ float sdA[HC][MAXH];
    __shared__ float sx[MAXH];
    __shared__ float sR[HC][MAXH];
    __shared__ float spost[HC], sdpost[HC], sdcomb[HC * HC];
    __shared__ float su[NOUT], sdu[NOUT], sdu0[NOUT], ssc[NOUT], smix[HC], sdmix[HC];
    __shared__ float sred[128];
    __shared__ float sgq, sdq, srg;

    float dSreg[HC];
    if (h < H) {
        for (int j = 0; j < HC; ++j) { sA[j][h] = At[j * H + h]; dSreg[j] = dSt[j * H + h]; }
        sx[h] = xt[h];
        for (int i = 0; i < HC; ++i) sR[i][h] = Rt[i * H + h];
    }
    if (h < HC) spost[h] = pt[h];
    __syncthreads();

    // q
    float pq = 0;
    if (h < H) for (int j = 0; j < HC; ++j) pq += sA[j][h] * sA[j][h];
    float q = block_sum(pq, sred, H);
    if (h == 0) sgq = 1.0f / sqrtf(q / (float)D + (float)p.hc_norm_eps);
    __syncthreads();
    float gq = sgq;

    // 系数 + sigmoid + Sinkhorn（thread 0 完成 24 维小计算）
    if (h == 0) {
        for (int k = 0; k < NOUT; ++k) {
            int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
            ssc[k] = (float)p.scales[seg];
            su[k] = u0t[k] * gq * ssc[k] + bases[k];
            sdu[k] = 0.0f;
        }
        float pre[HC], logits[HC][HC];
        for (int j = 0; j < HC; ++j) {
            pre[j] = sigmoid(su[j]) + (float)p.hc_pre_eps;
            smix[j] = shifted ? spm[t * HC + j] : pre[j];
            sdu[HC + j] = dPost[t * HC + j] * (float)p.hc_post_scale
                          * sigmoid(su[HC + j]) * (1.0f - sigmoid(su[HC + j]));
        }
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) logits[j][i] = su[2 * HC + j * HC + i];
        float M[HC][HC], seq[MAXNS][HC][HC]; int dims[MAXNS], ns = 0;
        float dMin[HC][HC], dlogits[HC][HC];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) dMin[j][i] = dM[(t * HC + j) * HC + i];
        sinkhorn_forward<float>(logits, p, M, seq, dims, ns);
        sinkhorn_backward<float>(dMin, seq, dims, ns, p, dlogits);
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) sdu[2 * HC + j * HC + i] += dlogits[j][i];
    }
    __syncthreads();

    // RMSNorm 反向
    float nh = 0;
    if (h < H) for (int j = 0; j < HC; ++j) nh += smix[j] * sA[j][h];
    float m = block_sum((h < H) ? nh * nh : 0.0f, sred, H) / (float)H;
    if (h == 0) srg = 1.0f / sqrtf(m + (float)p.rmsnorm_eps);
    __syncthreads();
    float rg = srg;
    float dnh = 0, dLdrg = 0;
    if (h < H) {
        dnh = dYt[h] * rg * gamma[h];
        dgamma_p[(size_t)t * H + h] = dYt[h] * nh * rg * (float)p.rmsnorm_scale;
        dLdrg = dYt[h] * nh * gamma[h];
    }
    float dm = block_sum(dLdrg, sred, H) * (-0.5f) * rg * rg * rg;
    if (h < H) { sdn[h] = dnh + dm * (2.0f / H) * nh; }
    __syncthreads();
    for (int j = 0; j < HC; ++j) {
        float pj = (h < H) ? sdn[h] * sA[j][h] : 0.0f;
        sdmix[j] = block_sum(pj, sred, H);
    }

    // 仿射反向（thread 0 完成），得到 du0
    if (h == 0) {
        for (int j = 0; j < HC; ++j) {
            float dpre_in = shifted ? dShift[t * HC + j] : sdmix[j];
            sdu[j] = dpre_in * sigmoid(su[j]) * (1.0f - sigmoid(su[j]));
        }
        float dgq = 0, dsc[3] = {0, 0, 0};
        for (int k = 0; k < NOUT; ++k) {
            int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
            sdu0[k] = sdu[k] * gq * ssc[k];
            dgq += sdu[k] * u0t[k] * ssc[k];
            dsc[seg] += sdu[k] * u0t[k] * gq;
            dbases_p[(size_t)t * NOUT + k] = sdu[k];
        }
        for (int s = 0; s < 3; ++s) dscales_p[(size_t)t * 3 + s] = dsc[s];
        sdq = dgq * (-0.5f) * gq * gq * gq / (float)D;
        if (shifted) for (int j = 0; j < HC; ++j) dshift[t * HC + j] = sdmix[j];
        for (int k = 0; k < NOUT; ++k) du0t[k] = sdu0[k];
    }
    __syncthreads();

    // 融合点：dA = dAbase + dz，其中 dz[j*H+h] = Σ_k du0[k]*fn[k*D + j*H+h]（片上算，不落全局）
    if (h < H) {
        for (int j = 0; j < HC; ++j) {
            float dAb = smix[j] * sdn[h] + dSreg[j] + sdq * 2.0f * sA[j][h];
            float dzz = 0.0f;
            for (int k = 0; k < NOUT; ++k) dzz += sdu0[k] * fn[(size_t)k * D + j * H + h];
            sdA[j][h] = dAb + dzz;
        }
    }
    __syncthreads();

    // K2：混合反向（dpost/dcomb 需跨 h 归约；dR 每 h 局部，comb 从全局读回原始值）
    if (h < H) {
        float dacc = 0;
        for (int j = 0; j < HC; ++j) dacc += sdA[j][h] * spost[j];
        dx[(size_t)t * H + h] = dacc;
    }
    for (int j = 0; j < HC; ++j) {
        float pj = (h < H) ? sdA[j][h] * sx[h] : 0.0f;
        float v = block_sum(pj, sred, H);
        if (h == 0) sdpost[j] = v;
    }
    __syncthreads();
    if (h < HC) dpost[t * HC + h] = sdpost[h];
    __syncthreads();
    for (int j = 0; j < HC; ++j)
        for (int i = 0; i < HC; ++i) {
            float pr = (h < H) ? sdA[j][h] * sR[i][h] : 0.0f;
            float v = block_sum(pr, sred, H);
            if (h == 0) sdcomb[j * HC + i] = v;
        }
    __syncthreads();
    if (h < HC * HC) dcomb[t * HC * HC + h] = sdcomb[h];
    if (h < H) {
        for (int i = 0; i < HC; ++i) {
            float acc = 0;
            for (int j = 0; j < HC; ++j) acc += sdA[j][h] * ct[j * HC + i];
            dR[((size_t)t * HC + i) * H + h] = acc;
        }
    }
}

// ===========================================================================
// 主机侧：生成数据、跑参考、跑 device、比较、计时
// ===========================================================================
struct HostData {
    int T, H;
    std::vector<float> x, R, post, comb, fn, gamma, bases;
    std::vector<float> dS, dPost, dM, dY, spm, dShift;
};

static HostData gen(int T, int H, int seed, bool shifted, const Params& p) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> ur(-1.0, 1.0), upos(0.0, 1.0);
    HostData d; d.T = T; d.H = H;
    d.x.resize((size_t)T * H);
    d.R.resize((size_t)T * HC * H);
    d.post.resize((size_t)T * HC);
    d.comb.resize((size_t)T * HC * HC);
    d.fn.resize((size_t)NOUT * HC * H);
    d.gamma.resize(H);
    d.bases.resize(NOUT);
    d.dS.resize((size_t)T * HC * H);
    d.dPost.resize((size_t)T * HC);
    d.dM.resize((size_t)T * HC * HC);
    d.dY.resize((size_t)T * H);
    d.spm.resize((size_t)T * HC);
    d.dShift.resize((size_t)T * HC);
    for (auto& v : d.x) v = (float)ur(rng);
    for (auto& v : d.R) v = (float)ur(rng);
    for (auto& v : d.post) v = (float)upos(rng);
    for (auto& v : d.comb) v = (float)(0.3 * ur(rng));
    for (auto& v : d.fn) v = (float)(0.2 * ur(rng));
    for (auto& v : d.gamma) v = (float)ur(rng);
    for (auto& v : d.bases) v = (float)(0.1 * ur(rng));
    for (auto& v : d.dS) v = (float)(0.1 * ur(rng));
    for (auto& v : d.dPost) v = (float)(0.1 * ur(rng));
    for (auto& v : d.dM) v = (float)(0.1 * ur(rng));
    for (auto& v : d.dY) v = (float)(0.1 * ur(rng));
    for (auto& v : d.spm) v = (float)upos(rng);
    for (auto& v : d.dShift) v = (float)(0.1 * ur(rng));
    (void)shifted; (void)p;
    return d;
}

static double relerr(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    return std::sqrt(num / (den + 1e-300));
}

// host double 参考：按 token 循环 ref_token，聚合 dfn/dscales/dbases/dgamma。
struct RefOut {
    std::vector<double> dx, dR, dpost, dcomb, dfn, dscales, dbases, dgamma, dshift;
};

static RefOut host_ref(const HostData& d, const Params& p, bool shifted) {
    int T = d.T, H = d.H, D = HC * H;
    RefOut o;
    o.dx.assign((size_t)T * H, 0);
    o.dR.assign((size_t)T * HC * H, 0);
    o.dpost.assign((size_t)T * HC, 0);
    o.dcomb.assign((size_t)T * HC * HC, 0);
    o.dfn.assign((size_t)NOUT * D, 0);
    o.dscales.assign(3, 0);
    o.dbases.assign(NOUT, 0);
    o.dgamma.assign(H, 0);
    o.dshift.assign((size_t)T * HC, 0);
    for (int t = 0; t < T; ++t) {
        RefIn<double> in; std::memset(&in, 0, sizeof(in));
        in.H = H; in.shifted = shifted;
        for (int h = 0; h < H; ++h) in.x[h] = d.x[(size_t)t * H + h];
        for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) in.R[j][h] = d.R[((size_t)t * HC + j) * H + h];
        for (int j = 0; j < HC; ++j) in.post[j] = d.post[(size_t)t * HC + j];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) in.comb[j][i] = d.comb[((size_t)t * HC + j) * HC + i];
        for (int k = 0; k < NOUT; ++k) for (int e = 0; e < D; ++e) in.fn[k][e] = d.fn[(size_t)k * D + e];
        for (int s = 0; s < 3; ++s) in.scales[s] = p.scales[s];
        for (int k = 0; k < NOUT; ++k) in.bases[k] = d.bases[k];
        for (int h = 0; h < H; ++h) in.gamma[h] = d.gamma[h];
        for (int j = 0; j < HC; ++j) in.spm[j] = d.spm[(size_t)t * HC + j];
        for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) in.dS[j][h] = d.dS[((size_t)t * HC + j) * H + h];
        for (int j = 0; j < HC; ++j) in.dPost[j] = d.dPost[(size_t)t * HC + j];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) in.dM[j][i] = d.dM[((size_t)t * HC + j) * HC + i];
        for (int h = 0; h < H; ++h) in.dY[h] = d.dY[(size_t)t * H + h];
        for (int j = 0; j < HC; ++j) in.dShift[j] = d.dShift[(size_t)t * HC + j];
        std::vector<double> rx(H), rR(HC * MAXH, 0), rp(HC), rc(HC * HC), rf((size_t)NOUT * D, 0),
            rsc(3), rb(NOUT), rg(H), rsh(HC);
        ref_token<double>(in, p, rx.data(), rR.data(), rp.data(), rc.data(), rf.data(),
                          rsc.data(), rb.data(), rg.data(), rsh.data());
        for (int h = 0; h < H; ++h) o.dx[(size_t)t * H + h] = rx[h];
        for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) o.dR[((size_t)t * HC + j) * H + h] = rR[j * MAXH + h];
        for (int j = 0; j < HC; ++j) o.dpost[(size_t)t * HC + j] = rp[j];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) o.dcomb[((size_t)t * HC + j) * HC + i] = rc[j * HC + i];
        for (size_t e = 0; e < (size_t)NOUT * D; ++e) o.dfn[e] += rf[e];
        for (int s = 0; s < 3; ++s) o.dscales[s] += rsc[s];
        for (int k = 0; k < NOUT; ++k) o.dbases[k] += rb[k];
        for (int h = 0; h < H; ++h) o.dgamma[h] += rg[h];
        for (int j = 0; j < HC; ++j) o.dshift[(size_t)t * HC + j] = rsh[j];
    }
    return o;
}

// 把 device 输出聚合成与 RefOut 对应的 double 向量（dscales/dbases/dgamma 由 kernel 原子累加）。
struct DevPtrs {
    float *x, *R, *post, *comb, *fn, *gamma, *bases, *dS, *dPost, *dM, *dY, *spm, *dShift;
    float *A, *u0, *du0, *dAbase, *dz, *dfn, *dx, *dR, *dpost, *dcomb, *dgamma, *dscales, *dbases, *dshift;
    float *dgamma_p, *dscales_p, *dbases_p;
};
struct DevBuf { std::vector<void*> all; };

static DevPtrs dev_alloc(const HostData& d, DevBuf& buf) {
    int T = d.T, H = d.H, D = HC * H;
    auto alloc = [&](size_t n, const std::vector<float>& src) {
        float* p = nullptr; CK(cudaMalloc(&p, n * sizeof(float)));
        if (!src.empty()) CK(cudaMemcpy(p, src.data(), n * sizeof(float), cudaMemcpyHostToDevice));
        else CK(cudaMemset(p, 0, n * sizeof(float)));
        buf.all.push_back(p); return p;
    };
    DevPtrs P;
    P.x = alloc((size_t)T * H, d.x);
    P.R = alloc((size_t)T * HC * H, d.R);
    P.post = alloc((size_t)T * HC, d.post);
    P.comb = alloc((size_t)T * HC * HC, d.comb);
    P.fn = alloc((size_t)NOUT * D, d.fn);
    P.gamma = alloc(H, d.gamma);
    P.bases = alloc(NOUT, d.bases);
    P.dS = alloc((size_t)T * HC * H, d.dS);
    P.dPost = alloc((size_t)T * HC, d.dPost);
    P.dM = alloc((size_t)T * HC * HC, d.dM);
    P.dY = alloc((size_t)T * H, d.dY);
    P.spm = alloc((size_t)T * HC, d.spm);
    P.dShift = alloc((size_t)T * HC, d.dShift);
    P.A = alloc((size_t)T * D, {});
    P.u0 = alloc((size_t)T * NOUT, {});
    P.du0 = alloc((size_t)T * NOUT, {});
    P.dAbase = alloc((size_t)T * D, {});
    P.dz = alloc((size_t)T * D, {});
    P.dfn = alloc((size_t)NOUT * D, {});
    P.dx = alloc((size_t)T * H, {});
    P.dR = alloc((size_t)T * HC * H, {});
    P.dpost = alloc((size_t)T * HC, {});
    P.dcomb = alloc((size_t)T * HC * HC, {});
    P.dgamma = alloc(H, {});
    P.dscales = alloc(3, {});
    P.dbases = alloc(NOUT, {});
    P.dshift = alloc((size_t)T * HC, {});
    P.dgamma_p = alloc((size_t)T * H, {});
    P.dscales_p = alloc((size_t)T * 3, {});
    P.dbases_p = alloc((size_t)T * NOUT, {});
    return P;
}

static void launch_all(const DevPtrs& P, const HostData& d, const Params& p, bool shifted, int H) {
    int T = d.T, D = HC * H;
    k_fwd_mix<<<T, 64>>>(P.x, P.R, P.post, P.comb, P.A, T, H);
    { int n = T * NOUT; k_u0<<<(n + 127) / 128, 128>>>(P.A, P.fn, P.u0, T, D); }
    k_coeff<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM, P.dY, P.spm, P.dShift,
                       P.du0, P.dAbase, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift, T, H, shifted, 0, p);
    CK(cudaMemset(P.dgamma, 0, H * sizeof(float)));
    CK(cudaMemset(P.dscales, 0, 3 * sizeof(float)));
    CK(cudaMemset(P.dbases, 0, NOUT * sizeof(float)));
    k_reduce<<<256, 128>>>(P.dgamma_p, P.dbases_p, P.dscales_p, P.dgamma, P.dbases, P.dscales, T, H);
    { int n = T * D; k_dz<<<(n + 127) / 128, 128>>>(P.du0, P.fn, P.dz, T, D); }
    { int n = NOUT * D; k_dfn<<<(n + 127) / 128, 128>>>(P.du0, P.A, P.dfn, T, D); }
    k_mix_bwd<<<T, 64>>>(P.dAbase, P.dz, P.x, P.R, P.post, P.comb, P.dx, P.dR, P.dpost, P.dcomb, T, H);
}

static std::vector<float> d2h(const float* p, size_t n) {
    std::vector<float> v(n); CK(cudaMemcpy(v.data(), p, n * sizeof(float), cudaMemcpyDeviceToHost)); return v;
}
static std::vector<double> as_d(const std::vector<float>& v) {
    std::vector<double> o(v.size()); for (size_t i = 0; i < v.size(); ++i) o[i] = (double)v[i]; return o;
}

static void check_case(int T, int H, int iters, bool shifted, int seed) {
    Params p; p.hc_norm_eps = 1e-5; p.hc_pre_eps = 1e-6; p.hc_post_scale = 1.0;
    p.rmsnorm_eps = 1e-6; p.rmsnorm_scale = 1.0; p.sinkhorn_eps = 1e-8; p.iters = iters;
    p.scales[0] = 0.7; p.scales[1] = 0.5; p.scales[2] = 0.9;
    int D = HC * H;
    HostData d = gen(T, H, seed, shifted, p);
    RefOut R = host_ref(d, p, shifted);
    DevBuf buf; DevPtrs P = dev_alloc(d, buf);
    launch_all(P, d, p, shifted, H);
    CK(cudaDeviceSynchronize());
    double mx = 0; const char* worst = "";
    auto cmp = [&](const char* nm, const float* dp, size_t n, const std::vector<double>& ref) {
        double e = relerr(as_d(d2h(dp, n)), ref); if (e > mx) { mx = e; worst = nm; } };
    cmp("dx", P.dx, (size_t)T * H, R.dx);
    cmp("dR", P.dR, (size_t)T * HC * H, R.dR);
    cmp("dpost", P.dpost, (size_t)T * HC, R.dpost);
    cmp("dcomb", P.dcomb, (size_t)T * HC * HC, R.dcomb);
    cmp("dfn", P.dfn, (size_t)NOUT * D, R.dfn);
    cmp("dscales", P.dscales, 3, R.dscales);
    cmp("dbases", P.dbases, NOUT, R.dbases);
    cmp("dgamma", P.dgamma, H, R.dgamma);
    if (shifted) cmp("dshift", P.dshift, (size_t)T * HC, R.dshift);
    printf("  T=%4d H=%2d it=%d %-7s  worst relerr(fp32 device vs double) = %.3e  (%s)\n",
           T, H, iters, shifted ? "shifted" : "normal", mx, worst);
    for (void* q : buf.all) cudaFree(q);
}

static void timing(int T, int H) {
    Params p; p.hc_norm_eps = 1e-5; p.hc_pre_eps = 1e-6; p.hc_post_scale = 1.0;
    p.rmsnorm_eps = 1e-6; p.rmsnorm_scale = 1.0; p.sinkhorn_eps = 1e-8; p.iters = 3;
    p.scales[0] = 0.7; p.scales[1] = 0.5; p.scales[2] = 0.9;
    int D = HC * H;
    HostData d = gen(T, H, 42, false, p);
    DevBuf buf; DevPtrs P = dev_alloc(d, buf);
    int reps = 50;
    auto timeit = [&](const char* nm, double bytes, std::function<void()> fn) {
        for (int i = 0; i < 5; ++i) fn();
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= reps;
        double gbps = bytes / (ms * 1e-3) / 1e9;
        printf("    %-10s %8.3f us   %8.1f GB/s\n", nm, ms * 1e3, gbps);
        cudaEventDestroy(a); cudaEventDestroy(b);
    };
    printf("  [计时] T=%d H=%d D=%d reps=%d（元素级三段的「有效字节」按 读+写 估算）\n", T, H, D, reps);
    double bK0 = (double)((size_t)T * H + (size_t)T * HC * H + (size_t)T * HC * HC + (size_t)T * HC) * 4
               + (double)((size_t)T * D) * 4;
    timeit("K0 fwd_mix", bK0, [&]{ k_fwd_mix<<<T, 64>>>(P.x, P.R, P.post, P.comb, P.A, T, H); });
    double bK1 = (double)((size_t)T * D * 2 + (size_t)T * NOUT * 2 + (size_t)T * HC * H) * 4;
    timeit("K1 coeff", bK1, [&]{ k_coeff<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM,
        P.dY, P.spm, P.dShift, P.du0, P.dAbase, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift, T, H, false, 0, p); });
    timeit("K1 -sinkhorn", bK1, [&]{ k_coeff<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM,
        P.dY, P.spm, P.dShift, P.du0, P.dAbase, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift, T, H, false, 1, p); });
    timeit("K1r reduce", (double)((size_t)T * (H + NOUT + 3)) * 4 * 2, [&]{
        CK(cudaMemset(P.dgamma, 0, H * sizeof(float)));
    CK(cudaMemset(P.dscales, 0, 3 * sizeof(float)));
    CK(cudaMemset(P.dbases, 0, NOUT * sizeof(float)));
    k_reduce<<<256, 128>>>(P.dgamma_p, P.dbases_p, P.dscales_p, P.dgamma, P.dbases, P.dscales, T, H); });
    double bK2 = (double)((size_t)T * D * 2 + (size_t)T * HC * H * 2 + (size_t)T * H) * 4;
    timeit("K2 mix_bwd", bK2, [&]{ k_mix_bwd<<<T, 64>>>(P.dAbase, P.dz, P.x, P.R, P.post, P.comb,
        P.dx, P.dR, P.dpost, P.dcomb, T, H); });
    for (void* q : buf.all) cudaFree(q);
}

static void launch_fused(const DevPtrs& P, const HostData& d, const Params& p, bool shifted, int H) {
    int T = d.T, D = HC * H;
    k_fwd_mix<<<T, 64>>>(P.x, P.R, P.post, P.comb, P.A, T, H);
    { int n = T * NOUT; k_u0<<<(n + 127) / 128, 128>>>(P.A, P.fn, P.u0, T, D); }
    CK(cudaMemset(P.dgamma, 0, H * sizeof(float)));
    CK(cudaMemset(P.dscales, 0, 3 * sizeof(float)));
    CK(cudaMemset(P.dbases, 0, NOUT * sizeof(float)));
    k_fused_k1k2<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM, P.dY, P.spm, P.dShift,
                            P.fn, P.x, P.R, P.post, P.comb, P.du0,
                            P.dx, P.dR, P.dpost, P.dcomb, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift,
                            T, H, shifted, p);
    k_reduce<<<256, 128>>>(P.dgamma_p, P.dbases_p, P.dscales_p, P.dgamma, P.dbases, P.dscales, T, H);
    { int n = NOUT * D; k_dfn<<<(n + 127) / 128, 128>>>(P.du0, P.A, P.dfn, T, D); }
}

static void check_fused_case(int T, int H, int iters, bool shifted, int seed) {
    Params p; p.hc_norm_eps = 1e-5; p.hc_pre_eps = 1e-6; p.hc_post_scale = 1.0;
    p.rmsnorm_eps = 1e-6; p.rmsnorm_scale = 1.0; p.sinkhorn_eps = 1e-8; p.iters = iters;
    p.scales[0] = 0.7; p.scales[1] = 0.5; p.scales[2] = 0.9;
    int D = HC * H;
    HostData d = gen(T, H, seed, shifted, p);
    RefOut R = host_ref(d, p, shifted);
    DevBuf buf; DevPtrs P = dev_alloc(d, buf);
    launch_fused(P, d, p, shifted, H);
    CK(cudaDeviceSynchronize());
    double mx = 0; const char* worst = "";
    auto cmp = [&](const char* nm, const float* dp, size_t n, const std::vector<double>& ref) {
        double e = relerr(as_d(d2h(dp, n)), ref); if (e > mx) { mx = e; worst = nm; } };
    cmp("dx", P.dx, (size_t)T * H, R.dx);
    cmp("dR", P.dR, (size_t)T * HC * H, R.dR);
    cmp("dpost", P.dpost, (size_t)T * HC, R.dpost);
    cmp("dcomb", P.dcomb, (size_t)T * HC * HC, R.dcomb);
    cmp("dfn", P.dfn, (size_t)NOUT * D, R.dfn);
    cmp("dscales", P.dscales, 3, R.dscales);
    cmp("dbases", P.dbases, NOUT, R.dbases);
    cmp("dgamma", P.dgamma, H, R.dgamma);
    if (shifted) cmp("dshift", P.dshift, (size_t)T * HC, R.dshift);
    printf("  T=%4d H=%2d it=%d %-7s  worst relerr(fp32 device vs double) = %.3e  (%s)\n",
           T, H, iters, shifted ? "shifted" : "normal", mx, worst);
    for (void* q : buf.all) cudaFree(q);
}

static void timing_fused(int T, int H) {
    Params p; p.hc_norm_eps = 1e-5; p.hc_pre_eps = 1e-6; p.hc_post_scale = 1.0;
    p.rmsnorm_eps = 1e-6; p.rmsnorm_scale = 1.0; p.sinkhorn_eps = 1e-8; p.iters = 3;
    p.scales[0] = 0.7; p.scales[1] = 0.5; p.scales[2] = 0.9;
    int D = HC * H;
    HostData d = gen(T, H, 42, false, p);
    DevBuf buf; DevPtrs P = dev_alloc(d, buf);
    launch_all(P, d, p, false, H);
    int reps = 50;
    auto timeit = [&](const char* nm, std::function<void()> fn) {
        for (int i = 0; i < 5; ++i) fn();
        CK(cudaDeviceSynchronize());
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) fn();
        cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= reps;
        printf("    %-22s %8.3f us\n", nm, ms * 1e3);
        cudaEventDestroy(a); cudaEventDestroy(b);
    };
    printf("  [计时] T=%d H=%d D=%d reps=%d  元素级 K1+G1a+K2（不含 G0/G1b GEMM）\n", T, H, D, reps);
    timeit("segmented K1+dz+K2", [&]{
        CK(cudaMemset(P.dgamma, 0, H * sizeof(float)));
        CK(cudaMemset(P.dscales, 0, 3 * sizeof(float)));
        CK(cudaMemset(P.dbases, 0, NOUT * sizeof(float)));
        k_coeff<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM, P.dY, P.spm, P.dShift,
                           P.du0, P.dAbase, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift, T, H, false, 0, p);
        { int n = T * D; k_dz<<<(n + 127) / 128, 128>>>(P.du0, P.fn, P.dz, T, D); }
        k_reduce<<<256, 128>>>(P.dgamma_p, P.dbases_p, P.dscales_p, P.dgamma, P.dbases, P.dscales, T, H);
        k_mix_bwd<<<T, 64>>>(P.dAbase, P.dz, P.x, P.R, P.post, P.comb, P.dx, P.dR, P.dpost, P.dcomb, T, H);
    });
    timeit("fused per-token worker", [&]{
        CK(cudaMemset(P.dgamma, 0, H * sizeof(float)));
        CK(cudaMemset(P.dscales, 0, 3 * sizeof(float)));
        CK(cudaMemset(P.dbases, 0, NOUT * sizeof(float)));
        k_fused_k1k2<<<T, 64>>>(P.A, P.u0, P.gamma, P.bases, P.dS, P.dPost, P.dM, P.dY, P.spm, P.dShift,
                                P.fn, P.x, P.R, P.post, P.comb, P.du0,
                                P.dx, P.dR, P.dpost, P.dcomb, P.dgamma_p, P.dscales_p, P.dbases_p, P.dshift,
                                T, H, false, p);
        k_reduce<<<256, 128>>>(P.dgamma_p, P.dbases_p, P.dscales_p, P.dgamma, P.dbases, P.dscales, T, H);
    });
    double bytes_rt = 2.0 * 2.0 * (double)T * D * 4.0;
    printf("    融合省掉的全局往返：dAbase[T,D]+dz[T,D] 各 写+读 = %.4f GB（T*D=%zu）\n",
           bytes_rt / 1e9, (size_t)T * D);
    for (void* q : buf.all) cudaFree(q);
}

int main() {
    printf("== A. device 六段（K0/G0/K1/G1a/G1b/K2）fp32 vs host 单片 double 参考 ==\n");
    check_case(64, 8, 3, false, 7);
    check_case(64, 8, 3, true, 7);
    check_case(128, 32, 4, false, 11);
    check_case(128, 32, 4, true, 11);
    check_case(256, 64, 2, false, 13);
    check_case(256, 64, 5, true, 13);
    printf("  （device 是 6 段拆分 + fn 全 batch 共享，参考是单片 double 逐 token；两者语义一致。）\n");
    printf("\n== B. 元素级三段在 H100 上的真实设备计时 ==\n");
    timing(65536, 64);
    printf("\n== C. P122-1 fused per-token worker（K1+G1a+K2 融进一个 block）vs 六段参考 ==\n");
    check_fused_case(64, 8, 3, false, 7);
    check_fused_case(64, 8, 3, true, 7);
    check_fused_case(128, 32, 4, false, 11);
    check_fused_case(128, 32, 4, true, 11);
    check_fused_case(256, 64, 2, false, 13);
    check_fused_case(256, 64, 5, true, 13);
    printf("  （融合只影响 K1/G1a/K2 的元素级部分；G0/G1b 仍是真实 GEMM，du0 仍落全局。）\n");
    printf("\n== D. 融合 vs 分段：元素级 K1+dz+K2 的真实设备计时 ==\n");
    timing_fused(65536, 64);
    printf("\nALL DONE\n");
    return 0;
}
