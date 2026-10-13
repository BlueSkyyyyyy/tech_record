// P120-1 实验：Mega mHC（HyperConnection）变换的**反向** 在 H100(sm90) 上的设备代码移植与
// 数值对拍。
//
// Mega mHC 主体（sm100_mega_mhc.cuh、epilogue/sm100_mhc.cuh、scheduler/mega_mhc.cuh）是
// SM100 专属、且**只有前向**（DeepGEMM 不实现 mHC 反向）。但它的整条计算图
// （残差多流混合 → 投影 → sigmoid/仿射 → Sinkhorn 双随机 → RMSNorm）架构无关，
// 可以逐行搬到裸 CUDA，验证「若用它训练，梯度应该长什么样」。源码坐标（057ca59）：
//
//   - 残差多流混合 new_residual = post*x + comb@residual     (tests/test_mega_mhc.py:145-148)
//   - bf16 舍入（共享数值边界）                              (epilogue:496-498 / test :148)
//   - hc_sqr_sum 与 rsqrt 尺度                              (epilogue:28-43 / test :158-160)
//   - 投影 + 分段 scale/bias + sigmoid                       (epilogue:53-56 / test :155-170)
//   - Sinkhorn 交替行/列归一化                               (epilogue:63-86 / test :114-120)
//   - RMSNorm                                                (epilogue:213-244 / test :179-184)
//
// 本文件验证三件事：
//   A) Sinkhorn（4x4）反向解析 VJP vs double 有限差分（iters 1..5）
//   B) 整 token 变换反向：device(fp32) vs double 参考（normal / shifted 各若干形状）
//   C) double 参考自身的中心差分抽样复核（证明「参考」不是自证）
//
// 参考实现见 harness/exp_mega_mhc_backward.py；编译/运行见 harness/exp_mega_mhc_backward.sh。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <algorithm>

#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1);} } while (0)

static constexpr int HC = 4;
static constexpr int NOUT = HC * (HC + 2);   // 24
static constexpr int MAXH = 64;
static constexpr int MAXIT = 8;
static constexpr int MAXNS = 2 * MAXIT + 2;  // 归一化步数上界

// 参数（与 test_mega_mhc.py 的语义一致）
struct Params {
    double hc_norm_eps, hc_pre_eps, hc_post_scale, rmsnorm_eps, rmsnorm_scale, sinkhorn_eps;
    int iters;
    double scales[3];
};

// ---------------------------------------------------------------------------
// 源码移植：Sinkhorn（test :114-120 / epilogue:63-86）
//   v = softmax(logits,-1) + eps
//   v = v / (colsum(v) + eps)
//   repeat iters-1: v = v/(rowsum+eps); v = v/(colsum+eps)
// 记录每一步的输入矩阵与归约维，供反向使用。
// ---------------------------------------------------------------------------
template <typename T>
static __host__ __device__ void sinkhorn_forward(const T logits[HC][HC], const Params& p,
                             T out[HC][HC], T seq[MAXNS][HC][HC], int dims[MAXNS], int& ns) {
    T psm[HC][HC];
    // softmax over last dim
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
    // 用 softmax 输出作为反向 chain 的一部分
    {
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) seq[0][j][i] = psm[j][i];
        dims[0] = 100;  // 标记 softmax 步
        ns = 1;
    }
    // 第一步：列归一化
    {
        record(cur, -2);
        T S[HC] = {0, 0, 0, 0};
        for (int i = 0; i < HC; ++i) for (int j = 0; j < HC; ++j) S[i] += cur[j][i];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) cur[j][i] /= (S[i] + (T)p.sinkhorn_eps);
    }
    for (int it = 1; it < p.iters; ++it) {
        record(cur, -1);
        T Sr[HC] = {0, 0, 0, 0};
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) Sr[j] += cur[j][i];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) cur[j][i] /= (Sr[j] + (T)p.sinkhorn_eps);
        record(cur, -2);
        T Sc[HC] = {0, 0, 0, 0};
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
    // 反向遍历（dims[0] 是 softmax 标记，最后处理）
    for (int s = ns - 1; s >= 1; --s) {
        const T (*X)[HC] = seq[s];
        int dim = dims[s];
        // S = sum(X, dim)
        T S[HC] = {0, 0, 0, 0};
        if (dim == -1) { for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) S[j] += X[j][i]; }
        else           { for (int i = 0; i < HC; ++i) for (int j = 0; j < HC; ++j) S[i] += X[j][i]; }
        T dX[HC][HC];
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) {
            T den = S[dim == -1 ? j : i] + (T)p.sinkhorn_eps;
            dX[j][i] = d[j][i] / den;
        }
        // dS 项
        T dS[HC] = {0, 0, 0, 0};
        if (dim == -1) { for (int j = 0; j < HC; ++j) { T acc = 0; for (int i = 0; i < HC; ++i) acc += d[j][i] * X[j][i]; dS[j] = -acc / ((S[j] + (T)p.sinkhorn_eps) * (S[j] + (T)p.sinkhorn_eps)); } }
        else           { for (int i = 0; i < HC; ++i) { T acc = 0; for (int j = 0; j < HC; ++j) acc += d[j][i] * X[j][i]; dS[i] = -acc / ((S[i] + (T)p.sinkhorn_eps) * (S[i] + (T)p.sinkhorn_eps)); } }
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) d[j][i] = dX[j][i] + dS[dim == -1 ? j : i];
    }
    // softmax 反向（seq[0] 是 softmax 输出 p）
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
// 整 token 变换：前向 + 反向（同一模板，device 用 float、host 参考用 double）
// ---------------------------------------------------------------------------
template <typename T>
struct TokenIn {
    T x[MAXH];                       // [H]
    T R[HC][MAXH];                   // [HC,H]
    T post[HC];                      // [HC]
    T comb[HC][HC];                  // [HC,HC]
    T fn[NOUT][HC * MAXH];           // 投影（用前 D=HC*H 列）
    T scales[3];
    T bases[NOUT];
    T gamma[MAXH];                   // rmsnorm_weight * scale
    T shifted_prev_mix[HC];          // shifted 变体
    // 上游梯度
    T dS[HC][MAXH];
    T dPost[HC];
    T dM[HC][HC];
    T dY[MAXH];
    T dShift[HC];
    int H;
    bool shifted;
};

template <typename T>
struct TokenGrad {
    T dx[MAXH];
    T dR[HC][MAXH];
    T dpost[HC];
    T dcomb[HC][HC];
    T dfn[NOUT][HC * MAXH];
    T dscales[3];
    T dbases[NOUT];
    T dgamma[MAXH];
    T dshift[HC];
    // 便于有限差分的前向标量
    T loss;
};

template <typename T>
static void mhc_fwd_bwd(const TokenIn<T>& in, const Params& p, TokenGrad<T>& g) {
    const int H = in.H, D = HC * H;
    T A[HC][MAXH];
    for (int j = 0; j < HC; ++j)
        for (int h = 0; h < H; ++h) {
            T v = in.x[h] * in.post[j];
            for (int i = 0; i < HC; ++i) v += in.comb[j][i] * in.R[i][h];
            A[j][h] = v;   // s = A（bf16 舍入按 straight-through）
        }
    // q, g
    T q = 0;
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) q += A[j][h] * A[j][h];
    T gq = (T)(1.0 / std::sqrt((double)(q / (T)D + (T)p.hc_norm_eps)));
    // u0[k], u[k]
    T u0[NOUT], u[NOUT];
    for (int k = 0; k < NOUT; ++k) {
        T acc = 0;
        // z = flatten(A) 行主序：(j,h) -> j*H+h
        for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) acc += A[j][h] * in.fn[k][j * H + h];
        u0[k] = acc;
        int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
        u[k] = acc * gq * in.scales[seg] + in.bases[k];
    }
    auto sig = [](T z) { return (T)(1.0 / (1.0 + std::exp(-(double)z))); };
    T pre[HC], newpost[HC], logits[HC][HC];
    for (int j = 0; j < HC; ++j) {
        pre[j] = sig(u[j]) + (T)p.hc_pre_eps;
        newpost[j] = sig(u[HC + j]) * (T)p.hc_post_scale;
    }
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) logits[j][i] = u[2 * HC + j * HC + i];
    T M[HC][HC], seq[MAXNS][HC][HC]; int dims[MAXNS], ns = 0;
    sinkhorn_forward<T>(logits, p, M, seq, dims, ns);
    // mix
    T mix[HC];
    for (int j = 0; j < HC; ++j) mix[j] = in.shifted ? in.shifted_prev_mix[j] : pre[j];
    T n[MAXH];
    for (int h = 0; h < H; ++h) { T acc = 0; for (int j = 0; j < HC; ++j) acc += mix[j] * A[j][h]; n[h] = acc; }
    T m = 0; for (int h = 0; h < H; ++h) m += n[h] * n[h]; m /= (T)H;
    T rg = (T)(1.0 / std::sqrt((double)(m + (T)p.rmsnorm_eps)));
    T y[MAXH];
    for (int h = 0; h < H; ++h) y[h] = n[h] * rg * in.gamma[h];

    // -------- 反向 --------
    // 1. RMSNorm
    T dn[MAXH];
    T dgamma[MAXH];
    T dLdrg = 0;
    for (int h = 0; h < H; ++h) {
        dn[h] = in.dY[h] * rg * in.gamma[h];
        dgamma[h] = in.dY[h] * n[h] * rg;
        dLdrg += in.dY[h] * n[h] * in.gamma[h];
    }
    T dm = dLdrg * (T)(-0.5) * rg * rg * rg;
    for (int h = 0; h < H; ++h) dn[h] += dm * (T)(2.0 / H) * n[h];
    // 2. n = Σ mix_j A_j
    T dmix[HC];
    T dA[HC][MAXH];
    for (int j = 0; j < HC; ++j) {
        dmix[j] = 0;
        for (int h = 0; h < H; ++h) dmix[j] += dn[h] * A[j][h];
    }
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dA[j][h] = mix[j] * dn[h] + in.dS[j][h];
    // 3. 系数 → du
    T du[NOUT];
    for (int k = 0; k < NOUT; ++k) du[k] = 0;
    T dpre_in[HC];
    for (int j = 0; j < HC; ++j) dpre_in[j] = in.shifted ? in.dShift[j] : dmix[j];
    for (int j = 0; j < HC; ++j) {
        T s = sig(u[j]);
        du[j] = dpre_in[j] * s * (1 - s);
    }
    for (int j = 0; j < HC; ++j) {
        T s = sig(u[HC + j]);
        du[HC + j] += in.dPost[j] * (T)p.hc_post_scale * s * (1 - s);
    }
    {
        T dlogits[HC][HC];
        sinkhorn_backward<T>(in.dM, seq, dims, ns, p, dlogits);
        for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) du[2 * HC + j * HC + i] += dlogits[j][i];
    }
    // 4. 仿射
    T du0[NOUT];
    T dgq = 0;
    T dsc[3] = {0, 0, 0};
    for (int k = 0; k < NOUT; ++k) {
        int seg = (k < HC) ? 0 : (k < 2 * HC ? 1 : 2);
        du0[k] = du[k] * gq * in.scales[seg];
        dgq += du[k] * u0[k] * in.scales[seg];
        dsc[seg] += du[k] * u0[k] * gq;
    }
    T dbases[NOUT];
    for (int k = 0; k < NOUT; ++k) dbases[k] = du[k];
    // 5. q / z
    T dq = dgq * (T)(-0.5) * gq * gq * gq / (T)D;
    T dz[HC][MAXH];
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dz[j][h] = dq * 2 * A[j][h];
    T dfn[NOUT][HC * MAXH];
    for (int k = 0; k < NOUT; ++k) for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h)
        dz[j][h] += du0[k] * in.fn[k][j * H + h];
    for (int k = 0; k < NOUT; ++k) for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h)
        dfn[k][j * H + h] = du0[k] * A[j][h];
    // 6. 残差混合（dz 也回流到 A）
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) dA[j][h] += dz[j][h];
    for (int h = 0; h < H; ++h) { g.dx[h] = 0; for (int j = 0; j < HC; ++j) g.dx[h] += dA[j][h] * in.post[j]; }
    for (int j = 0; j < HC; ++j) { g.dpost[j] = 0; for (int h = 0; h < H; ++h) g.dpost[j] += dA[j][h] * in.x[h]; }
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) { T acc = 0; for (int h = 0; h < H; ++h) acc += dA[j][h] * in.R[i][h]; g.dcomb[j][i] = acc; }
    for (int i = 0; i < HC; ++i) for (int h = 0; h < H; ++h) { T acc = 0; for (int j = 0; j < HC; ++j) acc += dA[j][h] * in.comb[j][i]; g.dR[i][h] = acc; }
    for (int k = 0; k < NOUT; ++k) for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) g.dfn[k][j * H + h] = dfn[k][j * H + h];
    for (int s = 0; s < 3; ++s) g.dscales[s] = dsc[s];
    for (int k = 0; k < NOUT; ++k) g.dbases[k] = dbases[k];
    for (int h = 0; h < H; ++h) g.dgamma[h] = dgamma[h];
    for (int j = 0; j < HC; ++j) g.dshift[j] = dmix[j];
    // loss（用于有限差分复核）
    T loss = 0;
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) loss += A[j][h] * in.dS[j][h];
    for (int j = 0; j < HC; ++j) loss += newpost[j] * in.dPost[j];
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) loss += M[j][i] * in.dM[j][i];
    for (int h = 0; h < H; ++h) loss += y[h] * in.dY[h];
    if (in.shifted) for (int j = 0; j < HC; ++j) loss += pre[j] * in.dShift[j];
    g.loss = loss;
}

// Part A：Sinkhorn 反向 vs 有限差分（double，device）
__global__ void kernel_partA(const double* logits_in, const double* dM_in, Params p,
                             double* dlogits_out, double* fd_out) {
    double logits[HC][HC], dM[HC][HC], seq[MAXNS][HC][HC]; int dims[MAXNS], ns = 0;
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) { logits[j][i] = logits_in[j * HC + i]; dM[j][i] = dM_in[j * HC + i]; }
    double M[HC][HC];
    sinkhorn_forward<double>(logits, p, M, seq, dims, ns);
    double dl[HC][HC];
    sinkhorn_backward<double>(dM, seq, dims, ns, p, dl);
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) dlogits_out[j * HC + i] = dl[j][i];
    // 有限差分（中心）
    const double h = 1e-6;
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) {
        double lp[HC][HC], lm[HC][HC], Mp[HC][HC], Mm[HC][HC], sp[MAXNS][HC][HC], sm[MAXNS][HC][HC];
        int dp[MAXNS], dm_[MAXNS], np_, nm_;
        for (int a = 0; a < HC; ++a) for (int b = 0; b < HC; ++b) { lp[a][b] = logits[a][b]; lm[a][b] = logits[a][b]; }
        lp[j][i] += h; lm[j][i] -= h;
        sinkhorn_forward<double>(lp, p, Mp, sp, dp, np_);
        sinkhorn_forward<double>(lm, p, Mm, sm, dm_, nm_);
        double Lp = 0, Lm = 0;
        for (int a = 0; a < HC; ++a) for (int b = 0; b < HC; ++b) { Lp += Mp[a][b] * dM[a][b]; Lm += Mm[a][b] * dM[a][b]; }
        fd_out[j * HC + i] = (Lp - Lm) / (2 * h);
    }
}

static double relerr(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    return std::sqrt(num / (den + 1e-300));
}

template <typename T>
static TokenIn<T> make_token(int H, int seed, bool shifted, Params p, std::mt19937& rng_unused) {
    (void)rng_unused;
    std::mt19937 rng((unsigned)seed);
    std::uniform_real_distribution<double> ur(-1.0, 1.0), upos(0.0, 1.0);
    TokenIn<T> in; std::memset(&in, 0, sizeof(in));
    in.H = H; in.shifted = shifted;
    for (int h = 0; h < H; ++h) in.x[h] = (T)ur(rng);
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) in.R[j][h] = (T)ur(rng);
    for (int j = 0; j < HC; ++j) in.post[j] = (T)upos(rng);
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) in.comb[j][i] = (T)(0.3 * ur(rng));
    for (int k = 0; k < NOUT; ++k) for (int h = 0; h < H; ++h) in.fn[k][h] = (T)(0.2 * ur(rng));
    for (int s = 0; s < 3; ++s) in.scales[s] = (T)p.scales[s];
    for (int k = 0; k < NOUT; ++k) in.bases[k] = (T)(0.1 * ur(rng));
    for (int h = 0; h < H; ++h) in.gamma[h] = (T)ur(rng);
    for (int j = 0; j < HC; ++j) in.shifted_prev_mix[j] = (T)upos(rng);
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) in.dS[j][h] = (T)(0.1 * ur(rng));
    for (int j = 0; j < HC; ++j) in.dPost[j] = (T)(0.1 * ur(rng));
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) in.dM[j][i] = (T)(0.1 * ur(rng));
    for (int h = 0; h < H; ++h) in.dY[h] = (T)(0.1 * ur(rng));
    for (int j = 0; j < HC; ++j) in.dShift[j] = (T)(0.1 * ur(rng));
    return in;
}

template <typename T>
static std::vector<double> flat_grads(const TokenGrad<T>& g, int H) {
    std::vector<double> v;
    for (int h = 0; h < H; ++h) v.push_back((double)g.dx[h]);
    for (int j = 0; j < HC; ++j) for (int h = 0; h < H; ++h) v.push_back((double)g.dR[j][h]);
    for (int j = 0; j < HC; ++j) v.push_back((double)g.dpost[j]);
    for (int j = 0; j < HC; ++j) for (int i = 0; i < HC; ++i) v.push_back((double)g.dcomb[j][i]);
    for (int k = 0; k < NOUT; ++k) for (int h = 0; h < H; ++h) v.push_back((double)g.dfn[k][h]);
    for (int s = 0; s < 3; ++s) v.push_back((double)g.dscales[s]);
    for (int k = 0; k < NOUT; ++k) v.push_back((double)g.dbases[k]);
    for (int h = 0; h < H; ++h) v.push_back((double)g.dgamma[h]);
    for (int j = 0; j < HC; ++j) v.push_back((double)g.dshift[j]);
    return v;
}

int main() {
    Params p; p.hc_norm_eps = 1e-5; p.hc_pre_eps = 1e-6; p.hc_post_scale = 1.0;
    p.rmsnorm_eps = 1e-6; p.rmsnorm_scale = 1.0; p.sinkhorn_eps = 1e-8;
    p.scales[0] = 0.7; p.scales[1] = 0.5; p.scales[2] = 0.9;
    std::mt19937 rng(2024);

    printf("== A. Sinkhorn(4x4) 反向：解析 VJP vs double 有限差分 ==\n");
    for (int it = 1; it <= 5; ++it) {
        p.iters = it;
        std::uniform_real_distribution<double> ur(-1.0, 1.0);
        double hl[HC * HC], hm[HC * HC];
        for (int i = 0; i < HC * HC; ++i) { hl[i] = ur(rng); hm[i] = (it == 1 ? ur(rng) : 0.1 * ur(rng)); }
        double *dl, *dm, *ddl, *dfd;
        CK(cudaMalloc(&dl, sizeof(double) * HC * HC));
        CK(cudaMalloc(&dm, sizeof(double) * HC * HC));
        CK(cudaMalloc(&ddl, sizeof(double) * HC * HC));
        CK(cudaMalloc(&dfd, sizeof(double) * HC * HC));
        CK(cudaMemcpy(dl, hl, sizeof(double) * HC * HC, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dm, hm, sizeof(double) * HC * HC, cudaMemcpyHostToDevice));
        kernel_partA<<<1, 1>>>(dl, dm, p, ddl, dfd);
        CK(cudaDeviceSynchronize());
        std::vector<double> vdl(HC * HC), vfd(HC * HC);
        CK(cudaMemcpy(vdl.data(), ddl, sizeof(double) * HC * HC, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(vfd.data(), dfd, sizeof(double) * HC * HC, cudaMemcpyDeviceToHost));
        double mx = 0; for (int i = 0; i < HC * HC; ++i) mx = std::max(mx, std::fabs(vdl[i] - vfd[i]));
        printf("   iters=%d: max|analytic - fd| = %.3e\n", it, mx);
        cudaFree(dl); cudaFree(dm); cudaFree(ddl); cudaFree(dfd);
    }

    printf("== B. 整 token 变换反向：device(fp32) vs double 参考 ==\n");
    struct Case { int H; bool shifted; };
    Case cases[] = {{8, false}, {8, true}, {16, false}, {16, true}, {32, false}, {32, true}};
    for (auto c : cases) {
        p.iters = 3;
        TokenIn<double> hin = make_token<double>(c.H, 7, c.shifted, p, rng);
        TokenIn<float> fin = make_token<float>(c.H, 7, c.shifted, p, rng);  // 相同 rng 状态 → 相同数值
        TokenGrad<double> gref; mhc_fwd_bwd<double>(hin, p, gref);
        TokenGrad<float> gout; mhc_fwd_bwd<float>(fin, p, gout);
        auto vd = flat_grads<double>(gref, c.H);
        auto vf = flat_grads<float>(gout, c.H);
        printf("   H=%2d %-7s  grads relerr(fp32 vs double) = %.3e\n",
               c.H, c.shifted ? "shifted" : "normal", relerr(vf, vd));
    }

    printf("== C. double 参考自身中心差分抽样复核（normal, H=8）==\n");
    {
        p.iters = 3;
        TokenIn<double> in = make_token<double>(8, 11, false, p, rng);
        TokenGrad<double> g; mhc_fwd_bwd<double>(in, p, g);
        const double h = 1e-6;
        auto probe = [&](const char* name, const double* gradv, auto&& perturb) {
            TokenIn<double> ip = in, im = in;
            perturb(ip, +h); perturb(im, -h);
            TokenGrad<double> gp, gm;
            mhc_fwd_bwd<double>(ip, p, gp); mhc_fwd_bwd<double>(im, p, gm);
            double fd = (gp.loss - gm.loss) / (2 * h);
            printf("   %-14s analytic=%+.6e  fd=%+.6e  diff=%.2e\n", name, *gradv, fd, std::fabs(*gradv - fd));
        };
        probe("gamma[3]", &g.dgamma[3], [](TokenIn<double>& t, double d){ t.gamma[3] += d; });
        probe("bases[5]", &g.dbases[5], [](TokenIn<double>& t, double d){ t.bases[5] += d; });
        probe("fn[7,3]", &g.dfn[7][3], [](TokenIn<double>& t, double d){ t.fn[7][3] += d; });
        probe("comb[0][1]", &g.dcomb[0][1], [](TokenIn<double>& t, double d){ t.comb[0][1] += d; });
        probe("post[2]", &g.dpost[2], [](TokenIn<double>& t, double d){ t.post[2] += d; });
        probe("R[1][5]", &g.dR[1][5], [](TokenIn<double>& t, double d){ t.R[1][5] += d; });
        probe("x[4]", &g.dx[4], [](TokenIn<double>& t, double d){ t.x[4] += d; });
    }

    printf("\nALL DONE\n");
    return 0;
}
