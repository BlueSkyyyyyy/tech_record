// P117-1 实验：Mega Gate 路由（sqrt-softplus 打分 + 带 bias 排名 + top-k 归一化权重）的
// **反向** 在 H100(sm90) 上的设备代码移植与数值对拍。
//
// Mega Gate 主体（sm100_bf16_mega_gate.cuh、epilogue/sm100_mega_gate.cuh）是 SM100 专属、
// 且**只有前向**（DeepGEMM 不实现 gate 反向）。但它的路由**打分与权重语义**架构无关，可以
// 逐行搬到裸 CUDA 验证「若用它训练，梯度应该长什么样」。本文对照的源码坐标（057ca59）：
//
//   - sqrt_softplus 标量               (epilogue/sm100_mega_gate.cuh:19-24)
//   - sqrt_softplus 向量               (epilogue:26-39)
//   - bias 参与排名                    (impls/sm100_bf16_mega_gate.cuh:397-398)
//   - 无 bias 时排名用原始打分（单调） (impls:420-423)
//   - top-k 权重 = s_i / (Σ s + 1e-20) * scaling (epilogue:210-213)
//
// 本文件验证四件事：
//   A) sqrt_softplus 前向与解析导数 vs double（含阈值 20 分支、连续性）
//   B) 排名的单调等价性（无 bias 时按 z 选 == 按 s 选；有 bias 时可能不同）
//   C) 权重归一化的反向 Jacobian（解析式 vs double 有限差分）
//   D) 整 token 的 gate 反向：device 产出 dz[T,E]，host 用 double 参考算 dWg/dx 对拍
//
// 参考实现见 harness/exp_mega_gate_backward.py；编译/运行见 harness/exp_mega_gate_backward.sh。
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

#define DEV __device__ __forceinline__
#define CK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(_e), __LINE__); exit(1);} } while (0)

// ---------------------------------------------------------------------------
// 源码移植：epilogue/sm100_mega_gate.cuh:19-39
// ---------------------------------------------------------------------------
DEV float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}

// 解析导数 d/dz sqrt(softplus(z))，严格按**实现的分段函数**求导：
//   z > 20 : 实现返回 sqrt(z)      => 1 / (2 sqrt(z))
//   z ≤ 20 : 返回 sqrt(log1p(exp z)) => sigmoid(z) / (2 sqrt(softplus))
DEV float sqrt_softplus_grad(const float& value) {
    if (value > 20.0f) {
        const auto s = sqrtf(value);
        return 0.5f / s;
    }
    const auto e = expf(value);
    const auto softplus = log1pf(e);
    const auto s = sqrtf(softplus);
    const auto sigmoid = e / (1.0f + e);
    return 0.5f * sigmoid / s;
}

// double 参考（同一分段定义）
static __host__ __device__ double sqrt_softplus_d(const double& value) {
    const auto softplus = value > 20.0 ? value : log1p(exp(value));
    return sqrt(softplus);
}
static __host__ __device__ double sqrt_softplus_grad_d(const double& value) {
    if (value > 20.0) return 0.5 / sqrt(value);
    const auto e = exp(value);
    const auto s = sqrt(log1p(e));
    return 0.5 * (e / (1.0 + e)) / s;
}

// ---------------------------------------------------------------------------
// Part A: 前向 / 导数误差（float device vs double device）
// ---------------------------------------------------------------------------
__global__ void kernel_partA(const float* x, int n, float* fwd_err, float* grad_err) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float z = x[i];
    const double ref_f = sqrt_softplus_d((double)z);
    const double ref_g = sqrt_softplus_grad_d((double)z);
    const double got_f = (double)sqrt_softplus(z);
    const double got_g = (double)sqrt_softplus_grad(z);
    atomicMax((unsigned int*)fwd_err, __float_as_uint((float)fabs(got_f - ref_f)));
    atomicMax((unsigned int*)grad_err, __float_as_uint((float)fabs(got_g - ref_g)));
}

// ---------------------------------------------------------------------------
// Part C: 权重归一化反向 Jacobian（k 个选中槽）
//   w_i = s_i / S * g,   S = Σ_j s_j + eps
//   ds_i = g * (dy_i * S - Σ_j dy_j s_j) / S^2
// 用 double 有限差分逐槽对拍（device）。
// ---------------------------------------------------------------------------
#define MAXK 32
static __host__ __device__ double w_of_s(const double* s, int k, int i, double g, double eps) {
    double S = eps;
    for (int j = 0; j < k; ++j) S += s[j];
    return s[i] / S * g;
}
__global__ void kernel_partC(const double* s_in, const double* dy_in, int k, double g, double eps,
                             double* ds_analytic_out, double* ds_fd_out) {
    double s[MAXK], dy[MAXK], ds[MAXK];
    for (int i = 0; i < k; ++i) { s[i] = s_in[i]; dy[i] = dy_in[i]; }
    double S = eps; for (int i = 0; i < k; ++i) S += s[i];
    double c = 0.0; for (int i = 0; i < k; ++i) c += dy[i] * s[i];
    for (int i = 0; i < k; ++i) ds[i] = g * (dy[i] * S - c) / (S * S);
    // 有限差分：dL/ds_i，L = Σ_j dy_j w_j(s)
    const double h = 1e-6;
    for (int i = 0; i < k; ++i) {
        double sp[MAXK], sm[MAXK];
        for (int j = 0; j < k; ++j) { sp[j] = s[j]; sm[j] = s[j]; }
        sp[i] += h; sm[i] -= h;
        double lp = 0.0, lm = 0.0;
        for (int j = 0; j < k; ++j) { lp += dy[j] * w_of_s(sp, k, j, g, eps); lm += dy[j] * w_of_s(sm, k, j, g, eps); }
        ds_fd_out[i] = (lp - lm) / (2.0 * h);
    }
    for (int i = 0; i < k; ++i) ds_analytic_out[i] = ds[i];
}

// ---------------------------------------------------------------------------
// Part D: 整 token gate 反向。每个 warp 处理一个 token：
// 注：E ≤ 256（waves = E/32 ≤ MAXK_WAVES）。
//   z_e = <x_t, Wg[:,e]>,  s = sqrt_softplus(z), 排名由 host 给定的 sel（top-k）确定
//   S = Σ_{l<k} s[sel_l] + eps,  dy = dL/dw
//   ds_l = g (dy_l S - Σ dy_l s_l)/S^2 ;  dz_e = ds_{l:e=sel_l} * sqrt_softplus_grad(z_e)
// 输出 dz[T,E]（未选中的专家 dz=0）。
// ---------------------------------------------------------------------------
static constexpr int MAXK_WAVES = 8;

__global__ void kernel_partD(const float* x, const float* Wg, const int* sel, const float* dy,
                             int T, int H, int E, int K, float g, float eps, float* dz) {
    const int lane = threadIdx.x & 31;
    const int warp_in_block = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int warp_id = blockIdx.x * warps_per_block + warp_in_block;
    if (warp_id >= T) return;
    const int t = warp_id;

    const int waves = E / 32;
    float s[MAXK_WAVES];
    for (int w = 0; w < waves; ++w) {
        const int e = lane + w * 32;
        float acc = 0.0f;
        for (int h = 0; h < H; ++h) acc += x[t * H + h] * Wg[h * E + e];
        s[w] = sqrt_softplus(acc);
    }
    // 收集选中槽的 s
    float sel_s[MAXK];
    for (int l = 0; l < K; ++l) {
        const int e = sel[t * K + l];
        sel_s[l] = __shfl_sync(0xffffffffu, s[e >> 5], e & 31);
    }
    float S = eps;
    for (int l = 0; l < K; ++l) S += sel_s[l];
    float c = 0.0f;
    for (int l = 0; l < K; ++l) c += dy[t * K + l] * sel_s[l];
    float ds[MAXK];
    for (int l = 0; l < K; ++l) ds[l] = g * (dy[t * K + l] * S - c) / (S * S);

    for (int w = 0; w < waves; ++w) {
        const int e = lane + w * 32;
        float dze = 0.0f;
        for (int l = 0; l < K; ++l) {
            if (sel[t * K + l] == e) {
                // 重新算 z 以求导（与上面同一条 dot，保证一致）
                float acc = 0.0f;
                for (int h = 0; h < H; ++h) acc += x[t * H + h] * Wg[h * E + e];
                dze = ds[l] * sqrt_softplus_grad(acc);
                break;
            }
        }
        dz[t * E + e] = dze;
    }
}

// ---------------------------------------------------------------------------
// host 参考：double 前向 + 反向
// ---------------------------------------------------------------------------
struct Ref {
    std::vector<int> sel;         // [T,K]
    std::vector<double> dz;       // [T,E]
    std::vector<double> dWg;      // [H,E]
    std::vector<double> dx;       // [T,H]
    std::vector<double> w;        // [T,K]
};
static Ref reference(const std::vector<double>& x, const std::vector<double>& Wg,
                     const std::vector<double>& dy, int T, int H, int E, int K,
                     double g, double eps, const std::vector<double>* bias) {
    Ref r; r.sel.assign(T * K, 0); r.w.assign(T * K, 0.0);
    r.dz.assign((size_t)T * E, 0.0);
    r.dWg.assign((size_t)H * E, 0.0);
    r.dx.assign((size_t)T * H, 0.0);
    for (int t = 0; t < T; ++t) {
        std::vector<double> z(E), s(E), rank(E);
        for (int e = 0; e < E; ++e) {
            double acc = 0.0;
            for (int h = 0; h < H; ++h) acc += x[(size_t)t * H + h] * Wg[(size_t)h * E + e];
            z[e] = acc; s[e] = sqrt_softplus_d(acc);
            rank[e] = s[e] + (bias ? (*bias)[e] : 0.0);
        }
        std::vector<int> idx(E); std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(),
                          [&](int a, int b) { if (rank[a] != rank[b]) return rank[a] > rank[b]; return a < b; });
        double S = eps; for (int l = 0; l < K; ++l) S += s[idx[l]];
        double c = 0.0; for (int l = 0; l < K; ++l) c += dy[(size_t)t * K + l] * s[idx[l]];
        for (int l = 0; l < K; ++l) {
            r.sel[t * K + l] = idx[l];
            r.w[t * K + l] = s[idx[l]] / S * g;
            const double ssl = (g * (dy[(size_t)t * K + l] * S - c)) / (S * S);
            r.dz[(size_t)t * E + idx[l]] = ssl * sqrt_softplus_grad_d(z[idx[l]]);
        }
    }
    for (int h = 0; h < H; ++h)
        for (int e = 0; e < E; ++e) {
            double acc = 0.0;
            for (int t = 0; t < T; ++t) acc += x[(size_t)t * H + h] * r.dz[(size_t)t * E + e];
            r.dWg[(size_t)h * E + e] = acc;
        }
    for (int t = 0; t < T; ++t)
        for (int h = 0; h < H; ++h) {
            double acc = 0.0;
            for (int e = 0; e < E; ++e) acc += r.dz[(size_t)t * E + e] * Wg[(size_t)h * E + e];
            r.dx[(size_t)t * H + h] = acc;
        }
    return r;
}

static double relerr(const std::vector<double>& a, const std::vector<double>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { num += (a[i] - b[i]) * (a[i] - b[i]); den += b[i] * b[i]; }
    return std::sqrt(num / (den + 1e-300));
}

int main() {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> ur(-1.0, 1.0);

    // ---- Part A ----
    {
        const int N = 1 << 22;
        std::vector<float> hx(N);
        std::uniform_real_distribution<double> ud(-60.0, 60.0);
        for (int i = 0; i < N; ++i) hx[i] = (float)ud(rng);
        float *dx, *fe, *ge;
        CK(cudaMalloc(&dx, sizeof(float) * N));
        CK(cudaMalloc(&fe, sizeof(float)));
        CK(cudaMalloc(&ge, sizeof(float)));
        float zero = 0.0f; CK(cudaMemcpy(fe, &zero, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ge, &zero, 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dx, hx.data(), sizeof(float) * N, cudaMemcpyHostToDevice));
        kernel_partA<<<(N + 255) / 256, 256>>>(dx, N, fe, ge);
        CK(cudaDeviceSynchronize());
        float hfe, hge; CK(cudaMemcpy(&hfe, fe, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hge, ge, 4, cudaMemcpyDeviceToHost));
        printf("== A. sqrt_softplus 前向 / 解析导数 vs double (n=%d, z~U[-60,60]) ==\n", N);
        printf("   forward  max_abs_err = %.4e\n", hfe);
        printf("   grad     max_abs_err = %.4e\n", hge);
        auto probe = [](double z){ printf("   z=%+9.4f  ssp=%.6f  d(ssp)/dz=%.6f\n", z, sqrt_softplus_d(z), sqrt_softplus_grad_d(z)); };
        printf("   阈值 20 连续性探针:\n");
        probe(19.999f); probe(20.0f); probe(20.001f);
        probe(-60.0f); probe(0.0f); probe(40.0f);
        cudaFree(dx); cudaFree(fe); cudaFree(ge);
    }

    // ---- Part B: 排名单调等价性 ----
    {
        const int T = 4096, E = 128;
        std::uniform_real_distribution<double> uz(-8.0, 8.0);
        std::vector<double> bias(E);
        for (int e = 0; e < E; ++e) bias[e] = 0.3 * ur(rng) * (e < 16 ? 3.0 : 1.0);
        long long mismatch_no = 0, mismatch_bias = 0;
        for (int t = 0; t < T; ++t) {
            std::vector<double> z(E), s(E), rz(E), rs(E), rb(E);
            for (int e = 0; e < E; ++e) { z[e] = uz(rng); s[e] = sqrt_softplus_d(z[e]); rz[e] = z[e]; rs[e] = s[e]; rb[e] = s[e] + bias[e]; }
            auto rank_of = [&](const std::vector<double>& r) {
                std::vector<int> idx(E); std::iota(idx.begin(), idx.end(), 0);
                std::partial_sort(idx.begin(), idx.begin() + 8, idx.end(),
                                  [&](int a, int b){ if (r[a] != r[b]) return r[a] > r[b]; return a < b; });
                return std::vector<int>(idx.begin(), idx.begin() + 8);
            };
            if (rank_of(rz) != rank_of(rs)) ++mismatch_no;
            if (rank_of(rz) != rank_of(rb)) ++mismatch_bias;
        }
        printf("== B. 排名单调等价性 (T=%d, E=%d, topk=8) ==\n", T, E);
        printf("   无 bias：按 z 选 vs 按 s=sqrt_softplus(z) 选 不符 = %lld / %d\n", mismatch_no, T);
        printf("   有 bias：按 z 选 vs 按 s+bias 选         不符 = %lld / %d (bias 破坏单调，符合预期)\n", mismatch_bias, T);
    }

    // ---- Part C: 归一化反向 Jacobian ----
    {
        const int k = 8; const double g = 1.5, eps = 1e-20;
        std::vector<double> hs(k), hdy(k);
        std::uniform_real_distribution<double> us(0.2, 3.0), udy(-2.0, 2.0);
        for (int i = 0; i < k; ++i) { hs[i] = us(rng); hdy[i] = udy(rng); }
        double *ds, *ddy, *d_an, *d_fd;
        CK(cudaMalloc(&ds, 8 * k)); CK(cudaMalloc(&ddy, 8 * k)); CK(cudaMalloc(&d_an, 8 * k)); CK(cudaMalloc(&d_fd, 8 * k));
        CK(cudaMemcpy(ds, hs.data(), 8 * k, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(ddy, hdy.data(), 8 * k, cudaMemcpyHostToDevice));
        kernel_partC<<<1, 1>>>(ds, ddy, k, g, eps, d_an, d_fd);
        CK(cudaDeviceSynchronize());
        std::vector<double> an(k), fd(k);
        CK(cudaMemcpy(an.data(), d_an, 8 * k, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(fd.data(), d_fd, 8 * k, cudaMemcpyDeviceToHost));
        printf("== C. 权重归一化反向 Jacobian (k=%d, g=%.2f, eps=%.0e) ==\n", k, g, eps);
        printf("   解析式 vs double 有限差分  max_abs_err = %.4e\n",
               [&]{ double m = 0; for (int i = 0; i < k; ++i) m = std::max(m, std::fabs(an[i] - fd[i])); return m; }());
        cudaFree(ds); cudaFree(ddy); cudaFree(d_an); cudaFree(d_fd);
    }

    // ---- Part D: 整 token gate 反向 ----
    {
        const int T = 4096, H = 64, E = 128, K = 6;
        const double g = 1.5, eps = 1e-20;
        std::vector<double> x((size_t)T * H), Wg((size_t)H * E), dy((size_t)T * K);
        for (auto& v : x) v = ur(rng);
        for (auto& v : Wg) v = ur(rng) * 0.5;
        for (auto& v : dy) v = ur(rng);
        std::vector<double> bias(E);
        for (int e = 0; e < E; ++e) bias[e] = 0.4 * ur(rng) * (e < 16 ? 4.0 : 1.0);

        for (int with_bias = 0; with_bias <= 1; ++with_bias) {
            Ref ref = reference(x, Wg, dy, T, H, E, K, g, eps, with_bias ? &bias : nullptr);
            // to float inputs
            std::vector<float> fx((size_t)T * H), fW((size_t)H * E), fdy((size_t)T * K);
            for (size_t i = 0; i < fx.size(); ++i) fx[i] = (float)x[i];
            for (size_t i = 0; i < fW.size(); ++i) fW[i] = (float)Wg[i];
            for (size_t i = 0; i < fdy.size(); ++i) fdy[i] = (float)dy[i];
            std::vector<int> sel_i(T * K);
            for (int i = 0; i < T * K; ++i) sel_i[i] = ref.sel[i];

            float *d_x, *d_W, *d_dy, *d_dz; int* d_sel;
            CK(cudaMalloc(&d_x, 4 * (size_t)T * H)); CK(cudaMalloc(&d_W, 4 * (size_t)H * E));
            CK(cudaMalloc(&d_dy, 4 * (size_t)T * K)); CK(cudaMalloc(&d_dz, 4 * (size_t)T * E));
            CK(cudaMalloc(&d_sel, 4 * T * K));
            CK(cudaMemcpy(d_x, fx.data(), 4 * fx.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_W, fW.data(), 4 * fW.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_dy, fdy.data(), 4 * fdy.size(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(d_sel, sel_i.data(), 4 * sel_i.size(), cudaMemcpyHostToDevice));
            const int threads = 128; const int warps = threads / 32;
            kernel_partD<<<(T + warps - 1) / warps, threads>>>(d_x, d_W, d_sel, d_dy, T, H, E, K, (float)g, (float)eps, d_dz);
            CK(cudaDeviceSynchronize());
            std::vector<float> fdz((size_t)T * E);
            CK(cudaMemcpy(fdz.data(), d_dz, 4 * fdz.size(), cudaMemcpyDeviceToHost));

            std::vector<double> ddz((size_t)T * E);
            for (size_t i = 0; i < ddz.size(); ++i) ddz[i] = (double)fdz[i];
            // 由 device dz 在 host 上以 double 还原 dWg / dx
            std::vector<double> dWg_got((size_t)H * E, 0.0), dx_got((size_t)T * H, 0.0);
            for (int h = 0; h < H; ++h)
                for (int e = 0; e < E; ++e) {
                    double acc = 0.0;
                    for (int t = 0; t < T; ++t) acc += x[(size_t)t * H + h] * ddz[(size_t)t * E + e];
                    dWg_got[(size_t)h * E + e] = acc;
                }
            for (int t = 0; t < T; ++t)
                for (int h = 0; h < H; ++h) {
                    double acc = 0.0;
                    for (int e = 0; e < E; ++e) acc += ddz[(size_t)t * E + e] * Wg[(size_t)h * E + e];
                    dx_got[(size_t)t * H + h] = acc;
                }
            printf("== D. 整 token gate 反向 (%s, T=%d,H=%d,E=%d,topk=%d,scaling=%.2f) ==\n",
                   with_bias ? "with bias" : "no bias", T, H, E, K, g);
            printf("   dz  (device fp32) vs double : relerr = %.4e\n", relerr(ddz, ref.dz));
            printf("   dWg (由 device dz) vs double: relerr = %.4e\n", relerr(dWg_got, ref.dWg));
            printf("   dx  (由 device dz) vs double: relerr = %.4e\n", relerr(dx_got, ref.dx));
            // 权重前向也对一下
            std::vector<double> w_got((size_t)T * K);
            for (size_t i = 0; i < (size_t)T * K; ++i) w_got[i] = ref.w[i];
            printf("   (选中集合由 host double 给出，device 只算反向)\n");
            cudaFree(d_x); cudaFree(d_W); cudaFree(d_dy); cudaFree(d_dz); cudaFree(d_sel);
        }

        // epsilon 影响
        {
            Ref r0 = reference(x, Wg, dy, T, H, E, K, g, 0.0, nullptr);
            Ref r1 = reference(x, Wg, dy, T, H, E, K, g, 1e-20, nullptr);
            printf("== E. eps=1e-20 的影响 (no bias) ==\n");
            printf("   dz  relerr(eps=0 vs 1e-20) = %.4e\n", relerr(r0.dz, r1.dz));
        }
    }
    printf("\nALL DONE\n");
    return 0;
}
