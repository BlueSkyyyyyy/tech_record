// P132-1 实验：Mega MoE **前向 L1 epilogue**（in-place SwiGLU + top-k 权重 + per-token UE8M0 量化）
// 的架构无关数值语义与**定时预算**，在 H100(sm90) 上的忠实移植。
//
// Mega MoE 主体是 SM100 专属（`mega_moe.hpp:367` 只在 arch_major==10 派发；UMMA/tcgen05/TMEM
// 在 sm90 不可用），H100 端到端跑不了。但 L1 epilogue 的数值原语是架构无关的，本文件把它们搬到裸
// CUDA 并**计时**：
//
//   A. L1 in-place SwiGLU（`sm100_fp8_fp4_mega_moe.cuh:1066-1134`）
//      - gate/up 是 fp32 累加（TMEM 读回，granularity-8 交错的 W1 布局）；
//      - silu(gate)=gate/(1+exp(-gate))：`kFastMath` 用 __expf + rcp.approx（:1125-1129）；
//      - clamp（:1108-1120，up 双侧、gate 单侧）；
//      - out = gate*up*weight（:1134 `__fmul2_rn(__fmul2_rn(gate,up),weights)`，weight 来自 combine 的 top-k）。
//   B. per-token UE8M0 量化到 FP8（:1179-1192）
//      - 每 128 个元素一个 scale：amax/448 → UE8M0 上取整到 2 的幂（`math.cuh:111-147` 的整数进位法）；
//      - q = round_to_e4m3(x / scale)。
//
// 参考实现用 double 在同文件内独立实现。编译/运行见 harness/exp_forward_roofline.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_bf16.h>

#define DEV __device__ __forceinline__
#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

// math.cuh:57-66 的 fma2（sm90 回退）与 math.cuh:68-72 的 fast_rcp 的等价 fp32 版本。
DEV float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

// `math.cuh:111-135` get_ue8m0_sf_exp 的架构无关复刻：把 amax/quant_max 向上取到 2 的幂（E4M3 quant_max=448）。
// 这里直接在浮点域做「ceil 到 2 的幂」，与整数进位法数值等价。
DEV float ue8m0_ceil_pow2(float v) {
    if (!(v > 0.0f)) return 1.0f;               // 全零块：任意合法正 scale
    int e;
    frexpf(v, &e);                              // v = frac * 2^e, frac in [0.5,1)
    float lo = ldexpf(1.0f, e - 1);             // 2^(e-1) <= v < 2^e
    if (v <= lo) return lo;                     // v 恰为 2 的幂
    return ldexpf(1.0f, e);                     // 否则最小的 > v 的 2 的幂
}

DEV unsigned char to_e4m3(float x) {
    // 用 __nv_fp8 的转换（RNE）；此处走 cutlass/cuda 的 fp8 类型。
    __nv_fp8_e4m3 y = __nv_fp8_e4m3(x);
    return *reinterpret_cast<unsigned char*>(&y);
}

// ---------------------------------------------------------------------------
// 核心：一段 128 元素的 SwiGLU + 量化（一个 block 一个 chunk）
// ---------------------------------------------------------------------------
// gu[row, 0:I] = gate, gu[row, I:2I] = up（真机里是 gran-8 交错的 W1 布局，值域等价）。
// 一个 warp 处理一个 128-chunk（lane 各 4 个元素），amax 用 warp shuffle 归约。
#define WARP_ELEMS 4
__global__ void l1_epilogue_kernel(const float* __restrict__ gu,
                                   const float* __restrict__ weights,
                                   int m, int I, float clamp, int fast_math,
                                   unsigned char* __restrict__ out_fp8,
                                   float* __restrict__ out_sf) {
    const int nchunks = I / 128;
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x & 31;
    if (warp >= m * nchunks) return;
    const int row = warp / nchunks;
    const int c = warp % nchunks;
    const int base = c * 128 + lane * WARP_ELEMS;
    const float w = weights[row];

    float hh[WARP_ELEMS];
    #pragma unroll
    for (int j = 0; j < WARP_ELEMS; ++j) {
        // gate/up 先 round 到 bf16（:1112-1113 __float22bfloat162_rn）——这是 FP8/BF16 两个变体共享的数值边界
        float gate = __bfloat162float(__float2bfloat16_rn(gu[row * I + base + j]));
        float up = __bfloat162float(__float2bfloat16_rn(gu[row * I + I + base + j]));
        if (clamp != INFINITY) {
            gate = fminf(gate, clamp);
            up = fmaxf(up, -clamp);
            up = fminf(up, clamp);
        }
        float neg = fast_math ? __expf(-gate) : expf(-gate);
        float denom = 1.0f + neg;
        gate = fast_math ? gate * fast_rcp(denom) : gate / denom;
        hh[j] = gate * up * w;
    }
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < WARP_ELEMS; ++j) amax = fmaxf(amax, fabsf(hh[j]));
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    float scale = ue8m0_ceil_pow2(amax / 448.0f);
    float inv = 1.0f / scale;
    #pragma unroll
    for (int j = 0; j < WARP_ELEMS; ++j) out_fp8[row * I + base + j] = to_e4m3(hh[j] * inv);
    if (lane == 0) out_sf[(row * I + c * 128) / 128] = scale;
}

// ---------------------------------------------------------------------------
// double 参考
// ---------------------------------------------------------------------------
static double ref_h(double g, double u, double w, double clamp) {
    // 与 kernel 一致：gate/up 先 RNE 到 bf16
    g = (double)__bfloat162float(__float2bfloat16_rn((float)g));
    u = (double)__bfloat162float(__float2bfloat16_rn((float)u));
    if (clamp != INFINITY) {
        g = std::fmin(g, clamp);
        u = std::fmax(u, -clamp);
        u = std::fmin(u, clamp);
    }
    double s = g / (1.0 + std::exp(-g));
    return s * u * w;
}

// host 端 E4M3 解码（bias=7，4 位指数、3 位尾数）
static float e4m3_decode(unsigned char b) {
    int s = (b >> 7) & 1, e = (b >> 3) & 0xF, mm = b & 7;
    if (e == 0) return (s ? -1.f : 1.f) * std::ldexp((float)mm / 8.0f, -6);
    if (e == 0xF && mm == 7) return NAN;
    return (s ? -1.f : 1.f) * std::ldexp(1.0f + (float)mm / 8.0f, e - 7);
}

int main() {
    cudaFree(0);
    std::mt19937_64 rng(20241013);
    std::normal_distribution<float> nd(0.0f, 2.0f);
    std::uniform_real_distribution<float> ud(0.3f, 1.2f);

    // ---- A) 小用例数值核对 ----
    {
        const int m = 8, I = 256;
        std::vector<float> h_gu(m * 2 * I), h_w(m);
        for (auto& v : h_gu) v = nd(rng);
        for (auto& v : h_w) v = ud(rng);
        float *d_gu, *d_w; unsigned char* d_out; float* d_sf;
        CK(cudaMalloc(&d_gu, h_gu.size() * 4));
        CK(cudaMalloc(&d_w, h_w.size() * 4));
        CK(cudaMalloc(&d_out, (size_t)m * I));
        CK(cudaMalloc(&d_sf, (size_t)m * I / 128 * 4));
        CK(cudaMemcpy(d_gu, h_gu.data(), h_gu.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_w, h_w.data(), h_w.size() * 4, cudaMemcpyHostToDevice));
        int blocks = (m * (I / 128) + 7) / 8;
        l1_epilogue_kernel<<<blocks, 256>>>(d_gu, d_w, m, I, INFINITY, 0, d_out, d_sf);
        CK(cudaDeviceSynchronize());
        std::vector<unsigned char> h_out(m * I);
        std::vector<float> h_sf(m * I / 128);
        CK(cudaMemcpy(h_out.data(), d_out, m * I, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(h_sf.data(), d_sf, m * I / 128 * 4, cudaMemcpyDeviceToHost));

        // 参考：double h → 每 128 chunk 的 UE8M0 scale → 反量化
        double max_abs_norm = 0.0, max_sf_rel = 0.0;
        for (int r = 0; r < m; ++r) {
            for (int c = 0; c < I / 128; ++c) {
                double amax = 0;
                for (int l = 0; l < 128; ++l)
                    amax = std::max(amax, std::fabs(ref_h(h_gu[r * I + c * 128 + l],
                                                         h_gu[r * I + I + c * 128 + l], h_w[r], INFINITY)));
                // 参考 scale：ceil 到 2 的幂（对精确 2 的幂稳定）
                double v = amax / 448.0;
                double sc = std::pow(2.0, std::floor(std::log2(std::max(v, 1e-30))));
                if (sc < std::max(v, 1e-30)) sc *= 2.0;
                max_sf_rel = std::max(max_sf_rel,
                                      std::fabs((double)h_sf[r * (I / 128) + c] - sc) / sc);
                for (int l = 0; l < 128; ++l) {
                    double href = ref_h(h_gu[r * I + c * 128 + l], h_gu[r * I + I + c * 128 + l], h_w[r], INFINITY);
                    float got = e4m3_decode(h_out[r * I + c * 128 + l]) * h_sf[r * (I / 128) + c];
                    // 用 chunk amax 归一（避免 href≈0 处的相对误差爆炸）
                    max_abs_norm = std::max(max_abs_norm, std::fabs(got - href) / (amax + 1e-30));
                }
            }
        }
        printf("== A) L1 epilogue 语义（fast_math=off）：8x256 vs fp64 参考  "
               "max|Δ|/amax=%.3e (fp8 地板 ~2^-4)  max_rel(scale)=%.1e ==\n", max_abs_norm, max_sf_rel);
        cudaFree(d_gu); cudaFree(d_w); cudaFree(d_out); cudaFree(d_sf);
    }

    // ---- B) DeepSeek 量级定时 ----
    {
        const int m = 4096, I = 2048;           // routed tokens（T=2048, topk=2）, I
        std::vector<float> h_gu((size_t)m * 2 * I), h_w(m);
        for (auto& v : h_gu) v = nd(rng);
        for (auto& v : h_w) v = ud(rng);
        float *d_gu, *d_w; unsigned char* d_out; float* d_sf;
        CK(cudaMalloc(&d_gu, h_gu.size() * 4));
        CK(cudaMalloc(&d_w, h_w.size() * 4));
        CK(cudaMalloc(&d_out, (size_t)m * I));
        CK(cudaMalloc(&d_sf, (size_t)m * I / 128 * 4));
        CK(cudaMemcpy(d_gu, h_gu.data(), h_gu.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_w, h_w.data(), h_w.size() * 4, cudaMemcpyHostToDevice));

        const int blocks = (m * (I / 128) + 7) / 8;
        for (int i = 0; i < 5; ++i) l1_epilogue_kernel<<<blocks, 256>>>(d_gu, d_w, m, I, INFINITY, 1, d_out, d_sf);
        CK(cudaDeviceSynchronize());
        cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
        const int iters = 200;
        cudaEventRecord(t0);
        for (int i = 0; i < iters; ++i) l1_epilogue_kernel<<<blocks, 256>>>(d_gu, d_w, m, I, INFINITY, 1, d_out, d_sf);
        cudaEventRecord(t1);
        CK(cudaDeviceSynchronize());
        float ms; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;
        double bytes = (double)m * 2 * I * 4          // 读 gate/up fp32
                     + (double)m * 4                   // 读 weight
                     + (double)m * I * 1               // 写 fp8
                     + (double)m * I / 128 * 4;        // 写 scale
        printf("== B) L1 epilogue 定时（m=%d, I=%d, fast_math=on）：%.1f us  %.1f GB/s  "
               "(读 %.1f MB + 写 %.1f MB = %.1f MB) ==\n",
               m, I, ms * 1e3, bytes / (ms * 1e-3) / 1e9,
               ((double)m * 2 * I * 4 + (double)m * 4) / 1e6,
               ((double)m * I + (double)m * I / 128 * 4) / 1e6, bytes / 1e6);
        cudaFree(d_gu); cudaFree(d_w); cudaFree(d_out); cudaFree(d_sf);
    }

    printf("\nALL DONE\n");
    return 0;
}
