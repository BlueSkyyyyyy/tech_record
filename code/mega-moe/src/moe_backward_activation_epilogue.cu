// P146-1 实验：把 MoE 反向的「SwiGLU 反向 + 激活梯度重量化」落地成 **SM100 dgrad 的
//   `epilogue_class` operator**——契约可行性精读（架构无关部分在 H100/sm90 真机跑）。
//
// 承接第 155 篇（P145-1）留下的口子①：
//   「把这段折叠在 SM100 上用新写的 epilogue_class operator 落地（当前 SM100 只有
//   `FP8QuantizationEpilogue`、无 SwiGLU 反向 operator；SM90 拒绝自定义 epilogue）」。
//   本篇回答：**只加一个 operator 结构体够不够？** 结论是**不够**——epilogue 契约是
//   「单输出 + 无逐元素索引 + 无状态」的，而 SwiGLU 反向是「双输出 + 需按元素读 gate/up」。
//
// 本文件（架构无关 CUDA，H100/sm90 真机可跑；SM100 专属的 TMEM/TMA 部分仅对照行号）：
//   [A] 契约尺寸账：真实 `EpilogueOperatorArgs`（types.cuh:47-52）sizeof vs 提议扩展 sizeof，
//       以及 `sizeof(epilogue_operator_t) == sizeof(EpilogueOperatorArgs)` 这个「算子不得加状态」
//       断言（sm100_fp8_fp4_gemm_1d1d.cuh:57-58、sm100_bf16_gemm.cuh:59-60）在扩展后是否仍成立。
//   [B] 用**真实** `common/math.cuh` 逐行复刻 `sm100_store_cd.cuh:144-198` 的 FP8 量化分支
//       （cast_into_bf16x2 → get_packed_bf16_amax → get_ue8m0_sf_exp/inv → scale_bf16x2_into_fp8x4），
//       把它套在「dact → SwiGLU 反向 → dgate/dup」上，与一份**独立手写**参考逐位对拍。
//   [C] 单输出 vs 双输出：量化 `d(gate)` 与 `d(up)` 的 SF 不可互换；坐实必须两个输出 + 两个 SFD。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// 真实 DeepGEMM 头：EpilogueOperatorArgs（types.cuh:47-52）与量化原语（common/math.cuh）
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/common/math.cuh>

using bf16 = __nv_bfloat16;
using deep_gemm::EpilogueOperatorArgs;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static __host__ __device__ __forceinline__ uint32_t f2u(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static __host__ __device__ __forceinline__ float u2f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

// 逐位复刻 `common/math.cuh:114-135` 的 UE8M0 选择（这里故意**手写**，与被测的真实 math.cuh 互为独立参考）
static __host__ __device__ __forceinline__ uint32_t hand_ue8m0_exp(float amax) {
    const uint32_t kMantissaMask = (1u << 23) - 1;
    const uint32_t kQuantMaxMantissa = 0x60u << (23 - 7);
    const uint32_t kQuantMaxExponent = 8;
    const uint32_t kMinSFExponent = 105;
    const uint32_t amax_bits = f2u(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> 23;
    return max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}
static __host__ __device__ __forceinline__ float hand_ue8m0_inv(uint32_t sf_exp) {
    return u2f((254u - sf_exp) << 23);
}

// SwiGLU 反向的逐元素数学（与前向 L1 epilogue 的逐元素逆一致，见第 154 篇 §2）
static __device__ __forceinline__ void swiglu_bwd(float gf, float uf, float dd, float w,
                                                  float* dg, float* du) {
    const float sigma = 1.0f / (1.0f + __expf(-gf));
    const float silu = gf * sigma;
    const float d_su = dd * w;
    *dg = d_su * uf * sigma * (1.0f + gf * (1.0f - sigma));
    *du = d_su * silu;
}

// ---------------------------------------------------------------------------
// [A] 提议的扩展 args：在 EpilogueOperatorArgs 的现有字段之后追加「逐元素输入 + 第二输出 + 第二 SFD」
//     —— 注意：这些字段进的是**共享的** args 基类（因为算子不得加状态，见 :57-58 的 sizeof 断言），
//        所以**每个** SM100 GEMM 的 kernel 参数都会变大，不只是 SwiGLU 这一个。
// ---------------------------------------------------------------------------
struct SwiGLUEpilogueArgs {
    // === 现有（逐字段同 types.cuh:47-52）===
    uint32_t* sfd = nullptr;          // 第一输出的 SFD（此处 = dgate 的 SF）
    uint32_t sfd_stride = 0;
    uint32_t shape_m = 0, shape_n = 0;
    float alpha = 1.0f;
    // === 提议扩展 ===
    const bf16* gate = nullptr;        // 逐元素输入（dgrad 不产出，须前向 dump 或反向重算）
    const bf16* up = nullptr;
    const float* weights = nullptr;    // combine 的 top-k 权重（每 token 一个）
    __nv_fp8_e4m3* dup8 = nullptr;     // 第二数据输出：d(up)
    uint32_t* dup_sfd = nullptr;       // 第二 SFD：d(up) 的 SF
};

// ---------------------------------------------------------------------------
// [B] 真实 store 路径的 FP8 分支算术：逐行对照 `sm100_store_cd.cuh:144-198`
//     一个 quant group = 32 个 fp32 累加器值（kSFGranN，:38）。真实 kernel 里一个 lane 一个 group
//     （注释 :93-94「one row, kSFGranN columns per lane」），amax 是**单线程内**的树归约（:154）。
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void store_quant_group_real(const uint32_t values[32],
                                                              __nv_fp8_e4m3* d8, uint32_t* sf_out) {
    using namespace deep_gemm;
    // :147-150 —— 先把累加器 round 成 BF16 对（输出逐位匹配「BF16 D 再走独立 cast」）
    nv_bfloat162 b[16];
    #pragma unroll
    for (int p = 0; p < 16; ++p) {
        uint32_t x = values[2 * p], y = values[2 * p + 1];
        b[p] = math::cast_into_bf16x2(x, y);
    }
    // :154-155 —— 组内 amax → UE8M0 SF → sf_inv
    const uint32_t sf_exp = math::get_ue8m0_sf_exp(math::get_packed_bf16_amax<16>(b));
    const auto sf_inv = __bfloat162bfloat162(math::get_ue8m0_sf_inv<nv_bfloat16>(sf_exp));
    // :158-168 —— 缩放并打包成 FP8x4（cd 是 e4m3，故一个 bank group = 16 元素）
    #pragma unroll
    for (int bg = 0; bg < 2; ++bg) {
        const int pi = bg * 8;
        uint32_t w0 = math::scale_bf16x2_into_fp8x4(b[pi + 0], b[pi + 1], sf_inv, sf_inv);
        uint32_t w1 = math::scale_bf16x2_into_fp8x4(b[pi + 2], b[pi + 3], sf_inv, sf_inv);
        uint32_t w2 = math::scale_bf16x2_into_fp8x4(b[pi + 4], b[pi + 5], sf_inv, sf_inv);
        uint32_t w3 = math::scale_bf16x2_into_fp8x4(b[pi + 6], b[pi + 7], sf_inv, sf_inv);
        uint32_t words[4] = {w0, w1, w2, w3};
        memcpy(reinterpret_cast<uint8_t*>(d8) + bg * 16, words, 16);
    }
    *sf_out = sf_exp;
}

// 提议算子：在 epilogue 里就地做 SwiGLU 反向，再各走一遍真实 FP8 store 分支（= 第 155 篇的折叠）。
// 注意：真实 `operators.cuh:13-14/22-29` 的 `apply_values(values)` **只拿得到 values、拿不到 (row,n)**，
//       所以这个算子**无法**只靠 `apply_values` 实现——它必须进 store 路径（能拿到 row_idx/n_idx，见
//       `operators.cuh:78` 的 store_sf 与 `sm100_store_cd.cuh:101` 的 row_idx）。
__global__ void epilogue_operator_kernel(const bf16* __restrict__ gate,
                                         const bf16* __restrict__ up,
                                         const bf16* __restrict__ dact,
                                         const float* __restrict__ weights,
                                         __nv_fp8_e4m3* __restrict__ dg8,
                                         __nv_fp8_e4m3* __restrict__ du8,
                                         uint8_t* __restrict__ sfg, uint8_t* __restrict__ sfu,
                                         int N, int64_t groups, int two_output) {
    const int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int gpr = N / 32;
    const int row = (int)(g / gpr);
    const int n0 = (int)((g % gpr) * 32);
    const float w = weights[row];
    uint32_t dg_raw[32], du_raw[32];
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        // 累加器（fp32 d(act)）先按 store 路径语义 round bf16
        const float dd = __bfloat162float(__float2bfloat16_rn(__bfloat162float(dact[(int64_t)row * N + n0 + k])));
        const float gf = __bfloat162float(gate[(int64_t)row * N + n0 + k]);
        const float uf = __bfloat162float(up[(int64_t)row * N + n0 + k]);
        float dg, du; swiglu_bwd(gf, uf, dd, w, &dg, &du);
        dg_raw[k] = f2u(dg); du_raw[k] = f2u(du);
    }
    uint32_t eg = 0, eu = 0;
    store_quant_group_real(dg_raw, &dg8[(int64_t)row * N + n0], &eg);
    if (two_output) {
        store_quant_group_real(du_raw, &du8[(int64_t)row * N + n0], &eu);
        sfu[g] = (uint8_t)eu;
    }
    sfg[g] = (uint8_t)eg;
}

// [B] 独立手写参考段（同第 144/145 篇的语义；量化用**手写**的 UE8M0，与被测的真实 math.cuh 路径互证）
__global__ void ref_segment_kernel(const bf16* __restrict__ gate,
                                   const bf16* __restrict__ up,
                                   const bf16* __restrict__ dact,
                                   const float* __restrict__ weights,
                                   __nv_fp8_e4m3* __restrict__ dg8,
                                   __nv_fp8_e4m3* __restrict__ du8,
                                   uint8_t* __restrict__ sfg, uint8_t* __restrict__ sfu,
                                   int N, int64_t groups) {
    const int64_t g = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int gpr = N / 32;
    const int row = (int)(g / gpr);
    const int n0 = (int)((g % gpr) * 32);
    const float w = weights[row];
    float dgf[32], duf[32], ag = 0.f, au = 0.f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        const float dd = __bfloat162float(__float2bfloat16_rn(__bfloat162float(dact[(int64_t)row * N + n0 + k])));
        const float gf = __bfloat162float(gate[(int64_t)row * N + n0 + k]);
        const float uf = __bfloat162float(up[(int64_t)row * N + n0 + k]);
        float dg, du; swiglu_bwd(gf, uf, dd, w, &dg, &du);
        // 与 store 路径一致：先 round bf16 再 amax/量化
        dgf[k] = __bfloat162float(__float2bfloat16_rn(dg));
        duf[k] = __bfloat162float(__float2bfloat16_rn(du));
        ag = fmaxf(ag, fabsf(dgf[k]));
        au = fmaxf(au, fabsf(duf[k]));
    }
    const uint32_t eg = hand_ue8m0_exp(ag), eu = hand_ue8m0_exp(au);
    const float ig = hand_ue8m0_inv(eg), iu = hand_ue8m0_inv(eu);
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
        dg8[(int64_t)row * N + n0 + k] = __nv_fp8_e4m3(dgf[k] * ig);
        du8[(int64_t)row * N + n0 + k] = __nv_fp8_e4m3(duf[k] * iu);
    }
    sfg[g] = (uint8_t)eg; sfu[g] = (uint8_t)eu;
}

template <typename T>
static T* dev_alloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }

int main() {
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("[env] device=%s sm_%d%d\n\n", prop.name, prop.major, prop.minor);

    // ---- [A] 契约尺寸账 ----
    printf("[A] epilogue 契约尺寸账（types.cuh:47-52 / sm100_*_gemm.cuh 的 sizeof 断言）\n");
    printf("    sizeof(EpilogueOperatorArgs)           = %zu B （真实头，types.cuh:47-52）\n",
           sizeof(EpilogueOperatorArgs));
    printf("    sizeof(SwiGLUEpilogueArgs)（提议扩展）  = %zu B\n", sizeof(SwiGLUEpilogueArgs));
    printf("    断言 `sizeof(epilogue_operator_t) == sizeof(EpilogueOperatorArgs)`：\n");
    printf("      - 只加**行为**（apply_values）时成立；\n");
    printf("      - 加**状态**（gate/up/dup 指针）必须进共享基类，于是**每个** SM100 GEMM 的 kernel 参数\n");
    printf("        从 %zu B 涨到 %zu B（+%zu B）——不是只影响 SwiGLU 这一个 kernel。\n\n",
           sizeof(EpilogueOperatorArgs), sizeof(SwiGLUEpilogueArgs),
           sizeof(SwiGLUEpilogueArgs) - sizeof(EpilogueOperatorArgs));

    // ---- [B] 被测（真实 math.cuh 路径）vs 手写参考 逐位对拍 ----
    const int T = 4096, N = 2048;         // d2g 输出形状 [Σt=T, I=N]
    const int64_t n = (int64_t)T * N;
    const int64_t groups = n / 32;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<bf16> hg(n), hu(n), hd(n);
    std::vector<float> hw(T);
    for (int64_t i = 0; i < n; ++i) {
        hg[i] = (bf16)nd(rng); hu[i] = (bf16)nd(rng); hd[i] = (bf16)nd(rng);
    }
    for (int t = 0; t < T; ++t) hw[t] = 0.5f + 0.5f * (float)(rng() % 1000) / 1000.f;

    bf16 *dg = dev_alloc<bf16>(n), *du = dev_alloc<bf16>(n), *dd = dev_alloc<bf16>(n);
    float* dw = dev_alloc<float>(T);
    auto f8 = [&](size_t m) { return dev_alloc<__nv_fp8_e4m3>(m); };
    auto u8 = [&](size_t m) { return dev_alloc<uint8_t>(m); };
    __nv_fp8_e4m3 *op_dg = f8(n), *op_du = f8(n), *rf_dg = f8(n), *rf_du = f8(n);
    uint8_t *op_sfg = u8(groups), *op_sfu = u8(groups), *rf_sfg = u8(groups), *rf_sfu = u8(groups);
    CK(cudaMemcpy(dg, hg.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(du, hu.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dd, hd.data(), n * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), T * 4, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (int)((groups + threads - 1) / threads);
    epilogue_operator_kernel<<<blocks, threads>>>(dg, du, dd, dw, op_dg, op_du, op_sfg, op_sfu, N, groups, 1);
    ref_segment_kernel<<<blocks, threads>>>(dg, du, dd, dw, rf_dg, rf_du, rf_sfg, rf_sfu, N, groups);
    CK(cudaDeviceSynchronize());

    std::vector<__nv_fp8_e4m3> a1(n), a2(n), b1(n), b2(n);
    std::vector<uint8_t> s1(groups), s2(groups), t1(groups), t2(groups);
    CK(cudaMemcpy(a1.data(), op_dg, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(a2.data(), op_du, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b1.data(), rf_dg, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b2.data(), rf_du, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s1.data(), op_sfg, groups, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(s2.data(), op_sfu, groups, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(t1.data(), rf_sfg, groups, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(t2.data(), rf_sfu, groups, cudaMemcpyDeviceToHost));
    int64_t vmis = 0, smis = 0;
    for (int64_t i = 0; i < n; ++i) {
        if (a1[i].__x != b1[i].__x) ++vmis;
        if (a2[i].__x != b2[i].__x) ++vmis;
    }
    for (int64_t i = 0; i < groups; ++i) {
        if (s1[i] != t1[i]) ++smis;
        if (s2[i] != t2[i]) ++smis;
    }
    printf("[B] 提议算子（真实 math.cuh store 分支） vs 手写独立参考段（T=%d,I=%d）\n", T, N);
    printf("    FP8 字节不符 %lld / %lld（dgate+dup 两输出），SF 不符 %lld / %lld\n\n",
           (long long)vmis, (long long)(2 * n), (long long)smis, (long long)(2 * groups));

    // ---- [C] 单输出 vs 双输出：dgate/dup 的 SF 不可互换 ----
    {
        // SF 字节 = UE8M0 指数（bias 127）；inv 用 2^(127-exp)。共享一个 SF 时 dup 的等效比例偏差。
        double ss = 0;
        int64_t cnt = 0;
        for (int64_t g = 0; g < groups; ++g) {
            const int e1 = (int)t1[g], e2 = (int)t2[g];
            const double ratio = std::pow(2.0, (double)(e1 - e2));
            ss += (ratio - 1.0) * (ratio - 1.0);
            ++cnt;
        }
        printf("[C] 单输出共享 SF 的代价：dgate/dup 每 32 元素独立 SF；若强行共用一个 SF，\n");
        printf("    dup 的等效比例偏差 RMS = %.4f（SF 比值 2^(e_dgate-e_dup) 偏离 1 的均方根）。\n",
               std::sqrt(ss / (double)cnt));
        printf("    两输出 SF 分布不同 ⇒ 契约必须给**两个** SFD。\n\n");
    }

    // ---- [C2] 单输出运行时：two_output=0 时 dup 全零（契约表达不了第二输出） ----
    {
        CK(cudaMemset(op_du, 0, n));
        epilogue_operator_kernel<<<blocks, threads>>>(dg, du, dd, dw, op_dg, op_du, op_sfg, op_sfu, N, groups, 0);
        CK(cudaDeviceSynchronize());
        std::vector<__nv_fp8_e4m3> z(n);
        CK(cudaMemcpy(z.data(), op_du, n, cudaMemcpyDeviceToHost));
        int64_t nz = 0; for (int64_t i = 0; i < n; ++i) if (z[i].__x != 0) ++nz;
        printf("[C2] 只走「单输出」分支（two_output=0）：dup 非零元素 %lld / %lld\n", (long long)nz, (long long)n);
        printf("     第二输出**根本不会被写**——不是精度问题，是契约留不出第二个张量。\n\n");
    }

    printf("[done] 架构无关；SM100 专属部分（TMEM/TMA/UTCCP 与 %zu B 的 __grid_constant__ args）见 .sh/.py。\n",
           sizeof(EpilogueOperatorArgs));
    return 0;
}
