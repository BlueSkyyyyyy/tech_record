// P144-1 实验：MoE 反向「非 GEMM 段」——SwiGLU 反向 + L1 激活梯度的 FP8/UE8M0 重量化。
//
// 背景（为什么这个段一直缺席）：
//   第 142/143 篇把 **前向** single-step 的预算补全（GEMM 339.4us + L2 写回/combine 292us），
//   第 15/135~151 篇把 **反向** 的四个 grouped GEMM（d2g/d1g/w2g/w1g）与它们的累积协议挖到底，
//   第 104 篇补了反向的两段通信。但**夹在两个 dgrad GEMM 之间的那段元素级算子**——把 L2 dgrad
//   出来的 `d(act)` 反解成 gate/up 的激活梯度、再量化成 FP8（供 wgrad 的 B 操作数）——从没被
//   单独量过。它就是前向 L1 epilogue（`sm100_fp8_fp4_mega_moe.cuh:1108-1219`：in-place SwiGLU +
//   top-k 权重 + UE8M0 量化，第 10/142 篇已拆）的**逐元素逆**。本文件给它的下界内核与真实带宽。
//
// 前向（kernel `:1108-1135`，语义钉死）：
//   bg = bf16_rne(g); bu = bf16_rne(u);            // 先入 bf16（`__float22bfloat162_rn`）
//   gf = float(bg);  uf = float(bu);
//   sig = 1/(1+exp(-gf));                          // 前向用 __expf + fast_rcp（`:1125-1132`）
//   act = sig*gf*uf*w;                             // silu(gf)*uf*w（`:1122-1134`，w=top-k 权重）
//   再对 32 个连续中间维元素取 amax，算 UE8M0 SF，`__nv_fp8x4_e4m3(act*sf_inv)`（`:1137-1181`）。
//
// 反向（本文件，chain rule；bf16 round 走 straight-through）：
//   d_su = dact*w
//   dg   = d_su * uf * silu'(gf) = d_su * uf * sig*(1 + gf*(1-sig))
//   du   = d_su * silu(gf)       = d_su * sig*gf
//   dg/du 各自按 32 一组 UE8M0 量化成 FP8 E4M3。
//
// 全部架构无关（只有 bf16 ld/st、fp32 浮点与 `__nv_fp8_e4m3`），H100(sm90) 真机可跑。
// 编译/运行见 harness/exp_moe_backward_activation.sh；也可单跑做自带 [A]/[B]/[C] 报告。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

using bf16 = __nv_bfloat16;

#define CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static __host__ __device__ __forceinline__ uint32_t f2u(float x) { uint32_t u; memcpy(&u, &x, 4); return u; }
static __host__ __device__ __forceinline__ float u2f(uint32_t u) { float x; memcpy(&x, &u, 4); return x; }

// bf16 round-to-nearest-even，host/device 通用（避免依赖构造函数在 host 的可否）
static __host__ __device__ __forceinline__ float bf16_rne(float x) {
    const uint32_t u = f2u(x);
    return u2f((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// --- 逐位复刻 `common/math.cuh:114-147` 的 UE8M0 SF 选择（E4M3 档） ---
static __host__ __device__ __forceinline__ uint32_t ue8m0_sf_exp(float amax) {
    const uint32_t kMantissaMask = (1u << 23) - 1;
    const uint32_t kQuantMaxMantissa = 0x60u << (23 - 7);  // mantissa(1.75)
    const uint32_t kQuantMaxExponent = 8;                  // 448 = 1.75 * 2^8
    const uint32_t kMinSFExponent = 105;                   // max(amax,1e-4) -> 2^-22
    const uint32_t amax_bits = f2u(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> 23;
    return max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}
static __host__ __device__ __forceinline__ float ue8m0_sf_inv(uint32_t sf_exp) {
    return u2f((254u - sf_exp) << 23);
}

// 参考实现（host，double）：以 bf16 舍入后的 gf/uf 为数据路径（镜像前向 kernel）
static void ref_backward_element(double g, double u, double dact, double w, bool use_bf16_round,
                                 double* dg, double* du) {
    double gf = g, uf = u;
    if (use_bf16_round) { gf = (double)bf16_rne((float)g); uf = (double)bf16_rne((float)u); }
    const double sig = 1.0 / (1.0 + std::exp(-gf));
    const double d_su = dact * w;
    *dg = d_su * uf * sig * (1.0 + gf * (1.0 - sig));
    *du = d_su * sig * gf;
}

// 半 warp（16 lane）内最大值归约，width=16。
static __device__ __forceinline__ float warp_max_f16(float v) {
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 16));
    return v;
}

// ---------------- 设备内核：SwiGLU 反向 + 逐 32 组 UE8M0 FP8 量化 ----------------
// 一个 quant group(32 元素) = 16 lane × 2 元素（bf16x2 向量化）。一个 warp 处理 2 组。
// grid-stride over group-pairs；组内同一行 ⇒ 同一 top-k 权重。
template <bool kQuant, int kGroup>
__global__ void swiglu_backward_kernel(
    const bf16* __restrict__ gate, const bf16* __restrict__ up,
    const bf16* __restrict__ dact, const float* __restrict__ weights,   // [T]
    void* __restrict__ dg_out, void* __restrict__ du_out,
    uint8_t* __restrict__ dg_sf, uint8_t* __restrict__ du_sf,
    const int I, const int64_t total_groups) {
    static_assert(kGroup == 32, "one quant group per half warp");
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;              // 0/1：warp 内两组之一
    const int l16 = lane & 15;               // 组内 lane（处理 2 个元素）
    const int64_t warp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t total_warps = (int64_t)gridDim.x * (blockDim.x >> 5);
    const int64_t total_pairs = (total_groups + 1) / 2;
    const int groups_per_row = I / kGroup;
    __nv_fp8_e4m3* dg8 = (__nv_fp8_e4m3*)dg_out;
    __nv_fp8_e4m3* du8 = (__nv_fp8_e4m3*)du_out;
    bf16* dgo = (bf16*)dg_out;
    bf16* duo = (bf16*)du_out;

    for (int64_t p = warp; p < total_pairs; p += total_warps) {
        const int64_t g = p * 2 + half;
        if (g >= total_groups) continue;
        const int64_t row = g / groups_per_row;
        const int64_t base = row * I + (g % groups_per_row) * kGroup + l16 * 2;
        const float* gw = &weights[row];
        const float w = gw[0];
        const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gate[base]);
        const __nv_bfloat162 u2 = *reinterpret_cast<const __nv_bfloat162*>(&up[base]);
        const __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dact[base]);
        float dg[2], du[2];
        float amax_g = 0.f, amax_u = 0.f;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float gf = bf16_rne(j == 0 ? __bfloat162float(g2.x) : __bfloat162float(g2.y));
            const float uf = bf16_rne(j == 0 ? __bfloat162float(u2.x) : __bfloat162float(u2.y));
            const float dd = j == 0 ? __bfloat162float(d2.x) : __bfloat162float(d2.y);
            const float sigma = 1.0f / (1.0f + __expf(-gf));   // 前向 fast-math（__expf + rcp）
            const float silu = gf * sigma;
            const float d_su = dd * w;
            dg[j] = d_su * uf * sigma * (1.0f + gf * (1.0f - sigma));
            du[j] = d_su * silu;
            amax_g = fmaxf(amax_g, fabsf(dg[j]));
            amax_u = fmaxf(amax_u, fabsf(du[j]));
        }
        const float mg = warp_max_f16(amax_g), mu = warp_max_f16(amax_u);
        if constexpr (kQuant) {
            const uint32_t eg = ue8m0_sf_exp(mg), eu = ue8m0_sf_exp(mu);
            const float ig = ue8m0_sf_inv(eg), iu = ue8m0_sf_inv(eu);
            dg8[base + 0] = __nv_fp8_e4m3(dg[0] * ig);
            dg8[base + 1] = __nv_fp8_e4m3(dg[1] * ig);
            du8[base + 0] = __nv_fp8_e4m3(du[0] * iu);
            du8[base + 1] = __nv_fp8_e4m3(du[1] * iu);
            if (l16 == 0) { dg_sf[g] = (uint8_t)eg; du_sf[g] = (uint8_t)eu; }
        } else {
            *reinterpret_cast<__nv_bfloat162*>(&dgo[base]) = __floats2bfloat162_rn(dg[0], dg[1]);
            *reinterpret_cast<__nv_bfloat162*>(&duo[base]) = __floats2bfloat162_rn(du[0], du[1]);
        }
    }
}

template <typename T>
static T* dev_alloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }

static double time_quant(const bf16* g, const bf16* u, const bf16* d, const float* w,
                         void* dg, void* du, uint8_t* dgsf, uint8_t* dusf,
                         int T, int I, int iters, bool quant) {
    const int64_t total_groups = (int64_t)T * (I / 32);
    const int threads = 256;                                    // 8 warps/block
    const int64_t blocks64 = (total_groups / 32 + threads / 32 - 1) / (threads / 32);
    const int blocks = (int)blocks64;
    for (int i = 0; i < 3; ++i)
        if (quant) swiglu_backward_kernel<true, 32><<<blocks, threads>>>(g, u, d, w, dg, du, dgsf, dusf, I, total_groups);
        else       swiglu_backward_kernel<false, 32><<<blocks, threads>>>(g, u, d, w, dg, du, nullptr, nullptr, I, total_groups);
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i)
        if (quant) swiglu_backward_kernel<true, 32><<<blocks, threads>>>(g, u, d, w, dg, du, dgsf, dusf, I, total_groups);
        else       swiglu_backward_kernel<false, 32><<<blocks, threads>>>(g, u, d, w, dg, du, nullptr, nullptr, I, total_groups);
    CK(cudaEventRecord(b)); CK(cudaEventSynchronize(b));
    float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b));
    CK(cudaEventDestroy(a)); CK(cudaEventDestroy(b));
    return ms / iters * 1e3;
}

int main() {
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, dev));
    printf("[env] device=%s sm_%d%d smem/SM=%zu KiB\n", prop.name, prop.major, prop.minor,
           prop.sharedMemPerMultiprocessor / 1024);

    // ---- [A] 正确性：kernel 输出 vs host 参考（逐位 SF、FP8 字节） ----
    {
        const int T = 512, I = 1024, group = 32;
        const int n = T * I, num_groups = T * (I / group);
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.f, 1.5f);
        std::vector<bf16> hg(n), hu(n), hd(n);
        std::vector<float> hw(T);
        for (int i = 0; i < n; ++i) { hg[i] = (bf16)nd(rng); hu[i] = (bf16)nd(rng); hd[i] = (bf16)nd(rng); }
        for (int t = 0; t < T; ++t) hw[t] = 0.5f + 0.5f * (float)(rng() % 1000) / 1000.f;

        bf16 *dhg = dev_alloc<bf16>(n), *dhu = dev_alloc<bf16>(n), *dhd = dev_alloc<bf16>(n);
        float* dw = dev_alloc<float>(T);
        __nv_fp8_e4m3* dg8 = dev_alloc<__nv_fp8_e4m3>(n);
        __nv_fp8_e4m3* du8 = dev_alloc<__nv_fp8_e4m3>(n);
        uint8_t* dgsf = dev_alloc<uint8_t>(num_groups); uint8_t* dusf = dev_alloc<uint8_t>(num_groups);
        CK(cudaMemcpy(dhg, hg.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dhu, hu.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dhd, hd.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw, hw.data(), T * sizeof(float), cudaMemcpyHostToDevice));
        const int threads = 256, blocks = (num_groups / 32 + threads / 32 - 1) / (threads / 32);
        swiglu_backward_kernel<true, 32><<<blocks, threads>>>(dhg, dhu, dhd, dw, dg8, du8, dgsf, dusf, I, (int64_t)num_groups);

        std::vector<__nv_fp8_e4m3> og(n), ou(n);
        std::vector<uint8_t> ogsf(num_groups), ousf(num_groups);
        CK(cudaMemcpy(og.data(), dg8, n * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ou.data(), du8, n * sizeof(__nv_fp8_e4m3), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ogsf.data(), dgsf, num_groups, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ousf.data(), dusf, num_groups, cudaMemcpyDeviceToHost));

        int sf_mismatch = 0, val_mismatch = 0, checked = 0;
        for (int t = 0; t < T; ++t) {
            for (int gg = 0; gg < I / group; ++gg) {
                const int64_t base = (int64_t)t * I + gg * group;
                const int64_t gidx = (int64_t)t * (I / group) + gg;
                double amax_g = 0, amax_u = 0;
                std::vector<double> dgv(group), duv(group);
                for (int k = 0; k < group; ++k) {
                    ref_backward_element((float)hg[base + k], (float)hu[base + k], (float)hd[base + k], hw[t], true, &dgv[k], &duv[k]);
                    amax_g = std::max(amax_g, fabs(dgv[k])); amax_u = std::max(amax_u, fabs(duv[k]));
                }
                const uint32_t eg = ue8m0_sf_exp((float)amax_g), eu = ue8m0_sf_exp((float)amax_u);
                if (ogsf[gidx] != (uint8_t)eg) ++sf_mismatch;
                if (ousf[gidx] != (uint8_t)eu) ++sf_mismatch;
                const float ig = ue8m0_sf_inv(eg), iu = ue8m0_sf_inv(eu);
                for (int k = 0; k < group; ++k) {
                    const __nv_fp8_e4m3 rg = __nv_fp8_e4m3((float)(dgv[k] * ig));
                    const __nv_fp8_e4m3 ru = __nv_fp8_e4m3((float)(duv[k] * iu));
                    if (rg.__x != og[base + k].__x) ++val_mismatch;
                    if (ru.__x != ou[base + k].__x) ++val_mismatch;
                    ++checked;
                }
            }
        }
        printf("[A] 正确性（T=%d,I=%d, %d 组）: SF 逐位不符 %d / %d，FP8 字节不符 %d / %d\n",
               T, I, num_groups, sf_mismatch, 2 * num_groups, val_mismatch, 2 * checked);
    }

    // ---- [B] 前向 bf16 round 传给反向的 straight-through 误差（数据路径镜像 vs 精确） ----
    {
        const int n = 1 << 20;
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.f, 1.5f);
        double e_bf = 0, den = 0;
        for (int i = 0; i < n; ++i) {
            const float g = nd(rng), u = nd(rng), d = nd(rng), w = 0.7f;
            double dg_ex, du_ex, dg_bf, du_bf;
            ref_backward_element(g, u, d, w, false, &dg_ex, &du_ex);
            ref_backward_element(g, u, d, w, true, &dg_bf, &du_bf);
            e_bf = std::max(e_bf, std::max(fabs(dg_bf - dg_ex), fabs(du_bf - du_ex)));
            den = std::max(den, std::max(fabs(dg_ex), fabs(du_ex)));
        }
        printf("[B] 前向 bf16 round 的 straight-through 误差: max|Δ|=%.3e（相对 %.3e）\n", e_bf, e_bf / den);
    }

    // ---- [C] 计时：量化路径 vs bf16 路径，扫形状 ----
    {
        const int shapes[][2] = {{2048, 2048}, {8192, 2048}, {2048, 7168}, {4096, 4096}, {512, 512}};
        printf("[C] SwiGLU 反向 + 量化 带宽（空闲 GPU，热 cache，迭代平均）\n");
        printf("    %-12s %-10s %-12s %-10s %-12s %-12s\n", "T,I", "quant_us", "quant_GB/s", "bf16_us", "bf16_GB/s", "q_MB");
        for (auto& s : shapes) {
            const int T = s[0], I = s[1], n = T * I;
            bf16 *g = dev_alloc<bf16>(n), *u = dev_alloc<bf16>(n), *d = dev_alloc<bf16>(n); float* w = dev_alloc<float>(T);
            __nv_fp8_e4m3* dg8 = dev_alloc<__nv_fp8_e4m3>(n); __nv_fp8_e4m3* du8 = dev_alloc<__nv_fp8_e4m3>(n);
            bf16* dgb = dev_alloc<bf16>(n); bf16* dub = dev_alloc<bf16>(n);
            uint8_t* sfg = dev_alloc<uint8_t>(T * (I / 32)); uint8_t* sfu = dev_alloc<uint8_t>(T * (I / 32));
            const int iters = 200;
            const double tq = time_quant(g, u, d, w, dg8, du8, sfg, sfu, T, I, iters, true);
            const double tb = time_quant(g, u, d, w, dgb, dub, nullptr, nullptr, T, I, iters, false);
            const double mb_q = (double)n * 8.0 / 1e6, mb_b = (double)n * 10.0 / 1e6;
            printf("    %-5d,%-6d %-10.2f %-12.1f %-10.2f %-12.1f %-12.1f\n",
                   T, I, tq, mb_q / (tq * 1e-6) / 1e3, tb, mb_b / (tb * 1e-6) / 1e3, mb_q);
            CK(cudaFree(g)); CK(cudaFree(u)); CK(cudaFree(d)); CK(cudaFree(w));
            CK(cudaFree(dg8)); CK(cudaFree(du8)); CK(cudaFree(dgb)); CK(cudaFree(dub));
            CK(cudaFree(sfg)); CK(cudaFree(sfu));
        }
    }

    printf("[done] 架构无关；Mega MoE 融合 kernel（UMMA/tcgen05/SF）SM100 专属、仅分析。\n");
    return 0;
}
