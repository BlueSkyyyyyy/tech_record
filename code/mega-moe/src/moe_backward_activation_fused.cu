// P145-1 实验：把 MoE 反向的「SwiGLU 反向 + 激活梯度重量化」段折进 **dgrad 的 epilogue**——
//   省掉 d(act) 的一轮全局往返，多出来的 gate/up 读能不能藏进 GEMM 的算力里？
//
// 背景（承接第 154 篇 P144-1）：
//   第 154 篇把「夹在两个 dgrad GEMM 之间的元素级段」单独量了：它在 T=2048,I=2048 是 20.1us（量化路径），
//   在反向 single-step 预算（d2g 111 + 段 20 + d1g 194 + w2g 500 + w1g 997 = 1823us）里只占 1.10%。
//   它留下的口子（可继续①）：把这段**折进邻近 GEMM 的 epilogue**（对照第 148/149 篇的「同 kernel 融合」），
//   量它能不能藏进 d2g 或 d1g 的 epilogue。
//
// 合同（为什么可以折）：
//   d2g = `m_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:181-250`），输出 D 必 bf16（`:209`）、
//   SM90 走 1D2D、epilogue `SM90_TMA_STORE_2D` 直写 d(act)（`sm90_fp8_gemm_1d2d.cuh:437`）。d(act) 只被
//   紧随其后的 SwiGLU 反向段消费（本文件/第 154 篇），**没有第二个消费者**⇒ 可以不落盘、在 epilogue 内
//   直接反解 gate/up 的梯度、重量化成 FP8（供 wgrad 的 B 操作数 / d1g 的 A 操作数）。这样省掉 d(act) 的
//   「写一遍 bf16 + 读一遍 bf16」= 4 B/元素，只多读 gate/up（本来段里就要读）。
//
// 本文件（架构无关，H100/sm90 真机可跑）：
//   ① `k_seg`      = 第 154 篇的独立段（读 dact+gate+up，写 FP8 对）——「串行流水」的消费核；
//   ② `k_prod_fresh` = 模拟 d2g 的**输出侧**：做 `kIters` 次依赖 FFMA（代表 tensor-core 收缩预算，可调）
//                      产出每元素 dact_val，再 **fresh 写 bf16 d(act)**（= 现实现）；
//   ③ `k_prod_fused` = 同一份算 + **融合 epilogue**：dact_val 不出寄存器，直接读 gate/up、SwiGLU 反向、
//                      量化、写 FP8 对（= 折叠后）。两者算力完全相同、只差 epilogue 的数据流。
//   扫 kIters（0=纯访存 → 大=计算受限）得到「融合额外读暴露多少 / 藏住多少」的曲线；并用 T=2048,I=2048
//   等价形状做**逐位正确性**（fused 输出 == fresh 落盘后独立段输出）。
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

static __host__ __device__ __forceinline__ float bf16_rne(float x) {
    const uint32_t u = f2u(x);
    return u2f((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

// 逐位复刻 `common/math.cuh:114-147` 的 UE8M0 SF 选择（E4M3 档）
static __host__ __device__ __forceinline__ uint32_t ue8m0_sf_exp(float amax) {
    const uint32_t kMantissaMask = (1u << 23) - 1;
    const uint32_t kQuantMaxMantissa = 0x60u << (23 - 7);
    const uint32_t kQuantMaxExponent = 8;
    const uint32_t kMinSFExponent = 105;
    const uint32_t amax_bits = f2u(amax);
    const uint32_t rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> 23;
    return max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}
static __host__ __device__ __forceinline__ float ue8m0_sf_inv(uint32_t sf_exp) {
    return u2f((254u - sf_exp) << 23);
}

static __device__ __forceinline__ float warp_max_f16(float v) {
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 16));
    return v;
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

// ---------------- ① 独立段（第 154 篇的 k_seg）：读 dact+gate+up，写 FP8 对 ----------------
// 一个 quant group(32 元素) = 16 lane × 2 元素。一个 warp 处理 2 组。
__global__ void seg_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                           const bf16* __restrict__ dact, const float* __restrict__ weights,
                           __nv_fp8_e4m3* __restrict__ dg_out, __nv_fp8_e4m3* __restrict__ du_out,
                           uint8_t* __restrict__ dg_sf, uint8_t* __restrict__ du_sf,
                           const int I, const int64_t total_groups) {
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;
    const int l16 = lane & 15;
    const int64_t warp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t total_warps = (int64_t)gridDim.x * (blockDim.x >> 5);
    const int64_t total_pairs = (total_groups + 1) / 2;
    const int groups_per_row = I / 32;
    for (int64_t p = warp; p < total_pairs; p += total_warps) {
        const int64_t g = p * 2 + half;
        if (g >= total_groups) continue;
        const int64_t row = g / groups_per_row;
        const int64_t base = row * I + (g % groups_per_row) * 32 + l16 * 2;
        const float w = weights[row];
        const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gate[base]);
        const __nv_bfloat162 u2 = *reinterpret_cast<const __nv_bfloat162*>(&up[base]);
        const __nv_bfloat162 d2 = *reinterpret_cast<const __nv_bfloat162*>(&dact[base]);
        float dg[2], du[2], amax_g = 0.f, amax_u = 0.f;
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float gf = bf16_rne(j == 0 ? __bfloat162float(g2.x) : __bfloat162float(g2.y));
            const float uf = bf16_rne(j == 0 ? __bfloat162float(u2.x) : __bfloat162float(u2.y));
            const float dd = j == 0 ? __bfloat162float(d2.x) : __bfloat162float(d2.y);
            swiglu_bwd(gf, uf, dd, w, &dg[j], &du[j]);
            amax_g = fmaxf(amax_g, fabsf(dg[j]));
            amax_u = fmaxf(amax_u, fabsf(du[j]));
        }
        const float mg = warp_max_f16(amax_g), mu = warp_max_f16(amax_u);
        const uint32_t eg = ue8m0_sf_exp(mg), eu = ue8m0_sf_exp(mu);
        const float ig = ue8m0_sf_inv(eg), iu = ue8m0_sf_inv(eu);
        dg_out[base + 0] = __nv_fp8_e4m3(dg[0] * ig);
        dg_out[base + 1] = __nv_fp8_e4m3(dg[1] * ig);
        du_out[base + 0] = __nv_fp8_e4m3(du[0] * iu);
        du_out[base + 1] = __nv_fp8_e4m3(du[1] * iu);
        if (l16 == 0) { dg_sf[g] = (uint8_t)eg; du_sf[g] = (uint8_t)eu; }
    }
}

// ---------------- ②/③ d2g 输出侧原型：算力（kIters 依赖 FFMA）+ 可切换 epilogue ----------------
// kMode: 0 = FRESH（写 bf16 d(act)，= 现实现）；1 = FUSED（就地 SwiGLU 反向 + 量化，写 FP8 对）。
// 每元素先算出一个 dact_val（代表 GEMM 累加器的输出），再走 epilogue。kIters=0 ⇒ 纯访存下界。
template <int kMode, int kIters>
__global__ void prod_kernel(const bf16* __restrict__ gate, const bf16* __restrict__ up,
                            const float* __restrict__ weights,
                            bf16* __restrict__ dact_out,                 // kMode==0
                            __nv_fp8_e4m3* __restrict__ dg_out,          // kMode==1
                            __nv_fp8_e4m3* __restrict__ du_out,
                            uint8_t* __restrict__ dg_sf, uint8_t* __restrict__ du_sf,
                            const int I, const int64_t total_groups) {
    const int lane = threadIdx.x & 31;
    const int half = lane >> 4;
    const int l16 = lane & 15;
    const int64_t warp = (int64_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int64_t total_warps = (int64_t)gridDim.x * (blockDim.x >> 5);
    const int64_t total_pairs = (total_groups + 1) / 2;
    const int groups_per_row = I / 32;
    for (int64_t p = warp; p < total_pairs; p += total_warps) {
        const int64_t g = p * 2 + half;
        if (g >= total_groups) continue;
        const int64_t row = g / groups_per_row;
        const int64_t base = row * I + (g % groups_per_row) * 32 + l16 * 2;
        const float w = weights[row];

        // -- 算力阶段：每元素 kIters 次依赖 FFMA（种子来自地址，兼防常量折叠） --
        float dd[2];
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            float x = (float)(base + j) * 1.7e-6f + 0.37f;
            #pragma unroll
            for (int it = 0; it < kIters; ++it) x = fmaf(x, 1.0000011f, 1.3e-4f);
            dd[j] = bf16_rne(x);   // 模拟 GEMM 输出 d(act)（bf16 落盘精度）
        }

        if constexpr (kMode == 0) {
            // FRESH：把 d(act) 写回全局（2 B/元素）
            *reinterpret_cast<__nv_bfloat162*>(&dact_out[base]) = __floats2bfloat162_rn(dd[0], dd[1]);
        } else {
            // FUSED：dact_val 不出寄存器，读 gate/up，反解 + 量化，写 FP8 对
            const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(&gate[base]);
            const __nv_bfloat162 u2 = *reinterpret_cast<const __nv_bfloat162*>(&up[base]);
            float dg[2], du[2], amax_g = 0.f, amax_u = 0.f;
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const float gf = bf16_rne(j == 0 ? __bfloat162float(g2.x) : __bfloat162float(g2.y));
                const float uf = bf16_rne(j == 0 ? __bfloat162float(u2.x) : __bfloat162float(u2.y));
                swiglu_bwd(gf, uf, dd[j], w, &dg[j], &du[j]);
                amax_g = fmaxf(amax_g, fabsf(dg[j]));
                amax_u = fmaxf(amax_u, fabsf(du[j]));
            }
            const float mg = warp_max_f16(amax_g), mu = warp_max_f16(amax_u);
            const uint32_t eg = ue8m0_sf_exp(mg), eu = ue8m0_sf_exp(mu);
            const float ig = ue8m0_sf_inv(eg), iu = ue8m0_sf_inv(eu);
            dg_out[base + 0] = __nv_fp8_e4m3(dg[0] * ig);
            dg_out[base + 1] = __nv_fp8_e4m3(dg[1] * ig);
            du_out[base + 0] = __nv_fp8_e4m3(du[0] * iu);
            du_out[base + 1] = __nv_fp8_e4m3(du[1] * iu);
            if (l16 == 0) { dg_sf[g] = (uint8_t)eg; du_sf[g] = (uint8_t)eu; }
        }
    }
}

template <typename T>
static T* dev_alloc(size_t n) { T* p; CK(cudaMalloc(&p, n * sizeof(T))); return p; }

template <typename F>
static double bench(F f, int iters) {
    for (int i = 0; i < 3; ++i) f();
    cudaEvent_t a, b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    CK(cudaEventRecord(a));
    for (int i = 0; i < iters; ++i) f();
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

    const int T = 4096, I = 2048;             // = d2g 的输出形状 [Σt = T·topk = 4096, I = 2048]
    const int64_t n = (int64_t)T * I;
    const int64_t total_groups = n / 32;
    const int threads = 256;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.5f);
    std::vector<bf16> hg(n), hu(n);
    std::vector<float> hw(T);
    for (int64_t i = 0; i < n; ++i) { hg[i] = (bf16)nd(rng); hu[i] = (bf16)nd(rng); }
    for (int t = 0; t < T; ++t) hw[t] = 0.5f + 0.5f * (float)(rng() % 1000) / 1000.f;

    bf16* dgate = dev_alloc<bf16>(n); bf16* dup = dev_alloc<bf16>(n);
    float* dw = dev_alloc<float>(T);
    bf16* dact = dev_alloc<bf16>(n);
    __nv_fp8_e4m3* dg8a = dev_alloc<__nv_fp8_e4m3>(n);
    __nv_fp8_e4m3* du8a = dev_alloc<__nv_fp8_e4m3>(n);
    __nv_fp8_e4m3* dg8b = dev_alloc<__nv_fp8_e4m3>(n);
    __nv_fp8_e4m3* du8b = dev_alloc<__nv_fp8_e4m3>(n);
    uint8_t* sfga = dev_alloc<uint8_t>(total_groups); uint8_t* sfua = dev_alloc<uint8_t>(total_groups);
    uint8_t* sfgb = dev_alloc<uint8_t>(total_groups); uint8_t* sfub = dev_alloc<uint8_t>(total_groups);
    CK(cudaMemcpy(dgate, hg.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dup, hu.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, hw.data(), T * sizeof(float), cudaMemcpyHostToDevice));

    // ---- [A] 正确性：FRESH 落盘 d(act) → 独立段；FUSED 直接出；两者应逐位相等 ----
    {
        const int blocks = (int)((total_groups / 32 + threads / 32 - 1) / (threads / 32));
        prod_kernel<0, 64><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups);
        seg_kernel<<<blocks, threads>>>(dgate, dup, dact, dw, dg8a, du8a, sfga, sfua, I, total_groups);
        prod_kernel<1, 64><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups);
        std::vector<__nv_fp8_e4m3> a1(n), a2(n), b1(n), b2(n);
        std::vector<uint8_t> s1(total_groups), s2(total_groups), s3(total_groups), s4(total_groups);
        CK(cudaMemcpy(a1.data(), dg8a, n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(a2.data(), du8a, n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b1.data(), dg8b, n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b2.data(), du8b, n, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s1.data(), sfga, total_groups, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s2.data(), sfua, total_groups, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s3.data(), sfgb, total_groups, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(s4.data(), sfub, total_groups, cudaMemcpyDeviceToHost));
        int64_t vmis = 0, smis = 0;
        for (int64_t i = 0; i < n; ++i) { if (a1[i].__x != b1[i].__x) ++vmis; if (a2[i].__x != b2[i].__x) ++vmis; }
        for (int64_t i = 0; i < total_groups; ++i) { if (s1[i] != s3[i]) ++smis; if (s2[i] != s4[i]) ++smis; }
        printf("[A] 正确性（T=%d,I=%d, d2g 输出形状）: FUSED vs (FRESH→独立段) —— FP8 字节不符 %lld / %lld，SF 不符 %lld / %lld\n",
               T, I, (long long)vmis, (long long)(2 * n), (long long)smis, (long long)(2 * total_groups));
    }

    // ---- [B] 扫算力：FRESH / FUSED（同算力）与独立段；看融合的额外读藏不藏 ----
    printf("\n[B] d2g 输出侧原型（T=%d,I=%d = d2g 输出形状；同一份算力，只差 epilogue）\n", T, I);
    printf("    字节：FRESH 写 2B；FUSED 读 gate+up 4B + 写 2B；独立段 读 dact+gate+up 6B + 写 2B\n");
    printf("    %-8s %-11s %-11s %-11s %-11s %-11s %-10s %-11s\n",
           "kIters", "fresh_us", "fused_us", "fused-fresh", "seg_us", "SEQ_us", "SEQ/FUSED", "saving_us");
    const double bw_peak = 3.35e12;
    int ks[] = {0, 32, 96, 192, 384, 768};
    for (int kiter : ks) {
        const int blocks = (int)((total_groups / 32 + threads / 32 - 1) / (threads / 32));
        double t_fresh, t_fused, t_seg;
        switch (kiter) {
            case 0:   t_fresh = bench([&]{ prod_kernel<0, 0><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
            case 32:  t_fresh = bench([&]{ prod_kernel<0, 32><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
            case 96:  t_fresh = bench([&]{ prod_kernel<0, 96><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
            case 192: t_fresh = bench([&]{ prod_kernel<0, 192><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
            case 384: t_fresh = bench([&]{ prod_kernel<0, 384><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
            case 768: t_fresh = bench([&]{ prod_kernel<0, 768><<<blocks, threads>>>(dgate, dup, dw, dact, nullptr, nullptr, nullptr, nullptr, I, total_groups); }, 50); break;
        }
        switch (kiter) {
            case 0:   t_fused = bench([&]{ prod_kernel<1, 0><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
            case 32:  t_fused = bench([&]{ prod_kernel<1, 32><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
            case 96:  t_fused = bench([&]{ prod_kernel<1, 96><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
            case 192: t_fused = bench([&]{ prod_kernel<1, 192><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
            case 384: t_fused = bench([&]{ prod_kernel<1, 384><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
            case 768: t_fused = bench([&]{ prod_kernel<1, 768><<<blocks, threads>>>(dgate, dup, dw, nullptr, dg8b, du8b, sfgb, sfub, I, total_groups); }, 50); break;
        }
        t_seg = bench([&]{ seg_kernel<<<blocks, threads>>>(dgate, dup, dact, dw, dg8a, du8a, sfga, sfua, I, total_groups); }, 50);
        const double seq = t_fresh + t_seg;
        const double extra = t_fused - t_fresh;     // 融合 epilogue 相对「纯写 d(act)」的边际成本
        const double saving = t_seg - extra;         // 折叠相对独立段的净节省
        const double t_read_gu_lb = (double)n * 4.0 / bw_peak * 1e6;  // 纯读 gate+up 的 HBM 下界(us)
        printf("    %-8d %-11.2f %-11.2f %-11.2f %-11.2f %-11.2f %-10.3f %-11.2f\n",
               kiter, t_fresh, t_fused, extra, t_seg, seq, seq / t_fused, saving);
        (void)t_read_gu_lb;
    }
    printf("    注：kIters=0 是纯访存下界；kIters 越大越计算受限。SEQ = FRESH + 独立段（现实现的串行流水）。\n");
    printf("    fused-fresh = 融合 epilogue 的边际成本（含读 gate/up + SwiGLU 反向 + 半 warp 归约 + 写 FP8）；\n");
    printf("    saving = seg - (fused-fresh) = 折叠相对两 kernel 串行流水的净节省（≈ d(act) 往返 - 内联开销）。\n");

    printf("\n[done] 架构无关；d2g/d1g 真实 kernel 与 SM100 epilogue_class 机制见 .sh/.py。\n");
    return 0;
}
