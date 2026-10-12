// P5-2 实验：Mega mHC 的**架构无关数值语义**在 H100(sm90) 上的忠实移植与对拍。
//
// Mega mHC 主体（csrc/apis/mega_mhc.hpp、deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh、
// epilogue/sm100_mega_mhc.cuh、scheduler/mega_mhc.cuh）是 SM100 专属（`mega_mhc.hpp:223` 硬断言
// arch_major==10，TF32 UMMA / TMEM / tcgen05 均不可用），H100 端到端跑不了。但它的**数学语义**
// 与几处**位级布局**是架构无关的。本文件把这些源码逐行搬到裸 CUDA，并用独立的 double 参考对拍：
//
//   A. HC 残差混合更新 + RMSNorm（run_normal_worker / Post worker）
//      - new_residual = bf16(post_mix * x + Σ_i comb[i,j] * residual[i])   (epilogue:481-501)
//      - hc_sqr_sum   = Σ (new_residual_bf16)^2                            (epilogue:501)
//      - X1           = Σ_j pre_mix_j * new_residual_bf16                  (epilogue:196-210)
//      - y_bf16       = bf16(X1_bf16 * rsqrt(mean(X1_bf16^2)+eps) * w * scale) (epilogue:213-244)
//   B. Mix 系数 + Sinkhorn（run_mix_task，epilogue:16-101）
//      - affine = hc_output * rsqrt(hc_sqr_sum/(hc*H)+eps) * scales + bases
//      - pre = sigmoid(affine[:hc]) + hc_pre_eps ; post = sigmoid(affine[hc:2hc]) * hc_post_scale
//      - comb = Sinkhorn(softmax(affine[2hc:].view(hc,hc)))（warp 级、逐行/逐列 shfl 归约）
//   C. 共享专家 SF 的分页布局 get_shared_sf_row_idx (epilogue:117-122)
//   D. FP8 UE8M0 scale 的 4-lane 打包 store_fp8_and_pack_sf (epilogue:124-136)
//
// 参考实现全部用 double 在同一文件内实现（独立于 kernel），类似 P5-1 的 mega_gate_routing.cu。
// 编译/运行见 harness/exp_mega_mhc_semantics.sh。
//
// 目标卡代号敏感：文中一律用「目标卡」指代。

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <random>
#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define DEV __device__ __forceinline__

// ---------------------------------------------------------------------------
// 源码移植：deep_gemm/common/math.cuh
// ---------------------------------------------------------------------------
DEV float fast_rcp(const float& x) {
    float ret;
    asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
    return ret;
}

// math.cuh:89-103 get_packed_bf16_amax（树形 HMNMX2）；这里只做 4 元素（lane 的 8 hidden）归约
DEV float bf16_round_d(float f) { return __bfloat162float(__float2bfloat16(f)); }

DEV nv_bfloat16 get_packed_bf16_amax4(const nv_bfloat162* v) {
    nv_bfloat162 tree[2] = {__hmax2(__habs2(v[0]), __habs2(v[2])),
                            __hmax2(__habs2(v[1]), __habs2(v[3]))};
    const nv_bfloat162 m = __hmax2(tree[0], tree[1]);
    return __hmax(m.x, m.y);
}

// math.cuh:106-110 把两对 bf16 乘幂次 sf_inv 后打包成 4 个 E4M3
DEV uint32_t scale_bf16x2_into_fp8x4(const nv_bfloat162& lower, const nv_bfloat162& upper,
                                     const nv_bfloat162& inv_lower, const nv_bfloat162& inv_upper) {
    return __nv_fp8x4_e4m3(__hmul2(lower, inv_lower), __hmul2(upper, inv_upper)).__x;
}

// math.cuh:115-137 get_ue8m0_sf_exp（careful 逐位实现，E4M3）
DEV uint32_t get_ue8m0_sf_exp(const nv_bfloat16& amax) {
    constexpr uint32_t kMantissaBits = 7;
    constexpr uint32_t kMantissaMask = (1u << kMantissaBits) - 1;
    constexpr uint32_t kQuantMaxMantissa = 0x60u;   // mantissa(1.75)
    constexpr uint32_t kQuantMaxExponent = 8;       // 448 = 1.75 * 2^8
    constexpr uint32_t kMinSFExponent = 105;        // max(amax, 1e-4)
    const uint32_t amax_bits = __bfloat16_as_ushort(amax);
    const auto rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits;
    return max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
}

// math.cuh:140-147 get_ue8m0_sf_inv<bf16>：reciprocal biased exponent 求和为 254
DEV nv_bfloat16 get_ue8m0_sf_inv_bf16(const uint32_t& sf_exp) {
    return __ushort_as_bfloat16((uint16_t)((254u - sf_exp) << 7));
}

// ---------------------------------------------------------------------------
// 源码移植：epilogue/sm100_mega_mhc.cuh
// ---------------------------------------------------------------------------
DEV unsigned get_lane_idx() { unsigned l; asm("mov.u32 %0, %%laneid;" : "=r"(l)); return l; }

// math.cuh:203-220 warp_reduce<4>：组内 4-lane 归约（shfl_xor 1,2）
template <typename T>
DEV T warp_reduce_sum4(T v) {
    v += __shfl_xor_sync(0xffffffffu, v, 1);
    v += __shfl_xor_sync(0xffffffffu, v, 2);
    return v;
}
template <typename T>
DEV T warp_reduce_max4(T v) {
    v = max(v, __shfl_xor_sync(0xffffffffu, v, 1));
    v = max(v, __shfl_xor_sync(0xffffffffu, v, 2));
    return v;
}

// epilogue:112-114
DEV float2 fmul2_rn(const float2& a, const float2& b) {
    return make_float2(__fmul_rn(a.x, b.x), __fmul_rn(a.y, b.y));   // sm90 无 __fmul2_rn
}
DEV nv_bfloat162 normalize_bf16_pair(const nv_bfloat162& input, const nv_bfloat162& weight, const float& scale) {
    return __float22bfloat162_rn(fmul2_rn(fmul2_rn(__bfloat1622float2(input), {scale, scale}),
                                          __bfloat1622float2(weight)));
}

// epilogue:117-122 共享专家 SF 分页：每个 128 行页内交错 4 个 32 行组
template <uint32_t SF_BLOCK_M>
DEV uint32_t get_shared_sf_row_idx(const uint32_t& token_idx) {
    const uint32_t idx = token_idx % SF_BLOCK_M;
    return token_idx / SF_BLOCK_M * ((SF_BLOCK_M + 127u) & ~127u) +
           (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
}

// epilogue:103-109
static constexpr uint32_t kHiddenPerNormPack = 256;
static constexpr uint32_t kHiddenPerNormLane = kHiddenPerNormPack / 32;   // 8
static constexpr uint32_t kHiddenPerSF = 32;
static constexpr uint32_t kNumLanesPerSF = kHiddenPerSF / kHiddenPerNormLane;   // 4
static constexpr uint32_t kNumLanesPerSFWord = 128 / kHiddenPerNormLane;        // 16

// epilogue:124-136 把 lane 的 8 个 bf16 量化成 FP8 并打包 SF（SF 部分）
DEV uint32_t pack_sf_word(const uint32_t& sf_exp) {
    const uint32_t paired_sf = __byte_perm(sf_exp, __shfl_xor_sync(0xffffffffu, sf_exp, kNumLanesPerSF), 0x1140);
    return __byte_perm(paired_sf, __shfl_xor_sync(0xffffffffu, paired_sf, 2 * kNumLanesPerSF), 0x5410);
}

// epilogue:16-101 run_mix_task 的核心（lane<24 的 warp 级 affine/sigmoid/sinkhorn）
// 返回值：new_prev_mix / new_post_mix / new_comb_res_mix。lane 布局 [Pre hc | Post hc | Comb hc*hc | inactive]。
template <uint32_t kNumRoutes>
DEV void run_mix_step(const float& hc_output, const float& hc_rms_scale,
                      const float* __restrict__ scales, const float* __restrict__ bases,
                      const float& hc_pre_eps, const float& hc_post_scale,
                      const float& sinkhorn_eps, const uint32_t num_sinkhorn_iters,
                      const uint32_t lane_idx,
                      float* __restrict__ new_prev_mix, float* __restrict__ new_post_mix,
                      float* __restrict__ new_comb_res_mix, const uint32_t token_idx) {
    constexpr uint32_t kNumHCOutputs = kNumRoutes * (kNumRoutes + 2);
    constexpr uint32_t kFirstCombLane = 2 * kNumRoutes;
    const bool is_hc_output_lane = lane_idx < kNumHCOutputs;
    const bool is_pre_lane = lane_idx < kNumRoutes;
    const bool is_pre_or_post_lane = lane_idx < kFirstCombLane;
    const bool is_comb_lane = is_hc_output_lane and not is_pre_or_post_lane;

    float mix_value = 0.0f;
    if (is_hc_output_lane) {
        const uint32_t mix_scale_idx = is_pre_lane ? 0u : is_pre_or_post_lane ? 1u : 2u;
        const float affine_value = hc_output * hc_rms_scale * scales[mix_scale_idx] + bases[lane_idx];
        if (is_pre_or_post_lane) {
            const float sigmoid = fast_rcp(1.0f + __expf(-affine_value));
            mix_value = is_pre_lane ? sigmoid + hc_pre_eps : sigmoid * hc_post_scale;
        } else {
            mix_value = affine_value;
        }
    }
    __syncwarp();

    // epilogue:63-86 sinkhorn_normalize（warp 级）
    float value = is_comb_lane ? mix_value : 0.0f;
    const float row_max = warp_reduce_max4(value);
    value = __expf(value - row_max);
    float row_sum = warp_reduce_sum4(value);
    value = value * fast_rcp(row_sum) + sinkhorn_eps;

    const auto get_col_sum = [](const float& element) {
        float sum = element + __shfl_xor_sync(0xffffffffu, element, kNumRoutes);
        sum += __shfl_xor_sync(0xffffffffu, sum, 6 * kNumRoutes);
        return sum;
    };
    value *= fast_rcp(get_col_sum(value) + sinkhorn_eps);
    #pragma unroll 1
    for (uint32_t iteration = 1; iteration < num_sinkhorn_iters; ++ iteration) {
        row_sum = warp_reduce_sum4(value);
        value *= fast_rcp(row_sum + sinkhorn_eps);
        value *= fast_rcp(get_col_sum(value) + sinkhorn_eps);
    }
    const float normalized_comb = value;

    if (is_hc_output_lane and not is_pre_lane) {
        if (is_comb_lane) {
            const uint32_t idx = lane_idx - kFirstCombLane;
            new_comb_res_mix[token_idx * kNumRoutes * kNumRoutes + idx] = normalized_comb;
        } else {
            new_post_mix[token_idx * kNumRoutes + (lane_idx & (kNumRoutes - 1))] = mix_value;
        }
    }
    if (is_hc_output_lane and is_pre_lane)
        new_prev_mix[token_idx * kNumRoutes + lane_idx] = mix_value;
    __syncwarp();
}

static constexpr uint32_t kNumRoutes = 4;

// ---------------------------------------------------------------------------
// 实验 A kernel：HC 残差混合更新 + X1 + RMSNorm（每 warp 一个 token，H=256）
// ---------------------------------------------------------------------------
template <uint32_t kHidden>
__global__ void mhc_update_kernel(const __nv_bfloat16* __restrict__ x,
                                  const __nv_bfloat16* __restrict__ residual,
                                  const float* __restrict__ post_mix,
                                  const float* __restrict__ comb_res_mix,
                                  const float* __restrict__ pre_mix,
                                  const __nv_bfloat16* __restrict__ weight,
                                  int T, float rms_eps, float rms_scale,
                                  __nv_bfloat16* __restrict__ new_residual,
                                  float* __restrict__ hc_sqr_sum,
                                  __nv_bfloat16* __restrict__ x1_out,
                                  __nv_bfloat16* __restrict__ y_out) {
    const int token = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = get_lane_idx();
    if (token >= T) return;
    constexpr uint32_t kNumLanes = 32;
    constexpr uint32_t kHPL = kHidden / kNumLanes;   // 8，与 kHiddenPerNormLane 一致
    const uint32_t h0 = lane * kHPL;

    // 参考 run_normal_worker：new_residual 在 **bf16 处舍入**（所有消费者共享的数值边界，
    // epilogue:496 NOTES），hc_sqr_sum / X1 都从 bf16 值再算。
    float hc_sqr = 0.0f;
    float x1[kHPL];
    #pragma unroll
    for (uint32_t e = 0; e < kHPL; ++e) x1[e] = 0.0f;

    #pragma unroll
    for (uint32_t r = 0; r < kNumRoutes; ++r) {
        const float post = post_mix[token * kNumRoutes + r];
        const float pre = pre_mix[token * kNumRoutes + r];
        #pragma unroll
        for (uint32_t e = 0; e < kHPL; ++e) {
            const uint32_t h = h0 + e;
            float v = post * __bfloat162float(x[(size_t)token * kHidden + h]);
            #pragma unroll
            for (uint32_t i = 0; i < kNumRoutes; ++i) {
                const float c = comb_res_mix[(token * kNumRoutes + r) * kNumRoutes + i];
                v = __fmaf_rn(c, __bfloat162float(residual[((size_t)token * kNumRoutes + i) * kHidden + h]), v);
            }
            const float vb = bf16_round_d(v);   // epilogue:497
            new_residual[((size_t)token * kNumRoutes + r) * kHidden + h] = __float2bfloat16(vb);
            hc_sqr = __fmaf_rn(vb, vb, hc_sqr);
            x1[e] = __fmaf_rn(pre, vb, x1[e]);   // X1 = Σ_j pre_j * new_residual_j（fp32）
        }
    }

    // hc_sqr_sum：全 32-lane 归约（对应 post worker 的 full_stats / WG3 汇聚）
    float hs = hc_sqr;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) hs += __shfl_xor_sync(0xffffffffu, hs, off);
    if (lane == 0) hc_sqr_sum[token] = hs;

    // RMSNorm：X1 先舍入到 bf16（epilogue:207-209 注释），平方和再全 warp 归约
    float sq = 0.0f;
    float x1_bf[kHPL];
    #pragma unroll
    for (uint32_t e = 0; e < kHPL; ++e) {
        x1_bf[e] = bf16_round_d(x1[e]);
        sq = __fmaf_rn(x1_bf[e], x1_bf[e], sq);
    }
    float ss = sq;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, off);
    const float rms = rsqrtf(ss * (1.0f / (float)kHidden) + rms_eps) * rms_scale;

    #pragma unroll
    for (uint32_t p = 0; p < kHPL / 2; ++p) {
        const uint32_t h = h0 + 2 * p;
        const nv_bfloat162 in = {__float2bfloat16(x1_bf[2 * p]), __float2bfloat16(x1_bf[2 * p + 1])};
        const nv_bfloat162 w = *reinterpret_cast<const nv_bfloat162*>(weight + h);
        *reinterpret_cast<nv_bfloat162*>(x1_out + (size_t)token * kHidden + h) = in;
        *reinterpret_cast<nv_bfloat162*>(y_out + (size_t)token * kHidden + h) = normalize_bf16_pair(in, w, rms);
    }
}

// ---------------------------------------------------------------------------
// 实验 B kernel：mix 系数 + sinkhorn（每 warp 一个 token）
// ---------------------------------------------------------------------------
__global__ void mix_kernel(const float* __restrict__ hc_output, const float* __restrict__ hc_sqr_sum,
                           const float* __restrict__ scales, const float* __restrict__ bases,
                           int T, float inv_hc_hidden, float hc_norm_eps, float hc_pre_eps,
                           float hc_post_scale, float sinkhorn_eps, uint32_t iters,
                           float* __restrict__ new_prev_mix, float* __restrict__ new_post_mix,
                           float* __restrict__ new_comb_mix) {
    const int token = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = get_lane_idx();
    if (token >= T) return;
    const float hc_rms_scale = rsqrtf(hc_sqr_sum[token] * inv_hc_hidden + hc_norm_eps);
    const float hc_output_lane = lane < kNumRoutes * (kNumRoutes + 2) ? hc_output[token * (kNumRoutes * (kNumRoutes + 2)) + lane] : 0.0f;
    run_mix_step<kNumRoutes>(hc_output_lane, hc_rms_scale, scales, bases, hc_pre_eps, hc_post_scale,
                             sinkhorn_eps, iters, lane, new_prev_mix, new_post_mix, new_comb_mix, token);
}

// ---------------------------------------------------------------------------
// 实验 C kernel：共享专家 SF 分页行号
// ---------------------------------------------------------------------------
template <uint32_t SF_BLOCK_M>
__global__ void shared_sf_row_kernel(uint32_t* __restrict__ out, int T) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < T) out[i] = get_shared_sf_row_idx<SF_BLOCK_M>((uint32_t)i);
}

// ---------------------------------------------------------------------------
// 实验 D kernel：FP8 量化 + SF 打包（每 warp 256 hidden，lane 持 8 hidden）
// ---------------------------------------------------------------------------
__global__ void fp8_pack_kernel(const __nv_bfloat16* __restrict__ src, int T,
                                uint8_t* __restrict__ out_fp8, uint32_t* __restrict__ out_sf, int* out_nwords) {
    const int pack = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // 每个 pack = 256 hidden
    const uint32_t lane = get_lane_idx();
    if (pack >= T) return;
    const uint32_t hidden = lane * kHiddenPerNormLane;
    const auto in = reinterpret_cast<const nv_bfloat162*>(src + (size_t)pack * kHiddenPerNormPack + hidden);
    // amax 覆盖 lane 的 8 hidden（4 个 bf16x2），再按 kNumLanesPerSF=4 归约成 32-hidden 块 amax
    // （对应 epilogue:246-247 `warp_reduce_max<kNumLanesPerSF>(get_packed_bf16_amax(...))`）
    nv_bfloat16 amax = get_packed_bf16_amax4(in);
    amax = __hmax(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = __hmax(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const uint32_t sf_exp = get_ue8m0_sf_exp(amax);
    const auto sf_inv = __bfloat162bfloat162(get_ue8m0_sf_inv_bf16(sf_exp));
    const uint2 fp8v = {scale_bf16x2_into_fp8x4(in[0], in[1], sf_inv, sf_inv),
                        scale_bf16x2_into_fp8x4(in[2], in[3], sf_inv, sf_inv)};
    *reinterpret_cast<uint2*>(out_fp8 + (size_t)pack * kHiddenPerNormPack + hidden) = fp8v;
    // 4-lane SF 打包：每 128 hidden 出 1 个 int32；每 256 pack 出 2 个（lane 0 与 16 写）。
    // 注意：pack_sf_word 内部的 shfl 必须由**全部 32 lane** 一起执行（与源码一致，
    // 只有最终 store 是 lane%kNumLanesPerSFWord==0 的分支）。
    const uint32_t word = pack_sf_word(sf_exp);
    if (lane % kNumLanesPerSFWord == 0)
        out_sf[(size_t)pack * 2 + lane / kNumLanesPerSFWord] = word;
    if (pack == 0 and lane == 0) *out_nwords = 2;
}

// ---------------------------------------------------------------------------
// host 侧
// ---------------------------------------------------------------------------
#define CUDA_CHECK(x) do { cudaError_t _e = (x); \
    if (_e != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

static float bf16_round(float f) { return __bfloat162float(__float2bfloat16(f)); }

// --- 独立 double 参考：实验 A ---
static void ref_mhc_update(int T, int H, const std::vector<float>& hx,
                           const std::vector<float>& hres, const std::vector<float>& hpost,
                           const std::vector<float>& hcomb, const std::vector<float>& hpre,
                           const std::vector<float>& hw, float rms_eps, float rms_scale,
                           std::vector<float>& new_res, std::vector<double>& hc_sqr,
                           std::vector<float>& x1, std::vector<float>& y) {
    new_res.assign((size_t)T * kNumRoutes * H, 0.0f);
    hc_sqr.assign(T, 0.0);
    x1.assign((size_t)T * H, 0.0f);
    y.assign((size_t)T * H, 0.0f);
    for (int t = 0; t < T; ++t) {
        std::vector<double> hc2(H, 0.0);
        for (int r = 0; r < (int)kNumRoutes; ++r) {
            for (int h = 0; h < H; ++h) {
                double v = (double)hpost[t * kNumRoutes + r] * bf16_round(hx[(size_t)t * H + h]);
                for (int i = 0; i < (int)kNumRoutes; ++i)
                    v += (double)hcomb[(t * kNumRoutes + r) * kNumRoutes + i] * bf16_round(hres[((size_t)t * kNumRoutes + i) * H + h]);
                const float vb = bf16_round((float)v);
                new_res[((size_t)t * kNumRoutes + r) * H + h] = vb;
                hc2[h] += (double)vb * vb;
            }
        }
        double hs = 0.0;
        for (int r = 0; r < (int)kNumRoutes; ++r)
            for (int h = 0; h < H; ++h) hs += (double)new_res[((size_t)t * kNumRoutes + r) * H + h] * new_res[((size_t)t * kNumRoutes + r) * H + h];
        hc_sqr[t] = hs;
        double sq = 0.0;
        for (int h = 0; h < H; ++h) {
            double e = 0.0;
            for (int r = 0; r < (int)kNumRoutes; ++r)
                e += (double)hpre[t * kNumRoutes + r] * (double)new_res[((size_t)t * kNumRoutes + r) * H + h];
            const float eb = bf16_round((float)e);
            x1[(size_t)t * H + h] = eb;
            sq += (double)eb * eb;
        }
        const double rms = std::sqrt(1.0 / (sq / H + rms_eps)) * rms_scale;
        for (int h = 0; h < H; ++h)
            y[(size_t)t * H + h] = bf16_round((float)((double)x1[(size_t)t * H + h] * rms * (double)hw[h]));
    }
}

// --- 独立 double 参考：实验 B ---
static void ref_mix(int T, const std::vector<float>& hc_out, const std::vector<float>& hc_sqr, int H,
                    const std::vector<float>& scales, const std::vector<float>& bases,
                    float inv_hc_hidden, float hc_norm_eps, float hc_pre_eps, float hc_post_scale,
                    float sinkhorn_eps, int iters,
                    std::vector<double>& pre, std::vector<double>& post, std::vector<double>& comb) {
    const int O = kNumRoutes * (kNumRoutes + 2);
    pre.assign(T * kNumRoutes, 0.0); post.assign(T * kNumRoutes, 0.0);
    comb.assign((size_t)T * kNumRoutes * kNumRoutes, 0.0);
    for (int t = 0; t < T; ++t) {
        const double rms = std::sqrt(1.0 / (hc_sqr[t] * inv_hc_hidden + hc_norm_eps));
        double aff[24];
        for (int m = 0; m < O; ++m) {
            const int idx = m < (int)kNumRoutes ? 0 : (m < 2 * (int)kNumRoutes ? 1 : 2);
            aff[m] = (double)hc_out[t * O + m] * rms * scales[idx] + bases[m];
        }
        for (int j = 0; j < (int)kNumRoutes; ++j) {
            pre[t * kNumRoutes + j] = 1.0 / (1.0 + std::exp(-aff[j])) + hc_pre_eps;
            post[t * kNumRoutes + j] = (1.0 / (1.0 + std::exp(-aff[(int)kNumRoutes + j]))) * hc_post_scale;
        }
        double M[4][4];
        for (int j = 0; j < (int)kNumRoutes; ++j)
            for (int i = 0; i < (int)kNumRoutes; ++i) M[j][i] = aff[2 * (int)kNumRoutes + j * (int)kNumRoutes + i];
        // 镜像 kernel 的 sinkhorn_normalize（epilogue:63-86）：
        //   1) softmax(dim=-1)（不含 eps）后 + eps
        //   2) 列归一化（分母 + eps）
        //   3) (iters-1) 轮 [行归一化(分母+eps), 列归一化(分母+eps)]
        for (int j = 0; j < (int)kNumRoutes; ++j) {
            double mx = M[j][0];
            for (int i = 1; i < (int)kNumRoutes; ++i) mx = std::max(mx, M[j][i]);
            double s = 0;
            for (int i = 0; i < (int)kNumRoutes; ++i) { M[j][i] = std::exp(M[j][i] - mx); s += M[j][i]; }
            for (int i = 0; i < (int)kNumRoutes; ++i) M[j][i] = M[j][i] / s + sinkhorn_eps;
        }
        auto colnorm = [&]() {
            for (int i = 0; i < (int)kNumRoutes; ++i) {
                double s = 0;
                for (int j = 0; j < (int)kNumRoutes; ++j) s += M[j][i];
                for (int j = 0; j < (int)kNumRoutes; ++j) M[j][i] /= (s + sinkhorn_eps);
            }
        };
        auto rownorm = [&]() {
            for (int j = 0; j < (int)kNumRoutes; ++j) {
                double s = 0;
                for (int i = 0; i < (int)kNumRoutes; ++i) s += M[j][i];
                for (int i = 0; i < (int)kNumRoutes; ++i) M[j][i] /= (s + sinkhorn_eps);
            }
        };
        colnorm();
        for (int it = 1; it < iters; ++it) { rownorm(); colnorm(); }
        for (int j = 0; j < (int)kNumRoutes; ++j)
            for (int i = 0; i < (int)kNumRoutes; ++i) comb[(size_t)t * kNumRoutes * kNumRoutes + j * kNumRoutes + i] = M[j][i];
    }
}

// --- 独立 double 参考：实验 D（UE8M0 exponent + 4-block 打包，byte order = 小端） ---
static uint32_t ref_sf_word(const std::vector<float>& hsrc, int pack) {
    uint32_t word = 0;
    for (int b = 0; b < 4; ++b) {
        float amax = 0.0f;
        for (int k = 0; k < 32; ++k) amax = std::max(amax, std::fabs(bf16_round(hsrc[(size_t)pack * 256 + b * 32 + k])));
        // 数学定义：fp8 = amax * 2^(127-e) 需落进 [448/2, 448]，
        // 即 2^(127-e) = 2^floor(log2(448/amax))，故 e = 127 - floor(log2(448/amax))。
        // 与 get_ue8m0_sf_exp 的位技巧等价（在 2 的幂/边界上可能差 1，随机数据不触发）。
        int e;
        if (amax < 1e-4f) e = 105;   // 与 kMinSFExponent 对应的下限 clamp
        else { const int q = (int)std::floor(std::log2(448.0f / amax)); e = 127 - q; if (e < 105) e = 105; }
        word |= (uint32_t)(e & 0xff) << (8 * b);
    }
    return word;
}

static double relerr(double a, double b) { return std::fabs(a - b) / std::max(std::fabs(b), 1e-30); }

// E4M3 位模式 -> float（bias=7）
static float e4m3_to_float(uint8_t b) {
    const uint32_t s = (b >> 7) & 1, e = (b >> 3) & 0xF, m = b & 7;
    if (e == 0) return m == 0 ? 0.0f : (s ? -1.0f : 1.0f) * std::ldexp((float)m / 8.0f, -6);
    return (s ? -1.0f : 1.0f) * std::ldexp(1.0f + (float)m / 8.0f, (int)e - 7);
}

// ---------------------------------------------------------------------------
int main() {
    printf("Mega mHC semantics port -- H100(sm90) numeric checks\n");
    printf("(Mega mHC 主体 sm100_mega_mhc.cuh 为 SM100 专属，本实验仅移植其架构无关的数值语义与位级布局)\n\n");

    std::mt19937 rng(2026);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    // ===================== 实验 A：残差混合 + RMSNorm =====================
    {
        const int T = 256, H = 256;
        std::vector<float> hx((size_t)T * H), hres((size_t)T * kNumRoutes * H), hpost(T * kNumRoutes),
            hcomb((size_t)T * kNumRoutes * kNumRoutes), hpre(T * kNumRoutes), hw(H);
        for (auto& v : hx) v = nd(rng);
        for (auto& v : hres) v = nd(rng);
        for (auto& v : hpost) v = ud(rng);
        for (auto& v : hpre) v = ud(rng);
        for (int t = 0; t < T; ++t) {
            // comb 每行归一化（模拟真值：doubly-stochastic 附近的组合矩阵）
            for (int r = 0; r < (int)kNumRoutes; ++r) {
                float s = 0;
                for (int i = 0; i < (int)kNumRoutes; ++i) { hcomb[(t * kNumRoutes + r) * kNumRoutes + i] = ud(rng); s += hcomb[(t * kNumRoutes + r) * kNumRoutes + i]; }
                for (int i = 0; i < (int)kNumRoutes; ++i) hcomb[(t * kNumRoutes + r) * kNumRoutes + i] /= s;
            }
        }
        for (auto& v : hw) v = 1.0f + 0.1f * nd(rng);
        const float rms_eps = 7e-6f, rms_scale = 1.25f;

        __nv_bfloat16 *dx, *dres, *dw, *dnewres, *dx1, *dy;
        float *dpost, *dcomb, *dpre, *dhc;
        CUDA_CHECK(cudaMalloc(&dx, hx.size() * 2)); CUDA_CHECK(cudaMalloc(&dres, hres.size() * 2));
        CUDA_CHECK(cudaMalloc(&dw, H * 2));
        CUDA_CHECK(cudaMalloc(&dnewres, hres.size() * 2));
        CUDA_CHECK(cudaMalloc(&dx1, hx.size() * 2)); CUDA_CHECK(cudaMalloc(&dy, hx.size() * 2));
        CUDA_CHECK(cudaMalloc(&dpost, hpost.size() * 4)); CUDA_CHECK(cudaMalloc(&dcomb, hcomb.size() * 4));
        CUDA_CHECK(cudaMalloc(&dpre, hpre.size() * 4)); CUDA_CHECK(cudaMalloc(&dhc, T * 4));
        std::vector<__nv_bfloat16> bhx(hx.size()), bhres(hres.size()), bhw(H);
        for (size_t i = 0; i < hx.size(); ++i) bhx[i] = __float2bfloat16(hx[i]);
        for (size_t i = 0; i < hres.size(); ++i) bhres[i] = __float2bfloat16(hres[i]);
        for (int i = 0; i < H; ++i) bhw[i] = __float2bfloat16(hw[i]);
        CUDA_CHECK(cudaMemcpy(dx, bhx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dres, bhres.data(), hres.size() * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dw, bhw.data(), H * 2, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpost, hpost.data(), hpost.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dcomb, hcomb.data(), hcomb.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dpre, hpre.data(), hpre.size() * 4, cudaMemcpyHostToDevice));
        mhc_update_kernel<256><<<(T * 32 + 255) / 256, 256>>>(dx, dres, dpost, dcomb, dpre, dw, T,
                                                             rms_eps, rms_scale, dnewres, dhc, dx1, dy);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__nv_bfloat16> bnewres(hres.size()), bx1(hx.size()), by(hx.size());
        std::vector<float> ghc(T);
        CUDA_CHECK(cudaMemcpy(bnewres.data(), dnewres, bnewres.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(bx1.data(), dx1, bx1.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(by.data(), dy, by.size() * 2, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(ghc.data(), dhc, T * 4, cudaMemcpyDeviceToHost));

        std::vector<float> rnewres, rx1, ry; std::vector<double> rhc;
        ref_mhc_update(T, H, hx, hres, hpost, hcomb, hpre, hw, rms_eps, rms_scale, rnewres, rhc, rx1, ry);

        double e_newres = 0, e_x1 = 0, e_y = 0, e_hc = 0;
        double a_newres = 0, a_x1 = 0, a_y = 0;
        for (size_t i = 0; i < bnewres.size(); ++i) {
            const double d = std::fabs((double)__bfloat162float(bnewres[i]) - rnewres[i]);
            e_newres = std::max(e_newres, d / std::max(std::fabs((double)rnewres[i]), 1e-30));
            a_newres = std::max(a_newres, d);
        }
        for (size_t i = 0; i < bx1.size(); ++i) {
            const double d = std::fabs((double)__bfloat162float(bx1[i]) - rx1[i]);
            e_x1 = std::max(e_x1, d / std::max(std::fabs((double)rx1[i]), 1e-30));
            a_x1 = std::max(a_x1, d);
        }
        for (size_t i = 0; i < by.size(); ++i) {
            const double d = std::fabs((double)__bfloat162float(by[i]) - ry[i]);
            e_y = std::max(e_y, d / std::max(std::fabs((double)ry[i]), 1e-30));
            a_y = std::max(a_y, d);
        }
        for (int t = 0; t < T; ++t) e_hc = std::max(e_hc, relerr(ghc[t], rhc[t]));
        printf("== A 残差混合 + RMSNorm (T=%d, H=%d, hc=%u) ==\n", T, H, kNumRoutes);
        printf("   new_residual  rel %.3e / abs %.3e   hc_sqr_sum rel %.3e\n", e_newres, a_newres, e_hc);
        printf("   X1(bf16)      rel %.3e / abs %.3e   y_bf16(RMSNorm) rel %.3e / abs %.3e\n",
               e_x1, a_x1, e_y, a_y);
        printf("   （rel 受 new_residual 近零点相消放大；abs 与 bf16 单 ulp 同量级）\n\n");
    }

    // ===================== 实验 B：mix 系数 + Sinkhorn =====================
    {
        const int T = 4096, H = 4096;
        const int O = kNumRoutes * (kNumRoutes + 2);
        std::vector<float> hc_out((size_t)T * O);
        std::vector<float> hc_sqr(T);
        for (auto& v : hc_out) v = nd(rng);
        // hc_sqr_sum = Σ_{hc*H} new_residual² ≈ hc*H（new_residual ~ O(1)），取小幅扰动
        for (int t = 0; t < T; ++t)
            hc_sqr[t] = (0.8f + 0.4f * ud(rng)) * (float)(kNumRoutes * H);
        std::vector<float> scales = {0.1f, -0.05f, 0.2f}, bases(O);
        for (auto& v : bases) v = 0.1f * nd(rng);
        const float hc_norm_eps = 2e-5f, hc_pre_eps = 3e-4f, hc_post_scale = 1.25f, sinkhorn_eps = 2e-6f;
        const int iters = 10;
        const float inv_hc_hidden = 1.0f / (float)(kNumRoutes * H);

        float *dout, *dscales, *dbases, *dpre, *dpost, *dcomb;
        float *dhc;
        CUDA_CHECK(cudaMalloc(&dout, hc_out.size() * 4));
        CUDA_CHECK(cudaMalloc(&dhc, T * 4));
        CUDA_CHECK(cudaMalloc(&dscales, 3 * 4)); CUDA_CHECK(cudaMalloc(&dbases, O * 4));
        CUDA_CHECK(cudaMalloc(&dpre, T * kNumRoutes * 4)); CUDA_CHECK(cudaMalloc(&dpost, T * kNumRoutes * 4));
        CUDA_CHECK(cudaMalloc(&dcomb, (size_t)T * kNumRoutes * kNumRoutes * 4));
        CUDA_CHECK(cudaMemcpy(dout, hc_out.data(), hc_out.size() * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dhc, hc_sqr.data(), T * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dscales, scales.data(), 3 * 4, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dbases, bases.data(), O * 4, cudaMemcpyHostToDevice));
        mix_kernel<<<(T * 32 + 255) / 256, 256>>>(dout, dhc, dscales, dbases, T, inv_hc_hidden,
                                                  hc_norm_eps, hc_pre_eps, hc_post_scale, sinkhorn_eps, iters,
                                                  dpre, dpost, dcomb);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> gpre(T * kNumRoutes), gpost(T * kNumRoutes), gcomb((size_t)T * kNumRoutes * kNumRoutes);
        CUDA_CHECK(cudaMemcpy(gpre.data(), dpre, gpre.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gpost.data(), dpost, gpost.size() * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gcomb.data(), dcomb, gcomb.size() * 4, cudaMemcpyDeviceToHost));

        std::vector<double> rpre, rpost, rcomb;
        ref_mix(T, hc_out, hc_sqr, H, scales, bases, inv_hc_hidden, hc_norm_eps, hc_pre_eps, hc_post_scale,
                sinkhorn_eps, iters, rpre, rpost, rcomb);
        double e_pre = 0, e_post = 0, e_comb = 0;
        for (size_t i = 0; i < gpre.size(); ++i) e_pre = std::max(e_pre, relerr(gpre[i], rpre[i]));
        for (size_t i = 0; i < gpost.size(); ++i) e_post = std::max(e_post, relerr(gpost[i], rpost[i]));
        for (size_t i = 0; i < gcomb.size(); ++i) e_comb = std::max(e_comb, relerr(gcomb[i], rcomb[i]));
        // doubly-stochastic 性质检查（kernel 输出）
        double worst_row = 0, worst_col = 0;
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < (int)kNumRoutes; ++j) {
                double rs = 0, cs = 0;
                for (int i = 0; i < (int)kNumRoutes; ++i) { rs += gcomb[(size_t)t * kNumRoutes * kNumRoutes + j * kNumRoutes + i]; cs += gcomb[(size_t)t * kNumRoutes * kNumRoutes + i * kNumRoutes + j]; }
                worst_row = std::max(worst_row, std::fabs(rs - 1.0));
                worst_col = std::max(worst_col, std::fabs(cs - 1.0));
            }
        }
        printf("== B mix 系数 + Sinkhorn (T=%d, iters=%d) ==\n", T, iters);
        printf("   new_prev_mix max_rel_err = %.3e   new_post_mix max_rel_err = %.3e\n", e_pre, e_post);
        printf("   new_comb_res_mix max_rel_err = %.3e\n", e_comb);
        printf("   Sinkhorn 输出 |行和-1| max = %.3e, |列和-1| max = %.3e (double 参考稠密到 ~1e-3)\n\n",
               worst_row, worst_col);
    }

    // ===================== 实验 C：共享专家 SF 分页行号 =====================
    {
        const uint32_t BLOCK_M = 224;
        std::vector<uint32_t> tokens = {1, 64, 65, 200, 1025, 4096, 32768};
        printf("== C 共享专家 SF 分页 get_shared_sf_row_idx (SF_BLOCK_M=%u) ==\n", BLOCK_M);
        int total_mismatch = 0;
        for (uint32_t T : tokens) {
            uint32_t* d; CUDA_CHECK(cudaMalloc(&d, T * 4));
            shared_sf_row_kernel<BLOCK_M><<<(T + 255) / 256, 256>>>(d, T);
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint32_t> got(T);
            CUDA_CHECK(cudaMemcpy(got.data(), d, T * 4, cudaMemcpyDeviceToHost));
            int mismatch = 0;
            const uint32_t align128 = (BLOCK_M + 127) & ~127u;
            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t ib = t % BLOCK_M;
                const uint32_t ref = t / BLOCK_M * align128 + ib / 128 * 128 + (ib % 32) * 4 + (ib % 128) / 32;
                if (got[t] != ref) mismatch++;
            }
            total_mismatch += mismatch;
            printf("   T=%5u: 与 test extra_sf_rows 不符 %d / %u\n", T, mismatch, T);
            cudaFree(d);
        }
        printf("   合计不符 %d（期望 0）\n\n", total_mismatch);
    }

    // ===================== 实验 D：FP8 UE8M0 + 4-lane SF 打包 =====================
    {
        const int T = 512;   // 512 个 pack，每个 256 hidden
        std::vector<float> hsrc((size_t)T * 256);
        for (auto& v : hsrc) v = 3.0f * nd(rng);
        std::vector<__nv_bfloat16> bsrc(hsrc.size());
        for (size_t i = 0; i < hsrc.size(); ++i) bsrc[i] = __float2bfloat16(hsrc[i]);
        __nv_bfloat16* dsrc; uint8_t* dfp8; uint32_t* dsf; int* dnw;
        CUDA_CHECK(cudaMalloc(&dsrc, bsrc.size() * 2));
        CUDA_CHECK(cudaMalloc(&dfp8, bsrc.size()));
        CUDA_CHECK(cudaMalloc(&dsf, (size_t)T * 2 * 4));
        CUDA_CHECK(cudaMalloc(&dnw, 4));
        CUDA_CHECK(cudaMemcpy(dsrc, bsrc.data(), bsrc.size() * 2, cudaMemcpyHostToDevice));
        fp8_pack_kernel<<<(T * 32 + 255) / 256, 256>>>(dsrc, T, dfp8, dsf, dnw);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> gfp8(bsrc.size());
        std::vector<uint32_t> gsf((size_t)T * 2);
        CUDA_CHECK(cudaMemcpy(gfp8.data(), dfp8, gfp8.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gsf.data(), dsf, gsf.size() * 4, cudaMemcpyDeviceToHost));

        int sf_mismatch = 0; double worst_ulp = 0; float dbg_target = 0, dbg_deq = 0; uint8_t dbg_byte = 0;
        for (int p = 0; p < T; ++p) {
            const uint32_t rw0 = ref_sf_word(hsrc, p);   // 前 128 hidden 的 4 个 exponent
            if (gsf[(size_t)p * 2 + 0] != rw0) sf_mismatch++;
            // E4M3 量化精度：在 fp8 单位下比较 kernel 的 fp8 值与「bf16 * 2^(127-e)」，以 ulp 计
            for (int b = 0; b < 4; ++b) {
                const uint32_t e = (rw0 >> (8 * b)) & 0xff;
                const float dv_inv = std::ldexp(1.0f, 127 - (int)e);   // = sf_inv
                for (int k = 0; k < 32; ++k) {
                    const int h = b * 32 + k;
                    const float target = bf16_round(hsrc[(size_t)p * 256 + h]) * dv_inv;   // 期望的 fp8 输入
                    const float deq = e4m3_to_float(gfp8[(size_t)p * 256 + h]);            // kernel 产出的 fp8 值
                    int ex; std::frexp(std::fabs(target), &ex);
                    // E4M3：normal 步长 2^(ex-4)；低于最小 normal 2^-6 时为 subnormal，步长固定 2^-9
                    const float ulp = ex <= -6 ? std::ldexp(1.0f, -9) : std::ldexp(1.0f, ex - 4);
                    const double u = (double)std::fabs(deq - target) / ulp;
                    if (u > worst_ulp) { worst_ulp = u; dbg_target = target; dbg_deq = deq; dbg_byte = gfp8[(size_t)p * 256 + h]; }
                }
            }
        }
        printf("== D FP8 UE8M0 + 4-lane SF 打包 (T=%d packs) ==\n", T);
        for (int p = 0; p < 3; ++p) {
            const uint32_t rw = ref_sf_word(hsrc, p);
            printf("   pack %d: got=0x%08x ref=0x%08x  ref_bytes=(%u,%u,%u,%u)\n",
                   p, gsf[(size_t)p * 2], rw, rw & 0xff, (rw >> 8) & 0xff, (rw >> 16) & 0xff, (rw >> 24) & 0xff);
        }
        printf("   打包 int32 与参考（小端 exp0|exp1<<8|exp2<<16|exp3<<24）不符: %d / %d\n", sf_mismatch, T);
        printf("   E4M3 量化误差 max = %.4f ulp （round-to-nearest 期望 <= 0.5）\n", worst_ulp);
        (void)dnw; (void)dbg_target; (void)dbg_deq; (void)dbg_byte;
    }

    printf("done.\n");
    return 0;
}
