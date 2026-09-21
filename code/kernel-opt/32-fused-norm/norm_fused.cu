// 32 模型场景算子：RMSNorm / QK-Norm / 融合残差 / 融合 FP8 量化
//
// shape 取自本地模型 config：
//   DeepSeek-V4-Pro：hidden=7168, 128 head, head_dim=512(qk_rope=64), eps=1e-6
//   Qwen3-8B：      hidden=5120, 40 q / 8 kv head, head_dim=128, eps=1e-6
//   GLM-5.2-FP8：   hidden=6144, eps=1e-5
// 训练/推理里每层都会跑的「归一化 + 残差 + 量化」链：
//   x1 = x + res;  y = rmsnorm(x1) * w;  yq = fp8(y)（DeepSeek FP8 的 1x128 动态 scale）
//   QK-Norm：对 q/k 的每个 head 在 head_dim 上做 RMSNorm（Qwen3/GLM 用）。
//
// 变体：
//   v0  rmsnorm 朴素（每行 block，全局读两遍，无 smem）
//   v1  rmsnorm 单读（行缓存进 smem，读一遍写一遍 = 最小流量）
//   v2  add + rmsnorm 融合（省一趟 x+res 的读写）
//   v3  v2 + FP8 e4m3 per-128 量化融合（再省输出字节）
//   v4  QK-Norm（每 head 一个 warp，head_dim 上的 RMSNorm）
//
// 运行：ARCH=sm_90 scripts/run.sh 32-fused-norm/norm_fused.cu
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <vector>

using bf16 = __nv_bfloat16;
using fp8 = __nv_fp8_e4m3;

constexpr int H = 7168;      // DeepSeek-V4-Pro hidden
constexpr int QH = 40, KH = 8, HD = 128;  // Qwen3-8B GQA head 数 / head_dim
constexpr float EPS = 1e-6f;
constexpr int QB = 128;      // 量化 block（DeepSeek 激活 1x128 动态 scale）

// ---------------------------------------------------------------------------
// v0: 朴素 rmsnorm —— 每行一个 block，读 x 两遍（无 smem 缓存）
// ---------------------------------------------------------------------------
__global__ void rmsnorm_v0_kernel(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                  bf16* __restrict__ y, int M, int h) {
  const int row = blockIdx.x;
  const int t = threadIdx.x, T = blockDim.x;
  float ss = 0.f;
  for (int c = t; c < h; c += T) {
    const float v = __bfloat162float(x[(size_t)row * h + c]);
    ss += v * v;
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float v = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (t == 0) red[0] = rsqrtf(v / h + EPS);
  }
  __syncthreads();
  const float inv = red[0];
  for (int c = t; c < h; c += T) {
    const float v = __bfloat162float(x[(size_t)row * h + c]);
    y[(size_t)row * h + c] = __float2bfloat16(v * inv * __bfloat162float(w[c]));
  }
}

// ---------------------------------------------------------------------------
// v1: rmsnorm 单读 —— 行缓存进 smem，全局只读一遍写一遍
// ---------------------------------------------------------------------------
template <int T>
__global__ void __launch_bounds__(T) rmsnorm_v1_kernel(const bf16* __restrict__ x,
                                                       const bf16* __restrict__ w,
                                                       bf16* __restrict__ y, int M, int h) {
  extern __shared__ bf16 srow[];
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  float ss = 0.f;
  // 向量化 8 bf16（16B）载入 smem
  for (int c8 = t; c8 < h / 8; c8 += T) {
    const uint4 v = *reinterpret_cast<const uint4*>(&x[(size_t)row * h + c8 * 8]);
    *reinterpret_cast<uint4*>(&srow[c8 * 8]) = v;
    const bf16* p = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float f = __bfloat162float(p[i]);
      ss += f * f;
    }
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float v = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (t == 0) red[0] = rsqrtf(v / h + EPS);
  }
  __syncthreads();
  const float inv = red[0];
  for (int c8 = t; c8 < h / 8; c8 += T) {
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      po[i] = __float2bfloat16(__bfloat162float(srow[c]) * inv * __bfloat162float(w[c]));
    }
    *reinterpret_cast<uint4*>(&y[(size_t)row * h + c8 * 8]) = o;
  }
}

// ---------------------------------------------------------------------------
// v2: 融合 residual add + rmsnorm —— s=x+res 写回（给下一层当残差），y=rmsnorm(s)
//     读 x,res（各 h*2B）写 s,y（各 h*2B）；不融合要 add(2r+1w)+norm(1r+1w)
// ---------------------------------------------------------------------------
template <int T>
__global__ void __launch_bounds__(T) add_rmsnorm_v2_kernel(const bf16* __restrict__ x,
                                                           const bf16* __restrict__ res,
                                                           const bf16* __restrict__ w,
                                                           bf16* __restrict__ s,
                                                           bf16* __restrict__ y, int M, int h) {
  extern __shared__ bf16 srow[];
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  float ss = 0.f;
  for (int c8 = t; c8 < h / 8; c8 += T) {
    const uint4 vx = *reinterpret_cast<const uint4*>(&x[(size_t)row * h + c8 * 8]);
    const uint4 vr = *reinterpret_cast<const uint4*>(&res[(size_t)row * h + c8 * 8]);
    const bf16* px = reinterpret_cast<const bf16*>(&vx);
    const bf16* pr = reinterpret_cast<const bf16*>(&vr);
    uint4 vs;
    bf16* ps = reinterpret_cast<bf16*>(&vs);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = __bfloat162float(px[i]) + __bfloat162float(pr[i]);
      ps[i] = __float2bfloat16(a);
      ss += a * a;
    }
    *reinterpret_cast<uint4*>(&srow[c8 * 8]) = vs;
    *reinterpret_cast<uint4*>(&s[(size_t)row * h + c8 * 8]) = vs;
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float v = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (t == 0) red[0] = rsqrtf(v / h + EPS);
  }
  __syncthreads();
  const float inv = red[0];
  for (int c8 = t; c8 < h / 8; c8 += T) {
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      po[i] = __float2bfloat16(__bfloat162float(srow[c]) * inv * __bfloat162float(w[c]));
    }
    *reinterpret_cast<uint4*>(&y[(size_t)row * h + c8 * 8]) = o;
  }
}

// ---------------------------------------------------------------------------
// v3: v2 + FP8 e4m3 per-128 动态量化。y 写 fp8（1B/elem），scale 写 [M, h/128]
//     scale = amax/448；yq = y/scale。DeepSeek FP8 激活就是 1x128 动态 scale。
// ---------------------------------------------------------------------------
constexpr float FP8_MAX = 448.f;

template <int T>
__global__ void __launch_bounds__(T) add_rmsnorm_fp8_v3_kernel(const bf16* __restrict__ x,
                                                               const bf16* __restrict__ res,
                                                               const bf16* __restrict__ w,
                                                               bf16* __restrict__ s,
                                                               fp8* __restrict__ yq,
                                                               float* __restrict__ scale,
                                                               int M, int h) {
  extern __shared__ char smem[];
  bf16* srow = reinterpret_cast<bf16*>(smem);                       // [h]
  float* amaxb = reinterpret_cast<float*>(smem + (size_t)h * 2);    // [h/QB]
  const int row = blockIdx.x;
  const int t = threadIdx.x;
  const int nb = h / QB;
  if (t < nb) amaxb[t] = 0.f;
  float ss = 0.f;
  for (int c8 = t; c8 < h / 8; c8 += T) {
    const uint4 vx = *reinterpret_cast<const uint4*>(&x[(size_t)row * h + c8 * 8]);
    const uint4 vr = *reinterpret_cast<const uint4*>(&res[(size_t)row * h + c8 * 8]);
    const bf16* px = reinterpret_cast<const bf16*>(&vx);
    const bf16* pr = reinterpret_cast<const bf16*>(&vr);
    uint4 vs;
    bf16* ps = reinterpret_cast<bf16*>(&vs);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const float a = __bfloat162float(px[i]) + __bfloat162float(pr[i]);
      ps[i] = __float2bfloat16(a);
      ss += a * a;
    }
    *reinterpret_cast<uint4*>(&srow[c8 * 8]) = vs;
    *reinterpret_cast<uint4*>(&s[(size_t)row * h + c8 * 8]) = vs;
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float v = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(~0u, v, o);
    if (t == 0) red[0] = rsqrtf(v / h + EPS);
  }
  __syncthreads();
  const float inv = red[0];
  // pass A：只求每个 128-block 的 amax（atomicMax on float bits），不写回 srow
  for (int c8 = t; c8 < h / 8; c8 += T) {
    float lamax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      const float yv = __bfloat162float(srow[c]) * inv * __bfloat162float(w[c]);
      lamax = fmaxf(lamax, fabsf(yv));
    }
    atomicMax(reinterpret_cast<int*>(&amaxb[(c8 * 8) / QB]), __float_as_int(lamax));
  }
  __syncthreads();
  // pass B：用 amax 算 scale 并量化
  if (t < nb) scale[(size_t)row * nb + t] = fmaxf(amaxb[t], 1e-6f) / FP8_MAX;
  for (int c8 = t; c8 < h / 8; c8 += T) {
    // 坑：不要用 `fp8(存储字节)` 构造再存——本工具链下会走错重载把 1 字节当 float。
    // 直接把 `__nv_fp8_storage_t`（uint8）写进 yq 的字节流。
    unsigned char ob[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      const float sc = fmaxf(amaxb[c / QB], 1e-6f) / FP8_MAX;
      const float yv = __bfloat162float(srow[c]) * inv * __bfloat162float(w[c]);
      ob[i] = __nv_cvt_float_to_fp8(yv / sc, __NV_SATFINITE, __NV_E4M3);
    }
    unsigned char* dst = reinterpret_cast<unsigned char*>(yq) + (size_t)row * h + c8 * 8;
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(ob);
  }
}

// ---------------------------------------------------------------------------
// v4: QK-Norm —— 每 (token, head) 一个 warp，在 head_dim 上做 RMSNorm（Qwen3/GLM）
// ---------------------------------------------------------------------------
__global__ void qk_norm_v4_kernel(const bf16* __restrict__ q, const bf16* __restrict__ qw,
                                  const bf16* __restrict__ k, const bf16* __restrict__ kw,
                                  bf16* __restrict__ qo, bf16* __restrict__ ko, int M, int nh_q,
                                  int nh_k, int hd) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  const int nw = (nh_q + nh_k);
  const int row = warp / nw, hh = warp % nw;
  const bool isq = hh < nh_q;
  const bf16* src = isq ? q : k;
  const bf16* wt = isq ? qw : kw;
  bf16* dst = isq ? qo : ko;
  const int head = isq ? hh : hh - nh_q;
  const size_t base = ((size_t)row * (isq ? nh_q : nh_k) + head) * hd;
  float ss = 0.f;
  for (int c = lane * 4; c < hd; c += 128) {
    const uint2 v = *reinterpret_cast<const uint2*>(&src[base + c]);
    const bf16* p = reinterpret_cast<const bf16*>(&v);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float f = __bfloat162float(p[i]);
      ss += f * f;
    }
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  ss = __shfl_sync(~0u, ss, 0);
  const float inv = rsqrtf(ss / hd + EPS);
  for (int c = lane * 4; c < hd; c += 128) {
    const uint2 v = *reinterpret_cast<const uint2*>(&src[base + c]);
    const bf16* p = reinterpret_cast<const bf16*>(&v);
    uint2 o2;
    bf16* po = reinterpret_cast<bf16*>(&o2);
#pragma unroll
    for (int i = 0; i < 4; ++i)
      po[i] = __float2bfloat16(__bfloat162float(p[i]) * inv * __bfloat162float(wt[c + i]));
    *reinterpret_cast<uint2*>(&dst[base + c]) = o2;
  }
}

// ---------------------------------------------------------------------------
// v1r / v2r：寄存器缓存版——每线程把自己的元素留在寄存器里，省掉 smem 往返（消 L1 瓶颈）
// ---------------------------------------------------------------------------
template <int T>
__global__ void __launch_bounds__(T) rmsnorm_v1r_kernel(const bf16* __restrict__ x,
                                                        const bf16* __restrict__ w,
                                                        bf16* __restrict__ y, int M) {
  constexpr int G = H / 8, NG = (G + T - 1) / T;
  const int row = blockIdx.x, t = threadIdx.x;
  uint4 v[NG];
  float ss = 0.f;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    v[g] = make_uint4(0, 0, 0, 0);
    if (c8 < G) {
      v[g] = *reinterpret_cast<const uint4*>(&x[(size_t)row * H + c8 * 8]);
      const bf16* p = reinterpret_cast<const bf16*>(&v[g]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float f = __bfloat162float(p[i]);
        ss += f * f;
      }
    }
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float vv = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) vv += __shfl_down_sync(~0u, vv, o);
    if (t == 0) red[0] = rsqrtf(vv / H + EPS);
  }
  __syncthreads();
  const float inv = red[0];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    if (c8 >= G) continue;
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
    const bf16* p = reinterpret_cast<const bf16*>(&v[g]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      po[i] = __float2bfloat16(__bfloat162float(p[i]) * inv * __bfloat162float(w[c]));
    }
    *reinterpret_cast<uint4*>(&y[(size_t)row * H + c8 * 8]) = o;
  }
}

template <int T>
__global__ void __launch_bounds__(T) add_rmsnorm_v2r_kernel(const bf16* __restrict__ x,
                                                            const bf16* __restrict__ res,
                                                            const bf16* __restrict__ w,
                                                            bf16* __restrict__ s,
                                                            bf16* __restrict__ y, int M) {
  constexpr int G = H / 8, NG = (G + T - 1) / T;
  const int row = blockIdx.x, t = threadIdx.x;
  uint4 vs[NG];
  float ss = 0.f;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    vs[g] = make_uint4(0, 0, 0, 0);
    if (c8 < G) {
      const uint4 vx = *reinterpret_cast<const uint4*>(&x[(size_t)row * H + c8 * 8]);
      const uint4 vr = *reinterpret_cast<const uint4*>(&res[(size_t)row * H + c8 * 8]);
      const bf16* px = reinterpret_cast<const bf16*>(&vx);
      const bf16* pr = reinterpret_cast<const bf16*>(&vr);
      bf16* ps = reinterpret_cast<bf16*>(&vs[g]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float a = __bfloat162float(px[i]) + __bfloat162float(pr[i]);
        ps[i] = __float2bfloat16(a);
        ss += a * a;
      }
      *reinterpret_cast<uint4*>(&s[(size_t)row * H + c8 * 8]) = vs[g];
    }
  }
  __shared__ float red[32];
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float vv = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) vv += __shfl_down_sync(~0u, vv, o);
    if (t == 0) red[0] = rsqrtf(vv / H + EPS);
  }
  __syncthreads();
  const float inv = red[0];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    if (c8 >= G) continue;
    uint4 o;
    bf16* po = reinterpret_cast<bf16*>(&o);
    const bf16* ps = reinterpret_cast<const bf16*>(&vs[g]);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      po[i] = __float2bfloat16(__bfloat162float(ps[i]) * inv * __bfloat162float(w[c]));
    }
    *reinterpret_cast<uint4*>(&y[(size_t)row * H + c8 * 8]) = o;
  }
}

// v3r：寄存器缓存 + per-128 动态量化的融合版（不落 smem，amax 走 shared atomicMax）
template <int T>
__global__ void __launch_bounds__(T) add_rmsnorm_fp8_v3r_kernel(const bf16* __restrict__ x,
                                                                const bf16* __restrict__ res,
                                                                const bf16* __restrict__ w,
                                                                bf16* __restrict__ s,
                                                                fp8* __restrict__ yq,
                                                                float* __restrict__ scale, int M) {
  constexpr int G = H / 8, NG = (G + T - 1) / T;
  __shared__ float amaxb[H / QB];
  __shared__ float red[32];
  const int row = blockIdx.x, t = threadIdx.x;
  if (t < H / QB) amaxb[t] = 0.f;
  uint4 vs[NG];
  float yv[NG * 8];
  float ss = 0.f;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    vs[g] = make_uint4(0, 0, 0, 0);
    if (c8 < G) {
      const uint4 vx = *reinterpret_cast<const uint4*>(&x[(size_t)row * H + c8 * 8]);
      const uint4 vr = *reinterpret_cast<const uint4*>(&res[(size_t)row * H + c8 * 8]);
      const bf16* px = reinterpret_cast<const bf16*>(&vx);
      const bf16* pr = reinterpret_cast<const bf16*>(&vr);
      bf16* ps = reinterpret_cast<bf16*>(&vs[g]);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float a = __bfloat162float(px[i]) + __bfloat162float(pr[i]);
        ps[i] = __float2bfloat16(a);
        ss += a * a;
      }
      *reinterpret_cast<uint4*>(&s[(size_t)row * H + c8 * 8]) = vs[g];
    }
  }
  for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(~0u, ss, o);
  if ((t & 31) == 0) red[t >> 5] = ss;
  __syncthreads();
  if (t < 32) {
    float vv = (t < (T + 31) / 32) ? red[t] : 0.f;
    for (int o = 16; o > 0; o >>= 1) vv += __shfl_down_sync(~0u, vv, o);
    if (t == 0) red[0] = rsqrtf(vv / H + EPS);
  }
  __syncthreads();
  const float inv = red[0];
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    if (c8 >= G) continue;
    const bf16* ps = reinterpret_cast<const bf16*>(&vs[g]);
    float lamax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      yv[g * 8 + i] = __bfloat162float(ps[i]) * inv * __bfloat162float(w[c]);
      lamax = fmaxf(lamax, fabsf(yv[g * 8 + i]));
    }
    atomicMax(reinterpret_cast<int*>(&amaxb[(c8 * 8) / QB]), __float_as_int(lamax));
  }
  __syncthreads();
  if (t < H / QB) scale[(size_t)row * (H / QB) + t] = fmaxf(amaxb[t], 1e-6f) / FP8_MAX;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    const int c8 = t + g * T;
    if (c8 >= G) continue;
    unsigned char ob[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = c8 * 8 + i;
      const float sc = fmaxf(amaxb[c / QB], 1e-6f) / FP8_MAX;
      ob[i] = __nv_cvt_float_to_fp8(yv[g * 8 + i] / sc, __NV_SATFINITE, __NV_E4M3);
    }
    unsigned char* dst = reinterpret_cast<unsigned char*>(yq) + (size_t)row * H + c8 * 8;
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(ob);
  }
}

__global__ void dequant_fp8_kernel(const fp8* __restrict__ in, float* __restrict__ out, long n) {
  const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  // 坑：本工具链下 `float(fp8)` 会返回原始 1 字节位模式（不是数值），必须走
  // `__nv_cvt_fp8_to_halfraw` 解码。见 fp8_test.cu / fp8_test2.cu。
  if (i < n)
    out[i] = __half2float(__nv_cvt_fp8_to_halfraw(reinterpret_cast<const __nv_fp8_storage_t*>(in)[i],
                                                  __NV_E4M3));
}

// ===========================================================================
static void rms_ref(int row, int h, const std::vector<bf16>& x, const std::vector<bf16>& w,
                    std::vector<double>& out) {
  double ss = 0;
  for (int c = 0; c < h; ++c) {
    const double v = __bfloat162float(x[(size_t)row * h + c]);
    ss += v * v;
  }
  const double inv = 1.0 / std::sqrt(ss / h + EPS);
  out.resize(h);
  for (int c = 0; c < h; ++c)
    out[c] = __bfloat162float(x[(size_t)row * h + c]) * inv * __bfloat162float(w[c]);
}

int main() {
  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double peak = d.mem_bw_gbps;
  const int M = 8192;
  std::printf("\nRMSNorm / QK-Norm fusion: M=%d H=%d (DeepSeek-V4-Pro)  Qwen3 QK: %d/%d x %d\n", M,
              H, QH, KH, HD);

  // ---- 数据 ----
  const size_t n = (size_t)M * H;
  std::vector<bf16> hx(n), hr(n), hw(H);
  srand(7);
  auto rnd = [] { return __float2bfloat16(0.5f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& v : hx) v = rnd();
  for (auto& v : hr) v = rnd();
  for (auto& v : hw) v = __float2bfloat16(0.5f + (float)rand() / RAND_MAX);
  bf16 *x, *res, *w, *y, *s;
  fp8* yq;
  float* scale;
  CUDA_CHECK(cudaMalloc(&x, n * 2));
  CUDA_CHECK(cudaMalloc(&res, n * 2));
  CUDA_CHECK(cudaMalloc(&w, H * 2));
  CUDA_CHECK(cudaMalloc(&y, n * 2));
  CUDA_CHECK(cudaMalloc(&s, n * 2));
  CUDA_CHECK(cudaMalloc(&yq, n));
  CUDA_CHECK(cudaMalloc(&scale, (size_t)M * (H / QB) * 4));
  CUDA_CHECK(cudaMemcpy(x, hx.data(), n * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(res, hr.data(), n * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(w, hw.data(), H * 2, cudaMemcpyHostToDevice));

  const int T = 256;
  auto set_shm = [](auto fn, int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
  };
  const int shm_row = H * 2;
  const size_t io_rms = n * 2 * 2;            // read x + write y
  const size_t io_add = n * 2 * 4;            // read x,res + write s,y
  const size_t io_fp8 = n * 2 * 3 + n;        // read x,res(2) + write s(2) + fp8(1)

  // v0
  {
    auto fn = rmsnorm_v0_kernel;
    auto run = [&] { fn<<<M, T>>>(x, w, y, M, H); };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
    std::vector<double> ref;
    rms_ref(1234, H, hx, hw, ref);
    double err = 0, rr = 0;
    for (int c = 0; c < H; ++c) {
      err = std::max(err, std::fabs((double)__bfloat162float(hy[(size_t)1234 * H + c]) - ref[c]));
      rr = std::max(rr, std::fabs(ref[c]));
    }
    double ms = bench_ms(run, 3, 20);
    std::printf("  v0 rmsnorm naive        %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e\n", ms,
                to_gbps(io_rms, ms), 100 * to_gbps(io_rms, ms) / peak, err);
  }
  // v1
  {
    auto fn = rmsnorm_v1_kernel<T>;
    set_shm(fn, shm_row);
    auto run = [&] { fn<<<M, T, shm_row>>>(x, w, y, M, H); };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
    std::vector<double> ref;
    rms_ref(999, H, hx, hw, ref);
    double err = 0;
    for (int c = 0; c < H; ++c)
      err = std::max(err, std::fabs((double)__bfloat162float(hy[(size_t)999 * H + c]) - ref[c]));
    double ms = bench_ms(run, 3, 20);
    std::printf("  v1 rmsnorm smem 1-read  %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e\n", ms,
                to_gbps(io_rms, ms), 100 * to_gbps(io_rms, ms) / peak, err);
  }
  // v2
  {
    auto fn = add_rmsnorm_v2_kernel<T>;
    set_shm(fn, shm_row);
    auto run = [&] { fn<<<M, T, shm_row>>>(x, res, w, s, y, M, H); };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hy(n), hs(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), s, n * 2, cudaMemcpyDeviceToHost));
    double err = 0, serr = 0;
    for (int c = 0; c < H; ++c) {
      const double e = (double)__bfloat162float(hx[(size_t)55 * H + c]) +
                       (double)__bfloat162float(hr[(size_t)55 * H + c]);
      serr = std::max(serr, std::fabs((double)__bfloat162float(hs[(size_t)55 * H + c]) - e));
    }
    std::vector<bf16> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2bfloat16(__bfloat162float(hx[i]) + __bfloat162float(hr[i]));
    std::vector<double> ref;
    rms_ref(55, H, tmp, hw, ref);
    for (int c = 0; c < H; ++c)
      err = std::max(err, std::fabs((double)__bfloat162float(hy[(size_t)55 * H + c]) - ref[c]));
    double ms = bench_ms(run, 3, 20);
    std::printf("  v2 add+rmsnorm fused    %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e/%.2e\n", ms,
                to_gbps(io_add, ms), 100 * to_gbps(io_add, ms) / peak, err, serr);
  }
  // v1r
  {
    auto fn = rmsnorm_v1r_kernel<T>;
    auto run = [&] { fn<<<M, T>>>(x, w, y, M); };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
    std::vector<double> ref;
    rms_ref(999, H, hx, hw, ref);
    double err = 0;
    for (int c = 0; c < H; ++c)
      err = std::max(err, std::fabs((double)__bfloat162float(hy[(size_t)999 * H + c]) - ref[c]));
    double ms = bench_ms(run, 3, 20);
    std::printf("  v1r rmsnorm reg cache   %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e\n", ms,
                to_gbps(io_rms, ms), 100 * to_gbps(io_rms, ms) / peak, err);
  }
  // v2r
  {
    auto fn = add_rmsnorm_v2r_kernel<T>;
    auto run = [&] { fn<<<M, T>>>(x, res, w, s, y, M); };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hy(n);
    CUDA_CHECK(cudaMemcpy(hy.data(), y, n * 2, cudaMemcpyDeviceToHost));
    std::vector<bf16> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2bfloat16(__bfloat162float(hx[i]) + __bfloat162float(hr[i]));
    std::vector<double> ref;
    rms_ref(55, H, tmp, hw, ref);
    double err = 0;
    for (int c = 0; c < H; ++c)
      err = std::max(err, std::fabs((double)__bfloat162float(hy[(size_t)55 * H + c]) - ref[c]));
    double ms = bench_ms(run, 3, 20);
    std::printf("  v2r add+rmsnorm reg     %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e\n", ms,
                to_gbps(io_add, ms), 100 * to_gbps(io_add, ms) / peak, err);
  }
  // v3
  {
    auto fn = add_rmsnorm_fp8_v3_kernel<T>;
    const int shm_fp8 = shm_row + (H / QB) * 4;
    set_shm(fn, shm_fp8);
    auto run = [&] { fn<<<M, T, shm_fp8>>>(x, res, w, s, yq, scale, M, H); };
    run();
    CUDA_CHECK_LAST();
    std::vector<float> hyq(n);
    std::vector<float> hsc((size_t)M * (H / QB));
    {
      float* dtmp;
      CUDA_CHECK(cudaMalloc(&dtmp, n * 4));
      dequant_fp8_kernel<<<(int)((n + 255) / 256), 256>>>(yq, dtmp, (long)n);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(hyq.data(), dtmp, n * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(dtmp));
    }
    CUDA_CHECK(cudaMemcpy(hsc.data(), scale, (size_t)M * (H / QB) * 4, cudaMemcpyDeviceToHost));
    std::vector<bf16> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2bfloat16(__bfloat162float(hx[i]) + __bfloat162float(hr[i]));
    std::vector<double> ref;
    rms_ref(77, H, tmp, hw, ref);
    double err = 0, rel = 0;
    for (int c = 0; c < H; ++c) {
      const float sc = hsc[(size_t)77 * (H / QB) + c / QB];
      const double got = (double)hyq[(size_t)77 * H + c] * sc;
      err = std::max(err, std::fabs(got - ref[c]));
      rel = std::max(rel, std::fabs(ref[c]));
    }
    double ms = bench_ms(run, 3, 20);
    std::printf("  v3 add+rmsnorm+fp8      %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e (rel %.2f%%)\n",
                ms, to_gbps(io_fp8, ms), 100 * to_gbps(io_fp8, ms) / peak, err,
                100 * err / std::max(rel, 1e-6));
  }
  // v3r
  {
    auto fn = add_rmsnorm_fp8_v3r_kernel<T>;
    auto run = [&] { fn<<<M, T>>>(x, res, w, s, yq, scale, M); };
    run();
    CUDA_CHECK_LAST();
    std::vector<float> hyq(n);
    std::vector<float> hsc((size_t)M * (H / QB));
    {
      float* dtmp;
      CUDA_CHECK(cudaMalloc(&dtmp, n * 4));
      dequant_fp8_kernel<<<(int)((n + 255) / 256), 256>>>(yq, dtmp, (long)n);
      CUDA_CHECK_LAST();
      CUDA_CHECK(cudaMemcpy(hyq.data(), dtmp, n * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaFree(dtmp));
    }
    CUDA_CHECK(cudaMemcpy(hsc.data(), scale, (size_t)M * (H / QB) * 4, cudaMemcpyDeviceToHost));
    std::vector<bf16> tmp(n);
    for (size_t i = 0; i < n; ++i) tmp[i] = __float2bfloat16(__bfloat162float(hx[i]) + __bfloat162float(hr[i]));
    std::vector<double> ref;
    rms_ref(77, H, tmp, hw, ref);
    double err = 0, rel = 0;
    for (int c = 0; c < H; ++c) {
      const float sc = hsc[(size_t)77 * (H / QB) + c / QB];
      err = std::max(err, std::fabs((double)hyq[(size_t)77 * H + c] * sc - ref[c]));
      rel = std::max(rel, std::fabs(ref[c]));
    }
    double ms = bench_ms(run, 3, 20);
    std::printf("  v3r add+rmsnorm+fp8 reg %7.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e (rel %.2f%%)\n",
                ms, to_gbps(io_fp8, ms), 100 * to_gbps(io_fp8, ms) / peak, err,
                100 * err / std::max(rel, 1e-6));
  }
  // v4 QK-Norm
  {
    const int Mq = 4096;
    const size_t nq = (size_t)Mq * QH * HD, nk = (size_t)Mq * KH * HD;
    std::vector<bf16> hq(nq), hk(nk), hqw(HD), hkw(HD);
    for (auto& v : hq) v = rnd();
    for (auto& v : hk) v = rnd();
    for (auto& v : hqw) v = __float2bfloat16(0.5f + (float)rand() / RAND_MAX);
    for (auto& v : hkw) v = __float2bfloat16(0.5f + (float)rand() / RAND_MAX);
    bf16 *dq, *dk, *dqw, *dkw, *dqo, *dko;
    CUDA_CHECK(cudaMalloc(&dq, nq * 2));
    CUDA_CHECK(cudaMalloc(&dk, nk * 2));
    CUDA_CHECK(cudaMalloc(&dqw, HD * 2));
    CUDA_CHECK(cudaMalloc(&dkw, HD * 2));
    CUDA_CHECK(cudaMalloc(&dqo, nq * 2));
    CUDA_CHECK(cudaMalloc(&dko, nk * 2));
    CUDA_CHECK(cudaMemcpy(dq, hq.data(), nq * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk, hk.data(), nk * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dqw, hqw.data(), HD * 2, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dkw, hkw.data(), HD * 2, cudaMemcpyHostToDevice));
    const int nw = (QH + KH) * Mq;
    const int warps_per_block = 8;
    const int blocks = (nw + warps_per_block - 1) / warps_per_block;
    auto run = [&] {
      qk_norm_v4_kernel<<<blocks, warps_per_block * 32>>>(dq, dqw, dk, dkw, dqo, dko, Mq, QH, KH,
                                                          HD);
    };
    run();
    CUDA_CHECK_LAST();
    std::vector<bf16> hqo(nq);
    CUDA_CHECK(cudaMemcpy(hqo.data(), dqo, nq * 2, cudaMemcpyDeviceToHost));
    // 参考：Qwen3-8B 某 token/head
    {
      const int row = 7, head = 3;
      const size_t b = ((size_t)row * QH + head) * HD;
      double ss = 0;
      for (int c = 0; c < HD; ++c) {
        const double v = __bfloat162float(hq[b + c]);
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / HD + EPS);
      double err = 0, rr = 0;
      for (int c = 0; c < HD; ++c) {
        const double e = __bfloat162float(hq[b + c]) * inv * __bfloat162float(hqw[c]);
        err = std::max(err, std::fabs((double)__bfloat162float(hqo[b + c]) - e));
        rr = std::max(rr, std::fabs(e));
      }
      const size_t bytes = (nq + nk) * 2 * 2 + 2 * HD * 2;
      double ms = bench_ms(run, 3, 20);
      std::printf("  v4 QK-Norm (q%d/k%d hd%d) %6.4f ms  %7.1f GB/s (%5.1f%% HBM)  err=%.2e\n", QH, KH,
                  HD, ms, to_gbps(bytes, ms), 100 * to_gbps(bytes, ms) / peak, err);
    }
  }
  return 0;
}
