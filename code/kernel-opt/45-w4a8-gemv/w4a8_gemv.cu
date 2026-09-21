// 45 W4A8 的 M=1 GEMV（主题 26e）——用 dp4a 整数点积补上 44 篇的 ALU gap
//
// 44 篇把 M=1 的 W4A16 decode 做成「每 warp 一行、warp 内沿 K 并行」的带宽
// 最优 GEMV，拿到 0.0284 ms / 1566.6 GB/s / 49.7% HBM。ncu 判决瓶颈是
// **ALU 发射口（pipe 64.9%）**而非访存：每个 int4 权重要先 `I2F` 变成 fp32、
// 再和 bf16 激活做 FMA，32 个权重就是 32 条 I2F + 32 条 FMA + 若干移位/掩码。
//
// 本篇问：M=1 时能不能把「反量化 + 点积」从浮点搬到整数？
//   · 激活也量化成 int8（per-128-group scale），权重保持 int4；
//   · 用 `__dp4a`（一条指令做 4 个 int8 乘加）替代 4 次 FMA + 4 次 I2F；
//   · int4→int8 用 `PRMT`（`__byte_perm`）+ 一次 XOR/SUB 做符号展开。
// 目标：把每元素 ALU 指令数从 ~5 降到 ~2，顶到 60%+ HBM。
//
// 真实 shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128, 对称 int4。
// 默认测 M=1，可扫小 M（MT 个 token 共享一次权重读）。
//
// 运行：
//   scripts/run.sh 45-w4a8-gemv/w4a8_gemv.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <random>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int GROUP = 128;  // 每 128 个 k 一个 fp32 scale
constexpr double BF16_PEAK = 989.0;

// ===========================================================================
//  W4A16 基线（44 篇最佳，原样搬来做同进程对拍）
// ===========================================================================
union U4 {
  uint4 u;
  bf16 b[8];
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}

template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a16_kernel(const bf16* __restrict__ A,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);
  float* ssm = reinterpret_cast<float*>(smem + (size_t)MT * KK * 2);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 8; i += 256) {
    const int m = i / (KK / 8), c8 = i % (KK / 8);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c8 * 8]) =
        *reinterpret_cast<const uint4*>(&A[(size_t)m * KK + c8 * 8]);
  }
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    ssm[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    U4 xu[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        xu[m][q].u = *reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0 + q * 8]);

    const int g = k0 / GROUP;
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      const uint32_t wcomp[4] = {wv.x, wv.y, wv.z, wv.w};
      float dq[32];
#pragma unroll
      for (int e = 0; e < 16; ++e) {
        const uint32_t byte = (wcomp[e >> 2] >> (8 * (e & 3))) & 0xFF;
        dq[2 * e] = (float)(int)(byte & 0xF) - 8.f;
        dq[2 * e + 1] = (float)(int)(byte >> 4) - 8.f;
      }
      const float s = ssm[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        float part = 0.f;
#pragma unroll
        for (int idx = 0; idx < 32; ++idx)
          part += dq[idx] * __bfloat162float(xu[m][idx >> 3].b[idx & 7]);
        acc[r][m] += s * part;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// ===========================================================================
//  W4A8 + dp4a
// ===========================================================================
// 把一个 uint32（8 个 nibble，n0..n7）展开成两个 int32，每 4 个 nibble 占
// 一个 int32 的 4 个 byte（符号已减 8）。用 PRMT 选字节 + 一次 XOR/SUB 做符号调整。
__device__ __forceinline__ void expand8(uint32_t v, uint32_t& a, uint32_t& b) {
  const uint32_t lo_nib = v & 0x0F0F0F0Fu;            // bytes: n0 n2 n4 n6
  const uint32_t hi_nib = (v >> 4) & 0x0F0F0F0Fu;     // bytes: n1 n3 n5 n7
  const uint32_t a0 = __byte_perm(lo_nib, hi_nib, 0x5140);  // [n0 n1 n2 n3]
  const uint32_t b0 = __byte_perm(lo_nib, hi_nib, 0x7362);  // [n4 n5 n6 n7]
  // 每个 byte 是 offset-binary（值 0..15 = q+8）。用 SIMD 有符号饱和减 8 做
  // 逐 byte 的 q=n-8，避免普通 32-bit 减法在 n<8 时的跨 byte 借位。
  a = __vsubss4(a0, 0x08080808u);
  b = __vsubss4(b0, 0x08080808u);
}

// 激活量化辅助：x 转 int8，per-128-group scale。
// 一个 warp 负责一个 128-element group：lane 处理 4 个元素 → `shfl` 归约 amax
// → 同 warp 写回 int8。避免「一线程一 group」在 M=1 时只有 40 个活跃线程、
// 被访存延迟拖成 ~11µs（比 GEMV 本体还贵）。
__global__ void quant_x_kernel(const bf16* __restrict__ A, int8_t* __restrict__ Aq,
                               float* __restrict__ Axs, int ngroup, int KK) {
  constexpr int EPL = GROUP / 32;  // 每 lane 元素数
  const int lane = threadIdx.x & 31;
  const int gid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (gid >= ngroup) return;
  const int m = gid / (KK / GROUP), g = gid % (KK / GROUP);
  const bf16* x = A + (size_t)m * KK + g * GROUP;
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < EPL; ++t) amax = fmaxf(amax, fabsf(__bfloat162float(x[lane + 32 * t])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
  const float s = amax > 0 ? amax / 127.f : 1.f;
  if (lane == 0) Axs[gid] = s;
  const float inv = 1.f / s;
  int8_t* q = Aq + (size_t)m * KK + g * GROUP;
#pragma unroll
  for (int t = 0; t < EPL; ++t) {
    int v = __float2int_rn(__bfloat162float(x[lane + 32 * t]) * inv);
    v = max(-127, min(127, v));
    q[lane + 32 * t] = (int8_t)v;
  }
}

// W4A8 GEMV：每 warp 一行（RWW 行），warp 内沿 K 并行，dp4a 做整数点积。
//   · lane 负责 32 个 k 的 chunk：16B 权重（32 个 int4）+ 32B 激活（int8）；
//   · 权重展开成 8 个 int32，与激活的 8 个 int32 做 8 次 dp4a；
//   · 每个 chunk 落在单个 scale group 内，整数点积后乘 s_w*s_x。
template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a8b_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  float* sws = reinterpret_cast<float*>(smem + (size_t)MT * KK + (size_t)MT * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    sws[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g = k0 / GROUP;

    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0]);
      const uint4 xa = xp[0], xb = xp[1];
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      uint32_t w8[8];
      expand8(wv.x, w8[0], w8[1]);
      expand8(wv.y, w8[2], w8[3]);
      expand8(wv.z, w8[4], w8[5]);
      expand8(wv.w, w8[6], w8[7]);
      const float sw = sws[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += sw * sxs[m * NGRP + g] * (float)dot;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// W4A8 变体：每 lane 只处理 16 个 k（1 个 uint4 激活 + 1 个 uint2 权重）。
// 目的：让 warp 内相邻 lane 的 x 地址步长 = 16B（而不是 32B），消除 smem 读
// 的 4.5-way bank conflict（ncu 实测 45% shared wavefront 是冲突）。
template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a8c_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 8 / 32;  // 每 lane 16 个 k => K/512
  static_assert((KK / 2) % 8 == 0 && ((KK / 2) / 8) % 32 == 0, "K 必须被 512 整除");
  static_assert(GROUP % 16 == 0, "一个 16-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  float* sws = reinterpret_cast<float*>(smem + (size_t)MT * KK + (size_t)MT * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    sws[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 16;
    const int g = k0 / GROUP;

    uint32_t x4[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4 xa = *reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0]);
      x4[m][0] = xa.x; x4[m][1] = xa.y; x4[m][2] = xa.z; x4[m][3] = xa.w;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint2 wv = __ldcs(reinterpret_cast<const uint2*>(Wp + (size_t)row * (KK / 2) + c * 8));
      uint32_t w4[4];
      expand8(wv.x, w4[0], w4[1]);
      expand8(wv.y, w4[2], w4[3]);
      const float sw = sws[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 4; ++q) dot = __dp4a((int)w4[q], (int)x4[m][q], dot);
        acc[r][m] += sw * sxs[m * NGRP + g] * (float)dot;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// W4A8 变体 e：在 b（32k chunk）基础上给权重做寄存器双缓冲预取。
// ncu 显示 b 的最大 stall 是 `long_scoreboard` 38.9%（等 global 权重 load），
// 把下一 chunk 的 uint4 提前发出去，与当前 chunk 的展开/dp4a 重叠。
template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a8e_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  float* sws = reinterpret_cast<float*>(smem + (size_t)MT * KK + (size_t)MT * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    sws[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  // 预取第一个 chunk
  uint4 wcur[RWW];
#pragma unroll
  for (int r = 0; r < RWW; ++r) {
    const int row = row0 + warp * RWW + r;
    wcur[r] = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + lane * 16));
  }

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g = k0 / GROUP;

    uint4 wnext[RWW];
    if (i + 1 < NCHUNK) {
#pragma unroll
      for (int r = 0; r < RWW; ++r) {
        const int row = row0 + warp * RWW + r;
        wnext[r] = __ldcs(
            reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + (c + 32) * 16));
      }
    }

    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0]);
      const uint4 xa = xp[0], xb = xp[1];
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      uint32_t w8[8];
      expand8(wcur[r].x, w8[0], w8[1]);
      expand8(wcur[r].y, w8[2], w8[3]);
      expand8(wcur[r].z, w8[4], w8[5]);
      expand8(wcur[r].w, w8[6], w8[7]);
      const float sw = sws[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += sw * sxs[m * NGRP + g] * (float)dot;
      }
      wcur[r] = wnext[r];
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// W4A8 变体 g：可调 warp 数（NWARP）。ncu 显示 grid 1088 = 2.06 个 wave
// （同规模纯读 kernel 在 2.06 wave 下只有 66.9%，而单 wave 的 528 块能到 74.7%）。
// 把 block 缩到 128 线程（4 warp），寄存器限制下每 SM 可放 16 个 block →
// 并发 CTA 翻倍，让 grid 落进单 wave。
template <int RWW, int MT, int KK, int NWARP>
__global__ void __launch_bounds__(NWARP * 32) gemv_a8g_kernel(const int8_t* __restrict__ Aq,
                                                              const float* __restrict__ Axs,
                                                              const uint8_t* __restrict__ Wp,
                                                              const float* __restrict__ Ws,
                                                              float* __restrict__ C, int M, int N,
                                                              int K) {
  constexpr int NT = NWARP * 32;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  float* sws = reinterpret_cast<float*>(smem + (size_t)MT * KK + (size_t)MT * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 16; i += NT) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += NT) sxs[i] = Axs[i];
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += NT) {
    const int n = i / NGRP, g = i % NGRP;
    sws[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g = k0 / GROUP;

    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0]);
      const uint4 xa = xp[0], xb = xp[1];
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      uint32_t w8[8];
      expand8(wv.x, w8[0], w8[1]);
      expand8(wv.y, w8[2], w8[3]);
      expand8(wv.z, w8[4], w8[5]);
      expand8(wv.w, w8[6], w8[7]);
      const float sw = sws[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += sw * sxs[m * NGRP + g] * (float)dot;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// W4A8 变体 h：不做逐 byte 的符号展开（`__vsubss4` 在 sm_90 上被 ptxas 展开成
// 大量 LOP3/PRMT，是 ALU 的主开销）。改为「unsigned nibble + 解析修正」：
//   q_w = u_w - 8, u_w∈[0,15]；把 u_w 当正 int8 直接喂 dp4a，得到
//   dot = Σ u_w·q_x = Σ(q_w+8)·q_x = S + 8·Σq_x，于是 S = dot - 8·Σq_x。
//   Σq_x 只依赖激活、与行无关，每 chunk 用 `__dp4a(ones, x)` 现算一次即可。
__device__ __forceinline__ void expand8u(uint32_t v, uint32_t& a, uint32_t& b) {
  const uint32_t lo_nib = v & 0x0F0F0F0Fu;
  const uint32_t hi_nib = (v >> 4) & 0x0F0F0F0Fu;
  a = __byte_perm(lo_nib, hi_nib, 0x5140);
  b = __byte_perm(lo_nib, hi_nib, 0x7362);
}

template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_a8h_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  float* sws = reinterpret_cast<float*>(smem + (size_t)MT * KK + (size_t)MT * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  const int row0 = blockIdx.x * BN;
  for (int i = tid; i < BN * NGRP; i += 256) {
    const int n = i / NGRP, g = i % NGRP;
    sws[n * NGRP + g] = Ws[(size_t)(row0 + n) * NGRP + g];
  }
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g = k0 / GROUP;

    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0]);
      const uint4 xa = xp[0], xb = xp[1];
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }
    // Σ q_x（与行无关）
    int sx[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      int s = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) s = __dp4a(0x01010101, (int)x8[m][q], s);
      sx[m] = 8 * s;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row0 + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      uint32_t w8[8];
      expand8u(wv.x, w8[0], w8[1]);
      expand8u(wv.y, w8[2], w8[3]);
      expand8u(wv.z, w8[4], w8[5]);
      expand8u(wv.w, w8[6], w8[7]);
      const float sw = sws[(warp * RWW + r) * NGRP + g];
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += sw * sxs[m * NGRP + g] * (float)(dot - sx[m]);
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row0 + warp * RWW + r] = v;
    }
}

// 只读权重（int4）+ scale 的上界（同 44 roof 口径，便于对比）
__global__ void wread_roofline_kernel(const uint8_t* __restrict__ Wp,
                                      const float* __restrict__ Ws, float* __restrict__ out,
                                      int N, int K) {
  const size_t nword = (size_t)N * (K / 2) / 16;
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nword;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(Wp + i * 16);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  const size_t ns = ((size_t)N * (K / GROUP)) / 8;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ns;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = *reinterpret_cast<const float4*>(Ws + i * 4);
    acc ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^
           __float_as_uint(v.w);
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}

// ===========================================================================
//  host
// ===========================================================================
static bf16 f2b(float x) { return __float2bfloat16(x); }

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  int M = (argc > 1) ? std::atoi(argv[1]) : 1;
  int N = (argc > 2) ? std::atoi(argv[2]) : 17408;
  int K = (argc > 3) ? std::atoi(argv[3]) : 5120;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  const double w_bytes_int4 = (double)N * K / 2;
  const double s_bytes = (double)N * (K / GROUP) * 4;
  const double total_w = w_bytes_int4 + s_bytes;
  const double x_bytes = (double)M * K;  // int8 激活
  std::printf("\nW4A8 GEMV: M=%d N=%d K=%d group=%d\n", M, N, K, GROUP);
  std::printf("FLOPs = %.3f GFLOP ; W_int4 = %.1f MB ; scales = %.1f MB ; W+scale = %.1f MB ; "
              "x_int8 = %.2f MB\n\n",
              flops / 1e9, w_bytes_int4 / 1e6, s_bytes / 1e6, total_w / 1e6, x_bytes / 1e6);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<bf16> hA((size_t)M * K);
  std::vector<float> hWf((size_t)N * K), hWdeq((size_t)N * K);
  std::vector<uint8_t> hWp((size_t)N * (K / 2));
  std::vector<float> hWs((size_t)N * (K / GROUP));
  std::vector<float> hC((size_t)M * N);
  for (auto& v : hA) v = f2b(0.5f * dist(rng));
  for (auto& v : hWf) v = 0.1f * dist(rng);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k)
        amax = std::max(amax, std::fabs(hWf[(size_t)n * K + g * GROUP + k]));
      const float s = amax > 0 ? amax / 7.f : 1.f;
      hWs[(size_t)n * (K / GROUP) + g] = s;
      for (int k = 0; k < GROUP; ++k) {
        const size_t idx = (size_t)n * K + g * GROUP + k;
        int q = std::max(-8, std::min(7, (int)std::lround(hWf[idx] / s)));
        hWdeq[idx] = q * s;
        const size_t pk = (size_t)n * (K / 2) + (g * GROUP + k) / 2;
        if (((g * GROUP + k) & 1) == 0)
          hWp[pk] = (uint8_t)((hWp[pk] & 0xF0) | ((q + 8) & 0xF));
        else
          hWp[pk] = (uint8_t)((hWp[pk] & 0x0F) | (((q + 8) & 0xF) << 4));
      }
    }

  bf16* dA;
  int8_t* dAq;
  float* dAxs;
  uint8_t* dWp;
  float *dWs, *dC, *dOut;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&dAq, (size_t)M * K));
  CUDA_CHECK(cudaMalloc(&dAxs, (size_t)M * (K / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dWp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&dWs, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)M * N * 4));
  CUDA_CHECK(cudaMalloc(&dOut, (size_t)N * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWs, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));

  auto cpu_ref = [&](int m, int n) {
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)__bfloat162float(hA[(size_t)m * K + k]) * (double)hWdeq[(size_t)n * K + k];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 32; ++i) {
      const int m = (i * 1009 + 13) % M, n = (i * 997 + 7) % N;
      const double e = cpu_ref(m, n);
      err = std::max(err, std::fabs((double)hC[(size_t)m * N + n] - e));
      ref = std::max(ref, std::fabs(e));
    }
    const bool ok = err / std::max(ref, 1e-6) < 3e-2;
    std::printf("  [%-22s] max_abs_err=%.3e (ref~%.2f) rel=%.2e %s\n", tag, err, ref,
                err / std::max(ref, 1e-6), ok ? "OK" : "FAIL");
    return ok;
  };
  auto report = [&](const char* tag, double ms) {
    const double tf = to_tflops(flops, ms);
    const double wbw = w_bytes_int4 * 1000.0 / ms / 1e9;
    const double tbw = total_w * 1000.0 / ms / 1e9;
    std::printf("%-26s %9.4f ms  %8.2f TFLOPS (%4.1f%% bf16)  W-bw %7.1f GB/s  W+s %7.1f (%4.1f%% HBM)\n",
                tag, ms, tf, 100.0 * tf / BF16_PEAK, wbw, tbw, 100.0 * tbw / d.mem_bw_gbps);
  };
  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  if (want("roof")) {
    const int nthr = 256;
    const int grids[] = {528, 1088, 1188, 2112, 4224};
    for (int nb : grids) {
      char nm[48];
      std::snprintf(nm, sizeof(nm), "roof_g%d", nb);
      auto launch = [&] { wread_roofline_kernel<<<nb, nthr>>>(dWp, dWs, dOut, N, K); };
      launch();
      CUDA_CHECK_LAST();
      report(nm, bench_ms(launch, 3, 30));
    }
  }

  // 量化激活（覆盖全部 M 行）
  {
    const int ngroup = M * (K / GROUP);
    // 256 线程 = 8 warp，每 warp 一个 group
    auto launch = [&] { quant_x_kernel<<<div_up(ngroup, 8), 256>>>(dA, dAq, dAxs, ngroup, K); };
    launch();
    CUDA_CHECK_LAST();
    const double qms = bench_ms(launch, 3, 50);
    std::printf("quant_x (M=%d, %d groups): %.5f ms  (W4A8 的额外前置 kernel)\n\n", M, ngroup,
                qms);
  }

#define RUN_A16(NAME, RWW, MT)                                                                     \
    do {                                                                                           \
      constexpr int BN_ = 8 * (RWW);                                                               \
      const size_t shm = (size_t)(MT) * K * 2 + (size_t)BN_ * (K / GROUP) * 4;                     \
      auto fn = gemv_a16_kernel<RWW, MT, 5120>;                                                    \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dA, dWp, dWs, dC, M, N, K); };                 \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

#define RUN_A8(NAME, RWW, MT)                                                                      \
    do {                                                                                           \
      constexpr int BN_ = 8 * (RWW);                                                               \
      const size_t shm = (size_t)(MT) * K + (size_t)(MT) * (K / GROUP) * 4 +                       \
                         (size_t)BN_ * (K / GROUP) * 4;                                            \
      auto fn = gemv_a8b_kernel<RWW, MT, 5120>;                                                    \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dAq, dAxs, dWp, dWs, dC, M, N, K); };          \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

  if (M == 1 && (want("all") || want("a16"))) {
    RUN_A16("w4a16_r1", 1, 1);
    RUN_A16("w4a16_r2", 2, 1);
    RUN_A16("w4a16_r4", 4, 1);
  }
  if (M == 1 && (want("all") || want("a8"))) {
    RUN_A8("w4a8_r1", 1, 1);
    RUN_A8("w4a8_r2", 2, 1);
    RUN_A8("w4a8_r4", 4, 1);
    RUN_A8("w4a8_r8", 8, 1);
  }
  if (M % 2 == 0 && (want("all") || want("mt2"))) RUN_A8("w4a8_r4_mt2", 4, 2);
  if (M % 4 == 0 && (want("all") || want("mt4"))) RUN_A8("w4a8_r2_mt4", 2, 4);

#define RUN_A8C(NAME, RWW, MT)                                                                     \
    do {                                                                                           \
      constexpr int BN_ = 8 * (RWW);                                                               \
      const size_t shm = (size_t)(MT) * K + (size_t)(MT) * (K / GROUP) * 4 +                       \
                         (size_t)BN_ * (K / GROUP) * 4;                                            \
      auto fn = gemv_a8c_kernel<RWW, MT, 5120>;                                                    \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dAq, dAxs, dWp, dWs, dC, M, N, K); };          \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

  if (M == 1 && (want("all") || want("c8"))) {
    RUN_A8C("w4a8c_r1", 1, 1);
    RUN_A8C("w4a8c_r2", 2, 1);
    RUN_A8C("w4a8c_r4", 4, 1);
    RUN_A8C("w4a8c_r8", 8, 1);
  }

#define RUN_A8E(NAME, RWW, MT)                                                                     \
    do {                                                                                           \
      constexpr int BN_ = 8 * (RWW);                                                               \
      const size_t shm = (size_t)(MT) * K + (size_t)(MT) * (K / GROUP) * 4 +                       \
                         (size_t)BN_ * (K / GROUP) * 4;                                            \
      auto fn = gemv_a8e_kernel<RWW, MT, 5120>;                                                    \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dAq, dAxs, dWp, dWs, dC, M, N, K); };          \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

  if (M == 1 && (want("all") || want("e8"))) {
    RUN_A8E("w4a8e_r1", 1, 1);
    RUN_A8E("w4a8e_r2", 2, 1);
    RUN_A8E("w4a8e_r4", 4, 1);
    RUN_A8E("w4a8e_r8", 8, 1);
  }

#define RUN_A8G(NAME, RWW, MT, NW)                                                                 \
    do {                                                                                           \
      constexpr int BN_ = (NW) * (RWW);                                                            \
      const size_t shm = (size_t)(MT) * K + (size_t)(MT) * (K / GROUP) * 4 +                       \
                         (size_t)BN_ * (K / GROUP) * 4;                                            \
      auto fn = gemv_a8g_kernel<RWW, MT, 5120, NW>;                                                \
      auto launch = [&] { fn<<<N / BN_, (NW) * 32, shm>>>(dAq, dAxs, dWp, dWs, dC, M, N, K); };    \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

  if (M == 1 && (want("all") || want("g128"))) {
    RUN_A8G("w4a8g_w4_r2", 2, 1, 4);
    RUN_A8G("w4a8g_w4_r4", 4, 1, 4);
    RUN_A8G("w4a8g_w4_r8", 8, 1, 4);
    RUN_A8G("w4a8g_w2_r8", 8, 1, 2);
    RUN_A8G("w4a8g_w2_r16", 16, 1, 2);
    RUN_A8G("w4a8g_w1_r32", 32, 1, 1);
  }

#define RUN_A8H(NAME, RWW, MT)                                                                     \
    do {                                                                                           \
      constexpr int BN_ = 8 * (RWW);                                                               \
      const size_t shm = (size_t)(MT) * K + (size_t)(MT) * (K / GROUP) * 4 +                       \
                         (size_t)BN_ * (K / GROUP) * 4;                                            \
      auto fn = gemv_a8h_kernel<RWW, MT, 5120>;                                                    \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dAq, dAxs, dWp, dWs, dC, M, N, K); };          \
      launch();                                                                                    \
      CUDA_CHECK_LAST();                                                                           \
      check(NAME);                                                                                 \
      report(NAME, bench_ms(launch, 5, 50));                                                       \
    } while (0)

  if (want("all") || want("h8")) {
    if (M == 1) {
      RUN_A8H("w4a8h_r1", 1, 1);
      RUN_A8H("w4a8h_r2", 2, 1);
      RUN_A8H("w4a8h_r4", 4, 1);
      RUN_A8H("w4a8h_r8", 8, 1);
    }
    if (M % 2 == 0) RUN_A8H("w4a8h_r4_mt2", 4, 2);
    if (M % 4 == 0) RUN_A8H("w4a8h_r2_mt4", 2, 4);
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dAq));
  CUDA_CHECK(cudaFree(dAxs));
  CUDA_CHECK(cudaFree(dWp));
  CUDA_CHECK(cudaFree(dWs));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dOut));
  return 0;
}
