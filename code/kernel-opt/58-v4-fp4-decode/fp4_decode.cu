// 58（主题 26m）：DeepSeek-V4-Pro routed expert 的 FP4 decode GEMV —— 把 M=1 的
// 权重带宽从 35% 推向 roof：split-K 加并行度 + 去 LUT bank conflict。
//
// 承接 56 篇：真实 checkpoint 的 routed experts 是 FP4 e2m1 + E8M0 block-32；
//   56 的 M=1 GEMV 用「256 项 uint16 smem LUT（1 byte -> 2 int8）」解码 +
//   `__dp4a`，拿到 0.0100 ms / 1165.6 GB/s / 34.8% HBM（纯读 roof 86%）。
//   56 自己指出下一堵墙是 LUT 的 shared bank conflict
//   （59% 多余 wavefront、`short_scoreboard` 4.54）与 `long_scoreboard` 5.96。
//
// 本轮先诊断：56 的 kernel `Waves Per SM = 0.48`（N=3072、BN=8 -> 只有 384 个
//   CTA，132 SM 根本喂不满），是延迟/并行度受限，不只是 bank conflict。
//
// 本文件三个 kernel（一次干净的消融）：
//   DECODE=1  LUT    ：56 原样（baseline）
//   DECODE=2  PRMT   ：把 nibble 用 `__byte_perm` 展开成 byte，再经 8 项常量表
//                      （2 个 PRMT）映射成 int8 —— 彻底不碰 smem（无 bank conflict）
//   + KSPLIT：沿 K 切，grid.y = KSPLIT，`atomicAdd` 归约，把 waves 抬到 ≥1
//   + 预取：block 内把下一个 16B 权重先读出来，提升 MLP
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, expert_dtype=fp4（w1 [3072,7168]）。
//
// 运行：scripts/run.sh 58-v4-fp4-decode/fp4_decode.cu [MT]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int H = 7168;    // hidden
constexpr int I = 3072;    // moe_intermediate
constexpr int GROUP = 128; // 激活量化组

// ---------------------------------------------------------------------------
// e2m1 解码
// ---------------------------------------------------------------------------
__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}

// 算术解码（56 v1）：一个 uint32（8 nibble）-> 两个 uint32（各 4 个 int8）
__device__ __forceinline__ void expand_arith(uint32_t w, uint32_t& a, uint32_t& b) {
  uint32_t o0 = 0, o1 = 0;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int v0 = e2m1_i8((w >> (4 * q)) & 0xF) & 0xFF;
    const int v1 = e2m1_i8((w >> (4 * q + 16)) & 0xF) & 0xFF;
    o0 |= (uint32_t)v0 << (8 * q);
    o1 |= (uint32_t)v1 << (8 * q);
  }
  a = o0; b = o1;
}

// LUT 解码（56 v2）：一个 uint32 -> 两个 uint32，靠 256 项 uint16 smem 表
__device__ __forceinline__ void expand_lut(uint32_t w, uint32_t& a, uint32_t& b,
                                           const uint16_t* __restrict__ lut) {
  const uint32_t r0 = lut[w & 0xFF], r1 = lut[(w >> 8) & 0xFF];
  const uint32_t r2 = lut[(w >> 16) & 0xFF], r3 = lut[(w >> 24) & 0xFF];
  a = r0 | (r1 << 16);
  b = r2 | (r3 << 16);
}

// PRMT 解码：e2m1 nibble -> 2*value 的 int8，全在寄存器里，零 smem 访问。
//   magnitude 表 f[0..7] = {0,1,2,3,4,6,8,12}（= 2*{0,.5,1,1.5,2,3,4,6}），
//   负表就是 -f。先 `__byte_perm` 把 nibble 摊成 byte，再两次 PRMT 查表 + 选符号。
__device__ __forceinline__ uint32_t decode4(uint32_t e) {
  const uint32_t k = e & 0x07070707u;         // 幅值码 0..7（每 byte）
  const uint32_t s = (e >> 3) & 0x01010101u;  // 符号位（每 byte）
  // 把 4 个 byte 的低 3 位 / 符号位压成 4 个 nibble 的 PRMT selector
  const uint32_t pk = (k & 0x0000000Fu) | ((k >> 4) & 0x000000F0u) |
                      ((k >> 8) & 0x00000F00u) | ((k >> 12) & 0x0000F000u);
  const uint32_t ps = (s & 0x1u) | ((s >> 4) & 0x10u) | ((s >> 8) & 0x100u) |
                      ((s >> 12) & 0x1000u);
  const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, pk);   // f[code]
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, pk);   // -f[code]
  const uint32_t sel = 0x3210u | (ps << 2);                          // i + 4*sign
  return __byte_perm(pos, neg, sel);
}
__device__ __forceinline__ void expand_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  const uint32_t lo = w & 0x0F0F0F0Fu;
  const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
  a = decode4(__byte_perm(lo, hi, 0x5140));  // n0 n1 n2 n3
  b = decode4(__byte_perm(lo, hi, 0x7362));  // n4 n5 n6 n7
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}

// ---------------------------------------------------------------------------
// FP4 GEMV：每 warp RWW 行，warp 内沿 K 并行（lane 一个 16B chunk = 32 个 fp4），
//   dp4a 整数点积；E8M0 的 32-k scale 与激活 per-128 scale 在 chunk 内折算。
//   KS：沿 K 切成 KS 段（grid.y），每段各自 atomicAdd 到 C。
//   DECODE：1=LUT / 2=PRMT / 0=算术。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, int KS, int DECODE>
__global__ void __launch_bounds__(256) gemv_fp4_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;   // 激活 scale 组数
  constexpr int NS32 = KK / 32;      // 权重 scale 组数（block 32）
  constexpr int CH = (KK / 2) / 16;  // 每行 16B 权重 chunk 数 = 32-k 组数
  static_assert(CH % KS == 0, "K 的 16B chunk 数必须被 KSPLIT 整除");

  const int span = CH / KS;
  const int base = blockIdx.y * span;  // 本 block 负责的 chunk 区间 [base, base+span)
  const int KN = span * 32;            // 本 block 负责的 k 数

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KN);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (DECODE == 1) {
    for (int i = tid; i < 256; i += 256)
      lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
  }
  // 只加载本 block 的激活 k-slice（避免 split-K 重复读整段激活）
  for (int i = tid; i < MT * KN / 16; i += 256) {
    const int m = i / (KN / 16), c16 = i % (KN / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KN + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + base * 32 + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  const int row_base = blockIdx.x * BN + warp * RWW;

  auto load_w = [&](int u) {
    return __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)(row_base) * (KK / 2) + u * 16));
  };

  // 预取：先把每个 lane 的第一个 chunk 读出来（clamp 防越界）
  uint4 wv_cur = load_w((base + lane < CH) ? (base + lane) : (CH - 1));

  for (int u = base + lane; u < base + span; u += 32) {
    const uint4 wv = wv_cur;
    if (u + 32 < base + span) wv_cur = load_w(u + 32);
    const int c = u;  // 16B chunk 下标 == 32-k scale 组
    const int k0 = c * 32;
    const int g128 = k0 / GROUP;

    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4* xp = reinterpret_cast<const uint4*>(&xs[(size_t)m * KN + (k0 - base * 32)]);
      const uint4 xa = xp[0], xb = xp[1];
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }

#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = row_base + r;
      uint32_t w8[8];
      const uint4 w = (r == 0) ? wv
                               : __ldcs(reinterpret_cast<const uint4*>(
                                     Wp + (size_t)row * (KK / 2) + c * 16));
      if (DECODE == 1) {
        expand_lut(w.x, w8[0], w8[1], lut);
        expand_lut(w.y, w8[2], w8[3], lut);
        expand_lut(w.z, w8[4], w8[5], lut);
        expand_lut(w.w, w8[6], w8[7], lut);
      } else if (DECODE == 2) {
        expand_prmt(w.x, w8[0], w8[1]);
        expand_prmt(w.y, w8[2], w8[3]);
        expand_prmt(w.z, w8[4], w8[5]);
        expand_prmt(w.w, w8[6], w8[7]);
      } else {
        expand_arith(w.x, w8[0], w8[1]);
        expand_arith(w.y, w8[2], w8[3]);
        expand_arith(w.z, w8[4], w8[5]);
        expand_arith(w.w, w8[6], w8[7]);
      }
      const float sw = __ldg(&Ws[(size_t)row * NS32 + c]);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += 0.5f * sw * sxs[m * NGRP + g128] * (float)dot;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) {
        if (KS == 1) C[(size_t)m * N + row_base + r] = v;
        else atomicAdd(&C[(size_t)m * N + row_base + r], v);
      }
    }
}

// ---------------------------------------------------------------------------
// v4：把每个 lane 负责的 NITER 个 16B 权重 chunk **一次性全部预取**到寄存器，
//     再解码 —— 用满 MLP（每 lane NITER 个 16B 在飞）。RWW 行/ warp 复用激活。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, int DECODE>
__global__ void __launch_bounds__(256) gemv_fp4_preload(const int8_t* __restrict__ Aq,
                                                        const float* __restrict__ Axs,
                                                        const uint8_t* __restrict__ Wp,
                                                        const float* __restrict__ Ws,
                                                        float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NS32 = KK / 32;
  constexpr int CH = (KK / 2) / 16;   // 224
  constexpr int NITER = CH / 32;      // 每 lane 的 chunk 数（=7）
  static_assert(CH % 32 == 0, "CH 必须被 32 整除");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (DECODE == 1) {
    for (int i = tid; i < 256; i += 256)
      lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
  }
  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];

  const int row_base = blockIdx.x * BN + warp * RWW;
  // 全量预取：RWW 行的所有 chunk
  uint4 wv[RWW][NITER];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int i = 0; i < NITER; ++i)
      wv[r][i] = __ldcs(reinterpret_cast<const uint4*>(
          Wp + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * i) * 16));
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NITER; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g128 = k0 / GROUP;
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
      const uint4 w = wv[r][i];
      uint32_t w8[8];
      if (DECODE == 1) {
        expand_lut(w.x, w8[0], w8[1], lut);
        expand_lut(w.y, w8[2], w8[3], lut);
        expand_lut(w.z, w8[4], w8[5], lut);
        expand_lut(w.w, w8[6], w8[7], lut);
      } else if (DECODE == 2) {
        expand_prmt(w.x, w8[0], w8[1]);
        expand_prmt(w.y, w8[2], w8[3]);
        expand_prmt(w.z, w8[4], w8[5]);
        expand_prmt(w.w, w8[6], w8[7]);
      } else {
        expand_arith(w.x, w8[0], w8[1]);
        expand_arith(w.y, w8[2], w8[3]);
        expand_arith(w.z, w8[4], w8[5]);
        expand_arith(w.w, w8[6], w8[7]);
      }
      const float sw = __ldg(&Ws[(size_t)(row_base + r) * NS32 + c]);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += 0.5f * sw * sxs[m * NGRP + g128] * (float)dot;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row_base + r] = v;
    }
}

// ---------------------------------------------------------------------------
// v5：流水预取 —— 每个 lane 维护 DEPTH 个在飞的 16B 权重（软件流水），
//     在解码第 i 个 chunk 时发出第 i+DEPTH 个的 load。DEPTH 越大 MLP 越高，
//     但寄存器越多。RWW 行/ warp 让一次 activation 读服务多行。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, int DEPTH, int DECODE>
__global__ void __launch_bounds__(256) gemv_fp4_pipe(const int8_t* __restrict__ Aq,
                                                     const float* __restrict__ Axs,
                                                     const uint8_t* __restrict__ Wp,
                                                     const float* __restrict__ Ws,
                                                     float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NS32 = KK / 32;
  constexpr int NITER = (KK / 2) / 16 / 32;  // =7

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  if (DECODE == 1) {
    for (int i = tid; i < 256; i += 256)
      lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
  }
  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  __syncthreads();

  const int row_base = blockIdx.x * BN + warp * RWW;
  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  uint4 ring[RWW][DEPTH];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int d = 0; d < DEPTH; ++d)
      ring[r][d] = __ldcs(reinterpret_cast<const uint4*>(
          Wp + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * (d < NITER ? d : NITER - 1)) * 16));

#pragma unroll
  for (int i = 0; i < NITER; ++i) {
    const int c = lane + 32 * i;
    const int k0 = c * 32;
    const int g128 = k0 / GROUP;
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
      const uint4 w = ring[r][i % DEPTH];
      if (i + DEPTH < NITER)
        ring[r][i % DEPTH] = __ldcs(reinterpret_cast<const uint4*>(
            Wp + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * (i + DEPTH)) * 16));
      uint32_t w8[8];
      if (DECODE == 1) {
        expand_lut(w.x, w8[0], w8[1], lut); expand_lut(w.y, w8[2], w8[3], lut);
        expand_lut(w.z, w8[4], w8[5], lut); expand_lut(w.w, w8[6], w8[7], lut);
      } else if (DECODE == 2) {
        expand_prmt(w.x, w8[0], w8[1]); expand_prmt(w.y, w8[2], w8[3]);
        expand_prmt(w.z, w8[4], w8[5]); expand_prmt(w.w, w8[6], w8[7]);
      } else {
        expand_arith(w.x, w8[0], w8[1]); expand_arith(w.y, w8[2], w8[3]);
        expand_arith(w.z, w8[4], w8[5]); expand_arith(w.w, w8[6], w8[7]);
      }
      const float sw = __ldg(&Ws[(size_t)(row_base + r) * NS32 + c]);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += 0.5f * sw * sxs[m * NGRP + g128] * (float)dot;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + row_base + r] = v;
    }
}

// ---------------------------------------------------------------------------
// v6：无 smem / 无 __syncthreads —— 激活直接从 global 读（7KB 常驻 L2），
//     PRMT 解码（自带寄存器表）。block 可以做得又多又小，纯靠 warp 数堆并发度。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, int KS>
__global__ void __launch_bounds__(256) gemv_fp4_nosmem(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NS32 = KK / 32;
  constexpr int CH = (KK / 2) / 16;
  static_assert(CH % KS == 0, "CH %% KS");

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int span = CH / KS;
  const int base = blockIdx.y * span;
  const int row_base = blockIdx.x * BN + warp * RWW;

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

  uint4 wv[RWW];
#pragma unroll
  for (int r = 0; r < RWW; ++r) {
    const int u = base + lane;
    wv[r] = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)(row_base + r) * (KK / 2) + u * 16));
  }

  for (int u = base + lane; u < base + span; u += 32) {
    const int c = u;
    const int k0 = c * 32;
    const int g128 = k0 / GROUP;
    uint32_t x8[MT][8];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const uint4 xa = __ldg(reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + k0]));
      const uint4 xb = __ldg(reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + k0 + 16]));
      x8[m][0] = xa.x; x8[m][1] = xa.y; x8[m][2] = xa.z; x8[m][3] = xa.w;
      x8[m][4] = xb.x; x8[m][5] = xb.y; x8[m][6] = xb.z; x8[m][7] = xb.w;
    }
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      uint32_t w8[8];
      expand_prmt(wv[r].x, w8[0], w8[1]);
      expand_prmt(wv[r].y, w8[2], w8[3]);
      expand_prmt(wv[r].z, w8[4], w8[5]);
      expand_prmt(wv[r].w, w8[6], w8[7]);
      const float sw = __ldg(&Ws[(size_t)(row_base + r) * NS32 + c]);
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        acc[r][m] += 0.5f * sw * __ldg(&Axs[m * NGRP + g128]) * (float)dot;
      }
    }
    // 预取下一个 chunk
    if (u + 32 < base + span) {
#pragma unroll
      for (int r = 0; r < RWW; ++r)
        wv[r] = __ldcs(reinterpret_cast<const uint4*>(
            Wp + (size_t)(row_base + r) * (KK / 2) + (u + 32) * 16));
    }
  }
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) {
        if (KS == 1) C[(size_t)m * N + row_base + r] = v;
        else atomicAdd(&C[(size_t)m * N + row_base + r], v);
      }
    }
}

// 纯读 roof
__global__ void read_fp4_kernel(const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
                                float* __restrict__ out, int N, int K) {
  const size_t nword = (size_t)N * (K / 2) / 16;
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nword;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(Wp + i * 16);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  const size_t ns = ((size_t)N * (K / 32)) / 8;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ns;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = *reinterpret_cast<const float4*>(Ws + i * 4);
    a ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^ __float_as_uint(v.w);
  }
  if (a == 0xDEADBEEFu) out[0] = 1.f;
}
__global__ void read_bytes_kernel(const uint8_t* __restrict__ p, float* __restrict__ out, size_t n) {
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 16;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + i * 16);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0xDEADBEEFu) out[0] = 1.f;
}

// 激活量化（per-128 int8），同 45/56
__global__ void quant_x_kernel(const bf16* __restrict__ A, int8_t* __restrict__ Aq,
                               float* __restrict__ Axs, int ngroup, int KK) {
  const int lane = threadIdx.x & 31;
  const int gid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
  if (gid >= ngroup) return;
  const int m = gid / (KK / GROUP), g = gid % (KK / GROUP);
  const bf16* x = A + (size_t)m * KK + g * GROUP;
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < GROUP / 32; ++t) amax = fmaxf(amax, fabsf(__bfloat162float(x[lane + 32 * t])));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
  const float s = amax > 0 ? amax / 127.f : 1.f;
  if (lane == 0) Axs[gid] = s;
  const float inv = 1.f / s;
  int8_t* q = Aq + (size_t)m * KK + g * GROUP;
#pragma unroll
  for (int t = 0; t < GROUP / 32; ++t) {
    int v = __float2int_rn(__bfloat162float(x[lane + 32 * t]) * inv);
    v = max(-127, min(127, v));
    q[lane + 32 * t] = (int8_t)v;
  }
}

// host
static std::vector<uint8_t> load_bin(const char* path, size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
  std::vector<uint8_t> v(expect);
  size_t got = std::fread(v.data(), 1, expect, f);
  std::fclose(f);
  if (got != expect) { std::fprintf(stderr, "read %s got %zu/%zu\n", path, got, expect); std::exit(1); }
  return v;
}
__host__ __device__ __forceinline__ float e2m1f(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  float v = (e == 0) ? (m * 0.5f) : ((1.f + m * 0.5f) * exp2f((int)e - 1));
  return s ? -v : v;
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int MT = (argc > 1) ? std::atoi(argv[1]) : 1;
  const int N = I, K = H;
  const size_t wpc = (size_t)N * (K / 2);
  const size_t wsc = (size_t)N * (K / 32);
  const size_t wf8c = (size_t)N * K;
  const size_t wbfc = (size_t)N * K * 2;

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nV4-Pro routed expert w1 [N=%d, K=%d]  M=%d\n", N, K, MT);
  std::printf("weight bytes: fp4=%.2f MB (%.1f%%)  fp8=%.2f MB  bf16=%.2f MB\n", wpc / 1e6,
              100.0 * wpc / wbfc, wf8c / 1e6, wbfc / 1e6);

  auto wpack = load_bin("w1_fp4.bin", wpc);
  auto wscal = load_bin("w1_scale.bin", wsc);
  auto wfp8 = load_bin("w1_fp8.bin", wf8c);
  auto wbf = load_bin("w1_bf16.bin", wbfc);

  std::vector<float> hws((size_t)N * (K / 32));
  for (size_t i = 0; i < hws.size(); ++i) hws[i] = exp2f((int)wscal[i] - 127);

  std::vector<bf16> hx((size_t)MT * K);
  unsigned st = 12345u;
  for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = __float2bfloat16(0.5f * ((st >> 8) / 8388608.0f - 1.f)); }

  uint8_t *dWp, *dW8, *dWbf;
  float *dWs, *dXs, *dC;
  bf16* dX;
  int8_t* dXq;
  CUDA_CHECK(cudaMalloc(&dWp, wpc));
  CUDA_CHECK(cudaMalloc(&dW8, wf8c));
  CUDA_CHECK(cudaMalloc(&dWbf, wbfc));
  CUDA_CHECK(cudaMalloc(&dWs, hws.size() * 4));
  CUDA_CHECK(cudaMalloc(&dX, hx.size() * 2));
  CUDA_CHECK(cudaMalloc(&dXq, (size_t)MT * K));
  CUDA_CHECK(cudaMalloc(&dXs, (size_t)MT * (K / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dC, (size_t)MT * N * 4));
  CUDA_CHECK(cudaMemcpy(dWp, wpack.data(), wpc, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dW8, wfp8.data(), wf8c, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWbf, wbf.data(), wbfc, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWs, hws.data(), hws.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  {
    const int ngroup = MT * (K / GROUP);
    quant_x_kernel<<<(ngroup + 7) / 8, 256>>>(dX, dXq, dXs, ngroup, K);
    CUDA_CHECK_LAST();
  }

  // ---- 纯读 roof ----
  double t4 = bench_ms([&] { read_fp4_kernel<<<1024, 256>>>(dWp, dWs, dC, N, K); }, 5, 50);
  std::printf("\n[roof] read fp4+scale %.4f ms  %.1f GB/s (%.1f%% HBM)\n", t4,
              to_gbps(wpc + wsc, t4), 100 * to_gbps(wpc + wsc, t4) / di.mem_bw_gbps);

  // ---- 正确性：对每个配置查 8 个点 ----
  std::vector<int8_t> hxq_i((size_t)MT * K);
  std::vector<float> hxs((size_t)MT * (K / GROUP));
  CUDA_CHECK(cudaMemcpy(hxq_i.data(), dXq, hxq_i.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hxs.data(), dXs, hxs.size() * 4, cudaMemcpyDeviceToHost));
  auto check = [&](const char* tag, float* out) {
    std::vector<float> hC((size_t)MT * N);
    CUDA_CHECK(cudaMemcpy(hC.data(), out, hC.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0, maxref = 0;
    for (int s = 0; s < 8; ++s) {
      const int row = (s * 353 + 7) % N;
      const int m = s % MT;
      double ref = 0;
      for (int k = 0; k < K; ++k) {
        const unsigned n = (wpack[(size_t)row * (K / 2) + k / 2] >> ((k & 1) * 4)) & 0xF;
        const float w = e2m1f(n) * hws[(size_t)row * (K / 32) + k / 32];
        const float xv = (float)hxq_i[(size_t)m * K + k] * hxs[(size_t)m * (K / GROUP) + k / GROUP];
        ref += (double)w * (double)xv;
      }
      const double got = hC[(size_t)m * N + row];
      const double rel = std::fabs(got - ref) / std::max(std::fabs(ref), 1.0);
      maxrel = std::max(maxrel, rel); maxref = std::max(maxref, std::fabs(ref));
    }
    std::printf("[check %-16s] max_rel=%.3e (ref~%.1f) %s\n", tag, maxrel, maxref,
                maxrel < 3e-2 ? "OK" : "FAIL");
  };

  const int smem = MT * K + MT * (K / GROUP) * 4;
  const double bytes = wpc + wsc;

  auto run = [&](int decode, int ks) {
#define DISP_KS(MT_, KS_)                                                                     \
  do {                                                                                        \
    dim3 g(N / 8, KS_);                                                                       \
    if (decode == 0)                                                                          \
      gemv_fp4_kernel<1, MT_, H, KS_, 0><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    else if (decode == 1)                                                                     \
      gemv_fp4_kernel<1, MT_, H, KS_, 1><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    else                                                                                      \
      gemv_fp4_kernel<1, MT_, H, KS_, 2><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    CUDA_CHECK_LAST();                                                                        \
  } while (0)
#define DISP_MT(MT_)                                                                          \
  switch (ks) {                                                                               \
    case 1: DISP_KS(MT_, 1); break;                                                           \
    case 2: DISP_KS(MT_, 2); break;                                                           \
    case 4: DISP_KS(MT_, 4); break;                                                           \
    case 7: DISP_KS(MT_, 7); break;                                                           \
    case 8: DISP_KS(MT_, 8); break;                                                           \
    default: std::fprintf(stderr, "bad ks\n"); std::exit(1);                                  \
  }
    if (MT == 1) DISP_MT(1)
    else if (MT == 2) DISP_MT(2)
    else DISP_MT(4)
#undef DISP_MT
#undef DISP_KS
  };

  const char* dname[3] = {"arith", "LUT", "PRMT"};
  // ---- v5：流水预取（KS=1，扫 DEPTH / RWW）----
  auto run5 = [&](int rww, int depth, int decode) {
#define DISP_D(RWW_, DEPTH_, MT_)                                                                 \
  do {                                                                                            \
    dim3 g(N / (8 * RWW_));                                                                       \
    if (decode == 1)                                                                              \
      gemv_fp4_pipe<RWW_, MT_, H, DEPTH_, 1><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    else if (decode == 2)                                                                         \
      gemv_fp4_pipe<RWW_, MT_, H, DEPTH_, 2><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    else                                                                                          \
      gemv_fp4_pipe<RWW_, MT_, H, DEPTH_, 0><<<g, 256, smem>>>(dXq, dXs, dWp, dWs, dC, MT, N, K); \
    CUDA_CHECK_LAST();                                                                            \
  } while (0)
#define DISP_MT(RWW_)                                                                             \
  switch (depth) {                                                                                \
    case 1: DISP_D(RWW_, 1, 1); break;                                                            \
    case 2: DISP_D(RWW_, 2, 1); break;                                                            \
    case 3: DISP_D(RWW_, 3, 1); break;                                                            \
    case 4: DISP_D(RWW_, 4, 1); break;                                                            \
    default: std::fprintf(stderr, "bad depth\n"); std::exit(1);                                   \
  }
    if (rww == 1) { DISP_MT(1); } else { DISP_MT(2); }
#undef DISP_MT
#undef DISP_D
  };
  if (MT == 1) {
    std::printf("\n[gemv fp4 v6: 无 smem/sync + PRMT + KSPLIT]\n");
    for (int rww : {1, 2, 4}) {
      for (int ks : {1, 2, 4, 8}) {
        if ((((H / 2) / 16) % ks)) continue;
        CUDA_CHECK(cudaMemset(dC, 0, (size_t)MT * N * 4));
        dim3 g(N / (8 * rww), ks);
        auto launch = [&]() {
          if (rww == 1) gemv_fp4_nosmem<1, 1, H, 1><<<g, 256, 0>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);
          else if (rww == 2) gemv_fp4_nosmem<2, 1, H, 1><<<g, 256, 0>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);
          else gemv_fp4_nosmem<4, 1, H, 1><<<g, 256, 0>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);
        };
        // 用 KS 模板实例
#define L6(R_, K_) gemv_fp4_nosmem<R_, 1, H, K_><<<dim3(N / (8 * R_), K_), 256, 0>>>(dXq, dXs, dWp, dWs, dC, MT, N, K)
        switch (ks) {
          case 1:
            if (rww == 1) L6(1, 1); else if (rww == 2) L6(2, 1); else L6(4, 1);
            break;
          case 2:
            if (rww == 1) L6(1, 2); else if (rww == 2) L6(2, 2); else L6(4, 2);
            break;
          case 4:
            if (rww == 1) L6(1, 4); else if (rww == 2) L6(2, 4); else L6(4, 4);
            break;
          case 8:
            if (rww == 1) L6(1, 8); else if (rww == 2) L6(2, 8); else L6(4, 8);
            break;
        }
        (void)launch;
        CUDA_CHECK_LAST();
        if (rww == 1 && ks == 1) check("v6-PRMT", dC);
        double t = bench_ms([&] {
          if (rww == 1) { if (ks==1) L6(1,1); else if (ks==2) L6(1,2); else if (ks==4) L6(1,4); else L6(1,8); }
          else if (rww == 2) { if (ks==1) L6(2,1); else if (ks==2) L6(2,2); else if (ks==4) L6(2,4); else L6(2,8); }
          else { if (ks==1) L6(4,1); else if (ks==2) L6(4,2); else if (ks==4) L6(4,4); else L6(4,8); }
        }, 5, 200);
        std::printf("  v6 PRMT RWW=%d KS=%d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)\n", rww, ks, t,
                    to_gbps(bytes, t), 100 * to_gbps(bytes, t) / di.mem_bw_gbps);
#undef L6
      }
    }
    std::printf("\n[gemv fp4 v5: 流水预取]\n");
    for (int rww : {1, 2}) {
      for (int d : {1, 2, 3, 4}) {
        for (int dec : {1, 0, 2}) {
          if (rww == 2 && dec != 1) continue;
          CUDA_CHECK(cudaMemset(dC, 0, (size_t)MT * N * 4));
          run5(rww, d, dec);
          if (rww == 1 && d == 2) check(dname[dec], dC);
          double t = bench_ms([&] { run5(rww, d, dec); }, 5, 200);
          std::printf("  %-5s RWW=%d DEPTH=%d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)\n",
                      dname[dec], rww, d, t, to_gbps(bytes, t),
                      100 * to_gbps(bytes, t) / di.mem_bw_gbps);
        }
      }
    }
  }

  // ---- v4：全量预取 ----
  const int smemP = smem;
  auto runP = [&](int rww, int decode) {
#define DISP_RWW(RWW_)                                                                          \
  do {                                                                                          \
    dim3 g(N / (8 * RWW_));                                                                     \
    if (MT == 1) {                                                                              \
      if (decode == 1)                                                                          \
        gemv_fp4_preload<RWW_, 1, H, 1><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else if (decode == 2)                                                                     \
        gemv_fp4_preload<RWW_, 1, H, 2><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else                                                                                      \
        gemv_fp4_preload<RWW_, 1, H, 0><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
    } else if (MT == 2) {                                                                       \
      if (decode == 1)                                                                          \
        gemv_fp4_preload<RWW_, 2, H, 1><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else if (decode == 2)                                                                     \
        gemv_fp4_preload<RWW_, 2, H, 2><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else                                                                                      \
        gemv_fp4_preload<RWW_, 2, H, 0><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
    } else {                                                                                    \
      if (decode == 1)                                                                          \
        gemv_fp4_preload<RWW_, 4, H, 1><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else if (decode == 2)                                                                     \
        gemv_fp4_preload<RWW_, 4, H, 2><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
      else                                                                                      \
        gemv_fp4_preload<RWW_, 4, H, 0><<<g, 256, smemP>>>(dXq, dXs, dWp, dWs, dC, MT, N, K);   \
    }                                                                                           \
    CUDA_CHECK_LAST();                                                                          \
  } while (0)
    if (rww == 1) { DISP_RWW(1); } else { DISP_RWW(2); }
#undef DISP_RWW
  };
  std::printf("\n[gemv fp4 v4: 全量预取]\n");
  for (int rww : {1, 2}) {
    for (int d : {1, 0, 2}) {
      CUDA_CHECK(cudaMemset(dC, 0, (size_t)MT * N * 4));
      runP(rww, d);
      if (MT == 1 && rww == 1) check(dname[d], dC);
      double t = bench_ms([&] { runP(rww, d); }, 5, 200);
      std::printf("  %-5s RWW=%d BN=%3d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)\n", dname[d], rww,
                  8 * rww, t, to_gbps(bytes, t), 100 * to_gbps(bytes, t) / di.mem_bw_gbps);
    }
  }

  // ---- 正确性 ----
  for (int d = 0; d < 3 && MT == 1; ++d) {
    CUDA_CHECK(cudaMemset(dC, 0, (size_t)MT * N * 4));
    run(d, 4);
    check(dname[d], dC);
  }

  // ---- 扫参 ----
  std::printf("\n[gemv fp4] B-read = fp4+scale 字节 / 耗时\n");
  const int chain[3] = {1, 2, 0};  // baseline / LUT / PRMT 顺序
  for (int d : chain) {
    for (int ks : {1, 2, 4, 7, 8}) {
      if ((H / 2) / 16 % ks) continue;
      CUDA_CHECK(cudaMemset(dC, 0, (size_t)MT * N * 4));
      double t = bench_ms([&] { run(d, ks); }, 5, 200);
      std::printf("  %-5s KS=%d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)  waves=%.2f\n", dname[d], ks,
                  t, to_gbps(bytes, t), 100 * to_gbps(bytes, t) / di.mem_bw_gbps,
                  (double)(N / 8) * ks / (132.0 * 6));
    }
  }
  if (argc > 2 && std::strcmp(argv[2], "ncu") == 0) {
    for (int i = 0; i < 3; ++i)
      gemv_fp4_pipe<1, 1, H, 2, 1><<<dim3(N / 8), 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }
  return 0;
}
