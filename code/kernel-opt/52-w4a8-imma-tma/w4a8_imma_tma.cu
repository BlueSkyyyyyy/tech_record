// 52 W4A8 IMMA 小 M GEMM：warp-private + TMA（主题 26k）
//
// 51 篇把 `barrier` 打到零之后，墙换成了 `long_scoreboard`（0.30→1.60）——每个 warp
// 一个 stage 只有 2KB W 在飞，warp 直接暴露在 `cp.async.wait_group` 的 DEPBAR 上。
// 本轮把 W（K-blocked 布局下一个 warp 一个 group = 连续 2KB）交给 TMA 引擎：
// 每个 warp 一个私有 mbarrier + 一个 stage 环，1D `cp.async.bulk` 一把搬入
// （W + sa + sc + sw 四条 bulk、一次 expect_tx），把「多组 cp.async + 每 stage
// 一次 DEPBAR」换成「mbarrier 完成信号」，在飞量由 TMA 兜底。
//
// 与 51 最佳（wA_s2k8 / wA_s3k4）同进程对照。
//
// shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128；对称 int4 权重。
//   N=17408（MLP up/gate），K=5120，测 M=1..16（decode）。
//
// 运行：
//   scripts/run.sh 52-w4a8-imma-tma/w4a8_imma_tma.cu tma
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <functional>
#include <random>
#include <string>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int GROUP = 128;  // 每 128 个 k 一个 fp32 scale
constexpr int LARGE_BK = GROUP;
constexpr int MPAD = 64;    // 转置 scale 布局的行距（AxsT/AsumT: [NGRP][MPAD]）
constexpr int NSM = 132;

// ===========================================================================
//  PTX 小工具（沿用 48）
// ===========================================================================
__device__ __forceinline__ void imma_m16n8k32_u8(int c[4], const uint32_t a[4],
                                                  const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// uint16（2 个 int4 字节 = 4 nibble）→ 4 个 int8（nibble 保持 0..15，配 u8.s8）
__device__ __forceinline__ uint32_t expand4u(uint32_t v) {
  const uint32_t lo = v & 0x0F0Fu;
  const uint32_t hi = (v >> 4) & 0x0F0Fu;
  return __byte_perm(lo, hi, 0x5140);
}

// ===========================================================================
//  激活动态量化（45/48 原样）：x bf16 → int8，per-128-group scale + Σq。
// ===========================================================================
__global__ void quant_x_kernel(const bf16* __restrict__ A, int8_t* __restrict__ Aq,
                               float* __restrict__ Axs, float* __restrict__ Axsum, int ngroup,
                               int KK) {
  constexpr int EPL = GROUP / 32;
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
  const float inv = 1.f / s;
  int8_t* q = Aq + (size_t)m * KK + g * GROUP;
  int lsum = 0;
#pragma unroll
  for (int t = 0; t < EPL; ++t) {
    int v = __float2int_rn(__bfloat162float(x[lane + 32 * t]) * inv);
    v = max(-127, min(127, v));
    q[lane + 32 * t] = (int8_t)v;
    lsum += v;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) lsum += __shfl_xor_sync(~0u, lsum, o);
  if (lane == 0) {
    Axs[gid] = s;
    Axsum[gid] = (float)lsum;
  }
}

// 把 [M][NGRP] 的 Axs / Axsum 转置成 [NGRP][MPAD]，并把 Axsum ×8（u8.s8 的解析修正）。
__global__ void transpose_scale_kernel(const float* __restrict__ Axs,
                                       const float* __restrict__ Axsum,
                                       float* __restrict__ AxsT, float* __restrict__ AsumT8, int M,
                                       int ngrp) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= M * ngrp) return;
  const int m = i / ngrp, g = i % ngrp;
  AxsT[(size_t)g * MPAD + m] = Axs[i];
  AsumT8[(size_t)g * MPAD + m] = 8.f * Axsum[i];
}

// 只读权重（int4）+ scale 的上界（同 44/45/46 的 roof 口径）
__global__ void wread_roofline_kernel(const uint8_t* __restrict__ Wp,
                                      const float* __restrict__ Ws, float* __restrict__ out, int N,
                                      int K) {
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
    acc ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^ __float_as_uint(v.w);
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}

// 用 GEMM 的「strided 访问模式」读同一份权重：每个 (n, stage) 读 64B，
// 行距 K/2 字节。用来验证「DRAM 行缓冲局部性」是不是 GEMM 到不了 roof 的原因。
__global__ void wread_strided_kernel(const uint8_t* __restrict__ Wp, float* __restrict__ out, int N,
                                     int K) {
  const int nstage = K / LARGE_BK;  // 40
  const int chunks = N * nstage * (LARGE_BK / 2 / 16);  // 个 16B 单元
  uint32_t acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < chunks; i += gridDim.x * blockDim.x) {
    const int c = i & 3;
    const int t = i >> 2;
    const int s = t % nstage, n = t / nstage;
    const uint4 v = *reinterpret_cast<const uint4*>(Wp + (size_t)n * (K / 2) + s * (LARGE_BK / 2) + c * 16);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}

// ===========================================================================
//  (A) 48 篇最佳基线（SCTAB=false，U8=true）——同进程 head-to-head
// ===========================================================================
template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1>
__global__ void __launch_bounds__(WM* WN* 32, MINB) gemm_mma_w4a8_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs, const float* __restrict__ Axsum,
    const uint8_t* __restrict__ Wp, const float* __restrict__ Ws, float* __restrict__ C, int M,
    int N, int K) {
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16，BM 必须 = WM*16");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;
  constexpr int WPAD = LARGE_BK / 2 + 16;
  constexpr int NGRP = 5120 / GROUP;
  constexpr int ASP4 = ASP / 4;
  constexpr int WPAD2 = WPAD / 2;
  constexpr int NC = LARGE_BK / 32;

  extern __shared__ __align__(16) char smem[];
  int8_t* As = reinterpret_cast<int8_t*>(smem);
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(As + (size_t)STAGES * BM * ASP);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / LARGE_BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nt = kb1 - kb0;

  auto load_stage = [&](int st, int kb) {
    const int k0 = kb * LARGE_BK;
    int8_t* a = As + (size_t)st * BM * ASP;
    uint8_t* wp = Wraw + (size_t)st * BN * WPAD;
    for (int i = tid; i < BM * (LARGE_BK / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 16), c = (i % (LARGE_BK / 16)) * 16;
      const int gr = block_row + r;
      if (gr < M) __pipeline_memcpy_async(&a[r * ASP + c], &Aq[(size_t)gr * K + k0 + c], 16);
      else *reinterpret_cast<uint4*>(&a[r * ASP + c]) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * (LARGE_BK / 2 / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 2 / 16), c = (i % (LARGE_BK / 2 / 16)) * 16;
      __pipeline_memcpy_async(&wp[r * WPAD + c],
                              &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
    }
    __pipeline_commit();
  };

  int acci[MTN][4];
  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acci[j][q] = 0;
      facc[j][q] = 0.f;
    }

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < nt) load_stage(p, kb0 + p);
    else __pipeline_commit();
  }

  const int arow0 = lane >> 2;
  const int arow1 = arow0 + 8;
  const int aqw = (lane & 3);
  const int bgroup = lane >> 2;

#pragma unroll 1
  for (int t = 0; t < nt; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if (tnext < nt) load_stage(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    const int g = kb0 + t;
    const uint32_t* As32 = reinterpret_cast<const uint32_t*>(As + (size_t)st * BM * ASP);
    const uint16_t* W16 = reinterpret_cast<const uint16_t*>(Wraw + (size_t)st * BN * WPAD);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      uint32_t a[4];
      a[0] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[1] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[2] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
      a[3] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int n = warp_col * WARP_N + j * 8 + bgroup;
        const int wo = n * WPAD2 + c * 8 + aqw;
        const uint32_t b0 = expand4u(W16[wo]);
        const uint32_t b1 = expand4u(W16[wo + 4]);
        const uint32_t bb[2] = {b0, b1};
        imma_m16n8k32_u8(acci[j], a, bb);
      }
    }
    {
      const int gm0 = warp_row * WARP_M + arow0;
      const int gm1 = gm0 + 8;
      const int rm0 = block_row + gm0, rm1 = block_row + gm1;
      const float a0 = (rm0 < M) ? __ldg(&Axs[(size_t)rm0 * NGRP + g]) : 0.f;
      const float a1 = (rm1 < M) ? __ldg(&Axs[(size_t)rm1 * NGRP + g]) : 0.f;
      const float corr0 = (rm0 < M) ? 8.f * __ldg(&Axsum[(size_t)rm0 * NGRP + g]) : 0.f;
      const float corr1 = (rm1 < M) ? 8.f * __ldg(&Axsum[(size_t)rm1 * NGRP + g]) : 0.f;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int ln0 = warp_col * WARP_N + j * 8 + (lane & 3) * 2;
        const float w0 = __ldg(&Ws[(size_t)(block_col + ln0) * NGRP + g]);
        const float w1 = __ldg(&Ws[(size_t)(block_col + ln0 + 1) * NGRP + g]);
        facc[j][0] += a0 * w0 * ((float)acci[j][0] - corr0);
        facc[j][1] += a0 * w1 * ((float)acci[j][1] - corr0);
        facc[j][2] += a1 * w0 * ((float)acci[j][2] - corr1);
        facc[j][3] += a1 * w1 * ((float)acci[j][3] - corr1);
#pragma unroll
        for (int q = 0; q < 4; ++q) acci[j][q] = 0;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int r0 = block_row + warp_row * WARP_M + arow0;
    const int c0 = block_col + warp_col * WARP_N + j * 8 + (lane & 3) * 2;
    const int r1 = r0 + 8;
    if (KSPLIT == 1) {
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = make_float2(facc[j][0], facc[j][1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = make_float2(facc[j][2], facc[j][3]);
    } else {
      if (r0 < M) {
        atomicAdd(&C[(size_t)r0 * N + c0], facc[j][0]);
        atomicAdd(&C[(size_t)r0 * N + c0 + 1], facc[j][1]);
      }
      if (r1 < M) {
        atomicAdd(&C[(size_t)r1 * N + c0], facc[j][2]);
        atomicAdd(&C[(size_t)r1 * N + c0 + 1], facc[j][3]);
      }
    }
  }
}

// ===========================================================================
//  (B) v1：转置 scale 布局 + per-stage cp.async 载入（非持久化，结构同 48）
//      AxsT/AsumT8: [NGRP][MPAD]（一行一个 group，BN/BM 连续）; WsT: [NGRP][N]。
// ===========================================================================
template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1,
          bool ONEBAR = false, bool TILED = false>
__global__ void __launch_bounds__(WM* WN* 32, MINB) gemm_mma_w4a8_pipe_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ AxsT, const float* __restrict__ AsumT8,
    const uint8_t* __restrict__ Wp, const float* __restrict__ WsT, float* __restrict__ C, int M,
    int N, int K) {
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16，BM 必须 = WM*16");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;
  constexpr int WPAD = LARGE_BK / 2 + 16;
  constexpr int ASP4 = ASP / 4;
  constexpr int WPAD2 = WPAD / 2;
  constexpr int NC = LARGE_BK / 32;

  extern __shared__ __align__(16) char smem[];
  int8_t* As = reinterpret_cast<int8_t*>(smem);
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(As + (size_t)STAGES * BM * ASP);
  float* sas = reinterpret_cast<float*>(Wraw + (size_t)STAGES * BN * WPAD);   // [STAGES][BM]
  float* scor = sas + (size_t)STAGES * BM;                                    // [STAGES][BM]
  float* sws = scor + (size_t)STAGES * BM;                                    // [STAGES][BN]

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / LARGE_BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nt = kb1 - kb0;

  auto load_stage = [&](int st, int g) {
    const int k0 = g * LARGE_BK;
    int8_t* a = As + (size_t)st * BM * ASP;
    uint8_t* wp = Wraw + (size_t)st * BN * WPAD;
    float* sa = sas + (size_t)st * BM;
    float* sc = scor + (size_t)st * BM;
    float* sw = sws + (size_t)st * BN;
    for (int i = tid; i < BM * (LARGE_BK / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 16), c = (i % (LARGE_BK / 16)) * 16;
      const int gr = block_row + r;
      if (gr < M) __pipeline_memcpy_async(&a[r * ASP + c], &Aq[(size_t)gr * K + k0 + c], 16);
      else *reinterpret_cast<uint4*>(&a[r * ASP + c]) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * (LARGE_BK / 2 / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 2 / 16), c = (i % (LARGE_BK / 2 / 16)) * 16;
      if constexpr (TILED) {
        // K-blocked 布局 Wt[g][N][64B]：一个 stage 的 BN 行 = 连续 8KB。
        __pipeline_memcpy_async(&wp[r * WPAD + c],
                                &Wp[(size_t)g * N * (LARGE_BK / 2) + (block_col + r) * (LARGE_BK / 2) + c],
                                16);
      } else {
        __pipeline_memcpy_async(&wp[r * WPAD + c],
                                &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
      }
    }
    // 转置后的 scale：一行 group 的 BM/BN 个 scale 连续，16B 对齐整体搬。
    for (int i = tid; i < BM / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sa[i * 4], &AxsT[(size_t)g * MPAD + i * 4], 16);
    for (int i = tid; i < BM / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sc[i * 4], &AsumT8[(size_t)g * MPAD + i * 4], 16);
    for (int i = tid; i < BN / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sw[i * 4], &WsT[(size_t)g * N + block_col + i * 4], 16);
    __pipeline_commit();
  };

  int acci[MTN][4];
  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acci[j][q] = 0;
      facc[j][q] = 0.f;
    }

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < nt) load_stage(p, kb0 + p);
    else __pipeline_commit();
  }

  const int arow0 = lane >> 2;
  const int arow1 = arow0 + 8;
  const int aqw = (lane & 3);
  const int bgroup = lane >> 2;

#pragma unroll 1
  for (int t = 0; t < nt; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if constexpr (ONEBAR) {
      // 单 barrier：wait → sync → 载下一 stage（写回已消费的 (t-1)%S）→ 消费。
      // 该 sync 同时保证「stage t 的 cp.async 对全 CTA 可见」与「所有线程已读完 t-1」。
      __pipeline_wait_prior(STAGES - 2);
      __syncthreads();
      if (tnext < nt) load_stage(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext);
      else __pipeline_commit();
    } else {
      if (tnext < nt) load_stage(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext);
      else __pipeline_commit();
      __pipeline_wait_prior(STAGES - 1);
      __syncthreads();
    }

    [[maybe_unused]] const int g = kb0 + t;
    const uint32_t* As32 = reinterpret_cast<const uint32_t*>(As + (size_t)st * BM * ASP);
    const uint16_t* W16 = reinterpret_cast<const uint16_t*>(Wraw + (size_t)st * BN * WPAD);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      uint32_t a[4];
      a[0] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[1] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[2] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
      a[3] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int n = warp_col * WARP_N + j * 8 + bgroup;
        const int wo = n * WPAD2 + c * 8 + aqw;
        const uint32_t bb[2] = {expand4u(W16[wo]), expand4u(W16[wo + 4])};
        imma_m16n8k32_u8(acci[j], a, bb);
      }
    }
    {
      const float* sa = sas + (size_t)st * BM;
      const float* sc = scor + (size_t)st * BM;
      const float* sw = sws + (size_t)st * BN;
      const int gm0 = warp_row * WARP_M + arow0;
      const int gm1 = gm0 + 8;
      const int rm0 = block_row + gm0, rm1 = block_row + gm1;
      const float a0 = (rm0 < M) ? sa[gm0] : 0.f;
      const float a1 = (rm1 < M) ? sa[gm1] : 0.f;
      const float corr0 = (rm0 < M) ? sc[gm0] : 0.f;
      const float corr1 = (rm1 < M) ? sc[gm1] : 0.f;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int ln0 = warp_col * WARP_N + j * 8 + (lane & 3) * 2;
        const float w0 = sw[ln0];
        const float w1 = sw[ln0 + 1];
        facc[j][0] += a0 * w0 * ((float)acci[j][0] - corr0);
        facc[j][1] += a0 * w1 * ((float)acci[j][1] - corr0);
        facc[j][2] += a1 * w0 * ((float)acci[j][2] - corr1);
        facc[j][3] += a1 * w1 * ((float)acci[j][3] - corr1);
#pragma unroll
        for (int q = 0; q < 4; ++q) acci[j][q] = 0;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int r0 = block_row + warp_row * WARP_M + arow0;
    const int c0 = block_col + warp_col * WARP_N + j * 8 + (lane & 3) * 2;
    const int r1 = r0 + 8;
    if (KSPLIT == 1) {
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = make_float2(facc[j][0], facc[j][1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = make_float2(facc[j][2], facc[j][3]);
    } else {
      if (r0 < M) {
        atomicAdd(&C[(size_t)r0 * N + c0], facc[j][0]);
        atomicAdd(&C[(size_t)r0 * N + c0 + 1], facc[j][1]);
      }
      if (r1 < M) {
        atomicAdd(&C[(size_t)r1 * N + c0], facc[j][2]);
        atomicAdd(&C[(size_t)r1 * N + c0 + 1], facc[j][3]);
      }
    }
  }
}

// ===========================================================================
//  (C) v2：跨 item 持久化流水（移植 43）。
//      BM=16（一个 m-tile），work item = (n_tile, k-chunk)，每 item CHUNK 个 group。
//      把 (ii,u) 展平成一维 stage 流，cp.async 永不 drain；item 末尾才落 facc。
// ===========================================================================
template <int BM, int BN, int STAGES, int CHUNK, int WM, int WN, int MINB = 2>
__global__ void __launch_bounds__(WM* WN* 32, MINB) imma_persist_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ AxsT, const float* __restrict__ AsumT8,
    const uint8_t* __restrict__ Wp, const float* __restrict__ WsT, float* __restrict__ C, int M,
    int N, int K, int total_items, int P) {
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16");
  static_assert(CHUNK >= STAGES, "item 比流水环短会打乱 stage 环");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;
  constexpr int WPAD = LARGE_BK / 2 + 16;
  constexpr int ASP4 = ASP / 4;
  constexpr int WPAD2 = WPAD / 2;
  constexpr int NC = LARGE_BK / 32;

  extern __shared__ __align__(16) char smem[];
  int8_t* As = reinterpret_cast<int8_t*>(smem);
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(As + (size_t)STAGES * BM * ASP);
  float* sas = reinterpret_cast<float*>(Wraw + (size_t)STAGES * BN * WPAD);
  float* scor = sas + (size_t)STAGES * BM;
  float* sws = scor + (size_t)STAGES * BM;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int ntiles = N / BN;
  const int nchunks = (K / LARGE_BK) / CHUNK;
  const int gid0 = blockIdx.x;
  const int nitems = (gid0 < total_items) ? (total_items - 1 - gid0) / P + 1 : 0;
  const int NS = nitems * CHUNK;

  // 展平 stage t → (item gid, group g)。item 内 n_tile 固定。
  auto item_of = [&](int t, int& gid, int& u) {
    u = t % CHUNK;
    gid = gid0 + (t / CHUNK) * P;
  };

  auto load_flat = [&](int t) {
    int gid, u;
    item_of(t, gid, u);
    const int nt = gid % ntiles;
    const int kc = gid / ntiles;
    const int col = nt * BN;
    const int g = kc * CHUNK + u;
    const int k0 = g * LARGE_BK;
    const int st = t % STAGES;
    int8_t* a = As + (size_t)st * BM * ASP;
    uint8_t* wp = Wraw + (size_t)st * BN * WPAD;
    float* sa = sas + (size_t)st * BM;
    float* sc = scor + (size_t)st * BM;
    float* sw = sws + (size_t)st * BN;
    for (int i = tid; i < BM * (LARGE_BK / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 16), c = (i % (LARGE_BK / 16)) * 16;
      if (r < M) __pipeline_memcpy_async(&a[r * ASP + c], &Aq[(size_t)r * K + k0 + c], 16);
      else *reinterpret_cast<uint4*>(&a[r * ASP + c]) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * (LARGE_BK / 2 / 16); i += NTHREADS) {
      const int r = i / (LARGE_BK / 2 / 16), c = (i % (LARGE_BK / 2 / 16)) * 16;
      __pipeline_memcpy_async(&wp[r * WPAD + c],
                              &Wp[(size_t)(col + r) * (K / 2) + k0 / 2 + c], 16);
    }
    for (int i = tid; i < BM / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sa[i * 4], &AxsT[(size_t)g * MPAD + i * 4], 16);
    for (int i = tid; i < BM / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sc[i * 4], &AsumT8[(size_t)g * MPAD + i * 4], 16);
    for (int i = tid; i < BN / 4; i += NTHREADS)
      __pipeline_memcpy_async(&sw[i * 4], &WsT[(size_t)g * N + col + i * 4], 16);
    __pipeline_commit();
  };

  int acci[MTN][4];
  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acci[j][q] = 0;
      facc[j][q] = 0.f;
    }

  const int arow0 = lane >> 2;
  const int arow1 = arow0 + 8;
  const int aqw = (lane & 3);
  const int bgroup = lane >> 2;

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < NS) load_flat(p);
    else __pipeline_commit();
  }

#pragma unroll 1
  for (int t = 0; t < NS; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if (tnext < NS) load_flat(tnext);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    const uint32_t* As32 = reinterpret_cast<const uint32_t*>(As + (size_t)st * BM * ASP);
    const uint16_t* W16 = reinterpret_cast<const uint16_t*>(Wraw + (size_t)st * BN * WPAD);
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      uint32_t a[4];
      a[0] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[1] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + aqw];
      a[2] = As32[(arow0 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
      a[3] = As32[(arow1 + warp_row * WARP_M) * ASP4 + c * 8 + 4 + aqw];
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int n = warp_col * WARP_N + j * 8 + bgroup;
        const int wo = n * WPAD2 + c * 8 + aqw;
        const uint32_t bb[2] = {expand4u(W16[wo]), expand4u(W16[wo + 4])};
        imma_m16n8k32_u8(acci[j], a, bb);
      }
    }
    {
      const float* sa = sas + (size_t)st * BM;
      const float* sc = scor + (size_t)st * BM;
      const float* sw = sws + (size_t)st * BN;
      const int gm0 = warp_row * WARP_M + arow0;
      const int gm1 = gm0 + 8;
      const int rm0 = gm0, rm1 = gm1;
      const float a0 = (rm0 < M) ? sa[gm0] : 0.f;
      const float a1 = (rm1 < M) ? sa[gm1] : 0.f;
      const float corr0 = (rm0 < M) ? sc[gm0] : 0.f;
      const float corr1 = (rm1 < M) ? sc[gm1] : 0.f;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int ln0 = warp_col * WARP_N + j * 8 + (lane & 3) * 2;
        const float w0 = sw[ln0];
        const float w1 = sw[ln0 + 1];
        facc[j][0] += a0 * w0 * ((float)acci[j][0] - corr0);
        facc[j][1] += a0 * w1 * ((float)acci[j][1] - corr0);
        facc[j][2] += a1 * w0 * ((float)acci[j][2] - corr1);
        facc[j][3] += a1 * w1 * ((float)acci[j][3] - corr1);
#pragma unroll
        for (int q = 0; q < 4; ++q) acci[j][q] = 0;
      }
    }
    __syncthreads();

    // item 末尾落 facc（在热循环里但只每 CHUNK 个 stage 一次）
    if ((t % CHUNK) == CHUNK - 1) {
      int gid, u;
      item_of(t, gid, u);
      const int col = (gid % ntiles) * BN;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int r0 = warp_row * WARP_M + arow0;
        const int c0 = col + warp_col * WARP_N + j * 8 + (lane & 3) * 2;
        const int r1 = r0 + 8;
        if (nchunks == 1) {
          if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = make_float2(facc[j][0], facc[j][1]);
          if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = make_float2(facc[j][2], facc[j][3]);
        } else {
          if (r0 < M) {
            atomicAdd(&C[(size_t)r0 * N + c0], facc[j][0]);
            atomicAdd(&C[(size_t)r0 * N + c0 + 1], facc[j][1]);
          }
          if (r1 < M) {
            atomicAdd(&C[(size_t)r1 * N + c0], facc[j][2]);
            atomicAdd(&C[(size_t)r1 * N + c0 + 1], facc[j][3]);
          }
        }
#pragma unroll
        for (int q = 0; q < 4; ++q) facc[j][q] = 0.f;
      }
    }
  }
}

// ===========================================================================
//  (D) v3：warp-private pipeline。每个 warp 拥有自己的 stage 环与 scale，
//      CTA 级 __syncthreads 全部换成 __syncwarp（无共享同步面）。
//  变体：TILED=K-blocked 权重；WMODE=1 时 A 由 CTA 一次性预载进共享缓冲（消冗余读）。
// ===========================================================================
template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1,
          bool TILED = false, int WMODE = 0, int SUB = 1>
__global__ void __launch_bounds__(WM* WN* 32, MINB) gemm_mma_w4a8_warp_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ AxsT, const float* __restrict__ AsumT8,
    const uint8_t* __restrict__ Wp, const float* __restrict__ WsT, float* __restrict__ C, int M,
    int N, int K) {
  constexpr int NWARP = WM * WN;
  constexpr int NTHREADS = NWARP * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16，BM 必须 = WM*16");
  static_assert(WM == 1, "warp-private 版固定 WM=1");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;
  constexpr int WPAD = LARGE_BK / 2 + 16;
  constexpr int ASP4 = ASP / 4;
  constexpr int WPAD2 = WPAD / 2;
  constexpr int NC = LARGE_BK / 32;
  constexpr int NBLK = 40;  // K/LARGE_BK（本 shape 固定）
  constexpr int NTMAX = (NBLK + KSPLIT - 1) / KSPLIT;
  constexpr int APER = BM * ASP;
  constexpr int WPER1 = WARP_N * WPAD;      // 一个 group 的 W 字节
  constexpr int SPER1 = 2 * BM + WARP_N;    // 一个 group 的 scale 浮点数
  constexpr int WPER = SUB * WPER1;         // 一个 stage（SUB 个 group）的 W
  constexpr int SPER = SUB * SPER1;         // 一个 stage 的 scale
  constexpr size_t AFULL_BYTES = (size_t)NTMAX * APER;

  extern __shared__ __align__(16) char smem[];
  int8_t* Afull = reinterpret_cast<int8_t*>(smem);
  uint8_t* Wbase = nullptr;
  float* sbase = nullptr;
  int8_t* Abase = nullptr;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int wcol = warp_col * WARP_N;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / LARGE_BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nt = kb1 - kb0;
  constexpr int ASPC = LARGE_BK + 16;  // WMODE=2 的紧凑 A 行距（须 16B 对齐）
  constexpr int ASPC4 = ASPC / 4;

  // WMODE: 0=A 每 stage 载入（per-warp 冗余）；1=A 每 CTA 一次性预载（BM 行）；
  //        2=A 每 CTA 一次性预载（仅 M 行，紧凑，smem 随 M 缩小）
  size_t aoff = 0;
  if constexpr (WMODE == 1) aoff = AFULL_BYTES;
  if constexpr (WMODE == 2) aoff = (size_t)nt * M * ASPC;
  Wbase = reinterpret_cast<uint8_t*>(Afull + aoff);
  sbase = reinterpret_cast<float*>(Wbase + (size_t)NWARP * STAGES * WPER);
  Abase = reinterpret_cast<int8_t*>(sbase + (size_t)NWARP * STAGES * SPER);

  auto load_w = [&](int st, int gbase) {
    for (int s = 0; s < SUB; ++s) {
      const int g = gbase + s;
      const int k0 = g * LARGE_BK;
      uint8_t* wp = Wbase + ((size_t)wid * STAGES + st) * WPER + (size_t)s * WPER1;
      float* sp = sbase + ((size_t)wid * STAGES + st) * SPER + (size_t)s * SPER1;
      float* sa = sp, *sc = sp + BM, *sw = sp + 2 * BM;
      for (int i = lane; i < WARP_N * (LARGE_BK / 2 / 16); i += 32) {
        const int r = i / (LARGE_BK / 2 / 16), c = (i % (LARGE_BK / 2 / 16)) * 16;
        if constexpr (TILED)
          __pipeline_memcpy_async(
              &wp[r * WPAD + c],
              &Wp[(size_t)g * N * (LARGE_BK / 2) + (block_col + wcol + r) * (LARGE_BK / 2) + c], 16);
        else
          __pipeline_memcpy_async(&wp[r * WPAD + c],
                                  &Wp[(size_t)(block_col + wcol + r) * (K / 2) + k0 / 2 + c], 16);
      }
      for (int i = lane; i < BM / 4; i += 32)
        __pipeline_memcpy_async(&sa[i * 4], &AxsT[(size_t)g * MPAD + i * 4], 16);
      for (int i = lane; i < BM / 4; i += 32)
        __pipeline_memcpy_async(&sc[i * 4], &AsumT8[(size_t)g * MPAD + i * 4], 16);
      for (int i = lane; i < WARP_N / 4; i += 32)
        __pipeline_memcpy_async(&sw[i * 4], &WsT[(size_t)g * N + block_col + wcol + i * 4], 16);
    }
    __pipeline_commit();
  };

  auto load_a = [&](int st, int gbase) {
    for (int s = 0; s < SUB; ++s) {
      const int g = gbase + s;
      const int k0 = g * LARGE_BK;
      int8_t* a = Abase + ((size_t)wid * STAGES + st) * (SUB * APER) + (size_t)s * APER;
      for (int i = lane; i < BM * (LARGE_BK / 16); i += 32) {
        const int r = i / (LARGE_BK / 16), c = (i % (LARGE_BK / 16)) * 16;
        if (block_row + r < M)
          __pipeline_memcpy_async(&a[r * ASP + c], &Aq[(size_t)(block_row + r) * K + k0 + c], 16);
        else
          *reinterpret_cast<uint4*>(&a[r * ASP + c]) = make_uint4(0, 0, 0, 0);
      }
    }
    __pipeline_commit();
  };

  int acci[MTN][4];
  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acci[j][q] = 0;
      facc[j][q] = 0.f;
    }

  if constexpr (WMODE == 1 || WMODE == 2) {
    // A 由 CTA 一次性预载整段（每 warp 不再重复读），只此一次 __syncthreads。
    const int rows = (WMODE == 2) ? M : BM;
    const int stride = (WMODE == 2) ? ASPC : ASP;
    for (int i = tid; i < nt * rows * (LARGE_BK / 16); i += NTHREADS) {
      const int t = i / (rows * (LARGE_BK / 16));
      const int rem = i % (rows * (LARGE_BK / 16));
      const int r = rem / (LARGE_BK / 16), c = (rem % (LARGE_BK / 16)) * 16;
      int8_t* a = Afull + (size_t)t * (rows * stride);
      if (block_row + r < M)
        __pipeline_memcpy_async(&a[r * stride + c],
                                &Aq[(size_t)(block_row + r) * K + (kb0 + t) * LARGE_BK + c], 16);
      else
        *reinterpret_cast<uint4*>(&a[r * stride + c]) = make_uint4(0, 0, 0, 0);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
  }

  const int nsup = nt / SUB;  // host 保证 nt % SUB == 0

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < nsup) {
      load_w(p, kb0 + p * SUB);
      if constexpr (WMODE == 0) load_a(p, kb0 + p * SUB);
    } else {
      __pipeline_commit();
      if constexpr (WMODE == 0) __pipeline_commit();
    }
  }

  const int arow0 = lane >> 2;
  const int arow1 = arow0 + 8;
  const int aqw = (lane & 3);
  const int bgroup = lane >> 2;

#pragma unroll 1
  for (int t = 0; t < nsup; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if (tnext < nsup) {
      load_w(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext * SUB);
      if constexpr (WMODE == 0) load_a(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext * SUB);
    } else {
      __pipeline_commit();
      if constexpr (WMODE == 0) __pipeline_commit();
    }
    __pipeline_wait_prior(STAGES - 1);
    __syncwarp();

#pragma unroll
    for (int s = 0; s < SUB; ++s) {
      const int gidx = t * SUB + s;  // 相对 group 下标
      const int8_t* aPtr;
      if (WMODE == 2) aPtr = Afull + (size_t)gidx * ((size_t)M * ASPC);
      else if (WMODE == 1) aPtr = Afull + (size_t)gidx * APER;
      else aPtr = Abase + ((size_t)wid * STAGES + st) * (SUB * APER) + (size_t)s * APER;
      const uint32_t* As32 = reinterpret_cast<const uint32_t*>(aPtr);
      const int ar0 = (WMODE == 2) ? ((arow0 < M) ? arow0 : M - 1) : arow0;
      const int ar1 = (WMODE == 2) ? ((arow1 < M) ? arow1 : M - 1) : arow1;
      constexpr int A4 = (WMODE == 2) ? ASPC4 : ASP4;
      const uint16_t* W16 = reinterpret_cast<const uint16_t*>(
          Wbase + ((size_t)wid * STAGES + st) * WPER + (size_t)s * WPER1);
      const float* sp = sbase + ((size_t)wid * STAGES + st) * SPER + (size_t)s * SPER1;
      const float* sa = sp, *sc = sp + BM, *sw = sp + 2 * BM;
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        uint32_t a[4];
        a[0] = As32[(ar0)*A4 + c * 8 + aqw];
        a[1] = As32[(ar1)*A4 + c * 8 + aqw];
        a[2] = As32[(ar0)*A4 + c * 8 + 4 + aqw];
        a[3] = As32[(ar1)*A4 + c * 8 + 4 + aqw];
#pragma unroll
        for (int j = 0; j < MTN; ++j) {
          const int n = j * 8 + bgroup;
          const int wo = n * WPAD2 + c * 8 + aqw;
          const uint32_t bb[2] = {expand4u(W16[wo]), expand4u(W16[wo + 4])};
          imma_m16n8k32_u8(acci[j], a, bb);
        }
      }
      {
        const int rm0 = block_row + arow0, rm1 = block_row + arow1;
        const float a0 = (rm0 < M) ? sa[arow0] : 0.f;
        const float a1 = (rm1 < M) ? sa[arow1] : 0.f;
        const float corr0 = (rm0 < M) ? sc[arow0] : 0.f;
        const float corr1 = (rm1 < M) ? sc[arow1] : 0.f;
#pragma unroll
        for (int j = 0; j < MTN; ++j) {
          const int ln0 = j * 8 + (lane & 3) * 2;
          const float w0 = sw[ln0];
          const float w1 = sw[ln0 + 1];
          facc[j][0] += a0 * w0 * ((float)acci[j][0] - corr0);
          facc[j][1] += a0 * w1 * ((float)acci[j][1] - corr0);
          facc[j][2] += a1 * w0 * ((float)acci[j][2] - corr1);
          facc[j][3] += a1 * w1 * ((float)acci[j][3] - corr1);
#pragma unroll
          for (int q = 0; q < 4; ++q) acci[j][q] = 0;
        }
      }
    }
    __syncwarp();
  }

#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int r0 = block_row + warp_row * WARP_M + arow0;
    const int c0 = block_col + wcol + j * 8 + (lane & 3) * 2;
    const int r1 = r0 + 8;
    if (KSPLIT == 1) {
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = make_float2(facc[j][0], facc[j][1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = make_float2(facc[j][2], facc[j][3]);
    } else {
      if (r0 < M) {
        atomicAdd(&C[(size_t)r0 * N + c0], facc[j][0]);
        atomicAdd(&C[(size_t)r0 * N + c0 + 1], facc[j][1]);
      }
      if (r1 < M) {
        atomicAdd(&C[(size_t)r1 * N + c0], facc[j][2]);
        atomicAdd(&C[(size_t)r1 * N + c0 + 1], facc[j][3]);
      }
    }
  }
}

// ===========================================================================
//  (E) v4：warp-private + TMA（cp.async.bulk）
//  每个 warp 一个 stage 环 + 一个私有 mbarrier。W 在 K-blocked 布局下，一个 warp
//  一个 group 恰是连续 2KB，用 1D cp.async.bulk 一把搬入（scale 同批），A 由 CTA
//  一次性预载。把「多组 cp.async + 每 stage 一次 DEPBAR」换成「4 条 bulk + mbarrier
//  完成信号」，在飞量交给 TMA 引擎。
// ===========================================================================
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes));
}
__device__ __forceinline__ void tma_load_1d(void* smem_dst, const void* gmem_src, uint32_t bytes,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
      ::"r"(smem_u32(smem_dst)), "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar))
      : "memory");
}

template <int BM, int BN, int STAGES, int WN, int KSPLIT = 1, int MINB = 1, int SUB = 1,
          bool PERM = true>
__global__ void __launch_bounds__(WN * 32, MINB) gemm_mma_w4a8_tma_warp_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ AxsT, const float* __restrict__ AsumT8,
    const uint8_t* __restrict__ Wt, const float* __restrict__ WsT, float* __restrict__ C, int M,
    int N, int K) {
  constexpr int NWARP = WN;
  constexpr int NTHREADS = NWARP * 32;
  constexpr int WARP_M = BM;  // WM=1
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;
  constexpr int ASP4 = ASP / 4;
  constexpr int NC = LARGE_BK / 32;
  constexpr int NBLK = 40;
  constexpr int NTMAX = (NBLK + KSPLIT - 1) / KSPLIT;
  constexpr int APER = BM * ASP;
  constexpr int WROW = LARGE_BK / 2;  // 64B
  constexpr int WPER1 = WARP_N * WROW;
  constexpr int WPER = SUB * WPER1;
  constexpr int SPER1 = 2 * BM + WARP_N;
  constexpr int SPER = SUB * SPER1;
  constexpr int WSTEP = WROW / 8;  // 每 (c,n) 单元 8 个 uint16
  constexpr int WPAD2 = WROW / 2;  // PERM=false 的连续行主序布局
  static_assert(WARP_N == 32, "Wt2 布局按每 warp 32 行排布");
  constexpr size_t AFULL_BYTES = (size_t)NTMAX * APER;

  extern __shared__ __align__(16) char smem[];
  int8_t* Afull = reinterpret_cast<int8_t*>(smem);
  uint8_t* Wbase = reinterpret_cast<uint8_t*>(Afull + AFULL_BYTES);
  float* sbase = reinterpret_cast<float*>(Wbase + (size_t)NWARP * STAGES * WPER);
  uint64_t* mbars = reinterpret_cast<uint64_t*>(sbase + (size_t)NWARP * STAGES * SPER);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_col = wid;
  const int wcol = warp_col * WARP_N;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / LARGE_BK;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;
  const int nt = kb1 - kb0;
  const int nsup = nt / SUB;

  uint64_t* mybar = mbars + (size_t)wid * STAGES;
  uint8_t* myW = Wbase + (size_t)wid * STAGES * WPER;
  float* myS = sbase + (size_t)wid * STAGES * SPER;

  for (int s = 0; s < STAGES; ++s) mbar_init(mybar + s, 1);
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

  // A 一次性预载（CTA 共享，只此一次 __syncthreads）
  for (int i = tid; i < nt * BM * (LARGE_BK / 16); i += NTHREADS) {
    const int t = i / (BM * (LARGE_BK / 16));
    const int rem = i % (BM * (LARGE_BK / 16));
    const int r = rem / (LARGE_BK / 16), c = (rem % (LARGE_BK / 16)) * 16;
    int8_t* a = Afull + (size_t)t * APER;
    if (block_row + r < M)
      __pipeline_memcpy_async(&a[r * ASP + c],
                              &Aq[(size_t)(block_row + r) * K + (kb0 + t) * LARGE_BK + c], 16);
    else
      *reinterpret_cast<uint4*>(&a[r * ASP + c]) = make_uint4(0, 0, 0, 0);
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();

  // 只由 lane0 发起 TMA：一个 stage = SUB 个 group 的 (W + sa + sc + sw)
  auto issue = [&](int st, int gbase) {
    for (int s = 0; s < SUB; ++s) {
      const int g = gbase + s;
      uint8_t* wp = myW + (size_t)st * WPER + (size_t)s * WPER1;
      float* sp = myS + (size_t)st * SPER + (size_t)s * SPER1;
      const uint32_t txW = WPER1;
      const uint32_t txA = BM * 4;
      const uint32_t txWt = WARP_N * 4;
      mbar_arrive_expect_tx(mybar + st, txW + 2 * txA + txWt);
      if constexpr (PERM)
        tma_load_1d(wp, &Wt[((size_t)g * (N / 32) + (block_col / 32 + warp_col)) * (32 * WROW)], txW,
                    mybar + st);
      else
        tma_load_1d(wp, &Wt[(size_t)g * N * WROW + (size_t)(block_col + wcol) * WROW], txW, mybar + st);
      tma_load_1d(sp, &AxsT[(size_t)g * MPAD], txA, mybar + st);
      tma_load_1d(sp + BM, &AsumT8[(size_t)g * MPAD], txA, mybar + st);
      tma_load_1d(sp + 2 * BM, &WsT[(size_t)g * N + block_col + wcol], txWt, mybar + st);
    }
  };

#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p)
    if (lane == 0 && p < nsup) issue(p, kb0 + p * SUB);

  int acci[MTN][4];
  float facc[MTN][4];
#pragma unroll
  for (int j = 0; j < MTN; ++j)
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      acci[j][q] = 0;
      facc[j][q] = 0.f;
    }

  const int arow0 = lane >> 2;
  const int arow1 = arow0 + 8;
  const int aqw = (lane & 3);
  const int bgroup = lane >> 2;

#pragma unroll 1
  for (int t = 0; t < nsup; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if (tnext < nsup && lane == 0) issue(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext * SUB);
    mbar_wait(mybar + st, (t / STAGES) & 1);
    __syncwarp();

#pragma unroll
    for (int s = 0; s < SUB; ++s) {
      const int gidx = t * SUB + s;
      const int8_t* aPtr = Afull + (size_t)gidx * APER;
      const uint32_t* As32 = reinterpret_cast<const uint32_t*>(aPtr);
      const uint16_t* W16 = reinterpret_cast<const uint16_t*>(myW + (size_t)st * WPER + (size_t)s * WPER1);
      const float* sp = myS + (size_t)st * SPER + (size_t)s * SPER1;
      const float* sa = sp, *sc = sp + BM, *sw = sp + 2 * BM;
#pragma unroll
      for (int c = 0; c < NC; ++c) {
        uint32_t a[4];
        a[0] = As32[(arow0)*ASP4 + c * 8 + aqw];
        a[1] = As32[(arow1)*ASP4 + c * 8 + aqw];
        a[2] = As32[(arow0)*ASP4 + c * 8 + 4 + aqw];
        a[3] = As32[(arow1)*ASP4 + c * 8 + 4 + aqw];
#pragma unroll
        for (int j = 0; j < MTN; ++j) {
          const int n = j * 8 + bgroup;
          const int wo = PERM ? ((c * 32 + n) * WSTEP + aqw) : (n * WPAD2 + c * 8 + aqw);
          const uint32_t bb[2] = {expand4u(W16[wo]), expand4u(W16[wo + 4])};
          imma_m16n8k32_u8(acci[j], a, bb);
        }
      }
      {
        const int rm0 = block_row + arow0, rm1 = block_row + arow1;
        const float a0 = (rm0 < M) ? sa[arow0] : 0.f;
        const float a1 = (rm1 < M) ? sa[arow1] : 0.f;
        const float corr0 = (rm0 < M) ? sc[arow0] : 0.f;
        const float corr1 = (rm1 < M) ? sc[arow1] : 0.f;
#pragma unroll
        for (int j = 0; j < MTN; ++j) {
          const int ln0 = j * 8 + (lane & 3) * 2;
          const float w0 = sw[ln0];
          const float w1 = sw[ln0 + 1];
          facc[j][0] += a0 * w0 * ((float)acci[j][0] - corr0);
          facc[j][1] += a0 * w1 * ((float)acci[j][1] - corr0);
          facc[j][2] += a1 * w0 * ((float)acci[j][2] - corr1);
          facc[j][3] += a1 * w1 * ((float)acci[j][3] - corr1);
#pragma unroll
          for (int q = 0; q < 4; ++q) acci[j][q] = 0;
        }
      }
    }
    __syncwarp();
  }

#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int r0 = block_row + arow0;
    const int c0 = block_col + wcol + j * 8 + (lane & 3) * 2;
    const int r1 = r0 + 8;
    if (KSPLIT == 1) {
      if (r0 < M) *reinterpret_cast<float2*>(&C[(size_t)r0 * N + c0]) = make_float2(facc[j][0], facc[j][1]);
      if (r1 < M) *reinterpret_cast<float2*>(&C[(size_t)r1 * N + c0]) = make_float2(facc[j][2], facc[j][3]);
    } else {
      if (r0 < M) {
        atomicAdd(&C[(size_t)r0 * N + c0], facc[j][0]);
        atomicAdd(&C[(size_t)r0 * N + c0 + 1], facc[j][1]);
      }
      if (r1 < M) {
        atomicAdd(&C[(size_t)r1 * N + c0], facc[j][2]);
        atomicAdd(&C[(size_t)r1 * N + c0 + 1], facc[j][3]);
      }
    }
  }
}

// ===========================================================================
//  host
// ===========================================================================
static bf16 f2b(float x) { return __float2bfloat16(x); }

struct Buf {
  bf16* A;
  int8_t* Aq;
  float* Axs;
  float* Axsum;
  float* AxsT;
  float* AsumT8;
  uint8_t* Wp;
  uint8_t* Wt;
  uint8_t* Wt2;
  float* Ws;
  float* WsT;
  float* C;
};

static size_t pipe_smem(int BM, int BN, int STAGES) {
  return (size_t)STAGES * (BM * (LARGE_BK + 16) + BN * (LARGE_BK / 2 + 16)) +
         (size_t)STAGES * (2 * BM + BN) * 4;
}

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1>
static void add_base(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                     const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a8_kernel<BM, BN, STAGES, WM, WN, KSPLIT, MINB>;
  const int NT = WM * WN * 32;
  constexpr int ASP = LARGE_BK + 16, WPAD = LARGE_BK / 2 + 16;
  const size_t shm = (size_t)STAGES * (BM * ASP + BN * WPAD);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm);
  specs.push_back([=](int M, float*) {
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.Aq, buf.Axs, buf.Axsum, buf.Wp,
                                                         buf.Ws, buf.C, M, N, K);
  });
}

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1,
          bool ONEBAR = false, bool TILED = false>
static void add_pipe(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                     const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a8_pipe_kernel<BM, BN, STAGES, WM, WN, KSPLIT, MINB, ONEBAR, TILED>;
  const int NT = WM * WN * 32;
  const size_t shm = pipe_smem(BM, BN, STAGES);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  const uint8_t* Wptr = TILED ? buf.Wt : buf.Wp;
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm);
  specs.push_back([=](int M, float*) {
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.Aq, buf.AxsT, buf.AsumT8, Wptr,
                                                         buf.WsT, buf.C, M, N, K);
  });
}

static size_t warp_smem(int BM, int BN, int STAGES, int WN, int WMODE, int ksplit, int M = 16,
                        int SUB = 1) {
  const int NWARP = WN;
  const int WARP_N = BN / WN;
  const size_t wper = (size_t)STAGES * NWARP * SUB * WARP_N * (LARGE_BK / 2 + 16);
  const size_t sp = (size_t)STAGES * NWARP * SUB * (2 * BM + WARP_N) * 4;
  size_t extra;
  const int ntmax = (40 + ksplit - 1) / ksplit;
  if (WMODE == 1) extra = (size_t)ntmax * BM * (LARGE_BK + 16);
  else if (WMODE == 2) extra = (size_t)ntmax * M * (LARGE_BK + 16);
  else extra = (size_t)STAGES * NWARP * SUB * BM * (LARGE_BK + 16);
  return wper + sp + extra;
}

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, int MINB = 1,
          bool TILED = false, int WMODE = 0, int SUB = 1>
static void add_warp(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                     const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a8_warp_kernel<BM, BN, STAGES, WM, WN, KSPLIT, MINB, TILED, WMODE, SUB>;
  const int NT = WM * WN * 32;
  const size_t shm_max = warp_smem(BM, BN, STAGES, WN, WMODE, KSPLIT, BM, SUB);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm_max));
  const uint8_t* Wptr = TILED ? buf.Wt : buf.Wp;
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm_max);
  specs.push_back([=](int M, float*) {
    const size_t shm = warp_smem(BM, BN, STAGES, WN, WMODE, KSPLIT, M, SUB);
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.Aq, buf.AxsT, buf.AsumT8, Wptr,
                                                         buf.WsT, buf.C, M, N, K);
  });
}

static size_t tma_warp_smem(int BM, int BN, int STAGES, int WN, int ksplit, int SUB = 1) {
  const int WARP_N = BN / WN;
  const size_t wper = (size_t)STAGES * WN * SUB * WARP_N * (LARGE_BK / 2);
  const size_t sp = (size_t)STAGES * WN * SUB * (2 * BM + WARP_N) * 4;
  const size_t mb = (size_t)WN * STAGES * 8;
  const int ntmax = (40 + ksplit - 1) / ksplit;
  const size_t afull = (size_t)ntmax * BM * (LARGE_BK + 16);
  return wper + sp + mb + afull + 1024;
}

template <int BM, int BN, int STAGES, int WN, int KSPLIT = 1, int MINB = 1, int SUB = 1,
          bool PERM = true>
static void add_tma_warp(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                         const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a8_tma_warp_kernel<BM, BN, STAGES, WN, KSPLIT, MINB, SUB, PERM>;
  const int NT = WN * 32;
  const size_t shm = tma_warp_smem(BM, BN, STAGES, WN, KSPLIT, SUB);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  const uint8_t* Wptr = PERM ? buf.Wt2 : buf.Wt;
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm);
  specs.push_back([=](int M, float*) {
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.Aq, buf.AxsT, buf.AsumT8, Wptr,
                                                         buf.WsT, buf.C, M, N, K);
  });
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* which = (argc > 1) ? argv[1] : "all";
  const char* only = (argc > 2) ? argv[2] : "";
  const int N = 17408;
  const int K = 5120;
  const int MAXM = 16;  // 持久化版 BM=16，单 m-tile

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double w_bytes_int4 = (double)N * K / 2;
  const double s_bytes = (double)N * (K / GROUP) * 4;
  const double total_w = w_bytes_int4 + s_bytes;
  std::printf("\nW4A8 IMMA pipe: N=%d K=%d group=%d ; W_int4=%.1f MB scales=%.1f MB total=%.1f MB\n",
              N, K, GROUP, w_bytes_int4 / 1e6, s_bytes / 1e6, total_w / 1e6);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::vector<bf16> hA((size_t)MAXM * K);
  std::vector<float> hWf((size_t)N * K), hWdeq((size_t)N * K);
  std::vector<uint8_t> hWp((size_t)N * (K / 2));
  std::vector<float> hWs((size_t)N * (K / GROUP));
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

  // 激活量化参考（CPU，用于对拍）
  std::vector<int8_t> hAq((size_t)MAXM * K);
  std::vector<float> hAxs((size_t)MAXM * (K / GROUP));
  std::vector<float> hAsum((size_t)MAXM * (K / GROUP));
  for (int m = 0; m < MAXM; ++m)
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k)
        amax = std::max(amax, std::fabs(__bfloat162float(hA[(size_t)m * K + g * GROUP + k])));
      const float s = amax > 0 ? amax / 127.f : 1.f;
      hAxs[(size_t)m * (K / GROUP) + g] = s;
      int sum = 0;
      for (int k = 0; k < GROUP; ++k) {
        int v = (int)std::nearbyint(__bfloat162float(hA[(size_t)m * K + g * GROUP + k]) * (1.f / s));
        v = std::max(-127, std::min(127, v));
        hAq[(size_t)m * K + g * GROUP + k] = (int8_t)v;
        sum += v;
      }
      hAsum[(size_t)m * (K / GROUP) + g] = (float)sum;
    }

  // 转置 scale（host）：AxsT[g][m]、AsumT8[g][m]（×8）、WsT[g][n]。
  const int NGRP = K / GROUP;
  std::vector<float> hAxsT((size_t)NGRP * MPAD, 0.f), hAsumT8((size_t)NGRP * MPAD, 0.f);
  for (int m = 0; m < MAXM; ++m)
    for (int g = 0; g < NGRP; ++g) {
      hAxsT[(size_t)g * MPAD + m] = hAxs[(size_t)m * NGRP + g];
      hAsumT8[(size_t)g * MPAD + m] = 8.f * hAsum[(size_t)m * NGRP + g];
    }
  std::vector<float> hWsT((size_t)NGRP * N);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < NGRP; ++g) hWsT[(size_t)g * N + n] = hWs[(size_t)n * NGRP + g];
  // K-blocked 权重布局：Wt[g][n][64B]，一个 stage 的 BN 行连续 8KB。
  std::vector<uint8_t> hWt((size_t)N * (K / 2));
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < NGRP; ++g)
      std::memcpy(&hWt[(size_t)g * N * 64 + (size_t)n * 64], &hWp[(size_t)n * (K / 2) + g * 64], 64);
  // TMA 友好布局 Wt2[g][w][c][n(32)][8 u16]：一个 warp 一个 group = 连续 2KB，
  // 且按 [c][n] 排列后 IMMA 读 W 无 bank conflict（n 步进 4 word）。
  std::vector<uint8_t> hWt2((size_t)N * (K / 2));
  {
    const int NW = N / 32;
    for (int g = 0; g < NGRP; ++g)
      for (int w = 0; w < NW; ++w)
        for (int c = 0; c < 4; ++c)
          for (int n = 0; n < 32; ++n) {
            const uint8_t* src = &hWt[(size_t)g * N * 64 + (size_t)(w * 32 + n) * 64];
            uint8_t* dst = &hWt2[(((size_t)g * NW + w) * 4 + c) * 32 * 16 + (size_t)n * 16];
            std::memcpy(dst, src + c * 16, 16);
          }
  }

  Buf buf;
  CUDA_CHECK(cudaMalloc(&buf.A, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&buf.Aq, hAq.size()));
  CUDA_CHECK(cudaMalloc(&buf.Axs, hAxs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.Axsum, hAsum.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.AxsT, hAxsT.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.AsumT8, hAsumT8.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.Wp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&buf.Wt, hWt.size()));
  CUDA_CHECK(cudaMalloc(&buf.Wt2, hWt2.size()));
  CUDA_CHECK(cudaMalloc(&buf.Ws, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.WsT, hWsT.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.C, (size_t)MAXM * N * 4));
  CUDA_CHECK(cudaMemcpy(buf.A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Aq, hAq.data(), hAq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Axs, hAxs.data(), hAxs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Axsum, hAsum.data(), hAsum.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Wp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Wt, hWt.data(), hWt.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Wt2, hWt2.data(), hWt2.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Ws, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));
  // device 端转置（与 quant 一致）：先跑 quant 灌 Axs/Axsum，再转置。
  {
    const int ngroup = MAXM * NGRP;
    quant_x_kernel<<<div_up(ngroup, 8), 256>>>(buf.A, buf.Aq, buf.Axs, buf.Axsum, ngroup, K);
    CUDA_CHECK_LAST();
    transpose_scale_kernel<<<div_up(ngroup, 256), 256>>>(buf.Axs, buf.Axsum, buf.AxsT, buf.AsumT8,
                                                         MAXM, NGRP);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaMemcpy(buf.WsT, hWsT.data(), hWsT.size() * 4, cudaMemcpyHostToDevice));
  }

  // 纯读上界（44/45/46 口径）：读 W_int4 + Ws，不含 A、无 mma。
  {
    auto launch = [&] { wread_roofline_kernel<<<8 * NSM, 256>>>(buf.Wp, buf.Ws, buf.C, N, K); };
    launch();
    CUDA_CHECK_LAST();
    const double ms = bench_ms(launch, 5, 50);
    std::printf("纯读上界 w+s: %.4f ms  %.1f GB/s (%.1f%% HBM)\n", ms, total_w * 1000.0 / ms / 1e9,
                100.0 * total_w * 1000.0 / ms / 1e9 / d.mem_bw_gbps);
    auto ls = [&] { wread_strided_kernel<<<8 * NSM, 256>>>(buf.Wp, buf.C, N, K); };
    ls();
    CUDA_CHECK_LAST();
    const double ms2 = bench_ms(ls, 5, 50);
    std::printf("纯读上界 strided(64B/2560B): %.4f ms  %.1f GB/s (%.1f%% HBM)\n", ms2,
                w_bytes_int4 * 1000.0 / ms2 / 1e9,
                100.0 * w_bytes_int4 * 1000.0 / ms2 / 1e9 / d.mem_bw_gbps);
  }

  std::vector<float> hC((size_t)MAXM * N);
  int CHK_M = (argc > 3) ? std::atoi(argv[3]) : MAXM;
  auto check_imma = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), buf.C, (size_t)CHK_M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 48; ++i) {
      const int m = (i * 1009 + 3) % CHK_M, n = (i * 997 + 7) % N;
      double s = 0;
      for (int k = 0; k < K; ++k) {
        const float av = (float)hAq[(size_t)m * K + k] * hAxs[(size_t)m * (K / GROUP) + k / GROUP];
        s += (double)av * (double)hWdeq[(size_t)n * K + k];
      }
      err = std::max(err, std::fabs((double)hC[(size_t)m * N + n] - s));
      ref = std::max(ref, std::fabs(s));
    }
    const bool ok = err / std::max(ref, 1e-6) < 3e-2;
    std::printf("  [%-20s] max_abs_err=%.3e (ref~%.1f) rel=%.2e %s\n", tag, err, ref,
                err / std::max(ref, 1e-6), ok ? "OK" : "FAIL");
    return ok;
  };

  std::vector<std::function<void(int, float*)>> specs;
  std::vector<std::string> names;

  if (strcmp(which, "all") == 0 || strcmp(which, "base") == 0) {
    add_base<16, 128, 2, 1, 4, 7>(specs, "base48_ns_s2k7", buf, N, K);
    names.push_back("base48_ns_s2k7");
  }
  if (strcmp(which, "all") == 0 || strcmp(which, "pipe") == 0) {
    add_pipe<16, 128, 2, 1, 4, 7>(specs, "pipe_s2k7", buf, N, K);
    names.push_back("pipe_s2k7");
    add_pipe<16, 128, 3, 1, 4, 7>(specs, "pipe_s3k7", buf, N, K);
    names.push_back("pipe_s3k7");
    add_pipe<16, 128, 4, 1, 4, 7>(specs, "pipe_s4k7", buf, N, K);
    names.push_back("pipe_s4k7");
    add_pipe<16, 128, 3, 1, 4, 5>(specs, "pipe_s3k5", buf, N, K);
    names.push_back("pipe_s3k5");
    add_pipe<16, 128, 3, 1, 4, 8>(specs, "pipe_s3k8", buf, N, K);
    names.push_back("pipe_s3k8");
    add_pipe<16, 128, 3, 1, 4, 4>(specs, "pipe_s3k4", buf, N, K);      // 1.03 波
    names.push_back("pipe_s3k4");
    add_pipe<16, 128, 3, 1, 4, 7, 5>(specs, "pipe_s3k7_b5", buf, N, K);  // 强制 5 CTA/SM
    names.push_back("pipe_s3k7_b5");
    add_pipe<16, 256, 2, 1, 8, 7>(specs, "pipe_bn256_s2k7", buf, N, K);
    names.push_back("pipe_bn256_s2k7");
    add_pipe<16, 256, 3, 1, 8, 7>(specs, "pipe_bn256_s3k7", buf, N, K);
    names.push_back("pipe_bn256_s3k7");
    add_pipe<16, 256, 2, 1, 8, 4>(specs, "pipe_bn256_s2k4", buf, N, K);
    names.push_back("pipe_bn256_s2k4");
    add_pipe<16, 512, 2, 1, 16, 4>(specs, "pipe_bn512_s2k4", buf, N, K);
    names.push_back("pipe_bn512_s2k4");
    // 单 barrier 版（每 stage 少一次 __syncthreads，攻 barrier stall）
    add_pipe<16, 128, 3, 1, 4, 7, 1, true>(specs, "one_s3k7", buf, N, K);
    names.push_back("one_s3k7");
    add_pipe<16, 128, 4, 1, 4, 7, 1, true>(specs, "one_s4k7", buf, N, K);
    names.push_back("one_s4k7");
    add_pipe<16, 128, 3, 1, 4, 7, 5, true>(specs, "one_s3k7_b5", buf, N, K);
    names.push_back("one_s3k7_b5");
    add_pipe<16, 128, 3, 1, 4, 5, 5, true>(specs, "one_s3k5_b5", buf, N, K);
    names.push_back("one_s3k5_b5");
    add_pipe<16, 256, 3, 1, 8, 7, 1, true>(specs, "one_bn256_s3k7", buf, N, K);
    names.push_back("one_bn256_s3k7");
    // K-blocked 权重布局（Wt[g][N][64B]）：stage 内 BN 行连续 8KB
    add_pipe<16, 128, 3, 1, 4, 7, 1, false, true>(specs, "tile_s3k7", buf, N, K);
    names.push_back("tile_s3k7");
    add_pipe<16, 128, 3, 1, 4, 7, 5, false, true>(specs, "tile_s3k7_b5", buf, N, K);
    names.push_back("tile_s3k7_b5");
    add_pipe<16, 128, 2, 1, 4, 7, 1, false, true>(specs, "tile_s2k7", buf, N, K);
    names.push_back("tile_s2k7");
    add_pipe<16, 256, 2, 1, 8, 7, 1, false, true>(specs, "tile_bn256_s2k7", buf, N, K);
    names.push_back("tile_bn256_s2k7");
    add_pipe<16, 128, 3, 1, 4, 7, 1, true, true>(specs, "tile_one_s3k7", buf, N, K);
    names.push_back("tile_one_s3k7");
    add_pipe<16, 128, 2, 1, 4, 5, 1, false, true>(specs, "tile_s2k5", buf, N, K);
    names.push_back("tile_s2k5");
    add_pipe<16, 128, 2, 1, 4, 8, 1, false, true>(specs, "tile_s2k8", buf, N, K);
    names.push_back("tile_s2k8");
    add_pipe<16, 128, 2, 1, 4, 4, 1, false, true>(specs, "tile_s2k4", buf, N, K);
    names.push_back("tile_s2k4");
    add_pipe<16, 256, 2, 1, 8, 5, 1, false, true>(specs, "tile_bn256_s2k5", buf, N, K);
    names.push_back("tile_bn256_s2k5");
    add_pipe<16, 128, 3, 1, 4, 8, 1, false, true>(specs, "tile_s3k8", buf, N, K);
    names.push_back("tile_s3k8");
    add_pipe<16, 128, 2, 1, 4, 10, 1, false, true>(specs, "tile_s2k10", buf, N, K);
    names.push_back("tile_s2k10");
    add_pipe<16, 128, 2, 1, 4, 16, 1, false, true>(specs, "tile_s2k16", buf, N, K);
    names.push_back("tile_s2k16");
    add_pipe<16, 128, 2, 1, 4, 8, 5, false, true>(specs, "tile_s2k8_b5", buf, N, K);
    names.push_back("tile_s2k8_b5");
  }

  if (strcmp(which, "warp") == 0) {
    // 50 篇代表配置作为同进程对照
    add_pipe<16, 128, 2, 1, 4, 8, 5, false, true>(specs, "base_tile_s2k8_b5", buf, N, K);
    names.push_back("base_tile_s2k8_b5");
    add_pipe<16, 128, 2, 1, 4, 4, 1, false, true>(specs, "base_tile_s2k4", buf, N, K);
    names.push_back("base_tile_s2k4");
    add_pipe<16, 128, 3, 1, 4, 7, 1, false, true>(specs, "base_tile_s3k7", buf, N, K);
    names.push_back("base_tile_s3k7");
    // warp-private（无 CTA barrier），K-blocked 权重
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 0>(specs, "w_s2k8", buf, N, K);
    names.push_back("w_s2k8");
    add_warp<16, 128, 3, 1, 4, 8, 1, true, 0>(specs, "w_s3k8", buf, N, K);
    names.push_back("w_s3k8");
    add_warp<16, 128, 4, 1, 4, 8, 1, true, 0>(specs, "w_s4k8", buf, N, K);
    names.push_back("w_s4k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 0>(specs, "w_s3k4", buf, N, K);
    names.push_back("w_s3k4");
    add_warp<16, 128, 3, 1, 4, 5, 1, true, 0>(specs, "w_s3k5", buf, N, K);
    names.push_back("w_s3k5");
    add_warp<16, 128, 3, 1, 4, 7, 1, true, 0>(specs, "w_s3k7", buf, N, K);
    names.push_back("w_s3k7");
    add_warp<16, 128, 4, 1, 4, 5, 1, true, 0>(specs, "w_s4k5", buf, N, K);
    names.push_back("w_s4k5");
    add_warp<16, 128, 6, 1, 4, 8, 1, true, 0>(specs, "w_s6k8", buf, N, K);
    names.push_back("w_s6k8");
    add_warp<16, 128, 3, 1, 4, 8, 1, true, 1>(specs, "wA_s3k8", buf, N, K);
    names.push_back("wA_s3k8");
    add_warp<16, 128, 4, 1, 4, 8, 1, true, 1>(specs, "wA_s4k8", buf, N, K);
    names.push_back("wA_s4k8");
    add_warp<16, 128, 4, 1, 4, 4, 1, true, 1>(specs, "wA_s4k4", buf, N, K);
    names.push_back("wA_s4k4");
    add_warp<16, 128, 2, 1, 4, 4, 1, true, 1>(specs, "wA_s2k4", buf, N, K);
    names.push_back("wA_s2k4");
    add_warp<16, 128, 2, 1, 4, 6, 1, true, 1>(specs, "wA_s2k6", buf, N, K);
    names.push_back("wA_s2k6");
    add_warp<16, 128, 2, 1, 4, 7, 1, true, 1>(specs, "wA_s2k7", buf, N, K);
    names.push_back("wA_s2k7");
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    add_warp<16, 128, 3, 1, 4, 5, 1, true, 1>(specs, "wA_s3k5", buf, N, K);
    names.push_back("wA_s3k5");
    add_warp<16, 128, 3, 1, 4, 6, 1, true, 1>(specs, "wA_s3k6", buf, N, K);
    names.push_back("wA_s3k6");
    add_warp<16, 128, 3, 1, 4, 7, 1, true, 1>(specs, "wA_s3k7", buf, N, K);
    names.push_back("wA_s3k7");
    add_warp<16, 128, 2, 1, 4, 10, 1, true, 1>(specs, "wA_s2k10", buf, N, K);
    names.push_back("wA_s2k10");
    add_warp<16, 128, 2, 1, 4, 16, 1, true, 1>(specs, "wA_s2k16", buf, N, K);
    names.push_back("wA_s2k16");
    // 强制更高 occupancy（launch_bounds MINB）
    add_warp<16, 128, 2, 1, 4, 8, 8, true, 1>(specs, "wA_s2k8_b8", buf, N, K);
    names.push_back("wA_s2k8_b8");
    add_warp<16, 128, 2, 1, 4, 8, 10, true, 1>(specs, "wA_s2k8_b10", buf, N, K);
    names.push_back("wA_s2k8_b10");
    add_warp<16, 128, 2, 1, 4, 4, 8, true, 1>(specs, "wA_s2k4_b8", buf, N, K);
    names.push_back("wA_s2k4_b8");
    add_warp<16, 128, 3, 1, 4, 8, 6, true, 1>(specs, "wA_s3k8_b6", buf, N, K);
    names.push_back("wA_s3k8_b6");
    add_warp<16, 128, 3, 1, 4, 4, 8, true, 1>(specs, "wA_s3k4_b8", buf, N, K);
    names.push_back("wA_s3k4_b8");
    // WMODE=2：紧凑 A（只存 M 行，smem 随 M 缩小）
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 2>(specs, "wC_s2k8", buf, N, K);
    names.push_back("wC_s2k8");
    add_warp<16, 128, 3, 1, 4, 8, 1, true, 2>(specs, "wC_s3k8", buf, N, K);
    names.push_back("wC_s3k8");
    add_warp<16, 128, 2, 1, 4, 4, 1, true, 2>(specs, "wC_s2k4", buf, N, K);
    names.push_back("wC_s2k4");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 2>(specs, "wC_s3k4", buf, N, K);
    names.push_back("wC_s3k4");
    add_warp<16, 128, 2, 1, 4, 8, 8, true, 2>(specs, "wC_s2k8_b8", buf, N, K);
    names.push_back("wC_s2k8_b8");
    add_warp<16, 128, 2, 1, 4, 8, 10, true, 2>(specs, "wC_s2k8_b10", buf, N, K);
    names.push_back("wC_s2k8_b10");
    add_warp<16, 128, 2, 1, 4, 6, 8, true, 2>(specs, "wC_s2k6_b8", buf, N, K);
    names.push_back("wC_s2k6_b8");
    add_warp<16, 128, 2, 1, 4, 4, 10, true, 2>(specs, "wC_s2k4_b10", buf, N, K);
    names.push_back("wC_s2k4_b10");
    add_warp<16, 128, 3, 1, 4, 4, 8, true, 2>(specs, "wC_s3k4_b8", buf, N, K);
    names.push_back("wC_s3k4_b8");
    // SUB：一个 stage 打包 SUB 个 group，一次 wait 覆盖更多字节（降 long_scoreboard）
    add_warp<16, 128, 2, 1, 4, 4, 1, true, 1, 2>(specs, "wA_s2k4_S2", buf, N, K);
    names.push_back("wA_s2k4_S2");
    add_warp<16, 128, 2, 1, 4, 4, 1, true, 1, 5>(specs, "wA_s2k4_S5", buf, N, K);
    names.push_back("wA_s2k4_S5");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 1, 2>(specs, "wA_s2k5_S2", buf, N, K);
    names.push_back("wA_s2k5_S2");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 1, 4>(specs, "wA_s2k5_S4", buf, N, K);
    names.push_back("wA_s2k5_S4");
    add_warp<16, 128, 3, 1, 4, 5, 1, true, 1, 2>(specs, "wA_s3k5_S2", buf, N, K);
    names.push_back("wA_s3k5_S2");
    add_warp<16, 128, 3, 1, 4, 5, 1, true, 1, 4>(specs, "wA_s3k5_S4", buf, N, K);
    names.push_back("wA_s3k5_S4");
    add_warp<16, 128, 2, 1, 4, 10, 1, true, 1, 2>(specs, "wA_s2k10_S2", buf, N, K);
    names.push_back("wA_s2k10_S2");
    add_warp<16, 128, 2, 1, 4, 10, 1, true, 1, 4>(specs, "wA_s2k10_S4", buf, N, K);
    names.push_back("wA_s2k10_S4");
    add_warp<16, 128, 3, 1, 4, 10, 1, true, 1, 2>(specs, "wA_s3k10_S2", buf, N, K);
    names.push_back("wA_s3k10_S2");
    add_warp<16, 128, 2, 1, 4, 20, 1, true, 1, 2>(specs, "wA_s2k20_S2", buf, N, K);
    names.push_back("wA_s2k20_S2");
    add_warp<16, 128, 2, 1, 4, 2, 1, true, 1, 2>(specs, "wA_s2k2_S2", buf, N, K);
    names.push_back("wA_s2k2_S2");
    add_warp<16, 128, 2, 1, 4, 2, 1, true, 1, 4>(specs, "wA_s2k2_S4", buf, N, K);
    names.push_back("wA_s2k2_S4");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 0>(specs, "w_s2k5", buf, N, K);
    names.push_back("w_s2k5");
    add_warp<16, 128, 2, 1, 4, 6, 1, true, 0>(specs, "w_s2k6", buf, N, K);
    names.push_back("w_s2k6");
    add_warp<16, 128, 4, 1, 2, 8, 1, true, 0>(specs, "w_wn2_s4k8", buf, N, K);
    names.push_back("w_wn2_s4k8");
    add_warp<16, 256, 3, 1, 8, 8, 1, true, 0>(specs, "w_bn256_s3k8", buf, N, K);
    names.push_back("w_bn256_s3k8");
    add_warp<16, 256, 4, 1, 8, 8, 1, true, 1>(specs, "wA_bn256_s4k8", buf, N, K);
    names.push_back("wA_bn256_s4k8");
  }

  if (strcmp(which, "tma") == 0) {
    // 51 篇最佳作为同进程对照（无 TMA 的 warp-private）
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 1, 2>(specs, "wA_s2k5_S2", buf, N, K);
    names.push_back("wA_s2k5_S2");
    // TMA warp-private
    add_tma_warp<16, 128, 2, 4, 4>(specs, "t_s2k4", buf, N, K);
    names.push_back("t_s2k4");
    add_tma_warp<16, 128, 2, 4, 5>(specs, "t_s2k5", buf, N, K);
    names.push_back("t_s2k5");
    add_tma_warp<16, 128, 2, 4, 6>(specs, "t_s2k6", buf, N, K);
    names.push_back("t_s2k6");
    add_tma_warp<16, 128, 2, 4, 7>(specs, "t_s2k7", buf, N, K);
    names.push_back("t_s2k7");
    add_tma_warp<16, 128, 2, 4, 8>(specs, "t_s2k8", buf, N, K);
    names.push_back("t_s2k8");
    add_tma_warp<16, 128, 3, 4, 4>(specs, "t_s3k4", buf, N, K);
    names.push_back("t_s3k4");
    add_tma_warp<16, 128, 3, 4, 5>(specs, "t_s3k5", buf, N, K);
    names.push_back("t_s3k5");
    add_tma_warp<16, 128, 3, 4, 6>(specs, "t_s3k6", buf, N, K);
    names.push_back("t_s3k6");
    add_tma_warp<16, 128, 3, 4, 7>(specs, "t_s3k7", buf, N, K);
    names.push_back("t_s3k7");
    add_tma_warp<16, 128, 3, 4, 8>(specs, "t_s3k8", buf, N, K);
    names.push_back("t_s3k8");
    add_tma_warp<16, 128, 4, 4, 4>(specs, "t_s4k4", buf, N, K);
    names.push_back("t_s4k4");
    add_tma_warp<16, 128, 4, 4, 5>(specs, "t_s4k5", buf, N, K);
    names.push_back("t_s4k5");
    add_tma_warp<16, 128, 4, 4, 8>(specs, "t_s4k8", buf, N, K);
    names.push_back("t_s4k8");
    add_tma_warp<16, 128, 5, 4, 8>(specs, "t_s5k8", buf, N, K);
    names.push_back("t_s5k8");
  }

  if (strcmp(which, "tma2") == 0) {
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    // STAGES × KSPLIT 精细扫
    add_tma_warp<16, 128, 2, 4, 4>(specs, "t_s2k4", buf, N, K);
    names.push_back("t_s2k4");
    add_tma_warp<16, 128, 3, 4, 4>(specs, "t_s3k4", buf, N, K);
    names.push_back("t_s3k4");
    add_tma_warp<16, 128, 2, 4, 8>(specs, "t_s2k8", buf, N, K);
    names.push_back("t_s2k8");
    add_tma_warp<16, 128, 3, 4, 8>(specs, "t_s3k8", buf, N, K);
    names.push_back("t_s3k8");
    add_tma_warp<16, 128, 4, 4, 8>(specs, "t_s4k8", buf, N, K);
    names.push_back("t_s4k8");
    add_tma_warp<16, 128, 6, 4, 8>(specs, "t_s6k8", buf, N, K);
    names.push_back("t_s6k8");
    add_tma_warp<16, 128, 2, 4, 10>(specs, "t_s2k10", buf, N, K);
    names.push_back("t_s2k10");
    add_tma_warp<16, 128, 3, 4, 10>(specs, "t_s3k10", buf, N, K);
    names.push_back("t_s3k10");
    add_tma_warp<16, 128, 4, 4, 10>(specs, "t_s4k10", buf, N, K);
    names.push_back("t_s4k10");
    add_tma_warp<16, 128, 3, 4, 16>(specs, "t_s3k16", buf, N, K);
    names.push_back("t_s3k16");
    add_tma_warp<16, 128, 4, 4, 16>(specs, "t_s4k16", buf, N, K);
    names.push_back("t_s4k16");
    // 强制 occupancy
    add_tma_warp<16, 128, 3, 4, 8, 6>(specs, "t_s3k8_b6", buf, N, K);
    names.push_back("t_s3k8_b6");
    add_tma_warp<16, 128, 2, 4, 8, 7>(specs, "t_s2k8_b7", buf, N, K);
    names.push_back("t_s2k8_b7");
    // BN=256
    add_tma_warp<16, 256, 3, 8, 8>(specs, "t_bn256_s3k8", buf, N, K);
    names.push_back("t_bn256_s3k8");
  }

  if (strcmp(which, "tma3") == 0) {
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    add_tma_warp<16, 128, 2, 4, 8>(specs, "t_s2k8", buf, N, K);
    names.push_back("t_s2k8");
    add_tma_warp<16, 128, 3, 4, 8>(specs, "t_s3k8", buf, N, K);
    names.push_back("t_s3k8");
    add_tma_warp<16, 128, 2, 4, 10>(specs, "t_s2k10", buf, N, K);
    names.push_back("t_s2k10");
    add_tma_warp<16, 128, 2, 4, 12>(specs, "t_s2k12", buf, N, K);
    names.push_back("t_s2k12");
    add_tma_warp<16, 128, 3, 4, 12>(specs, "t_s3k12", buf, N, K);
    names.push_back("t_s3k12");
    add_tma_warp<16, 128, 2, 4, 14>(specs, "t_s2k14", buf, N, K);
    names.push_back("t_s2k14");
    add_tma_warp<16, 128, 2, 4, 20>(specs, "t_s2k20", buf, N, K);
    names.push_back("t_s2k20");
    add_tma_warp<16, 128, 3, 4, 20>(specs, "t_s3k20", buf, N, K);
    names.push_back("t_s3k20");
    add_tma_warp<16, 128, 2, 4, 4>(specs, "t_s2k4", buf, N, K);
    names.push_back("t_s2k4");
    add_tma_warp<16, 64, 3, 2, 8>(specs, "t_bn64_s3k8", buf, N, K);
    names.push_back("t_bn64_s3k8");
    add_tma_warp<16, 64, 4, 2, 8>(specs, "t_bn64_s4k8", buf, N, K);
    names.push_back("t_bn64_s4k8");
  }

  if (strcmp(which, "abl") == 0) {
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    // 同一 TMA kernel：PERM=false 用行主序连续 W（warp 内 2KB），PERM=true 用
    // [c][n] 置换布局。对比 bank conflict 的影响。
    add_tma_warp<16, 128, 2, 4, 8, 1, 1, false>(specs, "t_s2k8_row", buf, N, K);
    names.push_back("t_s2k8_row");
    add_tma_warp<16, 128, 2, 4, 8, 1, 1, true>(specs, "t_s2k8_perm", buf, N, K);
    names.push_back("t_s2k8_perm");
    add_tma_warp<16, 128, 3, 4, 8, 1, 1, false>(specs, "t_s3k8_row", buf, N, K);
    names.push_back("t_s3k8_row");
    add_tma_warp<16, 128, 3, 4, 8, 1, 1, true>(specs, "t_s3k8_perm", buf, N, K);
    names.push_back("t_s3k8_perm");
    add_tma_warp<16, 128, 2, 4, 10, 1, 1, false>(specs, "t_s2k10_row", buf, N, K);
    names.push_back("t_s2k10_row");
    add_tma_warp<16, 128, 2, 4, 10, 1, 1, true>(specs, "t_s2k10_perm", buf, N, K);
    names.push_back("t_s2k10_perm");
  }

  if (strcmp(which, "best52") == 0) {
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    add_tma_warp<16, 128, 2, 4, 8>(specs, "t_s2k8", buf, N, K);
    names.push_back("t_s2k8");
    add_tma_warp<16, 128, 3, 4, 8>(specs, "t_s3k8", buf, N, K);
    names.push_back("t_s3k8");
    add_tma_warp<16, 128, 2, 4, 10>(specs, "t_s2k10", buf, N, K);
    names.push_back("t_s2k10");
    add_tma_warp<16, 128, 2, 4, 4>(specs, "t_s2k4", buf, N, K);
    names.push_back("t_s2k4");
  }

  if (strcmp(which, "final") == 0) {
    add_pipe<16, 128, 2, 1, 4, 8, 5, false, true>(specs, "base_tile_s2k8_b5", buf, N, K);
    names.push_back("base_tile_s2k8_b5");
    add_pipe<16, 128, 2, 1, 4, 4, 1, false, true>(specs, "base_tile_s2k4", buf, N, K);
    names.push_back("base_tile_s2k4");
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 0>(specs, "w_s2k8", buf, N, K);
    names.push_back("w_s2k8");
    add_warp<16, 128, 3, 1, 4, 8, 1, true, 0>(specs, "w_s3k8", buf, N, K);
    names.push_back("w_s3k8");
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 1>(specs, "wA_s2k8", buf, N, K);
    names.push_back("wA_s2k8");
    add_warp<16, 128, 2, 1, 4, 6, 1, true, 1>(specs, "wA_s2k6", buf, N, K);
    names.push_back("wA_s2k6");
    add_warp<16, 128, 3, 1, 4, 8, 1, true, 1>(specs, "wA_s3k8", buf, N, K);
    names.push_back("wA_s3k8");
    add_warp<16, 128, 3, 1, 4, 4, 1, true, 1>(specs, "wA_s3k4", buf, N, K);
    names.push_back("wA_s3k4");
    add_warp<16, 128, 2, 1, 4, 8, 1, true, 2>(specs, "wC_s2k8", buf, N, K);
    names.push_back("wC_s2k8");
    add_warp<16, 128, 2, 1, 4, 8, 10, true, 1>(specs, "wA_s2k8_b10", buf, N, K);
    names.push_back("wA_s2k8_b10");
    add_warp<16, 128, 6, 1, 4, 8, 1, true, 0>(specs, "w_s6k8", buf, N, K);
    names.push_back("w_s6k8");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 1, 2>(specs, "wA_s2k5_S2", buf, N, K);
    names.push_back("wA_s2k5_S2");
    add_warp<16, 128, 2, 1, 4, 5, 1, true, 1, 4>(specs, "wA_s2k5_S4", buf, N, K);
    names.push_back("wA_s2k5_S4");
  }

  std::vector<int> active;
  for (size_t si = 0; si < specs.size(); ++si)
    if (!*only || names[si] == only) active.push_back((int)si);

  std::printf("\n=== 精度校验（M=%d, 抽样 48 点）===\n", CHK_M);
  for (int si : active) {
    cudaMemsetAsync(buf.C, 0, (size_t)CHK_M * N * 4);
    specs[si](CHK_M, nullptr);
    CUDA_CHECK_LAST();
    check_imma(names[si].c_str());
  }

  const int Ms[] = {1, 2, 3, 4, 6, 8, 12, 16};
  const int NM = sizeof(Ms) / sizeof(int);
  std::printf("\n=== M 扫描（同进程、warmup=5 iters=50）===\n");
  std::printf("%-20s", "kernel\\M");
  for (int mi = 0; mi < NM; ++mi) std::printf("%10d", Ms[mi]);
  std::printf("\n");
  std::vector<std::vector<double>> tab(specs.size(), std::vector<double>(NM, 1e30));
  for (int si : active) {
    std::printf("%-20s", names[si].c_str());
    for (int mi = 0; mi < NM; ++mi) {
      const int M = Ms[mi];
      auto launch = [&] {
        cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
        specs[si](M, nullptr);
      };
      launch();
      CUDA_CHECK_LAST();
      const double ms = bench_ms(launch, 5, 50);
      tab[si][mi] = ms;
      std::printf("%8.4f  ", ms);
    }
    std::printf("\n");
  }
  if (*only) return 0;

  // 每个 M 的最优 + 有效权重带宽
  std::printf("\n=== 每个 M 的最佳 kernel（含 base 与全部变体；GB/s 按 W+s 字节）===\n");
  std::printf("%-6s %-20s %9s %9s %9s %9s\n", "M", "best", "ms", "TFLOPS", "W+s GB/s", "%HBM");
  double sum_best = 0, sum_base = 0;
  for (int mi = 0; mi < NM; ++mi) {
    int bi = 0;
    for (size_t si = 1; si < specs.size(); ++si)
      if (tab[si][mi] < tab[bi][mi]) bi = si;
    const int M = Ms[mi];
    const double ms = tab[bi][mi];
    sum_best += ms;
    for (size_t si = 0; si < specs.size(); ++si)
      if (names[si] == "base48_ns_s2k7") sum_base += tab[si][mi];
    const double gbs = total_w * 1000.0 / ms / 1e9;
    std::printf("%-6d %-20s %9.4f %9.2f %9.1f %8.1f%%\n", M, names[bi].c_str(), ms,
                to_tflops(2.0 * M * N * K, ms), gbs, 100.0 * gbs / d.mem_bw_gbps);
  }
  std::printf("汇总（8 个 M 点求和）：best=%.4f ms   base48=%.4f ms   best/base=%.3f\n", sum_best,
              sum_base, sum_best / sum_base);

  // 持久化扫描（单独一组：不同 CHUNK/P）
  if (strcmp(which, "all") == 0 || strcmp(which, "persist") == 0) {
    std::printf("\n=== 持久化版 M 扫描（CHUNK/P 组合，同进程）===\n");
    std::printf("%-20s", "kernel\\M");
    for (int mi = 0; mi < NM; ++mi) std::printf("%10d", Ms[mi]);
    std::printf("\n");
    std::vector<std::function<void(int, float*)>> ps;
    std::vector<std::string> pn;
    auto add_persist = [&](auto fn, const char* nm, int CHUNK, int P, size_t shm, int nthr) {
      CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
      std::printf("  [%s] P=%d smem=%zu\n", nm, P, shm);
      const int ntiles = N / 128;
      ps.push_back([=](int M, float*) {
        const int total_items = ntiles * ((K / LARGE_BK) / CHUNK);
        cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
        fn<<<P, nthr, shm>>>(buf.Aq, buf.AxsT, buf.AsumT8, buf.Wp, buf.WsT, buf.C, M, N, K,
                             total_items, P);
      });
      pn.push_back(nm);
    };
#define ADD_PERSIST(NAME, ST, CH, MB, P_CTAS)                                                        \
    add_persist(imma_persist_kernel<16, 128, ST, CH, 1, 4, MB>, NAME, CH, P_CTAS,                    \
                pipe_smem(16, 128, ST), 128)
    {
      const int P = 2 * NSM, P4 = 4 * NSM;
      ADD_PERSIST("pers_c8_s3_p2sm", 3, 8, 2, P);
      ADD_PERSIST("pers_c8_s3_p4sm", 3, 8, 2, P4);
      ADD_PERSIST("pers_c10_s3_p2sm", 3, 10, 2, P);
      ADD_PERSIST("pers_c8_s4_p2sm", 4, 8, 2, P);
      ADD_PERSIST("pers_c8_s2_p2sm", 2, 8, 2, P);
      ADD_PERSIST("pers_c5_s3_p2sm", 3, 5, 2, P);
      ADD_PERSIST("pers_c4_s3_p2sm", 3, 4, 2, P);
      ADD_PERSIST("pers_c40_s3_p2sm", 3, 40, 2, P);  // CHUNK=全 K → 无 atomic
    }
#undef ADD_PERSIST
    std::vector<std::vector<double>> ptab(ps.size(), std::vector<double>(NM, 1e30));
    for (size_t si = 0; si < ps.size(); ++si) {
      std::printf("%-20s", pn[si].c_str());
      for (int mi = 0; mi < NM; ++mi) {
        const int M = Ms[mi];
        auto launch = [&] { ps[si](M, nullptr); };
        launch();
        CUDA_CHECK_LAST();
        ptab[si][mi] = bench_ms(launch, 5, 50);
        std::printf("%8.4f  ", ptab[si][mi]);
      }
      std::printf("\n");
    }
  }

  CUDA_CHECK(cudaFree(buf.A));
  CUDA_CHECK(cudaFree(buf.Aq));
  CUDA_CHECK(cudaFree(buf.Axs));
  CUDA_CHECK(cudaFree(buf.Axsum));
  CUDA_CHECK(cudaFree(buf.AxsT));
  CUDA_CHECK(cudaFree(buf.AsumT8));
  CUDA_CHECK(cudaFree(buf.Wp));
  CUDA_CHECK(cudaFree(buf.Ws));
  CUDA_CHECK(cudaFree(buf.WsT));
  CUDA_CHECK(cudaFree(buf.C));
  return 0;
}
