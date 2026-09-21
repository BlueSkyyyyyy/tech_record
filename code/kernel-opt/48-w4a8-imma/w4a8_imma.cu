// 48 W4A8 的整数张量核（IMMA）小 M GEMM（主题 26h）
//
// 46/47 把 M∈[3,16] 交给了 `mma.sync.m16n8k16`（bf16）的 BM=16 小 tile，但它
// 的墙不是访存而是**反量化的 ALU**：ncu 显示 mma_b16_k8 里反量化的
// ALU+FMA 指令占 78%、张量核只占 2.3%（47 篇）。要把这块砍掉，必须换数制。
//
// 本篇把「反量化到 bf16 + bf16 mma」换成「int8 展开 + 整数张量核」：
//   · 激活 `quant_x_kernel` 动态量化成 int8（per-128 group，45 篇原样复用）；
//   · 权重 int4 在 GPU 上用 `PRMT` + `__vsubss4` 展开成符号 int8（每 8 个
//     nibble 只要 1 条 `__byte_perm` + 1 条 `vsubss4`）；
//   · 用 `mma.sync.m16n8k32.s32.s8.s8.s32`（IMMA，k=32 一条）替代 k=16 的
//     bf16 mma，张量核指令数直接减半；
//   · per-128 group scale 在「一个 stage = 一个 group」处用 fp32 折算回累加器
//     （int32 累加器每个 group 清零）。
//
// 关键实现点：int8 的 mma 片段每 lane 恰好一个 32-bit word（A: 4 个、B: 2 个），
// 所以**完全不需要 ldmatrix**——直接从行主序 smem 取 word 即可（见 imma_test.cu）。
// 而且 B 可以直接从 int4 的 smem 现场展开，省掉「展开后再写回 smem」的往返。
//
// shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128。
// 默认 N=17408（MLP up/gate），K=5120。int4 权重 44.6MB + scale 2.8MB。
//
// 运行（含 wgmma，需 sm_90a）：
//   ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
//     scripts/run.sh 48-w4a8-imma/w4a8_imma.cu [which] [only] [chkM]
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
constexpr int BK = 64;      // bf16 SW128 atom
constexpr double BF16_PEAK = 989.0;
constexpr double INT8_PEAK = 1978.0;  // int8/fp8 tensor TOPS

// ===========================================================================
//  通用小工具
// ===========================================================================
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}
__device__ __forceinline__ uint32_t pack2b(bf16 lo, bf16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// int8 版本：m16n8k32，累加器是 4×s32。
__device__ __forceinline__ void imma_m16n8k32(int c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
// 无符号权重版：nibble 直接用 0..15，配合激进的 −8 修正由折算阶段扣掉
// `8·Σq_a`。省掉每个 B 片段的 `__vsubss4`（ALU 的一大块）。
__device__ __forceinline__ void imma_m16n8k32_u8(int c[4], const uint32_t a[4],
                                                  const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}


// 把 8 个 int4（一个 uint32）展开成 8 个符号 int8（q=nibble-8），装进 2 个 uint32。
//   lo = [n0,n2,n4,n6]（每 byte 低位 nibble），hi = [n1,n3,n5,n7]
//   PRMT 交错成 [n0..n3]、[n4..n7]，再用 SIMD 饱和减 8 做逐 byte 的 -8。
__device__ __forceinline__ void expand8(uint32_t v, uint32_t& a, uint32_t& b) {
  const uint32_t lo = v & 0x0F0F0F0Fu;
  const uint32_t hi = (v >> 4) & 0x0F0F0F0Fu;
  const uint32_t a0 = __byte_perm(lo, hi, 0x5140);  // [n0 n1 n2 n3]
  const uint32_t b0 = __byte_perm(lo, hi, 0x7362);  // [n4 n5 n6 n7]
  a = __vsubss4(a0, 0x08080808u);
  b = __vsubss4(b0, 0x08080808u);
}
// 从 uint16（2 个 int4 字节 = 4 个 nibble）展开出 1 个 uint32 的 4 个符号 int8。
__device__ __forceinline__ uint32_t expand4(uint32_t v) {
  const uint32_t lo = v & 0x0F0F0Fu;
  const uint32_t hi = (v >> 4) & 0x0F0F0Fu;
  const uint32_t r = __byte_perm(lo, hi, 0x5140);
  return __vsubss4(r, 0x08080808u);
}
// 同上但保持 0..15（无符号），配合 u8.s8 的 mma。
__device__ __forceinline__ uint32_t expand4u(uint32_t v) {
  const uint32_t lo = v & 0x0F0Fu;
  const uint32_t hi = (v >> 4) & 0x0F0Fu;
  return __byte_perm(lo, hi, 0x5140);
}

// ===========================================================================
//  激活动态量化（45 篇原样）：x bf16 → int8，per-128-group scale。
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
  if (lane == 0) Axs[gid] = s;
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
  if (lane == 0) Axsum[gid] = (float)lsum;
}

// ===========================================================================
//  (A) bf16 基线：46/47 篇的 gemm_mma_w4a16_kernel（BKi=64，int4→bf16 反量化）
// ===========================================================================
template <int BM, int BN, int BKi, int STAGES, int WM, int WN, int KSPLIT = 1>
__global__ void __launch_bounds__(WM* WN* 32) gemm_mma_w4a16_kernel(
    const bf16* __restrict__ A, const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
    float* __restrict__ C, int M, int N, int K) {
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  constexpr int MTM = WARP_M / 16;
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = BKi + 8;
  constexpr int BKP = BKi + 8;
  constexpr int NGRP = 5120 / GROUP;
  constexpr int NW4 = BKi / 8;

  extern __shared__ __align__(16) char smem[];
  bf16* Araw = reinterpret_cast<bf16*>(smem);
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(Araw + (size_t)STAGES * BM * ASP);
  bf16* Bs = reinterpret_cast<bf16*>(Wraw + (size_t)STAGES * BN * (BKi / 2));

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wid = tid >> 5;
  const int warp_row = wid / WN;
  const int warp_col = wid % WN;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int nblk = K / BKi;
  const int kb0 = (nblk * (int)blockIdx.z) / KSPLIT;
  const int kb1 = (nblk * (int)(blockIdx.z + 1)) / KSPLIT;

  auto load_tile = [&](int st, int k0) {
    bf16* a = Araw + (size_t)st * BM * ASP;
    uint8_t* wp = Wraw + (size_t)st * BN * (BKi / 2);
    for (int i = tid; i < BM * (BKi / 8); i += NTHREADS) {
      const int r = i / (BKi / 8), c8 = (i % (BKi / 8)) * 8;
      const int gr = block_row + r;
      if (gr < M) __pipeline_memcpy_async(&a[r * ASP + c8], &A[(size_t)gr * K + k0 + c8], 16);
      else *reinterpret_cast<uint4*>(&a[r * ASP + c8]) = make_uint4(0, 0, 0, 0);
    }
    for (int i = tid; i < BN * (BKi / 2 / 16); i += NTHREADS) {
      const int r = i / (BKi / 2 / 16), c = (i % (BKi / 2 / 16)) * 16;
      __pipeline_memcpy_async(&wp[r * (BKi / 2) + c],
                              &Wp[(size_t)(block_col + r) * (K / 2) + k0 / 2 + c], 16);
    }
    __pipeline_commit();
  };
  auto dequant = [&](int st, int k0) {
    uint8_t* wp = Wraw + (size_t)st * BN * (BKi / 2);
    bf16* b = Bs + (size_t)st * BN * BKP;
    const int g = k0 / GROUP;
    for (int i = tid; i < BN * NW4; i += NTHREADS) {
      const int n = i / NW4, w = i % NW4;
      const uint32_t word = *reinterpret_cast<const uint32_t*>(&wp[n * (BKi / 2) + w * 4]);
      const float s = __ldg(&Ws[(size_t)(block_col + n) * NGRP + g]);
      bf16 out[8];
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int q0 = (int)((word >> (8 * q)) & 0xF) - 8;
        const int q1 = (int)((word >> (8 * q + 4)) & 0xF) - 8;
        out[2 * q] = __float2bfloat16((float)q0 * s);
        out[2 * q + 1] = __float2bfloat16((float)q1 * s);
      }
      *reinterpret_cast<uint4*>(&b[n * BKP + w * 8]) = *reinterpret_cast<const uint4*>(out);
    }
  };

  float acc[MTM][MTN][4];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  load_tile(0, kb0 * BKi);
  int stage = 0;
  const int nt = kb1 - kb0;
#pragma unroll 1
  for (int t = 0; t < nt; ++t) {
    const int k0 = (kb0 + t) * BKi;
    if (t + 1 < nt) load_tile(stage ^ 1, k0 + BKi);
    else __pipeline_commit();
    __pipeline_wait_prior(t + 1 < nt ? 1 : 0);
    __syncthreads();
    dequant(stage, k0);
    __syncthreads();
    const bf16* a = Araw + (size_t)stage * BM * ASP;
    const bf16* b = Bs + (size_t)stage * BN * BKP;
#pragma unroll
    for (int kk = 0; kk < BKi / 16; ++kk) {
      uint32_t af[MTM][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i) {
        const int row = (lane & 15);
        const int col = (lane >> 4) * 8;
        ldmatrix_x4((uint32_t)__cvta_generic_to_shared(
                        &a[(warp_row * WARP_M + i * 16 + row) * ASP + kk * 16 + col]),
                    af[i]);
      }
      uint32_t bf[MTN][2];
#pragma unroll
      for (int g = 0; g < MTN / 2; ++g) {
        const int n_row = (lane & 7) + ((lane >> 4) & 1) * 8;
        const int k_off = ((lane >> 3) & 1) * 8;
        uint32_t d[4];
        ldmatrix_x4((uint32_t)__cvta_generic_to_shared(
                        &b[(warp_col * WARP_N + g * 16 + n_row) * BKP + kk * 16 + k_off]),
                    d);
        bf[g * 2][0] = d[0];
        bf[g * 2][1] = d[1];
        bf[g * 2 + 1][0] = d[2];
        bf[g * 2 + 1][1] = d[3];
      }
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j) mma_m16n8k16(acc[i][j], af[i], bf[j]);
    }
    __syncthreads();
    stage ^= 1;
  }

  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      const int r0 = block_row + warp_row * WARP_M + i * 16 + group;
      const int c0 = block_col + warp_col * WARP_N + j * 8 + tig * 2;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + (q >= 2 ? 8 : 0);
        const int c = c0 + (q & 1);
        if (r < M) {
          if (KSPLIT == 1) C[(size_t)r * N + c] = acc[i][j][q];
          else atomicAdd(&C[(size_t)r * N + c], acc[i][j][q]);
        }
      }
    }
}

// ===========================================================================
//  (B) 48 篇主角：W4A8 的整数张量核（IMMA）。
//   · A  int8 [M][K]      -> As[STAGES][BM][ASP]（cp.async）
//   · W  int4 [N][K/2]    -> Wraw[STAGES][BN][WPAD]（cp.async，现场 expand）
//   · BK = 128 = GROUP，一个 stage 恰好一个 scale group，折算不再跨 stage。
//   · mma 片段每 lane 一个 word，直接从 smem 取（无 ldmatrix）；B 从 int4 现场展开。
// ===========================================================================
template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, bool SCTAB = true,
          bool U8 = true, int MINB = 1>
__global__ void __launch_bounds__(WM* WN* 32, MINB) gemm_mma_w4a8_kernel(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs, const float* __restrict__ Axsum,
    const uint8_t* __restrict__ Wp, const float* __restrict__ Ws, float* __restrict__ C, int M,
    int N, int K) {
  constexpr int LARGE_BK = 128;  // = GROUP
  constexpr int NTHREADS = WM * WN * 32;
  constexpr int WARP_M = BM / WM;
  constexpr int WARP_N = BN / WN;
  static_assert(WARP_M == 16, "IMMA 片段固定 m16，BM 必须 = WM*16");
  constexpr int MTN = WARP_N / 8;
  constexpr int ASP = LARGE_BK + 16;       // As 行距（int8），pad 消 bank conflict
  constexpr int WPAD = LARGE_BK / 2 + 16;  // Wraw 行距（int4 字节）
  constexpr int NGRP = 5120 / GROUP;
  constexpr int ASP4 = ASP / 4;
  constexpr int WPAD2 = WPAD / 2;
  constexpr int NC = LARGE_BK / 32;  // 每 stage 的 k32 chunk 数 = 4

  extern __shared__ __align__(16) char smem[];
  int8_t* As = reinterpret_cast<int8_t*>(smem);                       // [STAGES][BM][ASP]
  uint8_t* Wraw = reinterpret_cast<uint8_t*>(As + (size_t)STAGES * BM * ASP);
  uint8_t* Wbase = Wraw + (size_t)STAGES * BN * WPAD;
  float* sas = reinterpret_cast<float*>(Wbase);                        // [BM][NGRP]
  float* sws = sas + (size_t)BM * NGRP;                                // [BN][NGRP]

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

  // 一次性把这片 (m,n) 的 scale 表搬进 smem（SCTAB=false 时改从 global 读，省 23KB smem）。
  if constexpr (SCTAB) {
    for (int i = tid; i < BM * NGRP; i += NTHREADS) {
      const int r = i / NGRP, g = i % NGRP;
      sas[i] = (block_row + r < M) ? Axs[(size_t)(block_row + r) * NGRP + g] : 0.f;
    }
    for (int i = tid; i < BN * NGRP; i += NTHREADS) {
      const int n = i / NGRP, g = i % NGRP;
      sws[i] = Ws[(size_t)(block_col + n) * NGRP + g];
    }
  }

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

  // prologue：填 STAGES-1 个 stage
#pragma unroll
  for (int p = 0; p < STAGES - 1; ++p) {
    if (p < nt) load_stage(p, kb0 + p);
    else __pipeline_commit();
  }

  const int arow0 = lane >> 2;      // A 行（a0/a2）
  const int arow1 = arow0 + 8;      // A 行（a1/a3）
  const int aqw = (lane & 3);       // A/B 的 word 内偏移
  const int bgroup = lane >> 2;     // B 的 n = groupID

#pragma unroll 1
  for (int t = 0; t < nt; ++t) {
    const int st = t % STAGES;
    const int tnext = t + STAGES - 1;
    if (tnext < nt) load_stage(st == 0 ? STAGES - 1 : st - 1, kb0 + tnext);
    else __pipeline_commit();
    __pipeline_wait_prior(STAGES - 1);
    __syncthreads();

    const int g = kb0 + t;  // stage 就是 group
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
        const uint32_t b0 = U8 ? expand4u(W16[wo]) : expand4(W16[wo]);
        const uint32_t b1 = U8 ? expand4u(W16[wo + 4]) : expand4(W16[wo + 4]);
        const uint32_t bb[2] = {b0, b1};
        if constexpr (U8) imma_m16n8k32_u8(acci[j], a, bb);
        else imma_m16n8k32(acci[j], a, bb);
      }
    }

    // 折算：本 group 的 int32 结果乘 sa*sw 后进 fp32 累加器，int 累加器清零。
    // u8 权重时还要扣掉 `8·Σq_a`（每 (m,group) 一个标量）。
    {
      const int gm0 = warp_row * WARP_M + arow0;
      const int gm1 = gm0 + 8;
      const int rm0 = block_row + gm0, rm1 = block_row + gm1;
      float a0, a1;
      if constexpr (SCTAB) {
        a0 = sas[gm0 * NGRP + g];
        a1 = sas[gm1 * NGRP + g];
      } else {
        a0 = (rm0 < M) ? __ldg(&Axs[(size_t)rm0 * NGRP + g]) : 0.f;
        a1 = (rm1 < M) ? __ldg(&Axs[(size_t)rm1 * NGRP + g]) : 0.f;
      }
      float corr0 = 0.f, corr1 = 0.f;
      if constexpr (U8) {
        corr0 = (rm0 < M) ? 8.f * __ldg(&Axsum[(size_t)rm0 * NGRP + g]) : 0.f;
        corr1 = (rm1 < M) ? 8.f * __ldg(&Axsum[(size_t)rm1 * NGRP + g]) : 0.f;
      }
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int ln0 = warp_col * WARP_N + j * 8 + (lane & 3) * 2;
        float w0, w1;
        if constexpr (SCTAB) {
          w0 = sws[ln0 * NGRP + g];
          w1 = sws[(ln0 + 1) * NGRP + g];
        } else {
          w0 = __ldg(&Ws[(size_t)(block_col + ln0) * NGRP + g]);
          w1 = __ldg(&Ws[(size_t)(block_col + ln0 + 1) * NGRP + g]);
        }
        const float i00 = (float)acci[j][0] - corr0;
        const float i01 = (float)acci[j][1] - corr0;
        const float i10 = (float)acci[j][2] - corr1;
        const float i11 = (float)acci[j][3] - corr1;
        facc[j][0] += a0 * w0 * i00;
        facc[j][1] += a0 * w1 * i01;
        facc[j][2] += a1 * w0 * i10;
        facc[j][3] += a1 * w1 * i11;
#pragma unroll
        for (int q = 0; q < 4; ++q) acci[j][q] = 0;
      }
    }
    __syncthreads();
  }

  const int group = lane >> 2, tig = lane & 3;
#pragma unroll
  for (int j = 0; j < MTN; ++j) {
    const int r0 = block_row + warp_row * WARP_M + group;
    const int c0 = block_col + warp_col * WARP_N + j * 8 + tig * 2;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int r = r0 + (q >= 2 ? 8 : 0);
      const int c = c0 + (q & 1);
      if (r < M) {
        if (KSPLIT == 1) C[(size_t)r * N + c] = facc[j][q];
        else atomicAdd(&C[(size_t)r * N + c], facc[j][q]);
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
  uint8_t* Wp;
  float* Ws;
  float* C;
  float* Out;
};

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1>
static void add_mma16(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                      const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a16_kernel<BM, BN, BK, STAGES, WM, WN, KSPLIT>;
  const int NT = WM * WN * 32;
  const size_t shm = (size_t)STAGES * (BM * (BK + 8) * 2 + BN * (BK / 2) + BN * (BK + 8) * 2);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm);
  specs.push_back([=](int M, float*) {
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.A, buf.Wp, buf.Ws, buf.C, M, N, K);
  });
}

template <int BM, int BN, int STAGES, int WM, int WN, int KSPLIT = 1, bool SCTAB = true,
          bool U8 = true, int MINB = 1>
static void add_imma(std::vector<std::function<void(int, float*)>>& specs, const char* nm,
                     const Buf& buf, int N, int K) {
  auto fn = gemm_mma_w4a8_kernel<BM, BN, STAGES, WM, WN, KSPLIT, SCTAB, U8, MINB>;
  const int NT = WM * WN * 32;
  constexpr int ASP = 128 + 16, WPAD = 64 + 16;
  const size_t shm = (size_t)STAGES * (BM * ASP + BN * WPAD) +
                     (SCTAB ? (size_t)(BM + BN) * (5120 / GROUP) * 4 : 0);
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  std::printf("  [%s] threads=%d smem=%zu\n", nm, NT, shm);
  specs.push_back([=](int M, float*) {
    if (KSPLIT > 1) cudaMemsetAsync(buf.C, 0, (size_t)M * N * 4);
    fn<<<dim3(N / BN, div_up(M, BM), KSPLIT), NT, shm>>>(buf.Aq, buf.Axs, buf.Axsum, buf.Wp, buf.Ws,
                                                         buf.C, M, N, K);
  });
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char* which = (argc > 1) ? argv[1] : "all";
  const char* only = (argc > 2) ? argv[2] : "";
  const int N = 17408;
  const int K = 5120;
  const int MAXM = 64;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double w_bytes_int4 = (double)N * K / 2;
  const double s_bytes = (double)N * (K / GROUP) * 4;
  const double total_w = w_bytes_int4 + s_bytes;
  std::printf("\nW4A8 IMMA small-M sweep: N=%d K=%d group=%d ; W_int4=%.1f MB scales=%.1f MB\n", N,
              K, GROUP, w_bytes_int4 / 1e6, s_bytes / 1e6);
  std::printf("峰值: bf16 TC %.0f TFLOPS / int8 TC %.0f TOPS ; bf16 ridge M*=%.0f\n",
              BF16_PEAK, INT8_PEAK, BF16_PEAK * 1e3 / d.mem_bw_gbps / 3.77);

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

  // 激活量化参考（CPU 版，用于对拍 Aq/Axs 与 GEMM）
  std::vector<int8_t> hAq((size_t)MAXM * K);
  std::vector<float> hAxs((size_t)MAXM * (K / GROUP));
  for (int m = 0; m < MAXM; ++m)
    for (int g = 0; g < K / GROUP; ++g) {
      float amax = 0.f;
      for (int k = 0; k < GROUP; ++k)
        amax = std::max(amax, std::fabs(__bfloat162float(hA[(size_t)m * K + g * GROUP + k])));
      const float s = amax > 0 ? amax / 127.f : 1.f;
      hAxs[(size_t)m * (K / GROUP) + g] = s;
      for (int k = 0; k < GROUP; ++k) {
        int v = (int)std::nearbyint(__bfloat162float(hA[(size_t)m * K + g * GROUP + k]) * (1.f / s));
        v = std::max(-127, std::min(127, v));
        hAq[(size_t)m * K + g * GROUP + k] = (int8_t)v;
      }
    }

  Buf buf;
  CUDA_CHECK(cudaMalloc(&buf.A, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&buf.Aq, hAq.size()));
  CUDA_CHECK(cudaMalloc(&buf.Axs, hAxs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.Axsum, hAxs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.Wp, hWp.size()));
  CUDA_CHECK(cudaMalloc(&buf.Ws, hWs.size() * 4));
  CUDA_CHECK(cudaMalloc(&buf.C, (size_t)MAXM * N * 4));
  CUDA_CHECK(cudaMalloc(&buf.Out, (size_t)4096 * 256 * 4));
  CUDA_CHECK(cudaMemcpy(buf.A, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Aq, hAq.data(), hAq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Axs, hAxs.data(), hAxs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Wp, hWp.data(), hWp.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(buf.Ws, hWs.data(), hWs.size() * 4, cudaMemcpyHostToDevice));

  std::vector<float> hC((size_t)MAXM * N);
  int CHK_M = (argc > 3) ? std::atoi(argv[3]) : MAXM;
  // 参考：Σ (Aq*sa)*(q*sw)
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
    std::printf("  [%-16s] max_abs_err=%.3e (ref~%.1f) rel=%.2e %s\n", tag, err, ref,
                err / std::max(ref, 1e-6), ok ? "OK" : "FAIL");
    return ok;
  };

  std::vector<std::function<void(int, float*)>> specs;
  std::vector<std::string> names;

  if (strcmp(which, "all") == 0 || strcmp(which, "bf16") == 0) {
    add_mma16<16, 128, 2, 1, 4, 8>(specs, "mma_b16_k8", buf, N, K);
    names.push_back("mma_b16_k8");
    add_mma16<16, 128, 2, 1, 4, 16>(specs, "mma_b16_k16", buf, N, K);
    names.push_back("mma_b16_k16");
    add_mma16<32, 128, 2, 2, 4, 8>(specs, "mma_b32_k8", buf, N, K);
    names.push_back("mma_b32_k8");
  }
  if (strcmp(which, "all") == 0 || strcmp(which, "imma") == 0) {
    add_imma<16, 128, 2, 1, 4, 8>(specs, "imma_s2k8", buf, N, K);
    names.push_back("imma_s2k8");
    add_imma<16, 128, 3, 1, 4, 8>(specs, "imma_s3k8", buf, N, K);
    names.push_back("imma_s3k8");
    add_imma<16, 128, 4, 1, 4, 8>(specs, "imma_s4k8", buf, N, K);
    names.push_back("imma_s4k8");
    add_imma<16, 128, 3, 1, 4, 16>(specs, "imma_s3k16", buf, N, K);
    names.push_back("imma_s3k16");
    add_imma<16, 128, 3, 1, 4, 4>(specs, "imma_s3k4", buf, N, K);
    names.push_back("imma_s3k4");
    add_imma<32, 128, 3, 2, 4, 8>(specs, "imma_m32_s3k8", buf, N, K);
    names.push_back("imma_m32_s3k8");
    // no-smem-scale-table 变体（省掉 23KB scale 表，本意是抬 occupancy）
    add_imma<16, 128, 2, 1, 4, 8, false, true>(specs, "imma_ns_s2k8", buf, N, K);
    names.push_back("imma_ns_s2k8");
    add_imma<16, 128, 3, 1, 4, 8, false, true>(specs, "imma_ns_s3k8", buf, N, K);
    names.push_back("imma_ns_s3k8");
    add_imma<16, 128, 2, 1, 4, 16, false, true>(specs, "imma_ns_s2k16", buf, N, K);
    names.push_back("imma_ns_s2k16");
    add_imma<16, 128, 2, 1, 4, 4, false, true>(specs, "imma_ns_s2k4", buf, N, K);
    names.push_back("imma_ns_s2k4");
    add_imma<16, 128, 4, 1, 4, 8, false, true>(specs, "imma_ns_s4k8", buf, N, K);
    names.push_back("imma_ns_s4k8");
    // u8.s8 对照（U8=false 需要 vsubss4）
    add_imma<16, 128, 2, 1, 4, 8, false, false>(specs, "imma_s8_s2k8", buf, N, K);
    names.push_back("imma_s8_s2k8");
    add_imma<16, 128, 2, 1, 4, 8, true, false>(specs, "imma_sm_s8_s2k8", buf, N, K);
    names.push_back("imma_sm_s8_s2k8");
    // 强制 occupancy 负结果对照
    add_imma<16, 128, 2, 1, 4, 8, false, true, 7>(specs, "imma_ns_s2k8_b7", buf, N, K);
    names.push_back("imma_ns_s2k8_b7");
    // KSPLIT 细扫（wave 对齐）
    add_imma<16, 128, 2, 1, 4, 5, false, true>(specs, "imma_ns_s2k5", buf, N, K);
    names.push_back("imma_ns_s2k5");
    add_imma<16, 128, 2, 1, 4, 6, false, true>(specs, "imma_ns_s2k6", buf, N, K);
    names.push_back("imma_ns_s2k6");
    add_imma<16, 128, 2, 1, 4, 7, false, true>(specs, "imma_ns_s2k7", buf, N, K);
    names.push_back("imma_ns_s2k7");
    add_imma<32, 128, 2, 2, 4, 5, false, true>(specs, "imma_ns_m32_s2k5", buf, N, K);
    names.push_back("imma_ns_m32_s2k5");
    add_imma<32, 128, 2, 2, 4, 3, false, true>(specs, "imma_ns_m32_s2k3", buf, N, K);
    names.push_back("imma_ns_m32_s2k3");
  }

  std::vector<int> active;
  for (size_t si = 0; si < specs.size(); ++si)
    if (!*only || names[si] == only) active.push_back((int)si);

  // 先检查量化正确性
  {
    const int ngroup = CHK_M * (K / GROUP);
    quant_x_kernel<<<div_up(ngroup, 8), 256>>>(buf.A, buf.Aq, buf.Axs, buf.Axsum, ngroup, K);
    CUDA_CHECK_LAST();
    std::vector<int8_t> hq(ngroup * GROUP);
    std::vector<float> hs(ngroup);
    CUDA_CHECK(cudaMemcpy(hq.data(), buf.Aq, hq.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hs.data(), buf.Axs, hs.size() * 4, cudaMemcpyDeviceToHost));
    long bad = 0;
    for (size_t i = 0; i < hq.size(); ++i)
      if (hq[i] != hAq[i]) ++bad;
    double serr = 0;
    for (int i = 0; i < ngroup; ++i) serr = std::max(serr, (double)std::fabs(hs[i] - hAxs[i]));
    std::printf("\n激活量化对拍：int8 err=%ld/%zu, scale max_err=%.3e\n", bad, hq.size(), serr);
  }

  std::printf("\n=== 精度校验（M=%d, 抽样 48 点）===\n", CHK_M);
  for (int si : active) {
    cudaMemsetAsync(buf.C, 0, (size_t)CHK_M * N * 4);
    specs[si](CHK_M, nullptr);
    CUDA_CHECK_LAST();
    check_imma(names[si].c_str());
  }

  const int Ms[] = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
  const int NM = sizeof(Ms) / sizeof(int);
  std::printf("\n=== M 扫描（同进程、warmup=5 iters=50；括号=权重读遍数）===\n");
  std::printf("%-16s", "kernel\\M");
  for (int mi = 0; mi < NM; ++mi) std::printf("%11d", Ms[mi]);
  std::printf("\n");
  std::vector<std::vector<double>> tab(specs.size(), std::vector<double>(NM, 1e30));
  for (int si : active) {
    std::printf("%-16s", names[si].c_str());
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
      std::printf("%9.4f  ", ms);
    }
    std::printf("\n");
  }
  if (*only) return 0;

  std::printf("\n=== 每个 M 的最优 kernel（按 ms）===\n");
  std::printf("%-6s %-16s %9s %9s %9s %9s\n", "M", "best", "ms", "TFLOPS", "Wpass", "realGB/s");
  double sum_bf16 = 0, sum_imma = 0;
  for (int mi = 0; mi < NM; ++mi) {
    int bi = 0;
    for (size_t si = 1; si < specs.size(); ++si)
      if (tab[si][mi] < tab[bi][mi]) bi = si;
    const int M = Ms[mi];
    const int wpass = div_up(M, 16);
    const double real_bytes = (double)wpass * total_w + (double)M * K + (double)M * N * 4;
    const double ms = tab[bi][mi];
    std::printf("%-6d %-16s %9.4f %9.2f %9d %9.1f\n", M, names[bi].c_str(), ms,
                to_tflops(2.0 * M * N * K, ms), wpass, real_bytes * 1000.0 / ms / 1e9);
    double b16 = 1e30, bim = 1e30;
    for (size_t si = 0; si < specs.size(); ++si) {
      const bool is_imma = names[si].rfind("imma", 0) == 0;
      if (is_imma) bim = std::min(bim, tab[si][mi]);
      else b16 = std::min(b16, tab[si][mi]);
    }
    sum_bf16 += b16;
    sum_imma += bim;
  }
  std::printf("\n汇总（全部 %d 个 M 点各取各自最优之和）：\n", NM);
  std::printf("  always bf16-mma : %.4f ms\n", sum_bf16);
  std::printf("  always IMMA     : %.4f ms  (relative %.3f)\n", sum_imma, sum_imma / sum_bf16);

  CUDA_CHECK(cudaFree(buf.A));
  CUDA_CHECK(cudaFree(buf.Aq));
  CUDA_CHECK(cudaFree(buf.Axs));
  CUDA_CHECK(cudaFree(buf.Axsum));
  CUDA_CHECK(cudaFree(buf.Wp));
  CUDA_CHECK(cudaFree(buf.Ws));
  CUDA_CHECK(cudaFree(buf.C));
  CUDA_CHECK(cudaFree(buf.Out));
  return 0;
}
