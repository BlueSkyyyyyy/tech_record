// 62（主题 26q）：V4-Pro FP4 decode FFN 续 —— K1 的整 chunk 位拼装解码 / K2 的并行粒度
//
// 承接 61（主题 26p）：61 的 ncu 判词是
//   K1  Compute 75.9% ≈ DRAM 71.4%（16 项 nibble 表 LUT + dp4a 发射受限，ALU pipe 75.9%）
//   K2  occ 23.5% / long_scoreboard ≈35%（延迟受限）
//
// 本篇两条可执行路径：
//   ① K1 解码：每 16B 权重 chunk 现在要 32 次 LDS（16 项表，1 次/nibble）。
//      换成 (a) uint16[256] 字节表（1 次 LDS / byte，2 个 int8）、
//           (b) 纯 ALU 的 PRMT 位拼装（把 16 项有符号表用两次 prmt + 逐字节符号掩码拼出 4 个 int8）。
//   ② K2 并行粒度：K=I=3072 每行只有 3 个 16B chunk / lane，prefetch runway 太短。
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384, topk=6, expert_dtype=fp4
// 语义取自 inference/model.py:596（gate=w1(x), up=w3(x), clamp 10, silu(gate)*up, w2）。
//
// 运行：scripts/run.sh 62-v4-fp4-ffn2/fp4_ffn2.cu [B] [pipe] [dec] [mode]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int H = 7168;     // hidden（K）
constexpr int I = 3072;     // moe_intermediate（N）
constexpr int GROUP = 128;  // 激活量化组
constexpr int E_EXP = 384;  // routed experts
constexpr int TOPK = 6;

// e2m1 -> int8（值 = 2 * 真实值；dp4a 后整体乘 0.5 还原）
__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}
__host__ __device__ __forceinline__ float e2m1f(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  float v = (e == 0) ? (m * 0.5f) : ((1.f + m * 0.5f) * exp2f((int)e - 1));
  return s ? -v : v;
}
__device__ __forceinline__ uint16_t lut_entry(unsigned i) {
  return (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}
__host__ __device__ __forceinline__ float silu_f(float x) { return x / (1.f + expf(-x)); }

// DEC=0: uint16[256] 字节表    DEC=4: uint16[16] nibble 表    DEC=8: PRMT 位拼装
template <int DECODE> __host__ __device__ constexpr int L16N() {
  return DECODE == 0 ? 256 : (DECODE == 8 ? 1 : 16);
}

template <int DECODE>
__device__ __forceinline__ void init_lut(uint16_t* l16, int tid) {
  if (DECODE == 0) {
    for (int i = tid; i < 256; i += 256) l16[i] = lut_entry((unsigned)i);
  } else if (DECODE == 4) {
    for (int i = tid; i < 16; i += 256) l16[i] = (uint16_t)(uint8_t)e2m1_i8((unsigned)i);
  }
}

// PRMT 版：把一个 32-bit 权重字（8 个 nibble，k 顺序）译成 8 个有符号 int8。
//   prmt 的 selector 只用 field 的低 3 位（值 8-15 回绕到 0-7，bit3 被忽略），
//   所以把 orig nibble 直接当 selector 即可拿到幅值；再用 nibble 的 bit3 造逐字节
//   符号掩码在两张小表（+幅值 / -幅值）间选择。
__device__ __forceinline__ void decode_word_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  constexpr uint32_t T0 = 0x03020100u;  // 幅值 idx 0..3 -> 0,1,2,3
  constexpr uint32_t T1 = 0x0C080604u;  // 幅值 idx 4..7 -> 4,6,8,12
  constexpr uint32_t T2 = 0xFDFEFF00u;  // -幅值 idx 0..3 -> 0,-1,-2,-3
  constexpr uint32_t T3 = 0xF4F8FAFCu;  // -幅值 idx 4..7 -> -4,-6,-8,-12
  const uint32_t hi = w >> 16;  // fields0-3 = nibble 4..7
  const uint32_t mlo = __byte_perm(T0, T1, w);
  const uint32_t mhi = __byte_perm(T0, T1, hi);
  const uint32_t slo = __byte_perm(T2, T3, w);
  const uint32_t shi = __byte_perm(T2, T3, hi);
  // 逐字节符号掩码：nibble i 的符号位在 w 的 bit(4i+3)，移到 field 低位后 *5 -> 0/5，
  // 用 prmt(0, ~0) 展成 0x00 / 0xFF。
  const uint32_t klo = __byte_perm(0u, ~0u, ((w >> 3) & 0x11111111u) * 5u);
  const uint32_t khi = __byte_perm(0u, ~0u, ((w >> 19) & 0x11111111u) * 5u);
  a = mlo ^ ((mlo ^ slo) & klo);
  b = mhi ^ ((mhi ^ shi) & khi);
}

// 把一个 32-bit 权重字（8 个 nibble）译成 8 个 int8（a,b 各 4 个，k 顺序）。
template <int DECODE>
__device__ __forceinline__ void decode_word(uint32_t w, uint32_t& a, uint32_t& b,
                                            const uint16_t* __restrict__ l16) {
  if (DECODE == 0) {
    const uint32_t r0 = l16[w & 0xFFu], r1 = l16[(w >> 8) & 0xFFu];
    const uint32_t r2 = l16[(w >> 16) & 0xFFu], r3 = l16[(w >> 24) & 0xFFu];
    a = r0 | (r1 << 16); b = r2 | (r3 << 16);
  } else if (DECODE == 4) {
    const uint32_t t0 = (uint32_t)(uint8_t)l16[w & 0xFu] | ((uint32_t)(uint8_t)l16[(w >> 4) & 0xFu] << 8);
    const uint32_t t1 = (uint32_t)(uint8_t)l16[(w >> 8) & 0xFu] | ((uint32_t)(uint8_t)l16[(w >> 12) & 0xFu] << 8);
    a = t0 | (t1 << 16);
    const uint32_t t2 = (uint32_t)(uint8_t)l16[(w >> 16) & 0xFu] | ((uint32_t)(uint8_t)l16[(w >> 20) & 0xFu] << 8);
    const uint32_t t3 = (uint32_t)(uint8_t)l16[(w >> 24) & 0xFu] | ((uint32_t)(uint8_t)l16[(w >> 28) & 0xFu] << 8);
    b = t2 | (t3 << 16);
  } else {
    decode_word_prmt(w, a, b);
  }
}

// ---------------------------------------------------------------------------
// 通用 grouped GEMV：一个 kernel 覆盖全部 active expert，权重每专家只读一遍。
// ---------------------------------------------------------------------------
template <int KK, int RWW, int MTMAX, int DEPTH, int DECODE, int MINCTA, int PIPE, int GATHER>
__global__ void __launch_bounds__(256, MINCTA) gemv_fp4(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs,
    const uint8_t* __restrict__ Wall, const uint8_t* __restrict__ Wsall,
    const int* __restrict__ counts, const int* __restrict__ toff,
    const int* __restrict__ toks, const int* __restrict__ wexp,
    float* __restrict__ Y, int N) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int NS32 = KK / 32;
  constexpr int CH = KK / 32;
  constexpr int NITER = KK / 1024;

  const int gid = blockIdx.y;
  const int m = counts[gid];
  if (m == 0 || m > MTMAX) return;
  const int e = wexp[gid];
  const int off = toff[gid];

  extern __shared__ __align__(16) char smem[];
  int8_t* xs0 = reinterpret_cast<int8_t*>(smem);
  int8_t* xs1 = reinterpret_cast<int8_t*>(smem + (size_t)MTMAX * CH * 16);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MTMAX * KK);
  char* wsz = reinterpret_cast<char*>(smem + (size_t)MTMAX * KK + (size_t)MTMAX * NGRP * 4);
  __shared__ uint16_t l16[L16N<DECODE>()];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  init_lut<DECODE>(l16, tid);

  for (int i = tid; i < m * CH; i += 256) {
    const int mm = i / CH, c = i % CH;
    const int ar = GATHER ? toks[off + mm] : (off + mm);
    const uint4* src = reinterpret_cast<const uint4*>(Aq + (size_t)ar * KK);
    *reinterpret_cast<uint4*>(xs0 + ((size_t)mm * CH + c) * 16) = src[2 * c];
    *reinterpret_cast<uint4*>(xs1 + ((size_t)mm * CH + c) * 16) = src[2 * c + 1];
  }
  for (int i = tid; i < m * NGRP; i += 256) {
    const int mm = i / NGRP, g = i % NGRP;
    const int ar = GATHER ? toks[off + mm] : (off + mm);
    sxs[i] = Axs[(size_t)ar * NGRP + g];
  }
  __syncthreads();

  const int row_base = blockIdx.x * BN + warp * RWW;
  const uint8_t* We = Wall + (size_t)e * N * (KK / 2);
  const uint8_t* Wse = Wsall + (size_t)e * N * NS32;

  float acc[RWW][MTMAX];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) acc[r][mm] = 0.f;

  constexpr int WROW = 512;  // 32 lane * 16B

  if (PIPE == 0) {
    uint4 ring[RWW][DEPTH];
#pragma unroll
    for (int r = 0; r < RWW; ++r)
#pragma unroll
      for (int d = 0; d < DEPTH; ++d)
        ring[r][d] = __ldcs(reinterpret_cast<const uint4*>(
            We + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * (d < NITER ? d : NITER - 1)) * 16));
#pragma unroll
    for (int i = 0; i < NITER; ++i) {
      const int c = lane + 32 * i;
      const int g128 = (c * 32) / GROUP;
      uint32_t x8[MTMAX][8];
#pragma unroll
      for (int mm = 0; mm < MTMAX; ++mm) {
        if (mm < m) {
          const uint4 xa = *reinterpret_cast<const uint4*>(xs0 + ((size_t)mm * CH + c) * 16);
          const uint4 xb = *reinterpret_cast<const uint4*>(xs1 + ((size_t)mm * CH + c) * 16);
          x8[mm][0] = xa.x; x8[mm][1] = xa.y; x8[mm][2] = xa.z; x8[mm][3] = xa.w;
          x8[mm][4] = xb.x; x8[mm][5] = xb.y; x8[mm][6] = xb.z; x8[mm][7] = xb.w;
        }
      }
#pragma unroll
      for (int r = 0; r < RWW; ++r) {
        const uint4 w = ring[r][i % DEPTH];
        if (i + DEPTH < NITER)
          ring[r][i % DEPTH] = __ldcs(reinterpret_cast<const uint4*>(
              We + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * (i + DEPTH)) * 16));
        const float sw = __int_as_float((int)__ldg(&Wse[(size_t)(row_base + r) * NS32 + c]) << 23);
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        uint32_t w8[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) decode_word<DECODE>(wv[j], w8[2 * j], w8[2 * j + 1], l16);
#pragma unroll
        for (int mm = 0; mm < MTMAX; ++mm) {
          if (mm < m) {
            int dot = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[mm][q], dot);
            acc[r][mm] += 0.5f * sw * sxs[mm * NGRP + g128] * (float)dot;
          }
        }
      }
    }
  } else {
    char* wp = wsz + (size_t)warp * DEPTH * RWW * WROW;
    auto issue = [&](int s) {
#pragma unroll
      for (int r = 0; r < RWW; ++r) {
        const void* src = We + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * s) * 16;
        void* dst = wp + ((size_t)(s % DEPTH) * RWW + r) * WROW + lane * 16;
        __pipeline_memcpy_async(dst, src, 16);
      }
      __pipeline_commit();
    };
#pragma unroll
    for (int s = 0; s < DEPTH; ++s) issue(s);
#pragma unroll
    for (int i = 0; i < NITER; ++i) {
      __pipeline_wait_prior(DEPTH - 1);
      const int c = lane + 32 * i;
      const int g128 = (c * 32) / GROUP;
      uint32_t x8[MTMAX][8];
#pragma unroll
      for (int mm = 0; mm < MTMAX; ++mm) {
        if (mm < m) {
          const uint4 xa = *reinterpret_cast<const uint4*>(xs0 + ((size_t)mm * CH + c) * 16);
          const uint4 xb = *reinterpret_cast<const uint4*>(xs1 + ((size_t)mm * CH + c) * 16);
          x8[mm][0] = xa.x; x8[mm][1] = xa.y; x8[mm][2] = xa.z; x8[mm][3] = xa.w;
          x8[mm][4] = xb.x; x8[mm][5] = xb.y; x8[mm][6] = xb.z; x8[mm][7] = xb.w;
        }
      }
#pragma unroll
      for (int r = 0; r < RWW; ++r) {
        const uint4 w = *reinterpret_cast<const uint4*>(
            wp + ((size_t)(i % DEPTH) * RWW + r) * WROW + lane * 16);
        const float sw = __int_as_float((int)__ldg(&Wse[(size_t)(row_base + r) * NS32 + c]) << 23);
        const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
        uint32_t w8[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) decode_word<DECODE>(wv[j], w8[2 * j], w8[2 * j + 1], l16);
#pragma unroll
        for (int mm = 0; mm < MTMAX; ++mm) {
          if (mm < m) {
            int dot = 0;
#pragma unroll
            for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[mm][q], dot);
            acc[r][mm] += 0.5f * sw * sxs[mm * NGRP + g128] * (float)dot;
          }
        }
      }
      if (i + DEPTH < NITER) issue(i + DEPTH);
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) {
      if (mm < m) {
        const float v = warp_sum(acc[r][mm]);
        if (lane == 0) Y[(size_t)(off + mm) * N + row_base + r] = v;
      }
    }
}

// ---------------------------------------------------------------------------
// TMA 版 grouped GEMV（63 新增）：一个 warp = 一个专家的一组连续 RWW 行，
// 整 row-tile（RWW*KK/2 字节，天然连续）+ scale 段用两条 `cp.async.bulk` 搬进 smem，
// 每 warp 私有 mbarrier。对比 cp.async 版：每 warp 从 2*RWW 条 cp.async 降到 2 条 bulk。
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr), "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void tma_1d(uint32_t dst, const void* src, uint32_t bytes,
                                       uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ uint64_t l2_evict_first_policy() {
  uint64_t p;
  asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma_1d_hint(uint32_t dst, const void* src, uint32_t bytes,
                                            uint32_t mbar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;" ::"r"(dst),
      "l"(src), "r"(bytes), "r"(mbar), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, int phase) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tWAIT_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(phase));
}

template <int KK, int RWW, int NWARP, int MTMAX, int MINCTA, int DECODE, int GATHER>
__global__ void __launch_bounds__(NWARP * 32, MINCTA) gemv_fp4_tma(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs,
    const uint8_t* __restrict__ Wall, const uint8_t* __restrict__ Wsall,
    const int* __restrict__ counts, const int* __restrict__ toff,
    const int* __restrict__ toks, const int* __restrict__ wexp,
    float* __restrict__ Y, int N, int hint, int early) {
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int CH = KK / 32;
  constexpr int NS32 = KK / 32;
  constexpr int NITER = KK / 1024;
  constexpr int WBYTES = RWW * (KK / 2);
  constexpr int SBYTES = RWW * NS32;
  constexpr int T = NWARP * 32;

  const int gid = blockIdx.y;
  const int m = counts[gid];
  if (m == 0 || m > MTMAX) return;
  const int e = wexp[gid];
  const int off = toff[gid];

  extern __shared__ __align__(128) char smem[];
  int8_t* xs0 = reinterpret_cast<int8_t*>(smem);
  int8_t* xs1 = xs0 + (size_t)MTMAX * CH * 16;
  float* sxs = reinterpret_cast<float*>(xs1 + (size_t)MTMAX * CH * 16);
  uint8_t* wsm = reinterpret_cast<uint8_t*>(sxs + (size_t)MTMAX * NGRP);
  uint8_t* ssm = wsm + (size_t)NWARP * WBYTES;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(ssm + (size_t)NWARP * SBYTES);
  __shared__ uint16_t l16[1];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  (void)l16;

  // lane0 先 init 再 issue（同线程程序序）；其余 lane 在下面的 __syncthreads 之后才 wait，
  // 该 barrier 同时保证 init 对所有线程可见，无需额外同步。
  if (lane == 0) mbar_init(smem_u32(&mbar[warp]), 1);

  const int row_base = blockIdx.x * BN + warp * RWW;
  const uint8_t* We = Wall + (size_t)e * N * (KK / 2);
  const uint8_t* Wse = Wsall + (size_t)e * N * NS32;

  const uint64_t pol = hint ? l2_evict_first_policy() : 0ull;
  auto issue_tma = [&] {
    if (lane != 0) return;
    mbar_expect_tx(smem_u32(&mbar[warp]), (uint32_t)(WBYTES + SBYTES));
    if (hint) {
      tma_1d_hint(smem_u32(wsm + (size_t)warp * WBYTES), We + (size_t)row_base * (KK / 2), WBYTES,
                  smem_u32(&mbar[warp]), pol);
      tma_1d_hint(smem_u32(ssm + (size_t)warp * SBYTES), Wse + (size_t)row_base * NS32, SBYTES,
                  smem_u32(&mbar[warp]), pol);
    } else {
      tma_1d(smem_u32(wsm + (size_t)warp * WBYTES), We + (size_t)row_base * (KK / 2), WBYTES,
             smem_u32(&mbar[warp]));
      tma_1d(smem_u32(ssm + (size_t)warp * SBYTES), Wse + (size_t)row_base * NS32, SBYTES,
             smem_u32(&mbar[warp]));
    }
  };

  // early=1：先发 TMA 再搬激活，让 TMA 延迟被激活的全局读遮住（K1 长 tile 有效）。
  if (early) issue_tma();

  for (int i = tid; i < m * CH; i += T) {
    const int mm = i / CH, c = i % CH;
    const int ar = GATHER ? toks[off + mm] : (off + mm);
    const uint4* src = reinterpret_cast<const uint4*>(Aq + (size_t)ar * KK);
    *reinterpret_cast<uint4*>(xs0 + ((size_t)mm * CH + c) * 16) = src[2 * c];
    *reinterpret_cast<uint4*>(xs1 + ((size_t)mm * CH + c) * 16) = src[2 * c + 1];
  }
  for (int i = tid; i < m * NGRP; i += T) {
    const int mm = i / NGRP, g = i % NGRP;
    sxs[i] = Axs[(size_t)(GATHER ? toks[off + mm] : (off + mm)) * NGRP + g];
  }
  __syncthreads();
  if (!early) issue_tma();
  mbar_wait(smem_u32(&mbar[warp]), 0);

  float acc[RWW][MTMAX];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) acc[r][mm] = 0.f;

  const uint8_t* wbuf = wsm + (size_t)warp * WBYTES;
  const uint8_t* sbuf = ssm + (size_t)warp * SBYTES;
#pragma unroll
  for (int i = 0; i < NITER; ++i) {
    const int c = lane + 32 * i;
    const int g128 = (c * 32) / GROUP;
    uint32_t x8[MTMAX][8];
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) {
      if (mm < m) {
        const uint4 xa = *reinterpret_cast<const uint4*>(xs0 + ((size_t)mm * CH + c) * 16);
        const uint4 xb = *reinterpret_cast<const uint4*>(xs1 + ((size_t)mm * CH + c) * 16);
        x8[mm][0] = xa.x; x8[mm][1] = xa.y; x8[mm][2] = xa.z; x8[mm][3] = xa.w;
        x8[mm][4] = xb.x; x8[mm][5] = xb.y; x8[mm][6] = xb.z; x8[mm][7] = xb.w;
      }
    }
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const uint4 w = *reinterpret_cast<const uint4*>(wbuf + (size_t)r * (KK / 2) + c * 16);
      const float sw = __int_as_float((int)sbuf[(size_t)r * NS32 + c] << 23);
      const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
      uint32_t w8[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) decode_word<DECODE>(wv[j], w8[2 * j], w8[2 * j + 1], l16);
#pragma unroll
      for (int mm = 0; mm < MTMAX; ++mm) {
        if (mm < m) {
          int dot = 0;
#pragma unroll
          for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[mm][q], dot);
          acc[r][mm] += 0.5f * sw * sxs[mm * NGRP + g128] * (float)dot;
        }
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) {
      if (mm < m) {
        const float v = warp_sum(acc[r][mm]);
        if (lane == 0) Y[(size_t)(off + mm) * N + row_base + r] = v;
      }
    }
}

// ---------------------------------------------------------------------------
// SwiGLU + per-128 int8 量化（同 61）
// ---------------------------------------------------------------------------
__global__ void swiglu_quant_i8(const float* __restrict__ GU, int8_t* __restrict__ Hq,
                                float* __restrict__ Hs, int Pp, int NN) {
  const int lane = threadIdx.x & 31;
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int ngrp = NN / GROUP;
  if (wid >= Pp * ngrp) return;
  const int p = wid / ngrp, gg = wid % ngrp;
  const float* g = GU + (size_t)p * 2 * NN + gg * GROUP;
  const float* u = GU + (size_t)p * 2 * NN + NN + gg * GROUP;
  float vals[GROUP / 32];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < GROUP / 32; ++t) {
    const float gv = fminf(g[lane + 32 * t], 10.0f);
    const float uv = fminf(fmaxf(u[lane + 32 * t], -10.0f), 10.0f);
    const float hv = silu_f(gv) * uv;
    vals[t] = hv;
    amax = fmaxf(amax, fabsf(hv));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(~0u, amax, o));
  const float s = amax > 0 ? amax / 127.f : 1.f;
  if (lane == 0) Hs[(size_t)p * ngrp + gg] = s;
  const float inv = 1.f / s;
  int8_t* q = Hq + (size_t)p * NN + gg * GROUP;
#pragma unroll
  for (int t = 0; t < GROUP / 32; ++t) {
    int v = __float2int_rn(vals[t] * inv);
    v = max(-127, min(127, v));
    q[lane + 32 * t] = (int8_t)v;
  }
}

__global__ void unpermute_weighted(const float* __restrict__ O, const float* __restrict__ pw,
                                   const int* __restrict__ toks, float* __restrict__ Y,
                                   int Pp, int NN) {
  const size_t tot = (size_t)Pp * NN;
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < tot;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int p = (int)(idx / NN), n = (int)(idx % NN);
    atomicAdd(&Y[(size_t)toks[p] * NN + n], pw[p] * O[idx]);
  }
}

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

__global__ void replicate_packed(const uint8_t* __restrict__ base, uint8_t* __restrict__ out,
                                 size_t sz, int E) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)E * sz;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = base[i % sz] ^ (uint8_t)((i / sz) * 37);
}
__global__ void replicate_scale(const uint8_t* __restrict__ base, uint8_t* __restrict__ out,
                                size_t sz, int E) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)E * sz;
       i += (size_t)gridDim.x * blockDim.x)
    out[i] = base[i % sz];
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

static std::vector<uint8_t> load_bin(const char* path, size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
  std::vector<uint8_t> v(expect);
  size_t got = std::fread(v.data(), 1, expect, f);
  std::fclose(f);
  if (got != expect) { std::fprintf(stderr, "read %s got %zu/%zu\n", path, got, expect); std::exit(1); }
  return v;
}

#define SMEM_G(KK, RWW, MTMAX, DEPTH, PIPE) \
  ((size_t)MTMAX * (KK / 32) * 16 * 2 + (size_t)MTMAX * (KK / GROUP) * 4 + \
   ((PIPE) ? (size_t)8 * DEPTH * RWW * 512 : 0))

template <int KK, int RWW, int MTMAX, int DEPTH, int DECODE, int MINCTA, int PIPE, int GATHER>
static void launch_gemv(dim3 grid, int smem, const int8_t* Aq, const float* Axs,
                        const uint8_t* W, const uint8_t* Ws, const int* counts, const int* toff,
                        const int* toks, const int* wexp, float* Y, int N) {
  auto k = gemv_fp4<KK, RWW, MTMAX, DEPTH, DECODE, MINCTA, PIPE, GATHER>;
  cudaError_t e = cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) { std::fprintf(stderr, "setattr smem=%d failed: %s\n", smem, cudaGetErrorString(e)); std::exit(1); }
  k<<<grid, 256, smem>>>(Aq, Axs, W, Ws, counts, toff, toks, wexp, Y, N);
}

struct Route { std::vector<int> counts, toff, toks, wexp; int n_active; };

#define SMEM_TMA(KK_, RWW, NWARP, MTMAX) \
  ((size_t)(MTMAX) * ((KK_) / 32) * 16 * 2 + (size_t)(MTMAX) * ((KK_) / GROUP) * 4 + \
   (size_t)(NWARP) * (RWW) * ((KK_) / 2) + (size_t)(NWARP) * (RWW) * ((KK_) / 32) + (size_t)(NWARP) * 8)

template <int KK, int RWW, int NWARP, int MTMAX, int DECODE, int GATHER>
static void launch_tma(dim3 grid, int smem, const int8_t* Aq, const float* Axs,
                       const uint8_t* W, const uint8_t* Ws, const int* counts, const int* toff,
                       const int* toks, const int* wexp, float* Y, int N, int hint = 0, int early = 1) {
  auto k = gemv_fp4_tma<KK, RWW, NWARP, MTMAX, 2, DECODE, GATHER>;
  cudaError_t e = cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) { std::fprintf(stderr, "TMA setattr smem=%d failed: %s\n", smem, cudaGetErrorString(e)); std::exit(1); }
  k<<<grid, NWARP * 32, smem>>>(Aq, Axs, W, Ws, counts, toff, toks, wexp, Y, N, hint, early);
}

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int B = (argc > 1) ? std::atoi(argv[1]) : 64;
  const int PIPE = (argc > 2) ? std::atoi(argv[2]) : 1;
  const int DEC = (argc > 3) ? std::atoi(argv[3]) : 0;
  const char* mode = (argc > 4) ? argv[4] : "none";
  const int KTMA = (argc > 5) ? std::atoi(argv[5]) : 1;  // 0=全 cp.async 1=K1/K2 都 TMA 2=仅 K2 TMA
  const int HINT = (argc > 6) ? std::atoi(argv[6]) : 0;  // 1=TMA L2 evict_first
  const int EK = (argc > 7) ? std::atoi(argv[7]) : 3;    // bit0=K1 early TMA, bit1=K2 early TMA
  // sweep 只在 B<=64（m=1）下做，用固定 MTMAX=1 与主路径一致
  constexpr int MTMAX = 1;
  const int pairs = B * TOPK;
  const int N1 = 2 * I, K1 = H;
  const int N2 = H, K2 = I;

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nV4-Pro routed expert FFN FP4 decode: w1/w3 [%d,%d] + w2 [%d,%d], "
              "B=%d top-%d E=%d pairs=%d  PIPE=%d DEC=%d KTMA=%d mode=%s\n",
              I, H, H, I, B, TOPK, E_EXP, pairs, PIPE, DEC, KTMA, mode);

  const size_t w1sz = (size_t)I * (H / 2), w1ssz = (size_t)I * (H / 32);
  const size_t w2sz = (size_t)H * (I / 2), w2ssz = (size_t)H * (I / 32);
  auto w1p = load_bin("../56-v4-fp4-moe/w1_fp4.bin", w1sz);
  auto w1s = load_bin("../56-v4-fp4-moe/w1_scale.bin", w1ssz);
  auto w3p = load_bin("../56-v4-fp4-moe/w3_fp4.bin", w1sz);
  auto w3s = load_bin("../56-v4-fp4-moe/w3_scale.bin", w1ssz);
  auto w2p = load_bin("../56-v4-fp4-moe/w2_fp4.bin", w2sz);
  auto w2s = load_bin("../56-v4-fp4-moe/w2_scale.bin", w2ssz);

  std::vector<uint8_t> b13p(N1 * (H / 2)), b13s(N1 * (H / 32));
  std::memcpy(b13p.data(), w1p.data(), w1sz);
  std::memcpy(b13p.data() + w1sz, w3p.data(), w1sz);
  std::memcpy(b13s.data(), w1s.data(), w1ssz);
  std::memcpy(b13s.data() + w1ssz, w3s.data(), w1ssz);

  uint8_t *dB13, *dBs13, *dB2, *dBs2, *b13d, *b13sd, *b2d, *b2sd;
  CUDA_CHECK(cudaMalloc(&dB13, (size_t)E_EXP * b13p.size()));
  CUDA_CHECK(cudaMalloc(&dBs13, (size_t)E_EXP * b13s.size()));
  CUDA_CHECK(cudaMalloc(&dB2, (size_t)E_EXP * w2p.size()));
  CUDA_CHECK(cudaMalloc(&dBs2, (size_t)E_EXP * w2s.size()));
  CUDA_CHECK(cudaMalloc(&b13d, b13p.size())); CUDA_CHECK(cudaMalloc(&b13sd, b13s.size()));
  CUDA_CHECK(cudaMalloc(&b2d, w2p.size())); CUDA_CHECK(cudaMalloc(&b2sd, w2s.size()));
  CUDA_CHECK(cudaMemcpy(b13d, b13p.data(), b13p.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b13sd, b13s.data(), b13s.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b2d, w2p.data(), w2p.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b2sd, w2s.data(), w2s.size(), cudaMemcpyHostToDevice));
  replicate_packed<<<2048, 256>>>(b13d, dB13, b13p.size(), E_EXP);
  replicate_scale<<<2048, 256>>>(b13sd, dBs13, b13s.size(), E_EXP);
  replicate_packed<<<2048, 256>>>(b2d, dB2, w2p.size(), E_EXP);
  replicate_scale<<<2048, 256>>>(b2sd, dBs2, w2s.size(), E_EXP);
  CUDA_CHECK_LAST();

  Route R;
  R.counts.assign(E_EXP, 0);
  std::vector<int> assign(pairs);
  for (int t = 0; t < B; ++t)
    for (int j = 0; j < TOPK; ++j) { const int e = (t * TOPK + j) % E_EXP; assign[t * TOPK + j] = e; R.counts[e]++; }
  R.toff.assign(E_EXP + 1, 0);
  for (int e = 0; e < E_EXP; ++e) R.toff[e + 1] = R.toff[e] + R.counts[e];
  R.toks.assign(pairs, 0);
  std::vector<int> cur = R.toff;
  for (int p = 0; p < pairs; ++p) R.toks[cur[assign[p]]++] = p / TOPK;
  for (int e = 0; e < E_EXP; ++e) if (R.counts[e] > 0) R.wexp.push_back(e);
  R.n_active = (int)R.wexp.size();

  std::vector<float> hpw(pairs);
  for (int p = 0; p < pairs; ++p) hpw[p] = 0.5f + 0.1f * ((p / TOPK) % 5);
  std::printf("routing balanced active=%d/%d  max_m=%d\n", R.n_active, E_EXP,
              *std::max_element(R.counts.begin(), R.counts.end()));

  std::vector<bf16> hx((size_t)B * H);
  unsigned st = 12345u;
  for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = __float2bfloat16(0.5f * ((st >> 8) / 8388608.0f - 1.f)); }

  int8_t *dXq, *dHq; float *dXs, *dHs, *dGU, *dO, *dY; bf16* dX;
  CUDA_CHECK(cudaMalloc(&dX, hx.size() * 2));
  CUDA_CHECK(cudaMalloc(&dXq, (size_t)B * H));
  CUDA_CHECK(cudaMalloc(&dXs, (size_t)B * (H / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dGU, (size_t)pairs * N1 * 4));
  CUDA_CHECK(cudaMalloc(&dHq, (size_t)pairs * I));
  CUDA_CHECK(cudaMalloc(&dHs, (size_t)pairs * (I / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dO, (size_t)pairs * N2 * 4));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)B * H * 4));
  CUDA_CHECK(cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  int *dCounts, *dToff, *dToks, *dWexp; float* dPw;
  CUDA_CHECK(cudaMalloc(&dCounts, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToff, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToks, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dWexp, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dPw, pairs * 4));
  CUDA_CHECK(cudaMemcpy(dCounts, R.counts.data(), R.counts.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToff, R.toff.data(), E_EXP * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToks, R.toks.data(), R.toks.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWexp, R.wexp.data(), R.wexp.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dPw, hpw.data(), hpw.size() * 4, cudaMemcpyHostToDevice));

  const int MT = std::max(1, std::min(4, B / 64));

  // 三个 decode 版本的逐 M 分派实例
#define K1_LAUNCH(D_, M_) \
  (PIPE ? launch_gemv<H, 4, M_, 2, D_, 2, 1, 1>(dim3(N1 / 32, R.n_active), SMEM_G(H, 4, M_, 2, 1), dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1) \
        : launch_gemv<H, 4, M_, 2, D_, 2, 0, 1>(dim3(N1 / 32, R.n_active), SMEM_G(H, 4, M_, 2, 0), dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1))
#define K2_LAUNCH(D_, M_) \
  (PIPE ? launch_gemv<I, 8, M_, 2, D_, 2, 1, 0>(dim3(N2 / 64, R.n_active), SMEM_G(I, 8, M_, 2, 1), dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2) \
        : launch_gemv<I, 4, M_, 2, D_, 2, 0, 0>(dim3(N2 / 32, R.n_active), SMEM_G(I, 4, M_, 2, 0), dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2))

  auto run_quant = [&] {
    const int ngroup = B * (H / GROUP);
    quant_x_kernel<<<(ngroup + 7) / 8, 256>>>(dX, dXq, dXs, ngroup, H);
  };
  auto run_swi = [&] { swiglu_quant_i8<<<(pairs * (I / GROUP) + 7) / 8, 256>>>(dGU, dHq, dHs, pairs, I); };
  auto run_unp = [&] { unpermute_weighted<<<2048, 256>>>(dO, dPw, dToks, dY, pairs, N2); };

#define DISPATCH_K1(D_) do { \
    if (MT == 1) { K1_LAUNCH(D_, 1); } else if (MT == 2) { K1_LAUNCH(D_, 2); } else { K1_LAUNCH(D_, 4); } } while (0)
#define DISPATCH_K2(D_) do { \
    if (MT == 1) { K2_LAUNCH(D_, 1); } else if (MT == 2) { K2_LAUNCH(D_, 2); } else { K2_LAUNCH(D_, 4); } } while (0)

  // TMA row-tile 版（MT=1 专用；MT>1 回落到 cp.async）
#define K1_TMA(D_, M_) launch_tma<H, 2, 8, M_, D_, 1>(dim3(N1 / 16, R.n_active), SMEM_TMA(H, 2, 8, M_), dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1, HINT, (EK & 1))
#define K2_TMA(D_, M_) do { constexpr int RW = (M_) == 1 ? 8 : 4; constexpr int NW = (M_) == 1 ? 8 : 4; launch_tma<I, RW, NW, M_, D_, 0>(dim3(N2 / (NW * RW), R.n_active), SMEM_TMA(I, RW, NW, M_), dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2, HINT, ((EK >> 1) & 1)); } while (0)
#define DISPATCH_K1T(D_) do { \
    if (MT == 1) { K1_TMA(D_, 1); } else if (MT == 2) { K1_TMA(D_, 2); } else { K1_TMA(D_, 4); } } while (0)
#define DISPATCH_K2T(D_) do { \
    if (MT == 1) { K2_TMA(D_, 1); } else if (MT == 2) { K2_TMA(D_, 2); } else { K2_TMA(D_, 4); } } while (0)

  const double b1 = (double)R.n_active * ((double)b13p.size() + (double)b13s.size());
  const double b2 = (double)R.n_active * ((double)w2p.size() + (double)w2s.size());

  // ---- sweep 模式：K1/K2 的 DEC x 几何 ----
  if (!std::strcmp(mode, "sweep")) {
    if (B > 64) { std::printf("sweep only for B<=64 (MTMAX=1)\n"); return 0; }
    auto bk1 = [&](auto dec, auto rw, auto dp, auto pp) {
      constexpr int D_ = decltype(dec)::value, RWs = decltype(rw)::value,
                    DPs = decltype(dp)::value, PPs = decltype(pp)::value;
      const int smem = SMEM_G(H, RWs, MTMAX, DPs, PPs);
      dim3 g(N1 / (8 * RWs), R.n_active);
      auto fn = [&] {
        if (PPs) launch_gemv<H, RWs, MTMAX, DPs, D_, 2, 1, 1>(g, smem, dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1);
        else     launch_gemv<H, RWs, MTMAX, DPs, D_, 2, 0, 1>(g, smem, dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1);
      };
      double t = bench_ms(fn, 3, 30);
      std::printf("  K1 DEC%d RWW%d DEPTH%d PIPE%d  %.4f ms  %.1f GB/s (%.1f%%)\n", D_, RWs, DPs, PPs, t,
                  to_gbps(b1, t), 100 * to_gbps(b1, t) / di.mem_bw_gbps);
    };
    auto bk2 = [&](auto dec, auto rw, auto dp, auto pp) {
      constexpr int D_ = decltype(dec)::value, RWs = decltype(rw)::value,
                    DPs = decltype(dp)::value, PPs = decltype(pp)::value;
      const int smem = SMEM_G(I, RWs, MTMAX, DPs, PPs);
      dim3 g(N2 / (8 * RWs), R.n_active);
      auto fn = [&] {
        if (PPs) launch_gemv<I, RWs, MTMAX, DPs, D_, 2, 1, 0>(g, smem, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2);
        else     launch_gemv<I, RWs, MTMAX, DPs, D_, 2, 0, 0>(g, smem, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2);
      };
      double t = bench_ms(fn, 3, 30);
      std::printf("  K2 DEC%d RWW%d DEPTH%d PIPE%d  %.4f ms  %.1f GB/s (%.1f%%)\n", D_, RWs, DPs, PPs, t,
                  to_gbps(b2, t), 100 * to_gbps(b2, t) / di.mem_bw_gbps);
    };
#define SW1(D_, RW_, DP_, PP_) bk1(std::integral_constant<int,D_>{}, std::integral_constant<int,RW_>{}, std::integral_constant<int,DP_>{}, std::integral_constant<int,PP_>{})
#define SW2(D_, RW_, DP_, PP_) bk2(std::integral_constant<int,D_>{}, std::integral_constant<int,RW_>{}, std::integral_constant<int,DP_>{}, std::integral_constant<int,PP_>{})
    std::printf("== sweep K1 (DEC 0=LUT256 4=LUT16 8=PRMT) ==\n");
    SW1(4, 4, 2, 1); SW1(0, 4, 2, 1); SW1(8, 4, 2, 1);
    SW1(4, 2, 2, 1); SW1(0, 2, 2, 1); SW1(8, 2, 2, 1);
    SW1(4, 8, 2, 1); SW1(0, 8, 2, 1); SW1(8, 8, 2, 1);
    SW1(0, 4, 3, 1); SW1(0, 4, 4, 1); SW1(8, 4, 3, 1);
    SW1(8, 4, 2, 0); SW1(8, 4, 3, 0); SW1(8, 4, 4, 0); SW1(8, 2, 2, 0); SW1(8, 8, 2, 0);
    std::printf("== sweep K2 (DEC 0=LUT256 4=LUT16 8=PRMT) ==\n");
    SW2(4, 8, 2, 1); SW2(0, 8, 2, 1); SW2(8, 8, 2, 1);
    SW2(4, 4, 2, 1); SW2(0, 4, 2, 1); SW2(8, 4, 2, 1);
    SW2(0, 8, 3, 1); SW2(8, 8, 3, 1); SW2(0, 4, 3, 1); SW2(8, 2, 2, 1);
    SW2(8, 8, 2, 0); SW2(8, 8, 3, 0); SW2(8, 8, 4, 0); SW2(8, 4, 3, 0); SW2(8, 4, 4, 0); SW2(8, 2, 3, 0);
#undef SW1
#undef SW2
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  // ---- tsweep 模式：TMA row-tile 的 (RWW, NWARP) 几何 ----
  if (!std::strcmp(mode, "tsweep")) {
    if (B > 64) { std::printf("tsweep only for B<=64 (MTMAX=1)\n"); return 0; }
    auto tk1 = [&](auto rw, auto nw) {
      constexpr int RW = decltype(rw)::value, NW = decltype(nw)::value;
      const int sm = SMEM_TMA(H, RW, NW, MTMAX);
      dim3 g(N1 / (NW * RW), R.n_active);
      double t = bench_ms([&] { launch_tma<H, RW, NW, MTMAX, 8, 1>(g, sm, dXq, dXs, dB13, dBs13, dCounts, dToff, dToks, dWexp, dGU, N1); }, 3, 30);
      std::printf("  K1-TMA RWW%d NW%d  %.4f ms  %.1f GB/s (%.1f%%)  smem=%dKB\n", RW, NW, t,
                  to_gbps(b1, t), 100 * to_gbps(b1, t) / di.mem_bw_gbps, sm / 1024);
    };
    auto tk2 = [&](auto rw, auto nw) {
      constexpr int RW = decltype(rw)::value, NW = decltype(nw)::value;
      const int sm = SMEM_TMA(I, RW, NW, MTMAX);
      dim3 g(N2 / (NW * RW), R.n_active);
      double t = bench_ms([&] { launch_tma<I, RW, NW, MTMAX, 8, 0>(g, sm, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2); }, 3, 30);
      std::printf("  K2-TMA RWW%d NW%d  %.4f ms  %.1f GB/s (%.1f%%)  smem=%dKB\n", RW, NW, t,
                  to_gbps(b2, t), 100 * to_gbps(b2, t) / di.mem_bw_gbps, sm / 1024);
    };
#define T1(RW_, NW_) tk1(std::integral_constant<int,RW_>{}, std::integral_constant<int,NW_>{})
#define T2(RW_, NW_) tk2(std::integral_constant<int,RW_>{}, std::integral_constant<int,NW_>{})
    std::printf("== tsweep K1 (TMA row-tile) ==\n");
    T1(1, 8); T1(2, 8); T1(4, 8); T1(2, 4); T1(4, 4);
    std::printf("== tsweep K2 (TMA row-tile) ==\n");
    T2(2, 8); T2(4, 8); T2(8, 8); T2(4, 4); T2(8, 4); T2(2, 4);
#undef T1
#undef T2
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  auto run_k1 = [&] {
    if (KTMA == 1) {
      if (DEC == 0) { DISPATCH_K1T(0); } else if (DEC == 4) { DISPATCH_K1T(4); } else { DISPATCH_K1T(8); }
      return;
    }
    if (DEC == 0) { DISPATCH_K1(0); }
    else if (DEC == 4) { DISPATCH_K1(4); }
    else { DISPATCH_K1(8); }
  };
  auto run_k2 = [&] {
    if (KTMA >= 1) {
      if (DEC == 0) { DISPATCH_K2T(0); } else if (DEC == 4) { DISPATCH_K2T(4); } else { DISPATCH_K2T(8); }
      return;
    }
    if (DEC == 0) { DISPATCH_K2(0); }
    else if (DEC == 4) { DISPATCH_K2(4); }
    else { DISPATCH_K2(8); }
  };
  auto run_all = [&] { run_quant(); run_k1(); run_swi(); run_k2(); run_unp(); };

  CUDA_CHECK(cudaMemset(dY, 0, (size_t)B * H * 4));
  run_all();
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());

  if (std::strcmp(mode, "none") != 0 && std::strcmp(mode, "check") != 0) {
    if (!std::strcmp(mode, "k1")) { for (int i = 0; i < 3; ++i) { run_quant(); run_k1(); } }
    else if (!std::strcmp(mode, "k2")) { for (int i = 0; i < 3; ++i) { run_quant(); run_k1(); run_swi(); run_k2(); } }
    else if (!std::strcmp(mode, "swi")) { for (int i = 0; i < 3; ++i) run_swi(); }
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  // ---- 正确性检查 ----
  auto sample_val = [&](const uint8_t* Wp, const uint8_t* Ws, int N_, int K_, int e, int row,
                        int tok, const std::vector<int8_t>& xq, const std::vector<float>& xs) {
    double ref = 0;
    for (int k = 0; k < K_; ++k) {
      const uint8_t byte = Wp[(size_t)row * (K_ / 2) + k / 2] ^ (uint8_t)(e * 37);
      const unsigned n = (byte >> ((k & 1) * 4)) & 0xF;
      const float w = e2m1f(n) * exp2f((int)Ws[(size_t)row * (K_ / 32) + k / 32] - 127);
      const float xv = (float)xq[(size_t)tok * K_ + k] * xs[(size_t)tok * (K_ / GROUP) + k / GROUP];
      ref += (double)w * (double)xv;
    }
    return ref;
  };
  std::vector<int8_t> hxq((size_t)B * H);
  std::vector<float> hxs((size_t)B * (H / GROUP));
  CUDA_CHECK(cudaMemcpy(hxq.data(), dXq, hxq.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hxs.data(), dXs, hxs.size() * 4, cudaMemcpyDeviceToHost));
  std::vector<float> hGU((size_t)pairs * N1);
  CUDA_CHECK(cudaMemcpy(hGU.data(), dGU, hGU.size() * 4, cudaMemcpyDeviceToHost));
  {
    double mr = 0;
    for (int s = 0; s < 8; ++s) {
      const int g = (s * 97 + 11) % R.n_active;
      const int e = R.wexp[g], off = R.toff[e];
      const int mm = (s * 3) % R.counts[e];
      const int tok = R.toks[off + mm];
      const int row = (s * 761 + 13) % I;
      const double ref_g = sample_val(w1p.data(), w1s.data(), I, H, e, row, tok, hxq, hxs);
      const double got_g = hGU[(size_t)(off + mm) * N1 + row];
      const double ref_u = sample_val(w3p.data(), w3s.data(), I, H, e, row, tok, hxq, hxs);
      const double got_u = hGU[(size_t)(off + mm) * N1 + I + row];
      mr = std::max(mr, std::fabs(got_g - ref_g) / std::max(std::fabs(ref_g), 1.0));
      mr = std::max(mr, std::fabs(got_u - ref_u) / std::max(std::fabs(ref_u), 1.0));
    }
    std::printf("[check K1 GU  ] max_rel=%.3e %s\n", mr, mr < 3e-2 ? "OK" : "FAIL");
  }

  std::vector<int8_t> hHq((size_t)pairs * I);
  std::vector<float> hHs((size_t)pairs * (I / GROUP));
  CUDA_CHECK(cudaMemcpy(hHq.data(), dHq, hHq.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hHs.data(), dHs, hHs.size() * 4, cudaMemcpyDeviceToHost));
  {
    double mr = 0;
    for (int s = 0; s < 16; ++s) {
      const int p = (s * 211 + 5) % pairs;
      const int q = (s * 617 + 7) % I;
      const float gv = fminf(hGU[(size_t)p * N1 + q], 10.0f);
      const float uv = fminf(fmaxf(hGU[(size_t)p * N1 + I + q], -10.0f), 10.0f);
      const float href = silu_f(gv) * uv;
      const float sc = hHs[(size_t)p * (I / GROUP) + q / GROUP];
      const float got = (float)hHq[(size_t)p * I + q] * sc;
      mr = std::max(mr, (double)std::fabs(got - href) / (sc > 0 ? sc : 1.f));
    }
    std::printf("[check swiglu ] max_err=%.3e (unit=group scale; <0.51 OK) %s\n",
                mr, mr < 0.6 ? "OK" : "FAIL");
  }

  std::vector<float> hO((size_t)pairs * N2);
  CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * 4, cudaMemcpyDeviceToHost));
  {
    double mr = 0, mref = 0;
    for (int s = 0; s < 8; ++s) {
      const int g = (s * 131 + 7) % R.n_active;
      const int e = R.wexp[g], off = R.toff[e];
      const int mm = (s * 5) % R.counts[e];
      const int n = (s * 883 + 19) % N2;
      double ref = 0;
      for (int q = 0; q < I; ++q) {
        const unsigned wb = w2p[(size_t)n * (I / 2) + q / 2] ^ (uint8_t)(e * 37);
        const unsigned nib = (q & 1) ? (wb >> 4) : (wb & 0xF);
        const float w = e2m1f(nib) * exp2f((int)w2s[(size_t)n * (I / 32) + q / 32] - 127);
        ref += (double)w * (double)hHq[(size_t)(off + mm) * I + q] *
               hHs[(size_t)(off + mm) * (I / GROUP) + q / GROUP];
      }
      const double got = hO[(size_t)(off + mm) * N2 + n];
      mr = std::max(mr, std::fabs(got - ref) / std::max(std::fabs(ref), 1.0));
      mref = std::max(mref, std::fabs(ref));
    }
    std::printf("[check K2 O   ] max_rel=%.3e (ref~%.2f) %s\n", mr, mref, mr < 3e-2 ? "OK" : "FAIL");
  }

  std::vector<float> hY((size_t)B * H);
  CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * 4, cudaMemcpyDeviceToHost));
  {
    std::vector<float> ref2((size_t)B * H, 0.f);
    for (int p = 0; p < pairs; ++p)
      for (int n = 0; n < N2; ++n)
        ref2[(size_t)R.toks[p] * H + n] += hpw[p] * hO[(size_t)p * N2 + n];
    double mr2 = 0;
    for (size_t i = 0; i < ref2.size(); ++i)
      mr2 = std::max(mr2, std::fabs((double)hY[i] - ref2[i]) / std::max(std::fabs((double)ref2[i]), 1.0));
    std::printf("[check unperm ] max_rel=%.3e %s\n", mr2, mr2 < 1e-3 ? "OK" : "FAIL");
    std::vector<int> gexp(pairs, -1);
    for (int g_ = 0; g_ < R.n_active; ++g_) {
      const int e = R.wexp[g_];
      for (int k = R.toff[e]; k < R.toff[e + 1]; ++k) gexp[k] = e;
    }
    double mr = 0, mref = 0;
    const int ngrp = I / GROUP;
    for (int s = 0; s < 4; ++s) {
      const int t = (s * 53 + 3) % B;
      const int n = (s * 977 + 17) % H;
      double ref = 0;
      for (int p = 0; p < pairs; ++p) {
        if (R.toks[p] != t) continue;
        const int e = gexp[p];
        double o = 0;
        for (int q = 0; q < I; ++q) {
          const unsigned wb = w2p[(size_t)n * (I / 2) + q / 2] ^ (uint8_t)(e * 37);
          const unsigned nib = (q & 1) ? (wb >> 4) : (wb & 0xF);
          const float w = e2m1f(nib) * exp2f((int)w2s[(size_t)n * (I / 32) + q / 32] - 127);
          o += (double)w * (double)hHq[(size_t)p * I + q] * hHs[(size_t)p * ngrp + q / GROUP];
        }
        ref += (double)hpw[p] * o;
      }
      mr = std::max(mr, std::fabs(hY[(size_t)t * H + n] - ref) / std::max(std::fabs(ref), 1.0));
      mref = std::max(mref, std::fabs(ref));
    }
    std::printf("[check e2e Y ] max_rel=%.3e (ref~%.2f) %s\n", mr, mref, mr < 3e-2 ? "OK" : "FAIL");
  }

  const double full_bytes = (double)R.n_active * ((double)b13p.size() + (double)b13s.size() +
                                                  (double)w2p.size() + (double)w2s.size());
  double tr = bench_ms([&] { read_bytes_kernel<<<2048, 256>>>(dB13, dY, (size_t)E_EXP * b13p.size()); }, 3, 20);
  std::printf("\n[roof] read all w1+w3 packed (%.2f GB): %.4f ms  %.1f GB/s (%.1f%% HBM)\n",
              (double)E_EXP * b13p.size() / 1e9, tr, to_gbps((double)E_EXP * b13p.size(), tr),
              100 * to_gbps((double)E_EXP * b13p.size(), tr) / di.mem_bw_gbps);

  std::printf("\n== per-stage (B=%d, PIPE=%d DEC=%d) ==\n", B, PIPE, DEC);
  double tq = bench_ms(run_quant, 3, 30);
  double t1 = bench_ms(run_k1, 3, 30);
  double ts = bench_ms(run_swi, 3, 30);
  double t2 = bench_ms(run_k2, 3, 30);
  double tu = bench_ms(run_unp, 3, 30);
  std::printf("  quant_x     %8.4f ms\n", tq);
  std::printf("  K1 gemv     %8.4f ms  weights %.2f GB  %.1f GB/s (%.1f%% HBM)\n", t1, b1 / 1e9,
              to_gbps(b1, t1), 100 * to_gbps(b1, t1) / di.mem_bw_gbps);
  std::printf("  swiglu+quant%8.4f ms\n", ts);
  std::printf("  K2 gemv     %8.4f ms  weights %.2f GB  %.1f GB/s (%.1f%% HBM)\n", t2, b2 / 1e9,
              to_gbps(b2, t2), 100 * to_gbps(b2, t2) / di.mem_bw_gbps);
  std::printf("  unpermute   %8.4f ms\n", tu);
  const double tsum = tq + t1 + ts + t2 + tu;
  std::printf("  ---- sum    %8.4f ms  (weights %.2f GB, roofline %.4f ms, %.1f%% of roof)\n",
              tsum, full_bytes / 1e9, full_bytes / di.mem_bw_gbps * 1000 / 1e9,
              100 * (full_bytes / di.mem_bw_gbps * 1000 / 1e9) / tsum);
  double tall = bench_ms(run_all, 3, 30);
  std::printf("  end-to-end  %8.4f ms  (kernel sum %.4f)\n", tall, tsum);
  return 0;
}
