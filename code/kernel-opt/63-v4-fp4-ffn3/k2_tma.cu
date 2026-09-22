// 63（主题 26r）：K2（w2, N=7168, K=3072）用 TMA 1D bulk 搬整「warp row-tile」
//
// 承接 62（主题 26q）的 K2 判词：
//   K2 occ 34.3%（Block Limit Shared Mem=3，动态 smem 68.7KB，其中 cp.async staging 64KB）、
//   `long_scoreboard` 40.5%、NITER=K/1024=3 太短、DRAM 78.2% / L2 84.7%。
//   几何扫参（RWW/DEPTH/PIPE）全负 -> 得改结构。
//
// 本篇结构改动：K2 的一个 warp 的 RWW 行权重在显存里**天然连续**
//   （row_base..row_base+RWW-1，每行 KK/2=1536B，共 RWW*1536B），
//   所以可以用 **一条 `cp.async.bulk`（1D TMA）** 把整块搬进 smem，scale 段同理（RWW*96B）。
//   每 warp 私有 mbarrier（warp 内 lane0 发 + expect_tx，其余 lane 等 parity）。
//   对比 62 的 cp.async 版：每 warp 24 条 `cp.async`（DEPTH2×RWW8）-> 2 条 bulk。
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：hidden=7168, moe_inter=3072,
//   n_routed_experts=384, topk=6, expert_dtype=fp4（w2 [7168,3072]，FP4 e2m1 + E8M0 block-32）。
//
// 运行：scripts/run.sh 63-v4-fp4-ffn3/k2_tma.cu [B] [mode]
//   mode: none|check|bench|sweep
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int H = 7168;
constexpr int I = 3072;
constexpr int GROUP = 128;
constexpr int E_EXP = 384;
constexpr int TOPK = 6;

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
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}

// PRMT 位拼装（与 62 相同）：一个 32-bit 字（8 nibble）-> 8 个 int8（a,b 各 4 个，k 顺序）。
__device__ __forceinline__ void decode_word_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  constexpr uint32_t T0 = 0x03020100u, T1 = 0x0C080604u;
  constexpr uint32_t T2 = 0xFDFEFF00u, T3 = 0xF4F8FAFCu;
  const uint32_t hi = w >> 16;
  const uint32_t mlo = __byte_perm(T0, T1, w);
  const uint32_t mhi = __byte_perm(T0, T1, hi);
  const uint32_t slo = __byte_perm(T2, T3, w);
  const uint32_t shi = __byte_perm(T2, T3, hi);
  const uint32_t klo = __byte_perm(0u, ~0u, ((w >> 3) & 0x11111111u) * 5u);
  const uint32_t khi = __byte_perm(0u, ~0u, ((w >> 19) & 0x11111111u) * 5u);
  a = mlo ^ ((mlo ^ slo) & klo);
  b = mhi ^ ((mhi ^ shi) & khi);
}
__device__ __forceinline__ void decode_word(uint32_t w, uint32_t& a, uint32_t& b) {
  decode_word_prmt(w, a, b);
}

// ---- TMA / mbarrier PTX ----
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint32_t addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t addr, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
               "r"(bytes)
               : "memory");
}
__device__ __forceinline__ void tma_1d(uint32_t dst, const void* src, uint32_t bytes,
                                       uint32_t mbar) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
          dst),
      "l"(src), "r"(bytes), "r"(mbar)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t addr, int phase) {
  asm volatile(
      "{\n\t.reg .pred p;\n\tWAIT_%=:\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n\t"
      "@!p bra WAIT_%=;\n\t}" ::"r"(addr),
      "r"(phase));
}

// ---------------------------------------------------------------------------
// TMA 版 grouped GEMV：一个 warp = 一个专家的一组连续 RWW 行，整 row-tile 一条 bulk。
// ---------------------------------------------------------------------------
template <int KK, int RWW, int NWARP, int MTMAX, int MINCTA, int GATHER>
__global__ void __launch_bounds__(NWARP * 32, MINCTA) gemv_fp4_tma(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs,
    const uint8_t* __restrict__ Wall, const uint8_t* __restrict__ Wsall,
    const int* __restrict__ counts, const int* __restrict__ toff,
    const int* __restrict__ toks, const int* __restrict__ wexp,
    float* __restrict__ Y, int N) {
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int CH = KK / 32;
  constexpr int NS32 = KK / 32;
  constexpr int NITER = KK / 1024;
  constexpr int WBYTES = RWW * (KK / 2);   // 一个 warp 的权重 row-tile 字节
  constexpr int SBYTES = RWW * NS32;       // 一个 warp 的 scale 段字节
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
  uint8_t* wsm = reinterpret_cast<uint8_t*>(sxs + (size_t)MTMAX * NGRP);       // NWARP*WBYTES
  uint8_t* ssm = wsm + (size_t)NWARP * WBYTES;                                 // NWARP*SBYTES
  uint64_t* mbar = reinterpret_cast<uint64_t*>(ssm + (size_t)NWARP * SBYTES);  // NWARP

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (lane == 0) mbar_init(smem_u32(&mbar[warp]), 1);

  for (int i = tid; i < m * CH; i += T) {
    const int mm = i / CH, c = i % CH;
    const int ar = GATHER ? toks[off + mm] : (off + mm);  // K1 按 token gather，K2 per-pair
    const uint4* src = reinterpret_cast<const uint4*>(Aq + (size_t)ar * KK);
    *reinterpret_cast<uint4*>(xs0 + ((size_t)mm * CH + c) * 16) = src[2 * c];
    *reinterpret_cast<uint4*>(xs1 + ((size_t)mm * CH + c) * 16) = src[2 * c + 1];
  }
  for (int i = tid; i < m * NGRP; i += T) {
    const int mm = i / NGRP, g = i % NGRP;
    sxs[i] = Axs[(size_t)(GATHER ? toks[off + mm] : (off + mm)) * NGRP + g];
  }
  __syncthreads();

  const int row_base = blockIdx.x * BN + warp * RWW;
  const uint8_t* We = Wall + (size_t)e * N * (KK / 2);
  const uint8_t* Wse = Wsall + (size_t)e * N * NS32;

  if (lane == 0) {
    mbar_expect_tx(smem_u32(&mbar[warp]), (uint32_t)(WBYTES + SBYTES));
    tma_1d(smem_u32(wsm + (size_t)warp * WBYTES), We + (size_t)row_base * (KK / 2), WBYTES,
           smem_u32(&mbar[warp]));
    tma_1d(smem_u32(ssm + (size_t)warp * SBYTES), Wse + (size_t)row_base * NS32, SBYTES,
           smem_u32(&mbar[warp]));
  }
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
      for (int j = 0; j < 4; ++j) decode_word(wv[j], w8[2 * j], w8[2 * j + 1]);
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
// 62 的 cp.async 版（对照），K2 只用得到这一支。
// ---------------------------------------------------------------------------
template <int KK, int RWW, int MTMAX, int DEPTH, int MINCTA>
__global__ void __launch_bounds__(256, MINCTA) gemv_fp4_pipe(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs,
    const uint8_t* __restrict__ Wall, const uint8_t* __restrict__ Wsall,
    const int* __restrict__ counts, const int* __restrict__ toff,
    const int* __restrict__ toks, const int* __restrict__ wexp,
    float* __restrict__ Y, int N) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;
  constexpr int CH = KK / 32;
  constexpr int NS32 = KK / 32;
  constexpr int NITER = KK / 1024;
  constexpr int WROW = 512;

  const int gid = blockIdx.y;
  const int m = counts[gid];
  if (m == 0 || m > MTMAX) return;
  const int e = wexp[gid];
  const int off = toff[gid];

  extern __shared__ __align__(128) char smem[];
  int8_t* xs0 = reinterpret_cast<int8_t*>(smem);
  int8_t* xs1 = reinterpret_cast<int8_t*>(smem + (size_t)MTMAX * CH * 16);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MTMAX * KK);
  char* wsz = reinterpret_cast<char*>(smem + (size_t)MTMAX * KK + (size_t)MTMAX * NGRP * 4);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  for (int i = tid; i < m * CH; i += 256) {
    const int mm = i / CH, c = i % CH;
    const uint4* src = reinterpret_cast<const uint4*>(Aq + (size_t)(off + mm) * KK);
    *reinterpret_cast<uint4*>(xs0 + ((size_t)mm * CH + c) * 16) = src[2 * c];
    *reinterpret_cast<uint4*>(xs1 + ((size_t)mm * CH + c) * 16) = src[2 * c + 1];
  }
  for (int i = tid; i < m * NGRP; i += 256) {
    const int mm = i / NGRP, g = i % NGRP;
    sxs[i] = Axs[(size_t)(off + mm) * NGRP + g];
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

  char* wp = wsz + (size_t)warp * DEPTH * RWW * WROW;
  auto issue = [&](int s) {
    const int ss = s < NITER ? s : NITER - 1;  // DEPTH>NITER 时不越出行界
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const void* src = We + (size_t)(row_base + r) * (KK / 2) + (lane + 32 * ss) * 16;
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
      for (int j = 0; j < 4; ++j) decode_word(wv[j], w8[2 * j], w8[2 * j + 1]);
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

static std::vector<uint8_t> load_bin(const char* path, size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
  std::vector<uint8_t> v(expect);
  size_t got = std::fread(v.data(), 1, expect, f);
  std::fclose(f);
  if (got != expect) { std::fprintf(stderr, "read %s got %zu/%zu\n", path, got, expect); std::exit(1); }
  return v;
}

#define SMEM_TMA(KK_, RWW, NWARP, MTMAX) \
  ((size_t)(MTMAX) * ((KK_) / 32) * 16 * 2 + (size_t)(MTMAX) * ((KK_) / GROUP) * 4 + \
   (size_t)(NWARP) * (RWW) * ((KK_) / 2) + (size_t)(NWARP) * (RWW) * ((KK_) / 32) + (size_t)(NWARP) * 8)
#define SMEM_PIPE(RWW, MTMAX, DEPTH) \
  ((size_t)(MTMAX) * (I / 32) * 16 * 2 + (size_t)(MTMAX) * (I / GROUP) * 4 + \
   (size_t)8 * (DEPTH) * (RWW) * 512)

template <int KK, int RWW, int NWARP, int MTMAX, int GATHER>
static void launch_tma(dim3 grid, int smem, const int8_t* Aq, const float* Axs,
                       const uint8_t* W, const uint8_t* Ws, const int* counts, const int* toff,
                       const int* toks, const int* wexp, float* Y, int N) {
  auto k = gemv_fp4_tma<KK, RWW, NWARP, MTMAX, 2, GATHER>;
  cudaError_t e = cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) { std::fprintf(stderr, "TMA setattr smem=%d failed: %s\n", smem, cudaGetErrorString(e)); std::exit(1); }
  k<<<grid, NWARP * 32, smem>>>(Aq, Axs, W, Ws, counts, toff, toks, wexp, Y, N);
}
template <int RWW, int MTMAX, int DEPTH>
static void launch_pipe(dim3 grid, int smem, const int8_t* Aq, const float* Axs,
                        const uint8_t* W, const uint8_t* Ws, const int* counts, const int* toff,
                        const int* toks, const int* wexp, float* Y, int N) {
  auto k = gemv_fp4_pipe<I, RWW, MTMAX, DEPTH, 2>;
  cudaError_t e = cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (e != cudaSuccess) { std::fprintf(stderr, "PIPE setattr smem=%d failed: %s\n", smem, cudaGetErrorString(e)); std::exit(1); }
  k<<<grid, 256, smem>>>(Aq, Axs, W, Ws, counts, toff, toks, wexp, Y, N);
}

struct Route { std::vector<int> counts, toff, toks, wexp; int n_active; };

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int B = (argc > 1) ? std::atoi(argv[1]) : 64;
  const char* mode = (argc > 2) ? argv[2] : "none";
  const int pairs = B * TOPK;
  const int N2 = H, K2 = I;

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nK2 TMA row-tile: w2 [%d,%d] FP4, B=%d top-%d E=%d pairs=%d mode=%s\n",
              N2, K2, B, TOPK, E_EXP, pairs, mode);

  const size_t w2sz = (size_t)H * (I / 2), w2ssz = (size_t)H * (I / 32);
  auto w2p = load_bin("../56-v4-fp4-moe/w2_fp4.bin", w2sz);
  auto w2s = load_bin("../56-v4-fp4-moe/w2_scale.bin", w2ssz);

  uint8_t *dB2, *dBs2, *b2d, *b2sd;
  CUDA_CHECK(cudaMalloc(&dB2, (size_t)E_EXP * w2p.size()));
  CUDA_CHECK(cudaMalloc(&dBs2, (size_t)E_EXP * w2s.size()));
  CUDA_CHECK(cudaMalloc(&b2d, w2p.size())); CUDA_CHECK(cudaMalloc(&b2sd, w2s.size()));
  CUDA_CHECK(cudaMemcpy(b2d, w2p.data(), w2p.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b2sd, w2s.data(), w2s.size(), cudaMemcpyHostToDevice));
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
  std::printf("routing balanced active=%d/%d max_m=%d\n", R.n_active, E_EXP,
              *std::max_element(R.counts.begin(), R.counts.end()));

  // 输入激活（int8 per-pair + per-128 scale），模拟 swiglu 输出
  const int ngrp = I / GROUP;
  std::vector<int8_t> hHq((size_t)pairs * I);
  std::vector<float> hHs((size_t)pairs * ngrp);
  unsigned st = 987654u;
  for (auto& v : hHq) { st = st * 1664525u + 1013904223u; v = (int8_t)((int)((st >> 24) % 255) - 127); }
  for (auto& v : hHs) { st = st * 1664525u + 1013904223u; v = 0.5f + (st >> 20) * 1e-3f; }

  int8_t *dHq; float *dHs, *dO, *dO2;
  CUDA_CHECK(cudaMalloc(&dHq, hHq.size()));
  CUDA_CHECK(cudaMalloc(&dHs, hHs.size() * 4));
  CUDA_CHECK(cudaMalloc(&dO, (size_t)pairs * N2 * 4));
  CUDA_CHECK(cudaMalloc(&dO2, (size_t)pairs * N2 * 4));
  CUDA_CHECK(cudaMemcpy(dHq, hHq.data(), hHq.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dHs, hHs.data(), hHs.size() * 4, cudaMemcpyHostToDevice));
  int *dCounts, *dToff, *dToks, *dWexp;
  CUDA_CHECK(cudaMalloc(&dCounts, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToff, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToks, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dWexp, E_EXP * 4));
  CUDA_CHECK(cudaMemcpy(dCounts, R.counts.data(), R.counts.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToff, R.toff.data(), E_EXP * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToks, R.toks.data(), R.toks.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWexp, R.wexp.data(), R.wexp.size() * 4, cudaMemcpyHostToDevice));

  const int MT = std::max(1, std::min(4, B / 64));
  const double b2 = (double)R.n_active * ((double)w2p.size() + (double)w2s.size());

  auto run_tma = [&](int sm) { launch_tma<I, 4, 8, 1, 0>(dim3(N2 / 32, R.n_active), sm, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO2, N2); };
  auto run_pipe = [&](int sm) { launch_pipe<8, 1, 2>(dim3(N2 / 64, R.n_active), sm, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2); };

  if (!std::strcmp(mode, "sweep")) {
    if (B > 64) { std::printf("sweep only for B<=64 (MTMAX=1)\n"); return 0; }
    auto bk_pipe = [&](auto rw, auto dp, auto mt) {
      constexpr int RW = decltype(rw)::value, DP = decltype(dp)::value, MTs = decltype(mt)::value;
      const int sm = SMEM_PIPE(RW, MTs, DP);
      dim3 g(N2 / (8 * RW), R.n_active);
      double t = bench_ms([&] { launch_pipe<RW, MTs, DP>(g, sm, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO, N2); }, 3, 30);
      std::printf("  PIPE RWW%d DEPTH%d MT%d  %.4f ms  %.1f GB/s (%.1f%%)\n", RW, DP, MTs, t,
                  to_gbps(b2, t), 100 * to_gbps(b2, t) / di.mem_bw_gbps);
    };
    auto bk_tma = [&](auto rw, auto nw, auto mt) {
      constexpr int RW = decltype(rw)::value, NW = decltype(nw)::value, MTs = decltype(mt)::value;
      const int sm = SMEM_TMA(I, RW, NW, MTs);
      dim3 g(N2 / (NW * RW), R.n_active);
      double t = bench_ms([&] { launch_tma<I, RW, NW, MTs, 0>(g, sm, dHq, dHs, dB2, dBs2, dCounts, dToff, dToks, dWexp, dO2, N2); }, 3, 30);
      std::printf("  TMA  RWW%d NW%d MT%d  %.4f ms  %.1f GB/s (%.1f%%)  smem=%dKB\n", RW, NW, MTs, t,
                  to_gbps(b2, t), 100 * to_gbps(b2, t) / di.mem_bw_gbps, sm / 1024);
    };
#define S_PIPE(RW_, DP_, MT_) bk_pipe(std::integral_constant<int,RW_>{}, std::integral_constant<int,DP_>{}, std::integral_constant<int,MT_>{})
#define S_TMA(RW_, NW_, MT_) bk_tma(std::integral_constant<int,RW_>{}, std::integral_constant<int,NW_>{}, std::integral_constant<int,MT_>{})
    std::printf("== K2 sweep (baseline cp.async) ==\n");
    S_PIPE(8, 2, 1); S_PIPE(4, 2, 1); S_PIPE(2, 2, 1); S_PIPE(8, 3, 1); S_PIPE(4, 3, 1); S_PIPE(2, 3, 1);
    std::printf("== K2 sweep (TMA row-tile) ==\n");
    S_TMA(8, 8, 1); S_TMA(4, 8, 1); S_TMA(2, 8, 1); S_TMA(8, 4, 1); S_TMA(4, 4, 1); S_TMA(16, 4, 1);
    S_TMA(16, 8, 1); S_TMA(8, 2, 1); S_TMA(32, 4, 1);
#undef S_PIPE
#undef S_TMA
    CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  // 正确性：先跑 TMA 版，与 CPU 参考对拍
  const int sm_tma = SMEM_TMA(I, 4, 8, 1);
  const int sm_pipe = SMEM_PIPE(8, 1, 2);
  CUDA_CHECK(cudaMemset(dO2, 0, (size_t)pairs * N2 * 4));
  run_tma(sm_tma);
  CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> hO2((size_t)pairs * N2);
  CUDA_CHECK(cudaMemcpy(hO2.data(), dO2, hO2.size() * 4, cudaMemcpyDeviceToHost));
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
        ref += (double)w * (double)hHq[(size_t)(off + mm) * I + q] * hHs[(size_t)(off + mm) * ngrp + q / GROUP];
      }
      const double got = hO2[(size_t)(off + mm) * N2 + n];
      mr = std::max(mr, std::fabs(got - ref) / std::max(std::fabs(ref), 1.0));
      mref = std::max(mref, std::fabs(ref));
    }
    std::printf("[check TMA K2 O] max_rel=%.3e (ref~%.2f) %s\n", mr, mref, mr < 3e-2 ? "OK" : "FAIL");
  }

  // head-to-head
  CUDA_CHECK(cudaMemset(dO, 0, (size_t)pairs * N2 * 4));
  run_pipe(sm_pipe);
  CUDA_CHECK_LAST(); CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<float> hO((size_t)pairs * N2);
  CUDA_CHECK(cudaMemcpy(hO.data(), dO, hO.size() * 4, cudaMemcpyDeviceToHost));
  {
    double mr = 0;
    for (size_t i = 0; i < hO.size(); ++i)
      mr = std::max(mr, std::fabs((double)hO[i] - (double)hO2[i]) / std::max(std::fabs((double)hO2[i]), 1.0));
    std::printf("[check pipe-vs-tma] max_rel=%.3e %s\n", mr, mr < 1e-3 ? "OK" : "FAIL");
  }

  double tp = bench_ms([&] { run_pipe(sm_pipe); }, 3, 30);
  double tt = bench_ms([&] { run_tma(sm_tma); }, 3, 30);
  std::printf("\n== K2 head-to-head (B=%d) ==\n", B);
  std::printf("  pipe (cp.async RWW8 D2)  %.4f ms  %.1f GB/s (%.1f%%)\n", tp, to_gbps(b2, tp),
              100 * to_gbps(b2, tp) / di.mem_bw_gbps);
  std::printf("  tma  (row-tile RWW4 NW8) %.4f ms  %.1f GB/s (%.1f%%)\n", tt, to_gbps(b2, tt),
              100 * to_gbps(b2, tt) / di.mem_bw_gbps);
  (void)MT;
  return 0;
}
