// 59（主题 26n）：DeepSeek-V4-Pro routed expert 的 FP4 decode **grouped GEMV** ——
// 把一批 decode token 命中的 (token, expert) 对合并进一次 launch，并按 expert 分组，
// 让每个专家的 FP4 权重只从 HBM 读一次、被该专家的 m 个 token 复用。
//
// 承接 58 篇：单专家 M=1 GEMV 拿到 0.0090 ms / 1296.6 GB/s / 38.7% HBM，判词是
//   「形状（N=3072 小）决定天花板，单点指令优化补不了延迟墙」——`Waves/SM=0.36`、
//   `Issued Ipc=1.08`，并行度只有 N/RWW=3072 个 warp。真实 decode 一步要算 top-6 个
//   专家，把这一步的 6B 个 (token,expert) 对放进一个 kernel，并且**按 expert 折叠**，
//   权重读次数从 `pairs` 降到 `active_experts`（reuse = pairs/active = B/64）。
//
// 三个口径（同一份真实权重）：
//   naive_loop   ：58 的 pipe kernel 每个 (token,expert) 对单独 launch（M=1，带 launch 税）
//   batched_m1   ：一个 kernel，grid.y = pairs，每 CTA 一个 (token,expert)，M=1（只赢并行度）
//   grouped      ：一个 kernel，grid.y = 384 experts，每 CTA 吃下该 expert 的全部 m 个 token
//                  （权重 decode 一次、dp4a 复用 m 次；赢并行度 + 权重复用）
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384, topk=6,
//   expert_dtype=fp4（w1 [N=3072, K=7168]，FP4 e2m1 + E8M0 block-32）。
//   平衡路由 (t*6+j)%384 让每个专家恰好拿到 B/64 个 token（B=64/128/256/512 → m=1/2/4/8）。
//
// 运行：scripts/run.sh 59-v4-fp4-grouped/fp4_grouped.cu [B]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

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

// ---------------------------------------------------------------------------
// e2m1 编码：LUT 把 1 byte（2 个 nibble）-> 2 个 int8（2*value）
// ---------------------------------------------------------------------------
__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}
__device__ __forceinline__ void expand_lut(uint32_t w, uint32_t& a, uint32_t& b,
                                           const uint16_t* __restrict__ lut) {
  const uint32_t r0 = lut[w & 0xFF], r1 = lut[(w >> 8) & 0xFF];
  const uint32_t r2 = lut[(w >> 16) & 0xFF], r3 = lut[(w >> 24) & 0xFF];
  a = r0 | (r1 << 16);
  b = r2 | (r3 << 16);
}
// PRMT（寄存器表，零 smem）：e2m1 nibble -> 2*value 的 int8（见 58 篇）
__device__ __forceinline__ uint32_t decode4(uint32_t e) {
  const uint32_t k = e & 0x07070707u;
  const uint32_t s = (e >> 3) & 0x01010101u;
  const uint32_t pk = (k & 0x0000000Fu) | ((k >> 4) & 0x000000F0u) |
                      ((k >> 8) & 0x00000F00u) | ((k >> 12) & 0x0000F000u);
  const uint32_t ps = (s & 0x1u) | ((s >> 4) & 0x10u) | ((s >> 8) & 0x100u) |
                      ((s >> 12) & 0x1000u);
  const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, pk);
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, pk);
  const uint32_t sel = 0x3210u | (ps << 2);
  return __byte_perm(pos, neg, sel);
}
__device__ __forceinline__ void expand_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  const uint32_t lo = w & 0x0F0F0F0Fu;
  const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
  a = decode4(__byte_perm(lo, hi, 0x5140));
  b = decode4(__byte_perm(lo, hi, 0x7362));
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}

// ---------------------------------------------------------------------------
// 单专家 pipe kernel（58 的最佳版，作为 naive_loop 的 per-pair kernel）
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, int DEPTH>
__global__ void __launch_bounds__(256) gemv_fp4_pipe(const int8_t* __restrict__ Aq,
                                                     const float* __restrict__ Axs,
                                                     const uint8_t* __restrict__ Wp,
                                                     const float* __restrict__ Ws,
                                                     const int* __restrict__ tok,
                                                     float* __restrict__ C, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = H / GROUP;
  constexpr int NS32 = H / 32;
  constexpr int NITER = (H / 2) / 16 / 32;  // 7

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int i = tid; i < 256; i += 256)
    lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
  const int t = tok[blockIdx.y];
  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)t * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[t * NGRP + i];
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
      expand_lut(w.x, w8[0], w8[1], lut);
      expand_lut(w.y, w8[2], w8[3], lut);
      expand_lut(w.z, w8[4], w8[5], lut);
      expand_lut(w.w, w8[6], w8[7], lut);
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
// grouped kernel：grid = (N/BN, G)，G = experts（grouped）或 pairs（batched_m1）。
//   gid 是「组」编号：ge = wexp[gid] 给出权重用的专家；counts[gid] 是该组 token 数；
//   toks[toff[gid] ..] 是 token 列表。输出写到 Y[(toff[gid]+mm)*N + row]。
//   MTMAX 固定（=8），要求 counts[gid] <= MTMAX。
// ---------------------------------------------------------------------------
template <int RWW, int MTMAX, int DEPTH, bool USELUT = true>
__global__ void __launch_bounds__(256) gemv_fp4_grouped(
    const int8_t* __restrict__ Aq, const float* __restrict__ Axs,
    const uint8_t* __restrict__ Wall, const float* __restrict__ Wsall,
    const int* __restrict__ counts, const int* __restrict__ toff,
    const int* __restrict__ toks, const int* __restrict__ wexp,
    float* __restrict__ Y, int N) {
  constexpr int KK = H;
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = H / GROUP;
  constexpr int NS32 = H / 32;
  constexpr int NITER = (H / 2) / 16 / 32;  // 7

  const int gid = blockIdx.y;
  const int m = counts[gid];
  if (m == 0) return;
  if (m > MTMAX) return;  // host 保证不会发生

  const int e = wexp[gid];
  const int off = toff[gid];

  constexpr int CH = KK / 32;  // 每个 token 的 16B chunk 数（32 k 值/chunk）
  extern __shared__ __align__(16) char smem[];
  // 激活用「两平面」布局：chunk c 的第一个 16B 放 plane0[c]、第二个放 plane1[c]，
  // 这样 lane 读 x 时相邻 lane 地址差 16B（合并、零 bank conflict，原来差 32B 是 8-way）。
  int8_t* xs0 = reinterpret_cast<int8_t*>(smem);
  int8_t* xs1 = reinterpret_cast<int8_t*>(smem + (size_t)MTMAX * CH * 16);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MTMAX * KK);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int i = tid; i < 256; i += 256)
    lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);

  // 该组 m 个 token 的 int8 激活 + per-128 scale 一次性搬进 smem（两平面）
  for (int i = tid; i < m * CH; i += 256) {
    const int mm = i / CH, c = i % CH;
    const uint4* src = reinterpret_cast<const uint4*>(Aq + (size_t)toks[off + mm] * KK);
    *reinterpret_cast<uint4*>(xs0 + ((size_t)mm * CH + c) * 16) = src[2 * c];
    *reinterpret_cast<uint4*>(xs1 + ((size_t)mm * CH + c) * 16) = src[2 * c + 1];
  }
  for (int i = tid; i < m * NGRP; i += 256) {
    const int mm = i / NGRP, g = i % NGRP;
    sxs[i] = Axs[(size_t)toks[off + mm] * NGRP + g];
  }
  __syncthreads();

  const int row_base = blockIdx.x * BN + warp * RWW;
  const uint8_t* We = Wall + (size_t)e * N * (KK / 2);
  const float* Wse = Wsall + (size_t)e * N * NS32;

  float acc[RWW][MTMAX];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int mm = 0; mm < MTMAX; ++mm) acc[r][mm] = 0.f;

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
    const int k0 = c * 32;
    const int g128 = k0 / GROUP;
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
      uint32_t w8[8];
      if constexpr (USELUT) {
        expand_lut(w.x, w8[0], w8[1], lut);
        expand_lut(w.y, w8[2], w8[3], lut);
        expand_lut(w.z, w8[4], w8[5], lut);
        expand_lut(w.w, w8[6], w8[7], lut);
      } else {
        expand_prmt(w.x, w8[0], w8[1]);
        expand_prmt(w.y, w8[2], w8[3]);
        expand_prmt(w.z, w8[4], w8[5]);
        expand_prmt(w.w, w8[6], w8[7]);
      }
      const float sw = __ldg(&Wse[(size_t)(row_base + r) * NS32 + c]);
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

// 纯读 roof
__global__ void read_bytes_kernel(const uint8_t* __restrict__ p, float* __restrict__ out, size_t n) {
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 16;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + i * 16);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0xDEADBEEFu) out[0] = 1.f;
}

// 激活量化（per-128 int8）
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
  const int B = (argc > 1) ? std::atoi(argv[1]) : 256;
  const bool ncu_mode = (argc > 2 && std::strcmp(argv[2], "ncu") == 0);
  const int N = I, K = H;
  const int MMAX = B / 64;  // 平衡路由下每个专家恰好拿到 B/64 个 token
  if (MMAX < 1 || MMAX > 8) { std::fprintf(stderr, "B must be a multiple of 64, in [64,512]\n"); return 1; }
  const int pairs = B * TOPK;

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nV4-Pro routed expert w1 [N=%d,K=%d], B=%d tokens, top-%d, E=%d experts\n",
              N, K, B, TOPK, E_EXP);
  std::printf("balanced routing: m_e = %d tokens/expert, pairs = %d\n", MMAX, pairs);

  const size_t wpc = (size_t)N * (K / 2);       // 单专家 fp4 权重字节
  const size_t wsc = (size_t)N * (K / 32);      // 单专家 scale 个数（float）
  const size_t eWbytes = wpc + wsc * 4;         // 单专家 fp4+scale

  auto wpack = load_bin("w1_fp4.bin", wpc);
  auto wscal = load_bin("w1_scale.bin", wsc);

  // 构造 E 个专家：权重 = 真实 w1 逐字节 XOR (e*37)（仍是合法 e2m1，且可查索引错），
  // scale = 真实 2^exp * (1 + 0.05*(e%5))（也可查 scale 索引错）。
  std::printf("building %d experts (fp4 %.2f GB + scale %.2f GB) ...\n", E_EXP,
              E_EXP * (double)wpc / 1e9, E_EXP * (double)wsc * 4 / 1e9);
  std::vector<uint8_t> hWall((size_t)E_EXP * wpc);
  for (int e = 0; e < E_EXP; ++e) {
    uint8_t* dst = hWall.data() + (size_t)e * wpc;
    const uint8_t x = (uint8_t)(e * 37);
    for (size_t i = 0; i < wpc; ++i) dst[i] = wpack[i] ^ x;
  }
  std::vector<float> hWsa((size_t)E_EXP * (N * (K / 32)));
  for (int e = 0; e < E_EXP; ++e) {
    float* dst = hWsa.data() + (size_t)e * N * (K / 32);
    const float fac = 1.0f + 0.05f * (e % 5);
    for (size_t i = 0; i < wsc; ++i) dst[i] = exp2f((int)wscal[i] - 127) * fac;
  }

  // 路由：token t 的 6 个专家 = (t*6+j)%E（周期 64 -> 每个专家恰好 B/64 个 token）
  std::vector<int> counts(E_EXP, 0), toff(E_EXP + 1, 0), toks(pairs), wexp_g(E_EXP);
  for (int t = 0; t < B; ++t)
    for (int j = 0; j < TOPK; ++j) counts[(t * TOPK + j) % E_EXP]++;
  for (int e = 0; e < E_EXP; ++e) toff[e + 1] = toff[e] + counts[e];
  {
    std::vector<int> cur = toff;
    for (int t = 0; t < B; ++t)
      for (int j = 0; j < TOPK; ++j) toks[cur[(t * TOPK + j) % E_EXP]++] = t;
  }
  for (int e = 0; e < E_EXP; ++e) wexp_g[e] = e;

  // batched_m1 的口径：每组 = 一个 pair，counts=1、toks=pair->token、wexp=pair->expert
  std::vector<int> b_counts(pairs, 1), b_toff(pairs), b_toks(pairs), b_wexp(pairs);
  {
    int p = 0;
    for (int e = 0; e < E_EXP; ++e)
      for (int i = toff[e]; i < toff[e + 1]; ++i, ++p) {
        b_toff[p] = p; b_toks[p] = toks[i]; b_wexp[p] = e;
      }
  }

  // 激活
  std::vector<bf16> hx((size_t)B * K);
  unsigned st = 12345u;
  for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = __float2bfloat16(0.5f * ((st >> 8) / 8388608.0f - 1.f)); }

  uint8_t *dWall; float* dWsa;
  int8_t* dXq; float* dXs; bf16* dX; float* dY; int *dCounts, *dToff, *dToks, *dWexp;
  CUDA_CHECK(cudaMalloc(&dWall, hWall.size()));
  CUDA_CHECK(cudaMalloc(&dWsa, hWsa.size() * 4));
  CUDA_CHECK(cudaMalloc(&dX, hx.size() * 2));
  CUDA_CHECK(cudaMalloc(&dXq, (size_t)B * K));
  CUDA_CHECK(cudaMalloc(&dXs, (size_t)B * (K / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)pairs * N * 4));
  CUDA_CHECK(cudaMalloc(&dCounts, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToff, (E_EXP + 1) * 4));
  CUDA_CHECK(cudaMalloc(&dToks, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dWexp, E_EXP * 4));
  CUDA_CHECK(cudaMemcpy(dWall, hWall.data(), hWall.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWsa, hWsa.data(), hWsa.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dCounts, counts.data(), E_EXP * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToff, toff.data(), (E_EXP + 1) * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToks, toks.data(), pairs * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWexp, wexp_g.data(), E_EXP * 4, cudaMemcpyHostToDevice));
  {
    const int ngroup = B * (K / GROUP);
    quant_x_kernel<<<(ngroup + 7) / 8, 256>>>(dX, dXq, dXs, ngroup, K);
    CUDA_CHECK_LAST();
  }

  // ---- 纯读 roof：读全部 384 专家的 fp4+scale（= grouped 的最少字节）----
  const double gbytes_all = (double)E_EXP * eWbytes;
  const double gbytes_fp4 = (double)E_EXP * wpc;
  double t4 = bench_ms([&] { read_bytes_kernel<<<2048, 256>>>(dWall, dY, hWall.size()); }, 5, 50);
  std::printf("\n[roof] read all %d experts fp4 (%.3f GB): %.4f ms  %.1f GB/s (%.1f%% HBM)\n",
              E_EXP, gbytes_fp4 / 1e9, t4, to_gbps(gbytes_fp4, t4),
              100 * to_gbps(gbytes_fp4, t4) / di.mem_bw_gbps);
  double t4s = bench_ms([&] { read_bytes_kernel<<<2048, 256>>>((const uint8_t*)dWsa, dY, hWsa.size() * 4); }, 5, 50);
  std::printf("[roof] read all scales (%.3f GB): %.4f ms  %.1f GB/s (%.1f%% HBM)  combined fp4+scale %.1f GB/s\n",
              hWsa.size() * 4 / 1e9, t4s, to_gbps((double)hWsa.size() * 4, t4s),
              100 * to_gbps((double)hWsa.size() * 4, t4s) / di.mem_bw_gbps,
              to_gbps(gbytes_all, t4 + t4s));

  // ---- 正确性：抽样 8 个 (pair, row) 对 CPU 参考 ----
  auto check = [&](const char* tag, float* out, const std::vector<int>& c, const std::vector<int>& tf,
                   const std::vector<int>& tk, const std::vector<int>& wx, int G) {
    std::vector<float> hY((size_t)pairs * N);
    CUDA_CHECK(cudaMemcpy(hY.data(), out, hY.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> hxq_i((size_t)B * K);
    std::vector<float> hxs((size_t)B * (K / GROUP));
    CUDA_CHECK(cudaMemcpy(hxq_i.data(), dXq, hxq_i.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hxs.data(), dXs, hxs.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0, maxref = 0;
    for (int s = 0; s < 8; ++s) {
      const int g = (s * 173 + 11) % G;
      const int mm = (s * 3) % c[g];
      const int off = tf[g];
      const int row = (s * 353 + 7) % N;
      const int e = wx[g];
      const int tok = tk[off + mm];
      double ref = 0;
      const uint8_t* wp = hWall.data() + (size_t)e * wpc;
      const float* wsp = hWsa.data() + (size_t)e * N * (K / 32);
      const float fac = 1.0f + 0.05f * (e % 5);
      for (int k = 0; k < K; ++k) {
        const unsigned n = (wp[(size_t)row * (K / 2) + k / 2] >> ((k & 1) * 4)) & 0xF;
        const float w = e2m1f(n) * exp2f((int)wscal[(size_t)row * (K / 32) + k / 32] - 127) * fac;
        const float xv = (float)hxq_i[(size_t)tok * K + k] * hxs[(size_t)tok * (K / GROUP) + k / GROUP];
        ref += (double)w * (double)xv;
      }
      const double got = hY[(size_t)(off + mm) * N + row];
      const double rel = std::fabs(got - ref) / std::max(std::fabs(ref), 1.0);
      maxrel = std::max(maxrel, rel); maxref = std::max(maxref, std::fabs(ref));
    }
    std::printf("[check %-14s] max_rel=%.3e (ref~%.1f) %s\n", tag, maxrel, maxref,
                maxrel < 3e-2 ? "OK" : "FAIL");
  };

  const int smemP = 1 * K + 1 * (K / GROUP) * 4;
  auto smem_for = [&](int mt) { return mt * K + mt * (K / GROUP) * 4; };
#define SET_ATTR(R_, MT_)                                                                     \
  CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_grouped<R_, MT_, 2>,                 \
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_for(MT_)))
  SET_ATTR(1, 1); SET_ATTR(1, 2); SET_ATTR(1, 4); SET_ATTR(1, 8);
  SET_ATTR(2, 1); SET_ATTR(2, 2); SET_ATTR(2, 4); SET_ATTR(2, 8);
  SET_ATTR(4, 1); SET_ATTR(4, 2); SET_ATTR(4, 4); SET_ATTR(4, 8);
  SET_ATTR(8, 1); SET_ATTR(8, 2); SET_ATTR(8, 4); SET_ATTR(8, 8);
#undef SET_ATTR

  // ---- naive_loop：58 的 pipe kernel 每个 pair 一次 launch ----
  auto run_naive = [&]() {
    for (int p = 0; p < pairs; ++p) {
      dim3 g(N / 8, 1);
      gemv_fp4_pipe<1, 1, H, 2><<<g, 256, smemP>>>(dXq, dXs, dWall + (size_t)b_wexp[p] * wpc,
                                                   dWsa + (size_t)b_wexp[p] * N * (K / 32), dToks + p, dY, N, K);
    }
  };

  // ---- batched_m1：一个 kernel，grid.y=pairs，M=1 ----
  int *dCountsB, *dToffB, *dToksB, *dWexpB;
  CUDA_CHECK(cudaMalloc(&dCountsB, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dToffB, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dToksB, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dWexpB, pairs * 4));
  CUDA_CHECK(cudaMemcpy(dCountsB, b_counts.data(), pairs * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToffB, b_toff.data(), pairs * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dToksB, b_toks.data(), pairs * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWexpB, b_wexp.data(), pairs * 4, cudaMemcpyHostToDevice));
  auto run_batched_real = [&]() {
    dim3 g(N / 8, pairs);
    gemv_fp4_grouped<1, 1, 2><<<g, 256, smemP>>>(dXq, dXs, dWall, dWsa, dCountsB, dToffB, dToksB,
                                                 dWexpB, dY, N);
    CUDA_CHECK_LAST();
  };

  // MTMAX 按真实 reuse 选，smem/寄存器随 m 缩放
  const int mt = (MMAX <= 1) ? 1 : (MMAX <= 2) ? 2 : (MMAX <= 4) ? 4 : 8;
  auto run_grouped = [&](int rww) {
    dim3 g(N / (8 * rww), E_EXP);
    const int sm = smem_for(mt);
#define DISP(R_, MT_)                                                                          \
  gemv_fp4_grouped<R_, MT_, 2><<<g, 256, sm>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks,   \
                                               dWexp, dY, N)
    if (rww == 1) {
      if (mt == 1) DISP(1, 1); else if (mt == 2) DISP(1, 2); else if (mt == 4) DISP(1, 4); else DISP(1, 8);
    } else if (rww == 2) {
      if (mt == 1) DISP(2, 1); else if (mt == 2) DISP(2, 2); else if (mt == 4) DISP(2, 4); else DISP(2, 8);
    } else if (rww == 4) {
      if (mt == 1) DISP(4, 1); else if (mt == 2) DISP(4, 2); else if (mt == 4) DISP(4, 4); else DISP(4, 8);
    } else {
      if (mt == 1) DISP(8, 1); else if (mt == 2) DISP(8, 2); else if (mt == 4) DISP(8, 4); else DISP(8, 8);
    }
#undef DISP
    CUDA_CHECK_LAST();
  };

  // PRMT 解码（去 smem LUT）版：只测 RWW=4 的 mt=4（B=256），验证「LUT bank conflict 是墙」
  CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_grouped<4, 4, 2, false>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem_for(4)));
  auto run_grouped_prmt = [&]() {
    dim3 g(N / (8 * 4), E_EXP);
    gemv_fp4_grouped<4, 4, 2, false><<<g, 256, smem_for(4)>>>(dXq, dXs, dWall, dWsa, dCounts, dToff,
                                                              dToks, dWexp, dY, N);
    CUDA_CHECK_LAST();
  };

  // ---- 结果 ----
  const double grouped_bytes = (double)E_EXP * eWbytes;
  const double naive_bytes = (double)pairs * eWbytes;

  if (!ncu_mode) {
    CUDA_CHECK(cudaMemset(dY, 0, (size_t)pairs * N * 4));
    run_grouped(1);
    check("grouped", dY, counts, toff, toks, wexp_g, E_EXP);

    for (int rep = 0; rep < 3; ++rep) {
      double tn = bench_ms([&] { run_naive(); }, 2, 10);
      std::printf("  naive_loop   %8.4f ms  (weight read %.2f GB, %.1f GB/s, %.1f%% HBM)\n", tn,
                  naive_bytes / 1e9, to_gbps(naive_bytes, tn),
                  100 * to_gbps(naive_bytes, tn) / di.mem_bw_gbps);
      double tb = bench_ms([&] { run_batched_real(); }, 2, 10);
      std::printf("  batched_m1   %8.4f ms  (weight read %.2f GB, %.1f GB/s, %.1f%% HBM)\n", tb,
                  naive_bytes / 1e9, to_gbps(naive_bytes, tb),
                  100 * to_gbps(naive_bytes, tb) / di.mem_bw_gbps);
      for (int rww : {1, 2, 4, 8}) {
        double tg = bench_ms([&] { run_grouped(rww); }, 3, 30);
        std::printf("  grouped RWW=%d %8.4f ms  (weight read %.2f GB, %.1f GB/s, %.1f%% HBM)\n", rww, tg,
                    grouped_bytes / 1e9, to_gbps(grouped_bytes, tg),
                    100 * to_gbps(grouped_bytes, tg) / di.mem_bw_gbps);
      }
      if (mt == 4) {
        CUDA_CHECK(cudaMemset(dY, 0, (size_t)pairs * N * 4));
        run_grouped_prmt();
        check("grouped-PRMT", dY, counts, toff, toks, wexp_g, E_EXP);
        double tp = bench_ms([&] { run_grouped_prmt(); }, 3, 30);
        std::printf("  grouped PRMT (RWW=4) %8.4f ms  (weight read %.2f GB, %.1f GB/s, %.1f%% HBM)\n", tp,
                    grouped_bytes / 1e9, to_gbps(grouped_bytes, tp),
                    100 * to_gbps(grouped_bytes, tp) / di.mem_bw_gbps);
      }
    }
  } else {
    const char* which = (argc > 3) ? argv[3] : "g2";
    for (int i = 0; i < 3; ++i) {
      if (std::strcmp(which, "b1") == 0) run_batched_real();
      else if (std::strcmp(which, "g1") == 0) run_grouped(1);
      else if (std::strcmp(which, "g4") == 0) run_grouped(4);
      else if (std::strcmp(which, "g8") == 0) run_grouped(8);
      else run_grouped(2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  return 0;
}
