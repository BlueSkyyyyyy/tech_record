// 60（主题 26o）：DeepSeek-V4-Pro routed expert FP4 decode —— grouped GEMV 的
// L1/TEX 收尾（LUT 布局 / occupancy）与真实稀疏路由。
//
// 承接 59 篇：grouped GEMV 把一步 decode 的 B×6 个 (token,expert) 对按 expert 折叠，
// 权重每专家只读一次、被 m 个 token 复用 → B=256 2.63ms / 2008 GB/s / 59.9% HBM。
// 59 的 ncu：DRAM 60.1% / **L1/TEX 83.9%** / Compute 41.1%，残余 wavefront 来自
// 256 项 uint16 LUT 的随机读（本篇先量：shared_ld 181.7M、wavefront 475.8M=2.62/inst、
// bank conflict 244.6M=1.35/inst；LUT 读 ≈ 每 weight byte 一次 = 5.28e9 lane-read）。
//
// 消融：
//   LUTMODE 0 uint16[256] 基线 | 1 uint32[256] 独占字 | 2/3 uint16 ×2/×4 副本+lane 选副本
//   LUTMODE 4 16 项 nibble 表（索引空间 ≤32、独占 bank ⇒ 结构无冲突，每 byte 两次 LDS）
//   MINCTA   __launch_bounds__(256,MINCTA) 压寄存器换 CTA/SM（59 是 117 regs / 2 CTA）
//   路由     balanced / 75% hot（96 专家 0 token）/ 幂律（尾部大量 0）
//
// shape 取自 /ssd/models/DeepSeek-V4-Pro/config.json：
//   hidden=7168, moe_intermediate_size=3072, n_routed_experts=384, topk=6, expert_dtype=fp4。
//
// 运行：scripts/run.sh 60-v4-fp4-grouped2/fp4_grouped2.cu [B]
//      scripts/run.sh 60-v4-fp4-grouped2/fp4_grouped2.cu 256 ncu <LUTMODE> <MINCTA>
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

template <int LUTMODE> __host__ __device__ constexpr int L16N() {
  if (LUTMODE == 0) return 256;
  if (LUTMODE == 1) return 1;
  if (LUTMODE == 2) return 129 + 256;
  if (LUTMODE == 3) return 3 * 129 + 256;
  return 16;
}
template <int LUTMODE> __host__ __device__ constexpr int L32N() { return LUTMODE == 1 ? 256 : 1; }

template <int LUTMODE>
__device__ __forceinline__ void init_lut(uint16_t* l16, uint32_t* l32, int tid) {
  if (LUTMODE == 0) {
    for (int i = tid; i < 256; i += 256) l16[i] = lut_entry(i);
  } else if (LUTMODE == 1) {
    for (int i = tid; i < 256; i += 256) l32[i] = (uint32_t)lut_entry(i);
  } else if (LUTMODE == 2) {
    for (int c = 0; c < 2; ++c)
      for (int i = tid; i < 256; i += 256) l16[c * 129 + i] = lut_entry(i);
  } else if (LUTMODE == 3) {
    for (int c = 0; c < 4; ++c)
      for (int i = tid; i < 256; i += 256) l16[c * 129 + i] = lut_entry(i);
  } else {
    for (int i = tid; i < 16; i += 256) l16[i] = (uint16_t)(uint8_t)e2m1_i8((unsigned)i);
  }
}

template <int LUTMODE>
__device__ __forceinline__ void expand_v(uint32_t w, uint32_t& a, uint32_t& b,
                                         const uint16_t* __restrict__ l16,
                                         const uint32_t* __restrict__ l32, int lane) {
  if (LUTMODE == 0) {
    const uint32_t r0 = l16[w & 0xFF], r1 = l16[(w >> 8) & 0xFF];
    const uint32_t r2 = l16[(w >> 16) & 0xFF], r3 = l16[(w >> 24) & 0xFF];
    a = r0 | (r1 << 16); b = r2 | (r3 << 16);
  } else if (LUTMODE == 1) {
    a = l32[w & 0xFF] | (l32[(w >> 8) & 0xFF] << 16);
    b = l32[(w >> 16) & 0xFF] | (l32[(w >> 24) & 0xFF] << 16);
  } else if (LUTMODE == 2 || LUTMODE == 3) {
    const uint16_t* t = l16 + ((LUTMODE == 2 ? (lane & 1) : (lane & 3)) * 129);
    const uint32_t r0 = t[w & 0xFF], r1 = t[(w >> 8) & 0xFF];
    const uint32_t r2 = t[(w >> 16) & 0xFF], r3 = t[(w >> 24) & 0xFF];
    a = r0 | (r1 << 16); b = r2 | (r3 << 16);
  } else {
    const uint32_t t0 = (uint32_t)(uint8_t)l16[w & 0xF] | ((uint32_t)(uint8_t)l16[(w >> 4) & 0xF] << 8);
    const uint32_t t1 = (uint32_t)(uint8_t)l16[(w >> 8) & 0xF] | ((uint32_t)(uint8_t)l16[(w >> 12) & 0xF] << 8);
    a = t0 | (t1 << 16);
    const uint32_t t2 = (uint32_t)(uint8_t)l16[(w >> 16) & 0xF] | ((uint32_t)(uint8_t)l16[(w >> 20) & 0xF] << 8);
    const uint32_t t3 = (uint32_t)(uint8_t)l16[(w >> 24) & 0xF] | ((uint32_t)(uint8_t)l16[(w >> 28) & 0xF] << 8);
    b = t2 | (t3 << 16);
  }
}

template <int RWW, int MTMAX, int DEPTH, int LUTMODE, int MINCTA>
__global__ void __launch_bounds__(256, MINCTA) gemv_fp4_g2(
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
  if (m == 0 || m > MTMAX) return;
  const int e = wexp[gid];
  const int off = toff[gid];

  constexpr int CH = KK / 32;
  extern __shared__ __align__(16) char smem[];
  int8_t* xs0 = reinterpret_cast<int8_t*>(smem);
  int8_t* xs1 = reinterpret_cast<int8_t*>(smem + (size_t)MTMAX * CH * 16);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MTMAX * KK);
  __shared__ uint16_t l16[L16N<LUTMODE>()];
  __shared__ uint32_t l32[L32N<LUTMODE>()];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  init_lut<LUTMODE>(l16, l32, tid);

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
      uint32_t w8[8];
      expand_v<LUTMODE>(w.x, w8[0], w8[1], l16, l32, lane);
      expand_v<LUTMODE>(w.y, w8[2], w8[3], l16, l32, lane);
      expand_v<LUTMODE>(w.z, w8[4], w8[5], l16, l32, lane);
      expand_v<LUTMODE>(w.w, w8[6], w8[7], l16, l32, lane);
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

__global__ void read_bytes_kernel(const uint8_t* __restrict__ p, float* __restrict__ out, size_t n) {
  uint32_t a = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n / 16;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(p + i * 16);
    a ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (a == 0xDEADBEEFu) out[0] = 1.f;
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

static std::vector<uint8_t> load_bin(const char* path, size_t expect) {
  FILE* f = std::fopen(path, "rb");
  if (!f) { std::fprintf(stderr, "open %s failed\n", path); std::exit(1); }
  std::vector<uint8_t> v(expect);
  size_t got = std::fread(v.data(), 1, expect, f);
  std::fclose(f);
  if (got != expect) { std::fprintf(stderr, "read %s got %zu/%zu\n", path, got, expect); std::exit(1); }
  return v;
}

struct Route { std::vector<int> counts, toff, toks, wexp; int n_active; };

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IONBF, 0);
  const int B = (argc > 1) ? std::atoi(argv[1]) : 256;
  const bool ncu_mode = (argc > 2 && std::strcmp(argv[2], "ncu") == 0);
  const int N = I, K = H;
  const int MMAX = B / 64;
  if (MMAX < 1 || MMAX > 8) { std::fprintf(stderr, "B must be a multiple of 64, in [64,512]\n"); return 1; }
  const int pairs = B * TOPK;

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nV4-Pro routed expert w1 [N=%d,K=%d], B=%d tokens, top-%d, E=%d experts\n",
              N, K, B, TOPK, E_EXP);

  const size_t wpc = (size_t)N * (K / 2);
  const size_t wsc = (size_t)N * (K / 32);
  const size_t eWbytes = wpc + wsc * 4;

  auto wpack = load_bin("w1_fp4.bin", wpc);
  auto wscal = load_bin("w1_scale.bin", wsc);

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

  // ---- 路由构造：0 balanced / 1 75% hot / 2 幂律 ----
  auto build = [&](int mode) {
    Route R;
    R.counts.assign(E_EXP, 0);
    std::vector<int> assign(pairs, -1);
    if (mode == 0) {
      for (int t = 0; t < B; ++t)
        for (int j = 0; j < TOPK; ++j) { const int e = (t * TOPK + j) % E_EXP; assign[t * TOPK + j] = e; R.counts[e]++; }
    } else if (mode == 1) {
      const int n_hot = (E_EXP * 3) / 4;
      for (int t = 0; t < B; ++t)
        for (int j = 0; j < TOPK; ++j) { const int e = (t * TOPK + j) % n_hot; assign[t * TOPK + j] = e; R.counts[e]++; }
    } else {
      std::vector<double> w(E_EXP); double sum = 0;
      for (int e = 0; e < E_EXP; ++e) { w[e] = 1.0 / (e + 1); sum += w[e]; }
      std::vector<double> cum(E_EXP); double acc = 0;
      for (int e = 0; e < E_EXP; ++e) { acc += pairs * w[e] / sum; cum[e] = acc; }
      int p = 0;
      for (int t = 0; t < B; ++t)
        for (int j = 0; j < TOPK; ++j, ++p) {
          int lo = 0, hi = E_EXP - 1;
          while (lo < hi) { int mid = (lo + hi) / 2; if (cum[mid] > p) hi = mid; else lo = mid + 1; }
          assign[p] = lo; R.counts[lo]++;
        }
    }
    // clamp > 8，重分配到有空位的专家（保持稀疏结构）
    for (int e = 0; e < E_EXP; ++e)
      if (R.counts[e] > 8) {
        R.counts[e] = 8;
        int over = 0;
        for (int p = 0; p < pairs; ++p) if (assign[p] == e) over++;
        over -= 8;
        for (int p = 0; p < pairs && over > 0; ++p)
          if (assign[p] == e) { assign[p] = -1; over--; }
      }
    for (int p = 0; p < pairs; ++p)
      if (assign[p] < 0)
        for (int e = 0; e < E_EXP; ++e) if (R.counts[e] < 8) { assign[p] = e; R.counts[e]++; break; }
    R.toff.assign(E_EXP + 1, 0);
    for (int e = 0; e < E_EXP; ++e) R.toff[e + 1] = R.toff[e] + R.counts[e];
    R.toks.assign(pairs, 0);
    std::vector<int> cur = R.toff;
    for (int p = 0; p < pairs; ++p) R.toks[cur[assign[p]]++] = p / TOPK;
    for (int e = 0; e < E_EXP; ++e) if (R.counts[e] > 0) R.wexp.push_back(e);
    R.n_active = (int)R.wexp.size();
    return R;
  };
  auto print_route = [&](const char* tag, const Route& R, const Route& R0) {
    int mx = 0, zer = 0;
    for (int e = 0; e < E_EXP; ++e) { if (R.counts[e] == 0) zer++; mx = std::max(mx, R.counts[e]); }
    const double ab = (double)R.n_active * eWbytes, fb = (double)R0.n_active * eWbytes;
    std::printf("routing %-10s active=%d/%d  zero=%d  max_m=%d  weight=%.2f GB (%.0f%% of full)\n",
                tag, R.n_active, E_EXP, zer, mx, ab / 1e9, 100 * ab / fb);
  };

  std::vector<bf16> hx((size_t)B * K);
  unsigned st = 12345u;
  for (auto& v : hx) { st = st * 1664525u + 1013904223u; v = __float2bfloat16(0.5f * ((st >> 8) / 8388608.0f - 1.f)); }

  uint8_t *dWall; float* dWsa; int8_t* dXq; float* dXs; bf16* dX; float* dY;
  CUDA_CHECK(cudaMalloc(&dWall, hWall.size()));
  CUDA_CHECK(cudaMalloc(&dWsa, hWsa.size() * 4));
  CUDA_CHECK(cudaMalloc(&dX, hx.size() * 2));
  CUDA_CHECK(cudaMalloc(&dXq, (size_t)B * K));
  CUDA_CHECK(cudaMalloc(&dXs, (size_t)B * (K / GROUP) * 4));
  CUDA_CHECK(cudaMalloc(&dY, (size_t)pairs * N * 4));
  CUDA_CHECK(cudaMemcpy(dWall, hWall.data(), hWall.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dWsa, hWsa.data(), hWsa.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dX, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice));
  {
    const int ngroup = B * (K / GROUP);
    quant_x_kernel<<<(ngroup + 7) / 8, 256>>>(dX, dXq, dXs, ngroup, K);
    CUDA_CHECK_LAST();
  }

  const double gbytes_fp4 = (double)E_EXP * wpc;
  double t4 = bench_ms([&] { read_bytes_kernel<<<2048, 256>>>(dWall, dY, hWall.size()); }, 5, 50);
  std::printf("\n[roof] read all %d experts fp4 (%.3f GB): %.4f ms  %.1f GB/s (%.1f%% HBM)\n",
              E_EXP, gbytes_fp4 / 1e9, t4, to_gbps(gbytes_fp4, t4),
              100 * to_gbps(gbytes_fp4, t4) / di.mem_bw_gbps);

  int *dCounts, *dToff, *dToks, *dWexp;
  CUDA_CHECK(cudaMalloc(&dCounts, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToff, E_EXP * 4));
  CUDA_CHECK(cudaMalloc(&dToks, pairs * 4));
  CUDA_CHECK(cudaMalloc(&dWexp, E_EXP * 4));

  // 压缩后的分组布局：gc/go/gt/gw + coff[e]（专家 e 在压缩输出里的起始行）
  struct Comp { std::vector<int> gc, go, gt, gw, coff; };
  auto compress = [&](const Route& Rt) {
    Comp C;
    C.gc.assign(Rt.n_active, 0); C.go.assign(Rt.n_active, 0); C.gw.assign(Rt.n_active, 0);
    C.gt.assign(pairs, 0); C.coff.assign(E_EXP, -1);
    int p = 0;
    for (int g = 0; g < Rt.n_active; ++g) {
      const int e = Rt.wexp[g];
      C.gc[g] = Rt.counts[e]; C.go[g] = p; C.gw[g] = e; C.coff[e] = p;
      for (int i = Rt.toff[e]; i < Rt.toff[e + 1]; ++i) C.gt[p++] = Rt.toks[i];
    }
    return C;
  };
  auto upload_grouped = [&](const Comp& C) {
    CUDA_CHECK(cudaMemcpy(dCounts, C.gc.data(), C.gc.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dToff, C.go.data(), C.go.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dToks, C.gt.data(), C.gt.size() * 4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWexp, C.gw.data(), C.gw.size() * 4, cudaMemcpyHostToDevice));
  };

  auto check = [&](const char* tag, const Route& Rt, const Comp& C) {
    std::vector<float> hY((size_t)pairs * N);
    CUDA_CHECK(cudaMemcpy(hY.data(), dY, hY.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int8_t> hxq_i((size_t)B * K);
    std::vector<float> hxs((size_t)B * (K / GROUP));
    CUDA_CHECK(cudaMemcpy(hxq_i.data(), dXq, hxq_i.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hxs.data(), dXs, hxs.size() * 4, cudaMemcpyDeviceToHost));
    double maxrel = 0, maxref = 0;
    for (int s = 0; s < 8; ++s) {
      const int g = (s * 173 + 11) % Rt.n_active;
      const int e = Rt.wexp[g];
      const int mm = (s * 3) % C.gc[g];
      const int off = C.coff[e];
      const int row = (s * 353 + 7) % N;
      const int tok = Rt.toks[off + mm];
      double ref = 0;
      const uint8_t* wp = hWall.data() + (size_t)e * wpc;
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
    std::printf("[check %-16s] max_rel=%.3e (ref~%.1f) %s\n", tag, maxrel, maxref,
                maxrel < 3e-2 ? "OK" : "FAIL");
  };

  const int sm4 = 4 * K + 4 * (K / GROUP) * 4;  // MTMAX=4
  const int sm8 = 8 * K + 8 * (K / GROUP) * 4;  // MTMAX=8

  // 供 ncu 用：单配置
  if (ncu_mode) {
    Route R = build(0);
    upload_grouped(compress(R));
    const int l = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int c = (argc > 4) ? std::atoi(argv[4]) : 2;
#define ATTR(MT_, L_, C_) CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_g2<4, MT_, 2, L_, C_>, cudaFuncAttributeMaxDynamicSharedMemorySize, (MT_) == 4 ? sm4 : sm8))
    ATTR(4, 0, 2); ATTR(4, 0, 3); ATTR(4, 0, 4); ATTR(4, 1, 2); ATTR(4, 2, 2); ATTR(4, 3, 2); ATTR(4, 4, 2); ATTR(8, 0, 2);
#undef ATTR
    for (int i = 0; i < 3; ++i) {
      if (l == 0 && c == 2) gemv_fp4_g2<4, 4, 2, 0, 2><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 0 && c == 3) gemv_fp4_g2<4, 4, 2, 0, 3><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 0 && c == 4) gemv_fp4_g2<4, 4, 2, 0, 4><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 1) gemv_fp4_g2<4, 4, 2, 1, 2><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 2) gemv_fp4_g2<4, 4, 2, 2, 2><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 3) gemv_fp4_g2<4, 4, 2, 3, 2><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else if (l == 4) gemv_fp4_g2<4, 4, 2, 4, 2><<<dim3(N / 32, R.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      else gemv_fp4_g2<4, 8, 2, 0, 2><<<dim3(N / 32, R.n_active), 256, sm8>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
      CUDA_CHECK_LAST();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  Route R0 = build(0);
  const double full_bytes = (double)R0.n_active * eWbytes;

  // ---- 路由消融（MTMAX=8，各模式都不丢 token）----
  std::printf("\n== routing ablation (RWW4, MTMAX8, LUTMODE0, CTA2) ==\n");
  double rt[3] = {0, 0, 0};
  for (int mode = 0; mode < 3; ++mode) {
    Route R = build(mode);
    Comp C = compress(R);
    print_route(mode == 0 ? "balanced" : mode == 1 ? "hot-75%" : "zipf", R, R0);
    upload_grouped(C);
    CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_g2<4, 8, 2, 0, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm8));
    CUDA_CHECK(cudaMemset(dY, 0, (size_t)pairs * N * 4));
    gemv_fp4_g2<4, 8, 2, 0, 2><<<dim3(N / 32, R.n_active), 256, sm8>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
    CUDA_CHECK_LAST();
    check(mode == 0 ? "grouped" : mode == 1 ? "grouped-hot" : "grouped-zipf", R, C);
    double t = bench_ms([&] { gemv_fp4_g2<4, 8, 2, 0, 2><<<dim3(N / 32, R.n_active), 256, sm8>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); CUDA_CHECK_LAST(); }, 3, 30);
    rt[mode] = t;
    const double ab = (double)R.n_active * eWbytes;
    std::printf("  MT8 m=%d  %8.4f ms  read %.2f GB  %.1f GB/s  (%.1f%% HBM)  speedup %.3fx\n",
                MMAX, t, ab / 1e9, to_gbps(ab, t), 100 * to_gbps(ab, t) / di.mem_bw_gbps,
                (mode == 0 ? 1.0 : rt[0] / t));
  }

  // ---- LUT 布局 × occupancy 消融（balanced，MTMAX=4）----
  std::printf("\n== LUT layout x occupancy (balanced RWW4 MTMAX4) ==\n");
  upload_grouped(compress(R0));
#define ATTR4(L_, C_) CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_g2<4, 4, 2, L_, C_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm4))
  ATTR4(0, 2); ATTR4(0, 3); ATTR4(0, 4); ATTR4(1, 2); ATTR4(2, 2); ATTR4(3, 2); ATTR4(4, 2);
#undef ATTR4
  {
    Comp C = compress(R0);
    CUDA_CHECK(cudaMemset(dY, 0, (size_t)pairs * N * 4));
    gemv_fp4_g2<4, 4, 2, 4, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N);
    CUDA_CHECK_LAST();
    check("M4-nibble16", R0, C);
  }
  auto time_cfg = [&](auto tag, auto launcher) -> double {
    CUDA_CHECK(cudaMemset(dY, 0, (size_t)pairs * N * 4));
    double t = bench_ms([&] { launcher(); CUDA_CHECK_LAST(); }, 3, 30);
    std::printf("  %-22s %8.4f ms  %.1f GB/s  (%.1f%% HBM)\n", tag, t,
                to_gbps(full_bytes, t), 100 * to_gbps(full_bytes, t) / di.mem_bw_gbps);
    return t;
  };
  time_cfg("M0 uint16 CTA2 (59)", [&] { gemv_fp4_g2<4, 4, 2, 0, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M1 uint32 CTA2",      [&] { gemv_fp4_g2<4, 4, 2, 1, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M2 x2copy CTA2",      [&] { gemv_fp4_g2<4, 4, 2, 2, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M3 x4copy CTA2",      [&] { gemv_fp4_g2<4, 4, 2, 3, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 nibble16 CTA2",    [&] { gemv_fp4_g2<4, 4, 2, 4, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M0 uint16 CTA3",      [&] { gemv_fp4_g2<4, 4, 2, 0, 3><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M0 uint16 CTA4",      [&] { gemv_fp4_g2<4, 4, 2, 0, 4><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });

  // ---- 用冲突无关的 M4 再扫 geometry / 流水深度 ----
  std::printf("\n== geometry x pipeline depth (M4 nibble, balanced) ==\n");
#define ATTRG(R_, D_, C_) CUDA_CHECK(cudaFuncSetAttribute((const void*)gemv_fp4_g2<R_, 4, D_, 4, C_>, cudaFuncAttributeMaxDynamicSharedMemorySize, sm4))
  ATTRG(4, 1, 2); ATTRG(2, 2, 2); ATTRG(2, 3, 2); ATTRG(8, 1, 2); ATTRG(8, 2, 2); ATTRG(1, 2, 2);
#undef ATTRG
  time_cfg("M4 RWW4 D1", [&] { gemv_fp4_g2<4, 4, 1, 4, 2><<<dim3(N / 32, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 RWW2 D2", [&] { gemv_fp4_g2<2, 4, 2, 4, 2><<<dim3(N / 16, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 RWW2 D3", [&] { gemv_fp4_g2<2, 4, 3, 4, 2><<<dim3(N / 16, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 RWW8 D1", [&] { gemv_fp4_g2<8, 4, 1, 4, 2><<<dim3(N / 64, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 RWW8 D2", [&] { gemv_fp4_g2<8, 4, 2, 4, 2><<<dim3(N / 64, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  time_cfg("M4 RWW1 D2", [&] { gemv_fp4_g2<1, 4, 2, 4, 2><<<dim3(N / 8, R0.n_active), 256, sm4>>>(dXq, dXs, dWall, dWsa, dCounts, dToff, dToks, dWexp, dY, N); });
  return 0;
}
