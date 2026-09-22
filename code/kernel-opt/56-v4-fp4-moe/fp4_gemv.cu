// 56 DeepSeek-V4-Pro 的 routed expert 权重其实是 FP4（e2m1 + E8M0 block-32）
//
// 真实 checkpoint（/ssd/models/DeepSeek-V4-Pro）里：
//   · routed experts  w1/w3 [3072, 7168] w2 [7168, 3072]  →  存成 int8（2 个
//     e2m1 nibble/byte）+ E8M0 scale（block 32，值 = 2^(b-127)）
//   · shared expert / attention  →  FP8 e4m3 + ue8m0（block 128）
//   · lm_head  →  bf16
// 因为 e2m1 ⊂ e4m3 且 scale 是 2 的幂，FP4 可以**逐位无损**折进 e4m3（见 extract.py）。
//
// 本文件：用真实 FP4 权重（w1）做 decode 时的 up 投影 GEMV（M=1..小 M），
//   · 整数点积：激活量化成 int8（per-128 动态），权重 nibble→int8（v'=2*e2m1），
//     用 `__dp4a` 一次算 4 个；
//   · 每 32-k 的 E8M0 scale 折成 float（sw=2^(b-127)），点积后统一乘回。
// 对照：fp4 / fp8 / bf16 三种存储格式的纯读 roof，给出权重带宽的差距。
//
// 运行：scripts/run.sh 56-v4-fp4-moe/fp4_gemv.cu [MT] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using bf16 = __nv_bfloat16;

constexpr int H = 7168;   // hidden
constexpr int I = 3072;   // moe_intermediate
constexpr int GROUP = 128;  // 激活量化组

// ---------------------------------------------------------------------------
// e2m1 解码：nibble -> int8（2 倍值，整数化，[-12,12]）
//   value = e==0 ? m*0.5 : (1+m*0.5)*2^(e-1)
//   v' = 2*value = (e?2:0 + m) << (e?e-1:0)，再按符号取负
// ---------------------------------------------------------------------------
__device__ __forceinline__ int e2m1_i8(unsigned n) {
  const unsigned e = (n >> 1) & 3u, m = n & 1u, s = (n >> 3) & 1u;
  int v = (int)(((e ? 2u : 0u) + m) << (e ? e - 1 : 0u));
  return s ? -v : v;
}

// 一个 uint32（8 nibble = 8 个 fp4）-> 两个 uint32（每 4 个 int8）
__device__ __forceinline__ void expand_fp4(uint32_t w, uint32_t& a, uint32_t& b) {
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

// v2：256 项 uint16 LUT（一个输入 byte -> 两个 int8），省掉逐 nibble 的算术解码
__device__ __forceinline__ void expand_fp4_lut(uint32_t w, uint32_t& a, uint32_t& b,
                                               const uint16_t* __restrict__ lut) {
  const uint32_t r0 = lut[w & 0xFF], r1 = lut[(w >> 8) & 0xFF];
  const uint32_t r2 = lut[(w >> 16) & 0xFF], r3 = lut[(w >> 24) & 0xFF];
  a = r0 | (r1 << 16);
  b = r2 | (r3 << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
  return v;
}

// ---------------------------------------------------------------------------
// FP4 GEMV：每 warp RWW 行，warp 内沿 K 并行，dp4a 整数点积。
//   每个 lane 一个 32-k chunk：16B（uint4）权重 = 32 个 fp4；32B（int8）激活。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK, bool LUT = false>
__global__ void __launch_bounds__(256) gemv_fp4_kernel(const int8_t* __restrict__ Aq,
                                                       const float* __restrict__ Axs,
                                                       const uint8_t* __restrict__ Wp,
                                                       const float* __restrict__ Ws,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NGRP = KK / GROUP;   // 激活 scale 组数
  constexpr int NS32 = KK / 32;      // 权重 scale 组数（block 32）
  constexpr int NCHUNK = (KK / 2) / 16 / 32;
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");

  extern __shared__ __align__(16) char smem[];
  int8_t* xs = reinterpret_cast<int8_t*>(smem);
  float* sxs = reinterpret_cast<float*>(smem + (size_t)MT * KK);
  __shared__ uint16_t lut[256];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (LUT) {
    for (int i = tid; i < 256; i += 256)
      lut[i] = (uint16_t)(uint8_t)e2m1_i8(i & 0xF) | ((uint16_t)(uint8_t)e2m1_i8(i >> 4) << 8);
    __syncthreads();
  }

  for (int i = tid; i < MT * KK / 16; i += 256) {
    const int m = i / (KK / 16), c16 = i % (KK / 16);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c16 * 16]) =
        *reinterpret_cast<const uint4*>(&Aq[(size_t)m * KK + c16 * 16]);
  }
  for (int i = tid; i < MT * NGRP; i += 256) sxs[i] = Axs[i];
  __syncthreads();

  float acc[RWW][MT];
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.f;

#pragma unroll
  for (int i = 0; i < NCHUNK; ++i) {
    const int c = lane + 32 * i;          // 32-k chunk 下标
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
      const int row = blockIdx.x * BN + warp * RWW + r;
      const uint4 wv = __ldcs(reinterpret_cast<const uint4*>(Wp + (size_t)row * (KK / 2) + c * 16));
      uint32_t w8[8];
      if (LUT) {
        expand_fp4_lut(wv.x, w8[0], w8[1], lut);
        expand_fp4_lut(wv.y, w8[2], w8[3], lut);
        expand_fp4_lut(wv.z, w8[4], w8[5], lut);
        expand_fp4_lut(wv.w, w8[6], w8[7], lut);
      } else {
        expand_fp4(wv.x, w8[0], w8[1]);
        expand_fp4(wv.y, w8[2], w8[3]);
        expand_fp4(wv.z, w8[4], w8[5]);
        expand_fp4(wv.w, w8[6], w8[7]);
      }
      const float sw = __ldg(&Ws[(size_t)row * NS32 + c]);   // per-32 E8M0
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        int dot = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) dot = __dp4a((int)w8[q], (int)x8[m][q], dot);
        // v'=2*value -> *0.5
        acc[r][m] += 0.5f * sw * sxs[m * NGRP + g128] * (float)dot;
      }
    }
  }

#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + blockIdx.x * BN + warp * RWW + r] = v;
    }
}

// ---------------------------------------------------------------------------
// FP8 GEMV（34 篇假设的格式）：权重 1 byte/elem，无 scale（已折叠），激活 bf16。
// 对照用：同样的算子若权重是 fp8 会多快。
// ---------------------------------------------------------------------------
template <int RWW, int MT, int KK>
__global__ void __launch_bounds__(256) gemv_fp8_kernel(const bf16* __restrict__ A,
                                                       const uint8_t* __restrict__ W8,
                                                       float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;
  constexpr int NCHUNK = (KK / 32) / 32;  // 每 lane 一个 32-k chunk
  extern __shared__ __align__(16) char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  for (int i = tid; i < MT * KK; i += 256) xs[i] = A[i];
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
    float xf[MT][32];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int t = 0; t < 32; ++t) xf[m][t] = __bfloat162float(xs[(size_t)m * KK + k0 + t]);
#pragma unroll
    for (int r = 0; r < RWW; ++r) {
      const int row = blockIdx.x * BN + warp * RWW + r;
      const uint4 wv0 = __ldcs(reinterpret_cast<const uint4*>(W8 + (size_t)row * KK + c * 32));
      const uint4 wv1 = __ldcs(reinterpret_cast<const uint4*>(W8 + (size_t)row * KK + c * 32 + 16));
      const uint32_t w8[8] = {wv0.x, wv0.y, wv0.z, wv0.w, wv1.x, wv1.y, wv1.z, wv1.w};
      float wf[32];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const unsigned char* bp = reinterpret_cast<const unsigned char*>(&w8[q]);
#pragma unroll
        for (int t = 0; t < 4; ++t) wf[q * 4 + t] = __half2float(__nv_cvt_fp8_to_halfraw(bp[t], __NV_E4M3));
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        float dd = 0.f;
#pragma unroll
        for (int t = 0; t < 32; ++t) dd += wf[t] * xf[m][t];
        acc[r][m] += dd;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < RWW; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float v = warp_sum(acc[r][m]);
      if (lane == 0) C[(size_t)m * N + blockIdx.x * BN + warp * RWW + r] = v;
    }
}

// ---------------------------------------------------------------------------
// 纯读 roof
// ---------------------------------------------------------------------------
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
  if (a == 0xDEADBEEFu) out[0] = 1.f;  // 防优化
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

// 激活量化（per-128 int8），同 45 篇
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

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
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
  const char* which = (argc > 2) ? argv[2] : "all";
  const int N = I, K = H;            // w1: [I, H]
  const size_t wpc = (size_t)N * (K / 2);       // packed fp4 bytes
  const size_t wsc = (size_t)N * (K / 32);      // e8m0 scale bytes
  const size_t wf8c = (size_t)N * K;            // fp8 bytes
  const size_t wbfc = (size_t)N * K * 2;        // bf16 bytes

  DeviceInfo di = device_info(0);
  print_device_info(di);
  std::printf("\nV4-Pro routed expert w1 [N=%d, K=%d]  M=%d\n", N, K, MT);
  std::printf("weight bytes: fp4=%.2f MB (%.1f%%)  fp8=%.2f MB  bf16=%.2f MB\n",
              wpc / 1e6, 100.0 * wpc / wbfc, wf8c / 1e6, wbfc / 1e6);

  auto wpack = load_bin("w1_fp4.bin", wpc);
  auto wscal = load_bin("w1_scale.bin", wsc);
  auto wfp8 = load_bin("w1_fp8.bin", wf8c);
  auto wbf = load_bin("w1_bf16.bin", wbfc);

  // E8M0 -> float
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
  if (std::strcmp(which, "all") == 0 || std::strcmp(which, "roof") == 0) {
    double t4 = bench_ms([&] { read_fp4_kernel<<<1024, 256>>>(dWp, dWs, dC, N, K); }, 5, 50);
    double t8b = bench_ms([&] { read_bytes_kernel<<<1024, 256>>>(dW8, dC, wf8c); }, 5, 50);
    double tbf = bench_ms([&] { read_bytes_kernel<<<1024, 256>>>(dWbf, dC, wbfc); }, 5, 50);
    std::printf("\n[roof] read fp4+scale %.4f ms  %.1f GB/s (%.1f%% HBM)\n", t4,
                to_gbps(wpc + wsc, t4), 100 * to_gbps(wpc + wsc, t4) / di.mem_bw_gbps);
    std::printf("[roof] read fp8       %.4f ms  %.1f GB/s (%.1f%% HBM)\n", t8b,
                to_gbps(wf8c, t8b), 100 * to_gbps(wf8c, t8b) / di.mem_bw_gbps);
    std::printf("[roof] read bf16      %.4f ms  %.1f GB/s (%.1f%% HBM)\n", tbf,
                to_gbps(wbfc, tbf), 100 * to_gbps(wbfc, tbf) / di.mem_bw_gbps);
  }
  if (std::strcmp(which, "roof") == 0) return 0;
  if (std::strcmp(which, "ncu") == 0) {  // 单 kernel 剖析模式（LUT v2, RWW=1）
    const int smem = K + (K / GROUP) * 4;
    gemv_fp4_kernel<1, 1, H, true><<<N / 8, 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
    for (int i = 0; i < 3; ++i)
      gemv_fp4_kernel<1, 1, H, true><<<N / 8, 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }
  if (std::strcmp(which, "ncu1") == 0) {  // 算术解码 v1
    const int smem = K + (K / GROUP) * 4;
    for (int i = 0; i < 3; ++i)
      gemv_fp4_kernel<1, 1, H, false><<<N / 8, 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
  }

  // ---- 正确性 + 计时（按运行时 MT 分派）----
  auto dispatch = [&](int rww, int mt, float* out) {
    const int smem = mt * K + mt * (K / GROUP) * 4;
    if (mt == 1) {
      if (rww == 1) gemv_fp4_kernel<1, 1, H><<<N / 8, 256, smem>>>(dXq, dXs, dWp, dWs, out, 1, N, K);
      else if (rww == 2) gemv_fp4_kernel<2, 1, H><<<N / 16, 256, smem>>>(dXq, dXs, dWp, dWs, out, 1, N, K);
      else gemv_fp4_kernel<8, 1, H><<<N / 64, 256, smem>>>(dXq, dXs, dWp, dWs, out, 1, N, K);
    } else if (mt == 2) {
      if (rww == 2) gemv_fp4_kernel<2, 2, H><<<N / 16, 256, smem>>>(dXq, dXs, dWp, dWs, out, 2, N, K);
      else gemv_fp4_kernel<4, 2, H><<<N / 32, 256, smem>>>(dXq, dXs, dWp, dWs, out, 2, N, K);
    } else {
      if (rww == 2) gemv_fp4_kernel<2, 4, H><<<N / 16, 256, smem>>>(dXq, dXs, dWp, dWs, out, 4, N, K);
      else gemv_fp4_kernel<4, 4, H><<<N / 32, 256, smem>>>(dXq, dXs, dWp, dWs, out, 4, N, K);
    }
    CUDA_CHECK_LAST();
  };
  dispatch(1, MT, dC);
  std::vector<float> hC((size_t)MT * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), dC, hC.size() * 4, cudaMemcpyDeviceToHost));
  // CPU ref：真实 fp4 值，x 用 int8 反量化值
  std::vector<int8_t> hxq_i((size_t)MT * K);
  std::vector<float> hxs((size_t)MT * (K / GROUP));
  CUDA_CHECK(cudaMemcpy(hxq_i.data(), dXq, hxq_i.size(), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hxs.data(), dXs, hxs.size() * 4, cudaMemcpyDeviceToHost));
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
  std::printf("[check] fp4 GEMV vs CPU ref: max_rel=%.3e (ref~%.1f) %s\n", maxrel, maxref,
              maxrel < 3e-2 ? "OK" : "FAIL");

  // ---- 计时 ----
  std::printf("\n[gemv fp4] (B-read = fp4+scale 字节 / 耗时)\n");
  const double bytes = wpc + wsc;
  const std::vector<int> rwws = (MT == 1) ? std::vector<int>{8, 2, 1} : std::vector<int>{4, 2};
  for (int rww : rwws) {
    double t = bench_ms([&] { dispatch(rww, MT, dC); }, 5, 100);
    std::printf("  RWW=%d BN=%3d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)  %7.2f TFLOPS-equiv\n",
                rww, 8 * rww, t, to_gbps(bytes, t), 100 * to_gbps(bytes, t) / di.mem_bw_gbps,
                to_tflops(2.0 * N * K * MT, t));
  }

  // ---- v2：LUT 解码（MT=1）----
  if (MT == 1) {
    auto disp_lut = [&](int rww) {
      const int smem = K + (K / GROUP) * 4;
      if (rww == 1) gemv_fp4_kernel<1, 1, H, true><<<N / 8, 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
      else gemv_fp4_kernel<2, 1, H, true><<<N / 16, 256, smem>>>(dXq, dXs, dWp, dWs, dC, 1, N, K);
      CUDA_CHECK_LAST();
    };
    std::printf("\n[gemv fp4 v2: smem LUT decode]\n");
    for (int rww : {2, 1}) {
      double t = bench_ms([&] { disp_lut(rww); }, 5, 100);
      std::printf("  RWW=%d BN=%3d  %8.4f ms  %7.1f GB/s fp4-read (%4.1f%% HBM)\n", rww, 8 * rww, t,
                  to_gbps(bytes, t), 100 * to_gbps(bytes, t) / di.mem_bw_gbps);
    }
  }

  // ---- fp8 对照（34 篇假设的权重格式：1 byte/elem、无 scale）----
  auto dispatch8 = [&](int rww) {
    const int smem = MT * K * 2;
    if (rww == 1) gemv_fp8_kernel<1, 1, H><<<N / 8, 256, smem>>>(dX, dW8, dC, MT, N, K);
    else if (rww == 2) gemv_fp8_kernel<2, 1, H><<<N / 16, 256, smem>>>(dX, dW8, dC, MT, N, K);
    else if (rww == 4) gemv_fp8_kernel<4, 1, H><<<N / 32, 256, smem>>>(dX, dW8, dC, MT, N, K);
    else gemv_fp8_kernel<8, 1, H><<<N / 64, 256, smem>>>(dX, dW8, dC, MT, N, K);
    CUDA_CHECK_LAST();
  };
  std::printf("\n[gemv fp8] (34 篇假设的 fp8 权重；B-read = fp8 字节)\n");
  const std::vector<int> rwws8 = (MT == 1) ? std::vector<int>{8, 4, 2, 1} : std::vector<int>{4, 2};
  for (int rww : rwws8) {
    double t = bench_ms([&] { dispatch8(rww); }, 5, 100);
    std::printf("  RWW=%d BN=%3d  %8.4f ms  %7.1f GB/s fp8-read (%4.1f%% HBM)  %7.2f TFLOPS-equiv\n",
                rww, 8 * rww, t, to_gbps(wf8c, t), 100 * to_gbps(wf8c, t) / di.mem_bw_gbps,
                to_tflops(2.0 * N * K * MT, t));
  }
  return 0;
}
