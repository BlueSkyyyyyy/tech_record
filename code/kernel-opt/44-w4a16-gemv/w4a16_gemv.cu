// 44 W4A16 的 M=1 GEMV（主题 26d）——放弃张量核，贴权重带宽
//
// 40/41/43 把 W4A16 做成「wgmma + 主循环内反量化」的 GEMM，decode 最佳
// M=1 是 0.0788 ms / 565.5 GB/s（43 篇）。但它们都在做 `wgmma.m64n128k16`：
// M=64 的 tile 在真 decode（M=1）时 **63/64 的张量核 M 维全浪费**，
// kernel 完全不吃权重带宽（43 篇 ncu：DRAM 18%、Compute 47%）。
//
// 本篇问一个问题：M=1 时到底该不该用张量核？答案是不该——
//   · M=1 的「上投影」(y[1,N]=x[1,K]·W[N,K]^T) 是纯 GEMV，权重 int4
//     44.6 MB（N=17408,K=5120）必须全部读一遍，理想 ~13-16 µs；
//   · 张量核只有在 M 大（同一次 bf16 加载被多行复用）时才划算；
//   · 正确的做法是「每 warp 一行、warp 内 K 维并行 + shuffle 归约」的
//     带宽最优 GEMV，int4 用 16B 向量读、scale 预取进 smem。
//
// 真实 shape 取自 /ssd/models/qwen3-8B/config.json：
//   hidden_size=5120, intermediate_size=17408, group_size=128, 对称 int4。
// 测 M=1（默认）及小 M 扫（MT 个 token 共享一次权重读）。
//
// 运行：
//   ARCH="" scripts/run.sh 44-w4a16-gemv/w4a16_gemv.cu [M] [N] [K] [which]
//   （本文件无 wgmma，-arch=sm_90 即可；保留 ARCH 默认）
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

// ---------------------------------------------------------------------------
// v0 朴素：一个线程一个输出行，逐 k 标量读（基线）
// ---------------------------------------------------------------------------
__global__ void gemv_naive_kernel(const bf16* __restrict__ A, const uint8_t* __restrict__ Wp,
                                  const float* __restrict__ Ws, float* __restrict__ C, int N,
                                  int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= N) return;
  const uint8_t* wrow = Wp + (size_t)n * (K / 2);
  const float* srow = Ws + (size_t)n * (K / GROUP);
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    const uint8_t byte = wrow[k >> 1];
    const int q = ((k & 1) ? (byte >> 4) : (byte & 0xF)) - 8;
    acc += (float)q * srow[k / GROUP] * __bfloat162float(A[k]);
  }
  C[n] = acc;
}

// ---------------------------------------------------------------------------
// roofline：只把 Wp + Ws 读一遍（结果无意义），给出「权重读」的上界
// ---------------------------------------------------------------------------
__global__ void wread_roofline_kernel(const uint8_t* __restrict__ Wp, const float* __restrict__ Ws,
                                      float* __restrict__ out, int N, int K) {
  const size_t nword = (size_t)N * (K / 2) / 16;  // uint4 数
  uint32_t acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < nword;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = *reinterpret_cast<const uint4*>(Wp + i * 16);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  // 顺带读一点 scale，保证它也进带宽账
  const size_t ns = ((size_t)N * (K / GROUP)) / 8;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < ns;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 v = *reinterpret_cast<const float4*>(Ws + i * 4);
    acc ^= __float_as_uint(v.x) ^ __float_as_uint(v.y) ^ __float_as_uint(v.z) ^
           __float_as_uint(v.w);
  }
  out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = (float)acc;
}

// ---------------------------------------------------------------------------
// v1/v2/v3 带宽最优 GEMV
//   · block = 8 warps，每 warp 负责 RWW 个输出行、MT 个 token；
//   · x（MT×K bf16）与整块的 scale 预取进 smem，block 内复用；
//   · lane l 负责 K 维的 16B chunk c = l + 32*i（一次读 32 个 int4 = 32 个 k）；
//   · int4 反量化后与 x 做 FMA，warp shuffle 归约（无需跨 warp 同步）。
// ---------------------------------------------------------------------------
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
__global__ void __launch_bounds__(256) gemv_kernel(const bf16* __restrict__ A,
                                                   const uint8_t* __restrict__ Wp,
                                                   const float* __restrict__ Ws,
                                                   float* __restrict__ C, int M, int N, int K) {
  constexpr int NWARP = 8;
  constexpr int BN = NWARP * RWW;         // 每 block 输出行数
  constexpr int NGRP = KK / GROUP;         // 每行 scale 数
  constexpr int NCHUNK = (KK / 2) / 16 / 32;  // 每 lane 的 chunk 数（=K/1024）
  static_assert((KK / 2) % 16 == 0 && ((KK / 2) / 16) % 32 == 0, "K 必须被 1024 整除");
  static_assert(GROUP % 32 == 0, "一个 32-k chunk 必须落在单个 group 内");

  extern __shared__ __align__(16) char smem[];
  bf16* xs = reinterpret_cast<bf16*>(smem);                           // MT*KK
  float* ssm = reinterpret_cast<float*>(smem + (size_t)MT * KK * 2);   // BN*NGRP

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // x 预取：MT*KK bf16，16B 向量化（block 内复用；x 只占 10KB×MT，L2 兜住）
  for (int i = tid; i < MT * KK / 8; i += 256) {
    const int m = i / (KK / 8), c8 = i % (KK / 8);
    *reinterpret_cast<uint4*>(&xs[(size_t)m * KK + c8 * 8]) =
        *reinterpret_cast<const uint4*>(&A[(size_t)m * KK + c8 * 8]);
  }
  // scale 预取：BN*NGRP fp32
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
    // x：本 lane 需要的 32 个 k（4 个 uint4），一次读好给所有 RWW 行复用
    U4 xu[MT][4];
#pragma unroll
    for (int m = 0; m < MT; ++m)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        xu[m][q].u = *reinterpret_cast<const uint4*>(&xs[(size_t)m * KK + k0 + q * 8]);

    const int g = k0 / GROUP;  // 本 chunk 落在哪个 scale group
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

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
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
  std::printf("\nW4A16 GEMV: M=%d N=%d K=%d group=%d\n", M, N, K, GROUP);
  std::printf("FLOPs = %.3f GFLOP ; W_int4 = %.1f MB ; scales = %.1f MB ; W+scale = %.1f MB\n\n",
              flops / 1e9, w_bytes_int4 / 1e6, s_bytes / 1e6, total_w / 1e6);

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
  uint8_t* dWp;
  float *dWs, *dC, *dOut;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
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
    std::printf("  [%-22s] max_abs_err=%.3e (ref~%.2f) %s\n", tag, err, ref, ok ? "OK" : "FAIL");
    return ok;
  };
  auto report = [&](const char* tag, double ms) {
    const double tf = to_tflops(flops, ms);
    const double wbw = w_bytes_int4 * 1000.0 / ms / 1e9;
    const double tbw = total_w * 1000.0 / ms / 1e9;
    std::printf("%-22s %9.4f ms  %8.2f TFLOPS (%4.1f%% bf16)  W-bw %7.1f GB/s  W+s %7.1f (%4.1f%% HBM)\n",
                tag, ms, tf, 100.0 * tf / BF16_PEAK, wbw, tbw, 100.0 * tbw / d.mem_bw_gbps);
  };
  auto want = [&](const char* n) { return std::strcmp(which, "all") == 0 || std::strcmp(which, n) == 0; };

  // naive：1 thread/row
  if (want("naive") && M == 1) {
    auto launch = [&] { gemv_naive_kernel<<<div_up(N, 256), 256>>>(dA, dWp, dWs, dC, N, K); };
    launch();
    CUDA_CHECK_LAST();
    check("naive");
    report("naive", bench_ms(launch, 3, 20));
  }

  // roofline：只读 W，扫一波 grid 找纯读上界
  if (want("roof")) {
    const int nthr = 256;
    const int grids[] = {528, 1188, 2112, 4224};
    for (int nb : grids) {
      char nm[48];
      std::snprintf(nm, sizeof(nm), "roof_g%d", nb);
      auto launch = [&] { wread_roofline_kernel<<<nb, nthr>>>(dWp, dWs, dOut, N, K); };
      launch();
      CUDA_CHECK_LAST();
      report(nm, bench_ms(launch, 3, 30));
    }
  }

#define RUN_GEMV(NAME, RWW, MT)                                                                   \
    do {                                                                                          \
      constexpr int BN_ = 8 * (RWW);                                                              \
      const size_t shm = (size_t)(MT) * K * 2 + (size_t)BN_ * (K / GROUP) * 4;                \
      auto fn = gemv_kernel<RWW, MT, 5120>;                                                             \
      auto launch = [&] { fn<<<N / BN_, 256, shm>>>(dA, dWp, dWs, dC, M, N, K); };                \
      launch();                                                                                   \
      CUDA_CHECK_LAST();                                                                          \
      check(NAME);                                                                                \
      report(NAME, bench_ms(launch, 5, 50));                                                      \
    } while (0)

  if (want("all") || want("gemv") || want("g1")) RUN_GEMV("gemv_r1", 1, 1);
  if (want("all") || want("gemv") || want("g2")) RUN_GEMV("gemv_r2", 2, 1);
  if (want("all") || want("gemv") || want("g4")) RUN_GEMV("gemv_r4", 4, 1);
  if (want("all") || want("gemv") || want("g8")) RUN_GEMV("gemv_r8", 8, 1);
  if (M % 2 == 0 && want("mt2")) {
    constexpr int RWW = 4, MT = 2;
    const size_t shm = (size_t)MT * K * 2 + (size_t)(8 * RWW) * (K / GROUP) * 4;
    auto fn = gemv_kernel<RWW, MT, 5120>;
    auto launch = [&] { fn<<<N / (8 * RWW), 256, shm>>>(dA, dWp, dWs, dC, M, N, K); };
    launch();
    CUDA_CHECK_LAST();
    check("gemv_r4_mt2");
    report("gemv_r4_mt2", bench_ms(launch, 5, 50));
  }
  if (M % 4 == 0 && want("mt4")) {
    constexpr int RWW = 4, MT = 4;
    const size_t shm = (size_t)MT * K * 2 + (size_t)(8 * RWW) * (K / GROUP) * 4;
    auto fn = gemv_kernel<RWW, MT, 5120>;
    auto launch = [&] { fn<<<N / (8 * RWW), 256, shm>>>(dA, dWp, dWs, dC, M, N, K); };
    launch();
    CUDA_CHECK_LAST();
    check("gemv_r4_mt4");
    report("gemv_r4_mt4", bench_ms(launch, 5, 50));
  }

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dWp));
  CUDA_CHECK(cudaFree(dWs));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFree(dOut));
  return 0;
}
