// 22 FP8 GEMM（一）：e4m3 + per-tensor scaling，mma.m16n8k32
//
//   C[M,N] = (sa * sb) * A[M,K] @ B[N,K]^T     （A/B 均为 e4m3，fp32 累加）
//
// 真实 shape：DeepSeek-V4-Pro MoE expert 的 up/gate 投影
//   M=tokens, N=2*moe_inter? 这里取 K=hidden=7168, N=moe_inter=3072。
//   config: hidden_size=7168, moe_intermediate_size=3072, quantization_config=
//           {fmt:e4m3, scale_fmt:ue8m0, weight_block_size:[128,128], activation dynamic}
//
// 与 13 篇 bf16 GEMM 的差别：
//   - 指令换成 mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32（K 维 32）
//   - FP8 一字节，ldmatrix 不直接适用 → 本版先用「按 mma 片段坐标的 32-bit smem 取数」，
//     每个 a0..a3 / b0..b1 恰好是 4 个连续 fp8，靠 padding 消 bank conflict
//   - 每元素 4 个 fp8 打包成 uint32
//
// 运行：ARCH=sm_90a scripts/run.sh 22-fp8-gemm/fp8_gemm.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_fp8.h>
#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

using fp8 = __nv_fp8_e4m3;
constexpr double FP8_PEAK = 1978.0;  // H100 e4m3 dense TC TFLOP/s

// ---------------------------------------------------------------------------
// mma.m16n8k32 e4m3
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_fp8(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ uint32_t ld32(const void* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

// fp8 把两个相邻字节当 b16 → ldmatrix 可直接用：一个 m8n8.b16 矩阵 = 8 行 × 16 个 fp8。
// A(16x32 fp8) = 16x16 b16 = 四个 m8n8 矩阵，x4 一次取回 a0..a3。
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}

// BK=128，每行加 16 字节 padding：行距 144B = 36 word，36 % 32 = 4 →
// 同一列偏移下 8 个 row-group 落在 bank {0,4,...,28}，与 lane 内 t4(0..3) 一起铺满 32 banks。
template <int BM, int BN, int BK, int STAGES>
struct Cfg {
  static constexpr int ASLD = BK + 16;
  static constexpr int BSLD = BK + 16;
  static constexpr int NTHREADS = 256;
  static constexpr int NWARP = NTHREADS / 32;
};

// ---------------------------------------------------------------------------
// 同步版（baseline，无流水）：每次搬一块 K 到 smem，再算
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK>
__global__ void __launch_bounds__(Cfg<BM, BN, BK, 1>::NTHREADS)
fp8_gemm_sync(const fp8* __restrict__ A, const fp8* __restrict__ B, float* __restrict__ C,
              int M, int N, int K, float scale) {
  constexpr int ASLD = BK + 16, BSLD = BK + 16;
  constexpr int NT = 256;
  constexpr int WM = 4, WN = 2;
  constexpr int WARP_M = BM / WM;  // 32
  constexpr int WARP_N = BN / WN;  // 64
  constexpr int MTM = WARP_M / 16; // 2
  constexpr int MTN = WARP_N / 8;  // 8

  __shared__ fp8 As[BM][ASLD];
  __shared__ fp8 Bs[BN][BSLD];

  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int t = threadIdx.x;

  const int g = lane >> 2, t4 = (lane & 3) * 4;

  float acc[MTM][MTN][4];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // 载入 A/B 分块
    for (int i = t; i < BM * BK / 16; i += NT) {
      const int r = i / (BK / 16), c16 = (i % (BK / 16)) * 16;
      *reinterpret_cast<uint4*>(&As[r][c16]) =
          *reinterpret_cast<const uint4*>(&A[(size_t)(block_row + r) * K + k0 + c16]);
    }
    for (int i = t; i < BN * BK / 16; i += NT) {
      const int r = i / (BK / 16), c16 = (i % (BK / 16)) * 16;
      *reinterpret_cast<uint4*>(&Bs[r][c16]) =
          *reinterpret_cast<const uint4*>(&B[(size_t)(block_col + r) * K + k0 + c16]);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK / 32; ++kk) {
      const int koff = kk * 32;
      uint32_t a[MTM][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i) {
        const fp8* pa = &As[wr * WARP_M + i * 16 + g][koff + t4];
        a[i][0] = ld32(pa);
        a[i][1] = ld32(pa + 8 * ASLD);
        a[i][2] = ld32(pa + 16);
        a[i][3] = ld32(pa + 8 * ASLD + 16);
      }
      uint32_t b[MTN][2];
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const fp8* pb = &Bs[wc * WARP_N + j * 8 + g][koff + t4];
        b[j][0] = ld32(pb);
        b[j][1] = ld32(pb + 16);
      }
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j) mma_fp8(acc[i][j], a[i], b[j]);
    }
    __syncthreads();
  }

  // epilogue
  const int r0 = block_row + wr * WARP_M, c0 = block_col + wc * WARP_N;
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
        const int c = c0 + j * 8 + (t4 >> 1) + (q & 1);
        if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j][q] * scale;
      }
}

// ---------------------------------------------------------------------------
// cp.async 多级流水版
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int STAGES, int WM, int WN>
__global__ void __launch_bounds__(WM* WN * 32)
fp8_gemm_pipe(const fp8* __restrict__ A, const fp8* __restrict__ B, float* __restrict__ C,
              int M, int N, int K, float scale) {
  constexpr int ASLD = BK + 16, BSLD = BK + 16;
  constexpr int NT = WM * WN * 32;
  constexpr int WARP_M = BM / WM, WARP_N = BN / WN;
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
  static_assert(MTN % 2 == 0, "MTN must be even for b0/b1 pairing");

  extern __shared__ __align__(16) char smem[];
  fp8 (*As)[ASLD] = reinterpret_cast<fp8(*)[ASLD]>(smem);
  fp8 (*Bs)[BSLD] = reinterpret_cast<fp8(*)[BSLD]>(smem + (size_t)STAGES * BM * ASLD);

  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  const int t = threadIdx.x;
  const int g = lane >> 2, t4 = (lane & 3) * 4;

  auto load_stage = [&](int st, int k0) {
    fp8* a = (fp8*)As[st * BM];
    for (int i = t; i < BM * BK / 16; i += NT) {
      const int r = i / (BK / 16), c16 = (i % (BK / 16)) * 16;
      __pipeline_memcpy_async(&a[r * ASLD + c16], &A[(size_t)(block_row + r) * K + k0 + c16], 16);
    }
    fp8* b = (fp8*)Bs[st * BN];
    for (int i = t; i < BN * BK / 16; i += NT) {
      const int r = i / (BK / 16), c16 = (i % (BK / 16)) * 16;
      __pipeline_memcpy_async(&b[r * BSLD + c16], &B[(size_t)(block_col + r) * K + k0 + c16], 16);
    }
    __pipeline_commit();
  };

  float acc[MTM][MTN][4];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

  const int nblk = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s)
    if (s < nblk) load_stage(s, s * BK);

  for (int kb = 0; kb < nblk; ++kb) {
    const int st = kb % STAGES;
    __pipeline_wait_prior(STAGES - 2);
    __syncthreads();
    {
      fp8(*a)[ASLD] = reinterpret_cast<fp8(*)[ASLD]>((fp8*)As + (size_t)st * BM * ASLD);
      fp8(*b)[BSLD] = reinterpret_cast<fp8(*)[BSLD]>((fp8*)Bs + (size_t)st * BN * BSLD);
#pragma unroll
      for (int kk = 0; kk < BK / 32; ++kk) {
        const int koff = kk * 32;
        // A: ldmatrix.x4，lane 0-7/8-15 指前 16 fp8（两半行），16-23/24-31 指后 16 fp8
        const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
        const int acol = (lane >> 4) * 16;
        uint32_t av[MTM][4];
#pragma unroll
        for (int i = 0; i < MTM; ++i)
          ldmatrix_x4(smem_u32(&a[wr * WARP_M + i * 16 + arow][koff + acol]), av[i]);
        // B: ldmatrix.x2，两个矩阵分别对应 k 的 [0,16) 与 [16,32)，即 b0/b1
        const int brow = lane & 7;
        const int bcol = ((lane >> 3) & 1) * 16;
        uint32_t bv[MTN][2];
#pragma unroll
        for (int j = 0; j < MTN; ++j) {
          uint32_t d[2];
          ldmatrix_x2(smem_u32(&b[wc * WARP_N + j * 8 + brow][koff + bcol]), d);
          bv[j][0] = d[0];
          bv[j][1] = d[1];
        }
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j) mma_fp8(acc[i][j], av[i], bv[j]);
      }
    }
    __syncthreads();
    const int next = kb + STAGES - 1;
    if (next < nblk) load_stage(next % STAGES, next * BK);
  }

  const int r0 = block_row + wr * WARP_M, c0 = block_col + wc * WARP_N;
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
        const int c = c0 + j * 8 + (t4 >> 1) + (q & 1);
        if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j][q] * scale;
      }
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int N = (argc > 2) ? std::atoi(argv[2]) : 3072;
  int K = (argc > 3) ? std::atoi(argv[3]) : 7168;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nFP8 e4m3 GEMM (per-tensor): M=%d N=%d K=%d  FLOPs=%.2f GFLOP\n\n", M, N, K,
              flops / 1e9);

  const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  fp8 *A, *B;
  float* C;
  CUDA_CHECK(cudaMalloc(&A, aN));
  CUDA_CHECK(cudaMalloc(&B, bN));
  CUDA_CHECK(cudaMalloc(&C, cN * 4));

  std::vector<float> hAf(aN), hBf(bN);
  std::vector<fp8> hA(aN), hB(bN);
  std::vector<float> hC(cN);
  srand(1234);
  auto q = [](float x) { return fp8(x); };
  for (size_t i = 0; i < aN; ++i) {
    float x = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
    hAf[i] = (float)q(x);
    hA[i] = q(x);
  }
  for (size_t i = 0; i < bN; ++i) {
    float x = 2.0f * ((float)rand() / RAND_MAX - 0.5f);
    hBf[i] = (float)q(x);
    hB[i] = q(x);
  }
  CUDA_CHECK(cudaMemcpy(A, hA.data(), aN, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bN, cudaMemcpyHostToDevice));

  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)hAf[(size_t)r * K + k] * (double)hBf[(size_t)c * K + k];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), C, cN * 4, cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 32; ++i) {
      int r = (i * 257 + 13) % M, c = (i * 131 + 7) % N;
      double e = cpu_entry(r, c);
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - e));
      ref = std::max(ref, std::fabs(e));
    }
    std::printf("  [%-9s] sampled max_abs_err=%.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 3e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tf = to_tflops(flops, ms);
    std::printf("%-14s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp8 peak)\n", tag, ms, tf,
                100.0 * tf / FP8_PEAK);
  };

  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };

  {
    using K0 = Cfg<64, 64, 128, 1>;
    auto kern = fp8_gemm_sync<64, 64, 128>;
    if (want("sync")) {
      dim3 grid(div_up(N, 64), div_up(M, 64));
      kern<<<grid, K0::NTHREADS>>>(A, B, C, M, N, K, 1.0f);
      CUDA_CHECK_LAST();
      check("sync");
      double t = bench_ms([&] { kern<<<grid, K0::NTHREADS>>>(A, B, C, M, N, K, 1.0f); }, 5, 30);
      report("sync64x64", t);
    }
  }

#define RUN_PIPE(NAME, BM, BN, BK, ST, WM, WN)                                              \
  do {                                                                                      \
    if (want(NAME)) {                                                                       \
      auto fn = fp8_gemm_pipe<BM, BN, BK, ST, WM, WN>;                                      \
      const int nt = (WM) * (WN) * 32;                                                      \
      const size_t shm = (size_t)(ST) * ((BM) + (BN)) * ((BK) + 16);                        \
      CUDA_CHECK(                                                                           \
          cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm)); \
      dim3 grid(div_up(N, BN), div_up(M, BM));                                              \
      fn<<<grid, nt, shm>>>(A, B, C, M, N, K, 1.0f);                                        \
      CUDA_CHECK_LAST();                                                                    \
      check(NAME);                                                                          \
      double t = bench_ms([&] { fn<<<grid, nt, shm>>>(A, B, C, M, N, K, 1.0f); }, 5, 30);   \
      report(NAME, t);                                                                      \
    }                                                                                       \
  } while (0)

  RUN_PIPE("p128x128x64s3", 128, 128, 64, 3, 4, 2);
  RUN_PIPE("p128x128x64s4", 128, 128, 64, 4, 4, 2);
  RUN_PIPE("p128x128x128s3", 128, 128, 128, 3, 4, 2);
  RUN_PIPE("p128x64x128s4", 128, 64, 128, 4, 4, 2);
  RUN_PIPE("p64x128x128s4", 64, 128, 128, 4, 2, 4);
  RUN_PIPE("p128x256x64s3", 128, 256, 64, 3, 4, 4);
  RUN_PIPE("p256x128x64s3", 256, 128, 64, 3, 8, 2);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
