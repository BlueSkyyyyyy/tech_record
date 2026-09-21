// 13 Tensor Core 入门：从 WMMA API 到裸 mma.sync.m16n8k16 + ldmatrix
//
//   C[M,N] = A[M,K] * B[K,N]   （行主序，输入 BF16，FP32 累加，输出 FP32）
//
// 版本：
//   wmma      : WMMA API (nvcuda::wmma)，smem 分块 128×128×16，无流水线
//   wmma_pipe : 同上，用 cp.async 2 级双缓冲把载入藏起来
//   mma       : 手写 PTX mma.sync.aligned.m16n8k16 + ldmatrix
//   mma_pipe  : 手写 mma + ldmatrix + cp.async 双缓冲
//
// 运行：scripts/run.sh 13-tensor-core/tensor_core.cu [M] [N] [K] [which]
#include "../common/cuda_utils.cuh"

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <mma.h>

#include <cmath>
#include <cstring>

using namespace nvcuda;
using bf16 = __nv_bfloat16;

// block 计算 128×128，BK=16（bf16 的 TC K 维），256 线程 = 8 warp。
constexpr int BM = 128, BN = 128, BK = 32;
constexpr int NTHREADS = 256;

// warp 布局 4×2：每个 warp 算 32×64 输出
constexpr int WM = 4, WN = 2;
constexpr int WARP_M = BM / WM;  // 32
constexpr int WARP_N = BN / WN;  // 64

constexpr int MTM = WARP_M / 16;  // m16n8k16 的 m-tile 数 = 2
constexpr int MTN = WARP_N / 8;   // m16n8k16 的 n-tile 数 = 8

// smem 行主序数组的「行距」加 padding，避免 ldmatrix 读取时行地址落到同一 bank。
// A 行 16 个 bf16 = 32B；加 8 个后 48B/12 word，8 行 bank = 0,12,24,4,16,28,8,20。
// B 行 128 个 bf16 = 256B；加 8 个后 272B/68 word，8 行 bank = 0,4,8,...,28。
constexpr int ASP = BK + 8;  // A smem leading dim (padded)
constexpr int BNP = BN + 8;  // B smem leading dim (padded)

// ---------------------------------------------------------------------------
// BF16 / smem 工具
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t pack2(bf16 lo, bf16 hi) {
  return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}

// ---------------------------------------------------------------------------
// smem 分块载入（普通 LDG → STS，同步版用）
//   As[BM][BK] 行主序；   Bs[BK][BN] 行主序
// ---------------------------------------------------------------------------
template <int LDA, int LDB>
__device__ __forceinline__ void load_tiles(const bf16* __restrict__ A, const bf16* __restrict__ B,
                                           bf16 (*As)[LDA], bf16 (*Bs)[LDB], int M, int N, int K,
                                           int block_row, int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < BM * BK / 8; i += NTHREADS) {
    const int row = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gr < M && gc + 7 < K) v = *reinterpret_cast<const uint4*>(&A[(size_t)gr * K + gc]);
    *reinterpret_cast<uint4*>(&As[row][c8]) = v;
  }
  for (int i = t; i < BK * BN / 8; i += NTHREADS) {
    const int row = i / (BN / 8), c8 = (i % (BN / 8)) * 8;
    const int gr = k0 + row, gc = block_col + c8;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (gr < K && gc + 7 < N) v = *reinterpret_cast<const uint4*>(&B[(size_t)gr * N + gc]);
    *reinterpret_cast<uint4*>(&Bs[row][c8]) = v;
  }
}

// cp.async 版：把 A、B 两个分块异步拷进 smem，并 commit 成一个 group
template <int LDA, int LDB>
__device__ __forceinline__ void prefetch_tiles(const bf16* __restrict__ A,
                                               const bf16* __restrict__ B, bf16 (*As)[LDA],
                                               bf16 (*Bs)[LDB], int M, int N, int K, int block_row,
                                               int block_col, int k0) {
  const int t = threadIdx.x;
  for (int i = t; i < BM * BK / 8; i += NTHREADS) {
    const int row = i / (BK / 8), c8 = (i % (BK / 8)) * 8;
    const int gr = block_row + row, gc = k0 + c8;
    if (gr < M && gc + 7 < K) {
      __pipeline_memcpy_async(&As[row][c8], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&As[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  for (int i = t; i < BK * BN / 8; i += NTHREADS) {
    const int row = i / (BN / 8), c8 = (i % (BN / 8)) * 8;
    const int gr = k0 + row, gc = block_col + c8;
    if (gr < K && gc + 7 < N) {
      __pipeline_memcpy_async(&Bs[row][c8], &B[(size_t)gr * N + gc], 16);
    } else {
      *reinterpret_cast<uint4*>(&Bs[row][c8]) = make_uint4(0, 0, 0, 0);
    }
  }
  __pipeline_commit();
}

// ---------------------------------------------------------------------------
// WMMA 版：用 nvcuda::wmma 的 fragment
// ---------------------------------------------------------------------------
__global__ void gemm_wmma(const bf16* __restrict__ A, const bf16* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ bf16 As[BM][BK];
  __shared__ bf16 Bs[BK][BN];

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WN, warp_col = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MTM][MTN/2];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tiles<BK, BN>(A, B, As, Bs, M, N, K, block_row, block_col, k0);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag[MTM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag[MTN / 2];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        wmma::load_matrix_sync(a_frag[i], &As[warp_row * WARP_M + i * 16][kk * 16], BK);
#pragma unroll
      for (int j = 0; j < MTN / 2; ++j)
        wmma::load_matrix_sync(b_frag[j], &Bs[kk * 16][warp_col * WARP_N + j * 16], BN);
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN / 2; ++j)
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN / 2; ++j)
      wmma::store_matrix_sync(&C[(size_t)(block_row + warp_row * WARP_M + i * 16) * N +
                                 block_col + warp_col * WARP_N + j * 16],
                              acc[i][j], N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// WMMA + cp.async 双缓冲
// ---------------------------------------------------------------------------
__global__ void gemm_wmma_pipe(const bf16* __restrict__ A, const bf16* __restrict__ B,
                               float* __restrict__ C, int M, int N, int K) {
  __shared__ bf16 As[2][BM][BK];
  __shared__ bf16 Bs[2][BK][BN];

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WN, warp_col = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MTM][MTN/2];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  prefetch_tiles<BK, BN>(A, B, As[0], Bs[0], M, N, K, block_row, block_col, 0);

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + BK;
    if (next < K) {
      prefetch_tiles<BK, BN>(A, B, As[stage ^ 1], Bs[stage ^ 1], M, N, K, block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(next < K ? 1 : 0);
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> a_frag[MTM];
      wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::row_major> b_frag[MTN / 2];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        wmma::load_matrix_sync(a_frag[i], &As[stage][warp_row * WARP_M + i * 16][kk * 16], BK);
#pragma unroll
      for (int j = 0; j < MTN / 2; ++j)
        wmma::load_matrix_sync(b_frag[j], &Bs[stage][kk * 16][warp_col * WARP_N + j * 16], BN);
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN / 2; ++j)
          wmma::mma_sync(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
    }
    __syncthreads();
    stage ^= 1;
  }

#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN / 2; ++j)
      wmma::store_matrix_sync(&C[(size_t)(block_row + warp_row * WARP_M + i * 16) * N +
                                 block_col + warp_col * WARP_N + j * 16],
                              acc[i][j], N, wmma::mem_row_major);
}

// ---------------------------------------------------------------------------
// 裸 mma.sync.aligned.m16n8k16 + ldmatrix
//   As[BM][BK]、Bs[BK][BN] 都是行主序：
//     A 片段用 ldmatrix（非转置）从 As 取 16×16 tile；
//     B 片段用 ldmatrix.trans 从 Bs 取 16×8 tile（转置成 col 布局）
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// 从 As/Bs（行主序）用 ldmatrix 取出 m16n8k16 的 A/B 片段并做 mma
__device__ __forceinline__ void mma_stage(const bf16 (*As)[ASP], const bf16 (*Bs)[BNP],
                                          float acc[MTM][MTN][4], int warp_row, int warp_col) {
  const int lane = threadIdx.x & 31;
  int kk = 0;
  (void)kk;
#pragma unroll
  for (kk = 0; kk < BK / 16; ++kk) {
    uint32_t a[MTM][4];
#pragma unroll
    for (int i = 0; i < MTM; ++i) {
      // ldmatrix.x4：lane 0-15 指向 16 行、列 0..7；lane 16-31 指向列 8..15
      const int row = (lane & 15);
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&As[warp_row * WARP_M + i * 16 + row][kk * 16 + col]);
      ldmatrix_x4(addr, a[i]);
    }
    uint32_t b[MTN][2];
#pragma unroll
    for (int g = 0; g < MTN / 2; ++g) {
      // ldmatrix.x4.trans：每次取 16(K)×16(N) 两列 n-tile
      const int row = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int col = (lane >> 4) * 8;
      uint32_t addr = smem_u32(&Bs[kk * 16 + row][warp_col * WARP_N + g * 16 + col]);
      uint32_t d[4];
      ldmatrix_x4_trans(addr, d);
      b[g * 2][0] = d[0];
      b[g * 2][1] = d[1];
      b[g * 2 + 1][0] = d[2];
      b[g * 2 + 1][1] = d[3];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) mma_m16n8k16(acc[i][j], a[i], b[j]);
  }
}

__device__ __forceinline__ void zero_acc(float acc[MTM][MTN][4]) {
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
}

__device__ __forceinline__ void store_acc(float* __restrict__ C, const float acc[MTM][MTN][4],
                                          int N, int block_row, int block_col, int warp_row,
                                          int warp_col) {
  const int lane = threadIdx.x & 31;
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
        C[(size_t)r * N + c] = acc[i][j][q];
      }
    }
}

__global__ void gemm_mma(const bf16* __restrict__ A, const bf16* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
  __shared__ bf16 As[BM][ASP];
  __shared__ bf16 Bs[BK][BNP];

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WN, warp_col = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  float acc[MTM][MTN][4];
  zero_acc(acc);

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tiles<ASP, BNP>(A, B, As, Bs, M, N, K, block_row, block_col, k0);
    __syncthreads();
    mma_stage(As, Bs, acc, warp_row, warp_col);
    __syncthreads();
  }
  store_acc(C, acc, N, block_row, block_col, warp_row, warp_col);
}

// ---------------------------------------------------------------------------
// 裸 mma + ldmatrix + cp.async 双缓冲
// ---------------------------------------------------------------------------
__global__ void gemm_mma_pipe(const bf16* __restrict__ A, const bf16* __restrict__ B,
                              float* __restrict__ C, int M, int N, int K) {
  __shared__ bf16 As[2][BM][ASP];
  __shared__ bf16 Bs[2][BK][BNP];

  const int wid = threadIdx.x >> 5;
  const int warp_row = wid / WN, warp_col = wid % WN;
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;

  float acc[MTM][MTN][4];
  zero_acc(acc);

  prefetch_tiles<ASP, BNP>(A, B, As[0], Bs[0], M, N, K, block_row, block_col, 0);

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + BK;
    if (next < K) {
      prefetch_tiles<ASP, BNP>(A, B, As[stage ^ 1], Bs[stage ^ 1], M, N, K, block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(next < K ? 1 : 0);
    __syncthreads();
    mma_stage(As[stage], Bs[stage], acc, warp_row, warp_col);
    __syncthreads();
    stage ^= 1;
  }
  store_acc(C, acc, N, block_row, block_col, warp_row, warp_col);
}

// ---------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
  int K = (argc > 3) ? std::atoi(argv[3]) : 2048;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nBF16 TC GEMM: M=N=K=%d, FLOPs = %.2f GFLOP\n\n", M, flops / 1e9);
  if (M % BM || N % BN || K % BK) {
    std::printf("M/N must be multiple of %d, K multiple of %d\n", BM, BK);
    return 1;
  }

  const size_t aN = (size_t)M * K, bN = (size_t)K * N, cN = (size_t)M * N;
  bf16 *A, *B;
  float* C;
  CUDA_CHECK(cudaMalloc(&A, aN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&B, bN * sizeof(bf16)));
  CUDA_CHECK(cudaMalloc(&C, cN * sizeof(float)));

  std::vector<bf16> hA(aN), hB(bN);
  std::vector<float> hC(cN);
  srand(1234);
  for (size_t i = 0; i < aN; ++i)
    hA[i] = __float2bfloat16(1.0f + 0.5f * (float)rand() / RAND_MAX);
  for (size_t i = 0; i < bN; ++i)
    hB[i] = __float2bfloat16(0.5f + 0.5f * (float)rand() / RAND_MAX);
  CUDA_CHECK(cudaMemcpy(A, hA.data(), aN * sizeof(bf16), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), bN * sizeof(bf16), cudaMemcpyHostToDevice));

  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)__bfloat162float(hA[(size_t)r * K + k]) * __bfloat162float(hB[(size_t)k * N + c]);
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaMemcpy(hC.data(), C, cN * sizeof(float), cudaMemcpyDeviceToHost));
    double err = 0, ref = 0;
    for (int i = 0; i < 16; ++i) {
      int r = (i * 257) % M, c = (i * 131) % N;
      double e = cpu_entry(r, c);
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - e));
      ref = std::max(ref, std::fabs(e));
    }
    std::printf("  [%-9s] sampled max_abs_err = %.3e (ref~%.1f) %s\n", tag, err, ref,
                err / std::max(ref, 1.0) < 2e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tflops = to_tflops(flops, ms);
    std::printf("%-9s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of bf16 TC peak)\n", tag, ms, tflops,
                100.0 * tflops / 989.0);
  };

  dim3 block(NTHREADS);
  dim3 grid(div_up(N, BN), div_up(M, BM));
  auto want = [&](const char* name) {
    return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0;
  };
  auto run = [&](const char* name, void (*kern)(const bf16*, const bf16*, float*, int, int, int)) {
    if (!want(name)) return;
    kern<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check(name);
    double t = bench_ms([&] { kern<<<grid, block>>>(A, B, C, M, N, K); }, 5, 30);
    report(name, t);
  };

  run("wmma", gemm_wmma);
  run("wmma_pipe", gemm_wmma_pipe);
  run("mma", gemm_mma);
  run("mma_pipe", gemm_mma_pipe);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  return 0;
}
