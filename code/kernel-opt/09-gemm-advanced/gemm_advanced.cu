// 09 GEMM 进阶：寄存器分块（register tiling） + 向量化 + double buffering。
//
//   C[M,N] = A[M,K] * B[K,N]   （行主序，fp32）
//
// 版本：
//   tiled32  : 第 08 篇的 shared-memory tiling 基线（每线程 1 个输出）
//   reg      : 寄存器分块，每线程算 TM×TN = 8×8 个输出，smem 载入被 8 复用
//   reg_vec  : reg + 全局/共享内存都用 float4 向量化载入
//   reg_db   : reg_vec + cp.async 双缓冲流水线（边算当前分块边预取下一块）
//
// 运行：scripts/run.sh 09-gemm-advanced/gemm_advanced.cu [M] [N] [K]
#include "../common/cuda_utils.cuh"

#include <cuda_pipeline.h>

#include <cmath>

// ---------------------------------------------------------------------------
// 基线：第 08 篇的 smem tiling（每线程一个输出）
// ---------------------------------------------------------------------------
template <int T>
__global__ void gemm_tiled(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty;
  const int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// ---------------------------------------------------------------------------
// 寄存器分块参数
//
//   block 算 C 的 BM×BN 分块；256 线程（16×16）；
//   每线程算 TM×TN = 8×8 个输出，用「交错映射」避免 smem bank conflict：
//     线程 (tx,ty) 负责行 {ty + 16*i}、列 {tx + 16*j}
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8, TN = 8;
constexpr int THREADS = 256;  // 16 x 16
constexpr int TX = 16, TY = 16;

// 标量载入：把 A 的 BM×BK、B 的 BK×BN 分块协作搬进 smem
__device__ __forceinline__ void load_tile_scalar(const float* __restrict__ A,
                                                 const float* __restrict__ B, float (*As)[BK],
                                                 float (*Bs)[BN], int M, int N, int K,
                                                 int block_row, int block_col, int k0) {
  const int t = threadIdx.y * TX + threadIdx.x;
  for (int idx = t; idx < BM * BK; idx += THREADS) {
    const int row = idx / BK, col = idx % BK;
    const int gr = block_row + row, gc = k0 + col;
    As[row][col] = (gr < M && gc < K) ? A[(size_t)gr * K + gc] : 0.f;
  }
  for (int idx = t; idx < BK * BN; idx += THREADS) {
    const int row = idx / BN, col = idx % BN;
    const int gr = k0 + row, gc = block_col + col;
    Bs[row][col] = (gr < K && gc < N) ? B[(size_t)gr * N + gc] : 0.f;
  }
}

// 向量化载入：每个线程搬一个 float4（A、B 分块各 1024 个 float = 256 个 float4）
__device__ __forceinline__ void load_tile_vec(const float* __restrict__ A,
                                              const float* __restrict__ B, float (*As)[BK],
                                              float (*Bs)[BN], int M, int N, int K, int block_row,
                                              int block_col, int k0) {
  const int t = threadIdx.y * TX + threadIdx.x;
  {
    const int row = t / (BK / 4), q = t % (BK / 4);
    const int gr = block_row + row, gc = k0 + q * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gr < M && gc + 3 < K) v = *reinterpret_cast<const float4*>(&A[(size_t)gr * K + gc]);
    *reinterpret_cast<float4*>(&As[row][q * 4]) = v;
  }
  {
    const int row = t / (BN / 4), q = t % (BN / 4);
    const int gr = k0 + row, gc = block_col + q * 4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (gr < K && gc + 3 < N) v = *reinterpret_cast<const float4*>(&B[(size_t)gr * N + gc]);
    *reinterpret_cast<float4*>(&Bs[row][q * 4]) = v;
  }
}

// cp.async 预取：发起异步拷贝后 commit 成一个 group（不阻塞）
__device__ __forceinline__ void prefetch_tile(const float* __restrict__ A,
                                              const float* __restrict__ B, float (*As)[BK],
                                              float (*Bs)[BN], int M, int N, int K, int block_row,
                                              int block_col, int k0) {
  const int t = threadIdx.y * TX + threadIdx.x;
  {
    const int row = t / (BK / 4), q = t % (BK / 4);
    const int gr = block_row + row, gc = k0 + q * 4;
    if (gr < M && gc + 3 < K) {
      __pipeline_memcpy_async(&As[row][q * 4], &A[(size_t)gr * K + gc], 16);
    } else {
      *reinterpret_cast<float4*>(&As[row][q * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  {
    const int row = t / (BN / 4), q = t % (BN / 4);
    const int gr = k0 + row, gc = block_col + q * 4;
    if (gr < K && gc + 3 < N) {
      __pipeline_memcpy_async(&Bs[row][q * 4], &B[(size_t)gr * N + gc], 16);
    } else {
      *reinterpret_cast<float4*>(&Bs[row][q * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
    }
  }
  __pipeline_commit();
}

// 内层：从 smem 读 8+8 个值，做 64 次 FMA（一次载入被 8 次复用）
__device__ __forceinline__ void compute_tile(float (*As)[BK], float (*Bs)[BN], float acc[TM][TN]) {
  const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
  for (int k = 0; k < BK; ++k) {
    float a[TM], b[TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = As[ty + TY * i][k];
#pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + TX * j];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];
  }
}

__device__ __forceinline__ void store_tile(float* __restrict__ C, const float acc[TM][TN], int M,
                                           int N, int block_row, int block_col) {
  const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int r = block_row + ty + TY * i;
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = block_col + tx + TX * j;
      if (r < M && c < N) C[(size_t)r * N + c] = acc[i][j];
    }
  }
}

__global__ void gemm_reg(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_scalar(A, B, As, Bs, M, N, K, block_row, block_col, k0);
    __syncthreads();
    compute_tile(As, Bs, acc);
    __syncthreads();
  }
  store_tile(C, acc, M, N, block_row, block_col);
}

__global__ void gemm_reg_vec(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_vec(A, B, As, Bs, M, N, K, block_row, block_col, k0);
    __syncthreads();
    compute_tile(As, Bs, acc);
    __syncthreads();
  }
  store_tile(C, acc, M, N, block_row, block_col);
}

__global__ void gemm_reg_db(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[2][BM][BK];
  __shared__ float Bs[2][BK][BN];
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  float acc[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

  prefetch_tile(A, B, As[0], Bs[0], M, N, K, block_row, block_col, 0);
  int buf = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const bool more = (k0 + BK < K);
    if (more) prefetch_tile(A, B, As[buf ^ 1], Bs[buf ^ 1], M, N, K, block_row, block_col, k0 + BK);
    __pipeline_wait_prior(more ? 1 : 0);  // 等当前缓冲到齐（保留最后一组在飞）
    __syncthreads();
    compute_tile(As[buf], Bs[buf], acc);
    __syncthreads();
    buf ^= 1;
  }
  store_tile(C, acc, M, N, block_row, block_col);
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
  int K = (argc > 3) ? std::atoi(argv[3]) : 2048;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nM=N=K=%d, GEMM, FLOPs = %.2f GFLOP\n\n", M, flops / 1e9);

  float *A, *B, *C;
  CUDA_CHECK(cudaMalloc(&A, (size_t)M * K * 4));
  CUDA_CHECK(cudaMalloc(&B, (size_t)K * N * 4));
  CUDA_CHECK(cudaMalloc(&C, (size_t)M * N * 4));
  float* hA = (float*)std::malloc((size_t)M * K * 4);
  float* hB = (float*)std::malloc((size_t)K * N * 4);
  float* hC = (float*)std::malloc((size_t)M * N * 4);
  for (size_t i = 0; i < (size_t)M * K; ++i) hA[i] = 0.001f * (float)(i % 13);
  for (size_t i = 0; i < (size_t)K * N; ++i) hB[i] = 0.001f * (float)(i % 7);
  CUDA_CHECK(cudaMemcpy(A, hA, (size_t)M * K * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB, (size_t)K * N * 4, cudaMemcpyHostToDevice));

  auto cpu_entry = [&](int r, int c) {
    double s = 0;
    for (int k = 0; k < K; ++k) s += (double)hA[(size_t)r * K + k] * hB[(size_t)k * N + c];
    return s;
  };
  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaMemcpy(hC, C, (size_t)M * N * 4, cudaMemcpyDeviceToHost));
    double err = 0;
    for (int i = 0; i < 8; ++i) {
      int r = (i * 257) % M, c = (i * 131) % N;
      err = std::max(err, std::fabs((double)hC[(size_t)r * N + c] - cpu_entry(r, c)));
    }
    std::printf("  [%-10s] sampled max_abs_err = %.3e %s\n", tag, err,
                err < 1e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tflops = to_tflops(flops, ms);
    std::printf("%-12s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp32 peak)\n", tag, ms, tflops,
                100.0 * tflops / 66.9);
  };

  {
    dim3 block(32, 32);
    dim3 grid(div_up(N, 32), div_up(M, 32));
    gemm_tiled<32><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("tiled32");
    double t = bench_ms([&] { gemm_tiled<32><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("tiled32", t);
  }
  {
    dim3 block(TX, TY);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_reg<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("reg");
    double t = bench_ms([&] { gemm_reg<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("reg", t);
  }
  {
    dim3 block(TX, TY);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_reg_vec<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("reg_vec");
    double t = bench_ms([&] { gemm_reg_vec<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("reg_vec", t);
  }
  {
    dim3 block(TX, TY);
    dim3 grid(div_up(N, BN), div_up(M, BM));
    gemm_reg_db<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("reg_db");
    double t = bench_ms([&] { gemm_reg_db<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("reg_db", t);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  return 0;
}
