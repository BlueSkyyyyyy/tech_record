// 12 异步拷贝与流水线：cp.async 多级 software pipeline（producer-consumer）。
//
//   C[M,N] = A[M,K] * B[K,N]   （行主序，fp32）
//
// 版本：
//   sync     : 第 09 篇的 reg_vec 基线（载入 → __syncthreads → 计算 → __syncthreads）
//   db2      : 2 级双缓冲（等价 09 的 reg_db，边算边预取下一块）
//   pipeN    : N 级 cp.async 流水线（N = 3/4/5），异步组队列更深，覆盖访存延迟
//
// 运行：scripts/run.sh 12-async-pipeline/async_pipeline.cu [M] [N] [K]
#include "../common/cuda_utils.cuh"

#include <cuda_pipeline.h>

#include <cmath>
#include <cstring>

// ---------------------------------------------------------------------------
// 与第 09 篇相同的寄存器分块参数：block 算 128×128，256 线程，每线程 8×8 输出
// 「交错映射」：线程 (tx,ty) 负责行 {ty + 16*i}、列 {tx + 16*j}
// ---------------------------------------------------------------------------
constexpr int BM = 128, BN = 128, BK = 8;
constexpr int TM = 8, TN = 8;
constexpr int TX = 16, TY = 16;

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

// cp.async 预取一个分块到 smem，并 commit 成一个独立 group（不阻塞）。
// 越界元素本应填 0：这里对越界分支退化为「同步写零」——只发生在边界 block/尾块，
// 不影响主体流水的异步性。
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

__device__ __forceinline__ void zero_acc(float acc[TM][TN]) {
#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;
}

// ---------------------------------------------------------------------------
// 基线：同步载入 + 同步计算（09 的 reg_vec）
// ---------------------------------------------------------------------------
__global__ void gemm_sync(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  float acc[TM][TN];
  zero_acc(acc);
  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_vec(A, B, As, Bs, M, N, K, block_row, block_col, k0);
    __syncthreads();
    compute_tile(As, Bs, acc);
    __syncthreads();
  }
  store_tile(C, acc, M, N, block_row, block_col);
}

// ---------------------------------------------------------------------------
// N 级 cp.async 流水线。STAGES=2 即双缓冲；STAGES 越大，在飞的异步组越多。
//
// 主循环：先补发「STAGES-1 之后」那一块 → wait 到只剩 STAGES-1 组在飞
// （即当前要算的这块已到齐）→ 同步 → 计算。
// ---------------------------------------------------------------------------
template <int STAGES>
__global__ void gemm_pipe(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[STAGES][BM][BK];
  __shared__ float Bs[STAGES][BK][BN];
  const int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
  float acc[TM][TN];
  zero_acc(acc);

  // 序言：把前 STAGES-1 块排队（每块一个 commit group）
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s * BK < K) {
      prefetch_tile(A, B, As[s], Bs[s], M, N, K, block_row, block_col, s * BK);
    } else {
      __pipeline_commit();  // 空组，保持 group 计数整齐
    }
  }

  int stage = 0;
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int next = k0 + (STAGES - 1) * BK;
    const int nslot = (stage + STAGES - 1) % STAGES;
    if (next < K) {
      prefetch_tile(A, B, As[nslot], Bs[nslot], M, N, K, block_row, block_col, next);
    } else {
      __pipeline_commit();
    }
    __pipeline_wait_prior(STAGES - 1);  // 当前块到齐；更晚的 STAGES-1 组继续飞
    __syncthreads();
    compute_tile(As[stage], Bs[stage], acc);
    __syncthreads();
    stage = (stage + 1) % STAGES;
  }
  store_tile(C, acc, M, N, block_row, block_col);
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 2048;
  int N = (argc > 2) ? std::atoi(argv[2]) : 2048;
  int K = (argc > 3) ? std::atoi(argv[3]) : 2048;
  const char* which = (argc > 4) ? argv[4] : "all";

  DeviceInfo d = device_info(0);
  print_device_info(d);
  const double flops = 2.0 * M * N * K;
  std::printf("\nM=N=K=%d, GEMM, FLOPs = %.2f GFLOP\n\n", M, flops / 1e9);
  if (K % BK != 0) {
    std::printf("K must be a multiple of %d\n", BK);
    return 1;
  }

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
    std::printf("  [%-8s] sampled max_abs_err = %.3e %s\n", tag, err,
                err < 1e-2 ? "OK" : "FAIL");
  };
  auto report = [&](const char* tag, double ms) {
    double tflops = to_tflops(flops, ms);
    std::printf("%-8s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp32 peak)\n", tag, ms, tflops,
                100.0 * tflops / 66.9);
  };

  dim3 block(TX, TY);
  dim3 grid(div_up(N, BN), div_up(M, BM));
  auto want = [&](const char* name) { return std::strcmp(which, "all") == 0 || std::strcmp(which, name) == 0; };

  if (want("sync")) {
    gemm_sync<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("sync");
    double t = bench_ms([&] { gemm_sync<<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("sync", t);
  }

  if (want("pipe2")) {
    gemm_pipe<2><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("pipe2");
    double t2 = bench_ms([&] { gemm_pipe<2><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("pipe2", t2);
  }

  if (want("pipe3")) {
    gemm_pipe<3><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("pipe3");
    double t3 = bench_ms([&] { gemm_pipe<3><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("pipe3", t3);
  }

  if (want("pipe4")) {
    gemm_pipe<4><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("pipe4");
    double t4 = bench_ms([&] { gemm_pipe<4><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("pipe4", t4);
  }

  if (want("pipe5")) {
    gemm_pipe<5><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("pipe5");
    double t5 = bench_ms([&] { gemm_pipe<5><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("pipe5", t5);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  return 0;
}
