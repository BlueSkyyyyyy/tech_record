// 08 GEMM 入门：从朴素三重循环到 shared-memory tiling。
//
//   C[M,N] = A[M,K] * B[K,N]   （行主序，fp32）
//
//   naive : 一个线程算一个 C 元素，每次乘加都从全局内存读 A、B（无复用）
//   tiled : 一个 block 算一个 TILE×TILE 的 C 分块，先把 A/B 的 TILE 块搬进
//           shared memory 复用 TILE 次，减少全局访存
//
// 运行：scripts/run.sh 08-gemm/gemm.cu [M] [N] [K]
#include "../common/cuda_utils.cuh"

#include <cmath>

template <int T>
__global__ void gemm_naive(const float* __restrict__ A, const float* __restrict__ B,
                           float* __restrict__ C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
  C[(size_t)row * N + col] = acc;
}

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
    // 协作加载一个 TILE×TILE 的 A、B 分块（全局读合并）
    As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[(size_t)row * N + col] = acc;
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

  // CPU 参考（抽样校验，避免全量 O(MNK) 太慢）
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

  // GEMM 是计算受限，主指标是 TFLOPS；不报 GB/s（同一份 A/B 会被 L1/L2 反复复用，
  // 「逻辑访存字节」远大于实际 HBM 流量，算 GB/s 没有意义）。
  auto report = [&](const char* tag, double ms) {
    double tflops = to_tflops(flops, ms);
    std::printf("%-16s %8.4f ms  %8.2f TFLOPS  (%5.1f%% of fp32 peak)\n", tag, ms, tflops,
                100.0 * tflops / 66.9);
  };

  // naive
  {
    dim3 block(16, 16);
    dim3 grid(div_up(N, 16), div_up(M, 16));
    gemm_naive<16><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("naive");
    // 全局访存：每个 C 元素读 K 个 A、K 个 B
    double t = bench_ms([&] { gemm_naive<16><<<grid, block>>>(A, B, C, M, N, K); }, 3, 10);
    report("naive", t);
  }

  // tiled 16x16
  {
    dim3 block(16, 16);
    dim3 grid(div_up(N, 16), div_up(M, 16));
    gemm_tiled<16><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("tiled16");
    // 全局访存：A、B 各被读取 N/TILE 和 M/TILE 次左右
    double t = bench_ms([&] { gemm_tiled<16><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("tiled 16", t);
  }

  // tiled 32x32
  {
    dim3 block(32, 32);
    dim3 grid(div_up(N, 32), div_up(M, 32));
    gemm_tiled<32><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK_LAST();
    check("tiled32");
    double t = bench_ms([&] { gemm_tiled<32><<<grid, block>>>(A, B, C, M, N, K); }, 3, 20);
    report("tiled 32", t);
  }

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C));
  std::free(hA);
  std::free(hB);
  std::free(hC);
  return 0;
}
