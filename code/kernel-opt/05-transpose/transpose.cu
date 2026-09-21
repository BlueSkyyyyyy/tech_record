// 05 共享内存与 bank conflict：矩阵转置。
//
// 转置天然有一边访存不合并。用 shared memory 做「中转」把两边都变成合并访问，
// 再用 padding 消掉 shared memory 的 bank conflict。
//
//   naive        ：读合并、写跨行（写不合并）
//   smem_nopad   ：34/33 分块 smem 中转，但 tile[32][32] 有 bank conflict
//   smem_pad     ：tile[32][33] padding，消除 bank conflict
//
// 运行：scripts/run.sh 05-transpose/transpose.cu [M] [N]
#include "../common/cuda_utils.cuh"

constexpr int TILE = 32;

__global__ void transpose_naive(const float* __restrict__ A, float* __restrict__ B, int M, int N) {
  int x = blockIdx.x * TILE + threadIdx.x;  // A 的列 = B 的行
  int y = blockIdx.y * TILE + threadIdx.y;  // A 的行 = B 的列
  if (x < N && y < M) B[(size_t)x * M + y] = A[(size_t)y * N + x];
}

__global__ void transpose_smem_nopad(const float* __restrict__ A, float* __restrict__ B, int M, int N) {
  __shared__ float tile[TILE][TILE];  // 注意：没有 padding
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = A[(size_t)y * N + x];
  __syncthreads();
  // B 的行 = A 的列 = blockIdx.x*TILE + threadIdx.y
  int x2 = blockIdx.x * TILE + threadIdx.y;
  int y2 = blockIdx.y * TILE + threadIdx.x;
  if (x2 < N && y2 < M) B[(size_t)x2 * M + y2] = tile[threadIdx.x][threadIdx.y];
}

__global__ void transpose_smem_pad(const float* __restrict__ A, float* __restrict__ B, int M, int N) {
  __shared__ float tile[TILE][TILE + 1];  // padding：每行多一个元素，错开 bank
  int x = blockIdx.x * TILE + threadIdx.x;
  int y = blockIdx.y * TILE + threadIdx.y;
  if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = A[(size_t)y * N + x];
  __syncthreads();
  int x2 = blockIdx.x * TILE + threadIdx.y;
  int y2 = blockIdx.y * TILE + threadIdx.x;
  if (x2 < N && y2 < M) B[(size_t)x2 * M + y2] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // 无缓冲，崩溃时也能看到已打印的输出
  int M = (argc > 1) ? std::atoi(argv[1]) : 8192;
  int N = (argc > 2) ? std::atoi(argv[2]) : 8192;
  const size_t n = (size_t)M * N;
  const size_t bytes = n * sizeof(float);

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nM=%d N=%d, matrix %.1f MB, working set (A+B) %.1f MB\n\n", M, N,
              bytes / 1e6, 2.0 * bytes / 1e6);

  float *A, *B;
  CUDA_CHECK(cudaMalloc(&A, bytes));
  CUDA_CHECK(cudaMalloc(&B, bytes));
  float* hA = (float*)std::malloc(bytes);
  for (size_t i = 0; i < n; ++i) hA[i] = (float)(i % 1000);
  CUDA_CHECK(cudaMemcpy(A, hA, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid(div_up(N, TILE), div_up(M, TILE));
  const double gbytes = 2.0 * bytes;

  // 参考转置（CPU）
  float* hB = (float*)std::malloc(bytes);
  float* hOut = (float*)std::malloc(bytes);
  for (int y = 0; y < M; ++y)
    for (int x = 0; x < N; ++x) hB[(size_t)x * M + y] = hA[(size_t)y * N + x];

  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaMemcpy(hOut, B, bytes, cudaMemcpyDeviceToHost));
    double err = 0;
    for (size_t i = 0; i < n; i += 997) err = std::max(err, (double)std::fabs(hOut[i] - hB[i]));
    std::printf("  [%s] max_err = %.3e %s\n", tag, err, err == 0 ? "OK" : "FAIL");
  };

  transpose_naive<<<grid, block>>>(A, B, M, N);
  CUDA_CHECK_LAST();
  check("naive");
  double t0 = bench_ms([&] { transpose_naive<<<grid, block>>>(A, B, M, N); });
  report_mem("naive", t0, gbytes, d);

  transpose_smem_nopad<<<grid, block>>>(A, B, M, N);
  CUDA_CHECK_LAST();
  check("smem_nopad");
  double t1 = bench_ms([&] { transpose_smem_nopad<<<grid, block>>>(A, B, M, N); });
  report_mem("smem no-pad", t1, gbytes, d);

  transpose_smem_pad<<<grid, block>>>(A, B, M, N);
  CUDA_CHECK_LAST();
  check("smem_pad");
  double t2 = bench_ms([&] { transpose_smem_pad<<<grid, block>>>(A, B, M, N); });
  report_mem("smem padded", t2, gbytes, d);

  std::printf("\nspeedup smem_pad/naive = %.2fx, smem_pad/smem_nopad = %.2fx\n", t0 / t2, t1 / t2);

  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));

  std::free(hA);
  std::free(hB);
  std::free(hOut);
  return 0;
}
