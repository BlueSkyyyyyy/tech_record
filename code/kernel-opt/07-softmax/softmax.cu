// 07 Softmax 优化：把一个「多趟 kernel」的 softmax 融合成单 kernel。
//
// softmax(x)_j = exp(x_j - max(x)) / sum_j exp(x_j - max(x))
//
//   multipass    ：3 个 kernel（求 max / 求 exp 和 / 归一化），数据来回搬 3 遍
//   fused        ：一个 kernel 处理一行，block 内归约，输入读 2 遍（第 2 遍多半命中 L2）
//   fused_cache  ：把一行缓存进 shared memory，输入只读 1 遍
//
// 运行：scripts/run.sh 07-softmax/softmax.cu [rows] [cols]
#include "../common/cuda_utils.cuh"

#include <cfloat>
#include <cmath>

// ------------------------- block 级归约工具 -------------------------
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}
__device__ __forceinline__ float block_max(float v, float* sm) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_max(v);
  if (lane == 0) sm[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (blockDim.x >> 5)) ? sm[lane] : -FLT_MAX;
    v = warp_max(v);
    if (lane == 0) sm[0] = v;
  }
  __syncthreads();
  return sm[0];
}
__device__ __forceinline__ float block_sum(float v, float* sm) {
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  v = warp_sum(v);
  if (lane == 0) sm[wid] = v;
  __syncthreads();
  if (wid == 0) {
    v = (lane < (blockDim.x >> 5)) ? sm[lane] : 0.f;
    v = warp_sum(v);
    if (lane == 0) sm[0] = v;
  }
  __syncthreads();
  return sm[0];
}

// ------------------------- multipass 三个 kernel -------------------------
__global__ void k_row_max(const float* __restrict__ x, float* __restrict__ out, int rows, int cols) {
  __shared__ float sm[32];
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, p[c]);
  m = block_max(m, sm);
  if (threadIdx.x == 0) out[row] = m;
}
__global__ void k_row_expsum(const float* __restrict__ x, const float* __restrict__ m,
                             float* __restrict__ out, int rows, int cols) {
  __shared__ float sm[32];
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  const float mx = m[row];
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __expf(p[c] - mx);
  s = block_sum(s, sm);
  if (threadIdx.x == 0) out[row] = s;
}
__global__ void k_row_norm(const float* __restrict__ x, const float* __restrict__ m,
                           const float* __restrict__ s, float* __restrict__ y, int rows,
                           int cols) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)rows * cols) {
    const int row = i / cols;
    y[i] = __expf(x[i] - m[row]) / s[row];
  }
}

// ------------------------- fused：block per row -------------------------
__global__ void softmax_fused(const float* __restrict__ x, float* __restrict__ y, int rows,
                              int cols) {
  __shared__ float sm[32];
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  float* q = y + (size_t)row * cols;
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, p[c]);
  m = block_max(m, sm);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __expf(p[c] - m);
  s = block_sum(s, sm);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) q[c] = __expf(p[c] - m) * inv;
}

// ------------------------- fused + shared memory 缓存 -------------------------
template <int BLOCK>
__global__ void softmax_smem(const float* __restrict__ x, float* __restrict__ y, int rows,
                                    int cols) {
  extern __shared__ float sx[];  // cols floats
  __shared__ float sm[32];
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  float* q = y + (size_t)row * cols;
  for (int c = threadIdx.x; c < cols; c += BLOCK) sx[c] = p[c];  // 读入 smem，只读一遍
  __syncthreads();
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += BLOCK) m = fmaxf(m, sx[c]);
  m = block_max(m, sm);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += BLOCK) s += __expf(sx[c] - m);
  s = block_sum(s, sm);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += BLOCK) q[c] = __expf(sx[c] - m) * inv;
}

int main(int argc, char** argv) {
  int rows = (argc > 1) ? std::atoi(argv[1]) : 4096;
  int cols = (argc > 2) ? std::atoi(argv[2]) : 8192;
  const size_t n = (size_t)rows * cols;
  const size_t bytes = n * sizeof(float);
  const int block = 256;

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nrows=%d cols=%d, matrix %.1f MB, working set (x+y) %.1f MB\n\n", rows, cols,
              bytes / 1e6, 2.0 * bytes / 1e6);

  float *x, *y, *maxes, *sums;
  CUDA_CHECK(cudaMalloc(&x, bytes));
  CUDA_CHECK(cudaMalloc(&y, bytes));
  CUDA_CHECK(cudaMalloc(&maxes, rows * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&sums, rows * sizeof(float)));
  float* h = (float*)std::malloc(bytes);
  for (size_t i = 0; i < n; ++i) h[i] = 0.5f * sinf((float)(i % 1000)) + 0.01f * (float)(i % 7);
  CUDA_CHECK(cudaMemcpy(x, h, bytes, cudaMemcpyHostToDevice));

  // CPU 参考（双精度）
  float* ref = (float*)std::malloc(bytes);
  for (int r = 0; r < rows; ++r) {
    double mx = -1e300, s = 0;
    for (int c = 0; c < cols; ++c) mx = std::fmax(mx, h[(size_t)r * cols + c]);
    for (int c = 0; c < cols; ++c) s += std::exp((double)h[(size_t)r * cols + c] - mx);
    for (int c = 0; c < cols; ++c)
      ref[(size_t)r * cols + c] = (float)(std::exp((double)h[(size_t)r * cols + c] - mx) / s);
  }
  float* hout = (float*)std::malloc(bytes);

  auto check = [&](const char* tag) {
    CUDA_CHECK(cudaMemcpy(hout, y, bytes, cudaMemcpyDeviceToHost));
    double err = 0;
    for (size_t i = 0; i < n; i += 1009) err = std::max(err, (double)std::fabs(hout[i] - ref[i]));
    std::printf("  [%-14s] max_abs_err = %.2e %s\n", tag, err, err < 1e-3 ? "OK" : "FAIL");
  };

  // 每次必须读 x、写 y，共 2 * bytes
  const double gbytes = 2.0 * bytes;

  const int grid_norm = (int)((n + block - 1) / block);
  auto run_multipass = [&] {
    k_row_max<<<rows, block>>>(x, maxes, rows, cols);
    k_row_expsum<<<rows, block>>>(x, maxes, sums, rows, cols);
    k_row_norm<<<grid_norm, block>>>(x, maxes, sums, y, rows, cols);
  };

  run_multipass();
  CUDA_CHECK_LAST();
  check("multipass");
  double t0 = bench_ms(run_multipass);
  report_mem("multipass (3 kernels)", t0, gbytes, d);

  softmax_fused<<<rows, block>>>(x, y, rows, cols);
  CUDA_CHECK_LAST();
  check("fused");
  double t1 = bench_ms([&] { softmax_fused<<<rows, block>>>(x, y, rows, cols); });
  report_mem("fused (1 block/row)", t1, gbytes, d);

  size_t smem = (size_t)cols * sizeof(float);
  softmax_smem<256><<<rows, 256, smem>>>(x, y, rows, cols);
  CUDA_CHECK_LAST();
  check("fused_cache");
  double t2 = bench_ms(
      [&] { softmax_smem<256><<<rows, 256, smem>>>(x, y, rows, cols); });
  report_mem("fused + smem cache", t2, gbytes, d);

  std::printf("\nspeedup fused/multipass = %.2fx, cache/fused = %.2fx\n", t0 / t1, t1 / t2);

  CUDA_CHECK(cudaFree(x));
  CUDA_CHECK(cudaFree(y));
  CUDA_CHECK(cudaFree(maxes));
  CUDA_CHECK(cudaFree(sums));
  std::free(h);
  std::free(ref);
  std::free(hout);
  return 0;
}
