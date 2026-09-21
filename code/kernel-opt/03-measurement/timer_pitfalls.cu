// 03 正确测量：几个新手一定会踩的计时坑。
//
//   A. launch 开销：空 kernel 连发的平均「墙钟/kernel」
//   B. 冷启动：第一次调用 vs 稳态（模块加载 / 上下文 / lazy 初始化）
//   C. L2 常驻：小数组反复搬 → 有效带宽虚高；大数组才是真实 HBM 带宽
//
// 运行：scripts/run.sh 03-measurement/timer_pitfalls.cu
#include "../common/cuda_utils.cuh"

#include <chrono>

__global__ void empty_kernel() {}

__global__ void copy_v4(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) out[i] = in[i];
}

static double cpu_ms_now() {
  using namespace std::chrono;
  return duration_cast<duration<double, std::milli>>(high_resolution_clock::now().time_since_epoch())
      .count();
}

int main() {
  DeviceInfo d = device_info(0);
  print_device_info(d);

  // ---------------- A. launch 开销 ----------------
  std::printf("\n=== A. launch overhead (empty kernel) ===\n");
  const int L = 2000;
  // 先预热
  for (int i = 0; i < 50; ++i) empty_kernel<<<1, 32>>>();
  CUDA_CHECK(cudaDeviceSynchronize());

  double wall0 = cpu_ms_now();
  for (int i = 0; i < L; ++i) empty_kernel<<<1, 32>>>();
  double wall1 = cpu_ms_now();
  CUDA_CHECK(cudaDeviceSynchronize());
  double wall2 = cpu_ms_now();
  std::printf("  %d empty launches: host_issue=%.3f ms (%.3f us/launch), +sync=%.3f ms\n", L,
              wall1 - wall0, (wall1 - wall0) * 1000 / L, wall2 - wall0);
  double ev = bench_ms([&] { empty_kernel<<<1, 32>>>(); }, 100, 1000);
  std::printf("  CUDA-event avg over 1000 launches: %.3f us/launch (includes event overhead)\n",
              ev * 1000);

  // ---------------- B. 冷启动 vs 稳态 ----------------
  std::printf("\n=== B. cold start vs steady state ===\n");
  int n = 1 << 22;  // 4M floats = 16MB
  size_t bytes = (size_t)n * sizeof(float);
  float *a, *b;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMemset(a, 1, bytes));
  GpuTimer t;
  t.start();
  copy_v4<<<div_up(n / 4, 256), 256>>>((const float4*)a, (float4*)b, n / 4);
  t.stop();
  std::printf("  first (cold) launch : %.3f ms\n", t.ms());
  double steady = bench_ms([&] { copy_v4<<<div_up(n / 4, 256), 256>>>((const float4*)a, (float4*)b, n / 4); }, 20, 200);
  std::printf("  steady state        : %.3f ms  (%.1f GB/s)\n", steady,
              to_gbps(2.0 * bytes, steady));

  // ---------------- C. L2 常驻 ----------------
  std::printf("\n=== C. L2 residency (L2 = %.1f MB) ===\n", 52.4);
  std::printf("%12s %14s %12s %12s\n", "size", "working set", "ms", "GB/s");
  for (double mb : {1.0, 4.0, 16.0, 32.0, 64.0, 256.0, 1024.0}) {
    int nn = (int)(mb * 1e6 / sizeof(float));
    nn = (nn / 4) * 4;
    size_t bb = (size_t)nn * sizeof(float);
    float *x, *y;
    CUDA_CHECK(cudaMalloc(&x, bb));
    CUDA_CHECK(cudaMalloc(&y, bb));
    CUDA_CHECK(cudaMemset(x, 1, bb));
    copy_v4<<<div_up(nn / 4, 256), 256>>>((const float4*)x, (float4*)y, nn / 4);
    double ms = bench_ms([&] { copy_v4<<<div_up(nn / 4, 256), 256>>>((const float4*)x, (float4*)y, nn / 4); }, 10, 100);
    std::printf("%9.0f MB %11.1f MB %12.4f %12.1f\n", mb, 2 * bb / 1e6, ms, to_gbps(2.0 * bb, ms));
    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
  }
  std::printf("  注：working set << L2 时反复读同一份数据，带宽会远超 HBM 峰值，属正常现象。\n");

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  return 0;
}
