// 02 第一个 CUDA kernel：vector add 的三种写法。
//
//   v0  一线程一元素（最直观，但固定 grid）
//   v1  grid-stride loop（线程数可与数据量解耦，工业界常用）
//   v2  float4 向量化 + grid-stride（每个线程搬 16B，减少指令、提高访存效率）
//
// 三种写法结果必须与 CPU 参考一致，并实测带宽。
//
// 运行：scripts/run.sh 02-first-kernel/vector_add.cu [N]
#include "../common/cuda_utils.cuh"

// ------------------------------ kernels ------------------------------

__global__ void add_v0(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void add_v1(const float* __restrict__ a, const float* __restrict__ b,
                       float* __restrict__ c, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}

__global__ void add_v2(const float4* __restrict__ a, const float4* __restrict__ b,
                       float4* __restrict__ c, int n4) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 x = a[i];
    float4 y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

// ------------------------------ host ------------------------------

int main(int argc, char** argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);  // 16M floats = 64MB/数组
  n = (n / 4) * 4;
  const size_t bytes = (size_t)n * sizeof(float);

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nN = %d floats (%.1f MB/array, 3 arrays = %.1f MB)\n\n", n,
              bytes / 1e6, 3.0 * bytes / 1e6);

  float *h_a = (float*)std::malloc(bytes), *h_b = (float*)std::malloc(bytes),
        *h_c = (float*)std::malloc(bytes), *h_ref = (float*)std::malloc(bytes);
  fill_random(h_a, n);
  fill_random(h_b, n);
  for (int i = 0; i < n; ++i) h_ref[i] = h_a[i] + h_b[i];

  float *a, *b, *c;
  CUDA_CHECK(cudaMalloc(&a, bytes));
  CUDA_CHECK(cudaMalloc(&b, bytes));
  CUDA_CHECK(cudaMalloc(&c, bytes));
  CUDA_CHECK(cudaMemcpy(a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b, h_b, bytes, cudaMemcpyHostToDevice));

  const int block = 256;
  const int grid_full = div_up(n, block);
  const int grid_cap = 1024;  // 故意少于所需，给 grid-stride 版本用
  const double gbytes = 3.0 * bytes;  // 读 a、读 b、写 c

  auto check = [&](const char* name) {
    CUDA_CHECK(cudaMemcpy(h_c, c, bytes, cudaMemcpyDeviceToHost));
    double max_err = 0;
    for (int i = 0; i < n; ++i) max_err = std::max(max_err, (double)std::fabs(h_c[i] - h_ref[i]));
    std::printf("  [%s] max_err = %.3e %s\n", name, max_err, max_err < 1e-4 ? "OK" : "FAIL");
  };

  // v0：一线程一元素
  add_v0<<<grid_full, block>>>(a, b, c, n);
  CUDA_CHECK_LAST();
  check("v0");
  double t0 = bench_ms([&] { add_v0<<<grid_full, block>>>(a, b, c, n); });
  report_mem("v0  one-thread-per-elem", t0, gbytes, d);

  // v1：grid-stride，固定 1024 个 block
  add_v1<<<grid_cap, block>>>(a, b, c, n);
  CUDA_CHECK_LAST();
  check("v1");
  double t1 = bench_ms([&] { add_v1<<<grid_cap, block>>>(a, b, c, n); });
  report_mem("v1  grid-stride", t1, gbytes, d);

  // v2：float4 + grid-stride
  int n4 = n / 4;
  add_v2<<<grid_cap, block>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
  CUDA_CHECK_LAST();
  check("v2");
  double t2 = bench_ms([&] { add_v2<<<grid_cap, block>>>((const float4*)a, (const float4*)b, (float4*)c, n4); });
  report_mem("v2  float4 grid-stride", t2, gbytes, d);

  std::printf("\nspeedup v2/v0 = %.2fx, v2/v1 = %.2fx\n", t0 / t2, t1 / t2);

  // 启动配置扫描：同样计算量，block 大小 / block 数怎么影响带宽？
  std::printf("\n--- v2 float4 launch-config sweep (grid = SMs * k) ---\n");
  std::printf("%8s %8s %12s %10s\n", "block", "grid", "ms", "GB/s");
  for (int bs : {128, 256, 512, 1024}) {
    for (int k : {1, 2, 4, 8, 16}) {
      int gs = d.sms * k;
      double t = bench_ms(
          [&] { add_v2<<<gs, bs>>>((const float4*)a, (const float4*)b, (float4*)c, n4); }, 5, 30);
      std::printf("%8d %8d %12.4f %10.1f\n", bs, gs, t, to_gbps(gbytes, t));
    }
  }

  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  std::free(h_a); std::free(h_b); std::free(h_c); std::free(h_ref);
  return 0;
}
