// 04 访存合并与向量化：同一个 copy，换个遍历顺序，带宽差一个数量级。
//
//   copy_rowwise      ：线程 idx 直接对应连续地址（合并访存 coalesced）
//   copy_colwise       ：线程 idx 按「列优先」映射到行优先存储（一个 warp 内跨行，
//                        地址相隔 N 个元素 → 严重不合并）
//   copy_vec4  ：合并 + float4 向量化（每线程 16B）
//
// 运行：scripts/run.sh 04-coalescing/coalescing.cu [M] [N]
#include "../common/cuda_utils.cuh"

// 连续地址：第 idx 个线程处理第 idx 个元素。
__global__ void copy_rowwise(const float* __restrict__ in, float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx];
}

// 列优先遍历：把线性 idx 解释成「第 col 行、第 row 列」的列主序坐标，
// 再换算回行主序偏移。相邻线程 → 同一列、相邻行 → 地址差 N。
__global__ void copy_colwise(const float* __restrict__ in, float* __restrict__ out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    int col = idx / M;   // 列
    int row = idx % M;   // 行
    int off = row * N + col;
    out[off] = in[off];
  }
}

// 合并 + float4：每线程搬 16B，指令数降到 1/4。
__global__ void copy_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) out[idx] = in[idx];
}

__global__ void fill_one(float* __restrict__ p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = 1.0f;
}

int main(int argc, char** argv) {
  int M = (argc > 1) ? std::atoi(argv[1]) : 8192;
  int N = (argc > 2) ? std::atoi(argv[2]) : 8192;
  N = (N / 4) * 4;
  const int n = M * N;
  const size_t bytes = (size_t)n * sizeof(float);

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nM=%d N=%d, matrix %.1f MB, working set (in+out) %.1f MB\n\n", M, N,
              bytes / 1e6, 2.0 * bytes / 1e6);

  float *in, *out;
  CUDA_CHECK(cudaMalloc(&in, bytes));
  CUDA_CHECK(cudaMalloc(&out, bytes));
  fill_one<<<div_up(n, 256), 256>>>(in, n);  // 全部填 1.0f
  CUDA_CHECK_LAST();

  const int block = 256;
  const double gbytes = 2.0 * bytes;  // 读 in + 写 out

  // 正确性：三种写法都应该等价于逐元素 copy（用 memset 的 1 值即可，随便校验一处）
  auto check = [&](const char* tag) {
    float h[4];
    CUDA_CHECK(cudaMemcpy(h, out, sizeof(h), cudaMemcpyDeviceToHost));
    for (float v : h)
      if (v != 1.0f) { std::printf("  [%s] FAIL value=%f\n", tag, v); return; }
    std::printf("  [%s] OK\n", tag);
  };

  copy_rowwise<<<div_up(n, block), block>>>(in, out, n);
  CUDA_CHECK_LAST();
  check("rowwise");
  double t0 = bench_ms([&] { copy_rowwise<<<div_up(n, block), block>>>(in, out, n); });
  report_mem("rowwise (coalesced)", t0, gbytes, d);

  copy_colwise<<<div_up(n, block), block>>>(in, out, M, N);
  CUDA_CHECK_LAST();
  check("colwise");
  double t1 = bench_ms([&] { copy_colwise<<<div_up(n, block), block>>>(in, out, M, N); });
  report_mem("colwise (strided)", t1, gbytes, d);

  int n4 = n / 4;
  copy_vec4<<<div_up(n4, block), block>>>((const float4*)in, (float4*)out, n4);
  CUDA_CHECK_LAST();
  check("rowwise_vec4");
  double t2 = bench_ms([&] {
    copy_vec4<<<div_up(n4, block), block>>>((const float4*)in, (float4*)out, n4);
  });
  report_mem("rowwise vec4", t2, gbytes, d);

  std::printf("\nslowdown colwise/rowwise = %.1fx   speedup vec4/rowwise = %.2fx\n", t1 / t0,
              t0 / t2);

  CUDA_CHECK(cudaFree(in));
  CUDA_CHECK(cudaFree(out));
  return 0;
}
