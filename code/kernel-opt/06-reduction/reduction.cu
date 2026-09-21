// 06 归约与 warp shuffle：把一个大数组求和。
//
//   atomic_every : 每个元素都 atomicAdd（反面教材：全局原子争用）
//   smem_tree    : block 内 shared memory 树形归约，每个 block 一次 atomicAdd
//   warp_shuffle : warp 内用 __shfl_down_sync 归约，再跨 warp 用 smem
//   vec4_shuffle : warp_shuffle + float4 向量化读
//
// 运行：scripts/run.sh 06-reduction/reduction.cu [N]
#include "../common/cuda_utils.cuh"

__global__ void reduce_atomic_every(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) atomicAdd(out, in[i]);
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
  return v;
}

__global__ void reduce_smem_tree(const float* __restrict__ in, float* __restrict__ out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  float sum = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) sum += in[i];
  s[tid] = sum;
  __syncthreads();
  for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
    if (tid < step) s[tid] += s[tid + step];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, s[0]);
}

__global__ void reduce_warp_shuffle(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int tid = threadIdx.x;
  float sum = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) sum += in[i];
  sum = warp_reduce_sum(sum);
  __shared__ float wsum[32];
  const int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) wsum[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = (lane < (blockDim.x >> 5)) ? wsum[lane] : 0.f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) atomicAdd(out, sum);
  }
}

__global__ void reduce_vec4_shuffle(const float4* __restrict__ in, float* __restrict__ out, int n4) {
  const int tid = threadIdx.x;
  float sum = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n4; i += stride) {
    float4 v = in[i];
    sum += (v.x + v.y) + (v.z + v.w);
  }
  sum = warp_reduce_sum(sum);
  __shared__ float wsum[32];
  const int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) wsum[wid] = sum;
  __syncthreads();
  if (wid == 0) {
    sum = (lane < (blockDim.x >> 5)) ? wsum[lane] : 0.f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) atomicAdd(out, sum);
  }
}

int main(int argc, char** argv) {
  int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 26);  // 67M floats = 256MB
  n = (n / 4) * 4;
  const size_t bytes = (size_t)n * sizeof(float);

  DeviceInfo d = device_info(0);
  print_device_info(d);
  std::printf("\nN = %d floats (%.1f MB)\n\n", n, bytes / 1e6);

  float* in;
  CUDA_CHECK(cudaMalloc(&in, bytes));
  float* h = (float*)std::malloc(bytes);
  for (int i = 0; i < n; ++i) h[i] = 1.0f + 1e-3f * (float)(i & 7);
  CUDA_CHECK(cudaMemcpy(in, h, bytes, cudaMemcpyHostToDevice));

  double ref = 0;
  for (int i = 0; i < n; ++i) ref += h[i];

  const int block = 256;
  const int grid = d.sms * 8;
  const double gbytes = bytes;  // 只读 in

  auto reset_out = [&] {
    float* dummy;
    CUDA_CHECK(cudaMalloc(&dummy, sizeof(float)));
    CUDA_CHECK(cudaMemset(dummy, 0, sizeof(float)));
    return dummy;
  };
  auto check = [&](const char* tag, float* out) {
    float got = 0;
    CUDA_CHECK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
    double rel = std::fabs(got - ref) / ref;
    std::printf("  [%-14s] sum=%.1f ref=%.1f rel_err=%.2e %s\n", tag, got, ref, rel,
                rel < 1e-4 ? "OK" : "FAIL");
  };

  // 反面教材：每元素一次 atomicAdd
  {
    float* out = reset_out();
    reduce_atomic_every<<<grid, block>>>(in, out, n);
    CUDA_CHECK_LAST();
    check("atomic_every", out);
    double t = bench_ms([&] { reduce_atomic_every<<<grid, block>>>(in, out, n); });
    report_mem("atomic_every (bad)", t, gbytes, d);
    CUDA_CHECK(cudaFree(out));
  }

  // smem 树形归约
  {
    float* out = reset_out();
    size_t smem = block * sizeof(float);
    reduce_smem_tree<<<grid, block, smem>>>(in, out, n);
    CUDA_CHECK_LAST();
    check("smem_tree", out);
    double t = bench_ms([&] {
      CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
      reduce_smem_tree<<<grid, block, smem>>>(in, out, n);
    });
    report_mem("smem_tree", t, gbytes, d);
    CUDA_CHECK(cudaFree(out));
  }

  // warp shuffle
  {
    float* out = reset_out();
    reduce_warp_shuffle<<<grid, block>>>(in, out, n);
    CUDA_CHECK_LAST();
    check("warp_shuffle", out);
    double t = bench_ms([&] {
      CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
      reduce_warp_shuffle<<<grid, block>>>(in, out, n);
    });
    report_mem("warp_shuffle", t, gbytes, d);
    CUDA_CHECK(cudaFree(out));
  }

  // vec4 + shuffle
  {
    float* out = reset_out();
    reduce_vec4_shuffle<<<grid, block>>>((const float4*)in, out, n / 4);
    CUDA_CHECK_LAST();
    check("vec4_shuffle", out);
    double t = bench_ms([&] {
      CUDA_CHECK(cudaMemset(out, 0, sizeof(float)));
      reduce_vec4_shuffle<<<grid, block>>>((const float4*)in, out, n / 4);
    });
    report_mem("vec4_shuffle", t, gbytes, d);
    CUDA_CHECK(cudaFree(out));
  }

  CUDA_CHECK(cudaFree(in));
  std::free(h);
  return 0;
}
