// O42 冒烟：Hopper `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`
// （1D、无需 tensormap）是否可用。主 kernel 的 dK/dV 目前逐元素 `red.global.add.f32`
// （L2 扇区 ~70%）。若 1D bulk reduce 可用，可把 [rows][D] tile 先写 smem 再一次性
// 归约回 global（省掉逐元素 red 指令、天然 coalesced）。
//
// 关键：generic 写 smem 后必须 `fence.proxy.async.shared::cta` 才能被 async proxy 的
//       cp.reduce 读到（否则非法指令 / 结果错）。
// 编译：  ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" scripts/run.sh THIS
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at %d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while (0)

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define HAS_BULK_RED 1
#else
#define HAS_BULK_RED 0
#endif

__device__ __forceinline__ void bulk_reduce_add_f32(float* gdst, const float* ssrc, int nbytes) {
#if HAS_BULK_RED
  unsigned long long g = (unsigned long long)gdst;   // 64-bit global 地址（"l" 约束）
  unsigned s = (unsigned)__cvta_generic_to_shared(ssrc);
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
               ::"l"(g), "r"(s), "r"(nbytes) : "memory");
#else
  (void)gdst; (void)ssrc; (void)nbytes;
#endif
}
__device__ __forceinline__ void bulk_commit() {
#if HAS_BULK_RED
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}
template <int N>
__device__ __forceinline__ void bulk_wait() {
#if HAS_BULK_RED
  asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory");
#endif
}

template <int ROWS, int D>
__global__ void smoke_kernel(const float* __restrict__ src, float* __restrict__ dst) {
  extern __shared__ __align__(16) float s[];
  const int b = blockIdx.x;
  for (int i = threadIdx.x; i < ROWS * D; i += blockDim.x) s[i] = src[(size_t)b * ROWS * D + i];
  __syncthreads();
  if (threadIdx.x == 0) {
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    bulk_reduce_add_f32(dst, s, ROWS * D * (int)sizeof(float));   // 所有 CTA 归约进 dst[0..N)
    bulk_commit();
    bulk_wait<0>();
  }
}

int main() {
  constexpr int ROWS = 32, D = 128, NBLK = 7;
  constexpr int N = ROWS * D;
  float* h_src = (float*)malloc((size_t)NBLK * N * sizeof(float));
  float* h_dst = (float*)malloc((size_t)N * sizeof(float));
  for (int b = 0; b < NBLK; ++b)
    for (int i = 0; i < N; ++i) h_src[(size_t)b * N + i] = (float)(b + 1);
  float *d_src, *d_dst;
  CK(cudaMalloc(&d_src, (size_t)NBLK * N * sizeof(float)));
  CK(cudaMalloc(&d_dst, (size_t)N * sizeof(float)));
  CK(cudaMemcpy(d_src, h_src, (size_t)NBLK * N * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemset(d_dst, 0, (size_t)N * sizeof(float)));
  const int smem = N * (int)sizeof(float);
  CK(cudaFuncSetAttribute(smoke_kernel<ROWS, D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  smoke_kernel<ROWS, D><<<NBLK, 128, smem>>>(d_src, d_dst);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h_dst, d_dst, (size_t)N * sizeof(float), cudaMemcpyDeviceToHost));
  double sum = 0; for (int b = 0; b < NBLK; ++b) sum += (b + 1);
  double e = 0; for (int i = 0; i < N; ++i) e = fmax(e, fabs((double)h_dst[i] - sum));
  printf("1D bulk-reduce smoke: max_abs=%.3e (expect %.1f)\n", e, sum);
  printf("%s\n", (e == 0.0) ? "=== PASS ===" : "=== FAIL ===");
  return e == 0.0 ? 0 : 1;
}
