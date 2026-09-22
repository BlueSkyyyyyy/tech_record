// 最小复现：验证手写 SW128 布局与 TMA 的 SW128 布局是否逐字节一致（fp8）。
// 运行：ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
//         scripts/run.sh 57-v4-fp4-moe-fold/swz_test.cu
#include "../common/cuda_utils.cuh"
#include <cuda.h>

constexpr int M = 128, K = 128;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__host__ __device__ __forceinline__ int sw128_off(int r, int k, int Kd) {
  (void)Kd;
  const int rr = r & 7;
  return (r >> 3) * 1024 + (rr * 8 + (((k >> 4) & 7) ^ rr)) * 16 + (k & 15);
}
__device__ __forceinline__ void tma_load_2d(const CUtensorMap* tmap, void* dst, int c0, int c1,
                                            uint64_t* bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%3, %4}], [%2];" ::"r"(smem_u32(dst)),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
      : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra W;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}

__global__ void test_kernel(const __grid_constant__ CUtensorMap tm, const uint8_t* __restrict__ in,
                            int* __restrict__ bad, uint8_t* __restrict__ dump_tma,
                            uint8_t* __restrict__ dump_man) {
  __shared__ __align__(1024) char smem[1024 + M * K];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
  uint8_t* Bs = reinterpret_cast<uint8_t*>(smem + 1024);
  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_arrive_expect_tx(bar, M * K);
    tma_load_2d(&tm, Bs, 0, 0, bar);
  }
  __syncthreads();
  mbar_wait(bar, 0);
  __syncthreads();
  // 手动布局到另一个 buffer
  extern __shared__ __align__(1024) char sm2[];
  uint8_t* man = reinterpret_cast<uint8_t*>(sm2);
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
    const int r = i / K, k = i % K;
    man[sw128_off(r, k, K)] = in[(size_t)r * K + k];
  }
  __syncthreads();
  for (int i = threadIdx.x; i < M * K; i += blockDim.x) {
    if (Bs[i] != man[i]) atomicAdd(bad, 1);
    dump_tma[i] = Bs[i];
    dump_man[i] = man[i];
  }
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  DeviceInfo d = device_info(0);
  print_device_info(d);
  uint8_t* in; int* bad; uint8_t *dt, *dm;
  CUDA_CHECK(cudaMalloc(&in, M * K));
  CUDA_CHECK(cudaMalloc(&bad, 4));
  CUDA_CHECK(cudaMalloc(&dt, M * K));
  CUDA_CHECK(cudaMalloc(&dm, M * K));
  std::vector<uint8_t> h(M * K);
  for (int i = 0; i < M * K; ++i) h[i] = (uint8_t)(i * 31 + 7);
  CUDA_CHECK(cudaMemcpy(in, h.data(), M * K, cudaMemcpyHostToDevice));
  CUtensorMap tm;
  cuuint64_t dims[2] = {K, M}; cuuint64_t strides[1] = {K};
  cuuint32_t box[2] = {K, M}; cuuint32_t es[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, in, dims, strides, box,
                                      es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) { printf("tmap fail\n"); return 1; }
  CUDA_CHECK(cudaMemset(bad, 0, 4));
  test_kernel<<<1, 256, M * K>>>(tm, in, bad, dt, dm);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaDeviceSynchronize());
  int hb = -1; CUDA_CHECK(cudaMemcpy(&hb, bad, 4, cudaMemcpyDeviceToHost));
  std::vector<uint8_t> ht(M * K), hm(M * K);
  CUDA_CHECK(cudaMemcpy(ht.data(), dt, M * K, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hm.data(), dm, M * K, cudaMemcpyDeviceToHost));
  printf("TMA vs manual SW128: mismatched bytes = %d  %s\n", hb, hb == 0 ? "OK" : "FAIL");
  for (int i = 0; i < 40 && hb != 0; ++i)
    if (ht[i] != hm[i]) printf("  off %d: tma=%u man=%u\n", i, ht[i], hm[i]);
  // 打印若干位置的映射，帮助推导
  if (hb != 0) {
    printf("查找每行第一个元素的物理位置（manual）:\n");
    for (int rr = 0; rr < 8; ++rr)
      for (int c = 0; c < 2; ++c)
        printf("  (r=%d,k=%d) -> off %d (tma byte %u man byte %u)\n", rr, c * 16,
               sw128_off(rr, c * 16, K), ht[sw128_off(rr, c * 16, K)], hm[sw128_off(rr, c * 16, K)]);
  }
  return 0;
}
