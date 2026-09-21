// 最小复现：逐级验证 producer/consumer mbarrier + named bar.sync
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
  asm volatile(
      "{\n.reg .pred p;\nLAB_WAIT%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
      "@!p bra LAB_WAIT%=;\n}\n" ::"r"(smem_u32(bar)),
      "r"(phase));
}
__device__ __forceinline__ void named_bar_sync(int id, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(count) : "memory");
}

// MODE 0: 只 full 单向
// MODE 1: full + empty
// MODE 2: full + empty + named bar
template <int STAGES, int NCONS, int MODE>
__global__ void __launch_bounds__((1 + NCONS) * 128) test_kernel(int nblk) {
  constexpr int T = (1 + NCONS) * 128;
  extern __shared__ char smem[];
  uint64_t* full = reinterpret_cast<uint64_t*>(smem);
  uint64_t* empty = full + STAGES;
  int tid = threadIdx.x;
  for (int s = tid; s < STAGES; s += T) {
    mbar_init(full + s, 1);
    mbar_init(empty + s, NCONS);
  }
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  __syncthreads();

  if (tid < 128) {
    for (int t = 0; t < nblk; ++t) {
      int tn = t + 1;
      if (MODE >= 1 && tn < nblk && tn >= STAGES)
        mbar_wait(empty + tn % STAGES, (uint32_t)((tn / STAGES - 1) & 1));
      if (MODE >= 2) named_bar_sync(1, 128);
      if (tid == 0) mbar_arrive(full + t % STAGES);
    }
  } else {
    int cg = ((tid >> 5) - 4) >> 2;
    for (int t = 0; t < nblk; ++t) {
      mbar_wait(full + t % STAGES, (uint32_t)((t / STAGES) & 1));
      if (MODE >= 2) named_bar_sync(2 + cg, 128);
      if (MODE >= 1 && (tid & 31) == 0) mbar_arrive(empty + t % STAGES);
    }
  }
}

template <int MODE>
void run(const char* tag) {
  constexpr int STAGES = 3, NCONS = 1;
  size_t shm = 2 * STAGES * 8;
  cudaFuncSetAttribute(test_kernel<STAGES, NCONS, MODE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, shm);
  printf("%s: launching...\n", tag);
  test_kernel<STAGES, NCONS, MODE><<<1, (1 + NCONS) * 128, shm>>>(40);
  cudaError_t e = cudaDeviceSynchronize();
  printf("%s: err=%s\n", tag, cudaGetErrorString(e));
}

int main() {
  setvbuf(stdout, nullptr, _IONBF, 0);
  run<0>("MODE0");
  run<1>("MODE1");
  run<2>("MODE2");
  return 0;
}
