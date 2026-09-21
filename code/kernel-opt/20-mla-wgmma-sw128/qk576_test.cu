// SW128 QK^T 定点测试：A[64][576] @ B[64][576]^T -> C[64][64]，wgmma m64n64k16 x36 步。
// 用来隔离 MLA 中 K=576 多 atom 的 QK 部分（冒烟 GEMM 只覆盖 K=64 单 atom）。
// 运行：ARCH=sm_90a scripts/run.sh 20-mla-wgmma-sw128/qk576_test.cu
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
constexpr int M = 64, N = 64, KK = 576;
constexpr int SBO = (KK / 64) * 1024;  // 9216
constexpr int T = 256;

__global__ void qk_kernel(const bf16* __restrict__ A, const bf16* __restrict__ B,
                          float* __restrict__ C) {
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;                  // [64][576]
  char* Bs = smem + M * KK * 2;     // [64][576]
  const int tid = threadIdx.x, lane = tid & 31, W = tid >> 5;
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
  for (int idx = tid; idx < M * (KK / 8); idx += T) {
    const int row = idx / (KK / 8), cu = idx % (KK / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&A[row * KK + cu * 8]);
    sw128_store16(As, row, cu * 8, KK, v);
  }
  for (int idx = tid; idx < N * (KK / 8); idx += T) {
    const int row = idx / (KK / 8), cu = idx % (KK / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&B[row * KK + cu * 8]);
    sw128_store16(Bs, row, cu * 8, KK, v);
  }
  __syncthreads();
  float S[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) S[i] = 0.f;
  const uint32_t as_a = smem_u32(As), bs_a = smem_u32(Bs);
  wgmma_fence();
#pragma unroll
  for (int s = 0; s < KK / 16; ++s)
    wgmma_m64n64k16(S, make_desc_sw128(sw128_k16_addr(as_a, s), SBO),
                    make_desc_sw128(sw128_k16_addr(bs_a, s), SBO));
  wgmma_commit();
  wgmma_wait0();
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int col = t * 8 + (lane & 3) * 2;
    C[r0 * N + col] = S[t * 4 + 0];
    C[r0 * N + col + 1] = S[t * 4 + 1];
    C[r1 * N + col] = S[t * 4 + 2];
    C[r1 * N + col + 1] = S[t * 4 + 3];
  }
}

int main() {
  std::vector<bf16> hA(M * KK), hB(N * KK);
  srand(3);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& x : hA) x = rnd();
  for (auto& x : hB) x = rnd();
  bf16 *A, *B;
  float* C;
  CUDA_CHECK(cudaMalloc(&A, M * KK * 2));
  CUDA_CHECK(cudaMalloc(&B, N * KK * 2));
  CUDA_CHECK(cudaMalloc(&C, M * N * 4));
  CUDA_CHECK(cudaMemcpy(A, hA.data(), M * KK * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(B, hB.data(), N * KK * 2, cudaMemcpyHostToDevice));
  const size_t shm = 2 * M * KK * 2;
  CUDA_CHECK(cudaFuncSetAttribute(qk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  qk_kernel<<<1, T, shm>>>(A, B, C);
  CUDA_CHECK_LAST();
  std::vector<float> hC(M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), C, M * N * 4, cudaMemcpyDeviceToHost));
  double err = 0, ref = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double acc = 0;
      for (int k = 0; k < KK; ++k)
        acc += (double)__bfloat162float(hA[m * KK + k]) * (double)__bfloat162float(hB[n * KK + k]);
      err = std::max(err, std::fabs(hC[m * N + n] - acc));
      ref = std::max(ref, std::fabs(acc));
    }
  std::printf("QK576 max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  return err / std::max(ref, 1e-6) < 5e-2 ? 0 : 1;
}
