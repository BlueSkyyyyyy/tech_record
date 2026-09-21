// 参数化 MN-major 描述符探测：D[64][128]=A[64][64]@B[64][128]，B 为 MN-major SW128。
// 用法：./mn_test2.out <lbo_bytes> <sbo_bytes> <step_bytes>
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"
#include "./mn128.cuh"

#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;

constexpr int BM = 64, BK = 64, BN = 128, T = 128;

__device__ __forceinline__ uint64_t make_desc_mn_rt(uint32_t addr, uint32_t lbo_bytes,
                                                    uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_bytes >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}

__global__ void __launch_bounds__(T) mn_test2_kernel(const bf16* A, const bf16* B, float* D,
                                                     uint32_t lbo, uint32_t sbo, uint32_t step) {
  __shared__ __align__(1024) char as[BM * BK * 2];
  __shared__ __align__(1024) char bs[BK * BN * 2];
  const int tid = threadIdx.x, lane = tid & 31, W = tid >> 5;
  for (int idx = tid; idx < BM * (BK / 8); idx += T) {
    const int m = idx / (BK / 8), cu = idx % (BK / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&A[m * BK + cu * 8]);
    sw128_store16(as, m, cu * 8, BK, v);
  }
  for (int idx = tid; idx < BK * (BN / 8); idx += T) {
    const int kt = idx / (BN / 8), nu = idx % (BN / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&B[kt * BN + nu * 8]);
    mn128_store16(bs, kt, nu * 8, BN, v);
  }
  __syncthreads();

  const uint32_t aa = smem_u32(as), ba = smem_u32(bs);
  float d0[32] = {0}, d1[32] = {0};
  wgmma_fence();
#pragma unroll
  for (int s = 0; s < BK / 16; ++s) {
    const uint64_t da = make_desc_sw128(sw128_k16_addr(aa, s), (BK / 64) * 1024);
    wgmma_m64n64k16(d0, da, make_desc_mn_rt(ba + s * step, lbo, sbo));
    wgmma_m64n64k16(d1, da, make_desc_mn_rt(ba + 1024 + s * step, lbo, sbo));
  }
  wgmma_commit();
  wgmma_wait0();

  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
#pragma unroll
  for (int t = 0; t < 8; ++t) {
    const int col = t * 8 + (lane & 3) * 2;
    D[r0 * BN + col] = d0[t * 4 + 0];
    D[r0 * BN + col + 1] = d0[t * 4 + 1];
    D[r1 * BN + col] = d0[t * 4 + 2];
    D[r1 * BN + col + 1] = d0[t * 4 + 3];
    D[r0 * BN + 64 + col] = d1[t * 4 + 0];
    D[r0 * BN + 64 + col + 1] = d1[t * 4 + 1];
    D[r1 * BN + 64 + col] = d1[t * 4 + 2];
    D[r1 * BN + 64 + col + 1] = d1[t * 4 + 3];
  }
}

int main(int argc, char** argv) {
  uint32_t lbo = (argc > 1) ? atoi(argv[1]) : 1024;
  uint32_t sbo = (argc > 2) ? atoi(argv[2]) : 2048;
  uint32_t step = (argc > 3) ? atoi(argv[3]) : 4096;
  std::vector<bf16> hA(BM * BK), hB(BK * BN);
  std::vector<float> hD(BM * BN, 0.f);
  srand(7);
  auto rnd = [] { return __float2bfloat16(0.1f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& x : hA) x = rnd();
  for (auto& x : hB) x = rnd();
  bf16 *dA, *dB; float* dD;
  CUDA_CHECK(cudaMalloc(&dA, hA.size() * 2));
  CUDA_CHECK(cudaMalloc(&dB, hB.size() * 2));
  CUDA_CHECK(cudaMalloc(&dD, hD.size() * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice));
  mn_test2_kernel<<<1, T>>>(dA, dB, dD, lbo, sbo, step);
  CUDA_CHECK_LAST();
  CUDA_CHECK(cudaMemcpy(hD.data(), dD, hD.size() * 4, cudaMemcpyDeviceToHost));
  double err = 0, ref = 0;
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < BN; ++n) {
      double acc = 0;
      for (int k = 0; k < BK; ++k)
        acc += (double)__bfloat162float(hA[m * BK + k]) * (double)__bfloat162float(hB[k * BN + n]);
      err = std::max(err, std::fabs((double)hD[m * BN + n] - acc));
      ref = std::max(ref, std::fabs(acc));
    }
  std::printf("lbo=%u sbo=%u step=%u -> max_abs_err=%.3e (ref~%.3f) %s\n", lbo, sbo, step, err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  cudaFree(dA); cudaFree(dB); cudaFree(dD);
  return 0;
}
