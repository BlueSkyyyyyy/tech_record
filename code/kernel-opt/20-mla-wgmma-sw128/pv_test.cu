// SW128 PV 定点测试：P[64][64] @ Vt[512][64]^T -> O[64][512]，wgmma m64n256k16 x2。
// 运行：ARCH=sm_90a scripts/run.sh 20-mla-wgmma-sw128/pv_test.cu
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"
#include <cuda_bf16.h>
using bf16 = __nv_bfloat16;
constexpr int BM = 64, KT = 64, DV = 512, NWG = 2, DVW = DV / NWG, T = NWG * 128;
constexpr int SBO = 1024;

__global__ void pv_kernel(const bf16* __restrict__ P, const bf16* __restrict__ Vt,
                          float* __restrict__ O) {
  extern __shared__ __align__(1024) char smem[];
  char* ps = smem;                     // [64][64]
  char* vs = ps + BM * KT * 2;         // [512][64]
  const int tid = threadIdx.x, lane = tid & 31, wg = tid >> 7, W = tid >> 5;
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
  for (int idx = tid; idx < BM * (KT / 8); idx += T) {
    const int row = idx / (KT / 8), cu = idx % (KT / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&P[row * KT + cu * 8]);
    sw128_store16(ps, row, cu * 8, KT, v);
  }
  for (int idx = tid; idx < DV * (KT / 8); idx += T) {
    const int row = idx / (KT / 8), cu = idx % (KT / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&Vt[row * KT + cu * 8]);
    sw128_store16(vs, row, cu * 8, KT, v);
  }
  __syncthreads();
  float Oc[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) Oc[i] = 0.f;
  const uint32_t ps_a = smem_u32(ps), vs_a = smem_u32(vs) + (uint32_t)(wg * (DVW / 8) * 1024);
  wgmma_fence();
#pragma unroll
  for (int c = 0; c < KT / 16; ++c)
    wgmma_m64n256k16(Oc, make_desc_sw128(sw128_k16_addr(ps_a, c), SBO),
                     make_desc_sw128(sw128_k16_addr(vs_a, c), SBO));
  wgmma_commit();
  wgmma_wait0();
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int col = wg * DVW + t * 8 + (lane & 3) * 2;
    O[r0 * DV + col] = Oc[t * 4 + 0];
    O[r0 * DV + col + 1] = Oc[t * 4 + 1];
    O[r1 * DV + col] = Oc[t * 4 + 2];
    O[r1 * DV + col + 1] = Oc[t * 4 + 3];
  }
}

int main() {
  std::vector<bf16> hP(BM * KT), hV(DV * KT);
  srand(11);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  for (auto& x : hP) x = rnd();
  for (auto& x : hV) x = rnd();
  bf16 *P, *V;
  float* O;
  CUDA_CHECK(cudaMalloc(&P, BM * KT * 2));
  CUDA_CHECK(cudaMalloc(&V, DV * KT * 2));
  CUDA_CHECK(cudaMalloc(&O, (size_t)BM * DV * 4));
  CUDA_CHECK(cudaMemcpy(P, hP.data(), BM * KT * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V, hV.data(), DV * KT * 2, cudaMemcpyHostToDevice));
  const size_t shm = BM * KT * 2 + DV * KT * 2;
  CUDA_CHECK(cudaFuncSetAttribute(pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  pv_kernel<<<1, T, shm>>>(P, V, O);
  CUDA_CHECK_LAST();
  std::vector<float> hO((size_t)BM * DV);
  CUDA_CHECK(cudaMemcpy(hO.data(), O, (size_t)BM * DV * 4, cudaMemcpyDeviceToHost));
  double err = 0, ref = 0;
  for (int m = 0; m < BM; ++m)
    for (int d = 0; d < DV; ++d) {
      double acc = 0;
      for (int k = 0; k < KT; ++k)
        acc += (double)__bfloat162float(hP[m * KT + k]) * (double)__bfloat162float(hV[d * KT + k]);
      err = std::max(err, std::fabs(hO[m * DV + d] - acc));
      ref = std::max(ref, std::fabs(acc));
    }
  std::printf("PV max_abs_err=%.3e (ref~%.3f) %s\n", err, ref,
              err / std::max(ref, 1e-6) < 5e-2 ? "OK" : "FAIL");
  return 0;
}
