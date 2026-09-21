// 最小复现：用 mma.m16n8k16 + ldmatrix.x4.trans 从 **SW128 布局**的 B tile 读 V，
// 验证 PV 路径。A=P[64][64] row-major（padded），B=V[64][512] 以 SW128（K=512）存。
#include "../common/cuda_utils.cuh"
#include "../20-mla-wgmma-sw128/wgmma_sw128.cuh"
#include <cuda_bf16.h>
#include <random>
#include <vector>
using bf16 = __nv_bfloat16;
constexpr int BM = 64, KT = 64, DV = 512, KTP = KT + 8;
constexpr int T = 256;

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(T) pv_kernel(const bf16* P, const bf16* V, float* C) {
  extern __shared__ __align__(1024) char smem[];
  bf16 (*ps)[KTP] = reinterpret_cast<bf16(*)[KTP]>(smem);
  char* vs = smem + BM * KTP * 2;                       // [KT][DV] SW128
  const int tid = threadIdx.x, lane = tid & 31, W = tid >> 5;
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
  const int rh = tid >> 7, dvw = 256;                   // 前半/后半 warpgroup 各 256 列
  for (int i = tid; i < BM * KT; i += T) ps[i / KT][i % KT] = P[i];
  for (int i = tid; i < KT * (DV / 8); i += T) {
    const int kv = i / (DV / 8), cu = i % (DV / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&V[(size_t)kv * DV + cu * 8]);
    sw128_store16(vs, kv, cu * 8, DV, v);
  }
  __syncthreads();
  float O[32 * 4];
#pragma unroll
  for (int i = 0; i < 32 * 4; ++i) O[i] = 0.f;
  const int rrow = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int ccol = (lane >> 4) * 8;
  const uint32_t vs_a = smem_u32(vs);
#pragma unroll
  for (int c = 0; c < KT / 16; ++c) {
    uint32_t pa[4];
    ldmatrix_x4(smem_u32(&ps[16 * (W & 3) + (lane & 15)][c * 16 + (lane >> 4) * 8]), pa);
#pragma unroll
    for (int dn = 0; dn < dvw / 16; ++dn) {
      uint32_t d[4];
      const int keyrow = c * 16 + rrow;
      const int dv = rh * dvw + dn * 16 + ccol;
      ldmatrix_x4_trans(vs_a + sw128_off(keyrow, dv, DV), d);
      mma16816(O + dn * 2, pa, d);
      mma16816(O + dn * 2 + 1, pa, d + 2);
    }
  }
#pragma unroll
  for (int t = 0; t < dvw / 8; ++t) {
    const int col = rh * dvw + t * 8 + (lane & 3) * 2;
    C[(size_t)r0 * DV + col] = O[t * 4 + 0];
    C[(size_t)r0 * DV + col + 1] = O[t * 4 + 1];
    C[(size_t)r1 * DV + col] = O[t * 4 + 2];
    C[(size_t)r1 * DV + col + 1] = O[t * 4 + 3];
  }
}

int main() {
  std::vector<bf16> P(BM * KT), V(KT * DV);
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : P) x = __float2bfloat16(dist(rng));
  for (auto& x : V) x = __float2bfloat16(dist(rng));
  bf16 *dP, *dV; float* dC;
  CUDA_CHECK(cudaMalloc(&dP, P.size() * 2));
  CUDA_CHECK(cudaMalloc(&dV, V.size() * 2));
  CUDA_CHECK(cudaMalloc(&dC, BM * DV * 4));
  CUDA_CHECK(cudaMemcpy(dP, P.data(), P.size() * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), V.size() * 2, cudaMemcpyHostToDevice));
  size_t shm = BM * KTP * 2 + KT * DV * 2;
  CUDA_CHECK(cudaFuncSetAttribute(pv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  pv_kernel<<<1, T, shm>>>(dP, dV, dC);
  CUDA_CHECK_LAST();
  std::vector<float> C(BM * DV);
  CUDA_CHECK(cudaMemcpy(C.data(), dC, C.size() * 4, cudaMemcpyDeviceToHost));
  double err = 0, base = 0;
  for (int m = 0; m < BM; ++m)
    for (int n = 0; n < DV; ++n) {
      double acc = 0;
      for (int k = 0; k < KT; ++k)
        acc += (double)__bfloat162float(P[m * KT + k]) * (double)__bfloat162float(V[k * DV + n]);
      err = std::max(err, std::fabs(acc - C[m * DV + n]));
      base = std::max(base, std::fabs(acc));
    }
  std::printf("PV-from-SW128 max_abs_err=%.4e (ref~%.2f) %s\n", err, base,
              err / base < 1e-2 ? "OK" : "FAIL");
  return 0;
}
