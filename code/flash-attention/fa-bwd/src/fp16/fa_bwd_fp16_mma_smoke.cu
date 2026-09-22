// =============================================================================
// fa_bwd_fp16_mma_smoke.cu —— fp16 反向的 mma.m16n8k16 布局冒烟测试（O5 前置）
// =============================================================================
// 目的：在写整条 fp16 反向之前，先用最小 GEMM 验证三件事，避免在完整 kernel 里
// 花大量时间 debug 布局：
//   1) A[16][16] 行主序，用 `ldmatrix.x4` 取 A 片段（acol 以「元素」为单位：+8 half）；
//   2) B 存成 [N][K] 行主序（= col-major B），用 `ldmatrix.x2`（非转置）取 B 片段；
//   3) B 存成 [K][N] 行主序（= row-major B），用 `ldmatrix.x2.trans` 取 B 片段。
//   后两者得到的 C[m][n] = Σ_k A[m][k]·B[k][n] 必须与 CPU 参考一致（fp16 输入、fp32 累加）。
// 这三个布局正好对应反向里的：
//   GEMM1/2/5 : B=[N][K]（K/V），非转置；
//   GEMM3/4   : B=[K][N]（dO/Q），转置。
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4],
                                        const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A: [16][16] half row-major, stride asld (half)
// B: [8][16] half row-major (= col-major B), stride bsld (half)
__global__ void kmm_nt(const __half* A, int asld, const __half* B, int bsld,
                       float* C, int csld) {
  __shared__ __half sA[16][24];
  __shared__ __half sB[16][24];
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < 16 * 16; i += 32) sA[i / 16][i % 16] = A[i];
  for (int i = lane; i < 8 * 16; i += 32) sB[i / 16][i % 16] = B[i];
  __syncwarp();
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  uint32_t av[4];
  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int acol = (lane >> 4) * 8;  // 元素单位：+8 half
  ldmatrix_x4(smem_u32(sA[arow] + acol), av);

  uint32_t bv[2];
  const int brow = lane & 7;
  const int bcol = ((lane >> 3) & 1) * 8;
  ldmatrix_x2(smem_u32(sB[brow] + bcol), bv);

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  mma_f16(c, av, bv);
  C[g * csld + c2] = c[0];
  C[g * csld + c2 + 1] = c[1];
  C[(g + 8) * csld + c2] = c[2];
  C[(g + 8) * csld + c2 + 1] = c[3];
}

// B: [16][8] half row-major (= row-major B), stride bsld (half)
__global__ void kmm_t(const __half* A, int asld, const __half* B, int bsld,
                      float* C, int csld) {
  __shared__ __half sA[16][24];
  __shared__ __half sB[16][24];
  const int lane = threadIdx.x & 31;
  for (int i = lane; i < 16 * 16; i += 32) sA[i / 16][i % 16] = A[i];
  for (int i = lane; i < 16 * 8; i += 32) sB[i / 8][i % 8] = B[i];
  __syncwarp();
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  uint32_t av[4];
  const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
  const int acol = (lane >> 4) * 8;
  ldmatrix_x4(smem_u32(sA[arow] + acol), av);

  uint32_t bv[2];
  const int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
  ldmatrix_x2_trans(smem_u32(sB[krow]), bv);

  float c[4] = {0.f, 0.f, 0.f, 0.f};
  mma_f16(c, av, bv);
  C[g * csld + c2] = c[0];
  C[g * csld + c2 + 1] = c[1];
  C[(g + 8) * csld + c2] = c[2];
  C[(g + 8) * csld + c2 + 1] = c[3];
}

int main() {
  const int M = 16, K = 16, N = 8;
  std::vector<float> A(16 * 16 + 8), B(16 * 8 + 8), C1(16 * 8), C2(16 * 8),
      Cref(16 * 8);
  srand(123);
  auto rnd = []() { return (float)((int)(rand() % 7) - 3); };  // fp16 精确的整数
  for (int i = 0; i < 16 * 16; ++i) A[i] = rnd();
  for (int i = 0; i < 16 * 8; ++i) B[i] = rnd();
  // A: [16][16]; B_row: [16][8] row-major (k rows). C[m][n]=Σ A[m][k]B[k][n].
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 8; ++n) {
      float acc = 0.f;
      for (int k = 0; k < 16; ++k) acc += A[m * 16 + k] * B[k * 8 + n];
      Cref[m * 8 + n] = acc;
    }
  // B_col: [8][16] = B^T.
  std::vector<float> Bcol(8 * 16 + 8);
  for (int n = 0; n < 8; ++n)
    for (int k = 0; k < 16; ++k) Bcol[n * 16 + k] = B[k * 8 + n];

  std::vector<__half> Ah(16 * 16 + 8), Bh(16 * 8 + 8), Bcolh(8 * 16 + 8);
  for (int i = 0; i < 16 * 16; ++i) Ah[i] = __float2half(A[i]);
  for (int i = 0; i < 16 * 8; ++i) Bh[i] = __float2half(B[i]);
  for (int i = 0; i < 8 * 16; ++i) Bcolh[i] = __float2half(Bcol[i]);

  __half *dA, *dB, *dBc;
  float *dC;
  CUDA_CHECK(cudaMalloc(&dA, Ah.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, Bh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dBc, Bcolh.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC, 16 * 8 * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dA, Ah.data(), Ah.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, Bh.data(), Bh.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dBc, Bcolh.data(), Bcolh.size() * sizeof(__half),
                        cudaMemcpyHostToDevice));

  kmm_t<<<1, 32>>>(dA, 16, dB, 8, dC, 8);
  CUDA_CHECK(cudaMemcpy(C2.data(), dC, sizeof(float) * 16 * 8, cudaMemcpyDeviceToHost));
  kmm_nt<<<1, 32>>>(dA, 16, dBc, 16, dC, 8);
  CUDA_CHECK(cudaMemcpy(C1.data(), dC, sizeof(float) * 16 * 8, cudaMemcpyDeviceToHost));

  double e1 = 0, e2 = 0;
  for (int i = 0; i < 16 * 8; ++i) {
    e1 = std::max(e1, (double)std::fabs(C1[i] - Cref[i]));
    e2 = std::max(e2, (double)std::fabs(C2[i] - Cref[i]));
  }
  printf("ldmatrix.x2 (B=[N][K], non-trans) vs CPU: max_abs=%.3e\n", e1);
  printf("ldmatrix.x2.trans (B=[K][N])      vs CPU: max_abs=%.3e\n", e2);
  bool ok = (e1 < 1e-3) && (e2 < 1e-3);
  printf("%s\n", ok ? "PASS" : "FAIL");
  cudaFree(dA); cudaFree(dB); cudaFree(dBc); cudaFree(dC);
  return ok ? 0 : 1;
}
