// 32 MLA 极限冲刺（三）冒烟 v2：MN-major（免转置 V）GMMA 描述符暴力验证
// 用法：./mn_smoke.out [layout] [lbo] [sbo] [step]
//   layout: 0 = [kg][ng][key][n]（cute canonical），1 = [ng][kg][key][n]
//   step:   0 = s*2*SBO(字节)  1 = s*SBO  2 = s*32
#include "../common/cuda_utils.cuh"
#include "./wgmma_sw128.cuh"
#include <cuda_bf16.h>

using bf16 = __nv_bfloat16;
constexpr int M = 64, N = 256, K = 64;
constexpr int T = 128;

__device__ __host__ __forceinline__ int mn_off(int k, int n, int layout) {
  const int ch = ((n & 63) >> 3) ^ (k & 7);
  if (layout == 0)
    return (k >> 3) * (N / 64) * 1024 + (n >> 6) * 1024 + (k & 7) * 128 + ch * 16 + (n & 7) * 2;
  return (n >> 6) * (K / 8) * 1024 + (k >> 3) * 1024 + (k & 7) * 128 + ch * 16 + (n & 7) * 2;
}

__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr, uint32_t lbo_b, uint32_t sbo_b) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((lbo_b >> 4) & 0x3FFF) << 16;
  d |= (uint64_t)((sbo_b >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;
  return d;
}

__host__ __device__ __forceinline__ int kdiag(int m, int n) { return m*100000+n; }
__global__ void __launch_bounds__(T) mn_kernel(const bf16* __restrict__ A, const bf16* __restrict__ V,
                                               float* __restrict__ C, int layout, int lbo, int sbo,
                                               int step) {
  extern __shared__ __align__(1024) char smem[];
  char* As = smem;
  char* Vs = smem + M * K * 2;
  const int tid = threadIdx.x, lane = tid & 31, W = tid >> 5;
  const int r0 = 16 * (W & 3) + (lane >> 2), r1 = r0 + 8;
  for (int i = tid; i < M * K / 8; i += T) {
    const int m = i / (K / 8), cu = i % (K / 8);
    uint4 v = *reinterpret_cast<const uint4*>(&A[m * K + cu * 8]);
    sw128_store16(As, m, cu * 8, K, v);
  }
  for (int i = tid; i < K * N; i += T) {
    const int k = i / N, n = i % N;
    *reinterpret_cast<bf16*>(Vs + mn_off(k, n, layout)) = V[k * N + n];
  }
  __syncthreads();

  float O[128];
#pragma unroll
  for (int i = 0; i < 128; ++i) O[i] = 0.f;
  const uint32_t as_a = smem_u32(As), vs_a = smem_u32(Vs);
  wgmma_fence();
#pragma unroll
  for (int s = 0; s < K / 16; ++s) {
    uint32_t off = (step == 0) ? (uint32_t)(s * 2 * sbo) : (step == 1) ? (uint32_t)(s * sbo)
                                    : (uint32_t)(s * 32);
    wgmma_m64n256k16(O, make_desc_sw128(sw128_k16_addr(as_a, s), (K / 64) * 1024),
                     make_desc_mn(vs_a + off, lbo, sbo));
  }
  wgmma_commit();
  wgmma_wait0();
#pragma unroll
  for (int t = 0; t < 32; ++t) {
    const int col = t * 8 + (lane & 3) * 2;
    C[r0 * N + col] = O[t * 4 + 0];
    C[r0 * N + col + 1] = O[t * 4 + 1];
    C[r1 * N + col] = O[t * 4 + 2];
    C[r1 * N + col + 1] = O[t * 4 + 3];
  }
}

int main(int argc, char** argv) {
  int layout = argc > 1 ? atoi(argv[1]) : 0;
  int lbo = argc > 2 ? atoi(argv[2]) : 64;
  int sbo = argc > 3 ? atoi(argv[3]) : 256;
  int step = argc > 4 ? atoi(argv[4]) : 0;
  int diag = argc > 5 ? atoi(argv[5]) : 0;
  bf16 *A, *V;
  float* C;
  CUDA_CHECK(cudaMalloc(&A, M * K * 2));
  CUDA_CHECK(cudaMalloc(&V, K * N * 2));
  CUDA_CHECK(cudaMalloc(&C, M * N * 4));
  std::vector<bf16> hA(M * K), hV(K * N);
  srand(11);
  auto rnd = [] { return __float2bfloat16(0.2f * ((float)rand() / RAND_MAX - 0.5f)); };
  if (diag) {
    for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) hA[m*K+k] = __float2bfloat16(m==k?1.f:0.f);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) hV[k*N+n] = __float2bfloat16(diag==1?(float)k:(float)n);
  } else {
    for (auto& x : hA) x = rnd();
    for (auto& x : hV) x = rnd();
  }
  CUDA_CHECK(cudaMemcpy(A, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(V, hV.data(), K * N * 2, cudaMemcpyHostToDevice));
  std::vector<float> ref(M * N, 0.f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0;
      for (int k = 0; k < K; ++k)
        acc += __bfloat162float(hA[m * K + k]) * __bfloat162float(hV[k * N + n]);
      ref[m * N + n] = acc;
    }
  const size_t shm = M * K * 2 + K * N * 2;
  auto fn = mn_kernel;
  CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shm));
  fn<<<1, T, shm>>>(A, V, C, layout, lbo, sbo, step);
  CUDA_CHECK_LAST();
  std::vector<float> hC(M * N);
  CUDA_CHECK(cudaMemcpy(hC.data(), C, M * N * 4, cudaMemcpyDeviceToHost));
  double err = 0, rr = 0;
  for (int i = 0; i < M * N; ++i) {
    err = std::max(err, std::fabs((double)hC[i] - ref[i]));
    rr = std::max(rr, std::fabs((double)ref[i]));
  }
  std::printf("layout=%d lbo=%d sbo=%d step=%d  err=%.3e ref=%.3f %s\n", layout, lbo, sbo, step,
              err, rr, err / std::max(rr, 1e-6) < 3e-2 ? "OK" : "FAIL");
  if (diag) {
    for (int m = 0; m < 2; ++m) {
      {
        int printed = 0;
        for (int n = 0; n < N && printed < 24; ++n) {
          int v = (int)(hC[m*N+n] + 0.5f);
          std::printf("  m=%d n=%d -> read=%d\n", m, n, v);
          ++printed;
        }
      }
    }
  }
  return 0;
}
