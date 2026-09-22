// =============================================================================
// fa_bwd_fp16_atrans_smoke.cu —— O6b 第一步：fp16 用 `ldmatrix.x4.trans` 取 A 片段
// =============================================================================
// 目的（对应 ROADMAP「下一步」O6b 的「更省的转置布局」）：
//   反向主 kernel 的 GEMM3/GEMM4 需要 A 操作数 `PsT/dSsT`（P/dS 的转置副本）：
//     * GEMM3 : dV = Pᵀ·dO，A = Pᵀ[BN][BM]
//     * GEMM4 : dK = dSᵀ·Q，A = dSᵀ[BN][BM]
//   现状是把 P/dS（天然按 [BM][BN] 算出来）**额外复制一份转置**存成 `[BN][BM]`
//   行主序，用普通 `ldmatrix.x4` 读。这两份转置副本各占 BN*(BM+8) 个 half。
//
// 若能用 `ldmatrix.x4.trans` **直接从 [BM][BN] 原布局**读 A 片段，就能**同时**
// 消掉 `PsT` 与 `dSsT`（共 2*BN*(BM+8) 个 half ≈ 9.2KB），把 O6 的 K/V 双缓冲
// （83.97KB）压到 3 CTA/SM 的预算（≤77.8KB）以内。
//
// 推导（b16，非 fp8 的「2 个 fp8 = 1 个 b16」，所以配对方向简单）：
//   A 存成 [K][M] 行主序（即转置），用 `ldmatrix.x4.trans` 取 M×K 片段。地址模式是
//   非转置 A 的 bit3/bit4 互换：
//     非转置（A=[M][K]）：row=(lane&7)+((lane>>3)&1)*8, col=(lane>>4)*8
//     转置  （A=[K][M]）：krow=(lane&7)+((lane>>4)&1)*8, mcol=((lane>>3)&1)*8
//   直观：lanes 0-7 取「行 k0-7 × 列 m0-7」的 8×8，`.trans` 后即 (m0-7,k0-7)=a0；
//   lanes 8-15 取 (k0-7,m8-15)→(m8-15,k0-7)=a1；16-23 取 (k8-15,m0-7)=a2；
//   24-31 取 (k8-15,m8-15)=a3。
//
// 验证：同一组 half 数据，分别用
//   (1) 非转置 A[M][K] + `ldmatrix.x4`      （现有 GEMM4 的做法）
//   (2) 转置   A[K][M] + `ldmatrix.x4.trans`（O6b 方案）
//   喂给同一个 mma.m16n8k16，比较输出 C —— 预期 **逐位相同**。
//
// 运行：scripts/run.sh src/fp16/fa_bwd_fp16_atrans_smoke.cu
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

static constexpr int M = 16;   // mma 输出行（一个 warp 的 M_TILE）
static constexpr int N = 8;    // mma 输出列
static constexpr int K = 64;   // 归约维（BM=64）
static constexpr int NT = 32;  // 一个 warp

// 非转置 A[M][K] 行距；转置 P[K][M] 行距。都 +8 消 ldmatrix bank conflict。
static constexpr int ASLD = K + 8;
static constexpr int PSLD = M + 8;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
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

// A[M][K] 行主序 + B[N][K] 行主序；一路用 ldmatrix.x4，一路用 ldmatrix.x4.trans 从 Aᵀ[K][M] 读。
__global__ void atrans_smoke_kernel(const __half* __restrict__ A,
                                    const __half* __restrict__ B, float* __restrict__ C_notrans,
                                    float* __restrict__ C_trans) {
  extern __shared__ __align__(16) char smem[];
  __half* As = reinterpret_cast<__half*>(smem);         // [M][ASLD]
  __half* Bs = As + M * ASLD;                            // [N][K+8]
  __half* Ps = Bs + N * (K + 8);                         // [K][PSLD]（Aᵀ）

  const int t = threadIdx.x, lane = t & 31;
  for (int i = t; i < M * K; i += NT) {
    int r = i / K, c = i % K;
    As[r * ASLD + c] = A[r * K + c];
    Ps[c * PSLD + r] = A[r * K + c];  // 转置副本：P[k][m] = A[m][k]
  }
  for (int i = t; i < N * K; i += NT) {
    int n = i / K, k = i % K;
    Bs[n * (K + 8) + k] = B[n * K + k];
  }
  __syncthreads();

  float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
  for (int kk = 0; kk < K / 16; ++kk) {
    const int koff = kk * 16;
    // B[N][K] + ldmatrix.x2
    uint32_t bv[2];
    {
      const int brow = lane & 7;
      const int bcol = ((lane >> 3) & 1) * 8;
      ldmatrix_x4(smem_u32(&Bs[brow * (K + 8) + koff + bcol]), bv);  // x4 只用前 2 个也可
    }
    // (1) 非转置 A[M][K] + ldmatrix.x4
    {
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int acol = (lane >> 4) * 8;
      uint32_t av[4];
      ldmatrix_x4(smem_u32(&As[arow * ASLD + koff + acol]), av);
      mma_f16(c0, av, bv);
    }
    // (2) 转置 Aᵀ[K][M] + ldmatrix.x4.trans
    {
      const int krow = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int mcol = ((lane >> 3) & 1) * 8;
      uint32_t av[4];
      ldmatrix_x4_trans(smem_u32(&Ps[(koff + krow) * PSLD + mcol]), av);
      mma_f16(c1, av, bv);
    }
  }

  const int g = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const int r = g + (q >= 2 ? 8 : 0);
    const int c = c2 + (q & 1);
    C_notrans[r * N + c] = c0[q];
    C_trans[r * N + c] = c1[q];
  }
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 200;
  printf("=== fp16 ldmatrix.x4.trans（A[K][M] 转置布局）取 A 片段最小复现（O6b）===\n");
  printf("M=%d N=%d K=%d  ASLD=%d PSLD=%d\n", M, N, K, ASLD, PSLD);

  std::vector<__half> hA(M * K), hB(N * K);
  srand(20240923u);
  auto rh = [](float x) { return __float2half(x); };
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = rh(2.f * ((float)rand() / RAND_MAX - 0.5f));
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = rh(2.f * ((float)rand() / RAND_MAX - 0.5f));
  // 特殊值覆盖（0、±1、大值），避免全随机测不出布局错
  hA[0] = rh(0.f); hA[1] = rh(1.f); hA[K] = rh(-1.f); hA[K + 1] = rh(3.f);
  hB[0] = rh(0.5f); hB[1] = rh(-2.f);

  __half *dA, *dB;
  float *dC0, *dC1;
  CUDA_CHECK(cudaMalloc(&dA, M * K * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dB, N * K * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dC0, M * N * 4));
  CUDA_CHECK(cudaMalloc(&dC1, M * N * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * sizeof(__half), cudaMemcpyHostToDevice));

  int smem = (M * ASLD + N * (K + 8) + K * PSLD) * (int)sizeof(__half);
  atrans_smoke_kernel<<<1, NT, smem>>>(dA, dB, dC0, dC1);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) atrans_smoke_kernel<<<1, NT, smem>>>(dA, dB, dC0, dC1);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  ms /= iters;

  std::vector<float> hC0(M * N), hC1(M * N);
  CUDA_CHECK(cudaMemcpy(hC0.data(), dC0, M * N * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC1.data(), dC1, M * N * 4, cudaMemcpyDeviceToHost));

  double max_abs_nt = 0;
  int nbit = 0;
  for (int i = 0; i < M * N; ++i) {
    max_abs_nt = std::fmax(max_abs_nt, std::fabs((double)hC0[i] - hC1[i]));
    if (hC0[i] != hC1[i]) ++nbit;
  }
  printf("  notrans-vs-trans: max_abs=%.3e  bitwise_diff=%d/%d\n", max_abs_nt, nbit, M * N);
  printf("  time=%.5f ms\n", ms);
  printf("=== %s ===\n",
         (nbit == 0) ? "PASS（Aᵀ[K][M] + ldmatrix.x4.trans 与 A[M][K] + ldmatrix.x4 逐位一致）"
                     : "FAIL");
  cudaFree(dA); cudaFree(dB); cudaFree(dC0); cudaFree(dC1);
  return 0;
}
