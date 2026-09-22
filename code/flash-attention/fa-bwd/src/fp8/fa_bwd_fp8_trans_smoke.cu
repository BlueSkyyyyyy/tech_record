// =============================================================================
// fa_bwd_fp8_trans_smoke.cu —— O4b 第一步：fp8 `ldmatrix.x2.trans` 取 B 片段的最小复现
// =============================================================================
// 目的（对应 ROADMAP「下一步」O4b）：
//   反向主 kernel 的 GEMM3/4/5 需要 B 操作数 `dOt/Qt/Kt`（把 dO/Q/K 转置后的副本）。
//   目前是用「逐字节 scatter 写 [N][K] 行主序」的副本 + 普通 `ldmatrix.x2` 读。
//   若能用 `ldmatrix.x2.trans` **直接从原始 [K][N] 布局**读 B，就能免掉这些副本。
//
// 为什么 fp8 不能直接照搬 bf16 的 `.trans`（本文件的结论）：
//   `ldmatrix` 以 **b16** 为单位做转置。bf16 的 b16 就是 1 个元素，`.trans` 把
//   「行主序 [K][N]」变成 mma 需要的 col 布局。但 fp8 是「2 个相邻 fp8 = 1 个 b16」，
//   若原始数组按 [K][N]（N 为列）存，每个 b16 = **沿 N 的 2 个 fp8**，转置后寄存器里
//   配对方向仍是 N —— 不是 mma 需要的「沿 K 的 4 个 fp8」。所以不是 drop-in。
//
// 正确做法（本文件验证）：把操作数按 **K 方向配对**存成 `S[i][n]`（uint16），
//   其中 `i = k/2`，`S[i][n] = (B[n][2i]) | (B[n][2i+1])<<8`（两个相邻 k 值）。
//   此时 `ldmatrix.x2.trans` 的输入 8×8 b16 矩阵，转置后每个 lane 拿到的寄存器
//   恰好是 `B[n=l/4][k=4(l%4)..4(l%4)+3]` —— 与普通 `ldmatrix.x2` 从 [N][K] 读到
//   的 B 片段**逐位相同**。
//
// 验证：同一批 fp8 B 字节，分别用
//   (1) 非转置：Bs[N][K] 行主序，`ldmatrix.x2`（现有反向主 kernel 的做法）
//   (2) 转置配对：Sp[K/2][N]（K 配对），`ldmatrix.x2.trans`（O4b 方案）
//   喂给同一个 mma，比较输出 C —— 预期 **逐位相同**（同一组操作数、同一累加次序）。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_trans_smoke.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp8.h>

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

static constexpr int M = 16;   // mma 输出行
static constexpr int N = 8;    // mma 输出列（B 的行数）
static constexpr int K = 128;  // 归约维
static constexpr int NT = 32;  // 一个 warp

// 非转置 B[N][K]（行主序，含 padding）；Sp[K/2][N]（K 配对 uint16，含 padding）。
static constexpr int BSLD = K + 16;      // 字节行距
static constexpr int SPSLD = N + 8;      // uint16 行距（40 个 uint16 = 80B；80%128≠0）

__host__ __device__ __forceinline__ unsigned char cvt_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__host__ __device__ __forceinline__ float deq_e4m3(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}

__device__ __forceinline__ void mma_e4e4(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
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

// A[M][K] fp8 行主序；Bs[N][K] 非转置；Sp[K/2][N] K 配对。输出两路 C 逐位比对。
__global__ void trans_smoke_kernel(const unsigned char* __restrict__ A,
                                   const unsigned char* __restrict__ B,
                                   float* __restrict__ C_notrans,
                                   float* __restrict__ C_trans) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* As = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Bs = As + M * (K + 16);
  uint16_t* Sp = reinterpret_cast<uint16_t*>(Bs + N * BSLD);

  const int t = threadIdx.x, lane = t & 31;
  for (int i = t; i < M * K; i += NT) {
    int r = i / K, c = i % K;
    As[r * (K + 16) + c] = A[r * K + c];
  }
  for (int i = t; i < N * K; i += NT) {
    int n = i / K, k = i % K;
    Bs[n * BSLD + k] = B[n * K + k];
  }
  // Sp[i][n] = pack(B[n][2i], B[n][2i+1])
  for (int i = t; i < (K / 2) * N; i += NT) {
    int r = i / N, n = i % N;             // r = i = k/2, n
    unsigned char lo = B[n * K + 2 * r];
    unsigned char hi = B[n * K + 2 * r + 1];
    Sp[r * SPSLD + n] = (uint16_t)(lo | ((uint16_t)hi << 8));
  }
  __syncthreads();

  float c0[4] = {0, 0, 0, 0}, c1[4] = {0, 0, 0, 0};
  for (int kk = 0; kk < K / 32; ++kk) {
    const int koff = kk * 32;
    // ---- A 片段（ldmatrix.x4，与非转置版一致；每个 k-block 重取）----
    uint32_t av[4];
    {
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int acol = (lane >> 4) * 16;
      ldmatrix_x4(smem_u32(&As[arow * (K + 16) + koff + acol]), av);
    }
    // (1) 非转置 ldmatrix.x2：B[n][k] 行主序
    uint32_t bn[2];
    {
      const int brow = lane & 7;
      const int bcol = ((lane >> 3) & 1) * 16;
      ldmatrix_x2(smem_u32(&Bs[brow * BSLD + koff + bcol]), bn);
    }
    mma_e4e4(c0, av, bn);

    // (2) 转置配对 ldmatrix.x2.trans：Sp[k/2][n] 行主序
    uint32_t bt[2];
    {
      const int row = (lane & 15);  // lanes 0..15 -> Sp 行 koff/2 .. koff/2+15
      ldmatrix_x2_trans(smem_u32(&Sp[(koff / 2 + row) * SPSLD + 0]), bt);
    }
    mma_e4e4(c1, av, bt);
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

__global__ void ref_kernel(const unsigned char* __restrict__ A,
                           const unsigned char* __restrict__ B, float* __restrict__ C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx % N;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) acc += deq_e4m3(A[m * K + k]) * deq_e4m3(B[n * K + k]);
  C[m * N + n] = acc;
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 200;
  printf("=== fp8 ldmatrix.x2.trans（K 配对布局）取 B 片段最小复现（O4b）===\n");
  printf("M=%d N=%d K=%d  BSLD=%d SPSLD=%d(uint16)\n", M, N, K, BSLD, SPSLD);

  std::vector<unsigned char> hA(M * K), hB(N * K);
  srand(20240922u);
  for (size_t i = 0; i < hA.size(); ++i) hA[i] = cvt_e4m3(2.f * ((float)rand() / RAND_MAX - 0.5f));
  for (size_t i = 0; i < hB.size(); ++i) hB[i] = cvt_e4m3(2.f * ((float)rand() / RAND_MAX - 0.5f));
  // 也覆盖几个特殊字节（0、最大/最小指数），避免全是随机值测不出布局错
  hB[0] = 0x00; hB[1] = 0x7e; hB[K] = 0xfe; hB[K + 1] = 0x01;

  unsigned char *dA, *dB;
  float *dC0, *dC1, *dR;
  CUDA_CHECK(cudaMalloc(&dA, M * K));
  CUDA_CHECK(cudaMalloc(&dB, N * K));
  CUDA_CHECK(cudaMalloc(&dC0, M * N * 4));
  CUDA_CHECK(cudaMalloc(&dC1, M * N * 4));
  CUDA_CHECK(cudaMalloc(&dR, M * N * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K, cudaMemcpyHostToDevice));

  int smem = M * (K + 16) + N * BSLD + (K / 2) * SPSLD * 2;
  trans_smoke_kernel<<<1, NT, smem>>>(dA, dB, dC0, dC1);
  ref_kernel<<<(M * N + 255) / 256, 256>>>(dA, dB, dR);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) trans_smoke_kernel<<<1, NT, smem>>>(dA, dB, dC0, dC1);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  ms /= iters;

  std::vector<float> hC0(M * N), hC1(M * N), hR(M * N);
  CUDA_CHECK(cudaMemcpy(hC0.data(), dC0, M * N * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hC1.data(), dC1, M * N * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hR.data(), dR, M * N * 4, cudaMemcpyDeviceToHost));

  double max_abs_r = 0, max_abs_t = 0, max_abs_nt = 0;
  int nbit = 0;
  for (int i = 0; i < M * N; ++i) {
    max_abs_r = std::fmax(max_abs_r, std::fabs((double)hC0[i] - hR[i]));
    max_abs_t = std::fmax(max_abs_t, std::fabs((double)hC1[i] - hR[i]));
    max_abs_nt = std::fmax(max_abs_nt, std::fabs((double)hC0[i] - hC1[i]));
    if (hC0[i] != hC1[i]) ++nbit;
  }
  printf("  notrans-vs-ref : max_abs=%.3e\n", max_abs_r);
  printf("  trans  -vs-ref : max_abs=%.3e\n", max_abs_t);
  printf("  notrans-vs-trans: max_abs=%.3e  bitwise_diff=%d/%d\n", max_abs_nt, nbit, M * N);
  printf("  time=%.5f ms\n", ms);
  printf("=== %s ===\n",
         (nbit == 0) ? "PASS（K 配对 + ldmatrix.trans 与现有 ldmatrix.x2 路径逐位一致）"
                     : "FAIL");
  cudaFree(dA); cudaFree(dB); cudaFree(dC0); cudaFree(dC1); cudaFree(dR);
  return 0;
}
