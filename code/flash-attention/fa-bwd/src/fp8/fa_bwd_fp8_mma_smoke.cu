// =============================================================================
// fa_bwd_fp8_mma_smoke.cu —— P3-4 第一步：fp8 mma.m16n8k32 布局最小复现
// =============================================================================
// 目的（对应 ROADMAP「下一步」P3-4 的首个子任务）：
//   在把张量核合入 fp8 反向之前，先用一个**最小 GEMM**把 `mma.m16n8k32` 的
//   操作数/累加器片段布局、`ldmatrix` 取数、以及 **rowwise scale 折回 epilogue**
//   逐位验证正确，避免反向里「缩放布局错」和「硬件 MMA 布局错」两种 bug 混在一起。
//
// 复现的两种运算（就是反向里的两大类）：
//   A) S = scale·Q·Kᵀ   ：A=Q (E4M3) × B=K (E4M3)   —— e4m3 × e4m3
//   B) dP = dO·Vᵀ       ：A=dO (E5M2) × B=V (E4M3)  —— e5m2 × e4m3
//   B 均以行主序 [N][K] 存于 smem（N 为「key/列」维，K 为 head_dim）。
//
// 布局要点（沿用 code/kernel-opt/22-fp8-gemm 的已验证公式，见 kernel-opt.md）：
//   * fp8 把「2 个相邻字节 = 1 个 b16」，所以 16×32 的 fp8 A 片段 = 16×16 的 b16
//     = 4 个 m8n8 矩阵，一次 `ldmatrix.x4` 取回 a0..a3；B 用 `ldmatrix.x2` 取 b0/b1。
//   * A 行距 = K + 16 字节 padding（144B = 36 word，36%32=4）消 ldmatrix bank conflict。
//   * 同一 (m,n) 的分块因子：c0/c1 同行（groupID），c2/c3 在 groupID+8；
//     n8 内两列由 `(lane&3)*2 + (q&1)` 给出。
//   * **rowwise 折算**：真实值 = `sa[m]·sb[n]·Σ(a_q·b_q)`；sa 按 A 的行（m），
//     sb 按 B 的行（即输出的列 n）。在 epilogue 对每个累加器乘 `sa[r]*sb[c]`。
//
// 参考值：把同一批 fp8 字节反量化回 fp32（x' = float(fp8)·scale），用 fp32 标量算
//   `C_ref[m,n] = Σ_k A'[m,k]·B'[n,k]`，与 mma 结果逐元素比对（预期 ~1e-6，fp32 舍入）。
//
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_mma_smoke.cu
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

// ----------------------------- 编译期常量 -----------------------------
static constexpr int BM = 64;   // 输出 M（query 行）
static constexpr int BN = 32;   // 输出 N（key 列）
static constexpr int BK = 128;  // 归约维 head_dim
static constexpr int WM = 2;    // M 方向 warp 数
static constexpr int WN = 2;    // N 方向 warp 数
static constexpr int NT = WM * WN * 32;  // 128 线程
static constexpr int WARP_M = BM / WM;   // 32
static constexpr int WARP_N = BN / WN;   // 16
static constexpr int MTM = WARP_M / 16;  // 2
static constexpr int MTN = WARP_N / 8;   // 2
static constexpr int ASLD = BK + 16;     // A 行距（字节），+16 padding
static constexpr int BSLD = BK + 16;     // B 行距（字节）

static constexpr float kE4M3Max = 448.0f;
static constexpr float kE5M2Max = 57344.0f;

// ----------------------------- fp8 转换 -----------------------------
__device__ __forceinline__ unsigned char cvt_e4m3(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ unsigned char cvt_e5m2(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E5M2);
}
__device__ __forceinline__ float deq_e4m3(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  return __half2float(__half(h));
}
__device__ __forceinline__ float deq_e5m2(unsigned char q) {
  __half_raw h = __nv_cvt_fp8_to_halfraw(q, __NV_E5M2);
  return __half2float(__half(h));
}

// ----------------------------- mma / ldmatrix -----------------------------
__device__ __forceinline__ void mma_e4e4(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_e5e4(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_e4e5(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma_e5e5(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
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

// =============================================================================
// 1) quantize_row_kernel：fp32 [rows][D] -> fp8 + rowwise scale
// =============================================================================
__global__ void quantize_row_kernel(const float* __restrict__ x,
                                    unsigned char* __restrict__ xq,
                                    float* __restrict__ scale, int D, int is_e5m2) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  const float v = (d < D) ? x[(size_t)row * D + d] : 0.f;
  __shared__ float sh[128];
  sh[d] = fabsf(v);
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (d < off) sh[d] = fmaxf(sh[d], sh[d + off]);
    __syncthreads();
  }
  const float fp8_max = is_e5m2 ? kE5M2Max : kE4M3Max;
  const float s = (sh[0] > 0.f) ? (sh[0] / fp8_max) : 1.f;
  if (d == 0) scale[row] = s;
  if (d < D) xq[(size_t)row * D + d] = is_e5m2 ? cvt_e5m2(v / s) : cvt_e4m3(v / s);
}

// =============================================================================
// 2) mma GEMM：C[M,N] = sa[m]·sb[n]·Σ_k A_q[m,k]·B_q[n,k]
//    A dtype 由 AE5 决定（E5M2/E4M3），B 恒 E4M3。B 行主序 [N][K]。
//    grid=(N/BN, M/BM)
// =============================================================================
template <bool AE5, bool BE5>
__global__ void __launch_bounds__(NT)
mma_smoke_kernel(const unsigned char* __restrict__ A, const float* __restrict__ sa,
                 const unsigned char* __restrict__ B, const float* __restrict__ sb,
                 float* __restrict__ C, int M, int N, int K) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* As = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Bs = As + BM * ASLD;

  const int wid = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int block_row = blockIdx.y * BM;
  const int block_col = blockIdx.x * BN;
  const int t = threadIdx.x;

  // ---- 协作载入 A[BM][BK] 与 B[BN][BK]（padding 行距）----
  for (int i = t; i < BM * BK; i += NT) {
    int r = i / BK, c = i % BK;
    As[r * ASLD + c] = A[(size_t)(block_row + r) * K + c];
  }
  for (int i = t; i < BN * BK; i += NT) {
    int r = i / BK, c = i % BK;
    Bs[r * BSLD + c] = B[(size_t)(block_col + r) * K + c];
  }
  __syncthreads();

  float acc[MTM][MTN][4];
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

#pragma unroll
  for (int kk = 0; kk < BK / 32; ++kk) {
    const int koff = kk * 32;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[MTM][4];
#pragma unroll
    for (int i = 0; i < MTM; ++i)
      ldmatrix_x4(smem_u32(&As[(wr * WARP_M + i * 16 + arow) * ASLD + koff + acol]),
                  av[i]);
    const int brow = lane & 7;
    const int bcol = ((lane >> 3) & 1) * 16;
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      uint32_t d[2];
      ldmatrix_x2(smem_u32(&Bs[(wc * WARP_N + j * 8 + brow) * BSLD + koff + bcol]), d);
      bv[j][0] = d[0];
      bv[j][1] = d[1];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        if (AE5 && BE5) mma_e5e5(acc[i][j], av[i], bv[j]);
        else if (AE5) mma_e5e4(acc[i][j], av[i], bv[j]);
        else if (BE5) mma_e4e5(acc[i][j], av[i], bv[j]);
        else mma_e4e4(acc[i][j], av[i], bv[j]);
      }
  }

  // ---- epilogue：乘 rowwise scale 并写回（sa 按输出行 m，sb 按输出列 n）----
  const int g = lane >> 2;
  const int c2 = (lane & 3) * 2;
  const int r0 = block_row + wr * WARP_M, c0 = block_col + wc * WARP_N;
#pragma unroll
  for (int i = 0; i < MTM; ++i)
#pragma unroll
    for (int j = 0; j < MTN; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
        const int c = c0 + j * 8 + c2 + (q & 1);
        if (r < M && c < N)
          C[(size_t)r * N + c] = acc[i][j][q] * sa[r] * sb[c];
      }
}

// =============================================================================
// 3) 参考 kernel：反量化后 fp32 标量 GEMM
// =============================================================================
template <bool AE5, bool BE5>
__global__ void ref_kernel(const unsigned char* __restrict__ A,
                           const float* __restrict__ sa,
                           const unsigned char* __restrict__ B,
                           const float* __restrict__ sb, float* __restrict__ C, int M,
                           int N, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= M * N) return;
  int m = idx / N, n = idx % N;
  float acc = 0.f;
  for (int k = 0; k < K; ++k) {
    float a = AE5 ? deq_e5m2(A[(size_t)m * K + k]) : deq_e4m3(A[(size_t)m * K + k]);
    float b = BE5 ? deq_e5m2(B[(size_t)n * K + k]) : deq_e4m3(B[(size_t)n * K + k]);
    acc += a * b;
  }
  C[(size_t)m * N + n] = acc * sa[m] * sb[n];
}

// =============================================================================
// host
// =============================================================================
static void run_variant(bool ae5, bool be5, int M, int N, int K, int iters) {
  const size_t aN = (size_t)M * K, bN = (size_t)N * K, cN = (size_t)M * N;
  std::vector<float> hAf(aN), hBf(bN);
  srand(20240922u + (ae5 ? 1 : 0) + (be5 ? 2 : 0));
  for (size_t i = 0; i < aN; ++i) hAf[i] = 2.f * ((float)rand() / RAND_MAX - 0.5f);
  for (size_t i = 0; i < bN; ++i) hBf[i] = 2.f * ((float)rand() / RAND_MAX - 0.5f);

  float *d_Af, *d_Bf, *d_sa, *d_sb, *d_C, *d_Cref;
  unsigned char *d_A, *d_B;
  CUDA_CHECK(cudaMalloc(&d_Af, aN * 4));
  CUDA_CHECK(cudaMalloc(&d_Bf, bN * 4));
  CUDA_CHECK(cudaMalloc(&d_A, aN));
  CUDA_CHECK(cudaMalloc(&d_B, bN));
  CUDA_CHECK(cudaMalloc(&d_sa, M * 4));
  CUDA_CHECK(cudaMalloc(&d_sb, N * 4));
  CUDA_CHECK(cudaMalloc(&d_C, cN * 4));
  CUDA_CHECK(cudaMalloc(&d_Cref, cN * 4));
  CUDA_CHECK(cudaMemcpy(d_Af, hAf.data(), aN * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_Bf, hBf.data(), bN * 4, cudaMemcpyHostToDevice));

  quantize_row_kernel<<<M, 128>>>(d_Af, d_A, d_sa, K, ae5 ? 1 : 0);
  quantize_row_kernel<<<N, 128>>>(d_Bf, d_B, d_sb, K, be5 ? 1 : 0);

  dim3 grid(N / BN, M / BM);
  int smem = BM * ASLD + BN * BSLD;
  auto launch = [&](bool run_ref) {
#define DISPATCH(AE, BE)                                                          \
  do {                                                                            \
    mma_smoke_kernel<AE, BE><<<grid, NT, smem>>>(d_A, d_sa, d_B, d_sb, d_C, M, N, K); \
    if (run_ref)                                                                  \
      ref_kernel<AE, BE><<<(M * N + 255) / 256, 256>>>(d_A, d_sa, d_B, d_sb, d_Cref, M, N, K); \
  } while (0)
    if (ae5 && be5) DISPATCH(true, true);
    else if (ae5) DISPATCH(true, false);
    else if (be5) DISPATCH(false, true);
    else DISPATCH(false, false);
#undef DISPATCH
  };
  launch(true);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));
  CUDA_CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) launch(false);
  CUDA_CHECK(cudaEventRecord(e1));
  CUDA_CHECK(cudaEventSynchronize(e1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
  ms /= iters;

  std::vector<float> hC(cN), hR(cN);
  CUDA_CHECK(cudaMemcpy(hC.data(), d_C, cN * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hR.data(), d_Cref, cN * 4, cudaMemcpyDeviceToHost));
  double max_abs = 0, max_rel = 0, ref_amax = 0;
  for (size_t i = 0; i < cN; ++i) {
    double d = std::fabs((double)hC[i] - (double)hR[i]);
    if (d > max_abs) max_abs = d;
    double r = d / (std::fabs((double)hR[i]) + 1e-3);
    if (r > max_rel) max_rel = r;
    ref_amax = std::fmax(ref_amax, std::fabs((double)hR[i]));
  }
  double flops = 2.0 * M * N * K;
  const char* tag = ae5 && be5 ? "E5M2xE5M2"
                    : ae5    ? "E5M2xE4M3"
                    : be5    ? "E4M3xE5M2"
                             : "E4M3xE4M3";
  printf("  [%s] M=%d N=%d K=%d  mma-vs-ref max_abs=%.3e max_rel=%.3e (ref_amax=%.3f)\n",
         tag, M, N, K, max_abs, max_rel, ref_amax);
  printf("         time=%.4f ms  %.2f TFLOPS  (%.2f GFLOP)\n", ms,
         flops / (ms * 1e-3) / 1e12, flops / 1e9);

  cudaFree(d_Af); cudaFree(d_Bf); cudaFree(d_A); cudaFree(d_B);
  cudaFree(d_sa); cudaFree(d_sb); cudaFree(d_C); cudaFree(d_Cref);
}

int main(int argc, char** argv) {
  int iters = (argc > 1) ? atoi(argv[1]) : 100;
  printf("=== fp8 mma.m16n8k32 布局最小复现（P3-4a）===\n");
  printf("BM=%d BN=%d BK=%d WM=%d WN=%d threads=%d  A row stride=%d B row stride=%d\n",
         BM, BN, BK, WM, WN, NT, ASLD, BSLD);
  // 覆盖 4 种 dtype 组合（重点验证反向 dV 用的 E4M3×E5M2）
  run_variant(false, false, 64, 32, 128, iters);
  run_variant(true, false, 64, 32, 128, iters);
  run_variant(false, true, 64, 32, 128, iters);
  run_variant(true, true, 64, 32, 128, iters);
  run_variant(false, true, 128, 64, 128, iters);
  printf("=== done ===\n");
  return 0;
}
