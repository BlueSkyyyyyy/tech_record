// =============================================================================
// fa_bwd_fp8_mma_onefile.cu —— FlashAttention 反向（FP8）张量核版（P3-4）
// =============================================================================
// 在 golden（fa_bwd_fp8_onefile.cu，标量 fp32 模拟）基础上，把反向的 5 个矩阵乘
// 全部换成 `mma.sync.aligned.m16n8k32`（E4M3/E5M2），用 ldmatrix 从 smem 取操作数、
// 用 smem 行距 padding 消 bank conflict，rowwise scale 在 epilogue 折回。
//
// 与 golden 的差异（更接近 TE、且更适合张量核）：
//   * dP **不再量化**（它只进 dS = P∘(dP−D) 的 fp32 计算，不进张量核）。
//   * dS 也不再以 fp32 直接进张量核：分别形成 dS2 / dS3 两个 fp8 操作数（见下）。
//   * 所有 mma 累加器 fp32；最终 dq/dk/dv 仍是 fp32（与 golden 口径一致，便于对拍）。
//
// ----- 五个矩阵乘的量化/折算记账（rowwise）-----
// 记 Q/K/V/dO 的 rowwise scale（over head_dim）为 qs[m],ks[j],vs[j],dos[m]。
// mma 计算 Σ a_q·b_q（a_q,b_q 为 fp8 原始字节，不含 scale），scale 折回方式：
//   1) S = scale·QKᵀ : A=Q,B=K，两者 scale 都不在归约维 → epilogue 乘 scale·qs[m]·ks[j]。
//   2) dP = dO·Vᵀ   : A=dO,B=V，同理 → epilogue 乘 dos[m]·vs[j]。
//   3) dV = Pᵀ·dO   : 归约维 m 上 dO 的 rowwise scale dos[m] 随归约变。
//        定义 Ap[j][m] = P[m][j]·dos[m]，按 j 行 rowwise 量化得 sA[j]；
//        则 Σ P·dO = Σ (Ap/dos[m])·(do8·dos[m]) = sA[j]·Σ ap_q·do8，B 用**原始** do8。
//   4) dQ = scale·dS·K : 归约维 j 上 ks[j] 随归约变。定义 dS2[m][j]=dS[m][j]·ks[j]，
//        按 m 行 rowwise 量化得 sds2[m]；Σ dS·K = sds2[m]·Σ ds2_q·k8，B 用**原始** k8。
//   5) dK = scale·dSᵀ·Q : 归约维 m 上 qs[m] 随归约变。定义 dS3[j][m]=dS[m][j]·qs[m]，
//        按 j 行 rowwise 量化得 sds3[j]；Σ dS·Q = sds3[j]·Σ ds3_q·q8，B 用**原始** q8。
// 这样每个 mma 的折算因子都是「每输出行一个（或 epilogue 里两个独立 scale 相乘）」，
// 与 m16n8 累加器布局吻合（同一行 4 个 acc 共用行 scale；列由 (lane&3)*2 区分）。
//
// ----- mma 布局（沿用 22-fp8-gemm 已验证公式）-----
//   fp8 2 个相邻字节=1 个 b16；16×32 fp8 A = 4 个 m8n8 b16 → ldmatrix.x4 取 a0..a3；
//   B 用 ldmatrix.x2 取 b0/b1（b0 对应 k 0..15、b1 对应 k 16..31）。
//   A/B 行距均取 16B 的整数倍并 padding（+16B）以消 ldmatrix bank conflict。
//   输出：c0/c1 在行 g=lane>>2，c2/c3 在 g+8；n8 内两列 = (lane&3)*2+(q&1)。
//
// 参考：docs/02-fp8-bwd-design.md、src/fp8/fa_bwd_fp8_mma_smoke.cu（布局已验证）。
// 运行：scripts/run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int kHeadDim = 128;
static constexpr int BM = 64;
static constexpr int BN = 32;
static constexpr int THREADS = 128;
static constexpr int WN = 2;

// smem 行距（字节，均为 16 的整数倍以配合 ldmatrix）
static constexpr int ASLD = kHeadDim + 16;  // 144，A 行距离 head_dim=128
static constexpr int KTS  = 32 + 16;        // 48，[d][j] 转置 K（K=32）
static constexpr int QTS  = 64 + 16;        // 80，[d][m] / [j][m]（K=64）
static constexpr int DSS2 = 32 + 16;        // 48，dS2 [m][j]（K=32）

static constexpr float kE4M3Max = 448.0f;
static constexpr float kE5M2Max = 57344.0f;

// O1：preprocess 的 LSE 改用 mma 分块（见下），独立的 tile 常量。
static constexpr int LBM = 64;
static constexpr int LBN = 64;
static constexpr int kLseSmemBytes = LBM * ASLD + LBN * ASLD + (LBM + LBN) * (int)sizeof(float);

// 动态 smem 布局
static constexpr int kFp8Bytes = BM * ASLD            // Qs
                              + BN * ASLD            // Ks
                              + BN * ASLD            // Vs
                              + BM * ASLD            // dOs
                              + kHeadDim * KTS       // Kt
                              + kHeadDim * QTS       // Qt
                              + kHeadDim * QTS       // dOt
                              + BN * QTS             // Ap
                              + BM * DSS2            // dS2
                              + BN * QTS;            // dS3
static constexpr int kNScale = 3 * BM + 4 * BN;       // qs,dos,sds2 (BM) + ks,vs,sA,sds3 (BN)
static constexpr int kSmemBytes =
    kFp8Bytes + (kNScale + 2 * BM * BN) * (int)sizeof(float);

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t _e = (call);                                                    \
    if (_e != cudaSuccess) {                                                    \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),       \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

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
enum MmaKind { E4E4 = 0, E5E4 = 1, E4E5 = 2 };

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

// A[M_TILE][K_TILE]、B[N_TILE][K_TILE] 均行主序（行距 asld/bsld，含 padding）。
// 每 warp 负责 WARP_M×WARP_N 输出块；各 warp 旧 (wm,wn) 由调用方给出。
template <int WARP_M, int WARP_N, int K_TILE, int KIND>
__device__ __forceinline__ void mma_block(const unsigned char* As, int asld,
                                          const unsigned char* Bs, int bsld,
                                          float acc[WARP_M / 16][WARP_N / 8][4],
                                          int wm, int wn, int lane) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
#pragma unroll
  for (int kk = 0; kk < K_TILE / 32; ++kk) {
    const int koff = kk * 32;
    const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int acol = (lane >> 4) * 16;
    uint32_t av[MTM][4];
#pragma unroll
    for (int i = 0; i < MTM; ++i)
      ldmatrix_x4(smem_u32(As + (wm * WARP_M + i * 16 + arow) * asld + koff + acol),
                  av[i]);
    const int brow = lane & 7;
    const int bcol = ((lane >> 3) & 1) * 16;
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      uint32_t d[2];
      ldmatrix_x2(smem_u32(Bs + (wn * WARP_N + j * 8 + brow) * bsld + koff + bcol), d);
      bv[j][0] = d[0];
      bv[j][1] = d[1];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        if (KIND == E4E4) mma_e4e4(acc[i][j], av[i], bv[j]);
        else if (KIND == E5E4) mma_e5e4(acc[i][j], av[i], bv[j]);
        else mma_e4e5(acc[i][j], av[i], bv[j]);
      }
  }
}

// =============================================================================
// 1) quantize_row_kernel（与 golden 相同）：fp32 [rows][D] -> fp8 + rowwise scale
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
// 2a) lse_mma_kernel【O1 优化】：用 mma 分块 Q·Kᵀ 求 LSE
// =============================================================================
// 旧 preprocess 每个 (s,h) 行一个 block、128 线程标量扫 K，每对 (i,j) 反量化
// 2×128 个 e4m3 再 FFMA；S=4096 时 preprocess ~71ms，是端到端第一瓶颈。
//
// 新做法：与反向主 kernel 同源的 tensor-core QK：
//   * grid = (S/LBM, H, B)，每 CTA 处理 LBM=64 行 Q；4 个 warp 各 16 行（wm=wid），
//     沿 N 方向一次吃 LBN=64 列；Q/K 分块进 smem，mma.m16n8k32 E4M3×E4M3 算 S 块。
//   * P 不物化：mma 的 fp32 累加器直接做 online-softmax（running max/l），
//     LSE 的行 max/sum 在 warp 内按 lane 组（同 row 的 4 个 lane）shfl 归约。
//   * 与旧版数学一致（scale·qs·ks、causal mask、NaN 安全），只是走张量核并大幅
//     提升并行度（S=4096 时 1024 CTA vs 旧的 65536 个 1-行 CTA×标量）。
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
               const unsigned char* __restrict__ k8, const float* __restrict__ ks,
               float* __restrict__ lse, int S, int H, float scale, int causal) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks = Qs + LBM * ASLD;
  float* qs_s = reinterpret_cast<float*>(Ks + LBN * ASLD);
  float* ks_s = qs_s + LBM;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * LBM;

  for (int i = tid; i < LBM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim, d = i % kHeadDim;
    int qi = m0 + r;
    Qs[r * ASLD + d] =
        (qi < S) ? q8[(((size_t)(b * S + qi)) * H + h) * kHeadDim + d] : cvt_e4m3(0.f);
  }
  if (tid < LBM)
    qs_s[tid] = (m0 + tid < S) ? qs[((size_t)(b * S + m0 + tid)) * H + h] : 1.f;
  __syncthreads();

  const int ncols = causal ? min(S, m0 + LBM) : S;
  const int ntiles = (ncols + LBN - 1) / LBN;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * LBN;
    for (int i = tid; i < LBN * kHeadDim; i += THREADS) {
      int r = i / kHeadDim, d = i % kHeadDim;
      int jg = j0 + r;
      Ks[r * ASLD + d] =
          (jg < S) ? k8[(((size_t)(b * S + jg)) * H + h) * kHeadDim + d] : cvt_e4m3(0.f);
    }
    if (tid < LBN)
      ks_s[tid] = (j0 + tid < S) ? ks[((size_t)(b * S + j0 + tid)) * H + h] : 1.f;
    __syncthreads();

    float acc[1][8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
    mma_block<16, LBN, kHeadDim, E4E4>(Qs, ASLD, Ks, ASLD, acc, wid, 0, lane);

#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        int s = q >= 2 ? 1 : 0;
        int r = wid * 16 + g + (q >= 2 ? 8 : 0);
        int c = j * 8 + c2 + (q & 1);
        int qi = m0 + r, jg = j0 + c;
        float sv = -INFINITY;
        if (qi < S && jg < S && !(causal && jg > qi))
          sv = acc[0][j][q] * scale * qs_s[r] * ks_s[c];
        if (sv != -INFINITY) {
          float mn = fmaxf(mrow[s], sv);
          lrow[s] = lrow[s] * expf(mrow[s] - mn) + expf(sv - mn);
          mrow[s] = mn;
        }
      }
    __syncthreads();
  }

#pragma unroll
  for (int s = 0; s < 2; ++s) {
    float m = mrow[s], l = lrow[s];
#pragma unroll
    for (int off = 1; off <= 2; off <<= 1) {
      float m2 = __shfl_xor_sync(0xffffffffu, m, off);
      float l2 = __shfl_xor_sync(0xffffffffu, l, off);
      float mn = fmaxf(m, m2);
      float ca = (m == -INFINITY) ? 0.f : l * expf(m - mn);
      float cb = (m2 == -INFINITY) ? 0.f : l2 * expf(m2 - mn);
      l = ca + cb;
      m = mn;
    }
    if (c2 == 0) {
      int r = wid * 16 + g + (s ? 8 : 0);
      int qi = m0 + r;
      if (qi < S) lse[((size_t)(b * S + qi)) * H + h] = m + logf(l);
    }
  }
}

// =============================================================================
// 2b) delta_kernel：D = rowsum(dO ∘ O)（反量化 e5m2·dos 后与 fp32 O 点积）
// =============================================================================
__global__ void delta_kernel(const float* __restrict__ o,
                             const unsigned char* __restrict__ do8,
                             const float* __restrict__ dos, float* __restrict__ delta,
                             int S, int H) {
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const float* orow = o + row * kHeadDim;
  const unsigned char* dorow = do8 + row * kHeadDim;
  const float do_scale = dos[row];
  float dp = 0.f;
  for (int d = tid; d < kHeadDim; d += blockDim.x)
    dp += orow[d] * (deq_e5m2(dorow[d]) * do_scale);
  __shared__ float sh_delta[THREADS];
  sh_delta[tid] = dp;
  __syncthreads();
  for (int off = THREADS / 2; off > 0; off >>= 1) {
    if (tid < off) sh_delta[tid] += sh_delta[tid + off];
    __syncthreads();
  }
  if (tid == 0) delta[row] = sh_delta[0];
}

// =============================================================================
// 3) main kernel：1colblock 反向，5 个 GEMM 全部用 mma.m16n8k32
// =============================================================================
__global__ void __launch_bounds__(THREADS)
fa_bwd_fp8_mma_kernel(const unsigned char* __restrict__ q8,
                      const float* __restrict__ qs,
                      const unsigned char* __restrict__ k8,
                      const float* __restrict__ ks,
                      const unsigned char* __restrict__ v8,
                      const float* __restrict__ vs,
                      const unsigned char* __restrict__ do8,
                      const float* __restrict__ dos,
                      const float* __restrict__ delta,
                      const float* __restrict__ lse,
                      float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                      float* __restrict__ dv_acc, int S, int H, float scale,
                      int causal) {
  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks  = Qs + BM * ASLD;
  unsigned char* Vs  = Ks + BN * ASLD;
  unsigned char* dOs = Vs + BN * ASLD;
  unsigned char* Kt  = dOs + BM * ASLD;
  unsigned char* Qt  = Kt + kHeadDim * KTS;
  unsigned char* dOt = Qt + kHeadDim * QTS;
  unsigned char* Ap  = dOt + kHeadDim * QTS;
  unsigned char* dS2 = Ap + BN * QTS;
  unsigned char* dS3 = dS2 + BM * DSS2;
  float* scales = reinterpret_cast<float*>(dS3 + BN * QTS);
  float* Ps = scales + kNScale;              // P fp32 [BM][BN]
  float* Ss = Ps + BM * BN;                  // dS fp32 [BM][BN]
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  // ---- 载入 Q/dO（含转置副本 Qt/dOt，供 dK/dV 的 B 操作数）----
  for (int i = tid; i < BM * kHeadDim; i += THREADS) {
    int r = i / kHeadDim, d = i % kHeadDim;
    int qi = m0 + r;
    unsigned char qv = cvt_e4m3(0.f), ov = cvt_e5m2(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * kHeadDim + d;
      qv = q8[idx];
      ov = do8[idx];
    }
    Qs[r * ASLD + d] = qv;
    Qt[d * QTS + r] = qv;
    dOs[r * ASLD + d] = ov;
    dOt[d * QTS + r] = ov;
  }
  if (tid < BM) {
    int qi = m0 + tid;
    qs_s[tid] = (qi < S) ? qs[((size_t)(b * S + qi)) * H + h] : 1.f;
    dos_s[tid] = (qi < S) ? dos[((size_t)(b * S + qi)) * H + h] : 1.f;
  }
  __syncthreads();

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    // ---- 载入 K/V（Ks/Vs 原布局 + Kt 转置副本，供 dQ 的 B 操作数）----
    for (int i = tid; i < BN * kHeadDim; i += THREADS) {
      int r = i / kHeadDim, d = i % kHeadDim;
      int jg = j0 + r;
      unsigned char kv = cvt_e4m3(0.f), vv = cvt_e4m3(0.f);
      if (jg < S) {
        size_t idx = (((size_t)(b * S + jg)) * H + h) * kHeadDim + d;
        kv = k8[idx];
        vv = v8[idx];
      }
      Ks[r * ASLD + d] = kv;
      Kt[d * KTS + r] = kv;
      Vs[r * ASLD + d] = vv;
    }
    if (tid < BN) {
      int jg = j0 + tid;
      ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * H + h] : 1.f;
      vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * H + h] : 1.f;
    }
    __syncthreads();

    // ---- (1) S = scale·QKᵀ  →  P = exp(S − LSE)，存 fp32 ----
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 16, kHeadDim, E4E4>(Qs, ASLD, Ks, ASLD, acc, wr, wc, lane);
      const int r0 = wr * 32, c0 = wc * 16;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r, jg = j0 + c;
            float p = 0.f;
            if (qi < S && jg < S && !(causal && jg > qi)) {
              float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
              p = expf(sval - lse[((size_t)(b * S + qi)) * H + h]);
            }
            Ps[r * BN + c] = p;
          }
    }
    __syncthreads();

    // ---- (2) dP = dO·Vᵀ  →  dS = P∘(dP − D)，存 fp32 ----
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 16, kHeadDim, E5E4>(dOs, ASLD, Vs, ASLD, acc, wr, wc, lane);
      const int r0 = wr * 32, c0 = wc * 16;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r;
            float dpv = acc[i][j][q] * dos_s[r] * vs_s[c];
            float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
            Ss[r * BN + c] = Ps[r * BN + c] * (dpv - del);
          }
    }
    __syncthreads();

    // ---- 构造 dV/dK/dQ 的 fp8 操作数（fold 归约维上的 rowwise scale）----
    if (tid < BN) {  // Ap[j][m] = P[m][j]*dos[m]，按 j 行 rowwise (e4m3)
      int j = tid;
      float amax = 0.f;
      for (int m = 0; m < BM; ++m) amax = fmaxf(amax, fabsf(Ps[m * BN + j] * dos_s[m]));
      float sc = (amax > 0.f) ? amax / kE4M3Max : 1.f;
      sA[j] = sc;
      for (int m = 0; m < BM; ++m)
        Ap[j * QTS + m] = cvt_e4m3(Ps[m * BN + j] * dos_s[m] / sc);
    }
    if (tid < BM) {  // dS2[m][j] = dS[m][j]*ks[j]，按 m 行 rowwise (e5m2)
      int m = tid;
      float amax = 0.f;
      for (int j = 0; j < BN; ++j) amax = fmaxf(amax, fabsf(Ss[m * BN + j] * ks_s[j]));
      float sc = (amax > 0.f) ? amax / kE5M2Max : 1.f;
      sds2[m] = sc;
      for (int j = 0; j < BN; ++j)
        dS2[m * DSS2 + j] = cvt_e5m2(Ss[m * BN + j] * ks_s[j] / sc);
    }
    if (tid < BN) {  // dS3[j][m] = dS[m][j]*qs[m]，按 j 行 rowwise (e5m2)
      int j = tid;
      float amax = 0.f;
      for (int m = 0; m < BM; ++m) amax = fmaxf(amax, fabsf(Ss[m * BN + j] * qs_s[m]));
      float sc = (amax > 0.f) ? amax / kE5M2Max : 1.f;
      sds3[j] = sc;
      for (int m = 0; m < BM; ++m)
        dS3[j * QTS + m] = cvt_e5m2(Ss[m * BN + j] * qs_s[m] / sc);
    }
    __syncthreads();

    // ---- (3) dV = Pᵀ·dO  : A=Ap[j][m] (e4m3), B=dOt[d][m] 原始 do8 (e5m2) ----
    {
      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block<16, 64, BM, E4E5>(Ap, QTS, dOt, QTS, acc, wr, wc, lane);
      const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2 + (q & 1);
          int jg = j0 + r;
          if (jg < S)
            atomicAdd(dv_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim + c,
                      acc[0][j][q] * sA[r]);
        }
    }

    // ---- (4) dK = scale·dSᵀ·Q : A=dS3[j][m] (e5m2), B=Qt[d][m] 原始 q8 (e4m3) ----
    {
      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block<16, 64, BM, E5E4>(dS3, QTS, Qt, QTS, acc, wr, wc, lane);
      const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2 + (q & 1);
          int jg = j0 + r;
          if (jg < S)
            atomicAdd(dk_acc + (((size_t)(b * S + jg)) * H + h) * kHeadDim + c,
                      acc[0][j][q] * sds3[r] * scale);
        }
    }

    // ---- (5) dQ += scale·dS·K : A=dS2[m][j] (e5m2), B=Kt[d][j] 原始 k8 (e4m3) ----
    {
      float acc[2][8][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 64, BN, E5E4>(dS2, DSS2, Kt, KTS, acc, wr, wc, lane);
      const int r0 = wr * 32, c0 = wc * 64;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r;
            if (qi < S)
              atomicAdd(dq_acc + (((size_t)(b * S + qi)) * H + h) * kHeadDim + c,
                        acc[i][j][q] * sds2[r] * scale);
          }
    }
    __syncthreads();
  }
}

// =============================================================================
// 4) convert：fp32 累加缓冲 -> 输出（本版直接 fp32 拷贝）
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               float* __restrict__ dq, float* __restrict__ dk,
                               float* __restrict__ dv, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    dq[i] = dq_acc[i];
    dk[i] = dk_acc[i];
    dv[i] = dv_acc[i];
  }
}

// =============================================================================
// 极简 npy 读取（little-endian C-contiguous float32）
// =============================================================================
struct NpyF32 {
  std::vector<float> data;
  std::vector<long> shape;
};

static NpyF32 load_npy_f32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    fprintf(stderr, "无法打开 %s\n", path.c_str());
    std::exit(1);
  }
  char magic[6];
  f.read(magic, 6);
  unsigned char ver[2];
  f.read(reinterpret_cast<char*>(ver), 2);
  uint32_t hlen = 0;
  if (ver[0] == 1) {
    uint16_t h16 = 0;
    f.read(reinterpret_cast<char*>(&h16), 2);
    hlen = h16;
  } else {
    f.read(reinterpret_cast<char*>(&hlen), 4);
  }
  std::string header(hlen, '\0');
  f.read(&header[0], hlen);
  if (header.find("f4") == std::string::npos) {
    fprintf(stderr, "%s 不是 float32 npy\n", path.c_str());
    std::exit(1);
  }
  NpyF32 out;
  size_t sp = header.find("'shape'");
  size_t lp = header.find('(', sp);
  size_t rp = header.find(')', lp);
  std::string tup = header.substr(lp + 1, rp - lp - 1);
  for (size_t i = 0; i < tup.size();) {
    while (i < tup.size() && (tup[i] == ' ' || tup[i] == ',')) ++i;
    long v = 0;
    bool any = false;
    while (i < tup.size() && tup[i] >= '0' && tup[i] <= '9') {
      v = v * 10 + (tup[i] - '0');
      ++i;
      any = true;
    }
    if (any) out.shape.push_back(v);
  }
  std::streampos pos = f.tellg();
  f.seekg(0, std::ios::end);
  size_t total = (size_t)f.tellg();
  f.seekg(pos);
  size_t nbytes = total - (size_t)pos;
  out.data.resize(nbytes / sizeof(float));
  f.read(reinterpret_cast<char*>(out.data.data()), nbytes);
  return out;
}

struct DiffStat {
  double max_abs;
  double max_rel;
};

static DiffStat diff_stat(const std::vector<float>& a, const std::vector<float>& b) {
  DiffStat st{0.0, 0.0};
  for (size_t i = 0; i < a.size(); ++i) {
    double d = std::fabs((double)a[i] - (double)b[i]);
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  bool causal = true;
  int iters = 20;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdq = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) {
    fprintf(stderr, "期望 q 为 4D [B,S,H,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  if (D != kHeadDim) {
    fprintf(stderr, "本版本仅支持 head_dim=%d（当前 %d）\n", kHeadDim, D);
    return 1;
  }
  const size_t n = (size_t)B * S * H * D;
  const size_t rows = (size_t)B * S * H;
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d D=%d causal=%d scale=%.6f\n", B, S, H, D, (int)causal, scale);
  printf("FP8 mma: Q/K/V=E4M3, dO=E5M2, dS2/dS3=E5M2, Ap=E4M3 (rowwise); P/dS fp32\n");
  printf("smem = %d bytes (%.1f KB); lse smem = %d bytes (%.1f KB)\n", kSmemBytes,
         kSmemBytes / 1024.0, kLseSmemBytes, kLseSmemBytes / 1024.0);

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc, *d_dq, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_o_f, n * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, n));
  CUDA_CHECK(cudaMalloc(&d_k8, n));
  CUDA_CHECK(cudaMalloc(&d_v8, n));
  CUDA_CHECK(cudaMalloc(&d_do8, n));
  CUDA_CHECK(cudaMalloc(&d_qs, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows * 4));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dq, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, n * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, n * 4));

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), n * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), n * 4, cudaMemcpyHostToDevice));

  auto quant = [&]() {
    quantize_row_kernel<<<(int)rows, 128>>>(d_q_f, d_q8, d_qs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_k_f, d_k8, d_ks, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_v_f, d_v8, d_vs, kHeadDim, 0);
    quantize_row_kernel<<<(int)rows, 128>>>(d_do_f, d_do8, d_dos, kHeadDim, 1);
  };

  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_mma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBytes));
  dim3 pg(S, H, B);
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 mg((S + BM - 1) / BM, H, B);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((n + cvt_threads - 1) / cvt_threads, 65535);

  auto run_preprocess = [&]() {
    lse_mma_kernel<<<lg, THREADS, kLseSmemBytes>>>(d_q8, d_qs, d_k8, d_ks, d_lse, S, H,
                                                   scale, (int)causal);
    delta_kernel<<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
  };

  auto run_all = [&]() {
    quant();
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * 4));
    run_preprocess();
    fa_bwd_fp8_mma_kernel<<<mg, THREADS, kSmemBytes>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
        d_dk_acc, d_dv_acc, S, H, scale, (int)causal);
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                d_dv, n);
  };

  for (int i = 0; i < 3; ++i) run_all();
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_all();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  printf("[timing] total(quant+pre+main+cvt) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms,
         flops / (ms * 1e-3) / 1e12);

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_preprocess();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * 4));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, n * 4));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, n * 4));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    fa_bwd_fp8_mma_kernel<<<mg, THREADS, kSmemBytes>>>(
        d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
        d_dk_acc, d_dv_acc, S, H, scale, (int)causal);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main);

  std::vector<float> mdq(n), mdk(n), mdv(n);
  CUDA_CHECK(cudaMemcpy(mdq.data(), d_dq, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdk.data(), d_dk, n * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdv.data(), d_dv, n * 4, cudaMemcpyDeviceToHost));

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs fp32 ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);

  auto cmp_to = [&](const char* name, const std::vector<float>& mine, const char* fname) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(mine, t.data);
    printf("  %-4s vs %s: max_abs=%.3e  max_rel=%.3e\n", name, fname, st.max_abs,
           st.max_rel);
  };
  printf("[compare] ours vs TE FP8\n");
  cmp_to("dq", mdq, "te_dq");
  cmp_to("dk", mdk, "te_dk");
  cmp_to("dv", mdv, "te_dv");

  cudaFree(d_q_f); cudaFree(d_k_f); cudaFree(d_v_f); cudaFree(d_do_f); cudaFree(d_o_f);
  cudaFree(d_q8); cudaFree(d_k8); cudaFree(d_v8); cudaFree(d_do8);
  cudaFree(d_qs); cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_dos);
  cudaFree(d_delta); cudaFree(d_lse);
  cudaFree(d_dq_acc); cudaFree(d_dk_acc); cudaFree(d_dv_acc);
  cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_dv);
  return 0;
}
