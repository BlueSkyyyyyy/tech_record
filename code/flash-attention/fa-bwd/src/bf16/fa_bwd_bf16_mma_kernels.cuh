// =============================================================================
// fa_bwd_bf16_mma_kernels.cuh —— bf16 张量核反向（O5b）**两文件版的 device 部分**
// =============================================================================
// 由 fp16 张量核版（O5）`fa_bwd_fp16_mma_kernels.cuh` 做 **dtype 参数化**而来：
//   `__half` → `__nv_bfloat16`、`mma...f16.f16` → `mma...bf16.bf16`、
//   `__half2float/__float2half` → `__bfloat162float/__float2bfloat16`。
// 算法数据流、线程映射、smem 布局、padding（LD=HD+8/LDP=BM+8/LDS=BN+8）、
// dQ 寄存器累加、dK/dV `red_add2`（float2 atomicAdd）全部与 fp16 版**逐字同构**。
// host 侧见 `fa_bwd_bf16_mma_main.cu`。
//
// ----------（以下为 fp16 版原始说明，bf16 版完全适用）----------
// FlashAttention 反向（bf16）**张量核版**（O5b）
// 背景：P2 的 `fa_bwd_bf16_onefile.cu` 是**正确性优先的标量 golden**（CUDA-core FFMA），
// main 只有 ~1 TFLOPS。O5 已把 fp16 反向换成张量核（main 11.8–14.9×），本文件把同一
// 后端移植到 bf16：5 个 GEMM 全部用 `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
// + `ldmatrix`，数据流保持 FA2 的 1colblock（recompute P、D 预处理、dQ/dK/dV 归约）。
//
// ----- 5 个 GEMM 的 mma 布局映射（HD=128, BM=64, BN=32）-----
//   1) S  = scale·QKᵀ : A=Q [BM][HD], B=K [BN][HD]（[N][K]，非转置）
//   2) dP = dO·Vᵀ     : A=dO[BM][HD], B=V [BN][HD]（非转置）
//   3) dV = Pᵀ·dO     : A=PsT[BN][BM]（P 转置存), B=dO[BM][HD]（[K][N]，转置）
//   4) dK = scale·dSᵀ·Q: A=dSsT[BN][BM]（dS 转置), B=Q [BM][HD]（转置）
//   5) dQ = scale·dS·K : A=dSs[BM][BN], B=K [BN][HD]（转置）
// 与 fp8 张量核版（P3-4）同构，但没有 rowwise scale（bf16 无量化），也就没有 fold；
// P/dS 直接按 FA2 的做法转成 bf16 当 A 操作数（dS 用 fp32 算完再转 bf16）。
//
// ----- 与 fp8 版的关键差异 -----
//   * mma 是 m16n8k16（K_TILE/16 步），A 每段 +8 bf16、B 每段 +8 bf16；ldmatrix 行距
//     按 **元素**（bf16）计；padding 8 个元素（16B）即可消 ldmatrix bank conflict。
//   * 无 FP8 转换、无 Ap/dS2/dS3 折算操作数；PsT/dSsT/dSs 都是纯 bf16。
//   * dQ 无 split-K（grid=S/BM × H × B），每个 Q 块由唯一 CTA 独占 → dQ **寄存器累加**
//     后直接写 `dq_acc`（不需要跨 CTA atomic）；dK/dV 仍跨 CTA `atomicAdd`（float2，O4c）。
//   * bf16 版 head_dim 目前只做 **HD=128**（MHA/GQA）；MLA(HD=512) 的张量核留 backlog。
//
// 用法：run.sh src/bf16/fa_bwd_bf16_mma_onefile.cu [--dir=...] [--full|--causal] [--iters=N]
// =============================================================================

#ifndef FA_BWD_BF16_MMA_KERNELS_CUH_
#define FA_BWD_BF16_MMA_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ----------------------------- 编译期常量 -----------------------------
static constexpr int THREADS = 128;   // 4 warps
static constexpr int WN      = 2;     // N 方向 warp 数（2×2 warp 网格）

// =============================================================================
// mma / ldmatrix（smoke 已在 fp16 版验证三种操作数布局；bf16 位宽/布局同构）
// =============================================================================
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
__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4],
                                         const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// O4c 风格向量化归约：mma.m16n8 累加器里 q/q+1 两列相邻且同 row → 一次 float2 atomicAdd。
__device__ __forceinline__ void red_add2(float* p, float a, float b) {
  atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));
}

// ---- O6：cp.async 异步拷贝（16B）----
// 把「全局→smem」的 K/V 搬运从「同步 LDG + STS」改成硬件异步流水：`cp.async.cg` 走
// L2-only 路径（流式数据不污染 L1），发起后立即返回、不占寄存器、不阻塞发射；
// 用 `commit_group` 打组、`wait_group 0` 在消费前统一等待。这是消 `long_scoreboard`
// （O5b ncu：全局访存延迟占 ~63%）的标准手段（对齐 fp8 的 O3，但 fp8 因字节少用寄存器预取，
// fp16/bf16 字节翻倍 → 改用 cp.async 双缓冲 smem，避免 32 个额外寄存器的代价）。
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
  uint32_t s = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}

// O6：把一个 K/V 列块（BN 行 × HD 列，bf16）用 cp.async 发进 smem。
// 以 8 个 bf16（16B）为最小搬运单位 → 每行 HD/8 个 unit，共 BN*HD/8 个，按 THREADS 均分。
// 行越界（jg>=S）用普通 smem 写 0（cp.async 无谓词，混合写 + 后续 __syncthreads 可见）。
// O6b：DOK/DOV 分别控制是否发 K / V，便于把二者拆到不同 commit_group、在不同时点预取。
//   * O6  ：DOK=DOV=true（一次发 K+V）。
//   * O6b ：K 双缓冲、循环首预取（DOK=true,DOV=false）；V 单缓冲、GEMM2 之后才发下一 tile。
template <int HD, int BN, bool DOK = true, bool DOV = true>
__device__ __forceinline__ void kv_issue_async(const bf16* __restrict__ k,
                                               const bf16* __restrict__ v, int j0, int S,
                                               int Hkv, int hkv, int b, int tid, bf16* Kd,
                                               bf16* Vd, int LD) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 bf16) 数
  constexpr int NU  = BN * HDV;  // 总 unit 数
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int jg = j0 + row;
    if (jg < S) {
      const size_t off = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c8 * 8;
      if (DOK) cp_async16(Kd + row * LD + c8 * 8, k + off);
      if (DOV) cp_async16(Vd + row * LD + c8 * 8, v + off);
    } else {
      if (DOK) *reinterpret_cast<uint4*>(Kd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      if (DOV) *reinterpret_cast<uint4*>(Vd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// A[M_TILE][K_TILE] 行主序（行距 asld，bf16）；B 两种布局：
//   BTRANS=false：Bs=[N_TILE][K_TILE] 行主序（行距 bsld，bf16）→ ldmatrix.x2；
//   BTRANS=true ：Bs=[K_TILE][N_TILE] 行主序（行距 bsld，bf16）→ ldmatrix.x2.trans。
// ATRANS=true ：As=[K_TILE][M_TILE] 行主序 → ldmatrix.x4.trans（bit3/bit4 互换），
//   用于 O6b 免掉 `PsT/dSsT` 转置副本（fp16 版 smoke 验证逐位一致，bf16 位宽同构）。
template <int WARP_M, int WARP_N, int K_TILE, bool BTRANS, bool ATRANS = false>
__device__ __forceinline__ void mma_block_bf16(const bf16* As, int asld,
                                               const bf16* Bs, int bsld,
                                               float acc[WARP_M / 16][WARP_N / 8][4],
                                               int wm, int wn, int lane) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
#pragma unroll
  for (int kk = 0; kk < K_TILE / 16; ++kk) {
    const int koff = kk * 16;
    uint32_t av[MTM][4];
    if constexpr (ATRANS) {
      const int krow = (lane & 7) + ((lane >> 4) & 1) * 8;
      const int mcol = ((lane >> 3) & 1) * 8;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        ldmatrix_x4_trans(smem_u32(As + (koff + krow) * asld + (wm * WARP_M + i * 16 + mcol)),
                          av[i]);
    } else {
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int acol = (lane >> 4) * 8;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        ldmatrix_x4(smem_u32(As + (wm * WARP_M + i * 16 + arow) * asld + koff + acol), av[i]);
    }
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      uint32_t d[2];
      if (BTRANS) {
        const int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
        ldmatrix_x2_trans(smem_u32(Bs + (koff + krow) * bsld + wn * WARP_N + j * 8), d);
      } else {
        const int brow = lane & 7;
        const int bcol = ((lane >> 3) & 1) * 8;
        ldmatrix_x2(smem_u32(Bs + (wn * WARP_N + j * 8 + brow) * bsld + koff + bcol), d);
      }
      bv[j][0] = d[0];
      bv[j][1] = d[1];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) mma_bf16(acc[i][j], av[i], bv[j]);
  }
}

// =============================================================================
// 1) preprocess（O8：mma 分块 LSE + 独立 delta）——替代旧标量 preprocess_kernel
// =============================================================================
// 与 fp16 版（O8）逐字同构：旧版每 (s,h) 行一个 block、标量扫 K，S=4096 ~69ms 是
// 端到端第一瓶颈。改为 `lse_mma_kernel`（mma.m16n8k16 QKᵀ + online-softmax + warp 内
// shfl 归约，LBM=64 行 × LBN=64 列/CTA）+ 独立 `delta_kernel`。数学与旧版一致。
static constexpr int LBM = 64;   // LSE CTA 的 Q 行数
static constexpr int LBN = 64;   // LSE 一次吃的 K 列数

template <int HD>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
               float* __restrict__ lse, int S, int H, int Hkv, float scale, int causal) {
  constexpr int LD = HD + 8;
  extern __shared__ __align__(16) char smem[];
  bf16* Qs = reinterpret_cast<bf16*>(smem);
  bf16* Ks = Qs + LBM * LD;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * LBM;

  for (int i = tid; i < LBM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    Qs[r * LD + d] =
        (qi < S) ? q[(((size_t)(b * S + qi)) * H + h) * HD + d] : __float2bfloat16(0.f);
  }
  __syncthreads();

  const int ncols = causal ? min(S, m0 + LBM) : S;
  const int ntiles = (ncols + LBN - 1) / LBN;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * LBN;
    for (int i = tid; i < LBN * HD; i += THREADS) {
      int r = i / HD, d = i % HD;
      int jg = j0 + r;
      Ks[r * LD + d] =
          (jg < S) ? k[(((size_t)(b * S + jg)) * Hkv + hkv) * HD + d] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // S_tile = Q·Kᵀ（bf16×bf16→fp32），每 warp 16×64
    float acc[1][8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
    mma_block_bf16<16, LBN, HD, false>(Qs, LD, Ks, LD, acc, wid, 0, lane);

    // online-softmax：每个线程持有 2 个 row-slot（q<2 / q>=2），沿 N 就地更新
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        int s = q >= 2 ? 1 : 0;
        int r = wid * 16 + g + (q >= 2 ? 8 : 0);
        int c = j * 8 + c2 + (q & 1);
        int qi = m0 + r, jg = j0 + c;
        float sv = -INFINITY;
        if (qi < S && jg < S && !(causal && jg > qi)) sv = acc[0][j][q] * scale;
        if (sv != -INFINITY) {
          float mn = fmaxf(mrow[s], sv);
          lrow[s] = lrow[s] * expf(mrow[s] - mn) + expf(sv - mn);
          mrow[s] = mn;
        }
      }
    __syncthreads();
  }

  // 同一 row 由 4 个 lane（同 g、lane&3=0..3）持有，warp 内 shfl 归约
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

// delta_kernel：D = rowsum(dO ∘ O)（纯 O(S·H·D) 逐行归约，与 LSE 解耦）
template <int HD>
__global__ void delta_kernel(const bf16* __restrict__ o,
                             const bf16* __restrict__ do_, float* __restrict__ delta,
                             int S, int H) {
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const bf16* orow = o + row * HD;
  const bf16* dorow = do_ + row * HD;
  float dp = 0.f;
  for (int d = tid; d < HD; d += blockDim.x)
    dp += __bfloat162float(orow[d]) * __bfloat162float(dorow[d]);
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
// 2) main kernel（张量核）：1colblock 反向，5 个 GEMM 全 mma.m16n8k16
// =============================================================================
// smem 布局（bf16，除注明外）：
//   PIPE!=2：Qs[BM*LD] + dOs[BM*LD] + Ks[KSB] + Vs[VSB]
//     + PsT[BN*LDP]（P 转置，GEMM3 A） + dSs[BM*LDS]（GEMM5 A） + dSsT[BN*LDP]（GEMM4 A）
//   PIPE==2：Qs[BM*LD] + dOs[BM*LD] + Ks[2*KVL] + Vs[KVL]
//     + Ps[BM*LDS]（P，GEMM3 A 用 trans 读） + dSs[BM*LDS]（GEMM4 A trans / GEMM5 A 普通）
//   其中 LD=HD+8、LDP=BM+8、LDS=BN+8（+8 消 ldmatrix bank conflict）。
// PIPE=0：O5b 原版（同步标量 K/V 载入，3 CTA/SM）。
// PIPE=1：O6（cp.async 双缓冲 K/V，消全局访存延迟；smem 翻倍 → 2 CTA/SM）。
// PIPE=2：O6b（cp.async 只双缓冲 K，V 单缓冲且 GEMM2 后预取 + A 转置读免掉 PsT/dSsT）。
//   与 fp16 版逐字同构；smem 71.2KB → 3 CTA/SM。
template <int HD, int BM, int BN, int PIPE>
__global__ void __launch_bounds__(THREADS, PIPE == 1 ? 2 : 3)
fa_bwd_bf16_mma_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                       const bf16* __restrict__ v, const bf16* __restrict__ do_,
                       const float* __restrict__ delta, const float* __restrict__ lse,
                       float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                       float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                       int causal) {
  static_assert(HD == 128, "O5b bf16 mma 目前只支持 head_dim=128");
  constexpr int LD  = HD + 8;    // Q/K/V/dO 行距（bf16）
  constexpr int LDP = BM + 8;    // PsT/dSsT 行距（bf16，PIPE!=2）
  constexpr int LDS = BN + 8;    // Ps/dSs 行距（bf16，[BM][BN] 布局）
  constexpr int KVL = BN * LD;                     // 单个 K 或 V 缓冲（bf16）
  constexpr int KSB = PIPE >= 1 ? 2 * KVL : KVL;   // K 缓冲（O6/O6b 双缓冲）
  constexpr int VSB = PIPE == 1 ? 2 * KVL : KVL;   // V 缓冲（只有 O6 双缓冲）
  constexpr int PSZ = (PIPE == 2) ? BM * LDS : BN * LDP;  // P 存储大小

  extern __shared__ __align__(16) char smem[];
  bf16* Qs   = reinterpret_cast<bf16*>(smem);
  bf16* dOs  = Qs + BM * LD;
  bf16* Ks   = dOs + BM * LD;
  bf16* Vs   = Ks + KSB;
  bf16* Ps   = Vs + VSB;              // PIPE==2：[BM][LDS]；否则 [BN][LDP]（=PsT）
  bf16* dSs  = Ps + PSZ;              // 始终 [BM][LDS]
  bf16* dSsT = dSs + BM * LDS;        // PIPE!=2 的 dS 转置副本（PIPE==2 不用）

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  // ---- 载入 Q/dO（越界补 0）----
  for (int i = tid; i < BM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    bf16 qv = __float2bfloat16(0.f), ov = __float2bfloat16(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * HD + d;
      qv = q[idx];
      ov = do_[idx];
    }
    Qs[r * LD + d] = qv;
    dOs[r * LD + d] = ov;
  }

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;

  if constexpr (PIPE == 0) {
    __syncthreads();
  } else if constexpr (PIPE == 1) {
    // O6：prologue 直接异步发起 tile0 的 K/V（不占寄存器）；Q/dO 的可见性由
    // 循环首的 `wait_group + __syncthreads` 一并保证（Q/dO 与 K/V 写不同 smem）。
    if (ntiles > 0)
      kv_issue_async<HD, BN, true, true>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs, LD);
  } else {
    // O6b：tile0 的 K 发进双缓冲 stage0、V 发进单缓冲 Vs（两个独立 commit_group）。
    if (ntiles > 0) {
      kv_issue_async<HD, BN, true, false>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs, LD);
      kv_issue_async<HD, BN, false, true>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs, LD);
    }
  }

  // dQ 沿 nt 在寄存器里累加（每个 Q 块唯一 CTA，无需跨 CTA atomic）。
  float dqacc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    // O6/O6b：本 tile 的 K 用 stage = nt&1；PIPE=0 时 stage 恒 0（布局退化为原版）。
    const int stage = (PIPE >= 1) ? (nt & 1) : 0;
    bf16* Kt = Ks + stage * KVL;
    bf16* Vt = (PIPE == 1) ? Vs + stage * KVL : Vs;

    if constexpr (PIPE == 1) {
      // 等本 tile 的 cp.async 落地；此 barrier 同时保证「上一 tile 的 GEMM5 已读完
      // 那个 stage」，故随后把下一 tile 发进该 stage 是安全的。
      asm volatile("cp.async.wait_group 0;\n");
      __syncthreads();
      if (nt + 1 < ntiles)
        kv_issue_async<HD, BN, true, true>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                               Ks + ((nt + 1) & 1) * KVL, Vs + ((nt + 1) & 1) * KVL, LD);
    } else if constexpr (PIPE == 2) {
      // O6b：等 K[nt] 与上一轮发出的 V[nt] 落地；barrier 同时证明上一 tile 的 GEMM5
      // 已读完 Ks 的另一 stage，故可把 K[nt+1] 发进该 stage。V 单缓冲，见下。
      asm volatile("cp.async.wait_group 0;\n");
      __syncthreads();
      if (nt + 1 < ntiles)
        kv_issue_async<HD, BN, true, false>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                               Ks + ((nt + 1) & 1) * KVL, Vs, LD);
    } else {
      // ---- 载入 K/V 块（原版：同步标量读）----
      for (int i = tid; i < BN * HD; i += THREADS) {
        int r = i / HD, d = i % HD;
        int jg = j0 + r;
        bf16 kv = __float2bfloat16(0.f), vv = __float2bfloat16(0.f);
        if (jg < S) {
          size_t idx = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d;
          kv = k[idx];
          vv = v[idx];
        }
        Kt[r * LD + d] = kv;
        Vt[r * LD + d] = vv;
      }
      __syncthreads();
    }

    // ---- (1) S = scale·QKᵀ → P = exp(S − LSE) ----
    // 同一线程在 GEMM1/GEMM2 的 (r,c) 映射一致，故用寄存器 pval 保存 P 供 dS 用，
    // 同时把 P 转 bf16 写进转置布局 PsT（供 GEMM3 的 A）。
    float pval[2][2][4];
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bf16<32, 16, HD, false>(Qs, LD, Kt, LD, acc, wr, wc, lane);
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
            if (qi < S && jg < S && !(causal && jg > qi))
              p = expf(acc[i][j][q] * scale - lse[((size_t)(b * S + qi)) * H + h]);
            pval[i][j][q] = p;
            if constexpr (PIPE == 2)
              Ps[r * LDS + c] = __float2bfloat16(p);   // [BM][BN]，GEMM3 用 trans 读
            else
              Ps[c * LDP + r] = __float2bfloat16(p);   // [BN][BM]（PIPE!=2）
          }
    }

    // ---- (2) dP = dO·Vᵀ → dS = P∘(dP − D)（同 warp/累加器映射）----
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bf16<32, 16, HD, false>(dOs, LD, Vt, LD, acc, wr, wc, lane);
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
            float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
            float ds = pval[i][j][q] * (acc[i][j][q] - del);
            dSs[r * LDS + c] = __float2bfloat16(ds);
            if constexpr (PIPE != 2)
              dSsT[c * LDP + r] = __float2bfloat16(ds);
          }
    }
    __syncthreads();
    if constexpr (PIPE == 2) {
      // GEMM2 是 V 的唯一消费者；barrier 后把 V[nt+1] 发进单缓冲，延迟由 GEMM3/4/5 盖住。
      if (nt + 1 < ntiles)
        kv_issue_async<HD, BN, false, true>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                            Ks, Vs, LD);
    }

    // ---- (3) dV = Pᵀ·dO（A=PsT[BN][BM], B=dO[BM][HD] 转置）----
    {
      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      if constexpr (PIPE == 2)
        mma_block_bf16<16, 64, BM, true, true>(Ps, LDS, dOs, LD, acc, wr, wc, lane);
      else
        mma_block_bf16<16, 64, BM, true>(Ps, LDP, dOs, LD, acc, wr, wc, lane);
      const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2;
          int jg = j0 + r;
          if (jg < S)
            red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     acc[0][j][q], acc[0][j][q + 1]);
        }
    }

    // ---- (4) dK = scale·dSᵀ·Q（A=dSsT[BN][BM], B=Q[BM][HD] 转置）----
    {
      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      if constexpr (PIPE == 2)
        mma_block_bf16<16, 64, BM, true, true>(dSs, LDS, Qs, LD, acc, wr, wc, lane);
      else
        mma_block_bf16<16, 64, BM, true>(dSsT, LDP, Qs, LD, acc, wr, wc, lane);
      const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2;
          int jg = j0 + r;
          if (jg < S)
            red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     acc[0][j][q] * scale, acc[0][j][q + 1] * scale);
        }
    }

    // ---- (5) dQ += scale·dS·K（A=dSs[BM][BN], B=K[BN][HD] 转置）----
    {
      float acc[2][8][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bf16<32, 64, BN, true>(dSs, LDS, Kt, LD, acc, wr, wc, lane);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dqacc[i][j][q] += acc[i][j][q] * scale;
          }
    }
    // PIPE=0 需要尾 barrier；PIPE=1/2 由下轮循环首的 barrier 承担。
    if constexpr (PIPE == 0) __syncthreads();
  }

  // ---- 写回 dQ（寄存器累加结果，直接存；每个 Q 块由唯一 CTA 负责）----
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; q += 2) {
        int r = wr * 32 + i * 16 + g + (q >= 2 ? 8 : 0);
        int c = wc * 64 + j * 8 + c2;
        int qi = m0 + r;
        if (qi < S) {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          base[0] = dqacc[i][j][q];
          base[1] = dqacc[i][j][q + 1];
        }
      }
}

// =============================================================================
// 3) convert：fp32 累加缓冲 → bf16 输出
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc, bf16* __restrict__ dq,
                               bf16* __restrict__ dk, bf16* __restrict__ dv, size_t n_q,
                               size_t n_kv) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_q;
       i += (size_t)gridDim.x * blockDim.x)
    dq[i] = __float2bfloat16(dq_acc[i]);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_kv;
       i += (size_t)gridDim.x * blockDim.x) {
    dk[i] = __float2bfloat16(dk_acc[i]);
    dv[i] = __float2bfloat16(dv_acc[i]);
  }
}


#endif  // FA_BWD_BF16_MMA_KERNELS_CUH_
