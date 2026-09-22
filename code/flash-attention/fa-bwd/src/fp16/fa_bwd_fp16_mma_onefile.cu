// =============================================================================
// fa_bwd_fp16_mma_onefile.cu —— FlashAttention 反向（fp16）**张量核单文件版**（O5）
// =============================================================================
// 背景：P1 的 `fa_bwd_fp16_onefile.cu` 是**正确性优先的标量 golden**（CUDA-core FFMA），
// 实测 main 只有 ~1 TFLOPS（FA3 ~850、TE ~618）。ROADMAP 的 O5（最大杠杆）就是把它换成
// 张量核：5 个 GEMM 全部用 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` +
// `ldmatrix`，数据流保持 FA2 的 1colblock（recompute P、D 预处理、dQ/dK/dV 归约）。
//
// 前置：`fa_bwd_fp16_mma_smoke.cu` 已验证三种操作数布局（A=[M][K] + ldmatrix.x4；
// B=[N][K] + ldmatrix.x2；B=[K][N] + ldmatrix.x2.trans）与 CPU 参考一致（PASS）。
//
// ----- 5 个 GEMM 的 mma 布局映射（HD=128, BM=64, BN=32）-----
//   1) S  = scale·QKᵀ : A=Q [BM][HD], B=K [BN][HD]（[N][K]，非转置）
//   2) dP = dO·Vᵀ     : A=dO[BM][HD], B=V [BN][HD]（非转置）
//   3) dV = Pᵀ·dO     : A=PsT[BN][BM]（P 转置存), B=dO[BM][HD]（[K][N]，转置）
//   4) dK = scale·dSᵀ·Q: A=dSsT[BN][BM]（dS 转置), B=Q [BM][HD]（转置）
//   5) dQ = scale·dS·K : A=dSs[BM][BN], B=K [BN][HD]（转置）
// 与 fp8 张量核版（P3-4）同构，但没有 rowwise scale（fp16 无量化），也就没有 fold；
// P/dS 直接按 FA2 的做法转成 half 当 A 操作数（dS 用 fp32 算完再转 half）。
//
// ----- 与 fp8 版的关键差异 -----
//   * mma 是 m16n8k16（K_TILE/16 步），A 每段 +8 half、B 每段 +8 half；ldmatrix 行距
//     按 **元素**（half）计；padding 8 个元素（16B）即可消 ldmatrix bank conflict。
//   * 无 FP8 转换、无 Ap/dS2/dS3 折算操作数；PsT/dSsT/dSs 都是纯 half。
//   * dQ 无 split-K（grid=S/BM × H × B），每个 Q 块由唯一 CTA 独占 → dQ **寄存器累加**
//     后直接写 `dq_acc`（不需要跨 CTA atomic）；dK/dV 仍跨 CTA `atomicAdd`（float2，O4c）。
//   * fp16 版 head_dim 目前只做 **HD=128**（MHA/GQA）；MLA(HD=512) 的张量核留 backlog。
//
// 用法：run.sh src/fp16/fa_bwd_fp16_mma_onefile.cu [--dir=...] [--full|--causal] [--iters=N]
// =============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
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
static constexpr int THREADS = 128;   // 4 warps
static constexpr int WN      = 2;     // N 方向 warp 数（2×2 warp 网格）

// =============================================================================
// mma / ldmatrix（与 smoke 完全一致）
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
__device__ __forceinline__ void mma_f16(float c[4], const uint32_t a[4],
                                        const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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
// （O5 ncu：全局访存延迟占 ~63%）的标准手段（对齐 fp8 的 O3，但 fp8 因字节少用寄存器预取，
// fp16 字节翻倍 → 改用 cp.async 双缓冲 smem，避免 32 个额外寄存器的代价）。
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
  uint32_t s = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}

// O6：把一个 K/V 列块（BN 行 × HD 列，half）用 cp.async 发进 smem。
// 以 8 个 half（16B）为最小搬运单位 → 每行 HD/8 个 unit，共 BN*HD/8 个，按 THREADS 均分。
// 行越界（jg>=S）用普通 smem 写 0（cp.async 无谓词，混合写 + 后续 __syncthreads 可见）。
// O6b：DOK/DOV 分别控制是否发 K / V，便于把二者拆到不同 commit_group、在不同时点预取。
//   * O6  ：DOK=DOV=true（一次发 K+V）。
//   * O6b ：K 用双缓冲、在循环首预取（DOK=true,DOV=false）；V 只单缓冲，在 GEMM2 之后
//            （V 的最后一次使用）才发下一 tile（DOK=false,DOV=true），省下一整个 V 缓冲。
template <int HD, int BN, bool DOK = true, bool DOV = true>
__device__ __forceinline__ void kv_issue_async(const __half* __restrict__ k,
                                               const __half* __restrict__ v, int j0, int S,
                                               int Hkv, int hkv, int b, int tid, __half* Kd,
                                               __half* Vd, int LD) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 half) 数
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

// A[M_TILE][K_TILE] 行主序（行距 asld，half）；B 两种布局：
//   BTRANS=false：Bs=[N_TILE][K_TILE] 行主序（行距 bsld，half）→ ldmatrix.x2；
//   BTRANS=true ：Bs=[K_TILE][N_TILE] 行主序（行距 bsld，half）→ ldmatrix.x2.trans。
// ATRANS=false：As=[M_TILE][K_TILE] 行主序（行距 asld）→ ldmatrix.x4（常态）。
// ATRANS=true ：As=[K_TILE][M_TILE] 行主序（即 A 的转置副本）→ ldmatrix.x4.trans，
//   地址的 bit3/bit4 互换（见 `fa_bwd_fp16_atrans_smoke.cu` 的逐位验证），
//   这样 P/dS 只需存一份 [BM][BN]，省掉一份 `PsT/dSsT`（O6b）。
template <int WARP_M, int WARP_N, int K_TILE, bool BTRANS, bool ATRANS = false>
__device__ __forceinline__ void mma_block_f16(const __half* As, int asld,
                                              const __half* Bs, int bsld,
                                              float acc[WARP_M / 16][WARP_N / 8][4],
                                              int wm, int wn, int lane) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
#pragma unroll
  for (int kk = 0; kk < K_TILE / 16; ++kk) {
    const int koff = kk * 16;
    uint32_t av[MTM][4];
    if constexpr (ATRANS) {
      // As=[K][M]：krow 选 K 行（bit4→k8-15），mcol 选 M 列（bit3→m8-15）。
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
      for (int j = 0; j < MTN; ++j) mma_f16(acc[i][j], av[i], bv[j]);
  }
}

// =============================================================================
// 1) preprocess（O8：mma 分块 LSE + 独立 delta）——替代旧标量 preprocess_kernel
// =============================================================================
// 旧版每 (s,h) 行一个 block、128 线程标量扫 K，每对 (i,j) 做 128 次 FFMA；S=4096 时
// ~68ms，是端到端第一瓶颈（O5 main 才 4.5ms）。O8 照搬 fp8 的 O1：
//   * LSE 用 `lse_mma_kernel`：与主 kernel 同源的 QKᵀ 张量核（mma.m16n8k16），每 CTA 吃
//     LBM=64 行 Q × LBN=64 列 K，4 warp 各 16 行；P 不物化，accumulator 直接做
//     online-softmax，行 max/sum 在同 row 的 4 个 lane 间 `shfl_xor` 归约。
//   * delta=rowsum(dO∘O) 拆成独立 `delta_kernel`（纯 O(S·H·D) 归约，与 LSE 解耦）。
// 数学与旧版一致（fp16 乘积在 fp32 里累加），且 QKᵀ 的 k-loop 分块顺序与 main GEMM1
// 完全相同 ⇒ LSE 与 main 的 P 自洽，数值只会更稳。
static constexpr int LBM = 64;   // LSE CTA 的 Q 行数
static constexpr int LBN = 64;   // LSE 一次吃的 K 列数

template <int HD>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
               float* __restrict__ lse, int S, int H, int Hkv, float scale, int causal) {
  constexpr int LD = HD + 8;
  extern __shared__ __align__(16) char smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);
  __half* Ks = Qs + LBM * LD;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * LBM;

  for (int i = tid; i < LBM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    Qs[r * LD + d] =
        (qi < S) ? q[(((size_t)(b * S + qi)) * H + h) * HD + d] : __float2half(0.f);
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
          (jg < S) ? k[(((size_t)(b * S + jg)) * Hkv + hkv) * HD + d] : __float2half(0.f);
    }
    __syncthreads();

    // S_tile = Q·Kᵀ（f16×f16→fp32），每 warp 16×64
    float acc[1][8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
    mma_block_f16<16, LBN, HD, false>(Qs, LD, Ks, LD, acc, wid, 0, lane);

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

// =============================================================================
// O8b：LSE 预处理的两处优化（fp16，causal 专用）
// =============================================================================
// 动机（O8 ncu，S=4096）：`lse_mma_kernel` 1.03ms，占端到端 ~34%；SOL Compute 39.5%、
// L1 23%、DRAM 1%，stall `long_scoreboard 2.19 + wait 1.34`（延迟受限）；且因果下工作量
// 从第 0 块的 1 个 K tile 到第 63 块的 64 个 tile 线性增长，GPU 按 blockIdx 递增调度把重块
// 全排在最后，ncu 报「尾波可达 50%」。两处改动（用模板 `PIPE` 分开，便于消融）：
//   ① **镜像配对负载均衡**：每 CTA 同时处理配对的两个 m 块 `m` 与 `nblk-1-m`，工作量恒为
//      `nblk+1` 个 tile（完美均衡），grid.x 从 nblk 减半到 ceil(nblk/2)。
//   ② **K 的 cp.async.cg 双缓冲**（PIPE=1）：每 tile 的 K 从「同步标量读→sync→mma」改成
//      16B 异步预取下一块（不占寄存器、走 L2-only），消掉全局访存延迟（对齐 main 的 O6）。
//      PIPE=0 则退回「同步标量读 K」，用于和 PIPE=1 做同 session 消融。
// 数学与 O8 完全一致（同一 QKᵀ mma、online-softmax、同 row 4-lane `shfl_xor` 归约），
// 数值应逐位相同。仅用于 causal；非 causal 各块工作量相同，无需配对（host 走 O8 原版）。
// smem：Qs[LBM*LD] +（PIPE=1 时 2 份 / PIPE=0 时 1 份）Ks[LBN*LD]。
template <int HD, int PIPE>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal(const __half* __restrict__ q, const __half* __restrict__ k,
                   float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  constexpr int LD  = HD + 8;
  constexpr int KVL = LBN * LD;
  constexpr int HDV = HD / 8;   // 每行 uint4(8 half) 数
  extern __shared__ __align__(16) char smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);
  __half* Ks = Qs + LBM * LD;   // PIPE=1：2 × LBN × LD；PIPE=0：1 × LBN × LD

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // 把 K 的一个 tile（j0 起 LBN 行）写进 Kd：PIPE=1 用 16B cp.async（行越界写 0）；
  // PIPE=0 用普通标量 smem 写（随后由调用处的 __syncthreads 保证可见）。
  auto issue_k = [&](__half* Kd, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c8 = u % HDV;
      const int jg = j0 + row;
      if (jg < S) {
        const size_t off = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c8 * 8;
        if constexpr (PIPE) {
          cp_async16(Kd + row * LD + c8 * 8, k + off);
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) Kd[row * LD + c8 * 8 + e] = k[off + e];
        }
      } else {
        *reinterpret_cast<uint4*>(Kd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      }
    }
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) continue;  // 奇数 nblk 的中心块只做一次
    const int m0 = mblk * LBM;

    // ---- 载入本 m 块的 Q（越界补 0）----
    for (int i = tid; i < LBM * HD; i += THREADS) {
      int r = i / HD, d = i % HD;
      int qi = m0 + r;
      Qs[r * LD + d] =
          (qi < S) ? q[(((size_t)(b * S + qi)) * H + h) * HD + d] : __float2half(0.f);
    }

    const int ncols = min(S, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;

    // prologue：PIPE=1 时发 tile0 进 stage0（Q 与 K 写不同 smem，由循环首 wait+sync 保证可见）
    if constexpr (PIPE) {
      if (ntiles > 0) issue_k(Ks, 0);
    }

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int nt = 0; nt < ntiles; ++nt) {
      const int j0 = nt * LBN;
      __half* Kt = Ks + (PIPE ? (nt & 1) * KVL : 0);
      if constexpr (PIPE) {
        // 等本 tile 落地；此 barrier 同时证明「上一 tile 的 mma 已读完其 stage」，故可复用。
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (nt + 1 < ntiles) issue_k(Ks + ((nt + 1) & 1) * KVL, j0 + LBN);
      } else {
        issue_k(Ks, j0);
        __syncthreads();
      }

      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block_f16<16, LBN, HD, false>(Qs, LD, Kt, LD, acc, wid, 0, lane);

#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int s = q >= 2 ? 1 : 0;
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float sv = -INFINITY;
          if (qi < S && jg < S && jg <= qi) sv = acc[0][j][q] * scale;
          if (sv != -INFINITY) {
            float mn = fmaxf(mrow[s], sv);
            lrow[s] = lrow[s] * expf(mrow[s] - mn) + expf(sv - mn);
            mrow[s] = mn;
          }
        }
    }

    // 同一 row 由同 g 的 4 个 lane 持有，warp 内 shfl 归约
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
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）
    __syncthreads();
  }
}

// delta_kernel：D = rowsum(dO ∘ O)（纯 O(S·H·D) 逐行归约，与 LSE 解耦）
template <int HD>
__global__ void delta_kernel(const __half* __restrict__ o,
                             const __half* __restrict__ do_, float* __restrict__ delta,
                             int S, int H) {
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const __half* orow = o + row * HD;
  const __half* dorow = do_ + row * HD;
  float dp = 0.f;
  for (int d = tid; d < HD; d += blockDim.x)
    dp += __half2float(orow[d]) * __half2float(dorow[d]);
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
// smem 布局（half，除注明外）：
//   PIPE!=2：Qs[BM*LD] + dOs[BM*LD] + Ks[KSB] + Vs[VSB]
//     + PsT[BN*LDP]（P 转置，GEMM3 A） + dSs[BM*LDS]（GEMM5 A） + dSsT[BN*LDP]（GEMM4 A）
//   PIPE==2：Qs[BM*LD] + dOs[BM*LD] + Ks[2*KVL] + Vs[KVL]
//     + Ps[BM*LDS]（P，GEMM3 A 用 trans 读） + dSs[BM*LDS]（GEMM4 A trans / GEMM5 A 普通）
//   其中 LD=HD+8、LDP=BM+8、LDS=BN+8（+8 消 ldmatrix bank conflict）。
// PIPE=0：O5 原版（同步标量 K/V 载入，3 CTA/SM）。
// PIPE=1：O6（cp.async 双缓冲 K/V，消全局访存延迟；smem 翻倍 → 2 CTA/SM）。
// PIPE=2：O6b（cp.async **只双缓冲 K**，V 单缓冲且在 GEMM2 后预取）+ **A 转置读**：
//   * V 只在 GEMM2 里被读、且 GEMM2 在 tile 前段 ⇒ 发完 GEMM2（下方的 __syncthreads 之后）
//     就把下一 tile 的 V 发进同一缓冲，靠 GEMM3/4/5 的时间把延迟盖住。K 全程被 GEMM1/GEMM5
//     使用，必须双缓冲。
//   * `PsT/dSsT` 两份转置副本删掉：GEMM3/GEMM4 的 A 改用 `ldmatrix.x4.trans` 直接从
//     `Ps/dSs[BM][BN]` 读（`fa_bwd_fp16_atrans_smoke.cu` 验证逐位一致）。
//   ⇒ smem 降到 71.2KB，回到 **3 CTA/SM**，且少写两份转置副本（降 L1/TEX 压力）。
template <int HD, int BM, int BN, int PIPE>
__global__ void __launch_bounds__(THREADS, (BN > 32) ? 2 : 3)
fa_bwd_fp16_mma_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                       const __half* __restrict__ v, const __half* __restrict__ do_,
                       const float* __restrict__ delta, const float* __restrict__ lse,
                       float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                       float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                       int causal, int sched) {
  static_assert(HD == 128, "O5 fp16 mma 目前只支持 head_dim=128");
  static_assert(BM % 32 == 0 && BN % 16 == 0, "BM/BN 需为 2×2 warp 网格的整数倍");
  constexpr int LD  = HD + 8;    // Q/K/V/dO 行距（half）
  constexpr int LDP = BM + 8;    // PsT/dSsT 行距（half，PIPE!=2）
  constexpr int LDS = BN + 8;    // Ps/dSs 行距（half，[BM][BN] 布局）
  constexpr int KVL = BN * LD;                     // 单个 K 或 V 缓冲（half）
  constexpr int KSB = PIPE >= 1 ? 2 * KVL : KVL;   // K 缓冲（O6/O6b 双缓冲）
  constexpr int VSB = PIPE == 1 ? 2 * KVL : KVL;   // V 缓冲（只有 O6 双缓冲）
  constexpr int PSZ = (PIPE == 2) ? BM * LDS : BN * LDP;  // P 存储大小

  // ---- 由 (BM,BN) 派生的 2×2 warp 网格几何（O5c：支持 BN=64 等更大 tile）----
  // GEMM1/2（S/dP，输出 [BM][BN]）：每个 warp 吃 (BM/2)×(BN/2)
  // GEMM3/4（dV/dK，输出 [BN][HD]）：每个 warp 吃 (BN/2)×(HD/2)，wm=wr 铺 BN、wn=wc 铺 HD
  // GEMM5（dQ，输出 [BM][HD]）：每个 warp 吃 (BM/2)×(HD/2)
  // 记 MT* 为每个 warp 的 m16/n8 tile 数。当前 HD=128 ⇒ 每个 warp 的 N 半宽恒为 64。
  constexpr int GM1 = BM / 2, GN1 = BN / 2;   // GEMM1/2 warp tile
  constexpr int GMV = BN / 2, GNV = HD / 2;   // GEMM3/4 warp tile
  constexpr int GMQ = BM / 2, GNQ = HD / 2;   // GEMM5 warp tile
  constexpr int MTM1 = GM1 / 16, MTN1 = GN1 / 8;
  constexpr int MTMV = GMV / 16, MTNV = GNV / 8;
  constexpr int MTMQ = GMQ / 16, MTNQ = GNQ / 8;
  static_assert(GM1 % 16 == 0 && GN1 % 8 == 0 && GMV % 16 == 0 && GMQ % 16 == 0,
                "warp 几何需为 mma tile 的整数倍");

  extern __shared__ __align__(16) char smem[];
  __half* Qs   = reinterpret_cast<__half*>(smem);
  __half* dOs  = Qs + BM * LD;
  __half* Ks   = dOs + BM * LD;
  __half* Vs   = Ks + KSB;
  __half* Ps   = Vs + VSB;            // PIPE==2：[BM][LDS]；否则 [BN][LDP]（=PsT）
  __half* dSs  = Ps + PSZ;            // 始终 [BM][LDS]
  __half* dSsT = dSs + BM * LDS;      // PIPE!=2 的 dS 转置副本（PIPE==2 不分配/不用）

  // ---- O6c：causal 下按 blockIdx 到 Q 块（mblk）的映射重排，做负载均衡 ----
  // 因果下第 m 个 Q 块要算 m+1 个 K/V tile，工作量随 m 线性增长；GPU 按 blockIdx
  // 递增调度，会把重块排到最后 → 尾波最重（O6b ncu：Waves 2.59、尾波最多占 33%，
  // 且 SM active cycles 最大比均值高 23%、最小低 26%）。这里不改变每个 CTA 的数学，
  // 只重排「blockIdx.x → mblk」这个双射，把轻/重块均匀铺进每个波：
  //   sched=0：原样（mblk=bx，重块在后）
  //   sched=1：交错（0, n-1, 1, n-2, ...，每个波都轻/重混合）
  //   sched=2：逆序（n-1, n-2, ..., 0，重块先跑、尾波最轻）
  const int bx = blockIdx.x;
  const int nblk = (S + BM - 1) / BM;
  int mblk = bx;
  if (causal) {
    if (sched == 1)
      mblk = (bx & 1) ? (nblk - 1 - (bx >> 1)) : (bx >> 1);
    else if (sched == 2)
      mblk = nblk - 1 - bx;
  }
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  // ---- 载入 Q/dO（越界补 0）----
  for (int i = tid; i < BM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    __half qv = __float2half(0.f), ov = __float2half(0.f);
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
  float dqacc[MTMQ][MTNQ][4];
#pragma unroll
  for (int i = 0; i < MTMQ; ++i)
#pragma unroll
    for (int j = 0; j < MTNQ; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    // O6/O6b：本 tile 的 K 用 stage = nt&1；PIPE=0 时 stage 恒 0（布局退化为原版）。
    const int stage = (PIPE >= 1) ? (nt & 1) : 0;
    __half* Kt = Ks + stage * KVL;
    __half* Vt = (PIPE == 1) ? Vs + stage * KVL : Vs;

    if constexpr (PIPE == 1) {
      // 等本 tile 的 cp.async 落地；此 barrier 同时保证「上一 tile 的 GEMM5 已读完
      // 那个 stage」，故随后把下一 tile 发进该 stage 是安全的。
      asm volatile("cp.async.wait_group 0;\n");
      __syncthreads();
      if (nt + 1 < ntiles)
        kv_issue_async<HD, BN, true, true>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                               Ks + ((nt + 1) & 1) * KVL, Vs + ((nt + 1) & 1) * KVL, LD);
    } else if constexpr (PIPE == 2) {
      // O6b：等本 tile 的 K[nt] 与上一轮发出的 V[nt] 落地（同一个 wait_group 0 覆盖）。
      // 此 barrier 同时证明「上一 tile 的 GEMM5 已读完 Ks 的另一 stage」，故可把 K[nt+1]
      // 发进该 stage；V 单缓冲，下一 tile 的 V 留到 GEMM2 之后才发（见下）。
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
        __half kv = __float2half(0.f), vv = __float2half(0.f);
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
    // 同时把 P 转 half 写进转置布局 PsT（供 GEMM3 的 A）。
    float pval[MTM1][MTN1][4];
    {
      float acc[MTM1][MTN1][4];
#pragma unroll
      for (int i = 0; i < MTM1; ++i)
#pragma unroll
        for (int j = 0; j < MTN1; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_f16<GM1, GN1, HD, false>(Qs, LD, Kt, LD, acc, wr, wc, lane);
      const int r0 = wr * GM1, c0 = wc * GN1;
#pragma unroll
      for (int i = 0; i < MTM1; ++i)
#pragma unroll
        for (int j = 0; j < MTN1; ++j)
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
              Ps[r * LDS + c] = __float2half(p);   // [BM][BN]，GEMM3 用 trans 读
            else
              Ps[c * LDP + r] = __float2half(p);   // [BN][BM]，转置副本（PIPE!=2）
          }
    }

    // ---- (2) dP = dO·Vᵀ → dS = P∘(dP − D)（同 warp/累加器映射）----
    {
      float acc[MTM1][MTN1][4];
#pragma unroll
      for (int i = 0; i < MTM1; ++i)
#pragma unroll
        for (int j = 0; j < MTN1; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_f16<GM1, GN1, HD, false>(dOs, LD, Vt, LD, acc, wr, wc, lane);
      const int r0 = wr * GM1, c0 = wc * GN1;
#pragma unroll
      for (int i = 0; i < MTM1; ++i)
#pragma unroll
        for (int j = 0; j < MTN1; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r;
            float del = (qi < S) ? delta[((size_t)(b * S + qi)) * H + h] : 0.f;
            float ds = pval[i][j][q] * (acc[i][j][q] - del);
            dSs[r * LDS + c] = __float2half(ds);   // GEMM5 A（普通）
            if constexpr (PIPE != 2)
              dSsT[c * LDP + r] = __float2half(ds);  // GEMM4 A 转置副本（PIPE==2 用 trans 免掉）
          }
    }
    __syncthreads();
    if constexpr (PIPE == 2) {
      // GEMM2 是 V 的唯一消费者；上面的 barrier 保证所有 warp 已读完 V[nt]，
      // 于是把 V[nt+1] 发进同一个单缓冲，其延迟由随后的 GEMM3/4/5 盖住。
      if (nt + 1 < ntiles)
        kv_issue_async<HD, BN, false, true>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                            Ks, Vs, LD);
    }

    // ---- (3) dV = Pᵀ·dO（A=PsT[BN][BM], B=dO[BM][HD] 转置）----
    {
      float acc[MTMV][MTNV][4];
#pragma unroll
      for (int i = 0; i < MTMV; ++i)
#pragma unroll
        for (int j = 0; j < MTNV; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      if constexpr (PIPE == 2)
        mma_block_f16<GMV, GNV, BM, true, true>(Ps, LDS, dOs, LD, acc, wr, wc, lane);
      else
        mma_block_f16<GMV, GNV, BM, true>(Ps, LDP, dOs, LD, acc, wr, wc, lane);
      const int r0 = wr * GMV, c0 = wc * GNV;
#pragma unroll
      for (int i = 0; i < MTMV; ++i)
#pragma unroll
        for (int j = 0; j < MTNV; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2;
            int jg = j0 + r;
            if (jg < S)
              red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                       acc[i][j][q], acc[i][j][q + 1]);
          }
    }

    // ---- (4) dK = scale·dSᵀ·Q（A=dSsT[BN][BM], B=Q[BM][HD] 转置）----
    {
      float acc[MTMV][MTNV][4];
#pragma unroll
      for (int i = 0; i < MTMV; ++i)
#pragma unroll
        for (int j = 0; j < MTNV; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      if constexpr (PIPE == 2)
        mma_block_f16<GMV, GNV, BM, true, true>(dSs, LDS, Qs, LD, acc, wr, wc, lane);
      else
        mma_block_f16<GMV, GNV, BM, true>(dSsT, LDP, Qs, LD, acc, wr, wc, lane);
      const int r0 = wr * GMV, c0 = wc * GNV;
#pragma unroll
      for (int i = 0; i < MTMV; ++i)
#pragma unroll
        for (int j = 0; j < MTNV; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2;
            int jg = j0 + r;
            if (jg < S)
              red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                       acc[i][j][q] * scale, acc[i][j][q + 1] * scale);
          }
    }

    // ---- (5) dQ += scale·dS·K（A=dSs[BM][BN], B=K[BN][HD] 转置）----
    {
      float acc[MTMQ][MTNQ][4];
#pragma unroll
      for (int i = 0; i < MTMQ; ++i)
#pragma unroll
        for (int j = 0; j < MTNQ; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_f16<GMQ, GNQ, BN, true>(dSs, LDS, Kt, LD, acc, wr, wc, lane);
#pragma unroll
      for (int i = 0; i < MTMQ; ++i)
#pragma unroll
        for (int j = 0; j < MTNQ; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dqacc[i][j][q] += acc[i][j][q] * scale;
          }
    }
    // PIPE=0 需要尾 barrier 保护 Ks/Vs/dSs 在下一轮被覆盖；PIPE=1/2 由下轮
    // 循环首的 barrier 承担（K 写的是另一 stage；dSs/PsT 的覆盖也在下轮首 barrier 之后），
    // 故省掉一次同步。
    if constexpr (PIPE == 0) __syncthreads();
  }

  // ---- 写回 dQ（寄存器累加结果，直接存；每个 Q 块由唯一 CTA 负责）----
#pragma unroll
  for (int i = 0; i < MTMQ; ++i)
#pragma unroll
    for (int j = 0; j < MTNQ; ++j)
#pragma unroll
      for (int q = 0; q < 4; q += 2) {
        int r = wr * GMQ + i * 16 + g + (q >= 2 ? 8 : 0);
        int c = wc * GNQ + j * 8 + c2;
        int qi = m0 + r;
        if (qi < S) {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          base[0] = dqacc[i][j][q];
          base[1] = dqacc[i][j][q + 1];
        }
      }
}

// =============================================================================
// 3) convert：fp32 累加缓冲 → fp16 输出
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc, __half* __restrict__ dq,
                               __half* __restrict__ dk, __half* __restrict__ dv, size_t n_q,
                               size_t n_kv) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_q;
       i += (size_t)gridDim.x * blockDim.x)
    dq[i] = __float2half(dq_acc[i]);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n_kv;
       i += (size_t)gridDim.x * blockDim.x) {
    dk[i] = __float2half(dk_acc[i]);
    dv[i] = __float2half(dv_acc[i]);
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
    st.max_abs = std::max(st.max_abs, d);
    st.max_rel = std::max(st.max_rel, d / (std::fabs((double)b[i]) + 1e-3));
  }
  return st;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
// PIPE=0：K/V 都不双缓冲；PIPE=1：K/V 都双缓冲；PIPE=2：只 K 双缓冲（V 单缓冲 + 后段预取）。
// K/V 的 smem 份数：0→2（各 1）、1→4（各 2）、2→3（K 2 + V 1）。
template <int HD, int BM, int BN, int PIPE>
static void launch_bwd_mma(dim3 mg, const __half* q, const __half* k, const __half* v,
                           const __half* do_, const float* delta, const float* lse,
                           float* dq_acc, float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int sched) {
  constexpr int kvn = (PIPE == 0) ? 2 : (PIPE == 1 ? 4 : 3);
  constexpr int pds = (PIPE == 2) ? 2 * BM * (BN + 8) : 2 * BN * (BM + 8) + BM * (BN + 8);
  constexpr int smem =
      (2 * BM * (HD + 8) + kvn * BN * (HD + 8) + pds) * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE><<<mg, THREADS, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, sched);
}

int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp16";
  std::string o_name = "ref_o";
  bool causal = true;
  // pipe: -1=自动（按网格大小选 1/2）、0=O5、1=O6、2=O6b。
  int pipe = -1;
  // O6c：causal 下 mblk 重排（0=原样，1=交错，2=逆序）。
  int sched = 0;
  // O6c：主 kernel tile 配置覆盖（-1=自动）。
  int bm_opt = -1, bn_opt = -1;
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a == "--nopipe") pipe = 0;
    else if (a == "--pipe") pipe = 1;
    else if (a == "--pipe2") pipe = 2;
    else if (a.rfind("--sched=", 0) == 0) sched = atoi(a.c_str() + 8);
    else if (a.rfind("--bm=", 0) == 0) bm_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--bn=", 0) == 0) bn_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  auto q_np  = load_npy_f32(dir + "/q.npy");
  auto k_np  = load_npy_f32(dir + "/k.npy");
  auto v_np  = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np  = load_npy_f32(dir + "/" + o_name + ".npy");
  auto rdq   = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk   = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv   = load_npy_f32(dir + "/ref_dv.npy");

  if (q_np.shape.size() != 4) {
    fprintf(stderr, "期望 q 为 4D [B,S,H,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  const int Hkv = (int)k_np.shape[2];
  if (D != 128) {
    fprintf(stderr, "O5 fp16 mma 版目前只支持 head_dim=128；当前 %d（MLA 见标量版）\n", D);
    return 1;
  }
  if (H % Hkv != 0) {
    fprintf(stderr, "H(%d) 必须是 Hkv(%d) 的整数倍\n", H, Hkv);
    return 1;
  }
  const size_t n = (size_t)B * S * H * D;
  const size_t nkv = (size_t)B * S * Hkv * D;
  const float scale = 1.0f / sqrtf((float)D);

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);

  auto to_half = [&](const std::vector<float>& src) {
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2half(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  __half *dq, *dk, *dv;
  __half *d_q, *d_k, *d_v, *d_o, *d_do;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  CUDA_CHECK(cudaMalloc(&dq, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_q, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_k, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_v, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_o, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_do, n * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_delta, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, (size_t)B * S * H * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), n * sizeof(__half), cudaMemcpyHostToDevice));

  dim3 pg(S, H, B);
  dim3 mg((S + 63) / 64, H, B);
  const int lse_nblk = (S + LBM - 1) / LBM;
  dim3 lg(lse_nblk, H, B);
  dim3 lg_bal((lse_nblk + 1) / 2, H, B);   // O8b：镜像配对，grid.x 减半
  constexpr int kLseSmem = (LBM + LBN) * (128 + 8) * (int)sizeof(__half);
  // O8b：PIPE=0 单缓冲（与 O8 同尺寸），PIPE=1 双缓冲。
  constexpr int kLseSmemBal0 = (LBM + LBN) * (128 + 8) * (int)sizeof(__half);
  constexpr int kLseSmemBal1 = (LBM + 2 * LBN) * (128 + 8) * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<128>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 0>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal0));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 1>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  const int cvt_threads = 256;
  const int cvt_blocks =
      (int)std::min<size_t>((std::max(n, nkv) + cvt_threads - 1) / cvt_threads, 65535);

  // O6c：主 kernel 的 tile/PIPE 自动选择。
  //  - grid < 132（不到「每 SM 一个 CTA」）且 S 较小时，把 BM 减半到 32 → grid 翻倍、
  //    并行度翻倍（S=512 MHA：main 0.0876→0.0792ms，1.11×）。大 S 下 BM=32 会让
  //    dK/dV 的跨 CTA 原子量翻倍，故只在 S≤1024 用。
  //  - 否则 BM=64：大网格走 O6b（PIPE=2，只双缓冲 K）；小网格走 O6（PIPE=1）。
  const long long grid = (long long)((S + 63) / 64) * H * B;
  const bool tiny = (grid < 132) && (S <= 1024);
  const int auto_bm = tiny ? 32 : 64;
  const int auto_bn = (!tiny && S >= 4096) ? 64 : 32;
  const int auto_pipe = (!tiny && grid >= 396) ? 2 : 1;
  const int bm_sel = (bm_opt > 0) ? bm_opt : auto_bm;
  const int bn_sel = (bn_opt > 0) ? bn_opt : auto_bn;
  const int pp_sel = (pipe >= 0) ? pipe : auto_pipe;
  printf("[O6c] main grid=%lld auto=(BM=%d,BN=%d,PIPE=%d) sel=(BM=%d,BN=%d,PIPE=%d)\n", grid,
         auto_bm, auto_bn, auto_pipe, bm_sel, bn_sel, pp_sel);
  auto launch_cfg = [&](int bm, int bn, int pp) {
    dim3 g((S + bm - 1) / bm, H, B);
    if (bm == 32) {
      if (pp == 2)
        launch_bwd_mma<128, 32, 32, 2>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      else if (pp == 1)
        launch_bwd_mma<128, 32, 32, 1>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      else
        launch_bwd_mma<128, 32, 32, 0>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    } else if (bn == 64) {
      launch_bwd_mma<128, 64, 64, 2>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    } else if (pp == 2) {
      launch_bwd_mma<128, 64, 32, 2>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    } else if (pp == 1) {
      launch_bwd_mma<128, 64, 32, 1>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    } else {
      launch_bwd_mma<128, 64, 32, 0>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    }
  };
  auto run_main = [&]() { launch_cfg(bm_sel, bn_sel, pp_sel); };
  auto run_pre = [&]() {
    if (causal)
      lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                    Hkv, scale);
    else
      lse_mma_kernel<128><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                     (int)causal);
    delta_kernel<128><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
  };

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  auto run_all = [&]() {
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    run_pre();
    run_main();
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv, n,
                                                nkv);
  };
  for (int i = 0; i < 3; ++i) run_all();
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_all();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  ms /= iters;
  double flops = 4.0 * (double)B * S * H * S * D;
  double tflops = flops / (ms * 1e-3) / 1e12;
  printf("[timing] total(3 kernels) %.4f ms  %.2f TFLOPS (bwd FLOPs=4BS^2HD)\n", ms, tflops);

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_pre();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_main();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] preprocess %.4f ms | main %.4f ms | convert %.4f ms\n", ms_pre, ms_main,
         ms - ms_pre - ms_main);

  // ---- O8b A/B（仅 causal）：LSE 原版(O8) vs 镜像配对(单缓冲) vs 镜像配对+cp.async双缓冲 ----
  if (causal) {
    auto time_lse = [&](int mode, float* out_ms) {
      auto launch = [&]() {
        if (mode == 2)
          lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S,
                                                                        H, Hkv, scale);
        else if (mode == 1)
          lse_mma_kernel_bal<128, 0><<<lg_bal, THREADS, kLseSmemBal0>>>(d_q, d_k, d_lse, S,
                                                                        H, Hkv, scale);
        else
          lse_mma_kernel<128><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale, 1);
      };
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_o8 = 0.f, ms_bal = 0.f, ms_balp = 0.f;
    time_lse(0, &ms_o8);
    time_lse(1, &ms_bal);
    time_lse(2, &ms_balp);
    printf("[O8b A/B] lse O8 %.4f ms | bal(单缓冲) %.4f ms (%.3fx) | bal+cpasync(双缓冲) %.4f ms "
           "(%.3fx)\n", ms_o8, ms_bal, ms_o8 / ms_bal, ms_balp, ms_o8 / ms_balp);
  }

  // ---- O6 A/B：同 session 对比原版（PIPE=0）、K/V 双缓冲（PIPE=1）、只 K 双缓冲（PIPE=2）----
  auto launch_mode = [&](int m) {
    if (m == 2)
      launch_bwd_mma<128, 64, 32, 2>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    else if (m == 1)
      launch_bwd_mma<128, 64, 32, 1>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
    else
      launch_bwd_mma<128, 64, 32, 0>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
  };
  auto time_launch = [&](int m, float* out_ms) {
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    for (int i = 0; i < 3; ++i) launch_mode(m);
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < iters; ++i) launch_mode(m);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    float t = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
    *out_ms = t / iters;
  };
  float ms_nopipe = 0.f, ms_pipe = 0.f, ms_pipe2 = 0.f;
  time_launch(0, &ms_nopipe);
  time_launch(1, &ms_pipe);
  time_launch(2, &ms_pipe2);
  double main_flops = 4.0 * (double)B * S * H * S * D;
  printf("[O6 A/B] main nopipe %.4f ms (%.2f TF) | pipe(KV both) %.4f ms (%.2f TF) | "
         "pipe2(K only) %.4f ms (%.2f TF) => nopipe/pipe %.3fx, nopipe/pipe2 %.3fx\n",
         ms_nopipe, main_flops / (ms_nopipe * 1e-3) / 1e12, ms_pipe,
         main_flops / (ms_pipe * 1e-3) / 1e12, ms_pipe2,
         main_flops / (ms_pipe2 * 1e-3) / 1e12, ms_nopipe / ms_pipe,
         ms_nopipe / ms_pipe2);

  // ---- O6c A/B（仅 causal）：mblk 重排 sched=0/1/2（同一 pipe）----
  if (causal) {
    const int mdef = (pipe < 0) ? auto_pipe : pipe;
    float ms_s[3] = {0.f, 0.f, 0.f};
    for (int s = 0; s < 3; ++s) {
      sched = s;
      time_launch(mdef, &ms_s[s]);
    }
    printf("[O6c A/B] main sched0(原样) %.4f ms | sched1(交错) %.4f ms (%.3fx) | "
           "sched2(逆序) %.4f ms (%.3fx)\n",
           ms_s[0], ms_s[1], ms_s[0] / ms_s[1], ms_s[2], ms_s[0] / ms_s[2]);
    sched = 1;
  }

  // ---- O5c A/B（仅 causal）：不同 (BM,BN,PIPE) tile 配置 ----
  if (causal) {
    auto time_cfg = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp); };
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      for (int i = 0; i < 3; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms0 = 0.f, ms1 = 0.f, ms2 = 0.f, ms3 = 0.f;
    time_cfg(64, 32, 2, &ms0);
    time_cfg(64, 64, 2, &ms1);
    time_cfg(32, 32, 1, &ms2);
    time_cfg(32, 32, 2, &ms3);
    printf("[O5c A/B] main (64,32,2) %.4f ms (%.2f TF) | (64,64,2) %.4f (%.2f) | (32,32,1) %.4f "
           "(%.2f) | (32,32,2) %.4f (%.2f) => best %.3fx\n",
           ms0, main_flops / (ms0 * 1e-3) / 1e12, ms1, main_flops / (ms1 * 1e-3) / 1e12,
           ms2, main_flops / (ms2 * 1e-3) / 1e12, ms3, main_flops / (ms3 * 1e-3) / 1e12,
           ms0 / std::min(std::min(ms0, ms1), std::min(ms2, ms3)));
  }

  // ---- 数值对拍（重新跑一次完整 forward 保证累加缓冲清零）----
  run_all();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<__half> hdq(n), hdk(nkv), hdv(nkv);
  CUDA_CHECK(cudaMemcpy(hdq.data(), dq, n * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdk.data(), dk, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(hdv.data(), dv, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  std::vector<float> mdq(n), mdk(nkv), mdv(nkv);
  for (size_t i = 0; i < n; ++i) mdq[i] = __half2float(hdq[i]);
  for (size_t i = 0; i < nkv; ++i) {
    mdk[i] = __half2float(hdk[i]);
    mdv[i] = __half2float(hdv[i]);
  }

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);

  auto try_cmp = [&](const char* fname, const std::vector<float>& ref) {
    std::ifstream test(dir + "/" + fname + ".npy");
    if (!test.good()) return;
    auto t = load_npy_f32(dir + "/" + fname + ".npy");
    DiffStat st = diff_stat(t.data, ref);
    printf("  %-9s vs ref: max_abs=%.3e  max_rel=%.3e\n", fname, st.max_abs, st.max_rel);
  };
  try_cmp("fa_dq", rdq.data);
  try_cmp("fa_dk", rdk.data);
  try_cmp("fa_dv", rdv.data);
  try_cmp("te_dq", rdq.data);
  try_cmp("te_dk", rdk.data);
  try_cmp("te_dv", rdv.data);

  cudaFree(dq); cudaFree(dk); cudaFree(dv);
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o); cudaFree(d_do);
  cudaFree(d_delta); cudaFree(d_lse);
  cudaFree(d_dq_acc); cudaFree(d_dk_acc); cudaFree(d_dv_acc);
  return 0;
}
