// =============================================================================
// fa_bwd_fp16_mma_kernels.cuh —— fp16 张量核反向（O5）**两文件版的 device 部分**
// =============================================================================
// 由单文件 `fa_bwd_fp16_mma_onefile.cu` 拆分而来，device 代码与单文件**逐字一致**：
//   * mma/ldmatrix 封装、preprocess_kernel、fa_bwd_fp16_mma_kernel、convert_kernel。
// host 侧（npy 读取 / launcher / 自测）见 `fa_bwd_fp16_mma_main.cu`。
//
// ----------（以下为单文件版原始说明）----------
// FlashAttention 反向（fp16）**张量核版**（O5）
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

#ifndef FA_BWD_FP16_MMA_KERNELS_CUH_
#define FA_BWD_FP16_MMA_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int THREADS = 128;   // 4 warps
static constexpr int WN      = 2;     // N 方向 warp 数（2×2 warp 网格）

// ----------------------------- O11：快速指数/对数 -----------------------------
// 原实现用 libdevice 的精确 `expf`/`logf`（软件多项式，~10 多条指令），而 softmax 的
// exp/log 在主 kernel 与 LSE 预处理里都是**每元素**要算的热点（ncu：main 的 `wait`
// fixed-latency 占 2.04、LSE Compute 60%）。FA2/FA3 用的是硬件 `exp2f`（MUFU.EX2）。
// 这里把 `expf`/`logf` 换成硬件内建 `__expf`/`__logf`（MUFU.EX2/LG2，相对误差 ~2^-21），
// 对 fp16（容差 ~1e-3）绰绰有余。用 `FAST_EXP` 宏做 A/B（-DFAST_EXP=0 走精确版）。
#ifndef FAST_EXP
#define FAST_EXP 1
#endif
__device__ __forceinline__ float fexp(float x) {
#if FAST_EXP
  return __expf(x);
#else
  return expf(x);
#endif
}
__device__ __forceinline__ float flog(float x) {
#if FAST_EXP
  return __logf(x);
#else
  return logf(x);
#endif
}

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

// =============================================================================
// O9：Hopper `wgmma`（SS，A/B 均来自 smem 描述符）+ SW128（128B swizzle）K-major 布局
// =============================================================================
// 目的：把反向的 GEMM 从 SM80 兼容的 `mma.m16n8k16 + ldmatrix` 换成 Hopper warpgroup MMA，
// 免掉 ldmatrix（消 `short_scoreboard`）、减少发射指令、提高张量核吞吐。SW128 布局与
// wgmma 描述符自洽（同一 16B chunk 置换），由 `fa_bwd_fp16_wgmma_smoke.cu` 逐位验证：
//   * 元素 `(row,k)` 字节偏移见 `sw128_off`（布局 `[row/8][k/64][8 行][64 元素]`，atom 1024B）；
//   * 描述符 `SBO = (K/64)*1024`，`LBO` 恒 1（B128 下硬件忽略），k16 步进
//     `floor(s/4)*1024 + (s%4)*32`；
//   * `m64n64k16` 的累加器布局 = 每 warp 16 行、warp 内 `d[j*4+q]` ↔
//     `row=16*wid+g+(q>=2?8:0)`、`col=j*8+2*(lane%4)+(q&1)`（与 mma.m16n8 的 `acc[j][q]` 同构）。
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6, kk = k & 63;
  const int cc = (kk >> 3) ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
__device__ __forceinline__ void sw128_store16(char* tile, int row, int k0, int K,
                                              uint4 v) {
  *reinterpret_cast<uint4*>(tile + sw128_off(row, k0, K)) = v;
}
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;  // LBO = 1（B128 下硬件忽略）
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;  // base_offset（tile 1024B 对齐时相位 0）
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}
// CUDA 13 的 nvcc 只有 `-gencode=arch=compute_90a` 才让 ptxas 接受 wgmma；默认 `-arch=sm_90`
// 下这些 asm 会报错。用 `__CUDA_ARCH_FEAT_SM90_ALL` 把 asm 包起来：sm_90 编译时退化成空实现
// （只有显式 `--lsewgm` 才会走这条路径，且必须用 sm_90a 构建）。
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define FA_HAS_WGMMA 1
#else
#define FA_HAS_WGMMA 0
#endif
__device__ __forceinline__ void wgmma_fence() {
#if FA_HAS_WGMMA
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit() {
#if FA_HAS_WGMMA
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_wait0() {
#if FA_HAS_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_m64n64k16_f16(float (&d)[32], uint64_t da,
                                                    uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, 0, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1));
#else
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  (void)da;
  (void)db;
#endif
}
// 用 wgmma m64n64k16 算 Q[64][HD]·Kᵀ[HD][64]（K 存成 [64][HD] K-major SW128），
// 结果落到每线程 32 个 fp32（`d[j*4+q]` 的映射同 mma 的 `acc[j][q]`）。
__device__ __forceinline__ void wgmma_qkt64(const char* Qsw, const char* Ksw, int HD,
                                            float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t qa = smem_u32(Qsw), ka = smem_u32(Ksw);
  const uint32_t sbo = (uint32_t)((HD / 64) * 1024);
#pragma unroll
  for (int s = 0; s < HD / 16; ++s) {
    uint64_t da = make_desc_sw128(sw128_k16_addr(qa, s), sbo);
    uint64_t db = make_desc_sw128(sw128_k16_addr(ka, s), sbo);
    wgmma_m64n64k16_f16(d, da, db);
  }
  wgmma_commit();
  wgmma_wait0();
}

// O4c 风格向量化归约：mma.m16n8 累加器里 q/q+1 两列相邻且同 row → 一次 float2 atomicAdd。
__device__ __forceinline__ void red_add2(float* p, float a, float b) {
  atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));
}

// O7c：dK/dV 归约再把 float2 提升到 float4。mma.m16n8 里一个 quad（lane&3=0..3）的
// `c2=(lane&3)*2` 分别是 0/2/4/6，即同 row 的连续 8 列；把 quad 的 float2 用 `shfl_down 1`
// 拼成两个 float4（列 0-3 由 lane0 写、列 4-7 由 lane2 写），`red.global.add.v4.f32` 的
// 事务数相对 float2 再减半。调用者须保证整个 warp 参与 shfl（在 `if (jg<S)` 之外算好），
// 且仅 (lane&1)==0 的 lane 执行 st；偶 lane 的地址天然 16B 对齐（c2∈{0,4}）。
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
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

// O10：把一整块 Q/dO（BM 行 × HD 列，half）异步发进 smem。
// 与 `kv_issue_async` 同构，但用于 Q/dO（每个 CTA 只在 prologue 载入一次）：
//   * 用 16B `cp.async.cg` 向量化，取代原来的逐元素 `LDG.U16 + STS.U16`（ncu 报 Q/dO 的
//     标量全局读未被利用满 32B/sector，且 prologue 的全局延迟串在 K/V 之后）。
//   * 行越界（qi>=S）用普通 smem 写 0（与 K/V 一样，由后续 barrier 保证可见）。
// 数值与标量路径**逐位相同**（搬的是同样的 half）。PIPE>=1 时 Q/dO 与 K/V 各提交一个
// commit_group，循环首的 `cp.async.wait_group 0` 一并等待，从而让 Q/dO 的全局延迟与 K/V 重叠。
template <int HD, int BM>
__device__ __forceinline__ void qdo_issue_async(const __half* __restrict__ q,
                                                const __half* __restrict__ do_, int m0, int S,
                                                int H, int h, int b, int tid, __half* Qd,
                                                __half* dOd, int LD) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 half) 数
  constexpr int NU  = BM * HDV;  // 总 unit 数
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int qi = m0 + row;
    if (qi < S) {
      const size_t off = (((size_t)(b * S + qi)) * H + h) * HD + c8 * 8;
      cp_async16(Qd + row * LD + c8 * 8, q + off);
      cp_async16(dOd + row * LD + c8 * 8, do_ + off);
    } else {
      *reinterpret_cast<uint4*>(Qd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(dOd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
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
          lrow[s] = lrow[s] * fexp(mrow[s] - mn) + fexp(sv - mn);
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
      float ca = (m == -INFINITY) ? 0.f : l * fexp(m - mn);
      float cb = (m2 == -INFINITY) ? 0.f : l2 * fexp(m2 - mn);
      l = ca + cb;
      m = mn;
    }
    if (c2 == 0) {
      int r = wid * 16 + g + (s ? 8 : 0);
      int qi = m0 + r;
      if (qi < S) lse[((size_t)(b * S + qi)) * H + h] = m + flog(l);
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

  // O10：Q 的载入同样向量化（PIPE=1 用 16B cp.async，与 K 一起在循环首 wait 覆盖；
  // PIPE=0 用标量写）。Q 只依赖本 CTA 的 m 块，行越界写 0。
  auto issue_q = [&](__half* Qd, int m0) {
#pragma unroll
    for (int u = tid; u < LBM * HDV; u += THREADS) {
      const int row = u / HDV, c8 = u % HDV;
      const int qi = m0 + row;
      if (qi < S) {
        const size_t off = (((size_t)(b * S + qi)) * H + h) * HD + c8 * 8;
        if constexpr (PIPE) {
          cp_async16(Qd + row * LD + c8 * 8, q + off);
        } else {
#pragma unroll
          for (int e = 0; e < 8; ++e) Qd[row * LD + c8 * 8 + e] = q[off + e];
        }
      } else {
        *reinterpret_cast<uint4*>(Qd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
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
    issue_q(Qs, m0);

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
            lrow[s] = lrow[s] * fexp(mrow[s] - mn) + fexp(sv - mn);
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
        float ca = (m == -INFINITY) ? 0.f : l * fexp(m - mn);
        float cb = (m2 == -INFINITY) ? 0.f : l2 * fexp(m2 - mn);
        l = ca + cb;
        m = mn;
      }
      if (c2 == 0) {
        int r = wid * 16 + g + (s ? 8 : 0);
        int qi = m0 + r;
        if (qi < S) lse[((size_t)(b * S + qi)) * H + h] = m + flog(l);
      }
    }
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）
    __syncthreads();
  }
}

// =============================================================================
// O9：wgmma 版 LSE（causal 专用，仅 HD=128；镜像配对 + SW128 + wgmma.m64n64k16）
// =============================================================================
// 动机：O8b 后 LSE 仍占端到端 ~15%，ncu 报 Compute 60%（`mma.m16n8k16 + ldmatrix` 在
// Hopper 上只走 SM80 兼容路径，张量核利用率低、且每个 k-step 都有 ldmatrix 依赖）。
// 本 kernel 用 1 条 `wgmma.m64n64k16` 取代「4 warp × m16n64 × 8 k-step」的 mma：
//   * Q/K 以 **SW128 K-major** 存 smem（16B chunk 置换），wgmma 描述符直读，免 ldmatrix；
//   * 布局/累加器映射由 `fa_bwd_fp16_wgmma_smoke.cu` 逐位验证；
//   * 镜像配对、online-softmax、行 4-lane `shfl` 归约与 `lse_mma_kernel_bal` 相同 ⇒ 数值应
//     在 fp16 噪声内一致。仅 HD=128（LBN=64=N）启用；HD=512 的 SW128 tile 过大不走。
template <int HD, int PIPE>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_wgmma(const __half* __restrict__ q, const __half* __restrict__ k,
                         float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  static_assert(HD == 128, "wgmma LSE 目前只做 HD=128");
  constexpr int HDV = HD / 8;
  constexpr int TILE = (LBM / 8) * (HD / 64) * 1024;  // 单个 SW128 tile 字节数（HD=128→16KB）
  extern __shared__ char smem_raw[];
  // SW128 描述符 base_offset=0 要求 tile 1024B 对齐 → 手动对齐动态 smem 基址。
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;  // PIPE=1：2*TILE；PIPE=0：TILE

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // K tile（j0 起 LBN 行）用 cp.async 发进 SW128 tile（PIPE=1）或标量写（PIPE=0）。
  auto issue_k = [&](char* Kd, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c8 = u % HDV;
      const int jg = j0 + row;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (jg < S) {
        const size_t off = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c8 * 8;
        if constexpr (PIPE) {
          cp_async16(Kd + sw128_off(row, c8 * 8, HD), k + off);
          continue;
        }
        v = *reinterpret_cast<const uint4*>(k + off);
      }
      *reinterpret_cast<uint4*>(Kd + sw128_off(row, c8 * 8, HD)) = v;
    }
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };
  auto issue_q = [&](char* Qd, int m0) {
#pragma unroll
    for (int u = tid; u < LBM * HDV; u += THREADS) {
      const int row = u / HDV, c8 = u % HDV;
      const int qi = m0 + row;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (qi < S) {
        const size_t off = (((size_t)(b * S + qi)) * H + h) * HD + c8 * 8;
        if constexpr (PIPE) {
          cp_async16(Qd + sw128_off(row, c8 * 8, HD), q + off);
          continue;
        }
        v = *reinterpret_cast<const uint4*>(q + off);
      }
      *reinterpret_cast<uint4*>(Qd + sw128_off(row, c8 * 8, HD)) = v;
    }
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) continue;
    const int m0 = mblk * LBM;
    issue_q(Qs, m0);
    const int ncols = min(S, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    if constexpr (PIPE) {
      if (ntiles > 0) issue_k(Ks, 0);
    }

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int nt = 0; nt < ntiles; ++nt) {
      const int j0 = nt * LBN;
      char* Kt = Ks + (PIPE ? (nt & 1) * TILE : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (nt + 1 < ntiles) issue_k(Ks + ((nt + 1) & 1) * TILE, j0 + LBN);
      } else {
        issue_k(Ks, j0);
        __syncthreads();
      }

      float d[32];
      wgmma_qkt64(Qs, Kt, HD, d);

#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int s = q >= 2 ? 1 : 0;
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float sv = -INFINITY;
          if (qi < S && jg < S && jg <= qi) sv = d[j * 4 + q] * scale;
          if (sv != -INFINITY) {
            float mn = fmaxf(mrow[s], sv);
            lrow[s] = lrow[s] * fexp(mrow[s] - mn) + fexp(sv - mn);
            mrow[s] = mn;
          }
        }
    }
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      float m = mrow[s], l = lrow[s];
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        float m2 = __shfl_xor_sync(0xffffffffu, m, off);
        float l2 = __shfl_xor_sync(0xffffffffu, l, off);
        float mn = fmaxf(m, m2);
        float ca = (m == -INFINITY) ? 0.f : l * fexp(m - mn);
        float cb = (m2 == -INFINITY) ? 0.f : l2 * fexp(m2 - mn);
        l = ca + cb;
        m = mn;
      }
      if (c2 == 0) {
        int r = wid * 16 + g + (s ? 8 : 0);
        int qi = m0 + r;
        if (qi < S) lse[((size_t)(b * S + qi)) * H + h] = m + flog(l);
      }
    }
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
// =============================================================================
// O9b：主 kernel 的 wgmma 版（GEMM1/2 用 wgmma.m64n64k16 + SW128；GEMM3/4/5 仍 mma，
//      其转置 B 从同一块 SW128 tile 用 `ldmatrix.x2.trans` 读；只做 fp16 / HD=128 / BM=BN=64）。
// =============================================================================
// 动机：O9a 已把 LSE 的单个 QKᵀ 换成 wgmma，但 LSE 是 softmax epilogue bound、收益有限。
// 主 kernel 的墙是 `wait`（mma 依赖）+ L2（dK/dV 原子）+ 低 occupancy；wgmma 的异步 mma
// 可让 GEMM1/2 免掉 ldmatrix 并减少发射，同时 SW128 布局天然无 bank conflict。
//
// 数据通路（本冒烟 `fa_bwd_fp16_wgmma_main_smoke.cu` 逐位验证）：
//   * Q/dO/K/V 存成 **SW128 K-major** tile（`sw128_off` 写、wgmma 描述符直读）；
//   * GEMM1 `S=Q·Kᵀ`、GEMM2 `dP=dO·Vᵀ` 用 `wgmma.m64n64k16`（整 CTA 一个 64×64 tile）；
//   * GEMM3/4/5 的 B（dO/Q/K，BTRANS）用 `ldmatrix.x2.trans` + `sw128_off` 从同一 tile 转置读，
//     SW128 只在 16B 粒度置换，ldmatrix 每个 lane 只要一个 16B 地址；
//   * P/dS 仍按 `[BM][BN]` 行主序存（+8 行距），GEMM3/4 用 `ldmatrix.x4.trans`（ATRANS，O6b）。
//
// smem（HD=128,BM=BN=64）：Q/dO 各 16KB + K 双缓冲 32KB + V 单缓冲 16KB + Ps/dSs 18KB ≈ 98KB
// → 2 CTA/SM（与 O13 的 (64,64,2) mma 版同 occupancy）。
#ifdef FA_WGMMA

// SW128 K-major（A[M][K]、B[N][K] 均 K-major）的 wgmma QKᵀ，行宽 Kd。
__device__ __forceinline__ void wgmma_mn64(const char* Asw, const char* Bsw, int Kd,
                                           float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((Kd / 64) * 1024);
#pragma unroll
  for (int s = 0; s < Kd / 16; ++s) {
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(aa, s), sbo),
                        make_desc_sw128(sw128_k16_addr(ba, s), sbo));
  }
  wgmma_commit();
  wgmma_wait0();
}

// issue-only 版（不 wait）：GEMM1/GEMM2 两个 wgmma group 一起发、最后统一 wait0，
// 让两条异步 mma 重叠（原 `wgmma_mn64` 每次内部 wait0，串行）。
__device__ __forceinline__ void wgmma_mn64_issue(const char* Asw, const char* Bsw, int Kd,
                                                 float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((Kd / 64) * 1024);
#pragma unroll
  for (int s = 0; s < Kd / 16; ++s) {
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(aa, s), sbo),
                        make_desc_sw128(sw128_k16_addr(ba, s), sbo));
  }
  wgmma_commit();
}

// A=[M_TILE][K_TILE] 行主序（行距 asld，half）；ATRANS=true 时 As 存 [K][M]（ldmatrix.x4.trans）。
// B 为 SW128 K-major tile（行宽 BK=HD），BTRANS 读 [K=token][N=hd]（`ldmatrix.x2.trans`）。
template <int WARP_M, int WARP_N, int K_TILE, bool ATRANS, int BK>
__device__ __forceinline__ void mma_block_swb(const __half* As, int asld, const char* Bsw,
                                              float acc[WARP_M / 16][WARP_N / 8][4], int wm,
                                              int wn, int lane) {
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
        ldmatrix_x4_trans(smem_u32(As + (koff + krow) * asld +
                                   (wm * WARP_M + i * 16 + mcol)), av[i]);
    } else {
      const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int acol = (lane >> 4) * 8;
#pragma unroll
      for (int i = 0; i < MTM; ++i)
        ldmatrix_x4(smem_u32(As + (wm * WARP_M + i * 16 + arow) * asld + koff + acol),
                    av[i]);
    }
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      const int krow = (lane & 7) + ((lane >> 3) & 1) * 8;
      const int ncol = wn * WARP_N + j * 8;
      uint32_t d[2];
      ldmatrix_x2_trans(smem_u32(Bsw + sw128_off(koff + krow, ncol, BK)), d);
      bv[j][0] = d[0];
      bv[j][1] = d[1];
    }
#pragma unroll
    for (int i = 0; i < MTM; ++i)
#pragma unroll
      for (int j = 0; j < MTN; ++j) mma_f16(acc[i][j], av[i], bv[j]);
  }
}

// ---- O9b-2：MN-major（转置）描述符 ----
// 同一份 **K-major SW128** 存储的 tile，用 `Major::MN` 描述符 + `tnsp=1` 读，等于读它的转置
// （FA3 `dKV_swapAB` 的做法）。`fa_bwd_fp16_wgmma_bwd_smoke.cu` 逐位验证：
//   * LBO=64（相邻 64 列组的 u128 步长）；SBO=(W/64)*64（相邻 8 行组的 u128 步长）；
//   * trans 操作数第 s 个 k16 slab（K=行，前进 16 行=2 行组）地址 = base + s*2*SBO*16 字节。
__device__ __forceinline__ uint32_t trans_k16_addr(uint32_t base, int s, uint32_t W) {
  return base + (uint32_t)(s * 2 * ((W >> 6) * 64) * 16);
}
__device__ __forceinline__ uint64_t desc_k16_mn(uint32_t base, int s, uint32_t W) {
  uint64_t d = 0;
  d |= (uint64_t)((trans_k16_addr(base, s, W) >> 4) & 0x3FFF);
  d |= (uint64_t)(64u & 0x3FFF) << 16;  // LBO = 64
  d |= (uint64_t)((((W >> 6) * 64)) & 0x3FFF) << 32;
  d |= (uint64_t)1 << 62;  // B128
  return d;
}
__device__ __forceinline__ uint64_t desc_k16_k(uint32_t base, int s, uint32_t W) {
  return make_desc_sw128(sw128_k16_addr(base, s), (W >> 6) * 1024);
}
// wgmma m64n64k16，TA/TB = 0(K-major) / 1(MN-major)。
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n64k16_t(float (&d)[32], uint64_t da,
                                                  uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, 1, 1, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(TA), "n"(TB));
#else
  (void)da; (void)db;
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
#endif
}
// P/dS 的 16B 分块 SW128 存储：一个 quad（lane%4）覆盖同 row 的 8 个连续列（每个 lane 贡献
// 相邻 2 列），用 `shfl` 拼成 uint4 后由 lane%4==0 落 16B。`a` 为 row r0、`b` 为 row r0+8。
__device__ __forceinline__ void pds_store_sw128(char* tile, int j, int r0, int lane,
                                                 int c2, int W, __half a0, __half a1,
                                                 __half b0, __half b1) {
  const uint32_t h2a = (uint32_t)__half_as_ushort(a0) |
                       ((uint32_t)__half_as_ushort(a1) << 16);
  const uint32_t h2b = (uint32_t)__half_as_ushort(b0) |
                       ((uint32_t)__half_as_ushort(b1) << 16);
  const int c = j * 8 + c2;
  *reinterpret_cast<uint32_t*>(tile + sw128_off(r0, c, W)) = h2a;
  *reinterpret_cast<uint32_t*>(tile + sw128_off(r0 + 8, c, W)) = h2b;
  (void)lane;
}

// 把 K/V（[BN][HD] half）用 16B `cp.async` 发进 SW128 tile（DOK/DOV 拆不同 commit_group）。
template <int HD, int BN, bool DOK, bool DOV>
__device__ __forceinline__ void kv_issue_async_sw(const __half* __restrict__ k,
                                                  const __half* __restrict__ v, int j0, int S,
                                                  int Hkv, int hkv, int b, int tid, char* Kd,
                                                  char* Vd) {
  constexpr int HDV = HD / 8;
  constexpr int NU  = BN * HDV;
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int jg = j0 + row;
    if (jg < S) {
      const size_t off = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c8 * 8;
      if (DOK) cp_async16(Kd + sw128_off(row, c8 * 8, HD), k + off);
      if (DOV) cp_async16(Vd + sw128_off(row, c8 * 8, HD), v + off);
    } else {
      if (DOK) *reinterpret_cast<uint4*>(Kd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
      if (DOV) *reinterpret_cast<uint4*>(Vd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// 把 Q/dO（[BM][HD] half）用 16B `cp.async` 发进 SW128 tile。
template <int HD, int BM>
__device__ __forceinline__ void qdo_issue_async_sw(const __half* __restrict__ q,
                                                   const __half* __restrict__ do_, int m0,
                                                   int S, int H, int h, int b, int tid,
                                                   char* Qd, char* dOd) {
  constexpr int HDV = HD / 8;
  constexpr int NU  = BM * HDV;
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int qi = m0 + row;
    if (qi < S) {
      const size_t off = (((size_t)(b * S + qi)) * H + h) * HD + c8 * 8;
      cp_async16(Qd + sw128_off(row, c8 * 8, HD), q + off);
      cp_async16(dOd + sw128_off(row, c8 * 8, HD), do_ + off);
    } else {
      *reinterpret_cast<uint4*>(Qd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(dOd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

template <int HD>
__global__ void __launch_bounds__(THREADS, 2)
fa_bwd_fp16_wgmma_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                         const __half* __restrict__ v, const __half* __restrict__ do_,
                         const float* __restrict__ delta, const float* __restrict__ lse,
                         float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                         float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                         int causal, int sched) {
  static_assert(HD == 128, "wgmma 主 kernel 目前只做 HD=128");
  constexpr int BM = 64, BN = 64;
  constexpr int TILE  = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（16KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（16KB）
  constexpr int PSZ   = BM * BN * 2;                   // P/dS SW128 tile（8KB）

  extern __shared__ char smem_raw[];
  // SW128 描述符 base_offset=0 要求 tile 1024B 对齐 → 手动对齐动态 smem 基址。
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + TILE;
  char* Ks  = dOs + TILE;                 // 双缓冲 2*KTILE
  char* Vs  = Ks + 2 * KTILE;             // 单缓冲 KTILE
  char* Ps  = Vs + KTILE;                 // P  [BM][BN] SW128 K-major（8KB）
  char* dSs = Ps + PSZ;                   // dS [BM][BN] SW128 K-major（8KB）

  const int bx = blockIdx.x;
  const int nblk = (S + BM - 1) / BM;
  int mblk = bx;
  if (causal) {
    if (sched == 1) mblk = (bx & 1) ? (nblk - 1 - (bx >> 1)) : (bx >> 1);
    else if (sched == 2) mblk = nblk - 1 - bx;
  }
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  qdo_issue_async_sw<HD, BM>(q, do_, m0, S, H, h, b, tid, Qs, dOs);

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0) {
    kv_issue_async_sw<HD, BN, true, false>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs);
    kv_issue_async_sw<HD, BN, false, true>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs);
  }

  // LSE/D 预装：wgmma m64n64 的累加器里，warp wid 持行 [16*wid,16*wid+16)，每线程两行。
  const int r_lo = wid * 16 + g, r_hi = r_lo + 8;
  const int qi_lo = m0 + r_lo, qi_hi = m0 + r_hi;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < S) {
    const size_t idx = ((size_t)(b * S + qi_lo)) * H + h;
    lse_lo = lse[idx];
    del_lo = delta[idx];
  }
  if (qi_hi < S) {
    const size_t idx = ((size_t)(b * S + qi_hi)) * H + h;
    lse_hi = lse[idx];
    del_hi = delta[idx];
  }

  // dQ 寄存器累加（每个 Q 块唯一 CTA、无跨 CTA 原子）：dqacc[nh][j][q] ↔ wgmma 累加器布局。
  float dqacc[2][8][4];
#pragma unroll
  for (int nh = 0; nh < 2; ++nh)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] = 0.f;

  const uint32_t Qa = smem_u32(Qs), dOa = smem_u32(dOs);
  const uint32_t Pa = smem_u32(Ps), DSa = smem_u32(dSs);
  const int r0 = wid * 16 + g;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    char* Kt = Ks + (nt & 1) * KTILE;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, true, false>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                             Ks + ((nt + 1) & 1) * KTILE, Vs);

    // ---- (1)(2) S=QKᵀ 与 dP=dO·Vᵀ 两条 wgmma 一起发、统一 wait0（重叠异步 mma）----
    float sacc[32], dpacc[32];
    wgmma_mn64_issue(Qs, Kt, HD, sacc);
    wgmma_mn64_issue(dOs, Vs, HD, dpacc);
    wgmma_wait0();
    // (1) epilogue：P = exp(scale·S − LSE)，按 SW128 16B 分块写 Ps
    float pval[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int qi = m0 + r0 + (qq >= 2 ? 8 : 0);
        const int jg = j0 + j * 8 + c2 + (qq & 1);
        const float lv = (qq >= 2) ? lse_hi : lse_lo;
        float p = 0.f;
        if (qi < S && jg < S && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      pds_store_sw128(Ps, j, r0, lane, c2, BN,
                      __float2half(pval[j][0]), __float2half(pval[j][1]),
                      __float2half(pval[j][2]), __float2half(pval[j][3]));
    // (2) epilogue：dS = P∘(dP−D)，按 SW128 16B 分块写 dSs
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSs, j, r0, lane, c2, BN,
                      __float2half(d0), __float2half(d1),
                      __float2half(d2), __float2half(d3));
    }
    // barrier：P/dS 对所有 warp 可见；同时保证 GEMM2 已读完 V[nt]，可覆盖 V。
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, false, true>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid, Ks, Vs);

    // ---- O9b-2：(3) dV=Pᵀ·dO、(4) dK=scale·dSᵀ·Q、(5) dQ+=scale·dS·K 全上 wgmma ----
    // 三条 GEMM 的 A/B 均为「K-major 存储 + MN-major 描述符（转置读）」，唯一例外是 (5) 的
    // A=dS 用 K-major。按 N 半（nh=0/1）分两遍，每遍三条一起发、统一 wait0。
    wgmma_fence();
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
      const uint32_t Qn  = Qa  + (uint32_t)(nh * 1024);
      const uint32_t Kn  = smem_u32(Kt) + (uint32_t)(nh * 1024);
      float accv[32], acck[32], accq[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) { accv[i] = 0.f; acck[i] = 0.f; accq[i] = 0.f; }
#pragma unroll
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
      wgmma_commit();
#pragma unroll
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
      wgmma_commit();
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_t<0, 1>(accq, desc_k16_k(DSa, s, BN), desc_k16_mn(Kn, s, HD));
      wgmma_commit();
      wgmma_wait0();
      // (3) dV = Pᵀ·dO epilogue
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; qq += 2) {
          const int rr = r0 + (qq >= 2 ? 8 : 0);
          const int jg = j0 + rr;
          const int c = nh * 64 + j * 8 + c2;
          if (jg < S)
            red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     accv[j * 4 + qq], accv[j * 4 + qq + 1]);
        }
      // (4) dK = scale·dSᵀ·Q epilogue
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; qq += 2) {
          const int rr = r0 + (qq >= 2 ? 8 : 0);
          const int jg = j0 + rr;
          const int c = nh * 64 + j * 8 + c2;
          if (jg < S)
            red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
        }
      // (5) dQ += scale·dS·K（寄存器累加，循环结束后一次写出）
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] += accq[j * 4 + qq] * scale;
    }
  }

#pragma unroll
  for (int nh = 0; nh < 2; ++nh)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; qq += 2) {
        const int rr = r0 + (qq >= 2 ? 8 : 0);
        const int qi = m0 + rr;
        const int c = nh * 64 + j * 8 + c2;
        if (qi < S) {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) =
              make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
        }
      }
}

#endif  // FA_WGMMA

template <int HD, int BM, int BN, int PIPE, bool R4 = false, bool PREL = true>
__global__ void __launch_bounds__(THREADS, (BN > 32) ? 2 : 3)
fa_bwd_fp16_mma_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                       const __half* __restrict__ v, const __half* __restrict__ do_,
                       const float* __restrict__ delta, const float* __restrict__ lse,
                       float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                       float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                       int causal, int sched) {
  // O5c：head_dim 从「只 128」扩到 128/512（MLA）。HD>128 时 GEMM1/2（S/dP）的归约维是 HD，
  // 只是 k-loop 变长；GEMM3/4/5 的输出 N 维是 HD，需加一层 **N-tile 循环**（每遍 NTW=128 列），
  // 否则 `GNV=HD/2` 会让累加器/寄存器爆炸。HD=128 时 NDT=1，路径与 O5/O6/O6b/O6c/O7c 逐字等价。
  static_assert(HD % 128 == 0, "head_dim 需为 128 的整数倍");
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
  // GEMM3/4（dV/dK，输出 [BN][HD]）：每个 warp 吃 (BN/2)×(NTW/2)，NTW=WN*64=128（N-tile 宽）
  // GEMM5（dQ，输出 [BM][HD]）：每个 warp 吃 (BM/2)×(NTW/2)
  // 记 MT* 为每个 warp 的 m16/n8 tile 数。HD=128 时 NTW=HD ⇒ 与 O5c 逐字一致。
  constexpr int NTW = WN * 64;                // N-tile 宽（GEMM3/4/5 每遍覆盖的 head_dim 列）
  constexpr int NDT = HD / NTW;               // N-tile 遍数（HD=128→1，HD=512→4）
  constexpr int GM1 = BM / 2, GN1 = BN / 2;   // GEMM1/2 warp tile
  constexpr int GMV = BN / 2, GNV = NTW / 2;  // GEMM3/4 warp tile（N-tile 内）
  constexpr int GMQ = BM / 2, GNQ = NTW / 2;  // GEMM5 warp tile（N-tile 内）
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
  // O10：PIPE>=1 时用 16B `cp.async` 异步发 Q/dO（与 K/V 一起在循环首 `wait_group 0` 等待），
  // 让 Q/dO 的全局延迟与 K/V 重叠；PIPE==0 无流水语义，保持同步标量读。
  if constexpr (PIPE >= 1) {
    qdo_issue_async<HD, BM>(q, do_, m0, S, H, h, b, tid, Qs, dOs, LD);
  } else {
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

  // O7c-prel：LSE 与 delta 只依赖 CTA 自己的 Q 行（与 K tile 无关）。原来在每个 tile 的
  // GEMM1/2 epilogue 里按 (qi) 去 global 读，ncu 报「global load 每 thread 仅用 4.4/32B」。
  // 这里在 nt 循环前一次性把本线程需要的 MTM1×2 个 row 的 LSE/D 装进寄存器，循环内零 global 读。
  // 行号 r = wr*GM1 + i*16 + g + (s?8:0) 与 epilogue 的 (i, q>=2) 一一对应。
  float lse_r[MTM1][2], del_r[MTM1][2];
  if constexpr (PREL) {
#pragma unroll
    for (int i = 0; i < MTM1; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int r = wr * GM1 + i * 16 + g + (s ? 8 : 0);
        const int qi = m0 + r;
        const size_t idx = ((size_t)(b * S + qi)) * H + h;
        const bool ok = qi < S;
        lse_r[i][s] = ok ? lse[idx] : 0.f;
        del_r[i][s] = ok ? delta[idx] : 0.f;
      }
  }

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
            float lv = 0.f;
            if constexpr (PREL)
              lv = lse_r[i][q >= 2 ? 1 : 0];
            else if (qi < S)
              lv = lse[((size_t)(b * S + qi)) * H + h];
            float p = 0.f;
            if (qi < S && jg < S && !(causal && jg > qi))
              p = fexp(acc[i][j][q] * scale - lv);
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
            float del = 0.f;
            if constexpr (PREL)
              del = del_r[i][q >= 2 ? 1 : 0];
            else if (qi < S)
              del = delta[((size_t)(b * S + qi)) * H + h];
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

    // ---- (3)(4)(5) 沿 head_dim 的 N-tile 循环：GEMM3/4/5 的输出宽度是 HD，
    //      每遍覆盖 NTW=128 列（hd0 = nd*NTW）。B 操作数（dO/Q/K 的转置布局 [K][HD]）列方向
    //      连续，故基址 + hd0 即选中本遍的列；写回列号也加 hd0。HD=128 时 NDT=1、hd0=0，
    //      路径与 O5c/O6c/O7c 逐字等价。----
#pragma unroll
    for (int nd = 0; nd < NDT; ++nd) {
      const int hd0 = nd * NTW;

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
          mma_block_f16<GMV, GNV, BM, true, true>(Ps, LDS, dOs + hd0, LD, acc, wr, wc, lane);
        else
          mma_block_f16<GMV, GNV, BM, true>(Ps, LDP, dOs + hd0, LD, acc, wr, wc, lane);
        const int r0 = wr * GMV, c0 = wc * GNV;
#pragma unroll
        for (int i = 0; i < MTMV; ++i)
#pragma unroll
          for (int j = 0; j < MTNV; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = hd0 + c0 + j * 8 + c2;
              int jg = j0 + r;
              float* dst = dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c;
              if constexpr (R4) {
                // O7c：shfl 必须在 guard 之外（全 warp 参与），只有偶 lane 落 float4。
                float a2 = __shfl_down_sync(0xffffffffu, acc[i][j][q], 1);
                float b2 = __shfl_down_sync(0xffffffffu, acc[i][j][q + 1], 1);
                if (jg < S && (lane & 1) == 0)
                  red_add4(dst, acc[i][j][q], acc[i][j][q + 1], a2, b2);
              } else {
                if (jg < S) red_add2(dst, acc[i][j][q], acc[i][j][q + 1]);
              }
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
          mma_block_f16<GMV, GNV, BM, true, true>(dSs, LDS, Qs + hd0, LD, acc, wr, wc, lane);
        else
          mma_block_f16<GMV, GNV, BM, true>(dSsT, LDP, Qs + hd0, LD, acc, wr, wc, lane);
        const int r0 = wr * GMV, c0 = wc * GNV;
#pragma unroll
        for (int i = 0; i < MTMV; ++i)
#pragma unroll
          for (int j = 0; j < MTNV; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = hd0 + c0 + j * 8 + c2;
              int jg = j0 + r;
              float* dst = dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c;
              if constexpr (R4) {
                float a = acc[i][j][q] * scale, b = acc[i][j][q + 1] * scale;
                float a2 = __shfl_down_sync(0xffffffffu, a, 1);
                float b2 = __shfl_down_sync(0xffffffffu, b, 1);
                if (jg < S && (lane & 1) == 0) red_add4(dst, a, b, a2, b2);
              } else {
                if (jg < S) red_add2(dst, acc[i][j][q] * scale, acc[i][j][q + 1] * scale);
              }
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
        mma_block_f16<GMQ, GNQ, BN, true>(dSs, LDS, Kt + hd0, LD, acc, wr, wc, lane);
#pragma unroll
        for (int i = 0; i < MTMQ; ++i)
#pragma unroll
          for (int j = 0; j < MTNQ; ++j) {
            if constexpr (NDT == 1) {
              // 每个 Q 块由唯一 CTA 独占：寄存器累加，nt 结束后统一写回（O5 原路）。
#pragma unroll
              for (int q = 0; q < 4; ++q) dqacc[i][j][q] += acc[i][j][q] * scale;
            } else {
              // HD>128：dQ 的 HD 列放不进寄存器 → 直接全局累加。每个 (qi, 列) 由唯一线程
              // 拥有（唯一 CTA + 唯一 warp + 唯一 N-tile），非原子 RMW 无竞争；
              // dq_acc 由 host memset(0) 清零。
#pragma unroll
              for (int q = 0; q < 4; q += 2) {
                int r = wr * GMQ + i * 16 + g + (q >= 2 ? 8 : 0);
                int c = hd0 + wc * GNQ + j * 8 + c2;
                int qi = m0 + r;
                if (qi < S) {
                  float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
                  float2 old = *reinterpret_cast<float2*>(base);
                  old.x += acc[i][j][q] * scale;
                  old.y += acc[i][j][q + 1] * scale;
                  *reinterpret_cast<float2*>(base) = old;
                }
              }
            }
          }
      }
    }
    // PIPE=0 需要尾 barrier 保护 Ks/Vs/dSs 在下一轮被覆盖；PIPE=1/2 由下轮
    // 循环首的 barrier 承担（K 写的是另一 stage；dSs/PsT 的覆盖也在下轮首 barrier 之后），
    // 故省掉一次同步。
    if constexpr (PIPE == 0) __syncthreads();
  }

  // ---- 写回 dQ（寄存器累加结果，直接存；每个 Q 块由唯一 CTA 负责）----
  // HD>128（NDT>1）时 dQ 已在 GEMM5 epilogue 里直接全局累加，无需再写回。
  if constexpr (NDT == 1) {
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
            // O10：相邻两列打包成一次 8B 写（c 为偶数 → 自然 8B 对齐），减少全局 store 事务。
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[i][j][q], dqacc[i][j][q + 1]);
          }
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
  // O13：从逐元素「LDG.32 + STG.16」改成 **float4 读 + half2 写**（4 元素/次），减少访存指令与
  // 事务数；尾部不足 4 的元素走标量兜底。d*_acc 为 cudaMalloc 基址（256B 对齐），故 float4 安全。
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t t0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_q4 = n_q / 4;
  for (size_t i = t0; i < n_q4; i += stride) {
    float4 v = reinterpret_cast<const float4*>(dq_acc)[i];
    __half2* o = reinterpret_cast<__half2*>(dq) + i * 2;
    o[0] = __floats2half2_rn(v.x, v.y);
    o[1] = __floats2half2_rn(v.z, v.w);
  }
  for (size_t i = n_q4 * 4 + t0; i < n_q; i += stride) dq[i] = __float2half(dq_acc[i]);
  const size_t n_kv4 = n_kv / 4;
  for (size_t i = t0; i < n_kv4; i += stride) {
    float4 a = reinterpret_cast<const float4*>(dk_acc)[i];
    float4 b = reinterpret_cast<const float4*>(dv_acc)[i];
    __half2* ok = reinterpret_cast<__half2*>(dk) + i * 2;
    __half2* ov = reinterpret_cast<__half2*>(dv) + i * 2;
    ok[0] = __floats2half2_rn(a.x, a.y);
    ok[1] = __floats2half2_rn(a.z, a.w);
    ov[0] = __floats2half2_rn(b.x, b.y);
    ov[1] = __floats2half2_rn(b.z, b.w);
  }
  for (size_t i = n_kv4 * 4 + t0; i < n_kv; i += stride) {
    dk[i] = __float2half(dk_acc[i]);
    dv[i] = __float2half(dv_acc[i]);
  }
}


#endif  // FA_BWD_FP16_MMA_KERNELS_CUH_
