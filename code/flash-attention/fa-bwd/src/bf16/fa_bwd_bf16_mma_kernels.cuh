// =============================================================================
// fa_bwd_bf16_mma_kernels.cuh —— bf16 张量核反向（O5b）**两文件版的 device 部分**
// =============================================================================
// 由 fp16 张量核版 `fa_bwd_fp16_mma_kernels.cuh` 做 **dtype 参数化**而来：
//   `__half` → `__nv_bfloat16`、`mma...f16.f16` → `mma...bf16.bf16`、
//   `__half2float/__float2half` → `__bfloat162float/__float2bfloat16`。
// 算法数据流、线程映射、smem 布局、padding（LD=HD+8/LDP=BM+8/LDS=BN+8）、
// dQ 寄存器累加、dK/dV `red_add2`（float2 atomicAdd）全部与 fp16 版**逐字同构**。
// host 侧见 `fa_bwd_bf16_mma_main.cu`。
//
// O6c：主 kernel 的 2×2 warp 几何由 (BM,BN) 参数化（BM=32 / BN=64 等），host 按网格
// 大小自动选 tile（小网格 BM=32 提并行度、大 S BN=64 降 L1/TEX）；与 fp16 版逐字同构，
// 详见 `docs/01` §13b 与 ROADMAP O6c。
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

#ifndef FA_BWD_FP16_MMA_KERNELS_CUH_
#define FA_BWD_FP16_MMA_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

using bf16 = __nv_bfloat16;

// ----------------------------- 编译期常量 -----------------------------
static constexpr int THREADS = 128;   // 4 warps
static constexpr int WN      = 2;     // N 方向 warp 数（2×2 warp 网格）

// ----------------------------- O11：快速指数/对数 -----------------------------
// 与 fp16 版同源：把 softmax 热点的 libdevice 精确 `expf`/`logf` 换成硬件内建
// `__expf`/`__logf`（MUFU.EX2/LG2，相对误差 ~2^-21）。bf16 容差 ~1e-2，绰绰有余。
// `FAST_EXP=0` 可退回精确版做 A/B。
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
__device__ __forceinline__ void mma_bf16(float c[4], const uint32_t a[4],
                                        const uint32_t b[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// =============================================================================
// O9：Hopper `wgmma`（SS）+ SW128 布局（bf16，与 fp16 版逐字同构）
// =============================================================================
// bf16 与 fp16 同为 2 字节、SW128 布局/描述符/累加器映射**逐字节同构**，只差指令 dtype
// （`mma...f16.f16` → `mma...bf16.bf16`）。冒烟验证见 fp16 的 `fa_bwd_fp16_wgmma_smoke.cu`。
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
__device__ __forceinline__ void wgmma_m64n64k16_bf16(float (&d)[32], uint64_t da,
                                                     uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
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
// O18-bf16：wgmma m64n128k16（N=128，每线程 64 个 fp32 累加器）。累加器布局与 m64n64 同构：
// 行 = wid*16 + lane/4 (+8)，列 = j*8 + (lane&3)*2 (+1)，j=0..15（`d[j*4+qq]`）。
// bf16 与 fp16 同为 2 字节，指令/dtype/SW128 布局逐字节同构（fp16 冒烟
// `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位 PASS），仅 `f16.f16`→`bf16.bf16`。
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n128k16_bf16_t(float (&d)[64], uint64_t da,
                                                        uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
      "%64, %65, p, 1, 1, %67, %68;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]),
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]),
        "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]),
        "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]),
        "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]),
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]),
        "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]),
        "+f"(d[62]), "+f"(d[63])
      : "l"(da), "l"(db), "r"(1), "n"(TA), "n"(TB));
#else
  (void)da; (void)db;
  for (int i = 0; i < 64; ++i) d[i] = 0.f;
#endif
}
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
    wgmma_m64n64k16_bf16(d, da, db);
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
// bf16 字节翻倍 → 改用 cp.async 双缓冲 smem，避免 32 个额外寄存器的代价）。
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
  uint32_t s = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
}

// O6：把一个 K/V 列块（BN 行 × HD 列，bf16）用 cp.async 发进 smem。
// 以 8 个 bf16（16B）为最小搬运单位 → 每行 HD/8 个 unit，共 BN*HD/8 个，按 THREADS 均分。
// 行越界（jg>=S）用普通 smem 写 0（cp.async 无谓词，混合写 + 后续 __syncthreads 可见）。
// O6b：DOK/DOV 分别控制是否发 K / V，便于把二者拆到不同 commit_group、在不同时点预取。
//   * O6  ：DOK=DOV=true（一次发 K+V）。
//   * O6b ：K 用双缓冲、在循环首预取（DOK=true,DOV=false）；V 只单缓冲，在 GEMM2 之后
//            （V 的最后一次使用）才发下一 tile（DOK=false,DOV=true），省下一整个 V 缓冲。
template <int HD, int BN, bool DOK = true, bool DOV = true>
__device__ __forceinline__ void kv_issue_async(const bf16* __restrict__ k,
                                               const bf16* __restrict__ v, int j0, int S,
                                               int Hkv, int hkv, int b, int tid, bf16* Kd,
                                               bf16* Vd, int LD, int qbase = -1) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 bf16) 数
  constexpr int NU  = BN * HDV;  // 总 unit 数
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int jg = j0 + row;
    if (jg < S) {
      const size_t off = (((size_t)(tk + jg)) * Hkv + hkv) * HD + c8 * 8;
      if (DOK) cp_async16(Kd + row * LD + c8 * 8, k + off);
      if (DOV) cp_async16(Vd + row * LD + c8 * 8, v + off);
    } else {
      if (DOK) *reinterpret_cast<uint4*>(Kd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      if (DOV) *reinterpret_cast<uint4*>(Vd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// O10：把一整块 Q/dO（BM 行 × HD 列，bf16）异步发进 smem（与 fp16 版逐字同构）。
template <int HD, int BM>
__device__ __forceinline__ void qdo_issue_async(const bf16* __restrict__ q,
                                                const bf16* __restrict__ do_, int m0, int S,
                                                int H, int h, int b, int tid, bf16* Qd,
                                                bf16* dOd, int LD, int qbase = -1) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 bf16) 数
  constexpr int NU  = BM * HDV;  // 总 unit 数
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += THREADS) {
    const int row = u / HDV, c8 = u % HDV;
    const int qi = m0 + row;
    if (qi < S) {
      const size_t off = (((size_t)(tk + qi)) * H + h) * HD + c8 * 8;
      cp_async16(Qd + row * LD + c8 * 8, q + off);
      cp_async16(dOd + row * LD + c8 * 8, do_ + off);
    } else {
      *reinterpret_cast<uint4*>(Qd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(dOd + row * LD + c8 * 8) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// A[M_TILE][K_TILE] 行主序（行距 asld，bf16）；B 两种布局：
//   BTRANS=false：Bs=[N_TILE][K_TILE] 行主序（行距 bsld，bf16）→ ldmatrix.x2；
//   BTRANS=true ：Bs=[K_TILE][N_TILE] 行主序（行距 bsld，bf16）→ ldmatrix.x2.trans。
// ATRANS=false：As=[M_TILE][K_TILE] 行主序（行距 asld）→ ldmatrix.x4（常态）。
// ATRANS=true ：As=[K_TILE][M_TILE] 行主序（即 A 的转置副本）→ ldmatrix.x4.trans，
//   地址的 bit3/bit4 互换（见 `fa_bwd_bf16_atrans_smoke.cu` 的逐位验证），
//   这样 P/dS 只需存一份 [BM][BN]，省掉一份 `PsT/dSsT`（O6b）。
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
      for (int j = 0; j < MTN; ++j) mma_bf16(acc[i][j], av[i], bv[j]);
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
// 数学与旧版一致（bf16 乘积在 fp32 里累加），且 QKᵀ 的 k-loop 分块顺序与 main GEMM1
// 完全相同 ⇒ LSE 与 main 的 P 自洽，数值只会更稳。
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

    // S_tile = Q·Kᵀ（f16×f16→fp32），每 warp 16×64
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
// O8b：LSE 预处理的两处优化（bf16，causal 专用）
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
lse_mma_kernel_bal(const bf16* __restrict__ q, const bf16* __restrict__ k,
                   float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  constexpr int LD  = HD + 8;
  constexpr int KVL = LBN * LD;
  constexpr int HDV = HD / 8;   // 每行 uint4(8 bf16) 数
  extern __shared__ __align__(16) char smem[];
  bf16* Qs = reinterpret_cast<bf16*>(smem);
  bf16* Ks = Qs + LBM * LD;   // PIPE=1：2 × LBN × LD；PIPE=0：1 × LBN × LD

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // 把 K 的一个 tile（j0 起 LBN 行）写进 Kd：PIPE=1 用 16B cp.async（行越界写 0）；
  // PIPE=0 用普通标量 smem 写（随后由调用处的 __syncthreads 保证可见）。
  auto issue_k = [&](bf16* Kd, int j0) {
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

  // O10：Q 的载入向量化（与 fp16 版逐字同构）。
  auto issue_q = [&](bf16* Qd, int m0) {
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
      bf16* Kt = Ks + (PIPE ? (nt & 1) * KVL : 0);
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
      mma_block_bf16<16, LBN, HD, false>(Qs, LD, Kt, LD, acc, wid, 0, lane);

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
// 与 fp16 版 `fa_bwd_fp16_mma_kernels.cuh` **逐字同构**（仅 dtype 不同）。
template <int HD, int PIPE>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_wgmma(const bf16* __restrict__ q, const bf16* __restrict__ k,
                         float* __restrict__ lse, int S, int H, int Hkv, float scale,
                         const int* __restrict__ cu_seqlens = nullptr) {
  static_assert(HD == 128, "wgmma LSE 目前只做 HD=128");
  constexpr int HDV = HD / 8;
  constexpr int TILE = (LBM / 8) * (HD / 64) * 1024;  // 单个 SW128 tile 字节数（HD=128→16KB）
  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;  // PIPE=1：2*TILE；PIPE=0：TILE

  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  // VARLEN：cu_seqlens 给本序列 token 基址与长度；nullptr 退化为定长 b*S/S。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk = (len + LBM - 1) / LBM;
  if (pair >= (nblk + 1) / 2) return;  // VARLEN：短序列多余配对 CTA 退出
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  auto issue_k = [&](char* Kd, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c8 = u % HDV;
      const int jg = j0 + row;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (jg < len) {
        const size_t off = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c8 * 8;
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
      if (qi < len) {
        const size_t off = (((size_t)(qbase + qi)) * H + h) * HD + c8 * 8;
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
    const int ncols = min(len, m0 + LBM);
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
          if (qi < len && jg < len && jg <= qi) sv = d[j * 4 + q] * scale;
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
        if (qi < len) lse[((size_t)(qbase + qi)) * H + h] = m + flog(l);
      }
    }
    __syncthreads();
  }
}

// =============================================================================
// O31：TMA 版 LSE（bf16）——把 fp16 O30 的 4D-TMA LSE 逐字 dtype 参数化
// =============================================================================
// 与 fp16 `fa_bwd_fp16_mma_kernels.cuh` 的 O30 段**逐字同构**，仅 `wgmma.m64n64k16.f16`
// → `wgmma.m64n64k16.bf16`（bf16 与 fp16 同为 2B，SW128 布局/描述符/TMA box 内维 128B=64
// 元素、SBO=1024 的 2×K=64 chunk 拆分**完全一致**）。数学与 `lse_mma_kernel_bal_wgmma`
// 相同（镜像配对 + online-softmax + 4-lane `shfl` 归约），只换搬运方式，数值应在 fp32
// 求和次序内一致。仅 HD=128、causal（由 host 控制），TMA asm 需 sm_90a（FA_HAS_WGMMA）。
// =============================================================================
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
#if FA_HAS_WGMMA
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(count));
#else
  (void)bar; (void)count;
#endif
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* bar, uint32_t bytes) {
#if FA_HAS_WGMMA
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
      "r"(bytes));
#else
  (void)bar; (void)bytes;
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
#if FA_HAS_WGMMA
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "selp.b32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(smem_u32(bar)), "r"(phase));
  }
#else
  (void)bar; (void)phase;
#endif
}
// 一条 4D TMA：全局张量 dims={D,S,H,B}，把坐标 {k0,row,head,batch} 起的 [64 行][64 列]
// 搬进 dst（SW128 K-major，box 内维 64 bf16 = 128B）。
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int k0,
                                            int r0, int hd, int b, uint64_t* bar) {
#if FA_HAS_WGMMA
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar)));
#else
  (void)dst; (void)map; (void)k0; (void)r0; (void)hd; (void)b; (void)bar;
#endif
}
// 消费两个 K=64 chunk 的 QKᵀ：每个 chunk 用 `SBO=1024` 的描述符（见 fp16 O30）。
__device__ __forceinline__ void wgmma_qkt64_tma(const char* Q0, const char* Q1,
                                                const char* K0, const char* K1,
                                                float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t q0 = smem_u32(Q0), q1 = smem_u32(Q1);
  const uint32_t k0 = smem_u32(K0), k1 = smem_u32(K1);
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    wgmma_m64n64k16_bf16(d, make_desc_sw128(sw128_k16_addr(q0, s), 1024),
                         make_desc_sw128(sw128_k16_addr(k0, s), 1024));
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    wgmma_m64n64k16_bf16(d, make_desc_sw128(sw128_k16_addr(q1, s), 1024),
                         make_desc_sw128(sw128_k16_addr(k1, s), 1024));
  }
  wgmma_commit();
  wgmma_wait0();
}

template <int HD, int PIPE = 1>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_tma(const __grid_constant__ CUtensorMap qmap,
                       const __grid_constant__ CUtensorMap kmap,
                       float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  static_assert(HD == 128, "wgmma LSE TMA 目前只做 HD=128");
  constexpr int CH = (LBM / 8) * 1024;   // 单个 K=64 chunk 的 SW128 字节数（8KB）
  constexpr int TILE = 2 * CH;           // [LBM][HD] K-major tile（16KB）
  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;
  uint64_t* qbar = reinterpret_cast<uint64_t*>(Ks + (PIPE ? 2 : 1) * TILE);
  uint64_t* kbar = qbar + 1;

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  if (tid == 0) {
    mbar_init(qbar, 1);
    mbar_init(kbar, 1);
    if (PIPE) mbar_init(kbar + 1, 1);
  }
  __syncthreads();

  auto issue_q = [&](int m0) {
    if (tid == 0) {
      mbar_arrive_expect(qbar, TILE);
      tma_load_4d(Qs, &qmap, 0, m0, h, b, qbar);
      tma_load_4d(Qs + CH, &qmap, 64, m0, h, b, qbar);
    }
  };
  auto issue_k = [&](int stage, int j0) {
    if (tid == 0) {
      char* Kd = Ks + stage * TILE;
      mbar_arrive_expect(kbar + stage, TILE);
      tma_load_4d(Kd, &kmap, 0, j0, hkv, b, kbar + stage);
      tma_load_4d(Kd + CH, &kmap, 64, j0, hkv, b, kbar + stage);
    }
  };

  int quse = 0;
  int kuse[2] = {0, 0};
#pragma unroll 1
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) break;
    const int m0 = mblk * LBM;
    const int ncols = min(S, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    issue_q(m0);
    if (ntiles > 0) issue_k(0, 0);
    mbar_wait(qbar, (uint32_t)(quse & 1)); quse++;

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int nt = 0; nt < ntiles; ++nt) {
      const int st = PIPE ? (nt & 1) : 0;
      mbar_wait(kbar + st, (uint32_t)(kuse[st] & 1)); kuse[st]++;
      __syncthreads();
      const int j0 = nt * LBN;
      char* Kt = Ks + st * TILE;
      if (PIPE) {
        if (nt + 1 < ntiles) issue_k(st ^ 1, j0 + LBN);
      }

      float d[32];
      wgmma_qkt64_tma(Qs, Qs + CH, Kt, Kt + CH, d);

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
      __syncthreads();  // 所有 warp 读完本 tile 后才能覆盖该 stage / 下一轮 Q
      if (!PIPE && nt + 1 < ntiles) issue_k(0, j0 + LBN);
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

// O24：warp-per-row 向量化版（与 fp16 版逐字同构，仅 `__half2`→`__nv_bfloat162`）。
// 详见 fp16 `fa_bwd_fp16_mma_kernels.cuh` 的 `delta_warp_kernel` 注释。
template <int HD>
__global__ void delta_warp_kernel(const bf16* __restrict__ o,
                                  const bf16* __restrict__ do_, float* __restrict__ delta,
                                  int rows) {
  static_assert(HD % 2 == 0, "delta_warp 需要 HD 为偶数（按 bf162 读）");
  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5;
  const int gwarp0 = blockIdx.x * wpb + (threadIdx.x >> 5);
  const int nwarp = gridDim.x * wpb;
  for (int row = gwarp0; row < rows; row += nwarp) {
    const __nv_bfloat162* o2 = reinterpret_cast<const __nv_bfloat162*>(o + (size_t)row * HD);
    const __nv_bfloat162* d2 = reinterpret_cast<const __nv_bfloat162*>(do_ + (size_t)row * HD);
    float acc = 0.f;
#pragma unroll
    for (int k = lane; k < HD / 2; k += 32) {
      const float2 a = __bfloat1622float2(o2[k]);
      const float2 b = __bfloat1622float2(d2[k]);
      acc += a.x * b.x + a.y * b.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) delta[row] = acc;
  }
}

// =============================================================================
// O9b / O9b-2：主 kernel 的 wgmma 版（bf16，与 fp16 版 `fa_bwd_fp16_mma_kernels.cuh` 逐字同构）
//   O9b：GEMM1/2（S=QKᵀ、dP=dO·Vᵀ）用 `wgmma.m64n64k16` + SW128；
//   O9b-2：GEMM3/4/5（dV=Pᵀ·dO、dK=dSᵀ·Q、dQ=dS·K）也用 `wgmma.m64n64k16`，
//          对 K-major SW128 tile 用 **MN-major 描述符（转置读）**；P/dS 存 SW128 K-major。
//   只做 bf16 / HD=128 / BM=BN=64。
// =============================================================================
// 说明见 fp16 版同段注释与 `docs/01` §14g/§14h、`docs/01b` §6r。bf16 与 fp16 同为 2 字节，
// SW128 布局/描述符/累加器映射逐字节同构，只差 wgmma 指令 dtype（`f16.f16`→`bf16.bf16`）。
// 构建需 `-gencode=arch=compute_90a,code=sm_90a -DFA_WGMMA`。
#ifdef FA_WGMMA

// issue-only 版（不 wait）：GEMM1/GEMM2 两个 wgmma group 一起发、最后统一 wait0，
// 让两条异步 mma 重叠（原 `wgmma_qkt64` 每次内部 wait0，串行）。
__device__ __forceinline__ void wgmma_mn64_issue(const char* Asw, const char* Bsw, int Kd,
                                                 float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((Kd / 64) * 1024);
#pragma unroll
  for (int s = 0; s < Kd / 16; ++s) {
    wgmma_m64n64k16_bf16(d, make_desc_sw128(sw128_k16_addr(aa, s), sbo),
                         make_desc_sw128(sw128_k16_addr(ba, s), sbo));
  }
  wgmma_commit();
}

// O18-bf16：m64n128 的 issue-only 版（A/B 均 K-major，GEMM1/2 用；N=128 = BN）。
__device__ __forceinline__ void wgmma_mn128_issue(const char* Asw, const char* Bsw, int Kd,
                                                  float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((Kd / 64) * 1024);
#pragma unroll
  for (int s = 0; s < Kd / 16; ++s) {
    wgmma_m64n128k16_bf16_t<0, 0>(d, make_desc_sw128(sw128_k16_addr(aa, s), sbo),
                                  make_desc_sw128(sw128_k16_addr(ba, s), sbo));
  }
  wgmma_commit();
}

// A=[M_TILE][K_TILE] 行主序（行距 asld，bf16）；ATRANS=true 时 As 存 [K][M]（ldmatrix.x4.trans）。
// B 为 SW128 K-major tile（行宽 BK=HD），BTRANS 读 [K=token][N=hd]（`ldmatrix.x2.trans`）。
template <int WARP_M, int WARP_N, int K_TILE, bool ATRANS, int BK>
__device__ __forceinline__ void mma_block_swb(const bf16* As, int asld, const char* Bsw,
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
      for (int j = 0; j < MTN; ++j) mma_bf16(acc[i][j], av[i], bv[j]);
  }
}

// 把 K/V（[BN][HD] bf16）用 16B `cp.async` 发进 SW128 tile（DOK/DOV 拆不同 commit_group）。
// O17：NT 为参与搬运的线程数（默认 THREADS；2-warpgroup 版传 256）。
template <int HD, int BN, bool DOK, bool DOV, int NT = THREADS>
__device__ __forceinline__ void kv_issue_async_sw(const bf16* __restrict__ k,
                                                  const bf16* __restrict__ v, int j0, int S,
                                                  int Hkv, int hkv, int b, int tid, char* Kd,
                                                  char* Vd, int qbase = -1) {
  constexpr int HDV = HD / 8;
  constexpr int NU  = BN * HDV;
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += NT) {
    const int row = u / HDV, c8 = u % HDV;
    const int jg = j0 + row;
    if (jg < S) {
      const size_t off = (((size_t)(tk + jg)) * Hkv + hkv) * HD + c8 * 8;
      if (DOK) cp_async16(Kd + sw128_off(row, c8 * 8, HD), k + off);
      if (DOV) cp_async16(Vd + sw128_off(row, c8 * 8, HD), v + off);
    } else {
      if (DOK) *reinterpret_cast<uint4*>(Kd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
      if (DOV) *reinterpret_cast<uint4*>(Vd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// 把 Q/dO（[BM][HD] bf16）用 16B `cp.async` 发进 SW128 tile。
// O17：NT 为参与搬运的线程数（默认 THREADS；2-warpgroup 版传 256）。
template <int HD, int BM, int NT = THREADS>
__device__ __forceinline__ void qdo_issue_async_sw(const bf16* __restrict__ q,
                                                   const bf16* __restrict__ do_, int m0,
                                                   int S, int H, int h, int b, int tid,
                                                   char* Qd, char* dOd, int qbase = -1) {
  constexpr int HDV = HD / 8;
  constexpr int NU  = BM * HDV;
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += NT) {
    const int row = u / HDV, c8 = u % HDV;
    const int qi = m0 + row;
    if (qi < S) {
      const size_t off = (((size_t)(tk + qi)) * H + h) * HD + c8 * 8;
      cp_async16(Qd + sw128_off(row, c8 * 8, HD), q + off);
      cp_async16(dOd + sw128_off(row, c8 * 8, HD), do_ + off);
    } else {
      *reinterpret_cast<uint4*>(Qd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
      *reinterpret_cast<uint4*>(dOd + sw128_off(row, c8 * 8, HD)) = make_uint4(0, 0, 0, 0);
    }
  }
  asm volatile("cp.async.commit_group;\n");
}

// ---- O9b-2：MN-major（转置）描述符（bf16，与 fp16 版逐字同构；dtype 无关）----
// 同一份 **K-major SW128** 存储的 tile，用 `Major::MN` 描述符 + `tnsp=1` 读，等于读它的转置
// （FA3 `dKV_swapAB` 的做法）。逐位验证见 fp16 的 `fa_bwd_fp16_wgmma_bwd_smoke.cu`：
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
__device__ __forceinline__ void wgmma_m64n64k16_bf16_t(float (&d)[32], uint64_t da,
                                                       uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
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
// 相邻 2 列）。`a` 为 row r0、`b` 为 row r0+8。
__device__ __forceinline__ void pds_store_sw128(char* tile, int j, int r0, int lane,
                                                 int c2, int W, bf16 a0, bf16 a1,
                                                 bf16 b0, bf16 b1) {
  const uint32_t h2a = (uint32_t)__bfloat16_as_ushort(a0) |
                       ((uint32_t)__bfloat16_as_ushort(a1) << 16);
  const uint32_t h2b = (uint32_t)__bfloat16_as_ushort(b0) |
                       ((uint32_t)__bfloat16_as_ushort(b1) << 16);
  const int c = j * 8 + c2;
  *reinterpret_cast<uint32_t*>(tile + sw128_off(r0, c, W)) = h2a;
  *reinterpret_cast<uint32_t*>(tile + sw128_off(r0 + 8, c, W)) = h2b;
  (void)lane;
}

template <int HD>
__global__ void __launch_bounds__(THREADS, 2)
fa_bwd_bf16_wgmma_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                         const bf16* __restrict__ v, const bf16* __restrict__ do_,
                         const float* __restrict__ delta, const float* __restrict__ lse,
                         float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                         float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                         int causal, int sched) {
  static_assert(HD == 128, "wgmma 主 kernel 目前只做 HD=128");
  constexpr int BM = 64, BN = 64;
  constexpr int TILE  = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（16KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（16KB）
  constexpr int PSZ   = BM * BN * 2;                   // P/dS SW128 tile 字节数（8KB）

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

  // dQ 寄存器累加（每个 Q 块唯一 CTA，无跨 CTA 原子）。
  float dqacc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) dqacc[i][j][qq] = 0.f;

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
                      __float2bfloat16(pval[j][0]), __float2bfloat16(pval[j][1]),
                      __float2bfloat16(pval[j][2]), __float2bfloat16(pval[j][3]));
    // (2) epilogue：dS = P∘(dP−D)，按 SW128 16B 分块写 dSs
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSs, j, r0, lane, c2, BN,
                      __float2bfloat16(d0), __float2bfloat16(d1),
                      __float2bfloat16(d2), __float2bfloat16(d3));
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
        wgmma_m64n64k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
      wgmma_commit();
#pragma unroll
      for (int s = 0; s < BM / 16; ++s)
        wgmma_m64n64k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
      wgmma_commit();
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_bf16_t<0, 1>(accq, desc_k16_k(DSa, s, BN), desc_k16_mn(Kn, s, HD));
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

// =============================================================================
// O17-bf16：2 warpgroup（BM=128）wgmma 主 kernel —— **跨 warpgroup 归约**，把 dK/dV 的
//      跨 CTA atomic 字节数砍半。（与 fp16 `fa_bwd_fp16_wgmma2_kernel` 逐字 dtype 同构）
// =============================================================================
// 动机（O15a/O17 ncu）：S=4096 主 kernel 的 L2 扇区里 `red`（dK/dV 的跨 CTA `atomicAdd`）
// 占 73.1%、DRAM 仅 4.2% ⇒ main 是 **L2 原子字节数 bound**。一个 KV 元素被 `nblk` 个 CTA
// 贡献一次偏和；把每个 CTA 覆盖的 Q 行从 BM=64 扩到 **BM=128**，贡献它的 CTA 数从 `nblk`
// 降到 `nblk/2` ⇒ **red 字节直接砍半**。O16（错开 wait）与 O7c（float4 归约）都已证明
// 「动等待/事务数」无效，唯一杠杆是减少每个元素的贡献 CTA 数。
//
// 结构（HD=128, BM=128, BN=64, 256 线程 = 2 warpgroups）：
//   * 2 个 warpgroup 各持自己 64 行 Q/dO，各自算 GEMM1/2（S、dP）并把 P/dS 按 SW128
//     写进**共享**的 [128][BN] tile（wg 写自己的 64 行）；
//   * **GEMM3/4 只由 wg0 做，且对全 BM=128 归约**：把两个 m64 半（s=0..7，即 128 行）
//     连续喂给**同一个** `wgmma.m64n64k16` 累加器（输出都是同一个 [BN][HD]），得到的
//     就是两半之和 ⇒ **每个 KV 元素一次 red**（fp16 冒烟 `fa_bwd_fp16_wgmma2_smoke.cu`
//     逐位验证「[128][*] K-major tile + MN-major 描述符 s=0..7 转置读」，max_abs=0；
//     bf16 与 fp16 同为 2 字节，SW128 布局/描述符/累加器映射逐字节同构）；
//     wg1 同时做自己的 GEMM5（dQ 只依赖本 wg 的行，互不干扰）；
//   * GEMM5（dQ）两个 wg 各算自己 64 行、寄存器累加，无需跨 CTA 原子；
//   * K/V 用 NTH=256 线程 `cp.async`：K 双缓冲、V 单缓冲后段预取（同 O6b/O9b）。
//
// smem（HD=128）：Q 32KB + dO 32KB + K 双缓冲 32KB + V 16KB + P 16KB + dS 16KB ≈ 145KB
// → 1 CTA/SM（256 线程 = 8 warps/SM，与 O9b 的 2 CTA/SM × 4 warps 相同）。数值只改
// 归约次序（atomic 顺序），与 O9b 在 bf16 噪声内一致。
template <int HD, bool SPLIT = true>
__global__ void __launch_bounds__(256, 1)
fa_bwd_bf16_wgmma2_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                          const bf16* __restrict__ v, const bf16* __restrict__ do_,
                          const float* __restrict__ delta, const float* __restrict__ lse,
                          float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                          float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                          int causal, bf16* __restrict__ dq_h = nullptr,
                          const int* __restrict__ cu_seqlens = nullptr) {
  static_assert(HD == 128, "wgmma2 主 kernel 目前只做 HD=128");
  constexpr int NTH = 256;
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（32KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（16KB）
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;   // P/dS [128][64] SW128 tile（16KB）

  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + QTILE;
  char* Ks  = dOs + QTILE;                 // 双缓冲 2*KTILE
  char* Vs  = Ks + 2 * KTILE;              // 单缓冲 KTILE
  char* Ps  = Vs + KTILE;                  // [128][64] SW128（16KB）
  char* dSs = Ps + PTILE;                  // [128][64] SW128（16KB）

  const int bx = blockIdx.x;
  const int mblk = bx;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  // VARLEN：cu_seqlens 给本序列 token 基址与长度；nullptr 退化为定长 b*S/S。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int tid = threadIdx.x;
  const int wg = tid >> 7;                 // warpgroup id（0/1）
  const int wid = (tid >> 5) & 3;          // warpgroup 内 warp id（0..3）
  const int lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  qdo_issue_async_sw<HD, BM, NTH>(q, do_, m0, len, H, h, b, tid, Qs, dOs, qbase);

  const int ncols = causal ? min(len, m0 + BM) : len;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0) {
    kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, 0, len, Hkv, hkv, b, tid, Ks, Vs, qbase);
    kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, 0, len, Hkv, hkv, b, tid, Ks, Vs, qbase);
  }

  // 本 wg 的 Q 行 = wg*64 + [0,64)。每线程两行（r_lo / r_hi）。
  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < len) {
    const size_t idx = ((size_t)(qbase + qi_lo)) * H + h;
    lse_lo = lse[idx]; del_lo = delta[idx];
  }
  if (qi_hi < len) {
    const size_t idx = ((size_t)(qbase + qi_hi)) * H + h;
    lse_hi = lse[idx]; del_hi = delta[idx];
  }

  // dQ 寄存器累加（本 wg 的 64 行 × HD 列）：dqacc[nh][j][q]。
  float dqacc[2][8][4];
#pragma unroll
  for (int nh = 0; nh < 2; ++nh)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] = 0.f;

  // 本 wg 的 Q/dO/P/dS tile 基址（每 64 行 = 8 rowgroup）。
  char* Qw  = Qs  + wg * (64 / 8) * (HD / 64) * 1024;
  char* dOw = dOs + wg * (64 / 8) * (HD / 64) * 1024;
  char* Pw  = Ps  + wg * (64 / 8) * (BN / 64) * 1024;
  char* dSw = dSs + wg * (64 / 8) * (BN / 64) * 1024;

  const uint32_t Qa = smem_u32(Qs), dOa = smem_u32(dOs);
  const uint32_t Pa = smem_u32(Ps), DSa = smem_u32(dSs);
  const int r0 = wid * 16 + g;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    char* Kt = Ks + (nt & 1) * KTILE;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, (nt + 1) * BN, len, Hkv, hkv, b, tid,
                                                  Ks + ((nt + 1) & 1) * KTILE, Vs, qbase);

    // ---- (1)(2) 本 wg 的 S=QKᵀ 与 dP=dO·Vᵀ（m64n64），统一 wait0 ----
    float sacc[32], dpacc[32];
    wgmma_mn64_issue(Qw, Kt, HD, sacc);
    wgmma_mn64_issue(dOw, Vs, HD, dpacc);
    wgmma_wait0();
    float pval[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
        const int jg = j0 + j * 8 + c2 + (qq & 1);
        const float lv = (qq >= 2) ? lse_hi : lse_lo;
        float p = 0.f;
        if (qi < len && jg < len && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      pds_store_sw128(Pw, j, r0, lane, c2, BN,
                      __float2bfloat16(pval[j][0]), __float2bfloat16(pval[j][1]),
                      __float2bfloat16(pval[j][2]), __float2bfloat16(pval[j][3]));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN,
                      __float2bfloat16(d0), __float2bfloat16(d1),
                      __float2bfloat16(d2), __float2bfloat16(d3));
    }
    // 两个 wg 的 P/dS 都写好；同时 GEMM2 已读完 V[nt]，可覆盖 V。
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, (nt + 1) * BN, len, Hkv, hkv, b, tid,
                                                  Ks, Vs, qbase);

    if constexpr (SPLIT) {
      // ---- O17-2：GEMM3(dV)→wg0、GEMM4(dK)→wg1，两者都仍对全 BM=128 归约。原版只有
      //      wg0 串行做 dV+dK（张量工作量 3:1 失衡、4 条 red 链串行）；拆分后两 wg 各一条
      //      GEMM+red 链并发、工作量 1:1。每个 KV 元素仍只 `red` 一次。与 fp16 逐字同构。----
      if (wg == 0) {
        // ---- (3) dV = Pᵀ·dO（wg0，全 BM）----
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float accv[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) accv[i] = 0.f;
          const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
          wgmma_commit();
          wgmma_wait0();
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + rr;
              const int c = nh * 64 + j * 8 + c2;
              if (jg < len)
                red_add2(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c,
                         accv[j * 4 + qq], accv[j * 4 + qq + 1]);
            }
        }
      } else {
        // ---- (4) dK = scale·dSᵀ·Q（wg1，全 BM）----
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float acck[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) acck[i] = 0.f;
          const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
          wgmma_commit();
          wgmma_wait0();
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + rr;
              const int c = nh * 64 + j * 8 + c2;
              if (jg < len)
                red_add2(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c,
                         acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
            }
        }
      }
    } else {
    if (wg == 0) {
      // ---- (3) dV = Pᵀ·dO：wg0 对全 BM=128 归约（s=0..7 两个 m64 半进同一累加器）----
      wgmma_fence();
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        float accv[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) accv[i] = 0.f;
        const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
        for (int s = 0; s < BM / 16; ++s)
          wgmma_m64n64k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
        wgmma_commit();
        wgmma_wait0();
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int qq = 0; qq < 4; qq += 2) {
            const int rr = r0 + (qq >= 2 ? 8 : 0);
            const int jg = j0 + rr;
            const int c = nh * 64 + j * 8 + c2;
            if (jg < len)
              red_add2(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c,
                       accv[j * 4 + qq], accv[j * 4 + qq + 1]);
          }
      }
      // ---- (4) dK = scale·dSᵀ·Q：同构 ----
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        float acck[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) acck[i] = 0.f;
        const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
        for (int s = 0; s < BM / 16; ++s)
          wgmma_m64n64k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
        wgmma_commit();
        wgmma_wait0();
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int qq = 0; qq < 4; qq += 2) {
            const int rr = r0 + (qq >= 2 ? 8 : 0);
            const int jg = j0 + rr;
            const int c = nh * 64 + j * 8 + c2;
            if (jg < len)
              red_add2(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c,
                       acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
          }
      }
    }
    }

    // ---- (5) dQ += scale·dS·K：每个 wg 算自己 64 行（A=本 wg 的 dS 行块）----
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      float accq[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) accq[i] = 0.f;
      wgmma_fence();
      const uint32_t Kt_n = smem_u32(Kt) + (uint32_t)(nh * 1024);
      const uint32_t dSw_a = smem_u32(dSw);
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_bf16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit();
      wgmma_wait0();
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
        const int qi = m0 + wg * 64 + rr;
        const int c = nh * 64 + j * 8 + c2;
        if (qi < len) {
          // O24：dQ 唯一拥有 ⇒ 可直接写 bf16，省掉 convert 的 dQ 一趟。
          if (dq_h)
            *reinterpret_cast<__nv_bfloat162*>(dq_h + (((size_t)(qbase + qi)) * H + h) * HD + c) =
                __floats2bfloat162_rn(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          else {
            float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          }
        }
      }
}

// =============================================================================
// O18-bf16：BN=128 版 wgmma2 主 kernel —— tile 数减半，降每-tile 的 barrier / wgmma
//   固定开销。（与 fp16 `fa_bwd_fp16_wgmma2b_kernel` 逐字 dtype 同构）
// =============================================================================
// 动机（docs §14k.7 item 3 / §14m）：O17（BM=128,BN=64）后 main 的第一墙是 L2 原子
// （`red` 占 L2 扇区 ~72%），且 occupancy 只有 1 CTA/SM（12.5%）⇒ 延迟受限。BN 从 64→128
// 让 per-CTA 的 tile 数减半 ⇒ `__syncthreads` / `cp.async.wait` / wgmma commit-wait 序列都减半；
// GEMM1/2 用 `m64n128k16`（一条指令算两倍），发射/依赖链也减半。代价 smem 148→224KB（仍 1 CTA/SM）。
//
// 结构（HD=128, BM=128, BN=128, 256 线程 = 2 warpgroup）与 O17 完全同构，仅：
//   * GEMM1/2 用 `wgmma_mn128_issue`（m64n128，A/B 均 K-major）；
//   * P/dS tile 变 [128][128]，softmax epilogue 列组 j=0..15；
//   * GEMM3/4 输出 [BN=128][HD=128] ⇒ 2 个 m64 半（mh=0/1），转置描述符基址 +mh*1024
//     （存储列 64 的 SW128 kg=1 atom；fp16 冒烟 `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位 PASS，
//     bf16 与 fp16 同为 2 字节、布局逐字节同构）；
//   * GEMM5 输出 [64][HD=128]（每 wg 自己 64 行）⇒ 一条 m64n128，`dqacc[16][4]`。
// 数值只改跨 CTA `atomicAdd` 次序，与 O17 在 bf16 噪声内一致。
template <int HD, bool SPLIT = true>
__global__ void __launch_bounds__(256, 1)
fa_bwd_bf16_wgmma2b_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                           const bf16* __restrict__ v, const bf16* __restrict__ do_,
                           const float* __restrict__ delta, const float* __restrict__ lse,
                           float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                           float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                           int causal, bf16* __restrict__ dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b 主 kernel 目前只做 HD=128");
  constexpr int NTH = 256;
  constexpr int BM = 128, BN = 128;
  constexpr int NG = BN / 8;                           // 列组数（16）
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（32KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（32KB）
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;   // P/dS [128][128] SW128 tile（32KB）

  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + QTILE;
  char* Ks  = dOs + QTILE;                 // 双缓冲 2*KTILE
  char* Vs  = Ks + 2 * KTILE;              // 单缓冲 KTILE
  char* Ps  = Vs + KTILE;
  char* dSs = Ps + PTILE;

  const int mblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wid = (tid >> 5) & 3;
  const int lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  qdo_issue_async_sw<HD, BM, NTH>(q, do_, m0, S, H, h, b, tid, Qs, dOs);

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0) {
    kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs);
    kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, 0, S, Hkv, hkv, b, tid, Ks, Vs);
  }

  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < S) { const size_t idx = ((size_t)(b * S + qi_lo)) * H + h; lse_lo = lse[idx]; del_lo = delta[idx]; }
  if (qi_hi < S) { const size_t idx = ((size_t)(b * S + qi_hi)) * H + h; lse_hi = lse[idx]; del_hi = delta[idx]; }

  float dqacc[NG][4];
#pragma unroll
  for (int j = 0; j < NG; ++j)
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) dqacc[j][qq] = 0.f;

  char* Qw  = Qs  + wg * (64 / 8) * (HD / 64) * 1024;
  char* dOw = dOs + wg * (64 / 8) * (HD / 64) * 1024;
  char* Pw  = Ps  + wg * (64 / 8) * (BN / 64) * 1024;
  char* dSw = dSs + wg * (64 / 8) * (BN / 64) * 1024;

  const uint32_t Qa = smem_u32(Qs), dOa = smem_u32(dOs);
  const uint32_t Pa = smem_u32(Ps), DSa = smem_u32(dSs);
  const int r0 = wid * 16 + g;

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    char* Kt = Ks + (nt & 1) * KTILE;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                                  Ks + ((nt + 1) & 1) * KTILE, Vs);

    // ---- (1)(2) S=QKᵀ、dP=dO·Vᵀ：m64n128（A/B 均 K-major），统一 wait0 ----
    float sacc[NG * 4], dpacc[NG * 4];
    wgmma_mn128_issue(Qw, Kt, HD, sacc);
    wgmma_mn128_issue(dOw, Vs, HD, dpacc);
    wgmma_wait0();
    float pval[NG][4];
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
        const int jg = j0 + j * 8 + c2 + (qq & 1);
        const float lv = (qq >= 2) ? lse_hi : lse_lo;
        float p = 0.f;
        if (qi < S && jg < S && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < NG; ++j)
      pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2bfloat16(pval[j][0]),
                      __float2bfloat16(pval[j][1]), __float2bfloat16(pval[j][2]),
                      __float2bfloat16(pval[j][3]));
#pragma unroll
    for (int j = 0; j < NG; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2bfloat16(d0), __float2bfloat16(d1),
                      __float2bfloat16(d2), __float2bfloat16(d3));
    }
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                                  Ks, Vs);

    if constexpr (SPLIT) {
      // O17-2：GEMM3(dV)→wg0、GEMM4(dK)→wg1，都对全 BM=128 归约（dK 乘 scale）。
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float accv[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) accv[i] = 0.f;
          const uint32_t Pa_m = Pa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          // 本 mh 的输出行偏移 = mh*64（红 epilogue 里加）。
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                                   accv[j * 4 + qq], accv[j * 4 + qq + 1]);
            }
        }
      } else {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float acck[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) acck[i] = 0.f;
          const uint32_t DSa_m = DSa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                                   acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
            }
        }
      }
    } else {
      // 非拆分：wg0 串行做 dV 与 dK。
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float accv[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) accv[i] = 0.f;
          const uint32_t Pa_m = Pa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                                   accv[j * 4 + qq], accv[j * 4 + qq + 1]);
            }
        }
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float acck[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) acck[i] = 0.f;
          const uint32_t DSa_m = DSa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                                   acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
            }
        }
      }
    }

    // ---- (5) dQ = dS·K：每 wg 算自己 64 行，一条 m64n128，寄存器累加 ----
    {
      float accq[NG * 4];
#pragma unroll
      for (int i = 0; i < NG * 4; ++i) accq[i] = 0.f;
      wgmma_fence();
      const uint32_t dSw_a = smem_u32(dSw);
      const uint32_t Kt_n = smem_u32(Kt);
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n128k16_bf16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit(); wgmma_wait0();
#pragma unroll
      for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) dqacc[j][qq] += accq[j * 4 + qq] * scale;
    }
  }

#pragma unroll
  for (int j = 0; j < NG; ++j)
#pragma unroll
    for (int qq = 0; qq < 4; qq += 2) {
      const int rr = r0 + (qq >= 2 ? 8 : 0);
      const int qi = m0 + wg * 64 + rr;
      const int c = j * 8 + c2;
      if (qi < S) {
        // O24：dQ 唯一拥有 ⇒ 可直接写 bf16，省掉 convert 的 dQ 一趟。
        if (dq_h)
          *reinterpret_cast<__nv_bfloat162*>(dq_h + (((size_t)(b * S + qi)) * H + h) * HD + c) =
              __floats2bfloat162_rn(dqacc[j][qq], dqacc[j][qq + 1]);
        else {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) = make_float2(dqacc[j][qq], dqacc[j][qq + 1]);
        }
      }
    }
}

// =============================================================================
// O34：主 kernel 的 Q/K/V/dO 改用 4D-TMA 载入（把 O30–O32 的「TMA 化 operand」从 LSE
//       推到主 kernel）。
// =============================================================================
// 动机：O30/O31/O32 已把 LSE 的 Q/K 换成 4D-TMA（省 load 指令/地址运算）。主 kernel 的
//   Q/dO（prologue）与 K/V（每个 KV tile）仍用逐 16B `cp.async` + `sw128_off` 地址运算。
//   TMA 一个 box 内维固定 128B（bf16 = 64 元素）——对 HD=128 的 K-major tile，一个
//   `[8 行][64 列]` 的 box 恰好等于一个 1024B SW128 atom。因此**逐 atom 发 TMA**（每个
//   `(rg,kg)` 一条），把 atom 放到 `sw128_off(rg*8, kg*64, HD)`（= `(rg*(HD/64)+kg)*1024`）
//   的位置，就**原样复现了交织布局**（atom 次序 `rg*(HD/64)+kg`）⇒ 所有 wgmma 描述符
//   **完全不用改**，数值与 cp.async 路径**应逐位相同**（搬的是同样字节）。
//   * Q/dO 各 32 atom、K/V 各 32 atom；由 tid0 串行发射（异步、不占寄存器/不记
//     scoreboard），比 4096 条 cp.async 摊到 256 线程更省发射；OOB 行由 TMA 自动补 0。
//   * 同步改为 mbarrier（`mbar_init`/`mbar_arrive_expect`/`mbar_wait`，同 O30）。Q/dO 一次性；
//     K 双缓冲两个 barrier（phase 每 `nt` 翻一次）；V 单缓冲一个 barrier（后段预取，同 O6b）。
//   * 仅 `-DFA_WGMMA -DFA_TMA -lcuda` 构建、HD=128、BN=128（wgmma2b 几何）时由 host 选用；
//     作为 `--maintma` opt-in，与 cp.async 版做同 binary A/B。
template <int R, int HD_>
__device__ __forceinline__ void tma_fill_sw128(char* tile, const CUtensorMap* map, int r0,
                                               int hd, int b, uint64_t* bar) {
#if FA_HAS_WGMMA
  constexpr int NG = HD_ / 64;
#pragma unroll
  for (int rg = 0; rg < R / 8; ++rg)
#pragma unroll
    for (int kg = 0; kg < NG; ++kg) {
      char* dst = tile + (rg * NG + kg) * 1024;
      tma_load_4d(dst, map, kg * 64, r0 + rg * 8, hd, b, bar);
    }
#else
  (void)tile; (void)map; (void)r0; (void)hd; (void)b; (void)bar;
#endif
}

template <int HD, bool SPLIT = true>
__global__ void __launch_bounds__(256, 1)
fa_bwd_bf16_wgmma2b_tma_kernel(
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __grid_constant__ CUtensorMap dmap,
    const float* __restrict__ delta, const float* __restrict__ lse,
    float* __restrict__ dq_acc, float* __restrict__ dk_acc, float* __restrict__ dv_acc,
    int S, int H, int Hkv, float scale, int causal,
    bf16* __restrict__ dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b TMA 主 kernel 目前只做 HD=128");
  constexpr int NTH = 256;
  constexpr int BM = 128, BN = 128;
  constexpr int NG = BN / 8;                           // 列组数（16）
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（32KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（32KB）
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;   // P/dS [128][128] SW128 tile（32KB）

  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + QTILE;
  char* Ks  = dOs + QTILE;                 // 双缓冲 2*KTILE
  char* Vs  = Ks + 2 * KTILE;              // 单缓冲 KTILE
  char* Ps  = Vs + KTILE;
  char* dSs = Ps + PTILE;
  uint64_t* bars = reinterpret_cast<uint64_t*>(dSs + PTILE);  // qbar,kbar0,kbar1,vbar

  const int mblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int wid = (tid >> 5) & 3;
  const int lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  if (tid == 0) {
    mbar_init(bars + 0, 1);
    mbar_init(bars + 1, 1);
    mbar_init(bars + 2, 1);
    mbar_init(bars + 3, 1);
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect(bars + 0, 2 * QTILE);
    tma_fill_sw128<BM, HD>(Qs, &qmap, m0, h, b, bars + 0);
    tma_fill_sw128<BM, HD>(dOs, &dmap, m0, h, b, bars + 0);
  }

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0 && tid == 0) {
    mbar_arrive_expect(bars + 1, KTILE);
    tma_fill_sw128<BN, HD>(Ks, &kmap, 0, hkv, b, bars + 1);
    mbar_arrive_expect(bars + 3, KTILE);
    tma_fill_sw128<BN, HD>(Vs, &vmap, 0, hkv, b, bars + 3);
  }

  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < S) { const size_t idx = ((size_t)(b * S + qi_lo)) * H + h; lse_lo = lse[idx]; del_lo = delta[idx]; }
  if (qi_hi < S) { const size_t idx = ((size_t)(b * S + qi_hi)) * H + h; lse_hi = lse[idx]; del_hi = delta[idx]; }

  float dqacc[NG][4];
#pragma unroll
  for (int j = 0; j < NG; ++j)
#pragma unroll
    for (int qq = 0; qq < 4; ++qq) dqacc[j][qq] = 0.f;

  char* Qw  = Qs  + wg * (64 / 8) * (HD / 64) * 1024;
  char* dOw = dOs + wg * (64 / 8) * (HD / 64) * 1024;
  char* Pw  = Ps  + wg * (64 / 8) * (BN / 64) * 1024;
  char* dSw = dSs + wg * (64 / 8) * (BN / 64) * 1024;

  const uint32_t Qa = smem_u32(Qs), dOa = smem_u32(dOs);
  const uint32_t Pa = smem_u32(Ps), DSa = smem_u32(dSs);
  const int r0 = wid * 16 + g;

  int ku[2] = {0, 0}, vu = 0, qu = 0;
  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    const int st = nt & 1;
    char* Kt = Ks + st * KTILE;
    if (nt == 0) { mbar_wait(bars + 0, (uint32_t)(qu & 1)); qu++; }
    mbar_wait(bars + 1 + st, (uint32_t)(ku[st] & 1)); ku[st]++;
    mbar_wait(bars + 3, (uint32_t)(vu & 1)); vu++;
    __syncthreads();
    if (nt + 1 < ntiles && tid == 0) {
      mbar_arrive_expect(bars + 1 + (st ^ 1), KTILE);
      tma_fill_sw128<BN, HD>(Ks + (st ^ 1) * KTILE, &kmap, (nt + 1) * BN, hkv, b,
                             bars + 1 + (st ^ 1));
    }

    // ---- (1)(2) S=QKᵀ、dP=dO·Vᵀ：m64n128（A/B 均 K-major），统一 wait0 ----
    float sacc[NG * 4], dpacc[NG * 4];
    wgmma_mn128_issue(Qw, Kt, HD, sacc);
    wgmma_mn128_issue(dOw, Vs, HD, dpacc);
    wgmma_wait0();
    float pval[NG][4];
#pragma unroll
    for (int j = 0; j < NG; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
        const int jg = j0 + j * 8 + c2 + (qq & 1);
        const float lv = (qq >= 2) ? lse_hi : lse_lo;
        float p = 0.f;
        if (qi < S && jg < S && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < NG; ++j)
      pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2bfloat16(pval[j][0]),
                      __float2bfloat16(pval[j][1]), __float2bfloat16(pval[j][2]),
                      __float2bfloat16(pval[j][3]));
#pragma unroll
    for (int j = 0; j < NG; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2bfloat16(d0), __float2bfloat16(d1),
                      __float2bfloat16(d2), __float2bfloat16(d3));
    }
    __syncthreads();
    if (nt + 1 < ntiles && tid == 0) {
      mbar_arrive_expect(bars + 3, KTILE);
      tma_fill_sw128<BN, HD>(Vs, &vmap, (nt + 1) * BN, hkv, b, bars + 3);
    }

    if constexpr (SPLIT) {
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float accv[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) accv[i] = 0.f;
          const uint32_t Pa_m = Pa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S)
                red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                         accv[j * 4 + qq], accv[j * 4 + qq + 1]);
            }
        }
      } else {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float acck[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) acck[i] = 0.f;
          const uint32_t DSa_m = DSa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S)
                red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                         acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
            }
        }
      }
    } else {
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float accv[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) accv[i] = 0.f;
          const uint32_t Pa_m = Pa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S)
                red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                         accv[j * 4 + qq], accv[j * 4 + qq + 1]);
            }
        }
#pragma unroll
        for (int mh = 0; mh < BN / 64; ++mh) {
          float acck[NG * 4];
#pragma unroll
          for (int i = 0; i < NG * 4; ++i) acck[i] = 0.f;
          const uint32_t DSa_m = DSa + (uint32_t)(mh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n128k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S)
                red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                         acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
            }
        }
      }
    }

    // ---- (5) dQ = dS·K：每 wg 算自己 64 行，一条 m64n128，寄存器累加 ----
    {
      float accq[NG * 4];
#pragma unroll
      for (int i = 0; i < NG * 4; ++i) accq[i] = 0.f;
      wgmma_fence();
      const uint32_t dSw_a = smem_u32(dSw);
      const uint32_t Kt_n = smem_u32(Kt);
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n128k16_bf16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit(); wgmma_wait0();
#pragma unroll
      for (int j = 0; j < NG; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) dqacc[j][qq] += accq[j * 4 + qq] * scale;
    }
  }

#pragma unroll
  for (int j = 0; j < NG; ++j)
#pragma unroll
    for (int qq = 0; qq < 4; qq += 2) {
      const int rr = r0 + (qq >= 2 ? 8 : 0);
      const int qi = m0 + wg * 64 + rr;
      const int c = j * 8 + c2;
      if (qi < S) {
        if (dq_h)
          *reinterpret_cast<__nv_bfloat162*>(dq_h + (((size_t)(b * S + qi)) * H + h) * HD + c) =
              __floats2bfloat162_rn(dqacc[j][qq], dqacc[j][qq + 1]);
        else {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) = make_float2(dqacc[j][qq], dqacc[j][qq + 1]);
        }
      }
    }
}
// =============================================================================
// O36：BN=64 版 wgmma2 主 kernel 的 Q/K/V/dO 改用逐 atom 4D-TMA（对齐 fp16 O35）
// =============================================================================
// 动机：O34 只把 **BN=128** 的 `wgmma2b` 主 kernel 的 Q/K/V/dO 换成 4D-TMA；而 O23 的
//   默认档在 **S<4096 与 GQA/MQA** 走的是 **BN=64 的 `wgmma2`**——这条更常用的路仍用
//   逐 16B `cp.async` + `sw128_off` 地址运算。本 kernel 把 O34 的做法搬到 BN=64 几何：
//   逐 `[8 行][64 列]` atom 发 TMA（box 内维 128B=64 个 bf16、8 行=一个 1024B atom，
//   原样复现 SW128 交织布局、wgmma 描述符零改动）⇒ Q/dO 各 32 atom、K/V 各 16 atom。
//   K 双缓冲两个 barrier、V 单缓冲后段预取（同 O6b/O17/O34）。
//   * 数学与 `fa_bwd_bf16_wgmma2_kernel` **完全一致**（同一 wgmma 次序、同一 epilogue），
//     只换搬运方式 ⇒ 数值应逐位相同（差异仅跨 CTA `atomicAdd` 次序）。作为 `--maintma`
//     的 BN=64 分支，与 cp.async 版做同 binary A/B。
//   * 仅 `-DFA_WGMMA -DFA_TMA -lcuda` 构建、HD=128 时由 host 选用；`sm_90` 构建里
//     mbar/tma asm 退化为空实现、host 不会 launch。
// =============================================================================
template <int HD, bool SPLIT = true>
__global__ void __launch_bounds__(256, 1)
fa_bwd_bf16_wgmma2_tma_kernel(
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __grid_constant__ CUtensorMap dmap,
    const float* __restrict__ delta, const float* __restrict__ lse,
    float* __restrict__ dq_acc, float* __restrict__ dk_acc, float* __restrict__ dv_acc,
    int S, int H, int Hkv, float scale, int causal, bf16* __restrict__ dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2 TMA 主 kernel 目前只做 HD=128");
  constexpr int NTH = 256;
  constexpr int BM = 128, BN = 64;
  constexpr int NG = BN / 8;                           // 列组数（8）
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（32KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（16KB）
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;   // P/dS [128][64] SW128 tile（16KB）

  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + QTILE;
  char* Ks  = dOs + QTILE;                 // 双缓冲 2*KTILE
  char* Vs  = Ks + 2 * KTILE;              // 单缓冲 KTILE
  char* Ps  = Vs + KTILE;                  // [128][64] SW128（16KB）
  char* dSs = Ps + PTILE;                  // [128][64] SW128（16KB）
  uint64_t* bars = reinterpret_cast<uint64_t*>(dSs + PTILE);  // qbar,kbar0,kbar1,vbar

  const int mblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;                 // warpgroup id（0/1）
  const int wid = (tid >> 5) & 3;          // warpgroup 内 warp id（0..3）
  const int lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  if (tid == 0) {
    mbar_init(bars + 0, 1);
    mbar_init(bars + 1, 1);
    mbar_init(bars + 2, 1);
    mbar_init(bars + 3, 1);
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect(bars + 0, 2 * QTILE);
    tma_fill_sw128<BM, HD>(Qs, &qmap, m0, h, b, bars + 0);
    tma_fill_sw128<BM, HD>(dOs, &dmap, m0, h, b, bars + 0);
  }

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0 && tid == 0) {
    mbar_arrive_expect(bars + 1, KTILE);
    tma_fill_sw128<BN, HD>(Ks, &kmap, 0, hkv, b, bars + 1);
    mbar_arrive_expect(bars + 3, KTILE);
    tma_fill_sw128<BN, HD>(Vs, &vmap, 0, hkv, b, bars + 3);
  }

  // 本 wg 的 Q 行 = wg*64 + [0,64)。每线程两行（r_lo / r_hi）。
  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < S) { const size_t idx = ((size_t)(b * S + qi_lo)) * H + h; lse_lo = lse[idx]; del_lo = delta[idx]; }
  if (qi_hi < S) { const size_t idx = ((size_t)(b * S + qi_hi)) * H + h; lse_hi = lse[idx]; del_hi = delta[idx]; }

  float dqacc[2][8][4];
#pragma unroll
  for (int nh = 0; nh < 2; ++nh)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] = 0.f;

  char* Qw  = Qs  + wg * (64 / 8) * (HD / 64) * 1024;
  char* dOw = dOs + wg * (64 / 8) * (HD / 64) * 1024;
  char* Pw  = Ps  + wg * (64 / 8) * (BN / 64) * 1024;
  char* dSw = dSs + wg * (64 / 8) * (BN / 64) * 1024;

  const uint32_t Qa = smem_u32(Qs), dOa = smem_u32(dOs);
  const uint32_t Pa = smem_u32(Ps), DSa = smem_u32(dSs);
  const int r0 = wid * 16 + g;

  int ku[2] = {0, 0}, vu = 0, qu = 0;
  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    const int st = nt & 1;
    char* Kt = Ks + st * KTILE;
    if (nt == 0) { mbar_wait(bars + 0, (uint32_t)(qu & 1)); qu++; }
    mbar_wait(bars + 1 + st, (uint32_t)(ku[st] & 1)); ku[st]++;
    mbar_wait(bars + 3, (uint32_t)(vu & 1)); vu++;
    __syncthreads();
    if (nt + 1 < ntiles && tid == 0) {
      mbar_arrive_expect(bars + 1 + (st ^ 1), KTILE);
      tma_fill_sw128<BN, HD>(Ks + (st ^ 1) * KTILE, &kmap, (nt + 1) * BN, hkv, b,
                             bars + 1 + (st ^ 1));
    }

    // ---- (1)(2) 本 wg 的 S=QKᵀ 与 dP=dO·Vᵀ（m64n64），统一 wait0 ----
    float sacc[32], dpacc[32];
    wgmma_mn64_issue(Qw, Kt, HD, sacc);
    wgmma_mn64_issue(dOw, Vs, HD, dpacc);
    wgmma_wait0();
    float pval[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
        const int jg = j0 + j * 8 + c2 + (qq & 1);
        const float lv = (qq >= 2) ? lse_hi : lse_lo;
        float p = 0.f;
        if (qi < S && jg < S && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      pds_store_sw128(Pw, j, r0, lane, c2, BN,
                      __float2bfloat16(pval[j][0]), __float2bfloat16(pval[j][1]),
                      __float2bfloat16(pval[j][2]), __float2bfloat16(pval[j][3]));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN,
                      __float2bfloat16(d0), __float2bfloat16(d1),
                      __float2bfloat16(d2), __float2bfloat16(d3));
    }
    // 两个 wg 的 P/dS 都写好；同时 GEMM2 已读完 V[nt]，可覆盖 V。
    __syncthreads();
    if (nt + 1 < ntiles && tid == 0) {
      mbar_arrive_expect(bars + 3, KTILE);
      tma_fill_sw128<BN, HD>(Vs, &vmap, (nt + 1) * BN, hkv, b, bars + 3);
    }

    if constexpr (SPLIT) {
      // ---- O17-2：GEMM3(dV)→wg0、GEMM4(dK)→wg1，都对全 BM=128 归约（每个 KV 元素只 red 一次）----
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float accv[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) accv[i] = 0.f;
          const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
          wgmma_commit();
          wgmma_wait0();
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
        }
      } else {
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float acck[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) acck[i] = 0.f;
          const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
          wgmma_commit();
          wgmma_wait0();
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
        }
      }
    } else {
      if (wg == 0) {
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float accv[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) accv[i] = 0.f;
          const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
          wgmma_commit();
          wgmma_wait0();
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
        }
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float acck[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) acck[i] = 0.f;
          const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_bf16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
          wgmma_commit();
          wgmma_wait0();
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
        }
      }
    }

    // ---- (5) dQ += scale·dS·K：每个 wg 算自己 64 行（A=本 wg 的 dS 行块）----
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      float accq[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) accq[i] = 0.f;
      wgmma_fence();
      const uint32_t Kt_n = smem_u32(Kt) + (uint32_t)(nh * 1024);
      const uint32_t dSw_a = smem_u32(dSw);
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_bf16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit();
      wgmma_wait0();
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
        const int qi = m0 + wg * 64 + rr;
        const int c = nh * 64 + j * 8 + c2;
        if (qi < S) {
          if (dq_h)
            *reinterpret_cast<__nv_bfloat162*>(dq_h + (((size_t)(b * S + qi)) * H + h) * HD + c) =
                __floats2bfloat162_rn(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          else {
            float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          }
        }
      }
}
#endif  // FA_WGMMA

// =============================================================================
// 2) main kernel（张量核）：1colblock 反向，5 个 GEMM 全 mma.m16n8k16
// =============================================================================
// smem 布局（bf16，除注明外）：
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
//     `Ps/dSs[BM][BN]` 读（`fa_bwd_bf16_atrans_smoke.cu` 验证逐位一致）。
//   ⇒ smem 降到 71.2KB，回到 **3 CTA/SM**，且少写两份转置副本（降 L1/TEX 压力）。
template <int HD, int BM, int BN, int PIPE, bool R4 = false, bool PREL = true>
__global__ void __launch_bounds__(THREADS, (BN > 32) ? 2 : 3)
fa_bwd_bf16_mma_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                       const bf16* __restrict__ v, const bf16* __restrict__ do_,
                       const float* __restrict__ delta, const float* __restrict__ lse,
                       float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                       float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                       int causal, int sched) {
  // O5c：head_dim 从「只 128」扩到 128/512（MLA）。HD>128 时 GEMM1/2（S/dP）的归约维是 HD，
  // 只是 k-loop 变长；GEMM3/4/5 的输出 N 维是 HD，需加一层 **N-tile 循环**（每遍 NTW=128 列），
  // 否则 `GNV=HD/2` 会让累加器/寄存器爆炸。HD=128 时 NDT=1，路径与 O5b/O6/O6b/O6c 逐字等价。
  static_assert(HD % 128 == 0, "head_dim 需为 128 的整数倍");
  static_assert(BM % 32 == 0 && BN % 16 == 0, "BM/BN 需为 2×2 warp 网格的整数倍");
  constexpr int LD  = HD + 8;    // Q/K/V/dO 行距（bf16）
  constexpr int LDP = BM + 8;    // PsT/dSsT 行距（bf16，PIPE!=2）
  constexpr int LDS = BN + 8;    // Ps/dSs 行距（bf16，[BM][BN] 布局）
  constexpr int KVL = BN * LD;                     // 单个 K 或 V 缓冲（bf16）
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
  bf16* Qs   = reinterpret_cast<bf16*>(smem);
  bf16* dOs  = Qs + BM * LD;
  bf16* Ks   = dOs + BM * LD;
  bf16* Vs   = Ks + KSB;
  bf16* Ps   = Vs + VSB;            // PIPE==2：[BM][LDS]；否则 [BN][LDP]（=PsT）
  bf16* dSs  = Ps + PSZ;            // 始终 [BM][LDS]
  bf16* dSsT = dSs + BM * LDS;      // PIPE!=2 的 dS 转置副本（PIPE==2 不分配/不用）

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
  // O10：PIPE>=1 用 16B cp.async 异步发 Q/dO（与 K/V 一起在循环首 wait_group 等待）；
  // PIPE==0 保持同步标量读。
  if constexpr (PIPE >= 1) {
    qdo_issue_async<HD, BM>(q, do_, m0, S, H, h, b, tid, Qs, dOs, LD);
  } else {
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
    float pval[MTM1][MTN1][4];
    {
      float acc[MTM1][MTN1][4];
#pragma unroll
      for (int i = 0; i < MTM1; ++i)
#pragma unroll
        for (int j = 0; j < MTN1; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block_bf16<GM1, GN1, HD, false>(Qs, LD, Kt, LD, acc, wr, wc, lane);
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
              Ps[r * LDS + c] = __float2bfloat16(p);   // [BM][BN]，GEMM3 用 trans 读
            else
              Ps[c * LDP + r] = __float2bfloat16(p);   // [BN][BM]，转置副本（PIPE!=2）
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
      mma_block_bf16<GM1, GN1, HD, false>(dOs, LD, Vt, LD, acc, wr, wc, lane);
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
            dSs[r * LDS + c] = __float2bfloat16(ds);   // GEMM5 A（普通）
            if constexpr (PIPE != 2)
              dSsT[c * LDP + r] = __float2bfloat16(ds);  // GEMM4 A 转置副本（PIPE==2 用 trans 免掉）
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
    //      路径与 O5b/O6/O6b/O6c 逐字等价。----
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
          mma_block_bf16<GMV, GNV, BM, true, true>(Ps, LDS, dOs + hd0, LD, acc, wr, wc, lane);
        else
          mma_block_bf16<GMV, GNV, BM, true>(Ps, LDP, dOs + hd0, LD, acc, wr, wc, lane);
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
          mma_block_bf16<GMV, GNV, BM, true, true>(dSs, LDS, Qs + hd0, LD, acc, wr, wc, lane);
        else
          mma_block_bf16<GMV, GNV, BM, true>(dSsT, LDP, Qs + hd0, LD, acc, wr, wc, lane);
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
        mma_block_bf16<GMQ, GNQ, BN, true>(dSs, LDS, Kt + hd0, LD, acc, wr, wc, lane);
#pragma unroll
        for (int i = 0; i < MTMQ; ++i)
#pragma unroll
          for (int j = 0; j < MTNQ; ++j) {
            if constexpr (NDT == 1) {
              // 每个 Q 块由唯一 CTA 独占：寄存器累加，nt 结束后统一写回（O5b 原路）。
#pragma unroll
              for (int q = 0; q < 4; ++q) dqacc[i][j][q] += acc[i][j][q] * scale;
            } else {
              // HD>128：dQ 的 HD 列放不进寄存器 → 直接全局累加（每 (qi,列) 由唯一线程拥有）。
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
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[i][j][q], dqacc[i][j][q + 1]);
          }
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
  // O13：从逐元素「LDG.32 + STG.16」改成 **float4 读 + bf162 写**（4 元素/次），减少访存指令与
  // 事务数；尾部不足 4 的元素走标量兜底。d*_acc 为 cudaMalloc 基址（256B 对齐），故 float4 安全。
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t t0 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t n_q4 = n_q / 4;
  for (size_t i = t0; i < n_q4; i += stride) {
    float4 v = reinterpret_cast<const float4*>(dq_acc)[i];
    bf16* o = dq + i * 4;
    *reinterpret_cast<__nv_bfloat162*>(o) = __floats2bfloat162_rn(v.x, v.y);
    *reinterpret_cast<__nv_bfloat162*>(o + 2) = __floats2bfloat162_rn(v.z, v.w);
  }
  for (size_t i = n_q4 * 4 + t0; i < n_q; i += stride) dq[i] = __float2bfloat16(dq_acc[i]);
  const size_t n_kv4 = n_kv / 4;
  for (size_t i = t0; i < n_kv4; i += stride) {
    float4 a = reinterpret_cast<const float4*>(dk_acc)[i];
    float4 b = reinterpret_cast<const float4*>(dv_acc)[i];
    bf16* ok = dk + i * 4;
    bf16* ov = dv + i * 4;
    *reinterpret_cast<__nv_bfloat162*>(ok) = __floats2bfloat162_rn(a.x, a.y);
    *reinterpret_cast<__nv_bfloat162*>(ok + 2) = __floats2bfloat162_rn(a.z, a.w);
    *reinterpret_cast<__nv_bfloat162*>(ov) = __floats2bfloat162_rn(b.x, b.y);
    *reinterpret_cast<__nv_bfloat162*>(ov + 2) = __floats2bfloat162_rn(b.z, b.w);
  }
  for (size_t i = n_kv4 * 4 + t0; i < n_kv; i += stride) {
    dk[i] = __float2bfloat16(dk_acc[i]);
    dv[i] = __float2bfloat16(dv_acc[i]);
  }
}


#endif  // FA_BWD_FP16_MMA_KERNELS_CUH_
