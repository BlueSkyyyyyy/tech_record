// =============================================================================
// fa_bwd_fp16_mma_onefile.cu —— FlashAttention 反向（fp16）**张量核单文件版**（O5 / O5c）
// =============================================================================
// 背景：P1 的 `fa_bwd_fp16_onefile.cu` 是**正确性优先的标量 golden**（CUDA-core FFMA），
// 实测 main 只有 ~1 TFLOPS（FA3 ~850、TE ~618）。O5（最大杠杆）把它换成张量核：5 个 GEMM
// 全部用 `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` + `ldmatrix`，数据流保持 FA2
// 的 1colblock（recompute P、D 预处理、dQ/dK/dV 归约）。
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
//
// ----- O5c：head_dim 128 → 128/512（MLA）-----
//   * GEMM1/2（S/dP）的归约维是 HD ⇒ 只是 k-loop 从 8 步变 32 步；
//   * GEMM3/4/5 的输出 N 维是 HD ⇒ 加一层 **N-tile 循环**（每遍 NTW=WN*64=128 列），
//     否则 `GNV=HD/2` 会让累加器/寄存器爆炸；B（dO/Q/K 的 [K][HD] 转置布局）列连续，
//     基址 + hd0 即选中本遍列，写回列号也 + hd0；
//   * dQ 的 HD 列放不进寄存器 ⇒ HD>128 时直接全局累加（每个 (qi, 列) 由**唯一线程**拥有：
//     唯一 CTA + 唯一 warp + 唯一 N-tile，故非原子 RMW 无竞争；dq_acc 由 host memset(0)）；
//   * **HD=128 时 NDT=1、hd0=0，与 O5/O6/O6b/O6c/O7c 逐字等价，数值逐位不变**。
//
// 用法：run.sh src/fp16/fa_bwd_fp16_mma_onefile.cu [--dir=...] [--full|--causal] [--iters=N]
// =============================================================================

#include <cuda_runtime.h>
#include <cuda.h>
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
// O16：等「未完成的 wgmma group 数 ≤ N」。N=1/2 时允许后面已 issue 的 wgmma 继续在
// 张量核上飞，从而把前面 group 的 epilogue（exp/red/量化）与它们重叠 —— 只改等待时机，
// 不改任何累加次序 ⇒ 数值逐位不变。
template <int N>
__device__ __forceinline__ void wgmma_wait_group() {
#if FA_HAS_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
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
// O18：wgmma m64n128k16（N=128，每线程 64 个 fp32 累加器）。累加器布局与 m64n64 同构：
// 行 = wid*16 + lane/4 (+8)，列 = j*8 + (lane&3)*2 (+1)，j=0..15（`d[j*4+qq]`）。
// 冒烟 `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位验证（max_abs=0）。
template <int TA, int TB>
__device__ __forceinline__ void wgmma_m64n128k16_t(float (&d)[64], uint64_t da,
                                                   uint64_t db) {
#if FA_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
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

// =============================================================================
// O25：cluster 分布式归约（Hopper thread block cluster，只在 CL>1 的 wgmma2 路径用）
// -----------------------------------------------------------------------------
// 目标（ROADMAP「下一步」①，O7b 的直接延续）：fp16/bf16 main 的墙是 dK/dV 的跨 CTA
// `atomicAdd`（`red` 占 L2 扇区 ~72%）。O7b 的「partial 覆盖写 + 二次归约 kernel」消掉了
// 原子，但二次归约是一整趟 DRAM 扫描（90% 带宽 bound）⇒ 净负。本方案把「不同 mblk 的
// CTA 对同一 KV 行的偏和」**在 SM 间 smem 内合并**：
//   * cluster 沿 bx（`__cluster_ctarank` = blockIdx.x & 1，配对相邻两个 mblk）；
//   * 每个 CTA 仍算自己 mblk 的 dK/dV 偏和 [BN][HD]；
//   * 所有 rank 用 `red.shared::cluster.add.f32`（mapa 到 leader 的 smem 累加器）把偏和
//     推进 leader；leader 每 tile 只发**一次**全局 `red_add2` ⇒ 全局 red 字节减半，
//     且不落全局 partial 缓冲、不做二次归约。
// 见 `fa_bwd_fp16_cluster_reduce_smoke.cu`（机制逐位 PASS）。
// =============================================================================
__device__ __forceinline__ unsigned fa_cluster_rank() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  unsigned r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
#else
  return 0u;
#endif
}
__device__ __forceinline__ void fa_cluster_sync() {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("barrier.cluster.arrive.aligned;\n" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ unsigned fa_map_shared(unsigned local_addr, unsigned rank) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  unsigned r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n"
               : "=r"(r)
               : "r"(local_addr), "r"(rank));
  return r;
#else
  (void)rank;
  return local_addr;
#endif
}
__device__ __forceinline__ void fa_red_cluster_add(unsigned addr, float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("red.shared::cluster.add.f32 [%0], %1;\n" ::"r"(addr), "f"(v)
               : "memory");
#else
  (void)addr;
  (void)v;
#endif
}

// O7c：dK/dV 归约再把 float2 提升到 float4。mma.m16n8 里一个 quad（lane&3=0..3）的
// `c2=(lane&3)*2` 分别是 0/2/4/6，即同 row 的连续 8 列；把 quad 的 float2 用 `shfl_down 1`
// 拼成两个 float4（列 0-3 由 lane0 写、列 4-7 由 lane2 写），`red.global.add.v4.f32` 的
// 事务数相对 float2 再减半。调用者须保证整个 warp 参与 shfl（在 `if (jg<S)` 之外算好），
// 且仅 (lane&1)==0 的 lane 执行 st；偶 lane 的地址天然 16B 对齐（c2∈{0,4}）。
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
}

// O7b：确定性 dK/dV 归约（可选，`DET`）。跨 CTA 的 `atomicAdd` 换成「每个 (Q-block, KV 行)
//   的贡献写进独立 partial 缓冲」——partial 下标含 `mblk`，每个元素**只被本 CTA 写一次**
//   （非原子覆盖写），因此无竞争、无浮点加法的乱序；再由 `dkv_reduce_kernel` 按 `mblk`
//   升序求和 ⇒ 结果与执行/调度顺序无关（可复现）。代价：partial 多一趟写 + 一趟读，
//   全局字节数上升（见 docs/01 §15）。`DET=false` 仍走 O4c 的 float2 `atomicAdd`。
__device__ __forceinline__ void dkv_det_store(float* part, float a, float b) {
  *reinterpret_cast<float2*>(part) = make_float2(a, b);
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
template <int HD, int BN, bool DOK = true, bool DOV = true, int NTH = THREADS>
__device__ __forceinline__ void kv_issue_async(const __half* __restrict__ k,
                                               const __half* __restrict__ v, int j0, int S,
                                               int Hkv, int hkv, int b, int tid, __half* Kd,
                                               __half* Vd, int LD, int qbase = -1) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 half) 数
  constexpr int NU  = BN * HDV;  // 总 unit 数
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += NTH) {
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

// O10：把一整块 Q/dO（BM 行 × HD 列，half）异步发进 smem。
// 与 `kv_issue_async` 同构，但用于 Q/dO（每个 CTA 只在 prologue 载入一次）：
//   * 用 16B `cp.async.cg` 向量化，取代原来的逐元素 `LDG.U16 + STS.U16`（ncu 报 Q/dO 的
//     标量全局读未被利用满 32B/sector，且 prologue 的全局延迟串在 K/V 之后）。
//   * 行越界（qi>=S）用普通 smem 写 0（与 K/V 一样，由后续 barrier 保证可见）。
// 数值与标量路径**逐位相同**（搬的是同样的 half）。PIPE>=1 时 Q/dO 与 K/V 各提交一个
// commit_group，循环首的 `cp.async.wait_group 0` 一并等待，从而让 Q/dO 的全局延迟与 K/V 重叠。
template <int HD, int BM, int NTH = THREADS>
__device__ __forceinline__ void qdo_issue_async(const __half* __restrict__ q,
                                                const __half* __restrict__ do_, int m0, int S,
                                                int H, int h, int b, int tid, __half* Qd,
                                                __half* dOd, int LD, int qbase = -1) {
  constexpr int HDV = HD / 8;    // 每行 uint4(8 half) 数
  constexpr int NU  = BM * HDV;  // 总 unit 数
  const int tk = (qbase >= 0) ? qbase : b * S;   // VARLEN：token 基址
#pragma unroll
  for (int u = tid; u < NU; u += NTH) {
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
               float* __restrict__ lse, int S, int H, int Hkv, float scale, int causal,
               const int* __restrict__ cu_seqlens = nullptr) {
  constexpr int LD = HD + 8;
  extern __shared__ __align__(16) char smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);
  __half* Ks = Qs + LBM * LD;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * LBM;
  // VARLEN：cu_seqlens 给出本序列在 packed [T,H,D] 的 token 基址与长度；nullptr 退化为
  //   定长 b*S/S。非 causal 的 varlen 走本 kernel（工作均衡，无需镜像配对）。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  if (m0 >= len) return;  // VARLEN：超出本序列长度的 m 块直接退出

  for (int i = tid; i < LBM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    Qs[r * LD + d] =
        (qi < len) ? q[(((size_t)(qbase + qi)) * H + h) * HD + d] : __float2half(0.f);
  }
  __syncthreads();

  const int ncols = causal ? min(len, m0 + LBM) : len;
  const int ntiles = (ncols + LBN - 1) / LBN;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * LBN;
    for (int i = tid; i < LBN * HD; i += THREADS) {
      int r = i / HD, d = i % HD;
      int jg = j0 + r;
      Ks[r * LD + d] =
          (jg < len) ? k[(((size_t)(qbase + jg)) * Hkv + hkv) * HD + d] : __float2half(0.f);
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
        if (qi < len && jg < len && !(causal && jg > qi)) sv = acc[0][j][q] * scale;
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
      if (qi < len) lse[((size_t)(qbase + qi)) * H + h] = m + flog(l);
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
                   float* __restrict__ lse, int S, int H, int Hkv, float scale,
                   const int* __restrict__ cu_seqlens = nullptr,
                   float* __restrict__ lse_part = nullptr, int ksplit = 1) {
  constexpr int LD  = HD + 8;
  constexpr int KVL = LBN * LD;
  constexpr int HDV = HD / 8;   // 每行 uint4(8 half) 数
  extern __shared__ __align__(16) char smem[];
  __half* Qs = reinterpret_cast<__half*>(smem);
  __half* Ks = Qs + LBM * LD;   // PIPE=1：2 × LBN × LD；PIPE=0：1 × LBN × LD

  // O39：K 维 split（对齐 fp8 §41 与 fp16 TMA LSE 的 O38）。`ksplit>1` 时 grid.z = B*ksplit，
  //   每个 (pair,ks) CTA 只扫本 m 块 K 范围的第 ks 个连续 tile 切片，部分 (m,l) 写 `lse_part`，
  //   由 `lse_split_merge_kernel` 汇总。`ksplit==1` 时 b=blockIdx.z、ksp=0，逐位退回 O8b。
  const int pair = blockIdx.x, h = blockIdx.y;
  const int b = blockIdx.z / ksplit, ksp = blockIdx.z % ksplit;
  // VARLEN（第 81 轮）：cu_seqlens 给出本序列在 packed [T,H,D] 的 token 基址与长度；
  // nullptr 逐式退化为定长（qbase=b*S、len=S），定长路径逐位不变。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk  = (len + LBM - 1) / LBM;
  if (pair >= (nblk + 1) / 2) return;   // 短序列多余的对 CTA 直接退出
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
      if (jg < len) {
        const size_t off = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c8 * 8;
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
      if (qi < len) {
        const size_t off = (((size_t)(qbase + qi)) * H + h) * HD + c8 * 8;
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

    const int ncols = min(len, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    // O39：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;

    // prologue：PIPE=1 时发 tile0 进 stage0（Q 与 K 写不同 smem，由循环首 wait+sync 保证可见）
    if constexpr (PIPE) {
      if (nuse > 0) issue_k(Ks, nt0 * LBN);
    }

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int j0 = nt * LBN;
      __half* Kt = Ks + (PIPE ? (rnt & 1) * KVL : 0);
      if constexpr (PIPE) {
        // 等本 tile 落地；此 barrier 同时证明「上一 tile 的 mma 已读完其 stage」，故可复用。
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (rnt + 1 < nuse) issue_k(Ks + ((rnt + 1) & 1) * KVL, j0 + LBN);
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
          if (qi < len && jg < len && jg <= qi) sv = acc[0][j][q] * scale;
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
        if (qi < len) {
          if (ksplit == 1) {
            lse[((size_t)(qbase + qi)) * H + h] = m + flog(l);
          } else {
            size_t row = ((size_t)(qbase + qi)) * H + h;
            lse_part[(row * ksplit + ksp) * 2 + 0] = m;
            lse_part[(row * ksplit + ksp) * 2 + 1] = l;
          }
        }
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
                         float* __restrict__ lse, int S, int H, int Hkv, float scale,
                         const int* __restrict__ cu_seqlens = nullptr,
                         float* __restrict__ lse_part = nullptr, int ksplit = 1) {
  static_assert(HD == 128, "wgmma LSE 目前只做 HD=128");
  constexpr int HDV = HD / 8;
  constexpr int TILE = (LBM / 8) * (HD / 64) * 1024;  // 单个 SW128 tile 字节数（HD=128→16KB）
  extern __shared__ char smem_raw[];
  // SW128 描述符 base_offset=0 要求 tile 1024B 对齐 → 手动对齐动态 smem 基址。
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;  // PIPE=1：2*TILE；PIPE=0：TILE

  // O40：K 维 split（对齐 O38/O39/O40-fp8）。`ksplit>1` 时每个 (pair,mblk) 只扫本 m 块 K
  //   范围的第 `ksp` 个连续 tile 切片，部分 (m,l) 写 `lse_part`，由 `lse_split_merge_kernel` 汇总。
  const int pair = blockIdx.x, h = blockIdx.y;
  const int b = blockIdx.z / ksplit, ksp = blockIdx.z % ksplit;
  const int hkv = h / (H / Hkv);
  // VARLEN：cu_seqlens 给本序列 token 基址与长度；nullptr 退化为定长 b*S/S。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk = (len + LBM - 1) / LBM;
  if (pair >= (nblk + 1) / 2) return;  // VARLEN：短序列多余配对 CTA 退出
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // K tile（j0 起 LBN 行）用 cp.async 发进 SW128 tile（PIPE=1）或标量写（PIPE=0）。
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
    // O40：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;
    if constexpr (PIPE) {
      if (nuse > 0) issue_k(Ks, nt0 * LBN);
    }

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int j0 = nt * LBN;
      char* Kt = Ks + (PIPE ? (rnt & 1) * TILE : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (rnt + 1 < nuse) issue_k(Ks + ((rnt + 1) & 1) * TILE, j0 + LBN);
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
        if (qi < len) {
          if (ksplit == 1) {
            lse[((size_t)(qbase + qi)) * H + h] = m + flog(l);
          } else {
            size_t row = ((size_t)(qbase + qi)) * H + h;
            lse_part[(row * ksplit + ksp) * 2 + 0] = m;
            lse_part[(row * ksplit + ksp) * 2 + 1] = l;
          }
        }
      }
    }
    __syncthreads();
  }
}

// =============================================================================
// O30：TMA 版 LSE（N5 计划「TMA 化 operand」的第一步落地）
// =============================================================================
// 动机：O15a（第 51 轮）已用 `fa_bwd_fp16_tma_smoke.cu` 证明 `cuTensorMapEncodeTiled`
//   (SWIZZLE_128B) 写出的 smem 布局与 kernel 的 `sw128_off` **逐字节相同**，并发现
//   **HD=128 的 K-major tile 必须拆成 2 个 K=64 chunk**（TMA box 内维 128B = 64 个 fp16），
//   wgmma 侧用两个 `SBO=1024` 描述符读。但那条通路一直只停在冒烟，没落进真 kernel。
//   本 kernel 把 LSE 的 Q/K 载入从「逐 16B `cp.async` + 地址运算」换成 **4D TMA**
//   （坐标 {k0, row, head, batch}），省掉 load 指令/地址运算（1 条 bulk 指令搬一个 8KB
//   chunk），并复用同一份 SW128 tile 供 `wgmma.m64n64k16` 直读。
//   * 数学与 `lse_mma_kernel_bal_wgmma` **完全一致**（镜像配对、online-softmax、4-lane
//     `shfl` 归约），只换搬运方式 ⇒ 数值应在 fp32 求和次序内一致。
//   * 仅 HD=128、causal（由 host 控制）。TMA asm 需 sm_90a，用 `FA_HAS_WGMMA` 包裹，
//     `sm_90` 构建时退化为空实现（且 host 不会 launch 它）。
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
// 搬进 dst（SW128 K-major，box 内维 64 fp16 = 128B）。
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
// 消费两个 K=64 chunk 的 QKᵀ：每个 chunk 用 `SBO=1024` 的描述符（见 O15a 冒烟）。
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
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(q0, s), 1024),
                        make_desc_sw128(sw128_k16_addr(k0, s), 1024));
  }
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    wgmma_m64n64k16_f16(d, make_desc_sw128(sw128_k16_addr(q1, s), 1024),
                        make_desc_sw128(sw128_k16_addr(k1, s), 1024));
  }
  wgmma_commit();
  wgmma_wait0();
}

// O38（fp16 版，对齐 fp8 §41）：把 K 维 split 的 LSE 部分结果 `(m_ks, l_ks)` 沿 `ks`
//   二次归约成最终 LSE。`part` 布局 `[row][ks] -> (m,l)`（每行 `2*ksplit` 个 fp32），
//   输出 `lse[row]=m+log(l)`。online-softmax 合并（max 取大、sum 按 exp 重标定），
//   `-inf/0` 安全（空 split 得 -inf/0）。数学上与「单 CTA 顺序扫全部 K tile」完全等价，
//   只差 fp32 求和次序。
__global__ void lse_split_merge_kernel(const float* __restrict__ part,
                                       float* __restrict__ lse, long long nrows, int ksplit) {
  long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= nrows) return;
  const float* p = part + row * (long long)ksplit * 2;
  float m = -INFINITY, l = 0.f;
#pragma unroll 1
  for (int k = 0; k < ksplit; ++k) {
    float mk = p[2 * k], lk = p[2 * k + 1];
    float mn = fmaxf(m, mk);
    float ca = (m == -INFINITY) ? 0.f : l * fexp(m - mn);
    float cb = (mk == -INFINITY) ? 0.f : lk * fexp(mk - mn);
    l = ca + cb;
    m = mn;
  }
  lse[row] = m + flog(l);
}

template <int HD, int PIPE = 1>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_tma(const __grid_constant__ CUtensorMap qmap,
                       const __grid_constant__ CUtensorMap kmap,
                       float* __restrict__ lse, float* __restrict__ lse_part,
                       int S, int H, int Hkv, float scale, int ksplit) {
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
  // O38：K 维 split。grid = (pairs, H, B*ksplit)；每个 (pair,ks) CTA 只扫本 m 块 K 范围的
  //   第 ks 个连续切片（按 tile 粒度切分），把部分 (m,l) 写到 `lse_part`，由 merge kernel 汇总。
  //   ksplit==1 时切片即整段、直接写 `lse`（逐位退化为 O30 原路径）。
  const int pair = blockIdx.x, h = blockIdx.y;
  const int b = (int)blockIdx.z / ksplit, ksp = (int)blockIdx.z % ksplit;
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
    // O38：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;
    issue_q(m0);
    if (nuse > 0) issue_k(0, nt0 * LBN);
    mbar_wait(qbar, (uint32_t)(quse & 1)); quse++;

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int st = PIPE ? (rnt & 1) : 0;
      mbar_wait(kbar + st, (uint32_t)(kuse[st] & 1)); kuse[st]++;
      __syncthreads();
      const int j0 = nt * LBN;
      char* Kt = Ks + st * TILE;
      if (PIPE) {
        if (rnt + 1 < nuse) issue_k(st ^ 1, j0 + LBN);
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
      if (!PIPE && rnt + 1 < nuse) issue_k(0, j0 + LBN);
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
        if (qi < S) {
          if (ksplit == 1) {
            lse[((size_t)(b * S + qi)) * H + h] = m + flog(l);
          } else {
            size_t row = ((size_t)(b * S + qi)) * H + h;
            lse_part[(row * ksplit + ksp) * 2 + 0] = m;
            lse_part[(row * ksplit + ksp) * 2 + 1] = l;
          }
        }
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

// O24：`delta_kernel` 的 warp-per-row 向量化版（与 fp8 O14 的 `quantize_row_warp_kernel` 同思路）。
// 旧版每个 (s,h,b) 行一个 128 线程 CTA + `__shared__` 归约 + log2(THREADS) 次 `__syncthreads`，
// 对 HD=128 只有 128 个乘加，block/同步开销远大于计算（ncu：S=4096 delta 42.8µs、occ 71.9%）。
// 新版**每 warp 一行**：lane 沿 HD 以 `__half2`（4B）coalesced 读（每步 warp 读 32×4=128B），
// `__shfl_xor_sync` 树归约，**无 smem / 无 barrier**。grid-stride 覆盖任意行数。
template <int HD>
__global__ void delta_warp_kernel(const __half* __restrict__ o,
                                  const __half* __restrict__ do_, float* __restrict__ delta,
                                  int rows) {
  static_assert(HD % 2 == 0, "delta_warp 需要 HD 为偶数（按 half2 读）");
  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5;
  const int gwarp0 = blockIdx.x * wpb + (threadIdx.x >> 5);
  const int nwarp = gridDim.x * wpb;
  for (int row = gwarp0; row < rows; row += nwarp) {
    const __half2* o2 = reinterpret_cast<const __half2*>(o + (size_t)row * HD);
    const __half2* d2 = reinterpret_cast<const __half2*>(do_ + (size_t)row * HD);
    float acc = 0.f;
#pragma unroll
    for (int k = lane; k < HD / 2; k += 32) {
      const float2 a = __half22float2(o2[k]);
      const float2 b = __half22float2(d2[k]);
      acc += a.x * b.x + a.y * b.y;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) delta[row] = acc;
  }
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

// O18：m64n128 的 issue-only 版（A/B 均 K-major，GEMM1/2 用；N=128 = BN）。
__device__ __forceinline__ void wgmma_mn128_issue(const char* Asw, const char* Bsw, int Kd,
                                                  float (&d)[64]) {
#pragma unroll
  for (int i = 0; i < 64; ++i) d[i] = 0.f;
  wgmma_fence();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((Kd / 64) * 1024);
#pragma unroll
  for (int s = 0; s < Kd / 16; ++s) {
    wgmma_m64n128k16_t<0, 0>(d, make_desc_sw128(sw128_k16_addr(aa, s), sbo),
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

// O17b：把 `pds_store_sw128` 写下的本线程 4 个 half 读回（同一线程同一地址，无需 sync）。
__device__ __forceinline__ void pds_load_sw128(const char* tile, int j, int r0, int c2, int W,
                                               float& p0, float& p1, float& p2, float& p3) {
  const int c = j * 8 + c2;
  const uint32_t a = *reinterpret_cast<const uint32_t*>(tile + sw128_off(r0, c, W));
  const uint32_t b = *reinterpret_cast<const uint32_t*>(tile + sw128_off(r0 + 8, c, W));
  p0 = __half2float(__ushort_as_half((unsigned short)(a & 0xffff)));
  p1 = __half2float(__ushort_as_half((unsigned short)(a >> 16)));
  p2 = __half2float(__ushort_as_half((unsigned short)(b & 0xffff)));
  p3 = __half2float(__ushort_as_half((unsigned short)(b >> 16)));
}

// 把 K/V（[BN][HD] half）用 16B `cp.async` 发进 SW128 tile（DOK/DOV 拆不同 commit_group）。
// O17：NT 为参与搬运的线程数（默认 THREADS；2-warpgroup 版传 256）。
template <int HD, int BN, bool DOK, bool DOV, int NT = THREADS>
__device__ __forceinline__ void kv_issue_async_sw(const __half* __restrict__ k,
                                                  const __half* __restrict__ v, int j0, int S,
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

// 把 Q/dO（[BM][HD] half）用 16B `cp.async` 发进 SW128 tile。
// O17：NT 为参与搬运的线程数（默认 THREADS；2-warpgroup 版传 256）。
template <int HD, int BM, int NT = THREADS>
__device__ __forceinline__ void qdo_issue_async_sw(const __half* __restrict__ q,
                                                   const __half* __restrict__ do_, int m0,
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

template <int HD, bool OW = false>
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
    // O16：`wait_group<1>` 只等 S=QKᵀ 那个 group 完成，dP=dO·Vᵀ 继续在张量核上飞；
    // P 的 exp/写 smem 与 dP 重叠，随后再 wait0 等 dP（数值与顺序无关，逐位不变）。
    if constexpr (OW) wgmma_wait_group<1>(); else wgmma_wait0();
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
    if constexpr (OW) wgmma_wait0();  // 现在才需要 dP 累加器
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
      // O16：三条 GEMM 已 issue，按「dV→dK→dQ」顺序用 `wait_group<2>/<1>/wait0` 逐个收，
      // 让 dV 的 red / dK 的 red 与前一条仍在飞的 wgmma 重叠（只改等待时机 ⇒ 数值逐位不变）。
      if constexpr (OW) wgmma_wait_group<2>(); else wgmma_wait0();
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
      if constexpr (OW) wgmma_wait_group<1>();  // dK 累加器就绪
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
      wgmma_wait0();  // dQ 累加器就绪
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
// O17：2 warpgroup（BM=128）wgmma 主 kernel —— **跨 warpgroup 归约**，把 dK/dV 的
//      跨 CTA atomic 字节数砍半。
// =============================================================================
// 动机（O15a ncu）：S=4096 主 kernel 的 L2 扇区里 `red`（dK/dV 的跨 CTA `atomicAdd`）
// 占 **73.1%**、DRAM 仅 4.2% ⇒ main 是 **L2 原子字节数 bound**。一个 KV 元素被 `nblk`
// 个 CTA 贡献一次偏和；把每个 CTA 覆盖的 Q 行从 BM=64 扩到 **BM=128**，贡献它的 CTA 数
// 从 `nblk` 降到 `nblk/2` ⇒ **red 字节直接砍半**。O16（错开 wait）与 O7c（float4 归约）
// 都已证明「动等待/事务数」无效，唯一杠杆是减少每个元素的贡献 CTA 数。
//
// 结构（HD=128, BM=128, BN=64, 256 线程 = 2 warpgroups）：
//   * 2 个 warpgroup 各持自己 64 行 Q/dO，各自算 GEMM1/2（S、dP）并把 P/dS 按 SW128
//     写进**共享**的 [128][BN] tile（wg 写自己的 64 行）；
//   * **GEMM3/4 只由 wg0 做，且对全 BM=128 归约**：把两个 m64 半（s=0..7，即 128 行）
//     连续喂给**同一个** `wgmma.m64n64k16` 累加器（输出都是同一个 [BN][HD]），得到的
//     就是两半之和 ⇒ **每个 KV 元素一次 red**（前置冒烟 `fa_bwd_fp16_wgmma2_smoke.cu`
//     逐位验证「[128][*] K-major tile + MN-major 描述符 s=0..7 转置读」，max_abs=0）。
//     wg1 同时做自己的 GEMM5（dQ 只依赖本 wg 的行，互不干扰）；
//   * GEMM5（dQ）两个 wg 各算自己 64 行、寄存器累加，无需跨 CTA 原子；
//   * K/V 用 NTH=256 线程 `cp.async`：K 双缓冲、V 单缓冲后段预取（同 O6b/O9b）。
//
// smem（HD=128）：Q 32KB + dO 32KB + K 双缓冲 32KB + V 16KB + P 16KB + dS 16KB ≈ 145KB
// → 1 CTA/SM（256 线程 = 8 warps/SM，与 O9b 的 2 CTA/SM × 4 warps 相同）。数值只改
// 归约次序（atomic 顺序），与 O9b 在 fp16 噪声内一致。
template <int HD, bool SPLIT = true, int CL = 1>
__global__ void __launch_bounds__(256, 1)
fa_bwd_fp16_wgmma2_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                          const __half* __restrict__ v, const __half* __restrict__ do_,
                          const float* __restrict__ delta, const float* __restrict__ lse,
                          float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                          float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                          int causal, __half* __restrict__ dq_h = nullptr,
                          const int* __restrict__ cu_seqlens = nullptr,
                          int ksplit = 1) {
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
  // O25：cluster 版在 leader 的 smem 里维护本 tile 的 dK/dV 合并累加器 [BN][HD] fp32。
  float* dvacc = reinterpret_cast<float*>(dSs + PTILE);
  float* dkacc = reinterpret_cast<float*>(dSs + PTILE) + (CL > 1 ? BN * HD : 0);

  const int bx = blockIdx.x;
  // nblk 仅用于 DET/cluster 路径（varlen 不走），保持定长语义。
  const int nblk = (S + BM - 1) / BM;
  // O43：N 方向 split-K（仅 CL==1 使用）。同一 mblk 的 KV tile [0,ntiles) 被均分给
  //   ksplit 个 CTA；每个 CTA 只扫自己那一段，dQ 改走跨 CTA fp32 `atomicAdd`（ksplit==1
  //   时逐式退化：ksp=0/mblk=bx/nt_begin=0/nt_end=ntiles，数值与原路径逐位相同）。
  //   小 S（grid 不足一个波）时把 grid 从 S/BM×H 抬到 ksplit 倍，填满空 SM。
  const int ksp = (ksplit > 1) ? (bx % ksplit) : 0;
  const int mblk = (ksplit > 1) ? (bx / ksplit) : bx;
  // O25：cluster 沿 bx（clusterDim.x=CL），leader = cluster 内 rank 0（mblk 较小的那个）。
  const int crank = (CL > 1) ? (int)fa_cluster_rank() : 0;
  unsigned leader_dv = 0, leader_dk = 0;
  if constexpr (CL > 1) {
    leader_dv = fa_map_shared(smem_u32(dvacc), 0u);
    leader_dk = fa_map_shared(smem_u32(dkacc), 0u);
  }
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

  const int ncols = causal ? min(len, m0 + BM) : len;
  const int ntiles = (ncols + BN - 1) / BN;
  // O43：本 CTA 负责的 KV tile 切片 [nt_begin, nt_end)。ksplit==1 时即 [0,ntiles)。
  const int nt_begin = (ksplit > 1) ? (int)(((long)ntiles * ksp) / ksplit) : 0;
  const int nt_end = (ksplit > 1) ? (int)(((long)ntiles * (ksp + 1)) / ksplit) : ntiles;
  if (ksplit > 1 && nt_end <= nt_begin) return;   // 该切片无 tile（causal 小 mblk 可能被切空）

  qdo_issue_async_sw<HD, BM, NTH>(q, do_, m0, len, H, h, b, tid, Qs, dOs, qbase);

  // O25：cluster 配对「相邻两个 mblk」——高的那个（rank1）恒多做 BM/BN 个（=2）KV tile。
  //   低 mblk 的 rank0 把循环延到 rank1 的 tile 数；多出的 tile 因 causal mask 使 P=0、
  //   dK/dV 偏和为 0，只贡献 barrier 参与与 0 累加（不改数值）。这样 cluster 内两个 CTA
  //   锁步，per-tile 的 DSM 合并才有确定的同步点。
  const int nt_loop = (nt_end - nt_begin) +
                      ((CL > 1 && causal && crank == 0) ? (BM / BN) : 0);
  if (nt_loop > 0) {
    kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, nt_begin * BN, len, Hkv, hkv, b, tid, Ks,
                                                Vs, qbase);
    kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, nt_begin * BN, len, Hkv, hkv, b, tid, Ks,
                                                Vs, qbase);
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

  // O25：把 dK/dV 的偏和推进 cluster leader（CL==1 时退化为原来的全局 red_add2）。
  auto dkv_red = [&](float* gdst, unsigned lb, int rr, int c, float a, float b) {
    if constexpr (CL > 1) {
      const unsigned off = (unsigned)(rr * HD + c) * 4u;
      fa_red_cluster_add(lb + off, a);
      fa_red_cluster_add(lb + off + 4u, b);
    } else {
      red_add2(gdst, a, b);
    }
  };
  if constexpr (CL > 1) {
    for (int t2 = tid; t2 < BN * HD; t2 += NTH) { dvacc[t2] = 0.f; dkacc[t2] = 0.f; }
    fa_cluster_sync();   // leader 的累加器清零先于任何 rank 的第一次累加
  }

  for (int nt = 0; nt < nt_loop; ++nt) {
    const int j0 = (nt_begin + nt) * BN;   // O43：tile 的全局列偏移
    char* Kt = Ks + (nt & 1) * KTILE;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
    if (nt + 1 < nt_loop)
      kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, (nt_begin + nt + 1) * BN, len, Hkv, hkv,
                                                  b, tid, Ks + ((nt + 1) & 1) * KTILE, Vs, qbase);

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
                      __float2half(pval[j][0]), __float2half(pval[j][1]),
                      __float2half(pval[j][2]), __float2half(pval[j][3]));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN,
                      __float2half(d0), __float2half(d1),
                      __float2half(d2), __float2half(d3));
    }
    // 两个 wg 的 P/dS 都写好；同时 GEMM2 已读完 V[nt]，可覆盖 V。
    __syncthreads();
    if (nt + 1 < nt_loop)
      kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, (nt_begin + nt + 1) * BN, len, Hkv, hkv,
                                                  b, tid, Ks, Vs, qbase);

    if constexpr (SPLIT) {
      // ---- O17-2：把 GEMM3(dV) 交给 wg0、GEMM4(dK) 交给 wg1，两者都仍对全 BM=128 归约
      //      （s=0..7 两个 m64 半进同一 wgmma 累加器）⇒ 每个 KV 元素仍只 `red` 一次，但
      //      张量工作量从「wg0:wg1 = 3:1」变成 1:1（原版 wg0 串行做 dV+dK，wg1 只做 dQ），
      //      且 wg0 原本 4 条串行 red epilogue 链被拆到两个 wg 并发 ⇒ 关键路径缩短。
      //      数值：dV/dK 的 wgmma 归约次序与 O17 完全相同，只是换了个 wg 发射，值逐位相同。----
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
            wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
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
                dkv_red(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c, leader_dv,
                        rr, c, accv[j * 4 + qq], accv[j * 4 + qq + 1]);
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
            wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
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
                dkv_red(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c, leader_dk,
                        rr, c, acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
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
          wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
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
              dkv_red(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c, leader_dv,
                      rr, c, accv[j * 4 + qq], accv[j * 4 + qq + 1]);
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
          wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
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
              dkv_red(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c, leader_dk,
                      rr, c, acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
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
        wgmma_m64n64k16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit();
      wgmma_wait0();
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] += accq[j * 4 + qq] * scale;
    }

    // ---- O25：cluster 合并——本 tile 所有 rank 的 dK/dV 偏和都已在 leader 的 smem 里，
    //      leader 只发一次全局 `red_add2`（≈把跨 CTA red 字节砍半），然后清零复用。----
    if constexpr (CL > 1) {
      fa_cluster_sync();                 // (A) 所有 remote/local add 已落地
      if (crank == 0) {
        for (int t2 = tid; t2 < BN * (HD / 2); t2 += NTH) {
          const int rr2 = t2 / (HD / 2);
          const int cc2 = (t2 % (HD / 2)) * 2;
          const int jgf = j0 + rr2;
          if (jgf < S) {
            const size_t gb = (((size_t)(qbase + jgf)) * Hkv + hkv) * HD + cc2;
            red_add2(dv_acc + gb, dvacc[rr2 * HD + cc2], dvacc[rr2 * HD + cc2 + 1]);
            red_add2(dk_acc + gb, dkacc[rr2 * HD + cc2], dkacc[rr2 * HD + cc2 + 1]);
          }
        }
        for (int t2 = tid; t2 < BN * HD; t2 += NTH) { dvacc[t2] = 0.f; dkacc[t2] = 0.f; }
      }
      fa_cluster_sync();                 // (B) 所有 rank 都看到 leader flush+清零完成
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
          // O43：ksplit>1 时同一 Q 行由多个 CTA 贡献 ⇒ dQ 必须跨 CTA 原子累加（dq_acc）。
          if (ksplit > 1) {
            float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
            red_add2(base, dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          }
          // O24：同 wgmma2b，dQ 唯一拥有 ⇒ 可直接写 fp16，省掉 convert 的 dQ 一趟。
          else if (dq_h)
            *reinterpret_cast<__half2*>(dq_h + (((size_t)(qbase + qi)) * H + h) * HD + c) =
                __floats2half2_rn(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          else {
            float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          }
        }
      }
}

// =============================================================================
// O18：BN=128 版 wgmma2 主 kernel —— tile 数减半，降每-tile 的 barrier / wgmma 固定开销。
// =============================================================================
// 动机（docs §14k.7 item 3）：O17（BM=128,BN=64）后 main 的第一墙是 L2 原子（`red` 占 L2 扇区
// ~72%），且 occupancy 只有 1 CTA/SM（12.5%）⇒ 延迟受限。BN 从 64→128 让 per-CTA 的 tile 数
// 减半，⇒ `__syncthreads` / `cp.async.wait` / wgmma commit-wait 序列都减半；GEMM1/2 用
// `m64n128k16`（一条指令算两倍），发射/依赖链也减半。代价是 smem 148→224KB（仍 1 CTA/SM）。
//
// 结构（HD=128, BM=128, BN=128, 256 线程 = 2 warpgroup）与 O17 完全同构，仅：
//   * GEMM1/2 用 `wgmma_mn128_issue`（m64n128，A/B 均 K-major）；
//   * P/dS tile 变 [128][128]，softmax epilogue 列组 j=0..15；
//   * GEMM3/4 输出 [BN=128][HD=128] ⇒ 2 个 m64 半（mh=0/1），转置描述符基址 +mh*1024
//     （存储列 64 的 SW128 kg=1 atom；smoke `fa_bwd_fp16_wgmma2b_smoke.cu` 逐位 PASS）；
//   * GEMM5 输出 [64][HD=128]（每 wg 自己 64 行）⇒ 一条 m64n128，`dqacc[16][4]`。
// 数值只改跨 CTA `atomicAdd` 次序，与 O17 在 fp16 噪声内一致。
template <int HD, bool SPLIT = true, bool DET = false>
__global__ void __launch_bounds__(256, 1)
fa_bwd_fp16_wgmma2b_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                           const __half* __restrict__ v, const __half* __restrict__ do_,
                           const float* __restrict__ delta, const float* __restrict__ lse,
                           float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                           float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                           int causal, float* __restrict__ dk_part = nullptr,
                           float* __restrict__ dv_part = nullptr, int nblk = 0,
                           __half* __restrict__ dq_h = nullptr) {
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
      pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2half(pval[j][0]),
                      __float2half(pval[j][1]), __float2half(pval[j][2]),
                      __float2half(pval[j][3]));
#pragma unroll
    for (int j = 0; j < NG; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2half(d0), __float2half(d1),
                      __float2half(d2), __float2half(d3));
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
            wgmma_m64n128k16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          // 本 mh 的输出行偏移 = mh*64（红 epilogue 里加）。
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) {
                if constexpr (DET)
                  dkv_det_store(dv_part + (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + c,
                                accv[j * 4 + qq], accv[j * 4 + qq + 1]);
                else
                  red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                           accv[j * 4 + qq], accv[j * 4 + qq + 1]);
              }
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
            wgmma_m64n128k16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) {
                if constexpr (DET)
                  dkv_det_store(dk_part + (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + c,
                                acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
                else
                  red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                           acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
              }
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
            wgmma_m64n128k16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) {
                if constexpr (DET)
                  dkv_det_store(dv_part + (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + c,
                                accv[j * 4 + qq], accv[j * 4 + qq + 1]);
                else
                  red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                           accv[j * 4 + qq], accv[j * 4 + qq + 1]);
              }
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
            wgmma_m64n128k16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
          wgmma_commit(); wgmma_wait0();
          for (int j = 0; j < NG; ++j)
            for (int qq = 0; qq < 4; qq += 2) {
              const int rr = r0 + (qq >= 2 ? 8 : 0);
              const int jg = j0 + mh * 64 + rr;
              const int c = j * 8 + c2;
              if (jg < S) {
                if constexpr (DET)
                  dkv_det_store(dk_part + (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + c,
                                acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
                else
                  red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                           acck[j * 4 + qq] * scale, acck[j * 4 + qq + 1] * scale);
              }
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
        wgmma_m64n128k16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
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
        // O24：dQ 由本 CTA 唯一拥有（无跨 CTA 原子）⇒ 可直接写 fp16 输出，省掉 convert 的 dQ 一趟。
        if (dq_h)
          *reinterpret_cast<__half2*>(dq_h + (((size_t)(b * S + qi)) * H + h) * HD + c) =
              __floats2half2_rn(dqacc[j][qq], dqacc[j][qq + 1]);
        else {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) = make_float2(dqacc[j][qq], dqacc[j][qq + 1]);
        }
      }
    }
}

// =============================================================================
// O33：主 kernel 的 Q/K/V/dO 改用 4D-TMA 载入（把 O30–O32 的「TMA 化 operand」从 LSE
//       推到主 kernel）。
// =============================================================================
// 动机：O30/O31/O32 已把 LSE 的 Q/K 换成 4D-TMA（省 load 指令/地址运算）。主 kernel 的
//   Q/dO（prologue）与 K/V（每个 KV tile）仍用逐 16B `cp.async` + `sw128_off` 地址运算。
//   TMA 一个 box 内维固定 128B（fp16 = 64 元素）——对 HD=128 的 K-major tile，一个
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
fa_bwd_fp16_wgmma2b_tma_kernel(
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __grid_constant__ CUtensorMap dmap,
    const float* __restrict__ delta, const float* __restrict__ lse,
    float* __restrict__ dq_acc, float* __restrict__ dk_acc, float* __restrict__ dv_acc,
    int S, int H, int Hkv, float scale, int causal,
    __half* __restrict__ dq_h = nullptr) {
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
      pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2half(pval[j][0]),
                      __float2half(pval[j][1]), __float2half(pval[j][2]),
                      __float2half(pval[j][3]));
#pragma unroll
    for (int j = 0; j < NG; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2half(d0), __float2half(d1),
                      __float2half(d2), __float2half(d3));
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
            wgmma_m64n128k16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
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
            wgmma_m64n128k16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
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
            wgmma_m64n128k16_t<1, 1>(accv, desc_k16_mn(Pa_m, s, BN), desc_k16_mn(dOa, s, HD));
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
            wgmma_m64n128k16_t<1, 1>(acck, desc_k16_mn(DSa_m, s, BN), desc_k16_mn(Qa, s, HD));
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
        wgmma_m64n128k16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
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
          *reinterpret_cast<__half2*>(dq_h + (((size_t)(b * S + qi)) * H + h) * HD + c) =
              __floats2half2_rn(dqacc[j][qq], dqacc[j][qq + 1]);
        else {
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) = make_float2(dqacc[j][qq], dqacc[j][qq + 1]);
        }
      }
    }
}

// =============================================================================
// O35：BN=64 版 wgmma2 主 kernel 的 Q/K/V/dO 改用逐 atom 4D-TMA
// =============================================================================
// 动机：O33（fp16）/O34（bf16）只把 **BN=128** 的 `wgmma2b` 主 kernel 的 Q/K/V/dO 换成
//   4D-TMA；而 O23 的默认档在 **S<4096 与 GQA/MQA** 走的是 **BN=64 的 `wgmma2`**——这条
//   更常用的路仍用逐 16B `cp.async` + `sw128_off` 地址运算。本 kernel 把 O33 的做法
//   （逐 `[8 行][64 列]` atom 发 TMA：box 内维 128B=64 个 fp16、8 行=一个 1024B atom，
//   原样复现 SW128 交织布局、wgmma 描述符零改动）搬到 BN=64 几何：Q/dO 各 32 atom、
//   K/V 各 16 atom。K 双缓冲两个 barrier、V 单缓冲后段预取（同 O6b/O17/O33）。
//   * 数学与 `fa_bwd_fp16_wgmma2_kernel` **完全一致**（同一 wgmma 次序、同一 epilogue），
//     只换搬运方式 ⇒ 数值应逐位相同（差异仅跨 CTA `atomicAdd` 次序）。作为 `--maintma`
//     的 BN=64 分支，与 cp.async 版做同 binary A/B。
//   * 仅 `-DFA_WGMMA -DFA_TMA -lcuda` 构建、HD=128 时由 host 选用；`sm_90` 构建里
//     mbar/tma asm 退化为空实现、host 不会 launch。
// =============================================================================
template <int HD, bool SPLIT = true>
__global__ void __launch_bounds__(256, 1)
fa_bwd_fp16_wgmma2_tma_kernel(
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __grid_constant__ CUtensorMap dmap,
    const float* __restrict__ delta, const float* __restrict__ lse,
    float* __restrict__ dq_acc, float* __restrict__ dk_acc, float* __restrict__ dv_acc,
    int S, int H, int Hkv, float scale, int causal, __half* __restrict__ dq_h = nullptr,
    const int* __restrict__ cu_seqlens = nullptr) {
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
  // VARLEN（本轮）：cu_seqlens 给出本序列在 packed [T,H,D] 的 token 基址与长度。
  //   `qbase`/`len` 用于所有 global 索引与边界（定长时 qbase=b*S、len=S，逐式退化）。
  //   TMA 描述符在 varlen 下按 dims={D,T,H,1} 建，故行坐标用 `qbase+row`、batch 坐标恒 0；
  //   定长时 rowbase=0、tbatch=b，与原来的坐标 (row,b) 完全一致 ⇒ 定长逐位不变。
  const int qbase   = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len     = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int rowbase = cu_seqlens ? qbase : 0;
  const int tbatch  = cu_seqlens ? 0 : b;

  if (tid == 0) {
    mbar_init(bars + 0, 1);
    mbar_init(bars + 1, 1);
    mbar_init(bars + 2, 1);
    mbar_init(bars + 3, 1);
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect(bars + 0, 2 * QTILE);
    tma_fill_sw128<BM, HD>(Qs, &qmap, rowbase + m0, h, tbatch, bars + 0);
    tma_fill_sw128<BM, HD>(dOs, &dmap, rowbase + m0, h, tbatch, bars + 0);
  }

  const int ncols = causal ? min(len, m0 + BM) : len;
  const int ntiles = (ncols + BN - 1) / BN;
  if (ntiles > 0 && tid == 0) {
    mbar_arrive_expect(bars + 1, KTILE);
    tma_fill_sw128<BN, HD>(Ks, &kmap, rowbase, hkv, tbatch, bars + 1);
    mbar_arrive_expect(bars + 3, KTILE);
    tma_fill_sw128<BN, HD>(Vs, &vmap, rowbase, hkv, tbatch, bars + 3);
  }

  // 本 wg 的 Q 行 = wg*64 + [0,64)。每线程两行（r_lo / r_hi）。
  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < len) { const size_t idx = ((size_t)(qbase + qi_lo)) * H + h; lse_lo = lse[idx]; del_lo = delta[idx]; }
  if (qi_hi < len) { const size_t idx = ((size_t)(qbase + qi_hi)) * H + h; lse_hi = lse[idx]; del_hi = delta[idx]; }

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
      tma_fill_sw128<BN, HD>(Ks + (st ^ 1) * KTILE, &kmap, rowbase + (nt + 1) * BN, hkv,
                             tbatch, bars + 1 + (st ^ 1));
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
        if (qi < len && jg < len && !(causal && jg > qi)) p = fexp(sacc[j * 4 + qq] * scale - lv);
        pval[j][qq] = p;
      }
#pragma unroll
    for (int j = 0; j < 8; ++j)
      pds_store_sw128(Pw, j, r0, lane, c2, BN,
                      __float2half(pval[j][0]), __float2half(pval[j][1]),
                      __float2half(pval[j][2]), __float2half(pval[j][3]));
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float d0 = pval[j][0] * (dpacc[j * 4 + 0] - del_lo);
      const float d1 = pval[j][1] * (dpacc[j * 4 + 1] - del_lo);
      const float d2 = pval[j][2] * (dpacc[j * 4 + 2] - del_hi);
      const float d3 = pval[j][3] * (dpacc[j * 4 + 3] - del_hi);
      pds_store_sw128(dSw, j, r0, lane, c2, BN,
                      __float2half(d0), __float2half(d1),
                      __float2half(d2), __float2half(d3));
    }
    // 两个 wg 的 P/dS 都写好；同时 GEMM2 已读完 V[nt]，可覆盖 V。
    __syncthreads();
    if (nt + 1 < ntiles && tid == 0) {
      mbar_arrive_expect(bars + 3, KTILE);
      tma_fill_sw128<BN, HD>(Vs, &vmap, rowbase + (nt + 1) * BN, hkv, tbatch, bars + 3);
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
            wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
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
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float acck[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) acck[i] = 0.f;
          const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
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
        wgmma_fence();
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float accv[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) accv[i] = 0.f;
          const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
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
#pragma unroll
        for (int nh = 0; nh < 2; ++nh) {
          float acck[32];
#pragma unroll
          for (int i = 0; i < 32; ++i) acck[i] = 0.f;
          const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
          for (int s = 0; s < BM / 16; ++s)
            wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
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
        wgmma_m64n64k16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
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
          if (dq_h)
            *reinterpret_cast<__half2*>(dq_h + (((size_t)(qbase + qi)) * H + h) * HD + c) =
                __floats2half2_rn(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          else {
            float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
            *reinterpret_cast<float2*>(base) =
                make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
          }
        }
      }
}

// =============================================================================
// O17b：4 warpgroup（BM=256）wgmma 主 kernel —— 把 O17 的跨 wg 归约再砍半。
// =============================================================================
// 动机（O17 ncu）：BM=128 后 dK/dV 的跨 CTA `red` 已砍半（102.2M→51.9M），但仍占
// L2 扇区 ~72.6% ⇒ 仍是 **L2 原子字节数 bound**。BM=256 让每个 KV 元素只被 `nblk/4`
// 个 CTA 贡献 ⇒ red 再砍半。
//
// 可行性账（这是本项的关键，先算再做）：
//   * 寄存器：512 线程（4 wg）在 1 CTA/SM 下每线程上限 = 65536/512 = **128 regs**。
//     本算法每线程持有 dQ 累加器 `dqacc[2][8][4]=64` fp32；GEMM1/2 的 wgmma 累加器
//     `sacc/dpacc` 各 32（m64n64）、且两者必须同时存活到 wait0 后 ⇒ 峰值 64+64=128，
//     碰顶，实测 ptxas 会 spill 少量寄存器（见文档 §14k 的 `-Xptxas -v`）。
//   * smem：Q 64KB + dO 64KB + K 16KB + V 16KB + P 32KB + dS 32KB = 224KB（单缓冲 K/V）
//     ⇒ 1 CTA/SM（本卡上限 227KB）。K/V 用「单缓冲 + cp.async 后段预取」换 smem：
//     V 在 GEMM2 后、K 在 GEMM5 后各自预取下一 tile，延迟由 GEMM5/GEMM3/4 覆盖。
//
// 结构（HD=128, BM=256, BN=64, 512 线程 = 4 warpgroups）：
//   * 4 个 wg 各持自己 64 行 Q/dO，各自算 GEMM1/2（S、dP）并把 P/dS 写进共享 [256][BN] tile；
//   * **GEMM3 只由 wg0 做、GEMM4 只由 wg1 做，且都对全 BM=256 归约**（s=0..15 进同一累加器）
//     ⇒ 每个 KV 元素只 `red` 一次；
//   * GEMM5（dQ）4 个 wg 各算自己 64 行、寄存器累加，无跨 CTA 原子；
//   * 与 O17 同构，唯一区别：BM 翻倍、多两个 wg、K/V 单缓冲 + 后段 cp.async 预取。
// 数值只改归约次序（atomic 顺序），与 O9b/O17 在 fp16 噪声内一致。
template <int HD, bool SEQ = false>
__global__ void __launch_bounds__(512, 1)
fa_bwd_fp16_wgmma4_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                          const __half* __restrict__ v, const __half* __restrict__ do_,
                          const float* __restrict__ delta, const float* __restrict__ lse,
                          float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                          float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                          int causal) {
  static_assert(HD == 128, "wgmma4 主 kernel 目前只做 HD=128");
  constexpr int NTH = 512;
  constexpr int BM = 256, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;   // Q/dO SW128 tile（64KB）
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;   // K/V SW128 tile（16KB）
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;   // P/dS [256][64] SW128 tile（32KB）

  extern __shared__ char smem_raw[];
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* smem = smem_raw + pad;
  char* Qs  = smem;
  char* dOs = Qs + QTILE;
  char* Ks  = dOs + QTILE;                 // 单缓冲（后段 cp.async 预取）
  char* Vs  = Ks + KTILE;                  // 单缓冲
  char* Ps  = Vs + KTILE;                  // [256][64] SW128（32KB）
  char* dSs = Ps + PTILE;                  // [256][64] SW128（32KB）

  const int nblk = (S + BM - 1) / BM;
  const int mblk = blockIdx.x;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x;
  const int wg = tid >> 7;                 // warpgroup id（0..3）
  const int wid = (tid >> 5) & 3;          // warpgroup 内 warp id（0..3）
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

  // 本 wg 的 Q 行 = wg*64 + [0,64)。每线程两行（r_lo / r_hi）。
  const int r_lo = wg * 64 + wid * 16 + g;
  const int qi_lo = m0 + r_lo, qi_hi = qi_lo + 8;
  float lse_lo = 0.f, lse_hi = 0.f, del_lo = 0.f, del_hi = 0.f;
  if (qi_lo < S) {
    const size_t idx = ((size_t)(b * S + qi_lo)) * H + h;
    lse_lo = lse[idx]; del_lo = delta[idx];
  }
  if (qi_hi < S) {
    const size_t idx = ((size_t)(b * S + qi_hi)) * H + h;
    lse_hi = lse[idx]; del_hi = delta[idx];
  }

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

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * BN;
    asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();

    // ---- (1)(2) 本 wg 的 S=QKᵀ 与 dP=dO·Vᵀ（m64n64）----
    // SEQ：把两条 GEMM 串行、P 写 smem 后**读回**再算 dS，避免 `sacc` 与 `dpacc` 同时存活
    //      （512 线程只有 128 regs，非 SEQ 版峰值 64(dqacc)+32+32 会 spill 且 spill 占 L2 ~50%）。
    if constexpr (SEQ) {
      float sacc[32];
      wgmma_mn64_issue(Qw, Ks, HD, sacc);
      wgmma_wait0();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float pp[4];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
          const int jg = j0 + j * 8 + c2 + (qq & 1);
          const float lv = (qq >= 2) ? lse_hi : lse_lo;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi))
            p = fexp(sacc[j * 4 + qq] * scale - lv);
          pp[qq] = p;
        }
        pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2half(pp[0]), __float2half(pp[1]),
                        __float2half(pp[2]), __float2half(pp[3]));
      }
      float dpacc[32];
      wgmma_mn64_issue(dOw, Vs, HD, dpacc);
      wgmma_wait0();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float p0, p1, p2, p3;
        pds_load_sw128(Pw, j, r0, c2, BN, p0, p1, p2, p3);
        const float d0 = p0 * (dpacc[j * 4 + 0] - del_lo);
        const float d1 = p1 * (dpacc[j * 4 + 1] - del_lo);
        const float d2 = p2 * (dpacc[j * 4 + 2] - del_hi);
        const float d3 = p3 * (dpacc[j * 4 + 3] - del_hi);
        pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2half(d0), __float2half(d1),
                        __float2half(d2), __float2half(d3));
      }
    } else {
      // 合并版：两条 GEMM 一起发、统一 wait0；P 的 fp32 值只在**每个 j 内**临时存在（pp[4]），
      // 不再用 `pval[8][4]`（省 32 regs；exp 只算一次、dS 用 fp32 的 P ⇒ 与 O17 数值一致）。
      float sacc[32], dpacc[32];
      wgmma_mn64_issue(Qw, Ks, HD, sacc);
      wgmma_mn64_issue(dOw, Vs, HD, dpacc);
      wgmma_wait0();
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        float pp[4];
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) {
          const int qi = m0 + r_lo + (qq >= 2 ? 8 : 0);
          const int jg = j0 + j * 8 + c2 + (qq & 1);
          const float lv = (qq >= 2) ? lse_hi : lse_lo;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi))
            p = fexp(sacc[j * 4 + qq] * scale - lv);
          pp[qq] = p;
        }
        pds_store_sw128(Pw, j, r0, lane, c2, BN, __float2half(pp[0]), __float2half(pp[1]),
                        __float2half(pp[2]), __float2half(pp[3]));
        const float d0 = pp[0] * (dpacc[j * 4 + 0] - del_lo);
        const float d1 = pp[1] * (dpacc[j * 4 + 1] - del_lo);
        const float d2 = pp[2] * (dpacc[j * 4 + 2] - del_hi);
        const float d3 = pp[3] * (dpacc[j * 4 + 3] - del_hi);
        pds_store_sw128(dSw, j, r0, lane, c2, BN, __float2half(d0), __float2half(d1),
                        __float2half(d2), __float2half(d3));
      }
    }
    // 两个 wg 的 P/dS 都写好；同时 GEMM2 已读完 V[nt]，可覆盖 V（单缓冲后段预取）。
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, false, true, NTH>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                                  Ks, Vs);

    // ---- (5) dQ += scale·dS·K：每个 wg 算自己 64 行（A=本 wg 的 dS 行块）----
#pragma unroll
    for (int nh = 0; nh < 2; ++nh) {
      float accq[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) accq[i] = 0.f;
      wgmma_fence();
      const uint32_t Kt_n = smem_u32(Ks) + (uint32_t)(nh * 1024);
      const uint32_t dSw_a = smem_u32(dSw);
#pragma unroll
      for (int s = 0; s < BN / 16; ++s)
        wgmma_m64n64k16_t<0, 1>(accq, desc_k16_k(dSw_a, s, BN), desc_k16_mn(Kt_n, s, HD));
      wgmma_commit();
      wgmma_wait0();
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int qq = 0; qq < 4; ++qq) dqacc[nh][j][qq] += accq[j * 4 + qq] * scale;
    }
    // K 已被 GEMM5 读完（本 tile 最后一次用 K），可覆盖（单缓冲后段预取）。
    __syncthreads();
    if (nt + 1 < ntiles)
      kv_issue_async_sw<HD, BN, true, false, NTH>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
                                                  Ks, Vs);

    if (wg == 0) {
      // ---- (3) dV = Pᵀ·dO：wg0 对全 BM=256 归约（s=0..15 进同一累加器）----
      wgmma_fence();
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        float accv[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) accv[i] = 0.f;
        const uint32_t dOn = dOa + (uint32_t)(nh * 1024);
#pragma unroll
        for (int s = 0; s < BM / 16; ++s)
          wgmma_m64n64k16_t<1, 1>(accv, desc_k16_mn(Pa, s, BN), desc_k16_mn(dOn, s, HD));
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
    } else if (wg == 1) {
      // ---- (4) dK = scale·dSᵀ·Q：同构 ----
      wgmma_fence();
#pragma unroll
      for (int nh = 0; nh < 2; ++nh) {
        float acck[32];
#pragma unroll
        for (int i = 0; i < 32; ++i) acck[i] = 0.f;
        const uint32_t Qn = Qa + (uint32_t)(nh * 1024);
#pragma unroll
        for (int s = 0; s < BM / 16; ++s)
          wgmma_m64n64k16_t<1, 1>(acck, desc_k16_mn(DSa, s, BN), desc_k16_mn(Qn, s, HD));
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
          float* base = dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c;
          *reinterpret_cast<float2*>(base) =
              make_float2(dqacc[nh][j][qq], dqacc[nh][j][qq + 1]);
        }
      }
}

#endif  // FA_WGMMA

template <int HD, int BM, int BN, int PIPE, bool R4 = false, bool PREL = true,
          int NTH = THREADS, int NWAR = WN>
__global__ void __launch_bounds__(NTH, (NTH == 128) ? ((BN > 32) ? 2 : 3) : 1)
fa_bwd_fp16_mma_kernel(const __half* __restrict__ q, const __half* __restrict__ k,
                       const __half* __restrict__ v, const __half* __restrict__ do_,
                       const float* __restrict__ delta, const float* __restrict__ lse,
                       float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                       float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                       int causal, int sched, const int* __restrict__ cu_seqlens = nullptr,
                       int ksplit = 1) {
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

  // ---- 由 (BM,BN) 派生的 warp 网格几何（O5c：支持 BN=64 等更大 tile；O46：warp 几何参数化）----
  // GEMM1/2（S/dP，输出 [BM][BN]）：每个 warp 吃 (BM/NWM)×(BN/NWAR)
  // GEMM3/4（dV/dK，输出 [BN][HD]）：每个 warp 吃 (BN/NWM)×(NTW/NWAR)，NTW=128（N-tile 宽）
  // GEMM5（dQ，输出 [BM][HD]）：每个 warp 吃 (BM/NWM)×(NTW/NWAR)
  // 记 MT* 为每个 warp 的 m16/n8 tile 数。默认 NTH=128/NWAR=2（2×2 网格）与 O5c 逐字一致。
  // O46：MLA（HD=512）用 NTH=256/NWAR=4（2×4 网格）⇒ 同 1 CTA/SM 下每 scheduler 的 warp 数
  //   从 1 翻到 2（O45 的墙），改善延迟隐藏；smem 与 grid 不变、dK/dV 归约字节不变。
  constexpr int NWM = NTH / 32 / NWAR;        // warp 行数（M 方向）
  static_assert(NWM * NWAR * 32 == NTH, "NTH 必须 = NWM×NWAR×32");
  constexpr int NTW = 128;                    // N-tile 宽（GEMM3/4/5 每遍覆盖的 head_dim 列）
  constexpr int NDT = HD / NTW;               // N-tile 遍数（HD=128→1，HD=512→4）
  constexpr int GM1 = BM / NWM, GN1 = BN / NWAR;   // GEMM1/2 warp tile
  constexpr int GMV = BN / NWM, GNV = NTW / NWAR;  // GEMM3/4 warp tile（N-tile 内）
  constexpr int GMQ = BM / NWM, GNQ = NTW / NWAR;  // GEMM5 warp tile（N-tile 内）
  constexpr int MTM1 = GM1 / 16, MTN1 = GN1 / 8;
  constexpr int MTMV = GMV / 16, MTNV = GNV / 8;
  constexpr int MTMQ = GMQ / 16, MTNQ = GNQ / 8;
  static_assert(GM1 % 16 == 0 && GN1 % 8 == 0 && GMV % 16 == 0 && GNV % 8 == 0 &&
                    GMQ % 16 == 0 && GNQ % 8 == 0,
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
  // O44：N 方向 split-K（split-KV）。MLA（HD=512）主 kernel 的 grid = ceil(S/BM)·H·B
  //   太小（S1024H2 只有 64 CTA < 132 SM，ncu Waves 0.48，half-SM 空转），不能像 fp8
  //   (O29) / fp16 wgmma2 (O43) 那样把 K tile 均分到 ksplit 个 CTA。这里给 mma 主 kernel
  //   （含 HD=128 fallback 与 HD=512 MLA）加同款：每个 (mblk) 的 KV tile [0,ntiles) 均分给
  //   ksplit 个 CTA，dQ 相应改成跨 CTA `red_add2`（ksplit==1 时逐式退化：ksp=0/mblk=bx/
  //   nt_begin=0/nt_end=ntiles，且 dQ 仍走非原子 RMW，数值与原路径逐位相同）。
  const int bx = blockIdx.x;
  const int nblk = (S + BM - 1) / BM;
  int ksp = 0, mblk = bx;
  if (ksplit > 1) {
    ksp = bx % ksplit;
    mblk = bx / ksplit;                 // 同一 mblk 的 ksplit 个 CTA 共享 Q/dO，切 K
  } else if (causal) {
    if (sched == 1)
      mblk = (bx & 1) ? (nblk - 1 - (bx >> 1)) : (bx >> 1);
    else if (sched == 2)
      mblk = nblk - 1 - bx;
  }
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  // VARLEN（第 81 轮）：cu_seqlens 给出本序列 packed [T,H,D] 的 token 基址与长度；nullptr
  // 退化为定长（qbase=b*S、len=S），定长路径逐位不变。sched 仅定长 causal 用（varlen 传 0）。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  // ---- 载入 Q/dO（越界补 0）----
  // O10：PIPE>=1 时用 16B `cp.async` 异步发 Q/dO（与 K/V 一起在循环首 `wait_group 0` 等待），
  // 让 Q/dO 的全局延迟与 K/V 重叠；PIPE==0 无流水语义，保持同步标量读。
  if constexpr (PIPE >= 1) {
    qdo_issue_async<HD, BM, NTH>(q, do_, m0, len, H, h, b, tid, Qs, dOs, LD, qbase);
  } else {
    for (int i = tid; i < BM * HD; i += NTH) {
      int r = i / HD, d = i % HD;
      int qi = m0 + r;
      __half qv = __float2half(0.f), ov = __float2half(0.f);
      if (qi < len) {
        size_t idx = (((size_t)(qbase + qi)) * H + h) * HD + d;
        qv = q[idx];
        ov = do_[idx];
      }
      Qs[r * LD + d] = qv;
      dOs[r * LD + d] = ov;
    }
  }

  const int ncols = causal ? min(len, m0 + BM) : len;
  const int ntiles = (ncols + BN - 1) / BN;
  // O44：本 CTA 负责的 KV tile 切片 [nt_begin, nt_end)。ksplit==1 时即 [0,ntiles)。
  const int nt_begin = (ksplit > 1) ? (int)(((long)ntiles * ksp) / ksplit) : 0;
  const int nt_end = (ksplit > 1) ? (int)(((long)ntiles * (ksp + 1)) / ksplit) : ntiles;
  if (ksplit > 1 && nt_end <= nt_begin) return;   // 该切片无 tile（causal 小 mblk 可能被切空）

  // O44：prologue 的 stage 必须与循环首 `stage = nt&1` 一致；ksplit==1 时 nt_begin=0 ⇒ st0=0，
  //   与历史逐位相同。这是 split-KV 的第一个坑（只改数据偏移、忘改 stage 会让 K 读到空 buffer）。
  const int st0 = (ksplit > 1) ? (nt_begin & 1) : 0;
  if constexpr (PIPE == 0) {
    __syncthreads();
  } else if constexpr (PIPE == 1) {
    // O6：prologue 直接异步发起本切片首个 tile 的 K/V（不占寄存器）；Q/dO 的可见性由
    // 循环首的 `wait_group + __syncthreads` 一并保证（Q/dO 与 K/V 写不同 smem）。
    if (nt_end > nt_begin)
      kv_issue_async<HD, BN, true, true, NTH>(k, v, nt_begin * BN, len, Hkv, hkv, b, tid,
                                         Ks + st0 * KVL, Vs + st0 * KVL, LD, qbase);
  } else {
    // O6b：切片首 tile 的 K 发进双缓冲 stage、V 发进单缓冲 Vs（两个独立 commit_group）。
    if (nt_end > nt_begin) {
      kv_issue_async<HD, BN, true, false, NTH>(k, v, nt_begin * BN, len, Hkv, hkv, b, tid,
                                          Ks + st0 * KVL, Vs, LD, qbase);
      kv_issue_async<HD, BN, false, true, NTH>(k, v, nt_begin * BN, len, Hkv, hkv, b, tid,
                                          Ks + st0 * KVL, Vs, LD, qbase);
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
        const size_t idx = ((size_t)(qbase + qi)) * H + h;
        const bool ok = qi < len;
        lse_r[i][s] = ok ? lse[idx] : 0.f;
        del_r[i][s] = ok ? delta[idx] : 0.f;
      }
  }

  for (int nt = nt_begin; nt < nt_end; ++nt) {
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
      if (nt + 1 < nt_end)
        kv_issue_async<HD, BN, true, true, NTH>(k, v, (nt + 1) * BN, len, Hkv, hkv, b, tid,
                               Ks + ((nt + 1) & 1) * KVL, Vs + ((nt + 1) & 1) * KVL, LD, qbase);
    } else if constexpr (PIPE == 2) {
      // O6b：等本 tile 的 K[nt] 与上一轮发出的 V[nt] 落地（同一个 wait_group 0 覆盖）。
      // 此 barrier 同时证明「上一 tile 的 GEMM5 已读完 Ks 的另一 stage」，故可把 K[nt+1]
      // 发进该 stage；V 单缓冲，下一 tile 的 V 留到 GEMM2 之后才发（见下）。
      asm volatile("cp.async.wait_group 0;\n");
      __syncthreads();
      if (nt + 1 < nt_end)
        kv_issue_async<HD, BN, true, false, NTH>(k, v, (nt + 1) * BN, len, Hkv, hkv, b, tid,
                               Ks + ((nt + 1) & 1) * KVL, Vs, LD, qbase);
    } else {
      // ---- 载入 K/V 块（原版：同步标量读）----
      for (int i = tid; i < BN * HD; i += NTH) {
        int r = i / HD, d = i % HD;
        int jg = j0 + r;
        __half kv = __float2half(0.f), vv = __float2half(0.f);
        if (jg < len) {
          size_t idx = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d;
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
            else if (qi < len)
              lv = lse[((size_t)(qbase + qi)) * H + h];
            float p = 0.f;
            if (qi < len && jg < len && !(causal && jg > qi))
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
            else if (qi < len)
              del = delta[((size_t)(qbase + qi)) * H + h];
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
      if (nt + 1 < nt_end)
        kv_issue_async<HD, BN, false, true, NTH>(k, v, (nt + 1) * BN, S, Hkv, hkv, b, tid,
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
              float* dst = dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c;
              if constexpr (R4) {
                // O7c：shfl 必须在 guard 之外（全 warp 参与），只有偶 lane 落 float4。
                float a2 = __shfl_down_sync(0xffffffffu, acc[i][j][q], 1);
                float b2 = __shfl_down_sync(0xffffffffu, acc[i][j][q + 1], 1);
                if (jg < len && (lane & 1) == 0)
                  red_add4(dst, acc[i][j][q], acc[i][j][q + 1], a2, b2);
              } else {
                if (jg < len) red_add2(dst, acc[i][j][q], acc[i][j][q + 1]);
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
              float* dst = dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c;
              if constexpr (R4) {
                float a = acc[i][j][q] * scale, b = acc[i][j][q + 1] * scale;
                float a2 = __shfl_down_sync(0xffffffffu, a, 1);
                float b2 = __shfl_down_sync(0xffffffffu, b, 1);
                if (jg < len && (lane & 1) == 0) red_add4(dst, a, b, a2, b2);
              } else {
                if (jg < len) red_add2(dst, acc[i][j][q] * scale, acc[i][j][q + 1] * scale);
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
              // HD>128：dQ 的 HD 列放不进寄存器 → 直接全局累加。ksplit==1 时每个 (qi, 列)
              // 由唯一线程拥有（唯一 CTA + 唯一 warp + 唯一 N-tile），非原子 RMW 无竞争；
              // ksplit>1（O44）时同一 (qi,列) 会被 ksplit 个 CTA 各加一次 → 改用跨 CTA
              // `red_add2`（float2 atomicAdd），dq_acc 由 host memset(0) 清零。
#pragma unroll
              for (int q = 0; q < 4; q += 2) {
                int r = wr * GMQ + i * 16 + g + (q >= 2 ? 8 : 0);
                int c = hd0 + wc * GNQ + j * 8 + c2;
                int qi = m0 + r;
                if (qi < len) {
                  float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
                  if (ksplit > 1) {
                    red_add2(base, acc[i][j][q] * scale, acc[i][j][q + 1] * scale);
                  } else {
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
          if (qi < len) {
            float* base = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
            if (ksplit > 1) {
              // O44：split-K 下同一 Q 行被多个 CTA 贡献 ⇒ 跨 CTA 原子累加。
              red_add2(base, dqacc[i][j][q], dqacc[i][j][q + 1]);
            } else {
              // O10：相邻两列打包成一次 8B 写（c 为偶数 → 自然 8B 对齐），减少全局 store 事务。
              *reinterpret_cast<float2*>(base) =
                  make_float2(dqacc[i][j][q], dqacc[i][j][q + 1]);
            }
          }
        }
  }
}

// =============================================================================
// 2b) O7b：确定性 dK/dV 归约 —— 把 per-(Q-block) 的 partial 按 mblk 升序求和
// =============================================================================
// partial 布局：`part[((b*Hkv+hkv)*nblk + mblk) * S * HD + jg*HD + c]`（每个 CTA 一份）。
// 输出布局：`dk_acc[((b*S+jg)*Hkv + hkv)*HD + c]`。
// causal 下 KV 行 `jg` 只被 `mblk >= jg/BM`（BM=128）的 CTA 写 ⇒ 直接从 `jg/BM` 起求和；
// 非 causal 从 0 起。**求和顺序固定**（mblk 升序）⇒ 与调度无关，可复现（deterministic）。
// partial 按 **Q 头 h**（不是 KV 头）分片：GQA/MQA 下多个 Q 头 `h` 共享同一 KV 头 `hkv`，
// 它们对 dK/dV 的贡献要**求和**；若 partial 只用 hkv 索引，同 (mblk, hkv) 的多个 h 会互相
// 覆盖（race，实测 GQA 对拍错 O(1)）。这里 partial 按 `(b*H + h)*nblk + mblk` 分片，归约时
// 对每个 KV 头 `hkv` 把其 head group `[hkv*G, (hkv+1)*G)`（G=H/Hkv）与 mblk 一起求和。
template <int HD>
__global__ void dkv_reduce_kernel(const float* __restrict__ dk_part,
                                  const float* __restrict__ dv_part,
                                  float* __restrict__ dk_acc, float* __restrict__ dv_acc,
                                  int S, int H, int Hkv, int nblk, int causal) {
  const int hb = blockIdx.x;   // b*Hkv + hkv
  const int jg = blockIdx.y;
  const int c = threadIdx.x;
  if (c >= HD || jg >= S) return;
  const int b = hb / Hkv, hkv = hb % Hkv;
  const int G = H / Hkv;                    // 每 KV 头对应的 Q 头数（GQA 广播组）
  const int h0 = hkv * G;
  const int mblk0 = causal ? (jg / 128) : 0;
  float sk = 0.f, sv = 0.f;
  for (int hh = 0; hh < G; ++hh) {
    const size_t prow = (size_t)(b * H + h0 + hh);
    for (int m = mblk0; m < nblk; ++m) {
      const size_t base = ((prow * nblk + m) * (size_t)S + jg) * HD + c;
      sk += dk_part[base];
      sv += dv_part[base];
    }
  }
  const size_t o = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c;
  dk_acc[o] = sk;
  dv_acc[o] = sv;
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

#if defined(FA_WGMMA) && defined(FA_TMA)
// O30：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={64,64,1,1}）。
// globalStride（字节）：dim1(S) 的行距 = H*D*2，dim2(H) 的头距 = D*2，dim3(B) 的批距。
// 要求 16B 对齐（D=128 时 256 的整数倍，恒成立）。坐标 {k0,row,head,batch}。
static CUtensorMap make_lse_map(const void* ptr, long long H, long long S, long long D,
                                long long B) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 2), (uint64_t)(D * 2),
                         (uint64_t)(S * H * D * 2)};
  uint32_t box[4] = {64, 64, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled failed: %s\n", s);
    std::exit(1);
  }
  return map;
}

// O33：主 kernel 用的大张量描述符（dims={D,S,H,B}，SW128）。box={64,8,1,1} —— 内维 128B
// （64 个 fp16）= SW128 跨距，8 行 = 一个 1024B atom；逐 atom TMA 即可原样写出 `sw128_off`
// 的交织布局（见 `tma_fill_sw128`）。
static CUtensorMap make_main_map(const void* ptr, long long H, long long S, long long D,
                                 long long B) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D * 2), (uint64_t)(D * 2),
                         (uint64_t)(S * H * D * 2)};
  uint32_t box[4] = {64, 8, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, (void*)ptr, dims, strides, box, estr,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    const char* s = "?";
    cuGetErrorString(r, &s);
    fprintf(stderr, "cuTensorMapEncodeTiled(main) failed: %s\n", s);
    std::exit(1);
  }
  return map;
}
#endif

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
template <int HD, int BM, int BN, int PIPE, bool R4 = false, bool PREL = true,
          int NTH = THREADS, int NWAR = WN>
static void launch_bwd_mma(dim3 mg, const __half* q, const __half* k, const __half* v,
                           const __half* do_, const float* delta, const float* lse,
                           float* dq_acc, float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int sched,
                           const int* cu_seqlens = nullptr, int ksplit = 1) {
  constexpr int kvn = (PIPE == 0) ? 2 : (PIPE == 1 ? 4 : 3);
  constexpr int pds = (PIPE == 2) ? 2 * BM * (BN + 8) : 2 * BN * (BM + 8) + BM * (BN + 8);
  constexpr int smem =
      (2 * BM * (HD + 8) + kvn * BN * (HD + 8) + pds) * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE, R4, PREL, NTH, NWAR>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_mma_kernel<HD, BM, BN, PIPE, R4, PREL, NTH, NWAR><<<mg, NTH, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, sched,
      cu_seqlens, ksplit);
}

#ifdef FA_WGMMA
// O9b/O9b-2：wgmma 主 kernel（只 HD=128 / BM=BN=64）。Q/dO/K/V SW128 + P/dS SW128，≈97KB。
template <int HD, bool OW = false>
static void launch_bwd_wgmma(dim3 mg, const __half* q, const __half* k, const __half* v,
                             const __half* do_, const float* delta, const float* lse,
                             float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                             int Hkv, float scale, int causal, int sched) {
  constexpr int BM = 64, BN = 64;
  constexpr int TILE  = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int smem = 1024 + TILE * 2 + KTILE * 3 + 2 * BM * BN * (int)sizeof(__half);
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma_kernel<HD, OW>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma_kernel<HD, OW><<<mg, THREADS, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                          dk_acc, dv_acc, S, H, Hkv, scale,
                                                          causal, sched);
}

// O17：2 warpgroup（BM=128）wgmma 主 kernel（只 HD=128）。Q/dO/K/V + P/dS 全 SW128。
// smem = 1024(对齐) + Q 32KB + dO 32KB + K 2×16KB + V 16KB + P 16KB + dS 16KB ≈ 145KB。
template <int HD, bool SPLIT = true, int CL = 1>
static void launch_bwd_wgmma2(dim3 mg, const __half* q, const __half* k, const __half* v,
                              const __half* do_, const float* delta, const float* lse,
                              float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                              int Hkv, float scale, int causal,
                              __half* dq_h = nullptr,
                              const int* cu_seqlens = nullptr,
                              int ksplit = 1) {
  static_assert(HD == 128, "wgmma2 只做 HD=128");
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  // O25：CL>1 时 leader 的 smem 里多两个 [BN][HD] fp32 合并累加器（dK/dV）。
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 +
                       (CL > 1 ? 2 * BN * HD * (int)sizeof(float) : 0);
  static_assert(CL == 1 || CL == 2, "cluster 只做 2");
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  if constexpr (CL > 1) {
    // O25：用 cudaLaunchKernelEx 指定 cluster 维度（沿 x，相邻两个 mblk 配对）。
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = mg;
    cfg.blockDim = dim3(256);
    cfg.dynamicSmemBytes = smem;
    cfg.stream = nullptr;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&cfg, fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL>, q, k, v,
                                  do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale,
                                  causal, dq_h, cu_seqlens, ksplit));
  } else {
    fa_bwd_fp16_wgmma2_kernel<HD, SPLIT, CL><<<mg, 256, smem>>>(
        q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, dq_h,
        cu_seqlens, ksplit);
  }
}

// O18：BN=128 版 wgmma2（只 HD=128）。tile 数减半；smem = 1024 + Q32 + dO32 + K 2×32 + V32 + P32 + dS32
// ≈ 230400B（仍 1 CTA/SM）。
template <int HD, bool SPLIT = true, bool DET = false>
static void launch_bwd_wgmma2b(dim3 mg, const __half* q, const __half* k, const __half* v,
                               const __half* do_, const float* delta, const float* lse,
                               float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                               int Hkv, float scale, int causal,
                               float* dk_part = nullptr, float* dv_part = nullptr,
                               int nblk = 0, __half* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b 只做 HD=128");
  constexpr int BM = 128, BN = 128;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2b_kernel<HD, SPLIT, DET>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma2b_kernel<HD, SPLIT, DET><<<mg, 256, smem>>>(
      q, k, v, do_, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, dk_part,
      dv_part, nblk, dq_h);
}

#if defined(FA_WGMMA) && defined(FA_TMA)
// O33：把 wgmma2b 的 Q/K/V/dO 载入换成逐 atom 4D-TMA（布局逐字节不变）。smem 与 wgmma2b 同
// （+ 4 个 mbarrier 32B），故 1 CTA/SM 不变。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2b_tma(dim3 mg, CUtensorMap qmap, CUtensorMap kmap,
                                   CUtensorMap vmap, CUtensorMap dmap, const float* delta,
                                   const float* lse, float* dq_acc, float* dk_acc,
                                   float* dv_acc, int S, int H, int Hkv, float scale,
                                   int causal, __half* dq_h = nullptr) {
  static_assert(HD == 128, "wgmma2b TMA 只做 HD=128");
  constexpr int BM = 128, BN = 128;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 + 128;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2b_tma_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma2b_tma_kernel<HD, SPLIT><<<mg, 256, smem>>>(
      qmap, kmap, vmap, dmap, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
      dq_h);
}

// O35：把 **BN=64** 的 wgmma2 主 kernel 的 Q/K/V/dO 换成逐 atom 4D-TMA（布局逐字节不变）。
// smem 与 cp.async 版 wgmma2 同（+ 4 个 mbarrier 32B），故 1 CTA/SM 不变。
template <int HD, bool SPLIT = true>
static void launch_bwd_wgmma2_tma(dim3 mg, CUtensorMap qmap, CUtensorMap kmap,
                                  CUtensorMap vmap, CUtensorMap dmap, const float* delta,
                                  const float* lse, float* dq_acc, float* dk_acc,
                                  float* dv_acc, int S, int H, int Hkv, float scale,
                                  int causal, __half* dq_h = nullptr,
                                  const int* cu_seqlens = nullptr) {
  static_assert(HD == 128, "wgmma2 TMA 只做 HD=128");
  constexpr int BM = 128, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 3 + PTILE * 2 + 128;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma2_tma_kernel<HD, SPLIT>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma2_tma_kernel<HD, SPLIT><<<mg, 256, smem>>>(
      qmap, kmap, vmap, dmap, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
      dq_h, cu_seqlens);
}
#endif

// O17b：4 warpgroup（BM=256）wgmma 主 kernel（只 HD=128）。Q/dO/P/dS SW128，K/V 单缓冲 + 后段预取。
// smem = 1024(对齐) + Q 64KB + dO 64KB + K 16KB + V 16KB + P 32KB + dS 32KB = 224KB（1 CTA/SM）。
template <int HD, bool SEQ = false>
static void launch_bwd_wgmma4(dim3 mg, const __half* q, const __half* k, const __half* v,
                              const __half* do_, const float* delta, const float* lse,
                              float* dq_acc, float* dk_acc, float* dv_acc, int S, int H,
                              int Hkv, float scale, int causal) {
  static_assert(HD == 128, "wgmma4 只做 HD=128");
  constexpr int BM = 256, BN = 64;
  constexpr int QTILE = (BM / 8) * (HD / 64) * 1024;
  constexpr int KTILE = (BN / 8) * (HD / 64) * 1024;
  constexpr int PTILE = (BM / 8) * (BN / 64) * 1024;
  constexpr int smem = 1024 + QTILE * 2 + KTILE * 2 + PTILE * 2;
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp16_wgmma4_kernel<HD, SEQ>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fa_bwd_fp16_wgmma4_kernel<HD, SEQ><<<mg, 512, smem>>>(q, k, v, do_, delta, lse, dq_acc,
                                                        dk_acc, dv_acc, S, H, Hkv, scale, causal);
}
#endif

// =============================================================================
// VARLEN（变长 / cu_seqlens）自测入口（fp16，HD=128/512，causal/full）
// =============================================================================
// packed 布局：q/dO/dQ `[T,H,D]`；k/v/dK/dV `[T,Hkv,D]`；`cu_seqlens.npy`（fp32，B+1 个
// token 前缀和）。LSE/主 kernel 用 `cu_seqlens[b]` 作 token 基址、`maxlen` 传 S；每个
// `(b,h,mblk)` 只处理本序列内的 tile。ref_dq/dk/dv 也是 packed，逐元素比对。
// HD=128 走 wgmma2 路径（causal 镜像配对 wgmma LSE，非 causal 走 O8 mma LSE）；HD=512（MLA）
// 走 mma 主 kernel（BM=32/BN=32/PIPE=1，与定长 D=512 同几何）。FA/TE 变长在本机不可用 ⇒ 仅对 ref。
#ifdef FA_WGMMA
static int run_varlen(const std::string& dir, bool causal, int iters, int varlen_tma = 0,
                      int lse_split = 0, int wg2ksplit = -1) {
  auto q_np = load_npy_f32(dir + "/q.npy");
  auto k_np = load_npy_f32(dir + "/k.npy");
  auto v_np = load_npy_f32(dir + "/v.npy");
  auto do_np = load_npy_f32(dir + "/do.npy");
  auto o_np = load_npy_f32(dir + "/ref_o.npy");
  auto rdq = load_npy_f32(dir + "/ref_dq.npy");
  auto rdk = load_npy_f32(dir + "/ref_dk.npy");
  auto rdv = load_npy_f32(dir + "/ref_dv.npy");
  auto cu_np = load_npy_f32(dir + "/cu_seqlens.npy");
  if (q_np.shape.size() != 3 || k_np.shape.size() != 3) {
    fprintf(stderr, "VARLEN 期望 q 为 [T,H,D]、k 为 [T,Hkv,D]\n");
    return 1;
  }
  const int T = (int)q_np.shape[0], H = (int)q_np.shape[1], D = (int)q_np.shape[2];
  const int Hkv = (int)k_np.shape[1];
  const int B = (int)cu_np.data.size() - 1;
  if (D != 128 && D != 512) {
    fprintf(stderr, "VARLEN 目前只做 HD=128/512；当前 %d\n", D);
    return 1;
  }
  int maxlen = 0;
  for (int b = 0; b < B; ++b) {
    int L = (int)cu_np.data[b + 1] - (int)cu_np.data[b];
    if (L > maxlen) maxlen = L;
  }
  const size_t nq = (size_t)T * H * D;
  const size_t nkv = (size_t)T * Hkv * D;
  const size_t rows_q = (size_t)T * H;
  const float scale = 1.0f / sqrtf((float)D);
  printf("case = %s\n", dir.c_str());
  printf("VARLEN: B=%d T=%d maxlen=%d H=%d Hkv=%d D=%d causal=%d\n", B, T, maxlen, H, Hkv, D,
         (int)causal);
  printf("cu_seqlens =");
  for (int b = 0; b <= B && b < 12; ++b) printf(" %d", (int)cu_np.data[b]);
  printf("%s\n", (B > 11) ? " ..." : "");

  auto to_half = [&](const std::vector<float>& src) {
    std::vector<__half> h(src.size());
    for (size_t i = 0; i < src.size(); ++i) h[i] = __float2half(src[i]);
    return h;
  };
  auto qh = to_half(q_np.data), kh = to_half(k_np.data), vh = to_half(v_np.data),
       doh = to_half(do_np.data), oh = to_half(o_np.data);

  __half *dq, *dk, *dv, *d_q, *d_k, *d_v, *d_o, *d_do;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc;
  int* d_cu;
  CUDA_CHECK(cudaMalloc(&dq, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dk, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&dv, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_q, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_k, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_v, nkv * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_o, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_do, nq * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * sizeof(float)));
  // O40：varlen LSE 的 K 维 split 部分结果（[T*H][ksplit] 个 (m,l)），按最大 split=16 预留。
  float* d_lse_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, nq * sizeof(float)));   // dq_h 路径下不用，占位
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * sizeof(float)));
  std::vector<int> cu(B + 1);
  for (int b = 0; b <= B; ++b) cu[b] = (int)cu_np.data[b];
  CUDA_CHECK(cudaMalloc(&d_cu, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cu, cu.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q, qh.data(), nq * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k, kh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v, vh.data(), nkv * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do, doh.data(), nq * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o, oh.data(), nq * sizeof(__half), cudaMemcpyHostToDevice));

  // VARLEN TMA（本轮）：packed 布局 [T,H,D] 没有 batch 维，故描述符按 dims={D,T,H,1} 建
  //   （复用 make_main_map，令 S=T、B=1，strides 自动为 {H*D*2, D*2, T*H*D*2}）。kernel 侧
  //   用行坐标 `cu_seqlens[b]+row`、batch 坐标 0 选择序列。仅 fp16/HD=128。
#if defined(FA_WGMMA) && defined(FA_TMA)
  const int varlen_tma_use = (D == 128) ? varlen_tma : 0;
  CUtensorMap vqmap, vkmap, vvmap, vdmap;
  if (varlen_tma_use) {
    vqmap = make_main_map(d_q, H, T, D, 1);
    vkmap = make_main_map(d_k, Hkv, T, D, 1);
    vvmap = make_main_map(d_v, Hkv, T, D, 1);
    vdmap = make_main_map(d_do, H, T, D, 1);
  }
#else
  const int varlen_tma_use = 0;
#endif
  printf("VARLEN main backend = %s\n",
         varlen_tma_use ? "wgmma2 TMA (Q/K/V/dO 4D-TMA)" : "wgmma2 cp.async");

  const int lse_nblk = (maxlen + LBM - 1) / LBM;
  dim3 lg_bal((lse_nblk + 1) / 2, H, B);
  // non-causal 用 O8 的 mma `lse_mma_kernel<HD>`（Q/K 行距 D+8）；causal D=512 用
  // `lse_mma_kernel_bal<512,1>`（镜像配对 + cu_seqlens，PIPE=1 双缓冲）。
  const int kLseSmem = (LBM + LBN) * (D + 8) * (int)sizeof(__half);
  const int kLseSmemBal1 = (LBM + 2 * LBN) * (D + 8) * (int)sizeof(__half);
  const int kLseTileWgm = (D == 128) ? (LBM / 8) * (D / 64) * 1024 : 0;
  const int kLseSmemWgm1 = 1024 + kLseTileWgm * 3;
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm1));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<128>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
  } else {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<512>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  }
  const int d_rows = (int)rows_q;
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  const int main_bm = (D == 512) ? 32 : 128;   // D=512：MLA 主 kernel 几何（BM=32）
  dim3 mg((maxlen + main_bm - 1) / main_bm, H, B);

  // O43：varlen 主 kernel（D==128 的 wgmma2）N 方向 split-K——短序列时 base grid 很小
  //   （如 b1 [512] → 4×16=64 CTA < 132 SM），按需切 K 填满 SM；dQ 改走跨 CTA 原子累加。
  //   TMA 版（`--varlentma=1`）未加 ksplit ⇒ 不切；`--wg2ksplit=N` 可强制/关闭（=1）。
  int wg2_ks = 1;
  // 本轮实测：varlen 短序列切 K 反慢（Q/dO 重载 + 额外 convert 抵不过填 SM），故**不做 auto**，
  //   仅 `--wg2ksplit=N>=2` 显式开启（见 docs 的负结果）；TMA 版未加 ksplit。
  if (D == 128 && !varlen_tma_use && wg2ksplit >= 2) {
    wg2_ks = wg2ksplit;
    mg.x *= (unsigned)wg2_ks;
  }
  const bool vq_direct = (D == 128) && (wg2_ks <= 1);

  // O40：varlen LSE 的 K 维 split auto（D=128 目标 `grid*split≈528`=一个波、cap 8；D=512
  //   `≈132`、cap 16，与 O38/O39 同标定）；`--lsesplit=N`（>0）直接指定。再按最大序列 tile 数封顶。
  const int lse_nblk0 = (maxlen + LBM - 1) / LBM;
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long base = (long)((lse_nblk0 + 1) / 2) * H * B;
    const int target = (D == 512) ? 132 : 528;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && base * (sp * 2) <= target) sp *= 2;
    while (sp > lse_nblk0 && sp > 1) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("[O40] varlen lse k-split = %d (base=%ld)\n", lse_split_eff,
         (long)((lse_nblk0 + 1) / 2) * H * B);

  const int cvt_threads = 256;
  const int cvt_blocks =
      (int)std::min<size_t>((std::max(nq, nkv) + cvt_threads - 1) / cvt_threads, 65535);

  auto run_all = [&]() {
    if (D == 128 && wg2_ks > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    if (D == 128) {
      if (causal) {
        if (lse_split_eff > 1) {
          // O40：K 维切片 + 二次归约。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_wgmma<128, 1><<<gsp, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part, lse_split_eff);
          const long long nrows = (long long)rows_q;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu);
        }
      } else
        lse_mma_kernel<128><<<dim3(lse_nblk, H, B), THREADS, kLseSmem>>>(d_q, d_k, d_lse,
                                                                         maxlen, H, Hkv, scale,
                                                                         0, d_cu);
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      // O24：dQ 由主 kernel 直接写 fp16（dq_h），convert 跳过 dQ（n_q=0）。
      // O43：ksplit>1 时 dQ 跨 CTA 原子累加 ⇒ dq_h 传 nullptr、convert 转全部 nq/nkv。
      __half* dqo = vq_direct ? dq : nullptr;
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (varlen_tma_use)
        launch_bwd_wgmma2_tma<128, true>(mg, vqmap, vkmap, vvmap, vdmap, d_delta, d_lse,
                                         d_dq_acc, d_dk_acc, d_dv_acc, maxlen, H, Hkv, scale,
                                         (int)causal, dqo, d_cu);
      else
#endif
        launch_bwd_wgmma2<128, true, 1>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, maxlen, H, Hkv, scale, (int)causal,
                                        dqo, d_cu, wg2_ks);
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv,
                                                  vq_direct ? 0 : nq, nkv);
    } else {
      // HD=512（MLA）：causal LSE 走 mma 镜像配对版（带 cu_seqlens）；dQ 由主板 kernel 全局
      // 累加（NDT>1）⇒ 先清零 dq_acc，convert 转全部 nq/nkv。
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * sizeof(float)));
      if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part, lse_split_eff);
          const long long nrows = (long long)rows_q;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, maxlen, H, Hkv, scale, d_cu);
        }
      } else
        lse_mma_kernel<512><<<dim3(lse_nblk, H, B), THREADS, kLseSmem>>>(d_q, d_k, d_lse,
                                                                         maxlen, H, Hkv, scale,
                                                                         0, d_cu);
      delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      launch_bwd_mma<512, 32, 32, 1, false, true>(mg, d_q, d_k, d_v, d_do, d_delta, d_lse,
                                                  d_dq_acc, d_dk_acc, d_dv_acc, maxlen, H, Hkv,
                                                  scale, (int)causal, 0, d_cu);
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv, nq,
                                                  nkv);
    }
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
  double flops = 0.0;
  for (int b = 0; b < B; ++b) {
    double L = cu[b + 1] - cu[b];
    flops += 4.0 * H * L * L * D;  // 反向 ≈ 2×fwd，因果再乘系数（口径同 fp8 varlen）
  }
  printf("[timing] VARLEN total %.4f ms  %.2f TFLOPS (sum_b 4HL^2D)\n", ms,
         flops / (ms * 1e-3) / 1e12);
  printf("grid main = %d x %d x %d | lse grid = %d x %d x %d | T=%d\n",
         (maxlen + main_bm - 1) / main_bm, H, B, (lse_nblk + 1) / 2, H, B, T);

  std::vector<float> h_dq(nq), h_dk(nkv), h_dv(nkv);
  std::vector<__half> h_dq_h(nq), h_dk_h(nkv), h_dv_h(nkv);
  CUDA_CHECK(cudaMemcpy(h_dq_h.data(), dq, nq * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dk_h.data(), dk, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv_h.data(), dv, nkv * sizeof(__half), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < nq; ++i) h_dq[i] = __half2float(h_dq_h[i]);
  for (size_t i = 0; i < nkv; ++i) { h_dk[i] = __half2float(h_dk_h[i]); h_dv[i] = __half2float(h_dv_h[i]); }
  auto report = [](const char* nm, const std::vector<float>& a, const NpyF32& b) {
    size_t n = std::min(a.size(), b.data.size());
    double ma = 0.0, mr = 0.0, ao = 0.0, ar = 0.0;
    size_t arg = 0;
    for (size_t i = 0; i < n; ++i) {
      double d = fabs((double)a[i] - (double)b.data[i]);
      if (d > ma) { ma = d; arg = i; }
      ao = std::max(ao, fabs((double)a[i]));
      ar = std::max(ar, fabs((double)b.data[i]));
      double den = std::max(1e-3, fabs((double)b.data[i]));
      double r = d / den;
      if (r > mr) mr = r;
    }
    printf("  %-3s vs ref: max_abs=%.3e  max_rel=%.3e  (ours_amax=%.3e ref_amax=%.3e @%zu)\n",
           nm, ma, mr, ao, ar, arg);
  };
  printf("[compare] VARLEN ours vs fp32 ref\n");
  report("dq", h_dq, rdq);
  report("dk", h_dk, rdk);
  report("dv", h_dv, rdv);
  return 0;
}
#endif  // FA_WGMMA

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
  // O7c：dK/dV 归约宽度（-1=自动/float2，0=float2，1=float4）。
  int r4_opt = -1;
  // O7c：LSE/D 预装寄存器（-1=自动/开，0=关，1=开）。
  int prel_opt = -1;
  // O9：LSE 是否用 wgmma（仅 D==128 且 causal；0=用 O8b 的 mma 版，1=wgmma 版）。
  int lse_wgm = 0;
  // O23：lse_wgm 是否被用户显式指定（--lsewgm=0/1）。未指定时在 FA_WGMMA 构建下默认开。
  bool lse_forced = false;
  // O30：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal；1=用 4D TMA 载入 Q/K）。
  //   -1=自动（FA_TMA 构建下 D==128/causal 默认开），0/1 由 `--lsetma=` 强制。
  int lse_tma = -1;
  // O38（fp16 版）：LSE 的 K 维 split（0=auto，1=关；>1=切片数）。仅 TMA/causal 路径生效。
  int lse_split = 0;
  // O9b：主 kernel 是否用 wgmma（仅 FA_WGMMA 构建、D==128 且 sel=(64,64) 时生效）。
  int wgmma_sel = 0;
  // O16：wgmma 主 kernel 是否用「分段 wait_group」重叠 epilogue（-1=自动/开，0=关，1=开）。
  int ow_opt = -1;
  // O17：2 warpgroup（BM=128，跨 wg 归约）wgmma 主 kernel（仅 FA_WGMMA 构建、D==128）。
  // O23：默认自动选择（见下方 auto 段）；`--wg2=0/1` 可强制。wg_forced=用户显式指定。
  int wg2_sel = 0;
  // O18：BN=128 版 wgmma2（仅 FA_WGMMA 构建、D==128）。O23：默认自动选择。
  int wg2bn_sel = 0;
  bool wg_forced = false;
  // O17-2：wgmma2 的 GEMM3/GEMM4 是否拆分到两个 warpgroup（1=拆，0=原版 wg0 串行）。
  int wg2split_sel = 1;
  // O17b：4 warpgroup（BM=256，跨 wg 归约再砍半）wgmma 主 kernel（仅 FA_WGMMA 构建、D==128）。
  int wg4_sel = 0;
  // O17b：是否用「串行 GEMM1/2 + 读回 P」版（消 spill）；-1=自动（SEQ=1）。
  int wg4seq_opt = -1;
  // O7b：是否跑「确定性 dK/dV（partial + 二次归约）」A/B（仅 FA_WGMMA 构建、D==128）。
  int det_ab = 0;
  // O24：delta 用 warp-per-row 向量化版（1，默认）还是旧 block-per-row smem 版（0，A/B）。
  int delta_warp_sel = 1;
  // O24：D==128 wgmma2/2b 路径直接用 fp16 写 dQ、convert 跳过 dQ（1，默认；0=A/B）。
  int dq_direct_sel = 1;
  // O25：cluster 分布式归约（仅 HD=128、BN=64 的 wgmma2；cluster 沿 bx 配对相邻 mblk）。
  //   0=关（默认），2=开。开了会强制走 wgmma2(BN=64)（BN=128 的 2b 放不下合并累加器）。
  int cluster_sel = 0;
  // O33：主 kernel 的 Q/K/V/dO 是否用逐 atom 4D-TMA 载入（仅 FA_WGMMA+FA_TMA 构建、D==128、
  //   BN=128 的 wgmma2b 几何）。1=用 TMA，0=cp.async（默认）。同 binary A/B。
  int maintma_sel = 0;
  // O43：wgmma2（BN=64）主 kernel 的 N 方向 split-K。`-1`=自动（仅 D==128、非 cluster/maintma、
  //   且未切块 grid 不足一个波时按需切）；`1`=关（A/B）；`>=2`=强制。
  int wg2ksplit = -1;
  // O44：MLA（HD=512）mma 主 kernel 的 N 方向 split-K（split-KV）。`-1`=自动（D==512 时按
  //   「填满一个波」切，D==128 恒 1）；`1`=关（A/B）；`>=2`=强制。
  int mlaksplit = -1;
  // O46：MLA（HD=512）mma 主 kernel 的 warp 几何。0=4 warp/128 线程（2×2 网格）；
  //   1=8 warp/256 线程（2×4 网格，默认）——同 1 CTA/SM 下每 scheduler 的 warp 数 1→2，
  //   实测 main 1.075–1.109×、端到端 1.03–1.07×，且 8-warp 实例 148–151 regs/0 spill
  //   （4-warp 是 168 regs + spill）。`--mla8w=0` 供同 binary A/B。
  int mla8w = 1;
  // O48（候选 ①）：D=128 mma 主 kernel 是否用 8-warp（256 线程 / 2×4 网格）几何。仅 mma 路径
  //   （wgmma2 已是 256 线程、结构不同）。O49：默认改为 **-1=自动**——判据是「4-warp 的每
  //   scheduler warp 数是否 <2」= 逻辑 grid ≤ SM 数（与 O43 的 split-K「填满一个波」同思路，
  //   但 mma 路径不切 K、二者不冲突）。0=强制 4-warp，1=强制 8-warp。`--d128w=`。
  int d128w = -1;
  int varlen = 0;   // VARLEN：packed [T,H,D] + cu_seqlens.npy（fp16/HD=128/causal/wgmma2）
  // 本轮：VARLEN 主 kernel 是否用 4D-TMA 载入 Q/K/V/dO（仅 FA_WGMMA+FA_TMA 构建、HD=128）。
  //   默认 0（opt-in）：本轮实测 varlen 的 BN=64 主 kernel 用 TMA 中性/偏负（同 O35），
  //   与 cp.async 版做同 binary A/B 用 `--varlentma=1`。
  int varlen_tma = -1;
  int iters = 50;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--varlen") varlen = 1;
    else if (a.rfind("--varlen=", 0) == 0) varlen = atoi(a.c_str() + 9);
    else if (a.rfind("--varlentma=", 0) == 0) varlen_tma = atoi(a.c_str() + 12);
    else if (a == "--varlentma") varlen_tma = 1;
    else if (a == "--full") causal = false;
    else if (a == "--causal") causal = true;
    else if (a == "--nopipe") pipe = 0;
    else if (a == "--pipe") pipe = 1;
    else if (a == "--pipe2") pipe = 2;
    else if (a.rfind("--sched=", 0) == 0) sched = atoi(a.c_str() + 8);
    else if (a.rfind("--bm=", 0) == 0) bm_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--bn=", 0) == 0) bn_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--r4=", 0) == 0) r4_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--prel=", 0) == 0) prel_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--lsewgm=", 0) == 0) { lse_wgm = atoi(a.c_str() + 9); lse_forced = true; }
    else if (a == "--lsewgm") { lse_wgm = 1; lse_forced = true; }
    else if (a.rfind("--lsetma=", 0) == 0) lse_tma = atoi(a.c_str() + 9);
    else if (a == "--lsetma") lse_tma = 1;
    else if (a.rfind("--lsesplit=", 0) == 0) lse_split = atoi(a.c_str() + 11);
    else if (a.rfind("--wgmma=", 0) == 0) wgmma_sel = atoi(a.c_str() + 8);
    else if (a == "--wgmma") wgmma_sel = 1;
    else if (a.rfind("--ow=", 0) == 0) ow_opt = atoi(a.c_str() + 5);
    else if (a.rfind("--wg2=", 0) == 0) { wg2_sel = atoi(a.c_str() + 6); wg_forced = true; }
    else if (a.rfind("--wg2split=", 0) == 0) wg2split_sel = atoi(a.c_str() + 11);
    else if (a == "--wg2") { wg2_sel = 1; wg_forced = true; }
    else if (a.rfind("--wg2bn=", 0) == 0) { wg2bn_sel = atoi(a.c_str() + 8); wg_forced = true; }
    else if (a == "--wg2bn") { wg2bn_sel = 1; wg_forced = true; }
    else if (a.rfind("--maintma=", 0) == 0) maintma_sel = atoi(a.c_str() + 10);
    else if (a == "--maintma") maintma_sel = 1;
    else if (a.rfind("--wg2ksplit=", 0) == 0) wg2ksplit = atoi(a.c_str() + 12);
    else if (a.rfind("--mlaksplit=", 0) == 0) mlaksplit = atoi(a.c_str() + 12);
    else if (a.rfind("--mla8w=", 0) == 0) mla8w = atoi(a.c_str() + 8);
    else if (a == "--mla8w") mla8w = 1;
    else if (a.rfind("--d128w=", 0) == 0) d128w = atoi(a.c_str() + 8);
    else if (a == "--d128w") d128w = 1;
    else if (a.rfind("--wg4=", 0) == 0) wg4_sel = atoi(a.c_str() + 6);
    else if (a == "--wg4") wg4_sel = 1;
    else if (a.rfind("--wg4seq=", 0) == 0) wg4seq_opt = atoi(a.c_str() + 9);
    else if (a.rfind("--det=", 0) == 0) det_ab = atoi(a.c_str() + 6);
    else if (a == "--det") det_ab = 1;
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_sel = atoi(a.c_str() + 12);
    else if (a.rfind("--dqdirect=", 0) == 0) dq_direct_sel = atoi(a.c_str() + 11);
    else if (a.rfind("--cluster=", 0) == 0) cluster_sel = atoi(a.c_str() + 10);
    else if (a == "--cluster") cluster_sel = 2;
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  if (varlen) {
#ifdef FA_WGMMA
    if (varlen_tma < 0) varlen_tma = 0;   // 本轮判决：varlen TMA 中性/偏负 ⇒ opt-in
    return run_varlen(dir, causal, iters, varlen_tma, lse_split, wg2ksplit);
#else
    fprintf(stderr, "VARLEN 需要 -DFA_WGMMA（sm_90a）构建\n");
    return 1;
#endif
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
  if (D != 128 && D != 512) {
    fprintf(stderr, "O5 fp16 mma 版支持 head_dim=128/512；当前 %d\n", D);
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
  float* d_lse_part = nullptr;
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
  // O38：LSE K-split 的部分 (m,l)，每行最多 16 个 split × 2 个 fp32。
  CUDA_CHECK(cudaMalloc(&d_lse_part, (size_t)B * S * H * 16 * 2 * sizeof(float)));
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
  // LSE smem 随 head_dim 变化：Qs[LBM*LD] +（PIPE=0 时 1 份 / PIPE=1 时 2 份）Ks[LBN*LD]，LD=HD+8。
  const int LDl = D + 8;
  const int kLseSmem = (LBM + LBN) * LDl * (int)sizeof(__half);
  // O8b：PIPE=0 单缓冲（与 O8 同尺寸），PIPE=1 双缓冲。
  const int kLseSmemBal0 = (LBM + LBN) * LDl * (int)sizeof(__half);
  const int kLseSmemBal1 = (LBM + 2 * LBN) * LDl * (int)sizeof(__half);
  // O9：wgmma LSE（仅 D=128）用 SW128 tile：TILE=(LBM/8)*(D/64)*1024 B，另加 1024B 对齐余量。
  const int kLseTileWgm = (LBM / 8) * (D / 64) * 1024;
  const int kLseSmemWgm0 = 1024 + kLseTileWgm * 2;  // Q + K（单缓冲）
  const int kLseSmemWgm1 = 1024 + kLseTileWgm * 3;  // Q + 2×K
  // O30：TMA 版与 wgmma 版同布局（Q + 2×K）+ 3 个 mbarrier（24B，取 64B 余量）。
  const int kLseSmemTma1 = 1024 + kLseTileWgm * 3 + 64;
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 0>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm0));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemWgm1));
#if defined(FA_WGMMA) && defined(FA_TMA)
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kLseSmemTma1));
#endif
  }
#if defined(FA_WGMMA) && defined(FA_TMA)
  // O30：建 LSE 的 Q/K 4D TMA 描述符（一次，供所有 (h,b) CTA 用坐标选择）。
  CUtensorMap qmap_lse, kmap_lse;
  if (D == 128) {
    qmap_lse = make_lse_map(d_q, H, S, D, B);
    kmap_lse = make_lse_map(d_k, Hkv, S, D, B);
  }
  // O33：主 kernel 的 Q/K/V/dO 描述符（box={64,8}，逐 atom）。
  CUtensorMap qmap_m, kmap_m, vmap_m, dmap_m;
  if (D == 128) {
    qmap_m = make_main_map(d_q, H, S, D, B);
    kmap_m = make_main_map(d_k, Hkv, S, D, B);
    vmap_m = make_main_map(d_v, Hkv, S, D, B);
    dmap_m = make_main_map(d_do, H, S, D, B);
  }
#endif
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<128>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 0>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal0));
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<128, 1>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  if (D == 512) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<512>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmem));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 0>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal0));
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, kLseSmemBal1));
  }
  const int cvt_threads = 256;
  const int cvt_blocks =
      (int)std::min<size_t>((std::max(n, nkv) + cvt_threads - 1) / cvt_threads, 65535);

  // O6c：主 kernel 的 tile/PIPE 自动选择。
  //  - grid < 132（不到「每 SM 一个 CTA」）且 S 较小时，把 BM 减半到 32 → grid 翻倍、
  //    并行度翻倍（S=512 MHA：main 0.0876→0.0792ms，1.11×）。大 S 下 BM=32 会让
  //    dK/dV 的跨 CTA 原子量翻倍，故只在 S≤1024 用。
  //  - 否则 BM=64：大网格走 O6b（PIPE=2，只双缓冲 K）；小网格走 O6（PIPE=1）。
  // O13：auto tile 重新标定（O7c-PREL 之后）。
  //  O6c 时代 `grid<132 且 S≤1024` 用 (BM=32,BN=32,PIPE=1) 换并行度；但 O7c-PREL（LSE/D
  //  预装寄存器）与其它优化后，BM=64 的每 CTA 效率更高：实测 S=512 MHA 主 kernel
  //  (64,64,2) 比 (32,32,1) 快 1.23×（O5c A/B），端到端 1.09×。故取消 BM=32 分支。
  //  BN=64 能降 L1/TEX，但 smem 105KB→2 CTA/SM；当 grid 落在「2 个波」的坏量化点
  //  （256<grid≤600，实测 grid=512 的 S1024/GQA-kv4）时反而输给 BN=32（3 CTA/SM），
  //  故仅该区间保留 BN=32。
  const long long grid = (long long)((S + 63) / 64) * H * B;
  const bool bn64 = (S >= 4096) || (grid <= 256) || (grid > 600);
  int auto_bm = 64;
  int auto_bn = bn64 ? 64 : 32;
  int auto_pipe = (grid >= 396) ? 2 : 1;
  if (D == 512) {
    // MLA（HD=512）：K/V 行有 520 个 half，BM=64 时双缓冲会超 smem，BM=32 时 PIPE=1
    // （K 双缓冲）仍在 232KB 内且实测最快（1.67× vs PIPE=0）；S/H 小、grid 不足一个波。
    auto_bm = 32;
    auto_bn = 32;
    auto_pipe = 1;
  }
  const int bm_sel = (bm_opt > 0) ? bm_opt : auto_bm;
  const int bn_sel = (bn_opt > 0) ? bn_opt : auto_bn;
  const int pp_sel = (pipe >= 0) ? pipe : auto_pipe;
  printf("[O6c] main grid=%lld auto=(BM=%d,BN=%d,PIPE=%d) sel=(BM=%d,BN=%d,PIPE=%d)\n", grid,
         auto_bm, auto_bn, auto_pipe, bm_sel, bn_sel, pp_sel);
  // O49：D=128 mma 路径的 8-warp **自动档**。判据 = 逻辑 grid ≤ SM 数（每 SM 仅 1 个 CTA ⇒
  //   4-warp 时每 scheduler 只有 1 个 warp）。`--d128w=0/1` 可强制关/开；D=512（MLA）走
  //   O46 的 mla8w，不经此路。仅影响 mma 路径（wgmma2/wgmma2b 已是 256 线程）。
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const bool d128w_eff =
      (d128w > 0) ? true : ((d128w < 0) ? (D == 128 && grid <= (long long)sm_count) : false);
  printf("[O49] d128 8-warp = %d (d128w=%d, grid=%lld, sm=%d)\n", (int)d128w_eff, d128w, grid,
         sm_count);

  // O44：MLA（D=512）主 kernel 的 N 方向 split-K 生效值。base grid = ceil(S/BM)·H·B 太小
  //   （S1024H2 只有 64 CTA < 132 SM），切 K 填并发槽。目标取 **`grid*sp ≈ 528`（1 CTA/SM
  //   的 4 个波）**——sweep（`--mlaksplit=1..16`）显示单纯「填满一个波（132）」太保守：
  //   S256H2 sp=8、S512H4 sp=8、S1024H2 sp=8–16 最优，对应 `≈512`（与 fp8 O29 对 MLA 用
  //   `S/2` 的有效目标一致）。再用 `nblk=ceil(S/BN)` 封顶（切得比 K tile 还细只产生空切片）。
  //   `--mlaksplit=N` 可强制/关；D!=512 恒 1（HD=128 的 wgmma2 已有 O43 的独立 ksplit）。
  int mlaksplit_eff = 1;
  if (D == 512) {
    if (mlaksplit >= 1) mlaksplit_eff = mlaksplit;
    else {
      const int base_m = (S + bm_sel - 1) / bm_sel;
      const long base = (long)base_m * H * B;
      int sp = 1;
      while (sp < 16 && base * (sp * 2) <= 528) sp *= 2;
      const int nt_cap = (S + bn_sel - 1) / bn_sel;
      while (sp > nt_cap && sp > 1) sp >>= 1;
      mlaksplit_eff = sp;
    }
  }
  if (D == 512)
    printf("[O44] MLA main k-split = %d (auto)\n", mlaksplit_eff);
  // O7c：`r4`（float4 归约）与 `prel`（LSE/D 预装寄存器）为运行期开关，各自派发到两个
  // 模板实例，便于同一 session 内做 2×2 A/B。
#define LAUNCH_CFG(HD_, BM_, BN_, PIPE_)                                                   \
  do {                                                                                     \
    if (r4) {                                                                              \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, true>(g, d_q, d_k, d_v, d_do, d_delta,  \
                                                         d_lse, d_dq_acc, d_dk_acc,        \
                                                         d_dv_acc, S, H, Hkv, scale,       \
                                                         (int)causal, sched, nullptr, mlaksplit_eff); \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, false>(g, d_q, d_k, d_v, d_do, d_delta, \
                                                          d_lse, d_dq_acc, d_dk_acc,       \
                                                          d_dv_acc, S, H, Hkv, scale,      \
                                                          (int)causal, sched, nullptr, mlaksplit_eff); \
    } else {                                                                               \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, true>(g, d_q, d_k, d_v, d_do, d_delta, \
                                                          d_lse, d_dq_acc, d_dk_acc,       \
                                                          d_dv_acc, S, H, Hkv, scale,      \
                                                          (int)causal, sched, nullptr, mlaksplit_eff); \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, false>(                            \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched, nullptr, mlaksplit_eff);                      \
    }                                                                                      \
  } while (0)

  // O46：8-warp（256 线程 / 2×4 网格）变体。仅 MLA D==512/BM=32/PIPE=1 使用（`--mla8w`）；
  //   其余几何回退 4-warp 的 LAUNCH_CFG。同 binary A/B 用 `--mla8w=0/1`。
#define LAUNCH_CFG_W(HD_, BM_, BN_, PIPE_, NTH_, NW_)                                      \
  do {                                                                                     \
    if (r4) {                                                                              \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, true, NTH_, NW_>(                       \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched, nullptr, mlaksplit_eff);                       \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, true, false, NTH_, NW_>(                      \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched, nullptr, mlaksplit_eff);                       \
    } else {                                                                               \
      if (prel)                                                                            \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, true, NTH_, NW_>(                      \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched, nullptr, mlaksplit_eff);                       \
      else                                                                                 \
        launch_bwd_mma<HD_, BM_, BN_, PIPE_, false, false, NTH_, NW_>(                     \
            g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,    \
            Hkv, scale, (int)causal, sched, nullptr, mlaksplit_eff);                       \
    }                                                                                      \
  } while (0)

  auto launch_cfg = [&](int bm, int bn, int pp, bool r4, bool prel, bool w8) {
    dim3 g((S + bm - 1) / bm, H, B);
    if (D == 512 && mlaksplit_eff > 1) g.x *= (unsigned)mlaksplit_eff;   // O44：split-KV 抬 grid
    if (D == 512) {
      // MLA：BN=32；BM 可 32/64；只支持 PIPE=0/1（PIPE=2 的 V 单缓冲无意义且更费 smem）。
      if (bm == 64) {
        if (pp == 1) LAUNCH_CFG(512, 64, 32, 1);
        else LAUNCH_CFG(512, 64, 32, 0);
      } else {
        if (pp == 1) {
          if (mla8w) LAUNCH_CFG_W(512, 32, 32, 1, 256, 4);
          else LAUNCH_CFG(512, 32, 32, 1);
        } else {
          LAUNCH_CFG(512, 32, 32, 0);
        }
      }
      return;
    }
    // O48/O49：D=128 mma 路径的 8-warp（256 线程 / 2×4 网格）几何。`w8` = O49 的自动档
    //   （grid≤SM）或 `--d128w=1` 强制。wgmma 路径不走这里（wgmma2 已 256 线程）。
    //   BN=64 与 4-warp 路径一致地固定 PIPE=2（该几何下 V 单缓冲无意义且更费 smem）。
    if (w8) {
      if (bm == 32) {
        if (pp == 2) LAUNCH_CFG_W(128, 32, 32, 2, 256, 4);
        else if (pp == 1) LAUNCH_CFG_W(128, 32, 32, 1, 256, 4);
        else LAUNCH_CFG_W(128, 32, 32, 0, 256, 4);
      } else if (bn == 64) {
        LAUNCH_CFG_W(128, 64, 64, 2, 256, 4);
      } else if (pp == 2) {
        LAUNCH_CFG_W(128, 64, 32, 2, 256, 4);
      } else if (pp == 1) {
        LAUNCH_CFG_W(128, 64, 32, 1, 256, 4);
      } else {
        LAUNCH_CFG_W(128, 64, 32, 0, 256, 4);
      }
      return;
    }
    if (bm == 32) {
      if (pp == 2) LAUNCH_CFG(128, 32, 32, 2);
      else if (pp == 1) LAUNCH_CFG(128, 32, 32, 1);
      else LAUNCH_CFG(128, 32, 32, 0);
    } else if (bn == 64) {
      LAUNCH_CFG(128, 64, 64, 2);
    } else if (pp == 2) {
      LAUNCH_CFG(128, 64, 32, 2);
    } else if (pp == 1) {
      LAUNCH_CFG(128, 64, 32, 1);
    } else {
      LAUNCH_CFG(128, 64, 32, 0);
    }
  };
#undef LAUNCH_CFG
#undef LAUNCH_CFG_W
  const bool r4_sel = (r4_opt > 0);
  const bool prel_sel = (prel_opt >= 0) ? (prel_opt != 0) : true;
  // O16：默认关（实测中性 1.004×，且会改 dK/dV 的 atomic 次序、破坏历史逐位值）；保留 A/B。
  const bool ow_sel = (ow_opt > 0);
  // O17b：SEQ（串行 GEMM1/2 + 读回 half P）默认关（有 fp16 精度损失）；默认走合并版。
  const bool wg4seq_sel = (wg4seq_opt > 0);
  // O23：把 O17/O18 的 Hopper wgmma2 主 kernel 在 `-DFA_WGMMA`（sm_90a）构建下**默认打开**
  //   （对齐 fp8 的 O22）。仅 D==128：S>=4096 用 BN=128（O18，tile 数减半），否则 BN=64（O17）。
  //   用户显式传 `--wg2=`/`--wg2bn=` 时不做自动选择；`--wg2=0 --wg2bn=0` 退回 mma 路径做 A/B。
  //   非 FA_WGMMA 构建行为完全不变（wg2_sel/wg2bn_sel 恒 0）。
  // O43：wgmma2(BN=64) 主 kernel 的 N 方向 split-K 生效值（仅定长 D==128 的 cp.async 路径）。
  int wg2_ksplit_eff = 1;
#ifdef FA_WGMMA
  if (!wg_forced && D == 128) {
    if (S >= 4096) wg2bn_sel = 1; else wg2_sel = 1;
  }
  // O25：cluster 版只存在于 BN=64 的 wgmma2（BN=128 的 2b 放不下 [BN][HD]×2 合并累加器），
  //   故 --cluster 开启时强制走 wgmma2；grid.x=nblk 须能被 cluster 整除，否则退回普通 wg2。
  int cluster_use = 0;
  if (cluster_sel > 1 && D == 128) {
    const int nblk = (S + 127) / 128;
    if (nblk % cluster_sel == 0) { wg2_sel = 1; wg2bn_sel = 0; cluster_use = cluster_sel; }
  }
  // O43：只在「BN=64 的 wgmma2、非 cluster、非 TMA」上切 K。小 S（base grid < 132）按需
  //   「填满一个波」；`--wg2ksplit=N` 可强制/关闭（=1）。大 S / BN=128 网格已够，不切。
  if (D == 128 && wg2_sel && !wg2bn_sel && cluster_use == 0 && !maintma_sel) {
    const int base = (int)((S + 127) / 128) * H * B;
    if (wg2ksplit >= 1) wg2_ksplit_eff = wg2ksplit;
    else {
      int sp = 1;
      while (sp < 8 && base * (sp * 2) <= 132) sp *= 2;
      int nt_cap = (S + 63) / 64;
      while (sp > nt_cap && sp > 1) sp >>= 1;
      wg2_ksplit_eff = sp;
    }
  }
  // O23：LSE 预处理也默认走 Hopper wgmma 版（仅 causal / D==128；非 causal 自动落回 O8 原版）。
  if (!lse_forced && D == 128 && causal) lse_wgm = 1;
#endif
  // O30：TMA 版 LSE 需驱动 API（`cuTensorMapEncodeTiled`）⇒ 只有 `-DFA_TMA -lcuda` 构建才编译
  //   该路径；此时 D==128/causal 默认开（1.30–1.36× 于 wgmma+cp.async，且逐位相同）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (lse_tma < 0) lse_tma = (D == 128 && causal) ? 1 : 0;
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  // O38（fp16 版）：自动 split 档（0=auto）。目标 `grid*split ≈ 528`（= 4 CTA/SM × 132 SM，
  //   即「填满一个波」；实测 fp16 LSE TMA 的 smem ~50KB ⇒ 4 CTA/SM），上限 8；grid 已达一个
  //   波则退回 1（逐位）。仅 TMA 路径生效。与 fp8 的 `≈2048` 不同：fp16/bf16 的 LSE 并行度
  //   更高、且大 S 时 512 CTA 已铺满一个波（S4096 split>1 反而慢 3–7%），故以「一波」为准。
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long lg_grid = (long)lg_bal.x * H * B;
    // O39：D=512（MLA，mma LSE，全 dtype 适用）目标 `grid*split ≈ 256`、上限 16；D=128 的
    //   TMA LSE 维持 O38 的「一波」目标 528、上限 8。两者都只在小 grid 时生效。
    const int target = (D == 512) ? 132 : 528;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && lg_grid * (sp * 2) <= target) sp *= 2;
    // 再按 `nblk=ceil(S/64)` 封顶：切得比 K tile 数还细只会产生空切片 + 每 CTA 的 Q 载入开销。
    int nblk_cap = (S + 63) / 64;
    while (sp > nblk_cap) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("[O38] lse k-split = %d%s\n", lse_split_eff, (lse_split <= 0 ? " (auto)" : ""));
  printf("[O23] main backend = %s | lse = %s (D=%d S=%d)%s%s%s\n",
         wg2bn_sel ? "wgmma2b(BN=128)" : (wg2_sel ? "wgmma2(BN=64)" : "mma"),
         (D == 128 && causal && lse_wgm) ? "wgmma" : "mma", D, S,
#ifdef FA_WGMMA
         cluster_use ? " +cluster2" : "",
#else
         "",
#endif
         ((wg2bn_sel || wg2_sel) && maintma_sel) ? " +maintma" : "",
         (wg2_ksplit_eff > 1) ? " +ksplit" : "");
  printf("[O43] wgmma2 k-split = %d\n", wg2_ksplit_eff);
  // O24：D==128 的 wgmma2/wgmma2b 路径里 dQ 唯一拥有 ⇒ 主 kernel 直接写 fp16 `dq`，
  // `convert_kernel` 跳过 dQ（n_q 传 0）。其它路径（mma/wgmma/wgmma4/MLA）仍写 fp32 dq_acc。
  bool dq_direct = false;
  auto run_main = [&]() {
#ifdef FA_WGMMA
    if (wg4_sel && D == 128) {
      dim3 g((S + 255) / 256, H, B);
      if (wg4seq_sel)
        launch_bwd_wgmma4<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
      else
        launch_bwd_wgmma4<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
      return;
    }
    if (wg2bn_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      dq_direct = dq_direct_sel;
      __half* dqo = dq_direct_sel ? dq : nullptr;
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (maintma_sel) {
        if (wg2split_sel)
          launch_bwd_wgmma2b_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal, dqo);
        else
          launch_bwd_wgmma2b_tma<128, false>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                             d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                             (int)causal, dqo);
        return;
      }
#endif
      if (wg2split_sel)
        launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, nullptr,
                                      nullptr, 0, dqo);
      else
        launch_bwd_wgmma2b<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, nullptr,
                                       nullptr, 0, dqo);
      return;
    }
    if (wg2_sel && D == 128) {
      dim3 g((S + 127) / 128, H, B);
      // O43：ksplit>1 时 dQ 跨 CTA 原子累加、不能直写 fp16（cluster/TMA 分支不受影响，其
      //   wg2_ksplit_eff 恒 1）。
      dq_direct = dq_direct_sel && (wg2_ksplit_eff <= 1);
      __half* dqo = dq_direct ? dq : nullptr;
      if (cluster_use == 2) {
        // O25：cluster 分布式归约（仅 SPLIT 版）。
        launch_bwd_wgmma2<128, true, 2>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo);
        return;
      }
#if defined(FA_WGMMA) && defined(FA_TMA)
      // O35：BN=64 的 wgmma2 也走逐 atom 4D-TMA（`--maintma`）。
      if (maintma_sel) {
        if (wg2split_sel)
          launch_bwd_wgmma2_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                           d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                           (int)causal, dqo);
        else
          launch_bwd_wgmma2_tma<128, false>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal, dqo);
        return;
      }
#endif
      g.x *= (unsigned)wg2_ksplit_eff;   // O43：N 方向 split-K 抬高 grid
      if (wg2split_sel)
        launch_bwd_wgmma2<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo,
                                     nullptr, wg2_ksplit_eff);
      else
        launch_bwd_wgmma2<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, dqo,
                                      nullptr, wg2_ksplit_eff);
      return;
    }
    if (wgmma_sel && D == 128 && bm_sel == 64 && bn_sel == 64) {
      dim3 g((S + 63) / 64, H, B);
      if (ow_sel)
        launch_bwd_wgmma<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      else
        launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                     d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, sched);
      return;
    }
#endif
    launch_cfg(bm_sel, bn_sel, pp_sel, r4_sel, prel_sel, d128w_eff);
  };
  // O24：delta 的 warp-per-row 版；`--deltawarp=0` 回旧版 A/B。行数 = B*S*H。
  const int d_rows = S * H * B;
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  auto run_pre = [&]() {
    if (D == 512) {
      // O39：MLA（D=512）causal LSE 也用 K 维 split（此前只有 D=128/TMA 有 O38）。
      if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        }
      } else
        lse_mma_kernel<512><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                       (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      else
        delta_kernel<512><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
    } else {
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (causal && lse_tma) {
        if (lse_split_eff > 1) {
          // O38：K 维切片 + 二次归约。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_tma<128, 1><<<gsp, THREADS, kLseSmemTma1>>>(
              qmap_lse, kmap_lse, d_lse, d_lse_part, S, H, Hkv, scale, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows,
                                                       lse_split_eff);
        } else {
          lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
              qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
        }
      } else
#endif
      if (causal && lse_wgm) {
        if (lse_split_eff > 1) {
          // O40：K 维切片 + 二次归约（对齐 O38/O39 的 mma/TMA LSE）。
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal_wgmma<128, 1><<<gsp, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale);
        }
      } else if (causal) {
        if (lse_split_eff > 1) {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * lse_split_eff));
          lse_mma_kernel_bal<128, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, lse_split_eff);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, lse_split_eff);
        } else {
          lse_mma_kernel_bal<128, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        }
      } else
        lse_mma_kernel<128><<<lg, THREADS, kLseSmem>>>(d_q, d_k, d_lse, S, H, Hkv, scale,
                                                       (int)causal);
      if (delta_warp_sel)
        delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
      else
        delta_kernel<128><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
    }
  };

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  auto run_all = [&]() {
    // O13：HD=128（NDT==1）时 dQ 由主 kernel **覆盖写**（寄存器累加后一次写回），无需清零；
    // 只有 MLA（HD=512）的 GEMM5 走全局 RMW 累加才需要 memset。省掉一趟 n 个 float 的 memset。
    if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
    run_pre();
    run_main();
    // O24：dq_direct 时主 kernel 已直接写 fp16 dq ⇒ convert 跳过 dQ（n_q=0）。
    convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, dq, dk, dv,
                                                dq_direct ? 0 : n, nkv);
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

  if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O24 A/B：delta 旧 block-per-row(smem 归约) vs 新 warp-per-row(向量化) ----
  if (D == 128) {
    auto time_delta = [&](int mode, float* out_ms) {
      auto launch = [&]() {
        if (mode)
          delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o, d_do, d_delta, d_rows);
        else
          delta_kernel<128><<<pg, THREADS>>>(d_o, d_do, d_delta, S, H);
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
    float d_old = 0.f, d_new = 0.f;
    time_delta(0, &d_old);
    time_delta(1, &d_new);
    printf("[O24 A/B] delta old %.4f ms | warp %.4f ms (%.2fx)\n", d_old, d_new,
           d_old / d_new);
  }

  // ---- O8b A/B（仅 causal，HD=128）：LSE 原版(O8) vs 镜像配对 vs 镜像配对+cp.async ----
  if (D == 128 && causal) {
    auto time_lse = [&](int mode, float* out_ms) {
      auto launch = [&]() {
        if (mode == 3)
          lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse,
                                                                              S, H, Hkv, scale);
        else if (mode == 2)
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
    // 仅当显式开启 wgmma 时才跑 O9 A/B（sm_90 构建下 wgmma 是空实现，时序无意义）。
    if (lse_wgm) {
      float ms_wgm = 0.f;
      time_lse(3, &ms_wgm);
      printf("[O9 A/B] lse wgmma+SW128 %.4f ms (vs bal+cpasync %.3fx, vs O8 %.3fx)\n", ms_wgm,
             ms_balp / ms_wgm, ms_o8 / ms_wgm);
    }
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O30 A/B：LSE 的 wgmma+cp.async 版 vs 4D-TMA 版（同 session + 数值对拍）----
    if (lse_tma) {
      float* d_lse2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lse2, (size_t)B * S * H * sizeof(float)));
      auto launch_w = [&]() {
        lse_mma_kernel_bal_wgmma<128, 1><<<lg_bal, THREADS, kLseSmemWgm1>>>(d_q, d_k, d_lse2,
                                                                            S, H, Hkv, scale);
      };
      auto launch_t = [&]() {
        lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
            qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
      };
      auto time_one = [&](auto launch, float* out_ms) {
        for (int i = 0; i < 3; ++i) launch();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) launch();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float ms_w = 0.f, ms_t = 0.f;
      time_one(launch_w, &ms_w);
      time_one(launch_t, &ms_t);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> l1((size_t)B * S * H), l2((size_t)B * S * H);
      CUDA_CHECK(cudaMemcpy(l1.data(), d_lse, l1.size() * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(l2.data(), d_lse2, l2.size() * 4, cudaMemcpyDeviceToHost));
      double e = 0;
      for (size_t i = 0; i < l1.size(); ++i) e = std::max(e, (double)fabs(l1[i] - l2[i]));
      printf("[O30 A/B] lse tma %.4f ms | wgmma %.4f ms (wgmma/tma %.3fx) | "
             "max_abs(tma-vs-wgmma)=%.3e\n", ms_t, ms_w, ms_w / ms_t, e);
      cudaFree(d_lse2);
    }
    // ---- O38 A/B（fp16）：LSE 的 K 维 split + 二次归约（TMA 版，D=128/causal）----
    //   split=1 逐位退回 O30；split=2/4/8 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
    //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。
    if (lse_tma) {
      std::vector<float> ref_l((size_t)B * S * H, 0.f);
      float base_ms = 0.f, best = 1e9f;
      int bestk = 1;
      for (int sp = 1; sp <= 8; sp *= 2) {
        auto launch_sp = [&]() {
          if (sp == 1) {
            lse_mma_kernel_bal_tma<128, 1><<<lg_bal, THREADS, kLseSmemTma1>>>(
                qmap_lse, kmap_lse, d_lse, nullptr, S, H, Hkv, scale, 1);
          } else {
            dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * sp));
            lse_mma_kernel_bal_tma<128, 1><<<gsp, THREADS, kLseSmemTma1>>>(
                qmap_lse, kmap_lse, d_lse, d_lse_part, S, H, Hkv, scale, sp);
            const long long nrows = (long long)B * S * H;
            const int th = 256;
            const long long bl = (nrows + th - 1) / th;
            lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, sp);
          }
        };
        for (int i = 0; i < 3; ++i) launch_sp();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) launch_sp();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        t /= iters;
        std::vector<float> got((size_t)B * S * H);
        CUDA_CHECK(cudaMemcpy(got.data(), d_lse, got.size() * 4, cudaMemcpyDeviceToHost));
        double e = 0.0;
        if (sp == 1) {
          ref_l = got;
          base_ms = t;
        } else {
          for (size_t i = 0; i < got.size(); ++i)
            e = std::max(e, (double)std::fabs((double)got[i] - (double)ref_l[i]));
        }
        printf("[O38 A/B] lse split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\n",
               sp, t, base_ms / t, e);
        if (t < best) { best = t; bestk = sp; }
      }
      printf("[O38] best lse split=%d %.4f ms (%.3fx)\n", bestk, best, base_ms / best);
    }
#endif
  }

  // ---- O39 A/B（D=512/MLA/causal）：`lse_mma_kernel_bal<512>` 的 K 维 split + 二次归约 ----
  //   split=1 逐位退回 O8b 原路径；split>1 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
  //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。仅 causal（非 causal 走 O8 原版）。
  if (causal && D == 512) {
    std::vector<float> ref_l((size_t)B * S * H, 0.f);
    float base_ms = 0.f, best = 1e9f;
    int bestk = 1;
    for (int sp = 1; sp <= 16; sp *= 2) {
      auto launch_sp = [&]() {
        if (sp == 1) {
          lse_mma_kernel_bal<512, 1><<<lg_bal, THREADS, kLseSmemBal1>>>(d_q, d_k, d_lse, S, H,
                                                                        Hkv, scale);
        } else {
          dim3 gsp(lg_bal.x, lg_bal.y, (unsigned)(B * sp));
          lse_mma_kernel_bal<512, 1><<<gsp, THREADS, kLseSmemBal1>>>(
              d_q, d_k, d_lse, S, H, Hkv, scale, nullptr, d_lse_part, sp);
          const long long nrows = (long long)B * S * H;
          const int th = 256;
          const long long bl = (nrows + th - 1) / th;
          lse_split_merge_kernel<<<(unsigned)bl, th>>>(d_lse_part, d_lse, nrows, sp);
        }
      };
      for (int i = 0; i < 3; ++i) launch_sp();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_sp();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      t /= iters;
      std::vector<float> got((size_t)B * S * H);
      CUDA_CHECK(cudaMemcpy(got.data(), d_lse, got.size() * 4, cudaMemcpyDeviceToHost));
      double e = 0.0;
      if (sp == 1) { ref_l = got; base_ms = t; }
      else
        for (size_t i = 0; i < got.size(); ++i)
          e = std::max(e, (double)std::fabs((double)got[i] - (double)ref_l[i]));
      printf("[O39 A/B] lse(D=512) split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\n",
             sp, t, base_ms / t, e);
      if (t < best) { best = t; bestk = sp; }
    }
    printf("[O39] best lse split=%d %.4f ms (%.3fx)\n", bestk, best, base_ms / best);
  }

  double main_flops = 4.0 * (double)B * S * H * S * D;
  // ---- O6 A/B（仅 HD=128）：同 session 对比原版（PIPE=0）、K/V 双缓冲（PIPE=1）、只 K 双缓冲（PIPE=2）----
  if (D == 128) {
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
    if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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
  printf("[O6 A/B] main nopipe %.4f ms (%.2f TF) | pipe(KV both) %.4f ms (%.2f TF) | "
         "pipe2(K only) %.4f ms (%.2f TF) => nopipe/pipe %.3fx, nopipe/pipe2 %.3fx\n",
         ms_nopipe, main_flops / (ms_nopipe * 1e-3) / 1e12, ms_pipe,
         main_flops / (ms_pipe * 1e-3) / 1e12, ms_pipe2,
         main_flops / (ms_pipe2 * 1e-3) / 1e12, ms_nopipe / ms_pipe,
         ms_nopipe / ms_pipe2);

  // ---- O48 A/B（D=128，仅 mma 路径）：主 kernel 4-warp（128/2）vs 8-warp（256/4 网格）。
  //      同 session 计时 + 逐元素对拍（只换 warp 网格，数学/数据流不变）。候选 ①（O47）。
  {
    dim3 g48((S + 63) / 64, H, B);
    auto run_d128w = [&](int nth) {
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      if (nth == 256)
        launch_bwd_mma<128, 64, 32, 1, false, true, 256, 4>(
            g48, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, sched);
      else
        launch_bwd_mma<128, 64, 32, 1, false, true>(
            g48, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, sched);
    };
    auto bench_d128w = [&](int nth, float* out) {
      run_d128w(nth);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_d128w(nth);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m4w = 0.f, m8w = 0.f;
    bench_d128w(128, &m4w);
    bench_d128w(256, &m8w);
    std::vector<float> a4_dq(n), a4_dk(nkv), a4_dv(nkv), b8_dq(n), b8_dk(nkv), b8_dv(nkv);
    run_d128w(128);
    CUDA_CHECK(cudaMemcpy(a4_dq.data(), d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dk.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dv.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    run_d128w(256);
    CUDA_CHECK(cudaMemcpy(b8_dq.data(), d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dk.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dv.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    auto md = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O48 A/B] main D=128 4w(128/2) %.4f ms | 8w(256/4) %.4f ms (%.3fx) | "
           "max_abs(8w-vs-4w) dq/dk/dv=%.3e/%.3e/%.3e\n",
           m4w, m8w, m4w / m8w, md(b8_dq, a4_dq), md(b8_dk, a4_dk), md(b8_dv, a4_dv));
  }

  // ---- O16 A/B（仅 FA_WGMMA 构建）：wgmma 主 kernel 分段 wait_group（重叠 epilogue）0 vs 1 ----
#ifdef FA_WGMMA
  if (wgmma_sel && D == 128 && bm_sel == 64 && bn_sel == 64) {
    auto time_wgm = [&](bool ow, float* out_ms, float* dq_c, float* dk_c, float* dv_c) {
      dim3 g((S + 63) / 64, H, B);
      auto launch = [&]() {
        if (ow)
          launch_bwd_wgmma<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                      d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
        else
          launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
      };
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
      CUDA_CHECK(cudaMemcpy(dq_c, d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk_c, d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv_c, d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    };
    std::vector<float> q0(n), k0(nkv), v0(nkv), q1(n), k1(nkv), v1(nkv);
    float ms_ow0 = 0.f, ms_ow1 = 0.f;
    time_wgm(false, &ms_ow0, q0.data(), k0.data(), v0.data());
    time_wgm(true, &ms_ow1, q1.data(), k1.data(), v1.data());
    double dq_d = 0, dk_d = 0, dv_d = 0;
    for (size_t i = 0; i < n; ++i) dq_d = std::max(dq_d, (double)std::fabs(q0[i] - q1[i]));
    for (size_t i = 0; i < nkv; ++i) {
      dk_d = std::max(dk_d, (double)std::fabs(k0[i] - k1[i]));
      dv_d = std::max(dv_d, (double)std::fabs(v0[i] - v1[i]));
    }
    printf("[O16 A/B] main wgmma wait0 %.4f ms | wait_group(重叠) %.4f ms (%.3fx) | "
           "max|diff| dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_ow0, ms_ow1, ms_ow0 / ms_ow1, dq_d, dk_d, dv_d);
  }
#endif

  // ---- O17 A/B（仅 FA_WGMMA 构建，HD=128）：mma 最优档 vs O9b wgmma vs O17 wgmma2(BM=128) ----
#ifdef FA_WGMMA
  if (D == 128) {
    auto time_o17 = [&](int mode, float* out_ms, float* dq_c, float* dk_c, float* dv_c) {
      auto launch = [&]() {
        if (mode == 4) {
          dim3 g((S + 255) / 256, H, B);
          launch_bwd_wgmma4<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 3) {
          dim3 g((S + 255) / 256, H, B);
          launch_bwd_wgmma4<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 2) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 5) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 6) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
        } else if (mode == 7) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                         d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
#if defined(FA_WGMMA) && defined(FA_TMA)
        } else if (mode == 8) {
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2b_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                            d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                            (int)causal);
        } else if (mode == 9) {
          // O35：BN=64 的 wgmma2 走 TMA。
          dim3 g((S + 127) / 128, H, B);
          launch_bwd_wgmma2_tma<128, true>(g, qmap_m, kmap_m, vmap_m, dmap_m, d_delta, d_lse,
                                           d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                           (int)causal);
#endif
        } else if (mode == 1) {
          dim3 g((S + 63) / 64, H, B);
          launch_bwd_wgmma<128, false>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, 0);
        } else {
          launch_cfg(bm_sel, bn_sel, pp_sel, false, true, false);
        }
      };
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
      CUDA_CHECK(cudaMemcpy(dq_c, d_dq_acc, n * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk_c, d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv_c, d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    };
    std::vector<float> q0(n), k0(nkv), v0(nkv), q1(n), k1(nkv), v1(nkv), q2(n), k2(nkv),
        v2(nkv), q3(n), k3(nkv), v3(nkv), q4(n), k4(nkv), v4(nkv), q5(n), k5(nkv), v5(nkv);
    std::vector<float> q6(n), k6(nkv), v6(nkv), q7(n), k7(nkv), v7(nkv);
#if defined(FA_WGMMA) && defined(FA_TMA)
    std::vector<float> q8(n), k8(nkv), v8(nkv);
    std::vector<float> q9(n), k9(nkv), v9(nkv);
    float ms_wg2bt = 0.f, ms_wg2t = 0.f;
#endif
    float ms_mma = 0.f, ms_wgm = 0.f, ms_wg2 = 0.f, ms_wg4 = 0.f, ms_wg4s = 0.f;
    float ms_wg2ns = 0.f, ms_wg2b = 0.f, ms_wg2bns = 0.f;
    time_o17(0, &ms_mma, q0.data(), k0.data(), v0.data());
    time_o17(1, &ms_wgm, q1.data(), k1.data(), v1.data());
    time_o17(2, &ms_wg2, q2.data(), k2.data(), v2.data());
    time_o17(3, &ms_wg4, q3.data(), k3.data(), v3.data());
    time_o17(4, &ms_wg4s, q4.data(), k4.data(), v4.data());
    time_o17(5, &ms_wg2ns, q5.data(), k5.data(), v5.data());
    time_o17(6, &ms_wg2b, q6.data(), k6.data(), v6.data());
    time_o17(7, &ms_wg2bns, q7.data(), k7.data(), v7.data());
#if defined(FA_WGMMA) && defined(FA_TMA)
    time_o17(8, &ms_wg2bt, q8.data(), k8.data(), v8.data());
    time_o17(9, &ms_wg2t, q9.data(), k9.data(), v9.data());
#endif
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0; for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i]-b[i])); return d;
    };
    printf("[O17 A/B] main mma(%d,%d,%d) %.4f ms | O9b wgmma %.4f (%.3fx) | O17 wgmma2(BM128) "
           "%.4f (%.3fx) TF %.1f\n",
           bm_sel, bn_sel, pp_sel, ms_mma, ms_wgm, ms_mma / ms_wgm, ms_wg2, ms_mma / ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12);
    printf("[O17b A/B] main O17(BM128) %.4f ms | O17b wgmma4(BM256,ovlp) %.4f ms (%.3fx) "
           "| wgmma4(BM256,seq) %.4f ms (%.3fx) TF %.1f\n",
           ms_wg2, ms_wg4, ms_wg2 / ms_wg4, ms_wg4s, ms_wg2 / ms_wg4s,
           main_flops / (ms_wg4s * 1e-3) / 1e12);
    printf("[O17 A/B] max|diff| wg2-vs-mma dq/dk/dv=%.2e/%.2e/%.2e | wg2-vs-O9b "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q2, q0), mad(k2, k0), mad(v2, v0), mad(q2, q1), mad(k2, k1), mad(v2, v1));
    printf("[O18 A/B] main O17 wg2(BN64) %.4f ms (%.1f TF) | O18 wg2b(BN128,split) %.4f ms "
           "(%.1f TF) => %.3fx | O18 wg2b(BN128,串行) %.4f ms => %.3fx | "
           "max|diff| wg2b-vs-wg2 dq/dk/dv=%.2e/%.2e/%.2e | wg2bns-vs-wg2b %.2e/%.2e/%.2e\n",
           ms_wg2, main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2b,
           main_flops / (ms_wg2b * 1e-3) / 1e12, ms_wg2 / ms_wg2b, ms_wg2bns,
           ms_wg2 / ms_wg2bns, mad(q6, q2), mad(k6, k2), mad(v6, v2), mad(q7, q6), mad(k7, k6),
           mad(v7, v6));
#if defined(FA_WGMMA) && defined(FA_TMA)
    printf("[O33 A/B] main wg2b(cp.async) %.4f ms (%.1f TF) | wg2b+TMA %.4f ms (%.1f TF) => "
           "%.3fx | max|diff| tma-vs-wg2b dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2b, main_flops / (ms_wg2b * 1e-3) / 1e12, ms_wg2bt,
           main_flops / (ms_wg2bt * 1e-3) / 1e12, ms_wg2b / ms_wg2bt, mad(q8, q6), mad(k8, k6),
           mad(v8, v6));
    printf("[O35 A/B] main wg2(BN64,cp.async) %.4f ms (%.1f TF) | wg2(BN64)+TMA %.4f ms "
           "(%.1f TF) => %.3fx | max|diff| tma-vs-wg2 dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2, main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2t,
           main_flops / (ms_wg2t * 1e-3) / 1e12, ms_wg2 / ms_wg2t, mad(q9, q2), mad(k9, k2),
           mad(v9, v2));
#endif
    printf("[O17-2 A/B] main O17 wg2 wg0串行(dV+dK) %.4f ms (%.1f TF) | O17-2 拆分(wg0=dV,"
           "wg1=dK) %.4f ms (%.1f TF) => %.3fx | max|diff| dq/dk/dv=%.2e/%.2e/%.2e\n",
           ms_wg2ns, main_flops / (ms_wg2ns * 1e-3) / 1e12, ms_wg2,
           main_flops / (ms_wg2 * 1e-3) / 1e12, ms_wg2ns / ms_wg2, mad(q2, q5), mad(k2, k5),
           mad(v2, v5));
    printf("[O17b A/B] max|diff| wg4-vs-wg2 dq/dk/dv=%.2e/%.2e/%.2e | wg4-vs-mma "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q3, q2), mad(k3, k2), mad(v3, v2), mad(q3, q0), mad(k3, k0), mad(v3, v0));
    printf("[O17b A/B] max|diff| wg4seq-vs-wg4 dq/dk/dv=%.2e/%.2e/%.2e | wg4seq-vs-wg2 "
           "dq/dk/dv=%.2e/%.2e/%.2e\n",
           mad(q4, q3), mad(k4, k3), mad(v4, v3), mad(q4, q2), mad(k4, k2), mad(v4, v2));
  }
#endif

  // ---- O6c A/B（仅 causal，仍在 HD=128 块内）：mblk 重排 sched=0/1/2（同一 pipe）----
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
  }  // end if (D == 128)

  // ---- O5c A/B（仅 HD=128 且 causal）：不同 (BM,BN,PIPE) tile 配置 ----
  if (D == 128 && causal) {
    auto time_cfg = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true, false); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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

  // ---- O7c A/B（仅 HD=128 且 causal）：在 4 个几何上对比 {float2/float4} × {无/有 LSE-D 预装} ----
  if (D == 128 && causal) {
    auto time_r4 = [&](int bm, int bn, int pp, bool r4, bool prel, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, r4, prel, false); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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
    const int cfgs[4][3] = {{64, 32, 2}, {64, 64, 2}, {32, 32, 1}, {32, 32, 2}};
    printf("[O7c A/B] main-only: base(f2,noPREL) | f4 | f2+PREL | f4+PREL, main-only:\n");
    for (int ci = 0; ci < 4; ++ci) {
      const int bm = cfgs[ci][0], bn = cfgs[ci][1], pp = cfgs[ci][2];
      float a = 0.f, b = 0.f, c = 0.f, e = 0.f;
      time_r4(bm, bn, pp, false, false, &a);
      time_r4(bm, bn, pp, true, false, &b);
      time_r4(bm, bn, pp, false, true, &c);
      time_r4(bm, bn, pp, true, true, &e);
      auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
      printf("  (BM=%d,BN=%d,PIPE=%d): base %.4f (%.2f) | f4 %.4f (%.2f,%+.1f%%) | "
             "f2+PREL %.4f (%.2f,%+.1f%%) | f4+PREL %.4f (%.2f,%+.1f%%)\n",
             bm, bn, pp, a, tf(a), b, tf(b), (a / b - 1) * 100, c, tf(c), (a / c - 1) * 100, e,
             tf(e), (a / e - 1) * 100);
    }
  }

  // ---- MLA（HD=512）配置 A/B：BM=32/64 × PIPE=0/1，看哪种并行度/smem 组合最快 ----
  if (D == 512 && causal) {
    auto time_cfg2 = [&](int bm, int bn, int pp, float* out_ms) {
      auto launch = [&]() { launch_cfg(bm, bn, pp, false, true, false); };
      if (D == 512 || wg2_ksplit_eff > 1) CUDA_CHECK(cudaMemset(d_dq_acc, 0, n * sizeof(float)));
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
    float a = 0.f, b = 0.f, c = 0.f;
    time_cfg2(32, 32, 0, &a);
    time_cfg2(32, 32, 1, &b);
    time_cfg2(64, 32, 0, &c);
    auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
    printf("[MLA512 A/B] main: (32,32,0) %.4f (%.2f TF) | (32,32,1) %.4f (%.2f) | "
           "(64,32,0) %.4f (%.2f) => best %.3fx\n",
           a, tf(a), b, tf(b), c, tf(c), a / std::min(a, std::min(b, c)));
  }

  // ---- O44 A/B：MLA（D=512）主 kernel 的 N 方向 split-K（同 binary、同 session）----
  if (D == 512 && causal) {
    auto time_ks = [&](int ks, float* out_ms) {
      dim3 g((S + 31) / 32, H, B);
      g.x *= (unsigned)ks;
      auto launch = [&]() {
        launch_bwd_mma<512, 32, 32, 1, false, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                    scale, (int)causal, 0, nullptr, ks);
      };
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
    float k1 = 0.f, k2 = 0.f, k4 = 0.f;
    time_ks(1, &k1);
    time_ks(2, &k2);
    time_ks(4, &k4);
    auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
    printf("[O44 A/B] MLA main ksplit: 1 %.4f (%.2f) | 2 %.4f (%.2f, %.2fx) | "
           "4 %.4f (%.2f, %.2fx)\n",
           k1, tf(k1), k2, tf(k2), k1 / k2, k4, tf(k4), k1 / k4);
  }

  // ---- O46 A/B：MLA（D=512）主 kernel 的 warp 几何（4-warp 2×2 vs 8-warp 2×4，同 binary/session）----
  if (D == 512 && causal) {
    auto time_w = [&](bool w8, float* out_ms) {
      dim3 g((S + 31) / 32, H, B);
      g.x *= (unsigned)mlaksplit_eff;
      auto launch = [&]() {
        if (w8)
          launch_bwd_mma<512, 32, 32, 1, false, true, 256, 4>(
              g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
              Hkv, scale, (int)causal, 0, nullptr, mlaksplit_eff);
        else
          launch_bwd_mma<512, 32, 32, 1, false, true>(
              g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
              Hkv, scale, (int)causal, 0, nullptr, mlaksplit_eff);
      };
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
    float w4 = 0.f, w8 = 0.f;
    time_w(false, &w4);
    time_w(true, &w8);
    auto tf = [&](float ms) { return main_flops / (ms * 1e-3) / 1e12; };
    printf("[O46 A/B] MLA main warp: 4w %.4f (%.2f) | 8w %.4f (%.2f, %.3fx)\n",
           w4, tf(w4), w8, tf(w8), w4 / w8);
  }

  // ---- O7b A/B（仅 FA_WGMMA 构建、HD=128，`--det` 开启）：跨 CTA `atomicAdd` vs
  //      确定性 partial 缓冲 + 二次归约。跑两遍 DET 验证逐位可复现，再与 atomic 比 max|diff|。
#ifdef FA_WGMMA
  if (det_ab && D == 128) {
    const int nblk = (S + 127) / 128;
    const size_t part_elems = (size_t)B * H * nblk * S * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    dim3 g((S + 127) / 128, H, B);
    auto run_atomic = [&]() {
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * sizeof(float)));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * sizeof(float)));
      launch_bwd_wgmma2b<128, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal);
    };
    auto run_det = [&]() {
      launch_bwd_wgmma2b<128, true, true>(g, d_q, d_k, d_v, d_do, d_delta, d_lse, d_dq_acc,
                                          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                                          d_dk_part, d_dv_part, nblk);
      dim3 rg(B * Hkv, S);
      dkv_reduce_kernel<128><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H, Hkv,
                                          nblk, (int)causal);
    };
    auto time_fn = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_det = 0.f;
    time_fn(run_atomic, &ms_at);
    std::vector<float> ak(nkv), av(nkv);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    time_fn(run_det, &ms_det);
    std::vector<float> dk1(nkv), dv1(nkv);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    run_det();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> dk2(nkv), dv2(nkv);
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * sizeof(float), cudaMemcpyDeviceToHost));
    auto mad = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0;
      for (size_t i = 0; i < a.size(); ++i) d = std::max(d, (double)std::fabs(a[i] - b[i]));
      return d;
    };
    printf("[O7b A/B] main wgmma2b atomic %.4f ms (%.1f TF) | DET(partial+reduce) %.4f ms (%.3fx) "
           "| runs[1-2] bitwise-diff dk/dv=%.2e/%.2e | DET-vs-atomic dk/dv=%.2e/%.2e\n",
           ms_at, main_flops / (ms_at * 1e-3) / 1e12, ms_det, ms_at / ms_det, mad(dk1, dk2),
           mad(dv1, dv2), mad(dk1, ak), mad(dv1, av));
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
  }
#endif

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
