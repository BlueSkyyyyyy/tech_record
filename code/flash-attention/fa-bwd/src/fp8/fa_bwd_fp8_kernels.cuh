// =============================================================================
// fa_bwd_fp8_kernels.cuh —— FlashAttention 反向（FP8）**两文件版的 device 部分**
// =============================================================================
// 由 P3-4 单文件 `fa_bwd_fp8_mma_onefile.cu` 拆分而来（P3-5），device 代码与单文件
// **逐字一致**（preprocess 与 golden 相同，main 为张量核 mma 版）：
//   * quantize_row_kernel：fp32 [rows][D] -> fp8 + rowwise scale（P5-3 起支持 D>128）
//   * lse_mma_kernel（O1）：mma 分块 Q·Kᵀ + online-softmax 求 LSE
//   * delta_kernel：反量化 dO 与 fp32 O 逐行点积求 D=rowsum(dO∘O)
//   * fa_bwd_fp8_mma_kernel：1colblock 反向主 kernel，5 个 GEMM 全部 mma.m16n8k32
//   * convert_kernel：fp32 累加缓冲 -> fp32 输出
// host 侧（npy 读取 / launcher / 自测）见 `fa_bwd_fp8_main.cu`。
//
// P5-3（GQA/MQA）：第 h 个 Q 头映射到 KV 头 hkv=h/(H/Hkv)（与 ref 的 repeat_interleave
// 对齐）。Q/dO/dQ 按 [B,S,H,D] 索引，K/V/dK/dV 按 [B,S,Hkv,D] 索引；Hkv==H 时逐式退化为 MHA。
//
// P5-3（MLA head_dim=512）：与 fp16/bf16 版同样的思路，把 head_dim 变成模板参数 `HD`。
//   * **GEMM1/GEMM2 的 HD 是归约维**（K_TILE=HD）→ 只是把 k-loop 从 4 次变 16 次；
//   * **GEMM3/4/5 的 HD 是输出 N 维**（dOᵀ/Qᵀ/Kᵀ 的行）→ 原实现硬编码「2 warp × 64 = 128
//     列刚好铺满 HD=128」，HD>128 时必须再加一层 **N-tile 循环**（每遍 128 列，共 HD/128 遍），
//     B 操作数按 `d0*stride` 偏移、写回列号加 `d0`。
//   * `HD=128,BM=64,BN=32` 时 N-tile 循环只跑 1 遍，**与 P3-5/O1–O4a 逐位相同**。
//   * `HD=512` 选 `BM=64,BN=32`：smem=213.5KB（1 CTA/SM）；寄存器预取（O3）每线程要
//     `BN*HD/4/THREADS` 个 uint32，HD=512 时 64 个会爆寄存器，故只在 `KVU<=8` 时启用。
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
// =============================================================================

#ifndef FA_BWD_FP8_KERNELS_CUH_
#define FA_BWD_FP8_KERNELS_CUH_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

// ----------------------------- 编译期常量 -----------------------------
static constexpr int THREADS = 128;
static constexpr int WN = 2;

static constexpr float kE4M3Max = 448.0f;
static constexpr float kE5M2Max = 57344.0f;

// O1：preprocess 的 LSE 改用 mma 分块（见下），独立的 tile 常量。
// 每个 CTA 负责 LBM 行，4 个 warp 各 16 行（wm=wid），沿 N 一次 LBN 列。
static constexpr int LBM = 64;
static constexpr int LBN = 64;

// =============================================================================
// Fp8Cfg<HD,BM,BN>：把原来的一组全局常量按 head_dim/tile 参数化。
//   * ASLD  = HD + 16      （Qs/Ks/Vs/dOs 行距，A 的行是 head_dim）
//   * KTS/QTS/DSS2 = 32/64/32 + 16（转置副本与 dS2 的行距，只与 BN/BM 有关）
//   * smem_bytes：主 kernel 动态 smem（含 O2 折叠后的布局）
//   * lse_smem_bytes：LSE kernel 动态 smem（只与 HD 有关）
//   * use_prefetch：O3 寄存器预取只在每线程预取量小时启用（HD=128 时 KVU=8）。
// =============================================================================
template <int HD, int BM_, int BN_>
struct Fp8Cfg {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int ASLD = HD + 16;   // 144（HD=128）/ 528（HD=512）
  static constexpr int KTS = BN + 16;    // 48，[d][j] 转置 K（K=BN）
  static constexpr int QTS = BM + 16;    // 80，[d][m] / [j][m]（K=BM）
  static constexpr int DSS2 = BN + 16;   // 48，dS2 [m][j]（K=BN）
  static constexpr int KVU = BN * HD / 4 / THREADS;  // 每线程预取的 uint32 数
  static constexpr int kNScale = 3 * BM + 4 * BN;    // qs,dos,sds2 (BM) + ks,vs,sA,sds3 (BN)

  // O4d：P/S 两个 fp32 [BM][BN] 缓冲的行距 padding。行距 = BN = 32 word（128B）时，
  //   * fold 里按「列」读 `P[m][j]`（j 固定、m 步进 16）会全部落同一 bank（32|stride）→ 4-way；
  //   * GEMM1/2 epilogue 里 8 行 × 4 列同时写 `P[r][c]`，bank=c，同列不同行全撞 → 8-way。
  //   +1 word（33）让 bank 与行号线性相关。实测（S=4096 ksplit=1）：`op_ld` 冲突
  //   199.9M→77.3M（−61%）、`op_st` 231.6M→198.4M（−14%）、总多余 wavefronts
  //   613.6M→456.0M（−26%），main 6.56→5.85 ms（1.12×）。奇数才能保证
  //   m 步进 16 时 `16*33 mod 32 = 16 != 0`（+4/+8 的偶数 padding 无效）。
  static constexpr int PSS = BN + 1;     // P/S fp32 行距（=33）

  // O2：dS3（[BN][QTS]）折进 Ks、Ap（[BN][QTS]）折进 Vs。
  static constexpr int fp8_bytes = BM * ASLD   // Qs
                                 + BN * ASLD   // Ks（dS3 复用尾部）
                                 + BN * ASLD   // Vs（Ap 复用尾部）
                                 + BM * ASLD   // dOs
                                 + HD * KTS    // Kt
                                 + HD * QTS    // Qt
                                 + HD * QTS    // dOt
                                 + BM * DSS2;  // dS2
  static constexpr int smem_bytes =
      fp8_bytes + (kNScale + 2 * BM * PSS) * (int)sizeof(float);

  static constexpr int lse_smem_bytes =
      LBM * ASLD + LBN * ASLD + (LBM + LBN) * (int)sizeof(float);

  // O3 寄存器预取：pk/pv 各 KVU 个 uint32。HD=128 时 KVU=8（16 regs，可行）；
  // HD=512 时 KVU=32（64 regs，会挤掉累加器/地址寄存器）→ 关闭，走直接向量化读。
  static constexpr bool use_prefetch = (KVU * 2 <= 16);
};

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

// ----------------------------- O4c：向量化归约（red） -----------------------------
// 反向的 dQ/dK/dV 都靠跨 CTA 的 fp32 `atomicAdd` 汇总（ncu：这些 red 占 L2 扇区的 92%，
// 是 O2b+O4d 后的头号墙）。mma.m16n8 累加器里 q=0/1 两列相邻、q=2/3 两列相邻，且同一
// q 对内**行号相同**（row = r0+g 或 r0+g+8）⇒ 该对共用同一个行 scale（sA/sds3/sds2）。
// 于是把两个标量 `atomicAdd` 打包成一次 `atomicAdd(float2*)`（sm_90 支持），
// **red 请求数与 L2 扇区数各减半**，数值等价（硬件对 v2 的两个 f32 仍各自原子累加）。
__device__ __forceinline__ void red_add2(float* p, float a, float b) {
  atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));
}

// A[M_TILE][K_TILE]、B[N_TILE][K_TILE] 均行主序（行距 asld/bsld，含 padding）。
// 每 warp 负责 WARP_M×WARP_N 输出块；各 warp 旧 (wm,wn) 由调用方给出。
template <int WARP_M, int WARP_N, int K_TILE, int KIND>
__device__ __forceinline__ void mma_block(const unsigned char* As, int asld,
                                          const unsigned char* Bs, int bsld,
                                          float acc[WARP_M / 16][WARP_N / 8][4],
                                          int wm, int wn, int lane) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
// HD=128 时 K_TILE/32<=4，`unroll 4` 仍是全展开（逐位不变）；HD=512 的 GEMM1/2 有 16 个
// k-步，全展开会把寄存器顶到 255 并 spill，限成 4 让 ptxas 少保活一些操作数。
#pragma unroll 4
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

// ----------------------------- O3：K/V 寄存器预取流水 -----------------------------
// fp8 main kernel 的 per-tile 全局读（K/V）原本是「载入→同步→算」，ncu 显示
// No Eligible ~80%、long_scoreboard 为主（全局延迟无处可躲）。这里**不加 smem**（加
// 双缓冲会把 3 CTA/SM 压回 2 CTA）而是用**寄存器双缓冲**：
//   * 每个线程按下轮 tile 的地址 (`tid + e*THREADS`)*4 预取 KVU 个 uint32（=4 个连续 fp8，
//     行内对齐，故可一次 4B 读），结果留在 KVU 个寄存器里；
//   * 预取发在本轮 5 个 GEMM 之前，落盘在本轮末尾（GEMM 全部读完 smem 之后），
//     于是全局/L2 延迟被一整轮计算覆盖，且不占额外 smem、保持 3 CTA/SM。
//   * 仅当 `KVU*2 <= 16`（HD=128）时启用；HD=512 走直接向量化读（kv_load_direct）。
template <int KVU, int HD>
__device__ __forceinline__ void kv_prefetch(const unsigned char* __restrict__ k8,
                                            const unsigned char* __restrict__ v8,
                                            int j0, int S, int Hkv, int hkv, int b, int tid,
                                            uint32_t* pk, uint32_t* pv) {
#pragma unroll
  for (int e = 0; e < KVU; ++e) {
    int u = tid + e * THREADS;
    int i = u * 4, r = i / HD, d = i % HD;
    int jg = j0 + r;
    uint32_t kv = 0, vv = 0;
    if (jg < S) {  // 越界写 0 字节（等价 cvt_e4m3(0)=0x00）
      size_t idx = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d;
      kv = *reinterpret_cast<const uint32_t*>(k8 + idx);
      vv = *reinterpret_cast<const uint32_t*>(v8 + idx);
    }
    pk[e] = kv;
    pv[e] = vv;
  }
}

template <int KVU, int HD>
__device__ __forceinline__ void kv_commit(unsigned char* Ks, unsigned char* Vs,
                                          unsigned char* Kt, const uint32_t* pk,
                                          const uint32_t* pv, int tid, int asld, int kts) {
#pragma unroll
  for (int e = 0; e < KVU; ++e) {
    int u = tid + e * THREADS;
    int i = u * 4, r = i / HD, d = i % HD;
    *reinterpret_cast<uint32_t*>(Ks + r * asld + d) = pk[e];
    *reinterpret_cast<uint32_t*>(Vs + r * asld + d) = pv[e];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk)  // 转置副本 Kt[d][r]（供 GEMM5 的 B 操作数）
      Kt[(d + kk) * kts + r] = (pk[e] >> (8 * kk)) & 0xff;
  }
}

// HD>128 时的直接向量化载入（无寄存器预取）：4B/线程，K 同时写 Kt 转置副本。
template <int HD, int BN>
__device__ __forceinline__ void kv_load_direct(const unsigned char* __restrict__ k8,
                                               const unsigned char* __restrict__ v8,
                                               int j0, int S, int Hkv, int hkv, int b,
                                               int tid, unsigned char* Ks, unsigned char* Vs,
                                               unsigned char* Kt, int asld, int kts) {
  for (int i = tid * 4; i < BN * HD; i += THREADS * 4) {
    int r = i / HD, d = i % HD;
    int jg = j0 + r;
    uint32_t kv = 0, vv = 0;
    if (jg < S) {
      size_t idx = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d;
      kv = *reinterpret_cast<const uint32_t*>(k8 + idx);
      vv = *reinterpret_cast<const uint32_t*>(v8 + idx);
    }
    *reinterpret_cast<uint32_t*>(Ks + r * asld + d) = kv;
    *reinterpret_cast<uint32_t*>(Vs + r * asld + d) = vv;
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) Kt[(d + kk) * kts + r] = (kv >> (8 * kk)) & 0xff;
  }
}

// =============================================================================
// 1) quantize_row_kernel（与 golden 相同）：fp32 [rows][D] -> fp8 + rowwise scale
// =============================================================================
// P5-3：D 可以 >128（MLA head_dim=512），故 amax 与量化都改成线程内 grid-stride；
// D<=128 时每线程恰好一个元素，与原实现（`sh[d]=|v|`）逐位一致。
__global__ void quantize_row_kernel(const float* __restrict__ x,
                                    unsigned char* __restrict__ xq,
                                    float* __restrict__ scale, int D, int is_e5m2) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  const float* xr = x + (size_t)row * D;
  __shared__ float sh[128];
  float loc = 0.f;
  for (int i = d; i < D; i += blockDim.x) loc = fmaxf(loc, fabsf(xr[i]));
  sh[d] = loc;
  __syncthreads();
  for (int off = 64; off > 0; off >>= 1) {
    if (d < off) sh[d] = fmaxf(sh[d], sh[d + off]);
    __syncthreads();
  }
  const float fp8_max = is_e5m2 ? kE5M2Max : kE4M3Max;
  const float s = (sh[0] > 0.f) ? (sh[0] / fp8_max) : 1.f;
  if (d == 0) scale[row] = s;
  for (int i = d; i < D; i += blockDim.x)
    xq[(size_t)row * D + i] = is_e5m2 ? cvt_e5m2(xr[i] / s) : cvt_e4m3(xr[i] / s);
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
template <int HD>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
               const unsigned char* __restrict__ k8, const float* __restrict__ ks,
               float* __restrict__ lse, int S, int H, int Hkv, float scale, int causal) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int ASLD = Cfg::ASLD;
  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks = Qs + LBM * ASLD;
  float* qs_s = reinterpret_cast<float*>(Ks + LBN * ASLD);
  float* ks_s = qs_s + LBM;

  const int mblk = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  // P5-3：GQA/MQA——第 h 个 Q 头映射到 KV 头 h/(H/Hkv)（与 ref 的 repeat_interleave 对齐）。
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * LBM;

  // 载入 Q 块（rowwise scale 同步取回）
  for (int i = tid; i < LBM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    Qs[r * ASLD + d] =
        (qi < S) ? q8[(((size_t)(b * S + qi)) * H + h) * HD + d] : cvt_e4m3(0.f);
  }
  if (tid < LBM)
    qs_s[tid] = (m0 + tid < S) ? qs[((size_t)(b * S + m0 + tid)) * H + h] : 1.f;
  __syncthreads();

  const int ncols = causal ? min(S, m0 + LBM) : S;
  const int ntiles = (ncols + LBN - 1) / LBN;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * LBN;
    for (int i = tid; i < LBN * HD; i += THREADS) {
      int r = i / HD, d = i % HD;
      int jg = j0 + r;
      Ks[r * ASLD + d] =
          (jg < S) ? k8[(((size_t)(b * S + jg)) * Hkv + hkv) * HD + d] : cvt_e4m3(0.f);
    }
    if (tid < LBN)
      ks_s[tid] = (j0 + tid < S) ? ks[((size_t)(b * S + j0 + tid)) * Hkv + hkv] : 1.f;
    __syncthreads();

    // S_tile = Q·Kᵀ（e4m3×e4m3→fp32），每 warp 16×64
    float acc[1][8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
    mma_block<16, LBN, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wid, 0, lane);

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
// 2b) delta_kernel：D = rowsum(dO ∘ O)（反量化 e5m2·dos 后与 fp32 O 点积）
// =============================================================================
// 纯粹 O(S·H·D) 的逐行归约，与 LSE 解耦（原来和 LSE 挤在同一 kernel 里）。
template <int HD>
__global__ void delta_kernel(const float* __restrict__ o,
                             const unsigned char* __restrict__ do8,
                             const float* __restrict__ dos, float* __restrict__ delta,
                             int S, int H) {
  const int s = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int tid = threadIdx.x;
  const size_t row = ((size_t)(b * S + s)) * H + h;
  const float* orow = o + row * HD;
  const unsigned char* dorow = do8 + row * HD;
  const float do_scale = dos[row];
  float dp = 0.f;
  for (int d = tid; d < HD; d += blockDim.x)
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
template <int HD, int BM, int BN>
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
                      float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                      int causal, int ksplit) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int KTS = Cfg::KTS;
  constexpr int QTS = Cfg::QTS;
  constexpr int DSS2 = Cfg::DSS2;
  constexpr int KVU = Cfg::KVU;
  constexpr int kNScale = Cfg::kNScale;
  constexpr int PSS = Cfg::PSS;
  constexpr int kFp8Bytes = Cfg::fp8_bytes;
  // GEMM3/4/5 的输出 N 维 = head_dim；每遍处理 NTW = WN*64 = 128 列，共 HD/NTW 遍。
  constexpr int NTW = WN * 64;
  static_assert(HD % NTW == 0, "HD 必须是 128 的整数倍");

  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks  = Qs + BM * ASLD;
  unsigned char* Vs  = Ks + BN * ASLD;
  unsigned char* dOs = Vs + BN * ASLD;
  unsigned char* Kt  = dOs + BM * ASLD;
  unsigned char* Qt  = Kt + HD * KTS;
  unsigned char* dOt = Qt + HD * QTS;
  unsigned char* dS2 = dOt + HD * QTS;
  unsigned char* Ap  = Vs;                   // O2：复用 GEMM2 后死亡的 Vs
  unsigned char* dS3 = Ks;                   // O2：复用 GEMM1 后死亡的 Ks
  float* scales = reinterpret_cast<float*>(smem + kFp8Bytes);
  float* Ps = scales + kNScale;              // P fp32 [BM][PSS]
  float* Ss = Ps + BM * PSS;                 // dS fp32 [BM][PSS]
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;

  // ---- O2b：N 方向切块（split-K）。同一 (mblk,h,b) 的 K/V 列块 [0,ntiles) 被均分给
  //      ksplit 个 CTA；各自只算自己那一段，dQ/dK/dV 仍用跨 CTA 的 fp32 atomicAdd 汇总。
  //      小 S 时把 grid 从 S/BM×H 抬到 ksplit 倍，消「grid 不足一整个波」的空 SM；
  //      大 S 时用来削尾波（partial wave）。各部分数学上仍是同一个和，只是 fp 加法次序略变。----
  const int mblk = blockIdx.x / ksplit, part = blockIdx.x % ksplit;
  const int h = blockIdx.y, b = blockIdx.z;
  // P5-3：GQA/MQA——Q 头 h 对应 KV 头 hkv；Q/dO/dQ 用 H，K/V/dK/dV 用 Hkv。
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / WN, wc = wid % WN;
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ksplit;
  const int nt_end = (part + 1) * ntiles / ksplit;
  if (nt_end <= nt_begin) return;  // 该 part 无 tile（causal 下小 mblk 可能被切空）

  // ---- 载入 Q/dO（含转置副本 Qt/dOt，供 dK/dV 的 B 操作数）----
  for (int i = tid; i < BM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    unsigned char qv = cvt_e4m3(0.f), ov = cvt_e5m2(0.f);
    if (qi < S) {
      size_t idx = (((size_t)(b * S + qi)) * H + h) * HD + d;
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

  // ---- O3 prologue：寄存器预取本 part 首个 tile 并落盘；HD>128 时直接向量化读。----
  // ---- O4a：Q/dO 载入与 tile 的 K/V 落盘写的是互不重叠的 smem（Qs/Qt/dOs/dOt vs
  //            Ks/Vs/Kt），故把原来 prologue 的两处 __syncthreads 合并为一处。----
  uint32_t pk[KVU], pv[KVU];
  if (Cfg::use_prefetch) {
    kv_prefetch<KVU, HD>(k8, v8, nt_begin * BN, S, Hkv, hkv, b, tid, pk, pv);
    kv_commit<KVU, HD>(Ks, Vs, Kt, pk, pv, tid, ASLD, KTS);
  } else {
    kv_load_direct<HD, BN>(k8, v8, nt_begin * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kt, ASLD,
                           KTS);
  }
  if (tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
  }
  __syncthreads();

  for (int nt = nt_begin; nt < nt_end; ++nt) {
    const int j0 = nt * BN;
    // ---- O3：预取下一 tile 的 K/V 到寄存器（延迟被本轮 5 个 GEMM 覆盖）----
    const int nnt = nt + 1;
    if (Cfg::use_prefetch && nnt < nt_end)
      kv_prefetch<KVU, HD>(k8, v8, nnt * BN, S, Hkv, hkv, b, tid, pk, pv);

    // ---- (1) S = scale·QKᵀ  →  P = exp(S − LSE)，存 fp32 ----
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 16, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wr, wc, lane);
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
            Ps[r * PSS + c] = p;
          }
    }
    // ---- O4a：GEMM1 与 GEMM2 之间**不需要** barrier。GEMM2 只读 dOs/Vs（本 tile 前已
    //      就绪），其 epilogue 读回的 Ps[r*PSS+c] 正是**本线程**刚写入的同一地址（两次
    //      mma_block 的 (wm=wr, wn=wc) 与累加器映射完全一致），无线程间依赖。----

    // ---- (2) dP = dO·Vᵀ  →  dS = P∘(dP − D)，存 fp32 ----
    {
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 16, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc, wr, wc, lane);
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
            Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);
          }
    }
    __syncthreads();

    // ---- 构造 dV/dK/dQ 的 fp8 操作数（fold 归约维上的 rowwise scale）----
    // O4a：原来 fold 只让 `tid<BN`（32 线程）算 Ap/dS3、`tid<BM`（64 线程）算 dS2，其余
    // 线程空等；现改为**全部 128 线程均衡分工**：warp 内 4 lane 一组做一行 amax 的
    // `__shfl_xor_sync` 归约。fmaxf 可交换结合 ⇒ amax 结果与原顺序**逐位相同**，除法/量化
    // 也逐元素一致，故数值仍与 O3 逐位相同；但这一段的墙钟缩短约 4×（Amax/写入都并行）。
    {
      // Ap[j][m]=P[m][j]*dos[m] (e4m3, per-j) 与 dS3[j][m]=dS[m][j]*qs[m] (e5m2, per-j)
      const int jl = lane >> 2, sub4 = lane & 3;  // 每 warp 8 行 × 4 lane 分工
      const int j = wid * 8 + jl;
      float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        int m = sub4 * 16 + t;
        amaxA = fmaxf(amaxA, fabsf(Ps[m * PSS + j] * dos_s[m]));
        amax3 = fmaxf(amax3, fabsf(Ss[m * PSS + j] * qs_s[m]));
      }
      amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 1));
      amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 2));
      amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 1));
      amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 2));
      float scA = (amaxA > 0.f) ? amaxA / kE4M3Max : 1.f;
      float sc3 = (amax3 > 0.f) ? amax3 / kE5M2Max : 1.f;
      if (sub4 == 0) {
        sA[j] = scA;
        sds3[j] = sc3;
      }
      scA = __shfl_sync(0xffffffffu, scA, jl * 4);
      sc3 = __shfl_sync(0xffffffffu, sc3, jl * 4);
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        int m = sub4 * 16 + t;
        Ap[j * QTS + m] = cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA);
        dS3[j * QTS + m] = cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3);
      }
    }
    {
      // dS2[m][j] = dS[m][j]*ks[j] (e5m2, per-m)：每 warp 16 行 × 2 lane 分工
      const int ml = lane >> 1, sub2 = lane & 1;
      const int m = wid * 16 + ml;
      float amax2 = 0.f;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        int j = sub2 * 16 + t;
        amax2 = fmaxf(amax2, fabsf(Ss[m * PSS + j] * ks_s[j]));
      }
      amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
      float sc2 = (amax2 > 0.f) ? amax2 / kE5M2Max : 1.f;
      if (sub2 == 0) sds2[m] = sc2;
      sc2 = __shfl_sync(0xffffffffu, sc2, ml * 2);
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        int j = sub2 * 16 + t;
        dS2[m * DSS2 + j] = cvt_e5m2(Ss[m * PSS + j] * ks_s[j] / sc2);
      }
    }
    __syncthreads();

    // ---- (3)(4)(5) 沿 head_dim 的 N-tile 循环。GEMM3/4/5 的输出宽度是 HD，
    //      每遍覆盖 NTW=128 列：B 操作数（dOt/Qt/Kt）按 d0*stride 偏移、写回列加 d0。
    //      HD=128 时只跑 1 遍（与 O4a 逐位相同）。----
#pragma unroll
    for (int nd = 0; nd < HD / NTW; ++nd) {
      const int d0 = nd * NTW;

      // ---- (3) dV = Pᵀ·dO  : A=Ap[j][m] (e4m3), B=dOt[d0+..][m] 原始 do8 (e5m2) ----
      {
        float acc[1][8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
        mma_block<16, 64, BM, E4E5>(Ap, QTS, dOt + d0 * QTS, QTS, acc, wr, wc, lane);
        const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
            int r = r0 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2;
            int jg = j0 + r;
            if (jg < S)
              red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d0 + c,
                       acc[0][j][q] * sA[r], acc[0][j][q + 1] * sA[r]);
          }
      }

      // ---- (4) dK = scale·dSᵀ·Q : A=dS3[j][m] (e5m2), B=Qt[d0+..][m] 原始 q8 (e4m3) ----
      {
        float acc[1][8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
        mma_block<16, 64, BM, E5E4>(dS3, QTS, Qt + d0 * QTS, QTS, acc, wr, wc, lane);
        const int r0 = wr * 16, c0 = wc * 64;
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
            int r = r0 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2;
            int jg = j0 + r;
            if (jg < S)
              red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d0 + c,
                       acc[0][j][q] * sds3[r] * scale,
                       acc[0][j][q + 1] * sds3[r] * scale);
          }
      }

      // ---- (5) dQ += scale·dS·K : A=dS2[m][j] (e5m2), B=Kt[d0+..][j] 原始 k8 (e4m3) ----
      {
        float acc[2][8][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block<32, 64, BN, E5E4>(dS2, DSS2, Kt + d0 * KTS, KTS, acc, wr, wc, lane);
        const int r0 = wr * 32, c0 = wc * 64;
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2;
              int qi = m0 + r;
              if (qi < S)
                red_add2(dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + d0 + c,
                         acc[i][j][q] * sds2[r] * scale,
                         acc[i][j][q + 1] * sds2[r] * scale);
            }
      }
    }
    __syncthreads();
    // ---- O3：落盘预取的下一 tile 的 K/V（本轮 GEMM 已全部读完 smem），并更新 ks/vs ----
    if (nt + 1 < nt_end) {
      if (Cfg::use_prefetch) {
        kv_commit<KVU, HD>(Ks, Vs, Kt, pk, pv, tid, ASLD, KTS);
      } else {
        kv_load_direct<HD, BN>(k8, v8, (nt + 1) * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kt,
                               ASLD, KTS);
      }
      if (tid < BN) {
        int jg = (nt + 1) * BN + tid;
        ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
        vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
      }
      __syncthreads();
    }
  }
}

// =============================================================================
// 4) convert：fp32 累加缓冲 -> 输出（本版直接 fp32 拷贝）
// =============================================================================
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               float* __restrict__ dq, float* __restrict__ dk,
                               float* __restrict__ dv, size_t nq, size_t nkv) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nq;
       i += (size_t)gridDim.x * blockDim.x)
    dq[i] = dq_acc[i];
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < nkv;
       i += (size_t)gridDim.x * blockDim.x) {
    dk[i] = dk_acc[i];
    dv[i] = dv_acc[i];
  }
}

#endif  // FA_BWD_FP8_KERNELS_CUH_
