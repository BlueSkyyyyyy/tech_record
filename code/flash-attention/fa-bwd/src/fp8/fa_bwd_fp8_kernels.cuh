// =============================================================================
// fa_bwd_fp8_kernels.cuh —— FlashAttention 反向（FP8）**两文件版的 device 部分**
// =============================================================================
// 由 P3-4 单文件 `fa_bwd_fp8_mma_onefile.cu` 拆分而来（P3-5），device 代码与单文件
// **逐字一致**（preprocess 与 golden 相同，main 为张量核 mma 版）：
//   * quantize_row_kernel：fp32 [rows][D] -> fp8 + rowwise scale（P5-3 起支持 D>128）
//   * lse_mma_kernel（O1）：mma 分块 Q·Kᵀ + online-softmax 求 LSE
//   * delta_kernel：反量化 dO 与 fp32 O 逐行点积求 D=rowsum(dO∘O)
//   * delta_warp_kernel（O26）：上者的 warp-per-row 向量化版（对齐 fp16/bf16 O24）
//   * fa_bwd_fp8_mma_kernel：1colblock 反向主 kernel，5 个 GEMM 全部 mma.m16n8k32
//   * convert_kernel：fp32 累加缓冲 -> fp32 输出
// host 侧（npy 读取 / launcher / 自测）见 `fa_bwd_fp8_main.cu`。
//
// P5-3（GQA/MQA）：第 h 个 Q 头映射到 KV 头 hkv=h/(H/Hkv)（与 ref 的 repeat_interleave
// 对齐）。Q/dO/dQ 按 [B,S,H,D] 索引，K/V/dK/dV 按 [B,S,Hkv,D] 索引；Hkv==H 时逐式退化为 MHA。
//
// O4b：GEMM3/4/5 的 B 操作数（dOᵀ/Qᵀ/Kᵀ）从「逐字节 scatter 写的转置副本 Kt/Qt/dOt」改成
//   **K 配对布局 Kp/Qp/dOp**（uint16，元素 = 相邻两行同一 d），用 `ldmatrix.x2.trans` 读 B。
//   先在 `fa_bwd_fp8_trans_smoke.cu` 验证与 `[N][K]` + `ldmatrix.x2` **逐位一致**；见 docs/03 §17。
//
// P5-3（MLA head_dim=512）：与 fp16/bf16 版同样的思路，把 head_dim 变成模板参数 `HD`。
//   * **GEMM1/GEMM2 的 HD 是归约维**（K_TILE=HD）→ 只是把 k-loop 从 4 次变 16 次；
//   * **GEMM3/4/5 的 HD 是输出 N 维**（dOᵀ/Qᵀ/Kᵀ 的行）→ 原实现硬编码「2 warp × 64 = 128
//     列刚好铺满 HD=128」，HD>128 时必须再加一层 **N-tile 循环**（每遍 128 列，共 HD/128 遍），
//     B 操作数按 `d0*stride` 偏移、写回列号加 `d0`。
//   * `HD=128,BM=64,BN=32` 时 N-tile 循环只跑 1 遍，**与 P3-5/O1–O4a 逐位相同**。
//   * `HD=512` 选 `BM=64,BN=32`：1 CTA/SM；寄存器预取（O3）每线程要 `NPU` 个行对（4 个
//     uint32/unit），HD=512 时 NPU=16（64 regs）会爆寄存器，故只在 `NPU*4<=16` 时启用。
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

// ----------------------------- O11：快速指数/对数 -----------------------------
// softmax 热点的 libdevice 精确 `expf`/`logf` 换成硬件内建 `__expf`/`__logf`
// （MUFU.EX2/LG2）。fp8 容差 O(1)，无影响；`FAST_EXP=0` 退回精确版做 A/B。
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

static constexpr float kE4M3Max = 448.0f;
static constexpr float kE5M2Max = 57344.0f;

// O1：preprocess 的 LSE 改用 mma 分块（见下），独立的 tile 常量。
// 每个 CTA 负责 LBM 行，4 个 warp 各 16 行（wm=wid），沿 N 一次 LBN 列。
static constexpr int LBM = 64;
static constexpr int LBN = 64;

// =============================================================================
// Fp8Cfg<HD,BM,BN>：把原来的一组全局常量按 head_dim/tile 参数化。
//   * ASLD  = HD + 16      （Qs/Ks/Vs/dOs 行距，A 的行是 head_dim）
//   * PSLD  = HD + 8       （O4b 配对布局 [K/2][HD] 的 uint16 行距）
//   * DSS2 = BN + 16（dS2 的行距）
//   * smem_bytes：主 kernel 动态 smem（含 O2 折叠后的布局）
//   * lse_smem_bytes：LSE kernel 动态 smem（只与 HD 有关）
//   * use_prefetch：O3 寄存器预取只在每线程预取量小时启用（HD=128 时 NPU=4）。
// =============================================================================
template <int HD, int BM_, int BN_>
struct Fp8Cfg {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int ASLD = HD + 16;   // 144（HD=128）/ 528（HD=512）
  static constexpr int PSLD = HD + 8;    // 136 / 520，配对布局 uint16 行距
  static constexpr int QTS = BM + 16;    // 80，Ap/dS3 [j][m]（GEMM3/4 的 A）行距
  static constexpr int DSS2 = BN + 16;   // 48，dS2 [m][j]（K=BN）
  // O4b：K/V 载入按「行对 + K 配对」组织，每线程负责 NPU 个 unit（unit = 行对×4 个 d）。
  static constexpr int NPU = (BN / 2) * (HD / 4) / THREADS;
  static constexpr int kNScale = 3 * BM + 4 * BN;    // qs,dos,sds2 (BM) + ks,vs,sA,sds3 (BN)

  // O4d：P/S 两个 fp32 [BM][BN] 缓冲的行距 padding。行距 = BN = 32 word（128B）时，
  //   * fold 里按「列」读 `P[m][j]`（j 固定、m 步进 16）会全部落同一 bank（32|stride）→ 4-way；
  //   * GEMM1/2 epilogue 里 8 行 × 4 列同时写 `P[r][c]`，bank=c，同列不同行全撞 → 8-way。
  //   +1 word（33）让 bank 与行号线性相关。实测（S=4096 ksplit=1）：`op_ld` 冲突
  //   199.9M→77.3M（−61%）、`op_st` 231.6M→198.4M（−14%）、总多余 wavefronts
  //   613.6M→456.0M（−26%），main 6.56→5.85 ms（1.12×）。奇数才能保证
  //   m 步进 16 时 `16*33 mod 32 = 16 != 0`（+4/+8 的偶数 padding 无效）。
  //
  // O7e-3：PSS=33（=BN+1）仍让 **GEMM1/2 epilogue** 的 `Ps/Ss[r*PSS+c]` 4-way bank
  //   conflict：mma 累加器里同一 store 指令内 `r=R0+g`（g=lane>>2∈0..7）、`c=C0+2l`
  //   （l=lane&3），bank=(g*PSS+2l) mod32；PSS=33（≡1）时 `{g+2l}` 有大量重合 ⇒ 实测
  //   `Ss` 写 25.6M、`Ps` 写 12.8M、`Ps` 回读 12.8M 多余 wavefronts（占全 kernel shared
  //   多余 wavefronts 的 ~98%）。把 padding 改成 `+5`（37，仍 ≡1 mod 4）后
  //   `bank=(5g+2l) mod32` 最大重合降到 2-way，而 fold 的掩码读（`sub4*8+t+half*32`）
  //   仍无冲突（`37 mod 4 = 1`）。数值逐位不变（只改 smem 地址）。
#ifndef FA_PSS_EXTRA
#define FA_PSS_EXTRA 5
#endif
  static constexpr int PSS = BN + FA_PSS_EXTRA;   // P/S fp32 行距（默认 37；1=旧值 33，供 A/B）

  // O20：mma 主路径的 GEMM1/GEMM2 epilogue 融合开关。默认 1：GEMM1 算出的 P 直接留在寄存器
  //   `preg[2][2][4]`（16 个 fp32）里给 GEMM2 的 epilogue 用，省掉原实现对 `Ps[r*PSS+c]` 的
  //   smem 回读（O7e-3 ncu：该回读贡献 12.78M 多余 wavefronts）。`-DFA_FUSE_EPI=0` 供同 shape A/B
  //   （退回「写 Ps 再读回」）。只影响 smem 访问路径，数学与数值逐位不变。
#ifndef FA_FUSE_EPI
#define FA_FUSE_EPI 1
#endif

  // O22：mma 主路径的 GEMM1/GEMM2 **指令级交错**开关（需 FA_FUSE_EPI=1）。默认 0：保持原顺序
  //   （GEMM1 mma → GEMM1 epilogue → GEMM2 mma → GEMM2 epilogue）。置 1：先连发两条独立 GEMM
  //   的 mma（acc/acc2 两个累加器），再依次做 epilogue——让 GEMM2 的 8 条 mma 填在 GEMM1
  //   epilogue（exp/量化）之前，掩盖 mma 依赖延迟（O20 ncu：`wait` 1.55 是头号 stall）。数学
  //   与数值逐位不变（同一批 mma、同一 (r,c) 映射、同一次序的 fp32 累加），只改指令发射顺序。
#ifndef FA_ILV
#define FA_ILV 0
#endif

  // O4b：Kt/Qt/dOt 三个「逐字节 scatter 写的转置副本」→ Kp/Qp/dOp 三个 **K 配对布局**
  //   （uint16：[K/2][HD]，元素 = 2 个相邻 K 值），用 `ldmatrix.x2.trans` 读 B 片段。
  //   * Qp（[BM/2][HD]）供 GEMM4 的 B=Qᵀ；dOp 供 GEMM3 的 B=dOᵀ；Kp（[BN/2][HD]）供 GEMM5。
  //   * 每个配对数组 = (rows/2)*PSLD*2 bytes；比原 [HD][K+16] 副本更小（无 +16 行距放大）。
  static constexpr int qp_bytes = (BM / 2) * PSLD * 2;
  static constexpr int kp_bytes = (BN / 2) * PSLD * 2;
  static constexpr int fp8_bytes = BM * ASLD   // Qs
                                 + BN * ASLD   // Ks（dS3 复用尾部）
                                 + BN * ASLD   // Vs（Ap 复用尾部）
                                 + BM * ASLD   // dOs
                                 + BM * DSS2   // dS2
                                 + qp_bytes    // Qp（GEMM4 B）
                                 + qp_bytes    // dOp（GEMM3 B）
                                 + kp_bytes;   // Kp（GEMM5 B）
  static constexpr int smem_bytes =
      fp8_bytes + (kNScale + 2 * BM * PSS) * (int)sizeof(float);

  // O9c-2：WGMMA 模式下 Q/dO/K/V 存 SW128（tile 字节数 = (rows/8)*(HD/128)*1024），
  // 取代行主序的 ASLD 布局。+1024 是动态 smem 基址到 1024B 对齐的 slack（描述符 base_offset=0）。
  static constexpr int qs_sw_bytes = (BM / 8) * (HD / 128) * 1024;
  static constexpr int ks_sw_bytes = (BN / 8) * (HD / 128) * 1024;
  static constexpr int fp8_bytes_wgmma = qs_sw_bytes   // Qs
                                       + ks_sw_bytes   // Ks（dS3 复用尾部）
                                       + ks_sw_bytes   // Vs（Ap 复用尾部）
                                       + qs_sw_bytes   // dOs
                                       + BM * DSS2     // dS2
                                       + qp_bytes + qp_bytes + kp_bytes;
  static constexpr int smem_bytes_wgmma =
      fp8_bytes_wgmma + (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 1024;

  static constexpr int lse_smem_bytes =
      LBM * ASLD + LBN * ASLD + (LBM + LBN) * (int)sizeof(float);
  // O11：LSE 的镜像配对 + cp.async 双缓冲版本（PIPE=0 单缓冲 / PIPE=1 双缓冲）smem。
  static constexpr int lse_smem_bytes_bal0 =
      LBM * ASLD + LBN * ASLD + (LBM + LBN) * (int)sizeof(float);
  static constexpr int lse_smem_bytes_bal1 =
      LBM * ASLD + 2 * LBN * ASLD + (LBM + 2 * LBN) * (int)sizeof(float);
  // O9c：fp8 wgmma LSE（SW128 K-major，LBM=LBN=64）。tile 字节数 = (LBM/8)*(HD/128)*1024；
  //   +1024 是 1024B 对齐的 slack（SW128 描述符 base_offset=0 要求 tile 1024B 对齐）。
  static constexpr int lse_tile_wgmma = (LBM / 8) * (HD / 128) * 1024;
  static constexpr int lse_smem_bytes_balw0 =
      2 * lse_tile_wgmma + (LBM + LBN) * (int)sizeof(float) + 1024;
  static constexpr int lse_smem_bytes_balw1 =
      3 * lse_tile_wgmma + (LBM + 2 * LBN) * (int)sizeof(float) + 1024;

  // O3 寄存器预取：pk0/pk1/pv0/pv1 各 NPU 个 uint32。HD=128 时 NPU=4（共 16 regs，可行）；
  // HD=512 时 NPU=16（共 64 regs，会挤掉累加器/地址寄存器）→ 关闭，走直接向量化读。
  static constexpr bool use_prefetch = (NPU * 4 <= 16);
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
// O11：16B 异步拷贝（fp8 的 16B = 16 个元素）。fp8 主 kernel 用的是 O3 寄存器预取，
// 但 LSE 里 K/Q 只读一次、不需要留寄存器，用 `cp.async.cg`（L2-only）双缓冲更省。
__device__ __forceinline__ void cp_async16(void* dst_smem, const void* src_gmem) {
  uint32_t s = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(src_gmem));
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
// O4b：从「K 配对」布局 Sp[k/2][n]（uint16）取 mma 的 B 片段。已验证（trans_smoke）：
// 与从 [N][K] 行主序用 `ldmatrix.x2` 读到的片段**逐位相同**。见 docs/03 §17。
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t addr, uint32_t d[2]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(d[0]), "=r"(d[1])
               : "r"(addr));
}

// =============================================================================
// O9c（第一步）：fp8 Hopper `wgmma` 数据通路（SW128 K-major + wgmma.m64n64k32）
// =============================================================================
// 目的：把 fp8 反向的 `mma.m16n8k32 + ldmatrix`（在 Hopper 上走 SM80 兼容路径、ncu 显示
// 第一墙 L1/TEX 66–71% 来自 `ldmatrix`+smem 访存）换成 Hopper warpgroup MMA：SS 直读 smem
// 描述符，免 ldmatrix、减少 smem 访存。本步先在**风险最小的 LSE（单个 QKᵀ）**上建立数据
// 通路（与 fp16 的 O9a 同构），主 kernel 的 5 个 GEMM 上 wgmma 留作 O9c 后续。
//
// fp8 的 SW128 与 bf16 **逐字节同构**（atom 恒 8 行 × 128B），只差「一行 128B = 128 个
// 元素」（bf16 是 64 个），故 16B chunk 下标 = `k/16`（bf16 是 `k/8`）。元素 (row,k) 的字节
// 偏移见 `sw128_off_fp8`（布局 [row/8][k/128][8 行][128 元素]，atom 1024B；atom 内 16B
// chunk c' = (k/16 & 7) ^ (row&7)）。描述符 SBO=(K/128)*1024、LBO=1（B128 下硬件忽略），
// k32 步进地址 = `(s>>2)*1024 + (s&3)*32`（s=k/32，每步 32B = 32 个 fp8）。
// 由冒烟 `fa_bwd_fp8_wgmma_smoke.cu` 逐位验证（SW128 存 + 描述符 + m64n64k32 累加器映射）。
//
// 仅 `-DFA_WGMMA` 构建时编译（默认 sm_90 构建完全不含本块，行为/数值不变）；wgmma 需
// `-gencode=arch=compute_90a,code=sm_90a`（CUDA 13 的 `-arch=sm_90a` 会静默退化）。
// O9c-2：SW128 的三个纯整数 helper 移出 `#ifdef FA_WGMMA`，好让主 kernel 的
//   `if constexpr (WGMMA)` 存储分支在任意构建下都能做地址运算（wgmma asm 仍只在
//   `-DFA_WGMMA` 下编译）。它们在非 wgmma 构建里不会被实例化调用。
// SW128 K-major：元素 (row,k) 的字节偏移；K = 行宽（元素数，须为 128 的倍数）。
__device__ __forceinline__ int sw128_off_fp8(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (rg * (K >> 7) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
// k32 块索引 s（沿 K）对应的描述符起始地址增量（字节）。
__device__ __forceinline__ uint32_t sw128_k32_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
__device__ __forceinline__ uint64_t make_desc_sw128_fp8(uint32_t addr,
                                                        uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;  // LBO = 1（B128 下硬件忽略）
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;  // base_offset（tile 1024B 对齐时相位 0）
  d |= (uint64_t)1 << 62;  // layout_type = B128
  return d;
}

#ifdef FA_WGMMA
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define FA_FP8_HAS_WGMMA 1
#else
#define FA_FP8_HAS_WGMMA 0
#endif
__device__ __forceinline__ void wgmma_fence_fp8() {
#if FA_FP8_HAS_WGMMA
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_commit_fp8() {
#if FA_FP8_HAS_WGMMA
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wgmma_wait0_fp8() {
#if FA_FP8_HAS_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
// wgmma.m64n64k32 E4M3×E4M3（fp8 尾部操作数与 bf16 不同：`p, scaleA, scaleB` 三个；
// 我们在 epilogue 乘 rowwise scale，故两条 scale 立即数都取 1）。累加器映射与
// mma.m16n8 的 `acc[j][q]` 同构：warp w 持行 [16w,16w+16)，`d[j*4+q]` ↔
// row=16w+g+(q>=2?8:0)、col=j*8+2*(lane%4)+(q&1)。
__device__ __forceinline__ void wgmma_m64n64k32_e4e4(float (&d)[32], uint64_t da,
                                                     uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
#else
  (void)da; (void)db;
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
#endif
}
// E5M2×E4M3（GEMM2 `dP=dO·Vᵀ` 口径）。
__device__ __forceinline__ void wgmma_m64n64k32_e5e4(float (&d)[32], uint64_t da,
                                                     uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %34, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k32.f32.e5m2.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},\n"
      "%32, %33, p, %35, %36;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]),
        "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]),
        "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]),
        "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
#else
  (void)da; (void)db;
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
#endif
}
// O9c-2：主 kernel 的 GEMM1/2 用 **m64n32k32**（主 kernel 的 BN=32，正好 n32；比 n64
//   少一半累加器，128 线程每线程 16 个 fp32）。累加器映射与 m64n64 同构，只是 j<4：
//   warp w 持行 [16w,16w+16)，`d[j*4+q]` ↔ row=16w+g+(q>=2?8:0)、col=j*8+c2+(q&1)。
//   （与 mma.m16n8 的 `acc[j][q]` 同构，方便直接套现用 epilogue 的 (r,c) 公式。）
__device__ __forceinline__ void wgmma_m64n32k32_e4e4(float (&d)[16], uint64_t da,
                                                     uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "%16, %17, p, %19, %20;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
#else
  (void)da; (void)db;
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
#endif
}
__device__ __forceinline__ void wgmma_m64n32k32_e5e4(float (&d)[16], uint64_t da,
                                                     uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e5m2.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "%16, %17, p, %19, %20;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "l"(da), "l"(db), "r"(1), "n"(1), "n"(1));
#else
  (void)da; (void)db;
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
#endif
}
// 发 GEMM1/2 的 wgmma（A/B 均 SW128 K-major，K 归约维 HD），只 commit 不 wait，方便两条
// 异步 mma 重叠（与 fp16 O9b 的 `wgmma_mn64_issue` 同）。KIND=0 e4m3×e4m3，1 e5m2×e4m3。
template <int KIND>
__device__ __forceinline__ void wgmma_mn32_issue(const char* Asw, const char* Bsw, int HD,
                                                 float (&d)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
  wgmma_fence_fp8();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((HD / 128) * 1024);
#pragma unroll
  for (int s = 0; s < HD / 32; ++s) {
    uint64_t da = make_desc_sw128_fp8(sw128_k32_addr(aa, s), sbo);
    uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(ba, s), sbo);
    if (KIND == 0)
      wgmma_m64n32k32_e4e4(d, da, db);
    else
      wgmma_m64n32k32_e5e4(d, da, db);
  }
  wgmma_commit_fp8();
}

// Q[64][HD]·Kᵀ[HD][64]（Q/K 存成 SW128 K-major，K 归约维 HD，HD 须为 128 的倍数）。
__device__ __forceinline__ void wgmma_qkt64_fp8(const char* Qsw, const char* Ksw,
                                                int HD, float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence_fp8();
  const uint32_t qa = smem_u32(Qsw), ka = smem_u32(Ksw);
  const uint32_t sbo = (uint32_t)((HD / 128) * 1024);
#pragma unroll
  for (int s = 0; s < HD / 32; ++s) {
    uint64_t da = make_desc_sw128_fp8(sw128_k32_addr(qa, s), sbo);
    uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(ka, s), sbo);
    wgmma_m64n64k32_e4e4(d, da, db);
  }
  wgmma_commit_fp8();
  wgmma_wait0_fp8();
}
#endif  // FA_WGMMA

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

// O4b：「B 为 K 配对布局」的 mma。Bp[k/2][n] 是 uint16 行主序（行距 bp_sld，单位 uint16），
//   元素 = 「操作数第 n 行、第 2i/2i+1 两个 k」打包。A 与非转置版完全一致；B 用
//   `ldmatrix.x2.trans` 读出（已验证与 [N][K] + `ldmatrix.x2` 逐位相同，见 trans_smoke）。
//   nb0 是本次 N 方向分块（head_dim 的 nd*128）在配对数组里的**列**偏移。
template <int WARP_M, int WARP_N, int K_TILE, int KIND>
__device__ __forceinline__ void mma_block_bt(const unsigned char* As, int asld,
                                             const uint16_t* Bp, int bp_sld,
                                             float acc[WARP_M / 16][WARP_N / 8][4], int wm,
                                             int wn, int lane, int nb0) {
  constexpr int MTM = WARP_M / 16, MTN = WARP_N / 8;
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
    uint32_t bv[MTN][2];
#pragma unroll
    for (int j = 0; j < MTN; ++j) {
      uint32_t d[2];
      // lanes 0..15 给出 Bp 的 16 行地址（k/2 = koff/2 .. +15），列偏移 = nb0 + 本 n-tile。
      ldmatrix_x2_trans(
          smem_u32(Bp + (koff / 2 + (lane & 15)) * bp_sld + nb0 + wn * WARP_N + j * 8), d);
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

// ------------------- O4b：K/V 载入（行对 + K 配对），替代原逐字节转置副本 -------------------
// fp8 main kernel 的 per-tile 全局读（K/V）原本逐行载入，并把 K 逐字节 scatter 成转置副本
// Kt（ncu：op_st bank conflict 极高）。O4b 改成按「行对 rp（2 行）+ 4 个连续 d」的 unit：
//   * 原始两行写 Ks[2rp][dq..] / Ks[2rp+1][dq..]（供 GEMM1 的 B）；
//   * 打包写 Kp[rp][dq..]（uint16，元素 = 相邻两行的同一 d），供 GEMM5 用 `ldmatrix.x2.trans`。
//   * 打包用 `__byte_perm(a, b, 0x5140/0x7362)` 做字节交织（一次 4B 写两个 uint16）。
// 与 O3 一致仍用**寄存器双缓冲预取**（不额外占 smem），只是每线程预取的是 NPU 个 unit。
template <int NPU, int HD>
__device__ __forceinline__ void kv_prefetch_pair(const unsigned char* __restrict__ k8,
                                                 const unsigned char* __restrict__ v8,
                                                 int j0, int S, int Hkv, int hkv, int b,
                                                 int tid, uint32_t* pk0, uint32_t* pk1,
                                                 uint32_t* pv0, uint32_t* pv1) {
  const int nd4 = HD / 4;
#pragma unroll
  for (int e = 0; e < NPU; ++e) {
    int u = tid + e * THREADS;
    int rp = u / nd4, dq = (u % nd4) * 4;
    int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    if (jr < S) {  // 越界写 0 字节（等价 cvt_e4m3(0)=0x00）
      size_t i0 = (((size_t)(b * S + jr)) * Hkv + hkv) * HD + dq;
      k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
      v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
    }
    if (jr + 1 < S) {
      size_t i1 = (((size_t)(b * S + jr + 1)) * Hkv + hkv) * HD + dq;
      k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
      v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
    }
    pk0[e] = k0;
    pk1[e] = k1;
    pv0[e] = v0;
    pv1[e] = v1;
  }
}

// O9c-2：`SW=true` 时 Ks/Vs 写 SW128（供 wgmma GEMM1/2 直读），否则写行主序。Kp 恒定。
template <int NPU, int HD, bool SW = false>
__device__ __forceinline__ void kv_commit_pair(unsigned char* Ks, unsigned char* Vs,
                                               uint16_t* Kp, const uint32_t* pk0,
                                               const uint32_t* pk1, const uint32_t* pv0,
                                               const uint32_t* pv1, int tid, int asld,
                                               int psld) {
  const int nd4 = HD / 4;
#pragma unroll
  for (int e = 0; e < NPU; ++e) {
    int u = tid + e * THREADS;
    int rp = u / nd4, dq = (u % nd4) * 4;
    if constexpr (SW) {
      *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2, dq, HD)) = pk0[e];
      *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2 + 1, dq, HD)) = pk1[e];
      *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2, dq, HD)) = pv0[e];
      *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = pv1[e];
    } else {
      *reinterpret_cast<uint32_t*>(Ks + (rp * 2) * asld + dq) = pk0[e];
      *reinterpret_cast<uint32_t*>(Ks + (rp * 2 + 1) * asld + dq) = pk1[e];
      *reinterpret_cast<uint32_t*>(Vs + (rp * 2) * asld + dq) = pv0[e];
      *reinterpret_cast<uint32_t*>(Vs + (rp * 2 + 1) * asld + dq) = pv1[e];
    }
    uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * psld + dq);
    kpw[0] = __byte_perm(pk0[e], pk1[e], 0x5140);
    kpw[1] = __byte_perm(pk0[e], pk1[e], 0x7362);
  }
}

// HD>128（MLA）时的直接向量化载入（无寄存器预取）。NDQ 个 unit grid-stride。
template <int HD, int BN, bool SW = false>
__device__ __forceinline__ void kv_load_pair(const unsigned char* __restrict__ k8,
                                             const unsigned char* __restrict__ v8,
                                             int j0, int S, int Hkv, int hkv, int b, int tid,
                                             unsigned char* Ks, unsigned char* Vs,
                                             uint16_t* Kp, int asld, int psld) {
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;
  for (int u = tid; u < units; u += THREADS) {
    int rp = u / nd4, dq = (u % nd4) * 4;
    int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    if (jr < S) {
      size_t i0 = (((size_t)(b * S + jr)) * Hkv + hkv) * HD + dq;
      k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
      v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
    }
    if (jr + 1 < S) {
      size_t i1 = (((size_t)(b * S + jr + 1)) * Hkv + hkv) * HD + dq;
      k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
      v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
    }
    if constexpr (SW) {
      *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2, dq, HD)) = k0;
      *reinterpret_cast<uint32_t*>(Ks + sw128_off_fp8(rp * 2 + 1, dq, HD)) = k1;
      *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2, dq, HD)) = v0;
      *reinterpret_cast<uint32_t*>(Vs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = v1;
    } else {
      *reinterpret_cast<uint32_t*>(Ks + (rp * 2) * asld + dq) = k0;
      *reinterpret_cast<uint32_t*>(Ks + (rp * 2 + 1) * asld + dq) = k1;
      *reinterpret_cast<uint32_t*>(Vs + (rp * 2) * asld + dq) = v0;
      *reinterpret_cast<uint32_t*>(Vs + (rp * 2 + 1) * asld + dq) = v1;
    }
    uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * psld + dq);
    kpw[0] = __byte_perm(k0, k1, 0x5140);
    kpw[1] = __byte_perm(k0, k1, 0x7362);
  }
}

// O19：`kv_load_pair` 的线程数可参数化版（供 256 线程的 wg2 kernel 用）。逻辑逐字相同，
//   只把 grid-stride 的步长从全局 `THREADS` 换成模板 `NT`。
template <int HD, int BN, int NT>
__device__ __forceinline__ void kv_load_pair_nt(const unsigned char* __restrict__ k8,
                                                const unsigned char* __restrict__ v8,
                                                int j0, int S, int Hkv, int hkv, int b,
                                                int tid, unsigned char* Ks, unsigned char* Vs,
                                                uint16_t* Kp, int asld, int psld) {
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;
  for (int u = tid; u < units; u += NT) {
    int rp = u / nd4, dq = (u % nd4) * 4;
    int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    if (jr < S) {
      size_t i0 = (((size_t)(b * S + jr)) * Hkv + hkv) * HD + dq;
      k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
      v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
    }
    if (jr + 1 < S) {
      size_t i1 = (((size_t)(b * S + jr + 1)) * Hkv + hkv) * HD + dq;
      k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
      v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
    }
    *reinterpret_cast<uint32_t*>(Ks + (rp * 2) * asld + dq) = k0;
    *reinterpret_cast<uint32_t*>(Ks + (rp * 2 + 1) * asld + dq) = k1;
    *reinterpret_cast<uint32_t*>(Vs + (rp * 2) * asld + dq) = v0;
    *reinterpret_cast<uint32_t*>(Vs + (rp * 2 + 1) * asld + dq) = v1;
    uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * psld + dq);
    kpw[0] = __byte_perm(k0, k1, 0x5140);
    kpw[1] = __byte_perm(k0, k1, 0x7362);
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
// 1b) O14：quantize_row_warp_kernel —— 输入量化的「每 warp 一行」向量化版
// =============================================================================
// 旧 `quantize_row_kernel` 是**每行一个 CTA**（128 线程）：线程数远超一行元素数（D=128），
// 且用 `__shared__ float sh[128]` + 7 次 `__syncthreads` 做行 amax。S=4096 时 grid = S·H
// (=65536) 个 CTA、每个只搬 512B，实测 quant（四个张量 q/k/v/dO 各一遍）0.19ms，
// 比纯带宽下限（~30µs）慢 ~6.5×，是端到端第三大项（S=1024H32 时占 15%）。
//
// 本版改为**每 warp 一行**：lane 用 `float4` 读 `D/32` 个元素（D=128→1 个、D=512→4 个）
// 进寄存器，`__shfl_xor_sync` 树求行 amax，再就地量化、用 `uchar4` 写回。全程序无 `__shared__`、
// 无 `__syncthreads`，读/写均 16B/4B 向量化且完全合并。
//
// **数值逐位不变**：行 amax 用 `fmaxf`，可交换结合 ⇒ warp 树与旧 smem 树的归约顺序无关；
// scale 公式、每元素 `cvt_*` 与旧版逐元素一致。故 q8/qs 等与旧 kernel 完全相同（本篇 A/B 验证）。
// 只实例化 D%4==0 且 D/32 ∈ {4,16}（本项目的 head_dim 128/512）；其它 D 回退旧 kernel。
template <int VPT, bool E5M2>
__global__ void __launch_bounds__(128)
quantize_row_warp_kernel(const float* __restrict__ x, unsigned char* __restrict__ xq,
                         float* __restrict__ scale, long long nrows) {
  constexpr int D = VPT * 32;  // 一行元素数（VPT=4→128，16→512）
  const float fp8_max = E5M2 ? kE5M2Max : kE4M3Max;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NW = 4;  // 128 线程 = 4 个 warp
  for (long long row = (long long)blockIdx.x * NW + warp; row < nrows;
       row += (long long)gridDim.x * NW) {
    const float4* xr4 = reinterpret_cast<const float4*>(x + row * (long long)D);
    float v[VPT];
    float amax = 0.f;
#pragma unroll
    for (int t = 0; t < VPT / 4; ++t) {
      const float4 q = xr4[t * 32 + lane];
      v[t * 4 + 0] = q.x;
      v[t * 4 + 1] = q.y;
      v[t * 4 + 2] = q.z;
      v[t * 4 + 3] = q.w;
      amax = fmaxf(amax, fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fmaxf(fabsf(q.z), fabsf(q.w))));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    const float s = (amax > 0.f) ? (amax / fp8_max) : 1.f;
    if (lane == 0) scale[row] = s;
    unsigned char* out = xq + row * (long long)D;
#pragma unroll
    for (int t = 0; t < VPT / 4; ++t) {
      uchar4 o;
      o.x = E5M2 ? cvt_e5m2(v[t * 4 + 0] / s) : cvt_e4m3(v[t * 4 + 0] / s);
      o.y = E5M2 ? cvt_e5m2(v[t * 4 + 1] / s) : cvt_e4m3(v[t * 4 + 1] / s);
      o.z = E5M2 ? cvt_e5m2(v[t * 4 + 2] / s) : cvt_e4m3(v[t * 4 + 2] / s);
      o.w = E5M2 ? cvt_e5m2(v[t * 4 + 3] / s) : cvt_e4m3(v[t * 4 + 3] / s);
      *reinterpret_cast<uchar4*>(out + (t * 32 + lane) * 4) = o;
    }
  }
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
// 2a') lse_mma_kernel_bal【O11】：把 fp16/bf16 的 O8b 负载均衡 + cp.async 移植到 fp8
// =============================================================================
// 动机：fp8 的 `lse_mma_kernel`（O1）每个 m 块一个 CTA，因果下第 m 块要做 m+1 个 K tile，
// 重块排最后（尾波），且每个 tile 的 K 是「同步逐字节标量读」。fp16/bf16 的 O8b 证明：
//   ① **镜像配对**（每 CTA 处理 m 与 nblk-1-m，工作量恒 nblk+1，grid.x 减半）；
//   ② **K/Q 的 `cp.async.cg` 16B 双缓冲**再叠加。
// fp8 的 K/Q 是 1B/元素、一行 HD 字节，16B = 16 个 fp8，故 unit 数 = HD/16。
// 数学与 O1 完全一致（同 E4E4 mma、同 online-softmax、同 4-lane shfl 归约、同 rowwise
// scale 相乘顺序），数值应逐位相同。仅用于 causal；非 causal 走 O1 原版。
// smem：Qs[LBM*ASLD] +（PIPE=1 双缓冲 / PIPE=0 单缓冲）Ks[LBN*ASLD] + qs/ks 标量。
template <int HD, int PIPE>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                   const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                   float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int KVL  = LBN * ASLD;
  constexpr int HDV  = HD / 16;   // 每行 16B（16 个 fp8）unit 数
  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks = Qs + LBM * ASLD;                               // PIPE=1：2×KVL
  float* qs_s = reinterpret_cast<float*>(Ks + (PIPE ? 2 : 1) * KVL);  // [LBM]
  float* ks_s = qs_s + LBM;                                          // PIPE=1：2×LBN

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // 发本 m 块的 Q（PIPE=1 用 16B cp.async，行越界写 0=cvt_e4m3(0)）与 rowwise scale。
  auto issue_q = [&](int m0) {
#pragma unroll
    for (int u = tid; u < LBM * HDV; u += THREADS) {
      const int row = u / HDV, c16 = u % HDV;
      const int qi = m0 + row;
      unsigned char* d = Qs + row * ASLD + c16 * 16;
      if (qi < S) {
        const unsigned char* s = q8 + (((size_t)(b * S + qi)) * H + h) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBM) qs_s[tid] = (m0 + tid < S) ? qs[((size_t)(b * S + m0 + tid)) * H + h] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

  // 发一个 K tile（j0 起 LBN 行）到 Kd，并写本 tile 的 rowwise scale 到 KdS。
  auto issue_k = [&](unsigned char* Kd, float* KdS, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c16 = u % HDV;
      const int jg = j0 + row;
      unsigned char* d = Kd + row * ASLD + c16 * 16;
      if (jg < S) {
        const unsigned char* s = k8 + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBN)
      KdS[tid] = (j0 + tid < S) ? ks[((size_t)(b * S + j0 + tid)) * Hkv + hkv] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) continue;  // 奇数 nblk 的中心块只做一次
    const int m0 = mblk * LBM;
    issue_q(m0);

    const int ncols = min(S, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    if constexpr (PIPE) {
      if (ntiles > 0) issue_k(Ks, ks_s, 0);
    }
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

    for (int nt = 0; nt < ntiles; ++nt) {
      const int j0 = nt * LBN;
      unsigned char* Kt = Ks + (PIPE ? (nt & 1) * KVL : 0);
      float* KtS = ks_s + (PIPE ? (nt & 1) * LBN : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (nt + 1 < ntiles)
          issue_k(Ks + ((nt + 1) & 1) * KVL, ks_s + ((nt + 1) & 1) * LBN, j0 + LBN);
      } else {
        issue_k(Ks, ks_s, j0);
        __syncthreads();
      }

      float acc[1][8][4];
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block<16, LBN, HD, E4E4>(Qs, ASLD, Kt, ASLD, acc, wid, 0, lane);

#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int s = q >= 2 ? 1 : 0;
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float sv = -INFINITY;
          if (qi < S && jg < S && jg <= qi)
            sv = acc[0][j][q] * scale * qs_s[r] * KtS[c];
          if (sv != -INFINITY) {
            float mn = fmaxf(mrow[s], sv);
            lrow[s] = lrow[s] * fexp(mrow[s] - mn) + fexp(sv - mn);
            mrow[s] = mn;
          }
        }
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
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）
    __syncthreads();
  }
}

// =============================================================================
// 2a'') O9c：wgmma 版 LSE（causal 专用，仅 HD=128；镜像配对 + SW128 + wgmma.m64n64k32）
// =============================================================================
// 与 O11 的 `lse_mma_kernel_bal` 数学完全一致（同 E4M3×E4M3 QKᵀ、同 online-softmax、同
// 4-lane `shfl` 归约、同 rowwise scale 相乘顺序），只把「4 warp × m16n64 × 4 k-step 的
// mma+ldmatrix」换成 1 个 warpgroup 的 4 条 `wgmma.m64n64k32`（Q/K 存 SW128、描述符直读，
// 免 ldmatrix）。仅 `-DFA_WGMMA` 构建启用（默认 sm_90 构建走 O11 的 mma 版，行为不变）。
#ifdef FA_WGMMA
template <int HD, int PIPE>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_wgmma(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                         const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                         float* __restrict__ lse, int S, int H, int Hkv, float scale) {
  static_assert(HD == 128, "wgmma LSE 目前只做 HD=128");
  constexpr int HDV = HD / 16;                         // 每行 16B（16 个 fp8）unit 数
  constexpr int TILE = (LBM / 8) * (HD / 128) * 1024;  // 单个 SW128 tile 字节数（HD=128→8KB）
  extern __shared__ char smem_raw[];
  // SW128 描述符 base_offset=0 要求 tile 1024B 对齐 → 手动对齐动态 smem 基址。
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;  // PIPE=1：2*TILE；PIPE=0：TILE
  float* qs_s = reinterpret_cast<float*>(Ks + (PIPE ? 2 : 1) * TILE);
  float* ks_s = qs_s + LBM;  // PIPE=1：2*LBN

  const int nblk = (S + LBM - 1) / LBM;
  const int pair = blockIdx.x, h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // 发本 m 块的 Q（PIPE=1 用 16B cp.async，行越界写 0）到 SW128 tile + rowwise scale。
  auto issue_q = [&](int m0) {
#pragma unroll
    for (int u = tid; u < LBM * HDV; u += THREADS) {
      const int row = u / HDV, c16 = u % HDV;
      const int qi = m0 + row;
      char* d = Qs + sw128_off_fp8(row, c16 * 16, HD);
      if (qi < S) {
        const unsigned char* s = q8 + (((size_t)(b * S + qi)) * H + h) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBM)
      qs_s[tid] = (m0 + tid < S) ? qs[((size_t)(b * S + m0 + tid)) * H + h] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };
  // 发一个 K tile（j0 起 LBN 行）到 SW128 tile Kd，并写本 tile 的 rowwise scale。
  auto issue_k = [&](char* Kd, float* KdS, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c16 = u % HDV;
      const int jg = j0 + row;
      char* d = Kd + sw128_off_fp8(row, c16 * 16, HD);
      if (jg < S) {
        const unsigned char* s = k8 + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBN)
      KdS[tid] = (j0 + tid < S) ? ks[((size_t)(b * S + j0 + tid)) * Hkv + hkv] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) continue;  // 奇数 nblk 的中心块只做一次
    const int m0 = mblk * LBM;
    issue_q(m0);

    const int ncols = min(S, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    if constexpr (PIPE) {
      if (ntiles > 0) issue_k(Ks, ks_s, 0);
    }
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

    for (int nt = 0; nt < ntiles; ++nt) {
      const int j0 = nt * LBN;
      char* Kt = Ks + (PIPE ? (nt & 1) * TILE : 0);
      float* KtS = ks_s + (PIPE ? (nt & 1) * LBN : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (nt + 1 < ntiles)
          issue_k(Ks + ((nt + 1) & 1) * TILE, ks_s + ((nt + 1) & 1) * LBN, j0 + LBN);
      } else {
        issue_k(Ks, ks_s, j0);
        __syncthreads();
      }

      float d[32];
      wgmma_qkt64_fp8(Qs, Kt, HD, d);

#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int s = q >= 2 ? 1 : 0;
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float sv = -INFINITY;
          if (qi < S && jg < S && jg <= qi) sv = d[j * 4 + q] * scale * qs_s[r] * KtS[c];
          if (sv != -INFINITY) {
            float mn = fmaxf(mrow[s], sv);
            lrow[s] = lrow[s] * fexp(mrow[s] - mn) + fexp(sv - mn);
            mrow[s] = mn;
          }
        }
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
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）
    __syncthreads();
  }
}
#endif  // FA_WGMMA


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
// 2c) O26：`delta_kernel` 的 warp-per-row 向量化版（对齐 fp16/bf16 的 O24）。
// =============================================================================
// 旧 `delta_kernel` 每个 (s,h,b) 行一个 128 线程 CTA + `__shared__` 树归约 +
// log2(THREADS)=7 次 `__syncthreads`。HD=128 时一行只有 128 个乘加，block/同步开销
// 远大于计算；fp16/bf16 在 O24 已改成 warp-per-row（S=4096 42.5→14.5µs，3.35×），
// 但 fp8 的 delta 一直是旧版。这里照搬同一数据通路：
//   * **每 warp 一行**：lane 沿 HD 以 `float4`（O，fp32）与 `uchar4`（dO，e5m2）各读 4 个
//     元素（warp 每步 32×4=128 个 d），`__shfl_xor_sync` 树归约，**无 smem / 无 barrier**；
//   * grid-stride 覆盖任意行数（行 input 数 = B*S*H）。
// 数学与旧版逐元素同式 `O[d]*(deq_e5m2(dO[d])*dos[row])`；只有 fp32 求和次序不同
// （warp 树 vs 128 线程 smem 树），对 fp8 容差 O(1) 无影响。`--deltawarp=0` 退回旧版 A/B。
template <int HD>
__global__ void delta_warp_kernel(const float* __restrict__ o,
                                  const unsigned char* __restrict__ do8,
                                  const float* __restrict__ dos, float* __restrict__ delta,
                                  int rows) {
  static_assert(HD % 4 == 0, "delta_warp 需要 HD 为 4 的倍数（按 float4/uchar4 读）");
  const int lane = threadIdx.x & 31;
  const int wpb = blockDim.x >> 5;
  const int gwarp0 = blockIdx.x * wpb + (threadIdx.x >> 5);
  const int nwarp = gridDim.x * wpb;
  for (int row = gwarp0; row < rows; row += nwarp) {
    const float* orow = o + (size_t)row * HD;
    const unsigned char* dorow = do8 + (size_t)row * HD;
    const float do_scale = dos[row];
    float acc = 0.f;
#pragma unroll
    for (int k = lane * 4; k < HD; k += 128) {
      const float4 a = *reinterpret_cast<const float4*>(orow + k);
      const uchar4 b = *reinterpret_cast<const uchar4*>(dorow + k);
      acc += a.x * (deq_e5m2(b.x) * do_scale);
      acc += a.y * (deq_e5m2(b.y) * do_scale);
      acc += a.z * (deq_e5m2(b.z) * do_scale);
      acc += a.w * (deq_e5m2(b.w) * do_scale);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) delta[row] = acc;
  }
}

// =============================================================================
// 3) main kernel：1colblock 反向，5 个 GEMM 全部用 mma.m16n8k32
// =============================================================================
// O7：`REGDQ=true` 时把 dQ 沿 nt 累加在寄存器里（每 CTA 只 flush 一次跨 CTA 归约）。
//   这会多占用 64 个 fp32 累加器（168→254 regs，2 CTA/SM），故用 `__launch_bounds__(THREADS,3)`
//   把寄存器压回 168（~0.9KB spill）保住 3 CTA/SM。只有「平均每 CTA 有足够多 nt tile」时
//   才划算（accumulation 省下的 red ∝ ntiles/CTA，而寄存器/溢出代价固定）；小 S 的高 ksplit
//   使每 CTA 只有 ~1 个 tile，此时 `REGDQ=false` 退回原 per-tile 归约（168 regs，无溢出）。
// O9c-2：`WGMMA=true` 时 GEMM1/2 换 `wgmma.m64n32k32`（A/B 存 SW128 K-major），其余
//   （fold + GEMM3/4/5 + dQ 归约）与 mma 版**逐字相同**。仅 `-DFA_WGMMA` 构建可实例化。
// O12（O7c-PREL for fp8）：`PREL=true` 时把本线程负责的 Q 行的 LSE/D 预装进寄存器，nt 循环内
//   零 global 读。fp16/bf16 早在 O7c（第三十五轮）就做了这一步（main +14–19%），但 fp8 的
//   GEMM1/2 epilogue 一直按 (qi) 逐元素 global 读 lse/delta（ncu：global load 仅 9.8/32B/thread、
//   38M 多余扇区）。`PREL=false` 退回逐元素 global 读，便于同 session A/B。
// O7e-2：`F16B=true` 时修 fold 的 shared-load bank conflict（Ap/dS3 的 `Ps/Ss` 列读从
//   `sub4*16` 起始改成 `sub4*8` 起始，4 组 lane 的 bank 铺满 0..31 无冲突）并把每 lane
//   两段各 8 个连续 m 用 8B `st.shared.v2.u32` 落盘；dS2 用 16B `st.shared.v4.u32`。
//   `F16B=false` 退回 O7e 的「`sub4*16` 起始 + 4×4B 写」，便于同 session A/B。数值逐位不变。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true>
__global__ void __launch_bounds__(THREADS, (HD == 128) ? (BN <= 32 ? 3 : 2) : 1)
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
  constexpr int PSLD = Cfg::PSLD;
  constexpr int QTS = Cfg::QTS;
  constexpr int DSS2 = Cfg::DSS2;
  constexpr int NPU = Cfg::NPU;
  constexpr int kNScale = Cfg::kNScale;
  constexpr int PSS = Cfg::PSS;
  // O9c-2：WGMMA 模式下 Q/dO/K/V 是 SW128 tile（1024B 对齐），否则是行主序 ASLD 布局。
  constexpr int QS_SZ = WGMMA ? Cfg::qs_sw_bytes : BM * ASLD;
  constexpr int KS_SZ = WGMMA ? Cfg::ks_sw_bytes : BN * ASLD;
  static_assert(!WGMMA || (HD == 128), "WGMMA 主 kernel 目前只做 HD=128");
  // GEMM3/4/5 的输出 N 维 = head_dim；每遍处理 NTW = WN*64 = 128 列，共 HD/NTW 遍。
  constexpr int NTW = WN * 64;
  static_assert(HD % NTW == 0, "HD 必须是 128 的整数倍");
  // O21：主 kernel 的 KV tile BN 参数化（原写死 BN=32）。4 个 warp 按 2×2 排布：
  //   GEMM1/2 的 warp 块 = (BM/2, BN/2)；GEMM3/4 输出 M=BN（KV 行），warp 块 = (BN/2, NTW)；
  //   GEMM5 输出 M=BM，warp 块 = (BM/2, NTW)。这里把 mma 累加器的 m/n-tile 数派生出来。
  constexpr int MTM = (BM / 2) / 16;   // GEMM1/2 每 warp 的 m-tile 数（BM=64→2）
  constexpr int MTN = (BN / 2) / 8;    // GEMM1/2 每 warp 的 n-tile 数（BN=32→2，BN=64→4）
  constexpr int MTM34 = (BN / 2) / 16; // GEMM3/4 每 warp 的 m-tile 数（BN=32→1，BN=64→2）
  constexpr int NTFOLD = BN / 32;      // fold 时每 warp 负责的 j 行份数（BN=32→1，BN=64→2）
  // O7：本实例是否把 dQ 沿 nt 累加在寄存器里（见 kernel 上方的说明）。
  constexpr bool kRegDq = REGDQ && (HD / NTW == 1);
  // O7e：`REGDQ` 的 `dqacc[2][8][4]`（64 个 fp32）把寄存器预算占满，再叠上 O3 的
  //   寄存器预取（pk0/pk1/pv0/pv1，16 个 uint32）会让 ptxas 强制 spill（ncu：局部内存
  //   占 L1TEX sector 的 ~12%、Est. 27.6%）。这里在 REGDQ 生效时**关掉 O3 预取**，退回
  //   O4b 的 4B 向量化同步读（K/V 只占一小段，延迟被 5 个 GEMM 盖住），把寄存器让给 dqacc。
  //   实测（S=4096 ksplit=4）prefetch off 2.55→2.52ms；小 S（use_regdq=false）维持预取。
  // O21：BN=64 时 NPU=8（预取 32 regs），此时 kernel 用 `__launch_bounds__(...,2)`（2 CTA/SM）
  //   寄存器预算更宽，恢复 O3 预取以盖住 K/V 全局延迟。
  constexpr bool kPrefetch = (NPU * 4 <= (BN > 32 ? 32 : 16)) && !kRegDq;

  extern __shared__ __align__(16) char smem[];
  // WGMMA 的 SW128 描述符要求 tile 1024B 对齐（base_offset=0）→ 手动对齐动态 smem 基址
  // （宿主为 WGMMA 模式多给 1024B slack，见 `Fp8Cfg::smem_bytes_wgmma`）。
  char* base = smem;
  if constexpr (WGMMA) {
    const uint32_t a0 = smem_u32(smem);
    base = smem + ((1024u - (a0 & 1023u)) & 1023u);
  }
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(base);
  unsigned char* Ks  = Qs + QS_SZ;
  unsigned char* Vs  = Ks + KS_SZ;
  unsigned char* dOs = Vs + KS_SZ;
  // O4b：Kt/Qt/dOt 三个转置副本 -> Qp/dOp/Kp 三个 K 配对布局（uint16，行距 PSLD）。
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(dOs + QS_SZ);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + Cfg::qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + Cfg::qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + Cfg::kp_bytes;
  unsigned char* Ap  = Vs;                   // O2：复用 GEMM2 后死亡的 Vs
  unsigned char* dS3 = Ks;                   // O2：复用 GEMM1 后死亡的 Ks
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
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

  // ---- 载入 Q/dO：原始行 [m][d] 写 Qs/dOs（供 GEMM1/2 的 A）+ 打包行对写 Qp/dOp
  //      （供 GEMM4/3 的 B，ldmatrix.trans）。行对用 __byte_perm 交织，4B 一次。----
  {
    const int nd4 = HD / 4;
    for (int u = tid; u < (BM / 2) * nd4; u += THREADS) {
      int rp = u / nd4, dq = (u % nd4) * 4;
      int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
      uint32_t q0 = 0, q1 = 0, o0 = 0, o1 = 0;
      if (qa < S) {
        size_t idx = (((size_t)(b * S + qa)) * H + h) * HD + dq;
        q0 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o0 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      if (qb < S) {
        size_t idx = (((size_t)(b * S + qb)) * H + h) * HD + dq;
        q1 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o1 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      if constexpr (WGMMA) {  // SW128：行对同一 dq 落 4B（dq 是 4 的倍数，4B 对齐）
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD)) = q0;
        *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = q1;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD)) = o0;
        *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = o1;
      } else {
        *reinterpret_cast<uint32_t*>(Qs + (rp * 2) * ASLD + dq) = q0;
        *reinterpret_cast<uint32_t*>(Qs + (rp * 2 + 1) * ASLD + dq) = q1;
        *reinterpret_cast<uint32_t*>(dOs + (rp * 2) * ASLD + dq) = o0;
        *reinterpret_cast<uint32_t*>(dOs + (rp * 2 + 1) * ASLD + dq) = o1;
      }
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
  }
  if (tid < BM) {
    int qi = m0 + tid;
    qs_s[tid] = (qi < S) ? qs[((size_t)(b * S + qi)) * H + h] : 1.f;
    dos_s[tid] = (qi < S) ? dos[((size_t)(b * S + qi)) * H + h] : 1.f;
  }

  // ---- O3 prologue：寄存器预取本 part 首个 tile 并落盘；HD>128 时直接向量化读。----
  // ---- O4a：Q/dO 载入与 tile 的 K/V 落盘写的是互不重叠的 smem（Qs/Qp/dOs/dOp vs
  //            Ks/Vs/Kp），故把原来 prologue 的两处 __syncthreads 合并为一处。----
  uint32_t pk0[NPU], pk1[NPU], pv0[NPU], pv1[NPU];
  if (kPrefetch) {
    kv_prefetch_pair<NPU, HD>(k8, v8, nt_begin * BN, S, Hkv, hkv, b, tid, pk0, pk1, pv0,
                              pv1);
    kv_commit_pair<NPU, HD, WGMMA>(Ks, Vs, Kp, pk0, pk1, pv0, pv1, tid, ASLD, PSLD);
  } else {
    kv_load_pair<HD, BN, WGMMA>(k8, v8, nt_begin * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kp, ASLD,
                         PSLD);
  }
  if (tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
  }
  __syncthreads();

  // ---- O12（O7c-PREL for fp8）：把本线程负责的行的 LSE/D 预装进寄存器。----
  // lse/delta 只依赖 (m0+r, h)，与 nt tile 无关。mma 路径每线程最多 4 个行槽
  // （r = wr*32 + i*16 + g + (q>=2?8:0)，i∈{0,1}）；wgmma.m64n32 路径每线程 2 个行槽
  // （r = wid*16 + g + (q>=2?8:0)）。装进 `lse_r[4]/del_r[4]`，索引 (i*2 + (q>=2))，
  // wgmma 路径用前 2 个。数值与原逐元素 global 读逐位相同（同一地址、同一值）。
  float lse_r[4], del_r[4];
  if constexpr (PREL) {
#ifdef FA_WGMMA
    if constexpr (WGMMA) {
#pragma unroll
      for (int t = 0; t < 2; ++t) {
        const int r = wid * 16 + g + (t ? 8 : 0);
        const int qi = m0 + r;
        const size_t idx = ((size_t)(b * S + qi)) * H + h;
        const bool ok = qi < S;
        lse_r[t] = ok ? lse[idx] : 0.f;
        del_r[t] = ok ? delta[idx] : 0.f;
      }
      lse_r[2] = lse_r[0];
      lse_r[3] = lse_r[1];
      del_r[2] = del_r[0];
      del_r[3] = del_r[1];
    } else
#endif
    {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int s = 0; s < 2; ++s) {
          const int r = wr * 32 + i * 16 + g + (s ? 8 : 0);
          const int qi = m0 + r;
          const size_t idx = ((size_t)(b * S + qi)) * H + h;
          const bool ok = qi < S;
          lse_r[i * 2 + s] = ok ? lse[idx] : 0.f;
          del_r[i * 2 + s] = ok ? delta[idx] : 0.f;
        }
    }
  }

  // ---- O7：dQ 在寄存器里沿 nt 累加，**每个 CTA 只 flush 一次**。----
  // 现状（O4c 后）：dQ 的 epilogue 每个 nt 都对本 CTA 的 dQ tile 做一次跨 CTA
  // `atomicAdd`，而 CTA 内同一线程在不同 nt 上写的是**完全相同的 (r,c) 地址**
  // （GEMM5 的 warp/累加器映射与 nt 无关）→ 同一元素被 RMW 了 ntiles 次。
  // 这里把 mma 结果就地折算（`sds2[r]*scale`）后累进寄存器 `dqacc`，nt 循环结束后
  // 每元素只发一次 `atomicAdd`，把 dQ 的全局归约指令数从 O(ntiles) 降到 O(1)。
  //   * 仅 HD==128（`HD/NTW==1`，dQ 的 N 维一次铺满）时启用；HD=512 的 4 个 N-tile
  //     需要 4 份独立累加器（4×64=256 regs）不划算，仍走原 per-tile 归约。
  //   * 因为折算因子 `sds2[r]` 逐 tile 变化，必须先折算再累加（不能累加裸 mma 输出）。
  float dqacc[kRegDq ? 2 : 1][8][4];
  if constexpr (kRegDq) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;
  }

  for (int nt = nt_begin; nt < nt_end; ++nt) {
    const int j0 = nt * BN;
    // ---- O3：预取下一 tile 的 K/V 到寄存器（延迟被本轮 5 个 GEMM 覆盖）----
    const int nnt = nt + 1;
    if (kPrefetch && nnt < nt_end)
      kv_prefetch_pair<NPU, HD>(k8, v8, nnt * BN, S, Hkv, hkv, b, tid, pk0, pk1, pv0,
                                pv1);

    // ---- (1) S = scale·QKᵀ  →  P = exp(S − LSE)，存 fp32 ----
    // ---- (2) dP = dO·Vᵀ    →  dS = P∘(dP − D)，存 fp32 ----
#ifdef FA_WGMMA
    if constexpr (WGMMA) {
      // O9c-2：GEMM1/2 换 wgmma.m64n32k32（A/B 都从 SW128 直读 smem）。两条异步 mma 一起
      //   发、统一 wait0 重叠；累加器 wgmma 映射下 warp w 持行 [16w,16w+16)，同一线程在
      //   两条 GEMM 里拥有同一个 (r,c) ⇒ dS 直接用寄存器里的 P（pval）即可，无需读回 smem。
      float sacc[16], dpacc[16];
      wgmma_mn32_issue<0>(reinterpret_cast<const char*>(Qs),
                          reinterpret_cast<const char*>(Ks), HD, sacc);
      wgmma_mn32_issue<1>(reinterpret_cast<const char*>(dOs),
                          reinterpret_cast<const char*>(Vs), HD, dpacc);
      wgmma_wait0_fp8();
      float pval[16];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi)) {
            float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            float lv = 0.f;
            if constexpr (PREL) lv = lse_r[q >= 2 ? 1 : 0];
            else lv = lse[((size_t)(b * S + qi)) * H + h];
            p = fexp(sval - lv);
          }
          pval[j * 4 + q] = p;
          Ps[r * PSS + c] = p;
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r;
          float dpv = dpacc[j * 4 + q] * dos_s[r] * vs_s[c];
          float del = 0.f;
          if constexpr (PREL) del = del_r[q >= 2 ? 1 : 0];
          else if (qi < S) del = delta[((size_t)(b * S + qi)) * H + h];
          Ss[r * PSS + c] = pval[j * 4 + q] * (dpv - del);
        }
    } else
#endif
    {
#if FA_FUSE_EPI
      // O20：P 留在寄存器，供 GEMM2 epilogue 直接用，免去 Ps 的 smem 回读。
      float preg[MTM][MTN][4];
#endif
#if FA_ILV && FA_FUSE_EPI
      // ---- O22：GEMM1/GEMM2 指令级交错：两条独立 GEMM 的 mma 先连发（各自累加器），
      //      再做 epilogue。GEMM2 只读 dOs/Vs（本 tile 已就绪），与 GEMM1 的 epilogue 无依赖，
      //      故顺序交换在数学上等价、数值逐位不变（fp32 累加次序未动）。----
      {
        float acc[MTM][MTN][4], acc2[MTM][MTN][4];
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) { acc[i][j][q] = 0.f; acc2[i][j][q] = 0.f; }
        mma_block<BM / 2, BN / 2, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wr, wc, lane);
        mma_block<BM / 2, BN / 2, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc2, wr, wc, lane);
        const int r0 = wr * (BM / 2), c0 = wc * (BN / 2);
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2 + (q & 1);
              int qi = m0 + r, jg = j0 + c;
              float p = 0.f;
              if (qi < S && jg < S && !(causal && jg > qi)) {
                float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
                float lv = 0.f;
                if constexpr (PREL) lv = lse_r[i * 2 + (q >= 2 ? 1 : 0)];
                else lv = lse[((size_t)(b * S + qi)) * H + h];
                p = fexp(sval - lv);
              }
              Ps[r * PSS + c] = p;
              preg[i][j][q] = p;
            }
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2 + (q & 1);
              int qi = m0 + r;
              float dpv = acc2[i][j][q] * dos_s[r] * vs_s[c];
              float del = 0.f;
              if constexpr (PREL) del = del_r[i * 2 + (q >= 2 ? 1 : 0)];
              else if (qi < S) del = delta[((size_t)(b * S + qi)) * H + h];
              Ss[r * PSS + c] = preg[i][j][q] * (dpv - del);
            }
      }
#else
      float acc[MTM][MTN][4];
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<BM / 2, BN / 2, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wr, wc, lane);
      const int r0 = wr * (BM / 2), c0 = wc * (BN / 2);
#pragma unroll
      for (int i = 0; i < MTM; ++i)
#pragma unroll
        for (int j = 0; j < MTN; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r, jg = j0 + c;
            float p = 0.f;
            if (qi < S && jg < S && !(causal && jg > qi)) {
              float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
              float lv = 0.f;
              if constexpr (PREL) lv = lse_r[i * 2 + (q >= 2 ? 1 : 0)];
              else lv = lse[((size_t)(b * S + qi)) * H + h];
              p = fexp(sval - lv);
            }
            Ps[r * PSS + c] = p;
#if FA_FUSE_EPI
            preg[i][j][q] = p;
#endif
          }
      // ---- (2) dP = dO·Vᵀ  →  dS = P∘(dP − D)，存 fp32 ----
      // ---- O4a：GEMM1 与 GEMM2 之间**不需要** barrier。GEMM2 只读 dOs/Vs（本 tile 前已
      //      就绪），其 epilogue 读回的 Ps[r*PSS+c] 正是**本线程**刚写入的同一地址（两次
      //      mma_block 的 (wm=wr, wn=wc) 与累加器映射完全一致），无线程间依赖。----
      {
        float acc[MTM][MTN][4];
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block<BM / 2, BN / 2, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc, wr, wc, lane);
        const int r0 = wr * (BM / 2), c0 = wc * (BN / 2);
#pragma unroll
        for (int i = 0; i < MTM; ++i)
#pragma unroll
          for (int j = 0; j < MTN; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2 + (q & 1);
              int qi = m0 + r;
              float dpv = acc[i][j][q] * dos_s[r] * vs_s[c];
              float del = 0.f;
              if constexpr (PREL) del = del_r[i * 2 + (q >= 2 ? 1 : 0)];
              else if (qi < S) del = delta[((size_t)(b * S + qi)) * H + h];
#if FA_FUSE_EPI
              // O20：用寄存器里的 P（本线程刚算的同一 (r,c)），不再读回 Ps。
              Ss[r * PSS + c] = preg[i][j][q] * (dpv - del);
#else
              Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);
#endif
            }
      }
#endif  // FA_ILV
    }  // end else (mma path)
    __syncthreads();

    // ---- 构造 dV/dK/dQ 的 fp8 操作数（fold 归约维上的 rowwise scale）----
    // O4a：原来 fold 只让 `tid<BN`（32 线程）算 Ap/dS3、`tid<BM`（64 线程）算 dS2，其余
    // 线程空等；现改为**全部 128 线程均衡分工**：warp 内 4 lane 一组做一行 amax 的
    // `__shfl_xor_sync` 归约。fmaxf 可交换结合 ⇒ amax 结果与原顺序**逐位相同**，除法/量化
    // 也逐元素一致，故数值仍与 O3 逐位相同；但这一段的墙钟缩短约 4×（Amax/写入都并行）。
    {
      // Ap[j][m]=P[m][j]*dos[m] (e4m3, per-j) 与 dS3[j][m]=dS[m][j]*qs[m] (e5m2, per-j)
      // O21：BN 参数化——每 warp 负责 NTFOLD 份 j 行（每份 8 行），j = wid*8 + jl + jh*32。
      const int jl = lane >> 2, sub4 = lane & 3;  // 每 warp 8 行 × 4 lane 分工
      // O7e-2：把每个 lane 负责的 16 个 m 从「一整块 `sub4*16+t`」改成「两半 `sub4*8+t+
      //   half*32`」。原因是 `Ps[m*PSS+j]` 的 bank=(m*PSS+j) mod32=(m+j) mod32（PSS=33）
      //   在原映射下 sub4=0/2、1/3 的起始 m 相差 16 ⇒ bank 恒撞（16*PSS≡16 mod32），
      //   ncu 实测 fold 读是 shared load 2.1-way conflict（占 load 波前 32%）的主要来源。
      //   改成起始差 8 后 4 组 lane 的 bank 恰为 {0,8,16,24}+{0..7} = 0..31 各一次 ⇒ 无冲突。
      //   amax 的 `fmaxf` 可交换结合 ⇒ 归约结果与原映射逐位相同（数值不变）。
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh) {
        const int j = wid * 8 + jl + jh * 32;
        float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
        for (int half = 0; half < 2; ++half)
#pragma unroll
          for (int t = 0; t < 8; ++t) {
            int m = F16B ? (sub4 * 8 + t + half * 32) : (sub4 * 16 + half * 8 + t);
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
        // O7e：把逐 1B 的 `st.shared.u8` 折成向量写。`F16B=true` 用新映射（每 lane 两段
        //   各 8 个连续 m）⇒ 每段一次 8B `st.shared.v2.u32`；`false` 退回原映射的 4×4B。
        //   QTS=BM+16=80（16 与 8 的倍数）、数组基址 16B 对齐 ⇒ 合法。数值逐位不变。
        if constexpr (F16B) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            uint32_t pa2[2] = {0, 0}, d32[2] = {0, 0};
#pragma unroll
            for (int t4 = 0; t4 < 2; ++t4) {
#pragma unroll
              for (int tt = 0; tt < 4; ++tt) {
                int m = sub4 * 8 + t4 * 4 + tt + half * 32;
                pa2[t4] |= (uint32_t)cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA) << (8 * tt);
                d32[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3) << (8 * tt);
              }
            }
            *reinterpret_cast<uint2*>(Ap + j * QTS + sub4 * 8 + half * 32) =
                make_uint2(pa2[0], pa2[1]);
            *reinterpret_cast<uint2*>(dS3 + j * QTS + sub4 * 8 + half * 32) =
                make_uint2(d32[0], d32[1]);
          }
        } else {
          uint32_t pa4[4] = {0, 0, 0, 0}, d34[4] = {0, 0, 0, 0};
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
#pragma unroll
            for (int tt = 0; tt < 4; ++tt) {
              int m = sub4 * 16 + t4 * 4 + tt;
              pa4[t4] |= (uint32_t)cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA) << (8 * tt);
              d34[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3) << (8 * tt);
            }
          }
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            *reinterpret_cast<uint32_t*>(Ap + j * QTS + sub4 * 16 + t4 * 4) = pa4[t4];
            *reinterpret_cast<uint32_t*>(dS3 + j * QTS + sub4 * 16 + t4 * 4) = d34[t4];
          }
        }
      }
    }
    {
      // dS2[m][j] = dS[m][j]*ks[j] (e5m2, per-m)：每 warp 16 行 × 2 lane 分工。
      // O21：BN 参数化——amax2 需覆盖全部 BN 个 j（NTFOLD 份，j 跨 jh*32），
      //   dS2 的 16B 向量写每份落在 `sub2*16 + jh*32`。
      const int ml = lane >> 1, sub2 = lane & 1;
      const int m = wid * 16 + ml;
      float amax2 = 0.f;
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh)
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          int j = sub2 * 16 + t + jh * 32;
          amax2 = fmaxf(amax2, fabsf(Ss[m * PSS + j] * ks_s[j]));
        }
      amax2 = fmaxf(amax2, __shfl_xor_sync(0xffffffffu, amax2, 1));
      float sc2 = (amax2 > 0.f) ? amax2 / kE5M2Max : 1.f;
      if (sub2 == 0) sds2[m] = sc2;
      sc2 = __shfl_sync(0xffffffffu, sc2, ml * 2);
      // O7e：`j = sub2*16+t` 连续 ⇒ dS2[m][j] 的 16 个 j 也连续，同样折成 4 次 4B 写。
      // O7e-2：16 个连续 j 一次 16B `st.shared.v4.u32`（DSS2 是 16 的倍数、基址对齐）。
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh) {
        uint32_t d2_4[4] = {0, 0, 0, 0};
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
#pragma unroll
          for (int tt = 0; tt < 4; ++tt) {
            int j = sub2 * 16 + t4 * 4 + tt + jh * 32;
            d2_4[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * ks_s[j] / sc2) << (8 * tt);
          }
        }
        if constexpr (F16B) {
          *reinterpret_cast<uint4*>(dS2 + m * DSS2 + sub2 * 16 + jh * 32) =
              make_uint4(d2_4[0], d2_4[1], d2_4[2], d2_4[3]);
        } else {
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            *reinterpret_cast<uint32_t*>(dS2 + m * DSS2 + sub2 * 16 + jh * 32 + t4 * 4) =
                d2_4[t4];
          }
        }
      }
    }
    __syncthreads();

    // ---- (3)(4)(5) 沿 head_dim 的 N-tile 循环。GEMM3/4/5 的输出宽度是 HD，
    //      每遍覆盖 NTW=128 列：B 操作数（dOp/Qp/Kp 配对布局）列偏移加 d0（nb0）。
    //      HD=128 时只跑 1 遍（与 O4a 逐位相同）。----
#pragma unroll
    for (int nd = 0; nd < HD / NTW; ++nd) {
      const int d0 = nd * NTW;

      // ---- (3) dV = Pᵀ·dO  : A=Ap[j][m] (e4m3), B=dOp[m/2][d0+..] (e5m2, ldmatrix.trans) ----
      {
        float acc[MTM34][8][4];
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block_bt<BN / 2, 64, BM, E4E5>(Ap, QTS, dOp, PSLD, acc, wr, wc, lane, d0);
        const int r0 = wr * (BN / 2), c0 = wc * 64;
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2;
              int jg = j0 + r;
              if (jg < S)
                red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d0 + c,
                         acc[i][j][q] * sA[r], acc[i][j][q + 1] * sA[r]);
            }
      }

      // ---- (4) dK = scale·dSᵀ·Q : A=dS3[j][m] (e5m2), B=Qp[m/2][d0+..] (e4m3, ldmatrix.trans) ----
      {
        float acc[MTM34][8][4];
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block_bt<BN / 2, 64, BM, E5E4>(dS3, QTS, Qp, PSLD, acc, wr, wc, lane, d0);
        const int r0 = wr * (BN / 2), c0 = wc * 64;
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; q += 2) {
              // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2;
              int jg = j0 + r;
              if (jg < S)
                red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + d0 + c,
                         acc[i][j][q] * sds3[r] * scale,
                         acc[i][j][q + 1] * sds3[r] * scale);
            }
      }

      // ---- (5) dQ += scale·dS·K : A=dS2[m][j] (e5m2), B=Kp[j/2][d0+..] (e4m3, ldmatrix.trans) ----
      {
        float acc[2][8][4];
#pragma unroll
        for (int i = 0; i < 2; ++i)
#pragma unroll
          for (int j = 0; j < 8; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block_bt<32, 64, BN, E5E4>(dS2, DSS2, Kp, PSLD, acc, wr, wc, lane, d0);
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
              if constexpr (kRegDq) {
                // O7：折算后累进寄存器，nt 结束后统一 flush（见下方）。
                dqacc[i][j][q] += acc[i][j][q] * sds2[r] * scale;
                dqacc[i][j][q + 1] += acc[i][j][q + 1] * sds2[r] * scale;
              } else {
                int qi = m0 + r;
                if (qi < S)
                  red_add2(dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + d0 + c,
                           acc[i][j][q] * sds2[r] * scale,
                           acc[i][j][q + 1] * sds2[r] * scale);
              }
            }
      }
    }
    __syncthreads();
    // ---- O3：落盘预取的下一 tile 的 K/V（本轮 GEMM 已全部读完 smem），并更新 ks/vs ----
    if (nt + 1 < nt_end) {
      if (kPrefetch) {
        kv_commit_pair<NPU, HD, WGMMA>(Ks, Vs, Kp, pk0, pk1, pv0, pv1, tid, ASLD, PSLD);
      } else {
        kv_load_pair<HD, BN, WGMMA>(k8, v8, (nt + 1) * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kp, ASLD,
                             PSLD);
      }
      if (tid < BN) {
        int jg = (nt + 1) * BN + tid;
        ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
        vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
      }
      __syncthreads();
    }
  }

  // ---- O7：把寄存器里累加的 dQ flush 出去（每 CTA 每元素一次 `red_add2`）。----
  if constexpr (kRegDq) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = wr * 32 + i * 16 + g + (q >= 2 ? 8 : 0);
          int c = wc * 64 + j * 8 + c2;
          int qi = m0 + r;
          if (qi < S)
            red_add2(dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c, dqacc[i][j][q],
                     dqacc[i][j][q + 1]);
        }
  }
}

// =============================================================================
// 3b) fa_bwd_fp8_wg2_kernel —— fp8 跨 warpgroup 归约（BM=128, 2 warpgroups, 256 线程）
// =============================================================================
// 动机（对齐 fp16 O17）：fp8 main 的 `dK/dV` 用跨 CTA 的 `atomicAdd` 汇总，一个 KV 元素被
//   `S/BM` 个 CTA（每个 query 块一个）贡献。BM 64→128 后贡献 CTA 数减半 ⇒ red 字节砍半，
//   L2 原子压力随之减半（O17 在 fp16/bf16 上实测 red 精确减半、main 1.5×）。
// 结构：256 线程 = 2 个 warpgroup；每个 wg 算自己 64 行 Q 的 S/dP（几何与 BM=64 版逐字同构），
//   把 P/dS 写进 smem；fold 后 GEMM3/4/5 的重叠维是**全 BM=128**（直接从 smem 读），
//   每个输出元素只被本 CTA 的一个 warp `red` 一次 ⇒ 跨 CTA red 减半。
// 与已有 `fa_bwd_fp8_mma_kernel` 逐项对应，只是几何从 (4 warp, BM=64) 变成 (8 warp, BM=128)。
template <int HD, int BM = 128, int BN = 32>
__global__ void __launch_bounds__(256, 1)
fa_bwd_fp8_wg2_kernel(const unsigned char* __restrict__ q8,
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
  static_assert(HD == 128, "wg2 目前只做 HD=128");
  constexpr int TH = 256;
  constexpr int ASLD = HD + 16;    // 144
  constexpr int PSLD = HD + 8;     // 136
  constexpr int QTS = BM + 16;     // 144
  constexpr int DSS2 = BN + 16;    // 48
  constexpr int PSS = BN + 5;      // 37（O7e-3：与 mma 版一致，fold 掩码读无冲突）
  constexpr int qp_bytes = (BM / 2) * PSLD * 2;
  constexpr int kp_bytes = (BN / 2) * PSLD * 2;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int NTW = 64;          // 每条 mma 的 N 方向 64 列（GEMM3/4 的 WARP_N，GEMM5 的 WARP_N）
  static_assert(HD % 4 == 0, "HD 必须是 4 的倍数");
  static_assert(HD == 128, "目前只实例化 HD=128");

  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks  = Qs + BM * ASLD;
  unsigned char* Vs  = Ks + BN * ASLD;
  unsigned char* dOs = Vs + BN * ASLD;
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(dOs + BM * ASLD);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + kp_bytes;
  unsigned char* Ap  = Vs;         // O2：复用 GEMM2 后死亡的 Vs
  unsigned char* dS3 = Ks;         // O2：复用 GEMM1 后死亡的 Ks
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
  float* Ps = scales + kNScale;
  float* Ss = Ps + BM * PSS;
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;

  const int mblk = blockIdx.x / ksplit, part = blockIdx.x % ksplit;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wg = wid >> 2;              // 0/1：两个 warpgroup
  const int wl = wid & 3;               // warpgroup 内 warp 号
  const int wr = wl >> 1, wc = wl & 1;  // phase A：wg 内 2×2 warp 几何
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ksplit;
  const int nt_end = (part + 1) * ntiles / ksplit;
  if (nt_end <= nt_begin) return;

  // ---- 载入 Q/dO（行 [m][d] 写 Qs/dOs；行对打包写 Qp/dOp 供 GEMM4/3 的 B）----
  {
    const int nd4 = HD / 4;
    for (int u = tid; u < (BM / 2) * nd4; u += TH) {
      int rp = u / nd4, dq = (u % nd4) * 4;
      int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
      uint32_t q0 = 0, q1 = 0, o0 = 0, o1 = 0;
      if (qa < S) {
        size_t idx = (((size_t)(b * S + qa)) * H + h) * HD + dq;
        q0 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o0 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      if (qb < S) {
        size_t idx = (((size_t)(b * S + qb)) * H + h) * HD + dq;
        q1 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o1 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      *reinterpret_cast<uint32_t*>(Qs + (rp * 2) * ASLD + dq) = q0;
      *reinterpret_cast<uint32_t*>(Qs + (rp * 2 + 1) * ASLD + dq) = q1;
      *reinterpret_cast<uint32_t*>(dOs + (rp * 2) * ASLD + dq) = o0;
      *reinterpret_cast<uint32_t*>(dOs + (rp * 2 + 1) * ASLD + dq) = o1;
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
  }
  if (tid < BM) {
    int qi = m0 + tid;
    qs_s[tid] = (qi < S) ? qs[((size_t)(b * S + qi)) * H + h] : 1.f;
    dos_s[tid] = (qi < S) ? dos[((size_t)(b * S + qi)) * H + h] : 1.f;
  }
  // 首个 tile 的 K/V（直接向量化读，无寄存器预取）
  kv_load_pair_nt<HD, BN, TH>(k8, v8, nt_begin * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kp, ASLD,
                              PSLD);
  if (tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
  }
  __syncthreads();

  // ---- PREL：本线程负责行的 LSE/D 预装寄存器。phase A 的 wm = wg*2+wr（0..3），
  //      行槽 r = wm*32 + i*16 + g + (s?8:0)。----
  float lse_r[4], del_r[4];
  {
    const int wm = wg * 2 + wr;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int s = 0; s < 2; ++s) {
        const int r = wm * 32 + i * 16 + g + (s ? 8 : 0);
        const int qi = m0 + r;
        const size_t idx = ((size_t)(b * S + qi)) * H + h;
        const bool ok = qi < S;
        lse_r[i * 2 + s] = ok ? lse[idx] : 0.f;
        del_r[i * 2 + s] = ok ? delta[idx] : 0.f;
      }
  }

  // ---- dQ 寄存器累加（O7）：GEMM5 每 warp 32 行 × 64 列 = 2×8×4。----
  float dqacc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

  for (int nt = nt_begin; nt < nt_end; ++nt) {
    const int j0 = nt * BN;
    // ---- (1) S = scale·QKᵀ → P = exp(S−LSE)； (2) dP = dO·Vᵀ → dS = P∘(dP−D) ----
    {
      const int wm = wg * 2 + wr;
      float acc[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      mma_block<32, 16, HD, E4E4>(Qs, ASLD, Ks, ASLD, acc, wm, wc, lane);
      const int r0 = wm * 32, c0 = wc * 16;
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
              p = fexp(sval - lse_r[i * 2 + (q >= 2 ? 1 : 0)]);
            }
            Ps[r * PSS + c] = p;
          }
      // GEMM2：无需 barrier（读回本线程自己写的 Ps 同址）
      float acc2[2][2][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc2[i][j][q] = 0.f;
      mma_block<32, 16, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc2, wm, wc, lane);
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2 + (q & 1);
            int qi = m0 + r;
            float dpv = acc2[i][j][q] * dos_s[r] * vs_s[c];
            float del = del_r[i * 2 + (q >= 2 ? 1 : 0)];
            if (qi >= S) del = 0.f;
            Ss[r * PSS + c] = Ps[r * PSS + c] * (dpv - del);
          }
    }
    __syncthreads();

    // ---- fold：把 Ps/Ss（全 BM=128 行）量化成 GEMM3/4/5 的操作数并乘 rowwise scale ----
    {
      // Ap[j][m]=P[m][j]*dos[m] (e4m3)，dS3[j][m]=dS[m][j]*qs[m] (e5m2)：每 warp 4 个 j，
      //   8 个 lane 一组沿 m（每组 16 个 m）。
      const int jj = lane >> 3, sub = lane & 7;
      const int j = wid * 4 + jj;   // 0..31
      float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
      for (int t = 0; t < 16; ++t) {
        int m = sub * 16 + t;
        amaxA = fmaxf(amaxA, fabsf(Ps[m * PSS + j] * dos_s[m]));
        amax3 = fmaxf(amax3, fabsf(Ss[m * PSS + j] * qs_s[m]));
      }
      amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 1));
      amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 2));
      amaxA = fmaxf(amaxA, __shfl_xor_sync(0xffffffffu, amaxA, 4));
      amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 1));
      amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 2));
      amax3 = fmaxf(amax3, __shfl_xor_sync(0xffffffffu, amax3, 4));
      float scA = (amaxA > 0.f) ? amaxA / kE4M3Max : 1.f;
      float sc3 = (amax3 > 0.f) ? amax3 / kE5M2Max : 1.f;
      if (sub == 0) { sA[j] = scA; sds3[j] = sc3; }
      scA = __shfl_sync(0xffffffffu, scA, jj * 8);
      sc3 = __shfl_sync(0xffffffffu, sc3, jj * 8);
      uint32_t pa[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
#pragma unroll
      for (int t4 = 0; t4 < 4; ++t4)
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          int m = sub * 16 + t4 * 4 + tt;
          pa[t4] |= (uint32_t)cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA) << (8 * tt);
          d3[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3) << (8 * tt);
        }
      *reinterpret_cast<uint4*>(Ap + j * QTS + sub * 16) = make_uint4(pa[0], pa[1], pa[2], pa[3]);
      *reinterpret_cast<uint4*>(dS3 + j * QTS + sub * 16) = make_uint4(d3[0], d3[1], d3[2], d3[3]);
    }
    {
      // dS2[m][j] = dS[m][j]*ks[j] (e5m2)：每 warp 16 行 × 2 lane；8 warp 覆盖 128 行。
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
      uint32_t d2[4] = {0, 0, 0, 0};
#pragma unroll
      for (int t4 = 0; t4 < 4; ++t4)
#pragma unroll
        for (int tt = 0; tt < 4; ++tt) {
          int j = sub2 * 16 + t4 * 4 + tt;
          d2[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * ks_s[j] / sc2) << (8 * tt);
        }
      *reinterpret_cast<uint4*>(dS2 + m * DSS2 + sub2 * 16) = make_uint4(d2[0], d2[1], d2[2], d2[3]);
    }
    __syncthreads();

    // ---- (3) dV = Pᵀ·dO：A=Ap[j][m]（全 BM=128 重叠），B=dOp 配对布局 ----
    {
      float acc[1][4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block_bt<16, 32, BM, E4E5>(Ap, QTS, dOp, PSLD, acc, wg, wl, lane, 0);
      const int r0 = wg * 16, c0 = wl * 32;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2;
          int jg = j0 + r;
          if (jg < S)
            red_add2(dv_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     acc[0][j][q] * sA[r], acc[0][j][q + 1] * sA[r]);
        }
    }
    // ---- (4) dK = scale·dSᵀ·Q：A=dS3[j][m]，B=Qp 配对布局 ----
    {
      float acc[1][4][4];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block_bt<16, 32, BM, E5E4>(dS3, QTS, Qp, PSLD, acc, wg, wl, lane, 0);
      const int r0 = wg * 16, c0 = wl * 32;
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = r0 + g + (q >= 2 ? 8 : 0);
          int c = c0 + j * 8 + c2;
          int jg = j0 + r;
          if (jg < S)
            red_add2(dk_acc + (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c,
                     acc[0][j][q] * sds3[r] * scale, acc[0][j][q + 1] * sds3[r] * scale);
        }
    }
    // ---- (5) dQ += scale·dS·K：A=dS2[m][j]（全 BM=128 行），B=Kp 配对布局 ----
    {
      float acc[2][8][4];
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      const int wm = wid >> 1, wn = wid & 1;
      mma_block_bt<32, 64, BN, E5E4>(dS2, DSS2, Kp, PSLD, acc, wm, wn, lane, 0);
      const int r0 = wm * 32, c0 = wn * 64;
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 8; ++j)
#pragma unroll
          for (int q = 0; q < 4; q += 2) {
            int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
            int c = c0 + j * 8 + c2;
            dqacc[i][j][q] += acc[i][j][q] * sds2[r] * scale;
            dqacc[i][j][q + 1] += acc[i][j][q + 1] * sds2[r] * scale;
          }
    }
    __syncthreads();
    // 载入下一 tile 的 K/V（本轮 GEMM 已读完 smem）
    if (nt + 1 < nt_end) {
      kv_load_pair_nt<HD, BN, TH>(k8, v8, (nt + 1) * BN, S, Hkv, hkv, b, tid, Ks, Vs, Kp, ASLD,
                                  PSLD);
      if (tid < BN) {
        int jg = (nt + 1) * BN + tid;
        ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
        vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
      }
      __syncthreads();
    }
  }

  // ---- dQ flush（每 CTA 每元素一次）----
  {
    const int wm = wid >> 1, wn = wid & 1;
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = wm * 32 + i * 16 + g + (q >= 2 ? 8 : 0);
          int c = wn * 64 + j * 8 + c2;
          int qi = m0 + r;
          if (qi < S)
            red_add2(dq_acc + (((size_t)(b * S + qi)) * H + h) * HD + c, dqacc[i][j][q],
                     dqacc[i][j][q + 1]);
        }
  }
}

// =============================================================================
// 4) convert：fp32 累加缓冲 -> 输出（本版直接 fp32 拷贝）
// =============================================================================
// O14b：fp32→fp32 拷贝向量化 `float4`（旧的逐元素 grid-stride 只 ~1.4TB/s，
// 远低于 HBM 峰值）。nq/nkv 均为 4 的倍数（B·S·H·D，D∈{128,512}），尾部留标量兜底。
__global__ void convert_kernel(const float* __restrict__ dq_acc,
                               const float* __restrict__ dk_acc,
                               const float* __restrict__ dv_acc,
                               float* __restrict__ dq, float* __restrict__ dk,
                               float* __restrict__ dv, size_t nq, size_t nkv) {
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  const size_t nq4 = nq >> 2, nkv4 = nkv >> 2;
  for (size_t i = tid; i < nq4; i += stride)
    reinterpret_cast<float4*>(dq)[i] = reinterpret_cast<const float4*>(dq_acc)[i];
  for (size_t i = tid; i < nkv4; i += stride) {
    reinterpret_cast<float4*>(dk)[i] = reinterpret_cast<const float4*>(dk_acc)[i];
    reinterpret_cast<float4*>(dv)[i] = reinterpret_cast<const float4*>(dv_acc)[i];
  }
  for (size_t i = (nq4 << 2) + tid; i < nq; i += stride) dq[i] = dq_acc[i];
  for (size_t i = (nkv4 << 2) + tid; i < nkv; i += stride) {
    dk[i] = dk_acc[i];
    dv[i] = dv_acc[i];
  }
}

#endif  // FA_BWD_FP8_KERNELS_CUH_
