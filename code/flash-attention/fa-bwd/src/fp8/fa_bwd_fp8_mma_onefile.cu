// =============================================================================
// fa_bwd_fp8_mma_onefile.cu —— FlashAttention 反向（FP8）**单文件版**
// =============================================================================
// 自包含：preprocess（quantize + lse mma + delta）+ main（5 个 GEMM 全 mma.m16n8k32）
//          + convert + launcher + 自测对拍（读 npy dump）。
//
// P5-3：与两文件版 `fa_bwd_fp8_kernels.cuh`/`fa_bwd_fp8_main.cu` **device 代码逐字一致**，
// 本文件由后者拼接生成。支持 head_dim=128（MHA/GQA）与 512（MLA 主注意力，模板参数 HD）；
// `HD=128,BM=64,BN=32` 时数值与 P3-5/O1–O4c 逐位相同。device 侧设计说明见 kernels.cuh 顶部注释。
// O4b：GEMM3/4/5 的 B 操作数改「K 配对布局 + `ldmatrix.x2.trans`」（`Kt/Qt/dOt` → `Kp/Qp/dOp`）。
// O7：dQ 沿 nt 循环在**寄存器**里累加（折算后），每 CTA 只 flush 一次跨 CTA `atomicAdd`；
//
// 用法：run.sh src/fp8/fa_bwd_fp8_mma_onefile.cu [--dir=...] [--full|--causal] [--o=...] [--iters=N]
// =============================================================================

#include "../fa_bwd_dump.h"   // P3-3：ours 输出落 npy，供 harness/fa_bwd_compare.py
#include <cuda_runtime.h>
#include <cuda.h>  // O32：LSE 的 4D TMA 需要驱动 API（cuTensorMapEncodeTiled / CUtensorMap）
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

// O81（F3b 主体，第 176 轮）：把 GEMM5（dQ）从 `mma.m16n8k32` 换成 **wgmma RS**
//   （A=dS2 寄存器，B=Kᵀ no-swizzle 描述符）。仅 KVTMA+WGMMA+HD=128+BN=32 非 DET/DQONLY
//   路径生效。**默认 1（正结果，第 176 轮）**：同 binary A/B S4096 main 1.556→1.481ms
//   （1.051×）、total 1.789→1.724ms（1.038×）；instruction −11.1%、HMMA −26.9%、L2 read −9.3%、
//   `red` 114.52M 不变、regs 168 不变；vs fp32 ref relL2 dq/dk/dv 8.15/8.26/6.49% 不回退。
//   数据通路由 `fa_bwd_fp8_wgmma345_smoke.cu` 三验证（no-swizzle 描述符 + SW128 源转置
//   + 端到端 GEMM5 全 max_abs=0）。`-DFA_WGMMA5=0` 退回原 mma GEMM5 做 A/B。
#ifndef FA_WGMMA5
#define FA_WGMMA5 1
#endif

// O82（F3b 主体续，第 177 轮）：把 GEMM3(dV)/GEMM4(dK) 从 `mma.m16n8k32` 换成 **wgmma RS**
//   （A=Ap/dS3 寄存器；B=dOᵀ/Qᵀ 紧凑 no-swizzle 描述符）。这两条 GEMM 的 M=BN=32 < wgmma 最小 m64，
//   故把 A 的 M 维**零填充到 64**（warp 2/3 的 A 寄存器置 0，其输出行 32–63 丢弃）。
//   **判决（第 177 轮，负结果）**：S4096 同 binary A/B main 1.4777→1.5494ms（**0.954×**），
//   虽 instruction −7.3%、L1/TEX 74.4→67.6%、L2 78.8→75.5%，但 Duration 反升——本 kernel 是
//   **L2 `red` bound（75%，wgmma 一字不减）+ 延迟 bound**（No Eligible 53→58%），零填充使张量核
//   做 2× 无用功，q/dO 转置与 wgmma wait 又加延迟。⇒ **默认 0**；`-DFA_WGMMA34=1` 复现 A/B。
//   仅 KVTMA+WGMMA+HD=128+BN=32 非 DET/DQONLY/ILV34/BULKRED/R4 路径生效（数值与 mma 版同量级）。
#ifndef FA_WGMMA34
#define FA_WGMMA34 0
#endif

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

  // O29：GEMM3(dV)/GEMM4(dK) 的**指令级交错**开关。默认 0：保持原顺序（GEMM3 mma →
  //   GEMM3 epilogue(red) → GEMM4 mma → GEMM4 epilogue(red)）。置 1：先连发两条独立 GEMM 的
  //   mma（acc3/acc4 两个累加器），再做各自的 epilogue——让 GEMM4 的 mma 填在 GEMM3 的依赖
  //   延迟之前，掩盖 `wait`（O7e-3/O28 一致确认 fp8 main 第一墙是 mma 依赖延迟）。
  //   数学与数值逐位不变（同一批 mma、同一 (r,c) 映射、同一次序的 fp32 累加），只改发射顺序。
  //   代价：两个 MTM34×8×4 累加器同时存活（HD=128 时各 32 个 fp32），可能挤占 3 CTA/SM 的
  //   寄存器预算；用 `-DFA_ILV34=0/1` 同 session A/B，默认 0（不改现状）。
#ifndef FA_ILV34
#define FA_ILV34 0
#endif

  // O67（第 147 轮）：dK/dV 归约从 `red.global.add.v2.f32`（8B）提升到 `.v4.f32`（16B），
  //   把 red 请求/扇区数再减半。`-DFA_R4=0/1` 同 session A/B，默认 0（不改旗舰路径）。
  //   仅作用于非 DET 的 `red_add2` 点（epi_dv/epi_dk），DET/BULKRED 路径不受影响。
#ifndef FA_R4
#define FA_R4 0
#endif

  // O114（第 208 轮）：把 dK/dV 的跨 CTA 归约**元素宽度**从 fp32 收窄到 fp16——
  //   `red.global.add.noftz.f16x2`（显式 PTX，走 RED 路径）而非 fp32 `red.global.add.v2.f32`。
  //   动因：ROADMAP「阻塞」判「red 由工作划分决定、与归约机制（atomic/TMA/宽度 v2↔v4）无关」，
  //   但前提是**元素宽度恒 fp32**。**实测判决：负结果**——冒烟在**连续地址**下扇区精确减半
  //   （3,243,520→1,621,568、1.34×），但真实 `mma.m16n8` 片段使一次 warp red 请求覆盖 8 个
  //   不同行 ⇒ 固定吃 8 扇区；fp32 float2（4 lane×8B=32B/行）已填满扇区，fp16 f16x2
  //   （4 lane×4B=16B/行）仍占满但只写一半 ⇒ **真机 `lts op_red` 一字不变、Duration 中性**
  //   （见 `docs/03` §136）。保留为 opt-in 探针 + 可复现冒烟，**默认 0**。改的是精度口径
  //   （fp16 累加），A/B 误差护栏内（`max_abs` 变化 ≤0.01）。仅作用于 `fp8_mma_body` 非 DET/
  //   非 BULKRED/非 FA_R4 的 dK/dV red 点；DET/BULKRED/MLA/varlen 逐字不变。
#ifndef FA_REDHALF
#define FA_REDHALF 0
#endif

  // O42：dK/dV 的跨 CTA 归约从「逐元素 `red_add2`」改成「per-warp smem staging +
  //   `cp.reduce.async.bulk...add.f32`」。O42 实测 dK/dV 的 red 是 fp8 main 头号成本
  //   （短路掉 main 1.60→0.94ms，天花板 1.70×），但 bulk 版因 staging 的 smem 流量 +
  //   TMA 归约延迟，实测 **1.80ms（0.89×）——负结果**。故默认 0（opt-in），
  //   `-DFA_BULKRED=1` 复现；仅 TMA+WGMMA+HD=128+BN=32 的路径生效。
#ifndef FA_BULKRED
#define FA_BULKRED 0
#endif

  // O50：WGMMA 路径的 GEMM1(S=QKᵀ)/GEMM2(dP=dO·Vᵀ) 等待拆分。原实现两条 wgmma 各自
  //   commit 后统一 `wait_group 0`，再一起做 fold（P 与 dS）。但 fold 的 **P 段只依赖
  //   GEMM1 的累加器 sacc**、dS 段才依赖 GEMM2 的 dpacc；故改成提交后先 `wait_group 1`
  //   （只等 GEMM1），用算 P 的那段 CUDA-core 工作（`fexp`/量化）**掩盖 GEMM2 的剩余执行**，
  //   算完 P 再 `wait_group 0` 取 dP。数学/数值逐位不变（同一批 wgmma、同一累加器、
  //   同一 fold 表达式，只改 wait 时机）。`-DFA_WS1=0` 退回原「单次 wait0」做 A/B。
#ifndef FA_WS1
#define FA_WS1 0
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

  // O51：MLA（D=512）主 kernel 的 K/V `cp.async` 回填流水。fp8 MLA 走 mma 主 kernel
  //   （无 TMA），原先 K/V 在每 tile 末尾**同步**载入 ⇒ ncu 头号 stall 是 `long_scoreboard`
  //   （O45：long 2.33）。这里不改数学、只改搬运：用 `cp.async.cg` 提前发起**下一 tile** 的
  //   K/V，延迟被本轮计算覆盖；tile 末尾 `wait_group 0` 后从 Ks 重建 Kp。
  //   代价：① **K 双缓冲**（`Ks` 两个 stage）⇒ 下一 tile 的 K 可在**本轮 GEMM1 之前**发起、
  //   与整轮计算重叠；② 为让 Vs 在 GEMM2 后即可覆写，把「Ap 复用 Vs、dS3 复用 Ks」两处
  //   smem 别名拆开（各给独立缓冲，于是 V 在 GEMM1/2 后即可回填）。合计多
  //   `BN*ASLD + 2*BN*QTS` 字节（MLA 下 16896+5120=22016B；207872+22016=229888 ≤ 232448，
  //   仍 1 CTA/SM）。
  static constexpr int smem_bytes_kvpipe = smem_bytes + BN * ASLD + 2 * BN * QTS;

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
  // O37：Q/dO 4D-TMA 版在 wgmma 布局后再加 64B 放两个 mbarrier（qbar/dbar）。
  // O85：其实 qbar/dbar 落在 `smem_bytes_wgmma` 的 1024B 对齐 slack 内，故 HD>128 不再额外
  //   +64（HD=256 时多这 64B 会把 2 CTA/SM 挤成 1——ncu `occupancy_limit_shared_mem=1`）。
  static constexpr int smem_bytes_wgmma_tma = smem_bytes_wgmma + (HD > 128 ? 0 : 64);
  // O41：K/V 也走 4D-TMA（roadmap「下一步候选 ①」）。为让 K 双缓冲（TMA 的异步搬运能跨
  //   tile 重叠）而不掉出 3 CTA/SM（实测本卡 3-CTA 动态 smem 上限 = 76800B），布局做两处
  //   零成本折叠：**dS3 复用当前 K stage**（K 只被 GEMM1 读，fold 之后才写 dS3）、
  //   **Ap 复用 Vs**（V 只被 GEMM2 读，fold 之后才写 Ap）。于是只多一个 K stage 的
  //   `ks_sw_bytes`（4096B）× 1 + 64B mbarrier（qbar/dbar/kbar[2]/vbar）。
  //   合计 = 70656 + 4096 + 64 = 74816B ≤ 76800 ⇒ 仍 3 CTA/SM。数值与 cp.async 版只差
  //   跨 CTA `atomicAdd` 次序（K/V 的字节完全相同，Kp 由 SW128 K tile 逐字节重建）。
  static constexpr int smem_bytes_wgmma_kvtma = smem_bytes_wgmma + ks_sw_bytes + 64;

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
  // O32：TMA 版 LSE（Q + 2×K tile + rowwise scale + 3 个 mbarrier）。fp8 一行 128B = 一个
  //   SW128 atom 的整行 ⇒ **一个 TMA 搬完整块**（不需要 fp16 的 2×K=64 chunk 拆分）。
  static constexpr int lse_smem_bytes_tma1 =
      3 * lse_tile_wgmma + (LBM + 2 * LBN) * (int)sizeof(float) + 1024 + 64;

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
// O27：fold 量化的除数折算。`RCP=true` 时用每行预算的 `inv`（乘法），否则精确除法。
template <bool RCP>
__device__ __forceinline__ float folddiv(float x, float sc, float inv) {
  if constexpr (RCP) {
    (void)sc;
    return x * inv;
  } else {
    (void)inv;
    return x / sc;
  }
}

// O28：一次转换 2 个 float→fp8（`cvt.rn.satfinite.{e4m3,e5m2}x2.f32`）。与两次标量
//   `__nv_cvt_float_to_fp8`（同 RN + SATFINITE）**逐位相同**，但硬件 `PACK_AB_MERGE_C`
//   直接把两条结果拼成 4B，省掉标量路径的 shift/OR（见 `cuda_fp8.hpp`）。fold 是纯
//   CUDA-core 段（夹在 GEMM1/2 与 GEMM3/4/5 之间、张量核空转），减指令直接缩短关键路径。
__device__ __forceinline__ uint32_t cvt2_e4m3(float a, float b) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
}
__device__ __forceinline__ uint32_t cvt2_e5m2(float a, float b) {
  return (uint32_t)__nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E5M2);
}
// O28：把 4 个待量化浮点（已乘 rowwise 因子）折成 4 个连续 fp8 字节（低字节对应 x0）。
//   `E5=true` 用 E5M2、否则 E4M3；`folddiv` 负责 RCP/除法折算。与逐元素标量版逐位相同。
#ifndef FA_CVT2
#define FA_CVT2 1
#endif
template <bool RCP, bool E5>
__device__ __forceinline__ uint32_t foldpack4(float x0, float x1, float x2, float x3,
                                              float sc, float inv) {
  const float a = folddiv<RCP>(x0, sc, inv), b = folddiv<RCP>(x1, sc, inv);
  const float c = folddiv<RCP>(x2, sc, inv), d = folddiv<RCP>(x3, sc, inv);
#if FA_CVT2
  uint32_t lo, hi;
  if constexpr (E5) { lo = cvt2_e5m2(a, b); hi = cvt2_e5m2(c, d); }
  else              { lo = cvt2_e4m3(a, b); hi = cvt2_e4m3(c, d); }
  return lo | (hi << 16);
#else
  // O28 A/B：`-DFA_CVT2=0` 退回逐元素标量 cvt + shift/OR（结果逐位相同）。
  if constexpr (E5)
    return (uint32_t)cvt_e5m2(a) | ((uint32_t)cvt_e5m2(b) << 8) |
           ((uint32_t)cvt_e5m2(c) << 16) | ((uint32_t)cvt_e5m2(d) << 24);
  else
    return (uint32_t)cvt_e4m3(a) | ((uint32_t)cvt_e4m3(b) << 8) |
           ((uint32_t)cvt_e4m3(c) << 16) | ((uint32_t)cvt_e4m3(d) << 24);
#endif
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
// O51：带源字节数的 16B `cp.async.cg`（src-size=0 时零填充、不读 global），用于行越界时
//   给 smem 写 0（与 `kv_load_pair` 的「越界写 0 字节」语义一致）。
__device__ __forceinline__ void cp_async16_z(void* dst_smem, const void* src_gmem, int nbytes) {
  uint32_t s = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(src_gmem),
               "r"(nbytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}
__device__ __forceinline__ void cp_async_wait0() {
  asm volatile("cp.async.wait_group 0;\n");
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
// O81（F3b）：`ldmatrix.x4.trans`（转置读 16 行 × 32 fp8）。用于从 SW128 源做逐字节转置。
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t addr, uint32_t d[4]) {
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
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
// O85（第 180 轮）：**chunk-major SW128** —— 物理布局 `[k/128][row/8][8][128]`，与 4D-TMA 的
//   「一个 box = 128 列」逐 box 落位（`Qs + c*(nrows/8)*1024`）严格一致（对齐 O74 LSE 的
//   `wgmma_qkt64_fp8_chunked`）。HD=128（nrows 任意、只有 1 个 chunk）时与 `sw128_off_fp8`
//   逐字节相同；HD=256 时用于 4D-TMA 搬入的 Q/dO（TMA 一个 box 内维固定 128B=128 fp8，
//   两个 box 各写一块 canonical [nrows][128]，故物理布局是 chunk-major 而非 rg-major）。
__device__ __forceinline__ int sw128c_off_fp8(int row, int k, int nrows) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 7, kk = k & 127;
  const int cc = (kk >> 4) ^ rr;
  return (kg * (nrows >> 3) + rg) * 1024 + (rr * 8 + cc) * 16 + (kk & 15);
}
// chunk-major 的 k32 起始地址：chunk 内 4 个 k32 步进 + 跨 chunk 的 (nrows/8)*1024 字节。
__device__ __forceinline__ uint32_t sw128c_k32_addr(uint32_t base, int s, int nrows) {
  return base + (uint32_t)((s >> 2) * ((nrows >> 3) * 1024) + (s & 3) * 32);
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

// O81（F3b 主体，第 176 轮）：把 GEMM5（dQ）从 `mma.m16n8k32` 换到 **wgmma RS** 所需的两个
//   数据通路 helper。由 `src/fp8/fa_bwd_fp8_wgmma345_smoke.cu` 三验证（全部 max_abs=0）：
//   (A) CUTLASS canonical Major-K **INTERLEAVE**（无 swizzle）布局：
//       `((8,n),2):((1,SBO),LBO)` ⇒ 元素 (r,k) 字节偏移 = 16*((r&7)+SBO*(r>>3)+LBO*(k>>4))+(k&15)。
//       K=32（BN=32）时自然编码 LBO=8、SBO=16（单位 uint128=16B）。
//   (B) 从 **SW128 源**做逐字节转置（只改 `ldmatrix.x4.trans` 的 lane 地址走 `sw128_off_fp8`，
//       PRMT 数学与 O80 相同）⇒ 把 SW128 的 K tile 转成 Kᵀ 的 INTERLEAVE K-major tile。
//   描述符（no-swizzle，layout_type=0）：LBO 在 bit16-29、SBO 在 bit32-45（单位 16B）。
__device__ __forceinline__ int inter_k_off_fp8(int r, int k, int SBO_u, int LBO_u) {
  return 16 * ((r & 7) + SBO_u * (r >> 3) + LBO_u * (k >> 4)) + (k & 15);
}
__device__ __forceinline__ uint64_t make_desc_noswz_fp8(uint32_t addr, uint32_t lbo_u,
                                                        uint32_t sbo_u) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)(lbo_u & 0x3FFF) << 16;
  d |= (uint64_t)(sbo_u & 0x3FFF) << 32;
  d |= (uint64_t)0 << 62;  // layout_type = 0（no swizzle / interleave）
  return d;
}
// 从 SW128 源 [R][C]（K-major，C 连续）逐字节转置并写成 INTERLEAVE K-major 目标 [C][R]
//   （SBO=16、LBO=8，即 K=R 连续、8 行组 + 16 元素 core）。R 须为 8 的倍数且 ≤32（2 个 core）；
//   C 须为 64 的倍数（每个 x4.trans 覆盖 64 fp8 列）。4 个 warp 协作。
template <int R, int C>
__device__ __forceinline__ void transpose_sw128_to_inter(unsigned char* __restrict__ sDst,
                                                         const unsigned char* __restrict__ sSrc,
                                                         int wid, int lane) {
  static_assert(R % 8 == 0 && C % 64 == 0, "transpose_sw128_to_inter 尺寸约束");
  const int nblkC = C / 64;
  const int nwork = (R / 8) * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    ldmatrix_x4_trans(smem_u32(sSrc + sw128_off_fp8(r0 + row, c0 + mat * 16, C)), reg);
    const int p = lane & 3, q = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + q;
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const int d0 = c0 + 2 * k, d1 = d0 + 1;
      const int m = r0 + 2 * p;
      *reinterpret_cast<uint16_t*>(sDst + inter_k_off_fp8(d0, m, 16, 8)) =
          (uint16_t)(word & 0xffffu);
      *reinterpret_cast<uint16_t*>(sDst + inter_k_off_fp8(d1, m, 16, 8)) =
          (uint16_t)(word >> 16);
    }
  }
}

// O82：**紧凑 no-swizzle K-major**（`layout_type=0`）通用布局，支持 K=64（`inter_k_off_fp8`
//   的 LBO=8 编码只对 K≤32 无冲突——K=64 时 8*(k>>4) 的第 4 位会与 SBO*(r>>3) 撞）。
//   定义 [N=C 行][K=R 列] 的行主序分块：offset(d,m) = (m>>4)*(C*16) + d*16 + (m&15)。
//   即：K-core（16 元素）为主序、块内 C 行连续 16B。描述符用 `make_desc_noswz_fp8(addr,
//   /*lbo_u=*/C, /*sbo_u=*/8)`（LBO=core 间 16B 数 = C，SBO=8 行组间 = 8）。
__device__ __forceinline__ int noswz_k_off_c(int d, int m, int C) {
  return (m >> 4) * (C * 16) + d * 16 + (m & 15);
}
// 从 SW128 源 [R][C]（K-major，C 连续）逐字节转置并写成上面的紧凑 no-swizzle 目标 [C][R]。
//   R 须为 8 的倍数（K 维，可达 64）；C 须为 64 的倍数（每个 x4.trans 覆盖 64 fp8 列）。4 warp 协作。
template <int R, int C>
__device__ __forceinline__ void transpose_sw128_to_noswz(unsigned char* __restrict__ sDst,
                                                         const unsigned char* __restrict__ sSrc,
                                                         int wid, int lane) {
  static_assert(R % 8 == 0 && C % 64 == 0, "transpose_sw128_to_noswz 尺寸约束");
  const int nblkC = C / 64;
  const int nwork = (R / 8) * nblkC;
  for (int w = wid; w < nwork; w += 4) {
    const int rblk = w / nblkC, cblk = w % nblkC;
    const int r0 = rblk * 8, c0 = cblk * 64;
    const int mat = lane >> 3, row = lane & 7;
    uint32_t reg[4];
    ldmatrix_x4_trans(smem_u32(sSrc + sw128_off_fp8(r0 + row, c0 + mat * 16, C)), reg);
    const int p = lane & 3, q = lane >> 2;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int k = 8 * i + q;
      const uint32_t word = __byte_perm(reg[i], reg[i] >> 16, 0x5140);
      const int d0 = c0 + 2 * k, d1 = d0 + 1;
      const int m = r0 + 2 * p;
      *reinterpret_cast<uint16_t*>(sDst + noswz_k_off_c(d0, m, C)) = (uint16_t)(word & 0xffffu);
      *reinterpret_cast<uint16_t*>(sDst + noswz_k_off_c(d1, m, C)) = (uint16_t)(word >> 16);
    }
  }
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
// O50：只等「最老的那组」wgmma 完成（pending ≤ N）。用于拆开 GEMM1/GEMM2 的等待。
template <int N>
__device__ __forceinline__ void wgmma_wait_group_fp8() {
#if FA_FP8_HAS_WGMMA
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
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
// O81（F3b）：**RS_TN** 形式（A 来自寄存器 4×u32，B 走描述符）。用于 GEMM5（dQ）——
//   A=dS2[m][j]（e5m2，行主序 K-major，`ldmatrix.x4` 装入）、B=Kᵀ[j][d]（e4m3，no-swizzle）。
//   由 `fa_bwd_fp8_wgmma345_smoke.cu` 端到端验证（max_abs=0）。
__device__ __forceinline__ void wgmma_m64n32k32_rs_e5e4(float (&d)[16],
                                                        const uint32_t a[4],
                                                        uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e5m2.e4m3 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "{%16,%17,%18,%19}, %20, p, %22, %23;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1), "n"(1),
        "n"(1));
#else
  (void)a; (void)db;
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
#endif
}
// O82（F3b 主体续）：**RS_TN e4m3×e5m2**（GEMM3 dV：A=Ap e4m3 寄存器，B=dOᵀ e5m2 描述符）。
__device__ __forceinline__ void wgmma_m64n32k32_rs_e4e5(float (&d)[16],
                                                        const uint32_t a[4],
                                                        uint64_t db) {
#if FA_FP8_HAS_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e5m2 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},\n"
      "{%16,%17,%18,%19}, %20, p, %22, %23;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
        "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(db), "r"(1), "n"(1),
        "n"(1));
#else
  (void)a; (void)db;
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
#endif
}
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

// O85（第 180 轮）：GEMM1/2 的 **A 走 chunk-major**（4D-TMA 搬入的 Q/dO）、B 仍为 rg-major
//   （cp.async 的 K/V）变体。仅 D=256 的 qd-tma 路径使用；D=128（NCH=1）与 `wgmma_mn32_issue`
//   逐位相同。`AROWS`=A tile 的行数（Q/dO 为 BM），用于 chunk stride `(AROWS/8)*1024`。
template <int KIND, int AROWS>
__device__ __forceinline__ void wgmma_mn32_issue_cm(const char* Asw, const char* Bsw, int HD,
                                                    float (&d)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) d[i] = 0.f;
  wgmma_fence_fp8();
  const uint32_t aa = smem_u32(Asw), ba = smem_u32(Bsw);
  const uint32_t sbo = (uint32_t)((HD / 128) * 1024);
#pragma unroll
  for (int s = 0; s < HD / 32; ++s) {
    uint64_t da = make_desc_sw128_fp8(sw128c_k32_addr(aa, s, AROWS), 1024);
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

// O74（第 169 轮）：HD>128 的 TMA LSE 用的「分 chunk QKᵀ」。TMA 一个 box 内维固定 128B
//   （128 fp8），HD=512（MLA）要 4 个 box 才能搬完整行；4 个 box 各写一块 [64][128] 的
//   canonical SW128 tile 到 `Qs + c*CHUNK`（CHUNK=(LBM/8)*1024=8KB），故物理布局是
//   `[kg][rg]` 而非单 box 的 canonical `[rg][kg]`。这里对每个 chunk 各用 **SBO=1024** 的描述符
//   累加（对齐 fp16 O30/O71 的 2-chunk 写法），数学与 `wgmma_qkt64_fp8` 完全相同、只差同一
//   tile 内跨 chunk 的 fp32 累加次序（LSE 容差 O(1) 内）。`NCH=HD/128`、`CHUNK` 由调用方传。
__device__ __forceinline__ void wgmma_qkt64_fp8_chunked(const char* Qsw, const char* Ksw,
                                                        int nch, int chunk, float (&d)[32]) {
#pragma unroll
  for (int i = 0; i < 32; ++i) d[i] = 0.f;
  wgmma_fence_fp8();
  const uint32_t qa = smem_u32(Qsw), ka = smem_u32(Ksw);
#pragma unroll 1
  for (int c = 0; c < nch; ++c) {
    const uint32_t qc = qa + (uint32_t)(c * chunk), kc = ka + (uint32_t)(c * chunk);
#pragma unroll
    for (int s = 0; s < 4; ++s) {
      uint64_t da = make_desc_sw128_fp8(sw128_k32_addr(qc, s), 1024);
      uint64_t db = make_desc_sw128_fp8(sw128_k32_addr(kc, s), 1024);
      wgmma_m64n64k32_e4e4(d, da, db);
    }
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
#if FA_RED_STORE
  // O83（第 178 轮）：**诊断专用**——把跨 CTA 原子归约换成 plain store（last-writer-wins，
  //   数值错误），用来分离「原子语义的成本」与「epilogue 写流量本身的成本」。
  //   `FA_RED_STORE=1` 只改时序语义；写地址/字节数与原子版相同。默认 0。
  *reinterpret_cast<float2*>(p) = make_float2(a, b);
#else
  atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));
#endif
}

// O67（第 147 轮）：把 dK/dV 的 `red_add2`（8B `red.global.add.v2.f32`）再提升到
//   `red_add4`（16B `red.global.add.v4.f32`）。动机：默认 fp8 `kvtma` main 的 L2 墙里
//   `red` 占 114.5M 扇区（74%），而每个 red **请求**固定吃 8 个 L2 扇区（mma.m16n8 累加器
//   的 8 行分散在 8 个 32B 扇区）⇒ 扇区数正比于请求数。fp8 `m16n8k32` 的累加器里一个 quad
//   （`lane&3`=0..3）的 `c2=(lane&3)*2` 恰是 0/2/4/6 —— 同 row 的**连续 8 列**；把 quad 的
//   float2 用 `__shfl_down_sync(...,1)` 拼成两个 float4（列 0-3 由 lane0 写、列 4-7 由 lane2
//   写），请求数与 L2 扇区数再减半。调用者须保证 shfl 在整个 warp 上执行（在 `jg<len` guard
//   之外算好），且仅 `(lane&1)==0` 的 lane 落 st；偶 lane 地址天然 16B 对齐（`c2∈{0,4}`）。
//   默认关（`-DFA_R4=1` 同 binary A/B）；O7c 在 fp16 上试过同款为负（见 docs/01 §14q），
//   本轮在 fp8（L2 `red` 占比更高）上复测。
__device__ __forceinline__ void red_add4(float* p, float a, float b, float c, float d) {
  atomicAdd(reinterpret_cast<float4*>(p), make_float4(a, b, c, d));
}

// O114（第 208 轮）：fp16 版跨 CTA 归约——把同一对 (a,b) 先四舍五入到 fp16，再用
//   **显式 PTX `red.global.add.noftz.f16x2`**（无返回、走 RED 路径）落盘。CUDA 的
//   `atomicAdd(__half2*)` 头实现会退化成带返回的 `ATOM.E.ADD.F16x2`（读改写 → 反而多一次
//   读扇区），故这里必须手写 PTX 才能拿到纯 RED（连续地址下扇区减半；mma 行散列下不减，
//   见冒烟 `fa_bwd_fp8_redhalf_smoke` 与 `docs/03` §136）。
__device__ __forceinline__ void red_addh2(__half* p, float a, float b) {
  __half2 v = __floats2half2_rn(a, b);
  unsigned packed = *reinterpret_cast<unsigned*>(&v);
  asm volatile("red.global.add.noftz.f16x2 [%0], %1;" ::"l"(p), "r"(packed) : "memory");
}

// O114：`FA_REDHALF` 的收尾——把 fp16 的 dK/dV 累加缓冲读回并写成 fp32 输出（d_dk/d_dv）。
//   dK/dV 的跨 CTA red 走 fp16；最终输出仍是 fp32（与本文件对拍口径一致）。
//   （实测扇区未减 ⇒ 本路径为负结果探针，见 `docs/03` §136。）
__global__ void redhalf_finalize_kernel(const __half* __restrict__ dk_h,
                                        const __half* __restrict__ dv_h,
                                        float* __restrict__ dk, float* __restrict__ dv, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dk[i] = __half2float(dk_h[i]);
    dv[i] = __half2float(dv_h[i]);
  }
}

// ----------------------------- P3-4e：确定性 dK/dV 归约（`DET`） -----------------------------
// 动机（对齐 fp16/bf16 O7b、catalog §4.2 第 5 条）：fp8 反向的 dK/dV 用跨 CTA `atomicAdd`
//   汇总，一个 KV 元素被多个（Q 块 × Q 头）CTA 贡献；`atomicAdd` 的浮点加法**次序随调度
//   变化** ⇒ 同一 binary 两次跑末位会抖动、无法位复现。`DET=true` 时把每个 CTA 的贡献
//   写进按 `(Q 头, Q 块)` 分片的 partial 缓冲（**非原子覆盖写**，每元素只被一个 CTA 写），
//   再由 `dkv_reduce_kernel` 按固定次序（`h` 升序、`mblk` 升序）求和 ⇒ 与调度无关、可复现。
//   partial 布局：`part[((b*H + h)*nblk + mblk) * S * HD + jg*HD + c]`（每 CTA 一份）。
//   注意 partial 按 **Q 头 h**（不是 KV 头）分片：GQA/MQA 下多个 Q 头共享同一 KV 头，
//   它们的贡献要相加；若只按 hkv 分片会互相覆盖（race）。代价：多一趟写 + 一趟读。
__device__ __forceinline__ void dkv_det_store(float* part, float a, float b) {
  *reinterpret_cast<float2*>(part) = make_float2(a, b);
}

// ---- F4（第一百三十三轮）：把 fp16/bf16 O60/O62 的「partial 降精度 + 写扇区化」逐字搬到 fp8 ----
// 动机：fp8 默认 DET 的 partial 是 fp32（每 lane 一次 `float2` = 8B，同一 quad 的 4 个 lane
//   拼成连续 32B 扇区 ⇒ **已落满扇区**）。但 DET 的二次归约是纯 DRAM 带宽 bound（docs/03 §56：
//   reduce 731.6µs / DRAM 91.5%、partial 写 2.20GB），把 partial 改存 **fp16** 可把写/读字节
//   减半、直接打这条墙。代价与 fp16/bf16 O60 同：每 lane 一次只写 4B（`__half2`），同一 quad
//   仅 16B、**落不满 32B 扇区** ⇒ store 扇区数不减、主 kernel 写侧反而更碎。故必须叠加 O62 的
//   **扇区化**：把相邻两列组 `j`、`j+1` 的 half2 拼成一次 8B 写（`uint2`），4 个 lane 覆盖连续
//   32B；HD 维做 16 列块内置换（`dkv_p16_perm`）使同一 lane 的两个 half2 相邻。
//   partial 以 fp16 存储时指针仍按 `float*` 传入，内部重解释为 `__half*`（元素单位是 2B，故
//   `off` 是 **half 元素下标**，必须在 `__half*` 上做加法）。
__device__ __forceinline__ void dkv_det_store_h4(float* base, size_t off, float a0, float a1,
                                                 float b0, float b1) {
  __half2 h0 = __floats2half2_rn(a0, a1), h1 = __floats2half2_rn(b0, b1);
  uint2 u;
  u.x = *reinterpret_cast<unsigned*>(&h0);
  u.y = *reinterpret_cast<unsigned*>(&h1);
  *reinterpret_cast<uint2*>(reinterpret_cast<__half*>(base) + off) = u;
}

// F4：fp16 partial（`P16`）列布局的 16 列块内置换，公式与 fp16/bf16 `dkv_p16_perm` **逐字相同**。
//   原列 `c = j*8 + 2*L + h`（`j` 列组、`L=lane&3`、`h∈{0,1}`；fp8 的 mma 列 `c2=(lane&3)*2`）
//   映射到 `((j>>1)<<4) + (L<<2) + ((j&1)<<1) + h`：同一 lane 的 group `j`、`j+1` 两个 half2
//   相邻构成 8B，4 个 lane 拼成 32B 扇区。`c0=wc*GN34` 是 16 的倍数，故全局列索引可直接套用。
//   reduce 端按此索引读回，求和集合/次序不变 ⇒ 仍确定性、数值只差 fp16 舍入。
__device__ __forceinline__ int dkv_p16_perm(int c) {
  const int rem = c & 7, L = rem >> 1, h = rem & 1;
  const int j = c >> 3;
  return ((j >> 1) << 4) + (L << 2) + ((j & 1) << 1) + h;
}

// F4-c（第一百三十七轮）：把 `dkv_det_store_h4` 写的 8B 单元（4 个连续物理列的 fp16）
//   一次读回。`u.x` 是逻辑列对（16*blk + 2L, +1），`u.y` 是（16*blk + 8 + 2L, +1）——与
//   `dkv_p16_perm` 的 16 列块内置换一致（见上）。用于 P16 归约的向量化读。
__device__ __forceinline__ void h4_to_f4(uint2 u, float& a, float& b, float& c, float& d) {
  __half2 h0 = *reinterpret_cast<__half2*>(&u.x);
  __half2 h1 = *reinterpret_cast<__half2*>(&u.y);
  a = __low2float(h0);
  b = __high2float(h0);
  c = __low2float(h1);
  d = __high2float(h1);
}

// partial → dK/dV 累加（固定次序）。causal 下 KV 行 `jg` 只被 `mblk >= jg/BM` 的 CTA 写，
//   故从 `jg/BM` 起求和；非 causal 从 0 起。`BM` 由模板给出（fp8 默认 64；fp16 用 128）。
// F4：`P16=true` 时 dK/dV partial 以 fp16 存储且用扇区化 16 列块内置换布局（O62），按置换后
//   的列索引读回（求和集合/次序不变，仍确定性；数值只差 fp16 舍入）。
template <int HD, int BM = 64, bool P16 = false>
__global__ void dkv_reduce_kernel(const float* __restrict__ dk_part,
                                  const float* __restrict__ dv_part,
                                  float* __restrict__ dk_acc, float* __restrict__ dv_acc,
                                  int S, int H, int Hkv, int nblk, int causal) {
  if constexpr (P16) {
    // F4-c（第一百三十七轮）：P16 归约读取向量化——每条 `uint2` 读回 `dkv_det_store_h4`
    //   写的 8B 单元（4 个连续物理列）。每 block 处理 `RW=4` 行（`blockDim=HD`、每行 `HD/4`
    //   组），grid.y = ceil(S/4)。**每输出列的求和集合/次序与标量读逐字相同 ⇒ 逐位相同**；
    //   读指令 4×2B → 1×8B，把 fp16 partial 归约从 ~78% 提到 DRAM 峰值附近。
    constexpr int NPG = HD / 4;          // 每行 HD/4 个列组（每组 4 列）
    constexpr int RW = 4;                // 每 block 行数 = blockDim/NPG = HD/(HD/4)
    const int sub = threadIdx.x / NPG;   // block 内行号
    const int t = threadIdx.x % NPG;     // 列组
    const int jg = blockIdx.y * RW + sub;
    if (jg >= S) return;
    const int hb = blockIdx.x;  // b*Hkv + hkv
    const int b = hb / Hkv, hkv = hb % Hkv;
    const int G = H / Hkv;
    const int h0 = hkv * G;
    const int mblk0 = causal ? (jg / BM) : 0;
    const int blk = t >> 2, L = t & 3;
    float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    for (int hh = 0; hh < G; ++hh) {
      const size_t prow = (size_t)(b * H + h0 + hh);
      for (int m = mblk0; m < nblk; ++m) {
        const size_t base = ((prow * nblk + m) * (size_t)S + jg) * HD + t * 4;
        const uint2 uk =
            *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dk_part) + base);
        const uint2 uv =
            *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dv_part) + base);
        float a, bb, cc, dd;
        h4_to_f4(uk, a, bb, cc, dd);
        k0 += a; k1 += bb; k2 += cc; k3 += dd;
        h4_to_f4(uv, a, bb, cc, dd);
        v0 += a; v1 += bb; v2 += cc; v3 += dd;
      }
    }
    const size_t o = (((size_t)(b * S + jg)) * Hkv + hkv) * HD;
    const int cA = (blk << 4) + (L << 1);   // 16*blk + 2L
    dk_acc[o + cA] = k0;      dk_acc[o + cA + 1] = k1;
    dk_acc[o + cA + 8] = k2;  dk_acc[o + cA + 9] = k3;
    dv_acc[o + cA] = v0;      dv_acc[o + cA + 1] = v1;
    dv_acc[o + cA + 8] = v2;  dv_acc[o + cA + 9] = v3;
    return;
  }
  const int hb = blockIdx.x;  // b*Hkv + hkv
  const int jg = blockIdx.y;
  const int c = threadIdx.x;
  if (c >= HD || jg >= S) return;
  const int b = hb / Hkv, hkv = hb % Hkv;
  const int G = H / Hkv;  // 每 KV 头对应的 Q 头数（GQA 广播组）
  const int h0 = hkv * G;
  const int mblk0 = causal ? (jg / BM) : 0;
  float sk = 0.f, sv = 0.f;
  for (int hh = 0; hh < G; ++hh) {
    const size_t prow = (size_t)(b * H + h0 + hh);
    for (int m = mblk0; m < nblk; ++m) {
      const size_t base = ((prow * nblk + m) * (size_t)S + jg) * HD + c;
      if constexpr (P16) {
        const size_t bp = ((prow * nblk + m) * (size_t)S + jg) * HD + dkv_p16_perm(c);
        sk += __half2float(reinterpret_cast<const __half*>(dk_part)[bp]);
        sv += __half2float(reinterpret_cast<const __half*>(dv_part)[bp]);
      } else {
        sk += dk_part[base];
        sv += dv_part[base];
      }
    }
  }
  const size_t o = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c;
  dk_acc[o] = sk;
  dv_acc[o] = sv;
}

// P3-4f：dQ 的确定性归约（`DET && ksplit>1`）。dK/dV 的 partial 天然无需 part 维
//   （每个 (mblk, jg) 的 K tile 只属于一个 part，各 part 写不相交的 jg），但 dQ 的每个
//   `(row, h, c)` 会被同一 mblk 的 ksplit 个 part 各贡献一个偏和 ⇒ 必须按 part 分片，
//   再按 `part=0..ksplit-1` 固定次序求和。布局 `dq_part[((row*H + h)*ksplit + part)*HD + c]`，
//   row = b*S+qi（packed q token）。**空 part（causal 下小 mblk 被切空）写不到**，故
//   host 每次先把 dq_part 清零。每线程一个 `(row,h,c)`，求和次序固定 ⇒ 可复现。
template <int HD>
__global__ void dq_reduce_kernel(const float* __restrict__ dq_part, float* __restrict__ dq_acc,
                                 int S, int H, int ksplit) {
  const int row = blockIdx.x;  // 全局 q token（b*S + qi）
  const int h = blockIdx.y;
  const int c = threadIdx.x;
  if (c >= HD) return;
  const size_t b0 = ((size_t)row * H + h) * ksplit;
  float s = 0.f;
  for (int p = 0; p < ksplit; ++p) s += dq_part[(b0 + p) * HD + c];
  dq_acc[((size_t)row * H + h) * HD + c] = s;
}

// ------------------- P3-4n：把两个 reduce 融合进一个 kernel（候选 ②） -------------------
// 动机：DET 的二次归约原本是**两次 launch**——`dkv_reduce_kernel`（读 dk/dv partial）+
//   `dq_reduce_kernel`（读 dq partial）。两者数据不相交（各自 block 独立），合成一次 launch
//   可省掉一次 launch + 一次尾部，并让 dq 的轻量块与 dkv 的重块在同一网格里交错填满 SM。
//   **字节数不变**（两个 reduce 读的都是各自 partial；无法合并），故预期收益在「省一次
//   launch + 小 S 的尾部」，对大 S 的纯带宽墙中性。数学/求和次序逐字与两个 kernel 相同 ⇒
//   融合版与分开版**逐位相同**。grid = `dkv_blocks + dq_blocks`（1D，block=HD）；前
//   `dkv_blocks` 个块做 dK/dV，其余做 dQ。`ksplit==1` 时 `dq_blocks=0`，退化成纯 dkv。
template <int HD, int BM = 64, bool P16 = false>
__global__ void dkv_dq_reduce_kernel(const float* __restrict__ dk_part,
                                     const float* __restrict__ dv_part,
                                     const float* __restrict__ dq_part,
                                     float* __restrict__ dk_acc, float* __restrict__ dv_acc,
                                     float* __restrict__ dq_acc, int S, int H, int Hkv,
                                     int nblk, int causal, int ksplit, int dkv_blocks) {
  const int c = threadIdx.x;
  if (c >= HD) return;
  const int idx = blockIdx.x;
  if (idx < dkv_blocks) {
    if constexpr (P16) {
      // F4-c（第一百三十七轮）：dK/dV 分支的向量化读（与 `dkv_reduce_kernel` 的 P16 逐字同构）。
      //   此时 host 传 `dkv_blocks = B*Hkv*ceil(S/4)`，`idx` 按「每 4 行一 block」编码。
      constexpr int NPG = HD / 4;
      constexpr int RW = 4;
      const int rowblocks = (S + RW - 1) / RW;
      const int hb = idx / rowblocks;
      const int jg = (idx - hb * rowblocks) * RW + threadIdx.x / NPG;
      const int t = threadIdx.x % NPG;
      if (jg >= S) return;
      const int b = hb / Hkv, hkv = hb % Hkv;
      const int G = H / Hkv;
      const int h0 = hkv * G;
      const int mblk0 = causal ? (jg / BM) : 0;
      const int blk = t >> 2, L = t & 3;
      float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
      for (int hh = 0; hh < G; ++hh) {
        const size_t prow = (size_t)(b * H + h0 + hh);
        for (int m = mblk0; m < nblk; ++m) {
          const size_t base = ((prow * nblk + m) * (size_t)S + jg) * HD + t * 4;
          const uint2 uk =
              *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dk_part) + base);
          const uint2 uv =
              *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dv_part) + base);
          float a, bb, cc, dd;
          h4_to_f4(uk, a, bb, cc, dd);
          k0 += a; k1 += bb; k2 += cc; k3 += dd;
          h4_to_f4(uv, a, bb, cc, dd);
          v0 += a; v1 += bb; v2 += cc; v3 += dd;
        }
      }
      const size_t o = (((size_t)(b * S + jg)) * Hkv + hkv) * HD;
      const int cA = (blk << 4) + (L << 1);
      dk_acc[o + cA] = k0;      dk_acc[o + cA + 1] = k1;
      dk_acc[o + cA + 8] = k2;  dk_acc[o + cA + 9] = k3;
      dv_acc[o + cA] = v0;      dv_acc[o + cA + 1] = v1;
      dv_acc[o + cA + 8] = v2;  dv_acc[o + cA + 9] = v3;
      return;
    }
    // ---- 与 dkv_reduce_kernel 逐字相同的 dK/dV 归约 ----
    const int hb = idx / S, jg = idx - hb * S;
    const int b = hb / Hkv, hkv = hb % Hkv;
    const int G = H / Hkv;
    const int h0 = hkv * G;
    const int mblk0 = causal ? (jg / BM) : 0;
    float sk = 0.f, sv = 0.f;
    for (int hh = 0; hh < G; ++hh) {
      const size_t prow = (size_t)(b * H + h0 + hh);
      for (int m = mblk0; m < nblk; ++m) {
        const size_t base = ((prow * nblk + m) * (size_t)S + jg) * HD + c;
        if constexpr (P16) {
          const size_t bp = ((prow * nblk + m) * (size_t)S + jg) * HD + dkv_p16_perm(c);
          sk += __half2float(reinterpret_cast<const __half*>(dk_part)[bp]);
          sv += __half2float(reinterpret_cast<const __half*>(dv_part)[bp]);
        } else {
          sk += dk_part[base];
          sv += dv_part[base];
        }
      }
    }
    const size_t o = (((size_t)(b * S + jg)) * Hkv + hkv) * HD + c;
    dk_acc[o] = sk;
    dv_acc[o] = sv;
  } else {
    // ---- 与 dq_reduce_kernel 逐字相同的 dQ 归约（按 part 固定次序） ----
    const int qi = idx - dkv_blocks;
    const int row = qi / H, h = qi - row * H;
    const size_t b0 = ((size_t)row * H + h) * ksplit;
    float s = 0.f;
    for (int p = 0; p < ksplit; ++p) s += dq_part[(b0 + p) * HD + c];
    dq_acc[((size_t)row * H + h) * HD + c] = s;
  }
}

// P3-4i：varlen 版的确定性 dK/dV 归约（把 P3-4e 的 `--det` 扩到变长）。partial 布局沿用定长式
//   `part[((b*H + h)*nblk_max + mblk)*maxlen + jg]`（body 传 `S=maxlen`、`nblk=nblk_max`），
//   但每个序列长度/块数不同 ⇒ 归约按 `cu_seqlens` 定界：`len_b = cu[b+1]-cu[b]`、
//   `nblk_b = ceil(len_b/BM)`，并只对 `jg < len_b` 的行求和；输出按 packed `[T,Hkv,D]` 定位
//   （全局 KV token = `qbase + jg`）。grid=(B*Hkv, maxlen)、block=HD。k 维（GQA 广播组）与
//   mblk 的求和次序固定（hh 升序、m 升序）⇒ 与调度无关、两次跑逐位可复现。
// F4-b（第 134 轮）：把 F4 第一步的「fp16 partial + O62 写扇区化」从定长扩到变长——`P16=true`
//   时 dK/dV partial 以 fp16 存储且用 16 列块内置换布局（与 `fp8_mma_body` 的 `DET_HALF` 写一致），
//   按置换后的列索引读回（求和集合/次序不变，仍确定性；数值只差 fp16 舍入）。HD 无关。
template <int HD, int BM = 64, bool P16 = false>
__global__ void dkv_reduce_varlen_kernel(const float* __restrict__ dk_part,
                                         const float* __restrict__ dv_part,
                                         float* __restrict__ dk_acc, float* __restrict__ dv_acc,
                                         const int* __restrict__ cu_seqlens,
                                         int H, int Hkv, int nblk_max, int maxlen, int causal,
                                         const int* __restrict__ part_base = nullptr) {
  if constexpr (P16) {
    // F4-c（第一百三十七轮）：varlen P16 归约读取向量化（与定长版 `h4_to_f4` 同构）。
    constexpr int NPG = HD / 4;
    constexpr int RW = 4;
    const int sub = threadIdx.x / NPG;
    const int t = threadIdx.x % NPG;
    const int hb = blockIdx.x;  // b*Hkv + hkv
    const int jg = blockIdx.y * RW + sub;  // 序列内的 KV 行
    const int b = hb / Hkv, hkv = hb % Hkv;
    const int qbase = cu_seqlens[b];
    const int len = cu_seqlens[b + 1] - qbase;
    if (jg >= len) return;
    const int nblk_b = (len + BM - 1) / BM;
    const int G = H / Hkv;
    const int h0 = hkv * G;
    const int mblk0 = causal ? (jg / BM) : 0;
    const int blk = t >> 2, L = t & 3;
    float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
    float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
    for (int hh = 0; hh < G; ++hh) {
      const size_t prow = (size_t)(b * H + h0 + hh) * nblk_max;
      for (int m = mblk0; m < nblk_b; ++m) {
        const size_t row =
            part_base
                ? ((size_t)part_base[b] + ((size_t)(h0 + hh) * nblk_b + m) * len + jg) * HD
                : ((prow + m) * (size_t)maxlen + jg) * HD;
        const size_t base = row + t * 4;
        const uint2 uk =
            *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dk_part) + base);
        const uint2 uv =
            *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dv_part) + base);
        float a, bb, cc, dd;
        h4_to_f4(uk, a, bb, cc, dd);
        k0 += a; k1 += bb; k2 += cc; k3 += dd;
        h4_to_f4(uv, a, bb, cc, dd);
        v0 += a; v1 += bb; v2 += cc; v3 += dd;
      }
    }
    const size_t o = (((size_t)(qbase + jg)) * Hkv + hkv) * HD;
    const int cA = (blk << 4) + (L << 1);
    dk_acc[o + cA] = k0;      dk_acc[o + cA + 1] = k1;
    dk_acc[o + cA + 8] = k2;  dk_acc[o + cA + 9] = k3;
    dv_acc[o + cA] = v0;      dv_acc[o + cA + 1] = v1;
    dv_acc[o + cA + 8] = v2;  dv_acc[o + cA + 9] = v3;
    return;
  }
  const int hb = blockIdx.x;   // b*Hkv + hkv
  const int jg = blockIdx.y;   // 序列内的 KV 行
  const int c = threadIdx.x;
  if (c >= HD) return;
  const int b = hb / Hkv, hkv = hb % Hkv;
  const int qbase = cu_seqlens[b];
  const int len = cu_seqlens[b + 1] - qbase;
  if (jg >= len) return;
  const int nblk_b = (len + BM - 1) / BM;
  const int G = H / Hkv;
  const int h0 = hkv * G;
  const int mblk0 = causal ? (jg / BM) : 0;
  float sk = 0.f, sv = 0.f;
  for (int hh = 0; hh < G; ++hh) {
    const size_t prow = (size_t)(b * H + h0 + hh) * nblk_max;
    for (int m = mblk0; m < nblk_b; ++m) {
      // P3-4o：`part_base` 非空时读 compact per-sequence 布局（与 `fp8_mma_body` 的写一致）。
      const size_t row =
          part_base
              ? ((size_t)part_base[b] + ((size_t)(h0 + hh) * nblk_b + m) * len + jg) * HD
              : ((prow + m) * (size_t)maxlen + jg) * HD;
      if constexpr (P16) {
        // F4-b：fp16 partial + 16 列块内置换（`dkv_p16_perm`），按置换列读回。
        const size_t bp = row + dkv_p16_perm(c);
        sk += __half2float(reinterpret_cast<const __half*>(dk_part)[bp]);
        sv += __half2float(reinterpret_cast<const __half*>(dv_part)[bp]);
      } else {
        sk += dk_part[row + c];
        sv += dv_part[row + c];
      }
    }
  }
  const size_t o = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c;
  dk_acc[o] = sk;
  dv_acc[o] = sv;
}

// P3-4n：varlen 版的 dkv+dq 融合归约（同定长版：grid = dkv_blocks + dq_blocks，求和次序
//   逐字沿用 `dkv_reduce_varlen_kernel` / `dq_reduce_kernel` ⇒ 与分开版逐位相同）。
//   `dkv_blocks = B*Hkv*maxlen`（`jg>=len_b` 的块空转 return），`dq_blocks = T*H`。
// F4-b：varlen 融合归约的 P16 版（与 `dkv_reduce_varlen_kernel` 的 P16 读逐字一致）。
template <int HD, int BM = 64, bool P16 = false>
__global__ void dkv_dq_reduce_varlen_kernel(
    const float* __restrict__ dk_part, const float* __restrict__ dv_part,
    const float* __restrict__ dq_part, float* __restrict__ dk_acc, float* __restrict__ dv_acc,
    float* __restrict__ dq_acc, const int* __restrict__ cu_seqlens, int H, int Hkv, int nblk_max,
    int maxlen, int causal, int ksplit, int dkv_blocks,
    const int* __restrict__ part_base = nullptr) {
  const int c = threadIdx.x;
  if (c >= HD) return;
  const int idx = blockIdx.x;
  if (idx < dkv_blocks) {
    if constexpr (P16) {
      // F4-c（第一百三十七轮）：varlen 融合归约的 dK/dV 向量化读（与 `dkv_reduce_varlen_kernel`
      //   的 P16 逐字同构）。host 传 `dkv_blocks = B*Hkv*ceil(maxlen/4)`。
      constexpr int NPG = HD / 4;
      constexpr int RW = 4;
      const int rowblocks = (maxlen + RW - 1) / RW;
      const int hb = idx / rowblocks;
      const int jg = (idx - hb * rowblocks) * RW + threadIdx.x / NPG;
      const int t = threadIdx.x % NPG;
      const int b = hb / Hkv, hkv = hb % Hkv;
      const int qbase = cu_seqlens[b];
      const int len = cu_seqlens[b + 1] - qbase;
      if (jg >= len) return;
      const int nblk_b = (len + BM - 1) / BM;
      const int G = H / Hkv;
      const int h0 = hkv * G;
      const int mblk0 = causal ? (jg / BM) : 0;
      const int blk = t >> 2, L = t & 3;
      float k0 = 0.f, k1 = 0.f, k2 = 0.f, k3 = 0.f;
      float v0 = 0.f, v1 = 0.f, v2 = 0.f, v3 = 0.f;
      for (int hh = 0; hh < G; ++hh) {
        const size_t prow = (size_t)(b * H + h0 + hh) * nblk_max;
        for (int m = mblk0; m < nblk_b; ++m) {
          const size_t row =
              part_base
                  ? ((size_t)part_base[b] + ((size_t)(h0 + hh) * nblk_b + m) * len + jg) * HD
                  : ((prow + m) * (size_t)maxlen + jg) * HD;
          const size_t base = row + t * 4;
          const uint2 uk =
              *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dk_part) + base);
          const uint2 uv =
              *reinterpret_cast<const uint2*>(reinterpret_cast<const __half*>(dv_part) + base);
          float a, bb, cc, dd;
          h4_to_f4(uk, a, bb, cc, dd);
          k0 += a; k1 += bb; k2 += cc; k3 += dd;
          h4_to_f4(uv, a, bb, cc, dd);
          v0 += a; v1 += bb; v2 += cc; v3 += dd;
        }
      }
      const size_t o = (((size_t)(qbase + jg)) * Hkv + hkv) * HD;
      const int cA = (blk << 4) + (L << 1);
      dk_acc[o + cA] = k0;      dk_acc[o + cA + 1] = k1;
      dk_acc[o + cA + 8] = k2;  dk_acc[o + cA + 9] = k3;
      dv_acc[o + cA] = v0;      dv_acc[o + cA + 1] = v1;
      dv_acc[o + cA + 8] = v2;  dv_acc[o + cA + 9] = v3;
      return;
    }
    const int hb = idx / maxlen, jg = idx - hb * maxlen;
    const int b = hb / Hkv, hkv = hb % Hkv;
    const int qbase = cu_seqlens[b];
    const int len = cu_seqlens[b + 1] - qbase;
    if (jg >= len) return;
    const int nblk_b = (len + BM - 1) / BM;
    const int G = H / Hkv;
    const int h0 = hkv * G;
    const int mblk0 = causal ? (jg / BM) : 0;
    float sk = 0.f, sv = 0.f;
    for (int hh = 0; hh < G; ++hh) {
      const size_t prow = (size_t)(b * H + h0 + hh) * nblk_max;
      for (int m = mblk0; m < nblk_b; ++m) {
        // P3-4o：compact per-sequence 布局（与 `fp8_mma_body` / `dkv_reduce_varlen_kernel` 一致）。
        const size_t row =
            part_base
                ? ((size_t)part_base[b] + ((size_t)(h0 + hh) * nblk_b + m) * len + jg) * HD
                : ((prow + m) * (size_t)maxlen + jg) * HD;
        if constexpr (P16) {
          const size_t bp = row + dkv_p16_perm(c);
          sk += __half2float(reinterpret_cast<const __half*>(dk_part)[bp]);
          sv += __half2float(reinterpret_cast<const __half*>(dv_part)[bp]);
        } else {
          sk += dk_part[row + c];
          sv += dv_part[row + c];
        }
      }
    }
    const size_t o = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c;
    dk_acc[o] = sk;
    dv_acc[o] = sv;
  } else {
    const int qi = idx - dkv_blocks;
    const int row = qi / H, h = qi - row * H;
    const size_t b0 = ((size_t)row * H + h) * ksplit;
    float s = 0.f;
    for (int p = 0; p < ksplit; ++p) s += dq_part[(b0 + p) * HD + c];
    dq_acc[((size_t)row * H + h) * HD + c] = s;
  }
}

// ------------------- O42：Hopper bulk reduce（`cp.reduce.async.bulk`） -------------------
// 动机：dK/dV 的逐元素 `red_add2` 虽是 coalesced，但每 tile 要发 2048 条、占满 LSU/L2
//   流水（O42 实测：把 dK/dV 的 red 短路掉 main 1.60→0.94ms，天花板 1.70×）。改成
//   「累加器先写 per-warp smem staging，再由 `cp.reduce.async.bulk...add.f32` 一次性
//   coalesced 归约回 global」可把每 tile 的归约指令从 ~2048 条降到 64 条、且完全异步。
// 语义：dst 是 global [rows][HD] 里的连续 256B 行；src 是 smem 里 16B 对齐、256B 的段。
//   多 CTA 并发 reduce 到同一 global 行由硬件做原子加（冒烟 `fa_bwd_fp8_bulkred_smoke.cu`
//   逐位 PASS）。**generic 写 smem 后必须 `fence.proxy.async.shared::cta`** 才能被 async
//   proxy 读到。
__device__ __forceinline__ void bulk_reduce_fence() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void bulk_reduce_add_f32(float* gdst, const float* ssrc, int nbytes) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  unsigned s = (unsigned)__cvta_generic_to_shared(ssrc);
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], %2;\n"
               ::"l"(gdst), "r"(s), "r"(nbytes) : "memory");
#else
  (void)gdst; (void)ssrc; (void)nbytes;
#endif
}
__device__ __forceinline__ void bulk_reduce_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void bulk_reduce_wait0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
#endif
}

// ---- F7 第十二步：TMA **4D tensor store-reduce**（`cp.reduce.async.bulk.tensor.4d`）----
//   背景：O67/p147 用 ncu 重测 TE fp8 反向（`..._flash_bprop_wgmma_f8_...`）发现它全局归约
//   只有 **3168** 条 `smsp__inst_executed_op_global_red`（ours 9.54M、`l1tex_red` 请求），
//   且 **0 条** plain global store；SASS 直方图见 **`UTMAREDG.4D.ADD`**——即 TE 的 dK/dV/dQ
//   归约走 **TMA 4D 张量归约**（一次搬一整块 [rows][cols] 到 global 并原子加），而不是
//   逐 lane 的 `red.global.add`。这正是 O42 的「smem staging + bulk reduce」思路，但 O42 用
//   的是 **1D** `cp.reduce.async.bulk`（每行一条、且 per-warp），本 helper 用 **4D tensor**
//   变体：src 是 smem 里行主序的整块 [boxR][boxD]（与行主序 tile 完全一致），**一条指令**
//   归约整个 tile，与 TE 同构。
//   语义：`map` 描述 global 张量（用 `make_kvowner_dq_map_f32` 建，dims={D,S,H,B}、f32、
//   box={boxD,boxR,1,1}、SWIZZLE_NONE），`ssrc` 是 16B 对齐的 smem 源（boxD*boxR*f32 字节），
//   坐标 `{c0,c1,c2,c3}` 是 tile 原点。硬件对并发写到同一 global 元素的多个 CTA 做原子加。
//   generic 写 smem 后必须 `bulk_reduce_fence()`（`fence.proxy.async.shared::cta`）。
__device__ __forceinline__ void tma_reduce_add_4d_f32(const CUtensorMap* map, const float* ssrc,
                                                     int c0, int c1, int c2, int c3) {
  // 注意：此处用 `__CUDA_ARCH__` 直接守卫（不能用 `FA_FP8_HAS_TMA`——它在文件更后面才
  //   `#define`，宏按出现顺序展开，放在这里会恒为 0 而把 asm 编掉，实测 dQ 完全错）。
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  const unsigned s = smem_u32(ssrc);
  // 正确 PTX（对齐 CUDA `cuda/__ptx` 生成头）：`.redOp.tile.bulk_group`，类型由 tensormap
  //   element type（此处 FLOAT32）推断，指令**不带** `.f32` 后缀。
  asm volatile(
      "cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group"
      " [%0, {%1, %2, %3, %4}], [%5];\n" ::"l"((uint64_t)map),
      "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(s)
      : "memory");
#else
  (void)map; (void)ssrc; (void)c0; (void)c1; (void)c2; (void)c3;
#endif
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
template <int NPU, int HD, int BN, int NT = THREADS>
__device__ __forceinline__ void kv_prefetch_pair(const unsigned char* __restrict__ k8,
                                                 const unsigned char* __restrict__ v8,
                                                 int j0, int S, int Hkv, int hkv, int qbase,
                                                 int tid, uint32_t* pk0, uint32_t* pk1,
                                                 uint32_t* pv0, uint32_t* pv1) {
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;  // 本 tile 的 (行对, 4B) unit 数（O48 上界）
#pragma unroll
  for (int e = 0; e < NPU; ++e) {
    int u = tid + e * NT;
    int rp = u / nd4, dq = (u % nd4) * 4;
    int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    // O48：NTH 可 >128（D=128 的 8-warp 几何），此时 NPU*NT 会超过本 tile 的 unit 数
    //   `(BN/2)*nd4`，原实现无上界检查会越界读并写坏 Kp。加运行期上界守卫：越界 unit 读 0、
    //   由 `kv_commit_pair` 跳过写。NT=128（历史路径）时恒满足 u<units，逐字不变。
    if (u < units) {
      if (jr < S) {  // 越界写 0 字节（等价 cvt_e4m3(0)=0x00）
        size_t i0 = (((size_t)(qbase + jr)) * Hkv + hkv) * HD + dq;
        k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
        v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
      }
      if (jr + 1 < S) {
        size_t i1 = (((size_t)(qbase + jr + 1)) * Hkv + hkv) * HD + dq;
        k1 = *reinterpret_cast<const uint32_t*>(k8 + i1);
        v1 = *reinterpret_cast<const uint32_t*>(v8 + i1);
      }
    }
    pk0[e] = k0;
    pk1[e] = k1;
    pv0[e] = v0;
    pv1[e] = v1;
  }
}

// O9c-2：`SW=true` 时 Ks/Vs 写 SW128（供 wgmma GEMM1/2 直读），否则写行主序。Kp 恒定。
template <int NPU, int HD, int BN, bool SW = false, int NT = THREADS>
__device__ __forceinline__ void kv_commit_pair(unsigned char* Ks, unsigned char* Vs,
                                               uint16_t* Kp, const uint32_t* pk0,
                                               const uint32_t* pk1, const uint32_t* pv0,
                                               const uint32_t* pv1, int tid, int asld,
                                               int psld) {
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;  // O48：与 prefetch 一致的上界（NTH>128 时避免越界写）
#pragma unroll
  for (int e = 0; e < NPU; ++e) {
    int u = tid + e * NT;
    if (u >= units) continue;   // 越界 unit 不落盘
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
template <int HD, int BN, bool SW = false, int NT = THREADS>
__device__ __forceinline__ void kv_load_pair(const unsigned char* __restrict__ k8,
                                             const unsigned char* __restrict__ v8,
                                             int j0, int S, int Hkv, int hkv, int qbase, int tid,
                                             unsigned char* Ks, unsigned char* Vs,
                                             uint16_t* Kp, int asld, int psld) {
  // O45：诊断探针（仅用于量「K/V 全局载入」的天花板；结果无意义，勿用于正确性）。
  //   `-DFA_SKIPKVL=1` 时跳过本 tile 的 K/V 全局读与配对重建，只保留骨架供计时。
#ifdef FA_SKIPKVL
  (void)k8; (void)v8; (void)j0; (void)S; (void)Hkv; (void)hkv; (void)qbase; (void)tid;
  (void)Ks; (void)Vs; (void)Kp; (void)asld; (void)psld;
  return;
#endif
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;
  for (int u = tid; u < units; u += NT) {
    int rp = u / nd4, dq = (u % nd4) * 4;
    int jr = j0 + rp * 2;
    uint32_t k0 = 0, k1 = 0, v0 = 0, v1 = 0;
    if (jr < S) {
      size_t i0 = (((size_t)(qbase + jr)) * Hkv + hkv) * HD + dq;
      k0 = *reinterpret_cast<const uint32_t*>(k8 + i0);
      v0 = *reinterpret_cast<const uint32_t*>(v8 + i0);
    }
    if (jr + 1 < S) {
      size_t i1 = (((size_t)(qbase + jr + 1)) * Hkv + hkv) * HD + dq;
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

// O51：K/V 的 `cp.async` 单缓冲回填（MLA/mma 路径）。把下一 tile 的 K/V 按 16B chunk
//   异步搬进行主序 `Ks/Vs`（HD 行 = HD/16 个 chunk，行距 asld）。行越界时零填充。
template <int HD, int BN, int NT>
__device__ __forceinline__ void kv_issue_async(const unsigned char* __restrict__ k8, int j0,
                                               int len, int Hkv, int hkv, int qbase, int tid,
                                               unsigned char* Dst, int asld) {
  const int nch = HD / 16;
  const int units = BN * nch;
  for (int u = tid; u < units; u += NT) {
    int jr = u / nch, cc = (u % nch) * 16;
    int jg = j0 + jr;
    bool ok = jg < len;
    size_t off = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + cc;
    cp_async16_z(Dst + jr * asld + cc, k8 + off, ok ? 16 : 0);
  }
}

// O51：从行主序 Ks 重建 Kp（K 配对布局 [BN/2][PSLD] uint16），与 `kv_load_pair` 的配对
//   字节顺序逐位一致（`byte_perm` 0x5140/0x7362）。
template <int HD, int BN, int NT>
__device__ __forceinline__ void kp_build_rows(const unsigned char* __restrict__ Ks,
                                              uint16_t* __restrict__ Kp, int asld, int psld,
                                              int tid) {
  const int nd4 = HD / 4;
  const int units = (BN / 2) * nd4;
  for (int u = tid; u < units; u += NT) {
    int rp = u / nd4, dq = (u % nd4) * 4;
    uint32_t k0 = *reinterpret_cast<const uint32_t*>(Ks + (rp * 2) * asld + dq);
    uint32_t k1 = *reinterpret_cast<const uint32_t*>(Ks + (rp * 2 + 1) * asld + dq);
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
// O64：量化单行的 warp 例程（从 `quantize_row_warp_kernel` 抽出，供融合 kernel 复用）。
//   与 O14 版**逐字相同**：float4 读、lane 内 amax、warp `shfl_xor` 树、scale=amax/fp8_max、
//   `cvt_*` 逐元素、`uchar4` 写回。数值逐位不变。
template <int VPT, bool E5M2>
__device__ __forceinline__ void quant_row_warp(const float* __restrict__ x,
                                               unsigned char* __restrict__ xq,
                                               float* __restrict__ scale, long long row,
                                               int lane) {
  constexpr int D = VPT * 32;  // 一行元素数（VPT=4→128，16→512）
  const float fp8_max = E5M2 ? kE5M2Max : kE4M3Max;
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

// 清零单行（VPT*32 个 fp32，float4 写）。
template <int VPT>
__device__ __forceinline__ void zero_row_warp(float* __restrict__ x, long long row, int lane) {
  constexpr int D = VPT * 32;
  float4* p = reinterpret_cast<float4*>(x + row * (long long)D);
  const float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int t = 0; t < VPT / 4; ++t) p[t * 32 + lane] = z;
}

// O66：`quant_row_warp<E5M2>` 的「顺便算 delta」版——量化 dO(E5M2) 的同时就地算
//   delta[row] = Σ_d O[d]·(deq_e5m2(do8[d])·dos[row])。
//   动机：delta 只依赖 dO/O 两项、且是逐 query 行的归约；与 dO 的量化同域、同 warp-per-row
//   几何。把两者合进同一个 warp 任务，可省掉一次 delta kernel 的 launch 与对 do8 的回读。
//   **数值与 `delta_warp_kernel` 逐位相同**：lane 累加顺序同为「t 外层（元素 t*128+lane*4）、
//   float4 内 x/y/z/w 内层」，随后同一 `__shfl_xor_sync` 树；`s == dos[row]`、`cvt`/`deq` 与
//   量化/delta 现用函数逐字一致（A/B 里 `max_abs(fused-vs-delta_warp) == 0`）。
template <int VPT>
__device__ __forceinline__ void quant_delta_row_warp(const float* __restrict__ x,
                                                     unsigned char* __restrict__ xq,
                                                     float* __restrict__ scale,
                                                     const float* __restrict__ o,
                                                     float* __restrict__ delta, long long row,
                                                     int lane) {
  constexpr int D = VPT * 32;
  const float fp8_max = kE5M2Max;
  const float4* xr4 = reinterpret_cast<const float4*>(x + row * (long long)D);
  const float4* or4 = reinterpret_cast<const float4*>(o + row * (long long)D);
  float v[VPT];
  float ov[VPT];
  float amax = 0.f;
#pragma unroll
  for (int t = 0; t < VPT / 4; ++t) {
    const float4 q = xr4[t * 32 + lane];
    const float4 oo = or4[t * 32 + lane];
    v[t * 4 + 0] = q.x;
    v[t * 4 + 1] = q.y;
    v[t * 4 + 2] = q.z;
    v[t * 4 + 3] = q.w;
    ov[t * 4 + 0] = oo.x;
    ov[t * 4 + 1] = oo.y;
    ov[t * 4 + 2] = oo.z;
    ov[t * 4 + 3] = oo.w;
    amax = fmaxf(amax, fmaxf(fmaxf(fabsf(q.x), fabsf(q.y)), fmaxf(fabsf(q.z), fabsf(q.w))));
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
  const float s = (amax > 0.f) ? (amax / fp8_max) : 1.f;
  if (lane == 0) scale[row] = s;
  unsigned char* out = xq + row * (long long)D;
  float acc = 0.f;
#pragma unroll
  for (int t = 0; t < VPT / 4; ++t) {
    uchar4 c;
    c.x = cvt_e5m2(v[t * 4 + 0] / s);
    c.y = cvt_e5m2(v[t * 4 + 1] / s);
    c.z = cvt_e5m2(v[t * 4 + 2] / s);
    c.w = cvt_e5m2(v[t * 4 + 3] / s);
    *reinterpret_cast<uchar4*>(out + (t * 32 + lane) * 4) = c;
    acc += ov[t * 4 + 0] * (deq_e5m2(c.x) * s);
    acc += ov[t * 4 + 1] * (deq_e5m2(c.y) * s);
    acc += ov[t * 4 + 2] * (deq_e5m2(c.z) * s);
    acc += ov[t * 4 + 3] * (deq_e5m2(c.w) * s);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) delta[row] = acc;
}

template <int VPT, bool E5M2>
__global__ void __launch_bounds__(128)
quantize_row_warp_kernel(const float* __restrict__ x, unsigned char* __restrict__ xq,
                         float* __restrict__ scale, long long nrows) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NW = 4;  // 128 线程 = 4 个 warp
  for (long long row = (long long)blockIdx.x * NW + warp; row < nrows;
       row += (long long)gridDim.x * NW) {
    quant_row_warp<VPT, E5M2>(x, xq, scale, row, lane);
  }
}

// O64：把「4 次输入量化（q/dO/k/v）+ 3 次累加缓冲清零（dQ/dK/dV）」融合成 **1 个 launch**。
//   动机：默认路径的 4 个 quant kernel + 3 个 `cudaMemset` 是 7 次串行 launch，各自在
//   S1024H32 上只到 ~57% DRAM、且被尾延迟截断（实测 quant 40.8µs + 尾/清零残留 ~31µs，
//   合计约占端到端 ~19%）。融合后 DRAM 流水连续、省 6 次 launch 的间隙与每次的尾波；
//   数值**逐位不变**（每行 amax/scale/cvt 与 O14 完全相同，清零只是写 0）。每 warp 一行；
//   任务序（rq=rows_q、rkv=rows_kv）：
//     [0, rq) 量化 Q(E4M3)  [rq, 2rq) 量化 dO(E5M2)
//     [2rq, 2rq+rkv) 量化 K(E4M3)  [.., 2rq+2rkv) 量化 V(E4M3)
//     [.., +rq) 清零 dQ  [.., +rkv) 清零 dK  [.., +rkv) 清零 dV
template <int VPT>
__global__ void __launch_bounds__(128)
quantize_zero_warp_kernel(const float* __restrict__ q, const float* __restrict__ kf,
                          const float* __restrict__ v, const float* __restrict__ dof,
                          unsigned char* __restrict__ q8, unsigned char* __restrict__ k8,
                          unsigned char* __restrict__ v8, unsigned char* __restrict__ do8,
                          float* __restrict__ qs, float* __restrict__ ks,
                          float* __restrict__ vs, float* __restrict__ dos,
                          float* __restrict__ dq, float* __restrict__ dk,
                          float* __restrict__ dv, long long rq, long long rkv) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NW = 4;  // 128 线程 = 4 个 warp
  const long long b1 = rq, b2 = 2 * rq, b3 = 2 * rq + rkv, b4 = 2 * rq + 2 * rkv;
  const long long b5 = b4 + rq, b6 = b5 + rkv, b7 = b6 + rkv;
  for (long long t = (long long)blockIdx.x * NW + warp; t < b7;
       t += (long long)gridDim.x * NW) {
    if (t < b1)
      quant_row_warp<VPT, false>(q, q8, qs, t, lane);
    else if (t < b2)
      quant_row_warp<VPT, true>(dof, do8, dos, t - b1, lane);
    else if (t < b3)
      quant_row_warp<VPT, false>(kf, k8, ks, t - b2, lane);
    else if (t < b4)
      quant_row_warp<VPT, false>(v, v8, vs, t - b3, lane);
    else if (t < b5)
      zero_row_warp<VPT>(dq, t - b4, lane);
    else if (t < b6)
      zero_row_warp<VPT>(dk, t - b5, lane);
    else
      zero_row_warp<VPT>(dv, t - b6, lane);
  }
}

// O66：把 `quantize_zero_warp_kernel` 的 dO 量化任务升级为「量化 dO + 顺便算 delta」
//   （`quant_delta_row_warp`）。任务序与 O64 版**逐字相同**，只多 2 个入参 `o`/`delta`；
//   数值逐位不变（dO 的 amax/scale/cvt 逐字一致；delta 与 `delta_warp_kernel` 逐位一致）。
//   收益：默认路径少一次独立 `delta_warp_kernel` launch（并省掉对 do8 的一趟回读）。
template <int VPT>
__global__ void __launch_bounds__(128)
quantize_zero_delta_warp_kernel(const float* __restrict__ q, const float* __restrict__ kf,
                                const float* __restrict__ v, const float* __restrict__ dof,
                                const float* __restrict__ o, float* __restrict__ delta,
                                unsigned char* __restrict__ q8, unsigned char* __restrict__ k8,
                                unsigned char* __restrict__ v8, unsigned char* __restrict__ do8,
                                float* __restrict__ qs, float* __restrict__ ks,
                                float* __restrict__ vs, float* __restrict__ dos,
                                float* __restrict__ dq, float* __restrict__ dk,
                                float* __restrict__ dv, long long rq, long long rkv) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NW = 4;  // 128 线程 = 4 个 warp
  const long long b1 = rq, b2 = 2 * rq, b3 = 2 * rq + rkv, b4 = 2 * rq + 2 * rkv;
  const long long b5 = b4 + rq, b6 = b5 + rkv, b7 = b6 + rkv;
  for (long long t = (long long)blockIdx.x * NW + warp; t < b7;
       t += (long long)gridDim.x * NW) {
    if (t < b1)
      quant_row_warp<VPT, false>(q, q8, qs, t, lane);
    else if (t < b2)
      quant_delta_row_warp<VPT>(dof, do8, dos, o, delta, t - b1, lane);
    else if (t < b3)
      quant_row_warp<VPT, false>(kf, k8, ks, t - b2, lane);
    else if (t < b4)
      quant_row_warp<VPT, false>(v, v8, vs, t - b3, lane);
    else if (t < b5)
      zero_row_warp<VPT>(dq, t - b4, lane);
    else if (t < b6)
      zero_row_warp<VPT>(dk, t - b5, lane);
    else
      zero_row_warp<VPT>(dv, t - b6, lane);
  }
}
// O109：`quantize_zero_delta_warp_kernel` 的**分相版**（phase split）。
//   动机：LSE（`lse_mma_kernel_bal_tma`，SM 73%、DRAM 4%）与量化（DRAM 84%、SM 62%）
//   在资源上互补，但 LSE 依赖 q8/k8——只要先量化 Q/K，LSE 即可与「dO+delta / V / 清零」
//   的重活在不同 stream 上重叠。本 kernel 只把 O64/O66 的 7 段任务拆成两相：
//     phase==0：[0,rq) 量化 Q(E4M3)  [rq,rq+rkv) 量化 K(E4M3)
//     phase==1：[0,rq) 量化 dO(E5M2)+delta  [rq,rq+rkv) 量化 V(E4M3)
//               [..,+rq) 清零 dQ  [..,+rkv) 清零 dK  [..,+rkv) 清零 dV
//   每段的逐行例程（amax/scale/cvt/delta 次序）与合并版**逐字相同** ⇒ q8/k8/v8/do8/qs/ks/
//   vs/dos/delta 与合并版**逐位相同**，零值也相同。只改「谁在哪个 stream 上跑」。
template <int VPT>
__global__ void __launch_bounds__(128)
quantize_zero_delta_phase_kernel(const float* __restrict__ q, const float* __restrict__ kf,
                                 const float* __restrict__ v, const float* __restrict__ dof,
                                 const float* __restrict__ o, float* __restrict__ delta,
                                 unsigned char* __restrict__ q8, unsigned char* __restrict__ k8,
                                 unsigned char* __restrict__ v8, unsigned char* __restrict__ do8,
                                 float* __restrict__ qs, float* __restrict__ ks,
                                 float* __restrict__ vs, float* __restrict__ dos,
                                 float* __restrict__ dq, float* __restrict__ dk,
                                 float* __restrict__ dv, long long rq, long long rkv, int phase) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  constexpr int NW = 4;  // 128 线程 = 4 个 warp
  if (phase == 0) {
    const long long n0 = rq + rkv;
    for (long long t = (long long)blockIdx.x * NW + warp; t < n0;
         t += (long long)gridDim.x * NW) {
      if (t < rq)
        quant_row_warp<VPT, false>(q, q8, qs, t, lane);
      else
        quant_row_warp<VPT, false>(kf, k8, ks, t - rq, lane);
    }
  } else {
    const long long b1 = rq, b2 = rq + rkv, b3 = b2 + rq, b4 = b3 + rkv, b5 = b4 + rkv;
    for (long long t = (long long)blockIdx.x * NW + warp; t < b5;
         t += (long long)gridDim.x * NW) {
      if (t < b1)
        quant_delta_row_warp<VPT>(dof, do8, dos, o, delta, t, lane);
      else if (t < b2)
        quant_row_warp<VPT, false>(v, v8, vs, t - b1, lane);
      else if (t < b3)
        zero_row_warp<VPT>(dq, t - b2, lane);
      else if (t < b4)
        zero_row_warp<VPT>(dk, t - b3, lane);
      else
        zero_row_warp<VPT>(dv, t - b4, lane);
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
               float* __restrict__ lse, int S, int H, int Hkv, float scale, int causal,
               const int* __restrict__ cu_seqlens = nullptr) {
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
  // VARLEN：cu_seqlens 给出本序列在 packed [T,H,D] 的 token 基址与长度；nullptr 退化为
  //   定长 b*S/S。非 causal 的 varlen 走本 kernel（因果的负载均衡由 bal_wgmma 承担；
  //   非 causal 各 m 块工作量恒为 nblk 个 tile，本就均衡）。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  if (m0 >= len) return;  // VARLEN：超出本序列长度的 m 块直接退出

  // 载入 Q 块（rowwise scale 同步取回）
  for (int i = tid; i < LBM * HD; i += THREADS) {
    int r = i / HD, d = i % HD;
    int qi = m0 + r;
    Qs[r * ASLD + d] =
        (qi < len) ? q8[(((size_t)(qbase + qi)) * H + h) * HD + d] : cvt_e4m3(0.f);
  }
  if (tid < LBM)
    qs_s[tid] = (m0 + tid < len) ? qs[((size_t)(qbase + m0 + tid)) * H + h] : 1.f;
  __syncthreads();

  const int ncols = causal ? min(len, m0 + LBM) : len;
  const int ntiles = (ncols + LBN - 1) / LBN;
  float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

  for (int nt = 0; nt < ntiles; ++nt) {
    const int j0 = nt * LBN;
    for (int i = tid; i < LBN * HD; i += THREADS) {
      int r = i / HD, d = i % HD;
      int jg = j0 + r;
      Ks[r * ASLD + d] =
          (jg < len) ? k8[(((size_t)(qbase + jg)) * Hkv + hkv) * HD + d] : cvt_e4m3(0.f);
    }
    if (tid < LBN)
      ks_s[tid] = (j0 + tid < len) ? ks[((size_t)(qbase + j0 + tid)) * Hkv + hkv] : 1.f;
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
        if (qi < len && jg < len && !(causal && jg > qi))
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
      if (qi < len) lse[((size_t)(qbase + qi)) * H + h] = m + flog(l);
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
// O54：`FULL=true` 时本 kernel 也服务 **非 causal（full）** 路径——每个 CTA 只处理一个 m 块
//   （grid.x = nblk，无镜像配对），`ncols=len`、无因果掩码，其余（cp.async 双缓冲、O40 split）
//   逐字复用。`FULL=false`（默认）编译出与原版逐位相同的代码。
// O58：模板参数化为 `<HD,PIPE,FULL,NTH,LBN_>`（对齐 fp16/bf16 的 O56 改造）：`LBM_ = (NTH/32)*16`
//   （128→64，256→128）、`MTN = LBN_/8`、`KVL = LBN_*ASLD`；默认档 `<HD,PIPE,FULL,128,64>` 与
//   O54 版**逐位等价**。用于 causal 与 full 的「2 CTA/SM（LBN=16/32）与 8-warp（NTH=256/LBM=128）」
//   几何（O58），以及 fp8 侧的 O56/O57 同构复现。
template <int HD, int PIPE, bool FULL = false, int NTH = THREADS, int LBN_ = LBN>
__global__ void __launch_bounds__(NTH)
lse_mma_kernel_bal(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                   const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                   float* __restrict__ lse, int S, int H, int Hkv, float scale,
                   const int* __restrict__ cu_seqlens = nullptr,
                   float* __restrict__ lse_part = nullptr, int ksplit = 1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int KVL  = LBN_ * ASLD;
  constexpr int HDV  = HD / 16;   // 每行 16B（16 个 fp8）unit 数
  constexpr int LBM_ = (NTH / 32) * 16;   // warp 数×16 行（128→64，256→128）
  constexpr int MTN  = LBN_ / 8;          // 每 warp 的 n8 tile 数
  extern __shared__ __align__(16) char smem[];
  unsigned char* Qs = reinterpret_cast<unsigned char*>(smem);
  unsigned char* Ks = Qs + LBM_ * ASLD;                              // PIPE=1：2×KVL
  float* qs_s = reinterpret_cast<float*>(Ks + (PIPE ? 2 : 1) * KVL);  // [LBM_]
  float* ks_s = qs_s + LBM_;                                         // PIPE=1：2×LBN_

  // O39：K 维 split。`ksplit>1` 时 grid.z = B*ksplit，每个 (pair,ks) CTA 只扫本 m 块 K 范围
  //   的第 ks 个连续 tile 切片，部分 (m,l) 写 `lse_part`，由 `lse_split_merge_kernel` 汇总。
  //   `ksplit==1` 时 b=blockIdx.z、ksp=0，逐位退化为 O11 原路径。
  const int pair = blockIdx.x, h = blockIdx.y;
  const int b = blockIdx.z / ksplit, ksp = blockIdx.z % ksplit;
  // VARLEN（第 80 轮）：cu_seqlens 给出本序列在 packed [T,H,D] 的 token 基址与长度；
  // nullptr 逐式退化为定长（qbase=b*S、len=S），定长路径逐位不变。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk  = (len + LBM_ - 1) / LBM_;
  // O54：FULL 一个 CTA 一个 m 块；causal 一行镜像对。
  if constexpr (FULL) { if (pair >= nblk) return; }
  else { if (pair >= (nblk + 1) / 2) return; }   // 短序列多余的对 CTA 直接退出
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  // 发本 m 块的 Q（PIPE=1 用 16B cp.async，行越界写 0=cvt_e4m3(0)）与 rowwise scale。
  auto issue_q = [&](int m0) {
#pragma unroll
    for (int u = tid; u < LBM_ * HDV; u += NTH) {
      const int row = u / HDV, c16 = u % HDV;
      const int qi = m0 + row;
      unsigned char* d = Qs + row * ASLD + c16 * 16;
      if (qi < len) {
        const unsigned char* s = q8 + (((size_t)(qbase + qi)) * H + h) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBM_) qs_s[tid] = (m0 + tid < len) ? qs[((size_t)(qbase + m0 + tid)) * H + h] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

  // 发一个 K tile（j0 起 LBN 行）到 Kd，并写本 tile 的 rowwise scale 到 KdS。
  auto issue_k = [&](unsigned char* Kd, float* KdS, int j0) {
#pragma unroll
    for (int u = tid; u < LBN_ * HDV; u += NTH) {
      const int row = u / HDV, c16 = u % HDV;
      const int jg = j0 + row;
      unsigned char* d = Kd + row * ASLD + c16 * 16;
      if (jg < len) {
        const unsigned char* s = k8 + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c16 * 16;
        if constexpr (PIPE) cp_async16(d, s);
        else {
#pragma unroll
          for (int e = 0; e < 16; ++e) d[e] = s[e];
        }
      } else {
        *reinterpret_cast<uint4*>(d) = make_uint4(0, 0, 0, 0);
      }
    }
    if (tid < LBN_)
      KdS[tid] = (j0 + tid < len) ? ks[((size_t)(qbase + j0 + tid)) * Hkv + hkv] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    if constexpr (FULL) { if (t == 1) continue; }   // O54：full 无镜像配对，只做 t=0
    const int mblk = FULL ? pair : ((t == 0) ? pair : (nblk - 1 - pair));
    if (!FULL && t == 1 && pair == nblk - 1 - pair) continue;  // 奇数 nblk 的中心块只做一次
    const int m0 = mblk * LBM_;
    issue_q(m0);

    const int ncols = FULL ? len : min(len, m0 + LBM_);
    const int ntiles = (ncols + LBN_ - 1) / LBN_;
    // O39：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;
    if constexpr (PIPE) {
      if (nuse > 0) issue_k(Ks, ks_s, nt0 * LBN_);
    }
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int j0 = nt * LBN_;
      unsigned char* Kt = Ks + (PIPE ? (rnt & 1) * KVL : 0);
      float* KtS = ks_s + (PIPE ? (rnt & 1) * LBN_ : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (rnt + 1 < nuse)
          issue_k(Ks + ((rnt + 1) & 1) * KVL, ks_s + ((rnt + 1) & 1) * LBN_, j0 + LBN_);
      } else {
        issue_k(Ks, ks_s, j0);
        __syncthreads();
      }

      float acc[1][MTN][4];
#pragma unroll
      for (int j = 0; j < MTN; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[0][j][q] = 0.f;
      mma_block<16, LBN_, HD, E4E4>(Qs, ASLD, Kt, ASLD, acc, wid, 0, lane);

      // O63/F5：tile 内两趟 softmax（先求本 lane 各 s 的列 max，再统一 rescale+exp）；
      //   与旧逐元素 online update 数学等价、只差 fp32 求和次序。覆盖 D=512 MLA/sm_90 回退。
      const float qsc0 = scale * qs_s[wid * 16 + g];
      const float qsc1 = scale * qs_s[wid * 16 + g + 8];
      const int qi0 = m0 + wid * 16 + g;
      const int qi1 = qi0 + 8;
      const bool ge0 = qi0 < len, ge1 = qi1 < len;
      float mloc0 = -INFINITY, mloc1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        const int jg0 = j0 + j * 8 + c2, jg1 = jg0 + 1;
        const bool m00 = FULL || jg0 <= qi0, m01 = FULL || jg1 <= qi0;
        const bool m10 = FULL || jg0 <= qi1, m11 = FULL || jg1 <= qi1;
        const bool c00 = ge0 && jg0 < len && m00;
        const bool c01 = ge0 && jg1 < len && m01;
        const bool c10 = ge1 && jg0 < len && m10;
        const bool c11 = ge1 && jg1 < len && m11;
        const float k0 = KtS[j * 8 + c2], k1 = KtS[j * 8 + c2 + 1];
        const float v0 = c00 ? acc[0][j][0] * qsc0 * k0 : -INFINITY;
        const float v1 = c01 ? acc[0][j][1] * qsc0 * k1 : -INFINITY;
        const float v2 = c10 ? acc[0][j][2] * qsc1 * k0 : -INFINITY;
        const float v3 = c11 ? acc[0][j][3] * qsc1 * k1 : -INFINITY;
        acc[0][j][0] = v0; acc[0][j][1] = v1; acc[0][j][2] = v2; acc[0][j][3] = v3;
        mloc0 = fmaxf(mloc0, fmaxf(v0, v1));
        mloc1 = fmaxf(mloc1, fmaxf(v2, v3));
      }
      const float mn0 = fmaxf(mrow[0], mloc0), mn1 = fmaxf(mrow[1], mloc1);
      const float mr0 = (mn0 == -INFINITY) ? 0.f : mn0;
      const float mr1 = (mn1 == -INFINITY) ? 0.f : mn1;
      float add0 = 0.f, add1 = 0.f;
#pragma unroll
      for (int j = 0; j < MTN; ++j) {
        add0 += fexp(acc[0][j][0] - mr0) + fexp(acc[0][j][1] - mr0);
        add1 += fexp(acc[0][j][2] - mr1) + fexp(acc[0][j][3] - mr1);
      }
      if (mn0 != -INFINITY) { lrow[0] = lrow[0] * fexp(mrow[0] - mn0) + add0; mrow[0] = mn0; }
      if (mn1 != -INFINITY) { lrow[1] = lrow[1] * fexp(mrow[1] - mn1) + add1; mrow[1] = mn1; }
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
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）。
    // O59：还必须 drain 本 m 块仍在飞的 cp.async——当本切片 `nuse==0` 时循环内的
    //   `wait_group 0` 不执行，t=1 的 issue_q 会与 t=0 的 Q 拷贝写同一 Qs 而竞争
    //   （实测 cfg6/LBN=16 + ksplit 下 LSE 出现 ~1e-2 的非确定性抖动）。
    if constexpr (PIPE) asm volatile("cp.async.wait_group 0;\n");
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
                         float* __restrict__ lse, int S, int H, int Hkv, float scale,
                         const int* __restrict__ cu_seqlens = nullptr,
                         const int* __restrict__ pt_b = nullptr,
                         const int* __restrict__ pt_pair = nullptr,
                         float* __restrict__ lse_part = nullptr, int ksplit = 1) {
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

  // O40：K 维 split（与 O38/O39 的 mma/TMA LSE 同构）。`ksplit>1` 时每个 (pair,mblk) 只扫
  //   本 m 块 K 范围的第 `ksp` 个连续 tile 切片，部分 (m,l) 写 `lse_part`，由
  //   `lse_split_merge_kernel` 汇总。非紧凑 varlen 用 blockIdx.z 编 `(b,ksp)`；紧凑网格
  //   （pt_b/pt_pair 非空）用 blockIdx.z 仅编 `ksp`（b 由表给出）。
  // VARLEN 均衡分块（第八十二轮）：定长/旧 varlen 用 (pair=blockIdx.x, h=blockIdx.y,
  //   b=blockIdx.z)；紧凑表 `pt_b/pt_pair` 只枚举每个序列的有效镜像对（pair < ceil(nblk/2)），
  //   grid = (total_pairs, H, 1)。消掉「以 maxlen 为界」时短序列的越界早退 CTA。
  const int pair = pt_pair ? pt_pair[blockIdx.x] : blockIdx.x;
  const int h = blockIdx.y;
  const int b = pt_b ? pt_b[blockIdx.x] : (blockIdx.z / ksplit);
  const int ksp = pt_b ? blockIdx.z : (blockIdx.z % ksplit);
  // VARLEN：cu_seqlens 给出每个序列在 packed [T,H,D] 里的 token 基址与长度。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk = (len + LBM - 1) / LBM;
  if (pair >= (nblk + 1) / 2) return;
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
      if (qi < len) {
        const unsigned char* s = q8 + (((size_t)(qbase + qi)) * H + h) * HD + c16 * 16;
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
      qs_s[tid] = (m0 + tid < len) ? qs[((size_t)(qbase + m0 + tid)) * H + h] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };
  // 发一个 K tile（j0 起 LBN 行）到 SW128 tile Kd，并写本 tile 的 rowwise scale。
  auto issue_k = [&](char* Kd, float* KdS, int j0) {
#pragma unroll
    for (int u = tid; u < LBN * HDV; u += THREADS) {
      const int row = u / HDV, c16 = u % HDV;
      const int jg = j0 + row;
      char* d = Kd + sw128_off_fp8(row, c16 * 16, HD);
      if (jg < len) {
        const unsigned char* s = k8 + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + c16 * 16;
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
      KdS[tid] = (j0 + tid < len) ? ks[((size_t)(qbase + j0 + tid)) * Hkv + hkv] : 1.f;
    if constexpr (PIPE) asm volatile("cp.async.commit_group;\n");
  };

#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int mblk = (t == 0) ? pair : (nblk - 1 - pair);
    if (t == 1 && pair == nblk - 1 - pair) continue;  // 奇数 nblk 的中心块只做一次
    const int m0 = mblk * LBM;
    issue_q(m0);

    const int ncols = min(len, m0 + LBM);
    const int ntiles = (ncols + LBN - 1) / LBN;
    // O40：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;
    if constexpr (PIPE) {
      if (nuse > 0) issue_k(Ks, ks_s, nt0 * LBN);
    }
    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};

    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int j0 = nt * LBN;
      char* Kt = Ks + (PIPE ? (rnt & 1) * TILE : 0);
      float* KtS = ks_s + (PIPE ? (rnt & 1) * LBN : 0);
      if constexpr (PIPE) {
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();
        if (rnt + 1 < nuse)
          issue_k(Ks + ((rnt + 1) & 1) * TILE, ks_s + ((rnt + 1) & 1) * LBN, j0 + LBN);
      } else {
        issue_k(Ks, ks_s, j0);
        __syncthreads();
      }

      float d[32];
      wgmma_qkt64_fp8(Qs, Kt, HD, d);

      // O63/F5：同 `lse_mma_kernel_bal_tma` 的「tile 内两趟 softmax」（先求本 lane 各 s 的
      //   列 max，再统一 rescale+exp）。数学等价、只差 fp32 求和次序；exp 64→34、去逐元素分支。
      const float qsc0 = scale * qs_s[wid * 16 + g];
      const float qsc1 = scale * qs_s[wid * 16 + g + 8];
      const int qi0 = m0 + wid * 16 + g;
      const int qi1 = qi0 + 8;
      const bool ge0 = qi0 < len, ge1 = qi1 < len;
      float mloc0 = -INFINITY, mloc1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int jg0 = j0 + j * 8 + c2, jg1 = jg0 + 1;
        const bool c00 = ge0 && jg0 < len && jg0 <= qi0;
        const bool c01 = ge0 && jg1 < len && jg1 <= qi0;
        const bool c10 = ge1 && jg0 < len && jg0 <= qi1;
        const bool c11 = ge1 && jg1 < len && jg1 <= qi1;
        const float k0 = KtS[j * 8 + c2], k1 = KtS[j * 8 + c2 + 1];
        const float v0 = c00 ? d[j * 4 + 0] * qsc0 * k0 : -INFINITY;
        const float v1 = c01 ? d[j * 4 + 1] * qsc0 * k1 : -INFINITY;
        const float v2 = c10 ? d[j * 4 + 2] * qsc1 * k0 : -INFINITY;
        const float v3 = c11 ? d[j * 4 + 3] * qsc1 * k1 : -INFINITY;
        d[j * 4 + 0] = v0; d[j * 4 + 1] = v1; d[j * 4 + 2] = v2; d[j * 4 + 3] = v3;
        mloc0 = fmaxf(mloc0, fmaxf(v0, v1));
        mloc1 = fmaxf(mloc1, fmaxf(v2, v3));
      }
      const float mn0 = fmaxf(mrow[0], mloc0), mn1 = fmaxf(mrow[1], mloc1);
      const float mr0 = (mn0 == -INFINITY) ? 0.f : mn0;
      const float mr1 = (mn1 == -INFINITY) ? 0.f : mn1;
      float add0 = 0.f, add1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        add0 += fexp(d[j * 4 + 0] - mr0) + fexp(d[j * 4 + 1] - mr0);
        add1 += fexp(d[j * 4 + 2] - mr1) + fexp(d[j * 4 + 3] - mr1);
      }
      if (mn0 != -INFINITY) { lrow[0] = lrow[0] * fexp(mrow[0] - mn0) + add0; mrow[0] = mn0; }
      if (mn1 != -INFINITY) { lrow[1] = lrow[1] * fexp(mrow[1] - mn1) + add1; mrow[1] = mn1; }
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
    // 切换到下一个 m 块前，确保所有 warp 读完 Qs/Ks（随后要覆盖）。
    // O59：还必须 drain 本 m 块仍在飞的 cp.async——当本切片 `nuse==0` 时循环内的
    //   `wait_group 0` 不执行，t=1 的 issue_q 会与 t=0 的 Q 拷贝写同一 Qs 而竞争
    //   （实测 cfg6/LBN=16 + ksplit 下 LSE 出现 ~1e-2 的非确定性抖动）。
    if constexpr (PIPE) asm volatile("cp.async.wait_group 0;\n");
    __syncthreads();
  }
}
#endif  // FA_WGMMA

// =============================================================================
// 2a''') O32：TMA 版 LSE（fp8）—— 对齐 fp16 O30 / bf16 O31，4D-TMA 载入 Q/K
// =============================================================================
// 动机：fp8 LSE 的 Q/K 载入从 O9c 起一直是「逐 16B `cp.async` + 地址运算」（`issue_q`/
//   `issue_k` 里对每个 `u` 算 `sw128_off_fp8` 再 `cp_async16`）。换成 **4D TMA** 后一条
//   bulk 指令搬完整块 [64 行][128 列]（1024B），省掉全部 load 指令与地址运算。
//
// fp8 与 fp16 O30 的关键差异：fp8 一行 128B = **一个 SW128 atom 的整行**（fp16 一行 128
//   个元素是 256B，必须拆成 2 个 K=64 chunk）。所以 fp8 的 Q/K tile 各**只需一次 4D TMA**
//   （box 内维 128 个 UINT8 = 128B），描述符仍是 `SBO=(HD/128)*1024=1024`、`layout_type=B128`。
//   `cuTensorMapEncodeTiled` 的 dtype 用 `CU_TENSOR_MAP_DATA_TYPE_UINT8`（CUDA 13 无
//   `FLOAT8_E4M3` 枚举；字节搬运与 wgmma 的 e4m3 解释互不影响，见 23 篇坑）。
//
// 数学与 `lse_mma_kernel_bal_wgmma` **完全一致**（镜像配对、online-softmax、4-lane `shfl`
//   归约、rowwise scale `qs*ks` 相乘顺序），只换搬运方式 ⇒ 数值应逐位相同。
//   仅 HD=128、causal（由 host 控制）。TMA asm 需 sm_90a，用 `FA_FP8_HAS_TMA` 包裹，
//   纯 `sm_90`（或仅 `-DFA_WGMMA`）构建时退化为空实现、host 不会 launch。
// =============================================================================
// O37：mbar/TMA helper 不再整体包在 `FA_WGMMA && FA_TMA` 里——主 kernel 的 TMA 版需要这些
//   名字在**非 TMA 构建**中也可见（body 里是 `if constexpr(TMA)` 的「名字可见但被丢弃」代码，
//   否则 ptxas 前的名字查找会失败）。asm 仍以 `FA_FP8_HAS_TMA`（= sm90a 且 FA_TMA）守卫。
#if defined(__CUDA_ARCH__) && defined(__CUDA_ARCH_FEAT_SM90_ALL) && defined(FA_TMA)
#define FA_FP8_HAS_TMA 1
#else
#define FA_FP8_HAS_TMA 0
#endif
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
#if FA_FP8_HAS_TMA
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_u32(bar)),
               "r"(count));
#else
  (void)bar; (void)count;
#endif
}
__device__ __forceinline__ void mbar_arrive_expect(uint64_t* bar, uint32_t bytes) {
#if FA_FP8_HAS_TMA
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smem_u32(bar)),
      "r"(bytes));
#else
  (void)bar; (void)bytes;
#endif
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t phase) {
#if FA_FP8_HAS_TMA
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
// O121（F3b WS）：mbarrier arrive（无 tx）。用于 warp specialization 的 **empty** barrier——
//   consumer 侧的全部 256 个 compute 线程各 arrive 一次（count=NCOMP），producer 等其完成
//   后才复用该 stage 的 smem（同 O120 冒烟 / 41 篇「empty count = 所有消费者线程数」）。
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
#if FA_FP8_HAS_TMA
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smem_u32(bar)));
#else
  (void)bar;
#endif
}
// O121（F3b WS）：命名 barrier —— 只让 **2 个 compute warpgroup（256 线程）** 同步，把 K/V
//   搬运的 producer warpgroup 移出 `__syncthreads` 域（O120 直接检验到的核心机制：同一份 L2
//   字节、同样 warp 数，仅「别把 producer 串进全 CTA barrier」就值 ~1.33×）。
__device__ __forceinline__ void named_bar_sync_compute() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("bar.sync 1, 256;\n" ::: "memory");
#endif
}
// 一条 4D TMA：全局张量 UINT8 dims={D,S,H,B}，把坐标 {k0,row,head,batch} 起的
// [64 行][128 列]（128 列 = 128B）搬进 dst（SW128 K-major）。
__device__ __forceinline__ void tma_load_4d(void* dst, const CUtensorMap* map, int k0,
                                            int r0, int hd, int b, uint64_t* bar) {
#if FA_FP8_HAS_TMA
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3, %4, %5}], [%6];\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar)));
#else
  (void)dst; (void)map; (void)k0; (void)r0; (void)hd; (void)b; (void)bar;
#endif
}
// O119：TMA cluster multicast（GQA/MQA 的 K/V 读摊薄）。leader 发一条，硬件按同一 smem 偏移把
//   数据复制进 mask 内各 CTA，并对**每个** CTA 的本地 mbarrier 补 `complete_tx`（各自 KS_SZ）。
__device__ __forceinline__ uint32_t fp8_cluster_rank() {
#if FA_FP8_HAS_TMA
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return r;
#else
  return 0;
#endif
}
__device__ __forceinline__ void fp8_cluster_sync() {
#if FA_FP8_HAS_TMA
  asm volatile("barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void fp8_fence_mbar_init() {
#if FA_FP8_HAS_TMA
  asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
#endif
}
__device__ __forceinline__ void tma_load_4d_mc(void* dst, const CUtensorMap* map, int k0,
                                               int r0, int hd, int b, uint64_t* bar,
                                               uint16_t mask) {
#if FA_FP8_HAS_TMA
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n" ::"r"(smem_u32(dst)),
      "l"((uint64_t)map), "r"(k0), "r"(r0), "r"(hd), "r"(b), "r"(smem_u32(bar)), "h"(mask));
#else
  (void)dst; (void)map; (void)k0; (void)r0; (void)hd; (void)b; (void)bar; (void)mask;
#endif
}

// O38/O39：把 K 维 split 的 LSE 部分结果 `(m_ks, l_ks)` 沿 `ks` 二次归约成最终 LSE。
//   `part` 布局 `[row][ks] -> (m,l)`（每行 `2*ksplit` 个 fp32），输出 `lse[row]=m+log(l)`。
//   online-softmax 合并（max 取大、sum 按 exp 重标定），`-inf/0` 安全（空 split 得 -inf/0）。
//   数学上与「单 CTA 顺序扫全部 K tile」完全等价，只差 fp32 求和次序。
//   放在 `FA_WGMMA && FA_TMA` 守卫**之外**：O39 起 mma 版 `lse_mma_kernel_bal`（D=512 MLA 与
//   `sm_90` 构建）也用 K 维 split，需要这个 merge kernel。
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

// O70（第 164 轮）：加 `bool FULL`——`FULL=true` 服务**非 causal（full）** 定长路径。
//   一个 CTA 只处理一个 m 块（grid.x = nblk，不做镜像配对；full 各 m 块工作量相同），
//   `ncols = S` 且不做 `jg <= qi` 因果掩码；其余（cp.async/TMA 双缓冲、rowwise scale、
//   tile 内两趟 softmax、4-lane shfl 归约）逐字复用。`FULL=false` 编译出与 O32/O38 逐位
//   相同的 causal 代码。
// O72（第 166 轮）：加 `const int* cu_seqlens`——VARLEN 支持。cu 非空时用 `cu_seqlens[b]` 作
//   packed [T,H,D] 的 token 基址、`cu_seqlens[b+1]-qbase` 作本序列长度；nullptr 时逐式退化为
//   定长（qbase=b*S、len=S），定长路径逐位不变。TMA 描述符建在 packed 张量上（dims={D,T,H,1}，
//   batch 维恒 0），行坐标 = qbase + m0/j0。O68 之前 varlen full 只走 cp.async 版
//   `lse_mma_kernel_bal`，本改动补上 varlen full D=128 的 4D-TMA 分支。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD, int PIPE = 1, bool FULL = false>
__global__ void __launch_bounds__(THREADS)
lse_mma_kernel_bal_tma(const __grid_constant__ CUtensorMap qmap,
                       const __grid_constant__ CUtensorMap kmap,
                       const float* __restrict__ qs, const float* __restrict__ ks,
                       float* __restrict__ lse, float* __restrict__ lse_part,
                       int S, int H, int Hkv, float scale, int ksplit,
                       const int* __restrict__ cu_seqlens = nullptr) {
  static_assert(HD % 128 == 0, "fp8 wgmma LSE TMA 需 HD 为 128 的整数倍");
  // O74（第 169 轮）：HD>128 支持——TMA box 内维固定 128B=128 fp8，`NCH=HD/128` 个 box，
  //   每个搬一整块 [LBM][128]，写到 `Qs + c*CHUNK`（CHUNK=(LBM/8)*1024）。HD=128 时 NCH=1、
  //   与历史单 box 路径**逐位相同**。
  constexpr int NCH = HD / 128;
  constexpr int CHUNK = (LBM / 8) * 1024;              // 单个 K=128 chunk 的 SW128 字节数（8KB）
  constexpr int TILE = NCH * CHUNK;                    // 单个 [LBM][HD] SW128 tile
  extern __shared__ char smem_raw[];
  // SW128 描述符 base_offset=0 要求 tile 1024B 对齐 → 手动对齐动态 smem 基址。
  const uint32_t a0 = smem_u32(smem_raw);
  const uint32_t pad = (1024u - (a0 & 1023u)) & 1023u;
  char* Qs = smem_raw + pad;
  char* Ks = Qs + TILE;  // PIPE=1：2*TILE；PIPE=0：TILE
  float* qs_s = reinterpret_cast<float*>(Ks + (PIPE ? 2 : 1) * TILE);
  float* ks_s = qs_s + LBM;  // PIPE=1：2*LBN
  uint64_t* qbar = reinterpret_cast<uint64_t*>(ks_s + (PIPE ? 2 : 1) * LBN);
  uint64_t* kbar = qbar + 1;

  // O38：K 维 split。grid = (pairs, H, B*ksplit)；每个 (pair,ks) CTA 只扫本 m 块 K 范围的
  //   第 ks 个连续切片（按 tile 粒度切分），把部分 (m,l) 写到 `lse_part`，由 merge kernel 汇总。
  //   ksplit==1 时切片即整段、直接写 `lse`（逐位退化为 O32 原路径）。
  const int pair = blockIdx.x, h = blockIdx.y;
  const int b = blockIdx.z / ksplit, ksp = blockIdx.z % ksplit;
  // O72：VARLEN——cu 给出本序列 packed token 基址/长度；nullptr 时逐式退化（定长逐位不变）。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  const int nblk  = (len + LBM - 1) / LBM;
  // O72：短序列多余的对 CTA 直接退出（定长时 grid.x==nblk ⇒ 恒不触发，逐位不变）。
  if constexpr (FULL) { if (pair >= nblk) return; }
  const int bq = cu_seqlens ? 0 : b;   // packed [T,H,D] 的 TMA batch 维恒 0（基址由 qbase 给）
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, c2 = (lane & 3) * 2;

  if (tid == 0) {
    mbar_init(qbar, 1);
    mbar_init(kbar, 1);
    if (PIPE) mbar_init(kbar + 1, 1);
  }
  __syncthreads();

  // 发本 m 块的 Q（NCH 次 4D TMA 搬整块）+ rowwise scale（标量 global 读，TMA 带不了）。
  auto issue_q = [&](int m0) {
    if (tid < LBM)
      qs_s[tid] = (m0 + tid < len) ? qs[((size_t)(qbase + m0 + tid)) * H + h] : 1.f;
    if (tid == 0) {
      mbar_arrive_expect(qbar, TILE);
      for (int c = 0; c < NCH; ++c)
        tma_load_4d(Qs + c * CHUNK, &qmap, c * 128, qbase + m0, h, bq, qbar);
    }
  };
  // 发一个 K tile（j0 起 LBN 行）+ 本 tile 的 rowwise scale。
  auto issue_k = [&](int stage, int j0) {
    if (tid < LBN)
      ks_s[stage * LBN + tid] =
          (j0 + tid < len) ? ks[((size_t)(qbase + j0 + tid)) * Hkv + hkv] : 1.f;
    if (tid == 0) {
      char* Kd = Ks + stage * TILE;
      mbar_arrive_expect(kbar + stage, TILE);
      for (int c = 0; c < NCH; ++c)
        tma_load_4d(Kd + c * CHUNK, &kmap, c * 128, qbase + j0, hkv, bq, kbar + stage);
    }
  };

  int quse = 0;
  // O63/F5：mbarrier 相位用两个标量（原 `int kuse[2]` 按运行期 `st=rnt&1` 动态下标会被
  //   ptxas 落到 local memory——同 F3-a 在 `fp8_mma_body` 修的坑；此处 LSE 也有）。
  int kuse0 = 0, kuse1 = 0;
#pragma unroll 1
  for (int t = 0; t < 2; ++t) {
    if constexpr (FULL) { if (t == 1) continue; }   // O70：full 无镜像配对，只做 t=0
    const int mblk = FULL ? pair : ((t == 0) ? pair : (nblk - 1 - pair));
    if (!FULL && t == 1 && pair == nblk - 1 - pair) break;
    const int m0 = mblk * LBM;
    const int ncols = FULL ? len : min(len, m0 + LBM);   // O72：len（定长时 == S，逐位不变）
    const int ntiles = (ncols + LBN - 1) / LBN;
    // O38：本 CTA 负责的 K tile 切片 [nt0, nt1)（连续，按 tile 数均分）。
    const int nt0 = (int)(((long)ntiles * ksp) / ksplit);
    const int nt1 = (int)(((long)ntiles * (ksp + 1)) / ksplit);
    const int nuse = nt1 - nt0;
    issue_q(m0);
    if (nuse > 0) issue_k(0, nt0 * LBN);
    mbar_wait(qbar, (uint32_t)(quse & 1)); quse++;

    float mrow[2] = {-INFINITY, -INFINITY}, lrow[2] = {0.f, 0.f};
#pragma unroll 1
    for (int rnt = 0; rnt < nuse; ++rnt) {
      const int nt = nt0 + rnt;
      const int st = PIPE ? (rnt & 1) : 0;
      const uint32_t kphase = st ? (uint32_t)(kuse1 & 1) : (uint32_t)(kuse0 & 1);
      if (st) kuse1++; else kuse0++;
      mbar_wait(kbar + st, kphase);
      __syncthreads();
      const int j0 = nt * LBN;
      char* Kt = Ks + st * TILE;
      float* KtS = ks_s + st * LBN;
      if (PIPE) {
        if (rnt + 1 < nuse) issue_k(st ^ 1, j0 + LBN);
      }

      float d[32];
      if constexpr (NCH == 1)
        wgmma_qkt64_fp8(Qs, Kt, HD, d);   // HD=128：与历史逐位相同
      else
        wgmma_qkt64_fp8_chunked(Qs, Kt, NCH, CHUNK, d);

      // O63/F5：tile 内「两趟 softmax」——先求本 lane 两个 s 的列 max（第一趟，顺手把
      //   掩码后的 S 写回 `d`），再统一 rescale + 求和（第二趟）。与旧「逐元素 online
      //   update」**数学等价**（只差 fp32 求和次序），但把每 tile 的 `fexp` 从 64 降到
      //   34、并删掉逐元素的 `if (sv != -INF)` 分支（SASS 里 107×BSSY/BSYNC）。
      const float qsc0 = scale * qs_s[wid * 16 + g];
      const float qsc1 = scale * qs_s[wid * 16 + g + 8];
      const int qi0 = m0 + wid * 16 + g;
      const int qi1 = qi0 + 8;
      const bool ge0 = qi0 < len, ge1 = qi1 < len;
      float mloc0 = -INFINITY, mloc1 = -INFINITY;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int jg0 = j0 + j * 8 + c2, jg1 = jg0 + 1;
        const bool c00 = ge0 && jg0 < len && (FULL || jg0 <= qi0);
        const bool c01 = ge0 && jg1 < len && (FULL || jg1 <= qi0);
        const bool c10 = ge1 && jg0 < len && (FULL || jg0 <= qi1);
        const bool c11 = ge1 && jg1 < len && (FULL || jg1 <= qi1);
        const float k0 = KtS[j * 8 + c2], k1 = KtS[j * 8 + c2 + 1];
        float v0 = c00 ? d[j * 4 + 0] * qsc0 * k0 : -INFINITY;
        float v1 = c01 ? d[j * 4 + 1] * qsc0 * k1 : -INFINITY;
        float v2 = c10 ? d[j * 4 + 2] * qsc1 * k0 : -INFINITY;
        float v3 = c11 ? d[j * 4 + 3] * qsc1 * k1 : -INFINITY;
        d[j * 4 + 0] = v0; d[j * 4 + 1] = v1; d[j * 4 + 2] = v2; d[j * 4 + 3] = v3;
        mloc0 = fmaxf(mloc0, fmaxf(v0, v1));
        mloc1 = fmaxf(mloc1, fmaxf(v2, v3));
      }
      const float mn0 = fmaxf(mrow[0], mloc0), mn1 = fmaxf(mrow[1], mloc1);
      // 全掩码（mn 仍为 -inf）时跳过本 tile 的更新；否则用安全的 `mr` 参考值算 exp（掩码
      // 位 -inf 自动得 0）。
      const float mr0 = (mn0 == -INFINITY) ? 0.f : mn0;
      const float mr1 = (mn1 == -INFINITY) ? 0.f : mn1;
      float add0 = 0.f, add1 = 0.f;
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        add0 += fexp(d[j * 4 + 0] - mr0) + fexp(d[j * 4 + 1] - mr0);
        add1 += fexp(d[j * 4 + 2] - mr1) + fexp(d[j * 4 + 3] - mr1);
      }
      if (mn0 != -INFINITY) { lrow[0] = lrow[0] * fexp(mrow[0] - mn0) + add0; mrow[0] = mn0; }
      if (mn1 != -INFINITY) { lrow[1] = lrow[1] * fexp(mrow[1] - mn1) + add1; mrow[1] = mn1; }
      __syncthreads();  // 所有 warp 读完本 tile 后才能覆盖该 stage / 下一轮 Q
      if (!PIPE && rnt + 1 < nuse) issue_k(0, j0 + LBN);
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
#endif  // defined(FA_WGMMA) && defined(FA_TMA)


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
// O27：`RCP=true` 时 fold 量化把「逐元素 fp32 精确除法」换成「每行一次 `__frcp_rn` + 乘法」。
//   fold 对每个 (m,j) 元素都算 `Ps*[dos] / scA`（dS2 用 `*[ks] / sc2`）；`scX` 是每输出行
//   （j / m）一个的常量，却让每个元素都发一条精确除法（ptxas 默认 prec-div，~10+ 指令）。
//   改成 sub4==0/sub2==0 lane 算一次 `1/scX`（`__frcp_rn`）再经 `__shfl` 广播，元素处用乘法。
//   数学等价；fp32 舍入偶有 1 ULP 差，而 fp8 只有 3 位尾数，cvt 结果几乎不变（实测 vs-ref 数值
//   与 RCP=false 完全一致）。`RCP=false` 保留精确除法作同 binary A/B；默认 true。
// O37：把主 kernel 的**函数体**抽成 device 函数，好让「cp.async 版」与「TMA 版」两个 `__global__`
//   壳复用同一份逻辑（避免 700 行重复）。`TMA=true` 时 Q/dO 用 4D-TMA 一次性搬进 SW128 tile
//   （`qmap`/`dmap` 为描述符，仅 WGMMA/HD=128 路径实例化），K/V 仍走原路径。
// O47：warp 网格由模板参数 `NTH`（线程数）+ `NWAR`（N 方向 warp 数）派生——对齐 fp16 O46。
//   默认 `NTH=128/NWAR=2`（2×2 网格）与历史逐字等价；fp8 MLA（D=512）用 `NTH=256/NWAR=4`
//   （2×4 网格）⇒ 1 CTA/SM 下 8 warp、每 scheduler 2 warp（O45 诊断的唯一剩余杠杆）。
//   `NWM=NTH/32/NWAR` 为 M 方向 warp 数。各 warp tile 由 NWM/NWAR 派生（见下方 GM*/GN*）。
// F7 第十一步（第 158 轮）：`DQONLY` —— 只算 dQ 的 Q-owner pass（dK/dV 全跳过）。动机：
//   F7 主体「两 kernel」选项（ROADMAP「下一步候选 ①(a)」）——KV-owner 出 dK/dV（red=0）＋
//   本 pass 出 dQ。dQ 在 `kRegDq` 下寄存器 owned（每元素单写者），DQONLY 时用 **plain store**
//   写出（red=0）。跳过 Ap/dS3 fold + GEMM3/4(dV/dK) + 其 epilogue；保留 GEMM1/2/5 与 dS2 fold。
// O93（第 188 轮，F6-①/LPT 候选 ③）：`HSWAP` —— 把默认稠密网格的 **x/y 轴对调**（head 走
//   blockIdx.x 快轴、m 块走 blockIdx.y），于是硬件派发顺序变成「**所有 head 的最贵 m 块先跑**」
//   （配合 O89 的 `mt_m` 反转表）= 跨 head 的**全局 LPT**；历史路径（head=blockIdx.y）是
//   「每个 head 内 m 降序」的锯齿形，head 边界处会重置。纯调度、不改任何数据/数学：dK/dV 仍是
//   跨 CTA 原子（可交换）⇒ 数值只在 fp8 噪声内。仅定长（`mt_b==nullptr`）用；`HSWAP=false`
//   与历史逐位相同。代价：相邻 CTA 落在不同 head ⇒ 理论损 L2 读局部性（S4096 的 K/V 全体
//   ~17MB 仍装得下 50MB L2，实测见 docs）。
// O119（第 213 轮）：`MCAST` = TMA cluster multicast 宽度（1=关）。沿 grid.x（HSWAP 时 = Q 头）
//   组 cluster，同 cluster 内各 CTA 读**同一 KV 头**的同一批 K/V tile（GQA/MQA：G=H/Hkv 个 Q 头
//   共享一份 K/V）⇒ 由 leader（rank0）发一条 `...multicast::cluster` 广播，K/V 的 L2 读扇区按
//   宽度摊薄（直打 O117 的 read 2.38× 差距里「K/V 重复读」那一半）。只动 K/V 的 4D-TMA，
//   数学/累加/reshape 一行未改；仅 `HSWAP && KVTMA && TMA && WGMMA && HD==128` 实例化。
//   **正确性前提**：cluster 内各 CTA 的 (mt,part) 相同 ⇒ causal 的 nt 调度逐迭代一致；每 tile
//   出/入 TMA 点用 `barrier.cluster` 锁步（否则 leader 可能比 follower 超前而串 tile）。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true,
           bool RCP = true, bool TMA = false, bool KVTMA = false, int NTH = THREADS,
           int NWAR = WN, bool KVPIPE = false, bool DET = false, bool DET_HALF = false,
           bool DQONLY = false, bool HSWAP = false, int MCAST = 1>
__device__ __forceinline__ void fp8_mma_body(const unsigned char* __restrict__ q8,
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
                      int causal, int ksplit, const int* __restrict__ cu_seqlens = nullptr,
                      const CUtensorMap* qmap = nullptr, const CUtensorMap* dmap = nullptr,
                      const CUtensorMap* kmap = nullptr, const CUtensorMap* vmap = nullptr,
                      const int* __restrict__ mt_b = nullptr,
                      const int* __restrict__ mt_m = nullptr,
                      float* __restrict__ dk_part = nullptr,
                      float* __restrict__ dv_part = nullptr, int nblk = 0,
                      float* __restrict__ dq_part = nullptr,
                      const int* __restrict__ part_base = nullptr,
                      const int* __restrict__ slot_tab = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  // O47：warp 网格派生。默认 128/2 ⇒ NWM=2 与历史 2×2 一致（逐字等价）。
  constexpr int NWM = NTH / 32 / NWAR;
  static_assert(NWM * NWAR * 32 == NTH, "NTH 必须 = NWM×NWAR×32");
  static_assert(WGMMA == false || (NTH == THREADS && NWAR == WN),
                "WGMMA 是 warpgroup 级，此 body 只在 4-warp/1 个 warpgroup 下正确");
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
  // O41：K/V TMA 时 K 双缓冲（2 个 SW128 stage），V 单缓冲。O51：KVPIPE（mma）同样 K 双缓冲。
  constexpr int KSTAGES = (KVTMA || KVPIPE) ? 2 : 1;
  // O85（第 180 轮）：HD>128 的 Q/dO 走 4D-TMA 时，一个 TMA box 只能搬 128 列 ⇒ smem 物理
  //   布局是 chunk-major `[k/128][row/8][8][128]`（与 O74 LSE 一致）；用 `NCH=HD/128` 个 box、
  //   每 box 落 `Qs + c*CHQ`（CHQ=(BM/8)*1024）。A 描述符改用 `wgmma_mn32_issue_cm`。
  //   HD=128 时 NCH=1、chunk-major==rg-major，路径与历史逐位相同。
  constexpr int NCH_Q = HD / 128;
  constexpr int CHQ = (BM / 8) * 1024;
  constexpr bool kQChunk = TMA && (HD > 128);
  // O84（第 179 轮）：放开 WGMMA 主 kernel 到 HD=256。SW128 K-major helper（`sw128_off_fp8`
  //   /`sw128_k32_addr`/`make_desc_sw128_fp8`）与 `wgmma_mn32_issue`/`wgmma_qkt64_fp8` 本就按
  //   `SBO=(HD/128)*1024`、k32 步进 `(s>>2)*1024+(s&3)*32` 编写，HD 为 128 的整数倍即正确
  //   （K=256 的 canonical 布局 [row/8][2 k-blocks][8][128]，rg 跨步 = 2048 = SBO）。此前只
  //   在 HD=128 实例化过（D=256 走 mma 后端），故加锁保守。GEMM3/4/5 仍 mma（见 kWg5/kWg34）。
  static_assert(!WGMMA || (HD == 128 || HD == 256), "WGMMA 主 kernel 只做 HD=128/256");
  static_assert(!KVTMA || (TMA && WGMMA && HD == 128),
                "K/V TMA 只在 Q/dO-TMA + WGMMA + HD=128 路径");
  // O119：multicast 只对「HSWAP（head=blockIdx.x）+ KVTMA」实例化（cluster 沿 x 分组 Q 头）。
  static_assert(MCAST == 1 || (KVTMA && TMA && WGMMA && HD == 128 && HSWAP),
                "MCAST>1 只在 HSWAP+KVTMA+TMA+WGMMA+HD=128 路径");
  constexpr uint16_t kMCMask = (uint16_t)((1u << (MCAST > 1 ? MCAST : 1)) - 1);
  // O51：K/V cp.async 回填流水只用于 mma 后端（非 WGMMA/TMA），目前实例化于 MLA（HD=512）。
  static_assert(!KVPIPE || (!WGMMA && !TMA && !KVTMA), "KVPIPE 只用于 mma 后端");
  // GEMM3/4/5 的输出 N 维 = head_dim；每遍处理 NTW = WN*64 = 128 列，共 HD/NTW 遍。
  constexpr int NTW = WN * 64;
  static_assert(HD % NTW == 0, "HD 必须是 128 的整数倍");
  // O21：主 kernel 的 KV tile BN 参数化（原写死 BN=32）。O47：warp 网格从写死 2×2 改为由
  //   NWM/NWAR 派生——默认 128/2（2×2）与历史逐字一致；MLA 用 256/4（2×4）。
  //   GEMM1/2 的 warp 块 = (BM/NWM, BN/NWAR)；GEMM3/4 输出 M=BN（KV 行），warp 块 =
  //   (BN/NWM, NTW/NWAR)；GEMM5 输出 M=BM，warp 块 = (BM/NWM, NTW/NWAR)。
  constexpr int GM1 = BM / NWM, GN1 = BN / NWAR;     // GEMM1/2 warp tile
  constexpr int GM34 = BN / NWM, GN34 = NTW / NWAR;  // GEMM3/4 warp tile（N-tile 内）
  constexpr int GM5 = BM / NWM, GN5 = NTW / NWAR;    // GEMM5 warp tile（N-tile 内）
  static_assert(BM % NWM == 0 && BN % NWAR == 0 && NTW % NWAR == 0, "warp tile 必须整除");
  constexpr int MTM = GM1 / 16, MTN = GN1 / 8;        // GEMM1/2 每 warp 的 m/n-tile 数
  constexpr int MTM34 = GM34 / 16, NTM34 = GN34 / 8;  // GEMM3/4 每 warp 的 m/n-tile 数
  constexpr int MTM5 = GM5 / 16, NTM5 = GN5 / 8;      // GEMM5 每 warp 的 m/n-tile 数
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
  // O42：bulk-reduce 路径（见 Fp8Cfg 上方 FA_BULKRED 说明）。仅 kvtma 快路 + BN=32
  //   （per-warp staging 行映射要求 MTM34==1、每 warp 16 行 × 64 列）。
  // O45：本条件曾开放到 HD=512 的 mma 路径再测一次（D=512 的 L1/TEX 仅 ~20%，staging 的
  //   smem 往返看似有空间）——**实测仍是负结果**：S1024H2 main 0.2004→0.2400ms（0.835×），
  //   与 O42 在 D=128 的结论一致。故默认仍关；`-DFA_BULKRED=1` 复现。
  constexpr bool kBulkRed = FA_BULKRED && (BN == 32) && !FA_ILV34 &&
                            ((HD == 128) ? (TMA && WGMMA) : true);
  // O81（F3b）：GEMM5(dQ) 走 wgmma RS。仅 KVTMA+WGMMA+HD=128+BN=32 非 DET/DQONLY。
  //   由编译期总闸 `FA_WGMMA5` 控制（默认 0），`-DFA_WGMMA5=1` 同 binary A/B。
  constexpr bool kWg5 = FA_WGMMA5 && WGMMA && KVTMA && (HD == 128) && (BN == 32) &&
                        !DQONLY && !DET && kRegDq;
  // O82（F3b 主体续）：GEMM3(dV)/GEMM4(dK) 走 wgmma RS（A 寄存器、M 零填充到 64）。门控同 kWg5，
  //   另需 !ILV34/BULKRED（这两者与 wgmma epilogue 不兼容）。`-DFA_WGMMA34=0` 同 binary A/B。
  constexpr bool kWg34 = FA_WGMMA34 && WGMMA && KVTMA && (HD == 128) && (BN == 32) &&
                         !DQONLY && !DET && !FA_ILV34 && !kBulkRed && !FA_R4;

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
  unsigned char* Vs  = Ks + KSTAGES * KS_SZ;
  unsigned char* dOs = Vs + KS_SZ;
  // O4b：Kt/Qt/dOt 三个转置副本 -> Qp/dOp/Kp 三个 K 配对布局（uint16，行距 PSLD）。
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(dOs + QS_SZ);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + Cfg::qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + Cfg::qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + Cfg::kp_bytes;
  unsigned char* Ap  = Vs;                   // O2：复用 GEMM2 后死亡的 Vs
  unsigned char* dS3 = Ks;                   // O2：复用 GEMM1 后死亡的 Ks（KVTMA 时按 stage 重指）
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
  float* Ps = scales + kNScale;              // P fp32 [BM][PSS]
  float* Ss = Ps + BM * PSS;                 // dS fp32 [BM][PSS]
  // O51：KVPIPE 时把 Ap/dS3 的 smem 别名拆开（各给独立缓冲，落在 Ss 之后），于是 Ks/Vs 在
  //   GEMM1/2 结束后即可被 cp.async 覆写。`Ap` 8B 对齐（上面所有区段尺寸均为 8 的倍数）。
  if constexpr (KVPIPE) {
    Ap = reinterpret_cast<unsigned char*>(Ss + BM * PSS);
    dS3 = Ap + BN * QTS;
  }
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;
  // O37：TMA 版 Q/dO 的两个 mbarrier，落在 `smem_bytes_wgmma` 之外额外分配的 64B 区。
  uint64_t* qbars = reinterpret_cast<uint64_t*>(Ss + BM * PSS);

  // O42：per-warp smem staging（复用在 fold 之后死亡的 `Ps`/`Ss` 区，共 2*BM*PSS 个 float）。
  //   每 warp 一块 [BN/2][STGS]，STGS=68（64+4 padding）消写冲突；行 16B 对齐、每行 64 float。
  constexpr int STGR = BN / 2, STGC = 64, STGS = STGC + 4;
  float* pstg = reinterpret_cast<float*>(
      (reinterpret_cast<uintptr_t>(Ps) + 15u) & ~(uintptr_t)15u);
  static_assert(!kBulkRed ||
                    (MTM34 == 1 &&
                     4 * STGR * STGS * (int)sizeof(float) + 16 <= 2 * BM * PSS * (int)sizeof(float)),
                "O42 bulkred staging 放不下 Ps/Ss（需 ≤ 2*BM*PSS 字节）");

  // ---- O2b：N 方向切块（split-K）。同一 (mblk,h,b) 的 K/V 列块 [0,ntiles) 被均分给
  //      ksplit 个 CTA；各自只算自己那一段，dQ/dK/dV 仍用跨 CTA 的 fp32 atomicAdd 汇总。
  //      小 S 时把 grid 从 S/BM×H 抬到 ksplit 倍，消「grid 不足一整个波」的空 SM；
  //      大 S 时用来削尾波（partial wave）。各部分数学上仍是同一个和，只是 fp 加法次序略变。----
  // VARLEN 均衡分块（第八十二轮）：定长路径用 (mblk = blockIdx.x/ksplit, h = blockIdx.y,
  //   b = blockIdx.z)；varlen 路径改为「只发有效 tile」的**紧凑一维网格**——宿主按
  //   cu_seqlens 枚举每个 b 的 nblk_b = ceil(len_b/BM) 个 m 块，按 (b,mblk) 主序写进
  //   `mt_b/mt_m`（保持同序列 K/V 的 L2 局部性），grid.x = total_mt*ksplit、grid.y = H、grid.z = 1。
  //   这消掉了「以 maxlen 为界」时短序列大量越界早退的死 CTA（强倾斜 b8 实测 8192→~1376），
  //   并让最贵的 m 块先调度（削尾波）。`mt_b==nullptr` 时逐式退化为定长/旧 varlen 行为。
  // O93：`HSWAP` 时把 head 放到快轴（grid=(H, nblk*ksplit, B)）⇒ 跨 head 全局 LPT；
  //   历史路径 grid=(nblk*ksplit, H, B)。两者只在「哪个 CTA 算哪个 (h,mblk,part)」上不同。
  // O95：`slot_tab` 非空时，`(ks,part,mt)` 从一个「slot → 编码」表解出（编码 = `mt<<8 | part<<4
  //   | ks`，ks/part≤15、mt=nblk≤255）。表由 host 按全局 LPT 顺序构建，于是**同一 grid 上不同
  //   m 块可有不同 ksplit**（贵块多切、便宜块少切）——在 O93 的 ksplit=2 平衡点上再省 Q/dO
  //   重读。纯调度：非 DET 默认路径的 dQ/dK/dV 仍是可交换的跨 CTA `atomicAdd` ⇒ 数值只在
  //   fp8 噪声内。`slot_tab==nullptr` 时逐式退化为 O93/历史路径（位不变）。
  int ks_eff = ksplit;
  int part, mt;
  if (slot_tab) {
    const int sidx = HSWAP ? blockIdx.y : blockIdx.x;
    const int enc = slot_tab[sidx];
    ks_eff = enc & 15;
    part = (enc >> 4) & 15;
    mt = enc >> 8;
  } else {
    part = HSWAP ? (blockIdx.y % ksplit) : (blockIdx.x % ksplit);
    mt   = HSWAP ? (blockIdx.y / ksplit) : (blockIdx.x / ksplit);
  }
  const int mblk = mt_m ? mt_m[mt] : mt;
  const int h = HSWAP ? blockIdx.x : blockIdx.y;
  const int b = mt_b ? mt_b[mt] : blockIdx.z;
  // VARLEN：cu_seqlens 给出每个序列在 packed [T,H,D] 的 token 基址与长度。
  const int qbase = cu_seqlens ? cu_seqlens[b] : b * S;
  const int len   = cu_seqlens ? (cu_seqlens[b + 1] - qbase) : S;
  // P5-3：GQA/MQA——Q 头 h 对应 KV 头 hkv；Q/dO/dQ 用 H，K/V/dK/dV 用 Hkv。
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wr = wid / NWAR, wc = wid % NWAR;  // O47：warp 网格 NWM×NWAR
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;
  if (m0 >= len) return;  // VARLEN：超出本序列长度的 m 块直接退出
  // P3-4o：varlen DET 的 partial 支持 **compact per-sequence 布局**（`part_base` 非空）——每序列
  //   b 的 partial 只占 `H*nblk_b*len_b` 个 token 行（而非全局 `maxlen/nblk_max` stride），缩小
  //   地址跨度、提高二次归约的 DRAM 局部性。`part_base` 是行前缀和（行单位，见 host）；
  //   本序列的 m 块数 `nblk_seq = ceil(len_b/BM)`。空指针时逐式退化为定长/旧 varlen 布局。
  const int nblk_seq = cu_seqlens ? (len + BM - 1) / BM : nblk;

  const int ncols = causal ? min(len, m0 + BM) : len;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ks_eff;
  const int nt_end = (part + 1) * ntiles / ks_eff;
  if (nt_end <= nt_begin) return;  // 该 part 无 tile（causal 下小 mblk 可能被切空）

  // ---- 载入 Q/dO：原始行 [m][d] 写 Qs/dOs（供 GEMM1/2 的 A）+ 打包行对写 Qp/dOp
  //      （供 GEMM4/3 的 B，ldmatrix.trans）。行对用 __byte_perm 交织，4B 一次。----
  // O41：K/V TMA 的 mbarrier 相位计数器（K 两个 stage 各一个、V 一个）。非 KVTMA 不用。
  // F3-a：`kuse` 原为 `int kuse[2]` 且按运行期 `stg^1` 动态下标 → ptxas 落到 **local memory**
  //   （栈帧 8B + 每 tile LDL/STL）。改成两个标量 + 运行期三元选择，去 local 化、语义逐位不变。
  int kuse0 = 0, kuse1 = 0;
  int vuse = 0;
  (void)kuse0;
  (void)kuse1;
  (void)vuse;
  if constexpr (TMA) {
    static_assert(WGMMA, "fp8 Q/dO TMA 只在 WGMMA(SW128) 路径");
    // O37：4D-TMA 一次性把 Q/dO 搬进 SW128 tile（fp8 一行 128B = 一个 SW128 atom 的整行），
    //   再从 smem 重建 Qp/dOp 的 K 配对布局（供 GEMM3/4 的 `ldmatrix.x2.trans`）。
    // O41：KVTMA 时 K/V 也由 4D-TMA 搬入（K 双缓冲、V 单缓冲），Kp 从 SW128 K tile 重建。
    if (tid == 0) {
      mbar_init(qbars + 0, 1);
      mbar_init(qbars + 1, 1);
      if (KVTMA) { mbar_init(qbars + 2, 1); mbar_init(qbars + 3, 1); mbar_init(qbars + 4, 1); }
      if (MCAST > 1) fp8_fence_mbar_init();
    }
    __syncthreads();
    if (MCAST > 1) fp8_cluster_sync();  // 远端 mbarrier 必须已 init+可见，才能被 multicast 补 tx
    if (tid == 0) {
      // O37：fp8 一行 128B = 一个 SW128 atom 的整行 ⇒ HD=128 一次 box 搬完整块。
      // O85：HD>128 时 `NCH_Q` 个 box 各搬 128 列（同一 mbarrier、expect 总量），
      //   落位 `Qs + c*CHQ`，物理布局 chunk-major（与 LSE O74 同款）。
      mbar_arrive_expect(qbars + 0, QS_SZ);
      for (int c = 0; c < NCH_Q; ++c)
        tma_load_4d(Qs + c * CHQ, qmap, c * 128, m0, h, b, qbars + 0);
      mbar_arrive_expect(qbars + 1, QS_SZ);
      for (int c = 0; c < NCH_Q; ++c)
        tma_load_4d(dOs + c * CHQ, dmap, c * 128, m0, h, b, qbars + 1);
      if (KVTMA) {
        // K[nt_begin] -> stage 0；K[nt_begin+1] -> stage 1（如有）；V[nt_begin] -> Vs。
        // O119：MCAST>1 时本 CTA 仍各自 arrive_expect（本地 mbarrier），但只有 leader(rank0) 发
        //   multicast，硬件按同一 smem 偏移广播到 cluster 内各 CTA 的 Ks/Vs 与各自 mbarrier。
        const bool ld = (MCAST == 1) || (fp8_cluster_rank() == 0);
        mbar_arrive_expect(qbars + 2, KS_SZ);
        if (ld) {
          if (MCAST > 1) tma_load_4d_mc(Ks, kmap, 0, nt_begin * BN, hkv, b, qbars + 2, kMCMask);
          else tma_load_4d(Ks, kmap, 0, nt_begin * BN, hkv, b, qbars + 2);
        }
        if (nt_begin + 1 < nt_end) {
          mbar_arrive_expect(qbars + 3, KS_SZ);
          if (ld) {
            if (MCAST > 1) tma_load_4d_mc(Ks + KS_SZ, kmap, 0, (nt_begin + 1) * BN, hkv, b,
                                          qbars + 3, kMCMask);
            else tma_load_4d(Ks + KS_SZ, kmap, 0, (nt_begin + 1) * BN, hkv, b, qbars + 3);
          }
        }
        mbar_arrive_expect(qbars + 4, KS_SZ);
        if (ld) {
          if (MCAST > 1) tma_load_4d_mc(Vs, vmap, 0, nt_begin * BN, hkv, b, qbars + 4, kMCMask);
          else tma_load_4d(Vs, vmap, 0, nt_begin * BN, hkv, b, qbars + 4);
        }
      }
    }
    if (tid < BM) {
      int qi = m0 + tid;
      qs_s[tid] = (qi < len) ? qs[((size_t)(qbase + qi)) * H + h] : 1.f;
      dos_s[tid] = (qi < len) ? dos[((size_t)(qbase + qi)) * H + h] : 1.f;
    }
    mbar_wait(qbars + 0, 0);
    mbar_wait(qbars + 1, 0);
    if (KVTMA) {
      mbar_wait(qbars + 2, (uint32_t)(kuse0 & 1)); kuse0++;
      mbar_wait(qbars + 4, (uint32_t)(vuse & 1)); vuse++;
    }
    __syncthreads();
    const int nd4t = HD / 4;
    if constexpr (kWg34) {
      // O82：GEMM3/4 走 wgmma 时，把 Q/dO 从 SW128 逐字节转置成 Qᵀ/dOᵀ 的紧凑 no-swizzle
      //   K-major（[HD][BM]，K=BM=64），复用 Qp/dOp 的配对缓冲（8192B ≤ qp_bytes 8704B）。
      transpose_sw128_to_noswz<BM, HD>(reinterpret_cast<unsigned char*>(Qp), Qs, wid, lane);
      transpose_sw128_to_noswz<BM, HD>(reinterpret_cast<unsigned char*>(dOp), dOs, wid, lane);
      bulk_reduce_fence();
    } else {
    // O85：Q/dO 若为 chunk-major（HD>128 的 4D-TMA），用 `sw128c_off_fp8` 读源；HD=128 退化相同。
    auto qoff = [&](int r, int k) { return kQChunk ? sw128c_off_fp8(r, k, BM) : sw128_off_fp8(r, k, HD); };
    for (int u = tid; u < (BM / 2) * nd4t; u += NTH) {
      int rp = u / nd4t, dq = (u % nd4t) * 4;
      uint32_t q0 = *reinterpret_cast<const uint32_t*>(Qs + qoff(rp * 2, dq));
      uint32_t q1 = *reinterpret_cast<const uint32_t*>(Qs + qoff(rp * 2 + 1, dq));
      uint32_t o0 = *reinterpret_cast<const uint32_t*>(dOs + qoff(rp * 2, dq));
      uint32_t o1 = *reinterpret_cast<const uint32_t*>(dOs + qoff(rp * 2 + 1, dq));
      uint32_t* qpw = reinterpret_cast<uint32_t*>(Qp + rp * PSLD + dq);
      qpw[0] = __byte_perm(q0, q1, 0x5140);
      qpw[1] = __byte_perm(q0, q1, 0x7362);
      uint32_t* opw = reinterpret_cast<uint32_t*>(dOp + rp * PSLD + dq);
      opw[0] = __byte_perm(o0, o1, 0x5140);
      opw[1] = __byte_perm(o0, o1, 0x7362);
    }
    }
    if (KVTMA) {
      // O81：GEMM5 走 wgmma 时，从 SW128 K[0] 逐字节转置成 Kᵀ 的 INTERLEAVE K-major。
      if constexpr (kWg5) {
        transpose_sw128_to_inter<BN, HD>(reinterpret_cast<unsigned char*>(Kp), Ks, wid, lane);
        bulk_reduce_fence();
      } else {
        // 从 SW128 K[0] 重建 Kp（K 配对布局，供 GEMM5 的 ldmatrix.x2.trans）。
        for (int u = tid; u < (BN / 2) * nd4t; u += NTH) {
          int rp = u / nd4t, dq = (u % nd4t) * 4;
          uint32_t k0 = *reinterpret_cast<const uint32_t*>(Ks + sw128_off_fp8(rp * 2, dq, HD));
          uint32_t k1 = *reinterpret_cast<const uint32_t*>(Ks + sw128_off_fp8(rp * 2 + 1, dq, HD));
          uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * PSLD + dq);
          kpw[0] = __byte_perm(k0, k1, 0x5140);
          kpw[1] = __byte_perm(k0, k1, 0x7362);
        }
      }
    }
  } else {
  {
    const int nd4 = HD / 4;
    for (int u = tid; u < (BM / 2) * nd4; u += NTH) {
      int rp = u / nd4, dq = (u % nd4) * 4;
      int qa = m0 + rp * 2, qb = m0 + rp * 2 + 1;
      uint32_t q0 = 0, q1 = 0, o0 = 0, o1 = 0;
      if (qa < len) {
        size_t idx = (((size_t)(qbase + qa)) * H + h) * HD + dq;
        q0 = *reinterpret_cast<const uint32_t*>(q8 + idx);
        o0 = *reinterpret_cast<const uint32_t*>(do8 + idx);
      }
      if (qb < len) {
        size_t idx = (((size_t)(qbase + qb)) * H + h) * HD + dq;
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
    qs_s[tid] = (qi < len) ? qs[((size_t)(qbase + qi)) * H + h] : 1.f;
    dos_s[tid] = (qi < len) ? dos[((size_t)(qbase + qi)) * H + h] : 1.f;
  }
  }

  // ---- O3 prologue：寄存器预取本 part 首个 tile 并落盘；HD>128 时直接向量化读。----
  // ---- O4a：Q/dO 载入与 tile 的 K/V 落盘写的是互不重叠的 smem（Qs/Qp/dOs/dOp vs
  //            Ks/Vs/Kp），故把原来 prologue 的两处 __syncthreads 合并为一处。----
  uint32_t pk0[NPU], pk1[NPU], pv0[NPU], pv1[NPU];
  if constexpr (KVTMA) {
    // K/V 已由 4D-TMA 搬入 smem（Kp 也已在上面重建），这里只装本 tile 的 rowwise scale。
    (void)pk0; (void)pk1; (void)pv0; (void)pv1;
    if (tid < BN) {
      int jg = nt_begin * BN + tid;
      ks_s[tid] = (jg < len) ? ks[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
      vs_s[tid] = (jg < len) ? vs[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
    }
  } else if constexpr (KVPIPE) {
    // O51：首个 tile 的 K（stage 0）与 V 用 cp.async 搬入，稍后（本段末尾）wait + 重建 Kp。
    (void)pk0; (void)pk1; (void)pv0; (void)pv1;
    kv_issue_async<HD, BN, NTH>(k8, nt_begin * BN, len, Hkv, hkv, qbase, tid, Ks, ASLD);
    kv_issue_async<HD, BN, NTH>(v8, nt_begin * BN, len, Hkv, hkv, qbase, tid, Vs, ASLD);
    cp_async_commit();
  } else if (kPrefetch) {
    kv_prefetch_pair<NPU, HD, BN, NTH>(k8, v8, nt_begin * BN, len, Hkv, hkv, qbase, tid, pk0, pk1, pv0,
                              pv1);
    kv_commit_pair<NPU, HD, BN, WGMMA, NTH>(Ks, Vs, Kp, pk0, pk1, pv0, pv1, tid, ASLD, PSLD);
  } else {
    kv_load_pair<HD, BN, WGMMA, NTH>(k8, v8, nt_begin * BN, len, Hkv, hkv, qbase, tid, Ks, Vs, Kp, ASLD,
                         PSLD);
  }
  if (!KVTMA && tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < len) ? ks[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < len) ? vs[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
  }
  __syncthreads();
  if constexpr (KVPIPE) {
    cp_async_wait0();
    __syncthreads();
    kp_build_rows<HD, BN, NTH>(Ks, Kp, ASLD, PSLD, tid);
    __syncthreads();
  }

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
        const size_t idx = ((size_t)(qbase + qi)) * H + h;
        const bool ok = qi < len;
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
          const size_t idx = ((size_t)(qbase + qi)) * H + h;
          const bool ok = qi < len;
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
  // O81：wgmma GEMM5 的累加器 [n-tile 0..3][n8 0..3][q 0..3]（每线程 64 个 fp32，同 kRegDq）。
  float dqacc5[kWg5 ? 4 : 1][4][4];
  if constexpr (kRegDq) {
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;
  }
  if constexpr (kWg5) {
#pragma unroll
    for (int nn = 0; nn < 4; ++nn)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) dqacc5[nn][j][q] = 0.f;
  }

  for (int nt = nt_begin; nt < nt_end; ++nt) {
    const int j0 = nt * BN;
    // O41/O51：K/V TMA 或 KVPIPE 时本 tile 的 K 在 Ks[stg]（stg = 相对 nt_begin 的奇偶），
    //   dS3 复用它（KVPIPE 下 dS3 已是独立缓冲，这里不改写它）。
    const int stg = (nt - nt_begin) & 1;
    if constexpr (KVTMA) dS3 = Ks + stg * KS_SZ;
    if constexpr (KVPIPE) {
      // O51：K 双缓冲 ⇒ 本 tile 的 K 在 Ks[stg]；在本轮 GEMM1 之前发起下一 tile 的 K 到
      //   Ks[stg^1]（该 stage 的上一个 K 已被上一轮 GEMM4/dS3 消费）。V 单缓冲，等 GEMM1/2
      //   读完 Vs 后再发起（见下方 2287 之后）。两者延迟均被本轮计算覆盖。
      if (nt + 1 < nt_end) {
        kv_issue_async<HD, BN, NTH>(k8, (nt + 1) * BN, len, Hkv, hkv, qbase, tid,
                                    Ks + (stg ^ 1) * KS_SZ, ASLD);
        cp_async_commit();
      }
    }
    // ---- O3：预取下一 tile 的 K/V 到寄存器（延迟被本轮 5 个 GEMM 覆盖）----
    const int nnt = nt + 1;
    if (!KVTMA && kPrefetch && nnt < nt_end)
      kv_prefetch_pair<NPU, HD, BN, NTH>(k8, v8, nnt * BN, len, Hkv, hkv, qbase, tid, pk0, pk1, pv0,
                                pv1);

    // ---- (1) S = scale·QKᵀ  →  P = exp(S − LSE)，存 fp32 ----
    // ---- (2) dP = dO·Vᵀ    →  dS = P∘(dP − D)，存 fp32 ----
    // O51：KVPIPE 时本 tile 的 K 在 stage stg（K 双缓冲）。
    const unsigned char* Km = KVPIPE ? (Ks + stg * KS_SZ) : Ks;
#ifdef FA_WGMMA
    if constexpr (WGMMA) {
      // O9c-2：GEMM1/2 换 wgmma.m64n32k32（A/B 都从 SW128 直读 smem）。两条异步 mma 一起
      //   发、统一 wait0 重叠；累加器 wgmma 映射下 warp w 持行 [16w,16w+16)，同一线程在
      //   两条 GEMM 里拥有同一个 (r,c) ⇒ dS 直接用寄存器里的 P（pval）即可，无需读回 smem。
      // O41：KVTMA 时 K 从当前 stage 缓冲读。
      const unsigned char* Kcur = KVTMA ? (Ks + stg * KS_SZ) : Ks;
      float sacc[16], dpacc[16];
      // O85：HD>128 的 4D-TMA 把 Q/dO 存成 chunk-major ⇒ A 描述符走 `wgmma_mn32_issue_cm`
      //   （B=K/V 仍是 cp.async 的 rg-major）。HD=128 走原 `wgmma_mn32_issue`（逐位相同）。
      if constexpr (kQChunk) {
        wgmma_mn32_issue_cm<0, BM>(reinterpret_cast<const char*>(Qs),
                                   reinterpret_cast<const char*>(Kcur), HD, sacc);
        wgmma_mn32_issue_cm<1, BM>(reinterpret_cast<const char*>(dOs),
                                   reinterpret_cast<const char*>(Vs), HD, dpacc);
      } else {
        wgmma_mn32_issue<0>(reinterpret_cast<const char*>(Qs),
                            reinterpret_cast<const char*>(Kcur), HD, sacc);
        wgmma_mn32_issue<1>(reinterpret_cast<const char*>(dOs),
                            reinterpret_cast<const char*>(Vs), HD, dpacc);
      }
#if FA_WS1
      // O50：只等 GEMM1（S），让 GEMM2（dP）与下面算 P（P 段只读 sacc）的 CUDA-core
      //   工作重叠；P 算完再 `wait0` 取 dP。`-DFA_WS1=0` 退回原单次 wait0。
      wgmma_wait_group_fp8<1>();
#else
      wgmma_wait0_fp8();
#endif
      float pval[16];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < len && jg < len && !(causal && jg > qi)) {
            float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            float lv = 0.f;
            if constexpr (PREL) lv = lse_r[q >= 2 ? 1 : 0];
            else lv = lse[((size_t)(qbase + qi)) * H + h];
            p = fexp(sval - lv);
          }
          pval[j * 4 + q] = p;
          Ps[r * PSS + c] = p;
        }
#if FA_WS1
      wgmma_wait0_fp8();   // O50：P 段已掩盖 GEMM2 的部分执行，这里等 dP 就绪
#endif
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
          else if (qi < len) del = delta[((size_t)(qbase + qi)) * H + h];
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
        mma_block<GM1, GN1, HD, E4E4>(Qs, ASLD, Km, ASLD, acc, wr, wc, lane);
        mma_block<GM1, GN1, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc2, wr, wc, lane);
        const int r0 = wr * GM1, c0 = wc * GN1;
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
              if (qi < len && jg < len && !(causal && jg > qi)) {
                float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
                float lv = 0.f;
                if constexpr (PREL) lv = lse_r[i * 2 + (q >= 2 ? 1 : 0)];
                else lv = lse[((size_t)(qbase + qi)) * H + h];
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
              else if (qi < len) del = delta[((size_t)(qbase + qi)) * H + h];
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
      mma_block<GM1, GN1, HD, E4E4>(Qs, ASLD, Km, ASLD, acc, wr, wc, lane);
      const int r0 = wr * GM1, c0 = wc * GN1;
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
            if (qi < len && jg < len && !(causal && jg > qi)) {
              float sval = acc[i][j][q] * scale * qs_s[r] * ks_s[c];
              float lv = 0.f;
              if constexpr (PREL) lv = lse_r[i * 2 + (q >= 2 ? 1 : 0)];
              else lv = lse[((size_t)(qbase + qi)) * H + h];
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
        mma_block<GM1, GN1, HD, E5E4>(dOs, ASLD, Vs, ASLD, acc, wr, wc, lane);
        const int r0 = wr * GM1, c0 = wc * GN1;
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
              else if (qi < len) del = delta[((size_t)(qbase + qi)) * H + h];
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
    // O51：GEMM2 已读完 Vs（且 Ap 已拆成独立缓冲）⇒ 立即用 cp.async 发起下一 tile 的 V，
    //   延迟被随后的 fold + GEMM3/4/5 覆盖。K 已在循环开头发起。tile 末尾 `wait_group 0`
    //   后从 Ks[stg^1] 重建 Kp。
    if constexpr (KVPIPE) {
      if (nt + 1 < nt_end) {
        kv_issue_async<HD, BN, NTH>(v8, (nt + 1) * BN, len, Hkv, hkv, qbase, tid, Vs, ASLD);
        cp_async_commit();
      }
    }

    // ---- 构造 dV/dK/dQ 的 fp8 操作数（fold 归约维上的 rowwise scale）----
    // O4a：原来 fold 只让 `tid<BN`（32 线程）算 Ap/dS3、`tid<BM`（64 线程）算 dS2，其余
    // 线程空等；现改为**全部 128 线程均衡分工**：warp 内 4 lane 一组做一行 amax 的
    // `__shfl_xor_sync` 归约。fmaxf 可交换结合 ⇒ amax 结果与原顺序**逐位相同**，除法/量化
    // 也逐元素一致，故数值仍与 O3 逐位相同；但这一段的墙钟缩短约 4×（Amax/写入都并行）。
    if constexpr (!DQONLY) {
    if (wid < 4) {  // O47：BN≤64 ⇒ 4 个 warp 即可覆盖 j（8 行/warp×NTFOLD）；NTH=256 时余下 warp 空等
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
        // O27：每行一次 rcp，元素处用乘法（`RCP=true`）。RCP=false 时 inv 为占位。
        const float invA = RCP ? __frcp_rn(scA) : 0.f;
        const float inv3 = RCP ? __frcp_rn(sc3) : 0.f;
        // O7e：把逐 1B 的 `st.shared.u8` 折成向量写。`F16B=true` 用新映射（每 lane 两段
        //   各 8 个连续 m）⇒ 每段一次 8B `st.shared.v2.u32`；`false` 退回原映射的 4×4B。
        //   QTS=BM+16=80（16 与 8 的倍数）、数组基址 16B 对齐 ⇒ 合法。数值逐位不变。
        // O28：用 `cvt...x2` + `PACK_AB_MERGE_C` 把「4 元素 → 4 字节」从标量路径的
        //   ~4 cvt + shift/OR 折成 2 cvt（逐位相同）；见 `foldpack4`。
        if constexpr (F16B) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
            uint32_t pa2[2], d32[2];
#pragma unroll
            for (int t4 = 0; t4 < 2; ++t4) {
              const int m = sub4 * 8 + t4 * 4 + half * 32;
              pa2[t4] = foldpack4<RCP, false>(
                  Ps[(m + 0) * PSS + j] * dos_s[m + 0], Ps[(m + 1) * PSS + j] * dos_s[m + 1],
                  Ps[(m + 2) * PSS + j] * dos_s[m + 2], Ps[(m + 3) * PSS + j] * dos_s[m + 3],
                  scA, invA);
              d32[t4] = foldpack4<RCP, true>(
                  Ss[(m + 0) * PSS + j] * qs_s[m + 0], Ss[(m + 1) * PSS + j] * qs_s[m + 1],
                  Ss[(m + 2) * PSS + j] * qs_s[m + 2], Ss[(m + 3) * PSS + j] * qs_s[m + 3],
                  sc3, inv3);
            }
            *reinterpret_cast<uint2*>(Ap + j * QTS + sub4 * 8 + half * 32) =
                make_uint2(pa2[0], pa2[1]);
            *reinterpret_cast<uint2*>(dS3 + j * QTS + sub4 * 8 + half * 32) =
                make_uint2(d32[0], d32[1]);
          }
        } else {
          uint32_t pa4[4], d34[4];
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            const int m = sub4 * 16 + t4 * 4;
            pa4[t4] = foldpack4<RCP, false>(
                Ps[(m + 0) * PSS + j] * dos_s[m + 0], Ps[(m + 1) * PSS + j] * dos_s[m + 1],
                Ps[(m + 2) * PSS + j] * dos_s[m + 2], Ps[(m + 3) * PSS + j] * dos_s[m + 3],
                scA, invA);
            d34[t4] = foldpack4<RCP, true>(
                Ss[(m + 0) * PSS + j] * qs_s[m + 0], Ss[(m + 1) * PSS + j] * qs_s[m + 1],
                Ss[(m + 2) * PSS + j] * qs_s[m + 2], Ss[(m + 3) * PSS + j] * qs_s[m + 3],
                sc3, inv3);
          }
#pragma unroll
          for (int t4 = 0; t4 < 4; ++t4) {
            *reinterpret_cast<uint32_t*>(Ap + j * QTS + sub4 * 16 + t4 * 4) = pa4[t4];
            *reinterpret_cast<uint32_t*>(dS3 + j * QTS + sub4 * 16 + t4 * 4) = d34[t4];
          }
        }
      }
    }
    }  // if constexpr (!DQONLY)：DQONLY 跳过 Ap/dS3 fold（dV/dK 的操作数）
    if (wid < 4) {  // O47：BM=64 ⇒ 4 个 warp（16 行/warp）即可覆盖 m；NTH=256 时余下 warp 空等
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
      const float inv2 = RCP ? __frcp_rn(sc2) : 0.f;
      // O7e：`j = sub2*16+t` 连续 ⇒ dS2[m][j] 的 16 个 j 也连续，同样折成 4 次 4B 写。
      // O7e-2：16 个连续 j 一次 16B `st.shared.v4.u32`（DSS2 是 16 的倍数、基址对齐）。
#pragma unroll
      for (int jh = 0; jh < NTFOLD; ++jh) {
        uint32_t d2_4[4];
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4) {
          const int j = sub2 * 16 + t4 * 4 + jh * 32;
          d2_4[t4] = foldpack4<RCP, true>(
              Ss[m * PSS + j + 0] * ks_s[j + 0], Ss[m * PSS + j + 1] * ks_s[j + 1],
              Ss[m * PSS + j + 2] * ks_s[j + 2], Ss[m * PSS + j + 3] * ks_s[j + 3],
              sc2, inv2);
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
      // ---- (4) dK = scale·dSᵀ·Q : A=dS3[j][m] (e5m2), B=Qp[m/2][d0+..] (e4m3, ldmatrix.trans) ----
      // O29：把两段 epilogue 抽成 lambda，`FA_ILV34` 时先连发两条 mma 再做 epilogue（见宏说明）。
      auto zero34 = [&](float (&acc)[MTM34][NTM34][4]) {
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < NTM34; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
      };
      auto epi_dv = [&](float (&acc)[MTM34][NTM34][4]) {
        const int r0 = wr * GM34, c0 = wc * GN34;
        float* wstg = pstg + wid * (STGR * STGS);
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < NTM34; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2;
              int jg = j0 + r;
              if constexpr (FA_R4 && !kBulkRed && !DET) {
                // O67：非 DET/BULKRED 的 dV 归约走 16B `red_add4`（见 `red_add4` 上方说明）。
                if ((q & 1) == 0) {
                  float a2 = __shfl_down_sync(0xffffffffu, acc[i][j][q], 1);
                  float b2 = __shfl_down_sync(0xffffffffu, acc[i][j][q + 1], 1);
                  if (jg < len && (lane & 1) == 0)
                    red_add4(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c,
                             acc[i][j][q] * sA[r], acc[i][j][q + 1] * sA[r], a2 * sA[r],
                             b2 * sA[r]);
                }
              } else if (jg < len) {
                if constexpr (kBulkRed) {
                  // O42：写 per-warp staging（行=warp 内 KV 行，列=warp 内 64 列），
                  //   随后由 `bulk_flush` 一次性 coalesced 归约回 global。
                  wstg[(i * 16 + g + (q >= 2 ? 8 : 0)) * STGS + (j * 8 + c2 + (q & 1))] =
                      acc[i][j][q] * sA[r];
                } else if constexpr (DET && DET_HALF) {
                  // F4：fp16 partial + O62 扇区化——把 group `j`、`j+1` 的两个 half2 拼成一次
                  //   8B 写（4 lane 覆盖连续 32B），列地址走 16 列块内置换（与 reduce 读回一致）。
                  if ((q & 1) == 0 && (j & 1) == 0 && (j + 1) < NTM34) {
                    const size_t rb =
                        part_base
                            ? ((size_t)part_base[b] + ((size_t)h * nblk_seq + mblk) * len + jg) * HD
                            : (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD;
                    const size_t dpo = rb + d0 + c0 + (j >> 1) * 16 + (lane & 3) * 4;
                    dkv_det_store_h4(dv_part, dpo, acc[i][j][q] * sA[r], acc[i][j][q + 1] * sA[r],
                                     acc[i][j + 1][q] * sA[r], acc[i][j + 1][q + 1] * sA[r]);
                  }
                } else if constexpr (DET) {
                  // P3-4e：非原子写 partial（每元素本 CTA 唯一）→ 固定次序二次归约。
                  // P3-4o：`part_base` 非空时改用 compact per-sequence 布局（缩小地址跨度）。
                  if ((q & 1) == 0) {
                    const size_t dpo =
                        part_base
                            ? ((size_t)part_base[b] + ((size_t)h * nblk_seq + mblk) * len + jg) * HD +
                                  d0 + c
                            : (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + d0 + c;
                    dkv_det_store(dv_part + dpo, acc[i][j][q] * sA[r], acc[i][j][q + 1] * sA[r]);
                  }
                } else if ((q & 1) == 0) {
                  // O4c：q/q+1 两列相邻且同 row → 一次 float2 red。
                  // O114：`FA_REDHALF` 时改走 fp16 `red.global.add.f16x2`（负结果探针——
                  //   真机扇区未减，见宏说明 / `docs/03` §136）。
                  const size_t idx = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c;
                  if constexpr (FA_REDHALF)
                    red_addh2(reinterpret_cast<__half*>(dv_acc) + idx, acc[i][j][q] * sA[r],
                              acc[i][j][q + 1] * sA[r]);
                  else
                    red_add2(dv_acc + idx, acc[i][j][q] * sA[r], acc[i][j][q + 1] * sA[r]);
                }
              }
            }
      };
      auto epi_dk = [&](float (&acc)[MTM34][NTM34][4]) {
        const int r0 = wr * GM34, c0 = wc * GN34;
        float* wstg = pstg + wid * (STGR * STGS);
#pragma unroll
        for (int i = 0; i < MTM34; ++i)
#pragma unroll
          for (int j = 0; j < NTM34; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              int r = r0 + i * 16 + g + (q >= 2 ? 8 : 0);
              int c = c0 + j * 8 + c2;
              int jg = j0 + r;
              if constexpr (FA_R4 && !kBulkRed && !DET) {
                // O67：非 DET/BULKRED 的 dK 归约走 16B `red_add4`（与 epi_dv 同款）。
                if ((q & 1) == 0) {
                  float a = acc[i][j][q] * sds3[r] * scale;
                  float b = acc[i][j][q + 1] * sds3[r] * scale;
                  float a2 = __shfl_down_sync(0xffffffffu, a, 1);
                  float b2 = __shfl_down_sync(0xffffffffu, b, 1);
                  if (jg < len && (lane & 1) == 0)
                    red_add4(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c,
                             a, b, a2, b2);
                }
              } else if (jg < len) {
                if constexpr (kBulkRed) {
                  wstg[(i * 16 + g + (q >= 2 ? 8 : 0)) * STGS + (j * 8 + c2 + (q & 1))] =
                      acc[i][j][q] * sds3[r] * scale;
                } else if constexpr (DET && DET_HALF) {
                  // F4：fp16 partial + O62 扇区化（与 epi_dv 同布局；dK 额外乘 fold scale）。
                  if ((q & 1) == 0 && (j & 1) == 0 && (j + 1) < NTM34) {
                    const size_t rb =
                        part_base
                            ? ((size_t)part_base[b] + ((size_t)h * nblk_seq + mblk) * len + jg) * HD
                            : (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD;
                    const size_t dpo = rb + d0 + c0 + (j >> 1) * 16 + (lane & 3) * 4;
                    dkv_det_store_h4(dk_part, dpo, acc[i][j][q] * sds3[r] * scale,
                                     acc[i][j][q + 1] * sds3[r] * scale,
                                     acc[i][j + 1][q] * sds3[r] * scale,
                                     acc[i][j + 1][q + 1] * sds3[r] * scale);
                  }
                } else if constexpr (DET) {
                  if ((q & 1) == 0) {
                    const size_t dpo =
                        part_base
                            ? ((size_t)part_base[b] + ((size_t)h * nblk_seq + mblk) * len + jg) * HD +
                                  d0 + c
                            : (((size_t)(b * H + h) * nblk + mblk) * (size_t)S + jg) * HD + d0 + c;
                    dkv_det_store(dk_part + dpo, acc[i][j][q] * sds3[r] * scale,
                                  acc[i][j][q + 1] * sds3[r] * scale);
                  }
                } else if ((q & 1) == 0) {
                  // O114：dK 与 dV 同款——`FA_REDHALF` 走 fp16 `red.global.add.f16x2`。
                  const size_t idx = (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c;
                  if constexpr (FA_REDHALF)
                    red_addh2(reinterpret_cast<__half*>(dk_acc) + idx, acc[i][j][q] * sds3[r] * scale,
                              acc[i][j][q + 1] * sds3[r] * scale);
                  else
                    red_add2(dk_acc + idx, acc[i][j][q] * sds3[r] * scale,
                             acc[i][j][q + 1] * sds3[r] * scale);
                }
              }
            }
      };
      // O42：per-warp staging → `cp.reduce.async.bulk`。每 warp 的 staging 区私有 ⇒
      //   只用 `__syncwarp`（廉价），不用 `__syncthreads`；`bulk_issue` 发出后**不等**，
      //   让 TMA 归约与 GEMM4/GEMM5 重叠；只有要覆盖 staging 时才 `bulk_waitread`。
      auto bulk_issue = [&](float* acc_dst) {
        __syncwarp();
        if (lane < STGR) {
          const int jg = j0 + wr * STGR + lane;
          if (jg < len) {
            bulk_reduce_fence();
            float* src = pstg + wid * (STGR * STGS) + lane * STGS;
            float* gdst = acc_dst + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + wc * 64 + d0;
            bulk_reduce_add_f32(gdst, src, STGC * (int)sizeof(float));
            bulk_reduce_commit();
          }
        }
        __syncwarp();
      };
      auto bulk_waitread = [&]() {
        __syncwarp();
        if (lane < STGR) bulk_reduce_wait0();
        __syncwarp();
      };
      if constexpr (!DQONLY) {
      if constexpr (kWg34) {
#ifdef FA_WGMMA
        // O82（F3b 主体续）：GEMM3(dV)/GEMM4(dK) 走 wgmma RS。M=BN=32 零填充到 m64——warp 2/3
        //   的 A 寄存器置 0、其输出行 32–63 丢弃；只多花一半张量核周期（tensor 利用率 <20%）。
        //   A=Ap/dS3（[BN][QTS]，K=BM=64 分两个 k-step 用 `ldmatrix.x4` 装入）；
        //   B=dOᵀ/Qᵀ（复用 Qp/dOp 的紧凑 no-swizzle [HD][BM] 缓冲，LBO=HD、SBO=8）。
        const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
        const int acol = (lane >> 4) * 16;
        const uint32_t baQ = smem_u32(reinterpret_cast<const unsigned char*>(Qp));
        const uint32_t baO = smem_u32(reinterpret_cast<const unsigned char*>(dOp));
        // 分两组 n-tile（每组 2 个，acc[2][16]=32 fp32）以压低寄存器/溢出；两组各一次 wait。
        // ---- GEMM3 dV = Pᵀ·dO : A=Ap (e4m3), B=dOᵀ (e5m2) ----
        {
          uint32_t av[2][4];
          if (wid < 2) {
            ldmatrix_x4(smem_u32(Ap + (wid * 16 + arow) * QTS + 0 + acol), av[0]);
            ldmatrix_x4(smem_u32(Ap + (wid * 16 + arow) * QTS + 32 + acol), av[1]);
          } else {
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
              for (int i = 0; i < 4; ++i) av[s][i] = 0u;
          }
#pragma unroll
          for (int ng = 0; ng < 2; ++ng) {
            float acc[2][16];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
              for (int i = 0; i < 16; ++i) acc[nn][i] = 0.f;
            wgmma_fence_fp8();
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
              for (int nn = 0; nn < 2; ++nn) {
                uint64_t db = make_desc_noswz_fp8(
                    baO + (uint32_t)((ng * 2 + nn) * 512 + s * 4096), (uint32_t)HD, 8);
                wgmma_m64n32k32_rs_e4e5(acc[nn], av[s], db);
              }
            wgmma_commit_fp8();
            wgmma_wait0_fp8();
            if (wid < 2) {
#pragma unroll
              for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                  for (int q = 0; q < 4; q += 2) {
                    int r = wid * 16 + g + (q >= 2 ? 8 : 0);
                    int c = (ng * 2 + nn) * 32 + j * 8 + c2;
                    int jg = j0 + r;
                    if (jg < len)
                      red_add2(dv_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c,
                               acc[nn][j * 4 + q] * sA[r], acc[nn][j * 4 + q + 1] * sA[r]);
                  }
            }
          }
        }
        // ---- GEMM4 dK = scale·dSᵀ·Q : A=dS3 (e5m2), B=Qᵀ (e4m3) ----
        {
          uint32_t av[2][4];
          if (wid < 2) {
            ldmatrix_x4(smem_u32(dS3 + (wid * 16 + arow) * QTS + 0 + acol), av[0]);
            ldmatrix_x4(smem_u32(dS3 + (wid * 16 + arow) * QTS + 32 + acol), av[1]);
          } else {
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
              for (int i = 0; i < 4; ++i) av[s][i] = 0u;
          }
#pragma unroll
          for (int ng = 0; ng < 2; ++ng) {
            float acc[2][16];
#pragma unroll
            for (int nn = 0; nn < 2; ++nn)
#pragma unroll
              for (int i = 0; i < 16; ++i) acc[nn][i] = 0.f;
            wgmma_fence_fp8();
#pragma unroll
            for (int s = 0; s < 2; ++s)
#pragma unroll
              for (int nn = 0; nn < 2; ++nn) {
                uint64_t db = make_desc_noswz_fp8(
                    baQ + (uint32_t)((ng * 2 + nn) * 512 + s * 4096), (uint32_t)HD, 8);
                wgmma_m64n32k32_rs_e5e4(acc[nn], av[s], db);
              }
            wgmma_commit_fp8();
            wgmma_wait0_fp8();
            if (wid < 2) {
#pragma unroll
              for (int nn = 0; nn < 2; ++nn)
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                  for (int q = 0; q < 4; q += 2) {
                    int r = wid * 16 + g + (q >= 2 ? 8 : 0);
                    int c = (ng * 2 + nn) * 32 + j * 8 + c2;
                    int jg = j0 + r;
                    if (jg < len)
                      red_add2(dk_acc + (((size_t)(qbase + jg)) * Hkv + hkv) * HD + d0 + c,
                               acc[nn][j * 4 + q] * sds3[r] * scale,
                               acc[nn][j * 4 + q + 1] * sds3[r] * scale);
                  }
            }
          }
        }
#endif
      } else {
#if FA_ILV34
      {
        float a3[MTM34][NTM34][4], a4[MTM34][NTM34][4];
        zero34(a3);
        zero34(a4);
        mma_block_bt<GM34, GN34, BM, E4E5>(Ap, QTS, dOp, PSLD, a3, wr, wc, lane, d0);
        mma_block_bt<GM34, GN34, BM, E5E4>(dS3, QTS, Qp, PSLD, a4, wr, wc, lane, d0);
        epi_dv(a3);
        epi_dk(a4);
      }
#else
      {
        float acc[MTM34][NTM34][4];
        zero34(acc);
        mma_block_bt<GM34, GN34, BM, E4E5>(Ap, QTS, dOp, PSLD, acc, wr, wc, lane, d0);
        epi_dv(acc);
      }
      if constexpr (kBulkRed) bulk_issue(dv_acc);
      {
        float acc[MTM34][NTM34][4];
        zero34(acc);
        mma_block_bt<GM34, GN34, BM, E5E4>(dS3, QTS, Qp, PSLD, acc, wr, wc, lane, d0);
        // O42：GEMM4 mma 与 dV 的 TMA 归约重叠；覆盖 staging 前再等它读完。
        if constexpr (kBulkRed) bulk_waitread();
        epi_dk(acc);
      }
      if constexpr (kBulkRed) bulk_issue(dk_acc);
#endif
      }  // else：kWg34=false 的原 mma GEMM3/4
      }  // if constexpr (!DQONLY)：跳过 GEMM3/4（dV/dK）及其 epilogue

      // ---- (5) dQ += scale·dS·K : A=dS2[m][j] (e5m2), B=Kp[j/2][d0+..] (e4m3, ldmatrix.trans) ----
      if constexpr (kWg5) {
#ifdef FA_WGMMA
        // O81（F3b）：GEMM5 走 wgmma RS —— A=dS2 经 `ldmatrix.x4` 装入寄存器，
        //   B=Kᵀ（INTERLEAVE K-major，no-swizzle 描述符）从 Kp 缓冲（已由
        //   `transpose_sw128_to_inter` 从 SW128 的 K stage 重建）。m64n32k32 ×4（HD/32）。
        const int arow = (lane & 7) + ((lane >> 3) & 1) * 8;
        const int acol = (lane >> 4) * 16;
        uint32_t av[4];
        ldmatrix_x4(smem_u32(dS2 + (wid * 16 + arow) * DSS2 + d0 + acol), av);
        const uint32_t ba = smem_u32(reinterpret_cast<const unsigned char*>(Kp));
        float acc5[4][16];
#pragma unroll
        for (int nn = 0; nn < 4; ++nn)
#pragma unroll
          for (int i = 0; i < 16; ++i) acc5[nn][i] = 0.f;
        wgmma_fence_fp8();
#pragma unroll
        for (int nn = 0; nn < 4; ++nn) {
          uint64_t db = make_desc_noswz_fp8(ba + (uint32_t)(nn * 4 * 16 * 16), 8, 16);
          wgmma_m64n32k32_rs_e5e4(acc5[nn], av, db);
        }
        wgmma_commit_fp8();
        wgmma_wait0_fp8();
#pragma unroll
        for (int nn = 0; nn < 4; ++nn)
#pragma unroll
          for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const int r = wid * 16 + g + (q >= 2 ? 8 : 0);
              dqacc5[nn][j][q] += acc5[nn][j * 4 + q] * sds2[r] * scale;
            }
#endif
      } else {
      {
        float acc[MTM5][NTM5][4];
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        mma_block_bt<GM5, GN5, BN, E5E4>(dS2, DSS2, Kp, PSLD, acc, wr, wc, lane, d0);
        const int r0 = wr * GM5, c0 = wc * GN5;
#pragma unroll
        for (int i = 0; i < MTM5; ++i)
#pragma unroll
          for (int j = 0; j < NTM5; ++j)
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
                if (qi < len) {
                  // P3-4k：非 `kRegDq` 路径（MLA/HD=512 因寄存器墙恒走此分支）。`DET && ksplit>1`
                  //   时把 dQ 的每个 part 偏和累加进本 CTA **独占**的 partial 区（布局同 P3-4f，
                  //   按 part 分片），由 `dq_reduce_kernel` 按 part 固定次序求和 ⇒ 跨 part 也确定。
                  //   这里逐 tile 累加：同一 `(row,c)` 在 CTA 内由同一线程按 nt 程序序写，故
                  //   `red_add2`（原子）虽为跨 CTA 设计，对本 CTA 私有区仍是确定次序；空 part 的
                  //   partial 由 host 先清零。`ksplit==1` 保持原无竞争 `red_add2`（逐位不变）。
                  if constexpr (DET) {
                    if (ksplit > 1)
                      red_add2(dq_part +
                                   (((size_t)(qbase + qi) * H + h) * ksplit + part) * HD + d0 + c,
                               acc[i][j][q] * sds2[r] * scale,
                               acc[i][j][q + 1] * sds2[r] * scale);
                    else
                      red_add2(dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + d0 + c,
                               acc[i][j][q] * sds2[r] * scale,
                               acc[i][j][q + 1] * sds2[r] * scale);
                  } else {
                    red_add2(dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + d0 + c,
                             acc[i][j][q] * sds2[r] * scale,
                             acc[i][j][q + 1] * sds2[r] * scale);
                  }
                }
              }
            }
      }
      }  // else：kWg5=false 的原 mma GEMM5
      // O42：离开本 tile 前等 dK 的 TMA 归约读完 staging（下一 tile 的 GEMM1/2 epilogue
      //   会覆写 Ps/Ss）。与 GEMM5 重叠。
      if constexpr (kBulkRed && !DQONLY) bulk_waitread();
    }
    __syncthreads();
    // ---- O3：落盘预取的下一 tile 的 K/V（本轮 GEMM 已全部读完 smem），并更新 ks/vs ----
    // ---- O41：KVTMA 时改为「发 K[nt+2] → Ks[stg]（本迭代 dS3 已读完）、发 V[nt+1] → Vs
    //       （本迭代 Ap 已读完）」，再等 K[nt+1]（上一迭代已发）并重建 Kp；V 的 TMA 与
    //       Kp 重建重叠。----
    if constexpr (KVTMA) {
      // O119：MCAST>1 时先让 cluster 锁步——保证 leader 发的 tile 与每个 follower 当前 nt 一致，
      //   且 leader 覆写 stage 前所有 CTA 都已消费完该 stage 的旧 tile（否则串 tile/覆写协程）。
      //   实测去掉锁步会串 tile/死锁 ⇒ 锁步是 multicast 正确性的硬前提。
      if (MCAST > 1) fp8_cluster_sync();
      if (tid == 0) {
        const bool ld = (MCAST == 1) || (fp8_cluster_rank() == 0);
        if (nt + 2 < nt_end) {
          mbar_arrive_expect(qbars + 2 + stg, KS_SZ);
          if (ld) {
            if (MCAST > 1)
              tma_load_4d_mc(Ks + stg * KS_SZ, kmap, 0, (nt + 2) * BN, hkv, b,
                             qbars + 2 + stg, kMCMask);
            else
              tma_load_4d(Ks + stg * KS_SZ, kmap, 0, (nt + 2) * BN, hkv, b, qbars + 2 + stg);
          }
        }
        if (nt + 1 < nt_end) {
          mbar_arrive_expect(qbars + 4, KS_SZ);
          if (ld) {
            if (MCAST > 1)
              tma_load_4d_mc(Vs, vmap, 0, (nt + 1) * BN, hkv, b, qbars + 4, kMCMask);
            else
              tma_load_4d(Vs, vmap, 0, (nt + 1) * BN, hkv, b, qbars + 4);
          }
        }
      }
      if (nt + 1 < nt_end) {
        if (tid < BN) {
          int jg = (nt + 1) * BN + tid;
          ks_s[tid] = (jg < len) ? ks[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
          vs_s[tid] = (jg < len) ? vs[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
        }
        const int sk2 = stg ^ 1;
        const int kc = sk2 ? kuse1 : kuse0;
        mbar_wait(qbars + 2 + sk2, (uint32_t)(kc & 1));
        if (sk2) kuse1++; else kuse0++;
        __syncthreads();
        const unsigned char* Kb = Ks + (stg ^ 1) * KS_SZ;
        if constexpr (kWg5) {
          // O81：GEMM5 走 wgmma，B=Kᵀ 需 INTERLEAVE K-major ⇒ 从 SW128 的 K stage 逐字节
          //   转置写进 Kp 缓冲（kp_bytes=4352 ≥ HD*BN=4096）。generic 写 smem 后必须
          //   `fence.proxy.async`（`bulk_reduce_fence`）才能被 wgmma 的 async proxy 读到。
          transpose_sw128_to_inter<BN, HD>(reinterpret_cast<unsigned char*>(Kp), Kb, wid, lane);
          bulk_reduce_fence();
        } else {
          const int nd4k = HD / 4;
          for (int u = tid; u < (BN / 2) * nd4k; u += NTH) {
            int rp = u / nd4k, dq = (u % nd4k) * 4;
            uint32_t k0 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2, dq, HD));
            uint32_t k1 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2 + 1, dq, HD));
            uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * PSLD + dq);
            kpw[0] = __byte_perm(k0, k1, 0x5140);
            kpw[1] = __byte_perm(k0, k1, 0x7362);
          }
        }
        __syncthreads();
        mbar_wait(qbars + 4, (uint32_t)(vuse & 1));
        vuse++;
      }
    } else if constexpr (KVPIPE) {
      // O51：本 tile 的 K/V 已在 GEMM1/2 后由 cp.async 发起；这里等它落地并从行主序 Ks
      //   重建下一 tile 的 Kp（供 GEMM5）。数值与原同步 `kv_load_pair` 逐位相同。
      if (nt + 1 < nt_end) {
        if (tid < BN) {
          int jg = (nt + 1) * BN + tid;
          ks_s[tid] = (jg < len) ? ks[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
          vs_s[tid] = (jg < len) ? vs[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
        }
        cp_async_wait0();
        __syncthreads();
        kp_build_rows<HD, BN, NTH>(Ks + (stg ^ 1) * KS_SZ, Kp, ASLD, PSLD, tid);
        __syncthreads();
      }
    } else if (nt + 1 < nt_end) {
      if (kPrefetch) {
        kv_commit_pair<NPU, HD, BN, WGMMA, NTH>(Ks, Vs, Kp, pk0, pk1, pv0, pv1, tid, ASLD, PSLD);
      } else {
        kv_load_pair<HD, BN, WGMMA, NTH>(k8, v8, (nt + 1) * BN, len, Hkv, hkv, qbase, tid, Ks, Vs, Kp, ASLD,
                             PSLD);
      }
      if (tid < BN) {
        int jg = (nt + 1) * BN + tid;
        ks_s[tid] = (jg < len) ? ks[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
        vs_s[tid] = (jg < len) ? vs[((size_t)(qbase + jg)) * Hkv + hkv] : 1.f;
      }
      __syncthreads();
    }
  }

  // ---- O7：把寄存器里累加的 dQ flush 出去（每 CTA 每元素一次 `red_add2`）。----
  if constexpr (kWg5) {
    // O81：wgmma GEMM5 的累加器映射 —— warp wid 持行 [16*wid,16*wid+16)，列 = nn*32 +
    //   j*8 + c2 + (q&1)。每线程 4×j × 4×q × 4×nn = 64 个 fp32。
#pragma unroll
    for (int nn = 0; nn < 4; ++nn)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = wid * 16 + g + (q >= 2 ? 8 : 0);
          int c = nn * 32 + j * 8 + c2;
          int qi = m0 + r;
          if (qi < len)
            red_add2(dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c, dqacc5[nn][j][q],
                     dqacc5[nn][j][q + 1]);
        }
  } else if constexpr (kRegDq) {
    // O48：flush 的 warp 几何由 GM5/GN5/MTM5/NTM5 派生（原写死 `wr*32/wc*64`、i<2/j<8，
    //   只对默认 2×2/128 线程成立；D=128 的 8-warp 实例 NTM5=4 会越界读 dqacc 并写错列）。
    //   默认 128/2 时 GM5=32/GN5=64/MTM5=2/NTM5=8 ⇒ 与原式逐字等价。
#pragma unroll
    for (int i = 0; i < MTM5; ++i)
#pragma unroll
      for (int j = 0; j < NTM5; ++j)
#pragma unroll
        for (int q = 0; q < 4; q += 2) {
          int r = wr * GM5 + i * 16 + g + (q >= 2 ? 8 : 0);
          int c = wc * GN5 + j * 8 + c2;
          int qi = m0 + r;
          // P3-4f：`DET && ksplit>1` 时 dQ 也走 partial（每 (row,h) 按 part 分片），
          //   由 `dq_reduce_kernel` 固定次序求和 ⇒ 跨 part 也确定。ksplit==1 时保持原
          //   单 CTA 无竞争 `red_add2`（逐位不变）。空 part 的 partial 由 host 先清零。
          if (qi < len) {
            if constexpr (DET) {
              if (ksplit > 1)
                dkv_det_store(dq_part +
                                  (((size_t)(qbase + qi) * H + h) * ksplit + part) * HD + c,
                              dqacc[i][j][q], dqacc[i][j][q + 1]);
              else
                red_add2(dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c, dqacc[i][j][q],
                         dqacc[i][j][q + 1]);
            } else if constexpr (DQONLY) {
              // F7 第十一步：DQONLY 时 dQ 由本 CTA 独占（ksplit=1），用 plain store 写出
              //   ⇒ L2 `red` = 0（对比默认 `red_add2` 每元素一次 red）。
              float* dst = dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c;
              dst[0] = dqacc[i][j][q];
              dst[1] = dqacc[i][j][q + 1];
            } else {
              red_add2(dq_acc + (((size_t)(qbase + qi)) * H + h) * HD + c, dqacc[i][j][q],
                       dqacc[i][j][q + 1]);
            }
          }
        }
  }
}

// O37：两个薄 `__global__` 壳复用同一 `fp8_mma_body`。cp.async 版与 TMA 版签名只差两个
//   `__grid_constant__` 描述符（`TMA=true` 才用到）。
// O47：`NTH`/`NWAR` 同 `fp8_mma_body`（默认 128/2 与历史逐字等价；MLA 用 256/4）。
// O104（第 198 轮）：`HSWAP`（默认 false）——把 O93 的跨 head 全局 LPT 调度从 `kvtma` 快路
//   （D=128）推广到**通用 `fa_bwd_fp8_mma_kernel`**（D=256 默认走它）。`fp8_mma_body` 本就支持
//   HSWAP（见 body 上方说明），此处只是把模板参透传；`HSWAP=false` 与历史逐位相同。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true,
           bool RCP = true, int NTH = THREADS, int NWAR = WN, bool KVPIPE = false, bool DET = false,
           bool DET_HALF = false, bool DQONLY = false, bool HSWAP = false>
__global__ void __launch_bounds__(NTH, (NTH == THREADS && HD == 128) ? (BN <= 32 ? 3 : 2) : 1)
fa_bwd_fp8_mma_kernel(const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                      const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                      const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                      const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                      const float* __restrict__ delta, const float* __restrict__ lse,
                      float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                      float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                      int causal, int ksplit, const int* __restrict__ cu_seqlens = nullptr,
                      const int* __restrict__ mt_b = nullptr,
                      const int* __restrict__ mt_m = nullptr,
                      float* __restrict__ dk_part = nullptr,
                      float* __restrict__ dv_part = nullptr, int nblk = 0,
                      float* __restrict__ dq_part = nullptr,
                      const int* __restrict__ part_base = nullptr) {
  fp8_mma_body<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, false, false, NTH, NWAR, KVPIPE, DET,
               DET_HALF, DQONLY, HSWAP>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, nullptr, nullptr, nullptr, nullptr, mt_b, mt_m,
      dk_part, dv_part, nblk, dq_part, part_base);
}

// O37：Q/dO 4D-TMA 版（`-DFA_WGMMA -DFA_TMA` 构建、WGMMA 路径）。HD=128 与 HD=256 均实例化
//   （O85：HD=256 的 Q/dO TMA，smem 115,712B ≤ 116,224B ⇒ 2 CTA/SM）。
// P3-4g：`DET=true` 时复用同一 body 的确定性 dK/dV 路径（partial + 固定次序归约）。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          bool DET = false, bool DET_HALF = false>
__global__ void __launch_bounds__(THREADS, (HD == 128) ? (BN <= 32 ? 3 : 2)
                                                       : ((HD == 256 && BN <= 32) ? 2 : 1))
fa_bwd_fp8_mma_qdtma_kernel(const __grid_constant__ CUtensorMap qmap,
                            const __grid_constant__ CUtensorMap dmap,
                            const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                            const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                            const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                            const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                            const float* __restrict__ delta, const float* __restrict__ lse,
                            float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                            float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                            int causal, int ksplit, const int* __restrict__ cu_seqlens = nullptr,
                            float* __restrict__ dk_part = nullptr,
                            float* __restrict__ dv_part = nullptr, int nblk = 0,
                            float* __restrict__ dq_part = nullptr,
                            const int* __restrict__ part_base = nullptr) {
  fp8_mma_body<HD, BM, BN, REGDQ, true, PREL, F16B, RCP, true, false, THREADS, WN, false, DET,
               DET_HALF>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, &qmap, &dmap, nullptr, nullptr, nullptr, nullptr,
      dk_part, dv_part, nblk, dq_part, part_base);
}

// O41：Q/dO/K/V 全 4D-TMA 版（roadmap「下一步候选 ①」）。K 双缓冲、V 单缓冲；Kp 从 SW128
//   K tile 重建。仅 `-DFA_WGMMA -DFA_TMA` 构建、HD=128/WGMMA（BN=32）路径实例化。
// P3-4g：`DET=true` 时复用同一 body 的确定性 dK/dV 路径（partial + 固定次序归约），
//   把 `--det` 从默认 mma 路径扩到 Hopper TMA 快路。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
           bool DET = false, bool DET_HALF = false, bool DQONLY = false, bool HSWAP = false,
           int MCAST = 1>
__global__ void __launch_bounds__(THREADS, (HD == 128) ? (BN <= 32 ? 3 : 2) : 1)
fa_bwd_fp8_mma_kvtma_kernel(const __grid_constant__ CUtensorMap qmap,
                            const __grid_constant__ CUtensorMap dmap,
                            const __grid_constant__ CUtensorMap kmap,
                            const __grid_constant__ CUtensorMap vmap,
                            const unsigned char* __restrict__ q8, const float* __restrict__ qs,
                            const unsigned char* __restrict__ k8, const float* __restrict__ ks,
                            const unsigned char* __restrict__ v8, const float* __restrict__ vs,
                            const unsigned char* __restrict__ do8, const float* __restrict__ dos,
                            const float* __restrict__ delta, const float* __restrict__ lse,
                            float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                            float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                            int causal, int ksplit, const int* __restrict__ cu_seqlens = nullptr,
                            float* __restrict__ dk_part = nullptr,
                            float* __restrict__ dv_part = nullptr, int nblk = 0,
                            float* __restrict__ dq_part = nullptr,
                            const int* __restrict__ part_base = nullptr,
                            const int* __restrict__ mt_m = nullptr,
                            const int* __restrict__ slot_tab = nullptr) {
  fp8_mma_body<HD, BM, BN, REGDQ, true, PREL, F16B, RCP, true, true, THREADS, WN, false, DET,
               DET_HALF, DQONLY, HSWAP, MCAST>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, &qmap, &dmap, &kmap, &vmap, nullptr, mt_m,
      dk_part, dv_part, nblk, dq_part, part_base, slot_tab);
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
// 3c) fa_bwd_fp8_wgmma2_kernel —— F6 第二步：双 warpgroup（256 线程）+ wgmma 的
//     BM=128 主 kernel（GEMM1/2 走 `wgmma.m64n32k32` 直读 SW128，GEMM3/4/5 仍 mma）。
// =============================================================================
// 动机（ROADMAP『fp8 专项冲刺』F6）：默认 fp8 main 的墙是 dK/dV 跨 CTA 的 L2 `red`
//   （114.5M 扇区、占 L2 流量 ~74%）。唯一真杠杆是把 BM 64→128，让每个 KV 元素被一半的
//   CTA 贡献。已有 `fa_bwd_fp8_wg2_kernel`（BM=128、2 warpgroup、mma + 全同步载入）确实把
//   `red` 砍半（58.2M），但它 217 regs / 131KB smem → 1 CTA/SM、L2 只到 20.4%，反而慢
//   0.53×。F6 的解法是「双 warpgroup + wgmma（+后续 TMA）」用 Hopper 原语把寄存器/指令
//   压下来。第 140 轮冒烟 `fa_bwd_fp8_wgmma2_smoke.cu` 已证明双 WG wgmma 几何成立、
//   SASS 无 HMMA/LDSM、90 regs。本 kernel 是它的**主 kernel 化**：
//     * Q/dO/K/V 存 **SW128 K-major**（wgmma 描述符直读），Qp/dOp/Kp 由 SW128 重建
//       （供 GEMM3/4/5 的 `ldmatrix.x2.trans`）；fold 与 GEMM3/4/5 逐字沿用 wg2。
//     * GEMM1/2：每个 WG 各跑 `wgmma.m64n32k32`（A 描述符按 WG +`(64/8)*1024=8192B`），
//       累加器映射 `row=wg*64+wl*16+g+(q>=2?8:0)`、`col=j*8+c2+(q&1)`，epilogue 直接写
//       Ps/Ss；rowwise scale 仍在 epilogue 折算（口径与 wg2 逐项一致，仅 fp 归约次序略变）。
//     * Ap/dS3 从 wg2 的「别名 Ks/Vs」改为**独立缓冲**（SW128 的 Ks/Vs 只有 4096B/块，
//       放不下 BN×QTS=4608B，且 Vs 在 GEMM2 时仍被读）。
// 这是 F6 的**中间态**（未上 TMA），用于量「GEMM1/2 wgmma 化」单独能省多少寄存器/时间；
// 默认路径一行未改（仅 `--wg2wgmma` opt-in）。
#ifdef FA_WGMMA
// O91（F6-step4，multi-warpgroup）：把本 kernel 泛化为 `NWG` 个 warpgroup（`BM = NWG*64`），
//   默认 `NWG=2`（BM=128）行为逐位不变。`--wg3` 用 `NWG=3`（BM=192、384 线程、1 CTA/SM）：
//   每个 KV 元素被贡献的 CTA 数再 ÷1.5（相对 BM=128）⇒ L2 `red` 目标 ≈ 114.5M×128/192≈76M…
//   实际按 m 块数 S/BM 比例降（64→192 是 3×）。同时 12 warp/SM（vs wgmma2 的 8）缓解延迟。 
template <int HD, int BM = 128, int BN = 32, int NWG = 2>
__global__ void __launch_bounds__(NWG * 128, 1)
fa_bwd_fp8_wgmma2_kernel(const unsigned char* __restrict__ q8,
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
  static_assert(HD == 128, "wgmma2 目前只做 HD=128");
  static_assert(NWG >= 2 && BM == NWG * 64, "BM 必须 = NWG*64");
  constexpr int TH = NWG * 128;
  constexpr int PSLD = HD + 8;   // 136
  constexpr int QTS = BM + 16;   // 144
  constexpr int DSS2 = BN + 16;  // 48
  constexpr int PSS = BN + 5;    // 37
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024;  // 16384
  constexpr int KS_SZ = (BN / 8) * (HD / 128) * 1024;  // 4096
  constexpr int qp_bytes = (BM / 2) * PSLD * 2;
  constexpr int kp_bytes = (BN / 2) * PSLD * 2;

  extern __shared__ __align__(16) char smem[];
  // wgmma 的 SW128 描述符要求 1024B 对齐 ⇒ 手动对齐动态 smem 基址（host 多给 1024B）。
  char* base = smem;
  {
    const uint32_t a0 = smem_u32(smem);
    base = smem + ((1024u - (a0 & 1023u)) & 1023u);
  }
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(base);
  unsigned char* dOs = Qs + QS_SZ;
  unsigned char* Ks  = dOs + QS_SZ;
  unsigned char* Vs  = Ks + KS_SZ;
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(Vs + KS_SZ);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + kp_bytes;
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
  float* Ps = scales + kNScale;
  float* Ss = Ps + BM * PSS;
  // F6：Ap/dS3 独立缓冲（wg2 里别名 Ks/Vs；SW128 下 Ks/Vs 各 4096B 放不下 BN×QTS=4608B）。
  unsigned char* Ap  = reinterpret_cast<unsigned char*>(Ss + BM * PSS);
  unsigned char* dS3 = Ap + BN * QTS;
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
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ksplit;
  const int nt_end = (part + 1) * ntiles / ksplit;
  if (nt_end <= nt_begin) return;

  // ---- 载入 Q/dO 为 SW128（供 wgmma 直读）+ 重建 Qp/dOp（供 GEMM3/4 的 ldmatrix.trans）----
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
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD)) = q0;
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = q1;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD)) = o0;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = o1;
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
  // 首个 tile 的 K/V 直接读成 SW128 + 重建 Kp（`SW=true` 分支）。
  kv_load_pair<HD, BN, true, TH>(k8, v8, nt_begin * BN, S, Hkv, hkv, b * S, tid, Ks, Vs, Kp,
                                 PSLD, PSLD);
  if (tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
  }
  __syncthreads();

  // ---- PREL：本线程负责行的 LSE/D 预装寄存器（wgmma 行布局）----
  float lse_r[2], del_r[2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int r = wg * 64 + wl * 16 + g + (t ? 8 : 0);
    const int qi = m0 + r;
    const size_t idx = ((size_t)(b * S + qi)) * H + h;
    const bool ok = qi < S;
    lse_r[t] = ok ? lse[idx] : 0.f;
    del_r[t] = ok ? delta[idx] : 0.f;
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
    // O91：K/V/Q/dO 的 SW128 是 **generic 写**，下一句 wgmma（async proxy）读 smem 前
    //   必须 `fence.proxy.async.shared::cta`（同 p155/O81 的坑；NWG=3 的线程时序更易触发
    //   偶发 race ⇒ 首次实测出现 dk/dv=inf 的非确定错误）。每迭代顶部补一次。
    bulk_reduce_fence();
    // ---- (1) S = scale·QKᵀ → P=exp(S−LSE)； (2) dP = dO·Vᵀ → dS = P∘(dP−D) ----
    //   两条 wgmma 一起发、统一 wait0；每个 warpgroup 各算自己 64 行的 S/dP。
    {
      float sacc[16], dpacc[16];
      const char* qa_sw = reinterpret_cast<const char*>(Qs) + (size_t)wg * ((64 / 8) * 1024);
      const char* doa_sw = reinterpret_cast<const char*>(dOs) + (size_t)wg * ((64 / 8) * 1024);
      wgmma_mn32_issue<0>(qa_sw, reinterpret_cast<const char*>(Ks), HD, sacc);
      wgmma_mn32_issue<1>(doa_sw, reinterpret_cast<const char*>(Vs), HD, dpacc);
      wgmma_wait0_fp8();
      float pval[16];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi)) {
            float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            p = fexp(sval - lse_r[q >= 2 ? 1 : 0]);
          }
          pval[j * 4 + q] = p;
          Ps[r * PSS + c] = p;
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r;
          float dpv = dpacc[j * 4 + q] * dos_s[r] * vs_s[c];
          float del = del_r[q >= 2 ? 1 : 0];
          if (qi >= S) del = 0.f;
          Ss[r * PSS + c] = pval[j * 4 + q] * (dpv - del);
        }
    }
    __syncthreads();

    // ---- fold：把 Ps/Ss（全 BM 行）量化成 GEMM3/4/5 的操作数并乘 rowwise scale ----
    if (wid < 8) {
      // Ap[j][m]=P[m][j]*dos[m] (e4m3)，dS3[j][m]=dS[m][j]*qs[m] (e5m2)：每 warp 4 个 j，
      //   8 个 lane 一组沿 m（每组 16 个 m）。
      // O91：每个 (j) 行沿 m 覆盖全 BM 行。`BM=128` 时 `NPC=1,REM=0`，与旧代码逐位相同；
      //   `BM=192`（NWG=3）时 NPC=1 覆盖 m 0..127、REM=64 再覆盖 m 128..191（每 lane 组 8 行）。
      constexpr int NPC = BM / 128;        // 完整 128 行块数
      constexpr int REM = BM - NPC * 128;  // 尾块行数
      const int jj = lane >> 3, sub = lane & 7;
      const int j = wid * 4 + jj;   // 0..31
      float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
      for (int mh = 0; mh < NPC; ++mh)
#pragma unroll
        for (int t = 0; t < 16; ++t) {
          int m = mh * 128 + sub * 16 + t;
          amaxA = fmaxf(amaxA, fabsf(Ps[m * PSS + j] * dos_s[m]));
          amax3 = fmaxf(amax3, fabsf(Ss[m * PSS + j] * qs_s[m]));
        }
      if (REM > 0) {
#pragma unroll
        for (int t = 0; t < REM / 8; ++t) {
          int m = NPC * 128 + sub * (REM / 8) + t;
          amaxA = fmaxf(amaxA, fabsf(Ps[m * PSS + j] * dos_s[m]));
          amax3 = fmaxf(amax3, fabsf(Ss[m * PSS + j] * qs_s[m]));
        }
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
#pragma unroll
      for (int mh = 0; mh < NPC; ++mh) {
        uint32_t pa[4] = {0, 0, 0, 0}, d3[4] = {0, 0, 0, 0};
#pragma unroll
        for (int t4 = 0; t4 < 4; ++t4)
#pragma unroll
          for (int tt = 0; tt < 4; ++tt) {
            int m = mh * 128 + sub * 16 + t4 * 4 + tt;
            pa[t4] |= (uint32_t)cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA) << (8 * tt);
            d3[t4] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3) << (8 * tt);
          }
        *reinterpret_cast<uint4*>(Ap + j * QTS + mh * 128 + sub * 16) =
            make_uint4(pa[0], pa[1], pa[2], pa[3]);
        *reinterpret_cast<uint4*>(dS3 + j * QTS + mh * 128 + sub * 16) =
            make_uint4(d3[0], d3[1], d3[2], d3[3]);
      }
      if (REM > 0) {
        uint32_t pa[2] = {0, 0}, d3[2] = {0, 0};
#pragma unroll
        for (int t2 = 0; t2 < 2; ++t2)
#pragma unroll
          for (int tt = 0; tt < 4; ++tt) {
            int m = NPC * 128 + sub * (REM / 8) + t2 * 4 + tt;
            pa[t2] |= (uint32_t)cvt_e4m3(Ps[m * PSS + j] * dos_s[m] / scA) << (8 * tt);
            d3[t2] |= (uint32_t)cvt_e5m2(Ss[m * PSS + j] * qs_s[m] / sc3) << (8 * tt);
          }
        *reinterpret_cast<uint2*>(Ap + j * QTS + NPC * 128 + sub * (REM / 8)) =
            make_uint2(pa[0], pa[1]);
        *reinterpret_cast<uint2*>(dS3 + j * QTS + NPC * 128 + sub * (REM / 8)) =
            make_uint2(d3[0], d3[1]);
      }
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

    // ---- (3) dV = Pᵀ·dO：A=Ap[j][m]（全 BM 重叠），B=dOp 配对布局 ----
    //   GEMM3/4 输出只有 BN(32)×HD(128)，8 个 warp 足矣；NWG>2 时 wid≥8 的 warpgroup 跳过。
    if (wid < 8) {
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
    if (wid < 8) {
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
    // ---- (5) dQ += scale·dS·K：A=dS2[m][j]（全 BM 行），B=Kp 配对布局 ----
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
      kv_load_pair<HD, BN, true, TH>(k8, v8, (nt + 1) * BN, S, Hkv, hkv, b * S, tid, Ks, Vs, Kp,
                                     PSLD, PSLD);
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
// 3d) fa_bwd_fp8_wgmma2_tma_kernel —— O90（F6-step3）：在 3c) 的 wgmma2（BM=128、2 warpgroup、
//     GEMM1/2 wgmma + SW128）基础上，把 **K/V 改成 4D-TMA**（K 双缓冲、V 单缓冲），复现默认
//     `kvtma` 主 kernel 的 K/V TMA 时序（O41）。Q/dO 仍手工载入（每 CTA 一次，非热点）。
//     动机：wgmma2 只有默认档的 0.56×，除 1 CTA/SM 外，每 tile 的 K/V「标量 global 读 +
//     __syncthreads」串行是主因；用 TMA + mbarrier 把 K/V 搬运与 GEMM 重叠，同时保留 BM=128
//     把 L2 `red` 砍半的收益。opt-in `--wg2tma`，默认路径一行未改。
// =============================================================================
template <int HD, int BM = 128, int BN = 32>
__global__ void __launch_bounds__(256, 1)
fa_bwd_fp8_wgmma2_tma_kernel(const __grid_constant__ CUtensorMap kmap,
                             const __grid_constant__ CUtensorMap vmap,
                             const unsigned char* __restrict__ q8,
                             const float* __restrict__ qs,
                             const float* __restrict__ ks,
                             const float* __restrict__ vs,
                             const unsigned char* __restrict__ do8,
                             const float* __restrict__ dos,
                             const float* __restrict__ delta,
                             const float* __restrict__ lse,
                             float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                             float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                             int causal, int ksplit) {
  static_assert(HD == 128, "wgmma2_tma 目前只做 HD=128");
  constexpr int TH = 256;
  constexpr int PSLD = HD + 8;   // 136
  constexpr int QTS = BM + 16;   // 144
  constexpr int DSS2 = BN + 16;  // 48
  constexpr int PSS = BN + 5;    // 37
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024;  // 16384
  constexpr int KS_SZ = (BN / 8) * (HD / 128) * 1024;  // 4096
  constexpr int qp_bytes = (BM / 2) * PSLD * 2;
  constexpr int kp_bytes = (BN / 2) * PSLD * 2;

  extern __shared__ __align__(16) char smem[];
  char* base = smem;
  {
    const uint32_t a0 = smem_u32(smem);
    base = smem + ((1024u - (a0 & 1023u)) & 1023u);
  }
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(base);
  unsigned char* dOs = Qs + QS_SZ;
  unsigned char* Ks  = dOs + QS_SZ;              // K 双缓冲：2*KS_SZ
  unsigned char* Vs  = Ks + 2 * KS_SZ;
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(Vs + KS_SZ);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + kp_bytes;
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
  float* Ps = scales + kNScale;
  float* Ss = Ps + BM * PSS;
  unsigned char* Ap  = reinterpret_cast<unsigned char*>(Ss + BM * PSS);
  unsigned char* dS3 = Ap + BN * QTS;
  uint64_t* bars = reinterpret_cast<uint64_t*>(dS3 + BN * QTS);   // 3 个 mbarrier
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;

  const int part = blockIdx.x % ksplit, mblk = blockIdx.x / ksplit;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int wg = wid >> 2, wl = wid & 3;   // 两个 warpgroup
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ksplit;
  const int nt_end = (part + 1) * ntiles / ksplit;
  if (nt_end <= nt_begin) return;

  // ---- 手工载入 Q/dO 为 SW128（供 wgmma 直读）+ 重建 Qp/dOp（供 GEMM3/4 的 ldmatrix.trans）----
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
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD)) = q0;
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = q1;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD)) = o0;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = o1;
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

  // Kp 重建：从 SW128 K stage 逐字节转置成 K 配对布局（与 wgmma2 的 kv_load_pair 内建部分同款）。
  auto rebuild_kp = [&](const unsigned char* Kb) {
    const int nd4k = HD / 4;
    for (int u = tid; u < (BN / 2) * nd4k; u += TH) {
      int rp = u / nd4k, dq = (u % nd4k) * 4;
      uint32_t k0 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2, dq, HD));
      uint32_t k1 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2 + 1, dq, HD));
      uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * PSLD + dq);
      kpw[0] = __byte_perm(k0, k1, 0x5140);
      kpw[1] = __byte_perm(k0, k1, 0x7362);
    }
  };

  // ---- K/V 4D-TMA prologue：K[nt_begin]→stage0、K[nt_begin+1]→stage1、V[nt_begin]→Vs ----
  if (tid == 0) {
    mbar_init(bars + 0, 1);
    mbar_init(bars + 1, 1);
    mbar_init(bars + 2, 1);
  }
  __syncthreads();
  if (tid == 0) {
    mbar_arrive_expect(bars + 0, KS_SZ);
    tma_load_4d(Ks, &kmap, 0, nt_begin * BN, hkv, b, bars + 0);
    if (nt_begin + 1 < nt_end) {
      mbar_arrive_expect(bars + 1, KS_SZ);
      tma_load_4d(Ks + KS_SZ, &kmap, 0, (nt_begin + 1) * BN, hkv, b, bars + 1);
    }
    mbar_arrive_expect(bars + 2, KS_SZ);
    tma_load_4d(Vs, &vmap, 0, nt_begin * BN, hkv, b, bars + 2);
  }
  if (tid < BN) {
    int jg = nt_begin * BN + tid;
    ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
  }

  // ---- PREL：本线程负责行的 LSE/D 预装寄存器（wgmma 行布局）----
  float lse_r[2], del_r[2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int r = wg * 64 + wl * 16 + g + (t ? 8 : 0);
    const int qi = m0 + r;
    const size_t idx = ((size_t)(b * S + qi)) * H + h;
    const bool ok = qi < S;
    lse_r[t] = ok ? lse[idx] : 0.f;
    del_r[t] = ok ? delta[idx] : 0.f;
  }

  // ---- dQ 寄存器累加（O7）：GEMM5 每 warp 32 行 × 64 列 = 2×8×4。----
  float dqacc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

  int kc0 = 0, kc1 = 0, vuse = 0;
  mbar_wait(bars + 0, 0); kc0++;
  mbar_wait(bars + 2, 0); vuse++;
  __syncthreads();
  rebuild_kp(Ks);
  __syncthreads();

  int stg = 0;
  for (int nt = nt_begin; nt < nt_end; ++nt, stg ^= 1) {
    const int j0 = nt * BN;
    // ---- (1) S = scale·QKᵀ → P=exp(S−LSE)； (2) dP = dO·Vᵀ → dS = P∘(dP−D) ----
    {
      float sacc[16], dpacc[16];
      const char* qa_sw = reinterpret_cast<const char*>(Qs) + (size_t)wg * ((64 / 8) * 1024);
      const char* doa_sw = reinterpret_cast<const char*>(dOs) + (size_t)wg * ((64 / 8) * 1024);
      wgmma_mn32_issue<0>(qa_sw, reinterpret_cast<const char*>(Ks + stg * KS_SZ), HD, sacc);
      wgmma_mn32_issue<1>(doa_sw, reinterpret_cast<const char*>(Vs), HD, dpacc);
      wgmma_wait0_fp8();
      float pval[16];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi)) {
            float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            p = fexp(sval - lse_r[q >= 2 ? 1 : 0]);
          }
          pval[j * 4 + q] = p;
          Ps[r * PSS + c] = p;
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r;
          float dpv = dpacc[j * 4 + q] * dos_s[r] * vs_s[c];
          float del = del_r[q >= 2 ? 1 : 0];
          if (qi >= S) del = 0.f;
          Ss[r * PSS + c] = pval[j * 4 + q] * (dpv - del);
        }
    }
    __syncthreads();

    // ---- fold：把 Ps/Ss（全 BM=128 行）量化成 GEMM3/4/5 的操作数并乘 rowwise scale ----
    {
      const int jj = lane >> 3, sub = lane & 7;
      const int j = wid * 4 + jj;
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
    // ---- 发下一 tile 的 K[nt+2]（覆写本迭代读完的 stage stg）与 V[nt+1]；等 K[nt+1]、重建
    //      Kp、等 V[nt+1]（与 O41 的时序一致；K 双缓冲、V 单缓冲）----
    if (tid == 0) {
      if (nt + 2 < nt_end) {
        mbar_arrive_expect(bars + 0 + stg, KS_SZ);
        tma_load_4d(Ks + stg * KS_SZ, &kmap, 0, (nt + 2) * BN, hkv, b, bars + 0 + stg);
      }
      if (nt + 1 < nt_end) {
        mbar_arrive_expect(bars + 2, KS_SZ);
        tma_load_4d(Vs, &vmap, 0, (nt + 1) * BN, hkv, b, bars + 2);
      }
    }
    if (nt + 1 < nt_end) {
      if (tid < BN) {
        int jg = (nt + 1) * BN + tid;
        ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
        vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
      }
      const int sk = stg ^ 1;
      int kc = sk ? kc1 : kc0;
      mbar_wait(bars + 0 + sk, (uint32_t)(kc & 1));
      if (sk) kc1++; else kc0++;
      __syncthreads();
      rebuild_kp(Ks + sk * KS_SZ);
      __syncthreads();
      mbar_wait(bars + 2, (uint32_t)(vuse & 1));
      vuse++;
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
// 3e) fa_bwd_fp8_wgmma2_ws_kernel —— O121（F3b 主体化第一步）：把 O120 的
//     **warp specialization**（独立 producer warpgroup + mbarrier ring）落进
//     `wgmma2_tma`（BM=128、2 compute warpgroup、GEMM1/2 wgmma + K/V 4D-TMA）。
// =============================================================================
// 动机：O90 的 `fa_bwd_fp8_wgmma2_tma_kernel` 用 K 双缓冲把 dK/dV 跨 CTA 的 L2 `red`
//   精确砍半（114.52M→58.20M，O90/O91 ncu），但 **1 CTA/SM** 下每 tile 的 5 个
//   `__syncthreads` 把 K/V 搬运（TMA）与 wgmma 串成一条依赖链，跨 tile 延迟没有别的
//   独立 CTA 来填 ⇒ 时间反而 0.54× 默认档（O90/O91）。O91 把病因归为「单一 barrier 域」，
//   但只从「改成多 warpgroup 后变慢」间接推断。**O120 独立冒烟在同一份 fp8 K/V TMA 几何
//   （UINT8 / SW128 / box={128,32} / KS_SZ=4096B）上把同步机制本身换掉**：独立 producer
//   warpgroup 跑 ring 预取、consumer 只等 `mbarrier(full)`→wgmma→`arrive(empty)`、无
//   `__syncthreads`，同字节的 L2 读吞吐 1343→2576 GB/s（**2.0×**）、Duration 1.67→0.82ms，
//   而 `lts op_read`、warp 数**一字不变** ⇒ 病因确认、WS 是 1 CTA/SM 的解锁路径。
//
// 本 kernel 是 O120 的**主 kernel 化**（GEMM 数据通路逐字沿用 O90，只换同步结构）：
//   * 384 线程 = **2 个 compute warpgroup**（wid 0..7，跑全部 5 个 GEMM，与 O90 逐字一致）
//     + **1 个 producer warpgroup**（wid 8..11，只发 K/V 4D-TMA）。
//   * compute 侧仅 3 处需要跨 warp 同步（epilogue→fold、fold→GEMM3/4/5、GEMM5→下一次
//     rebuild_kp），改用**命名 barrier** `bar.sync 1, 256`（只含 2 个 compute WG）。
//   * K（KSTAGE=2）与 V（VSTAGE=2）各一个 **ring**：producer 用 `full/empty` mbarrier 把
//     TMA 与 compute 解耦（empty 的 count = 256 = 全部 compute 线程，41 篇的坑）。
//   * Kp/dS2/Ps/Ss/Ap/dS3 仍单缓冲（compute 侧由命名 barrier 串起来，无需翻倍）。
// `--wg2ws` opt-in，默认路径一行未改；数值应与 O90 `wgmma2_tma` 只在 fp32 归约次序上不同。
template <int HD, int BM = 128, int BN = 32, int KSTAGE = 2, int VSTAGE = 2>
__global__ void __launch_bounds__(384, 1)
fa_bwd_fp8_wgmma2_ws_kernel(const __grid_constant__ CUtensorMap kmap,
                            const __grid_constant__ CUtensorMap vmap,
                            const unsigned char* __restrict__ q8,
                            const float* __restrict__ qs,
                            const float* __restrict__ ks,
                            const float* __restrict__ vs,
                            const unsigned char* __restrict__ do8,
                            const float* __restrict__ dos,
                            const float* __restrict__ delta,
                            const float* __restrict__ lse,
                            float* __restrict__ dq_acc, float* __restrict__ dk_acc,
                            float* __restrict__ dv_acc, int S, int H, int Hkv, float scale,
                            int causal, int ksplit) {
  static_assert(HD == 128, "wgmma2_ws 目前只做 HD=128");
  constexpr int NTH = 384;      // 3 warpgroup：2 compute + 1 producer
  constexpr int NCOMP = 256;    // 2 compute warpgroup
  constexpr int PSLD = HD + 8;   // 136
  constexpr int QTS = BM + 16;   // 144
  constexpr int DSS2 = BN + 16;  // 48
  constexpr int PSS = BN + 5;    // 37
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024;  // 16384
  constexpr int KS_SZ = (BN / 8) * (HD / 128) * 1024;  // 4096
  constexpr int qp_bytes = (BM / 2) * PSLD * 2;
  constexpr int kp_bytes = (BN / 2) * PSLD * 2;

  extern __shared__ __align__(16) char smem[];
  char* base = smem;
  {
    const uint32_t a0 = smem_u32(smem);
    base = smem + ((1024u - (a0 & 1023u)) & 1023u);
  }
  unsigned char* Qs  = reinterpret_cast<unsigned char*>(base);
  unsigned char* dOs = Qs + QS_SZ;
  unsigned char* Ks  = dOs + QS_SZ;                 // KSTAGE * KS_SZ（ring）
  unsigned char* Vs  = Ks + KSTAGE * KS_SZ;         // VSTAGE * KS_SZ（ring）
  uint16_t* Qp  = reinterpret_cast<uint16_t*>(Vs + VSTAGE * KS_SZ);
  uint16_t* dOp = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(Qp) + qp_bytes);
  uint16_t* Kp  = reinterpret_cast<uint16_t*>(reinterpret_cast<unsigned char*>(dOp) + qp_bytes);
  unsigned char* dS2 = reinterpret_cast<unsigned char*>(Kp) + kp_bytes;
  float* scales = reinterpret_cast<float*>(dS2 + BM * DSS2);
  float* Ps = scales + kNScale;
  float* Ss = Ps + BM * PSS;
  unsigned char* Ap  = reinterpret_cast<unsigned char*>(Ss + BM * PSS);
  unsigned char* dS3 = Ap + BN * QTS;
  uint64_t* bars = reinterpret_cast<uint64_t*>(dS3 + BN * QTS);   // NBAR 个 mbarrier
  uint64_t* fullK  = bars;
  uint64_t* emptyK = bars + KSTAGE;
  uint64_t* fullV  = bars + 2 * KSTAGE;
  uint64_t* emptyV = bars + 2 * KSTAGE + VSTAGE;
  float* qs_s = scales;
  float* ks_s = qs_s + BM;
  float* vs_s = ks_s + BN;
  float* dos_s = vs_s + BN;
  float* sA = dos_s + BM;
  float* sds2 = sA + BN;
  float* sds3 = sds2 + BM;

  const int part = blockIdx.x % ksplit, mblk = blockIdx.x / ksplit;
  const int h = blockIdx.y, b = blockIdx.z;
  const int hkv = h / (H / Hkv);
  const int tid = threadIdx.x, wid = tid >> 5, lane = tid & 31;
  const int isprod = (wid >= 8);
  const int wg = wid >> 2, wl = wid & 3;   // compute：wg∈{0,1}
  const int g = lane >> 2, c2 = (lane & 3) * 2;
  const int m0 = mblk * BM;

  const int ncols = causal ? min(S, m0 + BM) : S;
  const int ntiles = (ncols + BN - 1) / BN;
  const int nt_begin = part * ntiles / ksplit;
  const int nt_end = (part + 1) * ntiles / ksplit;
  const int NT = nt_end - nt_begin;
  if (NT <= 0) return;

  // ---- 一次性初始化 mbarrier（tid==0）----
  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < KSTAGE; ++s) mbar_init(fullK + s, 1);
#pragma unroll
    for (int s = 0; s < KSTAGE; ++s) mbar_init(emptyK + s, NCOMP);
#pragma unroll
    for (int s = 0; s < VSTAGE; ++s) mbar_init(fullV + s, 1);
#pragma unroll
    for (int s = 0; s < VSTAGE; ++s) mbar_init(emptyV + s, NCOMP);
  }

  // ---- 一次性载入 Q/dO 为 SW128（供 wgmma 直读）+ 重建 Qp/dOp（全 384 线程）----
  {
    const int nd4 = HD / 4;
    for (int u = tid; u < (BM / 2) * nd4; u += NTH) {
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
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2, dq, HD)) = q0;
      *reinterpret_cast<uint32_t*>(Qs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = q1;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2, dq, HD)) = o0;
      *reinterpret_cast<uint32_t*>(dOs + sw128_off_fp8(rp * 2 + 1, dq, HD)) = o1;
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

  // ---- 全 CTA 同步一次：mbarrier 就绪 + Q/dO/配对布局可见；之后 producer 与 compute 分叉 ----
  __syncthreads();
  if (isprod) {
    // ============================ producer（wid 8..11）============================
    if (tid == 256) {
      // **K 与 V 必须在同一个循环里交替推进**（否则 K 的前瞻队列会把 producer 阻塞在
      //   emptyK 上、V 的后继 tile 永不发出 ⇒ consumer 等 fullV 死锁——S4096 首测踩到）。
      //   t<KSTAGE（VSTAGE）时 ring 尚未回绕，无需等 empty；之后每 tile 先等上一轮消费完。
      int ekp[KSTAGE], evp[VSTAGE];
#pragma unroll
      for (int s = 0; s < KSTAGE; ++s) ekp[s] = 0;
#pragma unroll
      for (int s = 0; s < VSTAGE; ++s) evp[s] = 0;
      for (int t = 0; t < NT; ++t) {
        {
          const int s = t % KSTAGE;
          if (t >= KSTAGE) {
            mbar_wait(emptyK + s, (uint32_t)(ekp[s] & 1));
            ekp[s] ^= 1;
          }
          mbar_arrive_expect(fullK + s, KS_SZ);
          tma_load_4d(Ks + s * KS_SZ, &kmap, 0, (nt_begin + t) * BN, hkv, b, fullK + s);
        }
        {
          const int s = t % VSTAGE;
          if (t >= VSTAGE) {
            mbar_wait(emptyV + s, (uint32_t)(evp[s] & 1));
            evp[s] ^= 1;
          }
          mbar_arrive_expect(fullV + s, KS_SZ);
          tma_load_4d(Vs + s * KS_SZ, &vmap, 0, (nt_begin + t) * BN, hkv, b, fullV + s);
        }
      }
    }
    return;   // 其余 producer 线程退出（不再参与任何 CTA-wide/命名 barrier）
  }

  // ============================ compute（wid 0..7）============================
  // ---- PREL：本线程负责行的 LSE/D 预装寄存器（wgmma 行布局）----
  float lse_r[2], del_r[2];
#pragma unroll
  for (int t = 0; t < 2; ++t) {
    const int r = wg * 64 + wl * 16 + g + (t ? 8 : 0);
    const int qi = m0 + r;
    const size_t idx = ((size_t)(b * S + qi)) * H + h;
    const bool ok = qi < S;
    lse_r[t] = ok ? lse[idx] : 0.f;
    del_r[t] = ok ? delta[idx] : 0.f;
  }

  // ---- dQ 寄存器累加（O7）：GEMM5 每 warp 32 行 × 64 列 = 2×8×4。----
  float dqacc[2][8][4];
#pragma unroll
  for (int i = 0; i < 2; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int q = 0; q < 4; ++q) dqacc[i][j][q] = 0.f;

  int fkp[KSTAGE], fvp[VSTAGE];
#pragma unroll
  for (int s = 0; s < KSTAGE; ++s) fkp[s] = 0;
#pragma unroll
  for (int s = 0; s < VSTAGE; ++s) fvp[s] = 0;

  const int nd4k = HD / 4;
  for (int t = 0; t < NT; ++t) {
    const int s = t % KSTAGE, sv = t % VSTAGE;
    const int j0 = (nt_begin + t) * BN;
    // ---- 等本 tile 的 K/V 到齐（producer 已异步搬运）----
    mbar_wait(fullK + s, (uint32_t)(fkp[s] & 1));
    fkp[s] ^= 1;
    mbar_wait(fullV + sv, (uint32_t)(fvp[sv] & 1));
    fvp[sv] ^= 1;
    if (tid < BN) {
      int jg = j0 + tid;
      ks_s[tid] = (jg < S) ? ks[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
      vs_s[tid] = (jg < S) ? vs[((size_t)(b * S + jg)) * Hkv + hkv] : 1.f;
    }
    // ---- 从当前 K stage 的 SW128 逐字节转置重建 Kp（仅 compute 的 256 线程）----
    {
      const unsigned char* Kb = Ks + s * KS_SZ;
      for (int u = tid; u < (BN / 2) * nd4k; u += NCOMP) {
        int rp = u / nd4k, dq = (u % nd4k) * 4;
        uint32_t k0 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2, dq, HD));
        uint32_t k1 = *reinterpret_cast<const uint32_t*>(Kb + sw128_off_fp8(rp * 2 + 1, dq, HD));
        uint32_t* kpw = reinterpret_cast<uint32_t*>(Kp + rp * PSLD + dq);
        kpw[0] = __byte_perm(k0, k1, 0x5140);
        kpw[1] = __byte_perm(k0, k1, 0x7362);
      }
    }
    named_bar_sync_compute();
    // O91：K/V/Q/dO 的 SW128 是 generic 写，wgmma（async proxy）读前必须 fence。
    bulk_reduce_fence();

    // ---- (1) S = scale·QKᵀ → P=exp(S−LSE)； (2) dP = dO·Vᵀ → dS = P∘(dP−D) ----
    {
      float sacc[16], dpacc[16];
      const char* qa_sw = reinterpret_cast<const char*>(Qs) + (size_t)wg * ((64 / 8) * 1024);
      const char* doa_sw = reinterpret_cast<const char*>(dOs) + (size_t)wg * ((64 / 8) * 1024);
      wgmma_mn32_issue<0>(qa_sw, reinterpret_cast<const char*>(Ks + s * KS_SZ), HD, sacc);
      wgmma_mn32_issue<1>(doa_sw, reinterpret_cast<const char*>(Vs + sv * KS_SZ), HD, dpacc);
      wgmma_wait0_fp8();
      float pval[16];
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r, jg = j0 + c;
          float p = 0.f;
          if (qi < S && jg < S && !(causal && jg > qi)) {
            float sval = sacc[j * 4 + q] * scale * qs_s[r] * ks_s[c];
            p = fexp(sval - lse_r[q >= 2 ? 1 : 0]);
          }
          pval[j * 4 + q] = p;
          Ps[r * PSS + c] = p;
        }
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          int r = wg * 64 + wl * 16 + g + (q >= 2 ? 8 : 0);
          int c = j * 8 + c2 + (q & 1);
          int qi = m0 + r;
          float dpv = dpacc[j * 4 + q] * dos_s[r] * vs_s[c];
          float del = del_r[q >= 2 ? 1 : 0];
          if (qi >= S) del = 0.f;
          Ss[r * PSS + c] = pval[j * 4 + q] * (dpv - del);
        }
    }
    // K/V 本 tile 已读完（GEMM1 读 K、GEMM2 读 V，wgmma wait0 已完成）⇒ 释放 ring stage。
    mbar_arrive(emptyK + s);
    mbar_arrive(emptyV + sv);
    named_bar_sync_compute();

    // ---- fold：把 Ps/Ss（全 BM=128 行）量化成 GEMM3/4/5 的操作数并乘 rowwise scale ----
    {
      const int jj = lane >> 3, sub = lane & 7;
      const int j = wid * 4 + jj;
      float amaxA = 0.f, amax3 = 0.f;
#pragma unroll
      for (int t2 = 0; t2 < 16; ++t2) {
        int m = sub * 16 + t2;
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
      const int ml = lane >> 1, sub2 = lane & 1;
      const int m = wid * 16 + ml;
      float amax2 = 0.f;
#pragma unroll
      for (int t2 = 0; t2 < 16; ++t2) {
        int j = sub2 * 16 + t2;
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
    named_bar_sync_compute();

    // ---- (3) dV = Pᵀ·dO：A=Ap[j][m]，B=dOp 配对布局 ----
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
    // ---- (5) dQ += scale·dS·K：A=dS2[m][j]，B=Kp 配对布局 ----
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
    // 下一次迭代的 rebuild_kp 会覆写 Kp、epilogue 会覆写 Ps/Ss ⇒ 必须等本轮 GEMM3/4/5 读完。
    named_bar_sync_compute();
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
#endif  // FA_WGMMA

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

// 第 162 轮 O68：非 causal（full）D=128 的 LSE 用 O54 的均衡版
//   `lse_mma_kernel_bal<HD,PIPE,FULL=true>`（一个 CTA 一个 m 块 + cp.async 双缓冲）
//   替代 O1 的 `lse_mma_kernel`（无流水，逐标量 global→smem）。默认 1；`--lsefull=0`
//   退回 O1 做同 binary A/B。定长与 varlen 两条 full 路径都读这个开关（故用文件作用域）。
static int g_lse_full_opt = 1;
// 第 166 轮 O72：varlen full D=128 的 LSE 是否走 4D-TMA（对齐定长 O70）。默认 1；`--lsetmavarlen=0`
//   退回 O68 的 cp.async 均衡版做同 binary A/B。需要 `-DFA_WGMMA -DFA_TMA` 构建（否则恒 0）。
static int g_lse_tma_varlen = 1;
// O101（第 195 轮）：**定长 full 的 LSE 补齐均衡/流水/split**——此前 D=512（MLA）定长 full 的
//   LSE 一直是 O1 的 `lse_mma_kernel`（逐标量 global→smem、无 cp.async 流水、无 K 维 split）。
//   默认 1 = 走 `lse_mma_kernel_bal<512,1,true>`；`--lse512old=1` 退回 O1 做同 binary A/B。
static int g_lse_full512_opt = 1;

#if defined(FA_WGMMA) && defined(FA_TMA)
// O32：为 LSE 的 Q/K 建 4D TMA 描述符（dims={D,S,H,B}，SW128，box={128,64,1,1}）。
// fp8 一行 = 128 字节 = SW128 atom 整行 ⇒ 一个 box 覆盖整个 head_dim（不像 fp16 需 2 chunk）。
// dtype 用 UINT8（CUDA 13 驱动枚举无 FLOAT8_E4M3）。globalStride（字节）：dim1(S) 行距 = H*D，
// dim2(H) 头距 = D，dim3(B) 批距 = S*H*D（元素即字节）。要求 16B 对齐（D=128 恒成立）。
static CUtensorMap make_lse_map_fp8(const void* ptr, long long H, long long S, long long D,
                                    long long B, uint32_t boxR = 64) {
  CUtensorMap map;
  uint64_t dims[4] = {(uint64_t)D, (uint64_t)S, (uint64_t)H, (uint64_t)B};
  uint64_t strides[3] = {(uint64_t)(H * D), (uint64_t)D, (uint64_t)(S * H * D)};
  uint32_t box[4] = {128, boxR, 1, 1};
  uint32_t estr[4] = {1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4, (void*)ptr, dims, strides, box, estr,
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
    if (d > st.max_abs) st.max_abs = d;
    double r = d / (std::fabs((double)b[i]) + 1e-3);
    if (r > st.max_rel) st.max_rel = r;
  }
  return st;
}

// =============================================================================
// 模板 launcher：按 HD 选择实例并设置动态 smem 上限。
// =============================================================================
// O7：REGDQ 选择是否把 dQ 沿 nt 累加在寄存器里（见 kernels.cuh 主 kernel 说明）。
// O9c-2：WGMMA=true 时 GEMM1/2 走 wgmma（Q/dO/K/V 存 SW128），smem 用 wgmma 布局。
// O12：PREL=true 时把本线程负责的 LSE/D 预装寄存器（见 kernels.cuh），默认开。
// O7e-2：F16B=true 时 fold 的 Ap/dS3/dS2 用 16B 向量化写（见 kernels.cuh），默认开。
// O27：RCP=true 时 fold 量化用「每行 rcp + 乘法」代替逐元素精确除法（见 kernels.cuh）。
// O47：`NTH`/`NWAR` 透传给主 kernel（默认 128/2 与历史逐字等价；MLA 用 256/4）。
// O104（第 198 轮）：`HSWAP`（默认 false）同 `fa_bwd_fp8_mma_kernel`——把 O93 的跨 head 全局 LPT
//   调度推广到通用 mma/wgmma 主 kernel（D=256 默认走它）。`HSWAP=false` 与历史逐位相同。
template <int HD, int BM, int BN, bool REGDQ, bool WGMMA = false, bool PREL = true, bool F16B = true,
          bool RCP = true, int NTH = THREADS, int NWAR = WN, bool HSWAP = false>
static void launch_bwd_main(dim3 mg, const unsigned char* q8, const float* qs,
                            const unsigned char* k8, const float* ks,
                            const unsigned char* v8, const float* vs,
                            const unsigned char* do8, const float* dos,
                            const float* delta, const float* lse, float* dq_acc,
                            float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                            float scale, int causal, int ksplit,
                            const int* cu_seqlens = nullptr,
                            const int* mt_b = nullptr, const int* mt_m = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = WGMMA ? Cfg::smem_bytes_wgmma : Cfg::smem_bytes;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, false, false,
                            false, false, HSWAP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, false, false,
                        false, false, HSWAP><<<mg, NTH, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit, cu_seqlens, mt_b, mt_m);
}

// P3-4e/P3-4f：确定性 dK/dV/dQ 版主 kernel（`DET=true`）。把 dK/dV 的跨 CTA `atomicAdd` 换成
//   「按 (Q 头, Q 块) 分片的 partial 覆盖写 + `dkv_reduce_kernel` 固定次序求和」；`ksplit>1` 时
//   dQ 也走 partial（`dq_part`，按 part 分片）+ `dq_reduce_kernel`。仅用于定长（非 varlen）、
//   默认 mma 路径（A/B 实验，见 docs/03 §45/§57）。
//   注：`ksplit>1` 时 kRegDq 路径（HD=128）先在寄存器累加再**覆盖写** partial；非 kRegDq
//   路径（MLA/HD=512，P3-4k）逐 tile **累加进**本 CTA 独占的 partial 区（同一 (row,c) 由同一
//   线程按 nt 程序序写 ⇒ 确定），两路都无需再要求 `REGDQ=true`。`ksplit==1` 时 dQ 仍走无竞争
//   `red_add2`（与 P3-4e 逐位不变）。
// P3-4h：`NTH`/`NWAR` 参数化（默认 THREADS/WN ⇒ D=128 路径逐字不变），好让 MLA（HD=512）
//   复用同一 DET 路径、与默认 8-warp/256 几何（O47/O51）同几何做 A/B。
// P3-4i：`WGMMA`（默认 false，定长路径逐字不变）让 varlen D=128（varlen 构建恒为
//   `-DFA_WGMMA`，默认主 kernel 走 WGMMA）能用同一 DET 路径；`cu_seqlens/mt_b/mt_m` 透传
//   varlen 的 packed 索引（定长不传 ⇒ 逐字不变）。
// P3-4m：`KVPIPE` 让 DET 的 MLA 主 kernel 也能用 O51 的 K/V `cp.async` 回填流水（device 的
//   `fp8_mma_body` 早已同时支持 `KVPIPE && DET`，此前 host 未接线 ⇒ MLA DET 一直走非 kvpipe
//   主 kernel，白扔 O51 的 1.78–1.86×）。数值与旧非 kvpipe DET 逐位相同（只改搬运）。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          int NTH = THREADS, int NWAR = WN, bool WGMMA = false, bool KVPIPE = false,
          bool DET_HALF = false>
static void launch_bwd_main_det(dim3 mg, const unsigned char* q8, const float* qs,
                                const unsigned char* k8, const float* ks,
                                const unsigned char* v8, const float* vs,
                                const unsigned char* do8, const float* dos,
                                const float* delta, const float* lse, float* dq_acc,
                                float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                float scale, int causal, int ksplit, float* dk_part,
                                float* dv_part, int nblk, float* dq_part = nullptr,
                                const int* cu_seqlens = nullptr, const int* mt_b = nullptr,
                                const int* mt_m = nullptr, const int* part_base = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = WGMMA ? Cfg::smem_bytes_wgmma
                              : (KVPIPE ? Cfg::smem_bytes_kvpipe : Cfg::smem_bytes);
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, KVPIPE, true,
                            DET_HALF>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, WGMMA, PREL, F16B, RCP, NTH, NWAR, KVPIPE, true,
                        DET_HALF>
      <<<mg, NTH, kSmem>>>(q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
                           dv_acc, S, H, Hkv, scale, causal, ksplit, cu_seqlens, mt_b,
                           mt_m, dk_part, dv_part, nblk, dq_part, part_base);
}

// O51：K/V `cp.async` 回填流水版主 kernel（mma 后端，MLA/HD=512）。smem 比 `launch_bwd_main`
//   多 `2*BN*QTS`（Ap/dS3 独立缓冲），K/V 在 GEMM1/2 后异步回填下一 tile。数值与旧路径逐位相同。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          int NTH = THREADS, int NWAR = WN>
static void launch_bwd_main_kvpipe(dim3 mg, const unsigned char* q8, const float* qs,
                                   const unsigned char* k8, const float* ks,
                                   const unsigned char* v8, const float* vs,
                                   const unsigned char* do8, const float* dos,
                                   const float* delta, const float* lse, float* dq_acc,
                                   float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                   float scale, int causal, int ksplit,
                                   const int* cu_seqlens = nullptr,
                                   const int* mt_b = nullptr, const int* mt_m = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_kvpipe;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, false, PREL, F16B, RCP, NTH, NWAR, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kernel<HD, BM, BN, REGDQ, false, PREL, F16B, RCP, NTH, NWAR, true>
      <<<mg, NTH, kSmem>>>(q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
                           dv_acc, S, H, Hkv, scale, causal, ksplit, cu_seqlens, mt_b, mt_m);
}

// O37：Q/dO 4D-TMA 版主 kernel（仅 `-DFA_WGMMA -DFA_TMA` 构建、HD=128、WGMMA 路径）。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true>
static void launch_bwd_main_qdtma(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                  const unsigned char* q8, const float* qs,
                                  const unsigned char* k8, const float* ks,
                                  const unsigned char* v8, const float* vs,
                                  const unsigned char* do8, const float* dos,
                                  const float* delta, const float* lse, float* dq_acc,
                                  float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                  float scale, int causal, int ksplit) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_tma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_qdtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_qdtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP>
      <<<mg, THREADS, kSmem>>>(qmap, dmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse,
                               dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, ksplit);
}

// O41：Q/dO/K/V 全 4D-TMA 版主 kernel（roadmap「下一步候选 ①」；仅 `-DFA_WGMMA -DFA_TMA`
//   构建、HD=128、WGMMA 路径）。K 双缓冲、V 单缓冲，Kp 由 SW128 K tile 重建。
// O119：K/V TMA cluster multicast 宽度（1=关）。由 `main()` 依 `--mcast` 与形状门控设置。
static int g_mcast_width = 1;

template <int MC, int HD, int BM, int BN, bool REGDQ, bool PREL, bool F16B, bool RCP, bool HSWAP>
static void launch_kvtma_mc(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                            const CUtensorMap& kmap, const CUtensorMap& vmap,
                            const unsigned char* q8, const float* qs, const unsigned char* k8,
                            const float* ks, const unsigned char* v8, const float* vs,
                            const unsigned char* do8, const float* dos, const float* delta,
                            const float* lse, float* dq_acc, float* dk_acc, float* dv_acc,
                            int S, int H, int Hkv, float scale, int causal, int ksplit,
                            const int* mt_m, const int* slot_tab) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_kvtma;
  if constexpr (MC == 1) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false, HSWAP,
                                    1>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false, HSWAP, 1>
        <<<mg, THREADS, kSmem>>>(qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta,
                                 lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal, ksplit,
                                 nullptr, nullptr, nullptr, 0, nullptr, nullptr, mt_m, slot_tab);
  } else {
    CUDA_CHECK(cudaFuncSetAttribute(
        fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false, HSWAP,
                                    MC>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = mg;
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = kSmem;
    cfg.stream = nullptr;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = MC;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    CUDA_CHECK(cudaLaunchKernelEx(
        &cfg, fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, false, false, false,
                                          HSWAP, MC>,
        qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
        dv_acc, S, H, Hkv, scale, causal, ksplit, nullptr, nullptr, nullptr, 0, nullptr, nullptr,
        mt_m, slot_tab));
  }
}

template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          bool HSWAP = false>
static void launch_bwd_main_kvtma(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                  const CUtensorMap& kmap, const CUtensorMap& vmap,
                                  const unsigned char* q8, const float* qs,
                                  const unsigned char* k8, const float* ks,
                                  const unsigned char* v8, const float* vs,
                                  const unsigned char* do8, const float* dos,
                                  const float* delta, const float* lse, float* dq_acc,
                                  float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                  float scale, int causal, int ksplit,
                                  const int* mt_m = nullptr, const int* slot_tab = nullptr) {
  if constexpr (HSWAP) {
    const int mc = g_mcast_width;
    if (mc == 2)
      launch_kvtma_mc<2, HD, BM, BN, REGDQ, PREL, F16B, RCP, HSWAP>(
          mg, qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
          dv_acc, S, H, Hkv, scale, causal, ksplit, mt_m, slot_tab);
    else if (mc == 4)
      launch_kvtma_mc<4, HD, BM, BN, REGDQ, PREL, F16B, RCP, HSWAP>(
          mg, qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
          dv_acc, S, H, Hkv, scale, causal, ksplit, mt_m, slot_tab);
    else if (mc == 8)
      launch_kvtma_mc<8, HD, BM, BN, REGDQ, PREL, F16B, RCP, HSWAP>(
          mg, qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
          dv_acc, S, H, Hkv, scale, causal, ksplit, mt_m, slot_tab);
    else
      launch_kvtma_mc<1, HD, BM, BN, REGDQ, PREL, F16B, RCP, HSWAP>(
          mg, qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
          dv_acc, S, H, Hkv, scale, causal, ksplit, mt_m, slot_tab);
  } else {
    launch_kvtma_mc<1, HD, BM, BN, REGDQ, PREL, F16B, RCP, HSWAP>(
        mg, qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc,
        dv_acc, S, H, Hkv, scale, causal, ksplit, mt_m, slot_tab);
  }
}

// P3-4g：把 `--det` 从默认 mma 路径扩到 Hopper TMA 快路（`launch_bwd_main_kvtma` 的
//   `DET=true` 版）。dK/dV 走 partial + `dkv_reduce_kernel` 固定次序归约（可复现）。
//   P3-4g 原版锁 `ksplit=1`；**O102（第 196 轮）**放开 `ksplit>1`：dQ 走 per-part `dq_part`
//   + `dq_reduce_kernel`（同 mma 路径 P3-4f），恢复 split-K 并行度、逐位仍可复现。
template <int HD, int BM, int BN, bool REGDQ, bool PREL = true, bool F16B = true, bool RCP = true,
          bool DET_HALF = false>
static void launch_bwd_main_kvtma_det(dim3 mg, const CUtensorMap& qmap, const CUtensorMap& dmap,
                                      const CUtensorMap& kmap, const CUtensorMap& vmap,
                                      const unsigned char* q8, const float* qs,
                                      const unsigned char* k8, const float* ks,
                                      const unsigned char* v8, const float* vs,
                                      const unsigned char* do8, const float* dos,
                                      const float* delta, const float* lse, float* dq_acc,
                                      float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                      float scale, int causal, float* dk_part, float* dv_part,
                                      int nblk, int ksplit = 1, float* dq_part = nullptr) {
  using Cfg = Fp8Cfg<HD, BM, BN>;
  constexpr int kSmem = Cfg::smem_bytes_wgmma_kvtma;
  CUDA_CHECK(cudaFuncSetAttribute(
      fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, true, DET_HALF>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_mma_kvtma_kernel<HD, BM, BN, REGDQ, PREL, F16B, RCP, true, DET_HALF>
      <<<mg, THREADS, kSmem>>>(qmap, dmap, kmap, vmap, q8, qs, k8, ks, v8, vs, do8, dos,
                               delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv, scale, causal,
                               ksplit, nullptr, dk_part, dv_part, nblk, dq_part);
}
#endif

// O19：跨 warpgroup 归约版主 kernel（BM=128, 2 wg, 256 线程）的 smem 与 launcher。
template <int HD>
static constexpr int wg2_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int ASLD = HD + 16, PSLD = HD + 8, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  return (2 * BM * ASLD + 2 * BN * ASLD) + 2 * (BM / 2) * PSLD * 2 + (BN / 2) * PSLD * 2 +
         BM * DSS2 + (kNScale + 2 * BM * PSS) * (int)sizeof(float);
}

template <int HD>
static void launch_bwd_wg2(dim3 mg, const unsigned char* q8, const float* qs,
                           const unsigned char* k8, const float* ks,
                           const unsigned char* v8, const float* vs,
                           const unsigned char* do8, const float* dos,
                           const float* delta, const float* lse, float* dq_acc,
                           float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                           float scale, int causal, int ksplit) {
  constexpr int kSmem = wg2_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wg2_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wg2_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

// F6 第二步：双 warpgroup + wgmma（GEMM1/2）的 BM=128 主 kernel。仅 `-DFA_WGMMA` 构建可用。
#ifdef FA_WGMMA
template <int HD>
static constexpr int wgmma2_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  return 2 * QS_SZ + 2 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 1024;
}

template <int HD>
static void launch_bwd_wgmma2(dim3 mg, const unsigned char* q8, const float* qs,
                              const unsigned char* k8, const float* ks,
                              const unsigned char* v8, const float* vs,
                              const unsigned char* do8, const float* dos,
                              const float* delta, const float* lse, float* dq_acc,
                              float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                              float scale, int causal, int ksplit) {
  constexpr int kSmem = wgmma2_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

// O91（F6-step4，multi-warpgroup）：泛化 NWG 个 warpgroup（BM=NWG*64）；`--wg3` 用 NWG=3、BM=192。
template <int HD, int BM, int BN, int NWG>
static constexpr int wgmma_nw_smem_bytes() {
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  return 2 * QS_SZ + 2 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 1024;
}

template <int HD, int BM, int BN, int NWG>
static void launch_bwd_wgmma_nw(dim3 mg, const unsigned char* q8, const float* qs,
                                const unsigned char* k8, const float* ks,
                                const unsigned char* v8, const float* vs,
                                const unsigned char* do8, const float* dos,
                                const float* delta, const float* lse, float* dq_acc,
                                float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                float scale, int causal, int ksplit) {
  constexpr int kSmem = wgmma_nw_smem_bytes<HD, BM, BN, NWG>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_kernel<HD, BM, BN, NWG>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_kernel<HD, BM, BN, NWG><<<mg, NWG * 128, kSmem>>>(
      q8, qs, k8, ks, v8, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}
#endif  // FA_WGMMA

// O90（F6-step3）：wgmma2 + K/V 4D-TMA（K 双缓冲）的 launcher。需 `-DFA_WGMMA -DFA_TMA`。
#if defined(FA_WGMMA) && defined(FA_TMA)
template <int HD>
static constexpr int wgmma2tma_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  return 2 * QS_SZ + 3 * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + 64 + 1024;
}

template <int HD>
static void launch_bwd_wgmma2tma(dim3 mg, const CUtensorMap& kmap, const CUtensorMap& vmap,
                                 const unsigned char* q8, const float* qs,
                                 const unsigned char* k8, const float* ks,
                                 const unsigned char* v8, const float* vs,
                                 const unsigned char* do8, const float* dos,
                                 const float* delta, const float* lse, float* dq_acc,
                                 float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                 float scale, int causal, int ksplit) {
  (void)k8; (void)v8;   // K/V 由 4D-TMA 搬入
  constexpr int kSmem = wgmma2tma_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_tma_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_tma_kernel<HD, 128, 32><<<mg, 256, kSmem>>>(
      kmap, vmap, q8, qs, ks, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}

// O121（F3b 主体化）：O120 的 warp specialization 主 kernel 化——384 线程（2 compute WG +
//   1 producer WG）、K/V mbarrier ring、compute 侧命名 barrier。`--wg2ws` opt-in，默认关。
template <int HD, int KSTAGE = 2, int VSTAGE = 2>
static constexpr int wgmma2ws_smem_bytes() {
  constexpr int BM = 128, BN = 32;
  constexpr int PSLD = HD + 8, QTS = BM + 16, DSS2 = BN + 16, PSS = BN + 5;
  constexpr int kNScale = 3 * BM + 4 * BN;
  constexpr int QS_SZ = (BM / 8) * (HD / 128) * 1024, KS_SZ = (BN / 8) * (HD / 128) * 1024;
  constexpr int qp_bytes = (BM / 2) * PSLD * 2, kp_bytes = (BN / 2) * PSLD * 2;
  constexpr int NBAR = 2 * (KSTAGE + VSTAGE);
  return 2 * QS_SZ + (KSTAGE + VSTAGE) * KS_SZ + 2 * qp_bytes + kp_bytes + BM * DSS2 +
         (kNScale + 2 * BM * PSS) * (int)sizeof(float) + 2 * BN * QTS + NBAR * 8 + 1024;
}

template <int HD>
static void launch_bwd_wgmma2ws(dim3 mg, const CUtensorMap& kmap, const CUtensorMap& vmap,
                                const unsigned char* q8, const float* qs,
                                const unsigned char* k8, const float* ks,
                                const unsigned char* v8, const float* vs,
                                const unsigned char* do8, const float* dos,
                                const float* delta, const float* lse, float* dq_acc,
                                float* dk_acc, float* dv_acc, int S, int H, int Hkv,
                                float scale, int causal, int ksplit) {
  (void)k8; (void)v8;   // K/V 由 4D-TMA 搬入
  constexpr int kSmem = wgmma2ws_smem_bytes<HD>();
  CUDA_CHECK(cudaFuncSetAttribute(fa_bwd_fp8_wgmma2_ws_kernel<HD, 128, 32>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  fa_bwd_fp8_wgmma2_ws_kernel<HD, 128, 32><<<mg, 384, kSmem>>>(
      kmap, vmap, q8, qs, ks, vs, do8, dos, delta, lse, dq_acc, dk_acc, dv_acc, S, H, Hkv,
      scale, causal, ksplit);
}
#endif  // FA_WGMMA && FA_TMA

template <int HD>
static void launch_lse(dim3 lg, const unsigned char* q8, const float* qs,
                       const unsigned char* k8, const float* ks, float* lse, int S, int H,
                       int Hkv, float scale, int causal,
                       const int* cu_seqlens = nullptr) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel<HD>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  Cfg::lse_smem_bytes));
  lse_mma_kernel<HD><<<lg, THREADS, Cfg::lse_smem_bytes>>>(q8, qs, k8, ks, lse, S, H, Hkv,
                                                           scale, causal, cu_seqlens);
}

// O11：LSE 的镜像配对（+可选 cp.async 双缓冲）版本，仅 causal。
//   O54：`FULL=true` 时也服务非 causal（full）——一个 CTA 一个 m 块（lg.x=nblk），无镜像配对。
//   O58：模板参数化为 `<HD,PIPE,FULL,NTH,LBN_>`（对齐 device 侧；默认档 `NTH=THREADS,LBN_=64`
//   与原版逐位等价）。smem 按 `LBM_=(NTH/32)*16`、`LBN_` 现算。
template <int HD, int PIPE, bool FULL = false, int NTH = THREADS, int LBN_ = 64>
static void launch_lse_bal(dim3 lg, const unsigned char* q8, const float* qs,
                           const unsigned char* k8, const float* ks, float* lse, int S, int H,
                           int Hkv, float scale, const int* cu = nullptr,
                           float* lse_part = nullptr, int ksplit = 1,
                           long long merge_rows = -1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int ASLD = Cfg::ASLD;
  constexpr int LBM_ = (NTH / 32) * 16;
  constexpr int kSmem = LBM_ * ASLD + (PIPE ? 2 : 1) * LBN_ * ASLD +
                        (LBM_ + (PIPE ? 2 : 1) * LBN_) * (int)sizeof(float);
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal<HD, PIPE, FULL, NTH, LBN_>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  // O39：K 维 split —— grid.z 由 B 扩成 B*ksplit，部分结果写 lse_part 后由 merge kernel 汇总。
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)((size_t)B * ksplit));
  lse_mma_kernel_bal<HD, PIPE, FULL, NTH, LBN_><<<g, NTH, kSmem>>>(
      q8, qs, k8, ks, lse, S, H, Hkv, scale, cu, lse_part, ksplit);
  if (ksplit > 1) {
    const long long nrows = (merge_rows >= 0) ? merge_rows : (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}

// O9c：fp8 wgmma 版 LSE（SW128 + wgmma.m64n64k32），仅 `-DFA_WGMMA` 构建存在。
#ifdef FA_WGMMA
template <int HD, int PIPE>
static void launch_lse_bal_wgmma(dim3 lg, const unsigned char* q8, const float* qs,
                                 const unsigned char* k8, const float* ks, float* lse, int S,
                                 int H, int Hkv, float scale,
                                 const int* cu_seqlens = nullptr,
                                 const int* pt_b = nullptr, const int* pt_pair = nullptr,
                                 float* lse_part = nullptr, int ksplit = 1,
                                 long long merge_rows = -1) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = PIPE ? Cfg::lse_smem_bytes_balw1 : Cfg::lse_smem_bytes_balw0;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_wgmma<HD, PIPE>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  // O40：K 维 split（对齐 O38/O39）。紧凑网格（pt_pair 非空）时 grid.z 只编 `ksp`（b 由表给出），
  //   否则编 `(b,ksp)`。`ksplit==1` 时内核逐位退化为 O9c 原路径、不写 part、不 launch merge。
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, pt_pair ? (unsigned)ksplit : (unsigned)((size_t)B * ksplit));
  lse_mma_kernel_bal_wgmma<HD, PIPE><<<g, THREADS, kSmem>>>(q8, qs, k8, ks, lse, S, H, Hkv,
                                                             scale, cu_seqlens, pt_b, pt_pair,
                                                             lse_part, ksplit);
  if (ksplit > 1) {
    const long long nrows = (merge_rows >= 0) ? merge_rows : (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}
#endif

// O32：fp8 TMA 版 LSE（4D-TMA 载入 Q/K，单块），仅 `-DFA_WGMMA -DFA_TMA` 构建存在。
//   O70：模板加 `bool FULL`，`FULL=true` 时 grid.x 应是 `nblk`（一个 m 块一个 CTA，非因果）。
#if defined(FA_WGMMA) && defined(FA_TMA)
//   O72（第 166 轮）：加 `const int* cu`——varlen full D=128 复用它（描述符建在 packed 张量上、
//   行坐标由内核用 `cu_seqlens[b]` 定界）。定长调用不传 ⇒ 逐位不变。
template <int HD, int PIPE, bool FULL = false>
static void launch_lse_bal_tma(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                               const float* qs, const float* ks, float* lse, int S, int H,
                               int Hkv, float scale, const int* cu = nullptr) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE, FULL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  lse_mma_kernel_bal_tma<HD, PIPE, FULL><<<lg, THREADS, kSmem>>>(qmap, kmap, qs, ks, lse, nullptr,
                                                           S, H, Hkv, scale, 1, cu);
}

// O38：LSE 的 K 维 split + 二次归约（仅 D=128/causal/TMA）。`lg.z` 是 batch B；内部把
//   grid.z 扩成 `B*ksplit`，kernel 每个 (pair,ks) 只扫本 m 块的第 ks 个 K tile 切片，把
//   部分 (m,l) 写入 `lse_part`（[B*S*H][ksplit] 个 (m,l)），随后 merge kernel 汇总成 `lse`。
//   `ksplit==1` 时等于原 O32 路径（kernel 内逐位退化，不写 part、不 launch merge）。
//   好处：小 S / 低 H 时镜像配对后的 grid（=`ceil(nblk/2)*H`）常 < 132 SM，split 直接补满
//   并发槽、缩短「单 CTA 顺序扫 nblk+1 个 tile」的临界路径；大 S 已铺满一个波时收益趋零。
//   O103（第 197 轮）：模板加 `bool FULL`——full D=256 也走 TMA+split（此前只有 mma/cp.async
//   版）；FULL=false（causal 镜像配对）时与历史逐字相同。
template <int HD, int PIPE, bool FULL = false>
static void launch_lse_bal_tma_split(dim3 lg, const CUtensorMap& qmap, const CUtensorMap& kmap,
                                     const float* qs, const float* ks, float* lse,
                                     float* lse_part, int S, int H, int Hkv, float scale,
                                     int ksplit) {
  using Cfg = Fp8Cfg<HD, 64, 32>;
  constexpr int kSmem = Cfg::lse_smem_bytes_tma1;
  CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<HD, PIPE, FULL>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem));
  const int B = (int)lg.z;
  dim3 g(lg.x, lg.y, (unsigned)(B * ksplit));
  lse_mma_kernel_bal_tma<HD, PIPE, FULL><<<g, THREADS, kSmem>>>(qmap, kmap, qs, ks, lse, lse_part,
                                                          S, H, Hkv, scale, ksplit);
  if (ksplit > 1) {
    const long long nrows = (long long)B * S * H;
    const int th = 256;
    const long long bl = (nrows + th - 1) / th;
    lse_split_merge_kernel<<<(unsigned)bl, th>>>(lse_part, lse, nrows, ksplit);
  }
}
#endif

// =============================================================================
// VARLEN（变长 / cu_seqlens）自测入口（fp8，HD=128，causal/full，非 TMA 路径）
// =============================================================================
// 输入是 **packed** 布局 [T,H,D]（q/dO）与 [T,Hkv,D]（k/v），外加 `cu_seqlens.npy`
// （float32 保存，值即 token 前缀和，B+1 个）。kernel 侧用 `cu_seqlens[b]` 作 token 基址、
// `S` 参数传各序列最大长度 maxlen；每个 (b,h,mblk) 只处理自己序列内的 tile。
// ref 输出 `ref_dq/dk/dv.npy` 也是 packed 布局。FA/TE 不支持变长（本机版本），只对 fp32 ref。
// causal / 非 causal 均支持：causal 走镜像配对的 wgmma LSE，非 causal（各块工作量相同）走
// O1 的 mma LSE；非 TMA 路径（LSE 用 wgmma/mma，主 kernel Q/dO 用 cp.async）。
static int run_varlen(const std::string& dir, bool causal, int iters, bool compact = false,
                      bool lse_compact = false, int lse_split = 0, int mla8w = -1,
                      int mla_kvp = -1, int lseocc = 0, int lse8w = 0,
                      const std::string& dump = "", int det_ab = 0, int det_ksplit = 1,
                       int fuse_reduce = 1, int part_compact = 0, int qfuseflag = 1,
                       int dfuseflag = 1, int vksplit = -1, int mrev_flag = 1, int ovl_ql = -1, int ovl_cap = 0) {
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
  // O108（第 202 轮）：补齐 fp8 变长覆盖的最后一块 —— **D=256 变长**（与两文件版同源）。
  if (D != 128 && D != 256 && D != 512) {
    fprintf(stderr, "VARLEN 只做 HD=128/256/512；当前 %d\n", D);
    return 1;
  }
  int maxlen = 0;
  for (int b = 0; b < B; ++b) {
    int L = (int)cu_np.data[b + 1] - (int)cu_np.data[b];
    if (L > maxlen) maxlen = L;
  }
  const size_t nq = (size_t)T * H * D;       // packed q/dO/dq 长度
  const size_t nkv = (size_t)T * Hkv * D;    // packed k/v/dk/dv 长度
  const size_t rows_q = (size_t)T * H;
  const size_t rows_kv = (size_t)T * Hkv;
  const float scale = 1.0f / sqrtf((float)D);
  printf("case = %s\n", dir.c_str());
  printf("VARLEN: B=%d T=%d maxlen=%d H=%d Hkv=%d D=%d causal=%d\n", B, T, maxlen, H, Hkv, D,
         (int)causal);
  printf("cu_seqlens =");
  for (int b = 0; b <= B && b < 12; ++b) printf(" %d", (int)cu_np.data[b]);
  printf("%s\n", (B > 11) ? " ..." : "");

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos, *d_delta, *d_lse, *d_dq, *d_dk, *d_dv;
  float* d_lse_part = nullptr;   // O40：varlen LSE 的 K 维 split 部分结果（[T*H][ksplit] 个 (m,l)）
  int* d_cu;
  CUDA_CHECK(cudaMalloc(&d_q_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, nq));
  CUDA_CHECK(cudaMalloc(&d_k8, nkv));
  CUDA_CHECK(cudaMalloc(&d_v8, nkv));
  CUDA_CHECK(cudaMalloc(&d_do8, nq));
  CUDA_CHECK(cudaMalloc(&d_qs, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * 4));
  // O40：按最大 split=16 预留（与定长路径一致；实际只用到 auto 选出的 ksplit）。
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, nkv * 4));
  std::vector<int> cu(B + 1);
  for (int b = 0; b <= B; ++b) cu[b] = (int)cu_np.data[b];
  CUDA_CHECK(cudaMalloc(&d_cu, (B + 1) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_cu, cu.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));
  // 均衡分块表：把每个 b 的有效 m 块 (mblk, b) 按 (b,mblk) 主序写进一维表，主 kernel 用
  //   blockIdx.x 查表得到 (b, mblk)（见 fp8_mma_body 顶部注释）。只发有效 tile，消死 CTA。
  constexpr int BV = 64;
  std::vector<std::pair<int,int>> tiles;   // (mblk, b)
  for (int b = 0; b < B; ++b) {
    int nb = (cu[b + 1] - cu[b] + BV - 1) / BV;
    for (int mb = 0; mb < nb; ++mb) tiles.push_back({mb, b});
  }
  // 保持「b 主序、mblk 次序」以复用同序列 K/V 的 L2 局部性（按 mblk 排序会把不同
  // 序列交错，实测反而慢 —— 见文档）。
  const int total_mt = (int)tiles.size();
  std::vector<int> h_mtb(total_mt), h_mtm(total_mt);
  for (int i = 0; i < total_mt; ++i) { h_mtm[i] = tiles[i].first; h_mtb[i] = tiles[i].second; }
  int *d_mtb = nullptr, *d_mtm = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mtb, std::max(1, total_mt) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mtm, std::max(1, total_mt) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_mtb, h_mtb.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mtm, h_mtm.data(), total_mt * sizeof(int), cudaMemcpyHostToDevice));
  // LSE 的镜像对表：每个 b 的有效对 = ceil(nblk_b/2)（nblk_b 按 LBM=64），按工作量降序。
  constexpr int LBMH = 64;
  std::vector<std::pair<int,int>> pairs;   // (pair, b)
  for (int b = 0; b < B; ++b) {
    int nb = (cu[b + 1] - cu[b] + LBMH - 1) / LBMH;
    int np = (nb + 1) / 2;
    for (int p = 0; p < np; ++p) pairs.push_back({p, b});
  }
  std::stable_sort(pairs.begin(), pairs.end(),
                   [&](const std::pair<int,int>& x, const std::pair<int,int>& y) {
                     int nx = (cu[x.second + 1] - cu[x.second] + LBMH - 1) / LBMH;
                     int ny = (cu[y.second + 1] - cu[y.second] + LBMH - 1) / LBMH;
                     return nx > ny;
                   });
  const int total_pairs = (int)pairs.size();
  std::vector<int> h_ptb(total_pairs), h_ptp(total_pairs);
  for (int i = 0; i < total_pairs; ++i) { h_ptp[i] = pairs[i].first; h_ptb[i] = pairs[i].second; }
  int *d_ptb = nullptr, *d_ptp = nullptr;
  CUDA_CHECK(cudaMalloc(&d_ptb, std::max(1, total_pairs) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_ptp, std::max(1, total_pairs) * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_ptb, h_ptb.data(), total_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_ptp, h_ptp.data(), total_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  // delta = rowsum(dO∘O) 需要前向输出 O（fp32 ref_o，与 dO 同 dtype/布局）。
  CUDA_CHECK(cudaMalloc(&d_o_f, nq * 4));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  // ksplit / REGDQ 自动档：与定长路径同公式（O29），用 maxlen 作为串长。
  //   D==128：S>=2048 → 8192，否则 max(2048, 4*base_grid)；
  //   D==512（MLA）：target = len/2（O29 标定），regdq 恒关（HD=512 的 dQ 一次铺不满 N）。
  //   **O107（第 201 轮）**：causal 变长的 O29 档从未复核（O96-O100 只审了 full），
  //   D=128 在 `base_grid > 8192`（大 H·B / 长序列）时被欠切到 k=1、D=512 的 `maxlen/2`
  //   也被欠切。见下面 `if (causal …)` 分支（只改调度，数值逐位不变）。
  constexpr int BM = 64;
  const long base_grid = (long)((maxlen + BM - 1) / BM) * H * B;
  const long target_ctas = (D == 128)
                               ? ((maxlen >= 2048) ? 8192L : std::max(2048L, 4L * base_grid))
                               : (long)(maxlen / 2);
  long kk = target_ctas / base_grid; if (kk < 1) kk = 1; if (kk > 16) kk = 16;
  long kp = 1; while (kp * 2 <= kk) kp *= 2;
  long auto_k = kp;
  // O98（第 192 轮）：**full（非 causal）变长的 ksplit 重标定**——把 O96/O97 的定长 full
  //   审计推广到 `run_varlen`。原自动档沿用 O29 的 **causal** 标定 target（D=128 用 8192、
  //   D=256/512 用 `maxlen/2`），对 full 过切/欠切；且 `target/base` 含**短序列的早退死
  //   CTA**（`base_grid` 按 `maxlen` 计）⇒ 名义网格被高估。本轮对 dump 的全部 6 个 varlen
  //   full shape 做 k∈[1,16] 全扫（docs/03 §120）：
  //   · **D=128**：O98 取「`max(kp,3)`」下限 3（当时测 b4_t3840 最优 4）；**O100（第 194 轮）
  //     在干净机器上重复实测**（k∈[2,5]×3 次、150 iters）——4 个 shape 的最优**一致为 k=3**
  //     （b4_t4096 1.114 vs k4 1.128、b4_t3840 1.387 vs k4 1.391、b5 2.690、b8 1.129），
  //     O98 的「b4_t3840 k3 反慢 3.9%」证实为当轮噪声/残留进程。故 D=128 full 变长**直接取
  //     k=3**（按 nblk 封顶）——full 无三角偏斜，k 越大只增 Q/dO 重读，实测 3 即最优点。
  //   · **D=512（MLA，1 CTA/SM→132 槽）**：按 O97 的 132 槽波对齐（b1_t512 最优 k=8、
  //     b3_t1792 k=11 恰整数波），实测 +8.6%/+1.8%。
  //   · **D=256**（无 varlen dump，沿用 O97 定长规则：`maxlen≥2048` 给足并发、否则波对齐）。
  //   `--ksplit=K` 显式给出时不覆盖。
  if (!causal) {
    if (D == 128) {
      const long nblk32 = (long)((maxlen + 31) / 32);  // main kernel BN=32
      auto_k = std::min(3L, nblk32);
      if (auto_k < 1) auto_k = 1;
    } else {
      const long SLOTS = (D == 512) ? 132L : 264L;
      const long KMAX = (D == 512) ? 16L : 12L;
      if (D == 256 && maxlen >= 2048) {
        // O113（第 207 轮）：**D=256 full 变长的 ksplit 重标定**（与两文件 `fa_bwd_fp8_main.cu`
        //   同源逐字）。O98 只给 D=128 标了 full 变长、D=256 沿用 O97 的定长公式，实测该档
        //   大 base 欠切到 4、base=1024 过切到 8。5 个新 dump shape 全扫 k∈[1,16]：最优一致
        //   k=6（每 CTA 约 `nblk/6≈5` 个 K tile），规则 `max(6, 2048/base)`、cap KMAX。
        long k = 2048L / base_grid;
        if (k < 6) k = 6;
        if (k > KMAX) k = KMAX;
        auto_k = k;
      } else {
        long best_k = 1, best_waste = -1;
        for (long k = 1; k <= KMAX; ++k) {
          const long g = base_grid * k;
          const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
          if (best_waste < 0 || waste < best_waste) { best_waste = waste; best_k = k; }
        }
        if (D == 512 && best_k < 2) best_k = 2;  // 长 K 循环仍偏好 ≥2 份并发（同 O97）
        auto_k = best_k;
      }
    }
  }
  // O99（第 193 轮）：**causal D=256 的 ksplit 重标定**（定长规则同源，用 `maxlen` 当串长）。
  //   定长 causal D=256 的 O29 `target=S/2` 在小/中 S 严重欠切（见定长路径注释 / docs/03 §121），
  //   变长沿用同一错；改为 `k = clamp(2*maxlen/base,1,16)` 再按 `nblk` 封顶。
  // O108（第 202 轮）：**D=256 变长首次实装并单独标定**（同两文件版）——O99 的 `2*maxlen/base`
  //   用名义网格、大 H×B 时被早退死 CTA 高估 ⇒ 欠切；实测共同点「每 CTA 约 4 个 K tile」
  //   （`nblk/k≈4`）⇒ 再抬下限 `nblk/4`。只改跨 CTA atomicAdd 次序、数值逐位不变。
  if (causal && D == 256) {
    const long nblk256 = (long)((maxlen + BM - 1) / BM);
    long k = (2L * maxlen) / base_grid;
    const long kq = nblk256 / 4;
    if (kq > k) k = kq;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    if (k > nblk256) k = nblk256;
    auto_k = k;
  }
  // O107（第 201 轮）：**causal 变长的 ksplit 重标定（D=128 / D=512）**——同两文件版
  //   `fa_bwd_fp8_main.cu`（device 逐字同源；此处为 host 自动档，两文件须一致）。
  //   · **D=128**：O29 的绝对 target 在 `base_grid > target` 时欠切到 k=1，抬下限到 3
  //     （b5_t3968 1.10× / b8_t2904 1.06×；小 base 的 k=16 仍最优，不受影响）。
  //   · **D=512**：`target=maxlen/2` 欠切，改 `target=maxlen`、cap=8、下限 2 ⇒ b3/b1 均取 8
  //     （1.10×/1.06×）。只改调度，数值逐位不变。
  if (causal && D == 128) {
    if (auto_k < 3) auto_k = 3;   // 大 H·B（base_grid>target）时 O29 欠切，抬下限（不按 nblk 封顶——小 grid 靠加 k 填波）
  } else if (causal && D == 512) {
    long k = 1;
    while (k * 2 <= maxlen / base_grid && k < 8) k *= 2;
    if (k < 2) k = 2;
    auto_k = k;
  }
  // O97：`--ksplit=K` 也可用于 varlen（同 binary A/B；K>=1 直接覆盖自动档）。默认 -1 自动。
  const int ksplit = (vksplit >= 1) ? vksplit : (int)auto_k;
  const bool use_regdq = (D == 128) && ((long)(maxlen / 32) / (causal ? 2 : 1) / ksplit >= 4);

  // O106（第 200 轮）：把 O89/O105 的**per-head LPT m 块反转**（`mblk = nblk-1-mt`）推广到
  //   **变长（varlen）causal 主 kernel**。同一序列内 K 循环长 ∝ `ceil(len_b/BM)`，causal 下
  //   m 越大越贵；默认非紧凑网格直接把 `mt = blockIdx.x/ksplit` 当 mblk，是 LPT 的反面（便宜块
  //   先派发、贵块压尾波）。LSE 早已按降序工作量排序（镜像对表），主 kernel 一直没接。只改
  //   「哪个 CTA 算哪个 m 块」，dK/dV 仍是可交换的跨 CTA `atomicAdd` ⇒ 数值仅在 fp8 噪声内。
  //   仅 causal、D==128/512、`nblk_max>=16` 生效（非紧凑网格；`--compact` 走 mt 表本身，不叠加）；
  //   `--mrev=0` 回退历史锯齿序。
  int* d_mrev_v = nullptr;
  const int nblk_mv = (maxlen + BM - 1) / BM;
  //   形状门控（关键，同 O104 hswap256）：只在**名义网格被短序列 padding**（`total_mt < nblk_max*B`，
  //   即存在早退死 CTA、序列长度不齐）时为正——4 个 D=128 shape 实测 b4_t3840 +1.6% / b8_t2904
  //   +1.5% / b5_t3968 +0.6%（均不齐、maxlen=2048），而**等长 b4_t4096（无 padding）为 −1.1%**
  //   （纯 LPT 打乱同序列相邻 m 的 K/V 微局部性、收益不抵）；D=512 b3_t1792 +7.1%。故门控在
  //   「有 padding」；等长档逐值不变。
  //   O108：门控扩到 D=256（与定长 D=256 的 hswap256/mrev 同源）。
  const bool mrev_v_elig = (mrev_flag != 0) && causal && (D == 128 || D == 256 || D == 512) &&
                           nblk_mv >= 16 && (long)total_mt < (long)nblk_mv * B;
  if (mrev_v_elig) {
    std::vector<int> hrev(nblk_mv);
    for (int i = 0; i < nblk_mv; ++i) hrev[i] = nblk_mv - 1 - i;
    CUDA_CHECK(cudaMalloc(&d_mrev_v, nblk_mv * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mrev_v, hrev.data(), nblk_mv * sizeof(int), cudaMemcpyHostToDevice));
    printf("O106: varlen mrev on (nblk=%d, LPT expensive-first)\n", nblk_mv);
  } else if (mrev_flag) {
    printf("O106: varlen mrev requested but ignored (need causal & D==128/256/512 & nblk>=16)\n");
  }

  // O40：varlen LSE 的 K 维 split auto（D=128 目标 `grid*split≈2048`、cap 8；D=512 `≈256`、
  //   cap 16，与定长 O38/O39 同标定）；`--lsesplit=N`（>0）直接指定。再按最大序列的 tile 数封顶。
  const int nblk0 = (maxlen + LBM - 1) / LBM;
  // O58：把 fp16/bf16 的 O56（8-warp）/O57（2 CTA/SM）LSE 几何同构到 fp8。causal 默认走 cfg6
  //   （PIPE1/LBN16/NTH=128/LBM=64，smem 99.8KB → 2 CTA/SM）；full 保持 O54 旧默认，`--lseocc=5/6`
  //   或 `--lse8w=1` 可选（与 O56/O57 一致：full 上该杠杆为混合/负，未默认）。`--lse8w` 优先 8-warp。
  const int lse_nblk8 = (maxlen + 127) / 128;
  const bool lse8w_use = (lse8w != 0) && (D == 512);
  const bool causal_cfg6 = causal && (D == 512) &&
                           (lseocc == 5 || lseocc == 6 || (lseocc == 0 && !lse8w_use));
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    long base;
    if (D == 128 && lse_compact) base = (long)total_pairs * H;
    else if (D == 512 && !causal) base = (long)nblk0 * H * B;   // O54：full 一个 CTA 一个 m 块
    else base = (long)((nblk0 + 1) / 2) * H * B;
    // O58：causal 的 2 CTA/SM 几何（cfg5/6）tile 更细（fp8 LBN=16/HD=512 时 tile 更多），split
    //   目标提到 1024（b1→8 cap、b3→16，均为 A/B 最优附近）；旧默认/8-warp 维持 256。
    const int target = (D == 512) ? (causal ? (causal_cfg6 ? 1024 : 256) : 768) : 2048;
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && base * (sp * 2) <= target) sp *= 2;
    while (sp > nblk0 && sp > 1) sp >>= 1;
    lse_split_eff = sp;
  }
  printf("O40: varlen lse k-split = %d (base=%ld)\n", lse_split_eff,
         (D == 128 && lse_compact) ? (long)total_pairs * H : (long)((nblk0 + 1) / 2) * H * B);

  // O111（第 205 轮）：把 O109/O110 的「量化分相 + LSE 跨 stream 重叠 + phase1 栅格封顶」
  //   推广到**变长**。变长 `run_all` 此前始终走合并量化（`quantize_zero_delta_warp_kernel`）
  //   → 串行 LSE；而 LSE（SM bound）与 phase1（DRAM bound）资源互补。
  //   与定长同源：phase0 量化 Q/K（LSE 就绪），phase1（dO+delta/V/清零）放 aux stream 与 LSE
  //   重叠，主 kernel 前等 aux。逐行例程与合并版**逐字相同** ⇒ 数值**逐位不变**（device 一行
  //   未改，复用 O109 的 `quantize_zero_delta_phase_kernel`）。门控：融合路径
  //   （qfuseflag && dfuse）；D=256/512 需 `rows_q>=2048`（小 shape 分相 launch 开销盖过被藏的
  //   phase1）；`--ovlql=0` 关、`-1/1`=auto/开；`--ovlcap` 同 O110（0=auto、>0 显式、<0 不封顶）。
  const bool ql_overlap_v = (ovl_ql != 0) && qfuseflag && (dfuseflag != 0) &&
                            ((D == 128) || (rows_q >= 2048));
  cudaStream_t sQuantAuxV = nullptr;
  cudaEvent_t ql_e0v = nullptr, ql_e1v = nullptr;
  auto quant_phase_v = [&](int ph, cudaStream_t st) {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long ntask = (ph == 0) ? (rq + rkv) : (2 * rq + 3 * rkv);
    int grid = (int)std::min<long long>((ntask + 3) / 4, 1048576);
    if (ph == 1) {
      long long cap;
      if (ovl_cap < 0) cap = grid;
      else if (ovl_cap > 0) cap = ovl_cap;
      else cap = std::min<long long>(std::max<long long>(132, ntask / 64), 4096);
      grid = (int)std::min<long long>(grid, cap);
    }
    if (D == 128)
      quantize_zero_delta_phase_kernel<4><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq, d_dk, d_dv, rq, rkv, ph);
    else if (D == 256)
      quantize_zero_delta_phase_kernel<8><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq, d_dk, d_dv, rq, rkv, ph);
    else
      quantize_zero_delta_phase_kernel<16><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq, d_dk, d_dv, rq, rkv, ph);
  };
  if (ql_overlap_v) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&sQuantAuxV, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ql_e0v, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ql_e1v, cudaEventDisableTiming));
    printf("O111: varlen quant-phase overlap ON (Q/K -> LSE(default) || dO+V+zero(aux)), D=%d\n", D);
  } else if (ovl_ql == 1) {
    printf("O111: varlen overlap requested but conditions unmet (D=%d qfuse=%d dfuse=%d rows_q=%zu) -> serial\n",
           D, qfuseflag, dfuseflag, (size_t)rows_q);
  }

  auto run_all = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const int gq = (int)std::min<long long>((rq + 3) / 4, 65535);
    const int gkv = (int)std::min<long long>((rkv + 3) / 4, 65535);
    // VPT = D/32：D=128→4、D=512→16（与定长路径一致；写错会把行距当 128 ⇒ 整行错位）。
    // O64：默认把 4 次量化 + 3 次清零融合成 1 个 launch（`qfuseflag=0` 退回旧路径做 A/B）。
    // O66：`dfuseflag`（默认 1）时把 delta 也算进 dO 的量化任务（省一次 delta launch）。
    const bool dfuse = (dfuseflag != 0);
    // O111：分相重叠仅在融合路径、且本次调用 qfuseflag 为真时生效（`time_all(false)` 走 unfused）。
    const bool ql_ov = ql_overlap_v && qfuseflag;
    if (ql_ov) {
      // Q/K 先量化（LSE 就绪）→ 下面 default stream 跑 LSE，同时 phase1 在 aux stream 上
      // 做 dO+delta / V / 清零；主 kernel 前等 aux 完成（见 LSE 块之后的 `ql_e1v` wait）。
      quant_phase_v(0, nullptr);
      CUDA_CHECK(cudaEventRecord(ql_e0v));
      CUDA_CHECK(cudaStreamWaitEvent(sQuantAuxV, ql_e0v, 0));
      quant_phase_v(1, sQuantAuxV);
    } else if (qfuseflag) {
      const long long total = 3 * rq + 4 * rkv;
      const int g = (int)std::min<long long>((total + 3) / 4, 1048576);
      if (dfuse) {
        if (D == 128)
          quantize_zero_delta_warp_kernel<4><<<g, 128>>>(
              d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks,
              d_vs, d_dos, d_dq, d_dk, d_dv, rq, rkv);
        else if (D == 256)  // O108：VPT = D/32 = 8（与定长 D=256 同）
          quantize_zero_delta_warp_kernel<8><<<g, 128>>>(
              d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks,
              d_vs, d_dos, d_dq, d_dk, d_dv, rq, rkv);
        else
          quantize_zero_delta_warp_kernel<16><<<g, 128>>>(
              d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks,
              d_vs, d_dos, d_dq, d_dk, d_dv, rq, rkv);
      } else if (D == 128)
        quantize_zero_warp_kernel<4><<<g, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                 d_do8, d_qs, d_ks, d_vs, d_dos, d_dq, d_dk,
                                                 d_dv, rq, rkv);
      else if (D == 256)
        quantize_zero_warp_kernel<8><<<g, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq, d_dk,
                                                  d_dv, rq, rkv);
      else
        quantize_zero_warp_kernel<16><<<g, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq, d_dk,
                                                  d_dv, rq, rkv);
    } else {
      if (D == 128) {
        quantize_row_warp_kernel<4, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
        quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
        quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
        quantize_row_warp_kernel<4, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
      } else if (D == 256) {
        quantize_row_warp_kernel<8, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
        quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
        quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
        quantize_row_warp_kernel<8, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
      } else {
        quantize_row_warp_kernel<16, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
        quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
        quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
        quantize_row_warp_kernel<16, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
      }
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    }
    // LSE：grid.x 按 maxlen，逐 b 由 cu_seqlens 定界。
    //   causal → 镜像配对 + cp.async 的 wgmma 版（工作量随 mblk 递增，需均衡）；
    //   非 causal → 各 m 块工作量恒为 nblk 个 tile，本已均衡，走 O1 的 mma `lse_mma_kernel`。
    const int nblk = (maxlen + LBM - 1) / LBM;
    // O72（第 166 轮）：varlen full D=128 的 LSE 走 4D-TMA（对齐定长 O70）。TMA 描述符建在
    //   packed 张量上（dims={D,T,Hkv,1}，batch 维恒 0），内核对每个 b 用 `cu_seqlens[b]` 定界。
#if defined(FA_WGMMA) && defined(FA_TMA)
    const bool lse_tma_v = (g_lse_tma_varlen != 0) && g_lse_full_opt && (D == 128) && !causal;
    CUtensorMap qmap_v{}, kmap_v{};
    if (lse_tma_v) {
      qmap_v = make_lse_map_fp8(d_q8, H, T, D, 1);
      kmap_v = make_lse_map_fp8(d_k8, Hkv, T, D, 1);
    }
#endif
    if (causal) {
      dim3 lg((nblk + 1) / 2, H, B);
      // D==128 走 wgmma 版（SW128 + wgmma）；D==512（MLA）只有 mma 版（HD>128 无 SW128 快路）。
      // O42：纯 `sm_90`（无 `-DFA_WGMMA`）构建下没有 wgmma 版 ⇒ D==128 退回 mma 镜像配对版
      //   （此前无条件调用导致 fp8 main 的 `-arch=sm_90` 构建失败，fp16/bf16 无此问题）。
#ifdef FA_WGMMA
      if (D == 128 && lse_compact)
        launch_lse_bal_wgmma<128, 1>(dim3(total_pairs, H, 1), d_q8, d_qs, d_k8, d_ks, d_lse,
                                     maxlen, H, Hkv, scale, d_cu, d_ptb, d_ptp,
                                     d_lse_part, lse_split_eff, (long long)rows_q);
      else if (D == 128)
        launch_lse_bal_wgmma<128, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                     d_cu, nullptr, nullptr, d_lse_part, lse_split_eff,
                                     (long long)rows_q);
      else
#endif
      if (D == 128)
        launch_lse_bal<128, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
      else if (D == 512 && (lseocc == 5 || lseocc == 6 || causal_cfg6)) {
        // O58：causal MLA varlen 的 LSE 默认走 cfg6（2 CTA/SM，镜像配对版）；`--lseocc=5` 可选
        //   PIPE0/LBN32。grid 与默认相同（LBM=64 的镜像对）。
        if (lseocc == 5)
          launch_lse_bal<512, 0, false, 128, 32>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                                 Hkv, scale, d_cu, d_lse_part, lse_split_eff,
                                                 (long long)rows_q);
        else
          launch_lse_bal<512, 1, false, 128, 16>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                                 Hkv, scale, d_cu, d_lse_part, lse_split_eff,
                                                 (long long)rows_q);
      } else if (D == 512 && lse8w_use) {
        launch_lse_bal<512, 1, false, 256, 32>(dim3((lse_nblk8 + 1) / 2, H, B), d_q8, d_qs, d_k8,
                                               d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                                               d_lse_part, lse_split_eff, (long long)rows_q);
      } else if (D == 256)
        // O108：D=256 变长 causal LSE（通用 mma 镜像配对版，同定长非 TMA 档）。
        launch_lse_bal<256, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
      else
        launch_lse_bal<512, 1>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, d_cu,
                               d_lse_part, lse_split_eff, (long long)rows_q);
    } else {
      dim3 lg(nblk, H, B);
      if (D == 128) {
        // O68（第 162 轮）：varlen full D=128 的 LSE 同定长，改走 O54 均衡版（cp.async 双缓冲）。
        // O72（第 166 轮）：再优先走定长 O70 同款的 4D-TMA（`--lsetmavarlen=0` 退回 cp.async 版）。
        bool did_tma_v = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma_v) {
          launch_lse_bal_tma<128, 1, true>(lg, qmap_v, kmap_v, d_qs, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu);
          did_tma_v = true;
        }
#endif
        if (!did_tma_v && g_lse_full_opt)
          launch_lse_bal<128, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                       scale, d_cu, nullptr, 1);
        else if (!did_tma_v)
          launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
      } else if (D == 256)
        // O108：D=256 变长 full LSE（通用 mma 均衡版 FULL=true，同定长非 TMA 档）。
        launch_lse_bal<256, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                     d_cu, d_lse_part, lse_split_eff, (long long)rows_q);
      else if (lseocc == 5 || lseocc == 6)
        // O58：full MLA varlen 的 LSE「2 CTA/SM」几何（O57 的 fp8 同构；默认仍是 O54 旧路）。
        if (lseocc == 5)
          launch_lse_bal<512, 0, true, 128, 32>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                                scale, d_cu, d_lse_part, lse_split_eff,
                                                (long long)rows_q);
        else
          launch_lse_bal<512, 1, true, 128, 16>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                                scale, d_cu, d_lse_part, lse_split_eff,
                                                (long long)rows_q);
      else if (lse8w_use)
        launch_lse_bal<512, 1, true, 256, 32>(dim3(lse_nblk8, H, B), d_q8, d_qs, d_k8, d_ks,
                                              d_lse, maxlen, H, Hkv, scale, d_cu, d_lse_part,
                                              lse_split_eff, (long long)rows_q);
      else if (lse_split_eff > 1)
        // O54：full MLA varlen 的 LSE 走 K 维 split（bal 的 FULL 版）。
        launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                     d_cu, d_lse_part, lse_split_eff, (long long)rows_q);
      else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0, d_cu);
    }
    // O111：LSE 已发在 default stream，此刻再等 aux 的 phase1（dO/delta/V/清零）——主 kernel
    //   读 v8/do8/delta、写 dq/dk/dv，必须等 phase1。事件使两条 stream 在此汇合。
    if (ql_ov) {
      CUDA_CHECK(cudaEventRecord(ql_e1v, sQuantAuxV));
      CUDA_CHECK(cudaStreamWaitEvent(nullptr, ql_e1v, 0));
    }
    const int d_rows = (int)rows_q;
    const int d_wpb = THREADS / 32;
    const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
    // O66：融合路径下 delta 已在 quant kernel 内算好，跳过独立的 delta launch。
    if (!(qfuseflag && dfuse)) {
      if (D == 128)
        delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
      else if (D == 256)  // O108：D=256 变长的 delta（VPT=8 布局）
        delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
      else
        delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    }
    // 均衡分块：`--compact` 走紧凑一维 m-tile 网格（grid.z=1，由 mt_b/mt_m 查表解出 (b, mblk)）；
    //   默认走旧的 maxlen 网格（含早退死 CTA，实测反而更快，见 docs/03 §40）。
    dim3 mg = compact ? dim3(total_mt * ksplit, H, 1)
                      : dim3((maxlen + BM - 1) / BM * ksplit, H, B);
    // O106：非紧凑 varlen 网格把 O89 的 m 块反转表当作 `mt_m`（`mt_b=null` 时 b=blockIdx.z）。
    const int* mtb = compact ? d_mtb : nullptr;
    const int* mtm = compact ? d_mtm : d_mrev_v;
    if (D == 512) {
      // MLA：非 wgmma / 非 regdq（与定长 D=512 路径一致）。
      // O52：把定长路径的 O47（8-warp/256 线程）+ O51（K/V cp.async 回填流水）搬进 varlen MLA。
      //   -1=自动（默认 8-warp + kvpipe，与定长一致）；`--mla8w=0/1`、`--mlakvp=0/1` 强制 A/B。
      const bool w8 = (mla8w != 0);
      const bool kvp = w8 && (mla_kvp != 0);
      if (kvp)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
      else if (w8)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
    } else if (D == 256)
      // O108：D=256 变长主 kernel（wgmma SW128 + cp.async，同定长默认档；REGDQ=false）。
      launch_bwd_main<256, 64, 32, false, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
    else if (use_regdq)
      launch_bwd_main<128, 64, 32, true, true, true, true, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
    else
      launch_bwd_main<128, 64, 32, false, true, true, true, true>(
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
          d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtb, mtm);
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
    flops += 4.0 * H * L * L * D;  // bwd ≈ 2×fwd，因果只算一半再用系数 2 近似
  }
  printf("[timing] VARLEN total %.4f ms  %.2f TFLOPS (sum_b 4HL^2D)\n", ms,
         flops / (ms * 1e-3) / 1e12);
  // ---- O64 A/B：varlen 的融合 quant+zero 同 binary 端到端对比。----
  {
    auto time_all = [&](bool f) {
      qfuseflag = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    qfuseflag = 1;
    printf("[O64 A/B] VARLEN total fused(1 launch) %.4f ms | unfused(4 quant + 3 memset) %.4f ms "
           "(%.4fx)\n", t_f, t_u, t_u / t_f);
  }
  if (compact)
    printf("grid main = %d x %d x %d (compact, ksplit=%d, total_mt=%d, old-grid=%d) | use_regdq=%d | T=%d\n",
           total_mt * ksplit, H, 1, ksplit, total_mt, (maxlen + BM - 1) / BM * B, (int)use_regdq, T);
  else
    printf("grid main = %d x %d x %d (nocompact, ksplit=%d, total_mt=%d) | use_regdq=%d | T=%d\n",
           (maxlen + BM - 1) / BM * ksplit, H, B, ksplit, total_mt, (int)use_regdq, T);

  // ---- P3-4i A/B（D=128 varlen，`--det` / `--detk=N`）：把 P3-4e/f 的确定性 dK/dV/dQ 扩到
  //      变长。partial 布局沿用定长式、但 `S=maxlen`、`nblk=nblk_max`；归约改用
  //      `dkv_reduce_varlen_kernel`（按 `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、输出按
  //      packed token 定位）。同 session 计时 + 跑两遍 DET 验逐位可复现，再与 atomic 比
  //      max|diff|。强制 `REGDQ=true`（ksplit>1 时 dQ 走 partial 才能确定）。----
#ifdef FA_WGMMA
  constexpr bool kWgmVarlen = true;   // varlen D=128 构建恒为 `-DFA_WGMMA`，默认主 kernel 走 WGMMA
#else
  constexpr bool kWgmVarlen = false;
#endif
  if (det_ab && D == 128) {
    const int nblk_max = (maxlen + BM - 1) / BM;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_max * maxlen * D;   // 旧（maxlen-strided）上界
    // P3-4o：compact per-sequence offset（行前缀和，行 = 一个 (h,mblk,jg) 的 HD 向量）。
    std::vector<int> part_base_h(B + 1, 0);
    for (int b = 0; b < B; ++b) {
      const int len_b = cu[b + 1] - cu[b];
      const int nblk_b = (len_b + BM - 1) / BM;
      part_base_h[b + 1] = part_base_h[b] + H * nblk_b * len_b;
    }
    const size_t part_elems_cmp = (size_t)part_base_h[B] * D;
    int* d_part_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_part_base, (B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_part_base, part_base_h.data(), (B + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const int* pb = part_compact ? d_part_base : nullptr;   // 空 ⇒ 旧 maxlen-strided 布局（默认）
    const size_t dqpart_elems = (size_t)T * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    printf("[P3-4o] varlen D=128 partial: compact=%.1fMB (%.0f%% of old %.1fMB) layout=%s\n",
           part_elems_cmp * 4 / 1e6, 100.0 * (double)part_elems_cmp / (double)part_elems,
           part_elems * 4 / 1e6, pb ? "compact" : "maxlen-strided");
    auto run_at = [&](int k) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      launch_bwd_main<128, 64, 32, true, kWgmVarlen, true, true, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, k, d_cu, nullptr, nullptr);
    };
    auto run_dt_p = [&](int k, const int* p) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));   // ksplit==1 时 dQ 用无竞争 red_add2
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<128, 64, 32, true, true, true, true, THREADS, WN, kWgmVarlen>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_max, d_dq_part,
          d_cu, nullptr, nullptr, p);
      // P3-4n：dkv+dq 融合归约（默认）vs 两次 launch（`--nofusered`）。
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (k > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, k, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, p);
        if (k > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, k);
        }
      }
    };
    auto run_dt = [&](int k) { run_dt_p(k, pb); };
    auto time_fn3 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f;
    time_fn3([&] { run_at(ks); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    time_fn3([&] { run_dt(ks); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    auto mad3 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4i A/B] varlen ksplit=%d | atomic %.4f ms | DET(partial+reduce) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad3(dq1, dq2), mad3(dk1, dk2), mad3(dv1, dv2),
           mad3(dq1, aq), mad3(dk1, ak), mad3(dv1, av));
    // ---- F4-b（第 134 轮）：把 F4 第一步的「fp16 partial + O62 写扇区化」从定长扩到 varlen。
    //      同 binary、只把 partial 存储从 fp32 换 fp16（`DET_HALF`）并让二次归约走 `P16`
    //      置换读（求和集合/次序不变）。两档 partial 应逐位可复现，数值只差 fp16 舍入。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth_p = [&](int k, const int* p) {
      dim3 g(nblk_max * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<128, 64, 32, true, true, true, true, THREADS, WN, kWgmVarlen, false,
                          true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                d_lse, d_dq, d_dk, d_dv, maxlen, H, Hkv, scale, (int)causal, k,
                                reinterpret_cast<float*>(d_dk_part_h),
                                reinterpret_cast<float*>(d_dv_part_h), nblk_max, d_dq_part, d_cu,
                                nullptr, nullptr, p);
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (k > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<128, 64, true><<<dkv_blocks + dq_blocks, 128>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (k > 1) ? d_dq_part : nullptr, d_dk,
            d_dv, d_dq, d_cu, H, Hkv, nblk_max, maxlen, (int)causal, k, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<128, 64, true><<<rg, 128>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk, d_dv, d_cu, H, Hkv, nblk_max,
            maxlen, (int)causal, p);
        if (k > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, k);
        }
      }
    };
    auto run_dth = [&](int k) { run_dth_p(k, pb); };
    float ms_dth = 0.f;
    time_fn3([&] { run_dth(ks); }, &ms_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] varlen ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, ms_dt, ms_dth, ms_dt / ms_dth, mad3(hdk1, hdk2), mad3(hdv1, hdv2),
           mad3(hdk1, dk1), mad3(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    // ---- P3-4n A/B：varlen 二次归约 分离 vs 融合（只测 reduce；partial 已由上面 run_dt 预置）。
    {
      auto only_sep = [&]() {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, pb);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq, T, H, ks);
        }
      };
      auto only_fus = [&]() {
        const int dkv_blocks = B * Hkv * maxlen, dq_blocks = (ks > 1) ? T * H : 0;
        dkv_dq_reduce_varlen_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, ks, dkv_blocks, pb);
      };
      float t_rs = 0.f, t_rf = 0.f;
      time_fn3(only_sep, &t_rs);
      std::vector<float> rsdk(nkv), rsdv(nkv), rsdq(nq);
      CUDA_CHECK(cudaMemcpy(rsdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_fn3(only_fus, &t_rf);
      std::vector<float> rfdk(nkv), rfdv(nkv), rfdq(nq);
      CUDA_CHECK(cudaMemcpy(rfdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4n A/B] varlen reduce separate(2 launch) %.4f ms | fused(1 launch) %.4f ms "
             "(%.3fx) | fused-vs-sep bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             t_rs, t_rf, t_rs / t_rf, mad3(rfdq, rsdq), mad3(rfdk, rsdk), mad3(rfdv, rsdv));
    }
    // ---- P3-4o A/B：同 binary 只改 partial 布局（compact vs maxlen-strided）。
    {
      auto time_full = [&](const int* p, float* out_ms) {
        auto fn = [&]() { run_dt_p(ks, p); };
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float t_cmp = 0.f, t_leg = 0.f;
      time_full(d_part_base, &t_cmp);
      std::vector<float> cdk(nkv), cdv(nkv), cdq(nq);
      CUDA_CHECK(cudaMemcpy(cdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_full(nullptr, &t_leg);
      std::vector<float> ldk(nkv), ldv(nkv), ldq(nq);
      CUDA_CHECK(cudaMemcpy(ldk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4o A/B] varlen ksplit=%d | maxlen-strided %.4f ms | compact %.4f ms (%.3fx) | "
             "compact-vs-legacy bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_leg, t_cmp, t_leg / t_cmp, mad3(cdq, ldq), mad3(cdk, ldk), mad3(cdv, ldv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    cudaFree(d_part_base);
    run_all();   // 恢复最终输出为 CLI 选中的路径
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- P3-4j/P3-4l A/B（D=512 MLA varlen，`--det`）：把 P3-4i 的确定性 dK/dV 从 D=128 扩到
  //      MLA（HD=512）的变长路径（DET 候选 ① 的最后一块）。dK/dV 走 partial +
  //      `dkv_reduce_varlen_kernel<512,64>`（按 `cu_seqlens` 的逐序列 `len_b/nblk_b` 定界、
  //      输出按 packed token 定位）。**P3-4l**：MLA 的 dQ 无法用寄存器累加
  //      （`kRegDq = REGDQ && (HD/NTW==1)` 恒 false），但 P3-4k 已让非 `kRegDq` 的 dQ epilogue
  //      在 `DET && ksplit>1` 时写按 part 分片的 `dq_part`（同 `(mblk,h,part)` 只被一个 CTA
  //      写 ⇒ 天然确定），故 ksplit>1 时再接 `dq_reduce_kernel<512>`（packed token = `qbase+qi`，
  //      与定长 `dq_reduce_kernel` 的 `row` 语义一致）按 part 固定次序求和；`ksplit==1` 仍走
  //      单写者 `red_add2`（逐位不变）。与默认 MLA varlen 主 kernel 同几何（O47/O51 的
  //      8-warp/256 线程），但用非 kvpipe 版（DET 未接进 kvpipe）。
  //      同 session 计时 + 跑两遍 DET 验逐位，再与 atomic 比 max|diff|。----
  if (det_ab && D == 512) {
    const int nblk_max = (maxlen + BM - 1) / BM;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_max * maxlen * D;   // 旧（maxlen-strided）上界
    // P3-4o：compact per-sequence offset（同 D=128 版；dK/dV 布局 HD 无关）。
    std::vector<int> part_base_h(B + 1, 0);
    for (int b = 0; b < B; ++b) {
      const int len_b = cu[b + 1] - cu[b];
      const int nblk_b = (len_b + BM - 1) / BM;
      part_base_h[b + 1] = part_base_h[b] + H * nblk_b * len_b;
    }
    const size_t part_elems_cmp = (size_t)part_base_h[B] * D;
    int* d_part_base = nullptr;
    CUDA_CHECK(cudaMalloc(&d_part_base, (B + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_part_base, part_base_h.data(), (B + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    const int* pb = part_compact ? d_part_base : nullptr;
    const size_t dqpart_elems = (size_t)T * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    printf("[P3-4o] varlen D=512 partial: compact=%.1fMB (%.0f%% of old %.1fMB) layout=%s\n",
           part_elems_cmp * 4 / 1e6, 100.0 * (double)part_elems_cmp / (double)part_elems,
           part_elems * 4 / 1e6, pb ? "compact" : "maxlen-strided");
    dim3 g(nblk_max * ks, H, B);
    auto run_at = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_cu, nullptr, nullptr);
    };
    // P3-4m：两个 DET 主 kernel 变体共用同一段 reduce（非 kvpipe 与 kvpipe 只差搬运，
    //   数值应逐位相同；kvpipe 是 O51 的 K/V cp.async 回填流水，此前 DET 未接线）。
    auto do_reduce = [&](const int* p) {
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (ks > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<512, 64><<<dkv_blocks + dq_blocks, 512>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk, d_dv, d_dq, d_cu, H, Hkv,
            nblk_max, maxlen, (int)causal, ks, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<512, 64><<<rg, 512>>>(d_dk_part, d_dv_part, d_dk, d_dv, d_cu, H,
                                                       Hkv, nblk_max, maxlen, (int)causal, p);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq, T, H, ks);
        }
      }
    };
    auto run_dt_p = [&](const int* p) {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));   // ksplit==1 时 dQ 用无竞争 red_add2
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_dk_part, d_dv_part, nblk_max, d_dq_part, d_cu,
          nullptr, nullptr, p);
      do_reduce(p);
    };
    auto run_dt = [&]() { run_dt_p(pb); };
    // P3-4m：DET + O51 K/V `cp.async` 回填流水（kvpipe）。
    auto run_dt_kv = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, d_dk_part, d_dv_part, nblk_max, d_dq_part, d_cu,
          nullptr, nullptr, pb);
      do_reduce(pb);
    };
    auto time_fn3 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f, ms_dt_kv = 0.f;
    time_fn3([&] { run_at(); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    time_fn3([&] { run_dt(); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    // P3-4m：DET + kvpipe（数值应与非 kvpipe 逐位相同）。
    time_fn3([&] { run_dt_kv(); }, &ms_dt_kv);
    std::vector<float> dkk1(nkv), dvk1(nkv), dqk1(nq), dkk2(nkv), dvk2(nkv), dqk2(nq);
    CUDA_CHECK(cudaMemcpy(dkk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk1.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    run_dt_kv();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dkk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk2.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    auto mad3 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4l A/B] MLA varlen ksplit=%d | atomic(8w) %.4f ms | DET(8w) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad3(dq1, dq2), mad3(dk1, dk2), mad3(dv1, dv2),
           mad3(dq1, aq), mad3(dk1, ak), mad3(dv1, av));
    printf("[P3-4m A/B] MLA varlen ksplit=%d | DET nonkv %.4f ms | DET kvpipe %.4f ms (%.3fx) | "
           "kvpipe runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | kvpipe-vs-nonkv dq/dk/dv="
           "%.2e/%.2e/%.2e\n",
           ks, ms_dt, ms_dt_kv, ms_dt / ms_dt_kv, mad3(dqk1, dqk2), mad3(dkk1, dkk2),
           mad3(dvk1, dvk2), mad3(dqk1, dq1), mad3(dkk1, dk1), mad3(dvk1, dv1));
    // ---- F4-b（第 134 轮）：MLA（HD=512）varlen 的「fp16 partial + O62 扇区化」。
    //      `dkv_p16_perm` 只在 16 列块内置换、且 GEMM3/4 的 warp n-tile `c0=wc*GN34`
    //      （HD=512/8w：GN34=32）是 16 的倍数 ⇒ 全局列索引可直接套用，与定长/D=128 同一布局。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth_p = [&](const int* p) {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, false, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk, d_dv,
          maxlen, H, Hkv, scale, (int)causal, ks, reinterpret_cast<float*>(d_dk_part_h),
          reinterpret_cast<float*>(d_dv_part_h), nblk_max, d_dq_part, d_cu, nullptr, nullptr, p);
      const int dkv_blocks = B * Hkv * maxlen;
      const int dq_blocks = (ks > 1) ? T * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_varlen_kernel<512, 64, true><<<dkv_blocks + dq_blocks, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (ks > 1) ? d_dq_part : nullptr, d_dk,
            d_dv, d_dq, d_cu, H, Hkv, nblk_max, maxlen, (int)causal, ks, dkv_blocks, p);
      } else {
        dim3 rg(B * Hkv, maxlen);
        dkv_reduce_varlen_kernel<512, 64, true><<<rg, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk, d_dv, d_cu, H, Hkv, nblk_max,
            maxlen, (int)causal, p);
        if (ks > 1) {
          dim3 dg(T, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq, T, H, ks);
        }
      }
    };
    float ms_dth = 0.f;
    time_fn3([&] { run_dth_p(pb); }, &ms_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth_p(pb);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] MLA varlen ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, ms_dt, ms_dth, ms_dt / ms_dth, mad3(hdk1, hdk2), mad3(hdv1, hdv2),
           mad3(hdk1, dk1), mad3(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    // ---- P3-4o A/B：同 binary 只改 partial 布局（compact vs maxlen-strided）。
    {
      auto time_full = [&](const int* p, float* out_ms) {
        auto fn = [&]() { run_dt_p(p); };
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float t = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
        *out_ms = t / iters;
      };
      float t_cmp = 0.f, t_leg = 0.f;
      time_full(d_part_base, &t_cmp);
      std::vector<float> cdk(nkv), cdv(nkv), cdq(nq);
      CUDA_CHECK(cudaMemcpy(cdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(cdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      time_full(nullptr, &t_leg);
      std::vector<float> ldk(nkv), ldv(nkv), ldq(nq);
      CUDA_CHECK(cudaMemcpy(ldk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(ldq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4o A/B] MLA varlen ksplit=%d | maxlen-strided %.4f ms | compact %.4f ms (%.3fx) | "
             "compact-vs-legacy bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_leg, t_cmp, t_leg / t_cmp, mad3(cdq, ldq), mad3(cdk, ldk), mad3(cdv, ldv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    cudaFree(d_part_base);
    run_all();   // 恢复最终输出为 CLI 选中的路径
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // ---- O52 A/B（D=512/MLA/varlen）：主 kernel 4-warp vs 8-warp(+kvpipe) 同 session 计时 ----
  //   只改 warp 网格与 K/V 搬运，数学口径不变；跨 CTA atomic 次序变 ⇒ 预期 max_abs ~1e-6。
  if (D == 512) {
    dim3 mgab = compact ? dim3(total_mt * ksplit, H, 1)
                        : dim3((maxlen + BM - 1) / BM * ksplit, H, B);
    const int* mtba = compact ? d_mtb : nullptr;
    const int* mtma = compact ? d_mtm : nullptr;
    auto go52 = [&](int mode) {  // 0=4w, 1=8w, 2=8w+kvpipe
      if (mode == 2)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
      else if (mode == 1)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true>(
            mgab, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq, d_dk,
            d_dv, maxlen, H, Hkv, scale, (int)causal, ksplit, d_cu, mtba, mtma);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv, 0, nkv * 4));
    };
    cudaEvent_t eva, evb;
    CUDA_CHECK(cudaEventCreate(&eva));
    CUDA_CHECK(cudaEventCreate(&evb));
    auto bench52 = [&](int mode, float* out) {
      zero_acc();
      go52(mode);
      CUDA_CHECK(cudaEventRecord(eva));
      for (int i = 0; i < iters; ++i) go52(mode);
      CUDA_CHECK(cudaEventRecord(evb));
      CUDA_CHECK(cudaEventSynchronize(evb));
      CUDA_CHECK(cudaEventElapsedTime(out, eva, evb));
      *out /= iters;
    };
    float m4 = 0.f, m8 = 0.f, mkv = 0.f;
    bench52(0, &m4);
    bench52(1, &m8);
    bench52(2, &mkv);
    auto grab52 = [&](int mode, std::vector<float>& dq, std::vector<float>& dk,
                      std::vector<float>& dv) {
      zero_acc();
      go52(mode);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    };
    std::vector<float> a4(nq), b4(nkv), c4(nkv), a8(nq), b8(nkv), c8(nkv), ak(nq), bk(nkv), ck(nkv);
    grab52(0, a4, b4, c4);
    grab52(1, a8, b8, c8);
    grab52(2, ak, bk, ck);
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double e = 0.0;
      for (size_t i = 0; i < x.size(); ++i) e = std::max(e, (double)fabs((double)x[i] - (double)y[i]));
      return e;
    };
    printf("[O52 A/B] varlen MLA main-only 4w %.4f ms | 8w %.4f ms (%.3fx) | 8w+kvpipe %.4f ms "
           "(%.3fx) | max_abs(8w-vs-4w) dq=%.3e dk=%.3e dv=%.3e | max_abs(kvp-vs-8w) "
           "dq=%.3e dk=%.3e dv=%.3e\n",
           m4, m8, m4 / m8, mkv, m4 / mkv, maxd(a4, a8), maxd(b4, b8), maxd(c4, c8),
           maxd(a8, ak), maxd(b8, bk), maxd(c8, ck));
    run_all();  // 恢复 CLI 选中路径（写回 d_dq/d_dk/d_dv）
  }

  // ---- O54 A/B（D=512/MLA/varlen/full）：非 causal LSE 的「bal FULL + K 维 split」vs 旧版。
  if (D == 512 && !causal) {
    const int nblk = (maxlen + LBM - 1) / LBM;
    auto go_lse = [&](int which, int sp) {
      if (which == 0)
        launch_lse<512>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale, 0,
                        d_cu);
      else if (sp <= 1)
        launch_lse_bal<512, 1, true>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                     Hkv, scale, d_cu, nullptr, 1);
      else
        launch_lse_bal<512, 1, true>(dim3(nblk, H, B), d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H,
                                     Hkv, scale, d_cu, d_lse_part, sp, (long long)rows_q);
    };
    cudaEvent_t ela, elb;
    CUDA_CHECK(cudaEventCreate(&ela));
    CUDA_CHECK(cudaEventCreate(&elb));
    auto bench_lse = [&](int which, int sp, float* out) {
      go_lse(which, sp);
      CUDA_CHECK(cudaEventRecord(ela));
      for (int i = 0; i < iters; ++i) go_lse(which, sp);
      CUDA_CHECK(cudaEventRecord(elb));
      CUDA_CHECK(cudaEventSynchronize(elb));
      CUDA_CHECK(cudaEventElapsedTime(out, ela, elb));
      *out /= iters;
    };
    float t0 = 0.f;
    bench_lse(0, 1, &t0);
    printf("[O54 A/B] varlen MLA full LSE: old(O1 mma) %.4f ms |", t0);
    for (int sp : {1, 2, 4, 8, 16}) {
      float t = 0.f;
      bench_lse(1, sp, &t);
      printf(" balFULL/split%d %.4f", sp, t);
    }
    printf(" ms (auto=%d)\n", lse_split_eff);
    run_all();  // 恢复 CLI 选中路径
  }

  // ---- O58 A/B（D=512/MLA/varlen）：LSE 几何 sweep（LSE-only、同 session；causal 镜像配对 / full
  //   单 m 块）。默认 causal=cfg6、full=O54 旧路；8-warp 需 `--lse8w=1`。----
  if (D == 512) {
    dim3 lgc = causal ? dim3((nblk0 + 1) / 2, H, B) : dim3(nblk0, H, B);
    dim3 lg8c = causal ? dim3((lse_nblk8 + 1) / 2, H, B) : dim3(lse_nblk8, H, B);
    auto go_lse58 = [&](auto full_tag, int which, int sp) {
      constexpr bool F = decltype(full_tag)::value;
      if (which == 5)
        launch_lse_bal<512, 0, F, 128, 32>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else if (which == 6)
        launch_lse_bal<512, 1, F, 128, 16>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else if (which == 8)
        launch_lse_bal<512, 1, F, 256, 32>(lg8c, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv,
                                           scale, d_cu, d_lse_part, sp, (long long)rows_q);
      else
        launch_lse_bal<512, 1, F>(lgc, d_q8, d_qs, d_k8, d_ks, d_lse, maxlen, H, Hkv, scale,
                                  d_cu, d_lse_part, sp, (long long)rows_q);
    };
    cudaEvent_t eca, ecb;
    CUDA_CHECK(cudaEventCreate(&eca));
    CUDA_CHECK(cudaEventCreate(&ecb));
    auto bench_lse_c = [&](auto full_tag, int which, int sp, float* out) {
      go_lse58(full_tag, which, sp);
      CUDA_CHECK(cudaEventRecord(eca));
      for (int i = 0; i < iters; ++i) go_lse58(full_tag, which, sp);
      CUDA_CHECK(cudaEventRecord(ecb));
      CUDA_CHECK(cudaEventSynchronize(ecb));
      CUDA_CHECK(cudaEventElapsedTime(out, eca, ecb));
      *out /= iters;
    };
    auto sweep58 = [&](auto full_tag, int which, const char* tag) {
      printf("[O58 A/B] %s:", tag);
      for (int sp : {1, 2, 4, 8, 16}) {
        float t = 0.f;
        bench_lse_c(full_tag, which, sp, &t);
        printf(" split%d %.4f", sp, t);
      }
      printf(" ms\n");
    };
    if (causal) {
      sweep58(std::integral_constant<bool, false>{}, 0, "legacy PIPE1/LBN64 4w (1 CTA)");
      sweep58(std::integral_constant<bool, false>{}, 6, "cfg6 PIPE1/LBN16 4w (2 CTA) [new default]");
      sweep58(std::integral_constant<bool, false>{}, 5, "cfg5 PIPE0/LBN32 4w (2 CTA)");
      sweep58(std::integral_constant<bool, false>{}, 8, "PIPE1/LBN32 8w (1 CTA)");
    } else {
      sweep58(std::integral_constant<bool, true>{}, 0, "legacy PIPE1/LBN64 4w (1 CTA) [default]");
      sweep58(std::integral_constant<bool, true>{}, 6, "cfg6 PIPE1/LBN16 4w (2 CTA)");
      sweep58(std::integral_constant<bool, true>{}, 5, "cfg5 PIPE0/LBN32 4w (2 CTA)");
      sweep58(std::integral_constant<bool, true>{}, 8, "PIPE1/LBN32 8w (1 CTA)");
    }
    run_all();  // 恢复 CLI 选中路径
  }

  std::vector<float> h_dq(nq), h_dk(nkv), h_dv(nkv);
  CUDA_CHECK(cudaMemcpy(h_dq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_dv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));
  auto report = [](const char* nm, const std::vector<float>& a, const NpyF32& b) {
    size_t n = std::min(a.size(), b.data.size());
    double ma = 0.0, mr = 0.0, ao = 0.0, ar = 0.0, l2r = 0.0;
    size_t arg = 0;
    for (size_t i = 0; i < n; ++i) {
      double d = fabs((double)a[i] - (double)b.data[i]);
      if (d > ma) { ma = d; arg = i; }
      ao = std::max(ao, fabs((double)a[i]));
      ar = std::max(ar, fabs((double)b.data[i]));
      double den = std::max(1e-3, fabs((double)b.data[i]));
      double r = d / den;
      if (r > mr) mr = r;
      l2r += (double)b.data[i] * (double)b.data[i];  // O108：relL2 护栏口径
    }
    const double diff_l2 = [&] {
      double s = 0.0;
      for (size_t i = 0; i < n; ++i) {
        double d = (double)a[i] - (double)b.data[i];
        s += d * d;
      }
      return sqrt(s);
    }();
    printf("  %-3s vs ref: max_abs=%.3e  max_rel=%.3e  relL2=%.3f%%  "
           "(ours_amax=%.3e ref_amax=%.3e @%zu)\n",
           nm, ma, mr, 100.0 * diff_l2 / std::max(1e-30, sqrt(l2r)), ao, ar, arg);
  };
  printf("[compare] VARLEN ours vs fp32 ref\n");
  report("dq", h_dq, rdq);
  report("dk", h_dk, rdk);
  report("dv", h_dv, rdv);
  // P3-3 varlen：`--dump=<prefix>` 把 packed 的 dq/dk/dv 落成 npy，供 fa_bwd_compare.py 汇总。
  if (!dump.empty()) {
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dq.npy", h_dq.data(), (long long)h_dq.size());
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dk.npy", h_dk.data(), (long long)h_dk.size());
    fa_bwd_save_npy_f32(dir + "/" + dump + "_dv.npy", h_dv.data(), (long long)h_dv.size());
    printf("[dump] ours -> %s/%s_{dq,dk,dv}.npy\n", dir.c_str(), dump.c_str());
  }
  return 0;
}

// =============================================================================
// host / launcher / self-test
// =============================================================================
int main(int argc, char** argv) {
  std::string dir = "/home/xieminglin/proj/output/fa-bwd/b1_s512_h16_d128_causal_fp8";
  std::string o_name = "ref_o";
  // P3-3：`--dump=<prefix>` 让 ours 的 dq/dk/dv 落成 `<prefix>_{dq,dk,dv}.npy`，
  //   供 harness/fa_bwd_compare.py 统一对拍（默认不落盘，行为逐位不变）。
  std::string dump_prefix;
  bool causal = true;
  int iters = 20;
  int ksplit = -1;  // -1 = 自动
  int ksplit2_opt = -1;  // O19：wg2 的 ksplit（-1 自动）
  // O89：定长 causal 主 kernel 的 m 块调度序（LPT）。0 = 历史；1 = 贵块先跑（削尾波）。
  int mrev_opt = 1;  // O89：默认开（LPT 贵块先跑）；--mrev=0 A/B
  // O93（第 188 轮，LPT 候选 ③）：跨 head 全局 LPT——grid 轴对调（head 走快轴），
  //   所有 head 的最贵 m 块一起先派发。默认开（正结果）；--hswap=0 A/B。配合 hswap_elig
  //   时自动把 ksplit 收到 2（Q/dO 重读 8×→2×）。
  int hswap_opt = 1;
  // O119（第 213 轮）：K/V TMA cluster multicast 宽度。`--mcast=C`（C∈{2,4,8}）显式；
  //   `--mcast` 或 `--mcast=-1` = auto（最大 2 的幂 ≤ min(G=H/Hkv, 8)）；默认 1 = 关。
  int mcast_opt = 1;
  // O95（第 189 轮，O93 候选 ③）：**变 ks 调度**——`--ksm=N` 让最贵的 N 个 m 块用 ksplit=2、
  //   其余 m 块用 ksplit=1（表驱动、全局 LPT 序）。目的：在 O93 的 ksplit=2 平衡点上再砍掉
  //   便宜块那部分 Q/dO 重读（`l2 read`）与 dQ 跨-part 原子；只改「哪个 CTA 算哪段 K」，
  //   非 DET 默认路径的 dK/dV/dQ 仍是可交换跨 CTA `atomicAdd` ⇒ 数值在 fp8 噪声内。
  //   `-1` = 关（历史 O93 均匀 ksplit）。仅在 hswap eligible（定长 causal D=128 nblk>=16）生效。
  int ksm_opt = -1;  // O95：--ksm=N 开启（N=最贵 m 块数）；默认关
  int ksm_hi = 2;    // O95：最贵 N 个 m 块的 ks（默认 2）；`--ksmhi=K` A/B（如 3/4）
  // O22：在 `-DFA_WGMMA`（sm_90a）构建下，默认启用 Hopper 路径（LSE + 主 kernel GEMM1/2 的
  //   wgmma）；sm_90 构建下这两个宏路径不存在，保持 mma。`--lsewgm=0/--wgmma=0` 可显式退回 mma
  //   做 A/B（S=4096 端到端 wgmma 比 mma 快 ~8%：2.70→2.49ms，preprocess 0.40→0.32、main 2.19→2.04）。
#ifdef FA_WGMMA
  int lsewgm = 1;
  int wgmma = 1;
#else
  int lsewgm = 0;
  int wgmma = 0;
#endif
  int wg2 = 0;      // O19：1 = 主 kernel 走跨 warpgroup 归约版（BM=128, 2 wg, 256 线程）
  int wg2wgmma = 0; // F6：1 = BM=128 双 warpgroup 主 kernel 的 GEMM1/2 走 wgmma（SW128）
  int wg2tma = 0;   // O90（F6-step3）：1 = wgmma2 的 K/V 改 4D-TMA（K 双缓冲）
  int wg2ws = 0;    // O121（F3b）：1 = wgmma2 + WS（384 线程：2 compute WG + 1 producer WG）
  int wg3 = 0;        // O91（F6-step4）：1 = BM=192 / 3 warpgroup / 384 线程主 kernel
  int ksplit3_opt = -1;  // O91：wg3 的 ksplit（-1 自动）
  int prel_opt = -1;  // O12：-1 自动（开）；0/1 强制 LSE/D 预装寄存器开关
  int qfast = 1;      // O14：1 = warp-per-row 向量化量化，0 = 旧 per-row 标量量化（A/B）
  int delta_warp_opt = 1;  // O26：1 = warp-per-row 向量化 delta（默认），0 = 旧 per-row smem 归约（A/B）
  int f16b_opt = 1;   // O7e-2：1 = fold 16B 向量化写（默认），0 = 退回 O7e 的 4B 写（A/B）
  int bn64_opt = 0;   // O21：1 = 主 kernel KV tile BN=64（mma 路径，D=128）
  int cvt_on = 0;     // O21b：1 = 保留冗余的 fp32→fp32 convert 拷贝（默认 0：直接累加进输出）
  // O64：1 = 把 4 次输入量化 + 3 次累加缓冲清零融合成 1 个 launch（默认 1，数值逐位不变）；
  //   0 = 退回 O14 的 4 个 quant kernel + 3 个 cudaMemset，供同 session A/B。
  int qfuse = 1;
  // O88（本机 A/B）：融合 quant+zero kernel 的**栅格上限**（0 = 历史行为：grid = 任务数/4，
  //   即「每 warp 一行、每 CTA 4 行」，S4096 时 = 114688 个 CTA）。该 kernel 是纯带宽/访存
  //   kernel，114688 个微小 CTA 的**块调度开销**可能是端到端非 main 开销的一部分；把栅格
  //   封顶、让每 warp 沿 grid-stride 多搬几行可摊薄块调度。`--qcap=N` 强制上限（N=0 历史）。
  int qcap_opt = 0;
  // O66：1 = 把 delta 也融进 quant kernel 的 dO 任务（默认 1，数值逐位不变；需 qfuse=1
  //   且 delta 走 warp-per-row 版）；0 = 独立 delta launch，供同 session A/B。
  int dfuse = 1;
  int foldrcp_opt = 1;  // O27：1 = fold 量化用「每行 rcp + 乘法」（默认），0 = 精确除法（A/B）
  int regdq_opt = -1; // O22：-1 自动；0/1 强制关/开寄存器 dQ 累加（同 session A/B）
  // O32：LSE 是否用 TMA 版（仅 FA_TMA 构建、D==128、causal）。-1=自动（默认开），0/1 由
  //   `--lsetma=` 强制。
  int lse_tma = -1;
  // O37：主 kernel 的 Q/dO 是否用 4D-TMA（仅 FA_WGMMA+FA_TMA、D==128）。-1=自动（默认关，
  //   作 opt-in 与 cp.async 版同 binary A/B），0/1 由 `--qdtma=` 强制。
  int qd_tma = -1;
  // O41：主 kernel 的 K/V 是否也用 4D-TMA（roadmap「下一步候选 ①」）。仅 `-DFA_WGMMA -DFA_TMA`
  //   构建、D==128；-1=自动（默认开，对齐 O37），0/1 由 `--kvtma=` 强制。
  int kv_tma = -1;
  // O84（第 179 轮）：head_dim=256 的主 kernel 是否走 Hopper wgmma（GEMM1/2 换 wgmma、Q/dO/K/V
  //   存 SW128；非 TMA）。-1=自动（`-DFA_WGMMA` 构建默认开），0/1 由 `--d256wgm=` 强制（A/B）。
  int d256wgm_opt = -1;
  // O85（第 180 轮）：head_dim=256 主 kernel 的 Q/dO 是否走 **4D-TMA**（chunk-major，NCH=2；
  //   K/V 仍 cp.async）。**A/B 中性 ⇒ 默认关**（`--d256tma=1` 开启，opt-in）。smem 与
  //   O84 的 cp.async wgmma 档相同（115,712B）⇒ 仍 2 CTA/SM。
  int d256tma_opt = -1;
  // O38：LSE 的 K 维 split 数（仅 D=128/causal/TMA 生效）。0=自动（目标 grid*split≈2048、上限 8），
  //   >=1 直接指定（`--lsesplit=N` 走「切片 partial + merge」；`--lsesplit=1` 退回 O32、保持历史逐位）。
  int lse_split = 0;
  // O47：MLA（D=512）主 kernel 的 warp 网格。-1=自动（D=512 默认开 8-warp/256 线程），
  //   0/1 由 `--mla8w=` 强制（同 session A/B）。默认 128/2 与历史逐字等价。
  int mla8w_opt = -1;
  // O51：MLA（D=512）主 kernel 的 K/V cp.async 回填流水。-1=自动（默认开），0/1 由 `--mlakvp=`
  //   强制（同 binary A/B；需 8-warp 几何）。
  int mla_kvp_opt = -1;
  // O58：MLA（D=512）varlen LSE 的几何开关。`--lseocc=5/6`（2 CTA/SM：PIPE0/LBN32 或
  //   PIPE1/LBN16）、`--lse8w=1`（8-warp/256 线程/LBM=128/LBN=32）；默认 causal 走 cfg6、
  //   full 走 O54 旧路。`--lseocc=4` 退回 causal 旧默认（PIPE1/LBN64）做 A/B。
  int lseocc_opt = 0;
  int lse8w_opt = 0;
  // O48（候选 ①）：D=128 主 kernel 是否也用 8-warp（256 线程 / 2×4 网格）几何。0=默认
  //   4-warp（128/2），1=8-warp。仅 mma 路径（WGMMA 版 GEMM1/2 是 warpgroup 级、结构上锁死
  //   2 warp）；用 `--d128w=0/1` 做同 binary A/B。见 docs/03 §48。O49：默认 **-1=自动**
  //   （仅 `!wgmma` 的 mma 路径、且含 ksplit 的有效 grid ≤ SM 数时开；fp8 的 auto split-K
  //   通常已把小 S 的 grid 抬到 ≫132，故自动档在默认 shape 下不触发、保持逐位）。
  int d128w_opt = -1;
  // P3-4e：1 = 跑「确定性 dK/dV（partial + 固定次序归约）」A/B（仅 D=128 定长 mma 默认路径，
  //   对齐 fp16/bf16 O7b；`--det` 或 `--det=1`）。默认关，不影响常规计时。
  int det_ab = 0;
  // P3-4f：DET 实验用的 ksplit。>1 时 dQ 也走 partial（确定性 + split-K 并行度）。
  //   `--detk=N`。**O102（第 196 轮）**：默认改为 0 = auto——定长 D=128 的 DET A/B 在
  //   causal 下取 k=4（标定触底）、full 取 k=1；`--detk=1` 可显式复现 P3-4e/g 原状。
  int det_ksplit = 0;
  // P3-4n：DET 的 dkv/dq 二次归约是否融合成一次 launch（默认 1）。`--nofusered` 退回两次
  //   launch 做同 binary A/B。两版求和次序逐字相同 ⇒ 数值逐位相同。
  int fuse_reduce = 1;
  // P3-4o：varlen DET 的 dK/dV partial 是否用 compact per-sequence 布局（默认 0 = 旧
  //   maxlen-strided，性能略优）；`--partcompact` 打开（分配大幅缩小、实测 ~0.966–0.975×）。
  int part_compact = 0;
  int varlen = 0;   // VARLEN：1 = packed [T,H,D] + cu_seqlens.npy（fp8/HD=128/causal）
  int compact_opt = 0;  // 第八十二轮：1 = varlen 主 kernel 紧凑均衡网格（opt-in；实测中性偏负）
  int lse_compact_opt = 0;  // 第八十二轮：1 = varlen causal LSE 紧凑对网格（opt-in，A/B）
  // O109：量化分相 + LSE 跨 stream 重叠（默认 -1=auto，定长 D=128 默认融合路径开）。
  //   `--ovlql=0` 退回原「合并量化 → LSE → main」串行做同 binary A/B。数值逐位相同。
  int ovl_ql = -1;
  // O110：phase1 栅格封顶（0=auto，>0 显式，<0 不封顶）。
  int ovl_cap = 0;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--full") causal = false;
    else if (a == "--varlen") varlen = 1;
    else if (a == "--compact") compact_opt = 1;
    else if (a == "--lsecompact") lse_compact_opt = 1;
    else if (a.rfind("--varlen=", 0) == 0) varlen = atoi(a.c_str() + 9);
    else if (a == "--causal") causal = true;
    else if (a == "--lsewgm") lsewgm = 1;
    else if (a == "--wgmma") wgmma = 1;
    else if (a == "--lsetma") lse_tma = 1;
    else if (a.rfind("--lsewgm=", 0) == 0) lsewgm = atoi(a.c_str() + 9);
    else if (a.rfind("--wgmma=", 0) == 0) wgmma = atoi(a.c_str() + 8);
    else if (a.rfind("--lsetma=", 0) == 0) lse_tma = atoi(a.c_str() + 9);
    else if (a.rfind("--lsesplit=", 0) == 0) lse_split = atoi(a.c_str() + 11);
    else if (a.rfind("--lsefull=", 0) == 0) g_lse_full_opt = atoi(a.c_str() + 10);
    else if (a == "--lsefull") g_lse_full_opt = 1;
    else if (a.rfind("--lse512old=", 0) == 0) g_lse_full512_opt = !atoi(a.c_str() + 12);
    else if (a == "--lse512old") g_lse_full512_opt = 0;
    else if (a.rfind("--lsetmavarlen=", 0) == 0) g_lse_tma_varlen = atoi(a.c_str() + 15);
    else if (a == "--lsetmavarlen") g_lse_tma_varlen = 1;
    else if (a.rfind("--mla8w=", 0) == 0) mla8w_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--mlakvp=", 0) == 0) mla_kvp_opt = atoi(a.c_str() + 9);
    else if (a == "--mlakvp") mla_kvp_opt = 1;
    else if (a.rfind("--lseocc=", 0) == 0) lseocc_opt = atoi(a.c_str() + 9);
    else if (a.rfind("--lse8w=", 0) == 0) lse8w_opt = atoi(a.c_str() + 8);
    else if (a == "--lse8w") lse8w_opt = 1;
    else if (a.rfind("--d128w=", 0) == 0) d128w_opt = atoi(a.c_str() + 8);
    else if (a == "--d128w") d128w_opt = 1;
    else if (a.rfind("--ovlql=", 0) == 0) ovl_ql = atoi(a.c_str() + 8);
    else if (a.rfind("--ovlcap=", 0) == 0) ovl_cap = atoi(a.c_str() + 9);
    else if (a.rfind("--det=", 0) == 0) det_ab = atoi(a.c_str() + 6);
    else if (a == "--det") det_ab = 1;
    else if (a.rfind("--detk=", 0) == 0) det_ksplit = atoi(a.c_str() + 7);
    else if (a == "--nofusered") fuse_reduce = 0;
    else if (a == "--partcompact") part_compact = 1;
    else if (a.rfind("--qdtma=", 0) == 0) qd_tma = atoi(a.c_str() + 8);
    else if (a.rfind("--kvtma=", 0) == 0) kv_tma = atoi(a.c_str() + 8);
    else if (a == "--kvtma") kv_tma = 1;
    else if (a.rfind("--d256wgm=", 0) == 0) d256wgm_opt = atoi(a.c_str() + 10);
    else if (a == "--d256wgm") d256wgm_opt = 1;
    else if (a.rfind("--d256tma=", 0) == 0) d256tma_opt = atoi(a.c_str() + 10);
    else if (a == "--d256tma") d256tma_opt = 1;
    else if (a == "--wg2") wg2 = 1;
    else if (a == "--wg2wgmma") wg2wgmma = 1;
    else if (a == "--wg2tma") wg2tma = 1;
    else if (a == "--wg2ws") wg2ws = 1;
    else if (a == "--wg3") wg3 = 1;
    else if (a.rfind("--ksplit3=", 0) == 0) ksplit3_opt = atoi(a.c_str() + 10);
    else if (a == "--bn64") bn64_opt = 1;
    else if (a.rfind("--cvt=", 0) == 0) cvt_on = atoi(a.c_str() + 6);
    else if (a.rfind("--qfuse=", 0) == 0) qfuse = atoi(a.c_str() + 8);
    else if (a.rfind("--dfuse=", 0) == 0) dfuse = atoi(a.c_str() + 8);
    else if (a.rfind("--qcap=", 0) == 0) qcap_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--foldrcp=", 0) == 0) foldrcp_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--qfast=", 0) == 0) qfast = atoi(a.c_str() + 8);
    else if (a.rfind("--deltawarp=", 0) == 0) delta_warp_opt = atoi(a.c_str() + 12);
    else if (a.rfind("--regdq=", 0) == 0) regdq_opt = atoi(a.c_str() + 8);
    else if (a.rfind("--prel=", 0) == 0) prel_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--f16b=", 0) == 0) f16b_opt = atoi(a.c_str() + 7);
    else if (a.rfind("--o=", 0) == 0) o_name = a.substr(4);
    else if (a.rfind("--dump=", 0) == 0) dump_prefix = a.substr(7);
    else if (a.rfind("--iters=", 0) == 0) iters = atoi(a.c_str() + 8);
    else if (a.rfind("--ksplit=", 0) == 0) ksplit = atoi(a.c_str() + 9);
    else if (a.rfind("--ksplit2=", 0) == 0) ksplit2_opt = atoi(a.c_str() + 10);
    else if (a.rfind("--mrev=", 0) == 0) mrev_opt = atoi(a.c_str() + 7);
    else if (a == "--mrev") mrev_opt = 1;
    else if (a.rfind("--hswap=", 0) == 0) hswap_opt = atoi(a.c_str() + 8);
    else if (a == "--hswap") hswap_opt = 1;
    else if (a.rfind("--mcast=", 0) == 0) mcast_opt = atoi(a.c_str() + 8);
    else if (a == "--mcast") mcast_opt = -1;
    else if (a.rfind("--ksm=", 0) == 0) ksm_opt = atoi(a.c_str() + 6);
    else if (a == "--ksm") ksm_opt = 0;
    else if (a.rfind("--ksmhi=", 0) == 0) ksm_hi = atoi(a.c_str() + 8);
    else if (a.rfind("--dir=", 0) == 0) dir = a.substr(6);
    else if (!a.empty() && a[0] != '-') dir = a;
  }

  if (varlen)
    return run_varlen(dir, causal, iters, compact_opt, lse_compact_opt, lse_split, mla8w_opt,
                       mla_kvp_opt, lseocc_opt, lse8w_opt, dump_prefix, det_ab, det_ksplit,
                       fuse_reduce, part_compact, qfuse, dfuse, ksplit, mrev_opt, ovl_ql, ovl_cap);

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
  if (k_np.shape.size() != 4 || v_np.shape.size() != 4) {
    fprintf(stderr, "期望 k/v 为 4D [B,S,Hkv,D]\n");
    return 1;
  }
  const int B = (int)q_np.shape[0], S = (int)q_np.shape[1];
  const int H = (int)q_np.shape[2], D = (int)q_np.shape[3];
  const int Hkv = (int)k_np.shape[2];   // P5-3：GQA/MQA 的 KV 头数（MHA 时 Hkv==H）
  if (D != 128 && D != 256 && D != 512) {
    fprintf(stderr, "本版本支持 head_dim=128（MHA/GQA）/ 256 / 512（MLA）；当前 %d\n", D);
    return 1;
  }
  if ((int)v_np.shape[2] != Hkv || (int)v_np.shape[3] != D) {
    fprintf(stderr, "k/v 形状不一致（不支持 Dv!=D）\n");
    return 1;
  }
  if (H % Hkv != 0) {
    fprintf(stderr, "H(%d) 必须是 Hkv(%d) 的整数倍\n", H, Hkv);
    return 1;
  }
  const size_t nq = (size_t)B * S * H * D;      // q/dO/dq 长度
  const size_t nkv = (size_t)B * S * Hkv * D;   // k/v/dk/dv 长度
  const size_t rows_q = (size_t)B * S * H;
  const size_t rows_kv = (size_t)B * S * Hkv;
  const float scale = 1.0f / sqrtf((float)D);

  // 主 kernel tile 固定 BM=64,BN=32；smem 随 HD 变化。
  // O76（第 171 轮）：head_dim=256 走 mma 主 kernel（无 wgmma/TMA），smem ≈115KB ⇒ 1 CTA/SM。
  const int smem_bytes = (D == 128)   ? Fp8Cfg<128, 64, 32>::smem_bytes
                         : (D == 256) ? Fp8Cfg<256, 64, 32>::smem_bytes
                                      : Fp8Cfg<512, 64, 32>::smem_bytes;
  const int lse_smem = (D == 128)   ? Fp8Cfg<128, 64, 32>::lse_smem_bytes
                       : (D == 256) ? Fp8Cfg<256, 64, 32>::lse_smem_bytes
                                    : Fp8Cfg<512, 64, 32>::lse_smem_bytes;

  printf("case = %s\n", dir.c_str());
  printf("B=%d S=%d H=%d Hkv=%d D=%d causal=%d scale=%.6f\n", B, S, H, Hkv, D, (int)causal,
         scale);
  printf("FP8 mma: Q/K/V=E4M3, dO=E5M2, dS2/dS3=E5M2, Ap=E4M3 (rowwise); P/dS fp32\n");
  printf("smem = %d bytes (%.1f KB); lse smem = %d bytes (%.1f KB)\n", smem_bytes,
         smem_bytes / 1024.0, lse_smem, lse_smem / 1024.0);
  if (D == 128)
    printf("main wgmma smem = %d bytes (%.1f KB)\n",
           Fp8Cfg<128, 64, 32>::smem_bytes_wgmma, Fp8Cfg<128, 64, 32>::smem_bytes_wgmma / 1024.0);

  float *d_q_f, *d_k_f, *d_v_f, *d_do_f, *d_o_f;
  unsigned char *d_q8, *d_k8, *d_v8, *d_do8;
  float *d_qs, *d_ks, *d_vs, *d_dos;
  float *d_delta, *d_lse, *d_dq_acc, *d_dk_acc, *d_dv_acc, *d_dq, *d_dk, *d_dv;
  CUDA_CHECK(cudaMalloc(&d_q_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_k_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_v_f, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_do_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_o_f, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_q8, nq));
  CUDA_CHECK(cudaMalloc(&d_k8, nkv));
  CUDA_CHECK(cudaMalloc(&d_v8, nkv));
  CUDA_CHECK(cudaMalloc(&d_do8, nq));
  CUDA_CHECK(cudaMalloc(&d_qs, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_ks, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_vs, rows_kv * 4));
  CUDA_CHECK(cudaMalloc(&d_dos, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_delta, rows_q * 4));
  CUDA_CHECK(cudaMalloc(&d_lse, rows_q * 4));
  // O38：LSE split 的部分结果缓冲（[rows_q][ksplit] 个 (m,l)）。按最大 split=16 预留。
  float* d_lse_part = nullptr;
  CUDA_CHECK(cudaMalloc(&d_lse_part, rows_q * 16 * 2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_dq_acc, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk_acc, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv_acc, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dq, nq * 4));
  CUDA_CHECK(cudaMalloc(&d_dk, nkv * 4));
  CUDA_CHECK(cudaMalloc(&d_dv, nkv * 4));
  // O21b：输出本就是 fp32，与 fp32 累加缓冲同 dtype ⇒ 让 main 的 atomicAdd 直接累加到输出，
  //   消掉 `convert_kernel` 这一趟纯 fp32→fp32 拷贝（S=4096 约 0.145ms / 端到端 ~5%）。
  //   把 acc 指针别名到输出即可（所有自测 A/B 仍读写同一份，结果不变）。
  CUDA_CHECK(cudaFree(d_dq_acc)); d_dq_acc = d_dq;
  CUDA_CHECK(cudaFree(d_dk_acc)); d_dk_acc = d_dk;
  CUDA_CHECK(cudaFree(d_dv_acc)); d_dv_acc = d_dv;

  // O89：定长 causal 的 LPT m 块调度序（`--mrev=1`），与两文件版 `fa_bwd_fp8_main.cu` 同源。
  int* d_mrev = nullptr;
  // O104（第 198 轮）：把 O93 的跨 head 全局 LPT（hswap）从 D=128 推广到 **D=256 的通用 wgmma
  //   主 kernel**。实测（本卡，见 docs/03 §126）：hswap 对 D=256 的收益**只在小 `base_grid`
  //   （=`nblk*H*B ≤ 256`，即 K/V 工作集小、hswap 损 L2 局部性可忽略）时为正**——
  //   S1024H8 1.15×、S1024H16kv4 1.09×、S2048H8 1.02×；而 S2048H16（base=512）、S4096H8
  //   （base=512）为负（0.95×/0.84×）。故只在 `base_grid ≤ 256` 且 causal 定长 nblk≥16 时启用。
  // O105（第 199 轮）：把 O89 的 **per-head LPT m 块反转**（`mblk = nblk-1-mt`，grid 与 head
  //   排布都不变）从 D=128 推广到 **D=512（MLA）causal 定长**。对照实验：D=256 大 `base_grid`
  //   （S2048H16/S4096H8）的 mrev 仅 1.003×（噪声内），故**不纳入 D=256 默认**（其小 base_grid
  //   档已由 O104 的 hswap256 覆盖）；D=512 的 S1024H2（nblk=16，1 CTA/SM、`target=S/2` 切得细）
  //   实测 main **1.068×**（反复重测稳定），是 O104 之后 MLA 主 kernel 的第一条调度正收益。
  //   只改「哪个 CTA 算哪个 m 块」，dK/dV 的跨 CTA 原子次序略变 ⇒ 数值仍在 fp8 噪声内。
  //   `--mrev=0` 回退历史（mt 升序、便宜块先跑）做同 binary A/B。
  const bool mrev_elig = (D == 128) || (D == 512) ||
                         (D == 256 && hswap_opt != 0 &&
                          (long)((S + 63) / 64) * H * B <= 256);
  if (mrev_opt && causal && mrev_elig && (S + 63) / 64 >= 16) {  // O89：nblk>=16（S>=1024）才启用，避免小 S 噪声
    const int nblk_m = (S + 63) / 64;
    std::vector<int> hrev(nblk_m);
    for (int i = 0; i < nblk_m; ++i) hrev[i] = nblk_m - 1 - i;
    CUDA_CHECK(cudaMalloc(&d_mrev, nblk_m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_mrev, hrev.data(), nblk_m * sizeof(int), cudaMemcpyHostToDevice));
    printf("O89: mrev on (nblk=%d, LPT expensive-first)\n", nblk_m);
  } else if (mrev_opt) {
    printf("O89/O105: mrev requested but ignored (need causal & D==128/256/512 & fixed-length & nblk>=16)\n");
  }
  // O93：跨 head 全局 LPT（轴对调）的启用条件（与 launch 分支一致）。仅 Hopper 构建生效。
  const bool hswap_elig = (hswap_opt != 0) && (mrev_opt != 0) && causal && D == 128 &&
                          (S + 63) / 64 >= 16;
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (hswap_elig)
    printf("O93: hswap on (grid=(H,nblk*ksplit,B), global LPT across heads)\n");
  else if (hswap_opt && D == 128)
    printf("O93: hswap requested but ignored (need causal & D==128 & nblk>=16)\n");
#else
  if (hswap_opt && D == 128) printf("O93: hswap requested but ignored (非 Hopper 构建)\n");
#endif
  // O104（第 198 轮）：把 O93 的跨 head 全局 LPT 从 `kvtma`（D=128）推广到**通用 wgmma 主 kernel**
  //   （D=256 默认档）。只需 `-DFA_WGMMA`（不需 TMA——D=256 走 cp.async）；`--hswap=0` 关闭。
  const bool hswap256_elig = (hswap_opt != 0) && (mrev_opt != 0) && causal && D == 256 &&
                             (S + 63) / 64 >= 16 &&
                             (long)((S + 63) / 64) * H * B <= 256;
#if defined(FA_WGMMA)
  if (hswap256_elig)
    printf("O104: hswap256 on (D=256, grid=(H,nblk*ksplit,B), global LPT across heads)\n");
  else if (hswap_opt && D == 256 && (long)((S + 63) / 64) * H * B > 256)
    printf("O104: hswap256 skipped for D=256 (base_grid=%ld > 256: L2 locality loss > LPT gain)\n",
           (long)((S + 63) / 64) * H * B);
#else
  if (hswap_opt && D == 256) printf("O104: hswap256 requested but ignored (非 Hopper 构建)\n");
#endif
  // O95：变 ks 调度表（`--ksm=N`）。仅 hswap eligible 时构建；表按 mt 升序（= 全局 LPT）排列，
  //   前 N 个（最贵）m 块 ks=2、其余 ks=1。`slot_tab` 由主 kernel 在 HSWAP 路径消费。
  int* d_ksm = nullptr;
  int ksm_nslots = 0;
  if (hswap_elig && ksm_opt >= 0) {
    const int nblk_m = (S + 63) / 64;
    int n2 = ksm_opt;
    if (n2 > nblk_m) n2 = nblk_m;
    if (n2 < 0) n2 = 0;
    std::vector<int> tab;
    tab.reserve((size_t)nblk_m + n2);
    const int ks_hi = (ksm_hi < 1) ? 1 : ((ksm_hi > 15) ? 15 : ksm_hi);
    for (int mt = 0; mt < nblk_m; ++mt) {
      const int ks_i = (mt < n2) ? ks_hi : 1;
      for (int part = 0; part < ks_i; ++part)
        tab.push_back((mt << 8) | (part << 4) | ks_i);
    }
    ksm_nslots = (int)tab.size();
    CUDA_CHECK(cudaMalloc(&d_ksm, ksm_nslots * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_ksm, tab.data(), ksm_nslots * sizeof(int), cudaMemcpyHostToDevice));
    printf("O95: ksm on (top %d/%d m-blocks ks=%d, rest ks=1; nslots=%d vs uniform ks=2 %d)\n", n2,
           nblk_m, ks_hi, ksm_nslots, 2 * nblk_m);
  } else if (ksm_opt >= 0) {
    printf("O95: ksm requested but ignored (need hswap-eligible: causal & D==128 & nblk>=16)\n");
  }

  CUDA_CHECK(cudaMemcpy(d_q_f, q_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_k_f, k_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v_f, v_np.data.data(), nkv * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_do_f, do_np.data.data(), nq * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_o_f, o_np.data.data(), nq * 4, cudaMemcpyHostToDevice));

  // O14：量化输入（q/k/v=E4M3、dO=E5M2，rowwise）。`qfast` 选 warp-per-row 向量化版。
  auto quant_old = [&]() {
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_q_f, d_q8, d_qs, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_k_f, d_k8, d_ks, D, 0);
    quantize_row_kernel<<<(int)rows_kv, 128>>>(d_v_f, d_v8, d_vs, D, 0);
    quantize_row_kernel<<<(int)rows_q, 128>>>(d_do_f, d_do8, d_dos, D, 1);
  };
  auto quant_new = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const int gq = (int)std::min<long long>((rq + 3) / 4, 65535);
    const int gkv = (int)std::min<long long>((rkv + 3) / 4, 65535);
    if (D == 128) {
      quantize_row_warp_kernel<4, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<4, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<4, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    } else if (D == 256) {
      quantize_row_warp_kernel<8, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<8, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<8, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    } else {
      quantize_row_warp_kernel<16, false><<<gq, 128>>>(d_q_f, d_q8, d_qs, rq);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_k_f, d_k8, d_ks, rkv);
      quantize_row_warp_kernel<16, false><<<gkv, 128>>>(d_v_f, d_v8, d_vs, rkv);
      quantize_row_warp_kernel<16, true><<<gq, 128>>>(d_do_f, d_do8, d_dos, rq);
    }
  };
  auto quant = [&]() { if (qfast) quant_new(); else quant_old(); };
  // O64：融合「4 次量化 + 3 次清零」的单一 launch（数值与 quant()+3×memset 逐位相同）。
  auto quant_zero = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long total = 3 * rq + 4 * rkv;
    int grid = (int)std::min<long long>((total + 3) / 4, 1048576);
    if (qcap_opt > 0) grid = std::min(grid, qcap_opt);  // O88：栅格封顶（A/B）
    if (D == 128)
      quantize_zero_warp_kernel<4><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                  d_dk_acc, d_dv_acc, rq, rkv);
    else if (D == 256)
      quantize_zero_warp_kernel<8><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                  d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                  d_dk_acc, d_dv_acc, rq, rkv);
    else
      quantize_zero_warp_kernel<16><<<grid, 128>>>(d_q_f, d_k_f, d_v_f, d_do_f, d_q8, d_k8, d_v8,
                                                   d_do8, d_qs, d_ks, d_vs, d_dos, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, rq, rkv);
  };
  // O66：在 O64 融合的基础上，把 delta 也算进 dO 的量化任务（省一次独立 delta launch）。
  auto quant_zero_delta = [&]() {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long total = 3 * rq + 4 * rkv;
    int grid = (int)std::min<long long>((total + 3) / 4, 1048576);
    if (qcap_opt > 0) grid = std::min(grid, qcap_opt);  // O88：栅格封顶（A/B）
    if (D == 128)
      quantize_zero_delta_warp_kernel<4><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
    else if (D == 256)
      quantize_zero_delta_warp_kernel<8><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
    else
      quantize_zero_delta_warp_kernel<16><<<grid, 128>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv);
  };
  // O109：分相量化（phase=0 只做 Q/K，phase=1 做 dO+delta/V/清零）。数值逐位同合并版。
  auto quant_phase = [&](int ph, cudaStream_t st) {
    const long long rq = (long long)rows_q, rkv = (long long)rows_kv;
    const long long ntask = (ph == 0) ? (rq + rkv) : (2 * rq + 3 * rkv);
    int grid = (int)std::min<long long>((ntask + 3) / 4, 1048576);
    // O110：phase1 栅格封顶（0=auto=ntask/64 clamp[132,4096]；>0 显式；<0 不封顶）。见两文件版。
    if (ph == 1) {
      long long cap;
      if (ovl_cap < 0) cap = grid;
      else if (ovl_cap > 0) cap = ovl_cap;
      else cap = std::min<long long>(std::max<long long>(132, ntask / 64), 4096);
      grid = (int)std::min<long long>(grid, cap);
    }
    if (D == 128)
      quantize_zero_delta_phase_kernel<4><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv, ph);
    else if (D == 256)
      quantize_zero_delta_phase_kernel<8><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv, ph);
    else
      quantize_zero_delta_phase_kernel<16><<<grid, 128, 0, st>>>(
          d_q_f, d_k_f, d_v_f, d_do_f, d_o_f, d_delta, d_q8, d_k8, d_v8, d_do8, d_qs, d_ks, d_vs,
          d_dos, d_dq_acc, d_dk_acc, d_dv_acc, rq, rkv, ph);
  };

  // ---- O2b：自动选择 N 方向切块数 ksplit。base = 未切块时的 CTA 数；切块把小 S 时
  //      不足一个波、或大 S 的尾波（partial wave）用更细的 CTA 补满并发槽。
  //      实测（docs/03 §15 的 ksplit sweep）：d128（smem 73.8KB→3 CTA/SM）在
  //      grid≈4096（≈10 个波）时最优；MLA d512（smem 223KB→1 CTA/SM）在 grid≈一个波
  //      （132）时最优——再切只增 prologue 与 dQ atomic 竞争。故按 head_dim 取目标：
  //        TARGET = (D==128) ? 4096 : 132;  k = clamp(TARGET/base, 1, 16) 后向下取 2 的幂。----
  constexpr int BM = 64;
  const long base_grid = (long)((S + BM - 1) / BM) * H * B;
  const bool ksplit_auto = (ksplit < 1);
  if (ksplit < 1) {
    // O29：自动切块数重新标定。原公式 `target=(D==128)?4096:132` 是早期（O2b）在
    //   「d128=3 CTA/SM、MLA=1 CTA/SM」下测的，随后的 O3/O4b/O9c-2/O22 等把数据通路改过之后
    //   已明显次优：
    //   * d128 固定 4096 在 base 小（S=1024，base=512）时**过切**——每个 CTA 只有 ~2 个 tile，
    //     且此时 `use_regdq=false`、dQ 逐 tile 跨 CTA red，多切反而慢。实测 S1024H32 auto k=8
    //     0.340 ms vs k=4 **0.302 ms（1.13×）**；GQA kv4 0.329→0.274（1.20×）。
    //   * S=4096 时 4096 又**欠切**（k=4 1.758 vs k=8 1.737，1.01×）。
    //   * MLA（D=512，1 CTA/SM）固定 132（≈1 个波）**欠切**——S1024H2 auto k=4 0.265 ms vs
    //     k=16 **0.201 ms（1.32×）**；S512H4 k=4 0.140 vs k=8 0.122。
    //   新标定（sweep 见 src/fp8/fa_bwd_fp8_o29_ksplit_sweep.out.txt）：
    //     D==128: S>=2048 → 8192；否则 max(2048, 4*base_grid)（覆盖 base∈{128,512,640,1024}）
    //     D==512: S/2（S256H2→128 / S512H4→256 / S1024H2→512，三者都取到各自最优）
    const long target_ctas = (D == 128)
                                 ? ((S >= 2048) ? 8192L : std::max(2048L, 4L * base_grid))
                                 : (long)(S / 2);
    long k = target_ctas / base_grid;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;  // 向下取 2 的幂，让 grid 对齐到整数个波附近
    ksplit = (int)kp;
  }
  // O96（第 190 轮）：full（非 causal）D=128 的 ksplit 重标定（与两文件版同源，详见
  //   `fa_bwd_fp8_main.cu` 同名注释 / docs/03 §118）。O29 的 target 是按 causal 三角偏斜标的，
  //   full 每块工作量相同 ⇒ 过细切分只剩 Q/dO 重读与 dQ 跨 part 原子。改为让 `grid=base*k`
  //   最接近整数个并发波（396 槽 = 3 CTA/SM × 132 SM），k∈[1,8]。显式 `--ksplit=K` 不覆盖。
  if (ksplit_auto && !causal && D == 128) {
    const long SLOTS = 396L;
    long best_k = 1, best_waste = -1;
    for (long k = 1; k <= 8; ++k) {
      const long g = base_grid * k;
      const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
      if (best_waste < 0 || waste < best_waste) {
        best_waste = waste;
        best_k = k;
      }
    }
    ksplit = (int)best_k;
  }
  // O97（第 191 轮）：full（非 causal）D=256 / D=512 的 ksplit 重标定（与两文件版同源，
  //   详见 `fa_bwd_fp8_main.cu` 同名注释 / docs/03 §119）。D=256/512 共用 O29 的 `target=S/2`
  //   （按 causal MLA 标定），对 full 不合适：D=512（1 CTA/SM→132 槽）base 小时过切到 k=16，
  //   波对齐修正（main 1.15×）；D=256（2 CTA/SM→264 槽）`S/2/base=32/H` 与 S 无关 ⇒ 大 S
  //   欠切，S≥2048 给足并发（grid≈8192，cap k=12），S<2048 按 264 槽波对齐。
  if (ksplit_auto && !causal && (D == 256 || D == 512)) {
    const bool wave = (D == 512) || (S < 2048);
    if (wave) {
      const long SLOTS = (D == 512) ? 132L : 264L;
      const long KMAX = (D == 512) ? 16 : 8;
      long best_k = 1, best_waste = -1;
      for (long k = 1; k <= KMAX; ++k) {
        const long g = base_grid * k;
        const long waste = ((g + SLOTS - 1) / SLOTS) * SLOTS - g;
        if (best_waste < 0 || waste < best_waste) {
          best_waste = waste;
          best_k = k;
        }
      }
      if (D == 512 && best_k < 2) best_k = 2;  // 见两文件版注释：消除大 base 的 k=1 回退
      ksplit = (int)best_k;
    } else {
      long k = 8192L / base_grid;
      if (k < 1) k = 1;
      if (k > 12) k = 12;
      ksplit = (int)k;
    }
  }
  // O99（第 193 轮）：**causal D=256 的 ksplit 重标定**（与两文件版同源，详见
  //   `fa_bwd_fp8_main.cu` 同名注释 / docs/03 §121）。`target=S/2`（causal MLA 标定）套到
  //   causal D=256（2 CTA/SM→264 槽）上 `k=32/(H*B)` 与 S 无关 ⇒ 小/中 S 欠切。实测 6 个
  //   causal D=256 shape 的最优都落在「`grid≈2*S`」（`k≈128/(H*B)`），小 S 由 `k≤nblk` 封顶。
  //   规则：`k = clamp(2*S/base,1,16)` 再按 `nblk` 封顶。`--ksplit=K` 显式时不覆盖。
  if (ksplit_auto && causal && D == 256) {
    const long nblk = (long)((S + 63) / 64);
    long k = (2L * S) / base_grid;
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    if (k > nblk) k = nblk;
    ksplit = (int)k;
  }
  // O112（第 206 轮）：**causal D=512（MLA）的 ksplit 重标定**（定长；O107 的姊妹审计，
  //   与两文件版同源，详见 `fa_bwd_fp8_main.cu` 同名注释 / docs/03 §134）。O29 的 `target=S/2`
  //   在 O51（K/V `cp.async` 回填流水）+ O105（mrev）之后已次优：S512H2/S1024H2 auto k=16 比
  //   k=8 慢 2.2–6.5%，S2048H2/S4096H2 慢 2.7–3.6%；最优稳定在 k≈8 ⇒ 定长沿用 O107 给 causal
  //   变长 D=512 的 `k=pow2floor(min(8, S/base_grid))`、下限 2。只改跨 CTA atomicAdd 次序。
  if (ksplit_auto && causal && D == 512) {
    long k = 1;
    while (k * 2 <= S / base_grid && k < 8) k *= 2;
    if (k < 2) k = 2;
    ksplit = (int)k;
  }
  // O93：hswap（跨 head 全局 LPT）启用时把自动 ksplit 收到 2（与两文件版同源）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (hswap_elig && ksplit_auto) ksplit = 2;
#endif
  // O104：D=256 hswap 把自动 ksplit 收到 **4**（实测最优；k=2 过细、并行度不足，k=8/16 又损
  //   Q/dO 重读 + 局部性）。`--ksplit=K` 显式给出时不覆盖。
#if defined(FA_WGMMA)
  if (hswap256_elig && ksplit_auto) ksplit = 4;
#endif
  // O7：只有 HD=128（dQ 一次铺满 N）且「平均每 CTA 的 nt tile 足够多」时才启用寄存器累加。
  // 因果下每 mblk 的 nt tile 数 ≈ (m0+BM)/BN，三角求和 /(mblk·ksplit) 后平均每 CTA
  // ≈ (S/BN)/2/ksplit；阈值取 4（实测 S=1024H32 平均=2、启用反而持平/略慢，S=4096=16 明显收益）。
  // O96：full 无三角折半 ⇒ 不 `/2`（否则 k 偏大时误关 regdq，dQ 逐 tile 跨 CTA `red`）。
  bool use_regdq = (D == 128) && ((long)(S / 32) / (causal ? 2 : 1) / ksplit >= 4);
  // O22：`--regdq=0/1` 强制开关（仅同 session A/B 用）；-1 = 用上面的启发式。
  if (regdq_opt >= 0) use_regdq = (D == 128) && (regdq_opt != 0);
  dim3 pg(S, H, B);
  // O26：delta 的 warp-per-row 版 grid-stride 覆盖 B*S*H 行（每 warp 一行）。
  const bool delta_warp_sel = (delta_warp_opt != 0);
  const int d_rows = (int)((size_t)B * S * H);
  const int d_wpb = THREADS / 32;
  const int d_blocks = (d_rows + d_wpb - 1) / d_wpb;
  dim3 lg((S + LBM - 1) / LBM, H, B);
  dim3 lg_bal((((S + LBM - 1) / LBM) + 1) / 2, H, B);   // O11：镜像配对，grid.x 减半
  dim3 mg((S + BM - 1) / BM * ksplit, H, B);
  // O19：wg2（BM=128）的自动 ksplit 与 grid（base 减半）。同样以 grid≈4096（3 CTA/SM
  //   的 d128 目标）为准；1 CTA/SM 时用更细的 grid 不利，故对 wg2 用 target=2048。
  const long base_grid2 = (long)((S + 127) / 128) * H * B;
  int ksplit2 = ksplit;
  if (ksplit2 < 1) ksplit2 = 1;
  if (ksplit2_opt >= 1) ksplit2 = ksplit2_opt;
  else {
    const long target_ctas = 2048L;
    long k = target_ctas / (base_grid2 > 0 ? base_grid2 : 1);
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;
    ksplit2 = (int)kp;
  }
  dim3 mg2((S + 127) / 128 * ksplit2, H, B);
  // O91（F6-step4）：wg3（BM=192、3 warpgroup、384 线程、1 CTA/SM）的 grid。
  int ksplit3 = 4;
  if (ksplit3_opt >= 1) ksplit3 = ksplit3_opt;
  else {
    const long base_grid3 = (long)((S + 191) / 192) * H * B;
    long k = 4096L / (base_grid3 > 0 ? base_grid3 : 1);
    if (k < 1) k = 1;
    if (k > 16) k = 16;
    long kp = 1;
    while (kp * 2 <= k) kp *= 2;
    ksplit3 = (int)kp;
  }
  dim3 mg3((S + 191) / 192 * ksplit3, H, B);
  printf("grid main = %d x %d x %d  (ksplit=%d, base_grid=%ld)\n", mg.x, mg.y, mg.z,
         ksplit, base_grid);
  printf("O19: wg2 grid = %d x %d x %d (ksplit2=%d, base_grid2=%ld)\n", mg2.x, mg2.y, mg2.z,
         ksplit2, base_grid2);
  printf("O7: use_regdq=%d (register dQ accumulation)\n", (int)use_regdq);
  const int cvt_threads = 256;
  const int cvt_blocks = (int)std::min<size_t>((nq + cvt_threads - 1) / cvt_threads, 65535);

  // O32：TMA 版 LSE 需驱动 API（`cuTensorMapEncodeTiled`）⇒ 只有 `-DFA_TMA -lcuda` 构建才编译
  //   该路径；此时 D==128/causal 默认开（对齐 fp16 O30，省 load 指令/地址运算）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (D == 128) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<128, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<128, 64, 32>::lse_smem_bytes_tma1));
  }
  // O74（第 169 轮）：MLA（D=512）的 causal LSE 也上 4D-TMA（此前只有 mma/cp.async 版）。
  if (D == 512) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<512, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<512, 64, 32>::lse_smem_bytes_tma1));
  }
  // O103（第 197 轮）：head_dim=256 的 LSE 也上 4D-TMA（kernel 已在 O74 泛化为 NCH=HD/128，
  //   此前只差 host 未接 D=256）。host 建 D=256 的 Q/K 描述符 + 设 `lse_smem_bytes_tma1`。
  if (D == 256) {
    CUDA_CHECK(cudaFuncSetAttribute(lse_mma_kernel_bal_tma<256, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    Fp8Cfg<256, 64, 32>::lse_smem_bytes_tma1));
  }
  if (lse_tma < 0) lse_tma = (D == 128 || D == 256) ? 1 : (D == 512 && causal ? 1 : 0);
#else
  if (lse_tma < 0) lse_tma = 0;
#endif
  printf("O32: lse backend = %s\n", lse_tma ? "tma" : "wgmma/mma");
  // O38：自动 split 档（0=auto）。目标 `grid*split ≈ 2048`（≈2 个满波；实测该目标在各 shape
  //   上距 per-shape 最优 ≤1.2%），上限 8；grid 已够大则退回 1（逐位）。
  // O59：把 O58 的 causal MLA LSE 几何（cfg6 默认，2 CTA/SM）从 varlen 推广到**定长**路径。
  //   O58 只改了 `run_varlen`，定长 MLA 的 causal LSE 一直用旧默认 `<512,1>`（PIPE1/LBN64）。
  //   以 `--lseocc=4` 退回旧默认、5=PIPE0/LBN32、6=PIPE1/LBN16；`--lse8w=1` 在定长不启用。
  const bool lse8w_fixed = (lse8w_opt != 0) && (D == 512);
  const bool causal_cfg6_fixed =
      causal && (D == 512) &&
      (lseocc_opt == 5 || lseocc_opt == 6 || (lseocc_opt == 0 && !lse8w_fixed));
  int lse_split_eff = lse_split;
  if (lse_split_eff <= 0) {
    // O101（第 195 轮）：base 随 causal/full 取不同网格——causal 走镜像配对（`lg_bal.x`），
    //   full 一个 CTA 一个 m 块（`lg.x=nblk`）。此前 full 也误用 `lg_bal.x`（只有 causal 的半格），
    //   对新增了 split 的 full D=256/D=512 会把并发目标低估一半。
    long lg_grid = (long)(causal ? lg_bal.x : lg.x) * H * B;
    // O39：D=512（MLA，mma LSE）目标 `grid*split ≈ 256`、上限 16；D=128 的 TMA LSE 维持
    //   O38 的 `≈2048`、上限 8。O59：cfg6 的 4 CTA/SM 把并发槽翻倍，目标抬到 1024
    //   （对齐 O58 varlen 的 fp8 档）。
    // O63（第 138 轮）：D=128 的 TMA LSE 在大 S 上原目标 2048 会比实测最优**多切一档**——
    //   S=4096 H16 时 base=`lg_grid`=512，2048→split=4，而同 binary 交替实测 split=2 的
    //   LSE 快 ~2.7%（端到端在噪声内）。S≥2048 时把目标降到 1024（仍 ≈2 个满波量级）；
    //   S<2048 维持 2048，避免小 shape 的二次归约/尾部回归（S512/S1024H32 实测两档相同）。
    // O101：full D=512 新接 split，目标 **256**；full D=256 新接 split，目标 **512**
    //   （实测见两文件 `fa_bwd_fp8_main.cu` 同段注释 / docs/03 §123）。causal 逐字不变。
    const int target = (D == 512) ? (causal ? (causal_cfg6_fixed ? 1024 : 256) : 256)
                                  : (causal ? (S >= 2048 ? 1024 : 2048) : 512);
    const int cap = (D == 512) ? 16 : 8;
    int sp = 1;
    while (sp < cap && lg_grid * (sp * 2) <= target) sp *= 2;
    // 再按 `nblk=ceil(S/64)` 封顶：切得比 K tile 数还细只会产生空切片 + 每 CTA 的 Q 载入开销。
    int nblk_cap = (S + 63) / 64;
    while (sp > nblk_cap) sp >>= 1;
    lse_split_eff = sp;
  }
#if defined(FA_WGMMA) && defined(FA_TMA)
  // O32：建 LSE 的 Q/K 4D TMA 描述符（一次，供所有 (h,b) CTA 用坐标选择）。
  CUtensorMap qmap_lse, kmap_lse;
  // 注：O32/O9c A/B 段无论 `--lsetma` 取值都会跑 TMA 版 LSE，故 D==128 时始终建描述符
  //     （否则 `--lsetma=0` 会拿未初始化 map 启动 TMA kernel → illegal instruction）。
  if (D == 128 || D == 256 || D == 512) {  // O74/O103：MLA（512）causal、D=256 LSE 需 Q/K 描述符
    qmap_lse = make_lse_map_fp8(d_q8, H, S, D, B);
    kmap_lse = make_lse_map_fp8(d_k8, Hkv, S, D, B);
  }
  // O37：主 kernel 的 Q/dO TMA 描述符（box={128,BM=64}，与 LSE 同一 dims={D,S,H,B}）。
  if (qd_tma < 0) qd_tma = 1;   // 默认开（对齐 O32 的 lsetma；`--qdtma=0` 供 A/B）
  // O41：K/V TMA 默认开（依赖 Q/dO TMA；`--kvtma=0` 供 A/B）。
  if (kv_tma < 0) kv_tma = 1;
  CUtensorMap qmap_main, dmap_main;
  // 注：O37 A/B 段无论 `--qdtma` 取值都会跑 TMA 版，故 D==128/256 时始终建描述符
  //     （否则 `--qdtma=0` 会拿未初始化 map 启动 TMA kernel → illegal instruction）。
  // O85：D=256 的 Q/dO 也走 4D-TMA（chunk-major，NCH=2），同样需要描述符。
  if (D == 128 || D == 256) {
    qmap_main = make_lse_map_fp8(d_q8, H, S, D, B);
    dmap_main = make_lse_map_fp8(d_do8, H, S, D, B);
  }
  // O41：主 kernel 的 K/V TMA 描述符（dims={D,S,Hkv,B}，box={128,BN=32}）。
  CUtensorMap kmap_main, vmap_main;
  if (D == 128) {
    kmap_main = make_lse_map_fp8(d_k8, Hkv, S, D, B, 32);
    vmap_main = make_lse_map_fp8(d_v8, Hkv, S, D, B, 32);
  }
#else
  if (qd_tma < 0) qd_tma = 0;
  kv_tma = 0;
#endif
  printf("O37: main qd-tma = %s\n", qd_tma ? "on" : "off");
  printf("O41: main kv-tma = %s\n", kv_tma ? "on" : "off");  printf("O38: lse k-split = auto(%d)\n", lse_split_eff);

  // O119：multicast 宽度解析 + 形状门控（同 `fa_bwd_fp8_main.cu`）。
#if defined(FA_WGMMA) && defined(FA_TMA)
  if (mcast_opt != 1 && hswap_elig && kv_tma && qd_tma && D == 128 && causal) {
    const int G = H / Hkv;
    int W = 1;
    if (mcast_opt > 1) W = mcast_opt;
    else { while (W * 2 <= G && W * 2 <= 8) W *= 2; }
    if (W > 1 && W <= 8 && (W & (W - 1)) == 0 && (H % W) == 0 && (G % W) == 0) {
      g_mcast_width = W;
      printf("O119: mcast width=%d (G=%d H=%d Hkv=%d)\n", W, G, H, Hkv);
    } else {
      printf("O119: mcast requested but ineligible (G=%d H=%d W=%d)\n", G, H, W);
    }
  } else if (mcast_opt != 1) {
    printf("O119: mcast requested but conditions unmet (D=%d causal=%d hswap=%d kv_tma=%d)\n", D,
           (int)causal, (int)hswap_elig, kv_tma);
  }
#endif

  // O66：`do_delta=false` 时跳过 delta（融合路径已在 quant kernel 内算好）。
  auto run_preprocess = [&](bool do_delta = true) {
    if (D == 128) {
      // O11：causal 走镜像配对 + cp.async 双缓冲（非 causal 各块工作量相同，走 O1 原版）。
      // O9c：`--lsewgm` 且以 `-DFA_WGMMA` 构建时，causal 走 wgmma.m64n64k32（SW128）。
      if (causal) {
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          // O38：split>1 走「K 维切片 + merge」，=1 逐位退回 O32。
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H,
                                       Hkv, scale);
        } else
#endif
#ifdef FA_WGMMA
        if (lsewgm)
          launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                       scale, nullptr, nullptr, nullptr, d_lse_part,
                                       lse_split_eff);
        else
#endif
          launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                 nullptr, d_lse_part, lse_split_eff);
      } else if (g_lse_full_opt) {
        // O68（第 162 轮）：非 causal D=128 的 LSE 从 O1 `lse_mma_kernel`（无 cp.async 流水）
        //   改走 O54 的均衡版 `lse_mma_kernel_bal<FULL=true>`——一个 CTA 一个 m 块、
        //   K 用 `cp.async` 16B 双缓冲。full 各 m 块工作量相同（均 nblk 个 tile）无需镜像配对。
        // O70（第 164 轮）：full D=128 定长**优先走 4D-TMA**（对齐 causal O32）；`--lsetma=0`
        //   退回 O68 的 cp.async 均衡版做同 binary A/B；`--lsefull=0` 仍退回 O1。
        bool did_tma = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          launch_lse_bal_tma<128, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                           scale);
          did_tma = true;
        }
#endif
        if (!did_tma)
          launch_lse_bal<128, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, nullptr, 1);
      } else
        launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    } else if (D == 256) {
      // O76（第 171 轮）：head_dim=256 的 LSE 走 O11 镜像配对 mma 版 / 均衡 FULL 版
      //   （`lse_mma_kernel_bal` 对 HD 是模板参数，HD=256 直接复用）。
      // O103（第 197 轮）：优先走 4D-TMA（对齐 D=128 O32 / D=512 O74）；`--lsetma=0` 退回
      //   O76/O101 的 mma/cp.async 版做同 binary A/B。
      bool did_tma256 = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
      if (lse_tma) {
        if (causal) {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<256, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<256, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                       scale);
        } else {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<256, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                                   d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<256, 1, true>(lg, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                             scale);
        }
        did_tma256 = true;
      }
#endif
      if (!did_tma256) {
        if (causal)
          launch_lse_bal<256, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, lse_split_eff);
        else
          // O101（第 195 轮）：full D=256 的均衡 LSE 也接上 K 维 split（此前恒 split=1）。
          launch_lse_bal<256, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, d_lse_part, lse_split_eff);
      }
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    } else {
      // O39：D=512（MLA）causal LSE 也用 K 维 split（此前只有 D=128/TMA 有）。
      // O59：定长 causal MLA 默认走 O58 的 cfg6（PIPE1/LBN16，4 CTA/SM）；`--lseocc=4` 退回旧默认。
      if (causal) {
        // O74（第 169 轮）：MLA causal LSE 优先走 4D-TMA（`--lsetma=0` 退回 mma/cp.async A/B）。
        bool did_tma512 = false;
#if defined(FA_WGMMA) && defined(FA_TMA)
        if (lse_tma) {
          if (lse_split_eff > 1)
            launch_lse_bal_tma_split<512, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                             d_lse_part, S, H, Hkv, scale, lse_split_eff);
          else
            launch_lse_bal_tma<512, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                       scale);
          did_tma512 = true;
        }
#endif
        if (did_tma512) {
          // 已走 4D-TMA
        } else if (lseocc_opt == 5)
          launch_lse_bal<512, 0, false, 128, 32>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, lse_split_eff);
        else if (lseocc_opt == 6 || causal_cfg6_fixed)
          launch_lse_bal<512, 1, false, 128, 16>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, lse_split_eff);
        else
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, lse_split_eff);
      } else if (g_lse_full512_opt) {
        // O101（第 195 轮）：定长 full MLA（D=512）的 LSE 补齐均衡 + cp.async + K 维 split
        //   （对齐 O54 只修了的 varlen full）。`--lse512old=1` 退回 O1 做同 binary A/B。
        if (lse_split_eff > 1)
          launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, d_lse_part, lse_split_eff);
        else
          launch_lse_bal<512, 1, true>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                                       nullptr, nullptr, 1);
      } else
        launch_lse<512>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, (int)causal);
      if (do_delta) {
        if (delta_warp_sel)
          delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        else
          delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
      }
    }
  };

  // O12：LSE/D 预装寄存器（默认开），`--prel=0` 关；为同 session A/B 派发到两个模板实例。
  const bool prel_sel = (prel_opt < 0) ? true : (prel_opt != 0);
  // O7e-2：fold 16B 向量化写（默认开），`--f16b=0` 退回 O7e 的 4B 写（仅作 A/B）。
  const bool f16b_sel = (f16b_opt != 0);
  // O47：MLA（D=512）8-warp 几何（默认开；`--mla8w=0` 退回 4-warp A/B）。
  const bool mla8w_sel = (mla8w_opt < 0) ? true : (mla8w_opt != 0);
  // O48/O49：D=128 mma 路径的 8-warp 几何（opt-in / 自动）。默认 4-warp 与历史逐字相同。
  //   O49 自动档判据 = 「mma 后端（!wgmma）且有效 grid ≤ SM 数」；`--d128w=1` 仍可强制。
  int sm_count = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
  const long fp8_grid = (long)mg.x * mg.y * mg.z;
  const bool d128w_sel =
      (d128w_opt > 0) ? true
                      : ((d128w_opt < 0) ? (D == 128 && !wgmma && fp8_grid <= sm_count)
                                         : false);
  if (D == 128) printf("[O49] d128 8-warp = %d (d128w=%d, grid=%ld, sm=%d, wgmma=%d)\n",
                       (int)d128w_sel, d128w_opt, fp8_grid, sm_count, wgmma);
  // O27：第 5 个开关 rcp 选 fold 量化用乘法（true，默认）还是精确除法（false，A/B）。
  auto launch128 = [&](bool reg, bool wg, bool prel, bool f16, bool rcp = true) {
#define GO2(REG_, WG_, PREL_, F16_)                                                          \
    if (rcp) {                                                                               \
      launch_bwd_main<128, 64, 32, REG_, WG_, PREL_, F16_, true>(                            \
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,    \
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);                         \
    } else {                                                                                 \
      launch_bwd_main<128, 64, 32, REG_, WG_, PREL_, F16_, false>(                           \
          mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,    \
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);                         \
    }
#define GO1(REG_, WG_, PREL_)                                                                \
    if (f16) { GO2(REG_, WG_, PREL_, true); } else { GO2(REG_, WG_, PREL_, false); }
#define GO0(REG_, WG_)                                                                       \
    if (prel) { GO1(REG_, WG_, true); } else { GO1(REG_, WG_, false); }
    if (reg) { if (wg) { GO0(true, true); } else { GO0(true, false); } }
    else     { if (wg) { GO0(false, true); } else { GO0(false, false); } }
#undef GO0
#undef GO1
#undef GO2
  };
  auto run_main = [&]() {
    // O76（第 171 轮）：head_dim=256 走 mma 主 kernel（无 fp8 wgmma/TMA）。
    // O84（第 179 轮）：放开 WGMMA——`-DFA_WGMMA` 构建下 D=256 默认走 wgmma 主 kernel
    //   （GEMM1/2 wgmma + SW128；K/V/dO 仍 cp.async 载入，非 TMA）。`--d256wgm=0` 退回 mma A/B。
    if (D == 256) {
#ifdef FA_WGMMA
      const bool d256_wg = (d256wgm_opt < 0) ? true : (d256wgm_opt != 0);
#if defined(FA_TMA)
      // O85（第 180 轮）：D=256 的 Q/dO 4D-TMA（chunk-major，NCH=2；K/V 仍 cp.async）。
      //   **A/B 判决为中性（±2%，见 docs/03 §108）⇒ 默认关**，`--d256tma=1` 复现。
      const bool d256_tma = (d256tma_opt > 0);
      if (d256_wg && d256_tma) {
        launch_bwd_main_qdtma<256, 64, 32, false>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        return;
      }
#endif
      if (d256_wg) {
        // O104：hswap256（跨 head 全局 LPT）——head 走快轴，grid=(H, nblk*ksplit, B)，配 O89 的
        //   `d_mrev`（LPT 反转表）⇒ 所有 head 的最贵 m 块一起先跑。只改调度，dK/dV 仍是可交换
        //   的跨 CTA 原子 ⇒ 数值只在 fp8 噪声内。`--hswap=0` 回退历史锯齿序。
        if (hswap256_elig) {
          dim3 mgh256(H, mg.x, mg.z);
          launch_bwd_main<256, 64, 32, false, true, true, true, true, THREADS, WN, true>(
              mgh256, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
          return;
        }
        // O105：大 base_grid（hswap256 跳过）时也透传 O89 的 per-head LPT 反转表 `d_mrev`
        //   （HSWAP=false：grid/head 排布不变，只有 m 块贵先跑）。`--mrev=0` 时 d_mrev=nullptr。
        launch_bwd_main<256, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                                  d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
                                                  d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit,
                                                  nullptr, nullptr, d_mrev);
        return;
      }
#endif
      launch_bwd_main<256, 64, 32, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
                                          d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                          scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      return;
    }
    if (D == 128 && wg3) {
#ifdef FA_WGMMA
      launch_bwd_wgmma_nw<128, 192, 32, 3>(mg3, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                           d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                           S, H, Hkv, scale, (int)causal, ksplit3);
      return;
#else
      fprintf(stderr, "--wg3 需要 -DFA_WGMMA（sm_90a）构建\n");
      std::exit(2);
#endif
    }
    if (D == 128 && wg2wgmma) {
#ifdef FA_WGMMA
      launch_bwd_wgmma2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                             d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                             ksplit2);
      return;
#else
      fprintf(stderr, "--wg2wgmma 需要 -DFA_WGMMA（sm_90a）构建\n");
      std::exit(2);
#endif
    }
    if (D == 128 && wg2) {
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
      return;
    }
#if defined(FA_WGMMA) && defined(FA_TMA)
    if (D == 128 && wg2tma) {
      // O90（F6-step3）：wgmma2（BM=128, 2 warpgroup）+ K/V 4D-TMA（K 双缓冲）。
      launch_bwd_wgmma2tma<128>(mg2, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                Hkv, scale, (int)causal, ksplit2);
      return;
    }
    if (D == 128 && wg2ws) {
      // O121（F3b 主体化）：wgmma2 + K/V 4D-TMA + warp specialization（384 线程）。
      launch_bwd_wgmma2ws<128>(mg2, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                               d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                               Hkv, scale, (int)causal, ksplit2);
      return;
    }
#endif
    if (D == 128 && d128w_sel) {
      // O48（候选 ①）：D=128 主 kernel 的 8-warp（256 线程 / 2×4 网格）几何。仅 mma 后端
      //   （`WGMMA=true` 的 GEMM1/2 是 warpgroup 级、static_assert 锁死 2 warp）。BN 可 32/64；
      //   `PREL/F16B/RCP` 取默认档（A/B 只关心 warp 几何）。默认 4-warp 路径不受影响。
      if (bn64_opt) {
        if (use_regdq)
          launch_bwd_main<128, 64, 64, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 64, false, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
      return;
    }
    if (D == 128 && bn64_opt) {
      // O21：KV tile BN=64（mma 路径），其余与 BN=32 版逐字同构。grid 不变（BM 仍 64）。
      if (use_regdq)
        launch_bwd_main<128, 64, 64, true, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<128, 64, 64, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      return;
    }
    const bool rcp_sel = (foldrcp_opt != 0);
#if defined(FA_WGMMA) && defined(FA_TMA)
    // O41：Q/dO/K/V 全 TMA 版（K 双缓冲、V 单缓冲）。仅默认 fold 选项下启用。
    if (D == 128 && wgmma && qd_tma && kv_tma && prel_sel && f16b_sel && rcp_sel) {
      // O93：跨 head 全局 LPT（grid 轴对调，head 走快轴）。仅当 O89 的 mrev 表已建时启用。
      if (hswap_opt && d_mrev) {
        // O95：`--ksm` 时 y 轴长度改为变-ks 槽位数，并把 slot_tab 传给主 kernel（消费 HSWAP 路径）。
        dim3 mgh(H, d_ksm ? (unsigned)ksm_nslots : mg.x, mg.z);
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true, true, true, true, true>(
              mgh, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
              d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
              scale, (int)causal, ksplit, d_mrev, d_ksm);
        else
          launch_bwd_main_kvtma<128, 64, 32, false, true, true, true, true>(
              mgh, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
              d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
              scale, (int)causal, ksplit, d_mrev, d_ksm);
        return;
      }
      if (use_regdq)
        launch_bwd_main_kvtma<128, 64, 32, true>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit, d_mrev);
      else
        launch_bwd_main_kvtma<128, 64, 32, false>(
            mg, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
            d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
            scale, (int)causal, ksplit, d_mrev);
      return;
    }
    // O37：Q/dO TMA 版（仅在默认 fold 选项下启用；其它组合回退 cp.async 版）。
    if (D == 128 && wgmma && qd_tma && prel_sel && f16b_sel && rcp_sel) {
      if (use_regdq)
        launch_bwd_main_qdtma<128, 64, 32, true>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
            ksplit);
      else
        launch_bwd_main_qdtma<128, 64, 32, false>(
            mg, qmap_main, dmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos,
            d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
            ksplit);
      return;
    }
#endif
#ifdef FA_WGMMA
    if (D == 128 && wgmma) { launch128(use_regdq, true, prel_sel, f16b_sel, rcp_sel); return; }
#endif
    if (D == 128) { launch128(use_regdq, false, prel_sel, f16b_sel, rcp_sel); return; }
    // O47：MLA（D=512）默认走 8-warp/256 线程几何（`--mla8w=0` 退回 4-warp 供 A/B）。
    // O51：8-warp 档下 K/V 默认走 cp.async 回填流水（`--mlakvp=0` 退回同步载入 A/B）。
    // O105：定长 causal MLA 也透传 O89 的 per-head LPT 反转表 `d_mrev`（grid/head 排布不变）。
    const bool mla_kvp_sel = (mla_kvp_opt < 0) ? true : (mla_kvp_opt != 0);
    if (mla8w_sel) {
      if (prel_sel && mla_kvp_sel)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      else if (prel_sel)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
      else
        launch_bwd_main<512, 64, 32, false, false, false, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit, nullptr, nullptr, d_mrev);
    } else if (prel_sel)
      launch_bwd_main<512, 64, 32, false, false, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                       d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                       d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                       (int)causal, ksplit, nullptr, nullptr, d_mrev);
    else
      launch_bwd_main<512, 64, 32, false, false, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                        d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                        d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                        (int)causal, ksplit, nullptr, nullptr, d_mrev);
  };

  // O109：量化分相 + LSE 跨 stream 重叠（定长融合路径；`--ovlql=0` 关）。
  // O110（第 204 轮）：扩到 D=256 / D=512（同 `quantize_zero_delta_phase_kernel<VPT>`，VPT=D/32），
  //   用 `rows_q=B*S*H ≥ 2048` 门控，避免小 shape 上分相 launch/event 开销盖过重叠收益。
  //   D=128 保持 O109 无条件默认。
  const bool ovlql_req = ((D == 128) || ((D == 256 || D == 512) && rows_q >= 2048)) && qfuse &&
                         qfast && dfuse && delta_warp_sel && (qcap_opt <= 0);
  const bool ql_overlap = ovlql_req && (ovl_ql != 0);
  cudaStream_t sQuantAux = nullptr;
  cudaEvent_t ql_e0 = nullptr, ql_e1 = nullptr;
  if (ql_overlap) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&sQuantAux, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&ql_e0, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&ql_e1, cudaEventDisableTiming));
    printf("O109/O110: quant-phase overlap ON (Q/K -> LSE(default) || dO+V+zero(aux)), D=%d\n", D);
  } else if (ovl_ql == 1) {
    printf("O109: overlap requested but conditions unmet (D=%d qfuse=%d qfast=%d dfuse=%d) -> serial\n",
           D, qfuse, qfast, dfuse);
  }

  auto run_all = [&]() {
    if (ql_overlap) {
      // O109：Q/K 先量化（LSE 就绪）→ LSE(default) 与 dO+delta/V/清零(aux) 重叠 → main。
      quant_phase(0, nullptr);
      CUDA_CHECK(cudaEventRecord(ql_e0));
      CUDA_CHECK(cudaStreamWaitEvent(sQuantAux, ql_e0, 0));
      quant_phase(1, sQuantAux);
      run_preprocess(false);
      CUDA_CHECK(cudaEventRecord(ql_e1, sQuantAux));
      CUDA_CHECK(cudaStreamWaitEvent(nullptr, ql_e1, 0));
      run_main();
      if (cvt_on)
        convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                    d_dv, nq, nkv);
      return;
    }
    if (qfuse && qfast) {
      // O66：默认把 delta 也融进 quant kernel（`dfuse && delta_warp_sel`）。
      if (dfuse && delta_warp_sel)
        quant_zero_delta();
      else
        quant_zero();   // O64：4 次量化 + 3 次清零融合为 1 个 launch
    } else {
      quant();
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    }
    // O66：delta 已在 quant 阶段算好时，preprocess 跳过独立 delta launch。
    run_preprocess(!(qfuse && qfast && dfuse && delta_warp_sel));
    run_main();
    if (cvt_on)
      convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                  d_dv, nq, nkv);
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
  if (qfuse && qfast) {
    if (dfuse && delta_warp_sel)
      for (int i = 0; i < iters; ++i) quant_zero_delta();
    else
      for (int i = 0; i < iters; ++i) quant_zero();
  } else
    for (int i = 0; i < iters; ++i) quant();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_quant = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_quant, ev0, ev1));
  ms_quant /= iters;

  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i)
    run_preprocess(!(qfuse && qfast && dfuse && delta_warp_sel));
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_pre = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_pre, ev0, ev1));
  ms_pre /= iters;

  CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
  CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
  CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int i = 0; i < iters; ++i) run_main();
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms_main = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms_main, ev0, ev1));
  ms_main /= iters;
  printf("[timing] quant %.4f ms | preprocess %.4f ms | main %.4f ms | convert %.4f ms (cvt_on=%d, qfuse=%d, dfuse=%d)\n",
         ms_quant, ms_pre, ms_main, ms - ms_quant - ms_pre - ms_main, cvt_on, qfuse, dfuse);

  // ---- O64 A/B：融合 quant+zero 的同一 binary 端到端对比。----
  {
    auto time_all = [&](bool f) {
      qfuse = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    // 数值检查：两模式的输出差异应只来自跨 CTA `atomicAdd` 的次序（~e-7）。
    std::vector<float> a(nq + 2 * nkv), b(nq + 2 * nkv);
    qfuse = 1; run_all();
    CUDA_CHECK(cudaMemcpy(a.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a.data() + nq, d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a.data() + nq + nkv, d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    qfuse = 0; run_all();
    CUDA_CHECK(cudaMemcpy(b.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data() + nq, d_dk, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data() + nq + nkv, d_dv, nkv * 4, cudaMemcpyDeviceToHost));
    qfuse = 1;
    double mq = 0, mk = 0, mv = 0;
    for (size_t i = 0; i < nq; ++i) mq = std::max(mq, (double)fabsf(a[i] - b[i]));
    for (size_t i = 0; i < nkv; ++i) mk = std::max(mk, (double)fabsf(a[nq + i] - b[nq + i]));
    for (size_t i = 0; i < nkv; ++i) mv = std::max(mv, (double)fabsf(a[nq + nkv + i] - b[nq + nkv + i]));
    printf("[O64 A/B] end2end fused(1 launch) %.4f ms | unfused(4 quant + 3 memset) %.4f ms "
           "(%.4fx) | max_abs(fused-vs-unfused) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t_f, t_u, t_u / t_f, mq, mk, mv);
  }

  // ---- O66 A/B：把 delta 融进 quant kernel（同一 binary）端到端对比 + delta 逐位校验。----
  {
    auto time_all = [&](bool f) {
      dfuse = f ? 1 : 0;
      for (int i = 0; i < 3; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_all();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      return t / iters;
    };
    const float t_f = time_all(true), t_u = time_all(false);
    std::vector<float> df((size_t)d_rows), du((size_t)d_rows);
    dfuse = 1; run_all();
    CUDA_CHECK(cudaMemcpy(df.data(), d_delta, (size_t)d_rows * 4, cudaMemcpyDeviceToHost));
    dfuse = 0; run_all();
    CUDA_CHECK(cudaMemcpy(du.data(), d_delta, (size_t)d_rows * 4, cudaMemcpyDeviceToHost));
    dfuse = 1;
    double mdelta = 0;
    for (int i = 0; i < d_rows; ++i) mdelta = std::max(mdelta, (double)fabsf(df[i] - du[i]));
    printf("[O66 A/B] end2end delta-fused %.4f ms | delta-separate %.4f ms (%.4fx) | "
           "max_abs(delta fused-vs-separate)=%.3e\n", t_f, t_u, t_u / t_f, mdelta);
  }

  // ---- O48 A/B（D=128，仅 mma 路径）：主 kernel 4-warp（128/2）vs 8-warp（256/4 网格）。
  //      同 session 计时 + 逐元素对拍（只换 warp 网格、数学/数据流不变）。
  if (D == 128 && !bn64_opt) {
    auto run_d128w = [&](int nth) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (nth == 256) {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true, true, 256, 4>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
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
    std::vector<float> a4_dq(nq), a4_dk(nkv), a4_dv(nkv), b8_dq(nq), b8_dk(nkv), b8_dv(nkv);
    run_d128w(128);
    CUDA_CHECK(cudaMemcpy(a4_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a4_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_d128w(256);
    CUDA_CHECK(cudaMemcpy(b8_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b8_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto md = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O48 A/B] main D=128 4w(128/2) %.4f ms | 8w(256/4) %.4f ms (%.3fx) | "
           "max_abs(8w-vs-4w) dq/dk/dv=%.3e/%.3e/%.3e\n",
           m4w, m8w, m4w / m8w, md(b8_dq, a4_dq), md(b8_dk, a4_dk), md(b8_dv, a4_dv));
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

  // ---- P3-4e/P3-4f A/B（D=128 定长，`--det` / `--detk=N`）：跨 CTA `atomicAdd`（非确定性）
  //      vs 确定性 partial + 固定次序二次归约（对齐 fp16/bf16 O7b）。`--detk>1` 时 dQ 也走
  //      partial（`dq_reduce_kernel`），验证「确定性 + split-K 并行度」。同 session 计时 +
  //      跑两遍 DET 验证逐位可复现，再与 atomic 比 max|diff|。BK=64（fp8 默认 BM）。
  //      注：dK/dV 的 partial 天然无需 part 维——每个 (mblk, jg) 的 K tile 只属于一个 part，
  //      各 part 写不相交的 jg；跨 part 的贡献经固定次序 mblk 归约。----
  if (det_ab && D == 128 && !varlen) {
    const int nblk_d = (S + 63) / 64;
    // O102（第 196 轮）：DET 的 auto ksplit 标定——causal 下 k=4 触底（P3-4f 已测；
    //   full 无三角偏斜、多切只增 Q/dO 重读 + dQ 跨 part 原子 ⇒ k=1）。`--detk=K` 仍可覆盖。
    const int ks = det_ksplit >= 1 ? det_ksplit : (causal ? (nblk_d < 4 ? nblk_d : 4) : 1);
    const size_t part_elems = (size_t)B * H * nblk_d * S * D;
    const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    auto run_at = [&](int k) {
      dim3 g((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_main<128, 64, 32, true, false, true, true, true>(
          g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
          d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, k);
    };
    auto run_dt = [&](int k) {
      dim3 g((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<128, 64, 32, true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                            d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                            S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part,
                                            nblk_d, d_dq_part);
      // P3-4n：dkv+dq 融合归约（默认）vs 两次 launch（`--nofusered`）。
      const int dkv_blocks = B * Hkv * S;
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    auto time_fn2 = [&](auto fn, float* out_ms) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      float t = 0.f;
      CUDA_CHECK(cudaEventElapsedTime(&t, ev0, ev1));
      *out_ms = t / iters;
    };
    float ms_at = 0.f, ms_dt = 0.f;
    time_fn2([&] { run_at(ks); }, &ms_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    time_fn2([&] { run_dt(ks); }, &ms_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    auto mad2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[P3-4f A/B] ksplit=%d | atomic %.4f ms | DET(partial+reduce) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, ms_at, ms_dt, ms_at / ms_dt, mad2(dq1, dq2), mad2(dk1, dk2), mad2(dv1, dv2),
           mad2(dq1, aq), mad2(dk1, ak), mad2(dv1, av));
    // ---- P3-4n A/B：二次归约「两次 launch（分离）」vs「一次 launch（融合）」，只测 reduce。
    //      partial 先用一次 DET 主 kernel 预置好；两版只差网格组织、求和次序逐字相同。----
    {
      auto only_sep = [&]() {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (ks > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<128><<<dg, 128>>>(d_dq_part, d_dq_acc, S, H, ks);
        }
      };
      auto only_fus = [&]() {
        const int dkv_blocks = B * Hkv * S;
        const int dq_blocks = (ks > 1) ? B * S * H : 0;
        dkv_dq_reduce_kernel<128, 64><<<dkv_blocks + dq_blocks, 128>>>(
            d_dk_part, d_dv_part, (ks > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, ks, dkv_blocks);
      };
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      {
        dim3 g((S + 63) / 64 * ks, H, B);
        launch_bwd_main_det<128, 64, 32, true>(g, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8,
                                               d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc,
                                               S, H, Hkv, scale, (int)causal, ks, d_dk_part,
                                               d_dv_part, nblk_d, d_dq_part);
      }
      float t_rs = 0.f, t_rf = 0.f;
      time_fn2(only_sep, &t_rs);
      std::vector<float> rsdk(nkv), rsdv(nkv), rsdq(nq);
      CUDA_CHECK(cudaMemcpy(rsdk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rsdq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      time_fn2(only_fus, &t_rf);
      std::vector<float> rfdk(nkv), rfdv(nkv), rfdq(nq);
      CUDA_CHECK(cudaMemcpy(rfdk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(rfdq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4n A/B] reduce separate(2 launch) %.4f ms | fused(1 launch) %.4f ms (%.3fx) | "
             "fused-vs-sep bitwise dq/dk/dv=%.2e/%.2e/%.2e\n",
             t_rs, t_rf, t_rs / t_rf, mad2(rfdq, rsdq), mad2(rfdk, rsdk), mad2(rfdv, rsdv));
    }
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

  // ---- P3-4h/P3-4k A/B（D=512 MLA 定长，`--det` / `--detk=N`）：把确定性 dK/dV 从 D=128
  //      扩到 MLA（HD=512）。P3-4e/f 的 partial 布局与 `dkv_reduce_kernel` 本就对任意 HD 成立，
  //      body 的 DET 分支也是 HD 无关的；故只需给 `launch_bwd_main_det` 加 NTH/NWAR 模板参数、
  //      在 host 补一条 MLA 的 A/B。atomic 参照用同 256/4 几何的非 kvpipe 主 kernel（仅 DET
  //      一个变量不同）。跑两遍 DET 验证逐位可复现。----
  //      P3-4k：MLA 的 `kRegDq` 恒 false（HD/NTW=4）⇒ 旧版锁 ksplit=1（单写者 `red_add2` 保
  //      dQ 确定）。本轮把非 `kRegDq` 的 dQ epilogue 在 `DET && ksplit>1` 时改写到 `dq_part`
  //      的 part 分片 + `dq_reduce_kernel` 固定次序求和，从而**在 ksplit>1 下也确定**、恢复
  //      split-K 并行度。`--detk=N` 触发；默认 `--det`（ks=1）与 P3-4h 逐位一致。----
  if (det_ab && D == 512 && !varlen) {
    const int nblk_d = (S + 63) / 64;
    const int ks = det_ksplit < 1 ? 1 : det_ksplit;
    const size_t part_elems = (size_t)B * H * nblk_d * S * D;
    const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
    float* d_dk_part = nullptr;
    float* d_dv_part = nullptr;
    float* d_dq_part = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
    if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
    auto run_at = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k);
    };
    auto do_reduce = [&](int k) {
      const int dkv_blocks = B * Hkv * S;
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<512, 64><<<dkv_blocks + dq_blocks, 512>>>(
            d_dk_part, d_dv_part, (k > 1) ? d_dq_part : nullptr, d_dk_acc, d_dv_acc, d_dq_acc, S,
            H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<512, 64><<<rg, 512>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H,
                                                Hkv, nblk_d, (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    auto run_dt = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_d, d_dq_part);
      do_reduce(k);
    };
    // P3-4m：DET + O51 K/V `cp.async` 回填流水（kvpipe），数值应与非 kvpipe 逐位相同。
    auto run_dt_kv = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, true>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, d_dk_part, d_dv_part, nblk_d, d_dq_part);
      do_reduce(k);
    };
    auto tm2h = [&](auto fn, float* o) {
      for (int i = 0; i < 3; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(o, ev0, ev1));
      *o /= iters;
    };
    auto mad2h = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i) m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    float t_at = 0.f, t_dt = 0.f, t_dt1 = 0.f, t_dt_kv = 0.f;
    tm2h([&] { run_at(ks); }, &t_at);
    std::vector<float> ak(nkv), av(nkv), aq(nq);
    CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    tm2h([&] { run_dt(1); }, &t_dt1);
    tm2h([&] { run_dt(ks); }, &t_dt);
    std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
    CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    // P3-4m：DET + kvpipe（数值应与非 kvpipe 逐位相同）。
    tm2h([&] { run_dt_kv(ks); }, &t_dt_kv);
    std::vector<float> dkk1(nkv), dvk1(nkv), dqk1(nq), dkk2(nkv), dvk2(nkv), dqk2(nq);
    CUDA_CHECK(cudaMemcpy(dkk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    run_dt_kv(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dkk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dvk2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dqk2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    printf("[P3-4k A/B] MLA ksplit=%d atomic(256/4) %.4f ms | DET %.4f ms (vs atomic %.3fx) | "
           "DET k=1 %.4f ms -> k=%d %.4f ms (%.3fx split speedup) | "
           "runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
           ks, t_at, t_dt, t_at / t_dt, t_dt1, ks, t_dt, t_dt1 / t_dt, mad2h(dq1, dq2),
           mad2h(dk1, dk2), mad2h(dv1, dv2), mad2h(dq1, aq), mad2h(dk1, ak), mad2h(dv1, av));
    printf("[P3-4m A/B] MLA ksplit=%d | DET nonkv %.4f ms | DET kvpipe %.4f ms (%.3fx) | "
           "kvpipe runs[1-2] bitwise dq/dk/dv=%.2e/%.2e/%.2e | kvpipe-vs-nonkv dq/dk/dv="
           "%.2e/%.2e/%.2e\n",
           ks, t_dt, t_dt_kv, t_dt / t_dt_kv, mad2h(dqk1, dqk2), mad2h(dkk1, dkk2),
           mad2h(dvk1, dvk2), mad2h(dqk1, dq1), mad2h(dkk1, dk1), mad2h(dvk1, dv1));
    // ---- F4-b（第 134 轮）：MLA（HD=512）定长的「fp16 partial + O62 扇区化」。
    //      `dkv_reduce_kernel<512,64,true>` 已支持 P16；此处只把主 kernel 换 `DET_HALF`。----
    __half* d_dk_part_h = nullptr;
    __half* d_dv_part_h = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
    auto run_dth = [&](int k) {
      dim3 g1((S + 63) / 64 * k, H, B);
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (k > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
      launch_bwd_main_det<512, 64, 32, false, true, true, true, 256, 4, false, false, true>(
          g1, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc,
          d_dv_acc, S, H, Hkv, scale, (int)causal, k, reinterpret_cast<float*>(d_dk_part_h),
          reinterpret_cast<float*>(d_dv_part_h), nblk_d, d_dq_part);
      const int dkv_blocks = B * Hkv * S;
      const int dq_blocks = (k > 1) ? B * S * H : 0;
      if (fuse_reduce) {
        dkv_dq_reduce_kernel<512, 64, true><<<dkv_blocks + dq_blocks, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), (k > 1) ? d_dq_part : nullptr, d_dk_acc,
            d_dv_acc, d_dq_acc, S, H, Hkv, nblk_d, (int)causal, k, dkv_blocks);
      } else {
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<512, 64, true><<<rg, 512>>>(
            reinterpret_cast<const float*>(d_dk_part_h),
            reinterpret_cast<const float*>(d_dv_part_h), d_dk_acc, d_dv_acc, S, H, Hkv, nblk_d,
            (int)causal);
        if (k > 1) {
          dim3 dg(B * S, H);
          dq_reduce_kernel<512><<<dg, 512>>>(d_dq_part, d_dq_acc, S, H, k);
        }
      }
    };
    float t_dth = 0.f;
    tm2h([&] { run_dth(ks); }, &t_dth);
    std::vector<float> hdk1(nkv), hdv1(nkv), hdk2(nkv), hdv2(nkv);
    CUDA_CHECK(cudaMemcpy(hdk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_dth(ks);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hdk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hdv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[F4 A/B] MLA ksplit=%d | DET-fp32 %.4f ms | DET-fp16(扇区化) %.4f ms (%.3fx) | "
           "runs[1-2] bitwise dk/dv=%.2e/%.2e | fp16-vs-fp32 dk/dv=%.2e/%.2e\n",
           ks, t_dt, t_dth, t_dt / t_dth, mad2h(hdk1, hdk2), mad2h(hdv1, hdv2),
           mad2h(hdk1, dk1), mad2h(hdv1, dv1));
    cudaFree(d_dk_part_h);
    cudaFree(d_dv_part_h);
    cudaFree(d_dk_part);
    cudaFree(d_dv_part);
    if (d_dq_part) cudaFree(d_dq_part);
    run_main();   // 恢复最终输出为 CLI 选中的路径
  }

#ifdef FA_WGMMA
  // ---- O9c-2 A/B（D=128）：主 kernel GEMM1/2 的 mma.m16n8k32 vs wgmma.m64n32k32。----
  //      同一 session 计时 + 逐元素对拍（证明只换计算后端、数学口径未变）。
  if (D == 128) {
    auto run_main_wg = [&](bool wg) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (use_regdq) {
        if (wg)
          launch_bwd_main<128, 64, 32, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                   d_vs, d_do8, d_dos, d_delta, d_lse,
                                                   d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                   Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, true, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
      } else {
        if (wg)
          launch_bwd_main<128, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false>(mg, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                     d_vs, d_do8, d_dos, d_delta, d_lse,
                                                     d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                     Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_main_wg = [&](bool wg, float* out) {
      run_main_wg(wg);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_wg(wg);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mmma = 0.f, mwgm = 0.f;
    bench_main_wg(false, &mmma);
    bench_main_wg(true, &mwgm);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_main_wg(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_wg(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O9c-2 A/B] main mma(regdq=%d) %.4f ms | wgmma(m64n32) %.4f ms (%.3fx) | "
           "max_abs(wg-vs-mma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           (int)use_regdq, mmma, mwgm, mmma / mwgm, maxd(b_dq, a_dq), maxd(b_dk, a_dk),
           maxd(b_dv, a_dv));
    // 恢复最终输出为 CLI 选中的路径（上面 A/B 最后一次跑的是 wgmma）。
    run_main_wg(wgmma != 0);
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O37 A/B（D=128）：主 kernel Q/dO cp.async vs 4D-TMA（WGMMA 路径，其余逐字相同）。----
    auto run_main_qd = [&](bool tma) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (tma) {
        if (use_regdq)
          launch_bwd_main_qdtma<128, 64, 32, true>(mg, qmap_main, dmap_main, d_q8, d_qs, d_k8,
                                                   d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                                   d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                   Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main_qdtma<128, 64, 32, false>(mg, qmap_main, dmap_main, d_q8, d_qs, d_k8,
                                                    d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                                                    d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H,
                                                    Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, true>(mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                    d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                    (int)causal, ksplit);
      }
    };
    auto bench_main_qd = [&](bool tma, float* out) {
      run_main_qd(tma);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_qd(tma);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mcp = 0.f, mtma = 0.f;
    bench_main_qd(false, &mcp);
    bench_main_qd(true, &mtma);
    run_main_qd(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_qd(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O37 A/B] main Q/dO cp.async %.4f ms | tma %.4f ms (%.3fx) | "
           "max_abs(tma-vs-cp) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mcp, mtma, mcp / mtma, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    run_main_wg(wgmma != 0);
    // ---- O41 A/B（D=128）：主 kernel Q/dO-TMA vs Q/dO/K/V 全 TMA（WGMMA 路径，其余逐字相同）。----
    auto run_main_kv = [&](bool kvt) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (kvt) {
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(mg, qmap_main, dmap_main, kmap_main,
                                                   vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                   d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ksplit);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(mg, qmap_main, dmap_main, kmap_main,
                                                    vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse,
                                                    d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                    scale, (int)causal, ksplit);
      } else {
        run_main_qd(true);
      }
    };
    auto bench_main_kv = [&](bool kvt, float* out) {
      run_main_kv(kvt);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_main_kv(kvt);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float mqd = 0.f, mkv = 0.f;
    bench_main_kv(false, &mqd);
    bench_main_kv(true, &mkv);
    run_main_kv(false);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_main_kv(true);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    printf("[O41 A/B] main Q/dO-TMA %.4f ms | Q/dO/K/V-TMA %.4f ms (%.3fx) | "
           "max_abs(kvtma-vs-qdtma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mqd, mkv, mqd / mkv, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    // ---- P3-4g/O102 A/B（D=128/Hopper TMA，`--det`）：把确定性 dK/dV 从默认 mma 路径扩到
    //      Q/dO/K/V-TMA 快路。P3-4g 原版锁 ksplit=1；O102（第 196 轮）放开 `--detk=N>1`——
    //      dQ 走 per-part `dq_part` + `dq_reduce_kernel`，恢复 split-K 并行度（同 mma 路径
    //      P3-4f）。跑两遍 DET 验证 dq/dk/dv 逐位可复现，再与 atomic 比 max|diff|。----
    if (det_ab && !varlen) {
      const int nblk_d = (S + 63) / 64;
      // O102：DET auto ksplit（causal k=4 触底 / full k=1）——Hopper 快路同 mma 路径标定。
      const int ks = det_ksplit >= 1 ? det_ksplit : (causal ? (nblk_d < 4 ? nblk_d : 4) : 1);
      const size_t part_elems = (size_t)B * H * nblk_d * S * D;
      const size_t dqpart_elems = (size_t)B * S * H * (size_t)ks * D;
      float* d_dk_part = nullptr;
      float* d_dv_part = nullptr;
      float* d_dk_part_h = nullptr;
      float* d_dv_part_h = nullptr;
      float* d_dq_part = nullptr;
      CUDA_CHECK(cudaMalloc(&d_dk_part, part_elems * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&d_dv_part, part_elems * sizeof(float)));
      // F4：fp16 partial（字节减半 + O62 扇区化）。元素数不变，仅存储类型改为 half。
      CUDA_CHECK(cudaMalloc(&d_dk_part_h, part_elems * sizeof(__half)));
      CUDA_CHECK(cudaMalloc(&d_dv_part_h, part_elems * sizeof(__half)));
      if (ks > 1) CUDA_CHECK(cudaMalloc(&d_dq_part, dqpart_elems * sizeof(float)));
      dim3 g1((S + 63) / 64 * ks, H, B);
      auto dq_reduce = [&]() {
        if (ks > 1)
          dq_reduce_kernel<128><<<dim3(B * S, H), 128>>>(d_dq_part, d_dq_acc, S, H, ks);
      };
      auto run_at = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (use_regdq)
          launch_bwd_main_kvtma<128, 64, 32, true>(g1, qmap_main, dmap_main, kmap_main,
                                                   vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
                                                   d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                   d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                   (int)causal, ks);
        else
          launch_bwd_main_kvtma<128, 64, 32, false>(g1, qmap_main, dmap_main, kmap_main,
                                                    vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                    d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
                                                    d_dk_acc, d_dv_acc, S, H, Hkv, scale,
                                                    (int)causal, ks);
      };
      auto run_dt = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));  // 空 part 需为 0
        if (use_regdq)
          launch_bwd_main_kvtma_det<128, 64, 32, true>(g1, qmap_main, dmap_main, kmap_main,
                                                       vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                       d_vs, d_do8, d_dos, d_delta, d_lse,
                                                       d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                       scale, (int)causal, d_dk_part, d_dv_part,
                                                       nblk_d, ks, d_dq_part);
        else
          launch_bwd_main_kvtma_det<128, 64, 32, false>(g1, qmap_main, dmap_main, kmap_main,
                                                        vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8,
                                                        d_vs, d_do8, d_dos, d_delta, d_lse,
                                                        d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                        scale, (int)causal, d_dk_part, d_dv_part,
                                                        nblk_d, ks, d_dq_part);
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64><<<rg, 128>>>(d_dk_part, d_dv_part, d_dk_acc, d_dv_acc, S, H, Hkv,
                                                nblk_d, (int)causal);
        dq_reduce();
      };
      // F4：fp16 partial + O62 扇区化的 DET（写侧与归约侧都走 P16 布局）。
      auto run_dth = [&]() {
        CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
        CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
        CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
        if (ks > 1) CUDA_CHECK(cudaMemset(d_dq_part, 0, dqpart_elems * 4));
        if (use_regdq)
          launch_bwd_main_kvtma_det<128, 64, 32, true, true, true, true, true>(
              g1, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
              d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
              (int)causal, d_dk_part_h, d_dv_part_h, nblk_d, ks, d_dq_part);
        else
          launch_bwd_main_kvtma_det<128, 64, 32, false, true, true, true, true>(
              g1, qmap_main, dmap_main, kmap_main, vmap_main, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs,
              d_do8, d_dos, d_delta, d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale,
              (int)causal, d_dk_part_h, d_dv_part_h, nblk_d, ks, d_dq_part);
        dim3 rg(B * Hkv, S);
        dkv_reduce_kernel<128, 64, true><<<rg, 128>>>(d_dk_part_h, d_dv_part_h, d_dk_acc, d_dv_acc,
                                                      S, H, Hkv, nblk_d, (int)causal);
        dq_reduce();
      };
      auto tm2 = [&](auto fn, float* o) {
        for (int i = 0; i < 3; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev0));
        for (int i = 0; i < iters; ++i) fn();
        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(o, ev0, ev1));
        *o /= iters;
      };
      float t_at = 0.f, t_dt = 0.f, t_dth = 0.f;
      tm2(run_at, &t_at);
      std::vector<float> ak(nkv), av(nkv), aq(nq);
      CUDA_CHECK(cudaMemcpy(ak.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(av.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(aq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      tm2(run_dt, &t_dt);
      std::vector<float> dk1(nkv), dv1(nkv), dq1(nq), dk2(nkv), dv2(nkv), dq2(nq);
      CUDA_CHECK(cudaMemcpy(dk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      run_dt();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(dk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      tm2(run_dth, &t_dth);
      std::vector<float> hk1(nkv), hv1(nkv), hq1(nq), hk2(nkv), hv2(nkv), hq2(nq);
      CUDA_CHECK(cudaMemcpy(hk1.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hv1.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hq1.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      run_dth();
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(hk2.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hv2.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(hq2.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      printf("[P3-4g A/B] kvtma ksplit=%d atomic %.4f ms | DET %.4f ms (%.3fx) | "
             "runs[1-2] dq/dk/dv=%.2e/%.2e/%.2e | DET-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_at, t_dt, t_at / t_dt, maxd(dq1, dq2), maxd(dk1, dk2), maxd(dv1, dv2),
             maxd(dq1, aq), maxd(dk1, ak), maxd(dv1, av));
      printf("[F4 A/B] kvtma ksplit=%d atomic %.4f ms | DET-fp32 %.4f ms (%.3fx) | "
             "DET-fp16(扇区化) %.4f ms (%.3fx vs atomic) | runs[1-2] dq/dk/dv=%.2e/%.2e/%.2e | "
             "fp16-vs-fp32 dq/dk/dv=%.2e/%.2e/%.2e | fp16-vs-atomic dq/dk/dv=%.2e/%.2e/%.2e\n",
             ks, t_at, t_dt, t_at / t_dt, t_dth, t_at / t_dth, maxd(hq1, hq2), maxd(hk1, hk2),
             maxd(hv1, hv2), maxd(hq1, dq1), maxd(hk1, dk1), maxd(hv1, dv1), maxd(hq1, aq),
             maxd(hk1, ak), maxd(hv1, av));
      cudaFree(d_dk_part);
      cudaFree(d_dv_part);
      cudaFree(d_dk_part_h);
      cudaFree(d_dv_part_h);
      if (d_dq_part) cudaFree(d_dq_part);
    }
    run_main_wg(wgmma != 0);
#endif
  }
#endif

  // ---- O11 A/B（仅 causal, D=128）：LSE O1 原版 vs 镜像配对(单缓冲) vs 镜像配对+cp.async ----
  if (causal && D == 128) {
    auto bench_lse = [&](int which, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (which == 0)
          launch_lse<128>(lg, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, 1);
        else if (which == 1)
          launch_lse_bal<128, 0>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
        else
          launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float a = 0.f, b = 0.f, c = 0.f;
    bench_lse(0, &a);
    bench_lse(1, &b);
    bench_lse(2, &c);
    printf("[O11 A/B] lse O1 %.4f ms | bal(单缓冲) %.4f ms (%.3fx) | bal+cpasync %.4f ms "
           "(%.3fx)\n",
           a, b, a / b, c, a / c);
#ifdef FA_WGMMA
    // O9c A/B：wgmma（SW128 + m64n64k32）vs O11 mma 版（同 PIPE=1）。同时核对 LSE 数值。
    auto bench_lse_wgm = [&](int which, float* ms_out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (which == 3)
          launch_lse_bal_wgmma<128, 0>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
        else
          launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(ms_out, ev0, ev1));
      *ms_out /= iters;
    };
    float w0 = 0.f, w1 = 0.f;
    std::vector<float> lse_wgm((size_t)B * S * H), lse_ref((size_t)B * S * H);
    bench_lse_wgm(3, &w0);
    bench_lse_wgm(4, &w1);
    // 与 O11 mma 版 LSE 对拍（应逐位相同：同数学、同 rowwise scale 顺序）。
    launch_lse_bal<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
    CUDA_CHECK(cudaMemcpy(lse_ref.data(), d_lse, lse_ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale);
    CUDA_CHECK(cudaMemcpy(lse_wgm.data(), d_lse, lse_wgm.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    double le = 0.0;
    for (size_t i = 0; i < lse_ref.size(); ++i)
      le = std::max(le, (double)std::fabs((double)lse_wgm[i] - (double)lse_ref[i]));
    printf("[O9c A/B] lse mma(bal+cpasync) %.4f ms | wgmma(单缓冲) %.4f ms | wgmma(双缓冲) "
           "%.4f ms (%.3fx) | LSE max_abs(vs mma)=%.3e\n",
           c, w0, w1, c / w1, le);
#endif
#if defined(FA_WGMMA) && defined(FA_TMA)
    // ---- O32 A/B：LSE 的 wgmma+cp.async 版 vs 4D-TMA 版（同 session + 数值对拍）----
    {
      float* d_lse2 = nullptr;
      CUDA_CHECK(cudaMalloc(&d_lse2, (size_t)B * S * H * sizeof(float)));
      auto launch_w = [&]() {
        launch_lse_bal_wgmma<128, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse2, S, H, Hkv, scale);
      };
      auto launch_t = [&]() {
        launch_lse_bal_tma<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse, S, H, Hkv,
                                   scale);
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
      printf("[O32 A/B] lse tma %.4f ms | wgmma %.4f ms (wgmma/tma %.3fx) | "
             "max_abs(tma-vs-wgmma)=%.3e\n", ms_t, ms_w, ms_w / ms_t, e);
      cudaFree(d_lse2);
    }
    // ---- O38 A/B：LSE 的 K 维 split + 二次归约（TMA 版，D=128/causal）----
    //   split=1 逐位退回 O32；split=2/4/8 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
    //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。
    {
      std::vector<float> ref_l((size_t)B * S * H, 0.f);
      float base_ms = 0.f, best = 1e9f;
      int bestk = 1;
      for (int sp = 1; sp <= 8; sp *= 2) {
        auto launch_sp = [&]() {
          launch_lse_bal_tma_split<128, 1>(lg_bal, qmap_lse, kmap_lse, d_qs, d_ks, d_lse,
                                           d_lse_part, S, H, Hkv, scale, sp);
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
      printf("[O38] best lse split=%d %.4f ms (%.3fx)\\n", bestk, best, base_ms / best);
    }
#endif
  }

  // ---- O39 A/B（D=512/MLA/causal）：`lse_mma_kernel_bal<512>` 的 K 维 split + 二次归约 ----
  //   split=1 逐位退回 O11 原路径；split>1 各扫 1/split 的 K tile 切片后 merge。逐元素对拍
  //   以 split=1 为基准（理论等价、只差 fp32 求和次序）。仅 causal（非 causal 走 O1 原版）。
  if (causal && D == 512) {
    std::vector<float> ref_l((size_t)B * S * H, 0.f);
    float base_ms = 0.f, best = 1e9f;
    int bestk = 1;
    for (int sp = 1; sp <= 16; sp *= 2) {
      auto launch_sp = [&]() {
        launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale,
                               nullptr, d_lse_part, sp);
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
      printf("[O39 A/B] lse(D=512) split=%d %.4f ms (%.3fx vs split1) | max_abs vs split1=%.3e\\n",
             sp, t, base_ms / t, e);
      if (t < best) { best = t; bestk = sp; }
    }
    printf("[O39] best lse split=%d %.4f ms (%.3fx)\\n", bestk, best, base_ms / best);
  }

  // ---- O59 A/B（D=512/MLA/causal/定长）：旧默认 LSE(`<512,1>` PIPE1/LBN64) vs O58 的 cfg6
  //   (`<512,1,false,128,16>` PIPE1/LBN16，4 CTA/SM)。只改 LSE 的 smem 几何/并行度，数学口径
  //   不变（split 只改 fp32 求和次序）。同 session、LSE-only 计时 + 逐元素对拍。----
  if (causal && D == 512) {
    auto auto_split = [&](int target) {
      long lg_grid = (long)lg_bal.x * H * B;
      int sp = 1;
      while (sp < 16 && lg_grid * (sp * 2) <= target) sp *= 2;
      int nblk_cap = (S + 63) / 64;
      while (sp > nblk_cap) sp >>= 1;
      return sp;
    };
    const int sp_leg = auto_split(256), sp_cfg = auto_split(1024);
    auto bench_lse = [&](bool cfg6, int sp, float* out) {
      auto go = [&]() {
        if (cfg6)
          launch_lse_bal<512, 1, false, 128, 16>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv,
                                                 scale, nullptr, d_lse_part, sp);
        else
          launch_lse_bal<512, 1>(lg_bal, d_q8, d_qs, d_k8, d_ks, d_lse, S, H, Hkv, scale, nullptr,
                                 d_lse_part, sp);
      };
      for (int i = 0; i < 3; ++i) go();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t_leg = 0.f, t_cfg = 0.f;
    std::vector<float> l_leg((size_t)B * S * H), l_cfg((size_t)B * S * H);
    bench_lse(false, sp_leg, &t_leg);
    CUDA_CHECK(cudaMemcpy(l_leg.data(), d_lse, l_leg.size() * 4, cudaMemcpyDeviceToHost));
    bench_lse(true, sp_cfg, &t_cfg);
    CUDA_CHECK(cudaMemcpy(l_cfg.data(), d_lse, l_cfg.size() * 4, cudaMemcpyDeviceToHost));
    double e = 0.0;
    for (size_t i = 0; i < l_leg.size(); ++i)
      e = std::max(e, (double)std::fabs((double)l_leg[i] - (double)l_cfg[i]));
    printf("[O59 A/B] LSE MLA legacy(sp=%d) %.4f ms | cfg6(sp=%d) %.4f ms (%.3fx) | "
           "max_abs(cfg6-vs-legacy)=%.3e\n",
           sp_leg, t_leg, sp_cfg, t_cfg, t_leg / t_cfg, e);
  }

  // ---- O12 A/B（D=128/512）：主 kernel 的 LSE/D 预装寄存器 ON vs OFF（同 session 计时）。----
  //  OFF 版就是旧的「GEMM1/2 epilogue 里逐元素 global 读 lse/delta」。数值应逐位相同。
  if (D == 128 || D == 512) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_sel = [&](bool prel) {
      if (D == 128) {
        launch128(use_regdq, wg_ab, prel, true);
      } else {
        if (prel)
          launch_bwd_main<512, 64, 32, false, false, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<512, 64, 32, false, false, false>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_prel = [&](bool prel, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_sel(prel);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_sel(prel);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m_off = 0.f, m_on = 0.f;
    bench_prel(false, &m_off);
    bench_prel(true, &m_on);
    // 逐元素对拍（应逐位相同）
    std::vector<float> p_off_dq(nq), p_on_dq(nq);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    launch_sel(false);
    CUDA_CHECK(cudaMemcpy(p_off_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    launch_sel(true);
    CUDA_CHECK(cudaMemcpy(p_on_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    double pd = 0.0;
    for (size_t i = 0; i < p_on_dq.size(); ++i)
      pd = std::max(pd, (double)std::fabs((double)p_off_dq[i] - (double)p_on_dq[i]));
    printf("[O12 A/B] main LSE/D preload off %.4f ms | on %.4f ms (%.3fx) | "
           "max_abs(on-vs-off) dq=%.3e\n",
           m_off, m_on, m_off / m_on, pd);
    run_main();  // 恢复 CLI 选中路径（写回 d_dq_acc，不影响 d_dq）
  }

  // ---- O47 A/B（D=512/MLA）：主 kernel 4-warp(128 线程) vs 8-warp(256 线程) ----
  //   只改 warp 网格/累加器划分，数学口径不变；跨 CTA atomic 次序变 ⇒ 不逐位（预期 ~1e-5）。
  if (D == 512) {
    auto go_mla = [&](bool w8) {
      if (w8)
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    };
    auto bench_mla = [&](bool w8, float* out) {
      zero_acc();
      go_mla(w8);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go_mla(w8);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float m4 = 0.f, m8 = 0.f;
    bench_mla(false, &m4);
    bench_mla(true, &m8);
    std::vector<float> dq4(nq), dq8(nq), dk4(nkv), dk8(nkv), dv4(nkv), dv8(nkv);
    auto grab = [&](bool w8, std::vector<float>& dq, std::vector<float>& dk,
                    std::vector<float>& dv) {
      zero_acc();
      go_mla(w8);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    };
    grab(false, dq4, dk4, dv4);
    grab(true, dq8, dk8, dv8);
    double dqd = 0, dkd = 0, dvd = 0;
    for (size_t i = 0; i < dq4.size(); ++i) {
      dqd = std::max(dqd, (double)std::fabs((double)dq4[i] - (double)dq8[i]));
      dkd = std::max(dkd, (double)std::fabs((double)dk4[i] - (double)dk8[i]));
      dvd = std::max(dvd, (double)std::fabs((double)dv4[i] - (double)dv8[i]));
    }
    printf("[O47 A/B] main MLA 4-warp %.4f ms | 8-warp %.4f ms (%.3fx) | "
           "max_abs(8w-vs-4w) dq=%.3e dk=%.3e dv=%.3e\n",
           m4, m8, m4 / m8, dqd, dkd, dvd);
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O51 A/B（D=512/MLA）：8-warp 主 kernel 的 K/V 同步载入 vs cp.async 回填流水 ----
  //   只改搬运（行主序 cp.async + 从 Ks 重建 Kp），数学/K/V 字节完全相同 ⇒ 应逐位相同。
  if (D == 512) {
    auto go51 = [&](bool kvp) {
      if (kvp)
        launch_bwd_main_kvpipe<512, 64, 32, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<512, 64, 32, false, false, true, true, true, 256, 4>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto zero_acc = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    };
    auto bench51 = [&](bool kvp, float* out) {
      zero_acc();
      go51(kvp);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) go51(kvp);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float s_sync = 0.f, s_kvp = 0.f;
    bench51(false, &s_sync);
    bench51(true, &s_kvp);
    std::vector<float> a_dq(nq), b_dq(nq), a_dk(nkv), b_dk(nkv), a_dv(nkv), b_dv(nkv);
    auto grab51 = [&](bool kvp, std::vector<float>& dq, std::vector<float>& dk,
                      std::vector<float>& dv) {
      zero_acc();
      go51(kvp);
      CUDA_CHECK(cudaMemcpy(dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    };
    grab51(false, a_dq, a_dk, a_dv);
    grab51(true, b_dq, b_dk, b_dv);
    auto md51 = [](const std::vector<float>& a, const std::vector<float>& b) {
      double d = 0;
      for (size_t i = 0; i < a.size(); ++i)
        d = std::max(d, (double)std::fabs((double)a[i] - (double)b[i]));
      return d;
    };
    printf("[O51 A/B] main MLA 8w sync %.4f ms | kvpipe %.4f ms (%.3fx) | "
           "max_abs(kvp-vs-sync) dq=%.3e dk=%.3e dv=%.3e\n",
           s_sync, s_kvp, s_sync / s_kvp, md51(a_dq, b_dq), md51(a_dk, b_dk), md51(a_dv, b_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O7e-2 A/B（D=128）：fold 的 Ap/dS3/dS2 用 16B 向量化写 vs O7e 的 4B 写 ----
  //   同 session 计时 + 逐元素对拍（同一份 fp8 操作数 ⇒ 应逐位相同）。
  if (D == 128) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_f16b = [&](bool f16) { launch128(use_regdq, wg_ab, prel_sel, f16); };
    auto bench_f16b = [&](bool f16, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_f16b(f16);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_f16b(f16);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float b4 = 0.f, b16 = 0.f;
    bench_f16b(false, &b4);
    bench_f16b(true, &b16);
    // 逐元素对拍（4B vs 16B，应逐位相同）。
    std::vector<float> x_dq(nq), x_dk(nkv), x_dv(nkv), y_dq(nq), y_dk(nkv), y_dv(nkv);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_f16b(false);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_f16b(true);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O7e-2 A/B] main fold 4B %.4f ms | 16B %.4f ms (%.3fx) | "
           "max_abs(16B-vs-4B) dq/dk/dv=%.3e/%.3e/%.3e\n",
           b4, b16, b4 / b16, maxd2(y_dq, x_dq), maxd2(y_dk, x_dk), maxd2(y_dv, x_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O27 A/B（D=128）：fold 量化 逐元素精确除法 vs 每行 rcp+乘法 ----
  //   `scA/sc3/sc2` 是每输出行一个的常量，原实现让每个元素都发一条精确 fp32 除法
  //   （prec-div，~10+ 指令）；改成每行一次 `__frcp_rn` + 乘法。数学等价，fp8 只有 3 位
  //   尾数 ⇒ cvt 结果几乎不变。同 session 计时 + 逐元素对拍。
  if (D == 128) {
#ifdef FA_WGMMA
    const bool wg_ab = (wgmma != 0);
#else
    const bool wg_ab = false;
#endif
    auto launch_rcp = [&](bool rcp) { launch128(use_regdq, wg_ab, prel_sel, f16b_sel, rcp); };
    auto bench_rcp = [&](bool rcp, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_rcp(rcp);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_rcp(rcp);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float bdiv = 0.f, brcp = 0.f;
    bench_rcp(false, &bdiv);
    bench_rcp(true, &brcp);
    std::vector<float> x_dq(nq), x_dk(nkv), x_dv(nkv), y_dq(nq), y_dk(nkv), y_dv(nkv);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_rcp(false);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(x_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_rcp(true);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd2 = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O27 A/B] main fold div %.4f ms | rcp-mul %.4f ms (%.3fx) | "
           "max_abs(rcp-vs-div) dq/dk/dv=%.3e/%.3e/%.3e\n",
           bdiv, brcp, bdiv / brcp, maxd2(y_dq, x_dq), maxd2(y_dk, x_dk), maxd2(y_dv, x_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O22 A/B（D=128）：主 kernel 的寄存器 dQ 累加（REGDQ）ON vs OFF（mma 路径）----
  //   O21 的 ncu 显示：REGDQ 的 `dqacc[2][8][4]`（64 个 fp32）把 168-reg 预算挤爆，PTXAS 报
  //   68B spill stores / 380B spill loads，local 占 L1TEX sector 9.57%（Est. 28–50%）。关掉
  //   REGDQ 则 0 spill、且恢复 O3 寄存器预取，代价是 dQ 每个 nt tile 发一次跨 CTA atomicAdd
  //   （归约指令 O(1)→O(ntiles)）。这里同 session 计时 + 逐元素对拍（数学口径相同，仅 dQ 归约
  //   指令数/次序不同；dk/dv 不受影响）。
  if (D == 128) {
    auto launch_reg = [&](bool reg) { launch128(reg, false, prel_sel, f16b_sel); };
    auto bench_reg = [&](bool reg, float* out) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_reg(reg);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) launch_reg(reg);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float ron = 0.f, roff = 0.f;
    bench_reg(true, &ron);
    bench_reg(false, &roff);
    std::vector<float> x_dq(nq), y_dq(nq);
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_reg(true);
    CUDA_CHECK(cudaMemcpy(x_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
    CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
    CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
    launch_reg(false);
    CUDA_CHECK(cudaMemcpy(y_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    double dqdiff = 0.0;
    for (size_t i = 0; i < nq; ++i)
      dqdiff = std::max(dqdiff, std::fabs((double)x_dq[i] - (double)y_dq[i]));
    printf("[O22 A/B] main REGDQ on %.4f ms | off %.4f ms (%.3fx) | max_abs(on-vs-off) dq=%.3e\n",
           ron, roff, ron / roff, dqdiff);
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O14 A/B：输入量化 旧 per-row 标量 vs 新 warp-per-row 向量化（同 session 计时 + 逐位对拍）----
  {
    auto bench_q = [&](bool fast, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) { if (fast) quant_new(); else quant_old(); }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float qo = 0.f, qn = 0.f;
    bench_q(false, &qo);
    bench_q(true, &qn);
    // 逐字节对拍（q8/qs/k8/ks/v8/vs/do8/dos 应完全相同）。
    std::vector<unsigned char> o_q8(nq), o_k8(nkv), o_v8(nkv), o_do8(nq);
    std::vector<float> o_qs(rows_q), o_ks(rows_kv), o_vs(rows_kv), o_dos(rows_q);
    quant_old();
    CUDA_CHECK(cudaMemcpy(o_q8.data(), d_q8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_k8.data(), d_k8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_v8.data(), d_v8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_do8.data(), d_do8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_qs.data(), d_qs, rows_q * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_ks.data(), d_ks, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_vs.data(), d_vs, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(o_dos.data(), d_dos, rows_q * 4, cudaMemcpyDeviceToHost));
    quant_new();
    std::vector<unsigned char> n_q8(nq), n_k8(nkv), n_v8(nkv), n_do8(nq);
    std::vector<float> n_qs(rows_q), n_ks(rows_kv), n_vs(rows_kv), n_dos(rows_q);
    CUDA_CHECK(cudaMemcpy(n_q8.data(), d_q8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_k8.data(), d_k8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_v8.data(), d_v8, nkv, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_do8.data(), d_do8, nq, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_qs.data(), d_qs, rows_q * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_ks.data(), d_ks, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_vs.data(), d_vs, rows_kv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(n_dos.data(), d_dos, rows_q * 4, cudaMemcpyDeviceToHost));
    long long mismatch = 0;
    auto cmp_u8 = [&](const std::vector<unsigned char>& a, const std::vector<unsigned char>& c) {
      for (size_t i = 0; i < a.size(); ++i) if (a[i] != c[i]) ++mismatch;
    };
    auto cmp_f = [&](const std::vector<float>& a, const std::vector<float>& c) {
      for (size_t i = 0; i < a.size(); ++i) if (a[i] != c[i]) ++mismatch;
    };
    cmp_u8(o_q8, n_q8); cmp_u8(o_k8, n_k8); cmp_u8(o_v8, n_v8); cmp_u8(o_do8, n_do8);
    cmp_f(o_qs, n_qs); cmp_f(o_ks, n_ks); cmp_f(o_vs, n_vs); cmp_f(o_dos, n_dos);
    printf("[O14 A/B] quant old(per-row) %.4f ms | new(warp-per-row) %.4f ms (%.3fx) | "
           "bitwise mismatch=%lld\n", qo, qn, qo / qn, mismatch);
    run_all();  // 恢复 CLI 选中路径的完整输出
  }

  // ---- O26 A/B：delta 旧 per-row(smem 归约) vs 新 warp-per-row 向量化（同 session 计时 + 对拍）----
  {
    auto bench_delta = [&](bool warp, float* out) {
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) {
        if (warp) {
          if (D == 128)
            delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
          else if (D == 256)
            delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
          else
            delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
        } else {
          if (D == 128) delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
          else if (D == 256) delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
          else delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
        }
      }
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float do_ms = 0.f, dw_ms = 0.f;
    bench_delta(false, &do_ms);
    bench_delta(true, &dw_ms);
    if (D == 128) delta_kernel<128><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    else if (D == 256) delta_kernel<256><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    else delta_kernel<512><<<pg, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, S, H);
    std::vector<float> d_old(rows_q);
    CUDA_CHECK(cudaMemcpy(d_old.data(), d_delta, rows_q * 4, cudaMemcpyDeviceToHost));
    if (D == 128)
      delta_warp_kernel<128><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    else if (D == 256)
      delta_warp_kernel<256><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    else
      delta_warp_kernel<512><<<d_blocks, THREADS>>>(d_o_f, d_do8, d_dos, d_delta, d_rows);
    std::vector<float> d_new(rows_q);
    CUDA_CHECK(cudaMemcpy(d_new.data(), d_delta, rows_q * 4, cudaMemcpyDeviceToHost));
    float md = 0.f, mr = 0.f;
    for (size_t i = 0; i < d_old.size(); ++i) {
      float ad = fabsf(d_old[i] - d_new[i]);
      if (ad > md) md = ad;
      float den = fmaxf(fabsf(d_old[i]), 1e-6f);
      mr = fmaxf(mr, ad / den);
    }
    printf("[O26 A/B] delta old(per-row) %.4f ms | new(warp-per-row) %.4f ms (%.3fx) | "
           "max_abs=%.3e max_rel=%.3e\n", do_ms, dw_ms, do_ms / dw_ms, md, mr);
    run_all();  // 恢复 CLI 选中路径的完整输出
  }

  // ---- O19 A/B（D=128）：主 kernel mma(BM=64,3 CTA/SM) vs wg2(BM=128,2 wg,256 线程,1 CTA/SM)
  //      同一 session 计时 + 逐元素对拍（只改归约结构/几何，数学口径不变）。----
  if (D == 128) {
    auto run_mma_sel = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (use_regdq)
        launch_bwd_main<128, 64, 32, true, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      else
        launch_bwd_main<128, 64, 32, false, false, true, true>(
            mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
            d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
    };
    auto run_wg2_sel = [&]() {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      launch_bwd_wg2<128>(mg2, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta,
                          d_lse, d_dq_acc, d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal,
                          ksplit2);
    };
    auto bench_sel = [&](auto&& fn, float* out) {
      fn();
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) fn();
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t_mma = 0.f, t_wg2 = 0.f;
    bench_sel(run_mma_sel, &t_mma);
    bench_sel(run_wg2_sel, &t_wg2);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_mma_sel();
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_wg2_sel();
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O19 A/B] main mma(BM64,grid=%d) %.4f ms | wg2(BM128,grid=%d) %.4f ms (%.3fx) | "
           "max_abs(wg2-vs-mma) dq/dk/dv=%.3e/%.3e/%.3e\n",
           mg.x, t_mma, mg2.x, t_wg2, t_mma / t_wg2, maxd(b_dq, a_dq), maxd(b_dk, a_dk),
           maxd(b_dv, a_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O21 A/B（D=128）：主 kernel KV tile BN=32（3 CTA/SM）vs BN=64（2 CTA/SM，tile 数减半）。
  //      同一 session 计时 + 逐元素对拍（只改 KV 分块宽度，数学口径不变）。----
  if (D == 128) {
    auto run_bn = [&](int bn) {
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      if (bn == 64) {
        if (use_regdq)
          launch_bwd_main<128, 64, 64, true, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 64, false, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      } else {
        if (use_regdq)
          launch_bwd_main<128, 64, 32, true, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
        else
          launch_bwd_main<128, 64, 32, false, false, true, true>(
              mg, d_q8, d_qs, d_k8, d_ks, d_v8, d_vs, d_do8, d_dos, d_delta, d_lse, d_dq_acc,
              d_dk_acc, d_dv_acc, S, H, Hkv, scale, (int)causal, ksplit);
      }
    };
    auto bench_bn = [&](int bn, float* out) {
      run_bn(bn);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_bn(bn);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float t32 = 0.f, t64 = 0.f;
    bench_bn(32, &t32);
    bench_bn(64, &t64);
    std::vector<float> a_dq(nq), a_dk(nkv), a_dv(nkv), b_dq(nq), b_dk(nkv), b_dv(nkv);
    run_bn(32);
    CUDA_CHECK(cudaMemcpy(a_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    run_bn(64);
    CUDA_CHECK(cudaMemcpy(b_dq.data(), d_dq_acc, nq * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dk.data(), d_dk_acc, nkv * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b_dv.data(), d_dv_acc, nkv * 4, cudaMemcpyDeviceToHost));
    auto maxd = [](const std::vector<float>& x, const std::vector<float>& y) {
      double m = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::fabs((double)x[i] - (double)y[i]));
      return m;
    };
    printf("[O21 A/B] main BN=32 %.4f ms | BN=64 %.4f ms (%.3fx) | "
           "max_abs(BN64-vs-BN32) dq/dk/dv=%.3e/%.3e/%.3e\n",
           t32, t64, t32 / t64, maxd(b_dq, a_dq), maxd(b_dk, a_dk), maxd(b_dv, a_dv));
    run_main();  // 恢复 CLI 选中路径
  }

  // ---- O21b A/B：冗余 fp32→fp32 convert 的代价（默认 cvt_on=0 已消掉）。acc 已别名到输出，
  //      `do_cvt=true` 即对同一缓冲做一次自拷贝 ⇒ 时间差就是 convert 的真实成本。----
  {
    auto run_pipe = [&](bool do_cvt) {
      quant();
      CUDA_CHECK(cudaMemset(d_dq_acc, 0, nq * 4));
      CUDA_CHECK(cudaMemset(d_dk_acc, 0, nkv * 4));
      CUDA_CHECK(cudaMemset(d_dv_acc, 0, nkv * 4));
      run_preprocess();
      run_main();
      if (do_cvt)
        convert_kernel<<<cvt_blocks, cvt_threads>>>(d_dq_acc, d_dk_acc, d_dv_acc, d_dq, d_dk,
                                                    d_dv, nq, nkv);
    };
    auto bench_pipe = [&](bool do_cvt, float* out) {
      run_pipe(do_cvt);
      CUDA_CHECK(cudaEventRecord(ev0));
      for (int i = 0; i < iters; ++i) run_pipe(do_cvt);
      CUDA_CHECK(cudaEventRecord(ev1));
      CUDA_CHECK(cudaEventSynchronize(ev1));
      CUDA_CHECK(cudaEventElapsedTime(out, ev0, ev1));
      *out /= iters;
    };
    float tt_on = 0.f, tt_off = 0.f;
    bench_pipe(true, &tt_on);
    bench_pipe(false, &tt_off);
    printf("[O21b A/B] end2end 保留convert %.4f ms | 直写输出(消convert) %.4f ms (%.4fx)\n",
           tt_on, tt_off, tt_on / tt_off);
    run_all();  // 恢复 CLI 选中路径
  }

  std::vector<float> mdq(nq), mdk(nkv), mdv(nkv);
  CUDA_CHECK(cudaMemcpy(mdq.data(), d_dq, nq * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdk.data(), d_dk, nkv * 4, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(mdv.data(), d_dv, nkv * 4, cudaMemcpyDeviceToHost));

  // P3-3：落盘 ours 输出（默认关，行为逐位不变）。
  if (!dump_prefix.empty()) {
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dq.npy", mdq.data(), (long long)mdq.size());
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dk.npy", mdk.data(), (long long)mdk.size());
    fa_bwd_save_npy_f32(dir + "/" + dump_prefix + "_dv.npy", mdv.data(), (long long)mdv.size());
    printf("[dump] ours -> %s/%s_{dq,dk,dv}.npy\n", dir.c_str(), dump_prefix.c_str());
  }

  auto print_cmp = [&](const char* name, const std::vector<float>& mine,
                       const std::vector<float>& ref) {
    DiffStat st = diff_stat(mine, ref);
    printf("  %-4s vs ref: max_abs=%.3e  max_rel=%.3e\n", name, st.max_abs, st.max_rel);
  };
  printf("[compare] ours vs fp32 ref (O from %s.npy)\n", o_name.c_str());
  print_cmp("dq", mdq, rdq.data);
  print_cmp("dk", mdk, rdk.data);
  print_cmp("dv", mdv, rdv.data);
  // P5-3 调试：head_dim>128 时按每 128 维一段给 max_abs，验证 GEMM3/4/5 的 N-tile
  // 循环每一段都正确（若某段漏算/错位，该段误差会 O(1) 明显大于其它段）。
  if (D > 128) {
    auto print_band = [&](const char* name, const std::vector<float>& mine,
                          const std::vector<float>& ref) {
      for (int bnd = 0; bnd < D / 128; ++bnd) {
        double mx = 0.0;
        for (size_t i = 0; i < mine.size(); ++i)
          if ((int)(i % (size_t)D) / 128 == bnd)
            mx = std::max(mx, std::fabs((double)mine[i] - (double)ref[i]));
        printf("    %s[d %3d..%3d] max_abs=%.3e\n", name, bnd * 128, bnd * 128 + 127, mx);
      }
    };
    print_band("dq", mdq, rdq.data);
    print_band("dk", mdk, rdk.data);
    print_band("dv", mdv, rdv.data);
  }

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
  // O21b：acc 指针已别名到 dq/dk/dv，不能重复 free。
  cudaFree(d_dq); cudaFree(d_dk); cudaFree(d_dv);
  return 0;
}
