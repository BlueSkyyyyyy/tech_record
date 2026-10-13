# Mega MoE 前向 single-step 的 H100 预算归因：前向是 dgrad 的转置对偶（第 15 篇的对偶）

> 本文对应路线图 **P132-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 142 篇。
>
> 专题到第 141 篇，前向被拆过很多次：第 2/3/9/10/11 篇逐段精读 SM100 融合 kernel，第 99 篇做了
> **前向的 H100 数值复刻 + 误差分解**（端到端 vs fp64 rel 6.3e-2、四层可加分解），第 131 篇把通信
> 放回图里。但**前向的「性能为什么是这样」从来没有被单独量过**——第 99 篇只给了一个端到端吞吐
> （L1/L2 312/303 TFLOPS），没有 ncu 硬指标、没有 roofline、没有与**反向**（第 15 篇做过 ncu 归因）
> 的对照。
>
> 第 15 篇（`docs/15`）把**反向**四个 grouped GEMM 用 ncu 归因清楚了：**wgrad 访存受限、dgrad 计算
> 受限**。本篇补上它的**对偶**：用同样的 ncu 方法量**前向**的两段 routed GEMM，并加上前向特有的
> **L1 epilogue（in-place SwiGLU + top-k 权重 + UE8M0 量化）**的定时预算。核心结论一句话：
>
> **前向的 L1/L2 与反向的 d1g/d2g 是「转置对偶」形状，落在同一个计算受限 regime；前向根本不存在
> 反向 wgrad 那种访存受限的瓶颈——因为前向的收缩维是隐藏维（大），wgrad 的收缩维是 token（小）。**
>
> 可跑性：Mega MoE 融合 kernel 本体是 **SM100 专属**（`csrc/apis/mega_moe.hpp:256/367` 只在
> `arch_major==10` 派发；UMMA/tcgen05/TMEM/UTCCP 在 sm90 不可用）→ **仅分析**；但它的两段 routed
> GEMM 与 H100 的 SM90 `m_grouped` kernel 是同一数学/同一形状 family，**H100 真机可跑**（走
> `sm90_fp8_gemm_1d2d`，1D2D，见第 22/32 篇）。epilogue 的数值原语架构无关，逐行移植到裸 CUDA 计时。
>
> 原始输出：`harness/exp_forward_roofline.out.txt`（`harness/exp_forward_roofline.{py,sh}`、
> `src/moe_forward_epilogue.cu`；H100 80GB HBM3, sm90, ncu 2026.1.1）。

## 1. 前向的两段 routed GEMM 是什么

Mega MoE 把前向压进一个 kernel，但它的**两段 routed GEMM** 与 MoE 的常规写法完全同形（第 11 篇的
`m_grouped` contiguous：M=token、N/K 固定）。按 `csrc/apis/mega_moe.hpp:188-197` 的权重布局契约：

| 段 | 权重形状（每 rank） | 数学 | grouped 形态 |
|---|---|---|---|
| linear1 | `W1: [E_r, 2I, H]`（`intermediate_hidden_2 == 2*intermediate_hidden` `:197`） | `[m, H] · W1ᵀ → [m, 2I]` | `m_grouped`（M=token，**N=2I，K=H**） |
| linear2 | `W2: [E_r, H, I]` | `[m, I] · W2ᵀ → [m, H]` | `m_grouped`（M=token，**N=H，K=I**） |

中间夹一个**逐元素段**：把 L1 输出的 gate/up 两半做 SwiGLU、乘 combine 的 top-k 权重、再量化成
FP8（per-token UE8M0 gran 32/128）喂给 L2。在融合 kernel 里这段是 L1 epilogue
（`sm100_fp8_fp4_mega_moe.cuh:1066-1192`），在主 kernel 里被 L2 直接消费。

本篇的实验规模：`T=2048, topk=2, E=8` → routed `m = T*topk = 4096`（每专家 512 行）；
`H=7168, I=2048`。**一个诚实的抽象**：融合 kernel 的 routed 池里同时有本 rank 发出去/收回来的
token（`num_tokens <= num_max_tokens_per_rank`，`mega_moe.hpp:194`），单卡复刻把「所有 routed token
都落到本 rank 的 8 个专家」当规模——这给出与真机同量级的 `m` 和每专家块大小，是合理的**规模代理**
（误差复刻早已在第 99 篇做过，本篇只关心**性能来源**）。

## 2. 先算账：字节 / 算术强度 / roofline

H100 SXM 的两个常数（与第 15 篇同一口径）：FP8 dense tensor-core 峰值 **1979 TFLOPS**、HBM **3.35 TB/s**
→ roofline 拐点 **591 FLOP/B**。

**L1**（`m=4096, N=2I=4096, K=H=7168, E=8`）：FLOP = `2·m·2I·H` = **240.5 GF**；
读 `A[m,H]` fp8 = 29.4 MB、读 `B[E,2I,H]` fp8 = **234.9 MB**（8 个专家全触）、写 `D[m,2I]` bf16 = 33.6 MB、
SF ≈ 0.97 MB → 共 **298.8 MB** → **AI = 805 FLOP/B > 拐点 591 → 计算受限**。

**L2**（`m=4096, N=H=7168, K=I=2048`）：FLOP = `2·m·H·I` = **120.3 GF**；
读 `A[m,I]` = 8.4 MB、读 `B[E,H,I]` = **117.4 MB**、写 `D[m,H]` bf16 = 58.7 MB、SF ≈ 0.29 MB
→ 共 **184.8 MB** → **AI = 651 > 591 → 计算受限**。

关键形状事实：**前向 L1 的 FLOP 与反向的 w1g 完全相等（240.5 GF）、前向 L2 与 w2g 相等（120.3 GF）**
（见 `harness/exp_moe_backward_kernels.py` 的 `perf()` 与第 14 篇）。但形状不同——

- 前向 L1 `N=2I, K=H` 恰好是反向 **d1g**（`N=H, K=2I`）的 **N↔K 转置**；
- 前向 L2 `N=H, K=I` 恰好是反向 **d2g**（`N=I, K=H`）的 **N↔K 转置**；
- 反向 **w1g/w2g** 是 `k_grouped`：收缩维是 **token（K=512 = T·topk/E，很小）**、输出是 fp32 权重梯度。

这就是本篇的核心对照点：**前向 = dgrad 的转置对偶，而 wgrad 在前向里没有对应物**。

## 3. ncu 硬指标：前向 L1 / L2

`harness/exp_forward_roofline.out.txt` §KERNEL（ncu 单次 launch、热 cache；实测 kernel 名
`sm90_fp8_gemm_1d2d_impl`）：

| kernel | 形状 | Duration | Compute(SM) | DRAM 吞吐 | DRAM 读 | DRAM 写 | GMMA 指令 | SM 频率 |
|---|---|---|---|---|---|---|---|---|
| **forward L1** | m4096 N4096 K7168 | **203.7 us** | **76.5 %** | 55.3 % | 345.5 MB | 31.6 MB | 1 261 568 | 1.47 GHz |
| **forward L2** | m4096 N7168 K2048 | **111.4 us** | **66.4 %** | 48.5 % | 131.2 MB | 49.2 MB | 622 592 | 1.53 GHz |

读法：

- **两者都是计算受限**（Compute(SM) > DRAM 吞吐），与第 2 节的 AI 判断一致。Tensor pipeline 利用率
  76.5 % / 66.4 %——与反向 d1g 的 75.7 %、d2g 的 65.5 %（第 15 篇）**几乎逐格吻合**。
- ncu 的 GMMA 指令数 L1 : L2 = 1 261 568 : 622 592 = **2.03**，正好等于 FLOP 比 2.0（240.5/120.3），
  说明两个 kernel 的 MMA 粒度/指令效率一致。
- 单次 launch 吞吐：L1 `240.5 GF / 203.7 us = 1181 TFLOPS`（59.7 % of 1979）、
  L2 `120.3 / 111.4 = 1080 TFLOPS`（54.6 %）——与反向 d2g/d1g 的 55–63 % 同一档。
- ncu 实测 DRAM 读（L1 345.5 MB）比我按「A+B」估的 264 MB 多约 80 MB，来自 TMA 对齐取块与
  权重 `B` 在多 M 块间被重复从 L2 挤出后的重取；这是真值，写进账里。
- 占用率 14 %、单波（`Waves Per SM = 1`）、block 384 线程、cluster 2、寄存器 168、动态 smem 215 KB——
  与第 15 篇的 SM90 FP8 kernel 同款（SM90 grouped FP8 一贯用满 smem、低占用、深流水）。

## 4. event 计时与「前向 step 预算」

`--report` 段单次 launch 之外，再用 event 量两档（`.out.txt` §event timing）：

| kernel | single-shot（热启动 3 次测 1 次） | sustained（热启动 10 次测 100 次） |
|---|---|---|
| forward L1 | 229.9 us · 1046 TFLOPS · 52.9 % | **204.1 us · 1178 TFLOPS · 59.5 %** |
| forward L2 | 139.7 us · 861 TFLOPS · 43.5 % | **111.1 us · 1082 TFLOPS · 54.7 %** |

（single-shot 带冷 cache / 冷时钟，sustained 是持续负载下的稳态；方向与第 15 篇一致——
sustained 因 cache 变热而更快、也因持续负载略降频，两者相互作用。）

**L1 epilogue 的预算**（`src/moe_forward_epilogue.cu`，逐行移植
`sm100_fp8_fp4_mega_moe.cuh:1112-1192`）：读 `[m,2I]` fp32 gate/up（67.1 MB）+ 写 `[m,I]` fp8（8.4 MB）
+ SF，共 75.8 MB → **24.2 us、3129.7 GB/s（≈ 93 % of HBM 3.35 TB/s）**。语义核对：
`max|Δ|/amax = 5.5e-2`（fp8 e4m3 在 amax 附近一个 ulp ≈ 32/448 = 7.1e-2，符合）、
UE8M0 scale 与 fp64 参考**逐位相同**。

于是**前向 single-step 的 H100 预算**（稳态）：

```
linear1 m_grouped       204.1 us   (1178 TFLOPS)
L1 epilogue (SwiGLU+量化) 24.2 us   (3129.7 GB/s, 纯带宽段)
linear2 m_grouped       111.1 us   (1082 TFLOPS)
------------------------------------------------
合计                    339.4 us
计算 roofline 下界      121.5 + 60.8 = 182.3 us  -> 达下界 54%
```

两条结论：**① epilogue 只占前向 GEMM 的 ~7 %**（SwiGLU + 权重 + 量化的逐元素段是干净带宽受限、几乎
免费）；**② 前向两段的计算 roofline 下界之和 182.3 us，实测 315.2 us（GEMM only）达 58 %**——
与第 15 篇反向四个 kernel「达各自 roofline 下界 55–63 %」是同一个工程完成度。

## 5. 与反向的对照：为什么前向没有「wgrad 瓶颈」

把第 15 篇（反向，ncu 单次 launch）与本篇（前向）并排：

| | 形状（M/N/K） | 收缩维 | Duration | Compute(SM) | regime |
|---|---|---|---|---|---|
| forward L1 | m4096, N4096, K7168 | **H=7168** | 203.7 us | 76.5 % | 计算受限 |
| backward d1g | m4096, N7168, K4096 | **2I=4096** | 194.3 us | 75.7 % | 计算受限 |
| forward L2 | m4096, N7168, K2048 | **I=2048** | 111.4 us | 66.4 % | 计算受限 |
| backward d2g | m4096, N2048, K7168 | **H=7168** | 111.0 us | 65.5 % | 计算受限 |
| backward w1g | M7168, N4096, **K=512** | **token** | 998.8 us | 12.2 % | **访存受限** |
| backward w2g | M2048, N7168, **K=512** | **token** | 499.9 us | 12.4 % | **访存受限** |

**转置对偶**：`forward L1`（N=2I,K=H）与 `backward d1g`（N=H,K=2I）是 N↔K 互换，两者 FLOP 相同
（240.5 GF），实测时长 **203.7 vs 194.3 us**、tensor 利用率 **76.5 % vs 75.7 %**——几乎同一条曲线。
`forward L2`（N=H,K=I）与 `backward d2g`（N=I,K=H）同理（111.4 vs 111.0 us、66.4 % vs 65.5 %）。
**前向 dgrad 形状 = 反向 dgrad 的转置，落在同一计算受限 regime。**

**为什么前向没有 wgrad 那样的瓶颈**：wgrad 的收缩维是 **token**（`K = T·topk/E = 512`，只有 4 个
BLOCK_K），而它要写出一个**巨大的 fp32 权重梯度**（w1g 940 MB / w2g 470 MB）——算术强度只有 ~120，
远低于拐点 591 → 访存受限（第 15 篇 §4）。**前向的两段根本没有这个形状**：前向的收缩维是隐藏维
（H=7168 或 I=2048，几十个 BLOCK_K），输出是激活（bf16，几十 MB），AI 在拐点附近或以上。所以前向
天然计算受限、轻松跑到 dgrad 的档位。

把前向 single-step（339.4 us）与反向四段（single-shot 499.9+998.8+111.0+194.3 = **1804 us**）相比：
**反向 ≈ 前向的 5.3×**——差距**全部**来自 wgrad 两段（共 1498.7 us，占反向 83 %），而前向根本没有
这两段。这正是「MoE 反向为什么天然是四个独立 GEMM、且为什么 wgrad 是训练里最贵的一环」的量化来源。

## 6. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 15 篇 | 反向四 kernel 的 **ncu 归因**（wgrad 访存受限 / dgrad 计算受限） | 同一方法的**前向对偶**：L1/L2 + epilogue 预算 |
| 第 99 篇 | 前向**数值复刻 + 误差分解**（312/303 TFLOPS） | 前向**为什么是这个吞吐**：ncu 硬指标 + roofline |
| 第 14 篇 | 反向 kernel **接进图**、给吞吐 | 前向 step 与反向 step 的**预算对照** |
| 第 11/32 篇 | `m_grouped` 布局与 SM90 FP8 主 kernel | 前向用同一 kernel 的**实测硬指标** |
| 第 10 篇 | L1 epilogue 的**实现**（UTCCP/UE8M0/SwiGLU） | epilogue 的**数值核对 + 定时预算** |

**能跑/能核**（`harness/exp_forward_roofline.{py,sh}` + `src/moe_forward_epilogue.cu`）：前向两段
`m_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:181-250`）的 event 计时、字节/AI/roofline、
ncu 五 section + 定向 metric；L1 epilogue 的 fp64 对拍与定时。均 H100(sm90) 真机。

**仅分析**：Mega MoE 融合 kernel 的 UMMA/tcgen05/TMEM/UTCCP、L1 的 FP8 输出（2 TMA stage）与
L2 的 NVLink 直写 combine——**SM100 专属**。本篇的 `m_grouped` 复刻用 bf16 输出（SM90 1D2D 契约
`sm90_fp8_gemm_1d2d.hpp:148` 要求 D=bfloat16），与融合 kernel 的 L1 FP8 输出是一个**边界**；但两段 GEMM 的
**算术强度与 roofline regime** 不因输出 dtype 改变（输出只占字节的小头）。本机的数（尤其带宽）
受**共享 8 卡节点**影响，是**受争用的下界**——已在 idle 的 GPU1 上测，但仍非独占。
