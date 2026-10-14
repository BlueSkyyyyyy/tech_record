# MoE grouped GEMM 的 SM 预算曲线：给通信让出一个 SM，计算腿要付多少？

> 本文对应路线图 **P180-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 190 篇。
>
> Mega MoE 把 dispatch/combine 的通信 warp 和 linear1/linear2 的 MMA warp 塞进**同一个**
> persistent kernel、**同一块** SM（第 2/9/10/11 篇）。前面几篇把「通信-计算重叠」从三个方向量过：
> 两条 stream + 两个 kernel 重叠（第 144 篇）、micro-batch 流水线藏通信（第 165 篇）、
> 单 kernel 内 `setmaxnreg` + warp specialization（第 167 篇）、按 SM 分区并发（第 182 篇）、
> 单 kernel 内计算腿 × 访存腿的资源账（第 183 篇）。但**一个更基础的量一直没被单独测过**：
> **一条 grouped GEMM 自己的『SM 预算曲线』长什么样**——把可用 SM 数从 132 一路砍到 4，
> 计算腿的时延/吞吐怎么变、边际代价是多少、拐点在哪。
>
> 这不是个抽象问题：只要方案是「给通信留 k 个 SM、剩下的给计算」（第 165 篇的正对照就是这么做
> 的），这条曲线就是**决策的分母**——「通信墙钟能不能盖过计算腿少 k 个 SM 的损失」。
> 本篇在 **H100（sm90）** 上用**真实 SM90 FP8 grouped kernel** 把这条曲线量出来，结论有两面：
>
> - **不是简单的 1/k**。三条曲线（前向 L1 / 前向 L2=dgrad / 反向 wgrad）在 **SM 充足区都有一段
>   「平台」**：`132→84` 砍掉 **36%** 的 SM，L1 只慢 ~37%、L2 只慢 ~30%、**wgrad（访存受限）
>   几乎不变（534→558 us，甚至 84 SM 比 96/108 SM 还快）**；过了平台才是陡峭的拐点。
> - **每少一个 SM 的边际代价强烈依赖「计算受限 vs 访存受限」**：L1 在平台区 ~1.4–3 us/SM，
>   wgrad 在平台区 ~0–3 us/SM（有噪声为负），而在拐点区（<48 SM）涨到 ~10–120 us/SM。
>   ⇒ **给通信让出「访存受限腿」的 SM 几乎是免费的；让出「计算受限腿」的要贵得多**。
>
> **可跑性 / 边界**：Mega MoE 主 kernel 及其 UMMA/tcgen05/TMEM 是 **SM100 专属、仅分析**，
> 且它 `DG_HOST_ASSERT(num_sms == device.get_num_sms())`（`heuristics/mega_moe.hpp:69`）——
> **主 kernel 不允许用更少的 SM 启动**，本篇量的是「按 SM 分区」这一候选方案下计算腿的代价
> （与第 165/182 篇同一条设计轴）。本篇所有数字来自 H100 真机 SM90 FP8 grouped kernel。
> 目标卡代号敏感：全文用「目标卡」指代。
>
> 参考坐标：`~/github/DeepGEMM@057ca59`（Public release 26/09/30，v2.8.1）：
> `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`、`csrc/runtime/runtime.hpp`、
> `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` / `sm90_fp8_gemm_1d1d.hpp`、
> `csrc/jit_kernels/heuristics/sm90.hpp`。
> 原始输出：`harness/exp_sm_budget_curve.out.txt`（`harness/exp_sm_budget_curve.{py,sh}`；
> H100 80GB HBM3, sm90, 驱动 570）。

---

## 1. 为什么是「SM 预算」而不是「SM 数」

先厘清一件事：**Mega MoE 主 kernel 不会用少于全部 SM 的网格启动。**
`get_num_sms_for_mega_moe()`（`csrc/jit_kernels/heuristics/mega_moe.hpp:67-70`）直接断言
`num_sms == jit->device.get_num_sms()`，JIT launch 也照抄这个值当 `grid_dim`
（`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146/195`）。原因很直接：它的三组 warp
（dispatch / non-epilogue / epilogue，`sm100_fp8_fp4_mega_moe.cuh:33-49`）是**在同一块 SM 上**
靠 warp specialization 分工的，通信和计算**共享**每个 SM 的寄存器/发射槽/访存带宽。

所以「SM 预算」有两种解读，本篇要量的是**第二种**：

1. **单 kernel 内的时间片预算**：同一块 SM 上，通信 warp 和 UMMA warp 各占多少 issue 槽/寄存器
   （`setmaxnreg`）——第 167 篇量过（CUDA-core 腿只重叠 30–60%、换 WGMMA 后 0.62→0.93），
   第 183 篇量过（计算腿 × 访存腿重叠效率 0.877、计算腿 × 计算腿只有 0.333）。
2. **按 SM 切分**：把一部分 SM **整块让给通信**（第 165 篇的正对照：给 NVLink peer scatter
   一个「小 SM 预算」，micro-batch 流水线就达到理想重叠的 ~99%；第 182 篇：两 kernel 各占一组
   SM 才真正 co-reside）。这时要回答的是：**计算腿少 k 个 SM，慢多少？**

第二种是「把计算和通信拆成两块 SM」这条设计轴的分母——它和第 165/182 篇是同一条线，
只是那两篇量的是「通信腿 + 计算腿合起来」，从没把**单条 grouped GEMM 自己**的曲线画出来。

---

## 2. 源码事实：`num_sms` 是编译期模板参数，不只是 grid 宽度

`deep_gemm.set_num_sms(k)`（Python）背后是 `Runtime::set_num_sms`（`csrc/runtime/runtime.hpp:78-80`）：

```cpp
void set_num_sms(const int& new_num_sms) {
    DG_HOST_ASSERT(0 < new_num_sms and new_num_sms <= jit->device.get_num_sms());
    num_sms = new_num_sms;
}
```

这个值进 kernel 配置（`sm90_fp8_gemm_1d2d.hpp:92 .num_sms = runtime->get_num_sms()`），
再由 JIT **编进模板参数 `kNumSMs`**（`scheduler/gemm.cuh:20/34/38`）。它同时影响三件事：

1. **persistent grid 的宽度**：kernel 的 `grid_dim = num_sms`
   （`sm90_fp8_gemm_1d2d.hpp:125`，wgrad 走 1D1D 的 `sm90_fp8_gemm_1d1d.hpp:212` / `KGroupedContiguous` 分支 `:170`），
   每个 block 认领 task 的步长是
   `next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x`（`scheduler/gemm.cuh:186`）。
   所以 `num_sms` 就是「一波（wave）」有多宽。
2. **L2 swizzle 的分组宽度**：`kNum1DBlocksPerGroup` 由 `get_num_1d_blocks_per_group`
   按 `kNumSMs` 在 `{8, 16}` 里挑（`scheduler/gemm.cuh:14-27`，默认值见 `:38`）。
   也就是说 `num_sms` 还**改变块到 `(m_block, n_block)` 的映射**，从而改变 L2 局部性
   （`get_swizzled_block_idx`，`scheduler/gemm.cuh:107-141`）。
3. **cluster 合法性**：配置枚举里 `if (desc.num_sms % (cluster_m * cluster_n) != 0) continue;`
   （`csrc/jit_kernels/heuristics/sm90.hpp:77-78`）——奇数 `num_sms` 直接排除 cluster 2 的候选。

因此曲线**不该**是干净的 $\propto 1/k$：并行度、L2 复用、可用配置都在随 `num_sms` 一起变。

---

## 3. 实验设计

三条形状（`harness/exp_sm_budget_curve.py`），都取 MoE 生产量级、FP8、SM90：

| 腿 | kernel | 形状 | FLOP |
|---|---|---|---|
| 前向 L1 | `m_grouped_fp8_gemm_nt_contiguous`（1D2D） | G=8、每专家 m=1024、n=2I=4096、k=H=7168 | 481 G |
| 前向 L2 / dgrad | `m_grouped_fp8_gemm_nt_contiguous`（1D2D） | G=8、每专家 m=1024、n=H=7168、k=I=2048 | 240 G |
| 反向 wgrad | `k_grouped_fp8_gemm_nt_contiguous`（1D1D，fp32 累积） | G=8、m=7168、n=2048、每组 k_g=1024 | 240 G |

扫描 `num_sms ∈ {132, 120, 108, 96, 84, 72, 60, 48, 36, 24, 12, 8, 4}`
（`deep_gemm.set_num_sms`），每点先 warmup（含 JIT 重编译）再用 profiler 取纯 device kernel 时长。
「完美线性」参考线统一以 `num_sms=132` 为基准：`t_lin(k) = t(132)·132/k`。

> 说明：M 分组的 `block_m` 被启发式钉死为 `mk_alignment=128`
> （`heuristics/sm90.hpp:33`），所以前向 M 块数 = $\lceil 8192/128 \rceil = 64$；
> 每条腿的 `num_blocks` 还乘上 $\lceil n/\text{BLOCK\_N} \rceil$（1D2D 的 BLOCK_N 上限 192、
> 1D1D 上限 160，`heuristics/sm90.hpp:47-55`），而 wgrad 的 G=8>4 触发 `disable_multicast`
> （`heuristics/sm90.hpp:63-65`）。这些块池都被 persistent grid 的 `kNumSMs` 步长扫过。

---

## 4. H100 结果

### 4.1 前向 L1（计算受限，481 GFLOP，132 SM 时 1345 TFLOPS）

| num_sms | t (μs) | TFLOPS | t / t_lin | TF/SM |
|---|---|---|---|---|
| 132 | 357.7 | 1345 | 1.000 | 10188 |
| 120 | 374.3 | 1285 | 0.951 | 10710 |
| 96 | 457.3 | 1052 | 0.930 | 10956 |
| 84 | 491.9 | 978 | 0.875 | 11642 |
| 72 | 552.7 | 870 | 0.843 | 12087 |
| 60 | 627.5 | 767 | 0.797 | 12777 |
| 48 | 773.5 | 622 | 0.786 | 12957 |
| 24 | 1490.2 | 323 | 0.757 | 13450 |
| 4 | 8697.8 | 55 | 0.737 | 13826 |

**每少一个 SM 的额外 μs（相邻点差）**：`132→120` **+1.38 us/SM**、`120→108` +4.17、
`108→96` +2.75、`96→84` +2.88、`84→72` +5.07、`72→60` +6.23、`60→48` +12.16、
`48→36` +21.0、`36→24` +38.7、`24→12` **+122**、`12→4` **+360~1075**。

### 4.2 前向 L2 / dgrad（240 GFLOP，132 SM 时 1256 TFLOPS）

| num_sms | t (μs) | TFLOPS | t / t_lin | TF/SM |
|---|---|---|---|---|
| 132 | 191.5 | 1256 | 1.000 | 9515 |
| 120 | 204.5 | 1176 | 0.971 | 9803 |
| 96 | 232.5 | 1035 | 0.883 | 10777 |
| 84 | 248.5 | 968 | 0.826 | 11523 |
| 72 | 291.2 | 826 | 0.829 | 11473 |
| 48 | 430.4 | 559 | 0.817 | 11643 |
| 24 | 849.5 | 283 | 0.807 | 11797 |
| 4 | 4852.9 | 50 | 0.768 | 12390 |

**边际**：`132→120` +1.08 us/SM、`108→96` +0.88、`96→84` +1.33、`84→72` +3.56、
`60→48` +7.05、`36→24` +23.5、`24→12` +66.4。

### 4.3 反向 wgrad（访存受限，240 GFLOP，132 SM 时 450 TFLOPS）

| num_sms | t (μs) | TFLOPS | t / t_lin | TF/SM |
|---|---|---|---|---|
| 132 | 534.5 | 450 | 1.000 | 3409 |
| 120 | 567.9 | 424 | 0.966 | 3529 |
| **108** | **567.3** | **424** | 0.868 | 3926 |
| 96 | 572.7 | 420 | 0.779 | 4375 |
| **84** | **558.3** | **431** | 0.665 | 5129 |
| 72 | 591.5 | 407 | 0.604 | 5648 |
| 60 | 636.9 | 378 | 0.542 | 6294 |
| 48 | 761.5 | 316 | 0.518 | 6580 |
| 36 | 924.7 | 260 | 0.472 | 7225 |
| 24 | 1367.5 | 176 | 0.465 | 7329 |
| 4 | 7827.9 | 31 | 0.444 | 7681 |

**边际**：`132→120` +2.79 us/SM、**`120→96` ≈ −0.05 ~ +0.45 us/SM**、
**`96→84` −1.20 us/SM（噪声内为负）**、`84→72` +2.77、`72→60` +3.78、`60→48` +10.4、
`48→36` +13.6、`36→24` +36.9、`24→12` +108、`12→4` +331~961。

---

## 5. 读曲线：为什么有平台、拐点在哪

**（1）三条腿都有一段平台，wgrad 的平台最宽、最平。**
`132→84`（砍 36% SM）：L1 慢 37%、L2 慢 30%、**wgrad 只慢 4.4%**。
wgrad 的 `132→108→96→84` 段在 534–573 us 之间来回、边际代价在噪声内**趋于 0 甚至为负**
（84 SM 的 558.3 us 比 96 的 572.7、108 的 567.3 都快）。原因正是它的根因
（第 15/145 篇 ncu 归因）：**wgrad 是访存受限的**——132 SM 时只有 450 TFLOPS、
远低于同一形状计算受限时的上千 TFLOPS，HBM 带宽在 ~84 个 SM 时就已饱和，
再加 SM 只是让更多 CTA 抢同一条 HBM 总线。反过来，L1 在 132 SM 时已跑到 1345 TFLOPS，
接近张量核的甜区，所以每砍一个 SM 的收益损失更明显。

**（2）平台的另一层原因：`num_sms` 太小则「一波」装不满、且改坏 L2 swizzle。**
persistent grid 每波固定 `kNumSMs` 个块（`scheduler/gemm.cuh:186`）。当块池远大于 SM 数时
（wgrad 的块池含 8 组 × 56×16 量级），把它切成几波几乎无损；但当 `num_sms` 掉到 24 以下，
`t/t_lin` 从 ~0.46 直接崩到 0.44→0.46 的**拐点**，边际代价从几十涨到几百 us/SM——这是
「少到一波装不下几条并行的 K 累加链、尾波占比暴涨」。同一区间 `get_num_1d_blocks_per_group`
挑的 L2 分组宽度也在变（`scheduler/gemm.cuh:14-27`），块到 (m,n) 的映射跟着变。

**（3）拐点位置不是常数，随「计算受限程度」右移。**
L1/L2 的边际代价在 ~60–72 SM 就开始明显上升，wgrad 拖到 ~48–60 才上升。
换句话说：**越访存受限的腿，越能承受 SM 被砍。**

---

## 6. 与 Mega MoE 的关系：SM 预算的量化依据

把曲线代回「按 SM 分区」这条设计轴（第 165/182 篇的方案），反推「reserve k 个 SM 给通信腿」
的**净收益上界**（藏住的通信墙钟 − 计算腿损失）：

| 腿 | reserve 24 SM 的损失 | reserve 48 SM 的损失 |
|---|---|---|
| 前向 L1 | +66.7 us | +134.2 us |
| 前向 L2 / dgrad | +30.4 us | +57.0 us |
| 反向 wgrad | **+32.8 us** | **+23.8 us（同平台噪声）** |

结论有三条，都和 Mega MoE 的设计选择对得上：

1. **wgrad（访存受限）是「免费让出 SM」的首选腿**：reserve 到 48 个 SM，计算腿时延在噪声内
   几乎不涨（534→558 us）。如果通信腿要的是「小 SM 预算、长墙钟」（第 165 篇的 NVLink peer
   scatter 正对照），把它和 wgrad 段搭在一起最划算。
2. **计算受限的 L1 让 SM 要贵一个量级**：reserve 48 SM 要付 +134 us（37%）。
   这正是第 183 篇「重算腿吃张量核、只能躲进访存受限的 wgrad 段、不能躲进计算受限的 dgrad 段」
   那条资源结论的**对称面**——计算的争用不仅在 SM 内（寄存器/槽位），也在「你能拿几块 SM」。
3. **两 kernel 两条 stream 的 `1.09×`（第 182 篇）远不是天花板**：本篇说明「整块 SM 让出去」
   本身代价就随腿而异。要让重算/通信真正藏住，得让**访存受限腿**去吃闲着的 SM，
   或干脆走**单 kernel 内的异步张量核**（第 167/183 篇）——这与 Mega MoE 主 kernel
   偏不用「分区」而用「单 kernel warp specialization」是同一结论的两种说法。

---

## 7. 边界与结论

**结论**：

1. MoE 的三条 grouped GEMM 腿都有**明确的 SM 预算平台**：`132→84`（−36%）L1 −37%、
   L2 −30%、**wgrad −4.4%**。
2. **边际 SM 代价由「计算受限 vs 访存受限」决定**：平台区 wgrad ~0 us/SM、L1 ~1.4–3 us/SM；
   拐点区（<48 SM）统一涨到 ~10–120 us/SM。
3. 因此「给通信让 k 个 SM」不是无条件划算的——**它取决于让的是哪条腿**：访存受限腿几乎白送、
   计算受限腿要按上表计价。

**边界 / 局限**：

- 本篇的 kernel 是 **SM90 WGMMA** 代理；目标卡 SM100 的 UMMA/TMEM 主 kernel
  （`sm100_fp8_fp4_mega_moe.cuh`）**SM100 专属、仅分析**，其 `kNumSMs` 更是被
  `heuristics/mega_moe.hpp:69` 钉死为「全设备」，无法用 `set_num_sms` 复现本曲线。
- 8 卡共享节点，报数已 warmup + 取 device kernel 时长，仍可能有争用；wgrad 平台区的
  「负边际」是噪声，只作「≈0」读。
- 「反推净收益」用 device kernel 时长对比，未含 launch/同步开销；通信墙钟取自第 165 篇的量级，
  不是同一次运行。

**真机证据台账**（`harness/exp_sm_budget_curve.{py,sh,out.txt}`，H100 80GB HBM3 / sm90 / DeepGEMM 2.8.1）：

| 项 | 数字 |
|---|---|
| L1 132→84 SM（−36%） | 357.7 → 491.9 us（**+37%**） |
| L2 132→84 SM | 191.5 → 248.5 us（+30%） |
| wgrad 132→84 SM | 534.5 → 558.3 us（**+4.4%**） |
| wgrad 平台区边际 | ≈ 0 us/SM（96→84 噪声内为负） |
| L1 平台区边际 | +1.4–3 us/SM（132→72） |
| 拐点区边际（<48 SM） | L1 +12→38、wgrad +10→37 us/SM |
| reserve 48 SM 的计算损失 | L1 +134 us / L2 +57 us / wgrad +24 us |

主 kernel（UMMA/tcgen05/TMEM/warp specialization）SM100 专属、仅分析；以上数字均来自 H100 真机 SM90 FP8 grouped GEMM。

---

## 8. 参考

- DeepGEMM README / release（`#462` v2.8.1）：<https://github.com/deepseek-ai/DeepGEMM>
- persistent 调度器：`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:14-27/34-38/107-141/186`
- 运行时 SM 开关：`csrc/runtime/runtime.hpp:22/78-87`
- SM90 GEMM 发射与启发式：`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:92/125`、
  `sm90_fp8_gemm_1d1d.hpp:170/177/212`、`csrc/jit_kernels/heuristics/sm90.hpp:33/47-55/63-65/77-78`
- Mega MoE 全 SM 断言与单 kernel warp 划分：`csrc/jit_kernels/heuristics/mega_moe.hpp:67-70`、
  `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp:146/195`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:33-49`
- 相邻篇：第 142/145/152 篇（访存受限归因 / grouped 对偏斜免疫）、第 144/165 篇（通信-计算重叠）、
  第 167/182/183 篇（单 kernel warp specialization / SM 分区 / 资源账）
