# Mega MoE 前向 step 的通信-计算重叠：H100 上的真机量化（正负两个对照）

> 本文对应路线图 **P134-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 144 篇。
>
> 第 142 篇（P132-1）给出**前向的计算侧**预算（L1 m_grouped 204 us + L1 epilogue 24 us +
> L2 m_grouped 111 us = 339 us），第 143 篇（P133-1）补上**通信 + 归约侧**（L2 远端写回 ~186 us +
> combine 归约 ~107 us ≈ 292 us），并得出「两腿同量级（54% vs 46%）→ 融合 kernel 必须把它们
> **重叠**」的结论。但那个结论是**串行相加**推出来的，**没有真机量过前向这台的 overlap_factor**：
> 第 65 篇（P75）量的是通用「D2D copy ∥ matmul」（全卡不重叠、限 SM 才重叠），第 94 篇量的是**反向**。
>
> 本篇把这块补上，并给出一个**必须知道的反直觉事实**：
>
> **`把 NCCL a2a 发在侧流、把 grouped GEMM 发在主**流，在 H100 上几乎完全不重叠（实测
> overlap_factor ≈ 1.0）**——因为 NCCL 的 a2a kernel 会占满 SM，把计算 kernel 挤出去。
> 只有**给通信一个小 SM 预算**（第 63 篇 `get_theoretical_num_sms` 的模型），重叠才成立
> （正对照里 overlap_factor 达 **~1.98**）。
>
> 这正解释了 Mega MoE 的设计：它**不是**「NCCL 侧流 + 计算主流」，而是把 dispatch/combine
> 做进**同一个 kernel**、按 **warp** 解耦（dispatch warp 用 48/96 个寄存器、MMA/epilogue warp 用
> 208/168 个，见下）——SM 预算在寄存器/线程级就被分区了。
>
> 可跑性：真实 grouped GEMM 是 SM90 kernel（H100 可跑）；a2a 是 `torch.distributed` NCCL（真机通信）；
> 正对照是单进程 8 卡 NVLink。Mega MoE 融合 kernel 本体 SM100 专属 → 仅分析。本机 8 卡共享、
> 无 NVLS（fabric manager 未开）→ 通信带宽为**受争用下界**，报数时注明。
>
> 原始输出：`harness/exp_forward_overlap.out.txt`
> （`harness/exp_forward_overlap.{sh,py}`、`src/forward_overlap_nvlink.cu`；H100 80GB HBM3, sm90, 驱动 570）。

## 1. 要回答的问题

Mega MoE 的前向是「dispatch（a2a）→ L1 → SwiGLU → L2 → combine（a2a + 归约）」的单融合 kernel，
且**通信与计算重叠**（第 2/3/9 篇）。第 142/143 篇算出的**串行口径**是：

```
计算腿（P142）：L1 204.1 + L1 epilogue 24.2 + L2 111.1      = 339.4 us
通信+归约腿（P143）：L2 写回 186 + combine 归约 106.5        = 292.5 us
两腿串行相加                                                = 631.9 us
```

问题：**把这台机器的两条腿分别放到两条 stream 上，能重叠多少？** 以及：**要重叠，通信能不能
随便占 SM？**

## 2. 两条腿的真机实现

### 通信腿（`exp_forward_overlap.py`）

用 `torch.distributed`(NCCL) 的 `all_to_all_single` 复刻 DeepSeek 量级（每 rank `T=2048` token、
`topk=2`、`H=7168`）的**两张网**：

- **dispatch**：`T·topk` 行 × `H` 字节（fp8 激活）= **29.4 MB/rank**；
- **combine**：`T·topk` 行 × `H` × 2 字节（bf16 结果）= **58.7 MB/rank**。

合计 **88.1 MB/rank**——正是第 1 篇通信量模型里 `T·k·H·(1B+2B)` 的落地，也是第 143 篇
combine 段「读 58.7 + 写 29.4 = 88.1 MB」的同一口径。

### 计算腿（复用第 142 篇的 `build_m_grouped`）

用 DeepGEMM SM90 **真实 grouped GEMM** `m_grouped_fp8_gemm_nt_contiguous` 跑两段 routed GEMM
（每专家 512 行，共 8 专家、4096 行）：

- **L1**：`[m,H]·W1[e]ᵀ → [m,2I]`（`N=2I=4096, K=H=7168`）；
- **L2**：`[m,I]·W2[e]ᵀ → [m,H]`（`N=H=7168, K=I=2048`）。

L1+L2 事件计时 **319.7 us**（1163 TFLOPS）——与第 142 篇的 339.4 us（含 epilogue）同量级。

### 调度

通信发在**高优先级侧流** `comm_s`，计算发在**主**（默认）流；`serial = comm → wait → compute`，
`overlap = comm ∥ compute`（末端主流程等 `comm_s`）。两者结果张量独立，重叠**不改变语义**
（实测 comm 输出**逐字节** `bitsame=True`）。

## 3. 负结果：侧流 NCCL a2a 与 grouped GEMM **不重叠**

扫 `NCCL_MAX_NCHANNELS`（通信占用的 channel 数，channel ≈ SM/block 数）× 计算腿挑一次/三次
（`reps`，缩放计算量）：

| `NCCL_MAX_NCHANNELS` | reps | t_comm | t_comp | serial | overlap | **ovf** | eff_vs_ideal | comm GB/s |
|---|---|---|---|---|---|---|---|---|
| 默认 | 1 | 987.3 us | 319.7 us | 1651.6 | 1641.6 | **1.006** | 0.60 | 89.2 |
| 默认 | 3 | 992.5 | 955.1 | 2003.8 | 1981.3 | **1.011** | 0.50 | 88.7 |
| 8 | 1 | 1985.3 | 319.2 | 2587.9 | 2566.9 | **1.008** | 0.77 | 44.4 |
| 8 | 3 | 1983.0 | 955.6 | 3367.6 | 3383.8 | **0.995** | 0.59 | 44.4 |
| 16 | 1 | 1268.3 | 320.3 | 1697.6 | 1692.3 | **1.003** | 0.75 | 69.4 |
| 16 | 3 | 1268.6 | 959.9 | 2558.2 | 2541.2 | **1.007** | 0.50 | 69.4 |
| 32 | 1 | 996.0 | 320.3 | 1650.1 | 1624.7 | **1.016** | 0.61 | 88.4 |
| 32 | 3 | 952.0 | 958.2 | 1997.4 | 1992.2 | **1.003** | 0.48 | 92.5 |

**结论**：

1. **overlap ≈ serial（ovf ≈ 1.0），`eff_vs_ideal = overlap / max(comm,comp)` 只有 0.48–0.77**：
   即使把通信缩到 8 个 channel（`NCCL_MAX_NCHANNELS=8`，通信带宽从 89 掉到 44 GB/s），
   计算腿**也几乎没被隐藏**（`hidden_frac` 全程 ≤ 0.08）。
2. **限 channel 不是免费的**：把通信从 32→8 个 channel，`t_comm` 从 ~990 us 涨到 ~1985 us（**2.0×**），
   而重叠收益几乎为零 → **净亏**。这说明**光靠调 NCCL 的 channel 数，无法把 SM 预算「让」给计算**。
   NCCL 的 a2a kernel（无论几个 channel）会持续占用/争抢 SM，计算 kernel 只能等它。
3. **注意本机 a2a 本身很慢**：88.1 MB / 987 us = **89 GB/s**（第 131 篇 8 卡 a2a 渐近 ~120 GB/s，
   本机无 NVLS、8 卡共享）。这个数比第 143 篇用的「2 卡 NVLink 直连 316 GB/s」低一个量级——
   因为 **NCCL a2a ≠ NVLink 直连**：前者是 ring/tree 协议 + 协议开销，后者是裸 peer 写。

一句话：**「侧流 NCCL + 主流计算」这条路，在 H100 上走不通。**

## 4. 正对照：给通信一个 **SM 预算**，重叠立刻成立

负结果说明「让通信占满 SM」是问题。那**把 SM 按预算切给通信**呢？用单进程 8 卡 NVLink 做一个
**限块数**的 peer scatter（模拟 dispatch/combine 的远端写，88 MB/rank）∥ 一个**固定总计算量**
的计算受限 kernel；扫通信块数 `B`（`src/forward_overlap_nvlink.cu`）：

| `B`（通信块/SM） | t_comm | t_comp | serial | overlap | **ovf** | ideal=max | comm GB/s |
|---|---|---|---|---|---|---|---|
| 4 | 3289.0 us | 323.8 us | 3631.8 | 3319.4 | 1.094 | 3289.0 | 24.5 |
| 8 | 1661.3 | 331.6 | 2011.8 | 1672.8 | 1.203 | 1661.3 | 48.6 |
| 16 | 857.2 | 369.9 | 1227.6 | 885.3 | 1.387 | 857.2 | 94.2 |
| 32 | 507.9 | 415.2 | 914.6 | 527.7 | 1.733 | 507.9 | 159.0 |
| **48** | 411.2 | 395.6 | 828.4 | 418.3 | **1.980** | 411.2 | 196.3 |
| 64 | 387.4 | 488.7 | 893.5 | 499.6 | 1.788 | 488.7 | 208.4 |
| 96 | 388.5 | 632.8 | 1028.7 | 644.1 | 1.597 | 632.8 | 207.8 |
| 120 | 405.7 | 1435.6 | 1839.5 | 1499.6 | 1.227 | 1435.6 | 199.0 |

**结论**：

1. **`overlap ≈ ideal = max(t_comm, t_comp)`（每一行都成立）**——一旦把 SM 预算切好，
   重叠就是**近乎完美**的（比负结果的「overlap ≈ serial」前进了一大步）。
2. **存在最优通信块数 `B* ≈ 48`**（132 个 SM 里给通信 48 个）：`ovf` 从 `B=4` 的 1.09 升到
   **1.98**，再升到 `B=120` 又跌回 1.23。物理意义清楚：
   - `B` 太小 → 通信只有 4 个 SM，带宽塌到 24 GB/s，`t_comm` 3.3 ms 成为瓶颈（overlap 被通信拖住）；
   - `B` 太大 → 计算只剩十几个 SM，`t_comp` 涨到 1.4 ms 成为瓶颈；
   - `B*` 处 `t_comm ≈ t_comp`（411 vs 396 us），两腿**同时**是瓶颈，`ovf = (411+396)/411 ≈ 1.96` 达到理论上界。
3. **NVLink peer 搬运只需要 ~48 个 SM 就能跑满 ~200 GB/s**（`B≥48` 后带宽不再涨）；
   这正是第 63 篇 `get_theoretical_num_sms` 想表达的「通信只需一小撮 SM」的物理来源。

理论模型（与表一致）：把 `B` 个 SM 给通信、其余给计算，则
`overlap(B) ≈ max(t_comm(B), t_comp(B))`，
`ovf(B) = (t_comm+t_comp)/max(...)`，
在 `t_comm(B*) = t_comp(B*)` 处取到最优
`ovf* = (t_comm* + t_comp*)/t_comm*`。

## 5. 为什么 Mega MoE 用**单 kernel + warp 解耦**而不是「侧流 NCCL + 主流计算」

负结果（第 3 节）证明 NCCL 侧流走不通；正对照（第 4 节）证明「SM 预算分区」才是关键。
Mega MoE 的做法正是把**分区做到线程/寄存器级**：在一个 kernel 里划分五类 warp，用**寄存器预算**
把 SM 资源硬切给通信侧和计算侧（`sm100_fp8_fp4_mega_moe.cuh`）：

- `kNumDispatchRegisters = 48/96`、`kNumEpilogueRegisters = 208/168`，
  且 `DG_STATIC_ASSERT` 保证三类 warp 的寄存器总和 ≤ 64512（`:329-334`）——
  这等于**在编译期就把 SM 的寄存器文件按角色分了配**；
- dispatch warp（`warp_idx < kNumDispatchWarps`，`:342-344` 用 `warpgroup_reg_dealloc` 释放寄存器）
  与 MMA/epilogue warp 在同一 grid 内**并行推进**：dispatch 用 `grid_sync`（`:393`）等本地同步后
  立即开始拉远端 token，MMA warp 同时吃上一步的数据；
- 调度器按 **locality domain** 分区（`scheduler/mega_moe.cuh:344-370`，第 24/34 篇），
  ring buffer / pool block 让「通信块」与「计算块」流水（第 2/9 篇）。

对照第 65 篇的通用实验（D2D copy ∥ matmul：全卡不重叠、限 1/4 SM 才重叠 1.78×）与第 63 篇的
SM 预算模型，本篇把「为什么必须这样」在**前向的真实两腿**上坐实了：
**不是「并行发两条 stream」就能重叠，而是必须先按 SM 预算分区。**

## 6. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 142 篇（P132-1） | 前向**计算侧**预算（339 us） | 用真实 grouped GEMM 复现 320 us，并与通信腿一起调度 |
| 第 143 篇（P133-1） | 前向**通信+归约侧**预算（292 us，串行） | 串行口径的真机验证 + **重叠**量化 |
| 第 65 篇（P75） | 通用「copy ∥ matmul」重叠（限 SM 才重叠） | **前向真实两腿**（grouped GEMM × NCCL/NVLink）的重叠 |
| 第 63 篇（P73） | `get_theoretical_num_sms` 模型 | 正对照坐实「通信只需 ~48 SM 跑满带宽」 |
| 第 1 篇（P1-2） | 通信量模型 `T·k·H·(1B+2B)` | 落地成 88.1 MB/rank 的真实 a2a |
| 第 2/3/9 篇 | 融合 kernel 的 warp 解耦重叠机制 | 用寄存器预算行号（`:329-334`）解释「SM 预算分区」 |

**能跑/能核**（`harness/exp_forward_overlap.{sh,py}` + `src/forward_overlap_nvlink.cu`）：
负结果 8 组（NCCL a2a × 真实 grouped GEMM，扫 channel 数 × 计算量）、正对照 8 组（NVLink 限块数 ×
计算受限 kernel，扫 SM 预算）——全 H100(sm90) 真机。

**仅分析**：Mega MoE 融合 kernel 本体的 TMEM/UTCCP/NVLink 直写与 warp 级 mbarrier
（第 9/10/11 篇）；本机 a2a 带宽受**共享 8 卡节点**与无 NVLS 影响，是**受争用下界**。

**边界**：正对照的计算腿是**手写的固定 FLOP 内核**（非真实 GEMM），因为要让「SM 预算 → 计算时间」
可控可解析；它不追求复现 grouped GEMM 的具体 TFLOPS，只回答「重叠对 SM 预算的依赖形状」。
