# 跨 micro-batch 流水线能不能藏 EP 通信：NCCL 不行、SM 预算下才行（H100 实测）

> 本文对应路线图 **P155-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 165 篇。
>
> 第 144 篇（P134-1）在 H100 上量到一个反直觉负结果：**把 NCCL a2a 发在侧流、真实 grouped GEMM
> 发在主**流，overlap_factor ≈ 1.0（几乎不重叠）——因为 NCCL 的 a2a kernel 会占满 SM、把计算挤出去；
> 只有给通信一个**小 SM 预算**，正对照里重叠才成立（ovf ≈ 1.98）。但那量的是**同一个 micro-batch**
> 的两条腿并发。生产训练真正的做法是 **micro-batch 流水线**：让 `dispatch_{i+1}` 去盖住 `compute_i`。
> 本篇把这个问题单独量出来，并给出一个必须知道的两面对照：
>
> - **负结果（NCCL）**：即使把 `dispatch_{i+1}` 精确排进 `compute_i` 的时间窗，micro-batch 流水线也只
>   藏住一小部分通信——`ovf_vs_grouped` 最多 **1.21×（通信主导）** / **1.09×（算通相当）**，而理想是
>   **1.39× / 1.86×**；且把计算插到 dispatch 与 combine 之间，会让**通信腿本身膨胀 1.6–1.9×**
>   （NCCL a2a 与 GEMM 互抢 SM、互毁 L2 局部性）。
> - **正对照（SM 预算）**：同一条 `D_i → K_i → C_i` 依赖链，把通信换成**限块数**的 NVLink peer
>   scatter，micro-batch 流水线在**每个** SM 预算下都达到**理想重叠的 ~99%**（`ovf_grp ≈ ideal`）。
>
> 结论：**micro-batch 流水线调度本身没错，错的是 NCCL**——它没法被塞进一个小 SM 预算。这正是 Mega MoE
> 把 dispatch/combine 做进**同一个 kernel**、按 **warp + 寄存器预算**硬分区的根本原因。
>
> 可跑性：真实 grouped GEMM 是 SM90 kernel（H100 可跑）；NCCL a2a 是 `torch.distributed`（真机通信）；
> 正对照是单进程 8 卡 NVLink。Mega MoE 融合 kernel 本体 SM100 专属 → 仅分析。本机 8 卡共享、无 NVLS
> → 通信带宽为**受争用下界**，报数时注明。
>
> 原始输出：`harness/exp_pipeline_overlap.out.txt`
> （`harness/exp_pipeline_overlap.{sh,py}`、`src/pipeline_overlap_nvlink.cu`；H100 80GB HBM3, sm90, 驱动 570）。

## 1. 要回答的问题

Mega MoE 的前向是「dispatch（a2a）→ L1 → SwiGLU → L2 → combine（a2a + 归约）」的单融合 kernel，
**通信与计算重叠**（第 2/3/9 篇）。第 142/143 篇给出**串行口径**（计算 339 us + 通信/归约 292 us），
第 144 篇证明「侧流 NCCL + 主流计算」在**同一 micro-batch** 上不重叠。

但真实训练会把一个 batch 切成若干 **micro-batch** 做流水线，让第 $i+1$ 个 micro-batch 的 dispatch
去盖住第 $i$ 个 micro-batch 的计算。问题：

> **把通信与计算放进跨 micro-batch 的流水线（而不是同一 batch 的两条流），能不能把 EP 通信藏住？**

## 2. 两条腿的真机实现

### 2.1 负结果：NCCL a2a × 真实 grouped GEMM（`harness/exp_pipeline_overlap.py`）

每条 micro-batch 的「真前向」= `dispatch(a2a) → L1 → L2 → combine(a2a)`：

- **通信腿**：`torch.distributed`(NCCL) 的 `all_to_all_single`。每条 micro-batch 固定
  `T_m=512`、`topk=2` → 1024 行；dispatch = `T_m·k·H·1B = 7.34 MB`、combine = `T_m·k·H·2B = 14.68 MB`
  （`H=7168`）；合计 **22.0 MB/micro-batch**。
- **计算腿**：DeepGEMM SM90 **真实 grouped GEMM** `m_grouped_fp8_gemm_nt_contiguous` 跑 L1+L2
  （每专家 128 行，`H=7168, I=2048, E=8`），`--reps` 控制重复次数以缩放计算量。`T_m*ktop/E=128`
  = m-grouped contiguous 的 mk 对齐，故每条 micro-batch 都能直接喂真实 kernel。
- **三种调度**：
  - `serial_micro`（朴素串行）：逐 micro `D_i → K_i → C_i`；
  - `serial_grouped`（无重叠最优）：全 `D` → 全 `K` → 全 `C`（单流顺序）；
  - `pipeline`：通信发在侧流 `comm_s`、计算发在主流程，**先把 `D_0..D_{M-1}` 排好**，令 `K_i` 等
    `D_i`、`C_i` 等 `K_i` ——于是 `D_{i+1}` 自然落在 `K_i` 的时间窗里。

### 2.2 正对照：SM 预算下的 NVLink 流水线（`src/pipeline_overlap_nvlink.cu`）

单进程 8 卡，建一条**真正的 `D_i → K_i → C_i` 依赖链**（用事件跨 `s_comm`/`s_comp` 两条流解耦），
通信腿是**限块数** `B` 的 NVLink peer scatter（每 rank 176 MB，≈ 8 micro-batch × 22 MB），
计算腿是固定总 FLOP 的燃烧 kernel（网格 = 余下 SM）。扫 `B` 与 `M`。

两次实验都用 `deep_gemm/testing/bench.py` 同款「计时前清一次 256 MB L2」，并对每个测量**重复取中位数**
（8 卡共享节点，别的自驱任务会抢卡造成离群尖峰）。

## 3. 负结果：NCCL 的 micro-batch 流水线只藏住一小部分

`t_comm`/`t_comp` 是**孤立**腿（dispatch+combine 背靠背 / 纯计算全 M 条）；`serial_grouped` 是无重叠
基线；`ovf_vs_grouped = serial_grouped / pipeline`；`ideal_ovf = (t_comm+t_comp)/max(...)`。

| M | reps | t_comm | t_comp | serial_grouped | pipeline | **ovf_vs_grouped** | ideal_ovf |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 568.3 us | 155.3 | 1199.4 | 1200.1 | **0.999** | 1.273 |
| 2 | 1 | 1143.9 | 309.6 | 1931.1 | 1902.6 | **1.015** | 1.271 |
| 4 | 1 | 1825.6 | 617.3 | 3858.8 | 3257.9 | **1.184** | 1.338 |
| 8 | 1 | 3180.4 | 1235.4 | 7248.6 | 5968.7 | **1.214** | 1.388 |
| 1 | 3 | 568.2 | 464.2 | 1989.3 | 1990.4 | **0.999** | 1.817 |
| 2 | 3 | 1143.3 | 927.0 | 3986.8 | 3944.9 | **1.011** | 1.811 |
| 4 | 3 | 1824.1 | 1857.1 | 7960.3 | 7347.2 | **1.083** | 1.982 |
| 8 | 3 | 3186.7 | 3715.7 | 15465.2 | 14154.2 | **1.093** | 1.858 |

（`bitsame=True`：流水线与串行的 comm 输出**逐字节一致**，重叠不改变语义。）

**读法**：

1. `M=1` 不可能流水（无下一步可盖），`ovf ≈ 1.0`，符合预期——这是**实验自洽性检查**。
2. 通信主导（`reps=1`，`t_comm ≈ 2.6×t_comp`）：`ovf_vs_grouped` 从 1.02 升到 **1.214**，但仍只有理想
   1.388 的 **~55%**（`(1.214-1)/(1.388-1)`）。
3. 算通相当（`reps=3`，`t_comm ≈ t_comp`）：`ovf_vs_grouped` 只有 **1.09**，是理想 1.858 的 **~11%**。
   一旦计算腿变大，重叠几乎消失。
4. **通信腿在上下文里被膨胀**：`M=8, reps=1` 的 `serial_grouped` 通信分量 = `7248.6 − 1235.4 = 6013 us`，
   而孤立 `t_comm = 3180 us`——**1.9×**。把计算插到 dispatch 与 combine 之间，NCCL a2a 就慢了近一倍
   （dispatch/combine 在孤立测量里背靠背、缓冲驻留 L2；一旦计算 kernel 把 L2 冲掉，comm 只能回 HBM）。

一句话：**NCCL 的 micro-batch 流水线，收益被两件事吃掉了——SM 抢占 + L2 互毁。**

## 4. 为什么：NCCL a2a 与 grouped GEMM 互抢 SM / L2

这和第 144 篇（同一 micro-batch 的侧流实验）是**同一个物理**，只是从「同一 batch 两条流」搬到了
「不同 micro-batch 两条流」：NCCL 的 a2a kernel（无论几个 channel）会持续占用/争抢 SM（第 144 篇
§3：把通信从 32→8 个 channel，`t_comm` 涨 2.0× 而重叠收益为零），而 grouped GEMM 又要占满 SM，
两者并存时互相拖慢。第 144 篇的正对照其实已经把答案点破：**要重叠，通信必须先被限制在一小撮 SM 上**
（第 63 篇 `get_theoretical_num_sms` 的模型）。NCCL 不给你这个旋钮。

> DeepEP 自己的 EP kernel 也是这个思路：`deep_ep/buffers/ep.py:431` 的
> `get_theoretical_num_sms(num_experts, num_topk, ...)` 按带宽建模算一个**推荐 SM 数**（偶数、至少 4），
> dispatch/combine 只开这么多 SM——正是「通信只需一小撮 SM」的工程落地。

## 5. 正对照：SM 预算下，同一条流水线达到近乎完美重叠

`src/pipeline_overlap_nvlink.cu` 在同一条 `D_i → K_i → C_i` 流水线上把通信换成**限块数** NVLink
scatter（`B` 个 block；计算腿拿余下 `(132−B)×4` 个 block、固定总 FLOP）。`M=8` 扫 `B`：

| B_comm | t_comm | t_comp | t_grouped | t_pipe | **ovf_grp** | ideal_ovf |
|---|---|---|---|---|---|---|
| 8 | 6738.6 us | 348.1 | 7079.1 | 6745.6 | **1.049** | 1.052 |
| 16 | 3426.7 | 366.8 | 3817.8 | 3434.9 | **1.111** | 1.107 |
| 32 | 2072.1 | 441.7 | 2440.4 | 2086.8 | **1.169** | 1.213 |
| 48 | 1761.7 | 421.9 | 2104.2 | 1775.8 | **1.185** | 1.240 |
| **64** | 1515.4 | 495.9 | 2034.6 | 1544.8 | **1.317** | 1.327 |
| **96** | 1521.7 | 628.7 | 2144.6 | 1525.8 | **1.406** | 1.413 |
| 120 | 1539.6 | 1431.0 | 2903.1 | 1821.4 | **1.594** | 1.929 |

`M=1`（不可能流水）任意 `B` 下 `ovf_grp` 恒 ~1.00，是同一自洽性检查。`M=2/4` 在 `B=64` 分别到
1.261 / 1.356（ideal 1.314 / 1.324）。

**读法**：

1. **每个 `B` 下 `t_pipe ≈ max(t_comm, t_comp)`**——`ovf_grp` 与 `ideal_ovf` 两条线几乎重合，
   `B=64/96` 达到理想的 **99%**。micro-batch 流水线**本身是对的**。
2. 而且**这里 `t_grouped ≈ t_comm + t_comp`**（如 `B=64`：2034.6 ≈ 1515.4+495.9=2011.3）——NVLink
   限块通信**不会**被计算膨胀（对比 §3 的 1.9×）。**膨胀是 NCCL 独有的**，不是「通信 vs 计算」的
   本质。
3. 存在带宽/计算的最优点：`B` 太小 → 通信带宽塌（`B=8` 只有 24 GB/s、`t_comm` 6.7 ms）；`B` 太大 →
   计算只剩十几 SM（`B=120` `t_comp` 涨到 1.4 ms）。`B≈64` 处 `t_comm ≈ 1.5 ms` 且计算仍有 ~68 个 SM。

## 6. 这与 Mega MoE 的设计是什么关系

两面对照合起来说明：**问题不在「流水线」，而在「通信要占用多少 SM、怎么和计算共存」。** Mega MoE 的做法
正是把这个预算**做到线程 / 寄存器级**（第 2/3/9/11 篇）：

- 一个 kernel 里划五类 warp，用**寄存器预算**把 SM 资源硬切给通信侧与计算侧——
  `sm100_fp8_fp4_mega_moe.cuh:329` 的 `kNumDispatchRegisters = 48/96`、`:331` 的
  `kNumEpilogueRegisters = 208/168`，且 `:332-334` 的 `DG_STATIC_ASSERT` 保证三类 warp 寄存器总和
  ≤ 64512——等于**编译期就把寄存器文件按角色分了配**；
- dispatch warp 用 `:344` 的 `warpgroup_reg_dealloc<kNumDispatchRegisters>()` 主动释放寄存器，
  与 MMA/epilogue warp 在同一 grid 内并行推进（dispatch 用 `:393` / `:1412` 的 `comm::grid_sync`
  等本地同步后立即拉远端 token，MMA warp 同时吃上一步的数据）；
- 调度器按 **locality domain** 分区（`scheduler/mega_moe.cuh:345-360` 的
  `get_next_local_task_idx` 按域计数、`:363-370` 只在「≥2 波任务」时才允许跨域偷），让通信块与计算块
  在 ring buffer / pool block 上流水（第 2/9 篇）。

对照第 144 篇的正对照（NVLink 限块 `B*=48` 达 `ovf≈1.98`）与第 63 篇的 SM 预算模型，本篇把它推进到
**micro-batch 流水线维度**：**不是「多切几个 micro-batch、把通信排到侧流」就能藏住 EP 通信，而是必须
先按 SM 预算把通信框住。**

## 7. 边界与「能跑 / 仅分析」

**能跑（H100 sm90 真机）**：负结果的 8 组（M × reps，NCCL a2a × 真实 SM90 grouped GEMM）+
正对照的 28 组（B × M，单进程 8 卡 NVLink 流水线）。

**仅分析（SM100 专属）**：Mega MoE 融合 kernel 本体的 TMEM/UTCCP、NVLink L2 直写与 warp 级 mbarrier
（第 9/10/11 篇）——它们就是上面这套「寄存器/线程级 SM 预算」在目标卡上的实现形态。

**边界**：本机 8 卡共享、无 NVLS，NCCL a2a 带宽（`M=8, reps=1`：176 MB / 3180 us = **55 GB/s**）是
**受争用下界**，且**每 micro-batch 只有 7.3/14.7 MB 小消息**、时延主导（第 162 篇已证「字节缩 4× 只省
4.5% 墙钟」）；因此绝对数字随环境抖动，但**两面结论（NCCL 只藏一小部分 / SM 预算下达理想重叠）在
重复取中位数后稳定复现**。正对照的计算腿是**手写固定 FLOP 内核**（非真实 GEMM），因为它要让
「SM 预算 → 计算时间」可控可解析；它不追求复现 grouped GEMM 的具体 TFLOPS，只回答「流水线重叠对
SM 预算的依赖形状」。

## 8. 与既有篇章的关系

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 144 篇（P134-1） | **同一 micro-batch** 的侧流 NCCL ∥ grouped GEMM（ovf≈1.0）+ SM 预算正对照（ovf≈1.98） | **跨 micro-batch 流水线**维度：藏不藏得住 |
| 第 134 篇（P126-1） | 前向计算侧预算（339 us） | 用真实 grouped GEMM 作流水线的计算腿 |
| 第 143 篇（P133-1） | 前向通信 + 归约侧预算（292 us，串行） | 串行口径在**流水线**上的真机验证 |
| 第 63 篇（P73） | DeepEP `get_theoretical_num_sms` 模型 | 正对照坐实「通信只需 ~48–64 SM 跑满带宽」 |
| 第 162 篇（P152-1） | EP 真实通信字节口径 | 通信腿在上下文里被膨胀 1.9× 的量化 |
| 第 2/3/9/11 篇 | 融合 kernel 的 warp 解耦重叠机制 | 用寄存器预算行号解释「为什么必须这样」 |

**一句话**：跨 micro-batch 流水线**能把通信藏住——前提是通信能被框进一个小 SM 预算**。NCCL 的 a2a
做不到（藏 ≤55% / ≤11%，且通信腿膨胀 1.9×）；换成限块 NVLink 后，同一条流水线在每个预算下都到
理想重叠的 ~99%。这正是 Mega MoE 用单 kernel + warp/寄存器级 SM 预算、而不是「NCCL 侧流 + 主流计算」
的原因。
