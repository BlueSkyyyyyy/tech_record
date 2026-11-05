---
title: "通算并行研读（专题）：把通信藏进计算，还是把计算让给通信"
date: 2026-11-02
draft: false
weight: 1
series: ["comm-overlap"]
tags: ["通算并行", "系列", "分布式训练", "compute-comm-overlap", "流水线并行", "MoE"]
categories: ["训练框架"]
---

这个专题只研究一件事：**GPU 在等通信的时候，能不能不闲着；以及为了让它不闲着，要付什么代价。**

听起来像一句废话——把通信提前发、把 wait 推后，不就完了？真做起来，你会发现每一条 overlap 路径上都站着三个不肯让路的东西：**数据依赖、SM/带宽争用、语义正确性**。过去三年的论文，本质上是在这三个约束之间找新的空隙。

本系列从流水线气泡讲到 MoE 的 all-to-all，从 Megatron 的 bucket hook 讲到 Flux 的 tile 级融合，再到 2025–2026 年把通信压到 1–2 个 SM 的 T-CCL、把整个 MoE 层塞进一个 persistent kernel 的 Weave。**每一篇都有论文数字、源码行号和可运行的 CPU 模型**；没有 GPU 也能把账算清楚。2026-11 又补了 4 篇：总览雷达 + UniEP / MegaFlux / Entwine·mKernel 三篇深读，专门处理“最新论文”和“逐位可复现”这两个缺口。

## 先记住一个画面

把一张 GPU 想成一个**只有两个工人的小车间**：

- **算工**：只会做矩阵乘、attention、norm，占着 SM 干活；
- **传工**：只会搬数据，走 NVLink/IB，但常常也要占几个 SM 来发指令、做 reduction。

串行执行时，算工干活，传工抽烟；传工搬数据，算工抽烟。通算并行想干的事，就是让两个人**同时有活干**。

但车间里只有一套工具（SM）、一条走廊（HBM/L2 带宽）、一份电（功耗预算）。传工一忙，算工的工具和走廊就被占了一部分——所以"两个人同时干"的总产出，**不一定**等于"两个人各自干"的产出之和。这就是本专题反复出现的 **overlap tax**。

> **类比的边界**：现实中的工人可以加班、可以加机器；GPU 上 SM 数、寄存器、L2 带宽、功耗帽都是固定的。所以 overlap 的收益不是"凭空多出算力"，而是**把原本空转的时间，换成另一部分被拖慢的时间**。算总账时，后者必须减掉。

## 四种动作：切、错、融、让

把 2020–2026 的论文摊开，几乎所有手段都能归到四个动词：

| 动作 | 做什么 | 典型代表 | 代价 |
|---|---|---|---|
| **切** (split) | 把张量/算子/批次切成更小的独立单元，制造可并行的窗口 | Megatron 的 grad bucket、Flux 的 GEMM tile、Comet 的 shared tensor、NanoFlow 的 nano-batch | 切得太细，launch 和同步开销反噬；切得太粗，尾部暴露 |
| **错** (stagger) | 重排执行顺序，让 A 的通信落进 B 的计算窗口 | GPipe → 1F1B → Zero Bubble、DualPipe 的双向流水、MoE 的 chunk 流水 | 要更多显存存放 in-flight 的激活/梯度；调度本身变复杂 |
| **融** (fuse) | 把通信指令塞进计算 kernel，或让通信走 TMA/交换机等专用硬件 | Flux 的 AG+GEMM、T-CCL 的 TMA collective、INC-MoE 的交换机内 all-gather | 与硬件/形状强绑定，移植性差；kernel 变大后寄存器/occupancy 吃紧 |
| **让** (yield) | 显式分配 SM、warp、带宽、流给通信或计算 | Comet 的 CTA specialization、Weave 的动态 SM 调度、Lagom 的 NCCL 参数搜索 | 分区是静态的，workload 一变就错配；分少了通信慢，分多了计算慢 |

四个动作不是互斥的。Flux 同时做了"切"（tile）和"融"（塞进 GEMM kernel）；Weave 同时做了"切"（chunk）、"融"（megakernel）和"让"（SM 分区）。**论文之间的差异，往往不是谁用了哪个动作，而是谁把粒度选得更准、谁把代价算得更清。**

## 三条硬约束

### 1. 依赖：数据没到，就是不能算

流水线并行的气泡，本质是"下游 stage 的输入还没来"。1F1B 能比 GPipe 省内存，是因为它让每个 stage 同时只有 $p$ 个 microbatch 在飞；Zero Bubble 能把气泡压到接近 0，是因为它发现**反向的 weight gradient（W）不依赖下游**，可以拿来填 input gradient（B）等待的空隙。

MoE 更难：dispatch 的 all-to-all 没完成，expert GEMM 就不知道自己要算哪些 token；combine 没完成，下一层 attention 就拿不到输入。所以 MoE 的 overlap 必须把 token 切成 chunk，让"第 $i$ 块的 expert 计算"和"第 $i+1$ 块的 dispatch"同时发生——**切是错的前提**。

### 2. 资源：通信不是免费的

这是最容易被"overlap ratio"掩盖的一条。

NCCL 的 collective 不是魔法，它要占 SM 来跑 reduction kernel，要占 L2/HBM 带宽来读写数据，要占功耗。Comet 论文里 MoE 层的通信占到 **47%** 的时间；T-CCL 测量发现，128 KB 的 all-gather 在 NCCL 下平均占 **3.30 个 SM**，而它的实现只占 **1.32 个**——同样一个 collective，SM 占用可以差 2.5 倍。

更反直觉的是：**overlap 会让计算本身变慢**。Characterizing（[2507.03114](https://arxiv.org/abs/2507.03114)）在 4×A100/H100/MI250/MI210 上系统测量，overlap 执行时计算平均慢 **18.9%**、最高 **40.0%**；而串行执行只比 overlap 平均慢 **10.2%**。也就是说，很多场景下"overlap"赢的那一点，是从"计算被拖慢"里抠出来的。

### 3. 语义：梯度要对，optimizer 要同步

DP 的梯度 all-reduce 不能省；PP 的激活不能错位；MoE 的 token 不能丢。大部分 overlap 只是**改执行顺序**，不改数学。但也有论文走得更远：CO2（[2401.16265](https://arxiv.org/abs/2401.16265)）直接做 local-SGD，$\tau$ 步不同步，用陈旧参数继续算，把同步整个藏进后续 $\tau$ 步——这已经是在改语义，需要用 staleness penalty 和 momentum clipping 把收敛拉回来。

**判断一个 overlap 方案能不能用，先问它改没改语义；如果改了，再看它拿什么换收敛。**

## 一笔账：$T = T_{\text{compute}} + T_{\text{exposed}} + T_{\text{bubble}}$

后面每一篇都会回到这个式子。三个词分别对应三种浪费：

- $T_{\text{compute}}$：真正做浮点运算的时间，理论上不可避免；
- $T_{\text{exposed}}$：通信没被藏住、暴露在关键路径上的部分；
- $T_{\text{bubble}}$：因为依赖没满足，SM 空转的时间（流水线气泡是典型）。

如果通信总量是 $T_{\text{comm}}$，定义**重叠效率**：

$$
\eta = 1 - \frac{T_{\text{exposed}}}{T_{\text{comm}}}
$$

$\eta=100\%$ 表示通信全被藏住；$\eta<0$ 表示 overlap 反而比串行还慢（Flux 论文明确讨论过这种负效率）。

但只有 $\eta$ 不够。还要看**代价**：

- 显存：Zero Bubble 的 ZB-H2 要把激活从 $p$ 份抬到 $2p-1$ 份；DualPipe 要存 $PP+1$ 份激活、参数量翻倍。
- 算力：overlap tax 让计算慢 18.9%（平均）。
- 工程复杂度：一个 fused kernel 可能只对某几种 shape 有效，换模型就要重调。

所以本系列的评价标准是三个数一起看：**暴露时间降了多少、显存/算力付了多少、换 workload 还成不成立。**

## 一个反直觉的起点

先看三组数字，它们构成本专题的问题意识：

1. **Comet**（[2502.19811](https://arxiv.org/abs/2502.19811)）：MoE 层通信占 47%，它用 shared-tensor 切分 + CTA specialization 藏住 **86.5%**，单层 **1.96×**、端到端 **1.71×**。
2. **Weave**（[2609.21483](https://arxiv.org/abs/2609.21483)）：在 Comet 的基础上做 persistent megakernel + 动态 SM 调度，overlap ratio 从 Comet 的 **14.3%** 提到 **47.1%**，单层 **2.89×**、端到端 **1.33×**——但即使这样，GEMM 中段仍有 `mid bubble`，通信 SM 整段空闲。
3. **Characterizing**（[2507.03114](https://arxiv.org/abs/2507.03114)）：overlap 平均让计算慢 18.9%，很多"收益"其实被 tax 吃掉了。

三组数字放在一起，结论很清楚：**通算并行已经过了"有没有 overlap"的阶段，进入"粒度选得对不对、资源分得准不准"的阶段。** 2026 年的论文，比的不是谁能让通信和计算同时跑，而是谁能在正确的粒度上、用正确的 SM 数、让正确的 chunk 先跑。

## 怎么读

### 路线 A：先建立框架（推荐）

读 **一**（性能账本）→ **二**（流水线气泡史）→ **六**（MoE 三段式）→ **十一**（总结）。这四篇把"切错融让"和"三笔账"串起来，不需要先懂 CUDA。

### 路线 B：做训练框架/并行策略

读 **二**（GPipe/1F1B/ZB）→ **三**（DualPipe）→ **四**（DP bucket）→ **五**（TP 融合）→ **六**（MoE EP）。每篇都落到 Megatron/DeepEP/Flux 的具体代码路径。

### 路线 C：做 kernel/编译器

读 **五**（Flux/T-CCL/TileLink）→ **七**（persistent kernel/SM 分区）→ **八**（Syncopate/Lagom）→ **十**（测量）。重点看"粒度—资源—调度"三者的耦合。

### 路线 D：做推理

读 **九**（NanoFlow/TokenWeave/FlashComm）→ **七**（SM 分区）→ **十**（怎么测）。推理侧的约束和训练不同：decode 是 memory-bound，通信占比随 batch 变小，overlap 的阈值更敏感。

## 目录

1. [先算清三笔账：关键路径、暴露通信与 overlap tax]({{< relref "comm-overlap-01-performance-model" >}})
2. [流水线气泡消灭史：GPipe → 1F1B → Zero Bubble]({{< relref "comm-overlap-02-pipeline" >}})
3. [DualPipe：双向流水如何把 bubble 压到 $(PP/2-1)$]({{< relref "comm-overlap-03-dualpipe" >}})
4. [数据并行：bucket、hook 与 reduce-scatter 的时差]({{< relref "comm-overlap-04-data-parallel" >}})
5. [张量并行：从 all-reduce 到 AG+GEMM 融合]({{< relref "comm-overlap-05-tensor-parallel" >}})
6. [MoE 专家并行：dispatch、expert、combine 的三段式]({{< relref "comm-overlap-06-moe" >}})
7. [Kernel 与 SM：persistent kernel、warp specialization 与分区]({{< relref "comm-overlap-07-kernel-sm" >}})
8. [编译与自动调度：Syncopate、Lagom 与 tile 级编程]({{< relref "comm-overlap-08-compiler" >}})
9. [推理侧：decode 的通算并行与它的阈值]({{< relref "comm-overlap-09-inference" >}})
10. [实测手册：用 nsys/ncu 把 overlap 账算出来]({{< relref "comm-overlap-10-measurement" >}})
11. [总结：2026 年的通算并行地图与五个开放问题]({{< relref "comm-overlap-11-summary" >}})
12. [补遗·总览：2026 最新论文雷达与阅读路线]({{< relref "comm-overlap-12-latest" >}})
13. [补遗一：UniEP 深读——persistent worker 与确定性 token mapping]({{< relref "comm-overlap-13-uniep" >}})
14. [补遗二：MegaFlux 深读——routing skew 与专家复制]({{< relref "comm-overlap-14-megaflux" >}})
15. [补遗三：Entwine、mKernel 与 2026 生态]({{< relref "comm-overlap-15-tile-ecosystem" >}})

## 配套代码

所有代码在 [`code/comm-overlap/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/comm-overlap)，**纯 CPU、无第三方依赖、每个文件都能 `--test` 自测**：

| 文件 | 作用 | 对应文章 |
|---|---|---|
| `overlap_model.py` | 统一性能账本：PP/DP/TP/MoE 的解析式 | 一 |
| `schedule_sim.py` | 流水线调度模拟器（GPipe/1F1B/ZB-H2），输出 Gantt | 二 |
| `bucket_sim.py` | DP bucket 的尾延迟 vs launch 开销 | 四 |
| `moe_overlap.py` | MoE 三段流水 + 通信 SM 数扫描 | 六 |
| `sm_split_model.py` | 通信/计算 SM 分区的最优解 | 七 |
| `decode_overlap.py` | 推理 token 切分的收益阈值 | 九 |
| `overlap_audit.py` | 从 trace CSV 算 overlap ratio / exposed | 十 |

这些是**模型，不是实测**。它们的作用是让你在没卡的时候也能把趋势算对；真实数字以论文和你的 `ncu` 为准。

补遗里还多了一套 [**源码精读**](../comm-overlap-src/)：把 UniEP 与 Flux 的关键 kernel 从上游逐行复制下来、
只插注释不改代码（12 个文件、4536 行代码 + 2932 行注释），配 6 篇精读笔记。
模型算的是账，源码精读看的是账本是怎么被写出来的 —— 谁在等、等什么、等的时候谁在干活。

## 结论

- 通算并行的四种动作：**切、错、融、让**；三条约束：**依赖、资源、语义**。
- 评价一个方案，至少看三个数：**暴露时间、显存/算力代价、跨 workload 的稳定性**。
- 2026 年的前沿不再是"让通信和计算同时跑"，而是**在 tile/chunk 粒度上，用可计算的 SM 预算，让正确的部分先跑**。
- 反直觉但重要：**overlap 有 tax，有时不 overlap 更快**。本系列会把这些负结果和正结果一起记下来。

下一篇从最基本的账本开始：一条关键路径上，通信到底暴露了多少；以及为什么"overlap ratio 90%"也可能是一个坏指标。

> 配套代码：`code/comm-overlap/overlap_model.py`、`schedule_sim.py`
> 自测：`python3 overlap_model.py --test && python3 schedule_sim.py --test`
