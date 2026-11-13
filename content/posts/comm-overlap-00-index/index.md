---
title: "通算并行研读（专题）：把通信藏进计算，还是把计算让给通信"
date: 2026-11-02
draft: false
weight: 1
series: ["comm-overlap"]
tags: ["通算并行", "系列", "分布式训练", "compute-comm-overlap", "流水线并行", "MoE"]
categories: ["训练框架"]
---

通算并行的本质就一句话：把**等**换成**干**，代价是资源、算力和显存——还有一笔不写在计时器上的，语义。

读完你会知道：四种手法（切、错、融、让）各自在换什么，一笔账怎么记，这个系列该按什么顺序读。

## 车间里只有两个工人，一套工具

把一张 GPU 想成一个只有两个工人的小车间。**算工**只会做矩阵乘、attention、norm，占着 SM 干活；**传工**只会搬数据，走 NVLink/IB，但发指令、做归约时也得占几个 SM。

串行执行时，算工干活，传工抽烟；传工搬数据，算工抽烟。通算并行想干的事，就是让两个人同时有活干。

可车间只有一套工具（SM）、一条走廊（HBM/L2 带宽）、一份电（功耗预算）。传工一忙，算工的工具和走廊就被占掉一部分——两人同时干的总产出，不一定等于各自单干之和。这笔额外开销就是本专题反复出现的 **overlap tax**。

> 类比的边界：现实中的工人可以加班、可以加机器；GPU 上 SM 数、寄存器、L2 带宽、功耗帽都是固定的。所以 overlap 的收益不是凭空多出算力，而是把原本空转的时间，换成另一部分被拖慢的时间。算总账时，后者必须减掉。

藏得好不好，单层和端到端是两个数：

| 方案 | 单层加速 | 端到端加速 |
|---|---|---|
| [Comet](https://arxiv.org/abs/2502.19811)：MoE 层藏住 86.5% 的通信 | 1.96× | 1.71× |
| [Weave](https://arxiv.org/abs/2609.21483)：overlap ratio 14.3% → 47.1% | 2.89× | 1.33× |

## 四种手法，论文通常只改其中一种

把 2020–2026 的论文摊开，几乎所有手段都能归到四个动词：

| 动作 | 做什么 | 代表 | 代价 |
|---|---|---|---|
| **切** split | 把张量/算子/批次切成更小的独立单元，制造可并行的窗口 | Megatron 的 grad bucket、Flux 的 GEMM tile、Comet 的 shared tensor、NanoFlow 的 nano-batch | 切太细，launch 和同步开销反噬；切太粗，尾部暴露 |
| **错** stagger | 重排执行顺序，让 A 的通信落进 B 的计算窗口 | GPipe → 1F1B → Zero Bubble、DualPipe 的双向流水、MoE 的 chunk 流水 | 要更多显存存 in-flight 的激活/梯度；调度本身变复杂 |
| **融** fuse | 把通信指令塞进计算 kernel，或让通信走 TMA/交换机等专用硬件 | Flux 的 AG+GEMM、T-CCL 的 TMA collective、INC-MoE 的交换机内 all-gather | 与硬件/形状强绑定，移植性差；kernel 变大后寄存器/occupancy 吃紧 |
| **让** yield | 显式分配 SM、warp、带宽、流给通信或计算 | Comet 的 CTA specialization、Weave 的动态 SM 调度、Lagom 的 NCCL 参数搜索 | 分区是静态的，workload 一变就错配；分少了通信慢，分多了计算慢 |

四个动作不互斥：Flux 同时做了“切”（tile）和“融”（塞进 GEMM kernel），Weave 同时做了“切”（chunk）、“融”（megakernel）和“让”（SM 分区）。论文之间的差异，往往不是谁用了哪个动作，而是谁把粒度选得更准、谁把代价算得更清。

## 这四招都绕不开一条：数据没到就是不能算

流水线的气泡，本质是下游 stage 的输入还没来。Zero Bubble 能把气泡压到接近零，是因为它发现权重梯度不在传递链上：算完只往 `main_grad` 里累加，谁都不等它，所以可以挪到任意空隙里执行。这不等于“W 没有依赖”——下游 dY 没回来，它一样算不了。

MoE 更狠：dispatch 的 all-to-all 没完成，expert GEMM 就不知道自己要算哪些 token。所以只能把 token 切成 chunk，让第 i 块的专家计算和第 i+1 块的 dispatch 同时跑——**切是错的前提**。

## 一笔账：$T = T_{\text{compute}} + T_{\text{exposed}} + T_{\text{bubble}}$

习惯上记成 $C+E+B$：真正做浮点运算的时间、露在关键路径上的通信、依赖没满足而空转的气泡。overlap 的全部工作，就是把后两项往第一项里塞。

> 一个容易误解的点：ZB-H2 的“零气泡”只在跨 step 稳态下成立，单步口径下仍有一个下界。两种口径在 `schedule_sim.py` 里用 `--barrier` / `--iters` 切换。

## 代价有三笔：资源、算力、显存

- **资源**：传工抢的工具、走廊和电，都是从算工那里借的。Comet 的 MoE 层有 47% 的时间在通信；NCCL 下一个 128 KB 的 all-gather 平均占 3.30 个 SM，T-CCL 只占 1.32 个——同一个 collective，SM 占用差 2.5 倍。
- **算力**：这份争用的另一面是计算本身变慢。Characterizing（[2507.03114](https://arxiv.org/abs/2507.03114)）在 4×A100/H100/MI250/MI210 上实测，overlap 时计算平均慢 18.9%、最高 40.0%，而串行只比 overlap 平均慢 10.2%。
- **显存**：藏得越深，要存的东西越多。ZB-H2 把在飞激活从 p 份抬到 $2p-1$ 份，DualPipe 要存 PP+1 份激活、每设备参数 2×。

所以评价一个方案，三个数要一起看：暴露时长降了多少、显存和算力付了多少、换个 workload 还成不成立。还有一笔不写在计时器上：一个 fused kernel 往往只对某几种 shape 有效，换模型就得重调。

## 第四笔最贵：它到底改没改语义

DP 的梯度 all-reduce 不能省，PP 的激活不能错位，MoE 的 token 不能丢。绝大多数 overlap 只改执行顺序、不改数学。

CO2（[2401.16265](https://arxiv.org/abs/2401.16265)）走得更远：直接做 local-SGD，$\tau$ 步不同步、用陈旧参数继续算，把同步整个藏进后续 $\tau$ 步，再靠 staleness penalty 和 momentum clipping 把收敛拉回来。

## 系列目录：模型层 → 算子层 → 系统层

第一篇定义后面所有篇共用的那笔账，其余按你关心的层挑：模型层决定“什么时候切”，算子层决定“切完落在哪块硬件上”，系统层决定“怎么量出来、怎么选”。

- [先算清三笔账]({{< relref "comm-overlap-01-performance-model" >}})：关键路径、暴露通信与 overlap tax，以及 overlap ratio 为什么会骗人

**模型层：一步训练怎么切成能重叠的段**

- [流水线气泡消灭史]({{< relref "comm-overlap-02-pipeline" >}})：GPipe 为什么必须全前向再全反向，气泡能不能归零
- [DualPipe：两个方向对开]({{< relref "comm-overlap-03-dualpipe" >}})：两个方向互相填空，代价是多少显存
- [数据并行：算完一桶，先发一桶]({{< relref "comm-overlap-04-data-parallel" >}})：梯度同步暴露在哪，桶该开多大
- [张量并行：通信挪不走，就把 GEMM 切开]({{< relref "comm-overlap-05-tensor-parallel" >}})：为什么只能按 tile 切、四条融合路线各付什么代价
- [MoE：通信比计算长，只能藏住一段]({{< relref "comm-overlap-06-moe" >}})：dispatch、expert、combine 哪一段能藏

**算子层：把通信塞进 kernel，或者给它划 SM**

- [Kernel 与 SM：不是 stream 叠不叠，是 SM 归谁]({{< relref "comm-overlap-07-kernel-sm" >}})：为什么这更像 SM 归属问题，而不是 stream 问题
- [参数之间不独立，所以要编译器]({{< relref "comm-overlap-08-compiler" >}})：四条路线各自自动化了什么

**系统层：测量、判断与前沿**

- [推理侧：decode 的阈值]({{< relref "comm-overlap-09-inference" >}})：为什么 decode 的通信是固定开销
- [实测手册]({{< relref "comm-overlap-10-measurement" >}})：用 nsys/ncu 把暴露时长量出来，而不是估出来
- [最新论文雷达]({{< relref "comm-overlap-12-latest" >}})：2025–2026 这批里哪几篇值得先读
- [UniEP 深读]({{< relref "comm-overlap-13-uniep" >}})：可复现性为什么是硬约束
- [MegaFlux 深读]({{< relref "comm-overlap-14-megaflux" >}})：routing skew 为什么要复制专家
- [Entwine、mKernel 与生态]({{< relref "comm-overlap-15-tile-ecosystem" >}})：tile 早完成为什么不等于通信能早开始
- [总结：一张按粒度分层的地图]({{< relref "comm-overlap-11-summary" >}})：全系列的位置和还没解决的坑

另有一个平行系列 [源码精读]({{< relref "comm-overlap-src" >}})：UniEP 与 Flux 的关键 kernel 从上游逐行复制、只插注释不改代码，配七篇笔记 —— [导读]({{< relref "comm-overlap-src" >}}) · [dispatch]({{< relref "comm-overlap-src-01-dispatch-gemm" >}}) · [combine]({{< relref "comm-overlap-src-02-gemm-combine" >}}) · [确定性]({{< relref "comm-overlap-src-03-determinism" >}}) · [AG+GEMM]({{< relref "comm-overlap-src-04-flux-ag-gemm" >}}) · [GEMM+RS]({{< relref "comm-overlap-src-05-flux-gemm-rs" >}}) · [两边对照]({{< relref "comm-overlap-src-06-compare" >}}) · [Triton vs CuTeDSL]({{< relref "comm-overlap-src-07-triton-vs-cutedsl" >}})

模型算的是账；源码精读看的是账本是怎么被写出来的——谁在等、等什么、等的时候谁在干活。

## 怎么读：三条路线

- 建立框架：账本 → 流水线 → MoE → 总结，不用先懂 CUDA。
- 做训练框架或并行策略：流水线 → DualPipe → 数据并行 → 张量并行 → MoE，每篇都落到 Megatron/DeepEP/Flux 的具体代码路径。
- 做 kernel、编译器或推理：张量并行 → SM 分区 → 编译调度 → 实测手册；推理侧先读 decode 那篇，它是 memory-bound，通信占比随 batch 变小，阈值比训练敏感得多。

## 代码在哪

全部代码在 [`code/comm-overlap/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/comm-overlap)：七个纯 CPU、无第三方依赖的文件，每个都能 `--test` 自测，覆盖账本、调度模拟、DP 桶、MoE 三段流水、SM 分区、decode 阈值和 trace 审计。

模型不是实测，没有卡也能把趋势算对；真实数字以论文和你的 `ncu` 为准。

## 带走一句

overlap 有 tax，有时不 overlap 反而更快；先问改没改语义，再问粒度对不对，最后才算省了多少。

下一篇从最基本的账本开始：一条关键路径上，通信到底暴露了多少，以及为什么 overlap ratio 90% 也可能是个坏指标。
