---
title: "Megatron 源码精读（二十五）：一个 training step 的耗时怎么估算——计算、通信与气泡的完整账单"
date: 2026-09-07
draft: false
tags: ["megatron-lm", "系列", "训练框架", "性能估算", "roofline", "MFU"]
categories: ["训练框架"]
weight: 25
series: ["megatron-code"]
---

上一篇[《显存估算》]({{< relref "megatron-code-24-memory-estimation" >}})算了「塞不塞得下」，这一篇算另一半问题：**塞下了，一个 training step 要跑多久？**

运行时间的估算比显存难得多——显存是「加法」（几块账直接相加），时间却是「归并」（几段并行的事取最大值，串行的事才相加）。核心矛盾是：**单卡的计算时间、跨卡的通信时间、流水线的气泡时间这三者，有的能重叠、有的互相等待**，不是简单累加。

本篇建立一套「先拆成最小时间片，再按重叠关系合成」的估算框架：

1. **计算时间**：用参数量推出总 FLOP，除以峰值算力和 MFU，得到「纯算力耗时」；
2. **通信时间**：TP/DP/PP 三类通信各自的字节量 ÷ 有效带宽，得到「通信耗时」，再看哪些能藏进计算；
3. **气泡时间**：1F1B 流水线的 bubble 率，乘以 step 时长；
4. **合成**：串行的相加、重叠的取最大，最后反推 token/s 和 MFU，用「实测 MFU『被低估』时的目标」收尾。

---

## 1. 记账的三个基本量：FLOP、字节、带宽

时间估算的一切起点是三个数：

| 量 | 含义 | 谁贡献 |
|---|---|---|
| FLOP | 一个 step 总共要做的浮点运算次数 | 计算时间 |
| 通信字节 | 一个 step 内跨卡要搬运的数据量 | 通信时间 |
| 带宽/算力 | 硬件单位时间能搬多少字节 / 算多少 FLOP | 把上面两个换算成时间 |

先记两个必须分清的量级：

- **算力峰值**（如 H100 FP16/BF16 tensor-core ≈ 989 TFLOPS，A100 ≈ 312 TFLOPS），这是「理论天花板」，实际能达到的比例叫 **MFU（Model FLOPs Utilization）**。
- **通信带宽**分两类：**单机内 NVLink**（H100 ≈ 900 GB/s/方向，A100 ≈ 600 GB/s）和**跨机 InfiniBand/RoCE**（H100 每卡约 50~100 GB/s，8 卡共享一台交换机）。跨机通信几乎总是瓶颈，这也是为什么大模型训练「机内 NVLink 全速、机间 IB 拖后腿」。

> 精度对方与这些量的换算（承接第 24 篇）：BF16 下「2 字节=1 个元素」，所以「FLOP/元素」决定算力占主导还是带宽占主导，这正是 roofline 的精髓（`te-perf-roofline` 篇在算子级讲过，这里升维到整模型）。

---

## 2. 计算时间：由参数量决定的总 FLOP

### 2.1 一个 step 的 FLOP 从参数推出来——著名的「6P」经验式

训练一个 token、每一步（前向 + 反向）的 FLOP 可近似为 **每参数 6 个 FLOP** 乘以参与更新的 token 数。反向的代价是前向的约 2 倍（因为要算「激活的梯度 + 权重的梯度」两路），所以：

$$
\text{FLOP}/\text{token} \approx 6P
$$

（前向 $2P$ + 反向 $4P$，$P$ 为参数量。这个 $6P$ 对 decoder-only Transformer 的 dense 结构成立，MoE 会因激活参数变多而显著高于 $6P$。）

一个 step 处理 $B$ 个 token（$B$ = 全局 batch，注意不是 per-GPU batch），则：

$$
\text{FLOP}/\text{step} = 6P \cdot B
$$

**这是全篇最值钱的一行**：总 FLOP 完全由「参数量 × 全局 batch」决定，与并行怎么切**无关**——并行只是把这份总 FLOP 分给更多卡一起算，不改变总量。

### 2.2 单卡计算时间

总 FLOP 均分给 $N$ 张卡，每张卡的实际算力是 `峰值 × MFU`，于是：

$$
T_{\text{compute}} = \frac{6PB}{N \cdot \text{peak} \cdot \text{MFU}}
$$

例：7B 模型，全局 batch $B=4\times10^6$ token（如 2M seq 长度级累加），$N=1024$ 张 H100（989 T），MFU 取 40%：

$$
T_{\text{compute}} = \frac{6 \times 7\times10^9 \times 4\times10^6}{1024 \times 989\times10^{12} \times 0.4}
\approx 420\ \text{ms}
$$

这个量级就是大模型一步的「纯计算下限」。现实中 `T_compute` 是最诚实的地板——**任何估算低于它都不对**，剩下的通信/气泡只能让它变长。

### 2.3 MFU 为什么很难到 100%

MFU 上不去的原因（这也是 roofline 在整模型尺度的体现）：

- **非 GEMM 部分**（LayerNorm、attention 的 softmax、dropout、mask、embedding 查找）算力强度低，落在 roofline 的**内存受限区**，消耗时间却不贡献很多「有效 FLOP」；
- **GEMM 本身也到不了峰值的 100%**：shape 不是完美 tile、尾块浪费、wave 不齐；
- **通信没被完全隐藏**（下一节），GPU 等数据时空转，这部分不产生任何 FLOP；
- **激活重计算（第 4 篇）**额外 +100% 的前向 FLOP，直接把 MFU 「账面」拉低（尽管省了显存）。

经验值：BF16 + 纯 dense、调得好的大模型训练，MFU 通常在 **35%~55%**；FP8 因算力翻倍且 HBM 压力减半，MFU 可到 50%~65%。

---

## 3. 通信时间：三类通信 + 各自能否隐藏

通信的账，关键是「**哪些能被算进 $T_{\text{compute}}$ 里藏掉，哪些必须暴露**」。三类通信按 GPU 拓扑逐一算：

### 3.1 TP 的 all-reduce（机内，最「贵」但可部分隐藏）

TP 每层前向、反向各有一到两次 **all-reduce**（第 21 篇）。一次 all-reduce 的时延约：

$$
T_{\text{ar}} \approx \frac{2 \times \text{bytes}}{bw_{\text{nvlink}}}
$$

（all-reduce 用 ring 算法，每个字节要跑约 $2(n-1)/n$ 趟，近似 $2\times$ 于单向带宽。）TP 通信**只在 TP 组内发生**，TP 组通常 ≤8 卡、落在同一台机器，走 NVLink，带宽高、字节量（激活/梯度）中等。

关键判断：**TP 通信总量 ∝ 总激活量**（每个 microbatch 的激活都要 all-reduce 多次），而 `T_compute` ∝ 总 FLOP。两者比值 → 当 batch/seq 不大、模型不大时，TP 通信占比高，很容易暴露；batch 一大，计算变长、TP 通信被摊薄。

### 3.2 DP 的 grad all-reduce / ZeRO reduce-scatter（跨机，大头）

梯度归约（DP）处理的是**参数量级**的字节：每个 step 要把 $P$ 个参数的梯度在 DP 组内归约一次。ZeRO 下用 reduce-scatter（第 8 篇）把梯度切分，每卡最后只留 $1/\text{dp}$。

$$
T_{\text{reduce}} \approx \frac{2P \cdot \text{bytes\_per\_param}}{bw_{\text{inter}}}
$$

DP 组通常**跨机器**，走 IB/RoCE，带宽只有机内的 $1/10\sim1/20$。但注意：**梯度只有 $P$ 个参数**（不像 TP 反复归约激活），所以 DP 通信总量是 $O(P)$ 而非 $O(\text{层数}\times\text{激活})$。经验上 DP 通信是「少次大量」，靠 `overlap_grad_reduce`（第 20 篇）把 all-reduce 切成 bucket 边算边传，**大部分能藏进反向计算**。

### 3.3 PP 的 p2p send/recv（机内为主，量小）

PP 只在相邻 stage 之间传激活（第 18 篇），字节量 = 激活量、很小，时延才是主角。`overlap_p2p_comm`（第 20 篇）延迟 wait 后基本能藏掉。PP 的**真正代价不在通信，而在 bubble**（第 4 节），通信本身可忽略。

### 3.4 合成规则：`max` 而不是 `+`

三类通信与计算的合成规则是——**能 overlap 的取 `max`，不能 overlap 的分段相加**：

$$
T_{\text{step}} = \underbrace{\max\left(T_{\text{compute}},\ T_{\text{fwd\_comm}}^{(暴露)}\right)}_{\text{计算与可隐藏通信}} \;+\; \underbrace{T_{\text{comm}}^{(无法隐藏)}}_{\text{暴露的通信}} \;+\; T_{\text{bubble}}
$$

其中「暴露的通信」最常见的是：**DP 梯度 all-reduce 的尾部**、**ZeRO-3 的 param gather 启动延迟**、以及**microbatch 太小时 TP 通信来不及藏**。这些正是第 20 篇各种 `overlap_*` 开关要消灭的东西。

---

## 4. 气泡时间：流水线的固定税

第 18 篇已经给出 1F1B 的 bubble 率：

$$
\text{bubble} = \frac{pp\_size - 1}{num\_microbatches + pp\_size - 1}
$$

把它理解为「step 中有 bubble 这么一大比例的时间，GPU 在空转」。所以：

$$
T_{\text{bubble}} \approx \text{bubble} \times T_{\text{step}}
$$

代入可得一个**显式的放大系数**：

$$
T_{\text{step}} = \frac{T_{\text{compute}} + T_{\text{comm}}^{(暴露)}}{1 - \text{bubble}}
$$

这解释了 PP 的权衡：**pp_size 越大，单卡显存越省（第 24 篇），但 bubble 越大、step 越长**。`num_microbatches` 越大 bubble 越小，但会挤压显存（更多并飞的激活）。这是「显存-时间」的直接 trade-off，和上一篇的结论严丝合缝。

> interleaved（vp_size>1）能把 bubble 进一步压到约一半（第 18 篇 `schedules.py:913-914` 的 warmup 公式背后就是这个效果），但调度更复杂。

---

## 5. 汇总：一步时间的完整公式

把 §2–§4 拼起来，得到**单步耗时**的通用表达式：

$$
\boxed{
T_{\text{step}} = \frac{1}{1 - bubble}\left[
\max\left(\frac{6PB}{N\cdot peak\cdot MFU},\ T_{\text{comm}}^{(\text{暴露})}\right)
+ T_{\text{comm}}^{(\text{无法隐藏})}
\right]
}
$$

各符号：

| 符号 | 含义 |
|---|---|
| $P$ | 参数量 |
| $B$ | 全局 batch（token 数/step） |
| $N$ | 总卡数 |
| `peak` | 单卡峰值算力（按精度取 BF16/FP8 的 tensor-core 峰值） |
| `MFU` | 有效算力利用率（0.35~0.65） |
| `bubble` | 1F1B 气泡率 $(pp-1)/(m+pp-1)$ |
| $T_{\text{comm}}$ | 暴露的 TP/DP/PP 通信，见 §3 |

从 $T_{\text{step}}$ 可立即推出两个下游指标：

**token/s（吞吐）**：

$$
\text{tokens/s} = \frac{B}{T_{\text{step}}}
$$

**MFU 反推**（用实测 step 时间校验估算）：

$$
\text{MFU} = \frac{6PB}{N \cdot peak \cdot T_{\text{step}}^{(实测)}}
$$

这个反推式在实践中最有价值：**你测出一个 step 时间，代入就能反解出真实 MFU**，进而判断「是算力没吃满，还是通信没藏掉」。

---

## 6. 完整算例：7B 在 64×H100 上

取 7B 模型，BF16，`tp=2, pp=2, dp=16`（64 卡），全局 batch $B=2\times10^6$ token，microbatch 32 个。

1. **总 FLOP**：$6\times7\times10^9\times2\times10^6 = 8.4\times10^{16}$ FLOP。
2. **单卡计算**：每卡 $8.4\times10^{16}/64 = 1.31\times10^{15}$ FLOP；H100 BF16 peak 989 T、MFU 取 45% → $T_{\text{compute}} = 1.31\times10^{15}/(989\times10^{12}\times0.45) \approx 2.95$ s。
3. **bubble**：$pp=2, m=32$ → $(2-1)/(32+2-1) = 3\%$，放大系数 $1/0.97 \approx 1.03$。
4. **通信暴露**：DP 梯度 reduce-scatter 跨机，$P=7\times10^9$，BF16 下 $1.4\times10^{10}$ B，跨机有效带宽约 50 GB/s → 约 0.28 s，被 `overlap_grad_reduce` 藏掉大半，暴露约 0.1 s。
5. **合成**：$T_{\text{step}} \approx 1.03 \times (2.95 + 0.1) \approx 3.14$ s。

反推：token/s = $2\times10^6/3.14 \approx 6.4\times10^5$ token/s；MFU ≈ 42%（略低于取的 45%，因为通信暴露 + bubble 各吃了一点）。

> 对比 FP8：peak 翻倍（~1978 T，稀疏更高），HBM 压力减半，MFU 可到 55% → $T_{\text{compute}}$ 降到约 1.2 s，`T_step` ≈ 1.35 s，提速 **2 倍以上**。这就是 FP8 训练（第 6 篇 TE）真实收益的来源——**不是改显存，是改时间**。

---

## 7. 精度怎么改这笔账

承接第 24 篇的精度表，但这次看它对「时间」的影响：

| 精度 | 每 token FLOP 变 | 峰值算力 | 对 $T_{\text{compute}}$ 的净影响 |
|---|---|---|---|
| FP32 | 基准 6P | CUDA core ~67 T（H100） | 慢一个数量级，训练几乎不用 |
| BF16/FP16 | 基准 6P | 989 T | 基准 |
| FP8（TE） | FLOP 可视为不变（数值压缩不算「变快」） | ~1978 T（E4M3 两倍） | **时间减半**（若 HBM 不重新成为瓶颈） |
| FP4（TE） | 同上 | 更高 | 进一步提速，但数值风险大 |

要点：

- **BF16→FP8 的加速本质是「峰值翻倍」**，不是「FLOP 变少」——TE 把 GEMM 换成 fp8 tensor-core 路径，峰值约 2 倍。前提是模型足够大、绑的不是 HBM 而是算力（roofline 右侧）；小模型绑 HBM 时 fp8 提速有限。
- **优化器/通信字节量不随精度降太多**：FP8 只加速前向/反向 GEMM，梯度在 reduce 前通常又转回 BF16/FP32，所以 §3 的通信账不变——这正是「算得快了，通信占比反而上升」的原因，大卡配 FP8 后通信 overlap 更重要。

---

## 8. 估算的「诚实地板」与实测兜底

这套框架给的是**下限 + 结构**，不是精确预测：

- 先算出 $T_{\text{compute}}$ 作为**绝对下界**（任何 model config，step 时间不可能低于它）。
- 再叠加「暴露通信 + bubble」，得到**乐观估算**。
- 实测必然 ≥ 乐观估算，差距 = 那些没料到的（碎片、launch、NCCL 抖动、反向不等工作负载）。可以把它写进 `torch.profiler` 的 kernel 级对比来归因。

经验锚点：

- **MFU 是「体检指标」**：dense BF16 练得好 40%+，FP8 50%+，低于 30% 基本是通信暴露或重计算开销没消化。
- **跨机 DP 是首查对象**：把每卡 IB 带宽和 `overlap_grad_reduce` 打开（第 20 篇）放一起看，通常能找到最大的一块「暴露时间」。

---

## 9. 小结

- **总 FLOP 由 `6P·B` 决定，与并行无关**；并行只是把它分给 $N$ 卡，单卡时间 = `6PB / (N·peak·MFU)`，这是不可逾越的下限。
- **时间 = 归并不是加法**：计算与可隐藏通信取 `max`、不可隐藏通信相加、bubble 再放大 $1/(1-bubble)$。
- **三类通信分账**：TP 反复归约激活（机内 NVLink、可部分隐藏）、DP 归约一次梯度（跨机、`overlap_grad_reduce` 藏）、PP 传小激活（`overlap_p2p_comm` 藏，代价在 bubble 不在通信）。
- **bubble 是 PP 的固定税**：$(pp-1)/(m+pp-1)$，`num_microbatches` 越大越小，但和显存（第 24 篇）直接互斥。
- **精度改时间不改说法**：FP8 靠峰值翻倍让 $T_{\text{compute}}$ 减半（模型够大、绑算力时），通信账不变、占比反升。
- **实用神器**：用实测 step 时间反解 `MFU = 6PB/(N·peak·T)`，一眼定位「算力没吃满」还是「通信没藏掉」。

下一篇预告：把第 24/25 两篇的「显存 + 时间」联合起来，做一次「给定硬件预算（N 卡 × 显存 × 算力），反推最大可训模型和最快配置」的端到端容量规划实战。

（本篇为方法论篇，未引入新源码行号。涉及的通信 overlap、1F1B 调度、FP8 训练实现请分别回看第 20、18、6 篇，其行号均基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。）
