---
title: "通算并行研读（二）：流水线气泡消灭史——GPipe、1F1B 与 Zero Bubble"
date: 2026-11-02
draft: false
weight: 3
series: ["comm-overlap"]
tags: ["通算并行", "系列", "流水线并行", "GPipe", "1F1B", "Zero Bubble", "megatron-lm"]
categories: ["训练框架"]
---

流水线并行是最早、也最直观的通算并行：把模型按层切成 $p$ 段，每段放在一张卡上，让 $m$ 个 microbatch 像工厂流水线一样往前走。它的通信是 stage 之间的 send/recv，量不大，但**依赖极强**——下游没有输入，就只能等。

于是所有流水线论文都在干同一件事：**把等待的时间，用别的 microbatch 的计算填上。** 从 GPipe 到 1F1B 再到 Zero Bubble，填法越来越精细，气泡从 24% 一路压到 1% 以下。这一篇按时间线讲清楚每一步具体改了什么、代价是什么。

## 1. GPipe：全 F 再全 B，简单但吃显存

GPipe（[1811.06965](https://arxiv.org/abs/1811.06965)）的做法最朴素：

1. 把 mini-batch 切成 $m$ 个 microbatch；
2. 先对所有 microbatch 做前向（F），把每段的激活都存下来；
3. 再做所有 microbatch 的反向（B）。

```text
stage 0:  F0 F1 F2 F3 ................ B0 B1 B2 B3
stage 1:  ... F0 F1 F2 F3 .......... B0 B1 B2 B3
stage 2:  ...... F0 F1 F2 F3 ...... B0 B1 B2 B3
stage 3:  ......... F0 F1 F2 F3 .. B0 B1 B2 B3
```

它的气泡是结构性的：最后一段要等前面 $p-1$ 段填满才能开始，反向又要等最后一段算完才能往回传。单步 makespan：

$$
T_{\text{GPipe}} = (m + p - 1)(T_F + T_B)
$$

理想工作量是 $m(T_F+T_B)$，所以气泡率：

$$
\text{bubble rate} = \frac{(p-1)(T_F+T_B)}{(m+p-1)(T_F+T_B)} = \frac{p-1}{m+p-1}
$$

$p=8,m=24$：$7/31=22.6\%$。想压低气泡，只能增大 $m$；但 GPipe 要同时存 $m$ 份激活，显存直接爆炸。GPipe 用**重计算**（discard + recompute）换显存，代价是约 20% 的额外计算（Zero Bubble 论文引用的数字）。

**GPipe 的遗产**：它确立了"microbatch + 流水"的基本结构，也暴露了核心矛盾——**气泡和显存是一对反比**。

## 2. 1F1B：让反向早点开始

1F1B（One Forward One Backward）最早在 PipeDream 的异步设定里出现，后来在 Megatron-LM（[2104.04473](https://arxiv.org/abs/2104.04473)）的同步设定里成为标配。它的做法：

- **warmup**：每个 stage 先连续做若干 F，把流水填满；
- **steady**：之后每做一个 F，就做一个 B（"1F1B"）；
- **cooldown**：最后把剩下的 B 排完。

```text
stage 0:  F0 F1 F2 F3  B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
stage 1:     F0 F1 F2  B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
...
```

### 2.1 气泡没变，显存变了

在相同 $m$ 下，1F1B 的气泡率和 GPipe **一样**：

$$
\text{bubble} = (p-1)(T_F+T_B)
$$

但它的峰值激活从 $m$ 份降到约 $p$ 份：因为反向做得早，前向的激活一用完就释放。**1F1B 是用"更早的反向"换显存，不是用"更聪明的调度"换气泡。** 这一点经常被误解——很多人以为 1F1B 比 GPipe 快，其实同 $m$ 下两者的理论气泡相同，1F1B 的优势是**同样显存能开更大的 $m$**，从而间接减小气泡。

Megatron 的实现里，warmup 数量是：

```python
# megatron/core/pipeline_parallel/schedules.py:906
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
```

即第 $s$ 个 stage 先做 $p-s-1$ 个 F（再和 `microbatch_group_size_per_vp_stage` 等参数交互）。`forward_backward_pipelining_without_interleaving` 在 `schedules.py:2279`，`with_interleaving` 在 `:1098`。

### 2.2 交错式 1F1B：用更多通信换更小气泡

Megatron 还支持 **interleaved 1F1B**（1F1B-I）：每个设备放多个 model chunk（虚拟 stage），让同一张卡在不同 chunk 之间交错做 F/B。气泡大约缩小 $v$ 倍（$v$ 是每卡的 chunk 数）：

$$
\text{bubble rate} \approx \frac{p-1}{m \cdot v + p - 1}
$$

代价是：

- 每张卡要存 $v$ 份 chunk 的激活，显存上升；
- stage 之间的 send/recv 次数变成 $v$ 倍，通信量上升。

Zero Bubble 论文的 1.5B、$p=8,m=24$ 实测：1F1B 气泡 **24.31%**、显存 30 GB；1F1B-I 气泡 **10.55%**、显存 40 GB。**气泡降了一半多，显存多了 33%。**

## 3. Zero Bubble：把反向拆成 B 和 W

Zero Bubble（[2401.10241](https://arxiv.org/abs/2401.10241)）的观察是整条线的转折点：

> 反向其实由两部分组成：
> - **B**（input gradient）：$\nabla x$，依赖下游传回来的梯度；
> - **W**（weight gradient）：$\nabla W$，只依赖本 stage 的输入和输出梯度。

**W 不依赖下游。** 所以当下游还没把梯度传回来、B 只能等着的时候，可以先算 W。W 就是免费的气泡填充物。

```text
1F1B:   F0 F1 F2 F3 [等] B0 B1 B2 B3
ZB:     F0 F1 F2 F3  W0 B0 W1 B1 W2 B2 ...
                    ↑ W 填了等 B 的空隙
```

### 3.1 三个手写调度

论文先假设 $T_F=T_B=T_W$，给出两个手写调度：

| 调度 | 气泡 | 峰值激活 | 直觉 |
|---|---|---|---|
| 1F1B | $(p-1)(T_F+T_B+T_W)$ | $pM_B$ | 基线 |
| ZB-H1 | $(p-1)(T_F+T_B-T_W)$ | $pM_B$ | 把 W 挪到 B 前，气泡降到 1/3，显存不变 |
| ZB-H2 | $(p-1)(T_F+T_B-2T_W)$ | $(2p-1)M_B$ | warmup 多做 F，尾部重排 W，理论可到 0 |

$T_F=T_B=T_W=1$ 时，1F1B 气泡 $3(p-1)$，ZB-H1 $1(p-1)$，ZB-H2 $0$。**ZB-H2 用约 2 倍的激活显存换零气泡。**

但真实模型里 $T_F,T_B,T_W$ 不相等（通常 $T_W<T_F<T_B$），手写调度会留下气泡。论文于是做了两件事：

1. **自动调度**：给定 $p,m$、显存上限、$T_F/T_B/T_W/T_{\text{comm}}$，用启发式 + ILP 搜每个 stage 的 F/B/W 顺序；
2. **optimizer 同步 bypass**：真正让气泡归零的最后一刀——optimizer step 的同步不在关键路径上等，而是用"参数版本 + 梯度版本"的握手绕过（论文 §4）。

### 3.2 实测：24.31% → 0.29%

论文 Table 5 的数字（1.5B / 6.2B，$p=8,m=24$，A100）：

| 调度 | 1.5B 气泡 | 6.2B 气泡 | 1.5B 显存 |
|---|---|---|---|
| 1F1B | 24.31% | 23.47% | 30 GB |
| 1F1B-I | 10.55% | 8.08% | 40 GB |
| ZB-H1 | 15.85% | 13.23% | 30 GB |
| ZB-H2 | 10.83% | 6.98% | 59 GB |
| ZB-1p | 15.85% | 13.23% | 32 GB |
| ZB-2p | **4.33%** | **0.29%** | 59 GB |

几个容易读错的点：

- **ZB-H2 不是零**。在不等长的 $T_F/T_B/T_W$ 下，手写 ZB-H2 只有 10.83%（1.5B），真正接近零的是自动搜出来的 **ZB-2p**（6.2B 上 0.29%）。
- **ZB-1p 和 ZB-H1 数字一样**，因为 1p 的显存上限把自动调度逼回了 H1 的形状。
- **ZB-2p 的代价是显存**：59 GB vs 1F1B 的 30 GB，接近 2 倍。论文标题里的 +23% 是"相似显存下"，+31% 是"放宽显存"。
- 吞吐提升不是线性的：气泡从 24% 降到 4%，理论上限约 $1/(1-0.24)=1.32$×，实测 +23% 左右，差额来自 overlap tax 和通信。

### 3.3 为什么"零气泡"需要跨 step

单步看，第一个 stage 的 F0 之前一定是空的，最后一个 stage 的 B 之后也一定有空隙。**严格单步零气泡是不可能的**（除非 $m\to\infty$）。

ZB 的"零"是**跨 step 稳态**：把 optimizer 的同步从关键路径上拿掉，让上一个 step 的 W 尾巴接下一个 step 的 F 头。论文 §4 的 bypass 就是干这个的。这也是为什么很多团队复现 ZB 时只拿到一半收益——调度抄了，optimizer 同步没动。

## 4. 用模拟器看三种调度

`schedule_sim.py` 把 F/B/W 建成 DAG，按 stage 串行 + 跨 stage 依赖做 list scheduling。取 $p=4,m=8,F=B=W=1$：

```text
$ python3 schedule_sim.py --schedule 1f1b --stages 4 --microbatches 8 --gantt
makespan=33 bubble=9 bubble_rate=27.27%

p0    FFFF......XXFXXFXXFXXFXX.XX.XX.XX
p1    .FFF....XXFXXFXXFXXFXXFXX.XX.XX..
p2    ..FF..XXFXXFXXFXXFXXFXXFXX.XX....
p3    ...FXXFXXFXXFXXFXXFXXFXXFXX......
```

- `X` 是反向（B+W 合并），`.` 是气泡；
- 每个 stage 的气泡都是 9，总 makespan 33，理想 24；
- 气泡率 $9/33=27.27\%$，和公式 $(p-1)/(m+p-1)=3/11=27.27\%$ 完全一致。

换成 ZB-H2（把 B/W 拆开、warmup 多做 F）：

```text
$ python3 schedule_sim.py --schedule zbh2 --stages 4 --microbatches 8
makespan=30 bubble=6 bubble_rate=20.00%
```

气泡从 27.27% 降到 20.00%。我们的 `zbh2` 只做了"W 填 B 的空隙"，没有自动重排、没有 optimizer bypass，所以离论文的 0.29% 还很远——**这个差距正好说明：Zero Bubble 的难点不在"拆 B/W"这个想法，而在调度搜索和同步绕过。**

## 5. 工程取舍：什么时候该用哪种

| 场景 | 建议 | 原因 |
|---|---|---|
| 显存紧、$m$ 小 | 1F1B | 峰值激活约 $p$ 份，最稳 |
| 显存够、想压气泡 | ZB-2p / DualPipe | 气泡 <1%，但显存 ~2× |
| 多机、IB 慢 | ZB-1p 或 1F1B-I | 减少 stage 间通信次数；ZB-1p 不增加通信 |
| 单机 NVLink、$m$ 大 | 1F1B-I | 气泡随 $v$ 缩小，通信便宜 |
| 想少改代码 | 1F1B + `overlap_p2p_comm` | Megatron 现成开关，先拿 P2P 重叠的收益 |

Megatron 里和 PP 通信重叠相关的开关：

- `overlap_p2p_comm`：P2P send/recv 与计算重叠（`schedules.py:1180` 附近有和 `batch_p2p_comm` 的互斥检查）；
- `overlap_p2p_comm_warmup_flush`：warmup/flush 阶段也重叠（`schedules.py:1678`）；
- `virtual_pipeline_model_parallel_size`：交错式 1F1B 的 chunk 数。

但注意：**P2P 通信重叠和气泡消除是两件事**。前者藏的是 stage 间 send/recv 的延迟，后者填的是 stage 等依赖的空转。很多团队开了 `overlap_p2p_comm` 发现只快 2%–3%，就是因为气泡才是大头。

## 6. 一个反直觉的结论

把 GPipe → 1F1B → ZB 串起来看，会发现一条清晰的线：

- GPipe 用**更多显存**换更小气泡（增大 $m$）；
- 1F1B 用**更早的反向**换显存，气泡不变；
- 1F1B-I 用**更多通信 + 显存**换气泡；
- ZB 用**更多显存 + 更复杂的调度 + optimizer 改造**换气泡。

**没有一条是"免费"的。** 每一步都在三个约束（显存、通信、语义）里挑一个付账。

下一篇看 DualPipe：DeepSeek 把这条线推到"双向流水"——每张卡同时处理两个方向的 chunk，让 F 和 B 在两个方向上互相填充，气泡公式变成 $(PP/2-1)(F\&B+B-3W)$。

## 结论

- GPipe 气泡 $(p-1)/(m+p-1)$，靠增大 $m$ 压，但显存吃 $m$ 份激活。
- 1F1B 同 $m$ 下气泡不变，优势是峰值激活降到约 $p$ 份；Megatron 的 warmup 是 `p - rank - 1`。
- 1F1B-I 用 $v$ 个虚拟 stage 把气泡缩小约 $v$ 倍，代价是显存和 P2P 通信。
- Zero Bubble 把反向拆成 B/W，W 填 B 的空隙；ZB-H1 显存不变、气泡约 1/3，ZB-H2 显存 ~2×、理论零气泡；真正接近零的是自动调度 ZB-2p（6.2B 上 0.29%），且需要 optimizer 同步 bypass。
- **单步零气泡不存在；零气泡是跨 step 稳态性质。**

---

**系列导航**：[一：性能账本]({{< relref "comm-overlap-01-performance-model" >}}) · [三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}})

> 配套代码：`code/comm-overlap/schedule_sim.py`、`overlap_model.py`
> 自测：`python3 schedule_sim.py --test && python3 overlap_model.py --test`
