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

于是所有流水线论文都在干同一件事：**把等待的时间，用别的 microbatch 的计算填上。** 从 GPipe 到 1F1B 再到 Zero Bubble，填法越来越精细：等时假设下气泡从 $3(p-1)$ 一路压到 $0$，真实硬件上最新一代自动调度在 6.2B 模型上做到了 0.29%。这一篇按时间线讲清楚每一步具体改了什么、代价是什么。

> 记号：$F$/$B$/$W$ 分别是前向、dX 反向、dW 反向，**整段反向 $=B+W$**。本节的 $T_B$ **只含 dX**（与 Zero Bubble 论文 Table 2 一致），所以 $(p-1)(T_F+T_B^{\text{整段}})$ 和 $(p-1)(T_F+T_B+T_W)$ 是同一个式子。完整约定见[专题的符号表]({{< relref "comm-overlap-00-index" >}})。

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

1F1B（One Forward One Backward）最早在 PipeDream 的异步设定里出现，后来在 Megatron-LM（[2104.04473](https://arxiv.org/abs/2104.04473)）的同步设定里成为标配。它的做法分三段：

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
# megatron/core/pipeline_parallel/schedules.py::forward_backward_pipelining_without_interleaving
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
```

即第 $s$ 个 stage 先做 $p-s-1$ 个 F（再和 `microbatch_group_size_per_vp_stage` 等参数交互）。非交错版在 `forward_backward_pipelining_without_interleaving`，交错版在 `forward_backward_pipelining_with_interleaving`（行号会随 Megatron 版本漂移，按函数名找更稳）。

### 2.2 交错式 1F1B：用更多通信换更小气泡

Megatron 还支持 **interleaved 1F1B**（1F1B-I）：每个设备放多个 model chunk（虚拟 stage），让同一张卡在不同 chunk 之间交错做 F/B。气泡大约缩小 $v$ 倍（$v$ 是每卡的 chunk 数）：

$$
\text{bubble rate} \approx \frac{p-1}{m \cdot v + p - 1}
$$

代价是两项：

- 每张卡要存 $v$ 份 chunk 的激活，显存上升；
- stage 之间的 send/recv 次数变成 $v$ 倍，通信量上升。

Zero Bubble 论文的 1.5B、$p=8,m=24$ 实测：1F1B 气泡 **24.31%**、显存 30 GB；1F1B-I 气泡 **10.55%**、显存 40 GB。**气泡降了一半多，显存多了 33%。**

## 3. Zero Bubble：把反向拆成 B 和 W

Zero Bubble（[2401.10241](https://arxiv.org/abs/2401.10241)）的观察是整条线的转折点：

> 反向其实由两部分组成：
> - **B**（input gradient，dX）：算完要立刻发给上游——**它在传递链上**；
> - **W**（weight gradient，dW）：只往 `main_grad` 里累加——**没人等它**。

两件事都要等同一份 dY，所以"W 必须先等 B 算完"并不成立。论文的约定是把 W 排在**同一个 B 之后**（dX 要尽早发出去），真正的关键性质是：**W 不在传递链上，它不产出任何别人需要的东西。**

> ⚠️ 这里最容易记错的一句话是"W 不依赖下游"。**W 依赖下游**——没有 dY 就没有 dW。正确的说法是"**W 不被下游依赖**"：下游不会因为 W 没算完而停下来。方向相反，但结论只从后一句来。

```text
1F1B:      F0 F1 F2 F3 [等 dY]  B0  B1  B2  B3     ← 反向不拆，一个 hop 要走完整反向
拆开 B/W:  F0 F1 F2 F3 [等 dY]  B0 W0  B1 W1 ...   ← W 紧跟 B：链上只走 dX
ZB-H1:     F0 F1 F2 F3 [F4..]   B0 W0  B1 W1 ...   ← W 的起点按级错开，越靠后越靠尾部
```

如果把 dY 的传递链看成流水线的"节拍"，那么拆开 B/W 之后，节拍上只剩 $F$ 和 $B$；被摘下来的 W 并没有消失（工作量一分没少），它被**堆到每一级的尾部**，正好去填原本纯空转的尾部楔子。

### 3.1 三个手写调度，气泡是这么一级一级掉下来的

论文先假设 $T_F=T_B=T_W$，给出两个手写调度。把"拆开但没重排"也算进来，一共四级（前三行的显存是论文 Table 2 的在飞数公式，最后一行是 $(2p-1)M_B$）：

| 调度 | 气泡（单步） | 峰值激活 | 具体做了什么 |
|---|---|---|---|
| 1F1B | $(p-1)(T_F+T_B+T_W)$ | $pM_B$ | 反向不拆，一个 hop 上要走完整反向 |
| 1F1B-split | $(p-1)(T_F+T_B)$ | $pM_B$ | 拆开 B/W，但 **W 紧跟 B**——"拆开了没重排" |
| ZB-H1 | $(p-1)(T_F+T_B-T_W)$ | $pM_B$ | warmup 不变；第 $s$ 级积压 $s$ 个 wgrad，W 按级往尾部推 |
| ZB-H2 | $(p-1)(T_F+T_B-2T_W)$ | $(2p-1)M_B$ | warmup 做到 $2p-1$；第 $s$ 级积压 $2s$ 个 wgrad |

$T_F=T_B=T_W=1$ 时四行分别是 $3(p-1)$、$2(p-1)$、$1(p-1)$、$0$：**每多让 W 离开链上一步，气泡就少一个 $(p-1)T_W$。**

这条式子可以自己验算，用一条谁都绕不过去的下界：最后一个 stage 最早只能在 $(p-1)T_F$ 之后拿到第一个 microbatch 的输入（前面有 $p-1$ 次前向要排队），而它自己的工作量是 $m(T_F+T_B+T_W)$，所以

$$
\text{cost} \;\ge\; (p-1)T_F + m(T_F+T_B+T_W)
$$

代入 $T_F=T_B=T_W=1,\;p=8,\;m=24$：下界 $=7+72=79$。而 1F1B 是 $72+21=93$（离下界差 $2(p-1)=14$）、1F1B-split 是 $72+14=86$，**ZB-H1 正好是 79**，ZB-H2 的 $72+0=72$ 则**低于下界**。

两句话：

- **等时假设下，ZB-H1 已经把单步气泡榨干了**：剩下的 $(p-1)T_F$ 就是那道填不掉的填充楔子（最后一级必须在门口等 $p-1$ 个前向）；
- **ZB-H2 的 0 低于下界，所以它不可能是单步性质**——这正是 §3.3 要讲的"跨 step"。

再看"减 $T_W$"到底减在哪：W 的 $mT_W$ 工作量一分没少，减的是**它占用传递链的时间**。链上少一个 $T_W$、尾部楔子里补一个 $T_W$，一省一补，等价于每个 hop 少 $T_W$。H2 是"再减一个"，因为它的 warmup 把 $2p-1$ 个 microbatch 压在手上，每个空洞都有两份 W 可填。

**别把"1/3"当普适结论。** $T_F=T_B=T_W$ 时 H1 的气泡是 1F1B 的 $1/3$，但论文的 1.5B profile 是 $T_F=18.5$、$T_B=18.1$、$T_W=9.3$：

$$
\frac{\text{H1}}{\text{1F1B}}=\frac{18.5+18.1-9.3}{18.5+18.1+9.3}\approx 0.59
$$

论文的实测也吻合（$24.31\%\to15.85\%$ 就是 0.65）。**真实硬件上 H1 拿到的是 0.6 倍，不是 1/3。** 顺带说一句，$T_W$ 在公式里是"越大越省气泡"，这只在 $T_W\lesssim(T_F+T_B)/2$ 的有效域里成立；按论文的 FLOPs 估计 $T_B+T_W=2T_F$，正好落在有效域边界附近，可以拿来做校验。

### 3.2 实测：24.31% → 0.29%

论文实测的气泡率（1.5B / 6.2B，$p=8,m=24$，A100）。显存列里 30/40/59 GB 是论文给的数字，ZB-H1 按"与 1F1B 相同的激活记账"记为 30 GB、ZB-H2 按 $(2p-1)M_B$ 推算为 59 GB（论文的表里没有 H1/H2 这两行）：

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
- **ZB-1p 和 ZB-H1 数字一样**，是论文的推测：1p 的显存上限把自动调度逼回了 H1 的形状。
- **ZB-2p 的代价是显存**：59 GB vs 1F1B 的 30 GB，接近 2 倍。论文摘要里的 +23% 是"相似显存下"，+31% 是"放宽显存"。
- 吞吐提升不是线性的：气泡从 24% 降到 4%，理论上限约 $1/(1-0.24)=1.32$×，实测 +23% 左右，差额来自 overlap tax 和通信。

### 3.3 为什么"零气泡"必须跨 step

§3.1 的下界已经给出结论：单步下界是 $(p-1)T_F+m(T_F+T_B+T_W)$，而 ZB-H2 的公式在等时下是 $m(T_F+T_B+T_W)$，比下界还小 $(p-1)T_F$。**所以"H2 = 0"只可能是跨 step 稳态**：本 step 尾部右移的 W 一路接上下一 step 的 warmup，两个 iteration 的拼接窗口里没有空转。

但这一步有个前提：step 之间不能有全局 barrier。而常规实现里 optimizer step 恰好有两个全局同步：

- 全局梯度范数（grad clip）；
- 全局 NaN/Inf 检查。

它们都要求所有 stage 的梯度先做完全局归约，于是每个 iteration 结尾被强制对齐，平行四边形又被切回梯形。论文 §4 的办法是把同步**后置**：用"部分归约的全局状态 + 本地值"乐观地先做一步更新，下一个 iteration 的 warmup 里再把全量归约结果传回来校验，不合法就用可逆的 AdamW 回滚（附录 C）。

**这就是为什么很多团队复现 ZB 时只拿到一半收益**——调度抄了，optimizer 同步没动。§4 的模拟器会把这件事量化出来。

## 4. 用模拟器把四级气泡打出来

`schedule_sim.py` 把 F/B/W 建成 DAG，按 stage 串行 + 跨 stage 依赖做 list scheduling。取 $p=4,m=8,F=B=W=1$：理想工作量 24，单步下界 $3+24=27$。

```text
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule 1f1b
makespan=33 bubble=9 bubble_rate=27.27%          # (p-1)(F+B+W) = 9
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule 1f1b-split
makespan=30 bubble=6 bubble_rate=20.00%          # (p-1)(F+B)   = 6
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh1
makespan=27 bubble=3 bubble_rate=11.11%          # (p-1)(F+B-W) = 3 ＝ 下界
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh2 --iters 6
per_step=24.50 bubble_rate=2.04%                 # 论文公式给 0；静态表只能到 2.04%
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh2 --iters 6 --barrier
per_step=27.00 bubble_rate=11.11%                # 加了全局同步，立刻退回 H1
```

Gantt 上看得更清楚（`--gantt`，1 字符 = 1 时间单位）：

```text
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule 1f1b-split --gantt
p0  FFFF...BWFBWFBWFBWFBW.BW.BW.BW
p1  .FFF..BWFBWFBWFBWFBWFBW.BW.BW.
p2  ..FF.BWFBWFBWFBWFBWFBWFBW.BW..
p3  ...FBWFBWFBWFBWFBWFBWFBWFBW...
```

```text
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh1 --gantt
p0  FFFF...BWFBWFBWFBWFBWBWBWBW
p1  .FFF..BFBWFBWFBWFBWFBWBWBWW
p2  ..FF.BFBFBWFBWFBWFBWFBWBWWW
p3  ...FBFBFBFBWFBWFBWFBWFBWWWW
```

- 第一张是"拆开但没重排"：每一级的 W 都紧跟自己的 B，链上的节拍仍然是"B 完 W"，所以只省掉链上的一个 $T_W$，气泡 6。
- 第二张是 ZB-H1：链上只剩 F 和 B（`BFBFBF...`），**W 被按级往后推**——级别越高，尾部堆积的 W 越多：第一级还是 `BW` 紧跟，最后一级整段 W 都堆在尾部（`WWWW`）。气泡 3，正好压到下界。
- 形状和论文的在飞数公式对得上：第 $s$ 级积压 $s$ 个 wgrad，在飞数 $=(p-s)M_B+s\,M_W$，总数仍是 $p$。

`zbh2` 那两行是同一份静态表在两种口径下的结果：不加 `--barrier`（跨 step）时每步 24.5，加了 barrier 就退回 27。**差的那 2.5 就是 optimizer 同步的价钱**——注意它连"H2 的 0"都没到，因为我们的静态表只是按论文形状手写的，没有自动调度搜出来的精确重排，也没有真正的 bypass 回滚。

这个差距本身比数字有用：**Zero Bubble 的难点不在"拆 B/W"这个想法，而在调度搜索和同步绕过。**

## 5. 工程取舍：什么时候该用哪种

| 场景 | 建议 | 原因 |
|---|---|---|
| 显存紧、$m$ 小 | 1F1B | 峰值激活约 $p$ 份，最稳 |
| 显存够、想压气泡 | ZB-2p / DualPipe | 气泡 <1%，但显存 ~2× |
| 多机、IB 慢 | ZB-1p 或 1F1B-I | 减少 stage 间通信次数；ZB-1p 不增加通信 |
| 单机 NVLink、$m$ 大 | 1F1B-I | 气泡随 $v$ 缩小，通信便宜 |
| 想少改代码 | 1F1B + `overlap_p2p_comm` | Megatron 现成开关，先拿 P2P 重叠的收益 |
| 只有 $p$ 小、$T_W$ 小 | 1F1B | H1 相对收益本来就只有 0.6 倍，$p$ 小则两者气泡都小 |

Megatron 里和 PP 通信重叠相关的开关：

- `overlap_p2p_comm`：P2P send/recv 与计算重叠（和 `batch_p2p_comm` 互斥检查）；
- `overlap_p2p_comm_warmup_flush`：warmup/flush 阶段也重叠；
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
- Zero Bubble 的关键不是"W 不依赖下游"（它依赖），而是"**W 不被下游依赖**"：把 W 从链上摘下来、按级堆到尾部。
- 四级气泡：$(p-1)(T_F+T_B+T_W)$ → $(p-1)(T_F+T_B)$ → $(p-1)(T_F+T_B-T_W)$ → $(p-1)(T_F+T_B-2T_W)$。等时下 H1 已是单步最优（等于下界 $(p-1)T_F+m(\cdot)$），**1/3 只在等时成立，真实约 0.6×**；H2 的"零"低于下界，必须跨 step + optimizer bypass。
- 真正接近零的是自动调度的 ZB-2p（6.2B 上 0.29%）；手写 H2 在 1.5B 上是 10.83%。

---

**系列导航**：[一：性能账本]({{< relref "comm-overlap-01-performance-model" >}}) · [三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}})

> 配套代码：`code/comm-overlap/schedule_sim.py`、`overlap_model.py`
> 自测：`python3 schedule_sim.py --test && python3 overlap_model.py --test`
