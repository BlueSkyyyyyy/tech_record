---
title: "通算并行研读（二）：流水线气泡消灭史——GPipe、1F1B 与 Zero Bubble"
date: 2026-11-02
draft: false
weight: 3
series: ["comm-overlap"]
tags: ["通算并行", "系列", "流水线并行", "GPipe", "1F1B", "Zero Bubble", "megatron-lm"]
categories: ["训练框架"]
---

**流水线的气泡是结构性的，量级是 $(p-1)$ 份单段耗时、跟段数 $p$ 成正比，三代方案谁也压不到零；GPipe、1F1B、Zero Bubble 分别解决气泡能不能小、显存能不能省、W 能不能从关键路径上摘掉，而 ZB 的零只在跨 step 且去掉 optimizer 同步时才成立**。

读完你会知道：

- 为什么 GPipe 必须先全 F 再全 B；
- 1F1B 省下的到底是显存，还是气泡；
- 复现 Zero Bubble 只拿到一半收益，缺的那一半在哪。

把模型按层切成 $p$ 段、每段放一张卡，$m$ 个 microbatch 像工厂流水线一样往前走。stage 之间只有 send/recv，量不大，但依赖极强：下游没收到输入就只能等。

## 1. 气泡是结构性的：$(p-1)$ 份等待躲不掉

GPipe（[1811.06965](https://arxiv.org/abs/1811.06965)）的做法最朴素：切 $m$ 个 microbatch，先对全部 microbatch 做完前向，再统一做反向。

它没得选。一个 microbatch 的反向要用它自己的激活，这份激活就得活到反向那一刻；而只要把 F 全排在前面，手上就会同时压着 $m$ 份激活。

```text
stage 0:  F0 F1 F2 F3 ................ B0 B1 B2 B3
stage 1:  ... F0 F1 F2 F3 .......... B0 B1 B2 B3
stage 2:  ...... F0 F1 F2 F3 ...... B0 B1 B2 B3
stage 3:  ......... F0 F1 F2 F3 .. B0 B1 B2 B3
```

最后一段要等前面 $p-1$ 段填满才有活干，反向又要等最后一段算完才能往回传。这两块空转合起来就是气泡，长度 $(p-1)(T_F+T_B)$（$T_F$、$T_B$ 是单段前向和单段 dX 反向的耗时）——**它只跟 $p$ 有关，跟 $m$ 无关**。

单步 makespan 是 $(m+p-1)(T_F+T_B)$，理想工作量是 $m(T_F+T_B)$，气泡率就是 $\frac{p-1}{m+p-1}$。$p=8,m=24$ 时是 $7/31=22.6\%$。

想压小它只有一条路：加大 $m$。但 GPipe 手上同时握着 $m$ 份激活，$m$ 一大显存就炸，只能用重计算（丢掉激活、反向时重算）来换，代价是约 20% 的额外计算（Zero Bubble 论文引用的数字）。

**GPipe 留下的遗产是这条反比**：气泡和显存，只能挑一个。

## 2. 1F1B：省的是显存，不是气泡

1F1B（One Forward One Backward）最早出现在 PipeDream 的异步设定里，后来成了 Megatron-LM（[2104.04473](https://arxiv.org/abs/2104.04473)）同步设定下的标配。它分三段：warmup 连续做若干 F 把流水填满，steady 每做完一个 F 就做一个 B，cooldown 把剩下的 B 排完。

同 $m$ 下它的气泡和 GPipe 一模一样，还是 $(p-1)(T_F+T_B)$。变的是峰值激活：从 $m$ 份降到约 $p$ 份，因为反向做得早，前向的激活一用完就释放。

**很多人以为 1F1B 比 GPipe 快，其实同 $m$ 下两者理论气泡相同**。1F1B 的真实优势是同样显存能开更大的 $m$，气泡是这么间接降下来的。

Megatron 里 warmup 的条数是 `p - rank - 1`，即第 $s$ 级先做 $p-s-1$ 个 F。实现在 `schedules.py` 的 `forward_backward_pipelining_without_interleaving` 和 `..._with_interleaving`。

### 2.1 交错式：拿通信和显存换 $v$ 倍气泡

交错式 1F1B（1F1B-I）让每张卡放 $v$ 个 model chunk（虚拟 stage），同一张卡在不同 chunk 之间交错做 F/B，气泡大约缩小 $v$ 倍，气泡率约 $\frac{p-1}{mv+p-1}$。

代价两项：每张卡要存 $v$ 份 chunk 的激活，显存上升；stage 之间的 send/recv 次数也变成 $v$ 倍。

Zero Bubble 论文在 1.5B、$p=8,m=24$ 上的实测：1F1B 气泡 **24.31%**、显存 30 GB；1F1B-I 气泡 **10.55%**、显存 40 GB。**气泡降了一半多，显存多了 33%**。

> **口径**：$T_B$ 只含 dX（与 Zero Bubble 论文 Table 2 一致），整段反向记作 $T_B+T_W$，所以 $(p-1)(T_F+T_B^{\text{整段}})$ 和 $(p-1)(T_F+T_B+T_W)$ 是同一个式子。完整符号见[第一篇]({{< relref "comm-overlap-01-performance-model" >}})。

## 3. Zero Bubble：把 W 从关键路径上摘下来

Zero Bubble（[2401.10241](https://arxiv.org/abs/2401.10241)）的观察是整条线的转折点：反向其实是两件事。

- **B**（dX）：算完要立刻发给上游，**它在传递链上**；
- **W**（dW）：只往 `main_grad` 里累加，**没人等它**。

两件事共用同一份 dY，所以"W 必须等 B 算完"并不成立。论文把 W 排在同一个 B 之后（dX 要尽早发出去），真正被利用的性质是：**W 不在传递链上，它不产出任何别人需要的东西**。

```text
1F1B:      F0 F1 F2 F3 [等 dY]  B0  B1  B2  B3     ← 反向不拆，一个 hop 走完整反向
拆开 B/W:  F0 F1 F2 F3 [等 dY]  B0 W0  B1 W1 ...   ← W 紧跟 B：链上只走 dX
ZB-H1:     F0 F1 F2 F3 [F4..]   B0 W0  B1 W1 ...   ← W 起点按级错开，越靠后越靠尾部
```

### 3.1 四级：每让 W 离开链一步，气泡少一个 $(p-1)T_W$

把 dY 的传递链看成流水线的节拍。拆开 B/W 之后，节拍上只剩 F 和 B；被摘下来的 W 工作量一分没少，它被**堆到每一级的尾部**，正好去填原本纯空转的尾部楔子。

论文假设 $T_F=T_B=T_W$ 给出手写调度，把"拆开但没重排"也算进来，一共四级。H2 能再减一个 $T_W$，因为它 warmup 就压着 $2p-1$ 个 microbatch，每个空洞都有两份 W 可填——峰值激活也正是在这里涨到 $(2p-1)M_B$。

| 调度 | 单步气泡 | 1.5B 实测 | 6.2B 实测 |
|---|---|---|---|
| 1F1B | $(p-1)(T_F+T_B+T_W)$ | 24.31% | 23.47% |
| 1F1B-split（W 紧跟 B） | $(p-1)(T_F+T_B)$ | — | — |
| ZB-H1 | $(p-1)(T_F+T_B-T_W)$ | 15.85% | 13.23% |
| ZB-H2 | $(p-1)(T_F+T_B-2T_W)$ | 10.83% | 6.98% |
| ZB-1p | 同 H1（自动调度被显存逼回去） | 15.85% | 13.23% |
| ZB-2p | 跨 step 才到 0 | 4.33% | 0.29% |

口径：1.5B / 6.2B，$p=8,m=24$，A100，数字取自论文；气泡率公式里 $p$、$m$ 都在分母上，换规模不能直接外推。

前四行取 $T_F=T_B=T_W=1$，分别是 $3(p-1)$、$2(p-1)$、$1(p-1)$、$0$：**每多让 W 离开链上一步，气泡就少一个 $(p-1)T_W$**。

**别把 1/3 当普适结论**。等时下 H1 的气泡是 1F1B 的 1/3，但论文的 1.5B profile 是 $T_F=18.5$、$T_B=18.1$、$T_W=9.3$：

$$
\frac{\text{H1}}{\text{1F1B}}=\frac{18.5+18.1-9.3}{18.5+18.1+9.3}\approx 0.59
$$

论文的实测也吻合：0.65。**真实硬件上 H1 拿到的是 0.6 倍，不是 1/3**。

$T_W$ 越大越省气泡，这话有前提：只在 $T_W\lesssim(T_F+T_B)/2$ 以内成立，再大公式就不作数了。按论文的 FLOPs 估计 $T_B+T_W=2T_F$，正好落在边界附近，可以拿它做校验。

### 3.2 实测：24.31% 到 0.29%，但不是免费的

三处容易读错。**ZB-H2 不是零**：等时公式给它 0，可 $T_F/T_B/T_W$ 不等长时它只落在 H1 和 ZB-2p 中间；真正接近零的是自动搜出来的 ZB-2p。

**ZB-1p 和 ZB-H1 数字完全一样**，是论文的推测：1p 的显存上限把自动调度逼回了 H1 的形状，代价是显存 32 GB。

**ZB-2p 的代价是显存**：59 GB vs 1F1B 的 30 GB，接近 2 倍。论文摘要里 +23% 是"相似显存下"，+31% 是"放宽显存"。

吞吐也不是线性的：气泡从 24% 降到 4%，理论上限约 $1/(1-0.24)=1.32$ 倍，实测只有 +23% 左右，差额来自 overlap tax 和通信。

### 3.3 那个「0」为什么难

有一条谁都绕不过去的下界。最后一个 stage 最早只能在 $(p-1)T_F$ 之后拿到第一个 microbatch，而它自己的工作量是 $m(T_F+T_B+T_W)$，所以

$$
\text{cost}\;\ge\;(p-1)T_F+m(T_F+T_B+T_W)
$$

代入 $T_F=T_B=T_W=1$、$p=8$、$m=24$：下界 $7+72=79$。1F1B 是 $72+21=93$，split 是 $72+14=86$，ZB-H1 正好 79，而 ZB-H2 的 $72+0=72$ 低于下界。

于是两句话：等时假设下 ZB-H1 已经把单步气泡榨干了，剩下的 $(p-1)T_F$ 是填不掉的填充楔子；ZB-H2 的 0 低于下界，**它不可能是单步性质**。

它只能是跨 step 稳态：本 step 尾部右移的 W，一路接上下一 step 的 warmup。

但这有个前提——step 之间不能有全局 barrier。而常规实现里 optimizer step 恰好带两个全局同步：全局梯度范数（grad clip）和全局 NaN/Inf 检查。两者都要求所有 stage 的梯度先做完归约，于是每个 iteration 结尾被强制对齐，平行四边形又被切回梯形。

论文 §4 的办法是把同步后置：用"部分归约的全局状态 + 本地值"先乐观地更新一步，下一个 iteration 的 warmup 里再把全量归约结果传回来校验，不合法就用可逆的 AdamW 回滚（附录 C）。

**这就是很多团队复现 ZB 只拿到一半收益的原因**——调度抄了，optimizer 同步没动。

## 4. 模拟器：地板是算出来的，踩不穿也绕不过

`schedule_sim.py` 把 F/B/W 建成 DAG，按 stage 串行加跨 stage 依赖做 list scheduling。取 $p=4,m=8,F=B=W=1$：理想工作量 24，单步下界 $3+24=27$。

同一份表跑出来的四条 makespan：1F1B 33、split 30、ZB-H1 27（正好压在下界）、ZB-H2 跨 step 每步 24.5。给 H2 加上 `--barrier`（模拟 optimizer 同步）立刻退回 27——**差的这 2.5 就是同步的价钱**。

```text
$ python3 schedule_sim.py --stages 4 --microbatches 8 --schedule zbh1 --gantt
p0  FFFF...BWFBWFBWFBWFBWBWBWBW
p1  .FFF..BFBWFBWFBWFBWFBWBWBWW
p2  ..FF.BFBFBWFBWFBWFBWFBWBWWW
p3  ...FBFBFBFBWFBWFBWFBWFBWWWW
```

这张图只说明一件事：链上只剩 F 和 B（`BFBFBF...`），W 被按级往后推——级别越高，尾部堆的 W 越多，最后一级整段 W 都堆在尾部（`WWWW`）。形状和论文的在飞数公式对得上：第 $s$ 级积压 $s$ 个 wgrad，在飞数 $=(p-s)M_B+s\,M_W$，总数仍是 $p$。

加 barrier 后 H2 也没回到 0，因为这张静态表只是照论文形状手写的，没有自动调度的精确重排，也没有真正的 bypass 回滚。**Zero Bubble 的难点从来不是"拆 B/W"这个想法，而是调度搜索和同步绕过**。

## 5. 该抄哪一档

| 场景 | 建议 | 原因 |
|---|---|---|
| 显存紧、$m$ 小 | 1F1B | 峰值激活约 $p$ 份，最稳 |
| 显存够、想压气泡 | ZB-2p / DualPipe | 气泡 <1%，显存约 2× |
| 多机、IB 慢 | ZB-1p 或 1F1B-I | 减少 stage 间通信次数，ZB-1p 不增加通信 |
| 单机 NVLink、$m$ 大 | 1F1B-I | 气泡随 $v$ 缩小，通信便宜 |
| 想少改代码 | 1F1B + `overlap_p2p_comm` | Megatron 现成开关，先拿 P2P 重叠的收益 |

Megatron 里和 PP 通信重叠相关的三个开关：`overlap_p2p_comm`（P2P send/recv 与计算重叠，和 `batch_p2p_comm` 互斥）、`overlap_p2p_comm_warmup_flush`（warmup 阶段也重叠）、`virtual_pipeline_model_parallel_size`（交错式的 chunk 数）。

**但 P2P 重叠和气泡消除是两件事**：前者藏的是 stage 间 send/recv 的延迟，后者填的是 stage 等依赖的空转。很多团队开了 `overlap_p2p_comm` 只快 2%–3%，就是因为气泡才是大头。

## 结论

**只记一件事**：这一路换来的三样东西——更小的气泡、更省的显存、把 W 摘出链外——没有一样免费；而 ZB 那个零必须跨 step，复现时 optimizer 的全局同步不动，就只能拿到一半。

下一篇看 DualPipe：DeepSeek 把这条线推成双向流水，每张卡同时处理两个方向的 chunk，F 和 B 互相填充，气泡公式变成 $(PP/2-1)(F\&B+B-3W)$。

---

**系列导航**：[一：性能账本]({{< relref "comm-overlap-01-performance-model" >}}) · [三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}})

> 配套代码：`code/comm-overlap/schedule_sim.py`、`overlap_model.py`
> 自测：`python3 schedule_sim.py --test && python3 overlap_model.py --test`
