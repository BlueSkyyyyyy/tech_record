---
title: "LLM 模型结构变迁（九）：残差连接的第二次生命——从 ResNet 到 AttnRes 与 Kimi-K3"
date: 2026-11-04
draft: false
weight: 10
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "残差连接", "skip connection", "AttnRes", "Kimi-K3", "mHC", "归一化"]
categories: ["LLM算法"]
---

第五篇讲的是残差流的**加宽**（HC / mHC）：把一条流变成 $n$ 条。这一篇把镜头拉远，回答三个更基本的问题：

1. **skip connection 到底在做什么**？为什么 2015 年那个 `x + F(x)` 能统治十年，又为什么 2025 年突然不够用了？
2. **AttnRes 是什么**？它把「深度」当成序列做 softmax，公式、初始化、复杂度、系统实现分别长什么样？
3. **Kimi-K3 真的这么用了吗**？按 K3 的真实形状（93 层、$d=7168$、block=12 层），这笔账一年要付多少毫秒、多少 GB？

先给结论：

- 残差有三个角色——**梯度高速公路**、**深度方向的聚合器**、**数值尺度的锚**。十年来的所有改动，动的都是后两个，第一个从未被放弃。
- PreNorm 的根本问题是**等权累加**：$\|h_l\|$ 随深度 $O(L)$ 增长，每层的相对贡献被稀释，深层只能靠「把输出放大」来抢影响力。这是**深度方向没有注意力**的后果。
- AttnRes 用 softmax 替换等权累加，是「深度方向的 RNN → Transformer」。全量版 $O(L^2d)$ 算得起、存不起；Block 版把 186 个 source 压成 8 个 block，$N\approx 8$ 就能拿回大部分收益。
- **它几乎不花 FLOPs，花的是带宽**。K3 形状下每迭代 186 次调用、344.6 GB 残差流量、T=4096 时约 **130 ms**——相当于注意力主干的 +21%，是普通 PreNorm 残差的 **10.5 倍**流量。
- 这是 2026 年最容易被低估的一类结构改动：参数只有 **2.7M**（187 套 $q$/$w$ = 186 个 sublayer + 输出头，占 2.78T 的 0.0001%），但它是**训练 step time 的百分比级开销**，必须 kernel 化、必须和 PP 一起设计。

## 0. 一个画面：从「接力」到「选人」

经典残差是**接力赛**：每一棒（层）接过上一棒的棒子（hidden state），跑一段，把棒子往前递。棒子始终是同一根，信息在一条线上传递。

问题是：跑到第 90 棒时，棒子上已经叠了 90 个人的手印。第 90 棒想施加影响，只能**更用力地握**（输出更大的值）——这就是 PreNorm dilution。

AttnRes 把接力改成**选人**：每一棒上场前，可以回头看之前所有棒的成绩单，用 softmax 决定「这次带多少第 3 棒、多少第 17 棒、多少上一棒」。权重是学出来的，而且**每个 token 可以不一样**。

> 类比的边界：回头看成绩单是要花钱的——每次都要把之前所有棒的成绩从显存里再读一遍。所以真正的设计问题不是「能不能看」，而是「看多少份、怎么缓存、怎么跨 PP stage 传」。

## 1. skip connection 到底在做什么

### 1.1 梯度高速公路

ResNet 的更新是

$$
h_l = h_{l-1} + f_{l-1}(h_{l-1}).
$$

展开递归，$h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)$——**每一层看到的是 embedding 加所有前层输出的等权和**。反传时

$$
\frac{\partial \mathcal{L}}{\partial h_l}
= \frac{\partial \mathcal{L}}{\partial h_L}\prod_{j=l}^{L-1}\left(I + \frac{\partial f_j}{\partial h_j}\right).
$$

乘积展开后**恒等项 $I$ 永远在**：无论多深，梯度都有一条不经过任何变换的直达路径。这是 ResNet 解决退化问题（degradation）的核心，也是后来所有变体都不敢丢的东西。

### 1.2 第二个角色：深度方向的聚合器

但残差还有第二个角色，常被忽略：它决定了**跨层信息怎么聚合**。上面那个展开式说明，第 $l$ 层拿到的是

$$
h_l = \sum_{i=0}^{l-1} M_{i\to l}\, v_i,
\qquad v_0 = h_1,\; v_i = f_i(h_i),
$$

标准残差的 $M_{i\to l}\equiv 1$——**固定单位权重**。序列方向早就从「固定权重求和」（RNN）进化到「学习权重的 softmax」（attention），专家方向也从稠密进化到路由；只有深度方向，十年没动。

### 1.3 第三个角色：数值尺度的锚

PreNorm 把 norm 放在子层**里面**：

$$
h_l = h_{l-1} + f_{l-1}\big(\mathrm{Norm}(h_{l-1})\big).
$$

恒等路径干净，但 $\|h_l\|$ 随深度线性增长。PostNorm 把 norm 放在残差**之后**，尺度有界，但每次都在残差主路径上做归一化，梯度被反复压缩，深层容易 vanishing——所以 Post-LN 必须 warmup，Pre-LN 可以不用。

这两个角色的冲突，就是过去六年残差研究的全部张力。

## 2. 十年演化：一张表

| 时间 | 工作 | 更新式 | 权重 | 访问范围 |
|---|---|---|---|---|
| 2015 | ResNet | $h_l=h_{l-1}+f(h_{l-1})$ | 固定 1 | 上一层 |
| 2015 | Highway | $h_l=(1-g)\odot h_{l-1}+g\odot f(h_{l-1})$ | 输入相关（门控） | 上一层 |
| 2017 | DenseNet | $h_l=\mathrm{Conv}([h_1;f_1;\dots;f_{l-1}])$ | 固定 | 所有前层（拼接） |
| 2019 | Pre-LN 分析 | 归一化移到子层内 | 固定 1 | 上一层 |
| 2020 | ReZero | $h_l=h_{l-1}+\alpha_l f(h_{l-1})$ | 静态标量，$\alpha$ 零初始化 | 上一层 |
| 2020 | LayerScale | $h_l=h_{l-1}+\mathrm{diag}(\lambda_l)f(h_{l-1})$ | 静态逐通道（$10^{-4}$ 量级） | 上一层 |
| 2022 | DeepNorm | $h_l=\mathrm{Norm}(\alpha h_{l-1}+f(h_{l-1}))$ | 固定，$\alpha=(2M)^{1/4}$ | 上一层 |
| 2022 | DenseFormer | $h_l=\sum_i \alpha_{i\to l}f_i(h_i)$ | **静态**学出来的标量 | 所有前层 |
| 2024 | Hyper-Connections | $X_l=B_lX_{l-1}+C_lf(A_lX_{l-1})$ | 输入相关矩阵 | $m$ 条流 |
| 2025 | mHC | 同上，$B_l$ 约束到双随机 | 输入相关 + 流形约束 | $m$ 条流 |
| 2026 | **AttnRes** | $h_l=\sum_i \alpha_{i\to l}v_i$，$\alpha=\mathrm{softmax}(\phi(w_l,\mathrm{RMSNorm}(v_i)))$ | **输入相关 softmax** | **所有前层输出** |

几个关键节点值得展开：

- **ReZero / LayerScale / DeepNorm 都在调「残差分支的增益」**：$\alpha$ 零初始化、$\lambda$ 取 $10^{-4}$、$\alpha=(2M)^{1/4}$。它们解决的是「深层残差分支太大导致不稳定」，但**没有改变等权聚合**这个事实——只是把整体的尺度调小。
- **DenseFormer 是第一个认真做「跨层加权」的**：给每层学一组静态标量 $\alpha_{i\to l}$。但 AttnRes 论文的消融显示它**完全没有收益**（1.767 vs baseline 1.766）——说明关键不是「有权重」，而是「权重随输入变化」。
- **HC / mHC 走的是「加宽」路线**：不改聚合的「看谁」，而是把一条流变成 $m$ 条、让流之间用矩阵混合。mHC 再加双随机约束保证连乘不炸。细节在第五篇，这里只取一个数：$m=4$ 时每层残差 I/O 是 $34d$，而 Block AttnRes 是 $5.5d$。
- **Value Residual Learning（2024）** 只让每层回看**第一层的 value**，是「跨层访问」的极简版；MRLA 用 sigmoid 对所有权重做逐元素门控，但它的 query-key 是可分离的，本质更接近线性注意力。

一句话总结这条演化线：**大家一直在「加宽状态」和「加大增益」，直到有人问：为什么不让每一层直接看所有历史？**

## 3. PreNorm 的 $O(L)$ 问题：dilution 的数学

设每层输出 $f_i(h_i)$ 的尺度大致相当（Norm 之后输入同分布，输出也同量级），则

$$
\|h_l\|^2 = \Big\|\sum_{i<l} v_i\Big\|^2 \approx \sum_{i<l}\|v_i\|^2 + \text{cross terms} \sim O(L).
$$

于是第 $l$ 层新加的 $v_l$ 在 $h_l$ 里的**相对份额**是 $1/l$——越深越小。反向看，梯度的恒等项虽然保住了，但每层对最终输出的边际影响被摊薄。

这不是纯理论担忧，有三个可观测后果：

1. **深层可以被剪掉**：Gromov 等人发现很多模型的最后若干层剪掉后 loss 几乎不变（[2403.17887](https://arxiv.org/abs/2403.17887)）。
2. **输出幅度单调增长**：AttnRes 论文 Fig. 5(b) 里，baseline 每个 transformer block 的输出幅度随深度单调上升；AttnRes 则被限制在**周期性的有界区间**内（每个 block 边界重置一次累积）。
3. **梯度分布极不均匀**：Fig. 5(c) 里 baseline 的梯度在最早几层特别大；AttnRes 因为 softmax 让 source 之间有竞争，梯度沿深度明显更均匀。

还有一条更隐蔽的机制：**PreNorm 让尺度信息在子层入口被抹掉**（RMSNorm 对整体缩放不变）。深层想「更重要」，唯一的手段就是把输出幅度做大——而它又只能通过同一个 norm 后的输入去学。这是一个**没有归一化的竞争**：谁声音大谁被听见。

AttnRes 的解法很直接：给「被听见」这件事加上 softmax 归一化，让竞争变成**零和的**（权重和恒为 1）。

## 4. 统一视角：深度混合矩阵

把上面所有方法写成同一个形式：

$$
h_l = \sum_{i=0}^{l-1} M_{i\to l}\, v_i,
\qquad M\in\mathbb{R}^{L\times L}\ \text{下三角}.
$$

$M$ 就是「深度混合矩阵」。不同方法的区别只有两个：**$M$ 怎么来**（固定 / 静态学习 / 输入相关）、**$M$ 的秩有多高**。

| 方法 | $M_{i\to l}$ | 结构 | 状态大小 |
|---|---|---|---|
| 标准残差 | $1$ | 全 1 下三角，semiseparable rank 1 | $d$ |
| Highway | $g_{i+1}\prod_{j=i+2}^{l}(1-g_j)$ | rank 1，但权重输入相关 | $d$ |
| (m)HC | $\beta_i^\top A^{i+1\to l}\alpha_l$ | $m$-semiseparable | $md$ |
| DenseFormer | $\alpha_{i\to l}$（静态） | 稠密但**与输入无关** | $Ld$ |
| **Full AttnRes** | $\mathrm{softmax}_i\big(w_l^\top\mathrm{RMSNorm}(v_i)\big)$ | **稠密、输入相关、秩 $L$** | $Ld$ |
| **Block AttnRes** | 块内共享 block key，块内 partial 单独一项 | 秩在 $N$ 与 $N+S$ 之间 | $Nd$ |

「semiseparable rank」这个词来自 SSM 文献（Dao & Gu, [2405.21060](https://arxiv.org/abs/2405.21060)）：一个下三角矩阵如果能用 $r$ 维状态递推表示，秩就是 $r$。**标准残差的 rank 是 1**（状态就是一个 $d$ 维向量），HC 是 $m$，AttnRes 是 $L$——也就是「没有常数状态，直接看全部历史」。

这条谱系和序列方向完全同构：

- 标准残差 ↔ RNN 的 $S_t=S_{t-1}+k_tv_t^\top$（线性注意力）；
- Highway / 门控 ↔ GLA / 带遗忘门的线性注意力；
- mHC ↔ 状态扩展（state expansion）；
- AttnRes ↔ softmax attention。

论文把它写成「sequence–depth duality」，并引了 TTT（test-time training）：残差的每次加法相当于对深度做一步「梯度更新」，而 AttnRes 相当于在深度上做一次完整的注意力检索。**深度方向的「RNN → Transformer」，就是这一篇的全部内容。**

## 5. Full AttnRes：公式与设计

### 5.1 定义

对第 $l$ 个 sublayer（注意：论文把**每个 self-attention 或 MLP 都算一层**，K3 里是 186 个）：

$$
q_l = w_l \in \mathbb{R}^d,\qquad
k_i = v_i =
\begin{cases}
h_1, & i=0\\
f_i(h_i), & 1\le i\le l-1
\end{cases}
$$

$$
\alpha_{i\to l}
= \frac{\exp\!\big(w_l^\top \mathrm{RMSNorm}(v_i)\big)}
{\sum_{j=0}^{l-1}\exp\!\big(w_l^\top \mathrm{RMSNorm}(v_j)\big)},
\qquad
h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i.
$$

**记号澄清：$v_i$ 不是 $h_i$。** $v_i=f_i(h_i)$ 是第 $i$ 个 sublayer 的**输出**，$h_i$ 是它的**输入**（上一层聚合出来的结果）；只有 $i=0$ 时 $v_0=h_1$ 恰好是 embedding（它还没经过任何 sublayer）。若记累积流 $R_i=v_0+\dots+v_i$（$R_0=h_1$），则 $v_i=R_i-R_{i-1}$——AttnRes 是在**增量**上做 softmax，不是在累积状态上。两者不等价：把 $h_l=\sum_i\alpha_iv_i$ 改写成 $R_i$ 的组合会得到

$$
h_l=\sum_{i=0}^{l-2}(\alpha_i-\alpha_{i+1})\,R_i+\alpha_{l-1}R_{l-1},
$$

系数可以为负；而直接对 $R_i$ 做 softmax，$h_l$ 就变成「不同长度前缀」的加权平均了。还有一个更硬的理由：$h_i$ 本身只含到 $f_{i-1}$，拿 $h_i$ 当 value 会把最近一层 $f_{l-1}$ 的输出直接漏掉。

Block 版同理：$b_n=R_{\text{end}_n}-R_{\text{end}_{n-1}}$，是这一块的**增量**；K3 的 $V$ 就是「embedding + 每个已完成 block 的增量 + 当前 block 的 partial」。

三个设计点：

- **pseudo-query $w_l$ 是每层一个可学习向量，不依赖输入**。这样同一 block 内所有层的 query 可以**并行**算（不用等前一层输出），这是后面 two-phase 策略的前提。消融显示：改成输入相关（从 hidden state 投影）能再降 0.006（1.731 vs 1.737），但要多一个 $d\times d$ 投影、且 decode 时变成串行访存，所以默认不用。
- **key 上做 RMSNorm**：防止输出幅度大的层垄断 softmax。Block 版尤其需要——block 代表是多个层输出之和，量级差异更大。去掉它：Full 1.743、Block 1.750（都变差）。
- **$i=0$ 永远是 embedding**：保证原始 token 信息在候选集里，不会被深层「忘掉」。学出来的权重图（Fig. 8）里，embedding 那一列的权重始终非零，pre-attention 层尤其明显。

softmax 而不是 sigmoid：sigmoid 各 source 独立，没有竞争，1.741 vs 1.737。**竞争性归一化**是收益的一部分。

### 5.2 初始化：为什么必须全零

论文明确要求**所有 pseudo-query 初始化为 0**。此时所有 logits 都是 0，$\alpha$ 均匀，AttnRes 退化成「对 depth 做等权平均」。因为 $h_l$ 紧接着过一个 RMSNorm，均匀权重带来的 $1/(k+1)$ 缩放会被 norm 消掉——所以起点在功能上接近 PreNorm，训练不会因为「一开始就乱选」而震荡。

这是一个**必须照做**的细节，不是可选技巧。我们后面会看到一个真实的坑：HuggingFace 的 `_init_weights` 会把 `attn_res_proj` 当普通 Linear 初始化成 $\mathcal{N}(0,0.02)$，**不是零**。

### 5.3 代价

- 算术：$O(L^2d)$ 每 token。
- 显存：要保留所有层输出，$O(Ld)$。
- 但**在普通训练里这两项都不额外花钱**：层输出本来就要为反传保留。真正的问题是**激活重算 + 流水并行**——重算会把本可以丢掉的输出重新需要，PP 则要把每一层的输出都跨 stage 传一遍，通信量 $O(Ld)$。

$L=186$、$d=7168$、$T=4096$、bf16 时，全量 source 是 $186\times4096\times7168\times2\ \text{B}\approx 10.9$ GB/rank——存不下，也传不起。这就是 Block AttnRes 的动机。

## 6. Block AttnRes：把 186 个 source 压成 8 个

### 6.1 构造

把 $L$ 个 sublayer 分成 $N$ 个 block（每块 $S=L/N$ 个），块内输出**求和**成一个代表：

$$
b_n = \sum_{j\in \mathcal{B}_n} f_j(h_j),
\qquad b_0 = h_1\ (\text{embedding}).
$$

第 $n$ 块的第 $i$ 个 sublayer 看到的 source 集合是

$$
V=
\begin{cases}
[b_0,b_1,\dots,b_{n-1}]^\top, & i=1\\
[b_0,b_1,\dots,b_{n-1},b_n^{\,i-1}]^\top, & i\ge 2
\end{cases}
$$

其中 $b_n^{\,i-1}$ 是**当前 block 内的部分和**（partial）。注意两个容易搞错的点：

1. **$b_n$ 是块内输出之和，不是累积残差流**。所以聚合结果是这些源的**凸组合**，不是加法。块的「重置」正是把 $O(L)$ 的累积增长切成了一段段 $O(S)$。
2. **块边界那一次聚合只有已完成 block，没有额外的 partial 项**（上一个块的部分和此刻正好等于 $b_{n-1}$，随后被 append）；从块内第二个 sublayer 开始，source 里才多一个「当前 partial」。

$N=1$ 退化成标准残差（embedding 单独作为 $b_0$），$N=L$ 就是 Full AttnRes。K3 取 $N\approx 8$——论文的消融说 $N\approx8$ 能拿回大部分收益。

### 6.2 复杂度与消融

| | Full | Block |
|---|---|---|
| 算术 | $O(L^2d)$ | $O(N^2d + NSd)$ |
| 每 token 存 source | $Ld$ | $Nd$ |
| PP 通信 | $O(Ld)$ | $O(Nd)$ |

16 层模型上的 block size 消融（baseline 1.766）：

| $S$ | 1（Full） | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| loss | **1.737** | 1.746 | 1.746 | 1.746 | 1.753 | 1.757 |

$S\le 8$ 基本打平，$S=16/32$ 开始向 baseline 退化。**这说明「跨 block 的远距离访问」比「块内细粒度」更重要**——和滑动窗口的对比也印证了这一点：只保留最近 8 层（SWA）只有 1.764，几乎白给。

## 7. 数字：AttnRes 到底值多少

### 7.1 scaling law

五个规模（194M–528M 激活），同算力对比：

| 变体 | 拟合 | 5.6 PFLOP/s-days 处 loss |
|---|---|---|
| Baseline | $1.891\times C^{-0.057}$ | 1.714 |
| Block AttnRes | $1.870\times C^{-0.058}$ | **1.692** |
| Full AttnRes | $1.865\times C^{-0.057}$ | — |

Block 版相当于 baseline 多训 **1.25× 算力**；Full 与 Block 的差距随规模缩小（最大规模只差 0.001）。同一张表里 mHC(-lite) 是 1.694——**Block AttnRes 用 5.5d 的 I/O 打平了 mHC 的 34d**。

### 7.2 消融（16 层，baseline 1.766）

| 变体 | loss |
|---|---|
| DenseFormer（静态权重） | 1.767 |
| mHC | 1.747 |
| Block AttnRes（$S=4$） | 1.746 |
| Full AttnRes | 1.737 |
| + 输入相关 query | **1.731** |
| 换成 input-independent 标量 | 1.749 |
| softmax → sigmoid | 1.741 |
| 去掉 key RMSNorm（Full / Block） | 1.743 / 1.750 |
| 多头 depth attention（$H=16$） | 1.752 |
| 滑动窗口（最近 8 层 + embedding） | 1.764 |

两个反直觉结论：**多头没用**（不同 channel 组想看不同层，这件事不成立——一层有用就是整体有用）；**输入相关 query 有小幅收益但工程上不划算**。

### 7.3 下游（Kimi Linear 48B/3B，1.4T tokens）

| 任务 | Baseline | AttnRes |
|---|---|---|
| MMLU | 73.5 | **74.6** |
| GPQA-Diamond | 36.9 | **44.4** |
| BBH | 76.3 | **78.0** |
| HumanEval | 59.1 | **62.2** |
| Math | 53.5 | **57.1** |
| GSM8K | 81.7 | **82.4** |
| C-Eval | 79.6 | **82.5** |

增益集中在**多步推理和代码**（GPQA +7.5、Math +3.6、HumanEval +3.1），知识型任务涨幅小（MMLU +1.1）。这和「深层能选择性地回取早期表示」的假设一致。

### 7.4 架构偏好：更深

固定算力（$6.5\times10^{19}$ FLOPs）和激活参数（$2.3\times10^8$）扫 25 个配置（$d_{model}/L_b$ × $H/L_b$）：baseline 最优在 $d_{model}/L_b\approx60$，AttnRes 移到 $\approx45$——**同样的算力，它更愿意变深变窄**。25 个格子全部更低（0.019–0.063）。

这解释了为什么 K3 愿意堆到 93 层：AttnRes 让深度的边际收益变高了。

## 8. 系统：AttnRes 真正的成本在通信

### 8.1 PP 的跨 stage 缓存

标准残差在 PP 里每个 stage 边界只传一个 hidden state，大小与深度无关。Block AttnRes 要传「到目前为止所有 block 代表」。朴素做法：第 $j$ 个 chunk 传 $jN_p$ 个 block，总通信

$$
\mathrm{Comm}_{\text{naive}}=\frac{C(C-1)}{2}N_p d,\qquad C=PV.
$$

跨 stage 缓存（每个物理 rank 把之前 virtual stage 收到的 block 留在本地）后：

$$
\mathrm{Comm}_{\text{cached}}
=\underbrace{\frac{P(P-1)}{2}N_p d}_{\text{第一个 virtual stage}}
+\underbrace{(V-1)P^2N_p d}_{\text{后续}}.
$$

峰值从 $O(C)$ 降到 $O(P)$，好 $V$ 倍，能完全 overlap 进 1F1B 的稳态。**没有这个优化，AttnRes 在 PP 下不可用**。

### 8.2 two-phase：把 query 批量算

因为 $w_l$ 与输入无关，一个 block 内 $S$ 个 sublayer 的 inter-block attention 可以**一次矩阵乘**算完：

- **Phase 1**：把 $S$ 个 query 堆成 $[S,d]$，对缓存的 $N$ 个 block 代表做一次 attention，返回输出 + softmax 统计量（max、log-sum-exp）；
- **Phase 2**：块内按顺序算 partial 的贡献，再用 online softmax 把两段结果合并。

每层 I/O 从 $S$ 次读降到 1 次读。典型 $L=128$、$N=8$、$S=16$ 时：

| 方案 | 每 token 每层 I/O |
|---|---|
| 标准残差 | $3d$ |
| mHC（$m=4$） | $34d$ |
| Full AttnRes（two-phase） | $24d$ |
| **Block AttnRes（two-phase）** | **$5.5d$** |

### 8.3 训练 / 推理开销

- 训练：无 PP 时可忽略；有 PP 时端到端 **< 4%**。
- 推理：典型负载 **< 2%**。
- 长上下文 prefill 的 block 缓存：128K × 8 blocks 要 15 GB；按 TP 切序列 → 1.9 GB；再配 16K chunked prefill → **< 0.3 GB**。

注意「<2% 推理开销」和「训练 130 ms/迭代」不矛盾：推理用 two-phase 把同一 block 内 24 个 sublayer 的 inter-block 读**摊薄成一次**；训练则是 186 次独立的 fwd/bwd kernel，每一次都要把 source 重新读一遍。**同一份数学，两种访存模式，成本差一个数量级。**

## 9. Kimi-K3：把 AttnRes 放进 2.78T 的模型

### 9.1 配置

K3 的结构（`config.json` + 技术报告）：

| 项 | 值 |
|---|---|
| 层数 | 93（69 KDA + 24 MLA，3:1 混合，最后一层是 MLA） |
| hidden | 7168 |
| attention heads | 96 |
| block size | `attn_res_block_size = 12`（层）= **24 个 sublayer** |
| block 数 | $\lceil 186/24\rceil = 8$（最后一块不满） |
| 最大 source 数 | 9（embedding + 7 个已完成 block + 当前 partial） |
| 每迭代调用 | **186 次**（185 次在层内 + 1 次输出头） |
| pseudo-query / key-norm | 每个 sublayer 各一个 $d$ 维向量，fp32 |

186 次的来源：93 层 × 2 个 sublayer = 186，减去第 0 层 attention 前的那一次（那时还没有任何 block，且单 source 的聚合在实现里被短路成恒等），再加上**最终输出头**（`_apply_output_attn_res` / `final_attn_res`，$L=9$）。

L 的分布：$L=2..8$ 各 24 次，$L=9$ 共 18 次。

### 9.2 代码长什么样

HF 的 `modeling_kimi_linear.py` 里，每个 sublayer 前调用一次 `_apply_attn_res`：

```python
v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
v_float = v.float()
variance = v_float.pow(2).mean(-1, keepdim=True)
k = v_float * torch.rsqrt(variance + norm.variance_epsilon)
score_weight = norm.weight.float() * proj.weight.squeeze(0).float()
scores = (k * score_weight).sum(-1)
probs = scores.softmax(-1)
hidden_states = torch.matmul(probs, v_float)
```

三个细节和公式严格对应：

- `score_weight = norm.weight * proj.weight`：**只有 $w\odot q$ 进入 logits**。这带来一个数学上的冗余——任意 $\delta w=w\odot\epsilon,\ \delta q=-q\odot\epsilon$ 都不改变输出；两个参数在功能上只有 $d$ 个自由度。
- `prefix_sum` 在 block 边界被置 `None`，之后只累加本 block 的输出；到下一个边界才 append 成 $b_n$。**所以 $b_n$ 是块内和**。
- 全程 fp32 计算、cast 回 bf16；`epsilon = 1e-5`。

Megatron 侧（`attention_residual.py`）多了几件事：每个 sublayer 独立的 `pseudo_query`（零初始化）和 `key_norm_weight`（全 1）、单 source 的恒等短路（同时保留零梯度以适配 DDP 的 grad-ready 统计）、输出头 `final_attn_res`、PP payload 的切分（`[num_slices*s, b, h]`）、以及可选的 norm fusion（把下一层的 RMSNorm 融进 AttnRes kernel）和 recompute（只丢聚合输出、保住 source）。`kimi_k3` 的 recipe 里写的是 `attn_res_block_layers: 24`——**按 sublayer 计**，对应 12 层，和 HF 的 `block_size=12` 一致。

### 9.3 kernel：FLA 的 `fused_attnres`

Triton 实现（`fla/ops/attnres/fused.py`）把整次聚合压成一个 kernel：

- **前向单 pass**：源指针表（`_build_ptr_table`）一次建好，逐 source 做「RMSNorm → 点积 → online softmax → 加权累加」；$L$ 不 specialize（`do_not_specialize=['L']`），避免每种 $L$ 都编译一份；$L>8$ 时指针表按 `next_pow2(L)` 补齐。
- **反向两个 kernel、无 atomic**：一个算 $dV$（每个 source 一份，fp32），一个沿 token 归约 $dq/dw$；`dq/dw` 需要 $[N,d]$ 的 fp32 scratch（正是前面 I/O 表里那部分额外流量）。
- **可选融合输出 RMSNorm**：`output_rms_weight` 传进来，mixed 残差不再落盘。
- **Gluon 后端**：静态 $L$ 编译、`cp.async`、smem 常驻/环形缓冲、32-token 分组把 `dqw` 缩小 32×。

### 9.4 实测（H100，bf16，$T=4096$，CUPTI）

单次调用（µs）：

| $L$ | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|
| fwd | 74 | 118 | 119 | 158 | 160 | 199 | 200 | 237 |
| bwd | 318 | 417 | 438 | 537 | 563 | 656 | 692 | 770 |
| 次数 | 24 | 24 | 24 | 24 | 24 | 24 | 24 | 18 |

（单次值来自 sweep 那一轮；下面的每迭代合计来自独立的 schedule 那一轮，186 次逐项相乘再求和。两轮单次值相差 ±2 µs，属正常噪声；直接用上表乘次数得到 129.7 ms，与 129.8 只差 0.1 ms。）

按 K3 schedule 聚合（186 次串行）：

| $T$ | fwd | bwd | 合计 | 流量 | roofline | 效率 |
|---|---|---|---|---|---|---|
| 1024 | 7.2 ms | 26.4 ms | **33.6 ms** | 86.2 GB | 25.7 ms | 76.5% |
| 4096 | 28.8 ms | 101.0 ms | **129.8 ms** | 344.6 GB | 102.9 ms | 79.4% |
| 8192 | 55.4 ms | 199.3 ms | **254.7 ms** | 689.2 GB | 205.7 ms | 80.9% |
| 16384 | 108.1 ms | 394.0 ms | **502.1 ms** | 1378.4 GB | 411.5 ms | 82.0% |

对照：

- 本目录先前实测的 KDA（527 ms）+ MLA（97 ms）= 625 ms，AttnRes 相当于再 **+20.8%**。
- 普通 PreNorm 残差 186 次约 32.8 GB，AttnRes 是 **10.5×** 流量——**它买的是表达力，付的是带宽**。
- 朴素 PyTorch 实现（`naive_attnres`）：fwd 3.32 ms、bwd 7.51 ms、显存 4.09 GB，比融合版慢 **9.0× / 7.7×**、显存 **5.2×**。
- Gluon：$L\le8$ 更快（$L=2$ 的 fwd/bwd 63/179 µs vs Triton 74/318），但 $L=9$ 的反向掉到 2844 µs（Triton 770），**在 K3 上必须为 $L=9$ 保留 Triton 路径**。
- norm fusion：$L=9$ 时 237 µs vs 282 µs（bf16 权重）/ 711 µs（fp32 权重）——Megatron 把 norm 权重留 fp32，torch 的 `rms_norm` 会退化成非融合路径，融合收益从 45 µs 涨到 474 µs。

### 9.5 三个真实的坑

**坑一：输出头容易漏。** 我们第一版 schedule 只数了层内的 185 次，漏掉最终输出头，每迭代少算 1.0 ms（0.8%）。它不在任何 layer 的 forward 里，而在 `KimiLinearModel.forward` 的最后一行。

**坑二：HF 的默认初始化不是零。** `post_init` 会把 `attn_res_proj` 当普通 `nn.Linear` 初始化成 $\mathcal{N}(0,0.02)$。论文要求 pseudo-query **全零**（保证初始均匀权重），Megatron 是显式 zeros；用 HF 代码从零训练必须手动置零，否则丢掉的正是「起点等价于 PreNorm」这个稳定性保证。

**坑三：block 边界和 PP stage 边界不对齐。** 一个 block 可以跨两个 stage，payload 的切片数要用 `attn_res_num_payload_slices` 按「已完成层数」算，不能按 stage 数算；开了 VPP 还要按「本 rank 上次访问以来新增的 source」做增量传输。

## 10. AttnRes vs mHC：约束还是学习

| | mHC | Block AttnRes |
|---|---|---|
| 混合矩阵 | $m\times m$，约束到双随机（Birkhoff） | softmax 权重，无约束（凸组合） |
| 关键性质 | 谱范数 $\le1$，连乘不放大 | 权重和为 1，尺度有界 |
| 每层 I/O | $34d$（$m=4$） | $5.5d$ |
| 16 层消融 | 1.747 | 1.746（Full 1.737） |
| 额外参数 | 每层 $O(m^2)$ | 每层一个 $d$ 向量 |
| 主要风险 | Sinkhorn 只是近似、实现分裂 | 访存/PP 通信、kernel 复杂度 |

两者不是互斥的：AttnRes 只改「聚合」，不碰归一化和门控，理论上可以叠在 mHC 的流上。论文也把 mHC 当作对照，结论是 **Full AttnRes 更好、Block 版打平但便宜 6 倍 I/O**。

选择上：如果只是想要「深层更稳」，mHC 的约束更便宜（$O(m^2)$ 参数、不需要保存历史）；如果想要「深层能选择性回取」，AttnRes 更直接，但要接受带宽账单。**K3 选了后者，因为它的瓶颈本来就在带宽——与其省一点流量，不如把深度用起来。**

## 11. 账本与判断

| 方案 | 参数增量 | FLOPs/token | 每迭代残差流量（K3, T=4096） |
|---|---|---|---|
| PreNorm 残差 | 0 | 0 | 32.8 GB |
| Block AttnRes（全量 schedule） | **+2.7M**（187×2×$d$，占 2.78T 的 0.0001%） | ~0.03 GFLOP | **344.6 GB** |
| AttnRes 的注意力部分 | — | 4·d·ΣL ≈ 28.6 MFLOP | — |

FLOPs 只占激活参数（104B ≈ 208 GFLOP/token）的 **0.014%**——可以忽略；但流量是 10.5 倍，T=4096 时每迭代 130 ms。
（表里 PreNorm 的 32.8 GB 是把它当独立 kernel 算的**下界**；实际训练里 residual add 通常融进前一个 GEMM 的 epilogue，接近 0。AttnRes 的 344.6 GB 则是实打实的新增流量。）这就是它的定价：**用带宽买深度的表达力**。

三个判断：

1. **残差会继续被设计，而且方向是「输入相关的深度注意力」**。DenseFormer 的失败（静态权重无效）和 AttnRes 的成功说明：权重必须随 token 变。
2. **Block 大小会是下一个调参点**。$N\approx8$ 是当前硬件下的折中；论文自己说「等内存/带宽变宽，应该往 Full 走」。K3 用 12 层/块（24 sublayer）而不是更细，就是工程妥协。
3. **它会和 PP/重算/低精度深度绑定**。AttnRes 的数学成本只是 2.7M 参数（187 套 $q$/$w$，0.0001%），工程成本却是「跨 stage 缓存 + two-phase + 融合 norm」这一整套；没有这套，它就是一个慢 10 倍的算子。

最后回到开头那个问题：**skip connection 是不是「恒等就够」？** 十年的答案是：恒等解决了梯度，但没有解决聚合；而当模型深到 90 层以上，聚合本身就是主要矛盾。

---

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
5. [MoE 十年]({{< relref "llm-arch-04-moe" >}})
6. [残差流的加宽：HC 与 mHC]({{< relref "llm-arch-05-residual" >}})
7. [位置的退场]({{< relref "llm-arch-06-position" >}})
8. [条件记忆]({{< relref "llm-arch-07-memory" >}})
9. [五个收敛、三个分叉与三个赌注]({{< relref "llm-arch-08-trends" >}})
10. 本篇：残差连接与 AttnRes

配套实测：[code/te-perf/results_attnres_kimi_k3.md](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/te-perf/results_attnres_kimi_k3.md)
