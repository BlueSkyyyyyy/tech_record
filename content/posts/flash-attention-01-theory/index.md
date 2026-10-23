---
title: "Flash Attention 精读（一）：从 softmax 的内存瓶颈到 online softmax 的数学推导"
date: 2026-08-28
draft: false
weight: 1
series: ["flash-attention"]
tags: ["flash-attention", "attention", "gpu", "softmax", "系列"]
categories: ["算子开发"]
---

本系列共 7 篇，从数学原理一路读到工业级实现的每一行代码：

1. **（本篇）原理与数学推导** —— 为什么 attention 慢、safe softmax、online softmax 的流式推导、分块算法与 IO 复杂度
2. [Triton 教程版前向逐行精读]({{< relref "flash-attention-02-triton-fwd" >}}) —— 以 200 行 Triton 代码对照本篇数学
3. [FlashAttention-2/3 CUDA 前向实现]({{< relref "flash-attention-03-cuda-fwd" >}}) —— 工业级 kernel 的调度、流水线与 warp specialization
4. [反向梯度推导与 recompute 实现]({{< relref "flash-attention-04-bwd" >}}) —— dQ/dK/dV 的完整推导与三种实现对照
5. [TileLang / Gluon / Liger 多实现对比]({{< relref "flash-attention-05-dsl-zoo" >}}) —— 不同抽象层级下的同一算法
6. [官方仓库源码全景]({{< relref "flash-attention-06-official-code" >}}) —— FA2 / FA3 / FA4 三套实现的目录结构、调用链与构建方式
7. [反向传播深潜]({{< relref "flash-attention-07-bwd-deep" >}}) —— 四个抽象层级的反向实现、MLA 反向与完整的反向数学

## 先记住一个画面

把 softmax 想成**给一列数称重，再按重量比例分东西**。

麻烦在于：想给一个数称重，你得先把整列数都摆到秤上、找到最重的那个当"皮重"、再求出总重当"分母"。数据少时这没什么；数据多到台面摆不下时，麻烦就来了——你必须反复把货搬上搬下。

Flash Attention 的全部智慧，就是**让这台秤可以一边来货、一边称重、一边记账**，永远不需要把所有货同时堆在台面上。它不改变称出来的结果，只改变"货怎么上台面"。

> **类比的边界**：称重是串行的，而 GPU 上的 attention 是"很多行同时称、每行各自沿 key 维度累加"。真正被流式化的是**每一行内部的归约**（max、sum、输出累加），行与行之间仍然完全并行。记住这一点，后面看 grid/循环的分工才不会混淆。

读完这一篇，你应该能回答三个问题：

1. 朴素 attention 到底把时间花在哪？为什么说它"不是算不动，而是搬不动"？
2. online softmax 凭什么只扫一遍就能得到与三遍扫描**逐位等价**的结果？那个修正因子 $\alpha$ 是什么？
3. 分块之后 HBM 访问量为什么能从 $\Theta(N^2)$ 降到 $\Theta(N^2d^2/M)$？这个式子里的每个字母代表什么？

本篇不涉及任何具体代码。目标只有一个：**把 Flash Attention 的数学地基打牢**。读完你应该能独立写出 online softmax 的递推式，并理解它为什么是精确计算，而不是某种近似。

---

## 1. Attention 的两个代价

标准自注意力（self-attention），输入 $Q, K, V \in \mathbb{R}^{N \times d}$：

$$
S = \frac{1}{\sqrt{d}} QK^\top \in \mathbb{R}^{N \times N}, \qquad
P = \mathrm{softmax}(S) \in \mathbb{R}^{N \times N}, \qquad
O = PV \in \mathbb{R}^{N \times d}
$$

先看**计算量**：两次 $N \times N \times d$ 的矩阵乘（$QK^\top$ 与 $PV$），共约 $4N^2d$ FLOPs。这和一个 $d \times N \times N$ 的 GEMM 同阶——对 GPU 的 tensor core 来说，这是它最擅长的形状，A100 上每秒能做几百 TFLOPs。

所以**计算量不是问题**。问题在**访存**。朴素的 PyTorch 实现是这么写的：

```python
S = (Q @ K.T) / math.sqrt(d)   # 物化 N×N 矩阵，写回 HBM
P = S.softmax(-1)              # 读 N×N，写 N×N
O = P @ V                      # 读 N×N
```

取 $N = 8192$、fp16 存储，一个 $N \times N$ 矩阵就是 **128 MB**——注意这还只是**一个 head、一个 batch**。而上面三步之间，$S$ 和 $P$ 的每一次读写都要经过 HBM（高带宽显存，A100 上约 2 TB/s）。GPU 的片上 SRAM（共享内存加寄存器）只有几百 KB，比 HBM 快一个数量级，却装不下这张表。

打个比方：这就像为了统计"每层楼有几户人家"，却先把整栋楼的门牌号抄成一张 8192×8192 的巨型表格，填一遍、读一遍、再划掉。**表格本身不产生任何价值**，它只是为了让"逐行求和"这件事能做下去。$S$ 和 $P$ 就是这张表。

把账算得更具体一点：

- **算**：$4N^2d \approx 4 \times 8192^2 \times 128 \approx 3.4 \times 10^{10}$ FLOPs。在 300 TFLOPS 的 tensor core 上，理论只要约 **0.1 ms**。
- **搬**：$S$ 和 $P$ 各写一次、各读至少一次，量级是 $4N^2 \times 2\text{B} \approx 5 \times 10^8$ 字节。单看数字不大，但在 2 TB/s 的 HBM 上要花约 **0.25 ms**——已经是计算时间的两倍多，而且这些搬运**不参与任何有效计算**。

这就是 Flash Attention 论文（Dao et al., 2022）的核心观察：

> Attention 朴素实现的瓶颈不是算力，而是 HBM 带宽被中间矩阵 $S$、$P$ 的往返浪费了。如果我们能**根本不把 $S$、$P$ 写出片上内存**，attention 就能从"内存受限"回到"算力受限"。

这里要诚实说明两件事，避免读者产生不切实际的期待：

1. **计算量没有变**。Flash Attention 仍然是 $O(N^2d)$ 的浮点运算，它省的是 $O(N^2)$ 的**显存占用**和大部分 $O(N^2)$ 的**访存**。
2. **省显存本身就是巨大收益**。训练时长序列的显存从 $O(N^2)$ 降到 $O(N)$——不存 attention matrix，意味着同样的卡能训练长得多的序列。这一点常被"省带宽"的叙事盖过，但对工程落地同样关键。

---

## 2. Safe Softmax：三遍扫描与数值稳定

要消灭中间矩阵，先得理解 softmax 为什么"必须看全一行"。对 $S$ 的第 $i$ 行：

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{N} e^{S_{ik}}}
$$

这里藏着两个麻烦。

**麻烦一：数值上溢。** $e^{x}$ 在 $x \gtrsim 88$（fp32）时溢出为 inf。而 $S_{ij}$ 是两个 $d$ 维向量的内积再乘 $1/\sqrt d$，量级完全可能到几十上百。一旦某个 $e^{S_{ij}}$ 变成 inf，整行归一化就废了。

**麻烦二：分母是全行求和。** 要归一化第 $i$ 行，必须先知道整行的 $e^{S_{ik}}$ 之和。这看起来要求"一次性看到所有 key"——正是我们想避免的事。

第一个麻烦有标准解法：**safe softmax**。它利用 softmax 对加法常数的不变性。对任意常数 $c$：

$$
\frac{e^{S_{ij} - c}}{\sum_k e^{S_{ik} - c}} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}
$$

证明只需分子分母同乘 $e^c$。**这个"同乘 $e^c$"就是全部诀窍**：softmax 关心的是"谁占多少比例"，而不是"绝对值多大"，所以给整行数统一减去一个常数，比例分毫不变。好比量身高前让所有人站上同一块 10 cm 高的垫子，谁高谁矮的结论不变，但数字都变"友善"了。

取 $c = m_i = \max_k S_{ik}$，则所有指数参数 $\le 0$，指数值落在 $(0, 1]$：不会上溢；极小的值下溢成 0 也无害，因为它本来就该贡献 0。这就是教科书式的三遍扫描（three-pass）softmax：

$$
\text{Pass 1: } m_i = \max_k S_{ik} \qquad
\text{Pass 2: } \ell_i = \sum_k e^{S_{ik} - m_i} \qquad
\text{Pass 3: } P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}
$$

三遍扫描要求 $S$ 的第 $i$ 行被完整地读三次（或者物化在显存里）。如果数据能装进片上内存，三遍也无所谓；问题是装不下。**所以真正的敌人不是"减 max"这个动作，而是"为了减 max 必须回头再看一遍"。**

> **技术直觉**：safe softmax 的"减 max"不是精度上的锦上添花，而是 fp16/fp32 下指数运算的生存条件。后面所有实现细节——哨兵值用 $-10^6$ 而不是 $-\infty$、scale 乘在内积之后减 max 之前——都服务于同一目标：保证送进 exp 的参数永远是一个 $\le 0$ 的有界数。

---

## 3. Online Softmax：把三遍压成一遍

Online softmax（Milakov & Gimelshein, 2018）解决第二个麻烦。关键洞察是：**max 和 sum 都可以增量维护**。

### 3.1 先看两个块

假设 key 序列被分成两块：先看到前 $n_1$ 个 key 的 $S^{(1)}$，再看到后 $n_2$ 个 key 的 $S^{(2)}$。先定义部分量：

$$
m^{(1)} = \max_{k \le n_1} S_{ik}, \qquad \ell^{(1)} = \sum_{k \le n_1} e^{S_{ik} - m^{(1)}}
$$

看到第二块后，正确的全局量是：

$$
m = \max(m^{(1)}, m^{(2)}), \qquad
\ell = \sum_{k \le N} e^{S_{ik} - m}
$$

问题来了：$\ell^{(1)}$ 是按旧 max $m^{(1)}$ 归一化的，怎么修正到新 max $m$？一行代数就够：

$$
\ell^{(1)}_{\text{corrected}} = \sum_{k \le n_1} e^{S_{ik} - m}
= \sum_{k \le n_1} e^{S_{ik} - m^{(1)}} \cdot e^{m^{(1)} - m}
= \ell^{(1)} \cdot e^{m^{(1)} - m}
$$

**旧的部分和乘上一个修正因子，就迁移到了新的坐标系。** 这个因子记作 $\alpha = e^{m^{\text{old}} - m^{\text{new}}}$，它只跟新旧 max 有关，与具体有多少个 key 无关。

于是得到流式递推。每来一个新的 key 块 $j$：

$$
\begin{aligned}
m^{\text{new}} &= \max(m^{\text{old}}, \max_j S_{ij}) \\
\alpha &= e^{m^{\text{old}} - m^{\text{new}}} \\
\ell^{\text{new}} &= \alpha \cdot \ell^{\text{old}} + \sum_j e^{S_{ij} - m^{\text{new}}} \\
P_{ij} &= e^{S_{ij} - m^{\text{new}}} \quad (\text{未归一化，最终再除 } \ell)
\end{aligned}
$$

用归纳法容易验证：处理完所有块后，$m = \max_k S_{ik}$，$\ell = \sum_k e^{S_{ik} - m}$，与三遍扫描结果完全一致。**这是精确算法，没有任何近似**——"approximate attention"（Linformer、Performer 等）是另一条技术路线，与 Flash Attention 无关。

> **小例子（两块走一遍）**。设某行 logits 为 $[1,2,3,4]$，分成 $[1,2]$、$[3,4]$ 两块：
>
> - 第一块：$m^{(1)}=2$，$\ell^{(1)}=e^{-1}+e^{0}=1.3679$；
> - 第二块到来：$m=4$，修正因子 $\alpha=e^{2-4}=0.1353$，于是 $\ell = \alpha\ell^{(1)} + (e^{3-4}+e^{4-4}) = 0.1851+1.3679=1.5530$；
> - 一次性算法验证：$m=4$，$\ell=\sum_k e^{S_k-4}=e^{-3}+e^{-2}+e^{-1}+e^{0}=1.5530$。**分毫不差**。
>
> 关键动作只有一步：新块来了，把**旧的和乘一个数**（$\alpha$），而不是回头重读第一块。这就是"流式"的全部含义。

### 3.2 输出也要跟着搬家

到这里，softmax 的归一化已经不需要"一次看全一行"了。但还剩下输出 $O = PV$：$P$ 的第 $i$ 行也要对全部 key 求和，而不同块贡献的 $P$ 是在不同 max 下算的。**同一个 $\alpha$ 因子可以再修一次**：

$$
\begin{aligned}
O_i &= \frac{1}{\ell} \sum_k e^{S_{ik} - m} V_k \\
&= \frac{1}{\ell} \left( e^{m^{\text{old}} - m} \cdot \underbrace{\sum_{k \in \text{旧块}} e^{S_{ik} - m^{\text{old}}} V_k}_{\text{running } \tilde{O}} + \sum_{k \in \text{新块}} e^{S_{ik} - m} V_k \right)
\end{aligned}
$$

也就是说，维护一个**未归一化的输出累加器** $\tilde{O}$，每来一个新块：

$$
\tilde{O} \leftarrow \alpha \cdot \tilde{O} + P_{\text{新块}} V_{\text{新块}}, \qquad \alpha = e^{m^{\text{old}} - m^{\text{new}}}
$$

注意这里有一个很容易忽略的设计：**循环内始终不除以 $\ell$**。原因有两个：

- 中间块的 $\ell$ 还不完整，除了也是错的；
- 除法（或乘倒数）在 GPU 上比乘法贵。每块除一次是 $O(N/B_N)$ 次，循环外只除一次。

最后一次性算 $O = \tilde{O}/\ell$，数学上完全等价。

> **技术直觉**：online softmax 的本质，是把"归一化坐标系"从静态（全局 max 一次定死）变成动态（每块更新一次），再用 $\alpha$ 把旧坐标系下的累计量搬到新坐标系。所有状态——$m_i$（行最大）、$\ell_i$（行和）、$\tilde{O}_i$（未归一化输出）——大小都是 $O(N)$ 或 $O(Nd)$，与 key 数量无关，可以永久驻留在寄存器里。

---

## 4. 分块算法：两级 tiling

把 online softmax 装进 GPU，就得到 Flash Attention 的骨架。形象地说，这是一场"**分田到户**"：把 Q 的每个行块分给一个 CTA（一个"工人"），工人拿着一小块 Q，**沿着整条 KV 流水线**逐块处理，边看边更新自己手头的一本小账（$m_i, \ell_i, \tilde{O}$）——全程不必把中间结果搬回仓库（HBM）。

```
并行维度（grid）：Q 的行块 —— 每个 CTA / program 拥有 [BLOCK_M, d] 的 Q
串行维度（循环）：K/V 的列块 —— 依次加载 [BLOCK_N, d] 的 K 和 V 到 SRAM

for 每个 Q 块 (并行):
    m_i = -inf, l_i = 0, acc = 0            # 寄存器状态
    load Q_block → SRAM（或直接驻留寄存器）
    for 每个 KV 块 (串行):
        load K_block, V_block → SRAM
        S_block = Q_block @ K_block^T / √d  # tensor core，结果在寄存器
        m_new = max(m_i, rowmax(S_block))
        α = exp(m_i - m_new)
        P_block = exp(S_block - m_new)       # 未归一化
        l_i = α·l_i + rowsum(P_block)
        acc = α·acc + P_block @ V_block      # tensor core
        m_i = m_new
    O = acc / l_i                            # 循环外一次性归一化
```

对照第 3 节的递推式，逐行都是数学的直接翻译。下面三个设计决策值得单独强调，因为它们解释了"为什么是这个形状"。

### 4.1 归一化推迟到循环外

循环内 $P$ 一直是"未归一化分子"，除法只在最后做一次。除了上面说的"中间 $\ell$ 不完整"和"除法更贵"，还有一个数值上的好处：**避免在每一步引入一次额外的舍入**。所有 rescale 都集中在寄存器里完成，误差不会随着块数累积放大。

### 4.2 因果掩码的分块处理

causal mask 下，第 $i$ 个 query 只能看到前 $i+1$ 个 key。一个 $B_M \times B_N$ 的 tile 与对角线的位置关系只有三种：

| 位置 | 处理方式 | 代价 |
|---|---|---|
| 完全在对角线左下（全可见） | 不做 mask，直接算 | 正常 |
| 完全在对角线右上（全不可见） | **整块跳过，一次 dot 都不算** | 零 |
| 跨越对角线 | 块内逐元素 mask | 一次 mask 开销 |

因果模型的下三角占一半面积，块级跳过直接省掉约一半计算——这是 Flash Attention 实现 causal attention 比 full attention 快接近 2 倍的原因。这个"三分类"思想会在后面每一篇实现里反复出现：Triton 版把它做成 off-band/on-band 两遍循环，CUDA 版把它做成 `n_masking_steps` 两段式。

### 4.3 循环方向：FA1 与 FA2 的关键差别

上面这个"外层 Q、内层 KV"的结构是 FlashAttention-**2** 的写法。FA1 恰好相反：外层 KV、内层 Q。

方向的差别不是风格问题，而是**数据流问题**：

- FA1 里，输出 $O$ 被多个 Q 块共享写入。每次 softmax 状态更新，都要把**部分输出从 HBM 读回、rescale、再写出**，中间读写又回来了。
- FA2 把循环换过来之后，每个 CTA 的输出 tile 有了唯一属主，rescale 全部发生在寄存器里，HBM 上的中间读写归零；Q 块也天然适配 tensor core 的行主布局。

这个"只是换了个循环顺序"的改动，在论文里带来了 1.7~1.8 倍的加速。**循环结构决定数据流向，数据流向决定性能**——这是本系列反复出现的主题，第 4 篇讲反向时还会看到它的另一面。

---

## 5. IO 复杂度：为什么快

论文里的定理值得自己推一遍。把 HBM 想成**大仓库**，把 SRAM 想成**手边的工作台**：朴素实现每做一步都要回仓库取一次料、放一次半成品；分块实现则是"搬一次料够做一个 Q 块，做完再回仓库"。谁跑仓库的次数少，谁就快。

设 SRAM 大小为 $M$ 字节，且 $d \le M/4$（一个 tile 的 K 或 V 装得下）。下面分别数一数两种实现要访问 HBM 多少次。

**朴素实现**：读写 $S$、$P$ 各 $\Theta(N^2)$ 次，加上读写 $Q,K,V,O$ 的 $\Theta(Nd)$，总计

$$
\Theta(N^2 + Nd)
$$

**分块实现**：每个 Q 块（大小 $B_M \times d$）要扫全部 K/V。Q 块共有 $\lceil N/B_M \rceil$ 个，每个都要把 $N \times d$ 的 K 和 V 各读一遍，于是总量约为

$$
\frac{N}{B_M} \cdot 2Nd = \Theta\!\left(\frac{N^2 d}{B_M}\right)
$$

取 $B_M \approx M/(4d)$（用 SRAM 的四分之一装一个 Q 块，剩下装 K/V tile），代入得

$$
\text{HBM 访问：朴素 } \Theta(N^2) \;\longrightarrow\; \text{Flash } \Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

这个式子值得逐项读一遍：

- $N^2$ 还在——因为每个 Q 块终究要"看"所有 key，这是算法本身决定的，躲不掉；
- 但前面多了一个 $d^2/M$ 的系数。当 $d = 128$、$M = 256\text{KB}$ 时，$d^2/M \ll 1$，Flash 的 HBM 流量比朴素实现低一个数量级以上；
- $M$ 在分母上：**SRAM 越大收益越大**。这也解释了为什么后续每一代架构加大 SRAM，attention kernel 就跟着变快。

L2 cache 会让实际数字更好看：多个 CTA 同时跑相似的 KV 访问序列时，K/V 的重复读大部分命中 L2，实际 HBM 流量接近 $\Theta(Nd)$ 级。工程上还会用 grouped scheduling（让相邻 CTA 处理相邻 Q 块，使它们读 KV 的进度接近）刻意提高 L2 命中率——第 3 篇讲 CUDA 实现时会看到 FA2/FA3 具体怎么做。

---

## 6. 给反向留的钩子：Log-Sum-Exp

前向省掉了 $S$、$P$ 的物化，但反向传播需要它们：$\partial L / \partial S$ 依赖 $P$。Flash Attention 的答案是**重算（recompute）+ 前向存一个 $O(N)$ 的统计量**。

这个统计量就是每行的 log-sum-exp：

$$
\mathrm{LSE}_i = \log \sum_k e^{S_{ik}} = m_i + \log \ell_i
$$

它恰好是前向 online softmax 两个状态量的免费组合（$m_i$ 和 $\ell_i$ 反正都在寄存器里）。有了它，反向时一行就能恢复精确的 softmax 概率：

$$
P_{ij} = e^{S_{ij} - \mathrm{LSE}_i}
$$

验证一下：$e^{S_{ij} - (m_i + \log \ell_i)} = e^{S_{ij} - m_i}/\ell_i$，正是 softmax 的定义。于是反向只需重算 $S_{ij} = q_i \cdot k_j / \sqrt d$，减去 LSE 直接指数化，分子分母一次到位——不需要重跑 online softmax，也不需要单独存 $\ell_i$ 数组。

这就是前向 kernel 的 epilogue 都要顺手写出一行 LSE 的原因。第 2 篇你会看到 Triton 教程里 `m_i + tl.math.log2(l_i)` 那行，以及实现里把 $1/\ln 2$ 折进 scale、让 LSE 以 2 为底存储的工程化处理。

> **技术直觉**：LSE 是 softmax 分布的"充分统计量"——它以 $O(N)$ 的成本封存了"归一化这一行需要知道的一切"。用 $O(N)$ 换 $O(N^2)$ 的重算自由，这笔交易是整个 Flash Attention 体系里最划算的一笔。

---

## 7. 本篇小结

- **瓶颈判断**：朴素 attention 的瓶颈是 $\Theta(N^2)$ 的中间矩阵在 HBM 上往返，显存占用也是 $\Theta(N^2)$；计算量本身并不是主要矛盾。
- **safe softmax**：减 max 解决数值上溢，但要求三遍扫描——"必须回头再看一遍"才是要消灭的东西。
- **online softmax**：用 $\alpha = e^{m^{\text{old}} - m^{\text{new}}}$ 把 max、sum、输出累加器全部变成流式增量维护，**单遍、精确**。
- **分块算法**：外层 Q 并行（grid）+ 内层 KV 串行（循环）+ 三组寄存器状态（$m_i, \ell_i, \tilde{O}$）+ 循环外归一化；causal 的块级跳过省掉约一半计算。
- **IO 复杂度**：从 $\Theta(N^2)$ 降到 $\Theta(N^2d^2/M)$，SRAM 越大收益越大；FA2 相对 FA1 的核心改动是循环换序，让输出 rescale 留在寄存器。
- **给反向留钩子**：前向输出 LSE（$O(N)$），反向用 recompute + LSE 恢复 $P$。

### 自检三问

1. 如果只做 safe softmax 而不做 online softmax，为什么仍然需要把 $S$ 物化？问题出在哪一遍扫描？
2. 若新块的最大值比旧 max 小，$\alpha$ 等于多少？为什么这个递推里永远不会出现 $\alpha > 1$、把 $\ell$ 撑爆？
3. 第 5 节的 $\Theta(N^2d^2/M)$ 中，把 $B_M$ 从 $M/(4d)$ 提高到 $M/(2d)$，K/V 的重复读会怎么变？为什么工程上不把 SRAM 尽量全划给 Q 块？

下一篇，我们用第 2 节和第 3 节的每一个公式，去逐行对照 Triton 官方教程 `06-fused-attention.py` 的前向实现——你会看到 $m_i$ 初值 $-\infty$ 如何让第一次迭代"自动退化"，看到 `p.to(tl.float16)` 背后的精度边界，以及 causal 掩码的两遍遍历技巧。

*配套代码：本系列第 2 篇起逐篇给出，仓库 [code/](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code) 目录与本文章目录同名。*
