---
title: "Flash Attention 精读（一）：从 softmax 的内存瓶颈到 online softmax 的数学推导"
date: 2026-08-28
draft: false
weight: 1
series: ["flash-attention"]
tags: ["flash-attention", "attention", "gpu", "softmax", "系列"]
categories: ["算子开发"]
---

本系列共 5 篇，从数学原理一路读到工业级实现的每一行代码：

1. **（本篇）原理与数学推导** —— 为什么 attention 慢、safe softmax、online softmax 的流式推导、分块算法与 IO 复杂度
2. [Triton 教程版前向逐行精读]({{< relref "flash-attention-02-triton-fwd" >}}) —— 以 200 行 Triton 代码对照本篇数学
3. FlashAttention-2/3 CUDA 前向实现 —— 工业级 kernel 的调度、流水线与 warp specialization
4. 反向梯度推导与 recompute 实现 —— dQ/dK/dV 的完整推导与三种实现对照
5. Gluon / TileLang / Liger 多实现对比 —— 不同抽象层级下的同一算法

本篇不涉及任何具体代码，只回答一个问题：**Flash Attention 到底解决了什么问题，它的数学基础是什么**。读完你应该能独立推导出 online softmax 的递推公式，并理解为什么它是**精确计算**而非近似。

## 1. Attention 的两个代价

标准自注意力（self-attention），输入 $Q, K, V \in \mathbb{R}^{N \times d}$：

$$
S = \frac{1}{\sqrt{d}} QK^\top \in \mathbb{R}^{N \times N}, \qquad
P = \mathrm{softmax}(S) \in \mathbb{R}^{N \times N}, \qquad
O = PV \in \mathbb{R}^{N \times d}
$$

计算量：两次 $N \times N \times d$ 的矩阵乘，共约 $4N^2 d$ FLOPs。**计算量不是问题**——这与一个 $d \times N \times N$ 的 GEMM 同阶，GPU 的 tensor core 每秒能做几百 TFLOPs。

问题在**访存**。朴素的 PyTorch 实现：

```python
S = (Q @ K.T) / math.sqrt(d)   # 物化 N×N 矩阵，写回 HBM
P = S.softmax(-1)              # 读 N×N，写 N×N
O = P @ V                      # 读 N×N
```

以 $N = 8192$、fp16 存储为例，一个 $N \times N$ 矩阵是 **128 MB**。而这三个操作之间的每一次中间结果读写都发生在 HBM（高带宽显存，A100 上约 2 TB/s）上。GPU 的片上 SRAM（共享内存 + 寄存器）只有几百 KB，比 HBM 快一个数量级以上，但装不下这个矩阵。

把账算清楚：

- **FLOPs**：$4N^2 d \approx 4 \times 8192^2 \times 128 \approx 3.4 \times 10^{10}$，在 300 TFLOPS 的 tensor core 上只需 ~0.1 ms（理论值）。
- **HBM 流量**：$S$ 和 $P$ 各写一次、各读一次以上，$\ge 4N^2 \cdot 2\text{B} \approx 5 \times 10^8$ 字节级流量……看似不大？但注意这只是一个 head、一个 batch。真正的要害在于：**这些中间矩阵的读写不参与任何有效计算**，它们的存在仅仅是因为 softmax 需要看到一整行才能归一化。

这就是 Flash Attention 论文（Dao et al., 2022）的核心观察：

> Attention 朴素实现的瓶颈不是算力，而是 HBM 带宽被中间矩阵 $S$、$P$ 的往返浪费了。如果我们能**根本不把 $S$、$P$ 写出片上内存**，attention 就能从"内存受限"回到"算力受限"。

（这里要诚实说明：$O(N^2)$ 的**计算量**并没有变，Flash Attention 省的是 $O(N^2)$ 的**内存**与大部分 $O(N^2)$ 的**访存**。训练时长序列下显存从 $O(N^2)$ 降到 $O(N)$，这本身就是巨大的收益——不存 attention matrix 意味着能用大得多的序列训练。）

## 2. Safe Softmax：三遍扫描与数值稳定

要消灭中间矩阵，先得理解 softmax 为什么"必须看全一行"。softmax 的定义（对 $S$ 的第 $i$ 行）：

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{N} e^{S_{ik}}}
$$

两个麻烦：

**数值上溢**。$e^{x}$ 在 $x \gtrsim 88$（fp32）时溢出为 inf，而 $S_{ij}$ 是两个 $d$ 维向量的内积再乘 $1/\sqrt d$，量级完全可能到几十上百。

**分母是全行求和**。要归一化第 $i$ 行，必须知道整行的 $e^{S_{ik}}$ 之和——这看起来要求"一次性看到所有 key"。

第一个麻烦的标准解法是 **safe softmax**：利用 softmax 对加法常数的不变性。对任意 $c$：

$$
\frac{e^{S_{ij} - c}}{\sum_k e^{S_{ik} - c}} = \frac{e^{S_{ij}}}{\sum_k e^{S_{ik}}}
$$

证明只需分子分母同乘 $e^c$。取 $c = m_i = \max_k S_{ik}$，则所有指数的参数 $\le 0$，指数值落在 $(0, 1]$，既不上溢、也不下溢。这就是教科书式的三遍扫描（three-pass）softmax：

$$
\text{Pass 1: } m_i = \max_k S_{ik} \qquad
\text{Pass 2: } \ell_i = \sum_k e^{S_{ik} - m_i} \qquad
\text{Pass 3: } P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}
$$

三遍扫描要求 $S$ 的第 $i$ 行被完整地读三次（或者物化在显存里）。如果数据能装进片上内存，三遍也无所谓；问题是装不下。

> **技术直觉**：safe softmax 的"减 max"不是精度锦上添花，而是 fp16/fp32 下指数运算的生存条件。后面所有实现细节——哨兵值用 $-10^6$ 而不是 $-\infty$、scale 乘在内积之后减 max 之前——都服务于同一目标：保证送进 exp 的参数永远是一个 $\le 0$ 的有界数。

## 3. Online Softmax：把三遍压成一遍

Online softmax（Milakov & Gimelshein, 2018）解决第二个麻烦。关键洞察：**max 和 sum 可以增量维护**。

假设 key 序列被分成两块，先看到 $S^{(1)}$（前 $n_1$ 个 key），再看到 $S^{(2)}$（后 $n_2$ 个）。定义部分量：

$$
m^{(1)} = \max_{k \le n_1} S_{ik}, \qquad \ell^{(1)} = \sum_{k \le n_1} e^{S_{ik} - m^{(1)}}
$$

看到第二块后，正确的全局量是：

$$
m = \max(m^{(1)}, m^{(2)}), \qquad
\ell = \sum_{k \le N} e^{S_{ik} - m}
$$

问题：$\ell^{(1)}$ 是按旧 max $m^{(1)}$ 归一化的，怎么修正到新 max $m$？一行代数：

$$
\ell^{(1)}_{\text{corrected}} = \sum_{k \le n_1} e^{S_{ik} - m}
= \sum_{k \le n_1} e^{S_{ik} - m^{(1)}} \cdot e^{m^{(1)} - m}
= \ell^{(1)} \cdot e^{m^{(1)} - m}
$$

**旧的部分和乘上一个修正因子，就迁移到了新的坐标系**。于是得到流式递推——每来一个新的 key 块 $j$：

$$
\begin{aligned}
m^{\text{new}} &= \max(m^{\text{old}}, \max_j S_{ij}) \\
\ell^{\text{new}} &= e^{m^{\text{old}} - m^{\text{new}}} \cdot \ell^{\text{old}} + \sum_j e^{S_{ij} - m^{\text{new}}} \\
P_{ij} &= e^{S_{ij} - m^{\text{new}}} \quad (\text{未归一化，最终再除 } \ell)
\end{aligned}
$$

用归纳法容易验证：处理完所有块后，$m = \max_k S_{ik}$，$\ell = \sum_k e^{S_{ik} - m}$，与三遍扫描结果完全一致。**这是精确算法，没有任何近似**——"approximate attention"（Linformer、Performer 等）是另一条技术路线，与 Flash Attention 无关。

到这里，softmax 的归一化已经不需要"一次看全一行"了。剩下的拦路虎是输出 $O = PV$：$P$ 的第 $i$ 行也要对全部 key 求和，而不同块贡献的 $P$ 是在不同 max 下算的——同样用修正因子迁移：

$$
\begin{aligned}
O_i &= \frac{1}{\ell} \sum_k e^{S_{ik} - m} V_k \\
&= \frac{1}{\ell} \left( e^{m^{\text{old}} - m} \cdot \underbrace{\sum_{k \in \text{旧块}} e^{S_{ik} - m^{\text{old}}} V_k}_{\text{running } \tilde{O}} + \sum_{k \in \text{新块}} e^{S_{ik} - m} V_k \right)
\end{aligned}
$$

即维护一个**未归一化的输出累加器** $\tilde{O}$，每次新块到来：$\tilde{O} \leftarrow \alpha \tilde{O} + P_{\text{新块}} V_{\text{新块}}$，其中 $\alpha = e^{m^{\text{old}} - m^{\text{new}}}$。

> **技术直觉**：online softmax 的本质是把"归一化坐标系"从静态（全局 max 一次定死）变成动态（每块更新一次），并用 $\alpha$ 因子把旧坐标系的累计量搬运到新坐标系。所有状态——$m_i$（行最大）、$\ell_i$（行和）、$\tilde{O}_i$（未归一化输出）——大小都是 $O(N \cdot d)$ 或 $O(N)$，与 key 数量无关，可以永久驻留在寄存器里。

## 4. 分块算法：两级 tiling

把 online softmax 装进 GPU，就得到 Flash Attention 的骨架：

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

对照第 3 节的递推式，逐行都是数学的直接翻译。三个值得强调的设计决策：

**（1）归一化推迟到循环外**。循环内 $P$ 一直是"未归一化分子"，除法只在最后做一次。原因有二：其一，中间块的 $\ell_i$ 不完整，除了也是错的；其二，除法（或乘倒数）在 GPU 上比乘法贵，每块做一次是 $O(N/B_N)$ 次，循环外做是 1 次。

**（2）因果掩码的分块处理**。causal mask 下，第 $i$ 个 query 只能看到前 $i+1$ 个 key。一个 $B_M \times B_N$ 的 tile 与对角线的位置关系只有三种：完全在下三角内（不需要 mask）、完全在外（**整块跳过，一次 dot 都不用算**）、跨越对角线（块内逐元素 mask）。因果模型的下三角占一半面积，块级跳过直接省一半计算——这是 Flash Attention 实现 causal attention 比 full attention 快约 2 倍的原因。

**（3）FA1 与 FA2 的循环方向之差**。上面这个"外层 Q、内层 KV"的结构是 FlashAttention-**2** 的写法。FA1 恰好相反：外层 KV、内层 Q，输出 $O$ 在内层被多个 Q 块共享写入，每次 softmax 状态更新都要把**部分输出从 HBM 读回、rescale、再写出**。FA2 把循环换过来之后，每个 CTA 的输出 tile 有了唯一属主，rescale 全部发生在寄存器里，HBM 上的中间读写归零，同时 Q 块天然适配 tensor core 的行主布局。这个"只是换了个循环顺序"的改动带来了 1.7~1.8 倍的加速——**循环结构决定数据流向，数据流向决定性能**，这是本系列反复出现的主题。

## 5. IO 复杂度：为什么快

论文里的定理值得自己推一遍。设 SRAM 大小为 $M$ 字节，$d \le M/4$（一个 tile 的 K 或 V 装得下）。

**朴素实现**的 HBM 访问：读写 $S$、$P$ 各 $\Theta(N^2)$ 次，加上读写 $Q,K,V,O$ 的 $\Theta(Nd)$，总计 $\Theta(N^2 + Nd)$。

**分块实现**：每个 Q 块（大小 $B_M \times d$）要扫全部 K/V——$\lceil N/B_M \rceil$ 个 CTA 各读一遍 K 和 V，共 $\Theta(N^2 d^2 / M)$（代入 $B_M \approx M/4d$）。

$$
\text{HBM 访问：朴素 } \Theta(N^2) \;\longrightarrow\; \text{Flash } \Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

当 $d = 128$、$M = 256\text{KB}$（约 tile 128×128 fp16 的规模）时，$d^2/M \ll 1$——Flash 的 HBM 流量比朴素实现低一个数量级以上，且 **SRAM 越大收益越大**（这也是后续架构每次加大 SRAM，attention kernel 就跟着变快的原因）。

L2 cache 的存在让实际数字更好：多个 CTA 同时跑相似的 KV 访问序列时，K/V 的重复读大部分命中 L2，实际 HBM 流量接近 $\Theta(Nd)$ 级。工程上还会用 grouped scheduling（相邻 CTA 处理相邻 Q 块，让它们读 KV 的进度接近）来刻意提高 L2 命中率——第 3 篇讲 CUDA 实现时会看到 FA2/FA3 具体怎么做。

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

即重算 $S_{ij} = q_i \cdot k_j / \sqrt d$ 后，减去 LSE 直接指数化，分子分母一次到位——不需要重跑 online softmax，也不需要存 $\ell_i$ 单独一份数组。这就是前向 kernel 的 epilogue 都要顺手写出一行 LSE 的原因（第 2 篇你会看到 Triton 教程里 `m_i + tl.math.log2(l_i)` 那行，以及实现里把 $1/\ln 2$ 折进 scale、让 LSE 以 2 为底存储的工程化处理）。

> **技术直觉**：LSE 是 softmax 分布的"充分统计量"——它以 $O(N)$ 的成本封存了"归一化这一行需要知道的一切"。用 $O(N)$ 换 $O(N^2)$ 的重算自由，这笔交易是整个 Flash Attention 体系里最划算的一笔。

## 7. 本篇小结

- 朴素 attention 的瓶颈：$\Theta(N^2)$ 的中间矩阵在 HBM 上往返，显存占用也是 $\Theta(N^2)$。
- safe softmax 解决数值稳定（减 max），但要求三遍扫描；online softmax 用 $\alpha = e^{m^{old} - m^{new}}$ 修正因子把 max/sum/输出累加器全部变成流式增量维护，**单遍、精确**。
- 分块算法 = 外层 Q 并行（grid）+ 内层 KV 串行（循环）+ 三组寄存器状态（$m_i, \ell_i, \tilde{O}$）+ 循环外归一化；causal 的块级跳过省一半计算。
- IO 从 $\Theta(N^2)$ 降到 $\Theta(N^2 d^2 / M)$；FA2 相对 FA1 的核心改动是循环换序（Q 外 KV 内），让输出 rescale 留在寄存器。
- 前向输出 LSE（$O(N)$），反向用 recompute + LSE 恢复 $P$，梯度推导见第 4 篇。

下一篇，我们用第 2 节和第 3 节的每一个公式，去逐行对照 Triton 官方教程 `06-fused-attention.py` 的前向实现——你会看到 `$m_i$ 初值 -inf、$\ell_i$ 初值 1` 这样的初始化如何让第一次迭代"自动退化"，看到 `p.to(tl.float16)` 背后的精度边界，以及 causal 掩码的两遍遍历技巧。

*配套代码：本系列第 2 篇起逐篇给出，仓库 [code/](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code) 目录与本文章目录同名。*
