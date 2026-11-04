---
title: "LLM 模型结构变迁（三）：线性注意力与 3:1 混合——把 KV 变成状态"
date: 2026-11-02
draft: false
weight: 4
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "线性注意力", "KDA", "GDN", "Mamba", "混合架构"]
categories: ["LLM算法"]
---

第二篇的结尾留了个问题：注意力怎么压，都还是要「每个 token 存点东西」。那能不能**根本不存**？

线性注意力给的答案是：不存历史，存一个**固定大小的状态**。历史的全部信息被压进一个 $d_k \times d_v$ 的矩阵里，decode 每步只读这个矩阵，与上下文长度无关。

这个想法 2020 年就有，2023 年被 Mamba 带火，2024 年差点成为主流，2025 年却集体退回**混合**：3 层线性 + 1 层全注意力。这一篇讲清楚三件事：状态到底是什么、为什么纯线性不行、以及「3:1」这个比例到底是怎么来的。

## 0. 先记住一个画面

全注意力是**每来一个新读者，就把图书馆所有笔记重读一遍**；线性注意力是**图书馆门口的一块白板**——每来一个人，用他的笔记把白板擦一部分、写一部分。后面的人只看白板。

白板大小固定（$d_k \times d_v$），所以不管来了多少人，读白板的成本不变。代价也很明显：**白板会被写满、会互相覆盖**。你三年前写的一句话，可能已经被后来的一万次擦写冲没了。

> **类比的边界**：白板不是「有损压缩」这么简单——它是一个**可学习的递推**，擦除和写入的规则（$A_t$）本身是模型参数。所以「忘了什么」不是随机的，是被训练出来的。这是线性注意力比「滑窗 + 摘要」强的地方，也是它比全注意力弱的地方：**训练目标只关心最终 loss，不保证任何一条具体信息被记住**。

## 1. 一条公式，六代 $A_t$

所有线性注意力 / SSM 都是同一个递推：

$$
S_t = A_t S_{t-1} + v_t k_t^\top, \qquad o_t = S_t^\top q_t
$$

$S \in \mathbb{R}^{d_k \times d_v}$ 就是「压缩后的 KV cache」——它等价于把历史所有 $(k_i, v_i)$ 做外积累加。$A_t$ 决定「旧记忆怎么衰减、怎么被覆盖」，六代模型的区别全在这一个矩阵上：

| 模型 | $A_t$ | 门控粒度 | 解决什么 |
|---|---|---|---|
| Linear Attention（2020） | $I$ | 无 | 把 $O(N^2)$ 降到 $O(N)$ |
| RetNet（2023） | $\mathrm{diag}(\gamma)$，固定 | per-channel，静态 | 让旧信息指数衰减 |
| GLA（2023） | $\mathrm{diag}(\alpha_t)$，数据相关 | per-channel，动态 | 衰减率随内容变 |
| Mamba-2（2024） | $\mathrm{diag}(\exp(\Delta_t A))$ | per-channel（state dim） | 选择性 SSM + 张量核友好 |
| DeltaNet（2024） | $I - \beta_t k_t k_t^\top$ | 标量 $\beta$ | **覆盖写**（解决 key 冲突） |
| GDN（2024） | $\mathrm{diag}(\alpha_t)(I - \beta_t k_t k_t^\top)$ | per-head 标量 $\alpha$ | 遗忘 + 覆盖，两者互补 |
| **KDA（2025）** | $\mathrm{Diag}(\alpha_t) - \beta_t k_t k_t^\top \mathrm{Diag}(\alpha_t)$ | **per-channel** $\alpha$ | 细粒度遗忘，可顶替 RoPE |
| GDN-2（2026） | $(I - k_t (b_t \odot k_t)^\top)\mathrm{Diag}(\exp(g_t))$ | 擦除/写入**分开** | 两个门各管一个轴 |

读这张表有一条捷径：**Delta 规则管「覆盖」，门控管「遗忘」**。纯累加 $S \mathrel{+}= v k^\top$ 的问题是同一个 key 写两次会叠成 $2v$，检索时读出错误结果；Delta 规则先读一次再减掉（$I - \beta k k^\top$ 是以 $k$ 为法向的 Householder 反射），保证 $S^\top k \approx v$。但它只擦除「当前 key 方向」，管不了「整块内存过时」——那是 $\mathrm{diag}(\alpha_t)$ 的活。

GDN-2 的代码注释把这三代的退化关系写全了（`fla/layers/gdn2.py:45-60`）：把 $b_t = w_t = \beta$（标量广播）就退回 KDA；再把 $g_t$ 塌成标量就退回 GDN v1。**一行注释就是一张架构演化相图。**

## 2. 状态有多大：换算成「等效 KV 长度」

线性注意力最容易被夸大的地方是「$O(1)$ 显存」。$O(1)$ 是相对 $N$ 的，绝对值并不小。换算公式：

$$
L_{\text{eq}} = \frac{H_v \cdot d_k \cdot d_v}{2 \cdot H_{kv} \cdot d_{\text{head}}}
$$

代 Qwen3-Next-80B-A3B 的真实配置（hidden 2048、48 层、16 个 Q 头 / 2 个 KV 头 / head_dim 256；线性侧 32 个 v-head × 128 × 128）：

- 每个线性层 state = $32 \times 128 \times 128 = 524{,}288$ 元素 = **1 MiB**（bf16）；
- 每层全注意力每 token KV = $2 \times 2 \times 256 = 1024$ 元素 = **2 KiB**；
- $L_{\text{eq}} = 524288 / 1024 = 512$ **个 token**。

也就是说：**一个线性层的 state 只相当于 512 个 token 的 KV**。整个模型 36 个线性层 = 36 MiB，要和 12 个全注意力层的 KV 打平，需要 $T \approx 1536$（bf16 state）到 $3072$（fp32 state）个 token。

**短于这个长度，纯注意力模型的 cache 反而更小。** 这是第一个反直觉的结论，也是很多「小模型上线性注意力没优势」的原因：2K 上下文时，线性层的固定 state 比 KV 还占地方。

## 3. 「3:1」不是魔法数字

2025-2026 年几乎所有混合模型都用 3:1（Qwen3-Next、Kimi Linear、Kimi-K3），少数用 7:1 或 8:1（Nemotron-H 约 8%）。这个比例从哪来？

**一半是算术**：3 层线性 + 1 层全注意力，KV cache 恰好是纯注意力的 $1/4$。1M 上下文下，Qwen3-Next 的 KV 从 91.6 GiB 降到 22.9 GiB。

**另一半是消融**，而且结果很残酷：

| 配置 | 验证 ppl（Kimi Linear，1.4T token 同配方） |
|---|---|
| 0:1（纯全注意力） | 5.77 |
| 3:1 | **5.65** |
| 7:1 | 5.70 |
| 15:1 | 5.82 |

Mamba-2 的消融更直接（350M / 48 层）：0 层注意力 8.60 → 6 层 8.26 → 24 层 8.50。**注意力层从 6 加到 24，反而变差**，连纯线性的 8.60 都没追回来。

所以 3:1 是「刚好够用」：**加注意力的收益极快饱和，而每加一层都在付 KV 的代价**。7:1 训练 loss 相同但验证差 0.05，说明泛化上全注意力还是重要；15:1 开始塌。

## 4. 纯行为什么不行：检索塌方

线性注意力的状态是**有损**的，而且损失是**不可控**的。同一份配置下，纯线性模型在「大海捞针」（NIAH）上的表现：

| 模型（1.3B 级） | NIAH-1 @8K | NIAH-2 @8K |
|---|---|---|
| Mamba-2 | 30.4 | — |
| DeltaNet | — | 14.4 |
| GDN（纯线性里最好） | — | 92.2 → **29.6** |
| Transformer++ | 99+ | 99+ |

真实检索任务的平均分（同参数同配方）：GDN 30.6 < Transformer++ 37.0 < GDN+1 层注意力 39.0 < GDN+2 层 40.1。**加一层全注意力就把差距补回来大半**——这就是混合架构的全部秘密。

原因不复杂：状态是 $d_k \times d_v$ 的矩阵，它能精确记住的「键值对」数量受维度限制（类似联想记忆的容量上界）。要在这块白板上精确检索第 8000 个 token 写的某个具体事实，本质上是要求它做**无碰撞的哈希**——做不到。

## 5. 但混合架构没有解决 prefill

这是最常被误读的一点。混合架构把 KV 降到 1/4，于是大家默认「算力也降到 1/4」。实际算一下 Qwen3-Next 在 1M 上下文下的 prefill FLOPs 构成：

| 上下文 | 注意力占比 | 线性层占比 |
|---|---|---|
| 32K | **98.3%** | 1.7% |
| 1M | **99.9%** | 0.1% |

因为注意力的 FLOPs $\propto T^2$，而线性层 $\propto T$。层数减到 1/4，只是把二次项系数乘了 1/4——**二次项还是二次项**。

所以混合架构真正解决的是 **decode 的显存和带宽**，不是 prefill 的算力。这解释了为什么长上下文服务（decode 主导）对混合架构最积极，而训练侧（prefill 主导）兴趣有限。

实测数字也印证这个分工：Kimi Linear 在 1M 上下文下 TPOT 1.84 ms vs 11.48 ms（6.3×），KV 降 75%；Nemotron-H 在 64K 下 3× 吞吐；Qwen3-Next 在 32K 以上比 Qwen3-32B 快 10× 以上。

## 6. 训练怎么并行：chunkwise 的四个阶段

线性注意力在训练时不能串行递推（$T$ 步依赖），必须用 **chunkwise parallel**：把序列切成 $C$ 大小的块，块内当小注意力算（$C \times C$ 矩阵，吃 tensor core），块间只传 state。

FLA 的实现路径（`fla/ops/.../chunk.py`）：

1. **门控 cumsum**（`chunk.py:63`）：$g_t = \mathrm{cumsum}(\log \alpha)$，块内取局部 cumsum，避免跨块数值爆炸；
2. **块内 UT / WY 表示**（`chunk.py:72` → `chunk_fwd.py:40/332`、`wy_fast.py:255`）：解 $M = I + \mathrm{tril}(\mathrm{diag}(\beta) K K^\top, -1)$，得 $w = M^{-1}(\beta \odot k)$、$u = M^{-1}(\beta \odot v)$——把块内一串 Householder 连乘压成一个稠密矩阵，这是 delta 规则能上 tensor core 的关键；
3. **块间状态递推**（`chunk.py:95` → `chunk_delta_h.py:688`）：$S_{[c]} = A_{[c]} S_{[c-1]} + K_{[c]}^\top u_{[c]}$，并把每块的 state 物化下来给反向用（显存 $[B, N_T, H_V, K, V]$）；
4. **输出**（`chunk.py:111`）：跨块 $q^\top S_{[c-1]}$ + 块内 $(q k^\top \odot \text{decay}) v$ + 输出门。

chunk 大小是个工程参数：GDN 只支持 16/32/64（默认 64），KDA 32/64，Mamba-2 256。块越大，块内二次项越贵；块越小，块间递推次数越多、state 物化越频繁。

**chunkwise 和递推是数学等价的**（只差浮点累加顺序），不是近似——这一点必须说清楚，否则会被误解成「稀疏近似」。FLA 里 layer 会自动切换：`q_len <= 64 and not training` 时走 recurrent 路径（`fla/layers/gated_deltanet.py:238-244`）。

## 7. KDA：为什么 per-channel decay 能顶替 RoPE

Kimi Linear 的 KDA 相对 GDN 只改了一个地方：$\alpha$ 从 per-head 标量变成 per-channel 向量，用低秩投影产生（rank = head_dim）。

这个改动带来一个连锁反应：

1. per-channel 的 decay 意味着**不同维度有不同的衰减频率**——高频维度记近处，低频维度记远处；
2. 这正是 RoPE 在做的事（不同维度不同旋转频率）；
3. 所以 **KDA 层自己就带位置感**，全注意力层可以上 **NoPE**（不加位置编码）。

Kimi-K3 就是这么干的：69 层 KDA + 24 层 NoPE 的 Gated MLA，位置信息由 KDA 的逐通道 decay 提供。这是第六篇的主题，这里先记住结论：**位置编码的退场，是线性注意力普及的副产品**。

代价是数值工程。细粒度 decay 会导致块内除法精度问题，KDA 给门加了个硬下界：

$$
g = \text{lower\_bound} \cdot \sigma\big(\exp(A_{\log})(g + dt_{\text{bias}})\big), \quad \text{lower\_bound} = -5
$$

即 $\alpha \in [e^{-5}, 1)$（`fla/ops/kda/gate.py:62-69`）。有了这个界，kernel 才能假设 $g \in [-5, 0)$ 并走 TensorCore 路径。**一个数学上更优雅的门，必须被砍成有界的才能上硬件**——这是线性注意力论文里最不会写、但工程上最要命的一类细节。

## 8. 三个坑

**坑一：短上下文反而更费。** 上面算过，$L_{\text{eq}} \approx 512$ token/层。2K 上下文时，线性层的 state 总量（36 MiB）比 KV（12 层 × 2K × 2 KiB = 48 MiB）小不了多少，而线性层的参数更分散、kernel 更多。**线性注意力的优势只在长序列 decode 兑现。**

**坑二：state 精度。** 训练时 state 通常 fp32，推理时降到 bf16 会掉点；Kimi 的做法是 chunk 内 bf16、块间 state fp32。这个选择和「等效 KV 长度」直接相关——fp32 state 让 $L_{\text{eq}}$ 翻倍到 1024。

**坑三：2K 训练时 Transformer++ 吞吐最高。** GDN 论文自己写的：短序列训练时，线性层的 chunkwise kernel 反而不如 FlashAttention 成熟。**结构优势不等于当天就能变现**，kernel 生态的成熟度经常比结构本身更决定成败。

## 9. 代码在哪

- **FLA**：`fla/ops/gated_deltanet/chunk.py`（四阶段）、`fla/ops/kda/gate.py`（门的下界）、`fla/layers/gdn2.py:45-60`（三代退化关系）；
- **transformers**：`models/qwen3_next/modeling_qwen3_next.py:672-674`（门在 kernel 内算）、`configuration_qwen3_next.py:87-103`（`full_attention_interval=4`）；
- **Kimi-K3**：`config.json` 的 `linear_attn_config`（69 层 KDA、96 头、head_dim 128）。

下一篇换方向：从「序列」转到「宽度」，看 MoE 怎么把 2.78T 参数塞进 104B 的算力里。

---

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. 本篇：线性注意力与 3:1 混合
5. [MoE 十年]({{< relref "llm-arch-04-moe" >}})
6. [残差流的加宽：HC 与 mHC]({{< relref "llm-arch-05-residual" >}})
7. [位置的退场]({{< relref "llm-arch-06-position" >}})
8. [条件记忆]({{< relref "llm-arch-07-memory" >}})
9. [五个收敛、三个分叉与三个赌注]({{< relref "llm-arch-08-trends" >}})
10. [残差连接与 AttnRes]({{< relref "llm-arch-09-attnres" >}})
