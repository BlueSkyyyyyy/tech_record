---
title: "LLM 模型结构变迁（二）：注意力四代——从 GQA 到 MLA 到稀疏压缩"
date: 2026-11-02
draft: false
weight: 3
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "attention", "MLA", "DSA", "稀疏注意力", "KV-cache"]
categories: ["LLM算法"]
---

上一篇的账本里，有一项数字最难堪：**Llama-3.1-405B 在 1M 上下文下要 541 GB 的 KV cache**——单请求，超过任何一张卡。而 DeepSeek-V4.1 只要 933 MB。580 倍。

这 580 倍不是某个技巧变出来的，是四代注意力各自压掉一个维度、乘起来的结果。这一篇就按这个乘法顺序讲：**每 token 每层的字节数（entry size）× 序列维（多少 token 一个条目）× 层维（多少层共用一个条目）**，再加上一个不在乘法里、但决定成败的因子：**每个 query 要打分多少个位置**。

## 0. 先记住一个画面

把注意力想成**每个 token 都拖着一个行李箱**：箱子里装着这个 token 的 K 和 V，后面每个 token 都要打开前面所有人的箱子看一遍。

- **MHA**：每个人一个箱子，箱子有 2 个头那么大（K 和 V 各一份，每个头一份）；
- **GQA**：8 个人共用一个箱子（8 个 KV 头），箱子没变小，但箱子数少了；
- **MLA**：把箱子换成**压缩包**，里面只放一个 512 维的 latent，用的时候现场解压；
- **DSA / CSA**：不再打开所有箱子，先用一个「目录」挑出 512 个最相关的，只开这些；
- **CSA2**：连目录都跨层共用，40 层只留 4 份。

> **类比的边界**：开箱子（读 KV）花的是**带宽**，挑箱子（索引器）花的是**算力**，箱子占的地方（显存）是**容量**。这三笔账不能互相抵消——GQA 省的是带宽和容量，DSA 省的是带宽，但索引器反过来吃算力。所以每换一代，总有一项变贵。

## 1. 四代的总账

| 代际 | 每层每 token KV | 每个 query 打分位置 | 每位置单价 | 新增的代价 |
|---|---|---|---|---|
| MHA（2017） | $2 n_h d_h$ = 32 KB（$n_h{=}64,d_h{=}128$） | $N$ | $2 d_h$ | — |
| GQA（2023） | $2 n_g d_h$ = 4 KB（$n_g{=}8$） | $N$ | $2 d_h$ | 质量损失，需 uptraining |
| MLA（2024） | $d_c + d_R$ = 576 元素 = **1152 B** | $N$ | $2(d_c{+}d_R)$ = 1152 | 单价 ×4.5，FLOPs 变高 |
| DSA（2025） | 同 MLA = 1152 B | $N$（索引器）+ $k$（core） | 索引器 $2H_I d_I$ | 索引器仍 $O(N)$ + 三步训练 |
| CSA2（2026） | **18 B**（890 B ÷ 40 层，含索引器 K） | 常数 16384 + $k$ | 同上 | 跨层依赖，训练要 shadow indexer |

四个数字放在一起看：**从 32 KB 到 18 B，是 1800 倍**。但注意第二列——MLA 之后「位置数」和「单价」开始反向运动，这才是 2025-2026 年所有稀疏注意力论文真正的战场。

## 2. 第一代压缩：GQA 只省字节，不省 FLOPs

GQA 的公式简单到不值得展开：

$$
\text{KV bytes/token} = b \cdot L \cdot 2 n_g d_h
$$

$n_g$ 从 $n_h$ 降到 8，KV 线性降 8 倍；但**注意力 FLOPs 一点没变**——因为 Q 还是 $n_h$ 个头，每个头还是要和 $N$ 个位置做点积：

$$
\text{FLOPs}_{\text{dense}} = 4 n_h d_h N L
$$

这是「entry-size 压缩」的本质：它把 decode 从 compute-bound 推向 memory-bound，但**不减少计算量**。GQA 论文（arXiv:2305.13245）真正的工程贡献是 uptraining：用 5% 的原始预训练算力把已有 MHA checkpoint 转成 GQA，转换后 GQA-8-XXL 的推理时间从 1.51 降到 0.28（相对 MHA-XXL），质量基本不变。

## 3. MLA：压缩包 + 解耦 RoPE + 吸收

DeepSeek-V2（arXiv:2405.04434）的出发点是：GQA 把 KV 头砍到 8 已经掉点了，还能不能**不砍头、砍维度**？

### 3.1 压缩

$$
c_t^{KV} = W^{DKV} h_t \in \mathbb{R}^{512},\qquad
k_t^C = W^{UK} c_t^{KV},\qquad v_t^C = W^{UV} c_t^{KV}
$$

只缓存 $c_t^{KV}$（512 维），K/V 用的时候现算。每层每 token 的元素数从 $2 n_h d_h = 32768$ 降到 $512 + 64 = 576$——**57 倍**。

### 3.2 为什么必须「解耦」RoPE

这是 MLA 最容易被误解的一步。RoPE 是对 K 做位置相关的旋转：

$$
k_j \leftarrow R_{\theta,j} k_j = R_{\theta,j} W^{UK} c_j
$$

问题是 $R_{\theta,j}$ 与 $j$ 有关，**不能**和 $W^{UK}$ 合并成一个与位置无关的矩阵。而 MLA 的省带宽全靠「把 $W^{UK}$ 吸收进 $W^Q$、从而不用物化 K」——一旦有 RoPE，这条路就断了：推理时必须为每个历史位置重新算一遍 K。

DeepSeek-V2 的解法是给 RoPE 单开一条**窄通道**：额外 64 维的 $k^R_t = \mathrm{RoPE}(W^{KR} h_t)$，每个头共享；q 侧对应 64 维 $q^R$。这样 RoPE 只作用在 64 维上，剩下的 512 维保持「可吸收」。

### 3.3 吸收之后，MLA 等价于什么

把 $W^{UK}$ 吸收进 Q 侧：

$$
q_{t,i}^\top k_{j,i} = [W_i^{UK\top} q^C_{t,i}]^\top c_j^{KV} + q^{R\top}_{t,i} k^R_j
\quad\Longrightarrow\quad
\tilde q_{t,i} := [\,W_i^{UK\top} q^C_{t,i};\; q^R_{t,i}\,]
$$

于是推理时的 MLA **严格等价于一个 MQA**：所有 64 个头共享同一份 576 维 latent 作为 K 和 V，外加每个头一次低秩升维（Q 侧）和一次低秩降维（输出侧）。FlashMLA 的 `head_dim_k=576, head_dim_v=512` 就是这么来的。

### 3.4 代价：单价贵 4.5 倍

吸收之后，每个头要在 **576 维**上做点积，而不是 128 维。同样 64 个头、同样 1M 上下文：

$$
\text{MLA}: 2 \times 64 \times (576 + 512) \times N \approx 4.5 \times \text{GQA}(d_h{=}128)
$$

**MLA 用 FLOPs 换带宽。** 在 128K 上下文、batch=1 的 decode 里，这笔交易稳赚（带宽是瓶颈）；但当 batch 大到权重和 KV 都能被摊薄、算力变成瓶颈时，MLA 反而更贵。这就是为什么 V4 把 576 维的 MLA 又改回了 512 维的共享 KV（MQA）+ 分组输出投影——**在更长的上下文里，连 4.5 倍的单价都嫌贵**。

## 4. DSA：把「位置数」从 $N$ 打到 $k$

MLA 之后，KV 字节已经不是主要矛盾了。DeepSeek-V3.2（arXiv:2512.02556）换了个维度：**不看所有位置，只看 2048 个**。

### 4.1 闪电索引器

$$
I_{t,s} = \sum_{j=1}^{H_I} w^I_{t,j} \cdot \mathrm{ReLU}\big(q^I_{t,j} \cdot k^I_s\big)
$$

三个设计细节值得记：

- **ReLU 而不是 softmax**：不需要跨序列归一化，纯 GEMM + 逐元素，FP8 友好；
- **索引器是「小注意力」**：$H_I$ 个头、$d_I=128$ 维，比主注意力便宜一个量级；
- **top-k 作用在 MLA 的 MQA 模式上**：每个 latent 被该 token 的所有 query 头共享，kernel 才能高效。

### 4.2 稀疏不免费：索引器仍是 $O(N)$

DSA 的 core attention 从 $O(N^2)$ 降到 $O(Nk)$，但**索引器要给每个可见位置打分，仍是 $O(N^2)$**——论文 §2.3 自己承认了这一点。算笔账（每 token，单层，GFLOP）：

| 上下文 | core（$k{=}2048$） | 索引器 | 索引器 / core |
|---|---|---|---|
| 128K | 35 | 131 | **3.8×** |
| 1M | 35 | 1,048 | **30×** |

也就是说：**上下文越长，DSA 越像是「索引器在算，主注意力在打酱油」**。这直接解释了 2026 年两条后续路线：把索引器也稀疏化（V4.1 的层次化候选池），或者把序列先压缩再索引（CSA）。

把两条曲线解出来，交点大约在 **$N^* \approx 35$K**：短于 35K 时 core 更贵，长于 35K 后索引器主导。2025 年之前的主流上下文是 8K-32K，所以 DSA 的索引器开销看起来「可接受」；2026 年上下文冲到 1M，索引器就成了主要矛盾。

顺带一个更少人注意的指标：**算术强度**（每读 1 字节 HBM 做多少次浮点运算）。H800 的平衡点约 295 FLOP/B：

| 模型 | @128K 算术强度 | 处于什么区间 |
|---|---|---|
| Llama-3.1-405B（GQA） | 16 | 严重 memory-bound |
| DeepSeek-V3（MLA） | **242** | 接近 compute-bound |
| DeepSeek-V3.2（DSA） | 18 | 又回到 memory-bound |
| DeepSeek-V4.1（CSA2） | 12-15 | memory-bound，且瓶颈变成权重 |

这张小表说明了一条完整的弧线：**MLA 用 FLOPs 换带宽，把 decode 推到接近 compute-bound；稀疏化又把 FLOPs 砍回去，重新变成 memory-bound**。所以 kernel 优化的重点也跟着变——MLA 时代优化 Tensor Core 利用率（FlashMLA 的 576 维点积），稀疏时代优化索引开销和访存 pattern。**「稀疏 = 省算力」是个误解，它省的是带宽，代价是算力。**

### 4.3 训练代价：三步、~946B token

DSA 不是「加个 top-k 就完了」。V3.2 的训练分三步：

1. **dense warm-up**：冻结主模型，用主注意力的分数当目标，KL 散度训练索引器，1000 步 × 16 seq × 128K = 2.1B token；
2. **sparse training**：加入 top-k，索引器只在选中集合上对齐，15000 步 × 480 seq × 128K = 943.7B token；
3. **梯度隔离**：索引器输入 detach，只由索引损失训练，主模型只由 LM loss 训练。

**索引器是一个需要单独预热的子系统**——这是所有「学出来的稀疏」共同的税。

## 5. NSA / MoBA：块级稀疏的两个坑

NSA（arXiv:2502.11089）用三条分支（压缩 / 选择 / 滑窗）加权求和；MoBA（arXiv:2502.13189）把 MoE 的门控搬到 KV 块上，块大小 512、top-3，论文报告 1M prefill 6.5×。

MoBA 的三个因果性设计值得单独抄下来，因为**少一个就会训崩**：

1. 未来块分数设为 $-\infty$；
2. **当前块强制入选**，且块内用因果 mask（mean-pool 会把块内未来 token 的信息泄进来）；
3. 块划分粒度可调。

第 2 条最隐蔽：如果只做 1 和 3，当前块可能落选，模型就会「看不见自己」——MoBA 论文把它类比成 MoE 里的 shared expert。

## 6. CSA / CSA2：三个维度一起压

DeepSeek-V4（arXiv:2606.19348）的 CSA 和 V4.1 的 CSA2，是这条线上目前走到最远的：**条目维、序列维、层维同时压，最后连索引器也压成常数**。

### 6.1 序列维：每 $m$ 个 token 一个条目

CSA 用两套 KV（$C^a, C^b$）和两套门（$Z^a, Z^b$），相邻压缩条目**重叠** $m$ 个源 token：

$$
[S^a_{mi:mi+m-1}; S^b_{m(i-1):mi-1}] = \mathrm{Softmax}_{\text{row}}\big([Z^a + B^a;\; Z^b + B^b]\big)
$$

$$
C_i^{\mathrm{Comp}} = \sum_j S^a_j \odot C^a_j + \sum_j S^b_j \odot C^b_j
$$

每个压缩条目由 $2m$ 个源条目算出，但两套索引区间重叠，**净压缩率仍是 $1/m$**。HCA 更简单：单套 KV、无重叠、$m'=128$，几乎不稀疏。

V4-Flash 的配置是 20 层 CSA（$m=4$）+ 20 层 HCA（$m'=128$）交替。CSA2 把重叠和绝对位置嵌入都去掉了——**简化本身也是收益**，因为压缩器不再需要维护两套门和位置偏置。

### 6.2 层维：40 层只留 4 份 KV

CSA2 把每层静态指定为三种模式之一：

| 模式 | main KV | indexer K | indexer Q | Top-K 索引 | 省什么 |
|---|---|---|---|---|---|
| **Full** | 本层算 | 本层投影 | 本层算 | 本层算 | 什么都不省（基准） |
| **Reindex** | 复用 | 复用 | 本层算 | 本层重算 | 省 KV + 索引 K 的**存储** |
| **Reuse** | 复用 | 复用 | 不算 | 复用 | 还省索引器的**全部算力** |

V4.1 的分配是：`kv_source_layers = [2, 8, 14, 20]`（只有 4 层产 KV），`index_source_layers = [2, 8, 14, 20, 24, 28, 32, 36]`（8 层产索引）。也就是 **4 个 Full + 4 个 Reindex + 32 个 Reuse**。三种模式都保留本层自己的 global Q 和 SWA KV——这两样不能共享。

### 6.3 索引器维：层次化候选池

Reindex 层虽然复用 KV，但仍要重算 top-k。V4.1 在 decoder 的第一个 Full 层（layer 20）扫全量，顺带做块级候选：每块取块内最大分数，选 2048 个块 × 每块 8 个位置 = **16384 个候选**。后续 Reindex 层只在这个池子里打分。

于是「每个 query 的评分位置数」从 $\Theta(N)$ 变成**常数 16384**（只有第一个 Full 层还扫全量）。代码里有个容易漏的正确性细节：最后一个没填满的块被强行钉住（分数设 $+\infty$），否则「最近但未填满的块」会被更旧的满块挤掉（`annotated/model.py:693-727`）。

### 6.4 CED：把 prefill 砍半

V4.1 还多了一个 V4 没有的东西：**Causal Encoder-Decoder**。decoder 的 global KV 不由本层隐状态产生，而由第 $L/2$ 层隐状态投影：

$$
C_l = H_{L/2} W_l^{KV}, \qquad l \gt L/2
$$

于是 prefill 时 decoder 那 20 层**根本不需要在整条 prompt 上跑**：

$$
\Theta\big(\underbrace{N \cdot L/2}_{\text{encoder}} + \underbrace{n_{win} \cdot L/2}_{\text{decoder SWA 重放}}\big) \approx \Theta(NL/2)
$$

这就是「prefill 激活 8B、decode 激活 16B」的来历——**同一个模型，prefill 只在前一半层上跑 prompt token**。代价是 SWA 需要「重放」：精确重建要处理 $n_{win} \times L/2$ 个 token；V4.1 的 SWA Bounded Replay 只重放最近 $n_{win}$ 个并截断，**这是近似**（论文承认 "not mathematically identical"），换来的是 SWA KV 不必落盘。

### 6.5 复算 890 B/token

V4.1 报告给的是 890 B/token。按 config 逐项复算：

| 项 | 条目字节 | 持有层 | 压缩 | 贡献 |
|---|---|---|---|---|
| 压缩 KV latent | 512 维 × 4 bit + 32 个 scale = **288 B** | 3 个 encoder source | $m=2$ | 432 B |
| 压缩 KV latent | 288 B | 1 个 decoder source | $m=1$ | 288 B |
| 索引器 K | 128 维 × 4 bit + 4 个 E8M0 scale = **68 B** | 3 个 encoder source | $m=2$ | 102 B |
| 索引器 K | 68 B | 1 个 decoder source | $m=1$ | 68 B |
| **合计** | | | | **890 B** |

复算和报告**逐字节吻合**。这条账还反证了两件事：(1) 报告的 "global KV" 包含索引器 K、不含 SWA 环形缓冲；(2) 跨层共享确实只有 4 层在存东西——若 40 层各存各的，按 $m=2/1$ 平均约 216 B/层，合计约 8.6 KB，差 10 倍。

配套代码 `budget.py` 里，`DeepSeek-V4.1-Flash` 的 KV 就是按这个式子算的，自测断言它等于 890。

## 7. 一张图看完四代

| | KV/token | 1M 上下文 KV | 每 token 注意力 FLOPs@1M | 主要代价 |
|---|---|---|---|---|
| Llama-3.1-405B（GQA-8） | 504 KiB | 541 GB | 8,659 GFLOP | KV 装不下 |
| DeepSeek-V3（MLA） | 68.6 KiB | 73.7 GB | 17,816 GFLOP | 单价 4.5× |
| DeepSeek-V3.2（DSA） | 68.6 KiB | 73.7 GB | 1,083 GFLOP（索引器占 97%） | 索引器 $O(N)$ |
| DeepSeek-V4-Flash（CSA+HCA） | 2.9 KiB | 3.1 GB | ≈26 GFLOP | 工程复杂度 |
| DeepSeek-V4.1（CSA2+CED+FP4） | **890 B** | **933 MB** | 3.2 + 5.4 GFLOP | 跨层依赖、训练要 shadow indexer |

两句话总结这条线：

1. **注意力压缩的每一步，都是把成本从一个科目挪到另一个科目**：GQA 挪给质量，MLA 挪给 FLOPs，DSA 挪给索引器，CSA2 挪给训练复杂度（跨层依赖要 shadow indexer、pipeline 要传中间态）。
2. **2026 年的胜负手不在 core attention，在索引器**。当 core 的成本已经与 $N$ 无关，剩下的 $O(N)$ 项就决定了长上下文的经济性——这也是为什么 V4.1 要用「4 层产 KV、8 层产索引、32 层复用」这种近乎抠门的结构。

## 8. 代码在哪

- **MLA 的最小实现**：`transformers/models/deepseek_v3/modeling_deepseek_v3.py:364-470`（`q_a_proj` / `kv_a_proj_with_mqa` / `kv_b_proj` 在 386-396 行，forward 在 416 行起）；
- **CSA2 + CED + 层次索引器**：`DeepSeek-V4.1-Flash/annotated/model.py`——`Compressor` 在 525 行，`Indexer` 在 589 行，`select_candidate_blocks` 在 693 行，`Attention` 在 730 行；
- **kernel 侧**：FlashMLA（`head_dim_k=576` 的由来）、FlashAttention 的 sparse 分支。

下一篇转向另一条路：既然注意力怎么压都还是要「存 KV」，那能不能干脆不存——**线性注意力与 3:1 混合**。

---

**系列导航**：

1. [导读：结构不是越新越好，是预算搬家]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. 本篇：注意力四代
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
