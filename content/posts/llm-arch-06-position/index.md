---
title: "LLM 模型结构变迁（六）：位置的退场——RoPE、部分 RoPE 与 NoPE"
date: 2026-11-02
draft: false
weight: 7
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "位置编码", "RoPE", "NoPE", "YaRN", "长上下文"]
categories: ["LLM算法"]
---

前面五篇改的是「算什么」和「存什么」。这一篇改一个更隐蔽的东西：**模型怎么知道谁在前谁在后**。

2026 年发生了一件很反常的事：Kimi-K3 的 24 层全局注意力**完全不用位置编码**（NoPE），却直接外推到 1M 上下文，不做任何 RoPE 缩放或插值；而 DeepSeek-V4.1 还在用 RoPE，但按理论口径它只能分辨约 **7.6K** 的距离——却宣称 1M。

这两个事实放在一起，指向同一个结论：**1M 上的「记住」不靠位置，靠内容寻址。** 位置编码没有被删掉，它是被**分层稀释**了。

## 0. 先记住一个画面

RoPE 是给每个 token 的 Q/K 装一个**转速不同的时钟**：第 $i$ 维的角速度是 $\theta_i = \text{base}^{-2i/d}$，位置 $m$ 就是「这个时钟走了 $m\theta_i$ 弧度」。两个 token 的内积只取决于**时差**，所以它是相对的。

问题在于：**训练时只见过「转过不超过训练长度那么多圈」的时钟**。到了 1M，低频维度的时钟早就转了几百圈，相位落在训练分布之外——模型没见过，自然外推不了。

> **类比的边界**：时钟的转速是**焊死**的（base 是超参），但你可以换时钟（改 base）、可以给时钟减速（位置插值）、也可以只留几个时钟（部分 RoPE）、甚至全部拆掉（NoPE）。这四种做法对应 2023-2026 年的全部位置编码工作。

## 1. RoPE 的软肋：训练没见过的相位

RoPE 按 2 维一组切成 $d/2$ 个复数，第 $j$ 组的波长是：

$$
\lambda_j = \frac{2\pi}{\theta_j} = 2\pi \cdot \text{base}^{2j/d}
$$

base=1e4、$d=128$ 时，$\lambda_0 \approx 6.28$（最快，管相邻 token），$\lambda_{63} \approx 5.4 \times 10^4$（最慢，管几万 token 的距离）。

**外推失败的根因是「critical dimension」**：只有那些在训练长度 $T_{\text{train}}$ 内**至少转完一圈**的维度被充分训练过：

$$
T_n = 2\pi \cdot \text{base}^{2n/d} \le T_{\text{train}}
\quad\Longrightarrow\quad
d_{\text{extra}} = 2\left\lfloor \frac{d}{2} \log_{\text{base}} \frac{T_{\text{train}}}{2\pi} \right\rfloor
$$

LLaMA-2（$d=128$、base=1e4、4K 训练）算出 $d_{\text{extra}} = 92$——**最后 36 维从来没转完过一圈**。它们在外推时的相位完全是 OOD 的。

更硬的版本来自 Theorem 1（arXiv:2405.14591）：定义

$$
B_{m,\theta} = \sum_{i \lt d/2} \cos(m\theta_i)
$$

$B_{m,\theta} \lt 0$ 之后，「相似 token 的注意力分数」会低于「随机 token 的分数」，注意力**失去分辨力**。于是可以算出每个配置的「最大可分辨距离」：

| 配置 | $d$ | base | 缩放 | $B_m$ 首次转负 |
|---|---|---|---|---|
| LLaMA-2 原始 | 128 | 1e4 | 无 | 1,707 |
| V4.1 的 SWA 层 | 64 | 1e4 | 无（YaRN 关闭） | 725 |
| V3 的解耦 $k^R$ | 64 | 1e4 | YaRN s=40 | 4,119 |
| **V4.1 的压缩层** | 64 | 1.6e5 | YaRN s=16 | **7,568** |

最后一行就是本文开头那个矛盾：**V4.1 的 RoPE 在 1M 上基本不提供可用的距离信号。** 第六节解释它靠什么。

## 2. 四种外推方法：改的是同一件事

统一记号 $f(x, m) = f(x, g(m), h(\theta))$：

| 方法 | $g(m)$ | $h(\theta)$ | 一句话 |
|---|---|---|---|
| PI（位置插值） | $s \cdot m$ | $\theta_d$ | 把所有位置等比例压缩，**高频也被压掉** |
| NTK-aware | $m$ | 换 base $b' = b \cdot s^{d/(d-2)}$ | 不改位置改基频，高频少动、低频多动 |
| NTK-by-parts | $m$ | 按波长分段插值 | $r \lt \alpha$ 全插值，$r \gt \beta$ 不动 |
| YaRN | $m$ | 同上（$\alpha=1, \beta=32$）+ **温度** | 分段插值 + 温度补偿 |
| LongRoPE | $m$ | 逐维搜 $s_d$，再逐位置段搜 | 非均匀 + progressive，2048K |

YaRN 的温度是 $t = 0.1 \ln s + 1$，实现上是 `cos/sin × √(1/t)`。物理意义：插值会让注意力分布变平、softmax 熵升高，用温度拉回来。各家都自己调过——DeepSeek-V2 用 $0.0707\ln s+1$，V3 改回 $0.1\ln s+1$。

**一个漂亮的恒等式**：YaRN 的 `beta_slow=1` 边界，就是 critical dimension 本身：

$$
\text{find\_correction\_dim}(1, d, \text{base}, L) = \frac{d}{2}\log_{\text{base}}\frac{L}{2\pi} = \frac{d_{\text{extra}}}{2}
$$

所以 `beta_fast`/`beta_slow` 不是调参玄学，是「波长 = 训练长度」的那个维度索引。V4.1 用 32/1，算出 32 对里只有 15 对被真正除以 16。

## 3. 部分 RoPE：64 维是「够用的固定成本」

MLA 只对 64 维做 RoPE，这件事经常被误解为「省算力」。真正的原因是**硬约束**：

MLA 的省带宽全靠把 $W^{UK}$ 吸收进 $W^Q$。但只要 $k$ 里带位置敏感成分（RoPE 旋转），$W^{UK}$ 就没法和位置无关地合并，推理时必须为所有前缀重算 K。所以**位置必须被隔离到一个单独的小通道**里。

那为什么是 64 维？看它的演化：

| 模型 | head_dim | 带 RoPE 的维度 |
|---|---|---|
| DeepSeek-V2 | 128 | 64（= $d_h/2$） |
| DeepSeek-V3 | 128 | 64 |
| DeepSeek-V4 | 512 | 64（= $d_h/8$） |
| Kimi-K3 | 128 | **0（NoPE）** |

head_dim 从 128 涨到 512，RoPE 维度**没动**。所以 64 不是「最优维度」，是**够用的固定成本**：位置部分只占 MLA KV 的 $64/(512+64) = 11\%$。

代价是部分 RoPE 天然更难外推。同样 base 下，$d=64$ 的可达距离比 $d=128$ 少 **2.3-4.3 倍**（见第一节表：base=1e4 时 725 vs 1707）。所以部分 RoPE 必须靠低频插值补回来。

还有一个反直觉的推论：**不带位置的维度不携带距离信息，但距离会通过 value 路径泄漏**。V4 就踩了这个坑——压缩 KV 同时当 K 和 V，value 加权求和后天然带绝对位置，所以要在 core attention 输出上再套一次 $\mathrm{RoPE}(-i)$ 抵消。论文原话："the naive core attention outputs will carry absolute position embeddings … we also apply RoPE with position $-i$."

## 4. 双 base：局部用 1e4，全局用 1.6e5

2026 年的标配是**每层用不同的 base**：

| 模型 | 全局 base | 局部 base | 局部窗口 |
|---|---|---|---|
| Gemma-3 | 1e6 | 1e4 | 4096（5 local : 1 global） |
| ModernBERT | 1.6e5 | 1e4 | — |
| DeepSeek-V4/V4.1 | 1.6e5（压缩层） | 1e4（SWA 层，YaRN 关闭） | 128 |

规律很简单：**base 只需覆盖这一层真实的可达距离**。

- SWA 层只看 128 个 token，base=1e4 的 $B_m$ 到 725 才转负，绰绰有余；
- 压缩层要看整个 1M，base 必须加到 1.6e5，再配 YaRN×16。

**这不是 DeepSeek 的发明**——Gemma-3 和 ModernBERT 早就在做。V4.1 的 160000 和 ModernBERT 撞车是巧合，但逻辑一致。

还有一个结构性细节：压缩 latent 的 RoPE 位置用的是「块首原始位置」，**步长等于压缩比 $m$**。所以相位推进是 $m$ 倍，必须换更大的 base——这就是 `compress_rope_theta` 这个独立超参存在的原因。

## 5. NoPE：把位置交给线性层

NoPE 不是新东西。arXiv:2305.19466 在 2023 年就证明：不加位置编码、只靠因果 mask，长度泛化**优于** APE / ALiBi / Rotary / T5 relative bias。原因是一层注意力可以表示绝对位置，两层可以表示相对位置——位置信息能从 mask 结构里自己学出来。

但 NoPE 有个前提：**模型要足够大**。arXiv:2609.11913 的消融显示，小模型上 NoPE 明显更差，得大到一定规模才能自己编码位置。这也是为什么 NoPE 在 2023 年只是论文里的好奇心，到 2026 年才进生产。

Kimi-K3 的做法更彻底：**24 层全局 MLA 全部 NoPE**，位置信息由 69 层 KDA 提供。KDA 的逐通道 decay $\alpha \in [0,1]^{d_k}$ 天然带「近处记得牢、远处忘得快」的归纳偏置——**这就是位置感**。

K3 报告的原话："Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA. As a result, the model extrapolates directly to 1M-token contexts without any positional-encoding modification, such as RoPE rescaling or interpolation."

这句话值得逐字读：**它把「外推」这件事从「改位置编码」变成了「不需要改」**。这是结构层面的胜利，不是调参的胜利。

## 6. 那 1M 的召回靠什么？

回到第一节的矛盾：V4.1 的 RoPE 只能分辨 7.6K，怎么做到 1M？

答案是三件事，没有一件是位置编码：

1. **content-addressed 选择**。Top-512 由索引器打分（FP4、32 头、128 维），候选 16,384 个位置——**选谁完全由内容决定，与距离无关**。只要内容匹配，100 万 token 外也能被选中。
2. **压缩把位置网格变粗**。一个 latent 代表 $m$ 个 token，细粒度位置在压缩里就丢了——但这没关系，因为压缩层负责的是「全局语义」，不是「精确位置」。
3. **局部顺序交给 SWA**。128 窗口的 SWA 层用 base=1e4，专门管「刚才说了什么」。位置精度需求被局部层吃掉，全局层不需要。

再加一条「减法」证据：V4 的 CSA 压缩器里有可学习的绝对位置偏置 $B \in \mathbb{R}^{m \times c}$，**CSA2 明确把它删了**（"removes this overlap and absolute positional embedding"）。2026 年最直接的一次位置减法。

所以「位置退场」的完整图景是**分层**的：

| 层型 | 位置编码 | 负责什么 |
|---|---|---|
| SWA 层（128 窗口） | RoPE base=1e4 | 精确的局部顺序 |
| 压缩/全局层 | RoPE base=1.6e5，只 64 维 | 粗粒度的相对位置 |
| 线性层（KDA/GDN） | 无（decay 隐式） | 近因偏置 |
| 索引器 | 无 | 纯内容寻址 |

**位置没有被删掉，它被拆成了四份，分别由最便宜的那个组件承担。**

## 7. 五个坑

**坑一：`beta_fast` 调大更保守？反了。** `beta_fast ↑` → `low ↓` → **更多维度被插值**，局部顺序破坏更多。V4.1 用 32/1 与 YaRN 给 LLaMA 的推荐一致。

**坑二：`mscale` 要成对传。** $\text{get\_mscale}(s) = 0.1 \cdot \text{mscale} \cdot \ln s + 1$；DeepSeek 传 `mscale = mscale_all_dim = 1` 使 `attention_factor = 1`。只传一个，温度会偷偷变成 $0.1\ln 16 + 1 = 1.277$——等于改了 attention scale 而不自知。

**坑三：`original_max_position_embeddings` ≠ `max_position_embeddings`。** V4.1 是 65536 vs 1048576。写成 1M，YaRN 的 low/high 会算到 32 对之外，整层退化成「什么都不插值」。

**坑四：packed sequence 的 position id 必须逐样本从 0 开始。** 若一路累加，第二条样本第一个 token 会带「距上一条结尾 N 个 token」的相位 = 注入跨文档距离。压缩层更敏感：块边界跟着 position id 走，不 reset 会把两个不相关文档的 token 池化到一起。

**坑五：动态 NTK 会破坏 prefix cache。** 每次 forward 按当前 seq_len 重算 `inv_freq`，同一 prefix 在不同 batch 里 KV 不一致，prefix caching 和 CUDA graph 全部失效。生产上几乎没人用。

## 8. 小结

| 年代 | 位置方案 | 一句话 |
|---|---|---|
| 2017-2020 | 正弦 APE | 绝对位置，不可外推 |
| 2021-2023 | RoPE + PI/NTK/YaRN | 相对位置，靠改频率外推 |
| 2024-2025 | 部分 RoPE（64 维） | 为 MLA 的吸收让路 |
| 2026 | 双 base + NoPE 混合 | 位置按层分工，全局层不需要 |

一句话总结：**位置编码的演化，是从「一个全局超参」变成「每层的一个设计选择」。** 当全局注意力只剩 1/4 层、其余都是带 decay 的线性层时，「给所有 token 装同一个时钟」这件事本身就不成立了。

下一篇讲最后一个方向：记忆——把参数花在「查」而不是「算」。

---

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
5. [MoE 十年]({{< relref "llm-arch-04-moe" >}})
6. [残差流的加宽：HC 与 mHC]({{< relref "llm-arch-05-residual" >}})
7. 本篇：位置的退场
8. [条件记忆]({{< relref "llm-arch-07-memory" >}})
9. [五个收敛、三个分叉与三个赌注]({{< relref "llm-arch-08-trends" >}})
10. [残差连接与 AttnRes]({{< relref "llm-arch-09-attnres" >}})
