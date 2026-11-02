---
title: "LLM 模型结构变迁（四）：MoE 十年——从 8 专家到 896 专家"
date: 2026-11-02
draft: false
weight: 5
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "MoE", "路由", "专家并行", "LatentMoE"]
categories: ["LLM算法"]
---

前两篇都在「序列」这一维上做减法：少存 KV、少看位置。这一篇换一维——**宽度**。

宽度的核心矛盾是：模型容量（参数）和推理成本（FLOPs）在稠密模型里是绑死的，想更聪明就得更贵。MoE 把它们解开了：Kimi-K3 有 2.78T 参数，每个 token 只激活 104B；DeepSeek-V4.1 有 748B 参数，decode 只激活 16B。

代价是什么？这一篇把 MoE 十年的路由数学、配置考古和工程账单一次算清。

## 0. 先记住一个画面

稠密 FFN 是**一家所有人都必须上的食堂**：不管你想吃什么，整个厨房都得开火。

MoE 是**一条美食街**：门口有个门卫（router），每个 token 进来，门卫挑 8 家店（top-8），只开这 8 家的火。总店数可以无限加，但每个 token 只付 8 家的钱。

门卫的麻烦在于：**他得保证 384 家店都有人去**。如果所有 token 都挤向 3 家网红店，剩下的店就废了——参数白存，算力还浪费在排队上。MoE 十年的历史，一大半是在解决门卫的排班问题。

> **类比的边界**：真实的美食街可以把冷门店关掉止损；MoE 不行——专家权重必须常驻显存，哪怕一个月没接过客。所以 MoE **不省显存，只省算力**，这是它和「稀疏化」最容易被混淆的地方。

## 1. 四次瓶颈迁移

| 阶段 | 代表 | 瓶颈 | 解法 |
|---|---|---|---|
| 2017-2021 | GShard、Switch | 负载不均导致丢 token | 容量因子 + aux loss（top-1/top-2） |
| 2022 | Expert Choice | 均衡 vs 因果性 | 专家选 token（推理不可用，只借思想） |
| 2024 | DeepSeekMoE | 专家太粗、公共知识重复学 | 细粒度切分 + 共享专家 |
| 2024.12 | DeepSeek-V3 | aux loss 与主任务冲突 | aux-loss-free（bias 只影响选谁） |
| 2026 | V4 / K3 | 通信与显存 | 去 node-limit + hash 路由 + LatentMoE |

**GShard**（arXiv:2006.16668）确立了两件事：group-level top-2 路由，和「容量因子」——每个专家最多收多少 token，超了就丢。Switch（arXiv:2101.03961）把路由砍到 top-1，把容量因子做成标准件，论文说丢弃率 "typically < 1%"。

**Expert Choice**（arXiv:2202.09368）是个漂亮的反转：让专家选 token，负载天然完美均衡，连 aux loss 都不要了。但它**在自回归推理时不可用**——一个 token 的专家集合依赖同 batch 里其它 token，因果性被打破。工业界只借了它的思想（「均衡可以靠分配机制，而不是损失函数」），没借形式。

**DeepSeekMoE**（arXiv:2401.06066）贡献两个结构性发明，今天所有开源 MoE 都在用：

- **细粒度切分**：把一个大 FFN 切成 $m$ 份、激活其中 $mK$ 份。组合数从 $C(N,K)$ 变成 $C(mN,mK)$，表达能力涨而算力不变；
- **共享专家隔离**：$K_s$ 个专家对所有 token 恒开，吸收公共知识，让路由专家专心学「差异化」的部分。

## 2. 路由数学：三种打分，三种归一化

记 token 表示 $x \in \mathbb{R}^d$，专家打分矩阵 $W \in \mathbb{R}^{E \times d}$，logits $z = Wx$。

### 2.1 三种打分函数

**softmax（2017-2024 默认）**

$$
s_i = \frac{e^{z_i}}{\sum_j e^{z_j}}, \qquad y = \sum_{i \in \mathrm{TopK}(s,k)} \frac{s_i}{\sum_{j\in\mathrm{TopK}} s_j} E_i(x)
$$

问题是它是**零和**的：抬高一个专家必然压低另一个。$E$ 从 256 涨到 384 时，单个专家的概率被摊薄到 $1/E$ 量级，top-k 的绝对分数越来越小，路由对 logits 的微小扰动越来越敏感。

**sigmoid（V3 起）**

$$
s_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

每个专家独立判断「我该不该接」，没有零和约束。MiniMax-M2 的论文把理由写得最直白："removing the zero-sum constraint imposed by softmax. This allows multiple experts to be activated simultaneously with high confidence."

代价是权重不再天然归一，必须显式处理：V3 在 top-k 内重新归一化再乘 `routed_scaling_factor=2.5`，Qwen3-MoE 则干脆不归一。

**sqrtsoftplus（V4，2026）**

$$
s_i = \sqrt{\mathrm{softplus}(z_i)} = \sqrt{\ln(1 + e^{z_i})}
$$

论文只有一句话，没有消融、没有解释。从函数形状能说的只有：值域 $[0,\infty)$，$z \to -\infty$ 时按 $\sqrt{e^z}$ 衰减（比 sigmoid 的 $e^z$ 慢，尾部更厚），$z \to +\infty$ 时按 $\sqrt{z}$ 增长（不饱和）。工程上它仍要算 exp——**别把 V4 §3.1 那句「用无指数运算的激活替代 SwiGLU」当成 sqrtsoftplus 的动机**，那句话讲的是 FFN 激活，不是路由。

### 2.2 归一化落在哪，数值上不等价

| 落点 | 公式 | 代表 | 后果 |
|---|---|---|---|
| pre-softmax | $p=\mathrm{softmax}(z)$，再 TopK | Switch、GLM-5 bridge | 概率先被 $E$ 摊薄，$E$ 大时 top-k 差异被压缩 |
| post-topk | TopK$(z)$，再对选中项 softmax | gpt-oss | 权重恒和为 1，与 $E$ 无关 |
| post-topk + 缩放 | TopK$(z)$，$s_{\text{sel}}/\sum s_{\text{sel}} \times c$ | V3/V4（$c=2.5/1.5$） | 保留「绝对置信度」，$c$ 是必调超参 |

Megatron 里有个真实的配置陷阱：`moe_router_topk_scaling_factor` 的注释写「只在 pre_softmax 时生效」，而 V3/V4 恰恰是 `pre_softmax=False` + `scaling_factor≠None`——bridge 里是在**绕过注释语义**用它。

### 2.3 aux loss 与 aux-loss-free

经典 aux loss（Switch Eq.4）：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i
$$

$f_i$ 是分到专家 $i$ 的 token 比例，$P_i$ 是专家 $i$ 分到的概率质量。它的问题是：**它和语言建模目标是竞争关系**，$\alpha$ 大了掉点，小了不均衡。

V3 的 aux-loss-free 换了个思路：给每个专家一个偏置 $b_i$，**只参与「选谁」，不参与「权重多大」**：

$$
s'_i = s_i + b_i, \qquad b_i \leftarrow b_i + \gamma \cdot \mathrm{sign}(\bar{f}_i - f_i)
$$

选完之后，权重仍用**未加偏置**的 $s_i$ 归一化。这样均衡不影响梯度，只影响路由。

但注意一个被广泛误传的点：**aux-loss-free 不是「没有 aux loss」**。V3 论文同时保留了一个 $10^{-4}$ 的 sequence-wise balance loss，V4 也保留（§4.2.2）。本仓库 7 个 Megatron-Bridge 配置里 5 个仍开 aux（K2/GLM-5/M2 是 $10^{-3}$），只有 gpt-oss 真的关了。

为什么必须兜底？因为 `sign()` 丢掉了不均衡的**幅度**——超载 2 倍和 10 倍，修正量一样。收敛慢，且与失衡程度无关。这才是 V4 加回 $10^{-4}$ 的真实原因，不只是「防单序列极端」。

## 3. 配置考古表

| 模型 | 年 | 总参 | 激活 | routed E | top-k | 共享 | 专家 FFN | 打分 | 均衡手段 |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeekMoE-145B | 2024 | 144.6B | 22.2B | 128 | 12 | 4 | 0.125× | softmax | expert + device aux |
| OLMoE-1B-7B | 2024 | 6.9B | 1.3B | 64 | 8 | 0 | 1024 | softmax | aux + z-loss，dropless |
| DeepSeek-V3 | 2024 | 671B | 37B | 256 | 8 | 1 | 2048 | sigmoid | bias $\gamma{=}10^{-3}$ + seq $10^{-4}$ |
| Qwen3-235B-A22B | 2025 | 235B | 22B | 128 | 8 | 0 | 未披露 | softmax | global-batch balance |
| Kimi K2 | 2025 | 1.04T | 32.6B | 384 | 8 | 1 | 2048 | sigmoid | seq $10^{-3}$ + bias |
| gpt-oss-120b | 2025 | 116.8B | 5.1B | 128 | 4 | 0 | 2880 | softmax（top-k 内） | **none** + MXFP4 |
| Llama 4 Maverick | 2025 | 400B | 17B | 128 | 1 | 1 | 8192 | softmax | — |
| Qwen3-Next-80B | 2025 | 80B | 3B | 512 | 10 | 1 | 512 | softmax | aux $10^{-3}$ |
| GLM-5 | 2026 | 744B | 40B | 256 | 8 | 1 | 2048 | sigmoid | seq aux + bias |
| MiniMax-M2 | 2026 | 229.9B | 9.8B | 256 | 8 | 0 | ~1536（反推） | sigmoid | aux $10^{-3}$ + 可学习 bias |
| DeepSeek-V4-Pro | 2026 | 1.6T | 49B | 384 | 6 | 1 | 3072 | sqrtsoftplus | bias + seq $10^{-4}$ + 前 3 层 hash |
| **Kimi-K3** | 2026 | 2.78T | 104B | **896** | **16** | 2 | 3072（latent 3584） | sigmoid | seq aux + bias |

读表有两条主线：

**第一，激活比在降，但专家没有一直变小。** V3 的 5.5% → K2 的 3.1% → V4-Pro 的 3.1% → K3 的 3.7%。但 expert FFN 从 2048（V3）涨到 3072（V4-Pro/K3）——**「多而小」在 2026 年已经反转**。真正变小的是**每专家每 step 的 token 数**（决定 grouped GEMM 的 M 维），而不是专家本身的宽度。

**第二，Kimi-K3 的 896 专家 + top-16 是新台阶。** 稀疏度 $896/16 = 56$，比 K2 的 48、V3 的 32 都高。做到这个规模靠的是 **LatentMoE**：专家不在 7168 维的 hidden 空间工作，而在 3584 维（$d/2$）的潜空间：

$$
P_{\text{expert}} = 3 \ell d_{\text{ff}} \quad(\ell = 3584),\qquad \text{通信量} \propto \ell \text{ 而不是 } d
$$

单专家参数减半、all-to-all 字节减半，于是「专家数翻倍」在系统上变得可行。代价是多了一次升维/降维投影，以及潜空间表达力的损失（论文用更大的 $d_{\text{ff}}=3072$ 补回来）。

但 K3 的报告里有个更重要的细节：**这个规模下 vanilla LatentMoE 会坏掉**，坏在两个地方（§2.3）：

1. **激活爆炸**。路由路径是 $W^\downarrow \to$ 专家 FFN $\to W^\uparrow$，串了将近四个矩阵乘；这种病态结构在 2.8T 参数下会把内部激活放大到发散。
2. **均衡失效**。896 个专家已经超出「aux-loss-free 的 bias 更新还能好好工作」的范围——bias 是 sign-based 的，专家一多，每步的修正信号就被稀释。

K3 的解法叫 **Stable LatentMoE**，三个组件：

- **上投影前加 RMSNorm**：把 $u$ 的尺度拉回来；
- **SiTU-GLU**：把 SwiGLU 的两支都套上有界 tanh，$\beta_1 \tanh(x/\beta_1) \cdot \sigma(x)\beta_2\tanh(x/\beta_2)$，界是 $\beta_1\beta_2 = 100$（K3 取 $\beta_1=4, \beta_2=25$），近原点处与 SwiGLU 几乎重合，大正输入处饱和；
- **Quantile Balancing（QB）**：用分位数而不是 bias 的符号来做负载均衡。

这三条里，**SiTU-GLU 是最值得记住的**：它不是「更好的激活函数」，而是**给 2.8T 参数的不稳定性买的一份保险**。2026 年之后，MoE 论文里「有界激活」会越来越多——原因不在效果，在可训练性。

## 4. 工程账单：all-to-all 与 EP

MoE 的算力是省的，通信是加的。每层每 token 的 all-to-all 字节数：

$$
\text{Bytes}_{\text{a2a}} \approx 2 \cdot k \cdot d \cdot b \quad(\text{dispatch} + \text{combine})
$$

V3 配置：$k=8, d=7168, b=2$ → 每 token 每层约 229 KB。61 层、每 token 61 次往返——**这就是为什么 MoE 训练必须靠 EP + 大 batch，单卡根本喂不饱**。

几个实战结论：

1. **EP 大小由「每卡专家数 × 单专家字节」决定**。V4-Pro 单专家 $3 \times 3072 \times 7168 \times 2\text{B} \approx 132$ MB，EP=8 时每卡 48 个专家 = 6.3 GB/层——这是它必须 EP 拉大的原因。
2. **共享专家是复制的，不能 EP 切**。它的 FLOPs 等价一个稠密 FFN（V3 占 routed 部分的 12.5%，V4-Flash top-6 下升到 16.7%）——省的是路由决策，不是显存或通信。
3. **group-limited routing 是「用路由自由度换通信确定性」**。V3 限制每个 token 最多去 4 个节点；V4 直接删了这条约束，因为它用 fused EP 把 dispatch/combine 和两个 GEMM 流水线化，靠 overlap 藏通信，而不是靠「少发节点」。

## 5. 三个误解

**误解一：MoE 省显存。** 不省。总参就是显存，2.78T 的 bf16 是 5.5 TB。MoE 省的是**算力**和**每 token 的权重读取量**。

**误解二：共享专家免费。** 它每个 token 都算，FLOPs 等价一个 dense FFN。它省的是「路由专家不用重复学公共知识」，不是算力。

**误解三：aux-loss-free 没有超参。** 它有 $\gamma$（bias 更新步长）、有兜底 loss 系数、有 `norm_topk_prob`、有 scaling factor。只是把「调 $\alpha$」换成了「调 $\gamma$ + 兜底」。

## 6. 代码：V3 的门卫只有 20 行

`transformers/models/deepseek_v3/modeling_deepseek_v3.py:214-236`：

```python
def route_tokens_to_experts(self, router_logits):
    router_logits = router_logits.sigmoid()                       # ① sigmoid 打分
    router_logits_for_choice = router_logits + self.gate.e_score_correction_bias  # ② bias 只进"选谁"
    group_scores = (router_logits_for_choice
        .view(-1, self.n_group, self.n_routed_experts // self.n_group)
        .topk(2, dim=-1)[0].sum(dim=-1))                          # ③ 组内 top-2 之和
    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1)[1]
    ...
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask, -inf)
    topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)[1]
    topk_weights = router_logits.gather(1, topk_indices)          # ④ 权重取未加 bias 的分数
    if self.norm_topk_prob:
        topk_weights /= topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_weights * self.routed_scaling_factor
```

三个细节值得记：

- **`router_logits` 和 `router_logits_for_choice` 是两个变量**——bias 只影响「选谁」，不影响「权重多大」，这是 aux-loss-free 能成立的全部秘密；
- **组内 top-2 之和**而不是最大值：让「一个组里有两个中等专家」和「一个组里有一个超强专家」得到相近评价，避免整组被单个专家绑架；
- **`masked_fill(-inf)` 之后再 topk**：只在选中的 4 组（128 个专家）里挑 8 个。

Kimi-K3 的 LatentMoE 在代码里对应的是「先降维到 3584 → 路由 → 专家计算 → 升维回去」，`transformers` 里目前还没有 K3 的实现，要读只能看 Kimi 官方仓库或 Megatron-Bridge 的 recipe。

## 7. 小结

MoE 这十年的账，可以压成一句话：**用通信和显存，换「参数/算力」的解耦**。

- 2017-2021：换容量，代价是丢 token；
- 2022：换均衡，代价是因果性（失败）；
- 2024：换表达能力，代价是路由复杂度（细粒度 + 共享专家）；
- 2026：换到 896 专家，代价是通信（LatentMoE 把通信减半来买单）。

下一篇转向深度方向：残差连接这个十年没动过的东西，为什么在 2025 年底被重新设计。

---

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. [先把账本立起来]({{< relref "llm-arch-01-budget" >}})
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
5. 本篇：MoE 十年
6. [残差流的加宽：HC、mHC 与 AttnRes]({{< relref "llm-arch-05-residual" >}})
