---
title: "LLM 模型结构变迁（一）：先把账本立起来——参数、算力、KV 三张预算表"
date: 2026-11-02
draft: false
weight: 2
series: ["llm-arch"]
tags: ["LLM算法", "系列", "模型结构", "KV-cache", "MoE", "MLA", "scaling"]
categories: ["LLM算法"]
---

这个专题要回答一个问题：**2023 到 2026 年，LLM 的结构到底改了什么，为什么这么改。**

在动手翻论文之前，先立一本账。因为过去三年几乎所有结构创新，最后都能折算成三个数字：

1. **参数量** $P$：决定容量，也决定权重占多少显存；
2. **每 token 的 FLOPs**：决定训练/推理要多少算力；
3. **每 token 的 KV 字节数**：决定长上下文能开多长、decode 能跑多快。

一个结构方案好不好，不看它名字新不新，看它**在这三个数字之间怎么搬家**：把参数从「计算」搬到「记忆」（Engram），把 KV 从「每层存」搬到「跨层共享」（CSA2），把序列混合从「$O(N)$ 的注意力」搬到「$O(1)$ 的状态」（KDA）。搬完之后，总有一项变贵——账本就是用来找那一项的。

本篇是方法论篇：给出三个数字的闭式公式，用 11 个真实模型复算校验（误差 ≤1.3%），并给出配套代码 [budget.py](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/llm-arch-evolution/budget.py)。后面每一篇讲一个方向时，都会回到这本账。

## 0. 先记住一个画面

把模型想成一个**有三个仓库的工厂**：

- **参数仓**（HBM）：原料堆在这里，MoE 和 Engram 决定每次只搬哪几垛；
- **算力车间**（SM）：矩阵乘在这里发生，FLOPs 是工时；
- **KV 货架**（HBM/SSD）：每个 token 的中间结果存在这里，decode 时每一步都要回来取。

decode 的每一步，工厂只做很少的活（一个 token 的矩阵乘），却要把**整个参数仓里用到的那部分**和**整个 KV 货架上这个请求的货**搬到车间。所以 decode 从来不是算力瓶颈，是**搬运瓶颈**。

> **类比的边界**：真实工厂可以加叉车、加仓库；GPU 的 HBM 带宽（H100 约 3.35 TB/s、B200 约 8 TB/s）和容量（80/141/192 GB）是焊死的。所以「少搬」比「多算」值钱——这就是为什么 2024 年之后的结构创新，一半以上是在打 KV 的主意。

## 1. 参数量：先分清 $P_{\text{total}}$ 和 $P_{\text{active}}$

稠密模型两者相等。MoE 之后必须分开：

$$
P_{\text{total}} = \underbrace{V d}_{\text{embedding}} + L \cdot P_{\text{attn}} + L \cdot P_{\text{ffn}} + P_{\text{extra}}
$$

$$
P_{\text{active}} = V d + L \cdot P_{\text{attn}} + L \cdot \big(k \cdot P_{\text{expert}} + s \cdot P_{\text{shared}}\big) + V d
$$

其中 $k$ 是 top-k，$s$ 是共享专家数。注意最后那个 $Vd$ 是输出 head——**它每 token 都要全量算**，在小模型上能占到激活参数的 20% 以上。

三种 FFN 的参数量：

| 结构 | 每层参数 | 说明 |
|---|---|---|
| 稠密 SwiGLU | $3 d \, d_{\text{ff}}$ | gate/up/down 三个矩阵 |
| MoE（全宽专家） | $n_r \cdot 3 d \, d_{\text{ff}} + n_s \cdot 3 d \, d_{\text{ff}}$ | $n_r$ 个路由专家全宽 |
| LatentMoE | $n_r \cdot 3 \ell \, d_{\text{ff}} + n_s \cdot 3 d \, d_{\text{ff}}$ | 路由专家在 $\ell$ 维潜空间工作（Kimi-K3：$\ell = 3584 = d/2$） |

LatentMoE 是 2026 年的新东西：专家数从 384 涨到 896，但每个专家的输入宽度减半，**总参数涨 2.7 倍、激活参数只涨 3.2 倍**，通信量按 $\ell$ 而不是 $d$ 计。这笔账在第（四）篇展开。

三种注意力的参数量（$d$ 为 hidden）：

| 结构 | 每层参数 | 每 token KV 条目维度 |
|---|---|---|
| GQA | $d n_h h + 2 d n_{kv} h + n_h h d$ | $2 n_{kv} h$ |
| MLA | $d r_q + r_q n_h (h_{nope}+h_{rope}) + d(r_{kv}+h_{rope}) + r_{kv} n_h (h_{nope}+h_v) + n_h h_v d$ | $r_{kv} + h_{rope}$ |
| 共享 KV（V4 式 MQA） | $d r_q + r_q n_h h + d h$ + 分组输出投影 | $h$（K=V=一条 latent） |

MLA 的妙处是把「每层 KV 条目维度」从 $2 n_{kv} h$ 压到 $r_{kv} + h_{rope}$：DeepSeek-V3 是 $512+64=576$，而同样 128 头的 GQA8 是 $2 \times 8 \times 128 = 2048$。

## 2. FLOPs：prefill 和 decode 是两种动物

前向每 token 的矩阵乘 FLOPs 约等于 $2 P_{\text{active}}$（一次乘加算 2 FLOPs）。注意力另算：

$$
\text{FLOPs}_{\text{attn}} = L \cdot 2 n_h (h_{qk} + h_v) \cdot n_{\text{entries}}
$$

- **稠密注意力**：$n_{\text{entries}} = N$（上下文长度），所以 prefill 的注意力成本是 $O(N^2)$、每 token 是 $O(N)$；
- **稀疏注意力**：$n_{\text{entries}} = k + w$（top-k + 滑窗），与 $N$ 脱钩；
- **线性注意力**：$n_{\text{entries}} = 0$，但每层多一次状态读写，约 $2 n_h d_k d_v$。

除此之外，稀疏注意力还有**索引器**的固定开销：每个 query 要给可见条目打分，$\text{FLOPs}_{\text{idx}} = L_{\text{idx}} \cdot 2 n_I c_I \cdot N$。这是「稀疏」的隐性税——第（二）篇会算，DeepSeek-V4.1 在 1M 上下文下，8 个索引层的扫描成本仍然能到每 token 几十 GFLOP。

## 3. KV：三个维度相乘

这是过去三年变化最大的一项。KV 常驻容量可以写成三个因子的乘积：

$$
\text{KV}_{\text{bytes/token}} = \underbrace{\frac{\text{条目字节}}{\text{entry size}}}_{\text{每个条目多少字节}} \times \underbrace{\frac{1}{m}}_{\text{序列维压缩}} \times \underbrace{\frac{1}{g}}_{\text{层维共享}}
$$

- **条目维度**：MHA → GQA（$n_{kv}$ 从 $n_h$ 降到 8）→ MLA（$2 n_{kv} h$ 降到 $r_{kv}+h_{rope}$）→ FP8/FP4 量化；
- **序列维度**：每 $m$ 个 token 压成一个条目（V4 的 CSA：$m=4$；HCA：$m=128$）；
- **层维度**：若干层共享同一份 KV（V4.1 的 CSA2：3 个编码器 source + 1 个解码器 source）。

还有第四个维度容易被忽略：**精度**。DeepSeek-V4 的 KV 用「64 维 RoPE 存 BF16 + 其余存 FP8」，等效 9 bit/维；V4.1 把主 KV 全压到 FP4（E2M1 + 每 16 通道一个 E4M3 scale），等效 4.5 bit/维。

一个必须区分的概念：**容量**和**每步带宽**。稠密注意力里两者相等（每步把整个 KV 读一遍）；稀疏注意力把两者解耦了——容量还是 $O(N)$，但每步只读 top-k 个条目。这个区别在第（二）篇会展开，它直接决定了长上下文 decode 的瓶颈从 KV 搬回权重。

## 4. 账本：11 个模型的实测表

下面两张表由 `budget.py` 生成（`python budget.py --ctx 131072` / `--ctx 1048576`），权重按 BF16 计，KV 按各模型实际精度。

**128K 上下文：**

| 模型 | 总参数 | 激活参数 | KV B/token | KV@ctx | prefill FLOPs/token | decode 读字节/token | 算术强度 |
|---|---|---|---|---|---|---|---|
| Llama-3.1-405B | 405.8B | 405.8B | 516096 | 67.6GB | 1.89T | 879.3GB | 2.14 |
| Qwen3-32B | 32.8B | 32.8B | 262144 | 34.4GB | 337.3B | 99.9GB | 3.38 |
| Qwen3-30B-A3B | 30.5B | 3.3B | 98304 | 12.9GB | 108.5B | 19.6GB | 5.55 |
| DeepSeek-V3 | 670.9B | 37.4B | 70272 | 9.2GB | 595.2B | 84.1GB | 7.08 |
| DeepSeek-V4-Flash | 283.8B | 13.2B | 3310 | 433.8MB | 70.9B | 26.5GB | 2.68 |
| DeepSeek-V4.1-Flash | 747.4B | 16.0B | 890 | 116.7MB | 17.9B | 32.0GB | 0.56 |
| Kimi-K2.6 | 1.03T | 32.7B | 70272 | 9.2GB | 322.7B | 74.6GB | 4.33 |
| Kimi-K3 | 2.77T | 97.9B | 24576 | 3.2GB | 346.0B | 199.1GB | 1.74 |
| GLM-5.2 | 743.1B | 40.9B | 89856 | 11.8GB | 107.4B | 82.1GB | 1.31 |
| gpt-oss-120b | 116.8B | 5.7B | 73728 | 9.7GB | 86.4B | 21.1GB | 4.10 |
| ERNIE-4.5-300B-A47B | 299.5B | 47.9B | 221184 | 29.0GB | 324.3B | 124.8GB | 2.60 |

**1M 上下文（同一批模型）：**

| 模型 | KV@1M | prefill FLOPs/token | decode 读字节/token | 算术强度 |
|---|---|---|---|---|
| Llama-3.1-405B | 541.2GB | 9.46T | 1.4TB | 6.99 |
| Qwen3-32B | 274.9GB | 2.26T | 340.4GB | 6.64 |
| Qwen3-30B-A3B | 103.1GB | 830.1B | 109.8GB | 7.56 |
| DeepSeek-V3 | 73.7GB | 4.26T | 148.6GB | 28.69 |
| DeepSeek-V4-Flash | 3.5GB | 371.6B | 26.5GB | 14.03 |
| DeepSeek-V4.1-Flash | 933.2MB | 26.3B | 32.0GB | 0.82 |
| Kimi-K2.6 | 73.7GB | 2.16T | 139.1GB | 15.51 |
| Kimi-K3 | 25.8GB | 1.43T | 221.6GB | 6.45 |
| GLM-5.2 | 94.2GB | 257.7B | 82.1GB | 3.14 |
| gpt-oss-120b | 77.3GB | 627.6B | 88.7GB | 7.07 |
| ERNIE-4.5-300B-A47B | 231.9GB | 1.95T | 327.7GB | 5.94 |

读表有三个要点：

**第一，参数和激活分了家。** Kimi-K3 有 2.77T 参数，但每 token 只激活 97.9B——比值 28×。DeepSeek-V4.1 更极端：总参数 747B（含 196B Engram），decode 只激活 16B，**比值 46×**。这让「模型容量」和「推理成本」第一次真正解耦。

**第二，KV 的差距是三个数量级。** 同样是 1M 上下文，Llama-3.1-405B 要 541GB KV（单请求，超过任何单卡显存），DeepSeek-V4.1 只要 933MB。**580 倍**。这不是靠一个技巧，是条目维度（576 vs 2048）、序列维度（$m=2$）、层维度（4 个 source / 40 层）、精度（FP4 vs BF16）四个因子相乘。

**第三，稀疏之后，decode 从 KV-bound 变回 weight-bound。** 注意 V4.1 在 1M 上下文下，算术强度只有 0.82——意思是每从 HBM 读 1 字节，只做 0.82 次浮点运算。看起来还是严重带宽受限，但原因变了：32GB 的读取里，KV 只占 7MB，**其余全是权重**（16B 激活参数 × 2 字节）。所以对 V4.1 这种模型，长上下文 decode 的优化重点不是 KV，而是权重的低精度和 MoE 的稀疏激活。

> 这里有个容易被忽略的换算：H100 的 HBM 带宽约 3.35 TB/s。decode 每步读 32GB，理论下限就是 9.5 ms/token——约 105 token/s，且与 batch 无关（batch 内可摊销权重）。这就是「单请求 decode 速度」的物理上限。

## 5. 复算校验：公式对不对，拿论文对一遍

结构公式最怕「看起来对」。配套代码用 9 个模型的公布参数量做对拍：

```
[PASS] Llama-3.1-405B      复算   405.8B  vs 公布   405.0B  ( 0.2%)
[PASS] Qwen3-32B           复算    32.8B  vs 公布    32.8B  ( 0.1%)
[PASS] Qwen3-30B-A3B       复算    30.5B  vs 公布    30.5B  ( 0.1%)
[PASS] DeepSeek-V3         复算   670.9B  vs 公布   671.0B  ( 0.0%)
[PASS] DeepSeek-V4-Flash   复算   283.8B  vs 公布   284.0B  ( 0.1%)
[PASS] DeepSeek-V4.1-Flash 复算   551.4B  vs 公布   552.0B  ( 0.1%)
[PASS] Kimi-K2.6           复算     1.03T  vs 公布     1.04T  ( 1.3%)
[PASS] Kimi-K3             复算     2.77T  vs 公布     2.78T  ( 0.3%)
[PASS] gpt-oss-120b        复算   116.8B  vs 公布   117.0B  ( 0.2%)
```

误差来自没算的 bias、norm、router 和对齐。**误差都在 1.3% 以内**，说明这些模型的结构确实可以用上面那几行公式描述——反过来说，如果某个新模型用这几行公式算不出来，那它多半真的引入了新东西。这也是本专题找选题的方法：**先算账，算不平的地方就是新闻。**

V4.1 的 890 B/token 值得单独说一句，因为它能**逐字节复算**出来（报告只给聚合值，没给拆项）：

| 项 | 每条目字节 | 持有层 | 压缩比 | 贡献 |
|---|---|---|---|---|
| 压缩 KV latent（512 维 FP4 + 32 个 scale） | 288 | 3 个 encoder source | $m=2$ | 432 |
| 压缩 KV latent | 288 | 1 个 decoder source | $m=1$ | 288 |
| 索引器 K（128 维 FP4 + 4 个 E8M0 scale） | 68 | 3 个 encoder source | $m=2$ | 102 |
| 索引器 K | 68 | 1 个 decoder source | $m=1$ | 68 |
| **合计** | | | | **890** |

注意两点：(1) 主 KV 只占 720 B，**索引器 K 占了 170 B（19%）**——稀疏注意力的「目录」本身就要钱；(2) 40 层里只有 4 层在存 global KV，若每层各存各的，按 $m=2/1$ 平均约 216 B/层，合计约 8.6 KB，差 10 倍。**层维共享是这里最大的一笔压缩。**

## 6. 这本账怎么用

后面每一篇，我们都会问三个问题：

1. 这个结构把哪一项预算降下来了？
2. 代价记在哪个科目上（参数？FLOPs？带宽？工程复杂度？）
3. 这个交换在什么规模/什么上下文长度下才划算？

剧透一下答案的形状：**序列方向**在拿「表达能力」换「KV」，**宽度方向**在拿「通信」换「参数」，**深度方向**在拿「显存流量」换「梯度稳定性」，**记忆方向**在拿「存储」换「计算」，**位置方向**则在悄悄退场——因为当全局注意力只剩 1/4 层、其余都是线性层时，位置信息可以由线性层的 decay 隐式提供。

下一篇从序列方向开始：注意力四代，以及 MLA 到底把什么吸收进了什么。

---

**配套代码**：`code/llm-arch-evolution/budget.py`（纯标准库，`python budget.py --test` 自测，`--ctx` 改上下文，`--detail` 看单项拆解）。

**系列导航**：

1. [导读：2026 年的结构地图]({{< relref "llm-arch-00-index" >}})
2. 本篇：先把账本立起来
3. [注意力四代]({{< relref "llm-arch-02-attention" >}})
4. [线性注意力与 3:1 混合]({{< relref "llm-arch-03-linear" >}})
5. [MoE 十年]({{< relref "llm-arch-04-moe" >}})
6. [残差流的加宽：HC 与 mHC]({{< relref "llm-arch-05-residual" >}})
7. [位置的退场]({{< relref "llm-arch-06-position" >}})
8. [条件记忆]({{< relref "llm-arch-07-memory" >}})
9. [五个收敛、三个分叉与三个赌注]({{< relref "llm-arch-08-trends" >}})
10. [残差连接与 AttnRes]({{< relref "llm-arch-09-attnres" >}})
