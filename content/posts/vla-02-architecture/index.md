---
title: "VLA 精读（二）架构：从 RT-2 到 π0 的 token 账"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 3
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "OpenVLA", "RT-2", "Octo", "pi0", "模型架构"]
categories: ["LLM算法"]
---

把 VLM 接上动作，架构上有四种常见做法：

1. **把动作写成文本**，复用 VLM 的 next-token prediction（RT-2）；
2. **在 VLM 后面加一个小动作头**，VLM 只做一次前向（Octo 的 diffusion head、OFT 的 L1 head）；
3. **把动作 token 塞进 VLM 的词表**，自回归解码（OpenVLA）；
4. **给 VLM 配一个独立的 action expert**，两套权重共享注意力（π0/π0.5/GR00T）。

四种做法没有本质高下，差别在**你愿意把算力花在哪、把延迟花在哪、把参数花在哪**。这一篇把这笔账拆开算。

## 1. RT-2：动作就是文本

RT-2 的洞察很简单：既然 VLM 能输出任意文本，那就把动作也写成文本。

具体做法（p.5–6）：7 个连续动作维度 + 1 个 terminate，每维离散成 256 个 bin，然后拼成一个字符串，比如 `"1 128 91 241 5 101 127"`，作为 VLM 的输出序列。训练就是标准的 next-token cross-entropy，和 VQA、captioning 混在一个 batch 里 co-fine-tune。

这带来一个纯架构上的好处：**不需要任何新参数**。RT-2 的参数量就是 VLM 的参数量——5B、12B、55B 三个档位（p.5）。动作头和语言头是同一个 `lm_head`。

代价也很直接：

- **控制频率 1–3 Hz**（55B）/ ~5 Hz（5B）（p.6）。每步要自回归解码 7–8 个 token，55B 模型在云端服务，这个速度已经是工程优化后的结果。
- **动作精度被 bin 限制**，和上一篇的 255 档问题一样。
- **物理技能不会从 web 数据里长出来**。论文自己说得很清楚："The model's physical skills are still limited to the distribution of skills seen in the robot data"（p.3, p.11）。web 数据带来的是**语义泛化**（认识没见过的物体、理解「把可乐罐放到 Taylor Swift 照片上」），不是新的动作技能。

RT-2 的实验也印证了这一点：seen 任务上 55B 的 91% 和 RT-1 的 92% 基本持平，优势全在 unseen（平均 62% vs 32%）；涌现任务平均 60 vs 17（p.23–24）。**大模型买的是 OOD 语义，不是精细控制。**

## 2. Octo：把去噪放进小头里

Octo 走的是另一条路：backbone 尽量小，动作生成放在一个独立的 diffusion head 里。

架构（p.2, p.4–5）：

- 浅 CNN patch encoder（16×16 patch）+ ViT 风格 transformer；Octo-Small 27M，Octo-Base 93M（不含 111M 的 t5 语言编码器）。
- 语言用 t5-base 编成 16 个 token。
- **readout token**：类似 BERT 的 `[CLS]`，只读不写，把整个观测序列压成一个向量，喂给动作头。
- 动作头：3 层 MLP、hidden 256、cosine 噪声表、**20 步去噪**。关键设计是：**backbone 每次动作预测只前向一次**，20 步去噪全在小 MLP 里做。

这笔账很划算：93M 的 backbone 前向一次，加一个几百万参数的 MLP 跑 20 步，总计算量远小于「7B 模型自回归 7 个 token」。Octo 在 6 个微调域上平均 **72%**，而从头训练的 ResNet+Transformer 只有 20%（p.7）。

但 Octo 的天花板也明显：它的 zero-shot 只是「和 RT-2-X 相当」，泛化分解里 **novel skill 只有 5%**（p.17）。小 backbone + 小数据（800k trajectories）能学会多任务，但学不会开放世界的语义。

## 3. OpenVLA：7B 自回归，但只算动作 token 的 loss

OpenVLA 基本是 RT-2 的开源缩小版，但有几个工程细节值得记：

- **双视觉编码器**：SigLIP + DINOv2，各 300M 左右，按通道 concat，再经 2 层 MLP projector 投到 Llama 2 7B 的维度（p.4）。DINOv2 提供空间/纹理特征，SigLIP 提供语义，这个组合后来被很多模型沿用。
- **每视角 256 个 patch**（224×224，patch 14）。单图输入，无 history、无 proprio（p.6）。
- **动作 token 覆写词表最后 256 个最不常用 token**，`lm_head` 直接输出这 256 个 logits。
- **loss 只算在动作 token 上**（"evaluating the cross-entropy loss on the predicted action tokens only", p.5）。

第 4 点很容易被忽略，但它是 VLA 训练的标准做法：prompt 和图像 token 不产生监督信号，否则模型会把容量浪费在复述指令上。代码里就是 `labels[:-(len(action)+1)] = IGNORE_INDEX`（`prismatic/vla/datasets/datasets.py:63`）。

代价是推理：单步动作在 A100 上 **0.33 s**（OFT p.3），RTX 4090 上约 **6 Hz**（p.6）。7 个 token 的自回归，每个 token 都要过一遍 7B 模型——即使有 KV cache，7 次前向也远比一次慢。

## 4. π0：两套权重，一条 mask

π0 的架构是当前开源 VLA 里最有代表性的：**一个 transformer，两套权重**。

- **VLM 部分**：PaliGemma（Gemma-2B + SigLIP），width 2048、depth 18、18 heads、1 个 KV head、head_dim 256（p.15）。
- **action expert**：width 1024、MLP 4096、约 **300M**，从零初始化（p.5）。
- 总参数 **3.3B**（3B VLM + 300M expert）。

前向分两段：

- **prefix**：图像 token + 语言 token，双向注意力，只算一次，KV 缓存下来；
- **suffix**：1 个 state token + 50 个动作 token，每个去噪步只算这一段，通过同一套 self-attention 读 prefix 的 KV（两段 K/V 拼接后一起做注意力）。

`code/vla/block_attn_mask.py` 把这条 mask 画了出来（π0 版，state 在 suffix）：

```
            img0 img1 img2 img3 txt0 txt1 txt2 stat act0 act1 act2 act3 act4
      img0     #    #    #    #    #    #    #    .    .    .    .    .    .
      ...
     state     #    #    #    #    #    #    #    #    .    .    .    .    .
      act0     #    #    #    #    #    #    #    #    #    #    #    #    #
      act1     #    #    #    #    #    #    #    #    #    #    #    #    #
```

三个性质：

1. **prefix 内部双向**：图像和语言可以互相看，不是因果 mask。
2. **prefix 看不到 suffix**：VLM 表征不被动作污染。π0.5 论文把这条写成 "no VLM embedding attends to the action expert"（p.19）。
3. **动作 token 之间全连接**：50 个动作是并行去噪的，不是自回归。

这条 mask 的工程价值在 **KV cache**：prefix 只算一次，10 步去噪每步只算 51 个 suffix token。`block_attn_mask.py` 算了一笔账：

```
     P    S         无缓存(元素)         有缓存(元素)      加速
   256   51       9.425e+05       2.221e+05    4.2x
   768   51       6.708e+06       1.008e+06    6.7x
  1024   51       1.156e+07       1.597e+06    7.2x
  2048   51       4.406e+07       5.265e+06    8.4x
```

P=768 对应 3 路 224×224 图像（每路 256 token）。没有缓存，10 步去噪要把 819 个 token 全算 10 遍；有缓存，prefix 只算一遍，后面每步只算 51 个。**前缀缓存不是优化项，是 50 Hz VLA 能成立的前提。**

## 5. 四种架构的账

| 模型 | 参数量 | 视觉 token | 动作表示 | 动作头 | 每步推理 | 控制频率 |
|---|---|---|---|---|---|---|
| RT-2 | 5B / 12B / 55B | ViT-22B patch（未披露具体数） | 7 维 × 256 bin 文本 | 复用 lm_head | 自回归 7–8 token | 1–3 Hz |
| Octo | 27M / 93M（+111M t5） | 第三人称 256 + wrist 64 | 连续 7 维 | 3 层 MLP diffusion，20 步 | backbone 1 次 + 头 20 步 | 5–15 Hz（评测域） |
| OpenVLA | 7B | 每视角 256（SigLIP+DINOv2） | 7 维 × 256 bin | 复用 lm_head | 自回归 7 token，0.33 s | ~6 Hz |
| π0 | 3.3B | 3×256 | 50×32 连续 | Gemma-300M expert + flow，10 步 | prefix 1 次 + 10 步 suffix | 50 Hz 机器人，13.7 Hz 推理 |

一个反直觉的结论：**参数量最大的 RT-2（55B）和最少的 Octo（93M），在各自评测集上都能用；决定成败的不是参数量，是动作表示和推理预算是否匹配任务频率。**

RT-2 用 55B 换语义泛化，接受 1–3 Hz；Octo 用 93M 换多任务微调，接受 diffusion 20 步；π0 用 3.3B + expert 换 50 Hz 的连续动作，接受 73 ms 推理。三者都在自己选择的约束里做到了极致，但换个任务就会露馅。

## 6. 那 VLM 到底贡献了什么？

这是 VLA 里最容易被过度宣传的问题。把论文里的证据摊开：

- **RT-2**：web 预训练带来 OOD 语义（unseen 62% vs RT-1 32%），但**不带来新技能**（p.3, p.11）。
- **OpenVLA**：7B VLM 在 29 任务上比 55B 的 RT-2-X 高 16.5 个绝对点（p.1），但作者自己说可靠性 <90%，且吞吐撑不起 50 Hz（p.11）。
- **π0**：VLM 预训练让 π0 明显优于 π0-small 和 scratch（p.9–10），但专家只有 300M，动作能力主要靠机器人数据。
- **π0.5**：web 数据的好处"much of the benefit ... lies in improving the high-level policy"（p.11）——它提升的是**子任务选择**，不是低层控制。

所以更准确的说法是：**VLM 提供语义先验和开放世界识别，动作精度和动力学靠机器人数据与动作头**。把 VLA 当成「VLM 大了就能控制好」，是把两件事混为一谈。

> **类比的边界**：VLM 像一个见过很多图片和文字的工程师，能认出杯子、听懂「把杯子放到水槽左边」；但他没练过手腕的力矩控制。你可以让他指挥，不能让他直接拧螺丝。π0 的 action expert、GR00T 的 System 1，本质上都是给这位工程师配一双练过的手。

## 小结

- 四种架构的差别，本质是「动作表示 + 推理预算 + 参数分配」的组合，不是谁比谁先进。
- RT-2 证明 VLM 的语义可以迁移到动作，但物理技能仍受机器人数据限制。
- Octo 证明小模型 + 小 diffusion 头可以多任务，但 novel skill 只有 5%。
- OpenVLA 是开源 7B 基线，0.33 s/步、6 Hz，靠 loss mask 只训动作 token。
- π0 的关键不是 3.3B，而是 **prefix/suffix 分离 + KV cache + 300M expert**，把 10 步去噪压到 73 ms。
- VLM 贡献语义，动作头贡献精度；把两者混为一谈，就会得出「模型越大控制越好」的错误结论。

下一篇进入动作头内部：[diffusion 和 flow matching 到底在算什么]({{< relref "vla-03-action-head" >}})，以及为什么 π0 从 diffusion 换成了 flow。
