---
title: "扩散模型精读（九）视频扩散：时空 token 与 3D VAE"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 10
series: ["diffusion"]
tags: ["扩散模型", "系列", "视频生成", "3D VAE", "Wan", "Sora"]
categories: ["LLM算法"]
---

图像扩散的 token 账还没算完，视频又加了一个维度。一张 720p 的图是 92 万像素；5 秒 24 fps 的 720p 视频是 **120 帧**，3.3 亿个元素。如果直接在像素上做 attention，矩阵是 $10^{11}$ 量级。

视频模型的几乎所有设计，都是被这个数字逼出来的：3D VAE、时空 patch、factorized / window attention、causal 结构。这一篇算清楚「一个视频有多少 token」，然后看主流模型怎么解。

## 1. 一个 5 秒 720p 视频的 token 账

Wan 的 3D VAE 压缩率是 **4×8×8**：时间 4 倍、空间 8×8 倍，latent 16 通道。5 秒 24 fps 720p：

- 帧数：$5\times24=120$；
- 压缩后：$120/4=30$ 帧，$720/8=90$，$1280/8=160$；
- latent：$30\times90\times160=432{,}000$ 个 token，16 通道。

对比一下其他 VAE：

| VAE | 压缩率 | 通道 | token 数 | 元素数 |
|---|---|---:|---:|---:|
| Wan-VAE | 4×8×8 | 16 | 432K | 6.91M |
| SVD | 1×8×8 | 4 | 1.73M | 6.91M |
| Open-Sora Plan | 4×8×8 | 4 | 432K | 1.73M |

有意思的是 Wan 和 SVD 的**元素数一样**（6.91M），因为 Wan 把时间压了 4 倍、通道翻了 4 倍。但 token 数差了 4 倍——attention 的代价看 token 数，不看元素数。

## 2. 432K token 的注意力矩阵有多大

full 3D attention 的矩阵是 $N\times N$，$N=432{,}000$：

$$
N^2=1.87\times10^{11}\ \text{个元素}
$$

按 bf16 存，单头就是 **373 GB**。这还没乘 head 数、没算 QK 的中间结果。所以视频模型不可能做 full 3D attention。

`latent_budget.py` 的输出：

```text
Wan-VAE 4x8x8,16ch: 30x90x160 x16 = 6.91M 元素, 432K token
full 3D attention（单头、bf16）: 186.6G 元素 = 373 GB
每个 block 的 QK/AV MACs: 430T（d=1152 时）
```

三种解法：

1. **factorized attention**：先做空间 attention（每帧内 $90\times160$），再做时间 attention（同位置跨帧 $30$）。代价从 $N^2$ 降到 $HW\cdot T^2+T\cdot(HW)^2$，仍然大，但可控。
2. **window / 稀疏 attention**：只在局部时空窗口内算，比如 $2\times8\times8$ 的窗口；代价线性于 token 数。
3. **3D 卷积/时空分离的 DiT block**：把时间维当额外的 batch 或通道，避免全连接。

Wan、HunyuanVideo、CogVideoX 都用了不同比例的 factorized/window；Sora 的技术报告则用「spacetime patches」把视频切成时空块，再喂给 Transformer。

## 3. 3D VAE：时间因果与压缩率

视频 VAE 和图像 VAE 最大的区别是**时间维**：

- **causal VAE**：第 $t$ 帧的 latent 只依赖 $\le t$ 的帧，支持流式/自回归；
- **非 causal**：可以用双向上下文，重建更好，但不能流式。

Wan-VAE 用 3D causal 结构，时间压缩 4 倍，空间 8×8，16 通道。它的解码按 chunk 做：每个 latent 对应最多 4 帧，避免一次性解码长视频爆显存。

> **坑**：时间压缩比不是越大越好。4 倍时间压缩意味着 24 fps 的视频在 latent 里只有 6 fps；快速运动（运动员、挥手）会产生拖影。很多模型在训练时用 1×8×8（SVD）保时间细节，推理时再用插帧补帧率。

## 4. 主流模型的结构选择

| 模型 | 参数 | VAE | 结构 | 备注 |
|---|---:|---|---|---|
| Sora（2024） | 未公开 | 视频压缩网络 | spacetime patches + Transformer | 可变时长/分辨率 |
| CogVideoX | 5B | 3D causal VAE | 时空 DiT | 开源 |
| HunyuanVideo | 13B | 3D VAE | 双流 + 时空 attention | 开源，2024-12 |
| Wan | 1.3B / 14B | 4×8×8, 16ch | 时空 DiT | 开源，2025-03 |

参数量的跨度很大：1.3B 能在消费卡上跑，14B 需要多卡；13B 的 HunyuanVideo 在开源时是「最大开源视频模型」。这些模型的共同点是：**VAE 先压时空，DiT 再在 latent 上做时空 attention**，区别在 attention 怎么分解、文本怎么注入。

## 5. 视频扩散 vs 自回归

视频生成有两条路线：

- **扩散**：一次生成整段（或分块），每步去噪整个时空 latent；
- **自回归**：逐帧/逐块预测，类似语言模型。

扩散的优点是全局一致性好、训练稳定；缺点是长视频的 token 数线性涨，且不能流式。自回归的优点是流式、可变长；缺点是误差累积、长期一致性差。

2025–2026 的一个明显趋势是**混合**：扩散做关键帧/块，自回归做长程衔接；或者用 causal 扩散做流式。Wan 的 causal VAE 就是为流式留的口子。

## 6. 长视频的三个硬问题

1. **漂移**：生成 10 秒后，人物/背景开始变形；原因是误差在时间维累积，且训练时很少有这么长的样本。
2. **显存**：432K token 的 latent 本身不大（6.91M 元素，bf16 约 14 MB），但 attention 的中间结果和激活是瓶颈；factorized attention 的峰值仍然很高。
3. **运动一致性**：时间压缩 + 少步采样会让快速运动糊；需要在 VAE 压缩率、帧率、步数之间折中。

## 7. 小结

- 5 秒 720p 视频经 4×8×8 VAE 压成 432K token；
- full 3D attention 的矩阵是 373 GB，必须 factorized/window；
- 3D VAE 的时间压缩比决定运动细节，causal 结构决定能否流式；
- Wan/HunyuanVideo/CogVideoX 的差异主要在 attention 分解和文本注入；
- 长视频的瓶颈是漂移、显存和运动一致性，不是单帧质量。

下一篇回到训练侧：一个 8B 扩散模型训一次要多少显存，哪些默认值不能省。[训练工程]({{< relref "diffusion-10-training" >}})。
