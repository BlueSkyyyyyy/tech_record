---
title: "扩散模型精读（十）训练工程：EMA、CFG dropout 与显存账本"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 11
series: ["diffusion"]
tags: ["扩散模型", "系列", "训练", "EMA", "显存", "FSDP"]
categories: ["LLM算法"]
---

前九篇讲的是「模型怎么设计」。这一篇讲「模型怎么训」：同样的结构，训练细节错了，FID 会差一大截；显存算错了，8B 模型根本跑不起来。

先给结论：一个 8B 的扩散模型，用 AdamW + bf16 混合精度，**光参数、梯度、优化器状态就要约 128 GB**，还没算激活。单卡 80 GB 装不下，必须用 FSDP/ZeRO 或 8-bit 优化器。

## 1. 显存账本：8B 模型的钱花在哪

混合精度训练的标准配置：

| 项 | 精度 | 大小（8B） |
|---|---|---:|
| 模型权重 | bf16 | 16 GB |
| fp32 master weights | fp32 | 32 GB |
| 梯度 | bf16 | 16 GB |
| Adam $m$ | fp32 | 32 GB |
| Adam $v$ | fp32 | 32 GB |
| **小计** | | **128 GB** |
| EMA 副本 | bf16 | 16 GB |
| 激活（取决于 batch/seq） | bf16 | 几十 GB |

如果不用 fp32 master（纯 bf16 训练），可以省 32 GB，但小模型/大学习率下容易不稳。8-bit Adam 把 $m,v$ 压到 8-bit，省 48 GB，代价是精度和实现复杂度。

FSDP/ZeRO-3 把这些状态按卡数切分：8 卡时每卡约 16 GB（不含激活），再加激活和临时 buffer。这就是为什么 8B 模型通常要 8×80 GB 才能舒服地训。

> **坑**：很多人只算「权重 16 GB」，然后惊讶 80 GB 卡 OOM。训练显存 ≈ 权重 × 8（AdamW 混合精度）+ 激活；推理才是权重 × 2（bf16）。

## 2. 时间步采样：均匀不是最优

DDPM 默认 $t\sim\mathcal{U}\{1,\dots,T\}$。但不同 $t$ 的难度差异很大：高噪声段学全局结构，低噪声段学细节，中间段最难。

SD3 用 **logit-normal** 采样：

$$
t\sim\mathrm{LogisticNormal}(0,1)
$$

把样本集中在中间时段（$t\approx0.5$），两端少采。直觉是：中间段的去噪最难，应该多训；两端相对容易，少训不影响。

另一种做法是 **min-SNR 加权**：给损失乘一个与 SNR 有关的权重，避免高噪声段/低噪声段的梯度主导。它和 logit-normal 不冲突，可以一起用。

> **坑**：换时间采样后，**采样器的步长也要重调**。训练分布变了，模型在不同 $t$ 的误差分布也变了；原来 20 步的 solver 可能需要 25 步。

## 3. EMA：为什么必须有

扩散模型的训练损失是随机的（每步采不同的 $t$、噪声、batch），权重会在最优点附近抖动。EMA（指数滑动平均）维护一份平滑的权重：

$$
\theta_{\text{EMA}}\leftarrow \lambda\theta_{\text{EMA}}+(1-\lambda)\theta
$$

常用 $\lambda=0.9999$，每步更新。采样用 EMA 权重，FID 通常比原始权重好一截——因为它平均掉了训练后期的噪声。

几个工程细节：

- EMA 通常存 **bf16** 或 fp32 副本；fp32 更准，但 8B 模型多 32 GB；
- 分布式训练时，各 rank 用相同的参数、相同的更新顺序维护同一份 EMA；也可以只在 rank 0 更新后广播。
- **别只保存原始权重**：很多开源模型只发 EMA，复现时要确认。

## 4. CFG dropout 与 caption dropout

CFG 需要模型同时会「有条件」和「无条件」。训练时以概率 $p$（常见 **10%**）把条件替换成空条件：

$$
c\leftarrow\varnothing\quad\text{with probability }p
$$

这就是 CFG dropout。$p$ 太小，无条件预测不准，guidance 效果差；$p$ 太大，条件信号被浪费，模型欠拟合。

caption dropout 是另一回事：随机丢掉整段文本，让模型学会「无文本也能生成」。它和 CFG dropout 可以叠加，但比例要控制，否则文生图的对齐会掉。

> **坑**：训练时 drop 的是「条件」，不是「文本 token」。实现上要把整段 caption 换成空串的 embedding，而不是把 token 置零；两者在 cross-attention 里不等价。

## 5. 数据：分辨率桶、重标注与审美过滤

扩散训练的数据工程，比损失函数更影响最终质量：

- **分辨率桶**：把不同长宽比的图分到若干桶里，每桶内 resize 到同一尺寸，避免拉伸；桶的边界要是 8/16 的倍数（VAE 下采样）。
- **重标注**：用更强的 VLM 给图片重新生成 caption（DALL-E 3 的做法），比原始 alt-text 干净得多；代价是标注成本。
- **审美过滤**：用 aesthetic predictor 打分，过滤低质量图；但过头会让风格单一。
- **合成数据**：用强模型生成数据再训弱模型，能补长尾，但要防「模型自噬」导致的分布窄化。

## 6. 训练稳定性

- **bf16 vs fp16**：bf16 动态范围大，扩散训练首选；fp16 需要 loss scaling，且在高噪声段容易溢出。
- **梯度裁剪**：常用 1.0；扩散的梯度在 $t$ 两端会尖，裁剪能防炸。
- **学习率 warmup**：大模型常用几千步 warmup；DiT/MMDiT 对 warmup 敏感。
- **QK norm**：MMDiT 在 attention 前归一化 Q/K，训练更稳（第六篇）。
- **gradient checkpointing**：用时间换显存，通常能省 30–50% 激活，代价是 20–30% 速度。

## 7. 小结

- 8B 模型 AdamW 混合精度约 128 GB 状态，必须分片；
- 时间步采样用 logit-normal / min-SNR，均匀不是最优；
- EMA 0.9999 是默认，采样用 EMA 权重；
- CFG dropout 10%，drop 的是整段条件；
- 数据工程（桶、重标注、过滤）比损失更影响质量；
- bf16 + 梯度裁剪 + QK norm + checkpointing 是稳定训练的四件套。

下一篇算推理：同样的模型，为什么有人 1 秒出图，有人 10 秒。[推理系统]({{< relref "diffusion-11-inference" >}})。
