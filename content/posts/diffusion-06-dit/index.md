---
title: "扩散模型精读（六）DiT 与 MMDiT：Transformer 接管去噪网络"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 7
series: ["diffusion"]
tags: ["扩散模型", "系列", "DiT", "MMDiT", "Transformer", "adaLN"]
categories: ["LLM算法"]
---

Latent Diffusion 把图像压成了 token，但去噪网络还是 U-Net：卷积、下采样、上采样、skip connection。DiT（Peebles & Xie, 2022）问了一个很自然的问题：既然 latent 已经是 token 序列，为什么不直接用 ViT？

答案是：可以，而且 scaling 更好。DiT-XL/2 在 ImageNet 256 上做到 **FID 2.27**，并且给出了一个清晰的「Gflops ↑ → FID ↓」曲线。MMDiT 进一步把文本和图像做成两条流，成了 SD3、Flux、Qwen-Image 的共同骨架。

## 1. Patchify：把 latent 切成 token

给定 latent $z\in\mathbb{R}^{h\times w\times c}$，用 patch size $p$ 切块，得到

$$
N=\frac{h}{p}\times\frac{w}{p}
$$

个 token，每个 token 的维度是 $p^2c$，再线性投影到 $d$。256×256 的图，f=8 的 latent 是 32×32；$p=2$ 时 256 个 token，$p=1$ 时 1024 个。**patch size 是最便宜的降本旋钮**：$p$ 翻倍，token 数除以 4，attention 除以 16。

DiT 的 block 就是 ViT block，外加条件注入。条件有三种做法：

| 方式 | 做法 | 结果 |
|---|---|---|
| in-context | 把 $t,c$ 的 embedding 当额外 token 拼进序列 | 简单，但序列变长 |
| cross-attention | 在 block 里加一层 cross-attn | 多一层 attention，贵 |
| **adaLN-Zero** | 用 $t,c$ 回归 scale/shift/gate，调制 LayerNorm | 参数少、效果好，DiT 的默认 |

adaLN-Zero 的关键是**零初始化**：每个 block 的 gate 初始为 0，于是 block 一开始是恒等映射。这让深模型训练更稳，也让「条件」在早期不破坏主干。

## 2. 118.6 Gflops 的对账

DiT 论文报告 DiT-XL/2 的计算量是 **118.6 Gflops**。这个数字经常被误读成「118.6 GFLOPs」，然后推理预算算错一倍。

按 fvcore 的惯例，一次乘加（MAC）记 1 个「Flop」。DiT block 的 MACs 是：

$$
\underbrace{12Nd^2}_{\text{QKV+proj+MLP}}
+\underbrace{2N^2d}_{\text{attention}}
+\underbrace{6d^2}_{\text{adaLN}}
$$

代入 $N=256,d=1152,\text{depth}=28$：

$$
28\times(12\times256\times1152^2+2\times256^2\times1152+6\times1152^2)
\approx1.186\times10^{11}
$$

正好 118.6 G。换成「一次乘加 = 2 FLOPs」的口径，是 **237 GFLOPs**。`latent_budget.py` 复算的结果：

```text
256 : 256 token, 119 GMACs  (237 GFLOPs)
512 : 1024 token, 524 GMACs (1.05 TFLOPs)
1024: 4096 token, 2.91 TMACs (5.82 TFLOPs)
```

> **坑**：论文、nvprof、fvcore、厂商 PPT 对「FLOPs」的定义不一样。看到「Gflops」先问一句：是 MACs 还是 2×MACs？混用会让「H100 一张图几秒」的估算差一倍。

## 3. Scaling：Gflops 比参数量更能预测质量

DiT 的核心实验不是「Transformer 更好」，而是「在 latent 上，FID 和 Gflops 强相关」。他们固定训练步数，扫 depth/width/patch size，发现：

- 模型越大（Gflops 越高），FID 越低；
- patch 越小（token 越多），Gflops 越高，FID 也越低；
- 到 DiT-XL/2（675M 参数、118.6 GMACs）时，256 上 FID 2.27，512 上 3.04。

这条曲线解释了为什么后来的模型都在堆 DiT：它不是「卷积归纳偏置更好」的问题，而是**在同样的算力下，Transformer 更容易 scale**。U-Net 的 skip connection 在高分辨率下有用，但也限制了深度的扩展。

## 4. MMDiT：两条流，一层 joint attention

SD3 的 MMDiT 把文本和图像分成两条独立的流：

- 文本 token 走 text stream，图像 token 走 image stream；
- 每个 block 里做一次 **joint attention**：把两边的 Q/K/V 拼在一起算 attention，再拆回去；
- 两条流各自有 MLP、LayerNorm、残差；
- attention 前对 Q/K 做 RMSNorm，防止 logit 爆炸（SD3 论文明确说这是为了训练稳定性）。

这比「把文本当 cross-attention 的 key/value」更贵，但表达能力更强：文本和图像在每一层都能双向交互，而不是单向注入。SD3 用 8B 参数、16 通道 VAE、rectified flow，把这三件事绑在一起。

后来的模型基本沿着这条线：

| 模型 | 参数 | 文本编码器 | VAE | 备注 |
|---|---:|---|---|---|
| SD3 | 8B | CLIP-L + CLIP-G + T5-XXL | 16ch, f=8 | MMDiT + rectified flow |
| FLUX.1 | 12B | CLIP-L + T5-XXL | 16ch, f=8 | 双流 + 蒸馏版 |
| FLUX.2 | 32B | — | — | 2025-11 开源权重，4MP 编辑 |
| Qwen-Image | 20B | Qwen2.5-VL 7B | 16ch, f=8 | MMDiT，中文渲染 |

参数量的膨胀不只是「更大更好」，还因为 16 通道 VAE 把更多信息塞进 latent，需要更大的网络去建模（第五篇的 Figure 10 结论）。

## 5. 分辨率与 patch size 的账

1024×1024、f=8、$p=2$ 时是 4096 个 token；$p=1$ 时是 16384 个。`latent_budget.py` 算的每步 FLOPs：

| 配置 | 每步 FLOPs | 30 步 + CFG 的纸面时间（H100, 35% MFU） |
|---|---:|---:|
| 1024, p=2 | 5.82 TF | 1.01 s |
| 1024, p=1 | 49.2 TF | 8.53 s |

**patch size 从 2 降到 1，纸面时间涨 8.5 倍**。这就是为什么高分辨率模型宁愿用更大的 patch + 更强的条件注入，也不愿在全分辨率上做 attention。2K/4K 生成要么用级联/超分，要么用窗口/稀疏 attention（第十一篇）。

## 6. 几个工程坑

1. **adaLN-Zero 的零初始化**：gate 初始为 0，训练初期 block 是恒等；如果忘了零初始化，深模型会不稳。
2. **QK RMSNorm**：MMDiT 在 attention 前归一化 Q/K，不加的话高分辨率/大 batch 下 logit 会爆炸。
3. **位置编码**：DiT 用固定的 2D sin-cos；SD3/Flux 用 2D RoPE，便于外推到不同分辨率。换分辨率时要检查位置编码是否插值。
4. **文本流不能省**：把文本和图像拼成一条流（in-context）也能训，但条件控制会变弱；MMDiT 的双流是为了让两种模态各保留自己的归纳偏置。
5. **参数量 ≠ 计算量**：FLUX.2 32B 比 FLUX.1 12B 大，但推理成本还取决于 token 数、步数和是否蒸馏；不能只看参数量。

## 7. 小结

- DiT 用 patchify + ViT 替换 U-Net，adaLN-Zero 做条件注入；
- 118.6 Gflops 是 MACs，真实 FLOPs 是 237 G；
- scaling 看 Gflops，不看参数量；patch size 是最便宜的降本旋钮；
- MMDiT 用双流 + joint attention，成了 SD3/Flux/Qwen-Image 的共同骨架；
- 16 通道 VAE 需要更大模型，参数膨胀和 VAE 选择是耦合的。

下一篇换路径：不再沿扩散的曲线走，而是学一条更直的线。[Flow Matching 与 Rectified Flow]({{< relref "diffusion-07-flow-matching" >}})。
