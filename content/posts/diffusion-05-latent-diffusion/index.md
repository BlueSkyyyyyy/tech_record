---
title: "扩散模型精读（五）Latent Diffusion：把像素换成 latent，省下了什么"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 6
series: ["diffusion"]
tags: ["扩散模型", "系列", "Latent Diffusion", "VAE", "Stable Diffusion"]
categories: ["LLM算法"]
---

到第四篇为止，扩散都发生在像素空间：512×512×3 的图像，就是 78.6 万个元素的向量。每一步去噪都要在这个空间里跑一次 U-Net，attention 的代价还随 token 数平方涨。

Latent Diffusion（Rombach et al., 2021）的做法很直接：先训一个 VAE，把图像压到 64×64×4 的 latent，再在 latent 上做扩散。元素数从 78.6 万降到 1.64 万，**少 48 倍**；token 数从 26.2 万降到 4096，**少 64 倍**。

代价也很直接：**VAE 的重建误差成为生成质量的天花板**。这一篇算清楚这笔交换。

## 1. 压缩比：$f^2$ 而不是 $f$

VAE 的编码器把 $H\times W$ 压到 $H/f\times W/f$，通道数 $c$。元素数的压缩比是

$$
\frac{H\cdot W\cdot 3}{(H/f)(W/f)\cdot c}=\frac{3f^2}{c}
$$

Stable Diffusion 用 $f=8$、$c=4$：压缩比 $3\times64/4=48$。如果只看空间边长，$f=8$ 像是 8 倍；实际元素数是 $8^2=64$ 倍，再乘通道比 $3/4$，才是 48 倍。**这是一个常见的口误来源。**

具体到 512×512：

| 空间 | 形状 | 元素数 | 相对像素 |
|---|---|---:|---:|
| 像素 | 512×512×3 | 786,432 | 1× |
| latent（f=8, c=4） | 64×64×4 | 16,384 | 1/48 |
| DiT token（patch=2） | 32×32 | 1,024 | — |

DiT 在 256×256 上只有 256 个 token，512 上 1024 个，1024 上 4096 个。这就是为什么 latent 扩散能把 attention 控制在可训练的范围里。

## 2. 结构：VAE + U-Net + cross-attention

LDM 的三个部件：

1. **VAE**：编码器 $\mathcal{E}$ 把图像压成 $z$，解码器 $\mathcal{D}$ 还原；训练时加 KL 和感知损失，保证重建质量。
2. **去噪 U-Net**：在 $z$ 上做扩散，结构还是 DDPM 的 U-Net，但输入通道是 4。
3. **条件注入**：文本/类别先编码成序列，再通过 cross-attention 注入 U-Net 的中间层。这是 LDM 相对 pixel diffusion 的另一个便宜之处：条件只和 latent 序列交互，不和 78 万个像素交互。

Stable Diffusion 1.5 用 CLIP ViT-L/14 的文本编码器；SDXL 加到两个（CLIP-L + OpenCLIP bigG），UNet 大约是 SD1.5 的 3 倍大；SD3/Flux 换成 16 通道 VAE + MMDiT（下一篇）。

## 3. 数字：LDM 到底做到了什么

LDM 论文里，class-conditional ImageNet 256 的 **LDM-4-G**（f=4，带 classifier-free guidance）做到 **FID 3.60**，用 250 步 DDIM，guidance scale 1.5。这个数字在 2021 年很有竞争力，但更重要的不是 FID，而是**训练和推理的成本曲线**：

- 像素空间的扩散，每步 U-Net 要处理 256×256×3；latent 空间只处理 32×32×4（f=8 时）；
- 同样的 GPU 小时，latent 可以训更大的模型、更多的步；
- 采样时每步的显存和 FLOPs 都按压缩比下降。

LDM 论文还比较了 $f=4/8/16$ 的重建质量：$f$ 越小，PSNR 越高、重建 FID 越低，但去噪网络越贵。Stable Diffusion 选 $f=8$ 是一个工程折中：重建够用，计算可控。

> **坑**：LDM 的 latent 不是「标准正态」，VAE 的输出有一个缩放因子。Stable Diffusion 用 `0.18215` 把 latent 的方差缩到 1 附近；训练和推理必须用同一个因子，否则颜色会整体偏。换 VAE（比如 SD1.5 → SDXL）时，这个因子和通道数都可能变，不能直接复用。

## 4. VAE 是天花板

latent 扩散的上限由 VAE 的重建质量决定。如果 VAE 重建不出文字，扩散模型再强也生成不出文字；如果 VAE 把细粒度纹理抹平，生成的人脸就会「塑料感」。

这就是为什么 2024 年之后，模型开始换 VAE：

- **SD3 / Flux**：16 通道 VAE，$f=8$，重建细节更好，但去噪网络需要更大容量（SD3 论文 Figure 10 显示，16 通道在浅层模型上不如 8 通道，深到 $d=22$ 才追平）；
- **Qwen-Image**：16 通道 VAE + 20B MMDiT，配 7B 的 Qwen2.5-VL 文本编码器；
- **视频**：Wan 用 4×8×8、16 通道的 3D VAE（第九篇）。

通道数不是越多越好：16 通道把更多信息塞进 latent，去噪网络要花更多参数去建模；4 通道更容易训，但重建上限低。**这是一个和模型规模耦合的选择**，不是单独的参数。

## 5. 成本账：省在哪，花在哪

`latent_budget.py` 把 512/1024 的 token 和 DiT MACs 算出来：

```text
512x512  : 像素 786K -> latent 64x64x4=16.4K -> token(p=2) 1.02K
1024x1024: 像素 3.15M -> latent 128x128x4=65.5K -> token(p=2) 4.1K

DiT-XL/2 @256 : 119 GMACs  (=237 GFLOPs)
DiT-XL/2 @512 : 524 GMACs  (=1.05 TFLOPs)
DiT-XL/2 @1024: 2.91 TMACs (=5.82 TFLOPs)
```

latent 扩散把「像素数」换成「token 数」，再把「token 数」换成 attention 的 $O(N^2)$。512 时 1024 个 token 很便宜；1024 时 4096 个 token，attention 开始主导；再往上（2K/4K），就必须用 patch size、窗口或序列并行。

VAE 解码本身也不便宜：1024×1024 的解码是一次上采样卷积网络，延迟通常占端到端 10–30%。很多「扩散模型推理慢」的抱怨，其实慢在 VAE 和文本编码器，而不是去噪主干。

## 6. 小结

- latent diffusion 用 VAE 把元素数压 48 倍，token 数压 64 倍；
- 代价是 VAE 重建成为天花板，文字/细节最先坏；
- $f=8$、4 通道是 SD1.5/SDXL 的折中；SD3/Flux 换 16 通道，配合更大的 DiT；
- 换 VAE 要同时换缩放因子、通道数和去噪网络容量，不能只换一半。

下一篇讲 latent 上的去噪网络怎么从 U-Net 换成 Transformer：[DiT 与 MMDiT]({{< relref "diffusion-06-dit" >}})。
