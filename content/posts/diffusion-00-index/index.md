---
title: "扩散模型精读（导读）：从去噪到传输，一条主线"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 1
series: ["diffusion"]
tags: ["扩散模型", "系列", "生成模型", "DDPM", "Flow Matching", "DiT"]
categories: ["LLM算法"]
---

2026 年再看扩散模型，很容易被一堆名词劝退：DDPM、DDIM、score SDE、DPM-Solver、CFG、LDM、DiT、MMDiT、rectified flow、LCM、DMD……每个词背后都有一篇论文，每篇论文都自称解决了某个问题。

但它们其实在描述**同一个对象**：一条从噪声分布到数据分布的概率路径，以及沿着这条路径该往哪走。

- DDPM 用离散的马尔可夫链描述它；
- score SDE 用随机微分方程描述它；
- DDIM / DPM-Solver 只关心它的 ODE 形式，好少走几步；
- flow matching 换了一条更直的路径；
- 一致性模型 / DMD 干脆不走路，学一个一步到位的映射；
- LDM / DiT 改的是「在什么空间走、用什么网络走」。

这个专题按这条主线读 13 篇。每一篇都尽量回答三个问题：**它改了什么、省了什么、代价是什么**。

## 先记住一个画面

想象一条从「纯噪声」开往「真实图像」的传送带。训练时，我们把一张真实图像慢慢揉进噪声，记录每一站的样子；采样时，从终点站（纯噪声）往回走，每一步都问模型：**上一站长什么样**。

关键在于：模型学的不是「图像」，而是**每一站该往哪个方向走、走多远**。这个「方向」可以用四种语言写：

| 语言 | 符号 | 典型论文 | 一句话 |
|---|---|---|---|
| 噪声 | $\varepsilon_\theta(x_t,t)$ | DDPM | 预测这一步加进去的噪声 |
| 干净样本 | $x_{0,\theta}(x_t,t)$ | 早期 DDPM / 超分 | 直接猜终点 |
| 分数 | $s_\theta(x_t,t)=\nabla\log p_t(x_t)$ | NCSN、score SDE | 指向密度上升的方向 |
| 速度 | $v_\theta(x_t,t)$ | flow matching、rectified flow | 沿路径的速度矢量 |

这四种参数化在数学上可以互相换算（第二篇会给出换算表）。**很多「新方法」只是换了一种语言，或者换了一条路径**；真正改变计算量的，是路径的弯曲程度和离散化的步数。

> **类比的边界**：把扩散模型说成「把加噪录像倒放」很直观，但不准确。模型训练时看到的是**边缘分布** $p_t(x_t)$，不是某一条具体的加噪轨迹；反向过程也不是前向过程的逐帧倒放，除非你额外条件在 $x_0$ 上。真正被学的是一组「从每个噪声水平指向数据」的向量场，轨迹是积分出来的，不是存下来的。

## 五次搬家

把 2020–2026 的论文摊开，影响最大的其实是五次「搬家」，而不是五个损失函数。

### 1. 从像素搬到 latent（2021，LDM）

在 512×512×3 的像素空间去噪，一步要处理 78.6 万个元素；先压到 64×64×4 的 latent，只剩 1.64 万个，少 **48 倍**。代价是 VAE 的重建误差成为天花板，文字和小物体最先坏。

### 2. 从 U-Net 搬到 Transformer（2022，DiT）

DiT 把 latent 切成 patch，当成 token 喂给 ViT。DiT-XL/2 在 ImageNet 256 上做到 **FID 2.27**，而计算量是 **118.6 GMACs**（注意是 MACs，换成 FLOPs 要乘 2）。代价是 attention 的 $O(N^2)$：分辨率翻倍，token 数变 4 倍，attention 变 16 倍。

### 3. 从 $\varepsilon$ 搬到 $v$（2024，SD3 / Flux）

flow matching 不再预测噪声，而是预测「从噪声到数据」的速度。rectified flow 把条件路径拉成直线，少步 Euler 的误差更小。SD3 用 8B 的 MMDiT + rectified flow，把「架构」和「路径」一起换了。

### 4. 从 1000 步搬到 4 步（2023–2025，LCM / DMD）

一致性模型把「走完整条路」压成「一次映射」；LCM 用 32 个 A100 小时、4000 步蒸馏，把 SD 做到 2–4 步；DMD 用分布匹配而不是逐点回归，避免一步生成器把多个峰平均掉。

### 5. 从图像搬到视频（2024–2026，Sora / Wan）

视频多了一个时间轴。5 秒、24 fps、720p 是 120 帧；Wan 的 3D VAE 按 **4×8×8** 压缩后是 30×90×160 = **43.2 万 token**。full 3D attention 的注意力矩阵按 bf16 存要 **373 GB**，所以视频模型必须做 factorized / window attention，或者把时空切成块。

这五次搬家对应本系列的五个技术节点：latent（第五篇）、DiT（第六篇）、flow（第七篇）、少步（第八篇）、视频（第九篇）。剩下的是底座：DDPM（第一篇）、SDE（第二篇）、快速采样（第三篇）、引导（第四篇），以及工程侧的训练（第十篇）、推理（第十一篇）、评测（第十二篇）。

## 四个设计轴

读任何一篇扩散论文，都可以先把它放到这四个轴上：

| 轴 | 选项 | 代表 | 主要代价 |
|---|---|---|---|
| **时间** | 离散 1000 步 / 连续 $t$ / 多分辨率 | DDPM / score SDE / 级联 | 离散难做高阶求解；连续要先验 |
| **参数化** | $\varepsilon$ / $x_0$ / score / $v$ | DDPM / 超分 / NCSN / FM | 高噪声段误差放大不同；换约定易错 |
| **采样** | 随机 SDE / 确定性 ODE / 蒸馏 | DDPM / DDIM、DPM-Solver / LCM、DMD | 随机性换多样性；ODE 换速度；蒸馏换可控性 |
| **条件** | 分类器引导 / CFG / 文本 cross-attn | ADM / SD / DiT | 引导换保真度；CFG 要双 batch；文本编码器决定上限 |

四个轴不独立。选 flow matching，就更容易少步；选少步蒸馏，就牺牲了 guidance scale 的可调性；选 16 通道 VAE，就换来更好的重建，但去噪网络要更大。**论文之间的差异，往往不是谁用了哪个组件，而是谁把某个轴的代价算得更清。**

## 本系列怎么读

13 篇，按「基础 → 表示 → 路径 → 系统 → 评测」组织。

| # | 篇目 | 一句话 |
|---|---|---|
| 0 | 本篇 | 一条主线、五次搬家、四个轴 |
| 1 | [DDPM：一千步去噪的账]({{< relref "diffusion-01-ddpm" >}}) | ELBO 怎么变成 $\varepsilon$ 回归；线性日程的 259 步有效窗口 |
| 2 | [分数与 SDE：DDPM、SDE、ODE 是同一件事]({{< relref "diffusion-02-score-sde" >}}) | score、VP/VE、probability flow ODE 的换算表 |
| 3 | [快速采样：DDIM、DPM-Solver 与 10 步的极限]({{< relref "diffusion-03-fast-sampling" >}}) | 为什么 1000 步能砍到 10–20 步；误差从哪来 |
| 4 | [引导：从 classifier guidance 到 CFG]({{< relref "diffusion-04-guidance" >}}) | guidance scale 的锐化、过饱和与多样性坍缩 |
| 5 | [Latent Diffusion：把像素换成 latent]({{< relref "diffusion-05-latent-diffusion" >}}) | 48 倍压缩省在哪；VAE 是天花板 |
| 6 | [DiT 与 MMDiT：Transformer 接管去噪网络]({{< relref "diffusion-06-dit" >}}) | 118.6 GMACs 的对账；adaLN-Zero 与双流 |
| 7 | [Flow Matching 与 Rectified Flow]({{< relref "diffusion-07-flow-matching" >}}) | 直线路径、速度场与 SD3/Flux 的选择 |
| 8 | [少步生成：一致性模型、LCM、DMD]({{< relref "diffusion-08-few-step" >}}) | 四种蒸馏的账；一步为什么会平均掉峰 |
| 9 | [视频扩散：时空 token 与 3D VAE]({{< relref "diffusion-09-video" >}}) | 43.2 万 token 与 373 GB 的注意力矩阵 |
| 10 | [训练工程：EMA、CFG dropout 与显存账本]({{< relref "diffusion-10-training" >}}) | 8B 模型一次训练要多少显存；哪些默认值不能省 |
| 11 | [推理系统：步数、CFG 双 batch 与缓存]({{< relref "diffusion-11-inference" >}}) | 349 TFLOPs 的纸面账；patch size 决定 8 倍差距 |
| 12 | [评测与 2026 地图：FID 的坑与四个分叉]({{< relref "diffusion-12-eval-2026" >}}) | 同分布 FID 也不为 0；开源侧的四条路线 |

三条阅读路线：

- **只想建立判断力**：0 → 1 → 3 → 4 → 11 → 12。知道「路径、步数、引导、延迟」四件事，就能读懂大多数论文的摘要。
- **要做工程复现**：0 → 1 → 5 → 6 → 10 → 11。重点是 VAE、DiT 结构和显存/吞吐，而不是推导。
- **要跟研究前沿**：0 → 2 → 7 → 8 → 9 → 12。重点是 score/flow 的等价性、少步蒸馏和视频的时空建模。

## 配套代码

`code/diffusion/` 下 8 个纯 CPU 脚本，全部带 `--test`。它们不是模型实现，而是把文章里的账**算给你看**：

| 文件 | 作用 | 对应文章 | 一个数字 |
|---|---|---|---|
| `schedules.py` | 线性 vs cosine 日程的 SNR | 一、二 | 线性日程只有 **259/1000** 步 SNR ≥ 0 dB |
| `ddpm_toy.py` | 8 高斯上训一个最小 DDPM | 一 | 1000 步半径 2.031；20 步朴素跳采 1.337 |
| `sampler_zoo.py` | DDIM / Euler / Heun 对比 | 二、三 | DDIM 10 步 W2=0.092，5 步 0.234 |
| `cfg_toy.py` | CFG 的锐化与坍缩 | 四 | s=7.5 时少数峰从 39.8% 掉到 5.6% |
| `latent_budget.py` | latent / DiT / 视频 token 账 | 五、六、九、十一 | DiT-XL/2 = **118.6 GMACs = 237 GFLOPs** |
| `flow_vs_diffusion.py` | FM 与扩散的步数对比 | 七 | 4 步 FM 0.106，DDIM 0.335 |
| `distill_toy.py` | 一步回归 / 配对 / 速度蒸馏 | 八 | 独立配对一步半径 **0.019**，配对 1.985 |
| `fid_bias.py` | FID 的有限样本偏差 | 十二 | 同分布 n=100 时 FID **22.1**，n=10000 时 0.22 |

```bash
cd code/diffusion
for f in *.py; do
  echo "== $f =="
  python3 "$f" --test || exit 1
done
```

## 一句话结论

扩散模型的进步，主要是三个动作：**把路径选得更直、把表示压得更紧、把采样算得更省**。论文之间的差异，大多能落到这三个动作的某一条上；剩下的是工程：VAE、显存、并行、缓存和评测。

> **类比的边界**：把扩散模型类比成「艺术家从大理石里凿出雕像」听起来很美，但凿是**去掉**材料，扩散是**加入**结构——每一步都在把无序的噪声变成更有序的样本。更准确的类比是「一群粒子在向量场里被推向数据流形」，而向量场是从数据分布估计出来的，不是人手设计的。

下一篇从最基础、也最容易被跳过的一篇开始：[DDPM 的一千步到底在算什么]({{< relref "diffusion-01-ddpm" >}})。
