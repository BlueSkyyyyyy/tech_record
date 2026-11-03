---
title: "扩散模型精读（四）引导：从 classifier guidance 到 CFG"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 5
series: ["diffusion"]
tags: ["扩散模型", "系列", "Classifier Guidance", "CFG", "引导", "条件生成"]
categories: ["LLM算法"]
---

前三篇都在讲「无条件生成」：给一团噪声，生成一个像数据的样本。但真正让扩散模型出圈的是**条件生成**——给一句话，生成对应的图。

条件生成的核心问题是：怎么让样本既符合条件、又保持真实感？这一篇讲两种引导方式：

- **classifier guidance**（ADM，2021）：外挂一个分类器，用它的梯度推样本；
- **classifier-free guidance**（CFG，2022）：不挂分类器，用「有条件/无条件」两个预测的差做引导。

CFG 是今天所有文生图模型的默认配置。它好用，但也不是免费的：guidance scale 每加一点，保真度换多样性，过头就是过饱和和模式坍缩。

## 1. Classifier guidance：用分类器的梯度推一把

条件生成的分数可以拆成

$$
\nabla\log p(x\mid c)=\nabla\log p(x)+\nabla\log p(c\mid x)
$$

第一项是数据分布的分数（无条件模型已经学了），第二项是「给定 $x$，它有多像类别 $c$」。ADM（Dhariwal & Nichol, 2021）的做法是：在噪声数据上训一个分类器 $p_\phi(c\mid x_t,t)$，然后用

$$
\tilde\varepsilon=\varepsilon_\theta(x_t,t)-s\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p_\phi(c\mid x_t,t)
$$

采样。$s$ 是引导强度，$s=0$ 退回无条件，$s$ 越大越「像 $c$」。

ADM 的数字很硬：ImageNet 128×128 **FID 2.97**、256×256 **4.59**、512×512 **7.72**；再接一个上采样扩散模型（super-resolution），256 和 512 分别到 **3.94 / 3.85**。这在 2021 年超过了当时最好的 GAN。

但 classifier guidance 有三个工程问题：

1. **要额外训一个分类器**，而且要在每个噪声水平上都准；
2. **分类器在噪声输入上外推**，$t$ 大时梯度不可靠；
3. **引导强度 $s$ 和分类器的校准绑死**，换数据集要重调。

CFG 把这三个问题一起绕开了。

## 2. CFG：用「条件/无条件」的差

CFG（Ho & Salimans, 2022）训练时随机丢掉条件（dropout），让同一个网络同时会做有条件预测 $\varepsilon_\theta(x_t,c)$ 和无条件预测 $\varepsilon_\theta(x_t,\varnothing)$。采样时，

$$
\tilde\varepsilon=(1+w)\,\varepsilon_\theta(x_t,c)-w\,\varepsilon_\theta(x_t,\varnothing)
$$

$w=0$ 是纯条件，$w>0$ 放大条件方向。把它写成 score：

$$
\nabla\log p_w(x\mid c)=(1+w)\nabla\log p(x\mid c)-w\nabla\log p(x)
$$

对应的分布是

$$
p_w(x\mid c)\propto \frac{p(x\mid c)^{1+w}}{p(x)^w}
$$

这就是「锐化」：条件密度高的地方被放大，条件密度低的地方被压低。$w$ 越大，分布越尖。

### 2.1 记号坑：$w$ 还是 guidance scale

CFG 论文里的 $w$ 是**增量**，diffusers 的 `guidance_scale` 是 $s=1+w$。所以：

- 论文说 $w=0.3$，代码里是 `guidance_scale=1.3`；
- 社区常说的 `guidance_scale=7.5`，对应论文的 $w=6.5$。

CFG 论文的实验里，**FID 最优在 $w=0.1$–$0.3$**，而 **IS 最优在 $w\ge4$**；论文 Figure 3 还专门展示了 $w=3$ 时颜色过饱和。这看起来和今天动辄 7.5 的默认值矛盾，其实是三件事不同：

1. 指标不同：FID 奖励「像真实分布」，IS 奖励「每张图都清晰可辨」；
2. 模型不同：CFG 论文是 128×128 的 ImageNet 类条件模型，今天的文生图模型大得多；
3. 任务不同：文生图要的是 prompt 对齐和视觉质量，不是 FID。

所以「guidance scale 多少」没有统一答案，它是**指标和审美的函数**。

### 2.2 代价：双 batch 与多样性

CFG 每步要算两次前向（有条件 + 无条件）。工程上有两种做法：

- **双 batch**：把 $[c,\varnothing]$ 拼成一个 batch，一次前向出两个预测，显存换延迟；
- **两次前向**：省显存，但延迟翻倍。

数学上，CFG 还会**压窄分布**。在 1D 四峰玩具上（`cfg_toy.py`），条件分布是 60/40 的两个峰，横扫 $s$：

| guidance scale $s$ | 论文 $w$ | 少数峰比例 | 均值 | 标准差 |
|---:|---:|---:|---:|---:|
| 1.0 | 0.0 | 39.8% | 1.397 | 0.495 |
| 2.0 | 1.0 | 29.9% | 1.328 | 0.453 |
| 5.0 | 4.0 | 11.8% | 1.073 | 0.365 |
| 7.5 | 6.5 | 5.6% | 0.959 | 0.277 |
| 15.0 | 14.0 | 0.5% | 0.872 | 0.087 |

三件事同时发生：

1. **少数峰被压掉**：40% → 5.6% → 0.5%，这就是「模式坍缩」的玩具版；
2. **标准差缩小**：0.495 → 0.087，分布变尖；
3. **均值偏移**：从 1.40 掉到 0.87，强引导会把样本推向条件密度更高的方向，甚至越过峰的边界。

> **坑**：$s=7.5$ 不是「质量更好」，而是「更符合条件、更少多样性」。在人像/产品图上，它让皮肤和材质更「干净」；在需要多样性的创作上，它是负资产。

## 3. 过饱和与 CFG rescale

CFG 的一个副作用是**过饱和**：$w$ 大时，像素值被推出训练分布，颜色变得刺眼。原因可以从 $x_0$ 的估计看出来：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\tilde\varepsilon}{\sqrt{\bar\alpha_t}}
$$

引导把 $\tilde\varepsilon$ 往条件方向推，$\hat x_0$ 的幅值也跟着变大。一个常见的补丁是 **CFG rescale**：把引导后的 $\hat x_0$ 按标准差缩回：

$$
\hat x_0\leftarrow \hat x_0\cdot\frac{\mathrm{std}(x_t)}{\mathrm{std}(\hat x_0)}
$$

它不是理论推导出来的，而是经验修正；能缓解过饱和，但会改变引导强度。

更进一步的做法是**动态 CFG**：在采样早期用大 $w$ 定构图，后期用小 $w$ 修细节；或者按 $\hat x_0$ 的范数自适应调 $w$。这些技巧在 2024–2026 的模型里很常见，但本质都是在「锐化」和「过饱和」之间找折中。

## 4. 负提示词：CFG 的另一个用法

把无条件项换成「负提示词」的预测：

$$
\tilde\varepsilon=(1+w)\,\varepsilon_\theta(x_t,c_{\text{pos}})-w\,\varepsilon_\theta(x_t,c_{\text{neg}})
$$

就得到负提示词。它不是新公式，只是把 $\varnothing$ 替换成另一个条件；效果是「远离 $c_{\text{neg}}$ 的方向」。这也是为什么负提示词有时会「误伤」：CFG 的线性外推不区分「不想要的特征」和「与它相关的特征」。

## 5. 训练侧的三个细节

1. **条件 dropout 的比例**：常见 10%。太少，无条件预测不准；太多，条件信号被浪费。CFG 论文用的是 10% 左右。
2. **空条件的 embedding**：不是零向量，而是一个可学习的 null token；不同实现里可能是 `""`、`<uncond>` 或单独的参数。
3. **训练/推理的 guidance 一致性**：蒸馏模型（LCM/DMD）通常把 guidance **烘进**学生，推理时不能再调 $w$；想改强度要重新蒸馏。这是少步模型可控性差的根源之一。

## 6. 小结

- classifier guidance 用外部分类器的梯度，CFG 用条件/无条件预测的差；
- CFG 的 $w$ 是增量，代码里的 `guidance_scale` 是 $1+w$；
- 引导的本质是 $p_w\propto p_c^{1+w}/p^w$，锐化条件、压窄分布；
- 代价是双倍前向、多样性下降、过饱和；rescale 和动态 CFG 是补丁；
- 蒸馏模型把 guidance 烘进权重，换强度要重训。

下一篇换空间：与其在像素上算，不如先压到 latent 再算。[Latent Diffusion]({{< relref "diffusion-05-latent-diffusion" >}})。
