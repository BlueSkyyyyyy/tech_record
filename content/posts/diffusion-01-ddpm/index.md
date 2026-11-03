---
title: "扩散模型精读（一）DDPM：一千步去噪的账"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 2
series: ["diffusion"]
tags: ["扩散模型", "系列", "DDPM", "去噪扩散", "生成模型"]
categories: ["LLM算法"]
---

DDPM（Ho et al., 2020）通常被介绍成「加噪再还原」。这句话没错，但它解释不了三件事：为什么训练目标最后变成了一句 MSE、为什么是 1000 步、以及为什么这 1000 步里有一大半其实没在干活。

这一篇把 DDPM 的账从头算一遍。核心结论先放在这里：

1. DDPM 的损失不是拍脑袋的 MSE，而是变分下界在「固定方差 + $\varepsilon$ 参数化」下的简化；
2. 线性噪声日程下，只有 **259/1000** 步的信噪比不低于 0 dB，后 175 步几乎是纯噪声；
3. 因此 1000 步里真正决定样本的窗口很窄，这既解释了 DDIM 为什么能加速，也解释了朴素跳步为什么会把样本「压小」。

## 1. 前向：把数据揉进噪声

前向过程是一个固定的马尔可夫链，每步只加一点高斯噪声：

$$
q(x_t\mid x_{t-1})=\mathcal{N}\left(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\right)
$$

记 $\alpha_t=1-\beta_t$，$\bar\alpha_t=\prod_{s=1}^{t}\alpha_s$。利用高斯分布的可加性，任意时刻的边缘分布有闭式：

$$
q(x_t\mid x_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\right)
$$

等价地，用重参数化写：

$$
x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I)
$$

DDPM 取 $T=1000$，$\beta_t$ 从 $10^{-4}$ 线性增到 $0.02$。这两个数不是随便定的：数据被缩放到 $[-1,1]$，$\beta_1=10^{-4}$ 保证第一步几乎不破坏信号，$\beta_T=0.02$ 保证最后一步的 $\bar\alpha_T$ 足够小。

但「足够小」到底是多小？把日程算出来看：

| $t$ | $\bar\alpha_t$ | $\sqrt{\bar\alpha_t}$ | $\sqrt{1-\bar\alpha_t}$ | SNR (dB) |
|---:|---:|---:|---:|---:|
| 100 | 0.897 | 0.947 | 0.321 | 9.40 |
| 300 | 0.396 | 0.630 | 0.777 | −1.83 |
| 500 | 0.0786 | 0.280 | 0.960 | −10.69 |
| 700 | 0.00697 | 0.0835 | 0.9965 | −21.54 |
| 900 | 2.75e−4 | 0.0166 | 0.99986 | −35.60 |
| 1000 | 4.04e−5 | 0.00635 | 0.99998 | −43.94 |

SNR 定义为 $\bar\alpha_t/(1-\bar\alpha_t)$。按 0 dB 划一条线：

- **只有 259 步**的 SNR ≥ 0 dB，即信号不弱于噪声；
- **175 步**的 $\bar\alpha_t\lt 10^{-3}$，输入几乎是纯噪声，模型能从中学到的「图像内容」极少；
- 剩下 566 步是过渡区，真正在做「从结构到细节」的决策。

这不是说后 175 步没用——它们决定了全局布局和颜色统计。但从**每步的边际信息量**看，1000 步是高度不均匀的。后面 DDIM 和 DPM-Solver 的加速，本质上就是把这条不均匀的日程重新分配。

## 2. 从 ELBO 到一句 MSE

DDPM 是一个潜变量模型：$p_\theta(x_0)=\int p(x_T)\prod_t p_\theta(x_{t-1}\mid x_t)\,dx_{1:T}$。直接算似然不可行，优化 ELBO：

$$
\mathcal{L}=\mathbb{E}_q\left[
\mathrm{KL}(q(x_T\mid x_0)\|p(x_T))
+\sum_{t>1}\mathrm{KL}(q(x_{t-1}\mid x_t,x_0)\|p_\theta(x_{t-1}\mid x_t))
-\log p_\theta(x_0\mid x_1)
\right]
$$

关键是把后验 $q(x_{t-1}\mid x_t,x_0)$ 写成闭式。因为前向是高斯，后验也是高斯：

$$
q(x_{t-1}\mid x_t,x_0)=\mathcal{N}\left(\tilde\mu_t(x_t,x_0),\ \tilde\beta_t I\right)
$$

$$
\tilde\mu_t=\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0
+\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t,
\qquad
\tilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t
$$

把 $p_\theta(x_{t-1}\mid x_t)$ 也设成同方差的高斯，KL 就只剩均值差：

$$
\mathrm{KL}=\frac{1}{2\tilde\beta_t}\left\|\tilde\mu_t-\mu_\theta(x_t,t)\right\|^2+C
$$

到这里还是「预测后验均值」。DDPM 的关键一步是换参数化：不直接回归 $\tilde\mu_t$，而是让网络预测噪声 $\varepsilon$。由 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$ 反解 $x_0$，再代回 $\tilde\mu_t$，可以得到

$$
\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)
$$

代进 KL，均值差变成 $\varepsilon$ 差乘一个与 $t$ 有关的系数。DDPM 把这个系数**丢掉**，得到

$$
\mathcal{L}_{\text{simple}}
=\mathbb{E}_{t,x_0,\varepsilon}\left[
\left\|\varepsilon-\varepsilon_\theta\left(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon,\ t\right)\right\|^2
\right]
$$

这就是那句 MSE。它不是 ELBO 的严格上界，但 Ho 等人发现样本质量更好——因为 ELBO 的系数在 $t$ 小时很大、在 $t$ 大时很小；把它丢掉相当于**压低低噪声步的权重**，让训练更关注中高噪声段。后来的 min-SNR 加权、$v$-prediction 都是在重新设计这个权重。

> **坑**：$\varepsilon$-prediction 和 $x_0$-prediction 在精确算术下等价，但**误差放大不同**。给定 $\varepsilon$ 的误差 $\delta$，反解出的 $x_0$ 误差是
> $$
> \delta_{x_0}=\frac{\sqrt{1-\bar\alpha_t}}{\sqrt{\bar\alpha_t}}\,\delta_\varepsilon
> =\frac{1}{\sqrt{\mathrm{SNR}}}\,\delta_\varepsilon
> $$
> 在 $t=900$ 时放大约 **60 倍**，$t=1000$ 时约 **157 倍**。所以高噪声段预测 $x_0$ 再 clip 到 $[-1,1]$ 是个很粗的补丁：它把误差藏起来了，但梯度也被截断。

## 3. 采样：从 $x_T$ 往回走

训练完，采样从 $x_T\sim\mathcal{N}(0,I)$ 开始，按

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\varepsilon_\theta(x_t,t)\right)
+\sigma_t z,\qquad z\sim\mathcal{N}(0,I)
$$

迭代 1000 次。$\sigma_t^2$ 可以取 $\beta_t$，也可以取后验方差 $\tilde\beta_t$；两者都 work，后者在 $t$ 小时更小。DDPM 论文报告，CIFAR-10 无条件生成做到 **FID 3.17、IS 9.46**，在 2020 年是 SOTA。

但 1000 次网络前向的代价是实打实的。更麻烦的是，这 1000 步的**信息分布极不均匀**：前 259 步信号强，后 175 步几乎纯噪声，中间才是主战场。

一个自然的想法是「跳步」：只取 $t=1000,950,\dots,0$，每步仍用原来的公式。这样做的**质量会掉**，DDIM 论文专门分析了这种朴素 respacing 的偏差。在 8 高斯玩具上，这个掉的机制看得很清楚：

| 采样步数 | 8 峰覆盖 | 平均半径 |
|---:|---:|---:|
| 1000 | 8/8 | 2.031 |
| 100 | 8/8 | 1.561 |
| 20 | 8/8 | 1.337 |

数据半径是 2.0。跳步之后峰还在，但样本整体**向中心收缩**——因为跳过的那些步对应的噪声方差没有被正确补偿，采样器把「还没走完的噪声」当成了「已经该收窄的信号」。这不是模型不行，是**离散化日程和采样公式不匹配**。

这也解释了为什么 20 步采样看起来「还行」：8 个峰还在，但分布已经变窄，细节和多样性都损失了。要真正少步，需要换采样器，而不是简单跳步——这是第三篇的主题。

## 4. 玩具实验：把放大系数和收缩都跑出来

`code/diffusion/ddpm_toy.py` 在 2D 的 8 高斯上训一个 3 层 MLP，T=1000，线性日程，训练 3000 步。几个实测数字：

```text
step 3000  loss 0.2291
t= 900  sqrt(ab)=0.0166  x0 误差放大= 60.27x
t=1000  sqrt(ab)=0.0064  x0 误差放大=157.41x
1000 步: 覆盖 8/8 峰, 平均半径 2.031
 100 步: 覆盖 8/8 峰, 平均半径 1.561
  20 步: 覆盖 8/8 峰, 平均半径 1.337
```

训练损失从 0.317 降到 0.229，说明模型确实在学噪声；1000 步采样的半径 2.031 接近数据半径 2.0；跳步后的收缩和理论一致。

`schedules.py` 则把日程的「有效窗口」算出来：

```text
线性日程: SNR >= 0 dB 的步数 259/1000；alpha_bar < 1e-3 的步数 175/1000
cosine  : SNR >= 0 dB 的步数 496/1000；alpha_bar < 1e-3 的步数  21/1000
```

cosine 日程（Nichol & Dhariwal, 2021）把有效步数几乎翻倍，末尾也更干净（$\bar\alpha_T\approx2.4\times10^{-9}$）。这就是为什么后来的模型大多不再用线性日程。

## 5. 几个容易踩的坑

1. **$\beta_t$ 和 $\alpha_t$ 混用**。$\beta_t$ 是方差，$\alpha_t=1-\beta_t$，$\bar\alpha_t$ 是连乘。代码里如果一边用 `betas` 一边用 `alphas_cumprod` 但索引错一位，训练仍然会收敛，只是采样偏色/偏灰，很难查。
2. **采样方差选错**。用 $\beta_t$ 还是 $\tilde\beta_t$ 都能出图，但和训练时的 $q$ 不一致会改变多样性；DDPM 论文 Appendix 对固定方差和学习方差有实验。
3. **高噪声段 clip $x_0$**。clip 能明显改善 FID，但它不是生成过程的一部分；在蒸馏/少步时继续 clip 会把误差带进学生模型。
4. **$T=1000$ 不是必须**。T 越小，训练越快，但离散化误差越大；cosine 日程在 T=1000 下的有效步数更多，等价于「每一步的信息量更均匀」。
5. **损失下降不等于样本变好**。$\mathcal{L}_{\text{simple}}$ 在不同 $t$ 上的权重被改过，loss 曲线和 FID 不是单调关系；调参时要以采样指标为准。

## 6. 小结

- DDPM 的前向有闭式，训练目标可以化到一句 $\varepsilon$ 回归；
- 线性日程的有效窗口只有约 259 步，后 175 步几乎是纯噪声；
- 因此「跳步」会带来分布收缩，少步必须换采样器；
- 高噪声段 $x_0$ 误差放大 $1/\sqrt{\mathrm{SNR}}$，这是 $x_0$-prediction 和 clip 的根源。

下一篇把 DDPM 放进连续时间：同一个东西用 score 和 SDE 写出来，会看到 DDPM、NCSN、DDIM 其实是一张表里的三行：[分数与 SDE]({{< relref "diffusion-02-score-sde" >}})。
