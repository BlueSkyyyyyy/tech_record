---
title: "扩散模型精读（二）分数与 SDE：DDPM、SDE、ODE 是同一件事"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 3
series: ["diffusion"]
tags: ["扩散模型", "系列", "Score SDE", "随机微分方程", "NCSN", "ODE"]
categories: ["LLM算法"]
---

上一篇的 DDPM 是离散的：1000 个 $\beta_t$、1000 次采样。这一篇把它放进连续时间，会看到一个更统一的图景：

- DDPM 的 $\varepsilon$ 预测，本质上是在学**分数** $\nabla\log p_t(x)$；
- NCSN 的退火 Langevin，和 DDPM 是同一个东西的两个端点（VE vs VP）；
- DDIM 的确定性采样，对应一条 **probability flow ODE**；
- 所谓「扩散模型」和「score-based model」，在连续极限下是同一类模型。

把这张表记住，后面读任何一篇 diffusion 论文都能快速定位它在哪一行。

## 1. 从 $\varepsilon$ 到 score：只差一个系数

给定 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\varepsilon$，条件分布是

$$
q(x_t\mid x_0)=\mathcal{N}\left(\sqrt{\bar\alpha_t}x_0,\ (1-\bar\alpha_t)I\right)
$$

对 $\log q$ 求梯度：

$$
\nabla_{x_t}\log q(x_t\mid x_0)
=-\frac{x_t-\sqrt{\bar\alpha_t}x_0}{1-\bar\alpha_t}
=-\frac{\varepsilon}{\sqrt{1-\bar\alpha_t}}
$$

而 $p_t(x_t)$ 是 $q(x_t\mid x_0)$ 对数据分布的积分，分数满足

$$
\nabla\log p_t(x_t)=\mathbb{E}\left[\nabla\log q(x_t\mid x_0)\mid x_t\right]
$$

所以只要模型能预测 $\varepsilon$，它就等价于预测分数：

$$
s_\theta(x_t,t)=\nabla\log p_t(x_t)
=-\frac{\varepsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}
$$

这个换算在所有 VP 类模型里都成立。**NCSN 的「去噪分数匹配」和 DDPM 的「$\varepsilon$ 回归」是同一个目标**，只差一个与 $t$ 有关的缩放。

有了分数，采样就可以用 Langevin 动力学：

$$
x\leftarrow x+\eta\,\nabla\log p(x)+\sqrt{2\eta}\,z
$$

只要 $\eta$ 足够小、步数足够多，样本就会收敛到 $p(x)$。问题是：数据分布是多峰的，从随机初始化直接跑 Langevin 会困在某个峰里。NCSN 的解法是**退火**：从大噪声开始，逐步降噪，每个噪声水平跑一段 Langevin。这其实就是扩散模型的采样过程，只是用 Langevin 而不是逆向马尔可夫链。

## 2. 连续时间：前向 SDE 与逆向 SDE

把离散的加噪写成 SDE：

$$
dx=f(x,t)\,dt+g(t)\,dw
$$

前向过程把数据分布 $p_0$ 变成噪声分布 $p_T$。反向过程（Song et al., 2021）是

$$
dx=\left[f(x,t)-g(t)^2\nabla\log p_t(x)\right]dt+g(t)\,d\bar w
$$

其中 $d\bar w$ 是反向时间的布朗运动。只要知道每个时刻的分数，就能从 $p_T$ 采样回 $p_0$。

两个最常用的选择：

| 过程 | $f(x,t)$ | $g(t)$ | 特点 |
|---|---|---|---|
| **VP**（variance preserving） | $-\frac12\beta(t)x$ | $\sqrt{\beta(t)}$ | 方差有界，$t\to\infty$ 时趋于标准高斯；DDPM 的连续版 |
| **VE**（variance exploding） | $0$ | $\sqrt{\frac{d\sigma^2(t)}{dt}}$ | 方差随时间增大；NCSN 的连续版 |

VE 的直觉更简单：不加漂移，只加噪声，噪声尺度 $\sigma(t)$ 一路涨。VP 的直觉更接近 DDPM：信号被 $\sqrt{\bar\alpha_t}$ 衰减，噪声被 $\sqrt{1-\bar\alpha_t}$ 补上，总方差保持在 1 附近。

> **坑**：VE 和 VP 的分数缩放不一样。VE 里 $\sigma(t)$ 就是噪声标准差，$s_\theta=-\varepsilon/\sigma$；VP 里是 $-\varepsilon/\sqrt{1-\bar\alpha_t}$。把两个公式混用，采样会发散或者过平滑。

## 3. Probability flow ODE：把随机性拿掉

同一个 $p_t(x)$ 可以由两类过程生成：

- **逆向 SDE**：每步注入噪声，轨迹随机；
- **probability flow ODE**：

$$
\frac{dx}{dt}=f(x,t)-\frac12 g(t)^2\nabla\log p_t(x)
$$

这个 ODE 的边缘分布和 SDE 完全相同，但轨迹是确定的。DDIM 就是它在离散 VP 下的一个实现（$\eta=0$）；DPM-Solver 是它的高阶数值解。

为什么 ODE 重要？因为**确定性意味着可以用高阶求解器**。SDE 每步都有随机噪声，高阶格式的截断误差会被噪声掩盖；ODE 没有这个问题，可以用 Heun、Runge-Kutta、指数积分器，用更少的步数达到同样的精度。

一张表总结四种「采样」：

| 方法 | 过程 | 随机性 | 典型步数 | 代表 |
|---|---|---|---:|---|
| DDPM | 离散逆向马尔可夫链 | 每步加噪 | 1000 | Ho et al. 2020 |
| DDIM | 离散 PF ODE（$\eta=0$） | 确定性（$\eta>0$ 可加） | 20–100 | Song et al. 2020 |
| 逆向 SDE | 连续随机微分方程 | 有 | 100–2000 | Song et al. 2021 |
| PF ODE + 高阶求解 | 连续确定性 ODE | 无 | 10–50 | DPM-Solver 等 |

## 4. 在玩具上对比：DDIM / Euler / Heun

`code/diffusion/sampler_zoo.py` 用 8 高斯的**解析 score**，把模型误差去掉，只留下离散化误差。三个求解器，同样从 $t=1$ 积到 $t=0$：

| 求解器 | 5 步 | 10 步 | 20 步 | 50 步 |
|---|---:|---:|---:|---:|
| DDIM（$\eta=0$） | 0.234 | 0.092 | 0.091 | 0.088 |
| Euler（1 阶） | 0.197 | 0.094 | 0.082 | 0.084 |
| Heun（2 阶，2 NFE/步） | 0.156 | 0.095 | 0.080 | 0.082 |

数字是 sliced Wasserstein-2（越小越好）。几个观察：

1. **10 步以后基本饱和**。不是求解器不够好，而是 8 个峰的宽度只有 0.08，有限样本估计本身有约 0.08 的底噪。
2. **Heun 的 5 步（10 NFE）打不过 DDIM 的 10 步（10 NFE）**。高阶格式在轨迹刚性强的区间不一定占优：步长太粗时，二阶项也救不回来。
3. **Euler 的 20 步和 Heun 的 20 步几乎一样**。到了这个精度，误差主要来自模型/样本，而不是积分阶数。

> **类比的边界**：把 ODE 采样类比成「沿着梯度场下山」很直观，但下山的目标是某个极小值，扩散的目标是**整个分布**。PF ODE 的轨迹不是走向单个最优解，而是把噪声样本一一映射到数据样本；不同的初始噪声会落到不同的峰，映射本身是多峰的。

## 5. 连续时间的两个实用记号

读论文时还会遇到两个记号，提前统一：

- **log-SNR**：$\lambda=\log(\bar\alpha_t/(1-\bar\alpha_t))$。$t=0$ 时 $\lambda=+\infty$，$t=T$ 时 $\lambda=-\infty$。DPM-Solver 的推导就是在 $\lambda$ 空间做的，因为 VP 的 ODE 在 $\lambda$ 下变成半线性形式，可以用指数积分器。
- **时间方向**：有的论文 $t=0$ 是数据、$t=1$ 是噪声（flow matching 常用）；有的反过来（DDPM 常用）。**读公式前先确认方向**，否则 score 和 velocity 的符号会反。

## 6. 这一篇的结论

- $\varepsilon$ 预测 = score 预测，只差 $-\sqrt{1-\bar\alpha_t}$；
- DDPM / NCSN 是 VP / VE 的离散版；DDIM 是 PF ODE 的离散版；
- SDE 和 ODE 边缘分布相同，但 ODE 可以用高阶求解器，这是少步的理论基础；
- 采样器的收益在 10–20 步后饱和，再往下要靠蒸馏或更好的路径。

下一篇专门算「少步」这笔账：[DDIM 为什么能 10 倍加速，DPM-Solver 又省在哪]({{< relref "diffusion-03-fast-sampling" >}})。
