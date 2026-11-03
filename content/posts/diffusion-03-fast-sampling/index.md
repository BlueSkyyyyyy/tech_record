---
title: "扩散模型精读（三）快速采样：DDIM、DPM-Solver 与 10 步的极限"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 4
series: ["diffusion"]
tags: ["扩散模型", "系列", "DDIM", "DPM-Solver", "采样器", "加速"]
categories: ["LLM算法"]
---

上一篇把 DDPM 放进了连续时间，得到一条 probability flow ODE。这一篇只算一笔账：**同一条 ODE，为什么 DDPM 要 1000 步，DDIM 只要 20–100 步，DPM-Solver 只要 10–20 步？**

答案分两层：DDIM 改的是**离散化的对象**（从随机链改成确定性映射），DPM-Solver 改的是**数值格式**（从一阶 Euler 改成指数积分器）。两者都不重新训练模型。

## 1. DDIM：不改模型，改采样链

DDPM 的逆向链是马尔可夫的，每步都要注入噪声。DDIM（Song et al., 2020）构造了一个**非马尔可夫**的前向过程，它的边缘分布仍然是 $q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}x_0,(1-\bar\alpha_t)I)$，但条件后验多了一个参数 $\sigma_t$：

$$
q_\sigma(x_{t-1}\mid x_t,x_0)
=\mathcal{N}\left(
\sqrt{\bar\alpha_{t-1}}\,x_0
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\frac{x_t-\sqrt{\bar\alpha_t}x_0}{\sqrt{1-\bar\alpha_t}},
\ \sigma_t^2 I
\right)
$$

采样时，先用模型估出 $\varepsilon$，反解 $x_0$：

$$
\hat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\,\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}
$$

再代回：

$$
x_{t-1}=\sqrt{\bar\alpha_{t-1}}\,\hat x_0
+\sqrt{1-\bar\alpha_{t-1}-\sigma_t^2}\;\varepsilon_\theta(x_t,t)
+\sigma_t z
$$

$\sigma_t=0$ 就是 DDIM（确定性）；$\sigma_t=\sqrt{\tilde\beta_t}$ 就退回 DDPM。通常写成 $\sigma_t=\eta\sqrt{\tilde\beta_t}$，$\eta\in[0,1]$ 控制随机性。

DDIM 论文的报告是：**20–100 步**就能达到 1000 步 DDPM 的质量，wall-clock 快 **10–50 倍**。注意它没有让单步变便宜，只是让步数变少。

为什么可以少？因为 $\eta=0$ 时，采样是一个确定性映射，每一步的误差不会被新注入的噪声掩盖；同时它对应 PF ODE 的一个**指数积分**形式，而不是朴素 Euler。换句话说，DDIM 已经偷偷用上了 ODE 的结构。

## 2. DPM-Solver：在 log-SNR 空间做指数积分

DPM-Solver（Lu et al., 2022）把 VP 的 PF ODE 写到半 log-SNR 坐标 $\lambda=\frac12\log(\bar\alpha_t/(1-\bar\alpha_t))=\log(\sqrt{\bar\alpha_t}/\sqrt{1-\bar\alpha_t})$ 下。在这个坐标里，ODE 变成

$$
\frac{dx}{d\lambda}=a(\lambda)x+b(\lambda)\,\varepsilon_\theta(x,\lambda)
$$

的形式，其中 $a,b$ 有闭式。对线性项做**指数积分**（把 $a(\lambda)$ 的吸收进积分因子），对非线性项做多步/多级近似，就得到 1/2/3 阶的 DPM-Solver。

论文的数字：CIFAR-10 上 **10 NFE 做到 FID 4.70，20 NFE 做到 2.87**，比此前的 training-free 采样器快 4–16 倍。这里的 NFE 是**网络前向次数**，不是步数；DPM-Solver-2 每步 2 次，所以 10 NFE 可能只有 5 步。

一个容易混淆的点：DPM-Solver 有 `DPM-Solver` 和 `DPM-Solver++` 两代。后者假设模型输出 $x_0$ 而不是 $\varepsilon$，数值更稳，也是 diffusers 里 `DPM++ 2M` 的来源。**换求解器前先确认模型的 prediction type**，否则公式里的系数会对不上。

## 3. 误差从哪来

少步采样的误差可以拆成三项：

1. **离散化误差**：$p$ 阶格式的全局误差是 $O(\Delta t^p)$。步数越少，$\Delta t$ 越大，误差按幂次涨。
2. **刚性（stiffness）**：$t\to0$ 时 $\sqrt{\bar\alpha_t}\to0$，$x_0$ 的反解要除以它，ODE 的 Lipschitz 常数变大。均匀步长在低噪声端会不够用，这就是为什么很多求解器在 $t$ 小的地方加密。
3. **模型误差**：网络在训练分布外的高噪声/低噪声段外推能力差。步数少时，每一步的跨度大，模型被要求在没见过的 $(x,t)$ 上预测，误差会被放大。

三项的相对大小决定了「多少步够用」。在玩具上（解析 score，无模型误差），10 步就饱和；在真实模型上（有模型误差），20–30 步才稳。**这也是为什么「10 步出图」通常要靠蒸馏，而不是换求解器。**

## 4. 求解器对照

| 求解器 | 阶数 | NFE/步 | 典型 NFE | 备注 |
|---|---:|---:|---:|---|
| Euler | 1 | 1 | 20–50 | 最简单，低步数误差大 |
| Heun | 2 | 2 | 20–40 | 2 阶，但每步 2 次前向 |
| DDIM | 1（指数积分） | 1 | 20–100 | $\eta=0$ 确定性 |
| DPM-Solver-2 | 2 | 2 | 10–20 | 半线性指数积分 |
| DPM-Solver++ 2M | 2 | 1–2 | 10–25 | 多步，假设 $x_0$ 输出 |
| UniPC | 2–3 | 1–2 | 10–20 | 预测-校正统一框架 |

「步数」和「NFE」必须分开算。一个 20 步的 Heun 是 40 NFE；如果再加 CFG 的双 batch，实际前向是 80 次。论文里说「20 步」时，先确认它指的是哪种。

## 5. 实践中的几条经验

- **SD 1.5 / SDXL**：DDIM 或 PNDM 20–30 步是默认档；少于 15 步开始糊，少于 10 步结构会崩。
- **蒸馏模型**：LCM 2–4 步、DMD 1 步，属于另一个体系（第八篇），不能拿 DDIM 的步数直接比。
- **$\eta>0$ 不一定更好**：加噪声能增加多样性，但会引入方差；在少步时通常用 $\eta=0$。
- **respacing 要重算方差**：DDPM 的朴素跳步（第一篇的 20 步实验）会让样本收缩；DDIM 的 respacing 之所以 work，是因为它在新的时间格点上重新算了 $\bar\alpha$ 和系数，而不是复用旧步长。
- **求解器要和 prediction type 匹配**：$\varepsilon$-prediction 用 DDIM/DPM-Solver，$x_0$-prediction 用 DPM-Solver++，$v$-prediction 用 flow 的 Euler/Heun。

## 6. 玩具上的数字

`sampler_zoo.py` 的结果（sliced W2，越小越好）：

```text
DDIM   5 步 0.234 | 10 步 0.092 | 20 步 0.091 | 50 步 0.088
Euler  5 步 0.197 | 10 步 0.094 | 20 步 0.082 | 50 步 0.084
Heun   5 步 0.156 | 10 步 0.095 | 20 步 0.080 | 50 步 0.082
```

两个结论：

1. **10 步是这道题的饱和点**。再往上加步数，误差被 8 个峰的有限宽度和样本估计噪声盖住。
2. **Heun 的 5 步（10 NFE）不如 DDIM 的 10 步（10 NFE）**。高阶格式在粗步长下不一定赢，因为轨迹在低噪声端太刚性。

> **类比的边界**：把少步采样类比成「用更少的直线段逼近曲线」只对了一半。扩散的 ODE 不是一条固定曲线，它的向量场依赖模型；步数少时，你不仅在逼近积分，还在逼模型去它没见过的区域做预测。**离散化误差和模型误差是两笔账，不能只用「阶数更高」来还。**

下一篇处理另一个乘数：条件。guidance 让每一步的 NFE 翻倍，也让「保真度 vs 多样性」变成可调的旋钮：[从 classifier guidance 到 CFG]({{< relref "diffusion-04-guidance" >}})。
