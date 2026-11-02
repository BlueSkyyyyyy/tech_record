---
title: "VLA 精读（三）动作头：diffusion 与 flow matching 到底在算什么"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 4
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "扩散模型", "Flow Matching", "动作头"]
categories: ["LLM算法"]
---

上一篇的架构表里，动作头一栏有四种写法：复用 lm_head、小 MLP、diffusion、flow matching。这一篇只讲后两种，因为它们决定了一个 VLA 的**多峰表达力**和**推理步数**——前者决定它会不会把两个动作平均成一个，后者决定它能不能跑 50 Hz。

先把问题说清楚：给定观测 $o_t$，要建模的是动作 chunk 的分布 $p(a_{t:t+H} \mid o_t)$。这个分布通常是**多峰**的：同一个画面，向左绕和向右绕都能拿起杯子。用 L2 回归，最优解是两个峰的平均，落在物理上不可行的中间地带。

## 1. Diffusion Policy：把动作生成写成去噪

Diffusion Policy 的做法是标准的 DDPM：

**前向**：对干净动作 $a^0$ 逐步加高斯噪声，得到 $a^k$；

**反向**：训练一个网络 $\varepsilon_\theta(a^k, o_t, k)$ 预测加入的噪声，损失是

$$
\mathcal{L} = \mathrm{MSE}\left(\varepsilon^k,\ \varepsilon_\theta(o_t,\ a^0 + \varepsilon^k,\ k)\right)
$$

其中 $\varepsilon^k$ 是对应噪声水平的采样噪声（p.3）。

**采样**：从 $a^K \sim \mathcal{N}(0,I)$ 开始，迭代

$$
a^{k-1} = \alpha\left(a^k - \gamma\,\varepsilon_\theta(a^k,o_t,k) + \mathcal{N}(0,\sigma^2 I)\right)
$$

论文把它解释成「带噪梯度下降」：$\varepsilon_\theta$ 近似能量函数的梯度，$(\alpha,\gamma,\sigma)$ 是随 $k$ 变化的学习率调度（p.2）。

Diffusion Policy 的几个关键工程数字：

- 训练/推理都用 iDDPM 的 **100 步**（p.16）；真机上用 DDIM 把推理降到 **10 步**，RTX 3080 上约 **0.1 s** 延迟（p.4）。
- 动作块 $T_a=8$ 常用，$T_p=16$；对 ≤4 步的模拟延迟仍保持峰值性能（p.8）。
- 视觉编码器是 **ResNet-18、无预训练**，global average pooling 换成 spatial softmax（p.4）。这不是 VLA，没有 VLM 骨干。

Diffusion Policy 证明了动作头表达力的价值：在 Block Push 这种多峰任务上，比 LSTM-GMM 高 **32%**；在 Kitchen 的 p4 指标上高 **213%**（p.8）。但它的代价也清楚：**100 步去噪**，实时性靠 DDIM 或小网络硬扛。

Octo 把去噪搬进了一个 3 层 MLP 小头，20 步，backbone 只前向一次（p.5）。这是「大 backbone + 小 diffusion 头」的典型折中：表达力够用，延迟可控。

## 2. Flow matching：把去噪拉直

Flow matching 和 diffusion 建模的是同一个东西——从噪声到数据的概率路径——但路径选得更「直」。

π0 的定义（p.5）：

$$
q(A^\tau \mid A) = \mathcal{N}\left(\tau A,\ (1-\tau)I\right)
$$

即插值

$$
A^\tau = \tau A + (1-\tau)\varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,I)
$$

- $\tau=0$：纯噪声；
- $\tau=1$：干净动作；
- 速度场目标：$u(A^\tau \mid A) = A - \varepsilon$；
- 损失：$\mathcal{L}_\tau(\theta) = \mathbb{E}\left\|v_\theta(A^\tau,o_t) - u\right\|^2$。

采样从 $A^0 \sim \mathcal{N}(0,I)$ 开始，前向 Euler：

$$
A^{\tau+\delta} = A^\tau + \delta\, v_\theta(A^\tau, o_t),\qquad \delta = 1/K
$$

π0 用 **10 步**，$\delta=0.1$（p.5）。

### 2.1 代码里的符号是反的

打开 openpi 的实现，你会看到另一种写法：

```python
# openpi/models/pi0.py:197-200
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
```

这里 $t$ 是**噪声端为 1、动作端为 0**：

$$
x_t = t\cdot\varepsilon + (1-t)\cdot A,\qquad u_t = \varepsilon - A
$$

和论文的 $\tau$ 差一个 $t = 1-\tau$，路径是同一条，方向相反。LeRobot 把这个约定写成了枚举，注释很清楚：

```python
# lerobot/policies/common/flow_matching.py:58-70
NOISE_AT_ONE = "noise_at_one"
"""openpi (pi0, pi05, smolvla, eo1): x_1 is noise, integrate t: 1 -> 0."""

NOISE_AT_ZERO = "noise_at_zero"
"""groot / evo1 / wall_x: x_0 is noise, integrate t: 0 -> 1."""
```

这不是吹毛求疵。**训练和推理必须用同一个约定**；换模型、换代码库时符号搞反，学到的速度场方向就是错的。SmolVLA 就是另一个例子：它的插值同样是 $\tau A + (1-\tau)\varepsilon$，但速度目标写成 $u=\varepsilon-A$（p.4），和 π0 的 $A-\varepsilon$ 相反。两篇论文都对，只是参数化不同。

### 2.2 为什么 10 步就够

Flow matching 的采样步数取决于路径有多「直」。如果速度场沿路径近似恒定，Euler 积分一步就能走完；如果路径弯曲，就需要更多步。

π0 的线性插值路径在**期望意义下**是直的：给定 $A$ 和 $\varepsilon$，$A^\tau$ 是一条直线，速度 $A-\varepsilon$ 不随 $\tau$ 变。但网络看到的是边缘分布，不同 $(A,\varepsilon)$ 配对会让条件速度在同一个 $x$ 处指向不同方向，边缘速度场仍然是弯的。10 步是在「边缘场弯曲程度」和「延迟」之间的经验折中。

对比一下：

| 方法 | 训练目标 | 推理步数 | 随机性 | 代表 |
|---|---|---|---|---|
| DDPM | 噪声 $\varepsilon$ | 100（DDIM 10） | 每步加噪 | Diffusion Policy |
| DDIM | 噪声 $\varepsilon$ | 10–20 | 可 deterministic | DP 真机 |
| Flow matching | 速度 $v$ | 10 | 确定性 ODE | π0、π0.5、GR00T |
| 一致性/蒸馏 | 直接映射 | 1–4 | 确定性 | 2026 的加速方向 |

## 3. L1 回归：最简单，也最容易被误解

OpenVLA-OFT 把动作头换成了一个 MLP + **L1 回归**，理由是：L1 比 L2 更接近中位数，在多峰不那么极端的任务上更稳，而且一次前向就够（p.4）。

实验结果很漂亮：LIBERO 从 OpenVLA 的 76.5% 提到 **97.1%**，吞吐 **26×**（p.6）。但论文自己承认局限：L1 通过 "encouraging the policy to learn the median mode" 平滑噪声，但面对真正的多峰分布 "may struggle to accurately model truly multimodal action distributions"（p.10）。

这句话很重要。OFT 的 97.1% 是在 LIBERO 上拿到的，而 LIBERO 的任务大多是**单峰**的（同一个任务，演示做法高度一致）。在多峰的真机任务上，L1 会退化成「平均动作」，diffusion/flow 的表达力仍然必要。

所以「L1 打败 diffusion」是一个**任务相关**的结论，不是普遍规律。

## 4. 一个双峰玩具实验

`code/vla/flow_matching_toy.py` 把这件事做成了最小可复现实验：

- 一维动作 $x$，真实值在 $-1.5$ 或 $+1.5$；
- 观测 $obs$ 是模式的带噪观测，当 $obs \approx 0$ 时后验双峰；
- 三组对照：flow matching（10 步 Euler）、L2 回归、flow 单步。

结果（歧义区，理想情况应一半左一半右、几乎不落中间）：

```
方法                  左峰      右峰      中间     |x|均值
flow matching   54.5%   45.5%    0.0%     1.457
L2 回归            0.0%    0.0%   77.2%     0.324
flow 单步          0.0%    0.0%   79.0%     0.335
```

三个结论：

1. **L2 回归 77% 的样本落在中间**——它学到的就是两个峰的平均，一个物理上不存在的动作。
2. **flow matching 两个峰各占一半，0% 落中间**——多峰表达力来自「从不同噪声出发走不同路径」，不是来自网络容量。
3. **flow 单步也塌到中间**——多峰表达力需要**多步积分**。一步 Euler 等价于一个确定性映射，仍然会把不同噪声映射到同一个平均点。

> **类比的边界**：flow matching 像给每个噪声样本发一张「地图」，让它沿自己的方向走到某个峰；L2 回归像把所有地图平均成一张，最后指向两峰之间的空地。地图越多步越准，但每一步都要时间——这就是为什么 VLA 的延迟和动作头步数直接绑定。

## 5. 动作头的选择矩阵

| 任务特征 | 推荐动作头 | 理由 |
|---|---|---|
| 单峰、结构化、追求低延迟 | L1 / 小 MLP | 一次前向，够用 |
| 多峰、精细、可接受 10 步 | flow matching | 表达力 + 10 步延迟 |
| 多峰、需要随机性、可接受 20–100 步 | diffusion | 成熟、稳定 |
| 高频、长 chunk、要复用 VLM 词表 | FAST / DCT+BPE + 自回归 | 压缩时序冗余 |
| 极低延迟（<10 ms） | 蒸馏 / 单步 flow | 2026 年的前沿，精度换速度 |

## 6. 2026 年的新变量：一步生成

2026 年出现了一批「一步动作头」的工作，核心思路是把多步去噪蒸馏成一个映射，或者用 drifting loss 直接学单步生成。

GR00T N1.7 的一个实验很能说明取舍：把 4 步去噪的头换成一步 drifting head，单次前向从 **45.3 ms 降到 5.0 ms**，但 LIBERO-Spatial 从 **97.6% 掉到 64.0%**（2609.18108）。**9 倍的延迟改善，换来 33 个点的精度损失**——在需要精细空间推理的任务上，这个交易不划算。

这也解释了为什么 2026 年的主流仍然是 4–10 步：不是大家不想快，是一步生成还撑不住多峰 + 精细控制。

## 小结

- Diffusion 和 flow matching 都在学「从噪声到动作」的路径；flow 的路径更直，所以 10 步就够，diffusion 通常要 20–100 步。
- π0 的 $\tau$ 和代码里的 $t$ 是反的（$t=1-\tau$），训练/推理必须用同一个约定；SmolVLA 的速度目标符号也和 π0 相反。
- L1 回归在单峰任务上又快又好（OFT：LIBERO 97.1%、26×），但多峰时会学中位数，论文自己承认这一点。
- 双峰玩具实验：flow matching 两峰各 54.5%/45.5%、0% 落中间；L2 回归 77.2% 落中间；单步 flow 也塌。
- 一步生成在 2026 年仍是「延迟换精度」：GR00T N1.7 的 drifting head 快 9 倍，但 Spatial 掉 33 个点。

下一篇进入 π0 的源码：[两个 Gemma、一条 block-causal mask，以及 73 ms 是怎么算出来的]({{< relref "vla-04-pi0" >}})。
