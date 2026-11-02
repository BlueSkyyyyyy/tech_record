---
title: "VLA 精读（七）数据：OXE 的 1M 轨迹为什么不够"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 8
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "Open X-Embodiment", "数据", "归一化", "跨本体"]
categories: ["LLM算法"]
---

VLA 论文最喜欢引用的数据规模是「1M+ trajectories」。这个数字来自 Open X-Embodiment（OXE），听起来很大，但把它拆开看，真正能用的部分远没有那么多。

这一篇讲三件事：

1. OXE 到底有多大，以及为什么 1M 轨迹仍然不够；
2. 归一化的三个坑：q01/q99 与 mean/std、padding、动作表示；
3. 跨本体的动作空间怎么对齐——从「pad 到 32」到「canonical 55 维」。

## 1. OXE：1M 轨迹的构成

OXE 的规模（p.2–3）：

| 指标 | 数字 |
|---|---|
| 轨迹 | **1M+** |
| 机器人 | **22 种** |
| 机构 | **21 家** |
| 数据集 | **60 个** |
| 技能 | **527 种** |
| 任务 | **160,266 个** |

本体清单包括 Franka、Google Robot、xArm、WidowX、Jackal、Hello Stretch、Kinova Gen3、Sawyer、Kuka iiwa、UR5、PR2、DLR SARA、Jaco 2、Unitree A1、xArm Bimanual、Cobotta、DLR EDAN、PAMY2、Fanuc Mate、RC Car、TurtleBot 2、Baxter（p.3）。

但 OXE 的实验只用了 **9 个本体**，不是 22 个。论文的理由是「数据集在实验后仍在扩充」（p.5）。也就是说，**「22 种机器人」是数据集覆盖，不是训练覆盖**。

数据格式是 RLDS（tfrecord），支持不同相机数、深度、点云（p.3）。每个数据集只取**一个** canonical view（p.4）。

## 2. 为什么 1M 轨迹不够

### 2.1 大数据域上，跨本体混合反而更差

OXE 的 Table I（p.5）：

| 评测 | 机器人 | 原方法 | RT-1 | RT-1-X | RT-2-X (55B) |
|---|---|---|---|---|---|
| Bridge @ IRIS | WidowX | 13% | 40% | **27%** | 50% |
| Bridge @ RAIL | WidowX | 13% | 30% | **27%** | 30% |
| RT-1 6 skills @ Google | Google Robot | — | 92% | **73%** | 91% |

RT-1-X 是在 9 个本体上联合训练的，但在 Bridge 和 Google Robot 这两个大数据域上**不如单本体 RT-1**。论文的解释是 underfit：混合数据太大，模型容量被稀释，每个域都学不精。

这直接反驳了「数据越多越好」的直觉。**跨本体混合的收益，取决于本体之间的相似性和模型容量，不是简单的加法。**

### 2.2 +50% 和 3× 都是相对数

两个被引用最多的数字：

- **RT-1-X +50%**：相对「各数据集原方法」和「单本体 RT-1」的**平均**成功率提升（p.5）。是相对提升，不是 50 个百分点。
- **RT-2-X ~3×**：相对 RT-2（同 55B、只用 Google Robot 数据）在 **emergent skills** 上的提升：**27.3% → 75.8%**（≈2.78×，p.6）。也是相对倍数。

RT-2-X 的 3× 只发生在 **5 个 emergent skills** 上，这 5 个任务出现在 Bridge（WidowX）数据里、不在 Google Robot 数据里，却在 Google Robot 上评测（p.6）。去掉 Bridge 数据后，数字掉到 **42.8%**（Table II 第 3 行）。所以「3×」证明的是**跨本体数据能迁移语义技能**，不是「RT-2-X 全面比 RT-2 强 3 倍」。

### 2.3 数据质量：论文自己承认的五个问题

OXE 论文的「限制」部分写得很坦诚（p.4）：

1. **只做粗对齐**：统一成 7 维末端动作（x, y, z, roll, pitch, yaw, gripper）。
2. **相机不可对齐**："camera observations still vary substantially across datasets, e.g. due to differing camera poses … or differing camera properties."
3. **坐标系不对齐、绝对/相对混用**：允许绝对位置、相对位置或速度，"the same action vector may induce very different motions for different robots."
4. **动作离散化后再按本体反归一化**："We normalize each dataset's actions prior to discretization … de-normalized differently depending on the embodiment." 这意味着 256 个 bin 的物理含义逐本体不同。
5. **覆盖不足**：不含传感/驱动差异很大的机器人，不研究对新本体的泛化；训练混合只覆盖 9/22 个本体。

论文没有给逐数据集的成功率标注、reward 或质量分级。也就是说，**1M 轨迹里混着成功、失败、低质量、重复的演示，没有标签告诉模型哪些该学**。这是 OXE 最大的结构性缺陷，也是后来 DROID、RoboMIND 等数据集强调「质量筛选」的原因。

## 3. 归一化：三个坑

### 3.1 q01/q99 还是 mean/std

OpenVLA、π0、π0.5 都用 **quantile 归一化**：把 [q01, q99] 映射到 [-1, 1]。openpi 的实现：

```python
# openpi/transforms.py:141-145
def _normalize_quantile(self, x, stats: NormStats):
    q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

为什么不用 mean/std？`code/vla/norm_stats.py` 用一个带离群点的夹爪维度算了一笔账：

```
方法               |z|<=1  |z|<=0.5    z.std   |z|max
mean/std         72.8%     47.0%    1.000    12.86
q01/q99          98.0%     72.0%    1.227    39.61

夹爪维度（含 0.5% 的 20.0 尖峰）：
mean/std: mean=0.611 std=1.508
q01/q99 : q01=0.011 q99=0.995
正常样本被压到 |z|<=0.5 的比例：mean/std=100.0%，q01/q99=50.1%
```

- mean/std 下，0.5% 的尖峰把 std 拉到 1.508，**所有正常夹爪值都被压到 |z|≤0.5**，等效分辨率掉一半；
- q01/q99 下，q99 仍然是 0.995（正常范围），正常值分布在 [-1,1]，分辨率保住。

代价是 q01/q99 会把 2% 的极端值映射到 [-1,1] 之外，推理时模型输出被限制在 [-1,1]，这些值**不可达**。对于夹爪尖峰这种传感器错误，这是好事；对于真实的大动作，这是信息损失。

> **类比的边界**：mean/std 像用平均身高做尺子，一个 2.2 米的人会把所有正常人的刻度压扁；q01/q99 像用第 1 和第 99 百分位做尺子，极端值被截断，但 98% 的人刻度均匀。前者对离群点敏感，后者对极端值不敏感——没有免费午餐。

### 3.2 padding 会污染统计量

跨本体拼接时，7 维单臂要 pad 到 32 维（π0）或 132 维（GR00T）。如果对 padded 的全 32 维算统计量，零填充会把均值/方差带偏：

```
 dim      全维均值    mask均值     全维std  mask std
   0     0.499     0.499     0.191     0.191
   ...
   6     0.499     0.499     0.199     0.199
第 7~31 维：全维统计 mean=0.00 std=0.00，这些维度根本没有数据。
```

单看 `mean_all[:7]` 和 `mean_masked` 是一样的（因为 padding 是 0，不影响前 7 维的均值），但**如果你把 32 维的统计量当成一个向量用**，后 25 维的 `std=0` 会导致除零或放大噪声。正确做法是按 mask 只在有效维度上统计，或者给 padding 维一个显式的 mask，训练/推理都跳过。

代码里 π0 就是这么做的：`NormStats` 的 `mean/std/q01/q99` 只保留有效维度，padding 维不参与归一化。

### 3.3 动作表示的坑：relative/delta 要在归一化之前转换

LeRobot 的文档写得很清楚：π0/π0.5/π0-FAST 默认在归一化**之前**做 relative 转换（`docs/source/action_representations.mdx`）。

为什么顺序重要？假设你先归一化再算 delta：

$$
\delta = \frac{a_t - q01}{q99-q01} - \frac{a_{t-1} - q01}{q99-q01} = \frac{a_t - a_{t-1}}{q99-q01}
$$

看起来等价，但 q01/q99 是从**绝对动作**统计出来的；delta 的分布和绝对动作完全不同（均值接近 0，方差小得多）。用绝对动作的 q01/q99 去归一化 delta，会把 delta 压到一个很窄的区间，等效分辨率大幅下降。正确做法是：先算 delta，再用 **delta 的** q01/q99 归一化。

## 4. 跨本体动作空间：三条路线

### 4.1 pad 到统一维度（π0/π0.5）

π0 把动作/状态 pad 到 **32 维**（代码 `action_dim=32`），论文里最大机器人是 18 维。π0.5 把 state 离散成字符串，维度由字符串长度隐式表达。

优点：实现简单，一个模型吃所有本体。
缺点：高维本体（人形 132 维）会撑爆 token 长度；低维本体的 padding 浪费算力；不同本体的同一维语义不同，模型要靠数据自己学。

### 4.2 每本体一套投影（GR00T）

GR00T 给每个本体一套 state/action 编解码 MLP，主干共享（上一篇详细讲过）。

优点：本体差异显式建模，高维人形也能处理。
缺点：本体数量增长时参数线性增长；新本体需要重新定义 tag 和投影；相近本体共享槽位需要人工设计。

### 4.3 canonical action space（2026 的新趋势）

LingBot-VLA 2.0 定义了一个 **55 维 canonical state/action space**，把所有本体映射到同一组语义槽位（2607.06403）。X-VLA 则用 **soft prompt** 吸收本体差异，只训 **9M 参数**的适配器（2510.10274）。

优点：既不用字符串也不用 per-embodiment MLP，维度固定、语义统一。
缺点：canonical 维度的定义是人工的，映射函数需要领域知识；映射本身可能丢失本体特有的信息。

三条路线没有定论。2026 年的趋势是：**低维本体用 canonical space，高维人形用 per-embodiment 投影，语义差异大的用字符串/soft prompt**。

## 5. 数据配比：比数据量更重要

几个可复用的配方：

- **π0**：每个 task-robot 组合按 $n^{0.43}$ 降权（p.6）。数据越多，权重增长越慢于线性。
- **π0.5**：97.6% 的训练样本不来自目标域（p.2）；VI 只占 11%，但去掉显著变差。
- **Octo**：25 个数据集，权重从 17.0%（Fractal/Kuka/Bridge）到 0.1%（DLR EDAN），「更多样」的数据集 ×2，高度重复的降权（p.14）。
- **OpenVLA**：沿用 Octo 的 mixture 权重；DROID 加 10%，但因动作 token 准确率低，最后 1/3 训练移除（p.5）。
- **GR00T**：真机 3,288.8h + 人类视频 2,517.0h + 仿真 1,742.6h + 神经 827.3h；三层金字塔，越往上越少。

共同点：**没有一个是按数据集大小线性加权的**。都在对抗「大数据集淹没小任务」和「低质量数据污染模型」。

## 6. 2026 年的数据军备竞赛

- **LingBot-VLA 2.0**：60,000 小时、20 个本体、55 维 canonical space（2607.06403）。
- **MolmoAct2**：720 小时双臂 YAM（最大的开放双臂数据集）+ 过滤后的 DROID + SO-100/101（2605.02881）。
- **X-VLA**：290K episodes，DROID + RoboMind + AgiBot，7 平台 / 5 种臂型（2510.10274）。
- **FineART-VLA**：40,543 episodes / 533,913 子任务标注（2609.36416）。

一个明显的转向：**从「堆轨迹数」转向「堆标注密度和本体多样性」**。1M 条无标注轨迹的价值，可能不如 40K 条带子任务标注的轨迹——因为后者能训高层推理，前者只能训低层模仿。

## 小结

- OXE 有 1M+ 轨迹、22 本体、60 数据集，但实验只用了 9 个本体；「22 种机器人」是覆盖，不是训练。
- 跨本体混合不是越多越好：RT-1-X 在 Bridge/Google Robot 上不如单本体 RT-1（underfit）。
- +50% 和 3× 都是相对数；RT-2-X 的 3× 只发生在 5 个 emergent skills 上，去掉 Bridge 数据掉到 42.8%。
- q01/q99 对离群点更稳；mean/std 会被 0.5% 的尖峰压扁正常值。
- padding 维不能参与统计；relative/delta 要在归一化之前转换，并用 delta 自己的统计量。
- 跨本体三条路线：pad（π0）、per-embodiment MLP（GR00T）、canonical space（LingBot 2.0）。
- 数据配比（$n^{0.43}$、分层、质量筛选）比数据总量更能决定效果。

下一篇算延迟：[50 Hz 下 73 ms 的推理意味着什么，chunk 和异步怎么把延迟藏起来]({{< relref "vla-08-serving" >}})。
