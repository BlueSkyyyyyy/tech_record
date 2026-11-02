---
title: "VLA 精读（专题）：让模型输出动作，究竟难在哪"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 1
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "机器人", "多模态", "动作表示"]
categories: ["LLM算法"]
---

VLA（Vision-Language-Action）这两年从论文标题卷到了招聘 JD。最常见的介绍句式是「在 VLM 后面接一个动作头」——这句话在架构图上是对的，在工程上几乎什么都没说。

因为真正决定一个 VLA 能不能用的，不是它接了什么头，而是三个接口：

- **动作怎么表示**：离散成 256 个 bin，还是连续回归，还是用 flow matching 生成一段 chunk？
- **动作怎么在时间上执行**：一次出 1 步还是 50 步？推理 300 ms 时机器人是等着还是继续动？
- **不同本体的动作怎么对齐**：7 维单臂、14 维双臂、132 维人形，如何塞进同一个模型？

这三件事没有一个是「接个头」能解决的。过去三年 VLA 的论文，本质上是在这三个接口上反复做取舍；而 2026 年的竞争点，已经从「模型结构」移到了「数据配比 + 推理时延 + 评测可信度」。

这个专题只研究这三件事，以及它们如何决定一个 VLA 系统的能力边界。

## 先记住一个画面

想象一个能看图说话的模型，被要求以 **50 Hz** 控制一台机械臂。

语言模型每生成一个 token，都有语法和语义在兜底：说错一个词，人还能听懂。动作没有这种冗余。7 维动作错 1 cm，夹爪就可能撞到杯子；50 Hz 下错 3 个控制周期（60 ms），轨迹就会抖。

更麻烦的是，语言 token 之间是**离散且低相关**的——「今天」后面接「天气」还是「下午」，信息量很大。而 50 Hz 的相邻动作之间几乎是同一个数：关节从 0.300 rad 走到 0.302 rad，第二个 token 的边际信息接近零。FAST 论文把这件事说得最直白：

> "this marginal information approaches zero as the control frequency of the training signal increases"（FAST, p.4）

所以「把动作当 token」在 3 Hz 的 RT-1 上成立，到了 50 Hz 的叠衣服任务上就会退化——模型学会的是复制上一个动作，而不是控制。这不是模型不够大，是**表示和频率不匹配**。

> **类比的边界**：语言模型可以靠「下一个 token 预测」吃掉整个互联网，因为文本本身是离散符号；机器人没有互联网规模的 action 语料，动作还带物理单位、坐标系和本体差异。把 VLA 当成「多模态 GPT」会低估数据工程的比重，也会高估 scaling 的收益。

## 三条硬约束

### 1. 多峰性：同一个画面，人类演示不止一种做法

给定同一帧图像，把杯子拿起来可以向左绕，也可以向右绕。如果动作头用 L2 回归，最优解是**两个峰的平均**——一个物理上不存在的动作，通常直直撞向杯子。

这不是理论担忧。Diffusion Policy 的对照实验里，同样是预测动作，MSE 头只有 **35%**，diffusion 头 **83%**（WidowX，Octo 论文 Table VI, p.17）；Octo 自己用离散 256-bin 头只有 **18%**。动作头的表达力，直接决定策略能不能覆盖多峰分布。

### 2. 时序：动作是轨迹，不是标签

语言模型一次前向出一个 token，错了下一个 token 还能纠正。VLA 一次前向出一段 chunk，chunk 内部是**开环**执行的：π0 在 50 Hz 机器人上每 0.5 s 推理一次，一次执行 25 步（p.16）；OpenVLA-OFT 在 LIBERO 上一次出 8 步、ALOHA 上 25 步，整块执行完再重规划。

开环意味着两件事：

- **误差会积分**。如果每步动作有独立的量化误差，delta 动作的累积误差按 $\sqrt{n}$ 增长（最坏按 $n$）；50 步 chunk 的最坏累积是单步的 50 倍。
- **延迟会变成陈旧动作**。π0 在 RTX 4090 上单次推理 **73 ms**（on-board，p.16），听起来很快；但 50 Hz 的控制周期是 20 ms，73 ms 已经是 **3.6 个周期**。如果推理和执行不重叠，机器人每 0.5 s 就要停 73 ms。

### 3. 本体：动作空间的维度、单位、坐标系都不一样

Open X-Embodiment 汇总了 **22 种机器人、60 个数据集、1M+ 轨迹**（p.2–3），但论文自己承认：相机位姿不可对齐、动作可以是绝对位置/相对位置/速度、同一个动作向量在不同机器人上产生完全不同的运动（p.4）。

跨本体不是「把维度 pad 到 32」就完了。GR00T N1 的做法是给每个本体一套 state/action 编解码 MLP，共享 VLM 和 DiT 主干；2026 年的 LingBot-VLA 2.0 则更进一步，定义一个 **55 维 canonical state/action space**，让所有本体映射到同一组语义槽位。两种路线背后是同一个问题：**哪些维度是「本体私有」的，哪些是「任务公有」的**。

## 四个设计轴

把 2022–2026 的 VLA 摊开，几乎所有差异都能落到这四个轴上：

| 轴 | 选项 | 代表 | 代价 |
|---|---|---|---|
| **动作表示** | 每维 256 bin 离散 / 连续 L1 / diffusion / flow matching / DCT+BPE | RT-2、OpenVLA / OFT / Octo、RDT / π0 / FAST | 离散损失精度、高频退化；连续头要多步采样；BPE 拉长序列 |
| **动作头位置** | 直接复用 LM head / 独立 action expert / 小 MLP head / DiT | OpenVLA / π0、GR00T / OFT / GR00T System 1 | 复用 LM head 受词表和解码方式限制；独立 expert 增加参数和显存；小 head 表达力有限 |
| **时间粒度** | 单步 / chunk / 双系统（高频小模型 + 低频大模型） | RT-1 / ACT、π0 / GR00T | chunk 越长越开环；双系统要两个模型协同 |
| **数据与本体** | 单本体 / OXE 混合 / 每本体适配器 / canonical space | RT-1 / OpenVLA / GR00T / LingBot 2.0 | 混合会互相拖累；适配器增加部署复杂度；canonical space 依赖人工定义 |

四个轴不是独立的。选离散表示，就更容易复用 VLM 的 LM head，但高频任务会退化；选 flow matching，就要接受 10 步去噪的延迟，于是必须用 chunk + 异步推理把延迟藏起来；选跨本体混合，就要处理归一化和动作空间对齐，否则混合数据会互相污染。

**论文之间的差异，往往不是谁用了哪个组件，而是谁把某个轴的代价算得更清。**

## 本系列怎么读

13 篇，按「表示 → 架构 → 动作头 → 三个模型精读 → 数据 → 推理 → 代码 → 评测 → 2026 地图」组织。

| # | 篇目 | 一句话 |
|---|---|---|
| 0 | 本篇 | 三条硬约束与四个设计轴 |
| 1 | [动作表示：256 个 bin 的天花板]({{< relref "vla-01-action-space" >}}) | 离散化损失了什么；FAST 为什么必须压缩 |
| 2 | [架构：从 RT-2 到 π0 的 token 账]({{< relref "vla-02-architecture" >}}) | 视觉 token、动作 token、KV cache 与参数量的取舍 |
| 3 | [动作头：diffusion 与 flow matching]({{< relref "vla-03-action-head" >}}) | 多峰分布、去噪步数、为什么 π0 选 flow |
| 4 | [π0 精读：两个 Gemma 和一条 mask]({{< relref "vla-04-pi0" >}}) | block-causal attention、action expert、73 ms 的账 |
| 5 | [π0.5：把 state 写成字符串]({{< relref "vla-05-pi05" >}}) | 离散 state、adaRMS、异构数据与「知识绝缘」 |
| 6 | [GR00T N1/N1.7：双系统与本体条件化]({{< relref "vla-06-groot" >}}) | System 1/2、per-embodiment MLP、数据金字塔 |
| 7 | [数据：OXE、归一化与跨本体]({{< relref "vla-07-data" >}}) | 1M 轨迹为什么不够；q01/q99 与 padding 的坑 |
| 8 | [推理：chunk、延迟预算与 RTC]({{< relref "vla-08-serving" >}}) | 50 Hz 下 73 ms 意味着什么；异步的两个门槛 |
| 9 | [LeRobot 代码走读：π0/π0.5]({{< relref "vla-09-lerobot-code" >}}) | PyTorch 版逐段读，和 JAX 版对照 |
| 10 | [OpenVLA 代码走读：binning 与 OFT]({{< relref "vla-10-openvla-code" >}}) | 256 bin 的真实实现；并行解码为什么快 26× |
| 11 | [评测：LIBERO 的 AUC 不是成功率]({{< relref "vla-11-evaluation" >}}) | 5 个统计陷阱；10 次试验能说明什么 |
| 12 | [2026 地图：RL、世界模型与饱和的榜单]({{< relref "vla-12-2026-map" >}}) | π*0.6、VLA-JEPA、canonical space 与开放问题 |

三条阅读路线：

- **只想快速建立判断力**：0 → 1 → 3 → 8 → 11。知道动作表示、动作头、延迟和评测四个坑，就足够读懂大多数 VLA 论文的摘要。
- **要做工程复现**：0 → 4 → 5 → 9 → 8 → 10。π0/π0.5 是当前开源代码最完整的一支，LeRobot 有 PyTorch 实现；读完能自己改 chunk、换 action head、接 RTC。
- **要跟 2026 的研究前沿**：0 → 3 → 6 → 7 → 12。重点在数据配比、跨本体和 RL 后训练，而不是再刷一遍 LIBERO。

## 配套代码

`code/vla/` 下 7 个纯 CPU 脚本，每个都能 `--test` 自测。它们不是模型实现，而是把文章里的账**算给你看**：

| 文件 | 作用 | 对应文章 |
|---|---|---|
| `action_binning.py` | 256 bin 到底是 255 还是 256 档；量化误差；DCT 压缩 | 一 |
| `block_attn_mask.py` | π0 的 block-causal mask；前缀缓存省多少 | 四 |
| `flow_matching_toy.py` | 双峰动作上，flow matching vs L2 回归 | 三 |
| `norm_stats.py` | q01/q99 vs mean/std；跨本体 padding 的统计污染 | 七 |
| `chunk_latency.py` | 同步/异步 chunk 的刷新率与断档条件 | 八 |
| `rtc_toy.py` | RTC 权重曲线；新旧 chunk 拼接处的跳变 | 八、九 |
| `eval_ci.py` | Wilson 区间；10 次试验能区分什么 | 十一 |

```bash
cd code/vla
for f in *.py; do echo "== $f =="; python3 "$f" --test || exit 1; done
```

## 一句话结论

VLA 的难点不在「让 VLM 输出动作」，而在**让动作在时间、本体和评测三个维度上仍然成立**。

模型结构可以抄，数据配比、延迟预算和评测协议抄不了——这三样才是这个专题想讲清楚的东西。

> **类比的边界**：把 VLA 类比成「机器人的 GPT」有助于理解预训练-后训练的范式，但 GPT 的 scaling law 建立在可无限复制的文本上；机器人的数据要人遥操作、要标定、要按本体归一化，复制成本高几个数量级。所以 VLA 的「大」目前更多体现在 VLM 骨干上，而不是动作数据上。

下一篇从最基础、也最容易被跳过的问题开始：[256 个 bin 到底损失了什么]({{< relref "vla-01-action-space" >}})。
