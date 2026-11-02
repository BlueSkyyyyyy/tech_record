---
title: "VLA 精读（十二）2026 地图：RL、世界模型与饱和的榜单"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 13
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "2026", "强化学习", "世界模型", "MoE"]
categories: ["LLM算法"]
---

前十一篇讲的是「怎么造一个 VLA」。这一篇讲「2025–2026 年，这个领域往哪走了」。

一个前提：2026 年的 VLA 论文，**架构差异已经小于数据、训练目标和推理预算的差异**。LIBERO 上 2.1B–6B 的模型全挤在 97–98.6%，精度不再是区分度；延迟差 24 倍，数据规模差 200 倍，训练目标从行为克隆扩展到 RL 和世界模型。

所以这篇不按「模型列表」组织，而按**五个正在发生的变化**组织：

1. 从行为克隆到 RL 后训练；
2. 从像素世界模型到 latent 世界模型；
3. 从 pad 到 canonical action space；
4. 从 dense expert 到 MoE；
5. 从刷榜到「榜单饱和后看什么」。

最后给一张模型地图和一组开放问题。

## 1. 从行为克隆到 RL 后训练：π*0.6 / RECAP

模仿学习的天花板是演示质量。每次部署都会产生失败，每次人工接管都被扔掉——这是 VLA 最大的数据浪费。

π*0.6（RECAP，arXiv 2511.14759，2025-11-18）的做法是把 RL 变成**监督学习**：

- 训练一个**分布价值函数** $p_\phi(V \mid o_t, \ell)$，用 670M Gemma-3 骨干，输入是 returns-to-go，数据来自演示 + 自主 rollout + 人工接管；
- 策略额外 condition 一个**二值化的 advantage 指示**（高于/低于阈值），相当于告诉模型「这条轨迹比平均好还是差」；
- 迭代：部署 → 收数据 → 重训价值 → 重训策略。

这样就不需要 policy gradient、不需要可微奖励、不需要仿真器。价值函数把「好坏」压缩成一个 token，策略用监督学习去拟合「好」的条件分布。

**数字**（作者报告）：

- 连续做 **13 小时**浓缩咖啡；
- 在新家叠衣服 **2 小时以上**无干预；
- 最难任务上吞吐 **>2×**、失败率约 **一半**；
- 策略规模 **~4.9B**（Gemma 3 4B + 860M flow expert），价值函数另算 **670M**（Gemma 3）。

**要打的折扣**：

- 价值函数需要**任务特定的奖励定义**，不是全自动；
- 最大收益来自**人工接管**，不是 self-play；
- 不是新架构，是 π0.5 + Gemma 3 + 更大的 expert + 价值头。

> **类比的边界**：RECAP 像给一个实习生配一个「这单做得好不好」的评分员。评分员只看结果，不教动作；实习生从「好」的样本里学。但如果评分标准错了，实习生会学得更自信地做错。

## 2. 从像素世界模型到 latent 世界模型：VLA-JEPA / LaWAM

「先想象未来，再行动」是 2025 年的热门叙事，但它的推理成本很高：生成几帧视频，再让策略看视频做动作。

2026 年的共识变成了：**价值在表征，不在想象的帧**。

### 2.1 VLA-JEPA

VLA-JEPA（arXiv 2602.10098，2026-02-10）用 **V-JEPA2** 做世界状态编码器，训练目标是「预测未来的 latent state」，不是像素：

- target encoder 看未来帧，student 只看当前观测；
- 监督在 latent 空间，避免像素级重建；
- 动作头是 DiT-B + flow matching，4 步。

结果（作者报告）：LIBERO **97.2%**，LIBERO-Plus **79.5%**，对比 OpenVLA-OFT 69.6、π0-FAST 61.6、**π0 53.6**。注意 π0 在扰动下掉到 53.6，而 π0-FAST 是 61.6——**同一个模型的不同动作表示，鲁棒性差 8 个点**，这在普通 LIBERO 上看不出来。

更反直觉的是数据：VLA-JEPA 用的机器人数据「不到 villa-X 的 1%」，却靠 human video（SSv2）预训练拿到了更好的扰动鲁棒性。**数据规模不再是能力的代理，预训练目标解释了更多方差。**

### 2.2 LaWAM / Fast-WAM

LaWAM（arXiv 2606.15768）把世界模型压到 **230M**：在预训练视觉 foundation model 的特征上训一个 latent action model，然后复用它的 decoder 预测未来观测特征，输出一个 compact latent subgoal，动作头 condition 在这个 subgoal 上，**一次非迭代生成** chunk。

- LIBERO **98.6%**，RoboTwin **91.22%**，**187 ms/chunk**；
- 对比像素级 WAM（LingBot-VA 4,482 ms）快 **24×**。

Fast-WAM（arXiv 2603.16666）做的是消融：保留视频 co-training，**推理时不做未来生成**，190 ms，比 imagine-then-execute 快 4 倍，且**没有 embodied pretraining**。结论是：**视频建模作为训练信号有用，作为推理步骤大多没用。**

这两篇合起来，把「世界模型」拆成了三个不同的东西：

| 用法 | 代表 | 延迟 | 价值 |
|---|---|---|---|
| 只做训练信号 | Fast-WAM | 190 ms | 有，且便宜 |
| 生成 latent subgoal | LaWAM、VLA-JEPA | 187 ms | 有，但要 230M 的小世界模型 |
| 生成完整视频再执行 | LingBot-VA | 4,482 ms | 在实时场景基本不可用 |

> **类比的边界**：世界模型像下棋时在脑子里推演几步。推演本身有用，但如果你每走一步都要把整盘棋画出来，就太慢了。latent 世界模型是「只推演关键局面」，像素世界模型是「把每一步都画成图」。

## 3. 从 pad 到 canonical action space

2024 年的答案是「全部 pad 到 32 维」（π0/π0.5 的 `max_state_dim=32`、`max_action_dim=32`）。2026 年的答案是**显式的、有名字的布局**。

### 3.1 LingBot-VLA 2.0 的 55 维

LingBot-VLA 2.0（arXiv 2607.06403，2026-07-07）定义了一个 **55 维 canonical state/action space**：

| 槽位 | 维度 |
|---|---|
| `arm.position` | 14 |
| `end.position` | 14 |
| `effector.position` | 2 |
| `hand.position` | 12 |
| `waist.position` | 4 |
| `head.position` | 2 |
| `base.velocity` | 3 |
| reserved | 4 |

每个机器人映射到这组语义槽位。它比 π0 的 32 维多了 **waist/head/base**——这些维度在 π0 的 padding 里是被**静默丢弃**的。

数据规模：**60,000 小时**（50K 机器人 + 10K 第一人称人类视频），20 个本体，15–30 Hz。动作 expert 是 **32 experts / top-4 的稀疏 MoE**，sigmoid routing + 无辅助损失的负载均衡（明确引用 DeepSeek-V3）。

### 3.2 其他路线

- **GR00T N1.7**：`embodiment_tag` + `modality_configs`，checkpoint 里带每个本体的 action `delta_indices`，LeRobot 从 processor 配置推断 horizon。
- **X-VLA**：soft prompt 作为本体描述符，30 个 domain，频率写成文本（"Freq: 30Hz"），只训 **9M 参数（~1%）** 就能到 LIBERO **93%**。
- **VLA-JEPA**：`reinit_modules`，动作维度变了就只重初始化 `action_encoder`/`state_encoder`，其余保留。

一个共同点：**relative action 成了一等公民**。π0.5、π0-FAST、GR00T、VLA-JEPA 都有 `use_relative_actions` 和 `relative_exclude_joints=["gripper"]`——预测相对当前状态的 delta，但夹爪保持绝对。这本身是一种小规模的 canonicalization。

## 4. 从 dense expert 到 MoE

dense action expert 会饱和。2026 年的做法是在 expert 里放 MoE：

- **LingBot-VLA 2.0**：每个 action-expert block 的 FFN 都是 MoE，32 experts / top-4，shared-expert isolation，sigmoid router + routing-bias balancing。论文声称增益「不只是来自总参数增加」。
- **WALL-OSS**：MoE action heads，按任务阶段激活，离散/连续动作都支持。
- **GR00T N1.7**：不是 MoE，但同一思路——一个 3B 骨干 + 每本体 modality adapter，而不是每个机器人一个模型。

MoE 在 VLA 里的动机和 LLM 不完全一样：LLM 是为了「同样的 FLOPs 塞更多知识」，VLA 是为了「不同本体/任务阶段用不同的动作专家，但共享视觉语言表征」。

## 5. 榜单饱和之后

### 5.1 LIBERO 已经饱和

2026 年的模型，2.1B–6B 全在 **97–98.6%**：

| 模型 | LIBERO 平均 | 延迟 |
|---|---|---|
| GR00T N1.7 | 97.0% | 70 ms（~14 Hz） |
| MolmoAct2 | 97.2% | — |
| MolmoAct2-Think | **98.1%** | 自适应深度 |
| LaWAM | **98.6%** | 187 ms |
| LingBot-VA | 98.5% | **4,482 ms** |
| Fast-WAM | 97.6% | 190 ms |

在 50 trials/task 下，98.6% vs 97.6% 的差距在噪声里。**「X 比 Y 高 1 个点」在 LIBERO 上已经没有信息量。**

### 5.2 延迟差 24 倍

同一精度档，延迟从 187 ms 到 4,482 ms。这意味着：

- 4.5 s 的模型在真机上基本不可用，它只是在一个回合几百步的仿真里刷到了高分；
- 「精度一样」是假象，**可用性差一个数量级**；
- 2026 年真正的竞争轴是**延迟 × 精度 × 数据效率**，不是单点精度。

### 5.3 评测在换

- **LIBERO-Plus**（2510.13626）：7 个扰动轴（相机、机器人、语言、光照、背景、噪声、布局）。模型在这里才分开：VLA-JEPA 79.5 vs OpenVLA-OFT 69.6 vs π0-FAST 61.6 vs π0 53.6。
- **RoboTwin 2.0**（2506.18088）：双臂、强域随机化，LaWAM 91.22%。
- **RoboEval**：行为 + 结果指标，MolmoAct2 44.3%。
- **LIBERO-Safety**（2606.23686）：19,664 条无碰撞演示，8 个 VLA + 2 个具身 foundation model。
- **真机长程**：10–15 分钟任务，目前只有少数团队能做。

## 6. 模型地图

| 模型 | 时间 | 参数 | 骨干 | 动作头 | chunk | 关键点 |
|---|---|---|---|---|---|---|
| π0 | 2024-10 | 3.3B | PaliGemma | flow，10 步 | 50 | prefix/suffix + KV cache |
| π0.5 | 2025-04 | 3.3B | PaliGemma | FAST 预训练 + flow | 50 | state 字符串、异构数据 |
| GR00T N1 | 2025-03 | 2.2B | Eagle-2 | DiT flow，4 步 | 16 | 双系统、per-embodiment MLP |
| GR00T N1.7 | 2026-04 | 3B | Cosmos-Reason2 | DiT flow，4 步 | 40 | 132 维、modality tag |
| SmolVLA | 2025-06 | 0.45B | SmolVLM-2 | flow，10 步 | 50 | 冻结 VLM、只训 100M expert |
| RDT-1B | 2024-10 | 1.2B | — | diffusion | 64 | 双臂、46 数据集 |
| OpenVLA | 2024-06 | 7B | Prismatic | 离散 256 bin | 1 | 开源基线 |
| OpenVLA-OFT | 2025-02 | 7B | Prismatic | L1 + 并行解码 | 8/25 | 26×/43× 吞吐 |
| X-VLA | 2025-10 | 0.9B | Florence-2 | flow，10 步 | 32 | soft prompt，9M 可训 |
| π*0.6 | 2025-11 | ~4.9B（+670M 价值） | Gemma 3 | flow + advantage | 50 | RL 后训练 |
| VLA-JEPA | 2026-02 | 3B | Qwen3-VL + V-JEPA2 | flow，4 步 | 7 | latent 世界模型 |
| MolmoAct2 | 2026-05 | 4B + 621M | Molmo2-ER | flow，per-layer KV | 30 | 自适应深度 |
| LingBot-VLA 2.0 | 2026-07 | 6B | Qwen3-VL-4B | MoE flow | — | 55 维 canonical、60K 小时 |
| LaWAM | 2026-06 | 2.3B + 230M | — | flow，非迭代 | — | latent subgoal，187 ms |
| Fast-WAM | 2026-03 | — | — | flow | — | 不做推理时想象，190 ms |

（表中部分参数/延迟来自作者报告，未做第三方复现；「—」表示论文/笔记未给。）

## 7. 还没解决的问题

### 7.1 数据飞轮

RECAP 证明了「部署 → 收失败 → 重训」的闭环可行，但它依赖**任务特定的奖励**和**人工接管**。真正的自主飞轮——机器人自己发现失败、自己定义奖励、自己收集——还没有。

### 7.2 评测

LIBERO 饱和，LIBERO-Plus/RoboTwin 还没有形成共识；真机评测成本高、样本少；安全评测刚刚起步。**「SOTA」在 2026 年是一个比 2024 年更弱的词。**

### 7.3 长程记忆

在长程任务里，π0.5 对 30 秒前的观测利用很差（MEM 论文的测量）。MEM（2603.03596）用 mixed-modal memory（短时视频 + 长时语言）做到「几十分钟」，但只在 π0.6 上验证。长程任务（收拾整个厨房）仍然需要显式的记忆机制。

### 7.4 跨本体契约

55 维 canonical、embodiment tag、soft prompt 三条路线并存。没有一个公认的「动作空间标准」，导致模型之间无法直接比较、数据无法直接复用。

### 7.5 实时与安全

RTC 解决了 chunk 拼接，但**安全层**还没有：LIBERO-Safety 是第一个安全基准，SafeDojo 是第一个 safe RL 方法，LeRobot 里没有任何 policy-level guardrail。2026 年写「安全的 VLA」，写的是研究方向，不是产品特性。

### 7.6 奖励模型

RECAP 需要价值函数，价值函数需要奖励。TOPReward（2602.19313）、Robometer（2603.02115）是这条线上的早期工作，但「通用机器人奖励模型」还远没有解决。

## 8. 回到三条硬约束

第一篇说 VLA 的难点是三条硬约束：**多峰性、时序、本体**。2026 年的进展，正好对应这三条：

- **多峰性**：flow matching + chunk 成为默认，L1 只在单峰任务上用；一步生成仍不成熟（GR00T drifting head 掉 33 个点）。
- **时序**：RTC + 异步推理 + 双系统，把 73 ms 的推理藏进 500 ms 的决策周期；但延迟仍是 24 倍的差距。
- **本体**：从 pad 到 canonical space / embodiment tag / soft prompt，三条路线都在试图把「本体差异」变成显式的契约，而不是让模型自己猜。

三条约束没有一条被「解决」，只是被更好地**工程化**了。

## 9. 这个专题的终点

如果你读完了这 13 篇，应该能：

- 看到一个 VLA 论文的摘要，判断它的动作表示、动作头、chunk、延迟预算；
- 看到一个「成功率 97%」，追问 trial 数、指标定义、checkpoint 选择、动作空间；
- 看到一个「跨本体」，判断它是 pad、per-embodiment MLP 还是 canonical space；
- 看到一个「实时」，算 $C\Delta t$ 和 $L$ 的关系，判断它是真实时还是刷榜。

这比记住任何一个模型的参数更重要。模型会过时，这三条约束不会。

## 小结

- 2025–2026 的五个变化：RL 后训练、latent 世界模型、canonical action space、MoE expert、评测换轨。
- π*0.6/RECAP 把 RL 变成 advantage-conditioned 监督学习，>2× 吞吐、~一半失败率，但依赖任务特定奖励和人工接管。
- VLA-JEPA/LaWAM 证明「世界模型的价值在表征，不在想象的帧」；像素级 rollout 4,482 ms，latent 只要 187 ms。
- LIBERO 饱和（97–98.6%），延迟差 24 倍；该看 LIBERO-Plus、RoboTwin、真机长程。
- 开放问题：数据飞轮、评测共识、长程记忆、跨本体契约、安全、奖励模型。
- 三条硬约束（多峰、时序、本体）没有消失，只是被工程化得更好。

---

**专题目录**

0. [导读：三条硬约束与四个设计轴]({{< relref "vla-00-index" >}})
1. [动作表示：256 个 bin 的天花板]({{< relref "vla-01-action-space" >}})
2. [架构：从 RT-2 到 π0 的 token 账]({{< relref "vla-02-architecture" >}})
3. [动作头：diffusion 与 flow matching]({{< relref "vla-03-action-head" >}})
4. [π0：两个 Gemma、一条 mask]({{< relref "vla-04-pi0" >}})
5. [π0.5：state 字符串与知识绝缘]({{< relref "vla-05-pi05" >}})
6. [GR00T N1/N1.7：双系统与本体条件化]({{< relref "vla-06-groot" >}})
7. [数据：OXE、归一化与跨本体]({{< relref "vla-07-data" >}})
8. [推理：chunk、延迟预算与 RTC]({{< relref "vla-08-serving" >}})
9. [LeRobot 代码走读]({{< relref "vla-09-lerobot-code" >}})
10. [OpenVLA/OFT 代码走读]({{< relref "vla-10-openvla-code" >}})
11. [评测：LIBERO 的 AUC 不是成功率]({{< relref "vla-11-evaluation" >}})
12. 本篇：2026 地图
