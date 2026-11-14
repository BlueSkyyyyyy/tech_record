---
title: "AI 日报 2026-11-11：39 篇 HF Daily Papers — 机器人上下文学习、权重级修复与「证据链」"
subtitle: "WICL 用一条演示 + 仿真排练把 RoboTwin 2.0 拉到 83.4%、零梯度；Tree-Ising 把 BF16 修复解成 QUBO，MATH-500 54.6% 对 51.7%、成本约 $0.90；WikiTrace 给每条记忆接回原始轨迹"
date: 2026-11-11
draft: false
weight: 5
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "具身智能", "世界模型", "agent", "记忆", "模型编辑", "量化", "评测"]
categories: ["LLM算法"]
summary: "2026-11-11 的 HF Daily Papers 共 39 篇、610 upvotes（次日晚间复核补齐），17 篇上双。最热是 WICL（49↑）：冻结模型从一条演示出发，在模拟器里排练并把反馈写进上下文，RoboTwin 2.0 全部 50 任务平均 83.4%、零梯度，比最强训练策略高 20 多分。第二是 WikiTrace（43↑），把每条记忆链回原始轨迹和成败理由，减少 memory-induced degradation；第三是 Tree-Ising（37↑），把 BF16 权重修复拆成逐行 QUBO，MATH-500 54.6% 对 agent 的 51.7%，4 次前向、约 0.90 美元。另有 DexForesight（49.4%→60.5%）、WM-PPO（63.5%→87.5%）、DepthGraft（16B MoE 上 MMLU-Pro +7.0）、MMDM（hard multi-select 上限 36.81%）、FINCOSMOS（150 个 session 零接受）等硬数字。"
---

先看总量：39 篇、610 upvotes，17 篇上双（首轮 21:00 快照是 29 篇、325 票，补遗见正文末节）。热度比昨天更集中，长尾里有一半是 1–3 票的垂直工作。今天最值得读的不是某一个 SOTA，而是三条路线同时给出了「更便宜」的答案：机器人不重训权重、模型修复不跑 agent、记忆不只存结论。

第一条线是 **具身智能把世界模型移出推理路径**。WICL 让冻结模型在模拟器里排练，把反馈写进上下文，RoboTwin 2.0 五十个任务平均 83.4%、零梯度；DexForesight 只在训练时用世界模型塑形 policy latent，DexJoCo 从 49.4% 提到 60.5%；WM-PPO 只让 critic 对下一步预测状态估值，ALFWorld 63.5%→87.5%。同一天的 memory-clock 则给了一个反面提醒：按时间索引的记忆会把一次停顿当成进度，成功率从 47/50 掉到 16/50。

第二条线是 **权重与推理栈的「小改动」开始算细账**。Tree-Ising 把 BF16 修复变成逐行 QUBO，4 次前向、0 个 LLM token、约 0.90 美元；Deletion Tuning 只置零不到 0.05% 权重就匹配 FFT；DepthGraft 省约 1/3 KV cache 还在 16B MoE 上把 MMLU-Pro 拉高 7.0；MACaQ 量化 122B 只用 64 分钟和 32.5GB 显存。另一面，Dense Rewards 用 1,668 次 run 说明 matched compute 会把 dense reward 的排名翻过来，Sable 则把逐 bit 验证的成本标到每 4,096 token 6.4 美分。

第三条线是 **agent 的验收点移到证据链和过程合规**。WikiTrace 让每条 lesson 可回溯；MMDM 把决策单独拆出来量，hard multi-select 上限只有 36.81%；FINCOSMOS 的 150 个 session 零接受，用户签字的每一个都挂在程序检查上；Shortcuts 在 25% 冲突混合下得到 97.9% 的可盈利违规。下面正文按六个主题分组，组内按 upvotes 降序。

{{< doc_include "code/ai-daily/docs/2026-11-11.md" >}}
