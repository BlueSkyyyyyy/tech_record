---
title: "AI 日报 2026-11-12：11 篇 HF Daily Papers — 教科书监督、GTA V 智能体与推理成本"
subtitle: "TOME-Math 把 2,118 本教材压成 380K 条结构化证明记录、平均分 +16.7；Self-Learning RL 让 35B 模型自己读教材，proof accuracy 51.5%→79.5%；GTAGym 在 GTA V 上自动判定 103 个任务，最佳 70.9%"
date: 2026-11-12
draft: false
weight: 2
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "数学推理", "agent", "评测", "推理成本", "量化", "优化器", "权重空间"]
categories: ["LLM算法"]
summary: "2026-11-12 的 HF Daily Papers 只有 11 篇、67 upvotes，是近期最冷的一期。最热是 TOME-Math（26↑）：2,118 本大学数学教材转成约 380K 条保留章节顺序与证明依赖的记录，全量微调后平均分比 base 高 16.7 个百分点。同机构同日的 Self-Learning RL（15↑）走另一条路，不给参考解、让 Intern-S2-Preview-35B 自己读教材写证明，proof accuracy 51.5%→79.5%。第三是 GTAGym（10↑），把 GTA V 的 103 个在线任务做成引擎状态自动判定的评测与 RL 环境，最佳配置 70.9%，LoRA RL 把原生任务从 15% 提到 23%。其余硬数字：PISA-E 集成路由质量 +2.09/+2.87 分、低成本点再省 36%/25%；SparseRead 总 token 最多降 92.9%；SteerDistill 在 W4A16 下距 BF16 不到 1 分、训练 GPU 时间少 67%；WMIP 单步算术 84.9% 但 3–7 次更新后只剩 31.7%。"
---

先看总量：11 篇、67 upvotes，3 篇上双，中位数 2 票，两篇 0 票。前一天还有 29 篇、325 票，今天只剩 11 篇、67 票，是近期最冷的一期——样本小到不足以谈趋势，但主题意外地整齐。

最值得看的是开头两篇的对照。TOME-Math 把 2,118 本大学数学教材离线压成约 380K 条结构化证明记录，保留章节顺序、命题—证明链接、依赖引用和来源，全量微调后在 held-out 与公开 benchmark 的无加权平均分上比 base 高 16.7 个百分点；同一天、同一机构、作者高度重叠的 Self-Learning RL 则不给参考解答，用 BookNavigator 加 RL 让 Intern-S2-Preview-35B 自己翻教材写证明，proof accuracy 从 51.5% 拉到 79.5%。一个把结构固化进数据，一个把结构变成可学习的策略。

其余三条线：GTAGym 和 Interactive Visual Gym 都在造带自动判定的闭环环境，前者用 GTA V 引擎状态判 103 个任务（最佳 70.9%），后者把模拟用户也放进环境（最强模型限时完成 53.4%）；PISA、SparseRead、SteerDistill 从路由、输入、权重三层各切一刀压推理成本；WMIP、CogBattery 和 Mixed Signals 则都在提醒同一件事——单点分数、输出协议和权重本身，都不足以支撑「模型会了」的结论。正文分五组，组内按 upvotes 降序。

{{< doc_include "code/ai-daily/docs/2026-11-12.md" >}}
