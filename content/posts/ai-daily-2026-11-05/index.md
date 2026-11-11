---
title: "AI 日报 2026-11-05：39 篇 HF Daily Papers — 回合级信用、机器人数据与推理的「物理账」"
subtitle: "TurnWise 把 GRPO 的信用拆到回合、DataFarm 用 coding agent 造机器人数据、SIMPLE 把推理能耗写成四参数定律"
date: 2026-11-05
draft: false
weight: 6
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "agent", "VLA", "推理系统", "扩散模型", "评测"]
categories: ["LLM算法"]
summary: "2026-11-05 的 HF Daily Papers 共 39 篇。当天主线是「把长程问题拆开算」：TurnWise 把 GRPO 的信用从整条轨迹拆到每个回合，PyRUA-DataFarm 用 coding agent 在仿真里造机器人数据，NEO-VLA 去掉固定视觉编码器，Halley/SIMPLE 把推理与训练的账本换成 goodput、容量和每焦耳智能。评测侧 FrontierPhysics 只过 46/168、VideoMemBench 最强 45.7%。"
---

先说结论：这一天没有新的大一统基础模型，但有三笔账被重新算了一遍——**agent 的信用该记到哪个回合**、**机器人数据的成本该由谁承担**、**推理效率该用什么单位衡量**。TurnWise 把 GRPO 的 advantage 拆到回合，DataFarm 把遥操作换成 coding agent + 仿真渲染，SIMPLE 则把能耗写成四参数定律，Halley 直接报机柜容量。

评测的两盆冷水放在前面：FrontierPhysics 让 agent 做完整的物理研究，22 个模型、26 种配置没有一种通过超过 46/168 次 rollout；VideoMemBench 里最强的 Claude Opus 5 也只有 45.7%。**「单步会做」和「整条路径做对」之间，仍然隔着信用分配与状态携带两道坎。**

下面是正文：39 篇按主题分七组，组内按 upvotes 降序，每篇标了论文号、机构与 upvotes；数字只取自摘要，摘要没给的会明确写出来。

{{< doc_include "code/ai-daily/docs/2026-11-05.md" >}}
