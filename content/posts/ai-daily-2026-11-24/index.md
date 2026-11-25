---
title: "AI 日报 2026-11-24：75 篇 HF Daily Papers — harness 决定上限、可规划的 4D 世界、权重空间编程"
subtitle: "Embo-Harness 只改外层配方就到 83.3%（+26.2 点），RUIBench 只换接口把 45% 拉到 81%；WorldLeague 给 13 个系统画物理能动性画像，最高 28.58/100；WeightCode 用权重程序把未见过的新指令组合从 0% 做到 99.8%"
date: 2026-11-24
draft: false
weight: 2
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "具身智能", "agent", "世界模型", "多模态", "LLM训练", "推理系统"]
categories: ["LLM算法"]
summary: "2026-11-24 的 HF Daily Papers 共 75 篇、1260 upvotes：32 篇上双、57 篇 ≥3 票，前三名占 31.7%、前十名占 57.7%。主线是 harness 成为独立变量——Embo-Harness（90↑）只改提示/规划/执行/工具就到 83.3%，RUIBench（22↑）只换机器人接口把前沿 agent 从 45% 拉到 81%，AutoPILOT（33↑）让模型写闭环控制器把实时成功率从 38.3% 提到 68.6%；世界模型转向可规划与可评测——ViDecide（39↑）严格成功率 83.8% vs 23.3%，WorldLeague（128↑）最高只有 28.58/100；权重空间编程与数据审计给出两条新杠杆——WeightCode（14↑）新指令组合 0%→99.8%，SWE-Gym 审计发现 74% 的「解不出」其实是测试假阴性。"
---

今天 75 篇、1260 票，是近期篇数最多的一天，但热度反而分散：榜首 Motion Mamba 2.0 只占 14.4%，前十名合计 57.7%。篇数多意味着不能逐篇看，主线反而更清楚——今天最强的信号不在模型本身，而在模型外面的那一层。

三条线值得跟：一是 harness 与接口成了可单独优化的变量（Embo-Harness 83.3%、RUIBench 45%→81%、AutoPILOT 38.3%→68.6%）；二是世界模型开始分头补几何（Plan4D、WorldLatent、PersistWorld 的 +19 IoU）与决策（ViDecide 83.8% vs 23.3%），而 WorldLeague 的 28.58/100 说明两者都还早；三是训练侧的两把新杠杆——权重空间编程与数据审计，前者把新能力直接写进权重，后者发现「模型解不出」有 74% 是测试标错。

下面按五个主题分组，75 篇全覆盖；赶时间可以先看 Top 10 表，再跳到每篇的「数字」一行。

{{< doc_include "code/ai-daily/docs/2026-11-24.md" >}}
