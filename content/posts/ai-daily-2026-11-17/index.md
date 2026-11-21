---
title: "AI 日报 2026-11-17：42 篇 HF Daily Papers — 科学 agent、552B VLA 与「交付率」"
subtitle: "Sage 320B 科学 MoE 在 33 个 benchmark 的对照里拿了四项第一；ScaleVLA 用 552B backbone 把 LIBERO-PRO 推到 45.1%、真机插充电器 4/10→9/10；SWE-CoWork 则证明约 63% 的测试通过率只换来不到 8% 的完整交付"
date: 2026-11-17
draft: false
weight: 5
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "模型发布", "agent", "具身智能", "评测", "生成模型", "安全", "推理加速"]
categories: ["LLM算法"]
summary: "2026-11-17 的 HF Daily Papers 共 42 篇、619 upvotes（次日复核补齐）。最热是 Sage S2.1 Flash（64↑）：320B 科学 MoE，分子图/生物序列/谱图用专用编码器进共享 backbone，33 个 benchmark、7 个领域里 OfficeQA Pro 72.2、MLE-bench Lite 90.3 HumanRank、Harvey LAB、SpreadsheetBench-2 第一，OSWorld-Verified 85.2 第二。第二是 AllocFlow（60↑）：从原始行情端到端生成联合仓位，模仿加组相对 RL 把年化净收益 27.79%→51.38%、Calmar 0.955→4.043，未见 6 个月仍有 4.90。第三是 K2-Horizon（56↑）：0.9B–375B-A23B 六个全开源模型，MoVA 把 MoE 用在 attention value 上，36B-A4B 以 1/7 active 参数接近 32B dense。其余硬数字：InclusionMed 医疗 Index 仅 17.5–49.1 且排名随角色翻转；SPILLWAY 在开源 RL attacker 下 in-scope ASR 0.007、零额外 backbone 调用；ScaleVLA 10 个 demo 超过 2B 的 50 个（87.8% vs 85.6%）、216 ms/chunk；SWE-CoWork 前两名满足约 63% F2P 但完整交付 <8%；VoxMel 0.6B 模型 test-en 1.51% WER、test-zh 1.29% CER。"
---

先看总量：42 篇、619 upvotes，13 篇上双（首轮 23:30 快照是 30 篇、463 票，补遗见正文末节）。42 篇的 daily 收录日都是 2026-11-17，但 arXiv 首发日跨 2026-10-26 到 2026-11-16。

今天的主线是「分数之外的两本账」。一本是交付账：SWE-CoWork 里前两名满足约 63% 的 fail-to-pass 测试，完整交付却不到 8%；Accelerate Bench 的 412 个正确结果里 125 个加速不到 1.25×；InclusionMed 的总分只有 17.5–49.1，排名还会随医疗角色翻转。另一本是规模账：Sage 用 320B 换 OfficeQA Pro 72.2，ScaleVLA 用 552B 换 10 个 demo 顶 50 个（87.8% vs 85.6%）和插充电器 4/10→9/10，代价是 216 ms 一个 action chunk；K2 则反着来，用 1/7 的 active 参数去接近 32B dense。

安全侧今天有两篇值得单看：SPILLWAY 把「动作能去哪」从模型判断改成 typed returns 加固定规则的确定性 floor，开源 RL attacker 下 in-scope ASR 0.007；Centered Ablation 把方向消融写成 logit bias 并闭式投影掉 common mode，Gemma-2-9B 的文本崩坏从 50% 降到 1%，还顺手证明两个标准安全 judge 会对崩坏文本给出相反打分。

正文 30 篇分七组，组内按 upvotes 降序；每篇保留摘要里的具体数字，摘要没给的会明确写出来。赶时间的话，读完 Top 10 表和「今天的三个观察」即可。

{{< doc_include "code/ai-daily/docs/2026-11-17.md" >}}
