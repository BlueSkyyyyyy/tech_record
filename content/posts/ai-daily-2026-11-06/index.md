---
title: "AI 日报 2026-11-06：32 篇 HF Daily Papers — 决策模型、潜空间通信与「评测的可靠性」"
subtitle: "决策模型把 CoT 换成一次前向、agent 之间开始传 KV cache、多轮编辑只在去噪轨迹尾部修"
date: 2026-11-06
draft: false
weight: 2
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "agent", "VLA", "扩散模型", "线性注意力", "评测"]
categories: ["LLM算法"]
summary: "2026-11-06 的 HF Daily Papers 共 32 篇。当天主线是「把推理/记忆/通信从文本搬到别的地方」：Drex 用决策模型替代 CoT、SwarmEvolve 与 PACT 用潜空间传信息、Coca-Cola 把工具输出压成 latent token、SLA 把空间对齐做进持续预训练、TailSteer 只在去噪尾部修多轮漂移。另有 BestRand/OpenFrontiers/VisSearch-Bench/NPCBench 四篇说明单点分数正在失效。"
---

先说结论：这一天没有新的大模型，但有**三条互相咬合的路线**同时出现——决策模型（一次前向出分布，不要 CoT）、潜空间通信（KV cache / latent token / hidden state，不要文本交接）、以及「只修轨迹尾部」的训练法（不改能力，只补长程稳定性）。

另外值得单独拎出来的是评测：BestRand 用理论证明了基准压缩的误差是长尾且不可消除，OpenFrontiers 用 Lean 4 + 三种信息条件说清「联网解题多半是检索」，VisSearch-Bench 说最强多模态 agent 只能完整解出 26.47%，NPCBench 说单项强不等于整条诊疗路径对。**单点分数越来越不可信**。

下面是正文：32 篇按主题分六组，每组内按 upvotes 降序，每篇都标了论文号、机构与一句「为什么值得看」。

{{< doc_include "code/ai-daily/docs/2026-11-06.md" >}}
