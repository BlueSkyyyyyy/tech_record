---
title: "AI 日报 2026-11-20：50 篇 HF Daily Papers — 自进化取证、Mind-Act 的 12%、后验匹配蒸馏"
subtitle: "WeVisForensics 用可自进化的取证检查员库把跨域检测做到 95.2%；Mind2Act World 量出最强 agent 在闭环组合任务上只剩 12.0%；PMD 用后验均值匹配做到 4 步生成同档最佳 FID"
date: 2026-11-20
draft: false
weight: 2
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "agent", "具身智能", "世界模型", "视频理解", "扩散模型", "推理加速", "可解释性"]
categories: ["LLM算法"]
summary: "2026-11-20 的 HF Daily Papers 共 50 篇、650 upvotes（次日 21:11 抓取快照）：23 篇上双、13 篇 ≥20 票，前三名 137 票占 21.1%。WeVisForensics（50↑）用主调查员 + 可自进化取证检查员库与 DTPO 双时间尺度优化，跨域平均 95.2%、高最强前作 9.0 点；Mind2Act World（46↑）把推理与执行拆成三视图，最强 agent 在 Mind 上 52.7%、Act 上 39.0%、闭环 Mind2Act 只剩 12.0%；NVIDIA 的 Posterior Matching Distillation（34↑）用后验均值匹配实现 any-step 蒸馏，SDXL 与 FLUX.2 上 4 步 FID 最佳；LightningMoBA（17↑）动态路由 + KV 中心执行，1.20–1.60× 且 RULER 反涨 0.86。"
---

先看总量：50 篇、650 upvotes，23 篇上双（其中 13 篇 ≥20 票），没有 0 票篇目。票比昨天更分散——前三名 137 票只占 21.1%，昨天同一口径是 31.7%，榜首也从 120↑ 落到 50↑。

今天的主线是「能力之外的东西」：WeVisForensics 让工具库自己进化，Mind2Act World 证明单项能力不迁移到闭环组合（52.7% / 39.0% → 12.0%），SimAudit 在 49 个具身任务里查出 40 个「假成功」，TuringSpace 发现模型会为错误信念编造证据。系统侧则一致地「少做一点」：PMD 的 any-step、LightningMoBA 的动态保留率、NarrowSpec 的窄草稿、FlashMorph2 的 3.57 GPU 小时层选择。

{{< doc_include "code/ai-daily/docs/2026-11-20.md" >}}
