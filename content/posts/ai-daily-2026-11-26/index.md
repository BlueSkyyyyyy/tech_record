---
title: "AI 日报 2026-11-26：60 篇 HF Daily Papers — 把视频编译成仿真、35B 医学模型、循环深度换参数"
subtitle: "Real2SimLab 用 6 个基准、47 个 coding agent 统一 real2sim；Sage-Med 以 35B 总参 / 3B 激活在 31 个公开医学多模态基准拿到 58.26%，超 235B 的 Hulu-Med；MoLT 用 KV 当跨 loop 记忆换 1.7× token 效率；VibeVideoAgent 让生成器边渲染边自查，Opus 5.5 的 L3 4.03→4.43"
date: 2026-11-26
draft: false
weight: 3
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "具身智能", "模型架构", "视频生成", "3D", "agent", "评测"]
categories: ["LLM算法"]
summary: "2026-11-26 的 HF Daily Papers 共 60 篇、902 upvotes：26 篇上双、43 篇 ≥3 票，前三名占 29.4%、前十名占 60.4%。主线是数据入口与参数分配——Real2SimLab（108↑）统一 6 个 real2sim 基准，PyRUA-Dex（44↑）让 coding agent 无演示写灵巧策略；Sage-Med（89↑）以 35B 总参 / 3B 激活在 31 个公开医学多模态基准拿到 58.26%，MoLT（68↑）用 KV 记忆换 1.7× token 效率，DyLoMo-DiT（57↑）靠权重共享循环把 GenEval 做到 0.714。"
---

今天 60 篇、902 票，比昨天多 1 篇、少 159 票，热度更平：榜首 Real2SimLab 只占 12.0%，前十名合计 60.4%。票数降下来之后，主线反而更清楚——今天最强的信号不是「模型更大」，而是「数据从哪来、参数在什么时候被用到」。

三条线值得跟：一是**视频、代码与仿真失败变成机器人的数据入口**（Real2SimLab 的 6 基准统一、PyRUA-Dex 的 30 分钟学琴、Entangle 的 69.5%）；二是**容量按需分配**（Sage-Med 的 3B 激活打赢 235B、MoLT 的 1.7× token 效率、DyLoMo-DiT 的 1.05B 激活超过 FLUX.1）；三是**生成与评测都开始自查过程**（VibeVideoAgent 的草稿感知回路、Puzzling-Bench 的 post-cutoff 切片、AIChipBench 里 69 个通过测试的设计有 45 个在新序列上失败）。

下面 60 篇全覆盖：赶时间可以先看 Top 10 表，再跳到每篇的「数字」一行；正文按主题分六组，组内按 upvotes 降序。

{{< doc_include "code/ai-daily/docs/2026-11-26.md" >}}
