---
title: "AI 日报 2026-11-10：39 篇 HF Daily Papers — 触觉基础模型、世界模型基础设施与「观测对齐」"
subtitle: "LingBot-Touch 用一套触觉 token 接进 π0.5 与 VLA、WorldFoundry 给 270 个世界模型建可复现评测、ReproBridge 把复现成功率从 76.4% 拉到 92.6%"
date: 2026-11-10
draft: false
weight: 4
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "具身智能", "世界模型", "agent", "训练系统", "扩散模型", "评测"]
categories: ["LLM算法"]
summary: "2026-11-10 的 HF Daily Papers 共 39 篇、560 upvotes（次日晚间复核补齐）。当天最热是 LingBot-Touch（90↑）：一套跨压阻、磁式、视触觉的统一触觉表示，tactile token 即插即用接进 diffusion policy 与 π0.5、LingBot-VLA；但摘要没有任何定量指标。给硬数字的是 ReproBridge（精修 artifact 把复现成功率从 76.4% 提到 92.6%、API 支出降 19.7%、跨模型极差 21.6→3.9 个百分点）、Flide（比 vLLM/SGLang 快最多 9.5×且更接近 fp32）、JoVox（8× 训练吞吐、1/15 前向 FLOPs、4 对 32 张 GPU）、InfraRSI（最高 2.75× 加速、MFU 17.4%→30.0%）。另外两条线是观测对齐（Reframe 把 observation state 当对齐对象）与可验证状态（WorldFoundry 的 270 模型/74 基准/62 指标、Sage-Taiyi 把化学收率从痕量做到 84%）。"
---

先说结论：这一天没有新基础模型，但三条线同时收紧。第一条是 **具身智能从「换传感器」转向「统一表示」**：LingBot-Touch 把压阻、磁式、视触觉的触觉数据压进同一套即插即用 token，直接接 diffusion policy 与 VLA，拿到 90↑；同一天 InstantHMR 把 6,890 顶点的人体网格压到手机 NPU 5.5 ms，Spatial Harness 则证明空间推理的瓶颈常在「把场景摆对」（位姿给对时 98.25% 对 71.81%）。

第二条是 **世界模型开始拼基础设施与可测量状态**：WorldFoundry 用 270 个模型条目、74 个 benchmark、62 个指标，把「能跑」与「能宣称」分开记账；WARC 把 test-time 学习量化成约 80 次 exposure 学会一半机制；JoVox 用 4 张对 32 张 GPU、1/15 前向 FLOPs 换来低 11% 的 Chamfer distance。

第三条是 **agent 的竞争同时落在训练信号与证据链上**：Reframe 把 observation state 本身当对齐对象，OPD mismatch 给出 FP8 下 1.93× KV 容量、1.62× 更快到同质量的预算；证据侧，ReproBridge 把复现成功率从 76.4% 拉到 92.6%，SIRIS 却提醒 13 个前沿 LLM「看出模拟器局限」并不等于「选对设计」。

下面是正文：39 篇按主题分六组，补遗 16 篇单列一节，组内按 upvotes 降序，每篇都标了论文号、机构与一句「为什么值得看」。

{{< doc_include "code/ai-daily/docs/2026-11-10.md" >}}
