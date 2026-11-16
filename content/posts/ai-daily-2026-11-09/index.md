---
title: "AI 日报 2026-11-09：34 篇 HF Daily Papers — agent harness 工程化、世界模型与「可审计的评测」"
subtitle: "HEELO 把工具做成带 LLM 的原语、EvoWorld 让世界模型推演视线外的后果、LongBanana 用 4,670 条专家检查项审图像合成"
date: 2026-11-09
draft: false
weight: 7
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "agent", "3DGS", "视频生成", "扩散模型", "kernel", "评测"]
categories: ["LLM算法"]
summary: "2026-11-09 的 HF Daily Papers 共 34 篇、644 upvotes（晚间补抓补齐）。当天主线是 agent harness 的工程化（HEELO 把资源包成带 resident LLM 的原语，Terminal-Bench 4.0 从 37.3% 到 55.5%；Beneath the Surface 提醒涨分常以丢掉旧任务为代价）、生成模型补因果与结构（EvoWorld 的视线外因果演化、Video-Coder 的 3D 草稿、SplatAnything 的高斯间距归一化），以及评测转向可审计证据（LongBanana 4,670 条专家检查项、BILBY 55→31→8、FinDendro 自报知识不可信、Sketchpad 用 Lean 认证算术）。"
---

先说结论：这一天没有新基础模型，但三条线同时收紧。第一条是 **agent 的竞争从「模型会不会」转到「harness 怎么造」**：HEELO 把每个资源包成一个带 resident LLM 的可复用原语，Beneath the Surface 则用 16 个进化 agent 证明「涨分」经常以丢掉旧任务为代价，AgentBaton 和 Authorization Drift 分别从成本和越权两侧补上了这套工程的边界。

第二条是 **生成模型开始补因果与结构**：EvoWorld 要求模型推演事件在视线之外的后果，Video-Coder 让 agent 写代码搭 3D 草稿给渲染器提供结构与运动信号，SplatAnything 修的是高斯间距归一化造成的监督漂移，LongBanana 则用 entity-relation graph 检查多参考合成有没有真的用上分散证据。像素保真已经不是主要瓶颈。

第三条是 **评测本身在变成「可审计的证据」**：LongBanana 的 4,670 条专家检查项、BILBY 把 55 个静态成功压到 8 个全过、FinDendro 发现 GPT-5 答对 76% 却只承认 38%、Sketchpad 用 Lean 证明有限视图覆盖等价于全输入精确——共同结论是分数必须连证据等级一起报。

下面是正文：34 篇按主题分六组，组内按 upvotes 降序，每篇都标了论文号、机构与一句「为什么值得看」。

{{< doc_include "code/ai-daily/docs/2026-11-09.md" >}}
