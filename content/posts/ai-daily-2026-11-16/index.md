---
title: "AI 日报 2026-11-16：40 篇 HF Daily Papers — 机器人 harness、4D 场景生成与「读动态」的视频模型"
subtitle: "Agentic Robotics 把 harness 与基础模型并列为决策核心；VibePrevisBench 用 242 个 Blender 任务证明 coding agent 会看渲染却不会改错；V-JEPA-LM 用反事实监督让 4B 模型在 11 个视频 benchmark 上得 72.1，超过最好开放 8B 的 64.8"
date: 2026-11-16
draft: false
weight: 11
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "具身智能", "agent", "评测", "视频生成", "世界模型", "推理加速", "科学计算"]
categories: ["LLM算法"]
summary: "2026-11-16 的 HF Daily Papers 共 40 篇、726 upvotes（次日复核补齐）。最热是 Agentic Robotics（96↑）：把 Embodied System 2 定义成「基础模型 + embodied harness」，提出模型能力与闭环系统双轴评测，并给出通往持续自主的路线图。第二是 VibePrevisBench（80↑）：242 个 Blender 4D 场景任务、11 个 coding agent，规则 Scene Check 查 9 维事实、VLM 查 proxy 视频，结论是 agent 会检查自己的渲染却不会据此改错。第三是 Gamer（69↑）：数万小时键鼠轨迹继续预训练 + sparse-thinking + 多轮 RL，一套 pipeline 通吃 2D GUI 与 3D 开放世界。其余硬数字：MMFlash 多模态投机解码最高 3.58× 无损加速；V-JEPA-LM 4B 在 11 个视频 benchmark 平均 72.1，超过最好开放 8B 的 64.8；MIRIX-2.0 在 LongMemEval-S 得 72.6%（+14.3 点）；MEM-SQL 执行准确率 37.1%→50.0%、输入 token 少 76.6%；DySENSE 用不到 25% 标注把下游预测提升 8% mIoU。"
---

先看总量：40 篇、726 upvotes，25 篇上双，前三名拿走 287 票（39.5%），最高 114、最低 0。40 篇的 daily 收录日都是 2026-11-16，但 arXiv 首发日跨 2026-10-14 到 2026-11-13，最老的一篇挂了 33 天才上榜。

今天的主线是「能力从模型本体往模型 + harness + 环境迁移」。Agentic Robotics 直接把 embodied harness 写进决策核心的定义；Gamer 的实时性主要靠云边推理引擎；VibePrevisBench 评的是 coding agent 的完整交付轨迹，而不只是最终视频。第二梯队则是两条硬账：MMFlash 把多模态投机解码的接受率拆到 token 级视觉依赖，最高 3.58× 无损加速；V-JEPA-LM 先证明动态信息就在冻结编码器里、只是语言模型没去读，再用反事实监督把 4B 模型推到 11 个视频 benchmark 平均 72.1。

正文 28 篇分六组，组内按 upvotes 降序；每篇保留摘要里的具体数字，摘要没给的会明确写出来。赶时间的话，读完 Top 10 表和「今天的三个观察」即可。

{{< doc_include "code/ai-daily/docs/2026-11-16.md" >}}
