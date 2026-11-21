---
title: "AI 日报 2026-11-18：39 篇 HF Daily Papers — Ascend 全栈、力觉闭环与 agent 的「记忆」"
subtitle: "285B 模型在 8,192 张 Ascend 910C 上预训练、AscendGYM 把 kernel 基准改成真实 shape + token 成本；ForceHarness 让冻结模型真机零样本从 3.3% 到 90.0%；SIM 给 harness 优化器建记忆，十个 source run 平均 +2.36 分"
date: 2026-11-18
draft: false
weight: 4
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "模型发布", "训练系统", "agent", "具身智能", "评测", "量化", "生成模型"]
categories: ["LLM算法"]
summary: "2026-11-18 的 HF Daily Papers 共 39 篇、693 upvotes（次日复核补齐）。16 篇上双、23 篇 0–9 票，前两名都是 Ascend：Xing 4（146↑）把 285B 模型在 8,192 张 Ascend 910C 上的预训练摊开写，MindSpore HyperParallel 做拓扑感知混合切分与分布式优化器，TeleGym 承接 agent 后训练；AscendGYM（90↑）用 48 个真实 workload shape 的 kernel、两种评测模式和新指标 STS 把 token 成本除进加速比。第三是权重空间编程（40↑）：三个前沿 LLM 自由设计架构时都选了稀疏表，最高 54.4%、参数少约 1,800×、GPU 快最多 17×。具身侧 ForceHarness（38↑）把力闭环下沉成 primitive、把力流翻成物理量当 tool result，GPT-6 Astra 真机零样本 3.3%→90.0%，HiddenPhys 上 +34.3 点并追平手写专家控制器；TCP4D（23↑）DROID 轨迹误差 7.60 mm、RoboTwin 69.19%，10 个示范适配新本体。效率侧 MBT（34↑）用分支并行 + Joint-KV 把 KV cache 降 34%、单张 H200 解码吞吐 +51.4%；FastFrames（19↑）1.3B 在单张 B200 上 36 ms/帧（28 RPS）。方法学最硬的是 Noise Floor（1↑）：非量化噪声翻动 0–0.92% 决策、4-bit 翻动 3.5–5.8%，而 Q4_K_M 的准确率只动 6 题（−0.02 点）却掩盖 436 条变差的决策。"
---

先看总量：39 篇、693 upvotes，16 篇上双（首轮 23:31 快照是 31 篇、586 票，补遗见正文末节）。39 篇的 daily 收录日都是 2026-11-18，arXiv 首发日却从 2026-11-06 跨到 2026-11-17，最老的一篇等了 12 天才上榜。

今天的主线是「系统账」和「接口账」。系统账这次由 Ascend 占了两席，正好是榜首和榜眼：Xing 4 给的是 285B 模型、8,192 张 910C 的预训练报告，拓扑感知混合切分、分布式优化器执行、CPU 亲和性与核隔离，全是多卡跑稳的基础设施问题；AscendGYM 反过来把 kernel benchmark 从「生成新算子」挪到「在真实训练/推理的 shape 下改已有实现」，还加了 STS 这个把 token 成本除进加速比的指标。接口账则在具身和 agent 侧：ForceHarness 把力闭环下沉到模型之下、把力流翻译成带不确定度的物理量，同一个冻结模型在真机上从 3.3% 跳到 90.0%；SIM 给 harness 优化器本身建「改进记忆」，coding 域十个 source run 成对延续平均 +2.36 分。

方法学上今天有一篇票很低但很值得读的：Noise Floor 把「量化对比」的对照组拆开，量出引擎、重启、batch、kernel 自选这些非量化来源能翻动 0–0.92% 的决策，而 4-bit 量化翻动 3.5–5.8%；准确率只动 6 题（−0.02 点）却掩盖了 436 条实际变差的决策。同一天 Breath-Dx 证明随机切分会把呼气筛查的 AUC 高估 0.05–0.12，VisualArtifactSpec 发现配置间 73.4% 的差距来自产物表现而不是功能满足——分数越容易刷，口径就越值钱。

正文 31 篇分七组，组内按 upvotes 降序；每篇保留摘要里的具体数字，摘要没给的会明确写出来。赶时间的话，读完 Top 10 表和「今天的三个观察」即可。

{{< doc_include "code/ai-daily/docs/2026-11-18.md" >}}
