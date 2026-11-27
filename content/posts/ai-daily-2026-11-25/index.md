---
title: "AI 日报 2026-11-25：59 篇 HF Daily Papers — agent 学会何时停、镜子照出机器人、矩阵残差流重接电路"
subtitle: "RLVB 用精确后验教 agent 何时提交，九个设置平均 +17.1%；NightWatch 外挂 0.8B watcher 让 OSWorld 墙钟时间降 33%；一面镜子让五类机器人策略掉 10.8–56.7 点，最高 62.5 点；Loom Attention 只多 2% FLOPs 就匹配 3 倍算力"
date: 2026-11-25
draft: false
weight: 3
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "agent", "具身智能", "模型架构", "多模态", "视频生成", "评测"]
categories: ["LLM算法"]
summary: "2026-11-25 的 HF Daily Papers 共 59 篇、1061 upvotes：25 篇上双、43 篇 ≥3 票，前三名占 28.4%、前十名占 65.2%。主线是 agent 的元决策——RLVB（98↑）用可验证后验教何时提交，平均 +17.1%；NightWatch（97↑）用 0.8B watcher 解决感知过期，OSWorld 墙钟时间 -33%；ACAP（62↑）让一组 coding agent 异步控制机器人，真机时间成本 -28.2%。评测侧 ICLWorld 暴露 82% vs 27% 的数字/物理落差，RoboDojo-Mirror 证明镜子能让策略掉最多 62.5 点；架构侧 Loom Attention（90↑）加矩阵残差流，2% 额外 FLOPs 匹配 3 倍算力。"
---

今天 59 篇、1061 票，比昨天少 16 篇，热度却更平均：榜首 Antom 只占 10.0%，前十名合计 65.2%。篇数降下来之后主线反而更清楚——今天最强的信号不在「模型更强」，而在「模型什么时候该动」。

三条线值得跟：一是 agent 的元决策成了可训练对象（RLVB 的 +17.1%、NightWatch 的 -33% 墙钟、ACAP 的 -28.2% 时间成本）；二是评测从排行榜变成诊断仪，ICLWorld 的 82% vs 27%、RoboDojo-Mirror 的 62.5 点、Clinical3D 的「给了 3D 图却不用」都在拆「看起来会」；三是架构侧集体反对均匀默认值——Loom 的 per-token 矩阵残差流、FFN 的中段更宽、Model Carving 的自剪拓扑、CodecOPSD 剪掉 66% 条件 token。

下面 59 篇全覆盖，赶时间可以先看 Top 10 表，再跳到每篇的「数字」一行。

{{< doc_include "code/ai-daily/docs/2026-11-25.md" >}}
