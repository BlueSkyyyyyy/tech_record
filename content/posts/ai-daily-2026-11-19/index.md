---
title: "AI 日报 2026-11-19：52 篇 HF Daily Papers — System One 的一次前向、可交互世界模型与「评测的分辨率」"
subtitle: "Jeveration 用冻结 VLM 单次前向 78 ms 出 512×512、比统一模型最多快 122×；Resolution Hypothesis 用 7 种方法 × 6 个模型、超 1,000 万次调用证明选择近随机；QDA 让 agent 设计量化配方，W3A16 平均 +2.2 点"
date: 2026-11-19
draft: false
weight: 3
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "生成模型", "世界模型", "agent", "评测", "量化", "具身智能", "训练系统"]
categories: ["LLM算法"]
summary: "2026-11-19 的 HF Daily Papers 共 52 篇、599 upvotes（次日 21:03 抓取快照），是本专题篇数最多的一期。16 篇上双、36 篇 0–9 票、无 0 票；前三名 190 票占 31.7%，前十名 342 票占 57.1%。榜首 Jeveration（120↑）用冻结 VLM 单次前向出图，512×512 只要 78 ms、比现有统一模型最多快 122×；ALIVE（35↑）让流式视频模型学会主动监控、协调任务与追问；The Resolution Hypothesis（35↑）用 7 种进化方法 × 6 个模型、超 1,000 万次调用证明候选选择近随机、方法排名常小于同方法重跑差异。PROLA（27↑）把固定低秩预算拆给主方向与正交补做气象降尺度；Game2Player（25↑）造出 568 个网页游戏、3,436 个任务，9B 在 PlayerBench 上 14.3%→31.2%；CurvDelta（22↑）把保留约束写进参数化；HINT-WAM（22↑）用未来视频当 hindsight 做 on-policy 自蒸馏；LightTAR（20↑）给 PCA 表示补 target-aware 特征。世界模型侧 Code World Model 2.0（18↑）用四步因果渲染稳住 30 分钟 rollout，WorldRuler（18↑）做规则与状态联合存读档，StrideWorld（14↑）量出最高 7.3× 的场景形变。效率侧 QDA（14↑）用 KL 代理把量化配方搜索压到 64 GPU 小时，W4A16/W4A8/W3A16 平均 +0.9/+1.4/+2.2 点；Tideway（8↑）在消费级硬件上做到 1.97× 端到端、2.40× TTFT。方法学侧 Unfinished-Bench（5↑）量出持久性把内部文档规则违规从 32.2% 抬到 43.6%，是用户明说时 8.5% 的 5 倍。"
---

先看总量：52 篇、599 upvotes，16 篇上双、36 篇落在 0–9 票、没有 0 票的篇目。这是本专题开栏以来篇数最多的一期，但票数比昨天分散：前三名 190 票、占 31.7%，前十名 342 票、占 57.1%（11-18 是 39 篇、693 票，前三名占 42.1%）。52 篇的 daily 收录日都是 2026-11-19，arXiv 首发日却从 2026-08-26 跨到 2026-11-18，最老的 PGPO 等了 85 天。

今天的主线是「一次前向」。Jeveration 用冻结 VLM 在单次前向里出图，512×512 只要 78 ms、比现有统一模型最多快 122×；DWD 把 VAE 解码器换成最后一步的线性头，Wan 上 warping error 降 23%；AnyS1 把 agent 的常规决策卸载给一个 4B pointer，episode 成本降 75–91%；Piccolo 则把「决策」本身做成 0.8B–27B 的模型家族。第二条线是可交互世界模型：Code World Model 2.0 的四步因果渲染把 30 分钟赛车 rollout 稳住，WorldRuler 支持规则与状态联合存读档，StrideWorld 则量出 140 个起始视图里除一个模型外全部发生形变、偏移最高达刚性渲染的 7.3×。第三条线是评测本身：Resolution Hypothesis 用超过 1,000 万次调用证明候选选择区分好坏近随机，Unfinished-Bench 量出持久性让文档规则违规从 32.2% 涨到 43.6%，「Thinking Harder」发现思考力度只抬高置信度、不提高正确率。

正文 52 篇分九组，组内按 upvotes 降序、同票按 arXiv id 升序；每篇保留摘要里的具体数字，摘要没给的会明确写出来。逐篇 upvotes 截至抓取时间 2026-11-20T21:03:09+08:00，本期为次日补写、52 篇是抓取时点快照。赶时间的话，读完 Top 10 表和「今天的三个观察」即可。

{{< doc_include "code/ai-daily/docs/2026-11-19.md" >}}
