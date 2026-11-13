---
title: "AI 日报 2026-11-13：24 篇 HF Daily Papers — System One 的可靠性、双流世界模型与 32× 压缩"
subtitle: "SO-Arena 用 21 个模型证明最强 System One 可靠性只排第七；G-STAR 把离散 token 压到 32×、单张 H200 2.5 秒出 1024² 图；BirdDreams 把 BEV 变成预测结果，不再依赖 HD map"
date: 2026-11-13
draft: false
weight: 2
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "System One", "agent", "世界模型", "视频生成", "扩散模型", "优化器", "评测"]
categories: ["LLM算法"]
summary: "2026-11-13 的 HF Daily Papers 共 24 篇、364 upvotes（21:00 首轮仅 3 篇，23:30 补齐）。最热是 SO-Arena（57↑）：11 个领域、5 类能力、21 个模型，总准确率第一的 System One 在决策可靠性上只排第七。第二是 BirdDreams（41↑），透视视频与 BEV 视频双流联合生成，把场景 layout 变成预测结果，推理只要一张相机图、不要 HD map。第三是 G-STAR（39↑），证明 group size 应随空间压缩因子二次增长，做到 32×/有效 64× 压缩，1024² 单张 H200 2.5 秒、比此前离散 AR 快 20×。其余硬数字：Actor/Judge/Teacher 用 JEV 把 1.5B 策略的 ALFWorld 从 72.8% 教到 82.9%；SimpleHC 只多 2.4% 训练时间；TTB 渲染最高 23×；OVB 在 Cityscapes 零样本 73.9 对 69.9 mIoU；FOAM 只用 SOAP 一半的 optimizer state。"
---

先看总量：24 篇、364 upvotes，13 篇上双，11 篇只有 1–3 票，中间层几乎是空的。21:00 首轮只看到 3 篇，23:30 补跑后才到 24 篇，所以这是一份晚间快照，榜单本身还会继续变。

今天最值得读的不是某个 SOTA，而是两组对照。第一组在问「怎么量」：SO-Arena 把 System One 的能力谱系与决策可靠性拆开，21 个模型里总准确率第一的那位可靠性只排第七；DeepStress 则发现，检索文档全错时不少 HotpotQA 问题仍能被参数知识答对。第二组在问「结构放哪」：BirdDreams 把 BEV 从输入条件变成预测结果，WorldDraft 让代码搭 4D 世界、视频模型只做 renderer；G-STAR 与 TTB 则从 token 压缩和 test-time baking 两侧把生成成本打下来。

System One 这条线今天意外地完整：SO-Arena 给尺子，Actor/Judge/Teacher 说明不会做事的模型仍能当老师，NestOne 说最后六层可以删，JustRUA 则用 lean harness 把 RoboDojo 从 29.7 拉到 50.3。生成侧同一天给出 32×/64× 压缩、23× 渲染加速和 10 万条物理视频，但后两者的定量增益仍然偏少。

下面是正文：24 篇按主题分六组，组内按 upvotes 降序，每篇标了论文号、机构与 upvotes；数字只取自摘要，摘要没给的会明确写出来。

{{< doc_include "code/ai-daily/docs/2026-11-13.md" >}}
