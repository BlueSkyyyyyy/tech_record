---
title: "AI 日报 2026-11-23：26 篇 HF Daily Papers — 流混合的几何收缩、穿透率 74%→16%、空间记忆的 50.5%"
subtitle: "cmHC 用 Birkhoff 多面体的几何收缩治超连接的两难，Ascend 融合算子降延迟 44.8%；NoPeWorld 只读冻结 DiT 的隐状态就把穿模率从 74% 压到 16%；ReSpaceBench 里最强零样本模型 50.5%、人类 90.1%"
date: 2026-11-23
draft: false
weight: 5
series: ["ai-daily"]
tags: ["AI 日报", "系列", "论文速览", "模型架构", "世界模型", "多模态", "具身智能", "agent", "视频生成", "扩散模型"]
categories: ["LLM算法"]
summary: "2026-11-23 的 HF Daily Papers 共 26 篇、384 upvotes：8 篇上双、7 篇 ≥20 票，前三名 190 票占 49.5%、前十名占 84.1%。cmHC（94↑）把 Birkhoff 多面体向内部收缩，解决超连接「混不够/混过头」，NPU 融合算子降延迟 44.8%；NoPeWorld（55↑）在冻结 DiT 隐状态上加预测头，六个世界模型平均穿模率 74%→16%；Future-OPSD（41↑）用未标注视频自蒸馏，开放式预测 +14.2、闭环规划 +15.9 点；Clover-35B-A3B（37↑）以 3B 激活拿到 SWE-bench Verified 73.6、单卡 680 tok/s。"
---

今天只有 26 篇、384 票，是近期最少的一天，但票数反而更集中：前三名占 49.5%、前十名占 84.1%，cmHC 一篇就占全天四分之一。少而尖——26 篇里有 20 篇给出了可核对的具体数字。

三条主线：残差流的几何约束（cmHC）、物理常识可以被读出来（NoPeWorld 74%→16%）、评测开始区分「记住了」和「算对了」（ReSpaceBench 50.5% vs 人类 90.1%，RadioVISTA 的 field-path dissociation）。副线是扩散语言模型 Clover 与全双工 agent Duopra——后者的 Pass@1 从 4.7% 拉到 44.7%，说明全双工的瓶颈不在语音延迟，而在请求发起。

{{< doc_include "code/ai-daily/docs/2026-11-23.md" >}}
