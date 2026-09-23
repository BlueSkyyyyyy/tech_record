---
title: "FlashAttention 反向（七）：对标分析——FA2 / FA3 / TE 与目标形状"
date: 2026-09-22
draft: false
weight: 8
series: ["fa-bwd"]
tags: ["flash-attention", "flash-attention-3", "transformer-engine", "反向", "ncu", "算子优化", "系列"]
categories: ["算子开发"]
---

写完了自己的实现，下一步就是「和谁比、差多少、差在哪」。这篇先把「和谁比」讲清楚：
pip 里的 `flash_attn 2.7.4` 是 **FA2（Ampere 内核）**，在 H100 上应该对标 **FA3（SM90，wgmma+TMA）**。
然后给出 FA2 / FA3 / TE 三方在目标形状（GQA/MQA/MLA）上的**纯反向时间**，并用 kernel 名、SASS 指令、
ncu SOL 三层证据解释快慢的来源，最后给出我们的差距与方向。

{{< fa_include "code/flash-attention/fa-bwd/docs/06-benchmark-and-shapes.md" >}}
