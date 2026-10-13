---
title: "Mega MoE 实现解析（一百六十二）：多卡 EP MoE 的真实通信字节口径——DeepEP 字段级拆解、真实 per-token 布局与 dtype 伸缩（H100 实测）"
date: 2026-10-14T05:00:00+08:00
draft: false
weight: 162
series: ["mega-moe"]
tags: ["DeepEP", "DeepSeek", "MoE", "EP", "专家并行", "FP8", "通信", "a2a", "CUDA", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/162-ep-comm-bytes.md" >}}
