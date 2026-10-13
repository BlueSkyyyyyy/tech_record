---
title: "Mega MoE 实现解析（一百四十五）：wgrad 的「强制累积」代价——fp32 D 回读只占 48%，每专家 fp32 输出才是主因"
date: 2026-10-14T00:57:00+08:00
draft: false
weight: 145
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "grouped GEMM", "访存受限", "roofline", "ncu", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/145-wgrad-accum-cost.md" >}}
