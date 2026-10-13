---
title: "Mega MoE 实现解析（一百四十二）：前向 single-step 的 H100 预算归因——前向是 dgrad 的转置对偶（ncu roofline）"
date: 2026-10-13T23:55:00+08:00
draft: false
weight: 142
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "前向", "grouped GEMM", "roofline", "ncu", "SwiGLU", "UE8M0", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/142-forward-roofline.md" >}}
