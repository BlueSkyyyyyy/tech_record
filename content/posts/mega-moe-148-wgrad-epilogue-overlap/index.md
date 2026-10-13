---
title: "Mega MoE 实现解析（一百四十八）：把 wgrad 的 S 步归约藏进 GEMM——epilogue 融合 / 侧流重叠到底行不行"
date: 2026-10-14T01:50:00+08:00
draft: false
weight: 148
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "grouped GEMM", "bf16", "H100", "性能", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/148-wgrad-epilogue-overlap.md" >}}
