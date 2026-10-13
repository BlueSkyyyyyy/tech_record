---
title: "Mega MoE 实现解析（一百四十六）：wgrad 换 bf16 输出的精度代价——GEMM 快 2.27×，含归约只剩 1.54×"
date: 2026-10-14T01:05:00+08:00
draft: false
weight: 146
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "grouped GEMM", "bf16", "量化", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/146-wgrad-bf16-output.md" >}}
