---
title: "Mega MoE 实现解析（一百四十四）：前向 step 的通信-计算重叠真机量化——侧流 NCCL 不重叠，SM 预算分区才重叠"
date: 2026-10-14T00:30:00+08:00
draft: false
weight: 144
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "前向", "通信-计算重叠", "NCCL", "NVLink", "grouped GEMM", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/144-forward-overlap.md" >}}
