---
title: "Mega MoE 实现解析（十五）：反向 grouped GEMM 性能归因（wgrad 为何只有 dgrad 的 1/5）"
date: 2026-10-12T23:20:00+08:00
draft: false
weight: 15
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "反向传播", "grouped GEMM", "wgrad", "dgrad", "性能分析", "ncu", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/15-moe-backward-profiling.md" >}}
