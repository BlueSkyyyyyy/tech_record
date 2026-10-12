---
title: "Mega MoE 实现解析（三十九）：DeepGEMM 的 legacy Triton MoE grouped GEMM——被现代 C++ 取代的「显式索引 gather」（H100 实测）"
date: 2026-10-13T03:05:00+08:00
draft: false
weight: 39
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "Triton", "grouped GEMM", "wgrad", "dgrad", "CUDA", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/39-legacy-grouped-gemm.md" >}}
