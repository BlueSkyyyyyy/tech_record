---
title: "Mega MoE 实现解析（三十一）：SM100 GEMM 的 epilogue —— store 路径与四个 epilogue 算子（H100 逐位/统计实测）"
date: 2026-10-13T01:40:00+08:00
draft: false
weight: 31
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "epilogue", "随机舍入", "FP8量化", "UE8M0", "TMEM", "SM100", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/31-sm100-gemm-epilogue-operators.md" >}}
