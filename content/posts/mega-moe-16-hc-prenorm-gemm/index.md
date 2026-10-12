---
title: "Mega MoE 实现解析（十六）：HyperConnection prenorm GEMM（系数投影 + RMSNorm 平方和融合，H100 实测）"
date: 2026-10-12T23:35:00+08:00
draft: false
weight: 16
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "HyperConnection", "mHC", "TF32", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/16-hc-prenorm-gemm.md" >}}
