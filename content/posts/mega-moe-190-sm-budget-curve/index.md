---
title: "Mega MoE 实现解析（一百九十）：MoE grouped GEMM 的 SM 预算曲线——给通信让出一个 SM，计算腿要付多少（H100 实测）"
date: 2026-10-14T10:45:00+08:00
draft: false
weight: 190
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "grouped GEMM", "SM 预算", "通信计算重叠", "persistent kernel", "H100", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/190-sm-budget-curve.md" >}}
