---
title: "Mega MoE 实现解析（一百五十）：把 wgrad 的 S 步梯度累积折进收缩维 K——每步累积税能不能消掉"
date: 2026-10-14T02:05:00+08:00
draft: false
weight: 150
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "grouped GEMM", "FP8", "big-K", "H100", "性能", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/150-wgrad-bigk-fold.md" >}}
