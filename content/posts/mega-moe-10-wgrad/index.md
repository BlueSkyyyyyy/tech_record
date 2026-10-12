---
title: "Mega MoE 实现解析（十）：MoE 反向 wgrad API（k_grouped tn/nt、contiguous/PSUM）"
date: 2026-10-12T22:35:00+08:00
draft: false
weight: 10
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "反向传播", "grouped GEMM", "wgrad", "k_grouped", "FP8", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/12-moe-backward-wgrad.md" >}}
