---
title: "Mega MoE 实现解析（十一）：MoE 反向 dgrad API（m_grouped 与 contiguous/masked/PSUM）"
date: 2026-10-12T22:41:00+08:00
draft: false
weight: 11
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "反向传播", "grouped GEMM", "dgrad", "m_grouped", "FP8", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/13-moe-backward-dgrad.md" >}}
