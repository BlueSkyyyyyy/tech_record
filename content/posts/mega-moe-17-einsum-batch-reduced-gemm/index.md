---
title: "Mega MoE 实现解析（十七）：DeepGEMM einsum 批归约 GEMM `bmk,bnk->mn`（跨 batch/rank 的权重梯度融合，H100 实测）"
date: 2026-10-12T23:40:00+08:00
draft: false
weight: 17
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "einsum", "wgrad", "grouped GEMM", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/17-einsum-batch-reduced-gemm.md" >}}
