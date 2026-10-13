---
title: "Mega MoE 实现解析（一百六十九）：MoE 一个训练步的分段成本账 + 共享专家反向的同图闭环"
date: 2026-10-14T06:45:00+08:00
draft: false
weight: 169
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "FP8", "grouped GEMM", "反向", "共享专家", "shared expert", "wgrad", "dgrad", "训练步", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/169-moe-train-step-ledger.md" >}}
