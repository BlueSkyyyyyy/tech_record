---
title: "Mega MoE 实现解析（一百八十一）：反向的激活是存还是重算——重算逐位无损、省全部显存、预算可藏"
date: 2026-10-14T08:45:00+08:00
draft: false
weight: 181
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "反向", "grouped GEMM", "activation checkpointing", "重算", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/181-moe-activation-recompute.md" >}}
