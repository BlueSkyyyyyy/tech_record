---
title: "Mega MoE 实现解析（一百八十五）：数据并行 all-reduce 的逐位一致性——真实 NCCL 下 ring/树/world-size 都改 bit"
date: 2026-10-14T09:20:00+08:00
draft: false
weight: 185
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "NCCL", "数据并行", "可复现性", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/185-dp-allreduce-bits.md" >}}
