---
title: "Mega MoE 实现解析（一百四十九）：把 wgrad 的 EPI-B 落到真 kernel——多出来的 235MB/步读，到底藏住了吗"
date: 2026-10-14T01:55:00+08:00
draft: false
weight: 149
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "grouped GEMM", "bf16", "TMA", "H100", "性能", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/149-wgrad-epib-accumulate.md" >}}
