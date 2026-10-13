---
title: "Mega MoE 实现解析（一百六十五）：跨 micro-batch 流水线能不能藏 EP 通信——NCCL 不行、SM 预算下才行"
date: 2026-10-14T05:50:00+08:00
draft: false
weight: 165
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "DeepEP", "EP", "NCCL", "NVLink", "通信-计算重叠", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/165-ep-pipeline-overlap.md" >}}
