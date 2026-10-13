---
title: "Mega MoE 实现解析（一百五十二）：MoE grouped GEMM 在专家负载不均下的计算侧代价——grouped kernel 为什么几乎免疫"
date: 2026-10-14T02:40:00+08:00
draft: false
weight: 152
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "grouped GEMM", "负载均衡", "调度器", "H100", "性能", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/152-moe-grouped-load-imbalance.md" >}}
