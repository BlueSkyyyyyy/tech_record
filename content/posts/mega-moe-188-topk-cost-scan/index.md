---
title: "Mega MoE 实现解析（一百八十八）：MoE 的 top-k 到底买到了什么——路由数 k 对前向/反向成本与 grouped GEMM 效率的完整扫描"
date: 2026-10-14T10:12:00+08:00
draft: false
weight: 188
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "top-k", "grouped GEMM", "EP", "启发式", "H100", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/188-topk-cost-scan.md" >}}
