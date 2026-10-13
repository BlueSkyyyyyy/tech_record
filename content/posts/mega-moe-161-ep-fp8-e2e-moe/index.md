---
title: "Mega MoE 实现解析（一百六十一）：多卡 EP MoE 接上真实 SM90 FP8 grouped kernel——EP 通信 × FP8 量化的端到端误差与预算（H100 实测）"
date: 2026-10-14T04:50:00+08:00
draft: false
weight: 161
series: ["mega-moe"]
tags: ["DeepGEMM", "DeepSeek", "MoE", "EP", "专家并行", "FP8", "grouped GEMM", "反向", "CUDA", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/161-ep-fp8-e2e-moe.md" >}}
