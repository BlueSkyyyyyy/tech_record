---
title: "Mega MoE 实现解析（三十二）：DeepGEMM 的 SM90 FP8 dense GEMM 主 kernel —— 1D1D vs 1D2D（H100 真机实测）"
date: 2026-10-13T01:45:00+08:00
draft: false
weight: 32
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "FP8", "block scale", "1D1D", "1D2D", "WGMMA", "TMA", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/32-sm90-fp8-gemm.md" >}}
