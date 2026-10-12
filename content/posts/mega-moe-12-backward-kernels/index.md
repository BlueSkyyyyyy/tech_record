---
title: "Mega MoE 实现解析（十二）：MoE 反向 kernel 接入与端到端对拍（vs autograd）"
date: 2026-10-12T22:45:00+08:00
draft: false
weight: 12
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "反向传播", "grouped GEMM", "wgrad", "dgrad", "FP8", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/14-moe-backward-kernels.md" >}}
