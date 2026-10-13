---
title: "Mega MoE 实现解析（一百五十一）：MoE wgrad 的 FP8 累加器——把每步 RMW 从 8B 压到 2B 的误差-字节边界"
date: 2026-10-14T02:20:00+08:00
draft: false
weight: 151
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "Mega MoE", "反向", "wgrad", "FP8", "累积器", "H100", "性能", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/151-wgrad-fp8-accumulator.md" >}}
