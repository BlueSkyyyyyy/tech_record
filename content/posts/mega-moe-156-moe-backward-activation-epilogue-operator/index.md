---
title: "Mega MoE 实现解析（一百五十六）：把 SwiGLU 反向落地成 SM100 dgrad 的 epilogue_class operator（契约缺口 + H100 原型）"
date: 2026-10-14T02:00:00+08:00
draft: false
weight: 156
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "反向", "dgrad", "SwiGLU", "epilogue", "TMEM", "FP8", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/156-moe-backward-activation-epilogue-operator.md" >}}
