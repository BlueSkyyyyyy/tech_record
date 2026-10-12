---
title: "Mega MoE 实现解析（五十八）：DeepEP combine 的归约 epilogue —— top-k 去重、rank/topk 双布局与 bf16 直加旁路"
date: 2026-10-13T06:55:00+08:00
draft: false
weight: 58
series: ["mega-moe"]
tags: ["DeepEP", "MoE", "EP", "combine", "归约", "CUDA", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/58-deepep-combine-reduce.md" >}}
