---
title: "Mega MoE 实现解析（一百五十五）：把反向的 SwiGLU 段折进 dgrad 的 epilogue（H100 实测）"
date: 2026-10-14T01:00:00+08:00
draft: false
weight: 155
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "反向", "dgrad", "SwiGLU", "epilogue", "FP8", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/155-fuse-activation-into-dgrad.md" >}}
