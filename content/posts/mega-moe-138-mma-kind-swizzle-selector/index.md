---
title: "Mega MoE 实现解析（一百三十八）：DeepGEMM 启发式的最底层公共选择器——元素位宽、TMA swizzle 模式与 GemmDesc 的 dtype→MmaKind 管道"
date: 2026-10-13T23:05:00+08:00
draft: false
weight: 138
series: ["mega-moe"]
tags: ["DeepGEMM", "启发式", "TMA swizzle", "MmaKind", "block_k", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/138-mma-kind-swizzle-selector.md" >}}
