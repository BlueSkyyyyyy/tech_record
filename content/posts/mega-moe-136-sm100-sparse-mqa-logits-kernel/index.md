---
title: "Mega MoE 实现解析（一百三十六）：SM100 sparse MQA logits 主 kernel 逐段精读——gather 式 KV 访问、双 Q 槽打包与压缩列映射"
date: 2026-10-13T22:30:00+08:00
draft: false
weight: 136
series: ["mega-moe"]
tags: ["DeepGEMM", "MQA logits", "稀疏注意力", "SM100", "cp.async", "smem swizzle", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/136-sm100-sparse-mqa-logits-kernel.md" >}}
