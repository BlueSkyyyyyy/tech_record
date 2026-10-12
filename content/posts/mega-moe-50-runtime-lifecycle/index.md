---
title: "Mega MoE 实现解析（五十）：DeepGEMM 运行时对象的生命周期与资源所有权（lazy init / shutdown / cuBLASLt handle 与 workspace）"
date: 2026-10-13T05:10:00+08:00
draft: false
weight: 50
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "运行时", "生命周期", "cuBLASLt", "workspace", "atexit", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/50-runtime-lifecycle.md" >}}
