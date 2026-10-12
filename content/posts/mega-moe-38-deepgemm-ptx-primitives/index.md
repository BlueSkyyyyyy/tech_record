---
title: "Mega MoE 实现解析（三十八）：DeepGEMM 的 PTX 原语底座——内存序 / 原子 / 共享内存搬运 / warp 通信（H100 实测）"
date: 2026-10-13T02:55:00+08:00
draft: false
weight: 38
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "PTX", "CUDA", "内存序", "原子操作", "mbarrier", "TMA", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/38-deepgemm-ptx-primitives.md" >}}
