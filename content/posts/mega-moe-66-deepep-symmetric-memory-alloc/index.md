---
title: "Mega MoE 实现解析（六十六）：DeepEP 的对称内存分配层——GPU/CPU 弹性段、跨进程 shard 与 NCCL 窗口的地基"
date: 2026-10-13T08:55:00+08:00
draft: false
weight: 66
series: ["mega-moe"]
tags: ["DeepEP", "DeepSeek", "MoE", "EP", "symmetric memory", "CUDA VMM", "NCCL", "pidfd", "CUDA", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/66-deepep-symmetric-memory-alloc.md" >}}
