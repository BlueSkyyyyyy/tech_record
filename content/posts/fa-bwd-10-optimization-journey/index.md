---
title: "FlashAttention 反向（十）：性能调优历程与方法论（从 138 ms 到 1.95 ms）"
date: 2026-09-22
draft: false
weight: 11
series: ["fa-bwd"]
tags: ["flash-attention", "反向", "CUDA", "性能调优", "ncu", "算子优化", "系列"]
categories: ["算子开发"]
---

这一篇把 ours 反向从「能跑对」调优到「尽量快」的**完整过程与思路**整理出来：每一轮改了什么、
`ncu` 说瓶颈在哪、打掉之后**墙迁移到哪**、收益多少、哪些是负结果。含「墙迁移总表」与可复用经验。
主指标统一用**时间**（ms/µs）。

{{< fa_include "code/flash-attention/fa-bwd/docs/08-optimization-journey.md" >}}
