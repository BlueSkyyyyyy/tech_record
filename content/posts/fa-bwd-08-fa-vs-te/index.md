---
title: "FlashAttention 反向（八）：目标形状重点分析 & FA 为何比 TE 慢（ncu 归因）"
date: 2026-09-22
draft: false
weight: 9
series: ["fa-bwd"]
tags: ["flash-attention", "transformer-engine", "反向", "CUDA", "ncu", "算子优化", "系列"]
categories: ["算子开发"]
---

本篇回答两个问题：① 对 GQA/MQA/MLA 目标形状做**重点分析**（数值/性能/bound）；
② 仔细审查「FA 比 TE 慢很多」到底是不是**测试方法**的问题，并用 **ncu + SASS** 找到根因
（结论：`flash_attn 2.7.4` 是 **FA2/SM80(Ampere)** 内核，而 TE 在 H100 上走 **cuDNN SM90 `wgmma`+TMA** 内核）。

{{< fa_include "code/flash-attention/fa-bwd/docs/06-fa-vs-te-and-shapes.md" >}}
