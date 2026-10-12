---
title: "Mega MoE 实现解析（三十三）：DeepGEMM 的 JIT 运行时 —— DeepJIT 的编译 / 缓存 / 加载 / 发射管线（H100 真机实测）"
date: 2026-10-13T01:50:00+08:00
draft: false
weight: 33
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "DeepJIT", "JIT", "kernel cache", "FNV1a", "nvcc", "cubin", "CUDA driver", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/33-deepjit-runtime.md" >}}
