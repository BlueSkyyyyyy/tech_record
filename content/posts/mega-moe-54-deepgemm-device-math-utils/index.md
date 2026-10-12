---
title: "Mega MoE 实现解析（五十四）：DeepGEMM 的设备端数学与工具层 —— UE8M0 取整、bf16 打包、warp/CTA 归约"
date: 2026-10-13T05:45:00+08:00
draft: false
weight: 54
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "UE8M0", "bf16", "warp reduce", "prefix sum", "CUDA", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/54-deepgemm-device-math-utils.md" >}}
