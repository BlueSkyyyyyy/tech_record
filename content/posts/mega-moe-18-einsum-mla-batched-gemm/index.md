---
title: "Mega MoE 实现解析（十八）：DeepGEMM einsum MLA 投影 `bhr,hdr->bhd` / `bhd,hdr->bhr`（按头 batched GEMM，H100 实测）"
date: 2026-10-12T23:50:00+08:00
draft: false
weight: 18
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "einsum", "MLA", "batched GEMM", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/18-einsum-mla-batched-gemm.md" >}}
