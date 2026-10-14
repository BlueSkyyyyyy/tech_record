---
title: "Mega MoE 实现解析（一百八十四）：反向梯度累积的「切分 × 结合序」——wgrad 逐位敏感、dgrad 免疫"
date: 2026-10-14T09:10:00+08:00
draft: false
weight: 184
series: ["mega-moe"]
tags: ["DeepGEMM", "MoE", "反向", "梯度累积", "可复现性", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/184-moe-grad-accum-assoc.md" >}}
