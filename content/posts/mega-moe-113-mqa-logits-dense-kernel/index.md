---
title: "Mega MoE 实现解析（一百一十三）：SM100 dense MQA logits 主 kernel 逐段精读（TMEM 头块遍历 / bf16x2 归约 / 分页生产者均分）"
date: 2026-10-13T18:20:00+08:00
draft: false
weight: 113
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "MQA logits", "sparse attention", "TMEM", "UMMA", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/113-mqa-logits-dense-kernel.md" >}}
