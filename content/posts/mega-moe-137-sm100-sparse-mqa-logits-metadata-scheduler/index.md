---
title: "Mega MoE 实现解析（一百三十七）：SM100 sparse MQA logits 的 metadata kernel 逐段精读——两 token 归并去重、双 Q 槽打包与负载均衡调度"
date: 2026-10-13T22:40:00+08:00
draft: false
weight: 137
series: ["mega-moe"]
tags: ["DeepGEMM", "MQA logits", "稀疏注意力", "metadata", "负载均衡", "归并去重", "H100", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/137-sm100-sparse-mqa-logits-metadata-scheduler.md" >}}
