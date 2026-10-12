---
title: "Mega MoE 实现解析（二十四）：SM100 dense MQA logits 的分页版 metadata 负载均衡调度器（SM100 主 kernel 仅分析 + H100 真实调度器实测）"
date: 2026-10-13T00:00:00+08:00
draft: false
weight: 24
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "稀疏注意力", "MQA", "调度器", "负载均衡", "分页KV", "varlen", "SM100", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/24-mqa-logits-paged-schedule.md" >}}
