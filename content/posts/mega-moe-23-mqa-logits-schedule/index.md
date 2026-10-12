---
title: "Mega MoE 实现解析（二十三）：SM100 dense MQA logits —— 主 kernel 与连续版负载均衡调度器（SM100 主 kernel 仅分析 + H100 真实调度器实测）"
date: 2026-10-13T00:00:00+08:00
draft: false
weight: 23
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "稀疏注意力", "MQA", "调度器", "负载均衡", "block-scale", "SM100", "GPU kernel", "算子", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/23-mqa-logits-sm100-schedule.md" >}}
