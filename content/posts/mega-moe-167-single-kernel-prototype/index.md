---
title: "Mega MoE 实现解析（一百六十七）：单 kernel 内通信+计算 warp specialization 原型——融合的前提是异步张量核"
date: 2026-10-14T06:20:00+08:00
draft: false
weight: 167
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "DeepEP", "Mega MoE", "warp specialization", "寄存器预算", "setmaxnreg", "WGMMA", "NVLink", "通信-计算重叠", "系列"]
categories: ["算子开发"]
---

{{< doc_include "code/mega-moe/docs/167-mega-moe-single-kernel-prototype.md" >}}
