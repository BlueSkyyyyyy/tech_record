---
title: "MegaKernel 资料与实现分析（专题）"
date: 2026-10-12T20:20:00+08:00
draft: false
weight: 1
series: ["mega-kernel"]
tags: ["MegaKernel", "持久化内核", "MPK", "kernel fusion", "推理优化", "系列"]
categories: ["算子开发"]
---

这个专题系统梳理 **MegaKernel**：把整张算子图**融合进单个持久化 kernel**、在 **SM 级别**做跨算子
软件流水与通信-计算重叠的设计与研究。

以 **MPK（Mirage Persistent Kernel，arXiv 2512.22219）** 论文为主参考，辅以 **DeepGEMM** 的 mega 家族
（Mega MoE / Mega Gate / Mega mHC）等生产级案例，以及 CUDA Graphs / PDL / persistent kernel 的对照与实测。

- 配套代码：[`code/mega-kernel/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/mega-kernel)

## 目录

1. [MegaKernel 是什么：动机、定义与谱系]({{< relref "mega-kernel-01-overview" >}}) —— kernel-per-operator 的损失、定义、谱系
2. MPK 论文精读：SM 级图表示 / 编译器 / in-kernel runtime（待写）
3. 生产案例：DeepGEMM mega 家族、Kimi K3（待写）
4. 相关技术对照：CUDA Graphs / PDL / persistent kernel（待写）
5. 实验：launch 开销、persistent kernel、PDL（待写）
6. 一页纸总结（待写）

> 各篇正文由仓库 `code/mega-kernel/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
