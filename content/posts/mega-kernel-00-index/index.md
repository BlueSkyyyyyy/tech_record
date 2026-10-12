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
2. [MPK 论文精读（一）：SM 级图表示 tGraph]({{< relref "mega-kernel-02-paper" >}}) —— task/event 交替、与 CUDA Graphs 对比
3. [生产案例：DeepGEMM mega 家族、Kimi K3 与 hpc-ops]({{< relref "mega-kernel-03-cases" >}}) —— 段级 mega-kernel 的源码坐标与范式
4. [相关技术对照：CUDA Graphs / PDL / persistent kernel]({{< relref "mega-kernel-04-related" >}}) —— 各类缓解手段消了什么、没消什么
5. [实验：launch 开销、两算子融合、PDL 链式流水]({{< relref "mega-kernel-05-experiments" >}}) —— 目标卡 H100 实测
6. [MegaKernel 一页纸总结：原理、设计 checklist 与适用边界]({{< relref "mega-kernel-06-summary" >}}) —— 收官汇总
7. [MPK 线上实现源码走读：worker/scheduler 运行层]({{< relref "mega-kernel-07-mpk-source" >}}) —— 论文落到可核对源码
8. [MPK 单 task device 实现走读：Hopper task 内部流水]({{< relref "mega-kernel-08-mpk-tasks" >}}) —— 一个 task = 一个 CTA 的 TMA/wgmma 流水
9. [MPK 源码走读：依赖分析器与 tGraph 构建]({{< relref "mega-kernel-09-mpk-compiler" >}}) —— 最近写者 SSA、event fusion、linearization
10. [MPK 构建与端到端冒烟测试]({{< relref "mega-kernel-10-mpk-build" >}}) —— 在目标卡 H100 上从零构建 mirage 并 launch 最小 megakernel
11. [手工 MegaKernel 先例精读：FlashMoE 与 Llama-1B megakernel]({{< relref "mega-kernel-11-precedents" >}}) —— 把 MPK 转引的两个手工 mega-kernel 回到一手资料

> 各篇正文由仓库 `code/mega-kernel/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
