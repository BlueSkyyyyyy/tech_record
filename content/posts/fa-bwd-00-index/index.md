---
title: "FlashAttention 反向工程（专题）"
date: 2026-09-22
draft: false
weight: 1
series: ["fa-bwd"]
tags: ["flash-attention", "反向", "CUDA", "算子优化", "系列"]
categories: ["算子开发"]
---

这个专题做一件事：**从零把 FlashAttention 的「反向」写出来、测准、调快，并面向目标 AI 卡做移植**。

参考对象是 flash-attention（FA2/FA3）和 TransformerEngine（TE）；我们自己的实现（下称 **ours**）
支持 fp16 / bf16 / **fp8**，每种都给 **单文件**（好读）和 **两文件**（工程拆分）两个版本，
每个关键结论都有本机 H100 的实测数字和 `ncu` 证据。

- 配套代码：[`code/flash-attention/fa-bwd/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-attention/fa-bwd)
- 数值 I/O dump：`/home/xieminglin/proj/output/fa-bwd/`（CPU npy，便于 load 比对）

## 怎么读（由浅入深）

1. 先看**原理与手段**（一）：反向的数学、FA 的分段结构、以及有哪些优化手段。
2. 再看**实现**（二~五）：fp16 → bf16 → fp8 设计 → fp8 实现。每一步都先说「为什么慢」，再说「怎么改」。
3. 然后看**结果**（六）：所有 dtype、所有形状的数值与性能汇总。
4. 接着看**对标**（七）：和 FA2 / FA3 / TE 比，谁快、差多少、差在哪。
5. 想看**调优过程**（八）：一轮轮改了什么、瓶颈怎么迁移、哪些是负结果。
6. 最后看**移植**（九）：换到目标卡要改哪些层。

不需要先读源码；每篇都能独立看懂，看源码时也能一一对上。

## 目录

1. [实现结构与优化手段梳理（FA/TE 对照）]({{< relref "fa-bwd-01-catalog" >}}) —— 反向的数学、FA 的三段式、优化手段清单
2. [fp16 反向实现与优化]({{< relref "fa-bwd-02-fp16" >}}) —— 单/两文件，preprocess + 主 kernel
3. [bf16 反向实现与优化]({{< relref "fa-bwd-03-bf16" >}}) —— dtype 参数化 + bank conflict 那些坑
4. [FP8 反向设计]({{< relref "fa-bwd-04-fp8-design" >}}) —— E4M3/E5M2 + rowwise scaling 怎么定
5. [FP8 反向实现与优化]({{< relref "fa-bwd-05-fp8-impl" >}}) —— 张量核 mma 与逐轮优化（O1–O11）
6. [数值与性能汇总（GQA/MQA/MLA）]({{< relref "fa-bwd-06-summary" >}}) —— 对拍表 + 性能表 + ncu 结论
7. [对标分析：FA2 / FA3 / TE 与目标形状]({{< relref "fa-bwd-07-analysis" >}}) —— 纯反向口径、kernel/SASS/SOL 对照
8. [性能调优历程与方法论：从 138 ms 到 1.95 ms]({{< relref "fa-bwd-08-optimization-journey" >}}) —— 逐轮记录、墙迁移总表
9. [目标卡移植注意事项]({{< relref "fa-bwd-09-porting" >}}) —— 可替换层与移植 checklist

> 说明：以上各篇正文由仓库 `code/flash-attention/fa-bwd/docs/` 下的 Markdown **实时内联**渲染，
> 随代码与实测一起更新，不另行复制。
