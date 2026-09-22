---
title: "FlashAttention 反向工程（专题）"
date: 2026-09-22
draft: false
weight: 1
series: ["fa-bwd"]
tags: ["flash-attention", "反向", "CUDA", "算子优化", "系列"]
categories: ["算子开发"]
---

本专题整理 **FlashAttention 反向（bwd）** 的实现、优化与对照分析：借鉴 flash-attention（FA2/FA3）
与 TransformerEngine 的实现，把反向整理成 **单文件 / 两文件** 两种形式（fp16 / bf16 / **fp8 最重点**），
编译、用 ncu 剖析 bound、与 TE 做数值与性能对标。

- 目标硬件：目标 AI 卡（当前在 H100 sm90 上开发验证，代码按可替换层抽象便于移植）
- 配套代码：[`code/flash-attention/fa-bwd/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-attention/fa-bwd)
- 数值 I/O dump：`/home/xieminglin/proj/output/fa-bwd/`（CPU npy，便于 load 比对）

## 本专题目录

1. [实现结构与优化手段梳理（FA/TE 对照）]({{< relref "fa-bwd-01-catalog" >}}) —— FA2/FA3 反向结构、优化手段清单、FP8 反向难点
2. [fp16 反向实现与优化]({{< relref "fa-bwd-02-fp16" >}}) —— preprocess + 1colblock，单/两文件
3. [bf16 反向实现与优化]({{< relref "fa-bwd-03-bf16" >}}) —— dtype 参数化 + bank conflict padding
4. [FP8 反向设计]({{< relref "fa-bwd-04-fp8-design" >}}) —— E4M3/E5M2 + rowwise scaling
5. [FP8 反向实现与优化]({{< relref "fa-bwd-05-fp8-impl" >}}) —— 张量核 mma 与逐轮优化（O1–O4、O2b）
6. [数值与性能汇总（GQA/MQA/MLA）]({{< relref "fa-bwd-06-summary" >}}) —— 对拍表 + 性能表 + ncu bound
7. [目标卡移植注意事项]({{< relref "fa-bwd-07-porting" >}}) —— 可替换层与移植 checklist
8. [目标形状重点分析 & FA 为何比 TE 慢（ncu 归因）]({{< relref "fa-bwd-08-fa-vs-te" >}}) —— 纯反向口径、kernel/SASS/SOL 对照
9. [用 SM90 的 FA3 重测：更正「FA 比 TE 慢」]({{< relref "fa-bwd-09-fa3-sm90" >}}) —— FA3 编译 + FA3/FA2/TE 三方对比

> 说明：以上各篇正文由仓库 `code/flash-attention/fa-bwd/docs/` 下的 Markdown **实时内联**渲染，
> 随代码与实测一起更新，不另行复制。
