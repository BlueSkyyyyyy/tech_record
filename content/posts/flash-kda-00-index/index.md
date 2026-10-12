---
title: "KDA 前反向算子实现解析（专题）"
date: 2026-10-12T20:00:00+08:00
draft: false
weight: 1
series: ["flash-kda"]
tags: ["KDA", "Kimi Delta Attention", "线性注意力", "CUDA", "算子", "系列"]
categories: ["算子开发"]
---

这个专题逐层拆解 **Kimi Delta Attention（KDA）** 的**前向**与**反向**算子实现。

参考实现是 MoonshotAI 的 **FlashKDA**（`~/github/FlashKDA`，SM90+，`CHUNK=16`，$D=128$）；
反向以 `flash-linear-attention` 的 `chunk_kda` 与自行推导为参照。我们会先把数学与分块公式讲清，
再逐阶段精读 CUDA kernel，给出可运行的 PyTorch 参考实现、数值对拍与实测性能。

- 配套代码：[`code/flash-kda/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-kda)

## 目录

1. [KDA 数学与分块公式]({{< relref "flash-kda-01-theory" >}}) —— gated delta rule、门控构造、CHUNK=16 分块前向
2. [KDA 前向 K1 精读]({{< relref "flash-kda-02-fwd-k1" >}}) —— TMA 载入 / L2 norm / 门激活+cumsum / 衰减 / $L$、$M_{qk}$ / 块内求逆 / 写 workspace
3. [KDA 前向 K2 精读]({{< relref "flash-kda-03-fwd-k2" >}}) —— warp specialization（4 MMA + 1 LOAD + 1 STORE）、输入 3-stage / 输出 2-stage TMA pipeline、状态搬运
4. KDA 前向 K2 八个 Phase 与状态递推（待写）
5. 数值对拍 —— 与逐 token 参考、FLA 对比（待写）
6. KDA 反向推导与实现 —— adjoint、chunked backward（待写）
7. 性能与 `ncu` 剖析（待写）

> 各篇正文由仓库 `code/flash-kda/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
