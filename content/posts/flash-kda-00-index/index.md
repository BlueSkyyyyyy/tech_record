---
title: "KDA 前反向算子实现解析（专题）"
date: 2026-10-12T20:00:00+08:00
draft: false
weight: 1
series: ["flash-kda"]
tags: ["KDA", "Kimi Delta Attention", "线性注意力", "CUDA", "算子", "系列"]
categories: ["算子开发"]
---

这个专题讲 **Kimi Delta Attention（KDA）** 的前向与反向算子：数学怎么来的、分块公式怎么推、
CUDA kernel 怎么写、数值和性能到底如何。

参考实现是 MoonshotAI 的 **FlashKDA**（`~/github/FlashKDA`，SM90+，`CHUNK=16`，$D=128$），
反向参照 `flash-linear-attention` 的 `chunk_kda`，并给出我们自己的目标卡 CUDA 反向设计。

- 配套代码：[`code/flash-kda/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-kda)

## 一条主线

> **KDA 的难点不在「算」，在「怎么把 16 步串行递推变成矩阵乘，又不把精度和带宽赔进去」。**

这句话可以拆成三个问题，整个专题就是回答它们：

1. **数学上**：逐 token 的 delta rule，怎么等价变换成块内矩阵运算？（第 1 篇）
2. **实现上**：一次前向为什么要拆成 K1、K2 两颗 kernel？各自怎么榨干 SM？（第 2、3 篇）
3. **工程上**：分块带来哪些数值坑（布局、bf16 状态、求逆精度）？反向怎么复用前向的中间量？（第 4–7 篇）

## 怎么读

按你的目的挑一条：

- **只想搞懂 KDA 在算什么**（30 分钟）：[数学与分块公式]({{< relref "flash-kda-01-theory" >}})
  → [一页纸汇总]({{< relref "flash-kda-08-summary" >}})。
- **要照着写 kernel**：[数学]({{< relref "flash-kda-01-theory" >}}) →
  [K1 精读]({{< relref "flash-kda-02-fwd-k1" >}}) → [K2 精读]({{< relref "flash-kda-03-fwd-k2" >}})
  → [数值对拍]({{< relref "flash-kda-04-numerics" >}})。
- **要做反向**：[反向推导]({{< relref "flash-kda-05-bwd" >}}) →
  [反向 kernel]({{< relref "flash-kda-06-bwd-kernel" >}}) →
  [性能与优化]({{< relref "flash-kda-07-perf" >}})。
- **只关心快不快**：[性能与 ncu 剖析]({{< relref "flash-kda-07-perf" >}})。

## 目录

| # | 篇目 | 一句话 | 你会拿到什么 |
|---|---|---|---|
| 1 | [数学与分块公式]({{< relref "flash-kda-01-theory" >}}) | 带逐通道遗忘的 delta rule，如何变成 16-token 分块 | 六步公式、源码地图、第一个数值自检 |
| 2 | [前向 K1：prepare]({{< relref "flash-kda-02-fwd-k1" >}}) | 一个 CTA 干完归一化、门、$L$/$M_{qk}$、块内求逆 | 九段流水逐段拆、16×16 求逆怎么写的、workspace 字节账 |
| 3 | [前向 K2：recurrence]({{< relref "flash-kda-03-fwd-k2" >}}) | 一个 CTA 串完整个序列，靠 warp 特化 + TMA 流水吃饱 | 六个 Phase 的数据流、`MOVM_T` 省了什么、状态更新精度 |
| 4 | [数值对拍]({{< relref "flash-kda-04-numerics" >}}) | 三个 oracle 交叉验证：fp64 golden / FLA / 官方 torch_ref | 误差表、$S^\top$ 布局坑、varlen 与 fp32 状态外壳 |
| 5 | [反向：推导]({{< relref "flash-kda-05-bwd" >}}) | 六个梯度的 adjoint 推导与纯 PyTorch 参考实现 | 逐 token 反向、gate 链、与 FLA 的三方核验 |
| 6 | [反向：kernel]({{< relref "flash-kda-06-bwd-kernel" >}}) | FLA Triton 反向逐 kernel 精读 + 目标卡 CUDA 反向设计 | 5 个反向 kernel 的公式对照、B1/B2/B3 的实测迭代 |
| 7 | [性能与优化]({{< relref "flash-kda-07-perf" >}}) | 前向 ncu 剖析、反向工程化、每一步优化的定量收益 | 端到端数字、瓶颈定位、优化日志（含失败的尝试） |
| 8 | [一页纸汇总]({{< relref "flash-kda-08-summary" >}}) | 公式 / 结构 / 误差 / 性能 / 反向要点速查 | 贴墙上的那一页 |

## 这个专题的可信度怎么保证

- **公式**：分块前向与逐 token 递推在 fp64 下逐元素相等；分块反向与 autograd 对拍。
- **代码**：每个 `文件:行号` 都在 `~/github/FlashKDA` 上核对过；引用格式统一为 `文件名:行号`。
- **数字**：所有性能/误差都来自本机实测，原始输出留在 `code/flash-kda/harness/*.out.txt`，
  正文标注「实测」；纯推导/纯资料分析会明确标注。
- **失败也写**：走不通的路、被证伪的假设、踩过的坑，都在各篇的「踩坑」小节里，不粉饰。

> 各篇正文由仓库 `code/flash-kda/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
