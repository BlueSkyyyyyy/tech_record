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
3. [KDA 前向 K2 精读]({{< relref "flash-kda-03-fwd-k2" >}}) —— warp specialization（4 MMA + 1 LOAD + 1 STORE）、输入 3-stage / 输出 2-stage TMA pipeline、八个 Phase 数据流与状态更新精度
4. [KDA 前向数值对拍]({{< relref "flash-kda-04-numerics" >}}) —— kernel vs fp64 golden，多 shape / seed / 门控区间；状态 $S^\top$ 布局约定；golden/FlashKDA/FLA 三方对比与共模抵消
5. [KDA 反向：逐 token adjoint 推导与参考实现]({{< relref "flash-kda-05-bwd" >}}) —— 六个梯度（$dq,dk,dv,d\beta,dg,dh_0$）完整推导、gate 链，不依赖 autograd 的显式反向与 $10^{-10}$ 三方核验，并与 FLA `chunk_kda` 的 chunked 反向在 bf16 $D=128$ 对拍
6. [KDA 反向 kernel：FLA Triton 分解精读与目标卡 CUDA 反向设计]({{< relref "flash-kda-06-bwd-kernel" >}}) —— 5 个 Triton 反向 kernel 与分块反向公式逐式对照、记号映射实测、复用 K1 workspace 的目标卡 CUDA 反向方案
7. [KDA 前向性能实测与 ncu 剖析]({{< relref "flash-kda-07-perf" >}}) —— 统一口径（CUPTI device + wall）下 FlashKDA vs FLA `chunk_kda`，扫 T/H/B；FLA 的 ~0.8ms host 派发税、K2 沿 T 串行 $\propto T$ 的定量证据、kernel 级分解；`ncu` 逐 kernel 剖析 K1/K2 bound（TensorCore 均非瓶颈、放大后都撞 L2 带宽）
8. [KDA 前反向算子实现解析：一页纸汇总]({{< relref "flash-kda-08-summary" >}}) —— 公式、结构、误差、性能、反向要点速查表 + 踩坑清单

> 各篇正文由仓库 `code/flash-kda/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
