---
title: "DeepSeek Mega MoE 前反向算子实现解析（专题）"
date: 2026-10-12T20:10:00+08:00
draft: false
weight: 1
series: ["mega-moe"]
tags: ["MoE", "DeepGEMM", "DeepEP", "专家并行", "CUDA", "算子", "系列"]
categories: ["算子开发"]
---

这个专题拆解 DeepSeek 的 **Mega MoE**：把 **EP dispatch → linear1 → SwiGLU → linear2 → EP combine**
融合进**单个 mega-kernel**、并让 **NVLink 通信与 Tensor Core 计算重叠**的设计；
以及 MoE 的**反向**（weight-gradient / input-gradient 的 grouped GEMM）实现。

参考实现是 **DeepGEMM** 与 **DeepEP**。Mega MoE 主实现是 **SM100（Blackwell）专属**，
本机 H100 跑不了端到端，因此以源码精读 + 线上资料为主线，能跑的部分（grouped GEMM / backward）给实测。

- 配套代码：[`code/mega-moe/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/mega-moe)

## 目录

1. [MoE 与专家并行（EP）基础]({{< relref "mega-moe-01-basics" >}}) —— 计算图、grouped GEMM、dispatch/combine
2. [Mega MoE 整体架构与通信-计算重叠]({{< relref "mega-moe-02-architecture" >}}) —— 数据流、缓冲区、调度器、warp 分工
3. [DeepEP dispatch / combine 与 symmetric memory]({{< relref "mega-moe-03-deepep" >}}) —— 对称显存、expanded layout、通信实现
4. [H100 环境探测与 kernel 可用性清单]({{< relref "mega-moe-04-env" >}}) —— 哪些 kernel 本机可跑、哪些 SM100 专属
5. [主体 kernel 精读（一）：模板参数与 block config]({{< relref "mega-moe-05-kernel" >}}) —— grid/cluster/线程、BLOCK_M 启发式、smem 流水
6. [主体 kernel 精读（二）：warp specialization 与 mbarrier 同步]({{< relref "mega-moe-06-mbarrier" >}}) —— full/empty 相位、TMA 坐标、task 槽双缓冲
7. [主体 kernel 精读（三）：tcgen05 MMA 与 L1/L2 epilogue]({{< relref "mega-moe-07-mma-epilogue" >}}) —— UMMA/UTCCP、UE8M0 block scale、in-place SwiGLU、combine
8. [主体 kernel 精读（四）：dispatch / combine 与 NVLink、locality domain]({{< relref "mega-moe-08-dispatch-combine" >}}) —— 分布式前缀和认领、远端 pull/直写、readiness、分域调度
9. [MoE 反向：wgrad / dgrad 数学与 grouped GEMM 映射]({{< relref "mega-moe-09-backward" >}}) —— 逐层反向、SwiGLU/router 反向、k_grouped vs m_grouped
10. [MoE 反向 wgrad API：`k_grouped` tn/nt 与 contiguous/PSUM]({{< relref "mega-moe-10-wgrad" >}}) —— tn/nt major 差异、按组改 tensormap、SF 打包、H100 实测
11. [MoE 反向 dgrad API：`m_grouped` 与三种布局]({{< relref "mega-moe-11-dgrad" >}}) —— M 轴分组、contiguous/masked/PSUM 调度、训练分工、H100 实测
12. [MoE 反向 kernel 接入与端到端对拍（vs autograd）]({{< relref "mega-moe-12-backward-kernels" >}}) —— 四梯度接 grouped kernel、gather/量化、两层误差对拍、吞吐
13. [Mega Gate：BF16 打分 GEMM 与 top-k 路由的融合]({{< relref "mega-moe-13-mega-gate" >}}) —— 融合链路、TMEM 打分、sqrt-softplus、warp 级确定性 top-k、重复专家映射、H100 语义对拍
14. Mega mHC（待写）

> 各篇正文由仓库 `code/mega-moe/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
