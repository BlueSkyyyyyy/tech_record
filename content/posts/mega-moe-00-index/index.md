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
13. [Mega Gate 与 Mega mHC：两个 MoE 相关融合算子]({{< relref "mega-moe-13-mega-gate" >}}) —— 打分 GEMM+topk 路由、TMEM 打分、sqrt-softplus、warp 级确定性 top-k、重复专家映射；HC 残差混合、Sinkhorn、RMSNorm、H100 语义对拍
14. [总结：前反向一页纸与实测证据台账]({{< relref "mega-moe-14-summary" >}}) —— 数据流图、关键机制、与朴素实现开销对比、反向要点、H100 实测汇总
15. [反向 grouped GEMM 性能归因：wgrad 为何只有 dgrad 的 1/5]({{< relref "mega-moe-15-backward-profiling" >}}) —— ncu 硬指标、访存/计算受限的 roofline、fp32 权重梯度与累积读回
16. [HyperConnection prenorm GEMM：系数投影 + RMSNorm 平方和融合]({{< relref "mega-moe-16-hc-prenorm-gemm" >}}) —— TF32 语义（实测 RNE）、平方和搭便车、split-K、访存受限 roofline、H100 实测
17. [DeepGEMM einsum 批归约 GEMM `bmk,bnk->mn`：跨 batch/rank 的权重梯度融合]({{< relref "mega-moe-17-einsum-batch-reduced-gemm" >}}) —— batch 并入收缩轴、split-K + atomicAdd 归约（非确定性实测）、SM100 TMA reduce-add、H100 性能
18. [DeepGEMM einsum MLA 投影 `bhr,hdr->bhd` / `bhd,hdr->bhr`：按头 batched GEMM]({{< relref "mega-moe-18-einsum-mla-batched-gemm" >}}) —— 头=group 的 3D TMA、K/MN-major 角色、带 padding 的权重、TMA reduce-add 累加、H100 性能
19. [DeepGEMM 稀疏注意力 indexer 的 MQA logits：连续 / 分页路径]({{< relref "mega-moe-19-attention-mqa-logits" >}}) —— 加权 ReLU MQA 打分、Q 在 B 侧/KV 在 A 侧、逐 token fp32 KV scale、两级 warp 归约、PDL、H100 实测
20. [BF16 Mega MoE 变体精读：`bf16_mega_moe`]({{< relref "mega-moe-20-bf16-mega-moe" >}}) —— 一套骨架两种 MmaKind、去掉 SF/UTCCP/UE8M0、block_k=64、L1 in-place SwiGLU（bf16 in/out）、L2 RNE cast、host 配置与缓冲区实测
21. [DeepGEMM FP8 einsum MLA 投影 `bhr,hdr->bhd`：block-scale 版按头 batched GEMM]({{< relref "mega-moe-21-fp8-einsum-projection" >}}) —— A per-token 1D / B per-block 2D SF、复用 `1d2d` 的 Batched 分支、逐头 SFB 基址、gran 128 契约、SM100 独占 FP8 输出对、H100 实测 581 TFLOPS
22. [稀疏 MQA logits：稀疏注意力 indexer 的第二级打分]({{< relref "mega-moe-22-sparse-mqa-logits" >}}) —— 选中 KV 块内打分并压缩输出、metadata 合并去重与负载均衡调度、contiguous TMA 快路、SM100 仅分析 + H100 语义实测（压缩列映射逐位对齐）

> 各篇正文由仓库 `code/mega-moe/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
