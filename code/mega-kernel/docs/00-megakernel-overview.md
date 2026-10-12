# MegaKernel 是什么：动机、定义与谱系

> 本文是「MegaKernel 资料与实现分析」的第 1 篇。先讲清 **kernel-per-operator 执行模型**的损失、
> MegaKernel 的定义与它想解决的问题，作为后续 MPK 论文精读与案例分析的坐标系。

## 1. 现状：kernel-per-operator

今天的推理/训练系统把模型表示为一张**张量程序 DAG**：节点是算子（matmul、attention、norm、activation…），
边是张量。主流执行方式是**一个算子一个 kernel**：每个算子要么手写优化，要么由编译器生成，然后按依赖顺序
依次 launch 到同一 stream 上。

这套模型简单、模块化，但有三类系统性损失：

1. **kernel barrier（内核间屏障）**：同一 stream 上相邻 launch 之间，GPU 会隐式等待前一个 kernel
   **所有** thread 完成才开始下一个。这保证了正确性，却**禁止了跨算子的软件流水（software pipelining）**，
   依赖算子只能严格串行。→ 计算与访存/通信无法跨算子重叠。
2. **launch 开销**：每个 kernel 有固定的 launch 延迟；小算子多时，CPU/GPU 两侧的 launch 开销占比可观。
   （CUDA Graphs 主要治这一项。）
3. **中间张量 HBM 往返**：相邻算子之间通过 HBM 上的中间张量传数据，写入再读出，既耗带宽又耗时间。

## 2. 已有缓解手段及其边界

| 手段 | 治什么 | 局限 |
|---|---|---|
| 算子融合（fuse） | 减少中间张量与 launch | 手工、适配面窄，通常只融相邻小算子 |
| CUDA Graphs | 消除 launch 开销 | **仍有 kernel barrier**，不改变串行性 |
| PDL（Programmatic Dependent Launch） | 让相邻 kernel 部分重叠 | 粗粒度、需要 kernel 配合 |
| 多 stream | 无依赖算子并行 | 有依赖的算子仍串行；调度粒度粗 |

结论：这些手段都没有触及**「以 SM 为粒度的跨算子调度与重叠」**这个根问题。

## 3. MegaKernel 的定义

**MegaKernel（巨型内核 / persistent mega-kernel）**：把（一整段甚至整个模型的）多个算子
**融合进一个持久化 kernel**——kernel 一次 launch、在 GPU 上长期驻留（persistent），
内部用**去中心化的 in-kernel 调度**把各算子的工作分给不同 SM / warp，并在 kernel 内实现
**跨算子的软件流水**与**通信-计算重叠**。

关键词：**persistent**（一个 kernel 跑完所有任务）、**单 kernel**（消除 kernel barrier）、
**SM 级调度**（粒度细到 SM）、**in-kernel 运行时**（调度在 device 上做，不靠 host 反复 launch）。

## 4. 谱系（从融合到什么程度）

1. **单算子内部优化**：一个 kernel 内软件流水、warp specialization（如 FlashAttention、DeepGEMM GEMM）。
2. **段融合 / 算子融合**：把相邻几个算子融成一个 kernel（如 fused MoE、fused norm+residual）。
3. **通信-计算重叠的双 kernel**：如 DeepEP + GEMM 分步、FlashMLA split（两个 kernel，靠 stream 重叠）。
4. **MegaKernel（整段/整图）**：整张图一个持久化 kernel、SM 级调度。代表：
   - **MPK / Mirage Persistent Kernel**（论文 *MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*，
     arXiv 2512.22219）：首个**自动**把多 GPU 模型推理编译成单个 mega-kernel 的编译器 + 运行时，
     SM 级图表示 + 去中心化 in-kernel runtime，相对 vLLM/SGLang 最高 **1.7×** 更低延迟、TP 下最高 **10×** 吞吐。
   - **DeepGEMM Mega MoE / Mega Gate / Mega mHC**：生产级「段级 mega-kernel」——把 dispatch + 两段 GEMM +
     SwiGLU + combine 融进单个 kernel，重叠 NVLink 通信与 Tensor Core。
   - **Kimi K3** 报告里把「fused mega-kernels」列为内核优化任务的一类。

## 5. 本专题怎么读

- 先读 **MPK 论文精读**（`docs/01-mpk-paper.md`）：SM 级图表示、编译器、in-kernel runtime、评估。
- 再看**生产案例**（`docs/02-production-cases.md`）：DeepGEMM mega 家族、Kimi K3。
- 然后做**相关技术对照**（`docs/03-related-tech.md`）：CUDA Graphs / PDL / persistent kernel。
- 最后是**实验**（`docs/04`）与**一页纸总结**（`docs/05`）。

> 术语提醒：社区里 “mega-kernel / megakernel / mega kernel” 常混用；本专题统一用 **MegaKernel**，
> 指「整段/整图融合 + 持久化 + SM 级调度」这一类，而非仅仅「两个算子融合」。
