---
title: "MegaKernel 资料与实现分析（专题）"
date: 2026-10-12T20:20:00+08:00
draft: false
weight: 1
series: ["mega-kernel"]
tags: ["MegaKernel", "持久化内核", "MPK", "kernel fusion", "推理优化", "系列"]
categories: ["算子开发"]
---

这个专题研究 **MegaKernel**：把整张算子图融进**一个常驻 kernel**，让调度权从 host 移到 device、
粒度从「算子」细到「SM」。

主参考是 **MPK 论文**（*MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*，
arXiv 2512.22219）和它的实现 [`mirage-project/mirage`](https://github.com/mirage-project/mirage)，
辅以 **DeepGEMM** 的 mega 家族（Mega MoE / Mega Gate / Mega mHC）等生产案例，以及 CUDA Graphs /
PDL / persistent kernel 的对照实验。

- 配套代码：[`code/mega-kernel/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/mega-kernel)
- 姊妹专题：[Mirage 项目分析]({{< relref "mirage-00-index" >}})（同一个仓库的「怎么用、怎么编、怎么跑」）

## 一条主线

> **MegaKernel 不是「kernel 变大」，而是「调度权从 host 移到 device、粒度从算子细到 SM」。**

这条主线可以拆成三个问题：

1. **为什么要它**：一个算子一个 kernel，到底亏在哪三处？（第 1 篇）
2. **怎么做到**：一张算子图要经过哪几步，才能变成设备内可调度的 task 图？（论文 + MPK 编译栈）
3. **代价是什么**：常驻、融合、设备内同步，各自要付什么账？（实验手册 + 总纲）

## 怎么读

这个专题有 75 篇，**篇号是写作顺序，不是阅读顺序**。按目的挑：

- **只想搞懂「为什么」**（约 1 小时）：[第 1 篇]({{< relref "mega-kernel-01-overview" >}}) →
  [一页纸总结]({{< relref "mega-kernel-06-summary" >}}) → 有空再看
  [总纲 v2]({{< relref "mega-kernel-24-capstone" >}})。
- **想读懂 MPK 论文**：[第 1 篇]({{< relref "mega-kernel-01-overview" >}}) →
  [论文精读]({{< relref "mega-kernel-02-paper" >}}) →
  [编译栈总图]({{< relref "mega-kernel-23-compiler-manual" >}})。
- **想读 MPK 源码**：[编译栈总图]({{< relref "mega-kernel-23-compiler-manual" >}}) →
  [运行层]({{< relref "mega-kernel-07-mpk-source" >}}) →
  [依赖分析器]({{< relref "mega-kernel-09-mpk-compiler" >}}) →
  [单 task 实现]({{< relref "mega-kernel-08-mpk-tasks" >}})。
- **想看生产代码**：[DeepGEMM mega 家族]({{< relref "mega-kernel-03-cases" >}}) →
  [生产 task 目录]({{< relref "mega-kernel-25-mpk-task-catalog" >}}) →
  按家族挑 device 走读（见下方「E. device 身体」）。
- **想自己动手**：[实验篇]({{< relref "mega-kernel-05-experiments" >}}) →
  [构建实录]({{< relref "mega-kernel-10-mpk-build" >}}) → [Mirage 项目分析]({{< relref "mirage-00-index" >}})。

## 按主题找篇目

### A. 入门、谱系与结论（7 篇）

先把「为什么」和「结论」吃掉，剩下的都是证据。

| # | 篇目 | 一句话 |
|---|---|---|
| 1 | [MegaKernel 是什么]({{< relref "mega-kernel-01-overview" >}}) | 三笔账、五条判据 C1–C5、四档谱系 |
| 2 | [MPK 论文：tGraph]({{< relref "mega-kernel-02-paper" >}}) | SM 级依赖图长什么样，和 CUDA Graphs 差在哪 |
| 3 | [生产案例]({{< relref "mega-kernel-03-cases" >}}) | DeepGEMM 的段级 mega、Kimi K3、hpc-ops |
| 4 | [相关技术对照]({{< relref "mega-kernel-04-related" >}}) | CUDA Graph / PDL / 多流各治哪一处、为什么不够 |
| 6 | [一页纸总结]({{< relref "mega-kernel-06-summary" >}}) | 原理 + checklist + 适用边界 |
| 11 | [手工先例]({{< relref "mega-kernel-11-precedents" >}}) | FlashMoE 与 Llama-1B megakernel 怎么手写的 |
| 24 | [总纲 v2]({{< relref "mega-kernel-24-capstone" >}}) | 五层结论地图 + 实测校准 + 工程坑清单 |

### B. MPK 编译栈：从模型到 megakernel（11 篇）

一条流水线：构图 → 依赖分析 → event fusion → 线性化 → 生成 CUDA → nvcc。

| # | 篇目 | 一句话 |
|---|---|---|
| 9 | [依赖分析与 tGraph 构建]({{< relref "mega-kernel-09-mpk-compiler" >}}) | 最近写者 SSA、bbox 阴影、GCD/LCM 事件融合 |
| 10 | [构建与冒烟测试]({{< relref "mega-kernel-10-mpk-build" >}}) | 从零编 mirage 到 launch，含真实挂起排查 |
| 15 | [前端与算子层]({{< relref "mega-kernel-15-mpk-frontend" >}}) | `PersistentKernel` 的 layer 怎么变成 `task_configs` |
| 16 | [superoptimizer]({{< relref "mega-kernel-16-mpk-superoptimizer" >}}) | 自动搜索路径：候选生成、验证、导出 |
| 17 | [搜索辅助模块]({{< relref "mega-kernel-17-mpk-search-passes" >}}) | e-graph、区间推导、符号图三块剪枝 |
| 19 | [transpiler]({{< relref "mega-kernel-19-mpk-transpiler" >}}) | thread-block 图 → CUDA 的五个 resolve 阶段 |
| 20 | [runtime 模板库]({{< relref "mega-kernel-20-mpk-runtime" >}}) | 生成代码调用的 `kn::` / `tb::` 两层 |
| 21 | [Triton 后端]({{< relref "mega-kernel-21-mpk-triton-backend" >}}) | 同一张图的另一条 codegen 路线 |
| 22 | [Blackwell codegen]({{< relref "mega-kernel-22-mpk-blackwell-transpiler" >}}) | 2SM UMMA + TMEM + cluster 怎么生成 |
| 18 | [多卡 / NVSHMEM]({{< relref "mega-kernel-18-mpk-multigpu" >}}) | 通信为什么被迫并进同一个 kernel |
| 23 | [编译栈总图]({{< relref "mega-kernel-23-compiler-manual" >}}) | 两条链、双链分叉、论文落差清单 |

### C. 运行时与调度（5 篇）

kernel 起来之后，谁在派活、谁在干活、成本多少。

| # | 篇目 | 一句话 |
|---|---|---|
| 7 | [worker / scheduler 运行层]({{< relref "mega-kernel-07-mpk-source" >}}) | 事件驱动、原子环形队列、单次 launch |
| 8 | [单 task device 实现]({{< relref "mega-kernel-08-mpk-tasks" >}}) | 一个 task = 一个 CTA 的 TMA/wgmma 流水 |
| 12 | [in-kernel scheduler 手册]({{< relref "mega-kernel-12-sched-design" >}}) | ready 池、死锁边界、队列协议、预取深度 |
| 13 | [SM 共驻成本手册]({{< relref "mega-kernel-13-coresidency" >}}) | 异构 task 抢同一台机器的税从哪来 |
| 14 | [端到端数据面手册]({{< relref "mega-kernel-14-endtoend-dataplane" >}}) | 数据面能到多快、要多少 producer |

### D. 生产路径：模型 → task → 接线（10 篇）

MPK 在真实模型上怎么落地，以及「有实现 ≠ 已接线」的审计。

| # | 篇目 | 一句话 |
|---|---|---|
| 25 | [TaskType 全表]({{< relref "mega-kernel-25-mpk-task-catalog" >}}) | 92 个 task 类型与代际 ID 分区 |
| 26 | [模型组装层]({{< relref "mega-kernel-26-mpk-model-builder" >}}) | builder 注册、元数据、TP/EP 切分 |
| 27 | [模型家族 builder]({{< relref "mega-kernel-27-mpk-family-builders" >}}) | DeepSeek / GLM / Inkling / GPT-OSS 怎么折平 |
| 28 | [权重预处理]({{< relref "mega-kernel-28-mpk-weight-prep" >}}) | MLA 吸收、gate/up 交织、FP8 重定标 |
| 29 | [test_mode 套件]({{< relref "mega-kernel-29-mpk-testmode" >}}) | 让常驻 kernel「跑一次就退」的测试装置 |
| 30 | [单 kernel 单测]({{< relref "mega-kernel-30-mpk-unit-tests" >}}) | 一个 task = 一整颗 kernel 的回归网 |
| 31 | [单测 wrapper 家族]({{< relref "mega-kernel-31-mpk-unit-wrappers" >}}) | 14 个 SM100 目录的三方对照 |
| 32 | [FP8 wrapper 与 TP]({{< relref "mega-kernel-32-mpk-fp8-wrappers" >}}) | UE8M0 scale 契约与机械 port |
| 50 | [接线可达性审计]({{< relref "mega-kernel-50-mpk-wiring-audit" >}}) | 登记 → 发射 → 编入 → 调用四层交叉 |
| 51 | [多卡残差契约]({{< relref "mega-kernel-51-mpk-multigpu-residual" >}}) | TP 下残差为什么只能加一次 |

### E. device 身体：每种 task 怎么写（26 篇）

这一组是「肌肉」：GEMM、attention、MoE、norm、采样……每个家族的 kernel 逐行读。

| # | 篇目 | 一句话 |
|---|---|---|
| 33 | [SM100 FP8 linear]({{< relref "mega-kernel-33-mpk-sm100-linear" >}}) | TMA 生产者 / UMMA leader / UTCCP / epilogue 四角色 |
| 34 | [SM100 FP8 group-GEMM]({{< relref "mega-kernel-34-mpk-fp8-group-gemm" >}}) | MoE grouped GEMM 的 DMA/MMA/SCALE/epilogue |
| 35 | [CuTe linear + MLA-MTP]({{< relref "mega-kernel-35-mpk-sm100-cute" >}}) | 第二套 device 身体与三类 device 适配 |
| 36 | [MLA 家族]({{< relref "mega-kernel-36-mpk-sm100-mla" >}}) | KV gather / decode / reduce / prefill |
| 37 | [2SM MLA]({{< relref "mega-kernel-37-mpk-sm100-mla-2sm" >}}) | 整颗 CUTLASS FMHA + 2SM 协作 |
| 38 | [非 GEMM/MLA 家族]({{< relref "mega-kernel-38-mpk-sm100-misc" >}}) | 路由、采样、RoPE、量化、paged attention |
| 39 | [共享原语]({{< relref "mega-kernel-39-mpk-common-primitives" >}}) | 被所有 task include 的地基 |
| 40 | [投机解码 device 层]({{< relref "mega-kernel-40-mpk-speculative-decoding" >}}) | n-gram 查找、MTP 验证、Eagle3 commit |
| 41 | [退役的 paged attention]({{< relref "mega-kernel-41-mpk-deprecated-paged-attention" >}}) | 归档实现与今天的拆分解法对比 |
| 42 | [概率簿记 task]({{< relref "mega-kernel-42-mpk-prob-bookkeeping" >}}) | softmax_gather / prob_scatter / prob_extract |
| 43 | [MTP 验证链]({{< relref "mega-kernel-43-mpk-mtp-verify-chain" >}}) | scatter → verify → commit 的消费链 |
| 44 | [Eagle3 commit]({{< relref "mega-kernel-44-mpk-eagle3-commit" >}}) | verify-aware 写回与跨迭代快照 |
| 45 | [通用 paged attention]({{< relref "mega-kernel-45-mpk-paged-attention-sm100" >}}) | online softmax + 手写 1x4x1/1x1x4 MMA |
| 46 | [norm / rope 家族]({{< relref "mega-kernel-46-mpk-norm-rope" >}}) | 三代实现与「融合边界」选择 |
| 47 | [Eagle3 draft loop]({{< relref "mega-kernel-47-mpk-eagle3-draft-loop" >}}) | K=1 的并行链与 K>1 的退化 |
| 48 | [Eagle3 权重装配]({{< relref "mega-kernel-48-mpk-eagle3-weight-prep" >}}) | shuffle 布局与 pad→argmax→d2t 契约 |
| 49 | [段内融合 task]({{< relref "mega-kernel-49-mpk-fusion-tasks" >}}) | `norm_linear` / `silu_mul_linear` 与接线审计 |
| 53 | [2SM MLA 的 CUTLASS 侧]({{< relref "mega-kernel-53-mpk-mla-2sm-cutlass" >}}) | QK/PV atom、TMA 描述符、UMMA 序列 |
| 54 | [DFlash 家族]({{< relref "mega-kernel-54-mpk-dflash-family" >}}) | 非因果 block attention、per-head norm+RoPE |
| 55 | [Inkling 家族]({{< relref "mega-kernel-55-mpk-inkling-family" >}}) | 相对位置 bias attention、短卷积、路由 |
| 56 | [路由 task]({{< relref "mega-kernel-56-mpk-routers" >}}) | GLM 标量路由与 DeepSeek group-aware 路由 |
| 57 | [采样链]({{< relref "mega-kernel-57-mpk-sampling" >}}) | argmax 与 top-k/top-p 的 partial+reduce |
| 65 | [FP8 tile 调度器]({{< relref "mega-kernel-65-mpk-fp8-scheduler" >}}) | DeepGEMM 调度器的「单 CTA 全扫」改写 |
| 66 | [SM100 BF16 linear]({{< relref "mega-kernel-66-mpk-sm100-bf16-linear" >}}) | CuTe 描述符 + 手写 TMA 的混合体 |
| 67 | [SM100 BF16 MoE linear]({{< relref "mega-kernel-67-mpk-sm100-moe-linear" >}}) | cp.async gather + register scatter |
| 68 | [MLA-MTP 的 TP 三兄弟]({{< relref "mega-kernel-68-mpk-mla-mtp-tp" >}}) | per-rank 分区与 Oa/La 部分量 ABI |

### F. 存储布局与硬件原语契约（15 篇）

这一组回答「为什么同样的数算出来不一样」：swizzle、TMEM、描述符、稀疏元数据。

| # | 篇目 | 一句话 |
|---|---|---|
| 52 | [SMEM swizzle 三套写法]({{< relref "mega-kernel-52-mpk-smem-layout" >}}) | 布局类型就是 SMEM 的 ABI |
| 58 | [SM100 共享原语]({{< relref "mega-kernel-58-mpk-sm100-primitives" >}}) | tcgen05/TMEM 裸 PTX 与 DeepGEMM 派生封装 |
| 59 | [TMA 版 SMEM 布局]({{< relref "mega-kernel-59-mpk-tma-smem-layout" >}}) | 一个 (B,M,S) 同时喂 TMA/wgmma/软件 |
| 60 | [TMEM 列布局与 UTCCP]({{< relref "mega-kernel-60-mpk-tmem-utccp-layout" >}}) | FP8 scale 怎么从 SMEM 进 TMEM |
| 61 | [scale-factor 的 TMEM trait]({{< relref "mega-kernel-61-mpk-scale-sf-tmem-trait" >}}) | CUTLASS `tmem_sf_frg` ↔ MPK `sf_id` |
| 62 | [Hopper CuTe WS GEMM]({{< relref "mega-kernel-62-mpk-cute-hopper-ws-gemm" >}}) | 三份身体、一份活 |
| 63 | [TMEM fragment 另两面]({{< relref "mega-kernel-63-mpk-tmem-frg-traits" >}}) | 稀疏元数据 E 与 WS 的 C/D |
| 64 | [稀疏 UMMA 的 A 通路]({{< relref "mega-kernel-64-mpk-sparse-mma" >}}) | `sparse_elem<2,T>` / 稀疏 SMEM / 描述符 |
| 69 | [FP8 linear 支撑层]({{< relref "mega-kernel-69-mpk-fp8-support" >}}) | 描述符位编码、TMA 分派、两份副本 |
| 70 | [SF 段并置]({{< relref "mega-kernel-70-mpk-sf-segments" >}}) | dense linear vs grouped MoE 的 scale 布局 |
| 71 | [稀疏元数据 E]({{< relref "mega-kernel-71-mpk-sparse-metadata" >}}) | 2:4 的位编码与 A 物理槽置换 |
| 72 | [WS UMMA 指令现状]({{< relref "mega-kernel-72-ws-instruction" >}}) | CUTLASS 的 WS 布局 vs 死掉的 `tcgen05.mma.ws` |
| 73 | [UMMA 原语多实现]({{< relref "mega-kernel-73-umma-paths" >}}) | DeepGEMM `ptx::` vs MPK 几条通路 vs CUTLASS |
| 74 | [UMMA 描述符多路构造]({{< relref "mega-kernel-74-umma-descriptors" >}}) | 一个母本、两份派生、一份手写 |
| 75 | [DeepGEMM 的 UMMA 封装]({{< relref "mega-kernel-75-umma-homegrown" >}}) | 9 个 struct、两个 helper 逐条对照 |

### G. 实测（1 篇，但很长）

| # | 篇目 | 一句话 |
|---|---|---|
| 5 | [实验手册]({{< relref "mega-kernel-05-experiments" >}}) | launch 开销、融合、PDL、足迹、调度器、内存序……全部实测 |

## 可信度：这个专题的数字怎么来的

- **论文**：引用 MPK 快照（2717 行）的真实章节号与行号，逐条核对。
- **源码**：mirage 快照 commit `f9eb70c`（`mpk` 分支）；每个 `文件:行号` 都核对过。
- **实测**：目标卡 H100（sm90）。能跑的用容器实跑，原始输出留在 `code/mega-kernel/harness/*.out.txt`，
  正文标「实测」；跑不了的（如 SM100 专属指令）明确标「仅资料分析」。
- **失败也写**：构建失败、kernel 挂起、假设被证伪，都保留在相应篇目里。

> 各篇正文由 `code/mega-kernel/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
