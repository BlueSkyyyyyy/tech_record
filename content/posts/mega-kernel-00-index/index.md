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
12. [MegaKernel in-kernel scheduler 设计手册]({{< relref "mega-kernel-12-sched-design" >}}) —— 把 P42–P69 的调度器微基准收口成设计决策表
13. [MegaKernel 的 SM 资源分配与异构 task 共驻成本手册]({{< relref "mega-kernel-13-coresidency" >}}) —— 把 P83–P97 的异构共驻微基准收口成 SM 资源/成本决策表
14. [MegaKernel 端到端骨架与 worker 数据面能力手册]({{< relref "mega-kernel-14-endtoend-dataplane" >}}) —— 把 P71–P82 的端到端骨架与数据面旋钮收口成能力/成本决策表
15. [MPK 前端与算子层走读：从 mirage 图 API 到 task_configs / bGraph]({{< relref "mega-kernel-15-mpk-frontend" >}}) —— 补齐运行层/device/依赖分析器三篇的上游：手工构图的 task API
16. [MPK 自动路径源码走读：Mirage superoptimizer 与 thread-block 级生成]({{< relref "mega-kernel-16-mpk-superoptimizer" >}}) —— 论文 §4.1/§4.2 的「自动分解 + thread-block 超优化」在 `src/search`/`src/transpiler` 的落点
17. [MPK 搜索器辅助模块走读：e-graph 等价类、区间推导、符号化图]({{< relref "mega-kernel-17-mpk-search-passes" >}}) —— 主搜索循环之外的三块剪枝/抽象子系统（含「已实现未接线」证据）
18. [MPK 多卡/NVSHMEM 路径源码走读：把通信编进同一个常驻 mega-kernel]({{< relref "mega-kernel-18-mpk-multigpu" >}}) —— 跨卡 event、两种 AllReduce（AllGather+Reduce / NVLS tile）与「多卡被迫单 kernel」
19. [MPK 自动路径（下）源码走读：transpiler 如何把 thread-block 图编成 CUDA]({{< relref "mega-kernel-19-mpk-transpiler" >}}) —— 运行时/device/依赖分析/搜索之外的下半段：图归一化、五步 resolve、schedule/swizzle/memory plan 与 warp-specialized 代码生成
20. [MPK 自动路径（runtime 库）源码走读：被生成 CUDA 调用的 device / thread-block 模板库]({{< relref "mega-kernel-20-mpk-runtime" >}}) —— `kn::`（整算子兜底，GEMM 走 cuBLAS）与 `tb::`（拷贝原子/epilogue 链/mbarrier 流水/wgmma·UMMA）两层，以及「同一套硬件惯用法在仓库里实现了两遍」
21. [MPK 的另一条 codegen 后端源码走读：Triton transpiler 与 Triton 路径]({{< relref "mega-kernel-21-mpk-triton-backend" >}}) —— `superoptimize(backend="triton")` → 子进程 profile → `TritonTranspiler` → Python/Triton 模块，以及「Triton codegen 也实现了两遍」
22. [MPK 自动路径（Blackwell 版）源码走读：2SM UMMA + TMEM + cluster codegen]({{< relref "mega-kernel-22-mpk-blackwell-transpiler" >}}) —— Hopper 后端升格为「一对 CTA 的 2SM UMMA」：cluster 布局、multicast TMA、TMEM 累加、`tcgen05.commit` 会合与 `launch_kernel_on_cluster`
23. [MPK 源码走读手册：从模型到常驻 mega-kernel 的完整编译栈]({{< relref "mega-kernel-23-compiler-manual" >}}) —— 把 docs/06–08、14–21 十一篇源码走读收口成一张编译栈总图、判据总映射、双链分叉与「已实现未启用 / 论文落差」清单
24. [MegaKernel 专题总纲 v2：结论地图、工程坑与实测校准]({{< relref "mega-kernel-24-capstone" >}}) —— 把 P0–P108 的 100+ 增量压成五层总图 + 六张实测校准表 + 工程坑清单 + 适用边界 v2
25. [MPK 生产路径的 task 目录：`TaskType` 全表与代际 ID 分区]({{< relref "mega-kernel-25-mpk-task-catalog" >}}) —— 把 `register_task` 的 95 分支大表打开：116 条枚举、92 个被登记 `TaskType`、模型家族 ID 块与死码候选

> 各篇正文由仓库 `code/mega-kernel/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
