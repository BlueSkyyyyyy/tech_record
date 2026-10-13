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
26. [MPK 前端的模型组装与请求/元数据管理层走读]({{< relref "mega-kernel-26-mpk-model-builder" >}}) —— 谁把 95 个 layer 拼成一个模型：builder 注册/组装、`MPKMetadata`、TP/EP 切分与集合通信选择
27. [MPK 模型家族 builder 走读：MLA 吸收、FP8 UE8M0、MoE 与投机 draft]({{< relref "mega-kernel-27-mpk-family-builders" >}}) —— 家族结构差异如何在 builder 层被"折平"成标准 task 管线：DeepSeek-V3 / GLM-4.6 / Inkling / GPT-OSS 与 DFlash/Eagle3
28. [MPK 权重预处理层走读：从 HF checkpoint 到 builder 的 state_dict]({{< relref "mega-kernel-28-mpk-weight-prep" >}}) —— MLA 权重吸收、V 反吸收进 o_proj、gate/up 交织、MoE 专家折叠、FP8 重定标与 TP 切分，以及离线/内联两条路径的 shard 规则差异
29. [MPK 运行层「可执行规范」走读：`test_mode` 套件与 `PersistentKernel` API]({{< relref "mega-kernel-29-mpk-testmode" >}}) —— `test_mode` 如何把常驻 kernel 变成「跑一次就退」的测试装置，fork/join/case 2·3/residual-stripping 的可执行规范，以及 C1–C5 的运行层见证
30. [MPK 单 kernel 单测层走读：`runtime_kernel_wrapper` 与 `tests/runtime_python`]({{< relref "mega-kernel-30-mpk-unit-tests" >}}) —— 夹在 device task 与 megakernel 端到端之间的「一个 task = 一整颗 kernel」回归网：五套 pybind extension、大面积注册失配，以及 `MIRAGE_UNIT_TEST` 宏的语义
31. [MPK 单测 wrapper 家族走读：blackwell 家族与 cute/hopper 的两类落地]({{< relref "mega-kernel-31-mpk-unit-wrappers" >}}) —— 14 个 SM100 目录的扩展名/符号/测试三方对照（同名扩展撞名、缺源文件、standalone 程序），以及「一个 task = 一颗 kernel」的单 CTA/cluster 形态与 `extern "C"`/CUTLASS 两条非 pybind 路线
32. [MPK FP8 单测 wrapper 与 TP standalone 走读：UE8M0 scale 布局契约与机械 port]({{< relref "mega-kernel-32-mpk-fp8-wrappers" >}}) —— 把 `sm100_linear_fp8`/`sm100_fp8_moe`/`sm100_mla_mtp_decode` 两个重目录打开：UE8M0 packed scale 的 row-major/DeepGEMM 列主序布局探测层、UTCCP+UMMA 的 warp 特化、2D grid 手推指针（emulating `runtime.cc`），以及 TP=2/4/8 对外部 standalone 的机械 port
33. [MPK SM100 手工线性 task 走读：`linear_fp8_sm100` 家族与生产布线]({{< relref "mega-kernel-33-mpk-sm100-linear" >}}) —— 把 SM100 block-scaled FP8 线性 task 的 device 身体打开（TMA 生产者/UMMA leader/UTCCP scale 转置/TMEM epilogue 四角色、两条独立身体），并与生产 `register_task` 的 `kNumSMs=1`/`kNumStages=25` 适配逐参数对照
34. [MPK SM100 block-scaled FP8 MoE group-GEMM device 内部走读]({{< relref "mega-kernel-34-mpk-fp8-group-gemm" >}}) —— 把 grouped FP8 MoE GEMM 的 device 身体逐段打开（DMA 的 TMA+手写 swizzle `cp.async`+routing 检查、MMA 的 UTCCP+4×block-scaled UMMA+描述符 `__shfl` 广播、SCALE 的 fp32→UE8M0→warp 转置、epilogue 的 TMEM→BF16→scatter），并与生产 `register_moe_fp8_sm100_task`（`TaskType` 248/249、`MMA_N=16`、8 stage）及 DeepGEMM `make_sf_desc`/`utccp transpose` 静态对应
35. [MPK SM100 第二套 device 身体：CuTe 线性 task 与 MLA-MTP decode 走读]({{< relref "mega-kernel-35-mpk-sm100-cute" >}}) —— 把 `docs/32` §7 的两条尾巴一起走完：CuTe 高层版 FP8 线性 task（身体 B，727 行，无调用者的第二实现）的四段流水，以及 DeepSeek-V3 MTP 投机解码 MLA attention + reduce 家族（QK→softmax→PV→online rescale，1 单卡 + TP2/4/8），并揭示常驻 kernel 内跨 task 复用的三类 device 适配（TMEM alloc warp 一致、named barrier ID 保留区、255→128 线程守卫）
36. [MPK SM100 MLA attention 家族走读：KV gather / decode / reduce / prefill]({{< relref "mega-kernel-36-mpk-sm100-mla" >}}) —— 把 MLA attention 段的其余 device 身体打开：paged→连续的 KV-cache gather 两变体（含扁平化 ILP 优化注释）、被 PR 651 MTP 版顶替的旧 decode/reduce（同族内死代码，静态 `grep` 证据）、吸收版 prefill（cp.async + m16n8k16、1-stage 受常驻 SMEM 预算约束、dual-dispatch gate）与未吸收 TP8 prefill（TMA K/V + 128 线程命名屏障），并对照生产布线与 DeepSeek-V3 `_use_prefill` 选路
37. [MPK SM100 2SM MLA attention 走读：整颗 CUTLASS FMHA kernel 与 2SM 协作]({{< relref "mega-kernel-37-mpk-sm100-mla-2sm" >}}) —— 打开 MLA 家族里唯一的「整颗 CUTLASS kernel」形态：`kIs2Sm=true` 派生 cluster `(2,1,1)`/`Allocator2Sm`/`2SmSm100` 调度、只有 leader CTA 发 UMMA、TMEM 对半切、softmax 的 2×2 数据并行 peer 归约（SMEM 交换 + 命名屏障）、七 warp 五 pipeline 的 QK→softmax→PV→rescale→epilogue，以及它为何只走单 kernel 单测层、从未进生产注册表

38. [MPK SM100 非 GEMM / 非 MLA task 家族走读：路由、top-k、采样、位置编码、归一化、量化与小算子]({{< relref "mega-kernel-38-mpk-sm100-misc" >}}) —— 把常驻 megakernel 里「轻但必需」的薄 task 逐个打开：MoE 路由三兄弟（group-aware 向量化 `topk_sigmoid`、160 专家标量 `glm_moe_router`、logsigmoid-softmax `inkling_moe_router`）、argmax/sampling 两段式（partial+reduce）与 softmax_gather/prob_scatter 簿记、SMEM 原地 RoPE/RMSNorm 与融合 partial RoPE、UE8M0 per-token 分组量化（含 `blockIdx.x == worker id` 的 row-identity 修复）、以及非 MLA 的通用 paged / DFlash / Inkling attention，并核对它们在 `TaskType`/`graph.cc`/`persistent_kernel.py`/builder 四层的生产布线

39. [MPK 共享原语层走读：`tasks/common/*` 与 `ampere/mma.cuh`]({{< relref "mega-kernel-39-mpk-common-primitives" >}}) —— 打开被所有 device task 反复 `#include` 却从未逐行读的 arch-agnostic 地基：FlashInfer 移植的一趟式 Gumbel-Max 采样（借 CUB `BlockReduce::Sum` 当 argmax 归约）、bf16/PTX/SMEM 助手、`cp.async`/`ldmatrix` 原语（正是本专题取 task 实测的 device 本体）、全局显存逻辑布局包装，以及跨代共用的手写 `m16n16k16` MMA 及其 load-bearing 的 `"+f"` in-place 累加器修复（决定 EAGLE3 投机解码 token 流正确）

> 各篇正文由仓库 `code/mega-kernel/docs/` 下的 Markdown **实时内联**渲染，随代码与实测一起更新。
