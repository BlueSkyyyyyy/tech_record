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

## 6. MPK 论文导航（章节地图）

下面这张地图基于本机快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
（arXiv:2512.22219v2，共 2717 行）逐节核对得到，供精读时按图索骥。

| 章节 | 标题（原文） | 快照行号 | 核心内容 |
|---|---|---|---|
| Abstract | — | 38 | 首个**自动**把多 GPU 推理编译成单个 mega-kernel 的编译器+运行时；SM 级图 + 去中心化 in-kernel runtime |
| §1 | Introduction | 52 | kernel-per-operator 三类损失；mega-kernel 的三个好处；MPK 两大组件 |
| §2 | Background | 196 |  |
| §2.1 | GPU Programming Model | 210 | kernel barrier；software pipelining（intra- vs cross-task）；fine-grained kernel overlap |
| §2.2 | Kernel Fusion | 311 | 融合推到极致 = mega-kernel；现有 mega-kernel 均为手工（FlashDMoE、Spector LLaMA-1B） |
| §3 | SM-Level Graph Representation | 345 | **tGraph**：task/event 交替、fine-grained event；与 CUDA Graphs 的对比 |
| §4 | The MPK Compiler | 449 | 编译流程总览（Figure 5） |
| §4.1 | tGraph Generation | 458 | operator decomposition→dependency analysis→event fusion（Def 4.1/4.2）→normalization→linearization |
| §4.2 | Task Implementation Generation | 1126 | 以 **thread block** 为单位做 superoptimization（复用 Mirage）生成 CUDA device function |
| §5 | In-Kernel Parallel Runtime | 1147 | 单 kernel 内跑完整 tGraph；SM 分为 workers / schedulers |
| §5.1 | Event-Driven Execution | 1175 | start event、activation 计数、事件驱动传播、拓扑无关 |
| §5.2 | Hybrid Task Launch | 1232 | **JIT vs AOT** 的取舍；data-dependent 算子标 JIT、全局 barrier 之后标 AOT；双队列 |
| §5.3 | Runtime Optimizations | 1310 | paged shared memory（32 KB/页）、cross-task software pipelining、task description 预取 |
| §6 | Evaluation | 1369 | 四个问题：端到端、动态负载、多 GPU、消融 |
| §6.1 | Implementation Details | 1391 | `torch.compile(backend=MPK)`；44K C++/42K CUDA/10K Python；NVSHMEM；decentralized scheduling |
| §6.2 | Experimental Setup | 1455 | 5 个模型 × A100/H100/B200；prompt 64 / 生成 1024 |
| §6.3 | End-to-end Results | 1467 | 单 batch 1.0–1.7×；Qwen3-8B/A100 14.5→12.5 ms（下界约 10 ms） |
| §6.4 | Case Study: Mixture-of-Experts | 1520 | hybrid workload balancer、fused gather–GEMM |
| §6.5 | Multi-GPU Results | 1556 | TP 下 1.1–1.4×（vs SGLang/vLLM）、最高 10×（vs PyTorch） |
| §6.6 | Ablation Study | 1583 | cross-task pipelining 1.2–1.3×；compute–comm overlap 1.1×；launch 次数 |
| §6.7 | Compiler-Stage Ablation | 1629 | decomposition 每算子 32–47 任务；event fusion 37–118×；linearization 4.4–15× |
| §7 | Discussion | 1739 | 资源占用（时间复用、寄存器取最坏值）、接入手写 kernel、跨架构移植 |
| §8 | Related Work | 1766 | manual kernels / ML compilers / mega-kernels / compute–communication overlap |
| §9 | Conclusion | 1919 |  |

> 注意：路线图早期把「评估」写成 §5——**实际评估在 §6**，§5 是 in-kernel runtime；本导航以论文为准。

### 关键图 / 表 / 算法清单

- **Figure 1**（行 139）：MPK 总体架构（compiler + in-kernel runtime）。
- **Figure 2**（行 208）：intra-task vs cross-task software pipelining 的差别——kernel barrier 只在 task 内流水。
- **Figure 3**（行 253）：`MatMul`→`AllGather` 为何能做 fine-grained overlap（每个 AllGather block 只依赖一个 MatMul block）。
- **Figure 4**（行 294）：kernel-level 计算图 → SM-level **tGraph**（节点是 task/event，而非算子）。
- **Figure 5**（行 415 a–f）：MPK 编译流水线，含 dummy task 与线性化后的存储布局。
- **Figure 6**（行 798 a/b）：normalization 的两种重写（fan-out→1、fan-in→1）。
- **Figure 7**（行 1139）：事件驱动执行模型（event↔task 交替推进）。
- **Figure 8**（行 1230 a/b）：JIT vs AOT task launch 的同步步数差异。
- **Figure 9**（行 1465）：端到端吞吐，5 模型 ×3 代 GPU。
- **Figure 10**（行 1518）：MoE（Qwen3-30B-A3B/B200）runtime（µs）。
- **Figure 11**（行 1554）：多 GPU（Qwen3-1.7B/4×H100 TP）对比。
- **Figure 12**（行 1581）：cross-task pipelining 消融（B200）。
- **Figure 13**（行 1616）：compute–communication overlap 消融（4×H100）。
- **Algorithm 1**（行 952）：tGraph linearization（BFS，保证同一 event 触发的 task 在 `T` 中连续）。
- **Table 1**（行 1437）：三种 GPU 的 SM/worker/scheduler 配置。
- **Table 2**（行 1687）：每编译阶段统计（Ops / Tasks-per-op / Events / Fusion / Linearization 缩减）。

### 一图记住 MPK 的两层结构

1. **编译层**：tensor program →（分块→细粒度依赖分析→event fusion→normalization→linearization）→
   静态 tGraph →（按 thread block 做 superoptimization）→ 每个 task 的 CUDA device function。
2. **运行层**：一个持久化 mega-kernel 内，SM 固定划分为 **schedulers**（每 SM 4 个 scheduler warp，
   维护 event queue）与 **workers**（每 SM 一个 worker，维护 JIT/AOT 双 task queue）；
   事件被激活→入 scheduler 队列→派发 task→task 完成→通知触发事件，循环推进直到整图跑完。
