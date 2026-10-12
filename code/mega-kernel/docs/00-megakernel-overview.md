# MegaKernel 是什么：动机、定义与谱系

> 本文是「MegaKernel 资料与实现分析」的第 1 篇。先讲清 **kernel-per-operator 执行模型**的损失、
> MegaKernel 的定义与它想解决的问题，作为后续 MPK 论文精读与案例分析的坐标系。

## 1. 现状：kernel-per-operator 及其三类系统性损失

> 本文以 **MPK 论文**（*MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*，arXiv:2512.22219v2）
> 的快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（下称「快照」，共 2717 行）为准，
> 引用的章节号/行号均为在该快照中逐行核对的结果。与本仓库 P0-3 实测（`docs/04-experiments.md` §0）互相印证。

### 1.1 执行模型：张量程序 DAG，一个算子一个 kernel

今天的推理/训练系统把模型表示为一张**张量程序 DAG**：节点是张量代数算子（matmul、attention、norm、
activation…），边是算子生产/消费的 *n* 维张量（MPK §1，快照行 53–56）。主流执行方式是**一个算子一个
kernel**（kernel-per-operator）：每个算子要么由领域专家手写优化，要么由 ML 编译器自动生成，然后按依赖
顺序依次 launch 到同一 stream 上（MPK §1，快照行 57–73）。

这套模型简单、模块化，但 MPK 明确指出它**限制了多项关键的跨算子 GPU 优化**（快照行 71–73）。下面把这三类损失
逐条展开，并给出论文里的量化锚点。

### 1.2 损失一：kernel barrier——禁止跨算子软件流水

同一 stream 上相邻 launch 之间，GPU 会隐式插入一个 **kernel barrier（内核间屏障）**：必须等前一个 kernel
的**所有** thread block 完成后，后一个 kernel 的 thread block 才能开始（MPK §1 快照行 74–77、§2.1
快照行 223–226）。它保证了数据依赖正确，代价是：

- **禁止跨算子软件流水（cross-operator software pipelining）**，迫使有依赖的算子严格串行；
- 现代 GPU 已高度异构（Tensor Core 做矩阵计算、TMA 做异步搬运、CUDA Core 做辅助运算），充分的软件流水
  需要把「数据搬运 / 矩阵计算 / 辅助运算」在**多个任务迭代之间**交错。而 kernel barrier 只允许在**单个
  task 内部**做流水（intra-task pipelining），跨 task 就断流，留下 pipeline bubble 让硬件利用率下降
  （MPK §2.1，快照行 227–240，对应 Figure 2a/2b，快照行 204–209）。
- **禁止细粒度 kernel 重叠（fine-grained kernel overlap）**：依赖只在**整个算子**粒度上表示，运行时必须
  等上游算子全部完成才能启动下游。以常见的 `MatMul`→`AllGather` 为例：AllGather 是逐元素操作，每个
  AllGather block 只依赖**一个** MatMul block 的输出，本可让不同 SM 并行跑两种算子，但 kernel barrier 要求
  AllGather 的所有 block 等 MatMul 的所有 block 完成（MPK §2.1，快照行 241–293，对应 Figure 3a/3b）。
- 论文给出该优化被剥夺后的代价：开启跨任务流水相对关闭可把 Qwen3-8B 最后一层 linear 的任务运行时降低
  **1.2–1.3×**（MPK §6.6，快照行 1586–1596，Figure 12）；开启细粒度计算-通信重叠相对关闭降低
  每迭代延迟 **1.1×**（MPK §6.6，快照行 1597–1615，Figure 13，4×H100 TP）。

NVIDIA 的 **PDL（Programmatic Dependent Launch）** 允许同 stream 上 kernel 间部分重叠，但 MPK 指出采用
PDL 需要显著工程改造，因为它根本上改变 kernel 结构与控制流（MPK §1，快照行 78–84）。

### 1.3 损失二：launch 开销——每 token 数百到上千次 launch

kernel-per-operator 每次推理迭代要 launch **数百到上千个 kernel**（MPK §1，快照行 87–89）。论文给出一组
可直接引用的量化锚点（MPK §6.6「Kernel-launch reduction」，快照行 1618–1628，测试于 B200）：

| 项 | 数值 |
|---|---|
| Qwen3-8B 每 token 的 kernel launch 次数（kernel-per-operator） | **293 次** |
| 单次 launch 成本（eager） | **3.8 µs** → 合计 **1.1 ms/token** |
| 单次 launch 成本（CUDA Graph） | **0.8 µs** → 合计 **0.2 ms/token** |
| MPK 的 in-kernel scheduler 占总运行时比例 | **0.28%** |

这组数字说明两件事：(a) launch 开销随算子数线性累积，在 decode 这种「每 token 计算量小」的场景占比可观；
(b) CUDA Graphs 能把单次成本压到约 1/5、总量压到约 1/5，但**并不能归零**——论文在 §6.3 明确指出，
即便用了 CUDA Graphs，kernel-per-operator 仍有 **kernel-switch 开销**（快照行 1504–1509）。MPK 用「单次
launch + device 内调度」把这部分开销换成一个占比 0.28%、由设备内原子操作与环形队列维护的轻量调度器
（实现细节见 MPK §6.1，快照行 1409–1413）。

**本机 P0-3 佐证**（`docs/04-experiments.md` §0，目标卡 H100/sm90）：200 个 trivial kernel 串行 launch
耗时 **0.575 ms**（约 2.9 µs/次）；而在一个常驻 kernel 内做 200 次 cooperative grid-wide 屏障只要
**0.330 ms**（约 1.65 µs/次）。虽是能力探测而非公平基准，量级上印证了「反复 launch 的固定成本 > 设备内同步」。

### 1.4 损失三：中间张量 HBM 往返

相邻算子之间通过 **HBM 上的中间张量**传数据：上游算子把结果写回 device memory，下游算子再读回来。
kernel 融合的收益正是「避免中间结果物化、减少 device memory 访问、并消除 launch 开销」（MPK §2.2，
快照行 311–313）。MPK 进一步指出现有编译器只能融合**局部小算子组**，因为「忠实实现整个复杂张量程序的
单个 kernel」在计算上极难、常常不可行（MPK §2.2，快照行 314–328）。mega-kernel 范式则把融合推到极致：
把整个张量程序的计算与通信融进一个持久化 kernel，用 device memory 上的同步原语跨 SM 协调（快照行 329–331）。

一个具体案例：Qwen3-30B-A3B 在 batch=1、SGLang 上，「为 TMA 把路由到同一 expert 的 token 聚成连续布局」
这一预处理步骤占 MoE 总执行时间的 **高达 11%**（MPK §6.4「Fused gather-GEMM」，快照行 1551–1553）。
MPK 把它融进 GEMM 的异步 token 级拷贝，消除独立 gather kernel 与额外的调度点。

### 1.5 三类损失与「已缓解但仍不够」的边界

| 损失 | 症状 | 已有缓解（详见 §2） | 为什么仍不够 |
|---|---|---|---|
| kernel barrier | 跨算子无法软件流水、无法细粒度重叠 | PDL、算子融合 | PDL 需改 kernel 结构；融合只覆盖局部算子 |
| launch 开销 | 每 token 293 次 launch，1.1 ms（eager） | CUDA Graphs（→0.2 ms） | 仍有 kernel-switch 开销；图是静态的，控制流/形状变化要重捕 |
| 中间张量 HBM 往返 | 上游写、下游读中间结果 | 算子融合 | 只能融相邻小算子，无法覆盖整图 |

MPK 在 §6.3 把这三项与「CPU 侧 page allocation / request scheduling 的 host–device 同步」并列为
kernel-per-operator 在**小模型 + 新 GPU**上更吃亏的原因（快照行 1501–1513）：计算量/算子越小、GPU 越快，
这些固定与串行开销占比越大。其给出的端到端锚点是 Qwen3-8B/A100：vLLM/SGLang 约 **14.5 ms/token**，
MPK **12.5 ms/token**，而按「16 GB 参数 ÷ 1.6 TB/s 带宽」估算的硬件下界约 **10 ms**（快照行 1513）。

> 观察：这三类损失并非独立——kernel barrier 让中间张量必须物化（损失三），物化 + 多次 launch 又共同
> 造成 launch 与访存开销（损失二）。MegaKernel 的思路是用**单个持久化 kernel + SM 级调度**同时消解三者的
> 共同根源：把「以整个算子为单位、靠 host 反复 launch 推进」换成「以 SM 任务为单位、在 device 内事件驱动
> 推进」。这正是 MPK §3 的 tGraph 表示与 §5 的 in-kernel runtime 要解决的（后续 P2 展开）。

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
