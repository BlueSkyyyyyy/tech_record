# MPK 论文精读：SM 级图表示、编译器与 in-kernel runtime

> 本文是「MegaKernel 资料与实现分析」的第 2 篇，精读 **MPK**（Mirage Persistent Kernel，
> arXiv:2512.22219v2）的表示、编译器与运行时。快照基准：本机
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（共 2717 行），
> 逐节核对行号；线上实现 `https://github.com/mirage-project/mirage`。
>
> 承接关系：动机/定义/谱系见 `docs/00-megakernel-overview.md`（§1–§4）；本文只做**论文本身**的精读，
> 不重复动机论证。生产案例（DeepGEMM mega 家族、Kimi K3）见 `docs/02-production-cases.md`，
> DeepGEMM MoE 的前/反向实现细节见 `../mega-moe/` 专题——本文遇到时只给坐标、不展开。
>
> 进度：**P2-1 已完成**（§3，下文 §2）；P2-2 编译器（§4）、P2-3 in-kernel runtime（§5）、
> P2-4 评估（§6）待续。

## 1. 全篇结构一页（方便定位）

| 论文节 | 标题 | 快照行 | 一句话 |
|---|---|---|---|
| §1 | Introduction | 52 | kernel-per-operator 三损失 + mega-kernel 三好处 + MPK 两组件 |
| §2 | Background | 196 | GPU 编程模型（kernel barrier）、kernel fusion |
| **§3** | **SM-Level Graph Representation** | **345** | **tGraph：以 SM 为粒度的 task/event 交替图** |
| §4 | The MPK Compiler | 449 | tGraph 生成 → task 实现生成 |
| §4.1 | tGraph Generation | 458 | 分解→依赖分析→event fusion→normalization→linearization |
| §4.2 | Task Implementation Generation | 1126 | 以 thread block 为单位 superoptimization |
| §5 | In-Kernel Parallel Runtime | 1147 | 单 kernel 内 scheduler/worker 去中心化执行 |
| §6 | Evaluation | 1369 | 端到端 / MoE / 多 GPU / 消融 |

（更细的章节地图与图/表/算法清单见 `docs/00` §6。）

## 2. §3 SM 级图表示：tGraph（P2-1）

§3 全节只有约 100 行（快照行 **345–448**），却是整篇论文的表示层核心。它回答一个问题：
**要把整张算子图塞进一个持久化 kernel，需要一种什么粒度的依赖表示？**

### 2.1 动机：算子级依赖太粗（行 347–354）

原文（行 347–354）：tGraph「expresses the computation of a tensor program at the granularity of
individual streaming multiprocessors (SMs)」，而传统计算图「expose dependencies only between tensor
operators」。这种细粒度表示「exposes additional parallelism and enables optimizations such as
**cross-operator software pipelining** and **fine-grained kernel overlap**」，二者「are not supported
by the existing kernel-per-operator execution model」。

这与 `docs/00` §1.2（损失一：kernel barrier 禁止跨算子软件流水）是一体两面：
kernel barrier 之所以挡住跨算子流水，本质是**依赖被表达在算子（kernel）粒度**；
MPK 的解法不是换执行引擎，而是**换表示粒度**——把依赖下沉到 SM。

### 2.2 定义：task / event 交替（行 360–370）

Figure 4 的图（快照行 294–298 图题）明确：「transforms a kernel-level computation graph into an
optimized SM-level tGraph」。节点两类：

- **task**（矩形）：一个**在单个 SM 上执行**的工作单元。
  - 蓝色 = compute，橙色 = communication（行 364）。
  - 对照 §4.1：task 由算子分解而来，每个 task 最终编译成一个 **thread block** 的 device function。
- **event**（绿色圆）：**跨 task 的同步**（行 364）。

结构约束（行 364–367，逐字核对）：

- task 与 event **交替**出现；
- 每个 task 只有到 **triggering events** 的出边，和来自 **dependent events** 的入边；
- **就绪条件**（行 368–370）：一个 task 在其 dependent events **全部 activated** 后 ready；
  task 完成时 **notify** 它的 triggering event；
- 一个 event 在收到**与它关联的所有 task** 的 notification 后 **activated**。

用一句话概括这套语义：**task 是执行体，event 是「一组 task 全完成」的屏障**，
但屏障的**范围是任意 task 集合**，而不是「整个 kernel」。这正是 kernel barrier（§2.1 行 259）的推广：
把「全 kernel 的粗屏障」换成「任意子集之间的细屏障」。

### 2.3 关键例子：MatMul → AllReduce 的细粒度重叠（行 371–386）

论文用多 GPU LLM serving 里最常见的 `MatMul` → `AllReduce` 说明问题（注意：§2.1 用的是
`MatMul` → `AllGather`，行 260–293，机制相同）：

- **现状**（行 378–379）：现有系统顺序执行两个 kernel，因为「coarse-grained **kernel barriers**
  synchronize entire kernels」——AllReduce 的**所有** thread block 要等 MatMul 的**所有**
  thread block 完成。
- **真实依赖其实更细**（行 379–382）：AllReduce 做 element-wise 通信+归约，因此
  **每个 AllReduce task 只依赖产出它输入 tile 的那一个 MatMul task**。
- **收益**（行 382–386）：在成对 task 间插入 fine-grained event，就能让 compute-intensive 的
  MatMul 与 communication-intensive 的 AllReduce **重叠**，提高 GPU 利用率
  （同时吃满 Tensor Core 与通信带宽）。

这直接对应 `docs/00` §1.2 的量化锚点（重叠 1.1×、流水 1.2–1.3×；§6.6 消融）。

### 2.4 同一计算图，多个 tGraph：图 4(c) 反例（行 387–401）

论文强调「**Multiple tGraphs may represent the same computation graph**」（行 387–390）。
Figure 4(c)（行 391–396）给出一个**次优** tGraph：event 只捕捉**算子级**依赖，
「analogous to traditional kernel barriers」——即退化回粗粒度。
所以表示能力不等于性能，**如何插入 event 才是优化点**；§4（行 397–401）正是讲 MPK 如何
「inferring **precise** data dependencies to maximize concurrency and minimize synchronization overheads」。

> 这条对工程实践很重要：megakernel 的收益不来自「合进一个 kernel」本身，而来自
> **event/依赖切得够细**。手工 mega-kernel（DeepGEMM 的 ring buffer + mbarrier）也是同一逻辑。

### 2.5 与 CUDA Graphs 的对比（行 402–414）

论文用单独一段「Comparison with CUDA Graphs.」做定位，逐点对比：

| 维度 | CUDA Graphs | tGraph |
|---|---|---|
| 实例化 | 静态 | 静态 |
| 依赖编码 | 显式，**kernel 级** | 显式，**SM task / sub-kernel event 级** |
| 表示内容 | kernel **launch 顺序** | intra- **and** cross-operator 依赖 |
| 同步机制 | stream 语义 | 显式 event，跨 SM 细粒度同步 |
| 融合/重叠边界 | 局限在 **kernel 边界** | 单 kernel 内即可重叠 compute 与 communication |

原文结论（行 411–414）：CUDA Graphs「confines overlap and fusion to kernel boundaries」，
而 tGraph「enabling fine-grained synchronization across SMs and overlap of computation and
communication **within a single kernel**」，因此能利用「parallelism that is inaccessible to
CUDA Graphs and other kernel-level execution models」。

这段是 `docs/00` §2.2（CUDA Graphs 边界）与 P4-1 的**论文字面依据**：
CUDA Graphs 治的是 **launch 开销**（把 N 次 launch 压成 1 次图提交），
但**不改变依赖粒度**，因而治不了 kernel barrier。

### 2.6 承接：Figure 5 的编译流水线（行 415–448）

§3 末尾紧接着 Figure 5 图题（行 415–448），预告 §4：

- Q/K/V/A/O/R = query/key/value/attention/output projection 与 RMSNorm 分解出的 **task 集合**（行 416–434）；
- D₁、D₂ = normalization 阶段插入的 **dummy task**，目的是「ensure that each task has a **single
  triggering event**」（行 435–445）——这是 P2-2 的关键细节；
- (f) 展示如何 **linearize** tGraph 并「uniform, canonical representation」存储（行 445–448）。

### 2.7 小结：tGraph 的四条设计要点

1. **粒度**：SM/task，而非算子/kernel——这是与一切 kernel 级模型的根本分野。
2. **同步单元**：event 是「任意 task 子集」的屏障，可细到「task↔task 一对一」。
3. **优化自由度**：同一计算图有多种 tGraph，质量取决于 event 插得是否 precise（图 4(c) 反例）。
4. **定位**：CUDA Graphs 的下沉延伸，但换来「单 kernel 内跨算子重叠」这一 CUDA Graphs 拿不到的能力。

> 与生产案例的呼应：DeepGEMM Mega MoE 的 L1→L2 依赖用位掩码（`scheduler/mega_moe.cuh`
> 行 140–160）手工表达 task 级依赖、用 ring buffer 调度（见 `docs/02` §1.1），
> 本质就是**手工版 tGraph**；MPK 的贡献是**自动**构造这种细粒度依赖。
> 这里的 MoE 细节不在本文展开，实现坐标见 `docs/02` 与 `../mega-moe/`。

> 实测说明：P2-1 为**纯资料分析**（论文 §3 精读），无 H100 实验。与 §3 相关的实测佐证
> （CUDA Graph 只治 launch 开销 1.56×、依赖链下 Graph 1.05× 不破串行）见
> `docs/04-experiments.md` §0 与 `docs/00` §2，原始输出 `harness/probe_env.out.txt`、
> `harness/mitigations.out.txt`。

## 3. P2-2 编译器（§4）

（待续）

## 4. P2-3 in-kernel runtime（§5）

（待续）

## 5. P2-4 评估（§6）

（待续）
