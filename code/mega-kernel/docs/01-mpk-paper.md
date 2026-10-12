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
> 进度：**P2-1 已完成**（§3，下文 §2）；**P2-2 已完成**（§4，下文 §3）；
> **P2-3 已完成**（§5 + §6.1/§6.6，下文 §4）；**P2-4 已完成**（§6 + §7，下文 §5）。

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

## 3. §4 MPK 编译器（P2-2）

§4（快照行 **449–1146**）回答：**给定一张 tensor program 计算图和一份推理配置，如何自动生成 §3 定义的
tGraph，并为其中每个 task 生成 device function？** 若说 §3 是「表示层」，§4 就是「把表示生产出来的编译层」。
它的核心贡献是：把 §2.4 指出的「event 插得够不够 precise」这一**优化自由度**，从手工经验变成一条**可自动
执行、且表示开销可证明有界**的编译流水线。

### 3.1 §4 总览：输入、输出与两阶段（行 449–457）

原文（行 451–454）：MPK 编译器以「computation graph + associated inference configuration」为输入，
输出「an optimized tGraph **specialized for the target configuration and underlying GPU architecture**」，
端到端流程见 Figure 5（行 455–457）。两个阶段：

1. **§4.1 tGraph Generation**（行 458–1125）：把算子图编译成静态 tGraph；
2. **§4.2 Task Implementation Generation**（行 1126–1146）：为每个 task 生成一个可在单个 SM 上跑的
    CUDA device function。

一句话结构：**§4.1 决定「有哪些 task、谁等谁」，§4.2 决定「每个 task 怎么算」。**
前者是调度图，后者是内核体——与 §5 的 in-kernel runtime（把这张图搬到 device 上事件驱动地跑）分工清晰。

### 3.2 §4.1 tGraph 生成：五步流水线

§4.1 依次做五件事（Figure 5 的 a→f 即对应）：

| 步 | 名称 | 快照行 | 做什么 |
|---|---|---|---|
| 1 | Operator decomposition | 462–468 | 按输出张量切分，把每个算子分解成若干 SM 级 task |
| 2 | Dependency analysis | 469–519 | 枚举 producer/consumer task 对，重叠即插 event |
| 3 | Event fusion | 520–779 | successor-set / predecessor-set 两种融合，消除冗余 event |
| 4 | Normalization | 780–951 | 用空 task 把每 task 的 event 扇入/扇出压到 1 |
| 5 | Linearization | 1101–1125 | BFS 重排，使同一 event 触发的 task 索引连续 |

前两步「生图」，后三步「化简并编码」。下面逐步展开。

#### 3.2.1 步一：Operator decomposition（行 462–468）

- **做法**（行 463–467）：把每个算子的**输出张量**分区，每个 task 计算输出中一个 **disjoint** 子集，
  使同一算子的多个 task 可跨 SM 并行。多数张量代数算子可沿**多个输出维度**切分——例如矩阵乘的输出可同时
  沿行、列两个维度分块以暴露并行度。
- **切分策略**（行 467–468）：性能同时取决于 problem shape 与目标 GPU。MPK 选择**最小化 device memory→
  shared memory 搬运量**的策略（device memory 访问远比 shared memory / CUDA Core / Tensor Core 计算贵）；
  默认生成的 task **数与 SM 数成正比**以促进负载均衡；用户也可通过设定各输出维度的并行度自定义策略。
- **实测规模**（§6.7，行 1638–1642，Table 2 行 1687–1737）：真实模型里平均**每算子 32–47 个 task**；
  Qwen3-8B 的 293 个算子被分解为 **13,867 个 task**。只有 batch=1 时的 pointwise 算子（如 normalization）
  退化为单 task。分解的目的正是「给 runtime 足够独立工作把 SM 喂满」。

#### 3.2.2 步二：Dependency analysis（行 469–519）

- **算法**（行 470–519）：对任意两个**共享同一张量**的算子，**枚举所有 task 对** `(t1,t2)`；当且仅当
  `t1` 产出的输出区域与 `t2` 消费的输入区域**重叠**时，为这一对引入一个 event `e`，并插入两条边
  `(t1,e)`、`(e,t2)`。语义：`t2` 在 `t1` 产出所需数据前不得开始。
- **性质**（行 519）：这种**细粒度依赖分析保留所有 producer-consumer 依赖，同时暴露独立 task 之间的并行**。
- **与 §3 的呼应**：这正是 §2.4（图 4(c) 反例）所说「MPK 通过推断 **precise** 数据依赖来最大化并发、
  最小化同步开销」的具体算法——依赖下沉到「输出 tile 重叠」这一最细可判定粒度。
- **代价**：不加处理时，「每个重叠 task 对一个 event」会爆炸（见下步的量化）。

#### 3.2.3 步三：Event fusion（行 520–779）

为消除冗余同步点，MPK 定义两个集合函数（行 528–551）：`InTasks(e)` = 触发 `e` 的 task 集合，
`OutTasks(e)` = 依赖 `e` 的 task 集合。据此做两种**互补**融合：

**（1）Successor-set fusion（Def 4.1，行 552–659）**——合并「同一批消费者 task 的前驱」。
条件：`OutTasks(e1) = OutTasks(e2)`。做法：删 `e1,e2`，引入融合 event `e'`，使
`InTasks(e') = InTasks(e1) ∪ InTasks(e2)`、`OutTasks(e') = OutTasks(e1)`。
理由（行 554）：这些消费者本来就要等**全部**这些 event 激活，分开表示不带来任何额外调度自由度。
例（行 637–659）：Figure 5(b) 的 `e10`、`e14` 都是 task `O1` 的前驱，融合为 (c) 的 `e4`。

**（2）Predecessor-set fusion（Def 4.2，行 660–779）**——合并「由同一批生产者 task 触发的 event」。
条件：`InTasks(e1) = InTasks(e2)`。做法：删 `e1,e2`，`e'` 满足 `InTasks(e') = InTasks(e1)`、
`OutTasks(e') = OutTasks(e1) ∪ OutTasks(e2)`。
理由（行 662）：这类 event 会**同时**被触发，分开维护只是徒增图复杂度。
例（行 745–779）：Figure 5(c) 的 `e4,e5,e6,e7` 都依赖 task `A1,A2`，合并为新 tGraph 中的单个 `e4`。

- **实测效果**（§6.7，行 1643–1653，Table 2）：不加融合时，依赖分析会为每个重叠 task 对发一个 event；
  融合后，最终 tGraph 只含 **1,142–2,366 个 event**，却编码了 **69,000–162,000** 条 task 对依赖，
  即同步事件**减少 37–118×**。MoE 模型收益最大，因为专家路由产生大量**多对多依赖**，特别适合融合。

#### 3.2.4 步四：tGraph normalization（行 780–951，Figure 6 行 792–802）

**要解决的表示难题**（行 780–785）：runtime 要跨 SM **并行**执行 task / 更新 event，因此需要
**uniform 且廉价**的表示，避免昂贵的间接索引。两个问题：

- task 可能依赖/触发**任意多个** event——按「每 task 预留最大数量」存会严重浪费内存；
- event fusion 后，一个 event 可能触发**任意多个** task——按「每 event 预留最大扇出」存同样昂贵。

**做法**（行 805–934，Figure 6）：做两种图重写，把图变成**功能等价**、但**每个 task 至多一个 dependent
event、至多一个 triggering event** 的形式。这样每个 task descriptor 只需存**一个** dependent-event id 和
**一个** triggering-event id，而非变长列表。

- **降触发扇出（图 6a，行 813–879）**：当 task `T0` 触发 `e1,…,ek` 时，引入新 event `e'` 和 `k` 个**空 task**
  `T1,…,Tk`（不计算、依赖 `e'`）。之后 `T0` 只触发 `e'`，每个 `Ti` 只触发对应的一个原 event `ei`。
- **降依赖扇入（图 6b，行 880–934）**：当 task `T0` 依赖 `e1,…,ek` 时，引入新 event `e'` 和 `k` 个**空 task**
  `Ti`（不计算、**触发** `e'`）。之后 `T0` 只依赖 `e'`，每个 `Ti` 只依赖对应的一个原 event `ei`。
  §2.6 提到的 Figure 5(e) 的 dummy task `D1`/`D2` 就是这个变换的产物。

- **开销**（行 937–951、§6.7 行 1656–1674）：normalization **只在图含 fan-in / fan-out 的 fork/join 处**
  才插辅助 task/event。真实模型通常「deep 而非 wide」（大量顺序算子），实践中开销 **始终 <1%**；
  评测的三个 LLM forward tGraph **几乎完全是顺序的**，没有 fork/join group（Q/K/V projection 等会扇出的
  算子已被 fusion 成单个算子），normalization 基本不改动它们。但它对**正确性**仍是必需的——如 Figure 5 的
  **未融合** Q/K/V 例子，normalization 把每个 task 的 event 扇入/扇出压到 1，保住「固定大小、无间接索引」的编码。

#### 3.2.5 步五：tGraph linearization（行 1101–1125，Algorithm 1 行 952–1098）

- **动机**（行 1102–1112）：normalization 约束的是**「task→event」**的元数据；linearization 约束的是
  **「event→task」**的存储。问题：融合后一个 event 可能触发很多 task（如 Figure 5(e) 的 `e5` 触发四个 task），
  逐条存出边索引同样昂贵。
- **算法**（Algorithm 1，行 952–1098）：基于 **BFS**。从「无 dependent task 的 event」起（行 1013–1015），
  每出队一个 event `e`，就把所有 `dependent_event = e` 的 task 依次入队（行 1033–1061）；当某 task 触发的
  event `e'` 的所有触发 task 都已在 `T` 中时，再把 `e'` 入队（行 1074–1093）。**关键保证**（行 956–969，
  Lines 7–9）：所有依赖同一 event 的 task 在最终 task 顺序 `T` 中**连续**。
- **编码**（行 1113–1125、Figure 5(f)）：于是 event 的扇出只用 **first + last task index** 两个索引即可
  表示（区间），无需显式存消费者列表，且不丢依赖。device memory 布局为：每个 task 只记其 dependent /
  triggering event 的索引；每个 event 存**激活所需的触发计数**，一旦激活，runtime 就 launch 落在
  `[first, last]` 区间内的所有 task。
- **实测效果**（§6.7，行 1677–1686，Table 2）：连续区间编码把 successor footprint 在 dense 模型上压
  **4.4–5.9×**（Qwen3-8B：**110,932 → 18,928 bytes**），在 MoE 模型上压 **15.0×**——MoE 的高扇出专家路由
  event 在区间编码下最省。

### 3.3 §4.2 Task Implementation Generation（行 1126–1146）

前四步产出的是「调度图」，但每个 task 具体怎么算还没定。§4.2（行 1126–1138）为每个 task 生成一个
在 GPU SM 上执行的 device function：

- **复用超优化**（行 1131–1133）：MPK 沿用 Mirage 的 **compiler superoptimization** 自动生成高性能实现。
- **关键差异——thread block 级而非 kernel 级**（行 1134–1138，重点）：MPK 的 superoptimization 粒度是
  **thread block**，而不是 kernel。每个 compute task 关联一份**参考 PyTorch 实现**，MPK 用 **Mirage
  superoptimizer [40]** 搜索一个优化的 **thread-block graph**，交给 **Mirage compiler** 生成 CUDA 实现。
- **生成物内容**（行 1138）：CUDA 实现包含 **intra-SM 优化**——software pipelining、register reuse、
  以及降低 shared-memory bank conflicts 的 layout 优化。

> 与 §2.1 的关系：这里生成的正是 Figure 2 所说的 **intra-task software pipelining**（单个 task 内部流水）。
> **cross-task** 的软件流水不在这里做——那属于 §5 in-kernel runtime 的职责（P2-3）。
> 换句话说：§4.2 把每个「积木」做到极致，§5 负责把这些积木在 device 内交错推进。

### 3.4 小结：编译流水线的四条要点

1. **五步流水线 = 「生图（decomposition + dependency analysis）→ 化简/编码（fusion + normalization +
   linearization）」**。前两步决定图「对不对、并行度够不够」，后三步决定图「省不省、能不能被 runtime
   无间接索引地读」。
2. **precision 靠自动推断，不靠手工**：dependency analysis 用「输出 tile 是否重叠」判定最细依赖，
   fusion 再机械化地消冗余——这正是 §3 图 4(c) 所要求的「precise event」的自动实现（与手工 mega-kernel
   的位掩码依赖表达对照，见 §3 末与 `docs/02`）。
3. **normalization / linearization 是「表示性问题」而非「性能问题」**：它们不改并行度，只把
   task→event 元数据压到一个 id、把 event→task 扇出压成一个区间，换来「固定大小、无间接索引」的
   runtime 友好编码（<1% / 4.4–15× 的数字均属表示开销）。
4. **§4.2 的粒度选择是范式关键**：把超优化下沉到 **thread block**，才使「一个 task = 一个 thread block
   = 一个 SM 工作单元」的映射成立，从而与 §3 的 SM 级表示、§5 的 SM 级调度自洽。

> 与生产案例的呼应（不展开，见 `docs/02`、`../mega-moe/`）：DeepGEMM Mega MoE 用**位掩码 + ring buffer**
> 手工表达 task 级依赖（`scheduler/mega_moe.cuh` 行 140–160），相当于**手工 tGraph**；
> MPK 的贡献是把「分解→依赖→融合→规范化→线性化」这套手工活全部自动化。

> 实测说明：P2-2 为**纯资料分析**（论文 §4 + §6.7 精读），无 H100 实验。§6.7 的 32–47 tasks/op、
> 37–118× fusion、4.4–15× linearization 均为论文报告值（B200 平台，Table 2 行 1687–1737）。
> 与 §4 表示层相关的本机实测佐证（CUDA Graph 只治 launch、不破依赖串行）见
> `docs/04-experiments.md` §0 与 `harness/probe_env.out.txt`、`harness/mitigations.out.txt`。

## 4. P2-3 in-kernel parallel runtime（§5，行 1147–1368）

> 承接：§4（P2-2）已把 tensor program 编译成 tGraph 并为每个 task 生成了 thread-block 级 CUDA 实现；
> 本节精读 MPK 如何在一个**已 launch 的 mega-kernel 内部**把这些 task 真正跑起来。这是 MPK 与
> CUDA Graphs / 静态融合的**范式分水岭**——调度不再发生 host 侧，而是常驻 device 内。

### 4.1 §5 总览：SM 划分为 worker 与 scheduler（行 1148–1174）

- **单次 launch、常驻**（行 1149–1154）：in-kernel parallel runtime 在**一个 mega-kernel 内**跨所有 SM
  执行整张 tGraph；一旦 launch，该 kernel **持续管理计算与通信直到推理负载结束**。这直接消掉了
  重复的 kernel launch（对应 §2 损失二）并暴露了对 scheduling / synchronization / execution order 的细粒度控制。
- **两类角色**（行 1155–1164）：
  - **worker**：每个 worker 占用**一个物理 SM**，维护一个独立的 **task queue**，跑一个轻量循环
    （dequeue task → 执行计算/通信 → 通知该 task 的 triggering event）。worker 之间异步推进，
    使 SM 保持满载。
  - **scheduler**：以 **warp 粒度**组织，**每个 SM 承载 4 个 scheduler warp**；每个 scheduler 维护一个
    **event queue**，反复轮询新激活的 event 并把对应 task 下发给 worker。
- **角色固定在 launch 时确定**（行 1164）：调度/工作划分**在 kernel-launch 时固定并与 GPU 物理 SM 数匹配**，
  从而**避免 kernel 内的动态角色切换开销**。
- **本节组织**（行 1165–1174）：§5.1 事件驱动执行模型 → §5.2 两种 task launch（JIT/AOT）及其取舍 →
  §5.3 进一步降开销的运行时优化。

> **范式要点**：这里就是 `docs/00` §3 判据 **C4「设备内协调」+ C5「SM 级调度」** 的机制落地——
> SMs 被显式切成「少数 scheduler + 多数 worker」，调度状态（event/task queue、semaphore）全在
> device memory，host 在 launch 后不再参与。

### 4.2 §5.1 事件驱动执行（行 1175–1231）

- **起点**（行 1177–1198）：每张 tGraph 有一个**无前驱的 start event**（如 Figure 7 的 e0），最先入 scheduler
  的 event queue；scheduler（如 s1）dequeue 后 launch 依赖它的所有 task（如 AT1…AT4）。
- **推进闭环**（行 1211–1217）：每个被 launch 的 task 下发到某 worker；worker 执行完**通知该 task 的
  triggering event**；当某 event 的**所有前驱都完成并累计触发到所需次数**时，event 变为 **activated**，
  被入队到某 scheduler 的 event queue——于是执行沿图继续传播。**event 就是驱动 task 执行的机制**，
  实现细粒度、异步调度。
- **无需拓扑画像的自适应**（行 1218–1225）：计算与通信**统一表示为同一张 tGraph 里的 task**、由同一
  scheduler 分发。依赖满足即可跑：**快的链路更早触发 event、更早释放下游；慢链路只拖它自己的下游**。
  因此 scheduler **无需显式知道链路带宽/拓扑距离**，同一张 tGraph+scheduler 就能处理**异构互连**
  （intra-node NVLink 与 inter-node 网络），靠「数据到达」而非「静态拓扑模型」反应。

> 与 §3（P2-1）呼应：这正是 tGraph「以 SM 为粒度、MatMul→AllReduce 细粒度重叠」（快照行 371–386）
> 在 runtime 侧的兑现方式；也解释了 §6.6 compute–communication overlap 消融为何能到 **1.1×**
> （见下文 §4.5）。

### 4.3 §5.2 混合任务下发：JIT 与 AOT（行 1232–1308）

- **两种下发语义**（行 1234–1238）：
  - **JIT（just-in-time）**：scheduler 在依赖 event **完全激活后**才把 task 派给 worker，task 可立即执行。
  - **AOT（ahead-of-time）**：runtime 在**前驱 event 激活之前**就把 task **预入队**到某 worker 的 task queue；
    worker 在本地**等到依赖满足**才执行。
- **为何互补**（行 1239–1247）：
  - JIT 能**适应负载不均**：LLM attention 因序列长度而耗时高度可变，静态分配无效；JIT 在 attention 完成后
    才 launch 下游（MatMul/AllReduce），**先完成的 worker 多干活**，改善端到端时延并均衡 SM 负载（Figure 7）。
  - 但 JIT 有**更高的下发时延**，因为多了一次 worker↔scheduler 往返。
- **同步步数对比**（行 1247–1295，Figure 8 重点）：沿 `T1 → e → T2` 触发链，
  - JIT（图 8a）：worker(T1) → scheduler 通知 → scheduler dequeue e、launch T2、入队 worker → worker dequeue T2，
    共**两步同步**（worker→scheduler、scheduler→worker）。
  - AOT（图 8b）：T2 已**预入队到预分配的 worker**，该 worker 只需**等 event e 激活**，
    经 event trigger **一步同步**即可，**降低 per-task launch 时延**。
- **混合策略（编译期分类）**（行 1296–1302）：
  - 编译期按「执行时间是否数据相关、易致运行时不均」给**整个 operator** 打 JIT/AOT 标签（**operator 粒度**，
    同一 operator 生成的所有 task 共享同一 mode）。
  - **数据相关耗时的 operator（如 attention）标 JIT**，其下游**保持 JIT 直到遇到 global barrier**
    （一个必须由**所有上游 task**触发的 event）；barrier 抹平累积不均，其后的 operator 适合 **AOT**。
  - worker 维护 **JIT / AOT 两条 queue**，**优先 JIT**（就绪即可执行）；JIT 空了再检查 AOT 队首依赖是否已激活，
    如此 worker **只在确实无就绪工作时才阻塞**。
- **下发侧优化**（行 1303–1308）：scheduler **只处理会 launch JIT task 的 event**；AOT task **全部预入队**。
  由于 operator 产生的 task 数通常与 worker 数成正比（§4），AOT task 按 **round-robin** 摊到各 worker 以均衡负载。
  **预入队 AOT 降低了 scheduler 负载并摊薄下发开销**，同时 JIT 动态均衡 SM 工作。

> **设计张力一句话**：JIT = **保负载均衡但贵**；AOT = **省下发但静态**。MPK 用「编译期标签 + global barrier
> 分界 + worker 双队列优先级」把两者缝在一起，用静态图拿到动态负载均衡——这是纯 CUDA Graph
> （`docs/00` §2.2）做不到的。

### 4.4 §5.3 运行时优化（行 1309–1368）

三个优化都服务于同一目标：**让比传统 kernel 细得多的 task 的下发/执行开销摊薄**。

#### 4.4.1 Paged shared-memory abstraction（行 1312–1325）

- **问题**（行 1313–1318）：常规模型下 shared memory 是 thread block 私有、**kernel 生命周期内**存在；
  kernel-per-operator 下每个 kernel **独占**其 shared memory。该假设**阻碍 cross-task software pipelining**——
  因为「为下一 task 预取数据」与「当前 task 计算」可能**同时需要 shared memory**。
- **机制**（行 1319–1325）：把 shared memory 切成**固定大小页**，task 实现改为**按页**操作；
  task **acquire** 若干页（按其 footprint）、不再需要时**release**。**一旦 release 过任何页，便不再允许
  acquire 新页**（单调使用模式，简化调度）。当前 task 释放页后，MPK 即可把可用页分给**下一个 task 立即预取数据**。
- 配置锚点（§6.1，行 1460）：page size **32 KB**，在 A100/H100/B200 上分别得 **5 / 7 / 7 页/SM**。

#### 4.4.2 Cross-task software pipelining（行 1326–1363）

- **拆分**（行 1330–1334）：把每个 task 拆成 **pre-loading 阶段**（发数据搬运指令，把 tensor 的一个 chunk
  从 device memory 取到 shared memory，**不做计算**，用来初始化 intra-task 软件流水）与 **compute 阶段**。
- **重叠条件**（行 1335–1363）：当**两条件同时成立**时，MPK 机会式地把当前 task **T1 的 compute** 与
  后继 task **T2 的 pre-loading** 重叠：(1) T1 **已发出自己全部数据搬运指令**；(2) 有**足够页**给 T2 预取。
  该流水**不干扰 T1**：MPK 插入必要的 **intra-SM 同步屏障**，确保 T2 的搬运不与 T1 正在进行的搬运冲突。
- 消融（§6.6，行 1586–1596，Figure 12）：Qwen3-8B final linear、**B200** 上，cross-task pipelining 把
  task 运行时降 **1.2–1.3×**，甚至**胜过 cuBLAS based compiled kernels**。

#### 4.4.3 Pre-fetching task descriptions（行 1364–1368）

- 每个 worker 在 device memory 维护 JIT/AOT 两条 task queue；每个 task 关联一份 **task description**
  （输入/输出 tensor 与配置参数），**当前实现每份占 352 bytes**（§6.1）。
- 为降低 enqueue/dequeue 时延并隐藏 device-memory 访问，MPK 用**轻量预取**把**即将用到的 task description
  先取到 shared memory**。

### 4.5 实现细节、配置与消融锚点

- **实现规模与组件**（§6.1，行 1396–1407）：MPK 约 **44K 行 C++、42K 行 CUDA、10K 行 Python**；
  in-kernel runtime 用 CUDA 编写，用 **device memory 中的 semaphore** 协调 worker 与 scheduler；
  compute task 的 CUDA 实现由 **Mirage superoptimizer** 生成，跨 GPU 通信用 **NVSHMEM**。
- **降 task-launch 开销**（行 1409–1413）：event/task queue 是 device memory 中的**环形缓冲**，
  enqueue/dequeue **只用低成本的 `atomicAdd`**；采用**去中心化调度**（每个 scheduler 用本地状态分配 task），
  避免全局协调的通信/同步开销。论文指出 runtime 也**支持改写为全局协调调度**（未来工作）。
- **动态负载支持**（行 1414–1436）：扩展了 **continuous batching** 与 **paged attention**；
  start event 的调度逻辑（移除已完成请求、接纳新请求、更新 per-request KV-cache 元数据）作为**单个 task** 执行，
  KV-cache 元数据放 device memory 供 attention task 直接访问；并按 batch size 生成**多张 tGraph**
  （2 的幂直到最大 batch），runtime 按当前 batch size 选图。
- **配置（Table 1，行 1437–1460）**：scheduler 预留 **4 个 SM / 16 个 scheduler warp**（每 SM 最多 4 个活跃
  scheduler warp），其余为 worker——A100 **108 SM（104 worker / 16 scheduler）**、
  H100 **132 SM（128 / 16）**、B200 **148 SM（144 / 16）**。这组数字给出「几十个 worker、个位数百分比
  scheduler 开销」的直观量级。

**§6.6 运行时三消融（行 1583–1628）** —— 与本节机制一一对应：

| 优化 | 机制所在 | 消融结果 | 行 |
|---|---|---|---|
| cross-task pipelining | §5.3（本文 §4.4.2） | Qwen3-8B final linear，B200：**1.2–1.3×**，胜过 cuBLAS | 1586–1596 |
| compute–communication overlap | §5.1 事件驱动（本文 §4.2） | Qwen3-1.7B，4×H100 TP：**1.1×** | 1597–1615 |
| kernel-launch reduction | §5 单次 launch（本文 §4.1） | Qwen3-8B 每 token **293** 次 launch；B200 eager **3.8µs/次→1.1ms/token**、Graph **0.8µs→0.2ms**；MPK in-kernel scheduler 仅占 **0.28%** 总运行时 | 1618–1628 |

### 4.6 小结：in-kernel runtime 的四条要点

1. **调度搬到 device 内**：SMs 固定切成少数 scheduler（4 SM/16 warp）+ 多数 worker（每 SM 一 worker），
   状态（event/task queue、semaphore、task description）全在 device memory——这是「单次 launch 跑完整个模型」
   的必要条件，也是与 CUDA Graphs（host 侧预录）最本质的差别。
2. **event 驱动 + 统一表示**：计算与通信同为 task，靠「数据到达触发 event」推进，天然适配异构互连，
   无需拓扑画像；细粒度 event 正是 §3 tGraph 的表达在 runtime 的兑现。
3. **JIT/AOT 混合是「动态负载均衡 vs 下发开销」的折中**：编译期按 operator 打标签、以 global barrier 分界，
   worker 双队列优先 JIT——用静态图拿到近似动态调度的效果。
4. **§5.3 三优化都在服务「细粒度 task 开销摊薄」**：paged shared memory 解锁跨 task 流水、cross-task
   software pipelining 偷出 1.2–1.3×、task descriptor 预取隐藏设备内存访问；叠加后 in-kernel scheduler
   只占 0.28% 运行时——**证明「把调度做进 kernel」的额外开销是可忽略的**。

> 实测说明：P2-3 为**纯资料分析**（论文 §5 + §6.1/§6.6 精读），无 H100 实验；
> §6.6 的 1.2–1.3× / 1.1× / 0.28% / 293 launch 均为论文报告值（B200、H100、Qwen3）。
> 与本机相关的佐证见 `docs/04-experiments.md` §0：persistent kernel（cooperative grid sync）可跑、
> CUDA Graph 1.56×、PDL ok（`harness/probe_env.out.txt`），以及「多流只并行无依赖、Graph 不破依赖串行」
> （`harness/mitigations.out.txt`）；把「N 小 kernel 串行 vs CUDA Graph vs 单 persistent kernel」的
> **端到端 launch 开销对比实验**留给 P5。

## 5. P2-4 评估（§6，行 1369–1737）

> 承接：§4（P2-3）已拆开 in-kernel runtime 的三个机制，本节看这些机制在**真实 LLM serving** 上的
> 端到端收益，并划清「论文证明了什么、没证明什么」。§6 的运行时消融（§6.6）与编译期消融（§6.7）
> 已分别在 P2-3、P2-2 引用，本节只做**汇总回顾**，不重复展开；重点放在端到端（§6.3）、
> MoE 动态负载（§6.4）、多 GPU（§6.5）三块，以及 §7 Discussion（行 1739–1764）给出的**边界**。

### 5.1 §6 组织：四个问题（行 1371–1390）

原文（行 1371–1383）明确 §6 要回答**四个问题**，对应四个小节：

| 问题 | 小节 | 快照行 | 一句话 |
|---|---|---|---|
| mega-kernel vs kernel-per-operator | §6.3 端到端 | 1467–1517 | 相对 vLLM/SGLang 的延迟/吞吐 |
| 支持动态负载 | §6.4 MoE 案例 | 1520–1553 | 混合负载均衡 + fused gather-GEMM |
| 多 GPU 可扩展性 | §6.5 多 GPU | 1556–1580 | TP 下 8×H100 |
| 运行时优化贡献 | §6.6 消融 | 1583–1628 | pipelining / overlap / launch |

论文把评测**聚焦在 LLM serving**（行 1384–1390），两个理由：一是 serving 有 SGLang/vLLM
这类高度优化的 kernel-per-operator 基线，对照「严格」；二是 serving 天然动态（batch size、序列长度、
prefill/decode 混合逐轮变化），能同时压测编译器与运行时。论文强调二者与模型无关
（「the MPK compiler and runtime are model-agnostic」，行 1390），但**未给出 serving 之外的评测**。

### 5.2 §6.1–§6.2 实现与实验设定（复现要点，行 1391–1464）

- **接入方式**（§6.1，行 1393–1395）：MPK 作为 **PyTorch compiler backend**，用法即
  `torch.compile(backend=MPK)`；生成 mega-kernel 并返回一个可调用的 PyTorch function，
  **每次调用只发一次 launch**。这是「最小改动 mega-kernel 化」（行 1514 呼应）。
- **实现规模**（行 1396–1407，P2-3 已引）：约 44K C++ / 42K CUDA / 10K Python；
  in-kernel runtime 用 CUDA + device-memory semaphore；compute task 由 **Mirage superoptimizer**
  生成；跨 GPU 通信走 **NVSHMEM**。
- **评测模型与硬件**（§6.2，行 1457–1464）：**5 个 LLM**（含 dense 与 MoE，跨多个规模），
  **三代 NVIDIA GPU：A100 / H100 / B200**；配置见表 1（行 1437–1460，P2-3 已引：
  A100 108 SM、H100 132 SM、B200 148 SM，均预留 4 SM / 16 scheduler warp）；
  shared-memory page size 统一 **32 KB**（A100/H100/B200 → 5/7/7 页/SM）。
- **负载设定**（行 1464）：**offline batched-inference**，最大 batch size 在 **1–16** 间扫；
  每条请求 **prompt 64 token、生成 1024 token**。论文用 offline 设定「消除请求并发不足引起的
  服务端停顿」，以获得可控对比——这同时意味着**未覆盖在线到达分布（arrival pattern）**。

### 5.3 §6.3 端到端：单 batch 延迟 1.0–1.7×（行 1467–1517）

- **基线**（行 1469–1497）：SGLang 与 vLLM，二者都是 kernel-per-operator，依赖 FlashInfer /
  FlashAttention（attention）、cuBLAS/cuTLASS（GEMM）、CUDA/Triton（其余）；均 bf16、启用
  paged attention 与 continuous batching。**关键架构差异**：MPK 把 **page allocation 与 request
  scheduling 直接做进 mega-kernel**，而两个基线在 **CPU** 上做，额外承担 host–device 同步与下发开销。
- **主结果**（行 1499–1504）：单 batch 推理下，MPK 跨模型/硬件提升 **1.0–1.7×**；
  **越小的模型、越新的 GPU 代际收益越明显**——因为每 token 计算量越小、GPU 越快，三类开销越突出：
  (1) kernel-switch 开销（即便用了 CUDA Graphs）；(2) kernel 边界造成 pipeline bubble、阻断
  cross-task pipelining（Figure 2）；(3) 基线在 CPU 上做 page allocation + request scheduling。
- **绝对锚点**（行 1513）：Qwen3-8B 在 **A100** 上，MPK 把 per-token decode 从
  vLLM/SGLang 的 **14.5 ms 降到 12.5 ms**，逼近论文估算的**硬件下界 ~10 ms**
  （按 16 GB 参数 @ 1.6 TB/s 带宽）。这个「离下界还有 25%」的量级很关键：说明端到端
  已非常接近纯内存搬运极限，剩余差距即计算/通信/调度。
- **可编程性**（行 1514–1517）：论文额外主张 MPK 的**开发效率**优势——基线需大量工程为新模型
  接专用 kernel，MPK 只需几行 PyTorch 改动即可 mega-kernel 化；相对**原生 PyTorch** 超过 **10×**。
  （注：10× 是 vs 原生 PyTorch，不是 vs vLLM/SGLang；后者是 1.0–1.7×。）

### 5.4 §6.4 MoE 案例：动态负载两项优化（行 1520–1553）

- **MoE 用同一套 task 抽象表达**（行 1523–1536）：一个 MoE block 被 lower 成
  routing → dispatch → expert compute → combine 的 **SM 级 task 链**；dispatch/combine 的
  all-to-all 通信**与 expert 计算放进同一张 tGraph、由同一事件驱动 scheduler 分发**——
  于是通信-计算在事件触发时**自动重叠，不依赖 CUDA stream 或 host 侧同步**。这是把
  `docs/00` §2.5「通信-计算 overlap 的双 kernel 方案」（DeepEP/FlashMLA）对照到 MPK 范式的位置。
- **Hybrid workload balancer**（行 1537–1550）：每专家 token 数只有运行时才知道，静态划分
  在倾斜路由下会严重不均，全动态 Grouped-GEMM 又有细粒度同步开销。MPK 取**混合策略**：
  编译期按专家切 task，运行时每一 task 拿到 `topk-softmax` 产出的 **meta-tensor**
  （含激活专家数、各专家 token 数），据此**动态微调**负载分配。Figure 10（行 1518–1519）
  报告：该策略**在所有 batch size 上胜过纯静态划分**。
- **Fused gather-GEMM**（行 1551–1553）：Hopper/Blackwell 上用 TMA 需先把路由到同一专家的
  token **gather 成连续布局**；Qwen3-30B-A3B、batch=1 下该预处理在 SGLang 上**占总 MoE 时间
  高达 11%**。MPK 把 TMA gather 换成**直接融进 GEMM task 数据加载阶段的异步 token 级拷贝**，
  消掉独立 gather kernel 与额外调度点。这项与 DeepGEMM Mega MoE 的「dispatch+GEMM 融合」
  （见 `docs/02` §1、`../mega-moe/`）是**同一动机的两种实现**：消除 gather 这类「只为对齐布局」
  的搬运。

### 5.5 §6.5 多 GPU：TP 下最高 10×（行 1556–1580）

- **设定**（行 1558–1566）：NVIDIA **H100 DGX**，采用 **tensor model parallelism**（Megatron-LM），
  用户只需在 attention 与 gated MLP 后插入 `AllReduce`；MPK 自动把 collective 编成两类 task：
  **inter-GPU 数据传输 task**（用 NVSHMEM `nvshmem_signal_wait_until` 实现）+ **local reduction task**，
  从而把同步 collective 变成可并入事件驱动 runtime 的**异步 task**。
- **结果**（Figure 11 行 1554–1555，行 1567–1577）：相对 **PyTorch**（手写 kernel + CUDA Graphs
  + `torch.compile` 组合）最高 **10×**；相对 **SGLang/vLLM**，扩展到 **8×H100** 时 **1.1–1.4×**。
  论文归因于基线缺失的三点：(1) page allocation + request scheduling 进 mega-kernel，消 CPU 下发；
  (2) 异步模型让 compute 与 collective **重叠**；(3) 消除 kernel barrier、启用 cross-task pipelining。
- **一个值得注意的边界**：多 GPU 主结果（Figure 11）用的是 **Qwen3-1.7B（dense）+ TP**；
  MoE 的 **expert parallelism** 虽有机制描述（§6.4），但**未作为多 GPU 端到端图表呈现**。
  也就是说「EP 的 all-to-all 重叠」是机制论证，端到端数字主要在 TP。

### 5.6 §6.6 / §6.7 消融回顾（行 1583–1737）

两块消融已分别在 P2-3、P2-2 展开，此处只做**表格化归档**，便于与生产案例对照：

**§6.6 运行时三消融**（行 1583–1628）：

| 优化 | 平台/模型 | 结果 | 行 |
|---|---|---|---|
| cross-task pipelining | B200 / Qwen3-8B final linear | **1.2–1.3×**，胜过 cuBLAS compiled kernel | 1586–1596 |
| compute–communication overlap | 4×H100 TP / Qwen3-1.7B | **1.1×** | 1597–1615 |
| kernel-launch reduction | B200 / Qwen3-8B | 每 token **293** launch；eager **3.8µs→1.1ms**、Graph **0.8µs→0.2ms**；in-kernel scheduler 仅 **0.28%** | 1618–1628 |

**§6.7 编译期三消融**（Table 2，行 1629–1737）：Qwen3-1.7B / 8B / 30B-A3B 三模型上：

| 阶段 | 指标 | Qwen3-1.7B | Qwen3-8B | Qwen3-30B-A3B | 行 |
|---|---|---|---|---|---|
| operator decomposition | tasks/op | 35.6 | 47.3 | 32.2 | 1638–1642 |
| （ops / tasks 总量） | ops | 229 | 293 | 533 | Table 2 |
| event fusion | 事件数 / 缩减 | 1,870 / 37× | 2,366 / 68× | 1,142 / **118×** | 1643–1653 |
| normalization | 变更 | 三模型 forward tGraph **近乎全顺序**，无 fork/join；开销 <1%，仅为正确性保留 | | | 1656–1674 |
| linearization | footprint 缩减 | 4.4× | **5.9×**（110,932→18,928 B） | **15.0×** | 1677–1686 |

一句话：**计算越多/MoE 越重，编译期收益越大**（MoE 的 118× fusion、15.0× linearization 都是峰值）；
而 normalization 在真实 LLM forward 上**几乎不动图**——印证 `docs/01` §3.4 所说它是「表示性问题而非性能问题」。

### 5.7 §7 Discussion：论文自己划的边界（行 1739–1764）

§7 虽不是评测，但直接给出「megakernel 常见顾虑」的论文回答，是复现/落地时最该读的两段：

- **Resource footprint（行 1740–1748）**：mega-kernel 的常见担忧是「单 kernel 必须在每个 SM 上
  预留所有被融合算子的资源之和」。MPK 的回答：task 在 SM 上是**时分复用而非静态划分**，
  每个 SM 只用**当前 task** 所需资源（§5）；shared memory 走 paged 抽象（§5.3）按 task 粒度
  acquire/release。**唯一例外是 per-thread register**——它固定在所有 task 类型的最大值。
  → 落地时**寄存器压力**是 mega-kernel 融合范围的硬约束，shared memory 则被 paged 抽象化解。
- **集成手写 kernel 与移植新硬件（行 1749–1764）**：每个 task 是统一调用约定的 CUDA device function，
  因此 (1) 手写 kernel 可**包成一个 task** 直接进事件驱动调度，运行时与 tGraph 不用改——
  MPK 能「手写 + 自动」混用；(2) 移植新 GPU **只需改 per-task code generator**，
  tGraph 表示、图级变换、运行时、scheduler 都不变——**per-task codegen 是唯一硬件相关组件**。
  这与 `docs/02`/`../mega-moe/` 里 DeepGEMM 的**全手工** mega-kernel 形成对照：MPK 把
  「硬件相关」收缩到最底层 codegen，其余抽象跨代共享。

### 5.8 复现要点与局限（小结）

**复现要点**（能照着做的）：
1. 接入面：`torch.compile(backend=MPK)`；模型架构从 HuggingFace 载入，bf16。
2. 设定：offline batched-inference，batch 1–16，prompt 64 / 生成 1024；page size 32 KB；
   预留 4 SM / 16 scheduler warp；TP 时在 attention 与 gated MLP 后插 `AllReduce`。
3. 基线：SGLang / vLLM（均 paged attention + continuous batching）+ 原生 PyTorch；
   多 GPU 时 PyTorch 基线是「手写 kernel + CUDA Graphs + `torch.compile`」组合。
4. 报告口径：MPK 的 **1.0–1.7×**（单 batch）与 **1.1–1.4×@8GPU** 是 **vs SGLang/vLLM**；
   **10×** 是 **vs PyTorch**，两者不可混用。

**局限 / 未覆盖**（论文范围之外或明确未做）：
- **只在 LLM serving 上评测**（行 1390、1457–1464）；论文自称 model-agnostic，但无训练/其它 DNN 数据。
- **offline batched**（行 1464）——未覆盖真实在线服务的 **arrival pattern** 与尾延迟。
- **batch 上界仅 16、prompt 64、生成 1024**——长上下文 / 大 batch / 变长生成未评估。
- **多 GPU 端到端只在 TP + dense（Qwen3-1.7B）**（行 1554、1568）；**EP 的 all-to-all 重叠是机制论证**，
  无端到端图表。
- **动态仍依赖「每 batch size 一张静态 tGraph」**（§6.1 行 1427–1436：按 2 的幂生成多张图，
  运行时按当前 batch size 选图）——是**有限离散的动态**，不是任意变长形状。
- **decoupled 调度的取舍未探**：当前默认**去中心化调度**，论文自己把「全局协调调度及其 trade-off」
  列为**未来工作**（行 1413）。
- **寄存器固定为各 task 最大值**（行 1748）——融合的算子越多，寄存器上限越可能成为瓶颈。
- **page size 固定 32 KB**（行 1460），论文未给 page size 的自适应/调优分析。

> 实测说明：P2-4 为**纯资料分析**（论文 §6 + §7 精读），无 H100 实验；
> §6.3/§6.4/§6.5 的 1.0–1.7×、14.5→12.5 ms、up to 10×、1.1–1.4× 与 §6.6/§6.7 消融
> 均为论文报告值。与本机相关的**可跑佐证**（launch 开销、CUDA Graph 1.56×、
> 「多流只并行无依赖、Graph 不破串行」）见 `docs/04-experiments.md` §0 与
> `harness/probe_env.out.txt`、`harness/mitigations.out.txt`；把「N 小 kernel 串行 vs CUDA Graph vs
> 单 persistent kernel」的**端到端 launch 对比**留给 P5 本机复现。
