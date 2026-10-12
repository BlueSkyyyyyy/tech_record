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
> 进度：**P2-1 已完成**（§3，下文 §2）；**P2-2 已完成**（§4，下文 §3）；P2-3 in-kernel runtime（§5）、
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

## 4. P2-3 in-kernel runtime（§5）

（待续）

## 5. P2-4 评估（§6）

（待续）
