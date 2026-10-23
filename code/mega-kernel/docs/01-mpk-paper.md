# MPK 论文精读：SM 级图表示、编译器与 in-kernel runtime

> MPK 的贡献可以压成一句话：**把「算子级依赖图」换成「SM 级 task/event 图」，再用一个常驻
> kernel 在设备内事件驱动地跑它。** 这篇按四个台阶读：问题 → 表示 → 编译 → 运行，最后用
> 评估与局限收口。
>
> 承接 `docs/00-megakernel-overview.md`：那里讲动机、定义与谱系；这里只精读论文本身。
> 生产案例（DeepGEMM mega 家族、Kimi K3）见 `docs/02-production-cases.md`，DeepGEMM MoE 的
> 实现细节见 `../mega-moe/` 专题——本文遇到时只给坐标、不展开。
>
> 快照基准：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
> （arXiv:2512.22219v2，共 2717 行），引用的章节号与行号都在该快照里逐行核对过；
> 线上实现见 `https://github.com/mirage-project/mirage`。

## 0. 先记住一个画面

把一次模型推理想成一家**只有一个中央厨房**的餐厅。

- **kernel-per-operator**：每道菜单独开一间厨房。菜做完，整间厨房清场、关灯，半成品端进仓库，
  下一间厨房重新点火。
- **MegaKernel**：所有菜都在中央厨房做。墙上有块**任务看板**，每道菜被拆成若干「工位任务」
  （task），每个任务在一口灶（SM）上完成；一张**齐套卡**（event）负责放行——指定的那批
  配菜到齐了，才准下锅。

对应到论文里的角色：

| 餐厅 | MPK |
|---|---|
| 菜谱 | tensor program 计算图 |
| 工位任务 | task（一个 SM 上的工作单元） |
| 齐套卡 | event（一组 task 的同步点） |
| 看板管理员 | scheduler（每 SM 4 个 warp） |
| 厨师 | worker（每 SM 一个） |
| 工艺员 | 编译器：拆任务、写齐套卡、排顺序 |

> **类比的边界**：后厨里人可以走动、看板可以随时改；MPK 不行。task 到 SM 的映射在 launch 时
> 固定（行 1164），event 是显式计数的屏障，计算图一变就要重新编译（静态 tGraph）。任务之间也
> 不能随便共用寄存器——每个线程的寄存器按所有 task 类型的最大值预留（行 1748）。最关键的
> 失效点：现实看板允许「差不多齐了」，event 只认精确的触发计数。

读完这一篇，你应该能回答：

1. tGraph 里的 task 和 event 各是什么？为什么这个粒度能拿到 kernel 级模型拿不到的并行？
2. 编译器五步分别在消什么麻烦？event fusion 的 37–118× 是从哪来的？
3. 运行时怎么在一个 kernel 里推进整张图？JIT/AOT 各付什么代价？论文证明了什么、没证明什么？

## 1. 地图：论文五节 ↔ 本文

论文的主干按五块读，正好对上本文的五个台阶：

| 论文 | 快照行 | 回答什么 | 本文 |
|---|---|---|---|
| §1–§2 Introduction / Background | 52–344 | kernel-per-operator 亏在哪、已有手段停在哪 | §2 |
| §3 SM-Level Graph Representation | 345–448 | tGraph：task/event 交替的 SM 级依赖图 | §3 |
| §4 The MPK Compiler | 449–1146 | 怎么自动生成 tGraph、怎么生成每个 task | §4 |
| §5 In-Kernel Parallel Runtime | 1147–1368 | 怎么在一个常驻 kernel 里推进整张图 | §5 |
| §6–§7 Evaluation / Discussion | 1369–1764 | 端到端赢多少、边界在哪 | §6 |

论文的主张只有两条：**换表示粒度**（§3）+ **换调度位置**（§5）。

中间的编译器（§4）负责把「换粒度」这件事自动化——否则就得回到手工 mega-kernel 的路子，
像 DeepGEMM 那样自己写 ring buffer、位掩码和 mbarrier。评估（§6）回答「值多少钱」，
讨论（§7）回答「边界在哪」。

> 一句话记住分工：**§3 定义语言，§4 生产程序，§5 执行程序，§6 给账单。**

## 2. 论文要解决什么（§1–§2，行 52–344）

**这一步解决什么问题**：不是 kernel 写得不够快，而是 kernel 之间那堵墙太贵。

今天的系统把模型表示成一张**张量程序 DAG**：节点是算子，边是张量（行 53–56）。
执行时一个算子一个 kernel，按依赖顺序依次 launch（行 57–73）。
同一条 stream 上相邻 kernel 之间，GPU 会隐式插一道 **kernel barrier**：前一颗 kernel 的
**所有** thread block 跑完，后一颗才能开始（行 74–77、223–226）。

这道屏障保证了正确性，但代价是三条：

1. **跨算子软件流水被截断**。现代 GPU 上 Tensor Core、TMA、CUDA Core 是异构的，要让它们都不闲，
   就得在多个任务迭代之间交错；kernel barrier 只允许在单个 task 内部流水（行 227–240，
   Figure 2 行 204–209）。
2. **细粒度重叠不可表达**。依赖只在「整个算子」这个粒度上写，运行时只能等上游算子全部完成。
   论文的例子是 `MatMul` → `AllGather`：每个 AllGather block 其实只依赖一个 MatMul block，
   但 barrier 要求它等全部（行 241–293，Figure 3 行 253）。
3. **launch 与中间张量**。每 token 几百到上千次 launch（行 87–89）；相邻算子靠 HBM 上的中间张量
   传数据，上游写完下游再读（行 311–313）。

这堵墙值多少钱？论文给了几组可以直接引用的锚点：

| 项 | 数值 | 出处 |
|---|---|---|
| Qwen3-8B 每 token 的 kernel launch 次数 | **293 次** | §6.6 行 1618–1628 |
| 单次 launch（eager / CUDA Graph） | **3.8 µs / 0.8 µs** | 同上 |
| 合计（eager / CUDA Graph） | **1.1 ms / 0.2 ms** per token | 同上 |
| 跨任务软件流水 | **1.2–1.3×** | §6.6 行 1586–1596 |
| 计算-通信细粒度重叠 | **1.1×** | §6.6 行 1597–1615 |
| Qwen3-8B / A100 端到端 | **14.5 ms → 12.5 ms**，硬件下界约 **10 ms** | §6.3 行 1513 |

已有手段不是没管，而是各自停在半路：

| 手段 | 治哪笔账 | 停在哪 |
|---|---|---|
| 算子融合 | 中间张量往返 + 局部 barrier | 只能融**局部小算子组**（行 314–328） |
| CUDA Graphs | launch 开销 | 静态；仍有 kernel-switch 开销（行 87–89、1504–1509） |
| PDL | 相邻 kernel 部分重叠 | 要改 kernel 结构与控制流（行 78–84） |

根因是同一个：**依赖表达在算子粒度，调度权在 host 手里。**

MPK 的答案不是「写一个更大的 kernel」，而是两件事一起做：把依赖下沉到 SM 级 task（§3），
把调度搬进 device（§5）。编译器（§4）把前者自动化。

> **不这么做会怎样**：每 token 293 次 launch、eager 下合计 1.1 ms 的固定成本照付；MatMul 与
> AllReduce 只能轮流用 Tensor Core 和网络带宽，硬件总有一半在等。模型越小、GPU 越快，这笔钱
> 占比越高（行 1501–1513）。

## 3. tGraph 怎么表示（§3，行 345–448）

**这一步解决什么问题**：把依赖从「算子之间」下沉到「SM 上的 task 之间」。

§3 全节只有约 100 行（行 345–448），却是整篇论文的地基。它回答一个问题：
**要把整张算子图塞进一个持久化 kernel，需要一种什么粒度的依赖表示？**

原文的说法是：tGraph「expresses the computation of a tensor program at the granularity of
individual SMs」，而传统计算图「expose dependencies only between tensor operators」
（行 347–354）。这种细粒度表示暴露了额外并行，才使**跨算子软件流水**和**细粒度 kernel 重叠**
成为可能。

### 3.1 两类节点，四条规则

Figure 4 的图题写得很清楚：它把 kernel 级计算图变成 SM 级 tGraph（行 294–298）。节点只有两类：

- **task**（矩形）：一个**在单个 SM 上执行**的工作单元。蓝色是计算，橙色是通信（行 364）。
- **event**（绿色圆）：**跨 task 的同步**（行 364）。

结构约束有四条（行 360–370，逐条核对；前两条见行 364–367）：

1. task 与 event **交替**出现；
2. 每个 task 只有到 **triggering events** 的出边，和来自 **dependent events** 的入边；
3. 一个 task 在其 dependent events **全部 activated** 后 ready，完成时 **notify** 它的
   triggering event（行 368–370）；
4. 一个 event 收到**与它关联的所有 task** 的 notification 后 **activated**。

一句话概括：**task 是执行体，event 是「一组 task 全完成」的屏障**——但屏障的范围是**任意
task 子集**，不是整个 kernel。这正是 kernel barrier（§2.1 行 259）的推广：把「全 kernel 的
粗屏障」换成「任意子集之间的细屏障」。

### 3.2 例子：MatMul → AllReduce 为什么能重叠

论文用多 GPU serving 里最常见的 `MatMul` → `AllReduce` 来说明（行 371–386）。
这个例子值得慢慢走一遍，因为它同时解释了「为什么需要 tGraph」和「收益从哪来」。
（§2.1 用的是 `MatMul` → `AllGather`，行 260–293，机制相同。）

**现状**（行 378–379）：两个 kernel 顺序执行，因为「coarse-grained kernel barriers synchronize
entire kernels」——AllReduce 的**所有** thread block 要等 MatMul 的**所有** thread block 完成。

**真实依赖其实更细**（行 379–382）：AllReduce 是逐元素通信 + 归约。把 MatMul 的输出切成若干
tile，第 *i* 个 AllReduce task 只消费第 *i* 块；它只依赖产出这一块的那个 MatMul task。

**于是可以这样插 event**（行 382–386）：只给「产出第 *i* 块」和「归约第 *i* 块」这一对 task
插一个 event。第 1 块算完，第 1 块的归约就能走，不必等最后一块。

方向很重要：不是让 AllReduce **整体**提前，而是让**每一块**在它自己的输入就绪时立刻开始。
计算密集的 MatMul 与通信密集的 AllReduce 于是在时间上交错，Tensor Core 和网络带宽同时忙。

### 3.3 同一张计算图，可以有多个 tGraph（行 387–401）

论文专门强调「**Multiple tGraphs may represent the same computation graph**」（行 387–390）。

Figure 4(c) 给了一个**次优**的 tGraph：event 只捕捉**算子级**依赖，「analogous to traditional
kernel barriers」（行 391–396）。它当然也能跑对，但并行度退回到 kernel 级。

所以表示能力不等于性能，**如何插入 event 才是优化点**。§4 做的事就是「inferring **precise**
data dependencies to maximize concurrency and minimize synchronization overheads」（行 397–401）。

> 这条对工程实践很重要：megakernel 的收益不来自「合进一个 kernel」本身，而来自
> **event/依赖切得够细**。DeepGEMM Mega MoE 的 ring buffer + mbarrier 也是同一逻辑。

### 3.4 与 CUDA Graphs 的边界

论文用单独一段做定位（行 402–414），逐点对比：

| 维度 | CUDA Graphs | tGraph |
|---|---|---|
| 实例化 | 静态 | 静态 |
| 依赖编码 | 显式，**kernel 级** | 显式，**SM task / sub-kernel event 级** |
| 表示内容 | kernel **launch 顺序** | intra- **and** cross-operator 依赖 |
| 同步机制 | stream 语义 | 显式 event，跨 SM 细粒度同步 |
| 融合/重叠边界 | 局限在 **kernel 边界** | 单 kernel 内即可重叠 compute 与 communication |

原文结论（行 411–414）：CUDA Graphs「confines overlap and fusion to kernel boundaries」；
tGraph 则「enabling fine-grained synchronization across SMs and overlap of computation and
communication **within a single kernel**」。

换句话说：CUDA Graphs 治的是 **launch 开销**（N 次 launch 压成 1 次图提交），
但**不改变依赖粒度**，所以治不了 kernel barrier。

### 3.5 承接：Figure 5 的编译流水线

§3 末尾紧接着 Figure 5 图题（行 415–448），预告 §4：

- Q/K/V/A/O/R 分别是由 query projection、key projection、value projection、attention、
  output projection、RMSNorm 分解出的 **task 集合**（行 416–434）；
- D₁、D₂ 是 normalization 阶段插入的 **dummy task**，目的是让每个 task 只有**一个**
  triggering event（行 435–445）；
- (f) 展示如何 **linearize** tGraph 并以「uniform, canonical representation」存储（行 445–448）。

这两个 dummy task 和「区间编码」是 §4.4、§4.5 的伏笔。先记住 Figure 5 里的几个名字：
**A₁、A₂** 是两个 attention task，**O₁** 是 output projection 的一个 task。

> **不这么做会怎样**：如果 event 只画在算子之间，tGraph 就退化成 CUDA Graph 的另一种写法——
> 图还是那张图，屏障还是整算子级。第 1 块 MatMul 早就算完了，第 1 块 AllReduce 也只能等
> 最后一块。

## 4. 编译器五步（§4，行 449–1146）

**这一步解决什么问题**：手工插 event 不可扩展，需要一条自动、且表示开销有界的流水线。

§4 的输入是「computation graph + associated inference configuration」，输出是「an optimized
tGraph **specialized for the target configuration and underlying GPU architecture**」
（输入输出见行 451–454，端到端流程见 Figure 5、行 455–457；整节行 449–457）。两个阶段：

1. **§4.1 tGraph Generation**（行 458–1125）：把算子图编译成静态 tGraph；
2. **§4.2 Task Implementation Generation**（行 1126–1146）：为每个 task 生成一个能在单个 SM 上
   跑的 CUDA device function。

一句话：**§4.1 决定「有哪些 task、谁等谁」，§4.2 决定「每个 task 怎么算」。**

§4.1 依次做五件事（Figure 5 的 a→f 对应）：

| 步 | 名称 | 快照行 | 做什么 |
|---|---|---|---|
| 1 | Operator decomposition | 462–468 | 按输出张量切分，把算子分解成若干 SM 级 task |
| 2 | Dependency analysis | 469–519 | 枚举 producer/consumer task 对，重叠即插 event |
| 3 | Event fusion | 520–779 | successor-set / predecessor-set 两种融合，消除冗余 event |
| 4 | Normalization | 780–951 | 用空 task 把每个 task 的 event 扇入/扇出压到 1 |
| 5 | Linearization | 1101–1125 | BFS 重排，使同一 event 触发的 task 索引连续 |

前两步「生图」，后三步「化简并编码」。下面逐步走，每一步都用一个论文里真实存在的例子。

### 4.1 步一：Operator decomposition（行 462–468）

**这一步解决什么问题**：一个算子太大，一张 SM 吃不下；只做成一个 task，又喂不满所有 SM。

做法是**切输出**（行 463–467）：把每个算子的输出张量分区，每个 task 计算输出中一个
**disjoint** 的子集，于是同一算子的多个 task 可以跨 SM 并行。

论文自己给的例子是矩阵乘：**输出可以同时沿行、列两个维度分块**，以暴露并行度（行 467）。
比如一个 `[M, N]` 的输出，横着切几刀、竖着切几刀，每个 task 负责其中一块不相交的矩形。
task `(i, j)` 只写第 `(i, j)` 块输出，读 A 的第 `i` 行块和 B 的第 `j` 列块。
这些 task 之间没有依赖，可以同时上不同的 SM。

切分策略不是拍脑袋（行 467–468）：

- 性能同时取决于 problem shape 与目标 GPU；
- MPK 选择**最小化 device memory → shared memory 搬运量**的策略，因为 device memory 访问
  远比 shared memory / CUDA Core / Tensor Core 计算贵；
- 默认生成的 task **数与 SM 数成正比**，以促进负载均衡；
- 用户也可以通过设定各输出维度的并行度来自定义策略。

真实模型里的规模（§6.7 行 1638–1642，Table 2 行 1687–1737）：

| 指标 | 数值 |
|---|---|
| 平均每算子 task 数 | **32–47 个** |
| Qwen3-8B：算子数 → task 数 | **293 → 13,867** |
| Qwen3-8B 的 Tasks/op | **47.3** |
| Qwen3-1.7B / 30B-A3B 的 Tasks/op | **35.6 / 32.2** |
| batch=1 时的 pointwise 算子（如 normalization） | 退化为 **1 个 task** |

分解的目的很直接：**给 runtime 足够多的独立工作，把 SM 喂满。**

> **不这么做会怎样**：一个算子只出一个 task，大 GEMM 会压在一个 SM 上，切分带来的并行度完全
> 拿不到；293 个算子也永远变不成 13,867 个 task。反过来切得过细，每个 task 的加载与调度开销
> 会盖过计算——「与 SM 数成正比」就是论文给的下限保护。

### 4.2 步二：Dependency analysis（行 469–519）

**这一步解决什么问题**：决定「谁等谁」，而且只等真正需要的那一块数据。

算法本身很短（行 470–519）：

1. 对任意两个**共享同一张量**的算子，**枚举所有 task 对** `(t1, t2)`；
2. 当且仅当 `t1` 产出的输出区域与 `t2` 消费的输入区域**重叠**时，为这一对引入一个 event `e`；
3. 插入两条边 `(t1, e)`、`(e, t2)`。

语义：`t2` 在 `t1` 产出所需数据前不得开始。

**还是走 MatMul → AllReduce 这个例子。** 设 MatMul 的输出被切成若干块，`t1` 是产出第 *i* 块的
那个 task；AllReduce 的第 *i* 个 task 是 `t2`，它只读第 *i* 块。枚举所有 task 对时，只有
`(t1, t2)` 这一对的「输出区域」和「输入区域」重叠，于是只给它插 `e_i`。

结果：`t2` 只等 `t1`，不等整个 MatMul。这就是 §3.2 那张图的机械化生成过程。

论文对这条算法的评价是：**它保留所有 producer-consumer 依赖，同时暴露独立 task 之间的并行**
（行 519）。这也是对 §3.3 图 4(c) 那个反例的回答——「precise dependency」是可以自动推断的。

**代价**：不加处理时，「每个重叠 task 对一个 event」会爆炸。整个评测里，最终要编码的
producer-consumer task 对依赖有 **69,000–162,000** 条（§6.7 行 1643–1653）。
下一步就是来收拾这个数量的。

> **不这么做会怎样**：要么退回整算子屏障，MatMul 与 AllReduce 的细粒度重叠直接消失；要么给
> 所有 task 对都插 event，event 数量与 task 对数量同阶，runtime 还没跑就先被同步点压垮。

### 4.3 步三：Event fusion（行 520–779）

**这一步解决什么问题**：依赖分析会发出海量 event；很多 event 的「等待集合」完全一样，
分开存不带来任何调度自由度，只是徒增图复杂度。

先定义两个集合函数（行 528–551）：

- `InTasks(e)`：触发 `e` 的 task 集合；
- `OutTasks(e)`：依赖 `e` 的 task 集合。

两种融合互为镜像。

**（1）Successor-set fusion：合并「同一批消费者的前驱」**

**条件**：`OutTasks(e1) = OutTasks(e2)`（Def 4.1，行 552–659）。

**做法**：删掉 `e1`、`e2`，引入融合 event `e'`，使
`InTasks(e') = InTasks(e1) ∪ InTasks(e2)`、`OutTasks(e') = OutTasks(e1)`。

**为什么合法**（行 554）：这些消费者本来就要等**全部**这些 event 激活，分开表示不会让它们
早开始一秒。

**论文的例子**（行 637–659）：Figure 5(b) 的 `e10` 和 `e14` 都是 task `O1` 的前驱，
融合为 Figure 5(c) 的 `e4`。

走一遍：`O1` 是 output projection 的一个 task，它要等 `e10`、`e14` 都变绿。这两个 event 的
消费者集合相同（都服务 `O1`）。分开存，`O1` 还是得等两个都绿；合并成一个 `e4`，`O1` 只需
等一个，语义不变，图少一个节点。

**（2）Predecessor-set fusion：合并「由同一批生产者触发的 event」**

**条件**：`InTasks(e1) = InTasks(e2)`（Def 4.2，行 660–779）。

**做法**：删掉 `e1`、`e2`，`e'` 满足 `InTasks(e') = InTasks(e1)`、
`OutTasks(e') = OutTasks(e1) ∪ OutTasks(e2)`。

**为什么合法**（行 662）：这类 event 会被**同时**触发，分开维护只是徒增图复杂度。

**论文的例子**（行 745–779）：Figure 5(c) 的 `e4`、`e5`、`e6`、`e7` 都依赖 task `A1`、`A2`，
合并为新 tGraph 中的单个 `e4`。

走一遍：`A1`、`A2` 是两个 attention task，完成后同时触发四个 event；四个 event 的触发者集合
相同。合并成一个 `e4` 后，扇出变成四个消费者集合的并集——该等的一个不少，节点少三个。

**（3）数字**

§6.7（行 1643–1653，Table 2）给了很硬的一组数：

| 指标 | 数值 |
|---|---|
| 最终 tGraph 的 event 数 | **1,142–2,366 个** |
| 这些 event 编码的 task 对依赖 | **69,000–162,000 条** |
| 同步事件缩减 | **37–118×** |
| 缩减最大的模型 | **MoE（Qwen3-30B-A3B，118×）** |

MoE 收益最大，因为专家路由产生大量**多对多依赖**，特别适合融合。

> **不这么做会怎样**：event 数量会与 69,000–162,000 条依赖同阶。runtime 每次 event 激活都要
> 多一次同步，scheduler 的 event queue 被大量「其实等价」的同步点淹没，§5 的事件驱动循环
> 根本跑不快。融合必须严格保持「等的东西不变」，所以论文用 `InTasks`/`OutTasks` 做判据，
> 而不是拍脑袋合并。

### 4.4 步四：Normalization（行 780–951，Figure 6 行 792–802）

**这一步解决什么问题**：runtime 要跨 SM **并行**执行 task、更新 event，表示必须**定长、无间接
索引**。可是 task 和 event 的邻居数量都是可变的。

两个具体麻烦（行 780–785）：

- task 可能依赖/触发**任意多个** event——按「每 task 预留最大数量」存会严重浪费内存；
- event fusion 后，一个 event 可能触发**任意多个** task——按「每 event 预留最大扇出」存同样贵。

**做法**（行 805–934，Figure 6）：做两种图重写，把图变成**功能等价**、但**每个 task 至多一个
dependent event、至多一个 triggering event** 的形式。这样每个 task descriptor 只需存**一个**
dependent-event id 和**一个** triggering-event id。

**（1）降触发扇出（图 6a，行 813–879）**

当 task `T0` 触发 `e1,…,ek` 时，引入新 event `e'` 和 `k` 个**空 task** `T1,…,Tk`
（不计算、依赖 `e'`）。之后 `T0` 只触发 `e'`，每个 `Ti` 只触发对应的一个原 event `ei`。

论文指出，Figure 5(e) 就是用这个变换把 `A1`、`A2` 的 triggering event 数量降到 1。

**（2）降依赖扇入（图 6b，行 880–934）**

当 task `T0` 依赖 `e1,…,ek` 时，引入新 event `e'` 和 `k` 个**空 task** `Ti`
（不计算、**触发** `e'`）。之后 `T0` 只依赖 `e'`，每个 `Ti` 只依赖对应的一个原 event `ei`。

§3.5 提到的 Figure 5(e) 的 dummy task `D1`、`D2`，就是 normalization 插入的这类空 task
（行 435–445）。

**论文的真实例子**（行 937–951）：Figure 5(d) 里 `A1`、`A2` 各有**两个 fan-out event**，
因为 **RMSNorm 和 output projection 都依赖 attention，可以并行执行**。这就是一个 fork：
一个 task 要喂两个下游分支。对 `A1`、`A2` 各做一次图 6a 的重写（插入一个 `e'` 和两个空
task），它们就都只触发一个 event 了。

**开销**（行 937–951、§6.7 行 1656–1674）：normalization **只在图含 fan-in / fan-out 的
fork/join 处**才插辅助 task/event。真实模型通常「deep 而非 wide」（大量顺序算子），实践中
开销**始终 <1%**。

评测的三个 LLM forward tGraph **几乎完全是顺序的**，没有 fork/join group（Q/K/V projection
等会扇出的算子已被 fusion 成单个算子），normalization 基本不改动它们。

但它对**正确性**仍是必需的——如 Figure 5 的**未融合** Q/K/V 例子，normalization 把每个 task
的 event 扇入/扇出压到 1，保住「固定大小、无间接索引」的编码。

> **不这么做会怎样**：task descriptor 就只能存变长 event 列表。按最大值预留会浪费内存；
> 用变长数组 + 间接索引，runtime 跨 SM 读取时每次都要二次寻址。normalization 不改变并行度，
> 它买的是「定长编码」这个工程属性。

### 4.5 步五：Linearization（行 1101–1125，Algorithm 1 行 952–1098）

**这一步解决什么问题**：normalization 只管住了「task → event」的元数据；「event → task」的
存储还没解决——一个 event 仍可能触发很多 task。

问题很具体（行 1102–1112）：Figure 5(e) 的 `e5` **触发四个 task**。如果逐条存出边索引，
这里就要存四个 id，还要处理变长。

**算法**（Algorithm 1，行 952–1098）：基于 **BFS**。

1. 从「无 dependent task 的 event」开始，全部入队（行 1013–1015）；
2. 每出队一个 event `e`，就把所有 `dependent_event = e` 的 task 依次入队（行 1033–1061）；
3. 当某 task 触发的 event `e'` 的所有触发 task 都已在 `T` 中时，再把 `e'` 入队（行 1074–1093）。

**关键保证**（行 956–969，Lines 7–9）：所有依赖同一 event 的 task，在最终 task 顺序 `T` 中
**连续**。

**编码**（行 1113–1125、Figure 5(f)）：于是 event 的扇出只用 **first + last task index**
两个索引表示（一个区间），不需要显式存消费者列表，而且不丢依赖。

device memory 布局是：每个 task 只记其 dependent / triggering event 的索引；每个 event 存
**激活所需的触发计数**，一旦激活，runtime 就 launch 落在 `[first, last]` 区间内的所有 task。

**数字**（§6.7 行 1677–1686，Table 2）：

| 模型 | footprint 缩减 | 绝对值 |
|---|---|---|
| dense 模型 | **4.4–5.9×** | Qwen3-8B：**110,932 → 18,928 bytes** |
| MoE 模型 | **15.0×** | — |

MoE 的高扇出专家路由 event 在区间编码下最省。

> **不这么做会怎样**：每个 event 都要存一张消费者列表。`e5` 这类高扇出 event 在 MoE 里尤其
> 多，dense 模型上 footprint 就会从 18,928 回到 110,932 bytes（Qwen3-8B），而且每次激活都要
> 二次寻址。一次 BFS 重排，换来的是「两个整数就能表示任意扇出」。

### 4.6 每个 task 怎么算：§4.2（行 1126–1146）

**这一步解决什么问题**：图只说了「谁等谁」，没说每个 task 内部怎么算。

§4.2 为每个 task 生成一个在 GPU SM 上执行的 device function（行 1126–1138）：

- **复用超优化**（行 1131–1133）：沿用 Mirage 的 **compiler superoptimization [40]**；
- **关键差异**（行 1134–1138）：粒度是 **thread block**，不是 kernel。每个 compute task 关联
  一份**参考 PyTorch 实现**，MPK 用 Mirage superoptimizer 搜索一个优化的 **thread-block
  graph**，交给 Mirage compiler 生成 CUDA 实现；
- **生成物**（行 1138）：包含 **intra-SM 优化**——software pipelining、register reuse、
  降低 shared-memory bank conflicts 的布局优化。

**cross-task** 的软件流水不在这里做——那是 §5 运行时的职责。换句话说：§4.2 把每个「积木」
做到极致，§5 负责把这些积木在 device 内交错推进。

> **不这么做会怎样**：如果按 kernel 粒度做超优化，生成的代码会假设自己独占整个 GPU；塞进
> mega-kernel 后 task 之间要共享 SM，这个假设直接崩掉。把粒度压到 thread block，才使
> 「一个 task = 一个 thread block = 一个 SM 工作单元」的映射成立。

### 4.7 五步合起来看

1. **前两步生图**：decomposition 决定并行度，dependency analysis 决定精确度。
2. **后三步化简并编码**：fusion 减同步点，normalization 压 task 元数据，linearization 压
   event 扇出。
3. **precision 靠自动推断，不靠手工**：依赖下沉到「输出 tile 是否重叠」这一最细可判定粒度，
   fusion 再机械化地消冗余。这正是 §3.3 所说的「precise event」的自动实现。
4. **normalization / linearization 是表示问题，不是性能问题**：它们不改并行度，只把
   task→event 压成一个 id、把 event→task 压成一个区间（<1% / 4.4–15× 都是表示开销）。
5. **§4.2 的粒度选择是范式关键**：thread block 级超优化，才让整条链路自洽。

> **不这么做会怎样**：少了前两步，图「对」但没并行度；少了后三步，图「能跑」但表示膨胀——
> 要么 event 多到 runtime 追不上，要么 descriptor 变长到无法定长寻址。五步是一套连招，
> 拆开就断。

## 5. 运行时怎么跑（§5，行 1147–1368）

**这一步解决什么问题**：图编译好了，谁在 device 内推进它，而不是靠 host 逐段 launch。

§5 的组织是（行 1148–1174）：§5.1 事件驱动执行 → §5.2 两种 task launch（JIT/AOT）及其取舍 →
§5.3 进一步降开销的运行时优化（行 1165–1174）。

### 5.1 一次 launch，SM 分成两种角色

in-kernel runtime 在**一个 mega-kernel 内**跨所有 SM 执行整张 tGraph；一旦 launch，该 kernel
**持续管理计算与通信直到推理负载结束**（行 1149–1154）。

SM 被显式分成两类（行 1155–1164）：

| 角色 | 部署 | 本地状态 | 循环做什么 |
|---|---|---|---|
| **worker** | 每个 worker 占一个物理 SM | task queue | 取 task → 执行计算/通信 → notify 它的 triggering event |
| **scheduler** | warp 粒度；预留 4 个 SM、共 **16 个 scheduler warp**（每 SM 最多 4 个活跃） | event queue | 轮询已激活 event → 把对应 task 下发给 worker |

角色划分**在 kernel-launch 时固定，并与 GPU 物理 SM 数匹配**（行 1164），从而避免 kernel 内的
动态角色切换开销。

> **范式要点**：这里就是 `docs/00` 判据 **C4「设备内协调」+ C5「SM 级调度」** 的落地——
> 调度状态（event/task queue、semaphore）全在 device memory，host 在 launch 后不再参与。

### 5.2 事件驱动：event 不是被动屏障，是驱动机制（行 1175–1231）

执行从哪开始（行 1177–1198）：每张 tGraph 有一个**无前驱的 start event**（如 Figure 7 的 e0），
最先入 scheduler 的 event queue。scheduler 取到它，就 launch 依赖它的所有 task。

闭环是这样转的（行 1211–1217）：

1. 被 launch 的 task 下发到某个 worker；
2. worker 执行完，**通知该 task 的 triggering event**；
3. 当某 event 的**所有前驱都完成、累计触发到所需次数**时，event 变为 **activated**；
4. 激活的 event 入队到某个 scheduler，继续传播。

**event 就是驱动 task 执行的机制**，不是「挡在 task 前面的一道墙」。

这套模型还有一个意外的好处：**不需要拓扑画像**（行 1218–1225）。计算与通信统一表示成同一张
tGraph 里的 task、由同一 scheduler 分发。依赖满足即可跑：**快的链路更早触发 event、更早释放
下游；慢链路只拖它自己的下游**。

于是 scheduler **无需显式知道链路带宽/拓扑距离**，同一张 tGraph + scheduler 就能处理异构互连
（intra-node NVLink 与 inter-node 网络），靠「数据到达」而不是「静态拓扑模型」反应。

### 5.3 JIT 与 AOT：两种下发语义（行 1232–1308）

**两种语义**（行 1234–1238）：

- **JIT（just-in-time）**：scheduler 在依赖 event **完全激活后**才把 task 派给 worker，
  task 可立即执行。
- **AOT（ahead-of-time）**：runtime 在**前驱 event 激活之前**就把 task **预入队**到某 worker
  的 task queue；worker 在本地**等到依赖满足**才执行。

**为什么互补**（行 1239–1247）：

- JIT 能**适应负载不均**。LLM attention 因序列长度而耗时高度可变，静态分配无效；JIT 在
  attention 完成后才 launch 下游，**先完成的 worker 多干活**，改善端到端时延。
- 但 JIT 有**更高的下发时延**，因为多了一次 worker↔scheduler 往返。

**同步步数对比**（行 1247–1295，Figure 8）：

| | JIT（图 8a） | AOT（图 8b） |
|---|---|---|
| `T1 → e → T2` 的路径 | worker(T1) → scheduler 通知 → scheduler 取 e、launch T2、入队 worker → worker 取 T2 | T2 已预入队；worker 只需等 `e` 激活 |
| 同步步数 | **两步**（worker→scheduler、scheduler→worker） | **一步**（event trigger） |
| 长处 | 动态均衡 | 低 per-task launch 时延 |
| 短处 | 多一次往返 | 静态分配，可能不均 |

**混合策略**（行 1296–1302）：

- 编译期按「执行时间是否数据相关、易致运行时不均」给**整个 operator** 打 JIT/AOT 标签
  （operator 粒度，同一 operator 的所有 task 共享同一 mode）；
- **数据相关耗时的 operator（如 attention）标 JIT**，其下游**保持 JIT 直到遇到 global
  barrier**（一个必须由**所有上游 task**触发的 event）；barrier 抹平累积不均，其后的
  operator 适合 **AOT**；
- worker 维护 **JIT / AOT 两条 queue**，**优先 JIT**；JIT 空了再检查 AOT 队首依赖是否已激活。
  这样 worker **只在确实无就绪工作时才阻塞**。

**下发侧优化**（行 1303–1308）：scheduler **只处理会 launch JIT task 的 event**；AOT task
**全部预入队**，并按 **round-robin** 摊到各 worker（operator 产生的 task 数通常与 worker 数
成正比）。预入队降低了 scheduler 负载并摊薄下发开销，JIT 则继续动态均衡 SM 工作。

> **设计张力一句话**：JIT = **保负载均衡但贵**；AOT = **省下发但静态**。MPK 用「编译期标签
> + global barrier 分界 + worker 双队列优先级」把两者缝在一起，用静态图拿到动态负载均衡。

### 5.4 三个运行时优化（行 1309–1368）

三个优化都服务于同一目标：**让比传统 kernel 细得多的 task，其下发/执行开销被摊薄**。

**（1）Paged shared memory（行 1312–1325）**

**问题**（行 1313–1318）：常规模型下 shared memory 是 thread block 私有、**kernel 生命周期
内**存在；kernel-per-operator 下每个 kernel **独占**其 shared memory。这个假设**阻碍
cross-task software pipelining**——「为下一 task 预取数据」与「当前 task 计算」可能**同时
需要 shared memory**。

**机制**（行 1319–1325）：把 shared memory 切成**固定大小页**，task 实现改为**按页**操作；
task **acquire** 若干页（按其 footprint）、不再需要时 **release**。**一旦 release 过任何页，
便不再允许 acquire 新页**（单调使用模式，简化调度）。当前 task 释放页后，MPK 即可把可用页
分给**下一个 task 立即预取数据**。

**配置锚点**（§6.1 行 1460）：page size **32 KB**，在 A100/H100/B200 上分别得
**5 / 7 / 7 页/SM**。

**（2）Cross-task software pipelining（行 1326–1363）**

**拆分**（行 1330–1334）：把每个 task 拆成 **pre-loading 阶段**（发数据搬运指令，把 tensor 的
一个 chunk 从 device memory 取到 shared memory，**不做计算**）与 **compute 阶段**。

**重叠条件**（行 1335–1363）：当两个条件同时成立时，MPK 机会式地把当前 task **T1 的 compute**
与后继 task **T2 的 pre-loading** 重叠：

1. T1 **已发出自己全部数据搬运指令**；
2. 有**足够页**给 T2 预取。

该流水**不干扰 T1**：MPK 插入必要的 **intra-SM 同步屏障**，确保 T2 的搬运不与 T1 正在进行的
搬运冲突。

**消融**（§6.6 行 1586–1596，Figure 12）：Qwen3-8B final linear、**B200** 上，cross-task
pipelining 把 task 运行时降 **1.2–1.3×**，甚至**胜过 cuBLAS based compiled kernels**。

**（3）预取 task description（行 1364–1368）**

每个 worker 在 device memory 维护 JIT/AOT 两条 task queue；每个 task 关联一份
**task description**（输入/输出 tensor 与配置参数），**当前实现每份占 352 bytes**（§6.1）。

为降低 enqueue/dequeue 时延并隐藏 device-memory 访问，MPK 用**轻量预取**把**即将用到的 task
description 先取到 shared memory**。

### 5.5 实现与配置

- **规模与组件**（§6.1 行 1396–1407）：约 **44K 行 C++、42K 行 CUDA、10K 行 Python**；
  in-kernel runtime 用 CUDA 编写，用 **device memory 中的 semaphore** 协调 worker 与
  scheduler；compute task 的 CUDA 实现由 **Mirage superoptimizer** 生成；跨 GPU 通信用
  **NVSHMEM**。
- **降 task-launch 开销**（行 1409–1413）：event/task queue 是 device memory 中的**环形缓冲**，
  enqueue/dequeue **只用低成本的 `atomicAdd`**；采用**去中心化调度**（每个 scheduler 用本地
  状态分配 task），避免全局协调的通信/同步开销。论文指出 runtime 也**支持改写为全局协调
  调度**，但列为未来工作。
- **角色配置**（Table 1，行 1437–1460）：scheduler 预留 **4 个 SM / 16 个 scheduler warp**，
  其余为 worker。

| GPU | SM 数 | worker | scheduler warp |
|---|---|---|---|
| A100 | **108** | **104** | **16** |
| H100 | **132** | **128** | **16** |
| B200 | **148** | **144** | **16** |

这组数字给出「上百个 worker、个位数百分比 scheduler 开销」的直观量级。

> **不这么做会怎样**：如果调度留在 host，就算把算子全融进一个 kernel，也还是要付 293 次
> launch 的账（行 1618–1628）；如果 task 还是 kernel 粒度，跨 task 流水和 JIT 的动态均衡都
> 无从谈起；如果 shared memory 仍是 thread block 私有，为下一个 task 预取就会和当前 task
> 抢空间（行 1313–1318）。

## 6. 评估与局限（§6，行 1369–1737；§7，行 1739–1764）

**这一步解决什么问题**：把「机制」换算成端到端数字，并说清这些数字的适用边界。

### 6.1 论文要回答的四个问题

原文（行 1371–1383）明确 §6 要回答四个问题，对应四个小节（§6 的组织见行 1371–1390）：

| 问题 | 小节 | 快照行 | 一句话 |
|---|---|---|---|
| mega-kernel vs kernel-per-operator | §6.3 端到端 | 1467–1517 | 相对 vLLM/SGLang 的延迟/吞吐 |
| 支持动态负载 | §6.4 MoE 案例 | 1520–1553 | 混合负载均衡 + fused gather-GEMM |
| 多 GPU 可扩展性 | §6.5 多 GPU | 1556–1580 | TP 下 8×H100 |
| 运行时优化贡献 | §6.6 消融 | 1583–1628 | pipelining / overlap / launch |

论文把评测**聚焦在 LLM serving**（行 1384–1390），两个理由：一是 serving 有 SGLang/vLLM 这类
高度优化的 kernel-per-operator 基线，对照「严格」；二是 serving 天然动态（batch size、序列
长度、prefill/decode 混合逐轮变化），能同时压测编译器与运行时。

论文强调二者与模型无关（「the MPK compiler and runtime are model-agnostic」，行 1390），
但**未给出 serving 之外的评测**。

### 6.2 实验设定（复现要点，行 1391–1464）

- **接入方式**（§6.1 行 1393–1395）：MPK 作为 **PyTorch compiler backend**，用法即
  `torch.compile(backend=MPK)`；生成 mega-kernel 并返回一个可调用的 PyTorch function，
  **每次调用只发一次 launch**（行 1514 呼应）。
- **模型与硬件**（§6.2 行 1457–1464）：**5 个 LLM**（含 dense 与 MoE，跨多个规模），
  **三代 NVIDIA GPU：A100 / H100 / B200**；配置见表 1（行 1437–1460）。
- **shared memory 页**：统一 **32 KB**，A100/H100/B200 → **5/7/7 页/SM**（行 1460）。
- **负载设定**（行 1464）：**offline batched-inference**，最大 batch size 在 **1–16** 间扫；
  每条请求 **prompt 64 token、生成 1024 token**。论文用 offline 设定「消除请求并发不足引起
  的服务端停顿」，以获得可控对比——这同时意味着**未覆盖在线到达分布（arrival pattern）**。

### 6.3 端到端：单 batch 延迟 1.0–1.7×（§6.3，行 1467–1517）

**基线**（行 1469–1497）：SGLang 与 vLLM，二者都是 kernel-per-operator，依赖 FlashInfer /
FlashAttention（attention）、cuBLAS/cuTLASS（GEMM）、CUDA/Triton（其余）；均 bf16、启用
paged attention 与 continuous batching。

**关键架构差异**：MPK 把 **page allocation 与 request scheduling 直接做进 mega-kernel**，
而两个基线在 **CPU** 上做，额外承担 host–device 同步与下发开销。

**主结果**（行 1499–1504）：单 batch 推理下，MPK 跨模型/硬件提升 **1.0–1.7×**；
**越小的模型、越新的 GPU 代际收益越明显**——因为每 token 计算量越小、GPU 越快，三类开销越
突出：

1. kernel-switch 开销（即便用了 CUDA Graphs）；
2. kernel 边界造成 pipeline bubble、阻断 cross-task pipelining（Figure 2）；
3. 基线在 CPU 上做 page allocation + request scheduling。

**绝对锚点**（行 1513）：Qwen3-8B 在 **A100** 上，MPK 把 per-token decode 从 vLLM/SGLang 的
**14.5 ms 降到 12.5 ms**，逼近论文估算的**硬件下界 ~10 ms**（按 16 GB 参数 @ 1.6 TB/s 带宽）。

这个「离下界还有 25%」的量级很关键：说明端到端已非常接近纯内存搬运极限，剩余差距即
计算/通信/调度。

**可编程性**（行 1514–1517）：论文额外主张 MPK 的**开发效率**优势——基线需大量工程为新模型
接专用 kernel，MPK 只需几行 PyTorch 改动即可 mega-kernel 化；相对**原生 PyTorch** 超过 **10×**。

> 别混用口径：**10× 是 vs 原生 PyTorch**，不是 vs vLLM/SGLang；后者是 1.0–1.7×。

### 6.4 MoE 案例：动态负载的两项优化（§6.4，行 1520–1553）

**MoE 用同一套 task 抽象表达**（行 1523–1536）：一个 MoE block 被 lower 成
routing → dispatch → expert compute → combine 的 **SM 级 task 链**；dispatch/combine 的
all-to-all 通信**与 expert 计算放进同一张 tGraph、由同一事件驱动 scheduler 分发**。

于是通信-计算在事件触发时**自动重叠，不依赖 CUDA stream 或 host 侧同步**。这是把
`docs/00` §2.5「通信-计算 overlap 的双 kernel 方案」（DeepEP/FlashMLA）对照到 MPK 范式的
位置。

**Hybrid workload balancer**（行 1537–1550）：每专家 token 数只有运行时才知道。静态划分在
倾斜路由下会严重不均；全动态 Grouped-GEMM 又有细粒度同步开销。

MPK 取**混合策略**：

- 编译期按专家切 task；
- 运行时每个 task 拿到 `topk-softmax` 产出的 **meta-tensor**（含激活专家数、各专家 token 数），
  据此**动态微调**负载分配。

Figure 10（行 1518–1519）报告：该策略**在所有 batch size 上胜过纯静态划分**。

**Fused gather-GEMM**（行 1551–1553）：Hopper/Blackwell 上用 TMA 需先把路由到同一专家的
token **gather 成连续布局**。Qwen3-30B-A3B、batch=1 下，该预处理在 SGLang 上**占总 MoE 时间
高达 11%**。

MPK 把 TMA gather 换成**直接融进 GEMM task 数据加载阶段的异步 token 级拷贝**，消掉独立
gather kernel 与额外调度点。这与 DeepGEMM Mega MoE 的「dispatch+GEMM 融合」是**同一动机的
两种实现**：消除 gather 这类「只为对齐布局」的搬运。

### 6.5 多 GPU：TP 下最高 10×（§6.5，行 1556–1580）

**设定**（行 1558–1566）：NVIDIA **H100 DGX**，采用 **tensor model parallelism**（Megatron-LM）；
用户只需在 attention 与 gated MLP 后插入 `AllReduce`。

MPK 自动把 collective 编成两类 task：**inter-GPU 数据传输 task**（用 NVSHMEM
`nvshmem_signal_wait_until` 实现）+ **local reduction task**，从而把同步 collective 变成
可并入事件驱动 runtime 的**异步 task**。

**结果**（Figure 11 行 1554–1555，行 1567–1577）：

- 相对 **PyTorch**（手写 kernel + CUDA Graphs + `torch.compile` 组合）最高 **10×**；
- 相对 **SGLang/vLLM**，扩展到 **8×H100** 时 **1.1–1.4×**。

论文归因于基线缺失的三点：

1. page allocation + request scheduling 进 mega-kernel，消 CPU 下发；
2. 异步模型让 compute 与 collective **重叠**；
3. 消除 kernel barrier、启用 cross-task pipelining。

**一个值得注意的边界**：多 GPU 主结果（Figure 11）用的是 **Qwen3-1.7B（dense）+ TP**；
MoE 的 **expert parallelism** 虽有机制描述（§6.4），但**未作为多 GPU 端到端图表呈现**。

也就是说「EP 的 all-to-all 重叠」是机制论证，端到端数字主要在 TP。

### 6.6 两块消融：钱花在哪（行 1583–1737）

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
| （算子总数） | ops | 229 | 293 | 533 | Table 2 |
| event fusion | 事件数 / 缩减 | 1,870 / 37× | 2,366 / 68× | 1,142 / **118×** | 1643–1653 |
| normalization | 变更 | 近乎全顺序、无 fork/join；开销 <1% | — | — | 1656–1674 |
| linearization | footprint 缩减 | 4.4× | **5.9×**（110,932→18,928 B） | **15.0×** | 1677–1686 |

一句话：**计算越多 / MoE 越重，编译期收益越大**（MoE 的 118× fusion、15.0× linearization
都是峰值）；而 normalization 在真实 LLM forward 上**几乎不动图**。

### 6.7 论文自己划的边界（§7，行 1739–1764）

**Resource footprint**（行 1740–1748）：mega-kernel 的常见担忧是「单 kernel 必须在每个 SM 上
预留所有被融合算子的资源之和」。

MPK 的回答：task 在 SM 上是**时分复用而非静态划分**，每个 SM 只用**当前 task** 所需资源
（§5）；shared memory 走 paged 抽象（§5.3）按 task 粒度 acquire/release。**唯一例外是
per-thread register**——它固定在所有 task 类型的最大值（行 1748）。

→ 落地时**寄存器压力**是 mega-kernel 融合范围的硬约束，shared memory 则被 paged 抽象化解。

**集成手写 kernel 与移植新硬件**（行 1749–1764）：每个 task 是统一调用约定的 CUDA device
function，因此有两层灵活性：

1. 手写 kernel 可**包成一个 task** 直接进事件驱动调度，运行时与 tGraph 不用改——MPK 能
   「手写 + 自动」混用；
2. 移植新 GPU **只需改 per-task code generator**，tGraph 表示、图级变换、运行时、scheduler
   都不变——**per-task codegen 是唯一硬件相关组件**。

这与 DeepGEMM 的**全手工** mega-kernel 形成对照：MPK 把「硬件相关」收缩到最底层 codegen，
其余抽象跨代共享。

### 6.8 复现要点与局限

**复现要点**（能照着做的）：

1. 接入面：`torch.compile(backend=MPK)`；模型架构从 HuggingFace 载入，bf16。
2. 设定：offline batched-inference，batch 1–16，prompt 64 / 生成 1024；page size 32 KB；
   预留 4 SM / 16 scheduler warp；TP 时在 attention 与 gated MLP 后插 `AllReduce`。
3. 基线：SGLang / vLLM（均 paged attention + continuous batching）+ 原生 PyTorch；
   多 GPU 时 PyTorch 基线是「手写 kernel + CUDA Graphs + `torch.compile`」组合。
4. 报告口径：**1.0–1.7×**（单 batch）与 **1.1–1.4×@8GPU** 是 **vs SGLang/vLLM**；
   **10×** 是 **vs PyTorch**，两者不可混用。

**局限 / 未覆盖**（论文范围之外或明确未做）：

- **只在 LLM serving 上评测**（行 1390、1457–1464）；论文自称 model-agnostic，但无训练/其它
  DNN 数据。
- **offline batched**（行 1464）——未覆盖真实在线服务的 **arrival pattern** 与尾延迟。
- **batch 上界仅 16、prompt 64、生成 1024**——长上下文 / 大 batch / 变长生成未评估。
- **多 GPU 端到端只在 TP + dense（Qwen3-1.7B）**（行 1554、1568）；**EP 的 all-to-all 重叠是
  机制论证**，无端到端图表。
- **动态仍依赖「每 batch size 一张静态 tGraph」**（§6.1 行 1427–1436：按 2 的幂生成多张图，
  运行时按当前 batch size 选图；相关范围行 1414–1436）——是**有限离散的动态**，不是任意
  变长形状。
- **去中心化调度的取舍未探**：当前默认**去中心化调度**，论文自己把「全局协调调度及其
  trade-off」列为**未来工作**（行 1413）。
- **寄存器固定为各 task 最大值**（行 1748）——融合的算子越多，寄存器上限越可能成为瓶颈。
- **page size 固定 32 KB**（行 1460），论文未给 page size 的自适应/调优分析。

> **不这么做会怎样**：如果不划这些边界，1.0–1.7× 很容易被读成「任何场景都快 1.7×」。
> 论文自己的数字只在 offline、batch 1–16、TP + dense 的设定下成立；在线到达、长上下文、
> EP 端到端都还没有数据。

## 7. 实验与出处索引

| 编号 | 覆盖内容 | 快照行 | 性质 | 备注 |
|---|---|---|---|---|
| P2-1 | §3 tGraph | 345–448 | **纯资料分析** | 无 H100 实验 |
| P2-2 | §4 + §6.7 | 449–1146、1629–1737 | **纯资料分析** | 32–47、37–118×、4.4–15× 为论文报告值（B200，Table 2 行 1687–1737） |
| P2-3 | §5 + §6.1/§6.6 | 1147–1368、1391–1413、1583–1628 | **纯资料分析** | 1.2–1.3× / 1.1× / 0.28% / 293 launch 为论文报告值 |
| P2-4 | §6 + §7 | 1369–1764 | **纯资料分析** | 无 H100 实验 |

**本机实测佐证（非论文，供对照）**：

| 项 | 结果 | 出处 |
|---|---|---|
| CUDA Graph 重放 200 次（32 节点图） | **1.56×**（vs 逐次 launch） | `docs/04-experiments.md` §0、`harness/probe_env.out.txt` |
| 有依赖链时 4 streams | **0.96×**（并行失效）；Graph 仅 **1.05×** | `docs/00` §2.4、`harness/mitigations.out.txt` |
| persistent kernel 的 grid-wide 屏障 | 200 次 **0.330 ms**，约 1.65 µs/次 | `docs/04-experiments.md` §0、`harness/probe_env.out.txt` |

> **与生产案例的呼应**：DeepGEMM Mega MoE 用**位掩码 + ring buffer** 手工表达 task 级依赖
> （`scheduler/mega_moe.cuh` 行 140–160），本质就是**手工版 tGraph**；MPK 的贡献是把
> 「分解 → 依赖 → 融合 → 规范化 → 线性化」这套手工活全部自动化。细节见 `docs/02` 与
> `../mega-moe/`，本文不展开。

> **实测说明**：本文四项（P2-1～P2-4）均为**纯资料分析**，无 H100 实验；论文数字全部照抄
> 论文报告值，本机只提供「launch 开销、CUDA Graph 不破依赖串行」这类旁证。把「N 个小 kernel
> 串行 vs CUDA Graph vs 单 persistent kernel」的端到端对比留给后续本机复现。

## 8. 三句话小结

- MPK 把依赖从**算子粒度**下沉到 **SM 级 task + event**；event 是「任意 task 子集」的屏障，
  同一张计算图可以有多个 tGraph，好坏取决于 event 插得够不够 precise。
- 编译器五步：**分解 → 依赖分析 → event fusion → normalization → linearization**。前两步生图，
  后三步把图压成定长、无间接索引的编码（fusion **37–118×**，linearization **4.4–15×**）。
- 运行时把 SM 切成 **scheduler / worker**，用事件驱动 + JIT/AOT 混合 + paged shared memory，
  换来单次 launch 和 **1.2–1.3×** 跨任务流水；代价是静态图、寄存器取最大值，以及只在
  LLM serving 上验证过。

下一篇：`docs/02-production-cases.md`，看 DeepGEMM 的 Mega MoE 怎么用 ring buffer + 位掩码
手工实现同一套「细粒度 task 依赖」——那是一个没有编译器的 tGraph。
