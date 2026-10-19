# MegaKernel 是什么：动机、定义与谱系

> 本文是「MegaKernel 资料与实现分析」的第 1 篇。我们先讲清今天「一个算子一个 kernel」这套做法
> 到底亏在哪、MegaKernel 想解决什么，为后面读 MPK 论文和生产案例建立一个坐标系。

## 0. 先记住一个画面

把一次模型推理想成一条有几十上百道工序的产线。

**今天的做法（kernel-per-operator）**：每道工序单独开一个车间。一道工序做完，全厂停机、清场，
把半成品搬进仓库，下一个车间再开工。工序越多，停机、开门、搬货的次数就越多。

**MegaKernel 的做法**：所有工序放进同一个车间。工人（SM）从头到尾不散场，半成品通过传送带
（显存里的 ring buffer 和信号）直接交给下一道工序——上一道工序还在做，下一道工序已经能开始。

于是本文要讲的三笔账，可以一眼对上：

| 产线里的浪费 | GPU 上的名字 | 本文 |
|---|---|---|
| 每道工序后全厂停机清场 | kernel barrier（内核间屏障） | §1.2 |
| 每个车间开门关门的固定成本 | kernel launch 开销 | §1.3 |
| 半成品搬进搬出仓库 | 中间张量在 HBM 上往返 | §1.4 |

> **类比的边界**（很重要，别把类比用过头）：现实里换个车间只是多走几步路；GPU 上的「清场」意味着
> 整颗 kernel 的**所有**线程块必须全部退出，SM 上的寄存器、共享内存、软件流水状态全部作废，
> 下一个 kernel 要从零重新预热。另外，MegaKernel 用的「传送带 + 对讲机」不是免费的——设备内同步
> 本身有开销，还要额外推导死锁上界（见 `docs/11`）。所以它不是「白捡」，而是**换一种账单**。

读完这一篇，你应该能回答三个问题：

1. 「一个算子一个 kernel」具体浪费在哪三处？为什么说这三笔账其实是同一笔？
2. 算子融合、CUDA Graph、PDL、多流这些现成手段，各治哪一处、为什么都不够？
3. 什么样的 kernel 才算 MegaKernel？五条判据（C1–C5）怎么用？

## 1. 现状：kernel-per-operator 与它的三笔账

> 本文以 **MPK 论文**（*MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*，
> arXiv:2512.22219v2）的快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
> 为准（下称「快照」，共 2717 行），引用的章节号/行号都在该快照中逐行核对过；并与本仓库 P0-3 实测
> （`docs/04-experiments.md` §0）互相印证。

### 1.1 执行模型：一张 DAG，一个算子一个 kernel

今天的推理/训练系统，都先把模型表示成一张**张量程序 DAG**：节点是算子（matmul、attention、norm、
activation……），边是算子之间生产/消费的 *n* 维张量（MPK §1，快照行 53–56）。

执行时，主流做法是**一个算子一个 kernel**（kernel-per-operator）：每个算子要么由领域专家手写优化，
要么由 ML 编译器自动生成，然后按依赖顺序依次 launch 到同一条 stream 上（MPK §1，快照行 57–73）。

这套模型简单、模块化，每个算子都能单独优化、单独测试。代价是 MPK 指出的：
它**堵死了好几类跨算子的 GPU 优化**（快照行 71–73）。下面逐条算账。

### 1.2 第一笔账：kernel barrier，让流水线断在算子边界

同一条 stream 上相邻的两个 kernel 之间，GPU 会隐式插一道 **kernel barrier**：
必须等前一颗 kernel 的**所有** thread block 跑完，后一颗 kernel 的 thread block 才能开始
（MPK §1 快照行 74–77、§2.1 快照行 223–226）。

这道屏障保证了数据依赖不出错，但代价有两个。

**代价一：算子之间不能做软件流水（cross-operator software pipelining）。**
现代 GPU 内部是高度异构的：Tensor Core 算矩阵、TMA 异步搬数据、CUDA Core 做辅助运算。
想让它们都不闲着，就得让「搬运 / 计算 / 辅助」在**多个任务迭代之间**交错起来。
可是 kernel barrier 只允许在**单个 task 内部**流水（intra-task pipelining）——一跨过 task 边界就断流，
留下 pipeline bubble，硬件利用率随之下降（MPK §2.1，快照行 227–240，对应 Figure 2a/2b，快照行 204–209）。

**代价二：算子之间不能细粒度重叠（fine-grained kernel overlap）。**
因为依赖只在**整个算子**这个粒度上表达，运行时只能等上游算子全部完成，才能启动下游。
论文举的例子是 `MatMul` → `AllGather`：AllGather 是逐元素操作，每个 AllGather block 其实只依赖
**一个** MatMul block 的输出，完全可以让不同 SM 同时跑这两种算子；但 kernel barrier 要求
AllGather 的所有 block 等 MatMul 的所有 block 完成（MPK §2.1，快照行 241–293，Figure 3a/3b）。

> 换成产线的话说：每道工序的**最后一个人**没干完，下一道工序的**第一个人**就不许动手——
> 哪怕他要加工的那个零件早就做好了。

这两项优化被剥夺，值多少钱？论文给了两个数：开启跨任务流水，相对关闭可把 Qwen3-8B 最后一层
linear 的任务运行时降低 **1.2–1.3×**（MPK §6.6，快照行 1586–1596，Figure 12）；开启细粒度
计算-通信重叠，相对关闭可降低每迭代延迟 **1.1×**（MPK §6.6，快照行 1597–1615，Figure 13，4×H100 TP）。

NVIDIA 的 **PDL（Programmatic Dependent Launch）** 能让同 stream 上 kernel 之间部分重叠，
算是官方补丁；但 MPK 指出采用 PDL 需要显著工程改造，因为它从根上改变了 kernel 的结构和控制流
（MPK §1，快照行 78–84）。这点 §2.3 再展开。

### 1.3 第二笔账：launch 开销，每 token 几百到上千次「开门」

kernel-per-operator 下一次推理迭代要 launch **几百到上千个 kernel**（MPK §1，快照行 87–89）。

单次 launch 听起来不贵，但乘法很残酷。论文给了一组可以直接引用的锚点
（MPK §6.6「Kernel-launch reduction」，快照行 1618–1628，测试于 B200）：

| 项 | 数值 |
|---|---|
| Qwen3-8B 每 token 的 kernel launch 次数（kernel-per-operator） | **293 次** |
| 单次 launch 成本（eager 模式） | **3.8 µs** → 合计 **1.1 ms/token** |
| 单次 launch 成本（CUDA Graph） | **0.8 µs** → 合计 **0.2 ms/token** |
| MPK 的 in-kernel scheduler 占总运行时比例 | **0.28%** |

这组数字说明两件事。

第一，launch 开销随算子数线性累积。在 decode 这种「每 token 计算量很小」的场景里，它的占比很可观。

第二，CUDA Graphs 能把单次成本压到约 1/5、总量压到约 1/5，但**压不到 0**——论文 §6.3 明确指出，
即便用了 CUDA Graphs，kernel-per-operator 仍然有 **kernel-switch 开销**（快照行 1504–1509）。
MPK 的办法是「只 launch 一次 + 在设备内调度」：把这几百次 launch 换成一个轻量调度器，
它只占运行时 0.28%，用设备内的原子操作与环形队列维护（实现见 MPK §6.1，快照行 1409–1413）。

**本机 P0-3 实测佐证**（`docs/04-experiments.md` §0，目标卡 H100/sm90）：200 个 trivial kernel
串行 launch 耗时 **0.575 ms**（约 2.9 µs/次）；而在一个常驻 kernel 内做 200 次 cooperative
grid-wide 屏障只要 **0.330 ms**（约 1.65 µs/次）。

这是能力探测，不是公平基准，但量级上支持同一个结论：**反复 launch 的固定成本，高于在设备内同步。**

### 1.4 第三笔账：中间张量在 HBM 上往返

相邻算子之间靠 **HBM 上的中间张量**传数据：上游算子把结果写回显存，下游算子再读回来。
所以 kernel 融合的收益从来都是三件事——「避免中间结果物化、减少 device memory 访问、消除 launch 开销」
（MPK §2.2，快照行 311–313）。

那为什么编译器不干脆全融了？因为 MPK 指出现有编译器只能融**局部的小算子组**：
「忠实实现整个复杂张量程序的单个 kernel」在计算上极难、常常不可行（MPK §2.2，快照行 314–328）。
mega-kernel 范式则把融合推到极致——整图的计算与通信塞进一个持久化 kernel，用 device memory 上的
同步原语跨 SM 协调（快照行 329–331）。

一个能感知到这笔账的具体案例：Qwen3-30B-A3B 在 batch=1、SGLang 上，
「为了 TMA 好搬，先把路由到同一 expert 的 token 聚成连续布局」这个预处理步骤，
占了 MoE 总执行时间的**高达 11%**（MPK §6.4「Fused gather-GEMM」，快照行 1551–1553）。
MPK 把它融进 GEMM 的异步 token 级拷贝里，独立的 gather kernel 和那个调度点一起消失。

### 1.5 三笔账其实是一笔

| 账 | 症状 | 已有缓解（详见 §2） | 为什么仍不够 |
|---|---|---|---|
| kernel barrier | 跨算子无法软件流水、无法细粒度重叠 | PDL、算子融合 | PDL 需改 kernel 结构；融合只覆盖局部算子 |
| launch 开销 | 每 token 293 次 launch，1.1 ms（eager） | CUDA Graphs（→0.2 ms） | 仍有 kernel-switch 开销；图是静态的，控制流/形状一变就要重捕 |
| 中间张量 HBM 往返 | 上游写、下游读中间结果 | 算子融合 | 只能融相邻小算子，覆盖不了整图 |

把这三行放在一起看，会发现它们互相纠缠：**kernel barrier 逼着中间张量必须落显存**（第三笔账），
而**落显存 + 反复 launch** 又共同制造了第二笔账。三个症状，一个病根。

MPK 在 §6.3 把这三项与「CPU 侧的 page allocation / request scheduling 引起的 host–device 同步」
并列，作为 kernel-per-operator 在**小模型 + 新 GPU** 上更吃亏的原因（快照行 1501–1513）：
计算量越小、GPU 越快，这些固定开销与串行开销的占比就越大。
端到端锚点：Qwen3-8B / A100 上 vLLM、SGLang 约 **14.5 ms/token**，MPK **12.5 ms/token**，
而按「16 GB 参数 ÷ 1.6 TB/s 带宽」估算的硬件下界约 **10 ms**（快照行 1513）。

> **一句话**：MegaKernel 的思路，是把「以整个算子为单位、靠 host 反复 launch 推进」，
> 换成「以 SM 上的 task 为单位、在 device 内事件驱动地推进」。MPK §3 的 tGraph 表示与 §5 的
> in-kernel runtime，做的就是这件事（后续 P2 展开）。

## 2. 已有缓解手段，以及它们停在哪

上面三笔账不是没人管。工业界和编译器社区有四类主流手段，此外还有一类通信-计算重叠的工作。
但它们各自只治一部分，且**都没有碰到「以 SM 为粒度的跨算子调度与重叠」这个根问题**。

下面逐条过，每条都给 MPK 快照的真实章节号/行号，并用本专题 P1-2 实测佐证
（`src/mitigations.cu`，原始输出 `harness/mitigations.out.txt`，目标卡 H100/sm90）。

### 2.1 算子融合（operator fusion）

**做法**：把同一份数据上**顺序执行**的多个 kernel 合并成一个语义等价的 kernel。
收益就是 §1.4 说的三件事，顺带把被融合算子之间的 kernel barrier 也消掉了（MPK §2.2，快照行 313）。

**现状**：PyTorch / JAX / TVM 用**规则启发式**融合相邻算子（快照行 314–321）；
Mirage / TASO 用**超优化**自动搜索融合规则（快照行 322–327）。

**边界**：只能融**局部的小算子组**。原因还是那句——「忠实实现整个复杂张量程序的单个 kernel」
在计算上极难、常常不可行（MPK §2.2，快照行 328）。所以融合是**手工/局部**的，粒度停在
「几个相邻算子」，跨不过整张图。

而 mega-kernel 范式把融合推到极致（整图 + 通信进一个持久化 kernel，用 device memory 同步原语
跨 SM 协调，快照行 329–331）；并且现有 mega-kernel 都是**手工**为特定模型打造的
（FlashDMoE、Spector 的 LLaMA-1B，快照行 332–344）。MPK 想用编译器自动化取代的，正是这部分手工活。

### 2.2 CUDA Graphs：把「开门」录下来重放

**做法**：把一串 GPU 操作**捕获**成图，之后**低开销重放**。它直接针对第二笔账（launch 开销）。

MPK 承认它是当前系统压 launch 开销的主要手段，但明确指出两个根本局限（MPK §1，快照行 87–89）：

1. **本质上是静态的**。控制流、张量形状或数据依赖一变，就必须**重新实例化或修改**已捕获的图，
   对推理中常见的动态负载不够灵活（快照行 89）。
2. **仍有 kernel-switch 开销**。论文 §6.3 指出即便用了 CUDA Graphs，kernel-per-operator 依然要付
   这笔钱，而且它正是小模型 + 新 GPU 上更吃亏的三项开销之一（快照行 1504–1509）。

更关键的是，CUDA Graph **不改变串行语义**：图里相邻节点之间仍然是 kernel barrier。
所以第一笔账（跨算子软件流水）和第三笔账（中间张量物化）原样保留。
数字上也印证了这点（§6.6，快照行 1618–1628）：graph 把单次 launch 从 3.8 µs 降到 0.8 µs、
总量从 1.1 ms 降到 0.2 ms，但**不归零**。

> 一句话：CUDA Graph 让「开门」变便宜了，但门还是得一道一道开。

### 2.3 PDL：让相邻两个 kernel 探个头

**做法**：允许**同一条 stream** 上相邻 kernel **部分重叠**——下游 kernel 可以先跑它的「前导」部分，
等上游数据就绪后再去读（MPK §1，快照行 78–84）。这本来能缓解第一笔账。

**边界**：MPK 指出采用 PDL 需要**显著的工程改造**，因为它**从根本上改变了 kernel 结构与控制流**
（快照行 84）。换句话说，它不是「加个属性就好」，而是要求每一对上下游 kernel 都按 PDL 语义重写：
producer 要主动放行，consumer 要显式等待依赖。这也是 P0-3 探测里 PDL 能跑通、但生产模型难以
自动套用的原因（`docs/04-experiments.md` §0）。

### 2.4 多流：只并行「没关系」的算子

**做法**：把**互不依赖**的算子放到不同 stream 上并行。

**边界**一句话就能说完：**多流只能并行「无依赖」的算子；有数据依赖的算子，分到几个 stream 上都得串行。**
根因还是依赖只在**整个算子**粒度上表达——运行时表达不了「AllReduce 的第 i 个 fragment 只依赖
MatMul 的第 i 个输出」这种细粒度关系，于是只能等整个上游算子完成（MPK §1，快照行 85–86）。
此外 stream 级调度仍是运行期**粗粒度**决策，做不了 SM 级负载均衡。

**P1-2 实测**（`src/mitigations.cu`，64 个自旋约 17 µs 的小 kernel，目标卡 H100/sm90，30 轮均值）：

| 对照 | 时间（每次 64 kernel） | 每 kernel | vs A |
|---|---|---|---|
| A 无依赖，1 stream | 1.076 ms | 16.81 µs | 1.00× |
| B 无依赖，4 streams | 0.295 ms | 4.61 µs | **3.65×（有效并行）** |
| C 有依赖链，4 streams（event 串接） | 1.125 ms | 17.57 µs | **0.96×（并行失效）** |
| D 有依赖链，CUDA Graph | 1.025 ms | 16.01 µs | 1.05× |

怎么读这张表：

- **B**：对无依赖算子，多流确实接近线性加速（3.65×；剩下的损耗来自 launch 与共享 SM 争用）。
- **C ≈ A**：只要引入依赖，多流立刻退化成串行，和单 stream 几乎没差别。
- **D** 只比 C 略低：说明 **CUDA Graph 只降 launch 开销，并不打破依赖串行**。

这三条正好在真实硬件上复现了 §1 的论断：多流和 Graph 都停在「算子粒度 + 静态」这一层。

### 2.5 通信-计算 overlap：在 kernel 边界处做重叠

还有一条主流路线：把**通信**与**计算**在 kernel 边界处重叠起来。
DeepEP 提供专家并行的 all-to-all kernel；Triton-Distributed 让 Triton 能表达重叠的分布式 kernel；
TokenWeave / FlashOverlap / ParallelKittens 把通信与依赖它的计算 fuse 或 co-schedule；
Google 的早期工作则把集合通信算子**分解**，好让它与依赖计算重叠（MPK §8，快照行 1878–1917）。

**边界**：这些系统都在**单个 kernel 或单个集合算子**的粒度上实现重叠。
MPK 的差别是把通信与计算**统一表示成同一张 tGraph 里的 task**，用同一套 in-kernel runtime 调度，
于是重叠从全局 task 调度中**自然涌现**，并且同时适用于张量并行集合通信与专家并行 all-to-all
（快照行 1910–1917）。

（与 `mega-moe` 专题聚焦的 DeepEP / FlashMLA 双 kernel 方案交叉引用，本文不重复展开。）

### 2.6 降低手工融合难度的抽象工具

还有一层工作是在「抽象」上省力：ThunderKittens 提供 tile 级抽象，PipeThreader 把软件流水的
kernel 执行暴露给程序员。但它们仍然要求开发者**自己决定融哪些算子、怎么流水、怎么协调**。

MPK 认为这类抽象与自己是**互补**的——可以直接拿来实现 MPK runtime 里的单个 task kernel
（MPK §8，快照行 1864–1872）。另外 TileRT 把算子分解成细粒度 tile 级 task 塞进单个持久化 kernel，
核心思路与 MPK 一致（快照行 1873–1877）。

### 2.7 小结：五种手段，各有各的天花板

| 手段 | 治哪笔账 | 粒度 | 为什么仍不够 |
|---|---|---|---|
| 算子融合（fuse） | 中间张量往返 + launch + 局部 barrier | 几个相邻算子（手工/局部） | 只能融局部小算子组，覆盖不了整图（快照行 328） |
| CUDA Graphs | launch 开销 | 整图（静态） | 仍有 kernel barrier 与 kernel-switch 开销；动态负载要重捕（行 87–89、1504–1509） |
| PDL | 相邻 kernel 部分重叠（第一笔账） | kernel 对（需重写） | 根本改变 kernel 结构与控制流，改造成本高（行 78–84） |
| 多流 | 无依赖算子并行 | 算子/stream（粗） | 有依赖即串行；依赖只在算子粒度表示（行 85–86；P1-2 实测 C≈A） |
| 通信-计算 overlap | 隐藏通信延迟 | 单 kernel / 单集合算子 | 非统一调度；MPK 用同一 tGraph + runtime 让重叠自然涌现（行 1910–1917） |

**结论**：这些手段要么停在**算子粒度**（融合、多流），要么停在**静态图 / 相邻 kernel**
（CUDA Graphs、PDL），都没有触及这个根问题：**以 SM 为粒度的跨算子调度与重叠**。
把依赖细化到 SM task、在 device 内动态事件驱动地调度——这正是 MPK §3 的 tGraph 与 §5 的
in-kernel runtime 要做的事（后续 P2 展开）。

## 3. MegaKernel 的定义与判据

### 3.1 定义

MPK 把 mega-kernel 与 **persistent kernel（持久化内核）** 当作同义词（MPK §1，快照行 90–95）：
把模型推理的**全部计算与通信**融合进**一个** kernel——「一次 launch 执行整个模型（含逐层计算与
跨 GPU 通信），中途不中断」。

据此给出本专题采用的定义：

> **MegaKernel（巨型内核 / persistent mega-kernel）**：把（一整段甚至整个模型的）多个算子
> **融合进一个持久化 kernel**——kernel 一次 launch、在 GPU 上长期驻留（persistent），
> 内部用**去中心化的 in-kernel 调度**把各算子的工作分给不同 SM / warp，并在 kernel 内实现
> **跨算子的软件流水**与**通信-计算重叠**。

四个关键词：

- **persistent**——一个 kernel 跑完所有任务；
- **单 kernel**——消除 kernel barrier；
- **SM 级调度**——粒度细到 SM；
- **in-kernel 运行时**——调度在 device 上做，不靠 host 反复 launch。

这里有个容易误会的地方：**「mega」说的是融合范围**（整段/整图），不是 kernel 的代码有多长、
grid 有多大。**一个只覆盖单算子的 persistent kernel，并不因此成为 mega-kernel。**

MPK 把 mega-kernel 相对 kernel-per-operator 的好处归纳为**三点**（MPK §1，快照行 96–101），
正好与 §1 的三笔账一一对应：

1. **消除重复 launch 开销**——单次 kernel 调用取代多次算子级 launch（← §1.3 第二笔账）；
2. **跨算子软件流水**——全融进一个 kernel 后，下一算子的数据可以在当前算子**还没算完**时预取
   （← §1.2 第一笔账，对应 MPK §2.1 Figure 2）；
3. **计算-通信细粒度重叠**——同一 kernel 内让计算与跨 GPU 通信并发，更有效地隐藏通信延迟
   （← §1.2 中的 fine-grained overlap，对应 MPK §2.1 Figure 3）。

### 3.2 五条判据：什么才算 MegaKernel

把 §2 的缓解手段与上面的定义一对照，可以提炼出**五条判据**，用来区分「真 mega-kernel」和
「只是压了 launch / 做了局部融合」。五条都来自 §3.1 定义里可验证的特征：

| 判据 | 含义 | 不满足的代表 |
|---|---|---|
| **C1 单次 launch** | 整段/整图只调用一次 kernel | kernel-per-operator、CUDA Graph（图内仍是多次 kernel 调用） |
| **C2 常驻执行（persistent）** | kernel 在整段执行期间持续驻留，SM 不因 kernel 边界而清空 | 普通多 kernel 串行 |
| **C3 多算子融合范围** | 覆盖**跨算子的依赖链**，而不是单个/几个相邻算子 | 单算子 persistent kernel、局部融合 |
| **C4 设备内协调** | 用 **device memory 上的同步原语**（信号 / ring buffer / 原子）跨 SM 推进，而不是 host 逐段 launch | 多流、PDL（host 侧仍按 kernel 推进） |
| **C5 跨算子/通信调度** | 调度粒度细到 **SM / 线程块级 task**，能表达跨算子流水与通信-计算重叠 | CUDA Graph / PDL（粒度停在整算子或 kernel 对） |

对照结论（与 §2.7 一致）：

- 算子融合满足 C3，但停在局部，且仍是多 kernel（不满足 C1/C2/C4）；
- CUDA Graphs 满足 C1（一个 graph），但在**语义上仍是多次 kernel 调用**（不满足 C2/C3/C4）；
- PDL、多流同样不满足 C2/C4。

**只有同时满足 C1–C5，才是本专题意义上的 MegaKernel。** 分水岭就在 **C4/C5**：
调度权从 host 移到 device，调度粒度从算子细到 SM——也就是 §1.5 说的那个「根问题」。

### 3.3 与相邻术语的澄清

社区里 “mega-kernel / megakernel / mega kernel / persistent kernel” 经常混用。
本专题统一写 **MegaKernel**，相邻术语按下面的边界使用：

- **persistent kernel**：只强调「一次 launch、长期驻留」（C1+C2）。它可以只承载单算子
  （比如 persistent GEMM、常驻采样器），这时**不是** mega-kernel。MPK 之所以把两者视为同义，
  是因为它的 mega-kernel 总是**整图**常驻（行 90–95），C3 也一并满足。
- **cooperative kernel / cooperative groups**：用 grid-wide 同步实现跨 SM 协调，是满足 C4 的
  **手段之一**，但本身不决定融合范围（C3）。P0-3 的 `probe_env.cu` 正是用 cooperative grid sync
  探测设备内同步的可行性（`docs/04-experiments.md` §0）。
- **operator fusion（算子融合）**：mega-kernel 的**必要不充分条件**——mega-kernel 一定融合（C3），
  但局部融合（只融相邻几个算子）不满足 C1/C2/C4/C5，属于谱系第 2 档（§4）。
- **CUDA Graph**：把多次 kernel 调用录成静态序列，满足「一次提交」，但**语义上仍是逐 kernel 执行**，
  依赖与 barrier 原样保留（P1-2 实测：依赖链 graph 仅 1.05×，见 §2.4）——不满足 C2/C3/C5。
- **PDL**：让**相邻两个 kernel** 部分重叠，仍以 kernel 为单位推进，不满足 C2/C4/C5（§2.3）。
- **TileRT**：把算子分解成细粒度 tile 级 task 放进**单个持久化 kernel**（MPK §8，快照行 1873–1877），
  与 C1–C5 一致，与 MPK 的差别是「手工/专用 vs 编译器自动」，不是范式之差。

## 4. 谱系：从单算子优化到整图单 kernel

### 4.1 四档谱系

按「融合范围」（C3）与「调度权归属 / 调度粒度」（C4/C5）这两条轴，可以把相关工作排成四档：

| 档 | 融合范围 | 推进方式 | 调度粒度 | 代表工作（MPK 快照行号） |
|---|---|---|---|---|
| **T1 单算子内优化** | 单算子 | 单 kernel 内 software pipelining / warp specialization | warp | FlashAttention / FasterTransformer / FlashInfer（§8 行 1783–1798）、DeepGEMM GEMM |
| **T2 局部/段融合** | 相邻几个算子 | 多 kernel 或单 kernel，host 启动 | 算子/kernel | 规则启发式：PyTorch / JAX / TVM（§2.2 行 314–321）；超优化：Mirage / TASO（行 322–327）；fused MoE、fused AllReduce+RMSNorm（`hpc-ops`） |
| **T3 通信-计算重叠** | 通信 + 依赖计算 | host 协调的多 kernel 重叠 | 单 kernel / 单集合算子 | DeepEP（§8 行 1880–1885）、Triton-Distributed（行 1886–1890）、TokenWeave / FlashOverlap / ParallelKittens（行 1891–1903）、Google 早期集合算子分解（行 1906–1909） |
| **T4 MegaKernel（整段/整图）** | 整图（含通信） | **单次 launch + 设备内 in-kernel 调度** | SM / 线程块级 task | 手工：FlashDMoE（§2.2 行 333–336、§8 行 1851–1854）、Spector LLaMA-1B（§2.2 行 337–343、§8 行 1855–1861）、TileRT（§8 行 1873–1877）；自动：**MPK** |

**怎么读这张表**：T1→T4 的演进**不是「kernel 越来越大」**，而是两件事同时发生——
融合范围从单算子扩到整图，**调度权从 host 移到 device、调度粒度从算子细到 SM**。
T2/T3 都还在 host 逐 kernel 推进（不满足 C4），只有 T4 把「推进」也搬进了 device。

### 4.2 T4 内部：手工 vs 自动

T4 这一档内部还有一条关键分野（MPK §2.2 行 332–344、§8 行 1850–1863）：

- **手工 mega-kernel**：FlashDMoE（把 MoE 计算 + 跨 GPU 通信融进单 kernel，§2.2 行 333–336）、
  Spector 等人为 LLaMA-1B 手写的低延迟 mega-kernel（行 337–343）。MPK 对这类做法的评价很直接——
  「需要大量工程投入与深厚的 GPU 经验，且**不能泛化到别的模型或 GPU**」（§8 行 1862）。
- **编译器自动 mega-kernel**：**MPK / Mirage Persistent Kernel**——首个把多 GPU 模型推理
  **自动**编译成单个 mega-kernel 的编译器 + 运行时（Abstract 行 38、§1 行 133–138），
  SM 级 tGraph + 去中心化 in-kernel runtime，相对 vLLM/SGLang 最高 **1.7×** 更低延迟、
  TP 下最高 **10×** 吞吐（§6.3/§6.5）。
- **互补的抽象层工作**：ThunderKittens（tile 抽象）、PipeThreader（显式软件流水）仍要开发者
  自己决定融什么、怎么流水（§8 行 1864–1872）；MPK 视其为**互补**——可以作为 MPK runtime 里
  单个 task kernel 的实现手段。

### 4.3 生产落地：段级 mega-kernel

T4 在工业界的落地，目前集中在**段级**而不是整图。

典型代表是 **DeepGEMM Mega MoE / Mega Gate / Mega mHC**（`~/github/DeepGEMM` @ `057ca59`）：
把一段（MoE 段 / 路由打分段 / HyperConnection 段）内的多个算子融进一个持久化 kernel，
用**设备内信号 + ring buffer**（满足 C1/C2/C4）驱动固定的 warp/CTA 角色（C4/C5），
重叠 NVLink 通信与 Tensor Core。

它们与 MPK 的差别有两条：**「手工静态角色」 vs 「编译器生成 + 去中心化动态调度」**，
以及**融合范围停在段级**（C3 只到一段）。源码坐标见 `docs/02-production-cases.md`
（与 `../mega-moe/` 专题交叉引用，本文不重复实现细节）。

另外，**Kimi K3** 报告把「fused mega-kernels」列为内核优化任务的一类（`docs/02` §3），
说明这个范式已经进入模型能力训练的产业视野。

## 5. 这个专题怎么读

- 先读 **MPK 论文精读**（`docs/01-mpk-paper.md`）：SM 级图表示、编译器、in-kernel runtime、评估。
- 再看**生产案例**（`docs/02-production-cases.md`）：DeepGEMM mega 家族、Kimi K3。
- 然后做**相关技术对照**（`docs/03-related-tech.md`）：CUDA Graphs / PDL / persistent kernel。
- 最后是**实验**（`docs/04`）与**一页纸总结**（`docs/05`）。

> 术语提醒：社区里 “mega-kernel / megakernel / mega kernel” 常混用；本专题统一用 **MegaKernel**，
> 指「整段/整图融合 + 持久化 + SM 级调度」这一类，而不是仅仅「两个算子融合」。

## 6. 附录：MPK 论文导航（速查表）

下面这张地图基于本机快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
（arXiv:2512.22219v2，共 2717 行）逐节核对得到，精读时按图索骥即可。

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

1. **编译层**：tensor program →（分块 → 细粒度依赖分析 → event fusion → normalization → linearization
   → 静态 tGraph）→（按 thread block 做 superoptimization）→ 每个 task 的 CUDA device function。
2. **运行层**：一个持久化 mega-kernel 内，SM 固定划分为 **schedulers**（每 SM 4 个 scheduler warp，
   维护 event queue）与 **workers**（每 SM 一个 worker，维护 JIT/AOT 双 task queue）；
   事件被激活 → 进 scheduler 队列 → 派发 task → task 完成 → 通知触发事件，如此循环，直到整图跑完。
