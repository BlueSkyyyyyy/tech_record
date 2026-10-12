# MegaKernel 相关技术对照

> 本文把 MegaKernel 与四类最相关的既有技术逐一对齐：**CUDA Graphs**（P4-1）、**PDL**（P4-2）、
> **persistent kernel / in-kernel 调度**（P4-3）、**通信-计算 overlap**（P4-4）。目标是回答同一个问题：
> 这些技术**各自消除了什么、没消除什么**，以及它们距离「真 mega-kernel」还差哪一条判据。
>
> 判据沿用 `docs/00-megakernel-overview.md` §3.2 的五条（C1 单次 launch / C2 常驻 / C3 多算子融合范围 /
> C4 设备内协调 / C5 SM 级调度）。引用 MPK 论文快照
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt` 的真实行号。
> 交叉引用 `docs/00` §2（缓解手段及其边界），本文在其基础上**展开到机理与对照表**，不重复其结论。

## 1. CUDA Graphs（P4-1）

### 1.1 机理：capture → instantiate → replay

CUDA Graphs 的核心是**把一串 GPU 操作录成一张静态图**，之后用**一次提交**重放（replay），
从而把「host 为每个 kernel 各做一次 launch」压成「host 做一次图提交」。标准三步（CUDA 10.1 引入，
后由 CUDA 12.x 扩展）：

1. **capture**：`cudaStreamBeginCapture(stream, mode)` 开始录制，正常 launch kernel/memcpy，再
   `cudaStreamEndCapture(stream, &graph)` 结束，得到 `cudaGraph_t`。捕获有全局/线程本地/宽松三种模式
   （`cudaStreamCaptureModeGlobal` / `ThreadLocal` / `Relaxed`），模式决定捕获期间哪些跨线程操作会报错。
2. **instantiate**：`cudaGraphInstantiate(&exec, graph, ...)` 生成可执行的 `cudaGraphExec_t`，
   一次性预初始化所有 kernel 的 work descriptor。
3. **replay**：每次迭代 `cudaGraphLaunch(exec, stream)`——**一次 host 调用提交整张图**；
   图只在第一次 capture+instantiate，后续复用同一 `exec`。

除流捕获外，CUDA 也提供**显式构图 API**（`cudaGraphAddKernelNode` / `cudaGraphAddMemcpyNode` /
`cudaGraphAddDependencies` …），图节点不限于 kernel，还包括 memcpy、memset、host 函数、
**子图（child graph）** 与 event 节点，并可跨多个 stream、多张 GPU。

关键点：**capture 录下的是「哪些 stream 上按什么顺序发了什么操作」，即依赖结构**。同一 stream 上的
连续 kernel 在图中仍是前后依赖的节点；跨 stream 用 event 建立的依赖也会被正确录成图的边。所以
CUDA Graph 是**对已有 kernel 级依赖的忠实重放**，它优化的是**提交路径（host 开销）**，不是执行语义。

> 真实资料：NVIDIA *Getting Started with CUDA Graphs*（Alan Gray, 2019）
> https://developer.nvidia.com/blog/cuda-graphs/ ——该文实测 V100 上单个 2.9 µs 的短 kernel：
> 逐 launch 且每次同步 **9.6 µs/kernel**，把同步移出内层循环让 launch 与执行重叠后 **3.8 µs/kernel**，
> 改用 CUDA Graph 后 **3.4 µs/kernel**；并指出 graph 的 create+instantiate 约 400 µs 一次性成本，
> 复用足够多次后摊销到近 0。这与本专题的结论一致：graph 治的是**每次 launch 的固定开销**。

### 1.2 它消除了什么：host 侧 launch 开销（目标卡 H100/sm90 实测）

实验 `src/cuda_graph.cu`（原始输出 `harness/cuda_graph.out.txt`）：把 N 个近乎空的 kernel
（`<<<1,32>>>`，单 stream 顺序 launch，天然依赖链）分别用 (a) eager 逐 launch、(b) 捕获成 graph 后
`cudaGraphLaunch` 重放，测每次迭代耗时（iters=200，目标卡 H100/sm90）：

| N（节点数） | eager ms/迭代 | graph ms/迭代 | eager ns/kernel | graph ns/kernel | 加速 |
|---|---|---|---|---|---|
| 4 | 0.0075 | 0.0046 | 1872 | 1157 | 1.62× |
| 16 | 0.0294 | 0.0137 | 1836 | 856 | 2.15× |
| 64 | 0.1094 | 0.0512 | 1710 | 800 | 2.14× |
| 256 | 0.4404 | 0.2015 | 1720 | 787 | 2.19× |
| 1024 | 2.3904 | 0.8024 | 2334 | 784 | 2.98× |

一次性与固定成本：

| 项 | 值 |
|---|---|
| 单节点 graph 重放固定成本（`cudaGraphLaunch` 的 host 侧基线） | **3.13 µs/次** |
| 64 节点 graph 的 capture + instantiate（一次性） | **103.6 µs/次** |

读法：

- **eager 的 per-kernel 成本基本恒定（~1.7–2.3 µs），与 N 近似无关**——这正是「host 每发一个 kernel
  都要付一次固定 launch 开销」，于是 N 越大总开销线性越大。
- **graph 的 per-kernel 成本随 N 增大被摊薄到 ~0.78 µs**：因为整图只做**一次** `cudaGraphLaunch`
  （固定约 3.1 µs，见上表），N 越大这笔固定成本占比越小，剩下的 ~0.78 µs/kernel 是 **device 侧的
  kernel 派发/切换开销**（图内节点仍逐个被 GPU 启动）。
- 数量级与 MPK 论文吻合：论文在 B200 上给出 kernel-per-operator 单次 launch **3.8 µs（eager）**、
  **0.8 µs（CUDA Graph）**，总量 1.1 ms → 0.2 ms（快照 §6.6 行 1618–1628）。本机 H100 的 eager ~1.7 µs、
  graph ~0.78 µs/kernel 与之同量级（B200 更快、eager host 开销反而更高，故论文的 3.8 µs 偏大）。
- **注意 graph 的 ~0.78 µs/kernel 并不归零**——这正是 MPK 说的「即便用 CUDA Graphs，kernel-per-operator
  仍有 **kernel-switch 开销**」（快照 §6.3 行 1504–1509）。它由 device 启动每个图节点的时间决定，是
  **执行语义层面**的残留，不是 host 开销。

### 1.3 它没有消除什么

**(1) kernel barrier 与依赖串行原样保留。** CUDA Graph 只是把「多次 launch」变成「一次提交」，
图中相邻节点之间仍由 **kernel barrier** 串起来，损失一（跨算子软件流水）与损失三（中间张量物化）
不受影响。本专题 P1-2 实测的直接证据（`harness/mitigations.out.txt`）：64 个约 17 µs kernel 的
**依赖链**，逐 launch 1.076 ms、4 流 event 串接 1.125 ms、**捕获成 graph 后 1.025 ms（仅 1.05×）**——
graph 只降 launch、**不破依赖串行**。

**(2) 图是静态的。** MPK 明确指出 CUDA Graphs「largely static：控制流、张量形状或数据依赖一旦变化，
就必须重新实例化 / 修改已捕获的图」，对推理中常见的动态负载不灵活（快照 §1 行 87–89）。本机也印证了
「建图有成本」：一次 capture+instantiate（64 节点）就要约 **104 µs**，形状一变就得重来；
而 MPK 的一条局限正是「**每个 batch size 一张静态图**」（`docs/01-mpk-paper.md` §5.8）。

**(3)「device graph launch」也没改变粒度。** CUDA 12.0 起支持**从 device 端 launch 图**
（device graph launch，动态控制流），允许 kernel 内部按数据决定下一段图——这让「图」能承载一定动态性。
但它仍以**图/kernel 为推进单位**，节点之间的依赖依旧是 kernel 级；它没有把粒度下沉到 SM task，
也没有把调度做成 device 内事件驱动的 worker/scheduler 模型。这与 MPK 的 tGraph（**SM 级 task / 子 kernel
event**）有本质区别（见 §1.4 对照）。

### 1.4 与 MegaKernel 的关系：判据映射

MPK 论文专门给了 tGraph 与 CUDA Graphs 的对照（快照 §3 行 402–414）：二者**结构上相似**——都是静态
实例化、显式编码操作间依赖；但 **CUDA Graphs 只在 kernel 级捕获依赖**，而 tGraph 细化到**单个 SM task
与子 kernel event**。CUDA Graphs 主要描述**kernel 的 launch 顺序、靠 stream 语义同步**，于是重叠与融合
被限制在 **kernel 边界**；tGraph 显式建模**算子内与跨算子**依赖，能在**单 kernel 内**做到 SM 间细粒度
同步与计算-通信并发。

用 §3.2 的判据看 CUDA Graphs：

| 判据 | CUDA Graph 是否满足 | 说明 |
|---|---|---|
| C1 单次 launch | **部分**（一次图提交，但语义上仍是多次 kernel 调用） | 一个 `cudaGraphLaunch`，图内 N 个 kernel 各被启动一次 |
| C2 常驻（persistent） | 否 | 节点之间 GPU 会 drain，无法跨节点常驻软件流水 |
| C3 多算子融合范围 | 否 | 不融合算子，只重放 kernel 序列 |
| C4 设备内协调 | 否（device graph launch 仅部分） | 依赖靠 stream / 图边，仍以 kernel 为推进单位 |
| C5 SM 级调度 | 否 | 粒度停在 kernel 级，无法表达「第 i 个 AllReduce fragment 只依赖第 i 个 MatMul tile」 |

**结论**：CUDA Graphs 精准地治了**损失二（launch 开销）**，把 host 侧成本从「每 kernel 一次」压到
「每图一次」，是本专题实测中**收益最确定的单一手段**（1024 节点仍 2.98×）。但它**不满足 C2–C5**：
不改变依赖串行（损失一）、不消除中间张量物化（损失三）、粒度停在 kernel 级。它是 mega-kernel 的
**降级替身**而非实现路径——两者共享「静态实例化 + 显式依赖」的思想，分水岭正是 MPK 所说：
**依赖表示与调度粒度能否下沉到 SM/task 级**（快照行 408–414）。

### 1.5 小结

| 维度 | CUDA Graphs |
|---|---|
| 治哪类损失 | 损失二（launch 开销），host 侧从每 kernel 一次 → 每图一次 |
| 本机实测（H100） | 1024 节点 2.98×；graph per-kernel 从 ~1.7–2.3 µs 降到 ~0.78 µs（不归零） |
| 固定成本 | `cudaGraphLaunch` ~3.1 µs/次；capture+instantiate ~104 µs（一次性） |
| 不改变 | kernel barrier / 依赖串行；中间张量物化；图静态（形状/控制流变要重捕） |
| 判据 | 仅部分满足 C1，不满足 C2–C5 |
| 与 MPK 的关系 | 同「静态实例化 + 显式依赖」思想；分水岭是依赖/调度粒度是否到 SM 级（MPK §3 行 402–414） |

> 与 `docs/00` §2.2 的分工：§2.2 只给「静态 + 残留 kernel-switch」的结论；本节补上**机理、API 三步、
> 本机量化与判据映射**。

---

## 2. PDL（Programmatic Dependent Launch，P4-2）

### 2.1 机理：让同 stream 的相邻 kernel「部分重叠」

CUDA 默认在同一 stream 的两个 kernel 之间插一道隐式 **kernel barrier**：上游所有线程退出后，
下游才被硬件启动。**PDL**（Hopper sm90 起，CUDA 11.8+）允许下游 kernel **提前被启动**：它的
block 可以先执行自己的「前导（prologue）」代码（算地址、预热、做与上游无关的准备工作），
再显式地**等待上游依赖**后才读上游写的数据。三个关键原语：

- **下游侧属性**：用 `cudaLaunchAttributeProgrammaticStreamSerialization`
  （`cudaLaunchKernelEx`，值 `programmaticStreamSerializationAllowed=1`）声明「这个 kernel 接受
  提前启动、由程序自己管依赖」。
- **上游侧放行**：上游 kernel 调用 `cudaTriggerProgrammaticLaunchCompletion()`，表示「我的关键部分
  已完成，下游可以越过 grid 依赖先行」。当上游**所有 block** 都触发（或上游整体结束）后，该触发生效。
- **下游侧等待**：下游在真正读上游数据前调用 `cudaGridDependencySynchronize()`，阻塞到上游放行/完成。

换言之，PDL 把「整 kernel 的隐式 barrier」细化为「**显式的、可提前启动的 kernel 对**」：下游不必
空等上游最后一段，只要它自己有「可提前做的活」，这段活就能与上游尾部重叠。MPK 论文对此的定位是：
PDL「允许同 stream 上 kernel 间部分重叠」，但「采用 PDL 需要显著工程改造，因为它从根本上改变 kernel
结构与控制流」——**每个相关 kernel 都要按 PDL 语义重写**（上游主动放行、下游显式等待）。（快照 §1 行 78–84）

> 真实资料：NVIDIA *CUDA C Programming Guide* 的 Programmatic Dependent Launch 一节
> https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
> （即 MPK 论文的引用 [26]，见快照行 2242–2244）。它定义了上述三个原语与「触发在全体 block 都调用
> `cudaTriggerProgrammaticLaunchCompletion` 或上游 kernel 退出时生效」的语义。

### 2.2 它重叠了什么：下游 prologue 与上游 tail 的重叠（目标卡 H100/sm90 实测）

实验 `src/pdl.cu`（原始输出 `harness/pdl.out.txt`）。模型：上游 `producer` = `pre` 段 → 写数据 →
`cudaTriggerProgrammaticLaunchCompletion()` 放行 → `tail` 段；下游 `consumer` = `prologue` 段
（不碰上游数据）→ `cudaGridDependencySynchronize()` → 读数据。固定 `pre=tail=200000` 周期，
**对每个 prologue 值同时测「无 PDL」与「有 PDL」**（工作量相同，公平对照），iters=500：

| prologue（cycles） | no-PDL ms/it | PDL ms/it | 加速 | 理论可省 overlap |
|---|---|---|---|---|
| 0 | 0.2045 | 0.2033 | 1.01× | 0 |
| 100000 | 0.2573 | 0.2035 | 1.26× | 100000 |
| 200000 | 0.3075 | 0.2036 | 1.51× | 200000 |
| 400000 | 0.4052 | 0.3039 | 1.33× | 200000（被 tail 封顶）|

读法（与 `no-PDL: pre+tail+prologue`、`PDL: pre+max(tail,prologue)` 的模型完全吻合）：

- **prologue=0 时加速 1.01×**——下游没有可提前做的活，PDL 等于没用。这是最重要的一条：
  **PDL 不是万能加速，它只重叠「下游本可提前完成的那部分」**。
- **prologue 增大，收益增大，但被上游 tail 封顶**：`min(tail, prologue)` 就是能省下的时间。
  prologue=200000 省满 200000 周期（1.51×）；prologue=400000 时 overlap 仍只有 200000（1.33×，
  因为上游 tail 只有 200000）。
- 数量级核对：`pre=200000` 周期在该卡约 0.104 ms（由 prologue=0 的 PDL 单轮 0.2033 ms ≈ pre+tail
  反推），与 1.8 GHz 附近 SM clock 一致；A/B 差值 0.104 ms ≈ 200000 周期，验证「重叠量 = min(tail,prologue)」。

对照 CUDA Graph（§1）：graph 治的是**损失二（launch 开销）**，不改变依赖串行（依赖链仅 1.05×）；
PDL 治的是**损失一的一个特例**——它确实能让**相邻两个 kernel** 在时间上重叠，但前提是下游有 prologue、
且上下游都按 PDL 重写。两者是**互补**的两种降级手段。

### 2.3 它没有消除什么

**(1) 粒度仍是「kernel 对」。** PDL 只在**相邻 kernel**之间做重叠，重叠的边界是「下游能提前跑多少
prologue」。它无法表达 MPK §3 的 tGraph 那种「第 i 个 AllReduce fragment 只依赖第 i 个 MatMul tile」
的**细粒度跨算子依赖**（快照 §2.1 行 241–293、§3 行 371–386）——PDL 里下游要么整体被放行、要么整体
等待，没有 fragment/tile 级的 event。

**(2) 需要逐 kernel 改造，且上游必须显式放行。** 每个 kernel 都要被改造为「prologue / 依赖点 /
epilogue」结构，上游还要插入 trigger、下游插入 grid-dependency-sync。这正是 MPK 说的「显著工程改造、
根本上改变 kernel 结构与控制流」（快照行 84）。对含数百个算子的模型，靠人工逐对改造不现实——
这也是 MPK 选择「编译器自动生成单 mega-kernel」而非「PDL 化整图」的原因（`docs/00` §2.3）。

**(3) 仍是 host 逐对 launch。** PDL 不减少 launch 次数（损失二不受影响），也不消除中间张量物化
（损失三）。它与 CUDA Graphs 可叠加，但即便如此，粒度和依赖表示都停在 kernel 级。

### 2.4 与 MegaKernel 的关系：判据映射

| 判据 | PDL 是否满足 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | 仍是 host 逐个 launch，只是相邻 kernel 可重叠启动 |
| C2 常驻（persistent） | 否 | 每个 kernel 仍会结束、GPU 会 drain |
| C3 多算子融合范围 | 否 | 不融合算子，只是让相邻 kernel 部分重叠 |
| C4 设备内协调 | 否 | 依赖由 host 插入的 trigger/sync 管理，仍以 kernel 为推进单位 |
| C5 SM 级调度 | 否 | 粒度停在 kernel 对，无法表达 tile/fragment 级依赖 |

**结论**：PDL 是 kernel-per-operator 下**缓解损失一（kernel barrier / 依赖串行）的定向补丁**，
能把「下游 prologue」与「上游 tail」重叠掉 `min(tail, prologue)`，实测收益 1.0–1.5×（取决于下游
有多少可提前做的活）。但它是**逐 kernel 对**的局部优化，不满足 C1–C5——`docs/00` §3.2 给 mega-kernel
的判据里，PDL 与多流同列「host 侧仍按 kernel 推进」的降级项。真正摆脱 kernel barrier 的是
把调度下沉到 SM 级、用设备内事件驱动（P4-3）。

### 2.5 小结

| 维度 | PDL |
|---|---|
| 治哪类损失 | 损失一（kernel barrier）的**特例**：相邻 kernel 部分重叠 |
| 本机实测（H100） | prologue=0 → 1.01×；prologue=tail → 1.51×；重叠量 = min(tail, prologue) |
| 前提 | 上下游都按 PDL 重写；下游要有 prologue；上游要 trigger |
| 不改变 | kernel 级粒度、launch 次数、中间张量物化 |
| 判据 | 全不满足 C1–C5，是与多流同档的降级项 |
| 与 MPK 的关系 | MPK 视其为「需重写 kernel 结构」的局部手段（快照行 78–84），故走整图 mega-kernel 路线 |

> 与 `docs/00` §2.3 的分工：§2.3 只给「PDL 能跑通、生产难套用」的结论；本节补上**三个原语机理、
> 本机量化（重叠量 = min(tail, prologue)）与判据映射**。

---

<!-- 后续子项（P4-3 persistent vs mega + in-kernel 调度、P4-4 通信-计算 overlap）
     将依次补入本文件，见 ROADMAP「当前进度/下一步」。 -->
