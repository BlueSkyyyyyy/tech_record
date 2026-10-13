# MegaKernel 实验：环境探测与实测

> 本文是「MegaKernel 资料与实现分析」的实验篇。先记录 **P0-3 环境探测**的结果——
> 目标卡（H100 / sm90）上 persistent kernel、CUDA Graph、PDL 三项能力是否可编译可运行，
> 作为后续 P4/P5 实测的基线；后续 P5 的 launch 开销 / 两算子融合实验继续写在本文件。

## 0. 环境探测（P0-3）

### 0.1 目的

MegaKernel 的实测需要三块基础能力，先确认它们在本机容器 `kernel_lab` 上「能编译、能跑」：

1. **persistent kernel + cooperative grid sync**：单 kernel 常驻、跨 block 同步——P5「单持久化 kernel」的前提。
2. **CUDA Graph capture/replay**：消除 launch 开销——P4-1 / P5-1 的对照组。
3. **PDL（Programmatic Dependent Launch）**：让相邻 kernel 部分重叠——P4-2 / P5-3。

探测程序：`src/probe_env.cu`（自包含，单文件，含三项的编译与运行自检）。

### 0.2 硬件 / 工具链

| 项 | 值 |
|---|---|
| GPU | NVIDIA H100 80GB HBM3（8 卡，本探测用 1 卡） |
| compute capability | 9.0（**sm90**） |
| SM 数 | 132 |
| shared mem / SM | 233,472 B（228 KB） |
| shared mem / block | 49,152 B（48 KB，默认） |
| 寄存器 / SM | 65,536 |
| 驱动 | 570.211.01 |
| nvcc | release 13.2, V13.2.78 |
| CUDA runtime | 13020 |

### 0.3 编译与运行

在容器内经 `scripts/lab.sh` 执行（绝对路径，workdir 为容器内 `/tmp`）：

```
scripts/lab.sh exec "nvcc -O3 -arch=sm_90 -rdc=true \
  /ssd/.../code/mega-kernel/src/probe_env.cu -o probe_env && ./probe_env 200"
```

原始输出见 `harness/probe_env.out.txt`。

### 0.4 结果

**(1) persistent kernel + cooperative grid sync —— 支持**

| 指标 | 值 |
|---|---|
| `cudaDevAttrCooperativeLaunch` | 1（YES） |
| `persistent_kernel` occupancy | 16 blocks/SM @128 threads |
| persistent kernel（grid=132，200 轮，每轮 grid sync） | **0.330 ms** |
| 对照：200 个 trivial kernel 串行 launch（无 sync） | **0.575 ms** |

一个常驻 kernel 内做 200 次 grid-wide 屏障（跨 132 个 block）总耗时 0.330 ms，
平均每次屏障约 1.65 µs；而 200 次独立 kernel launch 反而要 0.575 ms。
这说明 **cooperative grid sync 在本机可跑**，且在这些规模下跨 block 屏障比反复 launch 更省。
（注：这是能力探测而非公平的 launch 开销基准，量级仅供定位；正式的 launch 开销量在 P5-1。）

**(2) CUDA Graph capture/replay —— 支持**

| 指标 | 值 |
|---|---|
| 图中节点数 | 32 个小 kernel |
| plain：200 × 32 次串行 launch | 18.007 ms |
| graph：200 次 replay | 11.567 ms |
| 加速比 | **1.56×** |

用独立非默认 stream 以 `cudaStreamCaptureModeThreadLocal` 捕获（legacy default stream 在
global 模式下不可捕获，见下方「坑」）。32 个节点的图捕获后 replay，相对逐次 launch 约 **1.56×**
加速，印证 CUDA Graph 消除的正是 host 侧逐 kernel launch 开销。**但 graph 仍保留 kernel barrier**——
它不改变串行性（对照 `docs/03-related-tech.md` P4-1）。

**(3) PDL（Programmatic Dependent Launch）—— 支持**

| 项 | 结果 |
|---|---|
| precheck（sm90 + CUDA 13.2） | likely YES |
| `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` 启 producer | ok |
| 同属性启 consumer | ok |
| device sync | ok |
| 运行后 flag | producer=1, consumer=1 |

producer 用 `cudaTriggerProgrammaticLaunchCompletion()` 提前放行，consumer 用
`cudaGridDependencySynchronize()` 等待上游数据就绪，两个 kernel 都正常完成。
**PDL 在本机可编译可运行**，P5-3 可做真实的重叠度量。

### 0.5 结论与对后续实验的意义

| 能力 | 状态 | 支撑的子项 |
|---|---|---|
| persistent kernel + cooperative grid sync | 可编译可运行 | P5-2（两算子融进单持久化 kernel） |
| CUDA Graph capture/replay | 可编译可运行 | P4-1、P5-1（launch 开销对照） |
| PDL | 可编译可运行 | P4-2、P5-3（跨 kernel 重叠） |

三项均**实测通过**，无「仅资料分析」降级项。后续 P4/P5 的实测均可落地。

### 0.6 踩到的坑

- **legacy default stream 不能用 `cudaStreamCaptureModeGlobal` 捕获**：会报
  `operation not permitted when stream is capturing`。改用独立非默认 stream +
  `cudaStreamCaptureModeThreadLocal` 即正常。
- **CUDA 13 移除了 `cudaDeviceProp::cooperativeMultiDeviceLaunch`**：引用它会编译报错，
  多设备 cooperative 查询改用设备属性 API。

---

## 1. Launch 开销实测（P5-1）

### 1.1 问题

MPK §1 行 74–89 把「launch 开销」列为 kernel-per-operator 的三类损失之一。P4-1 已经用
CUDA Graph 量化了「graph 把 N 次逐 kernel launch 压成一次提交」，本节把对照补齐到第三种
策略——**单 persistent kernel**，回答：对 N 个独立等量的微小工作单元，三者的 host/设备侧
开销各是多少、persistent 相比 graph 又多省了什么。

### 1.2 方法

程序 `src/launch_overhead.cu`（目标卡 H100 / sm90，实测原始输出
`harness/launch_overhead.out.txt`）。对**完全相同的设备侧工作**（每单元一次 LCG 自旋 K 次
+ 一次 atomicAdd 计数）比较三种把 N 个单元跑到 GPU 上的策略：

| 策略 | 做法 | 每次迭代的 launch/调度 |
|---|---|---|
| **eager** | N 个独立 kernel 逐个 launch（同一 stream，天然串行依赖链） | N 次 host launch + N 次设备 grid 启动 |
| **graph** | 把上面 N 个 kernel 捕获成 CUDA Graph，一次 `cudaGraphLaunch` 重放 | 1 次 host 提交 + N 次设备 grid 启动 |
| **persistent** | 单个常驻 kernel，grid=#SM，每 block 用 grid-stride 循环消费 N 个单元 | 1 次 host launch + 1 次设备 grid 启动 |

Section A 固定每单元计算量 K=0（纯 launch 开销主导）扫 N；Section B 固定 N=256 扫 K，
并用两个 persistent 网格把「launch 开销」与「并行度」两个因素分开：
`persist1`（grid=1，单 block 串行跑完 N 个单元，设备侧工作与 eager 完全等价）与
`persistG`（grid=132，真 megakernel 并行度）。

### 1.3 Section A：纯 launch 开销（K=0）

| N | eager ms | graph ms | persist ms | eager ns/单元 | persist ns/单元 | eager/graph | eager/persist |
|---|---|---|---|---|---|---|---|
| 4 | 0.0075 | 0.0047 | 0.0020 | 1885 | 488 | 1.60× | 3.87× |
| 16 | 0.0294 | 0.0144 | 0.0019 | 1839 | 120 | 2.04× | 15.35× |
| 64 | 0.1123 | 0.0537 | 0.0020 | 1754 | 31 | 2.09× | 57.45× |
| 256 | 0.4490 | 0.2107 | 0.0020 | 1754 | 7.7 | 2.13× | 227.07× |
| 1024 | 1.7960 | 0.8381 | 0.0024 | 1754 | 2.3 | 2.14× | 752.35× |
| 4096 | 9.1736 | 3.4129 | 0.0046 | 2240 | 1.1 | 2.69× | 1987.83× |

读法：

- **eager 的 per-单元时间恒定在 ~1.75–2.2 µs**，与 N 基本无关——这就是 host 侧逐 kernel
  launch 的间隔（与 P4-1 测得的 eager per-kernel ~1.7–2.3 µs 一致）。总时间因此随 N 线性增长。
- **graph 把它压到 ~0.8 µs/单元（约 2.1×）**：重放只提交一次，但设备侧仍要启动 N 个 grid，
  且存在单次 `cudaGraphLaunch` 的固定延迟（P4-1 测得 ~3.1 µs），所以**不归零**。
- **persistent 总时间几乎恒定在 ~2 µs**（一次 launch 的全部开销），per-单元摊薄到接近 0，
  N=4096 时相对 eager 达 **1988×**。persistent 相比 graph 多省的正是「N 次设备侧 grid 启动」
  这一层——MPK 让 worker 在 kernel 内循环取任务（§5 行 1147–1174），正是要消掉它。

### 1.4 Section B：launch-bound → compute-bound 交叉（N=256）

| 每单元自旋 K | eager ms | graph ms | persist1 ms | persistG ms | eager/graph | eager/persist1 | eager/persistG |
|---|---|---|---|---|---|---|---|
| 0 | 0.6286 | 0.2109 | 0.0128 | 0.0026 | 2.98× | 48.92× | 239.46× |
| 64 | 0.6284 | 0.2398 | 0.0504 | 0.0025 | 2.62× | 12.46× | 251.79× |
| 256 | 0.6298 | 0.2817 | 0.0920 | 0.0026 | 2.24× | 6.84× | 239.61× |
| 1024 | 0.6966 | 0.4482 | 0.2583 | 0.0039 | 1.55× | 2.70× | 177.59× |
| 4096 | 1.3446 | 1.1141 | 0.9239 | 0.0091 | 1.21× | 1.46× | 147.05× |
| 16384 | 4.0047 | 3.7745 | 3.5846 | 0.0299 | 1.06× | 1.12× | 133.82× |

读法：

- **`persist1` 与 `eager` 随 K 增大而收敛**（48.9× → 1.12×）。二者设备侧工作等价（都是
  N 个单 block 串行自旋），唯一的差别是 eager 额外的 N 次 launch；K 小时它是主导、
  K 大时被计算淹没。这条曲线直接证明了「差额本来就是 launch 开销」。
- **`persistG` 保持领先**：K=16384 时仍有 133.8×，因为它用 132 个 block 把 N 个单元并行铺开，
  属于并行度收益（不是 megakernel 的 launch 收益本身，但说明 MPK 同时拿到两者）。
- **graph 也趋于 1.0×**：graph 只治 host 提交，不改变设备侧 N 次 grid 启动与串行依赖，
  所以计算主导时相对 eager 的优势同样消失（1.06×）。

### 1.5 结论与对 MegaKernel 的含义

1. **launch 开销确实只在 launch-bound 区间值得治**：单元越轻、越多，eager 越吃亏；一旦每单元
   计算足以淹没 ~1.8 µs 的 launch 间隔，三种策略趋同。
2. **graph 与 persistent 治的是不同层**：graph 消 host 侧 N 次提交（→ 1 次），persistent
   连设备侧 N 次 grid 启动也消掉（→ 1 次）。MPK 的整图 mega-kernel 属于后者，并且在
   kernel 内还能做跨算子软件流水（§5），这是 graph 给不了的。
3. **与 MPK 量化锚点一致**：MPK §6.6 报告 launch 占比从 293 次/token 降到 in-kernel
   scheduler 的 0.28%（§5.8 / §6.6，行 1586–1628），本节 Section A 的「eager 线性 vs
   persistent 恒定」正是同一现象的微观复现。
4. **诚实边界**：本节单元是「独立、无依赖」的，persistent 用 grid-stride 循环即可，无需
   grid barrier；真实 megakernel 的单元间有数据依赖，需要 event/barrier 协调（MPK §5.2、
   本专题 P4-3 §3.3），那部分开销本节未计入。K>0 的 persistG 收益主要来自并行度而非
   launch 消除，不能全部记在 megakernel 账上。

### 1.6 局限

- persistent 的 grid=#SM=132 固定，未扫网格大小；网格过小则并行度不足，过大则每 block 单元
  太少、调度摊薄。
- LCG 自旋是合成计算，不含访存/张量核，K 与真实算子的时间不能直接换算；此处只用它制造
  「计算主导」区间。
- 单卡、单 stream；多 stream / 多卡下 launch 开销与重叠行为另算（见 P4-2/P4-4）。

---

## 2. 两算子融合成单 persistent kernel（P5-2）

### 2.1 问题

P5-1 量化了 **损失二（launch 开销）**，并给出结论：它只在 launch-bound 区间值得治。
本节做 MPK 意义上的 **损失三（中间张量 HBM 往返）** 的最小示例——把两个有数据依赖的算子
融合进**单个 persistent kernel**，让中间结果只活在寄存器里、不落 HBM，并实测收益。

对应关系（`docs/00` §1 三类损失）：算子融合/单 kernel 同时触及损失二与损失三；本实验把
二者**分开度量**，避免把两类收益混为一谈。

### 2.2 方法

程序 `src/two_op_fusion.cu`（目标卡 H100 / sm90，原始输出 `harness/two_op_fusion.out.txt`）。
构造一对逐元素算子（float4 向量化）：

- `op1`：`y = f(x) = 1.5·x + 0.25`（producer）
- `op2`：`z = g(y) = sqrt(|y|) + 0.5·y`（consumer，全图依赖 op1）

四种策略（设备侧算的 op1/op2/元素数完全相同）：

| 策略 | launch 次数 | 设备侧 grid 启动 | 中间量 y | 依赖协调 |
|---|---|---|---|---|
| **eager2** | 2（host） | 2 | 落 HBM | 两次 launch 串行 |
| **graph2** | 1（host，graph replay） | 2 | 落 HBM | 图示串行边 |
| **barrier** | 1 | 1 | 落 HBM | kernel 内 `grid.sync()`（cooperative） |
| **fused** | 1 | 1 | **寄存器，不落 HBM** | 无需协调（每 block 各管自己的 tile） |

差分逻辑（都是干净的单一变量对照）：

- `eager2 → barrier`：消掉第 2 次 launch / grid 启动 → **损失二收益**。
- `barrier → fused`：中间量从 HBM 往返（读 x 写 y 读 y 写 z = 16 B/元素）降到只在
  寄存器（读 x 写 z = 8 B/元素）→ **损失三收益**。
- `eager2 → graph2`：只消 host 侧提交、设备侧与中间量都不变，作为对照。

算子刻意选便宜、数组放大到 256 MB 量级，使 kernel 落在 **HBM 带宽主导** 区间——此时
损失三应当显现为接近 **2×**（流量减半），而损失二应当**近乎看不见**（与 P5-1 结论一致）。

### 2.3 结果

| M (元素) | eager2 ms | graph2 ms | barrier ms | fused ms | eager2/graph2 | eager2/barrier | barrier/fused |
|---|---|---|---|---|---|---|---|
| 16 Mi | 0.0990 | 0.0972 | 0.1006 | 0.0504 | 1.02× | 0.98× | **2.00×** |
| 64 Mi | 0.3796 | 0.3777 | 0.3816 | 0.1918 | 1.01× | 0.99× | **1.99×** |
| 256 Mi | 1.5113 | 1.5092 | 1.4909 | 0.7682 | 1.00× | 1.01× | **1.94×** |

表观显存带宽 @ M=256 Mi 元素：eager2 = 2842、graph2 = 2846、barrier = 2881 GB/s
（按 16 B/elem 计），fused = 2796 GB/s（按 8 B/elem 计）——**四者的表观带宽一致
（~2.8 TB/s，H100 量级），说明 fused 的 2× 时间优势确实来自流量减半，而非更快的内存路径**。
正确性核对：`max|z_two − z_fused| = 0.000e+00`（100 万元素），融合不改变数值结果。

### 2.4 读法

1. **损失三是本区间的主导收益**：`barrier/fused ≈ 2.0×`，与「16 B/elem → 8 B/elem」的
   预测精确吻合。这正是 MPK §2.2（快照行 311–331）「算子融合消掉中间张量落 HBM 的读写」
   在最小规模上的复现，也和 §6.4（行 1551–1553）fused gather-GEMM 省下 gather 中间张量
   （SGLang 中占 11%）同源。
2. **损失二在本区间近乎为零**：`eager2/barrier ≈ 1.0`、`eager2/graph2 ≈ 1.0`。因为每
   kernel ~0.75 ms 是内存时间，~2 µs 的 launch 与 grid barrier 被彻底淹没。这不是
   「launch 不重要」，而是 **launch 只在 launch-bound 区间重要**（P5-1 结论的直接印证）：
   要看到损失二，需要「很多个、每个都很轻」的算子（MPK 报告 293 launch/token、§6.6 行
   1586–1628 把 launch 压到 0.28%，正是那种场景）。
3. **graph 与 megakernel 的收益维度不同**：graph2 把 2 次 host 提交压成 1 次，但对
   设备侧 grid 启动、中间量 HBM 往返、kernel 间 barrier 一无所改，故在本区间与 eager 持平。
   megakernel 的价值在**设备侧**（少一次 grid 启动 + 中间量不落 HBM + kernel 内可跨算子
   协调）。
4. **单 kernel 内的设备侧协调成本可忽略**：`barrier` 用 cooperative `grid.sync()`（660
   block × 256 线程）完成全 grid 屏障，相对 eager2 没有可见开销——说明「把多算子塞进一个
   persistent kernel 并在里面协调」的基础设施在目标卡上**确实便宜**（呼应 P0-3：200 次
   grid sync 0.330 ms，平均 ~1.65 µs/次）。

### 2.5 与 megakernel 判据（C1–C5，见 `docs/00` §3）的映射

| 策略 | C1 单 launch | C2 常驻 | C3 多算子 | C4 设备内协调 | C5 SM 级调度 |
|---|---|---|---|---|---|
| eager2 | ✗ | ✗ | ✗（2 kernel） | ✗（host 串行） | ✗ |
| graph2 | ✓（1 次提交） | ✗ | ✗ | ✗（kernel 间仍 barrier） | ✗ |
| barrier | ✓ | ✓ | ✓ | ✓（grid barrier） | 部分（grid-stride） |
| fused | ✓ | ✓ | ✓ | （无需） | 部分（grid-stride） |

- **fused 是最小的「单 kernel 多算子融合」**：满足 C1–C3，中间量不落 HBM；因为两个算子
  逐元素、每 block 各管自己的 tile，**不需要** C4 的设备内协调——它更像「持久化版算子融合」
  （`docs/00` §4 谱系 T2 的上限），而非完整 megakernel。
- **barrier 展示了完整的设备侧协调能力**（C1–C4）：单 launch 内做两阶段 + 全 grid 屏障，
  是「段级 megakernel」的最小骨架（对照 `docs/02` DeepGEMM MoE 的 ring barrier /
  Mega Gate 的 ScoreBarrier——真实案例用**异步原子 barrier**而非 cooperative 全 grid
  屏障，二者都是「单 kernel 内协调」，但前者可重叠、后者是硬同步）。
- 与 `docs/03` §3 术语边界一致：persistent 只是机制（C1+C2），本节的 fused 正是靠
  persistent 机制拿到「融合后不落 HBM」的收益。

### 2.6 局限与未覆盖项

- **只做了 2 个算子的逐元素融合**，未覆盖 MPK 真正的难点：跨算子 **software pipelining**
  （§5.3 行 1326–1363）——本实验 op2 逐元素依赖 op1，无跨 tile 独立性可流水；也未做
  in-kernel 去中心化调度（worker/scheduler 动态取任务，§5 行 1147–1308）。这两项在
  bandwidth-bound 的逐元素融合里本来也难显现。
- **barrier 用的是 cooperative 全 grid 同步**，与生产 megakernel 的**环形/异步原子
  barrier**（DeepGEMM，见 `docs/02` §1.5）不同：前者每次同步要全 grid 会齐，后者可在
  细粒度上重叠。本实验不宣称复现了生产级协调机制，只证明「单 kernel 内协调在本机可行且
  便宜」。
- 单卡、单 stream、合成算子；真实 MoE/Attention 的收益还取决于张量核占比与通信，需另一套
  实验（与 `code/mega-moe/` 专题互补，本专题不重复）。
- 未扫更小 M / 更多算子：那才是损失二显形的区间（见 2.4 第 2 点），可作为后续增量。

---

## 3. PDL 链式流水：一对、一条链能省多少（P5-3）

### 3.1 问题

P4-2（`docs/03` §2）只量了**一对** kernel 的 PDL 重叠，得到结论「重叠量 = `min(上游 tail, 下游 prologue)`」，
收益上限 1.5×。一个自然的追问是：如果把 **K 个 kernel 用 PDL 串成一条链**，这条「kernel 级流水线」
能不能像真正的软件流水那样**随 K 累积收益**？本节用实验回答——结论是**不能**：

> PDL 在链上的收益由**每条边的 `min(prologue, tail)`** 决定，**不随 K 叠加**；
> 速度比只取决于 `min(prologue, tail)` 在单 kernel 全时长中的占比。

这与 MPK 的跨算子软件流水（论文 §5.3 行 1326–1363）形成清晰对照：后者在**单个 kernel 内**、
以 **task 为粒度**、靠设备内事件队列让不同 task 的计算/通信/访存相互重叠，收益可随流水深度累积；
PDL 只是把**相邻两个 kernel** 沿时间轴错开，无法把重叠下沉到 task/SM 级。

### 3.2 方法

程序 `src/pdl_pipeline.cu`（目标卡 H100 / sm90，原始输出 `harness/pdl_pipeline.out.txt`）。
构造一条 K 个 kernel 的链，链上第 i 个 kernel 的时间轴为：

```
[prologue 自旋 pl] -> cudaGridDependencySynchronize -> [临界段 crit: 校验上游 seq、写自己的 seq]
    -> cudaTriggerProgrammaticLaunchCompletion -> [tail 自旋 tl]
```

- `idx=0` 无上游；`idx>=1` 的 kernel 用 `cudaLaunchKernelEx` +
  `cudaLaunchAttributeProgrammaticStreamSerialization` 启动（与 P4-2 同款）。
- 用 `gen` 代次与 `seq[K+1]` 数组做**正确性门限**：第 i 个 kernel 放行后校验 `seq[i]==gen`
  （即上游确实把数据交过来了），失败则原子加 `fail`。每轮的链收尾用 `cudaStreamSynchronize`，
  避免轮间重叠污染计时。
- 扫 K = 1/2/4/8/16，固定 `pl=200000`、`crit=2000` 周期，测两组：
  **A** `tl=200000`（prologue=tail）、**B** `tl=600000`（tail=3×prologue）。iters=30。

### 3.3 结果

**A 组：`prologue = tail = 200000`（crit=2000）**

| K | no-PDL ms | PDL ms | speedup | no-PDL/K ms | PDL/K ms | 预测 PDL/K ms |
|---|---|---|---|---|---|---|
| 1 | 0.2094 | 0.2095 | 1.00× | 0.2094 | 0.2095 | —（单 kernel 无链边）|
| 2 | 0.4132 | 0.3119 | 1.32× | 0.2066 | 0.1559 | 0.1038 |
| 4 | 0.8207 | 0.5166 | 1.59× | 0.2052 | 0.1291 | 0.1031 |
| 8 | 1.6360 | 0.9264 | 1.77× | 0.2045 | 0.1158 | 0.1028 |
| 16 | 3.2665 | 1.7452 | 1.87× | 0.2042 | 0.1091 | 0.1026 |

**B 组：`prologue = 200000`、`tail = 600000`（crit=2000）**

| K | no-PDL ms | PDL ms | speedup | no-PDL/K ms | PDL/K ms | 预测 PDL/K ms |
|---|---|---|---|---|---|---|
| 1 | 0.4099 | 0.4099 | 1.00× | 0.4099 | 0.4099 | — |
| 2 | 0.8142 | 0.7126 | 1.14× | 0.4071 | 0.3563 | 0.3056 |
| 4 | 1.6226 | 1.3184 | 1.23× | 0.4056 | 0.3296 | 0.3045 |
| 8 | 3.2403 | 2.5301 | 1.28× | 0.4050 | 0.3163 | 0.3040 |
| 16 | 6.4716 | 4.9508 | 1.31× | 0.4045 | 0.3094 | 0.3036 |

正确性：两组所有 K、所有轮次累计 `fail = 0`；链尾 `seq[16] == gen`，说明每个下游都正确
读到了上游写入的值（PDL 的内存可见性语义成立，与 P4-2 checksum 一致）。

### 3.4 读法

1. **no-PDL 严格串行**：每 kernel 时间恒定 ≈ `pl+crit+tl`（A 组 ~0.204 ms ≈ 402000 周期，
   B 组 ~0.405 ms ≈ 802000 周期），总时间随 K 线性增长。
2. **PDL 把每条链边重叠掉恰好 `min(prologue, tail)`**。
   - A 组稳态每 kernel ≈ 0.109 ms（≈ 215000 周期），比 no-PDL 少 ~200000 周期 = `min(200000,200000)`；
   - B 组稳态每 kernel ≈ 0.309 ms（≈ 610000 周期），比 no-PDL 少 ~192000 周期 ≈ `min(200000,600000)`。
   这正是 **P4-2 单对结论在链上的直接外推**：每条边独立重叠 `min(prologue, tail)`，边与边不叠加。
   表中「预测 PDL/K」列即按 `span − min(prologue, tail)` 计，A 组预测 0.1026 vs 实测 0.1091、
   B 组预测 0.3036 vs 实测 0.3094（差 ~6%，为链两端预热与 host 逐个 launch 的固定成本）。
3. **speedup 趋于一个由 `min(prologue,tail)` 决定的常数，而非随 K 增长**：
   A 组 → `402000/(402000−200000) ≈ 2.0×`（K=16 已达 1.87×）；
   B 组 → `802000/(802000−200000) ≈ 1.33×`（K=16 已达 1.31×）。
   **关键差异**：A 组 prologue=tail，重叠占半，接近 2×；B 组 tail 更长（看似「有更多可重叠的尾部」），
   但重叠被**较短的 prologue** 封顶，收益反而更小。这与直觉「尾巴越长越能藏」相反——
   因为能提前启动的是**下游的 prologue**，上游 tail 再长也无人来重叠它。
4. **这解释了 PDL 的适用区间**：只有当下游 prologue 与上游 tail **相当且都足够长**时收益才大；
   对「很多个、每个都很轻」的算子图（MPK 报告 293 launch/token 的场景，§6.6 行 1586–1628），
   单 kernel 太短、prologue 更短，`min(prologue, tail)` 极小，PDL 基本无收益。

### 3.5 与判据 / MPK 的关系

- **仍是 kernel 级**：链上每个 kernel 都由 host 逐个 launch（launch 次数 = K 不变，损失二不受影响），
  中间量若需跨 kernel 传递仍要落 HBM（损失三不受影响），重叠只发生在**相邻 kernel 边界**。
  判据映射与 P4-2 完全一致：**全不满足 C1–C5**。
- **与 MPK 软件流水的本质差别**：MPK §5.3（行 1326–1363）的 cross-task software pipelining 在
  **单 kernel 内**让**不同 task**（可来自不同算子）的访存/计算/通信重叠，且因为这些 task 属于同一
  常驻 kernel、由设备内事件驱动，流水可以**跨算子、跨流水级累积**；PDL 的流水线**级数永远只有
  1 级**（第 i 个对第 i−1 个），且每一级都要 host 先 launch 好。这就是 `docs/03` §2.4「粒度停在
  kernel 对」在链式场景下的量化证据。
- **对 mega-moe 的旁证**：`code/mega-moe/` 专题会看到 DeepGEMM 在单 kernel 内用 `l2_full_mask` 位掩码
  + ring generation 协调多个 CTA（属 T4 段级），其重叠正是 MPK 式「单 kernel 内、task/CTA 级」，
  而非本节这种「kernel 级、host 逐对 launch」。两者不可混淆。

### 3.6 局限

- 用 `grid=1` 的单一小 kernel 造流，未验证多 block / 多 SM 下 PDL 的资源约束（如后继 grid 的
  block 何时真正获得 SM 资源）；真实算子 block 更多，重叠边界可能被 occupancy 影响。
- `spin_cycles` 是合成忙等，`prologue`/`tail` 是人为切分的段；真实算子的「可提前做的 prologue」
  由 kernel 结构决定，未必能任意切。
- 只测了「均匀链」（每个 kernel 同构）与两组参数；异构链、以及 PDL 与 CUDA Graph 叠加
  （`docs/03` §2.3 提到可叠加）未覆盖，可作为后续增量。
- 有 6% 左右的实测 vs 预测偏差未进一步归因（含 host 逐个 launch 与链首尾的固定成本）。

---

## 4. N 算子链的融合：损失随 N 的标度（P9-1）

### 4.1 问题

P5-2（§2）只做了 **2 个**依赖算子的融合，并留了「未扫更小 M / 更多算子」的尾巴；
P5-3（§3）证明 PDL 的 kernel 级流水收益**不随 K 累积**，向上对照的正是 MPK 单 kernel 内
**可累积**的 task 级软件流水。本节把 P5-2 从 N=2 推广到**任意 N 个依赖算子**，回答：

1. **损失三（中间张量 HBM 往返）随 N 如何标度**？——融合把「读 x 写 z」固定为 8 B/元素，
   而未融合策略是 8N B/元素（N 个阶段各读一次写一次），预期 fused 优势随 N 增长。
2. **损失二（launch 开销）在什么区间显形**？——是否与 P5-1/P5-2 结论（只在 launch-bound
   区间重要）一致，以及「很多个、每个都很轻的算子」这一 MPK 场景（293 launch/token）长什么样。

### 4.2 方法

程序 `src/n_op_fusion.cu`（目标卡 H100 / sm90，原始输出 `harness/n_op_fusion.out.txt`）。
构造一条 N 个逐元素算子的依赖链 `a_i = op_i(a_{i-1})`（float4 向量化；算子便宜且有界，
按序号循环 4 个：`fma(v,.5,.25)`、`|v|·.5+.1`、`sqrt(|v|)`、`fma(v,-.25,.5)`）。
四种策略：

| 策略 | launch 次数 | 设备侧 grid 启动 | 中间量显存流量 | 阶段协调 |
|---|---|---|---|---|
| **eagerN** | N（host） | N | 8N B/元素 | N 次 kernel 间 barrier（同 stream） |
| **graphN** | 1（host，graph replay） | N | 8N B/元素 | 图示 N-1 条串行边 |
| **persistN** | 1 | 1 | 8N B/元素 | kernel 内 N-1 次 `grid.sync()` |
| **fusedN** | 1 | 1 | **8 B/元素**（与 N 无关） | 无需（每 block 连过全链，中间量在寄存器） |

差分逻辑与 P5-2 相同：`eagerN → persistN` 度量损失二（消 N−1 次 launch/grid 启动），
`persistN → fusedN` 度量损失三（消 N−1 组中间量往返）。三组扫描：

- **Section A**：固定 N=8，扫 M（16Ki → 16Mi 元素），看「launch 主导 → 带宽主导」过渡。
- **Section B**：固定大 M=16Mi（带宽主导），扫 N（1/2/4/8/16/32），看 fused 优势随 N 的标度。
- **Section C**：固定小 M=16Ki（launch 主导），扫 N，看轻负载下的 launch 显形。

### 4.3 结果

**Section A：N=8 固定，扫 M**

| M (元素) | eager ms | graph ms | persist ms | fused ms | e/g | e/persist | e/fused |
|---|---|---|---|---|---|---|---|
| 16 Ki | 0.0245 | 0.0168 | 0.0172 | 0.0031 | 1.46× | 1.43× | 7.95× |
| 64 Ki | 0.0245 | 0.0168 | 0.0179 | 0.0031 | 1.46× | 1.37× | 7.97× |
| 256 Ki | 0.0247 | 0.0170 | 0.0181 | 0.0033 | 1.45× | 1.37× | 7.47× |
| 1 Mi | 0.0300 | 0.0239 | 0.0262 | 0.0052 | 1.26× | 1.15× | 5.72× |
| 4 Mi | 0.0827 | 0.0721 | 0.0661 | 0.0138 | 1.15× | 1.25× | 5.99× |
| 16 Mi | 0.3901 | 0.3781 | 0.3938 | 0.0549 | 1.03× | 0.99× | 7.11× |

**Section B：M=16Mi 固定（带宽主导），扫 N**

| N | eager ms | graph ms | persist ms | fused ms | e/g | e/persist | e/fused | eager 表观带宽 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0486 | 0.0482 | 0.0514 | 0.0490 | 1.01× | 0.95× | 0.99× | ~2.76 TB/s |
| 2 | 0.0975 | 0.0956 | 0.0977 | 0.0491 | 1.02× | 1.00× | 1.98× | ~2.76 TB/s |
| 4 | 0.1955 | 0.1909 | 0.1980 | 0.0501 | 1.02× | 0.99× | 3.91× | ~2.75 TB/s |
| 8 | 0.3908 | 0.3792 | 0.3937 | 0.0548 | 1.03× | 0.99× | 7.13× | ~2.75 TB/s |
| 16 | 0.7824 | 0.7565 | 0.7854 | 0.0831 | 1.03× | 1.00× | 9.41× | ~2.75 TB/s |
| 32 | 1.5638 | 1.5112 | 1.5691 | 0.1524 | 1.03× | 1.00× | 10.26× | ~2.75 TB/s |

**Section C：M=16Ki 固定（launch 主导），扫 N**

| N | eager ms | graph ms | persist ms | fused ms | e/g | e/persist | e/fused |
|---|---|---|---|---|---|---|---|
| 1 | 0.0031 | 0.0031 | 0.0030 | 0.0031 | 0.99× | 1.02× | 1.01× |
| 2 | 0.0061 | 0.0053 | 0.0053 | 0.0031 | 1.15× | 1.15× | 1.98× |
| 4 | 0.0123 | 0.0094 | 0.0094 | 0.0031 | 1.31× | 1.31× | 3.97× |
| 8 | 0.0246 | 0.0167 | 0.0172 | 0.0031 | 1.47× | 1.43× | 7.96× |
| 16 | 0.0492 | 0.0322 | 0.0328 | 0.0037 | 1.53× | 1.50× | 13.21× |
| 32 | 0.0984 | 0.0628 | 0.0640 | 0.0052 | 1.57× | 1.54× | 18.74× |

正确性核对：`max|z_eagerN − z_fusedN| = 0.000e+00`（N=16，1 百万元素）——经 HBM 的链与
纯寄存器融合逐位一致，融合不改变数值结果。

### 4.4 读法

1. **损失三随 N 标度：只要 fused 仍是带宽主导，收益就趋近 N×。** Section B 里
   eagerN 的表观带宽恒定在 **~2.75 TB/s**（8N B/元素 ÷ 时间），说明它每加一个算子就多
   一读一写；而 fusedN 的**显存流量恒为 8 B/元素**，N=1/2/4/8 时时间几乎不变
   （0.049→0.055 ms），`e/fused` = 0.99→1.98→3.91→7.13×，与流量比 N 精确吻合。
   这正是 P5-2 的「16 B/elem vs 8 B/elem ≈ 2×」外推到 N 阶段：**中间张量往返的总成本
   随算子链长度线性增长，而融合把它压成常数**。对照 MPK §2.2（行 311–331）的算子融合
   收益、以及 §6.4（行 1551–1553）fused gather-GEMM 省中间张量，这是同一现象在 N 阶段的标度。
2. **fused 的优势在 N≥8 后被「融合自身的计算」封顶。** Section B 里 N=8→16→32 时
   `e/fused` 只从 7.13× 涨到 9.41×、10.26×，因为 fusedN 的时间开始增长
   （0.0548→0.0831→0.1524 ms），其表观带宽从 ~2.45 TB/s 掉到 ~1.6、~0.88 TB/s——
   **fused 从带宽主导转入计算主导**。所以「融合收益 = N×」只成立于「融合后仍带宽受限」
   的区间；当 N 大到 N 个算子的算术时间超过 8 B/元素的访存时间，收益饱和。这解释了为何
   生产中（DeepGEMM/MPK）融合的算子数要权衡算力/带宽比，而非越多越好。
3. **损失二只在 launch-bound 区间显形，且随 N 增长。** Section B（大 M）里
   `e/persist ≈ 1.00×`、`e/graph ≈ 1.03×`——N 个 ~0.25 ms 的阶段彻底淹没 ~2 µs 的
   launch。而 Section C（小 M=16Ki，每阶段仅 ~64 KB）里 `e/persist` 从 1.02× 升到
   **1.54×**、`e/graph` 升到 1.57×，eager 每 kernel 的边际时间 ≈ `0.0984ms/32 ≈ 3.1 µs`
   （≈ host launch 间隔），persist/graph 把它压到 ~2.0 µs。**算子越多、每个越轻，
   损失二占比越大**——这正是 MPK 报告 293 launch/token（§6.6 行 1586–1628）要治的场景，
   也是 P5-1 §1.5 结论（launch 只在 launch-bound 有意义）在「N 算子链」上的复现。
4. **graph ≈ persist（本节）**：Section C 中 `e/graph`（1.57×）与 `e/persist`（1.54×）
   几乎相同，graph 甚至略快。区别在于 graph 消 host 侧 N 次提交、设备侧仍 N 次 grid 启动，
   而 persist 消 N 次 grid 启动、但要付 N−1 次 `grid.sync()`——在本机这两种代价量级相当
   （grid 启动与 grid barrier 均在亚 µs 级）。**真正的量级差出现在 Section A 大 M 那一栏
   之外**：graph 与 persist 都不改变「中间量落 HBM」这个主导成本（Section B
   `e/fused` 远大于 `e/persist`），所以对 N 算子链，**融合（损失三）才是大头，
   launch 消除（损失二）是小头**——与 P5-2 §2.4 结论一致。
5. **Section A 的过渡**：N=8 时 `e/fused` 全程 ~5.7–8.0×（fused 恒 8 B/elem、基线 64 B/elem，
   理论 8×），小 M 端因 eager 叠加 launch 开销略高（7.95×）、大 M 端因 fused 逼近带宽峰值
   略低；`e/persist` 则从 1.43×（16Ki，launch 主导）单调降到 0.99×（16Mi，带宽主导），
   把「损失二随负载变重而消失」画成一条清晰的下降曲线。

### 4.5 与 megakernel 判据 / MPK 的关系

| 策略 | C1 单 launch | C2 常驻 | C3 多算子 | C4 设备内协调 | C5 SM 级调度 |
|---|---|---|---|---|---|
| eagerN | ✗ | ✗ | ✗（N kernel） | ✗（host 串行） | ✗ |
| graphN | ✓（1 次提交） | ✗ | ✗ | ✗（kernel 间仍 barrier） | ✗ |
| persistN | ✓ | ✓ | ✓ | ✓（grid barrier×N−1） | 部分（grid-stride） |
| fusedN | ✓ | ✓ | ✓ | （无需） | 部分（grid-stride） |

- **fusedN** 是 P5-2 `fused` 的 N 阶段推广：满足 C1–C3、中间量全程不落 HBM，但仍无跨
  算子的**软件流水**——每个 block 串行过完 N 个算子，属「持久化版算子融合」（谱系 T2 上限）。
- **persistN** 用 `grid.sync()` 在单 kernel 内做了 N−1 次全 grid 硬同步，满足 C1–C4，
  是段级 megakernel 的最小骨架；但它与生产 megakernel 的**异步原子 ring barrier**
  （DeepGEMM，见 `docs/02` §1.5）本质不同：硬同步无法让阶段间重叠。
- 与 MPK §5.3（行 1326–1363）的关系：MPK 的单 kernel 内 task 级软件流水之所以能**跨算子
  累积**，正是因为它把 N 个算子放进一个常驻 kernel、由设备内事件队列驱动（本节的
  persistN/fusedN 是这个方向的手工最小版本）。本节量化了「把 N 个算子塞进单 kernel」的
  **上界收益来自融合（损失三，可近似 N×）而非 launch（损失二，仅在轻负载显形）**。

### 4.6 局限

- 算子集合固定为 4 个便宜有界函数循环，N 增大后**融合侧转入计算主导**（§4.4 第 2 点），
  故「收益 = N×」在 N≥16 处饱和；真实算子（GEMM/Attention）的算力/带宽比不同，饱和点会移动。
- persistN 用 cooperative 全 grid 硬同步，与生产异步 barrier 不同（同 P5-2 §2.6）。
- Section C 的 `wide_grid = 132×16 = 2112` block 对 M=16Ki（4096 个 float4）严重过配，
  每阶段绝大多数 block 空转；这放大了 grid 启动的相对代价，是**刻意制造 launch 主导**，
  不应据此推断真实小算子的 launch 占比。
- 单卡、单 stream、合成逐元素算子；不含张量核 / 通信 / 跨 tile 流水，未与 `code/mega-moe/`
  的真实 MoE 前反向对比（该专题另有实测）。

---

## 5. persistent kernel 的网格大小与设备内协调成本（P10-1）

### 5.1 问题

前面几节反复用 **persistent kernel**，但网格大小基本固定在 `grid=#SM`（P5-1）或
`grid=#SM·16`（P5-2/P9-1 的 wide grid），并把「单 kernel 内协调」交给 cooperative 全 grid
屏障（P5-2 §2.2、P9-1 §4.2）。P5-1 §1.6 明确留了两条尾巴：

1. **网格大小未扫**：「grid 过小则并行度不足，过大则每 block 单元太少、调度摊薄」；
2. **全 grid 屏障的成本未随网格标度**：P0-3 只测了 `grid=132` 一个点的 ~1.65 µs/次。

本节用一个自包含实验同时回答这两点——它们正是生产 megakernel 的两个基本资源配置问题：

> **该开多大的 persistent 网格？在里面做全 grid 协调到底多贵？**

### 5.2 方法

程序 `src/grid_sweep.cu`（目标卡 H100 / sm90，原始输出 `harness/grid_sweep.out.txt`，
threads/block = 256，iters = 200，扫描点 `G ∈ {1,2,4,…,528,768,1056}`）。三个正交的子实验：

- **Section A（并行度标度）**：固定总工作量——`N=8192` 个独立工作单元，每单元 K=2048 次
  LCG 自旋 + 一次 atomicAdd——只改 `persist_units` 的 grid 大小 G（grid-stride 消费 N 个单元）。
  总设备侧工作与 G 无关，只看墙钟时间随 G 怎么变。
- **Section B（过配/摊薄）**：固定较小 `N=64` 个单元，grid 从 1 扫到 1056（**远超 N**）。看 G>N
  后大量 block 空转时，时间是否还在降。
- **Section C（协调成本）**：cooperative kernel 内做 `B=200` 次 `cg::grid_group::sync()`，
  报 **ns/barrier** 随网格 G 的曲线。该 kernel 的占用上限是 `maxBlocks/SM=8 → grid≤1056`。

### 5.3 结果

**Section A：固定工作量，扫 grid（N=8192 单元，K=2048）**

| grid G | time ms | speedup | units/blk | ns/unit | ns/unit/blk |
|---|---|---|---|---|---|
| 1 | 18.184 | 1.00× | 8192.0 | 2219.8 | 2219.8 |
| 8 | 2.288 | 7.95× | 1024.0 | 279.3 | 2234.7 |
| 32 | 0.601 | 30.28× | 256.0 | 73.3 | 2345.9 |
| 128 | 0.152 | 119.65× | 64.0 | 18.6 | 2374.7 |
| 132（#SM） | 0.147 | 123.50× | 62.1 | 18.0 | 2374.9 |
| 192 | 0.128 | 142.12× | 42.7 | 15.6 | 3046.5 |
| 384 | 0.097 | 187.12× | 21.3 | 11.9 | 4627.6 |
| 528 | 0.089 | 205.27× | 15.5 | 10.8 | 5905.9 |
| 1056（占用上限） | 0.085 | 212.94× | 7.8 | 10.4 | 12199.6 |

**Section B：小 N，扫 grid 跨过 N（N=64 单元，K=2048）**

| grid G | time ms | speedup | units/blk | active blks |
|---|---|---|---|---|
| 1 | 0.1440 | 1.00× | 64.0 | 1 |
| 8 | 0.0198 | 7.26× | 8.0 | 8 |
| 32 | 0.0066 | 21.90× | 2.0 | 32 |
| 64（=N） | 0.0043 | 33.83× | 1.0 | 64 |
| 132 | 0.0043 | 33.43× | 0.5 | 64 |
| 528 | 0.0043 | 33.17× | 0.1 | 64 |
| 1056 | 0.0044 | 33.08× | 0.1 | 64 |

**Section C：cooperative `grid.sync()` 成本 vs grid（B=200 屏障）**

| grid G | time ms | ns/barrier | ns/barrier/blk |
|---|---|---|---|
| 1 | 0.2266 | 1132.8 | 1132.8 |
| 32 | 0.2187 | 1093.4 | 34.2 |
| 128 | 0.2239 | 1119.3 | 8.7 |
| 132（#SM） | 0.2236 | 1117.8 | 8.5 |
| 192 | 0.2607 | 1303.7 | 6.8 |
| 528 | 0.3456 | 1727.8 | 3.3 |
| 1056 | 0.4806 | 2402.8 | 2.3 |

正确性核对：`grid=1/132/528` 三种网格对同一 `N=4096` 个单元都得到 `sink[0]=12288`，即
grid-stride 消费的单元数只由 N 决定、与网格大小无关。

### 5.4 读法

1. **并行度并非在 `#SM` 处饱和，而是在「占用率上限」处饱和。** Section A 里 speedup 在
   G=132（一个 block/SM）时 123.5×，但继续升到 G=1056（8 block/SM）时达 **212.9×**，仍
   在涨。原因很简单：256 线程/block、1 block/SM 只占 256/2048 = **12.5% 的 warp 槽位**，
   其余槽位空着，更多常驻 warp 才能隐藏延迟、拉高聚合吞吐。所以「persistent 用 grid=#SM」
   只在「每 block 已吃满一个 SM」时才对；**小 block 的 persistent 网格应开到占满占用率
   （本机 8 block/SM×132=1056）**，而非 #SM。
2. **但每 block 效率会随网格变大而下降。** 看 `ns/unit/blk`（每 block 处理一个单元的时间）：
   G≤132 时稳定在 ~2200–2375 ns，一旦 G>#SM 就抬升（192→3047、384→4628、1056→12200 ns）。
   即多个 block 挤在同一个 SM 上共享发射/L1/尾延迟，单 block 的单位吞吐变差。**总时间仍在降、
   单 block 效率在降**——这是一个「墙钟 vs 效率」的取舍，没有免费的午餐。
3. **过配是纯浪费。** Section B 的 speedup **恰好在 G=N=64 处封顶**（33.8×），G 再大（128…1056）
   一直平（33.0–33.4×，还略降 ~2%）。因为 64 个单元在 G=64 时已「一单元一 block」，多出来的
   block 全程空转，只增加 launch/调度噪声。**对独立单元，persistent 网格的合理取值是
   `min(占用上限, N)`**，再多无益。
4. **`grid.sync()` 有一个与网格几乎无关的 ~1.1 µs 固定地板。** Section C 里 G=1 的单 block
   屏障竟也要 **1132 ns**，G=1→132 基本恒定在 ~1092–1133 ns。这是 cooperative 全 grid 屏障
   里**全局内存栅栏（membar.gl）+ 原子往返**的固定延迟，跟有多少 block 无关。**只要做一次全
   grid 屏障，就先付 ~1.1 µs**——即使只有 1 个 block。
5. **超过 #SM 后屏障成本才随网格增长，且是次线性的。** G=132（1118 ns）→528（1728 ns，1.55×）
   →1056（2403 ns，2.15×）：约 4× / 8× 的 block 数只换来 1.5× / 2.1× 的成本。多出来的成本来自
   「一个 SM 上多个 CTA 必须先在 SM 内聚合再参与全局会合」。这条曲线比线性温和，但方向明确：
   **网格越大、全 grid 协调越贵**。

### 5.5 与判据 / MPK / 生产案例的关系

- **修正 P5-1/P5-2 的网格选择**：P5-1 用 `grid=#SM`、P5-2/P9-1 用 `wide_grid=#SM·16`，
  本节说明两者分别落在 Section A 曲线的「严重欠配」与「接近占用上限」两端——前者适合
  隔离 launch 开销的诊断（P5-1 的目的），后者才是吞吐导向的合理配置。**结论：persistent
  网格要按占用率选，不按 SM 数选。**
- **对照 MPK 的 worker/scheduler 划分**：MPK §5.1（快照行 1148–1174）把 H100 的 132 SM 划成
  「~128 个 worker SM + 16 个 scheduler warp（4 SM×4 warp）」，**每个 SM 一个常驻 block、块内多
  warp 吃满整 SM**——正是为了避开本节 Section A 的「小 block 欠配」。它也印证 Section B：
  MPK 的网格由 SM 数决定（≈N 的「角色数」），不是越多越好。
- **对照生产 megakernel 的 barrier 选择**：Section C 的 ~1.1 µs 地板解释了为什么
  DeepGEMM 的段级 mega-kernel（MoE 的 `l2_full_mask` 位掩码 + ring generation、Mega Gate 的
  `ScoreBarrier`、见 `docs/02` §1.2/§1.3/§1.5）**不用全 grid 硬同步**，而用**异步原子 barrier /
  ring**：pipeline 里每个阶段（或每个 tile）站的是几十 µs 级的工作，一次 1.1 µs 的全 grid 会合
  占比可观，且硬同步会让阶段间无法重叠。MPK §5/§6.1（行 1147–1231 / 行 1410–1413）的去中心化
  event queue 更进一步——worker 各自轮询事件计数、往 per-scheduler 的原子环形缓冲区入队，
  **根本没有全 grid 会合**，协调成本因而与网格规模解耦。
- **对 P5-3 §3.6「多 block PDL 资源约束」的旁证**：本节 Section C 证明 grid 可以开到占用上限
  （1056）且 cooperative 仍能全常驻（否则 `grid.sync()` 会死锁/报错），说明**多 block 网格在
  本机是资源可行的**；PDL 后继 grid 的 block 何时拿到 SM 资源则是另一层问题，留待后续增量。
- **与判据（C1–C5，`docs/00` §3）**：本节不引入新策略，只是给已建立的 persistent 机制
  （C1+C2，见 `docs/03` §3）**标定网格与协调成本两个工程参数**；C3–C5 仍需设备内调度器
  （MPK worker/scheduler）才满足，不在本节范围。

### 5.6 局限

- 单元是**合成 LCG 自旋 + atomicAdd**，不含访存/张量核；Section A 的「占用上限饱和」结论对
  访存密集核可能移动（访存延迟更长，需要更多 warp 才能隐藏，饱和点可能更高）。
- Section A 的 `ns/unit/blk` 抬升只归因到「多 block/SM 争用」，未用 ncu 拆解是发射、L1 还是
  尾延迟；精确归因需 profiler。
- Section C 测的是 **cooperative groups 的全 grid `grid.sync()`**，不是 DeepGEMM/MPK 那种
  异步原子 ring barrier——后者成本结构完全不同（无全 grid 会合）；本节只量化「硬屏障」这一
  参照物，不宣称代表生产 barrier 的代价。
- 未扫 threads/block、未测多 stream/多卡；`iters=200` 下 Section C 的 ~1.1 µs 地板是否含
  计时/launch 残留未进一步分离（已用 warmup + 事件计时尽量排除）。
- Section A 的 N=8192 固定，未扫 N 与 G 的联合；「G 应取 min(占用上限, N)」是结合 Section A/B
  的推断，未做二维网格。

---

## 6. 多 block PDL 的 occupancy 资源约束（P12-1）

### 6.1 问题

P4-2（`docs/03` §2）与 P5-3（§3）都用 **`grid=1` 的单个小 kernel** 造流，得到一条干净的结论：

> PDL 每条边重叠 `min(上游 tail, 下游 prologue)`，收益由这段「下游可提前做的活」封顶。

但真实算子的 grid 是**上百个 block**。于是在 P5-3 §3.6 与 P10-1 §5.5 各留了同一条尾巴：

> **PDL 后继 grid 的 block 何时真正拿到 SM 资源？上游占满 SM 时，重叠还剩多少？**

PDL 的硬件语义是「下游 grid 可以在上游触发后**提前被 lauch**」，但下游 block **仍要等某个 SM
有空闲槽位才能开始跑它的 prologue**。本节把一对 producer/consumer 从 `grid=1` 扩到多 block，
量化 **上游占用率（occupancy）** 对 PDL 有效重叠的约束。这直接关系到 megakernel：生产 megakernel
用一个常驻 kernel **占满全部 SM**（MPK §5.1 行 1148–1174；本文件 §5），它到底还能不能靠 PDL 重叠？

### 6.2 方法

程序 `src/pdl_multiblock.cu`（目标卡 H100 / sm90，原始输出 `harness/pdl_multiblock.out.txt`）。
一对 kernel，两段都可多 block：

```
producer: [pre 自旋] -> thread0 写 flag[0]=gen -> cudaTriggerProgrammaticLaunchCompletion -> [tail 自旋] -> exit
consumer: [prologue 自旋] -> cudaGridDependencySynchronize -> 校验 flag[0]==gen
```

- `pre=tail=prologue=80000` SM 周期、`crit=2000`、`iters=30`；理想重叠 = `min(tail,prologue)=80000` 周期。
- consumer 用 `cudaLaunchKernelEx` + `cudaLaunchAttributeProgrammaticStreamSerialization` 启动（与 P4-2/P5-3 同款）；
  no-PDL 基线则两个 kernel 都普通 launch（同 stream 严格串行）。
- **有效重叠** = `(no-PDL − PDL) / (no-PDL 每轮 ms/cycle)`；`ms/cycle` 由 no-PDL 跨度
  （`pre+tail+prologue`）标定，故「有效重叠 / 理想重叠」是无量纲比值（低占用处应 ≈1.00）。
- 两个 blockDim 各跑一遍以交叉验证：**256**（占用上限 `8 block/SM × 132 SM = 1056`）与
  **128**（上限 `16 × 132 = 2112`），blocks/SM 由 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 查询。
- 正确性门限：consumer 只由 block0/thread0 校验 `flag[0]==gen`，期望每轮 +1；`fail=0` 表示内存可见性成立。

X 轴分三组正交扫描：A 固定 `Gd=1` 扫上游 `Gu`；B 固定 `Gu=cap/2` 扫下游 `Gd`；C 上游占满 `Gu=cap` 扫 `prologue`。

### 6.3 结果

**blockDim=256（cap=1056）**

| Section | 变量 | 结果 |
|---|---|---|
| A | `Gu` 1→528（0–0.5 cap） | 有效重叠 **0.99–1.00×ideal**、speedup 1.49–1.50× |
| A | `Gu=1056`（=cap） | **0.74×ideal**、speedup 1.33× |
| B | `Gu=528`，`Gd` 1→528 | **0.99×ideal 恒定**（下游 grid 大小无关） |
| C | `Gu=cap`，`prologue` 0→320000 | 0→**0.75–0.86×ideal** 饱和（`prologue=0` 时 0.00×） |

**blockDim=128（cap=2112）**

| Section | 变量 | 结果 |
|---|---|---|
| A | `Gu` 1→528（0–0.25 cap） | 有效重叠 **0.99×ideal**、speedup 1.49× |
| A | `Gu=1056`（0.5 cap） | 0.93×ideal、1.45× |
| A | `Gu=1584`（0.75 cap） | **0.74×ideal**、1.33× |
| B | `Gu=1056`，`Gd` 1→1056 | 0.89–0.96×ideal（下游 grid 大小几乎无关） |
| C | `Gu=cap`，`prologue` 0→320000 | 0→**0.74–0.81×ideal** 饱和（`prologue=0` 时 0.00×） |

两组所有点 `fail=0`；低占用处 no-PDL 单对 ≈ 240000 周期 / 1980 MHz ≈ 0.121 ms，
与实测 0.125 ms 相符，确认无外部干扰（并行自驱任务会污染计时，本文件为无干扰复现）。

### 6.4 读法

1. **上游 occupancy 是 PDL 重叠的第一个约束。** Section A 里，只要上游 `Gu` 只占机器一小部分
   （两种 blockDim 下都到 `Gu≈528`、约 4 block/SM），有效重叠就贴着理想值（~1.00×）：
   下游 block 总能落脚、且所在 SM 不挤。上游 `Gu` 一旦逼近占用上限，重叠滑到 **~0.74×ideal**。
2. **两种 blockDim 收敛到同一个天花板 ~0.74×ideal。** B=256 在 `Gu=cap`、B=128 在
   `Gu=0.75cap` 处都降到 0.74——**当上游几乎占满机器时，PDL 只能拿回约 3/4 的理想重叠**，
   剩下 1/4 被「下游 block 与满负荷上游争 SM（发射/L1/尾延迟）」吃掉。这条天花板与 blockDim 无关。
3. **下游 grid 大小本身不是约束。** Section B 固定上游、把下游 grid 从 1 扫到空闲槽位上限，
   重叠几乎不变：**只要「上游 + 下游」的总占用不超过 cap，下游 grid 开多大都不影响重叠**。
   也就是说约束在**上游侧**（占不占得下、挤不挤），不在下游侧的 block 数。
4. **重叠饱和而非随 prologue 增长。** Section C 把下游 prologue 拉到 4×tail，有效重叠仍停在
   ~0.74–0.86×ideal，speedup 反而从 1.33× 回落到 1.17×（prologue 越长，未重叠的那部分也越长）。
   且 `prologue=0` 时重叠≈0——**没有「可提前做的活」就没有 PDL 收益**，这与 P4-2 一致。
5. **量级对照**：低占用处 PDL 让一对 kernel 从 `pre+tail+prologue` 压到 `pre+max(tail,prologue)`，
   即 speedup 1.50×；满占用处只剩 1.33×。**同样的 prologue/tail，收益取决于上游给不给 occupancy 余量。**

### 6.5 与判据 / MPK / 生产案例的关系

- **补齐 P5-3 §3.6 与 P10-1 §5.5 的尾巴**：§3.6 说「未验证多 block 下 PDL 的资源约束，
  后继 grid 的 block 何时真正获得 SM 资源」，本节给出答案——**受上游 occupancy 约束**：
  上游不占满时重叠理想，占满时天花板 ~0.74×ideal；且与下游 grid 大小无关。
- **对 megakernel 的直接含义**：生产 megakernel 恰恰**反常**——它用一个常驻 kernel **占满全部 SM**
  来最大化吞吐（本文件 §5 的占用率结论；MPK §5.1 行 1148–1174 的 132 SM 划分）。本节说明
  **在这个前提下 PDL 只能拿回约 3/4 的理想重叠**：跨 kernel 的 PDL 是「向上游借 occupancy 余量」，
  而 mega-kernel 把余量全用光了。这正是它改在**单 kernel 内**做 task 级重叠的根本原因——
  MPK §5.3（行 1326–1363）的 cross-task software pipelining 与去中心化 event queue（§5/§6.1）
  在 kernel 内部调度，不需要向别的 kernel 借 SM 槽位。
- **与 P5-3 的衔接**：P5-3 证明 PDL 收益**不随链长 K 累积**（每级只有 1 级流水）；本节进一步
  证明**同一级的收益还会被上游占用率打折**（1.50×→1.33×）。两条合起来：PDL 的适用区间是
  「上游轻、下游有长 prologue、且只有相邻一对」——这与整图融合的 server 负载正交。
- **生产旁证**：DeepGEMM/FlashMLA 等用 PDL 时，upstream/downstream 都是**单个或少数**相对轻的
  kernel（如 `splitkv_mla`+`combine`，见 `docs/03` §4.1）；没有谁把 PDL 架在一个占满 SM 的
  persistent kernel 之上。这与本节结论一致。
- **判据（C1–C5，`docs/00` §3）**：本节不改变映射。PDL 仍**全不满足 C1–C5**：即便多 block、
  重叠理想，它也只是「相邻两个 kernel 的边界错位」，没有单次 launch、没有常驻、没有设备内调度。
  本节只是把 PDL 的**收益边界**从 kernel 边界细化到 occupancy 边界。

### 6.6 局限

- 用 `spin_cycles`（`clock64` 忙等）造流，**不含访存/张量核**；真实 kernel 的 prologue 是访存/计算，
  与上游 tail 争的资源不同（如争 HBM 带宽 vs 争发射槽），~0.74× 天花板可能移动。
- 只测了 pair（K=2）、均匀段；**未与 P5-3 的链式结构叠加**（链上每一级的上游占用率不同），
  也未测 PDL 与 CUDA Graph 叠加（`docs/03` §2.3 提到可叠加）。
- **未用 profiler 归因**：把缩水归到「下游 prologue 与满负荷上游争 SM」是合理推断，但没拆成
  发射 / L1 / 尾延迟；也没有直接读到「后继 grid 的 block 何时**排队/上 SM**」的硬件计数。
- `Gu` 只扫到 `cap`，**未测上游过配**（`Gu>cap`，上游自身分多个 wave）对下游触发放行时机的影响；
  上游分波会使触发事件延后，是另一条机制，留作后续。
- `unspecified launch` 之外的 sm90 特性（如 cluster、`cudaLaunchAttributeProgrammaticEvent`）未覆盖。

---

## 7. 单 kernel 内跨 task 软件流水：把「损失一」测出来（P13-1）

### 7.1 问题

前面几节把三类损失里的**损失二**（launch 开销，P5-1）与**损失三**（中间张量 HBM 往返，P5-2/P9-1）
都量化过了，唯独**损失一（kernel barrier 禁止跨算子软件流水）**一直只有资料分析：
P5-2 的单 kernel 融合用的是 `grid.sync()` 硬同步，两段之间没有跨 tile 流水，§2.6 明确把它列为
未覆盖项：

> 未覆盖 MPK 真正的难点：跨算子 **software pipelining**（§5.3 行 1326–1363）——本实验 op2 逐元素
> 依赖 op1，无跨 tile 独立性可流水。

而这条机制正是 MPK 的核心卖点之一：§5.3（快照行 1326–1363）让 worker 做完一个 task 后**不等全局
barrier** 就开始下一个 task 的访存阶段；§6.6（行 1586–1596，Figure 12）报告 cross-task software
pipelining 把 task 运行时降 **1.2–1.3×**，甚至胜过 cuBLAS。线上实现里它以 `Kstages` 多层 SMEM
流水的形态出现（`docs/07` §1.2，`linear_hopper.cuh` 的 K/N 双重流水，快照 `Kstages=3/6`）。

本节在**同一个 kernel** 内造一条「访存阶段 → 计算阶段」依赖链，用一个**流水深度 `K`** 旋钮直接
度量：把「阶段串行」换成「深度 K 软件流水」，能拿回多少损失一。

### 7.2 方法

程序 `src/in_kernel_pipeline.cu`（目标卡 H100 / sm90，原始输出 `harness/in_kernel_pipeline.out.txt`）。

- `grid = 132`（=#SM，每 SM 一个常驻 block）、`blockDim = 256`、每线程 8 个 float（`E=8`）。
- 每个 block 负责一段独立输入区，循环处理 `chunks` 个 chunk；每个 chunk = `T·E = 2048` 个 float
  （8 KB），用 `cp.async`（`__pipeline_memcpy_async`，16 B/条）搬进共享内存，再用 `R` 轮 FMA 消费。
  输入总大小 = `132·chunks·2048·4 B`（chunks=256 时 **276 MB** ≫ L2 50 MB），访存基本打 HBM。
- **唯一变量是流水深度 `K`**：双缓冲/多缓冲 + `__pipeline_commit` / `__pipeline_wait_prior`。
  - `K=1`：issue copy s → wait → compute s，**两阶段严格串行**——等价于「kernel barrier 挡住的
    两个 kernel」，也等价于一个不做跨 task 流水的 persistent kernel（P5-2 的 barrier 形态）。
  - `K>1`：预取后 `K−1` 段拷贝，边算 chunk s 边让后续拷贝在飞。
- 两种模式的指令路径**完全相同**（同一条 `cp.async` 路径），唯一差别就是「下一次拷贝有没有提前
  发射」——所以 `K=1 → K` 的比值就是**软件流水能拿回的损失一**。
- 参照：纯访存（`R=0, K=8`）与纯计算 `comp_only`（只做 `chunks·E·R` 次 FMA、不读全局，用来算
  重叠理论上界 `(Tmem+Tcomp)/max(Tmem,Tcomp)`）。
- 正确性：不能有跨线程误读，因为每线程只读自己 `cp.async` 进 SMEM 的那 8 个 float；输出 checksum
  仅用于确认计算未被编译器优化掉。

`iters=30`，三组正交扫描：A 固定 `R=64` 扫 `K`；B 固定 `K=4` 扫 `R`；C 固定 `R=64,K=4` 扫 `chunks`。

### 7.3 结果

**参照**：纯访存（chunks=256, R=0, K=8）= **0.0961 ms → 2879 GB/s**（HBM 读）；
纯计算（chunks=256）：R=16/32/64/128/256 → 0.1450 / 0.2569 / 0.4811 / 0.9311 / 1.8301 ms
（≈15–19 GFLOPs，随 R 趋饱和）。

**Section A：R=64、chunks=256，扫流水深度 K**

| K | K=1 基准 ms | pipeline ms | speedup | us/chunk |
|---|---|---|---|---|
| 1 | 0.7385 | 0.7385 | 1.00× | 2.885 |
| 2 | 0.7386 | **0.6237** | **1.18×** | 2.436 |
| 3 | 0.7388 | 0.6241 | 1.18× | 2.438 |
| 4 | 0.7389 | 0.6277 | 1.18× | 2.452 |
| 6 | 0.7390 | 0.6277 | 1.18× | 2.452 |
| 8 | 0.7386 | 0.6282 | 1.18× | 2.454 |

**Section B：K=4、chunks=256，扫计算强度 R**

| R | K=1 ms | K=4 ms | speedup | 重叠理论上界 | speedup/上界 |
|---|---|---|---|---|---|
| 0（纯访存） | 0.2089 | 0.0961 | **2.17×** | 1.00× | — |
| 16 | 0.3843 | 0.2746 | 1.40× | 1.66× | 0.84× |
| 32 | 0.4983 | 0.3935 | 1.27× | 1.37× | 0.92× |
| 64 | 0.7398 | 0.6293 | 1.18× | 1.20× | 0.98× |
| 128 | 1.2144 | 1.1042 | 1.10× | 1.10× | 1.00× |
| 256 | 2.1529 | 2.0554 | 1.05× | 1.05× | 1.00× |

**Section C：R=64、K=4，扫 chunks S**

| chunks | K=1 ms | K=4 ms | speedup |
|---|---|---|---|
| 16 | 0.0450 | 0.0426 | 1.06× |
| 32 | 0.0913 | 0.0815 | 1.12× |
| 64 | 0.1882 | 0.1603 | 1.17× |
| 128 | 0.3717 | 0.3159 | 1.18× |
| 256 | 0.7385 | 0.6278 | 1.18× |

### 7.4 读法

1. **损失一在单 kernel 内可回收，且回收量级正好落在 MPK 报告的 1.2–1.3×。** Section A 里
   `K=1 → K=4`（R=64）给出 **1.18×**，与 MPK §6.6（行 1586–1596）cross-task pipelining 的
   1.2–1.3× 同档。这正是 P5-2 §2.6 留下的那条尾巴的实测答案：把「阶段之间硬等」换成「流水」，
   就是这 1.2× 左右。
2. **增益几乎全来自「深度 1 → 2」这一步，再加深没用。** 每 chunk 从 2.885 µs 降到 2.436 µs
   （−0.45 µs/chunk），`K≥2` 后完全平掉。原因：`K=1` 把**每段拷贝的完整 HBM 延迟都暴露**出来
   （深层纯访存仅 0.375 µs/chunk 吞吐，而 `K=1,R=0` 实测 0.816 µs/chunk，约 0.44 µs/chunk 是
   被暴露的延迟）；`K=2` 把「下一段拷贝」塞进「本段计算」（R=64 时 1.88 µs/chunk）之后，延迟就被
   完全藏住——本合成链里 chunk 拷贝短、计算长，一层预取足够。
3. **纯访存也能从流水里拿 2.17×（Section B, R=0）**：这说明损失一有两块——一块是「相邻拷贝之间
   的延迟重叠」（与计算无关，`K=1` 时每段拷贝串行等，`K>1` 时流水），一块是「访存与计算的重叠」。
   前者在纯流式 kernel 里就存在，后者随计算变长成为主导。
4. **越贴近理论上界，越说明机制被吃干。** Section B 的 `speedup/上界` 随 R 从 0.84→1.00：R=128 起
   流水时间就等于 `max(纯访存, 纯计算)`，即访存**完全**藏进计算。上界 `(Tmem+Tcomp)/max(Tmem,Tcomp)`
   由「访存时长 / 计算时长之比」决定——访存占比越大，可回收越多；计算越长，流水收益越趋 1×。
5. **稳态需要足够的 chunk 数（Section C）**：chunks 从 16 涨到 256，speedup 从 1.06× 收敛到 1.18×，
   因为每 block 的序言（填流水）是一次性成本；chunk 越多，这次性成本占比越小。

### 7.5 与判据 / MPK / 生产案例的关系

- **补齐 P5-2 §2.6 + §5.6 的未覆盖项**：P5-2/§5.6 说「未做跨 tile software pipelining」「barrier 是
  硬同步」。本节给出该机制的量化：**单 kernel 内把阶段串行换成深度 K 流水，R=64 时 1.18×**，
  且解释了它为什么是 1.2× 量级（访存延迟 vs 计算时长之比）。
- **对上 MPK §5.3（行 1326–1363）与 §6.6（行 1586–1596）**：论文的 cross-task software
  pipelining 正是「worker 做完一个 task，不等全局 barrier 就开下一个 task 的访存阶段」——本实验的
  `K≥2` 就是这个语义的最小模型（跨 task 边界预取一层）。论文在 Qwen3-8B final linear 上测到
  1.2–1.3×，本实验在合成链上测到 1.18–2.17×（取决于访存/计算比），**量级与机理一致**。
- **对上 P11 的 `Kstages`**：线上实现里流水的落地形态是多层 SMEM 流水——`docs/07` §1.2 的
  `linear_hopper.cuh`（`Kstages=3/6`，K 方向 × N 方向双重流水，`:219-260`）。本实验的 `K` 直接对应
  「预取几段在飞」这个旋钮：合成链里 K=2 就够，而真实 kernel 因为**迭代更长、TMA 延迟更大、还要
  同时喂 K 与 N 两个方向**，才用 3–6 级。**机制相同，深度取决于延迟/计算比与流水维度数。**
- **与判据（C1–C5，`docs/00` §3）**：本节是一个单次 launch 的常驻 kernel（C1+C2），把「访存 +
  计算」两个阶段融进一个 kernel 并用异步流水协调（C3，以及 C4 的「设备内协调」雏形）——比 P5-2 的
  `grid.sync()` 形态更接近 megakernel：**协调是异步的、可重叠的，不再是全局会合**（呼应 P10-1 §5.4
  的「~1.1 µs 硬屏障地板 → 生产用异步 ring barrier」）。它仍**不满足 C5**：没有 worker/scheduler
  去中心化调度，block 只处理自己那段区、不动态取 task。
- **边界（诚实标注）**：本节测的是**block 内、跨 stage 边界**的软件流水；MPK 的 cross-task 流水
  发生在**跨 worker、由 event 依赖驱动**的 task 之间。两者共享「预取下一段、重叠本段计算」的内核，
  但本实验未建模去中心化 event queue 与跨 block 依赖（那需要 scheduler/原子队列，见 `docs/06`
  §3/§4）。所以结论定位为「损失一的**机制上界与量级**」，不是 MPK 运行时的完整复现。

### 7.6 局限

- 合成链：`cp.async` + FMA，**不含张量核 / 真实访存模式 / TMA**；真实 kernel 的访存是 TMA bulk
  copy（延迟结构不同，`docs/07` §1.2），K 的最优深度会变。
- 每 block 只处理**自己独立的输入区**，task 之间无 data dependency；未建模跨 block 的依赖图与
  event 驱动调度，因此测的是「阶段流水」而非「去中心化跨 task 调度」。
- `R` 是规则的 FMA 轮数，不是真实的 compute/memory 比；`Kstages` 的「K 方向 × N 方向」双重流水
  未复现（本实验只有单一预取维度）。
- 单卡、固定 `grid=#SM`、固定 `blockDim=256`；未扫 grid/blockDim，也未与 P12-1 的 occupancy
  约束叠加（多 block/SM 时常驻与流水的交互留待后续）。
- 未用 profiler 分离「SMEM 读延迟」「cp.async 引擎占用」「syncthreads 成本」对 1.18× 天花板的贡献。

## 8. megakernel 的资源足迹代价：寄存器被固定为各 task 最大值（P14-1）

### 8.1 问题

前面 P5/P9/P13 都在量 megakernel 的**收益**（省 launch、省中间张量 HBM 往返、回收跨 task 软件流水）。
但 megakernel 有一项公认的**代价**：单个 kernel 必须在每个 SM 上容纳所有被融合算子的资源并集。
MPK §7 Discussion「Resource footprint」（论文快照行 1740–1748）专门回应这个顾虑：

> A common concern with mega-kernels is that a single kernel must reserve, on every SM, the
> combined resources required by all fused operators. MPK avoids this issue: tasks are
> **time-multiplexed** across SMs rather than statically partitioned, so each SM uses only the
> resources required by the task it is currently executing（行 1741–1745）。Shared memory
> follows the same pattern: the **paged abstraction**（§5.3）acquires and releases pages at task
> granularity. **The main exception is per-thread register usage, which is fixed for the
> mega-kernel at the maximum required across task types.**（行 1745–1748）

即：MPK 用**时分复用**化解了「资源并集常驻」——SMEM 靠 **paged abstraction**（§5.3，快照行
1313–1325）按 task 粒度申请/释放，不必按最大值常驻；**唯独 per-thread 寄存器不能规避**：一个
megakernel 编出来的寄存器数是「所有 task 类型的最大值」，于是**所有 task——包括本来很轻的——都被
按最重 task 的占用率运行**。这条代价此前在本专题只有资料分析（`docs/01` §5.8 的 8 条局限之一），
本节把它在目标卡 H100 / sm90 上**量化**出来。

### 8.2 方法

程序 `src/megakernel_footprint.cu`（目标卡 H100 / sm90，原始输出
`harness/megakernel_footprint.out.txt`）。`grid=528`（=4 block/SM @132 SM，让占用率真正起作用）、
`blockDim=256`、轻输入 537 MB（≫ L2）、`iters=30`。三个角色：

- **轻 task** `light_kernel`：纯访存（grid-stride 求和 + block 归约），访存受限，**占用率越高
  带宽越高**（藏 HBM 延迟靠驻留 warp）。
- **重 task** `heavy_kernel<H>`：`H` 个常驻寄存器做长 FMA 链（`H=16…192`），寄存器需求随 `H` 增大。
- **融合 kernel** `fused_light_kernel<H>` / `fused_two_kernel<H>`：**同一个 kernel** 里同时含
  轻、重两条路径（重路径用运行时 `heavy` 开关，取值 0 时只走轻路径），模拟 megakernel 把两类
  task 编进一个 kernel。这样「同样的轻工作，唯一变量是这个 kernel 里有没有重 task」。

四个 Section：A 寄存器/占用率表（`cudaFuncGetAttributes` + `cudaOccupancy…`）；B 轻路径代价扫描
（`fused_light<H>` 只走轻路径 vs `light_kernel`）；C 时分复用场景（`fused_two<H>` 单 kernel 轻后重
vs 两次 launch）；D shared memory 足迹的占用率影响（对照 MPK 的 paged SMEM）。

### 8.3 结果

**参照**：`light_kernel` 满占用（4 block/SM、32 warp/SM）读 537 MB = **0.2074 ms → 2589 GB/s**。

**Section A：寄存器 / 占用率（T=256，regs/SM=65536）**

| kernel | regs | spill | blocks/SM | warps/SM |
|---|---|---|---|---|
| `light_kernel` | 26 | 0 | 8 | 64 |
| `heavy_kernel<32>` | 40 | 0 | 6 | 48 |
| `heavy_kernel<64>` | 71 | 0 | 3 | 24 |
| `heavy_kernel<128>` | 168 | 0 | 1 | 8 |
| `heavy_kernel<192>` | 254 | 0 | 1 | 8 |
| `fused_light<32>` | 40 | 0 | 6 | 48 |
| `fused_light<64>` | 72 | 0 | 3 | 24 |
| `fused_light<128>` | 168 | 0 | 1 | 8 |
| `fused_light<192>` | 254 | 0 | 1 | 8 |

> `fused_light<H>` 的寄存器数**恰好等于** `heavy_kernel<H>`——即 megakernel 的 per-thread 寄存器
> 被抬到 `max(light=26, heavy=H)`，轻 task 完全继承重 task 的占用率。无 spill。

**Section B：轻访存路径的代价（`fused_light<H>` heavy=0 vs `light_kernel`）**

| H | light ms | fused ms | GB/s | 比率 |
|---|---|---|---|---|
| base | 0.2074 | 0.2074 | 2589.1 | 1.00× |
| 16 | 0.2074 | 0.2080 | 2581.0 | 1.00× |
| 32 | 0.2074 | 0.2079 | 2582.2 | 1.00× |
| 48 | 0.2074 | 0.2080 | 2581.0 | 1.00× |
| 64 | 0.2074 | 0.2762 | 1943.7 | **1.33×** |
| 96 | 0.2074 | 0.2758 | 1946.7 | 1.33× |
| 128 | 0.2074 | 0.4740 | 1132.7 | **2.29×** |
| 144 | 0.2074 | 0.4743 | 1131.9 | 2.29× |
| 192 | 0.2074 | 0.4744 | 1131.7 | 2.29× |

**Section C：轻+重时分复用——单 kernel（fused_two）vs 两次 launch**

| H | separate ms | fused ms | 比率 | heavy ms |
|---|---|---|---|---|
| 32 | 0.2387 | 0.2293 | 0.96× | 0.0308 |
| 64 | 0.2671 | 0.3394 | **1.27×** | 0.0587 |
| 128 | 0.3250 | 0.5829 | **1.79×** | 0.1172 |
| 192 | 0.3784 | 0.6358 | 1.68× | 0.1708 |

**Section D：shared memory 足迹的占用率影响（SMEM/SM=233472 B）**

| smem/block | blocks/SM | smem used/SM |
|---|---|---|
| 0 KB | 8 | 0 KB |
| 16 KB | 8 | 128 KB |
| 32 KB | 6 | 192 KB |
| 64 KB | 3 | 192 KB |
| 100 KB | 2 | 200 KB |
| 160 KB | 1 | 160 KB |
| 220 KB | 1 | 220 KB |

### 8.4 读法

1. **寄存器足迹是 megakernel 一项真实、可测的代价。** Section B：同样的 537 MB 轻访存工作，
   只要 kernel 里编进了一个重 task，轻 task 的带宽就从 2589 GB/s 掉到 1944（H=64）乃至
   1132 GB/s（H≥128），**慢 1.33×–2.29×**；`H≤48` 时 `max(light,heavy)` 仍是轻 task 自己，
   所以 1.00×。这个拐点正对 Section A 的 `fused_light` 寄存器/occupancy 台阶（32→24→8 warp/SM）。
2. **它随「最重 task 的寄存器需求」单调上升，并在占用率见底后饱和。** H 从 64→128，occupancy
   24→8 warp/SM，带宽再掉一档；H≥128 后 numRegs 撞到 254（sm90 上限 255）封顶，代价饱和在 2.29×。
   Section A 证实无 spill——所以这不是溢出开销，纯粹是**占用率被重 task 拉低**。
3. **时分复用的 megakernel 会为此付账，kernel-per-operator 不会。** Section C：把「轻后重」两阶段
   合进单 kernel（模拟 MPK §7 行 1742 的时分复用），随 H 从 32→128 总时间比两次 launch 慢
   **1.27×→1.79×**。账能对上：`separate ≈ light(0.207)+heavy`，`fused ≈ light_降速后(0.474, H=128)
   +heavy(0.117)` ≈ 0.583，与实测 0.5829 吻合——**差值几乎全是轻阶段被寄存器拖慢的那部分**。
   H=32 时 fused 反而略快（0.96×），因为占用率未受显著影响、单 kernel 省了一次 launch。
4. **SMEM 与寄存器的分野正是 MPK §7 的论点（Section D）。** SMEM 足迹若按各 task 最大值常驻，
   同样会把占用率从 8 block/SM 压到 1（32 KB/block 起就掉到 6，100 KB→2，160 KB→1）。
   MPK 用 **paged SMEM**（§5.3）按 task 粒度申请/释放来规避这一项（§6.1 行 1460：H100 上
   32 KB page → 每 SM 7 页）；**唯独寄存器无法分页**，只能固定为各 task 最大值——这就是本节的量化对象。

### 8.5 与判据 / MPK / 生产案例的关系

- **补齐 `docs/01` §5.8 / MPK §7（行 1740–1748）的「寄存器上限」局限**：P2-4 把它列为 8 条局限之一，
  但只是资料分析。本节给出该局限在目标卡上的**量化**：异构 task 融合时，轻 task 的访存带宽可因
  最重 task 的寄存器需求损失 **1.33×–2.29×（占用率 32→24→8 warp/SM）**；时分复用 megakernel 相对
  kernel-per-operator 慢 **1.27×–1.79×**。
- **对上 MPK §7「为什么只有寄存器是例外」**：MPK 的论证是「计算资源与 SMEM 都按 task 时分复用/分页，
  所以不按并集常驻，唯寄存器例外」。Section D 用 occupancy API 复现了「SMEM 若按最大值常驻会压占用率」
  这一前提，Section A/B 则验证「寄存器确实等于 max」——两条实验合起来**从正反两面印证了 §7 的论断**。
- **对上 P12-1（多 block PDL 的 occupancy 约束）与 P10-1（网格选占用率）**：三者都指向同一条设备内
  事实——**占用率是许多重叠/并行机制的硬约束**。P12-1 说 PDL 是「向上游借 occupancy 余量」，
  P10-1 说 persistent 网格应按占用上限选；本节补充：**megakernel 自己会主动吃掉占用率余量**
  （重 task 抬寄存器），所以它更需要 MPK 式的时分复用 + 分页来把「资源并集」拆开。
- **与生产案例（`docs/02`）对照**：DeepGEMM 的 mega MoE / Mega Gate / Mega mHC 都是 SM100 专属，
  用**固定的 warp 角色**（warp0 TMA / warp1 MMA / gate warps…）和 `setmaxnreg` 式的寄存器重配置
  （`docs/02` §1.2/§1.3、`docs/07` §2 的 `utils.cuh` setmaxnreg，注意线上实现里它被注释）。这正是
  在**工程上手工绕开**本节所测的限制：让不同角色按需增减寄存器，而不是被动接受「一个 kernel 一个
  寄存器数」。MPK 的自动路径（`docs/06`/`docs/07`）尚未做动态寄存器重配置，故保留这条上限。
- **判据（C1–C5，`docs/00` §3）**：本节是一个单次 launch、常驻、含多 task 类型的 kernel（C1+C2+C3），
  但**没有**去中心化调度（C5），也不改变判据——它是 megakernel 的**成本侧**度量，与收益实验互补。

### 8.6 局限

- 合成 workload：轻 task 是标量 grid-stride 求和、重 task 是规则 FMA 链，**不含张量核 / TMA / 真实
  混比**；真实 kernel 的最重 task（如超大 MMA 分块）寄存器需求与访存模式都不同，绝对倍率会变。
- 只测了**寄存器**这一项；未复现 MPK 的 paged SMEM 运行时（Section D 只给 occupancy 敏感性，
  没实现「按 task 粒度 acquire/release 页」）。
- `fused_two` 的时分复用是**每 block 固定先轻后重**，不是 MPK 的**去中心化 event queue 按依赖调度**；
  因此它测的是「资源足迹」而非调度行为。
- 重 task 的寄存器需求用 `H` 常驻累加器硬造，且 sm90 单线程寄存器上限 255 使 H≥128 后封顶；
  未测「重 task 也重访存」或多种重 task 并存的组合。
- 未用 `ncu` 分离「占用率」「L2 命中率」「DRAM 队列深度」对带宽下降的贡献（当前用 occupancy API +
  时间对照推断，未做 profiler 归因）。
- 单卡、固定 `blockDim=256`、单一 `grid=528`；未扫 blockDim 与 grid，也未与 P10-1 的网格标度叠加。

## 9. 最小「去中心化 in-kernel 调度」megakernel：把 C5 机制跑起来（P15-1）

### 9.1 问题

上面 §2（P5-2）和 §7（P13-1）都已经做出「单 kernel 内融合 + 单 kernel 内软件流水」，但两处的
协调手段都是 **cooperative grid 的 `grid.sync()`——集中式会合**，因此都明确标注**不满足 C5**
（无去中心化 worker/scheduler 调度）。§5（P10-1）又实测出 `grid.sync()` 有 **~1.1 µs 的固定地板**，
并据此推断：生产级 mega-kernel 之所以走「异步原子 ring barrier / 去中心化 event queue」，正是为了
绕开全局会合。`docs/01` §4 / `docs/03` §3.3 从 MPK 论文层面把去中心化调度讲清楚了（§5 总览行
1147–1174、§5.1 事件驱动行 1175–1231、§5.2 JIT/AOT 混合下发行 1232–1308），但**本专题还没有把
这套机制用原生 CUDA 真正跑起来过**——即 C5 此前只有资料分析与源码走读（`docs/06` §3/§4 的
`execute_worker`/`execute_scheduler`），没有可复现的本机实验。

本节补上这一格：写一个**最小而完整的 dataflow 调度器**——设备侧原子 ready 队列 + 每 task 事件
计数器，worker 自取自执行，**全程没有 `grid.sync()`**——把 C5 机制落到目标卡 H100 / sm90 上，
并与集中式 barrier 版、kernel-per-operator（eager）版做公平对照，量化「去中心化调度」的
**收益与开销**。

### 9.2 方法

程序 `src/decentralized_sched.cu`（目标卡 H100 / sm90，原始输出
`harness/decentralized_sched.out.txt`）。`G=132`（=1 block/SM）、`blockDim=256`、`iters=100`。

**任务图**（「多链依赖」这一最简 DAG）：`C` 条独立链、每条链 `L` 个 task 串行依赖，
`task(c,l)` 依赖 `task(c,l-1)`；`task(c,l)` 把输入第 `c` 条链第 `l` 段归约，累加到该链的运行和
`chain_sum[c]`——依赖是**真依赖**（不可跳过、不能重排）。三种策略的**总工作量完全相同**：

- **eager**：`L` 次 kernel launch，每次 kernel 以 grid-stride 处理该层全部 `C` 个 task，
  流序保证「层 l-1 先于层 l」的依赖。这是 kernel-per-operator 的等价物。
- **barrier**：单个 **cooperative persistent kernel**，逐层处理，每层后 `grid.sync()`。
  = 本专题此前 P5-2/P13-1 的形态（集中式，非 C5）。
- **decen**：单个 persistent kernel，**设备侧原子 ready 队列 + 每 task 事件计数器**，worker 自取。
  = MPK §5 的去中心化调度微缩版（C5）。

**去中心化调度器实现**（对应 MPK §5.1/§5.2 的 ring buffer + event，`docs/01` §4、`docs/06` §3）：

- 全局 `dep_count[task]`：`l==0` 置 0、`l>0` 置 1。初始只把 `C` 个 `l==0` 的 task 预入队。
- **入队** `q_enqueue`：`idx = atomicAdd(q_tail,1)` 预约槽位 → 写 `q_buf[idx]` → `__threadfence()`
  → `atomicExch(&q_ready[idx],1)` 发布（值先可见再发布标志）。
- **出队** `q_dequeue`：读 `q_head/q_tail`，`head<tail` 时用 `atomicCAS(q_head,h,h+1)` 认领槽位，
  等价于 MPK 的原子 `atomicAdd` 环形缓冲；空且 `done_count>=total` 则退出，否则 `__nanosleep`。
- **task 完成**：`atomicSub(&dep_count[succ],1)==1`（旧值 1→新值 0）时把后继入队；`atomicAdd(done_count,1)`。
- 每 block 线程 0 负责出队/入队，`__syncthreads()` 在 task 边界对齐整块。
- 终止条件：全局 `done_count == C*L`。

**正确性**：三种策略都应得到同一条 `chain_sum[c]`（链内按层升序累加）。程序同时与**串行参考**
（host 端逐链逐层同序累加）比 max 相对误差，并做**三策略互比**（barrier/decen vs eager）。
随机输入 `in[i] = (i%1000)*1e-3`。

### 9.3 结果

**Section A：扫链深 L（C=64，seg=32768，数据 33.6→536.9 MB）**

| case | tasks | eager ms | barrier | decen |
|---|---|---|---|---|
| C=64 L=4 | 256 | 0.106 | 1.082× | **0.556×** |
| C=64 L=16 | 1024 | 0.846 | 1.029× | 0.993× |
| C=64 L=64 | 4096 | 3.380 | 1.033× | 1.022× |

**Section B：扫链数 C（L=32，seg=32768）**

| case | tasks | eager ms | barrier | decen |
|---|---|---|---|---|
| C=16 L=32 | 512 | 1.675 | 1.026× | 1.016× |
| C=256 L=32 | 8192 | 3.296 | 1.018× | 0.995× |

**Section C：负载不均衡（C=528，L=16，seg=8192×伪随机 1–8×，数据 1245.7 MB）**

| 策略 | ms | vs eager |
|---|---|---|
| eager | 4.507 | 1.000× |
| barrier | 4.475 | 1.007× |
| **decen** | **3.863** | **1.167×** |

**正确性**：全部 case `fail=0`；三策略 vs 串行参考 max 相对误差 ≤ `4.1e-7`（纯浮点累加序差异），
**三策略互比 max 相对误差 = 0.00e+00（逐位一致）**。

### 9.4 读法

1. **C5 机制本身是廉价且正确的。** decen 全程无 `grid.sync()`、纯设备侧原子队列，在 6 个 case 里
   与 eager/barrier **逐位一致**（互比误差 0），说明事件计数器 + 原子 ready 队列这套 MPK 式
   调度是**可实现且不动数值**的——C5 不再是「资料里的机制」。
2. **对规则、均衡的负载，去中心化 ≈ 集中式 ≈ eager，且小负载时更慢。** Section A/B 里 decen 与
   barrier 都在 1.0× 附近（0.99–1.02×），**看不出收益**；而 Section A 的 `C=64 L=4`（每层只有 64
   个 task，却有 132 个 block）decen 反而慢到 **0.556×**——因为每层 task 数 < block 数时，
   多余的 block 在空队列上 `__nanosleep` 轮询，**调度/自旋开销**摊在很小的有效工作量上。这说明
   「去中心化」不是免费的：**task 太粗、并行度不足时，它比 `grid.sync()` 更亏。**
3. **收益出现在「task 多且负载异构」时——动态负载均衡。** Section C 让 `C=528≫G=132`（每层
   528 个 task、每 block 约 4 个）、每链段长伪随机 1–8×。此时 **barrier 只快 1.007×**（它的
   grid-stride 静态分配把重/轻链固定绑到同一批 block，逐层同步只能等最慢 block），而 **decen
   快 1.167×**——worker 自取就让快 block 连续吞下更多 task、把重链摊平。这正是 MPK §6.4 用混合
   负载均衡处理 MoE 不均衡的同一动机在最小图上的复现（对照 `docs/01` §5 的 §6.4 行 1551–1553）。
4. **与 P10-1 的 `grid.sync()` 地板对得上。** barrier 在 Section A/B/C 相对 eager 只快
   1.007–1.033×：因为 `G=132` 时每次 `grid.sync()` 约 1.1 µs（P10-1 §5.3），而每层工作量通常
   ≫ 1 µs，所以**逐层同步在 132 block 规模上本就很便宜**——去中心化要赢，必须赢在**均衡/重叠**
   上，不能指望单纯「省掉 barrier」。这纠正了「C5 主要是为了省全局屏障」的直觉。
5. **`grid.sync()` 的集中式限制是「结构性的」，不只是「成本」。** barrier 要求**所有 block 同时
   进入下一层**，因此无法表达「快链先跑」；decen 的 ready 队列天然允许任意 DAG（不只分层链），
   这正是 MPK 能处理真实算子图（§4 的 tGraph、`docs/01` §3）而 `grid.sync()` 骨架不能的原因。

### 9.5 与判据 / MPK / 生产案例的关系

- **补齐 C5 的实测空白**：此前 P5-2/P13-1 的 persistent 骨架都停在 C1–C4（`grid.sync()` 集中式），
  明确「不满足 C5」。本节 decen 是**单次 launch（C1）+ 常驻（C2）+ 多 task（C3）+ 设备内协调（C4）
  + 去中心化 worker 自取（C5）**的完整最小实现——即 `docs/00` §3 判据里的 T4 骨架，不再依赖全局屏障。
- **对上 MPK §5（快照行 1147–1368）**：本实现是论文运行时的**极小同构物**——`dep_count`≈论文的
  `EventDesc.num_triggers`、`q_buf/q_ready/q_head/q_tail`≈§5.2 的 atomic ring buffer、
  「后继计数到 0 才入队」≈§5.1 的 event 激活。差异是 MPK 有**固定角色的 worker/scheduler**
  （§5 行 1148–1174；`docs/06` §3/§4 的 `execute_worker`/`execute_scheduler`）与 **JIT/AOT 混合
  下发**（§5.2 行 1232–1308），本实验用「所有 block 都是 worker、纯 JIT 取 task」近似——够验证
  C5 机制，但省掉了 scheduler 的角色划分与预取。
- **与生产案例（`docs/02`）对照**：DeepGEMM mega 家族用**静态 warp 角色 + 原子 barrier**
  （MoE 的 `l2_full_mask` 位掩码依赖 + ring generation，`docs/02` §1.5）实现「设备内协调」，
  本质上是**把 ready 判定编译成静态位掩码**的 decen 变体；MPK 则是**运行时 ready 队列**。
  两者是「静态依赖 vs 动态依赖」的两条路，本节的 decen 对应后者。
- **与本节其它实验的拼接**：P13-1 把**损失一**（跨 task 软件流水）测出来、P14-1 把**成本侧**
  （寄存器足迹）测出来，本节把**C5 机制**测出来——三者合起来，本专题对 megakernel 的
  「收益 / 成本 / 调度机制」三面都有了本机实测。

### 9.6 局限

- 任务图是**合成的分层链**：每链每层等长（除 Section C），无张量核 / TMA / 真实算子；最简 DAG
  只用于隔离「调度机制」这一项，绝对倍率不代表真实模型。
- decen 是**纯 JIT 自取**，没有实现 MPK 的 **scheduler 角色**（§5 行 1148–1174）与 **JIT/AOT
  混合下发**（§5.2），也没有 descriptor 预取（§5.3 行 1364–1368）——空队列轮询用 `__nanosleep`
  退避，与生产级两级队列/固定角色不同（故 Section A 小负载下偏慢）。
- 每 block 用线程 0 单点出/入队，未做 MPK 的批量/预取；未测 task 数远超队列容量、多级依赖扇入
  扇出（当前每 task 恰好 1 个后继）。
- 未与 CUDA Graph 捕获 eager、也未与 PDL 组合对照；未用 `ncu` 归因自旋/原子流量。
- 单卡、固定 `G=132`（1 block/SM）；未扫 `G` 或 `blockDim`（可对照 P10-1 的网格标度）。

## 10. megakernel 的代码足迹代价：融合的 task 类型数 -> 执行代码工作集 -> I-cache（P16-1）

### 10.1 问题

§8（P14-1）把 megakernel 的一项代价——**寄存器被固定为各 task 类型的最大值**（MPK §7 行 1748
的「唯一例外」）——量化了出来。但资源足迹不止寄存器这一项。MPK 的每个 task 都是一个
**CUDA device function**：

> Each MPK task is a CUDA device function with a uniform calling convention defined by the tGraph.
> … per-task code generation is the only hardware-specific component of MPK.（§7，快照行 1750、
> 1760；§4.2 行 1138：以 thread block 为单位复用 Mirage superoptimizer 生成含软件流水/寄存器复用/
> bank-conflict 优化的 CUDA 实现）

于是整张 tGraph 的 task 都进**同一个常驻 kernel 镜像**。**融合的算子/task 类型越多、每个 task 的
device 代码越专门化，单个 kernel 的代码就越多**。寄存器（§8）和 SMEM（MPK 用 paged abstraction，
§5.3 行 1313–1325）之外，还有一条同样**无法分页**的足迹：**指令足迹**。一个 SM 在一段时间里若在
多个 task body 之间来回执行（同一 SM 上并发的 block 各跑不同 task，或同一执行流依次扫过很多
task），被执行的代码工作集一旦超过 L1 指令缓存（L1i）容量，取指就会 miss、出现停顿。这与 §8 的
寄存器足迹是**两条独立的融合代价**，本节在目标卡 H100 / sm90 上把它量化出来。

### 10.2 方法

程序 `src/code_footprint.cu`（目标卡 H100 / sm90，原始输出 `harness/code_footprint.out.txt`；
固定在空闲的 device 3 上跑，当时的 device 0 被其它并行任务占满）。`G=1056`（=8 block/SM @132 SM）、
`blockDim=256`、`steps/block=512`、共 1.38 亿元素、`iters=30`。

- **task body**：`body<I>` 是 `__device__ __noinline__` 模板函数，每个做同样 `R=64` 次 FMA
  （**名义计算量相同**），但带各自不同的立即数常量，保证 SASS 互不折叠。一个 body = 65 条连续
  FFMA = **1040 B SASS**（`cuobjdump -sass` 核对）。
- **融合 kernel** `mega<P>`：一个 kernel 内含 P 个互异 body，运行时以 **block 粒度**从函数指针表选
  一个 body 调用（类型只由步号/`blockIdx` 决定、不含 `threadIdx`，故 block 内无 warp divergence）。
  **总工作量与 P 无关**（每元素一次 body 调用、R 次 FMA）；唯一变量是 **P**——即这个 workload 实际
  扫过的 task 类型数。P=1 是「单个 task 类型（未融合）」基准。
- `fill_tab<P>` 用编译期递归只实例化 `body<0..P-1>`：`mega<P>` 实际会被执行的代码集合严格是前 P 个
  body，故**被执行代码工作集 ≈ P × 1.0 KB**（P=1→1 KB、P=32→33 KB、P=256→266 KB）。
- 分发是**一次间接 call**（函数指针表），开销与 P 无关；P=1 时表里只有一个 body。

两个 Section：A 代码足迹扫描（固定总工作量扫 P）；B **代码工作集隔离**——同一张 P 项函数指针表、
同样多的间接调用、同样工作量，只改「每条执行流扫过的 body 数」：`spread` 随步号轮转（≈P 个 body）
vs `sticky` 每 block 固定一个 body（=1 个 body）。另用 `ncu` 对 spread/sticky 采 instruction-fetch /
branch-resolving 停顿占比做归因。

### 10.3 结果

**Section A：代码足迹扫描（固定总工作量，扫 P）**

| P | ms | GFLOP/s | vs P=1 | 执行代码工作集 |
|---|---|---|---|---|
| 1 | 0.5523 | 32079.9 | 1.000× | ~1 KB |
| 2 | 0.5884 | 30111.1 | 1.065× | ~2 KB |
| 4 | 0.5883 | 30116.1 | 1.065× | ~4 KB |
| 8 | 0.5810 | 30495.9 | 1.052× | ~8 KB |
| 16 | 0.6079 | 29145.4 | 1.101× | ~17 KB |
| 32 | 0.7180 | 24676.7 | 1.300× | ~33 KB |
| 64 | 0.6448 | 27474.4 | 1.168× | ~67 KB |
| 128 | 0.7860 | 22540.2 | 1.423× | ~133 KB |
| 256 | 1.2017 | 14742.5 | **2.176×** | ~266 KB |

**Section B：代码工作集隔离（同 P、同工作量；spread=每流扫 P 个 body，sticky=每流固定 1 个）**

| P | spread ms | sticky ms | sticky GFLOP/s | spread/sticky |
|---|---|---|---|---|
| 1 | 0.5532 | 0.5541 | 31971.5 | 0.998× |
| 16 | 0.6075 | 0.5470 | 32386.4 | 1.111× |
| 64 | 0.6428 | 0.5421 | 32680.4 | 1.186× |
| 256 | 1.1985 | 0.5424 | 32662.0 | **2.209×** |

**ncu 归因（warp 停顿占比 / 周期 / 指令数）**

| 场景 | no_instruction | branch_resolving | imc_miss | cycles | inst_executed |
|---|---|---|---|---|---|
| spread P=1 | 0.63% | 4.92% | 0.01% | 1,088,638 | 376.7 M |
| spread P=256 | **59.19%** | 2.23% | 0.01% | 2,189,600 | 407.0 M |
| sticky P=256 | 1.61% | 4.88% | 0.02% | 1,073,461 | 372.4 M |

> `no_instruction` = 等待取指（instruction fetch）的停顿；`branch_resolving` = 等分支目标地址；
> `imc_miss` = 等立即数常量缓存。三者是「指令交付」里不同的环节。

### 10.4 读法

1. **代码足迹是 megakernel 一项真实、可测的代价。** 总 FMA 量恒定，只把「融合进这个 kernel 的
   task 类型数」P 从 1 拉到 256，整 kernel 就慢 **2.18×**（32.1 → 14.7 TFLOP/s）。拐点很清晰：
   P≤8（执行工作集 ≲8 KB）几乎无代价；P=32（~33 KB）起 1.30×；P=256（~266 KB）2.18×。这与 §8
   的寄存器足迹**独立**——本 kernel 的寄存器占用不随 P 变。
2. **代价来自「每条执行流的时间代码工作集」，而非分发表的大小。** Section B 是干净的隔离：spread
   与 sticky 用**同一张 P 项函数指针表**、同样多的间接调用、同样工作量，唯一差别是每条执行流扫过
   的 body 数。sticky 在所有 P 下都稳在 **~32.7 TFLOP/s**（= P=1 基准），而 spread 一路掉到
   **2.21×**。这正是 megakernel 的处境：一个 SM 的 warp 会随时间在多个 task body 之间轮转。
3. **ncu 把归因钉死在取指。** 从 spread P=1 到 P=256，`no_instruction` 从 0.63% 飙到 **59.19%**，
   而 `branch_resolving` 反而略降（4.9%→2.2%）、`imc_miss`≈0；执行的指令数只多 8%（376.7→407.0 M），
   周期数却涨 2.01×。sticky P=256 把 `no_instruction` 拉回 1.61%、周期回到基准。**所以这不是分支
   预测或常量缓存问题，而是纯粹的 L1 指令缓存 miss / 取指带宽**。1040 B/body 的台阶说明本机
   有效 L1i 工作集大致在**数十 KB** 量级（P=16→32 之间开始明显 miss；P=32/64 的非单调来自
   cache 组相联与 body 布局）。
4. **它解释了 MPK 为什么把「每 task 一个 device function」当成需要小心的设计点。** MPK §6.7 用
   event fusion / linearization 把 **successor 编码的 device-memory footprint 降了 4.4–5.9×**（行
   1680、1701）——那是**元数据**足迹；而 **per-task codegen 带来的指令足迹**论文并未量化（§7 只把
   「per-task codegen 是唯一硬件相关组件」当作灵活性优点）。本节补上：**融合的 distinct 代码体
   （distinct (op, shape) 实例）越多，单 kernel 的取指压力越大**。

### 10.5 与判据 / MPK / 生产案例的关系

- **与 §8（P14-1）合成 megakernel 的「双层足迹」**：寄存器足迹（按各 task 最大值固定、合上占用率，
  §8）与指令足迹（按各 task body 累积、合上 L1i，§10）是两条独立不可分页的成本。两者都随融合的
  task **异构度/多样性**上升，且都无法用 MPK 的 paged SMEM 规避。
- **对上 MPK §4.2 / §7（行 1138、1750–1760）**：论文把「每个 task = 一个 device function」「per-task
  codegen」当作与硬件解耦的优点；本节量化了它的**暗面**——device function 越多，单 kernel 的
  执行代码工作集越大。规模锚点：§6.7 报告每个算子被切成 **32–47 个 task**、Qwen3-8B 全图
  **293 → 13,867 个 task**（`docs/01` §3）；虽然很多 task 同 shape、可共享同一份模板代码，但
  **distinct 代码体数量**随「不同 (op, shape) 组合」增长——这正是本实验的 P。
- **解释生产案例的设计取舍（`docs/02`）**：DeepGEMM 的 mega MoE / Mega Gate / Mega mHC **只融合
  少数、但极重的 task 类型**（静态 warp 角色 + 大段 TMA/wgmma），不是「成百个小 task 类型」。按本节
  结论，这是**对的方向**：融合少量大 body 的取指工作集可控，而把上百个互异小 body 塞进单 kernel
  会先在 I-cache 上吃亏。反过来，MPK 的「自动把算子切成很多桶状 task」路线需要额外的**代码去重/
  同类聚合**来压住指令足迹（如 shape-generic kernel、或把同类 task 调度到一起）。
- **判据（C1–C5，`docs/00` §3）**：`mega<P>` 是单次 launch、常驻、含多 task 类型的 kernel
  （C1+C2+C3），但没有去中心化调度（C5）——它和 §8 一样是**成本侧**度量，与收益实验（P5/P9/P13）互补。

### 10.6 局限

- 合成 workload：body 是**纯寄存器 FMA** 且**等长**（1040 B），刻意不含访存以隔离「指令交付」；
  真实 task body（`docs/07` 的 `linear_hopper` 等）大得多、且混张量核/TMA/访存，绝对倍率与拐点会变。
- `mega<P>` 是单 kernel 内的**间接调用/jump-table 分发**模型；MPK 用统一调用约定的 device function，
  分发机制相近，但论文运行层的 JIT/AOT 队列（`docs/06`）有额外开销，未在本实验体现。
- 静态镜像对所有 P 相同（nvcc 把小 body 内联/jump-table 化），本实验测的是**执行代码工作集**——
  这恰对应「一个 workload 实际跑到的 task 类型数」，与 megakernel 的语义一致；但未单独测「静态
  镜像大小」本身（如 link 时间、module 载入）。
- 未精确测 L1i 容量/组相联；拐点（P=16→32）是行为观察，P=32/64 的非单调未深究。
- 单卡、固定 `G=1056`（8 block/SM）、`blockDim=256`；未扫并发度对「同时执行的 body 数」的影响
  （这点与 P10-1 的网格标度、P12-1 的 occupancy 约束相关，可作后续）。
- 未与 §8 的寄存器足迹在**同一 kernel 内叠加**测（两者独立但真实 megakernel 会同时付）。

## 11. megakernel「双层足迹」的叠加：寄存器足迹 × 代码足迹（P17-1）

### 11.1 问题

§8（P14-1）量化了 megakernel 的**寄存器足迹**——per-thread 寄存器被固定为各 task 类型的最大值
（MPK §7 行 1748 的「唯一例外」），把轻 task 的访存带宽拉低 **1.33×–2.29×**（占用率 32→24→8 warp/SM）。
§10（P16-1）量化了**代码足迹**——每个 task 是一个 CUDA device function（MPK §4.2 行 1138 / §7 行 1750），
融合的 distinct task 类型越多、被执行代码工作集越大，超出 L1 指令缓存后取指 miss，P=256 时慢 **2.18×**
（`no_instruction` 停顿 0.63%→59.19%）。

但两节的「局限」都明确写着**未把二者编进同一个 kernel 测叠加**（§8.6 / §10.6）。而真实 megakernel 里
这两笔代价**同时都要付**：一个常驻 kernel 里既有寄存器很重的 task，又有许多互异的 task body。
本节把它们放进**同一个 kernel**，问一个直接的问题：**融合的这两类足迹，代价是相乘、相加、还是相互放大？**
（§10.5 已论证二者「独立不可分页」；本节检验它们**在时间上**如何合成。）

### 11.2 方法

程序 `src/two_layer_footprint.cu`（目标卡 H100 / sm90，原始输出
`harness/two_layer_footprint.out.txt`，固定在空闲的 device 3）。`G=528`（=4 block/SM @132 SM，
与 §8 同）、`blockDim=256`、`steps/block=512`、访存数组 537 MB ≫ L2、`iters=30`。

一个 kernel `mega<P,H>` 同时含三条路径，运行时用 `do_mem` / `do_comp` 开关隔离：

- **访存路径**（occupancy 敏感，对应 §8 的轻 task）：对本 block 分到的段做 grid-stride 求和，
  读 537 MB（≫L2），占用率被拉低就直接掉带宽。
- **代码路径**（I-cache 敏感，对应 §10）：每 block 以 block 粒度从一张 **P 项函数指针表**里间接调用
  `body<0..P-1>`（P 个 SASS 互异的 `__noinline__` device function，各 65 条 FFMA ≈1 KB），
  做满 `steps` 步——一条执行流扫过的代码工作集 ≈ P 个 body。
- **重寄存器路径**：`heavy_path<H>` 保持 H 个常驻寄存器（`H=0` 时该分支被 `if constexpr` 丢弃，
  kernel 不含重代码），`H>0` 时整 kernel 的 `numRegs` 被抬到 `max(轻, H)`。

`P` 与 `H` 都是**编译期**参数，故被执行代码集合与寄存器数都精确可控；重寄存器路径只调用一次/block
（`reps=8`，计算量可忽略），**只为抬寄存器**。总工作量与 P、H 无关。

三个 Section：**A** 寄存器/占用率表；**B1** 访存路径隔离（`do_comp=0`）；**B2** 代码路径隔离
（`do_mem=0`，并扫 H 看低占用率是否放大取指代价）；**B3** 两条路径全开（叠加）。

### 11.3 结果

**Section A：寄存器 / 占用率表（T=256）**

| kernel | regs | localB | blocks/SM | warps/SM |
|---|---|---|---|---|
| `mega<1,0>` | 26 | 0 | 8 | 64 |
| `mega<1,64>` | 72 | 0 | 3 | 24 |
| `mega<1,128>` | 168 | 0 | 1 | 8 |
| `mega<256,0>` | 26 | 0 | 8 | 64 |
| `mega<256,64>` | 72 | 0 | 3 | 24 |
| `mega<256,128>` | 168 | 0 | 1 | 8 |

**Section B1：访存路径隔离（`do_comp=0`）——寄存器足迹**

| P | H | ms | vs mem-base | GB/s |
|---|---|---|---|---|
| 1 | 0 | 0.2075 | 1.000× | 2587 |
| 1 | 64 | 0.2785 | 1.342× | 1928 |
| 1 | 128 | 0.4783 | **2.305×** | 1122 |
| 256 | 128 | 0.4787 | 2.307× | 1122 |

**Section B2：代码路径隔离（`do_mem=0`）——代码足迹 + 与占用率的交互**

| P | H | ms | vs comp-base | GFLOP/s |
|---|---|---|---|---|
| 1 | 0 | 0.3334 | 1.000× | 26572 |
| 1 | 64 | 0.6039 | 1.811× | 14670 |
| 1 | 128 | 1.2043 | 3.612× | 7356 |
| 16 | 0 | 0.3963 | 1.189× | 22350 |
| 256 | 0 | 0.7022 | **2.106×** | 12616 |
| 16 | 128 | 1.2771 | 3.831× | 6936 |
| 256 | 128 | 1.5747 | **4.723×** | 5626 |

**Section B3：叠加（访存 + 代码全开）**

| P | H | ms | vs base | note |
|---|---|---|---|---|
| 1 | 0 | 0.5310 | 1.000× | base |
| 1 | 64 | 0.8773 | 1.652× | reg only |
| 1 | 128 | 1.6598 | **3.126×** | reg only |
| 16 | 0 | 0.6178 | 1.163× | code only |
| 256 | 0 | 1.2275 | **2.312×** | code only |
| 16 | 128 | 1.7379 | 3.273× | both |
| 256 | 128 | 2.0149 | **3.795×** | both |

### 11.4 读法

1. **两笔足迹都被如实复现。** B1 的 `H=128` 给 **2.305×**（对上 §8 的 2.29×）；B2 的 `P=256,H=0`
   给 **2.106×**（对上 §10 的 2.18×）；`mega<*,128>` 的 `numRegs=168`、`1 block/SM`、`8 warp/SM`
   也与 §8 一致。Section A 还给出一个关键结论：**同一 H 下 `P=1` 与 `P=256` 的 `numRegs` 完全相同**
   （26/72/168）——代码足迹**不改变寄存器占用**，二者在结构上独立（印证 §10.5「双层足迹」的前提）。
2. **在纯计算路径上，两笔代价在绝对时间上「相加」，而不是相乘。** 用 B2 拆：
   占用率差（`P=1`：H 0→128）= `1.2043−0.3334 = 0.8709 ms`；代码差（`H=0`：P 1→256）=
   `0.7022−0.3334 = 0.3688 ms`；**独立相加预测** = `0.3334+0.8709+0.3688 = 1.5731 ms`，
   **实测 `both` = 1.5747 ms**（差 0.1%）。而**相乘预测**为 `3.612×2.106 = 7.61×`，与实测 `4.723×`
   相去甚远。更直接地：**代码足迹的绝对增量几乎不随占用率变**——`H=0` 时 +0.3688 ms、
   `H=128` 时 +0.3704 ms。所以「低占用率会放大取指停顿」的假设**不成立**：两者各占总停顿预算的
   一块，彼此是**可加**的，不是乘性的。
3. **混合负载下总代价甚至略低于「边际相加」。** B3 里 `reg only` 增量 = `+1.1288 ms`、
   `code only` 增量 = `+0.6965 ms`；边际相加预测 = `0.5310+1.1288+0.6965 = 2.3563 ms`，
   实测 `both` = **2.0149 ms**（低 ~14%）。原因是访存路径与代码路径分属不同资源、可跨 block 部分
   重叠，且代码路径不受 P 影响的访存段在两种情形下都在跑。**乘性直觉（3.13×2.31=7.2×）严重高估**；
   真实叠加是先**加**再因重叠**略打折**。
4. **但绝对组合惩罚依然很大。** 纯计算路径 `P=256,H=128` 达 **4.723×**、混合负载 `both` 达
   **3.795×**。也就是说：一个同时融合「重寄存器 task」与「大量互异 task body」的 megakernel，
   即使两者各自都不是灾难（各 ~2×），合起来仍是 3.8–4.7× 的代价。这从**成本侧**解释了生产
   megakernel 的设计纪律：**要么少融合几个极重的 task 类型（DeepGEMM 路线，§10.5），要么用
   `setmaxnreg` 式动态寄存器重配置把重 task 的寄存器在时间上摊开（§8.5）**，否则两层足迹会在
   同一 kernel 里叠加兑现。

### 11.5 与判据 / MPK / 生产案例的关系

- **补齐 §8.6 / §10.6 两条「未叠加」的尾巴**：把 P14-1 的寄存器足迹与 P16-1 的代码足迹编进
  **同一个 kernel**，首次给出二者的**合成律**——绝对时间近似可加、略受跨 phase 重叠折扣，
  **不是乘性**。这修正了「把各 task 的足迹倍率相乘来估 megakernel 成本」的自然但错误的直觉。
- **对上 MPK §7（行 1740–1748、1750–1760）**：论文把「per-thread 寄存器固定为各 task 最大值」
  列为唯一例外，把「per-task codegen」当作优点；本节说明**这两个被论文分别讨论的点，在同一个
  常驻内核里会同时兑现**，且它们的成本分属两项互不复用的物理资源（寄存器/占用率 vs L1i/取指），
  因此**都无法靠 paged SMEM 规避**——这是自动 megakernel 合成器需要显式建模的两项约束。
- **对上 §8（P14-1）与 §10（P16-1）**：三者构成成本侧的完整链条——§8 寄存器足迹、§10 代码足迹、
  §11 二者叠加。收益侧（P5/P9/P13）与调度机制（P15）不受影响。
- **与生产案例（`docs/02`）对照**：DeepGEMM 的 mega 家族只融合**少数、极重** task 类型，用静态
  warp 角色 + 大段 TMA/wgmma：按本节结论，这恰好同时压住了寄存器足迹（角色数少、可 `setmaxnreg`
  重配）与代码足迹（body 少而大）。**「少而重」而非「多而杂」是同时规避两层足迹的同一个设计选择。**
- **判据（C1–C5，`docs/00` §3）**：`mega<P,H>` 是单次 launch、常驻、含多 task 类型的 kernel
  （C1+C2+C3），无去中心化调度（C5）——与 §8/§10 一样是**成本侧**度量。

### 11.6 局限

- 合成 workload：访存路径是标量 grid-stride 求和、代码路径是等长纯寄存器 FMA（65 FFMA/body）、
  重寄存器路径是规则 FMA 链；**不含张量核 / TMA / 真实算子混比**，绝对倍率会变。
- 三条路径在本 kernel 内是**先访存后计算再重寄存器**的串行相位（跨 block 可重叠），非 MPK 的
  去中心化 event queue 按依赖调度；测的是**资源足迹的合成**而非调度行为。
- `H` 用常驻寄存器累加器硬造，sm90 单线程上限 255 使 `H≥128` 后受限（`mega<*,128>` 实测 168 regs）；
  未测多个重 task 并存、也未测 `setmaxnreg` 动态重配置（生产绕开手段只在 `docs/02` 资料层面）。
- 代码路径的 P 只取 1/16/256；未与 §10 的 spread/sticky 隔离组合，也未用 `ncu` 对 `both` 场景做
  `no_instruction` × occupancy 的联合归因（当前用时间分解推断加法性）。
- 单卡、固定 `G=528`（4 block/SM）、`blockDim=256`；未扫网格/blockDim（可对照 P10-1/P12-1）。

## 12. megakernel 寄存器足迹的「可绕开性」：`setmaxnreg` warp 级再平衡（P18-1）

### 12.1 问题

§8（P14-1）把 MPK §7（论文快照行 1740–1748）点名的唯一例外——「per-thread 寄存器被固定为各
task 类型的最大值」——量化了出来：把一个重 task 编进单 kernel，轻访存 task 的带宽损失
**1.33×–2.29×**（占用率被拉低）。§8.5/§8.6 又指出，生产案例（DeepGEMM mega 家族）用**手工
warp 角色 + `setmaxnreg` 式动态寄存器重配置**在工程上绕开这一项，但本专题一直**只在 `docs/02`
资料层面**提到它，**没有实测**（§8.6 末条、§11.6 末条两次记为尾巴；§11.5 也只是把「用 setmaxnreg
把重 task 的寄存器在时间上摊开」列为假设）。

本节把这条尾巴补上：**`setmaxnreg` 到底能不能收回 P14-1 那 1.33×–2.29× 的寄存器足迹代价？**

### 12.2 方法

程序 `src/setmaxnreg_rebalance.cu`（目标卡 H100 / sm90，原始输出
`harness/setmaxnreg_rebalance.out.txt`）。`G=528`（4 block/SM）、`T=256`（8 warp = 2 warpgroup）、
`n=128Mi float = 537 MB ≫ L2`、`heavy reps=200`、`iters=30`。

**机制**（Hopper / `sm90a`）：`setmaxnreg.inc/dec.sync.aligned.u32 N` 让一个 **warpgroup（128 线程）**
动态增减自己的寄存器分配（`N∈[24,256]`，8 的倍数）。它必须配合一个**显式的「入口」寄存器数**才能生效：
`__launch_bounds__(T, minBlocks)` 给 kernel 一个入口值 `= 65536/(T·minBlocks)`，占用率（blocks/SM）
按这个入口值计算；运行时重 task 的 warpgroup `inc` 到高位、其余 `dec` 到低位，**在同一 block 内换手**，
ptxas 因此不再给重路径 spill。这正是 CUTLASS/DeepGEMM warp-specialized kernel 的做法
（`docs/02` §1.2/§1.3；线上 MPK `tasks/hopper/utils.cuh:19-46` 的 `wg_increase/decrease_regs`
包 setmaxnreg，但在 `linear_hopper.cuh:217/265` 里被注释掉，见 `docs/07` §2——**默认路径没做
warpgroup 级再平衡**）。

**编译注意（实测踩坑）**：`setmaxnreg` 只在 `sm90a` 上可用，必须
`nvcc -gencode arch=compute_90a,code=sm_90a`（用 `-arch=sm_90` 会报
`Instruction 'setmaxnreg.dec' not supported on .target 'sm_90'`）；若不加 `__launch_bounds__`
（或 `-maxrregcount`）给出入口寄存器数，ptxas 会打印
`(C7508) Potential Performance Loss: 'setmaxnreg' ignored; unable to determine register count at entry.`
并**静默忽略**该指令——即 setmaxnreg 不是「免费」的运行时魔法，它要求编译器知道入口分配。

**对照**（同一个「每 block 读整段 + 做 heavy 计算」的融合工作负载，工作总量对齐）：

| kernel | 说明 |
|---|---|
| `separate` | 两次 launch：`light_kernel` + `heavy_kernel<H>`，各自按自己占用率跑（kernel-per-operator 上界） |
| `fused_naive<H>` | 单 kernel，**所有线程**都背 heavy 的寄存器（P14-1 的朴素融合，occupancy 被拉低） |
| `ws<H,U0>` | 单 kernel，warp-specialized（WG0 heavy / WG1 light），`__launch_bounds__(256,2)`，**不调** setmaxnreg |
| `ws<H,U1>` | 同上 + **setmaxnreg**（WG0 `inc→232`、WG1 `dec→24`） |

对齐方式：`naive` 用 256 线程各做 `reps=200` 次 heavy；`ws` 用 128 个 heavy 线程各做 `2·reps=400`
次（同总 FLOPs），light warpgroup 用 128 线程覆盖整段（同总访存）。

### 12.3 结果

**Section A：寄存器 / 溢出 / 占用率（T=256）**

| kernel | regs | localB | blocks/SM | warps/SM |
|---|---|---|---|---|
| `light_kernel` | 26 | 0 | 8 | 64 |
| `heavy_kernel<128>` | 168 | 0 | 1 | 8 |
| `heavy_kernel<192>` | 254 | 0 | 1 | 8 |
| `heavy_kernel<232>` | 255 | 0 | 1 | 8 |
| `fused_naive<128>` | 168 | 0 | 1 | 8 |
| `fused_naive<192>` | 254 | 0 | 1 | 8 |
| `fused_naive<232>` | 254 | 0 | 1 | 8 |
| `ws<128,lb2,U0>` | **128** | **32** | **2** | **16** |
| `ws<128,lb2,U1>` | **128** | **0** | **2** | **16** |
| `ws<192,lb2,U1>` | **128** | **0** | **2** | **16** |
| `ws<232,lb2,U1>` | 128 | 40 | 2 | 16 |

**Section B：轻访存带宽 —— `fused_naive<H>`(heavy=0) vs `light_kernel`（复现 §8-B）**

| H | light ms | fused ms | GB/s | 比率 |
|---|---|---|---|---|
| base | 0.2070 | 0.2070 | 2593 | 1.00× |
| 64 | 0.2070 | 0.2767 | 1941 | 1.34× |
| 128 | 0.2070 | 0.4733 | 1134 | **2.29×** |
| 192 | 0.2070 | 0.4735 | 1134 | 2.29× |
| 232 | 0.2070 | 0.4741 | 1132 | 2.29× |

**Section C：融合工作负载（读整段 + heavy）总时间（ms）**

| H | separate | naive | ws(U0) | ws(U1) | U1/sep | U1/naive |
|---|---|---|---|---|---|---|
| 128 | 0.3260 | 0.5828 | 0.4312 | **0.3165** | **0.97×** | **0.54×** |
| 192 | 0.3793 | 0.6367 | 2.6855 | **0.3503** | **0.92×** | **0.55×** |
| 232 | 0.4128 | 0.6684 | 3.9625 | 0.5122 | 1.24× | 0.77× |

正确性：`light_kernel` 全线程求和 = 68786352，`light_wg`（仅 WG1 读取） = 68786504，
相对差 **2.21e-6**（浮点归约序不同）；`ws<128,U1>` 融合输出有限值。

### 12.4 读法

1. **`setmaxnreg` 确实能把 P14-1 的寄存器足迹代价基本收回。** Section C：朴素融合
   `fused_naive` 相对 `separate` 慢 **1.79×/1.68×/1.62×**（H=128/192/232，即 Section B 的
   带宽代价在融合工作负载上的兑现），而 warp-specialized + setmaxnreg 的 `ws(U1)` 把
   `U1/naive` 压到 **0.54×–0.77×**（融合反而比朴素单 kernel **快 1.3×–1.8×**），并逼近甚至
   优于 kernel-per-operator 的 `separate`（**0.92×–1.24×**）。H≤192 时 `ws(U1)` 比 `separate`
   **更快**（0.97×/0.92×），因为它让轻 warpgroup 与重 warpgroup **在同一常驻 kernel 内并发**
   （`separate` 是 light→heavy 串行两次 launch），这正是 megakernel 融合本该拿到的收益。
2. **关键在于「占用率按低入口值算，重路径靠 inc 不 spill」。** Section A：`ws<*,lb2,U1>` 的
   `numRegs` 被 `__launch_bounds__(256,2)` 压到 **128**（→ 2 block/SM、16 warp/SM，是
   `fused_naive` 的两倍），但 `heavy<192>` 的重路径 **localB=0（无 spill）**——ptxas 知道 WG0
   会在运行时 `inc→232`，于是不把重路径的寄存器压到 128。对照 `ws<128,lb2,U0>`（**不调**
   setmaxnreg）：同样 128 regs / 2 block/SM，但重路径 **localB=32**（spill）。
3. **不调 setmaxnreg 的 warp-specialized kernel，一旦重路径超出入口预算就会灾难性 spill。**
   Section C 的 `ws(U0)`：H=128 时靠 2 block/SM 的高占用率还略有优势（0.4312 vs naive 0.5828），
   但 H=192/232 时 spill 使其**爆炸到 2.69/3.96 ms**（比 `ws(U1)` 慢 **7.7×**）。这说明
   **warp-specialized 只是前提，真正避免 spill 的是 setmaxnreg**——两者必须成对使用。
4. **代价侧结论修正**：§11.5 曾把「用 setmaxnreg 把重 task 寄存器在时间上摊开」列为**假设**，
   本节给出实测——它**不是免费午餐但确实有效**：需要一个**显式的 warp 角色划分 + 入口寄存器预算**
   （`__launch_bounds__`/`-maxrregcount`），运行时按角色 `inc/dec` 换手。H=232 时 `ws(U1)` 仍有
   残余 spill（localB=40、`U1/sep=1.24×`），说明当重 task 的寄存器需求**逼近入口预算的 2 倍**时，
   再平衡的余量开始不够。

### 12.5 与判据 / MPK / 生产案例的关系

- **补齐 §8.6 / §11.6 两条「未测 setmaxnreg」尾巴，并兑现 §11.5 的假设**：P14-1 只测出寄存器
  足迹的**代价**，本节测出它的**可绕开性**——生产用的 `setmaxnreg` warp 级再平衡能把代价从
  `1.6×–1.8×` 降到 `≈1.0×`（甚至因并发而更快）。
- **对上 MPK §7（行 1740–1748）与线上实现**：MPK 把「寄存器固定为各 task 最大值」留作唯一例外，
  其自动路径**没有**做 warpgroup 级再平衡（`docs/07` §2：`linear_hopper.cuh` 里
  `wg_decrease_regs<32>()` / `wg_increase_regs<160>()` 被**注释**）。本节解释了为什么：
  setmaxnreg 要求「哪个 warpgroup 跑哪个 task」这一**静态角色结构**，而 MPK 的自动 codegen
  以「一个 task = 一个 CTA」为单位（`docs/07` §0），CTA 内不做异构角色拆分，因此用不上它。
  这与 MPK §4.2 行 1126–1146「以 thread block 为单位复用 superoptimizer」的粒度选择一致。
- **与生产案例（`docs/02`）对照**：DeepGEMM mega 家族正是**手工**写出静态 warp 角色
  （warp0 TMA / warp1 MMA / gate warps…）并配 `setmaxnreg` 式寄存器重配置——本节在目标卡上
  用原生 CUDA 复现并量化了这套做法的收益（C 表）。**「静态角色 + setmaxnreg」是自动 megakernel
  合成器尚未采纳、但生产已用熟的一条成本侧优化路径。**
- **判据（C1–C5，`docs/00` §3）**：`ws` 是单次 launch、常驻、含两类 task（C1+C2+C3），
  且块内用 warp 角色做寄存器再平衡——但仍**无**去中心化 event queue 调度（C5）。与 §8/§10/§11
  一样是**成本侧**度量。

### 12.6 局限

- 合成 workload：轻路径是标量 grid-stride 求和、重路径是规则 FMA 链；**不含张量核 / TMA /
  真实混比**，绝对倍率会变。
- 只用单一 `__launch_bounds__(256,2)`（入口 128 regs）与固定 `inc=232/dec=24`；未扫
  `minBlocks`、`INC/DEC` 值、`blockDim` 或 warp 角色配比（如 1 重 + 3 轻的 512 线程块）。
- 角色是**静态**的（WG0 永远重、WG1 永远轻），非 MPK 的去中心化 event queue 按依赖动态派发；
  测的是**寄存器再平衡这一机制**，不是调度行为。
- `setmaxnreg` 是 Hopper（`sm90a`）专属；Blackwell 起寄存器再平衡机制不同（`tcgen05` 路径），
  本节结论只对目标卡 H100 / sm90 有效。

---

## 13. PDL 与 CUDA Graph 的叠加：两种降级手段能不能「相乘」？（P19-1）

### 13.1 问题

`docs/03` 把 kernel-per-operator 的两大降级手段分得很清楚：**CUDA Graph**（§1）治**损失二（launch
开销）**、**PDL**（§2）治**损失一的一个特例**——相邻 kernel 重叠 `min(上游 tail, 下游 prologue)`
（P4-2 / P5-3 已实测）。文档里写了一句「PDL 与 CUDA Graphs 可叠加」（`docs/03` §2.3），但
本专题**从未实测**（`docs/04` §6.6 明确把它记为未覆盖项）。本节把这条尾巴补上，回答三个问题：

- **Q1** CUDA Graph 的 `capture → instantiate → replay` 是否**保留** PDL 语义？也就是把一条 PDL
  链 capture 成图后，replay 时程序化依赖（`cudaTriggerProgrammaticLaunchCompletion` /
  `cudaGridDependencySynchronize`）还生效吗？
- **Q2** 若都生效，两者是**相乘**互补，还是互相挤占？
- **Q3** 谁封顶谁——Graph 消的是 launch 开销、PDL 消的是 `min(tail, prologue)`，两者的上限独立吗？

### 13.2 方法

源码 `src/pdl_graph.cu`，原始输出 `harness/pdl_graph.out.txt`（目标卡 H100/sm90，固定 device 3，
iters=30）。把 P5-3 的 PDL 链原样搬过来：**K 个 `chain_kernel` 串成一条链**，每个 kernel 的 time 轴为

```
[prologue 自旋 pl] -> cudaGridDependencySynchronize -> [crit: 校验上游 seq, 写 seq] -> trigger -> [tail 自旋 tl]
```

`idx>=1` 的 kernel 用 `cudaLaunchKernelEx` 带 `cudaLaunchAttributeProgrammaticStreamSerialization`。
对同一条链做**四种公平对照**（工作量完全相同）：

| 策略 | 说明 |
|---|---|
| `eager` | K 次 host launch，无 PDL |
| `ePDL` | K 次 host launch，`k>0` 带 PDL 属性（P5-3 的形态）|
| `graph` | 把 `eager` 的 K 次 launch `cudaStreamBeginCapture` 成图、`cudaGraphInstantiate`、replay |
| `gPDL` | 把 `ePDL` 的 K 次 launch（带 PDL 属性）capture 成图、instantiate、replay |

**正确性门限（Q1 的硬证据）**：`expect=12345` 烘进 kernel；`seq` capture 前清 0。若 PDL 排序在
图里被破坏，下游会读到 `0 != expect`，每轮 replay 都 `atomicAdd(fail)`；若排序保留，下游恒读到
`12345`、`fail` 恒 0。三个 Section：**A** 大 span `pl=tail=200000`（launch 可忽略）、
**B** 大 span `tail=3×pl`（overlap 被 prologue 封顶）、**C** 小 span `pl=tail=4000/crit=1000`
（span≈9000 周期≈4.6 µs，与 launch 开销同量级，才能同时看见两种收益）。

### 13.3 结果

**Section A（pl=tail=200000，大 span；speedup 相对 eager）**：

| K | eager ms | ePDL ms | graph ms | gPDL ms | e/ePDL | gPDL/g | e/gPDL | fail |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2052 | 0.2053 | 0.2053 | 0.2052 | 1.00× | 1.00× | 1.00× | 0/0/0/0 |
| 2 | 0.4106 | 0.3085 | 0.4096 | 0.3085 | 1.33× | 1.33× | 1.33× | 0/0/0/0 |
| 4 | 0.8213 | 0.5148 | 0.8186 | 0.5140 | 1.60× | 1.59× | 1.60× | 0/0/0/0 |
| 8 | 1.6421 | 0.9276 | 1.6354 | 0.9256 | 1.77× | 1.77× | 1.77× | 0/0/0/0 |
| 16 | 3.2827 | 1.7528 | 3.2696 | 1.7487 | 1.87× | 1.87× | 1.88× | 0/0/0/0 |

**Section B（tail=3×pl，overlap 被 prologue 封顶）**：

| K | eager ms | ePDL ms | graph ms | gPDL ms | e/ePDL | gPDL/g | e/gPDL | fail |
|---|---|---|---|---|---|---|---|---|
| 16 | 6.5154 | 4.9827 | 6.5013 | 4.9790 | 1.31× | 1.31× | 1.31× | 0/0/0/0 |

**Section C（pl=tail=4000/crit=1000，小 span，launch 可感）**：

| K | eager ms | ePDL ms | graph ms | gPDL ms | e/ePDL | gPDL/g | e/gPDL | fail |
|---|---|---|---|---|---|---|---|---|
| 4 | 0.0264 | 0.0176 | 0.0244 | 0.0168 | 1.50× | 1.45× | 1.57× | 0/0/0/0 |
| 16 | 0.1089 | 0.0626 | 0.0925 | 0.0573 | 1.74× | 1.61× | 1.90× | 0/0/0/0 |
| 32 | 0.2172 | 0.1221 | 0.1845 | 0.1122 | 1.78× | 1.64× | 1.94× | 0/0/0/0 |
| 64 | 0.4351 | 0.2393 | 0.3679 | 0.2211 | 1.82× | 1.66× | 1.97× | 0/0/0/0 |

### 13.4 读法

**(1) Q1：CUDA Graph capture 完整保留 PDL 语义——`gPDL/g` 在所有 K、A/B/C 三个 Section 里
**逐行等于** `e/ePDL`（A：1.00→1.87×；B：1.31×），且**四策略的 `fail` 计数在 A/B/C 每一格恒为 0**。
这意味着程序化依赖（上游 trigger、下游 grid-dependency-sync）在 `cudaStreamBeginCapture` /
`cudaGraphInstantiate` 后仍按原语义生效——图并没有把 PDL 退化成「普通串行边」。
（反之，若 capture 丢掉 PDL，`gPDL` 会退化成 `graph`，`gPDL/g` 应为 1.00×、`fail` 会爆炸。）

**(2) Q2：两者**不是相乘**，而是**次乘性（sub-multiplicative）**。看 Section C 大 K 的分解：

| K=64 | 每 kernel (µs) | 相对 eager 加速 |
|---|---|---|
| eager | 6.80 | 1.00× |
| graph | 5.75 | **1.18×**（Graph 消损失二）|
| ePDL | 3.74 | **1.82×**（PDL 消损失一特例）|
| gPDL | 3.45 | **1.97×**（叠加）|
| 「相乘」预测 | — | 1.18×1.82 = **2.15×** |

实测叠加 **1.97×**，只有相乘预测的 **~0.92×**。原因是二者**挤占同一段「kernel 间串行缝」**：
Graph 单独时每 kernel 省 ~1.05 µs（host 提交 + grid 启动缝隙），但在 PDL 之下这段缝隙已被
「下游 grid 提前启动」部分吃掉，Graph 的边际收益降到 ~0.28 µs/kernel（`ePDL→gPDL`）。即
**PDL 一旦把 grid 启动也重叠掉，Graph 能省的「launch 缝隙」就少了**——两种优化的作用域在
kernel 边界上重叠。

**(3) Q3：两者的上限彼此独立、各自封顶，不互相抬高。**
- **Graph 的上限 = launch 开销占比**：Section A/B 的 span（0.2–0.8 ms）远大于 launch（~2 µs），
  `graph ≈ eager`（差 <0.5%），Graph 毫无收益——正是 P5-1「损失二只在 launch-bound 区间有意义」
  的又一次体现。Section C 把 span 压到 ~4.6 µs 才看见 Graph 的 1.18×。
- **PDL 的上限 = `min(tail, prologue)/span`**：A 组 pl=tail → 1.87×（趋 2.0×）、B 组 tail=3×pl →
  1.31×（被 prologue 封顶），与 P5-3 完全一致；加不加 Graph 都改不了这条上限
  （A/B 里 `gPDL/g` 逐行等于 `e/ePDL`）。
- 一句话：**Graph 只在 kernel 时长（span）短到 launch 占比可观时有用，PDL 只在有 prologue 可提前跑时有用；
  两者可叠加但不相乘，因为它们在「kernel 间缝」上重叠。**

### 13.5 与判据 / MPK / 生产案例的关系

- **补齐 `docs/04` §6.6「未测 PDL 与 CUDA Graph 叠加」与 `docs/03` §2.3 的「可叠加」论断**：
  本节给出硬证据——**在 CUDA Graph 里 PDL 语义被完整保留**（Q1），叠加收益**次乘**（Q2），
  两者上限独立（Q3）。
- **对 MPK 路线的含义**：Graph（消 launch）与 PDL（重叠相邻 kernel）都**停在 kernel 级**，
  二者叠加后仍受 `min(tail, prologue)` 与 span 的约束，**不产生跨算子的 tile/fragment 级重叠**
  （`docs/03` §2.3/§3）：MPK 走整图 mega-kernel（`docs/01` §3–§5）而非「Graph + PDL 化整图」，
  正是因为这两种手段再怎么叠加都改不了 kernel 的粒度上限。
- **判据（C1–C5，`docs/00` §3）**：`graph` / `gPDL` 仍由 host 提交、每个 kernel 仍会结束、
  依赖仍是 kernel 级（哪怕 PDL 让它部分重叠）——**全不满足 C1–C5**，与 P4-1/P4-2/P5-3 同档，
  是 kernel-per-operator 的两块拼在一起也补不齐的降级项。
- **与生产案例（`docs/02`）对照**：DeepGEMM mega 家族**没有**依赖 Graph/PDL 的叠加，而是把
  「单次 launch + 设备内异步原子 barrier + 常驻 CTA」做进一个 kernel——本节从反面量化了为什么：
  Graph+PDL 叠加在最好的情况下也只有 ~2×，且随 span 增大立刻退化（A/B 段两者都 = 1.0×）。

### 13.6 局限

- 合成 workload：链上每 kernel 是「自旋 + 极简 seq 校验」，**不含张量核 / 真实算子**；测的是
  **PDL × Graph 的机制交互**，非端到端收益。
- `grid=1`（每 kernel 单 block）；未测多 block PDL（P12-1 已知上游 occupancy 是 PDL 的第一约束）
  与 Graph 叠加时的资源交互。
- 只测了 **capture 方式**构造的图（`cudaStreamBeginCapture`）；未测**显式构图** API
  （`cudaGraphAddKernelNode` + `cudaGraphKernelNodeSetAttribute` 的 PDL 属性）以及
  graph 内部的 programmatic event edge，后者语义可能不同。
- 正确性检查用的是「下游读到上游哨兵」这一**存在性**判据（`fail==0` 即 PDL 排序保留），
  未构造更细的跨 fragment 依赖来压测 PDL 在 Graph 里的边界行为。

## 14. 跨算子细粒度重叠：为什么「切 block 做重叠」赢不过算子级屏障（P20-1）

### 14.1 问题

本专题的**损失一（kernel barrier）**此前有两个实测侧面：

- **P13-1**：单 kernel 内、**单个 task 内部**的访存-计算软件流水（`cp.async` 载入藏进 FMA），确认
  损失一在单 kernel 内可回收 ~1.18×；
- **P5-2 / P15-1**：单 kernel 内跨 task 的**集中式** `grid.sync()` 与**去中心化** ready-queue 调度。

但 MPK §2.1（快照行 241–293，Figure 3）点名的是一类更具体的东西：**跨算子（cross-operator）细粒度
kernel overlap**。以 `MatMul → AllGather` 为例，下游 AllGather 是逐元素算子，每个下游 block 只依赖
**上游一个** block 的输出 tile；kernel barrier 却要求 AllGather 的**所有** block 等 MatMul 的**所有**
block 完成。MPK §3（快照行 371–386）进一步以 `MatMul → AllReduce` 说明：把依赖精确到「一个 task 对应
一个 task」后，compute-heavy 的 MatMul task 可与 communication-heavy 的 AllReduce task 重叠，
**同时利用计算资源与通信/访存带宽**。

这里有两个常在直觉里被混为一谈的问题：

- **Q1（依赖表示）**：把算子级屏障换成**逐 tile 的 ready 标志**，数值上是否安全？
- **Q2（收益来源）**：细粒度依赖本身是否就等于收益？还是说收益另有前提？在**固定 SM 数**的 GPU 上，
  把机器按 block 静态切给「上游算子」和「下游算子」来做并发，到底能不能赢过算子级屏障？

### 14.2 方法

源码 `src/cross_op_overlap.cu`，原始输出 `harness/cross_op_overlap.out.txt`（目标卡 H100/sm90，
device 3，iters=30，G=132=SM 数，threads/block=256）。构造 Figure 3 型的 workload：

- **NT=8192 个 tile**，每 tile **TS=4096 float**（16 KB），共 M=33.5M float（134 MB/数组）；
- **producer（上游算子）**：`buf[t] = src[t] + 1`，用 `float4` 向量化；`PR` 轮 FMA 决定它是**访存受限**
  （PR=0）还是**计算受限**（PR 大）。`buf` 是下游要读的中间张量。
- **consumer（下游算子）**：把本线程的 16 个元素**读进寄存器一次**，在寄存器上做 `R` 轮 FMA 归约
  写回 `out[t]`——R 大时 consumer 是**计算受限**（不重复访问 HBM）。tile `t` 的 consumer **只依赖**
  producer 的 tile `t`（1:1），与其它 tile 无关。

三种策略（工作量完全相同）：

| 策略 | 说明 | Figure 3 对应 |
|---|---|---|
| `serial` | 两个 eager kernel：producer 全体 -> consumer 全体（kernel barrier） | 3a |
| `barrier` | 单个 cooperative persistent kernel：全体 produce -> `grid.sync()` -> 全体 consume | 3a（单 launch） |
| `pipe` | 单个 persistent kernel：`Gp` 个 producer block + `Gc=G−Gp` 个 consumer block **并发**，每个 tile 一个 ready 标志（代次 `gen` 区分轮次），consumer 就绪即算 | 3b（细粒度重叠） |

`ideal` 列给的是「静态分块 + **完美重叠**」的**理论下界** `max(Tp(Gp), Tc(Gc))`（分别用实际 block 数
测得的单阶段时间）。四组扫描：**A** 扫 `R`（producer 访存受限）、**B** 扫 `PR`（producer 变计算受限）、
**C** 扫 `Gp`、**D** 单阶段随 block 数 `GB` 的缩放。

**正确性**：三策略的浮点运算顺序完全一致（同一 `produce_tile`、同一 block 归约树），`out[]` 应逐位相等；
以 `serial` 为参考比较 `barrier`/`pipe`，报告 `max|Δ|`。

### 14.3 结果

**Section A（producer 访存受限 PR=0，Gp=32 / Gc=100；时间单位 ms，越小越快）**：

| R | Tp | Tc | serial | barrier | pipe | s/b | pipe/b | ideal | p/ideal |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.1031 | 0.0310 | 0.1346 | 0.1325 | 0.4518 | 1.016× | **0.293×** | 0.2057 | 2.196× |
| 16 | 0.1032 | 0.1525 | 0.2539 | 0.2519 | 0.4567 | 1.008× | **0.552×** | 0.2059 | 2.218× |
| 32 | 0.1035 | 0.1860 | 0.2873 | 0.2854 | 0.4573 | 1.007× | **0.624×** | 0.2416 | 1.893× |
| 64 | 0.1034 | 0.2534 | 0.3541 | 0.3523 | 0.4606 | 1.005× | **0.765×** | 0.3290 | 1.400× |
| 128 | 0.1035 | 0.3879 | 0.4884 | 0.4866 | 0.5587 | 1.004× | **0.871×** | 0.5040 | 1.108× |
| 256 | 0.1033 | 0.6567 | 0.7573 | 0.7552 | 0.9140 | 1.003× | **0.826×** | 0.8537 | 1.071× |

正确性（R=256）：`max|barrier−serial| = 0.000e+00`、`max|pipe−serial| = 0.000e+00`——**逐位一致**。

**Section B（consumer 固定 R=80，扫 producer 计算量 PR）**：

| PR | Tp | Tc | serial | barrier | pipe | s/b | pipe/b |
|---|---|---|---|---|---|---|---|
| 0 | 0.1031 | 0.2870 | 0.3878 | 0.3858 | 0.4641 | 1.005× | 0.831× |
| 32 | 0.1955 | 0.2872 | 0.4807 | 0.4784 | 0.8106 | 1.005× | 0.590× |
| 64 | 0.2242 | 0.2870 | 0.5111 | 0.5069 | 0.9267 | 1.008× | 0.547× |
| 128 | 0.2930 | 0.2871 | 0.5806 | 0.5754 | 1.1820 | 1.009× | 0.487× |
| 256 | 0.4392 | 0.2873 | 0.7276 | 0.7174 | 1.7413 | 1.014× | 0.412× |

**Section C（R=80 下扫 pipe 的 Gp）**：

| Gp | pipe | barrier | b/pipe |
|---|---|---|---|
| 8 | 1.6767 | 0.3860 | 0.230× |
| 16 | 0.8716 | 0.3861 | 0.443× |
| 32 | 0.4641 | 0.3862 | 0.832×（最优） |
| 64 | 0.6358 | 0.3860 | 0.607× |

**Section D（单阶段随 block 数 GB 缩放；比值 = 基准 G=132 的时间 / 该 GB 的时间）**：

| GB | Tp | Tp(132)/Tp | Tc | Tc(132)/Tc |
|---|---|---|---|---|
| 16 | 0.3576 | 0.289× | 2.3021 | 0.125× |
| 33 | 0.2006 | 0.515× | 1.1203 | 0.257× |
| 66 | 0.1253 | 0.825× | 0.5647 | 0.509× |
| 132 | 0.1032 | 1.001× | 0.2873 | 1.000× |
| 264 | 0.0987 | 1.048× | 0.1540 | 1.866× |
| 528 | 0.0972 | 1.064× | 0.1319 | 2.179× |
| 1056 | 0.0971 | 1.065× | 0.1192 | 2.411× |

### 14.4 读法

**(1) Q1：逐 tile 的细粒度依赖在数值上完全安全。** `barrier`/`pipe` 相对 `serial` 的输出逐位一致
（`max|Δ|=0.00e+00`），`s/b ≈ 1.00×` 也说明**单次 launch 在本区间不再带来收益**（损失二只在
launch-bound 才显形，见 P5-1/P9-1）。逐 tile ready 标志（`atomicExch`+`gen` 代次 + `__threadfence`）
正确表达了 Figure 3b 的依赖。

**(2) Q2：但「细粒度依赖」本身不产生收益——`pipe` 在所有情况下都**慢于**算子级屏障
（`pipe/b` 恒 < 1，0.29–0.87×）。** 原因有两层，且被 Section D 直接量出来：

- **第一层：两个阶段都要「几乎整台机器」。** Section D 显示：
  - **producer（访存受限）直到 GB≈132 才喂满 HBM**（Tp 从 16 block 的 0.358 ms 降到 132 block 的
    0.103 ms，之后到 1056 block 才 1.07×、几乎平），即**访存受限阶段也需要 ~全部 132 个 SM 才能满带宽**
    （H100 每 SM 仅 8 warp、256 线程/block 时占用率低，见 P10-1）；
  - **consumer（计算受限）直到 1056 block 仍在加速**（Tc 0.287→0.119 ms，2.41×），**根本没饱和**。
  于是把机器切成 `Gp`+`Gc`，两阶段各按 ≈`G/Gp`、`G/Gc` 反比变慢：`ideal = max(Tp(Gp), Tc(Gc))`
  在 R≤64 时约 0.206–0.329 ms，而 `barrier = Tp+Tc` 约 0.25–0.35 ms——**「完美重叠」的理论下界
  至多与屏障打平**（R=16 时 0.206 vs 0.252 仅 1.2×，R≥128 时 ideal 反而更大）。
- **第二层：朴素逐 tile 同步的固定开销很大。** 实测 `pipe` 比它的 `ideal` 还慢 1.07–2.22×
  （`p/ideal` 列，R 越小越糟）：每个 tile 一次 `atomicExch` + producer/consumer 各一次 `__threadfence`，
  加上 `Gc` 个 consumer block 在 `flag[]` 上自旋轮询（即便 `__nanosleep(64)`），在 R 小时把
  provider 过早完成留下的空档全吃掉。这说明**细粒度并不等于「零成本」**：要便宜，同步必须**批量化/
  环形化**（MPK 的 ring buffer + 事件计数器，`docs/06` §3–§4；DeepGEMM 的位掩码 ring barrier，
  `docs/02` §1.5），而不是每 tile 一次原子。

**(3) producer 变计算受限（Section B）把 `pipe/b` 从 0.83× 进一步压到 0.41×。** 上游越重，
`ideal` 越大、两阶段越抢同一资源（FMA 单元），叠加上第二层开销，`pipe` 越亏——**反向印证 MPK
「overlap kernels that utilize different hardware resources」的前提**：只有资源互补且能**共驻**
才谈得上重叠。

**(4) Section C 有内点最优但仍不翻盘。** `Gp` 太小（8/16）producer 喂不满带宽、太大（64）挤占
consumer，最优 `Gp=32` 时 `pipe=0.464 ms`，仍比 `barrier=0.386 ms` 慢（`pipe/b=0.83×`）。

### 14.5 与判据 / MPK / 生产案例的关系

- **补齐「跨算子细粒度重叠」这一损失一的跨算子形态**：P13-1 证**单 task 内**的访存-计算流水可回收
  （~1.18×），本节证**跨算子的静态分块重叠在固定 SM 上结构性不划算**——两者合起来说明：MPK 的
  fine-grained overlap 之所以成立，靠的不是「把依赖画细」，而是**SM 级 task 动态共驻 + 便宜的
  ring/事件同步**（MPK §5，`docs/01` §4）。
- **与 P15-1 呼应**：P15-1 里去中心化 ready-queue 在**均衡负载下 ≈ barrier**、只在**异构负载**时赢；
  本节进一步显示，连「资源互补」的两算子静态分区都赢不过屏障——**去中心化的价值在动态负载均衡，
  不在省屏障**。
- **与 P13-1 / P18-1 呼应**：真正能拿到跨资源重叠的是**同一 SM 内的 warp 特化**（P18-1 用
  `setmaxnreg` 让轻/重 warpgroup 共驻同一 kernel 并发，H≤192 时反超 `separate`）——即
  **不切 SM，而在 SM 内时分/分 warp**。这正是本节负结果的镜像：重叠的载体必须是**共享的 SM**，
  不是**分割的 block**。
- **判据（C1–C5，`docs/00` §3）**：`serial` 是 T1；`barrier` 满足 C1/C2/C4（单 launch、常驻、
  kernel 内协调）；`pipe` 额外把依赖做细（C3），但 `Gp`/`Gc` 是**静态**角色，且无去中心化调度
  （无 C5）——三者都**不是**真正的 mega-kernel 调度。
- **对生产案例（`docs/02`）的解释**：DeepGEMM mega 家族在同一 kernel 内让**固定 warp 角色**
  各司其职（TMA warp / MMA warp / scheduler warp），用**异步原子 ring barrier**协调——正是
  「SM 内共驻 + 批量同步」，而非「按 block 切算子」。本节从反面量化了这一设计选择的必要性。

### 14.6 局限

- 合成 workload：上游是 `copy+1`/FMA、下游是 FMA 归约，**不含张量核 / 真实算子 / 真实通信**；
  测的是**依赖粒度与静态分区的机制**，非端到端收益。
- `pipe` 用 cooperative launch 保证 `Gp+Gc` 全常驻；producer/consumer 用**静态 block 角色**，
  未测**动态**任务分配（同一 block 池自取 produce/consume 任务，P15-1 的形态）。
- consumer 归约要求 `TS = nthreads × 16`，规模固定在 33.5M float；未扫 tile 大小（TS）对
  sync 开销/粒度的影响。
- 「理想重叠下界」`ideal` 用的是**单独测**的单阶段时间，未考虑两阶段并发时的**相互带宽/占用率
  干扰**（实测 `p/ideal > 1` 即含此干扰）；故 `p/ideal` 只能上界估计「并发干扰 + 同步开销」之和。
- 未测 warp 特化版的**共驻重叠**（那是 P13-1/P18-1 的领域），本节只覆盖「静态切 block」这一反面。

## 15. 去中心化调度的 DAG 形状敏感性：扇入/扇出 × 静态分配 vs 动态队列（P22-1）

### 15.1 问题

P15-1 用「C 条链」把 MPK §5 的去中心化调度（C5）跑了起来，但那是最**退化**的 DAG：每个 task
恰好 1 个前驱、1 个后继。P15-1 §9.6 自己留了尾巴——

> 每 block 用线程 0 单点出/入队，未做 MPK 的批量/预取；**未测 task 数远超队列容量、多级依赖扇入
> 扇出**（当前每 task 恰好 1 个后继）。

P20-1 §14 又从「跨算子静态分块重叠」的负结果里推出：**同步必须批量/环形化**（MPK ring buffer +
事件计数器、DeepGEMM 位掩码 ring barrier），而非每 task/每 tile 一次原子。本节把 DAG 从「链」推广成
**分层扇入/扇出图**，在同一张图、同一总工作量上公平对照**四种设备内协调机制**，回答三个子问题：

1. 扇入/扇出（每个 task 有 FIN 个前驱、fan-out 个后继）会不会改变各机制的相对排名？
2. 完全动态的「逐 task 原子 ready 队列」（P15-1）相对集中式 `grid.sync()`、相对「层粒依赖 +
   细粒分配」、相对「静态分配 + 事件计数」的**代价到底多大、能不能被 task 长度摊薄**？
3. MPK §5.2 的**批量取 task**（一个原子领 K 个）在什么 K 上有收益？

### 15.2 方法

程序 `src/fanin_sched.cu`（目标卡 H100 / sm90，固定空闲 device 3，原始输出
`harness/fanin_sched.out.txt`；`iters=20`，`G=132`（=SM 数，1 block/SM），`T=256`）。

**任务图（真实依赖，非合成屏障）**：`t = l*W + j`（层 `l∈[0,L)`、列 `j∈[0,W)`）。
`t` 的前驱 = 层 `l-1` 的 `{ (j*FIN+k) mod W : k<FIN }`，层 0 无前驱。数值
`val[t] = seg_sum(t 的 SEG 个输入) + Σ_{p∈pred(t)} val[p]`，末层写 `out[j]`。**同一求和顺序**，
故四种机制与串行参考应**逐位一致**（正确性门限 `fail=0`、`max rel|Δ|`）。

四种机制（全部单次 launch、常驻、设备内协调）：

| 机制 | 依赖粒度 | 分配方式 | grid.sync | 说明 |
|---|---|---|---|---|
| `barrier` | 层 | grid-stride | **每层 1 次** | 集中式，P5-2/P13-1 形态 |
| `wave` | 层（软屏障） | 原子工作计数器（可批量） | 无 | 「层粒依赖 + 细粒分配」，worker 自取；无 `grid.sync` |
| `decen` | **逐 task** | 设备侧原子 ready 队列 | 无 | P15-1 机制 + 扇入扇出；每 task 事件计数器 |
| `static` | **逐 task** | **静态 grid-stride**（无队列） | 无 | block b 固定处理 `t=b,b+G,…`，前驱事件计数器轮询 |

五个 Section：**A** 扫宽度 `W`（固定 L/FIN/SEG）；**B** 扫扇入 `FIN`（固定 L/W）；
**C** 扫 `wave` 批量 `K`（固定 L/W/FIN）；**D** 扫 task 长度 `SEG`（固定 L/W/FIN）；
**E** 扫层数 `L`（薄层，固定 W/FIN/SEG）。

### 15.3 结果

**Section A：扫宽度 W（L=16，FIN=2，SEG=4096；时间 ms）**

| W | tasks | barrier | wave | decen | static | wave/b | decen/b | static/b |
|---|---|---|---|---|---|---|---|---|
| 256 | 4096 | 0.283 | 0.289 | 0.874 | 0.301 | 0.979× | **0.323×** | 0.940× |
| 512 | 8192 | 0.490 | 0.544 | 1.761 | 0.574 | 0.900× | **0.278×** | 0.854× |
| 1024 | 16384 | 0.955 | 1.053 | 3.454 | 1.127 | 0.907× | **0.276×** | 0.847× |
| 2048 | 32768 | 1.874 | 2.055 | 6.958 | 2.241 | 0.912× | **0.269×** | 0.836× |

**Section C：wave 批量 K（L=16，W=1024，FIN=2）**

| K | ms | vs K=1 |
|---|---|---|
| 1 | 1.053 | 1.000× |
| 2 | 1.028 | 1.024× |
| 4 | 1.024 | **1.028×** |
| 8 | 1.029 | 1.023× |
| 16 | 1.986 | 0.530× |
| 32 | 3.900 | 0.270× |

**Section B：扫扇入 FIN（L=16，W=1024，SEG=4096）**

| FIN | barrier | wave | decen | static | wave/b | decen/b | static/b |
|---|---|---|---|---|---|---|---|
| 1 | 0.952 | 1.054 | 3.490 | 1.118 | 0.903× | 0.273× | 0.852× |
| 4 | 0.955 | 1.092 | 3.455 | 1.135 | 0.874× | 0.276× | 0.842× |
| 8 | 0.987 | 1.154 | 3.507 | 1.203 | 0.855× | 0.281× | **0.821×** |

**Section D：扫 task 长度 SEG（L=16，W=512，FIN=2）**

| SEG | barrier | wave | decen | static | wave/b | decen/b | static/b |
|---|---|---|---|---|---|---|---|
| 4096 | 0.490 | 0.544 | 1.761 | 0.574 | 0.900× | **0.278×** | 0.854× |
| 16384 | 1.675 | 1.706 | 2.226 | 1.724 | 0.981× | 0.752× | 0.971× |
| 32768 | 3.237 | 3.252 | 3.387 | 3.254 | 0.995× | **0.956×** | 0.995× |

**Section E：扫层数 L（薄层，W=256，FIN=2，SEG=8192）**

| L | tasks | barrier | wave | decen | static | wave/b | decen/b | static/b |
|---|---|---|---|---|---|---|---|---|
| 32 | 8192 | 0.909 | 0.950 | 1.931 | 0.958 | 0.957× | 0.471× | 0.949× |
| 64 | 16384 | 1.810 | 1.881 | 3.823 | 1.899 | 0.962× | 0.473× | 0.953× |
| 128 | 32768 | 3.612 | 3.742 | 7.647 | 3.781 | 0.965× | 0.472× | 0.955× |

**正确性**：全部 12 个 case 四机制均 `fail=0`；vs 串行参考 `max rel|Δ|` 在 `6e-8`–`1.2e-7`
（纯浮点累加序）；四机制**互比**在同一 case 内比值恒定（同序求和）。

### 15.4 读法

**(1) 扇入/扇出下 C5 机制依然正确。** 四机制在 12 个 case 上全部 `fail=0`，说明「逐 task 事件
计数器」（`decen`/`static`）与「层软屏障」（`wave`）都能正确表达 FIN 路汇聚依赖——P15-1 的 C5
机制不是链特例。

**(2) 在严格分层的 DAG 上，集中式 `barrier` 几乎总是最快。** Section A/B/E 里 `barrier` 恒为
1.000×。原因有两层：**其一**，分层图的每层都有 `W≥256` 个 task、每层工作量 `W×SEG` 通常
≥ 几微秒，而 `grid.sync()` 地板在 132 block 上只有 ~1.1 µs（P10-1 §5.3），**每层同步的开销占比
很小**；**其二**，严格分层意味着「层 `l+1` 必须等层 `l` *全部*完成」——这**正是** `grid.sync()`
语义，于是没有任何跨层重叠空间可利用。Section E 把层数从 32 加到 128（每层工作量不变），
`barrier` 仍稳赢（`wave`/`static` 0.95–0.97×、`decen` 0.47×）——**「多而薄」本身不构成
`grid.sync()` 的致命伤，只要每层仍有足够工作量**。这与 P15-1 §9.4 一致：去中心化要赢在
**均衡/结构**，不在「省屏障」。

**(3) 完全动态的「逐 task ready 队列」（`decen`）在小 task 上代价极大，且是每 task 固定开销。**
Section A 里 `decen` 只有 **0.27–0.32×**（比 `barrier` 慢 3–3.7×），随 `W` 增大不收敛——因为
每个 task 要付：队列头 `atomicCAS`、后继 `atomicSub` + `__threadfence` + 队列尾 `atomicAdd` +
`atomicExch` 发布、以及每 task 一次的 block 级 `__syncthreads()`。这些是**每 task 固定**的。
Section D 证明这一点：固定 `W=512`、把 `SEG` 从 4096 拉到 32768（task 变长 8×），
`decen/b` 从 **0.278×** 单调升到 **0.956×**——**只要 task 足够长，动态队列的固定开销就被摊薄到
可忽略**。这给 P15-1「decen 小负载下慢到 0.556×」一个更一般的解释：不是「去中心化不行」，而是
**每 task 的调度原子/栅栏成本相对于 task 计算量太大**。生产上 MPK/DeepGEMM 的每个 task 是一个
tiled GEMM / 一段算子（远大于这里 4096 float 的归约，`docs/07` §1），所以这套机制在真实粒度下成立。

**(4) MPK §5.2 的批量取 task：甜点区在 K ≈ W/G，超过就崩。** Section C 里 `wave` 的
`atomicAdd(work_cnt, K)` 一次领 K 个，K=2–8 有 ~2–3% 增益（per-task 原子被摊掉），但 K=16
骤降到 **0.530×**、K=32 到 **0.270×**。原因是每层可领 task 数 `W/G = 1024/132 ≈ 7.8`——K 超过
「每 worker 平均每层 task 数」后，一次领走 K 个的 worker 把别人的活也领光了，**每层末尾大量
worker 空转**（领到的越少/领不到的越早退出本层、却领不到下一层）。这给出批量调度的经验律：
**批量大小应对齐每 worker 每层的平均 task 数（`W/G`），不是越大越好**。MPK 的 scheduler/worker
角色划分（§5.1，`docs/06` §3–§4）正是让取 task 的并发度与 worker 数匹配。

**(5) 静态分配（`static`）优于动态队列但劣于 `wave`/`barrier`，且随扇入变差。**
`static` 无中央队列、无每 task 的出/入队原子，只有「前驱事件计数器轮询 + 后继 `atomicSub`」，
于是比 `decen` 快得多（0.82–0.995×）。但它的代价随**扇入 `FIN` 上升**（Section B：
0.852→0.842→**0.821×**）：每完成一个 task 要 fan-out 次 `atomicSub`，每个 task 要轮询 FIN 个
计数器，且**没有 work-stealing**——一旦某 block 手上的 task 因前驱未就绪而阻塞，它所在的 SM
只能空等（它的其它 task 也全在更低的 tt 上、同样可能阻塞）。`wave` 的可自取工作计数器恰好补了
这一点，所以 `wave ≥ static`（Section A/B/E）。

**(6) 一个真实的内存序坑。** 最初 `static` 用「写 `val` → `__threadfence()` → `atomicExch(done)`」
发布、「轮询 `done` → `__threadfence()` → 读 `val`」消费，在 `FIN=8` 等大扇入 case 上**间歇性
读错**（`max rel|Δ|` 达 `1e-2`–`1e0`，即读到未发布的 `val`）。把依赖改成与 `decen` 相同的
**逐 task 事件计数器（`atomicSub` 归零）** 后就稳定了。教训与 MPK 的实现一致：跨 task 依赖用
**原子计数/事件**这一条原语表达（MPK §5.1 的 event、`docs/06` §3 的 `num_triggers`），
而不是自己拼「数据写 + 独立 flag 发布」（P20-1 §14 也建议同步要批量/环形化，而非散落的 flag）。

### 15.5 与判据 / MPK / 生产案例的关系

- **补齐 P15-1 §9.6 的两条尾巴**：把「链」推广到**扇入/扇出分层图**（测了 FIN/扇出、`W`、`L`），
  并实现/量化了 MPK §5.2 的**批量取 task**（Section C）。结论：事件计数器机制在任意扇入/扇出下
  正确；批量甜点在 `K≈W/G`。
- **与 P15-1 的关系**：P15-1 证「decen ≈ barrier（均衡）/ decen 赢（异构）」；本节补上**机制的
  代价标度**——`decen` 的每 task 固定开销在多 task、短 task 时让它在**严格分层均衡图**上远慢于
  屏障（0.27×），只有当 task 足够长（Section D）或负载异构（P15-1 §C）时才回本。两节合起来：
  **去中心化调度的价值来自「动态均衡 + 长 task 摊薄」，不是来自「去掉全局屏障」。**
- **对上 MPK §5（快照行 1147–1368）**：`wave` 是本文档第一次把「layer 粒依赖 + 原子工作计数器 +
  软屏障」这一介于 `grid.sync()`（集中式）与 `decen`（逐 task 动态）之间的形态跑出来——它对应
  MPK 编译器把 tGraph **分层线性化**（§4.1 linearlization、`docs/08` §5）后，运行层只需在
  **层边界**协调、层内自由自取的场景。`decen` 对应 MPK 的**完全动态 ready 队列**；
  `static` 对应 DeepGEMM 的**静态 warp 角色 + 位掩码依赖**（`docs/02` §1.5）——本节量化了三者的
  代价差。
- **与 P13-1 / P14-1 / P16-1 / P18-1 呼应**（成本侧）：本节是「调度机制×DAG 形状」的代价面；
  那些是「资源/代码足迹」的代价面。共同结论：**megakernel 的价值不在「省 launch/省屏障」，
  而在长 task、异构负载下的动态均衡与资源共驻**。
- **判据（C1–C5，`docs/00` §3）**：四种机制都满足 C1（单 launch）+C2（常驻）+C3（多 task）
  +C4（设备内协调）；`decen`/`static`（逐 task 依赖）与 `wave`（层边界软屏障）都**无固定角色的
  worker/scheduler**，故严格说仍差 C5 的「固定角色去中心化调度」一步；`barrier` 是集中式（只有
  C1/C2/C4）。四者都是 T4 的**机制骨架**，用于隔离「协调机制」这一变量。

### 15.6 局限

- 合成 workload：task 是 `SEG` 个 float 的归约 + FIN 路加法，**不含张量核/TMA/真实通信**；
  绝对倍率不代表真实模型，测的是**调度机制的 DAG 形状敏感性**。
- G 固定为 `#SM=132`、1 block/SM、`T=256`；未扫 `G`/`blockDim`（P10-1 已覆盖网格标度）。
- 分层 DAG **每层等宽**（除由 FIN 引起的 fan-out 差异），**无「长短链混合」的异构**——那正是
  P15-1 §C 赢 barrier 的场景，本节不重复；故「barrier 恒快」的结论**仅对严格分层的均衡图成立**。
- `wave` 的软屏障是「轮询层完成计数器」，本质仍是**该层的会合**，故它无法利用跨层重叠；真正的
  「跨层流水」需要环形缓冲/预取（MPK §5.3），本节未做。
- `static` 的阻塞式轮询未做退避调优（固定 `__nanosleep(64)`），其绝对时间对 sleep 粒度敏感。
- 未用 `ncu` 归因原子/栅栏流量；未与 CUDA Graph、PDL 组合对照。

## 16. 跨层软件流水 + 环形缓冲预取：kernel-per-operator 与持久化内核的真正分野（P23-1）

### 16.1 问题

P22-1 §15.6 留了一条明确的尾巴：

> `wave` 的软屏障是「轮询层完成计数器」，本质仍是**该层的会合**，故它无法利用跨层重叠；真正的
> 「跨层流水」需要环形缓冲/预取（MPK §5.3），本节未做。

这正是 MPK §5.3（快照行 1312–1368）第二项运行时优化 **cross-task software pipelining**，也是
§1（行 100）/§2.1（行 226）/§2.2（行 313–331）反复强调的「**kernel-per-operator 做不到、必须融合进
单内核才能做**」的那条机制：

> 在 kernel-per-operator 模型里，每个 kernel 都假设**独占**自己 block 的 shared memory；一旦
> kernel 结束，SMEM 立刻释放。这个假设**阻断了跨 task 软件流水**——它是把「下一个 task 的数据载入」
> 与「当前 task 的计算」重叠，而两个 task 可能**同时需要 shared memory**（快照行 1312–1320）。
> MPK 用 **paged shared-memory**（SMEM 分成等长页、task 按页 acquire/release、且 release 后不可再
> acquire 的单调模式）解除该假设，使当前 task 释放页后即可为下一 task 预取（行 1320–1325）。
> 跨 task 软件流水把 task 分成 **pre-loading 阶段**（只发射数据搬运指令到 SMEM）与 **compute 阶段**，
> 当 (1) 当前 task 已发射完自己的全部搬运、且 (2) 有足够空闲页给后继 task 时，就把 $T_1$ 的 compute
> 与 $T_2$ 的 pre-loading 重叠（行 1326–1345）。§6.6 在 Qwen3-8B 末层上实测这条优化带来
> **1.2–1.3×**（行 1586–1599）。

P13-1（§7）测的是**单 task 内部**的 tile 级流水（损失一）；P20-1（§14）测的是**静态切 block**的
跨算子重叠（结论：赢不过算子级屏障，重叠载体必须是 SM 内共驻）。本节补最后一格：**单个 worker 上、
跨 task/算子边界的软件流水**——在**严格跨层依赖**下，能否「边算层 l、边预取层 l+1 的权重 tile」？
回答三个子问题：

1. **单内核**（barrier/decen/ring）相对 **kernel-per-operator**（eager）到底快多少？其中多少来自
   消 launch、多少来自跨层流水？
2. 跨层预取的**收益随「访存/计算比」如何变化**？纯访存（无可藏的计算）时是否反而更慢？
3. **环形深度 K**：越深越好，还是像 P13-1 那样 1→2 就到头？

### 16.2 方法

程序 `src/cross_layer_pipeline.cu`（目标卡 H100 / sm90，固定空闲 device 3，原始输出
`harness/cross_layer_pipeline.out.txt`（E=16 主运行）与 `harness/cross_layer_pipeline_e32.out.txt`
（E=32 重载敏感性）；`iters=30`，`L=64` 层，`G=132`（=SM 数，1 block/SM），`T=256`，
每线程 `E` 个 float（`E=16` → tile 16KB；`-DELEM=32` → tile 32KB），权重总量 528MB / 1056MB ≫ L2）。

**依赖模型（真实依赖，模拟一层算子里「私有权重 tile（可预取）+ 跨 worker 的激活值（依赖前层）」）**：
层 `l∈[0,L)`、worker `j∈[0,G)` 的 task：
- **私有 bulk**：cp.async 载入权重 tile `W[l][j]`（`T*E` 个 float）——**与 DAG 无关、可预取**；
- **跨层依赖**：读激活 `act[l][(j+1) mod G]`（由 worker `(j+1) mod G` 在层 `l-1` 写出）——**必须等**；
- **计算**：对 SMEM tile 做 `E` 条链 × `R` 轮 FMA，块内归约 → `s`；写
  `act[l+1][j] = s + act[l][(j+1) mod G]`。

worker=block，`W=G`（每层每 worker 一个 task）。层 `l` 的依赖落在**相邻 worker**上（既有纵向
跨层、也有横向跨 worker），逼出真实的跨 SM 协调，而非纯本地链。四种策略算**同一张图**，故数值应
**逐位一致**（交叉对比门限）：

| 策略 | 单次 launch | 常驻 | 依赖协调 | 跨层预取 | 对应 |
|---|---|---|---|---|---|
| `eager` | 否（L 次） | 否 | stream 串行 | **不可**（SMEM 随 kernel 释放） | kernel-per-operator 基线 |
| `barrier` | 是 | 是 | 每层 `grid.sync()` | 无 | P5-2/P22-1 `barrier` |
| `decen` | 是 | 是 | 逐槽事件计数器（`atomicAdd` 轮询） | 无 | P15-1/P22-1 `decen` |
| `ring` | 是 | 是 | 逐槽事件计数器 | **K 页环形 SMEM 预取**（MPK §5.3） | P23-1 |

`ring` 在每个 worker 的**固定 task 序列**（层 `0..L-1`，同一 `j`）上做 K 页环形预取：先预取层
`0..K-2` 的权重 tile，之后每进入层 `l` 就把层 `l+K-1` 的 tile 发射进重用的环形页（该页上一轮计算
已在 `__syncthreads` 后结束），再等前驱事件、`__pipeline_wait_prior` 等本层页就绪、计算。**预取
发射在事件等待之前**，故「下一层（乃至更后面几层）tile 的 HBM 延迟」与「本层的前驱等待 + 计算」
重叠。

三个 Section：**A** 扫计算强度 `R`（0…256）；**B** 扫环形深度 `K`（取访存占比较高的 `R=16`）；
**C** 扫层数 `L`（8…256）。

### 16.3 结果

**Section A：扫计算强度 R（L=64，G=132；时间 ms；`ring` 用 K=4）**

E=16（tile 16KB）：

| R | eager | barrier | decen | ring | d/b | r/b |
|---|---|---|---|---|---|---|
| 0 | 0.259 | 0.171 | 0.154 | 0.178 | 0.900× | **1.036×** |
| 8 | 0.317 | 0.231 | 0.209 | 0.195 | 0.907× | **0.844×** |
| 16 | 0.347 | 0.253 | 0.235 | 0.221 | 0.930× | 0.876× |
| 32 | 0.407 | 0.309 | 0.293 | 0.279 | 0.947× | 0.903× |
| 64 | 0.527 | 0.426 | 0.407 | 0.396 | 0.954× | 0.929× |
| 128 | 0.762 | 0.658 | 0.639 | 0.630 | 0.971× | 0.957× |
| 256 | 1.239 | 1.129 | 1.098 | 1.095 | 0.972× | 0.970× |

E=32（tile 32KB，访存更重）：

| R | eager | barrier | decen | ring | d/b | r/b |
|---|---|---|---|---|---|---|
| 0 | 0.396 | 0.301 | 0.257 | 0.414 | 0.856× | **1.376×** |
| 8 | 0.607 | 0.510 | 0.462 | 0.425 | 0.905× | **0.834×** |
| 16 | 0.619 | 0.515 | 0.472 | 0.434 | 0.917× | **0.842×** |
| 32 | 0.694 | 0.580 | 0.542 | 0.506 | 0.933× | 0.871× |
| 64 | 0.939 | 0.795 | 0.763 | 0.744 | 0.959× | 0.936× |
| 128 | 1.429 | 1.261 | 1.215 | 1.204 | 0.964× | 0.955× |
| 256 | 2.369 | 2.208 | 2.122 | 2.120 | 0.961× | 0.960× |

**Section B：扫环形深度 K（L=64，R=16）**

E=16（轻载）：K=1→8 的 `r/d` 恒在 0.933–0.941×（**深度无影响**）。

E=32（重载）：`r/d` = K1 **0.970×** / K2 0.944× / K3 0.924× / K4 **0.919×** / K6 0.924×
（K8 跳过，SMEM 超单 block optin 上限）——**深度 K=1→4 单调增益、K≈4 饱和**。

**Section C：扫层数 L（R=64，ring K=4）**

| L | eager | barrier | ring | r/b | eager/barrier |
|---|---|---|---|---|---|
| 8 | 0.060 | 0.053 | 0.051 | 0.970× | 1.14× |
| 16 | 0.120 | 0.103 | 0.099 | 0.957× | 1.16× |
| 64 | 0.527 | 0.426 | 0.396 | 0.929× | 1.24× |
| 256 | 2.113 | 1.696 | 1.554 | **0.916×** | **1.25×** |

（E=32 时 L=256：eager 3.766 / barrier 3.171 / ring 2.939，`r/b`=0.927×、eager/barrier=1.19×。）

**正确性**：四个 Section 的策略在两种 `E` 下对 `act[L][*]` **逐位一致**（互比 `max rel|Δ|=0.00e+00`），
说明「逐槽事件计数器」的细粒度协调与环形预取**不影响数值**。

### 16.4 读法

**(1) 单内核相对 kernel-per-operator 的主要收益仍是「消 launch」，不是「消同步」。** Section C 里
`eager/barrier` 随层数从 1.14×（L=8）升到 **1.25×**（L=256）——这正是 P5-1/P9-1 的损失二（每层一次
launch 的固定成本被层数摊到总时间里）。`barrier` 与 `eager` 的**设备侧工作相同**，差额就是 L 次
launch。与 P5-2「launch 只在 launch-bound 区间有意义」一致：层越多、每层越短，launch 占比越高。

**(2) 细粒度事件调度（`decen`）稳定优于每层 `grid.sync()`（`barrier`）约 3–14%。** `d/b` 在两种 E
下都是 0.856–0.972×，且**访存越重、优势越大**（E=32 Row0 `d/b`=0.856×）。原因是 `barrier` 每层都要
付一次全图会合（P10-1 实测 `grid.sync()` ~1.1µs 地板），而 `decen` 只在**相邻 worker** 之间用事件
计数器精确等前驱。这也解释了 P22-1「严格分层均衡图上 barrier 恒快」——那张图每层 `W≥256` 个 task、
每层工作量 ≫1.1µs；本节 `W=G=132`、每层只有**一个** task/worker，`grid.sync()` 的固定成本就显出来了。

**(3) 跨层预取（`ring`）相对 `decen` 的增益 ≈「访存占 pk 的比例」，且随计算强度衰减。** Section A：
E=16 Row8 `r/b`=0.844×（ring 比 barrier 快 **1.19×**）、`r/d`≈1.07×；到 R=256（计算主导）只剩
0.970×（1.03×）。机制清楚：ring 把「下一层权重 tile 的 HBM 延迟」藏进「本层的计算」，可藏的部分就是
访存时间占串行时间 `(Tload+Tcomp)` 的比例，其上界 `(Tload+Tcomp)/max(Tload,Tcomp)`。**E=32（tile 翻倍、
访存占比更高）时 `r/b` 峰值到 0.834×（快 1.20×）**——与 MPK §6.6（行 1593）报告的 **1.2–1.3×** 同一量级。

**(4) 纯访存（R=0）时流水净亏。** 两行 `R=0` 的 `r/b` 都 >1（E=16 1.036×、E=32 **1.376×**）：没有
计算可以「藏」访存，环形预取只是多占 SMEM、多一层 `__pipeline_wait_prior`/同步开销。这与 P13-1 §7
的观察互补——**软件流水的收益来自「用计算盖住访存」，不是凭空加速访存**。

**(5) 环形深度 K 的收益取决于 task 的访存占比。** 轻载（E=16）下 K=1→8 的 `r/d` 几乎不动
（0.933–0.941×）：因为**前驱事件等待本身已经给了足够 slack** 把 16KB 的 tile 载入藏掉，再加深环形
没有额外东西可藏。重载（E=32）下 K=1→4 单调改善（0.970→0.919×）、K≈4 饱和——**访存越重，越需要
更深的环形预取**。这条比 P13-1「深度 1→2 就到头」更细：**「到几层」由单 task 的访存延迟与同步
slack 共同决定**，不是常数。生产上 MPK/DeepGEMM 用 `Kstages=3/6`（`docs/07` §1.2）正落在这个
「够深即饱和」的区间。

### 16.5 与判据 / MPK / 生产案例的关系

- **补齐 P22-1 §15.6 的尾巴**：把「跨层流水需要环形缓冲/预取」实测出来——跨层预取在访存占比高时
  可达 **1.19–1.20×（vs barrier）**，深度按 task 访存占比选 K≈4，纯访存时净亏。判据上，`ring`
  是满足 C1（单 launch）+C2（常驻）+C3（多 task）+C4（设备内协调）的**持久化跨 task 流水骨架**；
  它按静态固定序列取 task，**没有固定角色的 worker/scheduler**，故仍差 C5 一步（C5 的去中心化
  调度由 P15-1 §9 覆盖）。它把 MPK §5.3 的 paged SMEM（行 1312–1325）与 cross-task software
  pipelining（行 1326–1345）在原生 CUDA 上跑了最小实现。
- **与 P13-1（§7）的关系**：P13-1 是**单 task 内** tile 级流水（K=1→2 拿到 1.18×）；本节是**跨
  task/算子边界**，两者叠加的方式不同——本节 `ring` 的收益来自「不同层的 task 共享常驻 worker 的
  SMEM 生命周期」，这是 kernel-per-operator **结构上做不到**的（`eager` 列里根本没有这条路径）。
  两者合起来覆盖 MPK §5.3 说的「intra-task + cross-task」双层软件流水。
- **与 P20-1（§14）的关系**：P20-1 证明跨算子重叠不能靠**静态切 block**（两阶段都要整台机器、
  逐 tile 同步开销大）；本节证明靠**同一 worker 上的环形预取**可以——**重叠的载体是「SM 内共驻 +
  异步拷贝」，不是「分割 block」**。这正面支持 P20-1 §14.4 的结论。
- **与 P15-1/P22-1（§9/§15）的关系**：那两节测「调度机制」的代价面（谁去取 task、怎么传依赖）；
  本节固定调度（静态序列 + 事件计数），只测**预取/流水**这一条运行时优化。三者共同画全
  MPK §5「in-kernel runtime」的 mechanism × DAG × prefetch 三维代价图。
- **对上 MPK**：`ring` 的「K 页环形 SMEM + 单调 acquire/release」就是 §5.3 paged SMEM 的最小化；
  「pre-load 阶段先行、compute 阶段重叠」就是 cross-task software pipelining 的直译。实测倍率
  （1.19–1.20× / eager 单内核 1.25×）落在 §6.6 报告区间内。
- **对上生产案例**：DeepGEMM mega 家族（`docs/02` §1）用**静态 warp 角色 + 双/多缓冲 SMEM 环 +
  异步原子 barrier**，其 SMEM 环正是 paged SMEM 的手工版；本节给出「为什么必须把连续 task 放进
  同一个 kernel」的量化理由。

### 16.6 局限

- 合成 workload：task = `T*E` 个 float 的载入 + `E` 条链 ×`R` 轮 FMA，**不含张量核/TMA/真实通信**；
  绝对倍率不代表真实模型，测的是**跨层流水的机制与标度**。
- 权重视为**与 DAG 无关、可预取**；真实图里只有部分输入（权重/静态激活）可预取，依赖链上的数据
  （前层输出）不可预取——本模型把这一区分显式化（权重 bulk vs 跨层激活值）。
- `G=#SM=132`、1 block/SM、`T=256`；`W=G`（每层每 worker 一个 task）。未扫 `G`/`T`/`W>G`
  （P10-1 已覆盖网格标度；W>G 的批内分配是 P22-1 的批量取 task 范畴）。
- `eager` 用 L 次普通 launch，**未与 CUDA Graph capture 对照**（P4-1/P5-1/P19-1 已证 graph 只消
  host 提交、不破依赖串行、不消 kernel barrier）；加 Graph 只会抬高 `eager` 基线、放大本节结论。
- 环形页复用依赖 `__syncthreads` 保护，未用 `mbarrier`/`cp.async.bulk` 等更新的异步原语
  （`docs/07` §2 的 Hopper 形态）；故 `ring` 的同步开销偏保守。
- 未用 `ncu` 归因 HBM 利用率/`cp.async` 停顿，未测 `E`/SMEM 页大小连续扫描（只测 E=16/32 两点）。

## 17. 损失三的直接字节证据：用 ncu 量化「融合消掉多少 HBM 流量」（P24-1）

### 17.1 问题：损失三此前只有「推断」，没有「测量」

`docs/00` §1 把 kernel-per-operator 的第三类损失定义为**中间张量的 HBM 往返**：每个算子把结果写回
HBM、下一个算子再读回来，往返一趟就是 `2 × 张量大小` 的 DRAM 流量。但本专题此前只**间接**验证过它：

- P5-2（§2）用「表观带宽 × 假设字节数」反推：三种经 HBM 中间量的策略表观带宽都 ~2.8 TB/s，
  于是推断 fused 的 2× 收益来自「16 → 8 B/元素」的流量差。这是**算术推断**，不是流量实测。
- P9-1（§4）测到时间优势随算子数 `N` 近似线性放大（`e/fused` 0.99→7.13×），但那是**时间**标度，
  没有把「多出来的时间 = 多出来的 DRAM 字节」对上。

本节用 `ncu` 的 `dram__bytes_read.sum` / `dram__bytes_write.sum`（DRAM 物理读写字节）把这条损失
**直接测成硬数字**，并验证标度律：**中间张量 HBM 往返的字节数随融合的算子数线性增长。**

### 17.2 实验设置

构造 `Q` 个逐元素算子串成的依赖链 `y_q = f(y_{q-1})`（`y_0 = x`，`f(v)=fma(v,1.5,0.25)`，`float4`），
`N = 64Mi` 元素 = **256 MB/数组 ≫ 50 MB L2**（保证中间量真的落 DRAM）。四种策略公平对照：

- **eagerQ**：`Q` 个独立 kernel（kernel-per-operator），每步读张量、写张量，中间量落 HBM；
  用两个 scratch buffer ping-pong。
- **graphQ**：把上述 `Q` 个 kernel capture 成一张 CUDA Graph，一次 replay。
- **barrier**：单个 persistent kernel，逐算子 `grid.sync()`，中间量仍落 HBM scratch。
- **fused**：单个 persistent kernel，逐元素把 `Q` 个算子算完，中间量只活在寄存器里。

源程序 `src/dram_traffic.cu`，原始输出 `harness/dram_traffic.out.txt`（计时）与
`harness/dram_traffic_ncu.out.txt`（ncu 字节 + 原始 CSV）。`one <strategy> <Q>` 子命令只跑一遍策略，
`ncu -k regex:dt_` 只抓目标 kernel（排除初始化 `warm_fill`）。

### 17.3 计时：fused 与 Q 无关，eager/barrier 随 Q 线性

`harness/dram_traffic.out.txt`（iters=30，固定 device 3）：

```text
   Q |    eagerQ |    graphQ |   barrier |     fused |     e/g |     b/e |     f/b
-----+-----------+-----------+-----------+-----------+---------+---------+--------
   1 |    0.1879 |    0.1871 |    0.1917 |    0.1883 |   1.00x |   1.02x |   0.98x
   2 |    0.3755 |    0.3739 |    0.3805 |    0.1884 |   1.00x |   1.01x |   0.50x
   4 |    0.7511 |    0.7468 |    0.7578 |    0.1884 |   1.01x |   1.01x |   0.25x
   8 |    1.5020 |    1.4919 |    1.5203 |    0.1889 |   1.01x |   1.01x |   0.12x
```

- **fused 恒 ~0.1885 ms**，与 `Q` 无关（单遍读 + 单遍写）；eager 恰好 `≈ Q × 0.188 ms`，
  `f/b = 1/Q`（0.98→0.50→0.25→0.12）。
- `e/g ≈ 1.00–1.01×`：256 MB/数组下 kernel 已带宽主导，**CUDA Graph 的 host 提交收益 ≈ 0**
  （与 P5-1/P5-2 一致：损失二只在 launch-bound 区间显形）。
- `b/e ≈ 1.01–1.02×`：`grid.sync()`（660 block）相对 `Q` 次 launch 无可测优势，退化解。
- 正确性四档 `max|z_eager − z_fused| = 0.000e+00`。

### 17.4 ncu 实测 HBM 字节：融合消掉的流量恰好随 Q 线性

`harness/dram_traffic_ncu.out.txt`（每策略每 Q 对目标 kernel 求和）：

```text
   Q    strat      read_MB     write_MB     total_MB   B/elem   vs_fused
   1    eager        268.4        249.9        518.3     7.72      1.00x
   2    eager        536.9        497.1       1034.0    15.41      1.99x
   4    eager       1073.8        992.0       2065.8    30.78      3.98x
   8    eager       2147.6       1982.1       4129.6    61.54      7.96x
   2    graph        536.9        497.2       1034.1    15.41      1.99x
   4    graph       1073.8        992.3       2066.1    30.79      3.98x
   2  barrier        536.9        519.0       1055.9    15.73      2.04x
   4  barrier       1073.8       1056.3       2130.0    31.74      4.11x
   8  barrier       2147.5       2129.7       4277.2    63.74      8.25x
   1..8 fused    268.4..268.5 250.0..250.3 518.5..518.8  7.73      1.00x
```

三点结论：

1. **fused 的 DRAM 流量是常量**：`read 268.4 MB + write 250.0 MB ≈ 518.5 MB`，即
   `≈ 2 × N × 4 B = 8 B/元素`（读略高 4.85%、写略低 2.3%，是 HBM3 读写粒度/ECC 与 L2 回写效应，
   非模型误差）。
2. **eager/graph 的流量 ≈ `Q × 517 MB`**，`vs_fused` = 1.99 / 3.98 / 7.96（`Q=2/4/8`）——
   **精确等于 `Q`**。即每多融合一个算子，就消掉**一趟完整的全张量 HBM 往返**（读 256 + 写 256 MB）：
   `Q` 个算子里，共享的只有「首次读 x」与「末次写 z」，中间 `(Q-1)` 对读写全部是冗余流量。
3. **barrier 与 eager 字节几乎相同**（仅写多 2–4%）：`grid.sync()` 只省 launch，**完全不省流量**。
   这正说明 `barrier` 量的是损失二、`fused` 量的是损失三——两档在**时间与字节上都正交**。

### 17.5 时间与字节严格成正比：融合收益 = 纯 HBM 流量收益

把字节数除以时间，两档处于**同一条 HBM 带宽线**上：

```text
Q=4:  eager 2065.8 MB / 0.7511 ms = 2.75 TB/s   fused 518.5 MB / 0.1884 ms = 2.75 TB/s
Q=8:  eager 4129.6 MB / 1.5020 ms = 2.75 TB/s   fused 518.5 MB / 0.1889 ms = 2.75 TB/s
```

即 **fused 相对 eager 的加速完全等于它省下的 DRAM 字节比**（`b/f ≈ 4.0×` 时间 vs `4.11×` 字节，
`≈ 8.0×` vs `8.25×`），表观带宽恒 ~2.75 TB/s——与 P5-2 推断的 ~2.8 TB/s 吻合。这把 P5-2 的
**算术推断**升级为**字节实测**：损失三不是一个「拟合出来的倍率」，而是一条可被 profiler 直接
数出来的 HBM 流量。

### 17.6 与判据 / MPK / 生产案例的关系

- **补齐 P5-2 §2.6 / §4 的「未直接量 HBM 字节」缺口**：P5-2 只给了 `barrier/fused ≈ 2×` 的时间
  倍率（两个算子），本节把算子数扫到 8，并用 `dram__bytes_*.sum` 直接证明
  `eager_bytes / fused_bytes = Q`、`time ∝ bytes`。
- **对上 MPK**：论文 §6.4（行 1551–1553）指出 MoE 的 fused gather-GEMM 里 gather 占 SGLang 的 11%
  ——那正是这类「中间量被显式写回再做下一步」的流量；MPK 把连续 task 放进单内核、中间量留在
  寄存器/SMEM（§2.2 行 313–331、§5.3 行 1312–1325 的 paged SMEM），消的就是本节测到的
  `2 × N × 4 B`/算子的往返。§5.3 的 cross-task software pipelining（行 1326–1345）之所以**只在
  单内核里可能**（P13-1 §7 / P23-1 §16），前提正是「中间量不需要落 HBM、可以跨 task 传递」。
- **对上生产案例**：DeepGEMM mega 家族（`docs/02` §1）把 dispatch+GEMM+SwiGLU+combine 等编进
  一个持久化 kernel，正是为了不再让每段算子把中间张量写回 HBM——本节给出「每融合一段省一趟
  全张量往返」的量化下界（本实验纯访存主导，是最乐观的收益上界；真实算子里还会叠加 tensor core 计算，
  故实际倍率小于 `Q`）。
- **与 P9-1（§4）的关系**：P9-1 从**时间**角度测到 `e/fused` 随 `N`（算子数）线性；本节从
  **字节**角度给出同一条标度律的因果——时间线性正是因为**字节线性**（`time = bytes / BWHBM`）。
  两节互为表里。
- **判据**：`fused` 满足 C1（单 launch）+C2（常驻）+C3（多算子）；`barrier` 再加 C4（设备内协调）；
  两者都**无固定角色 worker/scheduler**，仍差 C5 一步（C5 由 P15-1 §9 覆盖）。路径与
  P5-2/§2 的 T2→段级骨架一致。

### 17.7 局限

- **纯带宽主导的合成负载**：单算子只 1 条 FMA，故 fused 的收益取到理论上界（`Q×`）；真实算子
  含矩阵乘/归约/通信，fused 后可能转为计算主导，收益小于 `Q×`（P9-1 §4 已见 `N≥16` 的饱和）。
- **中间量必须 > L2 才有可测 DRAM 流量**：本实验 256 MB/数组；若张量能整块驻留 L2（≤50 MB），
  eager 的中间量往返可能命中 L2 而不落 DRAM，损失三会小于本节数字——**megakernel 的损失三
  在「工作集超过 L2」时才充分显现**。
- **读写字节与名义值有 ±5% 偏差**（HBM3 读写粒度/ECC/L2 回写），故 `B/elem` 约 7.7 而非 8.0；
  结论用「比例」（`vs_fused ≈ Q`）表述比绝对值鲁棒。
- `Q` 只扫到 8（再大只是同一线性律延伸）；只测 `float4`、只测单条逐元素链，未测非线性 DAG、
  未测 tile 级张量核场景。`barrier` 的写字节略高（+2–4%）未做进一步归因。
- 计时用 device 3、`iters=30`；ncu 用一遍 `one` 运行采样，单次字节读数自带 ±0.1% 抖动。

---

## 18. Thread Block Clusters 与分布式共享内存（DSM）：跨 SM「不落 HBM」的中间量交换原语（P29-1）

### 18.1 问题

前面所有实验里，跨算子/跨 SM 的中间量只有两条出路：**落 HBM（损失三，P24-1 §17 量过字节）**
或**靠 `grid.sync()` 在单内核里串起来（P13-1 §7 / P23-1 §16，但仍经 SMEM/HBM）**。而 §6.6
明确留了一条尾巴：

> `unspecified launch` 之外的 sm90 特性（如 cluster、`cudaLaunchAttributeProgrammaticEvent`）未覆盖。

Hopper 的 **Thread Block Cluster + 分布式共享内存（DSM）** 恰恰是把「跨 SM 传中间量」做进硬件
的原语：同一 cluster 内的 block 可用 `cluster.map_shared_rank(ptr, rank)` 直接**读邻居 block 的
SMEM**，中间量完全不落 L2/HBM。生产案例已经在用——DeepGEMM **Mega Gate** 一次发出多 CTA 并带
`cluster_dim = num_mma_ctas`（`docs/02` §1.2，`sm100_bf16_mega_gate.cuh:32` 一带），FlashMoE
的非阻塞 NVSHMEM（`docs/10`）也是同一思路的跨卡版本。

本节回答三件事：(1) DSM 远端读比本地 SMEM 贵多少？(2) 把「跨 SM 中间量」放 DSM vs 放
HBM/L2 往返，快多少？(3) cluster 的**容量/拓扑边界**在哪——能不能把整张机器做成一个 cluster？

### 18.2 方法

程序 `src/cluster_dsm.cu`，脚本 `harness/cluster_dsm.sh`，原始输出 `harness/cluster_dsm.out.txt`
（目标卡 H100 / sm90，SMs=132，**固定空闲 device 1**，nvcc 13.2）。用 `cudaLaunchKernelEx` +
`cudaLaunchAttributeClusterDimension` 设置 cluster 维；kernel 内用
`cooperative_groups::this_cluster()` 取 cluster、`map_shared_rank` 取邻居 SMEM 指针。
**每个配置单独一次进程调用**——一次 cluster launch failure 会污染 CUDA context（sticky error），
必须隔离，这也是复现脚本用 shell 循环逐配置调用的原因。

四个 Section：

- **A**：cluster 内读「环形邻居」的 SMEM（`src[threadIdx.x]`，每次一个 `volatile` load/线程/轮）。
  `C=1` 时 `(rank+1)%1=0` 退化为读**自己** SMEM = 本地基线；`C≥2` 是**远端**读。`grid=128`、
  16 KB/block、`R=20000`。
- **Lg**：plain 全局流式读 256 MB（> 50 MB L2）作 HBM 参照。
- **S**：跨 SM **ring 段位移**交换。C 个 block 组成 cluster，每 block 拥有 `seg=K·256` 个 float 的段，
  初值 `gid·1000 + j`；双缓冲 ping-pong，第 `r` 轮读邻居槽 `r%2`、写自己槽 `(r+1)%2`、末尾一次
  `cluster.sync()`，每轮把整段「向左搬一格」。段放 **global scratch（`Sh`，hbm）** 或**自己 SMEM
  （`Sd`，dsm）**。`R` 轮后 block `gid` 的段 = 初始时 `(rank-R)%C` 的段，可与解析参考**逐位比较**。
- **C1/C3**：`cluster.sync()` vs cooperative `grid.sync()` 成本；**C2/C2b**：cluster 容量/合法性。

### 18.3 结果

**A. DSM 远端读 vs 本地 SMEM（16 KB/block, grid=128, R=20000）**

```text
   C |  us/launch | ns/lap |   vs C=1
-----+------------+--------+----------
   1 |     56.33  |  2.817 |   1.000  （本地 SMEM 基线）
   2 |     55.89  |  2.795 |   0.992
   4 |     65.49  |  3.275 |   1.163
   8 |     65.87  |  3.294 |   1.169
  16 |     66.29  |  3.315 |   1.177
```

**DSM 远端读与本地 SMEM 读同量级**：`C=2` 甚至与本地一致（0.99×），`C≥4` 只慢约 **17%**。
plain 全局流式读（256 MB）只有 **690.8 GB/s**（HBM 参照）。

**S. ring 段位移：DSM vs HBM(scratch) 往返（grid=128, R=200, 1 屏障/轮）**

```text
 seg |  Sh hbm GB/s |  Sd dsm GB/s | dsm/hbm   （C=2；C=4/C=8 同值，见 out.txt）
-----+--------------+--------------+---------
 1KB |    299.2     |    316.1     |  1.06x
 2KB |    506.4     |    560.7     |  1.11x
 4KB |    779.0     |    912.4     |  1.17x
 8KB |   1048.6     |   1330.2     |  1.27x
```

四档 `maxerr = 0.0000`（与解析参考**逐位一致**，且跨进程可复现）。**DSM 交换恒快于
HBM/L2 往返，且随段增大优势放大到 1.27×**。注意 `Sh` 的 scratch 为 `grid·2·seg·4`（8 KB 时
仅 2 MB），**驻留 L2**，所以 `Sh` 量的是 **L2 往返**而非 HBM；`Sd` 的 1330 GB/s 已超过 L2
往返的 1049 GB/s。

**C1. 协调成本（grid=128, R=200）**

```text
 C  | cluster.sync ns/sync |   对照：grid.sync (grid=132) = 1113.3 ns/sync
----+----------------------+
  2 |        504.0         |   cluster.sync ≈ grid.sync / 2.2
  4 |        705.8         |
  8 |        730.3         |
 16 |        767.3         |   cluster.sync ≈ grid.sync / 1.5
```

`cluster.sync()` 比 grid-wide 的 `grid.sync()` 便宜 **1.5–2.2×**，且随 cluster 增大缓慢上升
（504→767 ns）——**群屏障 = 只同步 GPC 内的这几个 SM**。

**C2/C2b. cluster 容量/合法性**

```text
C=1,2,4,8,16（单 cluster）          -> ok（16 需要 non-portable 属性）
C=32（单 cluster）                  -> cluster misconfiguration（硬上限 16）
C=8, grid=132（132 % 8 != 0）       -> cluster misconfiguration（grid 必须是 C 的整数倍）
C=16, grid=64 / C=4, grid=132       -> ok
```

**cluster 有硬上限 16 个 block，且 grid 大小必须是 C 的整数倍。**

### 18.4 读法

1. **DSM 是「SMEM 级」的跨 SM 通道**：远端 SMEM 读只比本地 SMEM 贵 ~0–17%（`C=2` 几乎为零），
   远快于任何经 L2/HBM 的路径。这解释了 Hopper 上为什么可以把「邻居 block 的中间结果」当作
   本地 SMEM 用。
2. **交换介质决定一档收益**：同为单 cluster kernel、同段大小、同屏障数，只把段从 global scratch
   换成邻居 SMEM，`dsm/hbm` = **1.06→1.27×**（段越大越显著，因为 L2 往返的带宽/延迟开始展露）。
   这是**损失三在「跨 SM」尺度**的对应物——P24-1 §17 量的是「落 / 不落 HBM」，这里量的是
   「经 / 不经 L2 往返」。
3. **屏障也便宜**：`cluster.sync()` 只有 `grid.sync()` 的 ~1/2，因为它只覆盖 cluster 内几个 SM，
   不是全机器 rendezvous。对 megakernel 的含义：**把同步范围缩到「真正有依赖的那几个 SM」比
   全机器屏障更划算**——这正是 MPK 去中心化 event queue（`docs/03` §3.3）与 DeepGEMM ring
   barrier（`docs/02` §1.5）「按需局部同步」思想在 CUDA 原语层的等价物。
4. **但 cluster 装不下整张机器**：上限 16 block（且必须落在同一 GPC 内、grid 是 C 的倍数）。
   132 SM 的整机 megakernel **不可能是一个 cluster**，只能切成 ~8 个 cluster（每 GPC 一个），
   跨 cluster 仍要走 L2/HBM 或全局 ring——**DSM 只把「最容易的那一跳」留在片上**。

### 18.5 与判据 / MPK / 生产案例的关系

- **对上 P20-1（§14）**：§14 的结论是「跨算子重叠的载体必须是**共享的 SM 内共驻**而非分块」，
  因为静态切 block 的重叠恒慢于屏障。DSM 给出了「共享共驻」在 Hopper 上的**硬件形态**——
  cluster 明确保证一组 block 共同驻留同一 GPC、且可互读 SMEM。它比 §14 的「同 kernel 分块」
  更彻底：数据不经 L2，屏障只覆盖 cluster。
- **对上 P24-1（§17）**：§17 证明「每融合一个算子消一趟全张量 HBM 往返」，本节证明
  「每把一跳中间量从 L2 往返改到 DSM，省 6–21%」——**两条合起来是损失三的两级**：
  HBM↔片上（§17 的主体）、L2↔DSM（本节）。
- **对上 MPK**：论文的运行时把 worker/scheduler 划分到固定 SM（§5.1 行 1148–1174）、用
  paged SMEM（§5.3 行 1312–1325）做跨 task 中间量、用去中心化 event queue 做局部同步
  （`docs/03` §3.3）——这些正是「在单 kernel 内自己实现 cluster 式局部交换与局部屏障」。
  MPK 论文/快照未见对 Hopper cluster 的显式使用（其粒度是「一个 task = 一个 CTA」，`docs/07`
  §0），故这里的 cluster 是**实现层可选的一档加速**而非论文机制；但两者目标一致：**让有依赖的
  SM 之间只做局部、片上的 rendezvous**。
- **对上生产案例**：DeepGEMM Mega Gate 已用 `cluster_dim = num_mma_ctas`（`docs/02` §1.2）把
  「打分行/排序/映射」几段绑在一个 cluster 里做 TMA multicast + DSM；本节的 1.06–1.27× 是这种
  「段级 cluster 融合」相对「经 L2 的段间交换」的量化下界。FlashMoE 的对称张量 + 非阻塞 NVSHMEM
  （`docs/10`）是同一机制在**跨卡**尺度的推广。
- **判据（C1–C5，`docs/00` §3）**：cluster kernel 仍满足 C1（单 launch）+C2（常驻）+C3（多算子），
  且 DSM/`cluster.sync` 给出了比 `grid.sync` 更细的 **C4（设备内协调）**；但它**不是 C5**——
  没有固定的 worker/scheduler 角色去动态派发 task，也没有跨 cluster 的全局协调。**cluster 是
  「C4 的更细实现」，C5 仍由 P15-1 §9 的去中心化队列覆盖。**

### 18.6 局限与踩到的坑

- **编译器坑（重要）**：用**变界 stride 循环**读远端 SMEM（`for (i=threadIdx.x;i<sb;i+=blockDim.x)
  acc += src[i];`，含 `#pragma unroll`）会稳定触发 `unspecified launch failure`（同设备同参数
  可复现，`C=1` 本地读却正常，环 kernel 的「每线程单元素读」也正常）——是 `-O3` 对该循环在
  generic/remote 指针上的代码生成问题。本实验 A/Section 都改用**每线程固定索引的单元素读**
  （`src[threadIdx.x]` / 展开的 `threadIdx.x + k·256`）规避。这是**实现坑，不是 DSM 语义问题**，
  但说明「cluster + 变界远端读」在生产代码里需要小心中间表示（真实 megakernel 多用 TMA/固定 tile）。
- **`Sh` 的对照是 L2 而非 HBM**：scratch 仅 1–2 MB（< 50 MB L2）。要测「DSM vs 真 HBM」需要把
  scratch 撑过 L2，但 cluster 段受 SMEM 容量限制、DSM 无法搬运大段——**这本身就是 DSM 的边界**：
  它只能交换「装得进 SMEM」的中间量。大中间量仍需分块或落 HBM（对应 P23-1 §16 的环形预取）。
- **同步主导区间**：`S` 每轮 1 次 `cluster.sync`（~500–770 ns），段小时（1–2 KB）屏障占大头，
  故 `dsm/hbm` 只 1.06×；段大（8 KB）才显出介质差（1.27×）。绝对带宽（1.3 TB/s）低于纯 SMEM，
  因为每轮只有 `seg` 大小的一跳、屏障不能摊薄。
- **忙等造流、单 cluster 模型**：`A` 用 `volatile` 单元素读测的是**每轮一跳的延迟/吞吐代理**，
  非满带宽 streaming；`S` 是合成段位移，不含 TMA/张量核。只用 1D cluster（`clusterDim.x`），
  未测 2D/3D cluster 与 cluster 内 TMA multicast。
- **共享机器**：device 1 上有其它进程的 context，cluster 启动对 GPC 连续性敏感（早期一版 kernel
  在 `grid≥8` 时间歇性 `unspecified launch failure`），故复现脚本每配置独立进程、并记录
  设备号；本节结论用「比例」，绝对数会随共驻负载小幅漂移。
- 未覆盖 `cudaLaunchAttributeProgrammaticEvent`（PDL 的 cluster 版本）与 cluster 内的
  `mbarrier`/TMA 组合，留作后续（cluster 内 TMA 组合由 P30-1 §19 补齐；2D/3D cluster 与
  子集掩码由 P31-1 §20 补齐）。

## 19. cluster 内 TMA multicast：一次读、C 个 SMEM 收的广播原语（P30-1）

### 19.1 问题

P29-1（§18）量的是 **DSM**——数据**已经到片上**之后，跨 SM 读邻居 block 的 SMEM。它明确留了两条
尾巴，本条补其中一条（§18.6 末行）：

> 未覆盖 …… cluster 内的 `mbarrier`/TMA 组合，留作后续。

Hopper 的 **TMA multicast**（`cp.async.bulk.tensor ... .multicast::cluster`）是另一档原语：
它做的是 **load path**——把一块**全局**数据**一次读入**，由拷贝引擎同时投递到 cluster 内 C 个
CTA 的 SMEM（每个目的 CTA 的 `mbarrier` 各自收到一次完成信号）。这正是 megakernel 里
「权重 / 打分 / 掩码一次读、多 CTA 共享」的机制。生产案例已经在用：**DeepGEMM Mega MoE** 的
heuristics 写死「**Always multicast on A**」并把 token 维按 `LOAD_BLOCK_M = BLOCK_M/2` 切给
2-CTA cluster（`csrc/jit_kernels/heuristics/mega_moe.hpp:141–143`、
`deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh:130`，`cluster_dim=(2,1,1)` 见
`csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp:138`）；DeepGEMM 的普通 sm90 GEMM 也按
`kIsTMAMulticastOnA` 在 A/B 之间选一路做 multicast（`impls/sm90_fp8_gemm_1d2d.cuh:127–202`）。

本节回答三件事：(1) multicast 广播写出来的结果对不对、语义是什么？(2) 相对「每个 CTA 各自 TMA
载入同一块」，delivery 吞吐与源侧（L2/DRAM）流量差多少？(3) 墙钟收益有多大、取决于什么？

### 19.2 方法

程序 `src/tma_multicast.cu`，脚本 `harness/tma_multicast.sh`，原始输出
`harness/tma_multicast.out.txt`（目标卡 H100 / sm90a，SMs=132，**固定空闲 device 3**，nvcc 13.2，
**每配置独立进程**——cluster launch failure 会 sticky 污染 context）。host 用
`cuTensorMapEncodeTiled` 建 2D tensor map（`{256, rows}`，box `{256, box1}`，swizzle/interleave
全 off），设备用 `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension` 设 cluster 维。

kernel 把 SMEM 切成 `L` 个 chunk 槽（≈192 KB），**每轮**对 `L` 个槽各发一次 TMA（同一源的 `L`
个不同分段），用一个 `mbarrier` 累计 `L·chunk` 字节后**一次等待**——于是 TMA 完全重叠（深度 = L），
测的是峰值广播吞吐而非单跳延迟。`R` 轮后线程 0 把累加和写出，对照解析参考算 `maxerr`。两种模式：

- **mode 0（独立）**：每个 CTA 各发 `L` 次普通 TMA（`shared::cluster.global...mbarrier`，无 multicast），
  C 个 CTA 各读同一块一遍。
- **mode 1（multicast）**：**仅 rank0** 发 `L` 次
  `...shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster`，`mask = (1<<C)-1`；
  所有 CTA 在自己的 `mbarrier`（**同一 SMEM 偏移**）上等待。

两个指标：**deliver**（全 grid 实际写进 SMEM 的字节 / 时间，程序计时）与 **源侧读取**
（ncu 的 `lts__t_sectors_srcunit_tex_op_read.sum` / `lts__t_bytes.sum` / `dram__bytes_read.sum`）。
注意程序里 `src-read` 那一列是**指令级模型**（mode 1 = 每 cluster 一份、mode 0 = 每 CTA 一份），
**ncu 才是真值**（见 §19.4）。

### 19.3 结果

**A. 单 cluster（grid=C）扫 C（chunk=8KB, R=500）**

```text
   C | mode0 deliver GB/s | mode1 deliver GB/s | mode1/mode0
 ----+---------------------+---------------------+------------
   2 |       215.6         |       216.0         |   1.002
   4 |       429.8         |       430.3         |   1.001
   8 |       845.2         |       850.1         |   1.006
  16 |      1553.9         |      1562.7         |   1.006
```

delivery 吞吐**随 C 线性放大**（单 CTA ≈ 97–105 GB/s），两模式墙钟几乎相同（差 ≤0.6%）。
**所有配置 `maxerr = 0.00000`**——multicast 逐位正确。

**A2. 多 cluster（grid=128）扫 C（chunk=8KB, R=500）**

```text
   C | mode0 GB/s | mode1 GB/s | mode1/mode0
 ----+------------+------------+------------
   4 |  6730.9    |  6746.9    |   0.998
   8 |  6692.0    |  6767.9    |   0.989
  16 |  6188.7    |  6194.0    |   0.999
```

两模式都在 ~6.2–6.8 TB/s **饱和**（SMEM 写入 / TMA issue 成瓶颈），multicast 一致地**快 0.1–1.1%**。

**B. chunk 扫描（C=8, grid=128, R=500）**

```text
 chunk | mode0 GB/s | mode1 GB/s | mode1/mode0
 ------+------------+------------+------------
   4KB |  5494.3    |  5551.5    |   0.990
   8KB |  6693.9    |  6762.5    |   0.990
  16KB |  6692.5    |  6744.1    |   0.992
  32KB |  6689.5    |  6742.7    |   0.992
```

multicast 稳定快 ~0.8–1.0%；delivery 吞吐在 8KB 后饱和。

**D. ncu 源侧流量（C=8, grid=128, chunk=8KB, R=200；`-c 1`）**

```text
                       |    mode0    |    mode1    | mode0/mode1
 ----------------------+-------------+-------------+------------
 L2 read sectors       |  41,219,228 |  19,660,800 |    2.10x
 lts__t_bytes.sum      |   1.51 GB   |   821.8 MB  |    1.84x
 dram__bytes_read.sum  |   71.28 MB  |   71.28 MB  |    1.00x
```

**C. cluster.sync() sanity（对比 P29-1）**：C=2/8/16 = 399.8 / 428.7 / 469.2 ns/sync
（P29-1 在 grid=128 时是 504–767 ns，本节 grid=C 更小故更低）。

### 19.4 读法

1. **multicast 语义被实测确认**：mode 1 只有 rank0 发指令、其余 CTA 什么都不发，却都在**自己的**
   `mbarrier`（同一 SMEM 偏移）上等到——且 `maxerr=0`。这证明
   **`.multicast::cluster` 把 transaction bytes 记到 mask 内每个目的 CTA 的 `mbarrier` 上**
   （拷贝引擎按 mask 把数据分别写进各 CTA 同偏移的 SMEM）。这与 DeepGEMM 的用法一致：
   `common/tma_copy.cuh:77–86` 正是「`block_rank_in_cluster()==0` 才发 `LoadMulticast`、
   mask=`(1<<num_tma_multicast)-1`、barrier 用 leader 自己的 `mbarrier` 指针」。
2. **delivery 吞吐两模式打平**：广播循环跑到饱和时，瓶颈在 **SMEM 写入 / TMA issue**，不在源侧，
   故 multicast 只赢 ~1%（6.2–6.8 TB/s vs 同样量级）。**广播的能量守恒**——无论怎么读，
   最终都要往 C 个 SMEM 各写 `chunk` 字节。
3. **multicast 的收益在源侧，且比「C×」小**：ncu 显示 L2 read sectors 少 **2.10×**、L2 流量少
   **1.84×**；而 **DRAM 字节完全相同**（71.28 MB）。原因两点：
   - 本实验 `grid=128` 个 CTA 读的是**同一序列**的分段、且被 barrier 同步到同一时刻，mode 0 的 C
     份重复读**被 L2 合并/命中**（首读命中 L2 后其余命中），故 **DRAM 并不因 multicast 而减少**；
   - L2 对并发同址请求本身有合并，使 mode 0 的**实测** L2 transaction 只比 mode 1 多 ~2×，
     而非指令级的 C=8×。**指令级 `src-read` 列（mode1 少 ~8×）是上界模型，真值以 ncu 为准。**
4. **所以墙钟收益取决于瓶颈落在哪一档**：SMEM-write 主导 → 两模式打平（本实验）。
   multicast 真正省的是 **L2 请求/带宽**（2×）——在 megakernel 里，这块被省下的 L2 带宽可以给
   **其它并发 task 的载入**用（DeepGEMM Mega MoE 把 A 的 multicast 与 B/权重的独立载入叠在一起，
   heuristics 注释 `Always multicast on A` 就是选「被多 CTA 共享、最值得省源带宽」的那一路）。

### 19.5 与判据 / MPK / 生产案例的关系

- **对上 P29-1（§18）**：P29-1 量「已到片上后跨 SM 交换」（DSM，dsm/hbm 1.06–1.27×、`cluster.sync`
  便宜 1.5–2.2×）；本节量「**从全局到多个 SMEM 的载入**」（TMA multicast：L2 事务少 2×、指令级源读
  少 C×、DRAM 不变）。**两条合起来是「跨 SM 数据搬运」的完整两跳**：global→C×SMEM（本节）与
  SMEM→邻居 SMEM（§18）。
- **对上 P24-1（§17）**：§17 量「落/不落 HBM」的全张量往返；本节量「同一块**同时**投递给 C 个
  CTA」时源侧 L2 请求的减少——是**不落 HBM**之外、**跨 SM 复用**这一维上的省流量。
- **对上 MPK**：MPK 的粒度是「一个 task = 一个 CTA」（`docs/07` §0），跨 task 中间量走 paged
  SMEM / ring buffer（§5.3 行 1312–1325），**不以 Hopper cluster 为一等公民**；论文/快照未见对
  TMA multicast 的显式使用。故本节是**实现层可选的一档**，与 MPK「局部、片上交换」目标一致
  （`docs/03` §3.3、`docs/06` §3）。
- **对上生产案例（一手核对，DeepGEMM `057ca59`）**：Mega MoE **2-CTA cluster + multicast on A**
  （`heuristics/mega_moe.hpp:141–143`、`impls/sm100_bf16_mega_moe.cuh:130`、`impls/sm100_bf16_mega_moe.hpp:138`），
  且 DeepGEMM 把 cluster multicast 收敛成一个 `tma::copy(...num_tma_multicast)` 包装
  （`common/tma_copy.cuh:56–106`，leader-only + mask，正是本节验证的形态）。fp8/fp4 版
  Mega MoE（`impls/sm100_fp8_fp4_mega_moe.cuh:165`）同。**Mega Gate** 用 `cluster_dim=num_mma_ctas`
  （`docs/02` §1.2）但本节未在其源码里核到 A/B 的 multicast 使用，故不声称。
- **判据（C1–C5，`docs/00` §3）**：cluster kernel 满足 C1+C2+C3，`cluster.sync`/multicast 细化了
  **C4**；**仍不是 C5**（无固定 worker/scheduler 动态派发）。与 P29-1 同档。

### 19.6 局限与踩到的坑

- **ptxas advisory**：`.multicast::cluster` 在 `-arch=sm_90` 下会告警「应使用 `sm_90a`/`sm_100a` 等
  目标，否则在未来架构上可能大幅降速」。本实验用 **`-arch=sm_90a`**（告警仍在，因为 PTX target 串
  打成 `compute_90`，属 advisory，H100 上功能/性能正常）。
- **DRAM 不变是本实验设置的产物**：working set（`R·L·chunk = 200·24·8KB ≈ 37.5 MB`）**装得进 50 MB
  L2**，且所有 CTA 同步读同一序列 → L2 去重。若数据大到反复跨 L2，multicast 才会在 **DRAM** 层面
  显现（但也依赖各 CTA 读的时刻是否分散，故本条**未给出 DRAM 收益的数字**，只给 L2 层面的实测）。
- **deliver 饱和在 ~6.6 TB/s**：这是 SMEM 写入 / TMA issue 的上限，不是源侧带宽；故「墙钟打平」
  是**饱和区间**的结论，launch/延迟受限区间另论。
- **单跳延迟未单独测**：本节只测了重叠后的峰值吞吐；`L`（流水深度）人为取到 ≈192 KB SMEM 上限，
  未扫 `L`。cluster 上限仍是 16 block、grid 须为 C 的倍数（P29-1 §18.3），故 `grid=132` 非法。
- **DSM 手工扇出未纳入对照**：曾实现「rank0 读 + 各 CTA 用 DSM 把数据拷进本地 SMEM」的 mode 2，
  但逐元素/`float4` 的远端 SMEM stride 循环受 §18.6 同类代码生成问题拖累（比 multicast 慢 ~19×），
  属**不平等的 strawman**，已移除；「读一次 + 片上扇出」的成本以 P29-1 的 DSM 数字为准。
- 未测 2D/3D cluster、`cudaLaunchAttributeProgrammaticEvent`（PDL 的 cluster 版）与 SM100 的
  `SM100_TMA_2SM_LOAD`（DeepGEMM sm100 路径用的是后者）。留作后续（2D/3D cluster 与 multicast
  子集掩码由 P31-1 §20 补齐）。

## 20. 2D/3D Thread Block Cluster：rank 布局、DSM 寻址与 TMA multicast 的子集掩码（P31-1）

### 20.1 问题

P29-1（§18）与 P30-1（§19）建立了「跨 SM 数据搬运」的两跳：**DSM**（数据已到片上后读邻居 SMEM）
与 **TMA multicast**（一次读、投递到 C 个 SMEM）。但两者都只用 **1D cluster**（`clusterDim.x`），
两条局限里写着同一个尾巴：

> §18.6 / §19.6：未测 **2D/3D cluster**、cluster 内 TMA multicast 的（子集）掩码。

这不是学术细节。生产案例的 cluster 恰恰是**多维**的：DeepGEMM **Mega MoE** 用
`cluster_dim = (2,1,1)`（`csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp:138`）、把 token 维按
`BLOCK_M/2` 切给 2-CTA cluster（`docs/02` §1.2、§19.1）；**Mega Gate** 用
`cluster_dim = num_mma_ctas`。cluster 一旦不是 1D，就有两个新问题：

1. **rank 到 (x,y,z) 的映射**——`map_shared_rank`（DSM）与 multicast 的 **16-bit mask 都是按线性
   `block_rank()` 编址的**，要「只广播给某一行/某一列」必须知道 rank 怎么排。
2. **子集广播**——真实 kernel（如 Gate 的「按行打分」）只想把数据发给 cluster 内的**一部分** CTA，
   而不是整群。

本节回答四件事：(1) 2D/3D cluster 的线性 rank 布局是什么？(2) DSM 环形交换在 2D cluster 下是否
等同 1D？(3) TMA multicast 的**子集掩码**语义对不对、源侧流量差多少？(4) `cluster.sync` 成本随
2D 形状怎么变、容量边界有没有变？

### 20.2 方法

程序 `src/cluster_2d.cu`，脚本 `harness/cluster_2d.sh`，原始输出 `harness/cluster_2d.out.txt`
（目标卡 H100 / **sm90a**，SMs=132，**固定空闲 device 3**，nvcc 13.2，**每配置独立进程**——cluster
launch failure 会 sticky 污染 context，与 §18/§19 同）。`cudaLaunchKernelEx` +
`cudaLaunchAttributeClusterDimension` 设 `{x,y,z}` 三维 cluster。五个 Section：

- **A（rank 布局）**：每个 block 打印 `block_rank()`，host 核对 `rank == x + y·cx + z·cx·cy`
  （`x=blockIdx.x%cx` 等）。覆盖 `(2,2,1)`/`(4,2,1)`/`(4,4,1)`/`(2,2,2)`。
- **S（2D DSM 环形段位移）**：与 §18 的 `Sd` 同机制（按线性 rank 环形搬、双缓冲 ping-pong、
  每轮一次 `cluster.sync()`），但 cluster 是 `(cx,cy)`；grid 取 `(cx·kx, cy·ky)` 保持 128 block
  （2D launch 要求**每个 grid 维都是对应 cluster 维的整数倍**）。R 轮后对解析参考逐位比较。
- **M（multicast：整群 vs 逐行子集）**：cluster 内 SMEM 切 `L` 槽，每轮对 `L` 槽各发一次 TMA、
  一个 `mbarrier` 累计 `L·chunk` 一次等待（深度 L 全重叠）。三模式：
  - **mode 0（独立）**：每 CTA 各发普通 TMA（同 §19 mode 0）；
  - **mode 1（整群）**：仅 `rank==0` 发 multicast，`mask=(1<<C)-1`（同 §19 mode 1）；
  - **mode 2（逐行子集）**：**每个 row 的 rank0**（`rank%cx==0`）发 multicast，
    `mask = ((1<<cx)-1) << (row·cx)`——只覆盖该行的 `cx` 个 rank；**且各 row 读各自不同**的源区域。
- **P（`cluster.sync()` 成本）** 与 **C2（容量/合法性）**：2D/3D 扫描。
- **D（ncu 源侧流量）**：mode 0/1/2 的 `lts__t_sectors_srcunit_tex_op_read.sum` / `lts__t_bytes.sum`
  / `dram__bytes_read.sum`。

mode 2 的**逐行不同源**是关键设计：如果 mask 语义正确，每个 CTA 只从**本行** leader 收到本行源区的
数据；若 mask 失效（误发给整群），则一个 CTA 会收到多个 leader 的数据、`mbarrier` transaction 数
对不上（挂起）或内容被踩（`maxerr≠0`）。故 `maxerr` 同时校验「投递范围」与「数据正确」。

### 20.3 结果

**A. rank 布局（全部 mismatch=0）**

```text
A cluster=(2,2,1) grid=(4,4,1)
  by= 0:   0   1   0   1          rank = x + y*cx + z*cx*cy
  by= 1:   2   3   2   3          （x 变化最快、其次 y、最后 z）
  by= 2:   0   1   0   1
  by= 3:   2   3   2   3
  rank==x+y*cx+z*cx*cy ? YES (mismatch=0)
A cluster=(4,4,1) grid=(8,8,1)   -> YES (0)
A cluster=(2,2,2) grid=(4,4,2)   -> YES (0)   # 3D：rank 4..7 出现在 z=1 层
```

2D/3D cluster 的 `block_rank()` **就是行主序的线性下标**：`rank = x + y·cx + z·cx·cy`。
这条是所有子集 mask 与 DSM 寻址的基础。

**S. 2D cluster 的 DSM 环形段位移（K=4, seg=2KB, grid=128, R=200）**

```text
   cluster | C  |   GB/s | maxerr
 ----------+----+--------+--------
  (1,2)    |  2 |  587.9 | 0.0000
  (2,2)    |  4 |  552.4 | 0.0000
  (2,4)    |  8 |  533.0 | 0.0000
  (4,4)    | 16 |  527.8 | 0.0000
```

**2D cluster 的 DSM 环形交换与 1D 完全同档**：`(2,2)`/`(4,4)` 为 552/528 GB/s，正落在 P29-1 §18
1D `Sd` 在 seg=2KB 的 **560.7 GB/s** 上（差 <6%）；`maxerr=0` 逐位一致。K 扫描（cluster=(2,2)）：
seg 512B/1KB/2KB/4KB → **230.9 / 392.5 / 555.1 / 723.8 GB/s**，与 §18 的「段越小、每轮 1 次屏障占比
越大」趋势一致。

**M. multicast：整群 vs 逐行子集（chunk=8KB, R=500）**

```text
   cluster (grid)      | mode0 独立 | mode1 整群 | mode2 逐行 | 全部 maxerr
  ---------------------+------------+------------+------------+-----------
  (2,2) grid=(8,8)=64  |  6707 GB/s |  6764 GB/s |  6194 GB/s | 0.00000
  (4,4) grid=(8,8)=64  |  6108 GB/s |  6215 GB/s |  5981 GB/s | 0.00000
```

三模式 `maxerr` 全 **0.00000**——包括**逐行子集**掩码：证明 `.multicast::cluster` 的 mask 语义
（按 `block_rank()` 的位）在 2D cluster 下**精确生效**，且「每个 CTA 只从本行 leader 收本行源区」
成立。chunk 扫描（`(4,4)` mode2）：4KB/8KB/16KB/32KB → **5073 / 5979 / 5980 / 5980 GB/s**，
与 §19 一样在 8KB 后饱和（SMEM 写入 / TMA issue 瓶颈）。

**D. ncu 源侧流量（cluster=(4,4), grid=(8,8)=4 cluster×16 CTA, chunk=8KB, R=200）**

```text
                         |  mode0 独立 | mode1 整群 |  mode2 逐行
  -----------------------+-------------+------------+-------------
  L2 read sectors        |  21,536,076 |  4,915,200 |  19,660,800
  lts__t_bytes.sum       |   795.93 MB |   264.07 MB|   853.88 MB
  dram__bytes_read.sum   |    39.34 MB |    39.34 MB|   157.30 MB
  vs mode0 (sectors)     |    1.00x    |    4.38x↓  |    1.10x↓
```

**P. `cluster.sync()` 成本（grid=(8,8)=64 block, R=200）**

```text
  cluster |  C  | ns/sync
 ---------+-----+---------
  (1,2)   |  2  |  448.1
  (2,2)   |  4  |  451.0
  (2,4)   |  8  |  463.0
  (4,4)   | 16  |  504.6
```

**C2. 容量/合法性（2D/3D）**

```text
  cluster=(4,4,1) C=16 grid=(8,8,1) -> ok
  cluster=(2,2,2) C=8  grid=(4,4,2) -> ok     （3D cluster 可用）
  cluster=(4,4,2) C=32              -> cluster misconfiguration（仍硬上限 16）
  cluster=(8,8,1) C=64              -> cluster misconfiguration
  cluster=(4,4,1) grid=(8,6,1)      -> cluster misconfiguration（grid 各维须为 cluster 维整数倍）
```

### 20.4 读法

1. **rank 是行主序线性下标**：`rank = x + y·cx + z·cx·cy`（A 段全部 mismatch=0）。所以「子集广播」
   的实现就是**按 rank 位拼 mask**：想发给第 `y` 行（`x=0..cx-1`）就 `mask=((1<<cx)-1)<<(y·cx)`；
   想发给第 `x` 列就 `mask = Σ_y 1<<(x+y·cx)`（步长为 `cx` 的位）。DSM 的 `map_shared_rank` 同理按
   rank 取邻居。
2. **DSM 是「按 rank」而非「按几何邻居」的**：2D cluster 的环形交换与 1D 完全同档（552/528 vs
   560 GB/s），因为 `map_shared_rank` 只认线性 rank，cluster 的 2D 形状只影响 `rank` 怎么铺。
   ⇒ 想把「2D 邻居」（上/下/左/右）做成 ring，需要在**设备端把 (dx,dy) 邻居换算成 rank**再取指针。
3. **multicast 的子集掩码语义被实测钉死**（M 段 `maxerr=0`、逐行不同源）：mask 位 = 线性 rank，
   **只发给 mask 命中的目的 CTA**，其余 CTA 的 `mbarrier` 收不到 transaction（故必须用**与投递范围
   匹配的 `expect_tx`**，否则挂起）。这解释了 DeepGEMM 为什么敢写死 mask（§19.1 的
   `tma_copy.cuh:77–86`）——mask 是**精确**的，不是「广播给整群再丢弃」。
4. **子集广播是「源侧读取 vs 投递范围」的旋钮**：ncu 显示同一份工作，mode1（整群、1 个源区/cluster）
   L2 read sectors 比 mode0 少 **4.38×**、DRAM 不变；mode2（逐行、`cy` 个源区/cluster）的
   L2 sectors 恰是 mode1 的 **cy=4 倍**（19.66M/4.92M=4.00）、DRAM 是 mode0 的 4 倍——**因为各 row
   读的是不同源区、没有 L2 去重**。要点：**multicast 省的是「同一块的重复读」，一旦各行读不同数据，
   源侧流量就随之上升**；mode2 的价值在**投递范围正确**（Gate 的按行打分），不是省流量。
5. **`cluster.sync` 与 cluster 形状无关，只与 block 数有关**：2D `(1,2)`→`(4,4)` 是 448→505 ns
   （C 2→16），与 §18 的 1D 同量级（§18 grid=128 时 504→767 ns，绝对差来自本段 grid 更小)。
   `cluster.sync` 覆盖的是「这个 cluster 的这几个 SM」，**怎么排成 2D 不改变它同步多少个 block**。
6. **容量边界不变**：3D cluster 可用（`(2,2,2)=8` ok），但**硬上限仍是 16 block**（`(4,4,2)=32`
   失败），且**grid 每个维度都必须是 cluster 对应维的整数倍**（`grid=(8,6)` 配 `cy=4` 失败——
   1D 时代的「grid 是 C 的倍数」在多维下变成**逐维**约束）。⇒ 132 SM 的整机 megakernel 切成
   cluster 时，切法必须让每个 grid 维对齐。

### 20.5 与判据 / MPK / 生产案例的关系

- **对上 P29-1（§18）/ P30-1（§19）**：本节把 §18 的 DSM 与 §19 的 multicast 从 1D 推广到 2D/3D，
  并补上两节都留的「2D/3D cluster」与「子集掩码」尾巴。结论是**机制不变、多一层 rank 换算**：
  DSM 与整群 multicast 行为同 1D；新增的只是「按线性 rank 拼子集 mask」这条实现细节。
- **对上 P24-1（§17）**：§17 量「落/不落 HBM」；§19 量「整群 multicast 省 L2 请求」；本节进一步显示
  **multicast 省不省取决于「被广播的块是否被多 CTA 共享」**——逐行不同源时（mode2）反而把 DRAM
  拉到 4×。这是「跨 SM 复用」这一维上更精确的边界。
- **对上 MPK**：MPK 的粒度是「一个 task = 一个 CTA」（`docs/07` §0），worker/scheduler 划分到固定
  SM（§5.1 行 1148–1174），跨 task 中间量走 paged SMEM / ring buffer（§5.3 行 1312–1325），
  **不以 Hopper cluster 为一等公民**，也未见对 2D cluster 的使用。故本节仍是**实现层可选的一档**，
  与 MPK「按需局部同步」目标一致（`docs/03` §3.3、`docs/06` §3）。
- **对上生产案例（一手核对，DeepGEMM `057ca59`）**：Mega MoE 的 `cluster_dim=(2,1,1)`
  （`impls/sm100_bf16_mega_moe.hpp:138`）在形式上就是一个 2D cluster——其 rank 布局正好是本节
  A 段验证的 `rank = x + y·cx` 在 `cy=1` 的退化；Mega Gate 的 `cluster_dim=num_mma_ctas`
  （`docs/02` §1.2）同理。故 DeepGEMM 的 cluster 用法**只需要 1D 语义**，是本节的退化情形；
  真正的多维子集广播（按行/列）在本机实测上**可用且精确**，但目前未见生产案例使用。
- **判据（C1–C5，`docs/00` §3）**：cluster kernel 满足 C1+C2+C3，`cluster.sync` / multicast / DSM
  细化了 **C4**；**仍不是 C5**（无固定 worker/scheduler 动态派发）。与 P29-1/P30-1 同档。

### 20.6 局限与踩到的坑

- **2D launch 的逐维对齐**：最初把 `S` 写成 1D grid（`grid=128`）配 2D cluster `(2,2)`，直接
  `cluster misconfiguration`——**grid 的每个维度都必须是对应 cluster 维的整数倍**（`gridDim.y=1`
  不是 `cy=2` 的倍数）。改成 `grid=(cx·kx, cy·ky)` 后正常。这是 1D（只要求 `grid % C == 0`）
  在多维下的自然推广，容易踩。
- **正确性基准要为多维重写**：DSM 环形的解析参考在 1D 下是「`gid` 的 cluster 连续一段」，2D 下
  cluster 的 block 在全局线性 id 里**不连续**（`gid = by·gx+bx`），必须按 `(bx%cx, by%cy)` 反推
  同 cluster 的邻居 id。首版沿用 1D 的 `cstart=g-(g%C)` 在 C=16 时给出 `maxerr=40000`（C≤4 时
  碰巧连续才为 0）——**纯 host 基准 bug，不是 DSM 语义**，修正后全部 `maxerr=0`。
- **mode2 的 DRAM 上升是实验设置**：逐行读**不同**源区是刻意为验证掩码而设的；真实 kernel 若各行
  读同一块，DRAM 会与 mode0 相同（如 mode1）。故不要把 mode2 的 4× DRAM 读成「子集广播更贵」，
  它是「源数据不同」的直接结果。
- **deliver 仍饱和在 ~6 GB/s 档**：与 §19 一样，测的是 SMEM 写入 / TMA issue 的饱和吞吐，不是源侧
  带宽；multicast 的墙钟收益在饱和区间被掩盖，要看 ncu 的 L2 计数。
- **cluster 上限仍 16 block**：3D cluster 可用但总量不变（`(8,8,1)=64` 失败）；故 132 SM 整机仍只能
  切 ~8 个 cluster。DSM 的「变界远端读」代码生成坑（§18.6）在本节同样规避（每线程固定索引读）。
- **仍未覆盖**：SM100 专属的 `SM100_TMA_2SM_LOAD`（DeepGEMM sm100 路径用的是后者），留作后续；
  本节只测到 sm90a 的 cluster 原语。~~`cudaLaunchAttributeProgrammaticEvent`（PDL 的 cluster 版）~~
  已由 **P32-1（§21）** 单独覆盖。

## 21. 事件可见性的前移原语：ProgrammaticEvent / LaunchCompletionEvent 能把「进度」提前到哪一点（P32-1）

### 21.1 问题

P4-2 / P5-3 / P12-1 / P19-1 已经把 **PDL（Programmatic Dependent Launch）** 测透：下游 grid 靠
`cudaGridDependencySynchronize()` 与上游 tail 重叠，重叠量 = `min(tail, prologue)`。但有一个
kernel 级原语一直没覆盖（§20.6 明留的尾巴）：CUDA 还提供**两种把「一个 cudaEvent 的触发点前移」的
launch attribute**——

- `cudaLaunchAttributeProgrammaticEvent`（`driver_types.h:3999`，值 7）：事件在**所有 block 都调用过
  `cudaTriggerProgrammaticLaunchCompletion()`（PTX `griddepcontrol.launch_dependents`）**后触发；
  也可用 `triggerAtBlockStart=1`（`driver_types.h:4182`）让每个 block 在**启动时**自动触发。
- `cudaLaunchAttributeLaunchCompletionEvent`（`driver_types.h:4067`，值 12）：在所有 block
  **开始**执行时触发（best effort）。

这对 megakernel 是核心问题：**「中间进度」能不能被别的执行体更早看到？** MPK 用 in-kernel event
counter + 设备内原子 ring buffer 把依赖粒度压到 **task**（`docs/06` §3、`docs/03` §3.3），是不满足
C1–C5 的 kernel 级原语做不到的。本节把 kernel 级能做到的程度量化出来：事件可见性到底停在
**完成点**、**block 启动点**，还是**中间某点**。

### 21.2 方法

程序 `src/programmatic_event.cu`，脚本 `harness/programmatic_event.sh`，原始输出
`harness/programmatic_event.out.txt`（目标卡 H100 / sm90，SMs=132，**固定空闲 device 3**，
nvcc 13.2，`iters=30`，`pro=tail=100000` SM cycles，`step=20000`）。三个 Section：

- **Section A（单 block，事件触发点在时间轴上的位置）**：A 一个 kernel =
  `[prologue 自旋 pro]` → `[可选 trigger]` → `[tail 自旋 tail]`；C 一个极短消费 kernel 放在**另一条
  stream**，靠 `cudaStreamWaitEvent(C, evX)` 等某个事件。用两个**计时事件** `evA0`（A 之前 record）与
  `evC1`（C 之后 record）测「A 启动 → C 被放行开始」的 GPU 时间轴间隔 = **放行信号的时间位置**。六档：
  `seq`（同 stream 直接排 C）/ `post`（普通 `cudaEventRecord`，完成点）/ `early`（ProgrammaticEvent +
  在 prologue 后调 trigger）/ `none`（ProgrammaticEvent 不调 trigger）/ `bstart`（ProgrammaticEvent +
  `triggerAtBlockStart=1`）/ `lce`（LaunchCompletionEvent）。
- **Section B（多 block 错峰，G 扫描）**：block i 的 prologue 加 `i*step`，看 cross-stream 放行点随
  G 怎么变。若事件在完成点放行，`A_start→C ≈ (G-1)*step + pro + tail`。
- **Section C（同 stream PDL，trigger 到底放行了什么）**：A 后紧跟 B（B 用
  `cudaLaunchAttributeProgrammaticStreamSerialization`，结构 `prol → cudaGridDependencySynchronize →
  crit`），看 stream 总耗时随「A 是否调 trigger / 是否挂 ProgrammaticEvent 属性」怎么变。

programmatic / LCE 事件必须 `cudaEventDisableTiming`（`cuda.h` 明确要求），故用**计时事件 evA0/evC1
夹住**来给 GPU 时间轴定位，而不是给事件本身计时。

### 21.3 结果

**A. 单 block：只有 LaunchCompletionEvent 真正前移了 cross-stream 放行点**

```text
    mode | A_start->C(us) |    A_total(us) | C-Ahead(us) |  dep
---------+----------------+----------------+-------------+------
     seq |         112.76 |         120.95 |     -112.76 |   30
    post |         108.26 |         116.51 |        0.00 |   30
   early |         108.21 |         116.45 |        0.05 |   30
    none |         108.15 |         116.36 |        0.11 |   30
  bstart |         108.21 |         116.39 |        0.05 |   30
     lce |           7.77 |         106.43 |      100.49 |   30
```

`post`/`early`/`none`/`bstart` 四档的 `A_start→C` **都在 ~108 µs ≈ A 整段（116 µs）**——即
**ProgrammaticEvent 的提前 trigger（包括 `triggerAtBlockStart`）并不放行 cross-stream 的
`cudaStreamWaitEvent`/host 依赖**。只有 `lce` 落到 **7.77 µs**：所有 block 一启动就放行，下游 C 立即
开始（差的 ~100 µs 就是 A 的 prologue+tail）。这与 CUDA 文档一致——`cuda.h:2227` 明说
ProgrammaticEvent 的依赖方（**含调用 `cuEventSynchronize()` 的 host 线程**）「不保证在 release 时刻
观察到放行，可能直到 kernel 完成后才观察到」；`driver_types.h:4082` 又补一句 LCE「名义上类似
`triggerAtBlockStart` 的 programmatic event，**只是对 `cudaGridDependencySynchronize()` 不可见**」。

**B. 多 block 错峰：lce 的放行点恒定，其余随 (G-1)*step 线性**

```text
    G | (G-1)*step(us) |  early(us) | bstart(us) |  post(us) |  lce(us)
------+----------------+------------+------------+-----------+---------
    1 |           0.00 |     108.18 |     108.18 |    108.34 |     7.50
    4 |          30.30 |     138.48 |     138.43 |    138.42 |     7.78
   16 |         151.52 |     259.73 |     260.06 |    259.75 |     7.76
   64 |         636.36 |     744.73 |     744.62 |    744.59 |     7.70
  128 |        1282.83 |    1391.19 |    1391.02 |   1391.09 |     7.64
```

`early`/`bstart`/`post` 三条线**完全重合**、随 `(G-1)*step` 线性增长（= 完成点放行）；`lce` 恒定
~7.5 µs、与 G 无关——**cross-stream 下唯一真正「全 block 开始即放行」的信号是 LCE**。
（这也顺带说明：无法用 cross-stream wait 去验证 ProgrammaticEvent 的「等所有 block 触发」语义，
因为那个 port 根本不向 stream/host 放行。）

**C. 同 stream PDL：trigger 的作用是「允许下游 grid 提前被调度」，不放行 `gridDependencySynchronize`**

```text
case                                           | stream total(us)
-----------------------------------------------+-----------------
A 不调 trigger                                  |           208.51
A 调 early trigger（无事件属性）                 |           157.72
A 调 early trigger + ProgrammaticEvent 属性      |           157.74
期望：无 trigger ≈ pro+tail+prol+crit = 202.0；有 trigger ≈ pro+max(tail,prol)+crit = 151.5
```

有 trigger 时 stream 总耗时 ≈ `pro + max(tail, prol) + crit`——**下游 B 的 prologue 与上游 A 的 tail
重叠**了 `min(tail, prol)`；无 trigger 时 ≈ `pro+tail+prol+crit`，B 整段排在 A 之后。两类 case 里 B 的
`cudaGridDependencySynchronize()` **都是等 A 完成才返回**（把 B 的 prologue 置 0、crit 拉长即回到
§21 的 `A_start→C ≈ A_total`）。**挂不挂 ProgrammaticEvent 事件属性结果相同**（157.72 vs 157.74）——
设备端口由 `cudaTriggerProgrammaticLaunchCompletion()` 本身提供，事件属性只是把同一个端口额外
**记录**到一个 `cudaEvent`，供 CUDA Graph 的 `PROGRAMMATIC` 边使用（`cuda.h:2047/2060`
`CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC`）。

### 21.4 讨论：三类事件的「可见性受众」

| 事件形态 | 触发点 | stream/host 可见的放行点 | 设备内 programmatic port |
|---|---|---|---|
| 普通 `cudaEventRecord` | 上游完成 | **完成** | 无 |
| `ProgrammaticEvent`（trigger / `triggerAtBlockStart`） | 运行中 / block 启动 | **完成**（文档：不保证在 release 时刻观察到） | 有，供 PDL「允许提前调度」+ 图 PROGRAMMATIC 边 |
| `LaunchCompletionEvent` | 所有 block 开始 | **block 启动（≈0）** | 无（文档：对 `cudaGridDependencySynchronize()` 不可见） |

换句话说，kernel 级原语里**没有**「中间进度可被 stream/host 看到、且早于完成」的通用信号：
- 想要「下游尽早启动（做自己的 prologue）」，用 **PDL / ProgrammaticEvent port**，但同步点仍是完成；
- 想要「下游尽早知道上游已开始」，用 **LCE**，但它只表达「started」，不能当数据依赖；
- 想要「上游写到第 i 个 tile 就放行第 i 个 task」，kernel 级原语**做不到**——只能把执行体放进**同一个
  常驻 kernel**、用**设备内 event counter / 原子 ring buffer** 表达（MPK `docs/06` §3；本专题
  P15-1 §9 的 decen 骨架）。

这正好解释了生产案例的设计：DeepGEMM mega 家族用「**warmup wave + `ScoreBarrier`（异步原子 + grid
标签）/ ring barrier**」在单 kernel 内做**比完成更细**的协调（`docs/02` §1.5/§1.2），而不是指望
kernel 级事件——因为本节的 A/B 表明那样的粒度根本不存在。

**判据（C1–C5，`docs/00` §3）**：本节六档全部是 kernel-per-operator 的降级项，**均不满足 C1–C5**
（仍需 host 逐个 launch、kernel 仍会结束）。`lce` 是唯一能「提前启动下游」的 kernel 级信号，比普通
event record 多一类用途，但推进单位仍是 kernel。

### 21.5 局限与踩到的坑

- **口径 necessity**：programmatic/LCE 事件禁用计时，故本节的「放行点」是**用一对计时事件夹住下游
  kernel 的起点**间接测出的（下游开始 = 放行 + 下游自身启动延迟），不是事件本身的硬件时间戳。
  Section A 的绝对差（0 vs 100 µs）远大于该延迟，结论不受影响。
- **Section C 只证到「允许提前调度」**：本机实测始终是 `gridDependencySynchronize()` 等**上游完成**，
  未构造出「sync 在 trigger 时刻返回」的用例——这与 CUDA 头文件对 `cudaGridDependencySynchronize`
  的描述（「block until all direct grid dependencies have completed」）一致，也解释了 P5-3 的
  `min(tail, prologue)`（重叠的只能是**下游 prologue**）。
- **未测 CUDA Graph 的 `PROGRAMMATIC` 边**：cross-stream 的 programmatic 依赖在 **stream** 上不
  提前放行，但 CUDA Graph 用 `CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC` 边是可以表达的
  （`cuda.h:2055–2065`）——本节只在 stream 上测，图版本留后续。
- **观察者与 G 上限**：Section B 的 G 最大 128 ≤ 132 SM，全部 block 同时常驻，`stagger` 语义干净；
  未测 G≫SM（此时 block 排队，`(G-1)*step` 模型需重算）。
- **正确性哨兵**：三 Section 的消费 kernel 均 `dep==iters`（全部被真正执行），无漏放行/重复。

## 22. 2SM 协作：SM100 `cta_group::2` TMA load 与 tcgen05 2-SM MMA 的 sm90 投影（P33-1）

### 22.1 问题

§18（Cluster + DSM）、§19（TMA multicast）、§20（2D/3D cluster / 子集掩码）、§21（进度前移原语）
已经把 sm90 cluster 的四类原语量完。§20.6 与「下一步」都还点名**两条 SM100 专属原语**没覆盖：

- **`SM100_TMA_2SM_LOAD`**（`cp.async.bulk.tensor ... .cta_group::2`）：一个 CTA 发起，把一块 tile
  **按 2 个 SM 分区投递**（CTA0 得 M 上半、CTA1 得 M 下半），两个 CTA 共用一个事务屏障。
- **tcgen05 2-SM MMA**（`tcgen05.mma.cta_group::2`）：**一对 SM 协作算一次 MMA**（2x1SM），
  每 CTA 的 Tensor Memory（TMEM）各存自己那半 M 的累加器。

这两条都**只在 SM100 可执行**，H100（sm90）上跑不了。但它们对 megakernel 的意义很直接：**2SM 把
「SM 级调度」这一轴（`docs/00` §3 的 C5）又往下压了一级**——从「单 SM 内的 warp 角色划分」
（P13-1 / P18-1）到「**两个 SM 当成一个逻辑计算单元**」。本节分两步：(a) 在 CUTLASS（DeepGEMM 内
`third-party/cutlass`，commit `057ca59`）里核对 2SM 的**一手语义**；(b) 在 H100 上用 sm90 原语把
「一次发、按 SM 分区投递」**投影**出来并实测量化它相对「各读各的」和「multicast 复制」的差异。

### 22.2 资料分析：CUTLASS / DeepGEMM 里的 2SM

**TMA 2SM load（`include/cute/arch/copy_sm100_tma.hpp`）**：一族 `SM100_TMA_2SM_LOAD_{1D..5D}`
（`:52/78/104/130/156`）、dispatcher `SM100_TMA_2SM_LOAD`（`:182`），另有 `.multicast::cluster`
变体 `SM100_TMA_2SM_LOAD_MULTICAST_*`（`:229` 起）与 im2col 变体（`:402` 起）。关键实现细节：

- PTX 形如 `cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes`
  （`:92`）——比 §19 的普通 multicast 多了 **`.cta_group::2`** 限定符。
- 障碍句柄做了一次位与：`smem_int_mbar = cast_smem_ptr_to_uint(mbar_ptr) & Sm100MmaPeerBitMask`
  （`Sm100MmaPeerBitMask = 0xFEFFFFFF`，`:45/63`），注释（`:62-63`）写明 *"Executed by both CTAs.
  Set peer bit to 0 so that the transaction bytes will update CTA0's barrier."*——即**一对 CTA 的 TMA
  完成字节都记到 CTA0 的同一个 mbarrier**（清掉 SMEM 地址里的 peer 位 = 指到对端 CTA0 的地址）。
- 集体层 `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:234` 注释 *"Only one thread
  issues the TMA and updates the barriers in a 2SM MMA, adjust bytes accordingly"*——**2SM 对里只有一个
  线程发 TMA / 更新屏障**，故事务字节要按对（而非按 CTA）计。

**tcgen05 2-SM MMA（`include/cute/arch/mma_sm100_umma.hpp`）**：如 `SM100_MMA_TF32_2x1SM_SS`
（`:382`）与 `SM100_MMA_F16BF16_2x1SM_SS`（`:422`），`static_assert(M == 128 || M == 256,
"...M-mode size should be 128 or 256 for 2 CTA cluster MMA.")`（`:384`）；PTX
`tcgen05.mma.cta_group::2.kind::{tf32,f16,...}`（`:406/446/...`）由 `cute::elect_one_sync()` 单线程发出。
配套的完成信号 `umma_arrive_2x1SM()`（`include/cutlass/arch/barrier.h:779-789`）用
`tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64`——**一次 MMA 提交对发射一对
SM 的屏障各到达一次**。2x1SM 的 M 枚举行（`AtomThrShapeMNK` 的 2 路）由两个 CTA 的 TMEM 各自持有。

**DeepGEMM 自身**（commit `057ca59`）：Mega MoE 用 `cluster_dim=(2,1,1)`
（`csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp:138`）、Mega Gate 用 `cluster_dim=num_mma_ctas`
（`csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp:74`）——**2SM 对就是它们 cluster 的最小单位**
（细节见 `docs/02` §1.2/§1.5，此处不重复）。

**一句话**：SM100 的 2SM = **一次 TMA 发、一个屏障、一对 SM 的 TMEM 分持累加器**——把两个物理 SM
在硬件层面绑成一个逻辑计算单元。sm90 没有 `.cta_group::2`，只能**软件投影**（下面）。

### 22.3 方法（sm90 投影）

程序 `src/two_sm_load.cu`，脚本 `harness/two_sm_load.sh`，原始输出 `harness/two_sm_load.out.txt`
（目标卡 H100 / sm90a，SMs=132，**固定空闲 device 2**，nvcc 13.2，cluster 恒为 2 CTA，128 线程/CTA，
每配置独立进程避免 cluster launch failure sticky）。一个 2-CTA cluster（一「pair」）每轮要把
**逻辑 tile = chunk A + chunk B** 装进两个 SM 的 SMEM，三种投递策略公平对照：

- **mode 0 `indep`**：CTA0 用 `cp.async.bulk` 把 A 载进自己 SMEM、CTA1 把 B 载进自己 SMEM
  （**两个 CTA 各发各的**，2 次源读）。
- **mode 1 `mcast`**：rank0 用 §19 的 **TMA multicast** 把 A 复制投递到两个 CTA 的 SMEM
  （**1 次源读、复制**；两 CTA 拿到同一份）。
- **mode 2 `split`**：rank0 用 **2 次 `cp.async.bulk.shared::cluster.global` 把 A 发给自己 SMEM、
  B 发给对端 SMEM**（`mapa.shared::cluster` 求对端地址与**对端 mbarrier**；**1 个 CTA 发起、按 SM 分区**，
  2 次源读）——这是 **`cta_group::2` TMA load 的 sm90 投影**。

每 CTA 在 `mbarrier` 上等到自己的 chunk 后，读 SMEM 的 `[0]`、`[128]` 两元素求和写入 `out[blockIdx.x]`，
核对解析参考（mode0/2 期望 CTA0=A、CTA1=B，mode1 期望两者=A）。四 Section：
**A** 单 pair（grid=2）扫 chunk（4/8/16/32 KB）、**B** 多 pair（grid=128，64 pair）扫 chunk、
**C** grid=128 扫轮数 R、**D** ncu 量源侧 L2/DRAM 字节。

### 22.4 结果

**A. 单 pair（grid=2，R=500）：三模式同量级，mode2 略慢**

```text
 chunk |   mode0 indep |   mode1 mcast |   mode2 split | (deliver GB/s: m0/m1/m2)
-------+---------------+---------------+---------------+
  4 KB |    308.0 us   |    318.8 us   |    328.9 us   |   13.3 / 12.8 / 12.5
  8 KB |    317.2 us   |    327.3 us   |    340.1 us   |   25.8 / 25.0 / 24.1
 16 KB |    335.6 us   |    345.9 us   |    374.3 us   |   48.8 / 47.4 / 43.8
 32 KB |    462.9 us   |    384.3 us   |    548.2 us   |   70.8 / 85.3 / 59.8
```

单 pair 下三模式被「2 CTA cluster 的 mbar 等待 + `cluster.sync()` 逐轮固定成本」主导，只在 chunk=32KB
拉开：**mcast 反而最快（384 vs 463 µs）**——它 1 次源读、且没有「一个 CTA 连发两个远端 bulk」的
issue 压力；**split（mode2）最慢（548 µs）**。

**B. 多 pair（grid=128，R=500）：multicast 越大越占优，split 越大越吃亏**

```text
 chunk |   mode0 indep |   mode1 mcast |   mode2 split | m1/m0 | m2/m0
-------+---------------+---------------+---------------+-------+------
  4 KB |    476.7 us   |    459.3 us   |    491.8 us   | 0.964 | 1.032
  8 KB |    501.8 us   |    495.8 us   |    522.7 us   | 0.988 | 1.042
 16 KB |    562.3 us   |    529.6 us   |    602.6 us   | 0.942 | 1.072
 32 KB |    732.7 us   |    604.2 us   |    828.2 us   | 0.825 | 1.130
```

**multicast 相对「各读各的」在 32KB 快 1.21×（`m1/m0=0.825`）**，且优势随 chunk 增大；**split 在 32KB
慢 1.13×（`m2/m0=1.130`）**。对照 §19：multicast 的收益来自**源侧流量减半**；split 与 indep 的源侧流量
**其实一样**（都是 A、B 各读一次），差异纯粹在**发起端/落点**——sm90 上「一个 CTA 连发两次、其中一次
写远端 SMEM」比「两 CTA 各发一次、各写本地 SMEM」多花约 13%（单 SM 的 TMA issue 端口 + 远端 SMEM 写路径）。

**C. grid=128，chunk=8KB，扫 R：稳态下三模式差 ~5%**

```text
    R |   mode0 indep |   mode1 mcast |   mode2 split
------+---------------+---------------+---------------
  100 |    104.0 us   |    102.9 us   |    108.0 us
  500 |    502.1 us   |    495.8 us   |    522.3 us
 2000 |   1995.1 us   |   1969.5 us   |   2076.4 us
```

三模式随 R **严格线性**（每轮固定成本恒定），说明没有隐藏的批处理效应。

**D. ncu 源侧字节：multicast 让 DRAM/L2 读**恰**减半，split 与 indep 完全相同**

```text
 mode | dram__bytes_read | lts__t_bytes | lts read sectors |  = sectors*32B
------+------------------+--------------+------------------+--------------
    0 |      209.76 MB   |   448.08 MB  |      6,553,600   |  209.72 MB
    1 |      104.89 MB   |   290.60 MB  |      3,276,800   |  104.86 MB
    2 |      209.73 MB   |   447.84 MB  |      6,553,600   |  209.72 MB
```

（grid=128、chunk=8KB、R=200：indep/split 每 pair 读 A、B 两块 = `64×8192×200×2 = 209.7 MB`；
mcast 只读 A = `104.9 MB`。）

- **mcast 的 DRAM 读 209.76→104.89 MB（恰 2.00×）**、L2 read sectors 6553600→3276800（**恰好一半**）——
  与 §19（P30-1）在 sm90 multicast 上量到的「源侧 L2 减半、DRAM 随场景」同源：**复制投递下同一个 tile
  只从源读一次**（L2 命中接力），故 32KB 时快 1.21×。
- **split 的源侧字节与 indep 完全相同**（209.73 vs 209.76 / sectors 一致）。这坐实了 **`cta_group::2`
  的分区投递并不省源流量**——A、B 是两份不同的数据，无论由几个 CTA 发起，每个字节都只读一次。2SM load
  的价值在 **硬件层**（一次发、一个屏障、一对 SM 分持累加器），**不在**减少 HBM/L2 流量；sm90 缺席
  `cta_group::2`，只能靠软件连发——而这恰恰在 sm90 上**更慢**。

### 22.5 读法

1. **「复制」与「分区」是两个正交方向，别混**。两个 SM 需要**同一份**数据 → `multicast` 复制投递，
   源流量减半（mode1，本机 32KB 1.21×）。两个 SM 需要**各持一半** → `partition` 分区投递，源流量不变
   （mode0/2），省的是协调而非字节。SM100 `cta_group::2` 把**分区**做进了硬件（一次 TMA 发、一个屏障、
   一对 SM 的 TMEM），sm90 只能用软件投影，且投影有 ~13% 代价。
2. **sm90 上「让两个 CTA 共享一次载入」正解是 multicast 而非「单发分区」**：若数据相同，multicast
   又快又省源；若数据不同，还是**两 CTA 各发各的**（mode0）比「一个 CTA 连发两块（含远端）」便宜。
   split 唯一必要的场合是「必须由单一 issuer 保证原子性/顺序」。
3. **2SM 是「SM 级调度」轴（C5）的下一级**：§18–§21 的 cluster 原语都是**软件协作**（SM 之间约定
   mapa / mbarrier / multicast）；2SM 则是**硬件把一对 SM 绑成一个计算单元**（共享屏障、分摊 TMEM、
   单线程发 MMA）。对 megakernel 的含义：生产级 SM100 mega-kernel（DeepGEMM Mega MoE/Gate）把
   `cluster_dim` 的最小单位设为 2，正是为了拿到这一级**免软件协调**的原子性。

### 22.6 与判据 / MPK / 生产案例的关系

- **判据（C1–C5，`docs/00` §3）**：本节的 2SM load 是**段内**原语，与 §18/§19 同档——满足 C1–C3、
  `cta_group::2`/multicast 细化 C4（设备内协调），**仍非 C5**（无 worker/scheduler 动态派发）。
  真正的 C5 由 P15-1 §9 / P22-1 §15 的去中心化队列覆盖。
- **与 MPK 的关系**：MPK 论文假设的并行单元是「一个 task = 一个 thread block」（`docs/01` §4.2、
  `docs/07` §0）。2SM 意味着**一个 task 可以占一对 SM 并硬件绑定**——这是 MPK §7 未展开的硬件轴；
  对「寄存器固定为各 task 最大值」（P14-1）而言，2SM 用 **TMEM 分摊**规避了 SMEM/寄存器的一部分压力
  （tcgen05 累加器不在寄存器里）。
- **与生产案例**：DeepGEMM Mega MoE `cluster_dim=(2,1,1)`、Mega Gate `cluster_dim=num_mma_ctas`
  （`057ca59`，`docs/02` §1.2/§1.5）的最小 cluster 就是 2SM 对；本节的 mode1（multicast）与 mode0
  （各发各的）对应它们在 cluster 内的两条载入路径，Mega MoE 的「**Always multicast on A**」正是选
  mode1 那一路（被多 CTA 共享、最值省源带宽）。

### 22.7 局限与踩到的坑

- **这是 sm90 投影，不是 SM100 `cta_group::2` 本体**：sm90 没有 `.cta_group::2`，mode2 用「单 CTA
  连发两次 bulk（含一次写远端 SMEM）」近似「一次发、分区投递」；真实 2SM 由硬件在一个事务里完成分区，
  且**屏障在 CTA0 单点**（`Sm100MmaPeerBitMask` 清 peer 位），本投影是两套独立 mbarrier。故 mode2
  的 ~13% 代价**不能**推广到 SM100 本体。
- **未测 tcgen05 2-SM MMA**：H100 无 tcgen05，2x1SM 的**算**侧（TMEM 分持、`umma_arrive_2x1SM`
  单点到达）仅为资料分析，未实测。
- **mode1 的语义与 mode0/2 不同**（复制 vs 分区），三者是在「把 2 个张量搬进 2 个 SM」这一目标下的
  三条实现路径，不是同一份数据的三种搬法；正确性各按各自语义核对（全部 `maxerr=0.00000`）。
- **未扫 cluster > 2**：本节固定 C=2（2SM 的对应形态）；§19/§20 已扫过 C=4/8/16 的 multicast/DSM。
- **效率上限不高**：本节测的是「载入原语」而非「端到端 task」，带宽只到几 TB/s（受 SMEM 写入与
  TMA issue 限），是相对比较，不是绝对性能。

## 23. Mega Gate 的 `cluster_dim` 路径：sm90 等价复刻（P34-1）

### 23.1 问题

`docs/02` §1.2 已把 DeepGEMM **Mega Gate**（MoE 路由段段级 mega-kernel）精读到位，但它是 **SM100
专属**（`sm100_bf16_mega_gate.cu` 的 tcgen05 / TMA / TMEM），本机 H100/sm90 跑不了。§22（P33-1）把
SM100 的 **`cta_group::2` 载入**投影到了 sm90，但那只覆盖「把 tile 搬进一对 SM」的**硬件原语**，
没有覆盖 Mega Gate 真正把多个 CTA 绑成**一个逻辑 CTA** 的**协调路径**——也就是
`cluster_dim = num_mma_ctas` 这条线（`csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp:74`）。
本节把这条**协调路径**（不是 SM100 指令）用 sm90 原语等价复刻出来并量化三件事：

1. **CTA pair 切专家维**：`num_mma_ctas=2` 时，一个逻辑 CTA 是一对 CTA，把打分 GEMM 的**专家维（M）**
   对半切开、各写自己那半（`sm100_bf16_mega_gate.cuh:55-58,136-142`）。
2. **split-K + 全局 score scratch**：每个 split 只算 K 的一个切片，把**部分分**写进全局 scratch
   `[token_global][split][expert]`（`layout/mega_gate.cuh:78-107`）。
3. **ScoreBarrier**：一个 token block 的**所有逻辑 CTA** 各 arrive 一次，到齐后才允许读全部 split 部分分
   做归约 + top-k（`scheduler/mega_gate.cuh:13-39`）。

### 23.2 资料分析：Mega Gate 的 `cluster_dim` 路径（一手核对 `057ca59`）

- **task 分解**：一个 token block 由
  `num_logical_ctas = num_expert_groups * num_mma_ctas * num_split_k` 个 CTA 协作
  （`heuristics/mega_gate.hpp:133`；device 侧 `constexpr uint32_t kNumLogicalCtas` 在
  `sm100_bf16_mega_gate.cuh:55`）。`num_mma_ctas ∈ {1,2}`（`heuristics/mega_gate.hpp:122` 的循环），
  `num_split_k` 到 `kNumMaxSplitK=8`（`:54,126`）。
- **cluster_dim**：launch 时 `cluster_dim = dim3(config.num_mma_ctas, 1, 1)`
  （`impls/sm100_bf16_mega_gate.hpp:74`）——**一个 CTA pair 就是一个 cluster**。
- **CTA pair 的切分**：`LOAD_BLOCK_M = UMMA_N / kNumMmaCtas`（tokens/2）、
  `LOAD_BLOCK_N = UMMA_M / kNumMmaCtas`（experts/2）（`sm100_bf16_mega_gate.cuh:57-58`）；
  TMA 载入 warp 里 `expert_base_idx = expert_group_idx*kNumExpertsPerGroup + mma_cta_rank*LOAD_BLOCK_N`、
  `x_token_idx = ... + mma_cta_rank*(effective_umma_n/kNumMmaCtas)`（`:136-142`），注释
  *"A CTA pair splits both the experts (UMMA M) and the tokens (UMMA N) between its two CTAs"*（`:137`）。
  MMA 只由 leader CTA 发（`warp_idx==1 and is_leader_cta`，`:168-169`），完成信号用
  `umma_arrive_multicast_2x1SM` 一次性到达**一对 SM 的屏障**（`:191-197`）——这两点是 SM100 硬件行为，
  sm90 无对应（见 §22）。
- **ScoreBarrier 编码**（`scheduler/mega_gate.cuh:17-38`）：状态是一个 64-bit 全局字
  `encode_state(grid_idx, arrived) = (grid_idx+1)*kNumMaxLogicalCtas + arrived`；`init` 写 `(grid+1)*64`，
  `wait_init` 用**无符号距离**拒绝陈旧 grid 标签，`arrive = red.async.inc.release`，`wait` 等到达数
  `== kNumLogicalCtas`。barrier 行占 128 B（`layout/mega_gate.cuh:22`），上限 `kNumMaxLogicalCtas=64`
  （`:23`）。
- **split-K 归约**：score scratch 视图 `[num_token_blocks*block_tokens, kNumSplits, kNumExperts]`
  （`layout/mega_gate.cuh:78-102`）；gate warp 在 top-k 前 `load_scores` 把 `kNumSplitK` 个部分分
  **跨 split 求和**（`sm100_bf16_mega_gate.cuh:304-321`）。softplus（`sqrt_softplus`，`epilogue/
  sm100_mega_gate.cuh:19-23`）**只在 split 部分分求和之后**做一次（否则非线性不可换序，`:284-286` 注释）。
- **为什么用 multi-CTA 逻辑任务**：`heuristics/mega_gate.hpp:137` 取
  `num_worker_groups = num_sms / num_logical_ctas`——**逻辑 CTA 越多，同一时刻能并行处理的
  token block 越少**；当 token 数很少（token block 数 < SM 数）时，用 split-K / CTA-pair 把机器**占满**，
  这正是本节要实测的那个权衡。

**一句话**：Mega Gate 的 `cluster_dim` 路径 = **split-K 写全局部分分 → ScoreBarrier 全逻辑 CTA 会合 →
跨 split 归约 → top-k**，其中 CTA pair 是 SM100 硬件绑定的两个 SM。

### 23.3 方法（sm90 等价复刻）

程序 `src/mega_gate_cluster.cu`，脚本 `harness/mega_gate_cluster.sh`，原始输出
`harness/mega_gate_cluster.out.txt`（目标卡 H100 / sm90a，SMs=132，**固定空闲 device 3**，nvcc 13.2）。
复刻一个最小 gate task：`score[t][e] = x[t]·w[e]`（`T` tokens × `E` experts，K 维打分 GEMM），
再按 Mega Gate 的语义做 top-k（`sqrt_softplus` 在归约后、加 bias、按 biased 排名、输出 unbiased 分数
归一化）。两个变体：

- **`gate`（忠实路径）**：`num_logical_ctas = SPLIT * MMA`；`MMA=2` 时用 **cluster（cluster_dim=2）**，
  一对 CTA 把专家维对半切开（`Eper = E/2`，rank 各写自己那半）；每个逻辑 CTA 只算 K 的一个
  `1/SPLIT` 切片，部分分写全局 scratch `[token_global][split][expert]`；`ScoreBarrier`（按 grid 标签 +
  到达数编码）会合后，每个逻辑 CTA 认领自己那份 token，**跨 split 归约** + top-k。grid = `NT*SPLIT*MMA`。
- **`fused`（对照）**：单 CTA（`SPLIT=1,MMA=1`），分数留在**一块 SMEM**里，**不写全局 scratch、不设
  barrier**，直接在 SMEM 上 top-k。作为「不做 score round-trip」的上界。

GEMM 用 smem 分块 + 寄存器 micro-tile（`TM=TN=4`，行 stride `KT+1` 避 bank conflict，float4 载入），
score 先落 SMEM staging 再 **coalesced** 写全局 scratch。四 Section：**A** 正确性（6 组
`SPLIT×MMA` + fused，对参考**逐 token** 核 top-k 专家集合与权重）；**B** 占满机器（固定 `NT=16`，
靠 `SPLIT/MMA` 增加逻辑 CTA 数）；**C** 已饱和（`NT=128`，grid 已 ≈ #SM）；**D** ncu 量 score scratch
的 L2/DRAM 流量。

### 23.4 结果

**A. 正确性：split-K / CTA pair / fused 与参考逐 token 一致**

```text
  SPLIT MMA mode  | grid | 延迟(us) | idx_mismatch | wmax
 -----------------+------+----------+--------------+---------
      1   1  gate |  128 |  232.9   |     0/16384  | 0.00e+00
      2   1  gate |  256 |  356.6   |     0/16384  | 1.79e-07
      4   1  gate |  512 |  673.2   |     0/16384  | 2.09e-07
      1   2  gate |  256 |  260.3   |     0/16384  | 0.00e+00
      2   2  gate |  512 |  395.4   |     0/16384  | 1.79e-07
      4   2  gate | 1024 |  591.6   |     0/16384  | 2.09e-07
      1   1  fused|  128 |  209.8   |     0/16384  | 0.00e+00
```

`NT=128, T=128, E=128, K=512, TOPK=4`。**所有配置 top-k 专家集合与参考完全一致**（`idx_mismatch=0`），
权重最大误差仅在 split-K 浮点累加序差异量级（`~2e-7`）。C5 式的协调机制（cluster + 全局 barrier）
**不动数值**。

**B. 占满机器：固定少量 token（NT=16），靠 split-K / CTA-pair 增加逻辑 CTA 数**

```text
 SPLIT MMA | logical | grid | 延迟(us) | 相对 SPLIT=1
 ----------+---------+------+----------+-------------
   1   1   |    1    |   16 |  590.8   |   1.00×
   2   1   |    2    |   32 |  361.8   |   1.63×
   4   1   |    4    |   64 |  240.5   |   2.46×
   1   2   |    2    |   32 |  376.8   |   1.57×
   2   2   |    4    |   64 |  251.0   |   2.35×
   4   2   |    8    |  128 |  168.2   |   3.51×
```

`NT=16, T=128, E=128, K=2048`（token 少 → SPLIT=1 时只 16 个 CTA、机器空转）。**token 数不变、
只增加逻辑 CTA 数，延迟单调下降**：`grid 16→128` 拿到 **3.51×**。这正是 Mega Gate 在 token block
数少时用 `num_logical_ctas`（split-K / CTA-pair）**填满 SM** 的动机（`heuristics/mega_gate.hpp:137-147`）。
`grid=32` 处 `SPLIT=2`（1.63×）略优于 `MMA=2`（1.57×）——同样 2 个逻辑 CTA，split-K 把 K 也切了、
单个 CTA 的 K 负载更小，故略快。

**C. 已饱和（NT=128，grid 已 ≈ 132 SM）：split-K 不再填机器，只加协调开销**

```text
 SPLIT MMA mode  | grid | 延迟(us) | 相对 gate SPLIT=1
 -----------------+------+----------+-------------------
      1   1  gate |  128 |  232.9   |   1.00×
      2   1  gate |  256 |  356.6   |   1.53×   (2 波)
      4   2  gate | 1024 |  530.3   |   2.28×   (8 波)
      1   1  fused|  128 |  209.2   |   0.90×
```

`K=512`。此时 `SPLIT=1` 已把 128 个 CTA 铺满机器，**再 split 只会让 grid 超过 SM 数、变成多波**
（grid 256 = 2 波、1024 = 8 波），wall time 反而涨。**`fused`（无 scratch / 无 barrier）比
`gate SPLIT=1` 快 ~11%**——这 ~24 µs 就是 score scratch 往返 + ScoreBarrier 在**饱和区间**的净成本。
与 B 合并的结论：**multi-CTA 逻辑任务的价值在「填机器」，不在「更快」**；机器一旦被填满，
它们只剩协调开销。

**D. ncu：score scratch 的 L2 / DRAM 流量**

```text
 配置             | L2 read sectors | L2 write sectors | DRAM read | DRAM write
 -----------------+-----------------+------------------+-----------+-----------
 gate SPLIT1 MMA1 |    2,100,297    |    2,687,074     | 33.86 MB  |  2.57 MB
 gate SPLIT4 MMA2 |    3,966,044    |    4,000,361     | 33.87 MB  | 19.84 MB
 fused SPLIT1     |    2,079,750    |    2,294,813     | 33.86 MB  |  1.94 MB
 ```

`NT=128, T=128, E=128, K=512`，取第 2 次 launch（`--launch-skip 1`，避开首次 L2 冷启动）。
**`fused` 与 `gate SPLIT=1` 的 L2 读几乎相同**（2.08M vs 2.10M sectors），**写差 +0.39M sectors
（≈+12.6 MB）**——score scratch 一次落盘的量级（`NTT*E*4 = 8.4 MB`，余下为写放大/对齐）；
`SPLIT4 MMA2` 因 8× 逻辑 CTA + 4× split，L2 写升到 4.0M sectors、**DRAM 写从 1.94 MB 涨到
19.84 MB**。含义：**split-K 的部分分必须经过 L2/DRAM 一次往返回合**，这正是 `docs/04` §17（P24-1）
量到的「损失三」在 gate 里的具体形态——只不过 Mega Gate 用**一次** score round-trip（而非每算子一趟）
把整段打分 + 排名压进单 kernel。

### 23.5 读法

1. **Mega Gate 的 `cluster_dim` 路径可整体投影到 sm90**：CTA pair 切专家维、split-K 写全局部分分、
   ScoreBarrier（grid 标签编码的 64-bit 原子）会合、跨 split 归约 + top-k——**数值与参考逐 token 一致**。
   真正 sm90 拿不到的是 §22 的 `cta_group::2`（一次 TMA、一个屏障、一对 SM 分持 TMEM），
   本节用**软件 cluster + 独立 barrier** 代替。
2. **multi-CTA 逻辑任务（split-K × CTA-pair）是「occupancy 填充器」**：token 少时 `grid 16→128`
   快 **3.51×**（Section B）；token 多、机器已满时它只剩协调开销（Section C，fused 快 11%）。
   这解释了 Mega Gate heuristic 为何按 `num_worker_groups = num_sms / num_logical_ctas` 选配置
   （`heuristics/mega_gate.hpp:137`）——**逻辑 CTA 数 × worker group 数 ≈ SM 数**才最优。
3. **split-K 的代价是「部分分往返回合」**：score 写全局 + barrier + 跨 split 读归约，在饱和区间约
   +24 µs / +11%（Section C），L2 写 +0.39M sectors（Section D）。Mega Gate 接受它，是因为
   整段打分 + 排名只走**一次** round-trip；若把它拆回 kernel-per-operator 会走多次（`docs/04` §17）。

### 23.6 与判据 / MPK / 生产案例的关系

- **判据（C1–C5，`docs/00` §3）**：本节的 gate 任务满足 C1（单 launch）、C2（CTA 常驻）、
  C3（段内多算子范围）、C4（设备内协调：cluster + 全局 ScoreBarrier），仍**非 C5**（无 worker/scheduler
  动态派发，按静态 grid 角色计算）。与 §18–§22 同档——它们都是**段级 T4 的设备内协调原语**；
  完整 C5 由 P15-1 §9 / P22-1 §15 覆盖。
- **与 MPK 的关系**：MPK 的 tGraph 用 event 计数器表达「同 event 触发的 task 连续」（`docs/01` §3）；
  Mega Gate 的 ScoreBarrier 是它的**手工版**——一个 token block 的 `kNumLogicalCtas` 到达即放行下游
  排名（`scheduler/mega_gate.cuh:31-38`）。MPK §5.1 的「consumer 等 `num_triggers*iteration`」与
  `wait(state == (grid+1)*64 + kNumLogicalCtas)` 是同一个思路（精确计数 + grid 代次去陈旧）。
- **与生产案例**：本节的 `Cluster ST1/2` 复刻了 Mega MoE `cluster_dim=(2,1,1)`
  （`sm100_bf16_mega_moe.hpp:138`）、Mega Gate `cluster_dim=num_mma_ctas`
  （`sm100_bf16_mega_gate.hpp:74`）的**最小单位 = 2SM 对**；Section B 的 filling 规律对应 Mega Gate
  heuristic 的 `num_logical_ctas` 选择（`heuristics/mega_gate.hpp:133-147`）。细节见 `docs/02` §1.2。

### 23.7 局限与踩到的坑

- **这是 sm90 等价复刻，不是 SM100 Mega Gate 本体**：无 tcgen05、无 TMEM、无 `cta_group::2`；
  CTA pair 用软件 cluster + 独立 SMEM/global barrier 代替硬件的「一次 MMA、一个屏障、一对 TMEM」。
  绝对性能（~7 TFLOP/s 的朴素 fp32 GEMM）远低于 SM100 的 UMMA，故本节只做**相对**比较。
- **GEMM 是 SIMT fp32**：打分用 register-tiled 朴素 GEMM，非 Tensor Core；`T*Eper` 受寄存器 micro-tile
  限制（`T=128, Eper≤128`）。这里量的是**协调路径**，不是 GEMM 峰值。
- **`lts__t_sectors_op_write` 绝对值偏大**：包含 score staging 之外的写放大与多波重复载入；
  本节只用**配置间的差**（fused vs gate）与 DRAM 写做定性证据，不把绝对值当精确 scratch 字节。
- **CTA pair 的切法是专家维（M），不是 Mega Gate 的「同时切 M 和 N」**：Mega Gate 的 2SM MMA
  同时切 experts（M）与 tokens（N）（`sm100_bf16_mega_gate.cuh:137`），sm90 无 2SM MMA，
  本节让两 CTA 都处理全部 token、各写半数专家——是对「切开 M」的忠实投影，但少了 N 方向的共享。
- **未测 gate warp / top-k 的内部并行**（Mega Gate 的 gate warpgroup 之间的 transpose/top-k 分摊，
  `epilogue/sm100_mega_gate.cuh:107-152`）：本节的 top-k 是每线程一 token 的朴素插入排序。
- **ScoreBarrier 的 grid 代次**由 host 每次 launch 递增传入（而非设备侧 `get_grid_idx()`）；
  语义（无符号距离去陈旧）与 `scheduler/mega_gate.cuh:17-29` 一致，实现细节不同。

## 24. 显式构图 + PROGRAMMATIC 边：CUDA Graph 能把下游调度点前移到上游完成之前吗（P35-1）

### 24.1 问题

CUDA Graph 在本专题有两个**未覆盖**的尾巴，都指向同一件事——**图能不能表达「比完成更早」的依赖**：

- `docs/04` §13.6（P19-1）只测了 **capture 方式**构造的图（`cudaStreamBeginCapture`），明确留了
  「未测**显式构图** API（`cudaGraphAddKernelNode` + `cudaGraphKernelNodeSetAttribute` 的 PDL 属性）
  以及 graph 内部的 programmatic edge，后者语义可能不同」。
- `docs/04` §21.6（P32-1）在 **stream 上**测出：programmatic/LCE 事件的前移 trigger **不放行**
  cross-stream `cudaStreamWaitEvent` 依赖（只在完成点放行），并留了「未测 CUDA Graph 的
  `PROGRAMMATIC` 边」——`CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC`（`cuda.h:2060`）、
  `CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC`（`cuda.h:2044`）。

本节把这两条尾巴一起补上，回答一个问题：**用显式构图把一个「programmatic 边」做进图里，能否让下游
kernel 在上游完成之前就被调度**？如果能，那它到底把放行点前移到哪一点、和 capture 图是否等价。

### 24.2 资料分析：图的 programmatic 边（头文件一手核对，CUDA 13.2）

图的依赖不只是布尔「有/无」，而是一条带**端口**的边（`cudaGraphEdgeData`，`driver_types.h:3719-3748`）：

- `from_port`：**上游**在哪个端口触发这条边。kernel 节点定义了三个端口
  （`driver_types.h:3750-3761`）：
  - `cudaGraphKernelNodePortDefault = 0`：上游**执行完毕**（含内存可见性）；
  - `cudaGraphKernelNodePortProgrammatic = 1`：**所有 block 都调用过**
    `cudaTriggerProgrammaticLaunchCompletion()`（或已终止）；
  - `cudaGraphKernelNodePortLaunchCompletion = 2`：**所有 block 都开始执行**。
- `type`：`cudaGraphDependencyTypeDefault = 0`（普通边）或
  `cudaGraphDependencyTypeProgrammatic = 1`（`driver_types.h:3708-3715`）。文档明确：
  programmatic 类型「**允许下游节点使用 `cudaGridDependencySynchronize()`**」，且必须与
  `Programmatic` 或 `LaunchCompletion` 端口搭配。
- `to_port`：下游的哪个部分依赖上游——**当前无节点类型定义非零 `to_port`，必须为 0**
  （`driver_types.h:3731-3734`）——即**下游永远是整块依赖**，没有「下游的某一片先跑」。

构造 API：`cudaGraphAddKernelNode`（`cuda_runtime_api.h:9796`）加节点、
`cudaGraphAddDependencies(graph, from, to, edgeData, n)`（`cuda_runtime_api.h:11783`）按端口加边。
`cudaGraphKernelNodeSetAttribute`（`cuda_runtime_api.h:9924`）可设节点属性，但——关键——
`cudaLaunchAttributeProgrammaticStreamSerialization`（=6）与 `cudaLaunchAttributeLaunchCompletionEvent`
（=12）在头文件里都标着 **"Valid for launches"**（`driver_types.h:3988 / 4067`），**不在** graph node
可用属性列表（`AccessPolicyWindow / Cooperative / ClusterDimension / …`）里。故本实验**预期**节点级
PDL 属性会被驱动拒绝，programmatic 端口**只由边本身提供**——这由实测验证（见 §24.4）。

### 24.3 方法

程序 `src/graph_programmatic.cu`，脚本 `harness/graph_programmatic.sh`，原始输出
`harness/graph_programmatic.out.txt`（主配置）与 `harness/graph_programmatic_sens.out.txt`
（`pro > tail` 的敏感性检查）（目标卡 H100 / sm90，SMs=132，**固定空闲 device 3**，nvcc 13.2，
`iters=30`）。一张图两个 kernel 节点 A→B，用 `%globaltimer`（**全局一致**的 ns 计时器；`clock64` 是
per-SM、不能跨 block 比时间）打点：

- **A（上游，G 个 block）**：开始打 `g[0]` → `prologue`（block `i` 额外加 `i*stagger`）→
  **可选** `cudaTriggerProgrammaticLaunchCompletion()` → `tail` → 结束打 `g[1]`；
- **B（下游，1 个 block）**：入口打 `g[2]` → `prologue` → `cudaGridDependencySynchronize()` → 打 `g[3]`
  → `crit` → `atomicAdd(g[4])`（正确性哨兵，应恒 `== iters`）。

观测量：`A_total = g[1]-g[0]`；**`sched = g[2]-g[0]`**（B **何时被调度拿到 SM**，本节的关键量）；
`rel = g[3]-g[0]`（B 的 `gridDependencySynchronize` 何时返回）。四档边（同一 A/B workload 公平对照）：

| 边 | `from_port` | `type` | A 是否 trigger |
|---|---|---|---|
| `block` | Default(0) | Default | 否（完成即放行） |
| `prog_nt` | Programmatic(1) | Programmatic | **否**（端口在终止时激活） |
| `prog` | Programmatic(1) | Programmatic | 是（prologue 后 trigger） |
| `lce` | LaunchCompletion(2) | Programmatic | 否（全体 block 启动即激活） |

### 24.4 结果

**A. G=1，四档边：programmatic 边把 `sched` 前移到 trigger 点，但 `rel` 仍在完成点**

```text
    edge | A_total(us) |  sched(us) |    rel(us) |    rel/A |     done
---------+------------+------------+------------+----------+---------
   block |     101.15 |     101.86 |     152.42 |    1.507 |       30
 prog_nt |     101.15 |     101.60 |     152.16 |    1.504 |       30
    prog |     101.12 |      51.01 |     101.60 |    1.005 |       30
     lce |     101.12 |       0.32 |     101.54 |    1.004 |       30
```

`pro=tail=100000` cycles（各 ≈50.5 µs），`A_total≈101.1 µs`。读法：

- **`block` / `prog_nt`**：`sched≈A_total`——普通边（完成即放行）与「programmatic 边但 A 不 trigger」
  都**要等 A 完成**才调度 B（`prog_nt` 的端口在 A 终止时才激活，等价于完成）；
- **`prog`**：`sched=51.01 µs ≈ pro`——A 在 prologue 后 trigger，**B 在上游 tail 还没跑完时就被调度**，
  其 prologue 与 A 的 tail 重叠。**这就是 stream 上 `cudaStreamWaitEvent` 做不到的前移**（§21）；
- **`lce`**：`sched=0.32 µs ≈ 0`——全体 block 一启动就放行；
- **但 `rel` 恒 `≈ A_total`**（1.004–1.507 里，`prog`/`lce` 都是 1.00）：`gridDependencySynchronize()`
  **依旧等上游完成**。端口只决定下游**何时被调度**，不改变「等完成」的语义——与 §21 的 stream 结论
  完全一致。`done` 四档恒 30。

节点级 PDL 序列化属性确实被驱动**拒绝**（输出里 `[note] …（invalid argument）——端口只由 programmatic
边提供`），印证 §24.2 的头文件判断。

**B. 多 block + stagger：programmatic 端口在「所有 block 都 trigger」时激活**

```text
    G | (G-1)*stag(us) | block A(us) | prog_sched |   prog_rel |    lce_rel |  prog_done
------+--------------+------------+------------+------------+------------+-----------
    1 |         0.00 |     101.12 |      51.01 |     101.60 |     101.57 |         30
    2 |        10.10 |     101.12 |      61.12 |     111.68 |     111.62 |         30
    4 |        30.30 |     101.12 |      81.28 |     131.87 |     131.78 |         30
    8 |        70.71 |     101.15 |     121.73 |     172.29 |     172.22 |         30
   16 |       151.52 |     101.12 |     202.53 |     253.09 |     253.02 |         30
   32 |       313.13 |     101.12 |     364.13 |     414.72 |     414.66 |         30
```

block `i` 的 prologue 加 `i*stagger`（`stagger=20000` cyc ≈10.1 µs）。**`prog_sched ≈ pro + (G-1)*stagger`**
（51.0、61.1、81.3、121.7、202.5、364.1 对 50.5、60.6、80.8、121.2、202.0、363.6）——programmatic
端口在**最后一个 block** 也 trigger 之后才激活（`driver_types.h:3753-3756` 的"all blocks"语义）。
`prog_rel`/`lce_rel` 两列都随 `(G-1)*stagger` 增长并趋同（两者都 `= max(调度点+prologue, 最后一个 A
block 完成)`；本配置里后者主导）。`block A(us)` 恒 101——`g[1]` 由 block0 写，只有公共段 `pro+tail`。

**敏感性检查**（`harness/graph_programmatic_sens.out.txt`，`pro=200000, tail=100000`）：
`prog` 的 `sched=101.54 ≈ pro=101.0 µs`、`rel=152.13 ≈ A_total=151.6 µs`——**`sched` 精确跟随
trigger 点**（而非 `tail` 或别的时间量），确认它测的就是「上游端口激活 → 下游被调度」。

**C. capture 图（P19-1 形态）vs 显式 programmatic 边：逐项一致**

```text
构图                   | A_total(us) |  sched(us) |    rel(us) |    rel/A |     done
-------------------------+------------+------------+------------+----------+---------
capture (PDL attr)       |     101.12 |      51.01 |     101.60 |    1.005 |       30
explicit PROGRAMMATIC    |     101.15 |      51.04 |     101.60 |    1.004 |       30
```

`capture`（A 带 `ProgrammaticEvent` 事件属性 + B 带 PDL 序列化属性、同 stream capture）与
`explicit`（`cudaGraphAddKernelNode` + programmatic 边）**`sched`/`rel`/`A_total` 逐项相同**
（51.01 vs 51.04、101.60 vs 101.60）。即：**capture 与显式构图在 PDL 语义上等价**——显式构图
多出来的能力是能**显式选端口**（`Programmatic` vs `LaunchCompletion` vs `Default`），capture 只能
从被 capture 的 launch 属性推出唯一的 programmatic 端口。

### 24.5 读法

1. **图的 programmatic 边 = 「调度级前移」，不是「依赖级前移」**。它能把下游 kernel 的**调度点**
   （`sched`）提前到上游完成之前（`prog`→trigger 点、`lce`→0），从而让下游 prologue 与上游 tail
   重叠（`min(tail, prologue)`，与 P5-3/P19-1 的 PDL 收益同源）；但下游一旦调用
   `cudaGridDependencySynchronize()`，**仍等上游完成**（`rel≈A_total`）。这与 §21 在 stream 上的结论
   **完全一致**：kernel 级没有「中间进度早于完成、可被依赖方看到」的原语。
2. **端口语义**：`Programmatic` 端口 = 「**所有** block 都 trigger」（故 `sched = pro + (G-1)*stagger`，
   Section B）；`LaunchCompletion` 端口 = 「所有 block 都开始」（`sched≈0`）。想早放行，上游必须
   **每个 block** 都到 trigger——单点 trigger 不够（这正是 MPK 把协调压到 task 级、DeepGEMM 用
   ring barrier 的原因：kernel 级端口太粗、且要全体会合）。
3. **capture ≡ 显式构图**（Section C），补齐 §13.6；**程序matic 边可用**、节点级 PDL 属性被拒
   （补齐 §21.6）。三者合起来把「CUDA Graph 这条降级路线能走到哪」钉死在**调度级**。

### 24.6 与判据 / MPK 的关系

- **判据（C1–C5，`docs/00` §3）**：本节四档边全是 **kernel-per-operator 的降级项**，**均不满足 C1–C5**
  （图仍是 host 提交的多次 kernel 边界，kernel 仍会结束，无设备内调度）。它补的是「Graph 能表达
  哪一级依赖」这条**下界证据**：也就是说，把 PDL 升级成图的 programmatic 边，**仍然只到调度级**。
- **与 MPK 的关系**：MPK 用 **in-kernel event counter**（`docs/01` §3、`docs/06` §3）把「一个 task
  完成」做成设备内、可被**任意** worker 观察的细粒度信号，粒度是 **task**（比 kernel 的
  「全体 block 会合」细得多），且**不需要全体块同步**。本节的 programmatic 边是它的**最粗版本**：
  粒度 = 整个下游 kernel、触发 = 全体 block、可见性 = 只对设备内 `gridDependencySynchronize`。
  这正是 `docs/03` §2.6 那句「kernel 级没有可被别的执行体看到的中间进度」的图版本复核。
- **与生产案例**：DeepGEMM Mega 家族用**单 kernel 内的原子 ring barrier**（`docs/02` §1.5）而不是
  kernel 级事件，理由与本节一致——kernel 级端口/事件无法提供 task 粒度的「谁做完了」；MPK §5.3 的
  cross-task software pipelining（P23-1 §16）也建立在 task 级事件队列上。

### 24.7 局限与踩到的坑

- **合成 workload**：A/B 都是「自旋 + 打点」，测的是**边端口语义**，非端到端收益；`B` 只 1 个 block。
- **`rel` 的口径**：`rel = 下游 gridDependencySync 返回`，它同时受「下游何时被调度 + 下游自身
  prologue」影响，不是纯放行信号；故本节的结论量用 **`sched`**（下游调度点），`rel` 只作「sync 仍等
  完成」的反证。
- **只测 2 节点、单 kind 边**：未测「一条 programmatic 边扇出到多个下游」「下游再作为上游」的级联，
  也未测 `to_port`（当前恒 0，见 §24.2）。
- **`LaunchCompletion` 端口的边类型**：本节用 `type=Programmatic`（头文件对 Programmatic 端口规定
  必须如此，`driver_types.h:3753-3756`；LaunchCompletion 端口未硬性要求），实测被接受；未对比
  `type=Default` 的行为差异。
- **device 3**：主运行与敏感性检查都在固定空闲 device 3；`%globaltimer` 已消除跨 SM 时钟差，
  但首轮图 instantiate 的固定开销未单独剔除（用 `iters=30` 平均摊薄）。

## 25. CUDA Graph 的条件/循环节点：一张图能不能表达「数据相关 / 可变行程」控制流（P36-1）

### 25.1 问题

CUDA Graph 在本专题已被反复测成**静态 DAG**：P4-1（§1）量了 launch 摊薄、P19-1（§13）量了 capture
保留 PDL 语义、P35-1（§24）量了 programmatic 边只把**调度点**前移。这条降级路线还有一条**没碰过的
维度**——**控制流**：图能不能表达「按数据决定跑不跑、跑几轮」的**动态/迭代**工作流？

这不是纯技法问题，它正对着 MPK 的一条**核心局限**：

- MPK 的 tGraph 是**每 batch size 一张静态图**（`docs/01` §5.8 局限 5；MPK §6.1 行 1427–1436）、
  Qwen3-8B 每 token 293 task（`docs/00` §1）；
- 「多轮服务」不是靠图循环，而是靠运行层的 **`iteration` 计数**：TaskId 高 32 位 = iteration
  （`docs/06` §2/tGraph，快照 `runtime_header.h:73-74`），事件阈值是 `num_triggers * iteration`
  （`runtime_header`/`persistent_kernel.cuh:1090-1093`），`EVENT_END_OF_TASK_GRAPH` 触发
  `prepare_next_batch` 会**在同一常驻 kernel 内**组织下一 batch（`docs/06` §4，`persistent_kernel.cuh:1378-1423`）。

CUDA 12.3 起，CUDA Graph 原生支持 **conditional graph nodes**（IF / WHILE / SWITCH）：条件值**在设备侧
求值**，允许**带循环/分支的工作流完全落在图内**（官方 CUDA Programming Guide §4.2.4「Conditional Graph
Nodes」）。本节把这条能力放到和 MPK 同一张桌上量：**一张图/一个 graphExec 能否覆盖「可变行程 N」和
「数据相关分支」？如果能，它的每行程成本是多少——和 MPK 的 in-kernel `iteration` 循环比，是更省还是更贵？**

### 25.2 资料分析：条件节点的语义与边界（头文件一手核对，CUDA 13.2）

- **三种条件节点**（`driver_types.h:3559-3561` 的 `cudaGraphCondTypeIf/While/Switch`）：
  `If` = 条件非零跑一次 body（可带 else）；`While` = 条件非零就反复跑 body；`Switch` = 条件等于 n
  就跑第 n 个 body。
- **条件句柄**：`cudaGraphConditionalHandleCreate(&h, graph, defaultLaunchValue, flags)`
  （`cuda_runtime_api.h:13362`），`flags=cudaGraphCondAssignDefault`（`driver_types.h:3551-3554`）
  表示**每次图执行开始把条件重置为 default**——正是「每次调用都重新进循环、由 body 决定何时退出」所需。
- **设备侧写条件**：`__device__ void cudaGraphSetConditional(handle, value)`
  （`cuda_device_runtime_api.h:476-480`；注释警告**并发调用是 UB**，须单线程写）。
- **body 图由驱动分配**：`cudaConditionalNodeParams` 的 `phGraph_out`（`driver_types.h:3413`）是
  *"CUDA-owned array populated with conditional node child graphs during creation of the node"*——即
  创建条件节点后从 `phGraph_out[0]` **读回**一张空 body 图，再用普通 kernel 节点填充（实测见 §25.4）。
- **限制**（`driver_types.h:3616-3627`）：条件图**不能作为 child node**、**同一时刻只能有一个
  instantiation**、**不能 clone**；body 节点类型仅限 kernel/empty/memcpy/memset/child graph/conditional。
  这三条对「同一 graphExec 多流并发重放」「把 megakernel 图嵌进更大图」是硬边界。

一句话：**条件节点把「控制流」从 host 搬进图/设备，但粒度仍是「kernel 节点级重分发」**——这一点决定了
它和 MPK 的 in-kernel `iteration` 循环**不在同一成本档位**（§25.5）。

### 25.3 方法

程序 `src/graph_conditional.cu`，脚本 `harness/graph_conditional.sh`，原始输出
`harness/graph_conditional.out.txt`（目标卡 H100 / sm90，SMs=132，**固定空闲 device 3**，nvcc 13.2，
`iters=50`，`Nmax=64`，`G=32` blocks × `T=256` 线程，`work=64` FMA/线程/行程）。迭代体
`iter_kernel`：每线程对自己独占的 `acc[i]` 做 `work` 轮 FMA → `thread0` 递增行程计数 `iterc`，
（可选）按 `iterc < N` 写条件句柄——与持久化内核的一个 task body 同构。四策略**公平对照**
（同一 workload、同一表达式与累加顺序）：

| 策略 | 结构 | 行程 N 变化时 |
|---|---|---|
| `eager` | host 循环 N 次 `iter_kernel` launch | 需要 host 循环 |
| `unroll` | **N 个串联 kernel 节点**的静态图 | **每个 N 重新 instantiate** |
| `cwhile` | **1 个 WHILE 条件节点**（前置 reset 节点 + body 一个 `iter_kernel`） | **同一个 graphExec 服务任意 N** |
| `kinternal` | **单 kernel 内 for 循环 N 次**（持久化内核微缩版） | 一次 launch |

Section A 扫 N（work=64）；Section B 扫 N（work=0，纯控制）；Section C 实例化成本（host 时间）与
数据相关分支（IF 节点）；Section D 四策略 `acc[]` 逐位交叉校验。

### 25.4 结果

**A. 行程 N 扫描（work=64，µs/调用）**

```text
     N |      eager(us) |     unroll(us) |     cwhile(us) |  kinternal(us)
-------+----------------+----------------+----------------+----------------
     1 |          2.509 |          2.332 |         11.780 |          2.287
     2 |          4.810 |          3.687 |         16.301 |          2.418
     4 |          9.411 |          6.482 |         25.626 |          2.854
     8 |         18.628 |         11.534 |         42.762 |          3.808
    16 |         37.701 |         21.517 |         80.589 |          5.627
    32 |         73.946 |         41.615 |        154.188 |          9.313
    64 |        142.824 |         82.033 |        301.716 |         16.729
```

四条曲线**全部对 N 线性**（斜率即每行程成本）：`eager` 2.23、`unroll` 1.27、`cwhile` **4.60**、
`kinternal` **0.229 µs/行程**；`cwhile` 固定项 ~7.2 µs（条件节点簿记 + reset 节点）。N=64 时相对
`eager`：`unroll` **0.574×**、`cwhile` **2.11×**、`kinternal` **0.117×**。

**B. 纯控制开销（work=0，µs/调用）——把「每行程」的机器成本单独拎出来**

```text
     N |      eager(us) |     unroll(us) |     cwhile(us) |  kinternal(us)
-------+----------------+----------------+----------------+----------------
     1 |          2.260 |          2.198 |         11.418 |          2.134
     8 |         17.508 |         10.004 |         41.391 |          2.139
    64 |        140.156 |         70.608 |        282.419 |          2.138
```

`eager` 2.19、`unroll` 1.09、`cwhile` **4.30 µs/行程**；`kinternal` **恒定 ~2.14 µs**（N 无关——一次
launch，循环在 kernel 内）。**关键**：`cwhile` 的每行程成本 ~4.3 µs **比 host `eager` 的一次 launch
（~2.2 µs）还贵**——条件节点只是把「一次 kernel 分发」从 host 挪到了**设备/graph 侧**，并没有把它
消掉。

**C. 实例化成本与数据相关分支**

```text
   实例化(host)  |  unroll(us) |  cwhile(us)
   --------------+-------------+-------------
   N=1           |         5.9 |        16.1
   N=64          |        66.5 |        15.7
```

`unroll` 的 instantiate 时间随 N **近线性**（~1.0 µs/节点；N=64 → 66.5 µs），`cwhile` **常数 ~15.5 µs**
（图只有 2 个节点，与 N 无关；N=16 处 38.5 µs 是噪声）。**IF 节点**（条件由上游 `pred_kernel` 从数据
`x[0]` 算出）：同一 graphExec，`x[0]=100>10` → body 执行（`flag=12345`），`x[0]=3` → body 跳过
（`flag=0`）——**一张静态图按设备侧数据走不同分支**。

**D. 正确性**：四策略在 N=64 的 `acc[]` **逐位一致**（`max|Δ|=0.000e+00`，`max rel|Δ|=0`）；
`cwhile` 的行程计数恒 `== N`（A 表最后一列）。即：控制流换成条件节点**不动数值**。

### 25.5 读法

1. **条件节点解决的是「表达能力」，不是「性能」**。它让**一个 graphExec** 覆盖任意行程 N
   （Section A/B 的 `cwhile`）和设备侧数据相关分支（Section C），从而在**结构上**解除 MPK
   「每 batch size 一张静态图」的限制——不再需要按 2 的幂预生成多张图。但代价是**每行程一次
   设备侧 kernel 重分发**（~4.3–4.6 µs），**比 host eager launch（~2.2 µs）还贵 ~2×**，
   且含 ~7 µs 固定项。
2. **`kinternal`（持久化内核内循环）是数量级更优的下限**：每行程 **~0.23 µs**（比 `cwhile` 便宜
   **~20×**），且**纯控制下与 N 无关**（一次 launch）。这正解释了 MPK / DeepGEMM 为什么把「多轮」
   做成 **in-kernel `iteration` 计数 + `prepare_next_batch`**（`docs/06` §4）而不是图循环——图循环
   是「设备侧反复 launch kernel」，in-kernel 循环是「同一批 block 直接进下一轮」，两者差一个
   **kernel 分发**的量级。
3. **静态展开（`unroll`）仍然是纯 launch 摊薄的赢家**（~0.5–0.57× eager、~1.1–1.3 µs/节点，
   与 P4-1 的 ~0.78 µs/节点同档），但**每个 N/形状要重新 instantiate**（~1 µs/节点，N=64 → 66 µs）。
   对一个 293 launch/token、需按 batch size 切换的整图，instantiate 成本本身就是 MPK 把它搬进
   常驻 kernel 的理由之一。
4. **三条路线各有硬边界**：`unroll` = 便宜但需按形状重实例化；`cwhile` = 一图通用但每行程贵且
   受「不能作 child、只能单 instantiation、不能 clone」限制（`driver_types.h:3616-3627`）；
   `kinternal` = 最省但需要「常驻 kernel + 设备内调度」这一整套机制（C1–C5）。
   CUDA Graph 的控制流**填不满** MPK 需要的那一格。

### 25.6 与判据 / MPK / 生产案例的关系

- **判据（C1–C5，`docs/00` §3）**：四策略**均不满足 C1–C5**——`eager`/`unroll` 是 host 提交的多次
  kernel 边界；`cwhile` 虽由设备求值条件，但每次迭代仍是**一个 kernel 节点整体重分发**，kernel 仍会
  结束，**无 task 级设备内调度**；`kinternal` 满足 C1/C2（单次 launch + 常驻）但**没有** C3（多算子
  范围）/C4（算子间设备内协调）/C5（去中心化调度）——它只是「同一 kernel 内重复同一段代码」。
- **与 MPK 的关系**：MPK 的 `iteration`（`docs/06` §2/§4）与本节 `cwhile` 是**同一诉求的两种实现**：
  前者把「下一轮」做成常驻 kernel 内的**下一批 task**（零 kernel 边界，`kinternal` 档），后者做成
  **图上的一次设备侧 kernel 重分发**（`cwhile` 档）。量级证据（~0.23 vs ~4.3 µs/行程）支持 MPK 的
  选择。条件节点**只能补「表达变长/数据相关」，不能补「省下 launch」**。
- **与其它降级项的关系**：§24 把 CUDA Graph 的**依赖**维度钉死在「调度级」；本节把**控制流**维度
  钉死在「节点重分发级」。二者正交，合起来说明 **CUDA Graph 这条降级路线的天花板**：它能表达动态
  依赖的*调度时机*与动态控制流，但**粒度恒停在 kernel 节点**，拿不到 task 级、in-kernel 的成本。

### 25.7 局限与踩到的坑

- **合成 workload / 单迭代体**：body 是 `G=32` block 的一次 FMA；`cwhile` 的固定项（~7 µs）与
  `iter_kernel` 的 grid 大小/实现细节相关，绝对数**只作量级**，不比生产端到端。
- **`cwhile` 的 instantiate 出现一个 N=16 离群点**（38.5 µs）：单次 host 计时噪声；趋势（unroll 线性、
  cwhile 常数）在两轮运行一致。
- **`G=32` 非整机**：未扫 grid 对条件节点每行程成本的影响（预期随 grid 增大而抬高，留待后续）。
- **只测 IF / WHILE**：未测 SWITCH、嵌套条件、条件体内的 memcpy/memset/child graph（语义已由头文件
  界定，未实测）。
- **device 3**、`iters=50` 平均摊薄首个 launch 的冷启动。

## 26. Device Graph Launch：设备侧图启动能不能当「in-kernel 分发」用（P37-1）

### 26.1 问题

`docs/04` §1（P4-1）走 CUDA Graph 时留下一句未核实的判词——**「device graph launch 仍停在 kernel 级」**：
CUDA 允许把一个图**实例化到可从设备启动**（`cudaGraphInstantiateFlagDeviceLaunch`），再由一个**设备上
运行的 kernel** 用 `cudaGraphLaunch` 启动它。这正好是「设备自己当派发器（dispatcher）」的最直接形态，
与 megakernel 的 **in-kernel 分发**（MPK worker/scheduler、DeepGEMM ring barrier，`docs/06` §3 / `docs/02`
§1.5）是同一个诉求的**最粗版本**。§25（P36-1）已把「设备侧 kernel 重分发」在**条件节点**里量到
**~4.3–4.6 µs/行程**、把 in-kernel 循环量到 **~0.23 µs/行程**。本节把 device graph launch 这一路也量出来，
回答：**设备侧启动一个图，比 host 启动便宜吗？能不能替代常驻内核自己的循环？**

### 26.2 资料分析：device graph launch 的 API 与约束（头文件一手核对，CUDA 13.2）

- **实例化**：`cudaGraphInstantiateWithFlags(&exec, g, cudaGraphInstantiateFlagDeviceLaunch)`，标志值
  **4**；头文件注明「只能用于支持 unified addressing 的平台，且**不能**与
  `cudaGraphInstantiateFlagAutoFreeOnLaunch` 并用」（`driver_types.h:3910`）。
- **上传**：`cudaGraphUpload(exec, stream)`（`cuda_runtime_api.h:12940`）后图才驻留在设备上可被设备启动。
- **设备侧入口**：`__device__ cudaGraphLaunch(cudaGraphExec_t, cudaStream_t)`
  （`cuda_device_runtime_api.h:290`）。设备侧只能用三个**特殊流句柄**：
  `cudaStreamGraphTailLaunch`（`0x0100…`，父图完成后以「尾巴」方式串行启动）、
  `cudaStreamGraphFireAndForget`（`0x0200…`，与父并发、无人等待）、
  `cudaStreamGraphFireAndForgetAsSibling`（`0x0300…`）（`cuda_device_runtime_api.h:175-177`）。
- **自引用**：`__device__ cudaGetCurrentGraphExec()`（`cuda_device_runtime_api.h:311`，读
  `%current_graph_exec`）返回**当前正在执行的 exec**——这是「图自重启 / 常驻图循环」的关键原语。
- **构建约束**：设备侧 CUDA runtime（含 `cudaGetLastError` 等）需要**分离编译**
  （`nvcc -rdc=true … -lcudadevrt`），否则 `ptxas` 报 `Unresolved extern function '__cudaCDP2GetLastError'`
  （本轮编译踩到）。

### 26.3 方法

程序 `src/device_graph.cu`，脚本 `harness/device_graph.sh`，原始输出 `harness/device_graph.out.txt`
（主配置 `N=20000, reps=5`）与 `harness/device_graph_sens.out.txt`（敏感性 `N=2000, reps=20`）
（目标卡 H100 / sm90、`-arch=sm_90a -rdc=true -lcudadevrt`、nvcc 13.2、**固定空闲 device 3**）。
一条「单元」= 一次 `tick`（对 `sink` 做一次 `atomicAdd`）。要跑 N 个单元，四种「分发缝」公平对照：

| 策略 | 谁派发 | 每单元怎么走 |
|---|---|---|
| `eager` | host | 循环 N 次 `tick<<<1,1>>>`（host 派发 kernel） |
| `hgraph` | host | 循环 N 次 `cudaGraphLaunch(1-node graph)`（host 派发 graph） |
| `srelaunch` | **device** | device-launchable 图，节点用 `cudaGraphLaunch(cudaGetCurrentGraphExec(), TailLaunch)` **自重启** N 次；host 只提交 **1 次** |
| `kinternal` | **device（常驻）** | 单 kernel 内 `for N` 循环（= P36-1 的 `kinternal` 档） |

host 循环用 CUDA event 计时，`srelaunch` 用一次 host 提交 + event（每 rep 前 `cudaMemset(cnt,0)`），
全部 `sink` 必须 `= N×reps`。另外：Section A 量两条**使用约束**；Section C 用 host-graph 里的 launcher
节点 `TailLaunch` 一个子图，测**一次 device 图启动的往返**（`%globaltimer` 打点，launcher 入口→子图入口）。

### 26.4 结果

**A. 两条使用约束（实测复现）**

```text
[Section A] 设备侧 cudaGraphLaunch 的使用约束
  A1 plain <<<>>> launcher 直接调 device cudaGraphLaunch -> operation not supported
  A2 dual_launch (host graph 节点内连发两次): 1st=no error 2nd=invalid argument
```

- **A1**：把「调用 `cudaGraphLaunch` 的 kernel」直接用 `<<<>>>` 启动 → `cudaErrorOperationNotSupported`
  （`operation not supported`）——**发起者 kernel 必须本身是 graph 节点**（放进一个 host-launched 图里才跑得通，
  本节后续、以及 A2/自重启都遵此）。这也解释了为什么「随便丢个 host kernel 进去当派发器」不行。
- **A2**：在 host-graph 的节点里对**同一个 exec** 连发两次 device launch：**第一次成功、第二次
  `cudaErrorInvalidValue`（`invalid argument`）**——**同一时刻同一 exec 只能有一个实例在飞**。换言之，一个
  launcher 节点**不能**靠「循环 launch 同一个 exec」来跑多行程；要循环只能走**自重启图**（`srelaunch`）。

**B. 每个单元的分发缝成本（ns/unit；主配置 N=20000×reps=5）**

```text
  eager          2078.5    sink=100000 (exp 100000)   [host loop ok]
  hgraph         2370.7    sink=100000 (exp 100000)   [host loop ok]
  kinternal        39.3    sink=100000 (exp 100000)
  srelaunch      4285.4    sink=100000 (exp 100000)   cnt/rep=20000
  ratios vs eager: hgraph=1.14x  kinternal=0.0189x  srelaunch=2.06x
```

敏感性配置（N=2000×reps=20）逐项复现：`eager 2105.9 / hgraph 2394.8 / kinternal 40.0 / srelaunch 4294.1`，
比值 `1.14x / 0.019x / 2.04x`，`sink` 全对。

- **`srelaunch`（设备派发图）≈ 2.06× eager（host 派发 kernel）**：设备侧自重启一次图要 **~4.29 µs**，
  比 host 直接 launch 一个 kernel（~2.08 µs）**还贵一倍**。
- **`hgraph`（host 派发 1-node 图）≈ 1.14× eager**：单节点图从 host 启动反而**略贵于**直接 launch kernel，
  与 P4-1「图只在摊薄很多节点后才赢」一致。
- **`kinternal`（常驻内核自己的循环）≈ 0.019× eager（~40 ns/单元）**：数量级最低——因为它**根本不跨
  kernel/图分发边界**。

**C. 一次 device 图启动的往返**

```text
  device-graph-launch round-trip = 4128.0 ns (feasible=50/50)
```

host-graph 的 launcher 节点写 `t0` → `TailLaunch` 子图 → 子图入口写 `t1`，`Δ = 4128 ns`，与 Section B
的 `srelaunch` 每行程 **4.29 µs** 同量级——即 **`srelaunch` 的成本几乎就是「一次 device 图启动」本身**，
与自重启循环体的工作量无关。

### 26.5 读法

1. **设备侧启动一个图并不比 host 便宜**：`srelaunch` 4.29 µs ≈ host 直接 launch 的 **2×**、≈ host 启动
   1-node 图的 **1.8×**。设备「能发图」不等于设备「发图便宜」——**分发缝的代价在硬件派发本身**，
   与发起者是 host 还是 device 无关。
2. **device graph launch 与 P36-1 的条件节点落在同一个 ~4 µs 地板**：`cwhile` 4.3–4.6 µs/行程（§25）
   vs 本节 `srelaunch` 4.29 µs/行程 vs 单次 device 图启动 4.13 µs——**「设备侧 kernel/图重分发」这种
   机制无论披什么外衣（条件节点 WHILE / 自重启图 / device graph launch），单价都在 ~4 µs**。
3. **唯一的数量级更低来自「不跨边界」**：`kinternal` ~40 ns/单元说明，只要执行体**留在同一个常驻
   kernel 内**，单元之间就没有 ~2–4 µs 的分发缝。这正是 megakernel 把「下一轮/下一个 task」做成
   **设备内 event/队列**而不是「设备再发一个 kernel/图」的根本原因。
4. **约束进一步削弱它**：发起者必须是图节点（A1）、同一 exec 只能单实例在飞（A2）——要循环就得
   **自重启图**（每行程一次设备图启动），这正是本节量到 4.29 µs 的形态。

### 26.6 与判据 / MPK / 生产案例的关系

- **判据（C1–C5，`docs/00` §3）**：四策略**均不满足 C1–C5**。`eager`/`hgraph` 是 host 提交的多次
  kernel 边界；`srelaunch` 虽**由设备发起**，但每次重启仍是**一整个图（= kernel 级）**的分发，kernel
  仍会结束、**没有 task 级设备内调度**；`kinternal` 满足 C1/C2（单次 launch + 常驻）但**没有** C3/C4/C5。
- **与 MPK 的关系**：MPK 的 in-kernel scheduler（`docs/06` §3）在**一个常驻 kernel 内**用事件计数 + 原子
  ring 把 task 派给 worker，**不产生新的 kernel/图分发缝**——量级上就是本节 `kinternal` 那一档；而
  device graph launch 是「设备侧派发」的**最粗版本**（粒度 = 整个图、发起者必须是图节点、单实例在飞），
  其单价（~4 µs）远达不到 MPK 需要的 task 级粒度。这与 P4-1 的判词一致，并把它从「资料判断」升级为
  **实测**：**设备侧图启动仍停在 kernel 级**。
- **与生产案例**：DeepGEMM Mega 家族用**单 kernel 内的原子 ring barrier**（`docs/02` §1.5）、MPK 用
  in-kernel event queue（`docs/07` §4 两套同步机制）——都不走「设备再发一个 kernel/图」，理由正是本节
  量到的 ~4 µs 分发缝 vs ~40 ns 的 in-kernel 边界。

### 26.7 局限与踩到的坑

- **合成 workload / 单 block**：`tick` 与 `kinternal` 的单元都是「单 block 一次 atomicAdd」，绝对数以
  **量级**读；`srelaunch`/`fireonce` 的 ~4 µs 是「分发缝」本身，不含真实 task 工作。
- **`hgraph` 用 1-node 图**：故它 **不**代表「多节点图的摊薄收益」（那是 P4-1/§25 的主题）；本节用它
  只为对齐「host 发一个图」的单价。
- **A2 只证「同 exec 连发第二次失败」**：未穷举 `FireAndForget`/`TailLaunch`/`AsSibling` 三种流的差异
  （主线自重启用的是 `TailLaunch`）；亦未测「两个不同 exec 从同一 kernel 并发 launch」。
- **未测**用**多 block / 有真实工作量**的图自重启，也未测 `kinternal` 里再套一层日付的 dispatch；
  `%globaltimer` 已消除跨 SM 时钟差，但 Section C 取 50 次最小值。
- **构建坑**：device 侧 runtime 必须 `-rdc=true -lcudadevrt`，否则链接期 `ptxas` 报
  `Unresolved extern function '__cudaCDP2GetLastError'`。
- **device 3**（固定空闲），主配置与敏感性两轮结果一致。

## 27. 多租户共存：整机 megakernel 的干扰 & Green Contexts SM 分区（P38-1）

> 本节把「megakernel = 单次 launch 占满整机」这一前提**翻到成本面**：当它真的占满整机时，
> 同机的**其它 kernel（另一个模型、另一个租户、一个 profiler、一次通信）还能不能跑**？
> 以及对 MPK §5「按物理 SM 数固定分配 worker/scheduler」的部署含义。源码 `src/green_ctx_sched.cu`、
> 脚本 `harness/green_ctx_sched.sh`、原始输出 `harness/green_ctx_sched.out.txt`
> （目标卡 H100 / sm90，**固定空闲 device 3**，nvcc 13.2，buf=256 MB，rounds=64；
> 同一份程序在 device 1 上复现同量级数字）。

### 27.1 背景与问题

「megakernel 把整张图融进一个常驻 kernel」有一个被前面所有小节默认接受的前提：**它独占整机**
——MPK §5 原文（快照行 1168）：

> "The allocation of workers and schedulers is **fixed at kernel-launch time and matches the GPU's
> physical SM count**, avoiding any dynamic role-switching overhead inside the kernel."

本专题 P10-1（网格按**占用率**选 `min(占用上限, N)`）、P14-1（异构 task 融合把轻 task 的占用率拉低）
都在**单一 kernel 内部**谈占用率；本节问的是**跨 kernel**：一个把占用率槽位占满的持久化内核，
对**共存的普通 kernel**意味着什么？换成部署语言：serving 集群里 Megakernel 与其它负载共享一张卡时，
谁来保证公平？分成三个可测的问题：

- **Q1（干扰）**：整机 persistent kernel 会不会把共存 kernel **饿死**？量级多大？
- **Q2（隔离）**：Hopper **Green Contexts** 的 SM 分区能否给共存负载硬隔离、且对 persistent 自身无损？
- **Q3（替代）**：流优先级能不能替代分区？分区的代价（persistent 自己少拿 SM）有多大？

### 27.2 方法

被测对象是一个「访存为主的整机 persistent kernel」（`persist_kernel`：`rounds` 轮 grid-stride 求和，
网格按占用率上限铺满），以及一个「victim」普通 kernel（`victim_kernel`：多 block grid-stride 求和，
最后一个完成的 block 写 mapped-pinned 的 `done` 标志）。三者对照：

- **无隔离**：persist 与 victim 同在默认 context 的两条流上（`sp` / `sv`）。
- **Green Contexts 分区**：`cudaDeviceGetDevResource` 取整机 SM → `cudaDevSmResourceSplit` 切成
  A/B 两组 → `cudaGreenCtxCreate` 建两个 green context（各带自己的 stream）。persist 在 A、victim 在 B。
- **流优先级**：persist 挂低优流、victim 挂高优流（`cudaStreamCreateWithPriority`，本机 `hi=-5`）。

计时：victim 的 host 墙钟（launch 前 `t0` → 轮询 `done` 返回），预热 5 次取 min 作基线；
干扰场景每次**重新起一个 persistent**、只采**第一次** victim（只有 «persistent 还在跑» 的那次才可能被饿死），
重复 3 次取 min。persist 自身的设备侧时长用 `%globaltimer` 记录。

### 27.3 原始结果

```text
=== P38-1 persistent kernel 共存干扰 / Green Contexts 分区 ===
device=NVIDIA H100 80GB HBM3 sm_90 SMs=132  buf=256MB rounds=64
occupancy: victim 8 blk/SM, persist 8 blk/SM -> 整机 cap=1056 blocks

[Section A] 整机 persistent 对同机 kernel 的干扰（无隔离，默认 context 两条流）
  A0 victim alone (baseline, min/5)         : 0.042 ms
  A1 Gp=132 （仅 1 blk/SM）[有空槽位可共驻] | victim=0.040 ms  (1.0x baseline) | persist 23.85 ms
  A1 Gp=1056（占满整机）[被饿死→等 persistent 结束] | victim=5.613 ms  (134.5x baseline) | persist 5.59 ms

[Section B] Green Contexts SM 分区隔离（A 放 persistent，B 放 victim）
  [gc] total=132 align=8 -> A=64 SM, B=68 SM
  B0 victim alone in gcB(68 SM)              : 0.041 ms (baseline)
  B1 persist@A(64 SM 占满) + victim@B(隔离)  : victim=0.031 ms (0.76x baseline) | persist 7.56 ms
  B2 persist@A + victim@A(同分区,另流,对照)  : victim=7.593 ms (184.30x baseline) | persist 7.52 ms

[Section C] 流优先级不能替代分区（默认 context，persist 低优 / victim 高优）
  priority range: low=0 high=-5
  C0 victim(高优) alone                       : 0.042 ms
  C1 persist(低优,占满) + victim(高优)        : victim=5.615 ms (133.65x baseline) | persist 5.60 ms

[Section D] 分区的代价：persistent 自身吞吐 vs 分到的 SM 数
  [gc] total=132 align=8 -> A=32 SM, B=100 SM
  D smA=32   (gcA=32 SM) grid=256   persist=12.87 ms  1335.3 GB/s
  [gc] total=132 align=8 -> A=64 SM, B=68 SM
  D smA=64   (gcA=64 SM) grid=512   persist=7.55 ms  2276.7 GB/s
  [gc] total=132 align=8 -> A=96 SM, B=36 SM
  D smA=96   (gcA=96 SM) grid=768   persist=6.09 ms  2821.3 GB/s

OK
```

### 27.4 读法

1. **干扰的开关是「剩余占用率槽位」，不是「占了几张 SM」**（A1）：Gp=132 时 persistent 铺满**全部
   132 个 SM**，但每 SM 只占 1/8 个 block 槽位 → victim **1.0× 基线**（可共驻）；Gp=1056（8 blk/SM
   占满整机）→ victim 被饿死到 **134.5× 基线**（5.613 ms ≈ persistent 自身 5.59 ms）。这与 P10-1
   「并行度在**占用率上限**而非 #SM 处饱和」、P14-1「占用率被拉低」是同一变量的跨 kernel 形态。
2. **Green Contexts 的 SM 分区给硬隔离，且互不拖慢**（B1）：persist 占满 gcA(64 SM)、victim 在
   gcB(68 SM) → victim **0.76× 基线**（≈ 无干扰），而 persist 自身 **7.56 ms 与 D 段 smA=64 单独跑
   的 7.55 ms 一致** → **隔离是双向的**（victim 不拖 persistent、persistent 不拖 victim）。
3. **隔离来自分区本身、不是「换个 context」**（B2 对照）：把 victim 放进 **A 的同分区另一条流** →
   又被饿死到 **184.3× 基线**（等满 persistent 的 7.52 ms）。同一对 green context，只是 victim 落 A 还是落 B。
4. **流优先级不能替代分区**（C1）：persist 低优、victim **高优**，victim 仍被饿死到 **133.65× 基线**
   —— 优先级只影响**就绪 block 的派发顺序**，**不会抢占已在 SM 上运行的 block**；要把整机 persistent
   挤开，只能改**它拿多少 SM**（分区），不能改**谁优先**。
5. **分区的代价是 linear scaling**（D）：persist 吞吐随分到的 SM 数近似线性——32 SM→1335 GB/s、
   64→2277、96→2821 GB/s（96 SM 已接近整机 HBM 上限，之后饱和）。换言之，**给共存租户留 SMs =
   按比例削减 megakernel 的可达带宽**，是必须显式付的账。

### 27.5 与判据 / MPK / 生产案例的关系

- **判据（C1–C5，`docs/00` §3）**：本节三者都是**单个 kernel**，判据落在「kernel 级」：`persist_kernel`
  满足 C1/C2（单次 launch + 常驻），`victim_kernel` 是普通 kernel-per-operator；**Green Contexts 是
  C4 的更粗一层实现**（设备内/资源层协调，把「谁用哪些 SM」在 launch 前定死，本专题 P29-1 的
  cluster 分区是它在**片内**的同目标形态）。**均不涉及 C3/C5**——本节谈的**不是** megakernel 内部调度，
  而是它**占满整机**这一外部性。
- **与 MPK 的关系**：MPK §5（行 1168）明确「worker/scheduler 分配**固定在 launch 时、匹配物理 SM 数**」
  ——这正是 A1 的 Gp=cap/1056 形态：megakernel 在**语义上**就假定独占整机。因此当**同卡还有别的负载**
  时，MPK 只能靠**外来约束**（green context 给它一个 `smCount < 132` 的资源、或 MPS/独占调度）来共存，
  而 kernel 内部无法感知「少给了我几个 SM」——这与 MPK §7「per-task codegen 是唯一硬件相关部分、
  寄存器固定为各 task 最大值」一样，是**资源假设**层面的边界。
- **与生产案例**：DeepGEMM mega 家族（`docs/02` §1）同样按整机 CTA/warpgroup 铺静态角色；真实 serving
  往往一卡多模型。本节给出**量化结论**：**「整机 megakernel + 共存负载」不分区就会把对方饿死
  ~130–180×，Green Contexts 分区可恢复到 ≈ 基线（代价是 megakernel 自身按 SM 数线性降带宽）**。
  这与 P29-1 的结论互补：cluster 是**片内**局部性分区（≤16 block、同一 GPC），green context 是
  **整机**租户分区（A/B 两片 SM），二者都是「按需局部化」的不同粒度。

### 27.6 局限与踩到的坑

- **victim 是合成访存 kernel**（多 block 求和），数字以**量级**读；A1 两种 Gp 的「共驻 vs 饿死」结论
  对具体 workload 稳健，但绝对倍数随 victim 工作量而变。
- **依赖 mapped-pinned 的 `done` + host 轮询**计时：host 墙钟含 host 提交延迟（~µs 级），基线 0.042 ms
  已高于它；被饿死时信号（>5 ms）远大于噪声。
- **未测 MPS**（Multi-Process Service）作为另一条隔离路线，也未测 **green context 内的 cluster/`smCoscheduledAlignment` 交互**；
  Section D 只扫 32/64/96 SM（`smA=128` 因 132 个 SM **不是 `smCoscheduledAlignment`=8 的整数倍**、
  余 4 个 SM 小于最小分组而 split 失败——**踩坑：H100 的 132 SM 与 8 对齐不整除，第二组必须用
  `cudaDevSmResourceGroupBackfill` 才能拿到余数**）。
- **Green Contexts 的 runtime API**（`cudaDevSmResourceSplit`/`cudaGreenCtxCreate`/`cudaExecutionCtxStreamCreate`）
  在本容器 CUDA 13.2 可用；更早 CUDA 只有 driver API 版本。官方语义见 NVIDIA CUDA C Programming Guide
  的 “Green Contexts” 一节（`https://docs.nvidia.com/cuda/cuda-c-programming-guide/`）。
- **device 3**（固定空闲），device 1 复现同量级（A1 5.612/5.613 ms、B1 0.031/0.031、D 同）。

## 28. 多租户共存隔离的第二条路线：MPS（Multi-Process Service）（P39-1）

> §27 用 Hopper **Green Contexts** 把整机 SM「按张数」硬分区，给共存负载确定性隔离，
> 并明确留了「未测 **MPS**（Multi-Process Service）作为另一条隔离路线」（§27.6）。本节把它补上。
> 关键区别是**进程边界**：§27 是同一进程内的两条流/green context；本节是**两个独立进程**——
> 没有 MPS 时，不同进程的 kernel 在 GPU 上**不能真正并发**（由驱动在 kernel 间时分/抢占）；
> 有了 MPS 才允许跨进程真并发，并可用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` 给每个 client 设
> 「线程容量上限」。源码 `src/mps_isolation.cu`、脚本 `harness/mps_isolation.sh`、原始输出
> `harness/mps_isolation.out.txt`（目标卡 H100 / sm90，**固定空闲 device 3**，nvcc 13.2，
> buf=256 MB，victim=20 趟求和，persist=40000 rounds；同一曲线在重复运行间稳定）。

### 28.1 背景与问题

「megakernel 独占整机」的前提（MPK §5 快照行 1168）在真实 serving 里很少成立：一卡常有多模型/
多租户。§27 给出**进程内**的确定性解（Green Contexts SM 分区，victim 0.76× 基线、persist 无损）。
但生产上更常见的是**多进程**共享一张卡（每个模型一个进程、每个训练任务一个进程）。这时有两条路：

- **不用 MPS**：把整机交给一个进程，另一个进程的 kernel 会被驱动**时分/抢占**——语义上是「轮流独
  占」，不是并发。
- **用 MPS**：多个进程连到同一个 MPS server，共享一个 CUDA context，kernel 真正并发；
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P` 限制某 client 的**并发线程容量 / 可见 SM 数**（软限）。

本节量三个问题：

- **Q1（无 MPS）**：跨进程时，整机 persistent kernel 会不会把另一个进程的 kernel 饿死？
- **Q2（MPS 真并发）**：MPS 默认（两 client 各 100%）下 victim 受多少干扰？
- **Q3（软分区）**：`ACTIVE_THREAD_PERCENTAGE` 能不能当「软 SM 分区」用？它的**代价曲线**长什么样？

### 28.2 方法

被测体是**两个独立进程**（`src/mps_isolation.cu` 一个二进制、两种 mode）：

- `persist`：整机访存为主的持久化 kernel（`rounds` 轮 grid-stride 只读求和），网格按
  **当前进程可见的** SM 数 × 每 SM 占用率上限（8 blk/SM）铺满，结束用 CUDA event 记时长/带宽。
- `victim`：普通 kernel（`passes` 趟 grid-stride 求和），用 CUDA event 计时（`record → launch →
  record → 有界等待`），**计时段包含「被饿死而排队」的时间**（5 s 超时兜底）；预热 5 次、正式 7 次，
  报 min/median。

对照组：

- **无 MPS**：直接两个进程、各自 `CUDA_VISIBLE_DEVICES=3`。
- **有 MPS**：`nvidia-cuda-mps-control -d` 起 server（server 用 `CUDA_VISIBLE_DEVICES=3`，**只占
  device 3**，pipe 用独立临时目录）；client **不设** `CUDA_VISIBLE_DEVICES`，经 pipe 连到 server
  暴露的 device。persist 进程按 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P` 设 cap，victim 不设（默认 100%）。
- 干扰场景：persist 后台跑（先 sleep 2 s 保证已在内核里）→ 跑 victim → 等 persist 自然结束读其带宽。

### 28.3 原始结果

```text
=== P39-1 跨进程共存隔离：MPS vs 无 MPS（目标卡 H100 / sm90）===
device=3  buf=256MB  victim_passes=20  persist_rounds=40000

[Section A] 无 MPS：不同进程的 kernel 不能真正共驻（由驱动时分/抢占）
  A0 victim alone (no MPS)      : VICTIM dev=NVIDIA H100 80GB HBM3 grid=1056 passes=20 min=1.779 median=1.781 ms
  A1 persist alone (no MPS)     : PERSIST_DONE ms=3520.206 gbps=3050.2
  A2 victim during persist      : VICTIM dev=NVIDIA H100 80GB HBM3 grid=1056 passes=20 min=1.791 median=1.792 ms
  A2 persist (concurrent victim): PERSIST_DONE ms=3570.907 gbps=3006.9

[Section B] MPS 默认：两 client 各 100%（真并发，共享占用/带宽）
  B0 victim alone (in MPS)       : VICTIM dev=NVIDIA H100 80GB HBM3 grid=1056 passes=20 min=1.780 median=1.783 ms
  B1 victim during persist       : VICTIM dev=NVIDIA H100 80GB HBM3 grid=1056 passes=20 min=2.565 median=2.780 ms
  B1 persist (concurrent victim) : PERSIST_DONE ms=3539.016 gbps=3034.0

[Section C] MPS 给 persistent 设 ACTIVE_THREAD_PERCENTAGE=P（线程容量上限）
  C P=100 victim during persist : VICTIM ... min=2.488 median=2.769 ms
  C P=100 persist rate (≈solo)  : PERSIST_DONE ms=3537.470 gbps=3035.3
  C P=90  victim during persist : VICTIM ... min=17.641 median=17.652 ms
  C P=90  persist rate (≈solo)  : PERSIST_DONE ms=3638.048 gbps=2951.4
  C P=75  victim during persist : VICTIM ... min=7.088 median=7.093 ms
  C P=75  persist rate (≈solo)  : PERSIST_DONE ms=3853.346 gbps=2786.5
  C P=60  victim during persist : VICTIM ... min=4.987 median=4.994 ms
  C P=60  persist rate (≈solo)  : PERSIST_DONE ms=4270.507 gbps=2514.3
  C P=50  victim during persist : VICTIM ... min=3.576 median=3.584 ms
  C P=50  persist rate (≈solo)  : PERSIST_DONE ms=4738.108 gbps=2266.2
  C P=40  victim during persist : VICTIM ... min=3.240 median=3.244 ms
  C P=40  persist rate (≈solo)  : PERSIST_DONE ms=5563.588 gbps=1929.9
  C P=33  victim during persist : VICTIM ... min=3.050 median=3.054 ms
  C P=33  persist rate (≈solo)  : PERSIST_DONE ms=6586.495 gbps=1630.2
  C P=25  victim during persist : VICTIM ... min=2.900 median=2.906 ms
  C P=25  persist rate (≈solo)  : PERSIST_DONE ms=8106.114 gbps=1324.6
  C P=10  victim during persist : VICTIM ... min=2.582 median=2.629 ms
  C P=10  persist rate (≈solo)  : PERSIST_DONE ms=21072.965 gbps=509.5
```

persist 进程打印的「可见 SM 数」随 cap 变化（`PERSIST dev=... SMs=... grid=...`）：
`P=100→132 SM/grid 1056`、`90→118/944`、`75→98/784`、`60→78/624`、`50→66/528`、`40→52/416`、
`33→42/336`、`25→32/256`、`10→13/104`。即 **`ACTIVE_THREAD_PERCENTAGE` 直接缩放了该进程可见的
`multiProcessorCount`**。

### 28.4 读法

1. **无 MPS 时跨进程 kernel 不共驻，victim 不被饿死**（A）：victim 1.792 ms ≈ 基线 1.781 ms（1.006×），
   persist 只慢 1.4%（3520→3571 ms）。设备在 kernel 间**时分/抢占**——victim 每轮拿到的是**整机
   时间片**，所以延迟不受损；代价转嫁到 persist（上下文切换 + 被抢占）。这与 §27 进程内两条流的
   134× 饿死**完全相反**：**同一张卡，「同进程并发」比「跨进程无 MPS」对邻居更糟**。
2. **MPS 默认（各 100%）是真并发**（B）：victim 2.780 ms = **1.55× 基线**——两个访存 kernel 真共驻、
   共享占用率与 HBM 带宽；persist 带宽几乎不变（3034 GB/s）。这才是「共享一张卡」的常规语义。
3. **`ACTIVE_THREAD_PERCENTAGE` 是硬「线程容量」上限**：它把 client 可见 SM 数按 P 缩放
   （见上表），persist 的网格随之下调。
4. **反直觉：干扰峰值出现在「cap 略低于 100」**（C）。victim 相对基线的 slowdown 对 **P≥50** 精确
   服从 `≈ 1/(1−P)`：

   | P | victim median (ms) | slowdown | `1/(1−P)` |
   |---|---|---|---|
   | 100 | 2.769 | 1.55× | 时间片（不适用） |
   | 90 | **17.652** | **9.90×** | 10.0 |
   | 75 | 7.093 | 3.98× | 4.0 |
   | 60 | 4.994 | 2.80× | 2.5 |
   | 50 | 3.584 | 2.01× | 2.0 |
   | 40 | 3.244 | 1.82× | 1.67 |
   | 33 | 3.054 | 1.71× | 1.49 |
   | 25 | 2.906 | 1.63× | 1.33 |
   | 10 | 2.629 | 1.47× | 1.11 |

   即：**给 persistent 留 10% 线程容量（P=90），就把 victim 饿死 ~10×，比完全不设 cap（1.55×）还糟
   近 6 倍**；cap 越低 victim 越恢复。机理：P<100 时 MPS 做**空间化**的线程容量切分，victim 只剩
   `(1−P)` 的容量 → 访存受限 kernel 的带宽/延迟按 `1/(1−P)` 退化；P=100 时两侧都满额，MPS 退回
   **时间片**，反而只付 1.55×。
5. **persist 自身的带宽随 cap 次线性**：3035→2951→2786→2514→2266→1930→1630→1325→509 GB/s
   （P=100→10）。即「分区账」：persistent 少拿线程容量，**带宽降幅小于线程降幅**（访存早已饱和的
   区间内，少量 SM 也能压出大部分带宽）；全在 P=10（13 SM）才崩塌。
6. **三条隔离路线对比**：Green Contexts（§27，同进程 SM 张数硬分区）victim **0.76× 基线**、persist
   无损、双向且**确定性**；MPS（跨进程，线程容量软限）**最坏 9.9×** 且非确定，还依赖 cap 取值；
   无 MPS（跨进程时分）victim ≈ 基线但**不是并发**。**结论：要确定性公平隔离用 Green Contexts/
   独占；MPS 只保证「能并发」，不保证公平**。

### 28.5 与判据 / MPK / 生产案例的关系

- **判据（C1–C5）**：`persist` 是单 kernel、满足 C1/C2；**MPS 是进程/驱动层的资源约束，比 Green
  Contexts 更外一层**（它不是设备内协调、也不改 kernel），是「C4 的更粗粒度容器」。三者都不涉 C3/C5
  ——本节谈的是 megakernel **占满整机**这一外部性，不是它内部调度。
- **与 MPK 的关系**：MPK §5（行 1168）在**语义上**假定独占整机（worker/scheduler 固定匹配物理 SM），
  内核无法感知「少给了几个 SM」。MPS 提供了「整机一个 context、多个 client」这一部署形态，但 §28.4.4
  说明**它不会自动保护共存负载，最坏情况比不隔离还差**——给整机 megakernel 做 MPS 隔离时，cap 应
  设得**明显低**（P≤25）而不是「尽量高」，否则会给邻居 ~10× 惩罚。
- **与生产案例**：DeepGEMM mega 家族（`docs/02` §1）按整机 CTA/warpgroup 铺静态角色；真实 serving
  一卡多模型。本节给量化结论：**「整机 megakernel + 共存进程」若无隔离，跨进程靠时分（邻延迟≈基线、
  但非并发）；用 MPS 反而可能把邻居饿死到 ~10×（P≈90）**。与 §27 合看：**进程内共享**用 Green
  Contexts 分区、**跨进程共享**优先独占/Green Contexts，MPS 只在「确实需要跨进程并发且已显式留足
  (1−P) 余量」时用。
- **三条粒度**：P29-1 的 cluster 是**片内**分区（≤16 block）、§27 的 green context 是**整机 SM 张数**
  分区、§28 的 MPS 是**跨进程线程容量**软限——同一动机（按需局部化）的三个粒度。

### 28.6 局限与踩到的坑

- **victim 是合成访存 kernel**（grid-stride 求和），倍数以**量级**读；`1/(1−P)` 是 P≥50 的经验律，
  P<50 有 ~1.4–1.6× 的固定地板（MPS 协调/带宽争用），未做 ncu 归因。
- **persist 网格随 cap 自动缩放**（可见 `multiProcessorCount` 变了），不是「固定网格下只改线程容量」；
  这既是 MPS 的真实行为，也意味着观测到的是「可见 SM + 线程容量」的合成效应。
- 跨进程重叠靠 `sleep 2 s` 保证（persist 跑 3.5–21 s，远超 2 s）；victim 有 5 s 超时兜底。
- **MPS 关键坑**：client 若设 `CUDA_VISIBLE_DEVICES` 会连不上 MPS server（报
  `no CUDA-capable device is detected`）。正确做法是 **server** 用 `CUDA_VISIBLE_DEVICES=3` 限定，
  **client 不设**（经 pipe 连到 server 暴露的 device 0）。MPS server 用独立 pipe/log 临时目录，
  实验后 `echo quit | nvidia-cuda-mps-control` 清理；只影响 device 3。
- 未测：**cap 加在 victim 侧**、多 client（>2）、MPS 的 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`、
  与 Green Contexts 叠加；也未测 MPS 内 cluster/`smCoscheduledAlignment` 交互（§27.6 同一条尾巴）。
- MPS 官方语义见 NVIDIA 文档 “Multi-Process Service”（`https://docs.nvidia.com/deploy/mps/index.html`）。

## 29. L2 常驻：损失三的第三档中间量落点（寄存器 → L2 → HBM）（P40-1）

> §17（P24-1）用 ncu 直接量了损失三（中间张量的 HBM 往返）的字节，其 §17.7 明留一条假设：
> 「若张量能整块驻留 L2（≤50 MB），eager 的中间量往返可能命中 L2 而不落 DRAM，损失三会小于本节
> 数字——megakernel 的损失三在『工作集超过 L2』时才充分显现。」本节把这条假设测出来，并回答：
> 中间量除了「寄存器」（P5-2 融合）与「HBM」（P24-1 损失三）之外，能不能落在 **L2** 这一档？
> 源码 `src/l2_residency.cu`、脚本 `harness/l2_residency.sh`、原始输出 `harness/l2_residency.out.txt`
> （目标卡 H100 / sm90，**固定空闲 device 3**，nvcc 13.2，iters=30）。

### 29.1 背景与问题

中间量在内存层级里的落点，本专题已测到两档：

- **寄存器**：把算子融合进同一个 kernel，中间量根本不落内存（P5-2 §2、P24-1 §17 的 `fused`，8 B/elem 恒定）；
- **HBM**：kernel-per-operator 每个算子把中间量写回 HBM、下个算子再读回（P24-1 §17，`eager` 字节 ≈ `Q`×全张量往返）。

中间还有一档没测：**L2**。H100 有 **50 MB** L2——足够装下小模型一个 batch 的若干中间张量。如果中间量
在一次写、一次读之间能留在 L2，那么「损失三」的 HBM 往返本可以省掉。§17.7 的假设正是这个。但
L2 是**共享、按需分配、可被任何并发流量逐出**的：真实 serving（尤其 megakernel 让整图很多张量同时
流经 L2）里，写读之间总会流过别的张量。于是关键问题不是「尺寸够不够装」，而是：

- **Q1**：中间量的 L2 常驻，是被「尺寸」决定，还是被「写读之间流过什么」决定？
- **Q2**：`cudaAccessPolicyWindow` 的 **persisting**（常驻）机制能不能把它保护住？代价是什么？

### 29.2 方法

构造一条 kernel-per-operator 序列（三算子，中间量 `y` 尺寸 `S` 可变）：

```
K1:  读 x(256 MB)  → 写 y(S)
polluter: 流式读+写 512 MB      ← 模拟「写读之间流过的并发流量」，把 L2 冲干净
K2:  读 y(S)       → 写 z(256 MB)
```

`K2` 从哪读 `y` 就是「中间量落在哪一档」的判据。四个模式：

| 模式 | polluter | 中间量 `y` 的 L2 处理 | 语义 |
|---|---|---|---|
| `base` | 不插 | 无 | 靠「最近性」自然留在 L2（写读紧邻） |
| `pollute` | 插 | 无 | `y` 被逐出 → 落 HBM（= 损失三） |
| `persist` | 插 | `accessPolicyWindow` persisting，carveout=`min(S, maxPersist)` | 按中间量大小配常驻区（合理用法） |
| `pfull` | 插 | 同上，但 carveout=`maxPersist`（31 MB 全划走） | 测「过度预留常驻区」的代价 |

L2 常驻用标准 API：`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, bytes)` 划出 persisting set-aside，
`cudaStreamSetAttribute(st, cudaStreamAttributeAccessPolicyWindow, &w)` 给 `y` 挂窗口
（`hitProp=cudaAccessPropertyPersisting`、`missProp=cudaAccessPropertyStreaming`、`hitRatio=1.0`），
模式切换时用 `cudaCtxResetPersistingL2Cache()` 清干净。H100 上
`cudaDevAttrMaxPersistingL2CacheSize` = **31 MB**、`cudaDevAttrMaxAccessPolicyWindowSize` = 128 MB。

- **计时**：K2-only（每 iter 先跑 pre、再用一对 CUDA event 只夹 K2），iters=30；基准是 `pollute`（= 中间量
  必落 HBM），`base/pollute`、`persist/pollute` 越低说明 `y` 越命中 L2。
- **字节/命中率**：ncu `-k regex:l2_k2` 取 `dram__bytes_read.sum`（K2 读 `y` 的真实 DRAM 字节，标称 `S`）
  与 `lts__t_sector_hit_rate.pct`（L2 读命中率）。

> **关键踩坑（首版就栽在这）**：ncu 默认 `--cache-control=all` 会在**每次 replay 前 flush L2**，三个模式
> 的 DRAM 读字节在第一版里**完全一样（都 = S）**，看不出任何常驻差异。必须显式
> `--replay-mode application --cache-control none`，让每个 pass 重放整个 app、保留自然的 L2 演化，才看得到。

### 29.3 原始结果

**计时表**（K2-only，ms；比值相对 `pollute`）：

```text
      | K2ms pollute | base/pollute | persist/pollute | pfull/pollute
------+--------------+--------------+-----------------+---------------
   4 MB |       0.0090 |        0.748 |           0.900 |         1.502
   8 MB |       0.0119 |        0.677 |           0.829 |         1.532
  16 MB |       0.0181 |        0.651 |           0.864 |         1.541
  32 MB |       0.0301 |        0.828 |           1.468 |         1.469
  48 MB |       0.0421 |        0.947 |           2.171 |         2.154
  64 MB |       0.0539 |        0.970 |           2.200 |         2.205
```

**ncu：K2 读 `y` 的 DRAM 字节 / L2 命中率**（标称 `S` 即「`y` 全从 HBM 读」的上限）：

```text
   S  | base DRAM read (hit%) | pollute DRAM read (hit%) | persist DRAM read (hit%) | pfull DRAM read (hit%)
------+-----------------------+--------------------------+--------------------------+-----------------------
  4MB | 6.66 KB    (95.9%)    | 4.20 MB    (55.7%)       | 3.20 MB    (62.8%)       | 3.08 MB    (64.4%)
  8MB | 6.66 KB    (96.4%)    | 8.40 MB    (53.6%)       | 5.99 MB    (62.5%)       | 6.77 MB    (60.9%)
 16MB | 1.68 MB    (89.9%)    | 16.79 MB   (52.0%)       | 11.88 MB   (61.7%)       | 11.85 MB   (62.4%)
 32MB | 24.82 MB   (60.6%)    | 33.56 MB   (51.2%)       | 28.60 MB   (57.3%)       | 28.70 MB   (57.4%)
 48MB | 48.06 MB   (52.7%)    | 50.34 MB   (51.1%)       | 50.06 MB   (52.9%)       | 50.07 MB   (53.0%)
 64MB | 66.61 MB   (51.4%)    | 67.12 MB   (51.0%)       | 67.10 MB   (52.4%)       | 67.10 MB   (52.7%)
```

正确性：`base` 与 `persist` 的 `z` 逐位一致（S=16MB `max|Δ|=0.000e+00`）——四种模式**只改中间量的
内存落点，不动数值**（persisting 是纯性能提示）。

### 29.4 读法

1. **中间量在不在 L2，由「写读之间流过什么」决定，而不是尺寸单独决定**（Q1）。`base` 在 `S≤16 MB`
   时 K2 几乎全命中 L2（DRAM 读 6.7 KB–1.7 MB、命中 90–96%），只要 K1 后紧接 K2（中间没有别的流量）；
   一插 512 MB 的 polluter（`pollute`），K2 的 DRAM 读立刻涨到**恰好 = S**、命中掉到 52%——`y` 被逐出。
   即「写读紧邻」本身就是一种隐式的 L2 常驻，一旦有并发流量就失效。
2. **`persisting` 窗口能保护，但只部分**（Q2）：`persist`（carveout 按 `S` 配）把 K2 的 DRAM 读相对
   `pollute` 降 **~23%（S=4）到 ~30%（S=8/16）**，命中率抬到 62%——**方向对、量级有限**。原因：能常驻
   的只有 31 MB carveout（H100 上限 = 50 MB L2 的 62%），且窗口内所有行都要挤这块小产区。
3. **超出 carveout 后 `persisting` 反噬**：`S≥32 MB` 时 `persist` 相对 `pollute` **更慢**（32 MB→1.47×、
   48 MB→**2.17×**、64 MB→**2.20×**）——中间量装不进常驻区却在里面反复换入换出，比直接走 HBM 还差。
   **损失三在 `S` 超过 persisting carveout 后无可回避**（对上 §17.7 的假设，且给出精确边界）。
4. **过度预留 carveout 本身有代价**（`pfull`）：即便 `S` 很小（4–16 MB）、匹配用法（`persist`）已经
   在赚，把 31 MB carveout 全划走的 `pfull` 却 **稳定慢 ~1.5×**——因为它压缩了留给普通（normal）访问
   的 L2，`x`/`z`/polluter 的流式流量吃不到 L2。**persisting 是拿普通 L2 换常驻 L2 的零和交易**，
   该按中间量大小配、不该拉满。
5. **`base` 的「免费 L2」有硬边界**：`S=32 MB` 起 `base` 也退化（K2 从 DRAM 读 24.8 / 32 MB），
   `S≥48 MB` 完全等价于 `pollute`（≈ 50–67 MB 全从 HBM）。**只有当中间量 ≤ ~L2/3 且写读之间无干扰时，
   「留在 L2」才近乎免费。**

### 29.5 与判据 / MPK / 生产案例的关系

- **它补的是「中间量的内存落点」这一维**，与损失三（P24-1 §17）互补：中间量共四档 =
  **寄存器**（P5-2 融合，`fused` 8 B/elem）→ **SMEM/DSM**（P29-1 §18，片上显式管理）→
  **L2**（本节，硬件自动/可提示，≤31 MB 且与普通流量争）→ **HBM**（P24-1，损失三）。
- **对 megakernel 的含义**：megakernel / MPK 的价值恰恰是让**整图很多张量同时流经 L2**
  （`docs/09` 的多算子图、MPK §5 的跨 task 流水），也就是**「polluter 恒存在」**的场景——因此「指望中间量
  自然留在 L2」在整图融合里是**最不可依赖**的一档；`persisting` 有硬上限（31 MB = L2 的 62%）且与
  normal 流量零和。这从**内存层级**角度解释了 MPK §5.3（快照行 1312–1325）为何用 **paged shared
  memory**（片上、可显式管理、不与他人争）而不是依赖 L2，也解释了为什么最稳的仍是**融合进寄存器**
  （P5-2/P24-1）与**锁进 SMEM**（P29-1）。
- **判据（C1–C5）**：四模式都是 kernel-per-operator（K1、K2 两次 launch），本节谈的是**中间量的落点**，
  不改 C 判据；`fused`（寄存器档）才满足 C1–C3。判据映射同 §17。
- **与生产案例**：DeepGEMM mega 家族把中间量锁在**寄存器/SMEM**并按 `Kstages` 流水（`docs/07` §1.2），
  而非指望 L2；MPK 的 intermediate 分页也是显式 SMEM 而非 L2——本节给出量化理由：**在整机融合里，
  L2 常驻要么免费但脆弱（无干扰才成立）、要么受 31 MB 硬上限且零和**。

### 29.6 局限与踩到的坑

- **合成负载**：K1/K2 是逐元素 `float4`，polluter 是 512 MB 流式读写；倍数以**量级/方向**读。
- **ncu 口径**：必须 `--replay-mode application --cache-control none`（默认 `--cache-control all` 会 flush L2，
  测不到常驻差异，见 §29.2）；每次采样仍受 replay 影响，命中率绝对值为参考、**跨模式相对关系**是结论。
- **只测单窗口、单 stream、单一 persisting 段**：未测多窗口（`cudaStreamSetAttribute` 每流一个窗口）、
  多 stream 各自窗口、`hitRatio<1` 的概率化分区、以及 `cudaMemAdvise`/`cudaMemPrefetchAsync` 等其它
  张量放置手段；`S` 只扫 6 点。
- **H100 参数**：`maxPersistingL2CacheSize`=31 MB、L2=50 MB；不同代 GPU（A100 40 MB L2、B200 更大）
  的边界会移动，结论的**形状**（免费 L2 脆弱、persisting 部分且零和、超 carveout 反噬）应保持。
- **`pfull` 与 `persist` 在大 `S` 时曲线重合**（都超 carveout，保护失效），属预期。
- device 3（固定空闲）；计时绝对值随驱动/频率略变，比值稳定。

## 30. L2 persisting 的精细化调参 + 多窗口/多流共存（P41-1）

> §29（P40-1）把 L2 常驻测成「损失三的第三档中间量落点」，结论里有两条尾巴：**persisting 是「普通 L2 ↔
> 常驻 L2」的零和交易**（`pfull` 拉满 31 MB carveout 即便中间量很小也稳定慢 ~1.5×），且 §29.6 明写
> 「只测单窗口、单 stream、单一 persisting 段，**未测多窗口、多 stream 各自窗口、`hitRatio<1` 的概率化
> 分区**」。本节把这条补上——它正好对应 megakernel 的真实处境：整图很多中间量**同时**活着、分属不同
> task/流，要争同一个 carveout。源码 `src/l2_persist_tuning.cu`、脚本 `harness/l2_persist_tuning.sh`、
> 原始输出 `harness/l2_persist_tuning.out.txt`（目标卡 H100 / sm90，**固定空闲 device 3**，nvcc 13.2，
> `iters=30`，X=Z=256 MB 流式、polluter=512 MB、中间量 `y` 每段 8–16 MB，L2=50 MB /
> maxPersistingL2=31 MB / maxWindow=128 MB）。

### 30.1 背景与问题

CUDA persisting 的两个旋钮（`driver_types.h:1390-1415`、`cuda_runtime_api.h:426`）：

- **carveout**：`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, C)` 在 50 MB L2 里划出一块
  device 级 **set-aside**（H100 上限 **31 MB**）给 persisting 行专用；
- **access policy window**：`cudaStreamSetAttribute(..., accessPolicyWindow)` 给**一条 stream 挂一个**
  连续窗口 `[base, base+num_bytes)`，窗口内按 `hitRatio=f` 的概率把访问标 `hitProp=Persisting`、其余标
  `missProp`。头文件注释（`driver_types.h:1401-1408`）明确：窗口被切成多段，**`sum(hit segments)/window
  ≈ hitRatio`**——所以 **`hitRatio` 就是「概率化分区」旋钮**：只有 `f×访问量` 真正去挤 carveout。
- **一条 stream 只能挂一个窗口** → 「多窗口」只能落在**多条 stream**上。

§29 只用了 `f=1.0`、单窗口、`carveout=min(S,31)`。三个新问题：

- **Q1**：固定 carveout，把 `hitRatio` 从 1.0 降到 `f`，保护量是否≈ `f×S`？时间上能否更优？
- **Q2**：`pfull`（carveout 拉满 31 MB）的 ~1.5× 代价，能否用「**更小 carveout + `hitRatio<1`**」收回？
- **Q3**：两个并发中间量 `y1`/`y2` 各挂窗口、共享一个 carveout，保护能否**叠加**？carveout 小于总需求
  （megakernel 的常态）时会怎样？

### 30.2 方法

沿用 §29 的三算子序列 `K1(读 x→写 y) → polluter(512 MB 冲刷) → K2(读 y→写 z)`，只夹 K2 计时；ncu 用
`--replay-mode application --cache-control none`（§29.2 的坑）取 K2 的 `dram__bytes_read.sum` +
`lts__t_sector_hit_rate.pct`。新增两个模式：

- **`one win <S> <f> <carve>`**：单窗口，扫 `f` 与 `carveout`；
- **`one2 <S> <fA> <carve> [fB]`**：`y1`（stream stA）、`y2`（stream stB）各挂窗口（`<0` 表示不挂），
  序列 `K1a,K1b → polluter → K2a,K2b`（K2a/K2b 串行、各自输出到独立 `z`/`z2`，排除写缓冲共享的干扰）。

### 30.3 原始结果

**§A 单窗口 hitRatio 扫描**（S=16 MB, carveout=16 MB，对 `pollute` 基准的比）：

```text
   hitRatio |  K2 ms  | vs pollute   ||  K2 DRAM 读 y  |  L2 命中率
   ---------+---------+--------------++----------------+-----------
       0.00 |  0.0223 |   1.221      ||  16.65 MB      |  52.6%
       0.25 |  0.0209 |   1.149      ||  15.77 MB      |  54.3%
       0.50 |  0.0194 |   1.065      ||  14.53 MB      |  56.6%
       0.75 |  0.0177 |   0.969      ||  12.80 MB      |  60.0%
       1.00 |  0.0158 |   0.869      ||  11.58 MB      |  62.1%
   (基准 pollute：17.9 µs / 16.78 MB / 51.8%；base 无 polluter：11.9 µs / 0.33 MB / 93.3%)
```

**§B carveout × hitRatio 联合**（S=16 MB，对 `pollute` 的比；`f=0.00` 列 = 纯 carveout 代价）：

```text
   carveout |   f=0.00  |   f=0.25  |   f=0.50  |   f=0.75  |   f=1.00
   ---------+-----------+-----------+-----------+-----------+----------
       4 MB |    0.996  |    0.959  |    0.970  |    0.993  |    1.008
       8 MB |    0.981  |    0.931  |    0.918  |    0.947  |    1.001
      16 MB |    1.204  |    1.132  |    1.066  |    0.962  |    0.860
      31 MB |    1.921  |    1.877  |    1.782  |    1.625  |    1.504
```

单窗口 carveout 边界（S=8 MB, f=1.0，ncu）：carveout 8→**6.05 MB**、16→6.42、31→6.96；无窗口 8.40。
（S=16 MB, f=1.0：carveout 16→11.29、31→11.75。）

**§C 多窗口/多 stream 共存**（`y1`=`y2`=8 MB，总需求 16 MB；K2a/K2b 串行、独立输出；和相对无窗口）：

```text
   场景                        | K2a ms | K2b ms | vs 无窗口(和)
   ----------------------------+--------+--------+--------------
   no window                  | 0.0137 | 0.0125 |      1.000
   carve=16 y1 only f=1       | 0.0110 | 0.0146 |      0.979
   carve=16 both f=1.0        | 0.0113 | 0.0099 |      0.814
   carve=16 both f=0.5        | 0.0131 | 0.0123 |      0.972
   carve=8  both f=1.0        | 0.0148 | 0.0120 |      1.026
   carve=8  both f=0.5        | 0.0118 | 0.0112 |      0.879
   carve=31 both f=1.0        | 0.0186 | 0.0165 |      1.343
```

ncu 双窗口读数（carveout=16 MB）：仅 stA 窗口 → K2a 6.77 / K2b 8.40；仅 stB 窗口 → 8.40 / 8.29；两个都挂
→ 7.42 / 8.35；carveout=8 两个都挂 → 8.28 / 8.32；诊断「stB 单独使用」→ 6.30。

正确性：`z` 在有/无窗口下**逐位一致**（S=16 MB `max|Δ|=0.000e+00`）——persisting 是纯性能提示。

### 30.4 读法

1. **零和代价的开关是 carveout 大小，不是 hitRatio**（Q2）。§B 的 `f=0.00` 列（挂了窗口但全部按
   `Streaming` 走、等于「只划 carveout 不保护」）随 carveout 单调恶化：4 MB→0.996、8 MB→0.981、
   **16 MB→1.204、31 MB→1.921**。即 `pfull` 的 ~1.5× 惩罚**主要**来自 carveout 抢走普通 L2，
   **无法用 `hitRatio` 收回**（31 MB 行从 f=0 的 1.921 只降到 f=1 的 1.504，仍 ≫1）。
2. **正确姿势：carveout ≈ 中间量实际大小，且配得上时 `f` 取高**（Q1/Q2）。§A/§B 的最优都在
   **carveout=16、f=1.0 → 0.860×**；单窗口 ncu 也显示 **carveout=8（=S）** 比 16/31 保护得更好
   （6.05 vs 6.42/6.96 MB）。**把 carveout 拉到最大是最差的选择**，与 §29.4 的结论一致，只是这里把
   「谁在付账」量清了：**是 carveout 大小，不是 `hitRatio`**。
3. **`hitRatio<1` 真正有用的地方：carveout 被压得比需求还小时「救火」**（Q3）。§C 里
   **carveout=8 MB < 总需求 16 MB、两个都挂 f=1.0 → 1.026×（崩，thrash）**；把 `f` 降到 **0.5**（只让
   8 MB 需求去挤 8 MB carveout）→ **0.879×（反超）**。这是「概率化分区」的**唯一**价值场景：**carveout
   固定且偏小时，用 `f≈carveout/需求` 把 persisting 需求裁到装得下**，避免超配抖动。
4. **多窗口保护可以叠加，但要求 carveout ≥ 总需求**（Q3）。`carve=16、both f=1.0 → 0.814×`（两个 K2
   都快），而 `carve=31、both f=1.0 → 1.343×`（过度预留的惩罚盖过保护）。**两个并发中间量不是简单地
   「各保护一半」**：carveout 必须按**同一时刻存活的中间量总量**配。
5. **ncu 在「多 kernel + replay」下会失真**：自然运行时 §C 的 `carve=16 both f=1.0` 让 K2a/K2b **都**加速
   （0.0113/0.0099），但 ncu 读数却显示只有 stA 侧 `y1` 受益（7.42/8.35）；单 stream 的 `oneB` 又证明
   stB 单独用完全能保护（6.30）。故**多窗口的 split 归因以自然计时为准，ncu 的逐 kernel 拆分受 replay
   干扰**，只作参考（§29.2 已提示 replay 会扰动 L2 演化，此处更明显）。
6. **保护上限仍 ~30%**：单窗口即使 `f=1.0`、carveout 配平，也只把 K2 的 DRAM 读削 ~30%（S=16 时
   16.78→11.58；S=8 时 8.40→6.05）——与 §29 的 ~23–30% 一致。persisting 是**提示**、不是承诺。

### 30.5 与判据 / MPK / 生产案例的关系

- **补的是 §29 留下的「怎么配 carveout / 多窗口」这一维**，把「L2 常驻」这一档从「能不能用」推进到
  「**怎么用才不亏**」：从内存层级看，L2 中间量档的三条边界是 **31 MB 硬上限**、**~30% 保护上限**、
  **carveout 与普通流量零和**；本节补上第四条——**多窗口不叠加、carveout 必须按同活中间量总量配**。
- **对 megakernel 的含义**：megakernel / MPK 恰恰让**整图很多中间量同时活着**（`docs/09` 多算子图、
  MPK §5 跨 task 流水），也就是「同一时刻多个 persisting 段争一个 carveout」——本节证明这条路
  **最多只能护住 ~一个中间量的量级**（31 MB 上限 + ~30% 命中 + 不叠加），且一旦 carveout 配大就净亏。
  这从**机制上**解释了 MPK §5.3（快照行 1312–1325）为何用 **paged shared memory**（片上、可显式管理、
  不受 31 MB 上限、不与他人零和）而不是依赖 L2；也再次指向最稳的两档：**融合进寄存器**（P5-2/P24-1）
  与**锁进 SMEM/DSM**（P29-1）。
- **判据（C1–C5）**：本节仍是 kernel-per-operator（K1/K2 两次 launch），谈的是**中间量的落点/调参**，
  不改 C 判据；`fused`（寄存器档）才满足 C1–C3。
- **与生产案例**：DeepGEMM mega 家族把中间量锁在寄存器/SMEM 并按 `Kstages` 流水（`docs/07` §1.2），
  MPK 的 intermediate 分页也是显式 SMEM——本节给出「为什么不用 L2 persisting」的量化理由：**它挡得住
  一次往返的 ~30%，但挡不住整图的同时多张量**。

### 30.6 局限与踩到的坑

- **合成负载**：K1/K2 逐元素 `float4`、polluter 512 MB 流式；倍数以**量级/方向**读，绝对值随驱动/频率略变。
- **ncu 与自然运行的背离**：多 kernel + 多流场景下，ncu `--replay-mode application` 的逐 kernel DRAM 读
  与自然计时不一致（§30.4 第 5 点）；**跨模式相对关系**仍是结论，逐 kernel 拆分谨慎使用。这是本节的
  主要口径 caveat。
- **只测本机 H100、两个窗口、`f∈{0,0.25,0.5,0.75,1}`**：未测 >2 窗口、`hitRatio<1` 下的多段错峰、
  `cudaMemAdvise`/`cudaMemPrefetchAsync` 等其它放置手段，也未跑到论文级的真实 serving 中间量尺寸分布。
- **H100 参数**：L2=50 MB、`maxPersistingL2CacheSize`=31 MB；不同代 GPU（A100 40 MB L2、B200 更大）
  数值会移，但结论的**形状**（carveout 大小是零和开关、`f<1` 只救小 carveout、多窗口不叠加）应保持。
- device 3（固定空闲）；每配置独立进程跑 ncu，避免 persisting 状态串扰。

## 31. 有界 ready 池 + warmup wave 的防死锁与容量下界（P42-1）

> 本节把 checklist（`docs/05` §2.3「防死锁：交错调度下上游可能饿死下游——DeepGEMM 用 warmup wave +
> ring 容量上界闭式推导」）里**只作资料转引**的一条做成实测：megakernel 的 in-kernel scheduler 把待跑 task
> 放进一个**有界** ready 池（ring block / SMEM 页 / pool block）时，「**准入顺序**」与「**池容量**」如何
> 共同决定**死锁**与**吞吐**。源码 `src/ring_deadlock.cu`，脚本 `harness/ring_deadlock.sh`，原始输出
> `harness/ring_deadlock.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`work_iters=20000`）。

### 31.1 背景：DeepGEMM Mega MoE scheduler 的两件防死锁机制

一手核对 `~/github/DeepGEMM`（commit `057ca59`）的 `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`：

- **`get_num_l1_warmup_waves()`（`:15-45`）**——注释原文 *"Get minimal L1 warmup waves to ensure no
  L1 -> L2 deadlock"*。它在交错调度 L1（专家 GEMM 第一部分）/ L2（第二部分）task 前，**先把前若干个
  m-block 的 L1 task 全部发出去**（warmup wave）；否则 L2 task 会先把 ring/pool 的槽位占住、却在等
  **发不出来的 L1 前置**，形成死锁（`:23` 注释 *"The first L2 wave may touch multiple M blocks; all their
  L1 N tasks must be issued first"*、`:30-35` 讲交错调度每 M-block 遗留的 pending L1 task 如何累积）。
- **`get_num_max_live_pool_blocks()`（`:47-80`）**——注释 *"Host-side ring capacity helper for the MegaMoE
  schedule"*。它给出 ring 容量的**闭式上界**：`num_live_blocks_after_warmup + frontier_growth + wave_margin`
  （`:67-79`）。容量必须装得下「warmup 之后的活跃 block 数 + frontier 增长 + 一个 wave margin」；并且
  `l2_full_mask` 的奇偶 **generation 复用**也依赖 ring 容量（`:140-141` 注释 *"the mask parity per ring
  generation relies on the ring capacity, see `get_num_max_live_pool_blocks`"*，`:153-164`）。
- **运行期就是动态执行这条不变式**：`get_next_task()`（`:372-407`）在返回一个 L2 task 前，会
  **等所有需要的 L1 task 都已被各调度队列取走**（`:398-403` *"Wait until all required L1 tasks are fetched
  from all scheduling queues"*）；`fetch_expert_recv_count()`（`:302-306`）把 `num_sched_l1_waves` 取成
  `min(min_l1_warmup_waves, num_total_l1_waves)`。即：**「任何 L2 task 不得先于其 L1 前置被发」**是
  warmup 的本质不变式，`get_num_l1_warmup_waves` 只是它的**最小**宽度闭式。

MPK 侧对应的是 event 驱动的去中心化调度（`docs/01` §4、快照行 1175–1231 / 1232–1308）：task 完成→触发
event→event 激活→入 scheduler 队列→派发下游 task。它同样需要保证「上游先发」，只是用 event 依赖而非
固定 warmup 宽度。本节用一个**单常驻 kernel**（scheduler/worker 角色分离，对应 MPK §5.1）把这条不变式
的**死锁边界**与**容量下界**量出来。

### 31.2 实验设计

单 kernel，`G+1` 个 block：**block 0 = scheduler**（唯一准入者），其余 `G` 个 = **worker**（消费者）：

- **任务图**：`M` 个 m-block，每块 `Kp` 个 **producer**（L1，可立即跑）与 `Kc` 个 **consumer**（L2，
  **必须等本块全部 `Kp` 个 producer 完成**，fan-in 依赖）。`N = M*(Kp+Kc)`。
- **有界 ready 池**：scheduler 按 `order[]` 把 task **准入**池；池内「已准入未完成」数 **≤ `CAP`**
  （占用语义 = task 在整个生命周期占一个池槽，对应 ring block / SMEM 页）。worker 从池 pop task；
  遇到 consumer 且前置未齐就**自旋等待（占着 worker + 池槽）**。
- **两种准入顺序**：
  - `naive`：每块 **consumer 先于 producer**（对抗性交错，模拟「L2 先于 L1 被发」的坏顺序）；
  - `warmup(W)`：先把前 `W` 个 m-block 的 producer 全发出（warmup wave），其余同 naive。
- **安全**：设备侧**无进度守门狗**——scheduler 用 `clock64()` 检测 `completed` 长时间不增即置
  `deadlock_flag`，所有 worker 在依赖自旋里检查该 flag 退出。故**死锁 case 也让 kernel 正常返回并打印**，
  host 不会挂（§A/§B/§E 里 `ms≈505` 的行就是守门狗判死锁的耗时，~1e9 cycle ≈ 0.5 s）。
- **正确性**：`warmup, CAP=N+1` 全跑完时 `sum` 相对解析参考 `rel=-1.84e-10`（float 累加，量级正确）。

### 31.3 Section A/B：死锁边界与 warmup 宽度

**§A（`M=32,Kp=8,Kc=4,N=384`，扫 `CAP`）**：`naive` 在 **`CAP ≤ Kc=4`** 时死锁、`CAP≥5` 即 ok；
`warmup(W=M=32)` 在 `CAP=1..12` **全部 ok**。原因：naive 顺序下前 `CAP` 个准入的全是 m0 的 4 个 consumer，
它们等 m0 的 producer（排在位置 4..11、因池满无法准入）→ 池被「等前置的 consumer」堵死；`CAP≥5` 时
至少挤进一个 producer → 脱困。**死锁阈值恰 = 首个 m-block 的 consumer 数 `Kc`**。

**§B（`CAP=4=Kc`，扫 warmup 宽度 `W`）**：`done` 精确 = `12·W`（每救一个 block 就完整跑完它的
`Kp+Kc=12` 个 task），`W<M=32` 都死锁、`W=M=32` 才全解：

```
   W  | deadlock | done
    0 |        1 | 0/384      <- 退化为 naive
    1 |        1 | 12/384
    8 |        1 | 96/384
   16 |        1 | 192/384
   32 |        0 | 384/384    <- W=M，全解
```

**§E（依赖宽度敏感性，`M=16,Kp=8`，`run` naive）**：死锁边界随 `Kc` 平移——`Kc=2` 时
`CAP≤2` 死锁（`CAP=4` ok）；`Kc=4` 时 `CAP≤4` 死锁（`CAP=8` ok）；`Kc=8` 时 `CAP≤8` 死锁
（`CAP=9` ok）。同 `Kc` 下 `warmup(W=M=16)` 在 `CAP=2/4/8` **全部脱困**。

**结论**：有界池 + 坏顺序会**真死锁**，边界 = 「先被准入的、等前置的 task 数」；warmup（把前置抢发在前）
是**充分**的解除手段，且**必须够宽**（本模型里要覆盖到每个 consumer 的 producer 都在其前被发）。
真实 DeepGEMM 的 `get_num_l1_warmup_waves` 是这条不变式的**最小**宽度的闭式（靠 wave/block 几何算），
比「全部 producer」小得多，但机理同源。

### 31.4 Section C/D：容量下界与 worker 标度

**§C（`warmup W=M`，扫 `CAP`，`G=64` workers）**：即便**不死锁**，池容量对吞吐是**硬约束**——
`CAP<G` 时同时在飞的 task 不足 G，worker 饿着：

```
   CAP |    ms    | us/task | vs CAP=G
     2 |    9.169 |   23.88 |   27.6×
     8 |    2.300 |    5.99 |    6.9×
    16 |    1.164 |    3.03 |    3.5×
    32 |    0.594 |    1.55 |    1.8×
    64 |    0.334 |    0.87 |   1.01×   <- CAP=G，饱和
   128 |    0.326 |    0.85 |   0.98×
```

`CAP≥G` 后吞吐饱和（0.334→0.326 ms）。即 **ring 容量的「吞吐下界」≈ 同时在飞 task 数 / worker 数**，
正对应 `get_num_max_live_pool_blocks` 里的 "max live"（`:67` `num_live_blocks_after_warmup`）——容量
必须 ≥ 活跃 block 数才能不饿着。

**§D（`CAP=128` 富余，扫 worker 数 `G`）**：`G=1→18.151 ms`、`G=128→0.207 ms`（相对 `G=16`
**5.58×**）；`G≤32` 接近线性（`G=16→32` 1.93×），`G>32` 转次线性（`G=64→128` 只再 1.57×），
因任务总数 `N` 与调度协调成为瓶颈。

### 31.5 与 DeepGEMM / MPK、判据的关系

- **给 `docs/05` §2.3「防死锁」一个量化锚点**：坏顺序 + `CAP≤Kc` 必死；`warmup` 任意 `CAP` 可解但**宽度要够**；
  即使不死锁，`CAP<` 活跃数也会**掉吞吐**（§C `CAP=2` 慢 **27.6×**）。这解释了为什么真实 megakernel
  的 ring 容量要按 `get_num_max_live_pool_blocks` 的**闭式上界**取，而不是随手取小。
- **与 `docs/05` §2.3「跨 launch 复位不靠 host」互补**：`l2_full_mask` 的奇偶 generation 复用依赖 ring
  容量（`mega_moe.cuh:140-141,153-164`）——ring 容量不仅决定「装得下」，还决定「能否用奇偶位掩码免复位」。
- **与 MPK 的关系**：MPK 用 event 依赖**隐式**保证「上游先发」（下游 task 只有在其 triggering event
  激活后才入 scheduler 队列），DeepGEMM 用**显式 warmup wave + 闭式容量**；本节把后者的两条不变式在
  单常驻 kernel 上复现。两者都不改「调度权在 device、粒度到 SM/task」（C4/C5）。
- **判据（C1–C5）**：`warmup`/`naive` 都是 **C1–C4** 的单常驻 kernel（block 0 是固定 scheduler 角色、
  设备侧原子/池推进），但 worker 无固定 task 角色、由池动态取——**仍差 C5 一步**（与 P15-1/P22-1 同档）。
  本节重点是**成本/正确性侧**（死锁与容量），与 P14-1/P16-1/P17-1 的资源足迹互补。

### 31.6 局限与踩到的坑

- **合成依赖模型**：consumer 依赖「本块全部 producer」（fan-in），顺序是「每块 consumer 先于 producer」；
  真实 DeepGEMM 的依赖与 wave 几何更细，故本节的**最小 warmup 宽度 = 全部 producer**，而真实闭式远小于此
  （靠 m-block × L1-N cluster 的 wave 结构）。结论的**形状**（坏顺序死锁、warmup 必要且要够宽、容量
  ≥ 活跃数才满吞吐）应保持，绝对宽度不可直接套用。
- **守门狗是近似**：用「`completed` 长时间不增」判死锁；若某 task 真跑得极慢可能误判。本节 `work_iters=20000`
  的 task 仅 ~0.1 ms，阈值 ~0.5 s，余量充足。死锁 case 的 `ms` 是**守门狗耗时**，非真实计算耗时。
- **单 GPU、`N≤384`、`Kp=8`**：未扫更大 `N`/更深依赖/多级扇入扇出（P22-1 覆盖了扇入/扇出策略但未覆盖
  「有界池容量」这一维）；未与真实 DeepGEMM SM100 路径对跑（SM100 专属）。
- 正确性只查 `sum`（float 量级），未逐 task 比对；`warmup`/`naive` 收敛 case 的 `done` 均 = `N`。
- 目标卡 H100/sm90、device 3（固定空闲）；每 case 独立进程，避免残留状态串扰。

## 32. 有界 ready 池 × 多级依赖 DAG：死锁边界的「阻塞数」与 (CAP,G) 联合约束（P43-1）

> 本节接续 §31（P42-1），把「有界 ready 池 + warmup wave 的防死锁」从**单层 fan-in** 推广到
> **分层依赖 DAG**（`L` 层 × 每层 `W` 个 task × 每 task 依赖上一层 `FIN` 个前驱），直接补掉 §31.6 明留的
> 尾巴「**未测多级扇入扇出 + 有界池的组合**」。源码 `src/layered_pool.cu`，脚本
> `harness/layered_pool.sh`，原始输出 `harness/layered_pool.out.txt`（目标卡 H100/sm90、
> **固定空闲 device 3**、nvcc 13.2、`work_iters=20000`、task ~0.1 ms、守门狗阈值 ~0.05 s）。

### 32.1 背景：单层 Kc → 多级「阻塞数」

§31 单层 fan-in 模型下，坏顺序（块内 consumer 先于 producer）的死锁边界恰等于 **fan-in 宽度 `Kc`**
（首个 m-block 的、先被准入却在等前置的 consumer 数）。多级 DAG 里，「先被准入却在等前置」的 task
不再只是一层的 `Kc` 个，而是**整条依赖深度上的所有等待者**。因此本节要回答：边界是随**深度 `L`**、
**层宽 `W`**、**扇入 `FIN`**、还是它们的某个组合？

抽象成同一个量——**「阻塞数」`B` = 在第一个可运行 task 之前被准入、却在等未准入前驱的 task 数**。
§31 的 `B = Kc`；本节预计 `B` 随层序几何累积。这个量与两个资源直接挂钩（对应 DeepGEMM 的两件机制）：

- **池容量**：`B` 个阻塞 task 若占满有界池（ring block / SMEM 页），scheduler 就再也塞不进一个可运行 task
  ——对应 `get_num_max_live_pool_blocks()`（`scheduler/mega_moe.cuh:47-80`，"Host-side ring capacity helper"）。
- **worker 占用**：§31 的 worker 在依赖自旋期间**占着 worker + 池槽**。若 `B` 个阻塞 task 把 **worker 也占满**，
  即便池里已有可运行 task，也没有空闲 worker 去跑它。多级 DAG 让 `B` 一下涨到 `(L-1)×W`，很容易超过 worker 数
  ——对应 `get_num_l1_warmup_waves()`（`:15-45`，*"ensure no L1 -> L2 deadlock"*）要抢发 L1 的本质。

MPK 的 event 驱动调度（`docs/01` §4、`docs/06` §3）从根上避开这一点：**task 只有在其 triggering event 激活
后才入 scheduler 队列**，即「不把没就绪的 task 放进池」——正是本节对照的 `gated` 策略。

### 32.2 实验设计

单常驻 kernel，`G+1` 个 block：**block 0 = scheduler**（唯一准入者），其余 `G` = **worker**。

- **分层 DAG**：`L` 层，每层 `W` 个 task（`task id = l*W + j`）。层 0 无依赖；层 `l≥1` 的 task `(l,j)`
  依赖层 `l-1` 的 `FIN` 个前驱 `{(l-1,(j+o) mod W) : o=0..FIN-1}`，须**全部 done**。`N = L*W`。
- **有界 ready 池**：同 §31——task 在整个生命周期占一个池槽，池内「已准入未完成」≤ `CAP`；worker pop 到
  未就绪 task 就**自旋（占着 worker + 池槽）**。
- **四种准入策略**：
  - `wide`：**层降序**（`L-1 → 0`），最对抗的顺序（把等前驱的深层 task 先塞满池）；
  - `level`：**严格拓扑层序**（`0 → L-1`，每层按列序）；
  - `warmup(D)`：先把前 `D` 层（升序）发出、其余按 `wide` 降序（= §31 `warmup` 的多级版）；
  - `gated`：**依赖门控**——scheduler 只在某 task 全部前驱**已 done** 时才准入（按 `level` 次序扫描）。
- **安全/正确性**：同 §31，设备侧 `clock64` 无进度守门狗判死锁并置 flag，所有 case 让 kernel 正常返回；
  `gated, CAP=N+1` 全跑完时 `sum` 相对解析参考 `rel=0.00e+00`（逐位一致）。

### 32.3 Section A/A2/A3：边界 = 阻塞数 B，且是 (CAP,G) 联合约束

**§A（`L=4,W=16,FIN=2`，扫 `CAP`，`G=64`）**：`wide` 在 `CAP≤32` 死锁、`CAP=64` ok；`level`/`gated` 在
`CAP=1` 起**全部 ok**（任意 CAP 无死锁）。精扫（`run`）确认 `wide` 在 `CAP=48` 死、`CAP=49` 活。

**§A2（扫 `L`，资源分量各自富余）**：把「池容量」与「worker 数」分开扫，两个边界**都精确 = `B+1`,
`B=(L-1)×W`**：

```
   L  | B=(L-1)*W | CAP*(G=256) | G*(CAP=256) | B+1
    1 |         0 |           1 |           1 | 1
    2 |        16 |          17 |          17 | 17
    3 |        32 |          33 |          33 | 33
    4 |        48 |          49 |          49 | 49
    6 |        80 |          81 |          81 | 81
    8 |       112 |         113 |         113 | 113
```

**§A3（`L=8`，`B=112`，`(CAP,G)` 联合平面）**：只有 `CAP>B` **且** `G>B` 同时成立才达：

```
   CAP\G | G=16  G=32  G=64  G=128 G=256
      32 | DEAD  DEAD  DEAD  DEAD  DEAD
     112 | DEAD  DEAD  DEAD  DEAD  DEAD
     113 | DEAD  DEAD  DEAD  ok    ok
     128 | DEAD  DEAD  DEAD  ok    ok
```

**结论**：多级 DAG 的死锁边界不是「池容量」单变量，而是 **`(CAP,G)` 联合约束**——必须同时满足
`CAP > B`（池里留得下一个可运行 task）**且** `G > B`（有空闲 worker 去跑它）。这个 `B=(L-1)×W` 把 §31
的单层 `Kc` 推广成「**完整依赖深度上累积的阻塞数**」。**部署含义**：真实 megakernel 的 `G = #SM` 是**固定**的，
所以「加大 ring 容量」单独救不了死锁——必须先把 `B` 压到 `< G`（warmup 或拓扑序），这正是 DeepGEMM 抢发
L1 warmup 的第一性理由。

### 32.4 Section B：warmup 深度须覆盖「完整依赖深度」

**§B（`CAP=16=W`，`G=64`，`L=4`，扫 warmup 深度 `D`）**：`D` 每 warm 一层就把剩余阻塞 `B'=(L-1-D)×W`
降 `W`；`D=0` 退化为 `wide` 死锁，`D=3≈L-1` 才全解：

```
   D  | deadlock | done | 剩余阻塞 B'=(L-1-D)*W
    0 |        1 |    0 | 48      <- 退化为 wide
    1 |        1 |   16 | 32
    2 |        1 |   32 | 16
    3 |        0 |   64 | 0       <- 须 warm 完整依赖深度
    4 |        0 |   64 | 0
```

对照 §31 单层「warmup 宽度须 `W=M`（覆盖全部 m-block）」，多级里对应的是「warmup 深度须 ≈ `L-1`
（覆盖完整依赖链）」。二者是同一件事：**warmup 必须宽/深到让每个「等前驱」的 task 都排在它的前驱之后**。

### 32.5 Section C/D/E：拓扑层序与依赖门控（零成本防死锁），FIN 无关

**§C（`level` vs `gated`，扫 `CAP`，`G=64`）**：二者都**任意 CAP 无死锁**；但吞吐上 `level` 更优——
`gated` 在大 CAP 下慢约 **2.2×**（`level/gated` 从 `CAP=2` 的 0.98 掉到 `CAP≥32` 的 **0.457**），
因为 `gated` 的 scheduler 每次都要**扫描剩余 order 找「前驱已 done」的 task**（O(N) 扫描 + `threadfence`），
成为串行瓶颈：

```
   CAP | level ms | gated ms | level/gated
     2 |    1.565 |    1.591 |      0.984
     8 |    0.400 |    0.425 |      0.941
    16 |    0.207 |    0.268 |      0.775
    32 |    0.198 |    0.434 |      0.457
    64 |    0.201 |    0.440 |      0.457
   128 |    0.200 |    0.439 |      0.456
```

**§D（`gated`，`CAP=128`，扫 `G`）**：`G=1→3.125 ms`、`G=16→0.387 ms` 后**不再下降**（`G=32/64/128`
反而 ~0.9×）——因本图 `N=64`，`G>N` 只是空转 worker，与 §31 §D（任务数/协调成瓶颈）同源。

**§E（`wide`，`L=4,W=16`，`G=256`，扫 `FIN`）**：边界恒 = `B+1 = 49`，**与扇入宽度 `FIN` 无关**
（`FIN=1/2/4/8` 全 49）——边界只由**层序几何 `L,W`**（即阻塞数 `B`）决定。这与 §31 单层边界 = `Kc`
看似不同，实则同构：两者都是「**阻塞在前的前驱等待 task 数**」，单层时它恰等于 fan-in 宽度 `Kc`。

### 32.6 与 DeepGEMM / MPK、判据的关系

- **给 §31/`docs/05` §2.3「防死锁」补齐多级形态**：单层结论（坏顺序在 `CAP≤Kc` 死锁、warmup 要够宽）
  在多级 DAG 上保持形状，但**边界从「宽度 `Kc`」变成「阻塞数 `B=(L-1)×W`」，且从「池容量单约束」
  变成「**池容量 × worker 数联合约束**」**。这解释了真实 megakernel（`G=#SM` 固定）为什么**必须**用
  warmup/拓扑序把 `B` 压到 `< G`，而不能只靠加大 ring。
- **两种正解、各有代价**：
  - **拓扑层序 `level`**（每层按序发）：任意 CAP/G 无死锁、**吞吐无损**（§C 最快）——对应 DeepGEMM 的
    wave/block 几何排布（L1 先于 L2 被发）与「最小 warmup 宽度闭式」；
  - **依赖门控 `gated`**（只准入就绪 task）：等价于 MPK 的 event 驱动入队（`docs/06` §3），天然无死锁，
    但若实现成「每次扫描找就绪 task」则大 CAP 下慢 ~2×——MPK 用 **event→task 的直接映射 + 原子 ring**
    避免扫描，故不付这笔代价（`docs/06` §2/§4）。
- **判据（C1–C5）**：四策略都是**单常驻 kernel**、scheduler/worker 角色分离、设备侧推进（C1–C4），
  worker 无固定 task 角色、由池/门控动态取——**仍差 C5 一步**（与 P42-1/P15-1/P22-1 同档）。
  本节是 megakernel 的**正确性/容量成本侧**，与 P14-1/P16-1/P17-1 的资源足迹、P42-1 的单层防死锁互补。

### 32.7 局限与踩到的坑

- **首次预测被实测推翻**：起初以为 `wide` 边界 = `B+1`（只算池容量），实测 `L=6/8` 在 `G=64` 下**任何
  CAP 都死锁**（§E 补点初版全 DEADLOCK）——才发现**第二个约束是 worker 数**（阻塞 task 占满 worker）。
  这正是「为什么必须真跑」的例子；改测 §A2（分别富余资源扫 CAP 与 G）+ §A3（联合平面）后才定量。
- **合成依赖模型**：每 task 只依赖上一层 `FIN` 个前驱（无跨层、无多级扇入扇出），宽/深几何规则；
  真实 DeepGEMM 的依赖 + wave 几何更细（闭式 `get_num_l1_warmup_waves` 远小于 `(L-1)×W`）。
  结论的**形状**（边界 = 阻塞数、`(CAP,G)` 联合、warmup 须覆盖深度、拓扑序/门控零成本防死锁）应保持，
  绝对宽度不可直接套用。
- **`gated` 的实现是朴素扫描版**：其 ~2× 慢是「每准入一次扫一遍剩余 order」的工程代价，**非门控语义
  固有**——真实实现（event→task 直接映射）不付这笔账；故 §C 的「`level` 更快」只对**本实现**成立。
- **单 GPU、`N≤384`、`FIN≤16`**：未测更大 `N`/多级扇入扇出组合/与真实 DeepGEMM SM100 路径对跑
  （SM100 专属）。正确性只查 `sum`（float 量级），未逐 task 比对。目标卡 H100/sm90、device 3（固定空闲）。

## 33. 有界 ready 池 × 不规则依赖 DAG：阻塞数 B(π) 的普适定义与「层号 ≠ 拓扑序」（P44-1）

> 本节接续 §31（P42-1）/ §32（P43-1），把「有界 ready 池 + warmup wave 的防死锁」从**规则分层
> DAG**（`L` 层 × 每层等宽 `W` × 上一层 `FIN` 扇入）推广到**不规则 DAG**：每层宽度不等、跨层
> skip 前向边、多级扇入、以及**层号乱序**（名义层号与依赖方向不一致）。直接补掉 §32.7 明留的
> 尾巴「依赖模型是规则的分层 × 每层等宽 × 上一层扇入——**未测不规则/跨层依赖、多级扇入扇出混合**」。
> 源码 `src/irregular_pool.cu`，脚本 `harness/irregular_pool.sh`，原始输出
> `harness/irregular_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
> `work_iters=20000`、task ~0.1 ms、守门狗阈值 ~0.05 s、`G=64`）。

### 33.1 背景：从 `(L-1)*W` 到普适的「阻塞数」`B(π)`

§31 单层 fan-in 的边界 = fan-in 宽度 `Kc`；§32 规则分层 DAG 的边界 = `B=(L-1)*W`。两者的共性是
「**在第一个可运行 task 之前，被准入却仍在等前驱的 task 数**」。本节要做两件事：

1. **给出这个量的普适定义与正确算法**。§32 用的直觉「第一个就绪 task 的位置」在**源在前、后面再堆满
   阻塞**的形态下会**算错**（见 §33.5 的 `scramble`）。本节改用**贪心模拟**：沿准入 order 逐个准入、
   并**立刻排空**任何「前驱全已 done」的就绪 task（等价于一个无限快 worker），池内「已准入未完成」的
   **峰值占用**即 `B(π)`。实测死锁边界 **`CAP* = G* = B(π)+1`**（`B` 用 `>` 判，故 +1）。
2. **测不规则几何下 §32 的闭式是否还成立**，以及**什么条件下「层号升序」不再是安全的准入序**。

`B(π)` 仍与 DeepGEMM 的两件机制直接挂钩：它同时是**池容量下界**（`get_num_max_live_pool_blocks()`，
`scheduler/mega_moe.cuh:47-80`）与**worker 占用下界**（阻塞 task 占着 worker，对应
`get_num_l1_warmup_waves()`，`:15-45`，*"ensure no L1 -> L2 deadlock"*）。

### 33.2 实验设计

单常驻 kernel，`G+1` 个 block：**block 0 = scheduler**（唯一准入者），其余 `G` = **worker**。
与 §31/§32 相同的池模型（task 全程占一个池槽、池内「已准入未完成」≤ `CAP`、worker pop 到未就绪 task
就自旋并**占着 worker + 池槽**）。DAG 由四类不规则性参数化：

- **不规则层宽 `W[l]`**：`reg`（等宽 `W0`）/ `grow`（`W0+l`）/ `shrink`（`W0-l`，下限 2）/ `zigzag`；
- **跨层前向 skip**：task `(l,j)` 的前驱层在 `[l-span, l-1]` 轮转取（`span=1` 即只取上一层）；
- **多级扇入**：`FIN` 个前驱跨多个更浅的层；
- **层号乱序 `scramble`**：把名义层号整体旋转 `s` 位（`layer_of = (topo_layer + s) mod L`）——
  对偶于「一般 DAG / 层号与依赖方向不一致」的普遍情形（真实模型里的跨层反馈、迭代子图、或
  编译器 IR 里算子被重新编号）。

准入策略与 §32 对齐并新增 **`topo`**：`wide`（名义层降序）/ `level`（名义层升序）/
`warmup(D)`（前 `D` 名义层升序 + 其余降序）/ **`topo`（Kahn 真拓扑序，小 id 优先）** /
`gated`（依赖门控，按 `level` 次序扫描就绪 task）。安全：设备侧 `clock64` 无进度守门狗检测死锁并置
flag，所有 case 让 kernel 正常返回（host 不挂）。

### 33.3 Section A：不规则层宽——闭式 `(L-1)*W0` 失效，真边界 = `B(π)+1`

`L=4, W0=16, FIN=2, span=1, scramble=0`，`wide` 策略，`G` 富余扫 `CAP`、`CAP` 富余扫 `G`：

```
  shape   | W[l] 序列            | N  | W0 | B(pi) | CAP* | G*  | (L-1)*W0+1
  --------+----------------------+----+----+-------+------+-----+-----------
  reg     | 16 16 16 16          | 64 | 16 |    48 |   49 |  49 |        49
  grow    | 16 17 18 19          | 70 | 16 |    54 |   55 |  55 |        49
  shrink  | 16 15 14 13          | 58 | 16 |    42 |   43 |  43 |        49
  zigzag  | 16 8 16 8            | 48 | 16 |    32 |   33 |  33 |        49
```

要点：**`(L-1)*W0+1` 恒为 49（只用了 `L,W0`），但对不规则宽度给出的是错值**——真边界
`CAP*=G*=B(π)+1` 随每层宽度变化（55/43/33）。等宽时 `B(π)=N-W0=(L-1)*W` 恰好对上 §32。
`CAP*` 与 `G*` 逐行相等，再次印证 §32 的 **`(CAP,G)` 联合约束**。

### 33.4 Section B：跨层前向 skip / 多级扇入——层号升序仍是拓扑序

`shrink`（`N=58, W0=16`），`span=1..4`：

```
  span | N  | B_wide(pi) | wide CAP* | wide G* | level ok@CAP=1 | topo ok@CAP=1
  -----+----+------------+-----------+---------+----------------+---------------
     1 | 58 |         42 |        43 |      43 | ok             | ok
     2 | 58 |         42 |        43 |      43 | ok             | ok
     3 | 58 |         42 |        43 |      43 | ok             | ok
     4 | 58 |         42 |        43 |      43 | ok             | ok
```

要点：**只要前驱只来自更浅的层（前向边），「层号升序」仍是一个合法拓扑序** —— `level`/`topo` 在
`CAP=1` 即无死锁；`wide` 的 `B(π)=N-W0=42` 与 `span` **无关**（`wide` 把层 0 之外的 task 全先准入、
全部在等前驱，阻塞数是完整的）。多级扇入只改变「每个 task 要等几个前驱」，不改变这个总量。

### 33.5 Section C：层号乱序——`level` 失效，必须真拓扑线性化（headline）

`reg`（`W=[16,16,16,16]`），把名义层号旋转 `s` 位：

```
   scr | N  | B_wide | B_level | B_topo | wide CAP* = B_wide+1 | level ok@CAP=1 | topo ok@CAP=1
   ----+----+--------+---------+--------+---------------------+----------------+---------------
     0 | 64 |     48 |       0 |      0 |                  49 | ok             | ok
     1 | 64 |     32 |      16 |      0 |                  33 | DEADLOCK       | ok
     2 | 64 |     16 |      32 |      0 |                  17 | DEADLOCK       | ok
     3 | 64 |     32 |      48 |      0 |                  33 | DEADLOCK       | ok
```

精扫（`scramble=1`）：`level` 在 `CAP=1` 死锁、`CAP=17` 才活（`B=16`，须 `CAP>B`）；`topo` 恒
`B=0`、`CAP=1` 即活。要点：

- **层号乱序后「层号升序」不再是一个拓扑序**：它先发「名义浅层」，而这些 task 的**真前驱恰在名义
  深层**（最后才发），于是 `B_level` 从 0 骤升到 16/32/48，小池直接死锁。
- **真拓扑序 `topo`（Kahn）恒 `B=0`**：因为它的准入序本身就是「每个 task 的前驱都在它之前」，故
  `CAP=1`、`G=1` 即可无死锁。这正对应 **MPK 依赖分析器用 Kahn 求 linearization**（`docs/08` §5）、
  而非按层名/op 名顺序遍历——**「按名义层/算子名遍历」≠「拓扑线性化」**。
- `B()` 的算法必须是 §33.1 的**贪心峰值**，不能是「第一个就绪 task 的位置」：`scramble=3` 下
  `wide` 的名义降序**第一个 task 就是源（就绪）**，但随后会堆满 32 个阻塞 task，真边界是 33 而非 1。

### 33.6 Section D / E：`gated` 的扫描代价；warmup 的判据是「剩余阻塞」不是「层数」

**§D（多级扇入下 `gated` vs 预计算 `topo`，`CAP=512` 富余）**：`gated` 对任意 DAG/任意顺序无死锁，
但 scheduler 每次要**扫描剩余 order** 找就绪 task，代价约为 `topo` 的 **1.65–2.51×**（`reg N=64`
~2.4×、`shrink N=58` ~1.7×；`FIN` 越大略增）。这与 §32 §C 的 ~2.2× 同源，**是朴素扫描实现的工程代价，
非门控语义固有**——MPK 用 **event→task 直接映射**避免扫描（`docs/06` §2/§4），真实实现不付这笔账。

**§E（`warmup(D)`，`CAP=W0, G=64`）**：`reg,scr=0` 时 `B(π)` 随 `D` 由 48→32→16→0，`D=3≈L-1` 才把
`B` 压到 `<CAP`（无死锁）；`grow,scr=0` 同样 `D=3` 才清。但 `scramble=1` 时**任何 `D` 都救不了**
（`B(π)` 始终 ≥16，因为 warm 名义层拿不到真正的浅层，阻塞在乱序下重新分布）。要点：**warmup 的正确
判据是「剩余阻塞 `B' < G`（且 `< CAP`）」而非「warm `D` 层」**——层内 task 数随宽度变，层号还可能
非拓扑；DeepGEMM 的 `get_num_l1_warmup_waves` 闭式正是建立在**规则 wave 几何**上的近似。

### 33.7 与 DeepGEMM / MPK、判据的关系

- **给 §31/§32 的防死锁理论收口**：单层 `Kc`（§31）→ 规则分层 `(L-1)*W`（§32）→ **不规则 DAG 的普适
  量 `B(π)`（本节的贪心峰值）**，边界一律 `CAP*=G*=B(π)+1`。闭式 `(L-1)*W` 只是等宽 + 拓扑层序的
  特例；不规则宽度、跨层、乱序都会让它失效，而 `B(π)` 总成立。
- **「层号 ≠ 拓扑序」是本节给生产/编译器路径的核心提醒**：只要图里有跨层反馈、迭代子图，或 IR 里
  算子被重新编号，**任何「按名义层/op 名遍历」的准入序都可能死锁**；安全且零阻塞的是**真拓扑序**。
  这与 MPK 依赖分析器「最近写者 SSA + Kahn 定序」（`docs/08` §2/§5）一致——它天然产出拓扑序，
  因此 `B=0`、不依赖 warmup。
- **两条工程正解**：(a) **预计算拓扑序**（`topo`，零阻塞、零扫描，但要先跑一遍依赖分析）；
  (b) **依赖门控**（`gated`，= MPK §5.1 event 驱动入队，`docs/06` §3，任意图无死锁，但用事件映射
  直取而非扫描）。`wide`/`level` 这类「按几何名字排」的启发式只在图确实分层时可用。
- **判据（C1–C5）**：四策略都是**单常驻 kernel**、scheduler/worker 角色分离、设备侧推进（C1–C4），
  worker 无固定 task 角色、由池/门控动态取——**仍差 C5 一步**（与 P42-1/P43-1/P15-1/P22-1 同档）。
  本节是 megakernel 的**正确性/容量成本侧**，与 §31/§32 的防死锁、P14-1/P16-1/P17-1 的资源足迹互补。

### 33.8 局限与踩到的坑

- **两次预测被实测推翻**（都靠真跑发现）：(a) 初版把 `B` 定义为「第一个就绪 task 的位置」，`scramble=3`
  下给出 `B=0` 却实测 `CAP*=33`——才发现源在前、阻塞在后的形态会漏算，改用**贪心峰值**；(b) 初版用
  「深层→浅层的回边」制造非拓扑序，但那在严格分层的前向图里**必然成环**（浅层本可到达深层），
  gated 永不收敛、`topo` 的 Kahn 输出残缺导致非法访存——改用**名义层号旋转**（保持 DAG 无环、
  只让层号与拓扑方向不一致）后才稳定。
- **合成依赖模型**：`N≤70`、`FIN≤8`、`span≤4`、`scramble` 只是层号整体旋转（非任意图重编号）；
  真实 DeepGEMM 的 wave 几何与依赖更细（闭式远小于 `(L-1)*W`），MPK 的 tGraph 含 view/event 融合后的
  复杂事件结构（`docs/08`）。结论的**形状**（边界 = `B(π)`、`(CAP,G)` 联合、拓扑序/门控零成本防死锁、
  warmup 判据 = 剩余阻塞）应保持，绝对数值不可直接套用。
- **`gated` 仍是朴素扫描版**：其 ~1.7–2.5× 慢是工程代价；`topo` 的计时含少量「先算拓扑序」的开销，
  但本实现里拓扑序在 host 预计算、设备侧只按序取，故代价主要在设备扫描侧。
- **单 GPU、单次运行**：未测更大 `N`、与真实 DeepGEMM SM100 路径对跑（SM100 专属）、未测
  worker 数 `G` 在不规则图上的独立边界（§A 只验证 `G*` 与 `CAP*` 同行相等）。正确性只查 `sum`
  （float 量级 `rel=0.0e+00`），未逐 task 比对。目标卡 H100/sm90、device 3（固定空闲）。

## 35. 有界 ready 环形缓冲：事件驱动调度「无界」假设的现实检验（P46-1）

> 本节接续 §31（P42-1）/ §32（P43-1）/ §33（P44-1）/ §34（P45-1）。四节把「有界 ready 池 + 准入顺序
> ⇒ 死锁」收口到普适量 `B(π)`、证明事件驱动（worker 完成即 push 就绪后继）把 `B(π)` 归零、`CAP=1`
> 且 `G=1` 即活。**但 §34.9 明确留了尾巴**：「`event` 的 ready 队列**无界**（按 `N+4` 开数组），
> 真实实现要**环形缓冲 + 容量上界**」。本节把「有界」补上。
>
> 这不是学院派问题：真实 megakernel 的 in-kernel 队列**就是固定容量的有界环形缓冲**——
> **MPK** 快照 `f9eb70c` 里 `global_runtime_config.per_worker_queue_len = 8192` /
> `per_sched_queue_len = 4096`（`persistent_kernel.cuh:1790-1791`），worker 侧用
> `assert(next_task_pos[q] + per_worker_queue_len > last_task_pos[q])`（`:1027`）保证不溢出；
> 上方注释（`:1782-1789`）明说**为什么从 2048 加到 8192**——"136-worker B200 runs can temporarily
> outpace the default queue depth on the 'fast' schedulers that only feed 2 workers … Bump to 8192"
> （即**速率/背压**决定容量，而非某个静态图量）。**DeepGEMM Mega MoE**（`057ca59`）则用
> `get_num_max_live_pool_blocks()`（`scheduler/mega_moe.cuh:47-80`，注释 *"Host-side ring capacity
> helper"*）在编译期**闭式**算出 ring 容量上界，配 `get_num_l1_warmup_waves()`（`:15-45`，
> *"ensure no L1 -> L2 deadlock"*）静态化防死锁。
>
> 源码 `src/bounded_event_ring.cu`，脚本 `harness/bounded_event_ring.sh`，原始输出
> `harness/bounded_event_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
> `work_iters=20000`、设备侧 `clock64` 无进度守门狗使死锁 case 也正常返回）。

### 35.1 机制：从「无界」到「有界 + 阻塞背压」

单常驻 kernel，`G+1` 个 block：**block 0 = 播种源任务**（layer 0，前驱数 0），其余 `G` = worker。
每个 task 一个 `dep_left`（未完成前驱数，即 §5.1 event 计数器的补集）；worker 完成 task 后对每个后继
`atomicSub(dep_left, 1)`，**归零即入队**（= MPK 的 `trigger_event`，`docs/06` §2/§4、`docs/08` §7）。
两种 ring 策略：

- **`unb`**（= §34 的 `event`）：无界，`slot = atomicAdd(tail,1)` 直接取槽（容量 `N+4`，从不失败）。
- **`bnd`**：有界容量 `CAP` 的环形缓冲 + **阻塞式背压**——入队前先 `CAS(tail)` 预留槽，**仅当
  `tail - head < CAP` 才成功**，否则自旋等空位（真实有界队列的做法）。worker 在 push 与 pop 之间
  不「让路」，于是一旦环被填满、又没有 worker 处于 pop 相，整机就卡在 push 上互相等待 → 死锁
  （守门狗 `clock64` 检测并置 flag，kernel 正常返回，host 不挂）。

DAG 生成复用 §33/§34 的分层模型（层宽 `reg`/`grow`/`shrink`/`zigzag`、跨层 `span` skip、`FIN` 扇入）。
记 `W_max` = 最大层宽（分层图的就绪前哨上界）。

### 35.2 Section A：`unb` 恒可完成；`bnd` 需一个小容量（远小于 `W_max`）

`L=5, W0=24, FIN=1, G=64`：

```
   shape   | N   | Wmax | unb | CAP_hard | CAP_safe | safe/Wmax
   --------+-----+------+-----+----------+----------+----------
   reg     | 120 |   24 | ok  |        2 |        2 |      0.08
   grow    | 130 |   28 | ok  |        3 |        3 |      0.11
   shrink  | 110 |   24 | ok  |        1 |        2 |      0.08
   zigzag  |  96 |   24 | ok  |        2 |        4 |      0.17
```

- **`unb` 恒可完成**（30/30 正确），与 §34 一致。
- **`bnd` 在 `G=64` 时只需 `CAP=2~4`**，远小于 `W_max=24`——因为 **64 个 worker 在并行取走就绪
  task**，环只需缓冲「宽 `W_max` 的那一层里**还没被取走**的那一小截」。
- §A2（`FIN∈{1,2,4,8}`、`span∈{1,2}`）只在 `CAP_hard` 上从 1–2 轻微抬到 2–3，`CAP_safe` 仍在 2–4：
  **扇入/跨层只轻微影响需求，主导仍是有几个 worker 在并行消费。**
- 精扫（`harness/bounded_event_ring.out.txt`）`reg, G=64`：**`CAP=1` 死锁（`done=118/120`，卡在
  尾巴上）、`CAP=2` 稳**；`unb` 在 `CAP=1` 恒活。

### 35.3 Section B：容量随 worker 数下降 —— `CAP_safe + G ≈ W_max + 1`

`shrink, L=5, W0=40, FIN=2`（`W_max=40`）：

```
   G   | CAP_hard | CAP_safe | CAP_safe+G | (Wmax+1)-(CAP_safe+G)
   ----+----------+----------+------------+------------------------
     1 |       39 |       39 |         40 |                     +1
     2 |       38 |       38 |         40 |                     +1
     4 |       36 |       37 |         41 |                     +0
     8 |       33 |       33 |         41 |                     +0
    16 |       25 |       26 |         42 |                     -1
    32 |        9 |        9 |         41 |                     +0
    64 |        2 |        2 |         66 |                    -25
   128 |        2 |        2 |        130 |                    -89
```

- **`G ≤ W_max` 时 `CAP_safe + G ≈ W_max + 1`**——**G 个 worker 在飞、环只需缓冲宽层的
  `W_max − G` 个未被取走的就绪 task**。G=1 时几乎要整层宽（39/40），G=32 时只要 9。
- **`G ≥ W_max` 时容量触底在 ~2**（G=64/128 都 2）——多到「宽层一放出来就被抢空」。
- 精扫 3（`CAP=26` 固定）：`G=1` 立即死锁（`done=2/190`）、`G=16`/`G=64` 均完成——**容量与 worker
  数是可互换的资源**。
- **这与 §31–§34 的 `B(π)+1` 是两种不同的量**：`B(π)` 由**准入顺序/依赖深度**决定、是图的**静态
  不变量**；这里的容量需求由**并发就绪数 ÷ worker 数**决定、是**排队/速率量**。MPK 的注释
  （`persistent_kernel.cuh:1782-1789`）说的正是这件事：容量是被速率「逼」上去的。

### 35.4 Section C：安全容量以上，加大容量不再提速

`reg, L=6, W0=64, G=64`：

```
   CAP  | ms(bnd) | vs CAP=512 | done
   -----+---------+------------+-----
      1 |  30.70  |   DEADLOCK | 298/384  <- 守门狗 (CAP<CAP_safe)
      2 |  30.68  |   DEADLOCK | 355/384  <- 守门狗 (CAP<CAP_safe)
      4 |  30.68  |   DEADLOCK | 381/384  <- 守门狗 (CAP<CAP_safe)
      8 |   0.375 |      0.99x | 384/384
     16 |   0.375 |      0.99x | 384/384
     64 |   0.376 |      0.99x | 384/384
    128 |   0.376 |      0.99x | 384/384
    256 |   0.378 |      1.00x | 384/384
    512 |   0.378 |      1.00x | 384/384
```

- **`CAP < CAP_safe` 必挂**（守门狗 ~30 ms 返回），**`CAP ≥ CAP_safe` 后延迟完全持平**（0.375 ms，
  差 <1%）。**容量不是吞吐旋钮，只是安全阀**——与 §31 §C「容量是吞吐硬下界」不同（那里的池容量
  直接映射到 worker 饥饿程度，因为 §31 的 worker 会**占着未就绪 task 自旋**；事件驱动不会，
  故一旦有就绪 task 可取，容量大小就不影响吞吐）。

### 35.5 Section D/E：正确性、规模标度、与生产的对照

**§D 正确性**：`unb` 与 `bnd(CAP=W_max+1)` 在 8 个 case（4 形状 × `scramble∈{0,1}`）上给出**逐位
相同的 `sum`**（`rel` 三档 1.5e-9 / 2.0e-9 / 2.4e-9，`done=N/N`）——**有界/无界只改「何时入队」，
不改「算什么」**（CAP 足够时 `bnd` 退化为 `unb` 的执行序）。

**§E 规模标度**（`reg, L=5, FIN=1`，`G=64` 与 `G=8` 两档）：

```
   W0  | N    | Wmax | CAP_safe(G=64) | /Wmax | CAP_safe(G=8) | /Wmax
   ----+------+------+----------------+-------+---------------+------
     16 |   80 |   16 |              2 |  0.12 |            10 |  0.62
     64 |  320 |   64 |              6 |  0.09 |            59 |  0.92
    256 | 1280 |  256 |            197 |  0.77 |           252 |  0.98
    512 | 2560 |  512 |            446 |  0.87 |           509 |  0.99
   1024 | 5120 | 1024 |            958 |  0.94 |          1021 |  1.00
```

- **容量随 `W_max` 增长、随 `G` 下降**，近似 `CAP_safe ≈ max(2, W_max − G + O(1))`（大 `W`、小 `G`
  时贴近 `W_max`；大 `G` 时贴近 2），有 ±几% 的速率噪声。
- **对生产的解释**：MPK 直接钉 `per_worker_queue_len=8192`（`:1790-1791`）+ `assert`（`:1027`）不溢出，
  DeepGEMM 用 `get_num_max_live_pool_blocks`（`:47-80`）闭式给上界——**两者都是「取一个远超实际
  峰值就绪数的安全常量」**，正好兜住本节量出的「实际需求其实很小、但边界不确定」。

### 35.6 与 §31–§34 / MPK / DeepGEMM 的关系、判据

- **两类容量下界，别混用**：§31–§34 的 `B(π)+1` 是**准入顺序**造成的「阻塞 task 占了池槽」；
  本节的 `≈W_max−G` 是**并发就绪数**造成的「环被填满、pusher 互相卡住」。前者是图的静态量、
  由依赖结构决定；后者是排队量、由 worker 数决定。
- **事件驱动不是「免容量」**：§34 的「`CAP=1`/`G=1` 即活」只成立于**无界**队列。一旦把队列做成
  有界环 + 阻塞背压，就**重新引入一个容量下界**（且这个下界随 `G` 变、且在边界上**不确定**）。
  这与 MPK 把队列钉成常量 8192、DeepGEMM 预推导容量上界是同一件事的两面。
- **判据（C1–C5）**：`unb`/`bnd` 都是**单常驻 kernel**、设备侧推进（C1–C4）；`bnd` 额外引入
  「环容量」这一设备内协调参数。两者都无固定角色 worker/scheduler 动态派发（`event` 模式最接近
  C5 但 scheduler 角色仍留在 block 0），**与 §31–§34 同档、仍差 C5 一步**（C5 的完整去中心化见
  P15-1 §9 / `docs/06` §3）。本节是 megakernel 的**正确性/容量成本侧**。

### 35.7 局限与踩到的坑

- **边界不确定（headline caveat）**：靠近安全边界时**单次可完成、多次可能死锁**（§A 中
  `CAP_hard < CAP_safe`），故 `CAP_safe` 用「5 次全不挂」的二分近似下界，绝对数值不可外推；
  `CAP_safe + G ≈ W_max+1` 只在同一次运行内成立。**触发点是「环满 + 无 worker 在 pop 相」的时序竞态。**
- **入队的内存序**：与 §34 相同的「先 `atomicAdd(tail)`/CAS 预留、后写数据」模式，理论上消费者可能
  在数据可见前读到槽（未加 release 发布）。实测 `unb` 30/30 正确、`bnd` 在 `CAP≥CAP_safe` 恒正确
  （写读延迟覆盖了窗口），但**这不是形式化保证**；真实实现（MPK 的 `atom.add.release.gpu` + `st.relaxed.gpu`
  + CAS-release 放行读游标，`docs/06` §3）用 release 语义把它做实。
- **合成依赖模型**：`N≤5120`、层号/扇入规则化、正确性只查 `sum`；未与真实 DeepGEMM SM100 路径对跑
  （SM100 专属，本机不具备）。`work_iters=20000` 让每 task ~µs 级、便于暴露调度差异。
- **`bnd` 是「最朴素」的有界实现**：真实实现会用「环+背压但允许 pusher 掉落重试」或「scheduler
  作为唯一生产者」（MPK），故本节量的是**下界形态**，不是生产实现的最优形态。目标卡 H100/sm90、
  device 3（固定空闲），nvcc 13.2。

## 34. 事件驱动的「就地入队」：MPK §5.1 如何从机制上消掉有界 ready 池的死锁（P45-1）

> 本节接续 §31（P42-1）/ §32（P43-1）/ §33（P44-1）。三节把「有界 ready 池 + 准入顺序 ⇒ 死锁」
> 收口到一个普适量 `B(π)`（贪心峰值池占用），边界 `CAP* = G* = B(π)+1`，并给出两条零成本正解
> 「预计算拓扑序 `topo`」与「依赖门控 `gated`」。§33.8 明确留下三条尾巴：合成依赖模型 `N≤70`、
> `scramble` 只是层号整体旋转、**未测更大 `N` / 未独立扫 `G`**；且 `gated` 仍是**朴素扫描**实现
> （每次准入扫剩余 order），其 ~1.7–2.5× 慢只是工程代价。本节把 `gated` 从「调度器扫描」改成
> **MPK §5.1 的真机制**——每个 task 一个**事件计数器**（未完成前驱数），由完成该前驱的 **worker 直接
> 把「计数归零」的后继 push 进 ready 队列**，调度器根本不参与准入——并补齐更大 `N` 与独立扫 `G`。
> 源码 `src/event_driven_pool.cu`，脚本 `harness/event_driven_pool.sh`，原始输出
> `harness/event_driven_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
> `work_iters=20000`、设备侧 `clock64` 无进度守门狗使死锁 case 也正常返回）。

### 34.1 背景：`gated` 是「扫描门控」，而 MPK 是「事件驱动直接入队」

§33 的两条工程正解之一 `gated` 的名字来自「依赖门控」，但其**实现**是：scheduler 每轮**扫描剩余
order**，用 `all_preds_done()` 找一个就绪 task 才准入。它**行为上**确实消除了死锁（只准入就绪 task
⇒ 池内无阻塞 task ⇒ `B≡0`），但代价有两笔：每轮扫描 `O(剩余 order)`，以及「就绪判断」这个本可由
依赖结构天然给出的信息被重复计算。

MPK §5.1 的机制完全不同（快照行 1175–1231）：tGraph 以一个 `start` event 起，scheduler 把依赖该
event 的 task **一次全部**派发；每个 task 完成后**通知它触发的 event**；一个 event 的**全部前置 task
都完成**（各触发到所需次数）时 event 被 **activated** 并入队，scheduler 再放行依赖它的下一批 task。
换言之，**「一个 task 就绪」不是一个需要被扫描发现的谓词，而是「它的最后一个前置完成」这个事件**——
完成者**顺手**把后继入队。这正对应本节的 `event` 模式：`dep_left[task]` 是事件计数器，worker 完成
task 后对每个后继 `atomicSub(dep_left[succ],1)`，归零者**直接 push**。

### 34.2 实验设计：四种准入机制

单常驻 kernel，`G+1` 个 block：**block 0 = scheduler**（`scan`/`topo`/`raw` 用；`event` 只用它**播种
源任务**），其余 `G` = worker。task 全程占一个池槽，池内「已准入未完成」≤ `CAP`。四种机制：

- **`raw`**（对照 §33 的 `wide`/`level`）：scheduler 按 `order` **直接准入、不查就绪**——坐实
  `B(π)+1` 死锁边界（worker pop 到未就绪 task 就自旋、占着 worker + 池槽）。
- **`scan`**（= §33 的 `gated`）：scheduler 扫剩余 order，**就绪才准入**；`B≡0`，付扫描账。
- **`topo`**：scheduler 按**预计算拓扑序**（Kahn，host 侧算）准入；`B≡0`，不付扫描账，但要先跑依赖分析。
- **`event`**：**无 scheduler 准入**。seed 由 block 0 把源任务（前驱数 0）入队；worker 完成后对每个
  后继 `atomicSub(dep_left,1)`，**归零即 push**。ready 队列**只装就绪 task**。

`B(π)` 仍用 §33.1 的**贪心峰值**定义。DAG 生成复用 §33 的四类不规则性（层宽 `reg`/`grow`/`shrink`/`zigzag`、
跨层 `span` skip、`FIN` 扇入、`scramble` 层号旋转）。

### 34.3 Section A：三机制数值一致（含不规则 / 乱序）

`shape ∈ {reg,grow,shrink,zigzag}`、`scramble ∈ {0,1}`、`CAP=N+1` 富余：

```
   shape   | scr | N  | scan rel | topo rel | event rel | done(scan/topo/event)
   --------+-----+----+----------+----------+-----------+-----------------------
   reg     |   1 | 64 |  0.0e+00 |  0.0e+00 |   0.0e+00 | 64/64/64
   grow    |   1 | 70 |  0.0e+00 |  0.0e+00 |   0.0e+00 | 70/70/70
   shrink  |   1 | 58 |  0.0e+00 |  0.0e+00 |   0.0e+00 | 58/58/58
   zigzag  |   1 | 48 |  0.0e+00 |  0.0e+00 |   0.0e+00 | 48/48/48
```

三机制对同一 DAG、同一 task 工作给出**逐位相同的 `sum`**（`rel=0.0e+00`，8/8 case）。`event` 的
「完成即 push」**不动数值**——依赖只影响执行顺序，`dep_left` 归零恰保证「所有前驱已写、后继才入队」。

### 34.4 Section B：死锁边界——`raw` 的 `B(π)+1`，以及 `scan`/`topo`/`event` 的 `B≡0`

```
   shape   | scr | N  | B_raw | raw CAP* | raw G* | scan ok@CAP=1 | topo ok@CAP=1 | event ok@CAP=1,G=1
   --------+-----+----+-------+----------+--------+---------------+--------------+-----------------
   reg     |   0 | 64 |     0 |        1 |      1 | ok            | ok           | ok
   reg     |   1 | 64 |    16 |       17 |     17 | ok            | ok           | ok
   shrink  |   0 | 58 |     0 |        1 |      1 | ok            | ok           | ok
   shrink  |   1 | 58 |    13 |       14 |     14 | ok            | ok           | ok
```

- **`raw` 复现 §33 的边界**：`scramble=1` 让「层号升序」不再是拓扑序，`B_raw=16/13`，
  **`CAP*=G*=B+1`**（17/14）；`scramble=0` 时 `B=0`、`CAP*=1`。精扫（`harness/…out.txt`）：`reg,scr=1`
  在 **`CAP=1` 死锁（守门狗 ~50.7 ms 后返回）、`CAP=17` 才活**。
- **`scan` 靠「就绪才准入」把 `B` 归零**，任意 `CAP=1` 即活（与 §33 一致）。
- **`topo` 恒 `B=0`**，`CAP=1` 即活（无需扫描，但需预计算拓扑序）。
- **`event` 更强：`CAP=1` **且** `G=1` 即活**——不仅池容量下限是 1，**worker 数下限也是 1**
  （ready 队列只装就绪 task，从不把阻塞 task 塞给 worker）。

### 34.5 Section C：更大 `N` 的准入代价标度（补 §33.8 的 `N≤70`）

`CAP=G` 富余、`scramble=0`，`N` 由 64 到 ~4096：

```
   shape   | L  | W0   | N    | scan ms | topo ms | event ms | scan/event | topo/event
   --------+----+------+------+---------+---------+----------+------------+-----------
   reg     |  4 |   16 |   64 |   0.336 |   0.200 |    0.205 |       1.64 |       0.97
   reg     |  4 |   64 |  256 |   0.720 |   0.233 |    0.238 |       3.02 |       0.98
   reg     |  4 |  256 | 1024 |   1.125 |   0.831 |    0.876 |       1.29 |       0.95
   reg     |  4 | 1024 | 4096 |   4.281 |   3.196 |    3.344 |       1.28 |       0.96
   shrink  |  4 | 1024 | 4090 |   4.283 |   3.191 |    3.347 |       1.28 |       0.95
```

- **`scan` 的扫描代价随 `N` 增长**（相对 `event` 1.3–3.0×，抖动源于「多少 task 同时就绪」），
  **`topo` 与 `event` 基本持平**（`topo/event ≈ 0.95–0.98×`）——两者都不做「扫描找就绪」。
- 大 `N` 区间（`N≥1024`）三者的差主要在**准入/取 task 的簿记**，与 `N` 同阶；`event` 因「完成者
  顺手 push」把准入开销**摊到计算路径**上，是三者里设备侧最省的（但要多一次 `atomicSub`/后继遍历，
  见 §34.7 的高扇入代价）。
- 规则 vs 不规则（`reg`/`shrink`）在 `N` 相同时曲线一致——`event`/`topo` 与图几何无关。

### 34.6 Section D：独立扫 `G`（补 §33.8 的「未独立扫 G」）

不规则 `shrink, scramble=1`，`G` 富余时扫 `CAP`、`CAP` 富余时独立扫 `G`：

```
   N  | B(pi) | raw CAP* | raw G* | scan ok@CAP=1 | event G*(@CAP=1) | event ms@G=1 | event ms@G=64
   ---+-------+----------+--------+---------------+------------------+--------------+---------------
   26 |     5 |        6 |      6 | ok            |                1 |        1.262 |         0.200
   122 |    29 |       30 |     30 | ok            |                1 |        5.889 |         0.211
   506 |   125 |      126 |    126 | ok            |                1 |       24.335 |         0.256
```

- **`raw` 的 `(CAP,G)` 联合约束**（§32/§33 结论）在独立扫 `G` 下再次成立：`raw CAP* = raw G* = B+1`
  （6/30/126）。精扫（`N=122, B=29`）：**`G=28/29` 都死锁、`G=30` 才活**——**必须先让 `G` 超过
  「阻塞数」`B`**，否则 worker 全被阻塞 task 占满。
- **`scan` 在 `CAP=1` 恒活**（无需 `G>B`）。
- **`event` 的 `G*=1`**：单 worker 也能跑通（`event ms@G=1` 列），把 `G` 抬到 64 后延迟降 6–95×
  （`N=506`：24.3→0.26 ms，纯并行度）。

**关键增量**：§33.8 只验证了 `G*` 与 `CAP*` **同行相等**，本节把 `G` **独立**扫出来并确认：
对 `raw`，两者都精确等于 `B(π)+1`；而对 **`event`，`G*` 恒为 1**——**有界池死锁问题在事件驱动机制
下根本不存在**（池里永远没有「等前驱」的 task）。

### 34.7 Section E：扇入 `FIN` 敏感

`reg, N=256, CAP=G` 富余，`FIN ∈ {2,4,8,16}`：

```
   FIN | B(order)| scan ms | topo ms | event ms | scan/event | topo/event
   ----+--------+---------+---------+----------+------------+-----------
     2 |      0 |   0.576 |   0.239 |    0.239 |       2.41 |       1.00
     4 |      0 |   0.432 |   0.237 |    0.240 |       1.80 |       0.99
     8 |      0 |   0.588 |   0.241 |    0.248 |       2.37 |       0.97
    16 |      0 |   0.923 |   0.248 |    0.261 |       3.53 |       0.95
```

- **高扇入抬高 `scan` 的扫描代价**：`FIN=2→16` 时 `scan/event` 2.4→**3.5×**——扇入越大，
  「同时就绪」的 task 越多（层更宽），scheduler 每轮要扫更远；`scan` 的时序抖动也更大。
- **`event` 的 push 代价近持平**（`event ms` 0.239→0.261，`topo` 同步微增）：每个 task 完成要遍历
  `succ` 并 `atomicSub`，扇入越大后继集合越大，但这笔账摊在计算路径上、且 `N` 固定时总量与 `FIN` 同阶，
  相对扫描的 `O(剩余 order)` 更平滑。

### 34.8 与 MPK / DeepGEMM、判据的关系

- **`event` 模式 = MPK §5.1 的最小设备侧复刻**（快照行 1175–1231）：`dep_left` 是 event 的
  前置计数、`atomicSub` 归零是「event activated」、「seed 源任务」是 `start` event。它**同时**拿到
  §31–§33 三条正解的全部好处：`B≡0`（像 `topo`/`scan`）、**无预排序**（不像 `topo`）、
  **无扫描**（不像 `scan`）、且 **`G*=1`**（比两者更省资源）。
- **对 DeepGEMM Mega MoE 的解释**：它的 in-kernel scheduler 用 `get_num_l1_warmup_waves()`
  （`scheduler/mega_moe.cuh:15-45`，*"ensure no L1 -> L2 deadlock"*）与 `get_num_max_live_pool_blocks()`
  （`:47-80`）保证有界池不饿死——这是**静态化**的防死锁（预排 warmup wave），代价是要闭式推导
  ring 容量与 wave 数。MPK 的**动态事件驱动**（每 task 一个 event 计数器 + 完成即触发）从机制上
  免掉这笔推导：**池只需装就绪 task、`G` 只需 1**。两者是「静态编译期算容量」vs「动态运行期触发」
  的同一问题的两种解。
- **对 §33 的收口**：`raw` 边界 `B(π)+1` 是「按序把尚不可运行的 task 也准入」的代价；`scan`/`topo`/`event`
  是三条 `B≡0` 的路径，差别只在**谁、何时、以何代价**把就绪 task 送进池——`scan` 用扫描、
  `topo` 用预计算、`event` 用**依赖计数的天然触发**。MPK 选最后一条。
- **判据（C1–C5）**：四机制都是**单常驻 kernel**、scheduler/worker 角色分离、设备侧推进（C1–C4）；
  `event` 里 worker 无固定 task 角色、由依赖事件动态入队，最接近 C5，但本节仍把 scheduler 角色
  留在 block 0（`scan`/`topo`/`raw` 用它准入），故**与 §31–§33 同档、仍差 C5 一步**（C5 的完整
  worker/scheduler 去中心化见 P15-1 §9 / `docs/06` §3）。本节是 megakernel 的**正确性/容量成本侧**。

### 34.9 局限与踩到的坑

- **`scan` 计时抖动大**：`scan` 的每轮「扫剩余 order」对「多少 task 同时就绪」敏感（§C/§E），
  同配置多次可差 2–3×（本实现 `§E` 取 best-of-10）。故 `scan/event` 的绝对倍数只在同一次运行内可比，
  **趋势（随 `N`/`FIN` 增大）稳定**，绝对值不可直接外推。
- **`event` 的 ready 队列无界**：本实现按 `N+4` 开数组（每个 task 恰好 push 一次）。真实实现要
  环形缓冲 + 容量上界（= DeepGEMM `get_num_max_live_pool_blocks` 的职责）；本实验刻意让 `event`
  免于容量约束，以测量「机制本身」的代价。
- **`raw` 的死锁 case 靠守门狗返回**（~50 ms），故 `deadlock=1 done=0` 的行时间是守门狗耗时、非真实进度。
- **仍是合成依赖模型**：`N≤4096`、`scramble` 仍是层号整体旋转（非任意图重编号）、正确性只查 `sum`
  （float 量级 `rel=0.0e+00`），未逐 task 比对；未与真实 DeepGEMM SM100 路径对跑（SM100 专属）。
  目标卡 H100/sm90、device 3（固定空闲）。

## 36. in-kernel 队列的「发布内存序」：不 release 会不会读错？release/acquire 值多少代价？（P47-1）

> 本节接续 §35（P46-1）。§35 把事件驱动调度的 ready 队列从「无界」装进「有界环形缓冲」，但 §35.7
> 明确留了一条尾巴：**入队是「先 `atomicAdd`/`CAS` 推 `tail`、后写数据」（无 release 发布），理论上
> 消费者可能在数据可见前读到槽；真实 MPK 用 release/acquire 把它做实**——可作后续实验。本节把它补上。
>
> 这不是合成实现独有的写法。**MPK**（mirage 快照 `f9eb70c`）的队列协议是**明确配对**的
> 「**`st.relaxed.gpu` 写数据 + `atom.add/.cas.release.gpu` 推游标 / `ld.acquire.gpu` 读游标 +
> `ld.relaxed.gpu` 读数据**」：内存序原语集中在 `mpk_atoms.cuh:17-96`
> （`atom.add.release.gpu.u64:26`、`atom.cas.release.gpu.b64:37`、`ld.acquire.gpu.u64:49`、
> `ld.acquire.sys.u64:56`、`ld.relaxed.gpu.u64:66`、`st.relaxed.gpu.u64:73`）；worker 入队
> `st.relaxed.gpu` 写事件索引（`persistent_kernel.cuh:1215-1218`）后用 `atom.cas.release.gpu` 放行读游标
> （`:1222-1228`）；scheduler 出队 `ld.acquire.gpu` 读 `sched_queue_last_ready_event_id`（`:1340`）再
> `ld.relaxed.gpu` 取 event_id（`:1357-1360`）；每次派发同样是「`st.relaxed.gpu` 写 worker 队列槽 →
> `atom.add.release.gpu` 放行游标」（`:1495-1502`）。**DeepGEMM** 的 ring barrier 亦同（`
> red.async.inc.release` + 无符号距离 wait）。而本专题 §31–§35 的合成队列是
> **「先推 `tail`、后写 payload」**（`src/bounded_event_ring.cu` 的 `unb`：`slot = atomicAdd(tail,1);
> pool[slot]=u; __threadfence();`；`bnd` 同理）——**程序序本身就是发布在前、写在后的**。
>
> 所以本节要回答三件事：**（Q1）sm90 的弱内存模型下，这个「发布先于写」到底会不会真的读错？
> （Q2）把语义写成 `release/acquire` 值多少代价？（Q3）MPK 为什么要拆成「relaxed 写 + release 推游标」？**
>
> 源码 `src/pub_order_ring.cu`，脚本 `harness/pub_order_ring.sh`，原始输出
> `harness/pub_order_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`__launch_bounds__(32)`、
> litmus 用 2 block×1 线程、环用 block0=生产者/其余=消费者；所有等待循环带 `clock64` 守门狗，设备侧不挂）。

### 36.1 机制：litmus 与 in-kernel 环形队列

**§A（litmus）**：GPU 弱内存的经典试纸，验证「本机的重排到底可观测在哪一类」。两条：

- **message-passing（MP，store→store）**：`P: data=1; flag=1` / `Q: while(flag==0); r1=data`（都 relaxed）。
  `r1==0`（即 `flag` 已置位却没看到 `data`）就是「发布先于写」要担心的那种重排。控制通道
  （`setup`/`consumed`）两模式都带 `__threadfence()`，**只操控 `data→flag` 通道**；`mode1` 在
  `data` 写后、`flag` 写前加 release fence，在 `flag` 读后加 acquire fence。两线程分处两 block（必然跨 SM）。
- **store-buffering（SB，store→load）**：`P: x=1; r1=y` / `Q: y=1; r2=x`（都 relaxed），观测
  `r1==0 && r2==0`。**正对照**：若它可观测、而 MP 不可观测，说明 sm90 保序的是 store→store、
  允许 store→load 绕过（经典 GPU 弱内存特征）。注意**同步只放在读之后**——若在写与读之间放一根
  带原子的屏障，它会顺带 flush store，把窗口抹掉（本实现初版即因此测出 0，已修正）。

**§B（环）**：单常驻 kernel，`block0`=生产者（1 线程），其余 `G-1` block=消费者（各 1 线程），全局环形缓冲
`CAP` 槽 × `PL` 字 payload，`payload[s][i]=mix(s,i)` 可逐字校验。生产者按模式「写 payload / 推 tail」，
用**已完全消费计数**（读+校验完才 `atomicAdd(consumed,1)`）做容量背压（`tail - consumed < CAP`）——
注意不能拿 `head`（抢槽即自增）当背压，否则生产者会在消费者读完前覆盖槽（本实现初版即踩此坑，
换来 19995/20000 的假阳性，已修）。消费者 `atomicCAS(head)` 抢槽、逐字校验，任何一字不符记一次 `bad`。
五种发布序：

| 模式 | 生产者程序序 | fence | 说明 |
|---|---|---|---|
| `wro` | 写 payload → 推 `tail` | 无 | **正确程序序**的最小实现 |
| `rel` | 写 payload → **`__threadfence()`** → 推 `tail` | prod release / cons acquire | **MPK 协议** |
| `pub` | 推 `tail` → 写 payload | 无 | **§35/§34 的真实写法**（发布在前） |
| `half` | 写半个 payload → 推 `tail` → 写另一半 | 无 | 更宽的「发布先于写」 |
| `pubslp` | 推 `tail` → 空转 ~2.2 µs → 写 payload | 无 | **正对照**（人为拉长窗口，检测器应抓错） |

### 36.2 Section A：sm90 保序 store→store，但允许 store→load

`harness/pub_order_ring.out.txt` 行 4–12：

- **MP litmus relaxed：`lost=0 / 20M`（×2）+ `0 / 10M`（×3，连跑）= 0 / 70M**；release/acquire 也 0。
  即 **sm90 上 `data=1; flag=1`（都 relaxed）从未被观测到「`flag` 置位却看不到 `data`」**——store→store
  在硬件上保序（`wro` 的「写 payload→推 tail」正是这一类）。
- **SB litmus relaxed：`both-zero = 5M/5M`（×2）= 100%**——**store→load 重排则完全可观测**
  （每个线程的读绕过了对方刚发出的写）。这**证明了试纸有效**：本机弱内存确实会重排，不是「怎么测都是 0」。
  两条合起来给出本机的序画像：**store→store 保序、store→load 可绕过**。

代价侧（行 32 vs 31、行 4 vs 6 的 cycle）也顺带量到：MP **relaxed 1497.5 ns/trial** vs **rel+acq 2226 ns/trial**，
每 trial 一对 `__threadfence()` 约 +728 ns；SB 593.9 ns/trial。

### 36.3 Section B：环的发布序——`pubslp` 必错、`pub`/`half`/`wro`/`rel` 在本机实测 0 错

`harness/pub_order_ring.out.txt` 行 15–27：

```
   ring pubslp NS=20000  CAP=64  PL=256  G=34  bad=20000 / 20000   （正对照：应错）
   ring pubslp NS=2000   CAP=4   PL=8192 G=2   bad=2000  / 2000
   ring pub    NS=200000 CAP=256 PL=256  G=34  bad=0 / 200000      （§35 真实写法：发布在前，无 fence）
   ring pub    NS=20000  CAP=4   PL=8192 G=2   bad=0 / 20000
   ring half   NS=200000 CAP=256 PL=256  G=34  bad=0 / 200000
   ring wro    NS=200000 CAP=256 PL=256  G=34  bad=0 / 200000      （正确程序序，无 fence）
   ring rel    NS=200000 CAP=256 PL=256  G=34  bad=0 / 200000      （MPK release+acquire）
   ring wro    NS=20000  CAP=4   PL=8192 G=2   bad=0 / 20000
   ring rel    NS=20000  CAP=4   PL=8192 G=2   bad=0 / 20000
```

三件事：

1. **检测器有效**：`pubslp`（强行拉长「发布→写」窗口 ~2.2 µs）**20000/20000、2000/2000 全错**——
   一旦消费者在 payload 落盘前读到槽，逐字校验立刻抓到。故「0 错」不等于「检测器瞎」。
2. **「发布先于写」在本机自然时序下读不出错**：`pub`（= §35/§34 的 `slot=atomicAdd(tail,1); pool[slot]=u;`
   真实程序序）与 `half` 在 `PL=256`/`PL=8192`、百万级槽上**全 0 错**。原因与 §36.2 一致：
   producer 的 payload store 在消费者「轮询 `tail` → CAS 抢槽 → 读槽」的往返（几百 ns 起）内早已落盘，
   producer 总是抢先——所以 §34/§35 **「没出错」是时序侥幸、不是形式保证**（§35.7 的判断正确）。
3. **正确程序序（`wro`）与 MPK 协议（`rel`）本机实测全 0 错**：`wro` 无 fence 也全对，
   因为 sm90 保序 store→store（§36.2），payload 先写、`tail` 后推，消费者只会看到「payload 已就绪」的槽。

### 36.4 Section C：release/acquire 的代价 ≈ +0.36 µs/槽（几乎恒定）

`harness/pub_order_ring.out.txt` 行 29–36。比较 `wro` 与 `rel`（差额即生产者一次 release + 消费者一次 acquire）：

| payload | G | `wro` ns/槽 | `rel` ns/槽 | 差 | 倍率 |
|---|---|---|---|---|---|
| `PL=1` | 2 | 1120.5 | 1492.6 | **+372.1** | 1.33× |
| `PL=1` | 34 | 385.0 | 740.1 | **+355.1** | 1.92× |
| `PL=256` | 34 | 6301.2 | 6663.2 | **+362.1** | 1.06× |

**每槽 release+acquire 的代价 ≈ 355–372 ns，几乎与 payload 大小、worker 数无关**（是 fence 的固定成本）：
在**小 payload**（`PL=1`、队列机制本身占主导）时它是 1.3–1.9× 的显著开销；在 **256 字 payload**（≈1 KB）时
被访存摊薄到 **1.06×**。这解释了 MPK 为什么把协议拆成「`st.relaxed.gpu` 写数据 + **单条**
`atom.cas.release.gpu` 推游标」而不是全程 fence——**每队列操作只付一次 release（prod）+ 一次 acquire（cons）**，
且靠「批量取 task」（`docs/04` §15 的 K≈W/G）、环形缓冲摊薄到可忽略。

### 36.5 与 §31–§35 / MPK / DeepGEMM 的关系、判据

- **对 §35.7 的收口**：§35/§34 的队列是**发布先于写**（程序序错误），在本机「没出事」纯属时序侥幸；
  只要窗口被拉开（`pubslp`）就 100% 读错。**修法有两条**：(a) 把程序序倒过来——先写 payload 再推游标
  （`wro`；sm90 保序 store→store，连 fence 都不必，但非形式保证）；(b) 写 payload 后 release 推游标、
  读游标后 acquire 读 payload（`rel`，MPK 做法，形式保证），代价 ~0.36 µs/槽（大 payload 摊薄后 ~1.06×）。
- **对 MPK**：`mpk_atoms.cuh:17-96` 的六条原语正是 `rel` 的工程化——不是「更保守」，而是**唯一**
  在任意 payload/任意消费者速度下都成立的写法；`docs/06` §5.3 的「relaxed 写数据 + release 推游标 /
  acquire 读游标 + relaxed 读数据」与本节 `rel` 逐字对应。**这也解释了 §35.7 的注脚不是杞人忧天**：
  生产实现必须写 release，合成实验可以侥幸。
- **对 DeepGEMM**：ring barrier 用 `red.async.inc.release` + 无符号距离拒绝陈旧标签
  （`docs/04` §23 引 `scheduler/mega_gate.cuh:13-39`），把「release 发布 + 单调序号校验」做到
  单条 PTX，是本节「fence 是固定成本、要靠单条原子吸收」的同一条工程经验。
- **判据（C1–C5）**：litmus 与环都与「几个常驻 kernel / 设备内协调」正交，是队列**内存正确性**的
  地基；环本身是单常驻 kernel、设备侧推进（C1–C4），无固定角色 worker/scheduler 动态派发，
  **仍差 C5 一步**（与 §31–§35 同档）。

### 36.6 局限与踩到的坑

- **litmus 的窗口**：MP/SB 都是「2 block×1 线程」的最小形态，未扫 SM 距离、未与 `ncu` 的内存序事件对拍；
  SB 的 100% 是「读紧跟写、对方写尚未广播」的必然结果，**只说明重排可观测、不给出概率分布的真实意义**。
  MP 的 0/70M 是「未观测到」，不是「不可能」（GPU 弱内存模型不保证 store→store 保序；本机实现恰好保序）。
- **环仍是合成模型**：payload 是 `mix(s,i)` 纯写、消费者逐字校验；未做真实 task 描述符、未扫多生产者
  （MPSC）、未与 DeepGEMM SM100 路径对跑。`pubslp` 的 2.2 µs 窗口是**人为**构造的最坏情形，
  用于证明检测器有效，不代表生产时序。
- **初版两个 bug（已修）**：① 背压用 `head`（抢槽即自增）而非「已消费」计数，导致生产者覆盖消费者
  正在读的槽，所有模式假阳性 19995/20000；② SB litmus 在写与读之间放了带原子的屏障，屏障顺带 flush store、
  把重排窗口抹掉，测出 0。两坑都记在此，提醒「0 错」先要排查是不是试纸坏了。
- **口径**：时钟换算用 H100 SM ~1.8 GHz（`ns = cycle / 1.8`），是近似口径；不同配置在不同空闲 device
  上可能有 ±几 % 抖动。目标卡 H100/sm90、device 3（固定空闲）。
