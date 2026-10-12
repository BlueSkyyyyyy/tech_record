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
