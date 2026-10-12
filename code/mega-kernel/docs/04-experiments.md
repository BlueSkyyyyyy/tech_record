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
