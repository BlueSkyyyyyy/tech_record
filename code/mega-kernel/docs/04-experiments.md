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
  `mbarrier`/TMA 组合，留作后续。

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
  `SM100_TMA_2SM_LOAD`（DeepGEMM sm100 路径用的是后者）。留作后续。
