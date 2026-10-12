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
