# KDA 前向性能实测：FlashKDA vs FLA `chunk_kda`

> 本文是「Kimi Delta Attention（KDA）前向 / 反向算子实现解析」的第 7 篇。前六篇把 KDA 的
> 数学、FlashKDA 前向 K1/K2 精读、数值对拍、反向推导与参照实现对清楚了。本篇（P7-1）只做
> 一件可验证的事：**在同一块 H100 上，用统一口径把 FlashKDA 前向和 FLA `chunk_kda`
> （Triton）的前向计时量出来**，给出端到端 wall 时间、纯 GPU（CUPTI/device）时间、
> kernel 级分解，并解释两者差异**来自哪里**。所有数字来自本机实测，原始输出见
> `harness/bench_fwd.out.txt`。

## 1. 计时口径与一个必须避开的坑

### 1.1 三个口径

同一份输入上分别调用两个实现，各测三个量：

| 口径 | 含义 | 采集方式 |
|---|---|---|
| `wall` | 单次调用端到端耗时，**含 host 派发**（Python/launch/`torch.empty` 分配） | 一次 CUDA event 对包住 `REPS=50` 次调用求均值，重复 3 轮取中位数 |
| `dev`  | **纯 GPU kernel 时间**（不含 host 派发、不含调度空隙） | `torch.profiler`（CUPTI）CUDA-activity，`PROF_REPS=20` 次求和除以次数 |
| `TF/s` | 用解析 FLOPs 模型算的有效算力 | 见 §1.3 |

### 1.2 坑：连续 event 对的「单次中位数」会读到一半

第一版我按最常见的写法——每次调用前后各插一个 event，然后取 `median(elapsed_time)`——
拿到的 FlashKDA 时间只有 CUPTI 的**一半**（T=1024 H=4：wall 115µs vs dev 220µs），一度
以为是 CUPTI 偏大。用 `torch.mm(4096³)` 做了对照才定位：**背靠背的相邻 event 对时间戳会
交叠**，`median` 读出来约是真值的一半，而 `sum/N`（即均值）才和 CUPTI 一致
（`mm` per-pair `sum=10233µs/30=341µs`，CUPTI/总时长 `339µs`）。

```
mm per-pair median us 174.38   sum us 10233.25
mm total/call us 339.57          # median 只有一半，sum/N 才对
```

本脚本因此改成**一个 event 对包住 N 次调用再除以 N**（`bench_fwd.py:time_flash`），
并用 `CUDA_VISIBLE_DEVICES` 隔离到一张空闲卡（8 卡机器上 GPU0 常被占满，实测前先
`nvidia-smi` 确认）。

### 1.3 解析 FLOPs / bytes 模型

KDA 前向是「分块 delta rule」，矩阵乘占绝对主导。每个 chunk（CHUNK=16，块数 $C=\lceil T/16\rceil$，
并行度 $n=B\cdot H\cdot C$）的乘加（按 2 FLOP/MAC 计）：

```
K1:  L = k_dec·k^T              2·D·C_h·C_h        （C_h=16）
     INV 块内求逆                O(C_h³)，可忽略（~0.27 MFLOP/chunk）
K2:  q@s, k@s（各 D·C_h·D）     2 · 2·D·C_h·D
     U = INV·U                2·C_h·C_h·C_h
     Mqk·U                    2·C_h·C_h·C_h
     s += k_rest^T·U          2·D·C_h·D
per_chunk ≈ 4·D·C_h·C_h + 4·D·C_h·D + 2·D·C_h·D
FLOPs = n · per_chunk
```

DRAM 下界（bf16=2B）：读 `q,k,v,g`（各 D）+ `beta`（1），写 `o`（D）：

```
bytes = (B·T·H·(4D+1) + B·T·H·D) · 2
```

这是**下界**：实际还有 workspace 读写、初始/终态、以及 K1→K2 间的中间量落盘。所以报告里的
`GB/s` 是「有效带宽下界比」，`TF/s` 是「按此模型的算力」，用于横向比较而非绝对上界。

环境：H100 80GB HBM3，`torch 2.12.0a0`（NVIDIA 版），FLA `0.5.2`，`triton 3.6`，
`bf16`，`D=128`，`CHUNK=16`（FLA `chunk_size=64`），两实现吃同一份输入、启用各自的
in-kernel gate / l2norm / beta-sigmoid。

## 2. 端到端结果

`spd(wall)=fla_wall/fk_wall`、`spd(dev)=fla_dev/fk_dev`，**大于 1 表示 FlashKDA 更快**。

| case | B | T | H | fk wall(µs) | fk dev(µs) | fla wall(µs) | fla dev(µs) | fk TF/s | spd(wall) | spd(dev) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| T=128 | 1 | 128 | 4 | 36.6 | 25.1 | 779.7 | 57.8 | 2170 | **21.3×** | 2.30 |
| T=256 | 1 | 256 | 4 | 47.8 | 37.2 | 846.0 | 62.6 | 2932 | 17.7× | 1.68 |
| T=512 | 1 | 512 | 4 | 65.9 | 61.5 | 910.0 | 68.5 | 3548 | 13.8× | 1.11 |
| T=1024 | 1 | 1024 | 4 | 112.8 | 108.7 | 883.5 | 81.4 | 4013 | 7.8× | 0.75 |
| T=2048 | 1 | 2048 | 4 | 208.7 | 204.8 | 880.2 | 122.9 | 4260 | 4.2× | 0.60 |
| T=4096 | 1 | 4096 | 4 | 407.3 | 402.8 | 860.8 | 199.5 | 4332 | 2.1× | 0.50 |
| H=1 | 1 | 1024 | 1 | 109.0 | 106.2 | 877.2 | 78.5 | 1027 | 8.0× | 0.74 |
| H=2 | 1 | 1024 | 2 | 112.7 | 108.7 | 887.6 | 77.2 | 2006 | 7.9× | 0.71 |
| H=4 | 1 | 1024 | 4 | 112.8 | 108.8 | 883.0 | 81.4 | 4009 | 7.8× | 0.75 |
| H=8 | 1 | 1024 | 8 | 117.5 | 113.3 | 901.7 | 101.6 | 7698 | 7.7× | 0.90 |
| H=16 | 1 | 1024 | 16 | 130.0 | 126.0 | 865.7 | 117.5 | 13849 | 6.7× | 0.93 |
| B=2 | 2 | 1024 | 4 | 117.5 | 113.3 | 879.6 | 101.5 | 7703 | 7.5× | 0.90 |
| B=4 | 4 | 1024 | 4 | 129.2 | 125.3 | 875.5 | 132.8 | 13929 | 6.8× | 1.06 |
| B=8 | 8 | 1024 | 4 | 151.0 | 145.2 | 927.8 | 211.2 | 24034 | 6.1× | 1.45 |
| T=1023 | 1 | 1023 | 4 | 118.6 | 114.6 | 889.4 | 81.8 | 3805 | 7.5× | 0.71 |
| big B8 T2048 H8 | 8 | 2048 | 8 | 337.4 | 333.9 | 886.7 | 741.3 | 41799 | 2.6× | 2.22 |
| T=8192 H=16 | 1 | 8192 | 16 | 915.3 | 903.7 | 897.4 | 717.0 | 15446 | **0.98×** | 0.79 |
| T=8192 H=64 | 1 | 8192 | 64 | 1261.9 | 1261.2 | 2327.9 | 2288.9 | 44271 | 1.84× | 1.81 |

### 2.1 观察一：FLA 的 wall 被 **host 派发**钉死在 ~0.86 ms

最刺眼的是 FLA 一列：从 `T=128` 到 `B=8`，无论 workload 多大，**wall 几乎恒为
0.86–0.93 ms**，而它同期的 `dev` 只有 58–211 µs。也就是 **FLA 每次调用有 ~0.7–0.8 ms 的
host 开销**——`chunk_kda` 的纯 Python 包装把前向拆成十几个独立的 torch/triton 调用
（§3 的 breakdown 可见 `l2norm`、`gate_cumsum`、`intra`、`inter_solve`、`recompute_w_u`、
`h`、`o` 等各是一次 launch），host 端不具备把 GPU 喂满的能力，于是**在中小 shape 上
FLA 是 host-bound，wall 不随 workload 增长**。

FlashKDA 反过来：它只有 **1 个 host 调用**（`flash_kda.fwd` → C++ 里 2 次 kernel launch +
一次 `torch.empty` workspace），所以 `wall ≈ dev`（T=1024：112.8 vs 108.7；T=8192 H=64：
1261.9 vs 1261.2，差 ~0.1%）。**它的成本几乎全是 GPU 工作，没有可观测的 host 税。**

结论：在中小 shape 上 FlashKDA 的 wall 领先（up to 21×）**主要是把 FLA 的 host 派发税
省掉了**，这部分收益与 GPU 快慢无关，是「算子融合 + 单入口」的工程收益。

### 2.2 观察二：纯 GPU 时间（dev）上 FLA 在中等 shape 反而更小

`spd(dev)<1`（T=512…4096、H≤8）说明**单看 GPU 工作，FLA 的 Triton 前向比 FlashKDA 的
CUDA 更小**（如 T=2048 H=4：FLA 122.9µs < Flash 204.8µs）。这并不矛盾，见 §3 的分解：

* FLA 的 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（状态递推）**分块沿 T 并行**，
  并行度是 `B·H·(T/64)`；
* FlashKDA 的 K2 `flash_kda_fwd_recurrence` **沿 T 串行**（每个 `(b,h)` 一个 CTA，顺序扫过
  `T/16` 个 chunk），并行度只有 `B·H`。

于是 FlashKDA 的 K2 时间是 $\propto T$ 的直线（§2.4 有数据），在 `B·H` 不够大时会输给
FLA 的高并行 h-kernel。**但 FLA 赢的这点 dev 时间被它自己的 0.8ms host 税整体吃掉**，
wall 上仍然是 FlashKDA 快。这也解释了上游 `BENCHMARK_H20.md` 为什么选 `T=8192,H=64/96`
作设计点：只有到那个规模，FLA 才 GPU-bound（host 税被摊薄），FlashKDA 的融合优势才在
wall 上以 1.8–2.3× 的形式显出来。

### 2.3 观察三：两个不同的「谁更快」区间

把上表按规模分三段，结论很清楚：

1. **小/中 shape（`B·H` 小、T 中等）**：FlashKDA wall 大胜（2×–21×），但**赢在 host**；
   纯 GPU 工作 FLA 更小。
2. **大 `B·H`（`big B8 T2048 H8`）**：两边都 GPU-bound；FlashKDA dev 333.9µs vs FLA
   741.3µs，**dev 与 wall 同时 2.2×/2.6× 领先**——这时 FlashKDA 的融合是真的 GPU 收益。
3. **大 T 但小 `B·H`（`T=8192 H=16`）**：FlashKDA dev 903.7µs ≈ FLA 717.0µs，wall 打平
   （**0.98×**）。K2 只有 16 个 CTA，沿 T 的串行链是瓶颈，FlashKDA 的并行度不足。
   H 升到 64 后（`T=8192 H=64`）FlashKDA 反超到 1.84×。

### 2.4 K2 沿 T 串行：$\propto T$ 的定量证据

固定 `B=1,H=4`，FlashKDA 的 dev 时间随 T 精确线性：

| T | dev(µs) | dev/T (ns/token) |
|--:|--:|--:|
| 1024 | 108.7 | 106 |
| 2048 | 204.8 | 100 |
| 4096 | 402.8 | 98 |

斜率 ~100 ns/token 基本恒定，印证 K2 recurrence 沿时间串行、无跨 chunk 并行。同时固定
`T=1024` 扫 `H=1→16`，dev 只从 106 涨到 126 µs——`H≤4` 时 K2 的 `H` 个 CTA 都填不满
GPU，加 CTA 几乎不加时间，**说明此区间是 per-CTA 的串行延迟受限，而不是吞吐受限**。
这正是 §2.3 第 3 段的根因。

## 3. kernel 级分解

用 CUPTI 抓每次调用里各 kernel 的 device 时间（`harness/bench_fwd.out.txt` 顶部）。

### 3.1 小 shape（T=1024, H=4）

```
FlashKDA                               FLA chunk_kda
 96.6us  flash_kda_fwd_recurrence (K2)   27.9us  chunk_kda_fwd_kernel_inter_solve_fused
  9.0us  flash_kda_fwd_prepare    (K1)   23.3us  chunk_gated_delta_rule_fwd_kernel_h_blockdim64
  2.0us  elementwise (workspace 初始化)    8.2us  chunk_kda_fwd_kernel_intra_token_parallel
                                          5.9us  recompute_w_u_fwd_kda_kernel
                                          4.1us  chunk_gla_fwd_kernel_o
                                          2.9us  l2norm_fwd_kernel (×2)
                                          2.7us  kda_gate_chunk_cumsum_vector_kernel
                                          2.3us  elementwise (×2) / 1.3us beta_sigmoid / 1.0us
```

FlashKDA 只有 **3 个 kernel**：K2 占 89%，K1 占 8%，还有一个初始化 workspace 的
elementwise。FLA 是 **10 个 kernel 的拼装**，最长单个 27.9µs，但**串起来 + host 派发**
导致 wall 远大于 FlashKDA。

### 3.2 大 shape（T=8192, H=64）——融合的收益一目了然

```
FlashKDA                                 FLA chunk_kda
 776.3us  flash_kda_fwd_recurrence (K2)   424.0us  chunk_kda_fwd_kernel_inter_solve_fused
 480.2us  flash_kda_fwd_prepare    (K1)   422.5us  chunk_kda_fwd_kernel_intra_token_parallel
   6.0us  elementwise                      357.9us  recompute_w_u_fwd_kda_kernel
                                           351.7us  chunk_gated_delta_rule_fwd_kernel_h_blockdim64
                                           339.7us  chunk_gla_fwd_kernel_o
                                           177.0us  l2norm_fwd_kernel (×2)
                                           147.1us  kda_gate_chunk_cumsum_vector_kernel
                                            62.9us  elementwise (×2) + 2.5us beta_sigmoid
```

* 单比状态递推：FlashKDA K2 = 776µs，FLA h-kernel = 352µs——**FLA 的这个 kernel 更快**
  （如前分析，它沿 T 并行，而 FlashKDA K2 沿 T 串行）。
* 但 FLA 要把中间量一遍遍落盘/重读（`intra` 422 + `inter_solve` 424 + `recompute_w_u`
  358 + `o` 340 + `l2norm` 177 + `cumsum` 147 + elementwise 63 ≈ **1.93 ms** 的非递推工作），
  而 FlashKDA 用 K1 的 **480µs** 一次性把 workspace 备好、K2 内部消化掉其余。
* 合计：FlashKDA 1262µs vs FLA 2289µs → **1.81×**。

这印证了 deep-dive 里 K1/K2 拆分的动机（`docs/01` §4）：**K1 把 token 并行的重活
（归一化/门控/衰减/块内求逆）一次算完并落盘，K2 专心做 head 并行的递推并复用 workspace，
避免 FLA 那套「多个独立 kernel 反复读写中间量」的带宽与 launch 开销**。前者在 wall 上
的净收益，正是 §2.3 三种区间的分界。

## 4. 小结与待办

* **口径**：本文所有 `dev` 来自 CUPTI（`torch.profiler` CUDA-activity），`wall` 用
  「一个 event 对包 N 次取均值」——已排除背靠背 event 对中位数减半的坑（§1.2）。
* **结论**：中小 shape 上 FlashKDA wall 领先（2×–21×），主因是**省掉 FLA 的 ~0.8ms host
  派发税**；大 `B·H` 时（如 B8 T2048 H8）FlashKDA 在 **wall 与 dev 同时**领先 ~2.2×，
  是真实 GPU 收益；大 T 小 `B·H`（T=8192 H=16）因 K2 沿 T 串行、并行度只有 `B·H` 而打平。
* **待验证**（留待后续）：
  - ~~FLA 的 host 税组成（Python 派发 vs Triton launch）尚未用 `nsys`/host 插桩拆开，
    只从「wall 恒定」推断为 host-bound~~——**已在 §9（P7-6）拆开**：固定 ~0.82–0.86 ms/次全在
    host，拆为 12 次 Triton kernel 的 Python launch 包装 + ~190 µs `dispatch` + ~75 µs driver launch；
  - FLOPs/bytes 模型是下界，未计 workspace 流量；
  - FlashKDA 的 `spd(dev)<1` 区间是否可通过加大 K2 的 T 维并行（如跨 chunk 扫描/两遍法）
    消除，属实现改进问题，非本专题范围。

## 5. ncu 剖析：K1/K2 到底 bound 在什么上（P7-2）

§3 只回答了「各 kernel 花多少时间」，没回答「**为什么**是这个时间、离硬件上限有多远」。本节用
Nsight Compute（`ncu`）对 K1/K2 逐 kernel 做一次 bound 分析，全部为本机实测；复现脚本
`harness/probe_ncu.sh`、原始输出 `harness/probe_ncu.out.txt`（每个 kernel 单独 profile，脚本把
超长的 CUTLASS demangled kernel 名裁掉、只留分节表）。

**方法学**：驱动 `harness/ncu_driver.py` 先 warmup 3 次再用 `ncu -k regex:... -c 1 --launch-skip 3`
只抓第 4 次调用的目标 kernel；`ncu` 会 replay 多次（本节指标 12–20 passes）。replay 会串行化并
插入 flush，故 ncu 报的 `Duration`（如 K2 T=1024 的 104.2µs）比 §2 的 CUPTI `dev`（108.7µs，同一
shape）略小，趋势一致。GPU 132 SM。

### 5.1 三个 shape 的关键指标

`tensor%`/`fma%` 是**每活跃 SM** 上对应 pipe 的 `pct_of_peak_sustained_active`；`L2%/DRAM%/L1TEX%`
取自 SOL 的 memory-pipe 利用率（`pct_of_peak_sustained_elapsed`，整卡口径）；`occ` 为
theoretical/achieved occupancy；两个 stall 是每发射一条指令的 stall 周期数。

| shape | kernel | grid | dur(µs) | tensor% | fma% | L2% | DRAM% | L1TEX% | occ(th/ach)% | barrier | wait |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| T=1024 H=4 | K1 prepare | 256 | 11.55 | 0.63 | 5.12 | 30.1 | 8.2 | 22.3 | 100/24.2 | **11.9** | 1.9 |
| T=1024 H=4 | K2 recurrence | 4 | 104.2 | 21.6 | 9.7 | 3.0 | 1.4 | 58.6 | 18.8/9.3 | 0.3 | **1.1** |
| T=8192 H=64 | K1 prepare | 32768 | 481.1 | 1.54 | 12.5 | **88.2** | 52.1 | 57.3 | 100/98.4 | **27.0** | 1.9 |
| T=8192 H=64 | K2 recurrence | 64 | 781.2 | 22.7 | 10.2 | 46.5 | 27.5 | 63.4 | 18.8/9.4 | 0.2 | **1.1** |
| B=4 T=1024 H=64 | K2 recurrence | 256 | 207.1 | 24.1 | 10.8 | **91.6** | 53.0 | 59.5 | 18.8/18.1 | 0.3 | **1.2** |

补充（smem bank conflict，ncu 归一到「占 shared wavefront 的比例」）：

| kernel / shape | shared-load 冲突占比 | shared-store 冲突占比 |
|---|--:|--:|
| K1 T=1024 H=4 | 45.0%（2.7-way） | 51.6%（2.3-way） |
| K1 T=8192 H=64 | 47.5%（2.7-way） | 54.4%（2.3-way） |
| K2 T=8192 H=64 | 9.1% | 44.9% |
| K2 B=4 T=1024 H=64 | 4.6% | 21.6% |

### 5.2 K1：TensorCore 几乎全闲，瓶颈是 L2 带宽 + barrier

**结论一：K1 的 TensorCore 通道全程 ≤1.6%，它不是矩阵乘瓶颈。** 即使 K1 里有
$L=k_{dec}k^\top$、$M_{qk}=q_{dec}k^\top$ 两个 16×16 的 bf16 GEMM 和求逆后的 16×16 HMMA 合并
（`fwd_kernel1.cuh:483/485/511`），这些只是 M=N=16 的小乘，FLOPs 被每 tile 的 q/k/g/beta 搬运与
workspace 落盘淹没。FMA pipe 也只有 5–12%（大部分是归一化/门控/decay 的逐元素标量）。

**结论二：放大到设计点后，K1 是 L2 带宽瓶颈。** T=8192 H=64 时 K1 的 SOL 最大 pipe 是 **L2 =
88.2%**（DRAM 52.1%，L1TEX 57.3%），此时它已是满占用（achieved occupancy 98.4%，waves/SM=31）。
DRAM 读 405MB + 写 434MB、L2 流量 1.97GB——**写 workspace（13824B/tile）和读 q/k/v/g/beta 的
流式流量**把 L2 打满。这与 §3.2「K1 一次算完落盘」的收益是一体两面：K1 省掉的是 launch/重复读写，
代价是把全部中间量落到 L2/DRAM，于是它自己成了 L2-bound。

**结论三：barrier stall 是 K1 的第二瓶颈。** 每发射一条指令 stall 在 CTA barrier 上的平均周期：
小 shape 11.9、大 shape **27.0**（`ncu` 提示这占两次发射间隔 43.5 周期的 **61%**）。来源是 K1 里
密集的 `__syncthreads()`——`fwd_kernel1.cuh:425/476/487/508/514` 一圈：读入寄存器 → 同步 → 覆写
union 复用后的 smem（q/k/g → k_decayed/q_decayed/k_inv → L → M）→ 同步 → 求逆 → 同步 → 落盘。
`SharedStorageK1` 的 union 复用（`fwd_kernel1.cuh:45–83`）省 shared memory 的代价，就是这些
「必须先同步再覆写」的串行点。此外 shared-load/store 的 **2.7-way/2.3-way bank conflict**（占
shared wavefront 45–54%）给 smem 访问再加约 5 成 wavefront。

**结论四：小 shape 是「grid 太小」而非算法问题。** T=1024 H=4 时 K1 grid 只有 256 个 CTA
（$BHC=1\cdot4\cdot64$），/132 SM ≈ 2 CTA/SM，achieved occupancy 仅 24.2%、`No Eligible` 85.3%，
纯延迟受限。`ncu` 同时报「global load 平均每 sector 只用 4/32 字节」（未合并，估 50% 局部加速空间，
具体是哪条 load 待验证）。

### 5.3 K2：不是 TensorCore bound；smem 限占用、L2 限吞吐

**结论一：K2 的 TensorCore 通道只有 22–24%（每活跃 SM），从来不是瓶颈。** 它是 K2 里最忙的
compute pipe 没错，但离打满还差 4 倍；FMA 仅 ~10%。K2 大头是 6 个 Phase（§3 Part II）里的
bf16 MMA + 大量 TMA/LDSM 搬运，属于**访存 + 依赖链**型。

**结论二：K2 的占用率被 shared memory 死死钉在 18.75%（2 CTA/SM）。** ncu LaunchStats：
Dynamic Shared Memory Per Block **98.43 KB**（就是 union 后的 `SharedStorageK2`，见 §P3-1 实测），
每 SM 最多放 2 个（第 3 个放不下，与 `Block Limit Shared Mem = 2` 一致）；理论 occupancy 18.75%。而 K2 grid = $B\cdot H$：设计点
T=8192 H=64 只有 **64 个 CTA < 132 SM**，achieved occupancy 仅 9.4%、`ncu` 直接报
「grid 64 blocks < 132 SMs，估 51.5% 局部加速空间」。这就是 §2.4「per-CTA 串行延迟受限」的硬件证据。

**结论三：一旦 grid 足以填满 GPU，K2 立刻变成 L2 带宽瓶颈。** 把 $B\cdot H$ 抬到 256（B=4,H=64，
T=1024）后 achieved occupancy 18.06% ≈ 理论值、waves/SM≈1，此时 SOL 最大 pipe 是 **L2 = 91.6%**
（DRAM 53.0%、L1TEX 59.5%），而 TensorCore 仍只有 24.1%。也就是说 K2 的稳态上限由
**每 chunk 从 L2 读 workspace（kd/qd/kr/gt/INV/Mqk，13824B/tile）** 决定——同一份 workspace
被 K2 沿 T 顺序读一遍，流量上不去就被 L2 卡住。（§6 的逐字节实测给出修正：workspace 约占 K2 读流量
的 **75%**，另有约 **22.5%** 来自 `v` 流——见 §6.2。）

**结论四：K2 的 stall 以 `wait` 为主（~1.1 周期/发射），不是 barrier/scoreboard。** 与 K1 相反，
K2 几乎不 `__syncthreads`（warp-specialized，只有跨 MMA warp 的 `NamedBarrier` 与流水线
barrier），所以 barrier stall 只有 0.2–0.3；主导的 `wait` 是**固定延迟依赖**（等待前一条
MMA/LDSM 的结果），加上 occupancy 太低、可切换 warp 不足（`Eligible Warps Per Scheduler` 0.38–0.50），
掩盖不住依赖延迟。

### 5.4 与 P7-1 结论的闭环

把 §5 和 §2/§3 对起来：

1. **K1/K2 都不是 TensorCore bound**（K1 ≤1.6%、K2 ≤24%），所以 §1.3 的 `TF/s` 只是「相对算力」
   而非上界；真正的上限是访存。
2. **K1 和 K2 在放大后都撞 L2**（K1 88.2%、K2 91.6%）。原因不同：K1 是「把中间量写进 L2」（融合的
   代价），K2 是「把中间量从 L2 读回来」（融合的兑现）。二者的 L2 流量都在 1.7–2.0GB 量级，
   与 P7-1 §3.2「FLA 非递推工作 ~1.93ms 全在读写的 workspace」互相印证——**KDA 前向的真正
   硬成本是 D×D 状态与 13824B/tile workspace 的搬运，不是乘法**。
3. **K2 的并行度只有 $B\cdot H$** 是 §2.4 的直接硬件根因：设计点 H=64 < 132 SM，占用率 9.4%，
   于是 dev ∝ T、`spd(dev)` 在 H≤16 时 <1；H=64 时 grid 才刚够、L2 又成新墙。
4. §2.1 的「FLA host 税」与本节的 GPU 结论不冲突：那部分收益在 host 端，本节量的是 GPU 端。

**待验证**：(a) K1「global load 4/32 字节/sector」具体落在哪条 load——**已在 §7 解决**（= `fwd_kernel1.cuh:258`
的 `A_log_ptr[head_idx]`，广播标量读、`Excessive=0`、DRAM 占比 <2%，属启发式误报）；
(b) K2 的 L2 流量里 workspace 读与 $o$/state 写各占多少（需 `nsys` 或按指针插桩）——**已在 §6 解决**；
(c) 把 K2 的 state 常驻寄存器/提高 smem 复用以突破 2 CTA/SM 后 L2 墙是否后移。

## 6. K1/K2 的访存账：谁贡献了那 1.7–2.0GB 的 L2 流量（P7-3）

§5 说「KDA 前向的硬成本是状态与 workspace 的搬运」，但那句里 workspace 读与 $o$/state 写各占多少、
K1 到底读了哪些张量，都是推断而非实测（§5 待验证 (b)）。本节把这块账**逐字节**算清：用 `ncu` 抓
DRAM 读写计数，再和闭式预测逐项对拍。复现脚本 `harness/check_traffic.py`、原始输出
`harness/check_traffic.out.txt`。

### 6.1 方法：为什么用 DRAM 计数而不是 L2 sector 计数

`ncu` 默认在每次 replay 前**清空 L2**（`--cache-control all`），所以一次 kernel 调用的
`dram__bytes_read/write` 就是它**唯一的、去重后的访存字节**——这正是「L2 流量」里真正来自外部的部分，
且不受 L1/TMA 重发、swizzle、cache 命中的干扰。相比之下 `lts__t_sectors_*` 对 TMA 有系统性放大：
实测在 K1 上 `lts` 读是 DRAM 读的 **1.5×**、`lts` 写是 workspace 的 **~3×**（同一份 workspace，
TMA store 被重复计数），所以 `lts` 计数只能当**同 kernel 内相对量**看，不能当字节数。故本节以 DRAM 为准。

预测模型（$D=128$，每 tile 的 workspace $=13824$ B，`utils.cuh:70–76`）：

- **K1**：读 `q,k,g,beta`（**不读 v**，K1 只用 q/k/g/beta 算 kd/qd/kr/gt/INV/Mqk）；写 workspace
  $=B\cdot H\cdot\lceil T/16\rceil\cdot 13824$。
- **K2**：读 workspace $+$ `v` $+$ `beta` $+$ 初态 `h0`；写 `o` $+$ 终态。

### 6.2 实测：DRAM 读基本精确，逐项归因

四组 shape（含设计点、grid 填满 GPU、非整除 $T$ 旁证）。`ratio = 实测/预测`：

| shape (B,T,H) | kernel | DRAM 读 MB | 预测 MB | ratio | workspace | v | beta | h0 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1,1024,4 | K1 | 3.18 | 3.15 | **1.010** | — | — | q,k,g,β 100% | — |
| 1,1024,4 | K2 | 4.76 | 4.73 | **1.008** | **74.9%** | 22.2% | 0.2% | 2.8% |
| 4,1024,64 | K1 | 202.66 | 201.85 | **1.004** | — | — | 100% | — |
| 4,1024,64 | K2 | 302.61 | 302.51 | **1.000** | **74.9%** | 22.2% | 0.2% | 2.8% |
| 1,8192,64 | K1 | 405.03 | 403.70 | **1.003** | — | — | 100% | — |
| 1,8192,64 | K2 | 590.52 | 590.35 | **1.000** | **76.7%** | 22.7% | 0.2% | 0.4% |
| 8,2048,8 | K1 | 101.28 | 100.93 | **1.003** | — | — | 100% | — |
| 8,2048,8 | K2 | 149.21 | 149.16 | **1.000** | **75.9%** | 22.5% | 0.2% | 1.4% |

读这一侧**四组 shape、两个 kernel 全部落在 1.000–1.010**，即 DRAM 读计数与闭式预测**逐项吻合**，
归因是硬的：

1. **K1 唯一的外部输入就是 `q,k,g,beta`**（$3\times BTHD + BTH$ 元素），**从不读 `v`**——
   与源码一致（`fwd_kernel1.cuh:231/233/235/243/253` 只 load q/k/beta/g/dt_bias，无 v 的 TMA 描述符）。
2. **K2 的读里 workspace 只占 ~75%，`v` 占 ~22.5%**。这不是小项：K2 每 chunk 还要
   `tma_load_v`/`tma_load_beta`（`fwd_kernel2.cuh:325/326, 358/367`）来算 $u=(v-u)\beta$。
   所以 §5.3「K2 的稳态上限由每 chunk 读 workspace 决定」**需修正**：workspace 是大头但不是全部，
   **`v` 的 22.5% 是不可省的输入流**，K2 一旦 grid 填满就同时被这两条流一起压。
3. `beta` 几乎免费（0.2%），`h0` 随 $B\cdot H/T$ 变化（$1{\times}1024{\times}64$ 时因 $BH$ 大、$T$ 小
   升到 2.8%，设计点 $1{\times}8192{\times}64$ 降到 0.4%）。

### 6.3 实测：DRAM 写与「dirty 行滞留 L2」的解释

写这一侧 `ratio` 小于 1：

| shape (B,T,H) | kernel | DRAM 写 MB | 预测 MB | ratio | 归因 |
|---|--:|--:|--:|--:|---|
| 1,1024,4 | K1 | **0.00** | 3.54 | 0.00 | workspace 3.54MB ≪ L2(50MB)，dirty 行留在 L2 不落 DRAM |
| 1,1024,4 | K2 | **0.00** | 1.18 | 0.00 | $o$+state 1.18MB 同样留在 L2 |
| 4,1024,64 | K1 | 208.16 | 226.49 | 0.919 | workspace 226MB > L2，约 8% 仍滞留 |
| 4,1024,64 | K2 | 65.10 | 75.50 | 0.862 | — |
| 1,8192,64 | K1 | 434.65 | 452.98 | 0.960 | — |
| 1,8192,64 | K2 | 130.19 | 136.31 | 0.955 | — |
| 8,2048,8 | K1 | 94.59 | 113.25 | 0.835 | — |
| 8,2048,8 | K2 | 29.38 | 35.65 | 0.824 | — |

小 shape 下 DRAM 写为 0 是**测量上的必然**而非「没写」：写出的字节先变 dirty 行，kernel 结束时
还没被换出，直到下一次调用或 flush 才落 DRAM。放大到 workspace ≫ L2 后，绝大部分在同一次调用内
就被换出，`ratio` 升到 0.82–0.96，与预测同量级，证明**写的归因同样成立**：K1 写的就是 workspace
（100%），K2 写的就是 `o`（94–98.5%）+ 终态（1.5–6%）。

### 6.4 结论与闭环

1. **K2 的读 ≈ 75% workspace + 22.5% `v`（+ β/h0 尾巴）**；把 §5.3 的「只读 workspace」修正为
   「以 workspace 为主、`v` 为辅」。K2 撞 L2 的墙是**两条流叠加**，workspace 仍是首要优化对象。
2. **K1 的读 = `q,k,g,β`，写 = workspace**，与 §3.2「K1 一次算完落盘」互为印证的字节账：
   设计点 K1 读 405MB + 写 435MB，K2 读 590MB + 写 130MB，合计约 **1.56GB** 的外部流量——
   与 §5.2/§5.3 用 `lts` 报的 1.7–2.0GB 同量级（差额即 TMA 的 `lts` 重发放大）。
3. 方法学收获：**TMA 密集 kernel 不要用 `lts__t_sectors` 当字节**（读 ×1.5、写 ×3）；要字节就用
   清 L2 后的 `dram__bytes`，且小 shape 的 DRAM 写会因 dirty 行滞留而读数为 0，须放大到超 L2 才有意义。

## 7. K1 那条「4/32 bytes/sector」的 global load 究竟是谁（P7-4）

§5.2 留了个待验证：`ncu` 报 K1「global load 平均每 sector 只用 4/32 字节」，但没说是哪条指令。
本节用「静态 SASS 计数 + `SourceCounters` 逐行归因 + DRAM 占比」把它钉死。复现脚本
`harness/probe_k1_gload.sh`、原始输出 `harness/probe_k1_gload.out.txt`。

### 7.1 静态：K1 里到底有几条「非 TMA」的 global load

把装好的 `flash_kda_C.cpython-312-x86_64-linux-gnu.so` 反汇编（`cuobjdump -sass`），按函数切段统计
**标量 global load `LDG.E`** 与 **TMA `UTMALDG`** 两类：

| K1 变体 | `LDG.E`（标量） | `UTMALDG`（TMA） |
|---|--:|--:|
| `IsVarlen=false`（本轮 profiling 走这条） | **1** | 5 |
| `IsVarlen=true` | **5** | 5 |

5 条 `UTMALDG` 正是 q/k/g/beta/dt_bias 五个 TMA 描述符（`fwd_kernel1.cuh:231/233/235/243/253`）。
非 varlen 只有 **1 条标量 global load**，SASS 是 `/*0a20*/ LDG.E R9, desc[UR14][R12.64]`，其前驱是

```
S2R  R2, SR_CTAID.Y              // R2 = blockIdx.y = head_idx
LDC.64 R12, c[0x0][0xda0]        // R12 = A_log_ptr
IMAD.WIDE R12, R2, 0x4, R12      // R12 = A_log_ptr + head_idx*4
LDG.E R9, desc[UR14][R12.64]     // = A_log_ptr[head_idx]
```

即它唯一对应源码 `fwd_kernel1.cuh:258` 的 `float a_log_exp = expf(A_log_ptr[head_idx]);`。
varlen 变体多出的 4 条（SASS `/*00e0*/ /*0190*/ /*01a0*/ /*01c0*/`）就是二分查找里的
`tile_prefix[mid]`/`tile_prefix[lo]`/`cu_seqlens[seq_idx]`/`cu_seqlens[seq_idx+1]`
（`fwd_kernel1.cuh:179/183/185/186`）。

### 7.2 动态：`SourceCounters` 与那条 `LDG.E` 一一对应

`ncu --section MemoryWorkloadAnalysis_Tables` 复现出的警告原文：

```
The memory access pattern for global loads from L1TEX might not be optimal. On average,
only 4.0 of the 32 bytes transmitted per sector are utilized by each thread.
This could possibly be caused by a stride between threads. ...
```

`ncu --section SourceCounters --page source` 里**整段 SASS 唯一一条 `Address Space=Global` 的 load**
就是上面那条 `LDG.E`，它的计数是：

| Source | Instructions Executed | Address Space | Access Size | L1 Tag Requests Global | L2 Sectors Global | Sectors Ideal | Excessive |
|---|--:|---|--:|--:|--:|--:|--:|
| `LDG.E R9, ...` | 2048 | Global | 32 | 2048 | 2048 | **2048** | **0** |

`B=1,T=1024,H=4` 时 grid `= ceil(1024/16)·4 = 64×4 = 256` 个 CTA × 8 warp = **2048 个 warp**，
每条 warp 恰好发射 1 条 `LDG.E`（`smsp__inst_executed_op_global_ld.sum` 实测也正好 =2048）。
所以 §5.2 那个「global load」统计**100% 就是这一条 `A_log_ptr[head]`**。

### 7.3 为什么是 4/32，以及为什么它其实没问题

`A_log_ptr[head_idx]` 的地址对 warp 内 32 个 lane **完全相同**（`head_idx=blockIdx.y` 是 CTA 一致的）。
硬件把它当**广播**处理：一个 warp 一条指令、只碰 **1 个 32B sector**（`Sectors Ideal` 也是 2048，
`Excessive=0`），但 32 个 lane 合起来只用掉其中 4 字节（一个 fp32）。`ncu` 的启发式把「每线程用到的
字节 / sector 字节」算成 `4.0/32` 并提示「可能有 stride」，这对**广播标量 load 是误报**——它已经是
该访问的最小代价（1 sector/warp），`Excessive sectors = 0` 才是有意义的可优化余量，而它是 0。

再做一次字节预算，看这条 load 值不值得动：

| shape | inst_global_ld（=warp 数） | 该 load 的字节 | K1 DRAM 读 | 占比 |
|---|--:|--:|--:|--:|
| B=1 T=1024 H=4 | 2 048 | 0.06 MB | 3.18 MB | **1.97%** |
| B=1 T=8192 H=64（设计点） | 262 144 | 8.00 MB | 405.02 MB | **1.98%** |

即把这条 load 完全消掉（例如 host 侧预展开 `exp(A_log)` 或走 kernel 参数），最多省下 K1 DRAM 读的
**~2%**；而 K1 本身在设计点只占总前向时间的约 8%（P7-1 §3.2），所以端到端收益 <0.2%。这解释了为什么
放大的 K1（§5.2 结论二）依然稳稳撞 L2——**它的瓶颈从来不是这条标量 load**。

### 7.4 结论与勘误

1. §5.2「global load 4/32」的具体归属 = **`fwd_kernel1.cuh:258` 的 `A_log_ptr[head_idx]`**（非 varlen
   是唯一一条标量 global load；varlen 再加 4 条同样是广播的二分查找标量 load）。
2. 它**不是**未合并访存：`Sectors Global == Sectors Ideal`、`Excessive = 0`，广播 load 已是 1 sector/warp
   的最小代价；`4/32` 是「每线程利用字节」启发式的误报。**勘误**：§5.2 记的「估 50% 局部加速空间」
   在本次复现的 `MemoryWorkloadAnalysis_Tables` 输出里**没有**对应到这条 global-load 警告（紧随的
   `Est. Local Speedup: 44.94%` 属于 shared-load bank conflict 警告），且与 `Excessive = 0` 矛盾，
   应视为误记/未复现。
3. 预算 <2% 且 K1 只占前向 ~8%，故 §5.4 的两条结论（**K1 = barrier stall + L2 带宽**）**不变**，
   无需为这条 load 做任何优化。

## 8. K2 突破 2 CTA/SM 后 L2 墙会不会后移？（P7-5）

§5.4 留了最后一个待验证 (c)：「把 K2 的 state smem 复用/减小以突破 2 CTA/SM 后 L2 墙是否后移」。
本节直接把它做成实验：**改造 K2 的 shared-memory 预算、编译出多个 pipeline 深度/占用率组合，
实测占用率、访存 SOL 与 K2 时间**。结论先给：**墙不会后移**——K2 是 L2 带宽 bound，多出来的占用率
用不上；而能真正把占用率抬到 3 CTA/SM 的配置都必须牺牲 pipeline 深度，反而更慢。
复现脚本 `harness/_build_k2_occ.py`（编译变体）、`harness/bench_k2_occ.py`（计时，
`bench_k2_occ.out.txt`）、`harness/probe_k2_occ_ncu.sh` + `harness/ncu_driver_occ.py`（ncu，
`probe_k2_occ_ncu.out.txt`）、`harness/_smem_probe.py`（精确 `sizeof`，`smem_k2_occ_probe.out.txt`）。

### 8.1 smem 预算：为什么「少放几个 pipeline stage」根本减不下去

`SharedStorageK2`（`fwd_kernel2.cuh:72–112`）的字节账是（$D{=}128$，CHUNK$=16$，bf16）：

```
sizeof = 32768                       // state_acc: bf16 [D,D]
       + max(IN*17984 + OUT*4096,    // union 第一臂：IN 个输入 stage + OUT 个输出 stage
             65536)                  // union 第二臂：fp32 状态转换缓冲 [D,D]*4  <-- 无条件预留
       + ~128                        // load/store pipeline + state barrier
```

每输入 stage `= v(4096) + beta(64) + 3*4096(kd/qd/kr) + g_total(512) + 2*512(INV/Mqk) = 17984` B，
每输出 stage `= out[16,128] bf16 = 4096` B。关键在 union 第二臂：`state_fp32_buf`（`fwd_kernel2.cuh:106`）
是 **65536 B 的 fp32 缓冲，只在 `StateFP32=true`（fp32 初/终态）时才用**，但对 bf16 路径也**原样预留**。

所以只要 `IN*17984 + OUT*4096 ≤ 65536`，union 就恒等于 65536，`sizeof` 恒为 **98432 B**——**与 stage 数无关**。
实测：

| 变体 | `sizeof`(B) | +driver smem(1.02KB) | ncu Block Limit Shared Mem |
|---|--:|--:|--:|
| orig / plain (3,2) | **98432** | 99456 | **2** |
| plain (2,1)（只减 stage） | **98432** | 99456 | **2** |
| cond (3,2)（bf16 去掉 fp32 缓冲） | 95232 | 96256 | 2 |
| cond (2,2) | 77184 | 78208 | **2**（3×78208=234624 > 233472）|
| cond (2,1) | 73088 | 74112 | **3** |
| cond (1,2) | 59136 | 60160 | **3** |

`cond` = 在实验分支上把 union 第二臂改成 `ReserveFP32 ? [D*D*4] : [1]`（`ReserveFP32 = StateFP32`），
即 bf16 路径不再预留那 64KB。可以看到：**只减 stage（plain 2,1）不动任何字节；去掉 fp32 缓冲（cond 3,2）
也只省 3200 B，仍是 2 CTA/SM；要摸到 3 CTA/SM 必须同时把 OUT 或 IN 砍到 1**。注意 ncu 的
`Block Limit Shared Mem` 严格用「dynamic + driver(1.02KB)/block」算，`cond(2,2)` 恰好差 384 B ×3 落回 2。

> 精度不变量：所有变体的前向输出与上游**逐位相同**（下表 `max|dO|` 全 0），此改造只动 smem 布局，
> 不改运算。

### 8.2 实测：抬高占用率并没有让 K2 变快

H100（132 SM），bf16，`CUDA_VISIBLE_DEVICES` 隔离空闲卡，wall = event 对包 50 次取 7 轮最小，
CUPTI 取 3 遍最小（脚本见 §8 头部）。B*H=grid_k2：

**B=1 T=8192 H=64（grid 64 < 132 SM，设计点：占用率上限根本用不到）**

| 变体 | CTA/SM | occ(理论) | K2 µs |
|---|--:|--:|--:|
| orig | 2 | — | 801 |
| plain (3,2) | 2 | — | 800 |
| cond (3,2) | 2 | — | 800 |
| **cond (2,2)** | **2** | — | **793** |
| plain (2,1) | 2 | — | 929 |
| cond (2,1) | 3 | — | 931 |
| cond (1,2) | 3 | — | 1468 |

grid 只有 64 个 CTA，`B*H<132`——**每个 SM 连 1 个 CTA 都分不到**，2 vs 3 CTA/SM 的帽根本不起作用；
这组里 `cond(2,2)` 的 793µs 与 orig 801µs 在噪声内，而砍 pipeline 深度的 (2,1)/(1,2) 明显更慢。

**B=8 T=2048 H=64（grid 512，真正能把 2/SM 用满）**

| 变体 | CTA/SM | K2 µs | ncu L2 SOL | ncu Duration |
|---|--:|--:|--:|--:|
| **orig** | **2** | **773** | **94.0%** | 766 |
| plain (3,2) | 2 | 771 | — | — |
| plain (2,1)（2/SM，只减 stage） | 2 | 774 | 93.1% | 774 |
| cond (3,2) | 2 | 772 | — | — |
| cond (2,2) | 2 | 775 | 93.2% | 774 |
| **cond (2,1)** | **3** | **885** | **81.4%** | 883 |
| cond (1,2) | 3 | 1086 | — | — |

三点：
1. **2→3 CTA/SM 没有换来任何加速**。`cond(2,2)`（去掉 64KB 死缓冲，但仍是 2/SM）与 orig 同速；
   唯一真正进 3/SM 的 `cond(2,1)` 反而 **慢 14%**（885 vs 773µs），而且它的 **L2 SOL 掉到 81.4%**
   （orig 94.0%）——多驻留的 CTA 只在抢同一份 L2 带宽、没有做更多有用功。
2. **占用率提升必须靠砍 pipeline 深度**（OUT2→1 或 IN→1），而深度才是 K2 的命门：`cond(1,2)`
   在 grid 512 上 K2=1086µs（1.4× orig），grid 64 上 1468µs（1.8× orig）——输入 stage 掉到 1
   后 MMA 每个 tile 都要等 LOAD，pipeline 断裂。
3. **上游 K2 已经贴着 L2 上限在跑**：设计点/填满 grid 时 `lts__throughput = 93–94%`、`dram=55–56%`、
   `sm=34%`，`sm__warps_active=18.6%`。这正是 §5 的结论——**K2 bound 在 L2 带宽**，不是延迟隐藏，
   所以再多 CTA/SM 也没有下一层可喂。

### 8.3 ncu 那句「Est. Local Speedup: 81.25%」怎么读

ncu 的 Occupancy section 对 orig K2 报：

```
Block Limit Shared Mem   2      Theoretical Occupancy  18.75%   Achieved  18.56%
OPT  Est. Local Speedup: 81.25%  ... theoretical occupancy limited by shared memory
```

这 81.25% 是**纯启发式**：它假设「占用率越高越快」，无视此时 L2 已 94% SOL。本节的实验直接证伪了
这条建议对该 kernel 的适用性——**把占用率从 18.75% 提到 28.12%（`cond(2,1)`）后，K2 反而更慢**。
正确的解读是：`Block Limit Registers = 4`、`Block Limit Warps = 10`、`Block Limit Barriers = 7`
都远高于 `Block Limit Shared Mem = 2`，smem 确实是**占用率**的瓶颈，但**占用率不是性能的瓶颈**；
性能的瓶颈是 `lts__throughput` 的 94%。

### 8.4 结论（关闭 §5.4 待验证 (c)）

- **L2 墙不会后移**。K2 在 grid 填满时 `lts` 已 93–94% SOL，是 L2 带宽 bound；实验证明把占用率
  2→3 CTA/SM（`cond(2,1)`）后 K2 反而慢 14%、L2 利用降到 81%。
- **想突破 2 CTA/SM，只能砍 pipeline 深度**（`sizeof` 由那个**无条件预留的 65536 B fp32 状态缓冲**
  主导：`IN*17984+OUT*4096 ≤ 65536` 时 stage 数怎么改都不动 smem）；而砍深度的代价
  （`cond(2,1) +14%`、`cond(1,2) +40%`）远大于任何占用率收益。唯一「免费」的改动是把 fp32 缓冲
  改为按 `StateFP32` 条件预留（省 3.2KB），但不足以跨过 3/SM 门槛，也无性能变化。
- 因此 §5.3/§5.4 的「K2 被 smem 钉在 2 CTA/SM、per-CTA 串行延迟受限」这一**描述**成立，但它
  **不是**「还可优化 51%」的意思——K2 已经没有占用率维度的余量可挖，要更快只能减少 L2 流量
  （workspace 体积/`v` 的复用，见 §6）或改并行结构。

## 9. FLA 那 0.8 ms「host 税」到底花在哪（P7-6）

§2.1 观察到 FLA `chunk_kda` 的 `wall` 恒定在 **~0.86 ms**、与 T 无关，由此推断它是 host-bound；
§4 把「host 税的组成（Python 派发 vs Triton launch）」留成待验证。本节把它拆开量到底，并给出
FlashKDA 的对照组。复现脚本 `harness/probe_nsys.sh`（内部调 `harness/probe_nsys.py` 的
`micro` / `cpu` / `profile` / `loop` 四种模式），原始输出 `harness/probe_nsys.out.txt`。
平台同 §5：H100 80GB、torch 2.12、FLA 0.5.2、triton 3.6、bf16、$D=128$。

### 9.1 口径：为什么这里用 `time.process_time`

`wall`（event 对）在 launch 队列被填满后会把 CPU 阻塞时间算进去，`torch.profiler` 的 CPU 活动
又会串行化并放大 device 时间（§1.2 同款问题）。本节的主口径改成 **`time.process_time()`**：
单线程进程里它等于真实消耗的 CPU 时间、不受 GPU 队列影响；已用「纯 Python 忙循环」校验
`process_time≡wall`（259 ms vs 259 ms）、用「大 kernel 连发」校验无后台线程干扰（6.4 µs/次）。
所有数字都先做 10 次 warmup（Triton 的 JIT/autotune 只发生一次，**不预热会把编译时间算进去**——
这是本节踩的坑，见 §9.5）。

另外用 `nsys profile --capture-range=cudaProfilerApi` 圈出稳态区间，拿到「每次调用到底发几个
kernel、driver 侧发一次多少钱」。

### 9.2 结论一：FLA 的 host CPU 是**固定 ~0.82–0.86 ms/次**，与 T 无关

`B=1, H=4`，扫 `T=16…2048`（`process_time`，纯 host CPU）：

| T | 16 | 128 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|
| FLA `chunk_kda` 前向 | 821.5 | 827.8 | 857.8 | 843.2 | 862.2 µs |
| FlashKDA `fwd`（对照） | 34.6 | 34.2 | 35.2 | 33.5 | 33.5 µs |

两者都**与 T 无关**——但相差 **~24×**。T=16 时 GPU 只要几微秒，FLA 仍要烧 0.8 ms CPU，这直接
坐实「FLA 前向是 host-bound，瓶颈在 Python/launch 而不是计算」。FlashKDA 每次调用只烧 ~34 µs
CPU，且它只发 2–3 个 kernel（§9.4），CPU 完全能压在 GPU 后面（所以 §2 里 `wall≈dev`）。

### 9.3 结论二：这 0.8 ms 的分解（`tt=1024`，`process_time`，单位 µs/次）

在 `fla.ops.kda.chunk` 上给三个顶层子调用挂计时器：

| 组成 | dispatch ON | dispatch OFF | 说明 |
|---|---|---|---|
| `chunk_kda_fwd` | 529 | 409 | 真正的递推+一堆 Triton kernel 的 launch |
| `l2norm_fwd`（×2） | 154（≈77/次） | 130（≈65/次） | 只归一化 q/k，GPU 端 ~1.6 µs，**host 端 77 µs** |
| `fused_beta_sigmoid` | 63 | 45 | 一个逐元素 kernel |
| autograd/`dispatch`/`input_guard` 等杂项 | ~95 | ~70 | `ChunkKDAFunction.forward` 自身 Python |
| **合计** | **~841** | **~654** | |

`FLA_DISABLE_BACKEND_DISPATCH=1`（`fla/ops/backends/__init__.py:28` 的 `_DISPATCH_DISABLED`，
`fla/modules/backends/__init__.py` 复用它）能让总量掉 **~190 µs**——即 FLA 那套「每次调用遍历
注册后端、`is_available/enabled` + `verify(...)`」的 Python 后端派发占了约 **22%**。注意它**不是**
一次性初始化：`dispatch` 的 `wrapper` 每次调用都重跑一遍（`__init__.py:167` 起）。

> 关键点：**单个 Triton kernel 的 host 成本 ≈ 77 µs**（`l2norm_fwd`），而 §9.5 的裸 launch 只要
> **2.4 µs**。多出来的几十 µs 是 FLA 在 Triton 之上叠的每调用 Python 包装（自己的
> `fla_cache_autotune`/`CachedAutotuner`、`AutotuneKey.build` 的 dtype 序列化等），不是驱动、不是
> 分配（裸 `torch.empty`/`empty_like` 实测 0.07 µs）。

### 9.4 结论三：nsys 稳态——FLA 发 12 个 kernel/次，FlashKDA 3 个

`nsys` 稳态（capture-range，N=200；driver API 用**中位数**，因为首个调用含 lazy 初始化的
2.5 ms 长尾）：

| | kernel 数/次 | driver launch API 中位 | 稳态 GPU kernel 合计* |
|---|---|---|---|
| FLA `chunk_kda`（B1 T1024 H4） | **12**（9 `cuLaunchKernelEx`+3 `cudaLaunchKernel`） | ~6.3 µs/次 | ~82 µs/次 |
| FlashKDA `fwd`（B1 T1024 H4） | **3**（`cudaLaunchKernel`） | ~5.6 µs/次 | ~109 µs/次 |
| FLA `chunk_kda`（B8 T2048 H8） | 12 | ~5.4–5.8 µs | ~728 µs/次 |
| FlashKDA `fwd`（B8 T2048 H8） | 3 | ~5.5 µs | ~332 µs/次 |

\* 取各 kernel 的 `Med`×实例数求和。

于是 0.8 ms 的账很清楚：**纯 driver 侧只需 ~75 µs**（12 次 × ~6.3 µs），GPU 计算 ~82 µs，
剩下 **~680 µs 全是 FLA 在 Python 里、每 kernel 一次地烧掉的 launch/包装开销**——12 个 kernel
每多一个就多一份这套开销。FlashKDA 只有 3 个 launch（且直接走 torch 的 C launch，没有 Python
包装栈），host 侧总成本 ~34 µs，完全被 GPU 时间盖住。

### 9.5 一个必须避开的坑：不预热会把 JIT 编译算成「launch 开销」

第一版直接在 `process_time` 区间里连续调 200 次未预热的 kernel，量到 **3843 µs/次**的荒唐值——
那其实是 **Triton 首次 JIT 编译 + autotune 的几十毫秒摊到 200 次里**。改成「先 warmup 10 次再计
时」后同一 kernel 稳定在 2.3–2.4 µs。所以本节所有 host 数字都带 warmup；谁做类似测量务必先
预热，否则会把编译一次性成本误读成稳态 launch 税。

### 9.6 小结（关闭 §4 待验证「FLA host 税组成」）

- FLA 前向的 ~0.8 ms host 税 ≈ **12 次 Triton kernel 的 Python launch 包装**（每次几十 µs）
  \+ **~190 µs 后端 `dispatch` 派发** \+ ~75 µs 真正的 driver launch；**与 GPU 计算、autograd、
  张量分配都无关**（GPU ~82 µs、裸 launch 2.4 µs、裸分配 0.07 µs）。
- 这正是 §2.1「FLA wall 被钉在 0.86 ms」的微观来源，也解释了 §3.2「融合成一个 K1、把 12 个
  launch 压到 2–3 个」为何在中小 shape 上能带来数量级的 wall 收益。
- 可迁移的经验：**逐 kernel 的 Python launch 开销是 Triton 类实现做小算子融合时最容易被忽视的
  成本**；KDA 这类「每个 chunk 一次小 kernel」的算法尤其敏感。（目标卡 CUDA 反向同理——§6 的
  B0/B1/B2/B3 每条支路一次 launch 已是最小化。）

## 10. 反向性能：workspace 复用链 vs FLA autograd（P7-7）

§1–§9 只剖析了**前向**。反向侧（`docs/06` P6-2…P6-8）此前只做过逐 kernel 的朴素秒表计时，
既没有统一口径的 `wall`/`dev` 拆分，也没有和 FLA 反向的端到端对照。本节补上：把 P6-7/P6-8 组装出的
**workspace 驱动反向链**（`dAv → B1(dhu) → B2(wy_dqkg) → B3(intra) → 反向 cumsum(B0)`）和
`torch.autograd` 穿 `fla.ops.kda.chunk_kda` 放在同一口径下比。复现脚本 `harness/bench_bwd.py`，
原始输出 `harness/bench_bwd.out.txt`；平台同 §5（H100 80GB、torch 2.12、FLA 0.5.2、triton 3.6、
bf16、$D=128$、`CHUNK=16`）。

### 10.1 口径：怎么只计「反向」

- `wall` = 一个 CUDA event 对包 `REPS=30` 次取均值、3 轮取中位（含 host 派发）；`dev` = `torch.profiler`
  （CUPTI）把所有 kernel 的 self time 求和再除以调用数（纯 GPU）。同 §1.1/§9.1。
- **只计反向**：两条路径都把前向排除在计时区间外。FLA 侧先在区间外建好 autograd 图，区间内只跑
  `loss.backward(retain_graph=True)`（随后清空 `.grad`，可反复重跑同一张图）；ours 侧把 FlashKDA
  的 K1 workspace 与 `fi` 中间量在区间外预先重建好（`check_bwd_ws_reuse.py` 的
  `flash_forward`+`reconstruct`），区间内只跑那条反向链的 closure。
- `host = wall − dev` 即 Python/launch 开销；与 §9 的 `process_time` 不同，这里保留 event-pair
  `wall` 口径以便和 GPU 时间直接相减（大 shape 下 host 会被队列隐藏，见下）。

### 10.2 主表：反向 wall / dev，四组 shape

单位 µs/次。`#k` = 一次反向发出的 CUDA kernel 数（CUPTI 计数）。`w-o/FLA`、`d-o/FLA` 越小越好。

| case（B T H） | ours wall | ours dev | ours host | FLA wall | FLA dev | FLA host | w-o/FLA | d-o/FLA | #k ours/FLA |
|---|---|---|---|---|---|---|---|---|---|
| 1 512 4 | 349.6 | 163.0 | 186.6 | 703.5 | 207.2 | 496.3 | **0.50** | **0.79** | 40 / 28 |
| 2 1024 8 | 1085.7 | 972.8 | 113.0 | 1419.9 | 1322.8 | 97.2 | **0.76** | **0.74** | 40 / 28 |
| 8 2048 8 | 5630.7 | 5513.4 | 117.3 | 9197.3 | 9119.6 | 77.7 | **0.61** | **0.60** | 40 / 28 |
| 1 250 3（非整除 T） | 355.5 | 97.6 | 257.8 | 872.4 | 143.9 | 728.5 | **0.41** | **0.68** | 40 / 28 |

三点观察：

1. **我们的反向链在 wall 与 dev 上都快过 FLA 反向**（wall 0.41–0.76×、dev 0.60–0.79×），
   尽管其中的 Triton 反向 kernel 都是**朴素实现**。原因见 §10.3：FLA 反向里包含了**前向状态重算**，
   而我们直接吃 FlashKDA K1 落盘的 workspace。
2. **host 税的量级**：FLA 反向的 host 成本约 **500–730 µs/次**（小 shape 下暴露、大 shape 下被
   9120 µs 的 GPU 时间完全隐藏 → `host≈78 µs`）；我们那条链约 **110–260 µs/次**（同样在大 shape
   下隐藏）。注意 ours 一次反向发 **40 个 kernel**（含十来条 torch elementwise 兜底，如各支路结果的
   `+` 相加、`chunk_local_cumsum` 收尾），FLA 28 个——但两者 host 都远小于 §9 里 FLA *前向* 的
   ~0.8 ms 固定税，因为反向的 GPU 时间更长、队列能把 host 藏住。
3. 前向对照（同脚本）：FlashKDA `fwd` dev = 63.5 / 128.2 / 668.7 / 42.9 µs（3 kernel）；
   `fwd+chain` 合计 dev = 226.5 / 1101.0 / 6182.1 / 140.5 µs，而 FLA `chunk_kda` **fwd+bwd** dev =
   325.1 / 1907.9 / 12635.0 / 250.2 µs。即端到端（前向+反向）我们也快 **1.4–2.0× dev**。

### 10.3 归因：FLA 反向重算了前向状态，我们复用了 K1 workspace

`B2 T=1024 H=8` 的逐 kernel 分解（CUPTI，µs/次）：

| ours chain | µs | | FLA backward | µs |
|---|---|---|---|---|
| `_dhu_kernel`（B1） | **539.8** | | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（前向重算） | **578.0** |
| `_wy_dqkg_kernel`（B2） | 125.1 | | `chunk_kda_bwd_kernel_wy_dqkg_fused`（B2） | 239.1 |
| `_intra_kernel`（B3） | 80.4 | | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`（B1） | 136.9 |
| `_dav_kernel`（dAv） | 11.3 | | `chunk_kda_bwd_kernel_intra`（B3） | 119.5 |
| `chunk_local_cumsum`（B0 收尾） | 6.1 | | `chunk_kda_bwd_kernel_dAv`（dAv） | 18.3 |
| torch elementwise（多支路相加/清零等） | ~45 | | `recompute_w_u_fwd_kda_kernel`（前向重算） | 32.7 |
| | | | `chunk_local_cumsum` + reduce/elementwise | ~15 |
| **合计** | **972.8** | | **合计** | **1322.8** |

关键在**最上面两行**：FLA 的反向 profile 里出现了前向的状态 kernel
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（578.0 µs）与 `recompute_w_u_fwd_kda_kernel`
（32.7 µs）——**FLA 在反向里重算了 chunk 状态与 `w/u`**（P6-1 §6.2 记录的重算策略），这两项合起来
**610.7 µs = FLA 反向 dev 的 46.2%**（$B1\,T512\,H4$ 时是 21.8%）。我们的链**从不重算前向**：六个
per-tile 字段（$k^{dec},q^{dec},k^{rest},E^{tot},P,M_{qk}$）直接从 FlashKDA K1 的 workspace 读，
`U/h` 由它们逐块重放（`check_bwd_ws_reuse.reconstruct`）——只在计时区间外做一次。这正好把
`docs/06` §5.2「复用 K1 workspace」的设计收益从**正确性**（P6-8 已核）兑现成了**性能**。

也要诚实指出短板：我们自己的 **B1 `_dhu_kernel` 是整条链的瓶颈**（539.8 µs = ours dev 的 55.5%；
$T512H4$ 时 102.3/163.0 = 62.8%），且比 FLA 的对应 kernel（136.9 µs）**慢 3.9×**。这是一条朴素 Triton
实现（P6-2/P6-3，未做软件流水/共享内存搬运优化），而 FLA 的 `dhu` 已沿 T 分块并行。B1 在目标卡
CUDA 里本应**直接镜像前向 K2 的 warp-specialized 八相递推**（`docs/06` §5.2），这正是下一步
工程化的首要目标（见 §10.4）。

### 10.4 结论

- **复用 K1 workspace 的反向在设计上就省掉了 FLA 反向最大的一块（前向状态重算 ≈ dev 的 22–46%）。**
  即便我们的四个 Triton 反向 kernel 都是朴素版，整条链的 dev 仍是 FLA 反向的 **0.60–0.79×**、
  wall **0.41–0.76×**；端到端（前向+反向）dev 是 FLA 的 **0.5–0.8×**。
- **下一步的优化空间集中在 B1（`dhu`）**：它占我们 dev 的 ~55–63%、且比 FLA 同类 kernel 慢近 4×；
  把它做成目标卡 CUDA、复用前向 K2 的 warp specialization / TMA pipeline，是整条反向最大的单点收益。
  B2（`wy_dqkg`，125 vs FLA 239，已更快）与 B3（`intra`，80 vs 120）反而已经领先，说明朴素实现
  在块内小矩阵上吃亏不大，**跨块状态递推的并行/流水**才是要害。
- **host 税不是反向的主要矛盾**：两条路径的 host 成本都在百 µs 量级且大 shape 下被隐藏，
  与 §9 前向那 0.8 ms 的固定 Python 税不是一个量级。（可优化项：ours 的 40 个 kernel 里有十余条
  torch elementwise 兜底，真实融合 CUDA 反向可把这些并入 kernel。）
- 方法学提醒：本节 FLA 侧用 `retain_graph=True` 反复跑同一张图的 backward 来隔离反向；若不保留图，
  就只能量到「fwd+bwd」的合并值，无法把前向重算从反向里区分出来。

## 11. 复核 §10：GPU 竞争污染与两个混淆变量（P7-8）

§10 有两个数字事后被证明**不可复现**，且它们恰好是「B1 是瓶颈」这一结论的主要依据：

- ours 的 B1 `_dhu_kernel` 在 $B2\,T1024\,H8$ 报 **539.8 µs**（→「比 FLA 慢 3.9×」）；
- FLA 的前向状态重算 `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` 报 **578.0 µs**（→「重算占
  FLA 反向 dev 的 46.2%」）。

本轮把 `bench_bwd.py` **钉到空闲卡**（`CUDA_VISIBLE_DEVICES=1`；P7-7 当时机器上 GPU 0 正被另一个
容器打到 100% 利用率）重跑，输出存 `harness/bench_bwd_idle.out.txt`，并写了一个**控制变量**脚本
`harness/check_bwd_dhu_dtype.py`（`harness/check_bwd_dhu_dtype.out.txt`）专门拆解 B1。

### 11.1 重跑的 §10 主表（空闲卡）

| case（B T H） | ours wall | ours dev | ours host | FLA wall | FLA dev | FLA host | w-o/FLA | d-o/FLA |
|---|---|---|---|---|---|---|---|---|
| 1 512 4 | 316.5 | 162.6 | 153.9 | 800.3 | 202.8 | 597.5 | **0.40** | **0.80** |
| 2 1024 8 | 518.0 | 483.0 | 35.0 | 877.9 | 652.3 | 225.6 | **0.59** | **0.74** |
| 8 2048 8 | 2551.8 | 2515.1 | 36.7 | 4135.5 | 4108.8 | 26.7 | **0.62** | **0.61** |
| 1 250 3（非整除 T） | 372.5 | 97.1 | 275.4 | 857.0 | 144.0 | 713.0 | **0.43** | **0.67** |

**结论不变**（ours dev 仍是 FLA 的 0.61–0.80×，wall 0.40–0.62×），但绝对值整体比 §10 低约 2×：
如 $B2\,T1024\,H8$ 的 ours dev 从 972.8→**483.0**、FLA dev 从 1322.8→**652.3**。即 §10 的两侧都被
竞争同时抬高了约一倍，**比值因此没暴露问题，但逐 kernel 的绝对时间和占比全被放大**。

### 11.2 被污染的恰恰是「关键证据」

$B2\,T1024\,H8$ 的逐 kernel 分解（空闲卡重测）：

| ours chain | §10 报 | 空闲卡 | | FLA backward | §10 报 | 空闲卡 |
|---|---|---|---|---|---|---|
| `_dhu_kernel`（B1） | 539.8 | **214.3** | | `..._fwd_kernel_h_blockdim64`（前向重算） | 578.0 | **83.6** |
| `_wy_dqkg_kernel`（B2） | 125.1 | 126.2 | | `chunk_kda_bwd_kernel_wy_dqkg_fused`（B2） | 239.1 | 237.8 |
| `_intra_kernel`（B3） | 80.4 | 81.0 | | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`（B1） | 136.9 | **137.4** |
| `_dav_kernel`（dAv） | 11.3 | 11.3 | | `chunk_kda_bwd_kernel_intra`（B3） | 119.5 | 120.3 |
| `chunk_local_cumsum`（B0 收尾） | 6.1 | 6.0 | | `chunk_kda_bwd_kernel_dAv` | 18.3 | 18.5 |
| torch elementwise（各支路相加等） | ~45 | ~45 | | `recompute_w_u_fwd_kda_kernel`（前向重算） | 32.7 | 32.8 |
| **合计** | **972.8** | **483.0** | | **合计** | **1322.8** | **652.3** |

只有**两行**对不上，且都是「大头的、带串行递推的」kernel：

- **ours B1**：539.8 → **214.3 µs**。FLA 侧同一个 `dhu`（137.4）几乎没变，说明竞争主要压在
  我们这条长串行链上。
- **FLA 前向重算** `fwd_kernel_h`：578.0 → **83.6 µs**（$B1\,T512\,H4$ 时是 34.6 µs，4× 工作量报
  578/34.6≈17× 本就反常）。修正后在 $B2\,T1024\,H8$ 重算两块 `fwd_h`(83.6)+`recompute_w_u`(32.8)
  = 116.4 µs = FLA 反向 dev 的 **17.8%**（不是 §10 的 46.2%）；$T512H4$ 时 (34.6+9.9)/202.8 =
  **21.9%**。

### 11.3 B1 到底比 FLA 慢多少：控制 chunk 与 dtype 两个变量

即便用干净数字，§10 的「B1 慢 FLA 3.9×」仍是把**两个不同口径**放在一起比：

1. **chunk 粒度**：我们的 B1 吃 FlashKDA 的 `CHUNK=16`（`docs/06` 的六个 per-tile 字段都在 16 块上），
   而 FLA 的 `dhu` 是 `chunk_size=64`（FLA 的 KDA intra 只支持 32/64，见其 `chunk_intra.py`）；
2. **操作数 dtype**：P6-8 的 `reconstruct` 把整个 workspace 都 `.float()` 了，于是我们的 B1 跑的是
   fp32 操作数 → `tl.dot` 落到 **tf32**（且状态 `b_dh` 保持 fp32）；FLA 的 `dhu` 每次
   `tl.dot(b_k, b_dh.to(b_k.dtype))` 把状态**舍到 bf16 进 mma**、走 bf16 TensorCore。

`check_bwd_dhu_dtype.py` 造了一个**只用改 `tl.dot` 操作数 dtype** 的克隆 kernel
（`_dhu_kernel_bf16`：状态仍 fp32 累加，仅在点积里 `.to(bf16)`，与 FLA 逐字一致），在**同一份操作数**
上扫 {ours, FLA} × {fp32/tf32, bf16} × {BT=32, BT=64}（kernel-only，CUPTI，空闲卡）：

| shape | ours32 | FLA32 | o/F | ours64 | FLA64 | o/F | dtype |
|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 70.6 | 45.6 | 1.55 | 65.6 | 53.2 | 1.23 | fp32 |
| B2 T1024 H8 | 176.9 | 107.1 | 1.65 | 147.5 | 126.3 | 1.17 | fp32 |
| B8 T2048 H8 | 589.1 | 482.0 | 1.22 | 593.0 | 523.8 | 1.13 | fp32 |
| B1 T512 H4 | 25.5 | 20.5 | 1.24 | 17.6 | 13.0 | 1.35 | bf16 |
| B2 T1024 H8 | 73.5 | 37.8 | 1.95 | 41.8 | 23.0 | 1.81 | bf16 |
| B8 T2048 H8 | 210.3 | 169.9 | 1.24 | 161.5 | 120.2 | 1.34 | bf16 |

读数：

- **同一 BT、同一 dtype 下，我们的 B1 只比 FLA 的 `dhu` 慢 1.1–1.9×**（不是 3.9×）。这不是「实现差
  一个数量级」，而是一个小的常数因子（我们的 kernel 是单次 `tl.dot` 到 `[K,BV]` 整块状态、没有 FLA
  的 K-64 分块与 `num_stages` 流水）。
- **chunk 粒度**是真实差距：ours 在 BT=16（`CHUNK`）比 BT=64 多约 **1.3–1.6×**（串行步数 ×4、每步
  tile 更小）。§11.3 的 ours-BT16 这里用 Part C 的量：$T512H4$ 91.3 µs、$B2\,T1024\,H8$ 215.0 µs，
  对上 BT=32/64 的 70.6/65.6、176.9/147.5。

### 11.4 一个更便宜的杠杆：让 B1 吃 workspace 的**原生 bf16**

既然 FlashKDA 的 workspace 六个字段里 `kd/qd/kr/INV/Mqk` **本来就是 bf16**（只有 $E^{tot}$ 是 fp32），
P6-8 为了对拍把它整套 `.float()` 成 fp32，反而让 B1 掉进最慢的 tf32 路径。把操作数保持 bf16（`do/dv`
也 bf16）再测同一 kernel：

| case | ours B1（P7-7 口径 fp32/tf32, BT=16） | ours B1（bf16 操作数, BT=16） | 加速 |
|---|---|---|---|
| B1 T512 H4 | 91.3 µs | **37.2 µs** | **2.45×** |
| B2 T1024 H8 | 215.0 µs | **97.9 µs** | **2.20×** |

即**不改算法、只把操作数 dtype 对齐 workspace 原生 bf16**，B1 就快 2.2–2.5×，$B2\,T1024\,H8$ 下
97.9 µs **已经快过** FLA 的 `dhu`（空闲卡 137.4 µs）。正确性（`check_bwd_dhu_dtype.py` Part A，都
BT=64、都 bf16）：ours-bf16 vs FLA-bf16 的 `dh`/`dv2` max_abs 6.1e-5–2.4e-4、`dh0` 7.5e-9–1.7e-7
（状态累加器仍 fp32，只有 mma 操作数被舍到 bf16）。

### 11.5 更正后的归因与下一步

- **§10 的「B1 慢 3.9×」不成立**：那是单点污染值（539.8→214.3）。干净口径下 B1 比 FLA 慢
  **1.6×**（$T512H4$ 102.0 vs 54.0 = 1.89×，$B2\,T1024\,H8$ 214.3 vs 137.4 = 1.56×），与表
  11.3 的 1.1–1.9× 一致。B1 仍是 ours 反向链里最大的单 kernel（$B2\,T1024\,H8$ 214.3/483.0 =
  **44.4%**，$T512H4$ 102.0/162.6 = 62.7%），但**差距的性质从「实现差 4×」变成「chunk 粒度 + 一个
  常数因子」**。
- **优化优先级重排**：先做**零算法改动**的那一步——B1（以及整条链）吃 workspace 原生 bf16，预期
  B1 ≈ **2.2–2.5×**（§11.4）；再考虑把跨块状态递推**按更大的 chunk（重组 64 块操作数）**跑，消掉
  §11.3 那 1.3–1.6× 的粒度损失；最后才是镜像前向 K2 的 warp-specialized/TMA 重写。
- **方法学修正**：本专题此前的前向/反向基准（P7-1、P7-7）都在**未隔离 GPU**的情况下测的，本轮起
  统一 `CUDA_VISIBLE_DEVICES=<空闲卡>`；P7-7 的比值结论侥幸成立，但其绝对时间与「重算占 46.2%」
  需按 §11.2 修正为 **17.8%**（$T512H4$ 21.9%）。

## 12. 把 §11.4 的杠杆落到真实反向链：B1 吃 workspace 原生 bf16（P7-9）

§11.4 的「B1 吃 bf16 操作数快 2.2–2.5×」是在**孤立 kernel** 上、**手工**把操作数换成 bf16 测的；
P6-8 那条真正跑起来的整链（`check_bwd_ws_reuse.py`）里，`reconstruct` 仍把整个 workspace 用
`.float()` 转成 fp32，于是 B1 依然掉在 tf32 路径——**§11.4 的收益从未落地到组装链**。本节把它落下来，
并重新量这条链的**正确性**与**端到端反向性能**。脚本 `harness/check_bwd_b1_bf16_e2e.py`
（`harness/check_bwd_b1_bf16_e2e.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 12.1 改动：只换 B1 的 mma 操作数，输出保持 fp32

新增一个 bf16-mma 的 B1 clone（`_dhu_kernel_bf16`，比 §11.4 的版本多了 P7-7 的 `HAS_DHT` 终态梯度入口）：
**状态 `b_dh` 仍 fp32 累加，只有进入 `tl.dot` 的操作数 `.to(bf16)`**——与 FLA `dhu` 逐字一致。
链条里只有 B1 换实现，B2/B3 不动。

一个实现要点：**`dh`/`dv2` 两个输出保持 fp32**（不是 FLA 的 bf16）。原因不是精度，是**接口兼容**——
P6-8 的 `wy_dqkg`(B2) 内核把 `dv` 与 fp32 的 `v` 放进同一个 `tl.dot`（`check_bwd_dqkg_kernel.py:341`
附近的 `b_dA += tl.dot(b_dv, tl.trans(b_v))`），`dh` 与 fp32 的 `h` 在同一条 `tl.sum` 里；若 B1 落盘
bf16，B2 立刻报 `Both operands must be same dtype. Got bf16 and fp32`。B1 的加速来自 **mma 操作数
dtype**（bf16 TensorCore 2× tf32 且加载字节减半），与输出 dtype 无关，故保持 fp32 只损失两处 store 的
带宽，不影响本节结论。

### 12.2 正确性：整链仍落在 bf16 底噪

`check_bwd_b1_bf16_e2e.py` Part A 用 **P6-8 完全相同的 6 组 case**（含非整除 $T=250/513$、$g=0$、$B=2$，
同一份激活后操作数、`dh0` 含终态上游），分别跑 fp32-B1 链与 bf16-B1 链，对 `torch.autograd`(FLA)：

| 梯度 | bf16-B1 链 vs FLA（max_rel） | fp32-B1 链 vs FLA（P6-8 基线） |
|---|---|---|
| `dq` | 3.42e-3 | 3.42e-3 |
| `dk` | 3.02e-3 | 2.98e-3 |
| `dv` | 6.80e-3 | 6.34e-3 |
| `dbeta` | 6.01e-3 | 5.97e-3 |
| `dg` | 7.06e-3 | 7.05e-3 |
| `dh0` | 8.27e-3 | 6.42e-3 |

六项全部与基线**同阶**（bf16 输入 / tf32 `tl.dot` 底噪）。把「操作数换 bf16」这一步自身的增量单列
（bf16-B1 链 vs fp32-B1 链，同 case 的跨 case 最坏）：`dq` **逐位相同（0）**（B1 不产 `dq`）、
`dk ≤1.03e-3`、`dv ≤3.76e-3`、`dbeta ≤2.30e-3`、`dg ≤1.08e-3`；B1 自己的 `dh`（bf16 vs fp32）跨
case max_rel **6.4e-4–3.9e-3**。即**把它换成 bf16 没有把任何一项梯度推出底噪**。

### 12.3 端到端反向性能：B1 快 1.7–2.1×，整链 dev 降到 0.71–0.91×

反向 only（复现 P7-8 口径：空闲卡、`wall`=一个 event 对包 30 次取中位、`dev`=CUPTI 纯 kernel），
四组形状（见 `harness/check_bwd_b1_bf16_e2e.out.txt`）：

| case（B T H） | ours fp32-B1 wall/dev | ours bf16-B1 wall/dev | FLA wall/dev | bf16/fp32 dev | bf16/FLA dev |
|---|---|---|---|---|---|
| 1 512 4 | 306.0 / 162.7 | 320.4 / **115.1** | 798.4 / 202.9 | **0.71** | **0.57** |
| 2 1024 8 | 517.8 / 479.7 | 429.3 / **393.2** | 880.6 / 656.5 | **0.82** | **0.60** |
| 8 2048 8 | 2564.3 / 2514.3 | 2312.8 / **2284.2** | 4122.2 / 4093.3 | **0.91** | **0.56** |
| 1 250 3（非整除 T） | 370.5 / 96.3 | 389.5 / **73.2** | 875.3 / 143.3 | **0.76** | **0.51** |

逐 kernel（device µs/call，`_dhu_kernel` 即 B1）：

| case | B1 fp32→bf16 | B1 加速 | B2 `wy_dqkg` | B3 `intra` | dAv |
|---|---|---|---|---|---|
| 1 512 4 | 101.5 → **53.2** | **1.91×** | 21.9（不变） | 14.8（不变） | 3.4 |
| 2 1024 8 | 208.2 → **118.8** | **1.75×** | 124.9 | 78.6 | 11.5 |
| 8 2048 8 | 651.9 → **387.0** | **1.68×** | 899.8 | 559.3 | 74.4 |
| 1 250 3 | 46.9 → **22.4** | **2.09×** | 14.9 | 12.3 | 2.4 |

读数：

- **B1 快 1.68–2.09×**，与 §11.4 孤立 kernel 的 2.2–2.5× 同向（略低，因为这里的 `dh`/`dv2` 落盘是
  **fp32**——为兼容 B2 而做的接口选择，见 §12.1——而不是 §11.4 的 bf16 落盘）。$B2\,T1024\,H8$ 下
  B1 = **118.8 µs，已快过**空闲卡 FLA `dhu` 的 137.4 µs（§11.2）。
- **整链 dev 降到 fp32-B1 链的 0.71–0.91×、FLA 反向的 0.51–0.60×**。形状越大 B2/B3（大 GEMM）占比
  越高，B1 的收益被摊薄（$B8\,T2048\,H8$ 只到 0.91×，因为 B2/B3/dAv 合计 1.63 ms 把 B1 从 652 降到
  387 省下的 265 µs 稀释了）；小 shape（$T512H4$）B1 占 dev 的 63%，故整链直接降到 **0.71×**。
- **wall 在小 shape 反而略高（320 vs 306）**：这是 **harness 的 host 税**，不是 kernel 变慢——
  `dhu_bf16_triton` 每个 timed call 里对来自 dAv 的 fp32 `dv` 做一次 `.to(bf16)`（一次 elementwise +
  分配）；小 shape 的整链 wall 本就由 host launch 主导（P7-6），这点额外 host 被放大。真实实现里
  `do/dv` 与 workspace 同为 bf16、无需运行期转换，故应以 **dev（纯 kernel）** 为口径。大 shape
  （$B8\,T2048\,H8$）dev 主导，wall 也随之下降（2564→2313）。

### 12.4 结论与下一步

- **§11.4 的杠杆已落地**：只把 B1 的 mma 操作数对齐 workspace 原生 bf16（零算法改动），整条
  workspace 驱动反向链在四组形状上 dev 降到 **0.71–0.91×**（相对自身 fp32-B1）、**0.51–0.60×**
  （相对 FLA 反向），正确性仍在 bf16 底噪（六项 max_rel ≤8.3e-3，与 P6-8 同阶）。
- 至此 §11.5 的「优先级」第 (1) 步（吃原生 bf16）**完成**。剩下的第 (2) 步——按更大 chunk（把 `CHUNK=16`
  的跨块状态递推重组到 64 块操作数）消掉 §11.3 那 1.3–1.6× 粒度损失——是下一个可做的工程化增量；
  第 (3) 步镜像前向 K2 的 warp specialization 仍然排在最后。第 (2) 步见 §13。

## 13. B1 更大 chunk 重组：从 16-chunk workspace 恢复 64-chunk 操作数（P7-10）

§11.5 第 (2) 步是「把 `CHUNK=16` 的跨块状态递推**重组到 64 块操作数**」以消掉粒度损失。它有一个先
天矛盾：B1 吃的是 FlashKDA K1 workspace 的 **16-chunk** 字段（`kd/qd/kr/gt`），而 FLA `dhu` 用 64。
本节回答两个问题：**(A) 64-chunk 操作数能不能从 16-chunk workspace「免前向重算」地恢复？(B) 恢复后
B1@64 是否真的更快、正确性如何？** 脚本 `harness/check_bwd_b1_rechunk.py`
（`harness/check_bwd_b1_rechunk.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 13.1 为什么可以直接重算（代数）

分块状态递推 `S_{c+1} = diag(E_c) S_c + k_rest_c^T U_c` **对任意 chunk 大小都精确成立**，因此
64-chunk 的 `(q_dec, k_rest, w, E_tot)` 只是把 16-chunk 字段按**子块重新标定**。对一个由 4 个 16-子块
`s=0..3` 组成的 64 组，记子块门控和 `G16_s`（log2）、前缀偏移 `off_s = Σ_{s'<s} G16_s'`：

| 64-chunk 量 | 由 16-chunk workspace 字段导出 |
|---|---|
| `q_dec64[t]` | `q_dec16[t] · 2^{+off_s}`（`qd` 字段，含 scale） |
| `k_rest64[t]` | `k_rest16[t] · 2^{G64 − off_s − G16_s}`（`kr` 字段；指数 ≤0，**有界**） |
| `ki64[t]` | `ki16[t] · 2^{−off_s}`，`ki16 = kr16 · 2^{−G16}`（**无界**，见 13.4） |
| `E_tot64` | `2^{G64} = Π_s E16_s`（`gt` 字段） |
| `P64 = (I+L64)^{-1}` | `L64 = tril(kd64 @ ki64^T, −1) ⊙ β_u`，**唯一需要新算的东西** |
| `w64` | `P64 diag(β) kd64` |

其中 `G16_s` 从 B0 重算的门控取（子块末 token 的 chunk-local cumsum），`off_s` 是它在组内的前缀和。
`qg64/kg64/kd64/E64` 全部是**逐子块标量重标定**，直接复用 workspace 字段即可；`P64/w64` 是 64 块内
那一次「求逆」（正是 K1 在 16 块上做过的活，现在要在 64 块上重做）。**注意：即使前向输出 `o/S` 与
chunk 大小无关，`w64` 也≠重排后的 `w16`**（实测 `max_rel≈1.0`）——强衰减下 64 块早段的 `E64` 小到
`3e-86…5e-22`，操作数被最后一个子块主导，64 块与 16 块的块内逆是不同对象。

### 13.2 Part A：恢复出的操作数 == FLA forward@64（bf16 底噪）

四组 case（含非整除 `T=250/513`、`g=0`），恢复值 vs **FLA 自己 forward@64** 的 `qg/kg/w/g_cum`：

| case | qg | kg | w | g_cum |
|---|---|---|---|---|
| T=256 H=2 | 3.67e-3 | 7.04e-3 | 8.44e-3 | 9.41e-4 |
| T=250 H=3 非整除 | 7.83e-3 | 6.58e-3 | 1.67e-2 | 9.05e-4 |
| B2 T=513 H=2 非整除 | 5.03e-3 | 6.06e-3 | 8.73e-3 | 1.08e-3 |
| T=192 H=2 g=0 | 8.37e-3 | 3.13e-3 | 1.10e-2 | 5.91e-3 |

全部落 bf16 底噪（`w` 因走 `P64` 略高、到 1.7e-2），**含非整除 T**。fp64 与 fp32（稳定分块）两条恢复
路径给出**逐项相同**的结果。

### 13.3 Part B/C：B1@64 正确且快 2.1–2.3×（但恢复成本是硬伤）

把恢复出的 64-chunk 操作数喂 P7-9 的 bf16-mma B1 核（`BT=64`，token 轴补零、以零操作数 mask），
与 FLA `dhu`@64 对拍（同一份 `do/dv`、`dht=None`，只切有效 token）：

| 梯度 | 最坏 max_rel（6 组×两精度） |
|---|---|
| `dh` | 1.55e-2 |
| `dv2` | 1.87e-3 |
| `dh0` | 1.44e-2 |

全落 bf16 底噪（与 P6-8/P7-9 同阶），**含非整除 T**。计时（kernel-only CUPTI，空闲卡）：

| case | ours B1@64 | ours B1@16（P7-9） | FLA dhu@64 | **o64/o16** | o64/FLA |
|---|---|---|---|---|---|
| T=512 H=4 | 19.5 µs | 45.5 µs | 13.0 µs | **0.43** | 1.50 |
| B2 T=1024 H=8 | 54.8 µs | 122.2 µs | 23.2 µs | **0.45** | 2.36 |
| B8 T=2048 H=8 | 187.4 µs | 393.2 µs | 119.5 µs | **0.48** | 1.57 |
| T=250 H=3 非整除 | 10.8 µs | 23.4 µs | 7.4 µs | **0.46** | 1.46 |

- **B1 本身快 2.1–2.3×**：在 bf16 下（P7-9 的口径）粒度损失比 §11.3 在 tf32 下测的 1.3–1.6× **更大**，
  `BT=16→64` 值 2.1–2.3×。相对 FLA 仍有 1.5–2.4× 常数因子（我们是单次 `[K,BV]` 整块状态 `tl.dot`、
  没有 FLA 的 K-64 分块与 `num_stages` 流水，§11.3）。
- **硬伤——恢复本身不便宜**：`recover_operands_64` 作为 torch 算子序列（含 B0 门控重算）实测
  **~2.0–5.7 ms**（`T512H4` 2.46 ms、`B8T2048H8` 5.68 ms），远大于 B1 的 10–187 µs。拆开看：纯
  「重标定」（`qg/kg/g_cum`，不算 `P64/w64`）就要 0.47–1.48 ms，**64 块内逆 `A64/P64/w64` 再占
  2.0–3.7 ms**。这是 torch 端逐算子 + `linalg.solve` 的 host/库开销，生产实现必须像 K1 那样**融成一个
  Triton pass**（`A64` 是 64×64 GEMM、`P64` 复用 K1 的 8×8 前代+Schur 补求逆思路），此处数字是高估的
  上界、**待验证**。

### 13.4 两个只有真做重组才暴露的坑

1. **`ki16 = k_rest16 · 2^{−G16}` 会溢出 fp32**（`G16≲−58` ⇒ `2^{58}`，乘上 `kr` 后仍逼近 2^38 上限，
   再乘 `2^{−off}` 直接 inf）。本节的做法：`A64` **逐子块对**（`s_t ≥ s_u`）算，把对偏移
   `2^{off_{s_t}−off_{s_u}}` 折进**较小的因子**（`ki16 ⊙ 2^{≤0}`），于是任何中间量 ≤~2^58；这就是
   `dt=float32` 也能跑出与 fp64 逐项相同结果的原因。直接在 64 块上做 `kd64 @ ki64^T` 会 inf。
2. **非整除 T 的补零不能丢「累积进位」**：B0 输出的 `g_cum` 只到第 `T` 个 token，把它零填充到
   `NT16·16` 会让最后一个尾块的子块门控和 `G16=0`（错），进而污染**前一个完整子块**的重标定指数
   （指纹：`kg` 只在尾块前 4 个 token 大错、`max_rel≈0.83`）。须先补**携带值**（尾块 pad 的 `g=0`，故
   `g_cum` 停在末有效 token）再差分出逐 token 门控、在完整轴重算 cumsum。另：workspace 字段是
   `[B,NT,H,CHUNK,D]`（H 在 CHUNK 前），展平成 token-major 必须 `permute(0,1,3,2,4)`（同 §6.10 的坑）。

### 13.5 结论与下一步

- **可行性成立**：64-chunk 操作数可**免前向重算**地从 16-chunk workspace 恢复（`qg/kg/kd/ki/E64` 是逐
  子块重标定），B1@64 正确（bf16 底噪）且**快 2.1–2.3×**。这正式兑现了 §11.5 第 (2) 步的性能动机。
- **但代价与集成缺口明显**：(a) 恢复要**重算 64 块内逆 `P64/w64`**——正是 K1 在 16 块上做过的活，所以
  重组 B1 会**部分抵消**「复用 K1 workspace、不重算」的设计（§10.3 的卖点）；且本节 torch 原型恢复 ~ms、
  必须融成 Triton pass；**(b) 整链并不干净**：B2/B3 仍在 16 块上跑、要 16 边界处的 `dS`，而 B1@64 只在
  64 边界落 `dh`。要把这条链端到端改成 64，需要 B2/B3 一并重组（或让 B1@64 额外吐子块 `dS`）——这是
  真正的下一步，故本节只把结论钉在**孤立的 B1 重组**上。
- **优先级更新**：§11.5 第 (2) 步从「纯收益」细化为「B1 快 2.1–2.3×，但需 (i) 融一个 64 块重算 pass、
  (ii) 决定整链是否跟到 64」。在 (i)(ii) 落地前，第 (1) 步（吃原生 bf16）仍是唯一零算法改动的杠杆；
  第 (3) 步镜像 K2 warp specialization 依旧排最后。

## 14. 把 16→64 re-chunk 恢复融成一个 Triton pass（P7-11）

§13.5 把 P7-10 的硬伤定在「恢复本身」：torch 原型 `recover_operands_64` 用 ~2.0–5.7 ms 造 64-chunk
操作数，比 B1 本体（10–187 µs）大一个多量级，且要求「必须像 K1 那样融成一个 Triton pass（数字为高估
上界、待验证）」。本节兑现，并给出**不是上界、而是实测**的数字。新增 `harness/check_bwd_rechunk_pass.py`
（输出 `harness/check_bwd_rechunk_pass.out.txt`，tf32 变体 `harness/check_bwd_rechunk_pass_tf32.out.txt`，
空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 14.1 关键点：`P64` 不需要任何新的 16×16 求逆

§13.1 只把「唯一需要新算的」记成了 `P64`；但 `L64` 的**对角 16×16 块恰好就是 K1 已在 16 块里求过逆的
`L16`**——workspace 的 `INV` 字段即 $P16=(I+L16)^{-1}$（`docs/06` §6.2 已钉死 $Akk=P$）。于是
$P64=(I+L64)^{-1}$ 用 **4×4 块前代**从 workspace `P16`（对角）与 6 个非对角块 $L_{ij}$ 直接拼出，与 FLA
`merge_16x16_to_64x64_inverse_kernel`（`fla/ops/utils/solve_tril.py:200-344`）同构：

$$
P_{ii}=P_i,\quad
P_{10}=-P_1L_{10}P_0,\quad
P_{21}=-P_2L_{21}P_1,\quad
P_{32}=-P_3L_{32}P_2,
$$
$$
P_{20}=-P_2(L_{20}P_0+L_{21}P_{10}),\quad
P_{31}=-P_3(L_{31}P_1+L_{32}P_{21}),\quad
P_{30}=-P_3(L_{30}P_0+L_{31}P_{10}+L_{32}P_{20}).
$$

真正的新算只有 6 个非对角 $A_{ij}$ 与其 $\beta$ 掩码 $L_{ij}=A_{ij}\cdot\beta_j$（列 $u$）：

$$
A_{ij}=kd16_i\,\bigl(kr_j\cdot 2^{\,e_{ij}}\bigr)^{\!\top},\qquad
e_{ij}=\sum_{s=j+1}^{i-1}G16_s\le 0 .
$$

把对偏移 $2^{e_{ij}}$（**恒 ≤0**）折进**更小的** $kr_j$ 因子，就绕开了 §13.4 坑 (1)「$ki16=kr\cdot2^{-G16}$
溢 fp32」——任何中间量都被 $kr$ 界住，直接吃到 workspace 的 bf16 `kr`。$w64=P64\,\mathrm{diag}(\beta_t)\,kd64$
（$kd64_j=kd16_j\cdot2^{off_j}$）。以上全部落进**一个** Triton kernel `_recover64_kernel`：grid
`(NG, B*H)`，每 program 处理一个 64-group 的一个 head，把重标定（`qg/kg/g_cum64`）、6 个 $A_{ij}$、
6 个 merge 式、4 段 `w` 一次算完。

### 14.2 Part A：pass 产出 == torch 原型 & FLA fwd@64

四组 case（含非整除 `T=250/513`、`g=0`），pass vs torch 原型（同一批 workspace 字段 + B0 门控）：

| 量 | pass vs torch（fp32 代数） | pass vs FLA fwd@64 |
|---|---|---|
| `qg` | 8.05e-8 | 8.37e-3 |
| `kg` | 7.77e-6 | 7.05e-3 |
| `w` | 1.64e-2 | 1.10e-2 |
| `g_cum` | 1.12e-7 | 5.91e-3 |

`qg/kg/g_cum` 与 torch 原型到 1e-7~1e-6（同一条 fp32 代数，只差 `tl.dot` 的 ieee 舍入）；`w` 到 1.6e-2
**不是 bug**：pass 的 $P64$ 对角块复用 **bf16 落盘的 `P16`**，与 §13/P7-10 记的 `w` 1.7e-2 同源；对
FLA fwd@64 反而只 1.1e-2。全部落 bf16 底噪，**含非整除 T**。

### 14.3 Part C：把 pass 操作数喂 bf16 B1@64（端到端）

直接拿 pass 产出的 64 操作数喂 P7-9 的 bf16-mma B1 核（`BT=64`）与 FLA `dhu`@64 对拍：

| 梯度 | 最坏 max_rel（4 组 case） |
|---|---|
| `dh` | 1.44e-2 |
| `dv2` | 1.87e-3 |
| `dh0` | 1.20e-2 |

与 P7-10 在 **torch 原型操作数**上得到的结果同阶（dh 1.55e-2／dv2 1.87e-3／dh0 1.44e-2）——证明融成
pass 后语义未变，整条「workspace → 恢复 → B1@64」端到端仍落 bf16 底噪（含 `T=250/513` 非整除、`g=0`）。

### 14.4 Part B：性能——恢复不再是 ~ms 的硬伤

| case | pass dev (ieee) | pass dev (tf32) | torch 原型 full | pass/torch |
|---|---|---|---|---|
| T=512 H=4 | 41.2 µs | 17.2 µs | 2482.7 µs | **0.12** |
| B2 T=1024 H=8 | 117.7 µs | 62.9 µs | 2693.1 µs | **0.11** |
| B8 T=2048 H=8 | 952.8 µs | 491.9 µs | 5627.8 µs | **0.25** |
| T=250 H=3 非整除 | 175.5 µs | 16.2 µs | 2146.9 µs | **0.15** |

（`pass dev` = 仅 `_recover64_kernel` 的 CUPTI device 时间；`torch full` = `recover_operands_64` 的
wall。）**kernel 比 torch 原型快 4–9×**（0.11–0.25×）。把 `tl.dot` 的 `input_precision` 从 `ieee` 换
`tf32` 再快 ~2.5×（量表列），Part C 的 `dh/dv2/dh0` 逐位不变（操作数反正要 round 到 bf16 再进 B1）。
`pass wall`（298.8/300.6/1420.2/328.1 µs）比 `dev` 高一个量级的差，来自本研究里仍在 torch 侧做的
**gather/`permute`/`b0_gate`/G16 前缀和**；真实集成中这些融进 K1 落盘与 B0 门控重算 kernel，故 `dev`
才是 kernel 的贡献口径（同 §9「host 税」的教训）。

### 14.5 结论与下一步

- **兑现 P7-10 的「融成 pass」**：16→64 恢复从 torch 原型的 ~2.0–5.7 ms 降到 kernel-only 数十~数百 µs
  （ieee 41–953 µs、tf32 16–492 µs），**相对 B1@64 已不是硬伤**；§13 记的「2.0–3.7 ms」确系 torch
  逐算子 + `linalg.solve` 的 host/库开销，非算法成本。
- **设计上仍守住「复用 K1 workspace」**：pass 只读 workspace 六个字段 + B0 门控，$P64$ 的对角块直接
  复用落盘的 $P16$，**不重算任何 16×16 求逆**，也不碰 q/k 原始张量。
- **局限 / 下一步**：(a) `w` 精度被 **bf16 的 `P16`** 顶到 1.6e-2，若要更准需让 K1 落盘 fp32 `INV`（或
  重算对角块）；(b) §13.5 的 (ii) 仍未解——整链 B2/B3 还在 16 块上跑，要把整条反向端到端跟到 64 得让
  B2/B3 一并重组（或 B1@64 额外吐子块 `dS`），这是把本节 kernel 真正接进生产反向的前提。

## 15. 把整条反向端到端跟到 64——闭合 §13.5 (ii)（P7-12）

§14.5 把最后一个未解项钉死在 **§13.5 (ii)**：P7-10/P7-11 只让 **B1** 跑 64 块，而 B2/B3 仍在 16 块上，
B1@64 只在 64 边界落 `dh`，与 B2/B3 需要的 16 边界处 `dS` 对不上，所以「整链不干净」。本节把**整条反向**
（`dAv → B1 → B2 → B3 → 反向 cumsum(B0)`）**统一到 64 块**，且仍**只吃 FlashKDA 自己的 16-chunk K1
workspace**（不重算前向），从而把 (ii) 从「未解」变成「有明确答案」。新增
`harness/check_bwd_ws_reuse64.py`（`harness/check_bwd_ws_reuse64.out.txt` 正确性 +
`harness/check_bwd_ws_reuse64_timing.out.txt` 计时，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 15.1 只差四个操作数：把 64 块的前向量从 16 块 workspace 补齐

P7-11 的 fused pass 已给出 64 块的 `qg/kg/w/g_cum/P64`（§14.1）。B2/B3/`dAv` 还需要四个：

- **`h64`（chunk 输入状态 `S_in`）**：状态是**物理量、与分块粒度无关**，所以 64 边界状态就是 16 边界状态
  的取样：`h64[g] = h16[4g]`。16 级递推 `S_{c+1}=E^{tot}_c\odot S_c + k_{rest,c}^\top U_c` 与 P6-8 的
  `reconstruct` 同式，直接 `h16[:, 0::4]`。
- **`Aqk64`（intra 注意力矩阵 `M_qk`）**：块内对角块就是 workspace 落盘的 `Mqk`（tril）；非对角块与
  §14.1 的 $A_{ij}$ **同一条代数**，只是把 `kd16` 换成 `qd16`：

  $$A^{qk}_{ij}=qd16_i\,\bigl(kr_j\,2^{\,e_{ij}}\bigr)^{\!\top},\qquad e_{ij}=\sum_{s=j+1}^{i-1}G16_s\le 0 ,$$

  指数恒 ≤ 0、折进更小的 `kr_j`，绕开 §13.4 坑 (1) 的 fp32 溢出。
- **`Akk = P64`**：即 §14.1 的 4×4 块前代合并结果（复用落盘 `P16`，零新求逆），本节直接把整个矩阵落盘。
- **`U64 = v_new`**：块内 WY 的同一递推，换成 64 粒度：
  $U64= P64\,\mathrm{diag}(\beta)\,(v64-kd64\,h64)$（`kd64_j=kd16_j\,2^{off_j}`，$off\le0$）。

本节用 **fp64 torch 原型**重建这四个量（快速验证端到端正确性；§14.1 的 `_recover64_kernel` 已示范如何融成
单个 Triton pass）。非整除 T 一律零填充到 `Tp=NG·64` 并把整链按 `T=Tp` 跑（pad token 的 q/v/β/g 全 0，惰性），
FLA golden 吃同一份 pad 输入，比较 apples-to-apples。

### 15.2 Part A：重建出的 64 前向量 vs FLA forward@64

四组 case（含非整除 `T=250/513`、`g=0`），有效 token 上比较：

| 量 | 最坏 max_rel | 备注 |
|---|---|---|
| `qg` | 8.3e-3 | 逐子块重标定 |
| `kg` | 7.0e-3 | |
| `w` | 8.7e-3 | 吃 bf16 落盘 `P16`（同 §13/§14） |
| `g_cum` | 2.6e-3 | |
| `v_new (U64)` | 1.8e-2 | 需要 `P64/h64/kd64` 三者 |
| `Akk (P64)` | 3.0e-2 | 4×4 块前代合并 |
| `h (S_in)` | 6.9e-3 | `Tp` 之前的计算态 |

全部落 bf16 底噪，含非整除 T。`Aqk` **故意不入表**：FLA 把定义的（下三角 `M_qk`）写进 `[BT,BT]` tile 的一个
三角、另一个三角**未初始化**，且它 `q_dec @ k_inv^T` 的朴素写法在深衰减块会**溢 fp32**（正是 §13.4 要避开的
溢出），无法与 FLA 原始缓冲干净对拍——它的重建正确性由 Part B 端到端核验（`dAv`/B3 直接消费它）。

### 15.3 Part B：整链 @64 vs `torch.autograd`（含非整除 T / g=0 / B=2）

| 梯度 | 最坏 max_rel |
|---|---|
| `dq` | 9.8e-3 |
| `dk` | 7.9e-3 |
| `dv` | 1.3e-2 |
| `dbeta` | 6.0e-3 |
| `dg` | 7.7e-3 |
| `dh0` | 8.3e-3 |

六项全落 bf16 输入 / tf32 `tl.dot` 底噪，**与 P6-8 的 16 链同阶**（P6-8：`dq 3.4e-3…dh0 6.4e-3`）。
**至此整条反向已能统一在 64 块上从 K1 workspace 端到端跑通且正确，§13.5 (ii) 闭合。**

### 15.4 Part C：整链 @64 反而**慢 2.0–2.5×**——B1 的省被 B2/B3 的涨吃掉

把 workspace→64 链与 P6-8 的 workspace→16 链放在同一口径（event-pair `wall` + CUPTI `dev`，只计反向）：

| case | ours64 dev | ours16 dev | dev 64/16 |
|---|---|---|---|
| T=512 H=4 | 240.0 µs | 119.7 µs | **2.00** |
| B2 T=1024 H=8 | 856.2 | 432.2 | **1.98** |
| B8 T=2048 H=8 | 6124.3 | 2476.3 | **2.47** |
| T=250 H=3 非整除 | 202.6 | 80.0 | **2.53** |

逐 kernel 归因（µs/call，`T=512 H=4`／`B2 T=1024 H=8`）：

| kernel | @64 | @16 | 谁涨谁省 |
|---|---|---|---|
| `_dhu`（B1 状态伴随） | 20.8 / 55.0 | 45.7 / 119.9 | **@64 快 2.1–2.2×**（兑现 §13/§14） |
| `wy_dqkg`（B2） | 62.1 / 223.7 | 21.7 / 125.7 | @64 慢 **1.8–2.9×** |
| `intra`（B3） | 82.8 / 381.4 | 15.1 / 80.2 | @64 慢 **4.8–5.5×** |
| `_dav`（输出支路） | 7.0 / 15.6 | 3.5 / 11.1 | @64 慢 ~1.4–2.0× |

**为什么**：B1 是**跨块递推**，64 块把串行步数砍 4×，收益真实；而 **B2/B3 是块内 intra**，其工作
$\propto BT^2\times(NT=T/BT)\propto T\cdot BT$，把 `BT` 从 16 抬到 64 直接**乘 4**。两者方向相反，且 B2/B3
在 16 链里本就比 B1 大（`B2 T1024 H8`：16 链 B2+B3=205.9 vs B1=119.9），所以 **B1 的省被 B2/B3 的涨整个
吃掉还有余**。

### 15.5 结论：§13.5 (ii) 的答案——**不要把整链跟到 64**

- **能**：整条反向可以在 64 块上从 16-chunk workspace 免前向重算地端到端跑通，六项梯度落 bf16 底噪
  （§15.3），证明 §14 的重组 pass 与 B2/B3 的 64 变体能拼成一条正确、干净的 64 链。
- **但不应**：统一到 64 让整链反向 **慢 2.0–2.5×**（§15.4）。B1@64 的 2.1–2.2× 省被 B2/B3@64 的
  4–5× 涨盖过。**B1@64 的速度只有放进「B2/B3 也便宜」的 64 链才兑现，而 B2/B3 的 intra 工作随 `BT` 线性放大**，
  故 64 链是净亏。
- **故 §13.5 的两个选项里**：「B1@64 额外吐子块 `dS`」在 FLOPs 上等价于再跑一遍 B1@16（子块边界 $dS$ 的
  块内递推正是 16 级递推，见 §15.1 的 `dS` 链是仿射的 $\bar S\!\to\!(\mathrm{diag}(E)-w^\top k_{rest})\bar S$），
  **无净收益**；「B2/B3 一并重组到 64」本节点清是**净亏**。**因此工程结论：反向维持 16 块链**（P6-8/P7-9），
  B1@64 仅作为**孤立**的状态伴随加速手段（当 B1 确实主导时，例如更大的 `V` 或没有 K1 workspace 复用的场景）。
- **可复用性**：本节的 `h64`/`U64`/`Aqk64`/`P64` 重建（§15.1）在**任何**需要 64 块前向量的下游（不止 B1）
  都可直接吃，且全从 16 块 workspace 恢复——这把 §13/§14「64 操作数可免重算恢复」的结论从 B1 扩展到
  **全部**前向量。

### 15.6 四个只有把整链架到 64 才暴露的坑

1. **FLA `initial_state` 布局**：FLA 要 `[K,V]`，而本 harness 的 `h0` 是 FlashKDA 的 $S^\top$==`[V,K]`
   （P4-2）；对照 FLA forward@64 时必须 `h0.transpose(-1,-2)`，否则 `h[0]`（初始状态槽）与 FLA 恰好互为转置
   （指纹：`h` chunk0 rel≈1.47 而 chunk≥1 rel≈7e-3）。
2. **门控的 ln2 单位**：`fla_forward` 家族吃的是**自然对数**门（内部 `chunk_local_cumsum(scale=RCP_LN2)`
   转 log2），误当 log2 再乘一次 `ln2` 会让所有 64 量比 FLA 差一个均匀因子 `1/ln2≈1.4427`
   （指纹：`g_cum` 样本整体 ×1.4427）。
3. **非整除 T 的 reshape 截尾**：`gate_tok.view(B,NTp,C,H,D)` 里只有前 `NT16` 个 16 子块有定义，必须
   `gate_tok[:, :NT16].reshape(...)`，否则 `B2 T=513`（`NTp=36 vs NT16=33`）直接 shape 报错。
4. **FLA `Aqk` 原始缓冲不可直接对拍**：未初始化三角 + 深衰减溢出（§15.2），须靠整链（Part B）核验。

计时口径与 §9/§10 一致：`wall` 为 event-pair 包 30 次取中位；**`dev`（CUPTI）才是 kernel 贡献口径**，`wall`
里 harness 侧每调用重建 Triton kernel 会引入假 host 税（本节已把 kernel 构建挪出计时闭包）。

## 16. ncu 剖析反向瓶颈 B1：它到底 bound 在什么上（P7-13）

P7-2 §5 只对**前向** K1/K2 做过 `ncu`；整条反向链（§10–§15）一直只用 CUPTI 计时，从没量过硬件 bound。
而 §11.5/§14.5/§15.5 把「剩余工程化 = 把 B1 做成目标卡 CUDA、镜像 K2 的 warp specialization」立为下一步——
**动它之前得先知道 B1 现在卡在哪**。本节补上这块：用 `ncu` 剖析 B1（跨块状态伴随 / FLA `dhu`）的
三种形态，新增 `harness/ncu_bwd_driver.py` + `harness/probe_bwd_ncu.sh`
（原始输出 `harness/probe_bwd_ncu.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

被剖析的两个实现：

- **ours** = Triton `_dhu_kernel_bf16`（P7-8/P7-9 的 bf16-mma 克隆，`harness/check_bwd_dhu_dtype.py:62-107`），
  状态 `b_dh` 是**单个** `[K, BV]`（K=128）fp32 累加器（`:76`），每次递推三条 `tl.dot`（`:98,:101-103`），
  `num_warps=4, num_stages=1`（`harness/check_bwd_dhu_kernel.py:214`）。喂 workspace 形状的 bf16 操作数，
  `--bt 16`（FlashKDA `CHUNK`，实际部署）或 `--bt 64`（FLA 粒度）。
- **fla** = `chunk_gated_delta_rule_bwd_dhu`（FLA 原生 `chunk=64`）。

### 16.1 主表：五个配置的 SOL / 占用 / 停顿（`ncu`，B2 T1024 H8 / B1 T512 H4，bf16）

| 配置 | grid | regs | smem/blk | theo occ | ach occ | dur µs | tensor% | SM% | L1% | L2% | DRAM% | warp cyc/issue | long_sb | wait |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ours B1 BT=16（部署） B2T1024H8 | 64 | 128 | 22.5 KB | 25% | **6.3** | **97.22** | 4.5 | 6.3 | 38.5 | 17.9 | 13.0 | 8.19 | **3.88** | 1.30 |
| ours B1 BT=64 B2T1024H8 | 64 | 186 | 65.5 KB | 12.5% | 6.3 | **53.82** | 6.8 | 6.1 | 37.9 | 18.9 | 13.6 | 9.68 | **5.29** | 0.82 |
| FLA `dhu` BT=64 B2T1024H8 | 64 | 146 | 180.2 KB | 6.25% | 6.2 | **25.15** | 14.9 | 8.7 | 53.7 | 39.8 | 28.6 | 5.06 | **1.06** | 0.96 |
| ours B1 BT=16 B1T512H4 | 16 | 128 | 22.5 KB | 25% | 6.2 | **48.38** | 4.7 | 1.6 | 39.8 | 5.7 | 1.7 | 7.80 | 3.53 | 1.29 |
| FLA `dhu` BT=64 B1T512H4 | 16 | 146 | 180.2 KB | 6.25% | 6.2 | **14.78** | 14.2 | 2.0 | 54.8 | 12.7 | 5.4 | 4.98 | 0.91 | 0.96 |

（`long_sb`/`wait` 是 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio`，单位「每次发射停顿的指令数」。
ncu 的 `Duration` 与 P7-8 的 CUPTI 口径数值基本吻合：ours-BT16 = 97.2µs（P7-8 bf16 报 97.9）、
ours-BT64 = 53.8（P7-8 报 41.8）、FLA = 25.2（P7-8 报 23.0）。）

### 16.2 结论一：B1 三种形态**都不是任何硬件单元的 bound**——它是「并行度 / 延迟」受限

- **不是 TensorCore bound**：ours tensor pipe 只 **4.5–6.8%**，FLA 14.2–14.9%；SM throughput 1.6–8.7%。
- **不是带宽 bound**：DRAM 最高 28.6%、L2 最高 39.8%，都远未饱和（对比前向 K2 是 L2 94% SOL，§8.4）。
- **也不是常规意义的占用率 bound**：theoretical occupancy 12.5–25%（ours）/ 6.25%（FLA），
  而 **achieved occupancy 三种形态一律 ~6.2%**——它们被**同一个东西**钉住：**grid 太小**。
  主表里 grid 只有 **64 个 block（B2H8）或 16 个（B1H4）**，而卡有 **132 个 SM**：
  `ncu` 明确报「grid 只覆盖 0.48 wave / 0.12 wave，Est. Local Speedup 51.5%」。
  B1 的 grid 结构是 `(cdiv(V,BV), B·H)`：并行度 = `B·H·(V/BV)`。$V{=}128$、$BV{=}32$ 时就是 `B·H·4`。
  **沿 T 的跨块递推是串行的**（`for i_t in range(NT-1,-1,-1)`），唯一能加的并行只有切 V（BV）和切 K。
  所以「ach occ 6%」是 **grid 小的症状，不是可调的旋钮**——`ncu` 那条 51.5%/74.9% 的 occupancy
  建议和前向 §8.4 的 `Est. Local Speedup` 一样是**纯启发式**（低占用率在这里是并发不足的结果，
  而不是原因）；强行拆 BV 只会把 tile 切碎、得不偿失。

### 16.3 结论二：ours 与 FLA 的差距 = 长记分牌停顿（没有 load 流水）+ 单一依赖链

在**同 BT=64** 下 ours 比 FLA 慢 2.14×（53.82 vs 25.15）；差在哪看两个数：

- **停顿构成**：ours 每次发射要等 **8.19（BT16）/9.68（BT64）** cycle，其中 **long_scoreboard 占
  3.88/5.29**（≈47%/55%）——即一半以上的等待是 `tl.dot` 在等 `k/q/w/do/dv` 的**全局加载**回来；
  FLA 只有 **1.06/5.06 ≈ 21%**。`ncu` 对 ours-BT16 直接给出「long scoreboard 占 47.4% issue 间隔、
  Est. Speedup 47.4%」。
- **依赖链宽度**：ours 的状态是**单个** `[128, BV]` 累加器，每步 `b_do`/`b_k`/`b_q`/`b_w` 都流进**同一条**
  `b_dh = b_dh*e + q^T do - w^T du` 链；FLA 把 K=128 拆成**两块 64**（`chunk_delta_h.py:401-415`
  的 `b_dh1`/`b_dh2`，各自独立更新于 `:585-604`，`b_dv` 由两条链相加 `:532-544`），于是有**两条独立
  累加链**可以交替发射。这正是 FLA `tensor%` 14.9 vs ours 6.8、`warp cyc/issue` 5.06 vs 9.68 的来源。

而 ours 之所以在等内存，是因为它**没有对分块循环做软件流水**：`num_stages=1`
（`check_bwd_dhu_kernel.py:214`），每轮的 5 个 `tl.load` 与紧随的 `tl.dot` 之间没有预取下一块的重叠。
FLA 的 `dhu` 用 `@fla_cache_autotune` 在 **`num_stages ∈ {2,3,4}`**（`chunk_delta_h.py:350-352`）
上自动调参，把 180 KB 动态 smem 全用在多级流水缓冲上——**代价是 occupancy 只有 6.25%（Block Limit
Shared Mem = 1，`chunk_delta_h.py` 的 `BV=32` 配置）**，但它照样更快，再次印证 B1 的瓶颈是**延迟而非占用率**。

### 16.4 结论三：BT=16 → BT=64 的 1.81× 来自「串行步数 ×4」和「`dh` 落盘 ×4」

ours 自身 BT=16 比 BT=64 慢 **1.81×**（97.22 vs 53.82）。两个可量化的机制：

- **串行步数**：NT = 64（BT=16）vs 16（BT=64），每步一轮加载 + 一轮三条 `tl.dot`，步数 ×4；
  但因为每步 tile 也小 4×、部分延迟被重叠，净效果不是 4× 而是 1.81×（寄存器反而从 128 涨到 186，
  因为 BT=64 的 `[BT,K]` 操作数更大、活跃区间更长）。
- **`dh` 落盘量**：`dh` 形状 `[B,NT,H,K,V]`，BT=16 时 NT ×4 ⇒ **DRAM 写 20.80 MB vs BT=64 的 3.43 MB
  （6.1×）**，`lts__t_bytes` 96.95 MB vs 57.56 MB（1.68×）。BT=16 那 1.7× 的额外 L2/DRAM 流量几乎全是
  给 B2 逐 chunk 落 `dh` 的 ×4 冗余（B2 只吃 16 边界；§15 已论证整链跟到 64 不划算）。

### 16.5 对目标卡 CUDA B1 的设计输入

把上面三条翻成「该怎么做」：

1. **流水化加载，而不是提高占用率。** B1 的收益来自把「下一块的 k/q/w/do/dv 预取」与「本块三条 `tl.dot`」
   重叠，正对 ours 那 47–55% 的 long-scoreboard。目标卡 CUDA 版应直接镜像前向 **K2 的 TMA 输入流水**
   （`docs/03` §8/§10，`utils.cuh:86-143` 的 `PipelineTmaAsync`）：逆序沿 chunk 走、TMA 预取、MMA
   消费。这是 **`docs/06` §5.2 早就写下的设计**，本节给出了它的**硬件动机量化**。
2. **把 D=128 状态拆成 2 条（或 4 条）独立累加链**，增加 ILP——FLA 用 K 两分（`:401-415`）已见效果；
   K2 的 `s_acc` 本身是 `D×D` 整块，B1 可沿 K 或 V 二分。
3. **不要为「小 shape 占用率」去拆 BV**：grid = `B·H·(V/BV)`，小 `B·H` 时（T512H4 grid=16）本就填不满，
   但那时 B1 的绝对时间也小、不是端到端瓶颈；大 `B·H` 时（如 P7-10 的 B8T2048H8，grid=256）grid 已过
   132。优先做 (1)(2)，它们对所有 shape 都收益。
4. **维持 16 块链**：BT=64 省下的 B1 时间（1.81×）会被 `dh` 落盘的省与 B2/B3@64 的涨（§15.4）抵消，
   §15.5 的结论不变。

### 16.6 口径与坑

- `ncu` 会 replay 并清 L2，绝对时间和 CUPTI 略有出入（本表与 P7-8 的 CUPTI 误差 ≤1.3×，结论一致）；
  跨实现**相对比较**在同一 `ncu` 运行内才可靠。
- `ncu` 的 occupancy / 「grid too small」建议是**启发式**，对「串行递推 + 天生小 grid」的 kernel 会误报
  加速空间（同前向 §8.4）——必须结合 `long_sb`/`lts`/`dram` 这些**实测计数**判断真实 bound。
- FLA 的 `num_stages` 实际取值被封在 `Heuristics` 包装后、`ncu` 从动态 smem（180.22 KB / block）推断其
  ≥2；未直接读出具体 stage 数（**待验证**，不影响结论）。

## 17. 在 Triton 原型上先验证 §16.5 的两条 B1 设计输入（P7-14）

§16.5 把「把 B1 做成目标卡 CUDA」拆成四条设计输入，其中前两条是并列的：

1. **流水化加载**（镜像前向 K2 的 TMA 输入流水），吃掉 ours 那 47–55% 的 `long_scoreboard`；
2. **把 D=128 状态拆成 2 条独立累加链**（FLA 把 K 两分，`fla/ops/common/chunk_delta_h.py:401-415`），增加 ILP。

动 CUDA 重写之前，先花很小代价在**现有的 bf16 Triton B1 原型**上把这两条各自值多少问清楚——
新增 `harness/check_bwd_b1_ilp.py`（对拍+计时，原始输出 `harness/check_bwd_b1_ilp.out.txt`）
与 `harness/probe_bwd_ilp_ncu.sh` / `harness/ncu_bwd_driver.py --variant`（`ncu`，原始输出
`harness/probe_bwd_ilp_ncu.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

四个变体（`_dhu_ilp`，与 P7-8 的 `_dhu_kernel_bf16` 逐式相同，只加了两个 `tl.constexpr` 开关）：

| 变体 | 说明 |
|---|---|
| **V0 base** | 单条 `[K=128,BV]` 累加链，循环 `num_stages=1`（= P7-8 基线） |
| **V1 pipe** | 单链，但用 `tl.range(..., num_stages=NS)` 对逆序块循环做**软件流水** |
| **V2 splitK** | 状态拆 2 条 `[64,BV]` 链（K 两分），`num_stages=1` |
| **V3 both** | 拆链 **且** 流水 |

### 17.1 Part A：四个变体数值上与 FLA `dhu` 同阶（含非整除 T）

三个 shape（含非整除 `T=250`）在 BT=64 下逐变体对 FLA `dhu`：`dh/dv2/dh0` 的 `max_abs` 与基线
**完全一致或更好**（如 `B=2 T=1024 H=8`：`dh 2.441e-4`、`dv2` 拆链后 2.441e-4→**1.221e-4**、
`dh0` `1.658e-7`→**7.451e-9**）。即两个开关**不改变代数**、只改变调度——`num_stages` 只影响
预取、不影响结果（各 ns 的 `max_abs` 到打印精度相同），拆链只是把 K 维点积换成两个半宽点积。

### 17.2 Part B：流水是**唯一**的杠杆，拆链单独没用

`B2 T1024 H8`、bf16、内核-only CUPTI（`us/call`），FLA `dhu` @BT=64 = **23.1**：

| 变体 | BT=32 (xFLA) | BT=64 (xFLA) |
|---|---|---|
| V0 base ns=1 | 73.0 (1.95) | 45.2 (1.96) |
| **V1 pipe ns=2** | 46.5 (1.24) | 27.3 (1.18) |
| **V1 pipe ns=3** | **32.3 (0.86)** | **23.0 (1.00)** |
| V2 splitK ns=1 | 74.0 (1.98) | 51.1 (2.21) |
| V3 both ns=3 | 39.5 (1.06) | 25.6 (1.11) |

三个 shape 的趋势一致（`B1 T512 H4` BT=64：base 17.6 → pipe ns=2 **12.7** → xFLA 0.97；
`B8 T2048 H8` BT=64：base 162.5 → pipe ns=3 **102.2** → xFLA 0.85）。读出来的结论很干净：

- **流水一个开关就吃掉了整个 2× 缺口**：`B2T1024H8` @BT=64 从 45.2µs（1.96×）降到 23.0µs（**1.00× FLA**），
  `B8T2048H8` 甚至 **0.85×**（反超），`B1T512H4` 0.97×。§16.3 把差距归到「没流水 + 单链」，这里直接
  证明**「没流水」是全部**。
- **拆链单独是负收益**：V2（`B2`@BT=64）= 51.1µs，比 base 的 45.2 **还慢 13%**；它想加的 ILP
  在 `num_stages=1` 下根本填不满（两条半宽 `tl.dot` 反而多了发射开销与寄存器）。
- **两者一起反而更慢**：V3 ns=3 = 25.6µs > V1 ns=3 的 23.0µs——流水已经把 ILP 预算填满，再拆链
  只是重复搬运/多余点积。

部署粒度 BT=16（操作数从 FlashKDA K1 workspace 的 `reconstruct` 取并转 bf16；FLA intra 无 BT=16 核，
故只与自身 base 比）：`B2T1024H8` base 99.0 → pipe ns=3 **50.9µs（1.96×）**、
`B1T512H4` 36.1 → **26.0µs（1.39×）**，同样由流水独得。

### 17.3 `ncu` 佐证：流水把 `long_scoreboard` 压到 FLA 水平（B2T1024H8 BT=64）

`harness/probe_bwd_ilp_ncu.sh`（`ncu -k regex -c 1 --launch-skip 3`，逐配置单独 profile）：

| 配置 | dur µs | long_sb | barrier | wait | tensor% | L2% | DRAM% | regs | smem/blk | theo occ |
|---|---|---|---|---|---|---|---|---|---|---|
| base ns=1 | 53.6 | **5.27** | 1.71 | 0.82 | 6.8 | 19.0 | 13.5 | 186 | 65.5 KB | 12.5% |
| pipe ns=2 | 31.3 | **1.46** | 1.34 | 0.64 | 13.0 | 32.1 | 23.1 | 154 | 123.4 KB | 6.25% |
| pipe ns=3 | **23.6** | **0.96** | 1.05 | 0.60 | 15.9 | 42.2 | 30.6 | 152 | 181.3 KB | 6.25% |
| splitK ns=1 | 60.0 | **6.88** | 1.59 | 0.88 | 6.0 | 17.1 | 12.0 | 168 | 61.4 KB | 18.8% |
| both ns=3 | 26.1 | 0.77 | 1.10 | 0.75 | 14.1 | 37.9 | 27.4 | 128 | 181.3 KB | 6.25% |
| （§16.1 FLA `dhu`） | 25.2 | 1.06 | — | 0.96 | 14.9 | 39.8 | 28.6 | 146 | 180.2 KB | 6.25% |

- **因果链闭合**：`long_sb` 5.27 → 1.46（ns=2）→ **0.96（ns=3）**，正好落到 FLA 的 1.06；`tensor%`
  6.8→15.9（FLA 14.9）、DRAM 13.5→30.6%（FLA 28.6%），都是「加载到得更多、算得更满」的体检读数。
- **拆链不改善停顿**：V2 的 `long_sb=6.88`（比 base 的 5.27 还高）。
- **占用率再次被证伪为旋钮**：流水把 smem 从 65.5 KB 抬到 **181.3 KB（≈FLA 的 180.2 KB）**，
  理论占用率从 12.5% 掉到 **6.25%**，却最快——与 §16.2「B1 是延迟/并行度受限、不是占用率受限」一致。

### 17.4 结论：把 §16.5 的前两条改成「只做 (1)」

- **§16.5 的第 (2) 条（拆 2 条累加链）在本原型上是冗余的**：`tl.range` 软件流水一旦开起来就已经把
  ILP 填满，拆链单独负收益、叠加更慢。这条从「并列的设计输入」降级为「不推荐」。
- **第 (1) 条（镜像 K2 的 TMA 输入流水）是全部收益来源**：单靠它就把 ours B1 从 1.96× FLA 拉到
  **1.00× 甚至 0.85×**，`long_scoreboard` 从 5.27 压到 0.96（=FLA 1.06）。
- **边界（待验证）**：这是 **Triton 自动流水器**的结论；手写 CUDA/TMA + warp-specialization 版把
  预取显式化后，是否还需要手工拆链以进一步增 ILP，须在目标卡上再测。但就现有证据，**目标卡 CUDA B1 的
  第一优先级就是「照抄 K2 的 TMA 输入流水」**（`docs/03` §8/§10，`utils.cuh:86-143`），拆链可以暂缓。

## 18. 用**手写 CUDA** B1 补 §17 的边界（P7-15）

§17 的结论（「流水是唯一杠杆，拆链冗余」）是在 **Triton 原型**上量的，B1 写到一半时自动软件流水器
已经把预取、寄存器分配全接管了。§17.4 明确留了一条边界：**手写 CUDA/TMA 把预取显式化后，是否还需要
手工拆链增 ILP？** 本节用一支**纯 CUDA**（非 CUTLASS、非 Triton）的 B1 原型回答它——先验证「手写 CUDA
需要哪些结构才让 §16.5 的流水杠杆兑现」。

### 18.1 原型：`harness/b1_cuda_probe.cu`

- 复刻 `check_bwd_dhu_kernel.py` 的递推（逐 (n,h)、逐 V-列块，逆序 chunk）：`du = k@S + dv`、
  `dv2=du`、`S = S*e + scale*(qᵀ do) - wᵀ du`（`fla/ops/common/chunk_delta_h.py:576-587` 同构）。
- mma 用 `nvcuda::wmma` 的 **bf16 m16n16k16**，运行态 `S` 保持 **fp32**（mma 累加 fp32），与 FLA 同路径。
- 网格 `(V/32, B·H)`，blockDim = **2 warps**，每个 warp 独立负责 **16 列**（warp 内 `S` 是 `[K=128,16]`，
  按 8 个独立的 mma m-tile 组织——**K 维天然被拆成 8 条互不依赖的累加 tile**，这正是 §17「拆链冗余」在
  CUDA 里的对应实现，无需跨 warp 归约）。
- 唯一的超参：`NSTAGE` = `cp.async` 预取深度（1 = 发完即等、无重叠；2/3 = 预取 1/2 个 chunk），
  ring buffer 按**处理序** `i % NSTAGE` 选址。运行为 `harness/check_bwd_b1_cuda.py`（ctypes 调
  `extern "C" b1_launch`），输出 `harness/check_bwd_b1_cuda.out.txt`。

### 18.2 Part A：与 FLA `dhu` **逐位相同**（含非整除 T）

| shape | BT | `dh` | `dv2` | `dh0` |
|---|---|---|---|---|
| B1 T512 H4 | 32 / 64 | **0** | **0** | **0** |
| B2 T1024 H8 | 32 / 64 | **0** | **0** | **0** |
| B1 T250 H3 | 32 / 64 | **0** | **0** | **0** |

`max_abs = 0`（`ref_max` 9.7e-2–1.2e-1）。这不是近似一致，而是**位级相同**：两侧都是 bf16 mma + fp32
累加，且**逐条运算顺序相同**（先 `k@S` 再加 `dv`；先 `scale*qᵀ do` 再减 `wᵀ du`），故浮点舍入逐位复现。
含**非整除 T=250**（尾块 mask 惰性）。这再次钉死了 B1 的运算契约：一个独立手写实现，只要 op 顺序与
FLA 一致，就能位级对齐——说明 §16/§17 的差异全是**调度/结构**，不是代数。

### 18.3 Part B：性能——本原型的流水**只有 ~2–6%**，且大 shape 反而变慢

kernel-only（CUPTI）`us/call`：

| shape | BT | FLA `dhu` | NS=1 | NS=2 | NS=3 |
|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 20.6 | 137.2 (1.00) | 134.6 (0.98) | 135.8 (0.99) |
| B1 T512 H4 | 64 | 12.9 | 106.8 (1.00) | 104.6 (0.98) | 104.5 (0.98) |
| B2 T1024 H8 | 32 | 37.2 | 284.4 (1.00) | 265.2 (0.93) | 267.8 (0.94) |
| B2 T1024 H8 | 64 | 22.7 | 217.4 (1.00) | 205.9 (0.95) | 206.3 (0.95) |
| B8 T2048 H8 | 32 | 170.2 | 729.3 (1.00) | 687.6 (0.94) | **1059.9 (1.45)** |
| B8 T2048 H8 | 64 | 119.9 | 574.1 (1.00) | **806.6 (1.40)** | 815.8 (1.42) |
| B2 T1024 H8 | 16 | — | 416.6 (1.00) | 390.6 (0.94) | 382.3 (0.92) |

两件事：

1. **绝对速度差 FLA 4.3–9.6×**（BT=16 无 FLA `dhu` 可比）。本原型远不是「load-bound」的——3 个 GEMM 每
   chunk 只 ~0.79 M MAC，`B1 T512 H4` 实测 ~1.9 TFLOP/s（H100 bf16 峰值的 ~0.2%），是**延迟/发射受限**：
   每 CTA 只有 2 warp、smem ~108KB（NS=1）→ 每 SM 仅 2 warp，几无延迟隐藏；grid 小（16–256 CTA）。
2. **`NSTAGE` 预取只值 ~2–6%**（小 shape 0.92–0.98×），**大 shape 反而 1.40–1.45× 变慢**：`B8` NS=2/3
   的 smem 到 168–226KB，占用率从 2 CTA/SM 掉到 1，预取省下的等待被占用率损失吃掉。smem 足迹：
   `BT=16/32/64`、`NS=1` = 58.9/76.3/111.1 KB，`NS=3` = 88.6/134.7/226.8 KB。

### 18.4 结论：§16.5/§17 的「流水杠杆」是**有条件的**

- **本手写 CUDA B1 不复现 §17 的 Triton 结论**（Triton 靠 `tl.range(num_stages)` 把 B1 从 1.96× 拉到
  1.00×，这里只值 ~2–6%）。原因不是流水无效，而是**本原型根本没到 load-bound**：2 warp/CTA、低占用、
  每 chunk 5 次 `__syncthreads`、wmma 从 smem 逐 tile 取数——瓶颈在**占用率/发射/同步**，不在等 global。
- 因此 §17.4 的边界应细化为：**「镜像 K2 的 TMA 输入流水」不是一个可单独加上的开关**——它只有在 kernel
  **已经是 K2 形状**（warp-specialized、TMA 大块搬运、占用率足够）时才成为「全部收益来源」。裸 wmma +
  `cp.async` 的原型得先把结构做到 K2，否则流水只是 2–6% 且会因为吃掉 smem 而反噬。
- 「拆链」在 CUDA 里**无从单独消融**：warp 内 `[K,16]` 状态本就是 8 个独立 mma m-tile，没有 Triton 那种
  「单一 `[128,BV]` 数据流依赖链」。这与 §17「Triton 拆链冗余」一致（CUDA 结构天然拆好了）。
- **待验证**：本原型是**研究用最小实现**，非优化实现（wmma-from-smem、无 ldmatrix/TMA、每 chunk 5 sync、
  2 warp/CTA），故 `xFLA` 的绝对数字不代表目标卡 CUDA 的天花板；可下结论的只是**定性**——裸 wmma 版
  不是 load-bound，流水杠杆要到 K2 结构才兑现。

### 18.5 两个只有手写 CUDA 才暴露的坑

1. **warp-local 的 smem 切片必须用 warp 内 lane（`lane&31`，步长 32）索引**，不能沿用 block 级
   `lane` + `blockDim.x`。本项目里 `S/dh`、`duf`、`du` 都是**每 warp 私有**的 `[.,16]` 切片，若用
   `idx = lane; idx += blockDim.x(=64)` 遍历，每个 warp 只覆盖自己切片里 `≡0..31 (mod 64)` 的一半元素，
   另一半永不写。指纹：**全零输入（Case A）通过**，但 `dv2/dh0` 在未写区域留 0、与参考差 O(1)
   （实测 col 全错、`argmax` 处 ours=0 而 ref=−4.2）。改成 `idx = wl; idx += 32` 即逐位归零。
2. **ring buffer 的 stage 号必须按「处理序」算（`i % NSTAGE`），不能按 chunk 号（`c % NSTAGE`）**：
   否则 `NSTAGE≥2` 时会去读一个**尚未装载**（或已被覆盖）的 stage——`NSTAGE=1` 时两者恒等，把 bug 藏住。
   指纹：只有 `NSTAGE=2/3` 结果错、`NSTAGE=1` 正确。

## 19. ncu 剖析**手写 CUDA B1**（P7-16）：把 §18.4 的「非 load-bound」变成定量

§18 只做了**计时**：裸 wmma CUDA B1 的 `cp.async` 预取只值 ~2–6%，且大 shape 反被 smem 占用反噬。
§18.4 由此*定性*断定「瓶颈在占用率/发射/同步，不在等 global」——但**没有剖析过这个 kernel**（P7-13/§16
剖析的是 Triton B1，§17/§18 都是计时）。本节用 `ncu` 把它的硬件 bound 钉死，并**修正其中的一个猜测**。

### 19.1 设置

`harness/probe_b1_cuda_ncu.sh`（输出 `harness/probe_b1_cuda_ncu.out.txt`）在同一空闲卡上剖析 7 个
单次 launch。每个配置**先在 ncu 外预编译**（`nvcc` 在 ncu 下会挂起），再 `--no-build` 进 ncu（`-k regex`,
`-c 1 --launch-skip 3`）。驱动 `harness/ncu_b1_cuda_driver.py` 的 `ours` 路径**直接合成** workspace 形状
操作数、不碰 FLA/Triton（后者一次 setup 会拉起几十个 kernel，在 ncu 下把 wall time 撑爆）：

- ours-CUDA `b1_dhu_cuda`：BT=64 NS=1 / NS=3 / `-DB1_SYNC`（把 `cp.async` 换成同步 smem 写）/ BT=16；
- FLA `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`（其原生 BT=64）作为参照。

### 19.2 SpeedOfLight / 占用 / 资源

| 配置 (T512 H4 除注明) | Duration µs | tensor% | fma% | L1/TEX% | DRAM% | reg | smem KB | theo/ach occ | waves |
|---|---|---|---|---|---|---|---|---|---|
| ours BT=64 NS=1       | 111.6 | 2.28 | 2.78 | 42.9 | 0.71 | 106 | 111.1 | 6.25 / **3.11** | 0.06 |
| ours BT=64 NS=3       | 107.2 | 2.39 | 2.87 | 45.1 | 0.74 | 104 | 226.8 | 3.12 / 3.12 | 0.12 |
| ours BT=64 NS=1 SYNC  | 183.3 | 1.38 | 1.64 | 27.9 | 0.43 | 128 | 111.1 | 6.25 / 3.13 | 0.06 |
| **FLA `dhu` BT=64**   | **15.0** | **14.09** | 7.37 | 54.3 | 5.29 | 146 | 180.2 | 6.25 / 6.17 | 0.12 |
| ours BT=16 NS=1       | 208.8 | 1.25 | 2.85 | 33.3 | 0.39 | 64 | 58.9 | 9.38 / 3.13 | 0.04 |
| ours BT=64 NS=1 (B8 T2048 H8) | 587.5 | 3.47 | 4.13 | **65.3** | 13.5 | 106 | 111.1 | 6.25 / 6.09 | 0.97 |
| FLA `dhu` BT=64 (B8 T2048 H8) | **120.0** | **14.38** | — | 41.8 | **64.9** | 146 | 180.2 | 6.25 / 6.19 | 1.94 |

两件事立刻可见：(a) **两侧的理论占用率都是 6.25%**（都被 smem 钉在 1–2 block/SM），achieved 也都在
3–6%——**占用率不是 ours 落后的原因**（大 shape ours 已 0.97 wave、achieved 6.09% 几乎与 FLA 6.19% 持平，
仍慢 4.9×）；(b) **FLA 的 tensor 吞吐是 ours 的 6×**（14.1% vs 2.28%）。大 shape 两者同样是「谁 bound」：
**FLA 撞 DRAM（64.9%，2.17 TB/s）**，而 **ours 撞 L1/TEX（65.3%）却只有 13.5% DRAM**——ours 把内存管线
的预算烧在 shared-memory 重放上，不是 DRAM 带宽。

### 19.3 stall 组成（每发射间隔的 warp 平均 stall 拍数）

| 配置 | `wait` | `short_sb` | `long_sb` | `barrier` | `lg_throttle` | bank-conflict / shared-load wavefronts |
|---|---|---|---|---|---|---|
| ours BT=64 NS=1       | **1.80** | 0.87 | 0.66 | 0.02 | 0.33 | 827520 / 1083520 = **76.4%** |
| ours BT=64 NS=3       | **1.83** | 0.77 | 0.57 | 0.02 | 0.31 | 830287 / 1086287 = **76.4%** |
| ours BT=64 NS=1 SYNC  | **1.87** | 0.75 | **3.96** | 0.09 | 0.37 | 827392 / 1083392 = 76.4% |
| FLA `dhu` BT=64       | 0.96 | 0.58 | 0.91 | 0.96 | 0.16 | 154 / 18586 = **0.8%** |
| ours BT=16 NS=1       | 1.65 | **1.03** | 0.96 | 0.04 | 0.00 | 853504 / 1330688 = 64.1% |
| ours BT=64 NS=1 (B8 T2048 H8) | **1.82** | 1.56 | 0.87 | 0.04 | 0.92 | 53096676 / 69382372 = **76.5%** |
| FLA `dhu` BT=64 (B8 T2048 H8) | 0.96 | 0.49 | **2.19** | — | — | 2411 / 1083755 = 0.2% |

`wait` = 等一条**固定延迟**执行依赖（这里是 mma/wmma 的流水延迟）；`long_sb` = 等**全局加载**；
`short_sb` = 等 **shared/L1 操作数取数**。

### 19.4 三个结论

1. **全局加载早已被 `cp.async` 隐藏，不是瓶颈**。NS=1→NS=3 只把 `long_sb` 从 0.66 挪到 0.57
   （预取几乎无事可做），而**把 `cp.async` 换成同步写（SYNC）后 `long_sb` 暴涨到 3.96、Duration 从 111.6
   涨到 183.3 µs（+64%）**——即 `cp.async` 确实在干活、且已经把可隐藏的加载延迟吃干净了，剩下的只是
   1.65–1.87 拍的 `wait`（mma 延迟）。这从 stall 侧**证实了 §18 的「非 load-bound」**，也解释了 NS 只值
   2–6%：没有 global 等待可省。
2. **瓶颈也不是同步**。§18.4 猜的「占用率/发射/同步」里，**`barrier` 只占 0.02–0.09 拍**（每 chunk 5 次
   `__syncthreads`，但 2 warp 下代价可忽略）——**这个猜测要更正**。真正的 grinder 是 **`wait`（1.80–1.87，
   约占 5.0 拍发射间隔的 36%）+ `short_sb`（0.87–1.56）**，也就是**每 warp 只有 1 条 mma 依赖链、每
   scheduler 只驻 1 个 warp**，延迟无处躲（`Eligible Warps Per Scheduler` 仅 0.15–0.21、`No Eligible` 80–85%）。
3. **shared-memory bank conflict 是 ours 独有的、量级上的浪费**。ours 每次 chunk 用 `wmma::load_matrix_sync`
   从 smem 逐 16×16 tile 取操作数（`k`/`q`/`w` 行 stride `KD=128` bf16=256B、`Sb` stride `BW=16` bf16=32B），
   **76% 的 shared-load wavefront 是冲突重放**（small 827K/1.08M；big 53.1M/69.4M），而 FLA 用 ldmatrix 只有
   **0.8%/0.2%**。这解释了为什么大 shape ours 撞 L1/TEX（65%）而 FLA 撞 DRAM：ours 的内存管线被 smem
   重放占满。再叠加 FLA 的 tensor 密度（14% vs ours 2.3%）与 `wait` 更低（0.96 vs 1.80），构成 4.9–7.4×
   的差距。

### 19.5 对目标卡 CUDA B1 的设计输入（修正 §18.4）

§18.4 把待办概括为「把结构做到 K2」。ncu 把「K2 结构」具体化成**两条并列的必要条件**，且**顺序不依赖
「先加流水」**：

1. **换掉 wmma-from-smem**：用 `ldmatrix`/TMA 的整块操作数（消 76% bank conflict），并提高每 chunk 的
   mma 密度（FLA 的 tensor 14% vs ours 2.3%）——这是**量级项**，不是流水能替代的。
2. **`cp.async`/TMA 预取只是「不再退化」的保底**（SYNC→cp.async 省 39%），**不是收益来源**——与 §17 的
   Triton 结论相反，因为裸 wmma 版的 global 等待本就被 5 次同步+小 tile 掩盖。只有当结构已是 K2
   （大 tile + ldmatrix + 足够 warp）时，把预取升到 TMA 才像 §17 那样值钱。

这**再次印证 §18.4**：流水杠杆**不能单独加上**；但它也**修正**了 §18.4 的归因——不是「占用率/同步」，
而是**mma 依赖延迟（`wait`）+ smem 取数（`short_sb`/bank conflict）**。占用率与 FLA 同为 6.25%、barrier
0.02，都不是抓手。

> 说明：本节用 ncu 单次 replay 的 `Duration`，与 §18 的 CUPTI 均值同量级（small ours 111.6 vs 106.8、
> FLA 15.0 vs 12.9；big ours 587.5 vs 574.1、FLA 120.0 vs 119.9），差异来自 replay 与首次冷启动。
> 仍然：本原型是**研究用最小实现**，绝对数字不代表目标卡天花板；可下的是**定性/归因**结论。

## 20. 隔离 CUDA B1 的 bank-conflict 项：smem padding 受控实验（P7-17）

§19 用 `ncu` 在**手写 CUDA B1**（`harness/b1_cuda_probe.cu`）上量到 **76% 的 shared-load wavefront
是 bank-conflict 重放**（FLA `dhu` 用 `ldmatrix` 只有 0.8%），§19.5 据此把「换掉 wmma-from-smem
（ldmatrix/TMA 整块 + 提 mma 密度，消 bank conflict）」列为**量级项**、目标卡 CUDA B1 的第一优先级。
本节用**最便宜的受控实验**检验「bank conflict 到底值多少」——不改内核结构，只把 smem 的行 stride
**各加 8 个 bf16（16 B）**（编译宏 `B1_PAD`），让 16×16 操作数 tile 的每一行落在不同 bank 上。

### 20.1 为什么 padding 就能消冲突

原版 wmma 从 smem 取操作数，bf16 行 stride = `KD=128` 元素 = **256 B = 64 word ≡ 0 (mod 32)**，
所以一个 16 行的 tile **每行都从同一个 bank 起步**——`ldmatrix`/`LDS` 一笔最多服务一个 bank，16 行
全撞 ⇒ 重放。每个 warp 自己的 B 操作数（`Sb`/`du`）行 stride `BW=16` 元素 = 32 B ≡ 0 (mod 32) 同病。
把 stride 分别改成 `KD+8=136`（272 B，每行移 4 bank）与 `BW+8=24`（48 B，每行移 12 bank），
16 行覆盖 16×4=64 个 bank 槽 → 落满 32 个 bank、零冲突。这是 `ldmatrix` 常规 swizzle 的**退化版**
（不改地址的 XOR、只加常量偏移），代价是 **smem 变大**。

实现上把 `Stage`/`Shared` 里所有操作数数组的行 stride 换成 `KDS/BWS/BVSCTA`，`wmma::load_matrix_sync`
的 `ldm` 参数同步改（`KD→KDS`、`BW→BWS`、`BV→BVSCTA`）；`k/q/w` 的 cp.async 落点、`duf/sacc` 的
`store_matrix_sync`、以及 fp32 状态 `S` 的标量循环全部改成 `stride=BWS` 的二维索引。**数学一字未动**。

### 20.2 Part A：padding 必须数值无感

`harness/check_bwd_b1_pad.py`（输出 `harness/check_bwd_b1_pad.out.txt`）同源编译两次
（`-DB1_PAD` 开/关），与 FLA `dhu` 同操作数比。三组 shape × BT=32/64（含非整除 T=250）：

| 比较 | `dh` | `dh0` | `dv2` |
|---|---|---|---|
| pad vs **FLA `dhu`** | max_abs/max_rel = **0** | **0** | **0** |
| pad vs nopad | **0** | **0** | **0** |

即 padding **逐位相同**（两个 build 之间、以及与 FLA 之间），证明它只改 smem 地址、不改算术
（与 §18 的「裸 wmma 与 FLA `dhu` 逐位相同」一致）。

### 20.3 Part B：kernel-only 计时（CUPTI 均值）

| shape | BT | nopad µs | pad µs | pad/nopad | FLA `dhu` µs | xFLA nopad → pad |
|---|---|---|---|---|---|---|
| B1 T512 H4 | 16 | 214.4 | 193.2 | **0.90×** | n/a | — |
| B1 T512 H4 | 32 | 144.5 | 123.6 | **0.86×** | 20.6 | 7.03 → 6.01 |
| B1 T512 H4 | 64 | 105.0 | 84.5 | **0.80×** | 12.9 | 8.16 → 6.56 |
| B2 T1024 H8 | 32 | 301.3 | 259.3 | **0.86×** | 37.1 | 8.12 → 6.99 |
| B2 T1024 H8 | 64 | 212.2 | 171.4 | **0.81×** | 22.8 | 9.32 → 7.52 |
| B8 T2048 H8 | 32 | 719.9 | 551.9 | **0.77×** | 169.0 | 4.26 → 3.27 |
| B8 T2048 H8 | 64 | 578.8 | 701.8 | **1.21×** | 118.5 | 4.88 → **5.92（变慢）** |

除最后一格外，pad 一律快 **10–23%**。**最后一格是 smem 变大的副作用**：BT=64 的 padding 把
`Shared` 从 111.1 KB 抬到 143.6 KB，`2×143.6=287 KB > 233 KB` ⇒ 占用率从 **2 CTA/SM 掉到 1**，
冲突的收益被占用率损失反超（且这只在大 shape、grid 填满时暴露）。

### 20.4 Part C：smem 足迹

| BT | ns=1 | ns=2 | ns=3 |
|---|---|---|---|
| 16 | 58880 → 82944 | 73728 → 99072 | 88576 → 115200 |
| 32 | 76288 → 103168 | 105472 → 134912 | 134656 → 166656 |
| 64 | 111104 → **143616** | 168960 → 206592 | 226816 → 269568 |

pad 一律 +29%（`KDS/KD=+6.25%`，`BWS/BW=+50%`）；BT=64/NS=1 正好跨过 2 CTA/SM 的 116.7 KB 门槛。

### 20.5 Part D：`ncu` 复核——冲突确实被消掉，但 `wait` 纹丝不动

`harness/probe_b1_cuda_pad_ncu.sh`（输出 `harness/probe_b1_cuda_pad_ncu.out.txt`）同场配对
nopad/pad。bank-conflict 占比与**新的** stall 组成：

| 配置 | bank-conflict / ld-wavefronts | L1/TEX% | DRAM% | `wait` | `short_sb` | `long_sb` | tensor% | theo/ach occ | Duration µs |
|---|---|---|---|---|---|---|---|---|---|
| BT64 T512H4 nopad | 827392/1083392 = **76.4%** | 43.4 | 0.72 | 1.79 | 0.70 | 0.50 | 2.31 | 6.25/3.12 | 110.1 |
| BT64 T512H4 **pad** | 59392/315392 = **18.8%** | **23.7** | 0.89 | 1.77 | 0.64 | 0.24 | 2.85 | 3.12/3.12 | **89.5** |
| BT64 B8T2048H8 nopad | 53115545/69401241 = **76.5%** | 65.0 | 13.4 | 1.82 | 1.37 | 0.64 | 3.46 | 6.25/6.08 | 595.5 |
| BT64 B8T2048H8 **pad** | 3702784/19988480 = **18.5%** | **23.0** | 10.9 | 1.81 | 0.66 | 0.39 | 2.81 | 3.12/3.11 | 722.8 |
| BT32 B2T1024H8 nopad | 6684672/9314304 = **71.8%** | 36.2 | 2.89 | 1.73 | 0.88 | 1.10 | 1.69 | 9.38/3.13 | 304.4 |
| BT32 B2T1024H8 **pad** | 860160/3489792 = **24.6%** | 22.4 | 3.34 | 1.60 | 1.03 | 0.39 | 1.96 | 6.25/3.13 | 262.1 |

三件事：

1. **冲突确实被消掉**：76% → 19–25%（残留来自 fp32 状态的标量 smem 访问与 `e` 数组，非 mma 操作数），
   L1/TEX 吞吐从 43–65% 掉到 22–24%。注意 `wavefronts_mem_shared_op_ld` **总量也从 1.08M→315K、
   69.4M→20.0M**——冲突重放本就计入 wavefront 总数，去掉重放后只剩 ~260K 底数（小 shape），
   与「base≈256K」吻合，即 **3.2× 的 shared-load 量确实是纯重放**。
2. **但 `wait` 几乎不动**（1.79→1.77、1.82→1.81、1.73→1.60）：`wait` 是**等 mma 的固定流水延迟**，
   与 smem 取数是否冲突无关。小 shape 的 pad 提速（110→89.5）来自 L1/TEX 压力下降（43%→24%）、
   而非 `wait`；大 shape 的 pad **变慢**则是占用率 2→1。
3. **占用率/并行度仍是天花板**：`Eligible Warps Per Scheduler` 全程 0.15–0.26、`No Eligible` 74–85%、
   tensor 密度 1.7–2.9%（FLA 14%）。每 CTA 只有 2 warp、每 warp 一条 mma 依赖链，即使冲突清零，
   `wait`+`short_sb` 仍主导。

### 20.6 结论：bank conflict 是 ~1.2× 项，不是量级项（修正 §19.5）

用 padding 把 bank-conflict 从 76% 打到 ~19%，**端到端只快 10–23%**（且大 shape BT=64 因 smem 膨胀
反而慢 21%）。这与 §19.5 把「消 76% bank conflict」列为**量级项**的说法**不符**——真正的量级差
（对 FLA 4.9–8×）来自**并行度/mma 密度**（tensor 2.3% vs 14%、每 scheduler 0.2 个 eligible warp），
bank conflict 只是其中一个 ~1.2× 的分项。修正后的目标卡 CUDA B1 优先级：

1. **先补并行度**：把「每 CTA 2 warp、每 warp 一条 mma 链」做到 K2 形状（多 warp / warp-specialized、
   每 chunk 更高的 mma 密度）——这是**量级项**，§18.4 早已指出的方向；
2. **bank conflict 用零成本的 swizzle 消**（XOR 地址 swizzle 或 `ldmatrix`），**不要用 padding**——
   本节证明 padding 的 smem 代价（+29%）会在 BT=64 把占用率打掉一级、把收益吃回去；
3. `cp.async`/TMA 预取仍只是「不退化」保底（§19.4）。

> 说明：本节仍是**研究用最小 CUDA 原型**的受控消融，`xFLA` 绝对值不代表目标卡天花板；可下的结论是
> 「bank conflict 的量级」与「residual 在 mma 延迟 + 并行度」，而非具体加速比。ncu 的 `Duration`
> （110.1/89.5 等）与 CUPTI 均值（105.0/84.5）同量级，差异来自 replay 与冷启动（同 §19 说明）。

## 21. 隔离 CUDA B1 的「mma 依赖 / 并行度」项：loop-reorder 与 CTA 形状受控实验（P7-18）

§20.6 把目标卡 CUDA B1 的**第一优先级**定为「补并行度：把每 CTA 2 warp、每 warp 一条 mma 链做到
K2 形状（多 warp / warp-specialized、每 chunk 更高的 mma 密度）——这是量级项」，但那条结论是从
`ncu` 的**横向对比**（tensor 2.3% vs FLA 14%、Eligible Warps/Scheduler 0.2）推出的**假设**，尚未在
同一内核上做受控实验。 本节在**同一份源码**上做两种最便宜的受控改动，检验「并行度/mma 密度」这个
量级项里，有多少是**免费的**（只调指令/循环顺序、不改结构），从而决定是否值得动 K2 形状的重写。

### 21.1 两个免费旋钮

**旋钮 A——warp 内 mma 指令级并行（ILP）。** 原内核里 op1（`du = k @ S`，`harness/b1_cuda_probe.cu:224`）
的循环是 `for m: for kk:`，同一个累加片段 `cdu[m]` 的 8 个 k-step 串行依赖；op2/op3
（`harness/b1_cuda_probe.cu:265`，`acc = scale·qᵀdo − wᵀdu`）每个 m-tile 内的 4 个 k-step 也串行。
把它们换成 `for kk: for m:`，就能在指令流里**同时暴露** MT1（op1）或 2·MT2（op2/op3）条**互相独立**
的累加链。实现为编译宏 `B1_ILP1`（`:228`）与 `B1_ILP2`（`:267`）。关键：每个累加片段**仍然按同样的
kk 顺序求和**，所以**逐位结果不变**；代价仅是同时驻留的片段数（op2/op3 要 2·MT2=16 个片段）。

**旋钮 B——CTA 形状 / 每 CTA warp 数。** 编译期宏 `B1_BV`/`B1_NW`（`:44–51`，约束 `BW=BV/NW=16`）：
默认 `BV=32,NW=2`；`BV=16,NW=1` 让 grid.x 翻 4 倍（小 `B·H` 时填更多 SM）；`BV=64,NW=4` 让每 CTA
更“胖”。这只改 work 划分、不改每元素算术。

### 21.2 Part A：两种改动都必须数值无感

`harness/check_bwd_b1_ilp_cuda.py`（输出 `harness/check_bwd_b1_ilp_cuda.out.txt`）对 3 组 shape ×
BT=32/64（含非整除 T=250）把 `base/ilp1/ilp2/ilp12/bv16/bv64` 六份 build 与 base 逐位比：

| 比较 | `dh` / `dh0` / `dv2` |
|---|---|
| ilp1 / ilp2 / ilp12 / bv16 / bv64 **vs base** | 全部 **max_abs = 0（逐位相同）** |

loop 重排与 CTA 形状都**只改调度、不改算术**（Σkk 顺序不变、每元素乘加不变），与 §18/§20 的
「裸 wmma 与 FLA `dhu` 逐位相同」自洽。

### 21.3 Part B：kernel-only 计时（CUPTI 均值）——两个旋钮都只值 ~1.1×

`xFLA` 是相对 FLA `dhu`（同 BT），`xbase` 是相对本 build 的 base：

| shape | BT | base µs | ilp1 | ilp2 | ilp12 | bv16 | bv64 | FLA µs |
|---|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 146.2 | 1.00× | **0.89×** | 0.90× | 1.02× | 1.28× | 20.5 |
| B1 T512 H4 | 64 | 105.0 | 1.02× | **0.94×** | 0.94× | 1.05× | 1.44× | 12.9 |
| B2 T1024 H8 | 32 | 303.1 | 1.00× | **0.90×** | 0.90× | 1.02× | 1.28× | 37.3 |
| B2 T1024 H8 | 64 | 213.0 | 1.02× | **0.95×** | 0.94× | 1.06× | 1.44× | 22.7 |
| B8 T2048 H8 | 32 | 732.1 | 1.02× | **0.90×** | 0.89× | 1.04× | 1.05× | 169.8 |
| B8 T2048 H8 | 64 | 578.7 | 1.00× | **0.88×** | 0.88× | 1.77× | 1.05× | 119.0 |

三点：
1. **`ilp1`（op1 重排）完全无效（1.00–1.02×）**：op1 只有 MT1=2/4 条链、且它只占约 1/3 的 mma，
   调度器本来就没被它卡住。
2. **`ilp2`（op2/op3 重排）稳定快 6–12%（0.88–0.94×）**，`ilp12`≈`ilp2`（叠加无额外收益）。
   这是**免费**的（0 smem、逐位不变），但离量级项（xFLA 4.9–8×）还差得远。
3. **CTA 形状两向都变差**：`bv16`（1 warp/CTA、grid×4）在多数 shape 反而慢 2–6%（大 shape BT=64 慢
   1.77×），`bv64`（4 warp/CTA）慢 5–44%。说明「多开 CTA / 多塞 warp」在**占用率被 smem 钉死**
   （§20）时不会增加每调度器可用 warp，只是改变串行度与尾部。

### 21.4 Part D：寄存器（`cuobjdump -res-usage`，无溢出）

| build | BT32/NS1 REG | BT64/NS1 REG | STACK | LOCAL |
|---|---|---|---|---|
| base | 72 | 128 | 0 | 0 |
| ilp1 | 72 | 96 | 0 | 0 |
| ilp2 | 168 | 168 | 0 | 0 |
| bv16 / bv64 | 72 | 128 | 0 | 0 |

`ilp2` 把寄存器抬到 168（同时驻留 16 个累加片段）仍**零 spill**；`ilp1` 反而降到 96（编译器把
op1 的累加片段压得更紧）。寄存器都不是占用率瓶颈（occupancy 由 smem 决定，§20）。

### 21.5 用 `ncu` 复核：`wait` 与「可用 warp」几乎不动

`harness/probe_b1_cuda_ilp_ncu.sh`（输出 `harness/probe_b1_cuda_ilp_ncu.out.txt`）同场配对
base/ilp1/ilp2：

| 配置 | `wait` | `short_sb` | `long_sb` | tensor% | Eligible W/S | theo/ach occ | Duration µs |
|---|---|---|---|---|---|---|---|
| BT64 T512H4 base | 1.79 | 0.70 | 0.50 | 2.31 | 0.21 | 6.25/3.12 | 110.4 |
| BT64 T512H4 ilp1 | 1.79 | 0.99 | 0.59 | 2.27 | 0.20 | 6.25/3.12 | 112.8 |
| BT64 T512H4 **ilp2** | 1.72 | 0.96 | 0.94 | **2.47** | 0.20 | 6.25/3.13 | **104.0** |
| BT64 B8T2048H8 base | 1.82 | 1.40 | 0.63 | 3.48 | 0.16 | 6.25/6.10 | 590.3 |
| BT64 B8T2048H8 ilp1 | 1.83 | 1.74 | 0.71 | 3.44 | 0.15 | 6.25/6.09 | 593.1 |
| BT64 B8T2048H8 **ilp2** | 1.76 | 1.69 | 1.35 | **3.94** | 0.16 | 6.25/6.09 | **522.1** |

**关键**：即使把 warp 内独立 mma 链从 1 条加到 16 条（ilp2），`wait` 只从 1.82→1.76、
`Eligible Warps/Scheduler` 仍是 **0.16**、tensor 密度只从 3.5%→3.9%（FLA 仍是 14%）。原因是
**每调度器只挂着 ~1 个 active warp**（theoretical occ 6.25% ⇒ 4 warp/SM ÷ 4 scheduler = 1），
这个 warp 一旦等 mma 固定延迟（`wait`）就没有别的 warp 可切——**warp 内 ILP 填不满一个被
warp 数卡死的调度器**。ILP2 的那点收益来自把「等 mma」的间隙塞进了同 warp 的别的 mma，但
`long_sb` 随之上升（1.35，独立链现在也要等操作数），净收益只有 ~10%。

### 21.6 结论：免费的 ILP/形状只值 ~1.1×；量级项要**提高每 warp 的 mma 密度**（修正 §20.6 的措辞）

把 §20（消 bank conflict，~1.2×）与本节（加 warp 内 ILP，~1.1×）合起来，两个「免费」杠杆各自
都只是**分项**，加起来也到不了 FLA 的十分之一——**§20.6 把「并行度/mma 密度」列为量级项是对的，
但它的可实现路径不是「多塞 warp / 加指令级并行」**：

- 占用率已被 smem（每 warp 的 fp32 状态 + 操作数）**钉死在 6.25%**，与 FLA 相同；`bv16/bv64` 的实验
  证明**动 CTA 形状换不到更多可用 warp**（反而更慢）。
- warp 内 ILP（`ilp2`）免费但只值 ~1.1×，且 `wait` 几乎不动——**瓶颈是每调度器只有 ~1 个 warp**，
  ILP 填不满它。
- 真正把 FLA 拉开的是**同一个 6.25% 占用率下、每 warp 的 mma 密度是 6×**（tensor 14% vs 2.3%）：
  FLA 用 `ldmatrix` 高效喂数、单 warp 背靠背发 mma；我们每笔 mma 都被 smem 取数（`short_sb` + 冲突）
  与固定 `wait` 拉开。**所以量级项的落点是「提高每 warp 有效 mma 发射密度」**（`ldmatrix`/TMA 整块喂数、
  `cp.async` 流水、把 `wait` 藏进更多独立 warp 的工作），**而不是**简单加 warp 数或加指令级并行。

修正后的目标卡 CUDA B1 优先级（取代 §20.6 的第 1 条措辞）：

1. **structural：把 wmma-from-smem 换成 `ldmatrix`/TMA-mma 的 K2 形状**，目标是把**每 warp 的 mma
   密度**从 2.3% 抬到 FLA 的 ~14%——这是**唯一**的量级项，免费旋钮（bank conflict、指令级 ILP）都已
   证明只值 ~1.1–1.2×；
2. bank conflict 随之用零成本 XOR swizzle 解决（勿用 padding，§20）；
3. `cp.async`/TMA 预取保底（§19.4）。

> 说明：本节与 §17.2 的 Triton 结论**不矛盾**：Triton 原型里 `tl.range` 流水把 B1 从 1.96× 拉到
> 1.00×，是因为 Triton 会把整个循环体（含 `tl.dot` 的 ldmatrix 与流水）一起生成；而手写裸 wmma
> 版把操作数取数留在了 `wmma::load_matrix_sync` 里，**流水杠杆只有在内核已是 K2 形状时才能兑现**
> （§18.4 早已指出）。本节进一步证明：**连「加指令级 ILP」这种更便宜的形式也兑现不了**。
> 所有数字仍是研究用最小原型，`xFLA` 绝对值不代表目标卡天花板。

## 22. 把手写 CUDA B1 改成 **K2 形状**：寄存器常驻状态 + `mma.sync`/`ldmatrix`（P7-19）

§21.6 把目标卡 CUDA B1 的**唯一量级项**定为「提高每 warp 的有效 mma 密度」（tensor 2.3% vs FLA 14%），
并证明免费旋钮（bank conflict ~1.2×、warp 内 ILP ~1.1×、CTA 形状）都填不满它。本节**直接兑现**这条
结论：不再在 `wmma`-from-smem 的骨架上调参，而是把它换成 **K2 形状**——运行态 $S$ **常驻寄存器**
（不再有 fp32 smem 主副本与 fp32 staging 往返），mma 用裸 `mma.sync.m16n8k16`、操作数用
`ldmatrix` 从 **XOR 填充后无冲突**的 smem 整块喂入。新增 `harness/b1_reg_probe.cu` +
`harness/check_bwd_b1_reg.py`（`harness/check_bwd_b1_reg.out.txt`）+ `harness/ncu_b1_reg_driver.py`
/ `harness/probe_b1_reg_ncu.sh`（`harness/probe_b1_reg_ncu.out.txt`）。

### 22.1 设计：把「状态、累加器、操作数」全部搬进寄存器

递推一字未动（逐 (n,h)、逐 16 列 V-块、逆序 chunk）：

```
for c = NT-1 .. 0:
    dh[c] = bf16(S)                       # 状态进入 chunk c
    du = k @ S + dv ;  dv2[c] = bf16(du)
    S  = S*e + scale*(q^T @ do) - (w^T @ du)
dh0 = S
```

与 `b1_cuda_probe.cu`（P7-15）的关键结构差异：

- **$S$ 常驻寄存器**：每 warp 持 `[K=128, BW=16]`，按 mma 累加器布局组织成 `8` 个 m16 块 × `2` 个
  n8 块 × `4` 个 fp32 = **64 个 fp32 寄存器**。旧版 $S$ 是 smem 里的 fp32 主副本（外加 bf16 镜像
  `sdh`），每 chunk 还要把 `du`/`acc` 经 fp32 smem staging（`sduf`/`sacc`）绕一圈。
- **`mma.sync` 而非 `wmma`**：`wmma::load_matrix_sync`/`mma_sync` 被替换为裸
  `ldmatrix.sync.aligned.m8n8.x4(.trans)` + `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`，
  A 操作数（`k` / $q^\top$ / $w^\top$）走 `ldmatrix`，B 操作数（$S$ 的 bf16 镜像 / `do` / `du`）走
  `ldmatrix.x2.trans`。
- **XOR 填充（退化 swizzle）**：操作数行 stride 从 `KD=128`/`BW=16` 改成 `+8`（`136`/`24` bf16），
  使每个 `ldmatrix` 的 8 行落在不同 bank 上——**只改地址、不改算术**（§20.1 同理）。
- **一个式子、一条累加链**：`S` 更新写成 `S = (S*e) + (scale*do-缩放后的 q^T do) + (-w^T du)`，
  用 mma 的 C 输入把 `S*e` 直接当初始累加值（`scale` 折进 `do` 的 bf16 操作数：`q^T@(scale·do)`），
  于是**只需一组 $S$ 累加器寄存器**（无需为 q/do、w/du 各开一组 fp32 暂存）。`-w^T` 用 bf16 符号位
  XOR 精确取反。

smem 足迹因此从「stage + fp32 状态 + 三块 fp32 staging」降到「stage + 每 warp 的 bf16 $S$/`du` 镜像」：
BT=64/NS=1 从 **111.1 KB → 81.4 KB**（少 27%），BT=16 从 58.9 → **30.0 KB**（‑49%）。理论占用率仍是
6.25%（被 stage 钉住，同 §16/§20）。

### 22.2 Part A：正确性——落到 bf16 底噪（含非整除 T）

`harness/check_bwd_b1_reg.py` 与 FLA `dhu`（bf16、同 BT）比 `dh`/`dv2`/`dh0`：

| shape | BT | `dh` max_abs/max_rel | `dv2` max_abs/max_rel | `dh0` max_abs/max_rel |
|---|---|---|---|---|
| B1 T512 H4 | 32/64 | 4.88e-4 / 5.0e-3 | 4.88e-4 / 4.4e-3 | 1.9e-4 / 2.1e-3 |
| B2 T1024 H8 | 32/64 | 4.88e-4 / 4.3e-3 | 4.88e-4 / 4.5e-3 | 2.1e-4 / 2.2e-3 |
| B1 T250 H3（非整除） | 32/64 | 4.88e-4 / 4.9e-3 | 4.9e-4 / 4.0e-3 | 1.5e-4 / 2.1e-3 |

`max_abs` 对所有 case 稳定在 **4.883e-4 = bf16 在 ~0.1 量级上的 1 ulp**，即落盘的 bf16 `dh`/`dv2`
与 FLA 至多差 1 个 bf16 格；`dh0`（fp32）相对 ~2e-3。与 P7-15 的「裸 wmma 与 FLA **逐位相同**」不同，
这里**不再逐位**：唯一原因是 `scale` 被折进 `do` 的 bf16 操作数（`round(scale·do)`）而非作用在 fp32
点积上，引入 ~1e-3 的 bf16 舍入——与 P7-9 的 bf16-B1 端到端误差（6.4e-3）同阶，属 bf16 底噪，非代数偏差。

### 22.3 Part B：kernel-only 计时——比 wmma 基线快 **2.3–2.7×**，xFLA 从 ~8× 收到 2.1–3.7×

`CUPTI` 均值（µs/call；`xbase`=相对 `b1_cuda_probe.cu` 同 BT、`xFLA`=相对 FLA `dhu`，FLA 无 BT<64 实现）：

| shape | BT | base(wmma,NS=1) | reg NS=1 | reg NS=2 | reg NS=3 | FLA `dhu` | reg NS=1 xFLA |
|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 16 | 214.4 | 81.8 | 79.2 | 79.2 | n/a | — |
| B1 T512 H4 | 32 | 145.1 | 56.4 | **53.7** | 56.8 | n/a | — |
| B1 T512 H4 | 64 | 105.4 | 41.5 | 41.7 | 43.0 | 13.0 | **3.20** |
| B2 T1024 H8 | 16 | 446.7 | 189.4 | 159.9 | 160.7 | n/a | — |
| B2 T1024 H8 | 32 | 302.4 | 124.3 | **108.1** | 112.0 | n/a | — |
| B2 T1024 H8 | 64 | 214.3 | 85.6 | 81.9 | 85.1 | 22.9 | **3.75** |
| B8 T2048 H8 | 16 | 1029.3 | 616.9 | 700.7 | 700.5 | n/a | — |
| B8 T2048 H8 | 32 | 716.2 | 368.3 | 387.0 | 367.8 | n/a | — |
| B8 T2048 H8 | 64 | 585.5 | **251.0** | 327.3 | 402.1 | 119.3 | **2.10** |

三点：

1. **相对旧 CUDA 基线稳定快 2.3–2.7×**（`xbase` 0.37–0.43，全部 shape、全部 BT）。这是「K2 形状」
   相对「wmma-from-smem」的直接红利，与 §21.6 预测的量级项落点一致。
2. **`xFLA` 从 §18 的 4.3–9.6× 收到 2.10–3.75×**（BT=64）。BT=16（部署粒度）81.8 µs vs 旧 214.4。
   **仍慢 FLA 2–3.7×**，说明「寄存器常驻 + ldmatrix」只吃掉了量级项的一部分，剩下的见 §22.4。
3. **`NSTAGE>1` 仍几乎无效、且大 shape 反噬**（B8 251→327→402）：与 §18/§19.4 一致——本内核已非
   global-load bound（见 §22.4 的 `long_sb`），预取加深度只增 smem 掉占用率。

> 诚实定位：在实验 GPU 上，**Triton B1**（P7-14/§17）本就已到 1.00–1.96× FLA，比本节手写 CUDA 更接近
> FLA；本节的产物是**纯 CUDA**（不依赖 CUTLASS/Triton 代码生成）的 K2 形状原型，供 Triton/CUTLASS
> 支持不确定的目标卡使用，并把「K2 形状能兑现多少」量化到「~2.4×、xFLA 收到 ~2–3.7×」。

### 22.4 Part D：`ncu` 复核——bank conflict 归零、tensor 密度翻倍、`wait` 下移

`harness/probe_b1_reg_ncu.sh`（输出 `harness/probe_b1_reg_ncu.out.txt`）同场配对 reg / base / FLA：

| 配置 | bank-conflict | tensor% | `wait` | `short_sb` | `long_sb` | barrier | theo/ach occ | Duration µs |
|---|---|---|---|---|---|---|---|---|
| base BT64 T512H4 | 827392 | 2.30 | 1.79 | 0.70 | 0.51 | 0.02 | 6.25/3.12 | 110.5 |
| **reg BT64 T512H4** | **0** | **5.83** | **1.54** | 0.94 | 0.83 | 0.03 | 6.25/3.13 | **46.5** |
| base BT64 B8T2048H8 | 53106635 | 3.48 | 1.82 | 1.42 | 0.64 | 0.03 | 6.25/6.10 | 591.1 |
| **reg BT64 B8T2048H8** | **556239** | **8.29** | **1.53** | 1.83 | 1.68 | 0.08 | 6.25/6.13 | **257.2** |
| FLA `dhu` B8T2048H8 | 2008 | 12.15 | 0.96 | 0.49 | 2.23 | 0.94 | 6.25/6.27 | 117.4 |

三件事印证了归因，也暴露了残差：

1. **bank conflict 被彻底消掉**：小 shape **827392 → 0**，大 shape **53.1M → 0.56M**（§20 的 padding
   只能压到 19–25%，这里靠 XOR stride 到 ~1%）。这正是 §21.6 说的「零成本 XOR swizzle」。
2. **每 warp mma 密度翻倍**：tensor **2.3% → 5.8%**（小）/ **3.5% → 8.3%**（大，逼近 FLA 的 12.2%），
   `wait` 也从 1.79→1.54 / 1.82→1.53——把「等 smem 取数」的洞补上后，mma 背靠背发得更多。
3. **残差在 `short_sb` 与并行度**：`short_sb` 反而略升（取数变密），且占用率仍是 6.25%、`Eligible
   Warps/Scheduler` 0.14–0.19（每调度器 ~1 warp）——**仍未到 FLA 的 `wait`=0.96**。大 shape 下 reg 的
   `lts` 已到 **82%**（旧 base 仅 15%），说明它已经把预算从「smem 重放」挪到了**真正的 L2 流量**上，
   接近 FLA 的 DRAM（66%）形态；但 tensor 密度与 `wait` 还差 FLA 一截，对应 §22.3 的剩余 2–3.7×。

### 22.5 结论与下一步

- **兑现了 §21.6 的量级项落点**：把「wmma-from-smem + smem fp32 状态」换成「寄存器常驻状态 +
  `mma.sync` + `ldmatrix` + XOR swizzle」后，bank conflict 归零、mma 密度翻倍、`wait` 下移，
  kernel 快 **2.3–2.7×**，`xFLA` 从 ~8× 收到 **2.1–3.7×**（BT=64）。这是**纯 CUDA** 的 K2 形状原型。
- **残差（仍慢 FLA 2–3.7×）的落点**：`wait`（1.53 vs 0.96）、`short_sb`、以及大 shape 的 stage smem
  把占用率钉在 6.25%。要进一步逼近 FLA，方向是 **TMA（`cp.async.bulk`）整块喂数 + warp-specialization
  的显式深流水**（§17 的 Triton 正是靠此到 1.0×），而非再加指令级并行（§21）或 padding（§20）。
- **更新后的目标卡 CUDA B1 优先级**：(1) 本节的**寄存器常驻 + `ldmatrix`/`mma.sync` + XOR swizzle**
  是已验证的起点（+2.4×）；(2) 在其上补 **TMA + warp-specialization 流水**（学 K2/§17）；(3) `CHUNK=16`
  部署粒度下 BT=16 的 reg 版 81.8 µs 已远好于旧 214.4 µs，可作为部署基线。

> 说明：本节仍是**研究用最小 CUDA 原型**；`xFLA` 绝对值不代表目标卡天花板（实验卡为 H100）。可下的
> 结论是**归因**：K2 形状能兑现约 2.4×，把量级项吃掉了大头但没吃干净，剩余在 `wait`/并行度/TMA 深度。

**本节踩坑**（都只有真写 `ldmatrix` 才暴露）：

1. **`ldmatrix.x4.trans` 的 4 个子块次序与非 trans 不同**：非 trans 输出 `a0..a3` 对应内存子块
   `(rowblk,colblk)=(0,0),(1,0),(0,1),(1,1)`；`.trans` 则对应 `(0,0),(0,1),(1,0),(1,1)`。用错次序的指纹是
   **`max_rel≈1.0`（完全错位）**，而非小误差——本次第一版即踩此坑，把 `q^T`/`w^T` 的列块与行块调换。
2. **B 操作数（`[K,N]` 行主序）必须用 `ldmatrix.x2.trans`**：非 trans 会让每 lane 拿到相邻 **N** 的两个
   元素，而 mma 的 B 片段要求相邻 **K**——用非 trans 的指纹同样是整片错位。
3. **`scale` 折进 bf16 操作数会丢「逐位相同」**：`round(scale·do)` 使结果落到 bf16 底噪（~1e-3），
   换来的是**只用一个累加器**（`S`）。若把 `scale` 留在 fp32 点积后，需要为 q/do 单开一组 $S$ 大小的
   fp32 暂存（+64 寄存器），本原型选择前者。

## 23. 在 K2 形状之上补 **TMA + warp-specialization**（P7-20）

§22.5 把「把 K2 形状再往前推」的下一步钉死为一条：**TMA（`cp.async.bulk.tensor`）整块喂数 +
warp-specialization 的显式深流水**。本节**照做**：在 P7-19 的 K2 形状内核（$S$ 常驻寄存器 + 裸
`mma.sync`/`ldmatrix`）之上，把「操作数怎么进 smem」从每线程 `cp.async` 换成 **一个专职 producer
warp 发 TMA**，用 **per-stage `mbarrier`**（`mbarrier::complete_tx::bytes` + `try_wait.parity`）
做 producer/consumer 握手，稳态循环里**不再有 `__syncthreads()`**。新增
`harness/b1_tma_probe.cu` + `harness/check_bwd_b1_tma.py`（`harness/check_bwd_b1_tma.out.txt`）
+ `harness/ncu_b1_tma_driver.py` / `harness/probe_b1_tma_ncu.sh`
（`harness/probe_b1_tma_ncu.out.txt`）。

### 23.1 设计

固定 `BV=64`（一个 128 B swizzle atom）、`NW=4` 个 compute warp，外加第 5 个 warp 作 producer；块内
共 160 线程。与 `b1_reg_probe.cu` 的差异：

- **TMA + 128 B swizzle 取代 `cp.async` + XOR padding**。q/k/w 的行是 256 B（KD=128 bf16），
  TMA 的 128 B swizzle 要求 box 内维 ≤128 B，故用 **5D 张量图**
  `globalDim={64,2,H,T,B}`（dim0 = 行内 64 列 = 一个 atom，dim1 = 两个半行），
  `box={64,1,1,BT,1}`，两次 TMA 把一行拆成**两个独立 `[BT,64]` 区域** `q[0]/q[1]`（dO/dV 同理，
  `vhalf=v0/64`）。TMA 的越界**自动零填充**顺带替掉非整除 T 的手工补零。
- **smem 布局是 `Swizzle<3,4,3>` 逐字节异或**：`swizzled_b = b ^ (((b>>7)&7)<<4)`。`ldmatrix`
  的每 lane 行地址据此改写（`ldsm_sw_x4` / `ldsm_sw_x4_t` / `ldsm_sw1_x2_t`）。
- **producer/consumer mbarrier**：每 stage 两个 barrier——`full[s]`（producer `arrive.expect_tx`
  1 次 + TMA 的 `complete_tx`）与 `empty[s]`（每个 compute warp **lane 0** arrive 1 次，count=NW）。
  producer 在复用 stage 前 `try_wait.parity(empty[s])`，compute warp `try_wait.parity(full[s])`；
  相位按 `i/NSTAGE` 交替。**`e`（$E^{tot}$）改由 compute warp 直接读 global**（每 chunk 两个标量，
  L2 常驻），省掉一次 TMA 与一个 512 B 的 stage 分量。

### 23.2 Part A：正确性——与 P7-19 的 reg 内核**逐位相同**（含非整除 T）

`harness/check_bwd_b1_tma.py` 在同操作数上与 FLA `dhu`、以及 P7-19 的 reg 内核比 `dh`/`dv2`/`dh0`：

| 指标（B1 T512 H4 / B2 T1024 H8 / B1 T250 H3，BT=32/64） | vs FLA | **vs reg（P7-19）** |
|---|---|---|
| `dh` max_abs / max_rel | 4.88e-4 / 5.0e-3 | **0 / 0** |
| `dv2` max_abs / max_rel | 4.88e-4 / 4.4e-3 | **0 / 0** |
| `dh0` max_abs / max_rel | 1.9e-4 / 2.1e-3 | **0 / 0** |

即：TMA 只换「怎么把同一批字节搬进 smem」，**算术一字未动**，输出与 reg 内核**逐位相同**；而 vs FLA 的
4.88e-4 = bf16 在 ~0.1 量级的 1 ulp（与 §22.2 同）。非整除 T（250）也用 TMA 的 OOB 零填充自然覆盖。

### 23.3 Part B：计时——只值 ~0.8×（大 shape），BT=16 反被拖累

`CUPTI` 均值（µs/call）。关键对照是**同 `BV=64`/`NW=4` 的 reg 内核** `reg64`（`-DB1R_BV=64 -DB1R_NW=4`），
这样差的只是「TMA + 专职 producer + mbarrier 流水」这一项：

| shape | BT | reg(BV32,NS1) | reg64(BV64,NS1) | **tma NS2** | tma xreg | **tma xreg64** | FLA `dhu` | tma xFLA |
|---|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 56.5 | 72.2 | 71.6 | 1.27 | **0.99** | n/a | — |
| B1 T512 H4 | 64 | 42.5 | 53.9 | 44.8 | 1.05 | **0.83** | 13.0 | 3.45 |
| B2 T1024 H8 | 64 | 85.2 | 107.8 | 86.8 | 1.02 | **0.81** | 22.8 | 3.80 |
| B8 T2048 H8 | 32 | 367.9 | 332.9 | 343.3 | 0.93 | **1.03** | n/a | — |
| B8 T2048 H8 | 64 | 252.4 | 240.6 | **205.5** | 0.81 | **0.85** | 118.8 | **1.73** |
| B2 T1024 H8 | 16 | 187.9 | 251.3 | 232.8 | 1.24 | 0.93 | n/a | — |

三点：

1. **同形状下 TMA + warp-spec 只值 ~15–20%**：`BT=64` 三档 `xreg64=0.81–0.85×`，`BT=32` 基本打平
   （0.99–1.03），`BT=16` 反而略慢（0.93–1.14，producer warp 的固定开销 + 更小的 tile 摊不动）。
2. **`xFLA` 从 §22 的 2.10–3.75× 收到 1.73–3.80×**（`B8T2048H8 BT=64` 达 **1.73×**，全表最好）。
3. 因此「TMA + warp-specialization」是真实但**次线性**的一步：把 K2 形状再推 ~15%，**没有**把 2× 的残差吃掉。

### 23.4 Part C：smem 足迹反而**变大**（XOR padding 省的那点被 stage 复制吃掉）

| BT | NS | tma | reg |
|---|---|---|---|
| 64 | 1 / 2 | 103424 / 168960 | 81408 / 144384 |
| 32 | 1 | 64512 | 47104 |
| 16 | 1 | 45056 | 29952 |

TMA 路径把 `[BT,128]` 拆成 `[BT,64]×2` 的 swizzle atom，**不再需要 `+8` padding**，但每 stage 因此多出
半行对齐的零头；`BT=64/NS=1` **103 KB > reg 的 81 KB**，理论占用率从 12.5% 掉到 7.8%（见下）。这是
「TMA 省 smem」直觉的反例——省的是 padding（~14%），赔的是 stage 粒度。

### 23.5 Part D：`ncu` —— 占用率与 mma 密度都上来了，但 `long_sb` 变贵

`harness/probe_b1_tma_ncu.sh`（`probe_b1_tma_ncu.out.txt`），同场配对 reg64 / tma / FLA：

| 配置 | bank-conflict | tensor% | `wait` | `short_sb` | `long_sb` | barrier | theo/ach occ | Duration µs |
|---|---|---|---|---|---|---|---|---|
| reg64 BT64 B8T2048H8 | 0 | 8.85 | 1.50 | 1.62 | 2.25 | 0.14 | 12.5/6.32 | 240.3 |
| **tma BT64 NS2 B8T2048H8** | 264781 | **10.73** | **1.42** | **0.91** | 3.52 | **0.00** | 7.81/**7.75** | **207.7** |
| FLA `dhu` B8T2048H8 | 2032 | 12.12 | 0.96 | 0.49 | 2.10 | 0.94 | 6.25/6.23 | 118.1 |
| reg64 BT64 T512H4 | 0 | 9.18 | 1.51 | 1.61 | 1.90 | 0.11 | 12.5/6.23 | 58.0 |
| tma BT64 NS2 T512H4 | 4841 | 11.44 | 1.43 | 0.90 | 2.61 | 0.01 | 7.81/7.41 | 48.2 |

四件事与 §23.3 自洽：

1. **占用率被 producer-warp 结构抬起来了**：理论占用率虽降到 7.81%（smem 变大 + 每 CTA 多 1 warp），
   但 **achieved 从 6.32% 升到 7.75%**（compute warp 不再被 `cp.async` 地址计算拖住，producer 专职取数）。
2. **mma 密度上台阶**：tensor **8.85→10.73%**（小 shape 9.18→11.44），**逼近 FLA 的 12.12%**；
   这正是 §22.5 想补的「每 warp 有效 mma 密度」。
3. **`wait` 微降、`short_sb` 减半、`barrier` 归零**：`wait` 1.50→**1.42**（仍未到 FLA 0.96）、
   `short_sb` 1.62→**0.91**（smem 取数变顺）、barrier 0.14→**0.00**（稳态无 `__syncthreads`）。
4. **代价是 `long_sb` 变贵**（2.25→**3.52**）：compute warp 现在把「等操作数」从 `short_sb` 挪到了
   `full[s]` 的 `try_wait` 上（TMA + mbarrier 完成延迟），且 bank-conflict 从 0 回升到 26.5 万
   （占 shared-load wavefront 8.13M 的 **3.3%**，来自 i) dV 的**逐标量 swizzle 读** `sw_read`
   与 ii) swizzle 后的 ldmatrix 残差）。这两项 + 更浅的 pipeline（NS2 vs FLA 的深层）合起来，就是
   剩下 1.73× 的落点。

### 23.6 结论与下一步

- **兑现了 §22.5 的下一步**：在 K2 形状上补 TMA + warp-specialization，**正确性逐位保持**（vs reg=0），
  把 achieved 占用率 6.32→7.75%、tensor 8.85→10.73%（逼近 FLA 12.1%）、`short_sb`/`barrier` 明显下降，
  kernel 时间 `BT=64` 只降 **~15–20%**（`xreg64=0.81–0.85×`），`xFLA` 收到 **1.73×**（全表最好）。
- **诚实定位**：这一步**没有**把 2× 的残差吃掉。残差已重新定位到 **`long_sb`（TMA/mbarrier 完成延迟）
  + 浅 pipeline + swizzle 读的 bank-conflict**，而**不是**占用率或 smem 取数。下一步要么**加深 pipeline
  （更多 stage / K 方向拆两半喂）**，要么换成 CUTLASS 的 `PipelineTmaAsync`（K2 用的那套）把 producer
  与 consumer 的握手交给成熟实现，而不是继续在手写 mbarrier 上叠。
- **可复用的副产品**：`sw_off` / `sw_read` / `ldsm_sw_*` 是一套经过验证（min 用例 mismatch=0、整核逐位）
  的 **128 B swizzle + ldmatrix 地址计算**，可直接搬去做目标卡上任何 K2 形状 kernel 的 TMA 输入路径。

> **后续（P7-21，§24）**：本节的两条下一步里，第一条「加深 pipeline」已被数据证伪——`long_sb` 的一块
> 其实来自 compute warp **逐 chunk 直读 global `E^tot`**（§23.1 省掉的那次 TMA）；把它用一条 512 B 的
> TMA 搬回 smem 后，最优配置反而回到 **`NS=1`（2 CTA/SM）**，`long_sb` 降 0.6–0.8，全表最好
> `xFLA=1.63`。见 §24。

**本节踩坑**（都只有真写 TMA/mbarrier 才暴露）：

1. **`mbarrier.arrive` 的到达数必须严格等于 init 的 count**。第一版让**每个 consumer warp 的全部 32
   lane** 都 `arrive` 一个 count=`NW` 的 `empty` barrier：一次握手内到达 128 > 4，**硬件在同一个 phase
   内被超额到达后直接 fault**（`compute-sanitizer` 报 `Unknown Error`，定位在 `mbarrier.arrive` 指令，
   而非后续的 TMA）。改成 **lane 0 arrive 1 次**（count=NW）即修复。指纹：kernel 立即
   `unspecified launch failure`，且把 `arrive` 换成 `arrive.expect_tx` 或 `_` 汇都不影响——错的是**到达次数**。
2. **TMA 128 B swizzle 是 `Swizzle<3,4,3>` 的字节异或**：`b ^ (((b>>7)&7)<<4)`。用最开始一版
   `ldmatrix` 只做**线性 stride**（没套这个异或）时，**只有行号 ≡0 (mod 8) 的 lane 落在正确位置**——
   指纹：`dv2` 逐 token 误差在「token 号 ≡0 (mod 8)」处=1e-4（1 ulp）、其余处 O(0.1)。先用一个
   512 行以内的最小 TMA 用例（读回逐元素对拍，`mismatch=0/4096`）把公式钉死再进大核。
3. **TMA 搬进来的 `dV` 是 swizzled 的，但 `du = k@S + dv` 那一步是「逐标量索引读」而非 `ldmatrix`**。
   必须用 `sw_read(row,col)` 走一遍 swizzle 逆变换；直接用 `dV[row][col]` 会读到物理邻居——
   指纹同 (2)（按 mod 8 周期性正确）。
4. **别把 dV 的 offset 漏掉 `warp*BW`**：P7-19 的 reg 内核读 `dV[r0][warp*BW+cc]`，改写时漏了
   `warp*BW` 会让 warp 1–3 全读 warp 0 的列——指纹：**跨整个 V 维均匀 ~30× 抬高的误差**（不是局部错位）。
5. `nvcc` 在 `ncu` 下会挂起，必须**预编译 + `--no-build`**（同 §18/§19）。

## 24. 把 TMA B1 残差的第一块钉死：per-chunk 的 `E^tot` 必须进流水（P7-21）

§23.6 把 TMA + warp-specialization 之后的残差重定位到「`long_sb`（TMA/mbarrier 完成延迟）
+ 浅 pipeline + swizzle 读的 bank-conflict」，并给出两条下一步：**加深 pipeline** 或换
**CUTLASS `PipelineTmaAsync`**。本轮先做两件更便宜、可证伪的事：(A) 用现有数据把「加深 pipeline」
这条排除掉；(B) 找出 `long_sb` 里**具体是哪一次 global 访问**——因为 §23 相对 §22 的 K2 形状把
`long_sb` 从 2.25 抬到 3.52，而**同一批算术在 reg 内核里只有 2.25**，说明 TMA 版引入了一个新的
关键路径 global 读。回读代码，`b1_tma_probe.cu` 在 mma 循环里**逐 chunk 读 `E^tot`**：

```cpp
// b1_tma_probe.cu（§23 版，op2/op3 循环内）
float e0 = etot[((n*NT + c)*H + h)*KD + m*16 + g];
float e1 = etot[((n*NT + c)*H + h)*KD + m*16 + g + 8];
```

而 reg 内核（P7-19）是把它 **`cp.async` 预取进 smem** 的（`b1_reg_probe.cu:150`）。§23.1 为了
「省一次 TMA 与 512 B stage 分量」改成了 compute warp 直读 global——代价是**每个 chunk 的
`S *= e` 都卡在一个冷的 global 标量读后面**（$E^{tot}$ 是 `[B,NT,H,KD]`，B8T2048H8 下 67 MB，
每 chunk 新地址、L2 不命中），正是 `long_sb`。

### 24.1 受控实验：把 `E^tot` 搬回 TMA 流水（`B1T_ETMA`）

新增一个编译开关 `-DB1T_ETMA`：给 `Stage` 加 `float e[KD]`（512 B），用一张 **3D 张量图**
（`dims={KD,NT*H,B}`、`stride={4,KD*4,NT*H*KD*4}`、`box={KD,1,1}`、`SWIZZLE_NONE`）**每 chunk
多发一条 `cp.async.bulk.tensor.3d`**，`arrive.expect_tx` 的字节数 `+KD*4`；compute warp 从
`st->e[...]` 读。**算术一字未动**——所以 §24.2 Part A 里 `tmae` 与基线 TMA **逐位相同**。
（`harness/b1_tma_probe.cu`、`harness/check_bwd_b1_tma.py`、`harness/ncu_b1_tma_driver.py
--which etma`、`harness/probe_b1_tma_ncu.sh`。）

### 24.2 Part A：正确性——`E^tot` 走 TMA 不改变任何数值

`harness/check_bwd_b1_tma.out.txt`，6 组 case（B1T512H4 / B2T1024H8 / B1T250H3，BT=32/64）：

| 指标 | tmae vs **tma** | tmae vs FLA |
|---|---|---|
| `dh` / `dv2` / `dh0` max_abs | **0 / 0 / 0**（全部 case） | 4.88e-4 / 4.88e-4 / 2.12e-4（=bf16 底噪，同 §23） |

即只换「`e` 从哪个内存读」不影响结果（与 §23.2 同样的「只换搬运方式」论证，含非整除 T=250）。

### 24.3 Part B：kernel-only 计时（CUPTI 均值，µs/call）

| shape | BT | ns | tma | **tmae** | xTMA | FLA `dhu` | tmae xFLA |
|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 16 | 1 | 123.0 | **106.5** | **0.87** | n/a | — |
| B1 T512 H4 | 16 | 2 | 111.1 | **103.0** | 0.93 | n/a | — |
| B1 T512 H4 | 32 | 1 | 66.4 | 65.5 | 0.99 | n/a | — |
| B1 T512 H4 | 64 | 2 | 44.9 | 43.4 | 0.97 | 12.9 | 3.36 |
| B2 T1024 H8 | 16 | 1 | 280.9 | **234.1** | **0.83** | n/a | — |
| B2 T1024 H8 | 32 | 2 | 145.7 | **133.1** | **0.91** | n/a | — |
| B2 T1024 H8 | 64 | 2 | 87.8 | 84.3 | 0.96 | 22.7 | 3.71 |
| B8 T2048 H8 | 16 | 1 | 635.7 | **492.4** | **0.77** | n/a | — |
| B8 T2048 H8 | 32 | 1 | 325.7 | **290.3** | **0.89** | n/a | — |
| B8 T2048 H8 | 64 | 1 | 218.5 | **195.0** | **0.89** | 119.4 | **1.63** |
| B8 T2048 H8 | 64 | 2 | 205.6 | 205.5 | 1.00 | 119.4 | 1.72 |

三点：

1. **收益一致为正、且在 `NS=1` 与 BT=16/32 上最大**（0.77–0.91×）；BT=64 `NS=2` 打平（1.00×），
   因为 stage 变大后里面已无余量、且下一条要讲的 bank-conflict 回升。
2. **产生本轮全表最好的一行**：`B8T2048H8 BT=64 NS=1` 由 218.5→**195.0 µs**，`xFLA` 从 §23 的
   1.72（该行 NS2）收到 **1.63**——把「TMA + warp-spec」的收益从 ~15–20% 再往前推。
3. **`NS=1` 之所以赢**：`E^tot` 一进 smem，关键路径上再无暴露的 global 读，于是「浅 stage
   （103 KB，**2 CTA/SM**）」比「深 stage（169 KB，1 CTA/SM）」更好——见 §24.4。

### 24.4 Part D：`ncu` 复核——`long_sb` 确实被削掉，占用率被排除

`harness/probe_b1_tma_ncu.out.txt`（空闲卡，同场配对）：

| 配置 | long_sb | short_sb | wait | tensor% | bank-conflict | theo/ach occ | regs | Duration µs |
|---|---|---|---|---|---|---|---|---|
| reg64 BT64 NS1 B8T2048H8 | 2.22 | 1.62 | 1.50 | 8.83 | 0 | 12.50/6.25 | 128 | 244.0 |
| **tma BT64 NS1 B8T2048H8** | **4.35** | 0.85 | 1.60 | 9.68 | 0 | **15.62**/7.75 | 166 | 224.4 |
| **tmae BT64 NS1 B8T2048H8** | **3.86** | 0.90 | **1.24** | **10.83** | 0 | 15.62/7.75 | 167 | **198.8** |
| tma BT64 NS2 B8T2048H8 | 3.46 | 0.91 | 1.42 | 10.75 | 261050 | 7.81/7.74 | 195 | 208.1 |
| **tmae BT64 NS2 B8T2048H8** | **2.69** | 1.36 | 1.28 | **11.15** | 458011 | 7.81/7.68 | 196 | 207.8 |
| tma BT64 NS2 T512H4 | 2.66 | 0.89 | 1.43 | 11.48 | 4829 | 7.81/7.39 | 195 | 49.0 |
| **tmae BT64 NS2 T512H4** | **2.05** | 1.18 | 1.28 | **12.15** | 9618 | 7.81/7.35 | 196 | **45.3** |
| FLA `dhu` B8T2048H8 | 2.07 | 0.50 | **0.96** | 12.21 | 1778 | 6.25/6.15 | 146 | 119.8 |

四件事：

1. **`E^tot` 进 TMA 把 `long_sb` 削掉 0.6–0.8**：`NS=1` 4.35→**3.86**、`NS=2` 3.46→**2.69**、
   T512H4 2.66→**2.05**（已很接近 FLA 的 2.07）。`wait` 也随之下移（1.60→1.24、1.42→1.28）——
   这直接**证明了 §23 的 `long_sb` 里有相当一块就是那个 per-chunk `e` 全局读**，而非纯 TMA 延迟。
2. **mma 密度继续上台阶**：tensor `9.68→10.83%`（NS1）、`10.75→11.15%`（NS2）、`11.48→12.15%`
   （T512），最接近 FLA 的 12.21%。这与「`e` 从关键路径移开→mma 更早发射」自洽。
3. **占用率不是残差**（本轮最重要的排除）：FLA `dhu` 的 achieved 占用率 **6.15%**、理论 6.25%，
   **比 ours（7.7% / 7.81–15.62%）还低**，却快 **1.7×**。同样地，`NS=1` 让 ours 摸到 **2 CTA/SM**
   （theo 15.62%）却没换来成比例的加速（grid=128 CTA < 132 SM，多数 SM 只有 1 CTA）。**故「加
   占用率/加 warp」可正式排除**——残差是**每 warp 的发射/指令效率**（FLA `wait` 0.96、`short_sb`
   0.50，ours 即使最优也 1.24 / 0.90）。
4. **代价是 bank-conflict 回升**（NS2: 261k→458k）：因为 stage 多了一个 `float e[128]`（512 B）
   使 `sdh/sdu` 的 smem 基址平移、且 `NS=1` 的 2 CTA/SM 让同一 SM 上两 CTA 争用——仍只占
   shared-load wavefront 个位数百分比（§23.5 量级），不是主因。

### 24.5 结论与下一步

- **钉死并修掉一块残差**：TMA 版把 per-chunk `E^tot` 从流水里拿掉，是 §23 相对 §22 的 `long_sb`
  抬升（2.25→3.52）的一个**具体来源**；把它用一条 512 B 的 TMA 搬回 smem 后，`long_sb` 降
  0.6–0.8、`wait` 降 0.16–0.36、tensor 密度逼近 FLA，**算术逐位不变**，kernel-only 快
  **0.77–0.98×**，并把全表最好一行推到 **`xFLA = 1.63`**（B8T2048H8 BT=64 **NS=1**，195 µs）。
- **「加深 pipeline」这条被数据排除**：§23 的 NS sweep 本就平坦（NS1≈NS2≈NS3），本轮 `ncu` 又证
  占用率（含 2 CTA/SM）不是杠杆。**在 `E^tot` 修好之后，最优配置反而回到 `NS=1`**——浅 stage 带
  来的 **2 CTA/SM** 比深 pipeline 更值。故 §23.6 的两条下一步里，「更多 stage」作废。
- **剩下的 1.63× 残差**：落在 `wait`（1.24 vs FLA 0.96）、`short_sb`（0.90 vs 0.50）与 `long_sb`
  （3.86 vs 2.07 尚余的 TMA/barrier 完成延迟）——即**指令级发射效率**，与占用率、stage 深度、
  swizzle bank-conflict 都无关。这正是 §14/§17 在 Triton 上一路走到的位置：**手写 mbarrier +
  `ldmatrix` 的最小实现已经逼近其结构上限，再往前要么换 CUTLASS `PipelineTmaAsync`（把
  producer/consumer 与 `cp.async.bulk` 的完成延迟交给成熟实现），要么直接接受 Triton 版**。
- **可复用副产品**：`make_map_e` / `tma3d`（3D、`SWIZZLE_NONE`、box `{KD,1,1}`）是「把一个
  per-chunk 的 `[128]` fp32 向量塞进 TMA 流水」的最小样例，可搬到任何 K2 形状 kernel 的门控/缩放
  因子载入，**替掉关键路径上的 global 标量读**。

**本节踩坑**：

1. **3D TMA 张量图的 `globalStrides` 从 dim1 起算，不是 dim0**。第一版写 `gstr={4, KD*4}`
   （把 dim0 的元素大小也放进去了），`cuTensorMapEncodeTiled` 直接返回错误——**静默** `fprintf`
   后 kernel 用未初始化的图启动（数值仍可能「看起来对」），且**污染后续 CUDA 状态**（同一进程里
   紧接着的 reg 内核 `cudaFuncSetAttribute` 竟报 `rc=-3`）。正确写法 `gstr[0]=KD*4`（dim1
   stride）、`gstr[1]=NT*H*KD*4`（dim2 stride）。修复后错误消失、reg 也恢复。
2. `box` 含 fp32 的 `[128]` 一行 = 512 B > 128 B，故**不能用 128 B swizzle**（那要求 box 内维
   ≤128 B），必须 `CU_TENSOR_MAP_SWIZZLE_NONE`；好在 `e` 是逐标量读（每 lane 同一地址），
   本就不需要 `ldmatrix`。

## 25. 把 §17 的 Triton 流水 B1 落进真实组装链（P7-22）

§17（P7-14）在**孤立**的 bf16 Triton B1 原型上量出：只加一条软件流水（`tl.range(...,
num_stages)`），B1 就从 1.96× 拉到 **1.00×/0.85× FLA**，并下结论「镜像 K2 的输入流水是**全部**
收益来源，拆链冗余」。但那条结论只停在**原型**上——`docs/07` §12（P7-9）把 B1 的 bf16 操作数落进
真实组装链时，仍写死 `num_stages=1`，**流水从未进链**。本节把它补上，回答「§17 的流水在**部署
粒度 BT=16 的真实反向链**里能否兑现」。

新增 `harness/check_bwd_b1_pipe_e2e.py`（`harness/check_bwd_b1_pipe_e2e.out.txt`，空闲卡
`CUDA_VISIBLE_DEVICES=1`）与 §12 同口径（同一份 workspace 重建、同一份 FLA golden、`wall`+CUPTI
`dev`）。

### 25.1 唯一的改动：反向 chunk 循环换成前向 `tl.range`

`_dhu_kernel_bf16` 的状态递推是沿 chunk **逆序**的串行 carry：

```
for c in range(NT - 1, -1, -1):   # 原始（§12）
```

Triton 的自动流水器只对 `tl.range` 生效，且对**负步长**循环不易识别，故改成前向 `tl.range` +
索引翻转（`c = NT - 1 - cc`），**循环体的算术一字未动**：

```
for cc in tl.range(0, NT, num_stages=NS):   # 本节的唯一改动
    c = NT - 1 - cc
    ... # 与 §12 完全相同
```

`NS` 作为 `tl.constexpr` 传入，一次编译扫 `{1,2,3,4}`（`num_stages=1` 的编译选项仍为 1，流水只由
`tl.range` 控制）。递推的 **loop-carried 依赖**（`b_dh`）不变，流水只重排**全局加载**（`q/k/w/do/dv/e`）
与计算，故数值应逐位保持。

### 25.2 Part A：流水不改变任何数值

6 组 case（含**非整除 T=250/513**、`g=0`、`B=2`），与 §12 同一批：

| 对比 | 结果 |
|---|---|
| `NS∈{2,3,4}` 链 vs `NS=1` 链 | `dq/dk/dv/dg` **max_abs=0（逐位相同）**、`dh0` **max_abs=0**、`dbeta` ≤1.5e-8（B2 里 `db` 的 `atomic_add` 归约序不同） |
| 流水链 vs FLA autograd（跨 6 case 最坏 `max_rel`） | `dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 / dbeta 6.01e-3 / dg 7.06e-3 / dh0 8.27e-3` |

末尾一行与 §12（P7-9）报的**逐位一致**——即本节 `NS=1` 精确复现了旧链，`NS>1` 只重排加载、
零数值代价。（6 组 case 的逐项明细见 `*.out.txt`。）

### 25.3 Part B：链级计时——流水把 B1 从「追平」推到「反超」FLA `dhu`

`dev` 为 CUPTI 纯 kernel 时间（含整条反向 `dAv→B1→B2→B3→反向 cumsum`），`dev/FLA` 越小越好：

| case | FLA dev | NS1 dev（旧链） | NS2 | **NS3** | NS4 |
|---|---|---|---|---|---|
| T=512 H=4 | 203.4 | 116.8 (0.57×) | 101.6 (0.50×) | **97.5 (0.48×)** | 96.8 (0.48×) |
| B2 T=1024 H=8 | 657.0 | 397.5 (0.61×) | 366.9 (0.56×) | **340.4 (0.52×)** | 338.7 (0.52×) |
| B8 T=2048 H=8 | 4119.1 | 2291.9 (0.56×) | 2231.4 (0.54×) | **2223.9 (0.54×)** | 2229.7 (0.54×) |
| T=250 H=3（非整除） | 143.7 | 74.6 (0.52×) | 70.2 (0.49×) | **70.0 (0.49×)** | 70.0 (0.49×) |

即整条反向 `dev` 从 §12 的 FLA **0.51–0.61×** 进一步降到 **0.48–0.54×**（`NS=3` 已到平台，`NS=4`
无额外收益）。逐 kernel 归属——**只 B1 变**：

| case | FLA `dhu` | NS1 (旧) | NS2 | **NS3** | NS4 |
|---|---|---|---|---|---|
| T=512 H=4 | 54.0 | 54.4 | 39.3 | **34.9** | 34.5 |
| B2 T=1024 H=8 | 137.1 | 122.1 | 87.3 | **61.9** | 61.9 |
| B8 T=2048 H=8 | 567.0 | 387.3 | 327.6 | **318.9** | 326.3 |
| T=250 H=3 | 27.7 | 23.5 | 19.2 | **19.0** | 19.0 |

两个要点：

1. **B1 现在所有 shape 都反超 FLA `dhu`**：`T512H4` 54.4→**34.9**（FLA 54.0）、`B2T1024H8`
   122.1→**61.9**（FLA 137.1，**2.2×**）、`B8T2048H8` 387.3→**318.9**（FLA 567.0，**1.78×**）、
   `T250H3` 23.5→**19.0**（FLA 27.7，1.46×）。这正是 §17 在孤立原型上预测的方向，现在在**真实
   链、部署粒度 BT=16** 上兑现。
2. **链级收益被 B2/B3 摊薄**：`B8T2048H8` 的链 `dev` 只降 0.97×（B1 省 68 µs，而 B2/B3 占 ~1.9 ms），
   小 shape（`T512H4`）B1 占比高、链降 0.83×。故「B1 流水」是**正确但非全局**的杠杆——与 §12/§15
   的「B2/B3 才是大头的反向后半段」一致。

### 25.4 Part C：孤立 B1 的 BT×NS 扫描（只看相对 trend）

`harness` 内另给一次**孤立** kernel-only 扫描（同一 build、`BV=32`）：

| case | BT=16 (NS1→NS3) | BT=32 (NS1→NS3) | BT=64 (NS1→NS3) |
|---|---|---|---|
| T=512 H=4 | 54.4 → 44.7 | 36.9 → 33.3 | 29.1 → 24.5 |
| B2 T=1024 H=8 | 143.1 → 80.0 | 91.3 → 60.3 | 75.0 → 46.3 |
| B8 T=2048 H=8 | 443.0 → 364.9 | 284.4 → 267.8 | 241.1 → 187.4 |
| T=250 H=3 | 32.8 → 28.2 | 23.9 → 21.7 | 20.1 → 18.1 |

三点：(a) **流水在部署粒度 BT=16 也有效**（`B2T1024H8` 143→80）；(b) `BT=64` 更快但整链不跟到 64
（§15 已验证净亏）；(c) **`BT=64/NS=4` 直接 OOM**（`Required 243200 > 232448`）——stage 数 ×
每 stage 的 `[64,K]`+`[64,V]` 操作数 smem 越界，反过来说明 `BT=64` 的深流水没有 smem 余量。

### 25.5 一个必须讲清的测量学陷阱：孤立 vs 链内归因差 ~1.3×

Part C（孤立）与 Part B（链内）对**同一个 kernel、同一组操作数、同一 NS** 给出不同绝对值：
`T512H4 BT=16 NS3` 在 Part C 是 **44.7 µs**，而 Part B 的链内归因是 **34.9 µs**。Part D 做了受控
诊断——把 B1 单独测 vs 紧跟在 `dav` 之后**背靠背**测、在同一 profiled 窗口里按名归因：

| | 孤立 dev | 链内（dav 后）归因 |
|---|---|---|
| T512H4 NS1 | 59.2 | **44.6** |
| T512H4 NS3 | 49.7 | **35.1** |

即**孤立的小 grid（16 CTA）launch 的固定启停/排空被算进其 device-active 窗口**，而背靠背时这段就
被后续 kernel 的排队吸收。结论：**链级口径一律用 Part B（链内归因）**，
Part C 只用来读**相对 trend**（孤立也会把 NS 收益压缩：BT16 `NS1→NS3` 孤立 1.22× vs 链内 1.56×）。
这一条对后续任何「孤立测小 kernel」都适用。

### 25.6 结论与下一步

- **兑现了 §17 的下一步**：把 Triton 软件流水从孤立原型落进真实链，`NS=3` 让 B1 在所有 shape
  **反超 FLA `dhu`**（最大 2.2×），整条反向 `dev/FLA` 从 §12 的 0.51–0.61× 收到 **0.48–0.54×**，
  且 **逐位不改变数值**。这是「**直接接受 Triton 版**」这条路线（§24.5 列的第二个选项）在链上的
  落地证据：B1 的 Triton 侧工程化**已到收敛**（`wait`/`short_sb` 残差是 Triton 自动流水的结构上限，
  与 §17 一致）。
- **剩余的链级瓶颈明确转向 B2/B3**：`B8T2048H8` 上 B1 只占 ~0.32 ms / 2.22 ms，即使 B1 再快一倍，
  链级也只再降 ~0.14×。§15 已证「整链跟到 64」净亏，故**反向的下一块硬骨头是 B2 `wy_dqkg`/B3
  `intra` 的 16 块实现**，而非继续压 B1。
- **手写 CUDA B1（§18–§24）的定位**：它已到「每 warp 发射效率」的结构上限（§24.5），目标卡 CUDA
  侧若仍要追，唯一未试的是换 CUTLASS `PipelineTmaAsync`（把完成延迟交给成熟实现）；但**在 Triton
  侧，本节已给出与 FLA 同档乃至更快的部署级 B1**——两条路线**都已经或接近**满足需求。
- **方法学**：§25.5 的「孤立小 kernel 归因偏高 ~1.3×」应作为本专题后续任何 kernel-only 微观基准
  的固定提醒（链内归因才是链级口径）。

## 26. 链级新瓶颈 B2/B3 的定位：B2 是 DRAM-bound、B3 是「无 TensorCore 的 FMA/L1-bound」（P7-23）

§25 的结论是「链级瓶颈从 B1 转向 B2/B3」——因为 B1 已经被软件流水推到反超 FLA `dhu`，而
`B8T2048H8` 的链 `dev` 里还剩 ~1.9 ms 是 B2/B3。§15 又早已证「整链跟到 64 净亏」。本节第一次
**把 B2/B3 单独量清楚**（此前 `ncu` 只剖析过 B1），并回答一个问题：**这两块是不是还能靠便宜的
launch/tiling 旋钮（§17/§21/§25 那一类杠杆）拿回来，还是必须结构性重写？**

新增 `harness/check_bwd_b2b3_e2e.py`（`harness/check_bwd_b2b3_e2e.out.txt`，空闲卡）+
`harness/probe_b2b3_ncu.sh` / `harness/ncu_b2b3_driver.py`（`harness/probe_b2b3_ncu.out.txt`）+
`harness/probe_b2b3_hdh_bf16.py`（`harness/probe_b2b3_hdh_bf16.out.txt`）。

### 26.1 Part A：链级 2/3 的开销确实是 B2/B3

把 §25 的组装链（pipe B1、`NS=3`）按 kernel 名做 CUPTI device 归因：

| case | chain dev | `dAv` | B1 `dhu` | **B2 `wy_dqkg`** | **B3 `intra`** | rcumsum | **B2+B3 占比** |
|---|---|---|---|---|---|---|---|
| T=512 H=4 | 98.0 µs | 3.5 | 35.0 | 21.7 | 15.0 | 2.0 | 37.4% |
| B2 T=1024 H=8 | 341.0 µs | 11.4 | 61.9 | 126.7 | 80.8 | 6.1 | 60.8% |
| B8 T=2048 H=8 | 2232.0 µs | 74.7 | 307.9 | **906.0** | **579.7** | 45.7 | **66.6%** |
| T=250 H=3（非整除） | 69.8 µs | 2.5 | 18.8 | 15.1 | 12.3 | 1.6 | 39.2% |

大 shape 上 **B2+B3 = 66.6%**，B1 只剩 13.8%。这直接给了 §25「别再压 B1」的定量依据。

### 26.2 Part B：免费的 launch/tiling 旋钮基本已用尽，只剩 B3 的 `num_warps`

对 B2/B3 做孤立受控扫描（BT=16，部署粒度；CUPTI 均值）：

- **`num_stages` 完全无效**（B2 的 `i_k`/`i_v` 与 B3 的 `j` 都是 2–16 次的短归约/标量循环，Triton
  自动流水器识别不出可重叠的全局加载；`ns=1/2/3` 三列在 ±0.5% 内）。**与 B1 相反**——B1 有一个长的
  chunk 串行循环才吃得到流水。
- **`num_warps` 有信号但形状相关**：B2 在 `T512H4` 上 `nw=8` 略好（13.6 vs 15.7 µs），但大 shape
  `nw=4` 最优（121.7 vs `nw=8` 152.7）；**B3 则一律 `nw=2` 最优**（`T512H4` 16.6 vs `nw=4` 19.0；
  `B2T1024H8` 76.8 vs 89.3）。
- **tiling 已是最优**：B2 的 `BK=BV=64` 最好（`BK=64/BV=64` 900 µs；`BK=128` → 2109 µs；
  `BK=BV=128` → **37.8 ms**，寄存器爆炸）；B3 的 `BC=16` 最好（`BC=32` → 2022 µs）。当前链的
  `min(64,K)`/`BC=CHUNK` 恰好命中。

### 26.3 Part C：把 B3 `num_warps=2` 落进链——逐位不变，链降 2–6%

B3 唯一的免费杠杆是 `num_warps=2`。落进组装链后：

- **正确性**（6 组 case，含**非整除 T=250/513**、`g=0`、`B=2`）：`nw=2` 链 vs `nw=4` 链的
  `dq/dk/dv/dg` **`max_abs=0`（逐位相同）**、`dbeta ≤1.5e-8`（B2 的 `atomic_add` 归约序）；vs FLA
  autograd 的最坏 `max_rel` 与 §25.2 逐位一致：`dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 /
  dbeta 6.01e-3 / dg 7.06e-3 / dh0 8.27e-3`。**`num_warps` 只改线程映射、不改算术**，故逐位保持。
- **计时**（链内归因）：B3 内核 `T512H4` 14.9→**12.9**（0.87×）、`B2T1024H8` 81.0→**68.8**（0.85×）、
  `B8T2048H8` 573.9→**452.9**（0.79×）、`T250H3` 12.2→**10.2**（0.84×）；整链 `dev` `nw2/nw4` =
  **0.94–0.98×**。

### 26.4 Part D：`ncu` 归因——B2 撞 DRAM、B3 撞「无 TensorCore 的 FMA/L1」

`B8T2048H8`、BT=16、`nw=4`（`probe_b2b3_ncu.out.txt`）：

| 指标 | B2 `wy_dqkg` | B3 `intra` |
|---|---|---|
| Duration（ncu） | 529.8 µs | 528.1 µs |
| **DRAM Throughput** | **59.0%** | 28.0% |
| **L1/TEX Throughput** | 36.1% | **70.9%** |
| L2 Throughput | 60.4% | 42.8% |
| **Compute (SM)** | 16.7% | **82.3%** |
| **tensor pipe** | 4.88% | **0.05%** |
| DRAM 读 / 写 | 778 / 270 MB | 311 / 184 MB |
| Registers/thread | **247**（→ Block Limit Registers=**2**） | 64 |
| Achieved occupancy | **12.4%** | 49.1% |
| 主 stall | **`long_scoreboard` 6.61** | `long_scoreboard` 3.20 / `not_selected` 2.28 |

两点：

- **B2 是 DRAM-bandwidth bound**（59% SOL；`lts` 1.68 GB、`long_sb` 6.6 拍 = 每条发射间隔 ~一半在等
  全局加载），**并被寄存器压住占用率**（247 regs/thread → 每 SM 只 2 个 block、achieved 12.4%）。
  它的 DRAM 流量大头是 **fp32 的 `h`/`dh`**（链里 `reconstruct` 把整套 workspace `.float()`，B1 又
  刻意把 `dh` 输出保持 fp32）；`h`/`dh` 各 `[K,V]`、按 `i_k` 分块被**重读**。
- **B3 是 L1/FMA bound 且几乎不用 TensorCore**（tensor 0.05%，SM 82.3%）——它本质是**逐元素外积**的
  `dq2 += dAqk[:,None]*kj[None,:]*…` 序列，不使用 `tl.dot` 的 MMA。这与 §15「B3@64 慢 4.8–5.5×」
  互洽：`ncu` 实测 B3 `BC=16/BT=64` 直接掉到 **2.29 ms**（REG 201、occupancy 12.4%），是寄存器/
  占用率塌陷。
- **口径提醒**：`ncu` 的 Duration（529.8）明显低于链内 CUPTI 归因（906），与 §25.5 的
  「孤立/不同 measured 窗口归因偏差」同源；本节结论只依赖**比值类指标**（SOL%、stall、占用率、寄存器）。

### 26.5 一个结构性杠杆：B2 的 `h`/`dh` 换 bf16

B2 撞 DRAM 且流量大头是 `h`/`dh`，最直接的省法是把这两个操作数从 fp32 换成 bf16（链里它们本是
fp32）。受控实验（`probe_b2b3_hdh_bf16.py`）：

| case | fp32 `h`/`dh` | bf16 `h`/`dh` | 速度 | 换 bf16 自身增量（`max_rel`） |
|---|---|---|---|---|
| T=512 H=4 | 16.9 µs | 14.6 µs | 1.16× | `dq 1.24e-3 / dk 1.63e-3 / dv 0 / dg 1.85e-3` |
| B8 T=2048 H=8 | 900.2 µs | 831.3 µs | 1.08× | `dq 1.50e-3 / dk 1.78e-3 / dv 0 / dg 1.95e-3` |

即 **只值 1.08–1.16×**（`h`/`dh` 只占 B2 总流量的一部分，`q/k/g/do/dv/v/vn` 仍 fp32），但代价是
给六项梯度加 ~1.5–2e-3 的 bf16 操作数误差。**结论：这是「有收益但有精度代价」的分项，不是量级项。**

### 26.6 结论与下一步

- **B2/B3 是链级新瓶颈已钉死**（`B8T2048H8` 占 66.6%），且**便宜旋钮已用尽**：`num_stages` 无效、
  tiling 已最优、只有 B3 的 `num_warps=2` 有 1.16–1.27× 的免费收益（落链后逐位不变、整链 −2–6%）。
- **两块的性质完全不同，须分别处理**：**B2 = DRAM-bandwidth bound + 寄存器压占用**（247 regs、
  12.4% occ、`long_sb` 6.6）；**B3 = 无 TensorCore 的 FMA/L1-bound**（tensor 0.05%、SM 82%）。
- **对目标卡 CUDA 的设计输入（B2/B3 支路）**：
  1. **B2 要减 DRAM 流量与寄存器**：把 `h`/`dh` 落盘为 bf16（1.08–1.16×，精度 ~1.5e-3）、去掉按
     `i_k` 的重复读、拆小 tile 降寄存器以提占用率；**不要**加大 `BK`（寄存器爆炸）。
  2. **B3 要消掉「逐元素外积」比重**：把对角/次对角块也走 `tl.dot`/MMA（把 tensor 0.05% 提起来），
     或减少每 chunk 的冗余重算；**不要**把 `BC` 或 `BT` 加大（`BC=32`→2022 µs、`BT=64`→2.29 ms）。
  3. **维持 BT=16 块链**（§15 已证 64 净亏，本节又证 B3@64 占用率塌陷），把省下的预算投到 B2/B3
     的结构上。
- **本轮性质**：这是一次**纯剖析 + 免费旋钮收口**的增量——给出了 B2/B3 的硬件 bound 与继续优化的
  方向，没有做结构性重写（那需要新 kernel，属下一轮）。B3 `num_warps=2` 可直接并入部署配置。

## 27. B3 `intra` 的结构性重写：把对角块 MMA 化（P7-24）

§26.6 给目标卡 CUDA 反向支路的第 (2) 条是「B3 要消掉逐元素外积比重（把对角/次对角块也走
`tl.dot`）」。本轮就做这一条，并把它从孤立 kernel 落进真实组装链。

### 27.1 为什么在部署粒度下 B3 *只有* 对角块

B3 的 grid 是 `(NK*NC, NT, B*H)`，其中 `NC = ceil(BT/BC)`（`check_bwd_intra_kernel.py:386`，
FLA 同构 `chunk_intra.py:395-423`）。链里部署的是 `BT=BC=CHUNK=16`，于是 **`NC=1`**、`i_i = i_kc % NC ≡ 0`：

- Part A（`if i_i > 0`，`check_bwd_intra_kernel.py:272`）**永不进入**；
- Part C（`if i_i < NC_eff - 1`，`:329`）**永不进入**。

也就是说，部署粒度下 B3 每个 program 执行的**全部**计算是 Parts B/D 那两个
`for j in range(bc)` 的逐 `j` 外积循环——而 baseline 里**所有 `tl.dot` 都在 Parts A/C**。这正好
解释了 §26.4 的 `tensor 0.05%`：TensorCore 一次也没被用到。（BT=64 时 `NC=4`，Parts A/C 才参与。）

### 27.2 恒等式：对角块就是两个 `tl.dot`

Part B（`dq2`，chunk_intra.py:534-546 的非 SAFE_GATE 版）对 query 行 `a`、通道 `l`：

```
dq2[a,l] = Σ_{j≤a} dAqk[a,j] · k[j,l] · 2^{g[a,l]-g[j,l]}
         = 2^{g[a,l]} · Σ_j  Mq[a,j] · ( k[j,l] · 2^{-g[j,l]} )        Mq = (a≥j) ⊙ dAqk
```

把门控指数 **移出收缩**（符号拆开，代数上严格相等）：一次 `[BC,BC]@[BC,BK]` 的 `tl.dot` 就顶掉
整个 `j` 循环。Part D（`dkt`，`:639-655`）同理：

```
dkt[a,l] = Σ_{j≥a} dAqk[j,a] · q[j,l] · 2^{g[j,l]-g[a,l]}
         = 2^{-g[a,l]} · Σ_j  Sq[a,j] · ( q[j,l] · 2^{g[j,l]} )      Sq = (a≤j) ⊙ dAqkᵀ
```

**数值上是稳的**：对 `j≤a`，被移出的 `2^{-g[j]}` 在 `j=a` 处与前置因子 `2^{g[a]}` 精确相消
（`g` 单调降到行内），故中间和中最大的项正好等于最终项，无灾难性放大——实测（§27.3）落在 tf32 底噪。

**这不是新发明**：FLA 的 `SAFE_GATE=True` 分支**早已**这么做——Part B 见 `chunk_intra.py:520-526`
（`b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk) * exp_b_g_diag_qk`），Part D 见 `:620-637`
（`b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp) * exp_neg_b_g_diag_kk`）。而默认路径
`safe_gate=False`（`chunk.py:47`）走的是我们继承的逐元素循环。所以本轮的实质是：**把 FLA
safe-gate 分支已验证过的对角 MMA 恒等式，移植到默认的逐元素路径上**，并量清它的收益与内容。

实现见 `harness/check_bwd_b3_mma.py::build_intra_mma`（Parts A/C 一字未改，只换 Parts B/D）。

### 27.3 Part A：孤立 kernel 正确性——落 tf32 底噪（含非整除 T、BT=16）

`check_bwd_b3_mma_correctness.out.txt`，8 组 case（`BT=64/16`、`T=64…1024`、非整除 `T=250/513`、
`g=0`）：MMA 对角版 vs 逐元素 baseline / vs FLA 的 `dq/dk/db/dg` **四项最坏 `max_rel`**：

| 对比 | `dq` | `dk` | `db` | `dg` |
|---|---|---|---|---|
| MMA vs baseline | 1.14e-3 | 9.83e-4 | 1.02e-3 | 1.04e-3 |
| MMA vs FLA | 1.14e-3 | 9.83e-4 | 1.02e-3 | 1.04e-3 |

全在 §26 记录的 tf32 底噪（~1e-3）上——换累积序（外积 FMA → `tl.dot` tf32）带来的差异，与
Parts A/C 本就走 `tl.dot` 是同一量级，**没有引入新误差来源**。

**踩坑（值得记）**：第一版把对角 `dAqk`/`dAkk` 平铺的**列**下标写成绝对 token `i_ti+o_i`，而
`dAqk` 的最后一维其实是 **chunk-内局部** key 下标（`i_i*BC+o_i`，见 baseline `:283-284` /
FLA `:515`）。指纹很有欺骗性：**只有 `i_t=0`（即 `T≤BT` 的单 chunk，如 `T=64 K=32`）正确，
其余 case 全错到 O(1)**——因为 `i_t=0 ⟹ i_ti = i_i*BC`，两种写法恰好相等。改成
`i_i*BC+o_i` 后所有 case 归位（本 harness 专门留了 `T=64 H=1 K=32 BT=64` 这一行来卡这个退化）。

### 27.4 Part B：孤立 kernel 计时——快 **1.5–2.0×**

`check_bwd_b3_mma_timing.out.txt`（CUPTI `dev` 与 event-pair `wall`，均值）：

| case | BT | nw | base dev | mma dev | dev 比 | wall 比 |
|---|---|---|---|---|---|---|
| T512 H4 | 64 | 2/4 | 22.1 / 22.3 | 11.9 / 12.0 | **0.54×** | 1.00× |
| B2 T2048 H8 | 64 | 2 / 4 | 262.4 / 263.8 | 134.3 / 167.3 | 0.51× / 0.63× | 0.53× / 0.64× |
| T512 H4 | 16 | 2 / 4 | 16.6 / 18.9 | 9.5 / 9.6 | 0.57× / 0.51× | ~1.0× |
| B2 T2048 H8 | 16 | 2 / 4 | 142.4 / 169.4 | 91.0 / 94.5 | 0.64× / 0.56× | 0.66× / 0.58× |
| B8 T2048 H8 | 16 | 2 / 4 | 522.3 / 631.6 | 348.4 / 365.1 | 0.67× / 0.58× | 0.67× / 0.58× |

`dev` 一律快 **1.5–2.0×**；小 shape 的 `wall` 打平（host 派发 60 µs 盖过 kernel）。与 §26 一致，
`nw=4` 与 `nw=2` 各有胜负（`nw=2` 在 B8 大 shape 更好），MMA 版把 B3 从「FMA/L1-bound」侧挪开。

### 27.5 Part C：落进真实组装链——梯度不变、链降 ~10%

`check_bwd_b3_mma_chain.out.txt`，把 §25 的 `tl.range` 流水 B1（`NS=3`）链里的 B3 换成 MMA 版，
6 组 case（含非整除 `T=250/513`、`g=0`、`B=2`）：

- **vs FLA autograd 的六项最坏 `max_rel`**：`dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 / dbeta 6.01e-3 /
  dg 7.05e-3 / dh0 8.27e-3`——**与 §25.2、§26.3 记录的 baseline 链逐值相同**（MMA 后没有退化）。
- **本次重写自身的增量**（MMA 链 vs baseline 链）：`dq 1.42e-3 / dk 1.09e-3 / dv 0 / dbeta 3.0e-5 /
  dg 6.08e-3`（`dg` 最坏在 `g=0`，与 §27.3 同源）。
- **链级 `dev`**（CUPTI，µs；FLA 对照）：

| case | FLA dev | base dev | MMA dev | MMA/base |
|---|---|---|---|---|
| T512 H4 | 204.4 | 97.9 | 89.0 | **0.91×** |
| B2 T1024 H8 | 662.1 | 339.4 | 298.0 | **0.88×** |
| B8 T2048 H8 | 4145.0 | 2216.4 | 1956.7 | **0.88×** |
| T250 H3 非整除 | 141.4 | 68.7 | 61.7 | 0.90× |

整链降 **9–12%**（B3 占链 ~26%、kernel 快 ~2×，乘积吻合）。小 shape `wall` 基本不动（host-bound）。

### 27.6 Part D：`ncu` 复核——TensorCore 起来了，但 kernel 转成 L2/访存 bound

`probe_b3_mma_ncu.out.txt`，`B8 T2048 H8 BT=16`（部署粒度）：

| 指标 | baseline（逐元素） | MMA 对角 | 变化 |
|---|---|---|---|
| **Duration** | 526.98 µs | 256.86 µs | **0.49×** |
| `tensor` pipe % | 0.05% | 2.61% | **×52** |
| `fma` pipe % | 38.79% | 10.11% | 0.26× |
| Compute (SM) % | 82.24% | 26.91% | 0.33× |
| L1/TEX % | 70.86% | 46.45% | 0.66× |
| L2 % | 42.96% | 74.20% | ×1.73 |
| DRAM % | 28.07% | 57.43% | ×2.05 |
| Memory throughput | 941 GB/s | 1.92 TB/s | ×2.04 |
| Registers/thread | 64 | 64 | = |
| Achieved occupancy | 49.01% | 48.82% | = |
| `stall long_sb` | 3.22 | 16.71 | ×5.2 |

读法：`tensor` 从 **0.05%→2.61%（×52）**、`fma` 从 38.8%→10.1%、SM 从 82%→27%——**工作确实从 FMA/LSU
搬到了 TensorCore**，Duration 随之 **0.49×**。但 kernel 随即**转成访存/L2 bound**（L2 43%→74%、
DRAM 28%→57%）：注意 DRAM%翻倍而 Duration 减半，二者相乘＝**总字节不变**（Mem throughput 941 GB/s→1.92 TB/s），
并非多读了数据，而是同样的字节在更短时间内搬完。代价是 L1 命中率 62.6%→20.0%、`long_sb` 3.22→16.71
——对角平铺 `[BC,BC]` 的加载模式比原来逐列取 L1 局部性差。`nw=2` 下同表趋势一致（Duration 547.9→260.3）。

### 27.7 结论与下一步

- **§26.6 第 (2) 条兑现**：B3 对角 MMA 化后孤立 kernel 快 **1.5–2.0×**、整链 `dev` 降 **9–12%**、
  六项梯度与 FLA 的误差**逐值不变**（仍落 bf16/tf32 底噪）。且证明了 §27.2 的恒等式：默认路径只要
  套上 FLA `SAFE_GATE` 分支早已在用的对角 `tl.dot` 就够，无需新数学。
- **但 B3 并未「修好」**：TensorCore 起来了却只到 2.6%，kernel 现在是 **L2/访存 bound**。B3 的下一个
  杠杆不再是 MMA，而是**减少对角平铺的重复读**（`dAqk/dAkk` 每 `i_k` 重读一份、`k/g` 每 `i_k` 重读）
  或提升其 L1 局部性。
- **链级瓶颈仍在 B2**（§26：B8T2048H8 上 B2+B3 曾占 66.6%；本轮砍掉 B3 一半后 B2 更突出）。下一步
  是按 §26.6 第 (1) 条做 **B2 的结构性重写**（减 DRAM 流量/寄存器：`h/dh` bf16、去按 `i_k` 重复读、
  拆 tile 降寄存器提占用），维持 BT=16 块链。
- **产物**：`harness/check_bwd_b3_mma.py`（`..._correctness/…_timing/…_chain.out.txt`）+
  `harness/ncu_b3_mma_driver.py` / `harness/probe_b3_mma_ncu.sh`（`..._b3_mma_ncu.out.txt`）。B3 这版
  可直接并入部署配置（`nw=2` 在大 shape 更好）。

## 28. B2 `wy_dqkg` 的结构性重写：延后 `q/k/g` 加载消掉寄存器溢出（P7-25）

§27.7 把链级瓶颈交回到 B2：§26 量出 B2 是 **DRAM-bound（59% SOL）+ 寄存器压占用**
（`Registers/thread=247 → Block Limit Registers=2 → achieved occupancy 12.4%`），而 P7-24 又砍掉
B3 一半，于是 B2 在链里更突出。本轮按 §26.6 第 (1) 条做 **B2 的结构性重写**——但先做最便宜、
**算术完全不变**的那一种：**把 `q/k/g` 的加载从 `i_k` 循环顶部挪到 `i_v` 循环之后**。

### 28.1 动机：部署配置下 B2 正在**溢出（spill）**

对**未改动**的 B2 直接读编译期寄存器（`kernel[grid](...)` 返回的 handle 上的 `n_regs/n_spills`，
BT=16、BK=BV=64、三种 shape 同值）：

| `num_warps` | baseline `n_regs` / `n_spills` | late-load `n_regs` / `n_spills` |
|---|---|---|
| 2 | 255 / **190** | 255 / 82 |
| 4（部署） | 255 / **12** | **252 / 0** |
| 8 | 254 / 0 | 224 / 0 |

部署用的 `nw=4` 正好卡在 Triton 的 255 寄存器上限并**溢出 12 个**，`nw=2` 更是溢出 190 个。
`ncu` 侧独立回读也确认了压力（`nw=4`：247 regs/thread，`Block Limit Registers=2`）。这正是 §26.4
「247 寄存器压占用」那句话背后更严重的事实——**它同时在往 local memory 打溢出流量**。

**根因（读代码就能定位）**：baseline 在 `i_k` 循环**顶部**就加载 `q/k/g`（`check_bwd_dqkg_kernel.py:244-249`），
但它们**只在 `i_v` 循环结束之后**才被用到（衰减重标定与门控链，`:290-305`）。于是 `b_q/b_k/b_g`
（各 `[BT,BK]`）在整个 `i_v` 循环里**一直活着**——而那正是 `b_h/b_dh`（各 `[BV,BK]`）和三个
`[BT,BK]` 累加器 `b_dq/b_dk/b_dw` 的活跃区。峰值压力落在 `i_v` 体里，超过 255 就溢出。

### 28.2 改动：只挪位置，不动算术

把三个 `tl.load(p_q/p_k/p_g, ...)` 从 `i_k` 循环顶部移到 `i_v` 循环**之后**（`b_gn` 的加载留在原处，
它本来就在 `i_v` 之前、量也小）。运算顺序、累加次序、`tl.debug_barrier()` 位置**一字未改**，因此
六个输出应**逐位相同**。实现见 `harness/check_bwd_b2_struct.py::build_dqkg_late`
（对应 `_wy_dqkg_kernel_late`）。

### 28.3 Part A：正确性——与 baseline **逐位相同**（含非整除 T、B=2、BT=16）

`check_bwd_b2_struct.out.txt`，7 组 case（`BT=64/16`、`T=64…1024`、**非整除 T=250/513**、`B=2`）：

| 对比 | `dq` | `dk` | `dv` | `db` | `dg` | `dA` |
|---|---|---|---|---|---|---|
| late vs baseline（最坏 `max_rel`） | 0 | 0 | 0 | 0 | 0 | 0 |
| late vs FLA（最坏 `max_rel`） | 0 | 0 | 0 | 1.7e-7 | 6.1e-8 | 0 |

**与 baseline 完全逐位相同（全部 `max_abs=0`）**，且与 baseline 一样，vs FLA 只在 `db/dg` 有
~1e-7 的差（B2 `db` 的 `atomic_add`/跨块归约序，§26.3 已记）。即这次重写**没有引入任何数值变化**。

### 28.4 Part B/C：孤立 kernel 快 **0.84–0.89×**（spill 消掉的直接收益）

`nw=4`、`BT=16`，三种口径互相印证（CUPTI `dev` / event-pair `wall` / `triton.testing.do_bench`）：

| case | dev base→late | dev 比 | wall 比 | do_bench base→late | bench 比 |
|---|---|---|---|---|---|
| T=512 H=4 | 14.9 → 12.7 µs | **0.85×** | 1.00× | 31.4 → 28.0 | **0.89×** |
| B2 T=1024 H=8 | 121.9 → 108.4 | **0.89×** | 0.89× | 130.7 → 117.2 | 0.90× |
| B8 T=2048 H=8 | 896.3 → 800.4 | **0.89×** | 0.89× | 905.4 → 809.5 | 0.89× |
| T=250 H=3 非整除 | 13.9 → 11.7 | **0.84×** | 1.00× | 28.1 → 23.7 | 0.85× |

即**纯粹消掉 12 个 spill 就近 11%**。`nw=4` 寄存器从 255→252（虽仍 >170，见 §28.6）但**零 spill**；
`nw=8` 从 254→224。这是「live range 缩短 → 峰值寄存器下降 → 不再溢出」的直接体现。小 shape 的
`wall` 不动是 host 派发主导（§26/§27 同现象）。

### 28.5 Part D：落进真实组装链——梯度**逐值不变**、链 `dev` 降 **3–5%**

把 late B2 替换进 §27 的组装链（P7-22 pipe B1 `NS=3` + P7-24 MMA B3），6 组 case（含非整除
`T=250/513`、`g=0`、`B=2`）：

- **vs FLA autograd 的六项最坏 `max_rel`**：`dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 / dbeta 6.01e-3 /
  dg 7.05e-3 / dh0 8.27e-3`——**与 §25.2、§26.3、§27.5 记录的 baseline 链逐值相同**（重写不改数值）。
- **重写自身增量**（late 链 vs baseline 链）：`dq/dk/dv/dg` **`max_abs=0`（逐位）**、`dbeta ≤3.7e-8`
  （`atomic_add` 序）。
- **链级 `dev`**（CUPTI，µs；FLA 对照）：

| case | FLA dev | base dev | late dev | late/base |
|---|---|---|---|---|
| T=512 H=4 | 206.2 | 88.0 | 85.4 | **0.97×** |
| B2 T=1024 H=8 | 662.6 | 297.9 | 285.1 | **0.96×** |
| B8 T=2048 H=8 | 4103.8 | 1961.4 | 1862.4 | **0.95×** |
| T=250 H=3 非整除 | 141.4 | 61.1 | 58.7 | 0.96× |

链降 **3–5%**（B2 占链 ~43%、kernel 快 ~11%，乘积吻合）；`B8T2048H8` 1961→**1862 µs**。小 shape
`wall` 基本不动（host-bound）。

### 28.6 Part E：`ncu` 复核——寄存器确实降下来，但占用率**没动**（口径提醒）

`probe_b2_struct_ncu.out.txt`，`B8 T2048 H8 BT=16`：

| 指标 | baseline | late-load |
|---|---|---|
| Registers/thread | **247** | **210**（`nw=8`：222→199） |
| Block Limit Registers | 2 | **2**（未变） |
| Achieved occupancy | 12.36% | 12.41% |
| DRAM 读 / 写 | 777.6 / 270.0 MB | **778.0 / 269.9 MB（不变）** |
| DRAM Throughput | 59.08% | 57.17% |
| tensor pipe | 4.88% | 4.73% |
| `stall long_scoreboard` | 6.61 | 6.85 |

两点读法：

- **字节数逐位不变**（`redirect` 只挪 live range，不挪数据）——确认这不是「少读」而是「少溢出」，
  且证实了 §26.4「B2 DRAM 大头是 `h`/`dh`」的画像（每 chunk `h`+`dh` = `2·K·V·4B` = 128 KB；
  `B8T2048H8` 有 `B·NT·H = 8192` 个 chunk）。
- **占用率没有变**（仍 2 blocks/SM）：`210×128×3 = 80640 > 65536`，**需 ≤170 regs 才够 3 blocks**。
  本轮的收益不是「占用率上来」，而是**去掉了溢出流量**。
- **口径提醒（重要）**：本次 `ncu` 自报的 `Duration` 给 late **反而略慢**（546.8 vs 529.0 µs），与
  CUPTI/`wall`/`do_bench` 三个独立口径（一致地 late 快 ~11%）**矛盾**。这与 §25.5「孤立小 grid 的
  `ncu` 归因窗口与 CUPTI 不同源、不可混用绝对值」同源；**本轮结论只依赖 CUPTI/`do_bench` 的比值**，
  `ncu` 仅取**静态寄存器回读**与**字节/SOL 比值类**指标。

### 28.7 结论与下一步

- **B2 的 register-spill 被消掉**：仅靠**把 `q/k/g` 的加载延后到 `i_v` 循环之后**（live range 缩短），
  部署配置 `nw=4` 的溢出 12→0、`nw=8` 寄存器 254→224；孤立 kernel **0.84–0.89×**、组装链 **0.95–0.97×**
  （`B8T2048H8` 1961→1862 µs），且**六项梯度逐位不变**（与 §25/§26/§27 baseline 链同值）。这是一次
  **零数值代价的结构性收益**。
- **它没有解决 B2 的根本 bound**：占用率仍被寄存器钉在 2 blocks/SM（需 ≤170 regs 才到 3），
  DRAM 字节数不变、仍是 DRAM-bound（~59% SOL）。§26.6 第 (1) 条里剩下的是**真正的结构项**：
  1. **把 `h/dh`/大 tile 的 live range 继续缩短或降精度**（`h/dh` bf16 在 §26.5 只值 1.08–1.16×）；
  2. **去掉按 `i_k` 对 `do/vn/dv/v` 的重复读**（`K/BK` 次）——需要改累加结构（把 `i_v` 提到外层、
     对 `dq/dk/dw` 用跨 `i_k` 的部分和落地），是下一轮的结构项；
  3. **拆 tile 到 ≤170 regs 换 3 blocks/SM**——但 §26.2/§28.4 都实测 `BK=BV=32` 反而更慢
     （MMA 效率/循环开销），故须配合别的改动；
  4. 更彻底的是 **B1→B2 融合**，让 `dh` 不落盘、不重读。
- **产物**：`harness/check_bwd_b2_struct.py`（`harness/check_bwd_b2_struct.out.txt`）+
  `harness/ncu_b2_struct_driver.py` / `harness/probe_b2_struct_ncu.sh`
  （`harness/probe_b2_struct_ncu.out.txt`）。这版 late-load B2 可直接并入部署配置。

## 29. B2 `wy_dqkg` 的 §26.6 两条结构项落链核验：`h/dh` bf16 与 `i_k` 去重（P7-26）

§26.6 给目标卡 CUDA B2 支路列了两条结构项：(1) 把 `h`/`dh` 落成 bf16；(2) 去掉按 `i_k` 对
V 侧操作数的重复读。P7-25 只做掉了最便宜的 spill，本轮把这两条在**真实组装链 + 部署 dtype** 下
核验，各自给出一个明确结论（(1) 有条件收益、(2) 基本被 L2 吸收、不做）。

新增 `harness/check_bwd_b2_hdh.py`（`harness/check_bwd_b2_hdh.out.txt`，空闲卡）+
`harness/ncu_b2_hdh_driver.py` / `harness/probe_b2_hdh_ncu.sh`（`harness/probe_b2_hdh_ncu.out.txt`）。

### 29.1 一个此前被忽略的事实：部署链里 B2 跑的是**全 fp32**

P6-8/P7-25 组装链的 `_prep_b` 只把 `qg/kg/w` 转 bf16——那是 **B1** 的操作数。B2 的 `inp2`
字段全部直接取自 `fi`（fp32）：`q,k,v_new,g,beta,A,h,do,dh,dv` 一律 fp32（实读 `fi` 的
`dtype`：只有 `Mqk` 是 bf16，而 B2 不读它）。因此 §26.4/§28.6 那些 `ncu`（driver 默认
`--dtype bf16`）描绘的 **778 MB 读不是部署 mix**。新 ncu 用部署 dtype（全 fp32）：

| 配置（`B8 T2048 H8`, BT=16, nw=4） | DRAM 读 | DRAM 写 | DRAM SOL | Duration | regs | occ | L2 字节 |
|---|---|---|---|---|---|---|---|
| **部署：全 fp32 `h/dh`（BK=64）** | **1.63 GB** | 274.5 MB | **70.6%** | 802.9 µs | 252 | 12.39% | 3.02 GB |
| `h/dh` bf16（BK=64） | 1.03 GB | 271.9 MB | 53.4% | 725.8 µs | 242 | 12.40% | 2.14 GB |
| 全 fp32，BK=128（=K，无 `i_k` 循环） | 1.76 GB | **1.49 GB** | 62.3% | 1.56 ms | 255 | 12.40% | 6.09 GB |

部署 mix 下每 chunk 读 = 184 KB，其中 **`h`+`dh` = 2·K·V·4B = 128 KB = 66%**；全 fp32 时
`h`+`dh` 合计 **1.07 GB**（不是 §26.4 的 524 MB——那是 bf16 driver）。所以 §26.4「B2 DRAM
大头是 h/dh」的画像**方向对、数字要按部署 mix 上调**。

### 29.2 item (1) `h/dh` bf16：kernel 值 ~10%，但**落链净负**（需原生 bf16）

隔离（部署 mix，`check_bwd_b2_hdh.out.txt` Part A/A2）：

- **正确性**（bf16 vs fp32 `h/dh`，含非整除 T=250）：`dq ≤1.6e-3 / dk ≤2.3e-3 / dv = 0
  (逐位) / db ≤2.1e-5 / dg ≤1.9e-3 / dA ≤4.0e-4`（`dv` 逐位不动，因为它走的是 dAv 支路，
  与 `h/dh` 无关）。
- **计时**：CUPTI `dev` `0.90–0.91×` 大 shape（`B2T1024H8` 108.5→98.1、`B8T2048H8`
  800.0→724.6 µs），小 shape `0.98–1.01×`；`do_bench` 同样 `0.90–0.95×`。
- **ncu**：读 1.63→1.03 GB（**−37%**）、L2 3.02→2.14 GB，**但占用率纹丝不动**
  （12.39→12.40%）且 DRAM SOL 只有 70.6→53.4%——即砍掉 37% 字节后 B2 **仍不在带宽墙上**，
  印证它本质是**延迟/并行度受限**（`long_scoreboard` 主导，§26/§28）。

落进真实组装链（Part C/D）是一个**反例**：

| case | FLA dev | fp32 链 dev | bf16 链 dev | 链 dev 比 | B2 内核 fp32 | B2 内核 bf16 | B2 内核比 |
|---|---|---|---|---|---|---|---|
| T=512 H=4 | 206.2 | 85.9 | 95.7 | **1.11×** | 18.4 | 17.9 | 0.97× |
| B2 T=1024 H=8 | 662.6 | 286.3 | 346.4 | **1.21×** | 113.7 | 102.3 | 0.90× |
| B8 T=2048 H=8 | 4103.8 | 1863.4 | 2307.9 | **1.24×** | 803.9 | 725.5 | 0.90× |

链 dev **变慢 7–24%**，但按 kernel 归因 **B2 内核本身是 0.90–0.97×**（大 shape −10%）。
原因：harness 的 `hdh_bf16` 分支在**每次调用**里对 `h`/`dh` 做一次 `.to(bf16)`，是两个
`K·V` 规模的独立 elementwise kernel（`h`/`dh` 各 ~1.07 GB 读 + 0.54 GB 写），把它们算进了
链 dev，收益被吃光还倒亏。

**结论**：`h/dh` bf16 的收益是真的（B2 内核 ~10%），但**只有在 `h`/`dh` 是原生 bf16 时才能
兑现**——即前向 workspace 的状态、以及 B1 的 `dh` 输出要**端到端 bf16**，B2 入口不能做运行期
转换。设计输入更新为「(1) 有条件成立：需与前向/B1 的 bf16 出口对齐，收益上界 ≈ B2 内核 10%
≈ 整链 ~4%」。链正确性 vs FLA autograd 六项最坏 `dq 3.6e-3 / dk 3.1e-3 / dv 6.8e-3 /
dbeta 6.0e-3 / dg 8.1e-3 / dh0 8.3e-3`，与 §25.2/§26.3/§27.5/§28.5 的 baseline 链同阶；
bf16 相对 fp32 链自身增量 `dq ≤1.0e-3 / dk ≤1.4e-3 / dv = 0 / dbeta ≤4.8e-5 / dg ≤1.8e-3`。

### 29.3 item (2) `i_k` 去重：基本被 L2 吸收，**不做**

部署 `BK=64`、`K=128` ⟹ `i_k` 循环 2 次 ⟹ V 侧 `do/v_new/dv/v` 按 `K/BK` **重读**。理论
V 侧读 = `4·T·V·4B·B·H·2 = 512 MB`，去重后 256 MB（省 256 MB = 总读的 **16%**）。但：

- ncu 实测 `lts__t_bytes`（L2）= **3.02 GB ≫ DRAM 1.63 GB**（比 ~1.85），说明**重读基本命中 L2**，
  只把其中一部分漏到 DRAM；
- 想用 `BK=128`（=K，**无 `i_k` 循环**）隔离去重反而更糟：读升到 1.76 GB、**写爆到 1.49 GB**
  （255 寄存器 + 溢出 292 个）、时长 **1.56 ms**——去重的字节被溢出流量淹没，根本测不出净收益；
- 要真正去重须改累加结构（把 `i_v` 提到外层、`dq/dk/dw` 跨 `i_k` 的部分和落地），而 B2 寄存器
  已顶满（252），**任何「多留活量」都会把刚消掉的 spill 打回来**（§28）。

**结论：`i_k` 去重最多值 ~16% 且被 L2 部分掩盖，实现代价（改累加结构 + 寄存器反弹）大于收益，
在 Triton 原型里不做。** 若目标卡 CUDA 用 TMA 大块搬运 + 寄存器常驻状态（§22–24 的 K2 形状），
这个问题会被「一次把 V 侧搬进 smem 喂多条 mmma」自然消掉，不在 Triton 层解决。

### 29.4 顺手复核：占用率确实不是旋钮（Part B）

对部署 kernel（late-load，P7-25）扫 `BK/BV/num_warps`（`B8T2048H8`）：

| 配置 | regs | spill | dev |
|---|---|---|---|
| BK=64 **BV=64 nw=4（部署）** | 252 | 0 | **795.7 µs** |
| BK=32 BV=32 nw=4（162 reg，**3 blocks/SM**） | 162 | 0 | 993.0 µs |
| BK=64 BV=32 nw=4 | 250 | 0 | 940.2 µs |
| BK=128 BV=64 nw=4 | 255 | 300 | 1570.9 µs |

唯一进到 3 blocks/SM 的 `BK=BV=32` 反而慢 **1.25×**。**「拆 tile 到 ≤170 regs 换 3 blocks/SM」
（§28.7 item 3）不成立**——B2 不是占用率受限（同 §26.2/§28.7）。

### 29.5 结论与下一步

- **B2 的廉价/半廉价杠杆已全部走完**：spill（P7-25，B2 内核 −11%）；`h/dh` bf16（本轮，
  B2 内核 −10%，**需原生 bf16**，否则落链净负）；`i_k` 去重（本轮，≤16% 且 L2 吸收，不做）；
  拆 tile 提占用（本轮，反慢 1.25×）。
- 部署链 B2 画像（全 fp32）：读 **1.63 GB**、写 274 MB、DRAM SOL **70.6%**、occ 12.4%、
  regs 252、`long_sb` 主导 → **延迟/并行度受限**，砍字节（−37%）只换来内核 ~10%。
- **唯一剩下的真结构项是 §26.6 item 4 / §28.7 item 4：B1→B2 融合**（`dh` 根本不落盘、不重读），
  或把 `h/dh` 的原生 bf16 出口与「目标卡 CUDA 的 K2 形状」（寄存器常驻 + TMA，§22–24）一起做。
- **产物**：`harness/check_bwd_b2_hdh.py`（`harness/check_bwd_b2_hdh.out.txt`）+
  `harness/ncu_b2_hdh_driver.py` / `harness/probe_b2_hdh_ncu.sh`
  （`harness/probe_b2_hdh_ncu.out.txt`）。

## 30. B1→B2 融合原型：`dh` 根本不落盘（P7-27）

§29.5 把反向链里「唯一剩下的真结构项」定为 **B1→B2 融合**——今天 B1（状态伴随）把每块的状态伴随
`dh[B,NT,H,K,V]` 写 global，B2（`wy_dqkg`）再读回；而 B1 的 program 在逆序扫块时**本来就持有**
`dS == dh[c]` 在寄存器里，若 B2 的逐块工作在同一 program 里做，`dh` 就永远不碰 DRAM。本轮把它做成
**最小可运行原型**并量清「省流量」与「丢并行度/爆寄存器」的净账。产物
`harness/check_bwd_fused.py`（`harness/check_bwd_fused.out.txt`）+
`harness/probe_fused_regs.py`（`harness/probe_fused_regs.out.txt`）+
`harness/ncu_bwd_fused_driver.py` / `harness/probe_bwd_fused_ncu.sh`
（`harness/probe_bwd_fused_ncu.out.txt`）。

### 30.1 设计：为什么原型取 `BV = V`（无跨 V 归约）

B1 沿 `V` 分块（grid `(cdiv(V,BV), B·H)`），B2 沿 chunk 分块（grid `(NT, B·H)`）。B2 对 **整个 V**
收缩（`b_dq = do @ b_hT`、`b_dk = vn @ b_dhT`、`b_dw`、`b_dgk = Σ_V h⊙dh`…），所以一个只持 `V` 切片
的 program 只能算出**部分和**，B2 后段的**非线性/衰减收尾**（`b_dg = q⊙dq − k⊙dk + …`、
`dA ← Aᵀ(β⊙dA)Aᵀ`）必须先跨 V 归约——要么 `atomic_add` + 第二个 epilogue kernel，要么在
program 内 V 全量。最干净的**无原子**原型就是 **`BV = V`**：一个 program 管一个 `(b,h)`，逆序扫
全部 chunk，`b_dS[K,V]` 常驻寄存器，每块做完 B1 立刻做 B2 的这一块（`dh = b_dS` 不落盘），再更新
`b_dS`。grid 退化为 `(B·H,)`——**正是前向 K2 的形状**（head 并行、沿 T 串行）。

### 30.2 Part A：融合内核与「B1+B2」**逐位相同**

把融合内核的 `dq/dk/dv/dbeta/dg/dh0` 接上**同一条** B3 + 反向 cumsum 尾巴，4 组 case
（`T=512H4` / `B2T1024H8` / `B8T2048H8` / 非整除 `T=250H3`，P6-8 的同一份激活操作数、loss 含终态项）：

- **融合链 vs FLA `torch.autograd`**：六项最坏 `dq 3.62e-3 / dk 3.10e-3 / dv 8.82e-3 /
  dbeta 5.62e-3 / dg 4.35e-3 / dh0 7.33e-3`——与部署链（P6-8/P7-26 的 3.4e-3…8.3e-3）**同阶**，
  即 bf16/tf32 底噪，融合没有引入可测量偏差。
- **融合 B1+B2（不含 B3）vs 未融合 B1+B2，同一份操作数**：`dq/dk/dv/dg/dh0` 全部
  **`max_abs = 0`（逐位相同）**、`db` `max_abs ≤ 1.2e-7`（`tl.sum` 的归约序差）。融合内核的每一
  行算术、每一条 mma 操作数与「B1 kernel → 落盘 dh → B2 kernel」完全同序，只是把 `dh` 留在寄存器
  ——**证明融合在代数/数值上是无损的**。

### 30.3 Part B：计时——融合反而慢 **14–30×**

反向链计时（CUPTI `dev`，CHUNK=16 部署粒度）：

| case | 未融合 B1+B2 | 融合 B1+B2 | 比 | 未融合整链 | 融合整链 | 比 |
|---|---|---|---|---|---|---|
| `T=512 H=4` | 81.3 µs | 2477.1 µs | **30.5×** | 115.3 | 2506.6 | 21.7× |
| `B2 T=1024 H=8` | 265.8 | 5406.5 | **20.3×** | 392.6 | 5531.0 | 14.1× |
| `B8 T=2048 H=8` | 1373.6 | 19132.2 | **13.9×** | 2282.6 | 20124.7 | 8.8× |
| `T=250 H=3`（非整除） | 47.6 | 1202.6 | **25.3×** | 72.9 | 1232.6 | 16.9× |

融合把 B1+B2 从 ~1.4 ms 抬到 ~19 ms。这**不是**「省下的 `dh` 流量不够」，而是融合内核**根本没算完**
——它先被寄存器压垮（§30.4）。

### 30.4 归因：1786 次寄存器溢出把「省流量」变成「爆流量」

**(1) 编译期寄存器（`harness/probe_fused_regs.py`）**：

| kernel | regs | spill | smem | nw |
|---|---|---|---|---|
| **融合 B1+B2** | **255（上限）** | **1786** | 96.3 KB | 8 |
| B1 `dhu_bf16` | 160 | 0 | 22.5 KB | 4 |
| B2 `wy_dqkg` | 255 | 12 | 51.2 KB | 4 |

融合内核在一个循环体里同时活着 `b_dS[K,V]`（fp32，64 KB）、`trans(b_dS)`、`b_h[K,V]`、
`trans(b_h)` 以及全部 `[BT,K]`/`[BT,V]` 中间量——**远超寄存器文件**，于是溢出 **1786** 次（vs B1 的 0、
B2 的 12）。`launch__occupancy_limit_registers = 1 block`（255 regs 钉死占用率）。

**(2) `ncu` DRAM 字节（`B8T2048H8`，`harness/probe_bwd_fused_ncu.out.txt`）**：

| kernel | DRAM 读 | DRAM 写 | 时长 | DRAM% | regs |
|---|---|---|---|---|---|
| **融合 B1+B2**（grid 64） | **12.92 GB** | **18.93 GB** | **19.10 ms** | 49.7% | 255 |
| B1 `dhu_bf16`（grid 4×64） | 178 MB | 588 MB | 0.386 ms | 59.2% | 160 |
| B2 `wy_dqkg`（grid 128×64） | 1.65 GB | 309 MB | 0.898 ms | 65.0% | 255 |

未融合一对合计 **读 ~1.83 GB / 写 ~0.90 GB**；融合内核 **读 12.92 GB / 写 18.93 GB（总 31.9 GB，
≈17×）**——融合的**真实操作数**只有 ~0.9 GB（`h` 536 MB + `q/k/g/do/dv/v/vn` 各 ~67 MB），多出的
~12 GB 读 / ~18 GB 写**全是溢出流量**（local memory 经 L1/L2/DRAM）。原意想省的 `dh`
写+读（约 0.59 GB 写 + B2 `dh` 读的一多半 ≈ 0.8 GB，合计 ~1.4 GB）被 Σ溢出流量的 +30 GB 彻底淹没。

**(3) 融合强制的 `BV=V` 分区本身也是 2–4× 代价（Part C，B1 单独）**：先只看 B1，把它的
`V` 分区从部署的 `BV=32` 拧到融合强制的 `BV=V`：

| case | BV=32 NS3 | BV=64 NS3 | **BV=128 NS3** | BV128/BV32 |
|---|---|---|---|---|
| `T=512 H=4` | 46.1 µs | 64.6 | **206.7** | **4.5×** |
| `B2 T=1024 H=8` | 85.4 | 126.6 | **475.8** | **5.6×** |
| `B8 T=2048 H=8` | 396.2 | 368.2 | **1024.2** | **2.6×** |
| `T=250 H=3` | 29.1 | 39.1 | **116.3** | **4.0×** |

即「一个 program 持有整条 `[K,V]` 状态」这件事，**在 B1 单独**就慢 2.6–5.6×（program 数从 `4·B·H`
掉到 `1·B·H`，每 program 串行全 T、状态 64 KB 常驻）；融合再把 B2 的整块工作压进同一 program，
溢出后直接慢 14–30×。

### 30.5 结论与设计输入

- **B1→B2 融合在代数与数值上是无损的**（Part A：`dq/dk/dv/dg/dh0` 与未融合 **逐位相同**），但
  **朴素 `BV=V` 单 kernel 融合不可行**：状态 `[K,V]` + 逐块 B2 工作集**超过寄存器文件**，溢出
  **1786** 次，DRAM 流量从 2.7 GB 爆到 **31.9 GB（≈17×）**，端到端慢 **14–30×**。
- **根因有两层**、且都在「融合把并行结构改成 K2 形状」这一步：(1) `BV=V` 让 grid 退化为
  `B·H`、B1 单独即慢 2.6–5.6×（§30.4(3)）；(2) 把 B2 的 `[BT,K]`/`[BT,V]`/A 工作集塞进同一个
  program，寄存器需求叠加到溢出。
- **设计输入（更新 §29.5）**：B1→B2 融合**不是**在 Triton 里「把两段并成一个 program」就能兑现的
  杠杆。若要做，必须**改并行结构**——即**用 §22–24 的 K2 形状**（warp-specialized：状态由多个
  MMA warp 的累加器寄存器分片常驻、操作数走 TMA/`ldmatrix` 大块搬运、`dh` 在 smem/寄存器里直接喂给
  同 CTA 的 B2 阶段），而不是把整条 `[K,V]` 状态塞进一个 monolithic Triton program。换言之，
  §29 把融合列为「唯一剩下的真结构项」是对的，但**它的前置条件是 K2 形状，而不是一个独立可加的
  开关**——这与 §15/§17/§20 对「流水/拆链/K2 形状」的结论一致。
- **待验证**：本轮只证了「无原子、`BV=V`、单 kernel」的融合形态不可行；「K2 形状内融合」
  （寄存器分片常驻 + TMA + 同 CTA 内 B1→B2 交接）是未实现的下一步，其可行性/收益须在目标卡上、
  以 K2 为基座再测。
- **产物**：`harness/check_bwd_fused.py`（`harness/check_bwd_fused.out.txt`，Part A 正确性 + Part B
  计时 + Part C `V` 分区扫描）+ `harness/probe_fused_regs.py`（`harness/probe_fused_regs.out.txt`）+
  `harness/ncu_bwd_fused_driver.py` / `harness/probe_bwd_fused_ncu.sh`
  （`harness/probe_bwd_fused_ncu.out.txt`）。

## 31. B3 `intra` 直接消费 B2 的累加器：消掉链外三个 `b2+b3` elementwise 求和（P7-28）

§27.7 把 B3 的下一个杠杆定为「减少对角平铺的重复读 / 提升 L1 局部性」。回读链代码时先撞上一个
**更便宜、且与 kernel 本体无关**的结构项：组装链（`check_bwd_b1_pipe_e2e.py:162-166`）让 B3 在
**一份全新置零**的 `dq/dk/dg` 上跑，然后在 kernel **之外**物化三个整张 `[B,T,H,K]` 的 elementwise
求和：

```python
b3 = intra_triton(intra_k, inp3, CHUNK, CHUNK)          # B3 内部 b_dq2 + load(zeros)
dg = chunk_local_cumsum(b2["dg"] + b3["dg"], ...)        # 3 张 [B,T,H,K] 的加
return dict(dq=b2["dq"] + b3["dq"], dk=b2["dk"] + b3["dk"], ..., dbeta=b2["db"] + b3["db"])
```

一次反向要为此付出 **3 ×（2 读 + 1 写）** 的 `B*T*H*K` fp32 流量，外加 3 个 `torch.zeros` 的
memset kernel。这不是 FLA 的契约——FLA 的 `chunk_kda_bwd_intra` 把**传进来的 `dq/dk/dg` 当累加器**
（读旧值、加块内贡献、写到**另一个** `dq2/dk2/dg2` 缓冲）：

- `chunk_intra.py:554-560`：`p_dq`/`p_dq2`；`b_dq2 = b_dq2 + tl.load(p_dq)`；`tl.store(p_dq2, ...)`。
- `chunk_intra.py:657-667`：`p_dk/p_dg`、`p_dk2/p_dg2`；`b_dg2 += ... + tl.load(p_dg)`、
  `b_dk2 += tl.load(p_dk)`；分别 store 到 `dk2`/`dg2`。

也就是说，FLA 的 intra 本就要求传入的 `dq` 是**非零累加器**（`wy_dqkg` 的输出），我们的「喂零 +
链外求和」纯属 harness 的偷懒写法：既白读一遍零缓冲，又把 B3 的产出再读一遍、写一遍。

本轮把 B3 改成**就地累积进 B2 的 `dq/dk/dg`**。合法性来自 grid：B3 的 `i_k`（K 分块）与内层
子块轴**恰好划分**每个 `(token, k)`——任一元素只被一个 program 产出（`check_bwd_b3_mma.py:215-231`
的 `grid=(NK*NC, NT, B*H)`），所以 `dq/dk/dg` 不需要跨 program 累加，就地读写无竞争；只有 `db`
仍靠 `atomic_add`（`check_bwd_b3_mma.py:163`），而 B2 已把 `db` 全量 store 过（`torch.empty` +
逐块 store），`atomic_add` 叠在其上即为 `b2.db + b3.db`。kernel 本体的**每一行算术、每一条
`tl.load`/`tl.store` 地址都不动**（`check_bwd_b3_mma.py:160-162`、`207-210` 保持原样），只是指针
从「新零缓冲」换成「B2 的缓冲」。

### 31.1 Part A：语义核对——就地累积 == 零填充 + 链外求和（含 FLA 契约）

`harness/check_bwd_b3_accum.out.txt`，三组 case（`T=512H4 D=128` 可整除、`T=250H3` 非整除、
`BT=16` 部署粒度）：

| 量 | `B=2 T=512 H=4 K=128` | `B=1 T=250 H=3`（非整除） | `B=1 T=128 H=2 BT=16` |
|---|---|---|---|
| `dq` | **0** | **0** | **0** |
| `dk` | 9.60e-8 | 1.06e-7 | 5.48e-8 |
| `db` | 1.25e-7 | 7.57e-8 | 8.91e-8 |
| `dg` | **0** | **0** | **0** |

`dq/dg` **逐位相同**；`dk/db` 的 ~1e-7 来自浮点结合序：kernel 里是 `(b_dk2 + 0) + b_dkt` 再由
链外加 `acc`，就地版是 `(b_dk2 + acc) + b_dkt`，两者仅加法次序不同（`b + 0` 本身精确）。这不是
误差、是 fp32 非结合性，量级 1e-7 = 单精度一个 ulp 量级。

同时验证这**就是 FLA 的契约**（让 FLA 的 intra 吃非零 `dq/dk/dg`，再与「FLA(零)+acc」比）：
`dq 0 / db 0 / dk 9.60e-8 / dg 1.03e-7`——与我们的就地版**同一组数**，证明「读旧值累加」是
FLA 的原意，我们只是把这条语义落回自己的链。

### 31.2 Part B/C：落链正确性不变、链 `dev` 降 12–16%

`harness/check_bwd_b3_accum.out.txt`。**Part B**（六组，含非整除 `T=250/513`、`g=0`、`B=2`）：
就地链 vs 现链的自身增量 `dq/dv/dg` **逐位 0**、`dk/dbeta ≤9.2e-8`；就地链 vs FLA autograd
六项最坏 `dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 / dbeta 6.01e-3 / dg 7.05e-3 / dh0 8.27e-3`
——与 §25.2/§26.3/§27 的**同一组数**，即 bf16/tf32 底噪、无退化。

**Part C**（`NS=3` pipe B1 + MMA B3，CUPTI `dev` + event `wall`，空闲卡 1）：

| case | FLA `dev` | base `dev` | ip `dev` | ip/base | base `wall` | ip `wall` | wall× |
|---|---|---|---|---|---|---|---|
| `T=512 H=4` | 205.7 | 88.3 | **74.2** | **0.84×** | 384.3 | 291.6 | 0.76× |
| `B2 T=1024 H=8` | 664.0 | 298.9 | **261.8** | **0.88×** | 398.1 | 307.4 | 0.77× |
| `B8 T=2048 H=8` | 4144.6 | 1957.3 | **1685.2** | **0.86×** | 1984.6 | 1700.3 | 0.86× |
| `T=250 H=3`（非整除） | 141.2 | 61.5 | **52.4** | **0.85×** | 393.0 | 301.4 | 0.77× |

链 `dev` 降 **12–16%**、`wall` 降 **14–24%**。按 kernel 归因（`torch.profiler`
`self_device_time_total`）能直接坐实机制——**消失的正是 elementwise 部分**：

| case | elementwise/add `dev` base | ip | elementwise kernel 数/次 base | ip |
|---|---|---|---|---|
| `T=512 H=4` | 17.9 µs | **7.3** | 14 | **6** |
| `B2 T=1024 H=8` | 52.6 | **14.3** | 15 | **7** |
| `B8 T=2048 H=8` | 316.0 | **49.2** | 15 | **7** |
| `T=250 H=3` | 16.7 | **7.0** | 14 | **6** |

`B8 T=2048 H=8` 上 elementwise 归因从 316 µs → 49 µs（省 ~267 µs），与链 `dev` 的
1957→1685（省 272 µs）吻合；每次反向少 **8 个 elementwise/memset kernel**（3 个 `torch.zeros`
fill + 3 个加法 + cumsum 前的中间物化）。`ip dev / FLA dev` 落到 **0.36–0.41×**（FLA 绝对值在本
窗口偏高，含 autograd 与 `retain_graph`，只作旁证，主口径仍是 ip/base 的同源比值）。

### 31.3 结论与下一步

- **B3 `intra` 应就地累积进 B2 的 `dq/dk/dg`**（FLA 的 `chunk_intra.py:559/662-663` 契约），
  而不是「喂零 + 链外求和」。这是**纯链结构**改动：kernel 算术逐行不动，正确性只差 fp32 加法次序
  （`dk/db ~1e-7`，`dq/dg` 逐位），链 `dev` **0.84–0.88×**、`wall` **0.76–0.86×**。
- **代价与约束**：B3 的契约从「输出独立缓冲」变成「消费 B2 的缓冲」，故 B2 的输出在 B3 之后不可
  再用（本链确实不再用）。若将来把 B2/B3 融成一个 kernel，需重新审视（那时 `dq/dk/dg` 的累加
  可完全留在寄存器）。
- **对 §27.7「减少对角平铺重复读」的定位**：本轮证明 B3 链级开销里有一块**与 kernel 无关**的
  elementwise 税（`B8T2048H8` 上 316 µs≈16% 的链时间），先把它消掉再谈 kernel 内的 L1 局部性——
  下一步才是 §27.7 说的「对角平铺 `dAqk/dAkk` 每 `i_k` 重读、k/g 每 `i_k` 重读」的 L1 局部性优化。
- **产物**：`harness/check_bwd_b3_accum.py`（`harness/check_bwd_b3_accum.out.txt`，Part A 语义 +
  Part B 落链正确性 + Part C 计时与 per-kernel 归因）。

## 32. B3 `intra` 的 L1 局部性：重复读是**空**、真正的杠杆是把 k 循环**软件流水**（P7-29）

§27.7/§31.3 把 B3 的下一杠杆定为「减少对角平铺的重复读（`dAqk/dAkk` 每 `i_k` 重读、`k/g` 每
`i_k` 重读）、提升 L1 局部性」。回读 P7-24 的 MMA kernel（`check_bwd_b3_mma.py:64-212`）后，这个
假设里其实藏着**两处具体的冗余读**，本轮把它们逐一做成受控实验 —— 结论是**两处都不是杠杆**，而
真正的杠杆是第三件当时没写进列表的事：**把 `k` 分块从 grid 轴改回 program 内的循环并软件流水**。

**(R1) `k`/`g` 被同一 program 读了两次。** Part B/D 读「对角 key 行」`j = i_ti + o_i`
（`check_bwd_b3_mma.py:142-143` 的 `p_kd`/`p_gd`），但 `o_c = i_ti + o_i`，所以 `p_kd`/`p_gd`
与前面 `:98-99` 的 `p_k`/`p_g` 是**逐字节相同的地址**，mask 也相同（`m_jk == m_ck`，因为
`m_jd == m_c`）。原 kernel 把这两块又 load 了一遍。

**(R2) 对角 `dAqk`/`dAkk` 每 `k` 分块重读一份。** grid 是 `(NK*NC, NT, B*H)` 且
`BK=min(32,K)`（`check_bwd_b3_mma.py:219`），所以 `K=128` 时 `NK=4` 个 program 各自把同一份只依赖
`(chunk, head)` 的 `[BC,BC]` 对角块读一遍（`:135-138`）。取 `BK=K`（`NK=1`）即读一次。

本轮新增 `harness/check_bwd_b3_l1.py`：四个变体——
`reuse`（R1：`b_kd=b_k`,`b_gd=b_g`，算术不动）、`noreuse`（R1 的对照，照旧重读）、
`bkK`（R2：`BK=K`）、`loop`（新增 kernel，见下）；另有 `harness/ncu_b3_l1_driver.py` /
`harness/probe_b3_l1_ncu.sh`。

### 32.1 Part A：正确性——`reuse`/`bkK` 与 P7-24 **逐位相同**，`loop` 只差 fp32 加法序

`harness/check_bwd_b3_l1_correctness.out.txt`，8 组 case（含非整除 `T=250/513/1023`、`g=0`、
`BT=16/64`、`BC=16`）：

| 变体 | `dq` | `dk` | `db` | `dg` | vs FLA（最坏） |
|---|---|---|---|---|---|
| `reuse` vs P7-24 | **0** | **0** | 1.3e-7 | **0** | `dq 1.14e-3 / dk 9.8e-4 / db 1.0e-3 / dg 1.04e-3` |
| `noreuse` vs P7-24 | **0** | **0** | 1.6e-7 | **0** | — |
| `bkK` vs P7-24 | **0** | **0** | 1.5e-7 | **0** | `dq 1.14e-3 / dk 9.8e-4 / db 1.0e-3 / dg 1.04e-3` |
| `loop_ns2` vs P7-24 | **0** | 1.1e-7 | 1.3e-7 | 8.5e-8 | — |

`reuse`/`bkK` **逐位相同**（`db` 的 ~1e-7 只是 `atomic_add` 顺序），因为它们是纯重排/常量折叠，
结果数学上同一串运算。`loop` 把 `b_eg` 提前复用（`b_QG=b_q*b_eg`），只差一个乘法的结合序，落
~1e-7。全部 vs FLA 仍在 tf32 底噪 ~1e-3。

### 32.2 Part B：R1/R2 在部署粒度**测不出**——`reuse` 恒 1.00×、`BK` 只值 ~5%

`harness/check_bwd_b3_l1_timing.out.txt`（孤立 kernel，CUPTI `dev`）。

- **R1 是空的**：`reuse` 与 `noreuse` 在**所有** shape/nw 上 `dev` 比值恒为 **1.00×**——编译器
  （Triton/CSE）早已把 `p_kd/p_gd` 那两条与 `p_k/p_g` 相同的 load 消掉，手工「复用」不产生任何
  差别。故 §27.7 说的「k/g 每 i_k 重读」在生成代码里根本不存在。
- **R2 只在非部署粒度帮忙**：`BK=64` 在部署 `BT=16` 的大 shape 稳定 **0.95–0.96×**
  （`B2 T=2048 H=8` 94.6→90.0、`B8 T=2048 H=8` 365.0→346.0），小 shape 打平或略慢；
  `BK=128`（`NK=1`，理论上彻底消 R2）在 `BT=16` 反而**中性**（94.6/365.0 不变），只在非部署的
  `BT=64` 有 `0.74–0.81×`。即 R2 的重复读被 L2 吸收了（见 §32.4 的 DRAM 字节）。

### 32.3 真正杠杆：把 `k` 分块改成 program 内 `tl.range` 循环并**软件流水**

`BT=BC=16/NC=1` 时 B3 每个 program 只做「一个 chunk × 一个 k 块」的少量工作，**所有 q/k/g 的
global load 都在 kernel 开头发出，随后长等**——这正是 P7-24 报的 `long_scoreboard 16.71`、
`L1 hit 20%`、而 `tensor` 只有 2.61% 的成因。修法是**换并行结构**：新增 `_intra_kernel_loop`
（`check_bwd_b3_l1.py`），grid 由 `(NK*NC,NT,B*H)` 改成 `(NT,B*H)`，program 内 `for i_k in
tl.range(0,NK,num_stages=NS)`；对角 `dAqk/dAkk`（R2）与 beta 循环不变、提前到循环外只读一次。
算术与 P7-24 **逐行同构**（`NC=1` 时 Parts A/C 不执行，只剩对角公式）。

`harness/check_bwd_b3_l1_timing.out.txt`（`reuse`/`noreuse` 已并入，此处列 `loop`）：

| case（部署 `BT=16`） | P7-24 mma | `BK=64` | `loop nw4 bk64 ns1` | `loop nw4 bk64 ns2` | `loop nw4 bk64 ns3` |
|---|---|---|---|---|---|
| `B1 T=512 H=4` | 9.4 | 9.5 | 12.0 (1.27×) | 10.1 | **9.6** |
| `B2 T=2048 H=8` | 94.6 | 89.9 | 100.8 (1.07×) | **76.6 (0.81×)** | 77.5 |
| `B8 T=2048 H=8` | 365.0 | 346.0 | 370.9 (1.02×) | 289.4 (0.79×) | **288.3 (0.79×)** |

**关键指纹**：`NS=1` 的循环版**比 P7-24 的 grid 版还慢**（1.02–1.27×，少了许多 program 的并行
度），只有 `NS≥2` 才反超——证明收益**全部来自 `tl.range` 的软件流水**（把下一 k 块的 load 提前到
当前 k 块的计算里），而不是「循环」本身。大 shape 上 `B2/B8` 各降 **~20%**；小 shape（`T=512`）
因归约太短、流水填不满而打平/略负。

### 32.4 Part C：落进真实组装链——梯度**逐值不变**、链 `dev` 0.97×

`harness/check_bwd_b3_l1_chain.out.txt`。**正确性**（六组，含非整除 `T=250/513`、`g=0`、`B=2`）：
`loop` 链 vs FLA autograd 六项最坏 `dq 3.42e-3 / dk 3.02e-3 / dv 6.80e-3 / dbeta 6.01e-3 /
dg 7.05e-3 / dh0 8.27e-3`——与 §25.2/§26.3/§27/§31.2 **同一组数**；`loop` 链 vs P7-24 链自身增量
`dq/dv` **逐位 0**、`dk ≤9.2e-8`、`dbeta ≤3.7e-8`、`dg ≤2.4e-6`（加法序）。

**计时**（`NS=3` pipe B1 + acc B3，CUPTI `dev`）：

| case | FLA `dev` | P7-24 `dev` | `loop` `dev` | loop/ref |
|---|---|---|---|---|
| `T=512 H=4` | 201.9 | 87.9 | 88.8 | 1.01× |
| `B2 T=1024 H=8` | 661.5 | 294.7 | **284.5** | **0.97×** |
| `B8 T=2048 H=8` | 4139.1 | 1940.1 | **1874.5** | **0.97×** |
| `T=250 H=3`（非整除） | 141.6 | 61.1 | 63.7 | 1.04× |

链级只 **0.97×**，因为 B3 在整链里占比已不大（B3 孤立省 ~77 µs @`B8T2048H8`，链省 ~66 µs，占比
吻合）；非部署小 shape 仍打平。

### 32.5 Part D：`ncu` 复核——DRAM 字节**逐位不变**，`long_scoreboard` 16.7→4.0

`harness/probe_b3_l1_ncu.out.txt`（`B8 T=2048 H=8`，`BT=16`；ncu 锁频故绝对 `Duration` 与 CUPTI
不同源，只看比值/机制）：

| 配置 | `long_sb` | `short_sb` | `wait` | `tensor` | DRAM% | L2% | occ | `Duration` |
|---|---|---|---|---|---|---|---|---|
| P7-24 mma | **16.74** | 1.36 | 1.29 | 2.61 | 57.3 | 73.8 | 48.6% | 257.4 |
| loop `ns=1` bk64 | 9.59 | 0.89 | 1.04 | 2.34 | 51.7 | 61.8 | 18.4% | 285.1 |
| loop `ns=2` bk64 | **5.72** | 0.91 | 0.81 | 3.41 | 74.3 | 82.5 | 18.3% | 198.1 |
| loop `ns=3` bk32 | **3.98** | 1.44 | 1.13 | 3.47 | 75.3 | 88.3 | 30.4% | 195.5 |

三个机制确认：

1. **DRAM 字节四档逐位相同**（读 **311.2 MB**、写 **182.2–182.9 MB**）——R1/R2 的「重复读」**根本
   不进 DRAM**（L2 全吸收）。这是 §32.2「R1 空、R2 只有 ~5%」的硬件解释：所谓重复读是 L2 命中，
   不是带宽。
2. **流水把 `long_scoreboard` 从 16.74 压到 3.98**（`ns` 越大越低），`Duration 257→195`（0.76×），
   与 CUPTI 的 0.79× 同向——**这就是收益来源**（隐藏 load 延迟）。
3. **占用率反降**（48.6%→18–30%）却更快：再证 B3 不是占用率受限；流水后 `L2 82–88%`、
   `DRAM 74–75%` 成为最高 pipe——B3 现在是**（更彻底的）L2/DRAM 带宽 bound**。

`reuse`/`bkK` 与 `mma` 的 DRAM 字节与 `long_sb` 基本相同（表中未列），印证「消重复读」无论从
指令还是字节角度都不成立。

### 32.6 结论与下一步

- **§27.7 的「减少对角平铺重复读 / 提升 L1 局部性」是空命题**：R1（k/g 双读）被编译器 CSE
  （`reuse` 恒 1.00×）；R2（`dAqk/dAkk` 每 `i_k` 重读）被 L2 吸收（DRAM 字节逐位不变，
  `BK=64` 只值 ~5%、`BK=128` 在部署粒度中性）。**B3 的重复读不是瓶颈**。
- **真正杠杆＝把 k 分块从 grid 轴搬进 program 内并 `tl.range` 软件流水**（`_intra_kernel_loop`）：
  孤立 kernel 大 shape **0.79–0.81×**、`long_scoreboard 16.7→4.0`、DRAM 字节不变；落链 `dev`
  **0.97×**（`B8T2048H8` 1940→1875），六项梯度逐值不变。**部署配置**：`nw=4`、`BK=64`、
  `num_stages=2`（大 shape）或 `3`（`BK=32`）；小 shape 打平，可回退 grid 版。
- **定位**：B3 kernel 级优化**到此收敛**——它已彻底是 L2/DRAM 带宽 bound，而流量大头是 P7-28 引入的
  `dq/dk/dg` **fp32 累加器读改写** + q/k/g 读。再往下只能靠**结构**（B2/B3 融合把累加器留寄存器，
  §31.3/§30 已列为下一步）而非 B3 内的 tiling/局部性。
- **产物**：`harness/check_bwd_b3_l1.py`（`..._correctness/…_timing/…_chain.out.txt`）+
  `harness/ncu_b3_l1_driver.py` / `harness/probe_b3_l1_ncu.sh`（`probe_b3_l1_ncu.out.txt`）。

## 33. B2+B3 融合：把 `dq/dk/dg` 累加器留在寄存器里（P7-30）

§32.6 / §31.3 / §30.5 把反向链的下一个结构项定为「**B2/B3 融合**」：既然 §32 证明 B3 已彻底
L2/DRAM 带宽 bound，而其流量大头正是 P7-28 引入的 `dq/dk/dg` **fp32 累加器读改写**，那么把
B3 折进 B2、让累加器根本不落盘，应当能省掉这块流量。本轮把它做成原型并量清楚。

### 33.1 为什么融合只需要「对角」B3

部署粒度是 `BT = BC = CHUNK = 16 ⟹ NC = 1`（`check_bwd_b3_mma.py:215-232`）。由 §27.1，此时 B3
的 grid `(NK*NC, NT, B*H)` 里 `i_i ≡ 0`，**Part A（`i_i>0`）与 Part C（`i_i < NC_eff-1`）都被
跳过**，B3 只剩对角 Parts B/D——恰好是两个 `tl.dot`（`check_bwd_b3_mma.py:130-210`）。

B2 的 program 是 `(NT, B*H)`，**天然覆盖整块**，且它在内部 `i_k` 循环结束时已算完整块
`dAkk`（其 `b_dA`，即落盘的 `dA`），也已有 `q/k/g/beta`。所以把 B3 折进 B2 只差：把 `b_dA` 与
从 `dAqk`（`dAv` 输出）读来的对角块拼出 Parts B/D，加到 B2 的 `dq/dk/dg` 累加器上，再存一次。
`dAkk` 于是**根本不必落盘**（它此前只被 B3 消费）。

融合后的单 kernel（`harness/check_bwd_b2b3_fused.py:build_fused_b2b3`，grid `(NT, B*H)`）与未融合
pair 的**代数逐行相同**，差别只在 `dq/dk/dg` 由「B2 写→B3 读→B3 写」变成「B2 寄存器累加→只写一次」。

### 33.2 Part A：孤立正确性——只剩 fp32 加法次序（含非整除 T）

`harness/check_bwd_b2b3_fused.out.txt`。把融合 kernel 与「B2(原) + B3(MMA 对角)」放在**同一份
fp32 操作数**上（`dAkk` 取 B2 的 `dA`），6 组 case（`K/V=64/128`、非整除 `T=250/513`、`B=2`）：

| 量 | 最坏 `max_rel` | 说明 |
|---|---|---|
| `dq` | **0（逐位）** | B2 与 intra 的 dq 贡献相加次序与原链一致 |
| `dv2` | **0（逐位）** | 完全在 B2 段，未动 |
| `dk` | 8.4e-8 | intra `dk2/dkt` 的 fp32 加法次序 |
| `db` | 2.2e-7 | B2 的 `db` + intra `db` 相加次序 |
| `dg` | 1.1e-7 | 同上 |

即融合在数值上无损，只剩单精度不可结合性（`b + 0` 精确，次序不同）量级 ~1e-7。

### 33.3 Part B：孤立计时——`BK=K` 的寄存器代价与解法

关键约束：要让 `dq/dk/dg` 常驻寄存器，`BK` 必须 = `K`（Triton 不能给 `[BT,K]` 切片赋值）。
但 `BK=K` 把 B2 原本 `[BV,64]` 的 `h/dh` 操作数、以及 `[BT,64]` 的各累加器全翻倍，**立刻溢出**：

| 配置 | regs/spills | `B8T2048H8` dev（vs 部署基线） |
|---|---|---|
| B2 原（nw=4 / nw=8） | 255/12 / 254/0 | — |
| fused `BK=K, BV=64, nw=4` | 255/**182** | 1.13× |
| fused `BK=K, BV=64, nw=8` | 255/**60** | 1.05× |
| fused `BK=K, BV=128, nw=4` | 255/**222** | 1.31× |
| fused `BK=K, **BV=128, nw=8**` | 255/**42** | **0.83×** |

解法是**把 `i_v` 循环也去掉**（`BV = V = 128`，单次迭代）：既少一层循环开销，又让编译器少一份
在飞行操作数，溢出从 60 降到 42。部署基线取 `B2(nw=4) + B3(nw=4)`（链里各 kernel 的实际配置），
`fused(BV=128,nw=8)` 的孤立 dev 为：

| case | 基线 dev | fused dev | 比 |
|---|---|---|---|
| `T=512 H=4` | 24.7 | 14.1 | **0.57×** |
| `B2 T=1024 H=8` | 170.3 | 143.9 | **0.85×** |
| `B8 T=2048 H=8` | 1263.6 | 1047.6 | **0.83×** |
| `T=250 H=3`（非整除） | 22.7 | 10.7 | **0.47×** |

`nw=4` 或 `BV=64` 都更差（溢出更多/循环更多）。**`BV=V` 是这批旋钮里唯一能压住寄存器的一项。**

### 33.4 Part C：落进真实组装链——六项梯度不变、链 dev 0.83–0.93×

`harness/check_bwd_b2b3_fused.out.txt`。把融合 kernel 替换掉 §12 链里的 B2+B3（对比基线取
**P7-28 就地累积链** `check_bwd_b3_accum.make_chain(accum=True)`，即当前最优未融合链），六项梯度
vs `torch.autograd`（FLA）最坏与 §25.2/§26.3 的**同一组数**（`dq 3.417e-3/dk 3.016e-3/dv 6.795e-3/
dbeta 6.012e-3/dg 7.054e-3/dh0 8.267e-3`，含非整除 `T=250/513`、`g=0`、`B=2`）；融合链 vs 基线的
**自身增量** `dq/dv` 逐位 0、`dk/dg ≤1.1e-7`、`dbeta ≤2.4e-7`。

链级计时（`NS=3` pipe B1、CUPTI `dev` + event `wall`，空闲卡 1）：

| case | FLA dev | base dev | fused dev | fus/base | base wall | fused wall | wall× |
|---|---|---|---|---|---|---|---|
| `T=512 H=4` | 203.5 | 74.2 | **67.8** | **0.91×** | 293.7 | 270.4 | 0.92× |
| `B2 T=1024 H=8` | 661.5 | 261.7 | **242.9** | **0.93×** | 299.7 | 270.0 | 0.90× |
| `B8 T=2048 H=8` | 4142.1 | 1681.7 | **1541.2** | **0.92×** | 1698.0 | 1553.4 | 0.91× |
| `T=250 H=3`（非整除） | 140.7 | 51.7 | **43.1** | **0.83×** | 295.5 | 265.6 | 0.90× |

按 kernel 归因（`B8T2048H8`）：基线 `wy_dqkg` 902 µs + `intra` 295 µs = 1197 µs；融合 kernel
**1053 µs（0.88×）**——与孤立 0.83× 一致。链 `dev/FLA` 落到 **0.31–0.37×**。

### 33.5 Part D：`ncu` 佐证——DRAM 字节降 23%、L2 降 17%

`harness/probe_b2b3_fused_ncu.sh`（`probe_b2b3_fused_ncu.out.txt`，fp32 操作数 = 部署 regime，
`B8T2048H8 BT=16`）：

| 指标 | B2 `wy_dqkg` | B3 `intra`(MMA) | **fused** | B2+B3 | fused/(B2+B3) |
|---|---|---|---|---|---|
| DRAM 读 | 1.64 GB | 420.5 MB | **1.58 GB** | 2.06 GB | 0.77× |
| DRAM 写 | 309.5 MB | 186.6 MB | **382.9 MB** | 497 MB | 0.77× |
| DRAM 合计 | 1.95 GB | 607 MB | **1.96 GB** | 2.56 GB | **0.77×** |
| L2 字节 | 3.21 GB | 1.19 GB | **3.66 GB** | 4.40 GB | 0.83× |
| tensor | 5.74% | 2.28% | **6.20%** | — | — |
| `long_sb` | 8.97 | 19.71 | **5.76** | — | — |
| Duration | 898.6 µs | 294.7 µs | **1050 µs** | 1193 µs | **0.88×** |
| 占用率 | 12.34% | 42.67% | **12.48%** | — | — |
| regs | — | — | 255 | — | — |

融合把 `dq/dk/dg` 的「写+读+写」压成「写一次」，省掉 B3 对它们的整轮读写 + B2 的 `dA` 落盘：
**DRAM 字节 −23%、L2 −17%**，与「省下 ~0.6 GB / 2.56 GB」的闭式预测吻合。`ncu` 的 `Duration` 1050 µs
与 CUPTI 归因的 1053 µs 一致（不同于 §25.5 某些窗口不同源的偏差）。融合 kernel 仍受
**255 寄存器 + 165.9 KB smem 钉在 1 block/SM（theoretical occ 12.5%）**，与 B2 同；故它不是占用率
翻身，而是**纯粹减字节**。

### 33.6 结论与下一步

- **B2/B3 融合成立**（第三次尝试，前两次 P7-27 的 B1→B2 朴素融合是反例）：在部署粒度
  `NC=1`（B3 只剩对角）下，把 B3 折进 B2、`dq/dk/dg` 累加器留寄存器，**代数逐行不变**
  （自身增量 ≤2.4e-7、vs FLA 逐值同 §25/§26），**链 dev 0.83–0.93×、wall 0.90–0.92×**，
  DRAM 字节 **−23%**、L2 **−17%**。
- **唯一的使能条件＝`BK=K` + `BV=V`**：融合要求 `dq/dk/dg` 以 `[BT,K]` 常驻，只能 `BK=K`；
  而 `BK=K` 必然溢出，必须靠 **`BV=V`（去掉 `i_v` 循环）+ `nw=8`** 把溢出从 222/60 压到
  **42** 才使融合净胜。这两项都不是「可加可不加」的旋钮，而是融合的**结构前提**（同 §30.5 的教训：
  朴素融合会被寄存器吃掉，必须换结构）。
- **契约变更**：融合后 `dA`（=`dAkk`）不再落盘（只被 intra 消费）；`dq/dk/dg` 只由融合 kernel 写一次。
  若后续把 B1 也并进来（§30 的 K2 形状融合），这三个字段与 `dh/dv2` 的交接要一并重审。
- **定位**：这是「Triton 侧三元组（B1 已流水、B2/B3 已融合）工程化」的最后一块——B1 由 §25 的
  `tl.range` 流水解决、B2/B3 由本轮融合解决；再往下是**把整条链做成 K2 形状的 CUDA**（目标卡工作），
  或 §30.5 的「K2 形状内 B1→B2 融合」（待验证，须在目标卡以 K2 为基座测）。
- **产物**：`harness/check_bwd_b2b3_fused.py`（`check_bwd_b2b3_fused.out.txt`，Part A 正确性 +
  B 孤立计时 + D 寄存器读 + C 落链正确性/计时）+ `harness/ncu_b2b3_fused_driver.py` /
  `harness/probe_b2b3_fused_ncu.sh`（`probe_b2b3_fused_ncu.out.txt`）。

## 34. 量化 §30.5「K2 形状内 B1→B2 融合」的收益上界：`dh` 落盘往返到底值多少（P7-31）

§30.5 把「K2 形状内 B1→B2 融合」（寄存器常驻状态 + TMA + 同 CTA 内交接）列为**唯一剩下的真结构项**，
却留在**待验证**——因为 P7-27 已证明朴素 `BV=V` 单 kernel 会被寄存器压垮。但这留下一个更前置的问题：
**这个融合的收益上界是多少？** K2 形状把 $dS$ 常驻寄存器，相对「两个 kernel 的链」它**唯一能省掉的流量**
就是每块的 `dh` 矩阵：

```
B1 写出 dh[c]  ->  DRAM   （每块 [K,V] 的 bf16）
B2 读入 dh[c]  <-  DRAM
```

本节不构建（很大的）融合 kernel，而是用**受控消融**把这两半分别量清：

- **Part A（B1 侧）**：K2 形状 TMA B1（`b1_tma_probe.cu`，取 §24 最优的 `-DB1T_ETMA`）编译两次：
  基线 vs `-DB1T_NODH`（**删掉 `dh_g` 的全局 store**；op1 需要的 per-warp bf16 `sdh` 镜像**保留**）。
  先证 `dv2/dh0` **逐位不变**，再用 CUPTI 计时——差值即 `dh` store 的成本。
- **Part B（B2 侧）**：Triton B2（`wy_dqkg`，`build_kernel(nodh=True)`）编译两次：基线 vs
  **不读 `dh`** 的变体（复用已加载的 `b_h` 顶替；值错，仅计时）。差值即 `dh` load 的成本。
- **Part C**：把两半合成往返上界，与 §26 的链级拆解对照，裁定这个融合值不值得做。

产物：`harness/check_bwd_dh_roundtrip.py`（`harness/check_bwd_dh_roundtrip.out.txt`）+ 编译开关
`B1T_NODH`（`b1_tma_probe.cu`）+ `build_kernel(nodh=)`（`check_bwd_dqkg_kernel.py`）+
`harness/ncu_dh_roundtrip_driver.py` / `harness/probe_dh_roundtrip_ncu.sh`
（`harness/probe_dh_roundtrip_ncu.out.txt`）。

### 34.1 Part A：`dh` store 是 B1 的主要成本（占 33–63%），且不是带宽

`harness/check_bwd_dh_roundtrip.out.txt`（`ETMA` 变体，CUPTI `us/call`，空闲卡）。`dv2/dh0` 在**所有 16 组**
配置下**逐位相同（`max_abs=0`）**，证明消融只动了 `dh` store：

| shape | BT | NS | base µs | nodh µs | nodh/base | dh-store 占 |
|---|---|---|---|---|---|---|
| B1 T512 H4 | 64 | 1 | 45.1 | 28.7 | 0.636 | 36.4% |
| B2 T1024 H8 | 64 | 1 | 91.2 | 58.6 | 0.642 | 35.8% |
| B2 T1024 H8 | 32 | 2 | 133.8 | 49.0 | **0.366** | **63.4%** |
| B8 T2048 H8 | 32 | 2 | 351.6 | 98.7 | **0.281** | **71.9%** |
| B8 T2048 H8 | 64 | 1 | 195.8 | 130.7 | 0.667 | 33.3% |
| B1 T250 H3 | 64 | 1 | 25.5 | 16.8 | 0.660 | 34.0% |

`ncu`（`probe_dh_roundtrip_ncu.out.txt`）给出**为什么**（B8 T2048H8 BT=64 NS=1）：

| 指标 | base | nodh | 说明 |
|---|---|---|---|
| Duration | 198.3 µs | **132.8** | −33% |
| DRAM **read** | 170.5 MB | 168.9 | **不变**（不是读） |
| DRAM **write** | 93.3 MB | **30.9** | 只降 62 MB（≈ `dh` 的 134 MB 被 L2 吸收一半） |
| **L2 bytes** | 866.4 MB | **417.7** | **−52%**（一次 134 MB 的 store 在 L2 放大出 ~448 MB 流量） |
| tensor % | 10.80 | **16.29** | mma 密度抬升 |
| `smsp...stalled_lg_throttle` | **0.67** | **0.06** | **store 队列反压（真凶）** |
| `stalled_long_scoreboard` | 3.84 | 2.56 | store 反压也拖慢后续 load |

即：**`dh` store 的成本不是 DRAM 写带宽**（read 不变、write 仅 −62 MB），而是
**每线程逐标量的 `st.global` 打满了 LSU/store 队列 → `lg_throttle` 0.67**，把整个 compute warp 的
发射卡住，同时制造 ~3×（写字节）的 L2 流量。删掉后 `lg_throttle` 归零、`tensor%` 10.8→16.3。
BT=32 NS=2 的极端档同理（`lg_throttle` 0.88→0.03、L2 161→57 MB、Duration 136.7→**50.5 µs**）。

### 34.2 Part B：`dh` load 在 B2 只值 ~5%（DRAM 底噪）

`wy_dqkg`（Triton，`build_kernel(nodh=True)`；`dq` 的 `max_abs` 非零证明确实动了 `dh`）：

| shape | K=V | base µs | nodh µs | nodh/base |
|---|---|---|---|---|
| B2 T512 H4 | 128 | 37.1 | 34.3 | 0.924 |
| B2 T2048 H8 | 128 | 194.7 | 184.8 | 0.950 |
| B1 T512 H4 | 128 | 29.5 | 30.0 | 1.014（噪声） |
| B1 T250 H3 | 64 | 14.9 | 14.6 | 0.984 |

`ncu`（B8 T2048H8 BT=64）：read 763.8→**672.2 MB**（只降 92 MB，`dh` 的 134 MB ≈ 68% 进 DRAM）、
Duration 754.9→**716.7 µs（−5.0%）**、`tensor%` 3.22→3.40、`long_sb` 16.57→15.38。**B2 是 DRAM bound 的
大 kernel（`long_sb` 16.6），多读/少读一个 134 MB 张量只是一小块**，与 §26（`h`+`dh` bf16 只值 B2 内核 ~10%）
和 §29 完全一致。

### 34.3 Part C：往返上界 ≈ B1+B2 的 10%，且大头是「store 效率」不是「融合」

把两半相加（B8 T2048H8 BT=64 NS=1）：**B1 store 65.2 µs（33.3%） + B2 load 38.2 µs（5.0%）≈ 103 µs**；
对照链级 B1（~196 µs）+ B2（~755–900 µs，§26/§28）≈ **950–1100 µs**，即往返上界 **≈ 10%**。据此：

1. **融合的收益上界只有 ~10%**，其中 **~2/3 在 B1 侧**、且 B1 侧那一大块已被 §34.1 定位成
   **`lg_throttle`（store 发射/ L2 放大）**，不是一个「融合才能解决」的问题——**只要把 `dh` 的落盘
   换成 TMA / 专职 STORE warp（镜像前向 K2 的 STORE warp，`fwd_kernel2.cuh`），就能在不做任何融合的
   前提下吃掉 B1 侧的 33%**。
2. **融合相对「高效 `dh` store」的净增量只剩 B2 侧的 `dh` load ≈ 5%**——与 §26/§29 对 `h/dh` 的
   判定同阶。故 **不值得为它建一整个 K2 形状的融合 kernel**。
3. **更新的设计输入（修正 §30.5）**：目标卡 CUDA 反向的**下一优先级 = 把 B1 的 `dh`（和 `dv2`）落盘
   改成 TMA/专职 store 路径**（量级 ~30% 的 B1），**而不是 B1→B2 融合**；融合的边际收益（~5%）
   排在很后面。

### 34.4 边界 / 待验证

- 本轮量的是**本原型 `b1_tma_probe.cu` 的朴素逐标量 store**的代价；`lg_throttle` 的幅度与具体 store
  指令（标量 vs 向量化 vs TMA）强相关——**待验证**：把 `dh`/`dv2` 改成 TMA store / 专职 STORE warp 后，
  能否复现这 33%（预期能吃掉大头，因为 TMA 有独立引擎、不求 LSU 发射）。
- B1 侧消融是**删掉整个 store**，故 65 µs 是「store 总成本」的上界；TMA store 仍要付少量 DRAM 带宽
  （~134 MB 的一半进 DRAM ≈ 30–45 µs），故实际收益应小于 65 µs、但显著大于 0——**待验证**用 TMA store
  变体直接测。
- Part B 的 nodh 是**数值错的**（顶替 `b_h`），只用于计时；其给出的 5% 与 ncu 的 read 字节下降自洽，
  结论不依赖其数值。


## 35. 兑现 §34.4：把 B1 的 `dh` 落盘改成 TMA store——`lg_throttle` 归零、吃掉那 33% 的大头（P7-32）

§34 用受控消融把「K2 形状内 B1→B2 融合」的收益上界量成 **≈ B1+B2 的 10%**，并指出其中 **~2/3 在 B1 侧、
且是逐标量 `st.global` 的 `lg_throttle`（store 队列反压）+ ~3× L2 写放大，不是 DRAM 带宽**。§34.4 于是把
「把 `dh`/`dv2` 落盘换成 **TMA store / 专职 STORE warp**（镜像前向 K2 的 STORE warp）能否复现这 33%」
列为**待验证**。本节兑现它：给 `b1_tma_probe.cu` 加编译开关 **`-DB1T_TMADH`**，把每块的 `dh` 改由 TMA 引擎落盘，
用一个受控消融（scalar / 无 store / TMA store 三者同源编译）给出明确答案。

### 35.1 实现：per-warp `[KD,BW]` tile 的 `cp.async.bulk.tensor` store

改动很小（`harness/b1_tma_probe.cu`）：

- **常量**：`KD=128`、`BV=64`、`NW=4`、`BW=BV/NW=16`。每个 compute warp 本就独占 V 方向 16 列
  （`vbase = v0 + warp*BW`），所以 `dh` 的天然分片就是 **per-warp `[KD, BW] = [128, 16]`**。
- **smem**：给 `Shared` 加 `bf16 dho[2][NW][KD][BW]`——**未 padding** 的 `[KD,BW]` tile，**双缓冲**，
  用于被 TMA 读取（TMA 要求行 stride 精确，故不能复用 padding 过的 `sdh` 镜像）。共 +32768 B（见 §35.4）。
- **发布 S 时**：把 warp 的 4 个 bf16 值同时写 `sdh`（给 op1 的 `ldmatrix`，保留）**和** `dho[i&1][warp]`；
  `i>=2` 时 lane 0 先 `cp.async.bulk.wait_group.read 1`（保证双缓冲里上上块的那份已被 TMA 读完）再写。
- **落盘**：`__syncwarp()` 后 lane 0 发一条
  `cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [dhmap, {vbase,0,h,c,n}], [smem]` 再
  `cp.async.bulk.commit_group`。tensor map（`make_map_dh`）把 `dh`（`[B,NT,H,KD,VD]` bf16）暴露成
  dims `{VD,KD,H,NT,B}`、box `{BW,KD,1,1,1}`、`SWIZZLE_NONE`（内维只有 `BW*2=32 B`）。
- **为什么不用跨 warp 的 STORE warp / barrier**：每 warp 的切片沿 V 方向连续，单 warp 就能发一条合法
  TMA store，于是**稳态循环里仍然没有任何 `__syncthreads()`**（与 §23 的 producer/consumer 结构一致）。
  K2 的「专职 STORE warp」在这里退化成了「每 compute warp 自己发一条 store」——因为被删掉的正是逐线程
  标量发射，不是某个 warp 的职责。

### 35.2 正确性：逐位不变（不是「同阶」）

`harness/check_bwd_b1_tma_store.py`（`harness/check_bwd_b1_tma_store.out.txt`）Part A，16 组配置
（`BT=16/32/64` 中取 32/64、`NS=1/2`、含非整除 `T=250`）：

| 对比 | dh | dv2 | dh0 |
|---|---|---|---|
| TMA vs scalar（全部 16 组） | **0** | **0** | **0** |
| TMA vs FLA `dhu`（sanity） | 4.88e-4（bf16 1 ulp） | — | — |

写进 `dho` 的就是原来的 `f2b(S[...])`，TMA 只是换了个搬运引擎，故**逐位相同**——这点比 §22 的 `scale` 折
bf16 更干净（那里丢了逐位）。非整除 `T=250` 也不受影响：`dh` 的 `[KD,VD]` 块本身不跨 `T`（`T` 只影响 `dv2`），
且 `V%BV==0` 由 host 保证。

### 35.3 计时：CUPTI 恢复 58–84%（均值 ~72%）的那 33%

Part B（CUPTI `us/call`，空闲卡）：

| shape | BT | NS | base µs | nodh µs | tma µs | nodh/base | tma/base | 恢复占比 |
|---|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 1 | 68.0 | 34.4 | 43.6 | 0.506 | 0.642 | 0.72 |
| B1 T512 H4 | 32 | 2 | 67.1 | 26.3 | 38.9 | 0.392 | 0.579 | 0.69 |
| B1 T512 H4 | 64 | 1 | 44.8 | 28.7 | 33.9 | 0.642 | 0.756 | 0.68 |
| B1 T512 H4 | 64 | 2 | 43.2 | 24.8 | 29.6 | 0.574 | 0.685 | 0.74 |
| B2 T1024 H8 | 32 | 1 | 141.6 | 69.3 | 99.5 | 0.490 | 0.703 | 0.58 |
| B2 T1024 H8 | 32 | 2 | 133.7 | 49.1 | 72.4 | 0.367 | 0.541 | 0.72 |
| B2 T1024 H8 | 64 | 1 | 91.2 | 56.1 | 70.8 | 0.616 | 0.776 | 0.58 |
| B2 T1024 H8 | 64 | 2 | 84.6 | 46.0 | 56.3 | 0.543 | 0.666 | 0.73 |
| B8 T2048 H8 | 32 | 1 | 296.4 | 163.2 | 215.1 | 0.551 | 0.726 | 0.61 |
| B8 T2048 H8 | 32 | 2 | 357.5 | 99.1 | 144.2 | 0.277 | 0.403 | 0.82 |
| B8 T2048 H8 | 64 | 1 | 194.5 | 129.3 | 152.1 | 0.664 | 0.782 | 0.65 |
| B8 T2048 H8 | 64 | 2 | 209.7 | 87.8 | 107.9 | 0.419 | 0.515 | 0.84 |
| B1 T250 H3 | 32 | 1 | 35.4 | 19.0 | 23.8 | 0.536 | 0.672 | 0.71 |
| B1 T250 H3 | 32 | 2 | 37.1 | 16.6 | 21.9 | 0.447 | 0.591 | 0.74 |
| B1 T250 H3 | 64 | 1 | 25.1 | 17.3 | 19.0 | 0.688 | 0.758 | 0.77 |
| B1 T250 H3 | 64 | 2 | 24.5 | 16.0 | 17.4 | 0.655 | 0.709 | 0.84 |

「恢复占比」= `(base−tma)/(base−nodh)`：**0.58–0.84（均值 ~0.72）**。即 TMA store **在没有做任何 B1→B2 融合
的前提下吃掉了 §34.1 那 33% 的大头**（tma/base 0.40–0.78，base→tma 提速 **1.28–2.48×**）。残留的那部分
（tma 仍比 nodh 慢）正是 §34.4 预判的「TMA 仍要付 `dh` 的真实 DRAM 写带宽」——下一节 ncu 坐实。

### 35.4 ncu：`lg_throttle` 归零、写放大减半，但 DRAM 写**照旧**

`harness/probe_b1_tma_store_ncu.sh`（`harness/probe_b1_tma_store_ncu.out.txt`）。两组：

**`B8 T2048H8 BT=64 NS=1`**（§34.1 的同一配置）：

| 指标 | scalar (base) | nodh | **TMA** | 读法 |
|---|---|---|---|---|
| Duration µs | 199.4 | 135.9 | **154.6** | −22% vs base |
| DRAM read MB | 171.0 | 168.9 | 168.9 | 不变 |
| DRAM write MB | 92.1 | 30.6 | **91.4** | **基本不变**（`dh` 仍进 DRAM） |
| L2 bytes MB | 860.3 | 414.8 | **486.6** | **−43%** |
| tensor % | 10.73 | 16.33 | 13.98 | 逼近 nodh |
| **`lg_throttle`** | **0.66** | 0.06 | **0.04** | **归零** |
| `long_scoreboard` | 3.88 | 2.47 | 2.53 | 跟 nodh 同档 |
| `short_scoreboard` | 0.90 | 0.80 | 0.80 | — |
| `wait` | 1.24 | 1.17 | 1.36 | 略升 |
| theo occ % | 15.62 | 15.62 | **7.81** | smem +32 KB 掉半 |
| achieved occ % | 7.78 | 7.85 | 7.77 | **不动** |

**`B2 T1024H8 BT=32 NS=2`**（§34.1 的最极端档）：

| 指标 | scalar (base) | nodh | **TMA** |
|---|---|---|---|
| Duration µs | 135.3 | 50.7 | **74.7** |
| DRAM read MB | 21.26 | 21.26 | 21.27 |
| DRAM write MB | 7.12 | 0.95 | **7.17** |
| L2 bytes MB | 161.1 | 57.3 | **86.3** |
| tensor % | 7.97 | 21.29 | 14.84 |
| `lg_throttle` | **0.89** | 0.03 | **0.05** |
| `long_scoreboard` | 2.86 | 0.65 | 0.83 |

三点结论：

1. **机制被钉死**：`lg_throttle` 从 0.66/0.89 掉到 **0.04/0.05**——正好落在 nodh 的 0.05/0.04 上。这**直接
   证明** §34.1 归因的「逐标量 `st.global` 打满 store 队列」就是真凶；TMA 有独立引擎、不占 LSU 发射，
   于是反压消失。守恒：`long_scoreboard` 也随之从 3.88/2.82 降到 2.52/0.87（store 反压曾连带拖慢 load）。
2. **不是省带宽**：DRAM **write 几乎不动**（91.4 vs base 92.1；7.17 vs 7.12）——`dh` 该进 DRAM 还是进，
   只有 nodh（不写）才是 30.6/0.95。**L2 bytes 掉 43%**（860.3→486.6）才是省的那块：TMA 写的是精确的
   128 B 行，消掉了标量 2 B store 在 L2 上的 ~3× 放大（§34.1）。故 TMA 相对 nodh 的残留时间
   （154.6−135.9=18.7 µs）≈ 真实写 `dh` 的 ~61 MB ÷ ~3 TB/s，与 §34.4 的估计吻合。
3. **占用率不是旋钮**：TMA 的 **theoretical occupancy 腰斩**（15.62→7.81%，因 `dho` 双缓冲 +32 KB smem），
   但 **achieved occupancy 纹丝不动（7.77%）且更快**——再次排除占用率（与 §22/§24/§17 一致）。

`dho` 双缓冲的 smem 代价（`b1t_smem_bytes`）：`BT=64 NS=1` 104448→**137216 B**、`BT=32 NS=2`
99328→**132096 B**、`BT=16 NS=1` 46080→**78848 B**，各 **+32768 B**。若日后 smem 吃紧，可把双缓冲退回
单缓冲（`wait_group.read 0`），代价是 store 与下一块 compute 不再重叠。

### 35.5 结论 / 对目标卡的设计输入

- **§34.4 的待验证有了答案：能。** 把 B1 的 `dh` 落盘从逐标量 `st.global` 换成 **TMA store**，
  逐位不变、**恢复 §34.1「删掉 store」收益的 58–84%（均值 ~72%）**，且不需要任何 B1→B2 融合。
- **更新设计输入（合并 §30.5 / §34.3）**：目标卡 CUDA 反向的**第一优先级 = 把 B1 的 `dh`（以及同一
  逐标量循环里的 `dv2`）改为 TMA store**（或专职 STORE warp）；这一步是**独立、低风险、逐位无损**的，
  吃掉 B1 侧那 33% 的大头。**B1→B2 融合**的净增量只剩 B2 侧的 `dh` load ≈ 5%（§34.2），排在最后。
- **对 `dv2` 的推广（待验证）**：本节只改了 `dh`（与 §34.1 的消融口径一致）。`dv2`（`[BT,BW]` per warp，
  逐 token 逐标量写，还有 `tr<T` 边界）**结构上同样可 TMA store**（box `{BW,1,1,BT,1}`，`SWIZZLE_NONE`，
  OOB 由 TMA 自动不写，天然处理非整除 `T`）——预期再省一小块，但未在本轮实测，标「待验证」。
- **未做**：`dh` 走 TMA 后 B1 的下一块瓶颈是每 warp 发射效率（tensor 13.9% vs nodh 16.4%，`wait` 1.36），
  即 §24.5 的「手写 mbarrier 结构上限」；继续压需换 CUTLASS `PipelineTmaAsync`，与本节的 store 优化正交。


## 36. 把 TMA store 从 `dh` 推广到同一逐标量循环里的 `dv2`——`lg_throttle` 归零、L2 再降一半（P7-33）

§35.5 留了最后一个**待验证**：`dh` 改 TMA（§35）只覆盖了两条逐标量 store 中的一条，另一条 **`dv2`**
（每个 warp 的 `[BT,BW]` 输出分片，且带 `tr<T` 边界判断）**结构上与 `dh` 完全同型**，预期「再省一小块」。
本节兑现它：给 `b1_tma_probe.cu` 加编译开关 **`-DB1T_TMADV`**，把 `dv2` 也交给 TMA 引擎落盘，并加一个
**`-DB1T_NODV2`** 消融（删掉整条 `dv2` 全局写、只留计时），用同源多版本编译把「`dv2` 这条 store 值多少 /
TMA 能回收多少」量清楚。

### 36.1 实现：per-warp `[BT,BW]` tile 的 4D TMA store，与 `dh` 共用一个 bulk group

改动与 §35.1 对称，落在 `harness/b1_tma_probe.cu`：

- **常量/分片**：每个 compute warp 处理 V 方向连续的 `BW=16` 列（`vbase = v0 + warp*BW`），每块覆盖
  `BT` 个 token，所以 `dv2` 的天然分片就是 **per-warp `[BT, BW]`**。
- **smem**：`Shared` 加 `bf16 duo[2][NW][BT][BW]`——**未 padding** 的 `[BT,BW]` tile、**双缓冲**（TMA 要求
  精确行 stride，不能复用 padding 过的 `sdu` 镜像）。
- **tensor map**（`make_map_dv2`）：`dv2` 是 token-major `[B,T,H,VD]`，暴露成 dims `{VD,T,H,B}`、
  box `{BW,BT,1,1}`、`SWIZZLE_NONE`（内维 `BW*2=32 B`）。**这里踩了一个 stride 坑**（见 §36.4）。
- **落盘**：前缀和写 `sdu`（给 op3 的 `ldmatrix`，保留）**和** `duo[i&1][warp]`；`i>=2` 时 lane 0 先
  `cp.async.bulk.wait_group.read 1` 再写双缓冲。`__syncwarp()` 后 lane 0 发一条
  `cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [dv2map, {vbase,c*BT,h,n}], [smem]`。
- **一个 bulk group / 块**：`dh` store 不再在发布 S 后单独 commit，而是**推迟**到 `dv2` 也发完后**一起**
  `cp.async.bulk.commit_group`（`#if defined(B1T_TMADH) && !defined(B1T_TMADV)` 才在 `dh` 处 commit）。
  这样仍是**每块一个 group**，双缓冲的 `wait_group.read 1` 语义与 §35 一字不变。
- **OOB 由 TMA 处理**：标量路径要 `if (tr < T)` 逐行判断，TMA 只按 tensor 边界（dim1 = `T`）裁掉越界 token
  行——**非整除 `T` 无需显式 mask**（这正是 §35.5 预判的那点便利）。

### 36.2 正确性：与标量路径逐位相同（含非整除 `T`、多 head）

`harness/check_bwd_b1_tma_dv2.py`（`harness/check_bwd_b1_tma_dv2.out.txt`）Part A，16 组配置
（`BT=32/64`、`NS=1/2`、含非整除 `T=250`）：

| 对比 | dh | dv2 | dh0 |
|---|---|---|---|
| TMA（dh+dv2）vs scalar（全部 16 组） | **0** | **0** | **0** |
| TMA vs FLA `dhu`（sanity） | 4.88e-4（bf16 1 ulp） | 4.88e-4（T250 BT64 处 2.44e-4，两边都裁尾行） | — |

写进 `duo` 的就是原来的 `f2b(d…)`，TMA 只换搬运引擎，故**逐位相同**——与 §35.2 的 `dh` 同级干净。

### 36.3 计时：`dv2` TMA 在 `dh` TMA 之上再值 ~11%

Part B（CUPTI `us/call`，空闲卡）。六列 = base（标量 dh+dv2）/ tma（仅 dh TMA，§35）/ **tma2（dh+dv2 都 TMA，
本轮）**/ nodv2（删 `dv2` 写、dh 仍标量）/ nodh（删 `dh` 写、dv2 仍标量）/ nodhv2（两条都删）：

| shape | BT | NS | base | tma | **tma2** | nodv2 | nodh | nodhv2 | tma2/base | tma2/tma |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 1 | 65.7 | 43.6 | **39.4** | 59.1 | 34.2 | 28.6 | 0.599 | 0.904 |
| B1 T512 H4 | 32 | 2 | 66.4 | 39.0 | **34.2** | 58.6 | 26.5 | 23.9 | 0.515 | 0.877 |
| B1 T512 H4 | 64 | 1 | 44.6 | 33.9 | **30.0** | 39.7 | 28.8 | 25.4 | 0.672 | 0.885 |
| B1 T512 H4 | 64 | 2 | 42.9 | 29.6 | **25.4** | 39.7 | 25.0 | 21.1 | 0.593 | 0.858 |
| B2 T1024 H8 | 32 | 1 | 142.4 | 99.3 | **89.3** | 128.2 | 68.3 | 56.4 | 0.627 | 0.899 |
| B2 T1024 H8 | 32 | 2 | 134.1 | 72.6 | **63.5** | 112.1 | 49.2 | 44.3 | 0.474 | 0.875 |
| B2 T1024 H8 | 64 | 1 | 91.4 | 70.9 | **62.6** | 80.5 | 55.7 | 48.3 | 0.685 | 0.883 |
| B2 T1024 H8 | 64 | 2 | 84.7 | 56.4 | **47.6** | 76.1 | 46.0 | 39.0 | 0.562 | 0.844 |
| B8 T2048 H8 | 32 | 1 | 294.0 | 215.2 | **198.6** | 272.8 | 162.8 | 141.1 | 0.675 | 0.923 |
| B8 T2048 H8 | 32 | 2 | 356.1 | 144.4 | **137.5** | 298.4 | 99.1 | 85.7 | 0.386 | 0.952 |
| B8 T2048 H8 | 64 | 1 | 194.2 | 151.8 | **137.6** | 175.9 | 129.6 | 112.5 | 0.709 | 0.907 |
| B8 T2048 H8 | 64 | 2 | 205.3 | 109.5 | **98.9** | 164.9 | 88.3 | 74.4 | 0.482 | 0.903 |
| B1 T250 H3 | 32 | 1 | 35.4 | 23.8 | **20.9** | 31.8 | 19.5 | 16.2 | 0.593 | 0.878 |
| B1 T250 H3 | 32 | 2 | 37.0 | 22.0 | **19.0** | 33.1 | 16.7 | 14.0 | 0.515 | 0.864 |
| B1 T250 H3 | 64 | 1 | 25.2 | 19.0 | **16.6** | 21.7 | 17.1 | 14.3 | 0.660 | 0.874 |
| B1 T250 H3 | 64 | 2 | 24.5 | 17.4 | **14.9** | 21.9 | 16.0 | 13.0 | 0.608 | 0.856 |

三点：

1. **`dv2` TMA 在 `dh` TMA 之上再快 5–16%**：`tma2/tma` = **0.84–0.95（均值 0.89）**，边际收益
   `tma−tma2` = **2.4–16.6 µs（均值 ~7.0）**，且 **16/16 组都为正**（无退化）。
2. **两条合起来吃掉 B1 逐标量 store 的大头**：`tma2/base` = **0.39–0.71（均值 0.58）**；
   相对「两条 store 全删」的上界 `nodhv2`，恢复占比 `(base−tma2)/(base−nodhv2)` = **0.62–0.83（均值 0.75）**，
   即两次 TMA 化后只剩约 1/4 的「真实 DRAM 写带宽」代价（与 §35.3 的 `dh` 单独口径同构）。
3. **`dv2` 的绝对值确实小**（与 §35.5 的「再省一小块」一致）：`nodv2` 相对 base 只省
   `base−nodv2` = **7–16%（小 shape ~10%、大 shape ~16%）**，而 `dh` 那条是 33–72%（§34.1）——所以本轮是
   把「小块」也收干净，**不是**新的量级项。

### 36.4 ncu：`lg_throttle` 归零、L2 再降 15%、DRAM 写照旧；一个 tensor-map stride 坑

`harness/probe_b1_tma_dv2_ncu.sh`（`harness/probe_b1_tma_dv2_ncu.out.txt`）。两组：

**`B8 T2048H8 BT=64 NS=1`**：

| 指标 | base | tma | **tma2** | nodv2 | 读法 |
|---|---|---|---|---|---|
| Duration µs | 198.5 | 155.1 | **141.4** | 180.0 | −29% vs base |
| DRAM read MB | 169.7 | 168.9 | 168.9 | 169.4 | 不变 |
| DRAM write MB | 91.6 | 91.7 | **91.2** | 61.0 | **不变**（`dv2` 仍进 DRAM；nodv2 才是不写） |
| L2 bytes MB | 863.4 | 490.7 | **416.2** | 643.0 | **−52%**（tma2 再吃掉 tma 的 −15%） |
| tensor % | 10.85 | 13.98 | **15.48** | 12.01 | 逼近 nodh 档 |
| **`lg_throttle`** | **0.66** | 0.04 | **0.04** | **0.83** | 归零（`dh`+`dv2` 都 TMA） |
| `long_scoreboard` | 3.86 | 2.51 | 2.81 | 4.03 | 跟 nodh 同档 |
| `short_scoreboard` | 0.91 | 0.80 | 0.75 | 0.79 | — |
| `wait` | 1.24 | 1.36 | 1.40 | 1.38 | 略升 |
| `SM%` | 14.5 | 19.2 | 17.9 | 13.7 | — |

**`B2 T1024H8 BT=32 NS=2`**（§34.1 的最极端档）：

| 指标 | base | tma | **tma2** | nodv2 |
|---|---|---|---|---|
| Duration µs | 134.3 | 74.8 | **65.1** | 114.8 |
| DRAM read MB | 21.30 | 21.27 | 21.26 | 21.27 |
| DRAM write MB | 7.49 | 7.05 | **7.14** | 4.73 |
| L2 bytes MB | 161.0 | 85.8 | **75.5** | 139.2 |
| tensor % | 7.98 | 14.83 | **16.89** | 9.33 |
| **`lg_throttle`** | **0.90** | 0.06 | **0.07** | **0.92** |
| `long_scoreboard` | 2.87 | 0.82 | 0.77 | 2.78 |
| `short_scoreboard` | 2.24 | 0.83 | 0.68 | 1.82 |

三点：

1. **机制与 §35 完全同构**：`dv2` 标量写单独看（nodv2 列）也把 `lg_throttle` 顶到 0.83/0.92；`dh`+`dv2`
   都走 TMA 后 **`lg_throttle` = 0.04/0.07**——两条 store 的 LSU 反压都被独立引擎消掉。`long_scoreboard`
   随之从 base 的 3.86/2.87 落到 nodh 档（store 反压曾连带拖慢 load）。
2. **不是省带宽、是消写放大**：DRAM **write 几乎不动**（91.6→91.2；7.49→7.14），只有 nodv2（不写）才降到
   61.0/4.73。省下的是 **L2 字节再降 15%/12%**（tma 490.7→tma2 416.2；85.8→75.5）——即 `dv2` 的逐标量
   2 B store 在 L2 上的放大也被 TMA 的整块写消掉。
3. **tensor 密度继续抬升**：同样 mma 工作量、时长更短 → tensor% 10.85→**15.48**（大 shape）/ 7.98→**16.89**
   （小 shape），把 §35.4「残差在每 warp 发射效率（`wait`/`short_sb`）」进一步暴露为下一步唯一瓶颈。

**tensor-map stride 坑（本轮踩到）**：`make_map_dv2` 的 `globalStrides` 按 **dim 从 1 起算**（`gstr[i]` =
第 `i+1` 维的字节 stride）。`dv2` 的 `[B,T,H,VD]` 里 dims 顺序是 `{VD,T,H,B}`，故正确值
`gstr = {H*VD*2, VD*2, T*H*VD*2}`（分别对应 T/H/B 维）。首版误按 `{VD,T,H}` 写成
`{VD*2, T*VD*2, H*T*VD*2}`，**恰好把 T 与 H 两维的 stride 写反**——`H==1` 时两者相等，**单 head 测试
全部逐位通过**（`B1 T512 H1` 连多 chunk 都对），只有 `H>1` 才暴露（`dv2` `max_abs` 0.11–0.17，`dh/dh0`
仍逐位 0）。指纹：**只有 `dv2` 错、且只在 `H>1` 错**。修法即按 `i+1` 维重排 stride。

### 36.5 结论 / 对目标卡的设计输入

- **§35.5 的待验证有了答案：能，且逐位无损。** `dv2` 的 TMA store（4D 张量图 `{VD,T,H,B}`、box
  `{BW,BT,1,1}`、`SWIZZLE_NONE`、与 `dh` 共用一个 bulk group）在 `dh` TMA 之上**再快 5–16%（均值 ~11%）**，
  `lg_throttle` 从 0.04 的 dh 档不再上升、L2 再降 15%，OOB 由 TMA 自动裁尾行。
- **最终设计输入（收敛 §30.5 / §34.3 / §35.5）**：目标卡 CUDA 反向的**第一优先级 = 把 B1 的 `dh` 与 `dv2`
  两条逐标量 `st.global` 全部改为 TMA store**（4D/5D 张量图 + 双缓冲 + 一块一个 bulk group），
  **独立、低风险、逐位无损**，两条合起来恢复「全删 store」上界的 **62–83%（均值 75%）**；
  `dh`+`dv2` 都 TMA 后 B1 的下一块瓶颈只剩**每 warp 发射效率**（`wait` 1.40、tensor 15.5%、`short_sb`），
  即 §24.5 的手写 mbarrier 结构上限——继续压需换 CUTLASS `PipelineTmaAsync`，与本节的 store 优化正交。
- **B1→B2 融合**的净增量仍只剩 B2 侧 `dh` load ≈ 5%（§34.2），排在最后。

## 37. 澄清 `docs/07` 里的两个 FLA `dhu` baseline——差的是 **operand dtype**（fp32 vs bf16），并更正 §25 的 B1／链级结论（P7-34）

本节是一次**测量学复核**，不引入新 kernel。起因是本专题在不同段落给同一条 FLA 状态伴随 kernel
`chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 报出了**相差 4–6 倍**的两个数（`B2 T1024 H8`、D=128）：

- **~23 µs**：§17（P7-14）起、一直到 §22/§23/§24/§34/§35/§36，都是**直接调用**
  `chunk_gated_delta_rule_bwd_dhu` 且喂 **bf16** 操作数；
- **~137 µs**：§10（P7-7）、§11（P7-8）、§25（P7-22）——**链内**（`torch.autograd` 穿 `chunk_kda`）
  且喂 **fp32** 操作数（`bench_bwd.py` / `check_bwd_b1_pipe_e2e.py` 的 FLA 侧都是 `.float()`）。

两者都当成了 `xFLA` 的分母，于是「我们的 bf16 B1 到底比 FLA 快还是慢」在文档里自相矛盾。本节用
`harness/check_bwd_fla_dhu_baseline.py`（`harness/check_bwd_fla_dhu_baseline.out.txt`，空闲卡
`CUDA_VISIBLE_DEVICES=1`）把两个混淆变量——**operand dtype** 与 **direct-call vs in-graph**——分开。

### 37.1 受控实验：dtype 是唯一变量

**Part A**（直接调 `chunk_gated_delta_rule_bwd_dhu`，同一套参数、只换 operand dtype，kernel-only CUPTI µs/call）：

| shape | bf16 | fp32 | fp32 / bf16 |
|---|---|---|---|
| B1 T512 H4 | 13.1 | 53.0 | **4.05** |
| B2 T1024 H8 | 23.3 | 127.6 | **5.48** |
| B8 T2048 H8 | 120.0 | 525.4 | **4.38** |

**Part B**（完整 `chunk_kda` autograd 链，复刻 `bench_bwd.py` 的 fp32 构造）——链内归因的 `dhu` =
**54.0 / 137.5 / 571.8 µs**，与 Part A 的 **fp32 直接调用（53.0 / 127.6 / 525.4）逐一对上**，而与本
shape 的 **bf16 直接调用（13.0 / 23.1 / 119.5）** 差 4.2–6.0×。即：**「链内 137」不是因为链内，
而是因为链是 fp32**；改用 bf16 重跑同一条链，`dhu` 立刻回到 ~23 µs 量级。

**Part D**（同一条 FLA 链端到端 dev，唯一差别是全 bf16 vs 全 fp32）：

| shape | FLA fp32 dev | FLA bf16 dev | bf16/fp32 | `dhu` fp32 | `dhu` bf16 |
|---|---|---|---|---|---|
| B1 T512 H4 | 203.5 | 113.1 | 0.56 | 53.9 | 13.9 |
| B2 T1024 H8 | 657.5 | 357.5 | 0.54 | 137.0 | 26.7 |
| B8 T2048 H8 | 4115.3 | 2404.6 | 0.58 | 570.8 | 126.0 |

**整条 FLA 反向链在 bf16 下只有 fp32 的 0.54–0.58×**——所以「用 fp32 FLA 当分母」把 FLA 整体抬高了
~1.8×，把 `dhu` 抬高了 ~4–5×。

### 37.2 根因：fp32 操作数的「2× 字节 + tf32 点积」

回读 `chunk_delta_h.py:359-671` 的 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`：所有 `tl.dot`
的操作数都 `.to(b_k.dtype)`（即跟随输入 dtype），而运行态 `b_dh` 恒为 fp32、在点积前 `to` 到操作数
dtype。故 fp32 输入时：**(a)** `q/k/w/do/dv` 的全局载入字节翻倍；**(b)** `tl.dot(fp32, fp32)` 落 Triton
默认的 **tf32** 路径（吞吐约为 bf16 mma 的一半）。两者相乘 ≈ 4×，与实测 4.0–5.5× 一致。（这与
§6.4 早先观察到的「`dhu`/`dAv`/`intra` 的误差底噪是 tf32 而非 fp32」同源：FLA 的 fp32 核并不走 ieee。）

### 37.3 更正 §25（P7-22）的两条结论

§25 把「**我们的 bf16 B1**」与「**FLA 的 fp32 `dhu`**」直接比，故 25.3 的 FLA 列与「反超」结论**不成立**。
用 **matched bf16** 重列（我们的 B1 取 §25.3 链内归因、FLA 取本节的 bf16 直接调用）：

| case | ours B1 bf16（NS3） | FLA `dhu` **bf16** | ours/FLA | （§25 旧写法）FLA fp32 @BT64 |
|---|---|---|---|---|
| T=512 H=4 | 34.9 | 13.9 | **2.5×** | 54.0 |
| B2 T=1024 H=8 | 61.9 | 26.7 | **2.3×** | 137.1 |
| B8 T=2048 H=8 | 318.9 | 126.0 | **2.5×** | 567.0 |

即：**部署粒度 BT=16 的 bf16 B1 比 FLA `dhu` 慢 ~2.3–2.5×，而不是快 1.46–2.2×**。这与 §22–24 在**纯
CUDA 原型**上量到的「ours 慢 FLA bf16 1.7–3.8×」量级自洽；也解释了为什么 §17 在**孤立 BT=64** 上说
ours≈FLA（§17.2：23.0 vs 23.1）——BT=64 的单 kernel 确实打平，差距来自**部署的 16 块粒度**（4× 串行
步数），正是 §13/§14「B1@64 快 2.1–2.3×」的同一件事。

整条反向链的 `dev/FLA` 同样要重算（§25.3 的分母是 fp32 FLA dev）：

| case | ours NS3 dev | FLA **bf16** dev | **更正后 dev/FLA** | §25 旧值（vs fp32 FLA） |
|---|---|---|---|---|
| T=512 H=4 | 97.5 | 113.1 | **0.86×** | 0.48× |
| B2 T=1024 H=8 | 340.4 | 357.5 | **0.95×** | 0.52× |
| B8 T=2048 H=8 | 2223.9 | 2404.6 | **0.92×** | 0.54× |

即：**在 matched bf16 下，我们复用 K1 workspace 的组装链与 FLA 反向只是大致打平（~0.9×）**，不是
§10/§25 反复写的「0.48–0.61×」。我们相对 FLA 的结构性优势（**不重算前向状态**）确实存在，但被
**B1 的 16 块粒度损失（~2.3×）+ B2/B3 的慢**吃掉了大部分。

### 37.4 影响面：哪些节用了哪个 baseline

| 段落 | FLA `dhu` 取值 | 是否 fair | 说明 |
|---|---|---|---|
| §10 P7-7 / §11 P7-8 | fp32（链内） | ✔（当时我方 B1 也是 tf32） | 那时 `reconstruct` 把 workspace `.float()`，两侧同 fp32 |
| §17 P7-14 … §24 P7-21 | bf16（直接调用） | ✔ | 我方原型也是 bf16 |
| **§25 P7-22** | **fp32（链内）** vs **bf16（我方）** | **✘** | 本节更正；P7-9 起我方 B1 已是 bf16 |
| §34/§35/§36 P7-31/32/33 | bf16（直接调用） | ✔ | CUDA store 消融，两侧 bf16 |
| §26–§33 | 六项**梯度**对比（正确性） | ✔ | 误差底噪对比，不受计时 dtype 影响 |

结论：只有 **§25 的计时结论**需要更正；其余节要么同 dtype，要么是正确性。§26「B2+B3 占 66.6%」是
**链内部分比**、不受 FLA dtype 影响，仍然成立（更正后 B1 占比只升不降，B2/B3 仍是链级瓶颈）。

### 37.5 结论 / 对后续基准的约定

- **两个 baseline 都是真的**：FLA `dhu` 在 fp32 操作数下 ~127 µs、bf16 下 ~23 µs（`B2T1024H8`）；
  差额全部来自 dtype（fp32 载入 2× + `tl.dot` 落 tf32）。
- **此后 `docs/07` 的 `xFLA` 一律以 matched-dtype 为准**；引用 FLA 反向绝对时间时必须写明 dtype。
  §25.3 的 B1「反超」与整链 0.48–0.54× 更正为 **B1 慢 ~2.3–2.5×、整链 ~0.86–0.95×**。
- **对本专题主线结论的影响**：目标卡反向的工程重点不变（B1 的 16 块粒度 + store 效率、B2/B3 融合），
  但**不能**再宣称「已经反超 FLA」；诚实定位是「**与 FLA 反向同档（~0.9×），且省掉 FLA 的前向状态重算**」。
- **方法学**：这条与 §25.5（孤立 vs 链内归因差 ~1.3×）、§11.4（chunk 粒度）并列，是本专题第三个
  「计时口径」陷阱——**跨实现的 `xFLA` 必须先对齐 dtype、chunk、归因窗口三者**。

## 38. 把「K2 形状 CUDA B1」的累计收益收口成 matched-bf16 的 `xFLA`，并暴露手写 pipeline 在 `NSTAGE≥2` 的竞态（P7-35）

§22–§36 把目标卡形状的 **CUDA B1**（寄存器常驻状态 + `ldmatrix`/`mma.sync` + TMA 输入 + warp
specialization）一轮轮垒起来，但每轮报的 `xFLA` 分母**各不相同**：§22 2.10–3.75、§23 1.73–3.80、
§24 最好 1.63（都是 `-DB1T_ETMA`、**标量** `dh`/`dv2` store），而 §35/§36 的 TMA store 只报了
**相对自身**的提速（`tma2/base`、`lg_throttle` 等），**从未**把当前最强组合
`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`（记作 `tma2`）重新对回 FLA。§37（P7-34）又把口径钉死为
**matched dtype**（FLA `dhu` 必须喂 **bf16**）。本节用
`harness/check_bwd_b1_xfla.py`（`harness/check_bwd_b1_xfla.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）
把这条线**收口**，并顺手加了一个 run-to-run 确定性扫描。

三个变体（都带 `-DB1T_ETMA`，`E^tot` 已在流水里）：

- `base`＝标量 `dh` + 标量 `dv2`（§24 的最好形态）；
- `tma` ＝TMA `dh` + 标量 `dv2`（P7-32）；
- `tma2`＝TMA `dh` + TMA `dv2`（P7-33，**当前最强**）。

### 38.1 Part A：正确性（bit-identical + bf16 1 ulp）

`tma2` vs `base` 在 `BT=32/64`、`NS=1/2`、4 组 shape（含非整除 `T=250`）上
`dh/dv2/dh0` **逐位相同（max_abs=0）**；`tma2` vs 直接 bf16 FLA `dhu` 仍 **bf16 1 ulp**
（`dh/dv2` max_abs `4.88e-4`，`T=250` 的 `dv2` 为 `2.44e-4`）——与 §35/§36 一致。

### 38.2 Part B/C：`tma2` 对 matched-bf16 FLA `dhu` 的 `xFLA`

FLA 分母 = **直接调用** `chunk_gated_delta_rule_bwd_dhu` 且喂 **bf16 操作数**（同 §37 口径），
CUPTI `us/call`。`xFLA = ours / FLA_bf16`，`BT=64` 与 FLA `dhu` 的默认 chunk 一致：

| shape | ours `tma2` NS=1 | ours `tma2` NS=2 | FLA `dhu` bf16 (BT=64) | `xFLA` NS=1 | `xFLA` NS=2 |
|---|---|---|---|---|---|
| B1 T512 H4 | 29.9 | 25.4 | 13.0 | **2.30** | 1.96 |
| B2 T1024 H8 | 62.1 | 47.9 | 23.1 | **2.69** | 2.07 |
| B8 T2048 H8 | 137.6 | 99.0 | 119.6 | **1.15** | **0.83** |
| B1 T250 H3 | 16.7 | 14.9 | 7.4 | **2.24** | 2.00 |

（CUPTI `us/call`；FLA 值 `13.0/23.1/119.6/7.4` 与 §37 Part A 的 `13.1/23.3/120.0` 一致。）

**读法**：

- **大 shape（B8 T2048 H8）**：`tma2` 把 §24 的 `xFLA=1.63`（标量 store）压到 **1.15（NS=1）/0.83
  （NS=2）**——即 TMA store 之后，K2 形状的 CUDA B1 在**与 FLA 同 chunk、同 dtype** 下已**追平甚至略快**
  （NS=2 时 99.0 µs vs FLA 119.6）。这坐实了 §35/§36「两条逐标量 store 改 TMA 恢复全删上界 ~75%」的
  累计效果：**它确实把最大 shape 的 xFLA 从 1.63 拉到 ≤1.15**。
- **小 shape（B1 T512 H4 / B2 T1024 H8 / B1 T250 H3）**：`xFLA` 仍 **2.0–2.7×**，与 §22–§24 基本未变
  ——因为这些 shape 的瓶颈不是 store 而是**每 warp 发射效率/并行度**（§35.5：`wait`/`tensor`），
  TMA store 对它无能为力。这也解释了为什么 §24 的「全表最好 xFLA=1.63」只出现在 `B8T2048H8`。
- **BT=16（部署粒度）无法在这里测 `xFLA`**：FLA 前向 `chunk_kda_fwd_intra` **只支持 chunk 32/64**
  （直接跑 `BT=16` 报 `ValueError: ... only supports chunk_size 32 or 64`），故 CUDA 探针**喂不进**
  16 块操作数；§37.3 的「部署粒度 BT=16 的 bf16 B1 慢 FLA 2.3–2.5×」仍是来自 Triton 组装链的口径。

### 38.3 Part D：手写 mbarrier pipeline 在 `NSTAGE≥2` 存在 run-to-run 竞态

为在正确性里覆盖 `BT=16`，本节给探针加了**重复运行确定性扫描**（同一 `base` 连跑两次、
`tma2` 与 `base` 对比；`base` 不含 TMA store，故竞态不能赖在 store 上）：

| shape | BT | NS | `base`/`base` `dh` | `tma2`/`base` `dh` | verdict |
|---|---|---|---|---|---|
| B2 T1024 H8 | 16 | 1 | 0 | 0 | ok |
| B2 T1024 H8 | 16 | 2 | 0 | 0 | ok |
| B8 T2048 H8 | 16 | **2** | **3.64e-2** | 1.29e-3 | **RACY** |
| B8 T2048 H8 | 16 | 3 | 0 | 0 | ok |
| B8 T2048 H8 | 32 | **3** | **4.52e-3** | 0 | **RACY** |
| B8 T2048 H8 | 32 | 1/2 | 0 | 0 | ok |
| B8 T2048 H8 | 64 | 1/2 | 0 | 0 | ok |

即：**`base` kernel 本身在 `NSTAGE≥2` 的大 shape 上不是 run-to-run 可复现的**（两处复现：
`BT=16 NS=2` 3.6e-2、`BT=32 NS=3` 4.5e-3），而 **`NSTAGE=1` 在所有 `BT` 上都稳定**。竞态是
**间歇性**的——小 shape、以及本轮多数 `NS=2` 行偶然全对，所以 §35/§36 当时「16 组逐位相同」是
**观测到的、而非有保证的**。含义有三：

1. **回授 §24 的结论**：P7-21「`E^tot` 修好后最优回到 `NS=1`」不只是性能选择，**`NS=1` 也是唯一
   有保证正确的配置**；目标卡 CUDA B1 应以 `NSTAGE=1` 为基准（或改 CUTLASS `PipelineTmaAsync` 重做
   多 stage）。
2. **本轮 `xFLA` 表的 NS=1 列可信**（`B8T2048H8` 1.15；小 shape 2.3–2.7），NS=2 列的时间仍然有效
   （计时不受竞态影响）但其数值来源是未完全排干的流水。
3. 竞态根因**未定位**（本节只确认它在 `base` 输入流水、非 store 路径，且 `NS=1` 免疫）——留作
   **待验证**：手写 `full`/`empty` 的 parity 协议在 `NSTAGE≥2` 下是否存在 off-by-one 的
   producer/consumer 覆写窗口。

### 38.4 结论 / 对上游的修正

- **K2 形状 CUDA B1 的累计 `xFLA`（matched bf16、BT=64）**：大 shape `B8T2048H8` **1.15（NS=1）/
  0.83（NS=2）**（§24 标量 store 时 1.63）；小 shape 仍 **2.0–2.7×**。即 **TMA store 兑现了「大
  shape 追平 FLA」这一半**，而**小 shape 的残差是每 warp 发射效率**（§24.5 的手写 mbarrier 上限），
  与 store 无关。
- **诚实定位**：目标卡 CUDA B1 **在大 shape 上已与 FLA bf16 同档或略快、在小 shape 上仍慢 ~2–2.7×**；
  部署粒度（BT=16）的对比只能由 Triton 链给出（§37.3：慢 ~2.3–2.5×，差距是**16 块粒度**）。
- **新增方法学约束**：多 stage 的手写流水 kernel 必须做 **run-to-run 确定性检查**（本节的 Part D），
  否则会把间歇竞态当正确性通过。这与 §25.5（归因窗口）、§37（dtype/chunk 对齐）并列。
- **下一步**：与 §35/§36 一致——手写 mbarrier 已到结构上限，若要安全地吃多 stage 的收益需换
  CUTLASS `PipelineTmaAsync`（本节的竞态是又一条动机）。

## 39. 定位并修掉手写 TMA 流水在 `NSTAGE≥2` 的竞态——缺一条 `fence.proxy.async`（P7-36）

§38.3（P7-35）在加 `BT=16` 覆盖时发现：K2 形状的手写 CUDA B1（`b1_tma_probe.cu`，`-DB1T_ETMA`
基线）在大 grid、`NSTAGE≥2` 下 **run-to-run 不可复现**，而 `NSTAGE=1` 稳定；当时只把它**观测**下来，
根因留作 **待验证**（猜是 `full`/`empty` parity 的 off-by-one）。本轮用一组受控变体把它**钉死并修掉**。
新增 `harness/check_bwd_b1_race.py`（`harness/check_bwd_b1_race.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 39.1 Part A：竞态由「消费 TMA 落盘的 `st->e`」触发，`fence.proxy.async` 是唯一非暴力解

在同一份 `b1_tma_probe.cu` 上编译 9 个变体，对每个变体**连跑同一个 kernel `NR=40` 次、逐次比 `dh`**
（`B8 T2048 H8`，大 grid = 复现区间）：

| BT | NS | base | noetma | dropread | **proxy** | **pro** | **con** | esync | evol | sync |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 2 | **39/40** | 0 | 0 | **0** | **0** | **0** | 39/40 | 10/40 | 0 |
| 16 | 3 | **11/40** | 0 | 0 | **0** | **0** | **0** | 10/40 | 39/40 | 0 |
| 32 | 3 | **19/40** | 0 | 0 | **0** | **0** | **0** | 7/40 | 0 | 0 |
| 32 | 2 | 1/40 | 0 | 0 | **0** | **0** | **0** | 0 | 0 | 0 |
| 64 | 2 | 0/40 | 0 | 0 | **0** | **0** | **0** | 0 | 0 | 0 |

（cell = 与第 0 次不同的 run 数 / 40；0 表示 run-to-run 逐位一致。变体：`noetma` = 关掉 ETMA、
E^tot 直接读 global；`dropread` = 仍发 E 的 TMA、但 op2/op3 读 global；`proxy` = 两个循环里各加一条
`fence.proxy.async`；`pro`/`con` = 只在 producer / 只在 consumer 一侧加；`esync` = op2/op3 前
`__syncwarp`；`evol` = `volatile` 读 `st->e`；`sync` = 两个循环各加 `__syncthreads`。）

**读法**：

- 竞态的**触发条件是「消费 TMA 落盘的 `st->e`」**：`noetma` 与 `dropread`（都不碰被复用的
  `st->e`）全程 0/40，而只要 op2/op3 读 `st->e` 就复现。
- **`fence.proxy.async` 是解，且两侧任一侧单独加都够**：`proxy`/`pro`/`con` 全 0/40。这不是
  parity 问题（`sep`——给 E 单独的 mbarrier——在 §38 的表里仍 racy），而是**跨 proxy 的内存序**问题。
- **`esync`/`evol` 修不了**：`__syncwarp` 只同步 warp 内执行、`volatile` 只约束编译器，二者都不
  建立 **generic proxy ↔ async proxy** 之间的顺序，所以 racy 照旧。
- **`sync`（`__syncthreads`）能修，但代价是把流水退化成串行**：它强制 producer 与所有 consumer
  每轮 rendezvous，等于把 `NSTAGE` 折回 1——§38 说「`NS=1` 是唯一有保证正确的配置」正是这个原因。
- base 的「racy 计数」本身**每次扫描都在抖**（`16/2` 这格先后测到 39、14、39；`32/3` 测到 19、24），
  这就是**间歇性竞态**的直接指纹。

### 39.2 Part B：损坏的形状——**一整个 compute warp 的 16 列 V-slice × 连续若干 chunk**

对 racy 的 `base`（`BT=32 NS=3`，`NR=60`）程序化统计每次失败 run 里 `dh` 的错误元素落点：

- **非可复现 run：29/59**（run 0 之后）。
- **chunk 局部性**：错的是**连续一段 chunk**（span 内的 chunk 基本都被污染，平均只漏 ~3.7 个），
  说明一个 chunk 的输入操作数错后会沿递推往后传几个 chunk。
- **K-row 直方图**（128 行）最高的是 **56–61 行各命中 29 次**（即整个 16 行的 `m=3` K-tile）。
- **V-col 直方图**（按 16 列一个 warp-slice 归并）：**`[0,16)` 命中 320 次、`[64,80)` 命中 240 次**，
  其余 slice（`[16,32)`/`[32,48)`/`[48,64)`）几乎不出现。

`cols=[0,16)` 是 `v0=0` 那个 CTA 的 **compute warp 0** 的 V 列，`[64,80)` 是 `v0=64` 那个 CTA 的
**compute warp 0**——即**永远是每个 CTA 的第 0 个 compute warp**，且是**整段 16 列**一起错。
这只有一种解释：**该 warp 正在读某个 stage 缓冲时，producer 用下一轮的 TMA 把这块缓冲整个覆写了**
（WAR）——被覆写的 q/k/w/dO/dV/E 一起错，所以整段 V-slice 丢失；而 `m=3` K-tile 高亮是**E 那 512 B
的 TMA 最先落盘**、只污染到 E 对应的行（`st->e[m*16+g]`）的时序残影。其他 warp 偶发落后不多、或
引用 §38 的 rance 窗口，故不被命中。

### 39.3 Part C：两个「安全解」的开销——`fence.proxy.async` ≈ 免费，回读 global 要贵 4–27%

CUPTI `us/call`，空闲卡：

| shape | BT | NS | base | noetma | **proxy** | noetma/base | **proxy/base** |
|---|---|---|---|---|---|---|---|
| B2 T1024 H8 | 16 | 1 | 236.1 | 283.8 | 236.3 | 1.20 | **1.00** |
| B2 T1024 H8 | 32 | 1 | 141.1 | 150.5 | 141.3 | 1.07 | **1.00** |
| B2 T1024 H8 | 32 | 3 | 132.6 | 138.2 | 133.3 | 1.04 | **1.01** |
| B2 T1024 H8 | 64 | 1 | 91.7 | 95.6 | 93.8 | 1.04 | **1.02** |
| B8 T2048 H8 | 16 | 1 | 499.7 | 636.9 | 499.8 | 1.27 | **1.00** |
| B8 T2048 H8 | 32 | 1 | 290.6 | 328.7 | 295.8 | 1.13 | **1.02** |
| B8 T2048 H8 | 32 | 3 | 370.1 | 332.8 | 356.4 | 0.90 | **0.96** |
| B8 T2048 H8 | 64 | 1 | 195.2 | 218.1 | 198.2 | 1.12 | **1.02** |

**放弃 ETMA（`noetma`，回读 global）要贵 4–27%**（正是 §21/§24 当初引入 ETMA 的理由）；而
**`proxy` 相对 racy 的 `base` 是 0.96–1.02（在噪声内）**——修竞态**不花性能**。这条正是 §21 的
`long_sb` 归因没看到的那一块：E 留在流水里是对的，只是**欠一条 proxy fence**。

### 39.4 Part D：修完之后多 stage 与 `NS=1` **逐位相同**（只修了竞态，没动数值）

`proxy` 变体在 `NS=2/3` 下 vs `base` 的 `NS=1`，`dh` **max_abs 全部 0.000e+00**（`BT=16/32/64`、
两组 shape，12 格）：

| shape | BT | proxy NS=2 vs base NS=1 | proxy NS=3 vs base NS=1 |
|---|---|---|---|
| B2 T1024 H8 | 16/32/64 | 0.000e+00 | 0.000e+00 |
| B8 T2048 H8 | 16/32/64 | 0.000e+00 | 0.000e+00 |

即：**加 fence 后多 stage 与单 stage 数值完全一致**，`NSTAGE≥2` 从此可安全使用；§38 里「`NS=1`
是唯一有保证正确的配置」这条**回授被撤回**（改为「需先加 proxy fence」）。

### 39.5 结论 / 对目标卡的设计输入

- **根因（已定位，非待验证）**：手写 TMA 流水的 `full`/`empty` mbarrier 握手**只在 generic proxy
  内定序**；consumer 用普通 `ld` 读 `st->e`（generic proxy），producer 用 `cp.async.bulk.tensor`
  回填同一缓冲（**async proxy**），这一对 **WAR 缺 `fence.proxy.async`**。因为每个 stage 的
  `st->e` 是「读后即被下一轮覆写」的复用缓冲，缺 fence 时 consumer 可能读到 producer 已回填的
  错误字节。`st->e` 最先暴露是因为它的 TMA（512 B）最早落盘、且它用的是普通标量 `ld`（q/k/w/dO
  走 `ldmatrix` 且结果进 `mma`，完成度被数据依赖逼出来），故同一 hazard 在 E 上最先可见——
  **但对所有被 TMA 复用的 stage 缓冲，这条 fence 都是必需的**。
- **修法**：在**任一**侧加一条 `fence.proxy.async.shared::cta`（`b1_tma_probe.cu:59-61` 的
  `fence_proxy_async()`；producer 在 `empty` wait 之后、发 TMA 之前 `:282-284`，或 consumer 在读完
  `st->e`、`empty` arrive 之前 `:518-520`）。**代价在噪声内**（Part C），值取 `-DB1T_EPROXY`。
- **撤回 §24.5/§38.4 的过度悲观**：多 stage 的安全**不需要**换 CUTLASS `PipelineTmaAsync`——那条
  路子仍更省心，但「手写 pipeline 不能安全开多 stage」的结论是**错的**，一条 fence 即可。**新增
  方法学约束**：手写「TMA 复用缓冲」的流水，其 buffer-reuse 握手上必须显式 `fence.proxy.async`；
  确定性扫描（§38 的 Part D）是抓它的必要手段，`__threadfence_block`/`volatile` 会漏。
- **本条的边界**：实测都在**标量 store 的 `-DB1T_ETMA`** 基线（§38 不可复现的那个变体）上做；
  `+TMADH/+TMADV`（`tma2`）的 store 路径用的是 `cp.async.bulk` store + `bulk_group`，其复用握手
  同样建议补 fence，**待验证**（本轮未对其做同样的确定性扫描修复，但 §38 Part A 的 `tma2/base`
  逐位一致结果不受影响）。

## 40. 关闭 §39.5 的 `tma2` store 侧待验证——同一条 `fence.proxy.async`，但本探针观测不到该竞态（P7-37）

§39 把 §38 的 `NSTAGE≥2` 竞态根因钉死为**输入侧**的 generic↔async WAR（consumer 普通 `ld` 读
被 TMA 回填的 `st->e`），并给出 `-DB1T_EPROXY` 修法。但它把**输出侧**留作**待验证**：`tma2`
（`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV`，当前最强组合）新增了一对反向的 generic↔async——
compute warp 用 generic store 把 per-warp `dh`/`dv2` tile 写进 smem，再由 `cp.async.bulk.tensor`
（async proxy）读出去（**RAW**），且 tile 被双重缓冲复用（由 `cp.async.bulk.wait_group.read` 兜
WAR）。本轮补上这条 fence 并做与 §39 同款的确定性扫描，同时把扫描对象从「标量 store 的 `base`」
扩到**部署最强变体 `tma2` 本身**。新增 `harness/check_bwd_b1_tma2_race.py`
（`harness/check_bwd_b1_tma2_race.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。

### 40.1 实现：写 tile 与 `cp.async.bulk` store 之间的一条 fence

给 `b1_tma_probe.cu` 加 `-DB1T_SPROXY`：在 `dh`（`:402-403`）与 `dv2`（`:473-474`）两处**写完
smem tile、`__syncwarp()` 之后、lane 0 发 `tma5d_store`/`tma4d_store` 之前**各插一条
`fence_proxy_async()`（即 `fence.proxy.async.shared::cta`，`:59-60`）。放在 `if (l == 0)` **之外**，
由全 32 lane 执行——这条 fence 只定序**执行线程自己**先前的 generic 写与后续 async proxy 访问，
而每个 lane 各写自己的行/列，故必须全 lane 都 fence，再由 lane 0 发射（`__syncwarp()` 已把全 warp
的写汇齐）。语义上与 CUTLASS 的 `tma_store_fence`（`fence_view_async_shared`）一致。

变体矩阵（都以 `-DB1T_ETMA` 为基）：

| 名 | flags | 含义 |
|---|---|---|
| `scalar` | `-DB1T_ETMA` | §38 的 racy 对照（标量 `dh`/`dv2`，无任何 fence） |
| `scalar_p` | `+ -DB1T_EPROXY` | 输入侧修好、无 store TMA（store 侧无 hazard） |
| `tma2` | `+ -DB1T_TMADH -DB1T_TMADV` | **部署最强**（§38），两侧都无 fence |
| `tma2_s` | `+ -DB1T_SPROXY` | 只加 store 侧 fence |
| `tma2_in` | `+ -DB1T_EPROXY` | 只加输入侧 fence（store RAW 仍开着） |
| `tma2_full` | `+ -DB1T_EPROXY -DB1T_SPROXY` | 两侧都修（安全部署配置） |

### 40.2 Part A：`scalar` 复现 racy，而 **`tma2` 全族（含无 fence 的 `tma2`）本探针观测为 clean**

同一 kernel 连跑 `NR=40`、逐次比 `dh`（大 grid `B8 T2048 H8`，§38/§39 的复现区间）：

| BT | NS | scalar | scalar_p | tma2 | tma2_s | tma2_in | tma2_full |
|---|---|---|---|---|---|---|---|
| 16 | 2 | **39/40** | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| 16 | 3 | **39/40** | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| 32 | 3 | **39/40** | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| 32 | 2 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |
| 64 | 2 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 | 0/40 |

（cell = 与第 0 次不同的 run 数 / 40。`scalar` 三次大配置**全 racy（39/40）**，与 §38/§39 一致。）

**读法（诚实口径）**：

- **`scalar`（标量 store）稳定复现输入侧竞态**，说明扫描本身有效、探针处于敏感窗口。
- **`tma2` 全族（含**两侧都没 fence**的 `tma2`）在本探针 `NR=40` 下都 0/40**——即**输出侧的
  generic→async RAW/WAR 在本探针上观测不到**。这与 §35/§36 当时「16 组逐位相同」的**观测**一致。
  机制推断：换成 `cp.async.bulk` store 后，`st->e` 的消费时点相对 producer 的 TMA 回填被推后
  （store 引擎与 `bulk_group` 等待改变了指令节奏），把 §39 那个输入侧覆写窗口拱出了敏感区。
- **但这只是「观测到 clean」，不是「有保证 clean」**（正是 §38.3 的教训）：`st->e` 的输入侧
  hazard 依旧存在（`tma2` 未加 `EPROXY`），只是本轮窗口没撞上；输出侧 RAW 同理。故**部署配置仍取
  `tma2_full`**（两侧 fence），代价见 Part C。

### 40.3 Part B：fence 只定序、不改数值——`tma2_full` vs `scalar_p` **逐位相同**

`tma2_full`（两侧 fence 的 TMA store）与 `scalar_p`（标量 store）在同一份操作数上比
`dh/dv2/dh0`，4 组 shape（含非整除 `T=250`）× `BT=32/64` × `NS=1/2` 共 16 格，**max_abs 全为
`0.00e+00`**；vs 直接 bf16 FLA `dhu` 仍 **1 ulp**（`dh` 4.88e-4、`dv2` 4.88e-4/6.10e-4、
`T=250` 的 `dv2` 2.44e-4）——与 §35/§36/§38 逐值一致：

| shape | BT | `dh` | `dv2` | `dh0` | vsFLA `dh` | vsFLA `dv2` |
|---|---|---|---|---|---|---|
| B2 T1024 H8 | 32/64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.88e-4 (BT64) | 4.88e-04 |
| B8 T2048 H8 | 32/64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.88e-4 (BT64) | 4.88e-04 / 6.10e-04 |
| B1 T512 H4 | 32/64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.88e-4 (BT64) | 4.88e-04 |
| B1 T250 H3 | 32/64 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 4.88e-4 (BT64) | 3.66e-04 / 2.44e-04 |

即：**store 侧 fence 是纯 ordering，不引入任何数值差异**（`dh` 只在 `BT=64` 与 FLA chunk 对齐时
才可逐元素比，故 `BT=32` 行 vsFLA `dh` 记 `--`）。

### 40.4 Part C：store 侧 fence 的代价在噪声内（CUPTI `us/call`）

| shape | BT | NS | tma2 | tma2_full | scalar_p | full/tma2 | scalar_p/full |
|---|---|---|---|---|---|---|---|
| B2 T1024 H8 | 16 | 1 | 137.7 | 140.7 | 239.3 | 1.021 | 1.70 |
| B2 T1024 H8 | 16 | 2 | 76.4 | 82.6 | 205.9 | 1.081 | 2.49 |
| B2 T1024 H8 | 32 | 3 | 56.3 | 56.9 | 133.4 | 1.011 | 2.34 |
| B2 T1024 H8 | 64 | 1 | 62.6 | 63.6 | 93.1 | 1.017 | 1.46 |
| B8 T2048 H8 | 32 | 1 | 197.7 | 201.2 | 295.0 | 1.018 | 1.47 |
| B8 T2048 H8 | 32 | 3 | 125.3 | 124.9 | 354.6 | 0.997 | 2.84 |
| B8 T2048 H8 | 64 | 1 | 137.8 | 139.6 | 198.3 | 1.013 | 1.42 |
| B8 T2048 H8 | 64 | 2 | 99.4 | 99.2 | 206.1 | 0.998 | 2.08 |

**`full/tma2` = 0.997–1.081（多数 ~1.01，噪声内）**——加两条 store 侧 fence **免费**（与 §39 Part C
的输入侧 fence 同为噪声内）。顺带复核 `tma2_full` 相对标量 store 的累计优势仍在
（`scalar_p/tma2_full` = 1.42–2.93），即 §35/§36 的 TMA store 收益不受本轮影响。

### 40.5 Part D：`tma2_full` 多 stage 与 `NS=1` **逐位相同**

`tma2_full` 在 `NS=2/3` vs `NS=1`，`dh` **max_abs 全 `0.000e+00`**（`BT=16/32/64`、两组 shape、
`BT=64` 的 `NS=3` 因 smem OOM 记 `--`）。与 §39 Part D 合并：**两侧各一条 `fence.proxy.async` 后，
手写 TMA 流水在 `NSTAGE≥2` 既 run-to-run 确定、又与单 stage 数值完全一致**。

### 40.6 结论 / 对目标卡的设计输入

- **同一条 `fence.proxy.async` 覆盖两个方向**：输入侧（`st->e` 的 WAR，§39）与输出侧（`dh`/`dv2`
  tile 的 RAW，本轮）。二者机制对称——**凡是「generic proxy 写/读」与「async proxy（TMA）读/写」
  共用同一块 smem 缓冲**，都必须显式 `fence.proxy.async`，`full`/`empty` mbarrier 只定序 generic
  proxy 内部。
- **本轮诚实定位**：输出侧 RAW/WAR 在 `X_FLASHKDA` 探针上**观测不到**（`tma2` 全族 0/40），只有
  输入侧在标量 store 基线可复现；但「观测 clean」不等于「有保证 clean」（§38.3 教训），故**部署
  配置取 `tma2_full`**（`-DB1T_EPROXY -DB1T_SPROXY`），代价在噪声内、数值逐位不变。
- **对目标卡 CUDA B1 的最终配置**：`-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY`，
  `NSTAGE` 可按需 >1（已确定性 + 逐位无损）；§22–§36 的 `xFLA`（大 shape ≤1.15、小 shape 2.0–2.7）
  具此配置不变。
- **方法学（补充 §38.3）**：确定性扫描应覆盖**部署最强变体本身**、而非只扫一个基座——本轮把
  §39 的「输入侧已定位、输出侧待验证」收口为「输出侧同需要 fence、但探针观测不到」，并留下唯一
  边界**待验证**：若换用更深 pipeline 或不同 store tile 形状（改变写→store 的时序），输出侧竞态的
  可观测性可能变化，届时仍以 `tma2_full` 为准。

## 41. 关闭 §40.6 的「更深 pipeline / 更深 store 缓冲」边界：输出侧 RAW 在本探针仍不可观测（P7-38）

§40.6 把 P7-37 收口后只留了一条边界**待验证**：「若换用更深 pipeline 或不同 store tile 形状
（改变写→store 的时序），输出侧竞态的可观测性可能变化」。本轮把**两个旋钮都推到探针上限**再扫一遍
确定性，回答这条边界。新增 `harness/check_bwd_b1_deep_ns.py`（`check_bwd_b1_deep_ns.out.txt`，
空闲卡）与 `b1_tma_probe.cu` 的两处参数化。

### 41.1 实现：`NSTAGE` 加深到 6、输出缓冲深度 `B1T_SDEPTH` 参数化

- **更深 pipeline**：`b1_tma_probe.cu` 的 `b1t_launch`/`b1t_smem_bytes` 新增 `NSTAGE=4`（BT=32/16）、
  `5,6`（BT=16）的实例化。§39 的输入侧 WAR 发生在 producer 用 `cp.async.bulk.tensor` 覆写
  `st->e`（`i>=NSTAGE` 才复用 stage，`b1_tma_probe.cu:287`），`NSTAGE` 越大、覆写窗口越靠后，
  是对该 WAR 的直接加压。
- **不同 store 时序**：新增宏 `B1T_SDEPTH`（默认 2，即 §35/§36 的 `dho[2]`/`duo[2]` 双缓冲），
  把输出 tile 缓冲改成 `B1T_SDEPTH` 深、按 `i % B1T_SDEPTH` 索引复用、由
  `cp.async.bulk.wait_group.read B1T_SDEPTH-1` 释放（`b1_tma_probe.cu:366-407/454-478`）。
  `SDEPTH=1` 单缓冲（每个 chunk 全排空，序列化）、`SDEPTH=3/4` 让若干 tile 的 generic 写与
  async-`cp.async.bulk.tensor` 读同时在飞——这正是 §40 输出侧 RAW（写 tile → TMA 读出去）与
  缓冲复用的 WAR（读未完成即覆写）最容易暴露的窗口。

### 41.2 Part A：`scalar` 复现 racy，而 **`tma2` 全族在 NS≤6 / SDEPTH≤4 仍 0/80**

同一 kernel 连跑 `NR=80` 次逐次比 `dh`（`B8T2048H8`，§38/§39 的失败 regime），覆盖
`BT∈{16,32}` × `NS`（16:1–6 / 32:1–4）× `SDEPTH∈{1,2,3,4}` × 变体
{`scalar`(仅 `-DB1T_ETMA`，已知输入侧 racy)、`scalar_p`(§39 输入侧 fence)、
`tma2`(无 fence)、`tma2_full`(部署，两侧 fence)}：

- **对照有效**：`scalar`（无 fence）在 `BT=16 NS=2/3/4` 稳定 **79/80 racy**、`BT=32 NS=3` **79/80**，
  换 `NS=1` 即 0/80（与 §38/§39 的「`NSTAGE≥2` 才触发」指纹一致）。`scalar_p` **全 0/80**——
  证明扫描有检出竞态的功率。
- **`tma2` 全族（含两侧都没 fence 的 `tma2_sd2`）在全部 60 个 `(BT,NS,SDEPTH)` 组合上 0/80**：
  pipeline 加深到 `NS=6`、store 缓冲加深到 `SDEPTH=4`、甚至单缓冲 `SDEPTH=1`，都**观测不到**
  输入侧 WAR（`st->e`）或输出侧 RAW/WAR（`dh`/`dv2` tile）。`tma2_full` 同样全 0/80。
- **一个值得记的细节**：`scalar` 的 racy 概率随 `NS` 变化（`NS=2/3/4`≈79/80 → `NS=5` 18/80 →
  `NS=6` 6/80），说明「更深 pipeline」并未单调放大该竞态窗口，反而把时序拱出敏感区。这提示
  §38/§39 的「racy/clean」是**强时序相关**的观测，任何单一配置的 clean 都不能外推。

### 41.3 Part B：加深后仍**逐位无损**（fence 与深度都只改定序）

`tma2_full`（BT=16 `NS=3,SDEPTH=3`；BT=64 `NS=2,SDEPTH=2`）vs `scalar_p`：`dh/dv2/dh0`
**逐位相同（max_abs=0，含非整除 `T=250`、`B1T512H4`）**；vs 直接 bf16 FLA `dhu` 在 `BT=64` 仍
**1 ulp**（`dh/dv2` = 4.88e-4/2.44e-4）。`BT=16` 的 `dv2` vs FLA 为 4.1e-3~7.0e-3——这是 **16 vs 64
的 chunk 粒度差**（FLA 前向 chunk 只支持 32/64），非本轮深度引入，与 §37.3 的部署粒度结论一致。

### 41.4 Part C：BT=16/32 上**更深 pipeline 确实更快**（与 §24 的 BT=64 结论相反）

部署配置 `tma2_full`、CUPTI `us/call`：`B8T2048H8` 上 BT=16 `NS=1→4` 从 **300.7→177.8µs**（1.69×，
`NS=5/6` 回到 ~188，最优 `NS=4`）；BT=32 `NS=1→3` 从 **203.4→124.6µs**（1.63×，`NS=4` 略回落）。
`B2T1024H8` 同趋势（BT=16 `NS=1→6` 140→75µs）。**`SDEPTH∈{1,2,3}` 在本 shape 上无差别**（同 `NS`
下差异 ≤3%＝噪声），即输出缓冲深度**不是**性能旋钮。对照 §24：那里 BT=64 + 标量 `dh` store 时
「最优回到 `NS=1`」，本轮换成 TMA store 且 BT=16/32 后深 pipeline 有正收益——说明「最优 stage
深度」取决于 **store 路径与 BT**，不可跨配置外推。

### 41.5 Part D：深 NS/SDEPTH 与 `NS=1` 基线**逐位相同**

`tma2_full`（`SDEPTH=3`）`NS=6`（BT=16）/`NS=4`（BT=32）vs 同库 `NS=1`：`dh` **max_abs=0**（四组
shape）。即加深 pipeline/缓冲只改时序、不改数值（与 `SDEPTH` 无关、与 `NS` 无关）。

### 41.6 结论 / 对目标卡的设计输入

- **§40.6 的边界关闭（答案：本探针仍不可观测）**：把 pipeline 加深到 `NS=6`、store 缓冲加深到
  `SDEPTH=4`（含单缓冲 `SDEPTH=1` 的极端复用）后，`tma2` 全族在 60 个组合上**仍 0/80**——更深
  pipeline / 不同 store 时序**没有**让输出侧（或 TMA 形状下的输入侧）竞态变得可观测。而**标量
  store 基线的输入侧竞态**在同一扫描里稳定复现（79/80），证明扫描有功率。故上一轮的判断不变，
  且更强：**在这条探针里，两侧 `fence.proxy.async` 都是「编程模型必需、但观测不到」**；部署继续取
  `-DB1T_EPROXY -DB1T_SPROXY`（免费、逐位无损），无需再为「更深的 pipeline」保留不确定项。
- **新增正面数据**：TMA store + `BT=16/32` 上，**`NS=4` 是明确的免费加速**（`B8T2048H8` 1.6–1.7×），
  输出 buffer 深度 `SDEPTH` 取 2 即可（3/4 无收益、且 `SDEPTH=4` 在 BT=32 `NS=4` 已超 smem）。
  目标卡 CUDA B1 的 `NSTAGE` 建议 ≥4（不再是 §24 的「回到 1」，那是 BT=64/标量 store 的结论）。
- **方法学（补充 §38.3/§40.6）**：(1) 确定性扫描的 `NR` 要足够大且覆盖**部署最强变体本身**；
  (2) 「更深/更极端」不必然放大竞态窗口（`scalar` 在 `NS=5/6` racy 概率反降），故 clean 的
  观测**只在被扫过的配置上**成立；(3) 性能旋钮（`NSTAGE`）的最优值依赖 store 路径 + BT，须实测。

## 42. 把 K2 形状的 CUDA B1 落进**完整反向链**：补 `dht` 入口、drop-in 核验与链级计时（P7-39）

§15–§41 把目标卡形状的 CUDA B1（`b1_tma_probe.cu`：状态常驻寄存器 + 裸 `mma.sync`/`ldmatrix` +
TMA 操作数流水 + TMA `dh`/`dv2` store + 两侧 `fence.proxy.async`）在**孤立情形**做到了收敛，部署配置为

```
-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY      (NSTAGE>=1)
```

但所有那些轮次都是 **kernel vs kernel**；CUDA B1 **从未**接进组装好的反向链
（`dAv -> B1 -> B2(wy_dqkg) -> B3(intra) -> 反向 cumsum`），而后者的 B1 一直是 **Triton** 版
（`dhu_bf16_pipe`，§25）。于是有两件事一直悬着：**(i) 接口**——CUDA 核没有 `dht`（终态伴随）
入口，而 Triton B1 有（`HAS_DHT`，P6-7）；**(ii) 链级图像**——换成 CUDA B1 后六项梯度是否仍
在 bf16 底噪、部署粒度 `BT=CHUNK=16` 下的链级时间如何。本节（`harness/check_bwd_b1_cuda_e2e.py`
+ 输出 `harness/check_bwd_b1_cuda_e2e.out.txt`）把这两件事一次做掉。

### 42.1 接口：给 CUDA B1 补 `dht` 入口

新增 kernel 参数 `const float* dht`（`b1_tma_probe.cu:258`），在 compute warp 的寄存器状态
初始化处按 [K,V] 片段布局写入（`b1_tma_probe.cu:341-357`）：

```cpp
float S[MTK][2][4];
if (dht != nullptr) {                       // [B,H,KD,VD] fp32，行=K、列=V
    const int64_t dbase = (int64_t)nh * KD * VD;
    for (int m...) for (int nn...) {
        int r0 = m*16+g, r1 = r0+8;         // mma C 片段的行（g=lane/4）
        int gc = v0 + warp*BW + nn*8 + t*2; // 该 warp 的 V 列（t=lane%4）
        S[m][nn][0] = dht[dbase + r0*VD + gc];   // (r0,gc)
        ...                                       // x=0..3 对应 (r0/1, gc/1)
    }
} else { /* 原路径：S=0 */ }
```

数值上与 Triton B1 的 `HAS_DHT` 完全相同：后者在反向扫描前置 `b_dh = dht`
（`check_bwd_b1_bf16_e2e.py:77-80`）。为不动既有调用者，新增独立符号 `b1t_launch_dht`
（`b1_tma_probe.cu:739`），`b1t_launch` 仍传 `nullptr`；两者共用 `dispatch_launch`
（`b1_tma_probe.cu:706`）。

### 42.2 Part A：drop-in 核验（同一份 workspace 操作数、`dht=dfin≠0`）

用组装链的真实操作数（`reconstruct` 出来的 workspace 场 + `dav` 的输出 `dv`），把 CUDA B1 与
Triton B1（`dhu_bf16_pipe`，`NS=3`）并排跑，`dht=dfin`（非零）。6 组 case（含非整除 T=250/513、
g=0、B=2）：

| 量 | CUDA vs Triton `max_abs` 区间 | 说明 |
|---|---|---|
| `dh`  | 1.24e-4 … 3.68e-4 | = **1 bf16 ulp**（参考量级 0.04–0.09），CUDA 落盘 `dh` 是 bf16、Triton 链保持 fp32 |
| `dv2` | 1.18e-4 … 2.35e-4 | 同上 |
| `dh0` | 1.41e-5 … 1.54e-4 | 同上（`dh0` 双方都 fp32） |

把 `dht` 去掉（`dht=None`）后 `dh` 与带 `dht` 的差 `max_abs` 高达 **3.8e-2 … 4.5e-2**
（`max_rel` 0.52–5.4）——证明 `dht` 入口确实生效、且是终态上游梯度的**必要**输入。**结论**：在
bf16 落盘精度内，CUDA B1 是 Triton B1 的**逐位等价 drop-in**。

### 42.3 Part B：六项梯度链 vs FLA autograd

把 CUDA B1（`BT=16`、`NSTAGE=4`）接进 §25 的同一条链（`dav -> B1 -> wy_dqkg -> intra -> 反向
cumsum`），与 Triton B1 链（`NS=3`）同输入、与 FLA `chunk_kda` autograd（`torch.autograd`）比。
6 组 case 的最坏 `max_rel`：

| 项 | CUDA-B1 链 vs FLA | Triton-B1 链 vs FLA | CUDA vs Triton |
|---|---|---|---|
| `dq`     | 3.423e-3 | 3.423e-3 | **0.000e+00（逐位）** |
| `dk`     | 3.308e-3 | 3.024e-3 | 1.898e-3 |
| `dv`     | 6.795e-3 | 6.795e-3 | 2.338e-3 |
| `dbeta`  | 6.012e-3 | 6.012e-3 | 1.609e-3 |
| `dg`     | 7.064e-3 | 7.064e-3 | 1.088e-3 |
| `dh0`    | 9.365e-3 | 8.267e-3 | 0.000e+00（逐位） |

`dq` 与 `dh0` 两链**逐位相同**（`dq` 只经 `do@h`、不碰 B1 的 `dh`/`dv2`；`dh0` 是 B1 的终态输出里
不含 `dh` 落盘的那部分）。其余四项两链差 ≤2.3e-3，全部落 bf16 底噪。**结论**：CUDA B1 换进去后，
整条反向链的六项梯度与 FLA 仍同阶、与 Triton 链等价。

### 42.4 Part C：链级计时（wall 为主口径）

反向-only，同一口径（`wall` = 一个 CUDA event 对包 `REPS=30` 次取均值、3 轮中位；`dev` = CUPTI）。
CUDA-B1 链的计时闭包内**只保留部署本就需要**的类型转换（`dv` 转 bf16、`dv2` 回 fp32 喂 B2），
不像 P7-39 早期版本那样把 `dh`/`dv2` 整体 `.float()`（那会在 `B8T2048H8` 上凭空多出 ~1.5ms 的
elementwise 税，是本轮先踩到的坑）。

| case | FLA wall | Triton-B1 链 wall | CUDA-B1 链 wall (NS2/NS4) | CUDA/Triton |
|---|---|---|---|---|
| T=512 H=4        | 685.7  | 274.9  | 282.2 / 283.1 | **1.03×** |
| B2 T=1024 H=8    | 2093.4 | 1155.3 | 1175.2 / 1176.4 | **1.02×** |
| B8 T=2048 H=8    | 13588.8| 7465.4 | 7047.2 / 7036.2 | **0.94×** |
| T=250 H=3 (non)  | 777.3  | 245.9  | 406.8 / 407.0 | **1.65×** |

kernel 级归因（CUPTI，同一窗口）：小 shape 上 CUDA B1 比 Triton B1 慢 ~1.2×（T512H4 40.8 vs 34.8、
B2T1024H8 79.2 vs 62.1、T250H3 22.6 vs 18.8），但在 `B8T2048H8` 上 `NS=4` 只有 **181.7µs vs
Triton 939.3µs**（CUPTI 在此形状读数波动大，见 §42.5，此处仅说明趋势）。链级 wall 因此**在
T 可整除的三个 shape 上是 0.94–1.03×**（大 shape 反而快 6%），与 Triton B1 链同档——即**把 B1
换成目标卡 CUDA 版不会拖慢整链**；`B8T2048H8` 的 6% 正是 §42.5 那个大 shape B1 优势被 B2/B3
（§26 已证占链 ~66%）稀释后的净效果。

`T=250 H=3` 的 **1.65× 回归**要单列：该 case 下 CUDA B1 的 kernel dev 只有 22.6µs（与 Triton 同档），
但链 wall 从上一轮（去掉多余 `.float()` 前）的 252µs 跳到 **407µs**——这是 **host 侧的开销**：
`launch_impl`（`b1_tma_probe.cu:660-684`）**每次 launch 都重新 `cuTensorMapEncodeTiled`**（×5–8 张
张量图）+ `cudaFuncSetAttribute` 检查，小 shape 的链是 host-bound，这笔账就显出来了。生产实现里
张量图应**一次性编码、跨 launch 复用**（目标卡落地时必须做），故此回归标为**待验证**（预期消除）。

### 42.5 Part D：部署粒度 `BT=16` 的 B1 单核头对头（wall 口径）

用 `torch.profiler` 的 CUPTI 归因测**孤立小 grid 的 B1** 在本机是**双峰**的（同一 kernel 相邻两次
可读成 ~41µs 或 ~130µs，§25.5 的窗口效应在这类 µs 级 kernel 上被放大），故 Part D 改用**一个
event 对包 `REPS=50` 次 launch** 的 wall（含 launch 开销但稳定），CUDA/Triton 交替测以抵消时钟漂移：

| shape | CUDA `NSTAGE` 最优 | Triton `NS` 最优 | 最优比 |
|---|---|---|---|
| B1 T512 H4   | 150µs | 143µs | 1.05× |
| B2 T1024 H8  | 285µs | 175µs | **1.63×** |
| B8 T2048 H8  | 581µs | 1094µs| **0.53×** |
| B1 T250 H3   | 38µs  | 34µs  | 1.11× |

**读数**：部署粒度 `BT=16` 下，手写 CUDA B1 在**小 shape 上与 Triton 同档或略慢**（1.05–1.63×），
但在 **`B8T2048H8` 上快 1.9×**——手写 TMA/深流水只有在**有足够并行度把延迟摊开**时才兑现，与 §41.4
「TMA store + BT=16/32 上深 pipeline 有正收益」自洽；小 shape 反而是 Triton 编译器流水更好。

### 42.6 结论 / 对目标卡的设计输入

- **接口补齐**：CUDA B1 现支持 `dht`（终态伴随）入口（`b1t_launch_dht`），与 Triton B1 的
  `HAS_DHT` 语义一致；去掉它会丢终态上游梯度（`dh` 变 ~50–500%）。
- **drop-in 成立**：CUDA-B1 链的六项梯度 vs FLA autograd 最坏
  `dq 3.42e-3 / dk 3.31e-3 / dv 6.80e-3 / dbeta 6.01e-3 / dg 7.06e-3 / dh0 9.37e-3`，与 Triton-B1 链
  **同阶**（`dq`/`dh0` 两链逐位相同），全部落 bf16 底噪。至此「目标卡 CUDA 反向」第一次**端到端
  可跑**（B1 是 CUDA、其余 Triton），不再只是孤立 kernel。
- **链级时间**：CUDA-B1 链在 T 可整除的 shape 上 **0.94–1.03×** Triton-B1 链（大 shape 快 6%），
  即换 B1 不拖慢整链；链的瓶颈仍是 **B2/B3**（§26），与「先接受 Triton、CUDA 版逐件替换」的路线一致。
- **单核头对头**：部署 `BT=16` 下 CUDA B1 小 shape 与 Triton 同档/略慢、大 shape 快 ~1.9×——
  「手写 CUDA 是否比 Triton 快」**依赖 shape**，不是无条件。
- **坑（新）**：(a) 早期把 CUDA B1 的 bf16 `dh`/`dv2` 整体 `.float()` 喂 B2，等于凭空加 elementwise
  税（`B8T2048H8` 上链 wall 从 7047 变 8600），**应让 B2 直接吃 bf16 `dh`**，只有 B2 里进 `tl.dot`
  的 `dv` 需 fp32；(b) `launch_impl` **每次 launch 重编码张量图**，小 shape 链是 host-bound，表现为
  `T=250` 的 1.65× 回归（**待验证**：张量图一次编码复用后消除）；(c) 孤立 µs 级 kernel 的 CUPTI 归因
  在本机双峰（~41 vs ~130µs），该场景一律用 event-pair wall。

## 43. 关闭 §42 的 host 税待验证：张量图「一次编码复用」省不下那 1.65×——它是**不可复现**的（P7-40）

§42.4/§42.6 把 CUDA-B1 链在 `T=250 H=3` 上的 **1.65×** 链级回归归因到「`launch_impl` 每次 launch
都重新 `cuTensorMapEncodeTiled`（5–8 张图）+ `cudaFuncSetAttribute` 检查，小 shape 链是 host-bound」，
并把它标为**待验证**（预期「一次性编码复用后消除」）。本轮（`harness/check_bwd_b1_launch_tax.py`
+ 输出 `harness/check_bwd_b1_launch_tax.out.txt`）把这个假设**直接测掉**，结论是：**张量图编码不是那
笔 host 税，而 1.65× 本身在空闲卡上根本复现不出来。**

### 43.1 加一个「一次编码复用」的 `B1T_MAPCACHE`（并证明它真的命中）

把 `launch_impl` 里的 5–8 次 `make_map*` 收成一个 `MapSet` + `encode_maps`（`b1_tma_probe.cu:661-687`），
再给 `-DB1T_MAPCACHE`：只要**所有操作数指针和 shape 都没变**就复用上一次的 `MapSet`，否则整组重编码
（`b1_tma_probe.cu:700-714`）。指针/shape 全进 key，故「同 shape 换指针」也会重编码，不会串味。
外加两个仅供探测的符号：`b1t_encode_only(...)`（只做编码、不 launch，`b1_tma_probe.cu:788`）与
`b1t_mapcache_miss()`（未命中计数，`:796`）。命中计数实测：连续 200 次 launch，miss 只 +1
（`harness/check_bwd_b1_launch_tax.out.txt`）⇒ **200 次里 199 次命中**，缓存确实生效。数值上
`B1T_MAPCACHE` 是**纯 host 端复用**：Part B 在非零 `dht` 下，cache vs no-cache 的
`dh/dh0/dv2` **逐位相同（max_abs=0，含非整除 T=250/513）**。

### 43.2 Part A：纯编码 ≈3.9µs，但它在 launch 里几乎全被隐藏

`process_time / call`（P7-6 口径，单线程 CPU）：

| shape | launch（no-cache） | launch（cache） | 省下 | `b1t_encode_only`（纯编码） |
|---|---|---|---|---|
| tiny B1 T16 H1 | 7.72 µs | 6.99 µs | **0.73** | 3.86 µs |
| T250 H3 | 18.68 µs | 18.45 µs | **0.23** | 3.86 µs |
| B2 T1024 H8 | 62.78 µs | 62.41 µs | **0.37** | 3.87 µs |

纯编码 **3.8–4.0µs/次**（8 张图，即每张 `cuTensorMapEncodeTiled` ≈0.5µs）——**但这 4µs 加进 launch
里只看得见 ≤0.75µs**。原因是这段 `process_time` 循环已经被 **launch 队列反压**主导：`B2 T1024 H8`
的 launch 读数 62.7µs 远大于其 kernel dev（§42 的 78µs @BT16 NS4 同量级，小 shape kernel 只有 10–20µs），
即循环速率被 GPU/驱动吞吐卡住，CPU 侧的编码被藏在队列等待里。**所以**：`T=250` 的 host 税若真是
编码，缓存应省 ~4µs/launch，实测只省 0.23µs——**不成立**。（顺带一个方法学点：用 `process_time`
测「host 税」时，如果 launch 队列被打满，读数≈GPU 吞吐速率、不是纯 CPU；要拆纯 CPU 得用
`b1t_encode_only` 这类**不发 launch** 的入口。）

### 43.3 Part C：1.65× 不可复现，空闲卡上 CUDA/Triton 链 wall 只有 0.95–1.05×

同一空闲卡（`CUDA_VISIBLE_DEVICES=1`）上重跑 §42 的链级口径（event-pair wall，3 轮取中位；
`harness/check_bwd_b1_launch_tax.out.txt`）：

| case | Triton-B1 链 wall | CUDA-B1（no-cache） | CUDA-B1（cache） | **CUDA/Triton** | cache/no-cache |
|---|---|---|---|---|---|
| T=512 H=4 | 312.6 µs | 335.9 µs | 327.1 µs | **1.046×** | 0.974 |
| B8 T=2048 H=8 | 2251.1 µs | 2146.3 µs | 2146.4 µs | **0.954×** | 1.000 |
| T=250 H=3 (non-div) | 313.2 µs | 327.6 µs | 328.5 µs | **1.049×** | 1.003 |

`T=250 H=3` 现在是 **1.05×**，不是 §42.4 记的 1.65×。而且 §42.4 的 Triton baseline（245.9µs）本身在
不同轮次漂得很大：本轮脚本内多次测到 **308 / 323 / 386µs**（同一 `make_chain_fn_pipe NS=3`），
§42.4 的 245.9µs 是一个**低离群**——1.65× 基本就是「Triton 基线抽到低值」的产物，非稳定回归。
`B1T_MAPCACHE` 对链 wall 的影响在噪声内（cache/no-cache 0.997–1.001），与 §43.2 一致。

### 43.4 结论 / 对目标卡的设计输入

- **撤回 §42.4 的 host 税归因**：`T=250 H=3` 的 1.65× **在空闲卡上不可复现**（实测 1.05×），且其
  Triton 基线自身跨轮次 245→386µs 波动（±25%），该 case 的「回归」是**基线采样的低离群**，不是 CUDA
  B1 的稳定慢。
- **张量图编码不是瓶颈**：纯编码 3.8–4.0µs/launch，加进 launch 后（队列反压主导）只看得见 ≤0.75µs；
  「一次编码复用」（`-DB1T_MAPCACHE`）**数值逐位无损、命中率 199/200**，但对链 wall 无可见收益。
  故 §42.6 坑 (b) 的「待验证」**关闭**：目标卡落地做一次编码复用是**合理的整洁做法**（免费、逐位、
  生产固定 workspace 时确实命中），但**它不解决任何已测到的性能问题**，不必再把它当优化项。
- **诚实的剩余差距**：CUDA-B1 链 vs Triton-B1 链 **0.95–1.05×**（大 shape 快 5%、小 shape 慢 5%），
  与 §42 的「换 B1 不拖慢整链、瓶颈在 B2/B3（§26）」一致；小 shape 的 ~5% 是 BT=16 下 CUDA 核本身
  比 Triton 慢 ~1.2×（§42.4 kernel 归因，非 host），与 §37.3「BT=16 粒度差距」自洽。
- **方法学（新）**：测「host 税」时，`process_time` 在 launch 队列被打满时读的是 **GPU 吞吐速率**、
  不是纯 CPU；要隔离纯 CPU 须用**不发 launch** 的入口（本轮的 `b1t_encode_only`）。这与 §25.5
  （归因窗口）、§37 dtype 对齐并列，进入本专题的计时口径清单。


## 44. 兑现 §24.5 的「换 CUTLASS `PipelineTmaAsync`」——一个**空结论**：手写 mbarrier 已在结构上限（P7-41）

§23–§41 用**手写** per-stage `mbarrier` 握手把 K2 形状的 CUDA B1 推到 `xFLA` ~1.7–3.8×，§24.5 把残差归到
「每 warp 发射效率」（`wait` = mma 固定延迟 ≈36% 发射间隔、`short_sb` = smem 取操作数），并留了两个选项：
**(a) 换 CUTLASS `PipelineTmaAsync`**（把 producer/consumer 与 `cp.async.bulk` 完成延迟交给成熟实现），或
**(b) 接受 Triton 版**。§36 又把 (a) 记成「唯一剩下的更大工程」。本轮把 (a) 做成**最小受控实验**：
给 `b1_tma_probe.cu` 加 `-DB1T_CUTLASS`，**只**替换 pipeline 的管理（producer 的 `empty` 等待 + `full`
`arrive_and_expect_tx`、consumer 的 `full` 等待 + `empty` arrive），TMA 拷贝（裸 PTX `cp.async.bulk.tensor`）、
`Swizzle<3,4,3>` 操作数、`mma.sync` 计算、`dh`/`dv2` 的 TMA store、`fence.proxy.async` **一字未改**——故任何
delta 只能归因于 pipeline 协议本身。

**实现**（`harness/b1_tma_probe.cu`，`#ifdef B1T_CUTLASS` 分支）：

- `Shared` 里用 `cutlass::PipelineTmaAsync<NSTAGE>::SharedStorage pipe;` 取代裸 `full[]`/`empty[]`
  （`b1_tma_probe.cu:243-244`）。
- 构造：`PipeT pipe(sh->pipe, pp, cute::Shape<_1,_1,_1>{});`，`pp.transaction_bytes` = 8 个 operand tile +
  `E^tot`（`KD*4`，同 `B1T_ETMA` 合并路径）、`pp.is_leader = (warp==NW && l==0)`、`pp.num_consumers = NW*32`
  （`b1_tma_probe.cu:285-300`）。CUTLASS 在 `initializing_warp==0`（=一个 compute warp）里 init barrier 对，
  再 `fence_barrier_init`，与手写版语义一一对应。
- producer：`auto pstate = cutlass::make_producer_start_state<PipeT>();`（起始 phase=1，令前 `NSTAGE` 次
  `empty` 等待**立即通过**）；循环里 `pipe.producer_acquire(pstate)`（内含一次性 `arrive_and_expect_tx`）→
  `pipe.producer_get_barrier(pstate)` 拿到 TMA 的 `mbarrier` 指针 → 原样发 8+1 条 TMA → `producer_commit`
  （TMA 路径下是 NOP）→ `++pstate`（`b1_tma_probe.cu:318-371`）。
- consumer：`PipeT::PipelineState cstate;`，循环里 `pipe.consumer_wait(cstate)` 取代
  `mbar_wait(&full[s], (i/NSTAGE)&1)`；块末 `pipe.consumer_release(cstate); ++cstate;` 取代
  `if(l==0) mbar_arrive(&empty[s])`——CUTLASS 的 `empty` 到达数 = `num_consumers = NW*32`，故**每线程都 arrive**
  （`b1_tma_probe.cu:415-428`、`:614-615`）。

### 44.1 Part A — 正确性：CUTLASS 是**逐位** drop-in

`harness/check_bwd_b1_cutlass.py`（`harness/check_bwd_b1_cutlass.out.txt`）四个 build 全部 `-DB1T_ETMA`：

| 变体 | pipeline | store |
|---|---|---|
| `base` | 手写 mbarrier | 标量 `dh`/`dv2` |
| `hnd` | 手写 mbarrier | TMA `dh`/`dv2` + `EPROXY`+`SPROXY`（= §40 `tma2_full`） |
| `cut` | CUTLASS `PipelineTmaAsync` | 标量 |
| `cut2` | CUTLASS `PipelineTmaAsync` | TMA + 两侧 fence |

16 格（4 shape × `BT∈{32,64}` × `NS∈{1,2}`，含非整除 `T=250`）实测：**`cut` vs `base` 与 `cut2` vs `hnd` 的
`dh`/`dv2`/`dh0` 全部 `max_abs = 0.00e+00`**；`cut2` vs 直接 bf16 FLA `dhu` 仍 **1 bf16 ulp**（`dh 4.88e-4`、
`dv2 4.88e-4`/`2.44e-4`）。即换 pipeline 库**不改一个 bit**（复现 §35/§36 的逐位结论）。

### 44.2 Part B — 性能：**空结论**，`cut2/hnd = 0.985–1.003`

CUPTI 内核时（µs/call，空闲卡），`FLAbf16` = 直接调用 bf16 FLA `dhu`（matched dtype，§37 口径）：

| shape | BT | NS | `base` | `hnd` | `cut` | `cut2` | FLA bf16 | `cut2/hnd` | `cut2/FLA` |
|---|---|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 32 | 1 | 66.3 | 40.3 | 63.6 | 39.7 | 20.5 | **0.987** | 1.94 |
| B1 T512 H4 | 64 | 1 | 44.0 | 30.6 | 44.5 | 30.6 | 13.0 | **0.999** | 2.36 |
| B1 T512 H4 | 64 | 2 | 42.7 | 25.9 | 42.6 | 25.6 | 13.0 | **0.992** | 1.98 |
| B2 T1024 H8 | 32 | 2 | 128.0 | 65.0 | 130.8 | 65.1 | 37.6 | **1.003** | 1.73 |
| B2 T1024 H8 | 64 | 2 | 84.1 | 48.2 | 82.8 | 47.4 | 23.3 | **0.985** | 2.03 |
| B8 T2048 H8 | 32 | 2 | 341.2 | 138.1 | 351.0 | 138.2 | 170.5 | **1.001** | 0.81 |
| B8 T2048 H8 | 64 | 1 | 194.6 | 138.7 | 196.0 | 138.7 | 119.1 | **1.000** | 1.16 |
| B8 T2048 H8 | 64 | 2 | 204.7 | 99.3 | 201.2 | 99.1 | 119.1 | **0.998** | 0.83 |
| B1 T250 H3 | 64 | 2 | 24.2 | 14.9 | 24.2 | 14.6 | 7.3 | **0.985** | 1.99 |

全部 16 格 `cut2/hnd ∈ [0.985, 1.003]`（均值 ~0.995，噪声内）；标量对照 `cut/base` 同样噪声内。即把
**手写 mbarrier 换成 CUTLASS `PipelineTmaAsync` 一点没变快**——§24.5 的残差（`wait`/`short_sb`）**不是
pipeline 协议的实现开销**。`cut2/FLA` 仍 0.81–2.72×（与 §38 的 `tma2` 逐值一致），BT=64 大 shape 追平、
小 shape 2× 残差仍是 §24.5/§37.3 的「每 warp 发射效率 + BT 粒度」。

### 44.3 Part C — CUTLASS 顺带是**确定性**的（手写 parity 曾是 bug 源）

同一 kernel 连跑两次比 `dh`（`max_abs`），`BT∈{16,32,64}` × `NS∈{1,2,3}`：

- **`base`（未加 fence 的手写 pipeline）** 在 `B8 T2048 H8` 上不稳：`BT=16 NS=3` **3.76e-2**、
  `BT=32 NS=3` **1.57e-3**（`NS=1/2` 免疫）——复现 §38/§39 的「手写 parity 在 `NS≥2` 的 WAR 竞态」。
- **`hnd`（手写 + `EPROXY`/`SPROXY`）**、**`cut2`（CUTLASS）** 在**全部 18 格** run-to-run `0.00e+00`，
  且 **`cut2` vs `hnd` 逐位相同**；`cut2-base` 只在 `base` 自身 racy 的格上非零（1.57e-3）。

即 CUTLASS 的 phase/parity 记账**天然**避开了手写版那处 off-by-one（§36 靠一条 `fence.proxy.async` 修掉）。
所以换 CUTLASS 的收益是**稳健性/可维护性**（省掉手写 parity 与 fence 的心智负担），**不是性能**。

### 44.4 Part D — `ncu`：两支逐指标几乎重合

`harness/probe_b1_cutlass_ncu.sh` / `ncu_b1_cutlass_driver.py`（`harness/probe_b1_cutlass_ncu.out.txt`），
`BT=64 NS=2`：

| metric | `hnd` (B8T2048H8) | `cut2` | `hnd` (B1T512H4) | `cut2` |
|---|---|---|---|---|
| Duration (µs) | 99.04 | **98.82** | 27.78 | **27.68** |
| `wait` (inst/issue) | 1.42 | **1.38** | 1.44 | **1.39** |
| `short_scoreboard` | 0.61 | 0.63 | 0.60 | 0.60 |
| `long_scoreboard` | 0.96 | 1.00 | 0.77 | 0.77 |
| `tensor` (% peak) | 22.29 | 22.37 | 20.77 | 21.04 |
| bank-conflict (`ld`) | 445448 | 476930 | 6911 | 7407 |
| regs/thread | 165 | 164 | 165 | 164 |
| dyn smem (KB) | 220.16 | 220.16 | 220.16 | 220.16 |
| achieved occ (%) | 7.77 | 7.79 | 7.40 | 7.34 |

逐指标在噪声内重合：`wait ≈1.4` **纹丝不动**（§24.5 已证它是 mma 固定流水延迟、与协议无关）、
`short_sb≈0.6`、`tensor≈22%`、`regs` 165→164、smem 同、占用率同。这从硬件计数上钉死「换 pipeline 库不动
残差」。

### 44.5 结论 / 对目标卡的设计输入

- **§24.5 的选项 (a) 是空杠杆**：CUTLASS `PipelineTmaAsync` 相对手写 mbarrier **逐位相同、性能噪声内
  （0.985–1.003×）**。K2 形状 CUDA B1 的残差（`wait`/`short_sb`，≈2× vs FLA bf16）**不是** pipeline 协议
  能修的东西——手写 mbarrier 已经触及该结构的**上限**。§23.6/§24.5/§36 反复记的「换 CUTLASS 是唯一剩下的
  更大工程」至此**关闭为空结论**。
- **故 §24.5 的选项 (b)（直接接受 Triton 版）是正确路线**：Triton 版 `tl.range` 软件流水（§17/§25）在链级
  已把 B1 追平/反超，而目标卡 CUDA 版停在 1.7–2.7×（小 shape）——省下的工程应投到**链级瓶颈 B2/B3**
  （§26–§33 已收敛）与**部署粒度**，而非继续打磨 B1 的流水协议。
- **CUTLASS 仍值得作为「实现基座」**：它天然确定（§44.3）、代码更短、省掉手写 parity/fence；若生产要长期
  维护一支 CUDA B1，用 CUTLASS pipeline 是**更稳的工程选择**，但**不带来性能**。
- **方法学**：把「换成熟实现」当优化假设时，先做**只替换该机制的受控实验**（本轮保留 TMA/计算/store/fence
  一字不改）——否则会把 TMA/store/计算侧的既有收益误记到 pipeline 头上。

## 45. 把两条**分别收敛**的反向优化合成一条链：CUDA B1 × 融合 B2/B3（P7-42）

§15–§41 把 **K2 形状的手写 CUDA B1** 打磨到收敛，§42（P7-39）第一次把它落进组装链——但那条链用的仍是
**未融合**的 B2（`wy_dqkg`）+ B3（`intra`）。而 §31–§33 把 **B2/B3 融合**收敛后，其链（§33 `make_chain_fused`）
用的仍是 **Triton** 的 pipe B1。**两条最新收获从未同时出现在一条链里**：一条最新链是「CUDA B1 + 旧 B2/B3」，
另一条是「Triton B1 + 新融合 B2/B3」。本轮把两者拼成**完全收敛的反向链**

    dAv -> CUDA B1 (dht, NSTAGE) -> fused B2+B3 -> reverse cumsum(B0)

仍只吃 FlashKDA 自己的 16-chunk K1 workspace，并把它与**两条半收敛链**及 FLA autograd 同口径对拍。

**实现**（`harness/check_bwd_cudab1_fused_e2e.py`）：把 §42 的 `make_chain_cuda`（CUDA B1 + 未融合
B2/B3）与 §33 的 `make_chain_fused`（Triton pipe B1 + 融合 B2/B3）各取一半拼起来——
`dav_triton` -> `run_cuda_b1`（deploy 配置 `-DB1T_ETMA -DB1T_TMADH -DB1T_TMADV -DB1T_EPROXY -DB1T_SPROXY`，
native bf16 `dh`/`dv2` 出口）-> `fused_b2b3_triton`（`num_warps=8`）-> `chunk_local_cumsum(reverse=True)`。
B1 吃 bf16 `qg/kg/w`、bf16 `do`，融合 B2/B3 吃 fp32 `q/k/v/g/beta` 与 bf16 `dh` + fp32 `dv2`（与 §42/§33
逐字一致）。三个变体：

| 记号 | B1 | B2/B3 |
|---|---|---|
| `cu` | **CUDA** K2 形状（NSTAGE=4） | 未融合（`wy_dqkg` + `intra`，= §42 链） |
| `fu` | **Triton** `dhu_bf16_pipe`（NS=3） | **融合**（= §33 链） |
| `cf` | **CUDA** K2 形状（NSTAGE=1..4） | **融合**（本轮） |

### 45.1 Part A — 正确性：合成链落 tf32/bf16 底噪，且继承了两个半链各自的位级性质

6 组 case（含非整除 `T=250/513`、`g=0`、`B=2`），`harness/check_bwd_cudab1_fused_e2e.out.txt`。六项梯度
worst-case `max_rel`：

| 梯度 | `cf` vs FLA | `cu` vs FLA | `fu` vs FLA | `cf` vs `cu` | `cf` vs `fu` |
|---|---|---|---|---|---|
| `dq` | 3.417e-3 | 3.423e-3 | 3.417e-3 | 1.424e-3 | **0.000e+00** |
| `dk` | 3.301e-3 | 3.308e-3 | 3.016e-3 | 1.093e-3 | 1.900e-3 |
| `dv` | 6.795e-3 | 6.795e-3 | 6.795e-3 | **0.000e+00** | 2.338e-3 |
| `dbeta` | 6.012e-3 | 6.012e-3 | 6.012e-3 | 3.007e-5 | 1.609e-3 |
| `dg` | 7.054e-3 | 7.064e-3 | 7.054e-3 | 6.078e-3 | 1.089e-3 |
| `dh0` | 9.365e-3 | 9.365e-3 | 8.267e-3 | **0.000e+00** | **0.000e+00** |

三个变体 vs FLA 全落 **tf32/bf16 底噪**，且 `cf` 的六个 worst-case 值与 §25.2/§26.3/§42.3 **逐值一致**
（`dq 3.42e-3 … dh0 9.37e-3`）——合成不改数值底噪。位级签名也自洽：

- `cf` vs `fu` 的 **`dq`、`dh0` 逐位相同**：两者共用同一份融合 B2/B3 与同一份上游 `dht`，差别只在 B1
  互换；`dq` 对 B1 的 bf16 出口不敏感（`dh`/`dv2` 的 1 ulp 不改变 `dq` 主项），`dh0` 则直接来自 B1 终态。
- `cf` vs `cu` 的 **`dv`、`dh0` 逐位相同**：两者共用同一个 CUDA B1（`dv2`/`dh0` 由它定），差别只在
  B2/B3 是否融合；融合只改 `dq/dk/dg/dbeta` 的 fp32 加法次序（≤1.4e-3）。
- 即「`dv`/`dh0` 由 B1 决定、`dq/dk/dg/dbeta` 由 B2/B3 形式决定」的归因在合成链上被逐位证实。

### 45.2 Part B — 链级计时：两条优化**可加**，但 B1 的选择**依 shape 交叉**

CUPTI `dev`（µs，空闲卡；`cf1..cf4` = NSTAGE∈{1,2,3,4}）与 event-pair `wall`（µs）：

| case | FLA dev | `cu` dev | `fu` dev | `cf1` | `cf2` | `cf3` | `cf4` | `cu` w | `fu` w | `cf4` w | `cf/cu` | `cf/fu` | `cf/FLA` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T=512 H=4 | 203.7 | 104.7 | 67.6 | 99.8 | 73.7 | 74.6 | 73.8 | 405.1 | 270.1 | 283.8 | 0.70 | 1.09 | 0.36 |
| B2 T=1024 H=8 | 659.8 | 359.0 | 243.3 | 318.2 | 258.4 | 259.9 | 256.4 | 414.6 | 274.4 | 289.3 | 0.71 | 1.05 | 0.39 |
| B8 T=2048 H=8 | 4137.5 | 2107.5 | 1537.3 | 1517.7 | 1420.8 | 1401.8 | 1401.6 | 2132.8 | 1559.9 | 1412.7 | 0.67 | 0.91 | 0.34 |
| T=250 H=3 nondiv | 142.3 | 75.2 | 43.7 | 60.5 | 50.7 | 51.0 | 50.6 | 407.2 | 266.9 | 285.1 | 0.67 | 1.16 | 0.36 |

其中 `cf/cu`、`cf/fu` 用最优 NSTAGE（3/4 打平）。B1 单核 `dev`（`b1_tma_kernel` vs Triton `_dhu_kernel`）：

| case | Triton B1 (`fu`) | CUDA B1 (`cf4`) |
|---|---|---|
| T=512 H=4 | 34.9 | 40.8 |
| B2 T=1024 H=8 | 61.7 | 77.9 |
| B8 T=2048 H=8 | 317.9 | **184.2** |
| T=250 H=3 | 19.0 | 22.8 |

三条结论：

1. **两条优化可加，不是零和**。`cf` 在**所有** shape 上都优于 `cu`（`cf/cu = **0.67–0.71×**`）——把
   CUDA B1 链的未融合 B2/B3 换成 §33 的融合版，dev 直接再砍 ~30%。这与 §30 两次失败的「朴素 B1→B2
   融合」不同：那是把整条 `[K,V]` 塞进一个 Triton program；这里融合是**正交于 B1 的独立收获**，在
   CUDA B1 之上一分不损地叠加。
2. **但 B1 的实现选择随 shape 交叉**，故 `cf` 相对 `fu` 只在**大 shape** 取胜（`B8 T2048 H8` `cf/fu = 0.91×`，
   dev 1537→1402、wall 1560→1413），小 shape 反而 **1.05–1.16× 略慢**。根因就在 B1 单核头对头：CUDA B1
   在部署粒度 `BT=16` 的小 grid 上比 Triton pipe **慢**（40.8 vs 34.9、77.9 vs 61.7、22.8 vs 19.0），
   在 `B8 T2048 H8` 才**大幅反超**（184.2 vs 317.9 = 1.73×）。这与 §37.3/§42.5/§40 的「CUDA vs Triton
   依赖 shape、小 shape 残差=每 warp 发射效率 + BT 粒度」逐值自洽。
3. **NSTAGE≥3**：`cf3`/`cf4` 打平且最优，`cf1` 明显最慢（T512H4 99.8 vs 73.8）；复现 §41.4 的
   「TMA store + BT=16 上深 pipeline 有正收益」。

据此「完全收敛链」的**部署最优**是 per-shape 选 B1：小 grid 用 Triton pipe B1、大 grid 用 K2 形状 CUDA
B1，**B2/B3 一律用融合版**。两条路线的 `dev/FLA` 都只有 **0.31–0.37×**（即整条反向比 FLA 反向 dev
快 **~2.7–3.2×**，且不含 FLA 那笔前向状态重算）。

### 45.3 结论 / 对目标卡的设计输入

- **合成成功且可加**：`cf` vs `fu` 的 `dq`/`dh0` 逐位、vs FLA 六项 worst-case 与半链逐值一致——两条
  优化（K2 形状 CUDA B1、B2/B3 融合）在**同一份 workspace 操作数**上互不干扰。§28 的「链外 elementwise
  税」、§31–§33 的融合、§35–§41 的 B1 TMA store 三条线至此**汇到一条链**。
- **目标卡 CUDA 反向的最终形态**＝`dAv -> B1 -> fused B2/B3 -> 反向 cumsum`；**B1 实现按 grid 大小切换**
  （小 grid 走 Triton pipe / 大 grid 走 K2 形状 CUDA），**B2/B3 走融合**。`cf` 相对 `cu` 的 0.67–0.71× 说明
  在已经上了 CUDA B1 的前提下，**B2/B3 融合仍是必须先落的**（收益比继续打磨 B1 大且零风险）。
- **不要再纠结 B1 的绝对胜负**：单核 CUDA B1 与 Triton pipe 是形状交叉的（§37/§42/§45.2），合成后整链
  0.31–0.37× FLA——继续压 B1 的边际已被 §44（换 CUTLASS=空）钉死；剩余投入应去做**B2/B3 的 CUDA 化**
  （把它们从 Triton 融合核搬成 K2 形状的 CUDA 核），那才是整链剩余的大头。
- **方法学**：当有多条**分别收敛**的优化时，**必须做合成实验**——本轮证明它们可加（§45.2 结论 1），
  也暴露了「最优 B1 随 shape 切换」这一单链实验看不到的部署约束（结论 2）。

## 46. 融合 B2/B3 的「K2 形状 CUDA 化」可行性研判：它到底 bound 在哪、能兑现多少（P7-43）

§45.3 把目标卡 CUDA 反向的**最终形态**定为 `dAv -> B1 -> fused B2/B3 -> 反向 cumsum`，并点名剩余大头是
**把融合 B2/B3 从 Triton 搬成 K2 形状 CUDA**。但在动一个 900+ 行的 CUDA 核之前，必须先把这个问题
**量清楚**：融合核现在到底 bound 在什么上？「K2 形状」能兑现多少？——因为本专题里「ncu 报 latency /
估 44% 加速」这条启发式**已经错过两次**（K2 的 §8.4「grid 太小」、B1 的 §16「占用率」），而**对**的那次
（B1 的 §22）恰好就是 K2 形状本身把 2.3% 的 tensor 密度拉到 14%、兑现 2.4×。所以本轮**不建 CUDA 核**，
用三个便宜实验给「K2 形状 CUDA 化」一个 go/no-go 和收益上界。

证据：`harness/check_bwd_b2b3_bound.py`（`harness/check_bwd_b2b3_bound.out.txt`，空闲卡）+ 调度器段
`harness/probe_b2b3_bound_ncu.sh`（`harness/probe_b2b3_bound_ncu.out.txt`）。

### 46.1 Part A — deploy 基线与被钉死的静态代价

部署配置 `B8 T2048 H8`、`BT=16`、`nw=8`、`BV=128`（= P7-30 的 `fused_b2b3_triton` 默认）、操作数
dtype 与真实链一致（`dh` 为 CUDA B1 的 bf16 出口、其余 fp32）：

| shape | `dev` | `wall` | regs | spills | smem | 解析占用率 |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` (grid 128×64) | 1002.7 µs | 992.6 µs | **255** | 34 | **114.0 KB** | 1 block/SM，8 warps/SM，12.5% |
| `T512 H4` (grid 32×4) | 12.2 µs | 45.8 µs | 255 | 34 | 114.0 KB | 1 block/SM，12.5% |

`B8 T2048 H8` 的 `dev≈wall`（无 host 税），1003 µs 与 §30/§45 的链内归因（融合核 ~1050 µs）一致。注意
占用率是**寄存器（255）与 smem（114 KB）同时**把它钉在 1 block/SM——这一点很关键，见 Part B。

### 46.2 Part B — 「降寄存器」不是杠杆：只会加 spill，占用率一动不动

用 Triton 的 `maxnreg` 强制寄存器上限（`launch_fused`，本脚本内；`B8 T2048 H8`）：

| `maxnreg` | `dev` | regs | spills | smem | 占用率 | vs base |
|---|---|---|---|---|---|---|
| 无（base） | 986.8 µs | 255 | 34 | 114.0 KB | 1 blk/SM 12.5% | 1.00× |
| 224 | 1096.7 | 224 | 58 | 114.0 KB | 1 blk/SM 12.5% | 1.11× |
| 192 | 1234.6 | 192 | 92 | 114.0 KB | 1 blk/SM 12.5% | 1.25× |
| 160 | 1503.6 | 160 | 136 | 114.0 KB | 1 blk/SM 12.5% | 1.52× |
| 128 | 1845.7 | 128 | 186 | 114.0 KB | 1 blk/SM 12.5% | 1.87× |
| 96 | 2455.7 | 96 | 264 | 114.0 KB | 1 blk/SM 12.5% | 2.49× |
| 80 | 3007.0 | 80 | 320 | 114.0 KB | 1 blk/SM 12.5% | 3.05× |

结论很干净：**砍寄存器把占用率从 1 block/SM 挪不动**（smem 114 KB 卡死，`228 KB/114 KB = 2` 但两块
需 2×(114+1) KB > 228 KB，故仍 1），却把 255→80 的每一档都换算成**线性增长的 spill**（34→320）与**变慢**
（最坏 3.05×）。即：融合核的寄存器压力是**真代价、不是可回收的占用率余量**——「先把寄存器降下来腾
占用率」这条路在 Triton 里走不通。

### 46.3 Part C — 同字节、无 `tl.dot` 的「内存地板」：融合核比它高 **1.94×**

关键受控实验：写一个**读写指针/掩码/字节与融合核逐一相同**、但**没有 `tl.dot`**（每个操作数只加载一次、
折进一个 store）的 `_traffic_kernel`。它 0 spill、0 smem、3 block/SM（高占用），因此是「**若延迟被完美
隐藏，同等 DRAM 流量能跑到多快**」的地板：

| shape | 融合核 `dev` | traffic 核 `dev`（nw=4 / nw=8） | 占用率 | fused/traffic |
|---|---|---|---|---|
| `B8 T2048 H8` | 988.4 µs | **510.7 / 508.8 µs** | 3 blk/SM，18.8 / 37.5% | **1.94×** |
| `T512 H4` | 12.2 µs | 5.1 / **5.0 µs** | 3 blk/SM | **2.43×** |

traffic 核的 `dev` 与 `nw` 无关（510 vs 509）——它已达**带宽饱和**（1.5 GB/509 µs ≈ **3.0 TB/s**，H100
HBM3 峰值的 ~90%），而融合核只跑到 DRAM 55.8% SOL（1.87 TB/s）。两条独立口径互证地板：
基于 §30 ncu 字节数的 `(1.58 GB 读 + 0.38 GB 写)/3.35 TB/s` = **586 µs**（融合核 **1.79×** 高于它），
与 traffic 核直接实测的 509 µs（**1.94×**）同量级。**即融合核有约 2× 的「延迟隐藏」头寸**。

诚实边界：traffic 核不做 dot、不落 smem staging，故它把「smem staging + 点积」那部分也省了；但融合核
的 tensor 密度只有 **6.2%**、SM 15.7%（Part 46.4），计算不是约束，故 509 µs 作为「内存延迟地板」是
合理代理，不是严格下界。

### 46.4 ncu — 融合核是**发射饥饿（eligible-warp starved）**，且被 regs+smem **同时**钉在 1 block/SM

`probe_b2b3_bound_ncu.sh`（`B8 T2048 H8`，deploy）调度器/占用率段：

| 指标 | 值 | 含义 |
|---|---|---|
| Active Warps / Scheduler | 1.99 | 每调度器只挂 ~2 个 warp（上限 16） |
| **Eligible Warps / Scheduler** | **0.18** | 14.28% 的周期「一个可发射 warp 都没有」 |
| Issued Warp / Scheduler | 0.14 /cyc | 每 **7 个周期**才发一条指令 |
| Warp Cycles / Issued Instruction | 13.92 | 其中 `long_scoreboard` **5.80（41.7%）** |
| Block Limit Registers / Shared Mem | **1 / 1** | 两者**都**限制到 1 block/SM |
| Theoretical / Achieved Occupancy | 12.50 / 12.48% | 8 warps/SM，远未填满 |
| DRAM / L2 / tensor / SM SOL | 55.8 / 62.4 / **6.2** / 15.7% | 内存未饱和、张量几乎闲置 |

这与 §45 之前对未融合 B2 的 `long_sb 6.61`（P7-23 §26）一致，但更极端：**85.7% 的周期无可发射 warp**，
瓶颈是把 6 个操作数的加载延迟暴露在唯一的 straight-line 体里（`i_v` 循环在 `BV=V=128` 下只有 1 次迭代、
**无循环可流水**），而占用率又被 regs+smem 锁在 1 block/SM。ncu 自报 `Est. Speedup ≈ 44%`（对齐
Part B 的「降寄存器反而变慢」——**那 44% 不是靠降寄存器能拿到的**）。

### 46.5 结论 / 对目标卡的设计输入

- **融合 B2/B3 是延迟/发射受限、不是带宽受限，且有 ~2× 头寸**：DRAM 只有 55.8% SOL、tensor 6.2%，
  而同字节、无 dot 的 traffic 核跑到 ~3.0 TB/s（带宽饱和）只用 **509 µs** → 融合核（988 µs）**1.94×**
  高于该地板；ncu 的 eligible warps/scheduler **0.18**、issue 0.14/cyc 直接坐实「发射饥饿」。
- **该头寸在 Triton 里够不着**：占用率被 **255 寄存器 + 114 KB smem 同时**钉在 1 block/SM；`maxnreg`
  从 255 降到 80 只会线性加 spill（34→320）并变慢到 **3.05×**。融合核的 straight-line 体（`BV=V` 无
  可流水循环）也不给软件流水的机会（`num_stages` 对无循环体无效，同 §26.2）。
- **因此「K2 形状 CUDA 化」是被数据支持的方向**（与 §45.3 点名一致）：唯一能同时(a)把状态/操作数搬到
  **TMA 异步流水**、(b)靠**warp specialization**把加载延迟与计算重叠、(c)让每 warp 的有效张量密度从
  6.2% 抬起来的手段，就是 B1 已走过的 K2 形状（寄存器分片 + TMA + 专职 producer，§19–§24）。收益上界
  由本轮给出：**~1.9×**（内存地板口径）/ **~1.8×**（ncu 字节口径），与 B1 当年的 2.3–2.7× 同量级。
- **但要诚实**：2× 是**上界**——K2 形状要**同时**做对「TMA 搬 `h/dh/q/k/...` + 把 6 个 GEMM 的累加器
  按 warp 分片常驻 + intra 对角折进同一 CTA」。B1 的经验（§23–§24）是 TMA+warp-spec 本身只值 ~15–20%、
  真正的大头来自**寄存器常驻状态 + `ldmatrix`/`mma.sync`**（§22 的 2.4×）。对融合 B2/B3，首要动作应是
  **把 `dq/dk/dw [BT,K]` 与 `dA [BT,BT]` 累加器分片到多 warp 的寄存器**（而非全塞一个 program），
  其次才是 TMA 输入流水。
- **不要先做的**：不要在 Triton 里继续扫 `maxnreg`/`nw`/`BV` 找占用率（本轮证伪）、也不要指望
  `num_stages`（无可流水循环体）。这些免费旋钮在 §26/§30 已扫过，本轮把「占用率」这条也钉死。
- **方法学（新增）**：判断「某核该不该做 K2 形状重写」时，最便宜的判据是**写一个同字节、去 compute 的
  traffic 等价核**测地板——它一次性给出(a)是否延迟受限（地板远低于实核）、(b)收益上界（实核/地板比）、
  (c)能否靠占用率拿到（配 `maxnreg` 受控扫）。这比直接 ncu「Est. Speedup」启发式可靠（后者在 K2/B1
  两次误报、且本轮给出的 44% 明确**不可**由降寄存器兑现）。

## 47. 「K2 形状 CUDA 化」的第一刀：B2 GEMM 核的**手写 SIMT CUDA** 被证明是**反向**方向（P7-44）

§46.6 把「融合 B2/B3 → K2 形状 CUDA」的首位动作写成「**把 `dq/dk/dw [BT,K]` 累加器按 warp 分片常驻
寄存器（先）**」。这句话有两种读法：(a) 在**保留 tensor core**（`ldmatrix`/`mma.sync`）的前提下把累加器分片
到多 warp 的寄存器（即 B1 §19–§24 的配方）；(b) 干脆**放弃 `tl.dot`、用普通 SIMT fp32 FMA** 写一个高占用、
寄存器常驻的核。这一轮先用**最便宜、最没有歧义**的方式把 (b) 试掉——因为它一次编译、逐位可核、可隔离
「寄存器常驻 + 高占用」本身值多少——结论是**负的**，从而把 (a) 钉成唯一方向。

证据：`harness/b2_core_probe.cu`（`b2_core_launch`）+ `harness/check_bwd_b2_core.py`
（`harness/check_bwd_b2_core.out.txt`，空闲卡）+ 调度器段 `harness/probe_b2_core_ncu.sh` /
`harness/ncu_b2_core_driver.py`（`harness/probe_b2_core_ncu.out.txt`）。

**被隔离的对象**：B2 (`wy_dqkg`) 里 FLOP 占大头的三条 V-归约 GEMM + 一条列归约——
`dq[t,k]=Σ_v do[t,v]·h[k,v]`、`dk=Σ_v v_new·dh`、`dw=Σ_v dv·h`、`dgk[k]=Σ_v h·dh`（`b_h[v,k]=h[k,v]`）。
**不**含 finalize（门链、`dkgb=A@dw`、`dA=dv@v^T`）、`dv2`/`db` 与 intra 对角折入（见 §47.6 边界）。

### 47.1 手写 SIMT 核的设计（为什么它「理应」代表 (b)）

`b2_core_probe.cu`：block = 256（8 warp），thread `tid` 固定拥有**列 `kk=tid%K`** 与 **8 行 `tbase..tbase+7`**
（`tbase=(tid/K)*8`），于是 `dq/dk/dw` 是每线程 **24 个 fp32 寄存器**累加器（8 行 ×3）、`dgk` 1 个；
warp 内 32 lane 拥有 32 个连续 `kk`（`h/dh` 的 smem 读拍成连续），`do/v_new/dv` 在同一 warp 内是广播读。
V 方向按 `BVT=32` 分 4 个 tile 暂存进 38 KB smem，无流水、无 TMA、无 tensor core——就是一个干净的高占用
SIMT GEMM。**它只用了 84 寄存器、38 KB smem**（`ncu` 实测，§47.4），解析/实测占用都远高于 Triton 版。

### 47.2 Part A — 正确性：手写核**逐位 = 真 fp32**（替身证明代数无误）

同操作数、`B2C_MODE=a`（四组 shape 含非整除 `T=250`；torch 参考关掉 tf32 走真 fp32）：

| 量 | CUDA vs torch(fp32) 最坏 `max_rel` | Triton 同 op 核 vs torch | CUDA vs Triton |
|---|---|---|---|
| `dq` | **0.000e+00（逐位）** | 8.97e-4 | 8.98e-4 |
| `dk` | **0.000e+00（逐位）** | 8.65e-4 | 8.66e-4 |
| `dw` | **0.000e+00（逐位）** | 8.36e-4 | 8.37e-4 |
| `dgk` | 5.2e-7 | 1.3e-7 | 5.2e-7 |

三点：**(1)** 手写核与真 fp32 torch **逐位相同**——所以下面它的慢**不是 bug**，是结构；**(2)** 同 op 的 Triton
核是 **tf32 底噪 8.4–9.8e-4**（`tl.dot` 默认 tf32，与 §6.6 一致）；**(3)** 即手写 SIMT 核**精度更高**
（真 fp32），但这在下面被证明**毫无价值**。

### 47.3 Part B/C — 头对头：SIMT 核**全部输给**同 op 的 Triton 核 3–5×

同操作数、只比这四条 op 的 kernel-only `dev`（`B2C_MODE=b/c`，CUPTI）：

| shape | CUDA core `dev` | Triton core `dev` (nw=8) | **CUDA/Triton** | 全 Triton B2 | 融合 B2/B3 | CUDA 核 GF/s |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 1727.7 µs | **556.2 µs** | **3.11×** | 903.0 µs | 1053.1 µs | 7409 |
| `B2 T1024 H8` | 228.4 µs | 82.2 µs | 2.78× | 121.9 µs | 144.5 µs | 7229 |
| `T512 H4` | 30.8 µs | 6.9 µs | 4.44× | 15.8 µs | 13.6 µs | 6642 |
| `T250 H3 (非整除)` | 30.9 µs | 6.0 µs | 5.20× | 14.1 µs | 10.8 µs | 2451 |

读法：**(1)** 手写 SIMT ~7.4 TFLOP/s，只有 H100 fp32 峰值的 ~11%、也远低于 Triton `tl.dot`——**放弃 tensor
core 是净亏**；**(2)** 关键结构数：**同 op 的 Triton GEMM 核（556 µs）已是完整 B2（903 µs）的 62%**，即
剩下 38% 才是 finalize + 读 `q/k/g/A/beta`（§47.5 用它重新定位 §46 的 1.94×）；**(3)** 手写核 1728 µs
**比完整的 Triton B2 还慢 1.93×**——「高占用寄存器常驻」这单个性质**不可能**是收益来源（否则 2 block/SM、
24.8% 占用的 SIMT 核不会全面落后 12.5% 占用的 Triton 核）。

### 47.4 ncu — 为什么慢：SIMT 核卡在 **L1/TEX 吞吐**（操作数重读），Triton 核卡在 **DRAM 带宽**

`probe_b2_core_ncu.sh`（`B8 T2048 H8`，两个核**读/写字节完全相同**——`dram__bytes_read.sum` 均为
1.28 GB、写 ~201 MB，是干净对照）：

| 指标 | 手写 SIMT 核 | Triton 同 op 核 (nw=8) | 含义 |
|---|---|---|---|
| Duration | 1739 µs | 552 µs | 3.15× |
| DRAM SOL / 有效带宽 | 25.3% / 0.85 TB/s | **79.8% / 2.67 TB/s** | SIMT 远离带宽；Triton 已带宽饱和 |
| **L1/TEX SOL** | **89.6%** | 41.1% | **SIMT 的真瓶颈**＝L1 管道 |
| FMA / tensor SOL | 14.3% / 0.01% | 2.3% / 7.4% | SIMT 算力也没打满 |
| regs / smem | 84 / 38 KB | 226 / 155.7 KB | 手写核更轻 |
| block/SM（theo） | **2**（reg 限） | **1**（reg+smem 双限） | 手写核占用更高 |
| 实测占用 | **24.8%** | 12.5% | —— |
| eligible warps/sched | 0.46 | 0.11 | Triton 更「发射饥饿」 |
| `long_scoreboard` | **9.42（59%）** | 4.13 | SIMT 在等 L1 数据 |
| bank conflict（shared-ld） | 151529 / 1.29e8 = **0.12%** | 0 | 都不是 bank conflict |

机制很清楚：SIMT 版每做 **24 个 FMA 要发 26 次 shared 读**（`do/v_new/dv` 各 8 行 + `h/dh` 各 1，
§47.1 的线程映射），**算术强度 ~0.9 op/word** → L1 管道 89.6% 打满、FMA 只 14%；而 Triton 用 `tl.dot`
（tensor core，一个 mma 吃 16×16×16）把每字节能做的乘加提高一个量级，于是同样的 1.28 GB 直接**撞到
DRAM 带宽（79.8% SOL）**，反而在 12.5% 占用下更快。**「寄存器常驻 + 高占用」并不能替代 tensor core 的
算术强度**——这正是 (b) 的失败点。

Triton 核的 552 µs 对照它的**同字节 DRAM 地板** `(1.28+0.20)GB / 3.35 TB/s = 442 µs` 只有 **1.25×**——
即**B2 的 GEMM 本身几乎无头寸**。

### 47.5 结论：这把 §46 的 1.94× 重新定位到**非 GEMM 段**，并钉死 K2 形状必须**保留 tensor core**

把 §46（融合核）与本轮（核内 GEMM）的字节/地板拆开：

| | 读+写字节 | HBM 地板 | 实测 `dev` | 实测/地板 |
|---|---|---|---|---|
| B2 GEMM 核（本轮 Triton） | 1.48 GB | 442 µs | 552 µs | **1.25×** |
| 融合 B2/B3（§46） | 1.96 GB | 586 µs | 988 µs | 1.69× |

融合核比 GEMM 核只多 **0.48 GB ≈ 146 µs** 的地板，实测却多 **436 µs**（988−552）。**所以 §46 的 ~1.9× 头寸
不在 GEMM 算术里（那部分已经 1.25× 贴地板），而在:(a) finalize/`dA`/`dv2`/`db` 与多出来的操作数
（`q/k/g/A/beta`）造成的寄存器/smem 压力与发射饥饿；(b) `h/dh` 流量本身（1.07 GB = GEMM 核读的 **84%**）。

**对目标卡的设计输入（修正/收窄 §46.6）**：

1. **K2 形状 CUDA 端口「必须保留 `ldmatrix`/`mma.sync` tensor core」**——本轮把「放弃 `tl.dot` 改 SIMT」这条
   显式证伪（3–5× 慢、且比完整 Triton B2 还慢 1.93×）。「寄存器分片常驻累加器」是与 tensor core **配套**
   的手段（B1 §22 的配方），不是可以单独拿来替代它的手段。
2. **GEMM 段（`dq/dk/dw/dgk`）不值得为它重写**：Triton 同 op 核已 1.25× 贴 DRAM 地板；重写的收益上界只有这
   25%。真正的头寸在**非 GEMM 段 + `h/dh` 流量**。
3. **最大单笔流量是 `h/dh`（fp32，1.07 GB = 84%）**——与 §26/§29 一致；B1→B2 融合（`dh` 不落盘）或 `h/dh`
   走 bf16 出口是比「重写 GEMM」更直接的减字节杠杆（§29 已量化 `h/dh` bf16 值内核 ~10%）。
4. **重排 §46.6 的动作顺序**：先做「**非 GEMM 段**的寄存器常驻分片 + 与 GEMM 段同 CTA 交接 + `h/dh` 少搬」，
   再谈 TMA 流水；**不要**从「用 SIMT 省掉 `tl.dot` 的 smem staging」入手。
5. **判据（方法学）**：判断「一个 SIMT/tensor 决策」时，先看**算术强度**（op/word）——本轮 SIMT 核
   ~0.9 op/word → 必撞 L1；`tl.dot` 的 mma 是把它抬到 10+ op/word 的手段。`ncu` 的 L1/TEX SOL 与
   `long_scoreboard` 是这条判据的直接读数。

### 47.6 边界 / 下一步（诚实）

- 本轮**只**隔离了 B2 的 GEMM 核；finalize/`dA`/`dv2`/`db` 与 intra 折叠**未**进 CUDA（在 §47.5 里作为
  「非 GEMM 段」按字节/时间差定位，但未单独建核量化）。**下一步（真正工程化）**：以本轮为反面依据，直接建
  **保留 `mma.sync` 的 K2 形状 B2/B3**（累加器按 warp 分片、操作数 TMA/`ldmatrix` 整块、非 GEMM 段同 CTA
  交接），并按 §47.5 优先减 `h/dh` 字节。
- 手写 SIMT 核若做**寄存器分块**（每线程 8×8 tile）可把算术强度抬到 ~8 op/word、脱离 L1 bound，但 tensor
  core 的 mma 已达 16 op/word 量级且 Triton 核已带宽饱和——因此**不值得**再往 SIMT 方向投入（本轮已判死）。
- **踩坑**：(a) `dgk` 输出首版错用了 `h/dh` 的 `KD*VD` large base offset（应 `((b*NT+c)*H+h)*K`），
  触发非法访存、错误只在**下一次 triton 载入**时才以 `illegal memory access` 冒出来；(b) torch 参考若不开
  `allow_tf32=False` 会走 cublas tf32，把「CUDA vs torch」污染成 ~3e-4（看起来像 bug），关掉后手写核**逐位=0**。

## 48. 融合 B2/B3 的 ~1.9× 头寸是**字节**还是**发射**？——受控字节消融（P7-45）

§47.5/§47.6 把「融合 B2/B3 → K2 形状 CUDA」的头寸**重新定位到非 GEMM 段 + `h/dh` 流量**，但那是从
「同字节去 compute 的地板核 509µs vs 融合核 988µs」以及「GEMM 核 1.48GB vs 融合核 1.96GB」两条**差值**
推出来的，**非 GEMM 段从未单独建核量化**；而且「头寸在 `h/dh` 流量」隐含「核是带宽受限」，与 §26/§29
（`h/dh` bf16 只值内核 ~10%、非带宽受限）**互相矛盾**。本轮用**最便宜的受控实验**把这两件事一次问清楚：
**在同一份融合核上，只把选定的输入 `tl.load` 换成 `tl.zeros`（`tl.dot` 全保留、store 全保留、程序形状不变），
看时间对哪股字节敏感**。这样既不扰动 compute 图、也不触发 DCE，把「字节」与「compute/发射」干净分开。

> 设计踩坑（先说）：**第一版想做「把 `tl.dot` 关掉」的消融**——用同 shape 的秩-1 归约外积替掉被关的
> `tl.dot`（读同样操作数、保字节）。结果 Triton 为这些归约做了**跨 lane 布局转换**，`no_post` 变体
> **4964 spills / 32 regs / 慢 50×**——是**编译器病态**而非测量。故改为**只渡字节**（`tl.zeros` 替 load）。

证据：`harness/check_bwd_b2b3_seg.py`（`harness/check_bwd_b2b3_seg.out.txt`，空闲卡）+
调度器段 `harness/probe_b2b3_seg_ncu.sh` / `harness/ncu_b2b3_seg_driver.py`
（`harness/probe_b2b3_seg_ncu.out.txt`）。变体：`full` / `no_hdh`（h/dh→0）/ `bf16_hdh`（h/dh 按 bf16 载入）/
`no_qkg`（q/k/g→0）/ `no_odv`（do/v_new/dv/v→0）/ `no_all`（四组全→0）。

### 48.1 Part A — 正确性：被门控的 `full` **逐位 = 未改动的融合核**

`full` 六个变体全关、什么都不渡，与 `build_fused_b2b3()` 的输出在 5 组 shape（含非整除 `T=250`、`T=513`、
`K=64/V=128`）上 `dq/dk/dv2/db/dg` **`max_abs=0`（逐位）**——证明加门控没有改动任何算术/布局，下面所有
差异都只来自被渡掉的字节。

### 48.2 Part B — 计时（CUPTI `dev`，fp32 部署 mix）

| shape | 变体 | 解析字节[MB] | `dev`[µs] | `dev/full` | 省字节[MB] | 省时间[µs] |
|---|---|---|---|---|---|---|
| B8 T2048 H8 | full | 1745 | 1050.8 | 1.00× | 0 | 0 |
| | **no_hdh** | 672 | **589.8** | **0.56×** | 1074 | 461.0 |
| | bf16_hdh | 1208 | 1566.2\* | 1.49×\* | 537 | −515\* |
| | no_qkg | 1544 | 762.4 | 0.73× | 201 | 288.4 |
| | no_odv | 1544 | 928.0 | 0.88× | 201 | 122.7 |
| | no_all | 269 | 170.1 | 0.16× | 1476 | 880.7 |
| B2 T1024 H8 | full | 218 | 144.3 | 1.00× | 0 | 0 |
| | no_hdh | 84 | 77.1 | 0.53× | 134 | 67.2 |
| | no_qkg | 193 | 109.1 | 0.76× | 25 | 35.2 |
| | no_all | 34 | 22.0 | 0.15× | 185 | 122.3 |
| T512 H4 | full | 27 | 13.5 | 1.00× | 0 | 0 |
| | no_hdh | 10 | 7.4 | 0.54× | 17 | 6.2 |
| | no_all | 4 | 4.1 | 0.30× | 23 | 9.4 |

\* `bf16_hdh` 的 `dev`/`wall` 把 harness 每次调用里的 `h/dh.to(bf16)` 两个 elementwise kernel 也算进去了
（与 §26.5/§29.1 同一个陷阱）；**干净口径看 §48.3 的 `ncu Duration`（=1.04ms）。**

**第一眼**：`no_hdh` 去 1074MB 省 461µs（边际 **2.33 TB/s**，接近 HBM3 峰值），像是「h/dh 很贵」；`no_all`
（所有大操作数不读、只留 compute+store）只要 **170µs = full 的 16%**。但 `no_qkg` 只去 **201MB 却省 288µs**
（边际 **0.70 TB/s**，远低于带宽）——**单位字节的收益差 3×**，说明**不是**简单带宽账，背后还有寄存器/发射的
二阶效应。`ncu` 把它拆开。

### 48.3 `ncu`（B8 T2048 H8）——为什么不是带宽、也不是占用率

| 变体 | Duration | DRAM 读 | DRAM 写 | DRAM% SOL | L2 字节 | tensor% | `fma`% | `long_sb` | issue/cyc | regs | smem[KB] | occ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **full** | 1.05ms | 1.58GB | 383MB | **55.9** | 3.66GB | 6.21 | 5.46 | 5.77 | 0.14 | 255 | 165.9 | 12.49 |
| no_hdh | 583µs | 487MB | 258MB | **38.1** | 1.16GB | 7.67 | 6.15 | 4.00 | 0.18 | 253 | **34.8** | 12.46 |
| bf16_hdh | **1.04ms** | **1.04GB** | 386MB | 40.6 | 3.39GB | 6.16 | 6.24 | 4.67 | 0.17 | 255 | 51.2 | 12.41 |
| no_qkg | 764µs | 1.36GB | 264MB | **63.4** | 2.47GB | 8.13 | 4.72 | 3.22 | 0.14 | **165** | 166.4 | 12.40 |
| full(T512H4) | 20µs | 24.5MB | 2.3MB | 40.0 | 58.7MB | 5.91 | 5.20 | 6.00 | 0.14 | 255 | 165.9 | 12.45 |

（`no_all` 的 `ncu` 记录**为空**——ncu 在该配置上卡住未出数据；其 `dev`/静态量见 Part B/C。）

四条读数：

1. **`full` 不是带宽受限**：DRAM 只 **55.9% SOL**（1.87 TB/s）、issue **0.14/cyc**、`long_sb` 5.77——
   ncu 直接判「latency issues」；与 §46/§47 一致。
2. **`h/dh` = 1.09GB = DRAM 读的 55%**（1.58GB 里），是单笔最大流；`no_hdh` 去它后 **1.78× 快**，
   但 **occupancy 一动不动（12.46% ≠ 变好）**、DRAM% 反**降到 38%**——**既不是占用率、也不是带宽**，
   收益来自「少搬半壁字节 + 去掉 `[128,128]` 常驻操作数及其 mma 取数」。注意 `no_hdh` 的 smem 从
   165.9 掉到 34.8KB，却仍被 **253 寄存器**钉在 1 block/SM（`Block Limit Registers=1`）。
3. **`bf16_hdh`（字节砍半）= 白干**：DRAM 读 1.58→**1.04GB**（省 537MB）却 `Duration 1.04ms ≈ full
   1.05ms`、L2 只从 3.66→3.39GB、`long_sb` 反从 5.77→4.67 但 `wait`/转换把时间吃回。**这是对「头寸在
   `h/dh` 带宽」的最直接反证**：只要那块张量还得常驻寄存器/走 mma 取数，**省字节不换时间**。与 §29 的
   「`h/dh` bf16 有条件成立、上界≈内核 ~10%」完全自洽。
4. **`no_qkg` 是异常项**：只去 201MB 却快 1.37×、DRAM% 升到 **63.4%**（反而更接近带宽），关键在
   **regs 255→165**——少保活 q/k/g 三个 `[BT,K]` 张量把寄存器打下来，`long_sb` 5.77→3.22。
   **寄存器保活（→ spill/发射）才是真正的一阶量**。

### 48.4 结论 / 对目标卡的设计输入（收窄 §47.5）

把「`full` 1051µs、compute 地板 `no_all` 170µs」与上表并看：

- **GEMM/张量算术不是瓶颈**：所有 `tl.dot` 一起跑（`no_all`，无大操作数读、只 store）只 **170µs＝16%**；
  `full` 的 `tensor%` 只 6.2%。这坐实 §47.5「GEMM 段不值得重写」。
- **头寸既不是「纯字节」也不是「占用率」，而是「大操作数常驻寄存器 → spill/发射饥饿」**：
  - 省字节不换时间（`bf16_hdh` 反证）；
  - 占用率 12.5% 在 `full`/`no_hdh`/`no_qkg` 三档**完全相同**（都被 regs 钉死），却相差 1.78×；
  - 唯一能让时间大幅下降的是**移走出寄存器保活的大 fp32 张量**（`no_hdh` 去掉 h/dh；`no_qkg` 去掉 q/k/g）。
- **对 K2 形状 CUDA 端口的设计输入（修正 §47.5 措辞，动作顺序不变）**：
  1. **不要指望「`h/dh` 换 bf16」当杠杆**——在寄存器常驻+mma 取数的结构下它**不换时间**（仅 ~10% 上界且
     本实验为 wash）。真正的杠杆是**别让 `[V,K]` 的 `h/dh` 整块常驻寄存器**：用 TMA 按 tile 流式喂入、
     warp 内只留小累加器（§22–§24 B1 的配方），这正是「寄存器分片常驻」的**反面**——分片的是**小累加器**，
     不是大操作数。
  2. 动作顺序仍是 §47.5 的：**非 GEMM 段寄存器分片 + 同 CTA 交接 + 少保活大操作数**，TMA 流水其次；
     `no_qkg` 的 255→165 寄存器是可达的近期目标。
  3. 判据补一条：**「这个优化减的是字节还是寄存器保活？」**——本实验显示在发射受限核里后者更值钱
     （用 `ncu` 的 `Block Limit Registers` + `regs/thread` + `long_scoreboard` 直接读）。

### 48.5 边界 / 下一步（诚实）

- 本轮**只**做了「渡输入字节」的受控消融（`tl.dot`/store/程序形状不动）；**没有**做「渡 compute」的消融
  （第一版失败于编译器病态，见开头踩坑）。compute 地板用 `no_all`（170µs）近似——它仍含全部 `tl.dot`
  与 store，故是**上界**。
- `no_all` 的 `ncu` 记录缺失（ncu 卡住）；`bf16_hdh` 的 Part B `dev/wall` 含 harness 转换税，只有 `ncu
  Duration` 是干净口径。
- **下一步（真正工程化）**与 §47.6 相同：建**保留 `mma.sync` 的 K2 形状 B2/B3**；但本轮把「怎么做」钉得更
  具体——**首位动作＝把大操作数（尤其 `h/dh`）从寄存器保活里拿掉、只留 warp 分片的小累加器**，
  并用「省字节 vs 省保活」判据逐项验证；`h/dh` bf16 不作为独立杠杆。

## 49. 「K2 形状 CUDA 化」的第二刀：B2 GEMM 核保留 `mma.sync` + 流式操作数——**首次正面**（P7-46）

§47（P7-44）把 §46.6 的「累加器按 warp 分片常驻寄存器」读法 (b)（**放弃 `tl.dot` 改 SIMT**）试掉并
证伪（慢 3–5×）；§48（P7-45）又把融合核的 ~1.9× 头寸定位到「**大操作数（`h/dh`）常驻寄存器 → spill/
发射饥饿**」，并给出首位动作：**「把大操作数从寄存器保活里拿掉、只留 warp 分片的小累加器」**。本轮
兑现这个首位动作——在**保留 `mma.sync` tensor core** 的前提下，把 B2 的 V-归约 GEMM 核写成 **K2 形状**
（操作数经 smem 流式喂入、寄存器里只剩每 warp 的小累加器），并在同一份操作数上与 P7-44 的 SIMT 核、
P7-44 的 identical-op Triton 核头对头。**这是本专题第一个跑赢 Triton 同 op 核的 CUDA 反向核。**

证据：`harness/b2_mma_probe.cu`（`b2mma_launch`）+ `harness/check_bwd_b2_mma.py`
（`harness/check_bwd_b2_mma.out.txt`，空闲卡）+ 调度器段 `harness/probe_b2_mma_ncu.sh` /
`harness/ncu_b2_mma_driver.py`（`harness/probe_b2_mma_ncu.out.txt`）。

### 49.1 设计：K2 形状落在 B2 的 V-归约 GEMM 上

被隔离的对象与 P7-44 **完全相同**（同一份 `dq/dk/dw/dgk` 四条 op、同一批 fp32 操作数、`K=V=128`、
部署 `BT=16`）：

```
dq[t,k] = Σ_v do[t,v]·h[k,v]      dk[t,k] = Σ_v v_new[t,v]·dh[k,v]
dw[t,k] = Σ_v dv[t,v]·h[k,v]      dgk[k]  = Σ_v h[k,v]·dh[k,v]
```

K2 形状（`b2_mma_probe.cu`）：

- **grid `(NT, B·H)`，block = 8 warp**；warp `w` 固定拥有**输出列（K 维）**`[w·16, w·16+16)`
  （即 N 分片），故每 warp 只背 `dq/dk/dw` 各 `[16,16]` 的 fp32 累加器（`accq/ak/aw[2][4]`，
  `b2_mma_probe.cu:131`）+ 一个 `dgk` 标量（`:136`）——**大操作数一个都不常驻**。
- `h/dh/do/vn/dv` **按 V（收缩维）分块 `BVK`、经 `cp.async` 双缓冲流水**搬进 smem
  （`prefetch` `:85-113`，`__pipeline_memcpy_async` `:99-112`），寄存器里流过即用即弃。
- 收缩用裸 `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`（`mma8` `:75-82`，与 `tl.dot` 默认
  tf32 同精度），A/B 片段**直接从 smem 按 mma 布局读**（`:158-177`），不用 `ldmatrix`（tf32 无 16-bit
  ldmatrix；代价是下面 `ncu` 里的 bank conflict）。
- 非整除 `/ T` 的尾块 token 行在 `prefetch` 里直接写 0（`:102-104`）；`h/dh` 是 chunk 粒度、无需 mask。

与 §48.4 的判据对表：**这个改动减的是「寄存器保活」不是「字节」**——`h/dh` 一个字节没省（仍 fp32、
仍读 1.07 GB），但 `[V,K]` 的整块不再进寄存器。

### 49.2 Part A — 正确性：K2-mma 核落 **tf32 底噪**（与 Triton 同精度档）

同操作数（四组 shape 含非整除 `T=250`；torch 参考关 tf32 走真 fp32）：

| 量 | CUDA(tf32) vs torch(fp32) | Triton(tf32) vs torch | CUDA vs Triton |
|---|---|---|---|
| `dq` | 3.58e-4 | 8.97e-4 | 9.85e-4 |
| `dk` | 3.28e-4 | 8.65e-4 | 9.45e-4 |
| `dw` | 3.41e-4 | 8.36e-4 | 9.80e-4 |
| `dgk` | 1.63e-7 | 1.34e-7 | 1.63e-7 |

四个量全部落 **tf32 底噪**；CUDA 与 Triton 互差 ~1e-3（都 tf32、累加序不同），`dgk`（纯 fp32 列归约）
~1.6e-7。**CUDA 侧甚至比 Triton 更贴近 fp32**（3.4e-4 vs 8.4e-4）——因为 `cvt.rna.tf32` 四舍五入而非
截断（`to_tf32` `:69-73`）。代数无误，形式与 P7-44 逐字对应。

### 49.3 Part B/C — 头对头：**大 shape 首次跑赢**同 op Triton 核

同操作数、只比这四条 op 的 kernel-only `dev`（CUPTI），默认 `BVK=32/NS=2`：

| shape | K2-mma CUDA `dev` | Triton 核 `dev` (nw=8) | **CUDA/Triton** | 全 Triton B2 | 融合 B2/B3 | 地板 |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | **517.3 µs** | 554.4 µs | **0.93×** | 899.5 | 1053.1 | 442 |
| `B2 T1024 H8` | **71.8 µs** | 82.7 µs | **0.87×** | 122.1 | 145.1 | — |
| `T512 H4` | 9.5 µs | **6.8 µs** | 1.40× | 15.7 | 13.4 | — |
| `T250 H3 (非整除)` | 8.6 µs | **5.9 µs** | 1.47× | 14.3 | 10.9 | — |

- **部署主 shape `B8 T2048 H8`：0.93×**（517 vs 554），且比 §47 的地板 **442 µs** 只高 **1.17×**
  （Triton 核是 1.25×）——**第一次把「K2 形状」变成一个正收益的 GEMM 核**，与 §47.5「GEMM 段已 1.25×
  贴地板、重写收益上界 25%」自洽：本轮吃回了那 25% 里的约 1/3。
- **小 shape 反而慢 1.4×**（`T512 H4`/`T250 H3`）：这两种 shape 的 grid 只有 128/48 个 CTA、单核 ~6–10µs，
  是**延迟/固定开销主导**（K2 核有 `cudaFuncSetAttribute` 后的绑定 smem + `cp.async` prologue 的固定段），
  Triton 的短核更贴合小 grid。故**核的选择应随 shape 切换**（与 §42.3 的 B1 结论同构）。
- 与 §47 的 SIMT 核并看：SIMT 1728 µs → 本轮 517 µs，**同一批 op 提速 3.3×**；对 `dq/dk/dw` 的计算
  吞吐 ~24.9 TFLOP/s（SIMT 只 7.4）——**tensor core 是量级项**（§47.5 点 1 再次坐实）。

### 49.4 Part D / `ncu`（`B8 T2048 H8`）——为什么它贴住 DRAM 地板

三个核读写字节几乎相同（DRAM 读 1.28 GB、写 ~200 MB，干净对照）：

| 指标 | **K2-mma CUDA** | P7-44 SIMT CUDA | Triton 同 op 核 (nw=8) |
|---|---|---|---|
| Duration | **520 µs** | 1730 µs | 551 µs |
| DRAM SOL / 带宽 | **84.6% / 2.84 TB/s** | 25.5% / 0.85 TB/s | 79.9% / 2.68 TB/s |
| L1/TEX SOL | 46.0% | **89.6%** | 41.0% |
| L2 SOL | 88.8% | 30.4% | 77.8% |
| Compute(SM) SOL | 31.0% | 24.6% | 14.8% |
| tensor / fma SOL | 7.76% / 9.00% | 0.01% / 14.3% | 7.34% / 2.28% |
| regs/thread | **70** | 84 | 226 |
| Block Limit Reg / smem | **3 / 2** | 2 / 3 | 1 / 1 |
| achieved occ | **24.6%** | 24.9% | 12.4% |
| `long_scoreboard` | 3.09 | **9.46** | 4.11 |
| `short_scoreboard` | **1.01** | 1.38 | 2.89 |
| `wait` | 1.29 | 0.89 | 1.49 |
| `mio_throttle` | 2.37 | 0.81 | 2.11 |
| smem bank conflict | 8.81 M | 0.15 M | **0** |
| L2 字节 | 3.03 GB | 2.25 GB | 2.24 GB |

读法：

1. **移出「大操作数常驻」直接兑现**：寄存器 **226→70**（Triton→CUDA），`Block Limit Registers 1→3`、
   achieved 占用 **12.4%→24.6%**，`long_scoreboard` 4.11→**3.09**；而 SIMT 核（无 tensor core）则撞
   **L1/TEX 89.6%**、`long_sb` 9.46。**「寄存器保活」是一阶量（§48.4）在这里被独立复现。**
2. **新核贴 DRAM 地板**：DRAM **84.6% SOL（2.84 TB/s）**、L2 88.8%（两个最高 pipe）——比 Triton 核的
   79.9% 更饿带宽，这就是 0.93× 的来源；tensor 7.76% / fma 9% 都低（**不是算力受限**）。
3. **残差重定位到 L2 流量 + smem 冲突**：(a) 本核 L2 字节 **3.03 GB 比 Triton 的 2.24 GB 高 35%**
   （DRAM 读字节相同）——`cp.async` 16B 粒度的 A/B 操作数请求放大到 32B sector；(b) 直接从 smem 读
   mma 片段（tf32 无 `ldmatrix`）产生 **8.81M bank conflict**（Triton 的 swizzle+`ldmatrix` 为 0），
   表现为 `mio_throttle` 2.37 偏高。**L2 88.8% 已是本核最高 pipe → 下一杠杆是降 L2 流量
   （XOR swizzle/`ldmatrix` 消冲突 + 更粗 cp.async），而不是再加 mma 密度。**

### 49.5 编译期旋钮扫描（`BVK`×`NS`）

同源重编译六档（`B2MM_CFLAGS`），`B8 T2048 H8` / 小 shape：

| `BVK`/`NS` | `B8 T2048 H8` | `B2 T1024 H8` | `T512 H4` | `T250 H3` |
|---|---|---|---|---|
| 32/2（默认） | 517.3 | 71.8 | 9.5 | 8.6 |
| 32/3 | **505.7** | 72.5 | **9.1** | **8.1** |
| 64/2 | **499.1** | 72.6 | 9.4 | 7.8 |
| 16/3 | 718.3 | 93.4 | 10.1 | 10.0 |
| 16/4 | 706.3 | 94.5 | 9.7 | 9.6 |
| 128/1 | 524.4 | 81.4 | 9.9 | 8.4 |

大 shape 最优 `BVK=64/NS=2`（0.90×）或 `BVK=32/NS=3`（0.91×）；`BVK=16`（细收缩、多 stage）**明显变慢**
（718 µs，cp.async 请求过碎）。小 shape 在 8–10 µs 之间对旋钮不敏感（固定开销主导）。默认取 `BVK=32/NS=2`。

### 49.6 结论 / 对目标卡的设计输入

- **§48.4 的首位动作方向正确且可兑现**：在**保留 tensor core**前提下，把 `h/dh` 从寄存器保活里移出、
  只留 warp 分片小累加器，使 CUDA B2 GEMM 核在部署主 shape 上**首次跑赢** identical-op Triton 核
  （0.93×，且离 DRAM 地板 1.17×），寄存器 226→70、占用 12.4%→24.6%。**与 §47 的 SIMT 反例合起来**，
  K2 形状被双向钉死：**必须 tensor core + 流式操作数 + 小累加器分片**，三件缺一不可。
- **收益上界仍受限**：GEMM 段本就 1.25× 贴地板（§47.5），本轮吃到 1.17×，**剩余头寸确实在非 GEMM 段**
  （finalize/`dA`/`dv2`/`db` 与 intra 折叠，§47.6）——B2 全核 899 µs 里本核只 517 µs，另 ~380 µs 是 GEMM
  之外。故下一步仍是 §47.6 的「非 GEMM 段进 CUDA + 同 CTA 交接」，本核可直接作为其中的 GEMM 段基座。
- **核按 shape 切换**：大 shape 用 K2-mma CUDA 核（0.90–0.93×），小 shape 仍交给 Triton 短核
  （CUDA 慢 1.3–1.5×，固定开销）；部署时按 grid/`T` 选。
- **本核自身下一杠杆 = 降 L2 流量**：当前 L2 88.8% 成最高 pipe 且 L2 字节比 Triton 高 35%——用 XOR
  swizzle/`ldmatrix` 消 8.81M bank conflict、把 `cp.async` 放宽到 32B/128B，可把 DRAM SOL 从 84.6% 再抬。

### 49.7 边界 / 下一步（诚实）

- 本核只隔离 B2 的**三条 V-归约 GEMM + `dgk` 列归约**；finalize（门链、`dkgb=A@dw`、`dA=dv@v^T`）、
  `dv2`/`db` 与 intra 对角折叠**未**进 CUDA（同 §47.6）。故「B2 全核」对照里另 ~380 µs 未覆盖。
- 未做 TMA（用 `cp.async`）；未做 `ldmatrix`/XOR swizzle（tf32 无 16-bit ldmatrix，需 CUTLASS 式手工重排）。
- 未做 bf16 变体：本轮刻意保持部署的 fp32/tf32 mix，以与 §47 的 identical-op Triton 核**同 dtype** 对照
  （§48.4 已证 `h/dh` bf16 在本结构下非独立杠杆）。
- **下一步（真正工程化）**：把本核作为 GEMM 段基座，向其中补非 GEMM 段（同 CTA 内交接 `dA`/`dv2`/`db`），
  并把 L2 流量/bank conflict 收口；同时验证小 shape 是否值得单独维护一个短核。

## 50. B2 K2-mma 核那 8.81M bank conflict 是谁：**不是 mma 片段读，是 `dgk` 列归约**（P7-47）

§49.4.3 把 K2-mma CUDA B2 核的 `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld = 8.81M`
归因到「直接从 smem 读 mma 片段（tf32 无 `ldmatrix`）」，并把它列为 §49.6 的下一杠杆之一。本轮用**受控消融**
把这个归因证伪并给出正确来源与修复。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_NODGK` / `-DB2M_DGKCF`）+ `harness/check_bwd_b2_dgk.py`
（`harness/check_bwd_b2_dgk.out.txt`，空闲卡）+ `harness/probe_b2_dgk_ncu.sh`（`harness/probe_b2_dgk_ncu.out.txt`）。

### 50.1 受控消融：`dq/dk/dw` 逐位不动，只动 `dgk`

三个变体共用同一份操作数、同一套 mma（`dq/dk/dw` 三条 V-归约 GEMM **一字不改**）：

- `base`：§49 原核。
- `nodgk`：**整段删掉** `dgk` 列归约（`b2_mma_probe.cu:199-209`；输出置 0，纯诊断）。
- `dgkcf`：把 `dgk` 列归约改成**无冲突映射**（`b2_mma_probe.cu:199-217`）。

`dgkcf` 的映射依据：smem word 地址 = `kcol*BVP + vv`、`BVP=36≡4 (mod 32)`，故 bank = `(4·kcol + vv) & 31`；
**16 个连续 `kcol` 会以 `(kcol, kcol+8)` 两两别名**（`4·kcol` 与 `4·(kcol+8)` 相差 32）。原实现让 lane
`l` 取 `kcol = warp*16 + (l&15)` 正好踩中这个别名（2-way）。改法：**一条指令只用 8 个 `kcol`**
（`c = l&7` 作 `kcol` 低位、`s = l>>3∈{0,1,2,3}` 作 `vv` 子类），则 bank = `4·c + s` 恰好铺满 `0..31`，
**与 `vv` 步进无关**；两半（`ph=0/1`）分别覆盖 `kcol` 的 `0..7` / `8..15`，最后对 `s` 方向做两次
`__shfl_xor` 归约（lane `l<8` 持有两半之和）。工作量不变（每 lane 仍 32 次 smem load），只是不再重放。

### 50.2 Part A — 受控消融正确性

四组 shape（含非整除 `T=250`）：`nodgk` / `dgkcf` 的 **`dq/dk/dw` 与 `base` 逐位相同（`max_abs=0`）**——
证明消融是受控的（只动了 `dgk`）。`dgkcf` 的 `dgk` vs torch fp32 **`max_rel=1.31e-7~1.63e-7`**，与 `base`
**完全相同**（`base` 也是 `1.31e-7`）；两者互差 `max_abs=1.9e-6`（fp32 加法序，落在近零元素上）。

### 50.3 Part B / `ncu` — 冲突 96% 来自 `dgk`，且**在部署 shape 上不值得换**

`B8 T2048 H8`、`ncu`（同一份操作数）：

| 变体 | bank conflicts | shared wavefronts | conflicts/ wavefronts | `mio_throttle` | regs | Duration | DRAM SOL | L2 SOL | L2 字节 |
|---|---|---|---|---|---|---|---|---|---|
| `base` | **8,807,285** | 42,650,502 | 20.6% | 2.37 | 70 | 519.9 µs | 84.6% | 88.8% | 3.03 GB |
| `nodgk` | **285,832** | 25,454,248 | 1.1% | 2.21 | 47 | 518.5 µs | 84.9% | 89.0% | 3.03 GB |
| `dgkcf` | **329,787** | 34,093,171 | 1.0% | 1.93 | 62 | 520.1 µs | 84.6% | 88.7% | 3.03 GB |

读法：

1. **归因证伪 + 定位**：删掉 `dgk` 后冲突从 **8.81M 掉到 0.29M（−96.7%）**，`dgkcf` 亦 **−96.3%**——那
   8.81M **几乎全是 `dgk` 列归约**，**不是** mma 片段读。旁证：mma 片段读按 lane `(g,t)` 地址
   `kcol/BVP = 4·g + t` 展开恰好铺满 `0..31`，`BVP=36` 的 padding **本身就是无冲突**的；删 `dgk` 后只剩
   ~0.29M（1.1%）残差（prefetch 写/边界）。
2. **`dgk` 的指令成本也是它**：`nodgk` 的 shared wavefronts 42.65M→**25.45M**（少 17.2M = 整个 `dgk`
   循环的 load+重放）；`dgkcf` 保留工作量但去冲突，wavefronts 34.09M（比重放版少 8.6M）。
3. **但在部署主 shape 上等于零**：三个变体 Duration 519.9 / 518.5 / 520.1 µs、DRAM 84.6–84.9%、L2
   88.7–89.0%——**完全被 DRAM/L2 带宽钉住**，去冲突只把 `mio_throttle` 2.37→1.93，不兑现时间。
   **L2 字节三个变体都是 3.03 GB**——§49.4.3 的「L2 +35%」也**与 `dgk` 无关**（另一条待查线索）。

### 50.4 小 shape：`dgk` 的收益真实但只 ~3–4%

`B2 T1024 H8` / `T512 H4` / `T250 H3`（`check_bwd_b2_dgk.out.txt`）：

| shape | base | nodgk (删) | dgkcf (无冲突) |
|---|---|---|---|
| `B8 T2048 H8` | 517.9 | 516.1 (1.003×) | 517.0 (1.002×) |
| `B2 T1024 H8` | 71.4 | 70.0 (1.020×) | 70.9 (1.006×) |
| `T512 H4` | 9.6 | **7.9 (1.209×)** | 9.2 (1.041×) |
| `T250 H3` | 8.7 | **7.1 (1.230×)** | 8.4 (1.032×) |

小 shape（grid 只有 32/16 个 CTA、单核 ~6–10µs）是**延迟主导**：整个 `dgk` 循环（含重放）占 ~20%；
`dgkcf` 消掉重放后拿回 **~3–4%**，剩余 ~17% 是「循环本身必须存在」的串行成本（与冲突无关，删掉才显示）。
与 §49.3「小 shape 交给 Triton 短核」的结论一致：小 shape 不值得为这点收益维护 CUDA 分支。

### 50.5 结论

- **更正 §49.4.3/§49.6**：那 8.81M smem bank conflict **不是** mma 片段读（tf32 无 `ldmatrix`），而是
  **`dgk` 列归约的 16-`kcol` 别名**；`BVP=36` 的 fragment padding 本就无冲突。
- **修复免费且位级安全**：`-DB2M_DGKCF` 去冲突 96%、`dgk` 精度不变（1.3e-7）、`dq/dk/dw` 逐位不变；
  但**部署主 shape 零收益**（DRAM/L2 bound）、小 shape 仅 3–4%——故 `base` 保持默认，`dgkcf` 留作
  小 shape 或将来非 memory-bound 结构下的储备。
- **新的待查线索**：§49.4.3 的「L2 字节比 Triton 高 35%（3.03 vs 2.24 GB）」在三个变体上**纹丝不动**，
  与 `dgk`/冲突均无关；下一杠杆应直接定位这 0.79 GB 来自哪条 global 访存（`cp.async` 请求粒度/`h/dh`
  的 L2 复用），而不是 smem swizzle。

## 51. B2 K2-mma 核的「L2 字节 +35%」是谁：**cp.async 的 L1-bypass 读路径（+0.64 GB）+ 逐标量 32-bit store（+0.13 GB）**（P7-48）

§50.5 把 §49.4.3 的「本核 L2 字节比 identical-op Triton 核高 35%（3.03 vs 2.24 GB）」立为**新待查线索**，
但它与 `dgk`/bank conflict 无关，三个变体上纹丝不动。本轮用**受控字节消融**把它完整拆开：0.79 GB 的差
**全部**由两条**独立、可分别消除**的机制构成——(a) **`cp.async` 的 L1-bypass 读路径**多出 **+0.64 GB**，
(b) 输出的**逐标量 32-bit `STG.E`** store 多出 **+0.13 GB**。每一条都能单独关掉，两条都关掉后本核
L2 字节降到 **2.23 GB，与 Triton 逐位相同**。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_NOHDH`/`-DB2M_NODOV`/`-DB2M_VECST`/`-DB2M_NOASYNC`）+
`harness/check_bwd_b2_l2.py`（`harness/check_bwd_b2_l2.out.txt`，空闲卡）+ `harness/probe_b2_l2_ncu.sh`
（`harness/probe_b2_l2_ncu.out.txt`）。

### 51.1 受控消融：同一份 mma、同一批操作数，只动「哪条字节路径」

被隔离的对象与 §49 完全一致（`dq/dk/dw` 三条 V-归约 GEMM + `dgk` 列归约、`K=V=128`、`BT=16`、
`B8 T2048 H8`）。所有变体共用同一套 `mma.sync.m16n8k8.tf32`，只改访存：

- `base`：§49 原核（`cp.async` 读；每个输出 2×32-bit `STG.E`）。
- `nohdh`：`h/dh` 的 `cp.async` 换成 smem 零填充（不读 global）——量出 `h/dh` 的字节侧。
- `nodov`：`do/vn/dv` 的 `cp.async` 换成零填充。
- `nohdh_nodov`：所有操作数都不读，只剩 store → 残差基线。
- `vecst`：输出改用 64-bit `float2` store（`b2_mma_probe.cu:273-287`），读数不变。
- `noasync`：把 16B `cp.async`（`LDGSTS.E.BYPASS.128`）换成同步 16B `float4` load+store
  （普通走 L1 的 `ld.global` 路径，`b2_mma_probe.cu:42-48`），**同 16B 宽度、同访问模式**。
- `noasync_vecst`：两者叠加。

`noasync*` 会破坏 async 流水（预期变慢）**只作字节诊断**；`vecst` 是**可部署、数值逐位无损**的那半个修复。

### 51.2 Part A — 正确性：`vecst` 与 `base` **逐位相同**

四组 shape（含非整除 `T=250`），`vecst` vs `base` 的 `dq/dk/dw/dgk` **`max_abs=0`**（全部逐位）；两者
vs torch(fp32) 同落 tf32 底噪（`dq 3.58e-4/dk 3.28e-4/dw 3.41e-4/dgk 1.63e-7`），与 §49.2 完全一致。
即 `float2` store 纯属 store 编队，**运算一字不改**。

SASS 佐证（`probe_b2_l2_ncu.out.txt` 末）：`base` = **25 × `STG.E`(32-bit)** + 10 × `LDGSTS`；
`vecst` = **1 × `STG.E`（`dgk` 标量）+ 12 × `STG.E.64`**——每个输出对从两条 4B store 合成一条 8B store。

### 51.3 `ncu` 字节分解（`B8 T2048 H8`，同一份操作数）

| 变体 | L2 字节 | srcunit 读扇区 | srcunit 写扇区 | l1tex ld 扇区 | l1tex st 扇区 | DRAM 读 | Duration | L2 SOL |
|---|---|---|---|---|---|---|---|---|
| `base` | **3.02 GB** | 59.77 M | 9.34 M | 44.83 M | 12.71 M | 1.28 GB | 519.5 µs | 88.8% |
| `nohdh` | 0.87 GB | 9.44 M | 9.63 M | — | 12.71 M | 0.20 GB | 323.5 µs | 56.6% |
| `nodov` | 2.61 GB | 50.33 M | 9.21 M | — | 12.71 M | 1.07 GB | 448.2 µs | 89.0% |
| `nohdh_nodov` | **0.44 GB** | **0** | 9.04 M | — | 12.71 M | ~0 | 313.3 µs | 34.6% |
| `vecst` | **2.89 GB** | 59.77 M | **6.42 M** | 44.83 M | **6.42 M** | 1.28 GB | 514.3 µs | 85.0% |
| `noasync` | 2.39 GB | **39.85 M** | 9.57 M | **39.85 M** | 12.71 M | 1.28 GB | 627.9 µs | 77.8% |
| `noasync_vecst` | **2.23 GB** | **39.85 M** | **6.42 M** | **39.85 M** | **6.42 M** | 1.28 GB | 605.9 µs | 73.1% |
| Triton identical-op 核 | **2.23 GB** | 39.85 M | 6.42 M | 39.85 M | 6.42 M | 1.28 GB | ~551 µs | 77.8% |

（1 sector = 32 B；`srcunit_tex` = L1TEX 发到 L2 的扇区数。）

### 51.4 拆解：两条独立的放大

**读侧 +0.637 GB（占 0.79 GB 的 ~81%）**——`h/dh` 与 `do/vn/dv` **各自**都正好是 **1.50×**：
`h/dh` 消耗 `59.77−9.44 = 50.33 M` 扇区 = 1.61 GB，而 `h/dh` 数据只有 1.07 GB（33.5 M 扇区）；
`do/vn/dv` 消耗 `59.77−50.33 = 9.44 M` 扇区 = 0.302 GB，而数据 0.201 GB（6.29 M 扇区）。DRAM 读
两侧都是 1.28 GB **不变**，故多出的扇区是 **L2 命中（re-touch）**，不是 DRAM 流量。关键对照是 `noasync`：
**换成同样的 16B、同样「每 warp 跨 4 行、每行 128 B」的普通 `float4` load**（`LDGSTS.E.BYPASS.128` →
`LD[.128]`），请求数**完全相同**（2,490,368），但 L1TEX 扇区从 **18/请求降到 16/请求**、srcunit 读从
59.77 M 降到 **39.85 M（= DRAM 扇区，1.0×）**。即这 1.5× 是 **`cp.async` 的 `.BYPASS` 路径**属性，
**不是** 16B 宽度、也**不是**「每 warp 跨 4 行」的访问模式（后者在普通 load 下 1.0× 无放大）。
扇区级的精确机制（为何恰是 1.5×：L1TEX 计 18/请求、L2 srcunit 计 24/请求）**待验证**——可确认的是
BYPASS 读不经过 L1 的跨 lane 128 B 合并，把这批 128 B 对齐、512 B 步进的请求放大到 1.5×。

**写侧 +0.093 GB（占 ~19%）**——`base` 的 `l1tex st 扇区 = 12.71 M`，是输出实际字节（`dq/dk/dw` 201 MB
+ `dgk` = 6.42 M 扇区）的 **1.98×**；原因是 25 条 **32-bit `STG.E`**：4 个 lane 各写 4 B 到同 sector 的
`{0,8,16,24}` 偏移 → **每次 store 指令只覆盖半个 sector**，`vecst` 合成 `float2` 后 l1tex st 扇区立刻
**12.71 M→6.42 M**、srcunit 写 **9.34 M→6.42 M**，与 Triton **逐位相同**。

**两条叠满**（`noasync_vecst`）= **2.23 GB = Triton 的 2.23 GB**，且 `l1tex`/`srcunit` 读写四项与 Triton
**逐值相同**——闭环：0.79 GB 差**没有第三项**。

### 51.5 Part B — 计时：`vecst` 是免费的小赢，`noasync` 不可部署

`vecst` 的 CUPTI `dev`（`check_bwd_b2_l2.out.txt`）：`B8T2048H8` **519.5→514.3 µs（0.99×）**、
`B2T1024H8` 0.98×、`T512H4` **0.87×**、`T250H3` **0.89×**——与 ncu `Duration` 一致。省下的 0.13 GB
L2 写流量在大 shape 只值 ~1%（本核 DRAM 84.7% / L2 85–89% bound，省的是被 L2 吸收的写放大），
**小 shape（延迟主导、store 指令数减半）收益 10–13%**。

`noasync` 虽把 L2 字节降到 2.39 GB，`Duration` 却 **520→628 µs（+21%）**——同步 load 丢掉 async 预取
overlap，证明那 0.64 GB 的「读放大」是 **async 流水的代价，不能用普通 load 换掉**。

### 51.6 结论 / 对目标卡的设计输入

- **闭环 §49.4.3/§50.5**：B2 K2-mma 核的 L2 +0.79 GB = **读侧 cp.async `.BYPASS` +0.64 GB** +
  **写侧逐标量 32-bit store +0.13 GB**；二者可分别消融，叠满后与 Triton **逐位相同（2.23 GB）**。
- **`vecst` 是免费、逐位无损的可部署修复**：`-DB2M_VECST`（输出改 `float2` store）把写侧放大整条消掉，
  大 shape ~1%、小 shape ~10–13%。目标卡 CUDA B2 应**一开始就用向量化 store**（本专题的 B1 §35/§36
  已证逐标量 store 打满 store 队列；这里是同一课在 output 侧）。
- **读侧放大是结构项、不是旋钮**：它是 `cp.async` L1-bypass 路径的属性，换成普通 load 会因丢流水
  **净亏 21%**。真正能同时「大块搬运 + 深流水」的只有 **TMA**（`cp.async.bulk.tensor`，B1 §22–24 的路线）
  ——故 K2 形状 B2 的读侧下一杠杆＝**TMA 输入**，与 §20/§24/§49 的结论收敛。
- **别把 L2 字节当唯一指标**：本核部署 shape 是 **DRAM 84.7% + L2 85–89% 双 bound**，`vecst` 消掉写放大
  后 L2 SOL 反降到 85%（L2 不再是最紧 pipe）。省字节要落在**最紧的 pipe** 上才兑现时间（与 §50 的
  `dgk`/冲突「零兑现」同构）。

### 51.7 边界 / 下一步（诚实）

- 读侧 1.5× 的**扇区级微架构成因**未钉死（L1TEX 18/请求、L2 srcunit 24/请求的由来），只确认「`cp.async`
  BYPASS 路径属性、非 16B 宽度/访问模式」；要坐实须 SASS/PC-sampling 或换 `cp.async.ca`（4/8B，会走 L1）
  对照——**待验证**。
- `noasync*` 变体破坏流水、**可能有竞态**（`prefetch` 的同步写不再受 `__pipeline_wait_prior` 保护），
  仅用于字节归因，**不得**作正确性/部署结论。
- `vecst` 只在 `BT=16/K=V=128`、fp32/tf32 部署 mix 上测；bf16 变体与更大 `BVK` 未测。
- 读侧 TMA 端口（把 `h/dh/do/vn/dv` 改 `cp.async.bulk.tensor`）**未做**，是本核读侧下一杠杆。

## 52. 兑现 §51.6 的「读侧下一杠杆 = TMA 输入」：K2-mma 核的读放大**整条消失**且**净赢 5%**（P7-49）

§51（P7-48）把 B2 K2-mma 核相对 identical-op Triton 的「L2 字节 +35%」拆成读侧 `cp.async` `.BYPASS`
的 **+0.64 GB** 与写侧逐标量 store 的 +0.13 GB，并点名读侧的下一步——**TMA**（只有它能同时「大块搬运 +
深流水」；`noasync` 换普通 `float4` 虽同样消掉读放大却因丢流水**净亏 21%**）。本轮就是把 B2 GEMM 核的
操作数路径从 `cp.async` 换成 **`cp.async.bulk.tensor`（TMA）+ SWIZZLE_128B**，验证「读放大消失」且
**兑现成时间**。答案是：读放大**整条消失**（srcunit 读 59.77M→39.85M = DRAM 扇区数 1.0×）、且**没有**
`noasync` 的流水代价——部署主 shape 端到端**快 5%**（516.7→491.2 µs）。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_TMA`）+ `harness/check_bwd_b2_tma.py`
（`harness/check_bwd_b2_tma.out.txt`，空闲卡）+ `harness/probe_b2_tma_ncu.sh`
（`harness/probe_b2_tma_ncu.out.txt`）。

### 52.1 设计：TMA 输入 + 128 B swizzle + per-stage mbarrier

被隔离的对象与 §49/§50/§51 **完全相同**（同一份 `dq/dk/dw/dgk` 四条 op、同一批 fp32 操作数、`K=V=128`、
部署 `BT=16`、`B8 T2048 H8`）。只改「操作数怎么进 smem」：

- `do/vn/dv`（`[B,T,H,V]` token-major）用 **4D 张量图** `{V,T,H,B}`、box `{BVK,BT,1,1}`、
  **SWIZZLE_128B**（内维 `BVK*4 == 128 B`）；`h/dh`（`[B,NT,H,K,V]`）用 **5D 张量图**
  `{V,K,H,NT,B}`、box `{BVK,K,1,1,1}`、同样 SWIZZLE_128B（`b2_mma_probe.cu:make_map_dov/make_map_hdh`）。
- 每 stage 一个 `mbarrier`（`full[s]`，arrive count 1）；**单线程**（`threadIdx.x==0`）每 stage 发 5 条
  TMA（`expect_tx = (3·BT + 2·KD)·BVK·4`），全体 warp `mbarrier.try_wait.parity` 等满（`b2_mma_probe.cu:issue_tma`）。
  稳态每 stage 两条 `__syncthreads()`（wait 后一次、compute 后一次保护 WAR），未做 producer/consumer
  warp 特化——本核 warp 数少（8），先把「TMA 输入」这条杠杆单独验出来。
- 128 B swizzle 的行内 16 B 子块按 `(row & 7)` 异或（`Swizzle<3,4,3>`）；mma 片段读改走 `rd()`
  （`b2_mma_probe.cu:rd`，非 TMA 时为恒等）。该映射下 A/B 片段读的 bank = `t + 4·((2kk^g)&7)` 恰好铺满
  `0..31`，**无冲突**（与 §49 的 `BVP=36` padding 等价，但省掉 padding）。
- 非整除 `T` 的尾块 token 行由 **TMA OOB 零填充**自动处理（无需在 `prefetch` 里手写 `t<T` 的置零）。

TMA 的内维必须 ≤128 B，故 `-DB2M_TMA` 下强制 `BVK==32`（`static_assert`，`BVK*4=128 B`）；宿主端用
`cuTensorMapEncodeTiled`（需 `-lcuda`）。

### 52.2 Part A — 正确性：`tma` 与 `base` **逐位相同**

同操作数四组 shape（含非整除 `T=250`；torch 参考关 tf32 走真 fp32）：

| 量 | `tma` vs torch(fp32) | Triton(tf32) vs torch | **base vs tma** |
|---|---|---|---|
| `dq` | 3.58e-4 | 8.40e-4 | **max_abs = 0** |
| `dk` | 3.28e-4 | 8.18e-4 | **max_abs = 0** |
| `dw` | 3.41e-4 | 8.36e-4 | **max_abs = 0** |
| `dgk` | 1.63e-7 | 1.34e-7 | **max_abs = 0** |

`tma`/`tma_vecst` 与 `base` 的 `dq/dk/dw/dgk` **逐位相同（max_abs=0，含 `T=250`）**——同一份操作数、
同一套 `mma.sync.m16n8k8.tf32`、逐条 op 序不变，只是搬运方式不同；OOB 零填充让尾块与显式置零**逐位一致**。
vs torch 落 **tf32 底噪**（与 §49.2 完全相同），且 CUDA(tf32) 比 Triton 更贴 fp32。

### 52.3 `ncu` 字节分解（`B8 T2048 H8`，同一份操作数）

| 变体 | L2 字节 | srcunit 读扇区 | srcunit 写扇区 | l1tex ld 扇区 | l1tex st 扇区 | DRAM 读 | Duration | L2 SOL | DRAM SOL |
|---|---|---|---|---|---|---|---|---|---|
| `base` | 3.02 GB | 59.77 M | 9.34 M | 44.83 M | 12.71 M | 1.28 GB | 519.8 µs | 89.0% | 84.6% |
| `vecst` | 2.89 GB | 59.77 M | 6.42 M | 44.83 M | 6.42 M | 1.28 GB | 514.5 µs | 85.0% | 85.5% |
| `tma` | **2.35 GB** | **39.85 M** | 8.63 M | **0** | 12.71 M | 1.28 GB | **494.2 µs** | 87.5% | **89.0%** |
| `tma_vecst` | **2.25 GB** | **39.85 M** | **6.42 M** | **0** | **6.42 M** | 1.28 GB | **491.0 µs** | 84.2% | **89.6%** |
| Triton identical-op 核 | 2.23 GB | 39.85 M | 6.42 M | 39.85 M | 6.42 M | 1.28 GB | ~551 µs | 77.8% | 79.9% |

读法：

1. **读放大整条消失**：`tma` 的 srcunit 读扇区 **59.77M → 39.85M**，正好等于 DRAM 扇区数（**1.00×**）——
   与 §51.4 的 `noasync` 预测**逐值吻合**（`.BYPASS` 路径的 1.5× 是属性、非 16B 宽度/访问模式）。DRAM 读
   两侧都是 **1.28 GB 不变**，故多出的 0.64 GB 本就是 **L2 re-touch**；TMA 把它整条拿掉。
2. **且确实走 TMA 引擎、不占 LSU**：`tma` 的 `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` = **0**
   （全局读请求全部由 TMA 发），SASS 里 `base` 的 **10 × `LDGSTS`** 全换成 **10 × `UBLKCP`**（TMA）——
   这也是它**不**像 `noasync` 那样丢流水的原因。
3. **省字节落在「最紧 pipe」上、并兑现时间**：`base` 的最高 pipe 是 **L2 89.0% > DRAM 84.6%**；删掉
   0.64 GB 后 `tma` 的 L2 降到 87.5%、**DRAM 升到 89.0% 成最高 pipe**——绑定点从 L2 转到 DRAM，Duration
   **519.8→494.2 µs（0.951×）**。这正是 §51.6「省字节要落最紧 pipe 才兑现时间」的正例：读放大虽被 L2 吸收、
   不增 DRAM 字节，却**压在最紧的 L2 pipe 上**，去掉即净赚。
4. **`tma_vecst` 与 Triton 逐值**：叠上 §51 的 `vecst`（写侧 `float2`）后 L2 **2.25 GB**、srcunit 读写
   `39.85M/6.42M`、l1tex st `6.42M` 与 Triton 的 2.23 GB 只差 1%（差在 dgk/边界）；Duration **491.0 µs**，
   比 identical-op Triton 核心（~551 µs）**快 1.12×**。§51 的两条放大至此**双双消除**。

### 52.4 Part B — 计时：TMA 在所有 shape 都是净赢

CUPTI kernel-only `dev`（`check_bwd_b2_tma.out.txt`）：

| shape | `base` | `vecst` | `tma` | `tma_vecst` | `tma/base` | `tma_vecst/base` |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 516.7 | 512.5 | **491.2** | **488.8** | **0.951×** | **0.946×** |
| `B2 T1024 H8` | 71.9 | 69.9 | **68.7** | **66.6** | **0.955×** | **0.926×** |
| `T512 H4` | 9.4 | 8.2 | 8.8 | **7.5** | 0.933× | 0.794× |
| `T250 H3 (非整除)` | 8.5 | 7.6 | 8.1 | **7.3** | 0.955× | 0.857× |

- **大 shape（部署主 shape）**：`tma` 单换读侧就 **0.951×**、叠 `vecst` **0.946×**；与 §49.5 的 `base`
  最优 `BVK=64/NS=2`（499.1 µs）相比，`tma`（`BVK=32`）**还更快**（491.2）。读侧 TMA 比「扫 `BVK`」更值。
- **小 shape**：`dev` 上 `tma` 0.93–0.96×、`tma_vecst` **0.79–0.86×**（`wall` 被 ~28 µs 的 host 派发钉死，
  不反映 kernel 收益，同 §49.3「小 shape 按 `dev` 选核」）。§49.5 曾说小 shape 交给 Triton 短核；现在
  `tma_vecst` 把小 shape `dev` 压到 ~7.5 µs，是否仍切 Triton 值得重估（本核 TMA prologue 的固定段是唯一的
  延迟成分，`vecst` 的指令数减半在小 shape 最显著）。

### 52.5 结论 / 对目标卡的设计输入

- **§51.6 的「读侧下一杠杆 = TMA 输入」兑现**：`cp.async` `.BYPASS` 的 +0.64 GB L2 读放大被 TMA **整条
  消除**（srcunit 读 59.77M→39.85M=1.0×），且 TMA 走独立引擎（l1tex 全局读请求 = 0）、**保留异步流水**——
  所以是**净 +5%**而非 `noasync` 的 −21%。**目标卡 CUDA B2 的操作数路径应直接上 TMA**（与 B1 §22–24 同课）。
- **「省字节落最紧 pipe」再得正例**：读放大不增 DRAM 字节、只压 L2（`base` L2 89% > DRAM 84.6%），去掉后
  绑定点转到 DRAM、Duration −5%。**看 `ncu` 哪个 pipe 最高再决定改哪条访存**（§50 的 `dgk`/冲突是反例：省了
  却不兑现）。
- **两条 §51 修复可叠加**：`tma_vecst` = **2.25 GB**（Triton 2.23）、Duration 491 µs，比 identical-op
  Triton 核快 **1.12×**，且 `tma`/`tma_vecst` 与 `base` **逐位相同**——两条都免费、位级安全。
- **本核自身下一杠杆**：`tma_vecst` 后 DRAM SOL **89.6%**、L2 84.2%，**DRAM 成唯一最高 pipe**——本核已基本
  贴住 DRAM 地板（§47.5 的 442 µs 地板 = `tma_vecst` 491 µs 的 1.11×）。**剩下的是**：把 `dgk` 无冲突映射
  （§50 的 `-DB2M_DGKCF`，仅小 shape 值）、以及 §47.6 的**非 GEMM 段**（finalize/`dA`/`dv2`/`db` + intra 折叠，
  不在本核内）。

### 52.6 边界 / 下一步（诚实）

- `-DB2M_TMA` 因 128 B swizzle 内维约束**强绑 `BVK==32`**（`static_assert`）；§49.5 的 `BVK=64` 需拆两个
  32 列 swizzle atom 或改 `SWIZZLE_NONE`，**未做**。`B2M_NOHDH/NODOV/NOASYNC` 与本变体**不组合**。
- 本核仍是 **GEMM 段隔离核**（三条 V-归约 GEMM + `dgk`），finalize/`dA`/`dv2`/`db`/intra 折叠**未进 CUDA**
  （同 §47.6）；故这 5% 是 GEMM 段（占完整 B2 约 57%）的收益。
- bf16 变体未测（本轮刻意保持部署 fp32/tf32 mix，与 Triton 同 dtype 对照）。
- **下一步**：把本核（`-DB2M_TMA -DB2M_VECST`）作为 GEMM 段基座，补非 GEMM 段与同 CTA 内交接；并把
  `BVK=64` 的 TMA swizzle 变体补上，看能否再压 DRAM 地板那 1.11×。

## 53. 把 B2 TMA 输入从 `BVK==32` 扩到 `BVK=64`：SWIZZLE_NONE 捷径**正确但不可用**（P7-50）

§52.6 留下一处硬边界：`-DB2M_TMA` 因 128 B swizzle 的内维约束被 `static_assert` 钉死在 `BVK==32`，
而 §49.5 的编译扫描里 cp.async 的**最优分块恰恰是 `BVK=64`**（`B8 T2048 H8`：`64/2` 499.1 µs vs
`32/2` 517.3 µs）——两个收获（TMA 读侧修复 + 最优分块）此前**永远无法同时出现**。本轮兑现 §52.6
点名的两条备选里最省的一条：`BVK==64` 时改用 **`SWIZZLE_NONE`**（内维 256 B，>128 B 故不能套 128 B
swizzle；同一 codebase 在 P7-21 的 512 B 行 `E^tot` 图已证明大内维配 `SWIZZLE_NONE` 可用）。

证据：`harness/b2_mma_probe.cu`（`-DB2M_TMA` 泛化到 `BVK∈{32,64}`）+ `harness/check_bwd_b2_tma64.py`
（`harness/check_bwd_b2_tma64.out.txt`，空闲卡）+ `harness/probe_b2_tma64_ncu.sh`
（`harness/probe_b2_tma64_ncu.out.txt`）。结论先给：**数值逐位正确，但太慢（2.6–2.8×）**——根因是稠密
256 B 行布局的 smem bank 全别名，`SWIZZLE_NONE` 这条捷径**被证伪**，`BVK=64` 的 TMA 只能走 §52.6 的
另一条路（两个 128 B swizzle atom 拆分）。

### 53.1 设计：只加一个 swizzle 选择，不动任何算术

对 `b2_mma_probe.cu` 的改动是**纯布局**，mma/累加序一字未动：

- 新增编译期宏 `TMA_SWZ128 = (B2M_BVK == 32)`（`b2_mma_probe.cu:117-121`），`BVP = BVK`（TMA 下稠密、
  无 padding，`:113`），`static_assert(BVK == 32 || BVK == 64)`（`:122`）。
- `rd()` 的 swizzle 逆变换按 `TMA_SWZ128` 二选一：`BVK==32` 走 `Swizzle<3,4,3>` 字节异或，`BVK==64`
  恒等（`:148`）。
- 两张张量图（`make_map_dov`/`make_map_hdh`）的 swizzle 参数改成
  `TMA_SWZ128 ? SWIZZLE_128B : SWIZZLE_NONE`（`:460` / `:477`），box 内维随 `BVK`。

因此三条变体同源对照：`base*`（cp.async + `BVP=BVK+4` padding）、`tma32`（TMA + `SWIZZLE_128B`）、
`tma64`（TMA + `SWIZZLE_NONE`）。`expect_tx` 仍为整 box 字节数 `(3·BT+2·KD)·BVK·4`，`rd` 之外的
kernel 主体不变。

### 53.2 Part A — 正确性：`tma64` 与 `base64` **逐位相同**、且 `base32==base64`

同操作数四组 shape（`BT=16`，含非整除 `T=250`；torch 参考关 tf32 走真 fp32）：

| 量 | `base64` vs `tma64` | `base32` vs `base64` | `tma64` vs torch(fp32) | Triton(tf32) vs torch |
|---|---|---|---|---|
| `dq` | **max_abs = 0** | **0** | 3.58e-4 | 8.40e-4 |
| `dk` | **0** | **0** | 3.28e-4 | 8.18e-4 |
| `dw` | **0** | **0** | 3.41e-4 | 8.10e-4 |
| `dgk` | **0** | 2.86e-6 | 1.97e-7 | 1.31e-7 |

两件事被钉死：(1) `SWIZZLE_NONE` 的字节解码**无误**（`tma64` 与 cp.async 的 `base64` 逐位相同，四位全 0）；
(2) **`base32==base64` 的 `dq/dk/dw` 逐位相同**——因为 mma 对收缩维的 k8 分块累加序不随 `BVK` 变（`BVK=32`
的 4 个 tile × 4 个 kk 与 `BVK=64` 的 2 个 tile × 8 个 kk 拼出同一条 `k=0,8,…,120` 序列）；只有 `dgk`
的标量列归约分组随 tile 拆分不同，差 2.86e-6 fp32 舍入（与 §49/§50 的稳定性一致）。`tma64` 本身仍落
tf32 底噪（3.6e-4），且**比 Triton 更贴真 fp32**（同 §49.2 的 CUDA-vs-Triton 微优势）。

### 53.3 Part B — 计时：`BVK=64` 的 TMA **慢 2.6–2.8×**

CUPTI `dev`（kernel-only，`B8 T2048 H8` 主 shape）：

| 变体 | `dev` (µs) | vs `base32` | vs `base64` |
|---|---|---|---|
| `base32`（cp.async, BVK=32） | 517.8 | 1.000× | 1.038× |
| `tma32`（TMA, BVK=32） | 491.4 | **0.949×** | 0.985× |
| `base64`（cp.async, BVK=64） | 499.0 | 0.964× | 1.000× |
| `tma64`（TMA, BVK=64） | **1415.5** | **2.734×** | **2.837×** |
| `tma64v`（+vecst） | 1388.8 | 2.682× | 2.783× |

四组 shape 一致：`base64/base32` 0.92–1.03×（复现 §49.5 的「64 略优」）、`tma32/base32` 0.92–0.96×
（复现 §52 的「TMA 净赢 5%」）、而 **`tma64/base64` 恒 2.5–3.0×**。即：`SWIZZLE_NONE` 不但没吃到
「最优分块」的 3%，反而把 TMA 的全部收益倒赔进去还多亏 ~2.8×。

### 53.4 `ncu` — 根因：稠密 256 B 行的 smem bank **全别名**（不是字节、不是带宽）

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_tma64_ncu.out.txt`）：

| 变体 | L2 bytes | srcunit 读 | l1tex 全局读 | **bank conflicts** | shared-load wavefronts | DRAM SOL | Duration |
|---|---|---|---|---|---|---|---|
| `base32` | 3.02 GB | 59.77M | 44.83M | 8.81M | 38.17M | 84.5% | 520.8 µs |
| `tma32` | 2.35 GB | 39.85M | **0** | 8.69M | 38.05M | 89.1% | 493.6 µs |
| `base64` | 2.96 GB | 59.77M | 44.83M | 8.39M | 37.75M | 88.2% | 499.7 µs |
| `tma64` | 2.35 GB | 39.85M | **0** | **272.6M** | **302.0M** | 30.0% | **1470 µs** |

- **读侧修复仍然在**：`tma64` 的 L2 bytes 2.35 GB = `tma32`、srcunit 读 39.85M（=DRAM 扇区 1.0×）、
  `l1tex__t_sectors_pipe_lsu_mem_global_op_ld = 0`（SASS `LDGSTS`→`UBLKCP`）——即 §52 的 `.BYPASS`
  读放大在 `BVK=64` 下同样消掉了，DRAM 读 1.28 GB 不变。
- **但 bank conflict 爆了 32×**（8.39M→**272.6M**），shared-load wavefront 同步涨 8×（37.7M→302.0M），
  DRAM SOL 掉到 **30%**、`sm__throughput` 掉到 8.9%。机制：`SWIZZLE_NONE` 稠密行 stride = 256 B =
  **64 个 float**，而 bank = `word_addr & 31`，`64 ≡ 0 (mod 32)` ⟹ 对固定列 `col`，**128 个 `h` 行全部
  落同一 bank**。而 mma 片段读恰是「固定列扫行」（`krow = warp*16 + …`、`col` 固定），于是 128-way
  别名。`base` 用 `BVP=BVK+4` 把行 stride 打成 68（mod 32 = 4，错开 bank），`tma32` 用 128 B swizzle 的
  行内异或把 bank 铺满 `0..31`——两条路都消失，唯独 `SWIZZLE_NONE` 的 256 B 行两者都占不到。

### 53.5 结论 / 对目标卡的设计输入

- **`SWIZZLE_NONE` 不是 `BVK=64` TMA 的可行捷径**：数值**逐位正确**（§53.2）但 **~2.8× 慢**（§53.3），
  根因是稠密 256 B 行让 smem bank 全别名（§53.4）。这与 §20/§50「省字节/消冲突要落对 pipe」同课——
  这里连「省到的字节」（L2 2.35 GB）都在 bank 重放里被淹没。
- **要 `TMA + BVK=64` 只能走 §52.6 的另一条路**：把每个操作数 tile 拆成**两个 128 B swizzle `[·,32]`
  atom**（各写一块独立 smem 子缓冲，`rd` 按 `col>>5` 选 atom 再做 `Swizzle<3,4,3>` 解码），而非把
  256 B 行整块交给 `SWIZZLE_NONE`。这条拆分**未做**，是本结果留下的**唯一**待验证路径。
- **当前可用最优组合不变**：cp.async 侧取 `BVK=64`（`base64` 0.964× `base32`），TMA 侧取 `BVK=32`
  （`tma32` 0.949×）——**分别最优**，二者尚未叠加。`tma32 ≈ base64`（0.985×）说明「TMA@32」与
  「cp.async@64」基本同档，真正要再进一步必须把两者合起来（即上面的 atom 拆分）。
- **方法学**：跨 `BVK` 对照时可**先用逐位校验**判断布局解码是否正确（`base32==base64` 逐位相同即证
  k8 累加序不随 `BVK` 变），再把时间/`ncu` 差异单独归因到布局——避免把「解码错」误判成「布局慢」。
  （本轮初版即因 `#if` 误用了 `constexpr` 而非宏，把 `BVK=32` 的 `rd` 也走成稠密而短暂出现
  「`tma32≠tma64`」，逐位校验立刻暴露。）

## 54. 兑现 §53.5 的**唯一**待验证路径——`TMA + BVK=64` 用**两个 128 B swizzle atom** 拆分：正确、无冲突、且是全程最快（P7-51）

§53（P7-50）把 `TMA + BVK=64` 走成 **`SWIZZLE_NONE`**（稠密 256 B 行）这条捷径，结论是**逐位正确但 2.6–2.8×
慢**：稠密行 stride 256 B = 64 个 float，`64 ≡ 0 (mod 32)`，对固定列（mma 片段的读法恰是「固定列扫行」）128
个 `h` 行**全部落同一 bank**（`ncu` 实测 bank conflict 8.39M→**272.6M**、Duration 520→1470 µs）。§53.5 因此
留下**唯一**一条未做路径：**把每个操作数 tile 拆成两个 128 B swizzle 的 `[·,32]` atom**（各写一块独立子缓冲、
`rd` 按 `col>>5` 选 atom），而不是把 256 B 行整块交给 `SWIZZLE_NONE`。本轮兑现它，答案是全省：**逐位正确、
读侧修复仍在、bank conflict 不再爆、并且把 §49.5 的「最优分块 BVK=64」与 §52 的「TMA 读侧修复」第一次叠加
成全程最快的一档**（`atom64/base32` dev **0.853–0.933×**，`B8 T2048 H8` ncu Duration 518.9→**485.9 µs**）。

证据：`harness/b2_mma_probe.cu`（TMA 路径改为 atom-major `[NAT][rows][32]` 布局）+ `harness/check_bwd_b2_tma_atom.py`
（`harness/check_bwd_b2_tma_atom.out.txt`，空闲卡）+ `harness/probe_b2_tma_atom_ncu.sh`
（`harness/probe_b2_tma_atom_ncu.out.txt`）。

### 54.1 设计：只改布局，算术一字未动

`b2_mma_probe.cu` 的 TMA 分支从「每个操作数一块 `[rows][BVK]` 子缓冲、一张 box 宽 `BVK` 的图」改成
**atom-major**：`NAT = BVK/32` 个 `[rows][32]` 子缓冲，每个 32-float（128 B）宽的 atom 由**自己的 TMA box**
（内维恒为 32、`SWIZZLE_128B`）搬入，`h/dh` 的 atom stride = `rows*128` 字节。

- `Stage` 里五个操作数都开成 `[NAT][rows][32]`（`BVK==32` 时 `NAT=1`，与 §52 的 P7-49 完全同构）（`b2_mma_probe.cu:130-142`）。
- `rd(buf,row,col,rows)`：`atom=col>>5`、`cc=col&31`，按 `[NAT][rows][32]` 算字节偏移后再做 `Swizzle<3,4,3>`
  解码 `b ^= ((b>>7)&7)<<4`（`:146-165`）。**为什么 atom 内部解码与 `BVK=32` 逐位同式**：atom 的字节 stride
  `rows*128` 是 **8 行的整数倍**（`BT=16`、`KD=128` 都含 8），故 `(atom*rows+row)&7 == row&7`——每个 atom 自己
  起一段 128 B swizzle，互不干扰。
- `issue_tma` 对每个 atom 各发 5 条 `cp.async.bulk.tensor`（`BVK=64` 时每 stage **10 条**，SASS `UBLKCP`
  10→**20**），`expect_tx` 仍是整 box 字节数 `(3·BT+2·KD)·BVK·4`（`:260-278`）。
- 两张张量图的 box 内维改成恒为 **32**、swizzle 恒 **`SWIZZLE_128B`**（`:463-485`）——`SWIZZLE_NONE` 分支被删除。

因此三条可用变体同源对照：`base*`（cp.async + `BVP=BVK+4` padding）、`tma32`（TMA，NAT=1）、`tma_atom64`
（TMA，NAT=2、两个 128 B atom）。

### 54.2 Part A — 正确性：`atom64` 与 `base64` **逐位相同**、与 `tma32` 的 `dq/dk/dw` 也逐位

同操作数四组 shape（`BT=16`，含非整除 `T=250`；torch 参考关 tf32 走真 fp32）：

| 量 | `atom64` vs `base64` | `atom64` vs `tma32` | `base32` vs `base64` | `atom64` vs torch(fp32) |
|---|---|---|---|---|
| `dq` | **0** | **0** | **0** | 3.58e-4 |
| `dk` | **0** | **0** | **0** | 3.28e-4 |
| `dw` | **0** | **0** | **0** | 3.41e-4 |
| `dgk` | **0** | 2.86e-6 | 2.86e-6 | 1.97e-7 |

三件事被钉死：(1) atom 布局的字节解码**无误**（`atom64`≡`base64` 四位全 0）；(2) `atom64`≡`tma32` 的
`dq/dk/dw` 逐位（证 **`BVK=32` 的单 atom 与 `BVK=64` 的双 atom 拼出同一个 `k=0,8,…,120` 收缩序列**，mma k8
累加序不随 atom 数变）；(3) `atom64` 自身仍落 tf32 底噪（3.6e-4），且**比 identical-op Triton 核更贴真 fp32**
（triton 8.4e-4，同 §49.2）。唯一非零是 `dgk` 的**标量列归约分组**（tile 拆分不同）差 2.86e-6 fp32 舍入。

### 54.3 Part B — 计时：atom 拆分让 TMA@64 成为**全程最快**，两收获首次叠加

CUPTI `dev`（kernel-only，µs；`wall` 含 host 派发，小 shape 被 ~20µs 尾税主导）：

| shape | `base32` | `tma32` | `base64` | `atom64` | `atom64v`(+vecst) | `atom64/base32` | `atom64/base64` |
|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 519.4 | 494.0 | 496.4 | **484.6** | 483.0 | **0.933×** | 0.976× |
| `B2 T1024 H8` | 70.7 | 68.3 | 71.6 | 68.9 | **68.0** | 0.974× | 0.962× |
| `T512 H4` | 9.4 | 8.9 | 9.0 | 8.3 | **6.9** | 0.887× | 0.927× |
| `T250 H3`（非整除） | 8.5 | 8.3 | 7.4 | 7.2 | **6.3** | 0.853× | 0.972× |

- **`base64` ≈ `base32`（0.956–1.14×）复现 §49.5 的「64 略优」**；`tma32/base32` 0.951× 复现 §52 的「TMA
  净赢 5%」；两者的乘积就该在 0.91× 附近——**`atom64/base32` = 0.853–0.974×（均值 ~0.91）正是这个乘积**，
  即两个此前**永不共现**的收获第一次叠加成功（对照 §53.3：`SWIZZLE_NONE` 版本 `tma64/base64` = 2.5–3.0×）。
- 加 `-DB2M_VECST`（写侧 `float2`，§51 的免费项）后再降一档（`T512 H4` 6.9µs）。

### 54.4 `ncu` — 读侧修复仍在，且 **§53.4 的 32× bank-conflict 爆炸不再出现**

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_tma_atom_ncu.out.txt`）：

| 变体 | L2 bytes | srcunit 读 | l1tex 全局读 | **bank conflicts** | shared-load wavefronts | DRAM SOL | sm | Duration |
|---|---|---|---|---|---|---|---|---|
| `base32` | 3.02 GB | 59.77M | 44.83M | 8.79M | 38.15M | 84.8% | 31.2% | 518.9 µs |
| `tma32` | 2.35 GB | 39.85M | **0** | 8.74M | 38.10M | 89.1% | 38.1% | 494.0 µs |
| `base64` | 2.96 GB | 59.77M | 44.83M | 8.39M | 37.75M | 88.9% | 31.9% | 495.2 µs |
| `atom64` | 2.39 GB | 39.85M | **0** | **8.39M** | 37.75M | **90.6%** | 34.7% | **485.9 µs** |
| （对照 §53.4 `tma64`=`SWIZZLE_NONE`） | 2.35 GB | 39.85M | 0 | **272.6M** | 302.0M | 30.0% | 8.9% | 1470 µs |

- **读侧修复整条保留**：`atom64` 的 srcunit 读 39.85M（= DRAM 扇区数 1.0×）、`l1tex__t_sectors_pipe_lsu_mem_global_op_ld`=0
  （SASS `LDGSTS`→`UBLKCP`，`base64` 10 条 → `atom64` 20 条，即 10 个 box × 2 atom）。
- **bank conflict 与 `base64` 逐值相同（8.39M），不再是 `SWIZZLE_NONE` 的 272.6M**——这 8.39M 是 §50 已定位的
  `dgk` 列归约冲突（与 mma 片段读/TMA 布局无关，§50.1 的 `-DB2M_NODGK` 可归零），**TMA 的 128 B swizzle 读
  本就不产生额外冲突**。
- 结果 `atom64` 的 DRAM SOL 最高（90.6%，已贴 DRAM 地板）、Duration 最短（485.9 µs）。

### 54.5 结论 / 对目标卡的设计输入

- **§53.5 的唯一待验证路径成立**：`TMA + BVK=64` 用**两个 128 B swizzle atom 拆分**后，数值**逐位正确**、
  读侧修复保留、且是本次四档里**最快**的一档。至此「TMA 读侧修复」与「cp.async 的 BVK=64 最优分块」**可以
  叠加**，不再二选一（§53.5 说的「当前可用最优 = cp.async@64 或 TMA@32、二者尚未叠加」被本条关闭）。
- **对目标卡 CUDA B2 的设计输入（更新 §52.5/§53.5）**：操作数路径**直接上 TMA**；`BVK` 想取 64（最优分块）
  时，**用 `NAT=BVK/32` 个 128 B swizzle atom 拆分**，**不要**用 `SWIZZLE_NONE` 稠密大行（后者 bank 全别名）。
  算法上这只是「把 box 内维砍到 128 B、多发几条 TMA」，不触碰 mma/累加序，故可与 `-DB2M_VECST`（向量化 store）
  无冲突叠加。
- **方法学**：跨 atom 数/布局对照**先做逐位校验**（`atom64≡base64` 且 `atom64≡tma32` 的 `dq/dk/dw`）再谈时间，
  可把「布局解码错」与「布局慢」分开（同 §53.5 教训）。
- **边界（诚实）**：本轮只覆盖 `BVK∈{32,64}`、fp32 操作数、`BT=16`；`BVK=128`（4 atom）未测；`atom64` 的 L2
  2.39 GB 比 `tma32` 的 2.35 GB 略高（0.04 GB，多出的 atom 边界 L2 re-touch），未逐项归因。

## 55. 「K2 形状 CUDA B2」补非 GEMM 段第一刀：**输出支路 `dv2 = beta·(A@dv)`** 落进 CUDA，数值落 tf32 底噪、只 +12%（P7-52）

§49.7（P7-46）把 K2 形状 CUDA B2 的下一步写成「把本核作为 GEMM 段基座，**向其中补非 GEMM 段（同 CTA 内交接
`dA`/`dv2`/`db`）**」。§50–§54（P7-47…51）随后把 GEMM 段的访存路径（`cp.async`→TMA、`BVK=64` atom 拆分、
向量化 store）收口到贴 DRAM 地板，但**非 GEMM 段一直没进 CUDA**——§49.6 里「B2 全核 899 µs，本核只 517 µs，
另 ~380 µs 是 GEMM 之外」还一直是差值推断。本轮落**第一刀**：B2 的 `i_k==0` **输出支路**

    dvb[o_A, v] = sum_{o_t} Akk[o_t, o_A] · dv[o_t, v]   (== A @ dv)
    dv2[o_A, v] = beta[o_A] · dvb[o_A, v]

它是 B2 里**唯一一个不依赖门控链的完整输出**（torch golden / FLA 在 `i_k==0` 分支算完 `dv2` 后**从不改写**
——`check_bwd_dqkg_kernel.py:294/297`），故可以**独立可证**。答案是：**逐位对得上自己的 GEMM 核心、落 tf32
底噪，且只给核心加 12–25%**。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_OUT` + 符号 `b2out_launch`）+ `harness/check_bwd_b2_out.py`
（`harness/check_bwd_b2_out.out.txt`，空闲卡）+ `harness/ncu_b2_out_driver.py` / `harness/probe_b2_out_ncu.sh`
（`harness/probe_b2_out_ncu.out.txt`）。

### 55.1 设计：为什么这段**不需要任何 smem 转置**（与三条 V-归约 GEMM 的关键差别）

核心的三条 GEMM（`dq/dk/dw`）收缩的是 **V**、输出行是 chunk 行 `o_t`；而输出支路 `A@dv` **收缩的是 chunk
行 `o_t`、输出行又是 chunk 行 `o_A`**。这带来两个便利：

- **A 操作数**：`Akk`（`[B,T,H,BT]`，`o_A` 连续）= `A[o_A, o_t]`，喂 mma A 片段只需**把 `(o_A,o_t)` 索引
  对调**——`a0=Akk(o_t=kc+t, o_A=g)`、`a1=Akk(o_t=kc+t, o_A=g+8)` …（`b2_mma_probe.cu:448-451`），**读的是
  原存储、不做转置**（`a_base` 见 `:334`）。非整除尾块按 `(i_t*BT + o_t) < T` 屏蔽（`:448-451`）。
- **B 操作数**：**直接复用 GEMM 段已经 stage 好的 `dv` tile**（`[BT,BVK]`，`:456-458`）——B 的 n 维是 `v`、
  收缩维是 `o_t`，而 staged `dv` 恰好是 `[o_t][v]`，`rd()` 按 `(o_t, v)` 读即可。**因此输出支路不新增任何
  `dv` 字节**。
- **V 列按 warp 私有**：每个 warp 仍拥有 `[w·16, w·16+16)` 的 V 列；一个 stage（`BVK=64`）装 64 列，故 warp
  只在**包含自己列的那一个 stage** 上做这 4 条 mma（`local0 = warp*BW - it*BVK` 守卫，`:441-444`），
  `BVK=32` 时占 2 个 stage。累加器 `acc2[2][4]`（`:329`）是 warp 私有的 `[16(o_A),16(v)]` 小 tile。
- **`beta[o_A]`**：行 `o_A` 与 token `i_t*BT+o_A` 同轴，直接读 global（`:530/534`）。

新增访存只有 **A 的小读**（`BT×BT=256` float/块）与 `beta`；输出新增 `dv2`（完整）与 `dvb`（中间量，供直接
核对 `A@dv`）两个 `[B,T,H,V]` 张量。

### 55.2 Part A — 正确性：`dv2`/`dvb` 落 tf32 底噪，且三种 CUDA 布局**逐位相同**

同操作数四组 shape（`BT=16`，含非整除 `T=250`；torch 参考关 tf32 走真 fp32）：

| 量 | `out` vs torch | `out_tma` vs torch | `out_tma64` vs torch | Triton B2 `dv2` vs torch | `out` vs Triton | `out` vs `out_tma64` |
|---|---|---|---|---|---|---|
| `dv2` | 3.63e-4 | 3.63e-4 | 3.63e-4 | 5.96e-4 | 8.78e-4 | **0** |
| `dvb` | 4.40e-4 | 4.40e-4 | 4.40e-4 | — | — | **0** |

（表内为四组 shape 的最坏 `max_rel`；`ref_max≈0.36–0.45`。）

三件事被钉死：(1) **CUDA 输出支路代数无误**——`dv2`/`dvb` 落 tf32 底噪（`~3.6e-4`），且**比 identical-op
Triton B2 更贴真 fp32**（Triton `dv2` 5.96e-4，同 §49.2/§54.2）；(2) 与 Triton 的差 `8.78e-4` 是**两侧 tf32
的分组差异**（CUDA m16n8k8 逐 k8 累加 vs Triton 的收缩序），非公式错；(3) **`out`≡`out_tma64` 逐位相同
（`max_abs=0`）**——非 TMA / TMA@32 / TMA@64（atom 拆分）三种 stage 布局对输出支路**零影响**，等于把 §54 的
布局解码正确性在**新的一段算术**上再证一遍。

### 55.3 Part B — 计时：输出支路只给 GEMM 核心加 12–25%，且仍 DRAM-bound

CUPTI `dev`（kernel-only，µs；`out` = `-DB2M_OUT` 非 TMA、`core_tma` = 只有 GEMM 的 TMA 基座）：

| shape | `core` | `core_tma` | `out` | `out_tma` | `out_tma64` | 全 Triton B2 | `out/core` | `out_tma/core_tma` |
|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 517.4 | 492.3 | 577.9 | 555.0 | 623.2 | 900.9 | 1.12× | 1.13× |
| `B2 T1024 H8` | 70.8 | 68.1 | 80.1 | 77.4 | 89.4 | 121.7 | 1.13× | 1.14× |
| `T512 H4` | 9.4 | 8.8 | 11.5 | 11.0 | 10.7 | 14.7 | 1.23× | 1.25× |
| `T250 H3`（非整除） | 8.6 | 8.4 | 10.0 | 9.8 | 9.3 | 14.0 | 1.16× | 1.17× |

- **非 GEMM 段第一刀很便宜**：加进 GEMM 核心只 **+12–25%**（大 shape +13%，小 shape 因固定开销占比高
  +17–25%）。加完输出支路的 CUDA 核（含 GEMM）**已到全 Triton B2 的 0.62–0.74×**（`B8` 555 vs 901 µs）。
- `out_tma` 比非 TMA `out` 略快（§52 的 TMA 读侧收益仍在）；但 **`out_tma64` 反而回退**（`B8` 623 µs）——`BVK=64`
  时每个 warp 只在一个 stage 上有活，其余 stage 的 warp 空转于输出支路（4 条 mma），`BVK=32` 摊得更平。故
  **输出支路偏好小 `BVK`**（与 GEMM 段偏好 `BVK=64` 相反，`out_tma` 非 TMA 是非整除小 shape 也稳）。

### 55.4 `ncu` — 增量是 **+130 MB DRAM 写（`dv2`+`dvb`）**，仍贴 DRAM 地板；不引入新瓶颈

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_out_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | DRAM SOL | L2 bytes | Duration | tensor% | occ | bank conflicts | l1tex 全局读 |
|---|---|---|---|---|---|---|---|---|---|
| `core_tma`（只有 GEMM） | 1.28 GB | 199.3 MB | 89.17% | 2.35 GB | 493.2 µs | 8.10% | 24.63% | 8.73 M | **0** |
| `out_tma`（+ 输出支路） | 1.29 GB | **329.9 MB** | 87.04% | 2.66 GB | 553.6 µs | 7.52% | 24.63% | 9.41 M | 4.19 M |

- **时间增量 = 写字节增量**：DRAM 写 `+130.6 MB`（= `dv2`+`dvb` 两个 `[B,T,H,V]` fp32 = `2×67 MB`），
  `Duration +60.4 µs`，二者之比 **≈ 2.16 TB/s**，即新增的那段**仍在冲 HBM 带宽**——`out_tma` DRAM SOL 仍
  **87%**、占用率/tensor 密度**一动不动**（24.63% / 7.52%），**没有把核从带宽受限打成延迟受限**。
- 新增 global 读＝**A 的 4.19 M sectors（≈134 MB L2）**（`core_tma` 为 0，因 GEMM 段 A 不参与）；
  `beta` 贡献可忽略。bank conflict `8.73→9.41 M`（+0.68 M，来自 `A@dv` 的两条 B-fragment 读），相对
  §50 的 8.39 M `dgk` 冲突是小头、非新瓶颈。
- SASS：`out_tma` `LDS` 88→96、`STG` 25→**41**（+16 = `dv2`/`dvb` 的标量 store；本例 store 是 32-bit
  标量，接 §51 的 `-DB2M_VECST` 可向量化——本轮输出支路未接该开关，留给生产形态）。

### 55.5 结论 / 对目标卡的设计输入

- **§49.7 的「补非 GEMM 段」第一刀成立**：B2 输出支路 `dv2 = beta·(A@dv)` 进 CUDA 后**独立可证**（完整输出、落
  tf32 底噪），**不新增 `dv` 字节**（复用 GEMM 段 staged 的 `dv`），成本仅 **+12–25%**、仍 DRAM-bound。关键
  使能是**索引重映射**——`A@dv` 的收缩维是 chunk 行而非 V，故 A/B 操作数都**直接按原存储读、无 smem 转置**。
- **设计输入**：目标卡 CUDA B2 里，凡「收缩维落在 chunk 行（`BT`）」的段（`A@dv`、`A@dw`、`dv@v^T`、最终的
  `A^T diag(beta) A^T`）都可照此**原地读 `Akk` 片段**完成；「收缩维落在 V」的段（三条主 GEMM）则走 staged
  `dv/do/vn` 的 B 操作数。两类段的操作数来源不同但共享同一份 stage。
- **剩余头寸仍在门控链**：本轮只做 `i_k==0` 输出支路；§49.6 那 ~380 µs 里更大的一块是**门控链 + `dkgb=A@dw` +
  `dA` 的 `dw@kg^T` 与 `A^T diag A^T` + `db`**（需 `q/k/g`/`A` 与跨 warp 归约）。下一步＝在 GEMM 段 + 本输出
  支路之上补**门控链**（同为 chunk-行收缩，可复用同一套索引重映射），把 B2 全核搬进 CUDA。
- **方法学**：「输出支路进 CUDA」的正确性**先按完整输出 `dv2` 对齐 torch 与 Triton，再证三种 stage 布局
  （非 TMA/TMA32/TMA64）逐位相同**——把「算术错」与「布局解码错」分开（同 §53.5/§54.5）。

### 55.6 边界 / 下一步（诚实）

- 本轮**只**落 `i_k==0` 输出支路（`dv2` 完整、`dvb` 中间量）；`dA`/`db` 的**后处理项**（`dw@kg^T`、`A^T diag A^T`、
  `dkgb`）与**门控链**（`dg`、`dk` 的 `exp2(g)` 标定）**未进 CUDA**（同 §49.7 未关闭）。
- **同 CTA 交接尚未做**：本段仍把 `dv2`/`dvb` 落盘再返回，未与 GEMM 段在同一寄存器生命周期里交接（§47.6 的
  「同 CTA 交接」目标）；本轮只证「能算对、加得便宜」。
- 只覆盖 `K=V=128`、`BT=16`、fp32/tf32；`out` store 用标量 32-bit（`-DB2M_VECST` 未接到本段）；`beta` 未做
  bf16。这些都不影响本节结论（增量机制＝写字节），但生产形态应接向量化 store。

## 56. 「K2 形状 CUDA B2」补非 GEMM 段第二刀：**`dkgb = A @ (-dw)`** 落进 CUDA，数值落 tf32 底噪、只 +10%（P7-53）

§55.6（P7-52）把 B2 非 GEMM 段的剩余工作列成「`dA`/`db` 的后处理项（`dw@kg^T`、`A^T diag A^T`、`dkgb`）
与**门控链**（`dg`、`dk` 的 `exp2(g)` 标定）」。§55 只落了 `i_k==0` 输出支路 `dvb=A@dv`。本轮落**第二刀**：
同一个 `i_k==0` 分支里紧邻的 **chunk-行收缩**段

    dkgb[o_A, k] = sum_{o_t} A[o_A, o_t] · (-dw[o_t, k])        (== A @ (-dw))

它对应 FLA / torch 的 `chunk_bwd.py:305-307`（`b_dw = -b_dw` 后 `b_dkgb = einsum("bhtu,bhuk->bhtk", b_A, b_dw)`，
`check_bwd_dqkg_kernel.py:180-182`）。`dkgb` 是**中间量**（下游喂 `db`、`dg`、`dk`、`dA`），故与 §55 暴露 `dvb`
一样，这里把它单独落盘以**独立核对这段 GEMM**。答案是：**逐位对得上自己的 GEMM 核心、落 tf32 底噪，且只给核心
加 10–13%**，与 §55 的输出支路同阶。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_DKGB` + 符号 `b2dkgb_launch`）+ `harness/check_bwd_b2_dkgb.py`
（`harness/check_bwd_b2_dkgb.out.txt`，空闲卡）+ `harness/ncu_b2_dkgb_driver.py` / `harness/probe_b2_dkgb_ncu.sh`
（`harness/probe_b2_dkgb_ncu.out.txt`）。

### 56.1 设计：与 §55 输出支路**同构**，只换 B 操作数并取负

§55.1 已讲清为什么「收缩维落在 chunk 行 `o_t`、输出行也是 chunk 行 `o_A`」的段最便宜：`Akk`（`[B,T,H,BT]`，
`o_A` 连续）喂 mma 的 **A 操作数只需索引重映射、无 smem 转置**。`A @ (-dw)` 的收缩维同样是 `o_t`、输出行同样是
`o_A`，所以**与输出支路共用同一套 A-fragment 读法**（`b2_mma_probe.cu:516-519` vs `:546-549`，逐字相同）；唯一
差别是：

- **B 操作数换成 `dw`**：三条主 GEMM 里 `dw = dv @ h`（收缩维是 V）是**输出**，这里把它当成 `A @ (-dw)` 的
  B 操作数（收缩维 `o_t`、自由维 `k`）。为独立可证，本段把 `dw` 作为**新操作数**经同一条流水 stage 搬进 smem
  （新增 `Stage::dwm`，`[BT, BVK]`，k-tiled；TMA 复用 `make_map_dov`，内维 `KD==VD==128`）。
- **取负在 B-fragment**：`b_dw = -b_dw`（`check_bwd_dqkg_kernel.py:180`），故读 B-fragment 时取反
  （`b2_mma_probe.cu:555-556` 的 `-rd(...)`），A-fragment 不动。
- **warp 列归属不变**：warp `w` 仍拥有输出列 `[w·16, w·16+16)`（这里输出列是 `k`），只在包含自己列的 stage 上做
  那 4 条 mma（`local0 = warp*BW - it*BVK` 守卫，同 `:510`/`:540`）。
- **`dw` 是 k-tiled、V-tile 也 k-tiled**：`VD==KD==128`，故 stage 粒度与三条主 GEMM 完全一致。

> 注：**同 CTA 交接尚未做**——本段仍把 `dw` 从 global 读进来、`dkgb` 落盘再返回；生产形态应让 GEMM 段
> 把 `dw` 的累加器写 smem（或直接在寄存器里做 C→B 片段的 shuffle/转置），省掉这一读一写。本轮只证「能算对、
> 加得便宜」。这与 §55.6 的「同 CTA 交接未做」是同一条边界。

### 56.2 Part A — 正确性：`dkgb` 落 tf32 底噪，且三种 stage 布局**逐位相同**

同操作数四组 shape（`BT=16`，含非整除 `T=250`；`dw` 操作数由 **CUDA GEMM 核心**产出，故两侧吃的都是同一份
tf32 `dw`；torch 参考关 tf32 走真 fp32）：

| 量 | `dkgb` vs torch | `dkgb_tma` vs torch | `dkgb_tma64` vs torch | `dkgb` vs `dkgb_tma` | `dkgb` vs `dkgb_tma64` |
|---|---|---|---|---|---|
| `dkgb` | 3.64e-4 | 3.64e-4 | 3.64e-4 | **0** | **0** |

（表内为四组 shape 的最坏 `max_rel`；`ref_max≈2.6–3.0`，`max_abs` 最坏 1.01e-3。）

两件事被钉死：(1) **CUDA `dkgb` 代数无误**——落 tf32 底噪（`~3.6e-4`，与 §55 输出支路的 `dvb` 4.40e-4、
§49/§52/§54 的四条 op 同量级）；torch 参考严格照 `check_bwd_dqkg_kernel.py:149-150`（`b_A[o_A,o_t]=A4[o_t,o_A]`）
与 `:180-182`（取负 + `A@dw`）写。(2) **`dkgb`≡`dkgb_tma`≡`dkgb_tma64` 逐位相同（`max_abs=0`）**——非 TMA / TMA@32 /
TMA@64（atom 拆分）三种 stage 布局对这段**零影响**，等于把 §54 的布局解码正确性在**第三段新算术**上再证一遍
（§55 已在输出支路证过一次）。

### 56.3 Part B — 计时：`dkgb` 段只给 GEMM 核心加 10–13%，且仍 DRAM-bound

CUPTI `dev`（kernel-only，µs；`dkgb` = `-DB2M_DKGB` 非 TMA、`core_tma` = 只有 GEMM 的 TMA 基座）：

| shape | `core` | `core_tma` | `dkgb` | `dkgb_tma` | `dkgb_tma64` | 全 Triton B2 | `dkgb/core` | `dkgb_tma/core_tma` |
|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 517.4 | 492.7 | 569.7 | 544.1 | 610.4 | 903.7 | 1.10× | 1.10× |
| `B2 T1024 H8` | 71.2 | 68.6 | 77.8 | 75.4 | 86.5 | 121.6 | 1.09× | 1.10× |
| `T512 H4` | 9.4 | 9.0 | 10.7 | 9.8 | 9.7 | 16.1 | 1.13× | 1.09× |
| `T250 H3`（非整除） | 8.6 | 8.4 | 9.7 | 9.3 | 8.7 | 14.1 | 1.13× | 1.10× |

- **第二刀同样便宜**：加进 GEMM 核心只 **+9–13%**（比 §55 输出支路的 +12–25% 还略低，因本段只有 2 条 mma、
  无 `beta` 乘法）。加完本段的 CUDA 核**已到全 Triton B2 的 0.63–0.70×**（`B8` 544 vs 904 µs）。
- `dkgb_tma` 比非 TMA `dkgb` 略快（§52 的 TMA 读侧收益仍在）；`dkgb_tma64` 在大 shape 反而回退
  （`B8` 610 µs）——与 §55.3「输出支路偏好小 `BVK`」同因：`BVK=64` 时每个 warp 只在一个 stage 上有活，
  4 条 mma 摊不平；小 shape 因固定开销占比高、三者接近。

### 56.4 `ncu` — 增量是 **+67 MB DRAM 读（`dw`）+ 65 MB DRAM 写（`dkgb`）**，仍贴 DRAM 地板；不引入新瓶颈

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_dkgb_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | DRAM SOL | L2 bytes | Duration | tensor% | occ | bank conflicts | l1tex 全局读 |
|---|---|---|---|---|---|---|---|---|---|
| `core_tma`（只有 GEMM） | 1.28 GB | 199.19 MB | 88.93% | 2.35 GB | 494.5 µs | 8.08% | 24.62% | 8.72 M | **0** |
| `dkgb_tma`（+ `dkgb` 段） | **1.35 GB** | **264.61 MB** | 88.72% | 2.62 GB | 543.2 µs | 7.68% | 24.51% | 9.18 M | 2.10 M |

- **时间增量 ≈ 写字节 + 读字节增量**：DRAM 读 `+0.07 GB`（=`dw` 一个 `[B,T,H,K]` fp32 = 67 MB）、DRAM 写
  `+65.4 MB`（=`dkgb` 一个 `[B,T,H,K]` fp32），`Duration +48.6 µs`，`(0.07+0.065) GB / 48.6 µs ≈ 2.78 TB/s`
  ——新增段**仍在冲 HBM 带宽**：`dkgb_tma` DRAM SOL 仍 **88.7%**、L2 **88.0%**、占用率/tensor 密度**几乎不动**
  （24.51% / 7.68%），**没有把核从带宽受限打成延迟受限**。
- 新增 global 读＝**A 的 2.10 M sectors（1 个 sector/元素，A 按标量读）≈67 MB**（`core_tma` 为 0）；`dw` 经
  TMA 读（`lts__srcunit` 39.85M→42.24M）；bank conflict `8.72→9.18 M`（+0.46 M，来自 `A@dw` 的两条
  B-fragment 读），相对 §50 的 8.39 M `dgk` 冲突是小头、非新瓶颈。
- SASS：`dkgb_tma` `LDS` 88→96、`STG` 25→**33**（+8 = `dkgb` 的标量 store）、`LDGSTS/UBLKCP` 10→**12**
  （+2 = `dw` 的 TMA）。`dkgb` 的 store 与 §55 的输出支路一样是 32-bit 标量，接 §51 的 `-DB2M_VECST` 可
  向量化（本轮未接，留给生产形态）。

### 56.5 结论 / 对目标卡的设计输入

- **§55.6 的「补非 GEMM 段」第二刀成立**：B2 的 `dkgb = A @ (-dw)` 进 CUDA 后**独立可证**（落 tf32 底噪
  `3.6e-4`）、**与三条主 GEMM 共用同一份 stage 与 A-fragment 索引重映射**（`dw` 经同一条流水搬入、取负在
  B-fragment），成本仅 **+10–13%**、仍 DRAM-bound、不引入新瓶颈。
- **设计输入（更新 §55.5）**：目标卡 CUDA B2 里，凡「收缩维落在 chunk 行（`BT`）」的段（`A@dv`、`A@dw`、
  以及最终的 `A^T diag(beta) A^T`）都走**同一套「原地读 `Akk` + staged 操作数」**；`dw` 这类**中间量**在
  生产形态应在 GEMM 段**就地（smem/寄存器）交接**，而非落盘再读（本轮为独立可证而落盘，多读 67 MB、
  多写 67 MB）。
- **剩余头寸仍是门控链**：本段只补 `dkgb`；`dA` 的 `dw@kg^T` 与 `A^T diag A^T`、`db`（`sum_k dkgb·kg`，
  需跨 warp 归约）与**门控链**（`dg`、`dk` 的 `exp2(g)` 标定）**仍未进 CUDA**（§49.7/§55.6 未关闭）。
- **方法学**：「段进 CUDA」先按**完整/中间量**对齐 torch（用**同一份**上游操作数，避免上游 dtype 混入），
  再证三种 stage 布局**逐位相同**——把「算术错」与「布局解码错」分开（同 §53.5/§54.5/§55.5）。

### 56.6 边界 / 下一步（诚实）

- 本轮**只**落 `dkgb`（`dw` 作为 global 输入、`dkgb` 落盘），**同 CTA 交接未做**；`dA` 的后处理项
  （`dw@kg^T`、`A^T diag A^T`）、`db` 的 `sum_k dkgb·kg`（跨 warp 归约）与门控链未进 CUDA。
- 只覆盖 `K=V=128`、`BT=16`、fp32/tf32；`dkgb` store 用标量 32-bit（`-DB2M_VECST` 未接到本段）；`dw` 未做
  bf16。这些都不影响本节结论（增量机制＝读/写字节），但生产形态应接向量化 store 并就地交接 `dw`。

## 57. 「K2 形状 CUDA B2」补非 GEMM 段第三刀：**`dApre = dv@v^T - dw@kg^T`**（`dA` 的后处理 GEMM）落进 CUDA（P7-54）

§55.6/§56.6 把 B2 非 GEMM 段的剩余工作列成「`dA` 的后处理项（`dw@kg^T`、`A^T diag A^T`）、`db` 的
`sum_k dkgb·kg`（跨 warp 归约）与**门控链**」。§55 落了输出支路 `dvb = A@dv`、§56 落了 `dkgb = A@(-dw)`
（两者都**收缩维落在 chunk 行 `BT`**）。本轮落**第三刀**：`dA` 的**两个 GEMM 贡献**（在最终
`-A^T diag(beta) A^T` sandwich **之前**），对应 FLA / torch 的

    dApre[o_t, o_u] = sum_v dv[o_t, v] · v[o_u, v]        (base;  `chunk_bwd.py:264` `b_dA += tl.dot(b_dv, v^T)`)
                    - sum_k dw[o_t, k] · kg[o_u, k]        (`chunk_bwd.py:280-281`: `b_dw = -b_dw; b_dA += tl.dot(b_dw, kg^T)`)

（`kg = k·exp2(g)`，`dw` 是 GEMM 核心的 `dw = dv @ h` 输出；`check_bwd_dqkg_kernel.py:168/180-181`）。
**更正（P7-55 / §58）**：第二项符号为 **`-`**——FLA 在 `chunk_bwd.py:280` 先 `b_dw = -b_dw` 再累加；
§57 最初写成 `+`（其独立核对两侧一致，未暴露；只有把 sandwich 也接上、对最终 `dA` 比 FLA 时才现形）。
这两项都是**块内**运算（无跨 chunk 耦合），故与 §55 暴露 `dvb`、§56 暴露 `dkgb` 一样，把 `dApre` 单独落盘以
**独立核对这两段 GEMM**。答案是：**逐位对得上真 fp32（比前两刀的 tf32 还准）、三种 stage 布局逐位相同，但代价明显
更大（+33–85% vs 前两刀的 +10–25%）**——因为这两项的**收缩维是完整的 `V` 和 `K`（各 128）**，而非 `BT=16`。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_DA` + 符号 `b2da_launch`）+ `harness/check_bwd_b2_da.py`
（`harness/check_bwd_b2_da.out.txt`，空闲卡）+ `harness/ncu_b2_da_driver.py` / `harness/probe_b2_da_ncu.sh`
（`harness/probe_b2_da_ncu.out.txt`）。

### 57.1 设计：两个**跨 warp 的部分和**，smem 归约后再落盘

与 §55/§56 的「收缩维落在 chunk 行」不同，本段的两个收缩维**都不是 `BT`**，故 warp 列归属按**自然分区**：

- **base `dv@v^T`（收缩维 `V`）**：warp `w` 拥有 **V 列** `[w·16, w·16+16)`（输出支路同款分区，`local0 =
  warp*BW - it*BVK` 守卫，只在含自己 V 列的 stage 累加）；lane `l` 拥有 `[BT,BT]` 部分和的 8 个条目
  （`r=l>>1, c=(l&1)·8 + i, i=0..7`），做成标量 `dv[r,v]*v[c,v]` 的 16 次循环。
- **`dw@kg^T`（收缩维 `K`）**：warp `w` 拥有**输出 k 列** `[w·16, w·16+16)`（GEMM 核心同款分区），故
  `dw=accw` 的这 16 列正好是 warp 私有；把它从 mma 片段镜像到 `swall[warp]`（row-major），再与
  `skg[o_u, k]=k·exp2(g)` 做标量点积（同样 8 条目/lane）。
- **归约**：8 个 warp 各写私有 `[BT,BT]` 部分和到 `dapart[w]`，`__syncthreads()` 后每线程一个条目求和成
  `da16[BT,BT]`，再落盘 `dApre[o_t,o_u]`（行 `o_t<T`）。
- `kg` 每 chunk 载入一次（`skg[BT,KD]`，**非** V-tiled；`k,g` 为 `[B,T,H,K]` 标量读，pad 行零）。

> 注：与 §55/§56 的「A-fragment 索引重映射、无转置」不同，本段两个操作数都**不是** `Akk`，而是
> `dv/v`（`[BT,V]`）与 `dw/kg`（`[BT,K]`）；`dw@kg^T` 还需要把 `accw` 从 mma 片段**镜像到 smem** 才能按
> row-major 重用（本段用**标量 FMA** 而非 mma，见下）。

### 57.2 Part A — 正确性：`dApre` 落 **fp32 底噪**（5.3e-7），且三种 stage 布局**逐位相同**

同操作数四组 shape（`BT=16`，含非整除 `T=250`；`dw` 操作数由 **CUDA GEMM 核心**产出，故两侧吃的都是同一份
tf32 `dw`；torch 参考关 tf32 走真 fp32）：

| 量 | `da` vs torch | `da_tma` vs torch | `da_tma64` vs torch | `da` vs `da_tma` | `da` vs `da_tma64` |
|---|---|---|---|---|---|
| `dApre` | 5.31e-7 | 5.31e-7 | 5.31e-7 | **0** | **0** |

（表内为四组 shape 的最坏 `max_rel`；`ref_max≈4.3–6.3`，`max_abs` 最坏 3.34e-6。）

两件事被钉死：(1) **CUDA `dApre` 代数无误**——且落的是 **fp32 底噪 `5.3e-7`**，比 §55 `dvb`/§56 `dkgb` 的
**tf32 底噪 `3.6e-4` 低约 3 个数量级**。原因：本段的两条收缩用**标量 fp32 FMA**（不是 `mma.sync.tf32`），
所以只有 `dw` 操作数带 tf32 舍入（且参考吃同一份 `dw`）——**收缩本身是真 fp32**。这是「便宜 vs 精度」的一个
显式取舍点（下方 §57.3 代价正是它换来的）。(2) **`da`≡`da_tma`≡`da_tma64` 逐位相同（`max_abs=0`）**——非 TMA /
TMA@32 / TMA@64（atom 拆分）对这段**零影响**，把 §54 的布局解码正确性在**第三段新算术**上再证一遍
（且这段含 `st->vv` 这个**新 staged 操作数**，等于把 TMA 的 `v` 图也证了一遍）。

### 57.3 Part B — 计时：本段明显贵于前两刀（**+33–85%**），因收缩维是完整的 `V`/`K`

CUPTI `dev`（kernel-only，µs；`da` = `-DB2M_DA` 非 TMA、`core_tma` = 只有 GEMM 的 TMA 基座）：

| shape | `core` | `core_tma` | `da` | `da_tma` | `da_tma64` | 全 Triton B2 | `da/core` | `da_tma/core_tma` |
|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 517.2 | 492.7 | 952.9 | 654.7 | 891.1 | 903.7 | 1.84× | 1.33× |
| `B2 T1024 H8` | 71.9 | 69.2 | 129.0 | 97.7 | 122.4 | 121.8 | 1.79× | 1.41× |
| `T512 H4` | 9.3 | 8.9 | 13.8 | 13.9 | 12.7 | 15.2 | 1.49× | 1.55× |
| `T250 H3`（非整除） | 8.6 | 8.4 | 12.8 | 13.5 | 11.5 | 14.1 | 1.49× | 1.61× |

- **第三刀贵得多**：非 TMA 下 `da/core` **1.49–1.84×**，TMA 下 `da_tma/core_tma` **1.33–1.61×**——远高于 §55 输出支路
  的 +12–25% 与 §56 `dkgb` 的 +9–13%。**根因是算术量**：`A@dv`/`A@dw` 的收缩维是 **`BT=16`**，而本段是
  **`V=128` 与 `K=128`**（各 8× 的 MAC 数），且用**标量 FMA**（无 tensor core）→ 每条 FMA 要 2 次 smem 读
  （算术强度 ~0.5 op/word）。含本段的 CUDA 核到全 Triton B2 只 **0.72–0.81×**（`B8` 655 vs 904 µs）——**仍不亏**，
  但已不是「白送」。
- `da_tma` 明显快过非 TMA `da`（§52 的 TMA 读侧收益在 `v` 上仍在）；`da_tma64` 在 `BVK=64` 大 shape 回退
  （同 §55.3/§56.3：warp 只在一个 stage 上有活，`[BT,BT]` 归约摊不平）。

### 57.4 `ncu` — 增量＝**标量 smem 流量翻倍 + bank conflict×2.8**，把核从 89% DRAM-bound 拉成 mix

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_da_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | DRAM SOL | L2 bytes | Duration | tensor% | occ | bank conflicts | shared wavefronts | l1tex 全局读 |
|---|---|---|---|---|---|---|---|---|---|---|
| `core_tma`（只有 GEMM） | 1.28 GB | 199.36 MB | 88.87% | 2.34 GB | 494.9 µs | 8.04% | 24.63% | 8.74 M | 38.10 M | **0** |
| `da_tma`（+ `dApre` 段） | **1.48 GB** | **209.79 MB** | 76.17% | 2.71 GB | 660.4 µs | 6.00% | 24.66% | **24.81 M** | **69.57 M** | 4.19 M |

- **时间增量 ≈ 标量 smem 流量**：DRAM 读 `+0.20 GB`（=`v` 一个 `[B,T,H,V]` fp32 = 67 MB × TMA + `k`,`g` 的
  标量全局读 4.19 M sectors ≈134 MB）、DRAM 写 `+10.4 MB`（=`dApre` 一个 `[B,T,H,BT]` fp32 ≈8.4 MB），
  但 **`Duration +165.5 µs`**——`(0.20+0.010) GB / 165.5 µs ≈ 1.27 TB/s`，**远低于**核的 ~2.9 TB/s 上限，
  即**新增段不是带宽、是延迟/发射**。`sm__throughput` 38.05→43.53%、`tensor` 8.04→**6.00%**（被标量段稀释）、
  `DRAM SOL` 反降到 76.17%——**核从 89% DRAM-bound 被拉成 mix**。
- **机制直接读数**：`l1tex` 共享读 wavefronts **38.10M→69.57M（+82%）**、bank conflict **8.74M→24.81M（×2.8，
  占 wavefront 的 ~36%）**、SASS `LDS 88→287`（+199 条标量 shared load）——这正是「标量 FMA 的 2 reads/FMA」
  与 `skg`/`swall` 行优先读的银行冲突。新增 global 读＝`k`/`g` 的 4.19 M sectors（1 sector/元素，标量读）。
- `STG 25→26`（+1 = `dApre` 的 store）、`LDGSTS/UBLKCP 10→12`（+2 = `v` 的 TMA + `k/g`？实为 `v` 的 TMA atom）。

### 57.5 结论 / 对目标卡的设计输入

- **§55.6/§56.6 的「补非 GEMM 段」第三刀成立**：`dA` 的两个 GEMM 贡献 `dv@v^T` 与 `-dw@kg^T`（pre-sandwich）
  进 CUDA 后**独立可证**、**逐位对得上真 fp32（5.3e-7）**、**三种 stage 布局逐位相同**（含非整除 `T=250`）。
- **但不是「便宜」的一刀**：与前两刀最本质的差别是**收缩维不再是 `BT`**——`dApre` 收缩完整的 `V`/`K`
  （各 128），且本段用**标量 FMA**，故 `+33–61%`（TMA）。**设计输入**：目标卡 CUDA B2 里凡「收缩维 = `V` 或 `K`
  全维」的段（`dv@v^T`、`dw@kg^T`，以及后续 `A^T diag A^T` 的 `[BT,BT]@[BT,BT]`），**应像三条主 GEMM 一样用
  `mma.sync`**（base `dv@v^T` 与 `dw@kg^T` 都是 `X @ Y^T` 形式，可直接复用 GEMM 核心的 A/B fragment 读法、
  warp 按 `V`/`K` 分片后 smem 归约），把 `LDS`/bank-conflict 压下去——本轮为**先证代数**取标量实现。
- **精度取舍**：标量收缩给 **fp32 底噪**（§55/§56 的 mma 是 tf32 底噪）；生产若走 mma 会回到 tf32 底噪，但
  如 §6.11/§35 所述目标卡对这类量用 bf16 出口更省钱——须按部署 dtype 重测。
- **设计输入（更新 §56.5）**：「收缩维落在 chunk 行」的段（`A@dv`、`A@dw`）走**原地读 `Akk` + staged 操作数**；
  「收缩维是全维 `V`/`K`」的段（本轮 `dApre`）走**GEMM 核心同款 mma + 部分和归约**。这两类要分开设计。

### 57.6 边界 / 下一步（诚实）

- 本轮只落 **`dApre`（`dw@kg^T` + base `dv@v^T`）**，**未落**最终的 `-A^T diag(beta) A^T` sandwich（§55.6 的另一半
  `dA` 后处理）、`db` 的 `sum_k dkgb·kg`（跨 warp 归约）与**门控链**（`dg`、`dk` 的 `exp2(g)` 标定）——即
  §49.7/§55.6/§56.6 仍未关闭。
- 本段的 base 与 `dw@kg^T` 用**标量 FMA**（非 tensor core），是「先证代数、后优化」的取舍；代价已在 §57.3/§57.4
  量化（+33–61%、LDS×3.3、bank conflict×2.8），下一步应换 mma。
- 只覆盖 `K=V=128`、`BT=16`、fp32/tf32；`dApre` store 用标量 32-bit（`-DB2M_VECST` 未接到本段）；**同 CTA 交接未做**
  （`v`/`k`/`g` 仍从 global 读、`dApre` 落盘再返回）；`k`/`g` 用标量全局读（可改 TMA/向量化）。这些都不影响本节
  结论（增量机制＝收缩维长度与访存机制），但生产形态应接向量化 store 并就地交接。

## 58. 「K2 形状 CUDA B2」补非 GEMM 段第四刀：**最终 `dA` sandwich `-A^T diag(beta) A^T`** 落进 CUDA，并更正 §57 的 `dw` 符号（P7-55）

§55.6/§56.6/§57.6 把 B2 非 GEMM 段的剩余工作列成「`dA` 的 `A^T diag A^T` 后处理、`db` 的 `sum_k dkgb·kg`
（跨 warp 归约）与**门控链**」。§55 落了输出支路 `dvb=A@dv`、§56 落了 `dkgb=A@(-dw)`、§57 落了 `dApre`
（在 sandwich 之前）。本轮落 **`dA` 的最后一步**：把 §57 的 `dApre` **就地**接上最终 sandwich，得到
**完整的 B2 `dA` 输出**：

    m_A[o_t,o_u] = (o_t > o_u) & valid
    dAm          = where(m_A, dApre · beta[o_u], 0)
    dA           = where(o_t > o_u & valid, -(b_A @ (dAm @ b_A)), 0)      # 存到 dA[o_t, o_A]

其中 `b_A[p,j] = A[o_t=j, o_A=p]`（FLA/Triton 的 `b_A` 取向，`check_bwd_dqkg_kernel.py:240`；即 `A^T`），
对应 FLA `chunk_bwd.py:300-304`（`tl.where(m_A, b_dA*beta[None,:],0)` → `tl.dot(b_dA,b_A)` →
`tl.dot(b_A,b_dA)` → `tl.where(m_A,-b_dA,0)`）。sandwich 只吃**已经在 smem 里的** `dApre`、只加两个
`[BT,BT]@[BT,BT]` 的 **fp32 标量** 矩阵乘与一个 `beta` 标量缩放，故**不重读 `dApre`**。

> **本轮的意外收获（更正 §57）**：把 sandwich 接上、对**最终 `dA`** 比 FLA 时，暴露了 §57 的一个**符号错误**——
> FLA 在 `chunk_bwd.py:280` 先 `b_dw = -b_dw` 再 `b_dA += tl.dot(b_dw, kg^T)`，即 `dApre` 的第二项是
> **`- dw@kg^T`**；§57 写成了 `+`。§57 的独立核验没发现，是因为它只跟**自己**的 torch 参考比（两侧同为 `+`）。
> 本轮已把 `.cu`（`dapart = dbpart - dkpart`）与 §57 参考（`base - dwkg`）改正；改后最终 `dA` vs FLA 落 tf32 底噪。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_SAND`，需 `-DB2M_DA`；新符号 `b2sand_launch`，输出 `dApre` 与
`dA`）+ `harness/check_bwd_b2_sand.py`（`harness/check_bwd_b2_sand.out.txt`，空闲卡）+ `harness/ncu_b2_sand_driver.py`
/ `harness/probe_b2_sand_ncu.sh`（`harness/probe_b2_sand_ncu.out.txt`）。

### 58.1 设计：`dApre` 留在 smem，加两个 `[BT,BT]` 的 fp32 标量矩阵乘

- **操作数**：`b_A`（`[BT,BT]`，每 chunk 从 `A[b,t,h,c]` 载入一次，行 `t<T`，pad 行零）与 `beta[o_t]`（每 chunk
  载入一次）——放在 `skg` 同批预取里（`Shared` 加 `sA[BT][BT]`/`sbeta[BT]`/`daB[BT][BT]` ≈ 3 KB）。
- **两步矩阵乘**：`X1 = dAm @ b_A`、`X2 = b_A @ X1`，用**标量 fp32 FMA**（`BT=16` 太小，不入 `mma`）；每个
  256 线程 block 各摊 `[BT,BT]` 的 ~1 条目、内层 `#pragma unroll` 的 `BT` 次点积，`__syncthreads()` 分段。
- **存储映射**：`X2` 的**第一个下标即最终 `dA` 的 token 行**（`dA[o_t,o_A] = X2[i,j]` 同下标，FLA/Triton
  的 store 也是这个映射，`check_bwd_dqkg_kernel.py:203` 的 `permute` 不构成转置）；mask `(i>j)` 作用在 `X2`
  自己的两个轴上。**坑**：`beta` 乘在 **`o_u`（列）** 上而非 `o_t`（`dApre·beta[o_u]`，`tl.where(..., b_dA*beta[None,:], ...)`
  的广播把 `beta` 落到第二个轴），写反则 `max_rel≈1`。
- **无重读**：`da16`（§57 归约后的 `dApre`）直接喂 `dAm`，`dApre` 落盘只是为了独立核对；生产可省。

### 58.2 Part A — 正确性：`dA` 与 torch（同一 `dApre`）**逐位相同**，三种 stage 布局逐位相同，vs FLA 落 tf32 底噪

同操作数四组 shape（`BT=16`，含非整除 `T=250`）。torch 参考吃**CUDA 自己产出的 `dApre`**（关 tf32 走真 fp32），
故比较隔离的正是 sandwich 本身：

| 量 | `sand` vs torch | `sand_tma` vs torch | `sand_tma64` vs torch | vs FLA 最终 `dA` |
|---|---|---|---|---|
| `dA` | **0** | **0** | **0** | 1.56e-3 ~ 2.34e-3 |

（表内为四组 shape 最坏 `max_rel`；`sand` 内部逐位、`sand` vs `sand_tma` vs `sand_tma64` 的 `dA` 与 `dApre`
都 **`max_abs=0`**。）三件事被钉死：(1) **CUDA sandwich 代数无误**——与 torch 在**同一份 `dApre`** 上**逐位相同
（`max_abs=0`）**，且因用 fp32 标量、连 tf32 底噪都没有；(2) **三种 stage 布局（非 TMA / TMA@32 / TMA@64 atom
拆分）对 `dA` 与 `dApre` 都零影响**；(3) **最终 `dA` vs FLA 落 tf32 底噪 1.6e-3~2.3e-3**——这里同时**证伪了 §57
的 `+`**，并**证实更正后的 `-`**（改任一侧符号都会把该项抬到 `max_rel≈0.8`）。

### 58.3 Part B — 计时：sandwich 只值 +2~6%（非 TMA）/ +3~10%（TMA）

CUPTI `dev`（kernel-only，µs；`da` = P7-54 `dApre`、`sand` = +sandwich）：

| shape | `da` | `da_tma` | `sand` | `sand_tma` | `sand_tma64` | 全 Triton B2 | `sand/da` | `sand_tma/da_tma` |
|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 3075.4 | 2076.0 | 3252.2 | 2245.7 | 3223.6 | 2914.6 | 1.06× | 1.08× |
| `B2 T1024 H8` | 363.0 | 216.1 | 371.3 | 222.8 | 369.0 | 358.4 | 1.02× | 1.03× |
| `T512 H4` | 13.9 | 14.0 | 14.5 | 15.5 | 13.8 | 14.6 | 1.04× | 1.10× |
| `T250 H3`（非整除） | 12.9 | 13.7 | 13.8 | 14.9 | 12.9 | 14.2 | 1.06× | 1.09× |

- **这第四刀是「便宜」的**：sandwich 只加 `sand/da` **1.02–1.06×**、`sand_tma/da_tma` **1.03–1.10×**——远低于
  §57 的 +33–61%。原因：`dApre` 已在 smem、sandwich 只做两个 `16×16` 的 fp32 标量乘（全块 ~8K MAC）+ 一次
  `beta` 缩放 + 多一次 `dA` 落盘，**没有任何新的 `[BT,V]`/`[BT,K]` 收缩**。
- **口径提醒**：本轮 CUPTI 绝对值是在**共享卡**上测得（同二进制比 P7-54 的 `check_bwd_b2_da.out.txt` 整体高
  ~3×，如 `da_tma@B8` 654.7→2076），故上表**只用比值**；绝对量以本轮 `ncu` 为准（`da_tma` **655.8 µs** →
  `sand_tma` **703.5 µs**，**+7.3%**，与比值自洽）。
- 含 sandwich 的 CUDA 核到全 Triton B2：非 TMA `sand` **1.12×**、TMA `sand_tma` **0.77×**（`B8`）——即
  **TMA 版 CUDA 核（含完整 `dA`）已比全 Triton B2 快 ~23%**，且仍比 §57 的 `da_tma`（0.72×）只贵一点。

### 58.4 `ncu` — 增量＝`dA` 落盘（+8.4 MB）+ `sA/beta` 标量读 + barrier，冲突/占用/带宽都不动

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_sand_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | Duration | L2 bytes | tensor% | occ | bank conflict | shared wavefronts | l1tex 全局读 sectors | SASS LDS | STG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `da_tma`（P7-54） | 1.48 GB | 210.19 MB | 655.8 µs | 2.71 GB | 5.99% | 24.66% | 24.82 M | 69.58 M | 4.194 M | 287 | 26 |
| `sand_tma`（本轮） | 1.49 GB | **218.22 MB** | **703.5 µs** | 2.75 GB | 5.58% | 24.67% | 24.87 M | 72.98 M | 4.588 M | 334 | **31** |

- **增量就是把 `dA` 写下去 + 把 `b_A/beta` 读上来**：DRAM 写 `+8.0 MB`（=`dA` 一个 `[B,T,H,BT]` fp32 = 8.4 MB，
  与 `dApre` 同尺寸）、DRAM 读 `+0.01 GB`、l1tex 全局读 sector `+0.39 M`（`b_A` 的 `[BT,BT]` 标量读 + `beta`）、
  SASS `LDS 287→334`（+47 条标量 shared load = 两个 `[BT,BT]` 矩阵乘）、`STG 26→31`（+5 = `dA` 的标量 store）。
- **不引入新瓶颈**：bank conflict（24.82→24.87 M）、占用率（24.66→24.67%）、tensor%（5.99→5.58，被标量段稀释）
  都**纹丝不动**；`Duration +47.7 µs`、`L2 2.71→2.75 GB`、`DRAM SOL 76.73→72.29%`（核仍 DRAM-bound）。即这部分
  是**加的一小块确定性工作**，不是新 bound。

### 58.5 结论 / 对目标卡的设计输入

- **B2 的 `dA` 支路在 CUDA 里闭环**：`dApre`（`dv@v^T - dw@kg^T`）→ sandwich（`-A^T diag(beta) A^T`）→ 完整
  `dA`，**独立可证**（vs torch 逐位、vs FLA 落 tf32 底噪）、**四种 stage 布局逐位一致**（含非整除 `T=250`）。
  sandwich 便宜（+2–10%），因为它纯做在 smem 的 `dApre` 上、无新收缩维。
- **更正 §57 的符号**：`dApre` 第二项是 **`- dw@kg^T`**（FLA `chunk_bwd.py:280` 的 `b_dw=-b_dw`）。
  **教训**：非 GEMM 段的独立核验若只跟**自己的**参考比，**符号/常数整体错**是查不出来的——必须让**下游
  （这里是 sandwich）接上、对最终对外量比 FLA**，中间量的错误才会被放大现形（`+` 时 `dA` vs FLA 的
  `max_rel≈0.83`，`-` 时 2e-3）。
- **设计输入（更新 §57.5）**：目标卡 CUDA B2 里「收缩维是全维 `V`/`K`」的段（`dv@v^T`、`dw@kg^T`）应换
  `mma.sync`（§57.5）；而**sandwich 这类 `[BT,BT]@[BT,BT]` 的小段**（本轮的 `-A^T diag A^T`）**留在标量 fp32
  即可**——它只值 +2–10%，换 mma 的收益微乎其微，反而不如把工量投到仍缺的 `db` 与门控链。
- **同 CTA 交接的收益**：`dApre` 不落盘、`dA` 就地交下游（B3/intra 或 dAv）可再省一次 `[B,T,H,BT]` 读写
  （这里约 16–17 MB/核），是生产形态应做的（本轮为独立核对仍落盘）。

### 58.6 边界 / 下一步（诚实）

- 本轮只落 **sandwich**；`db` 的 `sum_k dkgb·kg`（跨 warp 归约）与**门控链**（`dg`、`dk` 的 `exp2(g)` 标定）
  **仍未落 CUDA**——即 §58 的 4 刀把 `dA` 支路补全，但 B2 全核（`dq/dk/dv2/db/dg/dA` 六项）尚缺 `db`/门控链，
  下一步＝在同一核里补 `db`（需 `dvb`(§55) 与 `dkgb`(§56) 的跨 warp 归约）与门控链（`dg` 的 `q·dq`/`kdk`/
  `m_last·dgk`/`kg·dkgb·beta` 与 `dk += dkgb·gb`）。
- 只覆盖 `K=V=128`、`BT=16`、fp32；sandwich 用标量 fp32、`dA` store 用标量 32-bit（`-DB2M_VECST` 未接）；
  本轮 `-DB2M_SAND` 要求 `-DB2M_DA`（sandwich 吃 `dApre`），未与 §55/§56 的 `-DB2M_OUT`/`-DB2M_DKGB` 组合。
  这些都不影响本节结论（增量机制＝一小块确定性 smem 工作 + 一次落盘）。
- **本轮同时更正了 §57 的 `dw` 符号**并在 `check_bwd_b2_da.py`/`b2_mma_probe.cu` 落地；§57 的 `out.txt`
  数值（CUDA vs 自身参考 `5.3e-7`、逐位一致、计时比值）在符号改正后**不变**（两侧同步翻转），故未重跑覆盖；
  P7-54 的 ncu（字节/冲突/时长）与符号无关。

## 59. 「K2 形状 CUDA B2」补非 GEMM 段第五刀：**写入门梯度 `db = Σ_v dvb·v + Σ_k dkgb·kg`** 落进 CUDA（P7-56）

§55–§58 把 `dA` 支路（`dvb`→`dApre`→sandwich）与 `dkgb` 补进 CUDA 后，B2 六项里只剩 **`db`** 与**门控链**。
本轮落 `db`——它是唯一一个**纯归约、无新收缩**的段：

    db[o] = Σ_v dvb[o,v] · v[o,v]   +   Σ_k dkgb[o,k] · kg[o,k]

对应 FLA `chunk_bwd.py` / `check_bwd_dqkg_kernel.py`：先 `b_db = (b_dvb * b_v).sum(dim=3)`
（`check_bwd_dqkg_kernel.py:169-171`），后 `b_db = b_db + (b_dkgb * b_kg).sum(dim=3)`（`:179-183`），
Triton 里就是 `b_db += tl.sum(b_dvb * b_v, axis=1)`（`:295`）。其中 `dvb = A@dv`（P7-52 §55）与
`dkgb = A@(-dw)`（P7-53 §56）**都已在同一核里算出**，各自被一个 warp 的列切片私有，所以 `db` 就是
**对两个 warp 私有列和做一次跨 warp 归约**。`db` 是 B2 的**完整输出**（门控链不再改它），可独立核验。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_DB`，需 `-DB2M_OUT -DB2M_DKGB`；诊断开关 `-DB2M_NODB`；
新符号 `b2db_launch`，输出 `dvb`/`dv2`/`dkgb`/`db`）+ `harness/check_bwd_b2_db.py`
（`harness/check_bwd_b2_db.out.txt`，空闲卡）+ `harness/ncu_b2_db_driver.py` / `harness/probe_b2_db_ncu.sh`
（`harness/probe_b2_db_ncu.out.txt`）。

### 59.1 设计：两个 warp 私有列和 → smem 跨 warp 归约（`db` 无新操作数）

- **操作数已在手**：`dvb` 是输出支路（P7-52）的 `acc2` 累加器，`dkgb` 是 `dkgb` 段（P7-53）的 `acc3`；
  两者在 **相同 warp 列划分**下（`acc2` 按 V 列、`acc3` 按 k 列，都取 `[w*BW, w*BW+BW)`），`kg`（`skg`）与
  原始 `v` 都是逐 token 的 `[BT,·]`——所以 `db[o]` 只是把每个 warp 自己那 16 列的和加起来。
- **`dvb·v` 项需要一次 `v` 的持久镜像**（`Stage::vv` 是逐 V-tile 暂存、会被后续 stage 覆写，而 `acc2` 要等
  整个 V 循环跑完才完整）：在 warp 拥有该 V 列的 stage 里把 `vv` 的这 16 列抄进 `sh->dbv[warp][BT][BW]`
  （`b2_mma_probe.cu:750-762`）；`kg·dkgb` 项的 `skg` 本就是持久 buffer，直接读。
- **归约**（`b2_mma_probe.cu:848-878`）：`acc2/acc3` 的 mma 片段行是 `g/g+8`、列是 `nt*8+2t,nt*8+2t+1`，
  故每 lane 先把这 8 个片元乘对应 `dbv`/`skg` 求和，再在**行组内 4 个 lane** 上 `shfl_xor`（1、2）归约，
  由 `t==0` 写 `sh->dbrow[warp][g/g+8]`；最后每行读 8 个 warp 的部分和落盘。**全程无新 global 读**
  （`skg`/`dbv` 都在 smem）；`db` 落盘是 `[B,T,H]` 一个 fp32。
- `-DB2M_VV_KG_ON = (B2M_DA || B2M_DB)`（`:192`）复用 `vv` 暂存与 `skg`；`B2M_DB` 要求 OUT+DKGB
  （`:180-184`）。`-DB2M_NODB` 保留 build/签名/smem 但跳过归约+落盘，作计时基线（`:186`）。

### 59.2 Part A — 正确性：归约对 torch 落 fp32 序 ~1.8e-7，整链 vs FLA 落 tf32 底噪，四布局逐位相同

四组 shape（`BT=16`，含非整除 `T=250`）。`db vs torch(same dvb/dkgb)` 吃 **CUDA 自己产出的 `dvb/dkgb`**
（关 tf32 走真 fp32），隔离的正是归约本身：

| 量（四组最坏 `max_rel`） | 值 |
|---|---|
| `db` vs torch（同一 `dvb/dkgb`） | **1.3e-7 ~ 1.8e-7**（纯 fp32 归约序） |
| `db` vs torch（从操作数重算） | 2.0e-4 ~ 2.3e-4（=tf32 的 `dvb/dkgb`） |
| `dvb` / `dkgb` vs torch（从操作数） | 2.6e-4~4.4e-4 / 3.4e-4~3.6e-4 |
| `db` vs FLA `db` | **5.5e-4 ~ 6.3e-4**（=tf32 底噪，与 `dv2` vs FLA 的 6.2e-4~8.8e-4 同阶） |
| `db` vs `db_tma` / `db_tma64` | **0（逐位相同）** |

三件事被钉死：(1) **CUDA `db` 代数无误、且归约就是 fp32 序**——与 torch 在同一份 `dvb/dkgb` 上 ~1.8e-7
（非逐位：CUDA 按「每 warp 16 列、组内 4 lane、8 warp」的固定顺序累加，torch 按整条 V/k 轴，仅加法序不同）；
(2) **三种 stage 布局（非 TMA / TMA@32 / TMA@64 atom 拆分）对 `db` 零影响（`max_abs=0`）**——因为 `dvb/dkgb`
已逐位、归约序与 staging 无关；(3) **最终 `db` vs FLA 落 tf32 底噪 6.3e-4**——与 `dkgb`/`dv2` 的 tf32 误差同阶，
无算法级偏差。

### 59.3 Part B — 计时：`db` 段只值 +2~8%（非 TMA）/ +2~13%（TMA）

CUPTI `dev`（kernel-only，µs；`*_nodb` = 同 build 跳过 `db` 归约）：

| shape | `db_nodb` | `db` | `tma_nodb` | `db_tma` | `db_tma64` | 全 Triton B2 | `db_nodb→db` | `tma_nodb→tma` |
|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 755.5 | 774.6 | 742.5 | 763.1 | 1024.4 | 896.5 | 1.03× | 1.03× |
| `B2 T1024 H8` | 109.5 | 111.6 | 107.7 | 110.6 | 138.6 | 121.3 | 1.02× | 1.03× |
| `T512 H4` | 14.7 | 15.3 | 13.9 | 15.7 | 15.4 | 15.4 | 1.05× | 1.13× |
| `T250 H3`（非整除） | 13.6 | 14.5 | 13.0 | 14.6 | 13.7 | 14.1 | 1.07× | 1.13× |

- **第五刀最便宜的一类**：`db` 只加 **1.02–1.07×（非 TMA）/ 1.03–1.13×（TMA）**——因为它是纯 smem 归约
  （`dbv` 镜像 + 8 个 warp 的部分和），**没有任何新 global 读、没有新收缩维**。
- 含 OUT+DKGB+DB 的 CUDA 核（`db_tma`）到全 Triton B2：`B8` **0.85×**、`B2` **0.91×**、`T512` ~1.0×、
  `T250` 1.04×——即 CUDA 核已补进 `dv2`/`dkgb`/`db` 三条支路后仍**不慢于/略快于**全 Triton B2（差门控链）。
- `db_tma64`（BVK=64）：`B8` 1024µs——与本专题多次结论一致，`BVK=64` 在本核上因 smem/占用无正收益，取 `BVK=32`。

### 59.4 `ncu` — 增量＝`db` 落盘（+0.5 MB）+ `dbv/skg` 标量 smem 读，冲突/占用/带宽都不动

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_db_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | Duration | L2 bytes | tensor% | occ | bank conflict | shared wavefronts | l1tex 全局读 sectors | SASS LDS | STG |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `db_tma_nodb` | 1.55 GB | 398.1 MB | 742.2 µs | 3.28 GB | 5.77% | 24.55% | 9.89 M | 40.30 M | 10.49 M | 104 | 49 |
| `db_tma`（本轮） | 1.56 GB | **399.5 MB** | **762.3 µs** | 3.26 GB | 5.61% | 24.55% | **12.87 M** | 44.92 M | 10.49 M | **124** | **50** |

- **增量就是把 `db` 写下去 + 读 smem 里的 `dbv/skg`**：DRAM 写 `+1.4 MB`（`db` 一个 `[B,T,H]` fp32=0.5 MB，
  余为 L2 记账噪声）、DRAM 读 `+0.01 GB`、l1tex 全局读 sectors **不变**（`db` 不读 global）、SASS
  `LDS 104→124`（+20 = `dbv` 镜像 + 归约的标量 shared load）、`STG 49→50`（+1 = `db` 落盘）。
- **不引入新瓶颈**：占用率（24.55%，纹丝不动）、DRAM/L2 SOL（78.5→76.5% / 86.7→84.6%，核仍 **DRAM/L2 双
  bound**）、tensor%（5.77→5.61，被标量段稀释）都基本不动；`Duration +20 µs（+2.7%）`。bank conflict
  9.89→12.87 M（+30%，全部来自 `dbv/skg` 的标量 shared load），但 shared-load wavefront 只 +11%，**不改变
  核的 bound**（同 §57/§58 的「加一小块确定性工作」）。

### 59.5 结论 / 对目标卡的设计输入

- **B2 的 `db` 支路在 CUDA 里闭环**：`db = Σ_v dvb·v + Σ_k dkgb·kg`，**独立可证**（归约对 torch ~1.8e-7、
  整链 vs FLA 落 tf32 底噪 6.3e-4）、**四种 stage 布局逐位一致**（含非整除 `T=250`）。`db` 是**最便宜的一刀**
  （+2~13%），因为它纯做 smem 归约、无新收缩维、无新 global 读。
- **设计输入**：目标卡 CUDA B2 里 `db` 这类「**warp 私有列和 → 跨 warp 归约**」的段，落点就是
  **`shfl_xor` 行组归约 + 一次 smem 跨 warp 归约 + 一个 `[BT]` 落盘**，不要为它另起收缩；
  `dvb·v` 项的 `v` 需在拥有该列的 stage 里**存一份持久镜像**（`dbv`），否则 stage 缓冲早被覆写。
- **同 CTA 交接的收益**：本轮 `dkgb` 仍吃**从 global 重读的 `dw`**（见 §59.6 的坑）；生产形态应把 `dw`
  留在寄存器/同 CTA smem，`dkgb`/`db`/`dA` 都就地取——可再省 `dw` 的写+读（此处约 0.5 GB@`B8`）。

### 59.6 坑与边界（诚实）

- **坑（本轮真实踩到，值得记）**：harness 一开始把 `dw` 的**输入指针与 `dw_out` 输出指针设成同一个张量**
  （`dw = dv@h` 的 round-trip），结果 `dkgb` 大面积错（`max_rel≈2`）、`db` 随之错——**逐位对照 dkgb 专用 build
  才定位**。即：本原型里 `dw` 是**先算后写、再从 global 读回**的两趟契约，输入与输出**共用缓冲会被流水/落盘
  覆写**；harness 改为独立 `dw_out` 缓冲后，DB build 的 `dkgb` 与 §56 的 dkgb 专用 build **逐位相同
  （`max_abs=0`）**。这也正是「`dw` 应在同 CTA 交接、不走 global」的实证动机（§59.5 末条）。
- 本轮只落 `db`；**门控链**（`dg` 的 `q·dq`/`kdk`/`m_last·dgk`/`kg·dkgb·beta` 与 `dk += dkgb·gb`）**仍未落
  CUDA**——落完后 B2 的六项 `dq/dk/dv2/db/dg/dA` 才算全搬进 CUDA。下一步＝在 `db` 之后补门控链。
- 只覆盖 `K=V=128`、`BT=16`、fp32；`B2M_DB` 要求 `OUT+DKGB`，未与 `B2M_DA`/`B2M_SAND` 组合；`db` 落盘用
  标量 32-bit（`-DB2M_VECST` 未接）。这些都不影响本节结论（增量机制＝一小块 smem 归约 + 一次落盘）。

## 60. 「K2 形状 CUDA B2」补非 GEMM 段第六刀：**门控链**（最终 `dq/dk/dg`）落进 CUDA——`dq` 逐位、`dg` fp32 序 ~2e-7，整链落 tf32 底噪（P7-57）

§55–§59 把 B2 的 `dv2`/`dkgb`/`dA`/`db` 补进 CUDA 后，六项里只剩**门控链**——即把三条 raw GEMM 输出
（`dq=[do]@h^T`、`dk=[v_new]@dh^T`，及 §56 已在此核的 `dkgb=A@(-dw)`）与逐 chunk 的 `dgk=[Σ_V h⊙dh]`
（§49/§50 已在此核）**按门控标定成最终输出**。对应 FLA `chunk_bwd.py` / `check_bwd_dqkg_kernel.py:173-189`：

    gk_exp = exp2(g)                 # g = chunk 内 log2 前缀和（log2 域，≤0）
    gn     = g[c 的最后一个有效行]      # 逐 chunk 标量行
    dq     = dq_raw * gk_exp * scale
    dk_mid = dk_raw * exp2(gn - g)
    kdk    = k * dk_mid              # 注意：用**加 dkgb·gb 之前**的 dk
    dgk    = dgk_raw * exp2(gn) + Σ_t kdk          # 逐 (chunk,k) 的标量
    dg     = q*dq - kdk + (t==last)*dgk + k*gk*dkgb*beta
    dk     = dk_mid + dkgb*(gk*beta)

这是 B2 的**最后一段**：落完后六项 `dq/dk/dv2/db/dg/dA` 全部第一次同时存在于 K2 形状 CUDA 核里。它与前五刀的
结构区别是：**没有新的收缩维、没有新的 GEMM**——每个 warp 拥有的 k-列切片上，各 mma 片元位置的门控标定互相
独立；唯一的归约是 `Σ_t kdk`（chunk 内逐列和，一次 **warp 内 `shfl_xor`**，因为 mma 的 16 行恰好铺在 groupID
的 8 个 lane × 2 行上）。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_GATE`，需 `-DB2M_DKGB`；诊断开关 `-DB2M_NODG`；新符号
`b2gate_launch`，输出 raw `dw`/`dgk`/`dkgb` + 最终 `dqg`/`dkg`/`dg`）+ `harness/check_bwd_b2_gate.py`
（`harness/check_bwd_b2_gate.out.txt`，空闲卡）+ `harness/ncu_b2_gate_driver.py` / `harness/probe_b2_gate_ncu.sh`
（`harness/probe_b2_gate_ncu.out.txt`）。

### 60.1 设计：门控标定全在寄存器，唯一归约是 warp 内列和

- **操作数**：`accq`/`acck`（三条 V-归约 GEMM 的 raw 输出，warp 拥有 k 列 `[w·16, w·16+16)`）、`acc3`（§56 的
  `dkgb`，同样的 k-列划分）、raw `dgk`（§49 的逐 chunk `Σ_V h⊙dh`，已落 `dgk[B,NT,H,K]`）；输入 `q/k/g/beta`
  与标量 `scale`。
- **门控标定**：对每个 mma 片元位置 `(tr0/tr1, c0/c1)` 读 raw `q/k/g`，算 `exp2`、`dq`/`dk_mid`/`kdk`、以及
  `dg` 的三个非归约项，写 `dqg`/`dkg`。**不新增任何 `[BT,K]` 缓冲**——`q/k/g` 在 post-loop 里按片元地址从
  global 读（原型形态）。
- **`Σ_t kdk` 归约**（`b2_mma_probe.cu` 门控段的 `pk0/pk1`）：每 lane 把自己 2 行 × 2 列的 `kdk` 求和，再在
  groupID 上 `shfl_xor(4/8/16)`（3 步）得到整列和；`g==0` 的 lane 把 `dgk_raw*exp2(gn) + Σ_t kdk` 写进
  `sh->sdgkt[KD]`，`__syncwarp()` 后 `t==last` 的行读回。全程无新 global 收缩（`dgk` 的读回是**本 CTA 刚写**的
  地址）。
- **`m_last` 项**：`dg` 只在 chunk 最后一个有效行加 `dgk`（`lastrow = last - i_t*BT`，行判定用
  `(i<2? g : g+8) == lastrow`）。**注意这是逐行条件、不是逐 lane**（见 §60.6 的坑）。
- `-DB2M_GATE` 下**不再落盘 raw `dq`/`dk`**（生产形态：它们只活在寄存器里；harness 的对照从裸核 `core` build
  取 raw），故门控 build 的写出＝`dw`+`dkgb`+`dqg`+`dkg`+`dg`。

### 60.2 Part A — 正确性：`dq` 逐位，`dk`/`dg` 落 fp32 序，三布局逐位/近逐位

四组 shape（`BT=16`，含非整除 `T=250`）。isolate 门控算术：把 torch 门控作用在**同一份 CUDA raw
`dq/dk/dgk` + `dkgb`** 上：

| 量（四组最坏 `max_rel`） | 值 |
|---|---|
| `dq` vs torch（同一 raw） | **0（逐位相同）** |
| `dk` vs torch（同一 raw） | **4.3e-8**（fp32 归约序） |
| `dg` vs torch（同一 raw） | **2.2e-7**（fp32 序，`Σ_t kdk` 加法次序） |
| `dq` / `dk` / `dg` vs FLA | **1.05e-3 / 9.47e-4 / 1.01e-3**（tf32 底噪） |
| `dq`/`dk` vs `gate_tma` | **0（逐位相同）** |
| `dg` vs `gate_tma` / `gate_tma64` | **0 / 1.08e-7**（后者是 `BVK=64` 的 V-累加序） |

三件事被钉死：(1) **`dq` 的门控标定逐位无损**（`dq_raw*gk*scale`，与 torch 同一顺序）；(2) **`dk`/`dg` 只是
fp32 加法序差（~1e-7）**，无算法偏差——`Σ_t kdk` 的 warp 内归约序与 torch 的整行求和不同；(3) **最终三项 vs FLA
全落 tf32 底噪 ~1e-3**，与 `dv2`/`dkgb`/`db` 同阶。三种 stage 布局（非 TMA / TMA@32 / TMA@64 atom）对 `dq`/`dk`
**逐位零影响**，`dg` 仅 `TMA@64` 因 `BVK=64` 的 V 方向累加分组差 1 ulp。

### 60.3 Part B — 计时：门控链是**前五刀里最贵的一刀**（+30~38%），但仍不慢于全 Triton B2

CUPTI `dev`（kernel-only，µs；`gate_nodg` = 同 build 跳过门控归约+落盘）：

| shape | `gate_nodg` | `gate` | `gate_tma` | `gate_tma64` | 全 Triton B2 | `gate/nodg` | `gate_tma/B2` |
|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 517.9 | 673.3 | **655.1** | 852.3 | 897.7 | 1.30× | **0.73×** |
| `B2 T1024 H8` | 70.5 | 95.3 | 93.8 | 110.9 | 121.5 | 1.35× | 0.77× |
| `T512 H4` | 8.7 | 12.0 | 11.5 | 11.2 | 14.7 | 1.38× | 0.78× |
| `T250 H3`（非整除） | 8.0 | 10.9 | 10.7 | 10.2 | 14.0 | 1.37× | 0.76× |

- **门控链比前五刀都贵**：`db` 只 +2~13%（§59）、输出支路 +12~25%（§55）、`dkgb` +9~13%（§56），而门控
  **+30~38%**。原因见 §60.4（不是收缩，而是 **raw `q/k/g` 的片元式 global 读**——每个 lane 按 `(row,col)`
  散读 8 个位置 × 3 张量，未合流）。
- **K2 形状 CUDA B2（GEMM 核 + `dkgb` + 门控）到全 Triton B2 `gate_tma/B2` = 0.73–0.78×**——即补进门控后仍
  不慢于全 Triton B2（差 `dv2`/`db`/`dA` 三条已分别落地的支路）。
- `gate_tma64`（`BVK=64`）如前几刀无正收益（本核 smem/占用约束），取 `BVK=32`。

### 60.4 `ncu` — 增量 = **+199 MB 输出写 + 9× 的 global load sector（非合流 `q/k/g` 读）+ `exp2f`**

`B8 T2048 H8`，同一批 fp32 操作数（`probe_b2_gate_ncu.out.txt`）：

| 变体 | DRAM 读 | DRAM 写 | Duration | L2 bytes | tensor% | occ | bank conflict | shared wavefronts | l1tex 全局读 sectors | srcunit 读 | XU inst |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `gate_tma_nodg` | 1.35 GB | 135.7 MB | 495.9 µs | 2.35 GB | 2.99% | 24.54% | 9.01 M | 26.31 M | **2.10 M** | 42.22 M | 0.197 M |
| `gate_tma`（本轮） | 1.56 GB | **334.3 MB** | **655.3 µs** | 3.07 GB | 6.30% | 24.23% | 9.68 M | 39.83 M | **19.40 M** | 48.74 M | **1.507 M** |

- **写出**：`gate` 比 `nodg` 多写 **199 MB = `dqg`+`dkg`+`dg` 三张 `[B,T,H,K]` fp32**（各 67 MB）——门控的
  三项输出本身就是 B2 的交付物，非额外开销（生产的 B2 只写 `dq/dk/dg` 三张，与裸核写 `dq/dk/dw` 同量）。
- **读入**：l1tex 全局读 sector `2.10 M→19.40 M`（**9.2×**），srcunit 读 `42.2 M→48.7 M`（+6.5 M）。这
  17.3 M 新 sector（≈554 MB L1 侧字节）就是 **`q/k/g` 按 mma 片元地址的散读**——每 lane 8 位置 × 3 张量，
  地址在 warp 内不连续（`row=g/g+8`、`col=2t/2t+1`），合流度差。DRAM 读只 +0.21 GB，说明大头被 L2 吸收，但
  **L2 仍被推高 +0.72 GB**、L2 SOL 87–88%（核仍 DRAM/L2 双 bound）。
- **`exp2f`**：`sm__inst_executed_pipe_xu` `0.197 M→1.507 M`（+1.31 M = 门控的 `exp2(g)`/`exp2(gn-g)`/`exp2(gn)`
  的 MUFU 发射，SASS `MUFU 1→21`）。占用率（24.5%）、bank conflict（9.0→9.7 M，仅 +7%）、tensor%（被稀释）
  都不构成新瓶颈；`Duration +159 µs（+32%）`与「+199 MB 写（≈80 µs@2.5 TB/s）+ 更差的读合流 + `exp2` 发射」一致。

### 60.5 结论 / 对目标卡的设计输入

- **B2 的门控链在 CUDA 里闭环**：最终 `dq/dk/dg` 从 raw GEMM 输出 + `dgk` + `dkgb` 就地算出，**`dq` 逐位、
  `dk`/`dg` fp32 序 ~1e-7、整链 vs FLA 落 tf32 底噪**、三布局逐位/近逐位（含非整除 `T=250`）。至此 §5.2 的
  **B2 六项 `dq/dk/dv2/db/dg/dA` 第一次全部存在于 K2 形状 CUDA 核**。
- **设计输入（关键）**：门控的 `Σ_t kdk` 这类「chunk 内逐 k-列和」的落点是 **warp 内 `shfl_xor`（groupID
  3 步）+ 一个 `[KD]` 的 smem tile**，与 `dgk` 的逐 chunk 标量复用——**不要为它另起收缩**。`m_last` 项要按
  **行坐标**判定（`(i<2?g:g+8)==lastrow`），不能只看 lane。
- **设计输入（成本）**：门控是本核最贵的非 GEMM 段（+30~38%），成本不在算术而在**raw `q/k/g` 的片元式 global
  读**（9× 全局 sector）。目标卡应把 `q/k/g` **像 `kg` 一样在 chunk 开头合流载入 smem**（生产 B2 里 `k`/`g`
  本就为 `kg`/`db`/`dA` 载入，只需多一张 `q`），把散读换成一次合流读——这是把 +30% 收回去的下一个杠杆（待验证）。
- **整核定位**：K2 形状 CUDA（GEMM 核 + `dkgb` + 门控）已到全 Triton B2 的 **0.73–0.78×**；余下 `dv2`/`db`/`dA`
  三条支路已分别落地（§55/§56/§59 + §57/§58），把它们叠满即完整 K2 形状 CUDA B2。

### 60.6 坑与边界（诚实）

- **坑（本轮真实踩到，指纹很典型）**：`m_last` 项第一版写成
  `add = (g == lastrow || g + 8 == lastrow) ? … : 0`，**条件只依赖 lane 的 `g`、不依赖片元行 `i`**。结果
  lane `g=7`（其 `g+8=15=lastrow`）在**行 `g`（=7）与行 `g+8`（=15）两处**都加了 `dgk`。指纹：`dq`/`dk`
  逐位对、`dg` 只在 `行%16 ∈ {7,15}` 上错（`max_rel≈1`，`max_abs≈|dgk|`）。改成逐行 `rowrel=(i<2?g:g+8)`
  即归零。**教训**：mma 片元的「lane 条件」与「行条件」不是一回事，凡按行 mask 必须显式换算行坐标。
- 本轮只覆盖 `K=V=128`、`BT=16`、fp32；`B2M_GATE` **要求 `B2M_DKGB` 且不与 `B2M_OUT`/`B2M_SAND` 组合**
  （standalone，编译期 `#error` 拦截）。`q/k/g` 走**片元式 global 读**（未合流，见 §60.4）；`dqg`/`dkg`/`dg`
  落盘用标量 32-bit（`-DB2M_VECST` 未接）。这些都不影响本节结论（增量机制 = 门控标定 + warp 内列和 + 三次落盘）。

## 61. 「K2 形状 CUDA B2」门控链的下一杠杆——把 `q/k/g` 合流载入 smem：**证伪**（P7-58）

§60.5 把门控链这最贵的一刀（+30~38%）的成因归到 **raw `q/k/g` 的片元式 global 读**（§60.4：l1tex 全局读
sector `2.10 M→19.40 M`，9.2×），并给出下一个杠杆：**把 `q/k/g` 像 `kg` 一样在 chunk 开头合流载入 smem**，
把散读换成一次合流读（当时标「待验证」）。本轮把它直接测掉，答案是**负的**——合流载入 smem 不能收回那 +30%，
反而**净亏**。

证据：`harness/b2_mma_probe.cu`（新增 `-DB2M_QKG`，需 `-DB2M_GATE`；+ 诊断 `-DB2M_QKGP` 只把 staging 的行距
从 `KD`(128 字) 加到 `KD+4`）+ `harness/check_bwd_b2_gate_qkg.py`（`harness/check_bwd_b2_gate_qkg.out.txt`，
空闲卡）+ `harness/probe_b2_gate_qkg_ncu.sh`（`harness/probe_b2_gate_qkg_ncu.out.txt`）。

### 61.1 设计：一次合流的 `[BT,KD]` staging，门控改读 smem

- **staging**（`b2_mma_probe.cu:643-654`）：chunk 开头一次 `for idx in [0, BT*KD)` 循环，按 `row=idx/KD,
  col=idx%KD` **合流**读 `q/k/g`（各一次）写进 `sh->sq/skb/sgb[BT][QKGP]`（pad rows zeroed，`tt<T` 守卫），
  与 `kg`（`:604-618`）同构；随后 `__syncthreads()`（`:656-658`，本条把 `B2M_QKG` 加进该守卫）。
- **门控读**（`:960-976`）：P7-57 的 `qv=qptr[off]; kv=kraw[off]; gv=graw[off]` 与 `gl=graw[gl_base+…]`
  在 `B2M_QKG` 下换成 `sh->sq[rowrel][col]` / `sh->skb[…]` / `sh->sgb[…]`（`rowrel = i<2?g:g+8`）。
  **算术一字未动**——只把操作数来源从「按 mma 片元地址散读 global」换成「先合流落 smem、再按片元读 smem」。
- `-DB2M_QKGP`：staging 行距 `KD=128` 字（`128 % 32 == 0`）时，门控按固定 `col` 扫 `row=g/g+8`，8 个 row
  全落**同一个 bank**；`QKGP=KD+4`（`(4*row+col) % 32` 在 `row=0..7` 上取遍 `{0,4,…,28}`）即消掉该冲突。

### 61.2 Part A — 正确性：三种 stage 布局**逐位相同**（合流只是换取数）

| 量（四组最坏 `max_rel`） | 值 |
|---|---|
| `dqg`/`dkg`/`dg` vs `gate_qkg`（P7-57 门控） | **0 / 0 / 0（逐位相同）** |
| … vs `gate_qkgp`（pad 行距） | **0 / 0 / 0** |
| … vs `gate_tma_qkg` / `gate_tma_qkgp`（TMA 原子布局） | **0 / 0 / 0** |
| … vs `gate_tma64_qkg`（`BVK=64`） | **0 / 0 / 1.08e-7**（仅 `dg` 1 ulp，`BVK=64` 的 V-累加序） |
| `dq`/`dk`/`dg` vs FLA | 1.05e-3 / 9.47e-4 / 1.01e-3（tf32 底噪） |

即合流载入**逐位无损**：换源不改值，三布局（非 TMA / TMA@32 / TMA@64）一致（含非整除 `T=250`）。

### 61.3 Part B — 计时：**全为净亏**（非 TMA 1.11–1.51×，TMA 1.08–1.19×），padding 改不了结论

CUPTI `dev`（kernel-only，µs；`qkg`=合流，`qkgp`=合流+pad 行距）：

| shape | `gate` | `gate_qkg` | `gate_qkgp` | `gate_tma` | `tma_qkg` | `tma_qkgp` | 全 Triton B2 | `qkg/gate` | `qkgp/gate` | `tqkg/tma` | `tqkgp/tma` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 676.0 | 1021.6 | 1004.0 | 655.1 | 715.2 | 706.3 | 896.8 | **1.51×** | 1.49× | **1.09×** | 1.08× |
| `B2 T1024 H8` | 95.0 | 135.2 | 132.7 | 94.0 | 106.2 | 104.2 | 121.9 | 1.42× | 1.40× | 1.13× | 1.11× |
| `T512 H4` | 12.0 | 13.8 | 13.8 | 11.7 | 13.8 | 13.8 | 15.1 | 1.15× | 1.15× | 1.19× | 1.19× |
| `T250 H3`（非整除） | 11.1 | 12.4 | 12.1 | 10.9 | 12.8 | 12.4 | 14.1 | 1.12× | 1.09× | 1.18× | 1.14× |

- **合流 `q/k/g` 一律更慢**，大 shape 最明显（非 TMA `B8` 1.51×）；pad 行距只把 1.51→1.49、1.09→1.08，
  **不改变符号**。即「散读 global 换合流 smem」这条路在任何布局下都不回本。

### 61.4 `ncu` — 两个独立机制，但**都不指向 global 读合流**是那 +30% 的成因

`B8 T2048 H8`（`probe_b2_gate_qkg_ncu.out.txt`）：

| 变体 | dyn smem | occ 限(block) | warps_active | global ld sectors | bank conflict | shared wavefronts | DRAM 读 | L2 bytes | Duration |
|---|---|---|---|---|---|---|---|---|---|
| `gate` | 92.67 KB | 2 | 24.16% | 66.58 M | 9.87 M | 40.0 M | 1.56 GB | 3.79 GB | 672 µs |
| `gate_qkg` | 117.25 KB | **1** | **12.44%** | 60.03 M | 19.94 M | 51.9 M | 1.55 GB | 3.77 GB | 1020 µs |
| `gate_qkgp` | 118.02 KB | **1** | 12.45% | 60.03 M | **10.51 M** | — | 1.55 GB | 3.78 GB | 1000 µs |
| `gate_tma` | 82.56 KB | 2 | 24.21% | 19.40 M | 9.67 M | 39.8 M | 1.56 GB | 3.07 GB | 652 µs |
| `gate_tma_qkg` | 107.14 KB | 2 | 24.56% | **12.85 M** | 20.81 M | 52.8 M | 1.55 GB | 3.12 GB | 719 µs |
| `gate_tma_qkgp` | 107.90 KB | 2 | 24.57% | **12.85 M** | **11.46 M** | — | 1.55 GB | 3.15 GB | 709 µs |

- **机制 1（非 TMA 独有）= 占用率**：三张常驻 `[BT,KD]` tile（+24 KB）把 dynamic smem 从 92.7 KB 顶到
  117.3 KB，占用率上限从 **2 掉到 1 block/SM**（`warps_active` 24.2%→12.4%）。这就是非 TMA 那 1.5× 的主因
  ——pad 行距把 bank conflict 从 19.94 M 打到 10.51 M（≈基线），`Duration` 仍 1000 µs（1.49×），占用率没回来。
- **机制 2（两版共有）= 合流本身的代价**：TMA 版 smem 107 KB 仍保住 2 blocks/SM（`warps_active` 24.6%），
  global ld sector 从 **19.40 M→12.85 M（−34%）**——**预期收益确实兑现**（且 `gate_tma_qkgp` 的 bank conflict
  11.46 M 已回到基线）——可 `Duration` 反升 **652→709 µs（+8.7%）**、DRAM 读一字未动（1.56→1.55 GB）。
  **原因**：本核是 **DRAM/L2 双 bound**（DRAM SOL ~86%、L2 SOL ~88%），那 1.3 GB 的 l1tex 散读**大头被 L2
  吸收**（DRAM 读不变），删掉它并不动瓶颈；而 staging 多出一遍 smem 写 + 一个 block 同步 + 额外指令，
  净负。
- **XU 完全不变**（`sm__inst_executed_pipe_xu` 恒 1.507 M）：合流不改 `exp2` 发射，故 `exp2f` 那部分成本
  与取数无关。SASS 里 `LDG 62→49/48→35`、`LDS 164→190 / 98→124` 也印证「global 读换成 smem 读」，
  值没变、只是搬家。

### 61.5 结论 / 修正 §60.4 的归因

- **§60.5 的「合流 `q/k/g` 收回门控 +30%」证伪**：合流载入 smem **是净亏**（非 TMA 1.11–1.51×、TMA 1.08–1.19×，
  逐位无损但更慢）；pad 行距只把大 shape 的 1.51→1.49、1.09→1.08，**不改符号**。
- **修正 §60.4 的归因**：门控那 9.2× 的 global-load sector **是真的、但不在关键路径上**（被 L2 吸收，
  DRAM 读不变）；门控 +30% 的成本是 **三份 `[B,T,H,K]` 输出写（199 MB，交付物本身）+ `exp2f` XU 发射 +
  本核本就 DRAM/L2 bound**，**不是** raw `q/k/g` 取数的合流度。§60.4 那句「大头是散读」应改为
  「散读真实存在于 L1 计数，但被 L2 吸收、不构成端到端成本」。
- **对目标卡的设计输入（更新）**：门控链**不必**为 `q/k/g` 单设 smem staging（徒增占用/同步，见机制 1/2）；
  生产 B2 里 `k`/`g` 已为 `kg`/`db`/`dA` 载入，`q` 的少量散读**保持直读 global 更划算**。若一定要合流，
  必须**不新增常驻 smem 峰值**（复用既有缓冲、或让 staging 随 TMA 阶段滚动）且**验证不掉占用率**——
  但本轮已证即便做到（TMA+pad，占用率不掉、冲突回基线）仍 +8.7%，故**不作为优化项**。

### 61.6 边界（诚实）

- 只覆盖 `K=V=128`、`BT=16`、fp32；`B2M_QKG` 需 `B2M_GATE`。`sq/skb/sgb` 为常驻 `[BT,QKGP]`（`QKGP∈{128,132}`），
  `-DB2M_QKGP` 仅用于受控归因（消 bank conflict），非部署项。
- 「DRAM/L2 双 bound → 删 L1 散读不兑现」这一归因由 `gate_tma`（一对换算子只改取数来源、occupancy 不变）
  作受控对照，故非小 shape 噪声；小 shape（`T512/T250`）合流更慢还叠加了 staging 的固定延迟。

## 62. 「K2 形状 CUDA B2」把非 GEMM 段的 store 全部向量化：逐位无损、小 shape −8~16%（P7-59）

§51（P7-48）在 B2 **GEMM 核**上确立过：默认 store 是 25 条**逐标量 32-bit `STG.E`**，4 个 lane 各写 4 B 到同一
sector 的 `{0,8,16,24}` 偏移 ⇒ **每条 store 只覆盖半个 sector**（`l1tex st` 扇区 = 数据扇区的 **1.98×**），
换成合成的 64-bit **`float2` store**（`-DB2M_VECST`）后立刻回到 1.0×、**逐位相同**、且免费偏正（0.87–0.99×，
小 shape 10%+）。但 P7-52（`dv2/dvb`）、P7-53（`dkgb`）、P7-56（`db`）、P7-57（门控 `dqg/dkg/dg`）四轮
各自建在那支核上时，**本段的 store 都留成了标量**——每一轮的「边界」都写着「本段 store 未接 `-DB2M_VECST`」。
本节把这条边界收口。

### 62.1 设计：一个 `st2()` 成对 store，覆盖四段的 token-major 大输出

- `st2(float* p, float a, float b)`（`b2_mma_probe.cu:96`）：`B2M_VECST` 下一条
  `*reinterpret_cast<float2*>(p) = make_float2(a,b)`，否则两条标量 `p[0]=a; p[1]=b;`。成对的两列在
  store 里**地址相邻且列号为偶**（`KD=128` 且 `col=warp*BW+nt*8+2*t` 恒为偶）⇒ `float2` 天然 **8 B 对齐**。
- 落到四处（都只换取数/存数指令，**算术一字未动**）：
  - 输出支路 `dv2/dvb`（`:934-940`）；
  - `dkgb`（`:952-953`）；
  - 门控链 `dqg/dkg`——把原先 i-循环内逐元素 store 改成先写 `dqgv/dkgv[2][4]`（`:967`），循环后按行对
    成对存（`:1011-1014`）；`dg`（`:1054-1055`）——每行的 `last`-行补偿 `add`（`sh->sdgkt`）按列取 `dl0/dl1`，
    成对后即 `(dgv+a00, dgv+a01)`。
- `[BT,BT]` 的 `dA`/`dApre`（`B2M_DA`/`B2M_SAND`）**未动**：它是 `[B,T,H,BT]` 小张量、且是
  thread-per-entry 布局（一 thread 一个元素），向量化收益远小于四段的 `[B,T,H,K]`/`[B,T,H,V]` token-major 大输出。

### 62.2 证据

`harness/b2_mma_probe.cu`（新增 `st2` + 四处改写）+ `harness/check_bwd_b2_vecst_seg.py`
（`harness/check_bwd_b2_vecst_seg.out.txt`，空闲卡 GPU1）。**同一份编译源**跑两遍：base（无 `B2M_VECST`）
vs vecst（`-DB2M_VECST=1`），四段各建一份的 base/vecst 同源对照。

### 62.3 Part A — 正确性：全部段、全部 shape **逐位相同（`max_abs=0`）**

四组 shape（`B8 T2048 H8` / `B2 T1024 H8` / `T512 H4` / `T250 H3` 非整除）逐输出 `base vs vecst`：

| 段 | 输出 | 四组最坏 `max_abs` |
|---|---|---|
| `out` | `dq`/`dk`/`dw`/`dv2`/`dvb` | **0 / 0 / 0 / 0 / 0** |
| `dkgb` | `dq`/`dk`/`dw`/`dkgb` | **0 / 0 / 0 / 0** |
| `db` | `dv2`/`dvb`/`dkgb`/`db` | **0 / 0 / 0 / 0** |
| `gate` | `dqg`/`dkg`/`dg` | **0 / 0 / 0** |

即向量化**逐位无损**（含非整除 `T=250`）；`db` 的 `db` 输出（`B2M_VECST` 未触及、thread-per-row 布局）也
逐位相同。门控 vecst 对 FLA 的最坏 `max_rel` `dqg 1.051e-3 / dkg 9.466e-4 / dg 1.010e-3`——**与 P7-57 的
tf32 底噪逐值一致**，证明换 store 指令不动语义。

### 62.4 Part B — 计时：**全为正**（vecst/base，CUPTI `dev`）

| 段 | `B8 T2048 H8` | `B2 T1024 H8` | `T512 H4` | `T250 H3` |
|---|---|---|---|---|
| `out` | 0.983× | 0.950× | **0.835×** | **0.865×** |
| `dkgb` | 0.990× | 0.962× | **0.876×** | **0.887×** |
| `db` | **0.940×** | **0.943×** | **0.942×** | **0.913×** |
| `gate` | 0.987× | 0.959× | **0.919×** | **0.900×** |

- **所有 16 个格子全为正**：小 shape（`T512/T250`，store 指令数占比高、且本就延迟主导）**−8~16%**，大 shape
  （`B8T2048H8`，DRAM/L2 双 bound、store 字节被带宽掩盖）−1~2%。
- `db` 段（= OUT+DKGB+DB，一次把 `dv2/dvb/dkgb` 三个大张量都向量化）在大 shape 也吃到 **6%**，是全表大 shape
  最大收益——因为它的标量 store 密度最高。

### 62.5 机制：SASS 从 50 条标量 `STG.E` 降到 24 条 `STG.E.64`

`db` build 的 `cuobjdump -sass`（同源 base vs vecst）：

| 变体 | `STG.E`(32-bit) | `STG.E.64` |
|---|---|---|
| base | **50** | 0 |
| vecst | 2 | **24** |

50→24+2：`dv2/dvb`（4×2×2=… per row pair）+ `dkgb` 的成对 store 合成 64-bit；残留的 2 条标量是
`dgk`（列归约，非成对）与 `db_out`（thread-per-row）——都非 `float2` 形态，符合 §62.1 的边界。与 §51.4
「半 sector 标量写」是同一课：**store 一律向量化**，本次把 B2 四段补齐。

### 62.6 结论 / 边界

- **收口 §51.6 的「向量化 store 一律做」到非 GEMM 四段**：`dv2/dvb`、`dkgb`、门控 `dqg/dkg/dg` 全改
  `float2`（`st2`），**逐位无损**、小 shape **−8~16%**、大 shape −1~6%。目标卡 CUDA B2 的**所有 token-major
  store** 都应走 64-bit（乃至更宽）向量化。
- **边界**：`[BT,BT]` 的 `dA`/`dApre` 未向量化（小张量 + thread-per-entry 布局）；`dgk`/`db_out` 非成对形态
  仍标量；仅覆盖 `K=V=128`、`BT=16`、fp32。至此 §51.6 的设计输入在 B2 全核（含四段非 GEMM）落地。

## 63. 「K2 形状 CUDA B2」六段合成**单一生产 launch**：逐位无损、一次 launch 摊掉两份重复的 GEMM 核（−47~53%）（P7-60）

P7-52…P7-59 把 B2 的六段非 GEMM（输出支路 `dv2/dvb`、`dkgb=A@(-dw)`、`dA` 的 `dApre` 与
`-A^T diag(β) A^T` sandwich、写入门 `db`、门控链 `dq/dk/dg`）**逐段**搬进了 K2 形状 CUDA 核
（`harness/b2_mma_probe.cu`），但每一段都是**独立的编译变体 + 独立 launcher**（`b2out_launch` /
`b2dkgb_launch` / `b2da_launch` / `b2sand_launch` / `b2db_launch` / `b2gate_launch`）——上一轮的
「下一步」把这条留作**唯一剩余的真结构项**。事实上要把六项梯度全部产出，老设计至少要跑 **3 次** b2 核
（`db` = GEMM+OUT+DKGB+DB、`sand` = GEMM+DA+SAND、`gate` = GEMM+DKGB+GATE），每次各自重做一遍**同一个**
GEMM 核还有各自的一次 launch。本节给出**单一生产 launch** `b2prod_launch`（`-DB2M_PROD`），并证明
合并**逐位无损**、且把重复的 GEMM 核从 3 份摊成 1 份。

### 63.1 设计：六段旗标取并集 + `beta` 只声明一次，段体一字未动

- `-DB2M_PROD`（`b2_mma_probe.cu:129-142`）在**所有段旗标定义之前**把 `B2M_OUT/DKGB/DA/SAND/DB/GATE`
  六个全部 `#define` 为 1；各段原有的 `#ifndef`/`#error` 依赖检查因此全部通过。
- 六段的**代码体一行未动**：它们的累加器（`accq/acck/accw`、`acc2`、`acc3`、`dbpart/dkpart`、
  `dgv/dqgv/dkgv`）、smem 缓冲（`skg`、`swall/dapart/da16`、`dbrow/dbv`、`sA/sbeta/daB`、`sdgkt`）与输出
  （`dv2/dvb`、`dkgb`、`dApre/dA`、`db`、`dqg/dkg/dg`）**两两不相交**，所以并集天然无冲突。
- 唯一需要的共享改动：原先 `beta` 在 `B2M_OUT`、`B2M_GATE`、`B2M_SAND` 三处各声明一次（单独编译时各出现
  一次、合法），合并后变成**同名重复参数**。改成**只声明一次**——`beta` 在各段自己的位置按
  `#if B2M_GATE && !B2M_OUT` / `#if B2M_SAND && !B2M_OUT && !B2M_GATE` 让位给 `B2M_OUT` 处的那一份
  （`b2_mma_probe.cu:588-591/595-597/610-614`）。这样**既有的六个隔离 launcher 签名全部不变**，只有 `prod`
  看到并集后的参数顺序。
- 新增 `b2prod_launch`（`b2_mma_probe.cu`，`#if B2M_PROD`）：按宏展开后的 `b2mma_kernel` 签名给出联合参数
  表，含 TMA 七张张量图（`dom/vnm/dvm/hm/dhm/dwm/vvm`）。隔离 launcher 在 `prod` 下用 `&& !B2M_PROD`
  关掉，避免与并集签名冲突。

### 63.2 Part A — 正确性：`prod` 的**每一个**输出与对应隔离 build **逐位相同（`max_abs=0`）**

四组 shape（`B8 T2048 H8` / `B2 T1024 H8` / `T512 H4` / `T250 H3` 非整除），`prod`（一次 launch）对比
隔离 build：`dv2/dvb/dkgb/db` vs `db` build、`dApre/dA` vs `sand` build、`dqg/dkg/dg` vs `gate` build、
`dw/dgk` vs `core` build；另有 `prod_tma`（`-DB2M_PROD -DB2M_TMA`）vs `prod`（cp.async）。

| 字段 | 对照 | 四组最坏 `max_abs` |
|---|---|---|
| `dv2`/`dvb`/`dkgb`/`db` | `prod` vs `db` | **0 / 0 / 0 / 0** |
| `dApre`/`dA` | `prod` vs `sand` | **0 / 0** |
| `dqg`/`dkg`/`dg` | `prod` vs `gate` | **0 / 0 / 0** |
| `dw`/`dgk` | `prod` vs `core` | **0 / 0** |
| `dv2`/`dqg`/`dA` | `prod_tma` vs `prod` | **0 / 0 / 0** |

即**合并逐位无损**（含非整除 `T=250`；TMA 与 cp.async 两布局也逐位一致）。这与「段体一行未动」自洽：合并
只改了参数/旗标组合，未碰任何 mma、归约或 store 的算术与顺序。

### 63.3 Part B — 资源：`prod` 约 2× 寄存器但**零 spill**，smem 133.8 KB（1 block/SM）

`nvcc -Xptxas -v` + `b2mma_smem_bytes()`：

| build | regs | spill | smem(B) |
|---|---|---|---|
| `core` | 70 | 0 | 87 552 |
| `db` | 80 | 0 | 113 664 |
| `sand` | 80 | 0 | 119 936 |
| `gate` | 74 | 0 | 92 672 |
| **`prod`** | **140** | **0** | **133 760** |
| `prod_tma` | 128 | 0 | 123 008 |

六段并集把寄存器从单段的 70–80 抬到 140（各段累加器同时保活），但**没有溢出**——这是关键：P7-45/§48
曾证「大操作数常驻寄存器 → spill/发射饥饿」是融合核头寸的来源，而本轮六段并集**仍在 255 上限之内**。
`smem` 133.8 KB × `ncu` 的 `launch__occupancy_limit_shared_mem=1` ⇒ **1 block/SM（12.5% occupancy）**，
与 GEMM 核基座一致。

### 63.4 Part C — 计时：一次 `prod` launch vs 老设计的 3 次 launch，**0.47–0.53×**

CUPTI `dev`（`profile_dev`，每次计时跑 15 次取 `/rep`，空闲卡 GPU1）：

| shape | `prod`（1 次 launch） | `db`+`sand`+`gate`（3 次 launch） | `prod/old` |
|---|---|---|---|
| `B8 T2048 H8` | 1288.2 µs（15 个 b2 核） | 2450.4 µs（45 个 b2 核） | **0.526×** |
| `B2 T1024 H8` | 169.7 µs | 342.2 µs | **0.496×** |
| `T512 H4` | 20.1 µs | 42.2 µs | **0.477×** |
| `T250 H3`（非整除） | 18.4 µs | 39.4 µs | **0.467×** |

**机制**：老设计 3 次 launch 各自**重做一遍同一个 GEMM 核**（每次 15 个 b2 核 × 3 = 45），`prod` 只做一次
（15 个）。省下的正好是 **2 份 GEMM 核 + 2 次 launch**，与 0.47–0.53×（≈ −47~53%）一致。小 shape
（`T512/T250`）与大 shape 的比值接近，说明该收益是**结构性**的（核复用），非带宽/尺度相关。

### 63.5 `ncu` — 合并后仍**非 DRAM-bound**，是标量/发射受限（tensor 3%、occ 12.5%、DRAM 45%）

`harness/probe_b2_prod_ncu.sh`（`harness/ncu_b2_prod_ncu.out.txt`，`prod_tma`，空闲卡 GPU1）：

| 指标 | `B8 T2048 H8` | `B2 T1024 H8` |
|---|---|---|
| `gpu__time_duration` | 1.39 ms | 181.95 µs |
| DRAM 读 / 写 | 1.62 GB / 485 MB | 203 MB / 54 MB |
| `dram` / `lts` SOL | 45.4% / 55.8% | 42.1% / 53.2% |
| `sm` SOL / `tensor` | 23.9% / **3.07%** | 22.5% / **2.89%** |
| `warps_active` | 12.50% | 12.50% |
| regs / occ-limit-smem | 128 / 1 | 128 / 1 |

合并后的核**不再是 isolated GEMM 核那种 84.6% DRAM bound**（§49）：六段非 GEMM 基本是**标量/片元式
访存 + `exp2f`**，把 DRAM SOL 压到 45%、tensor 仍是 3%。也就是说 `prod` 的定位与 P7-57/58 的「B2 全核
DRAM/L2 双 bound、但含大量标量段」一致——**Part C 的 ~2× 是核复用/launch 摊薄的直接结果，不是换了瓶颈
regime**。合并把六个隔离核的「每人一次 GEMM 核」收敛成一次，本身就是把 P7-46（GEMM 核 0.93× Triton、
贴 DRAM 地板）的成果摊销到全部六段。

### 63.6 结论 / 对目标卡的设计输入

- **六段合成单一生产 launch 成立**：`b2prod_launch` 一次 launch 产出 B2 全部六项梯度 + raw `dw/dgk`，
  与六个隔离 build **逐位相同**（Part A，含非整除 T、两种 stage 布局），**零 spill**（Part B），
  相对老设计 3 次 launch **0.47–0.53×**（Part C）。公开的 §55.6/§56.6/§57.6/§58.6/§59.6/§60.5 一直点名的
  「各段仍分 launcher」这条**结构项关闭**。
- **设计输入**：目标卡 CUDA B2 的**生产形态就是这一个核**——GEMM 核做基座、六段在同一 `(NT, B·H)` grid
  内以各自的 warp 分片/寄存器累加器追加，`beta` 等共享操作数只入参一次；无需为每段单独起 launch。
- **边界 / 下一步**：本轮只验证「合并 == 隔离」与 launch 摊薄，未改变各段内部实现；`prod` 仍 1 block/SM
  （smem 133.8 KB 钉住）、且六段的标量/片元访存仍是主要 headroom（tensor 3%）。若继续，剩下的真结构项是
  §49.7 早已点名的**同 CTA 内交接**（`dw` 不落盘而是就地喂 `dkgb`/`dA`，`dv2`/`v` 不落盘喂 `db`）——
  本节把「六段共存于一个核」这个前提做实了，同 CTA 交接才具备落脚点。

## 64. 「K2 形状 CUDA B2」的 **同 CTA `dw` 交接**（`-DB2M_HDW`）：逐位无损、`dw` 中间量从 ABI 消失；隔离段快 7–9%，但六段生产核里是 wash（P7-61）

§49.7 与 §63.6 把「同 CTA 内交接」点名为 B2 六段合成后的**最后一条真结构项**：六段已共存于一个核
（P7-60），但 DKGB 段仍然消费一个**全局** `dw`（`dkgb = A @ (-dw)`，`chunk_bwd.py:305`），而这个 `dw`
恰恰是**同一个核**里 GEMM 基座刚算出来的 `accw`（`dw = dv @ h^T`）——即核心把 `dw` 写全局、
DKGB 段再从全局（经 `cp.async`/TMA）搬回来。本轮加 `-DB2M_HDW` 把这趟往返**彻底删掉**：核心的 `dw`
（`accw`）留在寄存器、镜像进 `swall`，DKGB 的 mma 改**在 it-loop 之后**从该镜像读。`dw` 因此从生产
ABI 里消失（不再需要调用方预算 `dw`、也不再落盘）。

### 64.1 设计：DKGB 的 mma 后移 + 读 `accw` 的 smem 镜像，`dw` 的 staging / 输出 store 全部删掉

- **为什么必须后移**：DKGB 的收缩维是 chunk 行 `o_t`，其 B 操作数是**完整**的 `dw[o_t, k]`（对所有
  V 列求和）。但 `accw` 是沿 V 逐 stage 累加的（`b2_mma_probe.cu:760-771` 的 `for kk` 在 `for it`
  内），只有在**所有** V-tile 累加完之后才完整。原 DKGB 段（`b2_mma_probe.cu:847-875`）之所以能在
  某个 stage 内跑，是因为它读的是**预先算好、按 V-tile stage**的全局 `dw`（每个 stage 都有全部
  token 行、只覆盖那一段 k 列）。交接后这个前提没了，DKGB 只能后移。
- **镜像**：后移的 DKGB 段（`b2_mma_probe.cu:997-1030`）先把本 warp 的 `accw` 片段（行 `g/g+8`、
  列 `nt*8+2t/2t+1`）写进 `swall[NW][BT][BW]`，`__syncwarp()` 后按 `-sh->swall[warp][o_t][lc]`
  （`lc = nt*8+g`，warp `w` 的 k 列恰好是其 own `swall` slice）喂 mma——**与 P7-54 的 `dw@kg^T`
  用的是同一份镜像**（`swall` 的分配由 `B2M_DA` 扩到 `B2M_DA || B2M_HDW`，`b2_mma_probe.cu:343-346`）。
- **删除的三处**：`Stage::dwm` 成员、`prefetch` 的 cp16 staging、TMA 的 `dwm` 张量图（含
  `expect_tx` 字节数）与核心的 `dw_out` 全局 store，全部按 `B2M_DKGB && !B2M_HDW` / `!B2M_HDW`
  条件关掉（`b2_mma_probe.cu:316/328/492/537/552/925/933/944/952`）。算术一字未改——只是把
  「写全局再读回」换成「写 smem 再读回」。
- **逐位无损是构造性的**：交接传的 `accw` **就是**原路径写进 `dw_out`、再被读回的同一个值（评测
  里 `db`/`gate`/`prod` 一直吃 `core.dw`），所以 DKGB 的 mma 操作数与累加序完全不变。

### 64.2 Part A — 正确性：`prod_hdw` 的每个输出与 `prod` **逐位相同**；且传进去的 `dw` 真的没被用

`harness/check_bwd_b2_hdw.py`（`harness/check_bwd_b2_hdw.out.txt`）。四组 shape（`B8 T2048 H8` /
`B2 T1024 H8` / `T512 H4` / `T250 H3` 非整除）：

| 字段 | 对照 | 四组最坏 `max_abs` |
|---|---|---|
| `dv2/dvb/dkgb/db/dApre/dA/dqg/dkg/dg/dgk` | `prod_hdw` vs `prod` | **0（全部）** |
| `dv2/dA/dqg` | `prod_hdw_tma` vs `prod_hdw` | **0** |

**Part A2（交接的自证）**：把 `prod_hdw` 的 `dw` 输入换成**全零**，十个输出与喂真实 `dw` 时
**逐位相同（`max_abs=0`）**——`dw` 输入在交接下**完全未被读取**，从侧面证明 DKGB 用的确实是核内
`accw` 而非任何外部/全局 `dw`。这是本轮最强的「交接成立」证据。

### 64.3 Part B — 资源：寄存器 140→110、smem 133.8→129.2 KB（零 spill）

`nvcc -Xptxas -v` + `b2mma_smem_bytes()`：

| build | regs | spill | smem(B) |
|---|---|---|---|
| `prod` | 140 | 0 | 133 760 |
| **`prod_hdw`** | **110** | **0** | **129 152** |
| `prod_tma` | 128 | 0 | 123 008 |
| **`prod_hdw_tma`** | 128 | 0 | **118 912** |

删掉 `dwm` 每-stage 缓冲（`NAT*BT*32*4` 或 `BT*BVP*4`）省下 ~4.1–4.6 KB smem，寄存器也降
（DKGB 后移使 stage 内不再同时保活 `dwm` 指针/地址）。但占用率**不变**——仍是 **1 block/SM**
（smem 128 KB 量级仍远超阈值）。

### 64.4 Part C/D — 计时：**隔离段快 7–9%，六段生产核里却是 wash**

CUPTI `dev`（`profile_dev`，每次 15 次取 `/rep`，空闲卡 GPU1）与 event-pair `wall`：

| shape | CUPTI `prod_hdw` | CUPTI `prod` | `hdw/prod` | `wall hdw/prod` |
|---|---|---|---|---|
| `B8 T2048 H8` | 1324.0 µs | 1288.4 µs | **1.028×** | 1.043× |
| `B2 T1024 H8` | 175.6 µs | 171.1 µs | **1.026×** | 1.027× |
| `T512 H4` | 20.2 µs | 20.4 µs | 0.991× | 1.014× |
| `T250 H3`（非整除） | 18.5 µs | 18.5 µs | 1.002× | 1.004× |

即大 shape 上交接**略慢 ~2.8%（CUPTI）/ ~4%（wall）**，小 shape 在噪声内。**但把 DKGB 单独拿出来**
（GEMM 基座 + DKGB 段，`Part E`）交接反而**稳定快 7–9%**：

| shape | `dkgb`（in-loop，全局 `dw`） | `dkgb_hdw`（post-loop，核内交接） | `hdw/base` |
|---|---|---|---|
| `B8 T2048 H8` | 570.5 µs | 521.6 µs | **0.914×** |
| `B2 T1024 H8` | 77.9 µs | 72.7 µs | **0.933×** |
| `T512 H4` | 10.8 µs | 9.8 µs | **0.905×** |
| `T250 H3` | 10.0 µs | 9.3 µs | **0.928×** |

**机制解释（两个数一致可解）**：交接**本身**是真的省——它删掉了 `dw` 的全局 store + staging 读
（在隔离的「核心+DKGB」里，staging 的开销没有别的东西可隐藏，故净赚 7–9%）。但在**六段齐全**的
`prod` 里，原 DKGB 段是**在 stage 内、与后续 stage 的 GEMM/TMA 流水重叠**的（它只需那一 stage 的
k 列，而其余 stage 还在跑），交接迫使 DKGB 后移成**串行尾巴**、再也无法重叠；而这趟全局往返在
`prod` 里本就**被 GEMM 流水藏住**了——于是「省下的字节」换不来时间，「丢掉的 overlap」却要还。
**结论：`dw` 的全局往返在隔离段里是实打实的一笔税，在完整六段生产核里不是**。

### 64.5 `ncu` — 字节确实降了（DRAM −126 MB / L2 −0.33 GB），但 ncu `Duration` 与 CUPTI/wall **反向**

`harness/probe_b2_hdw_ncu.sh`（`harness/probe_b2_hdw_ncu.out.txt`，`prod_tma` vs `prod_hdw_tma`，
空闲卡 GPU1）：

| 指标 | `prod_tma` B8 | `prod_hdw_tma` B8 | `prod_tma` B2 | `prod_hdw_tma` B2 |
|---|---|---|---|---|
| DRAM 读 | 1.62 GB | **1.56 GB** | 202.9 MB | **194.5 MB** |
| DRAM 写 | 485.4 MB | **419.3 MB** | 54.0 MB | **46.5 MB** |
| `lts` 字节 | 3.57 GB | **3.24 GB** | 447 MB | **407 MB** |
| `gpu__time_duration` | 1.38 ms | **1.29 ms** | 180.6 µs | **171.1 µs** |
| bank conflict / regs / occ-smem | 28.31 M / 128 / 1 | **28.31 M** / 128 / 1 | 3.54 M / 128 / 1 | **3.54 M** / 128 / 1 |
| tensor % | 3.07% | 3.28% | 2.90% | 3.05% |

- **字节账干净**：删掉的正是 `dw` 的输入 staging 读（~60 MB）+ `dw_out` 输出 store（~66 MB）≈
  **126 MB DRAM**，L2 少 **~0.33 GB**；bank conflict **完全不增**（交接只换取数来源）、occupancy /
  tensor 不动。`hdw` 仍非 DRAM-bound（DRAM ≤46% SOL、tensor ~3%），是标量/发射受限，与 §63.5 同。
- **口径冲突（诚实）**：ncu 自报 `Duration` 给 `prod_hdw` **更快（−7%）**，而 CUPTI `dev`/`wall`
  给的是**略慢（+2.8%/+4%）**——与 §25.5/§63.5 记录的「ncu `Duration` 与 CUPTI/wall 窗口不同源」
  一致（ncu 每次 replay 前清 L2，使那 60 MB `dw` 读变成真 DRAM 读，于是字节节省「兑现」成时间；
  背靠背的 CUPTI/wall 里它被流水藏住，只见串行尾巴）。**部署口径仍取 CUPTI/wall**：交接 = wash。

### 64.6 结论 / 对目标卡的设计输入

- **同 CTA `dw` 交接成立且逐位无损**：`prod_hdw` 的十个输出与 `prod` **逐位相同**（含非整除 T、
  两种 stage 布局），喂全零 `dw` 结果不变（`dw` 输入证实未被读）。§49.7 / §63.6 点名的最后一条结构项
  **收口**。
- **收益定性**：交接把 `dw` 从生产 ABI 删除（**不再需要预算/传参 `dw`、不落盘**），寄存器 140→110、
  smem −4.6 KB、DRAM −126 MB / L2 −0.33 GB、bank conflict 不增——**是内存足迹优化，不是速度杠杆**。
  隔离的「核心+DKGB」里净赚 **7–9%**，但在六段齐全的 `prod` 里**是 wash（大 shape +2.8%）**：
  `dw` 往返本就与 GEMM 流水重叠，而后移 DKGB 丢掉了 stage 内重叠。
- **设计输入**：目标卡 CUDA B2 的生产形态若**六段齐全**，`dw` 就地交接**可做可不做**（省中间量、
  不亏时间，利于显存/ABI）；若某实现里 DKGB 是**主导段**（或 GEMM 较浅、staging 不被隐藏），交接是
  **实打实的 7–9%**。判据＝「省下的 staging/store 是否本就被 GEMM 流水藏住」——这是继 §51.6「省字节
  要落最紧 pipe」之后对「省字节」的又一条前提检查：**还要看这笔访存是否在关键路径上**。
- **边界**：`dv2`/`v` 喂 `db` 的那半（`dbv` 镜像）在 P7-56 时就已是核内寄存器/smem（无全局往返），
  故本轮只动了 `dw` 一项；`dApre`/`dA` 在 P7-55 起也不落盘即入 sandwich。至此六段的中间量**要么本就
  核内、要么可核内**，且六段能一次 launch 产出（P7-60）——B2 的 K2 形状 CUDA 化在**结构上闭合**，
  剩余头寸（tensor 3%、标量/发射受限）不在这些中间量的搬运上。

## 65. B3 `intra` 折进 **K2 形状**（CUDA）：部署粒度下 B3 只剩对角，做成 `grid (NT, B*H)` / warp-per-k-column 的核，比 Triton MMA B3 快 1.3–1.45×（P7-62）

`ROADMAP`/§60.5 把「另一条「B3 intra 折进 K2 形状」仍待」列为 K2 形状 CUDA 化**最后一条**结构项。
本轮兑现它：`harness/b3_intra_probe.cu`（K2 形状）+ `harness/check_bwd_b3_intra_cuda.py`
（`harness/check_bwd_b3_intra_cuda.out.txt`）+ `harness/ncu_b3_intra_cuda_driver.py` /
`harness/probe_b3_intra_cuda_ncu.sh`（`harness/probe_b3_intra_cuda_ncu.out.txt`）。

### 65.1 为什么部署粒度下 B3 只是「两块对角 GEMM」

由 §27.1：B3 的 grid 是 `(NK*NC, NT, B*H)`、`NC=ceil(BT/BC)`。部署 `BT=BC=CHUNK=16 ⟹ NC=1 ⟹ i_i≡0`，
Part A（`i_i>0`）与 Part C（`i_i<NC_eff-1`）**永不进入**——B3 每 program 执行的全部计算就是 Parts B/D
那两个对角循环。§27 已证它们等价于两个 `tl.dot`（FLA `SAFE_GATE=True` 分支早已如此），并把 Triton
默认路径 MMA 化。本轮把这个**对角 MMA 恒等式**从 Triton 搬到 CUDA、并放进 **K2 形状**。

### 65.2 设计：K2 形状 + 四个 `[BT,BT]@[BT,KD]` 的 `mma.sync.tf32`

`grid (NT, B*H)`，一块管一个 chunk；8 warp，warp `w` 拥有输出 k 列 `[w*16, w*16+16)`（与 B2
`b2_mma_probe.cu` 核心的 warp/累加器分片**逐字同构**）。每 chunk 把 `q/k/g` `[BT,KD]`、`beta` `[BT]`
与两块对角伴随矩阵（`dAqk` 来自 `dAv`，`dAkk` = B2 落盘的 sandwich `dA`）`[BT,BT]` 合流进 smem，然后
（`NC=1`，全部为**对角**工作）：

```
Mq = tril(dAqk),  Mk = tril(dAkk),  Sq = triu(dAqk^T),  Sk = triu(dAkk^T)      [BT,BT]
KG = k·2^{-g},    QG = q·2^{g},     KGd = k·beta·2^{g}                        [BT,KD]
dq2 = 2^{g} ⊙ (Mq @ KG)                      (Part B, query 支路)
dk2 = 2^{g} ⊙ (Mk @ KG)                      (Part B, key 支路)
db += Σ_k dk2·k ;   dk2 *= beta              (写入门伴随)
dg  = q ⊙ dq2                                (衰减链：\bar g += \bar q^dec ⊙ q)
dkt = 2^{-g} ⊙ (Sq @ QG + Sk @ KGd)          (Part D)
dg += (dk2 - dkt) ⊙ k ;   dk2 += dkt
```

四个 `[16,16]@[16,16]`（warp 切片）矩阵乘走 `mma.sync.m16n8k8.tf32`（匹配 `tl.dot` 默认 tf32）；
门控 `2^{±g}` 在**累加器片元布局**里逐元素做（行 `grp/grp+8`、列 `nt*8+2t+(i&1)`）。`Mq` 的 A 片元直接
按 `sAqk[grp/grp+8][j]` 读；`Sq` 是 `dAqk` 的**转置**，A 片元按 `sAqk[j][grp/grp+8]` 读（转置被
`mma` 的 A 行主序代数「吃掉」，无需 smem 转置）。`db` 是先按 warp 拥有列求和、再跨 8 warp 归约
（与 P7-56 的 `db` 段同构）。**纯 fp32 的 `beta` 缩放与门控在核内**，只有四个矩阵乘落 tf32。

这一版是**独立可核验**的（自带输入/输出，不依赖 B2 核心的其它段），故能与 Triton MMA B3 在同一份
`dAqk/dAkk` 上**位级对拍**。

### 65.3 Part A：正确性——四项全落 tf32 底噪（含非整除 T、g=0）

`check_bwd_b3_intra_cuda.out.txt`，5 组 shape（`T=512/1024/2048`、`T=250` 非整除、`T=64` 单 chunk）。
CUDA K2 核 vs Triton MMA B3（同操作数）、vs per-chunk torch golden 的**最坏 `max_rel`**：

| 对比 | `dq` | `dk` | `db` | `dg` |
|---|---|---|---|---|
| CUDA vs Triton MMA | 1.27e-3 | 1.48e-3 | 1.22e-3 | 1.42e-3 |
| CUDA vs torch golden | 5.0e-4 | 6.4e-4 | 4.4e-4 | 5.3e-4 |
| Triton MMA vs torch golden | 1.13e-3 | 1.21e-3 | 1.13e-3 | 1.06e-3 |

CUDA 与 Triton 都在 §27 记录的 **tf32 底噪 ~1e-3** 上；CUDA vs torch golden 甚至更紧（≤6.4e-4，因为
**CUDA 的 `beta` 缩放/门控走 fp32 标量、只有矩阵乘落 tf32**，而 torch golden 是纯 fp32）。`g=0`（无衰减，
§27.2 说这是移出指数因子的稳定极限）下 CUDA vs Triton 仍 `dq 1.06e-3 / dk 1.05e-3 / db 1.03e-3 /
dg 1.18e-3`——**恒等式在 CUDA 上复现**。

> **踩到的坑**：第一版 `dq` 对（`Mq@KG` 门控对）而 `dk/db/dg` 全错到 O(10)。根因＝门控循环里只写了
> `adq *= 2^{g}` **漏了 `adk *= 2^{g}`**——`dk2` 的 `2^{g}` 因子没乘，误差随 `g` 幅度（可达 2^-8）放大。
> 指纹＝「同一 B 操作数/同一门控下，只 `dq` 对、`dk/db/dg` 同比例偏大」。

### 65.4 Part B：孤立计时——比 Triton MMA B3 快 **1.3–1.45×**

`check_bwd_b3_intra_cuda.out.txt`（CUPTI `dev` + event-pair `wall`，均值；Triton 取 `nw∈{2,4}` 最优）：

| case | CUDA dev (µs) | Triton dev (µs) | CUDA/Triton |
|---|---|---|---|
| `T=512 H=4` | 6.6 | 9.5 | **0.70×** |
| `B2 T=1024 H=8` | 35.0 | 44.6 | **0.79×** |
| `B8 T=2048 H=8` | 240.2 | 349.5 | **0.69×** |
| `T=250 H=3`（非整除） | 6.1 | 8.9 | **0.69×** |
| `T=64 H=1`（单 chunk） | 5.7 | 7.7 | **0.75×** |

全 shape 一致快 **1.3–1.45×**。机制见 Part D：Triton 的对角平铺 `[BC,BC]` 加载局部性差
（L1 命中 18.5%），CUDA K2 把 `q/k/g` 一次合流进 smem、8 个 warp 共享（L1 命中 51.4%）。

### 65.5 Part C/D：`ncu`——CUDA 侧 L1 局部性更好、DRAM 反更低、发射等待更短

`probe_b3_intra_cuda_ncu.out.txt`，`B8 T2048 H8 BT=16`：

| 指标 | Triton MMA B3 | **K2 CUDA intra** |
|---|---|---|
| Duration | 295.6 µs | **239.4 µs** |
| DRAM Throughput | 61.3% | **50.4%** |
| L1/TEX Throughput | 43.1% | **66.4%** |
| L1/TEX Hit Rate | 18.5% | **51.4%** |
| `tensor` pipe | 2.26% | 2.83% |
| `fma` pipe | 7.07% | 11.87% |
| Compute (SM) | 20.7% | 31.2% |
| `stall long_sb` | 19.89 | **11.37** |
| Registers/thread | 72 | **59** |
| Achieved occupancy | 42.6% | 48.3% |
| Memory throughput | 2.05 TB/s | 1.69 TB/s |

读法：两个核都**不是** tensor/core bound——对角 intra 在部署粒度下**总量太小**（`tensor` 2–3%、`fma`
7–12%）。CUDA 优势来自**访存侧**：`q/k/g` **一次合流入 smem 并被 8 warp 共享**（L1 命中 18.5%→51.4%），
故 `long_scoreboard` 近腰斩（19.9→11.4）、DRAM% 更低（61.3→50.4%），Duration 随之 **0.81×**（ncu 口径；
CUPTI Part B 为 0.69×，两窗口不同源，方向一致，同 §25.5/§63.5）。寄存器也更省（72→59）、占用率略高。
**注意**：B3 在部署粒度是「小的对角尾巴」，因此**它的绝对头寸有限**——把它做成 K2 形状的价值主要是
（a）结构闭合（B2/B3 都成 K2 形状、共享 warp 分片与 staging 约定）、（b）比 Triton 版快 1.3–1.45×。

### 65.6 结论与下一步

- **§60.5 最后一条结构项兑现**：B3 的对角 intra 现在是 **K2 形状的 CUDA 核**（`grid (NT, B*H)`、
  warp-per-k-column、四个 `mma.sync.tf32` + fp32 门控），与 Triton MMA B3 在 tf32 底噪上一致
  （含非整除 T、g=0），孤立计时快 **1.3–1.45×**。
- **还没做（下一步）**：把它**并进生产 launch**——即在 `b2_mma_probe.cu` 的 `-DB2M_PROD` 里，用同一块
  chunk 的 `dA`（`B2M_SAND` 的核内结果）、`dAqk` 输入与已 staging 的 `q/k/g` 算出对角贡献，**加到门控
  段最终的 `dq/dk/dg` 上**、并进 `db`（`B2M_DB`）。这需要处理**寄存器/smem 峰值**（prod 已是 129–134 KB
  smem / 110–140 regs、1 block/SM）与**累加器共享**（本轮的 `adq/adk/adt` 是独立的 `[2][4]`×3），是本
  轮的后续工程步，不改本轮已证的代数。做成后 B2/B3 就不再是两个 kernel、`dA` 也不必落盘（与 §33 的
  Triton 融合对应）。
- **产物**：`harness/b3_intra_probe.cu` + `harness/check_bwd_b3_intra_cuda.py`
  （`check_bwd_b3_intra_cuda.out.txt`）+ `harness/ncu_b3_intra_cuda_driver.py` /
  `harness/probe_b3_intra_cuda_ncu.sh`（`probe_b3_intra_cuda_ncu.out.txt`）。

## 66. B3 `intra` 折进**生产 launch**（`-DB2M_INTRA`）：逐位无损、单 kernel、单次 launch —— 但大 shape 不是净赢（P7-63）

§65.6 留下的**唯一**工程步：把 P7-62 的 K2 形状 B3 `intra`（`b3_intra_probe.cu`，孤立核）并进
`b2_mma_probe.cu` 的 `-DB2M_PROD` 生产 launch —— 用同一块 chunk 的**核内** sandwiched `dA`
（SAND 落的 `daB`）、`dAqk` 输入与已可用的 `q/k/g` 算出对角贡献，加到门控段最终的 `dq/dk/dg` 上、
并进 `db` 归约。本轮兑现它：新旗标 `-DB2M_INTRA`（要求 `-DB2M_PROD`；`b2_mma_probe.cu:217-230`）、
新入口 `b2prod_intra_launch`（多一个 `dAqk` 入参，`:1854-1871`）。产物：
`harness/check_bwd_b2_intra.py`（`harness/check_bwd_b2_intra.out.txt`）+ `harness/ncu_b2_intra_driver.py` /
`harness/probe_b2_intra_ncu.sh`（`harness/probe_b2_intra_ncu.out.txt`）。

### 66.1 设计：核内 `dAkk`、寄存器 merge、一次 launch

关键观察（§65.1/§65.2）：部署 `BT=CHUNK=16 ⟹ NC=1` 下 B3 只剩**对角**那两个 `tl.dot` 等价式。折进方式：

1. **`dAkk` 用核内结果**：`B2M_SAND` 落盘前 `daB[i][j]` 正是 sandwiched `X2`，而最终 `dA` =
   `where(i>j, -X2, 0)`（`b2_mma_probe.cu:1124-1132`）。故 intra 直接以
   `SAKK(a,j) = (a>j & valid) ? -daB[a][j] : 0`（`:1195`）读 B 的 A/B 片元，**无需 `dA` 落盘再读回**。
2. **`dAqk` 进 smem**、`q/k/g` 合流进 `siq/sik/sig`（`:393-401`，`q/k/g` 行 `t<T` 置零），
   唯一新输入是 `dAqk`。
3. **四个 `[BT,BT]@[BT,KD]` tf32 mma**（`:1198-1225`）逐式照抄 `b3_intra_probe.cu`，门控在累加器片元
   布局里逐元素做（`:1229-1254`），产出 `idq/idk/idg[2][4]`。
4. **merge 到门控**：PROD 的 gate 片元布局（行 `g/g+8`、列 `nt*8+2t+(i&1)`）与 intra **逐位同构**，
   故 `dqgv += idq`、`dkgv += idk`、`dgv += idg`（`:1319-1326`），`dg` 的 `m_last` 项照旧。
5. **merge 到 `db`**：intra 的 `Σ_k dkpre·k` 按 warp 列求和写 `idbrow[NW][BT]`（`:1260-1270`），
   在 `db` 段既有的跨 warp 归约里一并累加（`:1423-1426`）——**不新增 `db` 的 `__syncthreads`**。

为让 intra 读得到 SAND 的 `daB`、又能把片元 merge 进 GATE，把 **DA + SAND 两块整体前移到 GATE/DB 之前**
（纯文本重排，段内算术一字未动）；段间 buffer 两两不相交，新增仅 `siq/sik/sig`（24 KB）、`sda`（1 KB）、
`idbrow`（0.5 KB）。

### 66.2 Part A：折进 vs「prod + 独立 B3」**逐位相同**（`dg`/`db` 仅 fp32 加法序 ulp）

`check_bwd_b2_intra.out.txt`，6 组 shape（含非整除 `T=250`、单 chunk `T=64`）。把折进结果与
「`prod.<fld> + b3.<fld>`」（两者各自已锚定 torch/FLA）比，最坏 `max_abs`：

| 对比 | `dqg` | `dkg` | `dg` | `db` |
|---|---|---|---|---|
| `prod_intra` vs `prod + b3` | **0** | **0** | 4.8e-7 | 9.5e-7 |

`dq`/`dk` 的 merge 与参考同序（`dqf + idq`）故**逐位**；`dg` 在 gate 里先加 `idg` 再加 `m_last`
项、参考是先 `+m_last` 再 `+idg`，`db` 折进的是 `Σ_w(dbrow+idbrow)`、参考是 `Σ_w dbrow + Σ_w idbrow`，
两者仅**结合序**不同 → ulp 级（≤9.5e-7）。**未被 intra 触及的输出全部逐位不变**（`dv2/dvb/dkgb/dA/
dApre/dw/dgk` max_abs=0）。且 `(prod_intra − prod).dqg` 非零（如 `B8` 的 max_abs=1.45），证贡献确实加进去了。

### 66.3 Part B：资源 —— 寄存器反降、smem +26 KB、仍零 spill / 1 block/SM

| build | regs | spill | smem (B) |
|---|---|---|---|
| `prod` | 146 | 0 | 133760 |
| `prod_intra` | **144** | **0** | **159872** |

prereg 未涨（编译器把 intra 片元复用进 gate 的空档），smem +26112 B（≈25.5 KB）——`prod` 本就在
1 block/SM，新增 smem 后仍 1 block/SM（ncu `launch__occupancy_limit_shared_mem` 两者皆 1）。

### 66.4 Part C：计时 —— 大 shape 略慢（1.04–1.06×），小 shape 净赢（0.88–0.89×）

`check_bwd_b2_intra.out.txt`（CUPTI `dev`，均值；`prod+b3` = 老设计两次 launch）：

| case | `prod` (µs) | `prod_intra` (µs) | `prod+b3` (µs) | `incl/prod` | `fold/(p+b3)` |
|---|---|---|---|---|---|
| B8 T2048 H8 | 1306.2 | 1636.7 | 1538.1 | 1.253 | **1.064** |
| B2 T1024 H8 | 171.7 | 214.5 | 205.6 | 1.249 | **1.043** |
| T512 H4 | 20.5 | 24.5 | 27.9 | 1.198 | **0.878** |
| T256 H2 | 18.7 | 22.2 | 24.9 | 1.186 | **0.891** |
| T250 H3（非整除） | 18.9 | 22.5 | 25.3 | 1.189 | **0.889** |
| T64 H1（单 chunk） | 18.1 | 21.5 | 24.1 | 1.187 | **0.894** |

折进后**总是**比 `prod` 慢 ~19–25%（`incl/prod`）——intra 段的活是**串行追加在每个 CTA 的关键路径尾部**
（staging + 4 mma + 门控），不像独立 B3 那样是一批只做 intra 的 CTA。**小 shape** 省下一次 launch 摊薄
（固定开销占比高）→ 净赢 **~11%**；**大 shape** launch 已摊薄，串行尾巴占上风 → 略慢 **4–6%**。

### 66.5 Part D：`ncu` —— 占用率 / tensor% 不动，代价是 L2 流量 + bank conflict + 串行尾巴

`probe_b2_intra_ncu.out.txt`（`b8t2048h8` 与 `b2t1024h8`；两者都非 TMA、`-DB2M_PROD`）：

| 指标 | `prod` B8 | `prod_intra` B8 | `prod` B2 | `prod_intra` B2 |
|---|---|---|---|---|
| Duration | 1.30 ms | **1.64 ms** | 172.2 µs | **216.2 µs** |
| DRAM read / write | 1.63 GB / 486 MB | 1.64 GB / 486 MB | 203 MB / 54 MB | 205 MB / 54 MB |
| L2 bytes | 4.29 GB | **4.36 GB** | 536 MB | **542 MB** |
| global ld sectors | 77.7 M | **84.3 M** | 9.72 M | **10.53 M** |
| smem bank conflicts | 23.5 M | **36.3 M** | 2.94 M | **4.54 M** |
| tensor pipe | 3.3% | 3.0% | 3.0% | 2.8% |
| warps active | 12.46% | 12.46% | 12.46% | 12.46% |
| registers/thread | 146 | 144 | 146 | 144 |

读法：**DRAM 字节几乎不动**（`q/k/g` 本就被 gate 读过、命中 L2；仅 `dAqk` ~1 MB 新读），
**占用率与 tensor% 完全不变**——代价是（a）`siq/sik/sig/sda` 的 smem 读写把 **bank conflict 抬 ~55%**、
global ld sector +6.6 M（`q/k/g` 合流读）、L2 +70 MB；（b）一段**串行**的 staging+mma+门控尾。
所以 `Duration +26%` 是**串行尾巴 + smem 流量**，不是新瓶颈（kernel 本就在 1 block/SM、
12.5% 占用、tensor 3% 的标量/发射受限区）。

### 66.6 结论与设计输入

- **兑现 §65.6**：B3 对角 intra 已折进**单次** `-DB2M_PROD -DB2M_INTRA` launch，`dq/dk` **逐位**、
  `dg/db` ulp 级、未触及输出**逐位不变**；`dA` 不再落盘、B2/B3 不再是两个 kernel。
- **但不是大 shape 的净赢**：折进 vs「prod + 独立 B3」= 大 shape **1.04–1.06×（略慢）**、小 shape
  **0.88–0.89×（净赢）**。原因：部署粒度下 B3 的绝对工作量太小，独立 launch 会被 executor 更好地
  并行/流水掉，而**折进把它变成每个 CTA 关键路径尾部的一段串行尾巴**（并多扫一遍 `q/k/g` 的 smem staging）。
- **设计输入（更新）**：目标卡**保持 B2 与 B3 为两个 kernel**，除非目标是**小 batch/短序列**的 launch
  受限场景（此时折进净赢 ~11%）。这与 P7-41（CUTLASS 空杠杆）、P7-58（`q/k/g` 合流净亏）同属一类：
  **「结构闭合」与「性能净赢」不是同一回事**，须分别计量。
- **边界**：只测 `K=V=128`、`BT=16`、fp32 操作数；本折进只覆盖部署粒度 `NC=1` 的对角（非对角 Parts A/C
  只在 `BT>BC` 时出现，本专题未覆盖）。

### 66.7 坑与记录（诚实）

- **merge 的对齐前提**：intra 累加器片元（行 `g/g+8`、列 `warp*BW+nt*8+2t+(i&1)`）必须与 gate 的最终
  `dqgv/dkgv/dgv` **逐位同构**才可寄存器相加；P7-62 的独立核正是因为复用同一片元布局才能无缝折进。
- **`dAkk` 掩码别漏**：`daB` 存的是**未掩码** `X2`，`SAKK` 必须自带 `(a>j) & valid`，否则 `Mk/Sk`
  把上三角也吃进去（指纹：`dq` 对、`dk/dg` 错到 O(1)）。
- **顺序与 ulp**：把 intra 的 `dg` 折在 gate 的 `m_last` 项**之前**、`db` 折成 `Σ_w(dbrow+idbrow)`，
  只改结合序 → 实测 ≤9.5e-7；若要逐位，须把 `idg`/`idbrow` 与参考**同序**相加（可做但价值低）。
- **段重排**：为让 intra 读到 SAND 的 `daB` 并把片元 merge 进 GATE，DA+SAND 必须前移到 GATE/DB 之前；
  段间 buffer 不相交，故重排**不改任何段内算术**（Part A 的「未触及输出逐位=0」即证）。

## 67. 把 K2 形状 CUDA B2 **生产核**落进完整反向链（P7-64）：端到端正确、`dv`/`dh0` 与最优链**逐位**；但**整核比融合 Triton B2/B3 慢 ~1.6×**

P7-48…P7-63 把目标卡的 CUDA B2 一段一段做成了 K2 形状、再合成**单次生产 launch**（`-DB2M_PROD`，
P7-60；去掉 `dw` 往返 `-DB2M_HDW`，P7-61；折进 B3 对角 `-DB2M_INTRA`，P7-63）。但这些轮次的验证全在
**段与段之间**或**合成操作数**上（`check_bwd_b2_prod.py` / `check_bwd_b2_intra.py` 喂
`dqkg_make_inputs` 的合成数据）；生产核**从未落进端到端反向链**——链里的 B2/B3 一直是 Triton
（`check_bwd_cudab1_fused_e2e.py`，§45）。本轮把它接上：组装

    dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod[ -> Triton B3 intra] -> reverse cumsum(B0)

只吃 FlashKDA 的 16-chunk K1 workspace，对拍 `torch.autograd`(FLA) 与 §45 最优链 `cf`。新增
`harness/check_bwd_prod_e2e.py`（`harness/check_bwd_prod_e2e.out.txt`，空闲卡）。两个 CUDA-B2 变体：

- **`pc`** ＝ `-DB2M_PROD -DB2M_HDW` + **独立** Triton B3 `intra`（对应 §66.6「保持 B2/B3 两个 kernel」）；
- **`pci`** ＝ `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`，**单次** CUDA launch 产出全部六项（`dA` 不落盘的合体形）。

### 67.1 数据流：三处必须对齐的接口

1. **操作数**：CUDA prod 是 **fp32/tf32** 核（无 bf16 变体），故链里把 B1 的 bf16 `dh`/`dv2` 之一转
   fp32 喂入（`dh.float()`；`dv2.float()` 本就是 Triton 链的口径），其余 `q/k/g/beta/A/h/v/v_new`
   都取 workspace fp32 字段（`prep`/`reconstruct` 的原生 fp32）。
2. **`dw`**：`-DB2M_HDW` 下 DKGB 段读核内 `accw`，`dw` 入参**被忽略**——链里传一个零张量即可（§63
   Part A2 已证喂零逐位不变）。
3. **`dv` 终值**：CUDA prod 的 `dv2` 输出（`beta·(A@dv)`）与 Triton `wy_dqkg` 的 `dv` 是同一量，链里
   直接作为最终 `dv`（与 `cf` 一致）。

### 67.2 Part A：正确性——六项梯度全落 bf16/tf32 底噪，`dv`/`dh0` 与 `cf` **逐位相同**

6 组 shape（含非整除 `T=250`/`B2 T=513`、`g=0`、`B=2`）。跨组最坏 `max_rel`：

| 字段 | `pc` vs FLA | `pci` vs FLA | `cf` vs FLA | `pc` vs `cf` | `pci` vs `cf` |
|---|---|---|---|---|---|
| `dq`    | 3.07e-3 | 3.06e-3 | 3.42e-3 | 1.43e-3 | 1.70e-3 |
| `dk`    | 3.42e-3 | 3.41e-3 | 3.30e-3 | 1.10e-3 | 1.56e-3 |
| `dv`    | 6.80e-3 | 6.80e-3 | 6.80e-3 | **0** | **0** |
| `dbeta` | 6.01e-3 | 6.01e-3 | 6.01e-3 | 8.4e-5 | 6.8e-5 |
| `dg`    | 7.86e-3 | 7.76e-3 | 7.05e-3 | 6.1e-3 | 9.8e-3 |
| `dh0`   | 9.37e-3 | 9.37e-3 | 9.37e-3 | **0** | **0** |

- 六项对 FLA 的误差与 §25/§26/§42/§45 的 **`cf` 链逐值同阶**（全是 bf16 操作数 + tf32 mma 的底噪）。
- `dv`/`dh0` 与 `cf` **逐位相同（`max_abs=0`）**——两者都由 B1（CUDA，同一份 `dh`/`dv2`）与输出支路定，
  CUDA B2 只做等价的 `beta·A@dv`；这条位级签名再次证实接口等价。
- `dq/dk/dbeta` vs `cf` 只差 tf32 累加分组（~1e-3），`dg` 略大（6–9.8e-3）但 `dg` 的 `ref_max` 小
  （~6e-2，`g` 只在 `2^{g}` 指数与小量相减里定），相对误差被小分母放大、绝对量仍 ~1e-4 级。

**结论**：CUDA B2 生产核（含 `-DB2M_INTRA` 合体）是**忠实的 drop-in**——端到端六项梯度与最优链同底噪。

### 67.3 Part B：计时——CUDA B2 生产核**不是**净赢，`pci` 比融合 Triton `cf` 慢 ~1.6–1.7×

反向链 dev（CUPTI，µs；`wall` 为 event-pair）：

| case | FLA dev | `cf` dev | `pc` dev | `pci` dev | `pci/FLA` | `pc/FLA` | `pci/cf` |
|---|---|---|---|---|---|---|---|
| T512 H4 | 203.4 | **74.6** | 116.3 | 94.9 | 0.47 | 0.57 | 1.27 |
| B2 T1024 H8 | 661.7 | **257.2** | 462.6 | 387.4 | 0.59 | 0.70 | 1.51 |
| B8 T2048 H8 | 4124.6 | **1404.4** | 2908.4 | 2422.8 | 0.59 | 0.71 | 1.73 |
| T250 H3 | 141.8 | **50.2** | 83.1 | 64.6 | 0.46 | 0.59 | 1.29 |

逐 kernel（`B8 T2048 H8`，dev/call µs）：

| 链 | B1 | B2/B3 kernel |
|---|---|---|
| `cf`（CUDA-B1 + 融合 Triton） | 182.7 | **`_fused_b2b3_kernel` 998** |
| `pc`（CUDA-B1 + `prod_hdw` + Triton `intra`） | 182 | `prod` **1265.8**（+ 独立 `intra` 与 3 次链外加法） |
| `pci`（CUDA-B1 + `prod_hdw_intra`，单 launch） | 182.7 | `prod_intra` **1606.9** |

读法（诚实归因）：

- **CUDA `prod` 单核（六段，无 intra）= 1265.8 µs**，与 §63 的 `prod` = 1288 µs 互证；折进 intra 后
  **1606.9 µs**（与 §66.3 的 `prod_intra` = 1636 µs 互证，+26%）。
- 但 Triton 融合 B2/B3 **只有 998 µs**——**CUDA 六段生产核本身比 Triton 融合核慢 ~1.6×**。这不是
  输出体量的差异：`prod` 相对融合核多落盘的诊断量（`dgk/dkgb/dvb/dApre/dA`，`dw` 已由 HDW 消掉）
  在 `B8` 上合计仅 **~155 MB ≈ 65 µs（~4%）**，不足以解释 600 µs 的差。
- 差落在**非 GEMM 段的累积**：§57 的 `dA` sandwich（+33–61%）、§60 的门控（+30–38%）每一段相对
  「GEMM 核心」都只贵一两成，但六段叠在**同一个 1 block/SM、tensor ~3%、标量/发射受限**的核里
  （§63.5），累积后整核就落后于 **Triton 把它们融进一个 `tl.dot` 图 + 寄存器常驻**的形态
  （§46/§47：融合核的算子级浪费被 Triton 的 tiling/流水吃掉）。
- `pci`（单 launch）比 `pc`（`prod` + 独立 Triton `intra`）快 ~1.2×——与 §66「折进在小/中 shape 净赢、
  大 shape 靠省一次 launch 摊薄」方向一致（此处 `pc` 用的是 **Triton** intra，非 §66 的 CUDA `b3intra`）。

### 67.4 结论与设计输入

- **兑现「K2 形状 CUDA B2 落链」**：生产核（`-DB2M_PROD [-HDW] [-INTRA]`）第一次端到端可跑，六项梯度
  对 `torch.autograd` 落 bf16/tf32 底噪、`dv`/`dh0` 与 §45 最优链**逐位相同**。**接口忠实、代数无误**。
- **但整核不是净赢**：把六个非 GEMM 段单核化后，CUDA B2 比 **Triton 融合 B2/B3 慢 ~1.6–1.7×**
  （`pci/cf` 1.27–1.73×）。这与 §15/§17/§20/§41/§58/§63/§66 反复出现的主题一致——
  **「结构闭合 / 单核化」≠「性能净赢」**：K2 形状的 GEMM 核本身能赢同 op Triton（§49 0.93×），
  但把**全部**非 GEMM 段也搬进同一个受限核里后，反而追不上 Triton 把整条 B2/B3 融进一张
  `tl.dot` 图 + 寄存器常驻的形态。
- **设计输入（更新 §66.6）**：目标卡目前的最优反向形态仍是 §45 的
  **`dAv → B1 → 融合 Triton B2/B3 → 反向 cumsum`**（`cf`），CUDA 只在 **B1**（§42：大 shape 1.73×）
  净赢；CUDA B2 生产核当前是**正确但更慢**的研究产物。若要在目标卡上用 CUDA B2，下一杠杆不是再加段，
  而是把六段**重新融合回寄存器**（`dA` sandwich 与门控的中间量不落盘，逐 §64 的「同 CTA 交接」判据），
  即把 Triton 融合核的结构优点搬进 CUDA——这与 §49.7/§63.6 的收尾方向一致，但**收益上界已被本轮
  端到端计时钉在上方 ~1.6×**（先证「能追平」，再谈「超过」）。
- **边界**：只测 `K=V=128`、`BT=16`、fp32 操作数、非 TMA 路径（`B2M_TMA` 与 `B2M_INTRA` 未打通，
  `b2_mma_probe.cu:1809-1820` 的 TMA 分支未传 `dAqk_in`）；CUDA prod 仍落盘五类诊断中间量
  （真实生产应剪掉，本轮为与隔离 build 对齐保留）。`pc` 链的独立 Triton `intra` 计时受链外 3 次加法
  干扰，未单独拆净。

## 68. 把 §67 的「六段非 GEMM 成本在一个 1 block/SM 核里累积」**逐段归因**：生产核慢 ~1.6× 的真凶是三段**交付物 IO + 标量数学**（OUT / DA / GATE），不是可再融合掉的中间量（P7-65）

§67.4 把生产核（`-DB2M_PROD [-HDW]`）比 Triton 融合 B2/B3 慢的 ~1.6× 归给「六个非 GEMM 段在同一个
1 block/SM、tensor ~3% 的核里累积」，并留下「把六段**重新融合回寄存器**」作为下一杠杆——但这个累积
**从未被拆开**：P7-52…P7-57 的每段成本都是 `core + 一段` 在**各自独立**的 kernel/launch 里量的，生产
形态（六段同核）里的边际成本不可见。本轮把它量出来。

### 68.1 方法：生产核内的**累积消融**（`-DB2M_PROD -DB2M_HDW -DB2M_LEVEL=k`）

给 `b2_mma_probe.cu` 加一个 `B2M_LEVEL∈0..6`（默认 6）：它**保留所有 flag / launch 签名 / staging
路径**，只按**逆依赖序**删掉段的**函数体**（加法序 `1=OUT 2=DKGB 3=DA 4=SAND 5=DB 6=GATE(+INTRA)`，
`b2_mma_probe.cu:150-176`）。这与 P7-52…P7-57 的「独立 launcher」根本不同：那些 build 会连 kernel
签名一起换掉，故无法共存于一个生产核里做增量扫描。`B2M_LEVEL=6` 即 P7-60 的生产核；`B2M_LEVEL=0`
是**带全套 staging 的裸 GEMM 核**。于是 `T(k) − T(k−1)` 就是该段**叠加在生产核上**的真实代价——

```c
#define B2M_OFF_OUT   (B2M_LEVEL < 1)    // ... 六个 chained，INTRA 随 GATE 一起
```

> 代价：level < 5 时 `acc2/acc3` 按设计不被写出（段body被删），故此 build **仅供计时**；存活输出的
> 正确性在 Part A 单独锚定。

新增 `harness/check_bwd_b2_prod_levels.py`（`harness/check_bwd_b2_prod_levels.out.txt`，空闲卡）+
`harness/probe_b2_levels_ncu.sh`（`harness/probe_b2_levels_ncu.out.txt`）。计时口径 `dev`（CUPTI，
链式 `profile_dev`），四组 shape（含非整除 `T=250`）。全部 build 用部署形态 `-DB2M_PROD -DB2M_HDW`
（§64 的 `prod_hdw`）。

### 68.2 Part A：消融是**纯 compute-only**——各 level 的存活输出与 level 6 **逐位相同**

| 比较 | 结果 |
|---|---|
| `L1.dv2/dvb` vs `L6`（OUT 唯一生产者，下游无关） | `max_abs=0` |
| `L2.dkgb` vs `L6`；`L3.dApre`；`L4.dA`；`L5.db` vs `L6` | `max_abs=0` |
| `L0.dgk` vs `L6`（GEMM 核在所有 level 相同） | `max_abs=0` |

四组 shape（`B8T2048H8 / B2T1024H8 / T512H4 / T250H3`）**全部逐位**——删段只删 compute/store，
从不改变任何存活输出，故各 level 的**时间差**可干净归因到被删段。

### 68.3 Part B/C：逐段边际成本曲线（`dev` µs，`B8 T2048 H8`）

| L | 新增段 | `dev` | `Δ` | `Δ/core` | regs | spill |
|---|---|---|---|---|---|---|
| 0 | —（裸核心） | 610.5 | — | — | 31 | 0 |
| 1 | **OUT** | 761.6 | **151.1** | **24.7%** | 32 | 0 |
| 2 | DKGB | 840.8 | 79.2 | 13.0% | 48 | 0 |
| 3 | **DA** | 1024.7 | **183.8** | **30.1%** | 62 | 0 |
| 4 | SAND | 1021.6 | −3.0 | −0.5% | 64 | 36 |
| 5 | DB | 1057.8 | 36.2 | 5.9% | 79 | 0 |
| 6 | **GATE** | 1262.1 | **204.3** | **33.5%** | 112 | 0 |

- **总非 GEMM 开销 = 651.6 µs**（610.5→1262.1，与 §67 的 `prod` 1265.8 互证）。**三段吃掉 83%**：
  `OUT 151 + DA 184 + GATE 204 = 539 µs`（核的 43%）。
- **曲线≈加性**：每段的 prod-regime 边际与它在 P7-52…P7-57 的**独立**数字**同阶**——`OUT +25%`
  （§55 报 +12–25%）、`DA +30%`（§57 报 +33–61%）、`GATE +34%`（§60 报 +30–38%）、`DKGB +13%`
  （§56 报 +9–13%）、`DB +6%`（§59 报 +2–13%）、`SAND ~0`（§58 报 +2–6%）。**故 P7-60 的「一次
  launch 复用 GEMM 核」是加性摊薄，不是超加性的资源争用**——这修正 §67.4 把差整体归给「累积」的措辞。
- 小 shape 同结构（`B2T1024H8`：OUT 16.8/DA 23.2/GATE 26.2 占核心 20/27/31%；`T512H4` 同），只是
  DA 占比略升、SAND 恒 ~0。

### 68.4 `ncu`：三段的机制——**交付物 IO（OUT/GATE）+ 标量 smem 数学（DA）**

逐 level 的**边际** `ncu` 字节（`B8T2048H8`，`probe_b2_levels_ncu.out.txt`；`Duration` 与 CUPTI `dev`
本轮**逐值吻合**，610.2/759.6/837.2/1020/1020/1060/1270 µs，与 §63.5/§64.5 的「ncu 窗口不同源」相反）：

| 段 | Δ DRAM write | Δ DRAM read | Δ L2 bytes | tensor% | bank conflict | 读法 |
|---|---|---|---|---|---|---|
| 核(L0) | 6.7 MB | 1.49 GB | 2.92 GB | 0.00 | 8.39 M | 纯 GEMM，DRAM/L2 bound（73/78% SOL），occ 12.5% |
| OUT | **+130.8 MB** | 0 | +0.24 GB | 0.22 | 8.91 M | 写 `dv2+dvb` 两个 `[B,T,H,V]`（=交付物） |
| DKGB | +65.4 MB | 0 | +0.15 GB | 1.96 | 9.44 M | 写 `dkgb`（交付物）+ 其 mma 抬 tensor |
| DA | +8.1 MB | 0 | +0.02 GB | **1.61** | **20.84 M（×2.2）** | 写 `dApre` 极小，但**标量 FMA + shared 冲突翻倍**、tensor 不动 |
| SAND | +16.1 MB | 0 | +0.19 GB | 1.61 | 20.84 M | 写 `dA`（两 `[BT,BT]` matmul，但按 chunk 划分成本极低） |
| DB | ~0 | 0 | ~0 | 1.56 | 23.54 M | `db` 落盘仅 0.5 MB，纯跨 warp 归约 |
| GATE | **+198.2 MB** | **+70 MB** | +0.60 GB | 3.36 | 23.46 M | 写 `dq/dk/dg` 三个 `[B,T,H,K]`（=交付物）+ 散读 `q/k/g` + `exp2f` |

- **OUT 和 GATE 的成本 = 它们的交付物字节**：`OUT +130.8 MB`（= `dv2`+`dvb`，`2×67 MB`，逐一对上
  `131 MB`）、`GATE +198.2 MB`（= `dqg+dkg+dg`，`3×67 MB`）。这些是**最终梯度输出**，任何融合都删不掉。
- **DA 的成本 = 标量数学**：它只写 `8 MB`，却把 bank-conflict 从 `9.4M` 抬到 `20.8M`（×2.2）、tensor
  **纹丝不动（1.61%）**——正是 §57 记录的「收缩维 = 全维 `V`/`K`（各 128，8× MAC）+ 标量 FMA，shared
  wavefronts +82%」。**DA 是唯一仍用标量实现、非 `mma.sync` 的非 GEMM 段**（OUT/DKGB 早已是 mma）。
- **SAND/DB 是真·中间量段，但它们本来就几乎免费**（−3 / +36 µs），且 DB 的两个操作数 `dvb`/`dkgb`
  在 §56 已证是**核内寄存器归约**、`dA` 在 §55 已证由 SAND 就地吃 `dApre`——**这些才是「可再融合」的
  项，而它们合计 ≤ 40 µs**。

### 68.5 结论与设计输入（修正 §67.4）

- **兑现 §67 的逐段归因**：生产核的 `651 µs` 非 GEMM 开销拆成 `OUT 151 / DKGB 79 / DA 184 / SAND ~0 /
  DB 36 / GATE 204`；曲线**加性**，与各段的独立测量**同阶**——生产核慢不是超加性资源争用，就是六段
  真实成本之和。
- **§67.4 的「把六段重新融合回寄存器」杠杆 = 几乎无头寸**：能靠「消中间量落盘」省下的只有 SAND(~0)
  与 DB(~36 µs)，且它们的中间量（`dApre`、`dvb`/`dkgb`）**早已是核内**（§55/§56/§64）。三段的钱花在
  **交付物 IO**（OUT+GATE 共 +329 MB 最终梯度写）与 **DA 的标量数学**上，**重融合删不掉**。与 §67.3
  的「诊断中间量 ~155 MB ≈ 65 µs ≈ 4%」自洽——那 4% 就是重融合的**全部**上界，远小于 §67.4 暗示的 1.6×。
- **真正的逐段杠杆只有一条：把 DA 从标量 FMA 改成 `mma.sync`**（§57.6 早已写下的设计输入）。DA 的
  `184 µs`（核心的 30%）里，tensor 全程 1.6%、bank-conflict ×2.2——它是「收缩维 = 全维 `V`/`K`」的段，
  可像三条主 GEMM 那样用 `X@Y^T` 的 `mma.sync` + warp 分片归约，把标量 shared 流量换成 tensor 流量。
  这是生产核追上 Triton 融合核（§67 的 998 µs vs 1262 µs，差 264 µs）**唯一有量级希望的单项**：DA 的
  184 µs 只要压到 OUT 那样的 mma 成本量级（~几十 µs）即可补掉缺口的大半。
- **其余两段无免费杠杆**：OUT 已是 mma，成本=交付物写；GATE 的成本=交付物写 + 散读 + `exp2f`，而
  §61 已证「合流 `q/k/g`」是净亏——故这两段只能靠**减少交付物字节**（不现实）或**换 FLA 契约**（如
  `dq/dk/dg` 就地累积进上游缓冲，属链路级而非本核）。
- **设计输入（更新 §67.4）**：目标卡若仍要 CUDA B2 生产核，下一且几乎是唯一的动作是
  **`mma.sync` 化 DA**；「重新融合回寄存器」在本轮被量化到 ~4%、**不做**。结合 §67 的结论，目标卡的
  默认反向仍推荐 §45 的 `dAv→B1→融合 Triton B2/B3→cumsum`；CUDA B2 生产核要翻盘，须先证 `mma` 化的 DA
  能把 1262 µs 压进 Triton 融合核的 998 µs 附近。

### 68.6 坑与记录（诚实）

- `B2M_LEVEL` 只删**函数体**、保留 flag/签名/staging——若换成「按 flag 关段」会连带改 kernel 签名，
  使 P7-60 的固定签名生产 launcher 编译不过（arg 数不匹配）；这是选择 body-gating 的原因
  （`b2_mma_probe.cu` 的两处 `b2prod_launch_impl` kernel 调用是全参数列表，不随 flag 收缩）。
- level 4 (`SAND`) 的 `Δ` 在两组 shape 上都是 **≈0 甚至负**——不是噪声：SAND 的两次 `[BT,BT]@[BT,BT]`
  在 `BT=16` 下总量极小，且它把 DA 留下的 shared 冲突窗口「填满」（level 3→4 的 bank conflict 不变、
  时间不动），与 §58「sandwich 1.02–1.10×」互证。
- 本轮的 `ncu Duration` 与 CUPTI `dev` **逐值吻合**（7/7 档），而 §63.5/§64.5 记录过二者反向——差别在
  本轮每次只 profile **一个** `b2mma_kernel`（driver 直接合成操作数、无 FLA/Triton setup），窗口同源。
- 边界：只测 `K=V=128`、`BT=16`、fp32、非 TMA（`-DB2M_HDW` 部署形态）；`B2M_LEVEL` 对
  `-DB2M_TMA`/`-DB2M_INTRA` 未交叉验证（`INTRA` 随 `GATE` 一起、TMA 路径的 body-gating 未测）。

## 69. `mma.sync` 化 DA：把 §68 点名的「唯一量级杠杆」兑现——DA **段内**边际省 2–3×，但**生产核整体只 +5–7%**（P7-66）

§68.4 把 DA（`dApre = dv @ v^T - dw @ kg^T`）的 `184 us`（裸核心的 30%、六段里最大）钉在**标量
shared 数学**上：它把 smem bank conflict 从 `9.4M` 抬到 `20.8M`（×2.2）而 `tensor%` 纹丝不动
（1.6%）；§68.5 于是把「把 DA 从标量 FMA 改成 `mma.sync`」列为生产核追上 Triton 融合核**唯一有
量级希望的单项**。本轮把它做出来并**定量核验这个希望兑现了多少**。

### 69.1 方法：`-DB2M_DAMMA`（需 `-DB2M_DA`），算术不变、只换收缩的实现

`dApre` 的两项都是 `X @ Y^T`（与 GEMM 核的 `dq = do @ h^T` 同形，`b2_mma_probe.cu:874`）：
- base：`dApre[o_t,o_u] = Σ_{v∈warp V-slice} dv[o_t,v]·v[o_u,v]`
- dwkg：`dApre[o_t,o_u]-= Σ_{k∈warp k-slice} dw[o_t,k]·kg[o_u,k]`

P7-54 用标量：base 每 warp `BW×8 = 128` 次 shared 读、dwkg 再 `128` 次。P7-66 改成每 warp 用
`mma.sync.m16n8k8.tf32` 算**整块** `[BT,BT]` partial（2 个 n-tile × 每项 2 个 k-step，因 warp 只需
收缩自己的 `BW=16` 宽切片），8 个 warp 的 partial 仍过 `dapart`→`da16` 归约——
`b2_mma_probe.cu:985-1007`（base mma）、`:1177-1202`（dwkg mma + 按 mma C 布局写 `dapart`）。
A 操作数来自已 staged 的 `dv`/`swall`，B 操作数来自 `vv`/`skg`，与 GEMM 核读法逐字同构（tf32）。

### 69.2 Part A（§隔离）：代数是 mma 的、落到 tf32 底噪，且**跨布局逐位**

`harness/check_bwd_b2_da_mma.py`（`harness/check_bwd_b2_da_mma.out.txt`，空闲卡）。四组 shape
（含非整除 `T=250`），`dw` 取 CUDA GEMM 核的输出（本身 tf32）：

| 比较 | 最坏 `max_rel` |
|---|---|
| `da`（标量 P7-54）`dApre` vs torch(fp32) | **6.36e-7**（fp32 底噪，同 §57） |
| `da_mma` `dApre` vs torch(fp32) | **3.42e-4**（=tf32 底噪，同 OUT/DKGB 3.6e-4） |
| `da_mma` vs `da`（标量） | 3.42e-4（就只差 tf32 舍入） |
| `da_mma` vs `da_mma_tma` / `_tma64` | **0**（逐位，含非整除 T） |

即：**只把收缩换成 mma**，数值从 fp32 预算底噪（5e-7）抬到与其余 mma 段一致的 tf32 底噪
（3.4e-4）；三种 stage 布局（非 TMA / TMA@32 / TMA@64）逐位相同——§54 的布局解码在新算术上再验一遍。

### 69.3 Part B（§隔离）：孤立 `da_mma/da` = 0.87–0.92×，配 TMA 后 0.82–0.94×

`dev`（CUPTI，空闲卡），四组 shape：

| case | core | core_tma | da | da_tma | **da_mma** | dm_tma | dm_tma64 | fullB2 | mma/sc | mma_tma/sc_tma |
|---|---|---|---|---|---|---|---|---|---|---|
| B8 T2048 H8 | 518.6 | 492.7 | 958.0 | 656.2 | **866.5** | 610.7 | 852.7 | 903.3 | **0.90×** | 0.93× |
| B2 T1024 H8 | 70.9 | 68.5 | 128.0 | 97.7 | **117.2** | 91.6 | 115.4 | 121.5 | 0.92× | 0.94× |
| T512 H4 | 9.3 | 8.9 | 13.9 | 13.9 | **12.0** | 11.8 | 11.6 | 14.3 | 0.87× | 0.85× |
| T250 H3 非整除 | 8.6 | 8.4 | 12.9 | 13.6 | **11.2** | 11.2 | 10.6 | 14.1 | 0.87× | 0.82× |

孤立里 mma DA 只比标量 DA 快 **8–13%**（`da-core` 439→348 µs @B8）。

### 69.4 Part C（**生产核**，关键）：

`-DB2M_PROD -DB2M_HDW`（P7-60/61 部署形态）标量 DA vs mma DA：

| case | prod(sc) | prod(mma) | `mma/sc` | `dApre` rel | `dA` rel | 非 DA 输出 |
|---|---|---|---|---|---|---|
| B8 T2048 H8 | 1279.5 | **1213.5** | **0.95×** | 3.23e-4 | 3.06e-4 | **逐位相同** |
| B2 T1024 H8 | 169.4 | **161.7** | 0.95× | 2.94e-4 | 2.80e-4 | 逐位 |
| T512 H4 | 19.3 | **18.0** | 0.93× | 3.40e-4 | 3.15e-4 | 逐位 |
| T250 H3 非整除 | 18.8 | **17.4** | 0.93× | 2.65e-4 | 2.66e-4 | 逐位 |

**用 §68 的 L2→L3 累积消融测生产态 DA 的边际成本**（`L3=OUT+DKGB+DA` 减 `L2=OUT+DKGB`）：

| case | sc L3−L2（DA 标量） | mma L3−L2（DA mma） | **DA mma/sc** |
|---|---|---|---|
| B8 T2048 H8 | **185.4 µs** | **83.3 µs** | **0.45×** |
| B2 T1024 H8 | 23.6 | 11.4 | 0.48× |
| T512 H4 | 3.1 | 1.2 | 0.38× |
| T250 H3 非整除 | 3.3 | 1.1 | 0.32× |

**这是本轮的主结论**：`mma` 化把 DA 的**段内边际成本砍掉 2–3×**（`185→83 µs`，§68.5 的杠杆**方向
正确且兑现**）；但**整核只快 5–7%**（`1279→1214`），因为 DA 的 185 µs 只是 1279 µs 核的 ~14%。

### 69.5 `ncu`（`B8T2048H8`，`harness/probe_b2_damma_ncu.sh` / `probe_b2_damma_ncu.out.txt`）

| 指标 | 标量 DA | mma DA |
|---|---|---|
| `Duration` | 1.27 ms | **1.21 ms** |
| bank conflict | 23.46 M | **17.30 M**（−26%） |
| `tensor%` | 3.37 | **3.81** |
| L2 bytes | 3.92 GB | 3.82 GB |
| DRAM read / write | 1.56 GB / 419 MB | 1.56 GB / 419 MB（**不变**） |
| regs / occ | 112 / 12.46% | 104 / 12.46% |

DA 的 bank conflict 贡献从 `~15M`（23.46−8.39=核心基线）降到 `~9M`——**没有归零**，因为 tf32 mma
的操作数是 32-bit、**没有 16-bit `ldmatrix` 可走**（§49.6 的老限制），A/B 片段仍得逐元素从
row-major smem 直读（`rd`），所以 mma 只把冲突削掉 ~40%，不是 §68.4 设想的「换成 tensor 流量就干净」。

### 69.6 结论与设计输入（修正 §68.5）

- **§68.5 的「唯一量级杠杆」在段内成立、在整核不成立**：DA 段内 `185→83 µs`（0.45×），但生产核
  只 `0.93–0.95×`。原因是 §68.3 的加法曲线——DA 只占 651 µs 非 GEMM 开销的 ~28%，其余 `OUT+GATE`
  的 `329 MB` 交付物 IO（不可删）+ `DKGB` 才是大头。**mma 化 DA 不能把生产核从 1279 µs 拉进 Triton
  融合核的 998 µs**：`1214/998 = 1.22×`，仍在 1.2× 量级。**§67/§68 的默认建议不变**：目标卡反向默认
  `dAv→B1→融合 Triton B2/B3→cumsum`；CUDA B2 生产核是**正确**但**更慢**的研究产物。
- **DA 不是「几十 µs」**：mma 后仍 `83 µs`（B8），因为 tf32 片段读无 `ldmatrix`、bank conflict 只
  削 ~40%。要再往下，须给 `dv/vv/swall/skg` 做 smem swizzle 布局（本核其它段同样卡在这一点）——
  属结构性而非「换一条指令」。
- **正确性零代价**：非 DA 的七个输出（`dv2/dvb/dkgb/dqg/dkg/dg/db`）**逐位不变**，只 `dApre/dA` 落到
  tf32 底噪——因为它只换 DA 的收缩实现、不碰任何共享操作数/累加器。故 `-DB2M_DAMMA` 可作为**默认**
  部署开关（免费、正确），只是收益有限（5–7%）。
- **设计输入（更新 §68.5）**：目标卡若要 CUDA B2，`mma` 化 DA 值得做（免费 + 段内 2–3×），但**它不是
  翻盘点**；真正的大头是 OUT/GATE 的**交付物写**（只能靠链路级「就地累积进上游缓冲」减，见 §31/§62）
  与 `dk/dg` 的门控标定。**不推荐**再为 DA 单独投入（收益上界已钉在整核 ~5–7%）。

### 69.7 坑与记录（诚实）

- **`B2M_GATE` 下 raw `dq`/`dk` 缓冲是未初始化的**（生产形态只存门控后的 `dqg`/`dkg`）——首版 Part C
  拿 `dq`/`dk` 比，读出 `max_abs` O(1)（`dq` 标量侧恰为 0.5 的垃圾、mma 侧 0），**误判成「mma 改坏了
  门控」**。正确要比的输出集是 `dv2/dvb/dkgb/dqg/dkg/dg/db`；改对后**全部逐位相同**。凡比对
  `b2_mma_probe.cu` 的门控/生产输出，**先查该 flag 下哪些缓冲真的被写**（`#if !B2M_GATE` 会关掉 raw 存）。
- **`mma` 段与孤立段都能跑 ≠ 生产核能跑**：孤立 `b2da_launch`（`-DB2M_DA -DB2M_DAMMA`）的
  `dq/dk/dw/dgk` 与标量**逐位**、`dApre` 落 tf32；但孤立构建的 `dq/dk` 与生产构建是**不同缓冲契约**，
  两条链接的「bit-exact」不能互推（本轮即因此走了一趟弯路）。
- 边界：只测 `K=V=128`、`BT=16`、fp32、非 TMA 的 `-DB2M_HDW` 部署形态；`-DB2M_DAMMA` 与
  `-DB2M_INTRA`/`-DB2M_TMA` 的整核组合只做了孤立正确性（TMA 逐位），**未做生产链端到端**（待验证）；
  未测 bf16 操作数下的 DA mma。

## 70. 关闭 §67.4 的「`B2M_TMA` 与 `B2M_INTRA` 未打通」：一次实参转发的修复，逐位一致——但 TMA 在生产核里不划算（P7-67）

§67.4（P7-64）在记录 K2 形状 CUDA B2 生产核落链时留下一条明确的边界：
「**`B2M_TMA` 与 `B2M_INTRA` 未打通（TMA 分支未传 `dAqk_in`）**」。换句话说，P7-63 折叠进来的
B3 对角 `intra`（`-DB2M_INTRA`）无法与 P7-49/§52 的 TMA 操作数布局（`-DB2M_TMA`）**同时编译**。
本节把它关掉：定位根因（一处遗漏的实参转发）、修复、并核验「换布局不改数值」、再量清楚这对
生产核到底有没有用。

### 70.1 根因：生产 launch 的 TMA 分支漏传了内核的最后一个实参

`b2_mma_probe.cu` 里 `b2prod_launch_impl` 按 `#if B2M_TMA_ON` 分成两条 `b2mma_kernel<<<...>>>`
调用路径。内核签名在 `B2M_INTRA` 下于 SAND 参数之后多一个 `const float* dAqk`
（`b2_mma_probe.cu:707-709`）：

```c++
#if B2M_INTRA
             const float* __restrict__ dAqk,
#endif
             int B, int T, int H, int NT)
```

非 TMA 的 `#else` 分支把它传了（`b2_mma_probe.cu:1943-1945`），TMA 的 `#if B2M_TMA_ON` 分支
（`b2_mma_probe.cu:1920-1931`）却在 `dAout` 之后直接跳到 `B, T, H, NT`——**漏掉了 `dAqk`**。
于是 `-DB2M_TMA -DB2M_INTRA` 根本编不过。对照验证（把修复那 3 行删掉后编译）：

```
b2_nofix.cu(1934): error: argument of type "int" is incompatible with parameter of type "const float *"
b2_nofix.cu(1934): error: too few arguments in function call
```

（首条是这个漏参数的典型连报：多出的 `int B` 落到了 `const float* dAqk` 形参上。）

修复就是让 TMA 分支与非 TMA 分支写同一段守卫（**算术一字未动**）：

```c++
        (float*)db_out, (float*)dApre, (const float*)A4s, (float*)dAout,
#if B2M_INTRA
        (const float*)dAqk_in,
#endif
        B, T, H, NT);
```

值得注意的是 `b2prod_intra_launch` 的**宿主签名是 TMA-agnostic 的**：张量图是在
`b2prod_launch_impl` **内部**用普通操作数指针现编码的（`b2_mma_probe.cu:1912-1919`），
所以本次修复**不需要动 harness 侧**任何 launcher——同一个 `run_prod_intra` 对 TMA 与非 TMA
两个 `.so` 都能用。

### 70.2 Part A：换布局（cp.async → TMA）只改搬运、不改数值——11 项输出全 **逐位**

在 `-DB2M_PROD -DB2M_INTRA` 基础上只多加 `-DB2M_TMA`，用**同一份操作数**跑
（`harness/check_bwd_b2_intra_tma.py` Part A），逐项比 11 个输出：

| case | NT | `dv2` | `dvb` | `dkgb` | `db` | `dApre` | `dA` | `dqg` | `dkg` | `dg` | `dw` | `dgk` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B8 T2048 H8 | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| B2 T1024 H8 | 64 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T512 H4 | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T256 H2 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T250 H3 非整除 | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| T64 H1 单 chunk | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**全表 `max_abs = 0.000e+00`（BIT-IDENTICAL）**，6 组 shape 含非整除 `T=250` 与单 chunk。
这与 P7-49/§52、P7-60/§63 的结论一致：TMA 只换 stage 布局（`cp.async` → `cp.async.bulk.tensor`
+ per-stage `mbarrier`），所有 GEMM/门控/sandwich 的算术与累加序不变；而 `intra` 段的 `dAqk` 在
两种布局下都从 global 直读（不进 TMA stage），故折叠进来的对角贡献也逐位不变。

### 70.3 Part B：资源——TMA 反把 smem 降下来、寄存器略降，但多 12B spill

| build | regs | spill | smem (B) |
|---|---|---|---|
| `prod_intra`（cp.async） | 144 | 0 | **159872** |
| `prod_intra_tma`（TMA） | **128** | 12 | **149120** |

TMA 版把 smem 从 159.9 KB 降到 149.1 KB（−6.7%），寄存器 144→128，但引入 12 B spill
（`.B2M_DAMMA`，§69 同源）。两者都远超「2 block/SM」的 smem 门槛（227/2 ≈ 113.5 KB），
**占用率仍被钉在 1 block/SM**——与 §63/§67 的 `prod` 形态相同。故资源侧的变化只影响 spill/指令数，
不换占用率 regime。

### 70.4 Part C：计时——TMA 在生产核里反而慢 8–12%（dev）

| case | base dev (µs) | tma dev (µs) | **tma/base** | base wall | tma wall | wall ratio |
|---|---|---|---|---|---|---|
| B8 T2048 H8 | 1642.7 | 1788.3 | **1.089** | 1624.3 | 1770.4 | 1.090 |
| B2 T1024 H8 | 214.6 | 232.3 | **1.082** | 219.1 | 236.8 | 1.081 |
| T512 H4 | 23.6 | 25.8 | **1.092** | 76.6 | 77.7 | 1.015 |
| T256 H2 | 21.6 | 24.0 | **1.107** | 77.2 | 78.4 | 1.016 |
| T250 H3 非整除 | 22.5 | 25.0 | **1.110** | 77.9 | 79.3 | 1.018 |
| T64 H1 单 chunk | 21.4 | 24.1 | **1.123** | 77.9 | 78.9 | 1.013 |

CUPTI `dev` 上 TMA **在所有 shape 一律慢 8–12%**；event `wall` 大 shape 慢 ~9%、小 shape 被
~77 µs host 税压平（~1.5%）。**交叉校验**：非 TMA 的 `prod_intra` dev `1642.7 µs` @`B8 T2048 H8`
与 §66.4 的 `1636.7 µs` 互证（同一配置），说明本节基线口径正确。

### 70.5 结论与设计输入

- **边界关闭**：`-DB2M_TMA -DB2M_INTRA` 现在能编译，且 **11 项输出逐位一致**——「TMA 只换布局」
  在含 `intra` 折叠的生产核上同样成立（补上 §67.4 的缺口）。
- **但生产核默认取非 TMA**：在 **孤立 GEMM 核** 上 TMA 曾净赚 ~5%（P7-49/§52），到了**六段齐全的
  生产核**（§67/§68 已证它是 `tensor 3%`、标量/发射受限、非 DRAM bound）就变成 **慢 8–12%**。
  机制与本专题反复出现的判据一致：TMA 的收益来自「消 `cp.async` 的 L2 读放大 / 走独立引擎」，
  而那**只有在核被 DRAM/L2 双侧钉住时才兑现**（§51.6「省字节要落最紧 pipe」）；生产核既已不是
  带宽受限，TMA 增加的 per-stage `mbar_wait + __syncthreads` 反而成了净负。**「同一旋钮在孤立核
  与整核里可以反号」**（同 §61/§64）。
- **设计输入**：目标卡 CUDA B2 **默认用非 TMA 的 `prod[_hdw][_intra]`**；只有当核被重构成
  `h/dh` 大操作数搬运主导、真正回到 DRAM-bound 时，`-DB2M_TMA` 才值得重开（届时用 §54 的
  NAT 个 128 B swizzle atom）。本次修复保证这个开关**随时可叠加**且**数值无损**。
- **边界/待验证**：只测 `K=V=128`、`BT=16`、fp32、`-DB2M_INTRA` 的折叠形态；`B2M_TMA` 与
  `-DB2M_DAMMA`（§69）的组合、以及 TMA 版落进 P7-64 的整条反向链（`pci` 变体）未做端到端（待验证）。

## 71. 关闭 §70.6 的两条待验证：`-DB2M_TMA × -DB2M_DAMMA` 让生产核的 TMA 代价**归零**（代价原来就是标量 DA 的 bank conflict）；TMA 折叠版落整链（P7-68）

§70.6（P7-67）留下两条明确的待验证：(1) `-DB2M_TMA` 与 `-DB2M_DAMMA`（§69 把 DA 段从标量 FMA
改成 `mma.sync` 的开关）**组合**到生产核里会怎样——尤其 P7-66 只在**孤立** `-DB2M_DA` 核上验过
`da_mma_tma`；(2) TMA 版的生产 B2 落进 P7-64 的**整条反向链**（`pci` 变体，此前只跑过非 TMA 的
`prod_hdw[_intra]`）。本节两条一起关掉，并**修正 §70.5 对 TMA 生产代价的归因**。

### 71.1 方法

生产核全部是 `-DB2M_PROD -DB2M_HDW`（六段齐全、`dw` 往返已删，§63/§64），四个变体：

| tag | flags | DA 实现 |
|---|---|---|
| `sc` | `-DB2M_PROD -DB2M_HDW` | 标量 FMA（P7-54） |
| `sc_tma` | `+ -DB2M_TMA` | 标量 FMA，TMA stage |
| `mma` | `+ -DB2M_DAMMA` | `mma.sync.tf32`（P7-66） |
| `mma_tma` | `+ -DB2M_DAMMA -DB2M_TMA` | `mma.sync`，TMA stage |

`-DB2M_DAMMA × -DB2M_TMA` **本身无需任何修复**：生产 launch 的 TMA 分支照旧（§70.1 修好的那处
实参守卫）在核内现编码张量图（`b2_mma_probe.cu:1914-1937`），四个变体一次编译通过。

**注意**：`-DB2M_HDW` 下生产核**不写** `dw` 核心输出（`dw` 往返被删、`dw` 指针只是 scratch），
故逐位比对**必须排除 `dw`**（否则比的是未初始化显存）；`dgk` 仍被核心正常写出、纳入比对。

### 71.2 Part A：正确性——四类比对

`harness/check_bwd_b2_tma_damma.py` Part A，6 组 shape（含非整除 `T=250`、单 chunk `T=64`），
每项输出 `max_abs`：

| case | NT | `sc_tma` vs `sc` | `mma_tma` vs `mma` | `mma` vs `sc`（非 DA） | `mma` vs `sc`（`dApre/dA`） |
|---|---|---|---|---|---|
| B8 T2048 H8 | 128 | 全 0 | 全 0 | 0 | 1.90e-3 / 1.45e-3 |
| B2 T1024 H8 | 64 | 全 0 | 全 0 | 0 | 1.85e-3 / 1.24e-3 |
| T512 H4 | 32 | 全 0 | 全 0 | 0 | 1.64e-3 / 1.06e-3 |
| T256 H2 | 16 | 全 0 | 全 0 | 0 | 1.82e-3 / 1.10e-3 |
| T250 H3 非整除 | 16 | 全 0 | 全 0 | 0 | 1.29e-3 / 9.32e-4 |
| T64 H1 单 chunk | 4 | 全 0 | 全 0 | 0 | 1.22e-3 / 8.11e-4 |

两条独立结论：
- **`*_tma` vs 其非 TMA 双胞胎：8 项输出全部逐位（`max_abs=0`）**（`dv2/dvb/dkgb/db/dqg/dkg/dg/dgk`
  ＋ `dApre/dA`），即 TMA 只换 stage 布局、不改任何算术——把 §70.2 的结论从「`-DB2M_INTRA` 折叠形态」
  推广到「`-DB2M_DAMMA` 的 mma DA 形态」。
- **`mma* vs sc*`：非 DA 8 项逐位、只有 `dApre/dA` 落到 tf32 底噪（1.2–1.9e-3）**，与 §69.2 的
  孤立结论一致（这里是生产态：DA 的 `dv/dw` 操作数来自同一核的 GEMM 段，量级更大故相对误差略高，
  仍是纯 tf32 舍入）。

### 71.3 Part B（关键）：`-DB2M_DAMMA` 把 TMA 的符号**从 +4~6% 拨回 ~0%**

生产核 CUPTI `dev`（µs，`B8 T2048 H8` = `1278 / 1334 / 1212 / 1211`）：

| case | `sc` | `sc_tma` | `mma` | `mma_tma` | **`sc_tma/sc`** | **`mma_tma/mma`** | `mma/sc` |
|---|---|---|---|---|---|---|---|
| B8 T2048 H8 | 1278.3 | 1334.5 | 1211.8 | 1211.3 | **1.044** | **1.000** | 0.948 |
| B2 T1024 H8 | 167.9 | 174.4 | 160.6 | 161.6 | **1.039** | **1.006** | 0.957 |
| T512 H4 | 19.5 | 20.4 | 18.1 | 18.0 | **1.046** | 0.993 | 0.929 |
| T256 H2 | 18.7 | 19.7 | 17.5 | 17.1 | **1.057** | 0.978 | 0.936 |
| T250 H3 非整除 | 18.9 | 19.8 | 17.6 | 17.4 | **1.044** | 0.989 | 0.929 |
| T64 H1 单 chunk | 18.2 | 19.1 | 16.9 | 16.7 | **1.048** | 0.984 | 0.930 |

- 没有 DAMMA 时，TMA 仍单调慢 **4–6%**（与 §70.4 的 8–12% 同向；本节 case 集与基线不同，
  非 TMA 的 `sc` 在 `B8` 上 `1278 µs`，与 §67/§68 的 `1265–1288 µs` 互证口径）。
- **一旦 DA 改成 mma，TMA 的代价消失**：`mma_tma/mma` 落在 **0.978–1.006**（`B8` 上 `1.000`），
  且大 shape 反而是 TMA 略快。也就是说 §70.5 的「生产核默认取非 TMA」在 **DAMMA 打开后要重新判断**：
  此时 TMA 免费、还省 L2 字节（见 §71.5），是个纯收益开关。

### 71.4 Part C：TMA 折叠版落整链——数值逐位、链级 TMA 代价 ~1–2%

把 `-DB2M_INTRA` 折叠版（B3 对角进单 launch）也做 TMA 变体，落进 P7-64 的链
`dAv(Triton) → CUDA-B1 → CUDA-B2-prod_intra[+TMA] → 反向 cumsum`，4 组 TCASES：

- **数值**：`pci_tma` vs `pci` 的 **`dq/dk/dv/dbeta/dg/dh0` 全部 `max_abs=0`（逐位）**；
  两者 vs FLA autograd 的最坏 `max_rel` 完全一致（`dq 3.949e-3 / dk 3.196e-3 / dv 9.500e-3 /
  dbeta 5.456e-3 / dg 4.659e-3 / dh0 8.889e-3`），与 §67.2 的同阶、`dv/dh0` 与 §45 `cf` 逐位。
- **链级计时**（dev µs；`pci/FLA` 一列是 `pci` 相对 FLA 反向的比）：

| case | `cf` dev | `pci` dev | `pci_tma` dev | `pci/FLA` | **`pci_tma/pci`** |
|---|---|---|---|---|---|
| T=512 H4 | 73.8 | 94.6 | 95.1 | 0.47 | **1.006** |
| B2 T=1024 H8 | 257.5 | 387.8 | 395.2 | 0.59 | **1.019** |
| B8 T=2048 H8 | 1402.4 | 2440.3 | 2498.2 | 0.59 | **1.024** |
| T=250 H3 非整除 | 50.1 | 63.9 | 65.3 | 0.45 | **1.022** |

链级 TMA 代价只有 **~1–2%**（比生产核单独测量时的 ~4% 更小，因为链里还摊了 B1/dav/cumsum）。
这组数同时**复核 §67.3**：`pci`（CUDA B2 生产核）比融合 Triton `cf` 慢 **~1.6–1.7×**
（`B8` `2440/1402=1.74×`），与 §67 一致；目标卡反向默认仍是 `cf`。

### 71.5 机制（`ncu`）：TMA 的生产代价**就是标量 DA 的 bank conflict**

`harness/probe_b2_td_ncu.sh`（`harness/probe_b2_td_ncu.out.txt`，`B8 T2048 H8`，4 变体逐指标）：

| variant | Duration (ms) | **bank conflict** | L2 bytes (GB) | DRAM 读 (GB) | regs |
|---|---|---|---|---|---|
| `sc` | 1.27 | **23.46 M** | 3.92 | 1.56 | 112 |
| `sc_tma` | 1.33 | **28.31 M** | 3.25 | 1.56 | 128 |
| `mma` | 1.21 | **17.30 M** | 3.83 | 1.56 | 104 |
| `mma_tma` | 1.21 | **17.30 M** | 3.26 | 1.56 | 126 |

读法：
- TMA 两列的 **L2 字节都低 ~0.67 GB**（3.92→3.25 / 3.83→3.26）、DRAM 读不变——这正是 P7-49/§52
  说的「TMA 消 `cp.async` 的 L2 re-touch」。
- 但 **bank conflict 的行为分叉**：标量 DA 下 TMA 把冲突从 `23.46 M` 抬到 `28.31 M`（**+4.85 M**）；
  mma DA 下 TMA **一动不动**（`17.30 M` 两列完全相同）。
- `sc` 的 `23.46 M` 与 `mma` 的 `17.30 M` **与 §69.4 逐值吻合**（§69 记「只削 ~40%」）。

即：**生产核里 TMA 那 4–6% 的代价，本体就是标量 DA 段在 TMA stage 布局下多出来的 4.85 M 次
shared-load bank conflict**——不是 per-stage `mbar_wait + __syncthreads` 的协议开销（§70.5 当时的
推测），也不是带宽。DAMMA 把 DA 的 shared 读换成 `mma` 片元读（对 stage 行距不敏感），这 4.85 M
冲突连同标量 DA 原本的 6 M 一起消失，TMA 的 L2 字节收益就**免费**了。

**这修正了 §70.5 的归因**：TMA 在生产核的净负不是「非带宽受限后 per-stage 同步成净负」，而是
**它把标量 DA 的 bank conflict 放大了**；换掉标量 DA（§69 已因别的动机做了）之后，TMA 回到
中性偏正。

### 71.6 结论与设计输入

- **两条待验证全关**：(1) `-DB2M_TMA × -DB2M_DAMMA` 在生产核里能编译、非 DA 8 项逐位、`dApre/dA`
  只落 tf32 底噪；(2) TMA 折叠版落整链数值逐位、链级只慢 ~1–2%。
- **修正 §70.5**：生产核 TMA 的代价**由标量 DA 段的 bank conflict 承担**，而非 per-stage 同步。
  因此 **DAMMA 打开后 TMA 反号归零**（`mma_tma/mma ≈ 1.0`，大 shape 还略快），此时 TMA 是
  「省 L2 字节、免费」的开关。
- **设计输入（更新）**：目标卡 CUDA B2 若走 `-DB2M_DAMMA`（§69 已推荐：段内 2–3×、生产核 +5–7%），
  则**可以同时开 `-DB2M_TMA`**（免费省 L2）；若 DA 仍是标量，则保持非 TMA。这与 §70.5 的
  「默认非 TMA」并不矛盾——那是**标量 DA 形态**下的结论，本节给出随 DA 实现切换的判据。
- **判据（新，并入方法学）**：`ncu` 里「某开关的 bank-conflict 增量」要**分实现形态读**——
  同一 `-DB2M_TMA`，在标量段是 +4.85 M 冲突（净负），在 mma 段是 0（中性）。把「换搬运布局」
  的收益/代价归到具体的段实现上，而不是笼统记成「生产核 vs 孤立核反号」。
- **边界/待验证**：链级只测到 `pci_tma`（折叠版、标量 DA）；`pci_mma`/`pci_mma_tma`
  （DAMMA 进链）未做端到端——按本节机制预计 `pci_mma_tma` 与 `pci_mma` 链级差 ~1%（TMA 免费），
  留待有需求时验证。仅 `K=V=128`、`BT=16`、fp32。

## 72. 关闭 §71.6 的边界：`-DB2M_DAMMA`（mma 化 DA）落进**完整反向链**——链级数值无损（梯度差 ≤8e-6）、比标量 DA 链快 ~1–2.5%（P7-69）

§71.6（P7-68）在关掉两条待验证后，仍留了一句话的边界：「链级只测到 `pci_tma`（折叠版、标量 DA）；
`pci_mma`/`pci_mma_tma`（DAMMA 进链）未做端到端」。本节把这两条链补上，并回答一个此前只在
**合成操作数**上量过的量：`-DB2M_DAMMA` 那段 tf32 改写（§69：孤立 `dApre` 差 3.4e-4、生产核
`dApre/dA` 差 1.2–1.9e-3，§71.2）在**真实组装链**里到底还剩多少。

### 72.1 方法

四个折叠核（都是 `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`，即 B2 六段齐全、`dw` 往返已删、B3 对角
intra 折进单 launch，P7-61/§64 与 P7-63/§66），只有「DA 实现」与「stage 布局」两维：

| tag | 额外 flags | DA 实现 | stage |
|---|---|---|---|
| `pci` | — | 标量 FMA | `cp.async` |
| `pci_tma` | `-DB2M_TMA` | 标量 FMA | TMA |
| `pci_mma` | `-DB2M_DAMMA` | `mma.sync.tf32` | `cp.async` |
| `pci_mma_tma` | `-DB2M_DAMMA -DB2M_TMA` | `mma.sync` | TMA |

每条都链成 `dAv(Triton) → CUDA-B1 → CUDA-B2-prod_intra → 反向 cumsum(B0)`，只吃 FlashKDA 的
16-chunk K1 workspace（同 §67）。新增 `harness/check_bwd_b2_damma_e2e.py`
（`harness/check_bwd_b2_damma_e2e.out.txt`，空闲卡），4 组 TCASES（含非整除 `T=250`）。Part A 除
六项梯度外，还把每次 prod launch 的**原始 B2 输出**用 monkeypatch 捕获下来，直接对照「核级改写量」
与「链级残留量」。

### 72.2 Part A：`pci_mma_tma` 与 `pci_mma` **逐位相同**；`pci_mma` 与 `pci` 只差 ≤8e-6

6 组 shape 的 `max_abs`（`pci_mma` vs `pci`，以及 `pci_mma_tma` vs `pci_mma`）：

| case | `pci_mma_tma` vs `pci_mma` | `dq` | `dk` | `dv` | `dbeta` | `dg` | `dh0` |
|---|---|---|---|---|---|---|---|
| T512 H4 | 六项全 **BIT-IDENTICAL** | 0 | 1.03e-06 | 0 | 3.00e-07 | 1.36e-07 | 0 |
| B2 T1024 H8 | 六项全 **BIT-IDENTICAL** | 0 | 1.29e-06 | 0 | 3.81e-07 | 1.02e-07 | 0 |
| B8 T2048 H8 | 六项全 **BIT-IDENTICAL** | 0 | 5.45e-06 | 0 | 7.99e-06 | 9.89e-07 | 0 |
| T250 H3 非整除 | 六项全 **BIT-IDENTICAL** | 0 | 1.77e-08 | 0 | 1.86e-09 | 1.40e-09 | 0 |

- **`pci_mma_tma` vs `pci_mma`：六项梯度全部 `max_abs=0`（逐位）**——把 §71.2「TMA 只换 layout、
  不改算术」的结论从核级推广到**整链**。
- **`pci_mma` vs `pci`：`dq`/`dv`/`dh0` 逐位 = 0**（`dq` 由 `dAqk` 定、`dv`/`dh0` 完全由 B1 定，
  与 DA 实现无关），`dk/dbeta/dg` 只差 **≤8.0e-6**。

四个变体 vs FLA autograd 的最坏 `max_rel` **逐值完全相同**：

| fld | 四变体（`pci`/`pci_tma`/`pci_mma`/`pci_mma_tma`） |
|---|---|
| `dq` | 3.949e-03 |
| `dk` | 3.196e-03 |
| `dv` | 9.500e-03 |
| `dbeta` | 5.456e-03 |
| `dg` | 4.659e-03 |
| `dh0` | 8.889e-03 |

即 **DAMMA 改写对「链级 vs FLA」的误差没有可测量的贡献**（打印精度内四个变体一致），与 §71.4 的
`pci` 行逐值吻合。

### 72.3 为什么链级残留只有 ~5e-6，而 §71.2 是 ~1e-3：**真实链的 DA 操作数小 3 个数量级**

把 §72.2 的链级差值同每次 launch 的**原始 B2 输出**对照（同 4 组 shape，`max_abs`）：

| case | B2 `dApre` | B2 `dA` | B2 `dkg` | B2 `dqg` | 链级 `dk` 残留 |
|---|---|---|---|---|---|
| T512 H4 | 5.60e-06 | 4.98e-06 | 1.03e-06 | **0** | 1.03e-06 |
| B2 T1024 H8 | 6.32e-06 | 4.11e-06 | 1.29e-06 | **0** | 1.29e-06 |
| B8 T2048 H8 | 8.14e-06 | 6.26e-06 | 5.45e-06 | **0** | 5.45e-06 |
| T250 H3 非整除 | 2.37e-06 | 1.30e-06 | 1.77e-08 | **0** | 1.77e-08 |

两个关键事实：

1. **真实链里 DA 的 tf32 改写量本身就小 3 个数量级**（`dApre` ≤8.1e-6，而 §71.2 用 `make_inputs`
   的 O(1) 合成操作数测到 1.2–1.9e-3）。根因是**操作数幅度**：链里 `dApre = dv@v^T − dw@kg^T` 的
   `dv`（B1 的 bf16 出口）、`v_new`/`h`（K1 workspace 的 bf16 派生量）幅度小，tf32（10-bit 尾数）
   的**绝对**舍入误差随幅度线性缩小——tf32 vs fp32 的差是相对 5e-4 量级，操作数一小，绝对差就小。
   所以「DA 的 tf32 底噪」不是一个固定的绝对值，而是随那个 shape 的操作数幅度走（§71.2 的 1e-3
   是合成操作数下的上界）。
2. **`dqg`（门控后的最终 `dq`）在 B2 出口就是逐位 0**——`dq` 只由 `dAqk`（来自 dAv Triton 核）
   决定，与 DA 段无关；DA 改写只经 `dkg` 影响 `dk/dbeta`。

即：**链级残留 ≈ 核级 `dApre` 改写量本身（无放大、也无额外衰减）**——因为 intra 折进段把
`dAkk = −daB`（掩码后的 sandwiched `dA`）直接喂 mma，扰动线性传过去；而 §71.2 那 1e-3 是
**合成操作数**下的核级量，不该外推成链级量。

### 72.4 Part B：DAMMA 链比标量链快 ~1–2.5%，其上 TMA 再快 ~1–2%（与 §71.3 预测同向）

反向链 CUPTI `dev`（µs，空闲卡）：

| case | FLA dev | `cf` dev | `pci` | `pci_tma` | `pci_mma` | `pci_mma_tma` | `pci_mma/pci` | `pci_mma_tma/pci_mma` | `pci/FLA` |
|---|---|---|---|---|---|---|---|---|---|
| T512 H4 | 204.9 | 74.2 | 94.3 | 94.8 | 93.5 | 92.5 | 0.991 | 0.989 | 0.46 |
| B2 T1024 H8 | 660.0 | 256.4 | 388.6 | 395.8 | 381.2 | 374.1 | 0.981 | 0.981 | 0.59 |
| B8 T2048 H8 | 4102.7 | 1400.3 | 2423.1 | 2481.0 | 2361.5 | 2313.1 | 0.975 | 0.980 | 0.59 |
| T250 H3 非整除 | 141.4 | 50.3 | 63.9 | 65.2 | 64.0 | 63.5 | 1.001 | 0.992 | 0.45 |

- **`pci_mma/pci` = 0.975–1.001**：DAMMA 在整链里兑现 ~1–2.5%（大 shape 更明显，`B8` 2423→2362µs），
  与 §69 的「生产核 +5–7%、段占核 ~14%」自洽（链里还摊了 B1/dAv/cumsum，故比例更小）。
- **`pci_mma_tma/pci_mma` = 0.980–0.992**：**DAMMA 打开后 TMA 从「慢 4–6%」反号成「快 ~1–2%」**，
  兑现 §71.3/§71.6 的预测（TMA 代价本体是标量 DA 的 bank conflict，换成 mma DA 后免费）。
  `cf`（融合 Triton B2/B3）仍是链级赢家：`pci/cf ≈ 1.68–1.73`，复核 §67.3。
- `wall`（event-pair）远大于 `dev`（host 派发），数量级与 §71.4 一致，趋势同 `dev`。

### 72.5 结论与设计输入

- **§71.6 的边界关闭**：`pci_mma` / `pci_mma_tma`（DAMMA 进链）端到端核验——
  **`pci_mma_tma` 与 `pci_mma` 六项梯度逐位相同、四个变体 vs FLA 的 worst-case 完全一致**，
  **链级数值无损**。
- **修正 §71.2 的量级读法**：「DA 改写落 tf32 底噪 1.2–1.9e-3」是**合成操作数**下的核级上界；
  真实链里 DA 操作数小 3 个数量级，**核级改写只有 ~5–8e-6、链级残留同阶（≤8e-6）**。凡引用
  「某段的 tf32 底噪」都要**区分合成操作数与真实链操作数**——tf32 误差随操作数幅度走。
- **性能**：DAMMA 进链值 ~1–2.5%（大 shape），其上再加 TMA 再值 ~1–2%——两者**可叠加且免费/ch 正**，
  与 §71.6 的设计输入一致（目标卡 CUDA B2 走 DAMMA 时可同时开 TMA）。
- **默认部署不变**：目标卡反向默认仍是 §45 的 `cf`（`dAv→CUDA-B1→融合 Triton B2/B3→cumsum`）；
  CUDA B2 生产核（`pci*`）是「正确但更慢 ~1.7×」的研究形态，DAMMA/TMA 只把它从 1.73× 收到 ~1.67×。
- **边界**：仅 `K=V=128`、`BT=16`、fp32 操作数（B1 出口 bf16 转 fp32）；只测 `-DB2M_PROD -DB2M_HDW
  -DB2M_INTRA` 折叠形态（非折叠 `pc` 的 DAMMA 未单独落链，机制相同）。

## 73. 关闭 §62 的**最后一条**「store 未向量化」边界：`dA`/`dApre` 的 `[BT,BT]` store 改 `st2`——逐位无损、生产核 **0.93–0.96×**（P7-70）

§59（P7-59）把 `st2()` 成对 store 助手接到了 B2 的**大**非 GEMM 输出（`dv2/dvb`、`dkgb`、
门控 `dqg/dkg/dg`），并把每轮边界里写的「本段 store 未接 `-DB2M_VECST`」逐条划掉，但**明确留下**
了 `[BT,BT]` 的 `dA`/`dApre`（「小张量 + 一 thread 一 entry 的布局」）没做。那是整个专题边界清单里
**唯一**还挂着的「store 未向量化」注记（§57.6/§58.6/§62.6）。本节把它关掉，并给一个此前没量过的
问题定性：这类 store 与 §51/§59 那批**是否同类**。

### 73.1 关键区别：`dA`/`dApre` 的 store **本就完全合并**，不是半 sector 问题

§51（P7-48）发现 B2 默认 store 的危害机理很具体：门控/输出段的 store 是**片段跨步**布局
（4 个 lane 各写同一 32 B sector 的字节偏移 `{0,8,16,24}`），**每条 32-bit store 只覆盖半个 sector**，
`l1tex st` 扇区 = 数据扇区的 **1.98×**；合成 `float2` 后立刻回到 1.0×。§59 的收益正来自这里。

而 `dA`/`dApre` 的 store（`b2_mma_probe.cu`）是 `row = idx/BT, col = idx%BT`、`idx = threadIdx.x`，
即**相邻线程写相邻列**——warp 覆盖两行 `BT` 个 float = 两个 64 B 段，128 B 请求正好 4 个 sector，
**扇区数已是最小**。所以这两段的向量化**改不了访存字节**，最多把 store **指令数减半**
（256 条 `STG.E.32` → 128 条 `STG.E.64`）。预期收益因此**远小于** §59——本节实测确认它仍是
一个**小而确实正**的赢，而不是 wash。

改法（**算术一字未动**）：`dApre` store 把 `if (idx < BT*BT)` 改成 `if (idx < BT*(BT/2))`、每线程
经 `st2` 写一对相邻列（`b2_mma_probe.cu` 的 `-DB2M_DA` 段）；最终 `dA` sandwich store（`-DB2M_SAND`
段）同样按列成对、`v0/v1` 两个掩码分别作用于 `jj`/`jj+1`。`st2` 在 `-DB2M_VECST` 下一条
`STG.E.64`、否则两条标量——故**非 VECST 路径与旧代码逐位等价**（`BT` 为偶数，`BT=16`）。

### 73.2 Part A：standalone `da`/`sand` base vs vecst **逐位相同**

`harness/check_bwd_b2_vecst_da.py`（`harness/check_bwd_b2_vecst_da.out.txt`，空闲卡）。4 组 shape
（含非整除 `T=250`）上 `da`（`b2da_launch`，输出 `dApre`）与 `sand`（`b2sand_launch`，输出
`dApre`+`dA`）base vs vecst 的 `max_abs` **全部 = 0**（worst `max_abs=0`，BIT-IDENTICAL）。

### 73.3 Part B：standalone `da`/`sand` 的 CUPTI `dev`——**0.93–0.97×**

| shape | `da` base | `da` vecst | ratio | `sand` base | `sand` vecst | ratio |
|---|---|---|---|---|---|---|
| B8 T2048 H8 | 956.2 | 924.3 | **0.967** | 1008.2 | 973.7 | **0.966** |
| B2 T1024 H8 | 127.3 | 122.8 | **0.965** | 134.7 | 130.5 | **0.969** |
| T512 H4 | 13.6 | 12.8 | **0.945** | 14.7 | 13.7 | **0.931** |
| T250 H3 非整除 | 12.6 | 11.9 | **0.945** | 13.8 | 13.0 | **0.942** |

单核 **3–7%**。`da`/`sand` 都是 §57 里「标量 FMA、收缩维是全维」的那类段，**发射受限**
（§57：`da` 段把 bank conflict 拉到 ×2.8、`tensor` 纹丝不动），故**少一半 store 指令**能兑现时间。

### 73.4 Part C：生产核（`-DB2M_PROD -DB2M_HDW`）**十项输出逐位相同**、`dev` **0.93–0.96×**

| shape | prod base | prod vecst | vecst/base |
|---|---|---|---|
| B8 T2048 H8 | 1262.5 | 1208.5 | **0.957** |
| B2 T1024 H8 | 164.9 | 158.2 | **0.960** |
| T512 H4 | 18.9 | 17.8 | **0.943** |
| T250 H3 非整除 | 18.2 | 16.9 | **0.932** |

十项输出（`dv2/dvb/dkgb/db/dApre/dA/dqg/dkg/dg/dgk`）base vs vecst **全部 `max_abs=0`**。生产核
（§67/§68 已证**非 DRAM bound、标量/发射受限**）里这个免费旋钮值 **4–7%**——比 §59 在大形状的
0.983–0.990×更明显，与「本核瓶颈在发射侧、不在带宽/扇区」的历次归因（§61/§63/§68）一致。

### 73.5 Part D：SASS 计数——`STG.E.32` **37→19**、`STG.E.64` **0→18**

`sand` build 的 SASS：`STG.E`(32b) **37→19**、`STG.E.64` **0→18**——18 条标量 store 被合成了
64-bit（正是 `dApre`/`dA` 两处 `st2` 及其展开）。残留的 19 条标量来自 `dw`/`dgk` 列归约等
**非成对形态**的 store（不在本节范围）。

### 73.6 结论

- **§62.6 的最后一条边界关闭**：`dA`/`dApre` 的 `[BT,BT]` store 已接 `st2`，**逐位无损**、生产核
  **0.93–0.96×**。至此 B2 生产核里**所有** store 都已是向量化/已评估形态。
- **方法学（新）**：同叫「标量 store」，危害可以完全不同——§51/§59 的**片段跨步**（半 sector）是
  **扇区**问题，向量化**改字节**；本节 `dA`/`dApre` 的**相邻列**本就**合并最小**，向量化只**砍指令数**，
  收益靠**本核发射受限**兑现。判「该不该向量化」先看**线程→地址映射**（是否 `col = idx%BT` 型连续），
  再看本核是**扇区 bound**还是**发射 bound**。
- **边界**：`-DB2M_VECST` 只测到 `prod_hdw`（非 TMA、非 INTRA 折叠）；未与 `pci*`（`PROD+HDW+INTRA`）
  整链、`-DB2M_TMA` 交叉重测（机制相同、`dA`/`dApre` store 与 stage 布局无关）。目标卡反向默认仍
  是 §45 的 `cf`（CUDA B2 生产核整体仍比融合 Triton 慢 ~1.7×，见 §67）；本轮的 4–7% 是**该研究形态
  内部**的免费收尾，不改变默认部署。

## 74. 关闭 §73.6 的边界：`-DB2M_VECST` 落进**完整反向链**、并与 TMA / DAMMA 三向交叉（P7-71）

§73（P7-70）把 `st2` 成对 store 接到了 `[BT,BT]` 的 `dA`/`dApre`，但**只**在孤立的 `prod_hdw`
（`-DB2M_PROD -DB2M_HDW`，非 `-DB2M_INTRA` 折叠、非 `-DB2M_TMA`）上量过，明确留了一条边界：
「`-DB2M_VECST` 只测到 `prod_hdw`；未与 `pci*`（`PROD+HDW+INTRA`）整链、`-DB2M_TMA` 交叉重测」。
本节把它关掉，顺带把 §69–§73 的「DA 实现 × stage 布局 × store 宽度」收成一张完整的三向表。

新增 `harness/check_bwd_b2_vecst_e2e.py`（`harness/check_bwd_b2_vecst_e2e.out.txt`，空闲卡
`CUDA_VISIBLE_DEVICES=1`）。八份 fold 内核 = `{scalar DA, mma DA} × {非 TMA, TMA} × {标量 store,
VECST}`，全部 `-DB2M_PROD -DB2M_HDW -DB2M_INTRA`（P7-63 的单次 launch 折叠版），各自链成
`dAv(Triton) → CUDA-B1 → CUDA-B2-prod_intra → 反向 cumsum`，只吃 FlashKDA 16-chunk K1 workspace。

### 74.1 Part A：四对（VECST 开/关）孪生**六项梯度全部逐位相同**（`max_abs=0`）

4 组 case（含非整除 `T=250`、`B=2`）上，`pci_vec/pci`、`pci_tma_vec/pci_tma`、
`pci_mma_vec/pci_mma`、`pci_mma_tma_vec/pci_mma_tma` 四对孪生的 `dq/dk/dv/dbeta/dg/dh0` 全部
`max_abs = 0`（BIT-IDENTICAL）。这与机理一致：**VECST 只改 store 的位宽**（`STG.E.32`→`STG.E.64`），
不改任何累加/门控算术，也不管 stage 布局（§73.1）——故与 DA 实现、TMA 与否都正交。

八份变体逐 case 与 FLA autograd 的 worst-case `max_rel` **逐值完全相同**，且与 §72（P7-69）逐值
一致：`dq 3.949e-3 / dk 3.196e-3 / dv 9.500e-3 / dbeta 5.456e-3 / dg 4.659e-3 / dh0 8.889e-3`。
即 VECST **不改变**链级数值，只改指令形态。

### 74.2 Part B：链级 `dev`——VECST 值 **0.97–0.99×**、`pci*` 基线复现 §72/§70

只计反向的 `dev`（CUPTI，空闲卡；括号内是 `*_vec` 相对其非 VECST 孪生的比值）：

| case | FLA | `cf` | `pci` | `pci_vec` | `pci_tma` | `pci_tma_vec` | `pci_mma` | `pci_mma_vec` | `pci_mma_tma` | `mma_tma_vec` |
|---|---|---|---|---|---|---|---|---|---|---|
| T=512 H=4 | 203.8 | 75.4 | 94.3 | **94.1 (0.998)** | 95.0 | **95.3 (1.003)** | 93.9 | **92.1 (0.981)** | 92.8 | **92.7 (0.999)** |
| B2 T=1024 H=8 | 658.0 | 257.1 | 387.7 | **377.8 (0.974)** | 394.8 | **389.5 (0.987)** | 381.2 | **372.1 (0.976)** | 373.8 | **372.1 (0.995)** |
| B8 T=2048 H=8 | 4101.7 | 1399.5 | 2420.1 | **2345.1 (0.969)** | 2480.3 | **2443.4 (0.985)** | 2369.7 | **2294.9 (0.968)** | 2306.7 | **2295.5 (0.995)** |
| T=250 H=3 非整除 | 140.8 | 50.3 | 64.7 | **63.3 (0.979)** | 65.6 | **65.3 (0.995)** | 64.1 | **62.7 (0.979)** | 63.4 | **62.1 (0.979)** |

- **VECST 在整链上值 1–3%（`B2`/`B8`/`T250` 落 0.968–0.995）**，方向**基本恒正**；最小 shape
  `T512 H=4` 落 **0.98–1.00（噪声内）**——链里固定开销（`dAv`/B1/cumsum/launch）占比最大，B2 少一半
  store 指令被摊薄。与 §73 的单核 4–7% 同号、但被链路稀释。
- **收益随非 TMA > TMA 递减**：`pci`/`pci_mma`（非 TMA）落 0.968–0.998，而 `pci_tma*` 只 0.985–1.003。
  与 §67/§71「生产核里 TMA 版 store 本就不是最紧项」一致——TMA 把操作数访存从 LSU 发射挪到独立引擎，
  少一半 store 指令分到的红利也少（判据见 §74.3）。
- **`pci*` 基线复现前几轮结论**：`pci_tma/pci` = 1.007–1.025（TMA 略慢），而 `pci_mma_tma/pci_mma`
  = 0.973–0.989（**DAMMA 打开后 TMA 反号成略快**，兑现 §71.3/§72.3）；`pci_mma/pci` = 0.979–0.996、
  `pci/FLA` = 0.46–0.59、`cf/pci` = 0.58–0.80。全部与 §69–§72 同阶。
- `wall` 口径（含 host 派发）同样单调：如 `B8` `pci` 2431.0→`pci_vec` 2372.9、`pci_mma` 2394.0→
  `pci_mma_vec` 2326.3。

### 74.3 结论

- **§73.6 的边界关闭**：`-DB2M_VECST` 与 `pci*`（`PROD+HDW+INTRA`）整链、`-DB2M_TMA` 三向交叉
  ——**四对孪生六项梯度全部逐位相同**，链级值 1–3%（`B2`/`B8`/`T250` 落 0.968–0.995，最小 shape
  噪声内）。至此 B2 生产核的所有 store 变体（标量片段跨步 → `float2`；标量相邻列 → `st2`）在
  **隔离核与整链**两处都核验过：逐位无损、收益=「少 store 指令」在发射受限核上的兑现。
- **判据复用（§73.6）**：同叫「标量 store」，`dA`/`dApre` 是**相邻列**（扇区已最小，向量化只砍指令数）；
  其收益方向与大小由「核是否发射受限」和「该项是否在关键路径上的最紧 pipe」共同决定——TMA 版里它更小
  正因为 TMA 把访存压力从 LSU 发射挪走了一部分。
- **边界 / 不改变默认**：仅 `BT=16`、`K=V=128`、fp32 操作数；研究形态内部收尾。目标卡反向默认仍是
  §45 的 `cf`（`dAv→CUDA-B1→融合 Triton B2/B3→cumsum`），CUDA B2 生产核整体仍比它慢 ~1.7×（§67）。

## 75. 关闭 §54.5 的两条遗留：`BVK=128`（NAT=4）**正确但不划算**，且 `atom64` 的 L2 +0.04 GB 归因到**atom×stage 的 L2 重触**（P7-72）

§54（P7-51）把 `TMA + BVK=64` 走成两个 128 B swizzle atom、成为当时全程最快的一档，并在 §54.5 留了两条
明确的边界：「`BVK=128`（4 atom）未测」；「`atom64` 的 L2 2.39 GB 比 `tma32` 的 2.35 GB 略高（0.04 GB，
多出的 atom 边界 L2 re-touch），**未逐项归因**」。本节把这两条一起关掉。

新增 `harness/check_bwd_b2_tma_atom128.py`（`harness/check_bwd_b2_tma_atom128.out.txt`）+ `harness/probe_b2_tma_atom128_ncu.sh`
（`harness/probe_b2_tma_atom128_ncu.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=1`）。`b2_mma_probe.cu` 只放宽一处
`static_assert`（`BVK ∈ {32,64} → {32,64,128}`）——atom-major 布局本就是 `NAT=BVK/32` 泛化的（`Stage` 的
`[NAT][rows][32]` 与 `issue_tma` 的 `for a<NAT`），**算术一字未动**。

### 75.1 实验设计：干净的受控轴是 **NAT（=BVK/32）在 NSTAGE 固定下**

`BVK=128` 的 atom-major 单 stage smem = `NAT·(3·BT+2·KD)·32·4 B = 4·304·128 B = 155.6 KB`，两 stage 311 KB
**超过 227 KB 的动态 smem 上限**（launcher 的 `cudaFuncSetAttribute` 会直接 `rc=-3`）——故 BVK=128 **必须
`NSTAGE=1`**。因此为了不把「atom 数」与「流水深度」混在一起，本文把 TMA 的 `atom32/atom64/atom128`
**全部钉在 `NSTAGE=1`**（NAT=1/2/4）做 NAT 扫描，另保留 `NSTAGE=2` 的 `base32/tma32/atom64` 复现 §54 基线。

### 75.2 Part A：`atom128` 与 `base128` **逐位相同**（含非整除 T=250）

四组 shape（`BT=16`，torch 参考关 tf32 走真 fp32）：

| 量 | `atom128` vs `base128` | `base128` vs `base64` | `atom128` vs torch(fp32) | Triton(tf32) vs torch |
|---|---|---|---|---|
| `dq` | **0** | **0** | 3.583e-4 | 7.641e-4 |
| `dk` | **0** | **0** | 3.281e-4 | 6.571e-4 |
| `dw` | **0** | **0** | 3.408e-4 | 8.363e-4 |
| `dgk` | **0** | 3.815e-6 | 2.950e-7 | 1.311e-7 |

（最坏值；`atom128` vs `atom64` 的 `dq/dk/dw` 也逐位、`dgk` 差 2.86e-6。）两件事被钉死：(1) **NAT=4 的
atom 布局字节解码无误**（`atom128≡base128` 四位全 0）；(2) `base128≡base64` 的 `dq/dk/dw` 逐位——**k8 累加序
不随 BVK 变**（`base128` 的 1 tile × 16 kk 与 `base64` 的 2 tile × 8 kk 拼出同一条 `k=0,8,…,120` 序列）；
只有 `dgk` 的标量列归约分组随 tile 拆分不同、差 3.8e-6 fp32 舍入。`atom128` 自身仍落 tf32 底噪（3.6e-4），
**比 identical-op Triton 核（7.6e-4）更贴真 fp32**（同 §49.2）。

### 75.3 Part B：`BVK=128` **没有速度收益**——NS 被迫为 1、又失掉 V-tile 并行

CUPTI `dev`（kernel-only，µs）：

| shape | base32 | tma32ns1 | atom64ns1 | atom128ns1 | atom128ns1v | base128ns1 |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 520.8 | 498.9 | **486.1** | 502.5 | 489.9 | 504.3 |
| `B2 T1024 H8` | 72.4 | 68.8 | **67.7** | 75.8 | 73.7 | 76.2 |
| `T512 H4` | 9.3 | 9.9 | 9.9 | 10.4 | 8.5 | 10.2 |
| `T250 H3`（非整除） | 8.5 | 9.2 | 8.5 | 8.7 | 7.5 | 8.5 |

- **NSTAGE=1 下 NAT 扫描 `atom32/atom64/atom128` = 498.9/486.1/502.5（`B8`）**：`atom64` 最优，`atom128`
  反而略慢（1.034× `atom64ns1`）——因为 `VD/BVK=1` 把 V-tile 流水并行也丢掉，只剩 chunk 间双缓冲且 NS 还被
  钉死为 1。故 **`BVK=128` 是「正确但不可用」**（与 §53 的 `SWIZZLE_NONE` 同命、但原因不同：这里是并行度
  而非 bank 冲突）。
- `base128ns1` 同样无收益（cp.async 侧 3 tile→1 tile）。`-DB2M_VECST` 仍是免费项（`atom128ns1v` 比
  `atom128ns1` 快 2.5%，与 §51 一致）。

### 75.4 `ncu`：读侧修复与 bank conflict 都健康，L2 +0.04 GB 归因到 **atom×stage 的 L2 重触**

`B8 T2048 H8`（同一批 fp32 操作数）：

| 变体 | L2 bytes | srcunit 读 | l1tex 全局读 | bank conflict | DRAM 读 | DRAM SOL | Duration |
|---|---|---|---|---|---|---|---|
| `base32`（cp.async, NS=1） | 3.00 GB | 59.77M | 44.83M | 8.93M | 1.28 GB | 85.3% | 514.8 µs |
| `tma32ns1`（NAT=1, NS=1） | 2.35 GB | 39.85M | **0** | 8.49M | 1.28 GB | 88.2% | 498.9 µs |
| `atom64ns1`（NAT=2, NS=1） | 2.36 GB | 39.85M | **0** | 8.79M | 1.28 GB | 90.9% | 484.5 µs |
| `atom128ns1`（NAT=4, NS=1） | **2.42 GB** | 39.85M | **0** | **8.39M** | 1.28 GB | 87.0% | 507.6 µs |
| `tma32ns2`（NAT=1, NS=2） | 2.35 GB | 39.85M | **0** | 8.73M | 1.28 GB | 89.5% | 489.7 µs |
| `atom64ns2`（NAT=2, NS=2） | **2.39 GB** | 39.85M | **0** | 8.39M | 1.28 GB | 90.7% | 485.0 µs |

- **读侧修复与 bank conflict 在 `BVK=128` 下都健康**：`atom128` 的 `l1tex` 全局读 = 0（SASS `UBLKCP`
  10→**20**，5 操作数 × 4 atom）、srcunit 读恒 39.85M（=DRAM 扇区 1.0×）、bank conflict **8.39M**（不升反最低
  ——残留本就是 §50 定位的 `dgk` 列归约冲突，与 atom 数无关）。即 §54 的两条正面结论**外推成立**。
- **L2 归因**（把 L2 bytes 排成 NAT×NS 二维）：

  | | NAT=1 | NAT=2 | NAT=4 |
  |---|---|---|---|
  | **NS=1** | 2.35 | 2.36 | 2.42 |
  | **NS=2** | 2.35 | 2.39 | （311 KB 单 stage 超限、不可编译） |

  L2 bytes **随 NAT 增大、也随 NS 增大**（NS=1 下 NAT 1→2→4 为 2.35/2.36/2.42；NAT=2 下 NS 1→2 为
  2.36/2.39），而 **srcunit 读恒为 39,845,888、DRAM 读恒为 1.28 GB、l1tex 全局读恒为 0**。故 §54.5 的
  +0.04 GB **确认是「atom 边界 L2 重触」**（§54.5 的猜测方向正确），但它**不是「每个 atom 对固定 +0.02」
  的线性项**，而是随 **在飞的 box 数（∝ NAT×NS）** 增大——是 **L2 内部的边界/重触流量，不是 DRAM、也不是
  SM 侧读**（三者在本表都恒定）。故 `atom64` 相对 `tma32` 的 L2 差**既不影响带宽瓶颈（DRAM SOL 反升到
  90.9%）、也不影响正确性**。

### 75.5 结论 / 设计输入

- **§54.5 的两条遗留全部关闭**：`BVK=128`（NAT=4）**数值逐位正确、读侧修复与陷阱都健康，但无速度收益**
  （NS 被迫=1 + 无 V-tile 并行；`atom128ns1` ≥ `atom64ns1`）——**目标卡不需要 `BVK=128`**，`BVK=64`（NAT=2）
  仍是 atom 拆分的最优点。`atom64` 的 L2 +0.04 GB 归因为 **NAT×NS 相关的 atom 边界 L2 重触**（srcunit/DRAM/
  全局读三者恒定），**不是带宽或正确性问题**。
- **判据补充**：`lts__t_bytes` 这类「L2 总字节」含**重触**，与「DRAM 字节」「srcunit 读扇区」是三个不同
  口径——判「某个布局多花的字节是否为瓶颈」必须把它们分开读（本轮三者在 TMA 全族里分别恒定/单调），
  否则会把纯 L2 内部重触误当成访存瓶颈（同 §51.6「省字节要落最紧 pipe」的镜像）。
- **边界**：只测 `BT=16`、`K=V=128`、fp32、`B8T2048H8`（ncu）；`BVK=128` 的 `NSTAGE>1` 因 smem 超限
  不可编译（已由 launcher `rc=-3` 证实，非静默）；`atom128` 的 `dgk` 仍标量列归约。

## 76. B2 K2-mma GEMM 核的 **bf16 原生入口**：五个 streamed 操作数改 bf16 staging——与 fp32 核**逐位相同**、大 shape **0.73×**、DRAM 读**精确减半**（P7-73）

P7-64（§67.4）在把 CUDA B2 生产核落进完整反向链时留了一条明确的部署边界：**`b2_mma_probe.cu` 是
fp32/tf32 核，落链时 B1 的 bf16 `dh`/`dv2` 必须先 `.float()` 才能喂进来**（一次 elementwise 往返）。
§67/§72 的默认部署（`cf`）里 Triton B2/B3 直接吃 fp32，也一直走这条转换。ROADMAP 把「**bf16 原生入口**」
列为触发下一轮的真需求之一。本节给 K2 形状的 **B2 GEMM 核**加上 bf16-native 操作数入口（`-DB2M_BF16`），
把这条边界第一次量化。

新增 `harness/check_bwd_b2_bf16.py`（`harness/check_bwd_b2_bf16.out.txt`）+ `harness/ncu_b2_bf16_driver.py`
/ `harness/probe_b2_bf16_ncu.sh`（`harness/probe_b2_bf16_ncu.out.txt`，空闲卡 `CUDA_VISIBLE_DEVICES=2`）。

### 76.1 设计：只在 **staging** 换 bf16，mma 保持 `m16n8k8.tf32`——因为 bf16 ⊂ tf32

`-DB2M_BF16` 只改一处概念：五个 streamed 操作数 `do/vn/dv/h/dh` 在 smem 里按 **bf16** 暂存（`Stage` 的
数组类型 `float → __nv_bfloat16`、行距 `BVP=BVK+4 → OPS=BVK+8`，一次 `cp.async` 搬 8 个 bf16=16 B），
`rd()` 从 smem 读 bf16 后 `__bfloat162float` 扩到 fp32，再照旧 `to_tf32` 喂 `mma.sync.m16n8k8.tf32`。
**关键恒等式**：bf16 的 8 位尾数 ⊂ tf32 的 10 位尾数，故 bf16 值在 tf32 里**精确可表示**——`cvt.rna.tf32`
对已扩宽的 bf16 是恒等，乘积与「真正 bf16 mma」（`m16n8k16.bf16`）逐位相同，唯一差别是 fp32 累加序。
因此**不必重写 mma 指令/fragment 布局**（tf32 是 `m16n8k8`、bf16 是 `m16n8k16`、A/B 片段布局不同），
staging 换成 bf16 即可拿到「减半字节」的全部好处。本轮范围限定为**非 TMA 的 `cp.async` staging + 基础
GEMM 核**（TMA 需 bf16 张量图与更窄 swizzle；非 GEMM 六段按 fp32 寻址读 staged 操作数，均列为边界）。

### 76.2 Part A：bf16 核在 bf16 操作数上与 fp32 核在 `bf16.float()` 上 **逐位相同**（`max_abs=0`）

四组 shape（`BT=16`，含非整除 `T=250`，torch 参考关 tf32 走真 fp32）：

| 量 | fp32核 vs torch | **bf16核 vs torch(bf16)** | **bf16核 vs fp32核**（吃 `bf16.float()`） |
|---|---|---|---|
| `dq` | 3.583e-4 | 5.154e-7 | **0** |
| `dk` | 3.281e-4 | 4.416e-7 | **0** |
| `dw` | 3.408e-4 | 6.371e-7 | **0** |
| `dgk` | 1.628e-7 | 1.334e-7 | **0** |

（最坏 `max_rel`；四组 shape 全部如此。）三点被钉死：

- **`bf16核 vs fp32核` 全 0**：把 B1 的 bf16 出口直接喂 bf16 核（不做 `.float()`）与「`.float()` 后喂
  fp32 核」**逐位相同**——因为 fp32 核会把 `bf16.float()` 再 `cvt.rna.tf32` 截回同一个 bf16 值，两侧操作数、
  mma、累加序完全一致。故 **bf16 入口是 drop-in 替换**，不引入任何数值差。
- **`bf16核 vs torch(bf16)`（~5e-7）比 `fp32核 vs torch`（~3e-4）还紧**：fp32 核那 3e-4 来自把 **fp32
  输入截到 tf32** 的截断误差；bf16 核的输入本就 ⊂ tf32，**无截断**，只剩 fp32 累加序——bf16 入口反而更贴真 fp32。
- 含非整除 `T=250`（尾块 pad 行走 `OP_ZERO`）。

### 76.3 Part B：kernel-only `dev`——大 shape **0.73×**，shape 越大越受益

CUPTI `dev`/`wall`（µs，kernel-only）：

| shape | fp32 dev | fp32 wall | bf16 dev | bf16 wall | **bf16/fp32 dev** |
|---|---|---|---|---|---|
| `B8 T2048 H8` | 517.5 | 519.9 | **377.4** | 379.2 | **0.729×** |
| `B2 T1024 H8` | 71.0 | 73.1 | **52.3** | 54.5 | **0.736×** |
| `T512 H4` | 9.2 | 29.9 | 8.0 | 29.9 | 0.863× |
| `T250 H3`（非整除） | 8.5 | 30.1 | 7.8 | 30.0 | 0.907× |

收益随 shape 增大（大 shape 是本核的 DRAM-bound regime），小 shape（`wall` 由 host/launch 主导）收益被冲淡。

### 76.4 Part C：静态资源——smem **85.5→47.5 KB（−44%）**、blocks/SM 2→4、零 spill

| | smem | theo blocks/SM | regs | spill |
|---|---|---|---|---|
| fp32 | 87552 B（85.5 KB） | 2 | 70 | 0 |
| **bf16** | **48640 B（47.5 KB）** | **4** | 65 | 0 |

staged 操作数占本核 smem 的绝大部分（`NS=2` stage × 5 操作数），bf16 直接把它们减半。

### 76.5 `ncu`：DRAM 读**精确减半**、不再是 bound；占用率与 tensor 密度随之抬升

`B8 T2048 H8`（同一批操作数，`probe_b2_bf16_ncu.out.txt`）：

| 指标 | base（fp32） | **bf16** | 变化 |
|---|---|---|---|
| `dram__bytes_read` | 1.28 GB | **637.57 MB** | **−50.2%（精确减半）** |
| `dram__bytes_write` | 199.83 MB | 196.02 MB | 基本不变（输出仍是 fp32） |
| `lts__t_bytes`（L2） | 3.03 GB | **1.56 GB** | −48.5% |
| DRAM SOL | 84.72% | **65.96%** | 不再是最紧 pipe |
| L2 SOL | 88.51% | 79.63% | — |
| tensor 密度 | 7.84% | **10.83%** | ↑ |
| achieved 占用率 | 24.62% | **36.84%** | smem 87.5→48.6 KB 放开 block limit |
| `long_scoreboard` | 3.14 | **1.71** | 等 DRAM 的停滞减半 |
| Duration | 519.39 µs | **377.02 µs** | 0.726×（与 Part B 0.729× 一致） |

**归因干净**：五操作数的理论读取 = `(3·B·T·H·V + 2·B·NT·H·K·V)·4 B` = `(201+1073) MB = 1.27 GB`（fp32），
bf16 精确减半到 **637 MB**——`ncu` 实测 1.28 GB→637.57 MB **逐值吻合**，故收益**就是这五个操作数的 DRAM 读**
（以及被它连带减半的 L2 流量），**不是**占用率（占用率是 smem 减半的副产物：DRAM SOL 已降到 66%）。
`t512h4` 同向：DRAM 读 19.94→9.98 MB、L2 48.08→24.23 MB、Duration 12.51→10.30 µs（DRAM 49.1%→28.9%）。

### 76.6 结论与设计输入

- **「bf16 原生入口」在 B2 GEMM 核上兑现，且是免费的**：与 fp32 核在 `bf16.float()` 上**逐位相同**
  （drop-in），大 shape kernel **0.73×**，机制是**五操作数 DRAM 读/L2 精确减半**（不是占用率，占用率是 smem
  减半的副产物）。故 §67.4「`.float()` 往返是部署税」里 **B2 侧那部分可以用 bf16 入口整个消掉**（B1 侧
  `.float()` 是否也删取决于 B1 的 bf16 出口是否直接接 B2——§31/§39 已证 B1 原生即吐 bf16）。
- **设计输入**：目标卡 CUDA B2 的操作数路径在部署 dtype 为 bf16 时应走 **bf16-native staging**；它天然与
  §51/§59/§70 的向量化 store 正交（输出仍 fp32）。
- **边界（诚实）**：本轮只覆盖**非 TMA 的 `cp.async` staging**与**基础 GEMM 核**；TMA（需 bf16 张量图/更窄
  swizzle，或 `SWIZZLE_NONE` 的 128 B 行）、非 GEMM 六段（`-DB2M_OUT/DKGB/DA/DB/GATE/SAND/INTRA` 按 fp32
  寻址读 staged 操作数）**未做**，`-DB2M_BF16` 与它们通过 `#error` 互斥；也**未落进完整反向链**（生产核里
  `bf16 in → 门控/输出仍 fp32 out` 的组合待测）。本核是研究形态，不改变 §45 的 `cf` 默认部署。

## 77. B2 K2-mma GEMM 核的**原生 bf16 mma（`m16n8k16`）**：ISA 换代——指令数减半、tensor 吞吐翻倍，但大 shape 已 L2/DRAM bound、只值 ~1%（P7-74）

P7-73（§76）给 B2 K2-mma GEMM 核加了 **bf16 原生入口**（`-DB2M_BF16`：五操作数按 bf16 staging，DRAM 读
精确减半），但**保留了 tf32 mma**（`mma.sync.m16n8k8.tf32`）——这是合法的，因为 bf16 的 8 位尾数 ⊂ tf32 的
10 位尾数，bf16→fp32→`cvt.rna.tf32` 是恒等，乘积精确、只剩 fp32 累加序差。§76.6 把「bf16 原生 **mma**」
列为触发下一轮的需求，并点出它「可顺带消 §47.6 的 tf32 无 16-bit `ldmatrix` 限制」。本轮兑现这条：给核加
`-DB2M_BF16MMA`（需 `-DB2M_BF16`），把 GEMM 核的 mma 换成 **原生 bf16 `mma.sync.m16n8k16`**。

新增 `harness/check_bwd_b2_bf16mma.py`（`harness/check_bwd_b2_bf16mma.out.txt`）+ `harness/ncu_b2_bf16mma_driver.py`
/ `harness/probe_b2_bf16mma_ncu.sh`（`harness/probe_b2_bf16mma_ncu.out.txt`，空闲卡）。

### 77.1 设计：只换 **mma ISA**，A/B 片段值一字不变；GEMM 核本就不需要 `ldmatrix`

`-DB2M_BF16MMA` 只改 GEMM 核的两处：`mma.m16n8k8.tf32`（k8）→ `mma.m16n8k16.bf16`（k16），主循环
`kk` 步长 8→16。操作数**不重组**、字节数**不变化**——A 片段直接由两个相邻 bf16 打包成一个 32-bit 寄存器
（`rd2`，低位=`col`、高位=`col+1`），得到与 k8 路径**逐值相同**的操作数：

- A（m16k16，lane `l`，`g=l>>2`、`t=l&3`）：`a0={A[g][2t],A[g][2t+1]}`、`a1={A[g+8][2t],..}`、
  `a2={A[g][2t+8],..}`、`a3={A[g+8][2t+8],..}`；
- B（col-major k16n8，本核 B 即 `h[n][k]`，故 `B[k][n]=h[n][k]`）：`b0={B[2t][g],B[2t+1][g]}={h[g][kc+2t],h[g][kc+2t+1]}`、
  `b1={B[2t+8][g],B[2t+9][g]}`；
- C 片段布局与 m16n8k8 **完全相同**，故所有 store 代码一行未动。

**为什么不用 `ldmatrix`**：§47.6 的 tf32 限制（32-bit 片段、无 16-bit `ldmatrix`）在本核**没有可兑现收益**——
GEMM 核的 A/B 片段**直读**在 `OPS=BVK+8`（bf16）行距下**本就无 bank 冲突**：一次 A 片段读的字地址模 32 bank
= `(20g + 8·(kc/16) + t) mod 32`，`g∈0..7`、`t∈0..3` 给出的 `20g+t` 恰好铺满 `0..31`；B 片段读 `krow=const+g`
同构。实测印证：**bf16 与 bmm 两列的 shared-load bank conflict 恒 `8.90M / 8.81M`**（几乎不动），说明 §47 的
8.8M 冲突**不在** GEMM 片段读里（换 ISA 后不变），`ldmatrix` 对本核是多余的。故本轮把「原生 bf16 mma」与
「`ldmatrix`」解耦——只测 ISA 本身的收益。

### 77.2 Part A：与 tf32 路径只差 fp32 累加序（≤2.5e-7），`dgk` 逐位相同——片段布局被钉死

四组 shape（`BT=16`，含非整除 `T=250`，torch 参考关 tf32 走真 fp32），最坏 `max_rel`：

| 量 | **bmm vs torch(bf16)** | **bmm vs bf16(tf32)** | **bmm vs fp32核**（吃 `bf16.float()`） |
|---|---|---|---|
| `dq` | 4.582e-7 | 2.547e-7 | 2.547e-7 |
| `dk` | 3.676e-7 | 2.524e-7 | 2.524e-7 |
| `dw` | 4.779e-7 | 2.409e-7 | 2.409e-7 |
| `dgk` | 1.334e-7 | **0** | **0** |

- **`bmm vs bf16(tf32)` ≤ 2.5e-7 且量级恰为 fp32 累加序**：两侧操作数逐值相同、乘积精确（bf16×bf16 在 fp32
  精确），唯一差别是 k8×2 组 vs k16×1 组的 fp32 加法结合序。**这同时是对 A/B 片段布局的强证据**——若
  `2t / 2t+8` 的行列映射写错，误差会是 O(1)（对照 §65.3/§74 的「指纹」用法）。
- **`dgk` 逐位相同（`max_abs=0`）**：`dgk` 归约走的是 `rd`（bf16 扩 fp32 后点乘），不经 mma，故本次 ISA 改动
  对它零影响——正确。
- **`bmm vs torch(bf16)` ~4.6e-7**，与 §76.2 的 `bf16核 vs torch(bf16)`（~5e-7）同阶：bf16 输入 ⊂ tf32，
  **无截断**，只剩累加序，故比 fp32 核 vs torch（~3e-4，被 tf32 截断主导）更贴真 fp32。

### 77.3 Part B/C：kernel-only `dev` 大 shape 只 **0.99×**、小 shape **0.89–0.91×**；smem/regs 不变

CUPTI `dev`（µs，kernel-only）：

| shape | fp32 | bf16(tf32) | /fp32 | **bmm** | /fp32 | **bmm/bf16** |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 519.1 | 377.4 | 0.727× | **374.5** | 0.721× | **0.992×** |
| `B2 T1024 H8` | 71.5 | 53.0 | 0.741× | **51.7** | 0.723× | **0.976×** |
| `T512 H4` | 9.4 | 8.2 | 0.877× | **7.5** | 0.802× | **0.914×** |
| `T250 H3`（非整除） | 8.7 | 7.9 | 0.909× | **7.1** | 0.808× | **0.888×** |

`wall` 同向（大 shape 0.99×，小 shape 被 host/launch 主导）。静态资源：bf16 与 bmm **完全相同**（smem
48640 B=47.5 KB、regs 65、零 spill）——ISA 换代不动 staging，故不意外。

### 77.4 SASS：`24× HMMA.1688.F32.TF32` → `12× HMMA.16816.F32.BF16`（mma 指令精确减半）

`nvdisasm` 静态计数：bf16 核 **24 条** `HMMA.1688.F32.TF32`（m16n8k8），bmm 核 **12 条**
`HMMA.16816.F32.BF16`（m16n8k16）——每条 k16 指令做两条 k8 的活，指令数**精确减半**，且是真 bf16 HMMA。

### 77.5 `ncu`：指令数减半、tensor 周期**减半**（bf16 mma 吞吐是 tf32 的 2×），但核已 L2 bound

`B8 T2048 H8`（`probe_b2_bf16mma_ncu.out.txt`）：

| 指标 | bf16(tf32) | **bmm** | 变化 |
|---|---|---|---|
| `smsp__inst_executed` | 161.09 M | **90.83 M** | **0.564×** |
| `sm__pipe_tensor_cycles_active` | 10.73% | **5.39%** | **≈减半** |
| Compute (SM) Throughput | 45.27% | **27.42%** | 发压大降 |
| shared-load wavefronts | 40.67 M | **30.11 M** | −26% |
| bank conflict | 8.90 M | 8.81 M | 不动（不在 GEMM 片段读） |
| L2 Cache Throughput | 79.67% | **81.52%** | 仍最高 pipe |
| DRAM Throughput | 66.04% | 66.45% | 不变 |
| DRAM 读/写 | 637.6 / 195.6 MB | 637.6 / 195.4 MB | 不变 |
| Duration | 376.38 µs | **373.95 µs** | **0.993×** |

**关键读数**：`sm__pipe_tensor_cycles_active` **减半**——H100 上 bf16 mma 的吞吐是 tf32 的 **2×**，故同样 FLOP
只需一半 tensor 周期（注意：这个数**不是** FLOP 利用率，bf16 下它天然比 tf32 低一半，别误读成「tensor 更闲」）。
指令数也减半。但 **Duration 几乎不动（0.993×）**：核的最紧 pipe 是 **L2（79.7→81.5%）**，而 tensor 只占
10.7%、DRAM 66%。tf32→bf16 省下的**全是发射/tensor 槽位**，而这个核本就不缺发射槽——故大 shape 换 ISA
换不来时间。`t512h4` 则不 bound 于 L2（39.96%），发压/指令红利得以兑现：Duration 10.59→**9.76 µs（0.92×）**、
inst 2.52M→**1.42M**、tensor 7.97→4.50。

### 77.6 结论与设计输入

- **原生 bf16 mma 是正确的、drop-in 的 ISA 换代**：与 tf32 路径只差 fp32 累加序（≤2.5e-7）、`dgk` 逐位；与
  §76 的 bf16 staging **正交**（smem/regs 不变）。SASS 由 `HMMA.1688.F32.TF32` 变 `HMMA.16816.F32.BF16`、
  mma 指令精确减半，tensor 周期减半（bf16 吞吐 2× tf32）。
- **但它不是大 shape 的杠杆**：P7-73 的 bf16 核已把核推到 **L2/DRAM bound**（L2 80%、DRAM 66%），tensor 只
  10.7%，故 ISA 换代大 shape 只 **0.99×**。收益落在**存储不紧、发压主导**的小 shape（≤512 token）——
  **0.89–0.91×**（8–11%）。
- **`ldmatrix` 对本核是多余的（本轮副产物）**：GEMM 核的 A/B 片段直读在 `OPS=BVK+8` 行距下 32 lane 全落不同
  bank（`20g+t` 覆盖 `0..31`），换 ISA 前后 bank conflict 恒 8.8M——§47.6 的 tf32 无-16bit-`ldmatrix` 限制
  在本核**没有可兑现的成本**，8.8M 冲突在别处（dgk 归约）。故 §76 的「bf16 原生 mma 可顺带消 §47.6 限制」
  **被证伪为无收益**（此核不因缺 `ldmatrix` 而冲突）。
- **设计输入**：目标卡 CUDA B2 在部署 dtype 为 bf16 时，应走 **bf16-native staging（§76）× 原生 bf16 `m16n8k16`
  mma**——两者零成本叠加；但若核已 L2/DRAM bound（大 shape），应把优化预算投给**降 L2/DRAM 字节**（TMA/向量化
  store，§52/§73）而非 mma ISA。小 batch/短序列（launch/发压受限）则 ISA 红利可见。
- **边界（诚实）**：只覆盖**非 TMA 的 `cp.async` staging + 基础 GEMM 核**（`-DB2M_BF16` 的同一范围，
  `#error` 与 TMA/六段互斥）；`BVK∈{32,64}`（`static_assert BVK%16==0`）；输出仍 fp32；**未落进完整反向链**。
  本核是研究形态，**不改变 §45 的 `cf` 默认部署**。

## 78. 把 bf16 原生 staging 从「基础 GEMM 核」扩到**六段生产核**，并落进完整反向链——逐位无损、链 0.70–0.93×、DRAM 读 −0.56 GB（P7-75）

P7-73（§76）给 B2 K2-mma **基础 GEMM 核**加了 bf16 原生入口（`-DB2M_BF16`：`do/vn/dv/h/dh` 五操作数按
bf16 staging，DRAM 读精确减半），但用 `#error` 把它锁在「非 TMA 的基础 GEMM 核」内，**明确排除六段生产核
（`-DB2M_PROD`）与整链**——于是 §67.4 记下的部署税（链里必须把 B1 的 bf16 `dh`/`dv2`、workspace 的 bf16
`h`/`v_new`/`do` 先 `.float()` 才能喂 fp32/tf32 的 CUDA B2）仍然存在。P7-74 又给 GEMM 核换了原生
`mma.m16n8k16.bf16`。本轮兑现 §76.6/§77.6 点名的「**bf16 in 的生产核落链**」：把 `-DB2M_BF16` 放开到六段
生产核，并把它端到端接进反向链。

新增 `harness/check_bwd_b2_bf16_prod.py`（`harness/check_bwd_b2_bf16_prod.out.txt`）+
`harness/ncu_b2_bf16_prod_driver.py` / `harness/probe_b2_bf16_prod_ncu.sh`
（`harness/probe_b2_bf16_prod_ncu.out.txt`）+ `harness/check_bwd_b2_bf16_e2e.py`
（`harness/check_bwd_b2_bf16_e2e.out.txt`，空闲卡）。

### 78.1 设计：`bf16 ⊂ tf32` 让「扩到六段」也是 drop-in；关键是分清**哪些操作数永远是 fp32**

放开 P7-73 的 `#error` 只需解决两件事——六段读操作数的路径必须**按 dtype 分流**：

1. **五个 streamed 操作数**（`do/vn/dv/h/dh`）仍走 `rd()`：它在 `-DB2M_BF16` 下做 `__bfloat162float`
   （精确扩宽），六段（OUT 读 `dv`、DA 读 `dv`、DGK 归约读 `h/dh`）读到的都是**同一个 fp32 值**——和 fp32 核
   喂 `bf16.float()` 完全一致。
2. **fp32 scratch**（`vv`＝raw `v`、`dwm`＝`dw`）**永远是 fp32**。此前它们也用 `rd((const float*)st->vv,…)`
   读——在 fp32 核里 `rd` 就是按 fp32 读，但在 bf16 核里 `rd` 会**把 fp32 当 bf16 反量化**（错）。故新增
   `rds()`：非 TMA 下按 fp32 padded 行距（`BVP=BVK+4`）读、TMA 下退回 `rd` 的 atom 解码；把 `vv`/`dwm` 的 6
   处读全换成 `rds()`（`b2_mma_probe.cu` §DA/§DKGB/§DB 段）。
3. **prefetch 的粒度分流**：主循环按 `op_t`（bf16 时 `U=8`）搬 `do/vn/dv/h/dh`；`vv`/`dwm` 从主循环**拆出**、
   单独按 **fp32 的 `U4=4`** 搬（`b2_mma_probe.cu` 的 `prefetch`）。fp32 核里 `U4==U==4`，行为不变。

`Stage` 里 `vv`/`dwm` 本就是 `float[BT][BVP]`（与 `op_t` 无关），所以结构上无需改布局。放开范围后唯一保留的
`#error` 是 `-DB2M_BF16 && -DB2M_TMA`（TMA 需要 bf16 张量图 + 128 B 宽的 swizzle atom，未做）。

### 78.2 Part A：`prod_bf16` 与 fp32 `prod` **逐位相同**（11 项输出全 `max_abs=0`）

四组 shape（含非整除 `T=250`）上，把五操作数量化成 bf16（`bf16(x).float()` 喂 fp32 核、`bf16(x)` 喂 bf16 核），
`prod` 的 11 项输出（`dv2/dvb/dkgb/db/dApre/dA/dqg/dkg/dg/dw/dgk`）**全部逐位相同**：

| 量 | `prod_bf16` vs `prod`（fp32） |
|---|---|
| 11 项输出（四组 shape） | **max_abs = 0（BIT-IDENTICAL）** |

- **为什么必然逐位**：bf16 的 8 位尾数 ⊂ tf32 的 10 位尾数，bf16 值在 fp32 里精确、`cvt.rna.tf32` 恒等；五操作数
  两侧逐值相同，六段又都经 `rd()` 拿到同一 fp32 值，故整核只差**没有任何差别**（连累加序都不变——GEMM 仍走
  tf32 mma）。
- **`-DB2M_BF16MMA` 叠加**：只差 GEMM 核的 fp32 累加序——直通输出（`dv2/dvb/db`）**逐位**（OUT/DB 不走 GEMM
  mma 或走 tf32 未变），`dw/dgk/dqg/dkg/dg` 与 `dApre/dA` ≤ **9.5e-7**（`dgk` 仍逐位 0，同 §77.2）。
- **torch sanity**（Part D）：`prod_bf16` 的 `dv2 = beta·(A@dv)` vs torch **6.6e-6–9.9e-6**（OUT 的 tf32 mma
  底噪，因 dv 已是 bf16 值、幅度小）。

### 78.3 Part B/C：regs 146→128、smem −29%、kernel-only 大 shape **0.68×**

| build | regs | spill(B) | smem(B) |
|---|---|---|---|
| `prod`（fp32） | 146 | 0 | 133760 |
| `prod_bf16` | 128 | 4 | **94848** |
| `prod_bf16_mma` | 110 | 0 | **94848** |

smem **133760→94848（−29%）**：五操作数的 stage 数组由 fp32 变 bf16（近乎减半），六段的 scratch 不变。CUPTI
`dev`（kernel-only，单次 `b2prod` launch）：

| shape | fp32 | `bf16` | /fp32 | `bf16mma` | /fp32 |
|---|---|---|---|---|---|
| `B8 T2048 H8` | 1299.3 µs | **888.2** | **0.684×** | **856.6** | **0.659×** |
| `B2 T1024 H8` | 171.4 | **124.3** | **0.725×** | **120.6** | 0.704× |
| `T512 H4` | 19.7 | 19.0 | 0.966× | 18.5 | 0.938× |
| `T250 H3`（非整除） | 18.6 | 18.7 | 1.007× | 18.3 | 0.986× |

大 shape **0.68×**、小 shape（launch 主导）±1% 内。注意这比 §76 基础 GEMM 核的 0.73× **还多**——原因见 §78.4。

### 78.4 `ncu`：DRAM 读 −0.56 GB、L2 −29%**且占用率翻倍（1→2 block/SM）**

`B8 T2048 H8`：

| 指标 | fp32 | **bf16** | 变化 |
|---|---|---|---|
| `dram__bytes_read` | 1.63 GB | **1.07 GB** | **−0.56 GB（−34%）** |
| `dram__bytes_write` | 485.7 MB | 487.2 MB | 不变 |
| `lts__t_bytes`（L2） | 4.22 GB | **3.01 GB** | **−29%** |
| 全局读 sector | 77.73 M | **57.80 M** | −26% |
| bank conflict / shared wavefronts | 23.54 M / 70.27 M | 24.87 M / 71.73 M | 不动（六段 smem 流量不变） |
| DRAM Throughput | 49.06% | 52.80% | — |
| L2 Throughput | 60.78% | **70.53%** | 最紧 pipe |
| tensor cycles | 3.32% | 4.85% | — |
| **achieved occupancy** | **12.46%** | **24.64%** | **翻倍** |
| `launch__occupancy_limit_shared_mem` | 1 | **2** | smem 减半生效 |
| `gpu__time_duration` | 1290 µs | **879.6 µs** | **0.682×**（与 CUPTI 互证） |

`B2 T1024 H8` 同向：读 203.6→133.8 MB、L2 530.6→377.0 MB、duration 170.9→122.6 µs、occ 12.46→23.56%。

**关键（修正 §76.4 的归因读法）**：§76 的**基础 GEMM 核** bf16 收益被归因为「五操作数的 DRAM 读/L2，占用率
是 smem 减半的副产物」。到**六段生产核**这里，核**本就不 DRAM bound**（DRAM 49–53% SOL、tensor 3–5%、L2 最紧
仅 ~61–70%）——所以 0.68× 不是「省带宽」，而是 **省读字节 + 占用率翻倍（1→2 block/SM，为延迟/发射受限的六段
核多一倍 active warp）** 两者叠加。这与 §67/§68 把生产核判为「发射受限、非 DRAM bound」自洽：bf16 staging 恰
把该核缺的**并行度/延迟隐藏**补了一档，故收益比基础 GEMM 核更大。六段的 smem 流量（bank conflict/wavefronts）
**完全不动**——它们读的是 fp32 scratch 与 `[B,T,H,K]` 输出写，不在 bf16 范围内。

### 78.5 Part B（链级）：`pcb` 数值无损、`dev` 0.70–0.93× fp32 链、链 `wall` −29%

把 bf16 生产核落进完整反向链 `dAv(Triton)→CUDA-B1→CUDA-B2-prod[-DB2M_BF16]→[Triton B3 intra]→反向 cumsum`
（只吃 FlashKDA 的 16-chunk K1 workspace），五操作数**直接喂 bf16**（不再 `.float()`）：变体 `pc`（fp32 prod）、
`pcb`（bf16 prod + Triton intra）、`pcib`（bf16 单 launch 折叠），对照 `cf`（§45 最优：CUDA-B1 + 融合 Triton
B2/B3）与 FLA autograd。

**Part A（六项梯度 vs FLA autograd 最坏 `max_rel`）**：`pcb` **9.365e-3**、`pcib` **9.365e-3**——与 §67 的 `pc`
（9.4e-3）**同值**，全落 bf16/tf32 底噪；`pcb vs pc`（fp32 链）≤ **2.537e-3**、`pcb vs pci` ≤ **6.094e-3**，差
来自 `do/v_new/h` 在 bf16 链里被量化（**不是 bug**，`dv` 项 `pcb vs pc` **逐位 0**——OUT 的 tf32 mma 对已 bf16
的 dv 与 fp32 链一致）。

**Part B（链级 `dev`/`wall`，空闲卡）**：

| case | FLA dev | `cf` dev | `pc` dev | **`pcb`** | `pcib` | `pcb/FLA` | `pcb/pc` | wall pc/pcb/pcib |
|---|---|---|---|---|---|---|---|---|
| `T512 H4` | 204.8 | 75.9 | 116.1 | **104.8** | 84.0 | 0.51 | 0.90 | 465/444/322 µs |
| `B2 T1024 H8` | 661.0 | 257.1 | 463.0 | **352.0** | 326.2 | 0.53 | 0.76 | 495/458/345 µs |
| `B8 T2048 H8` | 4150.6 | 1409.6 | 2905.4 | **2043.7** | 1973.5 | 0.49 | 0.70 | 2928/2090/1992 µs |
| `T250 H3`（非整除） | 141.6 | 50.2 | 83.5 | **77.4** | 58.5 | 0.55 | 0.93 | 463/445/321 µs |

- 链 `dev` **`pcb/pc` 0.70–0.93×**（大 shape 0.70×），`pcb/FLA` **0.49–0.55**（vs `pc/FLA` 0.57–0.70）；
  链 `wall` 更明显（B8 2928→2090 µs）：fp32 链每次对 `dh/dv2/v_new/h/do` 做 `.float()` 的 elementwise 税被整体消掉。
- `pcib`（bf16 折叠单 launch）dev 再低一档（B8 1973 vs 2043；`T512` 84 vs 105），延续 §66 的「小/中 shape 折叠
  净赢、大 shape 略被串行尾吃掉」结论。
- **但 `cf` 仍更快**（B8 `cf` 1409.6 vs `pcb` 2043.7）——CUDA B2 生产核相对融合 Triton B2/B3 仍慢 ~1.45×，bf16
  staging 只是把它从 2.06×（`pc/cf`=2905/1409）收到 1.45×。**故默认部署仍取 §45 的 `cf`**；bf16 CUDA 核是
  「当目标卡选择 CUDA B2 时，部署 dtype 端到端原生」的收口，不改变默认路线。

### 78.6 结论与设计输入

- **`-DB2M_BF16` 已能覆盖六段生产核**：与 fp32 生产核在 bf16 值上**逐位相同**（11 项输出全 0），`-DB2M_BF16MMA`
  可选叠加（只差累加序 ≤9.5e-7）。§67.4 的部署税（`.float()` 往返）在**链级整个消掉**。
- **收益＝读字节 + 占用率**：六段生产核**不 DRAM bound**（49–53% SOL），故 0.68× 的大头是 **smem 减半把
  占用率从 1 提到 2 block/SM**（补该核缺的延迟隐藏），其次才是五操作数少读 0.56 GB。**这修正了 §76.4 把它单纯
  归因为字节的口径**——同一旋钮在「基础 GEMM 核（DRAM bound）」与「六段生产核（发射受限）」上机制不同。
- **设计输入**：目标卡 CUDA B2 部署 dtype 为 bf16 时，**`bf16 staging × 六段生产 launch` 零成本叠加**（`rds`
  分流 fp32 scratch、prefetch 按 `op_t`/fp32 双粒度搬）；生产核里 bf16 的下一杠杆应继续看 **smem 峰值/占用率**
  （而非带宽），与 §20/§22 的「占用率不是旋钮」在**六段核上是有旋钮**形成对照。
- **边界（诚实）**：`-DB2M_BF16` 仍只覆盖**非 TMA** 的 `cp.async` staging（`-DB2M_TMA` 需 bf16 张量图，未做）；
  `vv`/`dwm` 保持 fp32（本轮没有把 raw `v`/`dw` 也降 bf16）；输出仍 fp32；仅 `K=V=128/BT=16`。`pcb` 仍慢于
  `cf`，**不改变 §45 的默认部署**。

## 79. 关闭 bf16 原生入口的**最后一个 `#error`**：`-DB2M_BF16 × -DB2M_TMA` 正确可组合（逐位无损），但 TMA 在 bf16 核上是 **wash**——省下的 L2 字节落在不最紧的 pipe（P7-76）

§76（P7-73）给 B2 K2-mma **基础 GEMM 核**加了 bf16 原生入口（`-DB2M_BF16`：五操作数 `do/vn/dv/h/dh`
按 bf16 staging），§77（P7-74）又加了原生 `mma.m16n8k16.bf16`；但两条路线各自的 `#error` 一直把
`-DB2M_BF16` 与 `-DB2M_TMA`（P7-49 的 TMA 输入）**锁成互斥**——「TMA 需要 bf16 张量图 / 128 B 宽的
swizzle atom」（§78.1/§78.6 边界）。本轮把这两条路线**拼起来**，并诚实给结论：**组合正确（逐位），但
TMA 在 bf16 核上不再兑现它当年在 fp32 核上的读侧红利**。

新增 `harness/check_bwd_b2_bf16_tma.py`（`harness/check_bwd_b2_bf16_tma.out.txt`）+ `harness/ncu_b2_bf16_tma_driver.py`
/ `harness/probe_b2_bf16_tma_ncu.sh`（`harness/probe_b2_bf16_tma_ncu.out.txt`，空闲卡）。

### 79.1 设计：bf16 的 128 B swizzle atom 是 **64 个元素**，解码公式**一字不变**

TMA 的 `SWIZZLE_128B` 按 **128 字节**为单位做分布式 swizzle。fp32 时一个 128 B atom = **32 个元素**
（`NAT=BVK/32`，P7-49/§52）；bf16 时同样的 128 B = **64 个元素**（`NAT=BVK/64`，故 bf16 TMA 要求
`BVK∈{64,128}`，部署默认的 `BVK=32` 不是合法 bf16 atom 数）。关键观察是**解码公式不需要为 bf16 重写**：
`rd()` 里那条 `b ^= ((b>>7)&7)<<4` 作用在**字节地址**上，而 128 B 的行距无论 fp32/bf16 都把 `row`
搬进 bit 7，因此 swizzle 的语义（行内 16 B 块 ↔ 行号的低 3 位）完全一致，只差「一个 16 B 块是 4 个
fp32 还是 8 个 bf16」。改动因此很小（`b2_mma_probe.cu`）：

1. **常量**：`#if B2M_TMA_ON` 下引入 `tma_t`/`TAW`——bf16 为 `__nv_bfloat16`/64、否则 `float`/32，
   `NAT=BVK/TAW`，并 `static_assert` bf16 的 `BVK∈{64,128}`。
2. **`rd()`**：`atom=col/TAW, cc=col%TAW, b=((atom*rows+row)*TAW+cc)*sizeof(tma_t)`，swizzle 后按
   `tma_t` 取回（bf16 走 `__bfloat162float`）。fp32 时 `TAW=32`、`sizeof=4`，与旧式逐字节相同。
3. **`issue_tma`**：`expect_tx` 的字节数由 `BVK*4` 改 `BVK*sizeof(tma_t)`（bf16 减半），atom 循环里
   `va = v0 + a*TAW`（旧代码硬编码 `a*32`）。
4. **张量图**：`make_map_dov`/`make_map_hdh` 的 datatype / 元素大小 / box 内宽全部走 `TMA_DT`/`TMA_ESZ`/
   `TMA_INW`（bf16→`CU_TENSOR_MAP_DATA_TYPE_BFLOAT16`/2/64）。
5. **fp32 scratch（`vv`/`dwm`）在 bf16+TMA 下也走 bf16**：一张 bf16 图即可覆盖所有操作数，避免为
   `vv`/`dwm` 再建 fp32 atom 图（raw `v` 上线本就是 bf16；`dw` 是被 DKGB 的 tf32 mma 消费的中间量）。
   `rds()` 在 TMA 下本就委托 `rd()`，故无需改。

尚未组合的只剩 `-DB2M_BF16MMA × -DB2M_TMA`（`rd2()` 无 atom 解码），保留 `#error`。**算术一字未动。**

### 79.2 Part A：`bftma64` 与 `bf16_64` **逐位相同**（含非整除 `T=250`）

四组 shape（`B8 T2048 H8` / `B2 T1024 H8` / `T512 H4` / `T250 H3`）上，把两条 bf16 build 喂**同一份**
bf16 操作数，并把两条 fp32 build 喂同一份 fp32 操作数（全在 `BVK=64`）：

| 对比 | `dq` | `dk` | `dw` | `dgk` |
|---|---|---|---|---|
| `bftma64` vs `bf16_64` | **0** | **0** | **0** | **0** |
| `tma64` vs `base64`（fp32 对照） | **0** | **0** | **0** | 2.861e-6 |

- **`bftma64 ≡ bf16_64` 逐位**：两者 stage 的 bf16 值、mma（`m16n8k8.tf32`）的 k 序完全相同，TMA 只换
  smem 布局，故连累加序都不变——这同时把「bf16 TMA 的布局解码正确」在**五操作数 × 四 shape**上钉死。
- **`tma64 ≡ base64` 的 `dq/dk/dw` 也逐位**（`dgk` 差 2.861e-6＝`dgk` 列归约的 fp32 分组，同 §53.2 的
  `base32==base64` 指纹）；这是 §52 在 `BVK=32` 之外对 fp32 TMA 的一次旁证。
- vs torch（`allow_tf32=False`）：`bftma64` 落 **≤6.4e-7**（bf16⊂tf32、无 tf32 截断，比 fp32 核的
  `tma64/torch ≤3.6e-4` 还紧 3 个量级）；`tma64/torch ≤3.6e-4`＝tf32 底噪。

### 79.3 Part B/C：`bftma64/bf16_64` **0.97–1.03×（wash）**，`tma64/base64` 仍 0.85–0.97×

CUPTI `dev`（kernel-only，`b2mma_launch` 单次）：

| shape | `base64` | `tma64` | `bf16_64` | `bftma64` | `tma64/base` | `bftma/bf16` |
|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 519.9 | 484.7 | 308.4 | 317.3 | 0.932 | **1.029** |
| `B2 T1024 H8` | 71.5 | 69.0 | 47.9 | 46.6 | 0.966 | **0.973** |
| `T512 H4` | 9.3 | 8.2 | 8.4 | 8.6 | 0.886 | 1.028 |
| `T250 H3`（非整除） | 8.6 | 7.4 | 7.5 | 7.7 | 0.852 | 1.024 |

静态（Part C）：

| build | smem | regs | spill |
|---|---|---|---|
| `base64`（fp32 cp.async） | 87552 B（85.5 KB） | 70 | 0 |
| `tma64`（fp32 TMA，NAT=2） | 155776 B（152.1 KB） | 123 | 0 |
| `bf16_64`（bf16 cp.async） | 87552 B（85.5 KB） | 65 | 0 |
| `bftma64`（bf16 TMA，NAT=1） | **77952 B（76.1 KB）** | 125 | 0 |

**fp32 上 TMA 仍净赢**（`tma64/base64` **0.85–0.97×**，与 §52 的 `tma/base` 0.93–0.96 自洽）；**bf16 上
TMA 是 wash**（`bftma64/bf16_64` **0.97–1.03×**）。bf16 TMA 的 smem 反比 bf16 cp.async 小 11%（76.1 vs
85.5 KB，因 fp32 padding 换成了 64 元素稠密 atom），但两者占用率都仍是 **2 block/SM**，故 smem 的这点缩减
不兑现时间。regs 翻倍（65→125，TMA 的 mbarrier/描述符状态），不影响占用（仍 12 warp 上限由 smem 定）。

### 79.4 `ncu`：TMA 仍消掉 1.5× L2 读放大，**但 bf16 核已不 L2-bound**

`B8 T2048 H8`：

| 指标 | `base64` | `tma64` | `bf16_64` | `bftma64` |
|---|---|---|---|---|
| `dram__bytes_read` | 1.28 GB | 1.28 GB | **637.6 MB** | **637.6 MB** |
| `lts__t_bytes`（L2） | 2.97 GB | **2.40 GB** | 1.70 GB | **1.39 GB** |
| `lts__t_sectors_srcunit_tex_op_read` | 59.77 M | **39.85 M** | 29.88 M | **19.92 M** |
| `dram__throughput`（SOL） | 88.6% | **90.7%** | 79.6% | 77.9% |
| tensor cycles | 8.32% | 8.47% | **13.16%** | 12.77% |
| achieved occupancy | 12.54% | 12.48% | **24.37%** | 24.39% |
| `Duration` | 497.0 µs | **485.7 µs** | 312.7 µs | **319.3 µs** |

- **TMA 的读侧修复对 bf16 仍生效**：`srcunit` 读扇区 **29.88 M→19.92 M（×0.667）**、L2 **1.70→1.39 GB
  （−18%）**——即 `cp.async` 那条 `.BYPASS` 的 **1.5× L2 读放大在 bf16 下同样存在、同样被 TMA 消掉**
  （bf16 的 `srcunit/DRAM` 比 29.88 M / (637.6 MB/32 B=19.92 M) 仍是 1.5×，TMA 后正好 =1.0×）。DRAM 读两侧
  都 637.6 MB 不变。
- **但不兑现时间**：`Duration` 312.7→**319.3 µs（+2%）**。原因＝**bf16 核的最紧 pipe 不再是 L2**：DRAM
  SOL 掉到 **79.6%**、tensor 反升到 13.2%、占用率翻倍到 24.4%——核已从 fp32 时的「L2/DRAM 双紧」变成
  「延迟/发射主导」，TMA 省下的 L2 字节**不落在最紧的 pipe 上**（判据同 §51.6/§61.4：省字节要落最紧 pipe）。
  对比 `base64→tma64`：fp32 时 L2 2.97→2.40 GB 恰是能兑现的那一档（`base64` 的最紧 pipe 仍是 L2/DRAM 双 88–89%），
  故 fp32 上 TMA 净赢、bf16 上归零。
- `t512h4` 同向：`bf16_64` 11.01 µs vs `bftma64` 11.68 µs，L2 28.27→24.52 MB（读放大同样消掉，同样不兑现）。

### 79.5 结论与设计输入

- **组合正确**：`-DB2M_BF16 × -DB2M_TMA` 是可编译、且与 `bf16_64`（cp.async）**逐位相同**的 drop-in
  （五操作数、四 shape、含非整除 `T`）；bf16 的 128 B swizzle atom 只是「64 个元素」，解码公式无需改。
  bf16 原生入口**至此不再有 `#error`**（只剩 `BF16MMA × TMA` 未组合）。
- **但 TMA 在 bf16 核上不是杠杆**：fp32 上 `tma64/base64` 净赢 0.85–0.97×，bf16 上 `bftma64/bf16_64`
  **0.97–1.03×（wash）**。机制是**明确的**：TMA 依旧把 `cp.async .BYPASS` 的 1.5× L2 读放大消掉
  （`srcunit` −33%、L2 −18%），但 **bf16 staging 已把核从「L2/DRAM bound」推成「延迟/发射主导」**
  （DRAM 88.6→79.6%、tensor 8.3→13.2%、occ 12.5→24.4%），省下的字节落在不最紧的 pipe。
- **设计输入**：目标卡 CUDA B2 在部署 dtype 为 bf16 时，**操作数走 bf16 staging 即可，TMA 输入非必需**
  （若已为 fp32 留了 TMA 代码则零成本可用、逐位无损；但**不要指望它在 bf16 下再给时间**）——与 §67/§70
  「同一旋钮在孤立核与整核可反号」「省字节要落最紧 pipe」一脉相承。TMA 真正的杠杆场景是 **fp32/tf32 的
  DRAM/L2-bound 核**（§52 的孤立 GEMM 核）。
- **边界（诚实）**：只做**基础 GEMM 核**（非六段生产核）、`BVK=64`、`BT=16`、`K=V=128`、非 `BF16MMA`；
  bf16+TMA 下 `vv`/`dwm` 被降为 bf16（生产核组合时 raw `v`/`dw` 的 bf16 精度未单独消融）；链级未测
  （基础 GEMM 核不落链，机制与 §78.5 一致）。**不改变 §45 的 `cf` 默认部署。**

## 80. 关闭 bf16 原生入口的**最后一个 `#error`**（续）：`-DB2M_BF16MMA × -DB2M_TMA` 也可组合——逐位无损、且 TMA 在**原生 bf16 mma** 核上从 wash 变**小幅净赢**（P7-77）

§79（P7-76）把 `-DB2M_BF16` 与 `-DB2M_TMA` 拼起来后，bf16 原生入口只剩一个 `#error`：**原生 bf16 mma
（`-DB2M_BF16MMA`，§77/P7-74）与 TMA 输入（`-DB2M_TMA`）互斥**——因为 `rd2()`（把两个相邻 bf16 打包成
一个 32-bit mma 片段的读者）只认识非 TMA 的 `OPS` padded 行距布局，没有 atom swizzle 解码。本轮把
`rd2()` 泛化到与 `rd()` 同一套 `[…][TAW]` 128 B swizzle 解码，**关闭这最后一个 `#error`**，并给结论：
组合**逐位无损**，且 TMA 在原生 bf16 mma 核上不再像在 tf32 mma 的 bf16 核上那样是 wash——大 shape **小幅净赢**。

新增 `harness/check_bwd_b2_bf16mma_tma.py`（`harness/check_bwd_b2_bf16mma_tma.out.txt`）+
`harness/ncu_b2_bf16mma_tma_driver.py` / `harness/probe_b2_bf16mma_tma_ncu.sh`
（`harness/probe_b2_bf16mma_tma_ncu.out.txt`，空闲卡）。

### 80.1 设计：`rd2()` 复用 `rd()` 的 atom 解码，**算术一字未动**

k16 bf16 片段的 A/B 列偏移恒为**偶数**（`kc+2t` / `kc+8+2t`，见 §77.1 的片段布局）：一个 bf16 对的
两个元素 `col` / `col+1` 因此**必落在同一个 128 B atom、同一个 16 B swizzle 单元**内。故解码只需算一次
`col` 的**字节地址**（与 `rd()` 完全相同），再连读两个 bf16 即可。改动很小（`b2_mma_probe.cu`）：

1. **`rd2()` 签名**：`const __nv_bfloat16* buf, int row, int col` → `const void* buf, int row, int col,
   int rows`（对齐 `rd()`；`rows` 对 A 操作数是 `BT`、对 B 操作数是 `KD`，`b2_mma_probe.cu:631`）。
2. **TMA 分支**（`b2_mma_probe.cu:628-641`）：`atom=col/TAW, cc=col%TAW,
   b=((atom*rows+row)*TAW+cc)*sizeof(tma_t)`，套 `b ^= ((b>>7)&7)<<4`，再在 `(char*)buf+b` 与 `+sizeof(tma_t)`
   处取两个 bf16；非 TMA 分支仍走 `buf[row*OPS+col]`（原样）。
3. **调用点**（`b2_mma_probe.cu:1025-1043`）：12 个 A 片段调用补 `BT`、4 个 B 片段（`h`/`dh`）补 `KD`。
4. 删除 `b2_mma_probe.cu:428-430` 的 `#if B2M_BF16MMA && B2M_TMA_ON #error …`。

**算术一字未动**：stage 的 bf16 字节、mma 指令、k 序、片段值都与非 TMA 的 `bmm64` 完全相同，TMA 只换
smem 布局——故预期 `bmm_tma64 ≡ bmm64` **逐位**（下面 Part A 证实）。

### 80.2 Part A：`bmm_tma64` 与 `bmm64` **逐位相同**（含非整除 `T=250`）

四组 shape（`B8 T2048 H8` / `B2 T1024 H8` / `T512 H4` / `T250 H3`），同一份 bf16 操作数，全在 `BVK=64`；
bt16 四个 build 只差 `-DB2M_BF16MMA` / `-DB2M_TMA` 两个旋钮：

| 对比 | `dq` | `dk` | `dw` | `dgk` |
|---|---|---|---|---|
| `bmm_tma64` vs `bmm64`（TMA 只换布局） | **0** | **0** | **0** | **0** |
| `bmm64` vs `bf16_64`（换 mma ISA） | 7.63e-6 | 7.63e-6 | 7.15e-7 | **0** |

- **`bmm_tma64 ≡ bmm64` 逐位**（`max_abs=0`，四组 shape 全列四项）：`rd2` 的 atom 解码正确，且证明
  TMA 只改 smem 布局、连**累加序都不动**（同一 ISA、同一 k 序）——这是 §79.2 的等式在**原生 bf16 mma**
  上的延续，把「bf16 TMA 布局解码」在**原生 bf16 片段**上再钉死一次。
- `bmm64` vs `bf16_64` 的差异正是**换 mma ISA 的 fp32 累加序差异**（k16 vs k8 分组，同 §77.2 的
  `dq 2.55e-7/dk 2.52e-7/dw 2.41e-7` 量级；`dgk` 逐位 0 因它不经 K 累加）。
- vs torch（`allow_tf32=False`）：`bmm_tma64` 落 **≤4.8e-7**（bf16⊂tf32、无 tf32 截断，与 §79.2 同档）。

### 80.3 Part B/C：`bmm_tma64/bmm64` **0.96–1.04×**——大 shape 净赢、小 shape wash

CUPTI `dev`（kernel-only，`b2mma_launch` 单次）：

| shape | `bf16_64` | `bmm64` | `bf16_tma64` | `bmm_tma64` | `bmm/bf16` | `bmmtma/bmm` | `tma/bf16` |
|---|---|---|---|---|---|---|---|
| `B8 T2048 H8` | 308.3 | 292.1 | 316.9 | **280.7** | 0.947 | **0.961** | 1.028 |
| `B2 T1024 H8` | 48.0 | 43.2 | 46.5 | **41.5** | 0.900 | **0.960** | 0.967 |
| `T512 H4` | 8.4 | 6.9 | 8.7 | 6.9 | 0.815 | 1.008 | 1.029 |
| `T250 H3`（非整除） | 7.5 | 6.1 | 7.7 | 6.3 | 0.805 | 1.039 | 1.024 |

静态（Part C）：

| build | smem | regs | spill |
|---|---|---|---|
| `bf16_64`（bf16 cp.async，tf32 mma） | 87552 B（85.5 KB） | 65 | 0 |
| `bmm64`（bf16 cp.async，bf16 mma） | 87552 B（85.5 KB） | 73 | 0 |
| `bf16_tma64`（bf16 TMA，tf32 mma） | 77952 B（76.1 KB） | 125 | 0 |
| `bmm_tma64`（bf16 TMA，bf16 mma） | **77952 B（76.1 KB）** | **66** | 0 |

- **换 bf16 mma 仍兑现**（`bmm/bf16` **0.81–0.95×**，与 §77.3 同向：大 shape ~5%、小 shape 8–19%）。
- **TMA 在原生 bf16 mma 核上从 wash 变为大 shape 净赢**：`bmmtma/bmm` 大 shape **0.96×**，而
  §79.3 的 tf32-mma bf16 核里同旋钮是 **0.97–1.03×（wash）**；小 shape（`T512/T250`）仍 **1.01–1.04×**（launch
  主导，TMA 无空间）。也就是说，**换到原生 bf16 mma 后，TMA 又变得值得开**（在基础 GEMM 核上）。
- `bmm_tma64` 的 regs **66**（vs `bmm64` 73、`bf16_tma64` 125）：原生 bf16 mma 的片段更少、TMA 描述符
  状态也不再被 tf32 的双 k8 步放大——静态成本不升反降。smem 与 `bf16_tma64` 相同（同一 atom 布局）。
- 本轮新增的 `#error` 组合解禁**没有引入任何静态代价**（无 spill）。

### 80.4 `ncu`：TMA 对**原生 bf16 mma** 仍消掉 1.5× L2 读放大，且这次**兑现成时间**

`B8 T2048 H8`：

| 指标 | `bmm64` | `bmm_tma64` |
|---|---|---|
| `dram__bytes_read` | 637.6 MB | 637.6 MB |
| `lts__t_bytes`（L2） | 1.68 GB | **1.44 GB（−14%）** |
| `lts__t_sectors_srcunit_tex_op_read` | 29.88 M | **19.92 M（×0.667）** |
| `dram__bytes_write` | 195.7 MB | 195.2 MB |
| `l1tex` shared-load bank conflict | 8.638 M | 8.615 M（不动） |
| `long_scoreboard` | 3.60 | 3.92 |
| `short_scoreboard` | 1.27 | **0.38** |
| `wait`（固定延迟） | 1.51 | **1.10** |
| `inst_executed` | 80.8 M | 113.9 M |
| tensor cycles | 7.03% | 7.34% |
| DRAM SOL | 85.66% | 88.85% |
| L2 SOL | 86.74% | 91.07% |
| achieved occupancy | 24.33% | 24.42% |
| `Duration` | 290.18 µs | **279.65 µs（0.964×）** |

- **TMA 的读侧修复在原生 bf16 mma 下依旧生效**：`srcunit` 读扇区 **29.88 M→19.92 M（×0.667）**、
  L2 **1.68→1.44 GB**——与 §79.4 逐值同向（`cp.async .BYPASS` 的 1.5× L2 读放大被 TMA 消掉），DRAM 读
  不变（637.6 MB 已是五操作数最小值）。
- **但这次兑现成时间**（`Duration` 0.964×，与 Part B 的 0.962 吻合）：机制是 `short_scoreboard`（smem 取数）
  与 `wait`（固定延迟依赖）同时下降（**1.27→0.38、1.51→1.10**）——TMA 把操作数从 `cp.async`（走 L1TEX、
  仍进 short scoreboard 队列）换成 `cp.async.bulk.tensor`（不经 L1TEX），才让发射侧松下来。对比 §79.4 的
  **tf32-mma** bf16 核：那里 TMA 同样消掉 L2 读放大，却因核被更重的 tf32 mma 依赖（`wait` 更高）钉在
  延迟主导，省下的字节落不到最紧 pipe；**换成指令数减半的原生 bf16 mma 后，smem/发射侧成了可兑现项**。
- 值得注意：`inst_executed` 反而升高（80.8→113.9 M），并非矛盾——TMA 版多了描述符/mbarrier 指令、且
  `bmm64` 的 `cp.async` 循环本身极省指令；但少了 L1TEX 往返和 short-scoreboard 停顿，净结果是更快。

### 80.5 结论与设计输入

- **组合正确、逐位无损**：`-DB2M_BF16MMA × -DB2M_TMA` 可编译、且与 `bmm64`（cp.async）**逐位相同**
  （四组 shape、含非整除 `T=250`）。bf16 原生入口**至此不再有任何 `#error`**（原「只剩 `BF16MMA×TMA`」
  一事到此收口）。改动是纯粹的 `rd2()` atom 解码泛化，**算术一字未动**。
- **TMA 在原生 bf16 mma 核上又成为（小）杠杆**：基础 GEMM 核上 `bmmtma/bmm` 大 shape **0.96×**，
  与 §79.3 同核 tf32-mma 的 wash（0.97–1.03×）相反。**修正** §79.5 的一句「bf16 上 TMA 不是杠杆」——
  准确说法是：**TMA 是否兑现取决于该 bf16 核是否还留有 smem/发射侧头寸**；tf32-mma bf16 核已被 mma 依赖
  钉死故 wash，原生 bf16 mma 核换出指令数后 TMA 有 ~4% 可兑现。设计输入：目标卡 CUDA B2 若在 bf16 部署下
  **同时**采用原生 bf16 mma，则 TMA 输入**可以开**（逐位无损、大 shape ~4%）。
- **边界（诚实）**：只做**基础 GEMM 核**（非六段生产核）、`BVK=64`、`BT=16`、`K=V=128`；`ncu` 只在
  `b8t2048h8` / `t512h4` 两点；链级未测（基础 GEMM 核不落链，机制与 §78.5 一致）。`ncu` 的 `t512h4`
  `Duration`（9.66→10.40 µs，+7.7%）与 Part B 的 CUPTI（1.002×）**不同源**——小 shape 的 `Duration` 受
  replay/时钟影响，判趋势以 CUPTI `dev` 为准。**不改变 §45 的 `cf` 默认部署。**

## 81. 关闭 bf16 原生入口在**六段生产核**里的最后一条边界：`-DB2M_PROD × -DB2M_BF16 × -DB2M_TMA` 逐位无损，但 **TMA 在六段核里不是 bf16 的杠杆**（P7-78）

§78（P7-75）把 bf16 原生 staging 从基础 GEMM 核扩到**六段生产核**（`-DB2M_PROD`），明确留下边界
「**仍仅非 TMA**」；§79/§80（P7-76/77）把 `-DB2M_BF16 × -DB2M_TMA` 与 `-DB2M_BF16MMA × -DB2M_TMA`
打通，但都明确界定在「**仅基础 GEMM 核**」。也就是说，**bf16 入口与 TMA 的交叉从未在六段生产核上验过**。
本轮补上：`-DB2M_PROD -DB2M_BF16 -DB2M_TMA [-DB2M_BF16MMA]` 可编译（无 `#error`）、**逐位正确**，并第一次
量出——**同一旋钮在基础 GEMM 核（§80 大 shape 净赢 0.96×）与六段生产核（本节 1.07–1.11×）反号**，因为
六段核的 bf16 收益本就来自 `cp.async` bf16 staging 把 smem 压到 2 blocks/SM，TMA/`BVK=64` 又把 smem 顶回
1 block/SM、把这份收益吃掉。

新增 `harness/check_bwd_b2_bf16_tma_prod.py`（`harness/check_bwd_b2_bf16_tma_prod.out.txt`，空闲卡）。
**本轮 `b2_mma_probe.cu` 一行未改**——P7-75 的 `rds()`（fp32 scratch）与 P7-76 的 bf16 TMA 张量图早已并存，
生产核只差「把两个旋钮一起打开」的验证。

### 81.1 一个必须先说清的**线上格式**差异：`v`/`dw` 到底喂 fp32 还是 bf16

`-DB2M_BF16` 下 staged 操作数分两类，两条路径的**线上 dtype 不同**，这是移植时最容易踩的一处：

| 操作数 | 读法 | 非 TMA `-DB2M_BF16`（§78） | TMA `-DB2M_BF16`（§79） |
|---|---|---|---|
| `do/vn/dv/h/dh`（`op_t`） | `rd()` | **bf16** | **bf16** |
| raw `v`（`vv`）、`dw`（`dwm`） scratch | `rds()` | **fp32**（P7-75） | **bf16**（P7-76，一张 bf16 图覆盖全操作数） |

故 `prod_bf16`（非 TMA）只期待 `do/vn/dv/h/dh` 是 bf16、raw `v`/`dw` 仍是 fp32；而 `prod_bf16_tma` 期待
**全部**是 bf16。**首版 harness 把 bf16 张量的 `v` 指针喂给非 TMA build（其内核按 `const float*` 读）**，
`db/dApre/dA` 立刻出现 O(1)（6.6–8.8）的差——不是 kernel bug，是喂错了 dtype。按上表分流后全部逐位（§81.2）。
**移植判据**：切 `-DB2M_TMA` 开关的同时，调用方对 `v`/`dw` 的 dtype 也要跟着切。

### 81.2 Part A：`prod_bf16_tma` 与 `prod_tma`、`prod_bf16` 与 `prod` **双向逐位相同**（含非整除 `T=250`）

四组 shape（`B8 T2048 H8` / `B2 T1024 H8` / `T512 H4` / `T250 H3`），全部操作数取 **bf16-valued**（`bf16(x).float()`
喂 fp32 build），11 项输出（`dv2/dvb/dkgb/db/dApre/dA/dqg/dkg/dg/dw/dgk`）：

| 对比 | 含义 | worst `max_abs` |
|---|---|---|
| `prod_bf16_tma` vs `prod_tma`（同 TMA/`BVK=64`） | bf16 staging 对 bf16-valued 操作数是**精确**的 | **0（逐位）** |
| `prod_bf16` vs `prod`（同 cp.async/`BVK=32`） | 同上（非 TMA 路径，§78 的延续） | **0（逐位）** |
| `prod_tma` vs `prod`（fp32，跨布局） | `BVK=64` vs `BVK=32` 的 V-归约分组 | 非 `dgk` **≤9.5e-7**；`dgk` **≤1.9e-6** |
| `prod_bf16mma_tma` vs `prod_bf16_tma`（原生 bf16 mma） | k16 vs k8 的 fp32 累加序 | 直通 `dv2/dvb/db` **逐位 0**；其余 ≤9.5e-7 |

- **`prod_bf16_tma ≡ prod_tma` 全 11 项逐位**（含非整除 `T=250`）：bf16 staging 在 bf16-valued 操作数上
  与 fp32 staging 读到**同一个 fp32 值**（bf16→fp32 扩宽精确、`cvt.rna.tf32` 恒等），TMA 只改 smem 布局——
  这是 §78.2/§79.2/§80.2 那条等式在**六段生产核**上的重现，六段非 GEMM 段读到的操作数完全相同。
- `prod_tma` vs `prod` 的差异只在 `dgk`（V 方向 Σ 的分组）与一处 `dg`：与 §53.2「`base32==base64` 的
  `dq/dk/dw` 逐位、`dgk` 因 fp32 分组差 ~1e-6」同源，跨 `BVK` 的既有事实，非 bf16 引入。
- Part D 交叉核验：`prod_bf16_tma` 的 `dv2 = beta*(A@dv)` vs torch（同一 bf16-valued 操作数）落
  **6.6–9.9e-6**，与 §78 同档（bf16/tf32 底噪）。

### 81.3 Part B/C：**bf16 的收益全在 `cp.async` bf16 staging × `BVK=32`；TMA 与 `BVK=64` 都把它吃掉**

静态（Part B）与 CUPTI `dev`（Part C，kernel-only 单次 `prod` launch）：

| build | smem | regs | spill | `B8 T2048 H8` dev | ×prod | `B2 T1024 H8` | `T512 H4` | `T250 H3` |
|---|---|---|---|---|---|---|---|---|
| `prod`（fp32 cp.async 32） | 133760 B | 146 | 0 | 1298.9 µs | 1.000 | 169.2 | 19.2 | 18.7 |
| `prod_tma32`（fp32 TMA 32） | 123008 B | 128 | 0 | 1503.2 | **1.157** | 1.149 | 1.208 | 1.204 |
| `prod_tma`（fp32 TMA 64） | 209024 B | 112 | 0 | 1327.7 | 1.022 | 1.050 | 1.062 | 1.029 |
| `prod_bf16`（**bf16 cp.async 32**） | **94848 B** | 128 | 4 | **879.2** | **0.677** | **0.726** | 1.006 | 1.013 |
| `prod_bf16_64`（bf16 cp.async 64） | 141952 B | 104 | 0 | 1213.5 | 0.934 | 0.944 | 0.956 | 0.963 |
| `prod_bf16_tma`（bf16 TMA 64） | 123008 B | 112 | 0 | 1297.3 | 0.999 | 1.016 | 1.061 | 1.073 |
| `prod_bf16mma_tma`（bf16 TMA 64 + bf16 mma） | 123008 B | 128 | 0 | 1179.3 | 0.908 | 0.910 | 0.928 | 0.935 |

三个关键读数：

1. **fp32 上 TMA 在六段核里一律不划算**：`prod_tma32/prod` **1.15–1.21×**、`prod_tma/prod` 1.02–1.06×
   （大 `BVK` 把 TMA 的固定代价摊薄、但仍不快）。与 §67/§70/§71「六段生产核非 DRAM bound，TMA 是净负」一致。
2. **bf16 的收益来自 `BVK=32` 的 `cp.async` staging，不是 TMA**：`prod_bf16/prod` 大 shape **0.677×**
   （与 §78.4 的 0.684× 互证）——smem **133760→94848 B**，把占用率从 **1 顶到 2 blocks/SM**（2×94.8=189.7 KB
   < 227 KB）。一旦把 `BVK` 提到 64（`prod_bf16_64`，smem 141952 B > 227/2 KB）或改用 TMA（`prod_bf16_tma`，
   123008 B > 227/2 KB），smem 又越过 2-block 门槛、退回 1 block/SM，**0.677× 的收益整条丢掉**（0.934× / 0.999×）。
   **同 `BVK=64` 下 TMA 相对 cp.async 仍慢 1.07–1.11×**（`bf16_tma64/bf16_64`）。
3. **原生 bf16 mma 在 TMA 之上把丢掉的收益找回来一半**：`prod_bf16mma_tma/prod` **0.908–0.935×**
   （smem 不变 123008 B、regs 112→128）——默认部署应同时开 bf16 staging × 原生 bf16 mma，但**不要**为了让
   操作数走 TMA 而把 `BVK` 顶到 64。

### 81.4 结论与设计输入

- **正确性收口**：`-DB2M_PROD × -DB2M_BF16 × -DB2M_TMA`（及 `+ -DB2M_BF16MMA`）可编译、**逐位无损**
  （11 项输出、含非整除 `T=250`），且与 fp32 `prod` 在 bf16-valued 操作数上**逐位相同**。bf16 原生入口
  在**六段生产核**里的 TMA 边界至此关闭——`b2_mma_probe.cu` 一字未改（P7-75/76 的代码已并存）。
- **性能结论（修正 §80.5 的适用范围）**：§80 在**基础 GEMM 核**上测得 `bmm_tma64/bmm64` 大 shape **0.96×**
  （TMA 是 bf16 的小杠杆）；本轮同旋钮在**六段生产核**上反号为 **1.07–1.11× 变慢**。**判据**：TMA 的收益
  取决于该核**是否在 smem 上有 2-block 头寸**——基础 GEMM 核 bf16 TMA 后 smem 76 KB 仍 2 blocks/SM，
  生产核 bf16 TMA 后 123 KB 只剩 1 block/SM。这与 §51.6/§61/§70「同一旋钮在孤立核与整核可反号」同构。
- **目标卡 CUDA B2 部署输入**：bf16 部署时取 **`-DB2M_BF16`（`cp.async` staging，`BVK=32`）**，把 smem 压到
  2 blocks/SM 是第一杠杆；**TMA 与 `BVK=64` 在六段生产核里都不开**（前者净负、后者吃掉占用率）；若追求
  mma 密度可叠 **`-DB2M_BF16MMA`**（原生 bf16 mma，smem 不变、指令减半）。这与 §78.5「bf16 是 B2 侧
  `.float()` 部署税的收口」一致，只是明确了**生产核里 TMA 不参与这条路线**。**不改变 §45 的 `cf` 默认部署。**
- **边界（诚实）**：仅 `BT=16`、`K=V=128`、fp32 输出、非 `INTRA`/非 `HDW`/非 `DAMMA`；`prod_bf16` 有
  **4 B spill**（vs 其余 0，量级噪声内）；`ncu` 未做（机制已由 §78/§79/§80 的 ncu 与本节 smem/占用率读清）；
  链级未测（§78.5 已有 bf16 生产核落链的结论，本节的差异只在 staging 机制、不改链级结论）。

## 82. 把原生 bf16 mma（`-DB2M_BF16MMA`）与 mma 化 DA（`-DB2M_DAMMA`）落进 **bf16 折叠生产链**：逐位等价、链级 `pcibm/pcib` 0.98–0.99×、`pcibmd/pcib` 0.94–0.97×（P7-79）

§75（P7-75）把 `-DB2M_BF16` 落进链（`pcb`/`pcib`），但链级只测到 **非 DAMMA**、且 §81.4 推荐的
`-DB2M_BF16MMA`（原生 bf16 `mma.sync.m16n8k16`，P7-74）**只在基础 GEMM 核/生产核上 kernel-only 量过**
（§78 Part C：`prod_bf16_mma` 0.659×、`prod_bf16` 0.684×），**从未进链**。本轮把这两个 bf16 原生入口
一起接进完整反向链，关掉 §78.5/§81.6 的「非 BF16MMA / 非 DAMMA 进链」边界，并量出折叠形态下它们的边际。
新增 `harness/check_bwd_b2_bf16mma_e2e.py`（`check_bwd_b2_bf16mma_e2e.out.txt`，空闲卡）+
`harness/ncu_b2_bf16mma_e2e_driver.py` / `harness/probe_bf16mma_e2e_ncu.sh`
（`probe_bf16mma_e2e_ncu.out.txt`，空闲卡）。**本轮 `b2_mma_probe.cu` 一行未改**——只把已有旋钮组合起来。

组装链只吃 FlashKDA 16-chunk K1 workspace：

```
dAv(Triton) -> CUDA-B1 -> CUDA-B2-prod-fold[-DB2M_BF16][-DB2M_BF16MMA][-DB2M_DAMMA] -> 反向 cumsum
```

变体：`pc`/`pci`（fp32，§64 基线）、`pcb`（bf16 非折叠）、`pcib`（bf16 折叠，§75 基线）、
`pcibm`（`+ -DB2M_BF16MMA`）、`pcibmd`（`+ -DB2M_BF16MMA -DB2M_DAMMA`）、`cf`（§45 最优链）。

### 82.1 Part A：bf16 原生 mma / DAMMA 都是**代数无损**（只改 fp32 累加序 / tf32-DA）

各变体 vs FLA autograd 的六项梯度 worst-case `max_rel`（4 组 CASES，含非整除 `T=250`、`g=0`、`B=2`）：

| 字段 | `pcb` | `pcib` | `pcibm` | `pcibmd` | `pcibm` vs `pcib` | `pcibmd` vs `pcib` |
|---|---|---|---|---|---|---|
| `dq` | 3.922e-3 | 3.916e-3 | 3.916e-3 | 3.916e-3 | **9.0e-8** | **9.0e-8** |
| `dk` | 3.455e-3 | 3.456e-3 | 3.456e-3 | 3.456e-3 | **6.9e-8** | 3.5e-6 |
| `dv` | 6.795e-3 | 6.795e-3 | 6.795e-3 | 6.795e-3 | **0（逐位）** | **0（逐位）** |
| `dbeta` | 6.012e-3 | 6.012e-3 | 6.012e-3 | 6.012e-3 | **0（逐位）** | 3.0e-6 |
| `dg` | 7.790e-3 | 7.854e-3 | 7.854e-3 | 7.854e-3 | **1.1e-7** | 4.0e-6 |
| `dh0` | 9.365e-3 | 9.365e-3 | 9.365e-3 | 9.365e-3 | **0（逐位）** | **0（逐位）** |

- **四个变体 vs FLA 的 worst-case 与 §67/§75 逐值相同**（`dq 3.9e-3 … dh0 9.365e-3`）：bf16 staging、
  折叠、原生 bf16 mma、mma 化 DA 都不改「梯度对 FLA 的落点」，全在 bf16/tf32 底噪上。
- **`pcibm` vs `pcib`**：只有换 mma ISA 的 fp32 **累加序**（`dq/dk ~1e-7`、`dg 1.1e-7`），`dv/dbeta/dh0`
  **逐位**（由 B1 / 门控定，不经 GEMM 核的累加序）。这与 §77.2「`bmm` vs `bf16(tf32)` 只差累加序、
  `dgk` 逐位」同源。
- **`pcibmd` vs `pcib`**：再加 DAMMA 的 **tf32-DA** 改写（`dk 3.5e-6 / dbeta 3.0e-6 / dg 4.0e-6`），
  `dq/dv/dh0` 逐位——与 §72.1「真实链里 DA 的 tf32 改写量本身只有 ~1e-6 量级」互证。

### 82.2 Part B：链级收益——`pcibm` 稳赢 ~1–2%，`pcibmd`（叠 DAMMA）~3–6%

反向链 `dev`（CUPTI，空闲卡；wall 见原始输出，比值同向）：

| case | FLA | `cf` | `pc` | `pci` | `pcb` | `pcib` | `pcibm` | `pcibmd` | `pcibm/pcib` | `pcibm/cf` |
|---|---|---|---|---|---|---|---|---|---|---|
| T512 H4 | 203.1 | 74.4 | 116.0 | 95.3 | 104.4 | 80.2 | 79.0 | 78.0 | 0.986 | 1.062 |
| B2 T1024 H8 | 662.8 | 258.4 | 464.8 | 390.0 | 353.3 | 320.0 | 317.4 | 304.9 | 0.992 | 1.228 |
| B8 T2048 H8 | 4130.4 | 1402.4 | 2907.6 | 2423.5 | 2035.5 | 1918.4 | 1901.9 | 1805.5 | 0.991 | 1.356 |
| T250 H3 non-div | 141.1 | 50.3 | 83.0 | 65.6 | 78.6 | 59.6 | 58.5 | 56.7 | 0.981 | 1.163 |

- **`pcibm/pcib` = 0.981–0.992（4/4 稳定为正）**：原生 bf16 mma 在整链上是真实但**小幅**的赢
  （~1–2%），与 §77.3「大 shape 已 L2-bound 故 kernel-only 0.99×」一致——ISA 减半在这里换不到更多。
- **`pcibmd/pcib` = 0.94–0.97**：再叠 **mma 化 DA** 拿 ~3–6%（`B8` 1918→1805.5）。这与 §69.4「DA 段内
  2–3×、整核 +5–7%」同阶，是**本轮链级最大的单项**。
- `cf` 仍是全场最快（`pcibm/cf` 1.06–1.36）：与前几轮「CUDA B2 生产核正确但更慢」的结论一致，
  **默认部署不变**（§45 的 `cf`）。

### 82.3 Part C/D：折叠形态把 bf16 的「占用率红利」吃掉；native bf16 mma 的 tensor% 下降是**口径产物**

静态（ptxas）与 `ncu`（`b8t2048h8`，同一份操作数）：

| build | regs | spill | dyn smem | occ-limit (smem/reg) | achieved warps | tensor% | bank-conflict | `long_sb` | `wait` |
|---|---|---|---|---|---|---|---|---|---|
| `b_fold` | 128 | 8 | **116.35 KB** | 1 / 2 | 12.46% | 3.14 | 40.7M | 2.92 | 1.01 |
| `b_fold_mma` | 136 | 0 | 116.35 KB | 1 / **1** | 12.45% | **1.90** | 40.7M | **3.91** | 1.29 |
| `b_fold_mma_da` | 128 | 0 | 116.35 KB | 1 / 2 | 12.45% | 2.25 | **29.7M** | 4.37 | 1.34 |

三个读数：

1. **折叠形态下 bf16 的 2-block/SM 红利消失**：非折叠 bf16 生产核 smem 仅 **94848 B → 2 blocks/SM**
   （§78.4，`prod_bf16` 0.677× 的来源），但折进 `intra` 后 dyn smem 涨到 **116.35 KB > 227/2 KB**，
   `launch__occupancy_limit_shared_mem = 1`——**折叠把 §78 那份占用率红利整条让出去了**。这也解释为何
   `pcib` 相对 `pcb` 的链级优势（`B8` 2035.5→1918.4）**不是**来自占用率，而是来自「少一次 B3 kernel +
   不落盘 `dA`」。
2. **native bf16 mma 的 `tensor%` 从 3.14 降到 1.90 是口径产物、不是「tensor 更闲」**：H100 上 bf16 mma
   吞吐是 tf32 的 **2×**，同一 FLOP 的 tensor 周期**减半**（同 §77.5 的告警）。SASS 上 mma 指令精确减半，
   但本核本就不 tensor-bound（`sm__throughput` 24% vs `dram` 25%、`lts` 36%），故链级只值 ~1–2%。
   顺带：`b_fold_mma` regs 128→136 使 **reg-limit 也降到 1**（65536/(136·256)=1.88），但 achieved 占用
   **不变**（smem 已钉死 1 block），故寄存器不是旋钮。
3. **DAMMA 的收益落点是 bank conflict**：`b_fold_mma_da` 把 shared-load bank conflict 从 **40.7M 砍到
   29.7M**（−27%），与 §69.6/§71.4「标量 DA 是 bank-conflict 大户、mma 化削 ~40%」一致——这正是
   `pcibmd` 那 3–6% 链级赢的机制。
4. **ncu `wait`/`long_sb` 反而变差**（1.01→1.29 / 2.92→3.91）而链级 CUPTI 更快：再次踩到
   §25.5/§63.5「`ncu` 窗口与 CUPTI/wall 不同源、可反号」——**比值以 CUPTI `dev`/wall 为准**，ncu 只读
   静态量（smem/占用率/字节/bank conflict）。

### 82.4 结论与设计输入

- **边界关闭**：`-DB2M_BF16 × -DB2M_BF16MMA × -DB2M_DAMMA × [-DB2M_INTRA]` 在**完整反向链**上可跑、
  **逐位/底噪等价**（六项 vs FLA 与 §67/§75 逐值相同），`b2_mma_probe.cu` 一字未改。
- **链级边际**：`pcibm/pcib` **0.98–0.99×**（native bf16 mma，~1–2%）、`pcibmd/pcib` **0.94–0.97×**
  （再叠 mma 化 DA，~3–6%）。**两者可叠加、都免费**（只换 mma ISA / DA 实现，不改布局）。
- **目标卡 CUDA B2 bf16 部署输入（在 §81.4 基础上补充）**：若走**折叠**（单 launch、`dA` 不落盘）形态，
  建议 **`-DB2M_BF16 -DB2M_BF16MMA [-DB2M_DAMMA]`**——`BF16MMA` 免费 ~1–2%、`DAMMA` 免费 ~3–6%；
  但注意**折叠已把 smem 顶到 116 KB / 1 block/SM**，非折叠形态的 2-block 红利在折叠里拿不到，
  取舍同 §66.6（大 shape 保两 kernel、小 shape 折叠）。**默认部署仍不变**：§45 的
  `dAv→CUDA-B1→融合 Triton B2/B3→cumsum`（`cf`）。
- **边界（诚实）**：仅 `BT=16`、`K=V=128`、fp32 输出、非 TMA；`pcib`/`pcibm` 有小 spill（8/0 B）；
  `ncu` 只测 `b8t2048h8`/`b2t1024h8` 两档（机制由 §77/§78/§81 的 ncu 与本节 smem/bank-conflict 读清）。

## 83. 关闭 §79 遗留的「raw `v`/`dw` 的 bf16 精度未单独消融」：新增 `-DB2M_BF16_VDW`——逐位可核验、精度代价 ≤3e-3，但**单独开是 1.4–1.5× 回归**（寄存器 128→140 关掉 2-block/SM 窗口），叠 `BF16MMA` 才中性（P7-80）

§79（P7-76）把「bf16 原生入口 × TMA」拼通后，边界一句写明：「bf16+TMA 下 `vv`/`dwm` 被降为 bf16
（生产核组合时 raw `v`/`dw` 的 bf16 精度未单独消融）」。本节把这条**唯一遗留的精度问题**关掉：给
非 TMA 的 bf16 入口加一个 **staging-dtype 旋钮 `-DB2M_BF16_VDW`**，让 `vv`（raw `v`）与 `dwm`（`dw`）
也按 bf16（`op_t`/`OPS` 行距）搬入 smem——即 TMA 路径的布局在 `cp.async` 路径上补齐；`prefetch` 改用
bf16 粒度（U=8），`rds()` 折叠成与 `rd()` 相同的 bf16→fp32 精确扩宽。改动是**纯布局/搬运 dtype**，
`b2_mma_probe.cu` 的算术一字未动。

### 83.1 Part A：bf16 值下与 fp32 staging **逐位相同**（旋钮代数惰性）

`prod_bf16_vdw`（`-DB2M_BF16 -DB2M_BF16_VDW`，`v`/`dw` 走 bf16 线上）对 `prod_bf16`（`v`/`dw`
走 fp32 线上、fp32 staging）——两边喂**同一个 bf16 值**（一边 `bf16(x)` 张量、一边 `bf16(x).float()`）：

| case | 11 项输出 max_abs | 判定 |
|---|---|---|
| B8 T2048 H8 | 0.000e+00 | 逐位相同 |
| B2 T1024 H8 | 0.000e+00 | 逐位相同 |
| T512 H4 | 0.000e+00 | 逐位相同 |
| T250 H3（非整除） | 0.000e+00 | 逐位相同 |

因为 bf16→fp32 扩宽精确，把 bf16 值以 fp32 或 bf16 落 smem 得到**逐字节相同的 fp32 操作数**，
故 11 项输出全逐位（含非整除 `T=250`）。**这证明 `-DB2M_BF16_VDW` 只改布局、不改代数**。

### 83.2 Part B：全精度 `v`/`dw` 的 bf16 量化代价——scale-relative ≤3e-3（=bf16 底噪）

把**全精度** `v`/`dw` 分别量化到 bf16 再喂 `prod_bf16`（fp32 staging），对未量化基线比；度量用
`max_abs/|ref|max`（避免近零抵消元素的 `max_rel` 噪声）。四组 shape（含非整除）：

| 量化对象 | 受影响输出 | `max_abs/|ref|max`（四组区间） |
|---|---|---|
| **`v` only** | `dApre` / `dA`（SAND 的 `dv@v^T`）、`db`（`Σ dkgb·kg` 里带 `v`） | `dApre` 1.42–1.74e-3、`dA` 1.68–1.94e-3、`db` 1.53–1.78e-3 |
| **`dw` only** | `dkgb`（DKGB 的 `A@-dw`）、`dkg`/`dg`（门控链吃 `dkgb`）、`db` | `dkgb` 2.69–3.01e-3、`dkg` 2.0–3.0e-4、`dg` 1.6–2.3e-4、`db` 5.7–9.2e-4 |
| **both** | = 两者并集 | 与「各自量化」同阶叠加（`dkgb 2.69e-3 … dA 1.94e-3`） |

- `dgk`（GEMM 核输出）与 `dv2/dvb/dw/dqg` **逐位不受影响**（`vv`/`dwm` 只进 SAND/DKGB/DB 后处理）。
- 「两边都量化、一边 fp32 staging、一边 bf16 staging」的交叉核对仍是 **max_abs=0**（Part A 的机制）。
- **量级**：最坏 `dkgb` ≈3e-3，与 §78 已存在的 bf16 操作数底噪（`dq 3.9e-3` 等）**同阶**。且部署里
  **raw `v` 本就是 bf16**（`vv` 侧代价只对「v 是 fp32 却硬降 bf16」成立、生产不成立），真正新增的代价
  只有 `dw` 侧（`dkgb` ~3e-3）——而 `dw` 是 B2 自己的 fp32 GEMM 中间量，要降 bf16 还须让 `dw` 的
  **store 出口也改 bf16**（本节只做读侧 staging 消融，写侧留作边界）。

### 83.3 Part C/D：`vdw` 单独开是**寄存器回归**（2→1 block/SM），叠 `BF16MMA` 才中性

静态（ptxas）与 CUPTI（空闲卡，一次生产 launch）：

| build | regs | spill | smem (B) |
|---|---|---|---|
| `prod` | 146 | 0 | 133760 |
| `prod_bf16` | 128 | 4 | 94848 |
| `prod_bf16_vdw` | **140** | 0 | **90752** |
| `prod_bf16_vdw_mma` | **108** | 0 | 90752 |

| case | `bf16` µs | `vdw` µs（×bf16） | `vdw_mma` µs（×bf16） |
|---|---|---|---|
| B8 T2048 H8 | 885.4 | **1346.0（1.520×）** | 870.1（0.983×） |
| B2 T1024 H8 | 125.0 | **175.6（1.404×）** | 122.4（0.979×） |
| T512 H4 | 19.4 | 20.8（1.070×） | 19.8（1.020×） |
| T250 H3（非整除） | 19.0 | 20.1（1.061×） | 19.3（1.017×） |

**机制（`ncu`，`b8t2048h8`）**：`vdw` 的 smem **更小**（90752 < 94848）但**慢 1.5×**，根因**不是**
访存也不是 bank conflict，而是**寄存器数 128→140 越过了 2-block/SM 的寄存器门槛**：

| 读数 | `bf16` | `bf16_vdw` |
|---|---|---|
| `launch__registers_per_thread` | 128 | **140** |
| `launch__occupancy_limit_registers` | **2** | **1** |
| `sm__warps_active`（achieved） | 24.64% | **12.46%** |
| `dram__bytes_read` | 1.07 GB | 916 MB |
| `lts__t_bytes` | 3.01 GB | 2.64 GB |
| bank conflict（shared ld） | 24.87M | 26.84M |
| `gpu__time_duration` | 879.8 µs | 1340 µs |

H100 寄存器堆 65536/SM：`128×256×2 = 65536`（恰好 2 block），而 `140×256×2 = 71680 > 65536`（→1 block），
achieved warps 也正好**减半**（24.64→12.46%）。`vdw` 的 DRAM/L2 反而更低（`v`/`dw` 读减半，−15%/‑12%）、
bank conflict 只 +8%——**省下的字节落在不最紧的 pipe，而占用率被寄存器腰斩，净亏**。叠 `-DB2M_BF16MMA`
后 regs 落到 **108**（≤128）→ 恢复 2 block/SM，故 `vdw_mma` 回到中性（0.98–1.02×）。

### 83.4 结论与设计输入

- **边界关闭**：`vv`/`dwm` 的 bf16 staging 在非 TMA 路径上可开、**逐位可核验**（bf16 值下 max_abs=0）、
  精度代价 **≤3e-3（`dkgb`，=bf16 底噪）**；`b2_mma_probe.cu` 算术未动。
- **但它不是杠杆**：`-DB2M_BF16_VDW` **单独开**把 regs 128→140、关掉 2-block/SM 窗口 → **大 shape
  1.4–1.5× 回归**；**叠 `-DB2M_BF16MMA`**（108 regs）仅回到**中性**（0.98–1.02×），smem 省 4 KB、
  `v`/`dw` 读减 ~15% 都**换不来时间**。与 §81「TMA/`BVK=64` 在六段核反号、bf16 收益全在
  `cp.async`×`BVK=32` 的占用率头寸」**同构**——**再省一点字节的前提是别动 2-block/SM 头寸**。
- **目标卡 CUDA B2 部署输入（不变）**：$v$/$dw$ 的 staging **维持现状**（非 TMA 走 fp32、TMA 已 bf16），
  **默认部署仍是 §45 的 `cf`**；`-DB2M_BF16_VDW` 只作「若某天寄存器允许（例如叠 `BF16MMA`）可零成本
  对齐 TMA 布局」的备选，非建议开项。
- **边界（诚实）**：只测 `BT=16`、`K=V=128`、非 TMA、非折叠、未落整链；`dw` **写侧**（`dw_out` store）
  未改 bf16（故「`dw` 端到端 bf16」需生产侧一并改，属未做）；`ncu` 只取 `b8t2048h8`（机制由
  regs/`occupancy_limit_registers`/achieved warps 三者自洽钉死）。

## 84. 关闭「只测 K=V=128」边界：K2 形状 CUDA B2 **base GEMM core** 的 K≠V 泛化——逐位可核验、且是**唯一能跑 V=256** 的实现（P7-81）

KDA 的状态是 $S\in\mathbb{R}^{K\times V}$：**key / 输出 N 维 `K`** 与 **value / 收缩维 `V`** 在一般
情形下是**两个独立的量**。但从 P7-46（§49）到 P7-80（§83），CUDA B2 的**每一轮**都用单一常量
`KD = VD = 128` 编译与核验（`b2_mma_probe.cu:56-57`），ROADMAP 的每条边界都写着「只测 `K=V=128`」。
本轮把这两个维度**参数化**并做一次受控扫描，回答一个问题：**K2 形状的 base GEMM core 本身是否已经
对 `K≠V` 泛化？** 答案是——**是，且近乎免费**；而「六段非 GEMM 段」仍需 `V=K`。

改动只有两处（`b2_mma_probe.cu`，**算术一字未动**）：

- `b2_mma_probe.cu:64-74`：把 `constexpr int KD/VD` 改成宏 `B2M_KD`/`B2M_VD`（默认 128），
  `BW = KD/NW` 由 `KD` 导出并加 `static_assert(BW*NW == KD)`。
- `b2_mma_probe.cu:469-478`：新增 `static_assert(KD==128)`（**只扫 `V`**：`K=128` 被 `dgk` 列归约
  `l & 15` 与门控链的 16 宽 lane 布局写死）与一条 `#error`：**只要打开任一非 GEMM 段**
  （`B2M_OUT/DB/DA/INTRA/GATE/DKGB/HDW/VV_KG`）**就要求 `VD==KD`**——因为那六段按 warp 输出宽度
  `BW=KD/NW` 去**切 V 轴**（如 `dv2` 的 `[w*BW, w*BW+BW)`、`dbv` 的 raw-v 镜像），`V≠K` 会错位。

**为什么 base GEMM core 已经泛化**：它**本就分别索引两个轴**——`dq/dk/dw/dgk` 的地址一律用 `KD`
（`b2_mma_probe.cu` 里 `*KD + col` 一族），`do/vn/dv` 的操作数地址一律用 `VD`（`*VD + v0`），
`h/dh` 的 chunk 基址是 `KD*VD`（`:801`，`hbase`）、行步长 `VD`；V 轴只按 `BVK` 分块（`nvt = VD/BVK`，
`:1039`），`h/dh` 的 TMA 图 `make_map_hdh` 也是 `{VD, KD, H, NT, B}`（`:1851`）、`do/vn/dv` 的
`make_map_dov` 是 `{VD, T, H, B}`（`:1836`）——两张图本就各带 `VD`/`KD`。所以 GEMM core 对 `K≠V`
**零改动可用**，只差一个能编出非 128 `V` 的宏。

### 84.1 Part A：正确性（`K=128`，`V∈{64,128,256}`，含非整除 `T=250`）

四组 shape、每组喂同一份随机输入，比 torch fp32（`allow_tf32=False`，代数 golden）与 identical-op
Triton core（`check_bwd_b2_core.py`，本就泛化于 `K,V`）。全表最坏 `max_rel`：

| V | field | cuda/torch | trit/torch | cuda/trit |
|---|---|---|---|---|
| 64 | dq | 3.282e-04 | 8.832e-04 | 1.088e-03 |
| 64 | dk | 3.176e-04 | 9.524e-04 | 1.025e-03 |
| 64 | dw | 3.970e-04 | 8.723e-04 | 9.894e-04 |
| 64 | dgk | 2.067e-07 | 2.067e-07 | 2.067e-07 |
| **128** | dq | **3.583e-04** | 8.967e-04 | 9.851e-04 |
| **128** | dk | **3.281e-04** | 8.654e-04 | 9.446e-04 |
| **128** | dw | **3.408e-04** | 8.363e-04 | 9.804e-04 |
| **128** | dgk | 1.628e-07 | 1.335e-07 | 1.628e-07 |
| 256 | dq | 3.327e-04 | n/a | n/a |
| 256 | dk | 3.119e-04 | n/a | n/a |
| 256 | dw | 3.176e-04 | n/a | n/a |
| 256 | dgk | 1.901e-07 | n/a | n/a |

- **CUDA 一律落 tf32 底噪**（`dq/dk/dw` 3.1–4.0e-4、`dgk` 1.6–2.1e-7 = fp32 归约序），且**三档 V 同阶**；
  `V=128` 那一行**逐值复现 §49（P7-46）**的 baseline（`cuda/torch dq 3.6e-4` 等），证明宏参数化
  **未改任何生成代码**（默认 `KD=VD=128` 与改动前逐字节等价）。
- **`V=256` 之所以只有 `cuda/torch` 一列**：identical-op Triton core 把整块 `[V,K]` 操作数留在 smem，
  `V=256` 需 **311296 B > 232448 B**（硬件上限）→ `OutOfResources: shared memory`；而 **CUDA 按
  `BVK` 流式搬 V-tile**（smem 恒 87552 B，与 `V` 无关），故它是**唯一能跑 `V=256` 的实现**。这是
  「K2 形状（操作数流式、非整块常驻）」在 `V>K` 上的**结构性优势**，不是数值问题。

### 84.2 Part B：identical-op kernel 计时（CUDA vs Triton core）

| case | V | CUDA dev µs | CUDA wall µs | trit nw8 µs | CUDA/trit |
|---|---|---|---|---|---|
| B8 T2048 H8 | 64 | 297.3 | 299.0 | 282.9 | 1.05× |
| B2 T1024 H8 | 64 | 44.0 | 46.0 | 39.4 | 1.12× |
| T512 H4 | 64 | 6.6 | 28.8 | 4.3 | 1.54× |
| T250 H3（非整除） | 64 | 5.5 | 28.9 | 3.7 | 1.49× |
| B8 T2048 H8 | **128** | **517.7** | 536.1 | 567.4 | **0.91×** |
| B2 T1024 H8 | **128** | **70.9** | 72.9 | 81.7 | **0.87×** |
| T512 H4 | 128 | 9.1 | 28.5 | 6.8 | 1.34× |
| T250 H3（非整除） | 128 | 8.5 | 28.9 | 5.8 | 1.47× |
| B8 T2048 H8 | 256 | 994.5 | 994.7 | n/a | n/a |
| B2 T1024 H8 | 256 | 130.7 | 132.7 | n/a | n/a |
| T512 H4 | 256 | 15.5 | 28.3 | n/a | n/a |
| T250 H3（非整除） | 256 | 14.5 | 28.4 | n/a | n/a |

- **`V=128` 行复现 §49**：`B8` `CUDA 517.7 vs Triton 567.4 = 0.91×`（§49 报 0.93×、§75 同口径）；
  `B2` 0.87×；小 shape 1.3–1.5×（固定开销主导）——与既有结论一致。
- **随 V 近似线性**：`B8` 大 shape dev `297.3 (V=64) → 517.7 (V=128) → 994.5 (V=256)`，
  `V: 64→128` 比 1.74×（理想 2×）、`128→256` 比 1.92×——收缩工作量 $\propto V$，小 shape 的固定开销
  让低 V 端偏离线性更多。
- **`V=256` 无 Triton 对照**（其 smem OOM，见 §84.1）；CUDA 的 wall≈dev（994.5/994.7），仍是单核、
  无 host 税。

### 84.3 Part D：静态代价（smem 与 V 无关）

| V | smem (B) | block | nvt（BVK=32） |
|---|---|---|---|
| 64 | 87552 | 256 thr / 8 warp | 2 |
| 128 | 87552 | 256 thr / 8 warp | 4 |
| 256 | 87552 | 256 thr / 8 warp | 8 |

**per-stage smem 与 `V` 无关**——stage 缓冲只装**一个 `BVK` 宽的 V-tile**（`[BT][BVK]` 的
`do/vn/dv` + `[KD][BVK]` 的 `h/dh`）。`V` 只改**stage 数** `nvt=VD/BVK`，在 `it` 循环里逐 tile
累加；这正是它能跑 `V=256` 而「整块常驻」的 Triton core 不能的原因。

### 84.4 结论与设计输入

- **「只测 `K=V=128`」边界（对 base GEMM core）关闭**：把 `KD`/`VD` 宏参数化后，K2 形状 CUDA B2 的
  base GEMM core **对 `K≠V` 逐位/底噪可核验**（`V=64/256` 均落 tf32 底噪），且默认 `V=128` 下
  **与改动前逐字节等价**（复现 §49）。泛化性来自 core **本就分轴索引**（K-dim 用 `KD`、V-dim 用 `VD`、
  TMA 图各带 `VD`/`KD`），**不是新写的分支**。
- **结构性优势（新）**：因为操作数**按 `BVK` 流式**、smem 与 `V` 无关，CUDA core 是**唯一能跑
  `V=256`** 的实现——identical-op Triton core 把整块 `[V,K]` 留 smem，`V=256` 直接 311 KB > 232 KB
  **编不出**。即「K2 形状」在 `V>K` 上把『能否运行』都改变了。
- **边界（诚实）**：
  - **只参数化 `V`**：`K` 固定 128（`dgk` 列归约 `l&15`、门控链 lane 布局写死；`static_assert(KD==128)`）。
    支持 `K<128`/`K>128` 需重写这些 16 宽的 lane 布局，未做。
  - **只覆盖 base GEMM core**：六段非 GEMM 段按 `BW=KD/NW` 切 V 轴，**要求 `VD==KD`**（已加 `#error`
    守卫）；「六段核 K≠V」仍未做，是留给后续的真结构项。
  - 只测 `K=128`、`V∈{64,128,256}`、`BVK=32`、fp32/tf32 mma、非 TMA（TMA 分支未随 `V` 复测，但
    `make_map_*` 已带 `VD`/`KD`，机制上应随即可用——**待验证**）。
- **目标卡 CUDA B2 部署输入（不变）**：默认部署仍是 §45 的 `cf`；本轮的宏是**为泛化/健壮性**而加，
  对 `V=128` 部署**零行为差异**。若未来 KDA 出现 `V≠K` 的 head 配置，base GEMM core 可直接编出
  （K2 形状流式 V-tile 是必要条件），但六段非 GEMM 段须一并改造。

## 85. 关闭 P7-81 的两条遗留：base GEMM core 的 **TMA×K≠V** 复测 + 六段非 GEMM 段的 **K≠V** 泛化（P7-82）

§84（P7-81）把 CUDA B2 的 `K`/`V` 参数化并证了 **base GEMM core** 对 `K≠V` 泛化，但明确留了两条边界：
(a) **TMA 分支未随 `V` 复测（待验证）**；(b) **六段非 GEMM 段仍 `VD==KD`（未做）**——因为那六段按 warp
输出宽度 `BW=KD/NW` 切 V 轴。本轮把两条一并关闭，其中 (b) 是本轮的主要结构改动。

### 85.1 P7-81 待验证 (a)：base GEMM core 的 TMA×K≠V —— **成立且布局无关**

`check_bwd_b2_kv.py` 之前把 TMA 的构建漏了驱动库链接（TMA 走 `cuTensorMapEncodeTiled`，需 `-lcuda`），
`ctypes.CDLL` 直接报 `undefined symbol: cuTensorMapEncodeTiled`——已给 build 命令补 `-lcuda`
（`check_bwd_b2_kv.py:68-72`，对非 TMA 无害）。补上后以 `B2KV_CFLAGS=-DB2M_TMA` 重跑（`harness/
check_bwd_b2_kv_tma.out.txt`）：

- **正确性**：`K=128`、`V∈{64,128,256}`、四组 shape（含非整除 `T=250`），`dq/dk/dw` 全落 tf32 底噪
  **3.1–4.0e-4**、`dgk` **1.6–2.1e-7**；且 **worst 表与 §84.1（cp.async 版）逐值完全相同**
  （`V=64 dq 3.282e-04 … V=256 dgk 1.901e-07`）——TMA 只换 stage 布局，连 tf32 累加序都不动。
- **计时**（identical-op kernel，`B8T2048H8`）：CUDA dev `278.4 (V=64) / 493.6 (V=128) / 924.8 (V=256) µs`，
  随 V 近似线性；`V=128` 对 Triton core **0.90×**（§84.2 非 TMA 为 0.91×，一致）；`V=256` 无 Triton
  对照（其 smem OOM）。
- **静态**：TMA 的 smem **恒 77952 B（76.1 KB），与 V 完全无关**（同样只摆一个 `BVK` 宽 stage，
  `V` 只改 `nvt`）。对照 §84.3 非 TMA 的 87552 B——TMA 的 atom 布局更省。

**结论**：`make_map_dov`/`make_map_hdh` 本就各带 `VD`/`KD`（§84 已核），TMA×K≠V **随即可用、逐值正确、
与 stage 布局无关**；P7-81 的待验证 (a) 关闭。

### 85.2 六段非 GEMM 段 K≠V 泛化：把 V 轴段的 warp 宽度从 `BW` 换成 `BVW`

六段可分成两组：

- **K 轴组**（按 `K` 分区，`BW=KD/NW` 不变）：`DKGB` 的 `dkgb`、门控链的 `dqg/dkg/dg`、`dgk` 的列归约、
  `DA` 的 `dw@kg^T`、以及折叠进来的 B3 `intra`。
- **V 轴组**（按 `V` 分区，本轮改用 **`BVW = VD/NW`**）：`OUT` 的 `dv2/dvb`、`DA` 的 `dv@v^T`
  （标量与 mma 两支）、`DB` 的 raw-v 镜像。

改动（`b2_mma_probe.cu`，**算术一字未动**，只换 warp 宽度与循环界）：

- `:75-83`：新增 `BVW = VD/NW`、`NBV = BVW/8`（每 warp 的 mma `n=8` tile 数）与两条 `static_assert`
  （`VD % NW == 0`、`BVW % 8 == 0`）。
- `:558`：`dbv[NW][BT][BW]` → `[BVW]`；`:974`：`acc2[2][4]` → `acc2[NBV][4]`。
- V 轴段四处把 `local0 = warp*BW - it*BVK` / `local0+BW<=BVK` 换成 `BVW`（`:1197/1255/1274/1299`），
  mma 的 `nt<2` → `nt<NBV`（OUT，`:1209`）、`for j<BW` → `<BVW`（DA 标量，`:1259`）、
  `for kk<BW/8` → `<BVW/8`（DA mma，`:1277`）；`dv2/dvb` 的 store 也改 `BVW`/`nt<NBV`（`:1380-1381`）。
- `:1807-1822`：`DB` 的归约**拆成两项**——`dvb*v`（V 轴，`nt<NBV`）与 `dkgb*kg`（K 轴，`nt<2`）。
  为**保住 K==V 的逐位回归**，改成 `nt < max(NBV,2)` 的单一循环、按 `if (nt<NBV)`/`if (nt<2)` 分别加，
  在 `NBV==2` 时**逐字复现**原来的 acc2/acc3 交错序。
- `:482-490`：把 P7-81 那条「非 GEMM 段一律要求 `VD==KD`」的 `#error` **收窄**为唯一真实耦合——
  **全局 `dw` 的 DKGB staging（`dwm`）是 K-tiled 但骑在 V-tile 循环 `it` 上**，故
  `-DB2M_DKGB` 且**不带** `-DB2M_HDW` 时仍要求 `VD==KD`；`-DB2M_HDW` 的同 CTA 交接读核内 `swall`
  （K 轴、`BW` 分区），对 `K≠V` 是干净的。生产核的每个变体都用 `-DB2M_PROD -DB2M_HDW`，故不受影响。

### 85.3 核验（`harness/check_bwd_b2_kv_prod.py` + `..._kv_prod.out.txt`）

**Part A（回归，关键）**：K==V（`VD=128`）下，**改后源码 vs 改前源码（`git HEAD`）编译的
`-DB2M_PROD -DB2M_HDW` 核，在 10 项输出上逐位相同（`max_abs=0`，三组 shape 含非整除 `T=250`）**。
（首版 DB 归约拆成两个独立循环，`db` 差 ~7e-7 的 fp32 结合序；改成 §85.2 的 `max(NBV,2)` 单循环后
K==V 回归到**逐位 0**——这正是「泛化不得改变既有 `K=V=128` 路径」的可核验保证。）

**Part B（K≠V 正确性，`-DB2M_PROD -DB2M_HDW`，`dw` 由 torch 从操作数重算 `dw[t,k]=Σ_v dv[t,v]h[k,v]`）**：
V 轴三项 vs torch fp32 的 worst `max_rel`：

| V | dv2 (vs torch) | dvb (vs torch) | dvb (vs torch-ops) | db | dApre |
|---|---|---|---|---|---|
| 64 | 3.529e-04 | 2.889e-04 | 2.889e-04 | 2.629e-04 | 1.413e-04 |
| 128 | 2.923e-04 | 2.795e-04 | 2.795e-04 | 2.473e-04 | 1.319e-04 |
| 256 | 3.324e-04 | 4.456e-04 | 4.456e-04 | 2.430e-04 | 1.081e-04 |

全部落 **tf32 底噪**（对照 §55 `dvb` 3.6e-4、§56 `dkgb` 3.6e-4、§57 `dApre` 5.3e-7——本轮 `dvb`/`db`
同阶；`V=256` 的 `dvb` 略抬到 4.5e-4 仍是 tf32 级）。含非整除 `T=250` 与 `B=2`。

**Part C（stage 布局恒等）**：`prod` vs `prod + TMA`，`V∈{64,128,256}` × 三组 shape，**10 项输出全部
`max_abs=0`**——与 §85.1/§84 同结论（TMA 只换布局）。

**Part D（静态）**：`prod(HDW)` 的 smem `122.1 KB (V=64) / 126.1 KB (V=128) / 134.1 KB (V=256)`，
`NBV = 1/2/4`。比 base GEMM core（§84.3 的 87.5 KB）大，正是六段的 `dbv/swall/skg/da16` 等常驻缓冲；
`V` 增大的那部分来自 `dbv`（`[NW][BT][BVW]`）。

### 85.4 结论与边界

- **P7-81 的两条遗留全部关闭**：(a) base GEMM core 的 TMA×K≠V **成立且布局无关**；(b) 六段非 GEMM 段
  **对 K≠V 泛化**（V 轴段改 `BVW=VD/NW`），K==V 路径**逐位不变**、K≠V 落 tf32 底噪、TMA 版逐位同。
- **设计输入**：`K≠V` 时「六段核」唯一的结构耦合是**全局 `dw` 的 DKGB staging**——它必须改用
  同 CTA 交接（`-DB2M_HDW`）或对 `dwm` 单独做 K-tiling；目标卡 CUDA B2 的生产形态本就是
  `-DB2M_PROD -DB2M_HDW`（§64），故 **K≠V 部署无需额外工作**。
- **边界（诚实）**：
  - 仍 **`K` 固定 128**（`static_assert(KD==128)`：`dgk` 列归约 `l&15`、门控链 16 宽 lane 布局写死）；
    只扫 `V∈{64,128,256}`。`K≠128` 需重写这些 lane 布局，未做。
  - 只测 `BT=16`、`BVK=32`、fp32/tf32 mma、`-DB2M_HDW`（生产形态）；非 HDW 的 K≠V 由 `#error` 挡住。
    六段核的 bf16 入口（§78）叠 K≠V、以及 `-DB2M_INTRA` 折叠叠 K≠V 未逐一复测（两侧机制正交，**待验证**）。
  - 未做 `ncu`（本轮是泛化/正确性收口，性能结论沿用 §64/§67）。
- **目标卡部署输入（不变）**：默认部署仍是 §45 的 `cf`（`dAv→CUDA-B1→融合 Triton B2/B3→cumsum`）；
  本轮的 K≠V 泛化对 `V=128` **零行为差异**（Part A 逐位），仅拓宽了研究核能表达的 head 配置。
