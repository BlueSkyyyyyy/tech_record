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
