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
  - FLA 的 host 税组成（Python 派发 vs Triton launch）尚未用 `nsys`/host 插桩拆开，
    只从「wall 恒定」推断为 host-bound；
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
被 K2 沿 T 顺序读一遍，流量上不去就被 L2 卡住。

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

**待验证**：(a) K1「global load 4/32 字节/sector」具体落在哪条 load（`beta`/`g`/标量参数？）；
(b) K2 的 L2 流量里 workspace 读与 $o$/state 写各占多少（需 `nsys` 或按指针插桩）；
(c) 把 K2 的 state 常驻寄存器/提高 smem 复用以突破 2 CTA/SM 后 L2 墙是否后移。
