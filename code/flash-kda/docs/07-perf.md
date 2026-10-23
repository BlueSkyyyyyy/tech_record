# KDA 性能与优化：前向的 L2 墙，反向的三颗 kernel

> 这是 KDA 专题的第 7 篇。前六篇把数学、K1/K2 前向、数值对拍、反向推导与反向 kernel 讲完了；
> 本篇只回答两件事：**FlashKDA 前向被什么卡住**，以及**反向为什么要自己写、三颗 kernel 又是
> 怎么一轮轮改出来的**。结论先说：前向不是算力问题，是 1.7–2.0GB 的 L2 流量问题；
> 反向的收益几乎全来自「不重算前向状态」+「三颗 kernel 各自换结构」。

## 0. 先记住一个画面

把一次算子调用想成**给一条流水线做体检**。

体检不问「工人手快不快」，只量三件事：

1. 每个工位**等料**等了多久（stall：`long_scoreboard`、`wait`、`barrier`）；
2. 传送带堵在哪一段（哪个 pipe 的 SOL 最高：L2、DRAM、L1TEX，还是 TensorCore）；
3. 仓库门口多宽（smem 预算决定能同时驻几个 CTA）。

前向的结论有点反直觉：**工人（TensorCore）常年闲着**——K1 只有 ≤1.6%、K2 只有 22–24%——
真正堵的是「把半成品搬进搬出仓库」，即 workspace 在 L2 上的往返。

反向暴露的是另一种病：B1 的 grid 只有 64 或 16 个 CTA，而卡有 132 个 SM，
一半以上的发射间隔在等 global load 回来——**不是算得慢，是料没到**。

> **类比的边界**：GPU 没有实体队列和传送带。所谓「等料」是记分牌计数，
> 「传送带」是 L2（H100 约 50MB）与它的带宽，「仓库」是每个 SM 的 228KB shared memory。
> 更要紧的是：体检报告（`ncu`）会给**错的建议**——本篇里它的 `Est. Local Speedup`
> 至少误报三次（81.25%、51.5%、44%），照着改反而更慢。
> 所以下文每个结论都要落到「改完实测多少」，而不是「报告说能快多少」。

读完这一篇，你应该能回答：

1. 为什么 FlashKDA 前向 wall 上快 2–21×，纯 GPU 时间却在中等 shape 上输给 FLA？
2. 前向那 1.7–2.0GB L2 流量具体由谁贡献，为什么把占用率抬上去也救不了？
3. 反向三颗 kernel 各自 bound 在什么上，几十轮试错里哪几步真兑现、哪几步是空结论？

环境统一为：H100 80GB HBM3、`bf16`、$D=128$、`CHUNK=16`、`torch 2.12.0a0`、
FLA `0.5.2`、`triton 3.6`，FLA 侧 `chunk_size=64`；所有计时都在 `CUDA_VISIBLE_DEVICES`
隔离出的空闲卡上做。

## 1. 计时口径：为什么必须同时看 wall 与 device

**先给结论：这篇每个「快多少」都至少有两个口径，混用会得出相反结论。**

| 口径 | 含义 | 采集方式 |
|---|---|---|
| `wall` | 单次调用端到端耗时，**含 host 派发**（Python/launch/`torch.empty`） | 一个 event 对包 `REPS=50` 次求均值，3 轮取中位 |
| `dev` | **纯 GPU kernel 时间**（不含 host 派发与调度空隙） | `torch.profiler`（CUPTI）CUDA-activity，`PROF_REPS=20` 求和除以次数 |
| `TF/s` | 用解析 FLOPs 模型算的有效算力 | 见 §1.3 |

`wall` 回答「用户等多久」，`dev` 回答「GPU 干了多少活」，两者之差就是 host 税。
后面会看到，FLA 前向的这项高达 ~0.8 ms。

### 1.1 坑一：连续 event 对的「单次中位数」只读到一半

第一版按最常见写法——每次调用前后各插一个 event、取 `median(elapsed_time)`——
量到的 FlashKDA 只有 CUPTI 的**一半**（T=1024 H=4：wall 115µs vs dev 220µs）。
用 `torch.mm(4096³)` 对照才定位：**背靠背的相邻 event 对时间戳会交叠**，
`median` 读到约真值的一半，`sum/N` 才和 CUPTI 一致。

`mm` 的实测：per-pair `median` **174.38µs**、`sum` **10233.25µs**（/30 = **341µs**），
而 CUPTI 口径 **339.57µs**——median 只有一半，sum/N 才对。

于是全部脚本改成**一个 event 对包住 N 次调用再除以 N**。
这条教训在后面反向与链级计时里反复出现。

### 1.2 坑二：不预热会把 JIT 编译算成 launch 开销

在 `process_time` 区间里连调 200 次**未预热**的 Triton kernel，量到 **3843 µs/次**——
那是首次 JIT 编译 + autotune 的几十毫秒摊到 200 次里。
warmup 10 次后同一 kernel 稳定在 **2.3–2.4 µs**。后文所有 host 数字都带 warmup。

### 1.3 坑三：跨实现比快慢，必须对齐 dtype / chunk / 归因窗口

这条代价最大。FLA 的 `dhu` 在 fp32 操作数下 **~127 µs**、bf16 下 **~23 µs**（B2 T1024 H8），
差 4–6 倍全来自 dtype。同一专题里还并存另外两条：孤立 kernel 归因比链内**偏高 ~1.3×**；
`ncu` 的 `Duration` 与 CUPTI `dev` 不同源、两者可以差 **≤1.3×**（同一变体 ncu 报 0.81×、
CUPTI 报 0.69× 这类出入都出现过）。所以后文凡出现 `xFLA` 比值，都写明 dtype 与窗口。

### 1.4 解析 FLOPs / bytes 模型只是个下界

每 chunk（$C=\lceil T/16\rceil$，并行度 $n=B\cdot H\cdot C$）的乘加：

```
K1:  L = k_dec·k^T              2·D·C_h·C_h        （C_h=16）
     INV 块内求逆                O(C_h³)，可忽略（~0.27 MFLOP/chunk）
K2:  q@s, k@s（各 D·C_h·D）     2 · 2·D·C_h·D
     U = INV·U                   2·C_h·C_h·C_h
     Mqk·U                       2·C_h·C_h·C_h
     s += k_rest^T·U             2·D·C_h·D
per_chunk ≈ 4·D·C_h·C_h + 4·D·C_h·D + 2·D·C_h·D
```

DRAM 下界（bf16=2B）：读 `q,k,v,g`（各 D）+ `beta`（1），写 `o`（D）：

```
bytes = (B·T·H·(4D+1) + B·T·H·D) · 2
```

这是**下界**：没算 workspace 读写、初/终态、K1→K2 的中间量落盘。
所以 `GB/s` 是「有效带宽下界比」、`TF/s` 是「按此模型的算力」，只用于横向比较。

## 2. 前向端到端：wall 赢在 host，dev 输在并行度

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

**观察一：FLA 的 wall 被 host 派发钉死在 ~0.86 ms。** 从 `T=128` 到 `B=8`，无论 workload 多大，
FLA 的 wall 几乎恒为 0.86–0.93 ms（`T=128` 那档 779.7µs 是例外，见上表），
而同期 `dev` 只有 58–211 µs——每次调用有 **~0.7–0.8 ms host 开销**。
`chunk_kda` 的 Python 包装把前向拆成十几个独立 torch/triton 调用，host 喂不满 GPU，
于是中小 shape 上 FLA 是 host-bound。FlashKDA 只有 **1 个 host 调用**
（`flash_kda.fwd` → 2 次 kernel launch + 一次 `torch.empty`），`wall ≈ dev`
（T=1024：112.8 vs 108.7；T=8192 H=64：1261.9 vs 1261.2，差 ~0.1%）。
**wall 上的 21.3× / 7.8× 主要是省掉了 FLA 的 host 派发税**，与 GPU 快慢无关，是「融合 + 单入口」的工程收益。（P7-1）

**观察二：纯 GPU 时间上，FLA 在中等 shape 反而更小。** `spd(dev)<1`（T=512…4096、H≤8）说明
单看 GPU 工作，FLA 的 Triton 前向更小（T=2048 H=4：122.9 vs 204.8µs）。原因在并行结构：
FLA 的状态递推 kernel `..._fwd_kernel_h_blockdim64` **沿 T 并行**（并行度 `B·H·(T/64)`）；
FlashKDA 的 K2 **沿 T 串行**（每个 `(b,h)` 一个 CTA，顺序扫过 `T/16` 个 chunk），并行度只有 `B·H`。
所以 K2 时间是 $\propto T$ 的直线：固定 `B=1,H=4` 时 dev 为 108.7 / 204.8 / 402.8 µs
（T=1024/2048/4096），斜率 **~100 ns/token** 恒定；固定 `T=1024` 扫 `H=1→16`，
dev 只从 106.2 涨到 126.0 µs——`H≤4` 时 `H` 个 CTA 填不满 GPU，加 CTA 几乎不加时间。
**此区间是 per-CTA 串行延迟受限，不是吞吐受限。**（P7-1）

**观察三：两个不同的「谁更快」区间。**
①小/中 shape（`B·H` 小、T 中等）：wall 大胜（2.1×–21.3×），但赢在 host。
②大 `B·H`（`big B8 T2048 H8`）：两边都 GPU-bound，dev 333.9 vs 741.3、
**dev 与 wall 同时 2.2×/2.6× 领先**——这时融合是真的 GPU 收益。
③大 T 小 `B·H`（`T=8192 H=16`）：dev 903.7 ≈ 717.0，wall 打平（**0.98×**）；
H 升到 64 后反超到 **1.84×**。这也解释了上游 benchmark 为什么选 `T=8192, H=64/96` 作设计点。

## 3. kernel 级分解：3 个 kernel 对 10 个 kernel

用 CUPTI 抓每次调用各 kernel 的 device 时间（`harness/bench_fwd.out.txt`）。

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

FlashKDA 只有 **3 个 kernel**：K2 占 89%、K1 占 8%，剩一个初始化 workspace 的 elementwise。
FLA 是 **10 个 kernel 的拼装**，最长单个 27.9µs，串起来 + host 派发就让 wall 远大于 FlashKDA。（P7-1）

### 3.2 大 shape（T=8192, H=64）：融合的收益一目了然

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

- 单比状态递推：K2 776µs，FLA h-kernel 352µs——**FLA 这个 kernel 更快**（它沿 T 并行）。
- 但 FLA 要把中间量一遍遍落盘/重读：`intra` 422 + `inter_solve` 424 + `recompute_w_u` 358 +
  `o` 340 + `l2norm` 177 + `cumsum` 147 + elementwise 63 ≈ **1.93 ms** 的非递推工作。
- FlashKDA 用 K1 的 **480µs** 一次备好 workspace，K2 内部消化其余。合计 1262 vs 2289µs → **1.81×**。

这正是 K1/K2 拆分的动机：K1 把 token 并行的重活（归一化/门控/衰减/块内求逆）一次算完并落盘，
K2 专心做 head 并行的递推。代价见下一节。（P7-1）

## 4. ncu：K1/K2 都不 bound 在 TensorCore

方法：warmup 3 次后 `ncu -k regex:... -c 1 --launch-skip 3` 只抓第 4 次调用；
`ncu` 会 replay 多次（本节指标 12–20 passes），replay 串行化并插 flush，
故 `Duration`（K2 T=1024 的 104.2µs）比 CUPTI `dev`（108.7µs）略小，趋势一致。GPU 132 SM。
`tensor%`/`fma%` 是**每活跃 SM** 的 `pct_of_peak_sustained_active`；
`L2%/DRAM%/L1TEX%` 是整卡口径的 memory-pipe SOL。

| shape | kernel | grid | dur(µs) | tensor% | fma% | L2% | DRAM% | L1TEX% | occ(th/ach)% | barrier | wait |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| T=1024 H=4 | K1 prepare | 256 | 11.55 | 0.63 | 5.12 | 30.1 | 8.2 | 22.3 | 100/24.2 | **11.9** | 1.9 |
| T=1024 H=4 | K2 recurrence | 4 | 104.2 | 21.6 | 9.7 | 3.0 | 1.4 | 58.6 | 18.8/9.3 | 0.3 | **1.1** |
| T=8192 H=64 | K1 prepare | 32768 | 481.1 | 1.54 | 12.5 | **88.2** | 52.1 | 57.3 | 100/98.4 | **27.0** | 1.9 |
| T=8192 H=64 | K2 recurrence | 64 | 781.2 | 22.7 | 10.2 | 46.5 | 27.5 | 63.4 | 18.8/9.4 | 0.2 | **1.1** |
| B=4 T=1024 H=64 | K2 recurrence | 256 | 207.1 | 24.1 | 10.8 | **91.6** | 53.0 | 59.5 | 18.8/18.1 | 0.3 | **1.2** |

smem bank conflict（ncu 归一到占 shared wavefront 的比例）：K1 45.0%（2.7-way）/51.6%（2.3-way）
（大 shape 47.5%/54.4%），K2 9.1%/44.9%，K2 填满 grid 时 4.6%/21.6%。

### 4.1 K1：TensorCore 几乎全闲，瓶颈是 L2 带宽 + barrier

**结论一：K1 的 TensorCore 通道全程 ≤1.6%，它不是矩阵乘瓶颈。**
里面的 $L=k_{dec}k^\top$、$M_{qk}=q_{dec}k^\top$ 两个 16×16 bf16 GEMM 与求逆后的
16×16 HMMA 合并（`fwd_kernel1.cuh:483/485/511`）只是 M=N=16 的小乘，
FLOPs 被每 tile 的 q/k/g/beta 搬运与 workspace 落盘淹没；FMA pipe 也只有 5–12%。

**结论二：放大到设计点后，K1 是 L2 带宽瓶颈。** T=8192 H=64 时 SOL 最大 pipe 是
**L2 = 88.2%**（DRAM 52.1%、L1TEX 57.3%），此时已满占用（achieved 98.4%，waves/SM=31）。
DRAM 读 405MB + 写 434MB、L2 流量 1.97GB——**写 workspace（13824B/tile）与读 q/k/v/g/beta 的
流式流量**把 L2 打满。这与「K1 一次算完落盘」是一体两面：省掉 launch/重复读写，
代价是全部中间量落到 L2/DRAM，于是它自己成了 L2-bound。

**结论三：barrier stall 是 K1 的第二瓶颈。** 每次发射平均 stall 11.9（小）/ **27.0**（大 shape）
周期（`ncu` 提示这占两次发射间隔 43.5 周期的 61%）。来源是密集的 `__syncthreads()`——
`fwd_kernel1.cuh:425/476/487/508/514` 一圈：读入寄存器 → 同步 → 覆写 union 复用后的 smem
（q/k/g → k_decayed/q_decayed/k_inv → L → M）→ 同步 → 求逆 → 同步 → 落盘。
`SharedStorageK1` 的 union 复用（`fwd_kernel1.cuh:45–83`）省 smem 的代价，
正是这些「必须先同步再覆写」的串行点；2.7-way/2.3-way bank conflict 再给 smem 加约 5 成 wavefront。

**结论四：小 shape 是「grid 太小」而非算法问题。** T=1024 H=4 时 K1 grid 只有 256 个 CTA
（$1\cdot4\cdot64$），/132 SM ≈ 2 CTA/SM，achieved occupancy 24.2%、`No Eligible` 85.3%，
纯延迟受限。`ncu` 同时报「global load 平均每 sector 只用 4/32 字节」——
这条告警后来被证伪，见 §4.3。（P7-2）

### 4.2 K2：不是 TensorCore bound；smem 限占用、L2 限吞吐

**结论一：TensorCore 只有 22–24%（每活跃 SM），从来不是瓶颈。** 它是 K2 里最忙的 compute pipe，
但离打满差 4 倍，FMA 仅 ~10%。大头是六个 Phase 里的 bf16 MMA + 大量 TMA/LDSM 搬运，
属**访存 + 依赖链**型。

**结论二：占用率被 shared memory 钉在 18.75%（2 CTA/SM）。** LaunchStats 显示
Dynamic Shared Memory Per Block **98.43 KB**（union 后的 `SharedStorageK2`），每 SM 最多 2 个。
而 K2 grid = $B\cdot H$：设计点 T=8192 H=64 只有 **64 个 CTA < 132 SM**，
achieved occupancy 9.4%，`ncu` 直接报「grid 64 blocks < 132 SMs，估 51.5% 局部加速空间」。

**结论三：grid 一填满，K2 立刻变成 L2 带宽瓶颈。** 把 $B\cdot H$ 抬到 256
（B=4,H=64,T=1024）后 achieved occupancy 18.1% ≈ 理论值、waves/SM≈1，
此时 SOL 最大 pipe 是 **L2 = 91.6%**（DRAM 53.0%、L1TEX 59.5%），TensorCore 仍只有 24.1%。
即稳态上限由**每 chunk 从 L2 读 workspace（kd/qd/kr/gt/INV/Mqk，13824B/tile）** 决定。

**结论四：K2 的 stall 以 `wait` 为主（~1.1 周期/发射），不是 barrier。** 与 K1 相反，
K2 几乎不 `__syncthreads`（warp-specialized，只有跨 MMA warp 的 `NamedBarrier` 与流水线 barrier），
barrier stall 只有 0.2–0.3；主导的 `wait` 是**固定延迟依赖**（等前一条 MMA/LDSM 的结果），
加上 occupancy 太低、可切换 warp 不足（`Eligible Warps Per Scheduler` 0.38–0.50），盖不住延迟。（P7-2）

### 4.3 两条被证伪的读数：占用率 81.25%，和「4/32 bytes/sector」

`ncu` 的 Occupancy 段对 K2 报「Block Limit Shared Mem 2、Theoretical 18.75%、
**Est. Local Speedup: 81.25%**、limited by shared memory」。这很诱人：多驻一个 CTA 是否就能多 81%？
把它做成受控实验（`harness/_build_k2_occ.py`、`bench_k2_occ.py`、`probe_k2_occ_ncu.sh`）。

先算 smem 预算（`fwd_kernel2.cuh:72–112`）：

```
sizeof = 32768 + max(IN*17984 + OUT*4096, 65536) + ~128
#        state_acc(bf16 D×D)   ↑臂一: IN 输入+OUT 输出 stage   ↑臂二: fp32 状态缓冲, 无条件预留
```

每输入 stage $=v(4096)+beta(64)+3\times4096+\text{g\_total}(512)+2\times512=17984$ B，每输出 4096 B。
关键在臂二：`state_fp32_buf`（`fwd_kernel2.cuh:106`）只在 `StateFP32=true` 时用，
但 bf16 路径也**原样预留**——所以只要 $IN\cdot17984+OUT\cdot4096\le65536$，
`sizeof` 恒为 **98432 B，与 stage 数无关**。实测（`cond` = bf16 下发 `ReserveFP32 ? [D*D*4] : [1]`）：

| 变体 | `sizeof`(B) | CTA/SM | 64 CTA µs | 512 CTA µs | 512 CTA 的 L2 SOL |
|---|--:|--:|--:|--:|--:|
| orig = plain (3,2) | **98432** | 2 | 801 | **773** | **94.0%** |
| plain (2,1)（只减 stage） | **98432** | 2 | 929 | 774 | 93.1% |
| cond (3,2) | 95232 | 2 | 800 | 772 | — |
| cond (2,2) | 77184 | 2 | **793** | 775 | 93.2% |
| cond (2,1) | 73088 | **3** | 931 | **885** | **81.4%** |
| cond (1,2) | 59136 | **3** | 1468 | 1086 | — |

三读数：

1. **只减 stage 不动任何字节；去掉 64KB 死缓冲也只省 3200 B**，仍是 2 CTA/SM；
   要摸到 3 CTA/SM 必须把 IN 或 OUT 砍到 1。
2. **2→3 CTA/SM 没换来任何加速**。`cond(2,2)` 与 orig 同速；唯一真进 3/SM 的 `cond(2,1)`
   反而**慢 14%**（885 vs 773µs），**L2 SOL 掉到 81.4%**——多驻的 CTA 只在抢同一份 L2 带宽。
3. **占用率提升必须靠砍 pipeline 深度**，而深度才是命门：`cond(1,2)` 512 CTA 1086µs（1.4×）、
   64 CTA 1468µs（1.8×）。上游 K2 已贴着 L2 上限跑（`lts` 93–94%、`dram` 55–56%、`sm` 34%、
   `sm__warps_active` 18.6%），再多 CTA 也没有下一层可喂。所有变体前向输出与上游**逐位相同**。

**那句 81.25% 是纯启发式**：它假设「占用率越高越快」，无视 L2 已 94% SOL。
正确读法是：`Block Limit Registers=4`、`Warps=10`、`Barriers=7` 都远高于 `Shared Mem=2`，
smem 确实是**占用率**的瓶颈，但**占用率不是性能的瓶颈**。**L2 墙不会后移**。（P7-5）

第二条例外：K1 那条「4/32 bytes/sector 未合并、估 50% 局部加速空间」。
反汇编装好的 `.so`（`cuobjdump -sass`）按函数统计标量 `LDG.E` 与 TMA `UTMALDG`：
`IsVarlen=false` 时 **1 / 5**，`varlen=true` 时 **5 / 5**——5 条 `UTMALDG` 正是
q/k/g/beta/dt_bias 五个描述符（`fwd_kernel1.cuh:231/233/235/243/253`）。
非 varlen 的唯一标量 load 前驱是 `S2R SR_CTAID.Y` → `LDC.64 A_log_ptr` → `IMAD.WIDE head_idx*4`，
即 `fwd_kernel1.cuh:258` 的 `expf(A_log_ptr[head_idx])`。`SourceCounters` 里它独占
Global load 一行：2048 条指令、`L2 Sectors Global` 2048、`Sectors Ideal` **2048**、`Excessive` **0**
（`B1 T1024 H4` 时 256 CTA × 8 warp = 2048 warp，每 warp 恰好一条）。
**为什么 4/32 且无害**：该地址对 warp 内 32 lane 完全相同，硬件按**广播**处理——
一条指令只碰 1 个 32B sector，32 lane 合用其中 4 字节。`4.0/32` 是「每线程利用字节」的误报；
有意义的余量是 `Excessive`，而它是 0。字节预算：0.06MB / K1 DRAM 读 3.18MB = **1.97%**；
设计点 8.00MB / 405.02MB = **1.98%**；而 K1 只占总前向约 8%，端到端收益 **<0.2%**。
**勘误**：原文记的「估 50% 局部加速空间」对不上这条告警（紧随的 `Est. Local Speedup: 44.94%`
属 shared-load bank conflict），且与 `Excessive=0` 矛盾，应视为误记。
§4.1 的两条结论（**barrier stall + L2 带宽**）不变。（P7-4）

## 5. 访存账：那 1.7–2.0GB 的 L2 流量从哪来

**一句结论：workspace 是 K2 读流量的 ~75%，`v` 是不可省的另外 ~22.5%。**

### 5.1 为什么用 DRAM 计数而不是 L2 sector 计数

`ncu` 默认每次 replay 前**清空 L2**，所以一次调用的 `dram__bytes_read/write` 就是它
**唯一的、去重后的访存字节**，不受 L1/TMA 重发、swizzle、缓存命中干扰。
相比之下 `lts__t_sectors_*` 对 TMA 有系统性放大：实测 K1 的 `lts` 读是 DRAM 读的 **1.5×**、
`lts` 写是 workspace 的 **~3×**（同一份 workspace 被 TMA store 重复计数）。
**`lts` 只能当同 kernel 内的相对量，不能当字节数。**

预测模型（$D=128$，每 tile workspace $=13824$ B，`utils.cuh:70–76`）：K1 读 `q,k,g,beta`
（**不读 v**）、写 workspace $=B\cdot H\cdot\lceil T/16\rceil\cdot13824$；
K2 读 workspace + `v` + `beta` + 初态 `h0`，写 `o` + 终态。

### 5.2 读侧：四组 shape 全落在 1.000–1.010

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

归因是硬的：

1. **K1 唯一的外部输入是 `q,k,g,beta`**（$3\times BTHD+BTH$），**从不读 `v`**——
   与源码一致（没有 v 的描述符）。
2. **K2 的读里 workspace 只占 ~75%，`v` 占 ~22.5%**。这不是小项：K2 每 chunk 还要
   `tma_load_v`/`tma_load_beta`（`fwd_kernel2.cuh:325/326, 358/367`）来算 $u=(v-u)\beta$。
   所以 §4.2「稳态上限由读 workspace 决定」**需修正**：workspace 是大头但不是全部，
   **`v` 的 22.5% 是不可省的输入流**，grid 一填满就是两条流一起压。
3. `beta` 几乎免费（0.2%），`h0` 随 $B\cdot H/T$ 变化（`1×1024×64` 因 $BH$ 大、$T$ 小升到 2.8%，
   设计点 0.4%）。

### 5.3 写侧：小 shape 的 DRAM 写是 0，这是测量上的必然

| shape (B,T,H) | kernel | DRAM 写 MB | 预测 MB | ratio | 归因 |
|---|--:|--:|--:|--:|---|
| 1,1024,4 | K1 | **0.00** | 3.54 | 0.00 | workspace 3.54MB ≪ L2(50MB)，dirty 行留 L2 |
| 1,1024,4 | K2 | **0.00** | 1.18 | 0.00 | `o`+state 1.18MB 同样留 L2 |
| 4,1024,64 | K1 | 208.16 | 226.49 | 0.919 | workspace 226MB > L2，约 8% 仍滞留 |
| 4,1024,64 | K2 | 65.10 | 75.50 | 0.862 | — |
| 1,8192,64 | K1 | 434.65 | 452.98 | 0.960 | — |
| 1,8192,64 | K2 | 130.19 | 136.31 | 0.955 | — |
| 8,2048,8 | K1 | 94.59 | 113.25 | 0.835 | — |
| 8,2048,8 | K2 | 29.38 | 35.65 | 0.824 | — |

写出的字节先变 dirty 行，kernel 结束时还没被换出，直到下一次调用或 flush 才落 DRAM。
放大到 workspace ≫ L2 后，绝大部分在同一次调用内被换出，`ratio` 升到 0.82–0.96，
与预测同量级——**写的归因同样成立**：K1 写的就是 workspace（100%），K2 写的是
`o`（94–98.5%）+ 终态（1.5–6%）。（P7-3）

### 5.4 把账合起来

- 设计点 K1 读 405MB + 写 435MB，K2 读 590MB + 写 130MB，合计约 **1.56GB** 外部流量——
  与 §4.1/§4.2 用 `lts` 报的 1.7–2.0GB 同量级（差额即 TMA 的 `lts` 重发放大）。
- 两条方法学：**TMA 密集 kernel 不要用 `lts__t_sectors` 当字节**（读 ×1.5、写 ×3）；
  小 shape 的 DRAM 写会因 dirty 行滞留读数为 0，须放大到超 L2 才有意义。
- 对优化的含义：前向真正的硬成本是 **D×D 状态与 13824B/tile workspace 的搬运，不是乘法**。
  这条判断在反向的每一轮里都会再出现。

### 5.5 FLA 那 0.8ms host 税花在哪

主口径换成 **`time.process_time()`**（单线程进程里等于真实 CPU 时间，不受 GPU 队列影响）：
用纯 Python 忙循环校验 `process_time≡wall`（259 vs 259 ms），用大 kernel 连发校验无后台线程
（6.4 µs/次）；另用 `nsys profile --capture-range=cudaProfilerApi` 圈稳态区间。

**FLA 的 host CPU 是固定 ~0.82–0.86 ms/次**（`B=1,H=4` 扫 T=16…2048）：
821.5 / 827.8 / 857.8 / 843.2 / 862.2 µs；FlashKDA 对照 34.6 / 34.2 / 35.2 / 33.5 / 33.5 µs
——两者都**与 T 无关**，相差 **~24×**。T=16 时 GPU 只要几微秒，FLA 仍烧 0.8ms CPU。

分解（`tt=1024`，µs/次，`FLA_DISABLE_BACKEND_DISPATCH` 前后）：

| 组成 | dispatch ON | dispatch OFF | 说明 |
|---|---|---|---|
| `chunk_kda_fwd` | 529 | 409 | 递推 + 一堆 Triton kernel 的 launch |
| `l2norm_fwd`（×2） | 154（≈77/次） | 130（≈65/次） | GPU 端 ~1.6 µs，**host 端 77 µs** |
| `fused_beta_sigmoid` | 63 | 45 | 一个逐元素 kernel |
| autograd/`dispatch`/`input_guard` 等 | ~95 | ~70 | `ChunkKDAFunction.forward` 自身 Python |
| **合计** | **~841** | **~654** | |

后端派发占约 **22%**（~190 µs），且**不是**一次性初始化——`dispatch` 的 wrapper 每次调用都重跑。
单个 Triton kernel host 成本 ≈ **77 µs**，裸 launch 只要 **2.4 µs**、裸 `torch.empty` 只要 **0.07 µs**；
多出来的几十 µs 全是 FLA 在 Triton 之上叠的每调用 Python 包装（`fla_cache_autotune`、
`AutotuneKey.build` 的 dtype 序列化等）。

nsys 稳态（capture-range，N=200，driver API 取中位，因首个调用含 lazy 初始化 2.5ms 长尾）：
**FLA 发 12 个 kernel/次，FlashKDA 3 个**。FLA `B1 T1024 H4` 的 driver launch 中位 ~6.3 µs/次、
稳态 GPU kernel 合计 ~82 µs/次；FlashKDA ~5.6 µs/次、~109 µs/次。
`B8 T2048 H8` 两组分别是 ~5.4–5.8 µs / ~728 µs 与 ~5.5 µs / ~332 µs。

于是 0.8ms 的账很清楚：**纯 driver 侧只需 ~75 µs**（12 × ~6.3 µs）、GPU 计算 ~82 µs，
剩下 **~680 µs 全是 Python 里、每 kernel 一次地烧掉的 launch/包装开销**。
FlashKDA 只有 3 个 launch、host 总成本 ~34 µs，完全被 GPU 时间盖住。（P7-6）

**可迁移的经验**：逐 kernel 的 Python launch 开销，是 Triton 类实现做小算子融合时最容易被
忽视的成本；「每个 chunk 一次小 kernel」的算法尤其敏感。

## 6. 反向：为什么要自己写

**一句结论：FLA 的反向要在反向里重算前向状态，那部分占它 device 时间的 17.8–21.9%；
我们直接吃 K1 落盘的 workspace，一块都不用重算。**

### 6.1 口径：怎么只计「反向」

`wall` = 一个 event 对包 `REPS=30` 次取均值、3 轮取中位；`dev` = CUPTI 各 kernel self time 求和
除以调用数；`host = wall − dev`。**只计反向**：FLA 侧在区间外建好 autograd 图，
区间内只跑 `loss.backward(retain_graph=True)`；ours 侧把 K1 workspace 与 `fi` 在区间外预先重建，
区间内只跑反向链的 closure。

### 6.2 主表：四组 shape（空闲卡）

单位 µs/次。`#k` = 一次反向发出的 CUDA kernel 数。

| case（B T H） | ours wall | ours dev | ours host | FLA wall | FLA dev | FLA host | w-o/FLA | d-o/FLA | #k ours/FLA |
|---|---|---|---|---|---|---|---|---|---|
| 1 512 4 | 316.5 | 162.6 | 153.9 | 800.3 | 202.8 | 597.5 | **0.40** | **0.80** | 40 / 28 |
| 2 1024 8 | 518.0 | 483.0 | 35.0 | 877.9 | 652.3 | 225.6 | **0.59** | **0.74** | 40 / 28 |
| 8 2048 8 | 2551.8 | 2515.1 | 36.7 | 4135.5 | 4108.8 | 26.7 | **0.62** | **0.61** | 40 / 28 |
| 1 250 3（非整除 T） | 372.5 | 97.1 | 275.4 | 857.0 | 144.0 | 713.0 | **0.43** | **0.67** | 40 / 28 |

1. **我们的链在 wall 与 dev 上都快过 FLA 反向**（wall 0.40–0.62×、dev 0.61–0.80×），
   尽管其中的 Triton 反向 kernel 都是**朴素实现**。
2. **host 税量级**：FLA 反向 host 在小 shape 上是 **597.5 / 713.0 µs/次**（T512H4 / T250H3），
   大 shape 被 4108.8µs 的 GPU 时间藏住 → `host≈27 µs`；我们那条链 **35–275 µs/次**。
   ours 一次反向发 **40 个 kernel**（含十来条 torch elementwise 兜底），FLA 28 个——
   但两者 host 都远小于 FLA *前向* 的 ~0.8ms 固定税，因为反向 GPU 时间更长、队列藏得住 host。
3. 端到端（同脚本）：FlashKDA `fwd` dev = 63.5 / 128.2 / 668.7 / 42.9 µs（3 kernel）；
   `fwd+chain` 合计 = 226.5 / 1101.0 / 6182.1 / 140.5 µs；
   FLA `chunk_kda` **fwd+bwd** = 325.1 / 1907.9 / 12635.0 / 250.2 µs，即我们也快 **1.4–2.0× dev**。

### 6.3 归因：FLA 重算了前向状态，我们复用了 workspace

`B2 T=1024 H=8` 逐 kernel 分解（CUPTI，µs/次）：

| ours chain | µs | | FLA backward | µs |
|---|---|---|---|---|
| `_dhu_kernel`（B1） | **214.3** | | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（前向重算） | **83.6** |
| `_wy_dqkg_kernel`（B2） | 126.2 | | `chunk_kda_bwd_kernel_wy_dqkg_fused`（B2） | 237.8 |
| `_intra_kernel`（B3） | 81.0 | | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`（B1） | **137.4** |
| `_dav_kernel`（dAv） | 11.3 | | `chunk_kda_bwd_kernel_intra`（B3） | 120.3 |
| `chunk_local_cumsum`（B0 收尾） | 6.0 | | `chunk_kda_bwd_kernel_dAv` | 18.5 |
| torch elementwise（各支路相加等） | ~45 | | `recompute_w_u_fwd_kda_kernel`（前向重算） | 32.8 |
| **合计** | **483.0** | | **合计** | **652.3** |

FLA 的 profile 里出现了前向的状态 kernel（83.6µs）与 `recompute_w_u`（32.8µs）
——**它在反向里重算了 chunk 状态与 `w/u`**，合计 116.4µs = **FLA 反向 dev 的 17.8%**
（`B1 T512 H4` 时 (34.6+9.9)/202.8 = **21.9%**）。我们的链**从不重算前向**：
六个 per-tile 字段（$k^{dec},q^{dec},k^{rest},E^{tot},P,M_{qk}$）直接从 K1 workspace 读，
`U/h` 由它们逐块重放——只在计时区间外做一次。
这正好把「复用 K1 workspace」的设计收益从**正确性**兑现成了**性能**。（P7-7）

于是反向的起点很清楚：B1 是最大单 kernel（`B2T1024H8` **214.3/483.0 = 44.4%**，
`T512H4` **102.0/162.6 = 62.7%**），而 B2（126.2 vs 237.8）与 B3（81.0 vs 120.3）已经领先。
说明朴素实现在块内小矩阵上吃亏不大，**跨块状态递推的并行/流水才是要害**。
这也定下三颗 kernel 的分工：B1 打流水、B2 打带宽与寄存器、B3 打「没有 TensorCore 的标量数学」。（P7-7）

### 6.4 一次被 GPU 竞争污染的测量

第一版在**未隔离**的卡上测：`B2 T1024 H8` 的 ours dev 报 972.8、FLA dev 报 1322.8，
FLA 前向重算报 **578.0**、ours B1 报 **539.8**。钉到空闲卡重跑后：

| case（B T H） | ours dev | FLA dev | ours wall | FLA wall | d-o/FLA |
|---|---|---|---|---|---|
| 1 512 4 | 163.0 → 162.6 | 207.2 → 202.8 | 349.6 → 316.5 | 703.5 → 800.3 | 0.79 → 0.80 |
| 2 1024 8 | 972.8 → **483.0** | 1322.8 → **652.3** | 1085.7 → 518.0 | 1419.9 → 877.9 | 0.74 → 0.74 |
| 8 2048 8 | 5513.4 → **2515.1** | 9119.6 → **4108.8** | 5630.7 → 2551.8 | 9197.3 → 4135.5 | 0.60 → 0.61 |
| 1 250 3 | 97.6 → 97.1 | 143.9 → 144.0 | 355.5 → 372.5 | 872.4 → 857.0 | 0.68 → 0.67 |

**比值侥幸没变**，但绝对值整体低约 2 倍。被污染的恰恰是关键证据那两行：ours B1 539.8 → **214.3**，
FLA 前向重算 578.0 → **83.6**（`B1 T512 H4` 只要 34.6，4× 工作量报 578/34.6≈17× 本就反常）。
于是「重算占 FLA 反向 dev 的 46.2%」必须改成 **17.8%**（`T512H4` 21.9%）。

同一条链上还叠着两个变量：**chunk 粒度**（我们吃 `CHUNK=16`，FLA 的 `dhu` 是 `chunk_size=64`）
与**操作数 dtype**（早期 `reconstruct` 把整个 workspace `.float()`，`tl.dot` 落到 tf32；FLA 是 bf16）。
对齐后（`check_bwd_dhu_dtype.py`，同一份操作数，扫 {ours,FLA} × {fp32/tf32, bf16} × {BT=32,64}），
**同一 BT、同一 dtype 下我们的 B1 只比 FLA 慢 1.1–1.9×**（不是 3.9×）：

| shape | ours32 | FLA32 | o/F | ours64 | FLA64 | o/F | dtype |
|---|---|---|---|---|---|---|---|
| B1 T512 H4 | 70.6 | 45.6 | 1.55 | 65.6 | 53.2 | 1.23 | fp32 |
| B2 T1024 H8 | 176.9 | 107.1 | 1.65 | 147.5 | 126.3 | 1.17 | fp32 |
| B8 T2048 H8 | 589.1 | 482.0 | 1.22 | 593.0 | 523.8 | 1.13 | fp32 |
| B1 T512 H4 | 25.5 | 20.5 | 1.24 | 17.6 | 13.0 | 1.35 | bf16 |
| B2 T1024 H8 | 73.5 | 37.8 | 1.95 | 41.8 | 23.0 | 1.81 | bf16 |
| B8 T2048 H8 | 210.3 | 169.9 | 1.24 | 161.5 | 120.2 | 1.34 | bf16 |

两个读数：**chunk 粒度是真实差距**（BT=16 比 BT=64 多约 1.3–1.6×，串行步数 ×4）；
**同一 BT 下只剩一个 1.1–1.9× 的常数因子**（我们是单次 `tl.dot` 到 `[K,BV]` 整块状态，
没有 FLA 的 K-64 分块与 `num_stages` 流水）。
至于「两个 FLA `dhu` baseline」——fp32 下 ~127µs、bf16 下 ~23µs，**两个都是真的**，
差全在 dtype（fp32 载入 2× + `tl.dot` 落 tf32）。
所以此后引用 FLA 反向绝对时间都写明 dtype，凡 `xFLA` 一律以 matched-dtype 为准。（P7-8、P7-34）

顺带一个零算法改动的杠杆：workspace 六个字段里 `kd/qd/kr/INV/Mqk` **本来就是 bf16**
（只有 $E^{tot}$ 是 fp32），早期为对拍整套 `.float()` 反把 B1 推进最慢的 tf32 路径。
保持 bf16 再测：`T512H4` 91.3 → **37.2 µs（2.45×）**、`B2T1024H8` 215.0 → **97.9 µs（2.20×）**，
`dh/dv2` max_abs 6.1e-5–2.4e-4、`dh0` 7.5e-9–1.7e-7。这是整条反向第一个真正的收益项。（P7-8）

## 7. 三条优化主线：B1、B2、B3

### 7.1 B1：从「慢 2.14×」到「同档」

**问题**：`ncu` 说它**不是任何单元的 bound**。tensor pipe 只 **4.5–6.8%**（FLA 14.2–14.9），
DRAM 最高 28.6%、L2 最高 39.8%；achieved occupancy 三种形态一律 **~6.2%**，
因为 grid 只有 **64 个 block（B2H8）或 16 个（B1H4）**，而卡有 132 SM
（`ncu`：「0.48 wave / 0.12 wave，Est. Local Speedup 51.5%」）。
真正的病是**等料**：每次发射要等 **8.19（BT16）/9.68（BT64）** cycle，
其中 `long_scoreboard` 占 **3.88/5.29（≈47%/55%）**，FLA 只有 **1.06/5.06 ≈ 21%**；
同 BT=64 下 ours 53.82 vs FLA 25.15（**2.14×**）。根因是 `num_stages=1`
（`check_bwd_dhu_kernel.py:214`），5 个 `tl.load` 与紧随的 `tl.dot` 之间没有预取重叠。（P7-13）

**改法一：Triton 软件流水（`tl.range`，ns∈{2,3}）**。数字：`long_sb` 5.27 → 1.46（ns=2）→
**0.96（ns=3）**，正好落到 FLA 的 1.06；`tensor%` 6.8→15.9（FLA 14.9）、DRAM 13.5→30.6%（28.6%）；
smem 65.5 → **181.3 KB**（FLA 180.2），理论占用率 12.5%→**6.25%**，却最快；
ours B1 相对 FLA `dhu` 的 **1.96× → 1.00×/0.85×**。
同一轮把「把 D=128 状态拆成 2 条累加链」（FLA 的 K 两分做法）证伪：`splitK ns=1` 的
`long_sb=6.88` 比 base 的 5.27 还高，**拆链单独负收益、叠加更慢**。
**结论：流水的收益全部来自「把料提前搬进来」，与 ILP 拆分无关。**（P7-14）

**改法二：手写 CUDA（裸 wmma + `cp.async`）——先证明它没用**。加 preload 只值 **2–6%**，
因为结构还不够「K2 形状」：`ncu` 显示 **76% 的 shared-load wavefront 是 bank conflict**
（FLA 用 `ldmatrix` 只有 0.8%）、tensor 只 2.3%（FLA 14%）、`SYNC` 里 39% 在等 `cp.async`。
补 smem padding 后冲突 76%→**~19%**，端到端却只快 **10–23%**（BT=64 甚至 **−21%**，因为 padding
让 smem **+29%** 反噬占用率）；补 ILP 只有 **1.1–1.2×**。
**结论：bank conflict 与 ILP 都是「分项」，不是量级项；必须换结构。**（P7-15、P7-16、P7-17、P7-18）

**改法三：改成 K2 形状**——寄存器常驻状态 + `mma.sync`/`ldmatrix` + XOR swizzle。
数字：**2.4×**（BT=16 从 214.4 → **81.8µs**），因为状态不再每步过 smem、
`ldmatrix` 替掉 4 路标量 LDS。（P7-19）

**改法四：在 K2 形状之上补 TMA + warp specialization**。数字：`xFLA` 到 **1.63**
（`-DB1T_ETMA`、标量 `dh`/`dv2` store），残差是 `wait` **1.24**（FLA 0.96）、
`short_sb` **0.90**（FLA 0.50）、`long_sb` **3.86**（FLA 2.07）——即「每 warp 发射效率」，
不是访存。把 per-chunk 的 $E^{tot}$ 也塞进流水后，残差只剩这一块。（P7-20、P7-21）

**改法五：把两条逐标量 `st.global` 换成 TMA store。** 先量化上界：删掉 `dh` 落盘能让
B1+B2 快约 **10%**（其中 **2/3 在 B1 侧**，约 **33%** 的 B1 时间、65µs 量级）。
`dh` 改 TMA store 后**逐位不变**、恢复「删掉 store」收益的 **58–84%（均值 ~72%）**；
再把同一逐标量循环里的 `dv2` 也交给 TMA：**再快 5–16%（均值 ~11%）**、L2 再降 **15%**、
`lg_throttle` 归零，两条合起来恢复上界的 **62–83%（均值 75%）**，代价 smem +32768 B（占用率 7.77%）。
**结论：这一步独立、低风险、逐位无损，是反向里性价比最高的一刀。**（P7-31、P7-32、P7-33）

**收口：matched-bf16 下的诚实定位。** 把上面五步的收益收口成同 dtype 的 `xFLA`：
大 shape `B8T2048H8` 在 **NS=1 时 1.15×、NS=2 时 0.83×**，小 shape **2.0–2.7×**；
整链 ~0.86–0.95×。同时暴露并修掉一个真 bug：`NSTAGE≥2` 时手写 TMA 流水有**竞态**，
根因是缺一条 **`fence.proxy.async`**（generic proxy 写 smem 后没告诉 async proxy），
补上后 9 个变体各连跑 40 次逐位一致（`proxy` 档 0/40），NS=4 白拿 **1.6–1.7×**。
一个**空结论**也要记：按建议换成 CUTLASS `PipelineTmaAsync` 后是 **0.985–1.003×**，
`wait` 仍 ≈1.4——手写 mbarrier 已在该结构上限。（P7-35、P7-36、P7-37、P7-38、P7-39、P7-41）

**这条主线的结论**：B1 的收益排序是「流水 ≫ 结构（K2 形状）> store 引擎 > bank conflict ≈ ILP」。
反过来说，凡是先动 bank conflict/ILP 的尝试都只值 1.1–1.2×。（P7-13…P7-42）

### 7.2 B2：从「DRAM-bound 的 66.6%」到六段生产核

**问题**：链级归因显示大 shape 上 **B2+B3 = 66.6%**（B8T2048H8：B2 906.0 + B3 579.7 / 2232.0），
B1 只剩 13.8%。B2 的体检：**DRAM 59.0% SOL**、L1TEX 36.1%、tensor 4.88%，
**247 个寄存器**把占用率压到 12.4%，`long_sb` 6.6。（P7-23）

**先扫免费旋钮——已用尽**：`num_stages` 在 B2/B3 上**完全无效**（±0.5%，因为它们只有
2–16 次的短归约，自动流水器识别不出可重叠的 load）；tiling 上 `BK=BV=64` 最好（900µs）、
`BK=128` → 2109µs、`BK=BV=128` → **37.8ms**（寄存器爆炸）；B3 的 `BC=16` 最好（`BC=32`→2022µs）。
唯一免费信号是 B3 的 `num_warps=2`（1.16–1.27×，链 −2–6%，逐位不变）。（P7-23）

**改法一：延后 `q/k/g` 加载，先消寄存器溢出。** spill 12→0、`nw=8` 寄存器 254→224；
孤立快 **0.84–0.89×**，链内 0.95–0.97×（B8 1961→1862µs）。
同轮把 `h/dh` 换 bf16（**−10%**）、`i_k` 去重（≤16%，被吸收）——都是「分项」，
且 bf16 需要原生入口才有意义（见后）。（P7-25、P7-26）

**改法二：把它改写成 K2 形状的 CUDA 核（mma.sync + 流式操作数）。**
先算头寸：ncu 的字节下界给 **~1.8×**、内存地板给 **~1.9×**；但**直接写 SIMT 是反向方向**：
算术强度只有 0.9 op/word，`h/dh` 占 1.07GB = 84% 流量，实测慢 **3–5×**。
换成「保留 `mma.sync` + 操作数流式过 smem」后**首次正面**：**0.90–0.93×** vs Triton，
寄存器 226→**70**、占用率 12.4%→**24.6%**、DRAM 84.6% SOL、L2 88.8%。
（P7-43、P7-44、P7-45、P7-46）

**改法三：把读放大和写放大分别消掉。** 两个反直觉的定位：
① **8.81M bank conflict 不是 mma 片段读，而是 `dgk` 的列归约**——改成 warp 内列交换后
冲突降 96%，但部署 shape 上**收益为零**（只小 shape 3–4%）；
② **L2 字节 +35%**（3.03 vs 2.24GB）= `cp.async` 的 **L1-bypass 读路径 +0.64GB**
+ 逐标量 32-bit store **+0.13GB**（×1.98 系数吻合）；向量化 store 把系数压回 1.0×
（小 shape 10–13%），而 TMA 输入让**整条读放大消失**、净赢 **5%**（2.25 vs 2.23GB、491µs、
1.12× Triton、DRAM 89.6%，距地板 442µs 只差 1.11×）。
`BVK=64` 用两个 128 B swizzle atom 拆分是全程最快（485.9µs、DRAM 90.6%）；
`SWIZZLE_NONE` 捷径**正确但慢 2.6–2.8×**；`BVK=128` **无收益**。（P7-47…P7-55、P7-73、P7-74）

**改法四：把非 GEMM 六段逐个搬进 CUDA。** 代价依次是 `dv2` **+12–25%**、
`dkgb` **+10–13%**（均落 tf32 底噪）、`dApre` **+33–61%**（标量 FMA，后换 mma 才降下来）、
`sand` **+2–10%**、`db` **+2–13%**、门控链 **+30–38%**。
合起来是**一次 launch**（从 6 个独立 build 变成 1 个核）：**0.47–0.53×**、
零 spill、DRAM SOL 45%、tensor 3%。
`dw` 同 CTA 交接把寄存器 140→110、DRAM −126MB，但**端到端是 wash（+2.8%）**；
DAMMA（`dA` 段 mma 化）在核内 **+5–7%**、链内 1–2.5%。（P7-52…P7-60、P7-63、P7-64、P7-69）

**改法五：bf16 原生入口 + 原生 bf16 mma。** 五个 streamed 操作数改 bf16 staging 后
与 fp32 核**逐位相同**、大 shape **0.73×**，机制是**五操作数 DRAM 读精确减半**
（1.28 GB → **637.57 MB**，逐值吻合；`t512h4` 19.94→9.98MB、L2 48.08→24.23MB、12.51→10.30µs）；
扩到六段生产核并落链是 **0.70–0.93×**，占用率 1→2 block/SM。
原生 bf16 `m16n8k16` mma 是**免费的 ISA 换代**（`HMMA.1688.F32.TF32` → `HMMA.16816.F32.BF16`，
mma 指令精确减半、tensor 周期减半），但大 shape 只 **0.99×**（已 L2/DRAM bound），
收益落在**短序列**（≤512 token）的 **0.89–0.91×**。（P7-73、P7-74、P7-78）

**收口：结构闭合 ≠ 性能净赢。** 六段全 CUDA 的生产核 vs 融合 Triton B2/B3，
在真实链里仍**慢 ~1.6–1.7×**（`B8T2048H8`：融合 Triton **998µs** vs 单 launch 生产核 **1606.9µs**）；
原因是 Triton 融合版把 `dq/dk/dg` 累加器留在寄存器、
只在最后写一次，而六段版为「可独立核验」把中间量落盘再读。
所以**默认部署仍取 `cf` = CUDA B1 + 融合 Triton B2/B3**，生产核留作 `pci` 研究形态。（P7-67、P7-68、P7-70、P7-71、P7-74）

### 7.3 B3：一个「没有 TensorCore」的核

**问题**：B3 `intra` 的体检最极端：**tensor pipe 0.05%**、Compute(SM) **82.3%**、
L1TEX **70.9%**、DRAM 28.0%——它是纯 FMA/L1 的标量核。唯一的免费旋钮是 `num_warps=2`
（1.16–1.27×）。（P7-23）

**改法一：把对角块 MMA 化。** 局部 `C` 与 `D` 两个矩阵**同一个 `ldmatrix` 读、同一份 smem**，
MMA 版在同一个循环里算完两者，省掉重复加载。数字：孤立快 **1.5–2.0×**，链 dev **−9–12%**；
L2 43%→**74%**、DRAM 28%→57%、Mem throughput 941GB/s→**1.92TB/s**、tensor 2.6%；
代价是 L1 hit 62.6%→**20.0%**、`long_sb` 3.22→**16.71**。即**从「L1 复用」换成「L2 带宽」**，
而这一档正好还有 L2 余量。（P7-24）

**改法二：先证伪「重复读」，再去做软件流水。** 用 `reuse` 把 k 循环复用后实测**恒 1.00×**
（不同 BC 下），`dAqk/dAkk` 的每 `i_k` 重读也被 L2 吸收（DRAM 字节逐位不变，`BK=64` 只值 ~5%）；
真正的杠杆是把 k 分块**从 grid 轴搬进 program 内**并 `tl.range` 软件流水：
孤立 **0.79–0.81×**、`long_sb` 16.7→**4.0**、链 dev 0.97×（1940→1875µs）。（P7-29）

**改法三：直接把结果交到寄存器里。** B2→B3 之间原本有 3 个 `b2+b3` elementwise 求和
（316µs）；让 B3 直接消费 B2 的累加器后 → **49µs**（省 267µs），链 dev 1957→**1685µs**、
少 8 个 elementwise kernel，整体 0.84–0.88×。（P7-28）

**改法四：把 `intra` 也做成 K2 形状的 CUDA 核**：比 Triton MMA 版快 **1.3–1.45×**
（`ncu` duration 0.81×、CUPTI 0.69×），寄存器 72→59、`long_sb` 19.9→11.4。（P7-62）

**折叠进 B2 —— 小 shape 净赢、大 shape 不是**：把 `intra` 折进 B2 的 `i_k==0` 分支后，
小 shape **0.88–0.89×**，大 shape **1.04–1.06×**（略慢），因为它是**关键路径尾巴**（K 轴固定代价）。
再验证「smem staging 布局」这条建议：pad/swizzle 把 bank conflict 砍 **~28%**，时长**几乎不动**
（~1–2%），且 `K=256` 折叠加 bf16 staging 才装得下。**结论：`K=128` 保持「prod + 独立 b3」，
`K=64` 用折叠，按 K 切换。**（P7-66、P7-94、P7-95）

### 7.4 把三条线合成一条链

两条分别收敛的路线（CUDA B1 × 融合 B2/B3）合成 `cf` 后：

| case | FLA dev | `cu`（CUDA B1+未融合） | `fu`（Triton B1+融合） | `cf4` | `cf/cu` | `cf/fu` | `cf/FLA` |
|---|--:|--:|--:|--:|--:|--:|--:|
| T=512 H=4 | 203.7 | 104.7 | 67.6 | 73.8 | 0.70 | 1.09 | **0.36** |
| B2 T=1024 H=8 | 659.8 | 359.0 | 243.3 | 256.4 | 0.71 | 1.05 | **0.39** |
| B8 T2048 H8 | 4137.5 | 2107.5 | 1537.3 | 1401.6 | 0.67 | 0.91 | **0.34** |

- **可加**：`cf/cu` 0.67–0.71×、`cf/fu` 0.91–1.09×，两条优化的收益基本线性叠加。
- **B1 的选择依 shape 交叉**：小 shape 用 Triton B1（`fu` 更小），大 shape 用 CUDA B1（`cf4` 最小）。
- **正确性是位级自洽的**：`cf` vs `fu` 的 `dq/dh0` 逐位相同、`cf` vs `cu` 的 `dv/dh0` 逐位相同，
  即「`dv/dh0` 由 B1 决定、`dq/dk/dg/dbeta` 由 B2/B3 形式决定」，六个 worst-case 与单链逐值一致
  （`dq 3.417e-3 … dh0 9.365e-3`，全落 tf32/bf16 底噪）。
- **最终位置**：整链 dev 是 FLA 反向的 **0.34–0.39×**（汇总记 0.31–0.37×，含非整除 T 的第 4 例），
  即 **~2.7–3.2×**；wall 上 `cf4` 也全面更小（1412.7 vs 2132.8µs 等）。（P7-42）

### 7.5 泛化与口径：从 `K=V=128` 走到 `K≠V`

性能故事讲完，最后一段工作全是「把结论撑到别的形状、并修掉量它的尺子」。

**泛化**：把 `KD`/`VD` 参数化后，base GEMM core 在 `K∈{64,128,256}`、`V∈{64,128,256}` 上
逐位/落 tf32 底噪；六段生产核的 V 轴段改 `BVW=VD/NW`，`K==V` 路径逐位不变
（`K` 参数化 = `NBW=BW/8` 三处 + `dgk` 通用列归约两处，属**lane 布局重写**）。
一个结构性事实：**CUDA core 是唯一能跑 `V=256` 的实现**——identical-op Triton core 把整块
`[V,K]` 留 smem，`V=256` 要 **311 KB > 232 KB**，**编不出来**。
六段核的 smem 是 **122.1 / 126.1 / 134.1 KB**（V=64/128/256）；`intra` staging 的增量有闭式解
$\Delta = 3\cdot BT\cdot K\cdot4 + BT^2\cdot4 + NW\cdot BT\cdot4 = 192K+1536$，
与实测 **13824 / 26112 / 50688 B** 逐字节相等；`K=256` 的 `intra` 折叠在 fp32 下
**272.1 KiB > 227 KiB** 起不来，**只有 bf16 staging（206976 B）能装下**——容量杠杆，不是速度杠杆。（P7-81…P7-95）

**写侧的口径**：`dram__bytes_write` **不能当写量**，它由 L2 脏行逐出主导——同一份输出字节
在跨 dtype/TMA/占用率时散布在 **[−8, +33] MB**，只换 `dA` 段算法（DAMMA）就能把它从 +35 翻到 **−8 MB**，
同配置跨 `ncu` 重放还能差 **~19 MB**。要「比写量」应取 **L1 全局 store 扇区 $S_{L1}$**：
默认（32-bit 跨步 store）$S_{L1}\cdot32 = 2U$、开向量化 store 后 $=U$（$U$ 为解析输出体积）。
这与 §5.1「L2 总字节 / DRAM 字节 / srcunit 扇区是三个口径」是同一条教训的写侧版本。（P7-96…P7-103）

**尺子本身的污染**：`docs/05` §7.4 里「fp32 分块前向 GPU 比 CPU 差 $\sim3\times10^{-4}$」
曾被标 `待验证`。钉死后确认：**是本容器默认 tf32 matmul**（`nv26.04`，env
`TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`），不是 cuSOLVER——GPU fp32 tf32 的差是 **5.97e-04**、
显式 ieee 是 **1.03e-06**、fp64 是 **6.31e-16**，与 $\varepsilon_{tf32}\approx4.9\times10^{-4}$ 对上。
此后对拍一律显式 `allow_tf32=False`；求逆路径（cuSOLVER 与 kernel 8×8 前代）免罪。（P7-112、P7-113、P7-114）

## 8. 收益汇总：每一步值多少

单位与口径见各节。**「结论」列是这一步最终有没有进默认部署。**

| 步骤 | 机制 | 关键数字 | 结论 |
|---|---|---|---|
| 前向 K1/K2 拆分 | 一次算完落 workspace | 设计点 1262 vs 2289µs = **1.81×** | 保留 |
| 前向 K2 提占用率 | 2→3 CTA/SM | 885 vs 773µs（**−14%**）、L2 SOL 94.0%→**81.4%** | 不做 |
| B1 保持 bf16 操作数 | 免 `.float()`/tf32 路径 | 91.3→37.2µs（**2.45×**） | 保留 |
| B1 Triton 软件流水 | `ns=3` | `long_sb` 5.27→**0.96**；1.96×→**1.00×/0.85×** | 保留 |
| B1 换 K2 形状 | 寄存器常驻 + `ldmatrix` | 214.4→**81.8µs（2.4×）** | 保留 |
| B1 `dh`/`dv2` 改 TMA store | 双 4D 张量图 | 恢复上界 **62–83%（均值 75%）**、逐位无损 | 保留 |
| B1 换 CUTLASS `PipelineTmaAsync` | 手写 mbarrier → 库 | **0.985–1.003×**、`wait` ≈1.4 | **空结论** |
| B2 延后 `q/k/g` 加载 | spill 12→0 | 孤立 **0.84–0.89×**、链 0.95–0.97× | 保留 |
| B2 换 K2 形状 CUDA | `mma.sync` + 流式操作数 | **0.90–0.93×**、regs 226→**70**、occ 12.4%→**24.6%** | 研究形态 |
| B2 输入走 TMA | 消 `cp.async` 读放大 | 491µs、**1.12×** Triton、DRAM 89.6% | 大 shape 保留 |
| B2 bf16 原生 staging | 五操作数读减半 | **0.73×**（1.28GB→**637.57MB**） | 保留 |
| B2 六段合成单 launch | 6 个 build → 1 个核 | **0.47–0.53×** | 成立，但整核仍慢 ~1.6–1.7× |
| B3 对角块 MMA 化 | 一份 smem 算两个矩阵 | 孤立 **1.5–2.0×**、链 **−9–12%** | 保留 |
| B3 k 循环软件流水 | grid 打包换 `tl.range` | **0.79–0.81×**、`long_sb` 16.7→4.0 | 保留 |
| B3 折叠进 B2 | `i_k==0` 分支 | 小 shape **0.88–0.89×**、大 shape **1.04–1.06×** | 按 K 切换 |
| 融合 B2/B3 | 累加器留寄存器 | elementwise 316→**49µs**、链 **0.84–0.88×** | 默认 |
| 合成 `cf` | CUDA B1 × 融合 B2/B3 | `cf/FLA` **0.34–0.39×**（~2.7–3.2×） | 最终 |

## 9. 失败与教训

所有被证伪的假设都在这里，保留它们是因为**下一次还会有人这么想**。

| 当时的假设 | 实测 | 教训 |
|---|---|---|
| 占用率是旋钮，`ncu` 说能快 81.25% | `cond(2,1)` 反而慢 14%；B1 用 181.3KB smem、占用率 6.25% 却最快 | `Est. Local Speedup` 是**启发式**，要结合 `lts`/`dram` 实测计数读 |
| bank conflict 是主因（76% wavefront） | padding 后降到 19%，端到端只快 10–23%，BT=64 还 −21% | smem **+29%** 会反噬占用率；分项 ≠ 量级项 |
| 把 D=128 拆 2 条累加链能提 ILP | `splitK` 的 `long_sb` 6.88 > 5.27（base） | 流水一旦开起来，ILP 已被填满 |
| 朴素 B1→B2 融合能省掉 `dh` 往返 | spill 1786、DRAM 2.7→**31.9GB（17×）**、慢 **14–30×** | 融合只能在 K2 形状内做；`dh` 走 TMA store 才是正解 |
| 六段全搬进 CUDA 会更快 | 链内仍慢 **~1.6–1.7×**（998 vs 1606.9µs） | 「结构闭合并 ≠ 性能净赢」：为可独立核验而落盘，代价是字节 |
| `intra` 折叠总归更快 | 小 shape 0.88–0.89×，大 shape **1.04–1.06×** | 它是**关键路径尾巴**，按 K 切换 |
| bf16 一路降精度都赚 | `BF16_VDW` 让 regs 128→**140**、掉回 1 block/SM，**1.4–1.5× 回归** | 先看**寄存器台阶**再看字节 |
| `dram__bytes_write` 就是写量 | 跨配置散布 **[−8,+33] MB**，DAMMA 能翻符号，跨 replay 差 ~19MB | 三个口径：L1 store 扇区（$S_{L1}$）/ L2 写入 / DRAM 写回 |
| 竞态是 `full`/`empty` parity 的 off-by-one | 根因是缺 **`fence.proxy.async`**（proxy 写 smem 后没告诉 async proxy） | 「一次跑干净」≠「保证干净」，要按 stage 扫 |
| TMA tensor-map 的 stride 从 0 起算 | 写反 T/H 两维时 `H==1` **全部逐位通过**，只有 `H>1` 才暴露 | 单 head 测试会系统性漏掉布局错 |
| tf32 是库/版本的事 | 容器默认 `allow_tf32=True` 让**参考实现自己**带 ~3e-4 底噪 | 对拍前显式 `allow_tf32=False`，先修尺子再量 |
| 每轮结论可以直接接着上一轮用 | FLA 重算占比 46.2% → **17.8%**；`xFLA` 1.63 → 0.83 | 对齐 dtype / chunk / 归因窗口，三者缺一不可 |

## 10. 原小节号 → 新小节 对照

这一篇是压缩改写：原文 118 个追加式小节（P7-1…P7-114）在这里被合并成叙述。
旧文里「`docs/07` §NN」式的引用仍然存在（`docs/05`、`docs/06`、`docs/08` 等），
所以给一张对照表——**查旧§，读新节**。完整逐轮日志（含每轮原始表格与逐条边界）见
`code/flash-kda/docs/appendix/07-perf-log.md`。

| 原§ | 主题 | 新§ |
|---|---|---|
| §1 | 计时口径与 event-pair 坑 | §1.1–1.4 |
| §2 | 前向端到端表 | §2 |
| §3 | kernel 级分解 | §3.1–3.2 |
| §4 | 小结与待办 | §3.2、§9 |
| §5 | ncu：K1/K2 bound | §4、§4.1–4.2 |
| §6 | K1/K2 访存账（1.7–2.0GB） | §5.1–5.4 |
| §7–§8 | 4/32 bytes/sector 告警、2 CTA/SM 与 L2 墙 | §4.3 |
| §9 | FLA 的 0.8ms host 税 | §5.5 |
| §10 | 反向基线（workspace vs autograd） | §6.1–6.3 |
| §11–§12 | 复核 GPU 竞争污染、B1 吃原生 bf16 | §6.4 |
| §13–§15 | 16→64 re-chunk、整条反向跟到 64 | §6.4（BT 表） |
| §16 | ncu 剖析 B1 | §7.1（问题） |
| §17 | Triton 原型验证两条设计输入 | §7.1（改法一） |
| §18–§19 | 手写 CUDA B1 与它的 ncu | §7.1（改法二） |
| §20–§21 | smem padding、loop-reorder / CTA 形状 | §7.1（改法二）、§9 |
| §22 | 改成 K2 形状（reg + mma.sync） | §7.1（改法三） |
| §23–§24 | TMA + warp-specialization、`E^tot` 进流水 | §7.1（改法四） |
| §25 | Triton 流水落进真实链 | §6.4（末段） |
| §26 | B2 DRAM-bound / B3 FMA-L1-bound | §7.2（问题）、§7.3（问题） |
| §27 | B3 对角块 MMA 化 | §7.3（改法一） |
| §28–§29 | B2 延后 `q/k/g` 加载、`h/dh` bf16 与 `i_k` 去重 | §7.2（改法一） |
| §30 | B1→B2 融合原型（`dh` 不落盘） | §9（第 4 行） |
| §31 | B3 直吃 B2 累加器 | §7.3（改法三） |
| §32 | B3 的 L1 局部性 / 软件流水 | §7.3（改法二） |
| §33 | B2+B3 融合（累加器留寄存器） | §8（融合 B2/B3） |
| §34 | `dh` 落盘往返的上界 | §7.1（改法五） |
| §35–§36 | `dh` / `dv2` 改 TMA store | §7.1（改法五） |
| §37 | 两个 FLA `dhu` baseline 差在 dtype | §6.4（dtype 段） |
| §38–§41 | 收口 matched-bf16 `xFLA`、`fence.proxy.async` 竞态、store 侧边界 | §7.1（收口）、§9 |
| §42–§43 | CUDA B1 落完整链、host 税「一次编码复用」不可复现 | §7.4 |
| §44 | CUTLASS `PipelineTmaAsync` 空结论 | §7.1（收口）、§8 |
| §45 | 两条优化合成一条链（`cf`） | §7.4 |
| §46–§48 | B2 可行性研判、SIMT 反例、字节消融 | §7.2（改法二） |
| §49–§52 | B2 首次正面、`dgk` 冲突、读放大、TMA 输入 | §7.2（改法三） |
| §53–§54 | `BVK=64` 与两个 swizzle atom | §7.2（改法三） |
| §55–§60 | 非 GEMM 六段逐个进 CUDA | §7.2（改法四） |
| §61–§64 | 合并 `q/k/g`、向量化 store、单 launch、`dw` 交接 | §7.2（改法四） |
| §65–§66 | B3 的 K2 形状版、`intra` 折叠 | §7.3（改法四） |
| §67–§73 | 生产核落链、段归因、DAMMA、TMA×INTRA | §7.2（收口） |
| §74–§75 | 向量化 store 落链、`atom64` 的 L2 重触 | §7.2（改法三） |
| §76–§83 | bf16 原生入口 / 原生 mma / 六段 bf16 | §7.2（改法五） |
| §84–§98 | `K≠V` 泛化、`K∈{64,128,256}`、`intra` 折叠的 smem | §7.5（泛化） |
| §99–§108 | 写侧 `dram__bytes_write` 口径 | §7.5（写侧口径）、§9 |
| §109–§111 | bank-conflict 二级机制、`K=256` 容量杠杆 | §7.5（泛化） |
| §112–§115 | 写侧逐出机制与 replay 复测 | §7.5（写侧口径） |
| §116–§118 | 求逆路径免罪、容器默认 tf32、系统审计 | §7.5（尺子污染） |

## 三句话小结

- **前向的瓶颈不是乘法**：K1/K2 的 TensorCore 只有 ≤1.6% / 22–24%，1.7–2.0GB 的 L2 流量里
  ~75% 是 workspace、~22.5% 是不可省的 `v`；把占用率从 18.75% 提到 28.12% 反而慢 14%。
- **反向的收益几乎全来自「不重算前向」**：FLA 反向有 17.8–21.9% 花在重算 chunk 状态与 `w/u`；
  我们直接吃 K1 workspace，再加 B1 的流水/store 引擎、B2 的形状与 bf16、B3 的 MMA 化与融合，
  最终整链 dev 是 FLA 反向的 **0.34–0.39×**。
- **口径比数字更重要**：本篇至少四次「看起来的收益」是测量假象（event-pair 中位数、未预热 JIT、
  GPU 竞争、dtype/chunk 不对齐），也有至少三次「看起来的优化」被实测证伪（提占用率、
  降 bank conflict、朴素融合）——所以每个数字都带 dtype、窗口与结论限定词。

下一篇是专题收尾：把前六篇的结论、这套反向链的部署形态与还没关掉的 `待验证` 收成一张总表。

> 完整逐轮日志见 `code/flash-kda/docs/appendix/07-perf-log.md`。
