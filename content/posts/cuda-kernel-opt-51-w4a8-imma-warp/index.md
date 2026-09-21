---
title: "CUDA 算子调优（五十一）：W4A8 IMMA 的 warp-private pipeline——把 barrier 清零之后，墙去哪了"
date: 2026-09-22T05:30:00+08:00
draft: false
weight: 51
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A8", "int4", "IMMA", "cp.async", "软件流水", "warp specialization", "occupancy", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第十一篇。承接
[第 48 篇 W4A8 的整数张量核（IMMA）]({{< relref "cuda-kernel-opt-48-w4a8-imma" >}}) 与
[第 50 篇 IMMA 的收尾]({{< relref "cuda-kernel-opt-50-w4a8-imma-pipe" >}})。

50 篇把 `long_scoreboard`（1.71）换成了 `barrier`（1.61），留下了明确的任务：

> 50 篇把主 stall 从 `long_scoreboard` 换成了 `barrier`，根因是 128 线程 / 4 warp
> 共享同一批 A/W stage、每 stage 两次 `__syncthreads`，而 occupancy 只有 26%。
> 思路：让每个 warp 拥有**自己的 stage 环与 scale**，warp 内 `cp.async` +
> `__syncwarp` 即可，彻底去掉 CTA 级 barrier。

本篇就是做这件事。结论先给，**这是一个「技术成立、收益有限、但把瓶颈彻底讲清楚」的增量**：

- **warp-private pipeline 成立**：CTA 级 `__syncthreads` 全部换成 warp 级
  `__syncwarp`，`barrier` stall **1.61 → 0.12~0.21（≈8~13×）**，同进程实测
  `M=1` **1.07×**、`M=16` **1.11×**，全区间单配置 **1.09×**（相对 50 篇最佳）。
- **但墙只是换了个名字**：`long_scoreboard` **0.30 → 1.60~1.69**。没有 barrier 之后，
  warp 直接暴露在 `cp.async` 的完成等待上——**每个 stage 一次 DEPBAR**，而
  occupancy 只有 20~29%，藏不住 global 延迟。
- **A 的冗余读是第一个大坑**：W 本来就按 `warp_col` 沿 n 分片、各 warp 互不相交，
  但 A（16 行 × 5120）是所有 warp 共用的。per-stage 各 warp 冗余读一份，让
  `M` 越大越慢（A 流量 ∝ M）——`M=16` 时甚至**比 50 篇基线还慢 30%**。
  解法是 A 由 CTA **一次性预载**进共享缓冲（只一次 `__syncthreads`），这一招把
  M 的退化彻底修好。
- **一堆「想当然」的优化全负**：紧凑 A（按 M 裁行）、强制 occupancy（`launch_bounds`
  压寄存器 → spill）、加深 STAGES、把一个 stage 打包多个 group（SUB）、`BN=256`、
  `WN=2`——逐条实测在下面。
- **诚实的天花板**：`M=1` 的 IMMA 权重带宽 **1784 GB/s（53% HBM）**，仍输给
  [45 篇的 W4A8 GEMV]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}})
  （**2408 GB/s / 76% HBM**）1.35×。**M=1 就该用 GEMV**，IMMA 的价值在 `M≥3`。

shape 全部取自 `/ssd/models/qwen3-8B/config.json`：
`hidden_size=5120, intermediate_size=17408, group_size=128`，对称 int4；
测 `N=17408, K=5120, M=1..16`（decode）。权重 44.6 MB(int4) + scale 2.8 MB。

---

## 一、回顾：50 篇的 barrier 是怎么来的

50 篇的最优 kernel（`tile_s2k8_b5`）结构是这样的：

```
一个 CTA = 128 线程 = 4 个 warp，一起处理 (n_tile, k_chunk)
  共享 stage 环：[STAGES][BM=16 行 A][BN=128 行 W][A/W scale]
  每 stage 流程：
    cp.async 载 A(2KB) + W(8KB) + 转置 scale
    __pipeline_wait_prior(...)      # 等 cp.async 完成（per-thread）
    __syncthreads()                 # ① 让全 CTA 看到别线程搬的数据
    IMMA(m16n8k32) × 16  +  per-128 折算
    __syncthreads()                 # ② 确认所有线程读完了，才能覆盖 stage
```

`__syncthreads()` ①和②把 4 个 warp 死死绑在一起：任何一个 warp 慢一点，其余
三个就一起停在 barrier 上。ncu 的指纹非常清楚（50 篇 `ncu_tile_s2k8_b5_M1`）：

| stall（per issue-active） | 50 篇基线 `tile_s2k8_b5` |
|---|---|
| `barrier` | **1.61**（第一） |
| `wait`（固定延迟依赖） | 1.04 |
| `not_selected` | 1.01 |
| `short_scoreboard` | 0.42 |
| `long_scoreboard` | **0.30** |
| DRAM / L2 | 60.1% / 61.8% |
| Achieved occupancy | 26.3% |
| 指令数 | 11.56 M |

关键观察：**W 本来就不需要 CTA 级同步**。每个 warp 在 `BN` 方向只负责
`WARP_N = BN/WN = 32` 列，各 warp 的 W 片段**互不相交**。真正共享的只有 A
（`BM=16` 行）和「同一个 stage 缓冲」。于是：

> 给每个 warp 一个**私有的 stage 环**（W + 自己的 A 拷贝 + 自己的 scale），
> 同步就从 CTA 级 `__syncthreads` 降成 warp 级 `__syncwarp`。

`__syncwarp()` 也带 warp 范围的 shared memory 顺序保证，足以让「lane A 搬的
`cp.async` 数据」被「lane B 读到」。这就是 warp-private pipeline。

---

## 二、实现：把 stage 环拆到每个 warp

新 kernel `gemm_mma_w4a8_warp_kernel<BM,BN,STAGES,WM,WN,KSPLIT,MINB,TILED,WMODE>`：

```
smem 布局（NWARP = WN 个 warp，每 warp 自己的环）：
  [warp 0][stage 0..S-1]  A(16×144) | W(32×80) | sa(16) sc(16) sw(32)
  [warp 1][stage 0..S-1]  ...
  ...

每 warp 每 stage：
  for i = lane; i < 16*8;  i += 32   cp.async A 行
  for i = lane; i < 32*4;  i += 32   cp.async W 行（K-blocked 布局，连续 2KB）
  for i = lane; i < 4;     i += 32   cp.async sa / sc
  for i = lane; i < 8;     i += 32   cp.async sw
  __pipeline_commit()

主循环：
  load(stage_{t+S-1}) ; __pipeline_wait_prior(S-1) ; __syncwarp()
  IMMA × NC × MTN  + 折算
  __syncwarp()        # 只保证本 warp 读完，才能覆盖自己的 stage
```

整个 kernel 里**没有一条 CTA 级 `__syncthreads`**（除了 A 预载那一次，见下）。

`TILED` 开关沿用 50 篇的 K-blocked 权重布局 `Wt[g][N][64B]`：一个 warp 一个
stage 的 W = 32 行 × 64 B = **连续 2 KB**，是 `cp.async` 最喜欢的形态。

### 2.1 第一个坑：A 的冗余读让 M 越大越慢

W 各 warp 不重叠，但 A 是所有 warp 都要的。第一版 `WMODE=0` 让每个 warp
per-stage 各读一份 A（16×128 B = 2 KB）。同进程实测：

| 配置 | M=1 | M=4 | M=8 | M=16 |
|---|---|---|---|---|
| `base_tile_s2k8_b5`（50 篇） | 0.0291 | 0.0294 | 0.0309 | 0.0355 |
| `w_s2k8`（冗余 A，s2） | 0.0299 | 0.0350 | 0.0373 | **0.0459** |
| `w_s3k8`（冗余 A，s3） | 0.0278 | 0.0349 | 0.0379 | **0.0473** |

`M=1` 略赢，但 `M=16` 直接**比基线慢 30%**。原因很简单：A 的字节数 ∝ M，
4 个 warp 各读一份 = **A 的 L2 流量 ×4**；`M=16` 时 A 全量 80 KB，per-stage
冗余读把它顶到和 W 同量级（W 一个 CTA-stage 只有 8 KB，A 冗余 4×2 KB=8 KB）。
50 篇把 A 共享着读，`M=16` 时 A 只占 W 的 1/4。

### 2.2 解法：A 由 CTA 一次性预载（`WMODE=1`）

既然 per-stage 共享 A 需要同步、per-warp 冗余又太贵，那就**把整段 A 一次性
搬进共享缓冲**——只在 kernel 开头做一次 `__syncthreads()`：

```
整段 A = [nt 个 group][BM 行][144B]，nt = nblk/KSPLIT（如 KSPLIT=8 → nt=5）
开头：
  for i = tid; i < nt*16*8; i += NTHREADS   cp.async A 行
  commit ; wait_prior(0) ; __syncthreads()   # 唯一一次 CTA 同步
之后每个 stage 只搬 W + scale（A 直接从预载区取）
```

关键点：**A 的字节数 ∝ M，但它现在是「一次性」，不再随 warp 数翻倍**。
`M=16` 时 A 总量 = `nt·16·144`，和 50 篇的 per-stage 共享版一致；`M=1` 时
仍然读满 16 行（模板按 `BM` 预留），只是行 1..15 读的是掩码 0。修完之后：

| 配置 | M=1 | M=4 | M=8 | M=16 |
|---|---|---|---|---|
| `wA_s2k8`（预载 A，s2） | **0.0272** | 0.0283 | 0.0303 | 0.0353 |
| `wA_s3k8`（预载 A，s3） | 0.0276 | 0.0292 | 0.0306 | 0.0353 |
| `wA_s3k4`（预载 A，s3，KSPLIT=4） | 0.0277 | 0.0285 | 0.0294 | **0.0319** |

`M=16` 从 0.0459 拉回 0.0319（比 50 篇的 0.0355 快 1.11×）。

---

## 三、ncu：barrier 消失，`long_scoreboard` 补位

同进程实测（`final_sweep`，warmup=5 / iters=50）：

| M | `base_tile_s2k8_b5` | `wA_s2k8` / `wA_s3k4` | 加速 |
|---|---|---|---|
| 1 | 0.0291 | **0.0272** (`wA_s2k8`) | 1.070× |
| 2 | 0.0292 | **0.0280** | 1.043× |
| 3 | 0.0293 | **0.0281** | 1.043× |
| 4 | 0.0294 | **0.0283** | 1.039× |
| 6 | 0.0302 | **0.0290** (`wA_s3k4`) | 1.041× |
| 8 | 0.0309 | **0.0294** | 1.051× |
| 12 | 0.0331 | **0.0309** | 1.071× |
| 16 | 0.0355 | **0.0319** | 1.113× |
| **单配置总和**（`wA_s3k4`） | **0.2567** | **0.2341** | **1.097×** |

`M=1` 权重带宽 47.3 MB / 0.0272 ms = **1739 GB/s（51.9% HBM）**；
`M=16` = **1486 GB/s（44.3%）**。纯读上界（同口径读 W+s）**0.0183 ms /
2590 GB/s（77.3% HBM）**，所以 `M=1` 跑到 roof 的 ~67%。

ncu 把「墙去哪了」讲得明明白白（`wA_s2k8` M=1 / `wA_s3k4` M=16）：

| 指标 | 50 篇 `tile_s2k8_b5` (M=1) | `wA_s2k8` (M=1) | `wA_s3k4` (M=16) |
|---|---|---|---|
| stall `barrier` | **1.61** | **0.21** | **0.12** |
| stall `long_scoreboard` | 0.30 | **1.60** | **1.69** |
| stall `not_selected` | 1.01 | 1.57 | 1.04 |
| DRAM Throughput | 60.1% | 64.7% | 54.3% |
| L2 Throughput | 61.8% | 65.4% | 65.9% |
| SM Busy / Issue | — | 47.6% | 37.1% |
| Achieved occupancy | 26.3% | 29.4% | 20.7% |
| 寄存器 / 线程 | 86 | 76 | 72 |
| occupancy 上限（reg / smem） | — | 6 / 6 CTA/SM | 7 / 4 CTA/SM |

注意 `barrier` 并没有变成 0，还有 0.12~0.21——那来自 A 预载那一次
`__syncthreads()`，以及 `__syncwarp` 本身（占比极小）。

**`long_scoreboard` 从 0.30 涨到 1.60，不是变差了，而是「浮出来了」。**
50 篇里 warp 大部分时间停在 barrier 上，访存延迟被别的 warp 的 issue 掩盖；
把 barrier 拿掉后，每个 warp 直接暴露在「等自己那一组 `cp.async` 完成
（`cp.async.wait_group` → `LDGDEPBAR`）」上：

```
wA_s2k8 M=1 的实测：
  SM Busy 47.6%  ·  Issue Slots Busy 46.4%  ·  DRAM 64.7%  ·  L2 65.4%
  → 没有一级资源到顶；瓶颈是「在飞访存请求不够」，即 latency-bound。
```

为什么「不够」？每个 warp 一个 stage 只有 `2KB(W) + 2KB(A，预载后不计) + scale`
的在飞量，STAGES=2 时最多 1 组未完成；24 个 warp/SM 也填不满 cp.async 的
发射-完成窗口。这解释了后面的负结果：**加深 STAGES 会增加在飞量，但同时吃
smem、把 occupancy 打下来**，两者互相抵消。

---

## 四、负结果：五个「看起来应该有用」的方向

### 4.1 紧凑 A（`WMODE=2`，只存 M 行）——负

既然 A 的字节 ∝ M，那 `M=1` 时只存 1 行不就行了？实现成
`[nt][M][LARGE_BK+16]`（行距必须 16B 对齐，否则 `cp.async` 直接
`misaligned address`），并把动态 smem 按运行期 `M` 传。

实测 `wC_s2k8`（M=1 smem 34048→23168）**0.0316**，比 `wA_s2k8` 的 0.0272
**慢 16%**，而且 M 扫描出现非单调（M=2/4 反而比 M=1 快）。原因是它把
occupancy 从「reg/smem 都限制在 6」变成「reg 限制 6 / smem 可到 10」——
**瓶颈是寄存器，不是 smem**，缩小 smem 不产生任何收益，反而因为多了一层
运行期行数计算/掩码而更慢。教训：**先确认 occupancy 被谁限制，再决定缩谁。**

### 4.2 强制更高 occupancy（`__launch_bounds__(128, 8/10)`）——负

`wA_s2k8` 的寄存器是 76（2-CTA 门槛 85、3-CTA 64）。想上 8~10 CTA/SM
就得压到 ≤64/51。ptxas 直接给出 spill：

```
wA_s2k8_b10: 48 regs + 96 B spill stores / 64 B spill loads
实测 M=1 0.0291（慢 7%）、M=16 0.0393（慢 11%）
```

`acci[4][4]`（16）+ `facc[4][4]`（16）已经是 32 个累加器寄存器，压到 64
以下没有余量。这是 43 篇 `acc[64]` 寄存器墙的「小号版」：**小 M 的 IMMA
没必要为了 occupancy 去 spill**。

### 4.3 加深流水（`STAGES=3/4/6`）——负

| | smem | CTA/SM | M=1 | M=16 |
|---|---|---|---|---|
| `wA_s2k8` | 34048 | 6 | **0.0272** | 0.0353 |
| `wA_s3k8` | 45312 | 5 | 0.0276 | 0.0353 |
| `wA_s4k8` | 56576 | 4 | 0.0287 | 0.0348 |
| `w_s6k8` | 122880 | 1 | 0.0408 | 0.0513 |

`STAGES` 深了确实在飞量更多，但 smem 线性增长把 CTA/SM 从 6 压到 1，
净负。这是本篇最核心的权衡：**warp-private 想要 latency hiding，只能在
「warp 数 × 每 warp 在飞量」里分配固定的 smem 预算**。

### 4.4 一个 stage 打包多个 group（`SUB`）——负

另一个降「每字节一次 DEPBAR」的办法：让一个 stage 装 `SUB` 个 group
（W 2KB→4/8 KB），一次 `wait` 覆盖更多字节。

| | smem | M=1 | M=16 |
|---|---|---|---|
| `wA_s2k5_S2` | 63488 | 0.0284 | 0.0334 |
| `wA_s2k5_S4` | 108544 | 0.0298 | 0.0351 |
| `wA_s2k10_S2` | 54272 | 0.0275 | 0.0359 |

`SUB=2` 基本打平、`SUB=4` 因 smem 暴涨变慢。**「减少 wait 次数」不是瓶颈**——
每次 wait 覆盖多少字节并不重要，重要的是同时有多少 warp 在等。

### 4.5 `BN=256` / `WN=2`——负

`w_bn256_s3k8`（256 线程，8 warp）M=16 0.0490、`w_wn2_s4k8`
（64 线程，WARP_N=64）全线慢 5~15%。warp 太胖（每 warp 一个 stage 的 W
翻倍）或 CTA 太大都让「每 warp 私有环」的 smem 乘法更贵。

---

## 五、小结与定位

- **warp-private pipeline 是有效的结构改造**：去掉 CTA 级 `__syncthreads` 后
  `barrier` **1.61 → 0.12~0.21**，`M=1` 1.07×、`M=16` 1.11×、全区间单配置
  1.09×。代码见 `code/kernel-opt/51-w4a8-imma-warp/`。
- **A 必须一次性预载、不能每 warp 冗余**：W 天然按 n 分片，A 不是；per-stage
  冗余读让 A 的 L2 流量 ×warp 数、`M=16` 反而慢 30%。
- **墙从 barrier 换成了 `long_scoreboard`**：没有 CTA 同步之后，warp 直接
  暴露在 `cp.async.wait_group` 的完成等待上；而 smem 预算同时卡住
  「更多 warp」与「更深流水」——两个方向都试过，都负。
- **IMMA 在 `M=1` 不是最优工具**：53% HBM vs
  [45 篇 GEMV]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}}) 的 76% HBM。
  `M=1` 就该用 GEMV；IMMA 的价值在 `M≥3`（省掉反量化 ALU），本轮的
  warp-private 让它在 `M≥4` 稳定赢过 50 篇基线 4~11%。
- **数值**：整数 GEMM 精确，48 个采样点相对误差 1e-7 量级；量化 kernel 与
  CPU 参考逐位一致。

**下一篇预告**：这条路要继续往 roof 推，只剩两招——① 用 **TMA（`cp.async.bulk`）
搬 K-blocked 权重**（一个 stage 的 W 已是连续 2 KB，天然适合 1D bulk），把
「每 stage 一组 `cp.async` + 一次 DEPBAR」换成「mbarrier + 硬件异步引擎」，
在飞量由 TMA 引擎兜底；② 回到 42 篇暂缓的
[MLA decode producer/consumer warp specialization]({{< relref "cuda-kernel-opt-42-mla-decode-wgmma" >}})。
代码与原始输出见 `code/kernel-opt/51-w4a8-imma-warp/`。
