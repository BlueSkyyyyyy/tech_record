---
title: "CUDA 算子调优（五十二）：W4A8 IMMA 的 TMA warp-private——把 2 KB 权重交给 TMA 引擎（和一个必须自己造的无冲突布局）"
date: 2026-09-22T05:55:00+08:00
draft: false
weight: 52
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A8", "int4", "IMMA", "TMA", "cp.async.bulk", "mbarrier", "bank conflict", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第十二篇。承接
[第 51 篇 W4A8 IMMA 的 warp-private pipeline]({{< relref "cuda-kernel-opt-51-w4a8-imma-warp" >}})。

51 篇把 CTA 级 `__syncthreads` 全部拆成 warp 级 `__syncwarp`，`barrier` stall
从 **1.61 掉到 0.12~0.21（≈8~13×）**，但墙只是换了个名字：

> `long_scoreboard` **0.30 → 1.60**。没有 barrier 之后，warp 直接暴露在
> `cp.async` 的完成等待上——**每个 stage 一次 DEPBAR**，而每个 warp 一个 stage
> 只有 2 KB W 在飞，occupancy 只有 20~29%，藏不住 global 延迟。

51 篇给出的下一步非常明确：**把 W 交给 TMA 引擎**。K-blocked 布局
`Wt[g][N][64B]` 下一个 warp 一个 group 恰好是**连续 2 KB**，天生适合
`cp.async.bulk`——一条指令搬完，完成信号走 mbarrier，warp 不再发一堆
`cp.async` 也不再有「每 stage 一次 DEPBAR」。

本篇做这件事。结论先给，**这是一个「主线成立、且靠一个自造布局才兑现」的增量**：

- **朴素 TMA 版反而更慢**：把 W 用 1D bulk 搬进一个「行主序连续」的 smem
  缓冲后，IMMA 读 W 的 bank 数从 30 个塌成 2 个——ncu 报 **shared excessive
  wavefronts 62%**、`L1/TEX` 从 31% 飙到 **77%**，同进程比 51 篇慢
  5~15%。**TMA 不自动等于快**。
- **修法是一个 host 端的置换布局** `Wt2[g][warp][c][n][8×u16]`：每个 warp 每个
  group 仍是**连续 2 KB**（TMA 1D bulk 照搬），但把 `[n][c]` 换成 `[c][n]`
  后，IMMA 读 W 的 8 个 `bgroup` 落在 8 个不同 bank → **conflict 归零**
  （excess wavefronts 62% → **9%**、`L1/TEX` 77% → **35%**）。
- **结果**：同进程单配置（`t_s2k4` vs 51 的 `wA_s3k4`）全 M 段
  **1.07×**；逐 M 取最优 **1.09×**。`M=1` 最好 **0.0257 ms / 1841 GB/s /
  54.9% HBM**（51 篇 0.0268 / 1727 GB/s / 51.5%）。
- **一个干净的消融**：同一个 TMA kernel，只切「行主序」vs「`[c][n]` 置换」，
  `M=1` 差 **1.08~1.10×**——bank conflict 的代价被单独量了出来。
- **诚实的定位**：`M=1` 仍输给
  [45 篇的 W4A8 GEMV]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}})
  （2408 GB/s / 76.3% HBM）**1.31×**；IMMA 的价值仍在 `M≥3`。瓶颈仍是
  **latency**：ncu `DRAM 67% / L2 62% / Compute 51%`，没有一级到顶。

shape 全部取自 `/ssd/models/qwen3-8B/config.json`：
`hidden_size=5120, intermediate_size=17408, group_size=128`，对称 int4；
测 `N=17408, K=5120, M=1..16`（decode）。权重 44.6 MB(int4) + scale 2.8 MB。

---

## 一、为什么 51 篇的下一步是 TMA

回顾 51 篇 warp-private kernel 的一个 stage：

```
每个 warp 每 stage：
  for i = lane; i < 32*4; i += 32   __pipeline_memcpy_async(W 行 16B)   # 8 条/线程
  ... sa/sc/sw ...
  __pipeline_commit()
主循环：
  issue(stage_{t+S-1})              # 8×32 条 cp.async
  __pipeline_wait_prior(S-1)        # ← DEPBAR：warp 直接停在这里
  IMMA × NC×MTN + 折算
```

`cp.async` 的完成是**每线程一条 `cp.async.wait_group`**，也就是 51 篇 ncu 里那个
`long_scoreboard = 1.60` 的来源。而且每个 warp 一个 stage 只有 `32×64B = 2KB`
W，`STAGES=2` 时最多 1 组未完成，24 warps/SM 也填不满完成窗口。

TMA（`cp.async.bulk`）恰好是「发射即返回」的异步引擎：**一条指令搬一大块，
完成由 mbarrier 的 `complete_tx` 通知**。W 在 K-blocked 布局
`Wt[g][N][64B]` 下一个 warp 一个 group = **连续 2 KB**，正是 1D bulk 的理想形态。

于是设计很自然：

```
smem 布局（WN 个 warp，每个 warp 一个私有 stage 环 + 私有 mbarrier）：
  Afull：CTA 一次性预载的整段 A（16 行 × nt 个 group）
  Wbase：[warp][stage] 每格 2 KB 的 W
  sbase：[warp][stage] 每格 sa(16)+sc(16)+sw(32) 个 fp32
  mbars：[warp][stage] 每格一个 uint64 mbarrier

每个 group 每个 warp：lane0 发 4 条 1D bulk + 一次 expect_tx
  cp.async.bulk W  (2048 B)   ← 连续 2 KB
  cp.async.bulk sa (  64 B)
  cp.async.bulk sc (  64 B)
  cp.async.bulk sw ( 128 B)   总 tx = 2304 B
主循环：
  issue(mbar[st_next], tnext)       # 只 lane0 发，其它 lane 直接往下走
  mbar_wait(mbar[st], (t/STAGES)&1) # 等自己这个 stage 的 mbarrier
  __syncwarp()
  IMMA × NC×MTN + 折算
```

因为**每个 warp 既是它自己 stage 的生产者也是消费者**，pipeline 的「空槽」保证
可以由程序顺序免费给出（warp 先消费 `t-1`、后覆盖它的 buffer），所以**只需要
一个 full mbarrier、不需要 empty barrier**——这是它比 40/41 篇的 producer/consumer
warp specialization 简单的地方。

PTX 就三条（`w4a8_imma_tma.cu:913`、`:917`）：

```cpp
// 1D bulk：smem 目标 16B 对齐、字节数 16B 倍数、global 源 16B 对齐
asm volatile(
  "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
  " [%0], [%1], %2, [%3];\n"
  ::"r"(smem_u32(dst)), "l"(gsrc), "r"(bytes), "r"(smem_u32(bar)) : "memory");
```

mbarrier 的相位不需要数组（对比 23 篇动态下标掉 local memory 的坑），直接算
`(t / STAGES) & 1`。`mbarrier.init` 之后要一条
`fence.mbarrier_init.release.cluster` 再 `__syncthreads`。

---

## 二、第一版就翻车：TMA 不自动等于快

把 W 按「行主序连续」搬进 smem（就是 `Wt[g][N][64B]` 里那段 2 KB 原样），
IMMA 读 W 还是原来那套下标：

```
lane 读 uint16 下标 wo = n·32 + c·8 + aqw   (n = j·8 + (lane>>2), c∈[0,4))
两个 uint16 = expand4u → B 片段
```

同进程实测（`tma_sweep.out.txt` 的第一版，未存盘，值记录于此）：`t_s3k8` `M=1`
**0.0284 ms**——比 51 篇的 0.0268 还慢。ncu 一眼看出问题
（`ncu_tma_s2k10_row.out.txt`）：

| 指标 | 行主序 TMA（`t_s2k10_row`） | 51 篇 `wA_s2k8` |
|---|---|---|
| `L1/TEX Cache Throughput` | **77.36%** | 31.19% |
| shared excessive wavefronts | **62%** | （无） |
| DRAM Throughput | 61.66% | 64.69% |
| DRAM 有效带宽 @M=1 | — | — |

**bank conflict 的来历**：行主序 `[WARP_N=32 行][64B]`，行距 64B = 16 个
4 字节 word。`uint16` 地址落在 word `n·16 + (lane&3)/2`；`n = j·8 + bgroup`
里 `j·8` 对 32 取模是 0，于是同一时刻 8 个 `bgroup` 的 word 地址只落在
`{0,16}` 两个 bank 上 → **8 路冲突**。51 篇用 `WPAD=80`（行距 +16B）把它错开了，
而 TMA 写 smem 的布局是硬件定的、**只能是紧凑行主序**，那条 padding 就没了。

> 教训：**TMA 只负责「怎么搬」，不负责「搬到哪」**。当消费者是 `IMMA`
> 这种对 smem 行距极其敏感的取数模式时，TMA 的默认布局可能正好撞在最坏情况上。
> 这正是 20 篇「SW128 决定生死」的同一类问题，只是这次硬件不肯替我们 swizzle。

---

## 三、修法：造一个「TMA 友好 + 读无冲突」的置换布局

我们想要两件事同时成立：

1. **每个 warp 每个 group 的 W 是一段连续 2 KB**（否则 1D bulk 不成立）；
2. IMMA 读这 2 KB 时，8 个 `bgroup` 的地址落在 8 个不同 bank。

关键观察：IMMA 一个 `(c, j)` 取数只用到第 `c` 个 k-block 的 8 个 `uint16`
（`c*8 .. c*8+7`，`aqw` 选其中两个）。如果我们把这 2 KB 按 **`[c][n][8×u16]`**
排，而不是 `[n][c*8..c*8+7]`，那么对固定的 `c`：

```
word 下标 = ((c·32 + n)·8 + (lane&3)) / 2 = (c·32 + n)·4 + (lane&3)/2
           n = j·8 + bgroup
mod 32  →  (c·128) + (j·32) + bgroup·4 + const  ≡  bgroup·4 + const   (mod 32)
```

`bgroup = 0..7` → `{0,4,8,…,28}`，**8 个不同 bank**；同一 `bgroup` 的 4 条 lane
只落在同一个 word 的两个半字上（广播，无代价）。冲突归零。

host 侧就是一次 `memcpy`（`w4a8_imma_tma.cu:1315` 附近）：

```cpp
// Wt2[g][warp][c][n][8×u16]，warp = block_col/32 + warp_col
for g, w, c, n:
  src = Wt[g][w*32 + n][c*16 .. c*16+16)   // 16 字节 = 8 个 uint16
  dst = Wt2[((g*NW + w)*4 + c)*32*16 + n*16]
```

kernel 里 TMA 的 global 坐标相应改成
`Wt2 + ((g·(N/32) + block_col/32 + warp_col) · 2048)`，读下标改成
`wo = (c·32 + n)·8 + aqw`（`w4a8_imma_tma.cu:1061`）。**smem 内部布局、
IMMA 指令、折算逻辑一字未改**，只是一个 host 置换 + 两处坐标。

### 3.1 消融：bank conflict 值多少

同一个 TMA kernel 加一个 `PERM` 模板开关（行主序 vs `[c][n]` 置换），
同进程 head-to-head（`ablation.out.txt`）：

| M | `t_s2k8_row` | `t_s2k8_perm` | 置换收益 |
|---|---|---|---|
| 1 | 0.0300 | **0.0277** | **1.083×** |
| 4 | 0.0318 | **0.0285** | 1.116× |
| 16 | 0.0361 | **0.0343** | 1.052× |

| M | `t_s2k10_row` | `t_s2k10_perm` | 置换收益 |
|---|---|---|---|
| 1 | 0.0282 | **0.0262** | **1.076×** |
| 4 | 0.0294 | **0.0272** | 1.081× |
| 16 | 0.0364 | **0.0355** | 1.025× |

ncu 对照（同一 `t_s2k10`、同一 M）：

| 指标 | `row` | `perm` |
|---|---|---|
| `L1/TEX Cache Throughput` | **77.36%** | **35.18%** |
| shared excessive wavefronts | **62%** | **9%** |
| global excessive sectors | 36% | 7% |
| DRAM Throughput | 61.66% | **67.19%** |
| Compute (SM) | 47.21% | 51.16% |
| Achieved occupancy | 34.67% | 31.10% |

一个 host 端 `memcpy` 换掉近一半的 `L1/TEX` 占用、把 DRAM 抬了 5.5 个点——
这就是 20/42 篇「布局决定生死」的又一个实例。

---

## 四、结果：和 51 篇同进程对拍

`tma2_sweep.out.txt`（同进程、`warmup=5, iters=50`，纯读上界 2536 GB/s）：

| kernel | M=1 | M=2 | M=3 | M=4 | M=6 | M=8 | M=12 | M=16 |
|---|---|---|---|---|---|---|---|---|
| 51 `wA_s2k8` | 0.0268 | 0.0274 | 0.0274 | 0.0278 | 0.0287 | 0.0297 | 0.0322 | 0.0345 |
| 51 `wA_s3k4` | 0.0273 | 0.0279 | 0.0280 | 0.0281 | 0.0284 | 0.0288 | 0.0302 | 0.0312 |
| **`t_s2k4`** | 0.0270 | 0.0272 | 0.0272 | 0.0276 | 0.0278 | 0.0280 | 0.0293 | **0.0301** |
| **`t_s2k10`** | **0.0257** | **0.0260** | **0.0263** | **0.0267** | 0.0284 | 0.0290 | 0.0317 | 0.0351 |
| `t_s3k8` | 0.0263 | 0.0265 | 0.0268 | 0.0270 | 0.0278 | 0.0294 | 0.0308 | 0.0328 |

逐 M 最优（`t_s2k10` 在 M≤4、`t_s2k4` 在 M≥6）：

| M | 51 篇最优 | TMA 最优 | 有效权重带宽 | %HBM |
|---|---|---|---|---|
| 1 | 0.0272 | **0.0257** | **1841.5 GB/s** | **54.9%** |
| 2 | 0.0276 | **0.0260** | 1822.2 | 54.4% |
| 3 | 0.0277 | **0.0263** | 1798.7 | 53.7% |
| 4 | 0.0282 | **0.0267** | 1770.5 | 52.8% |
| 6 | 0.0284 | **0.0277** (s4k10) | 1712.1 | 51.1% |
| 8 | 0.0288 | **0.0280** (s2k4) | 1692.7 | 50.5% |
| 12 | 0.0302 | **0.0293** (s2k4) | 1615.4 | 48.2% |
| 16 | 0.0312 | **0.0301** (s2k4) | 1572.5 | 46.9% |

- **单配置**（`t_s2k4` 一把打天下）全 M 段 = 0.2242 ms，51 篇单配置
  `wA_s3k4` = 0.2399 ms → **1.070×**。
- **逐 M 取最优** = 0.2198 ms → 相对 51 篇最优（0.2328）**1.059×**。
- 纯读上界 0.0187 ms / 2536 GB/s（75.6%）；`M=1` 的 54.9% 意味着仍有一道
  延迟墙（见下）。

**正确性**：内置 CPU 参考对拍（48 个抽样点），所有变体 `rel ≤ 1.2e-7`
（int4/int8 整数点积 + fp32 折算，数值上精确），与 51 篇逐位同源。

---

## 五、扫参：STAGES × KSPLIT 与 occupancy

`tma2/tma3_sweep.out.txt` 的完整扫描（`N=17408, K=5120`）：

| 变体 | smem | M=1 | M=16 | 说明 |
|---|---|---|---|---|
| `t_s2k4` | 42.6 KB | 0.0270 | **0.0301** | 高 M 最佳 |
| `t_s2k8` | 31.0 KB | 0.0269 | 0.0335 | — |
| `t_s3k8` | 40.3 KB | 0.0262 | 0.0328 | 低 M 次佳 |
| `t_s2k10` | 28.7 KB | **0.0257** | 0.0351 | 低 M 最佳 |
| `t_s3k16` | 35.7 KB | 0.0263 | 0.0400 | k 太大，高 M 崩 |
| `t_s2k20` | 27.1 KB | 0.0272 | 0.0438 | 归约税吃掉 |
| `t_bn256_s3k8` | 68.0 KB | 0.0276 | 0.0333 | BN=256 不再赢 |
| `t_s3k8_b6` | 40.3 KB | 0.0262 | 0.0328 | 强制 6 CTA/SM：无差 |

几个可读的规律：

- **低 M 偏好大 KSPLIT**（`t_s2k10`/`t_s3k8`），高 M 偏好小 KSPLIT
  （`t_s2k4`）。这和 46/47 篇「KSPLIT 是并行度」的结论一致：M 小的时候
  `atomicAdd` 归约字节 ∝ M 很小，切细 K 换并行度稳赚；M 一大归约税就反超。
- **`STAGES` 从 2 加到 3 在低 M 有小幅收益**（0.0270→0.0262），但 4/6 反而
  变慢——smem 涨、CTA/SM 降，和 51 篇 `w_s6k8` 的结论同构。
- **occupancy 不是瓶颈**：`t_s2k10` ncu 是 `Block Limit Registers = 6`、
  `Block Limit Shared Mem = 6`，两者同时到顶（复现 51 篇 ZJ3）；强制
  `__launch_bounds__(128, 6/7)` 没有任何收益。

### 5.1 ncu：墙现在在哪

`ncu_tma_s2k10_perm.out.txt`（M=1 附近）对比 51 篇 `wA_s2k8`：

| 指标 | 51 `wA_s2k8` | TMA `t_s2k10` |
|---|---|---|
| DRAM Throughput | 64.69% | **67.19%** |
| L1/TEX Cache Throughput | 31.19% | 35.18% |
| L2 Cache Throughput | 65.35% | 61.90% |
| Compute (SM) | 47.63% | **51.16%** |
| Achieved occupancy | 29.39% | 31.10% |
| registers/thread | 76 | 75 |
| stall `long_scoreboard` | 1.60 | 3.04 |
| stall `barrier` | 0.21 | 0.22 |
| stall `wait` | 0.77 | 0.98 |

DRAM 从 64.7% 抬到 67.2%，`L1/TEX` 仍只有 35%（不再有 bank conflict）。
但**没有任何一级到顶**：这是典型的 latency-bound。TMA 把「每 stage 一次
DEPBAR」换成了「每 stage 一次 mbarrier 等待」，`long_scoreboard` 的绝对值反而
更高（3.04）——因为分母（发射的指令数）变小了，同样的等待摊到更少的指令上；
真正变好的是 DRAM 利用率和端到端时间。

---

## 六、小结与定位

- **TMA warp-private 成立**：K-blocked 权重一个 warp 一个 group = 连续 2 KB，
  1D `cp.async.bulk` + 私有 mbarrier 一把搬入，全 kernel 无 CTA 级
  `__syncthreads`（A 预载那一次除外）。同进程单配置 **1.07×**、逐 M 最优
  **1.09×**相对 51 篇。
- **TMA 不自动等于快；布局是第二道关**：默认行主序 smem 让 IMMA 读 W 变成
  8 路 bank conflict（`L1/TEX` 77%、excess 62%）。host 端一个 `[c][n]` 置换
  布局同时满足「连续 2 KB（TMA 可用）」和「8 个 bgroup 落 8 个 bank（读无冲突）」
  → `L1/TEX` 35%、excess 9%、`M=1` 快 1.08~1.10×。
- **规律**：`M` 小用大 `KSPLIT`（`t_s2k10`）、`M` 大用小 `KSPLIT`（`t_s2k4`）；
  `STAGES=2→3` 小赚，4/6 反亏；occupancy 被 regs 与 smem 同时锁在 6 CTA/SM，
  强制提高无收益。
- **诚实的边界**：`M=1` 仍输 45 篇 GEMV（2408 GB/s / 76.3%）**1.31×**；
  ncu 无一级到顶、纯 latency-bound。**`M=1` 该用 GEMV，IMMA 的价值在 `M≥3`**；
  TMA 版把这条线在高 M 段又推进了一小步。
- **可移植的判据**：当消费者是 `IMMA`/`ldmatrix` 这类「对 smem 行距敏感」的
  取数模式时，**TMA 的默认布局要先用 ncu 的 shared excessive wavefronts 验一遍**；
  连续 2 KB 的约束可以和「无 bank conflict」同时满足，只是布局得自己在 host
  排好，别指望硬件替你 swizzle。

**下一篇候选**：主题 25d（MLA decode 的 producer/consumer warp specialization，
42 篇评估因 smem/寄存器硬阻塞暂缓），或回到 MoE/FP8 主线把「TMA + 置换布局」
的思路用到别的低比特算子。

---

## 复现

```bash
cd code/kernel-opt
# 主实验（51 基线 + TMA 扫参）
scripts/run.sh 52-w4a8-imma-tma/w4a8_imma_tma.cu tma2
# 布局消融（行主序 vs [c][n] 置换）
scripts/run.sh 52-w4a8-imma-tma/w4a8_imma_tma.cu abl
# ncu
scripts/ncu.sh 52-w4a8-imma-tma/w4a8_imma_tma.cu \
  --set full --kernel-name regex:gemm_mma_w4a8_tma_warp --launch-skip 300 --launch-count 1 -- tma t_s3k8
```

代码：`code/kernel-opt/52-w4a8-imma-tma/w4a8_imma_tma.cu`（主 kernel
`gemm_mma_w4a8_tma_warp_kernel`，`:927`），实测原始输出
`tma2_sweep.out.txt` / `tma3_sweep.out.txt` / `ablation.out.txt` /
`final_sweep.out.txt`、ncu `ncu_tma_s2k10_perm.out.txt` /
`ncu_tma_s2k10_row.out.txt` / `ncu_tma_s3k8.out.txt`。
