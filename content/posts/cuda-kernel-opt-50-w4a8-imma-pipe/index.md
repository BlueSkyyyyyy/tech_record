---
title: "CUDA 算子调优（五十）：W4A8 IMMA 的收尾——转置 scale、K-blocked 权重布局与「挤不动的两倍」"
date: 2026-09-22T05:00:00+08:00
draft: false
weight: 50
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A8", "int4", "int8", "IMMA", "mma.sync", "cp.async", "软件流水", "访存局部性", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第十篇，也是 W4A8/IMMA 这条线的收尾。
承接 [第 46 篇小 M 分派]({{< relref "cuda-kernel-opt-46-w4a16-smallm-dispatch" >}})、
[第 47 篇小 M 的 warp specialization]({{< relref "cuda-kernel-opt-47-w4a16-mma-ws" >}}) 和
[第 48 篇 W4A8 的整数张量核]({{< relref "cuda-kernel-opt-48-w4a8-imma" >}})。

48 篇把「反量化 ALU 墙」用整数张量核（IMMA）砍掉了：同一批线程不再需要
`int4→bf16 + I2F + F2BF`，一条 `mma.m16n8k32.s8.u8` 直接吃 int8，指令数
−59%、权重带宽 650→1290 GB/s。但 48 篇结尾留了一句话：

> 换完数制后，墙会从 ALU 挪到**访存延迟**（`long_scoreboard` 1.58→2.74、
> `No Eligible` 57%、occ 23%）……该做的是加深流水 / 消 strided scale 读 / 持久化。

本篇把这三件事（外加 K-blocked 权重布局、单 barrier、wider BN）**逐条实测**。
结论先给：

- **真正有效的是「K-blocked 权重布局」**：把权重从 `Wp[N][K/2]`（一行一行取）
  重排成 `Wt[g][N][64B]`（一个 stage 的 BN 行连续 8 KB），让 `cp.async`
  从「128 段跨 2560 B 的 64 B 碎片」变成「一整块连续 8 KB」。这直接把
  `long_scoreboard` 从 **1.71 打到 0.30（5.7×）**、DRAM 利用率 **49%→60%**。
- 配合 `KSPLIT=8` 的波次对齐，M=1 权重带宽 **1446→1682 GB/s（+16%）**，
  全 M 段总和 **0.872×**（−12.8%）。
- **转置 scale、单 barrier、wider BN、跨 item 持久化全是小账或负结果**——
  尤其是 43 篇的持久化流水搬到这里**完全无效**，后面会解释为什么。

shape 全部取自 `/ssd/models/qwen3-8B/config.json`：
`hidden_size=5120, intermediate_size=17408, group_size=128`，对称 int4；
默认测 `N=17408, K=5120, M=1..16`（decode）。权重 44.6 MB(int4) + scale 2.8 MB。

---

## 一、先把 48 的墙量清楚

48 篇最优是 `imma_ns_s2k7`（`SCTAB=false` + `KSPLIT=7`，128 线程）：
`M=16` **0.0386 ms**、`M=1` **0.0328 ms**。它每一步（一个 128-k 的 group）：
`cp.async` 搬 A（2 KB）和 W（8 KB）→ `wait_prior` → `__syncthreads` →
4 条 `k32` 的 IMMA × 4 个 n8 tile + `PRMT` 展开 nibble → 每 group 用
`sa·sw` 折算回 fp32。

用 ncu 打 `M=1` 的 stall 分解（`--metrics smsp__average_warps_issue_stalled_*`）：

| 指标 | base48 `imma_ns_s2k7` (M=1) |
|---|---|
| DRAM Throughput | 49.2% |
| L2 Throughput | 54.5% |
| Achieved Occupancy | 22.2% |
| stall `long_scoreboard` | **1.71**（第一） |
| stall `wait` | 1.15 |
| stall `not_selected` | 0.85 |
| stall `barrier` | 0.69 |
| 指令数 | 11.91 M |
| 寄存器 | 101 |

也就是说：核函数**一半时间在等全局权重到 smem**（`long_scoreboard`），
occupancy 只有 22%（4 CTA/SM），DRAM 才跑到峰值的一半。典型的「延迟受限」。

---

## 二、先证伪一个直觉：不是 DRAM 访问模式的问题

看到「64 B 一段、行距 2560 B」的权重读法，第一反应是 **DRAM 行缓冲局部性差**：
每个 `(n, stage)` 只读 64 B，然后跳到 2.5 KB 外的下一行。

我们写了个诊断 kernel（`wread_strided_kernel`），用**完全相同的地址流**
（`Wp + n*(K/2) + s*64 + c*16`，grid-stride、只读不算），和顺序读的 roof 对照：

```text
纯读上界 w+s（顺序 uint4）      : 0.0184 ms  2575 GB/s (76.8% HBM)
纯读上界 strided(64B/2560B)     : 0.0175 ms  2546 GB/s (75.9% HBM)
```

**strided 读和顺序读一样快（甚至略快）**。所以 DRAM 完全不在乎这个 stride——
H100 的访存请求聚合器把大量并发 CTA 的 64 B 碎片合成了满带宽的流。
这条直觉被证伪，省得我们在错误的路上做 pre-transpose。

> 方法论：**怀疑某段访存的「模式」是瓶颈时，先写一个只做这段访存、不做别的
> 诊断 kernel**。如果它到不了 roof，才值得动布局；如果它和 roof 一样快，
> 说明瓶颈在「怎么发这些请求」（cp.async 粒度 / 流水深度 / 屏障），不是地址。

---

## 三、招式 A：转置 scale，消掉 strided 的散点读

48 的折算阶段按 `Ws[(col+n)*NGRP + g]` 取权重 scale：每个 lane 要读的
`(col+n)` 不同、步长 `NGRP=40`，于是每条 `__ldg` 命中一个**互不相邻的
32 B sector**。全局 scale 才 2.8 MB，但 L1 sector 的无效流量不便宜。

改法：**预先把 scale 转置成 `[g][N]` / `[g][M]`**（`WsT` / `AxsT`），
一个 group 的 128 个 scale 是**连续 512 B**，直接 `cp.async` 进 smem，
和权重走同一条流水；折算时从 smem 读、零 global：

```cuda
// load_stage 里：转置后的 scale 是连续 512B / 64B，整段搬
for (int i = tid; i < BN / 4; i += NTHREADS)
  __pipeline_memcpy_async(&sw[i * 4], &WsT[(size_t)g * N + block_col + i * 4], 16);
```

`AxsT` / `A-sum×8`（`u8.s8` 的 `8·Σq_a` 修正）也是同样处理。
实测（同进程，`M=16`）：`imma_ns_s2k7` 0.0386 → `pipe_s3k7` **0.0371（+4%）**。

**招式 A 只值 4%**——因为长 `long_scoreboard` 的主因是 8 KB 的权重，
2.8 MB 的 scale 只是陪跑。于是把注意力放回权重。

---

## 四、招式 B（主功）：K-blocked 权重布局

虽然 DRAM 不在乎 stride（第二节），但 **`cp.async` 在乎**。原布局下，
一个 stage 的权重是 BN=128 段、每段 64 B、段间跨 2560 B：`cp.async` 的
16 B 请求只能在**行内 4 个**之间合并成 64 B，跨行无法合并，于是每个
stage 产生 128 个独立的 64 B 请求、L2 sector 也被切得更碎。

改法：**把 K 维切块重排**——`Wt[g][N][64B]`，即把「第 g 个 128-k 块、
第 n 行的那 64 B」搬到 `Wt` 的 `g*N*64 + n*64` 处：

```text
原布局 Wp[N][K/2]：       Wt[g][N][64B]（K-blocked）：
  stage g 取每行 64B             stage g 的 BN 行 = 一整块连续 8KB
  n=0  [===].................     g=0: [n0 64B][n1 64B]...[n127 64B]  ← 8KB 连续
  n=1  ....[===].............     g=1: [n0 64B][n1 64B]...[n127 64B]  ← 8KB 连续
  n=2  ........[===].........     ...
  ...   行距 2560B, 段 64B
```

重排只是一次性 host 端 `memcpy`（或模型加载时完成），kernel 里只改一个地址：

```cuda
if constexpr (TILED) {
  __pipeline_memcpy_async(&wp[r * WPAD + c],
      &Wp[(size_t)g * N * (LARGE_BK/2) + (block_col + r) * (LARGE_BK/2) + c], 16);
} else {
  __pipeline_memcpy_async(&wp[r * WPAD + c],
      &Wp[(size_t)(block_col + r) * (K/2) + k0/2 + c], 16);
}
```

smem 里仍是 `[BN][64B]` 行主序，所以 IMMA 的 `W16[wo]` 取数**一字不改**。

效果（同进程，`M=1`）：

| 版本 | ms | W+s GB/s | %HBM | `long_scoreboard` | DRAM | occ |
|---|---|---|---|---|---|---|
| `base48_ns_s2k7` | 0.0328 | 1446 | 43.1% | 1.71 | 49.2% | 22.2% |
| `tile_s2k8_b5`（K-blocked + KSPLIT=8） | **0.0283** | **1672** | **49.9%** | **0.30** | **60.1%** | 26.3% |

**`long_scoreboard` 1.71 → 0.30（5.7×），DRAM 49%→60%。** 这就证明：
原布局的 `cp.async` 碎片化让「等权重」成了主 stall；重排后请求变粗，
访存延迟被藏住了。

顺带把寄存器从 101 压到 **86**（少了 `__ldg` 的地址计算），occupancy
22.2%→26.3%（4→5 CTA/SM）。

> **为什么会这样**：`cp.async` 的合并窗口比普通 `ld.global` 小得多——它
> 只把「同一 instruction、连续地址」的 lane 合并。原布局下一行只有 64 B
> 连续，跨行跳 2560 B，跨行永远合不了。K-blocked 之后，一个 stage 的
> 128×64 B = 8 KB 完全连续，`cp.async` 可以一路合成大块。
> （DRAM 侧的诊断之所以没差别，是因为它是 grid-stride 的 `uint4` load，
> 相邻线程天然连续，跟 `cp.async` 的行内合并窗口不是一回事。）

---

## 五、招式 C：波次对齐的 `KSPLIT`

`KSPLIT` 决定网格与每 CTA 的 stage 数：`grid = (N/BN) × KSPLIT`。
`N/BN=136`、2 CTA/SM… 5 CTA/SM 后一波 5×132=660 个 CTA：

| KSPLIT | grid=136×KSPLIT | 波数（660 槽） | M=1 ms | 每 CTA stage 数 |
|---|---|---|---|---|
| 4 | 544 | 0.82 | 0.0309 | 10 |
| 5 | 680 | 1.03 | 0.0314 | 8 |
| 7 | 952 | 1.44 | 0.0309 | ~6 |
| **8** | 1088 | 1.65 | **0.0283** | 5 |
| 10 | 1360 | 2.06 | 0.0292 | 4 |
| 16 | 2176 | 3.30 | 0.0298 | 2.5 |

`KSPLIT=8` 是甜点（每 CTA 5 个 group）。这和 46/47 的结论一致——**延迟受限的
小 M，`KSPLIT` 粒度本质是「同时可发射的独立访存流条数」**；太细（16）每 CTA
只剩 2.5 个 stage，流水预热/排空反噬。

---

## 六、ncu 对照与流水剖析

同进程对拍 `M=1`（`ncu --metrics dram__throughput...`）：

| 指标 | base48 | tile_s2k8_b5 | 变化 |
|---|---|---|---|
| DRAM Throughput | 49.2% | **60.1%** | +10.9pp |
| L2 Throughput | 54.5% | 61.8% | +7.3pp |
| Achieved Occupancy | 22.2% | 26.3% | +4.1pp |
| 寄存器 | 101 | 86 | −15 |
| stall `long_scoreboard` | **1.71** | **0.30** | **−82%** |
| stall `barrier` | 0.69 | 1.61 | +0.92 |
| stall `wait` | 1.15 | 1.04 | −0.11 |
| 指令数 | 11.91 M | 11.56 M | −3% |

瓶颈画像完全换了：原来等全局访存（`long_scoreboard` 第一），现在主 stall
变成 **`barrier`**——两个 `__syncthreads()`/stage 的同步代价浮出水面。
这正是下一道墙（见第八节）。

---

## 七、三条负结果（省你时间）

### 7.1 单 barrier 流水：中性

既然 `barrier` 成了第一 stall，就把「一 stage 两个 barrier」改成「一个」：
把下一次 `load_stage` 挪到 barrier 之后（写回 `(t-1)%S` 那个已消费的缓冲），
于是同一个 `__syncthreads()` 同时保证「stage t 的 `cp.async` 对全 CTA 可见」
与「所有线程已读完 t-1」：

```cuda
// ONEBAR：wait → sync → 载下一 stage → 消费
__pipeline_wait_prior(STAGES - 2);
__syncthreads();
if (tnext < nt) load_stage(st == 0 ? STAGES-1 : st-1, kb0 + tnext);
else __pipeline_commit();
```

实测 `one_s3k7` 0.0317 vs `pipe_s3k7` 0.0322——**只快 1.5%，在噪声内**。
`tile_one_s3k7` 0.0314 vs `tile_s3k7` 0.0313，同样中性。原因：`barrier` 高是
**低 occupancy 的症状**（13.8 个 active warp/SM），少一次 barrier 并不能凭空
造出更多可发射的 warp。真要治得靠 warp specialization，但 47 篇已经证明
把「重活」拆进 producer 只会让 `long_scoreboard` 裸奔。

### 7.2 Wider BN（256/512）：负

`BN=256`（8 warp）把 A 的重读从 136 次减到 68 次，但 smem 翻倍、每 bar
rier 要同步的 warp 更多，实测 `tile_bn256_s2k7` 0.0366 vs `tile_s2k7`
0.0353；`BN=512` 直接崩到 0.055（2 CTA/SM）。

### 7.3 跨 item 持久化流水：**完全无效**

43 篇在 W4A16 上把 `(item, kb)` 展平成一维 stage 流、跨 item 不停机，
赢了 1.04×。把它移植到 IMMA（`imma_persist_kernel`）后：

| 版本 | M=1 | M=16 |
|---|---|---|
| `tile_s2k8_b5`（KSPLIT 网格） | **0.0283** | 0.0349 |
| `pers_c8_s3_p4sm`（持久化，P=4×SM） | 0.0334 | 0.0380 |
| `pers_c8_s3_p2sm`（P=2×SM） | 0.0390 | 0.0411 |
| `pers_c40_s3_p2sm`（CHUNK=全 K，无 atomic） | 0.0596 | 0.0600 |

持久化**全线更慢**，慢了 18–110%。原因很直接：

1. **没有尾波可消**。43 篇的 W4A16 是 `wgmma.m64`，每个 CTA 的 item 很粗，
   网格小、尾波浪费大；IMMA 的 `BM=16` tile 本来就细，136×8 已有足够并发，
   尾波不是主要矛盾。
2. **每个 item 边界要折 fp32 + `atomicAdd`**，而 IMMA 每个 group 本来就要
   折算（不像 W4A16 只在 item 末尾折一次），持久化没有省下任何折算。
3. `CHUNK=40`（无 atomic、每 n-tile 一个 item）只有 136 个 item，`P=264`
   一半 CTA 空转，直接退化成 0.06 ms。

> 教训：**持久化的收益 ∝ 「被消掉的尾波 + 被摊薄的 prologue」**。
> 网格本身已经够大、prologue 已经够短时，持久化只有额外开销。

---

## 八、最终结果

同进程、同 warmup 的 M 扫描（`warmup=5, iters=50`，8 个 M 点）：

| M | base48 (ms) | 本篇最佳 (ms) | 加速 | 最佳 kernel | W+s GB/s | %HBM |
|---|---|---|---|---|---|---|
| 1 | 0.0328 | **0.0283** | 1.16× | `tile_s2k8_b5` | 1672 | 49.9% |
| 2 | 0.0330 | **0.0284** | 1.16× | `tile_s2k8_b5` | 1668 | 49.8% |
| 3 | 0.0334 | **0.0284** | 1.18× | `tile_s2k8_b5` | 1670 | 49.8% |
| 4 | 0.0339 | **0.0288** | 1.18× | `tile_s2k8` | 1645 | 49.1% |
| 6 | 0.0344 | **0.0297** | 1.16× | `tile_s2k8` | 1596 | 47.6% |
| 8 | 0.0351 | **0.0306** | 1.15× | `tile_s2k8` | 1546 | 46.1% |
| 12 | 0.0367 | **0.0327** | 1.12× | `tile_s2k8` | 1448 | 43.2% |
| 16 | 0.0386 | **0.0343** | 1.13× | `tile_s2k4` | 1380 | 41.2% |
| **总和** | **0.2756** | **0.2404** | **1.147×** | — | — | — |

纯读上界（同口径读 W+s）是 **0.0184 ms / 2575 GB/s（76.8% HBM）**。
所以 `M=1` 现在跑到 roof 的 **65%**（1672/2575），而 base48 只有 56%。
剩下的 35% 差距，就是下一节说的 `barrier` + 26% occupancy 的天花板。

对标 SOTA：48 篇给过口径——AWQ/GPTQ 的 Marlin 类 kernel 在 `M=1` 的有效
int4 带宽约 400 GB/s（受反量化限制），本篇 **1672 GB/s**，相对 40 篇的
~6.7× 差距收到 **~2.6×**（口径不同，仅供参考）；纯读上界 2575 GB/s 是这条
路的物理终点。

---

## 九、小结

- **K-blocked 权重布局（`Wt[g][N][64B]`）是主功**：`cp.async` 的合并窗口远
  小于 DRAM 的聚合能力，把「BN 段、跨 2560 B 的 64 B 碎片」拼成「连续 8 KB」
  后，`long_scoreboard` **1.71→0.30**、DRAM **49%→60%**、寄存器 101→86。
- **诊断先行**：用「同地址流的只读 kernel」证伪了「DRAM 行缓冲局部性」的
  直觉（strided 与顺序读同速）——**别在错误的方向做 pre-transpose**。
- **转置 scale** 只值 4%（scale 流量小）；**单 barrier** 中性；
  **wider BN** 负；**跨 item 持久化**在这条线上完全无效（没有尾波可消、
  折算省不下、item 边界还有额外 atomic）。
- **KSPLIT=8 是甜点**：小 M 的 `KSPLIT` 本质是「并发访存流条数」，太粗没并发、
  太细流水预热反噬。
- **最终**：`M=1` **0.0283 ms / 1672 GB/s / 50% HBM**，全 M 段 **0.872×**
  （−12.8%）；相对纯读上界还有 ~1.55× 的 gap，墙是 `barrier`（低 occupancy
  的症状），不是访存模式。
- **数值**：整数 GEMM 精确，相对误差 1e-7；量化 kernel 与 CPU 参考逐位一致。

**下一篇预告**：主题 25d（MLA decode 的 producer/consumer warp specialization）
在 42 篇评估后因 231 KB smem + 254 寄存器锁死 1 CTA/SM 暂缓；重新捡起来之前，
先把本篇的 `barrier` 墙用「warp-private pipeline（每 warp 独立 stage、零
`__syncthreads`）」或 TMA 版权重搬运试一遍。代码见
`code/kernel-opt/50-w4a8-imma-pipe/`。
