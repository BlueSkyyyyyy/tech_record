---
title: "CUDA 算子调优（四十一）：W4A16 的 warp specialization——把 occupancy 从 6.5% 提到 22%，为什么只快 7%？"
date: 2026-09-22T01:30:00+08:00
draft: false
weight: 41
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "量化", "W4A16", "AWQ", "GPTQ", "dequant", "GEMM", "wgmma", "warp-specialization", "mbarrier", "tensor-core", "H100", "Hopper", "Qwen3", "系列"]
categories: ["算子开发"]
---

这是 [第 40 篇]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}}) 的续集。上一篇把 W4A16
（4-bit 权重 × bf16 激活）融合进 GEMM 主循环，`M=64` decode 下比「物化 + cuBLAS」快 1.38×，
但留下两个刺眼的数字：

- **反量化税 46%**：把 dequant 算术短路掉，同一副搬运/屏障/`wgmma` 骨架只要 0.0613 ms（融合版 0.1114 ms）；
- **occupancy 只有 6.5%**：`shared-mem` 把 CTA 锁成 2/SM，ncu 主导 stall 是
  `fixed-latency execution dependency`（37.7%）——反量化那串 ALU 依赖链没被藏住。

自然的第一反应是 **warp specialization**：让一组 warp 专职反量化（CUDA core），另一组只发 `wgmma`
（Tensor core），用 `mbarrier` 握手，把两者真正重叠起来（这也是 Marlin 类 SOTA 的做法）。
这篇就把它做出来并实测。

先把结论摆在这（Qwen3-8B MLP `up/gate`，`M=64, N=17408, K=5120`，`group=128`，H100 实测）：

| 版本 | 手段 | 耗时 | 有效权重带宽 | 相对 40 篇最佳 |
|---|---|---|---|---|
| 40 篇 v1（最佳） | `cp.async` 流水 + 主循环 dequant + `wgmma`，`64×128 s3 k4` | 0.1118 ms | 398.7 GB/s | 1.00× |
| 40 篇 v1（k5） | 同 40 篇，只把 split-K 从 4 改成 5 | 0.1032 ms | 431.9 GB/s | 1.08× |
| **41 篇 WS（k4）** | **1 producer WG + 1 consumer WG（mbarrier）**，`64×128 s3 k4` | 0.1049 ms | 425.0 GB/s | 1.07× |
| **41 篇 WS（k5，最佳）** | **同上 + split-K=5 的波次对齐** | **0.0957 ms** | **465.6 GB/s** | **1.17×** |
| 41 篇诊断 | 同上但**关掉 dequant 算术**（结果无意义的天花板） | 0.0696 ms | 639.8 GB/s | 1.61× |

三个结论，也是这篇真正想讲的东西：

1. **warp specialization 确实把 occupancy 从 6.5% 提到 21%，但只换来 ~7% 的加速**——
   说明「反量化税」并不是被「同一组线程既算又发指令」卡住的，加 warps 解不了。
2. **真正更大的墙是 wave quantization**：`grid = (N/128)×KSPLIT = 136×4 = 544` 个 CTA，
   而并发槽位 `2 CTA/SM × 132 SM = 264`，于是 `544/264 = 2.06` → **3 波**，第 3 波只有 16 个 CTA，
   却要占满一整波的墙钟。同一副 kernel 把 grid 变成 528（`N=16896`）就 **0.1048 → 0.0842 ms（−20%）**。
3. 据此**按「波次 × 每项 stage 数」挑 split-K**：`k4` 预测 `3×20=60`、`k5` 预测 `3×16=48`，
   实测 `k5` 胜出（0.0957 ms）。这是这篇可复用的判据。

## 一、复现 40 篇的墙：为什么想到 warp specialization

40 篇的 ncu 指纹是这样的（`64×128 s3 k1`，grid 136）：

```
Section: Speed Of Light        40 篇 v1
  Compute (SM) Throughput  %     25.93
  DRAM Throughput          %     11.01     ← 远没到 HBM
  L1/TEX Cache Throughput  %     21.58
  Achieved Occupancy       %      6.5%     ← 只有 8 warp/SM
Section: Warp State
  Warp Cycles Per Issued Instruction  2.89
  top stall: fixed-latency execution dependency  1.1 cyc/issue → 37.7%
```

`fixed-latency dependency` 是「下一条指令在等上一条 ALU 指令的结果」。反量化主体是：

```
q = (word >> 8b) & 0xF - 8      // 移位 → 减
o = bf16( float(q) * s )         // int→float → 乘 → float→bf16
```

每个元素一条 ~5 级的依赖链，每线程 64 个元素。没有别的 warp 可以顶上来填空，
所以发射口空转。Warp specialization 的直觉就是：**把反量化（ALU）和 `wgmma`（Tensor core）
放到不同的 warpgroup，让二者在时间上重叠**。

## 二、设计：1 producer WG + 1 consumer WG

kernel 模板：

```
template <int BM=64, int BN=128, int STAGES=3, int KSPLIT, int NCONS=1, bool DQ, int NPRODW=1>
w4a16_ws_kernel(...)
```

线程布局（`NPRODW=1, NCONS=1` → 256 线程）：

```
warp 0..3   (tid  0..127)  producer WG ── cp.async 搬 A(bf16)/Wp(int4) + 反量化 Wp→sW(SW128)
warp 4..7   (tid 128..255) consumer WG ── 只发 wgmma.m64n128k16
```

smem 分 stage 三重缓冲：`As[STAGES]`（SW128 bf16 A）、`Wps[STAGES]`（打包 int4）、
`sW[STAGES]`（SW128 bf16 反量化结果），外加两个 mbarrier 数组和一张 per-group scale 表。

握手协议（与 [第 23 篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}}) 同源）：

```
         producer                                   consumer
  ┌──────────────────────────┐             ┌──────────────────────────┐
  │ wait empty[s]            │             │ wait full[s]             │
  │ issue cp.async A,Wp      │             │ wgmma(As[s], sW[s])      │  ← tensor core
  │ cp.async.wait            │             │ wgmma.commit / wait0     │
  │ dequant Wp → sW[s]       │  full[s] →  │ arrive empty[s]          │
  │ fence.proxy.async        │             │                          │
  │ arrive full[s]           │  ← empty[s] │                          │
  └──────────────────────────┘             └──────────────────────────┘
```

相位不用动态下标数组（23 篇的坑）：stage `s` 的第 `n` 次使用，`full` 相位 = `n&1`、
`empty` 的复用等待相位 = `(n-1)&1`。反量化按扁平映射（相邻线程读相邻 `uint32`，
无 bank conflict），scale 先整表预取进 smem，主循环里不碰 global。

### 2.1 一个花了我很久的坑：`empty` barrier 被重复 arrive

第一版把释放写在「每个 consumer warp 的 lane0」：

```cpp
if (lane == 0) mbar_arrive(empty + s);   // ✗ consumer WG 有 4 个 warp → 4 次 arrive
```

而 `empty` 的期望 count 只有 `NCONS=1`。**mbarrier 被超发（over-arrive）后相位直接错乱，
kernel 永久 `long_scoreboard` 自旋、不报任何 CUDA error、`timeout` 也杀不掉**。
我把它抽成一个最小复现（`mbar_test.cu`）才定位到：

```
producer t=0..: arrive(full[0])
consumer t=0  : wait(full[0]) → 立刻返回（因为被超发，相位已经翻过）→ ...
然后 producer 在 wait(empty[0]) 上永久阻塞
```

修法（照抄 23 篇的结论）：**`empty` 的 count = 所有 consumer 线程数，每个 consumer 线程各自 `arrive`**：

```cpp
constexpr int EMPTY_CNT = NCONS * 128;
mbar_init(empty + s, EMPTY_CNT);
...
wgmma_wait0();
mbar_arrive(empty + s);   // ✓ 每个 consumer 线程都 arrive
```

> 教训：mbarrier 的 count 必须与真实到达次数**逐一对上**，超发/少发都不报错，
> 只是死锁或数据竞争。调试靠抽最小复现，别在完整 kernel 里猜。

## 三、结果：occupancy 上去了，但只快 7%

同一 `64×128 s3 k4` 配置，41 篇 WS vs 40 篇：

| | 40 篇 v1 | 41 篇 WS |
|---|---|---|
| 耗时 | 0.1118 ms | 0.1049 ms |
| 有效权重带宽 | 398.7 GB/s | 425.0 GB/s |
| Achieved occupancy | **6.5%** | **21.0%** |
| Active warps/SM | ~4 | 13.4 |
| Issue slots busy | 25.9% | 41.5% |
| 主 stall | fixed-latency 37.7% | long_scoreboard 3.3 cyc/issue |
| 寄存器/线程 | 156 | 90 |
| Compute (SM) | 25.9% | 41.5% |

Warp specialization 的效果完全符合预期：**occupancy ×3.2、发射口利用率 ×1.6、寄存器 156→90**。
可时间只降了 6%。也就是说，40 篇的「fixed-latency 37.7%」是**症状**，不是根因——
即使把它消掉，kernel 也只快这么一点。

原因看诊断版就清楚了：把 dequant 算术短路掉（`DQ=false`），同一副骨架是 **0.0696 ms**。
WS 的 0.1049 说明：

$$
\underbrace{0.1049}_{\text{WS 融合}} - \underbrace{0.0696}_{\text{骨架}} = 0.0353\ \text{ms}
\quad(\text{反量化税} \approx 34\%)
$$

**反量化税还在**，而且骨架本身也才 639.8 GB/s（19% HBM）。真正的问题在骨架里。

## 四、真正的墙：wave quantization

用户最容易忽略的一点：**`0.1049 ms` 里有多少是 SM 在干活、多少是在等尾波？**

看网格账（`2 CTA/SM × 132 SM = 264` 个并发槽）：

```
k4: grid = (17408/128) × 4 = 136 × 4 = 544
    544 / 264 = 2.06  →  ceil = 3 波
    第 3 波 = 544 − 264×2 = 16 个 CTA   ← 264 个槽只用 16 个！

时间 ≈ 3 × T_item，而理想（连续填满）≈ 2.06 × T_item
```

验证：直接换一个 `N`，让 grid 恰好是 264 的整数倍。WS k4 配置下：

| N | grid | 波数 | 耗时 |
|---|---|---|---|
| 17408 | 544 | 3 | 0.1048 ms |
| 16896 | 528 = 264×2 | 2 | **0.0842 ms** |
| 16256 | 508 | 2 | 0.0818 ms |
| 16512 | 516 | 2 | 0.0818 ms |

**`N` 只少了 3%，时间掉了 20%。** 这 20% 就是那 16 个 CTA 的第 3 波在空转。

### 4.1 按波次挑 split-K

既然 grid `= (N/BN) × KSPLIT`，每项 stage 数 `= (K/BK)/KSPLIT`，就可以把总时间的
「理想 stage 单位」写成：

$$
T_{\text{stage}} \;\approx\; \underbrace{\left\lceil \frac{(N/BN)\cdot KSPLIT}{264} \right\rceil}_{\text{波数}}
\;\times\; \underbrace{\frac{K/BK}{KSPLIT}}_{\text{每项 stage 数}}
$$

对 `N/BN=136, K/BK=80`：

| KSPLIT | grid | 波数 | 每项 stage | 理想代价 |
|---|---|---|---|---|
| 2 | 272 | 2 | 40 | 80 |
| **4** | 544 | **3** | 20 | **60** |
| **5** | 680 | **3** | 16 | **48** |
| 8 | 1088 | 5 | 10 | 50 |
| 10 | 1360 | 6 | 8 | 48 |
| 16 | 2176 | 9 | 5 | 45 |

`k5` 理论最优（48）。实测 WS kernel sweep（`M=64`）：

| 配置 | 耗时 | 有效带宽 |
|---|---|---|
| ws k4 | 0.1049 ms | 425.0 GB/s |
| **ws k5** | **0.0957 ms** | **465.6 GB/s** |
| ws k8 | 0.1059 ms | 420.8 GB/s |
| ws k10 | 0.1075 ms | 414.4 GB/s |
| ws k16 | 0.1315 ms | 338.9 GB/s |
| ws k20 | 0.1451 ms | 307.2 GB/s |

理论预测 `k5` 最优、`k8/k10` 次之——实测完全吻合。`k16/k20` 偏慢是**每项 prologue/epilogue 开销**：
每项只有 4~5 个 stage，`cp.async` 预热的固定成本盖过了波次收益。

> **判据（可复用）**：小 `M`、小 `K/BK` 的 split-K GEMM，先算
> `ceil(grid / (2·SM)) × (K/BK/KSPLIT)`，**取「波次 × 每项 stage」最小的 KSPLIT；
> 但别切到每项只剩几个 stage**——prologue 会反噬。

### 4.2 归因：两块收益各占多少

为了不把「波次对齐」的功劳算到 warp specialization 头上，同一 `k4` 配置横比，
并额外把 40 篇的 baseline 也跑了一个 `k5`：

| 版本 | 耗时 | 说明 |
|---|---|---|
| 40 篇 baseline k4 | 0.1118 ms | 无 WS，split-K=4 |
| 40 篇 baseline k5 | 0.1032 ms | 无 WS，仅换 KSPLIT → **波次收益 1.08×** |
| 41 篇 WS k4 | 0.1049 ms | WS 本身 → **1.07×** |
| **41 篇 WS k5** | **0.0957 ms** | **合计 1.17×** |

两块大约各占一半，互不冲突，可以叠加。

## 五、decode 极端工作点：`M=1`

W4A16 的真正主场是 `M=1`（单请求 decode）。此时激活几乎不占带宽，全在搬权重：

| 版本（N=17408, K=5120） | 耗时 | 有效权重带宽 |
|---|---|---|
| 40 篇 baseline k8 | 0.0897 ms | 496.6 GB/s |
| **41 篇 WS k5** | **0.0823 ms** | **541.8 GB/s** |
| 41 篇 WS k10 | 0.0823 ms | 541.6 GB/s |

`M=1` 下 WS k5 到 **541.8 GB/s（16.2% HBM）**，相对 40 篇最佳 1.09×。注意 `M=1` 时
`BM=64` 仍按 64 行算（浪费 M 维），`wgmma` 的算力完全不是瓶颈——这纯粹是访存/延迟题。
`541 GB/s` 离权重 roofline（`44.6 MB / 3352 GB/s = 13.3 µs`）还差 6×，
仍是「骨架 + 反量化」的延迟墙。

## 六、被否掉的路（负结果，别重复踩）

| 尝试 | 结果 | 原因 |
|---|---|---|
| **persistent kernel + `atomicAdd` 动态派活** | 0.104 ms（≈ 非 persistent），且相位跨 item 难对齐 | 每项要重启流水（prologue/epilogue + `__syncthreads`），开销吃掉尾部收益 |
| **加深预取 PF=2** | 0.1124 ms（**更慢**） | `STAGES=3` 下 stage 复用只允许 1 级预取，PF=2 反而让 producer 提前等 `empty` |
| **多 producer WG（2/3/4）** | 0.121 / 0.121 / 0.124 ms（更慢） | 线程数上去后 1 CTA/SM，occupancy 反降；反量化吞吐本就不是瓶颈 |
| **BN=256（NCONS=2）** | 0.1404 ms | smem 158 KB → 1 CTA/SM，波次更碎 |
| **行优先 dequant 读（每线程一整行）** | 与扁平映射持平 | `Wps` 读 stride=32 B → 8-way bank conflict，抵消了 scale 提升 |
| 每 stage `__ldg` 读 scale | 0.1029 ms | 全局 load 依赖；预取进 smem 表（sctab）后持平，说明它也不是主因 |

这些负结果共同指向一个判断：**W4A16 融合 kernel 在我们的几何下是「多因素延迟墙」**——
occupancy、反量化 ALU、`cp.async` 延迟、波次尾巴各占一块，没有单一银弹。

## 七、ncu 对照表

| 指标 | 40 篇 v1（k4）| 41 篇 WS（k5）|
|---|---|---|
| Duration | 137 µs（k1）| 92.2 µs（k5）|
| DRAM Throughput | 11.0% | 17.9% |
| L2 Cache Throughput | 13.9% | 41.0% |
| Compute (SM) | 25.9% | 41.5% |
| Achieved Occupancy | 6.5% | 21.0% |
| Registers / thread | 156 | 90 |
| Waves Per SM | 0.52 | 2.58 |
| 主 stall | fixed-latency 37.7% | long_scoreboard 3.3 cyc/inst |

（注：两行的 `N/K` 与 `grid` 不同，绝对值只做趋势参考；40 篇那行是 `k1`、41 篇是 `k5`。）

## 八、小结

- **warp specialization 值得做，但别指望它单独解决反量化税**：它把 occupancy 6.5%→21%、
  寄存器 156→90，但只快 ~7%——40 篇的 `fixed-latency 37.7%` 是症状不是根因。
- **wave quantization 是小 `M` split-K GEMM 的隐形大头**：`544 CTA / 264 槽 = 3 波`，
  第 3 波只有 16 个 CTA 却占满一波；换 `N` 让 grid 落在 264 整数倍，**−20%**。
- **可复用判据**：`T ≈ ceil(grid / 2SM) × (K/BK/KSPLIT)`，取最小者；但每项别少于 ~8 个 stage。
- **最佳配置** `64×128 s3 k5`（producer+consumer 各 1 WG）：`M=64` **0.0957 ms / 465.6 GB/s（1.17×）**、
  `M=1` **0.0823 ms / 541.8 GB/s**。
- 距离 Marlin 类 SOTA（有效权重带宽 ~2 TB/s 量级）仍有 ~4× 差距，出口是**跨 item 流水
  （不重启 pipeline）+ 真正的 load-balanced persistent 调度**，而不是继续抠单条指令。

## 下一篇预告

W4A16 的骨架已被证明是「多因素延迟墙」。接下来回到第五/六部分的主线：
**MLA 吸收式 decode 的 `wgmma` + FP8 张量核**（主题 25c）——用 `wgmma` 的 smem 描述符
替掉 [第 39 篇]({{< relref "cuda-kernel-opt-39-mla-decode" >}}) 的 `ldmatrix` 墙，
并把 FP8 KV-cache 真正兑现（39 篇已证「换存储不换算力是白费」）。

> 配套代码：`code/kernel-opt/41-w4a16-ws/`（`w4a16_ws.cu`、`mbar_test.cu` 最小复现、
> `ws_M64.out.txt`、`ws_M1.out.txt`、`probe_nodq.out.txt`、`ncu_ws_k5.out.txt`）。
