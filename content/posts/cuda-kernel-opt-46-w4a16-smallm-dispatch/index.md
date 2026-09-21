---
title: "CUDA 算子调优（四十六）：M∈[1,64] 的 W4A16 decode——GEMV 与张量核的交叉点不是 M≈78，而是 M≈2.5 / M≈17"
date: 2026-09-22T04:05:00+08:00
draft: false
weight: 46
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A16", "int4", "AWQ", "GPTQ", "GEMV", "tensor-core", "wgmma", "mma.sync", "roofline", "分派", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第六篇。承接
[第 40 篇 W4A16 dequant-GEMM]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}})、
[第 41 篇 warp specialization]({{< relref "cuda-kernel-opt-41-w4a16-warp-specialization" >}})、
[第 43 篇跨 item 持久化流水]({{< relref "cuda-kernel-opt-43-w4a16-persist" >}})、
[第 44 篇 M=1 GEMV]({{< relref "cuda-kernel-opt-44-w4a16-decode-gemv" >}}) 和
[第 45 篇 W4A8+dp4a]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}})。

44 篇留了一个明确的预测：W4A16 的算术强度 $AI(M)\approx 3.77M$，H100 的 bf16 张量核
ridge 是 $989/3350\approx 295$ FLOP/byte，所以

$$M^\* \approx \frac{295}{3.77} \approx 78$$

「只有 $M\gtrsim 78$ 张量核才划算，M=1 就该回去做 GEMV」。这一篇要**用实测把这条预测
证伪**：真正的交叉点根本不在 78，而在 **M≈2.5**；而且张量核内部还分两段——小 M 属于
`mma.sync` 的小 tile，大 M 才轮到 `wgmma.m64`。我们给出三条路径、12 个 M 点的完整实测，
并落成一个按 M 选 kernel 的**运行时分派器**。

---

## 1. 为什么 78 这个数字靠不住

44 篇那张 roofline 图有一个隐含前提：**两条路都跑在各自的 roofline 上**。但小 M 的
张量核根本不在 roofline 上。43 篇实测 `wgmma.m64` 的 M=1 是 0.0798 ms，而纯读权重的
上界是 0.0182 ms——它只有 **18% HBM 带宽**，是被延迟/occupancy 卡住，不是被算力卡住。

反过来，GEMV 的时间随 M **线性增长**：每多一个 token 组（`MT` 个 token）就要把
47.3 MB 的权重再读一遍。

- 张量核：时间几乎**常数**（一个 64 行的 tile 不管 M 是多少都做同样多的活）；
- GEMV：时间 $\propto \lceil M/MT\rceil$。

两者相交的位置，取决于**谁先把权重读够**，而不是算术强度。下面用真实 shape 把它测出来。

## 2. 实验设计

真实 shape 取自 `/ssd/models/qwen3-8B/config.json`：

| 项 | 值 | 来源 |
|---|---|---|
| hidden（K） | 5120 | `hidden_size` |
| intermediate（N） | 17408 | `intermediate_size`（MLP up/gate） |
| group_size | 128 | AWQ/GPTQ 常见 |
| 权重 | 对称 int4，`[N, K/2]` u8 + `[N, K/128]` fp32 | |
| 权重字节 | int4 44.6 MB + scale 2.8 MB = **47.3 MB** | |
| 激活 | bf16 `[M, K]` | |
| 纯读权重上界 | **0.0182 ms / 2598 GB/s（77.5% HBM）** | `wread_roofline` |

三条路径（都在**同一个进程、同一份数据**上跑 head-to-head）：

1. **GEMV**：每 warp 一行（`RWW` 行）、warp 内沿 K 并行、`MT` 个 token 共享一次权重读
   （44 篇骨架 + 多 token 扩展），bf16 激活 + int4→fp32 反量化 + FMA；
2. **mma.sync 小 tile GEMM**：`mma.m16n8k16`，`BM∈{16,32,64}`、`BN=128`、`BK=64`，
   `cp.async` 双缓冲主循环内 int4→bf16 反量化 + `ldmatrix`；本系列新写；
3. **wgmma.m64 GEMM**：41/43 篇的 `wgmma.m64n128k16` + warp specialization（BM=64 固定）。

其中 (2) 是本篇的新代码，目的是检验一个假设：**把 BM 从 64 降到 16，能不能靠「不白算
63/64 行」把张量核的可用区间往左推？**

## 3. 现象：12 个 M 点的完整扫描

`N=17408, K=5120`，warmup=5 / iters=50，同进程。括号是该配置下的**权重读遍数**
`ceil(M/MT)` 或 `ceil(M/BM)`，这是理解一切的关键。

| kernel\M | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| `gemv_r2_mt1` | **0.0309**(1) | 0.0594(2) | 0.1164(4) | 0.2304(8) | 0.4590(16) | 0.9158(32) | 1.8293(64) |
| `gemv_r4_mt2` | 0.0374(1) | **0.0379**(1) | 0.0736(2) | 0.1447(4) | 0.2871(8) | 0.5722(16) | 1.1428(32) |
| `gemv_r2_mt4` | 0.0783(1) | 0.0789(1) | 0.0799(1) | 0.1574(2) | 0.3126(4) | 0.6233(8) | 1.2442(16) |
| `gemv_r2_mt8` | 0.1454(1) | 0.1461(1) | 0.1476(1) | 0.1513(1) | 0.3004(2) | 0.5991(4) | 1.1965(8) |
| `mma_b16_k8` | 0.0700(1) | 0.0702(1) | **0.0706**(1) | **0.0715**(1) | **0.0750**(1) | 0.1214(2) | 0.2188(4) |
| `mma_b32_k8` | 0.0778(1) | 0.0780(1) | 0.0782(1) | 0.0790(1) | 0.0816(1) | 0.0883(1) | 0.1505(2) |
| `mma_b64_k4` | 0.1131(1) | 0.1131(1) | 0.1132(1) | 0.1133(1) | 0.1149(1) | 0.1187(1) | 0.1277(1) |
| `wgmma_m64_k5` | 0.0829(1) | 0.0830(1) | 0.0832(1) | 0.0827(1) | 0.0837(1) | **0.0865**(1) | **0.0956**(1) |

（完整 12 点、含 M=3/6/12/24/48，见 `code/kernel-opt/46-w4a16-smallm/sweep.out.txt`。）

把它画出来：

```
  ms
0.10 ┤                                          wgmma m64 (1 pass) ─────────
     │                    ┌──────────────────── mma BM=16 (1 pass)
0.08 ┤            ┌───────┘         \
     │        ┌───┘                  \  mma BM=16 第 2 pass ↑ (M>16)
0.06 ┤    ┌───┘ GEMV(MT=2)            \
     │  ┌─┘                            \
0.04 ┤ ┌┘ GEMV(MT=1)                    \   GEMV: 每 MT 个 token 重读一遍权重
     │┌┘                                  \  (线性爬)
0.02 ┤
     └┬────┬────┬────┬────┬────┬────┬────┬──
      1    2    4    8   16   32   64   M
      └GEMV┘└──── mma BM=16 ───┘└── wgmma m64 ──┘
         ↑ M≈2.5               ↑ M≈17
```

**每个 M 的最优 kernel**（完整表见 `sweep.out.txt`）：

| M | best | ms | TFLOPS | 权重遍数 | 真实 HBM |
|---|---|---|---|---|---|
| 1 | `gemv_r2_mt1` | 0.0309 | 5.8 | 1 | 46.0% |
| 2 | `gemv_r4_mt2` | 0.0379 | 9.4 | 1 | 38.2% |
| 3–16 | `mma_b16_k8` | 0.070–0.075 | 7.6–38 | 1 | 19–22% |
| 24–64 | `wgmma_m64_k5` | 0.084–0.096 | 51–119 | 1 | 16–18% |

**交叉点：M≈2.5（GEMV → mma）和 M≈17（mma → wgmma）**，而不是 44 篇预测的 M≈78。

## 4. 根因：最优 kernel 由「权重读遍数 × 每遍成本」决定

把上表的括号列读一遍，规律很干净。记每个 kernel 的权重读遍数为
$P=\lceil M/d\rceil$（`d` = 一次权重读能覆盖的行数）：

| 路径 | d | 每遍成本 | 说明 |
|---|---|---|---|
| GEMV `r2_mt1` | 1 | ~0.030 ms | 一遍只覆盖 1 个 token |
| GEMV `r4_mt2` | 2 | ~0.038 ms | 2 个 token，计算略涨 |
| GEMV `r2_mt4` | 4 | ~0.079 ms | 4 个 token，计算再涨 |
| GEMV `r2_mt8` | 8 | ~0.150 ms | 8 个 token，ALU 已经很贵 |
| `mma_b16_k8` | 16 | ~0.070 ms | **一遍覆盖 16 行，成本还只有 0.07** |
| `mma_b32_k8` | 32 | ~0.078 ms | 一遍覆盖 32 行 |
| `mma_b64` | 64 | ~0.113 ms | 一遍覆盖 64 行，但成本高 |
| `wgmma_m64` | 64 | ~0.083 ms | 一遍覆盖 64 行，SS + WS 效率最高 |

于是：

- $M\le 2$：`gemv_r2_mt1` / `r4_mt2` 的 $P=1$、每遍 0.031–0.038 ms，**比任何张量核的
  0.07–0.083 ms 便宜**，GEMV 赢。
- $3\le M\le 16$：GEMV 一旦要 $MT\ge 3$ 就得多算 token，每遍涨到 0.073–0.08；
  而 `mma_b16_k8` 的 $P=1$、每遍仅 0.070 ms——**BM=16 的「少白算行」在这里兑现**，赢。
- $17\le M\le 64$：`mma_b16` 要读 2 遍（$P=\lceil M/16\rceil\ge 2$），而 `wgmma_m64`
  仍然只读 1 遍，**wgmma 赢**。
- $M>64$：`wgmma_m64` 也开始多遍，但那时已经是 prefill 的地盘，不在本篇范围。

> 关键点：**weight-bandwidth 场景下，「一遍覆盖多少行」比「每行算得多快」重要得多。**
> GEMV 的优势是每遍最便宜（0.031 ms），劣势是 d=1；wgmma 的优势是 d=64，劣势是每遍
> 0.083 ms；`mma_b16` 正好卡在中间（d=16、每遍 0.070 ms），于是赢下整个「小 batch」区间。

## 5. 优化过程：小 tile 一开始是惨败的

新写的 `mma.sync` 小 tile 版本**第一版全线垫底**，比 wgmma 慢近 2 倍。看 ncu 就明白了。

`mma_b32` 在 M=16（**加 K-split 之前**）：

| 指标 | 值 |
|---|---|
| Duration | 145.4 µs |
| **Waves Per SM** | **0.26** ← 只有 1/4 个波 |
| Achieved Occupancy | 13.0% |
| DRAM Throughput | 10.3% |
| Compute (SM) | 26.7% |
| Registers / thread | 63 |

M=16、BM=32 时 `grid.y=1`，网格只有 $N/BN=136$ 个 CTA，而每个 SM 能放 4 个——
**GPU 大面积空转**。而 wgmma 版带了 `KSPLIT=5`，网格 $136\times5=680$ 个 CTA，真正填满了机器。

修法：给小 tile GEMM 加 **K-split**（`blockIdx.z`，每段 K 各算一部分，`atomicAdd` 归约）。
`mma_b16_k8`（BM=16、KSPLIT=8）修完后的 ncu（M=16）：

| 指标 | wgmma 前基线 `mma_b32` | **`mma_b16_k8`** | M=1 GEMV `r2_mt1` |
|---|---|---|---|
| Duration | 145.4 µs | **66.3 µs** | 31.2 µs |
| Waves Per SM | 0.26 | **2.06** | 1.37 |
| Achieved Occupancy | 13.0% | 22.7% | 55.0% |
| DRAM Throughput | 10.3% | 22.6% | **47.4%** |
| Compute (SM) | 26.7% | 47.7% | 58.5% |
| Registers / thread | 63 | 64 | 38 |

网格从 136 → 1088 个 CTA，波数 0.26 → 2.06，时间腰斩到 66.3 µs。**「小 tile 输」不是
tile 的理念错，而是初版没并行度**。这也再次印证 41 篇的教训：先算 wave 数，再谈别的。

## 6. 自适应分派

由上面的规律，运行时分派器只需要三条分支：

```cpp
// 按 M 选 kernel（N=17408, K=5120, group=128 的 W4A16 decode）
if      (M <= 2)  gemv_r2_mt1(...);   // P=1, 每遍 0.031ms
else if (M <= 16) mma_b16_k8(...);    // P=1, 每遍 0.070ms，d=16
else              wgmma_m64_k5(...);  // P=1, d=64
```

用同一套 12 个 M 点求和（各点取最优），得到分派器相对「只用一种 kernel」的收益：

| 策略 | 总时间（12 点） | relative（以 always-wgmma 为 1.000） |
|---|---|---|
| always GEMV | 3.9746 ms | 3.884 |
| always mma | 0.9952 ms | 0.973 |
| always wgmma | 1.0233 ms | 1.000 |
| **adaptive** | **0.8576 ms** | **0.838** |

- 相对 always-wgmma **1.19×**，相对 always-GEMV **4.63×**；
- **always-mma 竟然略胜 always-wgmma（0.973）**——因为 12 个点里小 M 占了一半，
  `mma_b16_k8` 在 $M\le16$ 的领先足以抵消 $M\ge24$ 的落后。

单看真正的 decode 工作点 M=1：GEMV 0.0309 ms vs wgmma 0.0829 ms = **2.68×**；
M=2：0.0379 vs 0.0830 = **2.19×**。

## 7. 小结

- **实测证伪 44 篇的 $M^\*\approx78$**：交叉点在 **M≈2.5**（GEMV→mma）和 **M≈17**
  （mma→wgmma）。AI=3.77M 的 roofline 只在「两条路都贴 roofline」时成立；小 M 的张量核
  只有 16–18% HBM，是延迟/occupancy 受限，所以它远早于 M=78 就赢了。
- **最优 kernel 由权重读遍数决定**：weight-bandwidth 场景里「一遍覆盖多少行」（GEMV 1、
  mma.b16 16、wgmma 64）比「每行算得多快」更决定胜负。
- **小 BM 有效，但必须先给足并行度**：`mma.sync` 初版 waves 0.26、10% HBM；加 K-split
  后 waves 2.06、22.6% HBM、时间腰斩，成为 M∈[3,16] 的最优。
- **运行时分派器**在 M∈[1,64] 上比 always-wgmma 快 1.19×、比 always-GEMV 快 4.63×；
  M=1 时 GEMV 相对 wgmma **2.68×**。
- 仍未解决：`mma_b16_k8` 在 M=16 只有 22.6% HBM、Compute 47.7%，L1 40.3%——反量化 +
  `ldmatrix` 的寄存器往返是下一道墙；wgmma 版每遍 0.083 ms 也远高于纯读上界 0.018 ms。

## 8. 下一篇预告

M∈[2,16] 已经能稳定落在 `mma_b16` 上，但它的 HBM 利用率只有 22%。下一步可以：

1. 给 `mma_b16` 叠 **warp specialization**（生产者专职反量化、消费者专职 mma），
   看能否把 22.6% 推向 40%+；
2. 或者把这条分派逻辑接回 **W4A8+dp4a**（45 篇）：M≤2 用 dp4a GEMV、中间用
   `mma.sync.s8`（Hopper 支持 m16n8k32 IMMA），把「反量化」彻底从浮点搬到整数。

代码：`code/kernel-opt/46-w4a16-smallm/`（`w4a16_smallm.cu`、
`sweep.out.txt`、`ncu_gemv_r2mt1_M1.out.txt`、`ncu_mma_b16k8_M16.out.txt`、
`ncu_mma_b32_M16.out.txt`）。
