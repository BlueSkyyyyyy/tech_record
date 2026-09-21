---
title: "CUDA 算子调优（二十八）：MoE 门控 GEMM 的 wgmma + TMA 冲刺 —— 从 cuBLAS 的 28% 到 80%"
date: 2026-09-21
draft: false
weight: 28
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "gate", "router", "GEMM", "wgmma", "TMA", "SW128", "warp specialization", "DeepSeek", "DeepSeek-V4", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-27-moe-router" >}})把 DeepSeek-V4 的 MoE 前门整条链（router + top-k + token 置换）手写了一遍，唯一的"短板"是开头那个 **gate GEMM**：形状 `[M,7168] × [384,7168]ᵀ → [M,384]`，我们用 `mma.sync + ldmatrix + cp.async` 只做到 **204→257 TFLOPS**（约 cuBLAS 738 的 **27.7%**）。这一篇把它补上——用 [第 20 篇]({{< relref "cuda-kernel-opt-20-mla-wgmma-sw128" >}})的 **wgmma + SW128** 和 [第 23 篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}})的 **TMA + mbarrier + warp specialization** 两把武器重做。

结论先行（H100 SXM，实测）：

| M | 27 篇 `mma.sync` | 28 篇 `wgmma+cp.async` | 28 篇 `wgmma+TMA+WS`（最佳） | cuBLAS bf16 | 达 cuBLAS | 相对 27 |
|---|---|---|---|---|---|---|
| 16384 | 0.447 ms / **201.9 TFLOPS** | 0.268 ms / 337.1 | **0.160 ms / 565.4（57.2% 峰值）** | 0.123 ms / 734.6 | **77.0%** | **2.80×** |
| 32768 | 0.688 ms / **262.3 TFLOPS** | 0.502 ms / 359.6 | **0.301 ms / 600.3（60.7% 峰值）** | 0.240 ms / 752.6 | **79.8%** | **2.29×** |

一句话：**同一个算子，把指令换成 `wgmma`、把装载换成 TMA、把装载和计算拆成 producer/consumer 两条流水线，就从 cuBLAS 的 28% 干到 80%。** 本文把每一步的收益拆开量、并用 ncu 指出剩余 20% 卡在哪。

环境：H100 SXM 80GB（132 SM，实测 HBM **3352.3 GB/s**，bf16 TC dense 989 TFLOPS），CUDA 13.2。
数字来自 `code/kernel-opt/28-moe-gate-wgmma/gate_M{16384,32768}.out.txt`、`ncu_*.out.txt`、`cublas_gate.out.txt`。

---

## 1. 这个 GEMM 为什么"不好做"

真实参数取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：

```jsonc
"hidden_size": 7168,            // K = hidden
"n_routed_experts": 384,        // N = 专家数（这一层输出维度）
"num_experts_per_tok": 6
```

$$
\text{logits}[M,E] = X[M,H]\cdot W_g[E,H]^{\top},
\qquad M\in\{16384,32768\},\ H=7168,\ E=384
$$

它有三个"反常规"的地方：

1. **N 很小（384）、K 很大（7168）**：这是典型的 **tall-skinny / 高瘦** GEMM，算术强度 `2MNK/(MK·2) = N` FLOP/byte ≈ 384，离 bf16 TC 的 ridge（989/3.35≈295 FLOP/byte@BF16）很近，**既不是纯算力题也不是纯带宽题**。
2. **并行度天然不足**：输出只有 `M/BM × N/BN = (16384/128)×(384/128) = 128×3 = 384` 个 tile。132 个 SM、每 SM 最多 2 个 CTA 时，一次只能挂 264 个 CTA → **1.45 wave**，尾效应严重。
3. **A 会被重读 `N/BN` 次**：每个 `(m_tile, n_tile)` CTA 都要把整条 `BM×K` 的 A 读一遍，`N/BN=3` 就是 3 遍。A 是 `16384×7168×2 = 235 MB`，3 遍 705 MB，几乎等于整个 kernel 的 DRAM 流量。

27 篇的 `mma.sync` 版本已经踩到发射端口的墙（ncu：`L2 60.7%`、`Compute 35.3%`、`No Eligible 57.6%`、occupancy 20%）。要做上去，必须换 Hopper 的原生武器。

---

## 2. 第一步：`wgmma` + SW128 —— 先解决"指令"问题

### 2.1 为什么 `wgmma` 在 Hopper 上更快

`mma.sync.m16n8k16` 是 Ampere 的指令，操作数（A/B 片段）必须由每条线程用 `ldmatrix` 从 smem 读进寄存器，再喂给 tensor core。到了 Hopper，`wgmma.mma_async` 是**整条 warpgroup（128 线程）协作**、**直接从 smem 描述符读操作数（SS 形式）**的异步指令——省掉了 `ldmatrix` 的发射槽与寄存器，指令流从"每条线程各算各"变成"一条指令搬一整块"。

这在 [第 22 篇 FP8 GEMM]({{< relref "cuda-kernel-opt-22-fp8-gemm" >}})里已经验证过：同 shape 换 `wgmma` 是 **+2.89×**。gate 这里同样适用。

### 2.2 一个天然的便利：B 不用转置

`wgmma` 的 B 操作数要求 **K-major**（沿 K 连续）。MoE 的权重天然是 `W_g[E, H]`——**H 连续**，正好就是 `B[N=E][K=H]` 的 K-major 布局。所以 gate GEMM 里 B **不需要任何转置**，直接 `wgmma` 描述符指过去即可（对比 attention 里的 V 必须转置，见 [21 篇]({{< relref "cuda-kernel-opt-21-mla-wgmma-pipe" >}})）。

### 2.3 bf16 的 SW128：原子是 8 行 × 64 元素

[20 篇]({{< relref "cuda-kernel-opt-20-mla-wgmma-sw128" >}})推导过 CUTLASS canonical 的 K-major SW128：8 行 × 128 字节为一个 atom，16B 列做异或 `c' = c ^ r`（`r` = atom 内行号）。对 **fp8** 每行 128 字节 = 128 个元素；对 **bf16** 每行 128 字节只有 **64 个元素**。于是：

- 沿 K 的分块必须 `BK = 64`（一行正好 128B）；
- 描述符的 `SBO`（相邻 8 行组的字节距离）= `(BK/64)×1024 = 1024`；
- `k16` 步进的地址增量 = `(s>>2)*1024 + (s&3)*32`，`BK=64` 时 `s∈{0,1,2,3}` → `0, 32, 64, 96`。

和 20 篇完全同构，`make_desc_sw128/smem_u32` 直接复用。

### 2.4 实测：cgm + `cp.async`（先不等 TMA）

先把 A/B 用 `cp.async` 手动写进 SW128 布局 + N 级流水，得到对照版 `gate_async_kernel`：

| 配置 | M=16384 | M=32768 |
|---|---|---|
| `async128x128s3` | 337.1 TFLOPS | 359.6 |
| `async128x128s4` | 271.8 | 268.5 |
| `async256x128s3` | 268.6 | 366.1 |
| 27 篇 `mma.sync` BM=128 | 201.9 | 262.3 |

**换 `wgmma` 就先拿到 +67%~+37%**（16k：201.9 → 337.1）。但注意 `s4` 反而比 `s3` 慢（271 vs 337）：`s3` 的 smem 是 `3×(128+128)×128 = 98KB`，每 SM 能放 2 个 CTA；`s4` 是 131KB，只能放 1 个。**在延迟受限的 kernel 里，occupancy 比流水深度更值钱**——这个规律在下一步会反复出现。

---

## 3. 第二步：TMA + mbarrier + warp specialization

`cp.async` 版本的瓶颈是：**256 个线程每轮都要自己算 SW128 地址、发 `LDGSTS`**，既占发射槽又占寄存器，装载和计算互相抢 issue。23 篇的解法是把这件事整个交给硬件：

```
        ┌───────────────────────── CTA ──────────────────────────┐
        │  producer warp (1)          consumers: (BM/64) warpgroup │
        │  ┌──────────────┐           ┌───────────────────────┐    │
        │  │ TMA 2D SW128 │           │ wgmma.m64n128k16 (SS) │    │
        │  │ box [128B,BM]│  full[st] │  acc += A·B           │    │
        │  │ box [128B,BN]│ ────────► │                       │    │
        │  └──────┬───────┘           └──────────┬────────────┘    │
        │         ▲ empty[st] ◄───────────────────┘                 │
        │  mbarrier full/empty 相位 = (kb/STAGES)&1                  │
        └──────────────────────────────────────────────────────────┘
```

- **TMA**：`cp.async.bulk.tensor.2d` + `CU_TENSOR_MAP_SWIZZLE_128B`，一条指令搬整块 `[BM,64]`，硬件写出的 smem 布局**恰好等于 wgmma 的 K-major SW128**（23 篇已验证 fp8；bf16 用 `UINT8` 字节语义 + box 内维 128B，逐字节同构）。kernel 里不再有任何 swizzle 地址计算。
- **mbarrier 握手**：`full[st]` 由 producer `arrive.expect_tx(bytes)` 声明、TMA 完成后自动补齐；consumers 读完一个 stage 后每个线程都 `arrive(empty[st])` 回压。
- **相位用常量算，不用数组**：`phase = (q/STAGES)&1`。23 篇在这里踩过坑——`uint32_t phase[STAGES]` 动态下标会掉进 local memory。

### 3.1 实测：一跳从 337 → 565

| 配置 | M=16384 | M=32768 | 说明 |
|---|---|---|---|
| `ws128x128s2` | 505.5 | 559.9 | 66KB → 3 CTA/SM |
| **`ws128x128s3`** | **565.4** | 549.3 | 98KB → **2 CTA/SM**，16k 最佳 |
| `ws128x128s4` | 478.8 | 469.3 | 131KB → 1 CTA/SM |
| `ws128x128s5` | 468.0 | 452.8 | 1 CTA/SM |
| `ws256x128s3` | 456.8 | 561.9 | BM=256，线程 544 |
| `ws256x128s4` | 493.4 | **600.3** | 197KB → 1 CTA/SM，32k 最佳 |
| `ws64x128s4` | 450.4 | 420.4 | BM=64，只有 1 个 warpgroup |

TMA + WS 相对 `cp.async` 版本**再快 68%（337 → 565@16k）**，寄存器从 122 降到 **90**（0 spill）。而且最佳配置随 M 变化，正是前面"wave 数"的直接后果：

- **M=16384**：grid `(3,128)=384` CTA，2 CTA/SM 时 1.45 wave；此时**能塞进 2 个 CTA 的 `s3` 最好**。
- **M=32768**：grid `(3,256)=768` CTA；`ws256x128s4` 虽然只有 1 CTA/SM，但 **BM=256 让 A 的复用翻倍**（每个 A tile 服务 256 行输出），L2 压力下降，反超所有 128 行配置。

> 这跟 23 篇 FP8 GEMM 的结论一致：**算力受限的大 GEMM，"降 L2 流量（大 BM）"可能胜过"堆 occupancy"**。

---

## 4. ncu：钱花在哪、墙在哪

### 4.1 M=16384，`ws128x128s3`（565 TFLOPS）

| 指标 | 值 |
|---|---|
| Compute (SM) | 62.6%（由 Tensor 子管道主导） |
| Memory Throughput | 66.1%（其中 **L2 66.1%**、L1 55.6%、**DRAM 48.1%**） |
| Achieved / Theoretical occupancy | 22.4% / 28.1% |
| Registers / Dynamic smem | 90 / 99.3 KB（2 CTA/SM） |
| Waves per SM | **1.45** |
| No Eligible | 75.8% |
| 主 stall | `long_scoreboard` 占 49.2% issue 周期 |
| Bank conflict（ld/st） | 0 / 0 |

### 4.2 M=32768，`ws256x128s4`（600 TFLOPS）

| 指标 | 值 |
|---|---|
| Compute (SM) | **69.6%** |
| Tensor pipe（`sm__pipe_tensor_op_hmma` active） | **68.2%** |
| Memory Throughput | 62.4%（**L2 79.1%**、DRAM 53.4%） |
| Achieved occupancy | 26.1%（1 CTA/SM，197.6 KB smem） |
| Waves per SM | 2.91 |
| Bank conflict | 0 / 0 |

读法：

- **tensor 管道 68%、Compute 70%** → 主流已经压到 tensor 上了，方向没错；
- **L2 79% 比 DRAM 53% 高** → 正是第 1 节说的 **A 被重读 3 遍**：DRAM 只需读一遍 A（235MB）+ B，但 L2 要吃 `3×A`；
- occupancy 只有 26%、`long_scoreboard` 仍是头号 stall → **还有 30% 的 SM 时间去等数据**，这正是余下那 20% 对标 gap 的去处。

---

## 5. 两个负结果（省得你踩）

**① split-K（想把 1.45 wave 的尾巴摊平）——失败。**
把 K=7168 拆成 2/4 段，grid 从 384 涨到 768/1536，理论上能摊平尾效应；实测反而更慢：

| 配置 | M=16384 | M=32768 |
|---|---|---|
| `ws128x128s3`（不拆） | 565.4 | — |
| `ws128x128s3k2` | 429.8 | 428.2 |
| `ws128x128s3k4` | 347.8 | 370.7 |
| `ws128x128s2k2` | 423.0 | 424.3 |

原因：split-K 让**同一时刻有更多 CTA 抢 A 的 L2 行**（L2 本就 79%），再叠加 `atomicAdd` 归约与一次 `cudaMemsetAsync` 清零，得不偿失。**尾效应不是这个 kernel 的主要矛盾，L2 才是。**

**② `L2_PROMOTION_L2_256B`（想让 TMA 一次拉 256B 减半 L2 请求）——失败。**
32k `ws256x128s4`：600.3 → 576.3 TFLOPS。A 的行距是 14336B，256B promotion 预取的相邻数据大多用不上，反而多占 L2 带宽。

另外 `BM=64`（只有 1 个 warpgroup，`ws64x128*`）全线在 420–450，因为每个 CTA 的 A 复用只有 128 行配置的一半；`BN=384`（整块 N）虽然省 A，但 **TMA box 单维上限 256**，做不了。

---

## 6. 小结

- **指令**：`mma.sync` → `wgmma.m64n128k16`（SS，smem 描述符），gate 201.9 → 337.1 TFLOPS（**+67%**，16k）。
- **装载 + 流水**：`cp.async` → **TMA + mbarrier + warp specialization**（1 producer warp + 2/4 consumer warpgroup），337.1 → **565.4**（16k）/ 600.3（32k），寄存器 122 → **90**、0 spill。
- **相对 27 篇**：16k **2.80×**、32k **2.29×**；**达同 shape cuBLAS bf16 的 77.0% / 79.8%**（1.30× / 1.25×）。
- **config 选择**：小 M 时"能放 2 CTA/SM 的 `s3`"赢；大 M 时"BM=256 换 A 复用、压 L2"赢。
- **剩余 gap**：ncu 指向 tensor 68%、**L2 79%**、occupancy 26%。A 被 3 个 n-tile 重读是下一个明确目标。

### 下一篇预告

gate 的 A 重读 `N/BN=3` 次，而 [26 篇]({{< relref "cuda-kernel-opt-26-moe-cluster-multicast" >}})已经有现成的 **TMA cluster multicast**：沿 N 组的 3 个 CTA（同一 m-tile）用 `cluster.x=3`，让 leader 把 A 广播给整组，A 的 L2 流量直接降到 1/3。下一篇把它叠到 gate 上，目标把 600 推向 700+（逼近 cuBLAS）。
