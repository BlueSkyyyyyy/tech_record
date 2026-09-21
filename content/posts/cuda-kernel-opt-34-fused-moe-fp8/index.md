---
title: "CUDA 算子调优（三十四）：MoE 专家 FFN 的 FP8 化 —— 权重流量减半换来的 1.9~2.2×，以及 unpermute 融合在 FP8 下为什么失效"
date: 2026-09-21
draft: false
weight: 34
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "fused MoE", "expert FFN", "FP8", "per-block scaling", "SwiGLU", "wgmma", "TMA", "DeepSeek", "DeepSeek-V4", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 30 篇]({{< relref "cuda-kernel-opt-30-fused-moe" >}})把 MoE 专家 FFN 的五段流水融成三个 kernel，
却只拿到 1.02~1.08×，因为 ncu 告诉我们：**prefill 的 FFN 是权重带宽受限**——50.7 GB 的专家权重
对流 1.8 GB 的 activation，融合能省的 activation 只占约 13%。结论很清楚：**要再快，唯一的杠杆是把权重字节砍下来。**

[第 31 篇]({{< relref "cuda-kernel-opt-31-moe-grouped-fp8pb" >}})已经证明，DeepSeek-V4 的
`e4m3 + ue8m0 + weight_block 128×128` 能把 grouped GEMM 的权重字节直接减半，代价只是寄存器
→ occupancy 的约 18%。这一篇就把拼图合上：**给整个 MoE 专家 FFN 换上 FP8 的「减肥餐」**，
端到端实测 **1.9~2.2×**（相对 30 篇的 bf16 版本），并回答两个很实际的问题：

1. 30 篇那个「unpermute 融进 down 的 epilogue」的技巧，在 FP8 下还成立吗？（**不成立**，见「负结果一」）
2. 「一个 CTA 吃下 gate+up、直接 SwiGLU」的融合，能叠上 per-block FP8 吗？（**寄存器不允许**，见「负结果二」）

> 编号说明：原计划的第 33 篇（MLA 极限冲刺三 · producer warp 专门化）因 Hopper 上的
> smem/寄存器硬阻塞暂缓（见路线图「阻塞」），本篇先发；第 33 篇留待 1-warpgroup/248-reg 布局
> 或 Blackwell `tcgen05` 再攻。

shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`hidden=7168`、`moe_intermediate_size=3072`、
`n_routed_experts=384`、`num_experts_per_tok=6`、`e4m3 + ue8m0 + weight_block 128×128`。
实测 `M=8192/16384`（balanced 与 random 路由）。

## 一、把「融合」和「per-block FP8」接起来

### 1.1 数学

DeepSeek-V4 的 MoE FFN 是标准的 SwiGLU：

$$
\text{FFN}(x) = W_d \cdot \big(\mathrm{SiLU}(W_g x) \odot W_u x\big)
$$

第 31 篇的 per-block FP8 定义（`weight_block=128×128`、激活 `1×128`）为：

$$
C[m,n] = \sum_{k_b} \underbrace{s_a[m,k_b]}_{\text{激活 per-行 per-128}} \cdot
         \underbrace{s_b[n/128,k_b]}_{\text{权重 per-128×128}} \cdot
         \Big(\sum_{k\in k_b} A[m,k] B[n,k]\Big)
$$

也就是说：**每累加完 128 个 k，就要把 fp32 累加器折算一次**。这正是 24/31 篇反复出现的
`fin` 寄存器代价来源。

### 1.2 两条流水

两条路径共用同一份 grouped per-block GEMM（TMA + `mbarrier` + warp specialization + `wgmma.m64n128k32` + SW128），
差异**只有**被省掉的那几次「物化 + 回读」。A 的布局沿用第 31 篇的「已按 expert 分组、对齐到 BM=128」
（即已经 permute 过的连续行），这样 A 也能走 TMA；30 篇单独验证过「gather 省 permute」只值 1.11×，
与本文的权重主矛盾正交。

```
不融合（本篇 baseline，4 kernel + cast）：
  grouped up/gate  A1[P,H] fp8 ──► GU[P,2I] fp32        (N=2I=6144)
  swiglu+quant     GU ──────────► A2[P,I] fp8 + sa2[P,I/128]
  grouped down     A2 ──────────► D[P,H] fp32
  unpermute(w,D)   ─────────────► Yf[M,H] fp32 ──cast──► Y bf16

融合（3 kernel + cast）：
  grouped up/gate  A1 ──────────► GU
  swiglu+quant     GU ──────────► A2 fp8 + sa2
  down + unpermute A2 ──────────► Yf[M,H] fp32          (red.global.add.f32 直接归约回 token)
```

注意「融合」只动了最后一步：把 down 的输出 `D`（fp32，`Pp·H·4 = 1.41 GB`）不落盘，epilogue 里用
fire-and-forget 的 `red.global.add.f32` 直接按 `row_tok` 加权加到 `Yf` 上。**K1 这一侧不能融 SwiGLU**——
原因见负结果二。

## 二、先修一个「看起来不起眼」的 kernel：swiglu + 动态量化

SwiGLU 把 `GU[P,2I]`（`2·Pp·I·4 = 1.21 GB`）读出来，算 `silu(g)*u`，同时做 **per-128 列动态量化**
写 `A2` fp8（`Pp·I = 0.15 GB`）。第一版按「一个 block 处理 `(p, iblk)` 的 128 列、128 线程各管 1 列」
写，结果只有 **1.07 ms**、等效 39% HBM —— 它白白吃掉了端到端的 9%。

```
v1（128 线程/block，每线程 1 列标量读）：1.07 ms  1.36 GB → 1290 GB/s（39%）
v2（一个 block 一行，warp 分批，float4 读 + __shfl_xor 内 amax）：0.44 ms  1.36 GB → 3090 GB/s（92%）
```

关键就两条：**一行的 G 段和 U 段都用 `float4` 对齐读**；**每 128 列块的 `amax` 在一个 warp 内
用 `__shfl_xor_sync` 归约**（每 lane 负责 4 列 `ib*128+lane*4`，一次归约覆盖整块），
然后 lane0 写 scale、四字节 `fp8` 打包成 `uint32` 存。**2.4× 的收益来自向量化与去掉 block 级 `__syncthreads`
归约**，不是算法。

## 三、结果

### 3.1 端到端（M=8192 balanced，`Pp=49152`，m-tile/expert=1）

| 阶段 | 时间 (ms) | 实际流量 (GB) | 等效带宽 | HBM 占比 |
|---|---|---|---|---|
| K1 up/gate（`A1·W12`，N=6144） | 7.03 | 16.91(W)+0.35(A)+1.21(GU) = 18.47 | 2627 GB/s | **78.4%** |
| swiglu+quant v1 → v2 | 1.07 → **0.44** | 1.36 | 1290 → **3090 GB/s** | 39% → **92%** |
| K2 down（`A2·Wd`，N=7168） | 3.39 | 8.45(W)+0.15(A)+1.41(D) = 10.01 | 2953 GB/s | **88.1%** |
| unpermute | 0.54 | 1.64 | 3060 GB/s | 91.3% |
| cast | 0.15 | 0.35 | 2307 GB/s | 68.8% |

| 路径 | 本篇 FP8 (ms) | 30 篇 bf16 (ms) | 加速 |
|---|---|---|---|
| 不融合（4 kernel + cast） | **11.80** | 22.65 | **1.92×** |
| 融合（3 kernel + cast） | 12.25 | 22.18 | 1.81× |

### 3.2 更大规模与不规则路由

| 配置 | Pp / padding | 权重流量 (GB) | FP8 不融合 (ms) | bf16 不融合 (ms) | 加速 |
|---|---|---|---|---|---|
| M=8192 balanced | 49152 / 0% | 25.4 | **11.80** | 22.65 | **1.92×** |
| M=16384 balanced | 98304 / 0% | 50.7 | **20.58** | 45.44 | **2.21×** |
| M=8192 random | 72704 / 32.4% | 37.5 | **15.93** | 32.80 | **2.06×** |

M=16384 的加速比更高，因为此时每个 expert 有 2 个 m-tile、权重被重读 2 次（权重流量 50.7 GB），
**权重带宽的占比进一步压倒 activation**——而 FP8 砍的正是权重。

> 口径说明：bf16 的对比数字直接取第 30 篇同 shape 的 `[unfused 5-kernel]`/`[fused 3-kernel]`。
> 30 篇的 unfused 还含一个 permute kernel（M=8192 时 0.46 ms），本篇的 A 已是分组后的连续布局、
> 不含 permute；扣掉后 bf16 等效 22.18 ms，加速 1.88×，与上表同量级。

### 3.3 离 roofline 还有多远

| 配置 | 权重流量 | HBM 下限 @3.35 TB/s | 实测 K1+K2 | 达成率 |
|---|---|---|---|---|
| M=8192 bal | 25.4 GB | 7.57 ms | 10.80 ms | 70.1% |
| M=16384 bal | 50.7 GB | 15.14 ms | 18.76 ms | 80.7% |

K2 的 88.1% 已经很接近内存算子的北极星（90~95%）；K1 只有 78%，是下一步的主要空间。

## 四、两个负结果（都值得记下来）

### 负结果一：unpermute 融进 FP8 down 反而更慢

30 篇（bf16）里，`down + unpermute` 融合在 `bn128` 是**更快**的（8.67 vs 9.67 ms）；本篇照搬，
却变成 **4.28 vs 3.90 ms（K2+独立 unpermute）**，端到端 `fused` 反而比 `unfused` 慢（0.96×）。
ncu 给出了干净的指纹：

| kernel | DRAM Throughput | L2 Cache Throughput | 时间 |
|---|---|---|---|
| K2 down（写 `D`） | **89.3%** | 86.0% | 3.35 ms |
| K2 down+unperm（`red.add` 散写 `Yf`） | 73.2% | **94.4%** | 3.83 ms |

融合把写 `D` 的 1.41 GB 省了，但 `red.global.add.f32` 是 **6 路散写归约**（每个 token 最多被 6 个
expert 行命中），6 个地址在 L2 里打架，把瓶颈从 DRAM 顶成了 **L2 请求**（94.4%）。
在 bf16 时代，down GEMM 要 8.67 ms（权重是大头），散写的 L2 开销被藏得住；FP8 把 GEMM 压到
3.35 ms 后，**散写重新暴露在关键路径上**。

> 教训：**一个融合技巧的收益不是常数，它取决于被测算子有多快。** 同一个 unpermute 融合，
> 在 2× 权重字节的 FP8 版本上就从 +10% 变成 −10%。

### 负结果二：「双累加器 SwiGLU」叠不了 per-block FP8

30 篇的招牌技巧是「一个 CTA 吃下 gate 和 up、两个 fp32 累加器在寄存器里直接 SwiGLU」。
想把它搬到 FP8，寄存器账立刻爆炸：

```
gate/up 双累加器：2 × 64 = 128 fp32/线程
per-block 的 fin ：2 × 64 = 128 fp32/线程
────────────────────────────────────────
合计                  256 > 255（每线程寄存器硬上限）
```

30 篇的 bf16 版只有 128（无 `fin`），刚好；31 篇的 per-block 单累加器是 64(`acc`)+64(`fin`)=128，
也刚好。**两者叠加就超了 55 个寄存器**，任何调度都救不回来（第 24 篇 BM=256 的 `96 regs + 608 B spill`
就是同一堵墙）。所以本篇的 K1 只能老老实实写 `GU`、让 `swiglu+quant` 单独跑；用 WG 分工（两个
warpgroup 分别算 gate/up 再走 smem 交换）也许能绕过，留作后续。

### 扫参附注：`BN=256` 对 per-block 直接崩

在 K1/K2 上把 `BN` 从 128 拉到 256，`NSPLIT=2` 让每线程的 `acc`+`fin` 翻倍到 256——和负结果二同理，
实测 K1 从 7.1 ms 崩到 **21.7 ms**（ptxas 溢出）。**per-block 缩放的工作点只有 `BN=128` 一条路。**

## 五、ncu 定位下一步

K1（最佳配置 128×128 s3）的 SpeedOfLight：

```
DRAM Throughput      80.1%      ← 已经贴近但没打满
L2 Cache Throughput  80.4%
Compute (SM)         31.5%
Achieved Occupancy   13.9%      (156 regs / 132 KB smem -> 1 CTA/SM)
No Eligible          77.1%，其中 long_scoreboard 占 warp stall 的 51.3%
```

K1 是「权重流 + 每 128-k 折算出 `wgmma.wait0` 排空」的混合体：DRAM 80%、但 warp 有 51% 的时间
卡在 L1TEX 依赖（等 TMA 的下一块 B / 等 scale）。**和 31 篇一样，瓶颈是 1 CTA/SM 加上每块折算
打断 wgmma 流水**，不是算力。要再往上，路线是 DeepGEMM 那套「1 warpgroup + `setmaxnreg 248` +
`warpgroup_fence_operand`」把折算与下一块 wgmma 真正重叠——这是 [24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})
就点名的方向。

## 小结

- **FP8 是 MoE prefill FFN 目前最大的单点杠杆**：权重字节 50.7 → 25.4 GB，端到端 **1.92×**（M=8192）
  到 **2.21×**（M=16384）。
- K2 down 已达 **88.1% HBM**（ncu DRAM 89.3%）（ncu DRAM 89.3%），离内存算子的 90~95% 目标只差一点；
  K1 78% 是主要空间，卡在 1 CTA/SM + per-block `wait0`。
- SwiGLU+量化的正确写法是「一行一 block、`float4` 读、warp 内 `amax`」，从 39% HBM 提到 92%（2.4×）。
- **unpermute 融合在 FP8 下失效**：散写把瓶颈从 DRAM（89%）顶到 L2（94%），因为 GEMM 变快后
  散写不再被藏住。
- **per-block FP8 无力和双累加器 SwiGLU 共存**：256 > 255 寄存器，`BN=256` 同样崩。

下一篇继续把 K1 推向 DeepGEMM 的 1-warpgroup / 248-register 布局（也见 [24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}}) 的遗留），
或者转去做 DSA compressor / paged KV-cache。代码与原始输出见 `code/kernel-opt/34-fused-moe-fp8/`。
