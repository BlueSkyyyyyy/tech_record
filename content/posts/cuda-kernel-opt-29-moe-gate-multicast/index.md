---
title: "CUDA 算子调优（二十九）：MoE 门控 GEMM 的 TMA cluster multicast —— 广播 A 还是广播 B？"
date: 2026-09-21
draft: false
weight: 29
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "gate", "GEMM", "wgmma", "TMA", "cluster multicast", "warp specialization", "DeepSeek", "DeepSeek-V4", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 28 篇]({{< relref "cuda-kernel-opt-28-moe-gate-wgmma" >}})把 DeepSeek-V4-Pro 的 MoE 门控 GEMM
（`logits[M,384] = X[M,7168]·Wg[384,7168]ᵀ`）用 `wgmma + TMA + warp specialization` 从 cuBLAS 的
28% 干到了 80%，当时 ncu 指出剩下的墙是 **L2 79%**，并把怀疑对象写成了「**A 被 3 个 n-tile 重读**」。
这一篇就把 [第 26 篇]({{< relref "cuda-kernel-opt-26-moe-cluster-multicast" >}})的 **TMA cluster
multicast** 搬到 gate 上，实测到底是哪一块 operand 在被重读、广播它能不能把那 79% 的 L2 压下去。

结论先行（H100 SXM，实测，32k tokens）：

| 配置 | ms | TFLOPS | 峰值 | L2 读 sector | 相对基线 |
|---|---|---|---|---|---|
| baseline `256×128 s4`（BN=128，无 cluster） | **0.2947** | **612.2** | 61.9% | 46.2 M | —（本系列 gate 最好） |
| A-multicast `cluster.x=3` | 0.3848 | 468.7 | 47.4% | **34.0 M（−26%）** | **−23.4%** |
| B-multicast `cluster.y=2`（128×128 s3） | 0.3072 | 587.2 | 59.4% | 64.1 M | 该 config 上 **+5.6%** |
| baseline `128×128 s3`（对照） | 0.3243 | 556.3 | 56.3% | 71.2 M | — |

一句话：**广播确实能把 L2 读 sector 砍掉 1/4，但 gate 的瓶颈不是 L2 字节数——cluster 把 CTA 耦合起来后
tensor/SM 利用率从 69% 掉到 51%，净效果是变慢。** 唯一变快的是「沿 M 方向广播权重 B、cluster 只有 2」，
但它仍打不过「直接把 BM 提到 256」——因为**放大 BM 和广播 B 是同一件事（都在减少 B 的 m-tile 重读），
而前者没有跨 CTA 耦合**。这是一篇负结果为主、但把「什么情况下该上 multicast」讲清楚的实测记录。

环境：H100 SXM 80GB（132 SM，实测 HBM 3352.3 GB/s，bf16 TC dense 989 TFLOPS），CUDA 13.2。
数字来自 `code/kernel-opt/29-moe-gate-multicast/gate_M{16384,32768}.out.txt`、`ncu_*.out.txt`。
cuBLAS 对照沿用 28 篇：`734.55 TFLOPS @16k / 752.58 @32k`。

---

## 1. 先算清楚：谁在被重读

gate 的形状是 M×K 乘 K×N，N=384 很小、K=7168 很大：

$$
\text{logits}[M,E] = X[M,H]\cdot W_g[E,H]^{\top},\qquad
M\in\{16384,32768\},\ H=7168,\ E=384
$$

网格是 `grid = (E/BN, M/BM)`，`blockIdx.x` 是 n-tile（3 个），`blockIdx.y` 是 m-tile：

```
             n-tile →   x=0        x=1        x=2      （每个 128 列，共 E=384）
 m-tile  y=0   A(y=0) × B(x=0)  A(y=0) × B(x=1)  A(y=0) × B(x=2)
               └────────── 同一块 A，被 3 个 n-tile 各读一次 ──────────┘
 m-tile  y=1   A(y=1) × B(x=0)  ...
   │
   └── 同一块 B(x)，被 M/BM 个 m-tile 各读一次
```

- **A（激活）被重读 `N/BN = 3` 次**（同一 m-tile 的 3 个 n-tile）。
- **B（权重）被重读 `M/BM = 256`（BM=128）或 128（BM=256）次**（同一 n-tile 的所有 m-tile）。

B 只有 384×7168×2 = 5.5 MB，能常驻 50 MB 的 L2，所以它的每次「重读」都是 L2 命中；但次数多，
L2 读**字节**照样大。A 是 470 MB（32k）的流式数据，被重读 3 次。所以 28 篇猜「A 是墙」并不显然——
必须实测。

26 篇已经把 multicast 的协议写通了，这里直接把 `cluster` 做成**两种轴向可选**：

```cuda
// AXIS=1: cluster.x=CN，沿 N 组队，同 m-tile 的 CN 个 n-tile 共享 A → 广播 A
// AXIS=2: cluster.y=CN，沿 M 组队，同 n-tile 的 CN 个 m-tile 共享 B → 广播 B
template <int BM,int BN,int BK,int STAGES,int CN,int AXIS>
__global__ void gate_mcast_kernel(tmA, tmB, C, M, N, K);
```

`grid.x = E/BN`，`grid.y = ceil(M/BM)`。于是 `AXIS=1` 时 `CN` 必须整除 3（只能取 3），
`AXIS=2` 时 `CN` 必须整除 `M/BM`（2/4/8 都行）。

---

## 2. multicast 协议（与 26 篇完全一致）

cluster 内每个 CTA 都还是「1 producer warp + N consumer warpgroup」，区别只在共享 operand 由
**rank0 发一条 multicast TMA**，其余 CTA 不再自己搬：

```
   rank0 (leader)            rank1                    rank2
   ┌──────────┐          ┌──────────┐             ┌──────────┐
   │ TMA A ───┼──multicast::cluster──┬────────────►│ A        │
   │ TMA B    │          │ TMA B    │             │ TMA B    │
   └────┬─────┘          └────┬─────┘             └────┬─────┘
        │ full[s] = 1(本 CTA arrive.expect_tx(A+B))     │
        ▼                                              ▼
   consumer wgmma 消费；每 consumer 线程 arrive empty[s]（私有 B 的释放）
                        每 warp lane0 用 mapa 把 arrive 投到 rank0 的 sempty[s]（共享 A 的释放）
```

要点（都是 26 篇踩出来的）：

1. **`arrive.expect_tx` 每个 CTA 都做**，字节数写 `A+B`；multicast 会把 A 的完成信号投递到每个目标 CTA 的
   同一个 full barrier。
2. **共享 operand 的释放要用单独的 barrier**（`sempty`，count = `CN × 消费者 warp 数`），
   由每个 consumer warp 的 lane0 用 `mapa.shared::cluster` 投到 rank0；rank0 覆盖 A 前必须等到
   全 cluster 都读完。私有 operand 的 `empty` 仍是本地 arrive。**两者混用会死锁且不报错。**
3. `mbarrier.init` 之后、退出之前各来一次 `cluster_sync()`（远端 barrier 反构前所有 arrive 必须落地）。
4. 相位一律用常量 `(q/STAGES)&1`（23 篇的坑：动态下标相位数组会掉 local memory）。

---

## 3. 实测：广播 A 把 L2 砍了 26%，却慢了 23%

先看 32k 的完整扫描（`gate_M32768.out.txt`，只节选 128×128 s3 / 256×128 s4）：

| 配置 | ms | TFLOPS | 占峰值 |
|---|---|---|---|
| `ws64x128s3_c1` | 0.4378 | 412.1 | 41.7% |
| `ws128x128s2_c1` | 0.3191 | 565.3 | 57.2% |
| `ws128x128s3_c1` | 0.3243 | 556.3 | 56.3% |
| `ws256x128s3_c1` | 0.3119 | 578.4 | 58.5% |
| **`ws256x128s4_c1`** | **0.2947** | **612.2** | **61.9%** |
| `ws64x128s3_ax3` | 0.4192 | 430.4 | 43.5% |
| `ws128x128s2_ax3` | 0.3800 | 474.7 | 48.0% |
| `ws128x128s3_ax3` | 0.3601 | 500.9 | 50.7% |
| `ws256x128s4_ax3` | 0.3848 | 468.7 | 47.4% |
| `ws128x128s2_by2` | 0.3216 | 560.9 | 56.7% |
| `ws128x128s3_by2` | 0.3072 | **587.2** | 59.4% |
| `ws128x128s3_by4` | 0.3585 | 503.1 | 50.9% |
| `ws128x128s3_by8` | 0.3711 | 486.0 | 49.1% |
| `ws256x128s3_by2` | 0.3335 | 540.9 | 54.7% |
| `ws256x128s4_by2` | 0.3209 | 562.2 | 56.8% |
| `ws256x128s4_by4` | 0.4091 | 441.0 | 44.6% |

ncu 关键指标（32k，各抓一次单 kernel）：

| 配置 | duration | L2 读 sector | L2 吞吐 | SM(tensor) 吞吐 | DRAM |
|---|---|---|---|---|---|
| `256×128 s4 c1` | 290.3 µs | 46.19 M | 80.4% | **68.8%** | 53.8% |
| `256×128 s4 ax3` | 370.9 µs | **33.99 M (−26%)** | **48.5%** | **51.5%** | 43.5% |
| `128×128 s3 c1` | 316.3 µs | 71.20 M | 82.3% | 64.4% | 49.9% |
| `128×128 s3 by2` | 301.9 µs | 64.11 M (−10%) | 85.0% | 67.7% | 51.8% |

这组数字把因果讲得很直白：

**广播 A 确实命中目标**——L2 读 sector 从 46.2 M 掉到 34.0 M，`L2 Cache Throughput` 从 80% 砸到 48.5%。
**但 kernel 反而慢 28%**，因为 `SM (tensor)` 从 68.8% 掉到 51.5%。也就是说 **L2 在 80% 时根本不是
binding constraint**：tensor pipe 才是。cluster.x=3 让 3 个 CTA 必须同 GPC 共驻、且 rank0 的 producer 在
覆盖 A 前要等满 3 个 CTA 的全部 consumer 读完（`sempty` 的 count = 3×消费者 warp 数），跨 CTA 的握手把
访存耦合得更紧，producer 发不出去 → 消费者等操作数 → tensor 空转。

这和 26 篇 grouped GEMM 的结论是同一个：**multicast 砍的是 L2 字节，代价是跨 CTA 耦合；只有当瓶颈
真的在字节（而不是延迟/发射）时才划算。** 26 篇里 grouped 大 batch 赢了 +4.3%，这里 gate 直接输 23%。

---

## 4. 广播 B 和「直接放大 BM」是同一件事

再看沿 M 方向的 B 广播。B 被每个 m-tile 重读一遍，所以直觉上它比 A 更值得广播。在 `128×128 s3` 上
`by2` 实测 **587.2 vs 556.3 TFLOPS（+5.6%）**，L2 读 sector 71.2 M→64.1 M（−10%），tensor 利用率
64.4%→67.7%。为什么这个 cluster 不亏？因为它是 **CN=2**，只耦合两个 CTA，交叉握手的等待链短得多。

但它并没有赢过总冠军 `256×128 s4 c1`（612.2）。原因很关键：

> **B 的重读次数 = `M/BM`。把 BM 从 128 翻到 256，B 的重读次数直接砍半——这和 `cluster.y=2`
> 广播 B 做的事一模一样，却不需要任何 cluster、没有任何跨 CTA 耦合。**

所以 `256×128` 上再加 B 广播就几乎没收益甚至变慢：

| config | c1 | by2 |
|---|---|---|
| 128×128 s3 | 556.3 | **587.2（+5.6%）** |
| 256×128 s4 | **612.2** | 562.2（−8.2%） |
| 256×128 s3 | 578.4 | 540.9（−6.5%） |

**「BM 已经在消 B 重读」时再叠 B 广播是冗余的**；而 TMA box 单维上限 256（28 篇坑），BM 已经到顶，
所以 multicast 在这里救不了场。

那 A 广播能不能和 BM 叠加？`256×128 s4 ax3` 是 468.7，比 `256×128 s4 c1` 的 612.2 差更远——A 只被
重读 3 次，广播省下的字节本来就不多，却要付出 CN=3 的耦合。

---

## 5. 一次失败的「免 cluster」尝试：wide-N

既然 A 只需读一次，那让**一个 CTA 算完整个 N=384**（3 个 n128 累加器）就能免掉 A 的 3× 重读，
还不用 cluster。于是写了 `gate_wide_kernel<BM,STAGES>`：A 按 BM 装载一次，B 的 384 行用 3 次
TMA（box≤256）；每个 warpgroup 做 m64 × 3×128。

结果（32k）：

| 配置 | ms | TFLOPS |
|---|---|---|
| `wide64s3` | 0.4058 | 444.6 |
| `wide64s4` | 0.3953 | 456.3 |
| `wide128s2` | 1.2107 | 149.0 |
| `wide128s3` | 1.2401 | 145.5 |
| `wide256s2` | 4.4018 | 41.0 |

`wide128/256` 直接崩到 149/41——ptxas 明确报：

```
ptxas info : (C7511) Potential Performance Loss: wgmma.mma_async instructions are
  serialized due to insufficient register resources for the wgmma pipeline
```

**3 个 n128 累加器 = 3×64 = 192 个 fp32/线程**，加上描述符和地址，寄存器不够让 12 条 `wgmma` 在流水线里
同时在飞，ptxas 只能把它们串行化。这和 [第 24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})
per-block 的寄存器墙是同一个病：**wgmma 的并行度直接由「在飞累加器」的寄存器预算决定**。
`wide64`（只 1 个 warpgroup、每 WG 仍然 3 个累加器，但线程少、acc 分摊不同）没触发 C7511，却也只有
456 TFLOPS——因为一个 CTA 只 160 线程，occupancy 太低。

结论：**要让一个 CTA 吃掉整个 N，得等 tcgen05（Blackwell 把累加器放 TMEM）或拆成两个 kernel。**
在 Hopper 的寄存器-累加器模型下，这条路走不通。

---

## 6. 那 612 TFLOPS 卡在哪？

看最佳配置 `256×128 s4 c1` 的 ncu 指纹：

| 指标 | 值 |
|---|---|
| Duration | 290.3 µs |
| Compute (SM) | 69.5% |
| Tensor pipe 活跃 | ~69% |
| L2 Cache Throughput | 80.4% |
| DRAM Throughput | 54.0% |
| Achieved Occupancy | 26.1%（1 CTA/SM） |
| Registers / thread | 90，spill 0 |
| Shared bank conflicts | 0 |
| 主 stall | `long_scoreboard` **10.85** cyc/issue（占 ~63%） |
| `wait` / `not_selected` / `math_pipe` | 1.11 / 1.45 / 0.67 |

`long_scoreboard` 占主导，说明 **consumer 在等 `wgmma` 的操作数/结果**——smem 里的 A/B 喂不上来、
或 tensor 队列排满。但 occupancy 锁死 1 CTA/SM（256×128 s4 的 smem = 196 KB），STAGES=4 已经是
smem 上限（s5 会超 233 KB），加不了流水深度；`wait_group<STAGES-2>` 只能有 2 组在飞。**这就是
Hopper 单 CTA/单缓冲几何下 wgmma 的天花板。** 要再往上，方向是 TMA 直接喂更大的 K 块、
warp specialization 里给累加器做 `setmaxnreg` 重新分配（DeepGEMM 的 1-warpgroup/248-reg 布局），
或用 cs/分布式共享内存——都超出本篇范围。

**最终：29 篇 gate 最好 612.2 TFLOPS @32k（cuBLAS 752.6 的 81.3%）、547.0 @16k（cuBLAS 734.6 的 74.5%）。**
16k 的最佳是 `128×128 s3`（547.0），32k 是 `256×128 s4`（612.2）——和 28 篇的结论一致：
**tall-skinny GEMM 先算 wave 数再选 geometry。**

---

## 7. 什么情况下该上 cluster multicast

把 26/29 两篇合起来，可以给出一个可操作的判据：

| 条件 | 是否该 multicast |
|---|---|
| 瓶颈在 **L2 字节**（`lts__t_sectors` 高、tensor 不高） | ✅ 值得试，从小 cluster（CN=2）开始 |
| 瓶颈在 **tensor/发射**（`sm__throughput` 已 65%+） | ❌ 耦合只会让 tensor 更饿，实测 −23% |
| 想减少的重读，**放大 tile（BM/BN）也能做到** | ❌ 放大 tile 无耦合，严格更优 |
| 共享 operand 是被 **`M/BM` 次重读的权重**（大重读） | ✅（CN=2 时 +5.6%） |
| 共享 operand 只被 **`N/BN` 次重读**（小重读，如 3） | ❌ 省不了多少字节，不值得耦合 |
| 纯 DRAM 带宽场景（masked decode） | ❌ 26 篇实测 −3.9% |

一句话：**multicast 是「用跨 CTA 耦合换 L2 字节」的交易；先确认瓶颈是字节、且没有更便宜的
tile 放大方案，再上它。**

---

## 8. 小结

- 给 28 篇的 gate GEMM 实现了 **A 轴（`cluster.x`）与 B 轴（`cluster.y`）两种 TMA cluster multicast**，
  协议与 26 篇一致（multicast + `expect_tx` + 私有/共享 empty barrier 分离 + `mapa` 到 rank0 + 两次
  `cluster_sync`）。
- **A-multicast 实测把 L2 读 sector 砍 26%（46.2→34.0 M）、L2 吞吐 80%→48.5%，但整体慢 23%**：
  瓶颈是 tensor（69%）不是 L2，cluster 耦合把它压到 51%。
- **B-multicast `cluster.y=2` 在 128×128 上 +5.6%（556→587）**，因为它砍的是被重读 `M/BM` 次的权重；
  但它仍不如「直接把 BM 提到 256」——**放大 BM 和广播 B 是同一优化，前者无耦合**。
- **wide-N（一个 CTA 算完 N=384）失败**：3 个累加器 = 192 regs，触发 ptxas `C7511` 把 `wgmma` 串行化
  （149 / 41 TFLOPS）。Hopper 的寄存器-累加器模型下无解。
- 本系列 gate GEMM 保持 **612.2 TFLOPS @32k（cuBLAS 的 81.3%，61.9% 峰值）/ 547.0 @16k（74.5%）**。
- 产出一张 **「该不该上 multicast」的判据表**，并把它写进 `TECHNIQUES.md`。

## 下一篇预告

gate 只是前门。真正的大头在 [第 27 篇]({{< relref "cuda-kernel-opt-27-moe-router" >}})量出的
**端到端 69% 是 token 搬运**：`permute → grouped GEMM → unpermute` 里，`permuted_x` 的物化写+读
就占了 1.41 GB×2。下一篇（文章 30）做 **fused MoE**：让 grouped GEMM 直接按 `pos` gather 输入、
算完在 epilogue 里把加权结果写回 token，看能把这块搬运省掉多少。

---

### 复现

```bash
cd code/kernel-opt
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 29-moe-gate-multicast/gate_mcast.cu 32768
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/ncu.sh 29-moe-gate-multicast/gate_mcast.cu --set full -c 1 \
  --kernel-name regex:gate_mcast -- 32768 ws256x128s4_c1
```

代码：[`code/kernel-opt/29-moe-gate-multicast/gate_mcast.cu`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/kernel-opt/29-moe-gate-multicast)。
原始输出：`gate_M{16384,32768}.out.txt`、`ncu_best_c1.out.txt`、`ncu_multicast_cmp.out.txt`。

踩坑记录（同步入台账）：

1. **`cluster.x=2` 而 `grid.x=3`** → `cudaLaunchKernelEx` 报 `cluster misconfiguration`。cluster 维度必须
   整除对应 grid 维度；gate 的 `grid.x=E/BN=3` 在 BN=128 时只能取 CN=3。
2. **共享 vs 私有 operand 的 empty barrier 必须分开**：把两者合成一个 count 会死锁（且不报 CUDA error）。
3. **广播 A 省字节 ≠ 变快**：先看 `sm__throughput` 是否已经接近瓶颈，是的话 multicast 只会帮倒忙。
