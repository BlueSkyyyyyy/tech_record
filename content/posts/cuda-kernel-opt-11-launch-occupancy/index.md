---
title: "CUDA 算子调优（十一）：launch 配置与 occupancy — ILP、__launch_bounds__ 与 thread tile"
date: 2026-09-21
draft: false
weight: 11
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "occupancy", "ILP", "launch_bounds", "GEMM", "系列"]
categories: ["算子开发"]
---

上一篇（{{< relref "cuda-kernel-opt-10-ncu-deep" >}}）留了一个反直觉的观察：第 09 篇的 `gemm_reg` 被 128 个寄存器卡在 **25% occupancy**，性能却是好的；而强行把 occupancy 抬到 50% 的 `occ<8>` 反而慢了 **34 倍**。那到底什么时候该提 occupancy、什么时候不该提？

答案是：**occupancy 只是「隐藏延迟」的两条路之一，另一条是 ILP（指令级并行）。** 这篇把它讲透，并全部落地到第 09 篇的 GEMM 上：

1. **ILP × occupancy**：固定每 SM 的 block 数，改变每线程独立计算链的条数——两条路可以互相替代；
2. **`__launch_bounds__(threads, MINB)`**：对 GEMM 强行要求更多驻留 block，看寄存器和 spill 如何反噬；
3. **thread tile（每线程输出块）扫描**：每线程多算几个输出，复用变好、寄存器变多、occupancy 变低，谁说了算。

配套代码：[`code/kernel-opt/11-launch-occupancy/launch_occupancy.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/11-launch-occupancy/launch_occupancy.cu)，原始输出与 ncu 数据在同目录。硬件仍是 H100 80GB HBM3（132 SM，每 SM 65536 寄存器、2048 线程）。

---

## 0. 先建立一个「隐藏延迟」的模型

一条指令从发射到结果可用要等若干周期：FMA 大约 4~6 cycle，L1 访存几十 cycle，HBM 几百 cycle。在这段「等」的时间里，SM 的发射口不能闲着。硬件有三种办法填：

```
发射口（每 cycle 每条 scheduler 可发 1 条）
   ▲
   │  ① ILP：同一 warp 里塞几条互不依赖的指令
   │      acc0 = fma(acc0,a,b)   ← 轮到它等结果时
   │      acc1 = fma(acc1,a,b)   ← 先发这条（与上一条无依赖）
   │      acc2 = fma(acc2,a,b)
   │
   │  ② occupancy：换另一个 warp 来发
   │      warp0 在等 → 调度器切到 warp1、warp2 …
   │
   │  ③ 硬件乱序/多级流水（SM 只能部分做到）
   └──────────────────────────────────────────► cycle
```

「填满发射口所需的条件」可以粗略写成：

$$
\underbrace{\text{ILP}}_{\text{每 warp 可并行的独立指令}} \times \underbrace{\text{Eligible Warps Per Scheduler}}_{\text{每个发射口排队等着的 warp}} \ge \underbrace{\frac{\text{指令延迟}}{\text{发射间隔}}}_{\text{需要被盖住的空档}}
$$

右边由硬件延迟决定，所以**左边两个因子任何一个够大，都能把管道打满**。这就是为什么「低 occupancy + 高 ILP」和「高 occupancy + 低 ILP」都可以很快——也是这篇要验证的第一个结论。

---

## 1. 实验 1：ILP 与 occupancy 是可以互相替代的

我写了一个纯粹的 FMA 链 kernel `k_fma_ilp<ILP>`：每线程用 `ILP` 个独立累加器，做固定 `FMA_TOTAL=8192` 次乘加。链内 `acc[j]` 依赖上一次结果（有延迟），链间完全独立（提供 ILP）。用 `__launch_bounds__(256, 1)` 允许每 SM 放多个 block，分别用：

- **grid = 132**（每 SM 恰好 1 个 block）→ occupancy 固定 **12.5%**；
- **grid = 132×8**（每 SM 8 个 block）→ occupancy 拉到 **100%**。

实测（`launch_occupancy.out.txt`）：

| ILP（每线程独立链） | 1 blk/SM（12.5%） | 8 blk/SM（100%） |
|---|---|---|
| 1 | 29.9 TFLOPS（44.7%） | 63.7 TFLOPS（95.2%） |
| 2 | **53.6 TFLOPS（80.0%）** | 64.3 TFLOPS（96.1%） |
| 4 | 52.9 TFLOPS（79.0%） | 63.1 TFLOPS（94.3%） |
| 8 | 52.8 TFLOPS（79.0%） | 63.0 TFLOPS（94.1%） |

（fp32 峰值 66.9 TFLOPS。）

读这张表有两个层次：

- **在 12.5% occupancy 下**：ILP 从 1 提到 2，吞吐从 29.9 直接翻到 53.6 TFLOPS。因为 ILP=1 时每条 FMA 必须等上一条的结果，发射口一半时间是空的；ILP=2 就足以把 4~6 cycle 的延迟盖住。
- **在 ILP=1 下**：把 occupancy 从 12.5% 提到 100%，吞吐从 29.9 升到 63.7 TFLOPS——**用更多 warp 也能盖住同样的延迟**。

也就是说，`ILP=1 + 100% occupancy` 和 `ILP=2 + 12.5% occupancy` 都能到 ~80% 以上；**它们是同一件事的两种实现**。

`ncu` 的 stall 数据把机制坐实了（`ncu_ilp_stalls.out.txt`，均为 12.5% occupancy）：

| 版本 | `wait`（等固定延迟） | `not_selected`（warp 够多没人抢到） |
|---|---|---|
| `k_fma_ilp<1>` | **2.96** | 0.00 |
| `k_fma_ilp<8>` | **0.06** | 0.92 |

`wait` 就是「等指令结果」的 stall：ILP=1 时它是主要原因（2.96），ILP=8 时几乎消失（0.06），取而代之的是 `not_selected`——说明此时 warp 已经多到抢发射口了。

> **判断口径**：如果你看到 `wait` 高，先想 ILP（独立累加器、展开、重排依赖）；如果看到 `not_selected` 高，说明并行度已经足够，问题在别处（往往是复用/访存）。

---

## 2. 实验 2：对 GEMM 强行提高 occupancy 会发生什么

第 09 篇的 `gemm_reg_vec` 是 `BM=BN=128, BK=8`，每线程 `8×8` 输出，256 线程，自然占用 128 个寄存器 → 每 SM 只能放 2 个 block（25%）。

现在用 `__launch_bounds__(256, MINB)` 告诉编译器「至少要塞下 MINB 个 block」，编译器就会为了满足约束而压低寄存器、必要时 spill 到 local memory：

| 版本 | 寄存器 | local spill | blk/SM | occupancy | 耗时 | 算力 | %峰值 |
|---|---|---|---|---|---|---|---|
| `plain` | 128 | 0 B | 2 | 25.0% | 0.561 ms | 30.6 TFLOPS | 45.8% |
| `lb<1>` | 119 | 0 B | 2 | 25.0% | **0.554 ms** | **31.0 TFLOPS** | **46.3%** |
| `lb<2>` | 128 | 0 B | 2 | 25.0% | 0.561 ms | 30.6 TFLOPS | 45.8% |
| `lb<3>` | 80 | 296 B | 3 | 37.5% | 2.266 ms | 7.6 TFLOPS | 11.3% |
| `lb<4>` | 64 | 520 B | 4 | 50.0% | 4.869 ms | 3.5 TFLOPS | 5.3% |

这张表是本篇最想让人记住的一张：

- **从 25% 到 50% occupancy，性能掉了 ~8.7 倍**（31.0 → 3.5 TFLOPS）。
- 原因就是 spill：为了在每 SM 塞下 4 个 block，编译器把 8×8=64 个累加器的一部分放不进寄存器，只能写回 local memory 再读回来。local 的读写要经过 L1/L2，延迟长、带宽还占着。
- 更微妙的是 `lb<3>`：occupancy 只从 25% 抬到 37.5%，spill 也「只有」296 B，吞吐却已经腰斩到 7.6 TFLOPS。**spill 的伤害是非线性的。**

`ncu` 的 stall 对照（把 `plain` 和 `lb<4>` 放一起，`ncu_plain_stalls.out.txt` / `ncu_lb4_stalls.out.txt`）：

| stall reason | `plain`（25%） | `lb<4>`（50%，spill） |
|---|---|---|
| `long_scoreboard`（等访存/local） | 0.94 | **16.06** |
| `barrier`（等 `__syncthreads`） | 0.62 | 5.47 |
| `wait`（等固定延迟） | 0.20 | 1.44 |
| `not_selected`（warp 够多） | **2.24** | 0.21 |

`plain` 的主要 stall 是 `not_selected`（2.24）——**这是一个好信号**：warp 已经多到抢发射口，说明并行度不缺。而 `lb<4>` 的主要 stall 变成 `long_scoreboard`（16.06），正是 local spill 的读写延迟。occupancy 数字变好看了，实际却在等内存。

> **结论**：`__launch_bounds__` 的第二个参数（MINB）只有在「寄存器本来就没用满、且瓶颈真是发射口空着（`not_selected` 很小、`no_eligible` 高）」时才有意义。对寄存器受限的 kernel 硬压，只会换来 spill。第 10 篇的 `occ<8>`、这里的 `lb<4>` 是同一个坑。

---

## 3. 实验 3：thread tile —— 复用、寄存器与 occupancy 的三角

前面说过「每线程多算几个输出」能提升 smem 载入的复用：第 09 篇用 `TM=TN=8`，一次 `As[...][k]`/`Bs[k][...]` 载入被 8 次 FMA 复用。现在把每线程的输出块从 `2×2` 扫到 `8×8`：

| 配置（block 输出） | 每线程输出 | 线程/block | 寄存器 | blk/SM | occupancy | 算力 | %峰值 |
|---|---|---|---|---|---|---|---|
| `64×64` | 2×2 | 1024 | 32 | 2 | **100%** | 12.6 TFLOPS | 18.8% |
| `64×64` | 4×4 | 256 | 66 | 3 | 37.5% | 18.3 TFLOPS | 27.3% |
| `64×64` | 8×8 | 64 | 168 | 6 | 18.8% | 15.8 TFLOPS | 23.7% |
| `128×128` | 8×8 | 256 | 128 | 2 | 25.0% | **31.3 TFLOPS** | **46.8%** |

两个现象值得说：

1. **occupancy 和性能基本反向**：occupancy 最高的 `2×2`（100%）反而最慢（18.8%）。因为 `2×2` 每线程只有 4 个累加器、每算一个输出要从 smem 读一堆值，**载入/计算比最差、ILP 也不够**。`8×8` 用 25% 的 occupancy 拿到 46.8%，是 `2×2` 的 2.5 倍。
2. **光把 tile 调大也不够**：同样是 `8×8`，`64×64`（64 线程/block）只有 23.7%，`128×128`（256 线程/block）才是 46.8%。前者每 block 才 2 个 warp，`__syncthreads` 一同步就没人干活，而且寄存器 168 反而把 occupancy 压到 18.8%。**block 要有足够多的 warp 才能把同步和延迟的空档填上。**

这一节和第 09 篇的结论合起来就是：**GEMM 的 launch 配置是「复用度 / 寄存器 / 线程数」三者的平衡**，不是单看 occupancy。

---

## 4. 一张选 launch 配置的决策表

把三组实验收成一套可操作的流程：

```
性能不达预期
   │
   ├─ 看 ncu Scheduler：No Eligible 高吗？
   │     ├─ 高（发射口空着）→ 再分两类：
   │     │     ├─ wait 高        → 加 ILP：独立累加器 / 展开内层循环 / thread tile 调大
   │     │     └─ long_scoreboard 高 → 加 occupancy 或做访存优化（预取/cp.async）
   │     └─ 低（发射口满，Compute Throughput 高）→ 已经算力受限，别再动 occupancy
   │
   ├─ 看 Occupancy：限制因子是谁？
   │     寄存器 → 减小 thread tile / BK 可以降寄存器；但先确认降了不会丢复用
   │     smem     → 减小 BK 或分块
   │     线程/block → 调整 block 尺寸
   │
   └─ 看 Warp State：not_selected 很高？
         → 说明 warp 足够甚至偏多，**不要再提 occupancy**，考虑减小 block/线程改善局部性
```

几个可以背下来的经验值（H100）：

- FMA 延迟约 4~6 cycle，**ILP≥2~4** 通常就够盖住；不够时优先加 ILP 而不是加 occupancy。
- 寄存器受限的 kernel，`__launch_bounds__(T, MINB)` 的 `MINB` 设成「自然占用数」即可（本实验是 2），**不要为了好看的 occupancy 数字往上顶**。
- block 至少 **4 个 warp（128 线程）** 再用 `__syncthreads`，否则同步开销很显眼；GEMM 的 128×128 分块用 256 线程（8 warp）比较稳。

---

## 小结

- **ILP 与 occupancy 是隐藏延迟的两条可替代路径**：12.5% occupancy + ILP=2 能到 80%，ILP=1 + 100% occupancy 能到 95%；`wait` stall 从 2.96 掉到 0.06 就是证据。
- **对寄存器受限的 GEMM 硬提 occupancy 是负优化**：`__launch_bounds__(256,3/4)` 把寄存器压到 80/64、spill 296/520 B，性能从 31.0 TFLOPS 掉到 7.6/3.5（最惨慢 8.7×），`long_scoreboard` 从 0.94 飙到 16.06。
- **thread tile 决定复用与 ILP，occupancy 只是结果**：每线程 `8×8` 比 `2×2` 快 2.5 倍，哪怕后者的 occupancy 是它的 4 倍（100% vs 25%）。
- **block 要有足够 warp**：同样 `8×8`，64 线程/block 只有 23.7%，256 线程/block 有 46.8%。
- 本实验的 `128×128 / 8×8 / 256 线程`（`plain` 128 寄存器、25% occupancy、30.6 TFLOPS）与第 09 篇 `gemm_reg_vec` 的 30.1 TFLOPS 完全对齐，说明模板化没有引入额外开销。

> 下一篇：CUDA 算子调优（十二）· 异步拷贝与流水线——把第 09 篇的 `cp.async` 单级双缓冲拓展到多级流水，看它如何在不加 occupancy 的前提下把 `long_scoreboard` 进一步压下去。
