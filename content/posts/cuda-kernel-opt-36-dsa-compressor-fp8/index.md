---
title: "CUDA 算子调优（三十六）：DSA Compressor 投影的「L2 墙」拆解 —— 三种压 L2 的手段全负，以及用 FP8 把投影推到 1.4×"
date: 2026-09-21
draft: false
weight: 36
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "DeepSeek", "DeepSeek-V4", "DSA", "compressor", "KV压缩", "wgmma", "TMA", "warp specialization", "cluster multicast", "FP8", "e4m3", "per-block", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

上一篇（[第 35 篇]({{< relref "cuda-kernel-opt-35-dsa-compressor" >}})）把 DeepSeek-V4 的
`Compressor`（KV 门控池化）拆成了「合并投影 GEMM + 融合池化」两个 kernel，其中合并投影
`Y[M, 2C] = X[M, D] @ Wm[2C, D]ᵀ` 用 bf16 跑到 650 TFLOPS，并留下一个判决：

> **这个 GEMM 不是算力受限，而是 L2 带宽受限**（ncu `L2 Cache Throughput` 85~88%、`Compute` 只有 57~59%）。

这一篇就是来**拆这堵 L2 墙**的。按理说这种「权重只有 14.7 MB、却被 M/BM=128 个 m-tile 反复重读」
的 tall-skinny GEMM，正是 TMA cluster multicast 的主场。但我老老实实把
**A 广播 / B 广播 / L2 persisting / K-split / 消费者 fence** 五条路都试了一遍，
结论是：**全负**。而 ncu 也顺带把 35 的判决翻了个案——**最佳配置根本不是 L2 受限，是 tensor pipe 受限（76%）**。

既然 bf16 已经摸到本设计的 tensor 天花板，真正的提速只能从**换数据格式**来：DeepSeek-V4-Pro 的
`config.json` 里写着 `e4m3 + ue8m0 + weight_block 128×128`，于是我把投影按真实量化方案做了
**FP8 per-block** 版，**959 → 974 TFLOPS（FP8 峰值的 49%），相对 bf16 1.38~1.40×**。

先给结论：

| 手段 | 结果 | 原因 |
|---|---|---|
| bf16 最佳 `256×128×64 s4` | **692 TFLOPS（70% 峰值）** | ncu：tensor pipe **76%**、L2 58% → **tensor 受限** |
| TMA cluster multicast（A 沿 x / B 沿 y） | 全部更慢（−3.7% ~ −19%） | 字节确实省了，但跨 CTA 耦合把 tensor 活跃度拖下来 |
| L2 persisting window（钉住 Wm） | 618 vs 632（−2.2%） | hit rate 已 82%，瓶颈不在命中率 |
| K-split 双累加链 | 崩（449 / 70 TFLOPS） | ptxas `C7517/C7518` 把 wgmma 串行化（复现第 24 篇的坑） |
| **FP8 e4m3 per-block（128×128）** | **959（ratio128）/ 974 TFLOPS（ratio4）** | 相对 bf16 **+38% / +40%**；瓶颈翻转成 L2 73% + barrier |

> 编号说明：原「下一步」里的文章 36 是「compressor 投影的 L2 墙」。本轮把 L2 墙的正反两面都查完，
> 并把 FP8 化一起做了。全部代码在 `code/kernel-opt/36-dsa-compressor-mcast/`。

## 一、先把 shape 和基线固定住

shape 与 35 完全一致，取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`hidden_size=7168`、
`head_dim=512`、`qk_rope_head_dim=64`、`compress_ratios ∈ {128, 4, 0}`：

- ratio=128（`coff=1`）：`C=512`，合并权重 `Wm[1024, 7168]`（bf16 14.7 MB），
  `M=32768` → 投影 FLOPs `2·M·N·D = 481.0 GFLOP`；
- ratio=4（`coff=2`，overlap）：`C=1024`，`Wm[2048, 7168]`（29.4 MB），FLOPs `962.1 GFLOP`。

两个 GEMM 都是 **N 小 M 大** 的 tall-skinny：输出 tile 数 `(N/BN)×(M/BM)`，ratio=128 时只有
`8×128=1024` 个。kernel 直接复用 23/28/35 篇的骨架：

```
TMA(cp.async.bulk.tensor.2d + SW128) → mbarrier → 1 producer warp + (BM/64) consumer warpgroup
                                                  → wgmma.m64n128k16 SS(SW128) → fp32 epilogue
```

`code/kernel-opt/36-dsa-compressor-mcast/` 下三个文件：

| 文件 | 内容 |
|---|---|
| `compressor_mcast.cu` | bf16 骨架 + **TMA cluster multicast**（`AXIS=1` 沿 `cluster.x` 广播 A、`AXIS=2` 沿 `cluster.y` 广播 B）+ **L2 persisting window** + K-split 实验开关 |
| `compressor_fp8.cu` | **同进程内**跑 bf16 与 FP8 per-block 两条投影路径，便于同口径对照 + 量化精度 |
| `*.out.txt` | 全部实测原始输出 |

基线（`compressor_mcast.cu`，同一进程、同样的 warmup/iters）：

```
-- proj GEMM sweep, no multicast (BM x BN, BK=64) --
  128x128s2      0.8913 ms   539.69 TFLOPS
  128x256s3      0.7834 ms   614.02 TFLOPS
  256x128s3      0.8234 ms   584.18 TFLOPS
  256x128s4      0.7607 ms   632.32 TFLOPS   ← bf16 最佳几何
```

## 二、翻案：35 的「L2 墙」其实是 tensor 墙

35 篇的 ncu 是在 `128x128s2` 上采的（`L2 84%`、`Compute 58%`）。但那不是最佳配置。把 ncu 打到
**最佳几何 `256×128×64 s4`** 上，结论完全不同：

```
proj_mcast_kernel<256,128,64,4,0,1>  (8,128,1)x(544,1,1)   regs=90
  gpu__time_duration.sum                     691.94 us
  sm__pipe_tensor_cycles_active                76.35 %   ← 真正的天花板
  lts__throughput                              58.02 %   ← L2 反而不是瓶颈
  l1tex__throughput                            60.16 %
  dram__throughput                             31.29 %
  sm__warps_active                             25.98 %
  stalled_long_scoreboard / issue            9.56        ← 等 wgmma 操作数
```

为什么换个几何就差这么多？看两个操作数的重读次数：

$$
\text{A 重读} = \frac{N}{BN}, \qquad \text{B 重读} = \frac{M}{BM}
$$

- `BM=128,BN=128`：A 重读 8×、B 重读 **256×** → B 的 L2 流量爆炸，`L2 84%`；
- `BM=256,BN=128`：B 重读砍半到 **128×**，L2 掉到 58%，tensor 升到 76%。

也就是说 35 篇看到的 L2 墙**是 `BM=128` 自己的问题**——把 `BM` 从 128 放大到 256（TMA box 单维上限）
就是最有效的「压 L2」，不需要 cluster。这正好印证了第 29 篇的判据：**「放大 BM」和「广播 B」是同一个
优化，且前者没有跨 CTA 耦合**。

## 三、三种压 L2 的手段，全负

### 3.1 TMA cluster multicast

协议完全照搬 26/29 篇：leader 发 `cp.async.bulk.tensor.2d...multicast::cluster` + mask，每个 CTA
各自 `arrive.expect_tx`，私有 operand 本地 `arrive(empty)`、共享 operand 由每个消费者 warp 的 lane0
用 `mapa.shared::cluster` 投到 rank0。`AXIS=1` 走 `cluster.x`（同 m-tile 的 8 个 n-tile 共享 A），
`AXIS=2` 走 `cluster.y`（同 n-tile 的相邻 m-tile 共享 B）。协议跑通、结果与 `CN=1` 逐位一致，但：

```
  256x128s4           0.7607 ms   632.32 TFLOPS     ← baseline
  256x128s4 A-cx2     0.7897 ms   609.12 TFLOPS  (-3.7%)
  256x128s4 A-cx4     0.8569 ms   561.35 TFLOPS
  256x128s4 A-cx8     0.8959 ms   536.90 TFLOPS  (-15%)
  256x128s4 B-cy2     0.8063 ms   596.58 TFLOPS  (-5.6%)
  256x128s4 B-cy4     0.8911 ms   539.84 TFLOPS
  256x128s4 B-cy8     0.9373 ms   513.20 TFLOPS  (-19%)
```

A 广播把 A 的 L2 读字节砍到 `1/CN`（`CN=8` 时理论 −44%），B 广播同理，**但全都更慢**。原因和第
26/29 篇一致：广播省的是**字节**，代价是 cluster 内 CTA 的访存被 mbarrier 耦合在一起，tensor pipe
的活跃度被拖低。本 kernel 的瓶颈本来就是 tensor（76%），给一个非瓶颈单位减负只会添乱。

### 3.2 L2 persisting window

ratio=128 时 `Wm` 只有 14.7 MB，远小于 H100 的 50 MB L2，所以直觉上「把它钉进 L2」应该有用。
用 `cudaAccessPolicyWindow`（`hitProp=cudaAccessPropertyPersisting`，`num_bytes=14.7MB`）试了：

```
  256x128s4           632.32 TFLOPS
  256x128s4 persist   618.66 TFLOPS  (-2.2%)
```

没用。ncu 早告诉我们 `L2 Hit Rate` 已经 **82%**——B 本来就基本常驻 L2 了，把 82% 抬到 83% 没有意义。
**瓶颈在 L2→SM 的请求率/延迟，不在命中率。**

### 3.3 K-split 双累加链（想喂饱 tensor pipe）

既然 tensor pipe 只有 76%，一个自然的想法是给每个 warpgroup 两条**独立累加链**（`acc[2][64]`，k 块
交替），增加可并行的 wgmma。第一版用运行期下标 `acc[q & 1]`：

```
  128x128s3 K2     8.8557 ms     54.32 TFLOPS   ← 崩到 5%
```

ptxas 报 `C7518: wgmma instructions are serialized due to program dependence on compiler-inserted
WG.DP in divergent path`。把 `q` 循环按 `KSPLIT` 展开、让累加器下标变成编译期常量后，`128x128s3`
恢复到 449 TFLOPS，但 `128x256s3 / 256x128s4` 仍然崩到 ~70（`C7517` 串行化）。而且就算不崩，
`128x128s3 K2` 的 449 也远低于基线 527——多出来的 `acc[2][64]` 寄存器把 occupancy 从 2 CTA/SM
压到 1。**这正是第 24 篇的坑：动态选累加器 / 跨块保留累加器，ptxas 会主动把 wgmma 串行化。**

### 3.4 顺带否掉一个「免费优化」

第 23 篇的消费者循环里，`mbar_wait` 之后有一句 `fence.proxy.async.shared::cta`。从原理上讲，
TMA 写 smem 和 wgmma 读 smem **都在 async proxy**，mbarrier 已经完成两者排序，消费者侧再插一句
generic↔async 的 fence 是多余的。加开关 `CFENCE=0` 后，paired 测量：

```
CFENCE=1  652.11 / 646.39 / 647.36 TFLOPS
CFENCE=0  650.25 / 645.56 / 651.87 TFLOPS
```

**中性**（在噪声内）。结论：fence 不是瓶颈，但也不是必需的——可以删。第 23~36 篇的所有 TMA kernel
都可以去掉消费者侧这句 fence。

## 四、multicast 判据（更新版）

把这轮和 26/29 篇合起来，得到「该不该上 cluster multicast」的判据：

| 条件 | 说明 |
|---|---|
| ① ncu `lts__throughput` 是**最高项**（>75%） | multicast 只治「L2 子系统占用」这一种病 |
| ② 瓶颈确实在**字节**（`lts__t_sectors` 读放大高） | 不是请求率/延迟、不是命中率 |
| ③ 已经没法放大 `BM`/`BN`（TMA box 上限 256） | 「放大 tile」和「广播」是同一个优化，优先前者 |
| ④ `sm__pipe_tensor_cycles_active` < ~60% | tensor 已 >70% 时，广播的耦合只会更慢 |
| ⑤ 从 `CN=2` 起试 | `CN=4/8` 的耦合代价通常大于收益 |

本 kernel 满足 ③④ 的反面（tensor 76%），所以 multicast 全负；第 26 篇的 masked decode（纯 DRAM
带宽）也不满足 ①②，同样负。

## 五、换维度：FP8 e4m3 per-block，直接 1.4×

既然 bf16 的 tensor pipe 已经到 76%、又受限于 TMA box ≤ 256 和 Hopper 单 CTA 几何，那不如**换更
小的数制**——FP8 峰值是 bf16 的 2 倍。DeepSeek-V4-Pro 的 `config.json` 本来就写着：

```json
"quantization_config": {
  "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
  "weight_block_size": [128, 128], "scale_fmt": "ue8m0"
}
```

即 **e4m3 权重 + 128×128 权重块缩放 + 1×128 动态激活缩放**。这正是第 24/31 篇做过的
「`fin += sa·sb·acc` 每 128-k 块折算」，直接拿 `fp8_tma_pb_kernel`（`mma` 换成 `wgmma.m64n128k32`、
TMA 的 SW128 与 bf16 逐字节同构）接上即可。在**同一个进程**里跑 bf16 与 FP8 两条路径：

```
-- ratio=128, M=32768, N=1024  (481.0 GFLOP) --
  bf16 256x128s4        0.6947 ms   692.47 TFLOPS  (70.0% bf16 peak)
  fp8  128x128s4        0.5019 ms   958.49 TFLOPS  (48.5% fp8  peak)
  fp8  128x128s5        0.5015 ms   959.12 TFLOPS  (48.5% fp8  peak)
  fp8  128x128s3        0.5071 ms   948.54 TFLOPS
  fp8  192x128s4        0.6116 ms   786.49 TFLOPS
  fp8  64x128s4         0.6883 ms   698.88 TFLOPS
  fp8  128x256s3        2.4180 ms   198.94 TFLOPS  (寄存器墙)

-- ratio=4, M=32768, N=2048  (962.1 GFLOP) --
  bf16 256x128s4        1.3768 ms   698.76 TFLOPS  (70.7% bf16 peak)
  fp8  128x128s4        0.9876 ms   974.13 TFLOPS  (49.2% fp8  peak)
  fp8  128x128s5        0.9887 ms   973.08 TFLOPS  (49.2% fp8  peak)
```

- ratio=128：`692.5 → 959.1 TFLOPS`，**1.385×**；
- ratio=4：`698.8 → 974.1 TFLOPS`，**1.394×**；
- 权重字节 bf16→fp8 **减半**（14.7→7.3 MB、29.4→14.7 MB），对 L2 常驻也更友好。

### 5.1 ncu：瓶颈从 tensor 翻转成 L2 + barrier

把 ncu 打到 FP8 最佳配置上，和 bf16 最佳配置并排：

| 指标 | bf16 `256×128 s4` | FP8 `128×128 s4` |
|---|---|---|
| `gpu__time_duration` | 691.9 µs | 507.2 µs |
| `sm__pipe_tensor_cycles_active` | **76.4%** | 51.0% |
| `lts__throughput`（L2） | 58.0% | **73.0%** |
| `l1tex__throughput` | 60.2% | 49.4% |
| `dram__throughput` | 31.3% | 24.6% |
| registers / thread | 90 | **154** |
| achieved occupancy | 26.0% | **13.8%** |
| 主导 stall | `long_scoreboard` 9.56 | **`barrier` 1.18** |
| 次主导 stall | `wait` 1.09 | `long_scoreboard` 1.24 |

这张表讲了一个完整的故事：

1. **FP8 为什么快**：同一块 tensor 每个周期做 2× 的 MACs。bf16 时 tensor pipe 76% 已封顶，FP8 只
   需 ~51% 就能把时间砍到 0.73×（实测 507/692 = 0.73）。
2. **FP8 为什么没到 2×**：per-block 折算要求每线程多一个跨 k 保留的 fp32 `fin`（+64 regs），
   寄存器 90→154，越过 288 线程 2-CTA 门槛（65536/(2·288)=113），occupancy **2→1 CTA/SM**。
   于是 ncu 主导 stall 从「等 wgmma 操作数」（`long_scoreboard`）变成「每 128-k 块 `wgmma.wait0`
   折算时排空」（`barrier`），L2 占用升到 73%。
3. 这就是第 24/31 篇反复确认的 **per-block 寄存器墙**：代价是 occupancy，不是折算 FLOPs。
   而 `128×256` 的 `acc+fin=128` regs 直接 spill（168 regs + 1320 B spill）→ 只剩 199 TFLOPS。

想再往上（FP8 峰值 50% → 70%+）需要 DeepGEMM 的 **1-warpgroup / `setmaxnreg 248`** 布局，把
`fin(128)+acc(64)=192` 塞进单 warpgroup 且不掉 CTA——那是第 24 篇就记在 backlog 的下一步。

### 5.2 量化精度

用同分布随机数据（并非真实激活，真实激活更有结构、通常误差更小）在 CPU 上对拍「池化后的最终输出」，
FP8 投影（1×128 激活 + 128×128 权重、float scale）相对 bf16 全流程：

```
ratio=128:  pooled-output max_abs_diff = 1.211e-01  (ref ~3.820,  3.17%)
ratio=4:    pooled-output max_abs_diff = 1.324e-01  (ref ~3.466,  3.82%)
```

即**最大相对误差 ~3~4%**（bf16 自身的对拍误差是 0.2%）。DSA 的池化里还有 RMSNorm，会把整体尺度
拉回，所以这是偏保守的上界；真实部署的 ue8m0 幂次 scale 会比这里的 float scale 略粗，但换来的是
scale 可以只存 1 字节。

## 六、端到端

投影是 compressor 的绝对大头（池化是纯访存的 ~0.05 ms，见 35 篇）。ratio=128、M=32768 时：

| 路径 | 投影 | 端到端（含 35 篇池化 0.048 ms） |
|---|---|---|
| bf16（35 篇） | 0.740 ms / 650 TFLOPS | 0.790 ms |
| bf16（本篇同进程） | 0.695 ms / 692 TFLOPS | 0.743 ms |
| **FP8 per-block** | **0.502 ms / 959 TFLOPS** | **0.550 ms** |

端到端相对 35 篇的（自研 bf16）e2e **1.44×**；相对官方 eager 参考（35 篇测得 ratio128 e2e 2.00×）
可到 **~2.9×**。

## 七、小结

- **35 的「L2 墙」是配置假象**。ncu 打到最佳几何 `256×128×64 s4` 上，是 **tensor pipe 76% 受限**
  （L2 只有 58%）。`128×128` 的 L2 84% 是它自己 `BM=128` 造成 B 被重读 256× 的结果。
- **压 L2 的三种手段全负**：A/B cluster multicast（−3.7%~−19%）、L2 persisting（−2.2%）、K-split
  双累加链（崩）。判据更新为五条，核心是「tensor >70% 就别上 multicast」。
- **消费者侧的 `fence.proxy.async` 可删**（中性），因为 TMA 与 wgmma 同在 async proxy，mbarrier 已排序。
- **真正的提速来自换数制**：按 V4-Pro 真实的 `e4m3 + ue8m0 + 128×128` 做 FP8 per-block，
  **959（ratio128）/ 974（ratio4）TFLOPS，相对 bf16 1.38~1.40×**，权重字节减半，量化最大误差 ~3~4%。
- **剩余 gap**：FP8 只到峰值 49%，瓶颈是 per-block 的寄存器墙→1 CTA/SM→`barrier`+L2。出路是
  1-warpgroup / `setmaxnreg 248`（backlog）。

## 下一篇预告

第五部分还剩 **Paged KV-cache / flash-decoding**（主题 25）和 **W4A16 dequant-GEMM**（主题 26）
两块推理侧算子；第六部分的极限冲刺则欠着 **FP8 per-block 的 1-warpgroup/248-reg 布局**（把本篇的
974 推向 1300+）。下一篇优先做 1-warpgroup 布局，因为它是本篇、24、31、34 篇共同的未解瓶颈。
