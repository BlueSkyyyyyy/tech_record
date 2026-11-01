---
title: "Megatron 源码精读（二十五）：一个 training step 的耗时怎么估算——计算、通信与气泡的完整账单"
date: 2026-09-07
draft: false
tags: ["megatron-lm", "系列", "训练框架", "性能估算", "roofline", "MFU"]
categories: ["训练框架"]
weight: 25
series: ["megatron-code"]
---

上一篇[《显存估算》]({{< relref "megatron-code-24-memory-estimation" >}})算「塞不塞得下」，这一篇算「塞下了，一步要跑多久」。

**一句话结论**：总计算量由 `6 × 参数量 × 全局 token 数` 决定，跟并行怎么切无关；单卡时间 = 总计算量 / (卡数 × 峰值 × MFU)，再乘上「气泡放大」并加上「藏不掉的通信」。估算的价值不在于算准，而在于**判断瓶颈是算力、通信还是气泡**。

---

## 1. 三个基本量与两条硬件红线

时间估算只需要三类输入：

| 量 | 决定什么 | 从哪来 |
|---|---|---|
| FLOPs | 计算时间 | 参数量 × token 数 |
| 通信字节 | 通信时间 | 激活/梯度大小 ÷ 并行度 |
| 峰值算力 / 带宽 | 把上面两项换成秒 | 硬件规格 |

先记住两条红线（H100 SXM，**稠密**算力，不要把「稀疏 2×」的营销数字当默认值）：

- **算力**：BF16/FP16 tensor core 约 989 TFLOPS，FP8 约 1978 TFLOPS。A100 分别是 312 / 624 TFLOPS。
- **带宽**：机内 NVLink 约 900 GB/s **双向**（单向约 450 GB/s）；跨机 IB/RoCE 每卡约 25~50 GB/s 单向。

差了一个数量级，这决定了「哪些通信必须藏、哪些藏不掉」。

## 2. 计算时间：6PB 与 MFU

### 2.1 每 token 6P 是怎么来的

对 dense decoder-only Transformer：

- 前向：每个参数参与一次乘加，约 $2P$ FLOPs/token；
- 反向：对激活求导（$2P$）+ 对权重求导（$2P$），共 $4P$。

所以

$$
\text{FLOPs per token} \approx 2P + 4P = 6P
$$

一个 step 处理 $B$ 个全局 token：

$$
\text{FLOPs per step} = 6PB
$$

这个数与 TP/PP/DP 怎么切**完全无关**——并行只是把同一份计算分给更多卡。MoE 要把 $P$ 换成「每 token 激活的参数量」，并额外加上路由、dispatch 的开销。

### 2.2 单卡时间

$$
T_{\text{compute}} = \frac{6PB}{N \cdot \text{peak} \cdot \text{MFU}}
$$

其中 $N$ 是总卡数，`peak` 是所用精度的稠密峰值，MFU 是实际利用率。

例：7B、$B=4\times10^6$ token、1024 张 H100、BF16、MFU=40%：

$$
T_{\text{compute}} = \frac{6 \times 7\times10^9 \times 4\times10^6}
{1024 \times 989\times10^{12} \times 0.4} \approx 0.42\ \text{s}
$$

这就是「纯算力下限」。**任何低于它的估算都是错的**，剩下的通信与气泡只会让它变长。

### 2.3 MFU 为什么上不去

- **非 GEMM 部分**（LayerNorm、softmax、mask、embedding、dropout）算力强度低，落在 roofline 的内存受限区；
- **GEMM 本身**受 tile 形状、尾块、wave 量化影响，到不了 100%；
- **通信没藏干净**时 GPU 空转，不产生任何 FLOP；
- **重计算**多做一遍前向，账面 FLOPs +33%，MFU 被摊薄。

经验值：dense BF16 调得好 35%~55%；FP8 可以到 50%~65%。

## 3. 通信时间：三类，算法不同

### 3.1 TP：每层都通信，量正比于激活

TP 的每次 row-parallel 线性层都要一次 all-reduce。ring all-reduce 的耗时是

$$
T_{\text{AR}} \approx \frac{2(n-1)}{n}\cdot\frac{S}{bw}
\approx \frac{2S}{bw}
$$

其中 $n$ 是组大小，$S$ 是张量字节数，$bw$ 是单向带宽。

TP 组一般 ≤ 8 卡、走 NVLink，$bw$ 大；但**它每层都发生**，总字节量 $\propto$ 激活量 $\times$ 层数。所以：

- batch/seq 小、层多时，TP 通信容易露出来；
- batch 大时，计算变长，TP 通信被摊薄。

### 3.2 DP：每个 step 一次，量正比于参数

梯度归约的字节量是 $S_{\text{grad}} = P \times \text{bytes\_per\_param}$，与 batch、序列都无关。

$$
T_{\text{RS}} \approx \frac{dp-1}{dp}\cdot\frac{S_{\text{grad}}}{bw_{\text{inter}}}
$$

（all-reduce 前面多一个 2；开 ZeRO 后走 reduce-scatter，只算一份。）

DP 组通常跨机，带宽只有机内的 1/10~1/20，但它是「每 step 一次的大量通信」，可以靠 `overlap_grad_reduce` 切成 bucket 藏进反向。**它是大模型训练里最值得先查的一项。**

### 3.3 PP：量最小，代价在气泡

PP 只在相邻 stage 间传激活，字节量是 $s \cdot b \cdot h$ 量级，时延为主。`overlap_p2p_comm` 可以把 send/recv 藏进计算（注意：该开关在非 interleaved 1F1B 下不可用，见第 20 篇）。

除了超长序列 + 小 batch 的极端情况，**PP 通信本身可以忽略，真正贵的是 bubble**。

## 4. 气泡与合成

### 4.1 1F1B 的 bubble 率

$$
\text{bubble} \approx \frac{pp-1}{m + pp - 1}
$$

$m$ 是 microbatch 数。$m$ 越大 bubble 越小，但并飞的激活越多、显存越紧——这就是第 24 篇里「显存换时间」的那条 trade-off。interleaved（$vp>1$）大约能把 bubble 再除以 $vp$。

### 4.2 合成公式

别把三项直接相加，也别全取 max。正确的结构是：

$$
T_{\text{step}} \approx \frac{T_{\text{compute}}}{1-\text{bubble}} + T_{\text{comm}}^{(\text{暴露})}
$$

- 气泡只放大**计算**，因为空转的是计算流水线；
- 能藏进计算的通信（DP 的 bucket 化 all-reduce、PP 的异步 p2p）不计入第二项；
- 藏不掉的（DP 尾部、参数 all-gather 的启动延迟、microbatch 太小时的 TP all-reduce）计入 $T_{\text{comm}}^{(\text{暴露})}$。

## 5. 完整算例：7B，64 张 H100

配置：BF16，`tp=2, pp=2, dp=16`，全局 batch $B=2\times10^6$ token，$m=32$。

1. **总 FLOPs**：$6 \times 7\times10^9 \times 2\times10^6 = 8.4\times10^{16}$。
2. **计算**：每卡 $8.4\times10^{16}/64 = 1.31\times10^{15}$；MFU=45% → $T_{\text{compute}} = 2.95$ s。
3. **气泡**：$(2-1)/(32+2-1)=3\%$ → 放大 1.03 倍。
4. **DP 通信**：$S_{\text{grad}} = 1.4\times10^{10}$ B；跨机有效带宽 50 GB/s，$dp=16$ → $T_{\text{RS}} \approx (15/16)\times1.4\times10^{10}/50\times10^9 \approx 0.26$ s。开 overlap 后暴露约 0.1 s。
5. **合成**：$T_{\text{step}} \approx 1.03 \times 2.95 + 0.1 \approx 3.14$ s。

由此得到：

$$
\text{tokens/s} = \frac{B}{T_{\text{step}}} \approx 6.4\times10^5
$$

反推 MFU：

$$
\text{MFU} = \frac{6PB}{N \cdot \text{peak} \cdot T_{\text{step}}} \approx 42\%
$$

42% 低于假设的 45%，差额正好是通信暴露和气泡——账对得上，说明模型没算错。

**换 FP8**：峰值翻倍到 1978 TFLOPS，MFU 取 55%，$T_{\text{compute}} \approx 1.20$ s，$T_{\text{step}} \approx 1.34$ s，约 2.3 倍加速。但注意：FP8 只加速 GEMM，通信字节几乎不变，所以**通信占比会从 3% 升到 8% 左右**，overlap 变得更关键。

## 6. 用实测反推：排查顺序

拿到一个真实 step 时间，先算「理论下限」再逐层加回：

1. 算 $T_{\text{compute}}$。实测远高于它 → 不是算力问题，往下走。
2. 算 $T_{\text{RS}}$（DP）。如果它和「实测 − 计算」同量级 → 先查跨机带宽与 `overlap_grad_reduce`。
3. 算 bubble。$m$ 太小、$pp$ 太大时，这一项会非常显眼。
4. 都不是 → 看 kernel 级 profiler：非 GEMM 占比、launch 间隙、NCCL 等待。

经验锚点：dense BF16 长期低于 30% MFU，基本能确定有通信暴露或重计算没消化。

## 7. 小结

- **总量与并行无关**：$6PB$；单卡下限 $6PB/(N\cdot\text{peak}\cdot\text{MFU})$。
- **时间不是加法**：气泡放大计算，暴露的通信相加，可藏的通信取 max 后归零。
- **三类通信三种性格**：TP 每层小量（机内）、DP 每步大量（跨机、可藏）、PP 量最小但带来 bubble。
- **bubble 是 PP 的固定税**：$(pp-1)/(m+pp-1)$，$m$ 越大越小，但和显存直接冲突。
- **反推 MFU 是最实用的自检**：算出来对不上，说明有项没记进账。

---

**版本提示（2026-10）**：上游新增的 GTP 把权重 all-gather 做成逐权重预取链（第 26 篇），相当于把 ZeRO-3 的通信藏进上一层计算；batch-invariant 模式与 Triton autotune 固定表（`megatron/core/tuning/`）会影响 MFU 的可复现性，但不改变本篇的估算结构。

**行号基线**：本篇为方法篇，不引用行号；涉及机制见第 6、18、20、21 篇（commit `f713506ce`）。
