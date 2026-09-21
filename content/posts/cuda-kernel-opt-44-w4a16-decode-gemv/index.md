---
title: "CUDA 算子调优（四十四）：W4A16 的 M=1 GEMV——M=1 时张量核有 63/64 的行在空转，不如回去做带宽最优的 GEMV"
date: 2026-09-22T03:00:00+08:00
draft: false
weight: 44
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A16", "int4", "AWQ", "GPTQ", "GEMV", "带宽", "roofline", "tensor-core", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第四篇。承接
[第 40 篇 W4A16 dequant-GEMM]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}})、
[第 41 篇 warp specialization + 波次对齐]({{< relref "cuda-kernel-opt-41-w4a16-warp-specialization" >}})
和 [第 43 篇跨 item 持久化流水]({{< relref "cuda-kernel-opt-43-w4a16-persist" >}})。

前三篇一直在把 `wgmma + 主循环内反量化` 的 GEMM 往极致推：M=64 上做到 0.0957 ms
（41 篇 / 465.6 GB/s），M=1 上做到 0.0788 ms（43 篇 / 565.5 GB/s）。但每一篇的 ncu 都在
反复说同一句话——**DRAM 只有百分之十几，瓶颈是 occupancy / 发射 / 延迟**：

| 篇 | M | 时间 | 有效权重带宽 | ncu DRAM |
|---|---|---|---|---|
| 40 | 64 | 0.1114 ms | 400 GB/s | **11%** |
| 41 | 64 | 0.0957 ms | 465.6 GB/s | — |
| 43 | 1 | 0.0788 ms | 565.5 GB/s | **18%** |

这一篇问一个更基本的问题：**M=1 的时候，到底该不该用张量核？**

结论是**不该**。这是一个典型到有点荒谬的「工具选错」案例：

- M=1 的 W4A16 上投影 `y[1,N] = x[1,K] · W[N,K]^T` 是一个 **GEMV**，权重 int4 一共
  44.6 MB，**无论怎么算都要从 HBM 读一遍**，理想时间 ~14 µs；
- 而 `wgmma.m64n128k16` 的 M 维固定 64，M=1 时 **63/64 的张量核行在算空气**，kernel 变成
  延迟受限，DRAM 只有 18%；
- 换成「**每 warp 一行、warp 内沿 K 并行 + shuffle 归约**」的带宽最优 GEMV，用 16B 向量读
  int4、把 `x` 和 scale 预取进 smem，实测 **0.0284 ms / 1566.6 GB/s（49.7% HBM）**，
  比 43 篇的 tensor-core 最好成绩（0.0798 ms）**快 2.81×**。

shape 取自 `/ssd/models/qwen3-8B/config.json`（`hidden=5120, intermediate=17408`，
`group=128`，对称 int4），默认测 `M=1, N=17408, K=5120`。

---

## 一、先算一笔账：M=1 的 W4A16 到底是谁的瓶颈

W4A16 一次前向的算术强度（FLOP/Byte）：

$$
\text{AI}(M) = \frac{2MNK}{\underbrace{NK/2}_{\text{int4 权重}} + \underbrace{N(K/128)\cdot 4}_{\text{fp32 scale}} + \underbrace{MK\cdot 2}_{\text{bf16 激活}} + \underbrace{MN\cdot 4}_{\text{fp32 输出}}}
$$

`group=128` 时 scale 字节是权重的 `8/128 = 6.25%`，所以分母 ≈ `0.53·NK`，于是

$$
\text{AI}(M) \approx \frac{2MNK}{0.53\,NK} \approx 3.76\,M
$$

H100 上 bf16 张量核的 ridge point 是 `989e12 / 3.35e12 ≈ 295 FLOP/Byte`。代入：

$$
\text{AI}(M) \ge 295 \iff M \gtrsim 78
$$

**也就是说，只有当 `M ≳ 78` 时，这个算子才轮到张量核的性能说话；`M=1` 时 AI≈3.8，离
ridge 差 78×，是一个彻头彻尾的带宽问题。**

| M | AI ≈ 3.76M | 相对 bf16 ridge(295) | 谁是瓶颈 |
|---|---|---|---|
| 1 | 3.8 | 1/78 | 纯权重带宽 |
| 16 | 60 | 1/4.9 | 带宽 |
| 64 | 241 | 1/1.2 | 带宽/算力交界 |
| 128 | 481 | 1.6× | 算力 |

而 40–43 的 kernel 用 `BM=64` 的 `wgmma` tile。M=1 时它把 1 行 pad 成 64 行，白算 63/64
的 MAC，然后把「读一遍 44.6 MB 权重」这件事拖成了 0.0788 ms（565 GB/s）。
**这不是 GEMM 没优化好，是问题根本不该用 GEMM 解。**

对 M=1，正确的形状是 GEMV：

```
        K=5120
      ┌─────────┐
 x[1] │ a a a … │  bf16, 10 KB, 常驻 smem（所有输出行共享）
      └─────────┘
          ×
      ┌─────────┐
W[N]  │ 4 4 4 … │  int4 packed = N×K/2 = 44.6 MB，必须全部流一遍
 N=   │         │  每 128 个 k 一个 fp32 scale（2.8 MB）
17408 └─────────┘
          ↓ warp 内 shuffle 归约
      y[N] = float[N]  (70 KB)
```

---

## 二、GEMV 的设计

一个 block = 8 个 warp，每个 warp 负责 `RWW` 个输出行；`x`（`MT×K` bf16）与整块要用的
scale 预取进 smem，block 内复用。核心内层（`44-w4a16-gemv/w4a16_gemv.cu:98` 的
`gemv_kernel<RWW,MT,KK>`）：

```
for i in 0..NCHUNK-1:          // 每 lane 负责 K 维的 5 个 32-k chunk
  c  = lane + 32*i             // W 的 16B chunk，warp 内连续 → 全局合并读
  k0 = c*32
  xu = x[k0 .. k0+32]          // 4 个 uint4，一次读好给 RWW 行复用
  for r in 0..RWW-1:           // 同一份 x 喂 RWW 个输出行
    wv = W[row_r, c]           // __ldcs 16B（streaming，别污染 L2）
    展开 16 个 byte → 32 个 int4 → float，与 xu 做 FMA
  acc[r] += s * part           // s 来自预取的 scale
warp_sum(acc[r])               // 5 步 shfl_xor，lane0 写 C[row]
```

几个关键决策：

1. **每 warp 一行、K 维完全并行**：`K/2/16 = 160` 个 16B chunk，32 个 lane 每个拿 5 个
   （`c = lane + 32*i`）。同一 chunk 步内，相邻 lane 读相邻 16B → 一个 warp 一次搬
   512B 连续，完全合并。
2. **`x` 预取进 smem**：`x` 只有 10 KB，一个 block 的 8×RWW 个输出行共享一次读取；
   相对 44.6 MB 权重，`x` 的全局流量（`grid × 10 KB`）可以忽略。
3. **scale 预取进 smem**：每行 40 个 fp32，block 内 `BN×40` 个；内层直接命中 smem，
   不再走 global/L2。
4. **`__ldcs` 流式读权重**：权重只读一次、永不复用，用 `ld.global.cs` 让它 evict-first。
   实测把 `gemv_r2` 从 0.0297 ms 压到 **0.0284 ms（−4.4%）**。
5. **`RWW`（每 warp 行数）是唯一的调参旋钮**：`RWW` 越大，一份 `x` 喂的行越多、x 的
   smem 读被摊得越薄；但寄存器/ILP 也会变化。实测 **RWW=2 最优**（见下表）。

---

## 三、实测结果

环境：H100 80GB HBM3，峰值 3352 GB/s，`group=128` 对称 int4，`M=1, N=17408, K=5120`。
运行 `ARCH="" scripts/run.sh 44-w4a16-gemv/w4a16_gemv.cu 1 17408 5120 all`。

```
W4A16 GEMV: M=1 N=17408 K=5120 group=128
FLOPs = 0.178 GFLOP ; W_int4 = 44.6 MB ; scales = 2.8 MB ; W+scale = 47.3 MB

naive                     0.4583 ms      W-bw    97.2 GB/s   W+s   103.3 ( 3.1% HBM)
roof_g528                 0.0187 ms      W-bw  2380.0 GB/s   W+s  2528.8 (75.4% HBM)  ← 纯读上界
gemv_r1                   0.0310 ms      W-bw  1437.1 GB/s   W+s  1526.9 (45.5% HBM)
gemv_r2                   0.0284 ms      W-bw  1566.6 GB/s   W+s  1664.5 (49.7% HBM)  ← 最佳
gemv_r4                   0.0312 ms      W-bw  1428.8 GB/s   W+s  1518.1 (45.3% HBM)
gemv_r8                   0.0340 ms      W-bw  1310.3 GB/s   W+s  1392.1 (41.5% HBM)
```

| 版本 | 时间 | 有效权重带宽 | HBM 占比 | 相对 naive |
|---|---|---|---|---|
| v0 naive（1 thread/行，标量逐 k） | 0.4583 ms | 97.2 GB/s | 3.1% | 1× |
| v1 warp/行（RWW=1） | 0.0310 ms | 1437 GB/s | 45.5% | **14.8×** |
| **v2 warp/行（RWW=2，`__ldcs`）** | **0.0284 ms** | **1566.6 GB/s** | **49.7%** | **16.1×** |
| v3 RWW=4 | 0.0312 ms | 1429 GB/s | 45.3% | 14.7× |
| v4 RWW=8 | 0.0340 ms | 1310 GB/s | 41.5% | 13.5× |
| 纯读上界（只读 W+scale，无计算） | 0.0187 ms | 2380 GB/s | 75.4% | 24.5× |
| **43 篇 tensor-core GEMM M=1** | 0.0798 ms | 558.7 GB/s | 16.7% | 5.7× |

**最佳 0.0284 ms，相对 43 篇的 tensor-core 最好成绩（0.0798 ms）快 2.81×。**
正确性：内置 CPU 参考（对量化后的 `W_deq` 做 fp64 点积）抽样 32 点，
`max_abs_err=1.66e-7`（ref~2.79），相对误差 6e-8——因为我们和参考都走同一套量化权重，
这是纯粹的数值累加顺序差异。

---

## 四、优化过程里的负结果（都实测过）

这一节可能是本篇最有用的部分——**N=1 的带宽 GEMV 只有那几个旋钮，但很多「看起来对」的
优化在这个工具链上是负的**。

### 4.1 smem bank conflict：修了反而更慢

`x` 在 smem 里按 `[k]` 连续排，lane 读 `xs[k0..k0+32]`，`k0 = c*32 = (lane+32i)*32`
→ 所有 lane 的起始地址都是 128B 的倍数 → ncu 报 **210 万次 bank conflict（70% 的
shared wavefront 是多余的）**，并给出「Est. Speedup 59%」。

我按经典办法做了一版**免冲突置换布局**（把每个 32-k 块的第 q 个 16B 放到
`slot = (c&31) + 32*((c>>5)*4+q)`），冲突从 210 万降到 5384——结果 **0.0313 ms，
比不改还慢 6%**：

| 版本 | shared conflicts | 时间 |
|---|---|---|
| 连续布局（有冲突） | 2.1 M（70% 多余） | **0.0297 ms** |
| 置换布局（无冲突） | 5.4 K | 0.0313 ms |

根因：这个 kernel 的瓶颈在 **ALU 发射**，不在 LSU；把 4 个连续的 16B 读拆成 4 个相隔
512B 的读，反而牺牲了访存局部性。**「ncu 说 conflict 多」不等于「conflict 是瓶颈」**，
要看是哪一级 pipe 到顶（见第五节：ALU pipe 65%）。

### 4.2 smem LUT 反量化：慢 60%

int4→float 的 `(nibble)-8` + `I2F` 指令很多，很自然地想用一张 16 项（或 256 项）的 smem
LUT 替掉。实测**全部更慢**：

| LUT | 时间 |
|---|---|
| 无 LUT（I2F + FADD） | **0.0284 ms** |
| 16 项 nibble LUT | 0.0325 ms |
| 256 项 `float2` byte LUT | 0.0477 ms |

`float2` 版慢一倍：256 项随机 8B 读，每个 lane 命中两个 bank，warp 内冲突严重，把本来
空闲的 MIO/LSU 打爆。**用 LUT 把 ALU 换算成 smem 读，只在 smem 访问本身不冲突、且 LSU
有大量余量时才划算。**

### 4.3 `__int_as_float` 位技巧：数值上不可行

「用 `__int_as_float(0x4B000000 | nib) - 2^23` 造 float，把 `I2F` 挪到 FMA pipe」是个
常见技巧。但 `0x4B000000` 正是 2^23，凑出来的 `part` 里每项都被放大到 ~2e6，32 项累加
到 ~7e7，f32 的 ulp 已经 ~8，而真正的答案只有个位数——**灾难性抵消**，实测相对误差
15%。换成小 bias（16）又因为 f32 尾数 LSB 是 `16×2^-23` 而不是 1，根本不等于 `16+nib`。
这条路在「权重和激活量纲差很大」的 GEMV 上走不通。

### 4.4 `Σ(q-8)x = Σqx - 8Σx` 的代数化简：也没用

把 `-8` 折成每 chunk 一次 `−8·Σx`（`Σx` 与输出行无关，可以共享），理论上省掉每元素一个
`FADD`。实测 0.0315 ms，比不折的 0.0297 还慢——因为多出来「沿 x 做 32 次串行加法求 `Σx`」
的依赖链，反而拖慢了内层。**小 kernel 里，代数化简省下的算术常常抵不过新增依赖的代价。**

---

## 五、ncu：确实是 ALU/发射受限，不是带宽

对最佳版 `gemv_r2` 采 ncu（`scripts/ncu.sh ... --kernel-name regex:gemv_kernel -- 1 17408 5120 g2`）：

| 指标 | 值 |
|---|---|
| Duration | 31.0 µs（ncu 单次 replay，event 稳态 28.4 µs） |
| **Compute (SM) Throughput** | **57.8%** |
| DRAM Throughput | 47.7% |
| L1/TEX / L2 | 53.3% / 44.7% |
| Achieved Occupancy | 54.8% |
| **`sm__pipe_alu`** | **64.9%** |
| `sm__pipe_fma` | 27.1% |
| `smsp__issue_active` | 65.0% |

反量化把每个 int4 变成 float 需要「移位 + 掩码 + `I2F`」三串 ALU 指令，**ALU pipe 64.9%
是整个 kernel 的最高占用**。这也解释了 4.1：修 smem 冲突没用，因为 LSU 根本不忙；以及
4.2：把 ALU 换成 smem 读只在 LUT 命中不冲突时才成立。

**能继续做的方向**是减少每元素的整数指令数（例如把 4 个 lo-nibble 用一次
`LOP3` 批量抠出、或走 W4A8 + `dp4a` 整数点积），但这会把 kernel 从「忠实 bf16 激活」
变成「激活也量化」，属于另一篇的题目。

---

## 六、小 M 扫：GEMV 的收益边界

让一个 block 的 CTA 同时处理 `MT` 个 token（一次读 W、`MT` 次 FMA，`x` 换成
`MT×K`），看 GEMV 能撑到多大的 M：

| 配置 | 时间 | 权重带宽 | 单 token 时间 |
|---|---|---|---|
| M=1, RWW=2 | 0.0284 ms | 1566.6 GB/s | 0.0284 ms |
| M=2, MT=2, RWW=4 | 0.0352 ms | 1264.6 GB/s | 0.0176 ms |
| M=4, MT=4, RWW=4 | 0.0593 ms | 751.9 GB/s | 0.0148 ms |

`MT` 让权重被多个 token 复用，**单 token 成本确实在降**（0.0284 → 0.0176 → 0.0148），
但绝对时间在涨——因为 M 越大，FMA/ALU 工作量按 M 线性增长，而权重字节不变，kernel 逐渐
从带宽受限滑向 ALU 受限。**这正是第一节那张 AI 表说的**：`M` 一大，就该换回张量核
GEMM（40/41/43 的路径），用 `BM=64` 把同一份权重摊到 64 行上。于是得到一个干净的
分工：

```
decode 单 token（M=1）      → GEMV，贴权重带宽，2.8× 于 GEMM
小 M（2 ≤ M ≤ ~16）        → GEMV(MT) 或小 BM 的 GEMM，看 M
M ≳ 64                    → 张量核 GEMM（m64 tile 才不浪费）
```

---

## 七、对标 SOTA

- **40 篇**给过 W4A16 融合 GEMM 的「有效权重带宽 400 GB/s」，并据此说与 Marlin 类实现
  差 **≈6.7×**。本篇 GEMV 把有效权重带宽做到 **1566.6 GB/s（3.9× 于 400）**，剩下与
  「纯读上界（2380 GB/s）」的差距是 **1.52×**——也就是说，M=1 W4A16 的性能天花板已经从
  「差一个数量级」收窄到「差 1.5× 的纯带宽」。
- **2.81× vs 43 篇 tensor-core M=1**（同一 shape、同一进程内两版 binary 对拍）。
- **0.0284 ms = 49.7% HBM**；纯读上界 0.0187 ms = 75.4% HBM。剩下的 gap 全部在
  反量化的 ALU 发射上（ncu ALU pipe 65%）。

一个诚实的边界：Marlin 等 4-bit GEMM 库面向的是 **batched decode**（M 到几十），它们
用张量核 + 精细的分组调度，在 M=64 上仍然比「GEMV 打 64 遍」快得多。本篇不是说
「GEMV 取代 W4A16 GEMM」，而是说：**在 M=1 这个张量核最不划算的工作点上，回到带宽
本位的 GEMV，反而比硬上 m64 tile 快近 3 倍**——算子的形状要跟着 work point 走。

---

## 八、小结

- M=1 的 W4A16 上投影的 `AI ≈ 3.76M`，要到 `M ≳ 78` 才轮到张量核说话；40–43 的
  `m64` tile 在 M=1 上 63/64 的行在空转，kernel 变成延迟受限（DRAM 仅 18%）。
- 换「每 warp 一行、warp 内沿 K 并行 + `shfl` 归约、`x`/scale 常驻 smem、16B 向量读
  int4、`__ldcs` 流式权重」的 GEMV，实测 **0.0284 ms / 1566.6 GB/s / 49.7% HBM**，
  相对 43 篇 tensor-core 最好 **2.81×**，相对朴素标量版 **16.1×**。
- `RWW=2` 是最佳（一份 x 喂两行）；再大反而掉。
- **负结果**：修 smem bank conflict（−6%）、smem LUT 反量化（−14%~−40%）、
  `__int_as_float` 位技巧（数值不可行）、`Σ(q-8)x` 代数化简（−6%）——kernel 是
  **ALU 发射受限**（ALU pipe 65%），不是访存受限，别按「访存 kernel」的直觉调。
- M=2/4 用 `MT` 让权重复用，单 token 成本降到 0.0148 ms，但绝对时间在涨 → 大 M 该
  换回张量核 GEMM。

**下一篇预告**：把「纯读上界」和 GEMV 之间的 1.5× 差距补上——用 W4A8 + `dp4a` 整数
点积把反量化的 ALU 指令数砍掉一个数量级，看能不能把 M=1 顶到 60%+ HBM；或者反过来，
给 `M ∈ [2,16]` 做「GEMV ↔ GEMM 运行时自适应」的统一调度。

代码：`code/kernel-opt/44-w4a16-gemv/w4a16_gemv.cu`，原始输出
`gemv_M1.out.txt` / `gemv_M2_mt2.out.txt` / `gemv_M4_mt4.out.txt` / `ncu_gemv_r2.out.txt`。
