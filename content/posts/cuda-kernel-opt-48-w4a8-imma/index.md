---
title: "CUDA 算子调优（四十八）：W4A8 的整数张量核（IMMA）——把反量化的 ALU 砍掉一半，小 M 权重带宽翻倍"
date: 2026-09-22T04:00:00+08:00
draft: false
weight: 48
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A8", "int4", "int8", "IMMA", "mma.sync", "tensor-core", "PRMT", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第八篇。承接
[第 40 篇 W4A16 dequant-GEMM]({{< relref "cuda-kernel-opt-40-w4a16-gemm" >}})、
[第 44 篇 M=1 GEMV]({{< relref "cuda-kernel-opt-44-w4a16-decode-gemv" >}})、
[第 45 篇 W4A8 + `dp4a`]({{< relref "cuda-kernel-opt-45-w4a8-dp4a-gemv" >}})、
[第 46 篇小 M 分派]({{< relref "cuda-kernel-opt-46-w4a16-smallm-dispatch" >}}) 和
[第 47 篇小 M 的 warp specialization]({{< relref "cuda-kernel-opt-47-w4a16-mma-ws" >}})。

47 篇留下了一个非常明确的判决：小 M（`M∈[3,16]`）的 `mma_b16` 变体之所以慢，
不是张量核不够快，而是**同一批线程把绝大多数时间花在「把 int4 反量化成 bf16」的
ALU 上**——ncu 数出来反量化的 ALU+FMA 指令占了 **78%**，张量核只占 2.3%。把这段
活搬到另一个 warpgroup 去「重叠」是无效的（47 篇的核心负结果），因为 WS 只能重叠、
不能减少工作。

那么唯一的路就是**换数制**：既然激活也可以量化，为什么还要把 int4 权重先变成
bf16 再乘？直接用 **int8 的整数张量核（IMMA，`mma.sync.m16n8k32.s32`）**做整数
点积不行吗？这样：

- 一条 `mma.m16n8k32` 顶两条 `m16n8k16`（k 翻倍），张量核指令数减半；
- int4→int8 的展开用 `PRMT`（`__byte_perm`）一次处理 8 个 nibble，
  比「逐 nibble 取数 + `I2F` + 乘 scale + `F2BF`」便宜得多；
- int8 的 mma 片段**每 lane 恰好一个 32-bit word**，连 `ldmatrix` 都不需要。

结论先给：**这一换让小 M 的有效权重带宽从 650 GB/s 直接翻到 ~1290 GB/s
（19.4% → 37.5% HBM），相对 bf16 基线最高 1.92×，全 M 段总和 1.80×**。
而且这一步还顺手把数值误差从 bf16 的 3.6e-3 降到 **1e-7**（整数运算是精确的）。

---

## 一、先看墙：bf16 反量化到底贵在哪

shape 仍取自 `/ssd/models/qwen3-8B/config.json`：`hidden=5120,
intermediate=17408, group_size=128`。默认 `N=17408`（MLP up/gate）、`K=5120`，
权重 int4 打包成 `[N, K/2]` u8 + `[N, K/128]` fp32 scale，共 **47.3 MB**
（44.6 MB 权重 + 2.8 MB scale）。激活也按真实部署量化成 **int8（per-128 动态）**。

用 ncu 把 47 篇的 `mma_b16_k8`（M=16）和即将写的 IMMA 版放在一起看指令数：

| kernel | 执行指令数 @M=16 | ALU pipe | FMA pipe | LSU pipe | 主导 stall |
|---|---|---|---|---|---|
| bf16 `mma_b16_k8`（47 篇） | **30.60 M** | 43.0% | 14.8% | 13.6% | `long_scoreboard` 1.58 |
| **W4A8 IMMA（本篇）** | **12.42 M** | 34.1% | 11.5% | 20.0% | `long_scoreboard` 2.74 |

**指令数少了 2.46×**。这就是全部故事的起点：bf16 路径每两个权重元素要做
「取 nibble → `I2F` → 乘 scale → `F2BF`」约 5 条 ALU，IMMA 路径每个 nibble 只要
「掩码/移位 + `PRMT`」约 1 条。指令少了，反量化的墙塌了，瓶颈自然往外挪到访存。

---

## 二、IMMA 的片段布局：每 lane 就是一个 32-bit word

写整数 mma 之前，先把 `mma.sync.m16n8k32`（A/B 都是 8-bit）的片段布局钉死。
PTX 的定义（`g = lane>>2`，`t = lane&3`）：

```
A (m16 × k32)：每 lane 4 个 .b32（= 16 个 int8）
   a0 = A[row g,   col 4t .. 4t+3]      a2 = A[row g,   col 16+4t ..]
   a1 = A[row g+8, col 4t .. 4t+3]      a3 = A[row g+8, col 16+4t ..]

B (k32 × n8)：每 lane 2 个 .b32（= 8 个 int8），B 存成 [n][k] 行主序
   b0 = B[n g, k 4t .. 4t+3]            b1 = B[n g, k 16+4t ..]

C (m16 × n8)：4 × .s32
   c0/c1 = (row g,   col 2t / 2t+1)
   c2/c3 = (row g+8, col 2t / 2t+1)
```

关键在于：**4 个连续 k 的 int8 恰好是一个 32-bit word**。所以只要把 A 存成
`[m][k]`、B 存成 `[n][k]` 的行主序，每 lane 的片段就是**一次 32-bit smem 读取**，
无需 `ldmatrix`、无需 `.trans`：

```cpp
// A: 一次 word 读 (row, k=c*32+4t)
a[0] = As32[(arow0)*ASP4 + c*8 + t];
a[2] = As32[(arow0)*ASP4 + c*8 + 4 + t];
// B: 一次 word 读 (n, k=c*32+4t)
b[0] = Bs32[n*BSP4 + c*8 + t];
```

为了万无一失，先写一个 `imma_test.cu` 最小验证：16×32×8 的单条 mma，CPU 对拍
`C = A·B`。实测 **0/128 个元素不匹配**（`code/kernel-opt/48-w4a8-imma/imma_test.cu`）。
布局一旦钉死，后面的完整 kernel 就只是把片段拼起来。

---

## 三、kernel 结构：一个 stage 就是一个 scale group

W4A8 的难点在 **per-128 group scale**：整数 mma 只能在同一个 scale 下累加，而
group 是 128 个 k。最省事的做法是让 **`BK = 128` 恰好等于 `GROUP`**，于是
**一个流水 stage 就是一个 group**，折算频率和 group 一一对应，不需要跨 stage 记账。

```
每个 (m-tile, n-tile) CTA：
  As[STAGES][BM][144]   int8 激活（cp.async）
  Wraw[STAGES][BN][80]  int4 权重（cp.async，现场展开）

  prologue: 预取 STAGES-1 个 stage
  for t in 0..nblk-1:                       # nblk = K/128 = 40（KSPLIT 再切）
    预取 stage t+STAGES-1；__pipeline_wait_prior(STAGES-1)
    for c in 0..3:                          # 4 个 k32 chunk
      A 片段读一次（4 word）
      for j in 0..MTN-1:                    # 4 个 n8 tile
        b0 = expand4u(W16[n*40 + c*8 + t])  # 现场从 int4 展开 4 个 int8
        b1 = expand4u(W16[n*40 + c*8 + 4+t])
        imma(acci[j], a, {b0,b1})
    # 折算（本 group 结束）
    for j: facc[j][q] += sa[m,g]·sw[n,g]·(acci[j][q] − 8·Σq_a[m,g])
           acci[j][q] = 0
```

片段与 smem 地址的对应见 `w4a8_imma.cu:403`（内层循环）、`:412`（B 的 word 偏移
`n*WPAD2 + c*8 + aqw`）、`:416`（整数 mma）。B 的行距 `WPAD=80`（= 64 字节 int4 +
16 字节 padding）是精心选的：`g*20 mod 32` 对 8 个 `n` 落在 8 组互不重叠的 bank
上，实测 `smem` bank conflict 基本归零。

### 3.1 `expand4`：用 `PRMT` 一次摊出 4 个 int8

权重在显存里是每 2 个 int4 挤一个 byte。要把「4 个连续 k」变成 mma 想要的
「4 个 int8 挤一个 word」，只需要两条指令：

```cpp
__device__ uint32_t expand4u(uint32_t v) {           // v = 2 个 int4 字节
  const uint32_t lo = v & 0x0F0F;                    // 低 nibble: [n0, n2]
  const uint32_t hi = (v >> 4) & 0x0F0F;             // 高 nibble: [n1, n3]
  return __byte_perm(lo, hi, 0x5140);                // 交错成 [n0 n1 n2 n3]
}
```

读的时候直接从 int4 的 smem 里按 **uint16** 取，不再走「uint32 + 移位 + 选择半字」：

```cpp
const uint16_t* W16 = ...;                            // int4 的 uint16 视图
const int wo = n * WPAD2 + c * 8 + aqw;              // aqw = lane&3
const uint32_t b0 = expand4u(W16[wo]);
```

这样每个 B 片段只有 **1 次 LDS + 1 次 `LOP3` + 1 次 `SHF` + 1 次 `PRMT`**。

### 3.2 `u8.s8`：把 `-8` 从展开里拿掉

nibble 存的是 offset-binary（`u = q + 8 ∈ [0,15]`）。要变成有符号 int4 的
`q = u - 8`，最直觉的是 `__vsubss4(r, 0x08080808)`——但 45 篇已经实测过，这个
「逐 byte SIMD 减」在 sm_90 上被 ptxas 展开成一大堆整数指令。既然我们要的只是
点积，就干脆**别减**，让 mma 算 `Σ u·q_a`，真值再解析地扣掉：

$$
\sum_k q_w q_a = \sum_k (u_w - 8)\,q_a = \underbrace{\sum_k u_w q_a}_{\text{整数 mma}} \;-\; 8\sum_k q_a
$$

`Σ q_a` 只跟激活有关、**与输出列无关**，在量化 kernel 里顺带算出来即可（每
`(m, group)` 一个标量）。这样 mma 的 B 操作数就用 **无符号** 的 `u8`，
A 仍然是 `s8`：

```cpp
asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.u8.s32 ...");  // A=s8, B=u8
```

注意类型序是 `.s8.u8`——**A 在左、B 在右**。第一次我写反成 `.u8.s8`，结果
激活被当成无符号、权重被当成有符号，误差直接 116×（`rel=1.16e+2`），
而且因为指令本身合法、不报错，只能靠数值对拍抓到。

---

## 四、实测：小 M 权重带宽翻倍

运行环境同前作：kernel_lab 容器、H100 80GB HBM3（132 SM，HBM 3.35 TB/s）。
`warmup=5, iters=50`，同进程 head-to-head（原始输出
`48-w4a8-imma/sweep_all.out.txt`）。

### 4.1 主表：bf16 vs IMMA（同进程，单位 ms）

| kernel \ M | 1 | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|---|---|---|---|---|---|---|---|---|
| bf16 `mma_b16_k16` | 0.0632 | 0.0645 | 0.0666 | 0.0722 | 0.1196 | 0.1254 | 0.1745 | 0.2279 |
| bf16 `mma_b32_k8` | 0.0747 | 0.0752 | 0.0760 | 0.0785 | 0.0822 | 0.0852 | 0.1390 | 0.1455 |
| IMMA `imma_ns_s2k8` | 0.0331 | 0.0335 | 0.0343 | 0.0378 | 0.0558 | 0.0597 | 0.0809 | 0.1030 |
| IMMA `imma_ns_s2k5` | 0.0355 | 0.0360 | 0.0371 | 0.0398 | **0.0521** | **0.0548** | **0.0735** | 0.0942 |
| IMMA `imma_ns_s2k4` | — | — | — | — | 0.0535 | 0.0560 | 0.0736 | **0.0926** |

每个 M 点的最优（`imma_ns_s2k7` 拿下 `M≤8`，其余见上）：

| M | 最优 kernel | ms | 有效权重带宽 | %HBM | bf16 最优 | 加速 |
|---|---|---|---|---|---|---|
| 1 | `imma_ns_s2k7` | **0.0320** | 1483 GB/s | 44.2% | 0.0632 | **1.98×** |
| 8 | `imma_ns_s2k7` | **0.0341** | 1405 GB/s | 41.9% | 0.0666 | **1.95×** |
| 16 | `imma_ns_s2k7` | **0.0376** | 1290 GB/s | 38.5% | 0.0722 | **1.92×** |
| 24 | `imma_ns_s2k5` | **0.0521** | 1852 GB/s | 55.2% | 0.0822 | 1.58× |
| 32 | `imma_ns_s2k5` | **0.0548** | 1772 GB/s | 52.9% | 0.0852 | 1.55× |
| 48 | `imma_ns_s2k5` | **0.0735** | 1982 GB/s | 59.1% | 0.1390 | 1.89× |
| 64 | `imma_ns_s2k4` | **0.0926** | 2097 GB/s | 62.6% | 0.1455 | 1.57× |

12 个 M 点各取各自最优之和：**always-bf16 0.9804 ms、always-IMMA 0.5439 ms
（1.80×）**。单点最高 **1.92×**（M=16）。M=16 的有效 int4 权重带宽从 47 篇的
**650 GB/s（19.4% HBM）** 抬到 **1290 GB/s（38.5%）**——同一块 H100、同一份权重。

> 注：M=1 的绝对冠军仍是 45 篇的 W4A8 `dp4a` GEMV（0.0185 ms / 76.3% HBM），
> 因为 IMMA 的 `m16` tile 在 M=1 白算 15/16 行。IMMA 的战场是 **M≥2**：
> 46 篇的 GEMV 在 M=2 要读两遍权重（≈0.037 ms），IMMA 一遍覆盖 16 行只要 0.032 ms。

### 4.2 消融：两招各值多少（M=16）

| 版本 | scale 表 | 展开 | M=16 (ms) | 相对上一版 |
|---|---|---|---|---|
| bf16 `mma_b16_k16` | — | int4→bf16（`I2F`+`F2BF`） | 0.0722 | — |
| IMMA `sm_s8` | smem 23KB 表 | `vsubss4` 带 `-8` | 0.0467 | 1.55× |
| IMMA `s8` | **global（`__ldg`）** | `vsubss4` 带 `-8` | 0.0418 | **1.12×** |
| IMMA `s2k8` | smem 23KB 表 | **`u8.s8`** 免 `-8` | 0.0435 | 1.07× |
| IMMA `ns_s2k8` | **global** | **`u8.s8`** | **0.0378** | **1.11×** |

两招（去掉 23 KB 的 smem scale 表、`u8.s8` 免符号展开）各约 **+11%**，叠加后
0.0467 → 0.0378（**1.24×**）。

一个反直觉的现象：去掉 smem scale 表后寄存器**升高**了（92 → 101），occupancy
反而从 31% 掉到 23%，但就是更快。原因是那张表要在流水启动前串行地搬 23 KB
（`(BM+BN)·NGRP·4`），而且每个 block 都要搬一遍；改成按需从 global 读 scale
（`Ws` 才 2.8 MB，命中 L1/L2）反而把 prologue 省掉了。**「省 smem 抬
occupancy」不是这里的原因，别照抄结论。**

---

## 五、根因：墙从 ALU 挪到了访存延迟

对最佳几何 `imma_ns_s2k8` @M=16 做 ncu（`ncu_imma_ns_s2k8_M16.out.txt`）：

| 指标 | bf16 `mma_b16_k8` | W4A8 IMMA | 说明 |
|---|---|---|---|
| 执行指令数 | 30.60 M | **12.42 M** | −59% |
| ALU pipe | 43.0% | 34.1% | 反量化 ALU 让位 |
| DRAM Throughput | 22.6% | **42.1%** | HBM 利用率翻倍 |
| L2 Cache Throughput | — | 66.8% | 权重 + strided scale 读 |
| Compute (SM) | 47.7% | 38.1% | 不再是计算受限 |
| Achieved Occupancy | 24% | 23.0% | 101 regs，4 CTA/SM |
| Waves / SM | 2.06 | 2.06 | KSPLIT 对齐 |
| `No Eligible` | — | **56.8%** | 发射口常空——在等访存 |

stall 分解（per issue active）：**`long_scoreboard` 2.74 一骑绝尘**
（bf16 只有 1.58），后面是 `wait` 1.16、`barrier` 0.75、`not_selected` 0.75、
`math_pipe_throttle` 0.49。也就是说，把 ALU 砍掉之后，kernel 变成**访存延迟受限**：
只有 3.6 个 active warp/scheduler、0.76 个 eligible，`cp.async` 的 2 级流水填不满
权重读的延迟。

### 5.1 负结果：强压寄存器换 occupancy 没用

`long_scoreboard` 高、warp 又少，最自然的反应是把 occupancy 顶上去。
给 kernel 加 `__launch_bounds__(128, 7)` 逼 ptxas 把寄存器从 101 压到 72
——结果是 **spill 40 字节栈**，M=16 从 0.0384 掉到 **0.0517（慢 35%）**。
`__launch_bounds__(128,8)` 更惨。结论和 11/44 篇一致：**访存延迟不是靠堆
occupancy 能填的，压寄存器只会换来 spill**；真要填延迟得让数据流本身更早发出，
而不是多塞几个 warp。

---

## 六、正确性：整数运算是精确的

因为 IMMA 是整数点积、且我们的参考也用同一套量化，误差应当只有「两个 int32
求和」的精确性。实测抽样 48 点（`M=64`）：

| kernel | max_abs_err | 相对误差 |
|---|---|---|
| bf16 `mma_b16_k16` | 9.28e-3 | 3.61e-3（int4→bf16 丢精度） |
| **W4A8 IMMA** | **2.6e-7** | **1.01e-7**（精确） |

整条链里唯一的近似是**激活的 int8 per-128 动态量化**本身（相对误差 ~0.4%，
与 45 篇一致），这已经计入参考；GEMM 本体不再引入任何新误差。顺带一提，
量化对拍一开始有 3% 的 int8 元素不一致，排查出来只是 CPU 用 `std::lround`
（四舍五入远离零）、GPU 用 `__float2int_rn`（round-to-nearest-even）的舍入差，
把 CPU 侧换成 `std::nearbyint` + 同一套 `1/s` 就归零了——**低比特量化的对拍要连
舍入模式一起对齐**。

---

## 七、对标与定位

- **vs 同 shape 稠密 cuBLAS**：这里 N=17408、K=5120、M=16——cuBLAS 没有
  W4A16/W4A8 的融合入口（`torch._scaled_mm` 只支持 per-tensor/per-row），
  所以只能给「读 bf16 权重」的上界：权重 bf16 是 89 MB，要按 cuBLAS bf16
  的 ~2.6 TB/s 读，纯读下限就 ~0.034 ms，**还没算任何计算**。我们读的是 47 MB
  int4+scale、0.0376 ms，已经贴近这个「读更多字节」的 cuBLAS 下限本身。
- **vs Marlin 类 SOTA（AWQ/GPTQ）**：40 篇给出的有效权重带宽对比里，融合
  bf16 dequant 只有 400 GB/s（BF16 反量化税 46%）。本篇 IMMA 的 **1290 GB/s**
  相对它是 **3.2×**，把与 Marlin 类实现的差距从 ~6.7× 收窄到 **~2.1×**。
- **vs 纯读上界**（47 篇 `roof_g1056`）：读 47.3 MB 的物理下限是 2568 GB/s，
  我们 1290 GB/s 是它的 **50%**——剩下的一半就是 `long_scoreboard` 那 2.74。

---

## 八、该不该上 IMMA？一条判据

```
小 M 低比特 decode 的数制选择：

  数据在显存里的字节数已定（int4）
        │
        ├─ 张量核指令数 / 反量化 ALU 是不是墙？
        │     ncu: ALU+FMA > 50% 指令数、tensor pipe < 5%
        │        → 是：换 IMMA（m16n8k32，整数点积）
        │              激活量化 int8 per-128；权重 nibble 直接铺 u8
        │        → 否：保持 bf16/fp16 路径
        │
        └─ 换完如果 DRAM% 上去了、long_scoreboard 成第一 stall
              → 别再压寄存器抬 occupancy（会 spill）
                该做的是加深流水 / 消 strided scale 读 / 持久化
```

IMMA 给你的是「**用整数换掉浮点反量化**」，上限是**把权重带宽推到纯读 roofline**；
它不解决访存延迟，那要交给 43 篇的持久化和更深的 TMA 流水。

---

## 九、小结

- **换数制是砍反量化 ALU 的正解**：bf16 路径 30.6 M 指令 → W4A8 IMMA 12.4 M
  （−59%）；ALU pipe 43%→34%，DRAM 22.6%→42.1%。
- **小 M 权重带宽 650 → 1290 GB/s（19.4% → 37.5% HBM）**；相对 bf16 基线
  单点最高 **1.92×**（M=16）、全 M 段 **1.80×**。
- **int8 mma 片段每 lane 一个 word**：A 存 `[m][k]`、B 存 `[n][k]`，每片段一次
  32-bit smem 读，**不需要 `ldmatrix`**；布局先用 `imma_test.cu` 钉死再写 GEMM。
- **两个各 ~11% 的 ALU 小招**：① 权重按 uint16 直读、`PRMT` 一次摊 4 个 int8；
  ② `u8.s8` 省掉逐 byte 的 `-8`，解析扣 `8·Σq_a`。类型序是 `.s8.u8`（A 在左），
  写反不报错、只能靠对拍抓。
- **去掉 smem scale 表反而更快**（+12%），尽管寄存器/occupancy 变差——省的是
  每 block 的串行 prologue，不是 occupancy。
- **负结果**：`__launch_bounds__(128,7)` 强压寄存器 → 40 B spill、慢 35%；
  `long_scoreboard` 不能靠堆 occupancy 填。
- **数值**：整数 GEMM 精确，相对误差 1e-7（bf16 是 3.6e-3）；唯一的近似是
  激活 int8 量化本身。

**下一篇预告**：候选是 49（MLA decode 的 producer/consumer warp specialization，
42 篇遗留的 `wait` + 单 CTA/SM 问题），或把本篇的 IMMA 与 43 篇的跨 item 持久化
流水叠加、继续压 `long_scoreboard`。代码见 `code/kernel-opt/48-w4a8-imma/`。
