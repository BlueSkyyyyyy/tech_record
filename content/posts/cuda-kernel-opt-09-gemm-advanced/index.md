---
title: "CUDA 算子调优（九）：GEMM 进阶 — 寄存器分块、向量化与 double buffering"
date: 2026-09-21
draft: false
weight: 9
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "GEMM", "register tiling", "cp.async", "double buffering", "ncu", "系列"]
categories: ["算子开发"]
---

上一篇把 GEMM 从朴素三重循环（8.2%）用 shared-memory tiling 推到了 **13.6%**，`ncu` 告诉我们瓶颈仍是 **`L1/TEX` 载入管道（91%）**——因为内层每次 `FMA` 都要发两条 shared memory load。这一篇就来解决它：**寄存器分块**让一次 smem 载入被 8 次 FMA 复用，再加上**向量化**与 **cp.async 双缓冲流水线**。实测一路从 **9.11 TFLOPS（13.6%）爬到 34.46 TFLOPS（51.5%）**，而 cuBLAS 同口径是 **50.73 TFLOPS（75.8%）**，我们做到它的约 **68%**。

配套代码：[`code/kernel-opt/09-gemm-advanced/gemm_advanced.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/09-gemm-advanced/gemm_advanced.cu)。前一篇见 {{< relref "cuda-kernel-opt-08-gemm" >}}。

---

## 1. 问题在哪：一次 smem load 只喂了一次 FMA

回看第 08 篇 tiled 版本的内层循环：

```cuda
for (int k = 0; k < T; ++k)
  acc += As[ty][k] * Bs[k][tx];   // 每条 FMA 配 2 条 smem load
```

每个线程只算 **一个** 输出元素，`As`/`Bs` 的每个值从 smem 读出来只用一次就丢。`ncu` 显示 `L1/TEX` 吞吐 91%、`Compute` 只有 74.7%，就是被 load 指令堵死的。

**核心思路跟第 08 篇一模一样，只是把「复用的层级」从 block 降到了线程**：

> 第 08 篇：一次全局加载 → 被 block 内 TILE 个线程复用（存到 smem）
> 本篇：一次 smem 加载 → 被单个线程的多个 FMA 复用（存到寄存器）

具体地，让每个线程算一小片 `TM×TN = 8×8` 的输出：先把 `As` 的 8 个值、`Bs` 的 8 个值读进寄存器，然后做 **64 次 FMA**。载入:计算 从 `2:1` 变成 `16:64 = 1:4`。

```
一次 smem load 的复用倍数：
  tiled        :  1 load ──► 1 FMA      (2 load / FMA)
  register 8×8 : 16 load ──► 64 FMA     (0.25 load / FMA)  ← 8 倍
```

## 2. 版本 1：寄存器分块（register / thread tiling）

配置：block 算 `C` 的 `128×128` 分块，256 线程（16×16），每线程 `8×8`。`K` 方向按 `BK=8` 分片。

### 2.1 交错映射，顺手消掉 bank conflict

每个线程负责的 8 行、8 列怎么选？如果选连续的 `{ty*8..ty*8+7}` 和 `{tx*8..tx*8+7}`，smem 访存会踩到 bank conflict（跨线程步长 8，落在同一 bank 组上）。我们改成**交错映射**：

```cuda
// 线程 (tx,ty) 负责行 {ty + 16*i}、列 {tx + 16*j}
for (int i = 0; i < TM; ++i) a[i] = As[ty + TY * i][k];   // 同 ty 广播，TX 个线程同址
for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + TX * j];   // 同 j 下 tx 连续 → 无冲突
```

- `As[ty+16i][k]`：warp 内 16 个 `tx` 读**同一个地址**（广播），两个 `ty` 读相邻行、bank 相差 8，不冲突；
- `Bs[k][tx+16j]`：同一条指令里 `tx = 0..15` 访问**连续地址**，完美落在 16 个不同 bank。

不用 padding 就消掉了冲突。

### 2.2 内层：8+8 载入换 64 FMA

```cuda
__device__ __forceinline__ void compute_tile(float (*As)[BK], float (*Bs)[BN], float acc[TM][TN]) {
  const int tx = threadIdx.x, ty = threadIdx.y;
  #pragma unroll
  for (int k = 0; k < BK; ++k) {
    float a[TM], b[TN];
    #pragma unroll
    for (int i = 0; i < TM; ++i) a[i] = As[ty + TY * i][k];
    #pragma unroll
    for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + TX * j];
    #pragma unroll
    for (int i = 0; i < TM; ++i)
      #pragma unroll
      for (int j = 0; j < TN; ++j) acc[i][j] += a[i] * b[j];   // 64 次 FMA，全在寄存器里
  }
}
```

每个 `k` 里 `a[8]`、`b[8]` 常驻寄存器，`acc[8][8]` 一直累加。实测：

```
tiled32        1.8851 ms      9.11 TFLOPS  ( 13.6% of fp32 peak)
reg            0.6547 ms     26.24 TFLOPS  ( 39.2% of fp32 peak)
```

**2.9 倍。** `ncu` 一看瓶颈变了：

```
gemm_tiled<32> : L1/TEX 91.00%   Compute 74.68%   DRAM 0.78%
gemm_reg       : L1/TEX 39.29%   Compute 62.53%   DRAM 1.66%   ← L1 不再是瓶颈
```

`L1/TEX` 从 91% 掉到 39%，load 管道彻底松开了。但 `Compute` 只有 62.5%、时间 653 µs，离顶线还远。原因藏在 `Occupancy` 一节：

```
gemm_reg : Registers Per Thread = 128   Theoretical Occupancy = 25%   Achieved = 23.3%
```

`acc[8][8]`（64 个）+ `a[8]`/`b[8]` + 地址，单线程吃了 **128 个寄存器**，每个 SM 只能塞下 **2 个 block = 512 线程**，occupancy 只有 **25%**。计算单元经常「等不到可发射的 warp」。这正是下一篇（10 篇 ncu 深潜）要处理的问题，这里先记下。

## 3. 版本 2：向量化载入（float4）

`reg` 版把分块从全局搬进 smem 时是**逐 float 标量**载入：A 分块 1024 个 float 要发 1024 条 load。改成每个线程搬一个 `float4`：

```cuda
const int row = t / (BK / 4), q = t % (BK / 4);
float4 v = *reinterpret_cast<const float4*>(&A[(size_t)(block_row + row) * K + k0 + q * 4]);
*reinterpret_cast<float4*>(&As[row][q * 4]) = v;
```

`1024 float = 256 个 float4`，256 个线程一次搬完，load 指令数降到 **1/4**；B 分块的连续 `float4` 在全局还是 512 B 连续访问，合并得很好。

```
reg_vec        0.5707 ms     30.10 TFLOPS  ( 45.0% of fp32 peak)   ← +14.7%
```

`ncu`：`Duration 570 µs`，寄存器降到 118，`L1/TEX` 46%、`Compute` 54.75%。指令更少、跑得更快。

## 4. 版本 3：cp.async 双缓冲流水线

到目前为止每个 `k0` 都是「**先搬分块 → `__syncthreads()` → 再算**」，搬数据时计算单元在干等。用 Ampere 起的 `cp.async` 把「搬下一块」和「算当前块」重叠起来。

注意力放在**双缓冲**：准备两份 smem（`As[2][BM][BK]`、`Bs[2][BK][BN]`），一边算 buffer `b`，一边异步预取 buffer `b^1`。

```
时间轴（单 block）：
       ┌── 计算 tile k0 ──┐┌── 计算 tile k1 ──┐┌── 计算 tile k2 ──┐
       │                  ││                  ││                  │
cp.async │ 预取 k1 (async) ││ 预取 k2 (async) ││       ...        │
       └──────────────────┘└──────────────────┘└──────────────────┘
       预取在后台进行，计算不再等它 —— 只要在用到前 wait 一下
```

实现要点：

```cuda
prefetch_tile(..., As[0], Bs[0], k0 = 0);          // 预取第一块并 commit
int buf = 0;
for (int k0 = 0; k0 < K; k0 += BK) {
  const bool more = (k0 + BK < K);
  if (more) prefetch_tile(..., As[buf ^ 1], Bs[buf ^ 1], k0 + BK);  // 异步装下一块
  __pipeline_wait_prior(more ? 1 : 0);             // 等当前缓冲到齐（留最后一组在飞）
  __syncthreads();
  compute_tile(As[buf], Bs[buf], acc);             // 算当前块
  __syncthreads();
  buf ^= 1;
}
```

`__pipeline_memcpy_async(dst, src, 16)` 一次拷 16 字节（`float4`），`__pipeline_commit()` 把这一轮拷贝打包成一个 group；`__pipeline_wait_prior(1)` 表示「最多保留 1 个未完成的 group」——于是正在飞的是下一块，当前块已就绪。

```
reg_db         0.4986 ms     34.46 TFLOPS  ( 51.5% of fp32 peak)   ← +14.5%
```

`ncu`：`Duration 499 µs`，`Compute (SM) 66.43%`，`Warp Cycles Per Issued Instruction` 从 `reg` 的 5.84 降到 **5.28**——等待变少了。

## 5. 实测总览与 cuBLAS 对照

`2048³`，H100，FP32（TF32 关闭）：

| 版本 | 时间 (ms) | TFLOPS | 占 fp32 峰值 | 相对 tiled32 | 关键改动 |
|---|---|---|---|---|---|
| tiled32（08 基线） | 1.8851 | 9.11 | 13.6% | 1.00× | smem 分块，每线程 1 输出 |
| reg | 0.6547 | 26.24 | 39.2% | 2.88× | 每线程 8×8，交错映射 |
| reg_vec | 0.5707 | 30.10 | 45.0% | 3.30× | + float4 向量化载入 |
| **reg_db** | **0.4986** | **34.46** | **51.5%** | **3.78×** | + cp.async 双缓冲 |
| cuBLAS（torch） | 0.3386 | 50.73 | 75.8% | 5.57× | 厂商库 |

> cuBLAS 数字来自 `torch.matmul`（显式关掉 TF32，见 `cublas_ref.py`），同口径 FP32。我们的 `reg_db` 达到 cuBLAS 的 **67.9%**。

`ncu` 关键指标横向对比：

| kernel | Compute (SM) | L1/TEX | DRAM | 寄存器/线程 | 活跃 warp 占用 |
|---|---|---|---|---|---|
| tiled32 | 74.68% | **91.00%** | 0.78% | 31 | — |
| reg | 62.53% | 39.29% | 1.66% | 128 | 23.3% |
| reg_vec | 54.75% | 46.07% | 1.91% | 118 | 23.8% |
| reg_db | **66.43%** | 43.78% | 2.19% | 128 | 22.7% |

一个反直觉的点：`reg_vec` 的 `Compute` 占比（54.75%）比 `reg`（62.53%）**更低**，却更快。原因是向量化让总指令数变少——同样的 FLOPs 用了更少的周期，分母变小、占比自然不同。**看吞吐要用绝对时间（Duration），占比只用来定位瓶颈。**

## 6. 还剩什么没榨干

`ncu` 三条线索指向下一步：

1. **occupancy 被寄存器锁死在 25%**（128 regs → 2 block/SM）。`ncu` 建议 `__launch_bounds__(256, 2)` 或减小 `acc` 分块来提升驻留 warp 数；
2. `Compute` 才 66%，**warp 仍有约 5.3 个周期才发射一条指令**，说明延迟还没被完全掩盖；
3. cuBLAS 还在前面——它用了更大的 tile、更深的流水、更聪明的寄存器分配。

这些都建立在「读懂 `ncu` 报告」之上，所以下一篇先系统讲 **ncu 深潜**：occupancy 到底怎么算、warp stall reason 怎么读、roofline section 怎么用、source/SASS 怎么对照。第 12 篇再回到 `cp.async`/TMA 做更深的流水线。

## 小结

- **寄存器分块**是本篇最大的跳跃：每线程算 `8×8`，把一次 smem load 从「喂 1 次 FMA」变成「喂 8 次」，`L1/TEX` 从 91% 降到 39%，13.6% → **39.2%**（2.88×）。
- **交错线程映射**（`ty+16i` / `tx+16j`）不用 padding 就消掉了 smem bank conflict。
- **float4 向量化**把载入指令数降到 1/4，再涨到 **45.0%**。
- **cp.async 双缓冲**让「搬下一块」和「算当前块」重叠，`Warp Cycles Per Issued` 5.84 → 5.28，涨到 **51.5%**。
- 与同口径 **cuBLAS 50.73 TFLOPS（75.8%）** 相比，我们做到约 **68%**。
- 剩下的天花板是 **occupancy（寄存器限制 25%）**与 **warp 延迟掩盖**——交给第 10 篇的 ncu 深潜。

> 下一篇：CUDA 算子调优（十）· ncu 深潜：occupancy、warp stall 与 roofline
