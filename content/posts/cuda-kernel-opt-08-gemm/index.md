---
title: "CUDA 算子调优（八）：GEMM 入门 — 从朴素三重循环到 shared-memory tiling"
date: 2026-09-21
draft: false
weight: 8
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "GEMM", "shared memory", "tiling", "ncu", "系列"]
categories: ["算子开发"]
---

前面七篇的算子（elementwise、copy、transpose、reduce、softmax）全都是**内存受限**的，撞的是 3.35 TB/s 那条带宽线。从这一篇开始，我们进入另一个世界：**计算受限**的算子。最典型的代表就是 **GEMM（矩阵乘）**——它是所有深度学习计算的绝对核心，也是整个系列里优化空间最大、技巧最丰富的一个。这一篇先把最基本的优化讲透：**为什么朴素写法慢，以及 shared-memory tiling 怎么把全局访存降下来**。实测从 **5.46 TFLOPS 提升到 9.05 TFLOPS**，并用 `ncu` 找到下一步该优化的单元。

配套代码：[`code/kernel-opt/08-gemm/gemm.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/08-gemm/gemm.cu)。前一篇见 {{< relref "cuda-kernel-opt-07-softmax" >}}。

---

## 1. GEMM 为什么是计算受限

计算 `C[M,N] = A[M,K] × B[K,N]`（行主序，fp32）：

- **FLOPs** = `2·M·N·K`；
- **最少搬运** = 读 A、读 B、写 C = `4·(M·K + K·N + M·N)` 字节。

对 `2048³`：

$$
\text{AI} = \frac{2 \times 2048^3}{4 \times 3 \times 2048^2} = \frac{2 \times 2048}{12} \approx 341\ \text{FLOP/byte}
$$

而 H100 上 FP32 CUDA core 的 roofline 拐点只有约 **20 FLOP/byte**（66.9 TFLOPS ÷ 3.35 TB/s）。GEMM 的算术强度比拐点高一个数量级，**妥妥地落在计算受限区**——天花板是 66.9 TFLOPS。本文的优化目标就是把实测 TFLOPS 往这条顶线上推。

> 这也是 GEMM 和前面算子的根本区别：内存受限算子优化「怎么少搬、怎么搬得快」，计算受限算子优化「怎么喂饱计算单元」。后者几乎总是围绕「**数据复用**」做文章。

## 2. 版本 0：朴素三重循环

一个线程算一个输出元素：

```cuda
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += A[row * K + k] * B[k * N + col];   // 每次乘加都要读 A、B
  C[row * N + col] = acc;
}
```

它的问题：算一个 `C[row][col]` 要循环 `K` 次，每次从内存读一个 A 元素和一个 B 元素——**每个 A/B 元素被用了 1 次就丢**，没有任何复用。逻辑视角的总读取量是 `2·M·N·K` 个数，是理论最小值的几百倍。

实测（`2048³`）：

```
naive              3.1481 ms      5.46 TFLOPS  (  8.2% of fp32 peak)
```

只有峰值的 8.2%。`ncu` 一看瓶颈就清楚了：

```
gemm_naive:  L1/TEX Cache Throughput 99.35%   ← 瓶颈在这里
             Compute (SM) 65.28%   L2 20.11%   DRAM 0.47%
```

**`L1/TEX` 吞吐 99.35%，被打满了；DRAM 只有 0.47%。** 也就是说，瓶颈不是显存带宽，而是「每个 FMA 都要发两条 load」——`L1` 的载入管道处理不过来。数据其实都命中缓存（L2 命中 97%），但**访存指令本身把流水线堵死了**。

结论：想快，必须**让一次数据加载被多个计算复用**，减少 load 指令数。

## 3. 版本 1：shared-memory tiling

核心思想：**一个 block 负责算出 `C` 的一块 `TILE×TILE`，把计算这块需要的 A/B 分块先协作搬进 shared memory，然后在片上反复用**。

```
         B 的分块 [TILE x TILE]
        ┌────────┐
 A分块  │  C 分块 │   每个 block：
[TILEx │ [TILEx │   1. 从全局加载 A、B 的分块到 smem
 TILE]  │  TILE] │   2. 从 smem 读，循环 K 做乘加
        └────────┘   3. 写回 C 分块
```

关键收益：一次全局加载进 smem 的数据，会被 block 内的 `TILE` 个线程复用（A 元素被同一行的 TILE 个输出复用，B 元素被同一列的 TILE 个输出复用）。全局访存次数降到约 `1/TILE`。

```cuda
template <int T>
__global__ void gemm_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
  __shared__ float As[T][T];
  __shared__ float Bs[T][T];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * T + ty;
  const int col = blockIdx.x * T + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += T) {
    As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.f;   // 合并加载
    Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.f;
    __syncthreads();                        // 等分块到齐
    #pragma unroll
    for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];   // 片上复用
    __syncthreads();                        // 等大家都用完，再装下一块
  }
  if (row < M && col < N) C[row * N + col] = acc;
}
```

这里的 shared memory 访问没有 bank conflict：内层循环里 warp 内 `tx` 变化，`Bs[k][tx]` 是连续地址；`As[ty][k]` 对所有 `tx` 是同一个地址（广播）。

## 4. 实测对比

```
M=N=K=2048, GEMM, FLOPs = 17.18 GFLOP

naive              3.1481 ms      5.46 TFLOPS  (  8.2% of fp32 peak)
tiled 16           2.1072 ms      8.15 TFLOPS  ( 12.2% of fp32 peak)
tiled 32           1.8993 ms      9.05 TFLOPS  ( 13.5% of fp32 peak)
```

| 版本 | 时间 | TFLOPS | 占 fp32 峰值 | 关键改动 |
|---|---|---|---|---|
| naive | 3.148 ms | 5.46 | 8.2% | 每线程一个输出，load 无复用 |
| tiled 16 | 2.107 ms | 8.15 | 12.2% | 16×16 smem 分块，全局访存 ÷16 |
| tiled 32 | 1.899 ms | 9.05 | 13.5% | 32×32 分块 |

tiling 带来 **1.66 倍**提升。但 13.5% 离峰值还很远，`ncu` 告诉你为什么：

```
gemm_naive   :  L1/TEX 99.35%   Compute 65.28%   DRAM 0.47%
gemm_tiled32 :  L1/TEX 91.00%   Compute 74.68%   DRAM 0.78%
```

tiled 版本的瓶颈**还是 `L1/TEX` 管道（91%）**。为什么？因为 `As[ty][k] * Bs[k][tx]` 每次乘加仍然要发**两条 shared memory load**。shared memory 的访问虽然不占 HBM 带宽，但它走的是和 L1 同一套 `LSU` 载入管道，一次只能处理有限条 load。于是内层循环被「两条 smem load 喂一次 FMA」卡住，`Compute` 只能到 74.7%。

**要突破，就得让「一次 smem load 被多次 FMA 复用」。** 具体做法：每个线程不再只算一个 C 元素，而是算一小片 `m×n`（比如 4×4）——把 `As` 的 4 个值和 `Bs` 的 4 个值加载到寄存器后，做 16 次 FMA。这就是 **寄存器分块（register tiling / thread tiling）**，也是下一篇的主题。它能把 `As`/`Bs` 的 load 次数再降一个数量级，是逼近 cuBLAS 的关键一步。

## 小结

- GEMM 的算术强度约 **341 FLOP/byte**，远高于 FP32 拐点 20，是**计算受限**算子，天花板是 66.9 TFLOPS（CUDA core）。
- 朴素三重循环只有 **8.2%**，瓶颈是 **`L1/TEX` 载入管道（99.35%）**，不是 DRAM——每个 FMA 都要发两条 load。
- **shared-memory tiling** 把全局访存降为 `1/TILE`，提升到 **13.5%**；32×32 比 16×16 略好。
- tiling 后瓶颈仍在 `L1/TEX`（91%）：每次 FMA 仍要两条 **shared memory load**。
- 下一步是 **寄存器分块**，让一次 smem load 被多个 FMA 复用——这是第 09 篇。

到这里，系列已经走完「内存受限算子」的全套方法，并叩开了「计算受限」的大门。下一篇把 GEMM 一路推到寄存器分块 + double buffering，看它怎么从 13% 爬到 50%+。

> 下一篇：CUDA 算子调优（九）· GEMM 进阶：寄存器分块与 double buffering
