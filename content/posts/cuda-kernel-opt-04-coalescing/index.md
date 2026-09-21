---
title: "CUDA 算子调优（四）：访存合并与向量化 — 同一个 copy，差 7 倍"
date: 2026-09-21
draft: false
weight: 4
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "访存合并", "coalescing", "ncu", "系列"]
categories: ["算子开发"]
---

前一篇讲了怎么正确测量，这一篇进入真正的优化。第一个要刻进肌肉记忆的概念是**访存合并（memory coalescing）**：GPU 的内存系统以 **warp（32 个线程）** 为单位取数据，如果这 32 个线程访问的地址是连续的，硬件能用最少的显存事务满足它们；如果地址散开，同样一次取数会拆成几十个事务。代价有多大？这一篇用一个「什么都没算」的 copy 算子演示：只是换了个遍历顺序，有效带宽从 **2381 GB/s 掉到 323 GB/s，慢了 7.4 倍**。

配套代码：[`code/kernel-opt/04-coalescing/coalescing.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/04-coalescing/coalescing.cu)。前一篇见 {{< relref "cuda-kernel-opt-03-measurement" >}}。

---

## 1. 原理：一个 warp 一次能拿多少数据

GPU 的显存访问以 **32 字节的 sector** 为最小单位，L1/L2 缓存行是 **128 字节**。当一个 warp 的 32 个线程发起一次加载：

- **合并（coalesced）**：32 个线程访问连续地址（每个 4B），合起来正好 128B = 4 个 sector。**一次请求，4 个 sector**。
- **不合并（strided）**：每个线程的地址隔得很远，落在不同的 128B 行里，硬件只能拆成 **32 个独立请求、32 个 sector**。

```
合并：线程 0..31 访问 addr+0,4,8,...,124
      ┌─────────────────── 128B 缓存行 ───────────────────┐
      │ t0 t1 t2 ... t31 │                                │  → 4 sectors，1 次请求
      └──────────────────────────────────────────────────┘

不合并：线程 0..31 访问 addr+0, N, 2N, ..., 31N（N 很大）
      │t0│        │t1│        │t2│   ...   │t31│          → 32 sectors，32 次请求
       ↑ 每个线程各占一个 sector，实际拿回 32×32B=1024B
         却只用了 32×4B=128B
```

这就是为什么**遍历顺序**如此重要：内存里怎么摆数据是固定的，但「哪个线程读哪个地址」是你定的。

## 2. 实验：三种遍历

用一个 `M×N` 的矩阵（这里 8192×8192，行主序存储），写三个 copy：

```cuda
// (a) 行优先：第 idx 个线程读第 idx 个元素 → 连续地址，合并
__global__ void copy_rowwise(const float* __restrict__ in, float* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) out[idx] = in[idx];
}

// (b) 列优先：把线性 idx 当列主序坐标解释，再换算回行主序偏移
//     相邻线程 → 同一列、相邻行 → 地址相差 N 个元素
__global__ void copy_colwise(const float* __restrict__ in, float* __restrict__ out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < M * N) {
    int col = idx / M, row = idx % M;
    int off = row * N + col;
    out[off] = in[off];
  }
}

// (c) 合并 + float4 向量化：每线程 16B
__global__ void copy_vec4(const float4* __restrict__ in, float4* __restrict__ out, int n4) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) out[idx] = in[idx];
}
```

三者都先和预期值对拍（全部为 `1.0f`），保证正确性后再计时。

## 3. 实测：慢在哪

`scripts/run.sh 04-coalescing/coalescing.cu`，工作集 537 MB（远大于 52 MB L2）：

```
M=8192 N=8192, matrix 268.4 MB, working set (in+out) 536.9 MB

rowwise (coalesced)            0.2254 ms    2381.9 GB/s  ( 71.1% of peak)
colwise (strided)              1.6629 ms     322.8 GB/s  (  9.6% of peak)
rowwise vec4                   0.1792 ms    2996.7 GB/s  ( 89.4% of peak)

slowdown colwise/rowwise = 7.4x   speedup vec4/rowwise = 1.26x
```

- 合并的 copy 达到 2382 GB/s（71%），向量化后 **2997 GB/s（89%）**——这就是 H100 上内存受限算子的现实上限附近。
- 列优先的 copy 只剩 **323 GB/s（9.6%）**。它没有任何计算，只是地址算得不一样。

用 `ncu` 看每个 warp 请求拆成了多少 sector，一目了然（`l1tex__average_t_sectors_per_request`）：

```
copy_rowwise :  load 4 sectors/request   store 4 sectors/request   DRAM 2.35 TB/s
copy_colwise :  load 32 sectors/request  store 32 sectors/request  DRAM 0.34 TB/s
```

合并版本每次请求 4 个 sector（128B / 32B），正好是「一个 warp 一次拿满一条缓存行」的理想值；列优先版本是 **32 个 sector/请求**，即每个线程各占一个 sector，实际取回 32×32B = 1024B，却只用了 128B。

## 4. 更有意思的：它并不是被 DRAM 卡住

直觉会说「不合并 → 浪费 8 倍显存带宽」。但 `ncu` 的 SpeedOfLight 给出了更精确的答案：

```
copy_rowwise:  DRAM Throughput 70.1%   L2 Throughput 69.8%   L1 16.8%   Compute 35.9%
copy_colwise:  DRAM Throughput 10.2%   L2 Throughput 84.8%   L1 46.1%   Compute  5.8%
               L2 Hit Rate 94.1%
```

列优先版本 **L2 吞吐 84.8%，是全场唯一接近饱和的单元；而 DRAM 只有 10.2%，L2 命中率高达 94%**。怎么解释？

- 被「多取」的那些数据（同一 128B 行里其它列）并没有浪费——它们随后会被处理相邻列的 block 再次访问，命中了 L2。所以**DRAM 几乎没多搬数据**。
- 真正的代价在 **L2 的请求处理能力**：一次 warp 加载指令被拆成 32 个 sector 请求，L2 得处理 32 倍数量的小事务。哪怕数据都在 L2 里，L2 的**事务吞吐**也会被打满（84.8%），而每个事务只服务 4 字节有效数据，效率极低。
- 最终结果：DRAM 闲着，L2 忙死，kernel 卡在 L2 事务吞吐上，有效带宽只剩 10%。

所以「不合并」的准确表述是：**它把宝贵的 L2/内存事务带宽浪费在碎片化的小请求上**。是被 L2 事务吞吐卡住，而不是被 HBM 带宽卡住——这两种瓶颈要用不同的办法解（前者必须改访问模式，后者才谈得上减少数据量）。

> 这也解释了为什么「加上 8 倍」的粗略估算会错：数据有复用，DRAM 未必真的多搬。

## 5. 向量化的收益与代价

回到合并版本，`float4` 让每个线程一次搬 16B（一次请求 128B 变成 512B），把 71% 提到 89%。收益来自两点：

1. **指令更少**：4 个元素的搬运合并成 1 条 `LDG.128`/`STG.128`，发射和地址计算的开销除以 4；
2. **事务更宽**：每个线程一个大事务，更容易把 DRAM 的突发长度喂满。

代价/前提：

- 需要 **16 字节对齐** 和元素个数是 4 的倍数（`cudaMalloc` 的地址天然 128B 对齐）；
- 数据类型/算法要能凑成 4 个一起处理。对 elementwise、copy、reduce 这类连续算子都适用；对于本身就不连续的算子（如 attention 的 gather）则不适用。

## 6. 经验法则

- **让相邻线程访问相邻地址**。判断标准：`offset = blockIdx.x*blockDim.x + threadIdx.x` 且直接当数组下标用，通常就是合并的。
- 二维数组里，**行主序存储就按行遍历**（让 `threadIdx.x` 对应列）；按列遍历会把 stride 变成 `N`，立刻踩坑。
- 怀疑不合并时，看 `ncu` 的 `l1tex__average_t_sectors_per_request`：**理想是 4（128B/32B）**，明显大于 4 就有问题。
- 能用 **128 位向量（float4/int4）**就用，元素对齐时几乎白赚 20%+；注意对齐和整除。
- 「不合并」常常表现为 **L2 吞吐高、DRAM 吞吐低**，别只盯着 DRAM% 下判断。

## 小结

- warp（32 线程）是内存访问的基本单位；连续地址 → 4 sector/请求，散开地址 → 最多 32 sector/请求。
- 同一个 copy，仅把遍历顺序从行优先换成列优先，有效带宽 **2382 → 323 GB/s（慢 7.4×）**。
- `ncu` 揭示瓶颈其实是 **L2 事务吞吐（84.8%）**，而 DRAM 只有 10%——不合并浪费的是内层事务带宽，不一定浪费 HBM。
- `float4` 向量化把合并 copy 从 71% 提到 **89%**，是内存受限算子的常规操作。
- 记住那条准则：**相邻线程访问相邻地址**。

下一篇用这个原理干一件正经事：**矩阵转置**。转置的读和写天然有一边是不合并的，怎么用共享内存把它救回来，是 CUDA 优化最经典的一课。

> 下一篇：CUDA 算子调优（五）· 共享内存与 bank conflict（矩阵转置）
