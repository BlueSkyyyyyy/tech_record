---
title: "CUDA 算子调优（二）：第一个 kernel — 从 vector add 看懂线程层次"
date: 2026-09-21
draft: false
weight: 2
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "vector add", "系列"]
categories: ["算子开发"]
---

第一篇讲了 GPU 的执行模型和 roofline 这把尺子。这一篇动手写第一个 CUDA kernel——**向量加法** `c = a + b`。它简单到不能再简单，但麻雀虽小五脏俱全：线程怎么编号、grid 和 block 怎么配、为什么同样的计算量换个写法就快一点，全都能在这一个算子里看懂。

配套代码：[`code/kernel-opt/02-first-kernel/vector_add.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/02-first-kernel/vector_add.cu)。上一篇见 {{< relref "cuda-kernel-opt-01-overview" >}}。

---

## 1. 最直观的写法：一个线程算一个元素

CUDA kernel 就是一段用 `__global__` 修饰的函数，在 GPU 上由**很多线程同时**执行。以 vector add 为例：

```cuda
__global__ void add_v0(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // 算出「我是第几个元素」
  if (i < n) c[i] = a[i] + b[i];
}
```

关键是这两行：

- `threadIdx.x` 是**块内编号**（0..blockDim.x-1）；
- `blockIdx.x` 是**块的编号**；
- 两者拼起来得到全局唯一的 `i = blockIdx.x * blockDim.x + threadIdx.x`，正好落在 `[0, n)`。

这是一种「笛卡尔式」的一维映射：block 0 装线程 0..255，block 1 装 256..511……所以启动时：

```cuda
int block = 256;
int grid  = (n + block - 1) / block;   // 向上取整，保证覆盖 n 个元素
add_v0<<<grid, block>>>(a, b, c, n);
```

`<<<grid, block>>>` 就是「启动配置」：`grid` 个 block、每块 `block` 个线程，总共 `grid*block` 个线程。`if (i < n)` 是必需的边界检查——最后一个 block 通常会有多余的线程。

> 小知识：`__restrict__` 是给编译器的承诺——`a`、`b`、`c` 指向的内存互不重叠。有了它，编译器才敢做更激进的指令调度与缓存优化。写 kernel 时加上它几乎总是对的。

## 2. 工业界常用写法：grid-stride loop

`grid` 固定成「数据量 / block」有个问题：**线程数和数据量绑死了**。如果 `n` 很大，grid 会大到超过硬件上限（x 方向最多 2³¹-1 个 block，虽然一般够用）；更实际的问题是，数据量变化时启动配置也要跟着变，不利于调优。

更通用的写法是 **grid-stride loop（网格步长循环）**：启动固定数量的 block，让每个线程用「跨整个 grid 的步长」去遍历数据：

```cuda
__global__ void add_v1(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c, int n) {
  const int stride = gridDim.x * blockDim.x;       // 整个 grid 的线程总数
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    c[i] = a[i] + b[i];
}
```

这样**启动配置和问题规模解耦**了：不管 `n` 多大，都可以只启动「刚好填满 GPU」的 block 数，让每个线程循环几轮。这也是很多生产 kernel 的默认骨架。

## 3. 更省指令的写法：向量化 float4

前两种写法每个线程一次只搬 4 字节（一个 `float`）。如果让每个线程一次搬 **16 字节**（`float4`，四个 float 打包），指令数就降到 1/4，访存事务也更规整：

```cuda
__global__ void add_v2(const float4* __restrict__ a,
                       const float4* __restrict__ b,
                       float4* __restrict__ c, int n4) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}
```

要点：把指针转成 `float4*`，数组长度变成 `n/4`，一个线程处理一个 `float4`。**前提是数据按 16 字节对齐**（`cudaMalloc` 分配的内存天然满足），且 `n` 能被 4 整除。

## 4. 实测：它们真的差很多吗？

在 H100 上跑 `scripts/run.sh 02-first-kernel/vector_add.cu`，`n = 2²⁴ = 16M`（每个数组 64 MB），三个版本都用同样的配置、都先和 CPU 结果对拍：

```
N = 16777216 floats (67.1 MB/array, 3 arrays = 201.3 MB)

  [v0] max_err = 0.000e+00 OK
v0  one-thread-per-elem        0.0751 ms    2682.1 GB/s  ( 80.0% of peak)
  [v1] max_err = 0.000e+00 OK
v1  grid-stride                0.0746 ms    2698.3 GB/s  ( 80.5% of peak)
  [v2] max_err = 0.000e+00 OK
v2  float4 grid-stride         0.0717 ms    2807.5 GB/s  ( 83.7% of peak)

speedup v2/v0 = 1.05x, v2/v1 = 1.04x
```

「有效带宽」的算法：vector add 每个元素读 `a`、读 `b`、写 `c`，共 12 字节，所以

$$
\text{GB/s} = \frac{3 \times N \times 4\ \text{字节}}{\text{耗时}}
$$

比如 v2：`3 × 16M × 4 B / 0.0717 ms ≈ 2808 GB/s`，占 H100 峰值 3352 GB/s 的 **83.7%**。

**结论有点反直觉**：三种写法都摸到了 80%+ 的带宽，向量化只比最朴素的写法快 5%。为什么？

- 这三个版本**访存都是合并的（coalesced）**——一个 warp 的 32 个线程访问连续地址，硬件把它们合并成尽量少的显存事务。只要合并了，简单写法也能接近带宽上限。
- float4 省的是**指令数**，而这算子在 80% 带宽时并不是被指令发射卡住，所以收益有限。

这是一个重要的心态：**优化之前先判断瓶颈在哪**。如果访存已经合并、带宽已经接近峰值，那再怎么折腾指令也是白搭；反之如果访存没合并，性能可能直接掉到零头（第 04 篇会演示）。Roofline 那句话说得很对——低 AI 算子就是内存受限，先把带宽吃满再说。

## 5. 启动配置怎么选？扫一遍

既然带宽是瓶颈，那启动配置（block 大小、block 数量）影响大吗？把 v2 在 `block ∈ {128,256,512,1024}` × `grid = 132×{1,2,4,8,16}` 上扫一遍（132 是 H100 的 SM 数）：

```
   block     grid           ms       GB/s
     128      132       0.0989     2034.7
     128      264       0.0787     2558.9
     128      528       0.0705     2856.5
     128     1056       0.0704     2859.5
     128     2112       0.0714     2819.1
     256      132       0.0786     2561.4
     256      264       0.0703     2864.4
     256      528       0.0708     2844.7
     256     1056       0.0716     2810.9
     256     2112       0.0698     2885.6
     512      132       0.0703     2865.8
     512      264       0.0704     2860.8
     512      528       0.0715     2817.1
     512     1056       0.0698     2883.5
     512     2112       0.0693     2906.9
    1024      132       0.0705     2854.8
    1024      264       0.0708     2841.7
    1024      528       0.0697     2888.1
    1024     1056       0.0694     2902.2
    1024     2112       0.0686     2936.7
```

读出来的规律：

1. **并行度不够会明显掉带宽**：`block=128, grid=132`（每 SM 只有 4 个 warp）只有 2035 GB/s。每个 SM 驻留的 warp 太少，盖不住访存延迟。
2. **一旦并行度足够，就进入平台区**：2850~2940 GB/s 上下浮动，block 大小影响不大。最佳是 `block=1024, grid=132×16`（2937 GB/s）。
3. 平台区里那点波动（±3%）基本是噪声——调启动配置别追求最后 1%，先把「并行度是否足够」这个大问题解决。

> 经验法则：让 grid 至少是 SM 数的几倍（这里 528 = 132×4 起就基本饱和），block 取 128~1024 都行。真正的调优空间在访存模式和算法本身，不在这一两个参数。

## 小结

- CUDA 线程用 `blockIdx/blockDim/threadIdx` 三级索引定位；`if (i < n)` 处理边界。
- **grid-stride loop** 让启动配置与数据量解耦，是生产 kernel 的常用骨架。
- **float4 向量化**减少指令数，但在访存已合并、带宽接近峰值时收益有限（本例仅 1.05×）。
- 低 AI 算子是内存受限的——先看带宽利用率（本例 80–84%），别一上来就抠指令。
- 启动配置扫描：**并行度足够后进入平台区**，block 大小不敏感。

下一篇讲一个比写代码更容易翻车的事：**怎么正确地测量**。同样的 kernel，计时方法不对，结论能差好几倍。

> 下一篇：CUDA 算子调优（三）· 正确测量
