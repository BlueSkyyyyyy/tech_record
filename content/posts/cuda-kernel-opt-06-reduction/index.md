---
title: "CUDA 算子调优（六）：归约与 warp shuffle — 求和为什么能慢 1000 倍"
date: 2026-09-21
draft: false
weight: 6
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "归约", "warp shuffle", "ncu", "系列"]
categories: ["算子开发"]
---

前面几篇讲的都是「搬运」——怎么把数据搬得更整齐。这一篇换一类问题：**归约（reduction）**，也就是把一堆数合并成少数结果，最典型的就是「求一个大数组的和」。它看起来毫无难点，但一个天真的写法能让它比最优解**慢 1000 倍以上**（不是夸张，实测 2.3 GB/s vs 3057 GB/s）。归约也是 softmax、LayerNorm、attention 里所有「求最大值 / 求和」的底座，值得单独搞清楚。

配套代码：[`code/kernel-opt/06-reduction/reduction.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/06-reduction/reduction.cu)。前一篇见 {{< relref "cuda-kernel-opt-05-transpose" >}}。

---

## 1. 问题与理论天花板

把长度 `N = 2²⁶`（约 6700 万）的 `float` 数组求和。这个算子：

- **只读不写**：搬运 `N × 4 = 268 MB`；
- 每个元素 1 次加法，**AI ≈ 0.25 FLOP/byte**，远低于拐点——纯内存受限。

理论上限就是 HBM 带宽：`268 MB / 3.35 TB/s ≈ 80 µs`，即有效带宽 ~3352 GB/s。

看似简单：每个线程读一部分、加起来、最后汇总即可。难点全在「最后怎么汇总」。

## 2. 反面教材：每个元素一次 atomicAdd

最直接的「并行」思路——每个线程读一个元素，直接加到结果上：

```cuda
__global__ void reduce_atomic_every(const float* in, float* out, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    atomicAdd(out, in[i]);       // 6700 万次原子加，全砸在同一个地址上
}
```

实测结果触目惊心：

```
[atomic_every  ] sum=33554432.0 ref=67343745.0 rel_err=5.02e-01 FAIL
atomic_every (bad)         117.8601 ms       2.3 GB/s  (  0.1% of peak)
```

**117.9 毫秒，2.3 GB/s，只有峰值的 0.1%**，比理论值慢了 1400 多倍。`ncu` 一看，所有单元都在闲着：

```
reduce_atomic_every:  DRAM 0.07%   L2 1.01%   Compute 0.22%
```

DRAM、L2、算力全都没干活，那时间花哪了？花在**原子操作的串行化**上。`atomicAdd` 是对**同一个地址**的读-改-写，硬件必须把它们一个个排队执行：几千万次操作首尾相接，显存完全喂不饱。

> 还有一个隐藏的坑：这次求和结果也是**错的**（rel_err = 0.5）。因为所有加法都堆在同一个 float 累加器上，当累加值超过 2²⁴ 后，加上一个 1.0 已经无法精确表示，后面的加数被直接丢掉。**浮点求和的顺序会影响精度**，这也是并行归约要用「多路部分和」的另一个原因。

## 3. 正确姿势：先局部求和，再汇总

核心思想很朴素：**别让几千万个线程都去碰同一个地址**。让每个线程先在本地累加，每个 block 内部再合并成一个数，最后每个 block 只做一次 `atomicAdd`：

```
grid（1056 个 block）
 ├── block 0 ── 256 线程各自局部求和 → block 内合并成 1 个数 → atomicAdd 一次
 ├── block 1 ── ...
 ...
```

这样全局原子操作从 6700 万次降到 **1056 次**，可以忽略。

「block 内怎么合并」有两条路线：**shared memory 树形归约** 和 **warp shuffle**。

### 3.1 shared memory 树形归约

把 256 个线程的部分和写进 shared memory，然后两两合并，规模每次减半：

```cuda
__global__ void reduce_smem_tree(const float* in, float* out, int n) {
  extern __shared__ float s[];
  const int tid = threadIdx.x;
  float sum = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) sum += in[i];
  s[tid] = sum;
  __syncthreads();
  for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
    if (tid < step) s[tid] += s[tid + step];
    __syncthreads();               // 每轮都要同步，且半数线程在空转
  }
  if (tid == 0) atomicAdd(out, s[0]);
}
```

树形归约把比较次数从 O(n) 降到 O(log n)（256 个值只需 8 轮），但它要反复 `__syncthreads()`，而且每轮有一半线程闲置。

### 3.2 warp shuffle：在寄存器里直接换数据

前 5 轮（256 → 8，即直到每个 warp 内只剩 1 个值）其实**不需要 shared memory**。同一个 warp 的 32 个线程可以用 **`__shfl_down_sync`** 直接在寄存器之间传递数据——这是硬件提供的 warp 内数据交换指令：

```cuda
__device__ float warp_reduce_sum(float v) {
  for (int off = 16; off > 0; off >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, off);   // 取「后面第 off 个 lane」的值
  return v;   // lane 0 得到整个 warp 的和
}
```

`__shfl_down_sync(mask, v, off)` 的含义：warp 内每个线程拿到「自己 lane+off 的线程」的 `v`。于是 off=16 时，lane 0 拿到 lane 16 的值加上去；off=8、4、2、1……最后 **lane 0 手里就是 32 个值的总和**。整个过程没有 shared memory、没有 `__syncthreads`，全在寄存器里完成，特别快。

跨 warp 才需要 shared memory：每个 warp 把自己的和写进 `wsum[wid]`，再由 warp 0 把这至多 32 个数用同样的 shuffle 归约一次。

```cuda
__global__ void reduce_warp_shuffle(const float* in, float* out, int n) {
  const int tid = threadIdx.x;
  float sum = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + tid; i < n; i += stride) sum += in[i];
  sum = warp_reduce_sum(sum);                 // intra-warp：寄存器交换
  __shared__ float wsum[32];
  const int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) wsum[wid] = sum;
  __syncthreads();
  if (wid == 0) {                             // 只让第一个 warp 收尾
    sum = (lane < (blockDim.x >> 5)) ? wsum[lane] : 0.f;
    sum = warp_reduce_sum(sum);
    if (lane == 0) atomicAdd(out, sum);
  }
}
```

### 3.3 再向量化

读的部分还能用第 04 篇的 `float4`：一个线程一次读 16B，指令数除以 4。

```cuda
for (int i = ...; i < n4; i += stride) {
  float4 v = in[i];
  sum += (v.x + v.y) + (v.z + v.w);
}
```

## 4. 实测对比

```
N = 67108864 floats (268.4 MB)

atomic_every (bad)           117.8601 ms       2.3 GB/s  (  0.1% of peak)
smem_tree                      0.0905 ms    2966.1 GB/s  ( 88.5% of peak)
warp_shuffle                   0.0902 ms    2975.3 GB/s  ( 88.8% of peak)
vec4_shuffle                   0.0878 ms    3057.4 GB/s  ( 91.2% of peak)
```

| 版本 | 时间 | 有效带宽 | 占峰值 | 关键改动 |
|---|---|---|---|---|
| atomic_every | 117.86 ms | 2.3 GB/s | 0.1% | 每元素一次原子加（反面教材） |
| smem_tree | 0.0905 ms | 2966 GB/s | 88.5% | block 内 smem 树形归约，每 block 一次原子加 |
| warp_shuffle | 0.0902 ms | 2975 GB/s | 88.8% | warp 内用 `__shfl_down_sync` 取代 smem |
| vec4_shuffle | 0.0878 ms | 3057 GB/s | 91.2% | 再加 `float4` 向量化读 |

`ncu` 的 SOL 也印证了瓶颈的转移：

```
reduce_atomic_every :  DRAM  0.07%   L2  1.01%   Compute  0.22%     ← 全部空转，卡在原子串行
reduce_smem_tree    :  DRAM 88.47%   L2 83.86%   Compute 10.51%     ← 已经内存受限
reduce_vec4_shuffle :  DRAM 91.61%   L2 86.33%   Compute  4.66%     ← 逼近带宽峰值
```

两个结论：

1. **真正的 1000× 在于「不要每元素原子加」**。一旦改成两级归约，立刻从 0.1% 跳到 88%，已经贴着 roofline 的天花板了。
2. **shuffle 相对 smem 只快一点点**（88.5% → 88.8%），因为此时瓶颈已经是 DRAM，不是 block 内的合并方式。但 shuffle 省掉了 shared memory 和同步，在算子融合、warp 级流水里价值更大（后面 attention/softmax 会用到）。**优化要认准当前瓶颈**：纯归约已经是内存受限，block 内再怎么抠收益都有限。

> 精度小贴士：并行归约的误差取决于「部分和」的划分。这里先按线程、再按 warp、再按 block 逐级合并，累加器不会过早饱和，所以 `rel_err` 只有 ~1e-5，远好于每元素原子加的 0.5。

## 5. 归约还能怎么用

归约本身很少单独出现，它几乎总是作为**更大算子的一步**：

- **softmax**：先求每行最大值（max reduction），再求指数和（sum reduction），然后归一——第 07 篇的主角；
- **LayerNorm**：每行均值、方差，全是归约；
- **attention**：online softmax 里不断更新的 `m`（running max）和 `l`（running sum），是「边走边归约」。

所以「多级归约 + warp shuffle」这个模式，会在后面每一篇里反复出现。

## 小结

- 归约 AI 极低，理论天花板是 HBM 带宽；`N=2²⁶` 的理论下限约 80 µs。
- **每元素一次 `atomicAdd` 是灾难**：实测 2.3 GB/s（0.1%），瓶颈是全局原子串行（`ncu` 显示所有单元 <1%），而且浮点结果还错了。
- 正解是**两级归约**：线程局部累加 → block 内合并 → 每 block 一次原子加，立刻到 88%。
- block 内合并：**`__shfl_down_sync`（warp 内寄存器交换）** 比 shared memory 树更简洁；跨 warp 再借助 shared memory。
- 加 `float4` 后到 **91.2%**，真正贴着带宽跑。
- 指标看 `l1tex__data_bank_conflicts_*`（上一篇）和 SOL 的 DRAM%；**先确认瓶颈，再优化对应单元**。

下一篇把这些积木拼起来：**Softmax 与 LayerNorm 的优化**，从「多趟 kernel」到「一趟、融合、online」的完整优化历程。

> 下一篇：CUDA 算子调优（七）· Softmax / LayerNorm 优化
