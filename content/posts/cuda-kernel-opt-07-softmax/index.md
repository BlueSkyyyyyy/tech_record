---
title: "CUDA 算子调优（七）：Softmax 优化 — 从 3 个 kernel 到 1 个"
date: 2026-09-21
draft: false
weight: 7
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "softmax", "online softmax", "ncu", "系列"]
categories: ["算子开发"]
---

前面几篇攒下的积木——合并访存、共享内存、归约、warp shuffle——到这一篇终于拼成一个真实算子：**Softmax**。它是每个 Transformer 的注意力里都要跑的东西，也是「多趟 kernel → 融合 → 一次读入」这条优化主线的完美教材。我们从一个「三个 kernel 各扫一遍数据」的版本出发，一步步融合成单 kernel，最终把有效带宽从 **1000 GB/s 提到 2861 GB/s（2.86 倍）**，并引出大名鼎鼎的 **online softmax**。

配套代码：[`code/kernel-opt/07-softmax/softmax.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/07-softmax/softmax.cu)。前一篇见 {{< relref "cuda-kernel-opt-06-reduction" >}}。

---

## 1. Softmax 长什么样

对矩阵的**每一行**做 softmax：

$$
\text{softmax}(x)_j = \frac{\exp(x_j - \max(x))}{\sum_j \exp(x_j - \max(x))}
$$

其中减 `max` 是为了数值稳定（防止 `exp` 溢出）。行内数据必须全部参与 `max` 和 `sum` 两个归约，所以这是一个**「按行归约 + 逐元素」**的算子，天然要读同一行好几遍。

本文用 `rows=4096, cols=8192`（矩阵 134 MB，工作集 268 MB > 52 MB L2）。注意 `cols` 的每一行只有 32 KB，**能整个放进 shared memory**——这是后面优化的关键前提。

## 2. 版本 0：三个 kernel，各扫一遍

最直白的实现是拆成三步、三个 kernel：

```cuda
k_row_max    <<<rows, 256>>>(x, maxes, rows, cols);              // 每行求 max
k_row_expsum <<<rows, 256>>>(x, maxes, sums,  rows, cols);       // 每行求 exp 和
k_row_norm   <<<grid,  256>>>(x, maxes, sums,  y,    rows, cols); // y = exp(x-max)/sum
```

其中前两个是「block-per-row」的归约（用的就是第 06 篇的 warp shuffle + smem），第三个是逐元素。它的数据搬运账是这样的：

```
读 x（求 max） + 读 x（求 exp 和） + 读 x（归一化） + 写 y   = 4 趟全量读写
```

有效数据的下限是「读一遍 x、写一遍 y」共 2 趟，这里却搬了 4 趟——**2 倍的无谓搬运**，还没算三次 kernel launch 的开销。实测 **1000 GB/s（29.8%）**。

## 3. 版本 1：融合成单 kernel

既然三个 kernel 都在读同一份 `x`，那就让**一个 block 负责一整行**，把三步合进一个 kernel：

```cuda
__global__ void softmax_fused(const float* x, float* y, int rows, int cols) {
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  float* q = y + (size_t)row * cols;
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, p[c]);   // 1) 求 max
  m = block_max(m, sm);                                                       // block 归约
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += __expf(p[c] - m); // 2) 求 exp 和
  s = block_sum(s, sm);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) q[c] = __expf(p[c] - m) * inv; // 3) 归一化
}
```

数据搬运从 4 趟降到 **3 趟**（读 x 两遍 + 写 y 一遍），少了 launch、也少了一趟读。实测 **1255 GB/s（37.4%）**，提升 1.26 倍。

但为什么只有 37%？`ncu` 里 `DRAM Throughput 59%`、`L2 Hit Rate 39%`——第二次读 `x` 时，指望它命中 L2，可是同时有几千个 block 在跑，各自的行互相挤占 L2，命中率并不高。**「读两遍」这个浪费并没有被缓存完全吃掉**。

## 4. 版本 2：缓存进 shared memory，只读一遍

既然一行只有 32 KB、装得下，就干脆在读第一遍时把它**缓存到 shared memory**，后面所有操作都从片上读：

```cuda
template <int BLOCK>
__global__ void softmax_smem(const float* x, float* y, int rows, int cols) {
  extern __shared__ float sx[];          // cols 个 float
  const int row = blockIdx.x;
  const float* p = x + (size_t)row * cols;
  float* q = y + (size_t)row * cols;
  for (int c = threadIdx.x; c < cols; c += BLOCK) sx[c] = p[c];   // 全局读 x，一遍
  __syncthreads();
  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += BLOCK) m = fmaxf(m, sx[c]);  // 从 smem 求 max
  m = block_max(m, sm);
  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += BLOCK) s += __expf(sx[c] - m);
  s = block_sum(s, sm);
  const float inv = 1.f / s;
  for (int c = threadIdx.x; c < cols; c += BLOCK) q[c] = __expf(sx[c] - m) * inv;  // 全局写 y，一遍
}
```

全局内存的搬运降到理论下限：**读 x 一遍 + 写 y 一遍**。实测 **2861 GB/s（85.3%）**，比不缓存的融合版快 **2.28 倍**。

## 5. 实测对比

```
rows=4096 cols=8192, matrix 134.2 MB, working set (x+y) 268.4 MB

multipass (3 kernels)          0.2684 ms    1000.3 GB/s  ( 29.8% of peak)
fused (1 block/row)            0.2138 ms    1255.4 GB/s  ( 37.4% of peak)
fused + smem cache             0.0938 ms    2861.0 GB/s  ( 85.3% of peak)

speedup fused/multipass = 1.26x, cache/fused = 2.28x
```

（有效带宽按「读一遍 x + 写一遍 y = 2×134 MB」折算。）

| 版本 | 时间 | 有效带宽 | 占峰值 | 全局内存搬运 | 关键改动 |
|---|---|---|---|---|---|
| multipass | 0.2684 ms | 1000 GB/s | 29.8% | 4 趟 | 3 个 kernel，各扫一遍 |
| fused | 0.2138 ms | 1255 GB/s | 37.4% | 3 趟 | 合进一个 kernel，block 归约 |
| fused + smem | 0.0938 ms | 2861 GB/s | 85.3% | **2 趟（下限）** | 整行缓存进 shared memory |

`ncu` 的 SOL 把瓶颈变化看得很清楚：

```
softmax_fused :  DRAM 59.1%   L2 Hit 39.0%   Duration 210 us
softmax_smem  :  DRAM 83.4%   L2 80.8%       Duration  88 us
multipass     :  k_row_max 63.5%(64us) + k_row_expsum 61.2%(67us) + k_row_norm 56.5%(134us)
```

- multipass 的三个 kernel 各自都只跑到 56~64%，而且**时间累加**；
- fused 的第二遍读 `x` 没吃上 L2（命中率仅 39%），DRAM 只有 59%；
- smem 版本一次读入、DRAM 83%，已经把带宽吃满。

## 6. 再往前：online softmax

smem 版本虽好，但有个前提：**整行放得下 shared memory**。如果一行长到放不下（比如 attention 里很长的 KV 序列），或者我们根本不想让整个 block 同步等一行，该怎么办？

答案是 **online softmax**（也是 FlashAttention 的基石）：不预先知道 `max`，而是**边扫描边更新**一个「运行最大值 `m`」和「运行指数和 `l`」：

$$
m_{\text{new}} = \max(m, x_j),\qquad
l_{\text{new}} = l \cdot e^{m - m_{\text{new}}} + e^{x_j - m_{\text{new}}}
$$

每读到一个新元素，就把之前的和「按新的最大值重新缩放」。这样 **max 和 sum 一趟就算完**，不需要先把整行读一遍求 max。对本文这种「行能缓存」的场景，它是锦上添花；但对 attention 这种「不能缓存整行」的场景，它是唯一的解法——这也解释了为什么 FlashAttention 能一遍扫过 K/V 就得到结果。

## 7. LayerNorm 是同一个套路

**LayerNorm** 的优化路径和 softmax 一模一样：

- 先求每行均值 `μ` 和方差 `σ²`（两个归约）；
- 再逐元素 `(x-μ)/√(σ²+ε)`，加上仿射 `γ, β`。

所以配方也是：**block-per-row + warp shuffle 归约 + 整行缓存进 shared memory + 一遍读一遍写**。把 softmax 里「求 max/exp 和」换成「求和/平方和」，代码骨架几乎不动。会了 softmax，LayerNorm/RMSNorm 就是换个归约目标。

> 顺带一提：Trim 后的 LayerNorm/RMSNorm 在训练里往往是「内存受限的逐元素算子」，真正贵的是那几次全量读写；融合进前一个算子（如 fused residual + norm）比优化归约本身更值钱——这也是 Transformer Engine 那篇 [TE roofline 分析]({{< relref "te-perf-roofline" >}}) 的结论。

## 小结

- Softmax 是「按行归约 + 逐元素」，优化主线是**减少全量读写趟数**。
- 三个独立 kernel → 4 趟搬运（29.8%）；融合单 kernel → 3 趟（37.4%）；**整行缓存进 shared memory → 2 趟下限（85.3%）**。
- 关键收益来自「**全局内存只读一遍**」，而不是把归约写得更花哨。
- `online softmax` 把「求 max」和「求 sum」合并成一趟，是不缓存整行时的解法，也是 FlashAttention 的基础。
- LayerNorm/RMSNorm 是同一套路，换个归约量即可。

到这里，前七篇覆盖了「内存受限算子」的完整武器库。从下一篇开始进入**计算受限**的世界：**GEMM**——从朴素三重循环，到 shared-memory tiling，再到逼近 cuBLAS。

> 下一篇：CUDA 算子调优（八）· GEMM 入门：从 naive 到 tiled
