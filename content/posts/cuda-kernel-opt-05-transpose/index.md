---
title: "CUDA 算子调优（五）：共享内存与 bank conflict — 矩阵转置"
date: 2026-09-21
draft: false
weight: 5
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "共享内存", "bank conflict", "矩阵转置", "ncu", "系列"]
categories: ["算子开发"]
---

上一篇学会了「让相邻线程访问相邻地址」。但有一类算子天生做不到——**矩阵转置**：读的时候按行连续，写的时候就得按列跳着写；不管怎么排，总有一边不合并。这一篇的解法是 CUDA 优化里最经典的一招：**用 shared memory 做中转，把不合并的那一边挪到片上**；再用一个「加 1 个 padding」的小技巧消掉随之而来的 **bank conflict**。实测下来，转置从 **466 GB/s 提升到 1651 GB/s（3.5 倍）**。

配套代码：[`code/kernel-opt/05-transpose/transpose.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/05-transpose/transpose.cu)。前一篇见 {{< relref "cuda-kernel-opt-04-coalescing" >}}。

---

## 1. 问题：转置总有一边不合并

转置 `B[j][i] = A[i][j]`，A 形状 `M×N` 行主序，B 形状 `N×M` 行主序。最直接的写法：

```cuda
__global__ void transpose_naive(const float* A, float* B, int M, int N) {
  int x = blockIdx.x * 32 + threadIdx.x;   // A 的列，也是 B 的行
  int y = blockIdx.y * 32 + threadIdx.y;   // A 的行，也是 B 的列
  if (x < N && y < M) B[x * M + y] = A[y * N + x];
}
```

- 读 `A[y*N+x]`：`threadIdx.x` 对应 `x`，**相邻线程读相邻地址 → 合并**；
- 写 `B[x*M+y]`：`threadIdx.x` 改变的是 `x`，地址跳 `M` 个元素，**一个 warp 里地址散开 → 不合并**（第 04 篇演示过这种跨行访问有多惨）。

实测 naive 版本只有 **466 GB/s（13.9%）**，和上一篇的列优先 copy（9.6%）一个量级——瓶颈就是那个不合并的写。

## 2. 用 shared memory 中转

思路：既然「读要合并」和「写要合并」在全局内存里打架，那就把数据先搬到 **shared memory**（片上、每个 block 私有），在片上完成转置，再合并地写回全局：

1. **读**：每个线程把自己负责的 `A[y][x]` 合并地读进来，存到 `tile[y][x]`；
2. `__syncthreads()` 等整个 tile 到齐；
3. **写**：让线程换个坐标再读 `tile`，保证写回全局时 `threadIdx.x` 对应 B 的列（连续地址）。

```cuda
__global__ void transpose_smem(const float* A, float* B, int M, int N) {
  __shared__ float tile[32][32];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < N && y < M) tile[threadIdx.y][threadIdx.x] = A[y * N + x];  // 合并读
  __syncthreads();
  // 交换线程坐标：blockIdx.x 决定 B 的行块，blockIdx.y 决定 B 的列块
  int x2 = blockIdx.x * 32 + threadIdx.y;   // B 的行
  int y2 = blockIdx.y * 32 + threadIdx.x;   // B 的列
  if (x2 < N && y2 < M) B[x2 * M + y2] = tile[threadIdx.x][threadIdx.y];  // 合并写
}
```

这样全局内存的读和写**都是合并的**。先看效果（`transpose_smem_nopad`，即 `tile[32][32]`）：

```
naive                    466.4 GB/s  (13.9% of peak)
smem no-pad              984.6 GB/s  (29.4% of peak)
```

翻倍了，但离峰值还很远。问题出在第 3 步读 `tile` 的方式上——这就是 **bank conflict**。

## 3. Bank conflict：shared memory 的「合并」问题

Shared memory 被划分成 **32 个 bank**，每个 bank 宽 4 字节。地址到 bank 的映射是：

```
bank = (元素下标) % 32
```

一个 warp 的 32 个线程访问 shared memory 时，硬件会在一个周期内并行服务这 32 个请求——**前提是它们落在不同的 bank**。如果多个线程访问同一个 bank 的不同地址（同一地址广播除外），就会**串行化**，这就是 bank conflict。

看第 3 步：线程读 `tile[threadIdx.x][threadIdx.y]`，即行号是 `threadIdx.x`、列号是 `threadIdx.y`。对固定 `threadIdx.y`、变化的 `threadIdx.x = 0..31`：

```
元素下标 = threadIdx.x * 32 + threadIdx.y
bank      = (threadIdx.x * 32 + threadIdx.y) % 32
          = threadIdx.y % 32            ← 与 threadIdx.x 无关，恒为同一个值！
```

**32 个线程全部落到同一个 bank**，于是 32 路冲突，一次访问被拆成 32 个周期。这正好抵消了共享内存中转带来的收益。

图解（`tile[32][32]`，按列读）：

```
        列 threadIdx.y（固定）
        │
 行 t0 │ ●  → 下标 0*32+ty → bank ty
 行 t1 │ ●  → 下标 1*32+ty → bank ty
 行 t2 │ ●  → bank ty
 ...   │
 行 t31│ ●  → bank ty          ← 32 个线程抢同一个 bank，32-way conflict
```

### 解法：padding

把 `tile` 的第二维从 32 改成 **33**（`tile[32][33]`），每行末尾多留一个空位。下标变成 `threadIdx.x * 33 + threadIdx.y`：

```
bank = (threadIdx.x * 33 + threadIdx.y) % 32
     = (threadIdx.x + threadIdx.y) % 32     ← 随 threadIdx.x 变化，32 个线程落在 32 个不同 bank
```

冲突消失。代价是每行多 1 个 float 的 shared memory（32×33×4 = 4224 B），几乎可以忽略。

```cuda
__shared__ float tile[32][TILE + 1];   // padding：+1 错开 bank
```

## 4. 实测与 ncu 证据

```
M=8192 N=8192, matrix 268.4 MB, working set (A+B) 536.9 MB

naive                          1.1511 ms     466.4 GB/s  ( 13.9% of peak)
smem no-pad                    0.5453 ms     984.6 GB/s  ( 29.4% of peak)
smem padded                    0.3252 ms    1650.9 GB/s  ( 49.2% of peak)

speedup smem_pad/naive = 3.54x, smem_pad/smem_nopad = 1.68x
```

padding 相对无 padding 又快了 **1.68 倍**。用 `ncu` 数一下 shared memory 的 bank conflict（只看 load）：

```
transpose_smem_nopad :  bank_conflicts(ld) = 65,310,220   wavefronts(ld) = 67,407,372
transpose_smem_pad   :  bank_conflicts(ld) =    396,327   wavefronts(ld) =  2,493,479
```

`wavefronts` 是实际发生的访问波次。无 padding 时 6740 万次波次里 6531 万次是冲突造成的（**97% 在做无用功**）；加 padding 后冲突降到 40 万，几乎全是有效访问。kernel 时间也从 538 µs 降到 322 µs。

> 小提醒：bank conflict 只在**访问模式**层面产生，加 padding 改变的是 shared memory 的物理布局，不改变数据本身，所以完全不影响正确性。

## 5. 还能更快吗？

padded 版本达到 49%，对「教学版」够用了，但离 80%+ 还有空间。后续可以做的（留到后面几篇）：

- **每个线程处理多个元素**（如 4×4），让每次访问搬更多数据、摊薄地址计算与同步开销；
- **向量化**：shared memory 和全局内存都用 `float4`，一个线程一次搬 16B；
- 用 **异步拷贝 `cp.async`** 把「读下一块」和「算当前块」重叠（第 12 篇）；
- 更大 tile + 更好的 swizzle 排布（CUTLASS 里大量使用）。

## 小结

- 转置的读、写必然有一边不合并；**shared memory 中转**可以让全局内存两边都合并。
- Shared memory 分 32 个 bank，`bank = 元素下标 % 32`；一个 warp 内多线程命中同一 bank 即 **bank conflict**，会被串行化。
- 列访问 `tile[32][32]` 时所有线程落同一 bank（32 路冲突）；**padding 成 `tile[32][33]`** 即可错开。
- 实测：466 → 985（smem）→ **1651 GB/s（+padding）**，合计 **3.54×**；ncu 显示冲突从 6531 万降到 40 万。
- 判断 bank conflict 的 ncu 指标：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld/st.sum`。

下一篇换一类完全不同的算子：**归约（reduction）**。它挑战的是「线程之间怎么协作求一个和」，会引出 warp shuffle 和「怎么用多 block 归约」，也是 softmax / LayerNorm 的底座。

> 下一篇：CUDA 算子调优（六）· 归约与 warp shuffle
