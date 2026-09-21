---
title: "CUDA 算子调优（十三）：Tensor Core 入门 — 从 WMMA 到裸 mma.m16n8k16 + ldmatrix"
date: 2026-09-21
draft: false
weight: 13
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "Tensor Core", "WMMA", "mma", "ldmatrix", "m16n8k16", "BF16", "bank conflict", "GEMM", "系列"]
categories: ["算子开发"]
---

上一篇（{{< relref "cuda-kernel-opt-12-async-pipeline" >}}）用 cp.async 多级流水线把第 09 篇的 fp32 GEMM 推到 **35.84 TFLOPS / 53.6% 峰值**，结论是：瓶颈已经从「等访存」转移到「SM 计算/发射本身」，继续堆流水线深度没有意义。这一篇换执行单元——上 **Tensor Core**。

目标是搞明白三件事，并各写一个能跑的版本：

1. **WMMA API** 是什么（`nvcuda::wmma` 的 fragment 抽象），能拿到多少；
2. **裸 PTX `mma.sync.aligned.m16n8k16`** 怎么用：A/B/C 片段在每个线程手里到底长什么样；
3. **`ldmatrix`** 怎么把共享内存高效喂给 Tensor Core，以及为什么**共享内存的行距 padding** 是这一步的胜负手。

配套代码 [`code/kernel-opt/13-tensor-core/tensor_core.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/13-tensor-core/tensor_core.cu)，原始输出与 ncu 数据同目录。硬件仍是 H100 80GB HBM3（132 SM，BF16 Tensor Core dense 峰值 **989 TFLOPS**）。

核心结论先放这里（M=N=K=2048，BF16 输入、FP32 累加）：

| 版本 | 做法 | 耗时 | 算力 | %bf16 峰值 |
|---|---|---|---|---|
| `wmma` | WMMA API，smem 分块，无流水 | 0.2775 ms | 61.91 TFLOPS | 6.3% |
| `wmma_pipe` | WMMA + cp.async 双缓冲 | 0.2358 ms | 72.85 TFLOPS | 7.4% |
| `mma` | 裸 `m16n8k16` + `ldmatrix` | 0.1497 ms | 114.76 TFLOPS | 11.6% |
| `mma_pipe` | 裸 mma + ldmatrix + cp.async 双缓冲 | **0.0777 ms** | **220.98 TFLOPS** | **22.3%** |

同口径 **cuBLAS BF16** 是 0.0255 ms / **672.70 TFLOPS / 68.0%**。也就是说，从一个能跑的入门实现出发，我们做到了 cuBLAS 的 **32.8%**；相比第 12 篇 fp32 的最佳（35.84 TFLOPS），绝对算力是 **6.2 倍**。

一个反直觉的点值得先说：**fp32 版本能到峰值的 53.6%，TC 版本却只有 22.3%**。不是因为 TC 慢，而是因为 TC 峰值太高（989 vs 66.9 TFLOPS），而一个朴素实现卡在 L2 带宽上，离吃满 Tensor Core 还差得远——这正是下一篇要继续调的地方。

---

## 1. 背景：Tensor Core 与 `m16n8k16`

普通 CUDA core 一条 FMA 指令算 1 次乘加；Tensor Core 一条 `mma`（matrix multiply-accumulate）指令在**一个指令周期内**完成一个小矩阵乘：

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        D(16×8) = A(16×16) × B(16×8) + C(16×8)
```

- `m16n8k16`：输出 16×8，K 维 16；bf16 输入（16 bit）、fp32 累加。
- `.row.col`：A 按行主序、B 按列主序参与。
- **一条指令 16×8×16×2 = 4096 FLOP**，而它由一整个 warp 的 32 个线程协作完成——每个线程只持有这个矩阵的一小片。

H100 的 Tensor Core 峰值就是用「指令吞吐 × 4096 FLOP × 132 SM × 频率」堆出来的。要让它跑起来，核心难点不是算，而是**怎么把 A/B 的每个元素准确地送到该送的那个线程的寄存器里**。

---

## 2. WMMA：把"送数据"这件事交给编译器

CUDA 提供了一个高层 API `nvcuda::wmma`，用 fragment 抽象隐藏了寄存器布局：

```cpp
#include <mma.h>
using namespace nvcuda;

wmma::fragment<wmma::matrix_a, 16,16,16, __nv_bfloat16, wmma::row_major> a_frag;
wmma::fragment<wmma::matrix_b, 16,16,16, __nv_bfloat16, wmma::row_major> b_frag;
wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;

wmma::fill_fragment(c_frag, 0.f);
wmma::load_matrix_sync(a_frag, ptr_a, lda);   // 从内存/共享内存加载
wmma::load_matrix_sync(b_frag, ptr_b, ldb);
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
wmma::store_matrix_sync(ptr_c, c_frag, ldc, wmma::mem_row_major);
```

注意它叫 `load_matrix_sync`——这不是普通的拷贝：它会把一个 16×16 的矩阵**按 mma 要求的布局**分发到 32 个线程的寄存器中。用户完全不用关心哪个线程拿哪几个元素。

### 实测：WMMA + 双缓冲

沿用 09/12 的分块结构：block 算 128×128，BK=32，256 线程（8 warp，4×2 布局，每 warp 算 32×64）。`gemm_wmma` 每轮 `load_tiles` → `__syncthreads` → 16 次 `mma_sync` → `__syncthreads`；`gemm_wmma_pipe` 把载入换成 cp.async 双缓冲（和 12 一样的套路，见 `tensor_core.cu:167`）。

```
wmma        0.2775 ms     61.91 TFLOPS  ( 6.3% of bf16 TC peak)
wmma_pipe   0.2358 ms     72.85 TFLOPS  ( 7.4% of bf16 TC peak)
```

双缓冲有提升（+18%），但**整体只有峰值的 6~7%**。ncu 看一眼就知道问题：`gemm_wmma` 的 `Compute (SM) Throughput` 只有 17%，而 `L1/TEX Cache Throughput` 高达 77%——**瓶颈在共享内存搬运，根本不在 Tensor Core**。

问题出在 `load_matrix_sync` 的抽象上：每个 fragment 都要从共享内存里读 16×16 个元素，一个 warp 每轮要读 2 个 A fragment + 4 个 B fragment。数据被反复读，共享内存带宽被打满。要控制这件事，就得自己写 mma，自己决定怎么搬。

---

## 3. 裸 `m16n8k16`：每个线程手里到底有什么

裸 PTX 的 mma 长这样（`tensor_core.cu:229`）：

```cpp
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
    : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
```

- **A 片段**：4 个 32-bit 寄存器，每个打包 2 个 bf16，共 8 个元素；
- **B 片段**：2 个寄存器，4 个元素；
- **C 片段**：4 个 fp32。

每个线程持有的是矩阵的哪个位置？设 warp 内 `lane = 0..31`，`group = lane >> 2`（0..7），`tig = lane & 3`（0..3）：

| 片段 | 寄存器 | 元素 (行, 列) |
|---|---|---|
| A (16×16) | `a0` | (group, 2·tig), (group, 2·tig+1) |
| | `a1` | (group+8, 2·tig), (group+8, 2·tig+1) |
| | `a2` | (group, 2·tig+8), (group, 2·tig+8+1) |
| | `a3` | (group+8, 2·tig+8), (group+8, 2·tig+8+1) |
| B (16×8, K×N) | `b0` | (2·tig, group), (2·tig+1, group) |
| | `b1` | (2·tig+8, group), (2·tig+9, group) |
| C (16×8) | `c0` | (group, 2·tig) |
| | `c1` | (group, 2·tig+1) |
| | `c2` | (group+8, 2·tig) |
| | `c3` | (group+8, 2·tig+1) |

这张表就是 Tensor Core 编程的全部「暗号」。你（或 `ldmatrix`）必须按这个映射，把 smem 里的值摆到正确的寄存器里。

第一个版本我直接照着这张表用**标量 LDS** 一个个拼（`pack2(As[r][c], As[r][c+1])`）。功能对，但性能只有 60 多 TFLOPS，和 WMMA 差不多——因为每个元素都要单独从共享内存读一次。

---

## 4. `ldmatrix`：一条指令搬 4 个 8×8

`ldmatrix` 是 sm_75 起专门为 Tensor Core 设计的共享内存加载指令：

```
ldmatrix.sync.aligned.m8n8.x4.shared.b16 {d0,d1,d2,d3}, [addr];
```

- 一条指令加载 **4 个 8×8 的 16-bit 矩阵**（共 512 字节），分发给整个 warp；
- 每个线程提供一行（16 字节）的地址：lane 0-7 给矩阵 0 的 8 行，lane 8-15 给矩阵 1，以此类推；
- 加载完成后，每个线程恰好拿到 2 个连续元素——**这正是 mma 片段要的布局**。

也就是说，前面那张映射表，硬件替你做了。取 A 的 16×16 片段用一条 `ldmatrix.x4`：

```cpp
const int row = (lane & 15);          // lane 0-15 → 行 0-15（列 0..7）
const int col = (lane >> 4) * 8;      // lane 16-31 → 列 8..15
ldmatrix_x4(&As[warp_m + row][kk*16 + col], a);
```

B 需要的是 `.col` 布局（K×N 的列主序），用 **`ldmatrix.x4.trans`**：地址仍指向 `Bs[K][N]` 的行，硬件在加载时顺便转置，得到的就是 B 片段。

### 关键：bank conflict 把收益吃光了

第一版 ldmatrix 实现（行距不加 padding）实测：

```
mma_pipe    0.2254 ms     76.23 TFLOPS  (  7.7% of bf16 TC peak)
```

只比标量版好一点点。ncu 的 bank conflict 计数给出了答案：

```
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 35651584   (3564 万次)
```

**共享内存 bank 是按 4 字节（32-bit）编的，32 个 bank 循环。** 假设行主序数组行距是 `L` 个 bf16，则相邻两行的地址差是 `L×2` 字节，折算成 bank 偏移 `(L×2/4) mod 32 = (L/2) mod 32`。ldmatrix 一次读 8 行，如果这 8 行的起始 bank 撞在一起，就发生冲突。

| 矩阵 | 行距 (bf16) | 行距(字节) | bank 偏移 | 8 行的 bank | 冲突 |
|---|---|---|---|---|---|
| B 无 padding | 128 | 256 | 64 mod 32 = **0** | 8 行全在 bank 0 | **8 路** |
| B 加 8 | 136 | 272 | 68 mod 32 = **4** | 0,4,8,…,28 | 无 |
| A 无 padding (BK=32) | 32 | 64 | 16 mod 32 = 16 | 0,16,0,16… | 2 路 |
| A 加 8 | 40 | 80 | 20 mod 32 = 20 | 0,20,8,28,… | 无 |

解法很土但极其有效：**把 smem 数组的行距从 `BN` 改成 `BN+8`、`BK` 改成 `BK+8`**（`tensor_core.cu:39`）：

```cpp
constexpr int ASP = BK + 8;   // A smem 行距
constexpr int BNP = BN + 8;   // B smem 行距
```

实测（BK=32，同一份代码只改 padding）：

| | bank conflicts | `mma` | `mma_pipe` |
|---|---|---|---|
| 无 padding | 35,651,584 | 65.77 TFLOPS | 76.23 TFLOPS |
| 加 8 padding | **0** | 114.76 TFLOPS | **220.98 TFLOPS** |

**`mma_pipe` 从 76 → 221 TFLOPS，2.9 倍，全靠消掉 bank conflict。** 这是本篇最大的一个 takeaway：到了 Tensor Core 时代，性能瓶颈经常不在"算"，而在"喂"——而喂的手段（ldmatrix + padding）和经典 GEMM 优化里的套路是一脉相承的。

---

## 5. 完整实测对比

`scripts/run.sh 13-tensor-core/tensor_core.cu`，M=N=K=2048（17.18 GFLOP），原始输出见 `tensor_core.out.txt`：

| 版本 | 寄存器 | smem/block | 耗时 | 算力 | %bf16 峰值 |
|---|---|---|---|---|---|
| `wmma` | 96 | 16 KB | 0.2775 ms | 61.91 TFLOPS | 6.3% |
| `wmma_pipe` | 96 | 32 KB | 0.2358 ms | 72.85 TFLOPS | 7.4% |
| `mma` | 96 | 19 KB | 0.1497 ms | 114.76 TFLOPS | 11.6% |
| `mma_pipe` | 128 | 38 KB | **0.0777 ms** | **220.98 TFLOPS** | **22.3%** |
| cuBLAS BF16（参考） | — | — | 0.0255 ms | 672.70 TFLOPS | 68.0% |

（`nvcc -Xptxas -v`，`ptxas_regs.out.txt`；cuBLAS 参考见 `cublas_bf16_ref.out.txt`。第 12 篇 fp32 最佳为 35.84 TFLOPS / 53.6%，同为 128×128 分块、256 线程。）

读法：

- **WMMA → 裸 mma + ldmatrix**：74 → 221 TFLOPS，**3 倍**。高层 API 的 `load_matrix_sync` 在这里是负优化。
- **双缓冲**：无论 WMMA 还是裸 mma，cp.async 双缓冲都能再拿 15%～20%（`mma` 115 → `mma_pipe` 221，其实这里还叠加了流水线对 ldmatrix 的隐藏）。
- **离 cuBLAS 还有距离**：我们 221 TFLOPS = cuBLAS 的 32.8%。cuBLAS 用更大的分块（提高算术强度）、更细的流水、以及 Hopper 的 `wgmma`（整 warpgroup 的异步 TC 指令）。

---

## 6. ncu 证据：瓶颈已经从 L1 挪到 L2

对最终的 `mma_pipe` 采 SOL（`ncu_sol.out.txt`，`--set basic`）：

| 指标 | `gemm_wmma`（无 padding） | `mma_pipe`（有 padding） |
|---|---|---|
| Compute (SM) Throughput | 17.3% | **37.2%** |
| L1/TEX Cache Throughput | **77.4%** | 55.7% |
| L2 Cache Throughput | 13.4% | **69.9%** |
| DRAM Throughput | 2.0% | 7.3% |
| 耗时 | 289.8 µs | **78.7 µs** |

- WMMA 版本卡在 **L1/TEX 77%**（共享内存搬运），计算只有 17%；
- padding 之后 L1 降到 56%，**计算翻倍到 37%**，而 **L2 变成新的头号瓶颈（69.9%）**。DRAM 仍然只有 7%——2048³ 的数据基本驻留 L2，所以限制来自 L2 到 SM 的带宽，而不是 HBM。
- 要把 22% 继续往上推，方向就明确了：**更大分块**（BM/BN 加大 → 每份 A/B 被更多输出复用 → 降低 L2 流量）、**更深的流水**、以及 **`wgmma`（Hopper warpgroup MMA）**。

调度侧的 stall 也印证了这一点（`ncu_stalls.out.txt`）：

| 指标 | `mma_pipe` |
|---|---|
| Warp Cycles Per Issued Instruction | 9.23 |
| Issue Slots Busy | 37.2% |
| No Eligible | 59.7% |
| Eligible Warps / Scheduler | 0.63 |
| Executed IPC | 1.61 |

`Executed IPC` 1.61、每调度器只有 0.63 个可发射 warp——kernel 仍有延迟没盖住（寄存器 128 / 256 线程 → 2 block/SM = 25% occupancy）。这也是后续可以继续做文章的地方。

---

## 7. 和 fp32 版本怎么比

| | 第 12 篇 fp32 `pipe4` | 本篇 BF16 `mma_pipe` |
|---|---|---|
| 执行单元 | CUDA core（FMA） | Tensor Core（HMMA） |
| 峰值 | 66.9 TFLOPS | 989 TFLOPS |
| 实测 | 35.84 TFLOPS | **220.98 TFLOPS** |
| 占峰值 | 53.6% | 22.3% |
| 绝对算力 | 1× | **6.2×** |

**绝对算力翻了 6 倍，但占峰值的比例反而降了。** 这很好地说明了：算力峰值越高，「喂饱它」就越难。fp32 的 kernel 已经接近把 CUDA core 榨干，而 TC kernel 还饿着——差的就是数据供给和调度。

---

## 小结

- **Tensor Core 一条 `m16n8k16` 指令 = 4096 FLOP**，由 32 个线程协作；难点从来不是算，而是按片段的寄存器映射把数据准确送到位。
- **WMMA API 能跑但不够快**：`load_matrix_sync` 的抽象让共享内存流量失控，只有峰值 6~7%。
- **裸 `mma.m16n8k16` + `ldmatrix` 才能控制数据流**：`ldmatrix` 一条指令搬 4 个 8×8，天然对齐 mma 片段。
- **ldmatrix 的 bank conflict 是隐藏杀手**：B 无 padding 时 8 行全部落在同一 bank，实测 3565 万次冲突。**给 smem 行距加 8 个元素**后冲突归零，`mma_pipe` 从 76 → 221 TFLOPS（2.9×）。
- **瓶颈随优化不断转移**：WMMA 卡 L1/TEX（77%），padding 后计算利用率翻倍、瓶颈变成 L2（70%），DRAM 始终只有 7%。
- 最佳 **220.98 TFLOPS（22.3% 峰值）**，是 cuBLAS BF16（672.70）的 32.8%，是第 12 篇 fp32（35.84）的 6.2 倍。
- **下一步**：更大分块、更深流水、`wgmma` / TMA，以及把 GEMM 和 bias/激活融合的 epilogue。

> 下一篇：CUDA 算子调优（十四）· 融合与 epilogue——给 GEMM 加上 bias 和激活，串起一个生产算子的完整形态。
