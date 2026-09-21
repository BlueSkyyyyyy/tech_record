---
title: "CUDA 算子调优（十七）：MuonClip 的 Newton–Schulz 正交化 — 15 个 GEMM 的链式算子"
date: 2026-09-21
draft: false
weight: 17
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "Muon", "MuonClip", "优化器", "Newton-Schulz", "正交化", "Tensor Core", "GEMM", "Kimi", "系列"]
categories: ["算子开发"]
---

前十六篇我们把注意力类算子（MLA）推到了单 kernel 融合的 **170 TFLOPS**，也反复看到「访存复用」和「Tensor Core 布局」两条主线。从这一篇开始进入优化器一侧的模型算子：**Muon / MuonClip 的 Newton–Schulz（NS）正交化**。

Muon 对 2D 以上参数做更新时，不是像 Adam 那样逐元素缩放，而是先把动量矩阵「正交化」成近似的正交极因子。Kimi K2 系列就是用 MuonClip（Muon + QK-Clip）训出来的。这个正交化在 GPU 上就是 **5 步 Newton–Schulz 迭代、每步 3 个矩阵乘法**——一个天然的「GEMM 链式算子」，15 个 `N×N×N` GEMM，总计 `30N³` FLOPs。

这一篇回答三个问题：

1. 这个算子到底是**算力受限还是访存受限**？（roofline 先算清楚）
2. 怎么把它写成一个**自研 Tensor Core kernel**，并把 `b·A + c·(A@A)`、`a·X + (M@X)` 这类「`f·x + g(GEMM)`」融合进 GEMM 的 epilogue？
3. **离 cuBLAS 还有多远**？差在哪里、下一步怎么补？

先给结论（H100 80GB HBM3，bf16 dense 峰值 989 TFLOPS；`N=4096`，5 步，`30N³=2.06 TFLOP`）：

| 路径 | 做法 | 耗时 | 算力 | %峰值 | 相对 cuBLAS |
|---|---|---|---|---|---|
| `ref` | cuBLAS **fp32** 同算法（正确性参考） | 41.53 ms | 49.6 TFLOPS | × | — |
| `custom unfused` | 自研 GEMM + 独立 `axpby` kernel | 8.93 ms | 231 TFLOPS | 23.3% | 0.44× |
| **`custom fused`** | 自研 GEMM，**`f·x+g` 融进 epilogue**（cfg1） | **8.44 ms** | **244 TFLOPS** | **24.7%** | **0.46×** |
| `cublas` | cuBLAS bf16 GEMM + 独立 elementwise | 3.89 ms | 529 TFLOPS | 53.5% | 1.00× |

单看一个 4096³ GEMM：自研 **269 TFLOPS**，cuBLAS **883 TFLOPS**（89% 峰值）——自研是 cuBLAS 的 **30.5%**。融合 epilogue 省掉 3 趟 elementwise 读写，端到端快 **~5%**（8.93→8.44 ms）；但真正的瓶颈是自研 GEMM 本身（`ncu`：**L2 吞吐 76%、occupancy 23.8%、Compute 45%**），下一篇（wgmma + TMA）是补齐方向。

配套代码 [`code/kernel-opt/17-muonclip-ns/ns_muonclip.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/17-muonclip-ns/ns_muonclip.cu)，原始输出见同目录 `ns_all.out.txt` / `ns_sweep.out.txt` / `ns_single.out.txt` / `ns_ncu.out.txt`。

---

## 1. Muon 与 MuonClip：为什么优化器里会有 GEMM

Adam 类优化器对每个参数元素独立更新，天然是逐元素的。Muon（Keller Jordan 等）换了个思路：**对 2D 及以上的参数（权重矩阵）做「正交化」更新**，让更新量的奇异值全部拉到 1，从而对不同尺度的方向一视同仁。它带来更稳、更快收敛，但也把优化器 step 变成了矩阵运算。

Kimi K2 用的 **MuonClip** 是 Muon 的一个稳定化版本：用 Muon 更新矩阵参数，同时用 **QK-Clip** 在训练中途对注意力 logits 做裁剪、防止爆炸。MuonClip 的实现里，对每个 2D 参数 `G`（动量矩阵）做：

$$
\begin{aligned}
X_0 &= G / (\lVert G \rVert_F + \epsilon) \\
\text{重复 } 5 \text{ 次：}\quad
A &= X X^\top \\
B &= b\,A + c\,(A A) \\
X &\leftarrow a\,X + B X
\end{aligned}
$$

系数来自 Muon 论文：`a=3.4445, b=-4.7750, c=2.0315`。这是经典的 Newton–Schulz 迭代，用来逼近矩阵的**正交极因子**（polar factor）。做完后 `X` 近似列正交：`X Xᵀ ≈ I`。

> 代码对照：本地 `~/github/muonclip/src/muon_clip_pytorch.py:8` 的 `newton_schulz()`，以及 `~/github/Emerging-Optimizers` 里的 Muon 实现。本仓库代码系数、迭代次数与之一致。

**真实 shape**（读 `/ssd/models/*/config.json`）：

- **Kimi-K2.6**：`hidden_size=7168`、`intermediate_size=18432`、`moe_intermediate_size=2048`、384 experts、61 层 → Muon 处理的矩阵形如 `[7168,7168]`、`[7168,18432]`、专家 `[7168,2048]`；
- **DeepSeek-V4-Pro**：`hidden_size=7168`、`moe_intermediate_size=3072`、384 experts；
- **ERNIE-4.5-300B**：`hidden_size=8192`。

所以 `N` 在 2048~8192 之间。本文用 `N=2048 / 4096 / 8192` 三个尺寸覆盖。

---

## 2. 先算 roofline：这活是算力受限

一次迭代 3 个 `N×N×N` GEMM，5 次共 **15 个 GEMM**，所以

$$
\text{FLOPs} = 15 \times 2N^3 = 30N^3 .
$$

`N=4096` 时 `30N³ = 2.06 TFLOP`。若纯按 HBM 访存算：每个 GEMM 最少要读 `A`、`B` 各 `2N²` 字节、写 `2N²` 字节，15 个 GEMM ≈ `15×6N²×...`，但矩阵本身只有 `2N² = 32 MB`，工作集远小于算力/带宽的平衡点。把算术强度写成

$$
\text{AI} \approx \frac{30N^3}{\text{DRAM bytes}} = O(N)\ \text{FLOP/byte},
$$

`N=4096` 时是一个非常大的数——**理论上绝对的算力受限**。这也被实测证实：`ncu` 里 `DRAM Throughput` 只有 **13.4%**。

所以优化的核心不是省访存，而是**把 GEMM 本身做快**。这决定了整个实现思路：写一个尽量好的 Tensor Core GEMM，然后把三处 elementwise 加乘融进去。

```
             ┌──────────────────────── 重复 5 次 ────────────────────────┐
X (bf16 N×N) │  G1: A = X @ Xᵀ          ──► A (bf16)                    │
             │  G2: M = b·A + c·(A @ A) ──► M (bf16)   ← epilogue 融合 b·A│
             │  G3: X' = a·X + (M @ X)  ──► X' (bf16)  ← epilogue 融合 a·X│
             └───────────────────────────────────────────────────────────┘
                                   │
                                   └─► 输出 X'（正交化后的更新量）
```

`Xᵀ` 每个迭代物化一次（喂给 G1 的第二个操作数）；`A` 是对称的，所以 `A@A` 两个操作数都是 `A`。

---

## 3. 实现：三条路径对拍

[`ns_muonclip.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/17-muonclip-ns/ns_muonclip.cu) 里实现三条独立路径，互相验证：

1. **`custom`**：自研 bf16 GEMM（`mma.m16n8k16` + `ldmatrix` + `cp.async` 多级流水），带 **可融合 epilogue** `out = sc_c·Cin + sc_d·acc`；
2. **`cublas`**：cuBLAS bf16 GEMM + 独立 `k_axpby_bf16` kernel（「生产实现」口径）；
3. **`ref`**：cuBLAS fp32 走同一算法（计算结果的对拍基准）。

### 3.1 融合 epilogue：把 `f·x+g(GEMM)` 留在寄存器里

`mma.m16n8k16` 的 fp32 累加器，线程内 4 个数 `c0,c1,c2,c3` 的坐标是固定的：

```
c0 → (row = lane/4,     col = 2·(lane%4)    )
c1 → (row = lane/4,     col = 2·(lane%4) + 1)
c2 → (row = lane/4 + 8, col = 2·(lane%4)    )
c3 → (row = lane/4 + 8, col = 2·(lane%4) + 1)
```

因为坐标已知，**结果还在寄存器里时就能顺手做 `f·x + g`**：读一个 `Cin[(r,c)]`（bf16，和 `C` 的写出坐标完全一样，访存合并度一致），算完直接写 bf16 输出。于是 `G2`/`G3` 把原本要单独跑的 `axpby`（读 2 个矩阵 + 写 1 个矩阵）整段消掉。

```cpp
// 16-mla-fused 里用过的同一招；这里推广成通用 epilogue
template <class C>
__device__ void store_acc_epi(bf16* Cout, const bf16* Cin,
                              const float acc[C::MTM][C::MTN][4], int N,
                              int block_row, int block_col, int warp_row, int warp_col,
                              float sc_c, float sc_d) {
  ...
  float v = sc_d * acc[i][j][q];
  if (Cin) v += sc_c * __bfloat162float(Cin[(size_t)r * N + c]);
  Cout[(size_t)r * N + c] = __float2bfloat16(v);
}
```

三个 GEMM 用同一份 kernel，只是 epilogue 参数不同：

| GEMM | 左/右操作数 | `Cin` | `sc_c` | `sc_d` | 结果 |
|---|---|---|---|---|---|
| G1 | `X, Xᵀ` | — | 0 | 1 | `A = X@Xᵀ` |
| G2 | `A, A` | `A` | `b` | `c` | `M = bA + c(A@A)` |
| G3 | `M, X` | `X` | `a` | 1 | `X' = aX + (M@X)` |

（`G2` 里 `A` 对称，左右操作数同源；`Cin` 与输出 `M` 是不同 buffer，没有原地别名问题。）

### 3.2 GEMM 本体：沿用 13/14 篇的 128×128 分块

GEMM 用系列第 13 篇的骨架：每 block 算 `128×128`，走 `mma.m16n8k16` + `ldmatrix`（A 用非转置、B 用 `.trans`），smem 行距 `+8` 消 bank conflict，`cp.async` 多级软流水。为了让「融合」有说服力，epilogue 的改动**不增加寄存器**：实测 `126` 寄存器、`0` spill（和纯 GEMM 同级）。

`ncu` 也验证了 bank conflict 确实被消掉：`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 148472`，相对 2.19 亿条指令可以忽略。

---

## 4. 实测对比

### 4.1 三个尺寸（cfg1 = 128×128×64、3 级流水）

原始输出：`ns_all.out.txt`。

| N | FLOPs | `ref` fp32 | `custom unfused` | **`custom fused`** | `cublas+elementwise` | cuBLAS / custom |
|---|---|---|---|---|---|---|
| 2048 | 0.26 TFLOP | 5.68 ms (45.3 TF) | 1.350 ms (191 TF) | **1.312 ms (196 TF)** | 0.860 ms (300 TF) | 1.53× |
| 4096 | 2.06 TFLOP | 41.53 ms (49.6 TF) | 8.93 ms (231 TF) | **8.44 ms (244 TF)** | 3.89 ms (529 TF) | 2.17× |
| 8192 | 16.5 TFLOP | 325.1 ms (50.7 TF) | 66.4 ms (248 TF) | **65.4 ms (252 TF)** | 29.9 ms (551 TF)* | 2.2× |

- **融合收益**：`custom unfused → fused` 在 N=4096 是 8.93→8.44 ms（**−5.5%**），N=8192 是 66.4→65.4 ms（−1.5%）。尺寸越大，elementwise 那点流量占比越小，融合收益递减——但它是「免费」的。
- **对标 SOTA**：cuBLAS 链在 N=4096 领先自研 **2.17×**；自研融合版是 cuBLAS 链的 **46%**。
- \* N=8192 的 cuBLAS 用时在多次测量里为 26.7~29.9 ms（551~617 TF，受 GPU 时钟 boost 影响），这里是 `ns_all.out.txt` 里的值。

### 4.2 分块配置扫描（N=4096）

原始输出：`ns_sweep.out.txt`。所有配置都扫过，最优是 `BK=64 + 3 级流水`。

| cfg | 分块 (BM×BN×BK) | warp | 流水级 | 耗时 | 算力 |
|---|---|---|---|---|---|
| cfg0 | 128×128×32 | 4×2 | 2 | 8.87 ms | 232 TF |
| **cfg1** | **128×128×64** | **4×2** | **3** | **8.44 ms** | **244 TF** |
| cfg2 | 128×256×32 | 4×4 | 2 | 9.52 ms | 216 TF |
| cfg3 | 256×128×32 | 8×2 | 2 | 9.97 ms | 207 TF |
| cfg4 | 128×128×32 | 2×4 | 3 | 9.02 ms | 229 TF |
| cfg5 | 128×128×128 | 4×2 | 2 | 9.00 ms | 229 TF |
| cfg6 | 128×128×32 | 4×2 | 4 | 9.55 ms | 216 TF |

结论：**加大 BK 到 64 并上 3 级流水最有效**（同步开销摊薄、`cp.async` 预取更深）；单纯把块摊大（128×256 / 256×128）反而更慢——每线程 fp32 累加器翻倍会顶到寄存器墙、occupancy 掉下来。

---

## 5. 瓶颈在哪：ncu 说 L2 + occupancy

对 `cfg1` 的单个 GEMM（`4096³`，`ncu` 实测耗时 522.6 µs，对应 **263 TFLOPS**）采集 `--set full`（原始输出 `ns_ncu.out.txt`）：

| 指标 | 值 |
|---|---|
| Compute (SM) Throughput | 44.8% |
| **L1/TEX Cache Throughput** | **62.3%** |
| **L2 Cache Throughput** | **76.1%** |
| DRAM Throughput | 13.4% |
| Achieved Occupancy | 23.8% |
| Registers / Thread | 126（0 spill） |
| Block Limit（寄存器 / smem） | 2 / 2 |
| Executed IPC | 1.87 |
| Warp Cycles Per Issued Inst | 8.16 |

指纹很清楚：

1. **L2 吞吐 76% 是最高项**，DRAM 只有 13%。说明瓶颈是**片上的 L2 读放大**而不是 HBM。自研分块 `128×128` 下，A 被 `N/BN=32` 个列块重复读、B 被 `M/BM=32` 个行块重复读，每个 GEMM 的全局载入 ≈ `32×(32MB+32MB)=2GB`（全部命中的话走 L2）。
2. **occupancy 只有 23.8%**（寄存器限制每 SM 只能放 2 个 block）。每线程 `2×8×4=64` 个 fp32 累加器 + 操作数/地址寄存器 = 126，想再提 occupancy 就得减累加器，可 tile 又不能缩小，于是卡住。
3. **Compute 只有 45%**——Tensor Core 有一大半时间在等 smem/L2 的数据。

这两点其实是一个硬币的两面：`mma.m16n8k16` 路径下，输入必须经过 `ldmatrix` 从 smem 取，块一大寄存器就爆、块小了 L2 又放大。**要真正跨过 240 TFLOPS，必须换引擎——`wgmma` 直接从 smem 读操作数（绕开 `ldmatrix`），配合 TMA 把 L2 放大压低**。这正是路线图第六部分（31/38）要做的事，也是下面差距的根源。

---

## 6. 正确性与正交性

`ref`（cuBLAS fp32）与自定义 bf16 结果的相对误差都在 5% 以内（`max_abs_err ≈ 7e-4`，参考量级 `0.055`），且随 `N` 增大而下降：

| N | custom fused max_abs_err | ref~ |
|---|---|---|
| 2048 | 1.22e-3 | 0.0776 |
| 4096 | 7.32e-4 | 0.0554 |
| 8192 | 4.88e-4 | 0.0410 |

迭代确实收敛到正交：对 fp32 参考结果抽样验证 `X Xᵀ` 与单位阵的偏差

$$
\lVert XX^\top - I \rVert \text{（抽样）} = 0.0043\ (N{=}4096),\quad 0.0034\ (N{=}8192),
$$

5 步迭代后奇异值基本被拉到 1——这从数值上确认了「正交化」在干活。

---

## 7. 离 SOTA 还差多少

把「自研 GEMM」和「cuBLAS 单个 4096³ GEMM」放到同一口径（`ns_single.out.txt`）：

```
single GEMM custom cfg1      0.510 ms   269.30 TFLOPS
single GEMM cuBLAS           0.156 ms   883.14 TFLOPS
  -> custom = 30.5% of cuBLAS (time 3.28x)
```

- **单个 GEMM**：自研 269 TFLOPS（27.2% 峰值），cuBLAS 883 TFLOPS（89.3% 峰值）——自研是 cuBLAS 的 **30.5%**。
- **端到端 NS 链**：`15 × 0.156 ms = 2.34 ms` 是「全是 cuBLAS-GEMM 速度」的理论下界；cuBLAS+elementwise 实测 `3.89 ms`（下界的 1.66×，多出来的是 5 次转置和 elementwise）；自研融合版 `8.44 ms`（下界的 3.61×）。

也就是说：**融合消掉了 elementwise 税，但自研 GEMM 与 cuBLAS 的 3.3× 差距才是大头**。诚实地说，这个算子上 cuBLAS 目前全面胜出；自研版本的价值在于把「优化器算子 = GEMM 链」这件事拆清楚，并给出了可量化的追赶路径。

---

## 8. 小结与下一步

**这一篇做了什么**

- 把 MuonClip 的 Newton–Schulz 正交化还原成 **15 个 GEMM、`30N³` FLOPs 的链式算子**，用 `N=2048/4096/8192`（对应 Kimi-K2.6 / DeepSeek-V4-Pro 的 7168、8192 级权重矩阵）实测；
- 用 `ncu` 证明它是**算力受限**（DRAM 仅 13%），但自研 `mma` GEMM 卡在 **L2 读放大（76%）+ 低 occupancy（23.8%）**；
- 实现**融合 epilogue** 的通用 GEMM，把 `f·x+g(GEMM)` 留在寄存器里，端到端省 **~5%**；
- 诚实对标：单个 GEMM 自研 = cuBLAS 的 **30.5%**；端到端自研 = cuBLAS 链的 **46%**。

**踩坑记录**

- `cp.async` 多级流水的 `__pipeline_wait_prior` 参数：`STAGES` 级流水里 prologue 预取 `STAGES-1` 级，循环里再 commit 一级，要等「当前 stage 完成」应写 `wait_prior(STAGES-1)`；写成 `STAGES-2` 会过早放行、读到半写数据（本篇一开始就踩了，`max_abs_err` 直接爆到 1e6）。
- 动态 smem 手搓多级缓冲时，**B 的 stage 基址要放在 A 的全部 stage 之后**（`smem + STAGES×BM×LDP`），漏乘 `STAGES` 会让 A/B 的 stage 槽互相覆盖。
- 大块 `128×256`/`256×128` 看着能降 L2 放大，实际因为每线程累加器翻倍触到寄存器墙，**更慢**；先扫配置再下结论。

**下一步**

- 第六部分 **31/38**：用 `wgmma.mma_async`（SS 直读 smem）+ TMA 重写这个 GEMM，目标是单个 GEMM ≥ 400 TFLOPS、NS 链把与 cuBLAS 的差距从 2.2× 压到 ≤1.3×（这与 16 篇遗留的 SW128 swizzle 任务是同一个基础设施）。
- 把 `Xᵀ` 的转置融进 G3 的 epilogue（现在每迭代多两趟 32MB 读写）。
- 用 fused NS 做一个 v0→v1→v2 的性能回归看板（路线图 36）。

---

**系列导航**：上一篇 {{< relref "cuda-kernel-opt-16-mla-fused" >}}（MLA 单 kernel 融合）· 路线图与性能榜见 `code/kernel-opt/ROADMAP.md`、`code/kernel-opt/TECHNIQUES.md`。
