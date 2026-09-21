---
title: "CUDA 算子调优（十六）：MLA 注意力（二）— 单 kernel 融合，把 S/P 从显存里干掉"
date: 2026-09-21
draft: false
weight: 16
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MLA", "注意力", "Attention", "DeepSeek", "FlashMLA", "Tensor Core", "wgmma", "在线softmax", "kernel融合", "系列"]
categories: ["算子开发"]
---

上一篇（{{< relref "cuda-kernel-opt-15-mla-attn" >}}）把 MLA「吸收」成了 MQA，并用三个 Tensor Core kernel（QKᵀ → softmax → PV）跑到 **115.6 TFLOPS**。但三 kernel 要把中间量 `S`(fp32) 和 `P`(bf16) 各写读一遍显存——`Sq=Sk=1024` 时是 1.6 GB，`Sk=4096` 时是 **25.8 GB**，光这一项就吃掉 ~7.7 ms。

这一篇回答：**能不能把这三步融进一个 kernel，让 S/P 一次都不落显存？** 能。核心是两个洞察：

1. **`mma` 的 fp32 累加器布局，和下一步 PV 需要的 A 片段布局天然对齐**——softmax 后的 P 直接打包就能喂给下一个 `mma`，不需要任何 shuffle；
2. **online softmax 让 KV 分块处理**——QKᵀ 的累加器留在寄存器里做行 max/exp/rescale，P 永远只活在寄存器或 smem。

配套代码 [`code/kernel-opt/16-mla-fused/mla_fused.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/16-mla-fused/mla_fused.cu)（`mma` 版）与 [`mla_wgmma.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/16-mla-fused/mla_wgmma.cu)（`wgmma` 尝试），原始输出、`ncu` 数据同目录。硬件 H100 80GB HBM3（989 TFLOPS bf16 dense）。

先给结论（H=128，Sq=Sk=1024，DC=512/DR=64/DV=512）：

| 版本 | 做法 | 耗时 | 算力 | %峰值 | 相对上篇 |
|---|---|---|---|---|---|
| 15 `tc` | 三 kernel，S/P 物化 | 2.526 ms | 115.6 TFLOPS | 11.7% | 1.00× |
| `f4` | mma 单 kernel，DV 切 2（QK 重复 2×） | 1.993 ms | 146.5 TFLOPS | 14.8% | **1.27×** |
| **`f4s`** | mma 单 kernel，**共享 P 消重复**（dup=1） | **1.717 ms** | **170.1 TFLOPS** | **17.2%** | **1.47×** |
| `wgmma` | 用 warpgroup MMA 重写 | 3.452 ms | 84.6 TFLOPS | 8.6% | 0.73× |

`f4s` 在更长上下文更稳：`Sq=1024,Sk=2048` **183.5 TFLOPS**，`Sk=4096` **184.7 TFLOPS**，相对 15 的三 kernel 快 **2.96× / 5.78×**（三 kernel 的 softmax 物化随 `Sk` 线性膨胀，融合版几乎不涨）。离 FlashMLA 稀疏 prefill 在 H800 上公布的 **640 TFLOPS** 差距从 15 的 **5.7×** 收窄到 **3.5~3.8×**。

---

## 1. 为什么融合是「必选项」：先算一笔显存账

15 篇的三 kernel 结构：

```
QKᵀ : S[H,Sq,Sk] = Qcat[H,Sq,576] @ Kᵀ[576,Sk]      (fp32, 写显存)
softmax : P[H,Sq,Sk] = softmax_rows(S)              (bf16, 写显存)
PV   : O[H,Sq,512] = P[H,Sq,Sk] @ c_kv[Sk,512]
```

中间量 `S`/`P` 的显存流量（写+读）是 $\propto H\cdot S_q\cdot S_k$。实测：

| shape | S/P 物化流量 | 按 3.35 TB/s 折算 |
|---|---|---|
| Sq=1024, Sk=1024 | 1.61 GB | 0.48 ms |
| Sq=1024, Sk=2048 | 6.44 GB | 1.92 ms |
| Sq=1024, Sk=4096 | 25.77 GB | 7.69 ms |

`Sk=4096` 时，光是搬运 S/P 就要 7.69 ms，而三 kernel 总时长是 36.58 ms——**中间张量落显存是实打实的税**（占 21%）。而且它随 `S_k` 线性增长，长上下文必炸。

融合的目标就是：**`S` 只存在于寄存器，`P` 直接喂给下一个 mma，`c_kv` 常驻 smem**。

---

## 2. 融合 kernel 的三个关键点

### 2.1 布局对齐：累加器 C 就是下一步的 A 片段

这是整个融合的「Aha」。看 `mma.sync.m16n8k16` 的两个片段布局（PTX）：

| 片段 | 每线程 4 个 f32/bf16 的位置 |
|---|---|
| C/D（累加器，m16n8） | `c0,c1` → 行 `lane/4`，列 `(lane%4)*2, +1`；`c2,c3` → 行 `lane/4+8`，同列 |
| A（m16k16，两个 k8） | `a[0]={a0,a1}` → 行 `lane/4`，列 `(lane%4)*2, +1`；`a[1]={a2,a3}` → 行 `lane/4+8`；`a[2],a[3]` → 列 `+8` |

**一模一样。** 如果 QKᵀ 的 mma 按 `n8` 为步长产出 `S`，那么 PV 的一个 `k16` 片段要的 `A`，正好是**相邻两个 `n8` 的累加器**：

$$
\begin{aligned}
A[0] &= \mathrm{pack}_{bf16}(C_{\text{tile}\,0}.c_0,\; C_{\text{tile}\,0}.c_1)\\
A[1] &= \mathrm{pack}_{bf16}(C_{\text{tile}\,0}.c_2,\; C_{\text{tile}\,0}.c_3)\\
A[2] &= \mathrm{pack}_{bf16}(C_{\text{tile}\,1}.c_0,\; C_{\text{tile}\,1}.c_1)\\
A[3] &= \mathrm{pack}_{bf16}(C_{\text{tile}\,1}.c_2,\; C_{\text{tile}\,1}.c_3)
\end{aligned}
$$

（`tile 0` 覆盖 KV 列 `[0,8)`，`tile 1` 覆盖 `[8,16)`。）没有 shuffle、没有 smem 往返，只有 4 次 `__floats2bfloat162_rn`。这就是 FlashAttention 系列 P 不落盘的关键，也是本篇第一版 `f4` 的全部秘密。

### 2.2 online softmax：block 内精确，block 间递推

KV 太长，必须分块。每个 KV tile（`KT=64`）内部我们**持有全部 S**，所以 block 内做的是**精确 softmax**（不是逐元素 online）：

```text
对每个 KV tile:
  1. QKᵀ mma 得到 S[16, KT]（fp32，寄存器）
  2. block_max = max_j S[r,j]     （4-lane shuffle 归约）
     m_new = max(m_old, block_max); alpha = exp(m_old - m_new)
  3. P = exp(S - m_new); block_sum = Σ_j P
     l = l * alpha + block_sum
  4. O *= alpha                    （把历史累加器 rescale）
  5. PV mma: O += P @ V
最后 O /= l
```

行归约只在 **4 个 lane** 内（`lane/4` 决定行号，所以 `__shfl_xor_sync(...,1/2)` 就够），每线程手里正好有一整行的元素（同一行 `n8` tile 的 `c0,c1` 全在该 lane）。

### 2.3 寄存器墙与 DV 切分：dup 是唯一代价

融合后每个 warp 要持有 `O[16 行, DV]` fp32。`DV=512` 时是 `16×512/32 = 256` 个寄存器/线程——直接爆表（上限 255）。所以必须把 DV 切给不同 warp：

$$
\text{每线程 } O \text{ 寄存器} = \frac{16\cdot DVW}{32},\qquad \text{重复因子 } \text{dup}=\frac{DV}{DVW}
$$

`f4` 取 `DVW=256`（DVGRP=2）：`O=128` 寄存器，但**同一组 query 的 QKᵀ 被 2 个 warp 各算一遍**（dup=2）。这是寄存器和算力之间的硬取舍。

```
f4（DVGRP=2，QK 重复 2×）           f4s（共享 P，QK 不再重复）
block: 64 query × 512 DV           block: 64 query × 512 DV
warp (rowg, dvg) 16 行 × 256 DV    warp (rowg, dvg): QK 只算 KV 的 dvg 半边
  ├─ QKᵀ: 完整 KV  ← 重复！          ├─ QKᵀ: KV[dvg*32 : dvg*32+32]
  └─ PV : 自己的 256 DV             ├─ pmax/psum 经 smem 交换 → 全行统计
                                     ├─ P(bf16) 写进 smem 共享
                                     └─ PV : 读完整 P，算自己的 256 DV
```

### 2.4 消重复：`f4s` 用一点点 smem 换掉一半 QK

`f4` 的 L1 流量里，K 的 `ldmatrix` 占大头（每 warp 每 KV tile 读整块 K）。`f4s` 把同一 row 组的两个 warp **按 KV 对半分工**：各算 32 列 QK，行 max/sum 用 128 个 float 的 smem 交换，P(bf16) 写进 `smem[64][72]` 再被两 warp 共享读。

- 权衡：多了一次 `__syncthreads` + `P` 的 smem 写读（仅 8 KB）；
- 收益：**QK 的 mma 从 2× 降到 1×**，K 的 `ldmatrix` 也减半。

这是 `f4s` 比 `f4` 快 16% 的来源。

---

## 3. 实测：从 mma 到「共享 P」

### 3.1 主结果（H=128，Sq=Sk=1024）

| 版本 | BM | DVGRP | KT | warps | 耗时 (ms) | TFLOPS | %峰值 | dup |
|---|---|---|---|---|---|---|---|---|
| 15 `tc` | – | – | – | – | 2.526 | 115.6 | 11.7% | – |
| `f4s` | 64 | 2 | 64 | 8 | **1.717** | **170.1** | **17.2%** | 1 |
| `f4` | 64 | 2 | 64 | 8 | 1.993 | 146.5 | 14.8% | 2 |
| `f2s` | 32 | 4 | 64 | 8 | 2.565 | 113.9 | 11.5% | 1 |
| `f2` | 32 | 4 | 64 | 8 | 2.768 | 105.5 | 10.7% | 4 |
| `f4k32` | 64 | 2 | 32 | 8 | 2.247 | 130.0 | 13.1% | 2 |
| `f2k32` | 32 | 4 | 32 | 8 | 3.289 | 88.8 | 9.0% | 4 |
| `f1` | 16 | 8 | 64 | 8 | 4.662 | 62.6 | 6.3% | 8 |

几个规律：

- **dup 越小越快**：`f4s`(dup1) > `f4`(dup2) > `f2`(dup4) > `f1`(dup8)；
- **但 dup=1 也有代价**：`f2s` 虽然 dup=1，却比 `f4s` 慢——因为 DVGRP=4 时，同一个 row 组有 4 个 warp，**每个都要完整读一遍 Q**，Q 的 smem 流量翻了 4 倍，反而更亏；
- `KT=32` 明显更慢：KV tile 太小，Q 被重复读的次数翻倍，且同步更频繁。

### 3.2 长上下文：融合版几乎不随 `Sk` 变慢

| shape | 15 `tc` | `f4` | `f4s` | `f4s` vs 15 |
|---|---|---|---|---|
| Sq=1024, Sk=1024 | 115.6 | 146.5 | 170.1 | 1.47× |
| Sq=1024, Sk=2048 | 124.1 | 158.1 | 183.5 | **2.96×** |
| Sq=1024, Sk=4096 | 127.7 | 158.0 | 184.7 | **5.78×** |
| Sq=2048, Sk=2048 | – | 154.6 | 177.3 | – |

三 kernel 的时间随 `Sk` 线性涨（softmax 物化+三次 kernel 启动），算力卡在 ~125 TFLOPS；融合版算力反而**随 `Sk` 上升**（170→185），因为固定开销（Q 载入、kernel 启动）被摊薄。

### 3.3 wgmma 尝试：为什么它没赢

Hopper 的 `wgmma.mma_async` 让 A/B 操作数**直接从 smem 描述符读取**，理论上能干掉 `ldmatrix`。我照着 CUTLASS 的描述符格式实现了 `mla_wgmma.cu`（`m64n64k16` QKᵀ + `m64n256k16` PV，2 个 warpgroup，DV 各半）。结果只有 **84.6 TFLOPS / 3.45 ms**，比 `mma` 版还慢一倍。

`ncu` 定位：

```text
L1/TEX Cache Throughput   86.4%   ← 饱和
Compute (SM) Throughput   14.6%
sm__pipe_tensor_cycles_active  13.1%
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  2.7e8   ← 巨量 bank conflict
```

原因：为了省事，我用了 **无 swizzle 的 K-major `INTERLEAVE` 布局**（8×8 core matrix，行距 128 B）。这个布局有两个问题：

1. **smem 写冲突**：连续线程写相邻的 k-block（步长 128 B），全落在同一个 bank；
2. **tensor core 读操作数效率低**：没有 swizzle，操作数读取的 bank 利用率差。

生产级 kernel（FlashMLA、CUTLASS）都用 **SW128 swizzle**，配合 `cp.async`/TMA 写入，才能让描述符读取全速。SW128 需要正确处理 128 B 原子的 XOR 相位和描述符 `base_offset`，是下一篇/后续版本的事。**结论：wgmma 的胜负手不在指令本身，而在 smem 布局。**

---

## 4. ncu 剖析：还剩什么瓶颈

`f4s`（1.73 ms）的 Speed-of-Light：

| 指标 | 值 |
|---|---|
| Duration | 1.73 ms |
| L1/TEX Throughput | 63.2% |
| Compute (SM) | 25.8% |
| Tensor pipe active | 26.7% |
| DRAM Throughput | 7.0% |
| Achieved Occupancy | 12.5% |
| Issued warp / scheduler | 0.20 |
| No eligible | 79.8% |
| 主要 stall | `short_scoreboard` 3.05，`long_scoreboard` 2.05，`wait` 1.44 |

读出来的故事很清楚：

- **DRAM 只有 7%**：KV（`c_kv` 仅 1 MB）常驻 L2，完全没有访存压力——融合彻底解决了「中间量落显存」；
- **L1/TEX 63%** vs Compute 26%：瓶颈是 **smem 的 `ldmatrix` 流量**（`short_scoreboard` 是「等 smem 数据」的 stall）；
- **occupancy 只有 12.5%**：`O=128` 寄存器 + 159 KB smem 把 block 限死在 1 个/SM、8 warps。`No eligible 79.8%` 就是因为 warp 太少，`ldmatrix` 的延迟藏不住。

方向也因此明确：**要么用 SW128 + wgmma 把操作数读取从 LSU 挪走**，要么做 warp specialization（生产者 warp 专门用 TMA/cp.async 搬数据）把 occupancy 问题绕开。

---

## 5. 与 SOTA 的差距

FlashMLA 在 H800 SXM5（CUDA 12.8）公布的 MLA **稀疏 prefill 640 TFLOPS**、稠密 decode compute-bound 660 TFLOPS。本机是 H100，同为 SM90 全速 bf16 TC（989 TFLOPS），口径可比：

| 对比 | 15 `tc` | `f4s`（S=1024） | `f4s`（S=4096） | FlashMLA |
|---|---|---|---|---|
| TFLOPS | 115.6 | 170.1 | 184.7 | 640 |
| 差距 | 5.5× | 3.76× | **3.47×** | – |

差距从 15 的 ~5.5× 收到了 **3.5×**，但离路线图定的「压到 2× / 破 200 TFLOPS」还差一口气。那口气就是 smem 布局 + warp specialization。

---

## 6. 小结

- **融合是对的**：单 kernel 把 S/P 从显存里干掉，`Sk=4096` 时相对三 kernel 快 **5.78×**，因为省掉了 25.8 GB 的中间量读写；
- **布局对齐是免费午餐**：`mma` 累加器 C 就是 PV 的 A 片段，P 转换零 shuffle——这是所有 flash-attention 类 kernel 的基石；
- **DV 切分带来 dup**：`O` 寄存器墙逼着切 DV，dup 就是代价；用 8 KB 的 smem 共享 P（`f4s`）能把 QK 重复从 2× 降到 1×，提升 16%；
- **wgmma 不是银弹**：无 swizzle 的 `INTERLEAVE` 布局让 smem store 爆冲突、操作数读取低效，实测反而更慢。胜负手在 **SW128 swizzle**；
- 当前瓶颈是 **smem `ldmatrix` + 12.5% occupancy**（`short_scoreboard` 主导），DRAM 只用了 7%。

复现：

```bash
cd code/kernel-opt
scripts/run.sh 16-mla-fused/mla_fused.cu 1024 128 1024 f4s     # 170 TFLOPS
scripts/run.sh 16-mla-fused/mla_fused.cu 1024 128 1024 all     # 全变体扫描
scripts/ncu.sh 16-mla-fused/mla_fused.cu --section SpeedOfLight --kernel-name regex:mla_shared -- 1024 128 1024 f4s
# wgmma 版（注意 CUDA 13 的 nvcc 不认 -arch=sm_90a，需显式 gencode）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" scripts/run.sh 16-mla-fused/mla_wgmma.cu 1024 128 1024
```

**下一篇计划**：把 `wgmma` 的 smem 布局换成 **SW128 swizzle**（配 `cp.async` 写入），目标把执行引擎从 LSU 挪到 tensor pipe，冲 250+ TFLOPS；之后再上 DSA 稀疏注意力（第 17 篇）。
