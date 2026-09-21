---
title: "CUDA 算子调优（二十二）：FP8 GEMM（一）— e4m3 的 mma.sync 与 wgmma，per-tensor / per-block 缩放（266 → 768 TFLOPS）"
date: 2026-09-21
draft: false
weight: 22
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "FP8", "e4m3", "GEMM", "Tensor Core", "wgmma", "Hopper", "DeepSeek", "量化", "系列"]
categories: ["算子开发"]
---

前二十一篇我们把 MLA / DSA / MuonClip 这些**注意力与优化器**算子写了个遍，主线是「把算力受限的融合 kernel 推向 FlashMLA」。这一篇换到**另一条同样重要的推理主线——低精度 GEMM**：DeepSeek-V4 / GLM-5 的权重与激活都用 **FP8 e4m3 + ue8m0 块缩放** 存储，MoE 的每一次 expert 前向都是 FP8 GEMM。它在 H100 上的理论峰值是 BF16 的 **2 倍**（1978 vs 989 TFLOPS），做不快就等于白瞎了一半算力。

本篇按「朴素也能跑对 → 用 ncu 找到天花板 → 换指令越过天花板」的路径，把一个 e4m3 GEMM 从 **266 TFLOPS** 推到 **768 TFLOPS**，并给出与 cuBLAS 的差距：

| 版本 | 手段 | TFLOPS @M4096×N3072×K7168 | 峰值占比 | 相对 cuBLAS |
|---|---|---|---|---|
| mma.sync 同步 | `mma.m16n8k32` + 手工 smem 取片段 | 174.3 | 8.8% | 12.6% |
| **mma.sync 流水** | + `ldmatrix`(b16 技巧) + `cp.async` 4 级 | 266.3 | 13.5% | 19.3% |
| **wgmma 流水** | 换 `wgmma.m64n128k32` + SW128 | **768.3** | **38.8%** | **55.6%** |
| wgmma + per-block | + 每 128-k 块折算 scale | 519.0 | 26.2% | 38.5% |
| cuBLAS FP8 per-tensor | cuBLASLt | 1381.7 | 69.9% | 100% |
| cuBLAS BF16（对照） | — | 800.7 | 80.9% | 58.0% |

真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`hidden_size=7168`、`moe_intermediate_size=3072`、384 routed experts + top-6，量化配置 `{fmt:e4m3, scale_fmt:ue8m0, weight_block_size:[128,128], activation_scheme:dynamic}`。代码在 `code/kernel-opt/22-fp8-gemm/`（`fp8_gemm.cu` = mma.sync 版，`fp8_gemm_wgmma.cu` = wgmma 版，`cublas_fp8_ref.py` 取 cuBLAS 对照，原始输出 `*.out.txt`）。

前置阅读：[第十三篇]({{< relref "cuda-kernel-opt-13-tensor-core" >}})（`mma.m16n8k16` + `ldmatrix` + padding 消 bank conflict）、[第二十篇]({{< relref "cuda-kernel-opt-20-mla-wgmma-sw128" >}})（SW128 swizzle 描述符）。

---

## 1. 背景：为什么是 FP8、为什么是块缩放

FP8 e4m3 只有 3 位尾数，动态范围也窄（最大 ±448）。要让一个 `[M,K]×[K,N]` 的 GEMM 精度可用，标准做法是**分块量化**：把 A（激活）按每 **1×128**（每个 token 的每 128 个通道）算一个缩放因子 $s_a$，把 B（权重）按每 **128×128** 块算一个 $s_b$，都存成 `float32`（DeepSeek 用 `ue8m0`，即无符号 8 位指数、隐含 1.x 的幂次格式，此处按 fp32 处理不影响算法）：

$$
C[m,n] \;=\; \sum_{k_b} s_a[m,k_b]\, s_b[\lfloor n/128\rfloor,k_b] \sum_{k\in k_b} A[m,k]\,B[n,k],
\qquad k_b = \lfloor k/128 \rfloor .
$$

关键观察：**在固定 k 块 $k_b$ 内，$s_a,s_b$ 都是标量**，所以「先无缩放地把 128 个 k 累加完，再乘一个标量折进总累加器」是等价的。这决定了后面 per-block 版的实现方式。

先估一下 roofline，确认这个 shape 是**算力受限**：

$$
\frac{\text{FLOPs}}{\text{bytes}} = \frac{2MNK}{MK + NK + 4MN}
= \frac{2\cdot4096\cdot3072\cdot7168}{29.3\text{M}+22.0\text{M}+50.3\text{M}}
\approx 1785\ \text{FLOP/byte}.
$$

FP8 Tensor Core 的 ridge point ~ $\frac{1978\text{ TFLOP/s}}{3.35\text{ TB/s}} \approx 590$ FLOP/byte，算力强度远高于它——**这是纯 TC 吞吐问题，优化目标就是把 tensor pipe 打满**。

---

## 2. FP8 的 mma 片段：一个「两个字节当 16 位」的技巧

`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` 的 K 维是 **32**（比 bf16 的 `m16n8k16` 大一倍），A 是 16×32、B 是 32×8，累加器仍是 16×8 的 fp32。片段布局（PTX ISA）：

```
A(16×32, 每个 .b32 装 4 个 e4m3):
  a0: row=g,   k=t4*4        a1: row=g+8, k=t4*4
  a2: row=g,   k=t4*4+16     a3: row=g+8, k=t4*4+16     (g=lane/4, t4=lane%4)
B(32×8):
  b0: n=g, k=t4*4            b1: n=g, k=t4*4+16
```

朴素实现可以直接按坐标从 smem 取 4 字节（4 个 fp8 打包）——本篇的第一个能跑对的版本就是这么干的，先建立正确性基准。但那样每个 mma 要 4+2 次 32-bit smem load，指令数太多。

**`ldmatrix` 其实可以直接用**：`ldmatrix` 操作的是 8×8 的 **b16** 元素，而「2 个相邻 fp8」正好是 1 个 b16。于是 A 的 16×32 fp8 = **16×16 b16 = 4 个 m8n8 矩阵**，一次 `ldmatrix.x4` 就取回了 `a0..a3`：

```
A smem 视图（行=16，列=b16，1 格=2 个 fp8）：
  matrix0 = rows 0-7,  b16 0-7  (fp8 k 0-15)   → ldmatrix 输出 d0 = a0
  matrix1 = rows 8-15, b16 0-7                 → d1 = a1
  matrix2 = rows 0-7,  b16 8-15 (fp8 k16-31)   → d2 = a2
  matrix3 = rows 8-15, b16 8-15                → d3 = a3
lane 地址：l0-7→(lane,0)  l8-15→(lane-8+8,0)  l16-23→(lane-16,16)  l24-31→(lane-24+8,16)
```

B 存成 `[N][K]`（K-major）后同理，一个 n8 tile 用 `ldmatrix.x2` 即可（d0=b0、d1=b1）。

smem 行距仍要 padding 消 bank conflict：BK=128 时行距取 **BK+16=144 B**（36 个 4B word，$36 \bmod 32 = 4$），8 个 row-group 落在 bank $\{0,4,\dots,28\}$，与 lane 内偏移叠加正好铺满 32 个 bank。`fp8_gemm_sync` / `fp8_gemm_pipe` 就是「同步铺垫 → ldmatrix 取片段 → mma」的逐级版本。

---

## 3. 现象：mma.sync 卡在「发射端口」

4 级 `cp.async` 流水 + ldmatrix 的 `p128x64x128s4` 跑出 **266 TFLOPS（13.5%）**，ncu（`ncu_mma.out.txt`）：

```
Compute (SM) Throughput   57.2%      Memory Throughput   52.2%   DRAM  7.9%
sm__pipe_tensor_op_hmma_cycles_active   41.2%
stall: math_pipe_throttle 1.12   wait 1.59   barrier 0.31   long_scoreboard 0.26
occupancy 24.2% (Block Limit Registers=2, Shared Mem=2)
```

`DRAM 7.9%` 说明根本不是访存瓶颈；`Compute 57%`、`math_pipe_throttle 1.12` + `wait 1.59` 则是**tensor pipe 输入队列已满 + warp 在等 mma 定长延迟**的指纹。tensor pipe 只有 41% active，但发射口已经堵死——这说明 **`mma.sync` 在 Hopper 上就不是为打满 FP8 峰值设计的**。

原因是硬件层面的：Hopper 的第 4 代 Tensor Core 的 **满吞吐由 `wgmma`（warpgroup 级、异步、SS 直读 smem 描述符）提供**，`mma.sync` 走的是兼容 SM80 的路径，指令发射带宽和操作数读取都受限。cuBLAS 跑 1382 TFLOPS 用的正是 `wgmma`。于是第二个版本换 `wgmma`。

---

## 4. 换 wgmma：FP8 的 SW128 布局和 bf16 一模一样

关于 wgmma 的 SW128 布局，第二十篇已经推得很细（`wgmma_sw128.cuh`）。这里有个**省事的关键事实**：CUTLASS 里 SW128 的 canonical 布局是按 **bit** 定义的：

```cpp
Layout_K_SW128_Atom_Bits = ComposedLayout<Swizzle<3,4,3>, smem_ptr_flag,
                                          Layout<Shape<_8,_1024>, Stride<_1024,_1>>>;
Layout_K_SW128_Atom<T>   = upcast<sizeof_bits<T>::value>(Layout_K_SW128_Atom_Bits{});
```

upcast 到 fp8（8 bit）后是 `Shape<_8,_128>`——**8 行 × 128 字节**的 atom，`Swizzle<3,4,3>` 作用在字节上。bf16 是 8 行 × 64 个元素 = 8×128 字节。**两者的物理字节布局完全同构**，只是「一行」从 64 个 bf16 变成 128 个 e4m3。所以描述符公式原样复用：

- `layout_type=1`（B128）、`LBO=1`（K-major 忽略）、`SBO=(K/128)*1024`（相邻 8 行组字节距）
- `(row,k)` 字节偏移 `= (rg*(K/128)+kg)*1024 + (rr*8 + ((kk/16)^rr))*16 + (kk%16)`
- 一个 wgmma k32 步（32 字节）的地址增量仍是 `(s/4)*1024 + (s%4)*32`

主循环用 `wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3`（一个 warpgroup 算 64×128），BM=128 = 2 个 warpgroup，**A/B 都是 K-major**，所以 `B[N][K]` 天然符合 `Major::K` 要求，**不需要任何转置**（这也是 attention 里 V 必须转置、而 GEMM 里没有这个麻烦的原因）。

```
BM=128, BN=128, BK=128, STAGES=3
  cp.async 装载 A[128][128] / B[128][128]（SW128 写）+ commit
  loop kb:
    cp.async.wait_group(STAGES-2); __syncthreads()     // 本 stage 就绪
    wgmma.fence
    for s in 0..3:  wgmma(acc, desc(A+wg*64*BK), desc(B))   // 4 个 k32
    wgmma.commit_group
    // per-tensor：acc 到最后才读，不必 wait0
    if has_next: wgmma.wait_group(STAGES-2); __syncthreads(); cp.async 下一 stage
```

最关键的一处与第二十一篇同源的优化：**per-tensor 的累加器到 epilogue 才读，主循环里不需要 `wgmma.wait_group 0`**（wait0 会把整条 mma 流水串行化）。改成每次 commit 一个 group、覆盖某个 stage 前只 `wgmma.wait_group STAGES-2`，就允许 mma 跨块流水。仅此一项把 `wg256x128x128s3` 从 710 → **768 TFLOPS**。

### per-block 缩放

per-block 版必须每 128-k 块把该块的累加器乘 `sa*sb` 折进总累加器（DeepGEMM 的 `final_accum += scale_a*scale_b*accum`），所以每个 k 块都要等一次 mma（`wait0`），流水被切断——这是 per-block 明显慢于 per-tensor 的结构性原因。折算是按 mma 的累加器坐标做的：

```
每个 warpgroup 的 m64n128 累加器，线程 (lane) 持有 c0,c1(row=g, col+0/1)
                                          c2,c3(row=g+8, col+0/1)
per-block 折算： fin[c] += sa[row]*sb[nblock]*acc[c]
```

其中 `sb` 取 128 列块的值（DeepSeek `weight_block_size=[128,128]` → 同一 n-tile 内是**一个标量**，只 1 次 load），`sa` 按行取。踩坑见下节。

---

## 5. 两个真踩过的坑

**坑 1：两个相邻输出列共用一个 `sb`。** 直觉上「c0 和 c1 是相邻列，用同一个 `sb[col]` 就行」——错。`c1` 的列是 `col+1`，per-block 下它有自己的 scale。第一次用逐元素随机 scale 做正确性测试时，`max_abs_err` 稳定为 35%（部分位置对、部分错）；把 scale 换成常数就「对」，换了 kb 相关也「对」——正是这种「只在 scale 逐列变化时才错」逼出了这个 bug。修法是读 `sbv0=sb[col]`、`sbv1=sb[col+1]`，c0/c2 用 `sbv0`、c1/c3 用 `sbv1`。

**坑 2：FP8 的 wgmma 尾部操作数个数与 bf16 不同。** bf16 是 `..., p, 1, 1, 0, 0`（多两个 trans 立即数），FP8 是 `..., p, scaleA, scaleB`（只有 3 个）。照抄 bf16 模板会被 ptxas 报 `Arguments mismatch for instruction 'wgmma.mma_async with FP8 types'`。另外 asm 操作数编号要重新数：64 个累加器后，`da=%64, db=%65, pred=%66, scaleA=%67, scaleB=%68`。

（还有一个「安静得可怕」的坑：`__launch_bounds__` 写死 256 而 BM=256 需要 4 个 warpgroup / 512 线程时，只会算一半的行，却因为采样点恰好落在已算区域而显示 `OK`、性能还虚高一倍。**扫配置时一定要把 check 的采样行/列打散到全区间**。）

---

## 6. 实测：扫参与 SOTA 对照

`4096×3072×7168`（DeepSeek-V4-Pro 单 expert 的 up/gate 口径），H100，原始输出见 `fp8_wgmma.out.txt` / `fp8_mma.out.txt` / `cublas_fp8.out.txt`：

| kernel | 配置 | TFLOPS | 峰值占比 |
|---|---|---|---|
| mma.sync | p64×128×BK128 s4 | 264.7 | 13.4% |
| mma.sync | p128×64×BK128 s4 | **266.3** | 13.5% |
| wgmma | 128×128×BK128 s2 | 602.3 | 30.5% |
| wgmma | 128×128×BK128 s3 | 717.0 | 36.2% |
| wgmma | 128×128×BK128 s4 | 640.0 | 32.4% |
| wgmma | 128×256×BK128 s3 | 755.5 | 38.2% |
| wgmma | **256×128×BK128 s3** | **768.3** | **38.8%** |
| wgmma | 256×256×BK128 s3 | 218.4 | 11.0%（128 个累加器寄存器 → spill）|
| wgmma + per-block | 128×128×BK128 s3 | 519.0 | 26.2% |
| cuBLAS FP8 per-tensor | cuBLASLt | 1381.7 | 69.9% |
| cuBLAS FP8 per-row | cuBLASLt | 1346.9 | 68.1% |
| cuBLAS BF16 | — | 800.7 | 80.9%（bf16 峰值）|

换 wgmma 让同一个 shape 从 266 → **768 TFLOPS（+2.89×）**，已经**超过同 shape 的 cuBLAS BF16（800.7）的 96%**，达到 cuBLAS FP8 的 **55.6%**。per-block 因为每块折算切断流水，是 per-tensor 的 68%（519/768），但 per-block（1×128 激活 + 128×128 权重）本身就比 per-tensor 精度高得多，这个代价是值得的。

wgmma 版的 ncu（`ncu_wgmma.out.txt`）显示瓶颈已经换了：

```
Compute (SM) 38.5%   Memory 39.8%   DRAM 16.5%
occupancy 24.7%  No Eligible 59.9%  warp cycles/issue 9.86
Block Limit: Registers=1, SharedMem=1    ← 1 CTA/SM
```

`Compute 38.5%`、`No Eligible 60%`、`warp cycles/issue 9.86` 指向 **occupancy 只有 25%（且被寄存器和 smem 同时锁成 1 CTA/SM）+ wgmma 等待**：256×128×BK128×s3 的 smem 是 3×(256+128)×128 = 144KB、每线程 122 寄存器，只能 1 个 block。下一步的杠杆很清楚——**更深的流水 + TMA 减寄存器/指令 + 2 CTA/SM，或 warp specialization**。

---

## 7. 小结

- **FP8 GEMM 的胜负手和 MLA 一样是「选对指令 + 选对 smem 布局」**：`mma.sync.m16n8k32` 受发射端口限制，在 Hopper 上封顶 ~41% tensor pipe；换 `wgmma.m64n128k32` 直接 2.89×。
- **FP8 的 `ldmatrix` 可以用「两个 fp8 = 一个 b16」的技巧复用**；**FP8 的 SW128 atom 与 bf16 逐字节同构**，第二十篇的 `wgmma_sw128.cuh` 原样可用。
- **`wgmma.wait_group` 流水**（而非每块 `wait0`）是 per-tensor 版的关键 8%：710 → 768。
- **per-block 缩放的代价是流水**：每 128-k 块折算要等 mma，per-block 519 vs per-tensor 768。真实 DeepSeek 的 `weight_block=128×128` 让 `sb` 在每个 n-tile 内退化成标量，折算开销已经很小。
- 当前距 cuBLAS 1382 还有 **1.8×**，瓶颈是 25% occupancy / 1 CTA/SM / wgmma 等待，不是算法。

## 8. 下一篇

FP8 GEMM（二）：把上面这条留给下一轮的清单做掉——**TMA（`cp.async.bulk.tensor` + mbarrier）多级流水 + warp specialization**，把 smem 装载从「256 个线程各自算 swizzle 地址」变成 TMA 一条指令，腾出寄存器和发射槽；再试 2 CTA/SM。目标是把 FP8 per-tensor 推到 **≥ 1000 TFLOPS**（差距缩到 1.4× 以内），并顺手把 per-block 的折算做成「双累加器 + 跨块重叠」。同一套基础设施也会用在 MoE 的 grouped GEMM 上。
