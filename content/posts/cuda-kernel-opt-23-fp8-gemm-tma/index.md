---
title: "CUDA 算子调优（二十三）：FP8 GEMM（二）— TMA + mbarrier + warp specialization（768 → 1217 TFLOPS）"
date: 2026-09-21
draft: false
weight: 23
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "FP8", "e4m3", "GEMM", "Tensor Core", "wgmma", "TMA", "warp specialization", "mbarrier", "Hopper", "DeepSeek", "量化", "系列"]
categories: ["算子开发"]
---

[上一篇]({{< relref "cuda-kernel-opt-22-fp8-gemm" >}})我们把 DeepSeek-V4 形状（M=4096, N=3072, K=7168）的 FP8 e4m3 GEMM 从 266 TFLOPS 推到 **768 TFLOPS**，但 ncu 一眼就看穿了天花板不是算法，而是**装载方式**：

```
Compute (SM) 38.5%   Memory 39.8%   DRAM 16.5%
occupancy 24.7%  No Eligible 59.9%  warp cycles/issue 9.86
Block Limit: Registers=1, SharedMem=1     ← 1 CTA/SM
```

`cp.async` 版本里，A/B 的搬运是 **256 个线程各自算一遍 SW128 swizzle 地址、再发 `LDGSTS.16B`**：地址计算吃掉发射槽，地址/描述符吃掉寄存器，结果整块只塞得下 1 个 CTA/SM。这一篇把搬运整条链路换掉：

- **TMA（`cp.async.bulk.tensor.2d`）**：一条指令搬整个 `[BM,BK]` tile，而且硬件直接按 **SW128 swizzle** 写进 smem，布局与 wgmma 描述符逐字节一致；
- **mbarrier 生产者/消费者协议**：1 个 producer warp 专职发起 TMA，消费者 warpgroup 只做 `wgmma`；
- 顺带修掉两个把性能吃掉的坑（动态下标数组落 local memory、标量 store 不合并）。

结果：同一 shape、同一块 H100，**768.3 → 1217.3 TFLOPS（+58%）**，达到 cuBLAS FP8（1381.3）的 **88.1%**；per-block 缩放版 **519 → 906 TFLOPS（+74%）**。

| 版本 | 手段 | TFLOPS @4096×3072×7168 | 峰值占比 | 相对 cuBLAS FP8 |
|---|---|---|---|---|
| 22：cp.async + wgmma | `LDGSTS` + `wait_group` | 768.3 | 38.8% | 55.6% |
| **23：TMA + warp specialization** | `cp.async.bulk.tensor` + mbarrier | **1217.3** | **61.5%** | **88.1%** |
| 22：per-block | 每 128-k 块折算 | 519.0 | 26.2% | 37.6% |
| **23：per-block + TMA** | 同上，搬运算走 TMA | **906.2** | **45.8%** | **65.6%** |
| cuBLAS FP8 per-tensor | cuBLASLt | 1381.3 | 69.8% | 100% |

环境：H100 SXM 80GB（132 SM，HBM 3352 GB/s，FP8 dense 峰值 1978 TFLOPS），CUDA 13.2，`ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda"`。所有数字来自 `23-fp8-gemm-tma/fp8_tma.out.txt`。

---

## 1. 先把 `cp.async` 的账算清楚

22 篇的主循环（`22-fp8-gemm/fp8_gemm_wgmma.cu:95`）每次搬一个 stage 是这样：

```cpp
for (int i = tid; i < BM * (BK/16); i += NT) {
  const int r = i / (BK/16), c = i % (BK/16);
  __pipeline_memcpy_async(a + sw_off(r, c*16, BK),
                          &A[(block_row+r)*K + k0 + c*16], 16);   // 16B
}
```

一个 `BK=128` 的 `[256,128]` tile 要发 `256×8 = 2048` 次 `LDGSTS`；`[128,128]` 还要 1024 次。每次都要先算 `sw_off`（几次乘加 + 异或），这是**纯为搬运服务的指令**，却和 wgmma 抢同一个发射端口；同时地址寄存器、`cp.async` 的 stage 指针又占着寄存器文件。

TMA 的思路完全不同：**tile 的坐标、形状、swizzle 全部编码在一张 128 字节的 `CUtensorMap` 里，由 TMA 引擎自己完成。** 线程侧只剩一条指令：

```cpp
// fp8_gemm_tma.cu:98
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
      [dst_smem], [tensormap, {coord0, coord1}], [mbar];
```

一个 `[256,128]` 的 A tile，**一个线程发一条指令**就搞定（释放 255 个线程的发射槽），地址计算变成常量，swizzle 由硬件完成。这正是 ncu 说的「No Eligible 60%」的解法。

### 1.1 为什么 TMA 的 swizzle 刚好就是 wgmma 要的 SW128

第二十篇我们推导过 K-major SW128 的物理布局：`[row/8][k/128][8][128元素]`，元素 `(row,k)` 的字节偏移是

```text
byte = (rg*(K/64) + kg)*1024 + (rr*8 + ((kk/8) ^ rr))*16 + (kk%8)*2      (bf16)
```

换成 FP8 时一行 128 个 e4m3 = 128 字节，一个 atom 就是 **8 行 × 128B**，行内 16B 粒度按 `chunk ^ (row%8)` 异或 —— 这恰好是 TMA `CU_TENSOR_MAP_SWIZZLE_128B` 的定义。也就是说：

> **TMA 把 `[BM, BK]` tile 写进 smem 的布局，天然等于 wgmma 描述符期望的 SW128 布局。**

于是我们**不用在 kernel 里做任何 swizzle**，`make_desc_sw128` / `k32_addr`（`fp8_gemm_tma.cu:60`）原样复用。`BK` 也就锁定成 128（SW128 的内维上限是 128 字节），这刚好是 DeepSeek `weight_block=128×128` 的块宽。

TMA 需要 `CU_TENSOR_MAP_DATA_TYPE_UINT8`（CUDA 13 的驱动枚举里没有 FP8 类型；反正我们只搬字节，wgmma 自己解释成 e4m3）：

```cpp
// fp8_gemm_tma.cu:239
cuTensorMapEncodeTiled(&tm, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, ptr,
    /*globalDim  */ {K, R},        // dim0 = K（连续），dim1 = 行数 M/N
    /*globalStride*/ {K},          // 行间字节步长
    /*boxDim     */ {128, BM},     // 内维 128 字节 + 行数
    /*elemStride */ {1, 1},
    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
```

A 和 B 都用同一个模板（B 存的是 `[N][K]`，本来就是 K-major，不需要转置），这就是 22 篇说的「GEMM 的甜蜜点」。

---

## 2. warp specialization：1 个生产者 + N 个消费者

有了 TMA，装载变成「一个线程 + 一条异步指令」，就再没必要让所有线程都卷进来。我们按 **producer / consumer** 拆职责：

```text
        ┌─────────────────────── CTA (BM=256 时 544 线程) ───────────────────────┐
        │  producer warp (32)        consumer warpgroup ×4 (512)                  │
        │  ─────────────────         ─────────────────────────────                │
        │  for kb:                    for kb:                                     │
        │    wait(empty[st])  ◄────┐     wait(full[st])  ── wgmma ×Nsplit ──┐     │
        │    arrive_expect_tx       │     wgmma.commit_group()               │     │
        │    TMA(A) ──┐             │     if kb≥S-2: wgmma.wait_group(S-2)   │     │
        │    TMA(B) ──┤             │                arrive(empty[rs]) ──────┘     │
        │             ▼             │                                             │
        │     smem stage ring (STAGES) ── full[st] 由 TMA 完成时置位              │
        └─────────────────────────────────────────────────────────────────────────┘
```

两块 mbarrier 数组（`fp8_gemm_tma.cu:113` 起）：

- `full[s]`：**producer 以 `count=1` 到达 + `expect_tx(bytes)`**，TMA 搬完 `BM·BK + BN·BK` 字节后自动把这一相位补齐；消费者 `mbarrier.try_wait.parity` 等它。
- `empty[s]`：**每个消费者线程到齐一次**才算数（`count = NCONS`），表示这个 stage 的 smem 已被 wgmma 读完，producer 可以覆盖。

消费者对 stage 的释放用 `wgmma.wait_group<STAGES-2>`：每块 `commit_group` 一个 group，等「最多 STAGES−2 个 group 在飞」时，第 `kb-(STAGES-2)` 块的 mma 一定已完成，可以安全 `arrive(empty)`。这样 mma 管线里始终有多个 group 重叠，而不是每块 `wait0`。

### 2.1 两个必须踩的坑

**坑一：相位数组不要用动态下标。** 最自然的写法是给每个 stage 存一个 `uint32_t phase[STAGES]`，然后 `phase[st] ^= 1`。但 `st = kb % STAGES` 是运行期值，编译器只能把数组放进 **local memory**。第一版实测 ncu 直接点破：

```
Local Memory Spilling Requests 0        ← 没 spill
但：local memory 占 L1TEX 请求的 47.47%
```

`sm_90` 上 local memory 走 L1TEX，于是 `long_scoreboard` 飙到 **8.1 周期 / 条**，整个 kernel 只有 988 TFLOPS。修法很简单：mbarrier 的相位本来就是「这个 stage 第几次被使用」的奇偶 —— 第 `n` 次完成时相位是 `n&1`，不需要数组：

```cpp
// 消费者：full[st] 第 (kb/STAGES) 次完成 → 等待相位 (kb/STAGES)&1
mbar_wait(full + st, (uint32_t)((kb / STAGES) & 1));          // fp8_gemm_tma.cu:173
// 生产者：empty[st] 第 (kb/STAGES - 1) 次完成
mbar_wait(empty + st, (uint32_t)((kb / STAGES - 1) & 1));     // fp8_gemm_tma.cu:149
```

改完 local memory 归零，**988 → 1217 TFLOPS**。

**坑二：epilogue 的标量 store 不合并。** 22 篇的 epilogue 是 `C[r0*N+cc] = ...; C[r0*N+cc+1] = ...;` 两条 4B 存储。同一 warp 内相邻 lane 的 `cc` 相差 2 个 float，两条 store 各自只用到 sector 的一半，ncu 报 **50% 的 global sector 是 excessive 的**。改成一次 8B 向量存储即可（首地址 8B 对齐）：

```cpp
// fp8_gemm_tma.cu:231
*reinterpret_cast<float2*>(&C[(size_t)r0 * N + cc]) = make_float2(o[j*4+0], o[j*4+1]);
```

---

## 3. 实测：配置扫描

`scripts/run.sh 23-fp8-gemm-tma/fp8_gemm_tma.cu 4096 3072 7168 all`（原始输出 `fp8_tma.out.txt`）：

| 配置（BM×BN×BK, stages） | TFLOPS | 峰值占比 | 说明 |
|---|---|---|---|
| tma128×128×128 s3 | 1097.6 | 55.5% | smem 99KB → **2 CTA/SM** |
| tma128×128×128 s4 | 940.4 | 47.5% | |
| tma128×256×128 s3 | 1162.9 | 58.8% | |
| tma128×256×128 s4 | 1099.3 | 55.6% | |
| tma256×128×128 s2 | 1090.7 | 55.1% | |
| tma256×128×128 s3 | 1143.7 | 57.8% | |
| **tma256×128×128 s4** | **1217.3** | **61.5%** | 197KB smem，1 CTA/SM |
| pb128×128×128 s3（per-block） | 906.2 | 45.8% | 22 篇同口径 519（+74%）|
| pb256×128×128 s3（per-block） | 223.7 | 11.3% | 寄存器 150 → 折算流水断裂 |

几个值得记下的结论：

1. **BM=256（4 个 warpgroup）比 BM=128 好**：BM 越大，A tile 被同列 tile 复用的次数越多，L2 流量越小。256×128 比 128×128 高 ~11%。
2. **stage 不是越多越好，而是「够藏住 TMA 延迟」即可**：s4 最好，s5/s6 反而掉（smem 变大后 1 CTA/SM 且 stage 切换开销上升）。
3. **2 CTA/SM 并没有赢**：128×128 s3 的 `Block Limit Registers=2, Shared Mem=2`，实测 occupancy 26.7%、Compute 62.2%、**L2 79.9%**；256×128 s4 只有 1 CTA/SM，但 L2 只 56.7%、Compute 67.8%。对算力受限的大 GEMM，**降低 L2 流量比堆 occupancy 更重要**。
4. **per-block 不再被搬运算拖累**：906 vs per-tensor 1217（75%），而 22 篇的 per-block 只有 68%。per-block 的 ncu（`ncu_pb128x128s3.out.txt`）显示寄存器 **150**、occupancy 13.8%、Compute 48.6%——瓶颈已经变成「每 128-k 块折算时要等 mma 完成」的流水断裂，而不是搬运。

---

## 4. ncu：瓶颈换到了哪里

对最佳配置 `tma256x128×128 s4` 采集（`ncu_tma256x128s4.out.txt` / `ncu_stalls_best.out.txt`）：

```text
Compute (SM) 67.8%   Memory 57.9%   DRAM 27.0%
L1/TEX 62.4%   L2 56.7%   occupancy 25.5%   1 CTA/SM
sm__pipe_tensor_op_hmma_cycles_active = 71.98%     ← 张量管线活跃度
stall long_scoreboard 9.15   wait 1.15   not_selected 1.41   barrier 0.15
local memory spilling = 0
```

对比 22 篇：

| 指标 | 22 cp.async | 23 TMA+WS |
|---|---|---|
| TFLOPS | 768 | **1217** |
| tensor pipe active | ~41%* | **72.0%** |
| Compute (SM) | 38.5% | 67.8% |
| DRAM | 16.5% | 27.0% |
| local memory | — | 0 |
| occupancy | 24.7% | 25.5% |

\* 22 篇 mma.sync 版 tensor pipe 41%；wgmma 版未单独记录该指标，以 Compute 38.5% 为参照。

**张量管线已经跑到 72%**，剩下的 `long_scoreboard 9.15` 主要是消费者在 `mbarrier` 上等 TMA 数据（以及收尾的 epilogue 写）。也就是说：**wgmma 引擎的利用率接近饱和，下一步的边际收益要么来自更宽的 N 分块（减少 A 的重复装载与 TMA 次数），要么来自 cluster multicast（同一 A tile 跨 N 广播）。**

---

## 5. 与 SOTA 的差距

同 shape（4096×3072×7168，per-tensor）的 cuBLAS（`cublas_ref.out.txt`）：

| 实现 | TFLOPS | 占 FP8 峰值 |
|---|---|---|
| cuBLAS FP8 per-tensor | 1381.3 | 69.8% |
| cuBLAS FP8 per-row | 1345.4 | 68.0% |
| cuBLAS BF16 | 800.4 | 80.9%（bf16 峰值） |
| **本实现 per-tensor** | **1217.3** | 61.5% |
| 本实现 per-block | 906.2 | 45.8% |

- 距离 **cuBLAS FP8 只差 1.13×（88.1%）**——已经超过同 shape 的 cuBLAS BF16 的 **1.52×**。
- 22 篇的差距是 1.8×，这一篇收窄到 1.13×，靠的全是「把搬运算走 TMA + 修两个访存坑」。
- per-block 距 cuBLAS per-row（1345）还有 1.48×，留给下一篇的「双累加器跨块重叠」。

---

## 6. 小结

- **TMA 是把 wgmma 潜力兑现的必要条件**：搬运从「每线程算 swizzle + `LDGSTS`」变成「一条 `cp.async.bulk.tensor`」，同时消掉地址寄存器和发射槽。同 shape **768 → 1217 TFLOPS**。
- **TMA 的 SW128 与 wgmma 描述符逐字节一致**，kernel 里不需要任何 swizzle 代码；`BK=128` 是两者的天然交点。
- **生产者/消费者用 mbarrier 解耦**：producer 等 `empty`、consumer 等 `full`，消费者用 `wgmma.wait_group<STAGES-2>` 决定何时释放 stage。
- **相位数组别用动态下标**（会掉进 local memory，实测 988 vs 1217）；**epilogue 用 `float2` 向量存储**（消 50% excessive sector）。
- **对算力受限 GEMM，降 L2 流量（大 BM）比堆 occupancy（2 CTA/SM）更有效**：128×128 有 2 CTA/SM 但 L2 80%，反而不如 1 CTA/SM、L2 57% 的 256×128。
- 最终 **1217 TFLOPS = cuBLAS FP8 的 88.1%**，张量管线活跃度 72%。

## 7. 下一篇

留给下一轮的清单：

1. **per-block 的双累加器跨块重叠**：用两组累加器 ping-pong，让第 `kb` 块的折算与第 `kb+1` 块的 wgmma 重叠，把 906 往 1100+ 推；
2. **cluster（`cp.async.bulk.tensor` + multicast）**：同一 A tile 在 N 方向广播，砍掉重复装载；
3. **MoE grouped GEMM**：同一套 TMA + wgmma 基础设施直接用在 DeepSeek-V4 的 384 expert 上（主题 21），对照 `~/github/DeepGEMM`。

---

*代码：`code/kernel-opt/23-fp8-gemm-tma/fp8_gemm_tma.cu`；实测输出 `fp8_tma.out.txt`、`ncu_tma256x128s4.out.txt`、`ncu_tma128x128s3.out.txt`、`ncu_pb128x128s3.out.txt`、`ncu_stalls_best.out.txt`、`cublas_ref.out.txt`。*
