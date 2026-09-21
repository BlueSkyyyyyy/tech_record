---
title: "CUDA 算子调优（二十）：wgmma + SW128 swizzle — 把 MLA 的 smem 布局换对（84.6 → 105.3 TFLOPS）"
date: 2026-09-21
draft: false
weight: 20
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MLA", "wgmma", "Hopper", "swizzle", "Tensor Core", "DeepSeek", "系列"]
categories: ["算子开发"]
---

[第十六篇]({{< relref "cuda-kernel-opt-16-mla-fused" >}}) 里我们给融合 MLA 加了一个 `wgmma`（Hopper warpgroup MMA）版本，结果**比手写 `mma.m16n8k16 + ldmatrix` 还慢一倍**：`f4s` 170 TFLOPS，`wgmma`(无 swizzle 的 `INTERLEAVE` 布局)只有 **84.6 TFLOPS**。当时 ncu 给的原因是：共享内存 **store bank conflict 2.7e8 次**、tensor 读操作数低效。一句话——**`wgmma` 的胜负手不是指令，是 smem swizzle**。

这一篇把这个判断兑现：把 operands 的 smem 布局从 `INTERLEAVE` 换成 **K-major + 128B swizzle（SW128）**，先从**冒烟 GEMM** 把描述符和 k16 步进验证对，再移植回 MLA。结论：

| 版本 | 布局 | MLA TFLOPS（Sq=1024, Sk=1024） |
|---|---|---|
| `wgmma`（16 篇） | K-major `INTERLEAVE` | 84.60 |
| **`wgmma`（本篇）** | **K-major SW128** | **105.30（+24.5%）** |
| `mma+ldmatrix` `f4s`（16 篇） | 行主序 + padding | 170.11 |
| FlashMLA sm90（sparse prefill, H800） | CUTLASS | ~640 |

真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json` 的 MLA 吸收口径：$H{=}128$、$d_c{=}512$、$d_r{=}64$、$d_v{=}512$（bf16 dense 峰值 989 TFLOPS @H100）。

---

## 1. 为什么 `INTERLEAVE` 会输

`wgmma.mma_async.m64nNk16` 是 **SS 形式**：A、B 两个操作数都**不经过寄存器**，由硬件拿着一个 64-bit **smem 描述符**自己去 shared memory 里取。好处是彻底省掉了 `ldmatrix` 的 L1 流量（那正是 `mma` 版 `f4s` 的瓶颈，ncu 里 L1/TEX 63%）；代价是 **smem 里怎么摆，直接决定硬件取数的效率**。

16 篇用的 `INTERLEAVE`（`layout_type=0`，无 swizzle）布局是 8×8 core-matrix 分块：

```
off(mn,k) = (mn/8)*SBO + (k/8)*128 + (mn%8)*16 + (k%8)*2   (字节)
```

它的问题有两个，ncu 都抓到了：

- **写侧**：连续线程写相邻 k-block 会全部撞到同一个 bank——16 篇实测 store bank conflict **2.7e8 次**；
- **读侧**：core-matrix 的行距只有 16B，`wgmma` 一次要读 8 行，跨行步长小、访问被拆得很碎，tensor pipe 只有 13%。

**128B swizzle（SW128）** 就是硬件为这场景准备的布局：把每个 128B 行内的 16B 单元按行号做交换

$$c' = c \oplus r \qquad (c=16\text{B 列号},\ r=\text{行号}\bmod 8)$$

于是同一个 bank 上的访问被“打散”到不同列，读写都顺了。

---

## 2. SW128 的物理布局与描述符

### 2.1 物理地址公式

CUTLASS `cute/atom/mma_traits_sm90_gmma.hpp` 里 K-major SW128 的 canonical 布局是 `Swizzle<3,4,3> o ((8,n),2):((8,SBO),1)`（单位 `uint128_t`）。翻成人话，一个 **atom = 8 行 × 64 个 bf16（128B/行，行距 128B）**，atom 内做 `Swizzle<3,4,3>`。`Swizzle<3,4,3>` 的语义是「把字节地址的 bit[9:7] 异或进 bit[6:4]」，展开后正好就是上面的 $c'=c\oplus r$。于是元素 $(row,k)$ 的字节偏移为（`K` = tile 的行宽，须为 64 的倍数；布局按 `[行组][k组][8行][64元素]` 排列）：

```cpp
// 布局：[row/8][k/64][8 rows][64 elems]，每 atom 1024B；atom 内 16B 列做 c' = c ^ r
__device__ __forceinline__ int sw128_off(int row, int k, int K) {
  const int rg = row >> 3, rr = row & 7;
  const int kg = k >> 6,   kk = k & 63;
  const int c = kk >> 3,   cc = c ^ rr;
  return (rg * (K >> 6) + kg) * 1024 + (rr * 8 + cc) * 16 + (kk & 7) * 2;
}
```

16B 向量化写入时 `kk % 8 == 0`，退化成 `(rr*8 + (c^rr))*16`，天然 16B 对齐。

### 2.2 描述符：14-bit 字段 + 3-bit base_offset + 2-bit layout_type

描述符的 bit 分布（`cute/arch/mma_sm90_desc.hpp`）：

```
 bit  0-13  start_address        (>>4)
 bit 16-29  leading_byte_offset  (>>4)    # K-major SW128 固定 1（硬件忽略）
 bit 32-45  stride_byte_offset   (>>4)    # 相邻「8 行组」的字节距离 = (K/64)*1024
 bit 49-51  base_offset                   # 地址 1024B 对齐时为 0
 bit 62-63  layout_type                   # B128 = 1
```

```cpp
__device__ __forceinline__ uint64_t make_desc_sw128(uint32_t addr, uint32_t sbo_bytes) {
  uint64_t d = 0;
  d |= (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)((16u >> 4) & 0x3FFF) << 16;   // LBO = 1
  d |= (uint64_t)((sbo_bytes >> 4) & 0x3FFF) << 32;
  d |= (uint64_t)0 << 49;                        // base_offset
  d |= (uint64_t)1 << 62;                        // layout_type = B128
  return d;
}
```

### 2.3 k16 步进

一个 atom 覆盖 $K{=}64$ 个元素（4 个 k16）。第 $s$ 个 k16 块（$k=s\cdot16$）在 atom 内的起点是 $c=s\bmod4$ 对应的 16B 列，故

$$\text{addr}(s) = \text{base} + \lfloor s/4\rfloor\cdot1024 + (s\bmod4)\cdot32$$

```cpp
__device__ __forceinline__ uint32_t sw128_k16_addr(uint32_t base, int s) {
  return base + (uint32_t)((s >> 2) * 1024 + (s & 3) * 32);
}
```

整个 SW128 助手不到 60 行，全在 [`wgmma_sw128.cuh`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/20-mla-wgmma-sw128/wgmma_sw128.cuh)。

---

## 3. 先用冒烟 GEMM 把描述符验证对

改 swizzle 最怕“公式写对但差一个相位”。所以**先做小 GEMM**：`C[M,N]=A[M,K]·B[N,K]ᵀ`，A/B 均 K-major SW128。配置 `BM=128, BN=256, BK=64`，2 个 warpgroup，各跑 `wgmma.m64n256k16`。

```bash
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 20-mla-wgmma-sw128/gemm_sw128.cu 4096 4096 4096
```

实测（`gemm_sw128.out.txt`）：

| shape | 耗时 | TFLOPS | 占峰值 |
|---|---|---|---|
| 2048³ | 0.1156 ms | **148.66** | 15.0% |
| 4096³ | 1.0409 ms | **132.04** | 13.4% |

误差 `max_abs_err < 5e-2` 相对（朴素 kernel 全量对拍）通过。**描述符一次就对**——这也说明 16 篇的 84.6 纯粹是布局问题，不是指令用错。

注意这个冒烟 GEMM 是单缓冲、每 K 块 `__syncthreads`，所以 132 不是 `wgmma` 的上限（作为对照，13 篇的 `mma+ldmatrix+cp.async` 同尺寸 221）。它的任务只是**证明 SW128 描述符与 k16 步进正确**。

### 3.1 再做两个定点测试，把 MLA 的两段分别锁死

冒烟 GEMM 只覆盖 `K=64`（单 atom）。而 MLA 里 Q/K 的 $d_k = d_c+d_r = 576$，是 **9 个 atom**，k16 要跨 atom 步进。所以额外写了两个定点测试：

- **QK576**：`A[64][576] @ B[64][576]ᵀ`，`wgmma.m64n64k16 × 36`，`SBO=(576/64)*1024=9216` → `max_abs_err=1.6e-7`；
- **PV**：`P[64][64] @ Vᵀ[512][64]ᵀ`，`wgmma.m64n256k16 × 2` → `max_abs_err=1.5e-8`。

两段都过，才敢接回 MLA。

---

## 4. 一个把人坑惨的 nvcc “优化”：相邻 bf16 存储被合并

整合进 MLA 后第一次跑，`max_abs_err=3.6e-1`——**完全错**，但 QK576/PV 两个定点测试又是好的。问题出在 **P 矩阵从寄存器写回 smem** 这段。

SW128 里同一行的相邻两列 `(col, col+1)` 落在同一 16B 单元、且字节相邻（`col` 偶数时 `sw128_off` 是 4B 对齐）。直觉写法是两个 `bf16` 标量存储：

```cpp
*reinterpret_cast<bf16*>(ps + sw128_off(r0, col,   KT)) = S[t*4+0];
*reinterpret_cast<bf16*>(ps + sw128_off(r0, col+1, KT)) = S[t*4+1];   // ← 被吃掉
```

但 **nvcc 会把这两个 2B 存储合并成一个 `st.shared.u32`，并且只写了低 16 位**，`col+1` 全部变成 0。用一个小复现程序（`bf16_store_pitfall.cu`）对比三条路径的字节：

```
bad (2x bf16 store) : byte diff = 3553  MISMATCH <- nvcc 合并掉高 16 位
fix (packed u32)    : byte diff = 0     OK
```

修法：**显式打包成一次 u32 存储**（`col` 偶数 → `sw128_off` 4B 对齐，正好放两个 bf16）：

```cpp
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  __nv_bfloat162 h = __floats2bfloat162_rn(a, b);
  return *reinterpret_cast<uint32_t*>(&h);
}
*reinterpret_cast<uint32_t*>(ps + sw128_off(r0, col, KT)) = pack2(S[t*4+0], S[t*4+1]);
```

这个坑很隐蔽：**功能测试（QK/PV 单独）全过，只有 P 写回这一小段错**，症状是输出一半元素偏小。教训——**任何“相邻两个 bf16/半精度标量存储”都要么显式打包、要么用向量类型**。

---

## 5. 移植回 MLA

融合结构与 16 篇完全一致（online softmax 进 QK epilogue、$C\to$ PV 的 A 片段零 shuffle、KV 常驻 smem、P 走 smem 供 PV 复用），只把四处 smem 的存/取换成 SW128：

```
smem 布局（216 KB，1 CTA/SM）
┌────────────┬────────────┬────────┬──────────────┐
│ Q [64][576]│ K [64][576]│P[64][64]│ V [512][64] │
│  SW128     │  SW128     │ SW128  │  SW128(转置)│
│  73.7 KB   │  73.7 KB   │  8 KB  │   64 KB     │
└────────────┴────────────┴────────┴──────────────┘
SBO_K = (576/64)*1024 = 9216    SBO_P = SBO_V = 1024
```

- Q / K 用 `sw128_store16` 沿 $d_k{=}576$ 向量化写入（16B）；
- V 需要转置成 `[d_v][k]`（$wgmma$ 的 B 操作数必须 K-major），逐元素 gather 8 个 kv 凑成 16B 再 `sw128_store16`；
- P 用上面的 `pack2` 写回。

实测（`mla_wgmma_sw.out.txt`，`max_abs_err=9.1e-6` 通过）：

| 配置 | 耗时 | TFLOPS | 占峰值 |
|---|---|---|---|
| Sq=1024, Sk=1024 | 2.7735 ms | **105.30** | 10.6% |
| Sq=1024, Sk=2048 | 5.1733 ms | 112.91 | 11.4% |
| Sq=1024, Sk=4096 | 9.7440 ms | 119.89 | 12.1% |

相对 16 篇同结构的 `INTERLEAVE` 版（84.6 @1024），**+24.5%**；误差和 `mma` 版 `f4s` 逐位一致。

### 5.1 顺手试了一下“把 V 的 gather 藏到 QK 后面”

`wgmma` 是异步的，直觉上可以把 V 的 global gather 放到 `wgmma.commit()` 之后、`wgmma_wait0()` 之前，让 global 延迟叠在 tensor 计算上：

```
load K → sync → commit QK → [gather V] → wait0 → softmax → P → sync → PV
```

实测 `104.98 TFLOPS`（原 105.86），**没有收益**。说明这一版的瓶颈不是“V 的 global 延迟没藏住”，而是更结构性的东西——见下一节。

---

## 6. 还是打不过 `mma f4s`：ncu 说清楚了

对 `mla_wgmma_sw` 跑 `--set full`（`mla_sw_ncu.out.txt`）：

| 指标 | 值 | 解读 |
|---|---|---|
| `Compute (SM) Throughput` | **18.5%** | tensor pipe 远没吃饱 |
| `L1/TEX Cache Throughput` | **70.8%** | 共享内存/操作数读取是最高项 |
| `DRAM Throughput` | 5.1% | 远非访存受限 |
| `Achieved Occupancy` | **12.49%** | 8 warps/SM（1 CTA） |
| `Registers Per Thread` | 205 | `Block Limit Registers = 1` |
| `Dynamic Shared Memory` | 221 KB | `Block Limit Shared Mem = 1` |
| `No Eligible` | **80.9%** | 每周期只有 0.24 个可发射 warp |
| `Warp Cycles Per Issued Instruction` | 10.47 | 其中 **5.0 cycles（47.8%）** 等 L1TEX 数据 |
| shared store bank conflict | 4.4-way | 16 篇的 2.7e8 已消失，4.4≈16B 存储的理论下限 4.0 |

两条结论：

1. **SW128 确实修好了 swizzle 问题**：store bank conflict 从 2.7e8 降到接近理论下限，L1 也从 86% 降到 70.8%。
2. **但 `wgmma` 版仍被“低 occupancy + 无流水”锁死**：221KB smem 把 SM 锁成 1 个 CTA（8 warps），Q/K/V 单缓冲导致 load 与 tensor 计算完全串行，`No Eligible=80.9%`。相比之下 `mma` 版 `f4s` 有 8 warps × 更细的 warp 级任务分解，能靠 ILP 把延迟藏住。

换句话说：**SW128 让 `wgmma` 从 84.6 追到 105.3，但 `wgmma` SS 形式对 V 的“必须转置成 K-major”又引入了一次 gather；在单缓冲、1 CTA/SM 的结构下，这些延迟无处躲藏。** 与 `f4s` 的差距是 **1.62×**，与 FlashMLA（~640）是 **6.1×**。

---

## 7. 小结 & 下一步

- **`wgmma` 的胜负手是 swizzle**：同一份 MLA 计算，`INTERLEAVE` 84.6 → **SW128 105.3 TFLOPS（+24.5%）**；store bank conflict 2.7e8 → 4.4-way（理论下限）。
- **SW128 描述符**：14-bit `start_address`/`LBO`/`SBO` + 3-bit `base_offset` + 2-bit `layout_type(=1)`；K-major 下 `LBO=1`、`SBO=(K/64)*1024`、k16 步进 `+floor(s/4)*1024+(s%4)*32`。一次冒烟 GEMM 验证对（132 TFLOPS @4096³）。
- **踩坑**：nvcc 会把相邻两个 bf16 标量存储合并成 u32 并**丢掉高 16 位**（3553 字节错位）；必须 `__floats2bfloat162_rn` 显式打包。
- **现状**：`wgmma` SW128 版仍慢于手写 `mma+ldmatrix` 的 `f4s`（170），瓶颈从“swizzle”变成了“**1 CTA/SM + 单缓冲 + 无 producer/consumer 分工**”。
- **下一步（第六部分 32/33/39）**：① `cp.async.bulk.tensor`（**TMA**）+ 多级 mbarrier 流水，把 Q/K/V 的搬运与 tensor 计算解耦；② **warp specialization**（producer warp 搬数据 / consumer warp 算），让 8 warps 里始终有人在喂 tensor pipe；③ 目标把 tensor pipe 从 18.5% 推到 ≥40%，稠密 MLA 冲 **250+ TFLOPS**，把与 FlashMLA 的差距压到 ≤2.6×。SW128 的描述符基础设施本可直接复用。

> 代码：`code/kernel-opt/20-mla-wgmma-sw128/`（`wgmma_sw128.cuh`、`gemm_sw128.cu`、`qk576_test.cu`、`pv_test.cu`、`bf16_store_pitfall.cu`、`mla_wgmma_sw.cu`）。下一篇：TMA + 多级流水 / warp specialization。
