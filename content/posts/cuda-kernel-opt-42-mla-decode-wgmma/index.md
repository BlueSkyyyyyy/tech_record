---
title: "CUDA 算子调优（四十二）：MLA decode 的 wgmma 改造——消掉 QK dup，却撞上 V 转置税，再用混合 mma 绕开"
date: 2026-09-22T02:00:00+08:00
draft: false
weight: 42
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "Attention", "decode", "MLA", "DeepSeek-V4", "wgmma", "mma", "ldmatrix", "SW128", "swizzle", "tensor-core", "cp.async", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」推理方向的第三篇。上一篇 [第 39 篇]({{< relref "cuda-kernel-opt-39-mla-decode" >}})
用 `mma.sync + ldmatrix` 把 DeepSeek-V4 的 **MLA 吸收式 decode** 做成了单 kernel 融合
（`H=128, DC=512, DR=64, DV=512`），最佳 **3.076 ms / 189.9 TFLOPS**，
但留了两个明确的诊断：

1. **`short_scoreboard` 主导、L1/TEX 70.2%、254 寄存器**——`ldmatrix` 每个 KV tile 都要把
   Q 片段从 smem 重读进寄存器，寄存器被 `O[128]` 和 Q 片段占满，occupancy 只有 12.5%；
2. **FP8 KV 反而更慢**（3.87 vs 3.10 ms）——因为算子只有 ~8% 的算力效率，
   「换存储不换算力」是白费。

这一篇就是去啃第 1 堵墙：**把 QK 换成 `wgmma` SS**（操作数直接从 smem 描述符读，
绕开 `ldmatrix` 和寄存器墙），并且顺手消掉 [第 16 篇]({{< relref "cuda-kernel-opt-16-mla-fused" >}})
起融合 MLA 一直存在的 **QK dup**（两个 warpgroup 各算一份完整 `S`，工作量 ×1.53）。

剧透：这条路**前半程翻了车**——纯 wgmma 版反而比 39 慢；真正的转折点是一个诊断实验，
它把瓶颈精确定位到「V 必须转置」这一条 [第 32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}})
已经判过死刑的路上。最后的解法是**混合**：QK 用 wgmma，PV 用回 `mma + ldmatrix.x4.trans`，
但让 PV 直接从 wgmma 用的那块 SW128 K tile 里读 V。最终 **2.009 ms / 290.7 TFLOPS**，
比 39 篇 **快 1.53×**。

先把结论摆在这（H100 SXM，`B=64, L=32768`，KV 2.416 GB，`AI≈242`）：

| 版本 | 手段 | 耗时 | TFLOPS | 相对 39 |
|---|---|---|---|---|
| 39 篇最佳 | `mma.m16n8k16` + `ldmatrix` + `cp.async` | 3.0764 ms | 189.9 | 1.00× |
| 42 纯 wgmma SS · 拆键 4 WG | wgmma `m64n16k16` SS + V 转置进 smem | 3.2221 ms | 181.3 | 0.95× |
| 42 纯 wgmma SS · 拆键 2 WG | wgmma `m64n32k16` SS + V 转置进 smem | 3.5244 ms | 165.7 | 0.87× |
| 42 纯 wgmma SS · dup2（对照） | 两 WG 各算全部 KV | 3.6633 ms | 159.5 | 0.84× |
| （诊断）关掉 V 重载 · 拆键 2 WG | 孤立天花板，结果无意义 | 1.5863 ms | **368.2** | 1.94× |
| **42 混合 + K 双缓冲 · 拆键 2 WG** | **wgmma QK + mma/ldmatrix.trans PV（V 免转置）+ K 双缓冲** | **2.0094 ms** | **290.7** | **1.53×** |
| 42 混合 · 拆键 4 WG | 同上，4 个 warpgroup | 2.6176 ms | 223.1 | 1.18× |

---

## 1. 背景：39 篇留下的两堵墙，以及为什么是 wgmma

先把 shape 再钉一次。真实参数取自 `/ssd/models/DeepSeek-V4-Pro/config.json`
（`num_attention_heads=128, head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1`）。
吸收后所有头共享一份潜 KV：

```
QK:  S[h, j] = q_nope[h]·c_kv[j] + q_pe[h]·k_pe[j]     h=0..127, j=0..L-1
     DK = DC + DR = 576，1 个 query token / head / batch
PV:  O[h, :] = Σ_j softmax(S)[h, j] · c_kv[j]           DV = DC = 512
```

39 篇的核心矛盾是 **`O[H][DV] = 128×512 fp32 = 256 KB` 的累加器**：它把寄存器吃光
（254 regs），逼着 kernel 把 KV 按 `DVGRP` 列切给不同 warp，每个 warp 又要独立算一份
`S`（`dup=DV/DVGRP`）；同时每个 KV tile 都要 `ldmatrix` 把 Q 片段从 smem 拉到寄存器。

`wgmma` 的诱惑正在这里：**SS 形式的 `wgmma` 的 A、B 操作数都从 smem 描述符直接读，
不经过寄存器**，理论上能同时省掉 Q 片段的 `ldmatrix` 和它的寄存器占用。代价是一个
[第 32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}}) 已经判决过的事实（技巧台账 J4）：

> **Hopper `wgmma` 的 B 操作数只认 K-major（MN-major 描述符在 swizzle 布局下被硬件忽略）。**

对 QK 这没问题：`c_kv[j][dk]` 的 `dk` 连续，天然是 K-major 的 B。但对 PV 就是灾难：
`O = P·V` 里 V 的「收缩维」是 key，需要 `V^T[DV, key]` 的 K-major 布局，
即必须把 `c_kv[KT][DC]` **转置**成 `[DV][KT]`。而转置 = 每条 key 的 512 个值里
挑同一列 → 「8 条跨 512×2=1024 字节的标量 global 读 + 一次 SW128 16B 写」。

这正是 39 篇用 `ldmatrix.x4.trans` 绕开的税：`ldmatrix.trans` 让硬件顺带做转置，
V 可以原样以 `[KT][DC]` 行主序躺在 smem 里。所以这里有个天然的对立：

```
        QK                       PV
wgmma:  SS 免 ldmatrix    必须 V^T（转置税）
mma:    ldmatrix 重读 Q    ldmatrix.x4.trans 免转置
```

42 篇的主线就是把这两者**拼起来取长**。

---

## 2. 第一版：纯 wgmma SS——翻车

先老实做一个「纯 wgmma」版，结构照搬 20/21 篇的 prefill 融合 MLA，但换成 decode 的
batch/head 映射，并把 QK 拆键（split-KV）：

```
CTA = (batch b, head block BM=64)           grid = (H/BM, B) = (2, 64)
NWG 个 warpgroup，每个算 KT/NWG 个 key 的 S（wgmma.m64n{k}k16）
  → 跨 WG 用 smem 合并 row max / sum
  → 各写自己那段的 P 到 smem（SW128）
  → 各做 DV/NWG 列的 PV（wgmma.m64n128/256k16，B=V^T）
smem: Q[64,576]SW128 + K[64,576]SW128 + P[64,64]SW128 + V^T[512,64]SW128 ≈ 216KB
```

拆键让每个 key 的 QK 只算一次（对比融合 MLA 的 dup=2），V 的转置 gather 就是前面说的
「8 条标量读 + 16B 写」。实测：

| 配置 | 耗时 | TFLOPS |
|---|---|---|
| 拆键 4 WG（`m64n16`） | 3.2221 ms | 181.3 |
| 拆键 2 WG（`m64n32`） | 3.5244 ms | 165.7 |
| dup 2 WG | 3.6633 ms | 159.5 |

**全都不如 39 篇的 189.9。** ncu 判决很干脆（`B=64,L=8192`，拆键 4 WG）：

```
Tensor pipe (hmma)  : 19.1 %     ← 张量核空转
Achieved occupancy  : 24.97 %
stall long_scoreboard: 2.20      ← 在等 global，主导
stall wait          : 1.33
stall barrier       : 1.20
```

`long_scoreboard 2.20` 说明线程在等 global 访存。那是谁？**只有 V 的转置 gather**。
为确认，做一个「诊断版」——保留全部代码，只把每轮迭代里「载入下一块 V」那句 `if` 关掉
（结果当然错，但能给出「如果没有 V 重载」的天花板）：

| 配置 | 完整版 | 关掉 V 重载（诊断天花板） |
|---|---|---|
| 拆键 4 WG | 181.3 | **281.5** |
| 拆键 2 WG | 165.7 | **368.2** |
| dup 2 WG | 159.5 | **321.9** |

**V 的转置重载就是全部的墙**：去掉它，QK 拆键 2 WG 版能到 **368 TFLOPS**（39 的 1.94×）。
所以问题从「wgmma 行不行」变成了「怎么让 V 不要转置、或者让转置不落地到 global」。

---

## 3. 解法：混合——wgmma 算 QK，PV 从 SW128 K tile 里直接读 V

### 3.1 关键观察

`c_kv[KT][DK]` 这块 smem tile 同时是：

- QK 的 B 操作数（`dk` 连续 → K-major SW128）；
- PV 的 V（前 `DC=512` 列就是 V，`dv` 是输出维）。

`ldmatrix` 的硬件语义是「每个 lane 提供一个 16 字节行的地址，拼成 8×8 的 b16 矩阵」。
**SW128 只是把 16B 列按行号做了 `c' = c ^ r` 的异或置换，每个 16B chunk 本身完好**。
因此只要给 `ldmatrix` 正确的 16B 地址，它就能从 SW128 tile 里读出任意片段——
包括 PV 需要的、语义上「转置」的 V。

于是可以：

```
QK:  wgmma.m64n{k}k16 SS     A=Q(SW128)  B=K(SW128)   ← K-major，无需转置
PV:  mma.m16n8k16            A=P(row-major) B=V ← ldmatrix.x4.trans 从同一块 SW128 K tile
```

V 的地址由 `sw128_off(key, dv, DK)` 给出（就是把元素坐标翻译成 SW128 物理字节偏移），
再喂给 `ldmatrix.x4.trans`。`c_kv` **一个 KV tile 只从 global 读一次**，转置税归零。

```cpp
// PV：A=P（row-major，ldmatrix.x4），B=V（SW128 K tile，ldmatrix.x4.trans）
const int rrow = (lane & 7) + ((lane >> 3) & 1) * 8;
const int ccol = (lane >> 4) * 8;
for (int c = 0; c < KT / 16; ++c) {
  uint32_t pa[4];
  ldmatrix_x4(smem_u32(&ps[16*(W&3) + (lane&15)][c*16 + (lane>>4)*8]), pa);
  for (int dn = 0; dn < DVW / 16; ++dn) {
    uint32_t d[4];
    const int keyrow = c*16 + rrow;
    const int dv = wg*DVW + dn*16 + ccol;
    ldmatrix_x4_trans(cur + sw128_off(keyrow, dv, DK), d);   // V 免转置
    mma16816(O + (dn*2)*4,     pa, d);
    mma16816(O + (dn*2+1)*4,   pa, d + 2);
  }
}
```

> 踩坑：`mma16816(c, a, b)` 的 `c` 必须指向**连续的 4 个 float**。39 篇的 `O` 是
> `float O[NDH][4]`，传 `O[dn*2]` 得到一行 `float[4]`；我这里 `O` 是扁平的
> `float O[DVW/8*4]`，误写成 `O + dn*2` 会让相邻两个 n8 tile 的累加器**重叠**
> （`O+dn*2` 与 `O+dn*2+1` 差 1 而非 4），结果全错、而且只在特定 shape 才明显。
> 正解是 `O + (dn*2)*4`。

另一个坑来自跨 warpgroup 的 reduction：我一开始写
`red[wg*BM + 16*W + r0] = b0`。`r0 = 16*(W&3) + (lane>>2)` 已经是「CTA 内 0..63 的行号」，
再加 `16*W` 会把它推到 96、144 越界别名，跨 WG 读到别的行的 max/sum。
**症状极度隐蔽**：`L=32768` 时误差只有 7e-5（softmax 对 max 的小扰动不敏感），
但 `L=64`（单 tile）时相对误差 13% 直接 FAIL。改成 `red[wg*BM + r0]` 后 L=64 也回到 8e-5。

### 3.2 再叠 K 双缓冲

混合版的 smem 只有 `Q 73.7K + K 73.7K + P 9K = 156 KB`，还剩 **~71 KB**；
而一块 K SW128 正好 73.7 KB——**刚好塞得下第二块 K**（`227 KB` 上限，实测
`3×73728 + 9216 + red = 232448 B` 恰好顶格）。于是做 K 双缓冲：
QK 读完 buffer A 后立刻把下一块 K `cp.async` 进 buffer B，与 softmax/PV 重叠。
ncu 里 `long_scoreboard` 从纯 wgmma 的 **2.20** 掉到 **0.10**。

```
   ┌──────────────── 一个 KV tile 的流水 ────────────────┐
   QK(i) [wgmma]      softmax + P + O rescal   PV(i) [mma]
   │                  │        │                │
   └ prefetch K(i+1) ─┘        │                │
        (cp.async → 另一 buffer)│                └ prefetch K(i+2)→本 buffer
```

---

## 4. 实测：1.53×

### 4.1 主结果

`B=64, L=32768`（KV bf16 2.416 GB）：

| 版本 | 耗时 | 有效带宽 | TFLOPS | %bf16 峰值 |
|---|---|---|---|---|
| 39 篇 `mma+ldmatrix` | 3.0764 ms | 785 GB/s | 189.9 | 19.2% |
| **42 混合 + K 双缓冲（2 WG）** | **2.0094 ms** | **1202 GB/s** | **290.7** | **29.4%** |
| 42 混合（4 WG） | 2.6176 ms | 923 GB/s | 223.1 | 22.6% |

**1.53×**，正确性 `max_abs_err = 2.7e-6`（bf16，参考为 fp64 两趟 softmax）。

扫其它 shape（2 WG 混合版）：

| 配置 | 39 篇 | 42 混合 | 加速 |
|---|---|---|---|
| B=64, L=4096 | 175.2 | **261.2** | 1.49× |
| B=64, L=8192 | — | **277.3** | — |
| B=64, L=32768 | 189.9 | **290.7** | 1.53× |
| B=128, L=32768 | 191.2 | **293.9** | 1.54× |
| B=16, L=32768 | 47.2 | **70.8** | 1.50× |

各 shape 一致 ~1.5×，`B=16` 时格点不足（32 个 CTA 打不满 132 个 SM）所以绝对吞吐低，
但相对提升一致——说明收益来自 kernel 本身，不是并发度。

### 4.2 ncu 对照

`B=64, L=8192`，同为 2 warpgroup：

| 指标 | 纯 wgmma（V 转置） | 混合（V 免转置） |
|---|---|---|
| Tensor pipe (`hmma`) | 19.1 % | **36.7 %** |
| `long_scoreboard` | 2.20 | **0.10** |
| `wait` | 1.33 | 0.83 |
| `barrier` | 1.20 | 0.33 |
| `not_selected` | 0.87 | 0.45 |
| `math_pipe_throttle` | — | 0.39 |
| DRAM Throughput | 34 % | 34.3 % |
| L1/TEX | — | 61 % |
| occupancy | 25.0 % | 12.5 % |
| registers | 206 | 254 |

混合版把「等 global」彻底消掉，张量核活跃度翻了近一倍。**下一道墙变成 `wait`（等
`ldmatrix`→`mma` 依赖）和低 occupancy**（8 warp/SM，只有 12.5%），也就是经典的
「单 CTA/SM + 单缓冲」几何约束——和 prefill MLA 卡在同一个地方。

### 4.3 对标 SOTA 与 roofline

- **算力**：290.7 TFLOPS = 29.4% bf16 峰值；39 篇是 19.2%。
- **内存 roofline**：KV 2.416 GB / 3352 GB/s = **0.721 ms**，我们 2.009 ms，差 **2.79×**。
  注意 `AI = 2H(DK+DV)/DK = 241.8` 略低于 bf16 ridge（`989/3.352 = 295`），
  所以极限工作点本就该是「接近内存受限」；我们离它还有 2.8×。
- **对 FlashMLA**：39 篇给出的口径是距 FlashMLA sm90 decode ~3.4×（39 帖带宽）。
  本篇相对 39 收窄 1.53×，折算后**距 FlashMLA 约 2.2×**（同口径、未在本机直接跑
  FlashMLA 的 bf16 dense decode，仅作量级参考）。
- **FP8**：仍然不划算——KV 字节只占了 35.9% HBM（1202/3352），张量核 36.7%，
  两个都不到一半。**要先让 tensor 或带宽任一项逼近饱和，FP8 才有意义**（见 39 篇判据）。

---

## 5. 复现

```bash
cd ~/proj/tech_record/code/kernel-opt
scripts/lab.sh up
# 纯 wgmma（会看到 V 转置税）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 42-mla-decode-wgmma/mla_decode_wgmma.cu 64 32768 bf16
# 混合版（主结果）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 42-mla-decode-wgmma/mla_decode_hybrid.cu 64 32768 bf16
# PV-from-SW128 的最小复现（验证 ldmatrix 能从 swizzle tile 读转置）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a" \
  scripts/run.sh 42-mla-decode-wgmma/pvt_test.cu
```

代码与原始输出：`code/kernel-opt/42-mla-decode-wgmma/`。

---

## 小结

- **纯 wgmma SS 修不了 decode**：它免掉了 Q 的 `ldmatrix`/寄存器墙，但引入 V 必须转置进
  `V^T[DV,KT]` 的代价；实测 **181.3 < 39 篇的 189.9**。诊断实验（关掉 V 重载）把天花板
  亮出来：**281–368 TFLOPS**，证明墙就是这条转置。
- **混合才是答案**：QK 用 `wgmma`（B 天然 K-major），PV 用 `mma + ldmatrix.x4.trans`
  从**同一块 SW128 K tile** 读 V（SW128 只是地址置换，chunk 完好）。V 一个 tile 只读一次、
  不转置。再叠 **K 双缓冲**（smem 恰好顶格 227KB）把全局延迟藏掉。
- 最终 **2.009 ms / 290.7 TFLOPS**，相对 39 篇 **1.53×**；`long_scoreboard 2.20→0.10`，
  tensor pipe `19.1%→36.7%`。
- 两个坑：`mma16816` 的累加器指针要 `O + (dn*2)*4`（不是 `O + dn*2`，否则 n8 累加器重叠）；
  跨 WG 的 `red` 行号别再加 `16*W`（`r0` 已是 CTA 内行号，加上会越界别名，且只在短序列暴露）。
- **下一步**：`wait`/低 occupancy 是新墙——要么 producer/consumer warp specialization +
  更大几何（把 `O` 的寄存器墙继续拆），要么接受「KV 只读一次」后重估 FP8（需 tensor 先逼近饱和）。
  工程上还有一条更彻底的：用 CuTe 生成 **MN-major SW128 描述符**让 PV 直接吃 `[key, dv]`
  的 K-major 数据——FlashMLA 正是这么做的，值得单独立一篇。

**下一篇预告**：回到量化推理线——W4A16 的跨 work-item 持久化流水（41 篇遗留的波次尾巴），
或者 MLA decode 的 producer/consumer warp specialization。

> 环境：H100 80GB HBM3 SXM（CC 9.0，132 SM，HBM ~3.35 TB/s，BF16 TC dense ~989 TFLOPS）；
> 容器 `kernel_lab`（CUDA 13.2）。所有数字来自本机实测，原始输出见
> `code/kernel-opt/42-mla-decode-wgmma/*.out.txt`。
