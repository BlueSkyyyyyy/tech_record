---
title: "CUDA 算子调优（三十九）：MLA 吸收式 decode + FP8 KV-cache——一个张量核也吃不饱的「平衡」算子"
date: 2026-09-22T01:00:00+08:00
draft: false
weight: 39
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "Attention", "decode", "MLA", "DeepSeek-V4", "KV-cache", "FP8", "量化", "MQA", "mma", "tensor-core", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」推理方向的第二篇。上一篇 [第 38 篇]({{< relref "cuda-kernel-opt-38-paged-kv" >}})
把 **GQA decode**（Qwen3-8B，`G=5`，AI≈2.5）打到了 **92.8% HBM**——那是一个典型到不能再典型的
**纯访存**算子：算力强度低 60 倍，只要把「每个 KV 字节摊的指令数」压下去就赢。

这一篇换个模型：**DeepSeek-V4-Pro 的 MLA 吸收式 decode**。它和 GQA decode 长得像（`q_len=1`、
KV 分页、要一遍遍重读），但骨子里完全不同：

- MLA 用了 **latent KV**（`c_kv` 只有 512 维）+ decoupled RoPE（`k_pe` 64 维），所以 **128 个 head 共享同一份 KV**（MQA，`Hkv=1`）；
- 吸收后 `Q` 是 `[H=128, DK=576]`，`V` 维度是 `DV=512`。

结果是算术强度被拉到 **AI≈242**，几乎顶到 H100 bf16 tensor core 的 ridge（≈295）——
**这不是纯访存，而是一个卡在算力/带宽边界上的「平衡」算子**。本文会把这件事从头推一遍，
然后诚实地回答 ROADMAP 里那个设想：**「KV 换 FP8、字节减半，decode 是不是直接 ≈2×？」**

先给结论（真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：`num_attention_heads=128,
head_dim=512, qk_rope_head_dim=64, num_key_value_heads=1`；`B=64, L=32768`，KV bf16 = 2.42 GB）：

| 版本 | 手段 | 耗时 | bf16/FP8 KV 带宽 | 算力 | 占 bf16 峰值 |
|---|---|---|---|---|---|
| v1 | `mma.m16n8k16`，同步向量化装载 | 3.53 ms | 684 GB/s | 165 TFLOPS | 16.7% |
| **v2** | v1 + `cp.async` 双缓冲（`KT=32, DVGRP=2`） | **3.10 ms** | **779 GB/s** | **188 TFLOPS** | **19.0%** |
| v2' | `KT=64, DVGRP=2, ROWG=2`（4 warp） | 3.18 ms | 761 GB/s | 184 TFLOPS | 18.6% |
| v3 | FP8 KV（e4m3 + per-channel），非 pipeline | 3.53 ms | 343 GB/s | 166 TFLOPS | 16.7% |
| v3p | FP8 + 分页 | 4.26 ms | 284 GB/s | 137 TFLOPS | 13.9% |

**四个结论，先摆在这**：

1. **MLA decode 不是带宽受限**：ncu 实测最佳配置 `DRAM Throughput` 只有 **22.7%**、`L1/TEX` **70%**、
   tensor pipe 29%、occupancy **12.5%**——它是 **共享内存/发射延迟受限**（`short_scoreboard` 2.52
   占主导），离带宽顶还差得远。
2. **FP8 KV 在这个 kernel 上是负优化**：字节是减半了，但算力效率只有 8%，反而被多出来的反量化
   指令拖慢。**FP8 KV 只在「真带宽受限」或「上 FP8 张量核」时才划算**——本文给出定量判据。
3. 根因是一个 **`H×DV = 128×512 = 65536` 个 fp32 的巨型累加器**：它把寄存器和共享内存一起顶死，
   几何上逼出「1 CTA/SM、8~16 warp」的低占用。
4. 顺手挖出一个很隐蔽的几何坑：**`KT/DVGRP` 必须是 16 的倍数**，否则 QK 内层循环直接被展开成 0 次
   （kernel 不报错、结果全错、还"快"一倍）。

---

## 一、MLA 吸收式 decode 是什么

MLA（Multi-head Latent Attention）的核心是**把 KV 压成一个 latent**。DeepSeek-V4 里：

- 每个 token 的 KV 被压成 `c_kv ∈ R^{512}`（`kv_lora_rank`，也叫 NoPE 部分）+ `k_pe ∈ R^{64}`（decoupled RoPE）；
- query 侧 `q_nope` 通过吸收矩阵 `W_uk` 变到同一个 512 维空间，再加 64 维 rope。

于是单 head 的 score 与输出：

$$
s_{h,j} = \underbrace{q^{nope}_h \cdot c_{kv,j}}_{\text{NoPE, }\in R^{512}} + \underbrace{q^{pe}_h \cdot k_{pe,j}}_{\text{RoPE, }\in R^{64}},\qquad
o_h = \sum_j \mathrm{softmax}(s_{h,\cdot})_j \, c_{kv,j}
$$

也就是：**`DK = 512 + 64 = 576`，`DV = 512`，并且 K 和 V 是同一张 `c_kv`**。所有 128 个 head 共享
同一份 `c_kv`（`Hkv=1`，纯 MQA）。

decoder 阶段 `q_len=1`，所以一个请求每层只算一个 query token，但要对历史全部 key 做上面两件事。

### roofline

对 `B` 个请求、序列长 `L`：

$$
\text{FLOPs} = 2\cdot H \cdot (DK + DV) \cdot B L,\qquad \text{bytes}_{KV} = (DK) \cdot B L \cdot \text{sizeof}
$$

（QK 与 PV 各 2 FLOP/MAC；KV 只读一遍。）

$$
\text{AI} = \frac{2H(DK+DV)}{DK} = \frac{2\times 128 \times (576+512)}{576} \approx 242
$$

对比第 38 篇 GQA 的 `AI=5`。H100 bf16 ridge `989e12 / 3.35e12 ≈ 295`——
**MLA decode 的 AI 离 ridge 只差 1.2×，它落在「算力和带宽都要吃满」的窄缝里**。
`2.416 GB` 的 KV 理论最短读取时间 `= 2.416/3.35 = 0.72 ms`，算力理论最短 `= 584/989 = 0.59 ms`。

老老实实说：**我们这个 kernel 落在 3.10 ms，离两条 roofline 都在 4× 上下**——
算力和带宽都没打满。为什么，后面会逐条讲。

---

## 二、kernel 结构

CTA 的划分是「一个 batch × 一个 head block」：

```
grid  = (H / BH, B)          BH = ROWG * 16 个 head / CTA
block = ROWG * DVGRP * 32    线程

每个 CTA 循环整个序列 [0, L)，每轮吃 KT 个 key：
  ┌─ 载入 Q[BH][DK] 到 smem（整个 kernel 只搬一次，跨 tile 复用）
  │
  └─ for k0 in 0,KT,2KT,...,L:
       KV tile  [KT][DC] + [KT][DR]  -> smem
       QK^T : S[BH][KT] = Q · [c_kv | k_pe]^T          (mma.m16n8k16)
       尾 tile 掩码（key >= L 置 -inf）
       online softmax（行 = head，列 = key）：
           局部 max -> smem 交换 -> 全 max（跨 DVGRP 个 warp）
           exp / 求和 / 重标 O
       PV   : O[BH][DV] += P[BH][KT] · c_kv[KT][DV]     (mma, V 转置 ldmatrix)
```

关键复用点（沿用 16/19 篇）：

- **QK 的 `mma.m16n8k16`**：A=Q（`ldmatrix` 非转置），B=KV（`ldmatrix` 非转置，因为 `[key][dim]` 行主序天然满足 `.col`）；smem 行距 `DK+8` 消 bank conflict。
- **PV 的累加器 `O` 布局**：`P` 的 softmax 结果直接 `pack2` 成 PV 的 A 片段，**零 shuffle**（16 篇的老技巧）。
- **V 要转置**：`mma` 的 B 需要 `[k=key][n=dv]` 的 `.col` 布局，而 `c_kv` 是 `[key][dv]` 行主序，用
  `ldmatrix.x4.trans` 让硬件顺手转置。
- **跨 warp softmax**：`DVGRP` 个 warp 各算 KV 的一个纵切，`P` 写 smem 共享，max/sum 通过
  `pmax/psum` 小数组做一次跨 warp 交换（2 个 `__syncthreads`）。

`ROWG` 决定每个 CTA 管几个 head（越大，KV 复用越好、`O` 寄存器越多）；`DVGRP` 决定把 `DV=512`
切成几份（越大，每 warp 的 `O` 越少、但要更多 warp 做 QK 的列分工）。

代表性代码（`39-mla-decode/mla_decode.cu:279` 的 `compute_tile`，QK 主循环）：

```cpp
for (int kk = 0; kk < DK / 16; ++kk) {            // 36 个 k16
  uint32_t a[4];
  ldmatrix_x4(smem_u32(&qs[rowg*16 + (lane&15)][kk*16 + (lane>>4)*8]), a);   // Q
  for (int nb = 0; nb < KH / 16; ++nb) {          // 每个 warp 负责 KH 个 key
    uint32_t d[4];
    ldmatrix_x4(smem_u32(&kbuf[nbase + nb*16 + n_off][kk*16 + k_off]), d);   // KV
    mma16816(S[nb*2],     a, d);
    mma16816(S[nb*2 + 1], a, d + 2);
  }
}
```

---

## 三、踩坑一：`KT/DVGRP` 必须是 16 的倍数

`KH = KT / DVGRP` 是每个 warp 负责的 key 数。QK 内层按 **16 个 key 一个 `nb` 步**展开：

```cpp
constexpr int KH = KT / DVGRP;                   // 例如 KT=32, DVGRP=4 -> KH=8
for (int nb = 0; nb < KH / 16; ++nb) { ... }     // KH=8 -> KH/16 = 0，一次都不进！
```

我第一次扫参时看到 `ROWG=4, DVGRP=4, KT=32` 跑出 **308 TFLOPS（+60%）**，大喜过望。
结果一查：`KH = 8`，`KH/16 = 0`，**QK 的 mma 一次都没发射**——那个配置在「算一半 attention」
（只算了 PV），当然快。

> **教训**：attention kernel 的 `N`/`K` 分块要显式 `static_assert(KH % 16 == 0)`。
> 更阴的是：扫描脚本里如果**没接对拍**，你会一路把错误配置当成最佳结果。
> 本篇最终代码在 `mla_decode.cu:111` 加了：

```cpp
static_assert(KH % 16 == 0, "KH 必须是 16 的倍数（QK 内层按 16 key/nb 展开）");
static_assert(KT % 16 == 0 && DV / DVGRP % 16 == 0, "PV tile 必须是 16 的倍数");
```

加上约束后，合法几何只剩下面这些（`DVGRP=2` 时 `KT≥32`；`DVGRP=4` 时 `KT≥64`）。
真正的成绩是 **~190 TFLOPS**，而不是那个虚高的 308。

---

## 四、扫参：为什么 8 个 warp 反而赢 16 个

`B=64, L=32768`，KV bf16，逐配置实测（完整原始输出见 `scan_B64_L32768.out.txt`）：

| ROWG | DVGRP | KT | warps | O/warp | smem | pipe | 耗时 | 算力 |
|---|---|---|---|---|---|---|---|---|
| 4 | 2 | 32 | 8 | 128 reg | 157 KB | ✓ | **3.06 ms** | **191.0** |
| 2 | 2 | 32 | 4 | 128 reg | 117 KB | ✓ | 3.15 ms | 185.4 |
| 2 | 2 | 64 | 4 | 128 reg | 119 KB | ✓ | 3.17 ms | 184.1 |
| 2 | 4 | 64 | 8 | 64 reg | 119 KB | ✓ | 3.38 ms | 172.8 |
| 4 | 4 | 64 | 16 | 64 reg | 161 KB | ✗ | 3.45 ms | 169.3 |
| 4 | 2 | 64 | 8 | 128 reg | 161 KB | ✗ | 3.74 ms | 156.2 |
| 2 | 4 | 128 | 8 | 64 reg | 198 KB | ✗ | 4.59 ms | 127.4 |
| 2 | 8 | 128 | 16 | 32 reg | 196 KB | ✗ | 4.28 ms | 136.4 |

两条规律：

1. **`cp.async` 双缓冲是刚需**：同样 `ROWG=4, DVGRP=2`，`KT=32+pipe` 3.06 ms vs `KT=64` 无 pipe
   3.74 ms；`ROWG=2, DVGRP=4` 有无 pipe 是 3.38 vs 5.76（差点 1.7×）。global 延迟必须藏。
2. **`O` 寄存器数比 warp 数更重要**：`DVGRP=4`（`O=64 reg`）能开 16 warp，但它的 `KT` 被迫到 64，
   双缓冲 smem 超 227 KB 上限、只能退回无 pipe，反而慢。
   两个约束是**互相打架**的——这就是本篇的核心矛盾。

### 根因：一个 256 KB 的累加器

一个请求的全部输出是 `H × DV = 128 × 512 = 65536` 个 fp32 = **256 KB**。它必须在 online softmax
过程中一直活着。放进寄存器：

$$
\text{regs/thread} = \frac{65536}{T}
$$

`ROWG=4, DVGRP=2` 用 256 线程 → 每线程 `O` 就 128 个寄存器；加上 `S`、Q 片段、地址/循环开销，
ncu 实测**每线程 254 个寄存器**（`ncu_bf16_best.out.txt`）：

```
Registers Per Thread             register/thread      254
Block Limit Registers            block                  1
Achieved Occupancy               %                  12.50     <- 8 warp / 64
sm__pipe_tensor_cycles_active    %                  29.20
L1/TEX Cache Throughput          %                  70.16
DRAM Throughput                  %                  22.71
short_scoreboard (stall)         inst                 2.52     <- 主导
wait (stall)                     inst                 1.63
```

`short_scoreboard` = 等共享内存（MIO）依赖，典型来源就是 **`ldmatrix`**。而它为什么这么多？
因为 **Q 的片段每个 KV tile 都要重读一遍**：

```cpp
// compute_tile 每进一次就要 36 次 Q 的 ldmatrix，而这些 Q 值跨 tile 根本不变
for (int kk = 0; kk < DK/16; ++kk)
  ldmatrix_x4(&qs[rowg*16 + ...][kk*16 + ...], a);   // 每 tile 重复 36 次
```

`L/KT = 32768/32 = 1024` 个 tile × 36 次 = 每个 warp 光 Q 就重读 **36864 次 ldmatrix**。
理想做法是把 Q 片段在寄存器里常驻，但 36×4 = **144 个寄存器**——和 128 个 `O` 撞车，放不下。

这是一个**结构性**的墙：`O` 和「常驻 Q 片段」都想待在寄存器里，而两者之和超过了 255 上限。

---

## 五、FP8 KV-cache：字节减半，然后呢？

ROADMAP 当初的设想很直接：**decode 一遍遍读 KV，把它从 bf16 换 e4m3，字节减半 ≈ 2×**。
我们按 DeepSeek 的量化口径把它实现出来：

- `c_kv` / `k_pe` 各存 **per-channel（每个维度一个）scale**，`e4m3`；
- 装载时反量化。为了不引入逐元素乘法，把 scale **折进 Q**：`score = (q·s) · k8`，输出端再乘回
  `s`（因为 K/V 是同一张 `c_kv`，两个 scale 相同）——**反量化只剩「e4m3→bf16 转换」**，
  思路和 [第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}}) 的「把 2 的幂 scale 折进操作数」同源。
- 量化误差（per-channel e4m3，带 outlier）：`c_kv` 最大相对误差 **5.85%**、`k_pe` 5.85%。

实测（`B=64, L=32768`）：

| 配置 | KV 类型 | 耗时 | 带宽 | 算力 |
|---|---|---|---|---|
| `ROWG=4,DVGRP=2,KT=32,pipe` | bf16（2.42 GB） | **3.10 ms** | 779 GB/s | 188 TFLOPS |
| 同配置 | FP8（1.21 GB） | 3.87 ms | 312 GB/s | 151 TFLOPS |
| `ROWG=4,DVGRP=4,KT=64` | bf16 | 3.45 ms | 700 GB/s | 169 TFLOPS |
| 同配置 | FP8 | 3.53 ms | 343 GB/s | 166 TFLOPS |
| `ROWG=4,DVGRP=2,KT=32,pipe` + 分页 | FP8 | 4.26 ms | 284 GB/s | 137 TFLOPS |

**FP8 全线更慢**（同配置慢 10~25%），分页再叠加约 **10%**（4.26 vs 3.87）。

### 为什么？一条定量判据

因为 kernel 根本不在带宽上。用 roofline 的两条线就能算清楚「FP8 什么时候值得」：

$$
t_{\text{compute}} = \frac{\text{FLOPs}}{\text{peak}\cdot \eta_c},\qquad
t_{\text{mem}} = \frac{\text{bytes}}{\text{BW}\cdot \eta_m}
$$

取 `η_c` = 张量核效率、`η_m` = 带宽效率（H100 上 ~0.8 现实）。代入 `B=64,L=32768`
（`FLOPs=584 TFLOP`，KV `2.416 GB`）：

| KV 精度 | 算力峰 | 内存时间（η_m=0.8） | 达到内存线所需的算力效率 η_c | 说明 |
|---|---|---|---|---|
| bf16 | 989 | 0.90 ms | 0.66 | 582 TFLOPS 才算力=内存 |
| FP8 | 1978 | 0.45 ms | 0.33 | 653 TFLOPS 才算力=内存 |

**翻译成人话**：

- 只有当 kernel 的算力效率 `η_c ≳ 33%`（FP8 峰值的 653 TFLOPS）时，FP8 KV 的「字节减半」才开始兑现；
- 我们现在的 FP8 kernel 只有 **8.4%**（166 / 1978），差 4 倍，所以只吃到了反量化的开销、没吃到带宽的红利。
- 反过来说：**第 38 篇那种 `AI=5` 的纯访存 decode，FP8 KV 会直接 ≈2×**（那里 `η_m` 才是天花板）。

这解释了为什么业界上 FP8 KV 时通常**同时**上 FP8 张量核（`mma.m16n8k32` / wgmma）——
单换存储、不换算力，在 MLA 这种高 AI 场景是白费。

> 顺带一个反例佐证：`ROWG=2, DVGRP=4, KT=64` 的 FP8 版本只有 108 TFLOPS（比 bf16 慢 60%），
> 因为它 warp 少、反量化指令占比更高。反量化在**指令发得紧的配置**上更贵。

---

## 六、ncu 对照与 SOTA 差距

`B=64, L=32768`，bf16 最佳 vs FP8 最佳（`ncu_bf16_best.out.txt` / `ncu_fp8_best.out.txt`）：

| 指标 | bf16 `4,2,32,pipe` | FP8 `4,4,64` |
|---|---|---|
| Duration | 3.19 ms | 3.65 ms |
| Compute (SM) | 30.5% | 31.5% |
| L1/TEX Throughput | **70.2%** | 66.0% |
| DRAM Throughput | 22.7% | 10.0% |
| L2 Throughput | 28.7% | 11.6% |
| tensor pipe active | 29.2% | — |
| Registers / Thread | 254 | 128 |
| Achieved Occupancy | 12.5% | 25.0% |

两边都是 **L1/TEX（共享内存路径）先撞墙**，不是 DRAM。FP8 那一列 `DRAM` 只有 10%——
字节减半的结果是「更空的内存」，这本身就说明瓶颈不在内存。

### 距离极限多远

| 口径 | 时间 | 我们（3.10 ms）的倍数 |
|---|---|---|
| 内存 roofline（3.35 TB/s，2.42 GB） | 0.72 ms | 4.3× |
| 算力 roofline（bf16 989） | 0.59 ms | 5.2× |
| `mma.sync` bf16 现实效率（~50% = 494） | 1.18 ms | 2.6× |
| 假设 SOTA（FlashMLA 级、贴 80% HBM） | ~0.90 ms | 3.4× |

**结论：离「这条 kernel 用 `mma.sync` 应该能到的水平」差约 2.6×，离真正贴带宽的 SOTA 差约 3.4×。**
差距全部来自同一个地方——**累加器太大导致 occupancy 只有 12.5%，而 Q 片段又因为寄存器不够不能常驻**。

要跨过去，只有三条路（都超出本篇范围，留作下一轮）：

1. **`wgmma`**：操作数直接从 smem 描述符读，**砍掉全部 `ldmatrix`**（本篇最大的 stall 源），
   同时 FP8 的 K-major SW128 atom 与 bf16 同构、能直接用 FP8 张量核——正好解决第五节「换存储不换算力」的问题；
2. **sequence split（flash-decoding）**：把长序列切给更多 CTA，摊薄尾部长尾（本篇 0.97 wave，尾部损失有限，
   但长上下文下会更明显）；
3. **1-warpgroup / 248-register 布局**（[第 24/37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}}) 的老朋友）：
   给 math warpgroup 多发寄存器，把 `O` + 常驻 Q 片段一起塞下。

---

## 七、实测数据汇总

`mla_decode.cu` 自带参考实现对拍（内置独立的两趟 softmax 参考，抽样 3 个 `(head, batch)` 行，
`max_abs_err` 全部 < 2.4e-2 / 相对 < 0.5%）。

**B 维度扫描**（`L=32768`，最佳配置 `4,2,32,pipe`）：

| B | 网格 CTA | bf16 耗时 | 带宽 | 算力 | FP8 耗时 |
|---|---|---|---|---|---|
| 16 | 32 | 3.09 ms | 195 GB/s | 47 TFLOPS | 3.84 ms |
| 64 | 128 | 3.10 ms | 779 GB/s | 188 TFLOPS | 3.87 ms |
| 128 | 256 | 6.11 ms | 791 GB/s | 191 TFLOPS | 7.64 ms |

`B=16` 时 `grid.x=2`，只有 32 个 CTA，132 个 SM 空一半 → 算力腰斩；`B≥64` 后线性。

**L 维度扫描**（`B=64`，`4,2,32,pipe`）：`L=4096` 0.42 ms / 175 TFLOPS；
`L=32768` 3.10 ms / 188；`L=131072` 12.18 ms / 192 TFLOPS。**吞吐随 L 稳定**，
说明没有随长度劣化（KV 全读一遍，工作集线性增长）。

**分页开销**（fp8，`4,2,32,pipe`）：`B=64,L=32768` 连续 3.87 ms vs 分页 4.26 ms（**+10%**）；
`B=128` 7.64 vs 8.43 ms（+10.4%）。比第 38 篇 GQA 的 1.3% 高——因为这里 KV 行只有 576 字节，
分页时要按行跨页取，破坏了行内大段连续。

**量化质量**：per-channel e4m3 + 折 scale 后，kernel 输出与 host 参考（喂同一份反量化 KV）
对拍 `max_abs_err ≈ 2.9e-6`（相对 ~0.07%）——**kernel 数学与参考一致**，
真正的误差只剩 e4m3 量化本身（`c_kv` 最大相对误差 5.85%）。

---

## 八、小结

- **MLA 吸收式 decode 的 AI≈242**，贴着 bf16 tensor core 的 ridge，是个「算力/带宽平衡」算子——
  **不能照搬第 38 篇「纯访存」的直觉**。
- 一个 `H×DV=128×512` 的 fp32 累加器（256 KB）是所有矛盾的根源：它逼出 254 寄存器、12.5% occupancy，
  还挡住了「Q 片段常驻寄存器」的路，于是每个 tile 重读 36 次 Q → `short_scoreboard` 主导。
- **FP8 KV 在这个 kernel 上是负优化**（−10~25%）：字节减半只在「算力效率 ≥33%」或「真带宽受限」时才兑现。
  给出判据表——**上 FP8 存储要配套 FP8 张量核**。
- 一个隐蔽坑：**`KT/DVGRP % 16 != 0` 会让 QK 的 mma 循环展开成 0 次**，kernel 不报错、结果全错还"更快"。
- 本篇最佳 **191 TFLOPS（19.3%）/ 779 GB/s**，落在「离 `mma.sync` 现实上限 2.6×、离贴带宽 SOTA 3.4×」的地方。

**下一篇**：把这套 MLA decode 换成 `wgmma`——用 smem 描述符喂操作数，一次性干掉 `ldmatrix` 墙，
并顺势上 **FP8 张量核**，让第五节那张判据表真正兑现。
