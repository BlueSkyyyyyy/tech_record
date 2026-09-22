---
title: "CUDA 算子调优（六十二）：FP4 解码的整 chunk 位拼装 —— 把每 nibble 一次 LDS 换成两条 prmt，K1 从 73.6% 到 83.4% HBM"
date: 2026-09-22T11:00:00+08:00
draft: false
weight: 62
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "PRMT", "DeepSeek-V4", "GEMV", "grouped", "位运算", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 61 篇]({{< relref "cuda-kernel-opt-61-v4-fp4-ffn" >}}) 把 DeepSeek-V4-Pro 一步 decode 的整条
routed expert FFN（`w1/w3 → SwiGLU → w2`）跑通了，端到端 **5.97 ms / 67.6% 权重 roofline**。
ncu 给的判词很清楚：

```
K1 (w1+w3, N=6144, K=7168):  DRAM 71.4%  Compute 75.9%   ALU pipe 75.9%（LUT + dp4a 发射受限）
K2 (w2,    N=7168, K=3072):  DRAM 62.0%  occ 23.5%  long_scoreboard ~35%（延迟受限）
```

K1 的 `Compute ≈ DRAM` 意味着它**不在纯带宽墙上**：每读一个 16 B 权重 chunk（32 个 fp4 nibble），
61 的 `LUTMODE=4` 要查 **32 次共享内存**（16 项 nibble 表，一次/nibble），再加 ~32 条移位/掩码。
这些 LDS 走 LSU 管道、移位走 ALU 管道，把 SM 的发射口先占满了。

本篇就干这一件事：**把 e2m1 的解码从"每 nibble 一次 LDS"换成"整 16 B chunk 位拼装"** ——
只用 `__byte_perm`（PRMT）+ 几次掩码/乘法，**零 LDS** 地把 8 个 nibble 变成 8 个有符号 int8，
直接喂 `dp4a`。结果：

> **K1 3.64 → 3.21 ms（73.6% → 83.4% HBM）、K2 1.94 → 1.73 ms（69.0% → 77.6%）、
> 端到端 5.62 → 4.97 ms（71.8% → 81.1% 权重 roofline，1.13×）**；
> 共享内存 load 指令 **637M → 109M（−83%）**、总指令 **2.53e9 → 2.06e9（−18.5%）**。
> 解码本身不再是一堵墙——ncu 变成 **DRAM 83.6% ≈ L2 89.6%** 的访存受限。

先摆结论（都和 58/59/60 的"PRMT 换 LUT 是负优化"相反，原因见正文）：

1. **58/59/60 的 PRMT 输在别处**：那些场景要么 `RWW`/并行度不足（58），要么 L1/TEX 才是墙（59/60）。
   而 K1 是 **ALU/发射受限**——把 LDS 换成少量 ALU 恰好是往对的方向搬。
2. **PRMT 的 selector 只用 field 的低 3 位**（bit3 被忽略、值 8–15 回绕到 0–7）——这个硬件行为
   让"用原始 nibble 直接当 selector 查 8 项幅值表"成为可能，是整条路径的关键。
3. **K2 的并行粒度试了几种都输给现状**（PIPE0/更深 DEPTH/RWW 变化），它的墙是 `long_scoreboard`
   的访存延迟 + 34% occupancy；短 `NITER=3`（`K=I=3072`）让 cp.async 流水没有足够跑道。

---

## 一、背景：61 的 K1 到底在忙什么

K1 算的是 `GU[pairs, 2I] = X[B, H] · [w1; w3]ᵀ`，其中 `H=7168, I=3072`，权重是 FP4 e2m1：
每字节存两个 nibble，一个 16 B chunk 覆盖 32 个 k。61 的内层（`RWW` 行/warp）是：

```
读 16B uint4 权重 → 对每一个 uint32（4 字节 = 8 nibble）
   ├─ 8 次 LDS：16 项 nibble 表 l16[.] 查 e2m1→int8（每次只取低 8 位）
   ├─ 8 次 shift/and 抽 nibble + 若干 or 拼成两个 uint32
   └─ 8 次 __dp4a 与激活做整数点积
```

数一下 K1 的指令账（`ncu smsp__inst_executed_pipe_lsu.sum` 实测）：

| 版本 | 总指令 | ALU pipe | LSU pipe | FMA pipe |
|---|---|---|---|---|
| 61 `LUT16`（DEC=4） | **2.531e9** | 1.497e9 | **6.372e8** | 3.345e8 |
| 本篇 `PRMT`（DEC=8） | **2.063e9** | 1.230e9 | **1.086e8** | 6.484e8 |

LSU 掉了 **83%**（就是那 32 次/nibble 的 LDS 没了），ALU 也降了 18%，总指令 −18.5%。
ncu 的 SOL 顶在哪一级也随之改变：DEC=4 是 **ALU 75.9%** 最高；DEC=8 变成
**DRAM 83.6% / L2 89.6%** 最高。**这一步不是"把墙搬走"，而是"把墙挪回内存"**——剩下的差距要靠访存优化。

## 二、e2m1 → int8：能不能不进共享内存

FP4 e2m1 的 16 个码点（`s` 符号、`e` 2 位指数、`m` 1 位尾数）对应的整数值 `2 × real` 是：

```
n :  0  1  2  3  4  5  6  7 |  8  9 10 11 12 13 14 15
val:  0  1  2  3  4  6  8 12 |  0 -1 -2 -3 -4 -6 -8 -12
     \____(n & 7) 查幅值表___/   \____同幅值、取负____/
```

关键结构：**幅值只由 `n & 7` 决定，符号只由 `n` 的 bit3 决定**。而 `__byte_perm(a, b, s)` 恰好能
对 4 个字节做**并行的表查询**：selector 的每个 4-bit field 选 `{a,b}` 共 8 个字节里的一个。
也就是说，只要把幅值表 `[0,1,2,3 | 4,6,8,12]` 塞进 `a,b` 两个 32-bit 寄存器，就能一次
`prmt` 出 4 个幅值 byte。

这里有个**必须自己实测的硬件行为**（也是本次踩到的最大坑，见 §五）：

> `prmt` 的 selector field **只用低 3 位**（`.b32` 语义）：值 `0–3` 选 `a` 的字节、`4–7` 选 `b` 的字节，
> **值 `8–15` 会把 bit3 忽略、按 `v & 7` 回绕**——不是"选 0"。所以**原始 nibble 可以直接当 selector**，
> `n≥8` 会自动落到 `n-8` 对应的幅值上。

于是整条解码是（一个 uint32 = 8 个 nibble）：

```
              T0=0x03020100 (幅值 idx 0..3)   T1=0x0C080604 (幅值 idx 4..7)
              T2=0xFDFEFF00 (负值 idx 0..3)   T3=0xF4F8FAFC (负值 idx 4..7)

  mag_lo = prmt(T0,T1, w      )      // nibble 0..3 的幅值（selector 自动 &7）
  sgn_lo = prmt(T2,T3, w      )      // nibble 0..3 的负值
  mag_hi = prmt(T0,T1, w>>16  )      // nibble 4..7
  sgn_hi = prmt(T2,T3, w>>16  )

  // 逐字节符号掩码：nibble i 的符号位是 w 的 bit(4i+3)
  //   (w>>3) & 0x11111111 -> 每个 4-bit field 的 LSB = 0/1
  //   *5                   -> field 值 0 或 5（prmt 里 5 选 b 的 0xFF 字节）
  k_lo = prmt(0, ~0u, ((w      >> 3) & 0x11111111) * 5)
  k_hi = prmt(0, ~0u, ((w>>16  >> 3) & 0x11111111) * 5)

  out_a = mag_lo ^ ((mag_lo ^ sgn_lo) & k_lo)   // nibble 0..3 = k0..k3
  out_b = mag_hi ^ ((mag_hi ^ sgn_hi) & k_hi)   // nibble 4..7 = k4..k7
```

`out_a`/`out_b` 各 4 个 int8、**顺序与原始 k 完全一致**，所以后面 8 次 `dp4a` 与激活 `x8[q]`
的配对**一个字都不用改**（不像"偶/奇拆两半"还要重排激活）。整个 `decode_word` 约 **19 条指令/Word
= 2.4 条/nibble**，且**全是 ALU/PRMT、零 LDS**。

> 补充：为什么不是"一次 prmt 查 16 项表"？因为 PRMT 的 4-bit field 里 bit3 被忽略（见上），
> 没法用它同时编码"幅值 + 符号"两个维度。所以要用**两张 8 项表**（幅值/负值）+ 一个逐字节掩码，
> 共 4 次 `prmt`（2 张表 × 2 个半字）+ 2 次构造掩码的 `prmt`。

代码见 `62-v4-fp4-ffn2/fp4_ffn2.cu` 的 `decode_word_prmt`：

```cuda
__device__ __forceinline__ void decode_word_prmt(uint32_t w, uint32_t& a, uint32_t& b) {
  constexpr uint32_t T0 = 0x03020100u, T1 = 0x0C080604u;   // 幅值 0..3 / 4..7
  constexpr uint32_t T2 = 0xFDFEFF00u, T3 = 0xF4F8FAFCu;   // 负值 0..3 / 4..7
  const uint32_t hi = w >> 16;
  const uint32_t mlo = __byte_perm(T0, T1, w), mhi = __byte_perm(T0, T1, hi);
  const uint32_t slo = __byte_perm(T2, T3, w), shi = __byte_perm(T2, T3, hi);
  const uint32_t klo = __byte_perm(0u, ~0u, ((w     >>  3) & 0x11111111u) * 5u);
  const uint32_t khi = __byte_perm(0u, ~0u, ((hi    >>  3) & 0x11111111u) * 5u);
  a = mlo ^ ((mlo ^ slo) & klo);
  b = mhi ^ ((mhi ^ shi) & khi);
}
```

## 三、实测：B=64 端到端 5.62 → 4.97 ms

把 `DECODE` 做成模板参数（`DEC=0` 用 256 项 `uint16` 字节表、`DEC=4` 用 61 的 16 项 nibble 表、
`DEC=8` 用上面的 PRMT），同一份二进制、同一进程 head-to-head（`B=64, PIPE=1`）:

| 段 | DEC=0 (LUT256) | DEC=4 (LUT16, 61) | **DEC=8 (PRMT)** |
|---|---|---|---|
| `quant_x` | 0.0030 | 0.0027 | 0.0029 ms |
| **K1 gemv**（8.98 GB） | 3.664 ms / 73.1% | 3.641 ms / 73.6% | **3.213 ms / 83.4%** |
| `swiglu+quant` | 0.0055 | 0.0053 | 0.0055 ms |
| **K2 gemv**（4.49 GB） | 1.910 ms / 70.2% | 1.941 ms / 69.0% | **1.727 ms / 77.6%** |
| `unpermute` | 0.0091 | 0.0089 | 0.0090 ms |
| **kernel sum** | 5.592 ms | 5.600 ms | **4.957 ms（81.1% roof）** |
| **端到端** | 5.596 ms | 5.620 ms | **4.968 ms** |

读：`roof` 是同一个 `read_bytes` kernel 顺序读 8.46 GB `w1+w3` 的 3207 GB/s（95.7% HBM），
权重量按 384 活跃专家 ×（packed fp4 + E8M0 scale）= **13.48 GB**、roofline **4.02 ms**。

几个观察：

- **DEC=0（256 项字节表）≈ DEC=4**：字节表把 LDS 减半（1 次/byte），但随机索引让
  `uint16[256]` 读撞 bank，两个效应基本抵消——这也是 60 篇的结论。
- **DEC=8 把 K1 拉过 80%**：因为 K1 原本是 **ALU 发射受限**（L1/TEX 只 71.8%），
  去掉 LDS 后 SM 能连续发 `dp4a`，DRAM 才成为瓶颈。
- **K2 也顺带变快**（61.6% → 77.6%）：它原来就是访存延迟受限，少掉一部分 LDS 依赖后
  `long_scoreboard` 的占比下降、访存流更连续。

跨 batch（同进程，`MTMAX` 随 `B/64` 分派）：

| B | DEC=4 端到端 | DEC=8 端到端 | 提速 | DEC=8 的 K1 / K2 |
|---|---|---|---|---|
| 64 | 5.620 ms (71.8%) | **4.968 ms (81.1%)** | 1.13× | 3.213 (83.4%) / 1.727 (77.6%) |
| 128 | 6.499 ms (62.1%) | **5.644 ms (71.2%)** | 1.15× | 3.380 (79.3%) / 2.240 (59.8%) |
| 256 | 7.644 ms (52.8%) | **6.893 ms (58.4%)** | 1.11× | 4.225 (63.4%) / 2.594 (51.7%) |

B 越大有效带宽越低：`B=256` 时每专家 `m=4` 个 token，激活/`dp4a` 的工作量随 `m` 线性涨、
权重字节不变，"有效权重带宽"这个口径自然被摊薄（59 篇已记录同一现象）。

## 四、几何与 K2 并行粒度的尝试（大多是负结果）

**解码之外的几何**：在 B=64（`MTMAX=1`）把 `RWW × DEPTH × PIPE × DECODE` 扫了一遍
（`sweep_B64.out.txt`，GB/s 按 384 专家权重口径）：

| 配置 | K1 | K2 |
|---|---|---|
| DEC=8 RWW4 DEPTH2 **PIPE1** | **3.211 ms / 83.5%** | 1.801 ms / 74.4% |
| DEC=8 RWW2 DEPTH2 PIPE1 | 3.262 / 82.1% | 1.919 / 69.8% |
| DEC=8 RWW8 DEPTH2 PIPE1 | 3.530 / 75.9% | **1.719 / 77.9%** |
| DEC=8 RWW4 DEPTH3 PIPE1 | 3.322 / 80.7% | 1.801 / 74.4% |
| DEC=8 RWW4 DEPTH2 **PIPE0**（寄存器环） | 4.395 / 61.0% | 2.222 / 60.3% |
| DEC=8 RWW8 DEPTH3/4 PIPE0 | 4.538 / 59.1% | 2.374 / 56.4% |
| DEC=4 RWW4 DEPTH2 PIPE1 | 3.641 / 73.6% | 1.936 / 69.2% |

- **K1 甜点还是 `RWW4 s2`、K2 还是 `RWW8 s2`**，与 61 一致；`PIPE=1`（`cp.async`）稳定赢
  `PIPE=0`（寄存器环）；`DEPTH` 仍是 2，加深只会挤 smem/寄存器。
- **K2 的"并行粒度"没有免费午餐**：`RWW=2/4/8` + `DEPTH=2/3/4` + `PIPE=0/1` 一圈扫下来，
  最好仍是 `RWW8 s2 PIPE1`。原因在 ncu（§五）：K2 的 `NITER = K/1024 = 3`（`K=I=3072`）
  太短，`cp.async` 的 prologue/drain 占比高；而它已经是 **34% occupancy、`long_scoreboard` 40.5%**
  的访存延迟受限，单靠调 tile 补不上。想再进一步得改结构（TMA 搬整 tile，或把 K 摊到更多 CTA）。

## 五、ncu：墙从 ALU 挪回内存

**K1（DEC=8, B=64, RWW4 s2）**：

```
DRAM Throughput        83.61%     L2 Cache Throughput   89.58%
L1/TEX Cache Throughput 21.13%    Compute (SM)          72.87%   (ALU 最高 72.9%)
Duration               3.22 ms    Achieved Occupancy    46.21%   regs 56
Warp Cycles/Issue      12.07      stall long_scoreboard 31.6% (等 L1TEX)
```

对比 DEC=4 的 **ALU 75.9% / L1/TEX 71.8% / DRAM 73.6%**：解码的 LDS 没了，L1/TEX 从 71.8 掉到 21.1，
DRAM 升到 83.6——**瓶颈回到内存**。剩下 95.7% → 83.6% 的差距来自 L2（89.6%，权重 + scale 都过 L2）
和访问粒度；纯顺序读的 roof 是 95.7%。

**K2（DEC=8, B=64, RWW8 s2）**：

```
DRAM Throughput        78.23%     L2 Cache Throughput   84.68%
L1/TEX Cache Throughput 18.37%    Compute (SM)          68.16%
Duration               1.72 ms    Achieved Occupancy    34.27%   regs 62
Warp Cycles/Issue       9.56      stall long_scoreboard 40.5%
```

K2 比 K1 低 ~5 pp：occupancy 只有 34%（smem 被 64 KB 的 `cp.async` stage 环限制在 3 CTA/SM）、
且 `NITER=3` 的短循环让访存延迟藏不住。它现在是**彻底的访存延迟受限**。

> 相对 61：61 的 K1 是 `Compute 75.9% ≈ DRAM 71.4%`（两堵墙并立），K2 是 `long_scoreboard ~35%`。
> 本篇把 K1 的 ALU 墙拆了，两段的 `DRAM` 都往上走了一截，端到端 67.6% → **81.1% roofline**。

## 六、踩坑：`__byte_perm` 的 selector 语义

这次最值的坑。第一版按"标准"理解写 `idx = n4 & 0x07070707`（把 4 个 nibble 压进 4 个字节的低 4 位），
结果解码全错。抽最小复现（`dec_test.cu` / `probe.cu`）穷举后才发现：

```
sel=0x00000001 -> 0x01010102   // field0=1: output byte0 = a.b1 = 2 ✓
sel=0x00000200 -> 0x00020000   // 0x02 在 byte1，却是 output byte2 = 2 ✗（按字节理解）
```

真相：**selector 是按 4-bit field 从 LSB 排的**，`0x00000200` 的 `0x02` 落在 bit9–11，属于 **field2**，
所以选的是 output byte2。也就是说 selector 的 4 个 field 分别是 `bits[3:0] bits[7:4] bits[11:8] bits[15:12]`。
当初把 nibble 放在"字节"里（每个字节低 4 位）其实是把索引放到了 field 0/2/4/6，位置全错。

再穷举 selector field 值：

```
field=0..3  -> a 的 byte0..3
field=4..7  -> b 的 byte0..3
field=8..15 -> 与 field=(v & 7) 相同       ← bit3 被忽略、回绕！
```

正是这条"bit3 被忽略"让 §二 的写法成立：**原始 nibble（0–15）直接当 selector，`n & 7` 自动生效**，
省掉了压缩 nibble 到 field 的 6–10 条指令。

其它小坑：

- **偶/奇拆分的交错 selector** 第一版写成 `0x6240`（顺序 0,4,2,6），与权重偶 nibble 的
  `[k0,k2,k4,k6]` 顺序对不上 → 后来改用"自然顺序 + 符号掩码"，根本不需要交错，两者都省了。
- 验证 PRMT 语义**必须穷举**（16 个 nibble + 16 个 selector field），抽点会漏掉回绕行为。

## 七、小结

- **同一段解码，LDS → PRMT 位拼装**：K1 `3.64→3.21 ms`、K2 `1.94→1.73 ms`、端到端
  `5.62→4.97 ms`（**1.13×**，81.1% 权重 roofline）；LSU 指令 **−83%**、总指令 **−18.5%**。
- **判据**：`PRMT 换 LUT` 的胜负取决于**墙在哪一级**。58/59/60 里 LUT 的墙是并行度（58）或
  L1/TEX 吞吐（59/60），换 PRMT 是负优化；K1 的墙是 **ALU 发射**，换 PRMT 才是正解。
  动手前先看 `pipe_alu` 与 `pipe_lsu` 谁高。
- **PRMT 的 selector 只用低 3 位、值 8–15 回绕**——这个硬件行为是"nibble 直接当索引"的基础，
  但也让"按字节理解 selector"的写法静默错位，必须穷举验证。
- **K1 现在 DRAM 83.6% / L2 89.6% 受限**；K2 78.2%、`long_scoreboard` 40.5%、occ 34%，
  短 `NITER=3` 的访存延迟是下一堵墙。

**下一篇候选**：① K2 用 TMA 搬整 row-tile（免掉每 lane 8 条 `cp.async` 与短流水）；
② K1 的 L2 89.6% —— 权重与 scale 都过 L2，试 L2 promotion / 更大加载粒度 / cluster multicast；
③ 把这条 PRMT 解码推广回 58/59/60 的单矩阵/grouped GEMV（那里 `B=1` 时 `dp4a` 少、可能仍输，
但 `M≥4` 让解码复用后值得重测）。

---

**配套代码**：`code/kernel-opt/62-v4-fp4-ffn2/`

- `fp4_ffn2.cu` —— 主实验（`DECODE` = 0/4/8，五段流水，内建 CPU 参考对拍与分段计时）
- `dec_test.cu` / `probe.cu` —— PRMT selector 语义最小复现（穷举）
- `ffn_B64_dec{0,4,8}.out.txt`、`compare_B128_256.out.txt`、`sweep_B64.out.txt`
- `ncu_k1_dec8.out.txt`、`ncu_k2_dec8.out.txt`、`inst_cmp.out.txt`

运行：

```bash
cd code/kernel-opt
scripts/run.sh 62-v4-fp4-ffn2/fp4_ffn2.cu 64 1 8 none    # B=64, PIPE=1, DEC=8
scripts/run.sh 62-v4-fp4-ffn2/fp4_ffn2.cu 64 1 8 sweep   # 几何/解码 sweep
```
