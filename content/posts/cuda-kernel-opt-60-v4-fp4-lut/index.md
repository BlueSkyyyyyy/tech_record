---
title: "CUDA 算子调优（六十）：FP4 grouped decode 的 LUT 真的是墙吗 —— 244.6M 次 bank conflict 被消掉，只快了 2.9%"
date: 2026-09-22T09:58:00+08:00
draft: false
weight: 60
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "DeepSeek-V4", "GEMV", "grouped", "bank-conflict", "LUT", "occupancy", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 59 篇]({{< relref "cuda-kernel-opt-59-v4-fp4-grouped" >}}) 把 DeepSeek-V4-Pro 一步 decode 的
`B×6` 个 `(token, expert)` 对按 expert 折叠进一个 grouped GEMV：权重每专家只读一次，`B=256`
（每专家 4 个 token）拿到 **2.63 ms / 2008 GB/s / 59.9% HBM**，比逐对 launch 快 **5.9×**。
59 结尾把话说得很满：

> ncu 里墙从 **DRAM（60.1%）** 挪到了 **L1/TEX（83.9%）**，残余 48% 的多余 wavefront 几乎全来自
> **256 项 `uint16` LUT 的随机读**。下一堵墙是**免冲突的 LUT**。

本篇就是去兑现这句话。把 LUT 的每一条共享内存访问拆开数了一遍之后，结论反而很「打脸」：

> **那 2.44 亿次 bank conflict 是真的，但它们不是墙。**
> 把随机查表换成**结构上无冲突**的 16 项 nibble 表（`LUTMODE 4`）后，
> bank conflict **244.6M → 0.12M**、`short_scoreboard` 停顿 **3.01 → 0.64（4.7×）**、
> `L1/TEX` **83.4% → 68.8%** —— 可是墙钟只从 **2.62 ms 掉到 2.55 ms（+2.9%）**。
> 因为 grouped kernel 的墙其实是 **occupancy（24.4%，寄存器锁死 2 CTA/SM）** 带来的
> **访存延迟**，L1/TEX 的高利用率只是伴随现象。真正在模型层有杠杆的是另一件事：
> **真实路由下活跃专家变少** —— 75% hot 路由 **1.24×**、幂律路由只读 44% 权重时 **2.09×**。

---

## 一、先把「L1/TEX 84%」拆成可数的账

59 的 grouped kernel，`B=256`、balanced 路由（每专家恰好 4 个 token），
`w1 [N=3072, K=7168]` 的 FP4 权重（e2m1 + E8M0 block-32，56 篇实测）+ int8 激活 + `dp4a`。
用 ncu 把共享内存那部分单独拉出来（`60-v4-fp4-grouped2/ncu_grouped2_M0.out.txt`）：

| ncu 指标 | 值 | 含义 |
|---|---|---|
| Duration | 2.62 ms | |
| DRAM Throughput | **60.35%** | HBM 没喂满 |
| L1/TEX Throughput | **83.38%** | 最高的一级 |
| Achieved Occupancy | **24.44%** | 16 warp/SM（理论 25%） |
| `shared_ld` 指令（warp 级） | 181,665,792 | |
| shared load wavefronts | 475,815,269 | **2.62 wavefront/指令** |
| shared load bank conflict | **244,604,261** | **1.35 conflict/指令** |
| stall `short_scoreboard` | **3.01** | MIO/共享内存依赖，占停顿最大头 |
| stall `long_scoreboard` | 2.35 | 等 global |

两句话就能读懂：

1. **bank conflict 让每次共享读平均多花 1.35 个 wavefront**（理想是 1.0）——约占全部 shared
   wavefront 的 **52%**。`short_scoreboard = 3.01` 直接对应它。
2. **LUT 读的条数 = 权重字节数**。`expand_lut` 每读 1 个 weight byte（一个 `uint16` 表项给出两个
   nibble 的 int8），所以权重 5.28 GB 就是 **~5.3×10⁹ 次 lane 级查表**。每次查表撞一次 bank，
   就是那 2.44 亿次 conflict 的来源。这是 L1/TEX 被顶到 84% 的绝对主因。

```
              每次权重 byte 的解码路径
  Wp (fp4, 2 nibble/byte)
        │
        ▼  expand_lut(w): 每 byte 一次 LDS.U16
   ┌─────────────────────────────┐
   │ lut16[256]  （512 B = 128 word = 4 word/bank）
   │   lane0 ─► idx=0x?? ─► bank b0
   │   lane1 ─► idx=0x?? ─► bank b1      ← 32 个 lane 随机索引
   │   ...                                256 项散到 32 bank
   │   lane31─► idx=0x?? ─► bank b31       ⇒ 平均 2.62 wavefront/读
   └─────────────────────────────┘
```

### 为什么「分半表 / padding」没用，而「16 项表」有用

一个关键事实：**bank conflict 只有在索引空间 ≤ 32 且每个表项独占一个 bank 时才能被彻底消除**。
理由很简单——32 个 lane 随机查一个 `T` 项的表，只要 `T ≤ 32`，把第 `v` 项放在 bank `v`，
那么不同索引一定落在不同 bank（相同索引自动广播），**一次 wavefront 搞定**。

256 项的 byte 表做不到这点（项 `v` 和 `v+32` 天然同 bank），于是：

- 「拆成两个 128 项半表」「给表 padding」「错开副本基址」——都只是改变碰撞的分布
  （把 32 个 lane 分成两拨各查 128 项），**期望冲突只降一点**；
- 真正无冲突的办法是**把 byte 拆成两个 nibble**：`v & 0xF` 和 `(v>>4) & 0xF` 都是 4 bit，
  索引空间 16 ≤ 32。代价是**每 byte 两次查表**（LDS 指令数翻倍）。

---

## 二、LUT 布局消融：五种表，同 kernel 只换 `LUTMODE`

`code/kernel-opt/60-v4-fp4-grouped2/fp4_grouped2.cu` 里把查表封装成 `expand_v<LUTMODE>`，
其余（两平面激活、`dp4a`、ring 预取、warp 归约）逐字节相同：

| mode | 布局 | 项数×宽度 | smem | 说明 |
|---|---|---|---|---|
| **M0** | `uint16 lut[256]` | 256×2B | 512 B | 59 基线（128 word = 4 word/bank） |
| **M1** | `uint32 lut[256]` | 256×4B | 1 KB | 每项独占一个 word（8 word/bank） |
| **M2** | `uint16` ×2 副本 | 2×256×2B | 1.3 KB | lane 奇偶选副本，副本基址错开 1 个 bank |
| **M3** | `uint16` ×4 副本 | 4×256×2B | 1.7 KB | lane&3 选副本，错开 0/1/2/3 bank |
| **M4** | **16 项 nibble 表** | 16×2B | 32 B | 索引 ≤32 独占 bank ⇒ **结构无冲突**，每 byte 两次 LDS |

`B=256`（每专家 4 token）实测（`grouped2_B256.out.txt`）：

| 配置 | 耗时 | 有效权重带宽 | % HBM | 相对 M0 |
|---|---|---|---|---|
| M0 `uint16[256]`（59） | 2.6177 ms | 2018.9 GB/s | 60.2% | 1.000× |
| M1 `uint32[256]` | 2.8278 ms | 1868.9 GB/s | 55.7% | 0.926× |
| M2 ×2 副本 | 2.7516 ms | 1920.6 GB/s | 57.3% | 0.951× |
| M3 ×4 副本 | 2.8746 ms | 1838.4 GB/s | 54.8% | 0.911× |
| **M4 nibble16** | **2.5467 ms** | **2075.2 GB/s** | **61.9%** | **1.028×** |

要看懂这张表：**M1/M2/M3 全部更慢**。它们确实换了分布、也占了更多 smem，但没换来无冲突
（M1 的 8 word/bank 与 M0 的 4 word/bank 在「随机索引命中某 bank 的概率恒为 1/32」这点上等价），
反而把表变大、L1 命中变差。**只有 M4 把 conflict 真正归零**，拿到唯一的正向收益。

### M4 的 ncu：机制上大胜，墙钟上小胜

同 kernel 只切 `LUTMODE`（`ncu_grouped2_M0/M4.out.txt`）：

| 指标 | M0 | M4 | 变化 |
|---|---|---|---|
| Duration | 2.62 ms | **2.55 ms** | −2.9% |
| bank conflict | 244,604,261 | **122,786** | **÷1993** |
| shared wavefronts | 475.8M | 363.5M | −24% |
| shared_ld 指令 | 181.7M | **313.8M** | **+73%** |
| L1/TEX Throughput | 83.38% | **68.80%** | −14.6 pp |
| **stall `short_scoreboard`** | **3.01** | **0.64** | **÷4.7** |
| stall `long_scoreboard` | 2.35 | 1.40 | −40% |
| DRAM Throughput | 60.35% | 62.02% | +1.7 pp |
| Registers / Occupancy | 119 / 24.4% | 124 / 24.1% | 持平 |

```
   wavefront/指令          1 次吞吐              L1/TEX          墙钟
 M0  2.62  ──►  (512B 表, 4 word/bank)  ──►  83.4%  ────►  2.62 ms
 M4  1.00  ──►  (32B 表, 16 项各占 bank) ──►  68.8%  ────►  2.55 ms
      ▲ 冲突没了                              ▲ L1 松了       ▲ 只快 2.9%
      └ 但指令数 181.7M ─► 313.8M（每 byte 两次 LDS）          └ DRAM 只从 60→62%
```

`short_scoreboard` 掉了 4.7×、L1/TEX 掉了 14.6 pp，**时间却几乎没动**，因为瓶颈已经不在
L1/TEX 了：`short_scoreboard` 从 3.0 掉到 0.64 之后，**DRAM 只从 60.4% 升到 62.0%**。
也就是说，**去掉 L1 的阻塞并不能让 HBM 吃饱**——访存延迟（DRAM latency）才是真正的天花板。

> **结论一（本篇最重要）：ncu 报「L1/TEX 是最高的那一级」不等于 L1/TEX 是瓶颈。**
> 59 篇顺着 `L1/TEX 83.9%` + `short_scoreboard` 推断「消 LUT 冲突就能冲 70%」，
> 只对了一半：冲突确实消得掉，但墙钟由 DRAM 延迟决定，L1 只是「离得最近的那个高水位」。

---

## 三、为什么是延迟墙：occupancy 被寄存器锁死

既然瓶颈是延迟，标准解法就是**更多 warp 来藏延迟**。59/M4 的 occupancy 都只有 **24%**
（16 warp/SM），ncu 明说是 **`launch__occupancy_limit_registers = 2`（2 CTA/SM）**，
寄存器 **119（M0）/124（M4）**。`ptxas -v` 显示数据结构的寄存器账：

| 寄存器组 | 大小 | 说明 |
|---|---|---|
| `acc[4][4]` | 16 | 4 行 × 4 token 的 fp32 累加器（grouped 的「复用」就靠它） |
| `ring[4][2]` | 32 | 4 行 × 深度 2 的 `uint4` 权重预取环 |
| `x8[4][8]` | 32 | 4 个 token 的激活片段（每 token 32 个 int8） |
| `w8[8]` | 8 | 当前权重解码结果 |
| 地址/杂项 | ~30 | |

光数据就 ~88 个，离 3 CTA/SM 的门槛 `65536/(3×256) = 85` 只差一步但跨不过去。
实测两条强行提 occupancy 的路都是负结果：

| 配置 | 耗时 | 说明 |
|---|---|---|
| M0 `CTA2`（基线） | 2.6177 ms | 2 CTA/SM |
| M0 `__launch_bounds__(256,3)` | 2.9481 ms | 压到 85 regs 但**溢出** |
| M0 `__launch_bounds__(256,4)` | 5.0901 ms | 溢出更狠 |
| MTMAX=8（`__launch_bounds__(256,2)`） | 3.6009 ms | 128 regs + **884 B spill** |

`MTMAX` 必须匹配真实 `m`（59 已经踩过）：balanced `B=256` 的 `m=4`，用 `MTMAX=4` 是 2.62 ms；
一旦为了兼容 `m=6/8` 的路由把它开到 8，寄存器溢出一上来就掉到 3.60 ms——后面路由消融里所有
模式都统一用 `MTMAX=8`，就是为了让模式之间的对比公平（都吃同一份溢出惩罚）。

### 几何 × 流水深度扫描（都用 M4）

| 配置 | 耗时 | % HBM |
|---|---|---|
| **M4 RWW4 D1** | **2.5426 ms** | **62.0%** |
| M4 RWW4 D2 | 2.5467 ms | 61.9% |
| M4 RWW2 D2 | 2.9463 ms | 53.5% |
| M4 RWW2 D3 | 2.9358 ms | 53.7% |
| M4 RWW1 D2 | 4.1105 ms | 38.4% |
| M4 RWW8 D1 | 3.7131 ms | 42.5% |
| M4 RWW8 D2 | 6.5204 ms | 24.2% |

`RWW=4` 是甜点（59 已测过 `RWW=1/2/8` 都更慢）；`DEPTH=1` 和 `DEPTH=2` 打平
（都 32/16 个 ring 寄存器量级，2 CTA/SM 不变），**流水加深救不了**。`RWW=8` 直接把每 warp
的 ring 和 acc 翻倍，寄存器溢出去本地内存，崩到 6.5 ms。

> **结论二：grouped decode 的「复用」和「占用率」是一对矛盾。** `m` 个 token 的累加器 +
> 预取环是权重复用的前提，但它们把寄存器顶到 119+，锁死 2 CTA/SM。不牺牲复用（`MTMAX`、`RWW`）
> 就没法堆 warp；牺牲复用又退回 59 的 `batched_m1`。

---

## 四、真正在模型层有杠杆的：真实稀疏路由

59 用的是**平衡路由** `(t*6+j) % 384`，让每个专家恰好拿到 `B/64` 个 token——这是理想化的。
真实 MoE 推理一步里，很多专家**一个 token 都没分到**，而 grouped kernel 对 `counts==0` 的组
直接 `return`。所以「权重字节」应该按**活跃专家**算：

```text
grouped 实际权重字节 = active_experts × 13.76 MB   （不是 384 × 13.76 MB）
```

构造三种路由（`build(mode)`）实测（统一 `MTMAX=8, RWW4, M4=0`，读到的字节按活跃专家算）：

| B | 路由 | 活跃专家 | 0-token 专家 | 权重字节 | 耗时 | 有效带宽 | 相对 balanced |
|---|---|---|---|---|---|---|---|
| 64 | balanced | 384/384 | 0 | 5.28 GB | 2.949 ms | 1791.8 GB/s | 1.000× |
| 64 | hot-75% | 288/384 | 96 | 3.96 GB | 2.270 ms | 1746.4 GB/s | **1.300×** |
| 64 | zipf | **168/384** | 216 | **2.31 GB (44%)** | **1.414 ms** | 1635.1 GB/s | **2.086×** |
| 128 | balanced | 384/384 | 0 | 5.28 GB | 3.157 ms | 1673.9 GB/s | 1.000× |
| 128 | hot-75% | 288/384 | 96 | 3.96 GB | 2.500 ms | 1585.3 GB/s | 1.263× |
| 128 | zipf | 256/384 | 128 | 3.52 GB (67%) | 2.236 ms | 1575.6 GB/s | 1.412× |
| 256 | balanced | 384/384 | 0 | 5.28 GB | 3.601 ms | 1467.7 GB/s | 1.000× |
| 256 | hot-75% | 288/384 | 96 | 3.96 GB | 2.903 ms | 1365.5 GB/s | 1.241× |
| 256 | zipf | 350/384 | 34 | 4.82 GB (91%) | 3.298 ms | 1460.7 GB/s | 1.092× |

两个可用的结论：

1. **时间基本跟着「活跃权重字节」走**：75% hot → 读 75% 字节 → 快 1.24–1.30×；
   `B=64` 的幂律只读 44% → 快 **2.09×**。这就是 grouped（按专家折叠）相对 batched（按对）
   的第二个优势：**没被点到的专家是真正零成本**（59 的 batched_m1 对每个 pair 都要读一遍权重，
   稀疏与否都得读）。
2. **有效带宽随稀疏度下降**（balanced 60% → zipf 44%）：活跃专家变少，网格变小，H100 的
   132 个 SM 更容易空转；`B=256` 幂律只砍掉 34 个专家（91% 字节），收益立刻缩到 1.09×。
   稀疏的收益是**字节红利**，不是带宽红利。

> **结论三：MoE decode 的 grouped 折叠天然吃稀疏路由的红利。** 平衡路由是 grouped 的最坏情况
> （每个专家都被点到）；真实路由越偏，权重读得越少。但收益 ∝ 省下的字节，且要留意小网格下的
> 带宽下滑。

---

## 五、把 59/60 的判决合起来：FP4 decode 的完整账

| 版本 | 关键手段 | B=64 | B=128 | B=256 |
|---|---|---|---|---|
| 58 单专家 `pipe` | LUT + `dp4a`，一 (token,expert) 一 launch | 逐对 | 逐对 | 逐对 |
| 59 `batched_m1` | 一个 kernel、grid.y=pairs、M=1 | 74.5% | — | 74.4% |
| 59 `grouped` M0 | 按专家折叠（每专家权重读 1 次）+ 两平面激活 | 74.5% | 68.5% | 60.2% |
| **60 `grouped` M4** | **+ 无冲突 nibble LUT** | 73.6% | **70.3%** | **61.9%** |
| 60 路由（hot-75%） | + 只读活跃专家 | 1.30× | 1.26× | 1.24× |

M4 的收益**随 `m` 增大而出现**：`m=1`（B=64）时是 **−1.3%**（多一倍的 LDS 指令换不来回
复用，冲突本来就没那么致命）；`m=4`（B=256）才是 **+2.9%**（一次解码被 4 个 token 摊薄，
省下的冲突 wavefront 更值钱）。这正好复现 58 篇「LUT vs PRMT 胜负取决于 M」的规律。

纯读 roof（一次读完全部 384 专家的 fp4）：

```text
[roof] read all 384 experts fp4 (4.228 GB): 1.320 ms  3203 GB/s (95.6% HBM)
```

对应到 kernel 的最少字节 5.28 GB（fp4 + scale），**HBM 下限 1.58 ms**；最好的 grouped 是
2.54 ms，**距内存 roofline 1.6×**。差距全部来自那 24% 的 occupancy——这是「复用 vs 并行度」
的结构性代价，不是某条指令能补的。

---

## 六、收尾：这次「送进手术室」的结论

1. **244.6M 次 bank conflict 是真的，但不是墙。** M4 nibble 表把它归零（÷1993）、
   `short_scoreboard` ÷4.7、L1/TEX −14.6 pp，墙钟只 **+2.9%**。判据：**看 DRAM% 有没有跟着涨**，
   没涨就说明瓶颈在延迟而不是那一级吞吐。
2. **索引空间 ≤ 32 且表项独占 bank 才能结构性消冲突**；256 项 byte 表的「分半 / padding /
   多副本」都只是换分布。代价是每 byte 两次 LDS。
3. **grouped 的复用与 occupancy 互斥**：`acc[m]` + 预取环 + 激活片段把寄存器顶到 119+，
   锁死 2 CTA/SM；强行提 occupancy 一律溢出（CTA4 → 5.09 ms）。
4. **真实稀疏路由是模型层的真杠杆**：活跃专家少 → 权重字节少 → 时间少（75% hot 1.24×、
   44% active 2.09×），但有效带宽随网格变小而下滑。

下一篇的候选（见 ROADMAP「下一步」）：

- **把 grouped GEMV 接进 decode MoE FFN 端到端**（`w1/w3 → SwiGLU → w2`），量一步 decode 的
  真实延迟与 roofline；
- 用 **`cp.async` 把权重搬进 smem** 做深流水，把 global 延迟从寄存器预取环里解耦
  （本轮的延迟墙的正解方向）；
- 把 [第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}}) 的「ue8m0 折进操作数」思路
  用到 FP4：E8M0 是 2 的幂，`scale` 可折进 nibble→int8 的 LUT，省掉每次的 `sw` 读取。

代码与原始输出：`code/kernel-opt/60-v4-fp4-grouped2/`
（`fp4_grouped2.cu`、`grouped2_B{64,128,256}.out.txt`、
`ncu_grouped2_M0.out.txt`、`ncu_grouped2_M4.out.txt`、`ncu_grouped2_M4_bytes.out.txt`）。
