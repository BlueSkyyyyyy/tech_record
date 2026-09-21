---
title: "CUDA 算子调优（四十五）：W4A8 + dp4a——把 M=1 decode 的反量化 ALU 税从 60% 打到 30%，贴上 HBM 带宽"
date: 2026-09-22T03:40:00+08:00
draft: false
weight: 45
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "推理", "decode", "量化", "W4A8", "int4", "int8", "dp4a", "AWQ", "GPTQ", "GEMV", "带宽", "roofline", "Qwen3", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「模型场景算子」量化推理方向的第五篇。直接承接
[第 44 篇 W4A16 的 M=1 GEMV]({{< relref "cuda-kernel-opt-44-w4a16-decode-gemv" >}})。

44 篇把 M=1 的 W4A16 decode 从「硬上 `m64` 张量核」（0.0798 ms）拉回到「每 warp 一行、
warp 内沿 K 并行」的带宽最优 GEMV，做到 **0.0284 ms / 1566.6 GB/s / 49.7% HBM**，
比 43 篇快 **2.81×**。但那一篇的 ncu 留了一个明确的尾巴：

| 44 篇 `gemv_r2`（最佳） | 值 |
|---|---|
| **ALU pipe** | **64.9%** ← 最高占用 |
| FMA pipe | 27.1% |
| Compute (SM) | 57.8% |
| DRAM Throughput | 47.7% |

> 反量化把每个 int4 变成 float 需要「移位 + 掩码 + `I2F`」三串 ALU 指令，**ALU pipe 64.9%
> 是整个 kernel 的最高占用**……能继续做的方向是走 **W4A8 + `dp4a` 整数点积**。

这一篇就来补这 **1.52×** 的 gap（纯读上界 0.0187 ms / 75.4% HBM）。做法是把激活也量化成
int8（per-128-group），权重仍是 int4，用 `__dp4a`（一条指令做 4 个 int8 乘加）替代浮点点积。

结论：**M=1 的 GEMV 从 0.0284 ms → 0.0185 ms（1.53×），HBM 从 49.7% → 76.3%**，
ncu 的瓶颈从 ALU 60% 翻转为 DRAM 67%——**它终于变成了一个纯带宽 kernel**。

shape 取自 `/ssd/models/qwen3-8B/config.json`（`hidden_size=5120`,
`intermediate_size=17408`, `group_size=128`，对称 int4），测 `M=1, N=17408, K=5120`。

---

## 一、先看 gap：钱花在哪条指令上

44 篇的内层对每 32 个 int4 权重做（`w4a16_gemv.cu`）：

```
每 32 个权重：16 次「取 byte → 抠两个 nibble」 + 32 次 (I2F + FSUB) + 32 次 FFMA(×RWW)
约 5 条 ALU / 元素
```

而真正的「计算」只有 32 次 MAC。也就是说，**这个 kernel 的 80% 指令都在把 int4 变成 float**。
H100 的 `dp4a`（`IDP4A`，SASS 里叫 `IDP.4A`）一条指令能吃 4 对 int8 做乘加——理论上能把
「4 次 MAC」压成 1 条指令。剩下的问题只有一个：**怎么把 4-bit 的权重变成 8-bit 的权重**。

W4A8 的账：

$$
y = \sum_k x_k w_k,\quad
w_k = s^w_{g}\cdot q^w_k,\quad
x_k = s^x_{g}\cdot q^x_k,\quad
y = \sum_g s^w_g s^x_g \Big(\sum_{k\in g} q^w_k q^x_k\Big)
$$

- `q^w` 是 4-bit 对称量化（`-8..7`，存成带偏置的 `0..15`）；
- `q^x` 是 **int8 激活**（`-127..127`），per-128-group 动态量化；
- 内层就是纯整数点积 `Σ q^w q^x`，用 `dp4a` 做，最后乘一次 `s^w_g s^x_g`。

代价：引入激活的 int8 量化误差（实测额外 ~0.4% 相对误差），换来的是 ALU 指令数砍一个数量级。

---

## 二、第一版：int4 → int8 展开撞上 `__vsubss4` 的坑

权重在显存里是 packed int4，一字节两个 nibble（`q+8`，`0..15`）。`dp4a` 要的是每个 byte
一个**有符号** int8。把 nibble 铺成 byte 用 `PRMT`（`__byte_perm`）一行就够：

```
lo = w & 0x0F0F0F0F     // byte: n0 n2 n4 n6
hi = (w>>4) & 0x0F0F0F0F // byte: n1 n3 n5 n7
a  = __byte_perm(lo, hi, 0x5140)   // [n0 n1 n2 n3]
b  = __byte_perm(lo, hi, 0x7362)   // [n4 n5 n6 n7]
```

但每个 byte 还得做符号调整 `q = u - 8`（`u∈0..15`）。**注意不能写 `(a ^ 0x08080808)
- 0x08080808`**：这是个 32-bit 减法，当某个 byte 的 `u<8` 时会**向前一个 byte 借位**，把
高位字节全污染——我第一版就踩了这个坑（`rel=2.16`，全错）。逐 byte 有符号减 8 的「正统」
做法是 `__vsubss4`（SIMD 视频指令）。

改用 `__vsubss4` 之后数值对了，但性能只到 **0.0226 ms / 62.4% HBM**。ncu 显示它仍然是
**ALU 最高占用（60.1%）**，而且 `smsp__inst_executed.sum = 11.87 M`。看 SASS 才明白：

```
494 LOP3.LUT   240 PRMT   22 IMAD.WIDE ...
```

**`__vsubss4` 在 sm_90 上被 ptxas 展开成一大堆 `LOP3`/`PRMT`**（Volta 之后 SIMD 视频指令
都是软件模拟）。符号调整这一下，反而成了新的 ALU 大头。

---

## 三、第二版：干脆不调整符号，把偏置解析地扣掉

关键观察：`dp4a` 的两个操作数只要**同号**即可。权重 nibble 值是 `u = q+8 ∈ [0,15]`，它
本身就能当**正整数 int8** 用（`15 < 127`，不溢出）。于是令 `dot = Σ u·q^x`，则

$$
\sum (q^w)\,(q^x) = \sum (u-8)\,q^x = \underbrace{\sum u\,q^x}_{\text{dp4a 直接算}} \;-\; 8\sum q^x
$$

`Σ q^x` 只依赖激活、**与输出行无关**，每个 chunk（32 个 x）用 `__dp4a(ones, x, ·)` 现算一次
就够（`ones = 0x01010101`）。于是 `expand8u` 只剩 5 条廉价指令、没有符号处理：

```cuda
__device__ __forceinline__ void expand8u(uint32_t v, uint32_t& a, uint32_t& b) {
  const uint32_t lo_nib = v & 0x0F0F0F0Fu;
  const uint32_t hi_nib = (v >> 4) & 0x0F0F0F0Fu;
  a = __byte_perm(lo_nib, hi_nib, 0x5140);   // [u0 u1 u2 u3]
  b = __byte_perm(lo_nib, hi_nib, 0x7362);   // [u4 u5 u6 u7]
}
```

内层（`45-w4a8-gemv/w4a8_gemv.cu` 的 `gemv_a8h_kernel<RWW,MT,KK>`）：

```
for i in 0..NCHUNK-1:                    // 每 lane 一个 32-k chunk
  c  = lane + 32*i ; k0 = c*32 ; g = k0/128
  x8 = xs[m][k0 .. k0+32]                // 2×uint4，一次读好给 RWW 行 / MT token 复用
  sx[m] = 8 * dp4a(ones, x8[m])          // 该 chunk 的 8·Σq^x，与行无关
  for r in 0..RWW-1:
    wv = W[row_r, c]                     // __ldcs 16B = 32 个 int4
    expand8u(wv.x..wv.w) → w8[8]         // 4 次 expand8u，纯 PRMT/AND
    dot = Σ_q dp4a(w8[q], x8[q])          // 8 次 dp4a
    acc[r] += s * (dot - sx[m])          // s = s^w_g · s^x_g
warp_sum(acc[r]) ; lane0 写 C
```

`Σq^x` 只算一次（而不是沿 RWW 行各算一次），正好把「不做符号处理省下的指令」补齐还有余。
实测指令数从 **11.87 M → 7.17 M（−40%）**，时间 **0.0226 → 0.0185 ms**。

---

## 四、实测结果

环境：H100 80GB HBM3，峰值 3352 GB/s，`M=1, N=17408, K=5120`，`group=128`。
运行 `ARCH="" scripts/run.sh 45-w4a8-gemv/w4a8_gemv.cu 1 17408 5120 all`。
为避免跨进程干扰，**W4A16 基线（44 篇的 kernel）也编进同一个可执行文件、同进程同参对拍**。

```
W4A8 GEMV: M=1 N=17408 K=5120 group=128
FLOPs = 0.178 GFLOP ; W_int4 = 44.6 MB ; scales = 2.8 MB ; W+scale = 47.3 MB ; x_int8 = 0.01 MB

roof_g528     0.0190 ms   W-bw 2347.4 GB/s   W+s 2494.1 (74.4% HBM)   ← 纯读上界
quant_x       0.00253 ms  (W4A8 的额外前置 kernel：激活 int8 量化)

w4a16_r2      0.0283 ms   W-bw 1572.5 GB/s   W+s 1670.8 (49.8% HBM)   ← 44 篇口径基线
w4a8_r2       0.0226 ms   W-bw 1969.7 GB/s   W+s 2092.8 (62.4% HBM)   ← v1 (__vsubss4)
w4a8h_r1      0.0193 ms   W-bw 2306.8 GB/s   W+s 2450.9 (73.1% HBM)
w4a8h_r2      0.0200 ms   W-bw 2225.7 GB/s   W+s 2364.8 (70.5% HBM)
w4a8h_r4      0.0185 ms   W-bw 2408.4 GB/s   W+s 2558.9 (76.3% HBM)   ← 最佳
w4a8h_r8      0.0195 ms   W-bw 2280.2 GB/s   W+s 2422.7 (72.3% HBM)
```

| 版本 | 时间 | 有效权重带宽 | HBM 占比 | vs W4A16 |
|---|---|---|---|---|
| 44 篇 W4A16 GEMV（`RWW=2`） | 0.0283 ms | 1572.5 GB/s | 49.8% | 1× |
| W4A8 v1（`__vsubss4`，`RWW=2`） | 0.0226 ms | 1969.7 GB/s | 62.4% | **1.25×** |
| **W4A8 v2（unsigned + `sumx` 修正，`RWW=4`）** | **0.0185 ms** | **2408.4 GB/s** | **76.3%** | **1.53×** |
| 纯读上界（只流 W+scale，无计算） | 0.0190 ms | 2347.4 GB/s | 74.4% | 1.49× |
| naive（1 thread/行，标量逐 k） | 0.4583 ms | 97.2 GB/s | 3.1% | 24.8× |

**最佳 0.0185 ms，比 44 篇的 49.7% / 0.0284 ms 快 1.53×，HBM 利用率 76.3%——已经越过
同进程测出的「纯读上界」（74.4%）。** 这意味着 W4A8 的 GEMV 本体现在是**纯带宽受限**，
ALU 不再是瓶颈（见下一节）。

正确性：内置 CPU 参考（对量化后的 `W_deq` 与 **bf16 激活** 做 fp64 点积）抽样 32 点。
W4A16 相对误差 **6e-8**（同一套量化权重，仅 fp32 累加顺序差）；W4A8 相对误差
**4.0e-3**——这 0.4% 就是激活 int8 量化（per-128 动态）引入的额外误差，是 W4A8 相对
W4A16 的固有代价。

---

## 五、ncu：瓶颈从 ALU 翻转成 DRAM

两版同 shape 采 ncu（`--launch-skip 2 --launch-count 1`，抓第 3 个 launch = `RWW` 配置）：

| 指标 | v1（`gemv_a8b`, `__vsubss4`） | v2（`gemv_a8h`, unsigned） |
|---|---|---|
| Duration | 26.9 µs | 22.8 µs |
| **inst_executed** | **11,870,080** | **7,169,920（−40%）** |
| Compute (SM) | **60.1%** | 29.8% |
| **DRAM Throughput** | 56.9% | **66.7%** |
| L1/TEX | 42.8% | 50.4% |
| L2 | 52.5% | 60.3% |
| 最高 pipe | **ALU 60.1%** | — |
| 主 stall | `long_scoreboard` 38.9% | — |
| Achieved Occupancy | 81.9% | — |

v1 里 **ALU 是最高 pipe、`long_scoreboard` 是最大 stall**（等 global 权重），是一副典型的
「算得慢 + 还没喂饱带宽」的混合瓶颈；v2 把整数指令砍掉 40% 后，**Compute 掉到 29.8%、
DRAM 升到 66.7%**，kernel 干净地变成访存受限。44 篇预言的「补 ALU 就能贴带宽」成立了。

补充一句口径差异：ncu 单次 replay 的 DRAM 66.7% 与 event 稳态的 76.3% 不是一回事
（36 篇记过——ncu 锁频 + replay 会改变稳态）。**发布口径用 event 稳态的 76.3%**，
ncu 只用来看「瓶颈在哪一级」。

---

## 六、激活量化的成本：不能装作不存在

W4A8 比 W4A16 多一个前置 kernel：把 bf16 的 `x` 按 per-128-group 动态量化成 int8。

第一版量化 kernel 用「一线程负责一个 group」——M=1 时只有 **40 个活跃线程**，被访存延迟拖成
**0.0108 ms**，比 GEMV 本体还贵，端到端反而输给 W4A16。改成「**一个 warp 负责一个 group**
（lane 处理 4 个元素 → `shfl_xor` 归约 amax → 同 warp 写回 int8）」后：

| 量化 kernel | 时间（M=1） |
|---|---|
| 一线程/group（40 活跃线程） | 0.01084 ms |
| **一 warp/group** | **0.00253 ms（4.3×）** |

于是端到端（量化 + GEMV）：

| 路径 | 时间 | vs W4A16 |
|---|---|---|
| W4A16 GEMV（无量化 kernel） | 0.0283 ms | 1× |
| **W4A8（量化 0.0025 + GEMV 0.0185）** | **0.0210 ms** | **1.35×** |

**GEMV 本体的收益是 1.53×，计入量化后仍有 1.35×。** 真实部署里这一步通常和上游算子
（layernorm / 前一个 GEMM 的 epilogue）融合，成本还能再降。

---

## 七、小 M 扫：收益边界

`MT` 个 token 共享一次权重读（`<RWW, MT>` 模板），仍按 per-128-group 量化：

| 配置 | 时间 | 权重带宽 | HBM | 单 token 时间 |
|---|---|---|---|---|
| M=1, RWW=4 | 0.0185 ms | 2408 GB/s | 76.3% | 0.0185 ms |
| M=2, MT=2, RWW=4 | 0.0241 ms | 1847 GB/s | 58.5% | 0.01205 ms |
| M=4, MT=4, RWW=2 | 0.0263 ms | 1695 GB/s | 53.7% | 0.00658 ms |

`MT` 让权重被多 token 复用，**单 token 成本在降**（0.0185 → 0.0121 → 0.0066），但
**绝对时间在涨**——M 越大，整数/`dp4a` 工作量按 M 线性增长，权重字节不变，kernel 又从带宽
受限滑回算力受限。这与 44 篇的结论一致：`M=1` 用 GEMV 贴带宽，`M ≳ 78` 该换回张量核
GEMM。本次没有给 W4A8 的张量核路径（那需要 `mma` 的 int8 / `wgmma` fp8，是另一个方向），
所以只给到「GEMV 的适用区间」。

---

## 八、对标与边界

- **对 44 篇**：同 shape、同进程、同口径，W4A8 把 M=1 从 **0.0284 → 0.0185 ms（1.53×）**、
  HBM **49.7% → 76.3%**，并越过了同进程测出的纯读上界。44 篇与 Marlin 类 SOTA 差 **≈6.7×**；
  现在这个 6.7× 里，**M=1 W4A16 与纯带宽的部分已被我们走完**，剩下的是「M 大时张量核的
  复用优势」——那是 Marlin / AWQ 的主场，不是 M=1 的问题。
- **数值边界**：W4A8 用 int8 激活，per-128 动态量化的额外相对误差 ~0.4%。对 decode 的
  自回归误差累积通常可接受（AWQ/GPTQ 的 W4A16 权重误差才是主导项），但对精度敏感任务
  需要实测。
- **框架口径**：`dp4a` 的峰值吞吐是 ALU pipe 的一部分，本来不是 GPU 的「主算力」；但对
  M=1 这种「算力需求 ≈ 0、只差把权重喂完」的算子，它恰好是对的指令——**用对工具比堆资源重要**。

---

## 九、小结

- 44 篇的 M=1 GEMV 卡在 **ALU pipe 64.9%**：反量化把每个 int4 变成 3 串 ALU 指令。
- 换 **W4A8 + `dp4a`**：激活量化成 int8（per-128），权重保持 int4，内层用整数点积。
  第一版用 `__vsubss4` 做符号调整 → 被 ptxas 展开成大量 `LOP3`/`PRMT`，只到 62.4%。
- **关键一步**：不调整符号，把 nibble 当正整数喂 `dp4a`，解析地扣掉 `8·Σq^x`
  （`Σq^x` 与行无关，每 chunk 用 `__dp4a(ones, x)` 现算一次）。指令数 11.87 M → 7.17 M。
- 结果 **0.0185 ms / 2408 GB/s / 76.3% HBM**，相对 44 篇 **1.53×**；ncu 从
  「ALU 60%」翻转为「DRAM 66.7%、Compute 29.8%」——变成纯带宽 kernel。
- 激活量化 kernel 若「一线程/group」会成瓶颈（0.0108 ms），改「一 warp/group」
  → 0.0025 ms；计入后 **端到端仍 1.35×**。
- 数值：W4A8 额外 ~0.4% 相对误差（int8 激活量化）。

**下一篇预告**：`M ∈ [2,16]` 的张量核/向量核自适应调度（44 篇的 GEMV ↔ 40/41/43 的
`m64` GEMM，按 `M` 选形状），或者回到 MLA decode——把 42 篇残留的 `wait`/occupancy 墙
用 producer/consumer warp specialization 再攻一版。

代码：`code/kernel-opt/45-w4a8-gemv/w4a8_gemv.cu`，原始输出
`sweep_M1.out.txt` / `bench_M2.out.txt` / `bench_M4.out.txt` /
`ncu_b_r2.out.txt` / `ncu_h_r4.out.txt` / `ncu_inst_cmp.out.txt`。
