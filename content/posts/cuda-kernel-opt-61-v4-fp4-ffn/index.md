---
title: "CUDA 算子调优（六十一）：FP4 decode 的 MoE FFN 端到端 —— w1/w3→SwiGLU→w2，以及 cp.async 到底救不救得了延迟墙"
date: 2026-09-22T10:40:00+08:00
draft: false
weight: 61
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MoE", "FP4", "e2m1", "E8M0", "DeepSeek-V4", "SwiGLU", "GEMV", "grouped", "cp.async", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 59 篇]({{< relref "cuda-kernel-opt-59-v4-fp4-grouped" >}}) 和
[第 60 篇]({{< relref "cuda-kernel-opt-60-v4-fp4-lut" >}}) 把 DeepSeek-V4-Pro 一步 decode 的
`B×6` 个 `(token, expert)` 对按 expert 折叠，做出了一个 grouped GEMV：权重每专家只读一遍、
被若干个 token 复用，`w1 [N=3072, K=7168]` 的 FP4 权重在 `B=64` 时读到 **2772 GB/s（82.7% HBM）**。
但那只是 **一张矩阵**。模型里的 routed expert 是一个完整的 SwiGLU FFN：

```
            ┌─────────── 每个 (token, expert) 对 ───────────┐
  x[7168] ──► w1 ─► gate[3072] ─┐
          └► w3 ─► up  [3072] ─┴─► silu(clamp(gate)) * clamp(up) ─► h[3072] ─► w2 ─► y[7168]
```

一步 decode（`B` 个 token）要真实地跑完整条链路，并且给出**端到端延迟**。本篇做的事：

1. 用 `/ssd/models/DeepSeek-V4-Pro/config.json` 的真实形状（`hidden=7168, moe_inter=3072,
   384 experts, top-6, expert_dtype=fp4`）和 `inference/model.py:596` 的真实语义，
   把 `quant_x → w1/w3 → SwiGLU+per-128 int8 量化 → w2 → 加权反置换` 写成一条 5 段流水；
2. 全部对拍通过（端到端相对误差 **~1e-7**）；
3. 做两个 26p 提议的实验：**`cp.async` 双缓冲**、**把权重 scale 换成真实 E8M0 uint8**；
4. 给出端到端 **5.97 ms（B=64，读 13.48 GB FP4 权重，67.6% 权重 roofline）**。

结论先摆出来（都和直觉有点反）：

> **① `cp.async` 只在"双缓冲（DEPTH=2）"时略微赢寄存器预取环**（B=64 端到端 5.95 vs 7.88 ms，
> PIPE=1 对 PIPE=0）；**加深到 3 级反而更慢**（3.93 → 3.95 ms，K2 更明显）。
> **② 把 float scale 换成真实 checkpoint 的 E8M0 uint8（字节 −15%）只换来 5.5% 提速** ——
> grouped GEMV 在小 M 下不是纯带宽上限，**省字节 ≠ 省时间**。
> **③ 真正的坑是一个语义 bug**：K1 的激活是 **per-token**（按 `toks[]` gather），
> K2 的激活（SwiGLU 输出）是 **per-(token,expert) 对**（不能 gather）。同一份 kernel 必须区分。

---

## 一、真实形状与真实语义

### 1.1 形状（`/ssd/models/DeepSeek-V4-Pro/config.json`）

| 量 | 值 | 说明 |
|---|---|---|
| `hidden_size` H | 7168 | FFN 的输入/输出维 |
| `moe_intermediate_size` I | 3072 | expert 中间维 |
| `n_routed_experts` | 384 | routed expert 数 |
| `num_experts_per_tok` | 6 | top-6 |
| `expert_dtype` | `fp4` | 真实 routed expert 权重是 **FP4 e2m1 + E8M0 block-32**（56 篇实测） |
| `swiglu_limit` | 10.0 | 官方实现里 gate/up 的 clamp 上界 |

真实的 `w1 [3072,7168]` / `w3 [3072,7168]` / `w2 [7168,3072]` 三张 FP4 权重
（单专家打包字节 11 MB / 11 MB / 11 MB）直接从 56 篇 `extract.py` 落盘的 `.bin` 读取，
本程序把 384 个专家都铺满显存（而不是只放一个专家、全命中 L2），保证测的是 HBM 带宽。

### 1.2 语义（`inference/model.py:596`）

```python
gate = self.w1(x).float()          # [.., I]
up   = self.w3(x).float()          # [.., I]
if self.swiglu_limit > 0:
    up   = torch.clamp(up,   min=-limit, max=limit)
    gate = torch.clamp(gate, max=limit)      # 注意 gate 只 clamp 上界
x = F.silu(gate) * up              # [.., I]
return self.w2(x.to(dtype))
```

注意官方是 `gate=w1`、`up=w3`，**gate 只钳上界**、up 双边钳。这些细节写实现时必须对齐，
否则对拍只会发现一个"差一点"的错误。

---

## 二、五段流水与 kernel 设计

```
   ┌──────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌────────────┐   ┌───────────┐
   │ quant_x      │   │ K1 grouped GEMV  │   │ swiglu+quant │   │ K2 grouped │   │ unpermute │
   │ bf16→int8    │──►│ [w1;w3] fp4      │──►│ GU→h int8    │──►│ w2 fp4     │──►│ 加权求和  │
   │ per-128      │   │ → GU[pairs,2I]   │   │ per-128      │   │→O[pairs,H] │   │ Y[B,H]    │
   └──────────────┘   └──────────────────┘   └──────────────┘   └────────────┘   └───────────┘
     Xq,Xs              GU fp32               Hq,Hs              O fp32
```

**K1 把 w1 和 w3 拼成一张 `[2I, H] = [6144, 7168]` 的矩阵**，直接复用 59/60 的 grouped GEMV：
输入 `Aq/Axs`（int8 + per-128 scale），权重 `Wall[E, N, K/2]` FP4、`Wsall[E, N, K/32]` scale，
输出 `Y[pairs, N]`。每个 CTA 吃某个 expert 的一小段输出行 `BN = 8·RWW`，全 kernel 一次调度
全部 active expert。K2 同理，只是形状变成 `N=7168, K=3072`。

`swiglu+quant` 是一个独立小 kernel：一个 warp 处理一个 `(pair, 128-group)`，做
`silu(clamp gate)·clamp up` 后按 128 元素求 `amax`、动态量化到 int8。它的流量只有 ~10 MB
（`GU` 读 9.4 MB + `Hq/Hs` 写 1.2 MB），占端到端权重流量的 **0.08%** ——
所以**没有必要把它融进 GEMV**（详见第六节的负结果讨论）。

### 2.1 一个必须区分的语义：K1 gather、K2 不 gather

59/60 的 grouped GEMV 里，激活行号是 `Aq + toks[off+mm]*K` —— 按 **token 号** gather。
这对 K1 成立（输入 `X[B,H]` 是 per-token 的）。但 **K2 的激活是 SwiGLU 输出 `h[pairs,I]`，
它是 per-(token,expert) 对的**，行号就是 pair 索引 `off+mm`，**不能再 gather**。

我们第一版直接复用 K1 的 kernel 跑 K2，结果 8 个采样点全部 **max_rel ≈ 3.3（FAIL）**，
而 K1/swiglu 全过。加了一个 `GATHER` 模板参数（`ar = GATHER ? toks[off+mm] : (off+mm)`）后
K2 立刻回到 **1.4e-7**。这个 bug 的隐蔽之处在于：**`B=64` 时它碰巧不发作** ——
balanced 路由下每个 expert 恰好 1 个 pair，grouped 行号 `off+0` 与原 pair 号相等；
`B≥128` 每专家 2~4 个 pair，两者才分叉。**所以对拍必须跨 batch 做。**

---

## 三、性能：几何 sweep、MTMAX 分派、cp.async、uint8 scale

### 3.1 先把 geometry 扫一遍

固定 `MTMAX=4`，扫 `RWW ∈ {2,4,8}` × `DEPTH ∈ {2,3,4}` × `PIPE ∈ {0(寄存器环),1(cp.async)}`
（`sweep_uint8.out.txt`，B=64，balanced）：

| 配置 | K1 ms | K1 GB/s | K2 ms | K2 GB/s |
|---|---|---|---|---|
| RWW4 s2 PIPE0 | 4.06 | 2214 (66.0%) | 2.41 | 1867 (55.7%) |
| RWW4 s3 PIPE0 | 4.45 | 2018 (60.2%) | 2.57 | 1747 (52.1%) |
| RWW4 s4 PIPE0 | 4.07 | 2209 (65.9%) | 2.57 | 1747 (52.1%) |
| **RWW4 s2 PIPE1** | **3.93** | **2288 (68.3%)** | 2.30 | 1956 (58.3%) |
| RWW4 s3 PIPE1 | 3.95 | 2275 (67.9%) | 2.42 | 1854 (55.3%) |
| **RWW8 s2 PIPE1** | 12.70 | 707 (21.1%) | **2.27** | **1979 (59.0%)** |
| RWW2 s2 PIPE1 | — | — | 2.99 | 1502 (44.8%) |

三个结论：

- **`DEPTH` 不是越深越好**。PIPE=0 的寄存器环从 `DEPTH=2` 加深到 3/4 反而变慢
  （K1 4.06 → 4.45/4.07），PIPE=1 的 `cp.async` 也是 `s2 > s3`。寄存器/`cp.async` 组数一多，
  在途请求并没有变成有效 MLP，反而挤压了 x 片段的寄存器预算。
- **RWW=8 对 K1 是灾难**（12.7 ms，21%）：8 行 × 双 stream 的累加器 + 环把寄存器顶爆，
  掉到 1 CTA/SM；但对 K2（K 只有 3072、行更短）反而略优（2.27 vs 2.30）。
- **K1 选 `RWW4 s2 PIPE1`，K2 选 `RWW8 s2 PIPE1`**。这是我们随后所有测数的默认。

> 顺带：PIPE=1 与 PIPE=0 在同一 kernel 里只差"权重进 smem 再读"这一步，所以这个 sweep
> 是干净的 cp.async 消融。**cp.async 赢的幅度只有 ~3~5%，不是数量级。**

### 3.2 `MTMAX` 跟着 B 走

GEMV 的寄存器/smem 预算里，`acc[RWW][MTMAX]`、`x8[MTMAX][8]` 都乘 `MTMAX`（=每专家最多几个 token）。
一步 decode 的 `m = B/64`，所以 `B=64/128/256 → MTMAX=1/2/4`。把 `MTMAX` 做成模板、
按 `B/64` 分派后：

| B | PIPE1 K1 | PIPE1 K2 | 端到端 |
|---|---|---|---|
| 64 (m=1) | 3.76 ms（72 regs，occ 36.4%） | 2.18 ms | **5.97 ms** |
| 128 (m=2) | 4.05 ms | 2.40 ms | 6.48 ms |
| 256 (m=4) | 4.84 ms | 2.71 ms | 7.64 ms |

`B=64` 时 `MTMAX=1` 让 K1 从 3.93 → 3.76 ms、K2 从 2.27 → 2.18 ms，端到端 **+5%**。
代价是 `B=256` 没有 `MTMAX>4` 的档位（那会超 255 寄存器），`m=4` 只能 4 个 token 一组。

### 3.3 权重 scale：真实 E8M0 uint8（省 15% 字节，只快 5.5%）

59/60 里权重 scale 在 device 上存成 **float**（`exp2(e-127)` 预先算好），
而真实 checkpoint 里它就是 **E8M0 uint8**（每 32 个 k 一个字节）。改成读 uint8 后：

```cpp
// e[0..255] 是 E8M0 指数字节；2^(e-127) 的 IEEE754 位模式恰好是 (e<<23)
const float sw = __int_as_float((int)__ldg(&Wse[(size_t)(row_base + r) * NS32 + c]) << 23);
```

一个移位 + 一次 `int_as_float`，**精确**（`e∈[1,254]`；`e=0` 给 0，与 `2^-127` 数值上等价），
比 `exp2f` 便宜。字节账：K1 权重 8.46 GB packed + scale 0.53 GB（uint8）
对比 float scale 的 2.11 GB —— **总权重 13.48 GB vs 15.85 GB，−15%**。

但实测（同一 geometry，`RWW4 s2 PIPE1`，B=64）：

| scale 存法 | K1 (8.98 GB) | K1 GB/s | 端到端 |
|---|---|---|---|
| float（15.85 GB 总量） | 4.16 ms | 2542 (75.8% of 10.57GB) | ~6.6 ms |
| **uint8 E8M0（13.48 GB 总量）** | **3.93 ms** | 2288 (68.3% of 8.98GB) | ~5.97 ms |

**字节少了 15%，K1 只快了 5.5%。** 因为 grouped GEMV 在小 M 下从来不在纯带宽天花板上
（见下节 ncu：DRAM 71%，Compute 已达 76%）。真实 checkpoint 用 uint8 是"顺便"的收益，
但**别指望靠减字节解决小 M 的延迟/发射墙**。

---

## 四、ncu：墙在哪

`B=64, PIPE=1`（最终配置；`ncu_k1_B64.out.txt` / `ncu_k2_B64.out.txt` / `ncu_swi_B64.out.txt`）：

| 指标 | K1（w1+w3，8.98 GB） | K2（w2，4.49 GB） | swiglu+quant |
|---|---|---|---|
| Duration | 3.76 ms | 2.17 ms | 8.5 µs |
| **DRAM Throughput** | **71.4%** | 62.0% | 33.3% |
| **Compute (SM)** | **75.9%** | 65.3% | 25.9% |
| L1/TEX | 71.8% | 60.8% | 13.0% |
| L2 | 73.4% | 64.4% | 39.7% |
| Achieved Occupancy | **36.4%**（72 regs） | **23.5%**（96 regs） | 72.7% |
| Issued Ipc Active | 2.55 | 2.21 | 1.65 |
| 主导 stall | — | `long_scoreboard` ~35% | — |

解读：

- **K1 已经不是纯访存墙**：DRAM 71% 与 Compute 76% 几乎齐平。`nibble16` LUT
  （每 weight byte 两次 LDS + 若干 `LOP3` 拼装）+ `dp4a` + scale 解码一起把 SM 顶到了 76%。
  想再往上，要么换解码方式（59/60 已证 `PRMT` 更慢），要么减少每字节的 LUT 次数。
- **K2 是延迟/并行度受限**：DRAM 62%、Compute 65%、L1 61%、L2 64%，**没有任何一级到顶**，
  占用率 23.5%（96 regs → 2 CTA/SM），主导 stall 是 `long_scoreboard`。
  K2 的行只有 `K=3072`（3 个 16B chunk/行），每行工作量太小，延迟藏不住。
- **`swiglu+quant` 完全不值一提**：8.5 µs，DRAM 33%。它是 memory-bound 的小 kernel
  （~10 MB），占端到端 0.14%。**pipeline 里所有"非权重"环节加起来不到 0.2 ms。**

---

## 五、端到端结果

`B=64/128/256`，对照组 `PIPE=0`（寄存器预取环）vs `PIPE=1`（`cp.async` 双缓冲）。
有效带宽按"读到的权重字节（packed + uint8 scale）"计，roofline = 13.48 GB / 3352 GB/s = **4.02 ms**：

| B | PIPE | quant_x | K1 | swiglu | K2 | unperm | 端到端 | % roofline |
|---|---|---|---|---|---|---|---|---|
| 64 | 0 | 0.003 | 5.20 | 0.006 | 2.66 | 0.009 | 7.91 ms | 51.0% |
| **64** | **1** | 0.003 | **3.76** | 0.006 | **2.18** | 0.009 | **5.97 ms** | **67.6%** |
| 128 | 0 | 0.005 | 4.10 | 0.008 | 2.67 | 0.015 | 6.83 ms | 59.1% |
| **128** | **1** | 0.005 | **4.05** | 0.008 | **2.40** | 0.015 | **6.48 ms** | **62.1%** |
| 256 | 0 | 0.011 | 4.99 | 0.017 | 3.04 | 0.034 | 8.12 ms | 49.7% |
| **256** | **1** | 0.011 | **4.84** | 0.017 | **2.71** | 0.034 | **7.64 ms** | **52.8%** |

（单位 ms；`end-to-end` 与 kernel 之和差 <0.02 ms，即 launch 开销可忽略。）

- **`cp.async` 在 B=64 上带来 25% 的端到端提速**（7.91→5.97 ms），但要注意这是
  **PIPE0 在 `MTMAX=1` 下表现很差**（K1 5.20 ms）造成的；在 `MTMAX=4` 的同一 geometry 下
  PIPE0 是 4.06 ms、PIPE1 是 3.93 ms，**只差 3%**。所以正确说法是：
  **cp.async 是"小赚"，且它对 `MTMAX` 更鲁棒；寄存器环在 `MTMAX` 变小时会反常退化。**
- **`B` 越大越慢**（5.97→6.48→7.64 ms）：权重字节不变，但 `m` 从 1 涨到 4，
  `dp4a`/LUT/激活的工作量线性涨，而 HBM 利用率从 71% 掉到 55%。这是 decode GEMV 的固有形状账。

### 5.1 对标

- **纯权重读 roof**：13.48 GB / 3352 = 4.02 ms，我们 5.97 ms → **差距 1.49×**。
- **对 60 篇的单矩阵 grouped GEMV**：60 在 `B=64` 读 `w1` 5.28 GB 用 1.91 ms（2772 GB/s，82.7%）。
  按 60 的速率外推到我们的 8.98 GB（w1+w3+scale）应约 **3.24 ms**，我们 K1 是 3.76 ms
  → **慢 16%**。差距来自"把两张矩阵合进一个 kernel"（LUT/发射工作翻倍）与 K1 的几何。
- **数制红利**：同一份 FFN 若用 FP8 专家权重（34 篇的口径）要读 ~25 GB，bf16 要 ~50 GB；
  FP4 把这一步的权重字节砍到 **13.48 GB**，是 FP8 的 0.54×、bf16 的 0.27×。
  但要拿到这个红利，**必须接受一位/字节量的"解码税"**，而 decode 小 M 下这笔税正好卡在 SM 发射口。

---

## 六、负结果与踩坑

1. **"把 per-128 量化融进 K1 的 GEMV epilogue"没做，而且不该做。**
   grouped GEMV 的 warp 一次只看到 `RWW` 行输出，要凑齐一个 128-group 需要跨 CTA 归约。
   而独立的 `swiglu+quant` kernel 只花 8.5 µs、占端到端 0.14%。**先量搬运占比，再决定融不融。**
2. **`cp.async` 加深流水（DEPTH 3/4）全线更慢**：K1 `s3` 3.95 vs `s2` 3.93，K2 `s3` 2.42 vs `s2` 2.30。
   smem 变大、每 CTA 在途 stage 变多没有转成有效延迟隐藏，反而压低了 occupancy。
3. **`RWW=8` 对 K1 是寄存器墙**：12.7 ms（21%），`acc[8][MTMAX]` + 双 stream 环直接顶爆。
   `RWW` 要按 `K` 的长短选：K2 的 `K=3072` 行短，`RWW=8` 才划算。
4. **对拍的 batch 依赖性 bug**（第二节）：K2 激活是 per-pair，不能 gather；
   `B=64` 时 grouped 行号与原 pair 号巧合相等，**只有跨 batch 对拍才暴露**。
5. **`__int_as_float(e<<23)` 要保证 `e` 是无符号**：`uint8_t` 的 `__ldg` 返回 unsigned，
   若误走 `const char*` 重载会被符号扩展，`e≥128` 的 scale 全错（而小数值 scale 恰好看不出）。

---

## 七、小结

- 用真实 checkpoint 的 FP4 权重（e2m1 + E8M0 block-32）跑通了 DeepSeek-V4-Pro 一步 decode 的
  **完整 routed-expert FFN**（`w1/w3 → SwiGLU → per-128 int8 量化 → w2 → 加权反置换`），
  384 专家铺满显存、对拍端到端 **~1e-7**。
- 端到端 **5.97 ms @B=64**（读 13.48 GB FP4 权重，**67.6% 权重 roofline**）；
  `B=128/256` 分别 6.48 / 7.64 ms。K1（w1+w3）ncu **DRAM 71% + Compute 76%**（LUT/dp4a 发射受限），
  K2 **61% DRAM / 23.5% occupancy**（延迟受限）。
- 两个提议的实验都做了：`cp.async` 双缓冲是"小赚且更稳"（同 geometry 3~5%，但对 `MTMAX` 鲁棒）；
  **uint8 E8M0 scale 省 15% 字节只快 5.5%**，证明小 M grouped GEMV 不在纯带宽墙上。

下一步（ROADMAP 26q）候选：把 K1 的解码从"每 byte 一次 LUT"换成**整 16B chunk 的直接位拼装**
（`PRMT` 在 59/60 输过，但那时是单矩阵；K1 有两张矩阵、LUT 压力翻倍，值得再量一次），
或者按 token 的**路由复用**（一对 token 命中同一个 expert 时共享权重的更大 tile）去改 K2 的并行粒度。

---

*代码：`code/kernel-opt/61-v4-fp4-ffn/fp4_ffn.cu`；实测输出 `ffn_B{64,128,256}_pipe{0,1}.out.txt`、
`sweep_uint8.out.txt`；ncu `ncu_k1_B64.out.txt`、`ncu_k2_B64.out.txt`、`ncu_swi_B64.out.txt`。*
