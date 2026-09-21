---
title: "CUDA 算子调优（五十三）：融合 RoPE —— 交错配对、对半配对，以及一个被访问模式封顶的纯访存算子"
date: 2026-09-22T06:30:00+08:00
draft: false
weight: 53
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "RoPE", "旋转位置编码", "YaRN", "位置编码", "DeepSeek-V4", "Qwen3", "MLA", "GQA", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 32 篇]({{< relref "cuda-kernel-opt-32-fused-norm" >}})把 RMSNorm 家族推到了 87% HBM，
这一篇换 Transformer 里另一个「每层都要跑、又纯访存」的算子：**RoPE（旋转位置编码）**。

RoPE 看着简单——「把 q/k 的相邻两维转个角度」——但真正动手写会撞上三件事：

1. **两个主流模型用了两种不同的配对方式**：DeepSeek-V4 的 `apply_rotary_emb`
   用 `torch.view_as_complex`，是**交错配对** $(x_{2j}, x_{2j+1})$；而 Qwen3/LLaMA 的
   `rotate_half` 是**对半配对** $(x_j, x_{j+d/2})$。配对方式直接决定内存怎么向量化。
2. **YaRN 频率缩放**让「现场算频率」走进沟里——我们在 kernel 里逐元素 `powf`+`sincosf` 的朴素版
   只有 **8.3% HBM**。
3. 更微妙的是：这个算子的上限**不是** HBM 峰值，而是**访问模式**。DeepSeek 的 rope 区只占
   head_dim 的一小段（$64/512$），访问是「128 B 块、跨 1 KB 步长」——同模式的纯 copy 也只有
   **79%**（打平连续布局能到 86%）。把这一点量清楚，剩下全是「贴着天花板走」。

本篇把这条线走完，最后用一个 **uint4 流**版打到 **78.6%**（= 该访问模式 copy 天花板的 **99.5%**），
对半配对版打到 **82.1%**，相对 PyTorch eager 分别快 **22× / 6.4×**。所有 shape 取自本地模型
`/ssd/models/{DeepSeek-V4-Pro,Qwen3-8B}/config.json`。

前置阅读：[第 15/16 篇 MLA]({{< relref "cuda-kernel-opt-16-mla-fused" >}})（RoPE 就是 MLA 里唯一作用在
位置维的算子）、[第 32 篇 RMSNorm]({{< relref "cuda-kernel-opt-32-fused-norm" >}})（同为纯访存算子）。

---

## 一、RoPE 到底在算什么：两种配对，两个模型

对位置 $p$ 的 query 向量 $x\in\mathbb{R}^{d}$，RoPE 对每一对维度做一次平面旋转：

$$
\begin{aligned}
\text{out}_{i} &= x_i \cos(p\,\omega_{j}) - x_{i'}\sin(p\,\omega_{j})\\
\text{out}_{i'} &= x_i \sin(p\,\omega_{j}) + x_{i'}\cos(p\,\omega_{j})
\end{aligned}
$$

其中频率 $\omega_j = \text{base}^{-2j/d}$，$j=0,\dots,d/2-1$。差别只在「配对的 $(i,i')$ 是谁」：

```text
交错配对 interleaved（DeepSeek model.py: view_as_complex）
  d=8:  (x0,x1) (x2,x3) (x4,x5) (x6,x7)
         └j=0─┘ └j=1─┘ └j=2─┘ └j=3─┘
  → 一对在内存里是相邻的两个 bf16（4 B）

对半配对 split-half（HF rotate_half / Qwen3）
  d=8:  (x0,x4) (x1,x5) (x2,x6) (x3,x7)
         └j=0─┘ └j=1─┘ └j=2─┘ └j=3─┘
  → 一对相隔 d/2，分别落在两个「半区」
```

DeepSeek-V4-Pro 的语义（`/ssd/models/DeepSeek-V4-Pro/inference/model.py`）是：

```python
# precompute_freqs_cis(...)：base=10000，YaRN factor=16, beta_fast=32, beta_slow=1,
#                            original_seq_len=65536
freqs = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
if original_seq_len > 0:
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs  = freqs / factor * (1 - smooth) + freqs * smooth
freqs_cis = torch.polar(torch.ones_like(outer(t, freqs)), outer(t, freqs))

def apply_rotary_emb(x, freqs_cis):          # model.py:232
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))   # ← 交错配对
    return torch.view_as_real(x * freqs_cis).flatten(-2)
```

调用处 `apply_rotary_emb(q[..., -rd:], freqs_cis)`（`model.py:499`），$rd=\text{rope\_head\_dim}=64$，
q 的形状是 `[S, 128, 512]`——**只有每个 head 的最后 64 维需要旋转，前 448 维（nope）不动**。
kv 只有 1 个 head（MQA），形状 `[S, 512]`。

Qwen3-8B 则是标准 GQA：`head_dim=128`，$q$ 有 40 个 head、$k$ 有 8 个，`rope_theta=1e6`、无 YaRN、
`rotate_half` 对半配对。**两种模型 = 两种内存访问形态**，这就是本篇的主线。

---

## 二、Roofline：这是一个「0 FLOP/byte」的算子

一次旋转对一对元素做 4 次乘 + 2 次加（$6$ FLOP / $4$ B）——但 cos/sin 是预先算好的表，
所以对 DRAM 而言：

$$
\text{AI} = \frac{6\ \text{FLOP}}{4\ \text{B}} \approx 1.5 \ll \text{H100 bf16 ridge}\approx295
$$

**纯访存**。有效搬运量 `2 · T · H · rd · 2 B`（读一遍 rope 区、写一遍）。以
`T=16384, H=128, rd=64` 为例：$16384\times128\times64\times2\times2 = 537$ MB，在 3.35 TB/s 上
理论下限 $0.160$ ms。

> 注意「有效口径」的陷阱：整张 q 是 `16384×128×512×2 = 2.1 GB`，但**只有 rope 区（1/8）被读写**，
> 所以本文的 GB/s 一律按 rope 区的实际字节算。判断 HBM 利用率时，这 2.1 GB 的「经过但没碰」的
> 字节不算数。

---

## 三、v0：把 YaRN 算在 kernel 里 = 主动跳进沟里

最自然的写法是第一版就写在实际张量上：一线程一 pair，现场用 `powf` 算 $\omega_j$、
再做 YaRN 的线性 ramp、再 `sincosf`。结果：

| 版本 | 时间 @ds-q | 有效带宽 | 峰值占比 |
|---|---|---|---|
| **v0 现场算 YaRN + `powf`/`sincosf`** | **1.9193 ms** | **279.7 GB/s** | **8.3%** |

1.9 ms——比理论下限慢 **12×**。因为 `powf` + 双精度 `cos/sin` 是几十条 SFU/ALU 指令，
把一个 0 FLOP/byte 的算子硬生生变成了算力受限。

**唯一的正确做法是把频率表和 cos/sin 表在 host（或一个一次性 kernel）里预计算好**，
主 kernel 只做查表 + 4 乘 2 加。这就引出了 v1。

---

## 四、v1：查表版——但 elementwise 的形状会咬人

预计算 `cs[pos][j] = (cos(pos·ω_j), sin(pos·ω_j))`（fp32 或 bf16 存 `float2`），
然后「一线程一 pair」查表：

| 版本 | ds-q（交错） | qwen-q（对半） |
|---|---|---|
| v1a elementwise fp32 表 | 2024.0 GB/s（60.4%） | 1309.1 GB/s（**39.1%**） |
| v1b elementwise bf16 表 | 2021.0 GB/s（60.3%） | 1441.5 GB/s（**43.0%**） |
| v2a warp-per-row fp32 表 | 2027.6 GB/s（60.5%） | 2583.6 GB/s（77.1%） |
| **v2b warp-per-row bf16 表** | **2029.4 GB/s（60.5%）** | **2595.4 GB/s（77.4%）** |

两个现象：

- **交错配对（ds-q）v1/v2 都卡在 60%**：v1 和 v2 每 lane 都只搬 4 B（一个 `uint32`），
  且 v1 要算 `i % NP` / `i / NP` 两个 64 位整除。4 B/lane 的在飞字节太少。
- **对半配对（qwen-q）elementwise 只有 39–43%**，而 warp-per-row 有 77%：因为对半的两个元素
  相隔 $d/2=64$，elementwise 版每个线程发两条 2 B 标量读，warp 内地址还错位；warp-per-row 把
  一对固定在「lane $j$ ↔ 元素 $j$ 与 $j+d/2$」，warp 一次覆盖整行 256 B，才合并得上。

第一个通用结论：**RoPE 的性能取决于「同一条 rope 行能不能被向量化」**，而配对方式决定了它能不能。

---

## 五、v4：交错配对能当 `uint4` 流读——直接贴到模式天花板

交错配对有个漂亮的性质：**(x_{2j}, x_{2j+1}) 在内存里就是连续的 4 B**。于是一整行 rope 区
（$rd=64$ 个 bf16 = 128 B）可以当成 **8 个 `uint4` 的连续流**。让一线程吃一个 `uint4`
（8 个元素 = 4 对），连续线程地址连续：

```text
warp 的 32 个 lane，每个搬 16 B（uint4）
  lane0   lane1   lane2   lane3  ...           每行 128B = 8 个 uint4
 ┌──────┬──────┬──────┬──────┬─ … ─┬──────┐
 │ u4#0 │ u4#1 │ u4#2 │ u4#3 │     │ u4#7 │   ← row (t,h)
 └──────┴──────┴──────┴──────┴─ … ─┴──────┘
 一个 warp 覆盖 4 行（每行 8 个 uint4）
```

对每个 `uint4`：读 8 个 bf16、查 4 个 `float2`（bf16 表 4 B/对）、算 4 次旋转、写回一个 `uint4`。

| 版本 | 时间 @ds-q | 有效带宽 | 峰值占比 |
|---|---|---|---|
| v2b warp 标量（4 B/lane） | 0.2645 ms | 2029.4 GB/s | 60.5% |
| **v4 vec4（16 B/lane）** | **0.2038 ms** | **2634.0 GB/s** | **78.6%** |
| （v6 `uint4`×2/线程） | 0.2194 ms | 2446.6 GB/s | 73.0% |
| （v6 `uint4`×4/线程） | 0.2506 ms | 2142.2 GB/s | 63.9% |

换了向量宽度，**60.5% → 78.6%**。想再堆「每线程 2~4 个 uint4」反而更慢（v6 73%/64%）——
线程数变少、并行度下降，超过收益。

到 78.6% 时，直觉会问「还差 21% 去哪了」。答案的关键一步是**量出这个访问模式的天花板**。

---

## 六、关键一步：先量「同模式的 copy roof」

我们在同一份地址流上写两个诊断 kernel（保留完全相同的地址表达式，只去掉旋转/查表）：

- `pattern read-only`：只读 rope 区，xor 归约到全局 sink（防 DCE）；
- `pattern copy (r+w)`：读一个 `uint4`、异或一个 bit、写回（读+写、无旋转）。

| 场景（@T=16384） | read-only roof | copy (r+w) roof | v4 实测 |
|---|---|---|---|
| **ds-q：rope 64/512（128 B 块跨 1 KB）** | 3040.6 GB/s（90.7%） | **2647.0 GB/s（79.0%）** | 2634.0（**78.6%**） |
| **ds-q-packed：rope 区连续** | 3100.3 GB/s（92.5%） | **2909.7 GB/s（86.8%）** | 2897.5（**86.4%**） |

三个结论一次说清：

1. **v4 = 模式 copy 天花板的 99.5%**（78.6 / 79.0）。旋转 + 查表**几乎没有额外成本**，
   剩余的 21% 全是「读+写同一块 128 B 再回写」+「跨 1 KB 步长」的 DRAM 代价。
2. **rope 区连续（packed）能把天花板从 79% 抬到 86.8%**：同一份 kernel 一字未改，
   v4 也顺势从 78.6% 上到 86.4%。差的那 8 个百分点纯粹是 head_dim=512 带来的
   896 B 空洞——DRAM 只能吃下每 1 KB 里的 128 B。**如果你能控制布局，把 rope 维打包连续是最大的优化。**
3. **先量 roof 再谈优化**：有了 79% 这个数，就知道 v4 已经到顶，不该再去「抠查表」或「调 block」。
   （这是 [第 50 篇]({{< relref "cuda-kernel-opt-50-w4a8-imma-pipe" >}}) 那条「同地址流只读诊断」的又一应用。）

---

## 七、对半配对（Qwen3）：uint2×2 攻向它的天花板

对半配对没法用一个 `uint4` 覆盖（一对相隔 $d/2$）。但可以把一行拆成两个「半区」，
每个半区连续：**一线程吃 4 个连续 pair**——从半区 A 读 `uint2`（4 个 bf16 = 8 B）、
从半区 B 读 `uint2`，分别旋转后写回两个 `uint2`。

```text
split-half d=128（NP=64）：
  row:  [ A 半区 64 elems ][ B 半区 64 elems ]
         lane0 读 A[0:4] 与 B[0:4]（各一个 uint2），旋转写回
         lane1 读 A[4:8] 与 B[4:8]
  → 16 个线程覆盖一行；连续线程的 A 段 8 B 间隔 → 合并
```

| 版本 @qwen-q（T=16384,H=40,R=128） | 时间 | 有效带宽 | 峰值占比 |
|---|---|---|---|
| v1b elementwise | 0.2328 ms | 1441.5 GB/s | 43.0% |
| v2b warp 标量 | 0.1293 ms | 2595.4 GB/s | 77.4% |
| **v4s uint2×2** | **0.1220 ms** | **2750.7 GB/s** | **82.1%** |
| （模式 copy roof） | 0.1194 ms | 2809.2 GB/s | 83.8% |

v4s = 天花板的 **98.0%**。对半配对这条线也到顶了。

顺带一个 `H=8` 的小 shape（qwen3-k，67 MB）跑出 **98.4%**——但那是 **L2 助攻**（89 MB rope 区
被 50 MB L2 兜住一部分），不能当 HBM 结论，写下来是因为它提醒：**小 shape 的带宽数字必须看
ncu 的 DRAM%，别被 `Memory Throughput`（L1/L2/DRAM 最高级）骗**（系列老坑）。

---

## 八、ncu：它就是 DRAM 延迟受限

对两个最佳版本各采一次 full set（原始输出见 `ncu_v4_ds_q.out.txt` / `ncu_v4s_qwen_q.out.txt`）：

| 指标 | v4 @ds-q | v4s @qwen-q |
|---|---|---|
| Duration | 196.9 µs | 116.2 µs |
| **DRAM Throughput** | **78.45%** | **81.59%** |
| L2 Cache Throughput | 70.28% | 73.39% |
| L1/TEX Throughput | 18.96% | 21.53% |
| Compute (SM) | 25.99% | 26.83% |
| Achieved Occupancy | 90.85% | 90.95% |
| Registers/Thread | 30 | 28 |
| L2 Hit Rate | 53.5% | 50.1% |
| 主导 stall | `long_scoreboard` 50.3 / 55.1 cycle | 44.4 / 50.3 cycle |

**DRAM 到顶、Compute 只有 26%、occupancy 91%**——教科书式的访存受限。占 stall 九成的
`long_scoreboard`（等全局 load 返回）说明 warp 是在等 DRAM，不是等指令。寄存器 30 个、
occupancy 91%，也没有可压的空间。

---

## 九、两个「和直觉相反」的结果

**① 把 cos/sin 表常驻寄存器（v5）反而更慢。**
我的第一直觉是：同一 token 的 $H=128$ 个 head 共享同一份 rope 表，那让「一 warp 一 token、
表留寄存器、内层循环摊到 128 个 head」就能把表流量砍 $128\times$。实测：

| 版本 @ds-q | 时间 | 带宽 | 占比 |
|---|---|---|---|
| v2b（每 head 重读表） | 0.2645 ms | 2029.4 GB/s | 60.5% |
| **v5 表常驻寄存器（摊 128 head）** | **0.2563 ms** | **2094.6 GB/s** | **62.5%** |
| v4 vec4（根本不管表） | 0.2038 ms | 2634.0 GB/s | **78.6%** |

表流量**从来不是墙**——表只有 2 MB、命中 53% 的 L2，主 kernel 是 DRAM 受限。v5 反而因为
每 lane 只搬 4 B（又是 4 B/lane！）掉回 62%。**memory-bound 时别去优化 L1 侧的东西**，
这是 v1/v2/v5 三次踩到的同一个坑。bf16 表 vs fp32 表（v2a/v2b、v1a/v1b）也几乎无差，同理。

**② 表本身可以更小但不重要。** 理论上 cos/sin 可以只用 $\cos$ 表（$\sin$ 用相位搬移），
或者只在 token 级存一次；但既然表不构成流量，这些都不值得做。

---

## 十、对标 SOTA：PyTorch eager 的口径

同一份数据、同进程测 PyTorch 实现（`rope_ref.py`，容器内 GPU 计时；需求是「先算出 rope 区
再旋转」，与我们的口径一致）：

| 实现 | 时间 | 有效带宽 | 相对我们 |
|---|---|---|---|
| torch clone（连续布局 copy roof） | 0.1793 ms | 2994.8 GB/s（89.3%） | — |
| torch interleaved（`view_as_complex`，model.py 语义）@ds-q | 4.5654 ms | 117.6 GB/s | **我们 v4 快 22.4×** |
| torch `rotate_half`（HF/Qwen3）@qwen-q | 0.7811 ms | 429.6 GB/s | **我们 v4s 快 6.4×** |

torch 的 interleaved 版慢是因为它 `x.float()` 后再 `view_as_complex`，引入了 fp32 物化和
逐元素视图运算——正是我们用 `uint4` 一把绕过的部分。此外我们与 torch 的 reference 对拍
（fp64 复刻 `model.py`）最大绝对误差 **4.0e-3 / 4.2e-3**（bf16 量化级），正确。

> 对标口径：torch 的 `rotate_half` 没有官方融合 kernel，这里用纯 torch 张量算子；生产里
> vLLM/FlashAttention 会用自己的 rotary kernel，但其核心矛盾（配对方式 + 向量宽度 + 布局）
> 与本文一致，故本文以「torch eager」为下限、以「同模式 CUDA copy roof」为上界给出可复现区间。

---

## 十一、小结

- **纯访存算子，先预表**：kernel 里算 YaRN/`powf`/`sincosf` → **8.3%**；预计算 cos/sin 表后
  立刻到 60%。
- **配对方式决定向量化**：DeepSeek 交错配对 $(x_{2j},x_{2j+1})$ 是连续 4 B → 可当 `uint4` 流，
  **78.6%**（= 模式 copy 天花板的 99.5%）；HF 对半配对需 `uint2×2`，**82.1%**（= 98.0%）。
- **上限是访问模式，不是 HBM 峰值**：rope 64/512 = 「128 B 块跨 1 KB」，同模式 copy 只有
  **79%**；把 rope 区打包连续，同一 kernel 立刻 **86.4%**。**先量只读/copy roof，再谈优化。**
- **表流量不是墙**：把表常驻寄存器（摊 128 个 head）反而掉到 62%——DRAM 受限时别优化 L1 侧。
- 4 B/lane 是三个版本共同的低点（v1 60%、v2 60%、v5 62%），16 B/lane（v4）才是正解；
  ncu 确认 DRAM 78–82%、Compute 26%、`long_scoreboard` 占 stall 九成。
- SOTA：torch eager interleaved 慢 **22×**、`rotate_half` 慢 **6.4×**；最高成绩
  **v4 2634 GB/s（78.6%）/ v4s 2751 GB/s（82.1%）**。

**下一篇预告**：把这条「先量同模式 roof」的方法带回 FP8/MoE——也就是 [第 37 篇]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}})
把 ue8m0 的 2 的幂 scale 折进操作数后，per-block 的 fused MoE（[第 34 篇]({{< relref "cuda-kernel-opt-34-fused-moe-fp8" >}})）
能回收多少端到端损失；或继续 backlog 里的 26l（把 TMA + host 置换布局推广到别的低比特算子）。
