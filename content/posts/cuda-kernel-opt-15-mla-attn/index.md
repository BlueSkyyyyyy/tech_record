---
title: "CUDA 算子调优（十五）：MLA 注意力（一）— 从数学到第一个能跑的 kernel"
date: 2026-09-21
draft: false
weight: 15
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "MLA", "注意力", "Attention", "DeepSeek", "Kimi", "FlashMLA", "Tensor Core", "在线softmax", "系列"]
categories: ["算子开发"]
---

前十四篇我们一直在打磨「通用积木」：访存、归约、softmax、GEMM、Tensor Core、流水线、epilogue。从这一篇开始进入**第五部分「模型场景算子」**——把 DeepSeek-V4.x / Kimi-K2.6 / Qwen3 这些真实模型里的算子自己写出来、推到极致。第一站是**MLA（Multi-head Latent Attention）**，它是 DeepSeek-V2 提出、V3/V4 和 Kimi-K2 都在用的注意力变体。

配套代码 [`code/kernel-opt/15-mla-attn/mla_attn.cu`](https://github.com/BlueSkyyyyyy/tech_record/blob/main/code/kernel-opt/15-mla-attn/mla_attn.cu)，原始输出、`ncu` 数据同目录。硬件仍是 H100 80GB HBM3（132 SM，HBM ~3.35 TB/s，BF16 TC dense 峰值 989 TFLOPS）。

要回答的问题：

1. MLA 到底在算什么？为什么它能「省 KV cache」？所谓「吸收（absorb）」又是怎么来的？
2. DeepSeek-V3/V4、Kimi-K2.6 的 MLA 参数各是多少？kernel 该按什么 shape 写？
3. 一个**朴素实现**能跑到什么程度？瓶颈在访存还是算力？
4. 换到 Tensor Core 能快多少？离 FlashMLA 还有多远？

先给结论（H=128 heads，Sq=Sk=1024，DC=512/DR=64/DV=512，bf16 输入）：

| 版本 | 做法 | 耗时 | 算力 | %bf16 峰值 |
|---|---|---|---|---|
| `naive` | 一个 warp 一个 (query, head)，直接读 global | 17.11 ms | 17.07 TFLOPS | 1.7% |
| `head2` | 一个 warp 扛 2 个 head，KV 寄存器复用 | **15.50 ms** | **18.84 TFLOPS** | 1.9% |
| `smem8` | KV 分块进 smem，query/head 双重复用 | 17.91 ms | 16.31 TFLOPS | 1.6% |
| `tc` | **Tensor Core 三 kernel（QKᵀ / softmax / PV）** | **2.53 ms** | **115.6 TFLOPS** | **11.7%** |

**从标量 FFMA 到 Tensor Core 跳了 6.8×**；但即便这样，离 FlashMLA 在 H800 上公布的 **660 TFLOPS（compute-bound）** 仍有 ~5.7× 差距——这正是第十六篇要填的坑。

---

## 1. MLA 在算什么：低秩 + decoupled RoPE

标准 MHA 里，每个 token 要缓存 `H·(d_k+d_v)` 个元素；GQA/MQA 靠「多个 query head 共享 KV head」省 cache。MLA 走的是另一条路：**把 KV 压到一个低秩 latent `c_kv`，只缓存它**。

一次 MLA 前向（写标准 MHA 形状，方便理解）：

$$
\begin{aligned}
c_q &= W_{dq}\,h, & c_{kv} &= W_{dkv}\,h,\\
q_{nope} &= W_{uq}\,c_q, & q_{rope} &= W_{uqr}\,c_q,\\
k_{nope} &= W_{uk}\,c_{kv}, & v &= W_{uv}\,c_{kv},\\
k_{rope} &= W_{ukr}\,h.
\end{aligned}
$$

其中 `h` 是 hidden（DeepSeek 7168 维），`c_q ∈ R^{1536}`（q_lora_rank），`c_kv ∈ R^{512}`（kv_lora_rank）。关键设计：

- **Q 也低秩**：`c_q` 是 1536 维，再由它投影出 `q_nope/q_rope`；
- **KV 低秩 + 共享 RoPE**：只缓存 `c_kv`（512 维）和一份 `k_rope`（64 维），**与 head 数无关**；推理时 KV cache 每 token 只有 `512+64=576` 个 bf16，而标准 MHA 是 `H·(128+128)=128·256`。这就是 MLA 省 cache 的来源；
- **decoupled RoPE**：RoPE 只作用在 64 维的 `rope` 分量上，`nope` 分量不带位置信息。原因见 DeepSeek-V2 论文：对低秩 latent 直接做 RoPE 会破坏它「可以吸收进权重」的性质。

注意力本身和普通注意力一样：

$$
a_{ij} = \big(q_{nope,i} + \mathrm{RoPE}(q_{rope,i})\big)\cdot\big(k_{nope,j} + \mathrm{RoPE}(k_{rope,j})\big),
\qquad
o_i = \sum_j \mathrm{softmax}_j(a_{ij})\,v_j .
$$

### 吸收：把 `W_uk` / `W_uv` 折进计算

推理（乃至训练前向）里有一个等价变形。注意 `k_nope,j = W_uk c_kv,j`，所以

$$
q_{nope,i}\cdot k_{nope,j}
= q_{nope,i}^\top W_{uk}\, c_{kv,j}
= \underbrace{(W_{uk}^\top q_{nope,i})}_{=:\,\tilde q_{nope,i}}\cdot c_{kv,j}.
$$

同理，输出

$$
o_i = \sum_j p_{ij}\,W_{uv}c_{kv,j} = W_{uv}\underbrace{\Big(\sum_j p_{ij}\,c_{kv,j}\Big)}_{=:\,\tilde o_i}.
$$

于是整个注意力可以**只在 512 维 latent 空间**里做：

$$
a_{ij} = \tilde q_{nope,i}\cdot c_{kv,j} + q_{rope,i}\cdot k_{rope,j},
\qquad
\tilde o_i = \sum_j \mathrm{softmax}_j(a_{ij})\,c_{kv,j},
$$

最后再乘一次 `W_uv`（可与输出的 o-proj 合并）。beautiful 的地方在于：**所有 query head 共享同一份 `c_kv` 和 `k_rope`——吸收后的 MLA 就是一个 MQA**。这正是 FlashMLA kernel 的口径（MQA：`head_dim_k=576, head_dim_v=512`）。

### 真实模型的参数

从 `/ssd/models/*/config.json` 读到（`DC` = latent 维度、`DR` = 共享 RoPE 维度、`DV` = 吸收后 value 维度）：

| 模型 | heads | kv_head | DC | DR | qk_nope | v_head_dim | kernel 形态 |
|---|---|---|---|---|---|---|---|
| DeepSeek-V3 | 128 | 1 | 512 | 64 | 128 | 128 | absorb 576 / 512 |
| DeepSeek-V4-Pro | 128 | 1 | 448 | 64 | 448 | 512 | absorb 512 / 512 |
| DeepSeek-V4.1-Flash | 64 | 1 | 448 | 64 | 448 | 512 | absorb 512 / 512 |
| Kimi-K2.6 | 64 | (MHA) | 512 | 64 | 128 | 128 | materialize 192 / 128 |

几点说明：

- V4/V4.1 的 `head_dim=512`、`qk_rope_head_dim=64`，所以 `qk_nope=448`；FlashMLA 文档给出 V4 的 MQA 口径 `head_dim_k=512`（=448+64）也印证了这点；
- Kimi-K2.6 的 `text_config` 里 `qk_nope=128, qk_rope=64, v_head_dim=128, kv_lora_rank=512`，它是 64 个 KV head（不是 1），te-perf 里出现的 `(1,4096,64,192,128)` 就是它 materialize 后的 MHA 形状；
- 本篇 kernel 采用**最经典、文档最全的 DeepSeek-V3 口径**（`H=128, DC=512, DR=64, DV=512`），也就是 FlashMLA 密集 kernel 的 576/512 口径，方便和 SOTA 对齐。

---

## 2. 复杂度与 roofline：为什么标量版注定慢

一次前向的 FLOPs 与**理论最少读取量**（`c_kv+k_rope` 只读一遍、bf16）：

$$
F = 2\,S_q H S_k (D_C + D_R + D_V),
\qquad
B_{\min} = S_k (D_C + D_R)\cdot 2 .
$$

- 分数的 QK 部分算 `2·Sq·H·Sk·(DC+DR)`，P·V 部分算 `2·Sq·H·Sk·DV`；
- 读的量与 `Sq`、`H` 无关——**只要复用做得好，KV 只需读一遍**。

用 `H=128, DC+DR=576, DV=512` 算算术强度：

$$
\text{AI} = \frac{2H(D_C+D_R+D_V)}{(D_C+D_R)\cdot 2} = \frac{128\times1088}{576}\approx 241.8\ \text{FLOP/byte}.
$$

H100 的机器平衡点（ridge point）是 `989/3.35e3 ≈ 295 FLOP/byte`（bf16 TC），FFMA 的 ridge 只有 `66.9/3.35 ≈ 20 FLOP/byte`。这张图给出了整篇文章的走向：

```
 FLOP/byte
   ^
   |                         ideal MLA (241.8)
   |                                |
295|--------------------------------|-------- TC ridge
   |                                |
242|................................●  <-- 完美复用时的位置：略偏访存侧
   |                                |
 20|--------● FFMA ridge            |
   |        |                       |
   |   scalar kernel 落在这条线右边：被 FP32 pipe 顶住
   +-------------------------------------------> 带宽
```

- **理想实现**的 AI(241.8) 低于 TC ridge(295)，属于「轻微访存受限」——所以 FlashMLA 才会同时宣传「660 TFLOPS」和「3000 GB/s」两种极限配置；
- 但一个**标量 FFMA** 实现，算力上限只有 FP32 pipe 的 66.9 TFLOPS（= 6.8% 的 bf16 TC 峰值）。换句话说，**标量 MLA 无论怎么调，天花板就在 7% 以下**。Tensor Core 不是「优化」，是「入场券」。

---

## 3. 三个标量实现与一个反直觉的发现

代码里按「复用阶梯」写了三个版本（`mla_attn.cu:103/150/196`）：

- `naive`：一个 warp 负责一个 `(query, head)`，每个 key 直接 `ld.global` 读 `c_kv`，online softmax。KV 被 `Sq·H` 个 warp 各读一遍；
- `head_reuse<HG>`：一个 warp 负责一个 query 的 `HG` 个 head，把 `c_kv[key]` 载入寄存器后**在同一 key 上复用给 HG 个 head**。读放大降到 `H/HG`；
- `smem<HG,BQ,KT>`：一个 block 负责 `BQ` 个 query × `HG` 个 head，KV 以 `KT=32` 为 tile 搬进 smem，在 `BQ` 个 warp 间再复用一层。读放大降到 `H/(HG·BQ)`。

```
naive       : 每个 (q,h) 自己读一遍 KV        读放大 = H
              q0h0 q0h1 ... q0h127  各自 -> [c_kv]
head_reuse  : 同一 q 的 HG 个 head 共享一次读  读放大 = H/HG
              (q0,h0..h4) -> [c_kv] 载入寄存器
smem        : BQ 个 q × HG 个 head 共享 smem tile  读放大 = H/(HG·BQ)
              block(q0..q4, h0..h4) -> smem[KT][DC]
```

结果（Sq=Sk=1024，完整输出见 `mla_attn.out.txt`）：

| 版本 | 耗时 | 算力 | 有效读带宽 | 读放大 | 寄存器 |
|---|---|---|---|---|---|
| `naive` | 17.11 ms | 17.07 TFLOPS | **9035 GB/s** | 131072× | 64 |
| `head2` | **15.50 ms** | **18.84 TFLOPS** | 4987 GB/s | 65536× | 124 |
| `head4` | 19.86 ms | 14.71 TFLOPS | 1947 GB/s | 32768× | 225 |
| `smem` (HG4,BQ4) | 19.22 ms | 15.20 TFLOPS | 503 GB/s | 8192× | 190 |
| `smem8` (HG4,BQ8) | 17.91 ms | 16.31 TFLOPS | 270 GB/s | 4096× | 190 |

**反直觉的点来了**：读放大从 131072× 降到 4096×（32×），耗时几乎没变，甚至 `naive` 比 `head4/smem` 还快。为什么？

因为 `c_kv` 太小了：Sk=1024 时 `c_kv = 1024×512×2 = 1 MB`，Sk=4096 也只有 4.7 MB，**整个 KV 常驻 50 MB 的 L2**。所谓「读放大 131072×」读的都是 L2，不是 HBM。`ncu` 实测证实了这一点：

| 指标 | `naive` | `head2` |
|---|---|---|
| Compute (SM) Throughput | **83.47%** | 76.90% |
| L1/TEX Cache Throughput | 69.58% | 47.17% |
| L2 Cache Throughput | 23.70% | 7.55% |
| **DRAM Throughput** | **0.71%** | **0.78%** |
| Achieved Occupancy | 49.47% | 24.80% |
| Issued IPC | 3.35 | 3.09 |
| Registers / thread | 64 | 124 |

`DRAM Throughput` 0.7%——HBM 几乎在睡觉。瓶颈是 **FP32 FMA pipe + shuffle + `__expf` 的依赖链**，`ncu` 的 stall 指纹也印证了（`mla_smem` 上 `wait`=0.77、`not_selected`=0.57、`long_scoreboard`=0.50，FMA pipe 42.8%）。

这条「复用没收益」的结论对写 kernel 很有价值：**在 prefill 长度不大时，MLA 的 KV 住在 L2，省访存的努力全是白费；唯一的出路是换计算引擎。** 真正会变成访存受限的是极长上下文（Sk 大到 KV 冲出 L2）或 decode 阶段每步只算一个 query——那时 FlashMLA 才需要 3000 GB/s 的款。

> 顺带一提，`head8`（HG=8）虽然复用最高，但 `ptxas` 报了 **255 寄存器 + 328 B 栈溢出**（`ptxas_regs.out.txt`），实际比 `head2` 还慢——这是第 10/11 篇讲过的寄存器墙，这里以「复用换寄存器」的形式又出现了一次。

---

## 4. Tensor Core 版：把 MLA 拆成 QKᵀ / softmax / PV

既然瓶颈是算力，就请出第 13/14 篇的 `mma.m16n8k16 + ldmatrix + cp.async` GEMM。最直接的切法是把 MLA 拆成三个 kernel：

```
Qcat[H,Sq,576] = [ q_abs | q_rope ]        Kᵀ[576,Sk] = [ c_kvᵀ ; k_ropeᵀ ]
        │                                          │
        └────────────► S = QKᵀ  (mma) ◄────────────┘     ---- kernel 1
                          │
                     softmax(行方向)  -> P (bf16)          ---- kernel 2
                          │
        P[H,Sq,Sk] ──► O = P · c_kv  (mma) ──► O[H,Sq,512]  ---- kernel 3
                                     ▲
                                  c_kv[Sk,512]（所有 head 共享）
```

实现要点（`mla_attn.cu:410/441`）：

- 先把 `q_abs|q_rope` 拼成 `Qcat`、把 `c_kv|k_rope` 转置拼成 `Kᵀ`（用 host 端一次完成，避免额外 kernel）；
- 两个 GEMM 复用第 13 篇的 `gemm_mma_pipe`（BM=BN=128, BK=32, 256 线程），只是把 `blockIdx.z` 当作 head 做 batch，**B 矩阵（KV）所有 head 共享**——MQA 的结构优势在这里体现得淋漓尽致；
- softmax 逐行做（第 07 篇的套路），输出 bf16 的 P 喂给第二个 GEMM。

实测（`mla_attn.out.txt`）：

| kernel | 耗时 | 算力 | 备注 |
|---|---|---|---|
| `qk` (gemm) | 1.163 ms | 132.90 TFLOPS | L2 70.9% / Compute 23.2% |
| `softmax` | 0.476 ms | 1.69 TB/s | DRAM 50.4%，memory-bound |
| `pv` (gemm) | 0.919 ms | 149.55 TFLOPS | L2 63.7% / Compute 25.2% |
| **total** | **2.526 ms** | **115.64 TFLOPS（11.7%）** | vs `head2` **6.1×** |

`ncu` 显示两个 GEMM 又是**L2 瓶颈**（70.9% / 63.7%），Compute 只有 23~25%，occupancy 12.4%（164 寄存器）。原因和第 13 篇一样：这个 kernel 的算术强度随 K 变小而下降，`K=576`（QK）和 `K=Sk`（PV）都不大，L2 吞吐顶住了。想再快要么上更大 tile / `wgmma`，要么**别物化 S/P**。

### 物化 S/P 的代价

三 kernel 版的硬伤是 `S`（fp32）和 `P`（bf16）都要走一遍显存。Sk=Sq=S 时流量是 `H·S²·(4+2)·2` 字节（S 写+读、P 写+读）。扫一遍序列长度：

| S | `naive` | `head2` | `smem8` | `tc` 总 | `tc` 算力 | S/P 流量 | S/P 理论耗时 |
|---|---|---|---|---|---|---|---|
| 1024 | 17.11 ms | 15.50 ms | 17.91 ms | **2.53 ms** | 115.6 TFLOPS | 1.61 GB | 0.48 ms |
| 2048 | 67.49 ms | 60.58 ms | 69.45 ms | **9.41 ms** | 124.1 TFLOPS | 6.44 GB | 1.92 ms |
| 4096 | 268.0 ms | 239.7 ms | 273.8 ms | **36.58 ms** | 127.7 TFLOPS | 25.77 GB | 7.69 ms |

（原始输出 `mla_attn_s2048.out.txt` / `mla_attn_s4096.out.txt`。）

有意思的对照：

- 标量版随 S **线性**变慢（算力几乎不变，始终 17~20 TFLOPS），因为它是纯计算受限；
- TC 版随 S **超线性**变慢：算力慢升（115→128），但 S/P 物化流量按 `S²` 涨。到 S=4096，**25.77 GB 的 S/P 往返就要 7.69 ms，占总时间的 21%**；
- 也就是说，三 kernel 版在长上下文会被「写出 S、再读回来做 softmax、再写 P、再读回来做 PV」拖垮。**把 softmax 融进 QK 的 epilogue、把 PV 融进 P 的产生**（也就是 FlashAttention 那套 online softmax + 寄存器里做 P），是顺理成章的下一步——留到第十六篇。

---

## 5. 离 SOTA 还有多远

FlashMLA 公布的密集 MLA 性能（H800 SXM5，CUDA 12.8）：

- **compute-bound 配置：最高 660 TFLOPS**（decode，Sq=1，MQA，FP8 KV cache，bf16 计算）；
- **memory-bound 配置：最高 3000 GB/s**。

我们最好的 `tc` 版在 S=4096 是 **127.7 TFLOPS = 660 的 19.4%（约 5.2× 差距）**。口径差异要讲清楚：

| 维度 | FlashMLA dense decoding | 本篇 `tc` |
|---|---|---|
| 阶段 | decode（每步 Sq 小） | prefill（Sq 大） |
| 软件结构 | 单 kernel，online softmax 融合 | 三 kernel，物化 S/P |
| KV cache | FP8 + scale | bf16 |
| Tensor Core 指令 | 用满 Hopper（含 TMA/wgmma 级优化） | `mma.m16n8k16` |

即便有这些差异，差距的主因很清楚：**我们还没融合**。三 kernel 版的 FLOPs 利用率被 S/P 往返和 softmax 的额外 DRAM 流量稀释，而 FlashMLA 是「一个 kernel 从头算到尾、中间结果永不落显存」。

---

## 小结

- **MLA = 低秩 Q/KV + decoupled RoPE**；「吸收」后等价于一个 **MQA**：所有 query head 共享 `c_kv`（512 维）和 `k_rope`（64 维）。本篇用 DeepSeek-V3 的 576/512 口径，这也是 FlashMLA 的口径。
- **prefill 阶段的 MLA 不是访存受限，而是算力受限**：KV 只有 1~5 MB，常驻 L2（`ncu` 实测 DRAM Throughput 仅 0.7%）。三个标量实现都在 15~19 TFLOPS 打转，复用做多好都没用——瓶颈是 FP32 pipe + shuffle + `__expf`。
- **算术强度分析**：理想实现的 AI≈242 FLOP/byte，接近 bf16 TC 的 ridge（295）；但标量 FFMA 的 ridge 只有 20，天花板 <7%。**Tensor Core 是入场券。**
- **三 kernel Tensor Core 版 2.53 ms / 115.6 TFLOPS，比标量最好的 `head2` 快 6.1×**，达到 bf16 峰值的 11.7%、FlashMLA 660 TFLOPS 的 ~19%。
- **代价是物化 S/P**：S=4096 时 `S²` 的流量占总时间 21%，越长的上下文越亏。这直接指向下一篇：**融合**。
- 踩坑记录：`HG=8` 的寄存器复用策略触发 255 寄存器 + 栈溢出（第 10/11 篇的寄存器墙换了个马甲又来了）；smem 版即使做了 `+8` padding 仍有 `ldmatrix` 之外的 bank conflict（`l1tex` 报 6.8 亿 wavefront 过量），留给 16 的 smem 布局优化。

> 下一篇：CUDA 算子调优（十六）· MLA 注意力（二）——FlashMLA 式单 kernel：KV 分块 + online softmax 全融合、`wgmma` 与 split-KV，目标把这条 5× 的差距压到 2× 以内。
