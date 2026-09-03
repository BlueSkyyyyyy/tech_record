---
title: "H100 上 Transformer Engine 算子的性能与 Roofline 对比"
date: 2026-09-01
draft: false
tags: ["transformer-engine", "roofline", "性能分析", "kernel", "H100", "系列"]
categories: ["算子开发"]
series: ["te-perf"]
---

Transformer Engine（TE）是 NVIDIA 维护的一套训练加速库，训练框架（Megatron 等）会把 LayerNorm、attention 这些算子的 CUDA kernel 外包给它。但「外包出去」不等于「算得满」——TE 的算子到底吃到了 H100 的几成带宽/算力，和 roofline 模型差多远，只有实测才知道。本文在 H100 SXM（132 SM，CC 9.0）上对 TE 的 `rmsnorm` / `rmsnorm_bwd` / `rmsnorm_bwd_add`（残差融合反向）以及 `fused_attn_fwd` / `fused_attn_bwd` 做了一组微基准，把实测点放到 roofline 模型上看差距，并逐一拆解差距来源。

配套代码在 [`code/te-perf/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/te-perf)（`bench_te.py` 直接调 TE 原生 kernel，CUDA event 计时）。

---

## 1. 背景：roofline 模型与本次的「天花板」

Roofline 把一个算子归结到两个维度：**算术强度 AI（每搬运 1 字节能做的 FLOP 数）** 和 **实测性能**。机器能给的性能上限是一个「折线」：

- 斜率部分（低 AI）：内存受限，性能上限 = HBM 带宽 × AI；
- 平顶部分（高 AI）：计算受限，性能上限 = 峰值算力。

两段的交点是 **ridge point（拐点）**，由峰值算力 ÷ 峰值带宽得到。本次在 H100 上取的常量：

| 常量 | 值 | 谁用它 |
|---|---|---|
| FP16/BF16 tensor-core peak | ≈ 989.4 TFLOPS（132 SM × 1.980 GHz，dense，不开 sparsity） | MMA 类算子（fused_attn 的 QKᵀ/PV） |
| FP32（CUDA-core FMA）peak | ≈ 66.9 TFLOPS | elementwise/scan 类算子（rmsnorm 三兄弟） |
| HBM3 带宽 | ≈ 3.35 TB/s | 两类的内存天花板 |
| **ridge（FP16/BF16）** | **≈ 295 FLOP/byte** | tensor core 的拐点 |
| ridge（FP32） | ≈ 20 FLOP/byte | CUDA core 的拐点 |

这里要先分清两类硬件的**天花板**：FP16/BF16 矩阵法（`QKᵀ`、`PV`、GEMM）走 **tensor core**，峰值 989.4 TFLOPS；而 rmsnorm 这种逐元素扫描 + 归一的算子走的是 **CUDA core 的标量 FMA**，峰值只有 66.9 TFLOPS，对应的 ridge 也只有 20 FLOP/byte。

也就是说：AI 低于 295 FLOP/byte 的 FP16 tensor-core 算子一定是**内存受限**，理论上限就是 3.35 TB/s 那条斜线；AI 越过 295 之后才是**计算受限**。这个拐点非常陡（989.4 T / 3.35 T ≈ 295），意味着「是不是内存受限」这一个判断几乎决定了优化方向。而 CUDA core 那条线虽然 ridge 只有 20，对 rmsnorm 却**无关紧要**——因为它的 AI 只有 0.25–0.67，连 20 都够不到，算力天花板根本摸不着（详见 §3 开头）。

本次测的算子照 AI 从小到大排，正好横跨拐点两侧：

- **rmsnorm 三兄弟**（fwd / bwd / bwd_add）：AI 只有 0.25–0.67 FLOP/byte（每个元素读一次、写一次，外加几个乘加，FLOP 数远小于搬运字节数），远低于拐点 → 纯内存受限；
- **fused_attn**：AI 随 seqlen 线性增长（fwd 总共 `4·b·s²·h·d` FLOP，fp16 下 AI ≈ `s/2` FLOP/byte，即 seqlen=512 时 256、8192 时 4096）→ 初始内存受限，seqlen 一大就翻进计算受限区。

## 2. 一个必须先扣掉的坑：launch overhead

每个 case 的 `time(us)` 是 CUDA event 记的「一次调用」总耗时，里面混了两个东西：**内核真正执行时间** + **固定的内核启动开销**。小 shape 时后者会完全淹没前者。

我在脚本开头跑了一个最小 `fill_` kernel 测这个固定开销：

```
minimal-kernel round-trip latency = 12.0 us/call
```

12 µs 是什么概念？看 rmsnorm 最小 shape `(128, 512)` 的实测：

| shape | dtype | time(us) | 扣掉 12 µs 后的纯内核时间 |
|---|---|---|---|
| (128, 512) | fp32 | 17.9 | ≈ 5.9 µs |
| (128, 512) | bf16/fp16 | 17.7 | ≈ 5.7 µs |

**超过 2/3 的测时是启动开销**。`(128,512)` 一共才 65536（6.5 万）个元素，fp32 下 256 KB、bf16/fp16 下 128 KB，真正搬这份数据连 1 µs 都用不到——所以这类小 shape 测出来的「GB/s」奇低（fp32 是 29 GB/s、只有峰值带宽的 0.9%，fp16 更只到 14.8 GB/s / 0.4%），不是 kernel 慢，是**根本没来得及跑满**。这也是为什么下面所有小 shape 的 `%BW` 数字都不具参考意义，要看大 shape 才收敛。

## 3. 内存受限区：rmsnorm 三兄弟

三个 rmsnorm kernel 都是逐元素乘加 + 沿行归一的 scan，走的是 **CUDA core 标量 FMA**（不走 tensor core，也没有能让 tensor core 发挥的矩阵结构）。它们的 AI 只有 0.25–0.67 FLOP/byte，靠上那张 66.9 TFLOPS / ridge=20 的「CUDA-core roofline」：即便把算力提升到 CUDA core 的 100%，性能上限也是 `3.35 TB/s × AI`，约 0.84–2.2 TB/s 这条斜线——**算力天花板根本碰不到，瓶颈从一开始就是带宽**。所以下面 rmsnorm 全部拿 `%BW`（带宽占比）说话，算力占比不提也罢。

### 3.1 rmsnorm_fwd

AI = 0.25（fp32 每个元素 3 次读+写算下来 2 FLOP / 8 byte）或 0.50（fp16），整条折线远在拐点左侧。所以理论上限就是斜线 `3.35 TB/s × AI`。

看实测带宽随 shape 的收敛（fp16，避开 fp32 的算力陷阱）：

| shape | time(us) | GB/s | %BW |
|---|---|---|---|
| (2048, 2048) | 18.2 | 920 | 27.5% |
| (4096, 2048) | 19.1 | 1754 | 52.3% |
| (8192, 4096) | 62.2 | 2158 | 64.4% |
| (32768, 2048) | 109.2 | 2458 | 73.4% |

shape 一大，`%BW` 从 27% 一路爬到 73%，但**没到 100%**。rmsnorm 是纯 scan + 归一化，理论上只受 HBM 限制，为什么最大只到 ~73%？

拆开看几个来源：

1. **launch overhead 的残差**：12 µs 对 `(32768,2048)` 的 109 µs 已经只占 ~11%，但对中小 shape 影响仍在；
2. **kernel 本身没做极致访存优化**：rmsnorm 要先把整行读进来算 `rsigma`，normalize 时可能再读一遍（取决于实现是否在寄存器/SMEM 里 hold 住），读放大就吃掉了带宽；
3. **fp16 固定行方向的网格划分**：TE 的 rmsnorm 按行划分 CTA，`(32768,2048)` 每个 SM 分 256 行、每行 2048 个元素恰好装满一个 warp 分段，读写仍有对齐/分摊损耗，离纯流式的 3.35 TB/s 还有一段距离。

单看结论：**rmsnorm 这类内存受限算子，AI 定了上限之后，剩下的差距全是「搬字节的效率」，不是「算得快不快」**。

### 3.2 rmsnorm_bwd：同样内存受限，AI 略高、带宽略好

bwd 的 AI = 0.33（fp32）/ 0.67（fp16），比 fwd 略高（因为要多算 `dx` 与 `dw` 两路），但依旧远在拐点左侧。同样的 `(32768, 2048)`：

| kernel | fp16 GB/s | %BW |
|---|---|---|
| rmsnorm_fwd | 2458 | 73.4% |
| rmsnorm_bwd | 2551 | 76.1% |

bwd 要额外读 `rsigma`、额外写 `dw`，但 `dw` 是沿列 reduce 一条向量、`rsigma` 每行一个标量，摊到每行就很小，所以访存总量只比 fwd 多了约一半（AI 0.67 对 0.50 也印证了这点）。`%BW` 反比 fwd 略高 2.7 个百分点，一是因为 AI 略高、同样带宽下「有效算力」更高，二是扣掉 12 µs 固定开销后，bwd 纯内核时间（~146 µs）比 fwd（~97 µs）长，launch overhead 的相对拖累更小。

### 3.3 rmsnorm_bwd_add：融合残差反向量，AI 回落但带宽更高

这是本篇真正想测的新 case。forward 是 `z = rmsnorm(x) + add`（TE 里 add 是独立 elementwise，**不融合进 fwd kernel**），反向时普通写法是：

```
rmsnorm_bwd(x, dy, rsigma, gamma)   -> dx1, dw     # 一次全量读写
add 逐元素反向                               -> 再读一遍 add，写 dx2
dx = dx1 + dx2                                     # 又一次读写
```

而 TE 提供了一个真正的融合核 `rmsnorm_bwd_add(dz, x, add, rsigma, gamma, ...)`，**出一个 kernel 同时算 `dx`（含 add 的梯度）和 `dw`**，省掉了「单独 add 反向 + 二次累加」的那一整趟全量读写。

实测对比 `(32768, 2048)` fp16：

| kernel | time(us) | GB/s | %BW |
|---|---|---|---|
| rmsnorm_bwd | 157.9 | 2551 | 76.1% |
| rmsnorm_bwd_add | 194.0 | 2767 | 82.6% |

两点值得注意：

1. **绝对时间变长了**（158 → 194 µs），因为它确实多读了一个 `add` 张量（AI 从 0.67 掉回 0.50，同样的元素数、多读一路数据）；
2. **但等效带宽反而更高**（76% → 83%）。原因不是它「访问更聪明」，而是测出来的 `GB/s` 本来就混进了 12 µs 的固定开销：`rmsnorm_bwd` 纯内核 ~146 µs、`rmsnorm_bwd_add` 纯内核 ~182 µs，前者被 launch overhead 拉低的相对幅度更大。同时多读的一路 `add` 让稳态流式访问占比更高、访存更「顺」，所以跑到这个量级的 shape，`%BW` 比单纯的 bwd 更贴近峰值。这也提醒：**用含 launch 的 `/ (t)` 去比带宽，对小 shape 不公平，应该扣掉 12 µs 再算**。

融合的价值要跟「不融合」比：不融合时你得跑 `rmsnorm_bwd`（158 µs）+ add 反向（一次全量读写，约再一个 fwd 量级）+ 累加，总耗时显著超过 194 µs。所以 **`rmsnorm_bwd_add` 的意义不是更快地做同一个数学，而是把原本 2–3 个 kernel 串起来的内存流量压进 1 个 kernel**——这正是 Megatron 里 `LayerNormMLP` 之类把残差融合进 norm 反向的收益来源。

## 4. 计算受限区：fused_attn 穿越拐点

attention 的 AI 随 seqlen 线性增长（fwd 总共 `4·b·s²·h·d` FLOP，按「每元素」即 AI = `s/es`，fp16 下 = `s/2`；bwd 约 2 倍）。我们用 seqlen 从 512 一路拉到 8192，观察它从内存受限翻进计算受限：

| (batch, seq, head, dim) | fwd TFLOPS | %TC |
|---|---|---|
| (1, 512, 16, 128) | 50.4 | 5.1% |
| (1, 1024, 16, 128) | 177 | 17.9% |
| (1, 2048, 16, 128) | 423 | 42.8% |
| (2, 2048, 16, 128) | 614 | 62.0% |
| (4, 2048, 16, 128) | 790 | 79.8% |
| (8, 2048, 16, 128) | 943 | 95.3% |
| (1, 4096, 16, 128) | 768 | 77.6% |
| (2, 4096, 16, 128) | 953 | 96.4% |
| (4, 8192, 16, 128) | 1117 | **112.9%** |

三个观察：

1. **AI 穿越拐点**：小 seqlen 时 AI=256（< 295）、落在斜线上（内存受限，`%TC` 才 5%）；seqlen 一过 1024，AI 超过 295，翻上平顶，`%TC` 逼近 95%+——完全符合 roofline 的分段预测。
2. **batch 与 seqlen 的作用不同**：seqlen 涨会同时推高 AI（翻倍）和总 FLOP；而 batch 涨只增加并行度、AI 不变（仍是 1024）。`(1,2048)`→`(2,2048)`→`(4,2048)`→`(8,2048)` 的 `%TC` 从 43% 单调爬到 95%，纯靠**占用率**上来、尾浪被填平。对比 `(1,4096)`（77.6%）比 `(2,2048)`（62.0%）高，正是 seqlen 拉高 AI 的结果。
3. **`(4,8192)` 出现 112.9% > 100%**：这暴露了 TFLOPS 口径的问题。我按 `4·b·s·h·d` 计几何 FLOP，但对 causal attention 而言只算**下三角** `s(s+1)/2` 个位置是有用的，掩掉的上三角接近一半——也就是说几何口径把 FLOP **高估了约 2 倍**；把这个修正回来，`(4,8192)` 的真实利用率其实落在 ~56% 上下，并不是真超额。而之所以大 seqlen 反而缩水，是 8192 seq 下中间 S/P 矩阵（`b·s·h·s` 个 fp16 ≈ 4×8192×16×8192×2 ≈ 8.6 GB）远超 L2（50 MB），读写压力反噬。所以「超 100%」是**口径偏乐观 + 大 seqlen 记忆墙**共同造成的假象。

### 4.1 fused_attn_bwd：约 2× fwd 的算力，但 `%TC` 整体更低

bwd 每元素 FLOP 约是 fwd 的 2 倍，但实测 `%TC` 系统性低于 fwd：

| (batch, seq, head, dim) | bwd TFLOPS | %TC |
|---|---|---|
| (8, 2048, 16, 128) | 603 | 61.0% |
| (2, 4096, 16, 128) | 668 | 67.5% |
| (4, 8192, 16, 128) | 793 | 80.1% |

bwd 打不满的原因：forward 只写一次 O，backward 要写 `dQ/dK/dV` 三份、还要重算或读回 attention matrix，**访存量是 fwd 的数倍**，AI 实际更低，所以即便算力需求翻倍也没把 tensor-core 喂满。此外 backward 需要 recompute/软归一化中间量（`dS` 的 softmax 梯度），`NVTE_FUSED_ATTN_USE_FAv2_BWD` 默认关掉，走的是更保守的路径。

## 5. 小结：与 roofline 差距的三个来源

把整张图收敛成一句话：**rmsnorm 弟兄差在「搬字节的效率」，attn 差在「口径和记忆墙」，小 shape 差在「启动开销」**。具体：

- **launch overhead（~12 µs/call）**：小 shape 的绝对误差来源，测 kernel 前必须先扣；
- **内存受限区未饱和**：rmsnorm 最大 ~73–83% BW，来自读放大、行粒度网格划分的访存分摊，以及 bwd_add 多读一路 add 带来的流量；
- **计算受限区口径偏差 + 记忆墙**：`%TC` 的 `>100%` 是几何 FLOP 没扣 causal mask 下三角约一半的浪费、导致 FLOP 高估近 2 倍所致；大 seqlen 的中间 S/P 矩阵溢出 L2 后，实测会被带宽反噬。

**下一步**：把这张 roofline 图、尤其 `rmsnorm_bwd_add` 这类融合核的「带宽高效但流量更大」结论，进一步落到 TE 的真实模块（`LayerNormMLP`、`TransformerLayer` 的 memory 瓶颈），并纳入 PyTorch 的 CUDA Graph / greedy 流式复用来摊薄 launch overhead。

---

*环境：NVIDIA H100 SXM 80GB（132 SM，CC 9.0），transformer_engine 2.14.0，测试时 GPU 仅运行 benchmark。完整数据见 [`code/te-perf/perf.log`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/te-perf) 与 `te_perf.csv`。*
