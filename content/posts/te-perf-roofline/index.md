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

| 常量 | 值 |
|---|---|
| FP16/BF16 tensor-core peak | ≈ 989.4 TFLOPS（132 SM × 1.980 GHz，dense，不开 sparsity） |
| FP32（CUDA-core FMA）peak | ≈ 66.9 TFLOPS |
| HBM3 带宽 | ≈ 3.35 TB/s |
| **ridge（FP16/BF16）** | **≈ 295 FLOP/byte** |
| ridge（FP32） | ≈ 20 FLOP/byte |

也就是说：AI 低于 295 FLOP/byte 的 FP16 算子一定是**内存受限**，理论上限就是 3.35 TB/s 那条斜线；AI 越过 295 之后才是**计算受限**。这个拐点非常陡（989.4 T / 3.35 T ≈ 295），意味着「是不是内存受限」这一个判断几乎决定了优化方向。

本次测的算子照 AI 从小到大排，正好横跨拐点两侧：

- **rmsnorm 三兄弟**（fwd / bwd / bwd_add）：AI 只有 2–4 FLOP/byte（每个元素读一次、写一次，做几个乘加），远低于拐点 → 纯内存受限；
- **fused_attn**：AI = `4·s²·h·d` 的量级（fwd 约 `4s` 每元素），随 seqlen 增长从 ~256 一路涨到 4096 FLOP/byte → 初始内存受限，seqlen 一大就翻进计算受限区。

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

**超过 2/3 的测时是启动开销**。`(128,512)` 一共才 64 万个元素、160 KB 数据，真正搬这份数据连 1 µs 都用不到——所以这类小 shape 测出来的「GB/s」奇低（29 GB/s 只有峰值带宽的 0.9%），不是 kernel 慢，是**根本没来得及跑满**。这也是为什么下面所有小 shape 的 `%BW` 数字都不具参考意义，要看大 shape 才收敛。

## 3. 内存受限区：rmsnorm 三兄弟

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
3. **grid/CTA 调度的尾浪效应**：132 SM 分 32768 行，行粒度太小时 wave 数多、尾部不均衡。

单看结论：**rmsnorm 这类内存受限算子，AI 定了上限之后，剩下的差距全是「搬字节的效率」，不是「算得快不快」**。

### 3.2 rmsnorm_bwd：同样的内存受限，带宽反馈更差

bwd 的 AI = 0.33（fp32）/ 0.67（fp16），比 fwd 略高（因为要同时算 `dx` 和 `dw`），但依旧远在拐点左侧。同样的 `(32768, 2048)`：

| kernel | fp16 GB/s | %BW |
|---|---|---|
| rmsnorm_fwd | 2458 | 73.4% |
| rmsnorm_bwd | 2551 | 76.1% |

bwd 要额外读 `rsigma`、额外写 `dw`，但 `dw` 是沿列 reduce 一条向量、摊到每行就很小。整体带宽反而略高，是因为 bwd 每元素多做了几个乘加（AI 0.67 对 0.50），同样的带宽下「有效吞吐」更高。

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
2. **但等效带宽反而更高**（76% → 83%），因为多读的 `add` 是顺序流式访问，把 HBM 吞吐推得更满。

融合的价值要跟「不融合」比：不融合时你得跑 `rmsnorm_bwd`（158 µs）+ add 反向（一次全量读写，约再一个 fwd 量级）+ 累加，总耗时显著超过 194 µs。所以 **`rmsnorm_bwd_add` 的意义不是更快地做同一个数学，而是把原本 2–3 个 kernel 串起来的内存流量压进 1 个 kernel**——这正是 Megatron 里 `LayerNormMLP` 之类把残差融合进 norm 反向的收益来源。

## 4. 计算受限区：fused_attn 穿越拐点

attention 的 AI 随 seqlen 线性增长（fwd 每元素约 `4s`，bwd 约 `8s`）。我们用 seqlen 从 512 一路拉到 8192，观察它从内存受限翻进计算受限：

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

1. **AI 穿越拐点**：小 seqlen 时 AI=256、落在斜线上（内存受限，`%TC` 才 5%）；seqlen 一过 1024，AI 超过 295，翻上平顶，`%TC` 逼近 95%+——完全符合 roofline 的分段预测。
2. **batch 比 seqlen 更早把利用率拉满**：`(1,2048)`→`(2,2048)`→`(4,2048)`→`(8,2048)` 的 `%TC` 从 43% 单调爬到 95%，因为 batch 增长时 AI 不变（仍是 1024），但**并行度和占用率**上去了，尾浪被填平。
3. **`(4,8192)` 出现 112.9% > 100%**：这暴露了 TFLOPS 口径的问题——我按 `4·b·s·h·d`（几何定义的 FLOP，没算 softmax、dropout 的额外算力，也没扣 causal mask 浪费的那一半）,实际 kernel 干的活更多。同时 8192 seq 下中间矩阵放不进 L2，读写压力反噬。所以这里「超 100%」不是真超额，是**口径偏乐观 + 大 seqlen 记忆墙**共同造成的。

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
- **内存受限区未饱和**：rmsnorm 最大 ~73–83% BW，来自读放大、尾浪、以及 bwd_add 多读一路 add 带来的流量；
- **计算受限区口径偏差 + 记忆墙**：`%TC` 的 `>100%` 是几何 FLOP 口径乐观所致；大 seqlen 的中间矩阵溢出 L2 后，实测会被带宽反噬。

**下一步**：把这张 roofline 图、尤其 `rmsnorm_bwd_add` 这类融合核的「带宽高效但流量更大」结论，进一步落到 TE 的真实模块（`LayerNormMLP`、`TransformerLayer` 的 memory 瓶颈），并纳入 PyTorch 的 CUDA Graph / greedy 流式复用来摊薄 launch overhead。

---

*环境：NVIDIA H100 SXM 80GB（132 SM，CC 9.0），transformer_engine 2.14.0，测试时 GPU 仅运行 benchmark。完整数据见 [`code/te-perf/perf.log`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/te-perf) 与 `te_perf.csv`。*
