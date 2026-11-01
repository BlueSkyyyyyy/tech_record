---
title: "Megatron 源码精读（二十四）：显存到底怎么算——精度、并行与激活的完整账本"
date: 2026-09-07
draft: false
tags: ["megatron-lm", "系列", "训练框架", "显存估算", "混合精度"]
categories: ["训练框架"]
weight: 24
series: ["megatron-code"]
---

前 23 篇把并行、重计算、优化器、checkpoint 逐个拆开了。这一篇把它们合成一本账：**给定模型配置和并行配置，一张卡要多少显存？**

**一句话结论**：训练显存 = 参数 + 梯度 + 优化器状态 + 激活 + 余量。前三项按「并行度」打折，单价由精度决定；第四项按 `batch × seq × 层数` 增长，只能靠 FlashAttention、重计算、SP/CP 压。**算出 16 GB 就敢开 16 GB 的卡，必然 OOM——余量要单独记一笔。**

---

## 1. 记账单位：每个元素几字节

所有显存最终都是 `元素数 × 字节数`，所以先把单价定下来。

| 精度 | 字节/元素 | 典型用途 |
|---|---|---|
| FP32 | 4 | FP32 master weight、Adam 的一阶/二阶矩 |
| BF16 / FP16 | 2 | 前向/反向的权重、激活、梯度 |
| FP8 | 1 | TE 的 FP8 权重与激活 |
| FP4 | 0.5 | TE 的 NVFP4（实验） |

混合精度训练的关键在于**同一份参数在不同位置有不同副本**：

- 计算用 BF16 权重（2 B）；
- 更新用 FP32 master weight（4 B），否则 `w += lr * update` 在 BF16 下会丢精度；
- Adam 的 `exp_avg`、`exp_avg_sq` 必须 FP32（8 B）——二阶矩对精度尤其敏感。

于是「每个参数」的稳态开销约为 $2+4+8=14$ B，再加梯度 2 B，就是常说的 **16 B/参数**。

## 2. 参数、梯度、优化器状态

### 2.1 参数量怎么估

decoder-only Transformer 的参数量可拆成「每层 + 词表」：

$$
P \approx L \cdot 12h^2 + 2Vh
$$

每层 $12h^2$ 来自 attention 的四个投影 $4h^2$ 与 MLP 的 $8h^2$（FFN 扩到 $4h$ 再收回来）。$2Vh$ 是 embedding 与输出头（未共享时）。

代入 Llama-2 7B（$h=4096$、$L=32$、$V=32000$）：$32 \times 12 \times 4096^2 \approx 6.4\text{B}$，加 $2 \times 32000 \times 4096 \approx 0.26\text{B}$，合计约 6.7B，与公开数字一致。

> GQA 会显著减少 KV 投影的参数与 KV cache，但不改变 $4h^2$ 这个量级；做精细估算时把 attention 项换成 $(2+2g)h^2$（$g$ 为 KV head 比例）即可。

### 2.2 并行如何打折

不同并行切的东西不同，**不能用一个 `world_size` 一除了事**：

| 项 | 无 ZeRO | ZeRO-1（DistOpt） | ZeRO-2 | ZeRO-3 / FSDP |
|---|---|---|---|---|
| BF16 权重 | $2P$ | $2P$ | $2P$ | $2P/dp$ |
| FP32 master | $4P$ | $4P/dp$ | $4P/dp$ | $4P/dp$ |
| 梯度 | $2P$ | $2P$ | $2P/dp$ | $2P/dp$ |
| Adam 两矩 | $8P$ | $8P/dp$ | $8P/dp$ | $8P/dp$ |

（表中省略了 $/\mathrm{tp}/\mathrm{pp}$，所有项都要再除以 TP 与 PP。）

两个容易记错的点：

1. **ZeRO-1 只切优化器状态（含 master），不切参数**。Megatron 的 `param_data` 仍是完整 buffer，每个 rank 都分配；DistOpt 只是让每个 rank 只更新、只保存自己那 1/dp 的 state 与参数切片，前向前再 all-gather 出完整权重（第 8、11 篇）。
2. **梯度 buffer 按全量分配**。reduce-scatter 之后只有本地 1/dp 段有效，但显存是按 buffer 大小算的——除非用 FSDP 这类真正按 shard 分配的方案（第 8 篇）。

## 3. 激活：唯一随 batch 和序列增长的项

### 3.1 每层激活的经典估算

对 16-bit、GPT 风格的层，Korthikanti 等人给出的每层激活字节数为：

$$
M_{\text{act per layer}} \approx s \cdot b \cdot h \cdot \left(34 + 5\,\frac{a\,s}{h}\right)
$$

其中 $s$ 是序列长、$b$ 是 microbatch、$h$ 是 hidden、$a$ 是 head 数。

两项的含义完全不同：

- $34sbh$：LayerNorm 输入输出、QKV、attention 输出、MLP hidden 等「每 token 一份」的中间量；
- $5as^2b$：attention 分数矩阵 $QK^\top$ 的 $O(s^2)$ 项。

**FlashAttention 消掉的就是第二项**。以上面的 7B 为例（$s=2048$、$a=32$、$h=4096$）：

- 不融合 attention：每 token 约 $34+80=114$ 份，32 层合计约 30 GB；
- 用 FlashAttention：只剩 $34$ 份，约 9 GB。

**所以长序列下第一优先是 FlashAttention，不是重计算。**

### 3.2 三个减激活手段，别记混

| 手段 | 省多少 | 代价 |
|---|---|---|
| FlashAttention | 去掉 $O(s^2)$ 项 | 几乎无（核函数更快） |
| 激活重计算 | 见下 | 多一遍前向，FLOPs 约 +33% |
| SP / CP | 激活再 $/\mathrm{tp}$ 或 $/\mathrm{cp}$ | 增加通信 |

重计算要单独说清，因为常见说法互相矛盾：

- **全量重计算**（每层都 checkpoint）：只存每层输入，激活从 $34sbhL$ 降到约 $2sbhL$，**约 1/17**，但仍是 $O(L)$；
- **均匀分组**（每 $k$ 层存一次）：显存约 $O(L/k + k)$，取 $k=\sqrt{L}$ 时最优，才是 $O(\sqrt{L})$。

也就是说，「$O(\sqrt L)$」属于**每 $\sqrt L$ 层 checkpoint 一次**的策略；「full checkpoint」是**同阶但常数小一个量级**。两者不是一回事。

FLOPs 的账也要改：全量重计算相当于多做一遍前向。训练本身是 $2P$（前向）$+4P$（反向），多做一遍前向是 $+2P/6P \approx +33\%$，不是 +100%。

## 4. 汇总：一个能对得上的例子

单卡显存：

$$
M_{\text{total}} = M_{\text{weight}} + M_{\text{master}} + M_{\text{grad}} + M_{\text{adam}} + M_{\text{act}} + M_{\text{reserve}}
$$

**例：7B，BF16 + Adam，tp=2、pp=2、dp=4（16 卡），开 ZeRO-1，$s=2048$、$b=1$、FlashAttention 开、SP 开。**

| 项 | 计算 | 结果 |
|---|---|---|
| BF16 权重 | $2P/(\mathrm{tp}\cdot\mathrm{pp})$ | 3.5 GB |
| FP32 master | $4P/(\mathrm{tp}\cdot\mathrm{pp}\cdot dp)$ | 1.75 GB |
| 梯度 buffer | $2P/(\mathrm{tp}\cdot\mathrm{pp})$（全量分配） | 3.5 GB |
| Adam 两矩 | $8P/(\mathrm{tp}\cdot\mathrm{pp}\cdot dp)$ | 3.5 GB |
| 激活 | $34sbhL/\mathrm{tp}$ | 4.6 GB |
| 小计 | | 16.9 GB |
| 余量 15% | | 2.5 GB |
| **合计** | | **约 19.4 GB** |

这个结果能解释很多现象：7B 在 16 张 40 GB 卡上很宽松，在 16 张 24 GB 卡上就要靠重计算（激活 4.6 GB → 0.27 GB）或更大 TP 才塞得下。

### 4.1 余量里到底有什么

- **ZeRO-3 的参数 all-gather 缓冲**：前向时短暂出现「完整一层权重」；
- **PyTorch 分配器碎片**：小张量反复分配，实际可用率常掉 5%~10%；
- **CUDA context / cuBLAS workspace / NCCL buffer**：几 GB，和模型规模弱相关；
- **checkpoint 的 staging buffer**：保存瞬间的峰值。

经验上留 10%~20%。「算出来刚好 79 GB，卡是 80 GB」是最典型的 OOM 姿势。

## 5. 精度只改单价，不改结构

| 策略 | 权重 | 梯度 | 优化器 | 激活 |
|---|---|---|---|---|
| FP32 | 4 | 4 | 8 | 4 |
| BF16 混合精度 | 2 + 4 master | 2 | 8 | 2 |
| FP8（TE） | 1 + 4 master | 1~2 | 8 | 1 |
| FP4（TE） | 0.5 + 4 master | ~1 | 8 | 0.5~1 |

两个结论：

1. **优化器状态几乎不随计算精度下降**。FP8 训练里 Adam 仍是 FP32，8 B/参数是「顽固成本」。
2. **FP8 主要省激活和带宽，不省常驻的 master + 矩**。所以 FP8 对长序列、大 batch 的收益远大于对小模型。

反过来说，推理显存 = 权重 + KV cache，没有 master 和矩——拿训练账本去估推理会高估一个数量级。

## 6. 小结

- **单价表**：BF16 混合精度下「权重 2 + master 4 + 梯度 2 + 矩 8 = 16 B/参数」；FP8 只改前两项与激活。
- **打折规则**：TP/PP 切所有项；ZeRO-1 只切 state（含 master），ZeRO-2 再切梯度，ZeRO-3 才切权重。
- **激活先看 attention**：$O(s^2)$ 的分数矩阵靠 FlashAttention 消掉，其次才是重计算、SP/CP。
- **重计算量级**：全量重计算 ≈ 1/17 常数、仍是 $O(L)$；$O(\sqrt L)$ 属于每 $\sqrt L$ 层存一次的分组策略；FLOPs 代价约 +33%。
- **永远留 15% 余量**：碎片、workspace、all-gather 缓冲都在这一笔里。

---

**版本提示（2026-10）**：上游新增的 GTP 会再切一刀权重——每 rank 只常驻 $1/(\mathrm{TP}\times \mathrm{GTP\_remat})$ 的权重、梯度与优化器状态，GEMM 前按权重 all-gather（第 26 篇）。把上表的「权重/梯度/master/矩」再除以 `GTP_remat` 即可；激活账不变。MFSDP v2 则把优化器 state 的分配改成真正的 shard，第 2.2 节「buffer 全量分配」那条对它不适用。

**行号基线**：本篇为估算方法篇，不引用行号；机制细节回看第 4、6、8、11 篇（commit `f713506ce`）。
