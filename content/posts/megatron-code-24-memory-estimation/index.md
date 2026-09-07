---
title: "Megatron 源码精读（二十四）：显存到底怎么算——精度、并行与激活的完整账本"
date: 2026-09-07
draft: false
tags: ["megatron-lm", "系列", "训练框架", "显存估算", "混合精度"]
categories: ["训练框架"]
weight: 24
series: ["megatron-code"]
---

前 23 篇把注意力、SSM、并行、checkpoint、recompute、优化器都拆开讲了一遍，但有一个「工程师每天都要做、却很少被当成一个独立主题」的问题一直没有正面回答：**训练一个大模型，显存到底需要多少，怎么算？**

这一篇把所有散落的知识串成一本账：给定模型配置（层数、hidden、head 数、序列长度、batch、精度）和并行配置（TP/PP/DP/CP、ZeRO），如何一步步把「参数 + 梯度 + 优化器状态 + 激活 + 碎片/通信缓冲区」五块显存加起来，得到最终估算值。

它的价值不在「给一个公式背」，而在于用一个统一的**字节数记账法**，把「混合精度」「ZeRO 分片」「激活重计算」「序列并行/序列长度」这些前面讲过的机制，各自的显存影响**定量化**。读完这篇，你应该能独立推算出「模型能不能塞进 N 张卡」以及「瓶颈卡在哪一块」。

---

## 1. 先建立「记账单位」：每个元素占几字节

所有显存量最终都等于 `元素个数 × 每元素字节数`。因此精度决定了每一笔账的**单价**，这是全篇的基石。

| 精度 | 每元素字节 | 用途 |
|---|---|---|
| FP32 | 4 B | master 权重、Adam 的一阶/二阶矩 |
| BF16 / FP16 | 2 B | 前向/反向的**计算存储**（weights/activations/gradients） |
| FP8 | 1 B | Transformer Engine 下的前向激活/部分张量 |
| FP4 | 0.5 B | TE 的进一步压缩（少见） |

**混合精度（AMP）训练**之所以省显存，核心就在于把「存下来占大头的东西」降位：

- 前向/反向里流动的**权重、激活、梯度**用 BF16（2B）做矩阵乘，只有最重的那一步才 cast 回 FP32。
- 但**不能全用 BF16**：权重更新需要 FP32 精度，所以额外保留一份 **FP32 master weight（4B）**。
- 优化器（Adam/AdamW）的两个矩 `exp_avg`、`exp_avg_sq` 必须用 **FP32（4B）**，否则二阶矩 `v` 在 BF16 下精度不够，训练会发散。

这就是混精的账本：**权重 ×2B（BF16 副本）+ 权重 ×4B（FP32 master）+ 梯度 ×2B（BF16 累加前）+ 优化器 ×8B（两个 FP32 矩）**。后面每块展开时都套这个单价。

---

## 2. 大头之一：模型参数

一个 decoder-only Transformer（GPT 系）的参数量 $P$ 可拆成三部分：embedding、每层（attention + MLP）、输出头（通常与 embedding 共享）。

设 $h$ 为 hidden size，$V$ 为词表大小，$L$ 为层数，则每层：

- Attention：$4h^2$（Q/K/V/O 四个投影）
- MLP：$2 \times 4h^2 = 8h^2$（两个 FFN，扩到 $4h$ 再收回来）
- 每层合计约 $12h^2$

整体（忽略 bias/LayerNorm 的 $O(h)$ 小项，忽略 embedding/head 的 $O(Vh)$，或按需补上）：

$$
P \approx L \cdot 12h^2 + 2Vh
$$

例如 Llama-2 7B：$h=4096, L=32, V=32000$，代入得 $32\times12\times4096^2 \approx 6.4\mathrm{B}$，加 $2\times32000\times4096 \approx 0.26\mathrm{B}$，总计约 6.7B，与公开的 6.7B 吻合。这个式子本身不复杂，关键在于下面每一步**乘以精度单价和除以并行度**。

---

## 3. 并行如何「打折」：TP / PP / ZeRO 分别省什么

显存估算里，并行度不是简单除以总和，而是**每块显存各归各管**：

| 并行维度 | 谁被切 | 谁不被切 |
|---|---|---|
| **TP（张量并行）** | 权重、梯度、优化器状态、部分激活 | 数据、embedding（不切时） |
| **PP（流水线并行）** | 权重/梯度/优化器按层分段到不同 rank | 每 rank 只存自己那几层 |
| **DP / ZeRO（数据并行）** | 优化器状态（ZeRO-1）、梯度（Stage-2）、权重（Stage-3） | 数据、激活 |
| **CP / SP（序列并行）** | 激活（沿 seq 维度切） | 权重/梯度/优化器 |

### 3.1 TP：参数三件套一起除以 `tp`

TP 把一层内的矩阵按列/行切到多个 rank（第 2/21 篇已细讲），因此**权重、梯度、优化器状态每一份都除以 `tp_size`**。这也是为什么「开 TP 就能显著降 per-GPU 显存」——参数三件套全打折。代价是前向/反向每次 GEMM 后要 all-reduce（或 all-gather），这是通信不是显存。

### 3.2 PP：每 rank 只持有 `L / pp` 层

PP 把层序列切成 `pp_size` 段，每 rank 只存自己那一段的**参数 + 梯度 + 优化器状态**（这和第 18 篇 pipeline 调度里的 per-stage 划分一致）。所以 PP 和 TP 一样都省「参数三件套」，不同在于 PP 切的是**层维度**，TP 切的是**层内参数维度**。

关键：**PP 不能省激活**——1F1B 调度里每个 stage 同飞多个 microbatch 的激活（第 18 篇），省的是「计算等待」，激活总量反而可能更大。

### 3.3 ZeRO 的阶梯（第 8 篇）：先切优化器，再切梯度，最后切权重

数据并行的多份模型副本之间，唯一「本来就冗余」的是优化器状态和梯度（第 8 篇 ZeRO 精读）。记 DP 规模为 $dp$（含 sequence 维度拆出来的 DP，即 `dp = world / (tp × pp × cp)`）：

| 阶段 | 优化器状态 | 梯度 | 权重 | 相对节省 |
|---|---|---|---|---|
| ZeRO-1 | ÷`dp` | 全量（all-reduce 后） | 全量 | 主要省优化器 |
| ZeRO-2 | ÷`dp` | ÷`dp`（切分后再 reduce） | 全量 | 省优化器+梯度 |
| ZeRO-3 | ÷`dp` | ÷`dp` | ÷`dp`（用前 gather） | 三件套都切 |

ZeRO-3 相当于「把每一个 DP rank 里的参数三件套也彻底切掉」，只在前向用某块权重时临时 all-gather。代价是**全量参数会在通信缓冲区里短暂存在**——这引出后面的「碎片/缓冲区」那一笔显存。

---

## 4. 五块账，一次算清（以混合精度 BF16 为例）

设参数量 $P$，TP/PP/DP/CP 规模分别为 `tp/pp/dp/cp`。**单卡**显存由五块构成：

### 4.1 参数（BF16 副本 + FP32 master）

$$
M_{\text{param}} = \underbrace{2P/\text{tp}/\text{pp}}_{\text{BF16 权重}} \;+\; \underbrace{4P/\text{tp}/\text{pp}}_{\text{FP32 master}}
$$

（ZeRO-3 时再 ÷dp；但 master 与 BF16 副本在 ZeRO-3 下都切。）

### 4.2 梯度

$$
M_{\text{grad}} = 2P/\text{tp}/\text{pp}
$$

（BF16 下梯度以 2B 计；ZeRO-2/3 再 ÷dp。若梯度内部用 FP32 累加（`--fp32-allreduce` 或某些框架），这一项变 4B。）

### 4.3 优化器状态（Adam 两矩）

$$
M_{\text{opt}} = 2 \times 4P/\text{tp}/\text{pp} = 8P/\text{tp}/\text{pp}
$$

（Adam `exp_avg` 与 `exp_avg_sq` 各 4B；ZeRO-1 起再 ÷dp。）

> 三件套合计，在 BF16 + Adam、关 ZeRO 时约为 $16P$ /tp /pp。这就是「参数 1B 需要约 16GB 训练显存（不含激活）」这个经验值的来源——2B 权重 + 4B master + 2B 梯度 + 8B 优化器 = 16B 字节。

### 4.4 激活（大头中的大头，最看「手艺」）

激活是显存估算里**唯一随 batch 和序列长度线性增长、且高度依赖配置**的一块，也是估算最容易失真的地方。每个 Transformer 层前向要存的中间激活大致是（$s$ 序列长，$b$ microbatch，每层每 token 约 $34h$ 字节-ish，BF16）：

$$
M_{\text{act}} \approx L \cdot s \cdot b \cdot (\text{每层每 token 激活字节})
$$

具体字节数与「存不存 attention 分数矩阵 $QK^T$、MLP hidden」强相关。**三个减活化的大杀器**（前面都精读过）：

- **激活重计算（第 4 篇）**：把「存中间激活」换成「重算」，显存从 $O(L)$ 降到 $O(\sqrt L)$（full checkpoint 时约 $\sqrt L$，因为只有 checkpoint 边界存输入）。代价 flops +100%。
- **序列并行 SP（第 2/3 篇）**：激活沿 seq 维度 ÷`tp`（SP 通常在 TP 组内做）。
- **`distribute_saved_activations`（第 4 篇）**：checkpoint 存下的输入激活再沿 TP ÷`tp`。

**经验级 quick estimate**：无任何优化时，单个 microbatch 的激活峰值常被估为 `layer × hidden × seq × batch × 若干×2B`（若干在 5~18 之间，取决于实现）；重计算后砍到约 `< 10%`。真正落地建议直接 `nsys`/`torch.cuda.memory_summary` 实测，公式只用于事前「该往哪个数量级估」。

### 4.5 碎片 + 通信缓冲区（最容易被忽略的 10~20%）

前面四块是「理想账」，实际还要加一块**隐形成本**：

- **ZeRO-3 的 all-gather 缓冲区**：前向 gather 权重时，会在显存里短暂出现「全量参数」。
- **PyTorch 分配器碎片**：频繁小张量导致显存碎片化，常需预留 5~10%。
- **CUDA context / cuBLAS workspace / nccl buffer**：几 GB 量级，与模型规模弱相关。
- **checkpoint 保存时的 staging buffer**（第 10/22 篇）。

经验上这块给 **总显存的 10%~20% 余量**比较稳妥——这也是为什么「算出来刚好卡进显存」往往 OOM，要留安全系数。

---

## 5. 汇总公式 + 一个完整例子

把五块加起来，得到**单卡训练显存**的估算：

$$
\boxed{
M_{\text{total}} = M_{\text{param}} + M_{\text{grad}} + M_{\text{opt}} + M_{\text{act}} + M_{\text{reserve}}
}
$$

**例子：7B 模型，BF16 + Adam，符号算一版。**

- $P = 7\times10^9$，取 `tp=2, pp=2, dp=4`（共 16 卡），ZeRO-2。
- $M_{\text{param}} = (2+4)P/\text{tp}/\text{pp} = 6P/4 = 10.5\text{ GB}$
- $M_{\text{grad}} = 2P/\text{tp}/\text{pp}/\text{dp}=2P/4/4 = 0.875\text{ GB}$（ZeRO-2 切梯度）
- $M_{\text{opt}} = 8P/\text{tp}/\text{pp}/\text{dp}=8P/4/4 = 3.5\text{ GB}$（ZeRO-2 切优化器）
- $M_{\text{act}}$：取 seq=2048, micro=batch=1, 每层每 token 激活 ~$16h$=64KiB 低估 + 重计算后 ×0.1，$L=32$ → 约几十 MB 到数 GB 级（这一项强烈依赖配置，需当变量看）
- $M_{\text{reserve}}$：加 15% 余量

三件套合计约 14.9GB，激活视配置从 1GB（重计算激进）到 20GB+（不重算、长序列）变化——**激活才是让「线性能不能塞下」的关键变量**。

> 判断瓶颈的一句话：**关 ZeRO 时看「优化器+master」；开 ZeRO-3 后大头转到「激活+碎片」**。所以越大的模型，越是在「省参数（ZeRO/TP/PP）」与「省激活（重计算/SP）」之间反复权衡。

---

## 6. 精度怎么改账本：FP8 / FP4 与「纯 BF16」

上面的单价都可以按需替换，反映不同精度策略的显存：

| 策略 | 权重 | 梯度 | 优化器 | 激活 |
|---|---|---|---|---|
| FP32（全精度） | 4B | 4B | 8B | 4B |
| **BF16 混合精度**（主流） | 2B + 4B master | 2B | 8B | 2B |
| FP8（TE） | 1B（+4B master） | 1~2B | 8B（仍 FP32） | 1B |
| FP4（TE） | 0.5B（+master） | ~1B | 8B | 0.5~1B |

两个关键事实：

1. **优化器状态几乎不随「计算精度」降**——无论 FP8 还是 BF16 训练，Adam 的矩都保持 FP32，这 8B/参数是大头且「顽固」。（这也是为什么纯推理不需要这 8B，推理显存 = 权重 + KV cache，量级天差地别。）
2. **FP8 省的是「流动中的激活/梯度」**，不是「常驻的 master+优化器」。所以 FP8 对**激活显存**的削减比对参数削减更明显（第 6 篇 TE、第 4 篇 fp8 checkpoint 联动）。

---

## 7. 小结

- **记账单位**：显存 = 元素数 × 精度字节；混精 BF16 下「权重 2B + master 4B + 梯度 2B + 优化器 8B」是核心单价表。
- **五块账**：参数、梯度、优化器状态、激活、碎片/缓冲区；前三块「除以并行度」，第四块「除以 batch/seq 维度相关 + 重计算/SP」，第五块是安全余量。
- **并行打折规则**：TP/PP 切「参数三件套」，ZeRO 按 Stage 1/2/3 依次切优化器→梯度→权重，CP/SP 只切激活。
- **激活最看手艺**：随 seq×batch 线性增长，重计算（第 4 篇）降 $O(\sqrt L)$、SP 再 ÷tp，是长序列/大 batch 下的胜负手。
- **精度改单价不改结构**：FP8 省激活/梯度，优化器 8B 恒定；推理显存 = 权重 + KV cache，与本篇训练账本严格区分。

下一篇预告：把这篇的「估算」和 Megatron 实际落地的显存相关开关（`--recompute-*`、`--distribute-saved-activations`、ZeRO 配置到 `DistributedOptimizer` 的映射）串起来，做一次「给定硬件预算反推最大可训模型」的实战演练。

（本文为估算方法篇，未引入新源码行号；涉及的 `recompute` / `distribute_saved_activations` / ZeRO / 优化器实现请分别回看第 4、6、8、11 篇，其行号均基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。）
