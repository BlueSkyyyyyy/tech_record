---
title: "Flash Attention 精读（二）：Triton 教程版前向逐行精读"
date: 2026-08-28
draft: false
weight: 2
tags: ["flash-attention", "triton", "attention", "gpu", "系列"]
categories: ["算子开发"]
---

本篇精读 [Triton 官方教程 `06-fused-attention.py`](https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py) 的**前向**部分（约 L30–403 + autograd 封装 L755–905），约 200 行核心代码，实现了 Flash Attention v2。它是理解这个算法最好的教材——比 CUDA 版短一个数量级，又保留了所有关键的数学与工程决策。反向部分留给[第 4 篇]({{< relref "flash-attention-04-bwd" >}})。

建议对照[第 1 篇]({{< relref "flash-attention-01-theory" >}})的递推公式阅读：本篇的每一节都是"公式 → 代码"的映射。

## 1. 文件鸟瞰

| 区段 | 行号 | 内容 |
|---|---|---|
| 特性检测 | L30–58 | `is_hopper / is_blackwell / supports_host_descriptor`，决定 TMA、warp specialization 开关 |
| `_attn_fwd_inner` | L65–183 | **online softmax 主循环**（本篇核心） |
| autotune 配置 | L214–260 | `BLOCK_M/N × num_stages × num_warps` 搜索空间与剪枝 |
| `_attn_fwd` | L284–403 | 前向主 kernel：grid 映射、状态初始化、两阶段调用 inner、epilogue |
| `_attention` | L755–905 | `torch.autograd.Function` 封装 |

前向的数据流一句话：**grid 维度并行 Q 块，循环维度串行 KV 块，三组寄存器状态（$m_i, \ell_i, \mathrm{acc}$）贯穿始终**。

## 2. 主 kernel `_attn_fwd`：并行结构

### 2.1 Grid 映射（L320–323）

```python
start_m = tl.program_id(0)
off_hz = tl.program_id(1)
off_z = off_hz // H
off_h = off_hz % H
```

`program_id(0)` 是 Q 块索引，`program_id(1)` 是 `batch × head` 展平索引——正是第 1 篇说的"Q 分块是 grid 维度（块间并行）"。Q/K/V 被统一看成 `[Z*H*N_CTX, HEAD_DIM]` 的二维张量（L327–339），当前 batch/head 的行偏移是 `off_z*(N_CTX*H) + off_h*N_CTX`（L342）。二维展开避免了四维 stride 的一堆乘法，也方便 TMA descriptor 使用。

### 2.2 状态初始化（L355–357）

```python
m_i = tl.zeros([BLOCK_M], tl.float32) - float("inf")
l_i = tl.zeros([BLOCK_M], tl.float32) + 1.0
acc = tl.zeros([BLOCK_M, HEAD_DIM], tl.float32)
```

这三个初值就是第 1 篇递推式的寄存器化身，但**初始化本身藏着一个技巧**：$\ell_i$ 初值是 1.0 而不是 0。

代入第一次迭代：$m^{old}_i = -\infty$，则 $\alpha = \mathrm{exp2}(-\infty - m^{new}) = 0$，于是

$$
\ell_i \leftarrow 1.0 \times 0 + \ell_{ij} = \ell_{ij}, \qquad
\mathrm{acc} \leftarrow 0 \times \mathrm{acc} + P V = P V
$$

第一次迭代"自动退化"成普通计算，**不需要为第 0 块写特殊分支**。$\ell_i = 1.0$ 的作用是让被清零的旧值乘出来还是 0——数学上无所谓，代码上省一个 if。

### 2.3 scale 的折叠（L362–363）

```python
qk_scale = sm_scale * 1.44269504   # sm_scale / ln(2)
```

全文件没有一处自然指数。$1/\ln 2 = 1.44269504$ 被折进 scale，所有 `exp` 变成 `exp2`——GPU 的 SFU 只有 `ex2.approx.f32` 硬件指令，`exp(x)` 需要软件展开成 `ex2(x · log2 e)`，先折进去就省掉每次的乘法。同时前向存下的 LSE 也是以 2 为底的（见 §4），反向一行 `exp2(qk - M)` 就能恢复 softmax 概率——前后向的接口在数值域的选择上就设计好了。

**scale 乘在减 max 之前**（L123–125）：正数缩放是保序的，$\max(cx) = c\max(x)$，所以先乘后减与先减后乘等价；但必须保证"减去的 max"与"被指数的量"在同一量纲（缩放后域），$\mathrm{exp2}(S' - m') \le 1$ 才恒成立。若把 scale 留到 exp 之后补乘，未经缩放的 $S$ 稍大就会上溢。

## 3. 主循环 `_attn_fwd_inner`：递推式的逐行翻译

循环骨架（L113）：

```python
for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=...):
```

`tl.range` 的 `warp_specialize=True`（Hopper/Blackwell）让 Triton 把"加载 K/V"与 `tl.dot` 分派给不同 warp 组，形成软件流水（producer–consumer）。函数签名里 `offs_m / offs_n` 声明为 `tl.constexpr`（L70），让 mask 比较与地址计算在编译期折叠。

循环体一次处理一个 `[BLOCK_N, d]` 的 K/V 块，对照第 1 篇的框式递推：

### 3.1 算 S（L121 / L134）

```python
qk = tl.dot(q, k)                       # tensor core，fp32 累加
qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)   # STAGE==2（带 mask 分支）
# 或无 mask 分支：
m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
```

causal 掩码是 `offs_m[:, None] >= (start_n + offs_n[None, :])`（下三角）。**哨兵是 `-1.0e6` 而不是 `-inf`**：如果某行整块被 mask 且 $m_i$ 还是 $-\infty$，那么 $-\infty - (-\infty) = \mathrm{NaN}$ 会污染整个 softmax；$-10^6$ 减去有限的 $m_{ij}$ 后经 exp2 干净地下溢为 0。哨兵加在 scale 之后（L125），保证掩码值不随 scale 缩放。

### 3.2 更新 max 与修正因子（L127–144）

```python
m_ij = tl.maximum(m_i, tl.max(qk, 1))   # m^new = max(m^old, rowmax(S_block))
qk -= m_ij[:, None]                     # 参数 ≤ 0，exp2 安全
alpha = tl.math.exp2(m_i - m_ij)        # α = 2^(m_old − m_new)
```

这就是递推式的前两行。`alpha` 是旧坐标系到新坐标系的搬运工。

### 3.3 重缩放与第二次 GEMM（L162–172）

```python
p = tl.math.exp2(qk)                    # 未归一化分子 P_block ∈ [0,1]
l_ij = tl.sum(p, 1)                     # 本块的行和
acc = acc * alpha[:, None]              # 旧输出迁移到新坐标系
l_i = l_i * alpha + l_ij                # 行和迁移 + 累加
# 数值技巧：p 降到 fp16 再进 tensor core
v = desc_v.load([0, offsetv_y]).T       # （fp8 路径下 V 转置加载）
acc = tl.dot(p, v, acc)                 # acc += P_block @ V_block
```

三个细节值得停留：

**（1）`p.to(tl.float16)` 的安全性边界。** tensor core 的输入必须是 fp16/bf16/fp8。能安全降精的前提写在第 1 篇：减 max 之后 $p \in [0, 1]$，fp16 在 $[0,1]$ 区间有充分有效数字；而 `tl.dot(p, v, acc)` 的三参数形式让**硬件累加器是 fp32**。精度损失只发生在乘法操作数上，落点在 fp16 噪声级别。反过来，$\ell_i$、$m_i$、acc 的所有跨块累加全程 fp32——"降精只发生在 tensor core 强制要求的操作数位置"是贯穿前向反向的铁律。

**（2）`m_i/l_i` 的更新放循环末尾（L176–177）**，注释明说是让这两个中间量的生命周期尽早结束、降低寄存器压力。128×128 的 fp32 acc 本身就占满一半寄存器，生存分析这种编译器层面的考虑在这里是手写的。

**（3）非 Hopper 架构的寄存器拆分（L152–160）**：当 `BLOCK_M == HEAD_DIM == 128` 且开 warp specialization 时，把 acc reshape/permute/split 成两半、分别乘 `alpha` 再 join 回去。注释直言是为了避免 spilling——128×128 fp32 acc 整体 rescale 会让寄存器分配恶化，拆两半分两次做。这是"为特定 shape 手写指令级 workaround"的典型样本，第 3 篇 CUDA 版你会看到同款考虑如何上升为 warp specialization 的设计动机。

## 4. STAGE 两遍遍历：causal 的块级跳过（L91–100, L372–392）

教程版最漂亮的结构设计。STAGE 是**位标志**：bit0 = off-band 遍历，bit1 = on-band 遍历。

- **STAGE=1（非 causal）**：一次调用 inner，`lo, hi = 0, N_CTX`，全程无 mask。
- **STAGE=3（causal）**：两次调用 inner——
  - 第一次（L378–383）处理 $[0,\ start_m \cdot B_M)$：这些 key 严格早于当前 Q 块的所有行，**完全可见、零 mask**（off-band）；
  - 第二次（L387–392）只处理对角带 $[start_m \cdot B_M,\ (start_m+1) B_M)$：唯一需要 mask 的区域（on-band），宽度恰好一个 BLOCK_M。

`4 - STAGE`（L382）这个表达式把"非 causal 的全量扫描"与"causal 的第一次 off-band 扫描"统一进同一个调用点。收益：mask 的比较与 `tl.where` 开销只发生在对角带上（占比 $\approx B_M / N$），其余 $N - B_M$ 个 key 的内循环指令流更短；`tl.multiple_of` 提示（L97、L114）再帮编译器做对齐优化。

## 5. Epilogue（L399–403）：归一化与 LSE

```python
m_i += tl.math.log2(l_i)   # M_i = log2(ℓ) + m = 以 2 为底的 log-sum-exp
acc = acc / l_i[:, None]   # O = Σ P·V / ℓ
```

循环内不除 $\ell$（中间块的 $\ell$ 不完整，且除法比乘法贵——每块除一次是 $O(N/B_N)$ 次，循环外一次是 1 次）。写入 `M` 的是 $m_i + \log_2 \ell_i$——**以 2 为底的 LSE**，反向 kernel 里 `exp2(qk - M)` 一行恢复精确 softmax 概率（分子分母同除完成），连 $\ell_i$ 都不用单独存。这是前后向之间最重要的接口约定。

## 6. Autotune 与 TMA（L214–260, L190–207）

```python
configs = [triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN},
                         num_stages=s, num_warps=w)
           for BM in [64, 128] for BN in [32, 64] ... ]
```

- 搜索空间是 `BLOCK_M/N × num_stages × num_warps`；`keep` 静态过滤掉 Hopper 上"小 tile + 8 warps"的差配置；`prune_invalid_configs` 运行时剪枝 `BLOCK_M > N_CTX` 的非法项和 causal 下 `BLOCK_M < BLOCK_N` 的低效项（对角带内大量浪费的 mask 计算）。
- autotune key 含 `N_CTX`——**序列长度变化会触发重新调参与重编译**，这是线上服务的常见坑。
- `_maybe_make_tensor_desc`（L267–277）让同一份 kernel 兼容裸指针与 TMA descriptor 两种寻址；host 侧 pre-hook（L190–207）在 launch 前按最终 block 尺寸动态设置 `desc.block_shape`（构造时先用 `[1,1]` 占位）。

## 7. 测试里读出的工程事实（L914–993）

- 参考实现（L949–955）的 softmax **强制在 fp32 计算**再 cast 回 fp16——参考基准自身先规避了 fp16 softmax 的精度坑。
- 容差：fp16 前向/反向 `atol=1e-2, rtol=0`；**fp8 前向 `atol=3`**——e5m2 只有 2 位尾数，"量级正确即可"如实反映了 FP8 attention 的精度代价；AMD gfx90a 反向额外放宽 `rtol=1e-2` 是该架构 MFMA/超越函数的硬件特性妥协。
- 反向不支持 fp8（测试里直接跳过）；`N_CTX` 须被 128 整除（反向假设）。这些"教程没说的限制"正是它与生产实现的差距清单，第 3 篇逐项补齐。

## 8. 小结

200 行 Triton 把第 1 篇的数学变成了代码，同时展示了三类工程决策：

| 类别 | 决策 | 位置 |
|---|---|---|
| 数值 | exp2 折叠、scale 位置、`-1e6` 哨兵、p 降 fp16 / 累加 fp32 | L363, L125, L170 |
| 结构 | off-band/on-band 两遍、$\ell$ 循环外归一化、LSE 接口 | L377–392, L399–403 |
| 硬件 | constexpr 索引、寄存器拆分、warp specialization、TMA 双模态 | L70, L152–160, L113, L267 |

下一篇进入 CUDA：FlashAttention-2 在 sm80 上的 warp 分工与软件流水线，FlashAttention-3 在 Hopper 上的 TMA + warp specialization + pingpong——同一份数学，换一套"把 GPU 榨干"的工程。
