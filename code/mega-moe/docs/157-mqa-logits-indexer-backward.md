# MQA logits（lightning indexer）的训练反向：KL 蒸馏损失的梯度 + 打分算子反向（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的第 157 篇（P147 增量）。
> 第 19 篇（P9-1）把 MQA logits 的**连续 / 分页前向**、第 60 篇（P50-1）把它的**验收测试即正确性
> 规范**拆透了，但 DeepGEMM 只实现**前向打分**——lightning indexer 是要**训练**的，它的反向梯度
> 一直没人讲。本篇补上：① DeepSeek 用什么样的目标训练 indexer；② KL 蒸馏损失对 logits 的梯度；
> ③ 打分算子本身的四路反向（`dq / dw / dkv_data / dkv_scale`）；④ top-k 选择的直通与 FP8 量化 STE。
> 全部在 **H100(sm90)** 上用 **torch.autograd + 中心差分**对拍，并给一个可跑的 **CUDA 反向后端**。

源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`；论文坐标：DeepSeek-V3.2 技术报告
[arXiv:2512.02556](https://arxiv.org/abs/2512.02556) §2.1（式 (3)/(4)）与 §2.2。

## 1. 定位：DeepGEMM 只有前向，反向由谁补

DeepGEMM 把 indexer 的**打分**当作一条独立 kernel family（README `:88-108`「V3.2 MQA kernels for
the indexer」），但它**只输出 logits、不产生梯度**。训练侧的反向由框架（Megatron 系 MoE/attention
的 autograd）完成——这层从没在专题里出现过，而它恰恰是 MoE 前反向之外的**第二条 DeepSeek 关键
计算链**。

DeepSeek-V3.2 训练 indexer 的目标（arXiv:2512.02556 §2.1）分两段：

- **Dense warm-up（式 (3)）**：先把主注意力各头分数**沿头求和**，再沿序列维做 **L1 归一化**得到目标
  分布 `p_{t,:}`（detach），然后最小化

  $$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\!\left(p_{t,:}\,\middle\|\,\mathrm{Softmax}(I_{t,:})\right).$$

- **Sparse training（式 (4)）**：引入 fine-grained token selection 后，**只在选中集**
  $\mathcal{S}_t=\{s\mid I_{t,s}\in\text{Top-}k(I_{t,:})\}$ 上对齐（每个 query 选 2048 个 KV）：

  $$\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\!\left(p_{t,\mathcal{S}_t}\,\middle\|\,\mathrm{Softmax}(I_{t,\mathcal{S}_t})\right).$$

关键约定：**indexer 的输入从主模型的计算图里 detach**（论文原话「we detach the indexer input from
the computational graph for separate optimization」）——indexer 只由 $\mathcal{L}^I$ 训、主模型只由
语言建模损失训。这决定了反向链路的入口就是上游的 `dL/dI`。

## 2. 前向语义（真实行号）

打分算子的逐 token 定义，README `:100-108` 与 `tests/test_attention.py:78-101`（`ref_mqa_logits`）
完全一致：

```python
kv_j = kv[0][j, :] * kv[1][j].unsqueeze(1)          # [head_dim]，每 KV token 一个 fp32 scale
out_ij = q[i, :, :] @ kv_j                          # [num_heads]，每头点积
out_ij = out_ij.relu() * weights[i, :]              # [num_heads]，ReLU 后按 per-head 权重加权
out_ij = out_ij.sum()                               # 标量 logit
```

写成公式（`t`=query token、`s`=KV token、`h`=head、`d`=head_dim）：

$$
\hat{k}_{s,d}=kv_{s,d}\cdot \mathrm{sf}_s,\qquad
z_{t,s,h}=\sum_d q_{t,h,d}\,\hat{k}_{s,d},\qquad
I_{t,s}=\sum_h w_{t,h}\,\mathrm{relu}(z_{t,s,h}).
$$

几个与实现绑定的细节（都影响反向）：

- **KV 的 scale 是「先乘进 KV」的**：kernel 把 `scale` 当 KV 的一部分（`ref_mqa_logits` 的 `k = kv.float()`
  实际在测试里 `kv` 已是缩放后的融合布局，SM90 从 `tests/test_attention.py:270-283`
  `kv_cache_cast_to_fp8` 的 per-token fp32 scale 读出）。所以反向要**同时**求 `kv_data` 与 `kv_scale`
  两路梯度。

- **输出按行压缩**：每行只存有效区间 `[ks_t, ke_t)` 从第 0 列起（README `:99`）；无效位置在 softmax
  里是 `-inf`，梯度恒 0。

- **`weights` 行按 16 字节 TMA 对齐**（`csrc/apis/attention.hpp:41-43`）：这是布局要求，不改变数学。

- **Q / KV 的精度**：SM90 用 E4M3 + 逐 token fp32 KV scale（`attention.hpp:32` 的 `is_mx_sf ==
  (arch_major == 10)`），SM100 用 MXFP4/MXFP8 + packed UE8M0。反向以「反量化后的操作数」为准，
  量化层用 STE（§5）。

## 3. KL 蒸馏损失的梯度：`dL/dI = softmax(I) - p`

对式 (3) 逐行求导。记 $q_{t,s}=\mathrm{softmax}_s(I_{t,s})$，目标 $p_{t,\cdot}$ 已 detach：

$$
\frac{\partial}{\partial I_{t,s'}}\sum_s p_{t,s}\log q_{t,s}
=\sum_s p_{t,s}(\delta_{ss'}-q_{t,s'})
=p_{t,s'}-q_{t,s'}\sum_s p_{t,s}.
$$

因 $\sum_s p_{t,s}=1$（L1 归一化），得

$$\frac{\partial \mathcal{L}^I}{\partial I_{t,s'}}=q_{t,s'}-p_{t,s'}.$$

即「**模型分布减目标分布**」——标准 cross-entropy / KL 的梯度形状。两个可直接验证的性质：
**行和为 0**（softmax 平移不变，KL 对 $I$ 整体加常数不变）、**无效位置梯度为 0**。
H100 实测（`harness/exp_mqa_logits_indexer_backward.py` Part A）：

```
[PASS] dL/dI 解析 vs autograd     relerr=2.28e-17
[PASS] dL/dI 行和 = 0（softmax 平移不变）        max=2.7e-16
[PASS] 目标分布 p 每行 L1 归一化                max|sum-1|=1.1e-16
```

**式 (4) 的稀疏版**只在 $\mathcal{S}_t$ 内做 softmax 与 KL：解析梯度仍为 `q_S - p_S`，但
`q_S = softmax(I_{S})` 与 `p_S` 都在 $\mathcal{S}_t$ 上重新归一化，选中集外恒 0。实测式 (4)
`relerr=3.51e-17`；并验证「全序列 softmax + 只取 S 的 loss」与式 (4) **不同**（relerr 0.28）——
这是稀疏阶段必须在 S 内归一化的实证。

## 4. 打分算子的反向：四路梯度

给定上游 `dI_{t,s} = dL/dI_{t,s}`（SPOILER：由 §3 给出），对 §2 的链式求导：

$$
\frac{\partial \mathcal{L}}{\partial w_{t,h}}=\sum_s dI_{t,s}\,\mathrm{relu}(z_{t,s,h}),\qquad
\frac{\partial \mathcal{L}}{\partial z_{t,s,h}}=dI_{t,s}\,w_{t,h}\,\mathbf{1}[z_{t,s,h}>0],
$$

$$
\frac{\partial \mathcal{L}}{\partial q_{t,h,d}}=\sum_s \frac{\partial \mathcal{L}}{\partial z_{t,s,h}}\,\hat{k}_{s,d},\qquad
\frac{\partial \mathcal{L}}{\partial \hat{k}_{s,d}}=\sum_{t,h}\frac{\partial \mathcal{L}}{\partial z_{t,s,h}}\,q_{t,h,d}.
$$

最后从 $\hat{k}_{s,d}=kv_{s,d}\,\mathrm{sf}_s$ 拆开：

$$
\frac{\partial \mathcal{L}}{\partial kv_{s,d}}=\frac{\partial \mathcal{L}}{\partial \hat{k}_{s,d}}\,\mathrm{sf}_s,\qquad
\frac{\partial \mathcal{L}}{\partial \mathrm{sf}_s}=\sum_d \frac{\partial \mathcal{L}}{\partial \hat{k}_{s,d}}\,kv_{s,d}.
$$

三点非平凡：

1. **ReLU 是唯一的非线性**（无 softmax、无 online-max），故反向只需 $z$ 的符号掩码
   $\mathbf{1}[z>0]$，与第 19 篇前向 kernel「ReLU+加权求和的两级 warp 归约」逐元素对偶
   （`sm90_fp8_mqa_logits.cuh:282-303`）。
2. **两路收缩方向不同**：`dq` 沿 `s`（每个 query 独立，输出为 query 维）、`dkv` 沿 `t`
   （**跨 token 汇聚**，天然是 scatter/atomic）。
3. **每 KV token 一个 scale**，`dkv_scale` 是标量（沿 `d` 收缩）——它把 $q$ 与 `kv_data` 的点积再点一次。

H100 实测（Part B，全部 fp64 autograd 对拍）：四路 `relerr` 0 / 0 / 1.84e-16 / 1.87e-16；Part C 用
中心差分二次核验 4 个标量位置，最差 `relerr=1.32e-08`。

## 5. top-k 直通与 FP8 STE

**top-k 选择不可微**：式 (4) 的 $\mathcal{S}_t$ 由 $I$ 自身选出（`torch.topk`），反向视作常量掩码——
梯度只经「选中集内的 softmax」，未选中位置 `dI=0`（实测严格 0）。这正是第 129 篇讨论的
DeepSeek「控制回路」在此处的体现。

**FP8 量化走 straight-through**：前向 KV 被量化成 E4M3（每 token 一个 scale，
`tests/test_attention.py:270-283`），反向令 `dL/d(kv_fp32) = dL/d(反量化值)`。用标准 STE 写法

$$x_{\text{ste}}=x+\big(\mathrm{quant}(x)-x\big).\text{detach}()$$

（值 = 舍入值、梯度 = 1）实测三路梯度 relerr ≤9.12e-17；反例「舍入值不回接」使 kv 直接**不在图里**
（梯度 `None`）。E4M3 per-token 量化误差量级 `relerr=0.0109`。这与 MoE 反向里的量化 STE
（第 4/14 篇）是同一套路。

## 6. CUDA 反向后端（真实设备代码）

`src/mqa_indexer_backward.cu` 实现打分算子反向：`grid=T`（每 query token 一 block）、
`block=(32 lane, H)`，每个 head 一个 warp——沿 lane 跨步点积 + `__shfl_xor` 归约得 $z_{s,h}$，
`dq` 累在寄存器（block 独占一行 `t`），`dkv_data/dkv_scale` 跨 `t` 汇聚走全局 `atomicAdd`。
H100(sm90) 编译运行（sha256 `5c74e749…`），自带 `[A]` 正确性 + `[B]` 吞吐报告：

```
[A] 正确性：dq 8.107e-08 / dw 8.724e-08 / dkv_data 1.003e-07 / dkv_scale 9.484e-07  => ALL PASS
[B] 吞吐（ragged 有效区间、含 scatter 原子归约，受争用下界）
    T=1024,H=8,KV=512,D=128    2.039 ms/call
    T=1024,H=8,KV=2048,D=128   8.480 ms/call
    T=4096,H=8,KV=512,D=128    6.798 ms/call  ~1.3 GFLOP/s
```

**为什么 `dkv` 这么慢（~1 GFLOP/s）**：`dkv_{s,d}=\sum_{t,h}\dots` 是**跨 token 的 scatter**，本实现
把它做成 `atomicAdd`，scatter 量 $O(T\cdot KV\cdot H\cdot D)$。对 long-context（$T$ 可达 128K、
每 token 选 2048 KV），这个量惊人——这正是 **indexer 头数被设计得很小**（论文 §2.1「a small number
of heads」）的工程原因之一。生产实现不会逐元素 `atomicAdd`，而是按 KV 分块做局部归约（此处的
简单实现只用于验证语义 + 给量级）。前向的 `1 GFLOP/s` 不是算力瓶颈（索引器 FLOP 很小），
瓶颈是**反向的 scatter 归约 + 访存**，与第 152 篇「偏斜代价在显存/通信而非计算」同构。

## 7. 与 MoE 反向的关系

| 维度 | MoE 反向（第 4/12/14 篇） | indexer 反向（本篇） |
|---|---|---|
| 上游损失 | LM loss → `dY` | KL 蒸馏 $\mathcal{L}^I$ → `dI = softmax(I) - p` |
| 主算子 | 四个 grouped GEMM | 一个「点积+ReLU+加权求和」+ 不可微 top-k |
| 非平凡点 | SwiGLU 反向、四布局、累积协议 | softmax/KL Jacobian、top-k 直通、跨 token scatter |
| 计算/访存 | wgrad 访存受限（第 15/145 篇） | 反向 scatter 归约受限 |

两者共享同一套**量化 STE**（FP8 forward + straight-through backward），也共享「前向 kernel 由
DeepGEMM 提供、反向由框架 autograd 提供」这一分工——只是 MoE 反向已有 grouped GEMM 的
k_grouped/m_grouped 现成 kernel，而 indexer 反向**没有对应 kernel**，只能靠框架或自写（本篇原型）。

## 8. 小结

- indexer 的训练目标 = **对主注意力分布做 KL 蒸馏**（式 (3)、(4)），indexer 输入 detach、只由
  $\mathcal{L}^I$ 训；入口是 `dI = softmax(I) - p`（稀疏阶段只在 top-k 选中集内）。
- 打分算子反向四路 `dw / dq / dkv_data / dkv_scale` 全部与 autograd fp64 逐位一致
  （relerr ≤2e-16，中心差分 1.3e-8）；top-k 直通、FP8 STE 均验证。
- CUDA 后端 `[A] ALL PASS`；吞吐揭示 indexer 反向的**真正瓶颈是 `dkv` 的跨 token scatter 归约**，
  这是「indexer 头数很小、每 token 只选 2048 KV」的工程动机。
- **边界**：SM100 的 MXFP4/MXFP8/packed UE8M0 与分页/稀疏主 kernel 仍属 SM100 专属、仅分析
  （第 19/22/23/24/60 篇）；本篇的数学/量化/直通语义**架构无关**，H100 真机可跑。

实验与原始输出：`harness/exp_mqa_logits_indexer_backward.{py,sh,out.txt}` + `src/mqa_indexer_backward.cu`
（H100 真机 `PASS=23 / FAIL=0`、设备端 `ALL PASS`）。
