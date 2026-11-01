---
title: "Megatron 源码精读（十六）：Context Parallel 细节"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "上下文并行", "context-parallel"]
categories: ["训练框架"]
weight: 16
series: ["megatron-code"]
---

把序列按 token 切开分给多个 rank 之后，每个 rank 只有自己那一段的 Q，却要算出「看过全序列」的 attention——这件事在 Megatron 里有几种做法？打包变长序列（THD）时，为什么还要在两种切法之间搬 token？

一句话结论：**CP 的复杂度全部来自「Q 是局部的、K/V 必须是全序列」这一个约束；`cp_comm_type` 的四种取值只是用不同通信把它做出来，而 THD 场景额外多一层「布局搬运」。**

上一篇讲多模态，本篇收束并行主线：从 TP/PP/SP 到 CP，切分的维度从权重走到了 token。

---

## 1. 先纠正一个常见混淆：SP 和 CP 都切序列

结论：**SP 与 CP 切的是同一维——序列（token）维**，区别在通信语义，不在切哪一维；切 head 的是 TP。

`scatter_to_sequence_parallel_region` 的实现是 `_split_along_first_dim`（`megatron/core/tensor_parallel/mappings.py`），把 `[s, b, h]` 沿 **dim 0** 切开，反向再 all-gather 回来。它切的是激活，目的是让 LayerNorm/dropout 这些逐 token 算子不必在 TP 组里留全量副本。

TP 才沿 head/hidden 切权重（`ColumnParallelLinear` 切输出维）。所以「SP 切 head」是错的：SP 切 token，TP 切 head。

CP 与 SP 的差别在于：CP 是独立的并行维，序列切完之后 attention 无法自洽——每个 rank 只有 $S/P$ 个 token 的 Q，但 K/V 覆盖整条序列。这就是 CP 必须引入新通信原语的原因。

## 2. 四种通信路线：`cp_comm_type` 决定怎么通信

结论：**同一个 CP 语义有四种实现，选哪个是显存、带宽、能否与计算重叠三者的权衡。**

| 取值 | 做法 | 代价 |
|---|---|---|
| `p2p` | ring 式逐段交换 KV，异步 | 默认值；可与 attention 计算重叠 |
| `all_gather` | 先 gather 整条 KV 再算 | 不异步、不能重叠，实现最简单 |
| `a2a` | Ulysses 式：QKV 做一次换轴 all-to-all | 每 rank 得全序列，但只有 `1/cp` 的 head |
| `a2a+p2p` | 分层：低层 a2a（NVLink）+ 高层 p2p（IB） | 需 hcp 组，TE ≥ 1.12 |

取值语义写在 `transformer_config.py` 的 `cp_comm_type` 字段里；真正干活的是 Transformer Engine。MCore 只负责把 group 传下去：TE 版本必须 ≥ 1.0.0，`cp_comm_type` 为 None 时按 `p2p` 传，`a2a+p2p` 还要额外要一个层次化 CP 组（`extensions/transformer_engine.py:1683` 附近）。

## 3. 代码落点一：本地 attention 后端的 all-gather 版

结论：**当层 spec 里选的 core attention 是 MCore 自带的 `DotProductAttention`（而非 TE 版）且 `cp_size > 1` 时，走 `AttentionFuncionWithContextParallel`——它按 KV head 逐个 all-gather，不是一次性搬整条 KV。**

注意类名在上游拼写少一个 t（`Funcion`，`dot_product_attention_context_parallel.py:150`），按正确拼写搜不到；ViT block 的 spec 就是选 `DotProductAttention` 的例子。另外这条路径硬性要求 `attention_dropout == 0`（`dot_product_attention.py:60`），非零直接 assert 失败。

具体做法是**逐个 KV head 循环**（`heads_k_stride = 1`）：双缓冲 `kv_buffer` 形状为 `(2, s_local*cp_size, b, 1, d)`，先用异步 `all_gather_into_tensor` 拉第 0 个 head，之后每轮先 `wait()` 再发下一个 head 的 gather，边等边算。

```python
kv_buffer = torch.empty((2, k.shape[0] * cp_size, k.shape[1], heads_k_stride, k.shape[3]), ...)
comm.all_gather(kv_buffer_copy[0], k_0)   # 异步发起
for i in range(0, nheads_k, heads_k_stride):
    comm.wait()                            # 等上一轮
    kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
    if i < nheads_k - heads_k_stride:
        comm.all_gather(kv_buffer_copy[0], send_k)   # 发下一轮
```

显存要说清楚：单个 head 的 buffer 装的是**整条序列**，即 $O(S)$ 个 token，相当于本地该 head 分片乘以 `cp_size`。因为逐 head 循环，同一时刻只驻留一个 head 的全序列副本——额外开销是「$1/nheads_k$ 层的 KV 再乘 `cp_size`」，不是整层 KV 的 `cp_size` 份。

掩码这里有个细节：`to_zz_mask_attn_bias`（`:135`）把 `[b, 1, s, s]` 掩码沿最后一维切成 $2P$ 块，再按「第 $i$ 块配第 $2cp-1-i$ 块」交错。原因是 gather 出来的 KV 顺序就是 `[chunk0, chunk_{2cp-1}, chunk1, ...]`——每个 rank 本地 KV 已是 zigzag 布局。该路径还 `assert attention_mask is not None`，不给掩码无法开工。

反向是 `reduce_scatter_tensor` 把 `dk`/`dv` 规约回 owner（`:333`），与正向 all-gather 互补。但它是 eager attention、会保存完整 `probs` 供反向，$O(S^2)$ 的注意力矩阵决定了它只能当短序列兜底。

## 4. 代码落点二：THD 打包下的 zigzag ↔ contiguous 搬运

训练默认的 CP 切法就是 zigzag：把序列按 $2P$（$P$ 为 `cp_size`）等分，rank $r$ 取第 $r$ 块和第 $2P-1-r$ 块。实现只有三行（`megatron/core/utils.py:2440` 附近）：`view` 成 $2P$ 块、`index_select` 取 `[r, 2P-1-r]`、再 `view` 回去。

$$
\begin{aligned}
\ell &= S/(2P) \\
\text{first} &= r \\
\text{second} &= 2P - 1 - r
\end{aligned}
$$

这样每 rank 都是「一段轻（靠前）+ 一段重（靠后）」，causal attention 的三角负载被摊平；THD 场景下 `_build_thd_layout_segments` 对每条 packed 序列逐条做同样的切分（`context_parallel_layout/routes.py:63`）。

问题来了：数据侧切好的布局不一定符合各模块要求。attention 层入口把输入转成 **zigzag**（`transformer/attention.py:1367`），GDN/KDA 的 chunkwise 模式则要求 **contiguous**（`ssm/gated_delta_net/gdn.py:150`），而输入布局由 `cp_partition_mode` 决定，默认 `zigzag`（`transformer_config.py:325`），设成 `contiguous` 时必须来自 `sequence_packing_scheduler`。

转换本身是**一次 all-to-all**（`conversion.py:326` 的 `_redistribute_thd_layout`）：先按目标 rank 分组 `index_select` 打包发送缓冲，`all_to_all`（`:397`），收到后按接收索引 scatter 回本地顺序。THD 张量始终保持 packed，不展开成 batch。

路由是一次性算好缓存的。`build_thd_cp_partition_route`（`routes.py:148`）对当前 microbatch 的 `cu_seqlens` 分别算出两种布局下每个 rank 的 segment，再求交集（`_intersect_thd_layout_segments`）得到「我发哪些行给谁、从谁收哪些」，固化成 `ThdCpRoute` 的四个字段：`zigzag_index`/`zigzag_split_sizes`/`contiguous_index`/`contiguous_split_sizes`（`types.py:14`）。同一份路由正反两个方向都能用，所以只需建一次。

硬约束在 `_validate_thd_route_partitioning`（`routes.py:41`）：pack 后总 token 数必须被 `cp_size` 整除，**每条** packed 序列还必须被 `2*cp_size` 整除，否则 zigzag 切不出整数块，直接抛错。

## 5. 工程细节与坑

- **路由缺失会静默降级成同步**：`get_thd_cp_partition_route` 找不到预建路由时会现场构建，代价是一次 `cu_seqlens` 的 D2H 同步，并抛 `FutureWarning`——训练热路径上看到这个警告就是没预建路由。
- **非 THD 走另一条路**：SBHD/BSHD 布局用 `_redistribute_sbhd_layout`，要求 TP 为 1 或偶数，奇数 TP 直接报错（`conversion.py:425`）。
- **布局组合有互斥表**：`contiguous` 只能配 THD + scheduler，且与 GDN 的 `headwise` 线性 CP 互斥；DSv4 Hybrid 反过来强制 `contiguous`；DSA 的 CP 只支持 `all_gather`（`transformer_config.py:1846` 附近）。
- **dynamic CP 是逐 microbatch 换组**：TE 侧每个 microbatch 从 `packed_seq_params.cp_group` 取 CP 组，`local_cp_size == 1` 时干脆把组置 None（`extensions/transformer_engine.py:1805` 附近），短序列直接退化为无通信。

## 6. 小结

- **SP 与 CP 都切 token 维，TP 才切 head**；CP 的特殊性在于 attention 必须看到全序列 K/V。
- **`cp_comm_type` 决定通信形态**：p2p（ring、可重叠）、all_gather（简单、不可重叠）、a2a（Ulysses）、a2a+p2p（分层）。
- **本地 attention 后端用逐 head all-gather**：单 head 全序列缓冲是本地的 `cp_size` 倍，反向 reduce-scatter，且强制无 dropout、必须给 mask。
- **THD 的 zigzag 切分是 `[r, 2P-1-r]` 两段**，每条序列长度须被 `2*cp_size` 整除。
- **布局转换 = 预计算 `ThdCpRoute` + 一次 all-to-all**，attention 要 zigzag、GDN chunkwise 要 contiguous，入口转、出口转回。

本篇收束并行主线：从 TP/PP/SP 到重计算、优化器、checkpoint、MoE、多模态，再到 CP，覆盖了 Megatron-LM 训练框架的一条主干。17–25 是补充篇，接着从《[MCore 架构与 layer spec 机制]({{< relref "megatron-code-17-mcore-arch" >}})》继续。

---

**版本提示（2026-10）**：上游新增 `megatron/core/context_parallel/` 包（`groups.py`/`layout.py`/`sequence_roll.py`，#6776/#6764），提供 `CPLayout`、`convert_cp_layout` 与可微的 contiguous 序列 roll；旧的 `context_parallel_layout/` 仍在（attention/GDN 仍用它），但已不是唯一入口。dynamic CP 已接入标准 train/eval schedule（#7047）；GDN2（GDP）默认走 `headwise` 线性 CP；DSA 要求 `cp_comm_type="all_gather"`。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
