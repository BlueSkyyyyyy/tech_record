---
title: "Megatron 源码精读（十六）：Context Parallel 细节"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "上下文并行", "context-parallel"]
categories: ["训练框架"]
weight: 16
series: ["megatron-code"]
---

这是系列的最后一篇。前面把 TP/PP/SP、重计算、优化器、MoE、蒸馏、强化学习、多模态都过了一遍，本篇收在 **Context Parallel（上下文并行，CP）**——当序列太长、连序列并行（SP）沿 head 切都不够时，把**序列长度维度本身切成多段分给不同 rank** 的并行方式。

CP 和前几种并行有个本质区别：TP/PP/SP 切的是**权重或激发里彼此正交的维度**，切完各 rank 基本能独立算；而 CP 切的是**序列位置**，attention 里每个 query 都要看**整条序列的 K/V**，所以切完之后 attention 就「天然依赖其它 rank 的数据」，必须跨 rank 通信。**怎么在切开序列的同时保证 attention 语义正确、负载还均衡**，就是 CP 的全部内容。核心代码在两处：`megatron/core/transformer/dot_product_attention_context_parallel.py`（native CP attention）和 `megatron/core/context_parallel_layout/`（varlen 下的布局路由）。

---

## 1. 问题背景：序列维度切分后的 attention 怎么办

先回忆前几篇的并行：TP 沿 hidden/head 切权重、SP 切序列的 head 维度（`scatter_to_sequence_parallel_region`，第 11 篇提过）、PP 沿 layer 切。它们的共同点是——**切了之后每个 rank 的数据是自洽的**，不需要看别人的 token。

CP 不同：把一条 `seq_len = S` 的序列切成 `cp_size` 段，每段 `S/cp_size` 个 token 分给一个 rank。但 attention 的 QKT 是要「每个 token 对**全部** token」的：

```
attention(Q,K,V) = softmax(Q @ K^T / √d) @ V
```

Q 在第 0 段 rank 上，但 K、V 覆盖整条序列、散在 cp_size 个 rank 上。于是每个 rank 都必须拿到**别人的 K/V 片段**才能算自己那段的 attention。这就是 CP 通信的根源。

---

## 2. 现象一：native CP 用 all-gather 把整条 KV 都拉过来

先看最直白的一种实现。`AttentionFuncionWithContextParallel`（`dot_product_attention_context_parallel.py:150-237`）是 native（不依赖 TE）的 CP attention，文件头注明改编自 ring-flash-attention（`dot_product_attention_context_parallel.py:3`）。

它的策略不是 ring 式的「逐段 send/recv」，而是**干脆把整条 K/V all-gather 到每个 rank**。核心通信封装是 `AllGatherComm`（`dot_product_attention_context_parallel.py:108-132`）：

```python
# dot_product_attention_context_parallel.py:115-124（节选）
def all_gather(self, output_tensor, input_tensor):
    if self.group is None:
        output_tensor.copy_(input_tensor)
    else:
        handle = torch.distributed.all_gather_into_tensor(
            output_tensor, input_tensor, group=self.group, async_op=True
        )
        self.handles.append(handle)
```

forward 里（`dot_product_attention_context_parallel.py:173-185`）先开一个 `kv_buffer`，形状按 `cp_size` 倍扩大，然后把本 rank 的 K/V 首个 chunk `all_gather` 进去；之后在循环里（`dot_product_attention_context_parallel.py:196-222`）一边等上一次 gather 完成、一边异步 gather 下一段，同时用已经 gather 到的**整条序列 K/V** 和本 rank 的 Q 算 eager attention。

关键点有两处：

1. **异步流水**：`comm.wait()`（`dot_product_attention_context_parallel.py:198`）在 gather 下一段（`dot_product_attention_context_parallel.py:201-207`）之前把通信和计算叠起来，减少空等；
2. **反向 reduce-scatter**：`backward` 里（`dot_product_attention_context_parallel.py:333-334`）对 `dk_i`/`dv_i` 做 `reduce_scatter_tensor`，把 all-gather 造成的「每 rank 都有一份 K/V 梯度」规约回各自 owner。all-gather 的通信正反互补——正向 gather、反向 reduce-scatter，这是标准的「all-gather 类并行」模式。

不过这种把整条序列塞进每个 rank 的做法，显存是 `O(cp_size * S)`，长序列下会爆。所以它是 native 兜底路径，真正长序列要走 TE 的 ring attention 或下述布局方案。

---

## 3. 现象二：varlen 下 CP 的「zigzag ↔ contiguous」布局转换

第二个、也是更新颖的部分是 `megatron/core/context_parallel_layout/`。它解决的是 **varlen（变长）序列打包 + CP** 场景下的一个具体问题：序列被 pack（THD，total-hidden-dim）后，不同 rank 分到的 token 数、attention 负载都不均匀。于是引入**两种序列布局**（`context_parallel_layout/types.py:10`）：

```python
# context_parallel_layout/types.py:10
CpPartitionMode = Literal["zigzag", "contiguous"]
```

- **contiguous**：把 pack 后的 token 按顺序平均切成 cp_size 段，每 rank 一段（自然、好算 loss/对 optimizer 友好）；
- **zigzag**：每 rank 拿**两段**——一段来自序列头（`first_chunk = cp_rank`）、一段来自序列尾（`second_chunk = 2*cp_size - cp_rank - 1`）（`routes.py:83-86`）。

zigzag 的动机很好理解：causal attention 的负载是「靠前的 token 看很少 K/V、靠后的 token 看很多 K/V」。如果按 contiguous 切，最后一个 rank 会扛下最重的 lower-triangular 计算。zigzag 把「头（负载轻）」和「尾（负载重）」**交错配给每个 rank**，让每个 rank 的 attention 计算量大致均衡。这正是 LLM 社区里「CP 负载均衡 / zigzag ring attention」那个经典布局。

### 3.1 zigzag 的两段到底取哪

`_build_thd_layout_segments`（`routes.py:63-90`）把逻辑写得很清楚。对每条 packed 序列：

```python
# context_parallel_layout/routes.py:81-88（节选）
for seq_start, seq_end in zip(cu[:-1], cu[1:]):
    seq_len = seq_end - seq_start
    chunk_len = seq_len // (2 * cp_size)
    first_chunk = cp_rank
    second_chunk = 2 * cp_size - cp_rank - 1
    segments.append((seq_start + first_chunk * chunk_len, chunk_len, local_start))
    segments.append((seq_start + second_chunk * chunk_len, chunk_len, local_start + chunk_len))
```

`chunk_len = seq_len / (2 * cp_size)` 说明每条序列被切成 `2 * cp_size` 个小块；rank `cp_rank` 拿走第 `cp_rank` 块（头侧）和第 `2*cp_size-cp_rank-1` 块（尾侧）。排名靠前的 rank 拿轻尾巴、排名靠后的拿重尾巴，整体看每 rank 都是「一段轻 + 一段重」，负载被摊平。

注意 `_validate_thd_route_partitioning`（`routes.py:41-60`）里的硬约束：`total_tokens` 必须被 `cp_size` 整除，且每条序列长度必须被 `2*cp_size` 整除（`routes.py:49-59`）——否则 zigzag 切不出整数块。这是 CP 在 varlen 下一个很实际的限制。

### 3.2 布局转换 = 一次 all-to-all

两个布局之间不能白切，得有个`转换`动作。`CpPartitionModeConverter`（`conversion.py:27`）和 `convert_cp_partition_mode`（`conversion.py:208`）负责在 zigzag 和 contiguous 之间搬 token。核心实现 `_redistribute_thd_layout`（`conversion.py:326-412`）用**一次 all-to-all** 完成这个 permutation，路由信息来自预计算好的 `ThdCpRoute`（`types.py:13-25`，存 `zigzag_index`/`contiguous_index`/`split_sizes`）。

路由是一次性算好的：`build_thd_cp_partition_route`（`routes.py:148-210`）对当前 microbatch 的 `cu_seqlens` 先 `_build_thd_layout_segments` 算出每个 rank 在两种布局下的 segment，再 `_intersect_thd_layout_segments`（`routes.py:93-121`）求交集得到「我有哪些 token 要发给谁、从谁那里收哪些」，把结果固化成 `row_order` 索引 + `split_sizes`。之后每次转换直接拿这个索引做 gather/scatter（`conversion.py:358-367`），不再现算。

一句话总结这半段的架构：**attention 想用 zigzag（均衡），loss/optimizer 想用 contiguous（自然），于是两者之间靠一张预计算路由表 + 一次 all-to-all 来回切换。**

---

## 4. 小结

- **CP 的根因是「切序列后 attention 必须看全序列」**：Q 在本 rank，K/V 散在 cp_size 个 rank，必须通信。
- **最直白的实现 = all-gather KV**：`AllGatherComm` 把整条 K/V gather 到每个 rank 再算 eager attention，正向 gather、反向 reduce-scatter（`dot_product_attention_context_parallel.py:108-132`、`333-334`）。
- **长序列显存爆 → 用 zigzag 布局做负载均衡**：contiguous 按序切、zigzag 头尾交错，把 causal attention 的重负载摊平（`routes.py:63-90`）。
- **两种布局切换 = 预计算路由 + 一次 all-to-all**：`ThdCpRoute` 缓存 `row_order`/`split_sizes`，`_redistribute_thd_layout` 用 all-to-all 搬 token（`routes.py:148-210`、`conversion.py:326-412`）。
- **硬约束**：pack 后总 token 数须被 `cp_size` 整除、每条序列长度须被 `2*cp_size` 整除（`routes.py:41-60`）。

至此 16 篇走完：从并行拓扑（TP/PP/SP）到重计算、优化器、checkpoint、数据集、fused 算子、强化学习、MoE、多模态，再到本篇的 Context Parallel，覆盖了 Megatron-LM 训练框架的一条主干。CP 这一篇的意图其实也是整个系列的意图——**并行度越高，越多的「正确性」要搬到通信和布局上，读懂源码就是读懂这些 trade-off 落在哪一行。**

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/transformer/dot_product_attention_context_parallel.py`、`megatron/core/context_parallel_layout/routes.py`、`megatron/core/context_parallel_layout/conversion.py`、`megatron/core/context_parallel_layout/types.py`。）
