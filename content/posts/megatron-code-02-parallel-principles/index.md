---
title: "Megatron 源码精读（二）：模型并行的原理——TP/SP/PP/CP/DP/FSDP"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "分布式"]
categories: ["训练框架"]
weight: 2
series: ["megatron-code"]
---

这是「Megatron-LM 源码精读」系列的第二篇。第 1 篇《[整体代码结构与启动链路]({{< relref "megatron-code-01-structure" >}})》里说并行能力「分散在 `parallel_state.py`、`tensor_parallel/`、`pipeline_parallel/`、`distributed/`」，本篇把这几种并行**各自在做什么、为什么这么做、数学上长什么样**讲清楚——即「原理」这一层。下一篇《[并行拓扑]({{< relref "megatron-code-03-parallel-topology" >}})》再回答这些并行「在真实 rank 网格里如何排布」。

一句话先给结论：**Megatron 的「并行」不是一种，而是五种正交维度在叠加**——数据并行（DP）复制模型，张量并行（TP）切算子，流水线并行（PP）切层，上下文并行（CP）切序列长度，ZeRO/FSDP 切优化器/梯度和参数。理解它们的钥匙是「**每种并行切的是什么、省的是什么、通信代价是什么**」。

---

## 1. 总览：一张表先立起来

| 并行 | 切分对象 | 省的是 | 通信 | 每步通信量 |
|---|---|---|---|---|
| DP | batch（数据） | 计算时间 | 梯度 all-reduce / reduce-scatter | 与**模型大小**成正比，与 batch 无关 |
| TP | 矩阵算子 | 权重 + 优化器状态 | 每层多次 all-reduce / all-gather | 与 `batch × seq × hidden` 成正比 |
| SP | 序列维（非注意力的逐元素层） | **激活**显存 | all-gather / reduce-scatter（沿 seq） | 与 TP 同量级、分得更细 |
| PP | 层 | 每卡的层数 | P2P send/recv（激活/梯度） | 每 microbatch 边界一次，几乎与层数无关 |
| CP | 序列长度（attention） | 长上下文显存 | all-gather KV（ring attention） | 与 seq 切分份数有关 |
| FSDP/ZeRO | 优化器状态 → 梯度 → 参数 | 优化器/梯度/参数显存 | all-gather / reduce-scatter | 与模型大小成正比 |

这张表是全文的地图。下面按「为什么需要 → 数学 → 代码落点」的顺序，逐维拆开。

---

## 2. DP：最朴素、也必须先立住的基线

**动机**：一张卡放不下一个大 batch，或者想用多张卡并行算多个样本。DP 让每个 rank 持有一份**完全相同的模型**，各算各的 `batch_i`，最后把梯度一平均。

**数学**：训练用的是全局平均梯度。设世界大小为 `P`，第 `i` 个 rank 的局部梯度为 $g_i = \nabla_\theta L(\theta; batch_i)$，则

$$
g = \frac{1}{P} \sum_{i=1}^{P} g_i
$$

这就是 `all_reduce`。Megatron 在这里有一个容易忽略的细节：**PyTorch 的 NCCL 后端 `all_reduce` 默认是 SUM 而不是 AVG**，所以框架要把「1/P」这个系数显式写进缩放因子。`param_and_grad_buffer.py:720-723` 里：

```python
reduce_op = ReduceOp.SUM
if ddp_config.average_in_collective:
    reduce_op = ReduceOp.AVG
```

而梯度缓冲在此之前先乘了 `gradient_scaling_factor`（`param_and_grad_buffer.py:716-718`），注释明确说这个系数「已经把平均还是求和考虑进去了」。这是 DP 的数学核心，其余都是工程加速。

**代码落点**：DP 的通信不在 `forward` 末尾一次性触发，而是挂在 backward 钩子上、按 bucket 粒度触发。`DistributedDataParallel.__init__` 给每个 `requires_grad` 参数注册钩子（`distributed_data_parallel.py:424`），钩子内部（`_make_backward_post_hook`，`distributed_data_parallel.py:500-529`）先把 `param.grad` 累加进 `param.main_grad`，再在 `overlap_grad_reduce` 时调 `register_grad_ready`（`param_and_grad_buffer.py:913`）——只有最后一个 microbatch 的梯度就绪时才真正派发通信。真正的通信在 `start_grad_sync`（`param_and_grad_buffer.py:651`）里二选一：

- 普通 DDP → `torch.distributed.all_reduce`（`param_and_grad_buffer.py:781-783`）
- distributed optimizer（ZeRO）→ `dist_reduce_scatter_func`（`param_and_grad_buffer.py:769-775`）

这个「RS 还是 AR」的分支，正是后文 ZeRO-1/2 的分水岭，注释直接写在 `distributed_data_parallel_config.py:29` 的 `use_distributed_optimizer` docstring 上。DP 是基线：它的通信量与模型大小成正比（每个参数都要 reduce 一份梯度），与 batch 大小无关。

---

## 3. TP：把矩阵乘法切成「列并行 + 行并行」

DP 有一堵墙：模型太大时，单卡连**一份参数 + 一份优化器状态**都放不下，DP 复制 N 份只是更放不下。TP 的思路是**把算子本身切开**，让每张卡只持有权重矩阵的一块。

### 3.1 核心观察：两次 GEMM 中间只差一次 all-reduce

一个 `Y = X A + b` 的矩阵乘，只要把权重沿某一维切成 `p` 份，`p` 张 GPU 各算局部结果，再配一次集合通信还原。

**列并行（ColumnParallel）**：权重 $A$ 沿**输出维**切成 $p$ 段。每张卡用自己的 $A_i$ 与**完整输入** $X$ 相乘，得到输出的一块列：

$$
Y = X A = \bigl[\, X A_1,\ X A_2,\ \dots,\ X A_p \,\bigr],
\qquad A = [A_1\ A_2\ \cdots\ A_p]
$$

**行并行（RowParallel）**：上一层列并行的输出已经是按列切分的 $X = [X_1, \dots, X_p]$，把权重 $A$ 沿**输入维**切成 $p$ 段，各卡算局部和，再 all-reduce：

$$
Y = X A = \sum_{i=1}^{p} X_i A_i,
\qquad A = \begin{bmatrix} A_1 \\ A_2 \\ \vdots \\ A_p \end{bmatrix}
$$

两种并在一起，中间**只差一次 all-reduce**。这就是 TP 通信成本低的根本原因：通信量与「激活张量大小」成正比（$batch \times seq \times hidden$），而不是与权重大小成正比。

### 3.2 代码落点：权重怎么切、标志位怎么走

`layers.py` 里两类线性层的权重切分方式：

- `ColumnParallelLinear`：`output_size_per_partition = divide(output_size, world_size)`，权重形状 `[output_size_per_partition, input_size]`，`partition_dim=0`（沿输出维切，`layers.py:886`、`layers.py:916-922`）。
- `RowParallelLinear`：`input_size_per_partition = divide(input_size, world_size)`，权重形状 `[output_size, input_size_per_partition]`，`partition_dim=1`（沿输入维切，`layers.py:1236`、`layers.py:1267-1273`）。

两个标志位串起前向的数据流：

- `input_is_parallel=True`（RowParallel，`layers.py:1326-1327`）：输入已经是列并行输出（已按列切分在各 rank），无需再 scatter。
- `gather_output` / `output_is_parallel`（ColumnParallel，`layers.py:1091-1105`）：列并行的局部输出是否 all-gather 拼回完整 hidden。不 gather（`gather_output=False`）就把切分结果直接交给下一层行并行消费。

`ColumnParallelLinear.forward`（`layers.py:1000-1107`）的调用顺序是：**copy 区域 → 局部 GEMM →（可选）all-gather**；`RowParallelLinear.forward`（`layers.py:1314-1370`）是：**（可选）scatter → 局部 GEMM → all-reduce / reduce-scatter → bias**。其中的通信原语都收在 `mappings.py`：

- `_CopyToModelParallelRegion`（`mappings.py:201-218`）：**前向是恒等 copy（无通信），反向才是 all-reduce**。用于列并行 GEMM 前广播完整输入 $X$——前向本该每卡都拿完整 $X$，反向却要把 $dX$ 汇总回完整梯度。
- `_ReduceFromModelParallelRegion`（`mappings.py:221-237`）：**前向 all-reduce**（`_reduce`，`mappings.py:22-37`），反向恒等。用于行并行累加 $\sum_i X_i A_i$。

### 3.3 TP 的通信成本

每个 Transformer block 里，attention 的 QKV/输出投影、MLP 的两个 GEMM 都要各来一轮 all-reduce。通信量与 `batch × seq × hidden` 成正比、与层数成正比。这也是为什么 TP 通常**只用在单机 NVLink 域内**（TP 度数一般 ≤ 8）：它要求极低的卡间延迟和高带宽。

---

## 4. SP：TP 的顺风车，专省「激活显存」

TP 省的是**权重 + 优化器状态**的显存，但**激活**没省——每张卡都存一份完整的 $[s, b, h]$ 激活。SP 补的就是这一块。

**动机**：LayerNorm 和 Dropout 是「跨其他维共享的逐元素/归约」运算，**不涉及不同 token 之间的交互**，因此可以让每个 rank 只持有 `seq` 维的一部分 $[s/p, b, h]$，把这类层的激活显存按 TP 度数缩小 $p$ 倍；而在真正需要完整 hidden 的 GEMM 之前，再 all-gather 恢复完整序列。

**代码落点**：SP 不是一个独立进程组，而是一个配置开关 `config.sequence_parallel`（体现在 `layers.py:958`、`layers.py:1212`）。它改变的是两个映射函数的路径——同一套 `scatter`/`gather` 原语，SP 沿**第一维（seq）**，TP 沿**最后一维（hidden）**：

- `_ScatterToSequenceParallelRegion`（`mappings.py:280-297`）沿第一维切分，反向沿第一维 gather；对比 `_ScatterToModelParallelRegion`（`mappings.py:240-257`）沿最后一维。
- `_GatherFromSequenceParallelRegion`（`mappings.py:300-352`）前向沿第一维 gather，反向沿第一维 reduce-scatter。

具体到 data 流：列并行 GEMM 之前，`sequence_parallel` 开启时先 `gather_from_sequence_parallel_region` 把输入从 $[s/p, b, h]$ all-gather 回 $[s, b, h]$（`layers.py:459-463`）；算完之后由 RowParallel 的 `reduce_scatter_to_sequence_parallel_region`（`layers.py:1358-1361`）沿 seq 维 reduce-scatter，把残差送回局部 seq，供下一层 LayerNorm（SP 模式）使用。

**一句话**：SP 是「无通信的序列切分」用于省激活显存。注意它和 CP 不是一回事——见第 6 节。

---

## 5. PP：切层，通信最省、但有气泡

TP 的通信量随层数线性增长，模型一大、层一多，光 all-reduce 就吃不消。PP 换个维度：**按层切**。

### 5.1 数学与动机

设模型共 $L$ 层，切成 $P$ 个 stage，每个 stage 持有 $L/P$ 层。前向第 $p$ 个 stage 收到上一层输出 $a^{(p-1)}$，算完发 $a^{(p)}$；反向梯度沿反方向流动：

- 前向：$a^{(p)} = f_p(a^{(p-1)})$，边界 $a^{(0)} = x$
- 反向：$\dfrac{\partial L}{\partial a^{(p-1)}} = \bigl(\tfrac{\partial f_p}{\partial a^{(p-1)}}\bigr)^T \cdot \dfrac{\partial L}{\partial a^{(p)}}$

关键收益：相邻 stage 之间**只传一次激活（前向）和一次梯度（反向）**，通信量几乎与层数无关。这是 PP 「通信最省」的由来。

代价是 **pipeline bubble（气泡）**：如果没有调度优化，任一时刻只有一个 stage 在算，其余 $P-1$ 个都闲着，利用率掉到约 $1/P$。

### 5.2 1F1B：把气泡压下去

Megatron 用 1F1B（one-forward-one-backward）调度缓解气泡。核心入口 `forward_backward_pipelining_with_interleaving`（`schedules.py:1098`），把一次 step 分成三段：

1. **Warmup（预热）**：只做前向，把流水线「灌满」。循环 `for k in range(num_warmup_microbatches)`（`schedules.py:1675`）。
2. **Steady-state（稳态）**：每 stage 做一次前向、紧接着做一次反向，交错进行（1F1B 得名）。循环在 `schedules.py:1835`。
3. **Cooldown（收尾）**：只做反向，把流水线「排空」，只遍历 backward graph（`schedules.py:1088`）。

1F1B 的显存收益在于：把「同时在飞的 microbatch 数」从 $P$ 降到约 $P/2$，配合 activation checkpointing 时，允许按 `forward_k % max_outstanding_backprops` 决定哪些 microbatach缓存激活（`schedules.py:1840-1844`），把激活显存瓶颈从「层数 × microbatch 数」压到近似「层数 × 1」。

### 5.3 p2p 通信

stage 之间的激活/梯度走点对点通信，封装在 `P2PCommunicator`（`p2p_communication.py:140`）。两个值得记住的细节：

- **奇偶 rank 错峰**：`_p2p_ops`（`p2p_communication.py:55`）让偶数 rank 先发 next 再发 prev、奇数 rank 反过来，避免双向 head-of-line 阻塞（`p2p_communication.py:79-128`）。
- **高层语义**：`send_forward`/`recv_forward`（`p2p_communication.py:507`/`:445`）、`send_backward`/`recv_backward`（`:528`/`:476`），以及带通信重叠的 `send_forward_recv_forward`（`:614`，返回 wait handle 供调度器延迟等待）。

首尾 stage 的边界条件由 `is_pipeline_first_stage`（`parallel_state.py:1679`）与 `is_pipeline_last_stage`（`parallel_state.py:1689`）判定：首 stage 不 recv 前向激活、末 stage 不发前向激活/不收反向梯度。

---

## 6. CP：长上下文的「必需通信」序列并行

SP 能切序列，是因为 LayerNorm/Dropout 不跨 token 交互。但 **attention 恰恰跨 token 交互**：softmax 的分母需要对 $Q$ 与**所有** $K$ 的点积求和。要把超长序列的注意力摊到多卡，就必须通信。

**与 SP 的联系与区别**：

- **SP（Sequence Parallel）**：切的是 attention 之外的逐元素层，**无跨 token 交互 → 无需通信**，目标是省激活显存。它常与 TP 联合使用。
- **CP（Context Parallel）**：切的正是 attention 本身，需显式 all-gather KV（等效于 ring attention）。目标是支持超长上下文。

两者进程组状态同名，都叫 context parallel：`get_context_parallel_group`（`parallel_state.py:1518`）、`get_context_parallel_world_size`（`parallel_state.py:1837`）。CP 的 attention 走 `AttentionFunctionWithContextParallel`（`dot_product_attention.py:184`，条件 `context_parallel_size > 1`），底层在 `dot_product_attention_context_parallel.py:109-121` 用流水线化的 **all-gather KV** 实现「每块只算自己的 query 段、KV 按需 all-gather」的 ring attention。

CP 与 TP 里那个 Sequence Parallel 是两码事：一个省激活、一个省上下文显存，注意别混。CP 单独深入会放到第 16 篇。

---

## 7. FSDP / ZeRO：从「切算子」到「切优化器/梯度/参数」

DP 复制整份模型，显存反而翻倍；TP/PP 切模型但各有通信代价。ZeRO 家族换一个角度：**训练显存的大头其实不是参数本身，而是优化器状态（Adam 的 $m$、$v$）**。于是有了三阶段：

| 阶段 | 分片内容 | 参数副本 | 通信变化 |
|---|---|---|---|
| ZeRO-1 | 优化器状态（$m$/$v$） | 全量 | 梯度仍 all-reduce |
| ZeRO-2 | 优化器状态 + **梯度** | 全量 | all-reduce → **reduce-scatter** |
| ZeRO-3（=FSDP） | 优化器状态 + 梯度 + **参数** | 分片 | all-gather / reduce-scatter 按需 |

这对应第 2 节 `start_grad_sync` 里的那个二选一：`use_distributed_optimizer` 为真时走 reduce-scatter（ZeRO-2 及以上，每个 rank 只保留自己那一份 `grad_data` shard，见 `param_and_grad_buffer.py:763-775`），否则 all-reduce（ZeRO-1/DDP）。

Megatron 的 FSDP 是对 PyTorch FSDP2（`torch.distributed.fsdp.fully_shard`）的封装，实现在 `torch_fully_sharded_data_parallel.py`：用 `DeviceMesh.from_group` 把 DP 进程组变成 1D mesh（`torch_fully_sharded_data_parallel.py:81`），`reshard_after_forward` 默认 `True`（配置 `torch_fully_sharded_data_parallel_config.py:13`），逐个子模块 `fully_shard`（含 `TransformerLayer`、`LanguageModelEmbedding`、`ColumnParallelLinear` 等，`torch_fully_sharded_data_parallel.py:126-146`）。参数、梯度、优化器状态全部分片，forward 时 all-gather、结束时 reshard，内存占用最小。详细的 ZeRO-1/FSDP 实现留给第 8 篇。

---

## 8. 五种并行的分工：一起用时的画面

回到开头那张表的关键补充——它们**作用在不同维度、可以叠加**。一个典型的大模型训练配置同时开着 DP、TP、PP、CP：

- **DP** 决定「同一份模型复制几份，各喂不同样本」——梯度 all-reduce 把全局样本梯度聚合起来；
- **TP** 决定「一层内的算子切成几块」——每层的 GEMM 靠 all-reduce 拼回完整结果；
- **PP** 决定「模型切成几段」——段间用 send/recv 传激活；
- **CP** 决定「一条序列切成几段」——attention 里 all-gather KV；
- **FSDP/ZeRO** 决定「每卡存完整模型还是分片」——省优化器/梯度/参数显存。

`parallel_state.py` 里这几个组独立建立：`_DATA_PARALLEL_GROUP`（`parallel_state.py:964`）、`_TENSOR_AND_DATA_PARALLEL_GROUP`（`parallel_state.py:1168`，TP×DP 交叉组，用于 MoE 路由 expert bias 更新等）。梯度收尾 `finalize_model_grads`（`finalize_model_grads.py:454`）的顺序正好体现了三者的协作：先 DP 侧 `finish_grad_sync`（`finalize_model_grads.py:503-507`），再 TP/SP 侧的 layernorm 等非张量并行模块梯度 all-reduce（`finalize_model_grads.py:528`），最后对 embedding 做跨 PP 段的 all-reduce（`finalize_model_grads.py:537-538`）。

---

## 9. 小结

- **DP 是基线**：复制模型、平均梯度，通信量与模型大小成正比；「平均」靠 `average_in_collective` 显式把 1/P 写进缩放因子。
- **TP 切算子**：列并行 + 行并行中间只差一次 all-reduce，通信量与 `batch × seq × hidden` 成正比，靠 NVLink 高带宽撑起，通常限定在单机。
- **SP 切序列的逐元素层**：无跨 token 交互、无需额外通信，专省激活显存，是 TP 的顺风车。
- **PP 切层**：通信最省（每个 microbatch 边界只传一次激活/梯度），代价是 pipeline bubble，靠 1F1B 调度缓解。
- **CP 切 attention 的序列**：需 all-gather KV（ring attention），支持超长上下文。
- **ZeRO/FSDP 切优化器/梯度/参数**：训练显存大头其实是优化器状态，三阶段依次多省一块，FSDP 省到极致。

## 10. 下一篇预告

下一篇《[并行拓扑：parallel_state.py 精读]({{< relref "megatron-code-03-parallel-topology" >}})》回答一个更具体的问题：这五种并行在真实多卡集群里，如何用一行 `order` 字符串 + 混合进制分解，把「全局 rank 编号」映射到一张张 GPU 上。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应 `megatron/core/` 下的 `tensor_parallel/`、`pipeline_parallel/`、`distributed/`、`parallel_state.py`。）
