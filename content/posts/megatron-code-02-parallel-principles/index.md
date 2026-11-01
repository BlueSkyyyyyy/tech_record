---
title: "Megatron 源码精读（二）：模型并行的原理——TP/SP/PP/CP/DP/FSDP"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "分布式"]
categories: ["训练框架"]
weight: 2
series: ["megatron-code"]
---

第 1 篇《[整体代码结构与启动链路]({{< relref "megatron-code-01-structure" >}})》说过，并行能力分散在 `parallel_state.py`、`tensor_parallel/`、`pipeline_parallel/`、`distributed/` 几个目录里。这一篇只回答一个问题：**每一种并行到底切开了什么、省下了什么、又付出了多少通信。**

一句话结论：**五种切分各管一个维度——DP 切样本、TP 切算子、SP 切序列上的逐元素层、PP 切层、CP 切 attention 的序列；ZeRO/FSDP 则在 DP 轴上继续切优化器状态、梯度和参数。** 2026 年的上游又加了两个正交补充（GTP 与 dynamic CP），见文末版本提示。

## 1. 一张表：切什么、省什么、通信多少

| 并行 | 切分对象 | 省的是 | 每步通信 |
|---|---|---|---|
| DP | batch（样本） | 计算时间 | 梯度 all-reduce / reduce-scatter，与参数量成正比 |
| TP | 矩阵算子 | 权重 + 优化器状态 | 每层多次 all-reduce（attention 与 MLP 各两次），与 `batch×seq×hidden` 成正比 |
| SP | 序列（非 attention 的逐元素层） | **激活** | 每层 all-gather + reduce-scatter，与 TP 同量级 |
| PP | 层 | 每卡层数 | 每 microbatch 一次 P2P，几乎与层数无关 |
| CP | 序列（attention） | 长上下文激活 | 按 KV 块 all-gather，总量随 cp 增长 |
| ZeRO/FSDP | 优化器状态 → 梯度 → 参数 | 优化器/梯度/参数显存 | all-gather / reduce-scatter，与参数量成正比 |

读表要抓两点：DP 与 FSDP 的差别不在「切不切」，而在切到第几层；SP 与 CP 都切序列，但 SP 是拿 all-gather / reduce-scatter 换激活显存，CP 是拿 KV 交换换长上下文——后者是第 4 节的主线。

## 2. DP：复制模型，把梯度求平均

DP 让每个 rank 持有一份完整模型、各算不同样本，再把梯度平均：

$$
g = \frac{1}{P}\sum_{i=1}^{P} g_i
$$

这就是 all-reduce。坑在默认语义：**NCCL 的 all-reduce 是 SUM，不是 AVG**。所以平均要么让集合通信自己算（`average_in_collective=True` 时用 `ReduceOp.AVG`），要么提前把 $1/P$ 乘进 `gradient_scaling_factor`。`param_and_grad_buffer.py:712-723` 两条路都在，注释写明「这个系数已经把求平均还是求和考虑进去了」。

工程上，DP 通信不挂在 forward 末尾，而是挂在 backward 钩子上、按 bucket 触发：`DistributedDataParallel` 给每个 `requires_grad` 参数注册 post-hook，梯度先累进 `main_grad`，只有最后一个 microbatch 的梯度就绪才真正派发。

发什么由 `start_grad_sync` 二选一：普通 DDP 走 `all_reduce`，开 distributed optimizer 则走 reduce-scatter，每个 rank 只留自己那份 grad shard。这个分支就是后文 ZeRO 的分水岭。

## 3. TP：列并行 + 行并行，中间只差一次 all-reduce

模型大到一份参数都放不下时，DP 复制更多份只会更糟。TP 改切算子：权重沿一维切成 $p$ 份，每卡只存一片。

**列并行**沿输出维切，各卡拿完整输入算输出的列块：

$$
Y = XA = \bigl[\,XA_1,\ XA_2,\ \dots,\ XA_p\,\bigr]
$$

**行并行**沿输入维切，各卡算部分和，再一次 all-reduce 求和：

$$
Y = \sum_{i=1}^{p} X_i A_i
$$

两块拼起来，一层里无论是 QKV/输出投影还是 MLP 的两个 GEMM，都是「列并行 → 行并行 → 一次 all-reduce」。

代码落点是 `ColumnParallelLinear`（`partition_dim=0`，权重 `[out/tp, in]`）与 `RowParallelLinear`（`partition_dim=1`，权重 `[out, in/tp]`）。前者的输入靠 `copy_to_tensor_model_parallel_region` 补齐：**前向是恒等 copy，反向才 all-reduce**；后者的输出走 `reduce_from_tensor_model_parallel_region`，前向 all-reduce、反向恒等。

通信量只与激活大小 $B \times S \times H$ 成正比，与权重无关——这是 TP 便宜的原因，也是它必须待在 NVLink 域内的原因。`initialize_model_parallel` 的 docstring 特意提醒：相邻 rank 要落在同一台 DGX。

## 4. SP 与 CP：都切序列，通信完全不同

### 4.1 SP 省的是激活，不是通信

LayerNorm、Dropout 都是逐 token 运算，不跨 token 交互，所以激活可以沿 seq 切成 $[s/p, b, h]$，省下 $p$ 倍激活显存。但它省不了通信：

- 进 column-parallel GEMM 前，必须 all-gather 回完整序列（`linear_with_grad_accumulation_and_async_allreduce` 的 `sequence_parallel` 分支；权重冻结时走 `linear_with_frozen_weight`）；
- RowParallel 之后，再把部分和 reduce-scatter 回 $[s/p, b, h]$，供下一层 LayerNorm 使用。

于是 SP 每层的前向、反向各多一轮 all-gather 与一轮 reduce-scatter，量与 TP 的 all-reduce 同量级。它换的是显存，不是带宽。把 SP 写成「无通信的序列切分」，是这类文章最常见的错。

第二个易错点：**SP 没有自己的进程组**。`gather_from_sequence_parallel_region`、`reduce_scatter_to_sequence_parallel_region` 内部都调 `get_tensor_model_parallel_group_if_none`，复用的就是 TP 组。开关只有一个 `config.sequence_parallel`；打开后 `allreduce_dgrad` 恒为 False（两者同时为真会直接报错），同一笔输入梯度通信改由 reduce-scatter 承担。

### 4.2 CP 切的是 attention，必须交换 KV

Attention 天生跨 token：softmax 分母要对所有 key 求和。CP 把序列切给多个 rank，每个 rank 只算自己那段 query，但要看全量 KV。

基线的做法是 all-gather KV：`dot_product_attention.py` 在 `context_parallel_size > 1` 时分派给 `dot_product_attention_context_parallel.py`，后者按 KV head 分块，用 `AllGatherComm` 流水化地把各 rank 的 K/V 拼起来，再算本地 query 段的注意力。

**它不是 ring attention**——ring 是 KV 块在 rank 间 P2P 轮转，这里是 all-gather，别混为一谈。

CP 有独立的组：`get_context_parallel_group` / `get_context_parallel_world_size`（`parallel_state.py:1518`）。它与 SP 唯一的共同点是「都切序列」：SP 省激活、复用 TP 组；CP 省长上下文显存、自成一组。

## 5. PP：切层，通信最省、但有气泡

PP 按层切：$L$ 层分给 $P$ 个 stage，每 stage 只存 $L/P$ 层。相邻 stage 每 microbatch 只传一次激活、反向传一次梯度，通信量与层数无关。

代价是气泡：不调度时同一时刻只有一个 stage 在算，利用率约 $1/P$。1F1B 让每个 stage 做一次前向就跟一次反向，把「在飞」的 microbatch 数压到与流水线深度同阶。

配了 `num_microbatches_with_partial_activation_checkpoints` 时，代码才把上界设成 `max_outstanding_backprops = num_warmup_microbatches + 1`（否则为 `None`，即全部保存）；稳态循环用 `forward_k % max_outstanding_backprops` 决定当前 microbatch 是全量缓存激活还是重算，`num_microbatches_with_partial_activation_checkpoints` 控制最多几份全量激活同时在飞。

三段结构在 `forward_backward_pipelining_with_interleaving` 里：warmup 只前向、steady 走 1F1B、cooldown 只反向。stage 间走 `P2PCommunicator`；`_p2p_ops` 让偶/奇 rank 的发收顺序相反，并且在 PP 组只有 2 个 rank 时把一半通信挪到全局组上，好让两条独立的 P2P 真正重叠。

## 6. ZeRO/FSDP：在 DP 轴上继续切

训练显存的大头常常不是参数，而是 Adam 的 $m$、$v$。ZeRO 分三档：优化器状态（ZeRO-1）、再加梯度（ZeRO-2）、再加参数（ZeRO-3，即 FSDP）。第 2 节那个 RS/AR 分支就是入口——`use_distributed_optimizer` 为真时走 reduce-scatter，每个 rank 只保留自己的 shard。

基线 commit 里有三条并列的路线，选择在 `_ddp_wrap`（`megatron/training/models/dist_utils.py`）里：

| 路线 | 开关 | 做法 |
|---|---|---|
| `DistributedDataParallel` | 默认 | 参数全量复制，只切梯度通信；配 distributed optimizer 时优化器状态分片 |
| `FullyShardedDataParallel`（Megatron-FSDP） | `use_megatron_fsdp` | Megatron 自研，`data_parallel_sharding_strategy` 四档：`no_shard` / `optim` / `optim_grads` / `optim_grads_params` |
| `TorchFullyShardedDataParallel` | `use_torch_fsdp2` | 封装 PyTorch FSDP2 的 `fully_shard`，`reshard_after_forward` 默认 True |

**Megatron 的 FSDP 不等于 PyTorch FSDP2 的封装**：只有第三条是封装，Megatron-FSDP 是另一套自研的分片实现。细节留第 8 篇。

## 7. 小结

- **DP 是基线**：复制模型、平均梯度，通信量与参数量成正比；平均靠 `ReduceOp.AVG` 或预先缩放的 `gradient_scaling_factor`。
- **TP 切算子**：列并行接行并行，中间一次 all-reduce，通信量只与激活大小成正比，所以被限制在 NVLink 域内。
- **SP 切逐元素层的序列**：省激活显存，但每层的 all-gather / reduce-scatter 一样不少，复用 TP 组。
- **CP 切 attention 的序列**：必须 all-gather KV，有自己的组，和 SP 不是一回事。
- **PP 切层**：每个 microbatch 边界一次 P2P，代价是气泡，靠 1F1B 与部分激活重算压住。
- **ZeRO/FSDP 切优化器/梯度/参数**：基线三条路线，Megatron-FSDP 与 Torch FSDP2 是两套实现。

---
**版本提示（2026-10）**：上游新增两个正交补充。其一是 **GTP**（`tensor_parallel/generalized_tensor_parallelism.py`）：把权重并行域拆成 TP × GTP_remat，GTP_remat 沿模型并行轴做 ZeRO-3 式的「单权重 all-gather → 用 → 释放 → wgrad reduce-scatter」，需 TE ≥ 2.19，开关 `--tensor-parallel-num-weight-shards`。其二是 **dynamic CP**（`context_parallel/groups.py`）：CP 组不再固定，可按 microbatch 由 sequence packing scheduler 选择。于是并行的说法从「五种」变成「五种 + 两个正交补充」。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
