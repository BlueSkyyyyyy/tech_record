---
title: "Megatron 源码精读（二十一）：TP 通信原语与并行线性层"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "张量并行", "tensor-parallel", "collective", "线性层"]
categories: ["训练框架"]
weight: 21
series: ["megatron-code"]
---

上一篇《[通信与计算 overlap]({{< relref "megatron-code-20-overlap" >}})》讲怎么把通信藏进计算，但通信本身长什么样、由谁发起，还散在书缝里。第 3 篇《[并行拓扑]({{< relref "megatron-code-03-parallel-topology" >}})》只讲了 rank 和进程组怎么组织，没有展开真正干活的那批原语。

这一篇补上这块：`tensor_parallel/mappings.py`（集合通信原语）、`layers.py`（并行线性层与词表嵌入）、`cross_entropy.py`（词表并行 loss）。

一句话结论：**TP 的执行层只有一个套路——把每个集合通信包成 `torch.autograd.Function`，让 forward 和 backward 互为对偶；并行线性层则把「一次通信 + 一次 GEMM」拼成一层。**

## 1. 问题：切与合，为什么必须包 autograd

TP 的操作只有两类：切（split/scatter）与合（gather/reduce）。裸调 `torch.distributed` 的麻烦在于反向不会自动跟着走——forward 做了 all-gather，backward 就得做对应的 reduce-scatter，否则梯度的形状和语义都不对。

所以 Megatron 给每个通信配一个 `autograd.Function`：forward 做通信，backward 写它的对偶。后面看到的「成对互逆」都是这条规则的产物。

## 2. 原语层：`mappings.py`

### 2.1 最底下是 `_reduce` 和五个 helper

`_reduce` 是 all-reduce 的薄封装：`world_size == 1` 直接返回（单卡 TP 没有通信），否则 `contiguous()` 后 all-reduce。注释提醒：输入本来就 contiguous 时会被**原地改写**，调用方必须用返回值承接。

五个 helper 按「沿哪一维」分工：

| helper | 维 | 动作 |
|---|---|---|
| `_split_along_last_dim` | last | 切完取本 rank 那份 |
| `_split_along_first_dim` | first | 切之前断言第一维可整除 |
| `_gather_along_last_dim` | last | all-gather 后按 chunk 沿 last 拼 |
| `_gather_along_first_dim` | first | all-gather，支持不等长 `output_split_sizes` |
| `_reduce_scatter_along_first_dim` | first | reduce-scatter，支持 `input_split_sizes` |

`_reduce_scatter_along_last_dim` 没有独立实现：它把 last 维 reshape 成 `(-1, H)`、按 hidden 切块 cat 到第一维，复用 first-dim 版本再 reshape 回去。这样「沿最后维」的 reduce-scatter 不用重写，统一落到同一份实现上。

### 2.2 六个 autograd 原语，两两互逆

- `_CopyToModelParallelRegion`：forward 恒等，backward all-reduce。
- `_ReduceFromModelParallelRegion`：forward all-reduce，backward 恒等。
- `_ScatterToModelParallelRegion`：forward 沿 last split，backward 沿 last gather。
- `_GatherFromModelParallelRegion`：反过来，forward gather、backward split。
- `_ScatterToSequenceParallelRegion`：forward 沿 **first（seq）** split，backward gather。
- `_GatherFromSequenceParallelRegion`：forward 沿 first gather，backward 见下。

对外的 wrapper（`copy_to_tensor_model_parallel_region`、`gather_from_sequence_parallel_region` 等）只做两件事：用 `get_tensor_model_parallel_group_if_none` 取默认组，再 `.apply()`。单卡时原语内部自己旁路，wrapper 不用判。

### 2.3 `_GatherFromSequenceParallelRegion` 的 backward 走哪支

它的 backward（`mappings.py:332`）按 `tensor_parallel_output_grad` 二选一：

- 为 `True`：下游还是 TP，每个 rank 只消费自己那份，走 reduce-scatter；
- 为 `False`：下游是复制计算，各 rank 只需自己那段梯度，走 split（此时断言 `output_split_sizes is None`）。

这不是性能取舍，而是语义：**下游是不是 TP，决定了梯度该 reduce 还是该切**。写错这一支，loss 不报错，只是收敛变慢。

### 2.4 `_AllToAll` 与 sp2hp/hp2sp

`_AllToAll` 的 forward 是 `all_to_all_single`（支持不等长 split）；backward 把 `output_split_sizes` 与 `input_split_sizes` **对调**再交换一次——all-to-all 的对偶就是反方向的 all-to-all。

`all_to_all_sp2hp` / `all_to_all_hp2sp` 用它做布局转换，形状别记反：

- `sp2hp`：`[tokens/tp, H]` → 沿 H 切 tp 块、cat 到第一维 → all-to-all → `[tokens, H/tp]`。
- `hp2sp`：`[tokens, H/tp]` → all-to-all → 沿第一维切 tp 块、cat 到最后一维 → `[tokens/tp, H]`。

### 2.5 异步原语

`async_gather_from_sequence_parallel_region` 与 `async_reduce_scatter_along_first_dim` 返回 `_AsyncCollectiveHandle`，`wait()` 在首个消费者处才同步。handle 里有个 `_input_buffer` 字段，专门让输入张量活到 wait：NCCL 可能还没读完输入，提前释放就是随机崩溃。

## 3. 并行线性层：一次通信，正反各一次

### 3.1 `ColumnParallelLinear`：切输出维

权重 `[out/tp, in]`、`partition_dim=0`。forward 三件事：

1. 决定输入走哪条路：只有 `allreduce_dgrad`、`sequence_parallel`、`explicit_expert_comm`、`disable_grad_reduce` **全关**时才过一遍 copy 原语（前向恒等、反向 all-reduce）；否则输入已是各 rank 该有的那份。
2. 调 `linear_with_grad_accumulation_and_async_allreduce` 做 GEMM。
3. `gather_output=True` 时 all-gather 拼回完整 `Y`，否则把 `Y_i` 留给下游 `RowParallelLinear`。

`allreduce_dgrad` 的定义是 `world_size > 1 and not sequence_parallel and not disable_grad_reduce`——它与 SP 互斥，构造时直接报错。

### 3.2 `RowParallelLinear`：切输入维

权重 `[out, in/tp]`、`partition_dim=1`；输入按 last 维切（`input_is_parallel=False` 时才 scatter）。GEMM 得到的是**部分和**，输出三选一：专家并行时原样返回；SP 时 `reduce_scatter_to_sequence_parallel_region`；否则 `reduce_from_tensor_model_parallel_region` 做全量 all-reduce。

列并行 + 行并行成对出现：前者的列切分正好被后者的归约抵消。

### 3.3 权重初始化：先建全量，再各取一片

`_initialize_affine_weight_cpu`（`layers.py:158`）的做法是：每个 rank 都先建完整 master 权重、用同一个 `init_method` 初始化，再沿 `partition_dim` 切块，各 rank 取 `weight_list[rank::world_size]` 这些块、cat 起来拷进自己的 `weight`（有 stride 时块更小）。GPU 侧 `_initialize_affine_weight_gpu` 则靠 `get_cuda_rng_tracker().fork()` 切 RNG 流。

两条路的共同目标只有一个：**不同 rank 上拼出来的分片必须来自同一份权重**。否则 TP 的数学从一开始就不成立——这是「初始化看似无关紧要、实则是并行正确性前提」的典型例子。

## 4. 异步 backward：三段流水

### 4.1 forward 的 SP all-gather

`sequence_parallel=True` 时，forward 得先把输入拼回完整序列。它不新分配输出，而是从 `get_global_memory_buffer()` 取一块 `"mpu"` buffer 承接 all-gather 结果，再 matmul——省一次分配，buffer 还能跨层复用。

### 4.2 backward 的三段顺序

backward（`layers.py:522`）的原则是「先发通信、再算本地、最后统一 wait」：

1. **重 gather 输入（异步）**：SP 下 `async_op=True` 发起 all-gather，紧接着先算 `grad_input = grad_output @ weight`，让 gather 在飞。
2. **input 梯度 all-reduce（异步）**：非 SP 且 `allreduce_dgrad` 时发起异步 all-reduce。
3. **SP 的 reduce-scatter（异步）**：SP 时把 `grad_input` reduce-scatter 回 seq 分片。

之后才算 weight 梯度，最后统一 `wait()`。顺序不能乱：每段通信都要赶在 wgrad GEMM 之前下发，才能被 GEMM 盖住。代码注释反复强调依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1`——它不影响正确性，只是让 kernel 严格按调用顺序调度。

### 4.3 wgrad 融合与延迟

开 `gradient_accumulation_fusion` 时不算 `grad_weight` 再相加，而是 `wgrad_gemm_accum_fp32/fp16` 直接累进 `weight.main_grad`；Megatron-FSDP 的参数（带 `__fsdp_param__`）改走 `te_general_gemm`，以支持 bf16 输入、fp32 梯度这类混精度。

`grad_output_buffer` + `wgrad_deferral_limit` 则把最后 embedding 层的 wgrad 推迟几个 microbatch，凑满再一起 GEMM。

## 5. 词表并行：embedding 与 loss

`VocabParallelEmbedding` 每 rank 只存 `[vocab/tp, hidden]`。坑在掩码：token id 落在别的 rank 段时查不出值，所以先算 `input_mask`、把越界 id 归零，查完再把对应行清零；不做这一步，跨 rank 归约后就是错值。`deterministic_mode` 下用 `self.weight[masked_input]` 而不是 `F.embedding`，因为后者 backward 不确定。

输出布局也一样要对：`reduce_scatter_embeddings=True` 时先 `[b,s,h] → [s,b,h]` 再 reduce-scatter 成序列并行布局；为 `False` 且 tp>1 时则 all-reduce 汇总。

输出端 `VocabParallelCrossEntropy` 在词表维切 logits，固定三次 all-reduce：全局 max（MAX，`cross_entropy.py:130`）、target logits（SUM）、softmax 分母（SUM）。backward 直接用 forward 存下的归一化 `exp_logits` 算梯度，不再通信。

## 6. 小结

- 原语成对互逆；`_GatherFromSequenceParallelRegion` 的 backward 由**下游是否 TP** 决定 reduce-scatter 还是 split。
- `_AllToAll` 的 backward 是对调 split sizes；`sp2hp` 出 `[tokens, H/tp]`、`hp2sp` 出 `[tokens/tp, H]`。
- `ColumnParallelLinear` 一次可选 all-gather，`RowParallelLinear` 三选一的归约，两者成对抵消。
- 三段异步 + `CUDA_DEVICE_MAX_CONNECTIONS=1` 是 overlap 的前提，不是正确性前提。
- 权重初始化必须「全量 init + 按 rank 取 slice」，否则各 rank 的分片拼不成同一份权重。
- 词表并行 loss 固定三次 all-reduce，backward 不再通信。

---
**版本提示（2026-10）**：上游新增 GTP（`tensor_parallel/generalized_tensor_parallelism.py`），带来自己的权重 all-gather / remat 路径与 symmetric memory allocator（`gtp_symmetric_memory.py`），量化（FP8/MXFP8/NVFP4）参数 all-gather 可与 GTP 分片叠加；`layers.py` 有改动，但并行线性层的语义未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
