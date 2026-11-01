---
title: "Megatron 源码精读（八）：ZeRO-1 / FSDP 实现"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "数据并行", "zero", "fsdp"]
categories: ["训练框架"]
weight: 8
series: ["megatron-code"]
---

上一篇[《CPU offload》]({{< relref "megatron-code-07-cpu-offload" >}})结尾提到分布式优化器（DistOpt）会在 DP 之上继续切分优化器 state，但没展开。本篇补上这块：MCore 的 `use_distributed_optimizer` 到底把什么切开了、什么没切开。

**结论先说：它只把优化器 state 按 DP 切成 1/dp；梯度经 reduce-scatter 后每 rank 只留自己那一段；参数本身仍是完整的连续 buffer，前向之前靠一次 all-gather 补齐。**

这个区分很要紧。把它误读成「参数也只存 shard」，后面所有显存账和通信账都会算错。

---

## 1. 问题：DP 的三份冗余，各自被谁消掉

朴素 DDP 里每个 rank 存三份完整数据：参数、梯度、优化器 state。ZeRO 论文把它们拆成三档——ZeRO-1 切 state，ZeRO-2 再切梯度，ZeRO-3 再切参数。

MCore 提供了三个选项：`use_distributed_optimizer=true` 走 ZeRO-1，`use_torch_fsdp2=true` 外包 PyTorch FSDP2，`use_megatron_fsdp=true` 走自研 Megatron-FSDP。三者显存账完全不同，而最容易被讲错的是第一个。

---

## 2. 一个入口，三个开关

MCore 不为每种形态建类，只在 `_ddp_wrap`（`megatron/training/models/dist_utils.py:291`）里按开关选 wrapper。

```python
if use_megatron_fsdp:
    DP = FullyShardedDataParallel      # 与 use_torch_fsdp2 互斥，同时开会直接抛错
elif use_torch_fsdp2:
    DP = TorchFullyShardedDataParallel
else:
    DP = DistributedDataParallel       # ZeRO-1 也走这里
```

**关键观察：ZeRO-1 不是独立类。** 它就是 `DistributedDataParallel` 在 `ddp_config.use_distributed_optimizer=true` 下的内部行为，所以我们得钻进 buffer 里看。

---

## 3. 代码落点：`_ParamAndGradBuffer`

### 3.1 参数和梯度先拼成两块连续 buffer

`DistributedDataParallel.__init__` 按 `(param_dtype, grad_dtype, is_expert_parallel)` 给参数分组，每组交给一个 `_ParamAndGradBuffer`（`megatron/core/distributed/param_and_grad_buffer.py`）。buffer 里一块 `param_data`、一块 `grad_data`，`param_index_map` 记录每个参数在其中的偏移。

两块 buffer 的分配策略并不对称，这是全篇的枢纽：

- 开 DistOpt：`param_data` 按**完整 numel** 分配（每 rank 都分配一份完整大小），`param.data` 被重映射成这块 buffer 的 view；
- 不开 DistOpt：`param_data` 根本不存在，`param.data` 就是权威参数。

开 DistOpt 时还有一条强约束：`numel` 必须被 `data_parallel_world_size` 整除，不开则要求 `numel == numel_unpadded`（不许 padding）。前者是为等分 shard，后者是为整块 all-reduce。

代价是 TE 侧要手动拆缓存：参数 remap 之后，模块上的 `weight_tensor` 会再持有一份完整副本，所以 DistOpt 分支会 `apply(unmap_weight_tensor)` 把它置空（`distributed_data_parallel.py:391`）。漏掉这一步，所谓「只留一份」就白做了。

### 3.2 梯度：reduce-scatter 后只有本地段有效

`start_grad_sync`（`param_and_grad_buffer.py:651`）是分叉点。开 DistOpt 时它用 `shard_buffer` 把 `grad_data` 切成 `dp` 段，只把本 rank 那段当输出交给 `dist_reduce_scatter_func`；不开则整块 `all_reduce`。

`shard_buffer`（同文件 `:80`）返回的是 **view 列表**，不是新分配——所以 reduce-scatter 是「原地求和、只往本 rank 那段落结果」。`grad_data` 依旧是全量分配，只是 `1/dp` 之外的部分此后再没人读。

缩放和聚合方式在这里合流：`average_in_collective=true` 时 `reduce_op` 取 `AVG`、`gradient_scaling_factor` 为 1.0；为 false 时用 `SUM`，先在通信前把梯度乘 `1/dp_cp_size`。开了 per-token loss 则两者都是 1.0，且禁止在 collective 里求平均。这几个值是成组配对的，不能随手混搭。

### 3.3 参数：完整 buffer，前向前 all-gather

既然每 rank 的 `param_data` 是完整的、而 step 只更新了本地 shard，那么在下次前向之前，必须把各 rank 的 shard 拼回去。这就是 `start_param_sync`（`param_and_grad_buffer.py:448`）的职责，两条路径：

- **DistOpt 路径**：在 `_coalescing_manager` 里对每个 bucket 调 `dist_all_gather_func`，输出直接写进 `bucket.param_data`。因为 `param.data` 本来就是 `param_data` 的 view，gather 完成即参数就绪，**不需要回拷**；
- **非 DistOpt 的 layer-wise 路径**：layer-wise 布局下每个 rank 拥有的参数数量可能不等，于是用变长的 `torch.distributed.all_gather`，接收缓冲借用空闲的 `grad_data`，gather 完还要 `_finalize_layerwise_param_sync` 把数据拷回各参数。

这条路径不是人人都会走到：`start_param_sync` 开头就断言「开 DistOpt 或开 `overlap_param_gather`」，普通 DDP 根本不调它。

这也解释了前面那个「为什么参数要 remap 成 view」：只有 remap 过，all-gather 才能原地生效，省掉一次全量拷贝。

### 3.4 一个 step 的双向通信

前向前 all-gather 参数 → 反向后 reduce-scatter 梯度 → step 只更本地 shard 的 state 和参数。落入 DistOpt 的那部分 state 是真省下来的显存；参数和梯度 buffer 依然按全量分配。

顺带一个常被忽略的开关：`num_distributed_optimizer_instances > 1` 时，切分组是 instance 内的子组，shard 变成 `1/(dp/instances)`，之后还要跨 instance 把 shard all-reduce 一次。此时「1/dp」的说法要打折扣。

---

## 4. 另一种路线：FSDP2 与 Megatron-FSDP

### 4.1 FSDP2：逐层 shard + 即时 gather

`TorchFullyShardedDataParallel` 只做封装，真正的切分交给 `torch.distributed.fsdp.fully_shard`。它默认把 `TransformerLayer`、`LanguageModelEmbedding`、`RotaryEmbedding`、`ColumnParallelLinear` 各包成一个 FSDP 单元（`torch_fully_sharded_data_parallel.py:60`），做到每层参数「用前才 gather、用完即 reshard」。

`reshard_after_forward` 控制前向后的释放：为 `True`（默认）是 ZeRO-3 形态；置 `False`（`--torch-fsdp2-no-reshard-after-forward`）保留前向参数到反向，退回 ZeRO-2 形态。

开了激活重计算后，FSDP2 自动推导的反向预取顺序会失准，所以 MCore 显式调 `set_modules_to_backward_prefetch`，把每层的预取目标钉成「上一层」。

### 4.2 约束：比 ZeRO-1 严得多

`arguments.py` 里那一串 assert 是硬边界：要求 PyTorch ≥ 2.4；不支持 pipeline 并行、expert 并行、`gradient_accumulation_fusion`、fp16；与 `use_distributed_optimizer` **互斥**；`ckpt_format` 只能是 `torch_dist` 或 `torch_dcp`；必须 `--untie-embeddings-and-output-weights`；且 `CUDA_DEVICE_MAX_CONNECTIONS` 不能是 1。

互斥的理由不难想：FSDP2 自己已经切了参数、梯度和 state，再叠一层 DistOpt 就是两套切分打架。

### 4.3 别把两种 FSDP 混为一谈

Megatron 里有两条 FSDP 路线，`use_torch_fsdp2` 只是其中一条。`use_megatron_fsdp` 走的是自研 `FullyShardedDataParallel`，用 `--data-parallel-sharding-strategy` 在 `no_shard` / `optim` / `optim_grads` / `optim_grads_params` 四档间选——切到 state、梯度还是参数，由你显式指定，不像 FSDP2 那样基本由 `reshard_after_forward` 一个布尔量决定。它的约束也不同：**强制搭配 DistOpt**（不给就自动打开）、`ckpt_format` 必须是 `fsdp_dtensor`、只支持 `sgd`/`adam`、不支持 single grouped MoE 权重。

选型时先看你需要哪一档 sharding，再看哪条路线允许它。

---

## 5. 小结

- **ZeRO-1 不是类**：它是 `DistributedDataParallel` 在 `use_distributed_optimizer=true` 下的内部行为。
- **省的只有 state**：参数是完整 `param_data` 的 view，梯度 buffer 也全量分配；只有优化器 state 真按 DP 切成 1/dp。
- **双向通信，总量相当**：前向前 all-gather 参数，反向后 reduce-scatter 梯度；梯度省下的一半又被参数 gather 补回来，省的是显存不是带宽。
- **整除是硬前提**：开 DistOpt 后 buffer 元素数必须被 dp 整除，否则 `shard_buffer` 的等分断言先崩。
- **两条 FSDP 路线**：`use_torch_fsdp2` 外包 PyTorch，限制严；`use_megatron_fsdp` 自研，强制配 DistOpt。

下一篇回到「模型吃进去的数据」：[数据集处理]({{< relref "megatron-code-09-dataset" >}})。

---

**版本提示（2026-10）**：上游 MFSDP 进入 v2（`megatron_fsdp/experimental/`），新增 `TensorAtomic` / `BlockAtomic` / `RowAtomic` 三种 sharding placement，并支持 Muon 的 sharded momentum（#8313）；MFSDP v2 的 checkpoint 与 optimizer 设计另有专门文档。`use_torch_fsdp2` 路径仍在，本文引用的约束未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
