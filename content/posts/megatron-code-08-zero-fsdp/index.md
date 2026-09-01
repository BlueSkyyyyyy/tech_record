---
title: "Megatron 源码精读（八）：ZeRO-1 / FSDP 实现"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "数据并行", "zero", "fsdp"]
categories: ["训练框架"]
weight: 8
series: ["megatron-code"]
---

上一篇[《CPU offload》]({{< relref "megatron-code-07-cpu-offload" >}})结尾提到，分布式优化器（DistOpt）会在 DP 之上继续分片优化器 state，但没展开它的内部。本篇就补上这块缺失：讲清 Megatron Core（下称 MCore）里**两种把「数据并行」做成「显存友好」的路线**——

1. **ZeRO-1**：`DistributedDataParallel` + `use_distributed_optimizer=true`，梯度用 **reduce-scatter** 而非 all-reduce 聚合，并销毁冗余的参数副本；
2. **FSDP2**：`TorchFullyShardedDataParallel` 封装 PyTorch 原生 FSDP2（`fully_shard`），把参数/梯度/优化器 state 全部按 DP 维切碎。

两条路线共享同一个思想：**数据并行的每份 rank 不再各自持有完整模型，而是各自留一片 shard，用时再 all-gather。** 先把这个思想立住，再读代码就顺了。

---

## 1. 问题背景：DP 的显存冗余在哪里

朴素数据并行（DDP）里，每一份 DP rank 都持有一份**完整**的模型参数、完整的梯度、完整的优化器 state。这三者在 HBM 里各占一份，DP 越大，冗余越多：

- **参数**：前后向都要用完整参数，DDP 里天然每 rank 一份；
- **梯度**：反向算完后每个 rank 都有一份完整梯度，先 all-reduce 再各自 step；
- **优化器 state**：Adam 的一阶/二阶矩、FP32 master weight，是参数体积的 2~4 倍，每个 rank 也各存一份——这是最大的浪费。

ZeRO 论文把这三类冗余拆成三档：ZeRO-1 只切优化器 state，ZeRO-2 再切梯度，ZeRO-3 再切参数本身。**MCore 的 `use_distributed_optimizer` 走的是 ZeRO-1**（并把参数副本也一起销毁，见 §3），而 `TorchFullyShardedDataParallel` 走的是 **PyTorch FSDP2**（等价 ZeRO-2/3，取决于 `reshard_after_forward`）。

要回答的核心问题只有一个：**用 all-gather / reduce-scatter 这两种「先切后聚」的集合通信，把上面的冗余换成通信开销，值不值、怎么实现。**

---

## 2. 现象：三条路，一个分叉口

MCore 并不会把 DDP / ZeRO-1 / FSDP2 分成三个类。它只有一个统一入口 `_ddp_wrap`（`megatron/training/models/dist_utils.py:263`），根据三个布尔开关决定用哪个 wrapper：

```python
# megatron/training/models/dist_utils.py:291-301
if use_megatron_fsdp:
    DP = FullyShardedDataParallel
    if use_torch_fsdp2:
        raise ValueError("Using use_megatron_fsdp and use_torch_fsdp2 at the same time is not supported.")
elif use_torch_fsdp2:
    assert HAVE_FSDP2, "Torch FSDP2 requires torch>=2.4.0"
    DP = TorchFullyShardedDataParallel
else:
    DP = DistributedDataParallel
```

也就是说：

| 开关 | 选中的 wrapper | 梯度聚合方式 |
|---|---|---|
| 都为 false | `DistributedDataParallel` | all-reduce |
| `use_distributed_optimizer=true`（仍进 `DistributedDataParallel`） | 同上，但内部走 reduce-scatter（ZeRO-1） | reduce-scatter |
| `use_megatron_fsdp=true` | `FullyShardedDataParallel`（Megatron 自研 FSDP） | 见后续篇（本文聚焦 ZeRO-1 + FSDP2） |
| `use_torch_fsdp2=true` | `TorchFullyShardedDataParallel`（PyTorch FSDP2） | PyTorch FSDP 内部处理 |

**关键观察：ZeRO-1 不是一个独立的 wrapper 类，它和普通 DDP 是同一个类 `DistributedDataParallel`，区别只是 `ddp_config.use_distributed_optimizer` 这个标志。** 这正是为什么 `_ddp_wrap` 里看不到一个 `ZeRO1Wrapper` 的分支——ZeRO-1 完全内嵌在 `DistributedDataParallel` + `_ParamAndGradBuffer` 的实现里。

这条「一个 wrapper 吃遍三态」的设计是理解 MCore DP 的钥匙，下面顺着它往下挖。

---

## 3. 根因：`use_distributed_optimizer` 如何把 all-reduce 换成 reduce-scatter

### 3.1 参数/梯度先拼成连续 buffer

`DistributedDataParallel.__init__` 先收集所有 `requires_grad` 参数，按 `(param_dtype, grad_dtype, is_expert_parallel)` 分组，然后每一组交给一个 `_ParamAndGradBuffer`（`param_and_grad_buffer.py:1066`）：

```python
# distributed_data_parallel.py:163-169
buffer_groups = group_params_for_buffers(
    all_params,
    self.ddp_config.grad_reduce_in_fp32,
    merge_layerwise_fp8_grads=...,
)
```

`_ParamAndGradBuffer.__init__` 的核心是把这一组参数**拼成一个连续的 `param_data` 张量、梯度拼成一个连续的 `grad_data` 张量**，并维护 `param_index_map` 记录每个参数在 buffer 里的偏移（`param_and_grad_buffer.py:1153-1155`）。这样后续的集合通信一次就能覆盖整片内存，而不是逐参数发一次。

注意一个**只有 ZeRO-1 才有的强约束**（`param_and_grad_buffer.py:1220-1225`）：

```python
if self.ddp_config.use_distributed_optimizer:
    assert self.numel % self.data_parallel_world_size == 0
    ...
else:
    assert self.numel == self.numel_unpadded
```

开了 DistOpt 后，buffer 的元素数必须是 `dp_world_size` 的整数倍，才能等分 shard；而普通 DDP 则不允许 padding。这行 assert 是「reduce-scatter 要均匀切分」的直白证据。

### 3.2 梯度聚合：reduce-scatter vs all-reduce 的分叉点

真正的分叉在 `_ParamAndGradBucketGroup.start_grad_sync`（`param_and_grad_buffer.py:651`）。片段：

```python
# param_and_grad_buffer.py:752-783
if self.ddp_config.use_distributed_optimizer:
    communication_group = self.intra_distributed_optimizer_instance_group
else:
    communication_group = self.data_parallel_group

for idx, bucket in enumerate(self.buckets):
    if self.ddp_config.use_distributed_optimizer and not force_all_reduce:
        if self.cached_grad_buffer_shard_list[idx] is None:
            self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                bucket.grad_data, self.intra_distributed_optimizer_instance_size
            )
        local_data_view = self.cached_grad_buffer_shard_list[idx][
            self.intra_distributed_optimizer_instance_rank
        ]
        grad_reduce_handle = dist_reduce_scatter_func(
            local_data_view, bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op,
        )
    else:
        torch.distributed.all_reduce(
            bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
        )
```

这里的逻辑一句话总结：**开了 `use_distributed_optimizer` 就用 `reduce_scatter` 把梯度切成 `dp_size` 份、每个 rank 只留自己那份；没开就老老实实 `all_reduce` 让每个 rank 都拿全量梯度。**

`shard_buffer`（`param_and_grad_buffer.py:80-89`）的实现在这里很关键：

```python
def shard_buffer(buffer, data_parallel_world_size):
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    return [buffer[(r * shard_size):((r + 1) * shard_size)] for r in range(data_parallel_world_size)]
```

它返回的是**视图（view）**列表，不是新分配。所以 `local_data_view` 就是 `grad_data` 的一段连续切片，reduce-scatter 把「每 rank 的完整梯度求和后」，只把第 `rank` 段落回 `grad_data` 的那一段。

reduce-scatter 的本意是：**每个 rank 只需要自己那 1/dp_size 的梯度去更新自己那 1/dp_size 的参数**，没必要像 all-reduce 一样让所有人都拿全量。这正是 ZeRO-1「切优化器 state」的梯度侧前提——分布式优化器只维护自己那份参数对应的 state，所以它只需要自己那份梯度。

### 3.3 参数同理：用 all-gather 换「只存一份」

梯度切了，参数也要跟着切才能在 step 时对上号。`_ParamAndGradBuffer.__init__` 里，`param_data` **只在 `use_distributed_optimizer=true` 时才分配**（`param_and_grad_buffer.py:1323-1332`）：

```python
if self.ddp_config.use_distributed_optimizer:
    numel = self.nvfp4_packed_numel if self.has_nvfp4_params else self.numel
    with param_mem_alloc_context():
        self.param_data = torch.zeros(numel, dtype=self.param_dtype, ...)
```

为什么普通 DDP 不分配 `param_data`？因为普通 DDP 里 `param.data` 就是权威参数，直接 all-reduce 梯度后 in-place 更新即可，不需要把参数搬到连续 buffer 里。而 ZeRO-1 要销毁冗余的 `param.data` 副本、把参数重映射到这块连续 buffer 上（`distributed_data_parallel.py:384-395` 里的 `unmap_weight_tensor` 就是配合这个映射清理 TE 的 `weight_tensor` 残留）。

前向要用完整参数时，走 `start_param_sync`（`param_and_grad_buffer.py:448`）里的 all-gather：开了 DistOpt 走 `dist_all_gather_func`（`param_and_grad_buffer.py:594-599`），把各 rank 的 shard 聚回完整参数；没开则走 legacy 的 `torch.distributed.all_gather`（`param_and_grad_buffer.py:566`）。

### 3.4 一条主线的双向通信

把上面的机制串起来，ZeRO-1 一个 step 的通信是**双向的**：

- **前向前**：all-gather 参数（`start_param_sync`）；
- **反向后**：reduce-scatter 梯度（`start_grad_sync`）；
- **step 时**：分布式优化器只更新本 rank 的那一份参数/state，无需额外通信。

普通 DDP 同样是双向（前向前 all-reduce 参数以同步初始化、反向后 all-reduce 梯度），区别只在「梯度聚合」这一环换了 collectives，以及「参数是否被销毁成单份 shard」。

---

## 4. PyTorch FSDP2 封装

`use_torch_fsdp2=true` 时，MCore 干脆不自己实现并行切分了，而是包一层 PyTorch 的 FSDP2 API（`torch.distributed.fsdp.fully_shard`），要求 PyTorch ≥ 2.4（`torch_fully_sharded_data_parallel.py:70-72`）。

### 4.1 选哪些子模块做 shard

`TorchFullyShardedDataParallel.__init__`（`torch_fully_sharded_data_parallel.py:55`）默认对下列子模块做 FSDP（`torch_fully_sharded_data_parallel.py:60-65`）：

```python
sub_modules_to_wrap: Set[torch.nn.Module] = {
    TransformerLayer,
    LanguageModelEmbedding,
    RotaryEmbedding,
    tensor_parallel.ColumnParallelLinear,
}
```

即**每个 TransformerLayer 一层单独一个 FSDP 单元**，嵌入层、RoPE、最终输出层也各一个。这样做的目的是让参数 all-gather 做到「just-in-time」——每层前向需要时才 gather 这一层的参数，用完就 reshard 释放，而不是每个 iteration 一开始 gather 全部参数（注释见 `torch_fully_sharded_data_parallel.py:127-129`，指向 PyTorch issue #114299）。

### 4.2 核心封装：`fully_shard` + 反向预取调度

真正干活的是这一段（`torch_fully_sharded_data_parallel.py:120-148`）：

```python
for sub_module in self.module.modules():
    if any(isinstance(sub_module, sub_module_to_wrap) for sub_module_to_wrap in sub_modules_to_wrap):
        fully_shard(sub_module, **kwargs)
        if config.recompute_granularity is not None:
            sub_module.set_modules_to_backward_prefetch([prev_module] if prev_module else [])
        prev_module = sub_module

# Wrap the root module as required by the FSDP API.
fully_shard(self.module, **kwargs)
```

`kwargs` 里最关键的参数是 `reshard_after_forward`（`torch_fully_sharded_data_parallel.py:82-85`），它决定「前向 gather 的参数在前向结束后是否立即 reshard 回 shard」：

- `True`（默认）：前向结束后立刻释放全量参数副本，只剩 shard，显存最优但反向要重新 all-gather；
- `False`：前向的参数留到反向，省一次 gather 但显存多占。等价于 ZeRO-2 与 ZeRO-3 的差异——`False` 让 FSDP2 退化成「只切梯度+state 不切前向参数」的形态（即 ZeRO-2 语义）。

### 4.3 显式反向预取：压榨「计算-通信重叠」

FSDP2 默认会根据参数计算图自动推导 backward prefetch 顺序，但 MCore 的激活重计算会打乱这个推断（注释见 `torch_fully_sharded_data_parallel.py:136-138`）。所以当 `config.recompute_granularity is not None` 时，显式调用 `set_modules_to_backward_prefetch`，把每层的 prefetch 目标硬设定为「上一层的模块」，让第 N 层反向计算时，第 N-1 层的参数已经悄悄 all-gather 完毕——这正是 DDP 里 `overlap_grad_reduce`/`overlap_param_gather` 想达到的「通信藏在计算后面」在 FSDP2 里的对应实现。

### 4.4 它与 ZeRO-1 的权限边界

`use_torch_fsdp2` 的约束比 ZeRO-1 更严格，集中在 `arguments.py:1087-1111`：

```python
if args.use_torch_fsdp2:
    assert is_torch_min_version("2.4.0"), ...
    assert args.pipeline_model_parallel_size == 1, ...      # 不支持 pipeline
    assert args.expert_model_parallel_size == 1, ...        # 不支持 expert
    assert not args.use_distributed_optimizer, ...          # 与 MCore DistOpt 互斥
    assert not args.gradient_accumulation_fusion, ...       # 不支持梯度累加融合
    assert args.ckpt_format in ('torch_dist', 'torch_dcp'), ...  # 只支持分布式 ckpt
    assert args.untie_embeddings_and_output_weights, ...    # embedding 与输出解绑
    assert not args.fp16, ...                               # 暂不支持 fp16
```

核心信息有二：**一是 `use_torch_fsdp2` 与 `use_distributed_optimizer` 互斥**——因为 FSDP2 已经自带参数/梯度/state 切分（PyTorch 原生 optimizer in FSDP2 直接消费 shard 后的参数和梯度），再叠一层 MCore 分布式优化器就重复了；**二是不支持 pipeline 和 expert 并行**，这是当前版本的限制，选 torch_fsdp2 意味着你只能在 TP×DP×CP 的组合里用它。

---

## 5. 一条主线串起来：从参数到 state 的完整生命周期

把 §3 和 §4 合起来，MCore 的数据并行「显存优化」其实是一个连续的谱系：

| 形态 | 参数 | 梯度 | 优化器 state | 梯度聚合 |
|---|---|---|---|---|
| 普通 DDP | 每 rank 完整 | 每 rank 完整 | 每 rank 完整 | all-reduce |
| ZeRO-1（`use_distributed_optimizer`） | 单份 shard | reduce-scatter 后单份 shard | 分布式优化器单份 shard | reduce-scatter |
| FSDP2（`reshard_after_forward=True`） | 单份 shard | 单份 shard | 单份 shard | PyTorch FSDP 内部 |

越往下越省显存，但通信和实现复杂度越高。MCore 把前两档塞进同一个 `DistributedDataParallel` 类，第三档外包给 PyTorch——这种「能复用就复用、不能复用就委托」的分层，正是它并行层设计的味道。

还有一个容易被忽略的细节：**reduce-scatter 的通信量并不会比 all-reduce 少**。all-reduce 每个 rank 收全量，reduce-scatter 每个 rank 收 1/dp_size，但 ring 算法下两者的总线通信量其实是同一量级。ZeRO-1 省的是**内存**，不是**带宽**——这点在讨论「开 DistOpt 值不值」时经常被搞混。

---

## 6. 小结

- **一个分叉口**：`_ddp_wrap`（`dist_utils.py:291-301`）用三个开关 `use_megatron_fsdp` / `use_torch_fsdp2` / 默认，选出 `FullyShardedDataParallel` / `TorchFullyShardedDataParallel` / `DistributedDataParallel` 三者之一。
- **ZeRO-1 不是一个类**：它是 `DistributedDataParallel` 在 `use_distributed_optimizer=true` 下的内部行为——梯度从 all-reduce 换成 reduce-scatter（`param_and_grad_buffer.py:761-775`），参数映射到连续 buffer 且每 rank 只留 shard（`param_and_grad_buffer.py:1323-1332`）。
- **切分的几何约束**：开 DistOpt 后 buffer 元素数必须被 `dp_world_size` 整除（`param_and_grad_buffer.py:1220-1221`），否则 `shard_buffer` 的等分断言（`param_and_grad_buffer.py:84`）直接崩。
- **双向通信主线**：前向前 all-gather 参数（`start_param_sync`），反向后 reduce-scatter 梯度（`start_grad_sync`），step 时分布式优化器只更自己那份。
- **FSDP2 是「外包」**：`TorchFullyShardedDataParallel` 包 PyTorch `fully_shard`，默认按 TransformerLayer 粒度逐层 shard + just-in-time gather（`torch_fully_sharded_data_parallel.py:120-146`），并显式设置 backward prefetch 对抗重计算打乱调度（`torch_fully_sharded_data_parallel.py:138-141`）。
- **省内存不省带宽**：ZeRO-1/FSDP 的收益是显存（消灭每 rank 的冗余副本），不是通信量。
- **互斥约束**：FSDP2 与 MCore 分布式优化器、pipeline、expert 并行都不兼容（`arguments.py:1087-1111`）。

下一篇回到「模型吃进去的数据」，讲 MCore 的数据集处理（数据混合、序列打包、varlen），也就是《数据集处理》（待写）。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/distributed/distributed_data_parallel.py`、`megatron/core/distributed/param_and_grad_buffer.py`、`megatron/core/distributed/torch_fully_sharded_data_parallel.py`、`megatron/core/distributed/distributed_data_parallel_config.py`、`megatron/training/models/dist_utils.py`、`megatron/training/arguments.py`。）
