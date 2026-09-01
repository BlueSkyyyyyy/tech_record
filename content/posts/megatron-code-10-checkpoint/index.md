---
title: "Megatron 源码精读（十）：checkpoint 处理"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "checkpoint", "分布式"]
categories: ["训练框架"]
weight: 10
series: ["megatron-code"]
---

上一篇[《数据集处理》]({{< relref "megatron-code-09-dataset" >}})把数据喂进了训练循环；本篇讲训练的「存档与复活」——checkpoint。MCore 的 checkpoint 逻辑集中在 `megatron/training/checkpointing.py`（2600+ 行），表面是「把 `state_dict` 存到磁盘 / 从磁盘读回来」，但真正难的是三个分布式特有问题：

1. **命名与布局**：一个模型被 TP/PP/EP 切成几十上百个 shard，每个 rank 各存各的，目录结构长什么样？
2. **格式演进**：从老式「每 rank 一个 `.pt`」到分布式 checkpoint（Meta 的 `torch.distributed.checkpoint`），如何兼容与自动探测？
3. **结构一致性**：断点续训时，checkpoint 里存的模型结构和启动参数必须和当前一致，怎么防「结构对不上还硬读」？

本篇按「布局 → 保存 → 加载 → 一个坑」的顺序拆。

---

## 1. 问题背景：为什么不能简单 `torch.save(model.state_dict())`

朴素想法是每个 rank `torch.save` 自己的那份 `state_dict`，文件名带上 rank 编号。这在 TP/PP/EP 三维并行下会立刻暴露两个问题：

- **找文件难**：恢复时 rank 0 得「猜」目标 check 用没用 PP、用没用 EP（历史 check 可能是不同的并行配置），命名规则一变就找不到；
- **数据冗余/不一致**：DP 组的每个 rank 各存一份模型权重是等价的，全存是浪费；而分布式优化器又要求「同一个 rank 的优化器 state」和「模型参数」能对上（索引不能错位），这不是简单 `torch.save` 能保证的。

所以 MCore 把 checkpoint 拆成「布局」「格式」「内容」三个正交的维度，分别解决。本章核心问题：**这三者分别是什么、怎么落地到代码。**

---

## 2. 现象：目录布局是「iter / mp_rank / 文件」三层

保存的物理布局由 `get_checkpoint_name`（`megatron/training/checkpointing.py:203-249`）决定。核心逻辑是把目录名按并行 rank 拼出来：

```python
# megatron/training/checkpointing.py:219-244（节选）
if release:
    directory = 'release'
else:
    directory = 'iter_{:07d}'.format(iteration)
...
if not pipeline_parallel:
    common_path = os.path.join(checkpoints_path, directory, f'mp_rank_{tensor_rank:02d}')
else:
    common_path = os.path.join(
        checkpoints_path, directory, f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}'
    )
if expert_parallel:
    common_path = common_path + f'_{expert_rank:03d}'
```

即：`iter_0001000/mp_rank_TP_PP[_EP]/model_optim_rng.pt` 这种三层结构。关键观察：

- **文件名本身编码了 shard 位置**：`mp_rank_00_003_002` = TP rank 0、PP rank 3、EP rank 2。目录和 rank 是一一对应的，这是「布局」维度的答案。
- **`return_base_dir`**（`checkpointing.py:220-222`）决定返回「到 `iter_NNNNNNN/` 为止」还是「下钻到某个 rank 的文件名具体路径」。分布式 checkpoint 只需要 base dir（它是一个目录），legacy checkpoint 才需要精确到文件。
- 「release」目录（`checkpointing.py:216-217`）是特殊分支——发布版模型不按 iteration 存，统一放 `release/`。

对应地，`find_checkpoint_rank_0`（`checkpointing.py:273-345`）在做恢复时，是用一套「探测」逻辑：依次试「无 PP 无 EP → 无 PP 有 EP → 有 PP 无 EP → 有 PP 有 EP」四种命名，哪个文件存在就选哪个，最后再试分布式 checkpoint 的目录。这就是解决 §1 第一个痛点的代码。

---

## 3. 根因一：格式分三档，靠 `CheckpointType` 分叉

目录布局解决「在哪存」，但「以什么格式存」是另一个维度。`save_checkpoint`（`checkpointing.py:562`）开头就根据参数决定 `ckpt_type`：

```python
# megatron/training/checkpointing.py:644-667（节选）
ckpt_type = CheckpointType.GLOBAL if args.use_dist_ckpt else CheckpointType.LEGACY
...
ckpt_format = args.ckpt_format if ckpt_type == CheckpointType.GLOBAL else 'torch'
```

`CheckpointType`（`checkpointing.py:468`）大致三档：`LEGACY`（每 rank 一个 `.pt`）、`GLOBAL`（分布式 checkpoint，格式又分 `torch_dist` / `torch_dcp` / `fsdp_dtensor`）、`LOCAL`（non-persistent 本地 checkpoint，配合 resiliency-ext 做故障恢复中间态）。这三档在 `save_checkpoint` 里是一路 if/elif 分叉到底：

- **legacy**：`generate_state_dict` 生成完整 state_dict，某几个 rank（`dp_rank==0` 或 `expt_dp_rank==0`）负责写盘；
- **global + `torch_dist`**：调 Meta 的 `dist_checkpointing.save`（`checkpointing.py:857`），配合 `TorchDistSaveShardedStrategy`（`checkpointing.py:805`），把 state_dict 按 sharding 元数据自动切分落到各 rank；
- **global + `torch_dcp`/`fsdp_dtensor`**：直接走 `torch.distributed.checkpoint.save`（`checkpointing.py:922`），`fsdp_dtensor` 还要先 `preprocess_fsdp_dtensor_state_dict`（`checkpointing.py:877`）。

这里的关键点：**legacy 和 global 的「内容生成」根本不同**。看 `generate_state_dict`（`checkpointing.py:1284-1366`）：

```python
# megatron/training/checkpointing.py:1309-1321（节选）
if args.ckpt_format == "torch_dist":
    model_sd = model[i].sharded_state_dict(...)
else:  # torch, torch_dcp, fsdp_dtensor
    model_sd = model[i].state_dict_for_save_checkpoint()
```

`torch_dist` 下调 `sharded_state_dict()`——它返回的不是真实张量，而是**带 sharding 元数据的「惰性引用」**（每个 `ShardedTensor` 记录它应该在哪个 rank 存、占哪一段），`dist_checkpointing.save` 据此决定谁写哪块。legacy 下则是 `state_dict_for_save_checkpoint()` 返回真实张量，`torch.save` 直接落盘。**同一个模型，两种序列化抽象**——这是理解 MCore checkpoint 最关键的一层。

---

## 4. 根因二：tracker 文件 + 优化器独立文件

再补两个「内容如何定位」的机制。第一是 **tracker 文件**：恢复时用 `get_checkpoint_tracker_filename`（`checkpointing.py:348-351`）得到 `latest_checkpointed_iteration.txt`，里面只写一行「最近一次成功保存的 iteration」，`read_metadata`（`checkpointing.py:361`）读它决定从哪一步恢复。这样恢复方不需要扫目录，O(1) 找到最新迭代。

第二是 **优化器 state 的独立落盘**。开 `use_distributed_optimizer` 时，优化器 state 不是塞进主 state_dict，而是单独写一个 `distrib_optim.pt`：

```python
# megatron/training/checkpointing.py:710-720（节选）
if (
    args.use_distributed_optimizer
    and not args.no_save_optim
    and optimizer is not None
    and ckpt_type == CheckpointType.LEGACY
):
    optim_checkpoint_name = get_distributed_optimizer_checkpoint_name(checkpoint_name)
    ...
    optimizer.save_parameter_state(optim_checkpoint_name)
```

`get_distributed_optimizer_checkpoint_name`（`checkpointing.py:269-270`）就是在主文件同目录下拼 `distrib_optim.pt`。分开存的理由：分布式优化器的 state 是「每个 DP rank 只 hold 一段参数切片」，它的 sharding 布局和模型参数的 `mp_rank` 布局**不是同一套坐标系**，混在一个 state_dict 里反而难恢复。

---

## 5. 解法：加载路径的三个守护关卡

`load_checkpoint`（`checkpointing.py:2003`）是对称的加载入口。它内部有三个值得记住的守护点：

1. **格式自动探测**（`checkpointing.py:2045-2063`）：`args.auto_detect_ckpt_format` 或 `ckpt_format=="torch_dist"` 时，先 `_load_base_checkpoint` 读出来，再根据 `ckpt_type` 反推格式字符串（`LEGACY→torch`、`GLOBAL/LOCAL→torch_dist` 等）。这让「老 check 用 torch、新 check 用 torch_dist」能被同一份代码加载。

2. **结构一致性断言**：`check_checkpoint_args`（`checkpointing.py:143-182`）逐字段校验收到的 checkpoint 参数和当前启动参数，比如：

```python
# megatron/training/checkpointing.py:164-166
_compare('num_layers')
_compare('hidden_size')
_compare('num_attention_heads')
```

任何一个对不上直接 `assert` 失败——宁可报错也不硬读，否则「结构变了还悄悄加载」会训出静默错误。

3. **RNG state 条件加载**（`checkpointing.py:2092-2115`）：只有「ckpt 的 (TP, PP) 和当前一致、非 release、非 finetune、未 `no_load_rng`、且 ckpt 确实存了 rng」时才恢复随机数状态，否则 `ignore_rng_state=True`。因为 TP/PP 变了，同样 seed 的随机行为路径就变了，恢复 RNG 只会制造假的可复现性。

---

## 6. 小结

- **布局 = 三层目录**：`iter_NNNNNNN/mp_rank_TP_PP[_EP]/model_optim_rng.pt`，文件名编码 shard 位置（`checkpointing.py:219-249`）；`find_checkpoint_rank_0` 用「试四种命名」探测历史 check 的并行配置（`checkpointing.py:273-345`）。
- **格式 = 三档分叉**：`CheckpointType`（LEGACY/GLOBAL/LOCAL）× `ckpt_format`（torch/torch_dist/torch_dcp/fsdp_dtensor），在 `save_checkpoint` 里一路 if/elif（`checkpointing.py:644-667`）。
- **两种序列化抽象**：`torch_dist` 用 `sharded_state_dict()`（惰性 sharding 元数据），legacy 用 `state_dict_for_save_checkpoint()`（真实张量）（`checkpointing.py:1309-1321`）。
- **tracker + 独立优化器文件**：`latest_checkpointed_iteration.txt` 定位迭代（`checkpointing.py:348-361`），分布式优化器 state 单独写 `distrib_optim.pt`（`checkpointing.py:710-720`）。
- **加载三道守护**：格式自动探测（`checkpointing.py:2045-2063`）、结构一致性断言（`checkpointing.py:143-182`）、RNG 条件恢复（`checkpointing.py:2092-2115`）。

下一篇讲**优化器**：MCore 的分布式优化器如何在 DP 维度切分 state、`sharded_state_dict` 到底怎么产生那些惰性张量——正好和本篇「优化器 state 单独落盘」接上，也就是《[优化器]({{< relref "megatron-code-11-optimizer" >}})》。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/training/checkpointing.py`。）
