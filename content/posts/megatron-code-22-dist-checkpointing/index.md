---
title: "Megatron 源码精读（二十二）：分布式 checkpoint 底层（dist_checkpointing）"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "checkpoint", "分布式", "sharded-tensor", "序列化"]
categories: ["训练框架"]
weight: 22
series: ["megatron-code"]
---

第 10 篇[《checkpoint 处理》]({{< relref "megatron-code-10-checkpoint" >}})讲的是**训练侧的编排层**——`megatron/training/checkpointing.py` 决定「存哪、什么格式、怎么探测」。那一篇里反复出现一个词 `sharded_state_dict()`，它返回的「带 sharding 元数据的惰性引用」到底怎么被真正落盘、又怎么在任意并行布局下无损恢复，第 10 篇一句带过了。

本篇下钻到真正干活的底层库 **`megatron/core/dist_checkpointing/`**（约 3600 行），讲清三件事：

1. **抽象层**（`mapping.py`）：`ShardedTensor` / `ShardedObject` / `ShardedTensorFactory` 如何描述「局部张量 ↔ 全局张量」的映射。
2. **流程层**（`serialization.py`）：`save` / `load` 两个入口如何把抽象分 6 步落盘、7 步读回。
3. **校验层**（`validation.py`）：保存/加载前的三重校验——access integrity、strict mismatch、sharding integrity，以及 `integrity.json` 的 SHA-256 清单。

行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。

---

## 1. 问题背景：不能简单 `torch.save(model.state_dict())`

TP/PP/DP 把一个大模型切成散落在几百张卡上的「碎片」。简单 `torch.save` 只有两个选择，都不行：

1. **各 rank 存自己的碎片**：存一堆局部张量，下次换并行布局（如 TP=2 改 TP=4、或换 DP）就拼不回来。
2. **全 rank 各自存完整模型**：重复 `dp_size` 份，显存和 I/O 都爆炸。

分布式 checkpoint 的思路是第三条路：**存「全局张量的无冗余视图」**——每个全局张量只落一份，但**记录它的切分元数据**，谁持有哪个局部切片、切片在全局里的 offset 和 fragmentations。这样存的时候每 rank 只写自己的局部片，读的时候按目标布局「对号入座」取自己该拿的切片。`dist_checkpointing/mapping.py` 里的 `ShardedTensor` 就是这份元数据的载体。

---

## 2. 抽象层：三种「带元数据的对象」

### 2.1 `ShardedTensor`：局部张量 ↔ 全局张量的映射

`ShardedTensor`（`mapping.py:52`）的核心是几条元数据，回答「我手里的这块张量，是全局张量的哪一块」：

- `local_shape` / `global_shape`：局部形状与全局形状。
- `global_offset`：局部张量在全局张量里的**起始偏移**（按元素数计）。
- `axis_fragmentations`：每一维的切分数（如 TP 沿最后一维切 `tp_size` 份）。
- `replica_id`：**副本标识**（`mapping.py:28` 定义 `ReplicaId`），用于标记「这个局部张量在多个进程里存在重复份」——如 DP 语义下每个 DP rank 都有同一份参数，但只会由「主副本」`replica_id=0` 写入一次。
- `prepend_axis_num` / `allow_shape_mismatch` / `flattened_range`：处理可变形状（如 padding）与展平存储的扩展字段。

`__post_init__`（`mapping.py:93`）里 `validate_metadata_integrity`（`mapping.py:96`）会在构造时强制这些元数据自洽——比如 `local_shape + prepend_axis_num == global_shape`（`mapping.py:120-124`），否则直接抛 `CheckpointingException`。这是第一道「元数据必须正确」的防线，且**构造即校验**。

### 2.2 `from_rank_offsets`：用 rank 语义描述切分

`ShardedTensor` 最关键的是 `from_rank_offsets`（`mapping.py:190`）这个 classmethod。它接受若干 `(axis, axis_rank_offset, axis_fragm)` 三元组，表示「沿 `axis` 维切成 `axis_fragm` 片，我 rank 拿第 `axis_rank_offset` 片」。

```python
# mapping.py:219-231（节选）
global_offset = [0] * (data.ndim + prepend_axis_num)
global_shape = ([1] * prepend_axis_num) + list(data.shape)
axis_fragmentations = [1] * (data.ndim + prepend_axis_num)
for axis, axis_rank_offset, axis_fragm in rank_offsets:
    local_axis_shape = 1 if axis < prepend_axis_num else data.shape[axis - prepend_axis_num]
    global_shape[axis] = axis_fragm * local_axis_shape
    global_offset[axis] = axis_rank_offset * local_axis_shape
    axis_fragmentations[axis] = axis_fragm
```

数学很直白：`global_shape[axis] = fragm * local_shape`，`global_offset[axis] = rank_offset * local_shape`。这正对应第 21 篇 `ColumnParallelLinear.sharded_state_dict()` 里 `{"weight": 0}` 那种「沿第 0 维按 TP 切」的声明——只是这里是库侧的通用化实现。配套的 `local_chunk_offset_in_global`（`mapping.py:159`）把元素偏移换算成「chunk 坐标」（`offset // local_shape`），供校验层判断每块被访问了几次。

### 2.3 `ShardedObject`：不可切分的对象

张量能按维切，但有的东西切不了（如一个 int、一个字符串、一个 argparse 对象）。`ShardedObject`（`mapping.py:360`）就是「**原子对象的分片**」：它只有 `global_shape` / `global_offset` / `replica_id` 三条元数据，`unique_key`（`mapping.py:397`）把 key + offset + shape 拼成唯一字符串 `key/shard_off.off_shape.shape`，作为落盘时的独立文件/对象名。`empty_from_unique_key`（`mapping.py:409`）能从唯一 key 反推回元数据（反向解析），供加载侧用。

`LocalNonpersistentObject`（`mapping.py:342`）是它的对偶：**保存时跳过、加载时本地重建**——用于那些「不该进 checkpoint、但需要在 state dict 里占位」的对象（如非持久的运行态句柄）。

### 2.4 `ShardedTensorFactory`：先变形再序列化

有些张量存的形态和模型里不一样——典型是优化器状态：分布式优化器（第 11 篇）里参数被 `main_param` 展开成 flat buffer，但 checkpoint 里希望**按原始参数各自的形状落盘**，且「optimizer state 和 model param 用同一套变换」。

`ShardedTensorFactory`（`mapping.py:438`）就是干这个的：`build_fn` 在保存前把原始张量拆成一个 sub-sharded-state-dict，`merge_fn` 在加载后把读回来的子树再合成一个张量。`apply_factories`（`mapping.py:483`，就地 `build`）和 `apply_factory_merges`（`mapping.py:502`，递归 `merge`）分别驱动正反两个方向。这让「优化器状态」的变形逻辑和模型参数完全解耦。

---

## 3. 流程层：`save` 与 `load` 的 6+7 步

### 3.1 `save`：六步落盘

`serialization.py` 的 `save`（`serialization.py:332`）文档字符串列了 7 步，去重后核心是这几步：

1. **apply factories**：把 `ShardedTensorFactory` 展开成真正的 `ShardedTensor`（`serialization.py:422-424` 的 `save_preprocess`）。
2. **剥离 local & sharded**：非持久对象丢弃，sharded 对象从普通 state dict 里抽出来单独处理。
3. **普通对象存 `common_state`**：rank 0 把「不可切分」的普通 Python 对象（args、iteration 等）打包成一个 `ShardedObject("common_state", ...)`（`serialization.py:426-432`），`replica_id = rank`。
4. **sharded 张量按策略写**：交给 `sharded_strategy.save`（`serialization.py:448`）——`TorchDistSaveShardedStrategy`（`strategies/torch.py`）内部调 PyTorch 的 DCP（`torch.distributed.checkpoint.save`）按 `ShardedTensor` 元数据把每个局部片写到它该去的文件。
5. **写 metadata.json**：`metadata_finalize_fn`（`serialization.py:434-440`）在 rank 0 用 `save_config`（`core.py:76`）写 `CheckpointingConfig`（`core.py:23`，记录 `sharded_backend` + 版本）。
6. **（可选）integrity manifest**：`verify_integrity=True` 时 `save_integrity_manifest`（`validation.py:521`）对目录每个文件算 SHA-256 写 `integrity.json`。

异步版 `async_sharded_save`（`serialization.py:454`）把第 4 步变成 `async_save`，返回 `AsyncRequest`，`finalize_fns` 里挂 5、6 两步——**metadata.json 只在 checkpoint 完整写完后才落**，避免读到半成品。

### 3.2 `load`：七步读回

`load`（`serialization.py:62`）是对称的：

1. **先验检查**：`verify_checkpoint`（`validation.py:204`）确认目录存在且是分布式 checkpoint（即 `metadata.json` 能被 `maybe_load_config` 解析，`core.py:49`）。
2. **FP8 反量化**：`force_all_tensors_to_non_fp8`（`serialization.py:125`）把 state dict 里的 FP8 张量转回高精度——防止 `delay scaling` 下误写 TE 的 `amax_history`。
3. **读 common_state**：`load_common_state_dict`（`serialization.py:189`）兼容两种格式：legacy 的独立 `common.pt`，或当前「单个 `ShardedObject('common_state')`」（`serialization.py:210-220`，走 `torch.distributed.checkpoint.load(..., no_dist=True)` 单进程读）。
4. **校验**（详见第 4 节）：strict mismatch + access integrity + sharding integrity。
5. **按目标布局读 sharded 张量**：`sharded_strategy.load`（`serialization.py:168`），`sharded_state_dict` 里的 `ShardedTensor` 元数据决定了「每个 rank 该读哪一段」——这就是**跨并行布局恢复**的关键：加载时用**当前**模型的 sharding 元数据，而不是保存时的。
6. **合并 common + loaded**（`serialization.py:134,170`）。
7. **apply factory merges**（`serialization.py:172`）：把通过 factory 拆开的子树再合成回去（如优化器状态重新 merge 成 flat buffer）。

一个值得记住的细节：**加载用的是「当前模型的 sharding 元数据」当指导**，而非 checkpoint 里存的元数据。`load` 的 `sharded_state_dict` 参数就是「现有模型填充了 `ShardedTensor` 的 state dict，用作 mapping 判断该读全局张量的哪部分」（`serialization.py:83-86`）。这解释了为什么换个 TP size 还能恢复——只要新模型能构造出覆盖同一批全局 key 的 sharding 描述。

---

## 4. 校验层：三重校验 + integrity 清单

### 4.1 access integrity：每个分片恰好被访问一次

保存前的第一道关是 `validate_access_integrity`。`determine_global_metadata`（`validation.py:484`）用 `all_gather_object` 把各 rank 的 sharding元数据（`without_data()` 后的 `ShardedBase`，`mapping.py:46`）汇聚到 rank 0，`validate_sharding_integrity`（`validation.py:369`）检查：

- **张量**：`_compute_shards_access`（`validation.py:454`）统计每个 chunk 的「主副本访问次数」，必须**全 1**（`validation.py:443-449`）——即每个全局分片恰好被一个 rank 以主副本身份覆盖，不重叠、不遗漏。
- **对象**：`_validate_objects_for_key`（`validation.py:464`）检查 `unique_key` 无重复、且数量等于 `prod(global_shape)`。

这防的是最隐蔽的 bug：**保存时忘记某个 rank 的某个分片（有遗漏）或两个 rank 声明同一块（会覆盖写）**，一旦发生，恢复出来静默错误，比保存失败更糟。

### 4.2 strict：请求与 checkpoint 的 key 失配

`StrictHandling`（`validation.py:46`）是一个 8 档枚举，从 `ASSUME_OK_UNEXPECTED`（默认，零开销，靠底层策略报错）到 `RETURN_ALL`（返回全部不匹配 key）。`_determine_missing_and_unexpected_keys`（`validation.py:243`）算两组：

- `unexpected_keys = 本地要加载的 key - checkpoint 的 key`（`validation.py:274`）——只靠本地元数据。
- `missing_keys = checkpoint 的 key - 全局所有 rank 要加载的 key`（`validation.py:279`）——必须靠 global metadata，因为别的 rank 可能访问当前 rank 没请求的 key。

这里有个文档强调的不对称性（`validation.py:250-255`）：**missing keys 各 rank 相同，unexpected keys 各 rank 可能不同**。`adjust_non_strict_load`（`validation.py:222`）会把 unexpected key 从 state dict 里摘掉，避免底层 DCP 对「checkpoint 里没有的 key」报错。

### 4.3 integrity manifest：SHA-256 防静默损坏

`verify_integrity=True` 保存时，`save_integrity_manifest`（`validation.py:521`）对目录每个文件流式计算 SHA-256（`_compute_file_hash`，`validation.py:501`，1 MiB 分块）写 `integrity.json`。加载时 `verify_integrity_manifest`（`validation.py:622`）重算比对，任一文件 hash 不符或缺失即抛错。这针对「checkpoint 文件在存储介质上静默翻转」这类极难排查的问题——代价是保存多读一遍（`serialization.py:387-391`）。

---

## 5. 小结

- **`dist_checkpointing/` 是分布式 checkpoint 的底层库**，第 10 篇的 `checkpointing.py` 只是调它的编排层。
- **`ShardedTensor` 用 `global_offset` + `axis_fragmentations` + `replica_id` 描述「局部↔全局」映射**（`mapping.py:52`），`from_rank_offsets` 用 rank 语义构造（`mapping.py:190`）；`ShardedObject` 处理原子对象（`mapping.py:360`），`ShardedTensorFactory` 处理「先变形再序列化」的优化器状态（`mapping.py:438`）。
- **save 六步 / load 七步**（`serialization.py:332` / `:62`）：普通对象归 `common_state`，sharded 张量交策略层按元数据落盘，metadata.json + integrity.json 收尾。
- **跨布局恢复的关键是「加载用当前模型的 sharding 元数据」**（`serialization.py:83-86`），而非保存时的元数据。
- **三重校验**：access integrity（分片恰好访问一次，`validation.py:369`）、strict mismatch（`validation.py:126`）、sharding integrity + SHA-256 清单（`validation.py:521`）。

下一篇收束系列，补上最后一块独立拼图——**Mamba/SSM 与 Hybrid 混合架构**：SSM 的状态空间如何与注意力模块在 MCore 里编排，gated-MLP、mamba mixer 与 hybrid block 的分层实现。

