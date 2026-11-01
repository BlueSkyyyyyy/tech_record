---
title: "Megatron 源码精读（二十二）：分布式 checkpoint 底层（dist_checkpointing）"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "checkpoint", "分布式", "sharded-tensor", "序列化"]
categories: ["训练框架"]
weight: 22
series: ["megatron-code"]
---

第 10 篇[《checkpoint 处理》]({{< relref "megatron-code-10-checkpoint" >}})讲的是编排层——`megatron/training/checkpointing.py` 决定「存哪、什么格式、怎么探测」。那一篇里反复出现的 `sharded_state_dict()` 到底怎么被落盘、又怎么在另一种并行布局下无损读回，只一句带过了。

本篇下钻到真正干活的 `megatron/core/dist_checkpointing/`，回答一个问题：**一堆散落在几百张卡上的局部张量，凭什么能当成一个「全局张量」来存取？**

一句话结论：**用一个可交换的元数据表（`ShardedTensor`）描述「局部块在全局里的坐标」，保存时按主副本去重落盘，加载时用当前模型的同一张表反查自己该拿哪一块——换并行布局之所以可行，靠的是这张表而不是数据本身。**

## 1. 为什么两种朴素存法都不行

各 rank 存自己的碎片：存下来是一堆局部张量，下次 TP 从 2 改 4 就拼不回来。

全 rank 各存一份完整模型：重复 `dp_size` 份，I/O 和空间都浪费。

第三条路是「存全局视图 + 记录切分元数据」：每个全局张量只落一份，但每个局部片都声明「我在全局的第几块、从哪个 offset 开始、全局沿这维切了几份」。`mapping.py` 的 `ShardedTensor` 就是这份声明。

## 2. 抽象层：元数据怎么描述「局部 ↔ 全局」

### 2.1 `ShardedTensor`：六个字段说清「我是全局的哪一块」

`ShardedTensor`（`mapping.py:52`）的关键字段：`key`（全局张量标识）、`local_shape` / `global_shape`、`global_offset`（按元素计的起点）、`axis_fragmentations`（每维切几份）、`replica_id`（副本编号）。

`__post_init__` 立刻调 `validate_metadata_integrity`（`mapping.py:96`），**构造即校验**：维数要一致，`global_offset` 必须能被对应的 `local_shape` 整除（否则这块根本不对齐网格），`data` 的 dtype 和 shape 要和声明一致。

有个容易误解的字段：`flattened_range` 看着像「支持展平存储」，实际校验里直接抛 `ShardedTensor.flattened_range is not supported`。真正处理可变形状的是 `allow_shape_mismatch`——它让加载时的全局形状允许和 checkpoint 不一致（如 padded 张量）。

### 2.2 `from_rank_offsets`：把 rank 语义换算成坐标

手写 offset 容易错，所以模型侧几乎都用 `from_rank_offsets`（`mapping.py:190`）。它接收若干 `(axis, rank_offset, fragm)` 三元组，含义是「沿 axis 切 fragm 份，我拿第 rank_offset 份」，然后反推出全局形状与起点（`mapping.py:228-231`）。设轴 $a$ 上局部长度 $L_a$、切分数 $f_a$、rank 偏移 $r_a$，则：

$$
\begin{aligned}
G_a &= f_a \cdot L_a \\
O_a &= r_a \cdot L_a
\end{aligned}
$$

其中 $G_a$ 是全局长度、$O_a$ 是全局起点。这也解释了 `ColumnParallelLinear` 那种 `{"weight": 0}` 声明为什么能通用——它只说「沿第 0 维按 TP 切」，剩下的由这个公式补齐。校验层要的 chunk 坐标由 `local_chunk_offset_in_global`（`mapping.py:159`）算——把每维的元素偏移除以该维局部长度即可。

### 2.3 `ShardedObject`：切不动、也不能重切的原子对象

args、iteration、字符串这类东西没法按维切，用 `ShardedObject`（`mapping.py:360`）表示：只有 `global_shape` / `global_offset` / `replica_id` 三条元数据，加上 `key` 和 `data`。

它和 `ShardedTensor` 有一个本质区别，docstring 写得很明确：**加载时无法改变全局切分**（相当于把每个元素当成不可分的原子）。所以 `common_state` 这类对象只能按原样读回，不能跨布局重塑。

落盘名由 `unique_key` 拼成 `key/shard_<offset>_<shape>`（各维用点号连接），`empty_from_unique_key`（`mapping.py:409`）能反解析回元数据，供只读元数据的场景用。

`LocalNonpersistentObject`（`mapping.py:342`）是它的对偶：保存时直接丢弃，加载时用本地那份占位——典型用途是「运行时句柄不该进 checkpoint，但 state dict 里要有这个 key」。

### 2.4 `ShardedTensorFactory`：先变形、再序列化

分布式优化器把参数拼成 flat buffer，但 checkpoint 里希望按原始参数各自的形状落盘。`ShardedTensorFactory`（`mapping.py:438`）就是这层适配：`build_fn` 在保存前把一个张量展开成子 state dict，`merge_fn` 在加载后把子树合回一个张量。

`apply_factories`（`mapping.py:483`）就地展开，`apply_factory_merges`（`mapping.py:502`）递归合并。注意它的 `validate_metadata_integrity` 是空实现（注释写「没有合理的检查可做」）——工厂产出什么元数据，这一层不会拦。

## 3. 流程层：`save` 七步、`load` 五步

### 3.1 `save` 的 docstring 列了七步

`save`（`serialization.py:332`）的 docstring 原文是 7 步：展开工厂 → 丢弃非持久对象 → 抽出所有 `ShardedBase` → 其余存 common → 可选存 `ShardedObject` → 存 `ShardedBase` → 写 `metadata.json`。

实际执行里，前五步被打包进 `state_dict_utils.save_preprocess`（`serialization.py:422` 调用），它做的是：`apply_factories`、`extract_nonpersistent`、`extract_sharded_base`、`filter_out_empty_flatten_tensor`，最后按需 `validate_sharding_integrity`。

docstring 第 4 步说「存到 common.pt」，代码里其实没有这个文件了：普通对象被塞进一个 `ShardedObject("common_state", ..., replica_id=rank)`，与张量一起进同一个分布式 checkpoint。`common.pt` 只在**读**的时候作为 legacy 格式兼容（`load_common_state_dict`，`serialization.py:189`）。

收尾靠两个 finalize 函数：`metadata_finalize_fn` 让 rank 0 写 `metadata.json`（backend + version）后 barrier；`verify_integrity=True` 时再加一个算 SHA-256 的。异步保存时这两个函数挂在 `AsyncRequest.finalize_fns` 上，所以**元数据只在整个 checkpoint 写完之后才落**，不会出现「有 metadata、数据没写完」的半成品。

### 3.2 `load` 的 docstring 只有五步

`load`（`serialization.py:62`）的 docstring 列 5 步：读 common 打底 → 展开工厂 → 抽 `LocalNonpersistentObject` → 可选读 `ShardedObject` → 抽 `ShardedBase`、读、应用 merge。

代码的实际顺序略有出入：`verify_checkpoint` → 可选 `verify_integrity_manifest` → `force_all_tensors_to_non_fp8`（把 FP8 张量还原成高精度，避免延迟缩放往 TE 的 `amax_history` 里多写一笔）→ `load_preprocess`（展开工厂、抽出非持久对象）→ 读 common 并合并非持久对象 → `extract_sharded_base` → 校验 → `sharded_strategy.load` → `merge` → `apply_factory_merges`。

### 3.3 为什么换个 TP 还能读回来

这是全篇最该记住的一句：**加载时用的是当前模型的 sharding 元数据，不是 checkpoint 里的**。`load` 的第一个参数就是「现有模型填好 `ShardedTensor` 的 state dict」，docstring 说它「用作判断该读全局张量哪一部分的映射」；checkpoint 里的元数据只用来算 missing / unexpected key。

于是跨布局恢复的条件很明确：新模型能构造出覆盖同一批全局 key、且全局形状一致的描述。形状不一致会被 `MCoreLoadPlanner` 拦下——只有 `allow_shape_mismatch=True` 的张量才跳过形状校验。所以「TP 2→4」可以，「全局形状从 4096 变 8192」不可以。

## 4. 校验层：三道关，各管一类事故

### 4.1 access integrity：每块恰好被访问一次

`validate_sharding_integrity`（`validation.py:369`）统计各 rank 的元数据后，**只由 rank 0 判**。对张量：所有 rank 的 dtype、`global_shape`、`local_shape` 必须一致；`_compute_shards_access`（`validation.py:454`）按 chunk 坐标累加「主副本访问次数」，要求全为 1。

两个细节值得记：一，只统计主副本（`replica_id` 为全 0）；二，**不规则网格直接跳过**——`axis_fragmentations is None` 时把校验 defer 给 DCP，不会报「全 1」失败。

对 `ShardedObject`：`unique_key` 不许重复，且数量必须等于 `prod(global_shape)`（`validation.py:464`）。这一关防的是最隐蔽的 bug——漏存一块（恢复后静默缺数据）或两个 rank 声明同一块（互相覆盖）。

### 4.2 strict：missing 与 unexpected 天生不对称

`StrictHandling` 是 8 档枚举，默认 `ASSUME_OK_UNEXPECTED` 零额外开销，靠底层策略在遇到意外 key 时报错；`RETURN_ALL` 则把所有不匹配 key 一起返回。

两组 key 的算法不同（`validation.py:243`）：`unexpected = 本地要加载的 key − checkpoint 的 key`，查本地元数据就够；`missing = checkpoint 的 key − 全局所有 rank 要加载的 key`，必须 `all_gather_object` 拿到全局视图。因此 docstring 特意点明：**missing 各 rank 相同，unexpected 各 rank 可能不同**。非严格模式下 `adjust_non_strict_load` 会把 unexpected key 从 state dict 里摘掉，免得底层实现报错。

### 4.3 integrity manifest：防静默损坏

`verify_integrity=True` 保存时，`save_integrity_manifest`（`validation.py:521`）对目录**顶层**每个文件流式算 SHA-256（`_compute_file_hash`，1 MiB 分块，`validation.py:501`），写进 `integrity.json`，并跳过这个文件自身。

加载时 `verify_integrity_manifest`（`validation.py:622`）重算比对，缺文件或哈希不符都抛 `CheckpointingException`。多 rank 场景下只有 rank 0 真去算，再把错误 `broadcast` 出去。代价是保存要多读一遍全部数据；另外这条开关只允许配 `torch_dist`，TrainingConfig 里直接断言。

## 5. 小结

- **元数据就是协议**：`ShardedTensor` 用 `global_offset` + `axis_fragmentations` + `replica_id` 描述局部与全局的关系，构造时即校验对齐性。
- **`from_rank_offsets` 是最常用的换算口**：`global_shape = fragm × local_shape`、`global_offset = rank_offset × local_shape`，模型侧只声明 rank 语义。
- **`ShardedObject` 不可重切**，`LocalNonpersistentObject` 不落盘，`ShardedTensorFactory` 负责「存的和跑的不一样」。
- **`save` docstring 七步、`load` docstring 五步**，两者顺序并不严格对应代码；`common.pt` 已是 legacy 兼容路径，新格式把普通对象也塞进 `common_state`。
- **加载靠当前模型的元数据**，全局形状必须一致才能跨 TP 恢复。
- **三道校验各管一类事故**：access integrity 管漏写/重写，strict 管 key 对不上，manifest 管介质静默损坏。

下一篇补上系列最后一块独立拼图——**Mamba/SSM 与 Hybrid 混合架构**：状态空间模块怎么和注意力在 MCore 里共存，mamba mixer、gated-MLP 与 hybrid block 如何分层实现，见《[Mamba/SSM 与 Hybrid]({{< relref "megatron-code-23-mamba-ssm-hybrid" >}})》。

---
**版本提示（2026-10）**：DCP v2 已接入 dynamic CP 的标准 train/eval schedule（#7047）；`dist_ckpt` 支持 FP8/FP4 param-gather 的仅推理流式反量化（`strategies/torch.py` 的 `stream_ckpt_dequant`，#7405）；新增 `tools/checkpoint/weighted_merge.py` 做纯元数据驱动的多 checkpoint 加权合并。底层抽象没变：`save` / `load` 的 docstring 步数与 `from_rank_offsets` 语义一致，`mapping.py` 只是删掉了 `global_slice` / `max_allowed_chunks` 两个辅助方法。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
