---
title: "Megatron 源码精读（九）：数据集处理"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "数据集", "dataloader"]
categories: ["训练框架"]
weight: 9
series: ["megatron-code"]
---

上一篇[《ZeRO-1 / FSDP 实现》]({{< relref "megatron-code-08-zero-fsdp" >}})讲的是「模型怎么在 DP 上省显存」，本篇掉头，讲「喂给模型的数据是怎么组织起来的」。MCore 的数据管线（`megatron/core/datasets/`）是一套**从磁盘上的 `.bin`/`.idx` 到训练循环里的 `DataLoader` 的完整流水线**，核心概念高度浓缩成一句话：

> 离线把语料 token 化成 `.bin`（连续 token 字节流）和 `.idx`（每条文档/序列的长度与边界），训练时**在不碰原始文本的前提下**，用三级索引把「文档 → 样本 → 打乱后的样本」的映射做成纯整数运算，再用一个 `BlendedDataset` 在外层做多数据源的加权混合。

本篇按这条主线的顺序拆：先讲最底层的 `IndexedDataset`（为什么是 `.bin` + `.idx` 两个文件），再讲 `GPTDataset` 的三级索引，最后讲 `BlendedDataset` 与 `BlendedMegatronDatasetBuilder` 的混合/切分逻辑。

---

## 1. 问题背景：为什么不用「原始文本 + 在线 tokenize」

朴素的 GPT 数据管线想法是：Dataloader 读原始文本行 → 在线 tokenize → batch。这条路线有两个致命问题：

1. **在线 tokenize 是 CPU 密集且不可共享的**。每个 worker、每个 epoch 都要重新跑一遍 tokenizer，而 tokenize 的负载和矩阵乘一样重，会卡成一个训练前场外的大瓶颈；
2. **随机访问难**。GPT 的训练样本是「从某个文档的某个位置切一段 `sequence_length` 长的 token」，还可能跨文档。原始文本是变长的、带换行的，没法 O(1) 按偏移切序列。

所以业界（包括 GPT-NeoX、HF 的 `load_dataset`、MCore）统一的做法是 **ETL 阶段一次性离线 tokenize，训练阶段零 tokenize**。MCore 落地的格式就是 `IndexedDataset`：一个 `.bin` 文件装所有 token id 的连续字节流，一个 `.idx` 文件装每条序列的长度和文档边界。这样就同时解决了第 1 点（tokenize 只做一次）和第 2 点（按偏移 O(1) 取序列）。

本章要回答的核心问题：**这套格式具体长什么样、怎么存、怎么读**。

---

## 2. 现象：两个文件，一份「变长序列的稀疏索引」

`IndexedDataset` 的构造签名（`megatron/core/datasets/indexed_dataset.py:611-671`）第一句就是路径前缀，而不是文件名：

```python
# megatron/core/datasets/indexed_dataset.py:709-710
idx_path = get_idx_path(path_prefix)
bin_path = get_bin_path(path_prefix)
```

给定 `path_prefix`，它会去找 `<prefix>.bin` 和 `<prefix>.idx` 两个文件，缺一不可（`indexed_dataset.py:711-715` 里的 `assert os.path.exists` 同时检查二者）。这就是「现象」层的关键观察：**一个数据集永远成对出现，`.bin` 是数据，`.idx` 是元数据**。

`.idx` 文件由 `_IndexReader` 解析（`indexed_dataset.py:246` 附近），内容就三块：

- `sequence_lengths`：每条序列的长度（`int32` 数组）；
- `sequence_pointers`：每条序列在 `.bin` 里的字节偏移；
- `document_indices`：每个文档的「结束序列号」（`int64` 数组）。

`indexed_dataset.py:674-676` 的断言非常关键，直接点破了 `.idx` 的设计约束：

```python
# megatron/core/datasets/indexed_dataset.py:674-676
assert self.index.sequence_lengths.shape[0] == self.index.document_indices[-1]
assert self.index.sequence_lengths.shape[0] == len(self.index)
assert self.index.sequence_lengths.shape[0] == self.index.sequence_count
```

三条断言翻译成人话：**「序列数 == 最后一个文档的结束序号 == 总元素数」**。也就是说，`.idx` 里序列是「文档切分成的一段段」，`document_indices` 用「这段序列的结束下标」来标注每个文档的边界——文档 i 覆盖序列 `[document_indices[i-1], document_indices[i])`。这是变长文档（每个文档 token 数不等）被压成「定长序列 + 边界指针」的关键手段。

---

## 3. 根因：GPT 样本 = 文档 → 三级索引的纯整数映射

有了 `.bin` + `.idx`，训练时一个样本怎么来的？答案在 `GPTDataset` 的 `_build_document_sample_shuffle_indices`（`megatron/core/datasets/gpt_dataset.py:461-687`）。它构建了**三级索引**，是这个文件乃至整个数据管线的灵魂：

1. **document_index**（1-D）：`num_epochs × 文档数` 的有序文档 id 数组，并整体 shuffle。它决定了「每个 epoch 里文档以什么顺序出场」。

2. **sample_index**（2-D）：每行 `(doc_index_beg, offset_beg)`，标注「每一个样本从哪个文档的哪个偏移开始」。它由 C++ helper `helpers.build_sample_idx` 算出（`gpt_dataset.py:604-612`），把 `document_index` 这条文档流水线，切成一个个 `sequence_length` 长的样本——**这是把「文档」切成「样本」的切分轴**。

3. **shuffle_index**（1-D）：`[0, num_samples)` 的随机排列（`_build_shuffle_index`，`gpt_dataset.py:754`）。它决定了「样本以什么顺序被 DataLoader 取走」。注意打乱的是**样本的索引**，而不是样本内容——这是为了 shuffle 不破坏文档内 token 的连续性。

三级索引的构建顺序、职责、与「打乱」的关系，浓缩成一段（`gpt_dataset.py:578-622`）：

```python
# megatron/core/datasets/gpt_dataset.py:578-612（节选）
document_index = _build_document_index(
    self.indices, num_epochs, numpy_random_state, separate_final_epoch
)
...
sample_index = helpers.build_sample_idx(
    sequence_lengths_for_cpp,
    document_index,
    sequence_length,
    num_epochs,
    num_tokens_per_epoch,
    drop_last_partial_sequence,
    self.config.add_extra_token_to_sequence,
)
```

这里有个非常容易被忽略的细节：**第一级（文档）和第二级（样本）的 shuffle 都发生在 `build_sample_idx` 之前**；第三级 shuffle 发生在最后。换句话说，shuffle 的顺序是「先打乱文档 → 再切样本 → 最后打乱样本」。为什么切样本之后还要再打乱一次？因为 `build_sample_idx` 是按文档顺序线性扫描切样本的，相邻样本高度相关（同一个文档、相近位置），不二次打乱的话一个 batch 里全是同一篇文档的内容，训练会崩。

再看「跨文档样本」是怎么处理的。一个样本可能横跨两个文档（前文档尾部 + 后文档头部），`_query_document_sample_shuffle_indices`（`gpt_dataset.py:374-459`）里用 `doc_index_beg == doc_index_end` 分支区分：

```python
# megatron/core/datasets/gpt_dataset.py:409-423（节选）
if doc_index_beg == doc_index_end:
    # 样本落在单个文档内
    document_ids.append(self.document_index[doc_index_beg])
    sample_parts.append(self.dataset.get(
        self.document_index[doc_index_beg],
        offset=int(doc_index_beg_offset),
        length=doc_index_end_offset - doc_index_beg_offset
        + self.config.add_extra_token_to_sequence,
    ))
else:
    # 样本横跨多个文档，逐段拼接
    for i in range(doc_index_beg, doc_index_end + 1):
        ...
```

即：样本 = `[起始文档 offset 处]` + `[中间完整文档]` + `[结束文档 0 到 end_offset 处]` 的拼接。最后不足 `sequence_length` 的部分用 pad token 补齐（`gpt_dataset.py:449-453`）。

还有 `add_extra_token_to_sequence` 这个「+1」：GPT 训练需要 labels 是 tokens 右移一位，于是每个样本实际上多取 1 个 token，作为 `text[1:]` 的标签（`gpt_dataset.py:267-273` 的 `tokens = text[:-1]; labels = text[1:]`）。

一个值得深挖的坑：**`num_epochs` 是怎么来的**。`_get_num_epochs`（`gpt_dataset.py:697-717`）里，`num_samples` 指定时它会反复累加 `num_tokens_per_epoch`，直到总 token 数能满足请求的样本数。也就是说，**优先于「正好一个 epoch」，MCore 允许数据不够时自动多训几个 epoch 直到凑够 `num_samples` 个样本**——这是 why `document_index` 的长度是 `num_epochs × 文档数` 的根本原因。

---

## 4. 解法（上）：混合数据 = 再加一级「数据集索引」

单数据源是三级索引，多数据源（blend）则是**再往上叠一级**。`BlendedDataset`（`megatron/core/datasets/blended_dataset.py:25`）同样构建了两个一维索引：

- `dataset_index[idx]`：第 `idx` 个样本来自哪个数据集；
- `dataset_sample_index[idx]`：在那个数据集里取第几个样本。

二者加起来就是一次「外层查表 + 内层查表」：

```python
# megatron/core/datasets/blended_dataset.py:107-109
dataset_id = self.dataset_index[idx]
dataset_sample_id = self.dataset_sample_index[idx]
return {"dataset_id": dataset_id, **self.datasets[dataset_id][dataset_sample_id]}
```

内层的 `self.datasets[dataset_id]` 就是 §3 的 `GPTDataset`，`[dataset_sample_id]` 触发它的 `__getitem__` → 三级索引。所以**混合数据的本质，是在三级索引之上再加一级「数据集级索引」**，把「该不该混、按什么比例混」这件事也降级成纯整数索引。

混合比例怎么落地成索引？看 `_build_indices`（`blended_dataset.py:111-243`）。两种路径：

- 指定了 `size`：调 C++ 的 `helpers.build_blending_indices`（`blended_dataset.py:172-179`），按归一化权重 `weights` 生成长度 `size` 的 `dataset_index`；
- 未指定 `size`：调 `helpers.build_exhaustive_blending_indices`（`blended_dataset.py:184-186`），按权重「满配」把每个数据集的所有样本都排进去。

权重归一化在 `normalize`（`megatron/core/datasets/utils.py:33-46`），就是朴素的 `w / sum(w)`。

这里还有个关键的守护逻辑（`blended_dataset.py:188-199`）：

```python
# megatron/core/datasets/blended_dataset.py:188-199（节选）
dataset_indices, dataset_sizes = numpy.unique(dataset_index, return_counts=True)
for i, (_index, _size) in enumerate(zip(dataset_indices, dataset_sizes)):
    if len(self.datasets[_index]) < _size:
        raise IndexError(
            f"The {self.split.name} blend oversamples the contributing datasets ..."
        )
```

混合索引 build 完后，会统计每个数据集实际被请求了多少样本，一旦「外层请求数 > 内层样本数」就抛 `IndexError`，并提示调大 `mid_level_dataset_surplus`。**这就是 `mid_level_dataset_surplus`（默认 0.005，`blended_megatron_dataset_config.py:72-78`）存在的意义**：预先给每个 mid-level 数据集多切 0.5% 的样本，留出余量，避免外层混合采样时因浮点/向上取整而「越界」。

---

## 5. 解法（下）：Builder 把「混合 + 切分」编排成三种情况

前三节讲的是「一个数据集怎么索引」，但用户填的参数是一堆（weight, prefix）对 + 一个 train/valid/test 切分比例。把这些参数编排成「train/valid/test 三个 split × 每个 split 里若干数据集」的，是 `BlendedMegatronDatasetBuilder`（`megatron/core/datasets/blended_megatron_dataset_builder.py:29`）。

入口 `build`（`blended_megatron_dataset_builder.py:79-136`）只是调 `_build_blended_dataset_splits`（`blended_megatron_dataset_builder.py:138`）并做一致性断言。真正的大头是 `_build_blended_dataset_splits` 里按三种形态分叉：

1. **单一数据源 + `split`**（`blended_megatron_dataset_builder.py:170-171`）：只有一个 prefix 且无权重，直接 `_build_megatron_dataset_splits`，train/valid/test 从同一个分布按 `split_matrix` 切。`split_matrix` 由 `split` 字符串「非重叠 book-ends」自动生成（`blended_megatron_dataset_config.py:48-52`），例如 split=`"9,1,0"` 变成 `[(0, .9), (.9, 1.0), None]`。

2. **同一分布多数据源（`blend`）**（`blended_megatron_dataset_builder.py:162-233`）：train/valid/test 共享同一组权重。会先并行 build 每个 mid-level 数据集（`_build_megatron_datasets_parallel`，`blended_megatron_dataset_builder.py:333`，用 `ThreadPoolExecutor` 多线程），再为每个 split 各 build 一个 top-level `BlendedDataset`。

3. **每 split 独立分布（`blend_per_split`）**（`blended_megatron_dataset_builder.py:238-331`）：train 用一组数据源，valid 用另一组。对每个 split 分别「spoof」一个只有该 split 非空的 split 向量，走一遍和情况 2 相同的流程。

三种情况的共同点是，最终都要算「每个数据集、每个 split 要 build 多少个样本」，这件事由 `_get_size_per_split_per_dataset` 完成：

```python
# megatron/core/datasets/blended_megatron_dataset_builder.py:555-583（节选）
def _get_size_per_split_per_dataset(normalized_weights, target_size_per_split, surplus=0.0):
    assert numpy.isclose(sum(normalized_weights), 1.0)
    sizes_per_dataset = [
        [int(math.ceil(math.ceil(target_size * weight) * (1 + surplus)))
         for target_size in target_size_per_split]
        for weight in normalized_weights
    ]
    return sizes_per_dataset
```

注意这里的**双重向上取整**（`ceil(ceil(target_size * weight) * (1 + surplus))`）：内层把「目标样本数 × 权重」向上取整，外层再乘以 `(1 + surplus)` 并向上取整。两次 ceil 是为了保证「凑出来的 mid-level 样本数 ≥ 外层混合所需」，`surplus` 就是 §4 提到的余量。

还有一个容易踩的坑：**多数据源且有权重时，`size` 不能为 None**（否则报 `RuntimeError`，见 `blended_megatron_dataset_builder.py:312-313`）。因为「按权重混合」必须有明确的总样本数才能定比例，不像「无权重、按各自长度混合」那样可以从数据集长度反推。

---

## 6. 小结

- **两个文件是根**：`.bin`（token 字节流）+ `.idx`（`sequence_lengths` / `document_indices`），用「序列数 == 文档结束序号」的断言强制文档被切成定长序列（`indexed_dataset.py:674-676`）。
- **三级索引是核心**：`document_index`（打乱文档顺序）→ `sample_index`（切样本，`build_sample_idx`）→ `shuffle_index`（打乱样本顺序），shuffle 先后顺序决定 batch 的 token 相关性（`gpt_dataset.py:461-687`）。
- **跨文档样本靠拼接**：`_query_document_sample_shuffle_indices` 把「同文档单段」和「跨文档多段」统一成拼接 + pad（`gpt_dataset.py:409-459`）。
- **`num_epochs` 是「凑出来的」**：`num_samples` 指定时自动多训几个 epoch 直到 token 数够用，这就是 `document_index` 长度含 `num_epochs` 的由来（`gpt_dataset.py:697-717`）。
- **混合 = 再加一级索引**：`BlendedDataset` 用 `dataset_index` + `dataset_sample_index` 把「多数据集加权混合」也变成纯整数查表（`blended_dataset.py:107-109`）。
- **surplus 防越界**：`mid_level_dataset_surplus`（默认 0.005）给 mid-level 数据集预留余量，`_get_size_per_split_per_dataset` 的双重向上取整是它的落地点（`blended_megatron_dataset_builder.py:555-583`）。

下一篇离开数据，讲训练的「存档与复活」——checkpoint 的目录布局、格式分档与加载守护，也就是《[checkpoint 处理]({{< relref "megatron-code-10-checkpoint" >}})》。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/datasets/indexed_dataset.py`、`megatron/core/datasets/gpt_dataset.py`、`megatron/core/datasets/blended_dataset.py`、`megatron/core/datasets/blended_megatron_dataset_builder.py`、`megatron/core/datasets/blended_megatron_dataset_config.py`、`megatron/core/datasets/utils.py`。）
