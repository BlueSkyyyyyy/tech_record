---
title: "Megatron 源码精读（九）：数据集处理"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "数据集", "dataloader"]
categories: ["训练框架"]
weight: 9
series: ["megatron-code"]
---

上一篇[《ZeRO-1 / FSDP 实现》]({{< relref "megatron-code-08-zero-fsdp" >}})讲模型怎么在 DP 上省显存，本篇掉头：喂进训练循环的一个个样本，是从磁盘上的哪几个数组算出来的。

一句话结论：**MCore 把语料离线 tokenize 成 `.bin`（连续 token 流）加 `.idx`（长度与文档边界），训练时只做整数查表——先按文档流水线切样本，再用一层排列打乱样本的下标，全程不碰原始文本。**

这条管线分三层：`IndexedDataset` 管磁盘格式，`GPTDataset` 管「文档 → 样本」的三级索引，`BlendedDataset` 管多数据源加权混合。

## 1. 为什么不能「读文本行、在线 tokenize」

在线 tokenize 是 CPU 密集的，每个 worker、每个 epoch 都要重跑一遍，很容易成为训练前的瓶颈；更麻烦的是随机访问——GPT 样本要从某篇文档的任意位置切一段 `sequence_length`（还可能跨文档），变长文本做不到 O(1) 偏移定位。

所以业界通行做法是 ETL 阶段一次性 tokenize、训练阶段零 tokenize。MCore 把它固化成一对齐名的文件：`.bin` 装所有 token id 的连续字节流，`.idx` 装「每条序列多长、每篇文档到第几条序列结束」。前者解决「不用重跑分词」，后者解决「按偏移取序列」。

## 2. `.idx` 是三块数组加一个 34 字节头

`_IndexReader` 先读 34 字节头：9 字节 magic、8 字节 version、1 字节 dtype code、8 字节序列数、8 字节文档数，之后才是三块定长数组（`indexed_dataset.py:264-278`）：

- `sequence_lengths`（int32，长度 = 序列数）：每条序列的 token 数；
- `sequence_pointers`（int64，等长）：每条序列在 `.bin` 里的字节偏移；
- `document_indices`（int64，长度 = 文档数 + 1）：文档边界，首元素恒为 0。

多模态数据集还会多一块 `sequence_modes`（int8）；纯文本走不到它。

`IndexedDataset.__init__` 末尾用三条断言把这套结构钉死（`indexed_dataset.py:674-676`，`fast_cache_load` 时跳过）：

```python
assert self.index.sequence_lengths.shape[0] == self.index.document_indices[-1]
assert self.index.sequence_lengths.shape[0] == len(self.index)
assert self.index.sequence_lengths.shape[0] == self.index.sequence_count
```

翻译过来只有一句：**序列总数 == 最后一个文档的结束下标**。于是文档 i 覆盖序列区间 `[d_i, d_{i+1})`，这就是「变长文档压成定长序列 + 边界指针」的手段。

容易写错的地方是 `document_indices` 的长度：它在 builder 里以 `[0]` 初始化、每篇文档 append 一次结束位置，所以是「文档数 + 1」个元素。日志也印证了这点——文档数打印的是 `document_indices.shape[0] - 1`（`indexed_dataset.py:333`）。把它说成「每个文档的结束序号」会少一个位置。

## 3. 三级索引：先打乱文档，再切样本，最后打乱样本

`_build_document_sample_shuffle_indices`（`gpt_dataset.py:461`）的 docstring 把三者的形状和职责写得很直白：document index 是 1-D 有序文档 id 数组，sample index 是 2-D 的「每个样本的起点」，shuffle index 是 sample index 下标范围的一个随机排列。

### 3.1 document_index：长度是 `num_epochs × 文档数`，但 shuffle 可能分两段

`_build_document_index`（`gpt_dataset.py:720`）用 `numpy.mgrid` 把文档 id 平铺 `num_epochs` 份、`reshape(-1)`、再 `shuffle`。但打乱范围受 `separate_final_epoch` 控制：为真时前 `num_epochs - 1` 轮打乱成一段、最后一轮单独打乱后拼接（`gpt_dataset.py:741-751`），docstring 的说法是「把最后一个 epoch 排除在全局 shuffle 之外」。

这个开关由「最后一轮样本数是否低于整轮的 80%」决定（阈值写死 `0.80`）。为什么要拆？最后一轮常常只装了很少的数据，混进全局 shuffle 会被打散到每一轮里，等于每轮都掺一点尾巴；单独成段才能让它只出现在最后。

还有一个反直觉的点：`num_epochs > 1` 且不拆段时，文档是被全局打散的，「第几个 epoch」由 token 计数（`num_tokens_per_epoch`）划出，而不是由数组位置。

### 3.2 sample_index 不是 shuffle 出来的

它由 C++ `helpers.build_sample_idx` 顺序扫描上面那条已打乱的文档流水线生成（`gpt_dataset.py:604-612`）。形状 `[样本数 + 1, 2]`，每行 `(doc_index, offset)`；`[idx]` 与 `[idx+1]` 两行共同界定第 idx 个样本，所以多出来的那一行是哨兵——`__len__` 就写成 `sample_index.shape[0] - 1`。

样本数由 token 总量除出来（`drop_last_partial_sequence=True` 时向下取整，验证集可关掉）：

$$
n_{\text{samples}} = \left\lfloor \frac{k \cdot T_{\text{epoch}} - a}{L} \right\rfloor
$$

其中 $k$ 是 epoch 数、$T_{\text{epoch}}$ 是单轮 token 数、$a$ 是 `add_extra_token_to_sequence`（0 或 1）、$L$ 是 `sequence_length`。

### 3.3 shuffle_index：打乱的是「取第几个样本」

`_build_shuffle_index`（`gpt_dataset.py:754`）造 `[0, num_samples)` 的排列，`__getitem__` 第一件事就是 `idx = self.shuffle_index[idx]`。打乱的是下标，样本内容（文档内连续的 token）不变——这是「shuffle 又不破坏文档内连续性」的全部技巧。

它同样能分两段：`separate_final_epoch` 时第一段是 `[0, num_samples_sans_final_epoch)`、第二段是剩下的，各自打乱再拼接。效果是最后一轮的样本不与前面的混排。

**一个常见误解是「文档和样本的 shuffle 都在 `build_sample_idx` 之前」**，实际顺序是「打乱文档 → 切样本 → 打乱样本下标」：切样本之前只有文档在乱，`sample_index` 本身从不被 shuffle。

## 4. 跨文档样本：拼起来，不够就 pad

`_query_document_sample_shuffle_indices`（`gpt_dataset.py:374`）先取 `sample_index[idx]` 和 `sample_index[idx+1]`，按 `doc_index_beg == doc_index_end` 分两支（`gpt_dataset.py:409-439`）：单文档时取 `[beg_offset, end_offset + a)` 一段；跨文档时循环 `[beg, end]`，起始文档从 `beg_offset` 取到篇尾、中间文档整篇取、结束文档取到 `end_offset + a`，最后 `numpy.concatenate`。

长度不足 `sequence_length + a` 就补 pad token（`gpt_dataset.py:449-453`）。函数还返回每段的 `document_lengths`，供 `inter_document_masking` 构造 `cu_seqlens`——开启后注意力被限制在单篇文档内，position id 也按文档重置。

那个 `+a` 是为标签右移服务的：`add_extra_token_to_sequence` 开时 `tokens = text[:-1]`、`labels = text[1:]`；关时改成 `labels = roll(text, -1)` 并把末位设成 pad（`gpt_dataset.py:267-273`）。两条分支的形状一样、语义不同，改代码时很容易只改一边。

## 5. num_epochs：不指定 num_samples 就只跑一轮

`_get_num_epochs`（`gpt_dataset.py:697`）：`num_samples is None` 直接返回 1；否则算出需要的 token 数 `num_samples × sequence_length + a`，从 1 个 epoch 起累加，`while num_tokens < num_tokens_requested` 就一直加。

这就是「数据不够就多训几轮，凑够请求的样本数」，也是 `document_index` 长度为 `num_epochs × 文档数` 的由来。代价要说清：多出来的轮次是重新扫同一批文档（顺序不同），不复制数据，也不改变总量。

## 6. BlendedDataset：在三级索引之上再加一级

`BlendedDataset` 维护两个等长一维数组：`dataset_index[idx]`（第 idx 个样本来自哪个数据集，int16）和 `dataset_sample_index[idx]`（在那里取第几个样本，int64）。`__getitem__` 就是两次查表后转调内层（`blended_dataset.py:107-109`）。

造表有两条路径：给了 `size` 走 `build_blending_indices`，没给走 `build_exhaustive_blending_indices`（`blended_dataset.py:169-186`）。

前者不是「按比例分段切」，而是逐样本的贪心误差最小化——每步选误差 `w_i × n - used_i` 最大的那个数据集（`n` 是当前样本序号），取走后把 `used_i` 加一；全程确定性、无随机数。后者按权重整数份把每个数据集的全部样本铺进去，所以要求权重是整数，且不归一化（`blended_dataset.py:55-56`）。

建完表会统计每个数据集被请求了多少次，一旦超过它的实际长度就抛 `IndexError`，并在消息里提示调大 `mid_level_dataset_surplus`（`blended_dataset.py:188-199`）。这个默认 0.005 的余量，落地点是 `_get_size_per_split_per_dataset` 的双重向上取整（`blended_megatron_dataset_builder.py:575-581`）：

$$
s_i = \left\lceil \left\lceil W \cdot w_i \right\rceil \cdot (1 + \text{surplus}) \right\rceil
$$

内层先把「目标样本数 × 权重」取整，外层再留出 surplus 的余量，两次 ceil 保证 mid-level 数据集不会比外层要的少。surplus 为 0 时退化成单次 `ceil(W·w_i)`，两道 ceil 的差别只在 surplus 大于 0 时体现。

## 7. 工程细节与坑

- **给权重不给 size 是错的，但两条 blend 路径报的异常不同**：同一分布（`blend`）抛 `ValueError`，`blend_per_split` 抛 `RuntimeError`（`blended_megatron_dataset_builder.py:213-215`、`:313`）。按其中一种写文档，另一种就对不上。
- **缓存按内容哈希命名**：索引文件名里带 `unique_description_hash`（md5），改 seed、权重或 split 会生成新缓存，不会误读旧表。
- **无权重时按体量混合**：权重取各 mid-level 数据集的长度，`size` 再取 `min(size, sum(len))`，等价于按数据量分配，不需要显式比例。

## 8. 小结

- **格式是根**：`.bin` 存 token 流，`.idx` 存 `sequence_lengths` / `sequence_pointers` / `document_indices`，断言强制「序列数 == 最后文档的结束下标」。
- **三级索引各司其职**：文档索引打乱出场顺序，`build_sample_idx` 切样本（2-D，末行是哨兵），shuffle index 打乱取样本的下标。
- **shuffle 只发生两次**，都在文档打乱之后：一次是文档，一次是样本下标；`sample_index` 自身从不被 shuffle。
- **跨文档样本是拼接加 pad**，`document_lengths` 顺带支撑 `inter_document_masking` 的 `cu_seqlens`。
- **`num_epochs` 是凑出来的**：`num_samples` 给定时重复扫数据直到 token 够用。
- **混合是第四级索引**：`dataset_index` + `dataset_sample_index` 把加权混合降级成查表，surplus 与双重 ceil 负责不越界。

下一篇离开数据，讲训练的「存档与复活」：checkpoint 的目录布局、格式分档与加载守护，也就是《[checkpoint 处理]({{< relref "megatron-code-10-checkpoint" >}})》。

---
**版本提示（2026-10）**：`megatron/core/datasets/` 的索引格式（`.idx` 头 + 三块数组）与三级索引构建逻辑在 dev 上未变，`helpers.cpp` 与 `blended_dataset.py` 一字未动。新增的是两头：`IndexedDatasetBuilder.add_documents` 支持按 awkward 数组批量写入（`.idx` 布局不变）；`megatron/training/datasets/varlen_dataset.py` 支持 Nemotron-SFT / RP2 / CodeForge 三种样本格式，并在 `data_samplers.py` 里为 packed global batch 预取（#5771）。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
