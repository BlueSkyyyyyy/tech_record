---
title: "Megatron 源码精读（三）：并行拓扑——parallel_state.py 精读"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "分布式"]
categories: ["训练框架"]
weight: 3
series: ["megatron-code"]
---

第 2 篇讲了每种并行切什么，这一篇回答更具体的问题：**这些并行组是怎么从 `world_size` 和一行 `order` 算出来的？** 答案集中在 `megatron/core/parallel_state.py`（基线约 2266 行）。

一句话结论：**所有并行组都由同一套混合进制分解生成——`order` 定各轴的进制权重，`token` 定 mask 哪几个轴，`generate_masked_orthogonal_rank_groups` 把 rank 编号翻回坐标。**

## 1. 问题：一个 rank 属于哪些组

每种并行都要做集合通信，而集合通信的载体是 PyTorch 的 `ProcessGroup`——它规定「哪几个 rank 在一起通信」。`parallel_state.py` 的职责就是从世界大小推导出所有组的成员，缓存起来供全库查询（`get_*_group()` / `get_*_rank()` / `get_*_world_size()`）。

除了 DP、TP、PP、CP、EP 这五个基本轴，Megatron 还要 `tp-dp`、`tp-pp`、`tp-cp`、`dp-cp` 这些交叉组，分别对应 MoE 路由统计、模型并行域、带 CP 的 TP 域和带 CP 的 DP 域。它们不是特例，只是同一公式换一个 mask。

## 2. 混合进制：global_rank 就是一组坐标

`generate_masked_orthogonal_rank_groups` 的 docstring 写死了三轴约定：

$$
\text{global\_rank} = r_{tp} + r_{dp}\cdot tp + r_{pp}\cdot tp\cdot dp
$$

这本质上是以 `[tp, dp, pp]` 为权的混合进制数。`decompose` 内嵌函数用 `(index // stride) % size` 把编号拆回坐标；它接收的是压缩后的 shape，stride 由 `prefix_product` 现算，所以坐标不会因为某个轴 size 为 1 而错位。

mask 是关键抽象：`mask[i]=True` 表示这个轴在组内变化。函数把轴分成 masked（组内）与 unmasked（组间）两组，组内坐标乘 masked stride、组间坐标乘 unmasked stride，最后用 `inner_product` 拼回 global rank。于是「要哪个组」就退化成「mask 哪几个轴」。

docstring 里的例子可以直接对答案：8 张卡、`order='tp-pp-dp'`、size 为 `2-2-2` 时，dp 组是 `[[0,4],[1,5],[2,6],[3,7]]`——组索引由 tp 与 pp 坐标组合，组内按 dp stride 走，正好 4 组、每组 2 个 rank。

组大小就是 masked 各轴 size 的乘积，组数是 `world_size // 组大小`——第 6 节用的判据就来自这两行。

## 3. RankGenerator：order 决定权重

`RankGenerator.__init__` 把五个 size 放进 `name_to_size`，再按 `order` 里 token 的出现顺序生成 `ordered_size`——这个列表就是混合进制的进制序列。默认 `order="tp-cp-ep-dp-pp"`（`initialize_model_parallel` 的默认参数）。

两条容易被写错的约束：

- **order 里没出现的轴**：若 size 为 1，自动补到末尾；若 size 不为 1，直接 `RuntimeError`。
- **EP 与 CP 不能同时大于 1**：`assert ep == 1 or cp == 1`，源码给的理由是「CP 只出现在默认 generator，EP 只出现在 expert generator」。

`get_ranks(token)` 是唯一出口：`get_mask` 把 `'tp-dp'` 翻成布尔 mask，交给生成函数；`rank_offset > 0` 时统一平移。注意 `get_mask` 按 token 在 `order` 里出现的位置置位，所以 token 字符串内部的顺序无所谓，`'dp-tp'` 与 `'tp-dp'` 等价。

## 4. initialize_model_parallel：装配顺序有讲究

第一步反推 dp：`model_size = tp * pp * cp`，`data_parallel_size = world_size // model_size`。EP 不在其中——专家侧另建 `expert_decoder_rank_generator`，把 tp 换成 `expert_tensor_parallel_size` 并新开 ep 轴。

创建顺序不是随手写的。**`dp-cp` 组最先建**，原因在注释里：NCCL 只允许第一个带 `NCCL_COLLNET_ENABLE=1` 创建的 communicator 使用 SHARP，而 SHARP 默认给数据并行域（`sharp_enabled_group="dp"`）。所以代码先建 dp-cp 组、用一个 barrier 逼出真正的 communicator，再删掉 `NCCL_COLLNET_ENABLE`，后续组退回普通集合通信。

之后依次是：`dp` → `cp` → `tp-pp` → `tp` → `pp` → embedding / position-embedding → `tp-dp-cp` / `tp-dp` → `tp-cp` → 专家系列（`ep` / `etp` / `tp-ep` / `dp`）。开 `dynamic_context_parallel` 时，还会在 dp-cp 的 rank 列表上按 2 的幂建一批动态组。

创建是集合式的：所有 rank 都要参与每一次 `create_group`，但每个 rank 只把自己的那个组存进模块级全局变量（`if rank in ranks: _XXX_GROUP = group`）。所以 `initialize_model_parallel` 必须全员调用，不能只让部分 rank 进。

还有一个和 MoE 相关的断言：`order` 不以 `pp` 结尾时，必须满足 `expert_data_parallel_size == data_parallel_size`，否则注意力层与 MoE 层的 DP 域对不齐。默认的 `tp-cp-ep-dp-pp` 把 pp 放最后，正好绕开这个限制。

embedding 组不按轴生成，而由回调决定：默认 `default_embedding_ranks` 取每个 PP 组的首、尾两个 stage，`default_position_embedding_ranks` 只取首个。设了 `hierarchical_context_parallel_sizes` 时，CP 组还会被 `create_hierarchical_groups` 再切成多层，供分层 CP 用。

## 5. 查询层：四种 rank 语义

- `get_tensor_model_parallel_rank()`：TP 组内的「位」，管权重切哪一片。
- `get_pipeline_model_parallel_rank()`：PP 组内的「层」，配合 `is_pipeline_first_stage` 判断首尾。
- `get_data_parallel_rank()`：模型副本编号，也就是「第几份样本」。
- `get_context_parallel_rank()`：序列的第几段。

这些 getter 大多带 `check_initialized=True`，没初始化就断言失败，避免调用方静默拿到 `None` 再在集合通信里出玄学问题。

## 6. 算一遍：16 GPU，tp=2，pp=4，cp=1，ep=1

`dp = 16 / (2 × 4 × 1) = 2`；`ordered_size = [2, 1, 1, 2, 4]`，prefix product 是 `[1, 2, 2, 2, 4, 16]`，所以 **tp stride=1、dp stride=2、pp stride=4**（cp/ep 的 size 为 1，不占位）。

- **TP 组 8 个**：`[0,1] [2,3] … [14,15]`。tp 在最低位，TP 组拿相邻 rank——这也是 docstring 要求「相邻 rank 落在同一台 DGX」的原因。
- **PP 组 4 个**：`[0,4,8,12] [1,5,9,13] [2,6,10,14] [3,7,11,15]`，步长 `tp×dp=4`。
- **DP 组 8 个**：`[0,2] [1,3] [4,6] …`，步长正是 tp 的 size。

判据很简单：**组数 = world_size / 该轴 size，步长 = 该轴所有内层轴 size 的乘积**。PP 有 4 个 stage，就该有 4 组、每组 4 个 rank。把 PP 组写成 2 组 `[0,2,4,6]`、`[1,3,5,7]` 是典型错误：那是把 `dp=2` 当成了 `pp=4` 的组大小。

同一套分解也能直接拿到交叉组：`tp-dp` 得到 `[0,1,2,3] [4,5,6,7] …`，`tp-pp` 得到 `[0,1,4,5,8,9,12,13]` 与 `[2,3,6,7,10,11,14,15]`。不需要为交叉组另写一套逻辑，这是 mask 设计的直接收益。

## 7. 小结

- 一个 rank 是混合进制里的一个数：`order` 定权重，`token` 定 mask。
- `RankGenerator` 只做三件事：排 `ordered_size`、翻译 mask、平移 offset。
- dp-cp 最先建是为了 SHARP；默认 SHARP 只给 dp 域。
- 组数 = world_size / size，步长 = 内层轴 size 连乘。
- EP 用第二套 generator，且与 CP 在同一 generator 内互斥。

## 8. 没讲到的与下一篇

- **EP 的独立网格**：`expert_decoder_rank_generator`，细节在第 14 篇。
- **虚拟流水线**：`_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK` 与 `_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE`，一个物理 GPU 持多个 stage。
- **参数 all-gather 组**：`create_all_gather_groups`，与 FSDP/ZeRO 相关，第 8 篇。

下一篇《[激活重计算]({{< relref "megatron-code-04-recompute" >}})》进 `megatron/core/recompute.py`，看「重算激活 + 选择性 checkpoint」怎么把显存换计算。

---
**版本提示（2026-10）**：上游新增 `megatron/core/context_parallel/groups.py` 与 dynamic CP——CP 组不再固定，可按 microbatch 由 sequence packing scheduler 选择；EP/CP 相关的组构造有重构，但 `RankGenerator` 的混合进制数学没变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
