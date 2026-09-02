---
title: "Megatron 源码精读（十九）：并行组装地图"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "parallel-state", "process-group", "rank-mapping"]
categories: ["训练框架"]
weight: 19
series: ["megatron-code"]
---

这是本系列「并行」主线的收口篇。前面第 2 篇讲了各类并行的原理，第 3 篇精读了 `parallel_state.py`，第 16 篇展开 CP、第 18 篇展开 PP 调度。本篇回答一个贯穿始终的问题：**给定一个物理 rank，它此刻在 tp/pp/dp/cp/ep 的哪个位置？反过来，某组参与同一集合通信的 rank 是谁？** 这就是「并行组装地图」——把抽象的并行度参数映射成真实的进程组（process group）与 rank 列表。

核心代码在 `megatron/core/parallel_state.py`（2266 行）与 `megatron/core/process_groups_config.py`（718 行）。

---

## 1. 问题背景：一桌子通信，谁跟谁算

分布式训练的每一步都要回答两个问题：

1. **谁是同组**：比如做一次 tensor parallel 的 all-reduce，哪些 rank 参与？
2. **我是谁**：我在某个轴上排第几（`tp_rank` / `pp_rank` / ...），用于决定分片边界（如 `layer = pp_rank * layers_per_stage`）。

早期 Megatron（老 `mpu`）用一堆散落的全局变量 `_TENSOR_MODEL_PARALLEL_GROUP` 之类的来存这些答案，谁要用就 `from megatron import mpu` 再调 `get_xxx_group()`。MCore 化之后，这些「答案」被收敛成三类对象：

- **`RankGenerator`**：纯数学，按一个 `order` 字符串 + 各轴大小，算出任意子集（tp、tp-dp、pp…）的 rank 分组。
- **`initialize_model_parallel`**：用 `RankGenerator` 算出的 rank 分组去真实 `new_group` 建进程组，并记录每个 rank 在轴上的下标。
- **`ProcessGroupCollection`**：一个 `dataclass`，把散落的进程组统一收进一个对象，`use_mpu_process_groups()` 一键拉取，再传给 `TransformerBlock` / `DDP` / `finalize_model_grads`。

需要澄清一点：本系列第 18 篇结尾预告的 `ParallelismConfig` 在当前基准 commit 里并不存在（搜索 `megatron/` 无此符号）。真正扮演「配置」角色的，是 `RankGenerator` 的构造参数和 `initialize_model_parallel` 的几十个形参。所以本篇以这三者为骨架。

---

## 2. 现象：假设与 rank 地图

先看一个 16 GPU 的经典例子（`initialize_model_parallel` 的 docstring 原文，`parallel_state.py:682-692`）：`tp=2`、`pp=4`、`cp=1`，那么 `dp = 16 / (2*4) = 2`。默认 `order="tp-cp-ep-dp-pp"`（`parallel_state.py:560`），Min 掉 size=1 的轴后实际起作用的顺序是 `tp-dp-pp`。

此时：

- 8 个 dp 组：`[g0,g2] [g1,g3] [g4,g6] [g5,g7] [g8,g10] [g9,g11] [g12,g14] [g13,g15]`
- 8 个 tp 组：`[g0,g1] [g2,g3] ... [g14,g15]`
- 4 个 pp 组：`[g0,g4,g8,g12] [g1,g5,g9,g13] [g2,g6,g10,g14] [g3,g7,g11,g15]`

关键观察：**tp 组是「相邻奇数偶数对」，pp 组是「隔 4 取 1」，dp 组是「隔 2 对偶」**。这不是偶然——它精确对应下面第 3 节的基公式。

---

## 3. 根因：秩生成器的基公式

`generate_masked_orthogonal_rank_groups`（`parallel_state.py:250-356`）是整张地图的数学心脏。它的 docstring 直接给出正交并行的基公式（以 `tp-dp-pp` 为例）：

```
global_rank = tp_rank + dp_rank * tp_size + pp_rank * tp_size * dp_size   (1)
```

这就是**「rank ↔ 多维下标」的互转公式**。要理解它，用 `prefix_product`（`parallel_state.py:303-307`）这个 stride 数组：

- 若 `parallel_size = [tp, dp, pp] = [2, 2, 4]`，则 `global_stride = prefix_product([2,2,4]) = [1, 2, 4, 8]`。
- 每个轴的下标 `idx_i` 乘上 `stride[i]` 再加总，就是 global_rank：`tp_rank*1 + dp_rank*2 + pp_rank*4`。

这等价于把 rank 的二进制/混合进制逐位拆开。于是「求某个子组的 rank 列表」就变成一道很干净的数学题：

1. `mask` 标记哪些轴是这次组的成员。例如求 `dp` 组，mask = `[False, True, False]`（在 `order` 对应的位置）。
2. `masked_shape` = 组内轴的大小（这里是 `[dp=2]`），`unmasked_shape` = 组外轴（`[tp=2, pp=4]`）。
3. `group_size = prod(masked_shape) = 2`，`num_of_group = world_size / 2 = 8`。
4. 每个组：`group_index` 在组外轴上 `decompose`（`parallel_state.py:313-331`，即「除以 stride 取模」，把一维编号拆回多维），组内 rank 在掩码轴上 `decompose`，最后 `inner_product` 拼回 global_rank。

`decompose`（`parallel_state.py:313-331`）本质是 `index → idx`：`idx[i] = (index // stride[i]) % shape[i]`，并断言 `sum(idx[i]*stride[i]) == index` 自校验。`inner_product`（`parallel_state.py:310-311`）就是基公式的求和版。

`RankGenerator`（`parallel_state.py:444-519`）把这套数学包起来：

- 构造时校验「EP 和 CP 不能同时 >1」（`parallel_state.py:450-452`），因为 CP 只属于默认生成器、EP 只属于 expert 生成器。
- `world_size = tp * dp * pp * cp * ep`（`parallel_state.py:461`）。
- `order` 缺的轴若 size ≠ 1 直接 `RuntimeError`，size == 1 的轴自动补到结尾（`parallel_state.py:473-480`）。
- `get_ranks(token)`（`parallel_state.py:503-519`）把 `token`（如 `"tp-dp"`）转成 mask 再交给 `generate_masked_orthogonal_rank_groups`，最后统一加 `rank_offset`。

所以 `initialize_model_parallel` 里那一大段 `for ranks in decoder_rank_generator.get_ranks('tp')`（`parallel_state.py:1018`）到 `get_ranks('pp')`（`parallel_state.py:1094`），本质都是「用同一套基公式，换不同 mask，枚举出每一种正交子组」。

---

## 4. 两种 rank 生成器：attention 域 vs expert 域

`initialize_model_parallel` 建了**两个** `RankGenerator`（`parallel_state.py:769-800`）：

- `decoder_rank_generator`：`ep=1`、`cp=context_parallel_size`，覆盖 attention/embedding 等非 MoE 部分。
- `expert_decoder_rank_generator`：`cp=1`、`ep=expert_model_parallel_size`、`tp=expert_tensor_parallel_size`（默认等于 `tp`），覆盖 MoE 专家部分。

为什么拆两个？因为 **EP（专家并行）和 CP（上下文并行）都是「切 batch/序列」类的轴，但切的对象不同**，且两者 rank 布局通常不能简单叠加。注释和断言直接点破：

- 生成器内部强制 `ep == 1 or cp == 1`（`parallel_state.py:450-452`），一个生成器里不能同时有 EP 和 CP。
- 两个生成器的 PP 组必须一致：断言 `decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp")`（`parallel_state.py:808-811`）。这是合理的——同样的物理 GPU 既跑 attention 又跑 expert，pipeline 切分必须对齐，否则一个 microbatch 在两种域里的 pp rank 就不一致。
- 除非 `order` 以 `pp` 结尾，否则要求两个域的 `data_parallel_size` 相等（`parallel_state.py:802-806`）。

这条断言也解释了第 14 篇（MoE）里 EP group 为什么是另一套：`get_ranks('ep')` 走的是 expert 生成器，`expt_tp` / `tp_ep` / `tp_ep_pp` 这些组合组（见 `ProcessGroupCollection` 的 `pg_to_func` 映射，`process_groups_config.py:204-214`）也都是基于 expert 生成器算的。

---

## 5. 组装：统一进一个 `ProcessGroupCollection`

算完 rank 分组后，`initialize_model_parallel` 逐个 `create_group`（`parallel_state.py:228-247`，内部 `torch.distributed.new_group` 包一层，负责 `group_desc`/timeout 兼容与登记 `_global_process_group_list`）。但下游模块不想关心几十个全局变量名，MCore 于是用 `ProcessGroupCollection`（`process_groups_config.py:26-260`）做了一个**统一收纳 + 懒加载**。

它是个 `dataclass`，每个字段 `init=False`（`process_groups_config.py:71-140`），必须在实例化后单独赋值。三类字段：

- **模型并行组**：`tp / pp / mp / embd / pos_embd / cp / tp_cp / hcp`（`process_groups_config.py:70-92`）。
- **专家并行组**：`ep / expt_tp / tp_ep / tp_ep_pp / tp_dp_cp`（`process_groups_config.py:94-108`）。
- **数据并行组**：`dp / dp_cp / dp_cp_ag / expt_dp / intra_dp_cp / intra_expt_dp / inter_dist_opt / intra_dist_opt`（`process_groups_config.py:110-140`）。

两个关键类方法：

1. `use_mpu_process_groups(required_pgs=None)`（`process_groups_config.py:168-260`）：默认把所有字段都从 `parallel_state` 拉一遍；也可以传 `required_pgs=['tp','cp']` 只拉需要的（见第 15 篇 LLaVA 里 attention 只收 `['tp','cp']`，`attention.py:333`）。它内部是一张 `pg_to_func` 映射表（`process_groups_config.py:191-250`），把字段名映射到 `parallel_state.get_xxx_group(check_initialized=False)` 的 `partial`。
2. `setup_process_groups_for_optimizer(...)`（`process_groups_config.py:262-`）：给优化器/DDP 准备 dp 系进程组，必要时 gloo 版本与 `intra_/inter_` 拆分（对应 distributed optimizer 的多实例，第 11 篇提过）。

下游用法高度一致，都是「塞进构造、默认回退到 `use_mpu_process_groups()`」：

```python
# transformer_block.py:289-295（节选）
pg_collection: Optional[ProcessGroupCollection] = None
...
if pg_collection is None:
    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
```

这套「一个对象贯穿模型前向、DDP、梯度 finalize」的设计，正式把第 3 篇那些散落全局变量收敛成了一张可传参的地图。

---

## 6. 秩在轴上的位置：`get_xxx_rank`

有了进程组，还要知道「我自己在轴上的下标」。这些下标不单独存，而是**从全局 rank 反向分解**得到。`get_tensor_model_parallel_rank`（`parallel_state.py:1663`）、`get_pipeline_model_parallel_rank`（`parallel_state.py:1671`）、`get_data_parallel_rank`（`parallel_state.py:1824`）、`get_context_parallel_rank`（`parallel_state.py:1845`）、`get_expert_model_parallel_rank`（`parallel_state.py:1904`）等，在 chunk 维度返回对应的 `_xx_RANK` 全局变量。

这些 `_xx_RANK` 的值，是在建组时顺手记下的。以 pp 为例，分 chunk 后 `_PIPELINE_MODEL_PARALLEL_GROUP` 会从单个 group 升级成 list（`parallel_state.py:1113-1121`），配合 `_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE`（在 `initialize_model_parallel` 里当 `virtual_pipeline_model_parallel_size` 非空时置为 0/vp_size，`parallel_state.py:744-747`），支撑第 18 篇的 interleaved 调度。

至此，「谁跟我一组」用进程组回答，「我是第几段」用 `get_xxx_rank` 回答——两张表合起来，就是这张完整的并行组装地图。

---

## 7. 小结

- **一张公式撑起整张地图**：`global_rank = tp_rank + dp_rank*tp_size + pp_rank*tp_size*dp_size`（`parallel_state.py:273`），`generate_masked_orthogonal_rank_groups` 用 `stride`（`prefix_product`）+ `decompose`/`inner_product` 把「求任意子组的 rank 列表」变成纯数学（`parallel_state.py:250-356`）。
- **`RankGenerator` 是配置载体**：`order` 字符串决定轴顺序，缺轴校验（`parallel_state.py:473-480`），`get_ranks(token)` 换 mask 枚举子组（`parallel_state.py:503-519`）。
- **两个生成器分域**：decoder（CP 域）与 expert（EP 域）分开，但强制 PP 组一致（`parallel_state.py:769-811`）。
- **`ProcessGroupCollection` 统一收纳**：字段 `init=False`（`process_groups_config.py:71-140`），`use_mpu_process_groups()` 一键懒加载（`process_groups_config.py:168-260`），下游默认回退（`transformer_block.py:295`）。
- **下标反向分解**：`get_xxx_rank` 从 `_xx_RANK` 取，PP 分 chunk 后升级为 list 支持 interleaved（`parallel_state.py:1113-1121`）。

行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。到这一篇，并行主线（原理 → 拓扑 → CP → PP 调度 → 组装地图）闭合。
