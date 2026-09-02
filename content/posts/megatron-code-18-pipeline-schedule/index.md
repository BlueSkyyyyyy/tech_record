---
title: "Megatron 源码精读（十八）：Pipeline 调度细节"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "pipeline-parallel", "1f1b"]
categories: ["训练框架"]
weight: 18
series: ["megatron-code"]
---

承接上一篇[《MCore 架构与 layer spec 机制》]({{< relref "megatron-code-17-mcore-arch" >}})，本篇把第 3 篇并行拓扑里一笔带过的 **pipeline parallel 调度（schedule）** 展开讲。核心代码在 `megatron/core/pipeline_parallel/schedules.py`（2653 行）。

PP 的难点不在切模型（`num_layers / pp_size`，见第 17 篇的 `get_num_layers_to_build`），而在**调度**：模型被切成前到后的几个 stage，一个 microbatch 必须依次走完所有 stage 才能算 loss、才能反向。如果串行跑完一个再跑下一个，GPU 大部分时间是空的。**1F1B 调度就是「如何让前向和反向交错、把 GPU 空闲的 bubble 压到最小」**。

---

## 1. 问题背景：bubble 从哪来

先明确 bubble 是什么。假设 `pp_size = 4`、1 个 microbatch：rank0 算完前向把激活发给 rank1，rank0 就**空着等** rank1 算、再等 rank2、rank3，直到 loss 从 rank3 传回来才开始反向。这整段「rank 在等别人、自己不干活」的时间就是 bubble。

如果 microbatch 数量 `num_microbatches` 足够多（≥ `pp_size`），经典 1F1B 能把 bubble 摊薄。一个简单的 bubble 率估计是：

```
bubble = (pp_size - 1) / (num_microbatches + pp_size - 1)
```

`num_microbatches` 越大，bubble 占比越小（但会牺牲一点通信重叠和显存）。Megatron 对这一约束有硬断言：`num_microbatches` 必须 ≥ `pp_size` 才能让流水线「填满」，代码里在 `get_num_microbatches` 校验过（见 `parallel_state` 与 data 模块，本系列第 9 篇数据集篇也提过 micro_batch 概念）。

---

## 2. 现象：三种 schedule，由一个 dispatcher 决定

MCore 不直接暴露「跑哪种 schedule」，而是 `get_forward_backward_func`（`schedules.py:48-163`）根据并行配置返回对应函数：

```python
# schedules.py:156-163（节选）
if pp_size > 1:
    if vp_size is not None:
        forward_backward_func = forward_backward_pipelining_with_interleaving
    else:
        forward_backward_func = forward_backward_pipelining_without_interleaving
else:
    forward_backward_func = forward_backward_no_pipelining
```

三种情况对应三种 schedule：

| 条件 | schedule | 特点 |
|---|---|---|
| `pp_size == 1` | `no_pipelining`（eager） | 每个 microbatch 独立前向+反向，无流水线 |
| `pp_size > 1` 且无 vp | `without_interleaving` | 朴素 1F1B |
| `pp_size > 1` 且有 vp | `with_interleaving` | 每个 rank 交替跑多个 virtual stage 的 1F1B |

`vp_size`（virtual pipeline parallel size）是理解后两者的钥匙：它让**一个物理 rank 交替持有多个模型 chunk**（`num_model_chunks` 个），交错调度进一步压 bubble。第 17 篇 `get_num_layers_to_build` 里 `num_layers_per_virtual_stage`（`transformer_block.py:188-198`）就是算「每个 virtual stage 分到几层」。

---

## 3. 根因：warmup 数量公式是 1F1B 的心脏

1F1B 分三段：**warmup**（只前向，把流水线灌满）、**steady**（前向一个、反向一个，交替）、**cooldown**（只反向，把流水线排空）。其中 warmup 要做多少次，决定了前面 rank 要「提前跑多远」，代码集中在 `get_pp_rank_microbatches`（`schedules.py:878-935`）：

```python
# schedules.py:901-923（节选）
if forward_only:
    num_warmup_microbatches = total_num_microbatches
elif pipeline_parallel_size > 1:
    if virtual_pipeline_parallel_size is None:
        # without_interleaving
        num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
    else:
        # with_interleaving
        num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
        num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
else:
    num_warmup_microbatches = 0
```

两条公式值得逐字读：

1. **朴素 1F1B**（`schedules.py:906`）：`num_warmup = pp_size - pp_rank - 1`。rank0 要 warmup `pp_size-1` 个 microbatch（因为它离 loss 最远，要提前跑最多），rank `pp_size-1`（最后一层）warmup 0 个（它第一个就能出 loss）。这就是「越靠前的 stage 越早开始、跑得越远」。

2. **interleaved 1F1B**（`schedules.py:913-914`）：多出两项。`* 2` 是因为每个 microbatch 在 interleaved 下会经过「前向 chunk + 后向 chunk」两个阶段；`(num_model_chunks - 1) * microbatch_group_size_per_vp_stage` 是**从 chunk0 切到 chunk1 的额外前向**——virtual stage 交替时，`num_model_chunks` 个 chunk 之间要多塞一段前向才能接上。

`schedules.py:925-928` 还有个边界处理：当 `num_warmup >= total_num_microbatches`（microbatch 太少，连 warmup 都填不满），就 `are_all_microbatches_in_warmup = True`，退化成了「纯前向再纯反向」，没有 steady 段。

---

## 4. 解法：warmup / steady / cooldown 三段怎么落成循环

warmup 段的循环（`schedules.py:1675` 起）只做前向 + send/recv，`convert_schedule_table_to_order`（`schedules.py:968-991`）则把三段拼成一条可执行的 order：

```python
# schedules.py:985-990（节选）
order = forward_order[:num_warmup_microbatches]               # warmup：纯前向
for i in range(num_warmup_microbatches, len(forward_order)):
    order.append(forward_order[i])                            # steady：前向+反向交替
    order.append(backward_order[i - num_warmup_microbatches])
if num_warmup_microbatches > 0:
    order.extend(backward_order[-num_warmup_microbatches:])   # cooldown：纯反向
```

docstring 里那个 `PP2 N3M5 with VP2` 的例子（`schedules.py:970-981`）把这条 order 具体化了：`forward_order` 是 `[1,1,1,2,2,2,1,1,2,2]`（1/2 是 chunk 号），`backward_order` 是 `[-2,-2,-2,-1,-1,-1,-2,-2,-1,-1]`（负号表示反向）。`num_warmup=5` 时，最终顺序是「5 个前向 → 前向反向交替 → 5 个反向」，正好是三段。

这段代码还和 TE 的 `make_graphed_callables` 配合（`convert_schedule_table_to_order` 的名字就是「转成 graph 可接受的顺序」）——schedule 一旦静态确定，就能被 cudagraph 捕获固定下来，避免每次迭代现算。

---

## 5. 小结

- **bubble 源于 stage 间等待**，靠「microbatch 足够多（≥ pp_size）+ 前反向交错」摊薄，bubble ≈ `(pp_size-1)/(num_microbatches+pp_size-1)`。
- **dispatcher 选 schedule**：`pp_size` 和 `vp_size` 决定走 eager / 朴素 1F1B / interleaved 1F1B（`schedules.py:156-163`）。
- **warmup 公式是心脏**：朴素 `pp_size - pp_rank - 1`，interleaved 加乘 2 和 chunk 切换项（`schedules.py:906,913-914`）。
- **三段合成一条 order**：warmup（纯前向）→ steady（前反向交替）→ cooldown（纯反向），`convert_schedule_table_to_order` 拼装（`schedules.py:985-990`）。
- **静态 order 可被 graph 捕获**：schedule 确定后交给 TE 的 `make_graphed_callables`（`schedules.py:968-981` docstring）。

下一篇补最后一块拼图：**并行组装地图**——`ParallelismConfig` / `ProcessGroupCollection` 如何把 tp/pp/dp/cp/ep 映射到物理 rank，也就是「某个 tensor 此刻在哪个 rank 上」这张总图。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/pipeline_parallel/schedules.py`。）
