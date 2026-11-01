---
title: "Megatron 源码精读（十八）：Pipeline 调度细节"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "pipeline-parallel", "1f1b"]
categories: ["训练框架"]
weight: 18
series: ["megatron-code"]
---

承接上一篇[《MCore 架构与 layer spec 机制》]({{< relref "megatron-code-17-mcore-arch" >}})，本篇把第 3 篇并行拓扑里一笔带过的 **pipeline 调度**展开。核心代码在 `megatron/core/pipeline_parallel/schedules.py`。

模型被切成前后几段后，一个 microbatch 必须依次走完所有 stage 才能算 loss、才能反向。串行跑，GPU 大部分时间在等别人——这篇回答三个问题：三段式调度怎么排、warmup 数量从哪来、bubble 什么时候压不下去。

一句话结论：**1F1B 的本质是一条静态执行序——warmup 只前向、steady 前后向交替、cooldown 只反向；bubble 由 warmup 长度决定，而 warmup 长度只取决于 `pp_size`、`pp_rank` 和 `vp`。**

## 1. 问题：bubble 从哪来，能压到多小

设 `pp_size = p`、每 rank 的 microbatch 数为 `m`。rank0 前向完必须等 loss 从最后一级传回来才能反向，这段空转就是 bubble。忽略通信、并假设一个 microbatch 的前向与反向耗时相同，1F1B 的 bubble 占比是

$$
\text{bubble ratio} = \frac{p-1}{m+p-1}
$$

`m` 越大越接近 0，但 `m` 受 global batch 与显存限制。注意这是「1F1B、无 interleave、忽略通信」下的估计；开了 interleaved 后每个 rank 同时持有 `vp` 个 chunk，等效并行度约乘 `vp`，bubble 占比近似按 `vp` 缩小（数量级估计，不是精确等式）。

代码里**没有**「`num_microbatches ≥ pp_size`」这类硬断言。真正的保护在 `get_pp_rank_microbatches`：一旦 warmup 数达到或超过总步数，就把 warmup 截到总数并置 `are_all_microbatches_in_warmup=True`（`schedules.py:921-924`），整轮退化成「先全前向、再全反向」，没有 steady 段可交错，bubble 占比接近 1。所以 batch 太小时不报错，只是悄悄变慢。

## 2. 三种 schedule，由一个 dispatcher 选

MCore 不直接暴露「跑哪种 schedule」，而是 `get_forward_backward_func`（`schedules.py:48`）按并行配置返回对应函数：

| 条件 | 返回的函数 | 形态 |
|---|---|---|
| `pp_size == 1` | `forward_backward_no_pipelining` | 每 microbatch 独立前反向，无流水线 |
| `pp_size > 1`、无 vp | `forward_backward_pipelining_without_interleaving` | 朴素 1F1B |
| `pp_size > 1`、有 vp | `forward_backward_pipelining_with_interleaving` | 每 rank 交替持有多个 chunk 的 1F1B |

分派本体在 `schedules.py:156-163`。另有一个例外：当传入的是 `MultiModuleProcessGroupCollection`（一个 rank 同时承载 encoder + LLM 之类的多模块流水）时，函数在 pp/vp 判断之前就直接返回非 interleaved 版本（`schedules.py:149-150`）。

`vp` 是理解后两行的钥匙：它让一个物理 rank 交替持有多段 chunk，每段层数是 `num_layers_per_virtual_stage = num_layers_per_pipeline_rank // vp_size`（`transformer_block.py:191`）。第 17 篇的 `get_num_layers_to_build` 就是在这之上再处理首尾 stage 的 embedding/loss 扣减。

## 3. warmup 数量：一条公式决定整条流水线

`get_pp_rank_microbatches`（`schedules.py:878`）返回四个值：总步数、是否全 warmup、warmup 数、剩余数。总步数是 `num_microbatches * num_model_chunks`——interleaved 下每个 microbatch 要在每个 chunk 上各走一次，所以总量翻 vp 倍。

非 interleaved 只有一行：`num_warmup = pp_size - pp_rank - 1`（`schedules.py:906`）。rank0 离 loss 最远，要提前跑 `p-1` 个；最后一级 warmup 为 0，第一个 microbatch 就能出 loss。

interleaved 是两项相加（`schedules.py:913-914`）：

1. `(pp_size - pp_rank - 1) * 2`：按「到最后一个 stage 的距离」补前向，越靠前补得越多。
2. `(num_model_chunks - 1) * g`：每个 rank 先把自己前 `vp-1` 个 chunk 在当前 group 上跑完，才轮到最后一个 chunk 出 loss；`g` 是 `microbatch_group_size_per_vp_stage`，默认等于 `pp_size`（`model_parallel_config.py:601`）。

这里最容易讲错的是那个 `* 2`：这一段是纯 warmup，只有前向。它也与 vp 无关：vp=2 时这个 2 恰好等于 vp，vp=4 时代码照样乘 2。所以别把它读成「前向段 + 后向段」，也别读成 vp 的倍数——它是写死的预留量。

按代码注释读：先在所有 worker 上跑 `(vp-1)*g` 个前向，再按 stage id 补若干（越靠前补得越多，最后一级补 0，可以立刻进 1F1B）。

另有一个修正项：打开 `overlap_moe_expert_parallel_comm` 时 warmup 再 `+1`（`schedules.py:918-919`），注释给的理由是让每个 1F1B 步的前向与反向计算互相独立。此外 warmup 达到或超过总步数时，会按上一节所述截断。

## 4. 代码落点：三段循环，和 order 的真实用途

interleaved 主循环分三段。warmup 从 `schedules.py:1675` 起，`for k in range(num_warmup_microbatches)` 里只做前向 + send/recv；steady 跑 `num_microbatches_remaining` 步，第 k 步的前向取 virtual id `k + num_warmup`、反向取 `k`；cooldown 再从 `num_microbatches_remaining` 循环到 `total_num_microbatches`（`schedules.py:2073`），只反向。

三段顺序从哪来？先由 `get_schedule_table` 生成「第几个 virtual microbatch 对应哪个 (microbatch_id, chunk_id)」的查找表，再由 `convert_schedule_table_to_order` 把它转成正负号序列。

表的展开是三层：group → chunk → microbatch。默认 `g = pp_size`，意味着同一个 chunk 一次连喂 `pp_size` 个 microbatch 再换下一个 chunk，也就是配置注释里说的深度优先（depth-first）。

```python
# schedules.py:983-990（节选）
forward_order = [chunk_id + 1 for chunk_id in model_chunk_id_table]
backward_order = [chunk_id - num_model_chunks for chunk_id in model_chunk_id_table]
order = forward_order[:num_warmup_microbatches]
for i in range(num_warmup_microbatches, len(forward_order)):
    order.append(forward_order[i])
    order.append(backward_order[i - num_warmup_microbatches])
```

docstring 里 PP2 N3M5 VP2 的例子：`forward_order` 是 `1 1 1 2 2 2 1 1 2 2`，`backward_order` 是 `-2 -2 -2 -1 -1 -1 -2 -2 -1 -1`，`num_warmup=5` 时输出 `1 1 1 2 2 2 -2 1 -2 1 -2 2 -1 2 -1 -1 -2 -2 -1 -1`（`schedules.py:969-981`）。

关键一点：docstring 明说这个函数是把「tunable schedule lookup table」转成 `te.make_graphed_callables()` 接受的 order 格式，**用途是 CUDA graph 捕获**，不是运行时通用调度器。eager 路径直接用 `schedule_table` 做索引，不会先转成 order 再解释。

反向的 chunk 次序是前向的镜像：`get_model_chunk_id` 在 `forward=False` 时返回 `num_model_chunks - 1 - id`（`schedules.py:1344-1350`）。这也是 `backward_order` 用 `chunk_id - num_model_chunks` 写成负数的原因。

## 5. 工程细节与坑

**interleaved 对 pp_size 有真断言。** 开 `overlap_p2p_comm` 时要求 `pp_size > 1`；不开时要求 `pp_size > 2`，理由是同一对 rank 之间一个通信批次里会出现多组 send/recv，没法重叠（`arguments.py:1049-1063`）。

**`overlap_p2p_comm` 只在 interleaved 下存在。** 非 interleaved 函数开头直接 raise（`schedules.py:2313-2316`），参数解析也会在 vp 为空时把它和 `align_param_gather` 一起强制关掉（`arguments.py:1064-1066`）。细节见第 20 篇。

**`g` 是「通信 vs bubble」的旋钮。** 默认 `g = pp_size` 即深度优先编排；显式调大，同一个 virtual stage 会连续喂更多 microbatch，集合通信次数下降但流水线填充变慢，需要按模型实测。

**vp 不是随便设。** vp 为 1 会被归一成 None（退回非 interleaved），且每 rank 层数必须能被 vp 整除（`transformer_block.py:188-191`）。

## 6. 小结

- bubble 源于 stage 间等待，1F1B 用「warmup 前向 → steady 交错 → cooldown 反向」把它摊薄；`m` 太小时退化成全 warmup，bubble 接近 1。
- `get_forward_backward_func` 只按 `pp_size` / `vp_size` 分派，多模块集合是提前返回的例外。
- warmup 数决定一切：非 interleaved 是 `p - rank - 1`；interleaved 是「按 stage 距离补前向」加「(vp-1) 个 chunk 的 group」。
- interleaved 的 `* 2` 是纯前向阶段里每跨一级 stage 的预留步数，与「前向+后向」无关，也与 vp 无关。
- `convert_schedule_table_to_order` 服务于 CUDA graph 捕获，不是 eager 路径的调度器。

---

**版本提示（2026-10）**：上游把 dynamic CP 接进了标准 train/eval schedule（#7047，配套 `sequence_packing_scheduler=default_dynamic_cp`）；新增 packed MTP 在 CP 边界的交换（#8389）与 sequence-packing scheduler 的修复（#8378）。1F1B 的三段骨架与 `get_pp_rank_microbatches` 的公式没有变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
