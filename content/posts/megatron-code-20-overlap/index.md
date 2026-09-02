---
title: "Megatron 源码精读（二十）：通信与计算 overlap"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "compute-comm-overlap", "ddp", "overlap", "流水线并行"]
categories: ["训练框架"]
weight: 20
series: ["megatron-code"]
---

上一篇[《并行组装地图》]({{< relref "megatron-code-19-parallel-assembly" >}})讲清了「谁跟谁通信」。但通信本身是要花时间的——一张 A100/H100 的 NVLink 带宽有限，all-reduce / all-gather / pipeline 的 send-recv 如果都同步等，GPU 算力就会空转。本篇展开并行主线的最后一环：**通信与计算如何 overlap 提升性能（通算并行）**。

核心素材分布在两个文件：

- `megatron/core/distributed/distributed_data_parallel.py`（700 行）与 `megatron/core/distributed/param_and_grad_buffer.py`（2041 行）：DP 侧的 **param all-gather 与 grad reduce-scatter 重叠**。
- `megatron/core/pipeline_parallel/schedules.py` 与 `p2p_communication.py`：PP 侧的 **send/recv 与计算重叠**、输出张量伪释放。

---

## 1. 问题背景：通信暴露在哪

分布式训练里，通信按出现时机分两类：

1. **前向的 param all-gather**：FSDP/ZeRO 下每 rank 只持有参数的 1/dp 分片，前向前必须 all-gather 出完整权重。这是「通信在前，计算在后」，若不重叠，每个 microbatch 前都硬等一次。
2. **反向的 grad reduce**：算完所有 microbatch 的梯度后做 all-reduce（或 reduce-scatter + 分布式优化器）。这是「计算在前，通信在后」，若不重叠，反向尾部空等。
3. **pipeline 的 send/recv**：stage 间传激活/梯度，天然和计算交错但默认同步 `wait`。

上述每一处「硬等」都会把 GPU 暴露成 `bubble`（第 18 篇已算过 PP 的 bubble）。通算并行的目标，就是用**双 stream + 惰性 dispatch + 反向 hook** 把这三处通信藏到计算里。

下面按「现象 → 根因 → 解法」逐类拆。

---

## 2. grad reduce 与反向计算重叠

### 2.1 现象

默认 `overlap_grad_reduce: bool = False`（`distributed_data_parallel_config.py:18`）。关闭时，反向 pass 跑完所有层的梯度，再一次性 `all-reduce`——这中间的 all-reduce 时间全暴露。开启后，all-reduce 会被**切成很多小 bucket 的异步操作，边算边传**。

### 2.2 根因：为什么能切？

梯度是按「bucket」聚合成连续 buffer（`param_and_grad_buffer.py` 的 `_ParamAndGradBucket`），把若干参数的梯度拼成一段连续内存。**反向计算天然从输出层往输入层逐层出梯度**——即 bucket 是按「反向到达顺序」依次 ready 的。所以「最后一个输出层的 bucket 梯度算完」的那一刻，就可以先把它 reduce 掉，不必等后面输入层的梯度。

### 2.3 解法：反向 hook 驱动 dispatch

关键机制三件套：

**① 反向 hook 登记就绪**：`DistributedDataParallel.__init__` 里给每个 `param` 的 `grad_acc` 注册 backward post hook（`distributed_data_parallel.py:424`）。hook 主体 `_make_backward_post_hook`（`distributed_data_parallel.py:500-529`）做两件事：先把 `param.grad` 累进 `param.main_grad`，再调 `register_grad_ready`（若开了 overlap）。

**② `register_grad_ready` 判定「bucket 满没满」**（`param_and_grad_buffer.py:913-935`）：每个 bucket group 维护 `per_param_grad_ready_counts`，当它等于 `golden_per_param_grad_ready_counts`（`param_and_grad_buffer.py:307`，首个 batch 记账得到的期望计数）时，说明这个 bucket 里所有参数的梯度都到位了，立即 `start_grad_sync` 发起异步 reduce。

**③ `start_grad_sync` / `finish_grad_sync` 的 dispatch-wait 分离**（`param_and_grad_buffer.py:651` / `:833`）：overlap 模式下 `start_grad_sync` 只 dispatch（`async_op=True`），真正 `wait` 延后到 `finish_grad_sync`。`finish_grad_sync`（`param_and_grad_buffer.py:833-878`）里对 `grad_reduce_finished` 做幂等保护——因为 bucket 之间还会互相提前 drain（见下）。

还有一个常配的开关 `delay_wgrad_compute`（`distributed_data_parallel_config.py:202`）：它把 wgrad（权重梯度）的计算本身也往后拖到更靠近其 reduce 的时机，配合 TE ≥ 2.8 才能用的 `overlap_grad_reduce`（`arguments.py:2103-2106`）。

---

## 3. param all-gather 与前向计算重叠

### 3.1 现象

`overlap_param_gather: bool = False`（`distributed_data_parallel_config.py:21`），默认关闭。开启后，前向之前不再同步 `all_gather` 全量权重，而是**每个 bucket 按需异步拉取**：模块真正用到某参数的那一刻才等它 ready。

### 3.2 解法：forward pre-hook + 反向链式 dispatch

**① 前向 pre-hook 触发等待**：`enable_forward_pre_hook`（`distributed_data_parallel.py:436-446`）给每个子模块注册 forward pre-hook，`_make_forward_pre_hook`（`distributed_data_parallel.py:468-491`）里对模块直接持有的参数调 `_finish_param_sync_for_bucket_group`——即「这个模块要前向了，先确保它的参数 gather 完成」。于是 gather 的等待被推到「最晚必须用到参数」的时刻，前面别的模块已经边算边等。

**② 链式 dispatch 让下一次 gather 提前飞**：`next_param_gather_bucket_group` 按**反向顺序**串联 bucket（`distributed_data_parallel.py:342-348`，倒序迭代给每个 bucket 挂上「下一个」指针）。`finish_param_sync`（`param_and_grad_buffer.py:611-649`）里，当前 bucket wait 完之后，顺手 `start_param_sync` 把下一个 bucket 的 gather 也发出去——这样第 N 个 bucket 的 gather 在第 N-1 个 gather 完成时就已在飞，而前向还在用更早的 bucket。

**③ dispatch 本身的可选异步**：`start_param_sync`（`param_and_grad_buffer.py:448-609`）里 `async_op = self.ddp_config.overlap_param_gather and not force_sync`（`param_and_grad_buffer.py:477`），决定 `all_gather` / `_coalescing_manager` 走异步还是同步；分布式优化器路径用 `_coalescing_manager` 把多个 `all_gather_into_tensor` 合并成一次集合通信（`param_and_grad_buffer.py:583-601`）。

> 细节坑：`skip_next_bucket_dispatch`（`param_and_grad_buffer.py:611`）用于 `align_param_gather` 或「与 optimizer step 重叠」的路径，避免链式 dispatch 和显式调度打架。`overlap_param_gather_with_optimizer_step` 则把 gather 进一步延后到和优化器更新重叠（`distributed_data_parallel.py:578`）。

---

## 4. pipeline send/recv 与计算重叠

PP 侧的 overlap 分两个层次：

**① `overlap_p2p_comm`：send 后不立即 wait。** `p2p_communication.py` 的两个核心函数在 `overlap_p2p_comm` 时把 `wait_on_reqs` 置 False（`p2p_communication.py:631` / `:657`），把当前 stage 的下游 send 与前向上游 recv 发出的通信句柄暂存，延后统一 wait。schedules 里 `config.overlap_p2p_comm` 决定传输路径（`schedules.py:1738`、`1767`、`1775` 处按此参数选 recv/send 的 overlap 版本）。注意它和 `batch_p2p_comm` 互斥（`schedules.py:1180-1181`）。

**② `deallocate_output_tensor`：送出去就伪释放激活。** 一个 stage 把激活 send 给下游后，这个输出张量只剩 `.grad_fn` 有用了（反向还要走 graph），`.data` 可以扔。`deallocate_output_tensor`（`schedules.py:166-196`）把 `out.data` 换成 `torch.empty((1,))`——释放几乎整段激活显存，只留图结构。配套的 `custom_backward`（`schedules.py:199-210`）直接调 C++ autograd 引擎，绕开 PyTorch `backward` 里「input shape == grad shape」的检查（因为 input 已被伪标量化了）。这是「用显存换时间」之外的**「用图结构换显存」**，间接让前向能塞更多 microbatch、radically 摊薄 bubble。

---

## 5. MoE 专家并行的通信重叠

MoE 的 expert all-to-all 是另一处重通信。`overlap_moe_expert_parallel_comm` 把 EP 的 token dispatch all-to-all 和专家计算重叠，但它约束极多：仅支持 alltoall/flex dispatcher、仅 EP（`transformer_config.py:3536-3553`）、需 bf16/fp16、禁 full recompute、Hopper 及更早的卡上不建议和 TP/CP 混用（`arguments.py:1638-1643`）。schedules 侧有专用 `get_overlap_moe_expert_parallel_comm_order`（`schedules.py:994`）来安排这种重叠下的层序。它本质是前一节的「延迟 dispatch」思想在 MoE 域的复刻——属进阶优化，此处点到为止。

---

## 6. 小结

- **通算并行 = 把同步等通信改成「惰性 dispatch + 双 stream + 反向 hook 触发」**，把三类通信藏进计算里。
- **grad reduce 重叠**：反向 hook（`distributed_data_parallel.py:424`）→ `register_grad_ready` 判桶满（`param_and_grad_buffer.py:913`）→ 及时 `start_grad_sync` 异步 reduce，`wait` 延后（`param_and_grad_buffer.py:833`）。
- **param gather 重叠**：forward pre-hook 按需等待（`distributed_data_parallel.py:468`）+ 反向链式提前 dispatch 下一个 bucket（`param_and_grad_buffer.py:611`），分布式优化器走 `_coalescing_manager` 合并集合通信。
- **PP 重叠**：`overlap_p2p_comm` 延迟 wait（`p2p_communication.py:631,657`）；`deallocate_output_tensor` 伪释放已发送的激活（`schedules.py:166-196`），`custom_backward` 绕 shape 检查（`schedules.py:199-210`）。
- **MoE 重叠**：`overlap_moe_expert_parallel_comm` 复刻同思想，但约束审查严格（`transformer_config.py:3536-3553`）。

行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。到此，并行主线的「原理 → 拓扑 → CP → PP 调度 → 组装地图 → 通算并行」六环闭合。
