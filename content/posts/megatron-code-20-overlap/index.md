---
title: "Megatron 源码精读（二十）：通信与计算 overlap"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "compute-comm-overlap", "ddp", "overlap", "流水线并行"]
categories: ["训练框架"]
weight: 20
series: ["megatron-code"]
---

上一篇[《并行组装地图》]({{< relref "megatron-code-19-parallel-assembly" >}})讲清了「谁跟谁通信」。但通信要花时间：param all-gather、grad reduce、pipeline 的 send/recv，只要同步等，算力就空转。本篇讲这些通信怎么藏进计算里。

一句话结论：**所有 overlap 都是同一个套路——把「发起通信」和「等待完成」拆开，用 hook 或事件把 wait 推到最晚一刻；DP 侧靠 bucket 化 + 反向 hook，PP 侧只对 interleaved schedule 开放，MoE 是同一套路在 all-to-all 上的复刻。**

判断一处通信有没有被重叠，只看一件事：关键路径上是否还有一个同步 `wait`。

## 1. 问题：三类通信，三个暴露点

**前向的 param all-gather**：分布式优化器下每 rank 只存参数的 1/dp 分片，前向前必须把整桶参数 gather 回来。通信在前、计算在后，不重叠就是每个 microbatch 前硬等一次。

**反向的 grad reduce**：所有 microbatch 的梯度算完才 all-reduce/reduce-scatter。计算在前、通信在后，不重叠就是反向尾部空等。

**PP 的 send/recv**：stage 间传激活和梯度，本身与计算交错，但默认每个通信句柄发完就 `wait`。

三类暴露点的共同解法是惰性 dispatch：通信尽早发、wait 尽量晚，等的时机由「谁真的要用这份数据」决定；需要时再配独立通信 stream。

## 2. grad reduce：bucket 就绪就发，wait 留到最后

### 2.1 为什么能边算边传

参数按 bucket 聚成连续 buffer（`param_and_grad_buffer.py` 的 `_ParamAndGradBucket`），反向又是从输出层往输入层逐层出梯度，所以 bucket 天然按「反向到达顺序」依次 ready。最后一个输出层的 bucket 梯度到手时，就可以先把它 reduce 掉，不必等输入层。

开关默认关闭：`overlap_grad_reduce: bool = False`（`distributed_data_parallel_config.py:18`）。

### 2.2 三件套：hook 登记、计数判定、dispatch/wait 分离

**① 反向 hook 把梯度交出来。** DDP 初始化时给每个参数的 `grad_acc` 注册 backward post hook（`distributed_data_parallel.py:424`）。hook 体（`distributed_data_parallel.py:500`）先把 `param.grad` 累进 `param.main_grad`、把 `param.grad` 置 None，再调 `register_grad_ready`；CUDA graph 捕获期间直接返回。

**② 用「每个参数的到达次数」判定整桶就绪。** `register_grad_ready`（`param_and_grad_buffer.py:913`）只在 `is_last_microbatch` 时记账：每个参数计数 +1，计数与 `golden_per_param_grad_ready_counts` 完全相等才发通信。golden 是第一个 batch 结束时录下的期望值（`param_and_grad_buffer.py:307`、`reset` 在 336-341 行）。

为什么是「计数字典相等」而不是「参数集合齐全」？因为同一个参数可能在一个 batch 里被前向图经过多次（权重共享、tied embedding），期望值本来就是次数而不是布尔。这点在源码注释里写明，也是把 bucket 判满讲成「字节数够了」时最容易错的地方。

**③ dispatch 与 wait 分离。** `start_grad_sync`（`param_and_grad_buffer.py:651`）在 overlap 模式下只发 `async_op=True` 的通信；真正的等待在 `finish_grad_sync`（`param_and_grad_buffer.py:833`）。后者用 `grad_reduce_finished` 做幂等（856 行）：后继 bucket group 会在 dispatch 时提前 drain 前驱（`previous_grad_reduce_bucket_group`，只在 `reduce_scatter_with_fp32_accumulation` 路径下串联），末尾的 finalize 循环还会再调一遍，不能重复 wait。

`delay_wgrad_compute`（`distributed_data_parallel_config.py:202`）把 wgrad 计算本身也往后挪，进一步靠近它的 reduce。它只在 `transformer_impl=transformer_engine` 下可用，且与 `overlap_grad_reduce` 同开要求 TE ≥ 2.8.0（`arguments.py:2099-2106`）。

## 3. param all-gather：前向用到才等

开关同样默认关闭：`overlap_param_gather: bool = False`（`distributed_data_parallel_config.py:21`）。

**① forward pre-hook 把等待推到最晚。** `enable_forward_pre_hook`（`distributed_data_parallel.py:436`）给每个子模块注册 pre-hook，hook 体（468 行）只对模块直接持有的参数调 `_finish_param_sync_for_bucket_group`：模块要前向了，才确保它这一桶参数 gather 完成。前面别的模块在算时，gather 一直在飞。

**② 链式预取让下一次 gather 提前发。** 构造时按 bucket 逆序串链（`distributed_data_parallel.py:342-348`，注释说 all-gather 按 bucket 逆序发生），当前桶 wait 完就顺手 `start_param_sync` 下一个桶（`param_and_grad_buffer.py:611`），于是前向用到下一个桶时它已经完成大半。

**③ 异步与合并。** `async_op = overlap_param_gather and not force_sync`（`param_and_grad_buffer.py:477`）；分布式优化器路径用 `_coalescing_manager` 把多个 `all_gather_into_tensor` 合并成一次 coalesced launch（583 行起）。

两个坑：`skip_next_bucket_dispatch=True`（`distributed_data_parallel.py:495-498`）在 `align_param_gather` 或「与 optimizer step 重叠」时生效，避免链式 dispatch 和显式调度重复发同一个桶；`overlap_param_gather_with_optimizer_step` 打开后，DDP 的 `start_param_sync` 在前反向里直接 return，改由优化器 step 发第一个 chunk 的 gather。

`align_param_gather` 的语义是「所有 PP stage 同时发起 gather」，它与 `overlap_param_gather` 同开时 schedule 会在主循环前用 `param_sync_func` 先同步前两个 chunk（`schedules.py:1320-1323`）。

## 4. PP：overlap_p2p_comm 只属于 interleaved

**这是本篇最容易写错的一处。** 基线里 `forward_backward_pipelining_without_interleaving` 开头就抛错：

```python
# schedules.py:2313-2316
if config.overlap_p2p_comm:
    raise ValueError(
        "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
    )
```

参数解析侧同样处理：vp 为空时直接把 `overlap_p2p_comm` 和 `align_param_gather` 置 False（`arguments.py:1064-1065`）。所以「朴素 1F1B 用 overlap_p2p_comm 重叠 send/recv」不成立，它只在 interleaved 下可用。

interleaved 下机制是「先拿句柄、后统一 wait」：`send_forward_recv_forward` 与 `send_backward_recv_backward` 在 `overlap_p2p_comm` 时传 `wait_on_reqs=False`（`p2p_communication.py:631`、`657`），返回 `(tensor, wait_handles)`，句柄留到 schedule 里合适的点再等。它与 `batch_p2p_comm` 互斥（`schedules.py:1180-1181`）；warmup/flush 阶段是否也重叠另由 `overlap_p2p_comm_warmup_flush` 控制。

**另一件 PP 专属优化是伪释放输出。** `deallocate_output_tensor`（`schedules.py:166-196`）在输出发给下游后，把 `out.data` 换成一个 `torch.empty((1,), device=..., dtype=...)`：图结构（`grad_fn`）保留，这份激活的存储释放。训练入口把它硬编码为 True（`arguments.py:2236`），没有对应的 CLI 开关，所以这条路径一直在生效。

配套的 `custom_backward`（`schedules.py:199`）直接调 `Variable._execution_engine.run_backward`：`torch.autograd.backward` 会检查 output 与 grad 形状一致，而此时的 output 已被伪标量化成 1 个元素。

## 5. MoE A2A overlap：同一套路，约束一长串

`overlap_moe_expert_parallel_comm` 把 EP 的 token dispatch all-to-all 与专家计算重叠，约束集中在 `transformer_config.py:3536` 起的校验块：

- 需 torch ≥ 2.6（更早版本有 hang）；EP > 1；dispatcher 只支持 `alltoall` / `flex`；bf16 或 fp16；不能开 `moe_shared_expert_overlap`；MTP 层数只能是 1。
- PP > 1 时必须显式给 vp——否则没有 interleaved 骨架可用。
- full recompute 允许，但走「按 layer segment 手写 replay」：不能开 `distribute_saved_activations`、dropout 必须为 0、FP8 不能用 delayed scaling；非 full 时 `recompute_method` / `recompute_num_layers` 必须为 None，且 `recompute_modules` 里不能有 moe。

调度侧由 `get_overlap_moe_expert_parallel_comm_order`（`schedules.py:994`）把每个 chunk 拆成单层节点，让相邻 chunk 的层能互相 overlap；warmup 也相应多一个 microbatch。Hopper 及更早的卡上不建议与 TP/CP 混用，因为 `CUDA_DEVICE_MAX_CONNECTIONS` 的要求冲突（`arguments.py:1638`，只是 warning）。

## 6. 小结

- 通算并行 = 惰性 dispatch + 事件等待：通信尽早发，wait 推到数据真正被需要时。
- grad reduce：反向 hook 交梯度 → 按参数计数判桶满 → `start_grad_sync` 异步发、`finish_grad_sync` 幂等等。
- param gather：forward pre-hook 按模块等待 + 按逆序链式预取 + `_coalescing_manager` 合并 launch。
- PP：`overlap_p2p_comm` 只在 interleaved 下可用，非 interleaved 直接报错；`deallocate_output_tensor` + `custom_backward` 用图结构换激活显存。
- MoE：同一范式复刻到 A2A，但 full recompute、dropout、FP8 recipe 都有限制。

---

**版本提示（2026-10）**：EP A2A overlap 的重算按 layer segment 组织（#6429 系列改动），基线里已有逐 layer segment 手写 replay 的注释；dev 里 MoE overlap 改走新增的 `combined_1f1b.py`（chunk 拆成节点、计算/通信双流共调度）。GTP 权重材质化用逐权重预取链 + ticket 缓存；新增 `grad_buffer_offload`，配合 NCCL UBR 把梯度缓冲放进独立内存池，可在训练阶段间释放重建（#6925）。「惰性 dispatch + 事件等待」的范式没变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
