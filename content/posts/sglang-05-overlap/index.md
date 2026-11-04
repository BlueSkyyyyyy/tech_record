---
title: "SGLang 研读（五）两级 overlap：批间流水、批内 TBO 与 CUDA Graph"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 6
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "Overlap", "CUDA Graph", "流水线"]
categories: ["推理框架"]
---

上一篇把调度器拆到了「一个 batch 怎么凑出来」。但 SGLang 的调度器从不「凑完就等 GPU」：它凑第 N+1 步时，第 N 步的 forward 还在卡里跑；而第 N 步自己的 forward 内部又被切成两半，一半在等 all-to-all，另一半在算 GEMM。

同一时刻有三条时间轴：**CPU** 填第 N+1 步的 `seq_lens`，**GPU** 跑第 N 步的 61 层 decoder，**采样**卡在最后——它要等第 N-1 步的 token 落地才能建 grammar bitmask。三条轴各被不同的东西挡住：CPU 被 Python 与 H2D 挡，GPU 被权重带宽挡，采样被**上一步的结果**挡。

1. `event_loop_overlap` 改了哪几行？为什么非得有 `result_queue` 和 `last_batch`？`_apply_war_barrier` 防的是哪个 race？
2. TBO 怎么把一条 batch 切成两个 micro-batch，依赖方向怎么表达，为什么能省时间？
3. CUDA Graph 捕获哪些 batch size，decode 与 prefill 为什么是两套网格？三级流水各自被谁挡住？

## 1. 两个循环，六处差别

`event_loop_normal` 在 `managers/scheduler.py:2107`，`event_loop_overlap` 在 :2143，相隔 36 行。默认走后者：`disable_overlap_schedule` 默认 `False`（`arg_groups/fields/schedule.py:193`），只有 MLX 与部分 Mamba `no_buffer` 路径会关（`arg_groups/overrides.py:477`）。

| | `event_loop_normal`（:2107） | `event_loop_overlap`（:2143） |
|---|---|---|
| 结果处理 | `run_batch` 后立刻 process（:2131） | 压进 `result_queue`（:2193），下一步 pop（:2201） |
| 存什么 | `batch` 本身 | `batch.copy()` 的快照（:2193） |
| 采样 | forward 返回前采完 | 迭代末尾才采（:2209） |
| 流同步 | 无 | 每次 launch 后 `_apply_war_barrier`（:2192） |
| 结束判断 | 结果已落地 | 长度算术预判（:4755） |
| 队列深度 | 0 | 恒 ≤ 2 |

```python
# managers/scheduler.py:2166  删掉 pause / 内存检查后的骨架
process_last_batch_early = self.should_process_last_batch_early(self.last_batch)
if process_last_batch_early:
    pop_and_process()
plan = self.get_next_batch_to_run(          # :2181 CPU 在这里跑，GPU 在跑第 N 步
    running_batch=self.running_batch, last_batch=self.last_batch)
batch = plan.batch_to_run
if batch:
    batch_result = self.run_batch(batch)    # :2190 只入队，不等 GPU
    self._apply_war_barrier()               # :2192
    self.result_queue.append((batch.copy(), batch_result))
if self.last_batch and not process_last_batch_early:
    pop_and_process()                       # :2201 处理第 N-1 步
if self.is_generation:
    self.launch_batch_sample_if_needed(batch_result, batch)   # :2209
self.last_batch = batch
```

`normal` 里对应只有三句：`run_batch` → `process_batch_result` → `last_batch = batch`。所以 **overlap 的全部魔法是「把结果处理挪后一格」**：挪完之后，`get_next_batch_to_run` 那一大坨纯 CPU 工作正好落进 GPU 跑 forward 的时间窗。这一挪欠下三笔语义账（第 2 节）和一道栅栏（第 3 节）。

`batch.copy()`（`schedule_batch.py:3428`）不是洁癖：它的注释写着「只包含 `process_batch_result` 会用到的字段」。因为下一轮 `merge_batch` 会原地改这个对象，队列里的快照必须保持第 N 步的样子。

## 2. 延后一格欠下的三笔账

**prefill 请求提前进 running。** prefill 结果没处理，不代表这些请求不能进 decode。`get_next_batch_to_run` 一进门就把上一批 prefill 合进 running（`scheduler.py:4131`）：

```python
# scheduler.py:4131
if last_batch and last_batch.forward_mode.is_extend():
    last_batch.filter_batch(reqs_to_exclude=list(reqs_to_exclude))
    running_batch = last_batch if running_batch.is_empty() else running_batch.merge_batch(last_batch)
```

这就是 `last_batch` 必须当参数传进来的原因。代码把时序写在注释里（:4163）：「overlap 会在 prefill 的结果 finish 掉它们之前就把请求合进 running」。合进去的请求带着旧的 `output_ids`（prefill 的采样还在队列里），所以紧接着的 `update_running_batch` 必须容忍「KV 已分配、输出未落地」。

**结束判断只能靠算术。** 第 N 步结果未处理，调度器却要决定第 N+1 步还带不带它。多带一步，用户就多收一个 token——单请求压测里几乎复现不出来。SGLang 用纯算术预判（`scheduler.py:4755`）：`inflight_steps > 0` 且 `next_output_finishes_by_length()`，后者就是 `len(output_ids) + 1 >= max_new_tokens`（`schedule_batch.py:1087`）。`inflight_steps` 是每个 `Req` 上的计数器（:663），launch/process 时加减（:2144、:2149）。`enable_skip_finishing_decode = True` 是 `event_loop_overlap` 的第一句（`scheduler.py:2145`）。注意这个判断假定「在飞那一步恰好产出一个 token」，只对非投机解码成立。

**token 不走 CPU。** `normal` 里第 N 步的 token 要 D2H 回来再 H2D 上去；`overlap` 里等不起。SGLang 把 relay 整个放进显存：`FutureMap`（`managers/overlap_utils.py:284`）是一组按 `req_pool_idx` 索引的 buffer，forward 用 `publish`（:652）/`stash`（:683）写，下轮用 `resolve_forward_inputs`（:113）gather 出来当 `input_ids`。于是 `run_batch` 末尾有这一句（`scheduler.py:5055`）：`batch.input_ids = None`——此刻它是**占位符**，真正的 token 还在 `output_tokens_buf` 里。只有 `seq_lens`（长度，不是身份）要回 CPU，`resolve_seq_lens_cpu`（:590）用私有 D2H 流加静态 pinned buffer 去拉，`wait_event(publish_ready)` 而不同步整条 schedule 流。**vLLM 让调度器拿 `-1` 当占位 token 往下算，SGLang 干脆不让 token 进调度器。**

**采样为什么排最后。** 注释写得很直白（`scheduler.py:2207`）：「它依赖上一批的结果（比如 grammar）」。具体是 grammar 的 FSM：第 N 步采样要读第 N-1 步接受 token 推进后的 bitmask，而推进发生在 `process_batch_result` 里。但**默认只有 grammar 才延迟采样**（`managers/tp_worker.py:736`）：`self.enable_overlap and not self.enable_spec and (forward_batch.sampling_info.grammars is not None or envs.SGLANG_ENABLE_DELAY_SAMPLE...)`。普通请求的 sample 紧跟 forward 发在同一条流上，CPU 不阻塞——`launch_batch_sample_if_needed`（:5358）第一句就是 `if batch_result.delay_sample_func is None: return`。投机解码反过来了：bitmask 是 verify 的**输入**，所以 SGLang 把 FSM 推进塞进 worker 中间（`scheduler.py:5001` 传 `grammar_barrier`，落到 `_advance_pending_grammar` :2238）。

## 3. `_apply_war_barrier` 防哪个 race

两个流：`schedule_stream`（调度器所在，:1217）与 `forward_stream`（模型执行）。RAW 方向有显式等待——进 forward 前 `self.forward_stream.wait_stream(self.schedule_stream)`（:4972）。WAR 方向没有任何天然的序：第 N+1 步要往同一批张量写新值，第 N 步可能还在读；更狠的是 `process_batch_result` 会**释放 KV slot**，释放出来的可能立刻被新请求拿走，一边读一边复用就串了。

```python
# scheduler.py:2089
runner = self.model_worker.last_shared_read_runner
ev = runner.shared_read_done_event
runner.shared_read_done_event = None
if ev is not None and not envs.SGLANG_FORCE_COARSE_WAR_BARRIER.get():
    self.schedule_stream.wait_event(ev)                      # :2098 细粒度
else:
    self.schedule_stream.wait_stream(self.forward_stream)    # :2100 粗粒度
```

两档决定 overlap 的上限。粗粒度等整个 forward，overlap 等于归零。细粒度靠 backend 申报「共享读到哪结束」：`SharedReadEnds`（`layers/attention/base_attn_backend.py:260`）里 decode/verify 默认 `IN_REPLAY`，意思是「图一开始把 metadata 读进常量内存之后就不再碰调度器的张量」。decode CUDA Graph 会把一个 external event 嵌进图里当标记（`runner/decode_cuda_graph_runner.py:718`），replay 到那儿就 record，调度器等到它就能安全写。顺带一个坑：`is_war_barrier_enabled()`（`managers/utils.py:67`）是 `is_cuda() or env`，而 `is_cuda()` 不含 ROCm——**HIP 上默认没有这道栅栏**。

## 4. 账本一：批间 overlap 能藏多少

把一步拆成 CPU 调度 $S_{\text{cpu}}$、GPU 前向 $T_{\text{gpu}}$、采样 $S_{\text{smp}}$：

$$
T_{\text{step}}^{\text{normal}} = S_{\text{cpu}} + T_{\text{gpu}} + S_{\text{smp}}
\qquad
T_{\text{step}}^{\text{ovl}} \approx \max\left(S_{\text{cpu}} + S_{\text{smp}},\ T_{\text{gpu}}\right)
$$

收益上限就是 $S_{\text{cpu}} + S_{\text{smp}}$，即**吞吐提升 ∝ CPU/GPU 时间比**：模型越小、batch 越小越划算。反过来说，**overlap 在大 batch 高端几乎不值钱**——分母是 $T_{\text{gpu}}$，而 $S_{\text{cpu}}$ 增长远慢于它。它值钱的地方是小 batch 低延迟，而它偏偏被设成默认，说明默认配置服务的是最坏情况。

| 场景 | $S_{\text{cpu}}$ | $T_{\text{gpu}}$ | normal | overlap | 吞吐 |
|---|---|---|---|---|---|
| 7B / 8 卡 / bs=8 | 1.2 ms | 2.8 ms | 4.0 ms | 2.8 ms | +43% |
| 70B / 8 卡 / bs=64 | 1.5 ms | 9 ms | 10.5 ms | 9 ms | +17% |
| 70B / 8 卡 / bs=256 | 2.5 ms | 28 ms | 30.5 ms | 28 ms | +9% |
| 671B / EP16 / bs=128 | 3 ms | 45 ms | 48 ms | 45 ms | +7% |

> 假设说明：$S_{\text{cpu}}$ 是「ingest + 组 batch + 填张量 + 提交 H2D」的量级估计（Python 逐请求循环，几百请求到毫秒级），$T_{\text{gpu}}$ 按权重带宽估（70B bf16 ≈ 140 GB，8 卡合计约 24 TB/s，理论下界 5.8 ms，实际 2~3 倍）。只说明趋势与量级，不是实测。

## 5. 批内 overlap：TBO 把一条 batch 切成两半

批间藏 CPU，批内藏**通信**。开关在 `arg_groups/fields/exec_.py`：`enable_two_batch_overlap` 默认 `False`（:1087），`tbo_token_distribution_threshold` 默认 `0.48`（:1094）。文档口径是「unlock up to 2x throughput」（`docs/docs/advanced_features/expert_parallelism.mdx:237`）。

**三道门。** ① 命令行：没有 a2a backend 时还必须开 attention DP，否则报错（`arg_groups/validation_hook.py:763`）；DeepEP v2、DWDP、unified-memory、UNO tree 各有互斥断言。② 模型：`OperationsStrategy.init_new_tbo`（`batch_overlap/operations_strategy.py:34`）按层类名分派，只认 DeepSeek-V2 / Qwen3-MoE / MiMoV2 / DSV4，其余抛 `NotImplementedError`；且**只对 sparse MoE 层生效**（`assert layer.is_layer_sparse`，:94），DeepSeek 前几层 dense 走普通路径（`models/deepseek_v2.py:3271` 用 `first_k_dense_replace` 切边界）。③ DP 一致性：所有 rank 必须同时同意，`compute_output`（`two_batch_overlap.py:452`）对 `local_can_run_tbo` 取 `min` 并要求 `forward_mode` 一致，否则整步退回。

**在哪切。** 切分点是**序列下标**不是 token 下标（`two_batch_overlap.py:80`）。prefill 默认按「两边 token 数最接近」找序列边界（`_split_array_by_balanced_sum`，:154）。但长度天然不均：一个 8K 请求加七个 100 token 的，按边界切就是 8000/700，等于没切。所以有 fallback——一边占比不到 48%（或超 52%）时改成一个序列**从中间劈开**（`_is_two_chunk_split_enabled` 的 `want_two_chunk`，:107；切点 `sum(extend_seq_lens) // 2`，:288）。decode 最简单：请求数的一半（:91 的 `(num_tokens // token_num_per_seq) // 2`）。**推论**：TBO 下 decode capture bs 必须是偶数，所以 `get_cuda_graph_batch_size_alignment` 在 TBO 时 `alignment *= 2`（`utils/common.py:4219`）。

**依赖方向：一张 stage 列表。** TBO 不建图，它建一条线。每层 forward 拆成一串 `op_*`，中间插 `YieldOperation()` 当分隔符：

```python
# operations_strategy.py:136  DeepSeek decode
[op_comm_prepare_attn, attn.op_prepare, YIELD,
 attn.op_core, op_comm_prepare_mlp, mlp.op_gate, mlp.op_select_experts, YIELD,
 mlp.op_dispatch_a, mlp.op_shared_experts, YIELD,
 mlp.op_dispatch_b, mlp.op_experts, mlp.op_combine_a, YIELD,
 mlp.op_combine_b, YIELD,
 mlp.op_output, op_comm_postprocess_layer]
```

`_convert_operations_to_stages`（`operations.py:228`）就按它把列表切段。**stage 内顺序执行，stage 之间可以插入另一个 micro-batch**——依赖方向全靠「作者知道哪两个算子之间能松手」，没有显式 DAG。粒度按 shape 调：prefill 每层 2 个 yield → 3 个 stage（`operations_strategy.py:123`、:127），decode 5 个 yield → 6 个 stage。decode 每层更短，切得更细才藏得住 a2a 的固定延迟。

真正让切分有意义的是 DeepEP 的 issue/wait 分离：`dispatch_a` 只发（`previous_event = Buffer.capture()`，`token_dispatcher/deepep.py:574`），`dispatch_b` 才等（`event.current_stream_wait()`，:613）。中间那个 stage 就是另一个 micro-batch 的 GEMM 能挤进去的窗口。两个 micro-batch 各有一套 dispatcher——`MaybeTboDeepEPDispatcher.__init__` 直接 `num_inner_dispatchers = 2 if is_tbo_enabled() else 1`（`two_batch_overlap.py:1160`），因为 DeepEP 的 `handle` 是 per-instance 状态，共用会串。

**错位执行。** 交错在 `execute_overlapped_operations`（`operations.py:38`）：

```python
for _ in range(delta_stage): executor_a.next()                      # A 先跑 delta 个 stage
for _ in range(a.num_stages - delta_stage): a.next(); b.next()      # 之后一比一交替
for _ in range(delta_stage): executor_b.next()                      # B 收尾
```

`delta_stage` 来自策略的 `tbo_delta_stages`：prefill `0`（`operations_strategy.py:114`，齐步走），decode `2`（:139，A 领先两个 stage）。为什么 decode 要领先？`dispatch_a` 与 `dispatch_b` 只有一 stage 之隔，完全同步时 A 的「等」紧跟自己的「发」，窗口里没活；领先 2 个 stage，A 等的时候 B 正好在算 attention 前的准备。这是**软件流水**：`delta_stage` 是流水深度，填满/排空各损失 `delta_stage` 个 stage。DeepSeek-V3 有 61 层 × 6 stage = 366 个 stage，这点损失可忽略。

上下文也要劈开：hidden states 按 token 段切并 pad 到 `tbo_padded_len`（`two_batch_overlap.py:1080` 的 `_model_forward_filter_inputs`），跑完 merge（:1114）；attention 后端也跟着切，`TboAttnBackend`（`layers/attention/tbo_backend.py:18`）一个 `primary` 加**两个 child**，每个 child 有自己的 metadata，`_StageExecutor` 在每个 stage 前把 `ForwardContext` 换成对应 child 的（`operations.py:161` 的 `context(child_ctx)`）。少换一次，两个 micro-batch 就会拿错的 `seq_lens` 去读 KV。

## 6. 账本二：TBO 的收益与「SM 税」

设每层计算 $C$、通信 $K$，$r = K/C$。不重叠 $C+K$，完美重叠 $\max(C,K)$：

$$
S(r) = \frac{1 + r}{\max(1,\ r)}
$$

$r=1$ 时 $S=2$，这就是「2x」的来源；$r=0.2$ 时只剩 1.2x。但重叠不免费：a2a kernel 也占 SM。通信拿走 $n$ 个后计算只能用 $N-n$ 个，计算时间按 $N/(N-n)$ 膨胀，净收益为正（$r<1$）要求：

$$
r > \frac{n}{N-n}
\qquad\text{with}\qquad T_{\text{tbo}} = \max\left(C \cdot \frac{N}{N-n},\ K\right)
$$

H100 上 $N=132$，vLLM 的 DBO 默认 $n=20$（`vllm/envs.py:308` 的 `VLLM_DBO_COMM_SMS`），阈值 $20/112 \approx 0.18$：

| $r = K/C$ | 无重叠 | 有重叠（含税） | 加速 |
|---|---|---|---|
| 0.1 | 1.10 C | 1.18 C | 0.93x（**亏**） |
| 0.3 | 1.30 C | 1.18 C | 1.10x |
| 0.5 | 1.50 C | 1.18 C | 1.27x |
| 1.0 | 2.00 C | 1.18 C | 1.70x |
| 2.0 | 3.00 C | 2.00 C | 1.50x |

三个边界一目了然：通信太少没得藏（$r<0.18$ 被税吃掉）、通信太多藏不住（$r>1$ 瓶颈就是通信）、甜点在 $r\approx1$。SGLang 自己对这笔税的处理很能说明问题：prefill 策略**显式切 SM**（`operations_strategy.py:110` 的 `deep_gemm_num_sms = total - DeepEPConfig.get_instance().num_sms`），因为 prefill 计算密集、少给 SM 真掉吞吐；decode 策略传 `deep_gemm_num_sms=None`（:138）不切——decode 的 GEMM 是权重带宽受限，SM 少几个不掉多少，税接近 0，于是 $S$ 能真的往 2 靠。

## 7. CUDA Graph：每 phase 一个 backend、一套网格

早年的 `--cuda-graph-max-bs` 已拆成 `cuda_graph_max_bs_decode` / `cuda_graph_max_bs_prefill`（`fields/exec_.py:655`、:658），收进 `CudaGraphConfig`——一个 phase 一个 `PhaseConfig`（`model_executor/cuda_graph_config.py:146`），backend 只有三个值：`full` / `breakable` / `disabled`（:48，`tc_piecewise` 已移除）。

| phase | 默认 backend | 含义 |
|---|---|---|
| decode | `full`（:150） | 整个 forward 一张图，按 bs 桶 replay |
| prefill | `breakable`（:123，仅 CUDA） | 分段捕获，attention metadata 与 LM head 在图外 |

这个不对称是刻意的：decode 每步 shape 简单（每请求 1 token），整图最划算；prefill 的 attention metadata 依赖真实长度，整图要多捕一堆拓扑，默认给分段。

**两套网格。** decode 的桶是**请求数**（`arg_groups/cuda_graph_hook.py:545`）：`[1,2,4,8,12] + range(16,257,8) + range(272,512,16) + range(512,max_bs+1,32)`。prefill 的桶是**聚合 token 数**（:519），4 起步、4096 之后步长才 256。同一个字段 `bs` 在两个 phase 含义不同，`ALLOWED_KEYS_PER_PHASE` 的注释专门写了这件事（`cuda_graph_config.py:70`）。`max_bs` 按显存档位自动定（`arg_groups/memory_hook.py:92` 起的分档）：<20 GB → 8，<35 GB → 48（tp≥4 则 160），<90 GB → 256（tp≥4 则 512），≥160 GB → 512；prefill 的取 `chunked_prefill_size`，MLA 封顶 8192（:235）。数下来（tp=1）：4090 档 decode 4 桶、prefill 42 桶；H100 档 decode 36 桶、prefill 58 桶；B200 档 decode 52 桶。

**padding 与回退。** replay 时把真实 bs 抬到最近一档（`runner/base_cuda_graph_runner.py:180` 的 `_pad_to_bucket`，就是 `bisect_left`）；真实 bs 超过最大桶直接回退 eager（`decode_cuda_graph_runner.py:925` 的 `cuda_graph_bs <= self.max_bs`）——不报错，只是慢，最容易在压测里被忽略。

**和 TBO 的四条硬耦合。** ① 只捕 TBO=true 的图：`TboCudaGraphRunnerPlugin.capture_one_batch_size`（`two_batch_overlap.py:344`）算完 `tbo_split_seq_index` 就 `assert ... is not None`，注释写「为简单起见」。② replay 时切分点从 bs 反推（`compute_split_indices_for_cuda_graph_replay`，:309 用 `bs * token_num_per_seq`），所以切分规则必须只依赖 bs。③ TBO 会关掉 prefill 图：`cuda_graph_hook.py:220` 的规则表里 `("two-batch overlap", lambda: cfg.enable_two_batch_overlap)` 命中即把 prefill backend 设 `disabled`。④ capture bs 必须偶数（第 5 节）。

**显存账。** `reserve_for_graph_mb`（`memory_hook.py:359`）把图的预留算进 `mem_fraction_static`：decode 是 `max_bs * 2` MB（:370），prefill 是 `len(bs) * 8` MB（:388）。H100 档就是 256×2 + 58×8 ≈ 976 MB，直接从 KV 池里扣（`mem_fraction_static = (gpu_mem - reserved_mem)/gpu_mem`，:326）。**所以调大 `cuda_graph_max_bs_decode` 的代价不是峰值显存，而是并发上限**；反过来桶太稀会让 replay 频繁抬到高一档，白算 padding 的 token。

## 8. 与 vLLM 对照

- **异步的载体不同**。vLLM 让调度器拿占位 token 往下排：`AsyncScheduler._update_after_schedule` 给每个 decode 请求 `num_output_placeholders += ...`，并 `request.spec_token_ids = [-1] * n`（`vllm/v1/core/sched/async_scheduler.py:36`、:35）。SGLang 不让 token 进调度器，用 `FutureMap` 在显存里 relay。**一个是「乐观假设 + 事后打补丁」，一个是「延后一格 + 显存中转」。**
- **在飞步数都是 2**。vLLM `max_concurrent_batches` 在 async scheduling 下返回 2（`vllm/config/vllm.py:605-616`），`batch_queue` 的 `maxlen` 就是它；SGLang 的 `result_queue` 深度恒 ≤ 2。同一个数字，一个实现成 future 队列，一个实现成 deque。
- **structured output 撞同一堵墙**。vLLM 在 engine core 显式分支：「We need to defer sampling until we have processed the model output from the prior step」（`vllm/v1/engine/core.py:799`）；SGLang 是 `delay_sample_func` + 迭代末尾补采。**同一条约束，两家都把「采样」而不是「调度」推后。**
- **max_tokens 边界：语义强弱正好相反**。vLLM 判断 `num_computed_tokens + 2 - num_output_placeholders >= num_prompt_tokens + max_tokens` 就跳过（`vllm/v1/core/sched/scheduler.py:799-808`），是**保守充分条件**（假设 draft 全被拒也够）；SGLang 的 `next_output_finishes_by_length` 是**精确条件**，代价是只在非投机路径成立。
- **批内重叠：DBO ≈ TBO，形态不同**。vLLM 开两个 CPU 线程，各带 compute/comm 两条流，`threading.Barrier` + event 打乒乓，`dbo_yield()` 是对称会合点（`vllm/v1/worker/ubatching.py:182`、`docs/design/dbo.md:9`、:76）；SGLang 是单线程 + stage 列表 + `delta_stage` 错位，`YieldOperation` 是单向分隔符。线程模型决定约束：DBO 只支持整图（「once a DBO CUDA graph has been captured, it can be replayed without any multithreading」，`docs/design/dbo.md:62`），TBO 反而关掉 prefill 图。
- **适用面与档位**。vLLM DBO「currently only targets DP+EP deployments」（`docs/design/dbo.md:5`），要 `--enable-dbo`；SGLang TBO 允许无 a2a 的 DP TP-MoE 路径（`validation_hook.py:763`），但模型只认四个家族。CUDA Graph 一侧：vLLM `CUDAGraphMode` 五档 `NONE`/`PIECEWISE`/`FULL`/`FULL_DECODE_ONLY`/`FULL_AND_PIECEWISE`（`vllm/config/compilation.py:53-63`），是「全局模式 + 两段拆分」；SGLang 是「每 phase 独立选」，没有 piecewise 中间档，但多一个 `breakable`。

**谁把 CPU 藏得更彻底？** 机制上 vLLM 更狠：占位 token 让它真能排到第 N+1 步而不需要上一步的任何值，SGLang 至少还要把 `seq_lens` 拉回 CPU。代价是语义：占位 token 意味着「未确认的 token」会流进长度记账和 block 分配，错一次就是静默多算或少算；SGLang 把不确定性关在一个 deque 里，代价是算术预判结束（2.2 节）和一道 WAR 栅栏。**一个用语义复杂度换 overlap 深度，一个用 overlap 深度换语义简单。**

## 9. 常见坑

- **`batch.input_ids` 在 overlap 下是占位符**（`scheduler.py:5055` 直接置 `None`）。任何在 `process_batch_result` 里读它的代码都会拿到空值或上一步的残值。
- **别靠「结果已落地」判断结束**，要读 `inflight_steps`（`schedule_batch.py:663`）。否则用户多收一个 token，而且只在特定长度上出现。
- **`_apply_war_barrier` 的细粒度路径依赖 backend 申报**。`SharedReadEnds` 默认 `UNKNOWN`（`base_attn_backend.py:265`），落到粗粒度 `wait_stream(forward_stream)` 后 overlap 直接归零；自接后端忘了重写 `shared_read_ends`，性能会莫名掉。
- **TBO 下 capture bs 必须偶数**，切分点又由 bs 反推（`two_batch_overlap.py:309`）；手工给奇数 `--cuda-graph-bs-decode` 会在 replay 时切错位。另外 `tbo_token_distribution_threshold=0` 会关掉 two-chunk（字段注释原话），一个 8K 请求就能把 TBO 变回串行。

## 10. 读代码时的三个检查点

**检查点一：这个 `process_batch_result` 处理的是第几步？** `normal` 处理当前步，`overlap` 处理上一步。看到 `result_queue.popleft()` 就要问：此刻 `running_batch` 里有没有「结果还没落地」的请求？有的话，任何按 `output_ids` 长度的决策都得换成 `inflight_steps` 口径。

**检查点二：这道栅栏等的是「整个 forward」还是「metadata 读完」？** 差别是一个数量级。`shared_read_done_event` 为 `None` 就走粗粒度（`scheduler.py:2100`），而它是否被设取决于 backend 的 `SharedReadEnds` 申报。

**检查点三：这个 batch size 是请求数还是 token 数？** decode 的桶是请求数，prefill 的桶是聚合 token 数，TBO 的切分又是请求数一半（decode）或序列边界（prefill）。同一个 `bs` 三种含义，混用不崩，只在某个桶边界上慢一截或读错 KV。

## 小结

- 批间 overlap 的全部内容是「把结果处理挪后一格」：`result_queue` + `last_batch` + 延迟采样，队列深度恒 ≤ 2；这一挪欠三笔账——prefill 提前合进 running（:4131）、结束改成算术预判（:4755）、token 改走显存 relay（`FutureMap`）。
- `_apply_war_barrier` 防的是「第 N+1 步写共享张量 / 释放 KV slot」与「第 N 步还在读」的 WAR；等整段 forward 会让 overlap 归零，所以 backend 要申报 `SharedReadEnds`。
- 批间收益上限是 CPU 时间，**大 batch 高端不值钱，小 batch 低延迟才值钱**。
- TBO 用 `YieldOperation` 把每层切成 3 个（prefill）或 6 个（decode）stage，两个 micro-batch 按 `delta_stage`（0 / 2）错位；切分按序列边界，长度不均时把一个序列对半劈（阈值 0.48）。
- TBO 收益是 $S=(1+r)/\max(1,r)$，但通信占 SM，净收益为正要求 $r > n/(N-n)$（H100 上约 0.18）；prefill 显式切 SM，decode 不切。
- CUDA Graph 是「每 phase 一个 backend、一套网格」：decode 默认整图按请求数分桶，prefill 默认分段按聚合 token 分桶；显存预留约 `max_bs × 2 MB + 桶数 × 8 MB`，直接从 KV 池扣。

配套脚本 [`code/sglang/overlap_sim.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang)：serial / overlap / TBO 三种模式的时间线，画出每步谁在等谁，附 `comm_frac=0` 时 TBO 反而更慢的对照（`--test` 自测）。

下一篇看 attention 后端本身，以及 page table 怎么把逻辑位置翻成物理槽：[Attention 后端与 page table]({{< relref "sglang-06-attention-backends" >}})。调度器怎么凑 batch 见[调度器那篇]({{< relref "sglang-04-scheduler" >}})，KV 池的账本见[显存那篇]({{< relref "sglang-03-memory-pool" >}})。
