---
title: "vLLM 研读（五）执行侧：InputBatch、attention metadata 与 CUDA Graph"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 6
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "ModelRunner", "CUDA Graph", "torch.compile"]
categories: ["推理框架"]
---

调度器交出来的东西薄得离谱：一个 `dict[str, int]`，写着每个请求这一步要跑几个 token。它不知道 KV 落在显存哪一块，不知道 attention kernel 该起多大 grid，也不关心上一步的 batch 长什么样。把这些变成 GPU 能直接吃的一张表，就是执行侧的全部工作。

先记住一个画面：GPU 上第 N 步还没算完，CPU 已经在给第 N+1 步填表——把 `token ids`、`positions`、`slot_mapping` 三行写进**地址永远不变**的 pinned buffer；然后拿一个 `BatchDescriptor` 当 key 去查表，命中就直接 `cudaGraphLaunch`，整层模型一条命令发完。

这一篇回答三个问题：

1. 一个 step 里 `SchedulerOutput` 怎么变成三件套？为什么 slot mapping 是 paged attention 的**接口**，而不是实现细节？
2. 为什么 batch 必须持久化、只能增量更新，代价是什么？
3. CUDA Graph 能盖住哪一段、盖不住哪一段？padding 和 capture 的账怎么算？

## 1. 从 SchedulerOutput 到三件套

`SchedulerOutput` 的字段就是调度器能说的全部（`vllm/v1/core/sched/output.py:232`）：`scheduled_new_reqs`、`scheduled_cached_reqs`（只发 diff）、`num_scheduled_tokens`（:244）和 `total_num_scheduled_tokens`（:247）。EngineCore 的一步只有两句：`scheduler.schedule()` 和非阻塞的 `execute_model(non_block=True)`（`vllm/v1/engine/core.py:719`、`:720`）。

执行侧收到它，先改**持久状态**，再算**这一步的张量**。`_update_states`（`vllm/v1/worker/gpu_model_runner.py:1176`）里全是这种小动作：把新的 `num_computed_tokens` 写回行（:1452）、把新分配的 block 追加到块表行尾（:1454）、把空出来的行用末尾请求填掉（`condense()`，:1496）、最后刷新采样元数据（:1500）。

### 1.1 三件套怎么算出来

`_prepare_inputs`（:1924）是纯 CPU 的 numpy/torch 流水线，四步：

```python
# vllm/v1/worker/gpu_model_runner.py:1948
req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
# :1957  positions = 已算 token 数 + 本请求内偏移
positions_np = (self.input_batch.num_computed_tokens_cpu[req_indices]
                + self.query_pos.np[: cu_num_tokens[-1]])
# :1971  二维 token 表的扁平下标
token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
```

`token_ids_cpu` 是 `[max_num_reqs, max_model_len]` 的二维表，所以「第 r 个请求第 p 个位置」的 token 落在 `r * max_model_len + p`。一次 `torch.index_select` 把所有要算的 token 抽成连续的 `input_ids.cpu`（:1979），再整段 H2D。

`positions` 和 `seq_lens` 则干脆在 GPU 上算，省掉一次全量拷贝：

```python
# :2139
self.positions[:total_num_scheduled_tokens] = (
    self.num_computed_tokens[req_indices_gpu].to(torch.int64)
    + self.query_pos.gpu[:total_num_scheduled_tokens])
# :2143
self.seq_lens[:num_reqs] = self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
```

`query_start_loc` 是这批 token 的「请求边界」，即 `[0, cu1, cu2, ...]`，并且要求单调不减——FlashAttention 这类 kernel 会读越界的尾巴，所以要把它填成最后一个值（:2034-:2037）。

### 1.2 slot_mapping：把逻辑位置翻译成物理槽

paged attention 的 KV cache 是 `[num_blocks, block_size, num_kv_heads, head_dim]` 的扁平池。一个 token 的 KV 该写到哪，只由两件事决定：它属于哪个请求（查块表哪一行）、它的逻辑位置是几（查行内第几个块）。

$$\text{slot}(t) = \text{block\_table}\big[r(t),\ \lfloor p(t)/B \rfloor\big]\cdot B + \big(p(t) \bmod B\big)$$

这不是 Python 算的，是一个 Triton kernel（`vllm/v1/worker/block_table.py:374`）：

```python
slot_ids = block_numbers * KV_CACHE_BLOCK_SIZE + local_block_offsets
```

padding 出来的空位要显式写 `-1`（`gpu_model_runner.py:3870`），因为 `reshape_and_cache`（`vllm/_custom_ops.py:2870`）不做边界判断，`-1` 是它约定的「丢弃」哨兵（`PAD_SLOT_ID = -1`，`vllm/v1/attention/backends/utils.py:55`）；块表里对应的空行则填 `NULL_BLOCK_ID = 0`（`gpu_model_runner.py:2309`），指向专门保留的第 0 块。**两个哨兵各管一半：slot 管写，block 管读。**

这就是为什么 slot mapping 是接口而不是细节。attention kernel 的入参里没有「请求」这个概念，只有 `block_table`、`seq_lens`、`query_start_loc`；写 KV 的 kernel 也只认 `slot_mapping`。前缀缓存命中、抢占换块、chunked prefill 切段，所有这些都只改块表和 slot mapping，kernel 一行不用动。

> **类比的边界**：块表像文件系统的 FAT，slot mapping 是这一批写请求的 inode 列表。区别是文件系统可以懒分配、写时复制，KV cache 不行——块必须在那次 forward 之前就分配好，且一个 slot 在一步内只能被写一次。

| 张量 | 形状 / dtype | 谁产生 | 谁消费 |
|---|---|---|---|
| `input_ids` | `[num_tokens]` int32 | `token_ids_cpu` 的 `index_select`（:1979） | embedding |
| `positions` | `[num_tokens]` int64 | GPU 上按 `num_computed_tokens` 加偏移（:2139） | RoPE |
| `slot_mapping` | `[num_tokens]` int64 | Triton kernel（`block_table.py:374`） | `reshape_and_cache` |
| `block_table` | `[num_reqs, ⌈L/B⌉]` int32 | 调度器的 block_ids → pinned 行（:1944） | attention kernel |
| `seq_lens` / `query_start_loc` | `[R]` / `[R+1]` int32 | :2143 / :2034 | attention kernel |

## 2. 为什么不能每步重建张量

答案藏在一次分配里（`vllm/v1/worker/gpu_input_batch.py:131`）：

```python
self.token_ids_cpu_tensor = torch.zeros(
    (max_num_reqs, max_model_len), device="cpu", dtype=torch.int32, pin_memory=False)
```

`max_num_reqs=1024`、`max_model_len=131072` 时，这一张表就是 1024 × 131072 × 4 B = **512 MiB**，旁边的 `is_token_ids` 还要 128 MiB。每步重扫一遍不可能，所以它天生是持久的：`InputBatch` 里所有 buffer 按 `max_num_reqs` 预分配，per-step 只写被调度到的那几行。

第二个理由是**地址必须稳定**。CUDA graph 重放时只按地址读输入，replay 路径甚至会断言地址没变（`vllm/compilation/cuda_graph.py:352`）。任何 `torch.cat`、任何重新分配，都会让上一次捕获的图失效。

于是「增量」成了唯一的选择，`condense()`（`gpu_input_batch.py:698`）是它最脏的一段：请求结束留下空洞，就把末尾的活跃行搬下来填，同时搬走采样参数、generator、块表行（:777 `block_table.move_row`）、spec token。搬完之后 `req_id → index` 全部重写，所以 `prev_req_id_to_index` 这种「上一步的行号」必须单独留一份，否则异步调度下根本对不上号。

采样侧也一样按需：全是 greedy 时 `temperature` 连拷都不拷（`gpu_input_batch.py:853` 的 `if not self.all_greedy`），惩罚项则看 `no_penalties` / `no_top_p` 这些标记。**「没有请求需要它」比「拷得快」更省时间。**

### 账本一：每步要搬多少字节

设 8B 模型、`block_size=16`、decode 一步 256 个请求（`num_tokens=256`），pinned H2D 有效带宽按 25 GB/s 估。块表每步整行提交（`block_table.py:181` 的 `copy_to_gpu(num_reqs)` 只切行、不切列），所以成本随上下文长度线性涨：

| `max_model_len` | 块表行宽 | 每步块表 H2D | 25 GB/s 下耗时 | 占比（假设 GPU decode 一步 20 ms） |
|---|---|---|---|---|
| 8K | 512 | 512 KiB | ≈ 21 μs | 0.1% |
| 32K | 2048 | 2 MiB | ≈ 84 μs | 0.4% |
| 128K | 8192 | 8 MiB | ≈ 336 μs | 1.7% |

同一 step 里 `input_ids` 只有 1 KiB、`positions` 2 KiB、`query_start_loc` 1 KiB——块表一项比其余全部还大两到三个数量级。这解释了 MRV2 为什么专门造 `StagedWriteTensor`：只把 diff 的行段打包上传，而不是整行重传（`vllm/v1/worker/gpu/model_runner.py:1838` 的 `apply_staged_writes()`）。

> 假设说明：20 ms 是 8B/bf16 在单卡上 256 并发 decode 的量级估计（权重复用一次、每步读约 16 GB），只用来做比例尺，不是实测值。模型越小、batch 越小，分母越小，上表的占比就越难看。

## 3. attention metadata

三件套是 batch 级输入，metadata 是**每层、每后端**的输入。跨层共享的部分被抽成 `CommonAttentionMetadata`（`vllm/v1/attention/backend.py:395`）：

| 字段 | 含义 | 备注 |
|---|---|---|
| `query_start_loc` / `query_start_loc_cpu` | `[R+1]`，每个请求 query 的起点 | GPU 版给 kernel，CPU 版给 Python 决策（:402） |
| `seq_lens` | `[R]`，每个请求算到哪 | decode 时就是 KV 长度（:406） |
| `block_table_tensor` / `slot_mapping` | 块表与槽位 | 每个 KV cache group 一份（:419） |
| `max_query_len` | 本批最长 query | **选 kernel routine 的关键**（:414） |
| `max_seq_len` | 最长上下文，可能是上界 | split-K 启发式用（:416） |
| `num_reqs` / `num_actual_tokens` | 真实请求数 / token 数 | 与 padded 值区分（:409、:412） |
| `is_prefilling` / `seq_lens_cpu_upper_bound` | 混合模型与异步投机用 | （:447、:452） |

Builder 是唯一的抽象点：`build()`（:699）把 common metadata 翻译成后端私有结构；`update_block_table()`（:719）在混合 KV cache 分组之间复用已建好的 metadata，只换块表；`build_for_cudagraph_capture()`（:733）允许后端在捕获时走一条特殊路径（比如把 `max_seq_len` 拉到 `max_model_len`，免得选中依赖真实长度的 kernel）。

三种 batch 的差别不在于形状，而在于**后端能不能用同一套 kernel 覆盖它们**：

| | prefill / chunked prefill | uniform decode | mixed（chunked prefill + decode） |
|---|---|---|---|
| `max_query_len` | 可能到几千 | 1（投机解码时 `1+n`） | 批内最大值 |
| `slot_mapping` 有效长度 | 等于本 chunk 的 token 数 | 等于请求数 | 两者相加 |
| 典型后端行为 | 走 prefill kernel，切 split-K | 走 decode kernel，GQA 打包 | 按 `max_query_len` 二选一 |
| FULL CUDA Graph | 多数后端不支持 | 主力场景 | 只有 `AttentionCGSupport.ALWAYS` 支持 |

`num_actual_tokens` 和 padded 值分开存，就是因为一个 batch 同时存在两套坐标：kernel 按 padded 长度起 grid，采样只认真实 token。

一个具体例子：FlashAttention 3 的 builder 在 `build()` 里算 `max_num_splits`，并且把 AOT schedule 结果写回**持久 buffer**（`vllm/v1/attention/backends/flash_attn.py:670` 的 `_store_scheduler_metadata`）——因为这块 metadata 在 FULL CUDA Graph 里必须是同一个地址，否则重放时读到的还是捕获那一刻的旧值。

## 4. CUDA Graph

### 4.1 BatchDescriptor 是 dispatch key

`BatchDescriptor`（`vllm/forward_context.py:30`）是一个 frozen dataclass，五个字段：`num_tokens`、`num_reqs`、`uniform`、`has_lora`、`num_active_loras`。它的唯一职责是**唯一标识一张 padded batch**，越少字段越好。

`CudagraphDispatcher` 维护两组 key（`vllm/v1/cudagraph_dispatcher.py:39`）：`PIECEWISE` 一组、`FULL` 一组。dispatch 时按 `FULL > PIECEWISE > NONE` 顺序查（:305-:322），查不到就退回 eager。

三个细节决定了 key 的形状：

- **padding 是向上取整**。`_compute_bs_to_padded_graph_size`（:71）把 `[0, max]` 每个 batch size 映射到「不小于它的最小捕获尺寸」。
- **uniform decode 的 `num_reqs` 是推出来的**，不是传进来的：`num_reqs = num_tokens_padded // uniform_decode_query_len`（:142），并要求整除。所以 decode 的 key 里 `num_reqs` 只是个派生量。
- **PIECEWISE 的 key 更宽松**：`num_reqs=None, uniform=False`（:314），因为分段图对请求数不敏感；FULL 的 key 必须精确，否则 FA3 的 scheduler metadata 对不上。

### 4.2 捕获与重放

捕获发生在第一次带非 `NONE` 模式跑 `_dummy_run` 的时候。`capture_model`（`vllm/v1/worker/gpu_model_runner.py:6407`）先抓大 shape 再抓小的，让小的复用大的显存池（:6423），完成后打印耗时——代码注释给的范围是 **5~20 秒**（:6505）。重放路径就一句 `entry.cudagraph.replay()`（`vllm/compilation/cuda_graph.py:361`）。

默认模式是 `FULL_AND_PIECEWISE`：O2 与 O3 两个优化档都写死它（`vllm/config/vllm.py:326`、:347），O1 是 `PIECEWISE`（:305），O0 才是 `NONE`（:284）。语义是「uniform decode 走整图，其余走分段图」——理论上最优，代价是显存和捕获时间都最高。

### 4.3 为什么 attention 曾经进不了 graph

vLLM 最早的分段编译就是为了把 attention 排除在外：「excluding cudagraph-unsupported operations (mainly attention)」（`docs/design/cuda_graphs.md:25`）。原因有三层，今天仍然能在代码里看到残留：

1. **能力分级**。`AttentionCGSupport` 只有四档：`ALWAYS` / `UNIFORM_BATCH` / `UNIFORM_SINGLE_TOKEN_DECODE` / `NEVER`（`backend.py:570`），混布模型取所有后端的最小值，再降级模式（`gpu_model_runner.py:6773`）。
2. **控制流依赖真实长度**。`max_query_len`、`max_seq_len` 决定 kernel 选型与 split 数，graph 里这些都是常量。
3. **metadata 里有 CPU 算的东西**。AOT schedule、序列边界搜索都在 host 侧，图里录不进去，只能像 FA3 那样写回固定地址再重放。

### 账本二：padding 浪费多少 token

捕获尺寸不是连续的，默认网格是 `[1,2,4] + range(8,256,8) + range(256,max+1,16)`（`vllm/config/vllm.py:2467`），`max` 取 `min(max_num_seqs × decode_query_len × 2, 512)`（:2525，Blackwell 家族 1024）。于是 decode 一步的真实 token 数会被抬到最近的一档：

| 真实 token | 捕获尺寸 | 多算的 token | 相对浪费 |
|---|---|---|---|
| 100 | 104 | +4 | 4.0% |
| 200 | 200 | 0 | 0% |
| 250 | 256 | +6 | 2.4% |
| 257 | 272 | +15 | 5.8% |
| 500 | 512 | +12 | 2.4% |
| 513 | 超过上限 | 不捕获，退回 eager | 0%（但丢了整图收益） |

最坏点在 257：网格刚从步长 8 切到 16，一跳 15 个 token。但**多算 15 行不等于多花 5.8% 的时间**：decode 是权重带宽受限的，权重每步只读一遍，多出来的行只在 attention 与 MLP 的 token 维上摊薄——真实开销远小于 token 比例。真正贵的是 513 那一行：整图失效，Python 逐层发 kernel 的 overhead 全暴露。

## 5. 2026 年的两个新变量

### 5.1 异步调度：CPU 不等 GPU

调度器不再等本步结果才排下一步。`AsyncScheduler._update_after_schedule`（`vllm/v1/core/sched/async_scheduler.py:36`）给每个 decode 请求累加 `num_output_placeholders`，用「假设这一步会出 token」的乐观值继续排第 N+1 步。代价转移到了执行侧：

- **采样结果不回 CPU**。上一步采到的 token 直接从 GPU 上的 `prev_sampled_token_ids` scatter 进 `input_ids`（`vllm/v1/worker/gpu_model_runner.py:1849`），不再走 D2H→Python→H2D。
- **复用 CPU buffer 要设栅栏**。`synchronize_input_prep`（:3572）用 event 保证「上一步的异步 H2D 已经读完这块 pinned 内存」才允许重写（:3580）。
- **索引要跨版本对得上**。`condense()` 会搬行，所以 `prev_req_id_to_index` 与 `prev_positions` 是必需品，不是优化。

### 5.2 MRV2：把持久状态和每步输入拆开

实验性的 Model Runner V2（`vllm/v1/worker/gpu/model_runner.py`，开关 `VLLM_USE_V2_MODEL_RUNNER`）认为 V1 的持久 batch 本身就是技术债：持久张量**同时**当模型输入，就锁死了行序，于是必须靠整表重排来维持（`docs/design/model_runner_v2.md:23`），还得留一份 `CachedRequestState` 做备份。

MRV2 的做法是三层解耦：

1. **持久状态**：请求拿到一个终身不变的行号（`vllm/v1/worker/gpu/states.py:100` 的 `free_indices.pop()`），抢占视为结束、恢复视为新请求。
2. **每步输入**：`InputBatch` 退化成一个纯 dataclass（`vllm/v1/worker/gpu/input_batch.py:43`），靠 `idx_mapping`（:50）间接指向持久行。行序由 attention 后端定，改行序只是改一个 int64 向量。
3. **写入方式**：块表用 `StagedWriteTensor`（`vllm/v1/worker/gpu/buffer_utils.py:150`）暂存 diff 再一次 kernel 落地；token 表干脆放 UVA（host 内存），GPU kernel 直接读，连 H2D 都省了。

最关键的差别是**去掉了 async barrier**。V1 里「CPU 改 pinned buffer」与「GPU 还在读它」会打架，只能加同步；MRV2 把非 pinned 的 CPU 真相和一份副本分开（`docs/design/model_runner_v2.md:84`），大张量干脆放 UVA 让 GPU 直接读 host 内存（:143）。CPU 写自己的，GPU 读副本，竞争自然消失。这不是性能技巧，是正确性模型换了。

## 6. 常见坑

- **padded token 不能参与采样**：CUDA Graph 按 padded shape 跑，logits 只取前 `num_actual_tokens` 行；取错会把 padding 的随机 logits 采成 token。
- **`condense()` 会搬块表行**：请求结束时把末尾行搬下来填空洞，块表、采样参数、generator 都要跟着搬；行号映射错一次，KV 就写到别的请求上。
- **CUDA Graph 的 shape 网格不是越密越好**：每个捕获尺寸都占显存和捕获时间，网格过密会让启动慢、KV 池变小。
- **后端不支持 FULL graph 时是静默回退**：`AttentionCGSupport` 不满足就退回 piecewise/eager，日志里只有一行，性能却差一截。

## 7. 读代码时的三个检查点

**检查点一：这个 `num_tokens` 是 padded 还是 unpadded？**
同一个函数里三个变量并存：`num_tokens_unpadded`（调度器给的）、`num_tokens_padded`（`batch_desc.num_tokens`）、`num_actual_tokens`（metadata 里的）。用错不会报错，只会让 slot_mapping 尾部少填几个 `-1`，然后在某个 batch size 上偶发 NaN。

**检查点二：padding 的两道防线是谁在守？**
写侧靠 `slot_mapping = -1`（`vllm/v1/worker/gpu_model_runner.py:3870`），读侧靠 `block_table = 0`（:2309）。如果你改了 padding 逻辑却只改了一边，症状是「某些步的 logits 里混进了别的请求的 KV」——比崩溃难查得多。

**检查点三：你看到的行号是 batch 行还是 request 行？**
`condense()`、`_may_reorder_batch()`（`gpu_model_runner.py:1105`，后端可以把 decode 提到前面）、异步调度三者都会改行序。任何跨步保存的索引都要问一句「它属于哪一版行序」，否则在单请求压测里永远复现不出来。

## 小结

- 一个 step = 改持久状态（`_update_states`）+ 算三件套（`_prepare_inputs`）+ 建 metadata + 查表重放；`slot_mapping` 是调度器与 attention kernel 之间唯一的写接口。
- 持久 batch 不是优化而是约束：`token_ids_cpu` 在 128K 上下文下就是 512 MiB，而且 CUDA graph 要求输入地址恒定。
- 增量更新最脏的地方是 `condense()`：搬行会重写 `req_id → index`，所以跨步索引必须显式保存。
- 块表每步整行重传，128K 上下文、256 并发时是 8 MiB/步（25 GB/s 下约 336 μs）；这是 MRV2 用 `StagedWriteTensor` 直接针对的目标。
- `BatchDescriptor` 是 dispatch key，`FULL > PIECEWISE > NONE` 顺序匹配；默认 `FULL_AND_PIECEWISE`，捕获一次 5~20 秒。
- CUDA graph 的 padding 最坏浪费 5.8%（257 → 272），但 decode 受权重带宽限制，真实开销远小于这个比例；真正的悬崖是超出最大捕获尺寸后整图失效。
- 异步调度让 CPU 排第 N+1 步、GPU 跑第 N 步；MRV2 进一步把持久状态、每步输入、写入通道三者拆开，从而去掉 async barrier。

下一篇看 attention 后端本身：[Attention 后端：FlashAttention、FlashInfer 与 MLA 怎么选]({{< relref "vllm-06-attention-backends" >}})。调度器怎么切 token budget 见[调度器那篇]({{< relref "vllm-04-scheduler" >}})。

参考文档：<https://docs.vllm.ai/en/latest/design/cuda_graphs/>、<https://docs.vllm.ai/en/latest/design/model_runner_v2/>（对应仓库 `docs/design/cuda_graphs.md`、`docs/design/model_runner_v2.md`）。
