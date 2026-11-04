---
title: "vLLM 研读（二）PagedAttention：块、块表与 KV 显存账本"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 3
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "PagedAttention", "KV Cache", "显存管理"]
categories: ["推理框架"]
---

上一篇把请求送到了调度器门口。真正决定"能同时服务多少请求"的，是接下来这一步：**给每个 token 的 KV 找一块显存**。

朴素做法是给每个请求按 `max_model_len` 预留一段连续显存。这个做法在小模型、短上下文时还凑合；一旦上下文到 8k、并发到几百，显存会在两个地方同时漏：请求实际只用了几百 token，剩下的预留在空转（内部碎片）；请求长短不一，连续分配留下大量无法利用的缝隙（外部碎片）。

PagedAttention 的答案是把 KV 切成固定大小的块，按需分配，用一张块表把"逻辑上的第 i 个 token"映射到"物理上的第 b 块第 j 个槽位"。这一篇讲清楚三件事：

- 一个 KV 块里到底存了什么，块表长什么样？
- 分配器怎么在 O(1) 时间拿到/归还块，又怎么和前缀缓存共用同一套结构？
- 从块表到 attention kernel，中间那一步 `slot_mapping` 是怎么算的？

## 1. 先记住一个画面

一个请求的 8k 个 token，在逻辑上是连续的 `[0, 1, 2, ..., 8191]`。在物理显存里，它们被切成 512 个块，可能散落在显存的任意位置：

```
逻辑:   token 0-15   token 16-31  token 32-47   ...   token 8176-8191
物理:   block 173    block 902    block 44      ...   block 611
```

块表（block table）就是每个请求一行、每行 512 列的整数数组：

```text
request A: [173, 902, 44, ..., 611]
request B: [ 17,  18, 19, ..., 530]   # 和 A 共享前 3 块（同一段系统提示）
```

attention kernel 不再假设"第 i 个 token 在基址 + i×stride"，而是先查 `block_table[req][i // block_size]` 拿到块号，再加上块内偏移。**一个间接层，换来了显存利用率**。

## 2. 碎片账：为什么必须分页

继续用上一篇的配置：H800 80 GB，`gpu_memory_utilization=0.9`，Llama-3-8B bf16（权重 16 GB），约 54 GB 给 KV。

Llama-3-8B 的 KV 是 32 层、8 个 KV head、head_dim 128、bf16：

$$
\text{KV/token} = 2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{B} = 128\ \text{KiB}
$$

按块大小 16 算，一个块是 $16 \times 128\ \text{KiB} = 2\ \text{MiB}$。54 GB 约合 **27,648 个块**。

| 方案 | 单请求占用 | 54 GB 能装 | 浪费 |
|---|---|---|---|
| 连续预留 8k | 1 GiB（512 块） | 54 个请求 | 实际 500 token 也占满 8k |
| 分页 + 按需 | 500 token → 32 块（64 MiB） | 约 864 个请求 | 每请求最多浪费 15 个 token 的槽位 |

差距是 **16 倍**。这就是为什么 PagedAttention 不是"一个 kernel 优化"，而是推理框架的底座：没有它，continuous batching 根本拼不出大 batch。

> **类比的边界**：分页很像操作系统的虚拟内存，但 vLLM 的块**没有 swap**。块不够时不会换出到磁盘，而是抢占请求、丢弃 KV、之后重算。所以"显存不足"的代价是**重复的 prefill 计算**，不是"慢一点"。

## 3. 三个数据结构

### 3.1 `KVCacheBlock`：块的元数据

`vllm/v1/core/kv_cache_utils.py:164`：

```python
@dataclass(slots=True)
class KVCacheBlock:
    block_id: int
    ref_cnt: int = 0
    pool: BlockPool | None = None
    _block_hash: BlockHashWithGroupId | None = None
    _block_hash_num_tokens: int | None = None
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    is_null: bool = False
```

三个字段撑起了整套机制：

- `ref_cnt`：有多少个请求在用这块。前缀缓存命中时加一，请求结束或抢占时减一；减到 0 才可能被回收。
- `_block_hash`：块内容（token 序列 + 父块哈希）的指纹。只有**写满的块**才有哈希；有哈希且 `ref_cnt=0` 的块是"可复用但可被淘汰"的缓存块。
- `prev/next_free_block`：侵入式双向链表指针，让"从空闲队列中间摘掉一块"也是 O(1)。

注意 `block_id` 是物理块号，范围 `[0, num_blocks)`；它同时也是 `slot_mapping` 计算里的乘数。

### 3.2 `FreeKVCacheBlockQueue`：LRU 链表

`vllm/v1/core/kv_cache_utils.py:260`。它没有用 Python 的 `deque`，而是自己维护双向链表，原因写在注释里：需要 O(1) 从中间删除（前缀缓存命中时要把块从空闲队列摘出来），又不想为每次操作分配 Python 对象。

队列的排序规则是整套缓存淘汰策略的核心：

1. **最近最少使用的块在最前面**（LRU）；
2. 同一请求释放的块中，哈希覆盖 token 更多的（链尾）排更前——因为链尾块被复用的概率更低。

### 3.3 `BlockPool`：块池

`vllm/v1/core/block_pool.py:137`。它是"物理块的全集 + 哈希表 + 空闲队列"的组合：

| 方法 | 作用 |
|---|---|
| `get_new_blocks(n)` | 从空闲队列头部取 n 块，ref_cnt 置 1；若块还挂在缓存哈希表里，先淘汰 |
| `cache_full_blocks(...)` | 请求算满一块后，把块哈希登记进 `cached_block_hash_to_block` |
| `get_cached_block(hash)` | 前缀缓存查询：命中则 ref_cnt++ 并从空闲队列摘除 |
| `touch(blocks)` | 命中已有块时增加引用计数 |
| `free_blocks(blocks)` | 归还块；无哈希的块插到队头（LIFO，GPU 局部性好），有哈希的块追加到队尾（FIFO，LRU 淘汰） |

`free_blocks` 里那个"有哈希追加、无哈希前插"的分支（`vllm/v1/core/block_pool.py:818` 起）值得单独看一眼：它不是随意选择，而是同时服务两个目标——刚释放的临时块尽快被同一请求的下一个 chunk 复用（局部性），而缓存块要尽量活久一点（命中率）。

## 4. 分配：`allocate_slots` 的三段式

调度器每步对每个被选中的请求调用 `KVCacheManager.allocate_slots()`（`vllm/v1/core/kv_cache_manager.py:379`）。它的 docstring 里画了一张布局图，简化后是：

```text
|<-- comp -->|<-- new_comp -->|<-- ext_comp -->|<-- new -->|<-- lookahead -->|
                              |<---------- 本次要分配 ---------->|
|<-------------- 已被前缀缓存/连接器覆盖 -------------->|
```

- `comp`：请求已经算过的 token（`num_computed_tokens`）；
- `new_comp`：本次新命中的前缀缓存块；
- `ext_comp`：连接器（P/D 分离）从远端拿到的 token；
- `new`：本次真正要算的 token，包含未验证的 draft token；
- `lookahead`：投机解码为 draft 预留的槽位。

三个阶段：

1. **清理**：把滑出滑动窗口的块、不再需要的块释放；如果剩余空闲块不够，直接返回 `None`；
2. **前缀**：为 `new_comp` / `ext_comp` 部分做引用计数或新分配；
3. **新 token**：为 `new + lookahead` 分配块，可能跨越多个块。

返回 `None` 是调度器的信号：这个请求这一步放不下。调度器要么把它跳过，要么抢占别的请求（下一篇细讲）。

一个容易忽略的细节：**未验证的 draft token 也会占块**。投机解码提议 4 个 token，就要为这 4 个位置预留 KV；验证后如果只接受 2 个，多出来的块要还回去。这就是 `allocate_slots` 里 `num_lookahead_tokens` 参数的来历。

## 5. 从块表到 kernel：`slot_mapping`

模型 forward 时，每个 token 的 hidden state 要写进 KV cache 的某个位置。这个位置由 `slot_mapping` 给出：

$$
\text{slot}(t) = \text{block\_table}[r][t // B] \times B + (t \bmod B)
$$

其中 $B$ 是块大小，$r$ 是请求在 batch 里的行号，$t$ 是该 token 在序列里的位置。

这个计算现在是一个 Triton kernel：`ComputeSlotMappingKernel`（`vllm/v1/worker/block_table.py:305` 起）。核心几行：

```python
virtual_block_indices = pos // virtual_block_size
virtual_block_offsets = pos - virtual_block_indices * virtual_block_size
block_numbers = tl.load(block_table_ptr + row_offset + virtual_block_indices, ...)
slot_ids = block_numbers * KV_CACHE_BLOCK_SIZE + local_block_offsets
```

之所以要写成 kernel 而不是 Python 循环，是因为 `max_num_batched_tokens` 可以到 16384，每步 1.6 万个 slot 的 Python 计算会直接吃掉几毫秒 CPU 时间。kernel 还顺手把超出 `num_tokens` 的槽位填成 `PAD_ID`，让 CUDA Graph 的固定 shape 不会越界。

KV cache 本身在显存里的布局是 `[num_blocks, num_kv_heads, block_size, head_size]`（不同后端有 `NHD`/`HND` 两种排布），所以拿到 slot 后，写入地址是：

```text
k_cache[slot, head, :, :]  # 或 k_cache[slot, :, :, head]
```

attention kernel 读的时候则用 `block_table + seq_lens` 重建每个请求的 KV 序列。**这就是 paged attention 的全部魔法**：写的时候按 slot 散着写，读的时候按块表 gather。

## 6. 显存账：`num_blocks` 从哪来

块数不是拍脑袋定的，是启动时 profile 出来的。`GPUWorker.determine_available_memory()`（`vllm/v1/worker/gpu_worker.py:716`）的流程：

1. 用 dummy 输入跑一次 forward（`profile_run`），记录峰值显存；
2. 计算 `requested_memory = total_memory × gpu_memory_utilization`；
3. `available_kv_cache_memory = requested_memory - non_kv_cache_memory - cudagraph_memory_estimate`（`gpu_worker.py:824` 附近）；
4. `num_blocks = available_kv_cache_memory // 每块字节数`。

`non_kv_cache_memory` 包含权重、激活、workspace；`cudagraph_memory_estimate` 是 capture 所有 CUDA Graph 的额外开销——这也是为什么开了 `FULL_AND_PIECEWISE` 之后 KV 池会变小，以及为什么有时候"调大 `gpu_memory_utilization` 反而 OOM"：你把该留给 graph 的余量也吃掉了。

2026 年还多了一个选项：`--kv-cache-memory-bytes`，直接指定 KV 池大小，跳过 profile（`gpu_worker.py:729` 起的提前返回分支）。它不尊重 `gpu_memory_utilization`，适合"我知道这块卡只跑这一个服务"的场景。

每块字节数不是简单的 `block_size × 128 KiB`，而是该 KV 组里**所有层**的 `page_size_bytes` 之和（`vllm/v1/kv_cache_interface.py:590`）：

$$
\text{bytes/block} = \sum_{\text{layer} \in \text{group}} \text{block\_size} \times n_{\text{kv\_heads}} \times (d_{\text{head}} + d_{\text{head\_v}}) \times \text{dtype\_size}
$$

对 Llama-3-8B，就是 $16 \times 32 \times 8 \times 128 \times 2 \times 2 = 2\ \text{MiB}$。

## 7. 混合模型：块大小怎么统一

sliding window、Mamba、稀疏注意力混在一个模型里时，"每层都要 block_size 个 token 的槽位"这个假设不成立：

- full attention 层要为**所有** token 留 KV；
- sliding window 层只需要最近 `window_size` 个 token；
- Mamba 层存的是状态（state），不是逐 token 的 K/V。

vLLM 的做法是**一个共享块池 + 多个 KV 组**（KV cache group），每组有自己的 `page_size_bytes` 和分配策略，但块号从同一个池子里出。为了保证不同组的"页大小"一致，混合管理器会调整每组的层数或块大小（`docs/design/hybrid_kv_cache_manager.md` 里有完整的推导）。对外表现就是：

- 一个逻辑块在每个组里各占一份物理空间，所以 `num_blocks = 可用显存 / 各组 page_size_bytes 之和`——不是取最贵的那个组；
- 一个请求会同时持有多个组的块表（`block_tables` 是一个列表，每个 KV 组一个）；
- sliding window 组的块表会出现 `is_null` 的占位块，表示"这段被窗口跳过了"。

这也解释了为什么混合模型的 `block_size` 可能不是 16：不同 backend 对块大小有偏好，平台层会取所有 backend 都能接受的值（`vllm/platforms/interface.py:705` 起）。

## 8. 常见坑

- **块表按 `max_model_len` 开**：`max_num_blocks_per_req = ceil(max_model_len / block_size)`，每个请求一行。`max_model_len=128k` 时，1024 个请求的块表就是 1024 × 8192 × 4 B = 32 MiB，且每行都要清零/填充。长上下文场景下这不是零成本。
- **块大小不是越小越好**：块小、碎片少，但块表更长、kernel 的间接寻址更多；块大则相反。16 是默认的折中，MLA/FA3 等后端会偏好 64。
- **`ref_cnt` 泄漏**：前缀缓存命中后忘记 `touch`（或 `free` 多减一次），块会永远"看起来有人用"，表现为 KV 使用率只涨不跌。排查时看 `KVCacheBlock.ref_cnt` 和 `free_block_queue.num_free_blocks` 是否对得上。
- **null block**：sliding window 跳过的位置用共享的 null block 占位，它的 `is_null=True`，不参与引用计数和缓存。看到块表里一片相同的 block_id，先确认是不是 null block，而不是分配器坏了。
- **`num_gpu_blocks_override`**：测试抢占时可以用它强行把块数调小，但生产环境别用——它会让 profile 结果失真。

## 小结

- PagedAttention 把 KV 的分配单位从"请求"变成"块"（默认 16 token），碎片从"按 max_model_len 预留"降到"每请求最多 15 个 token"。
- 8B 模型每 token 128 KiB、每块 2 MiB；54 GB 的 KV 池约 2.7 万块，500 token 的请求只占 32 块。
- 三个结构：`KVCacheBlock`（引用计数 + 哈希 + 链表指针）、`FreeKVCacheBlockQueue`（O(1) 摘除的 LRU 链表）、`BlockPool`（分配 + 缓存 + 淘汰）。
- `allocate_slots` 三段式：清理 → 前缀 → 新 token；返回 `None` 就是"放不下"的信号。
- `slot_mapping = block_id × block_size + offset`，由 Triton kernel 批量算，兼顾 CUDA Graph 的固定 shape。
- `num_blocks = (可用显存 - 权重/激活/CUDA Graph) / 每块字节数`，是启动时 profile 出来的，不是配置项。
- 混合模型用"一个块池 + 多个 KV 组"统一页大小，块表因此变成"每组一张"。

下一篇讲这些块怎么被复用：[前缀缓存的哈希链、引用计数与淘汰策略]({{< relref "vllm-03-prefix-caching" >}})。

> 官方文档：<https://docs.vllm.ai/en/latest/design/paged_attention/>（历史设计，开头已声明不再描述当前实现）、<https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/>（混合模型的块大小统一）
