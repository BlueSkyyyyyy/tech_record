---
title: "vLLM 研读（三）前缀缓存：一条哈希链省下多少 prefill"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 4
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "前缀缓存", "Prefix Caching", "KV Cache"]
categories: ["推理框架"]
---

在线服务的 prompt 有一个统计规律：**开头高度重复，结尾各不相同**。系统提示、few-shot 示例、工具定义、代码库上下文——这些内容在成百上千个请求里一字不差。如果每次都重新 prefill，你花的算力有相当一部分是在重复同一件事。

前缀缓存（Automatic Prefix Caching, APC）就是把"已经算过的 KV"留下来复用。听起来简单，但难点在一个约束：**命中缓存不能改变输出**。这意味着你不能"差不多就复用"——必须能证明两个请求的某段前缀在语义上完全相同（同 token、同 LoRA、同多模态输入、同隔离域），并且 KV 的物理块可以被安全共享。

vLLM 的答案是给每个块算一条**哈希链**。这一篇讲清楚：

- 块哈希为什么必须包含父块哈希，`extra_keys` 里放什么？
- 命中之后块的引用计数怎么变，为什么还会涉及块拷贝？
- 淘汰、隔离和多租户安全怎么处理，代价是什么？

## 1. 先记住一个画面

假设你的服务有一个 2,000 token 的系统提示，后面接用户问题：

```text
[system prompt: 2000 tokens][user: "帮我改这段代码"][assistant: ...]
```

第一个请求：prefill 2,000 token，算完把前 125 个块（2,000 / 16）标记为"已缓存"。
后面 99 个请求：同样 2,000 token 的系统提示 → 直接命中 125 个块，prefill 从 2,000 token 降到几十 token（用户问题）。

用 8B 模型估一下省下的算力：每个请求的 prefill 是 $2 \times 2000 \times 8\text{e}9 \approx 32$ TFLOP；按 H100 上 400 TFLOPS 有效算力算，约 80 ms。99 个请求就是 **约 8 秒的 GPU 时间**，还没算这 99 次 prefill 期间被阻塞的 decode。

而代价只是显存里留着这 125 个块（250 MiB）——只要它们不被淘汰。

## 2. 哈希链：块的身份

### 2.1 为什么不能只哈希块内 token

如果块哈希只由块内 16 个 token 决定，那么两个不同请求里"恰好相同的 16 个 token"会被误认为同一段前缀。例如：

```text
请求 A: [The capital of France is][Paris, a city ...]
请求 B: [I went to France is][Paris, a city ...]
```

第 2 块都是 `[Paris, a city ...]`，但前面完全不同，KV 不能共享。

所以 vLLM 的块哈希是**链式**的（`vllm/v1/core/kv_cache_utils.py:709`）：

```python
def hash_block_tokens(hash_function, parent_block_hash, curr_block_token_ids, extra_keys=None):
    if not parent_block_hash:
        parent_block_hash = NONE_HASH
    return BlockHash(hash_function((parent_block_hash, tuple(curr_block_token_ids), extra_keys)))
```

第 $i$ 块的哈希 $h_i = H(h_{i-1}, \text{tokens}_i, \text{extra}_i)$。任何一块不同，后面所有块的哈希都不同——这正好对应"前缀必须完全一致"的语义。

第一块没有父块，用 `NONE_HASH` 代替。这个种子不是随便取的：

- 对 SHA-256 这类密码学哈希，种子是固定的（`"vllm-none-hash"`），所以**不同进程、不同机器**对相同内容算出相同哈希，可以跨实例共享缓存；
- 对 xxhash 这类非密码学哈希，种子每进程随机，否则攻击者可以离线构造碰撞（`init_none_hash` 的注释里写得很清楚）。

### 2.2 `extra_keys`：让"相同 token"变成"相同 KV"

光有 token 还不够。以下情况 token 完全相同，KV 也不能共享：

| 情况 | 放进 extra_keys 的东西 |
|---|---|
| LoRA | LoRA 的 id/权重指纹 |
| 多模态 | 图片/音频的内容哈希（同一个占位符可能对应不同图片） |
| 多租户隔离 | `cache_salt`（请求级盐） |
| prompt embeds | embedding 张量的哈希（只有向量、没有 token id） |

`generate_block_hash_extra_keys()`（`vllm/v1/core/kv_cache_utils.py:639`）负责收集这些键。`cache_salt` 只加在第一块上（`start_token_idx == 0`），因为盐一旦不同，后续哈希链自然全不同。

另外，块的哈希还要带上 **KV 组 id**（`make_block_hash_with_group_id()`，`kv_cache_utils.py:62`），因为混合模型的不同组（full attention / sliding window / Mamba）对同一段 token 存的是不同的东西。

### 2.3 默认算法与可复现性

`--prefix-caching-hash-algo` 有四个选项（`vllm/config/cache.py` 的 `prefix_caching_hash_algo`）：

- `sha256`（默认）：Pickle 序列化 + SHA-256，安全，但**跨 Python/vLLM 版本不保证可复现**；
- `sha256_cbor`：CBOR 规范化序列化 + SHA-256，跨语言/跨版本可复现，适合多实例共享缓存；
- `xxhash` / `xxhash_cbor`：更快，但非密码学安全，多租户场景要自己评估碰撞风险。

这条选择在"单机单实例"时无所谓，在"多个 vLLM 实例共享一个 KV 存储（如 LMCache/Mooncake）"时很关键：哈希不一致，缓存就永远不命中。

## 3. 命中路径：从请求到块表

调度器为一个新请求做前缀查询时，调用 `KVCacheManager.get_computed_blocks()`（`vllm/v1/core/kv_cache_manager.py:266`）：

1. 请求在创建时就已经算好了 `block_hashes`（`Request.update_block_hashes()`，`vllm/v1/request.py:290`），每追加 token 就增量更新；
2. `coordinator.find_longest_cache_hit()` 从第一块开始逐块查哈希表，直到某个哈希缺失或超过长度上限；
3. 命中的块逐个 `touch()`（`ref_cnt++`），从空闲队列摘除；
4. 返回命中的块列表和命中的 token 数。

这里有三个容易忽略的边界：

**只缓存写满的块。** 最后不满 16 个 token 的块不会被缓存——它的 KV 不完整，复用了也没用。2026 年的 `prefix_match_unit`/partial hash hits 对混合模型（尤其是 Mamba 的 align 模式）放宽了这一点，允许命中落在物理块内部，但需要额外的块拷贝来保持"块内前缀"的一致性。

**最后一个 token 总要重算。** `Request.get_max_cache_hit_length()`（`vllm/v1/request.py:329`）里写着 `min(self.num_tokens - 1, ...)`：即使整个 prompt 都命中，也要重算最后一个 token 才能拿到 logits。而 `allocate_slots` 要求 `num_computed_tokens` 块对齐，所以实际可能重算一整块（最多 15 个 token 的浪费）。

**有些请求不能读缓存。** `prefix_cache_lookup_enabled()` 在两种情况下返回 `False`：

- 请求要 `prompt_logprobs`：需要 prompt 每个位置的 logits，跳过前缀就没有了；
- pooling 模型做全序列池化：需要每个 token 的 hidden state。

## 4. 引用计数：共享的代价

前缀缓存把块从"一个请求独占"变成"多个请求共享"，于是需要引用计数：

- 命中：`ref_cnt++`；
- 请求结束/被抢占：`ref_cnt--`；
- `ref_cnt == 0` 且块有哈希 → 块进入空闲队列尾部，**仍然可读**，直到被重新分配；
- `ref_cnt == 0` 且块无哈希 → 直接可复用。

这就是为什么 `free_blocks()` 要区分"有哈希"和"无哈希"（`vllm/v1/core/block_pool.py:818`）：无哈希的块是纯临时资源，插到队头尽快复用；有哈希的块是缓存资产，追加到队尾尽量活久。

> **一个反直觉的后果**：前缀缓存会让"显存使用率"看起来很高，但那不一定是坏事。`ref_cnt=0` 的缓存块随时可以被抢占，它们只是"还没被用掉的免费额度"。看监控时要区分 **active KV**（ref_cnt>0）和 **cached KV**（ref_cnt=0 但有哈希）。

## 5. 淘汰与隔离

### 5.1 淘汰

淘汰是**惰性**的：没有独立的 eviction 线程，分配器在 `get_new_blocks()` 里发现队头的块还挂着哈希，就现场把它从哈希表摘掉、`reset_hash()`，然后交出去（`_maybe_evict_cached_block()`，`vllm/v1/core/block_pool.py:771`）。

这带来一个可预测的行为：**空闲块越少，缓存命中率越低**。因为队头是 LRU 最久没用的块，压力大时最先牺牲的就是它们。

vLLM 还提供了两个手动入口：

- `reset_prefix_cache()`：清空整个前缀缓存（调度器暂停后调用，保证没有在途请求引用旧块）；
- `evict_blocks(block_ids)`：按块号精确淘汰（给 KV 事件/外部缓存管理用）。

### 5.2 隔离

多租户场景下，前缀缓存是一个**侧信道**：如果租户 A 和 B 用相同的系统提示，B 可以通过命中率/延迟推断出 A 用过什么。`cache_salt` 就是为此设计的：请求带上不同的盐，哈希链从第一块就分叉，缓存物理隔离。

代价是：同一个租户内的复用仍在，但跨租户的复用完全消失。要不要开，取决于你的威胁模型——公有云 API 通常默认关闭跨租户复用，私有部署则可以全开。

## 6. 账：命中率、收益与成本

### 6.1 收益

前缀缓存的收益可以写成：

$$
\text{saved\_prefill\_tokens} = \sum_{\text{req}} \text{hit\_tokens}(\text{req})
$$

命中率 $\text{hit\_rate} = \frac{\sum \text{hit\_tokens}}{\sum \text{prompt\_tokens}}$。它对 TTFT 的影响近似线性：prefill 时间 $\propto$ 未命中 token 数。

一个真实的服务形态（系统提示 2k、用户输入平均 200 token）：

| 场景 | 命中率 | 平均 prefill token | TTFT 改善 |
|---|---|---|---|
| 无缓存 | 0% | 2,200 | 基线 |
| 有缓存，系统提示稳定 | ~90% | 220 | 约 10× |
| 有缓存，但每请求都改系统提示 | ~0% | 2,200 | 无 |

### 6.2 成本

- **显存**：缓存块占用 KV 池，直接减少可并发请求数。2k token 的系统提示 = 125 块 = 250 MiB。
- **哈希计算**：每块一次 SHA-256，8k prompt 约 512 次，量级在毫秒；请求创建时在输入处理线程算，和模型 forward 重叠。
- **块表与元数据**：命中的块要 `touch`、要维护引用计数，Python 层的开销随命中块数线性增长。
- **淘汰抖动**：如果工作集（所有活跃前缀）大于 KV 池，会出现"刚缓存就被淘汰"的抖动，命中率骤降。这时要么加显存，要么减少 `max_num_seqs`。

## 7. 常见坑

- **改了系统提示的一个空格，命中率归零**。哈希对 token 序列敏感，tokenizer 版本、chat template 的空格、特殊 token 都会改变 token 序列。
- **多实例共享缓存时哈希算法要一致**。默认 `sha256` 用 Pickle，不同 Python 版本可能序列化不同；跨实例场景用 `sha256_cbor`。
- **EAGLE 会主动丢掉最后命中的一块**。投机解码需要最后一块的 hidden state 来喂 draft head，所以 `find_longest_cache_hit()` 在 EAGLE 下会 `drop_eagle_block`，强制重算一块。命中率会因此略低，这是设计使然。
- **`prompt_logprobs` 与 APC 互斥**。要 prompt logprobs 的请求不会读缓存，别指望它享受命中。
- **`reset_prefix_cache` 不是免费的**。它会暂停调度、等在途请求排空，在线服务里应该当作运维操作，而不是定期任务。
- **KV 事件与外部缓存**。如果你用 LMCache/Mooncake 把 KV 存到 CPU/远端，`kv_events` 的块事件是另一套一致性机制，别和本地 APC 的引用计数混为一谈。

## 小结

- 块哈希是链式的：$h_i = H(h_{i-1}, \text{tokens}_i, \text{extra}_i)$，任何前缀差异都会让后续哈希全变。
- `extra_keys` 把 LoRA、多模态内容、`cache_salt`、prompt embeds 纳入身份，保证"同 token 同 KV"。
- 默认 SHA-256 + 固定种子，跨进程可复现；跨实例共享要用 `sha256_cbor`。
- 命中路径：增量算哈希 → 逐块查表 → `touch` 增引用 → 返回块表；只缓存满块，最后一个 token 总要重算。
- 淘汰是惰性 LRU：分配时才清哈希；空闲块越少，命中率越低。
- 收益近似正比于命中 token 数；成本是显存、哈希和引用计数开销。
- `cache_salt` 用哈希分叉换隔离，是多租户场景的必需项。

下一篇进入调度器：[一个 step 的 token budget 怎么分、长 prompt 怎么切、KV 不够时抢占谁]({{< relref "vllm-04-scheduler" >}})。

> 官方文档：<https://docs.vllm.ai/en/latest/design/prefix_caching/>、<https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/>
