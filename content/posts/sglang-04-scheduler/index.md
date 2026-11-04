---
title: "SGLang 研读（四）调度器：PrefillAdder、chunked prefill 与 retract decode"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 5
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "调度器", "Chunked Prefill", "Continuous Batching"]
categories: ["推理框架"]
---

前三篇讲的是"资源长什么样"：[第 2 篇]({{< relref "sglang-02-radix-attention" >}})是基数树怎么认前缀，[第 3 篇]({{< relref "sglang-03-memory-pool" >}})是 KV 池和 slot 分配器怎么发牌。这一篇讲这些资源每一步怎么花出去。

调度器的输入是一堆状态各异的请求（刚来的、切了一半的、decode 到一半的、被 retract 过又排回队列的），输出是一个 `ScheduleBatch`：这一步跑谁、各自跑多少 token。约束有三个，互相耦合：

- **算力**：一步最多算 `max_prefill_tokens` 个 token（默认 16384），单条长 prompt 还要被 `chunked_prefill_size` 再切一刀；
- **请求数**：一步最多收几条，受 `req_to_token_pool` 空闲行数、PP 的 micro-batch 上限、`max_running_requests` 限制；
- **KV**：准入一个新请求，不只买它这一步的 KV，还要买它"以后可能生成"的 KV。

SGLang 的答案和 vLLM 结构上不同：它没把 prefill 和 decode 融成一条连续的 token 流，而是**每一步二选一**，并且让两种请求分别去抢两本不同的账。

## 1. 先记住一个画面：两步一循环

一步不是"所有请求各跑一点"，而是两种步交替：

```text
step N   : [ 一条 32k prompt 的第 2/4 块，8192 token ]     ← EXTEND
step N+1 : [ 64 条 decode，各 1 token ]                     ← DECODE
step N+2 : [ 新来的 3 条短 prompt，共 2100 token ]          ← EXTEND
step N+3 : [ 64 条 decode，各 1 token ]                     ← DECODE
```

decode 步里没有 prefill，prefill 步里也没有 decode（除非开 `--enable-mixed-chunk`）。这就是为什么 SGLang 的 ITL 曲线是"整体很平、周期性鼓包"：鼓包高度等于一个 chunk 的 prefill 时间，而不是一整条 prompt 的 prefill 时间。

## 2. 一步的四个阶段，和 ScheduleBatch 的两副面孔

`event_loop_normal`（`scheduler.py:2107`）把一步拆成四段：

```python
self.ingest_requests()                       # 2115 收请求，进 waiting_queue
plan = self.get_next_batch_to_run(...)       # 2121 决定这一步跑谁
result = self.run_batch(batch)               # 2130 发 forward
self.process_batch_result(batch, result)     # 2131 收结果、放 KV、推状态机
```

`event_loop_overlap`（`2143`）把后两段错开一拍，那是[第 5 篇]({{< relref "sglang-05-overlap" >}})的事；本篇只看调度骨架。

同一份 `ScheduleBatch` 在两种模式下是完全不同的东西，这是读懂后面所有预算代码的前提：

| 字段 | EXTEND（`prepare_for_extend`，`schedule_batch.py:2285`） | DECODE（`prepare_for_decode`，`schedule_batch.py:3172`） |
|---|---|---|
| `input_ids` | 每 req 的 `[prefix_len:extend_end]` 首尾相接，长 `extend_num_tokens` | 上一步采样出的那 1 个 token，长 `bs` |
| `seq_lens` | `extend_end`（含本块要算的 token） | 上一步长度 +1 |
| `prefix_lens`/`extend_lens` | 有，一请求一项 | `None` |
| `out_cache_loc` | `extend_num_tokens` 个待写 slot | `bs` 个待写 slot |

一句话：**extend batch 是"变长 token 段的集合"，decode batch 是"等长 1 token 的集合"**。所有调度账都围绕这个差别展开——extend 的预算是"还能算多少 token"，decode 的预算是"还能写多少个 slot"。

## 3. get_next_batch_to_run：算力给 prefill，KV 给 decode

真正做选择的是 `get_next_batch_to_run`（`scheduler.py:4090`），骨架只有五行：

```python
if new_batch is not None:
    ret = new_batch                      # 4193: Run prefill first if possible
else:
    if not running_batch.is_empty() and not running_batch.is_prefill_only:
        decode_batch = self.update_running_batch(running_batch)
```

进函数后按顺序做四件事：

1. 上一步若是 extend 批，先过滤（去掉已完成的、去掉 chunked_req）再 `merge_batch` 进 running（`4153-4161`）；
2. `chunked_req` 从批里摘出来单独持有（`4107-4116`），合并时不会把"没切完的请求"当成普通请求；
3. `get_new_batch_prefill`（`4252`）试着从 waiting_queue 组一个 prefill 批；
4. 组出来了 → 这一步跑它，running 的 decode 让路；组不出来（或 `_should_defer_prefill` 命中）→ `update_running_batch`（`4767`）跑 decode。

所以"先续跑还是先塞新 prefill"的准确答案是：**算力上新 prefill 优先，KV 账本上 decode 优先**。

decode 让出这一步的 GPU，但它未来几百步要用的 KV，已经在 `PrefillAdder.__init__` 里被预扣掉了（`schedule_policy.py:679-685`）。这个不对称是设计核心：prefill 是"能等"的（多等一步只伤 TTFT），decode 是"等不起"的（每步都在涨 KV，用户已经在盯着字往外蹦），所以让 decode 占账本、prefill 占算力。反方向也成立：KV 账本紧到 PrefillAdder 一个都塞不进，这一步就只剩 decode——**SGLang 不会为塞 prefill 去挤 decode 的 KV**。

## 4. PrefillAdder 的三重预算

`PrefillAdder` 在 `schedule_policy.py:609`，构造在 `scheduler.py:4373-4393`。参数里有三个数字决定它多能吃：`new_token_ratio`（4378）、`max_prefill_tokens`（4379）、`chunked_prefill_size`（4359 起先做过一次动态调整）。

### 4.1 第一重：token（这一步能算多少）

两个计数器、两个不同的上限：`rem_input_tokens = max_prefill_tokens - 本步已花`，默认 16384（`arg_groups/fields/schedule.py:71`）；`rem_chunk_tokens = chunked_prefill_size - 本步已花`，默认按显存分档（`arg_groups/memory_hook.py:92-170`）：<20GB 给 2048，<35GB / <60GB 给 4096，<90GB / <160GB 给 8192，≥160GB 给 16384。H100 上就是"块 8192、上限 16384"。

`budget_state()`（`849`）一查，任一 ≤0 就返回 `OTHER`，等待队列的循环在 `scheduler.py:4435` 直接 break。这是**两个数不是一回事**：上限管"这一步 kernel 的最大形状"，块大小管"单条长 prompt 一次能进来多少"。

### 4.2 第二重：请求数（这一步能收几条）

`get_num_allocatable_reqs`（`scheduler.py:4227`）取两者最小：`pp_max_micro_batch_size - running_bs`，以及 `req_to_token_pool` 的空闲行数（加上本次已持有的行，再扣掉 beam 成员预留）；行池本身的大小由 `max_running_requests` 决定。行数不够就 `batch_is_full=True`，本步不再准入（`4331-4337`）；循环内每收一条还会再查一次（`4454-4460`）。

### 4.3 第三重：KV（这些请求以后能不能活到结束）

三本账里唯一"往前看"的一本。`memory_budget` 的总偏移量由两部分组成（`schedule_policy.py:679-685` 与 `_update_prefill_budget:869`）：

$$
R = (A + E) - \sum_{r \in \mathrm{running}} \min(m_r - |o_r|,\ 4096)\cdot \rho
$$

$A$ 是 allocator 空闲量，$E$ 是 radix tree 可驱逐量，$m_r$ 是 `max_new_tokens`，$|o_r|$ 是已生成数，$\rho$ 是下一节的 `new_token_ratio`。准入判据（`mem_cache/prefill_budget.py:140-156` 的 `check_prefill`，`total_tokens` 在 `add_one_req:1410-1425` 组装）：

$$
\mathrm{admit} \iff \lceil e \rceil + m + p + s < R
$$

$e$ 是本块 extend 长度，$m$ 是截到 4096 的 `max_new_tokens`，$p$ 是补一页的对齐开销，$s$ 是 state slot / request slot 的折算。三个数合起来看：**token 决定这一步算多少，请求数决定这一步收几条，KV 决定这些请求以后会不会半路被 retract**。前两个是硬的（超了 kernel 直接崩），第三个是估的（估错了靠第 6 节兜底）。

### 4.4 chunked prefill 只切一条

长 prompt 的分块路径只有一条：`add_one_req`（`1399`）→ `_select_prefill_admission`（`1596`）。当 `extend_len` 超过 `chunk_tokens_limit` 时截成上限（`1650-1675`，按 `page_size` 向下取整），置 `is_chunked=True`、`max_new_tokens=0`（本块不采样），在 `_commit_prefill_admission`（`1689`）里记成 `new_chunked_req`。下一轮先 `init_next_round_input()`，再 `adder.add_chunked_req(self.chunked_req)`（`scheduler.py:4400` → `schedule_policy.py:1122`）接着切；直到某一块 `truncated=False`，`add_chunked_req` 返回 `None`，这个槽才被放掉。

关键在于：**同一时刻只有一条 `chunked_req`**，`scheduler.py:4555` 那句 `assert self.chunked_req is None` 就是守卫。其余长 prompt 只能排队等它切完——这是 SGLang 吞吐上可见的队头阻塞，也是它和 vLLM"任何请求都能停在半路"最根本的世界观差异。一条 32k prompt、块 8192：4 步吃完，TTFT 从"1 步"变成"4 步"，换来中间三步能插 decode。

## 5. new_token_ratio：给"未来"打折

`NewTokenRatioTracker`（`scheduler_components/new_token_ratio_tracker.py:14`）：初值 $\rho_{init} = \min(0.7c,\ 1.0)$（`21-33`，$c$ 即 `schedule_conservativeness`），下限 $\rho_{min}=0.14\,\rho_{init}\approx 0.098$，每跑一次 decode 掉一格、600 步到底（`decay_step`，`34-35`）：

$$
\rho_t = \max\left(\rho_{init} - t\cdot\frac{\rho_{init}-\rho_{min}}{600},\ \rho_{min}\right)
$$

算一笔账。64 条 running、每条 `max_new_tokens=4096`：刚起来时 $64 \times 4096 \times 0.7 = 183{,}500$ token 被预扣给"未来 decode"；600 步后只剩 $64 \times 4096 \times 0.098 \approx 25{,}690$。约 7 倍差。按 20 ms 一步算，600 步约 12 秒——**服务启动后十几秒内，账本上"给未来留的座位"会自己缩掉一大半，腾出来的空间全给 prefill**。

为什么要缩：`max_new_tokens` 是用户写的上限，不是真实需求；一直按满额预留等于拒绝承认"这些 token 大部分不会生成"，吞吐永远起不来。缩的代价是请求真跑长了 KV 不够——这时触发第 6 节的 retract。所以 $\rho$ 是"吞吐 vs retract 率"的旋钮，而 `schedule_conservativeness`（`arg_groups/fields/schedule.py:139`）就是给它乘系数，那个参数自己的 help 写得很直白：retract 频繁就调大。一个细节：**新准入的请求按满额预留**（`_update_prefill_budget` 传的是 `min(max_new_tokens, 4096)`，不打折），进了 running 之后才开始按 $\rho$ 打折。所以刚准入那一步 KV 账最紧，随后逐步松——这也是 SGLang 的准入是"一条一条挤进去"而不是"一把全收"的原因。

## 6. retract_decode：丢谁、丢多少、KV 去哪

触发点不在准入，而在 decode 之前（`scheduler.py:4780`）：

```python
kv_full_retract_flag = not batch.check_decode_mem()
```

`check_decode_mem`（`schedule_batch.py:2865`）算的是**下一步 decode 要多少 slot**：`new_tokens_required_next_decode`（`2836`）在 `page_size=1` 时每请求恰好 1 个 slot，判据退化成"池子里还有没有 `bs` 个空位"。也就是说：**retract 不是因为"未来不够"，而是因为"下一个 token 都写不进去"**。未来的账在第 4 节准入时就算过了，这里只是最后一道闸。

丢谁？`_get_decode_retraction_order`（`2985`）：

```python
def length_key(req):
    return (len(req.output_ids), -len(req.origin_input_ids))
sorted_indices.sort(key=..., reverse=True)   # 最想留的排前面
idx = sorted_indices.pop()                   # 从尾部丢
```

翻译成中文：**先丢"生成得最少"的；生成数相同，先丢 prompt 最长的**。两条理由：(a) 已生成 0 个 token 的请求用户还没看到任何输出，丢掉最不伤体验；(b) 同等资历下先丢 KV 占用最大的，一次释放最多空间。循环继续直到 `check_decode_mem` 通过，且**至少留一条**（`2895-2897`）；连最后一条都装不下就 abort 它，而不是让调度器崩（`2948-2973`）。

丢掉的 KV（`release_req:3025` → `mem_cache/common.py:449`）：`checkpoint=False`，**不进 radix tree**（注释在 `schedule_batch.py:1815`：这块内存要立刻用，别拿去做前缀缓存）；随后 `evict_from_tree_cache(tree_cache, 剩余请求数 × 20)`（`1816-1817`）顺手再腾一批；`reset_for_retract`（`1476`）把 `prefix_len`、`last_node`、`lock` 清空、`retraction_count += 1`，但 **`output_ids` 保留**；最后 `_add_request_to_queue(req, is_retracted=True)`（`scheduler.py:4850`）把它塞回**队尾**（`3663` 的 `append`，不是队头）。

下一轮是重算还是续跑？**重算**：`prefix_len=0`，整个 prompt + 已生成的 `output_ids` 一起重新 prefill，生成从断点接着走，用户看不出接缝（除非 input_embeds 场景，`1505-1512` 会把 `output_ids` 也丢掉从头生成）。这和 vLLM 的 recompute 抢占本质一样，差别只在"丢谁"和"KV 是否留给缓存"。

还有一处和 vLLM 不同：`retract_decode` 返回的新 $\rho$（`estimate_new_token_ratio_after_retract`，`2978`）是

$$
\rho' = \min\left(1,\ \frac{\sum |o_r| + 20\,n}{\sum m_r + 1}\right)
$$

64 条请求、各已生成 200、`max_new=4096` 时 $\rho' = (12800+1280)/262145 \approx 0.054$，比衰减到 0.098 还低。也就是说 **retract 不只释放 KV，还把"未来折扣"打得更狠**，下一轮准入更激进。这个正反馈是 retract 循环的温床，也是调参时要盯 `#new_token_ratio` 这行日志的原因。

## 7. 缓存感知排序：lpm 与它的代价

`--schedule-policy` 的默认值是 **`fcfs`**（`arg_groups/fields/schedule.py:86`），不是很多人以为的 `lpm`。缓存感知那一档（`CacheAwarePolicy:164`）里有 `lpm` / `dfs-weight` / `hrrn` / `shortest-prefill-first`，lpm 是这一档里最常用的那个。

lpm 的排序键就是前缀命中长度（`_sort_by_longest_prefix:443`）：按 `-num_matched_prefix_tokens` 升序，命中越长的越靠前。但命中长度不是白来的——`calc_priority`（`205`）先 `_compute_prefix_matches`（`348`）把整个 waiting_queue 走一遍，每条 req 拿 `origin_input_ids + output_ids` 去 radix tree match 一次，然后才排序。这一步的代价是「等待队列长度 × 树深」，而 decode 步每 20 ms 就来一次。

代价有硬上限：等待队列超过 128 条时 `_determine_active_policy`（`306`）直接把 lpm / hrrn 降级成 fcfs，因为"每步 O(128 × 树深)"已经开始吃 CPU（`WAITING_QUEUE_PREFIX_MATCH_MAX=128`，`129`）。`_compute_prefix_matches` 还兼职做 in-batch 前缀去重：短前缀（≤32 token，`102`）在模拟树里撞车的请求会被 `temporary_deprioritized`，本步只放一个进来共享同一份 KV。

所以 lpm 的收益是"同样的 KV 命中更多"，代价是"每步多一次全队列树查询"。前缀重复率高的负载（RAG、few-shot、多轮）值；随机长 prompt 不值。

## 8. process_batch_result：收尾的判定顺序

`process_batch_result`（`scheduler.py:5388`）按 `forward_mode` 分派：decode → `process_batch_result_decode`（`batch_result_processor.py:1045`），extend → `process_batch_result_prefill`（`317`），prebuilt / idle 各有分支。

decode 里单条请求的顺序（`1104-1155`）：`req.output_ids.extend(next_token_id)`（`1143`）→ `req.update_finish_state(new_accept_len)`（`1146`）→ finished 就 `release_kv_cache(..., checkpoint=True)`（`1558`，**这次进树**，前缀留给下一个请求）→ 没 finished 就什么都不动，KV 继续涨。

`update_finish_state`（`schedule_batch.py:1434`）的判定顺序有讲究，不能随便换：

| 顺序 | 判据 | 行号 |
|---|---|---|
| 1 | 已 finished → return（只判一次） | `1435` |
| 2 | `to_finish`（外部 abort、retract OOM） | `1438` |
| 3 | vocab 越界 / NaN | `1446` |
| 4 | stop string | `1453` |
| 5 | EOS / `stop_token_ids` | `1461` |
| 6 | `max_new_tokens` 长度 | `1465` |
| 7 | grammar 终止 | `1472` |

顺序错了就是 bug：长度先判会把"同一 step 里既撞 EOS 又超长"的输出截错；stop string 让位给 EOS 则会漏 stop 字符串进输出（`1450-1452` 的注释就是讲这个）。这里还发生两件事：`output_streamer.stream_output`（`1198`）把增量发给 detokenizer；`_on_step_processed`（`scheduler.py:4747`）把完成的请求从 `running_reqs` 摘掉。所以"发给用户"和"推进状态机"是同一步，但"释放 KV"和它们只隔一脚——overlap 下会晚一拍（`1119-1124`）。

## 与 vLLM 对照

| 维度 | vLLM V1（c7a7607） | SGLang（1ef3335） |
|---|---|---|
| 有没有阶段 | 没有。`num_computed_tokens` 追 `num_tokens_with_spec`（`vllm/v1/core/sched/scheduler.py:701-710`） | 有。`EXTEND` 与 `DECODE` 是两种 batch 形态（`schedule_batch.py:2285` / `3172`） |
| 一步跑谁 | running 先跑（`836-843` 用 `token_budget` 截断，`1001` 才真扣），waiting 用剩余预算 | 二选一：组得出 prefill 批就整步跑 prefill（`4193`），否则 decode |
| chunked prefill | 任何请求都能停在中途，由 `max_num_batched_tokens` 统一分配（`841`） | 只有一条 `chunked_req`（`4400`/`4555`），其余整段准入 |
| 预算单位 | 单一 token budget（H100 server 默认 8192，B200 16384） | token + 请求数 + KV 三重，且 decode 的 KV 先扣 |
| 装不下怎么办 | preempt：丢块、`num_computed_tokens=0`，`prepend` 回 waiting 队头（`1798`/`1841`） | retract：按 (输出长度, -prompt 长度) 从尾部丢到 `check_decode_mem` 通过，`append` 回队尾（`4850`） |
| KV 释放给谁 | 块还给 block pool | 还给 allocator，不进 radix tree（`1815`），再 evict 20×剩余请求数 |

## 常见坑

- **把 `--chunked-prefill-size` 设得很小**：SGLang 的块是"一条请求的连续段"，开到 512 会让长 prompt 的 TTFT 线性爆掉，ITL 收益却早已饱和。
- **以为默认是 lpm**：默认 `fcfs`。想靠前缀缓存提吞吐得显式 `--schedule-policy lpm`，并接受每步一次全队列树查询；等待超 128 条还会被自动降级。
- **把 `schedule_conservativeness` 当安全阈值**：它只乘 `new_token_ratio` 的初值，不改变 retract 的触发条件（那是 `check_decode_mem` 的硬判据）。
- **retract 频繁时先调 policy**：先看 `max_running_requests` / `max_total_tokens`。retract 的根因是"准入时估的 KV"和"真实生成长度"的偏差，换 policy 治不了。
- **`--enable-mixed-chunk` 不是免费的**：`mix_with_running`（`schedule_batch.py:2700`）会让 extend 批混进 decode 行，prefill kernel 形状变差，只在 prefill 很碎、decode 很多时才有意义。
- **忽略 `chunked_req` 的单槽特性**：一条 128k prompt 在切块期间，别的长 prompt 只能排队等它切完。

## 读代码时的检查点

1. **这一步是 prefill 还是 decode？** 看 `get_next_batch_to_run` 返回的 `batch.forward_mode`。`new_batch` 非 `None` 时 running 的 decode 这步一定没跑（`4193`）——排查 ITL 尖刺先确认这一点。
2. **预算被谁扣了？** token 在 `_update_prefill_budget`（`869`，`rem_input_tokens` / `rem_chunk_tokens`），KV 在 `memory_budget.reserve`（`prefill_budget.py:188-201`，`total_offset` / `current_offset`），请求数在 `get_num_allocatable_reqs`（`4227`）。三处都扣了才算扣干净。
3. **retract 之后 ratio 变成了多少？** 日志里 `#new_token_ratio: 0.7000 -> 0.0512`（`scheduler.py:4845`）是排查 retract 循环的入口：ratio 掉得越低，下一轮准入越激进。

## 小结

- SGLang 的一步是二选一：能组出 prefill 批就跑 prefill（算力优先），否则跑 decode；decode 的 KV 在准入时已被预扣（KV 优先）。两件事分开记，是它和 vLLM 最大的结构差异。
- `ScheduleBatch` 有两副面孔：EXTEND 带 `prefix_lens`/`extend_lens`、写 `extend_num_tokens` 个 slot；DECODE 每请求 1 token。
- 三重预算：`max_prefill_tokens`（16384）管算力、`req_to_token_pool` 行数管条数、`available + evictable - running 预扣` 管 KV。
- chunked prefill 只切一条 `chunked_req`，上限是 `chunked_prefill_size`（按显存 2k–16k），跨步持有到最后一个块。
- `new_token_ratio` 从 0.7 衰减到 0.098（600 步），把"未来 decode"的 KV 预留折价换吞吐；折过头就 retract，而 retract 又把 ratio 再打低。
- retract 丢"生成最少、prompt 最长"的那条，丢到下一步 decode 装得下为止；KV 立刻还、不进树，请求 append 回队尾重算。
- 默认 policy 是 `fcfs`；lpm 是 cache-aware 档的武器，代价是每步 O(等待数 × 树深)，超 128 条自动降级。

配套脚本 [`code/sglang/scheduler_sim.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang)：把「decode 优先 + 预算装箱」与两种 retract/preempt 的挑人规则跑成可对比的步数账（`--test` 自测）。

下一篇看 overlap：[CPU 调度怎么藏进 GPU 计算、`future_map` 怎么中继、CUDA Graph 怎么把 decode 包成一次 launch]({{< relref "sglang-05-overlap" >}})。
