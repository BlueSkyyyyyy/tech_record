---
title: "vLLM 研读（四）调度器：token budget、chunked prefill 与抢占"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 5
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "调度器", "Chunked Prefill", "Continuous Batching"]
categories: ["推理框架"]
---

前三篇讲的是"资源长什么样"：请求怎么进来、KV 怎么分块、块怎么复用。这一篇讲"每一步怎么花这些资源"——调度器。

调度器的输入是一堆状态各异的请求（有的刚来、有的算了一半、有的在 decode、有的被抢占过），输出是一个 `SchedulerOutput`：这一步哪些请求各跑几个 token。它要在三个约束下做决策：

- **token 预算**：一步最多算 `max_num_batched_tokens` 个 token，不能超；
- **KV 块**：每个请求要推进 token，就得有块；没有就得抢别人的；
- **延迟目标**：长 prefill 不能把正在 decode 的请求饿死，否则 ITL 会出现尖刺。

三个约束互相耦合，所以调度器没有"最优解"，只有"当前负载下的折中"。这一篇把这套折中拆开。

## 1. 先记住一个画面

把一步想象成一个容量 8,192 token 的**传送带**：

```text
|---- 已 decode 的 100 个请求，各 1 token = 100 ----|
|---- 剩下的 8,092 个位置，分给 prefill / 新请求 ----|
```

先上来的是 running 队列：每个已经在跑的请求至少要 1 个 token（decode），否则它就停滞了。然后才轮到 waiting 队列里的新请求或 prefill 块。如果某个 prefill 太大，就把它的 8,092 拆成若干块，这一步只算一块（chunked prefill）；如果连一块都放不下，就跳过它。

如果所有请求的 KV 加起来超过块池，调度器要做一件更痛的事：**抢占**——挑一个 running 请求，把它已经算好的 KV 全部丢掉，放回等待队列，之后从头重算。

## 2. 没有 prefill/decode 阶段，只有"追平"

V1 调度器的第一句注释就否定了传统认知（`vllm/v1/core/sched/scheduler.py:698`）：

> There's no "decoding phase" nor "prefill phase" in the scheduler.

每个请求只有两个数：

- `num_computed_tokens`：已经算过 KV 的 token 数；
- `num_tokens_with_spec`：prompt + 已生成 + 投机 token 的总数。

每一步调度器做的事，就是给请求分配若干个 token，让 `num_computed_tokens` 去追 `num_tokens_with_spec`：

$$
\text{num\_new\_tokens} = \text{num\_tokens\_with\_spec} + \text{placeholders} - \text{num\_computed\_tokens}
$$

（`placeholders` 是异步调度的占位 token，见第 6 节。）

这个统一视角把四件事变成了同一个动作：

| 场景 | 表现 |
|---|---|
| prefill | `num_computed_tokens` 从 0 开始追 prompt 长度 |
| chunked prefill | 一次只追一部分，`is_prefill_chunk=True` |
| decode | 每次追 1 个（或 1+投机数） |
| prefix cache 命中 | 起点直接跳过命中部分，追得更少 |

`is_prefill_chunk` 的判断在 `_update_after_schedule()`（`scheduler.py:1844`）：调度完把 `num_computed_tokens` 加上本步分配数，如果还小于 `num_tokens + placeholders`，就说明这个请求还在 prefill 中。

## 3. token budget 怎么分

`schedule()`（`scheduler.py:698`）的结构是两段循环：

### 3.1 先跑 running

对每个 running 请求：

```python
num_new_tokens = (request.num_tokens_with_spec
                  + request.num_output_placeholders
                  - request.num_computed_tokens)          # 还差多少
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold          # 长 prefill 封顶
num_new_tokens = min(num_new_tokens, token_budget)         # 预算约束
num_new_tokens = min(num_new_tokens,
                     max_model_len - num_computed_tokens - num_sampled_tokens_per_step)
```

（`scheduler.py:838` 起。）

然后尝试 `allocate_slots()`。成功就扣预算、继续下一个；失败就进入抢占（第 5 节）。

注意两个细节：

- **`long_prefill_token_threshold` 只在"有别人在等"时生效**。代码里显式判断 `num_eligible_reqs > 1`（`scheduler.py:755` 附近）：如果整个引擎只有这一个请求，就不该切块——没有别人会被饿死，切块只会让 TTFT 变差。
- **自适应下限**：开了 `long_prefill_token_threshold_adaptive` 时，阈值不会低于 `max_num_batched_tokens / 请求数`，避免把请求切得比公平份额还碎。

### 3.2 再跑 waiting

running 跑完（或预算耗尽）后，从 waiting 队列取请求：

1. 先做前缀缓存查询（`get_computed_blocks`），命中的块直接算进 `num_computed_tokens`；
2. 用剩下的预算给这个请求分配一个 chunk（可能是整个 prompt，也可能是一块）；
3. 如果 KV 不够、或者预算不够，就跳过（放回 `deferred_waiting`），继续看下一个。

waiting 队列的排序由 `--scheduling-policy` 决定：

| 策略 | 数据结构 | 顺序 |
|---|---|---|
| `fcfs`（默认） | `deque` | 先到先服务 |
| `priority` | 小顶堆 | `(priority, arrival_time)`，priority 小者先 |

优先级策略的抢占也相应改变：FCFS 抢占 `running[-1]`（最后进来的），priority 抢占 `max(running, key=(priority, arrival_time))`（优先级最低的）——`scheduler.py:937`。

### 3.3 准入控制

不是所有 waiting 请求都能进 running。除了 token 预算和 KV，还有几道闸：

- `max_num_active_reqs`（`scheduler.py:133`）：活跃请求数上限，默认等于 `max_num_seqs`。它和 `max_num_running_reqs` 分开，是为了给"流式输入会话"之类占着 model runner 槽位但不在 running 的请求留空间。
- `scheduler_reserve_full_isl`（默认 True）：准入时检查**整个输入序列**能否放进 KV，而不只是第一个 chunk。开着 chunked prefill 时，如果不做这个检查，会先放进来一堆长请求，每个只算了一块就发现 KV 不够，然后互相抢占——这就是"KV 抖动"。
- `watermark`（`vllm/config/scheduler.py:196`，默认 0）：准入时保留一定比例的块不分配，避免刚进来就触发淘汰。生产环境在内存紧张时值得设成 0.01–0.05。

## 4. chunked prefill：用 TTFT 换 ITL

假设一个 8k prompt、预算 8,192、当前没有别的请求：

- 不分块：一步吃完 8k，prefill 时间约 300 ms（8B 模型、2·N·P ≈ 131 TFLOP、有效算力 400 TFLOPS 量级）。这 300 ms 里，所有 decode 请求都停下——如果此时有 50 个请求在 decode，它们的 ITL 就从 20 ms 变成 320 ms。
- 分块（比如每块 2,048）：4 步吃完，每步约 75 ms。decode 请求每步都能插进来，ITL 的尖刺从 300 ms 降到 75 ms，代价是 TTFT 多了几步调度开销（通常可忽略）。

| chunk 大小 | 每步 prefill 时间（8B，量级） | 长 prompt 的 TTFT | decode 的 ITL 尖刺 |
|---|---|---|---|
| 8,192（不分块） | ~300 ms | 最好 | ~300 ms |
| 2,048 | ~75 ms | +几步 | ~75 ms |
| 512 | ~20 ms | 明显变差 | ~20 ms |
| 128 | ~5 ms | 很差 | 接近纯 decode |

默认情况下 `long_prefill_token_threshold=0`，不额外封顶，chunk 大小实际由**剩余预算**决定（通常接近 `max_num_batched_tokens`，按卡型 2k–16k 不等）；真正的切块来自预算不足、`long_prefill_token_threshold` 显式设置，或者 hybrid 模型的块对齐要求。

2026 年还多了两个节流开关，用来进一步稳定 ITL：

- `prefill_schedule_interval`（`vllm/config/scheduler.py:203`）：每 N 步才准入一次 prefill，中间的步只跑 decode。代价是长 prompt 的 TTFT 最多变成 N 倍。
- `--enable-dbo` 之外的 DP 预填充均衡（`throttle_prefills`）：多 DP rank 时让各 rank 的 prefill 错峰。

## 5. KV 不够：抢占

当 `allocate_slots()` 返回 `None`，调度器进入抢占循环（`scheduler.py:912` 起）：

```python
while True:
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens, ...)
    if new_blocks is not None:
        break
    preempted_req = self.running[-1] if self.policy == FCFS else max(...)
    self._preempt_request(preempted_req, scheduled_timestamp)
```

`_preempt_request()`（`scheduler.py:1798`）做四件事：

1. 释放该请求的所有块（`_free_request_blocks`）；
2. `num_computed_tokens = 0`，状态改成 `PREEMPTED`；
3. `num_preemptions += 1`，记一条事件；
4. 把请求**插回 waiting 队列头部**（`prepend_request`），下次优先重算。

**V1 没有 swap。** 被抢占的 KV 直接丢弃，恢复时从头重算。这是和操作系统分页最大的区别，也是为什么抢占的代价很高：

- 一个 8k prompt 被抢占一次，重算要再花约 300 ms；
- 如果 KV 长期不足，会出现"算一点 → 被抢 → 再算 → 再被抢"的活锁，吞吐断崖式下跌。

所以调度的第一原则不是"尽量多接请求"，而是**别让 KV 进入持续抢占区**。手段就是前面说的 `scheduler_reserve_full_isl`、`watermark`、以及把 `max_num_seqs` 设在 KV 池能长期支撑的范围内。

## 6. 异步调度：把 CPU 藏到 GPU 后面

默认开启的 async scheduling（`AsyncScheduler`，`vllm/v1/core/sched/async_scheduler.py`）做了一件很反直觉的事：**在第 N 步的 GPU 还没算完时，就把第 N+1 步调度好**。

问题在于：第 N 步的输出 token 还没回来，第 N+1 步怎么知道要算什么？答案是 **placeholder**：

- 调度时先假设每个 decode 请求会产生 `num_sampled_tokens_per_step` 个 token（通常 1，投机解码时是 1+spec 数），把 `num_output_placeholders` 加上去；
- 用占位符先把这一步的输入准备好；
- 等第 N 步的输出回来，`update_from_output()` 用真实 token 替换占位符，并把多算/少算的部分修正回来。

`AsyncScheduler._update_after_schedule()`（`async_scheduler.py:18` 起）里能看到具体做法：给每个非 prefill 请求加 placeholder，给投机 token 塞 `-1` 占位。

好处是把调度、输入准备、采样元数据更新这些 CPU 工作完全藏进 GPU 计算时间里；代价是调度器里的所有计数都要处理"在途但未确认"的状态。这也是为什么你会在调度器里看到 `num_stale_output_tokens`、`num_in_flight_tokens`、`drop_stale_output` 这些字段——它们都是为了在异步下保持状态机正确。

对 PP（流水线并行）还有一个额外约束：`decode_stagger = pp_size`，同一个请求两次 decode 之间至少隔 `pp_size` 步，以匹配 worker 侧 sampled token 的 ring buffer 节奏（`async_scheduler.py:14`）。

## 7. 常见坑

- **`max_num_batched_tokens` 太小**：长 prompt 被切成很多块，TTFT 线性变差；太小还会让 decode 和 prefill 抢预算，吞吐和延迟一起下降。
- **`max_num_seqs` 与 KV 池不匹配**：请求数上限设得比 KV 能支撑的更大，调度器会不断准入再抢占，`num_preemptions` 飙升而 goodput 下降。
- **优先级反转**：priority 策略只影响准入和 victim 选择，不能让已经在跑的低优先级长请求"降级"；它占着 KV 时，高优先级请求仍可能被堵住。
- **`scheduler_reserve_full_isl` 与流式输入冲突**：会话式请求的输入会持续增长，整段 ISL 检查可能过度保守；这类负载要单独评估。
- **watermark 只影响准入**：它保留的是"未来"的余量，已经 running 的请求该抢占还是会抢占。
- **PP>1 的 decode 间隔**：`decode_stagger = pp_size`，同一个请求两次 decode 之间至少隔 pp 步，ITL 的下限被流水线深度抬高。

## 8. 读代码时的三个检查点

1. **`token_budget` 在哪被扣？** running 循环扣一次，waiting 循环扣一次；如果看到预算没扣干净，检查是不是在 `continue` 分支里漏了。
2. **`allocate_slots` 失败后发生了什么？** 一定是抢占或跳过，不会"部分分配"。部分分配会让块表和 `num_computed_tokens` 不一致。
3. **`num_computed_tokens` 何时推进？** 只在 `_update_after_schedule()` 里统一推进，且是在 `SchedulerOutput` 生成之后。这样 worker 拿到的 `positions` 还是"本步开始前"的值，不会错位。

## 小结

- V1 没有 prefill/decode 阶段，只有 `num_computed_tokens` 追 `num_tokens_with_spec`；chunked prefill 和投机解码都是这个模型的特例。
- 预算分配顺序：running 优先（保 decode 不停滞），waiting 用剩余预算；单请求时不做长 prefill 切块。
- 准入有三道闸：`max_num_active_reqs`、整段 ISL 预留（`scheduler_reserve_full_isl`）、`watermark`。
- 抢占是"丢 KV + 从头重算"，不是 swap；victim 在 FCFS 下是最后进来的请求，在 priority 下是优先级最低的。
- chunked prefill 是 TTFT 与 ITL 的交易：块越小 ITL 越稳，但长 prompt 的 TTFT 越差。
- async scheduling 用 placeholder 把 CPU 调度藏进 GPU 时间，代价是状态机里多了"在途未确认"的计数。
- 调度器的第一目标不是"接更多请求"，而是"别让 KV 进入持续抢占"。

下一篇下到执行层：[这些调度结果怎么变成 GPU 上的 token ids、positions、slot_mapping，以及 CUDA Graph 怎么把它们包起来]({{< relref "vllm-05-execution" >}})。

> 官方文档：<https://docs.vllm.ai/en/latest/configuration/optimization/>（调度相关参数）、<https://docs.vllm.ai/en/latest/usage/v1_guide/>（chunked prefill 默认行为）
