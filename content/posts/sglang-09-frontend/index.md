---
title: "SGLang 研读（九）前端 DSL、Router 与 cache-aware 负载均衡"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 10
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "DSL", "Router", "负载均衡"]
categories: ["推理框架"]
---

做 best-of-3 的时候，一个很自然的写法是：把 8k 的上下文拼一次，然后分三个方向各续写一段，最后挑一个。写完之后你觉得「前缀只算一遍，三条尾巴各算各的」，于是心安理得。直到有人把它改写成三个 HTTP 客户端调用同一个 `/v1/chat/completions`——QPS 没变，prefill 却涨了三倍，显存里同时躺着三份一模一样的 8k KV。

这篇讲的就是这条缝隙：**SGLang 的「前端」不是花架子，它决定了同一份 KV 能不能被后面的请求接上。** 顺着它往下看，会碰到三个层次的复用：引擎内的 `n>1`、前端 DSL 的 `fork`、以及集群里 Rust router 的前缀感知选路。前置阅读：[第一篇的进程与请求路径]({{< relref "sglang-01-architecture" >}}) 和 [第二篇的 radix 树]({{< relref "sglang-02-radix-attention" >}})。

## 1. 两个入口，只有一个是「程序」

SGLang 服务端对外的生成入口分两类：一个自有的 `/generate`，和一堆 OpenAI 兼容端点（`/v1/chat/completions`、`/v1/completions`、`/v1/responses`…）。它们最后都汇到 `TokenizerManager.generate_request()`（`python/sglang/srt/managers/tokenizer_manager.py:934`），但表达力差很远。挑两条有代表性的：

| 入口 | 处理链 | 一次请求能表达什么 |
|---|---|---|
| `POST /generate` | `serve_generate_request`（`python/sglang/srt/entrypoints/http_server.py:935`） | 一段 `text` 或 `input_ids`，直接进引擎 |
| `POST /v1/chat/completions` | `openai_v1_chat_completions`（`http_server.py:1862`）→ `serving_chat` → 组装 `GenerateReqInput`（`python/sglang/srt/entrypoints/openai/serving_chat.py:1388`） | 一次对话轮，一次生成，一个 rid |

`/generate` 收的是**已经渲染好的字符串**，服务端不再套模板；chat 路径收的是 messages，服务端渲染、还要补上特殊 token。这不是风格差异：OpenAI 协议里没有「同一个前缀下有多个不同后继」这个表达位，一个 HTTP 请求就是一个生成结果。DSL 之所以能做出分支，是因为它走 `/generate`，并且**在前端就把每个分支的完整 prompt 拼好了**。

## 2. DSL 的运行时：不是 IR 解释器，是「表达式队列 + 后台线程」

先破一个常见误解。`python/sglang/lang/ir.py` 里确实有一整套节点（`SglGen` 451、`SglSelect` 533、`SglFork` 551、`SglConcateAndAppend` 601），但**运行时并不解释这棵树**：

- `@sgl.function` 只是把函数包一层（`ir.py:141`），调用时直接执行 Python 函数体：`state.ret_value = program.func(state, ...)`（`python/sglang/lang/interpreter.py:44`）。
- 函数体里每次 `s += something`，走到 `ProgramState.__iadd__`（`interpreter.py:1023`）→ `SglExpr.__add__` 造 `SglExprList`（`ir.py:350` 的 `concatenate_ir`）→ `submit` 进队列（`interpreter.py:342`）。
- 每个 `StreamExecutor` 背后一个后台线程（`interpreter.py:323`），线程串行消费队列（`_thread_worker_func` 422），遇到 `SglGen` 就一次性把当前 `text_` 当作 prompt 发出去（`_execute_gen` 593）。

所以「IR」在运行时更接近**日志/可视化结构**：`SglExpr.print_graph_dfs`（`ir.py:361`）能把程序打成 `%3 = %2 + Gen('x')` 这样的图。它唯一被真正构造出来的地方是 tracer（`python/sglang/lang/tracer.py:114`），而 `StreamExecutor._execute`（`interpreter.py:461`）的分派表里**根本没有 `SglFork` 这个分支**——运行时的 fork 走的是另一条路。

## 3. fork 的真身：复制文本 + 新开线程 + 多个在途请求

`StreamExecutor.fork()`（`interpreter.py:370`）做的事很土：为每个分支新建一个 `StreamExecutor`，把 `text_`、`variables`、`messages_` 拷过去，并记下 `fork_start_text_pos = len(self.text_)`（`interpreter.py:397`）。注意它**不合并请求**——三个分支就是三个线程、三个 `sid`、三次 `/generate`，服务端三个 rid。

```python
@sgl.function
def best_of_three(s, question):
    s += "请从三个角度回答：" + question
    forks = s.fork(3)
    forks += lambda i: f"\n角度{i}："
    forks += sgl.gen("ans", max_tokens=256)
    forks.join()
    s += "三个答案里最好的是：" + sgl.gen("pick", max_tokens=8)
```

每个 `+=` 都是一次入队；`fork` 那行之后，后面两行在三个线程里各跑一遍。一份典型写法到线上的完整映射：

| 用户写法 | 前端产物 | 服务端看到 |
|---|---|---|
| `s += "请从三个角度回答："` | `SglConstantText` 入队，`text_` 追加 | 无（纯本地字符串） |
| `s += sgl.gen("a", max_tokens=256)` | `SglGen`（`ir.py:451`，采样参数冻进 `SglSamplingParams`，`ir.py:18`） | 一个 rid，prompt = 当前 `text_` |
| `forks = s.fork(3)` | 3 个 `StreamExecutor`，共享 `fork_start_text_pos` | 三个 rid，prompt 都以同一段字节开头 |
| 每个分支 `f += sgl.gen(...)` | 各自队列、各自 HTTP | 三个请求同时在途，尾部各不相同 |
| `forks.join()` | `gather_variable`：把分支新变量拷回父状态（`interpreter.py:1052`） | 无（不发请求） |

省下来的账可以算。设共享前缀 $P$ 个 token、第 $i$ 个分支自己那截 $b$ 个 token，共 $k$ 个分支：

$$
W_{\text{fork}} = P + \sum_{i=1}^{k} b_{i}, \qquad
W_{\text{naive}} = k \cdot P + \sum_{i=1}^{k} b_{i}
$$

差值 $(k-1)P$ 就是省掉的 prefill。按 prefill FLOPs 约 $2NP$ 的粗估，8B 模型、8k 前缀、3 分支，一份前缀约 131 TFLOP，省两份 ≈ 0.6 s 的 GPU 时间——这不是小数。

## 4. 引擎侧：三个 rid 怎么落回同一棵树

前端只是把三个请求发出去，真正让它们共享 KV 的是引擎。链路很短：

1. 请求进调度器前，`match_kv_cache`（`python/sglang/srt/mem_cache/common.py:253`）拿 `origin_input_ids + output_ids` 去 `tree_cache.match_prefix`（`python/sglang/srt/mem_cache/radix_cache.py:369`）里找最长前缀，命中的终点写进 `req.last_node`。
2. 跑起来之后，每算完一段就 `checkpoint_kv_cache`（`common.py:303`）→ `UnifiedRadixCache.checkpoint`（`python/sglang/srt/mem_cache/unified_radix_cache.py:1286`）→ `checkpoint_into_tree`（`:1291`），把「已经算出来的 token」**在请求还没结束时就发布进树**。这是后两个分支能命中的前提。
3. 命中之后，请求对匹配路径加锁：`BasePrefixCache.lock`（`python/sglang/srt/mem_cache/base_prefix_cache.py:563`）返回一个 `TreeLock`（`:255`），存进 `req.lock`（`python/sglang/srt/managers/schedule_batch.py:646`）。`inc_lock_ref`（`radix_cache.py:587`）沿路径一路 `+1` 到根，把字节从 `evictable_size_` 挪进 `protected_size_`（unified cache 下同一语义由 `unified_radix_cache.py:1197` 转给 tree core）。

于是三条分支的 KV 关系在树上非常直白：**同一串祖先节点（共享前缀），三个不同的 `last_node`（三条尾巴）**；共享那段的 `lock_ref` 等于 3，谁都别想在这期间把它淘汰掉。第 2 篇讲过 `evict` 只看可淘汰字节，这里就是「为什么共享前缀不会被踢」的答案。

```text
root
 └─ [请从三个角度回答：…]        lock_ref=3，evictable→protected
     ├─ [\n角度0：][ans…]        分支 0 的 last_node
     ├─ [\n角度1：][ans…]        分支 1 的 last_node
     └─ [\n角度2：][ans…]        分支 2 的 last_node
```

三个请求是三次 `match_prefix`，落点分别在三条尾巴上；但它们往上走的路径是同一串节点，KV 只有一份。

还有一个容易被忽略的细节：如果三条分支**同批**被准入，它们会各算一遍前缀。调度策略 `SchedulePolicy` 的 `_compute_prefix_matches`（`python/sglang/srt/managers/schedule_policy.py:348`）会维护一棵只属于等待队列的 radix 树（`:388`），发现后来的请求和队列里已准入的请求共享前缀，就把它塞进 `temporary_deprioritized`，排序时给 `inf` 键（`_sort_by_longest_prefix`，`:443`）推到队尾。但**这是降优先级不是拒绝**：prefill 预算宽松时三条分支照样同批各算一遍。真正稳定的命中，来自「第一条先 checkpoint、后两条下一轮再进来」。默认阈值是 32 token（`:102`、`:109`），前缀短于它连这个检查都不做。

## 5. Router：把「最长前缀」从单机搬到集群

单机引擎知道自己的树，但不知道隔壁那台的树。于是 SGLang 在仓库里放了一个 Rust router（`sgl-router/`，配置面在 `sgl-router/src/config/`），把「谁缓存了什么」变成路由状态。

**状态从哪来。** 主力是工作节点主动上报的 KV 事件：`BlockStored` / `BlockRemoved` / `AllBlocksCleared` 被喂进一棵以 block hash 为键的基数树（`sgl-router/src/state/kv_events/tree.rs:1` 的模块注释写得很清楚，还解释了为什么要按根 hash 分片、为什么读路径不拿写锁）。也可以换成外部的 KV indexer（`--kv-indexer-endpoint`，`sgl-router/src/config/cli.rs:359`），由它做权威的缓存信号（`--cache-prefix-provider`，`:326`）。代价在这里：router 自己算 block hash 时必须和引擎**逐字节一致**——`state/kv_events/hash.rs:1` 开头就写明这是 `radix_cache` 的镜像实现，连 EAGLE 的 bigram（重叠 token 对）哈希都要对上，引擎侧对应 `python/sglang/srt/mem_cache/utils.py:91` 的 `get_hash_str`。差一个字节，就是静默的 0 命中。

**怎么打分。** 命中候选不是加权求和，而是**分层**：

1. `cache_candidate_proposal`（`sgl-router/src/policies/cache_aware.rs:28`）把 indexer/树的匹配结果转成候选，`passes_cache_gate`（`:117`）用绝对命中 token 数或命中比例做门槛；
2. 候选数被 `candidate_limit`（`:130`）压住（默认取健康 worker 的 5%，最少 8 最多 32）；
3. `compare_candidate_seed`（`:139`）排序：**先按命中前缀 token 数降序，再按负载**。

也就是说「最长前缀」是主键，「最小负载」是次键；压力是真的大到不行，才由压力守卫翻盘。

一次请求进 router，顺序是这样：

1. **tokenize**：router 自己拿一份 tokenizer 把 prompt 变成 id（`request_tokens_for`，`sgl-router/src/policies/mod.rs:52`），chat 请求要先渲染模板——这也是为什么 router 依赖 `dynamo-tokenizers` 和 `sglang-processor`；
2. **算 block hash**：用与引擎相同的分页 SHA256 链，把 token 变成一串 `i64`；
3. **查树**：`RadixTreePrefixProvider.match_request_tokens`（`sgl-router/src/state/kv_events/prefix_provider.rs:55`）拿到「每个 worker 命中了多少个 block」；
4. **选**：走上面的分层打分，然后把 `routed_dp_rank` 塞进转发的请求里。

这四步里，第 1、2 步最容易出错，也最没有报错提示。

可调的旋钮和它们真正在权衡的东西：

| 参数 | 默认 | 在权衡什么 |
|---|---|---|
| `--cache-affinity-min-matched-tokens` | 1024 | 命中的绝对值不值得为它牺牲均衡；太小会把请求吸到一台 |
| `--cache-affinity-min-match-ratio` | 未设 | 同上，但按比例——短 prompt 上更公平 |
| `--cache-switch-margin-tokens` | 1024 | 未缓存 token 差多少才允许「改嫁」到更空的机器 |
| `--worker-queue-limit` | 未设 | 排队到这个深度就放弃亲和；失败开放（数据不新则保持亲和） |
| `--cache-pending-prefix-ttl-ms` | 0（关） | 冷前缀在事件确认前先「记账」多久，让同一 burst 落在一起 |

前四个是「亲和 vs 均衡」的缩放系数，最后一个才是 fork 场景的关键：三条分支同时到达时，第一条的 prefill 还没产生 `BlockStored` 事件，树里什么都没有。默认 `0` 意味着**不记账**——三个请求可能被撒到三台机器上，各自 prefill 一遍。把它设成几百毫秒，router 才会在路由第一条之后就把这段前缀「押」在这个 worker 名下（pending stake，`prefix_provider.rs:70` 处决定要不要查它、要不要写它），后两条分支即使事件还没到也能跟着走。

> 老笔记里的 `--cache-threshold` 在这个基线上已经不存在了（`grep -rn cache_threshold sgl-router/` 零命中）。它被拆成了两类东西：一类是**绝对量门槛**（`cache-affinity-min-matched-tokens`、`cache-switch-margin-tokens`），一类是**比例门槛**（`cache-affinity-min-match-ratio`、`cache-candidate-ratio`）。理解这套参数的关键，是记住它们量的都是「token 数」，不是百分比。

选路的账可以写成一句话：

$$
T(w) \approx \frac{\left(1 - r(w)\right) P}{\beta} + \frac{Q(w)}{\mu}
$$

$r(w)$ 是 worker $w$ 上已缓存的命中率，$Q(w)$ 是它的队列深度。cache-aware 的实质是：**只在 $r(w)$ 明显更大时才接受更长的 $Q(w)$**，而「明显」就是 `cache_switch_margin_tokens`。

## 6. DP：rank 是第二层，而且必须由外面指定

一台机器上还有 `--dp-size` 个 rank，每个 rank 有自己的 radix 树。这层的均衡在 `DataParallelController` 里，只有四种**纯粹看负载**的方法：`ROUND_ROBIN`、`FOLLOW_BOOTSTRAP_ROOM`、`TOTAL_REQUESTS`、`TOTAL_TOKENS`（`python/sglang/srt/managers/data_parallel_controller.py:105`），预算来自共享内存快照（`DPBudget` `:122`，刷新 `:335`）。它从 tokenizer 收 `PULL`、往各 rank `PUSH`（`:182`、`:441`），和第 1 篇的结论一致。

但每种方法的**第一行**都是 `maybe_external_dp_rank_routing`（`:854`）：`req.routed_dp_rank` 有值就直接发给那个 rank，跳过自己的均衡。这就是两层的接缝——router 算好 rank，塞进请求里。具体是：

- router 侧先按 sticky key 哈希，否则取「缓存最深」的 rank，最后才比 in-flight（`sgl-router/src/policies/dp_rank.rs:16`，`:31`）；
- 结果同时写进 header `x-data-parallel-rank`（`sgl-router/src/server/routes/chat/forward.rs:37`、`:736`）和 body 的 `routed_dp_rank`（`:495`）；
- Python 端 header 优先于 body（`python/sglang/srt/entrypoints/openai/serving_base.py:281`，函数名就叫 `extract_routed_dp_rank_from_header`）。

为什么不让 DP controller 自己做前缀感知？因为它看不到别的 rank 的树——树在各自进程里，DP controller 只有 `num_running_reqs` / `num_total_tokens` 这种标量。**KV 感知只能在同时看得见所有树的那个进程里做**，这就是 router 存在的理由，也是它必须多一个组件、多一层状态的代价。

## 7. 复用的三级台阶（顺便回答「为什么必须是同一个设计」）

| 层级 | 谁在 fork | 前缀保证来自 | 能改采样参数吗 |
|---|---|---|---|
| `n>1`（同一请求） | 引擎（一个 rid 内多路采样） | 引擎自己 | 不能，n 路共用 |
| DSL `fork(k)` | 前端（k 个 rid） | `fork_start_text_pos` 精确切点 | 能，每个分支独立 |
| 手搓 k 个请求 | 客户端 | 全靠人肉保证 | 能 |

所以更准确的说法不是「拆成三次请求复用就白费」——只要 token 前缀逐字相同、且同时在途，radix 树照样命中。真正的问题是**拆开之后没有任何一层能保证这两件事**：

- **逐 token 相同**：chat 路径每条分支都可能带上不同的 system prompt 或不同的渲染顺序，第一个 token 就分叉，匹配直接掉到 0；而 DSL 的 fork 是从同一个 `text_` 切出来的，字节前缀关系是构造出来的，不是碰巧的。
- **同时在途**：手工串行发三个请求，第一个跑完、树被 checkpoint、第二个才来，这叫「顺序命中」，省的是 prefill 但要等；fork 的三个线程是并发的，省的是时间。
- **可检查**：`join(mode="gather_variable")` 至少会在变量层面把分支收回来；手搓的写法连「这三条是一个程序」都没人知道。

一句话：前端并行与后端复用之所以必须一起设计，是因为**「共享前缀」是个端到端不变量**，而前端的表达力决定了它能不能被构造、被断言。

## 8. 与 vLLM 对照

基线 `c7a7607`。

- **没有等价 DSL。** vLLM 对外就是 `LLM` 类（`vllm/entrypoints/llm.py:62`）和 OpenAI 兼容层；仓库里所有 `fork` 都是进程 fork 的语境（如 `vllm/v1/engine/llm_engine.py:113` 的注释），没有一个是「程序分支」。想在一个程序里表达「共享前缀、分头续写」，只能自己发多个请求，或者用 `SamplingParams.n` 做同前缀多采样。
- **DP 均衡只看三个标量。** `DPLBAsyncMPClient.get_core_engine_for_request`（`vllm/v1/engine/core_client.py:1798`）的打分是 `max(client_count * inflight, waiting + running)`，再加一条 KV 压力惩罚 `waiting * 6.0 * max(0, kv_cache_usage - 0.5)`。**没有 token 身份**——它不知道这个 prompt 在哪个 engine 上已经有一半 KV。
- **也能钉 rank，但没人算。** `request.data_parallel_rank` 一旦有值就直接用（同上函数开头），和 `routed_dp_rank` 同构；区别是 vLLM 仓库里没有东西负责算出这个值。
- **前缀感知路由在仓库外。** vLLM 侧要靠外部组件（llm-d 的 EPP、Gateway API Inference Extension 的 prefix-cache 打分插件）去看各 engine 的 KV 事件再做选路，vLLM 自己只暴露指标。
- **判断：** SGLang 把「复用」从单机推到了集群，代价是仓库里多一个 Rust 组件、多一层必须与引擎逐字节对齐的镜像状态（tokenizer、page hash、`cache_salt`、EAGLE 的 bigram），以及一个新的失效模式——router 与引擎版本不一致时，命中率归零而错误码是 200。vLLM 把这件事留给生态，好处是内核简单，坏处是端到端前缀复用没有单一负责人。

## 9. 常见坑

- **`join(mode="concate_and_append")` 是死路。** 客户端会去调 `/concate_and_append_request`（`python/sglang/lang/backend/runtime_endpoint.py:315`），但 `python/sglang/srt/` 里已经一处都不剩（`grep -rn concate_and_append --include=*.py` 在 `srt/` 下 0 命中，13 处全在 `lang/` 与 `test/test_programs.py:291`）。而且它传的 `src_rids` 是客户端的 `sid`（`interpreter.py:751`），服务端 rid 是另生成的（`python/sglang/srt/managers/io_struct.py:397`），两套 id 从来不是一回事。默认的 `gather_variable` 才是活的路径。
- **`--policy` 默认不是 cache_aware**，是 `round_robin`（`sgl-router/src/config/cli.rs:250`）。不显式打开，前缀感知等于没装。
- **`--cache-pending-prefix-ttl-ms` 默认 0**，冷前缀 burst 不会「先记账」，三个分支可能散开——这是 fork 在集群上最常见的失效方式。
- **前缀短于 32 token 时，in-batch 去重不生效**（`schedule_policy.py:102`），小 prompt 的分支会各算一遍，但绝对值小，通常不值得管。
- **tokenizer/模板版本不一致 = 静默 0 命中**。router 的 hash 是引擎的镜像实现，任何一边升级分词器，前缀就对不上，而且没有任何报错。

## 10. 读代码时的三个检查点

1. **这个「分支」到底产生几个 rid？** 看它是不是走到了 `StreamExecutor.fork`（`interpreter.py:370`）。如果是 `n>1`，rid 只有一个；如果是 DSL fork，rid 有 k 个，共享前缀靠树而不是靠请求对象。
2. **前缀什么时候被发布进树？** 找 `checkpoint_kv_cache`（`common.py:303`）的调用点。没有它，并发的第二个请求只能等，或者重复算。
3. **router 的树和引擎的树是不是同一套 hash？** 对 `state/kv_events/hash.rs` 和 `mem_cache/utils.py:91` 的算法，看 page size、`cache_salt`、bigram 三个开关是否一致。

## 小结

- DSL 的运行时不是 IR 解释器：`@sgl.function` 直接跑 Python 函数体，表达式进队列，后台线程串行消费；`SglFork` 只是 tracer 的产物。
- `fork(k)` = 复制文本 + k 个线程 + k 个 rid；`fork_start_text_pos`（`interpreter.py:397`）是精确的分叉坐标，也是 `join` 能拼回去的依据。
- 三个 rid 靠 `match_prefix` 落到同一串祖先节点上，`TreeLock` / `inc_lock_ref` 让共享段 `lock_ref = k`，谁都淘汰不掉。
- 同批准入时 in-batch 去重只是「降优先级」，预算宽松仍会重复 prefill；真正的命中来自「先 checkpoint、后进来」。
- Router 的分层打分是「前缀做主键、负载做次键、压力做翻盘」，状态来自 KV 事件构成的 radix 树，必须与引擎 hash 逐字节一致。
- DP 那层的四种方法全都只看标量负载，所以 KV 感知必须由外部算好 `routed_dp_rank` 塞进来；这是多一个组件、多一层状态的全部代价。

下一篇：[PD 分离与 HiCache 三级缓存]({{< relref "sglang-10-pd-disagg" >}})——当 prefill 和 decode 不在一个进程里，「前缀能不能接上」就变成了跨机的 KV 搬运问题。
