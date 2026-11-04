---
title: "SGLang 研读（一）进程与请求的一生"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 2
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "进程模型", "ZMQ", "请求生命周期"]
categories: ["推理框架"]
---

[导读篇]({{< relref "sglang-00-index" >}})讲的是「为什么是树」，那是数据结构的账。这一篇先把最土的问题钉死：**代码跑在哪几个进程里，一个请求怎么从 HTTP 走到 GPU 再走回来**。后面所有篇（KV 池、调度器、overlap、PD 分离）都长在这张进程图上，这张图画错了，后面每一篇都会读歪。

SGLang 的进程图简单到有点反直觉：三个进程、五条单向管道、没有 client/core 抽象层。但正因为简单，几个问题才有意思——为什么 `rid` 必须在 tokenize **之前**就登记进表？为什么 detokenizer 要单独占一个进程，而 vLLM 把它塞在前端进程里？为什么 DP 一来就凭空多出一个 controller 进程？

## 1. 先记住一个画面

`tp=1` 的 SGLang 服务，一条 `POST /generate` 进来的那一刻，场上只有三个进程：

```text
进程 A（main 进程）：uvicorn(uvloop) + TokenizerManager + 一份 tokenizer
  │  HTTP handler 与 generate_request 在同一个 event loop 上，是函数调用，不是 IPC
  │  tokenize → TokenizedGenerateReqInput；rid_to_state[rid] = ReqState(out_list, event)
  │
  │ PUSH ──scheduler_input_ipc_name──▶
  ▼
进程 B（scheduler，每个 TP rank 一个）：Scheduler + TpModelWorker + ModelRunner + GPU
  │  recv → get_next_batch_to_run → run_batch → process_batch_result → stream_output
  │  BatchTokenIDOutput
  │
  │ PUSH ──detokenizer_ipc_name──▶
  ▼
进程 C（detokenizer）：DetokenizerManager + 另一份只读 tokenizer
  │  增量 detokenize → BatchStrOutput
  │
  │ PUSH ──tokenizer_ipc_name──▶
  ▼
回到进程 A：handle_loop 收到 BatchStrOutput → state.out_list.append → state.event.set()
  → _stream_one_response 醒来 → 一行 data: {...}
```

三个箭头**全是单向**的。回程不是「函数返回」，而是另一条管道把结果推回来，落在 `rid_to_state` 的那张表上。这个「去路和回路不对称」的设计，是理解后面全部细节的钥匙。

这不是我画的示意图，是 `_launch_subprocesses` 的 docstring 原话（`python/sglang/srt/entrypoints/engine.py:1179`）：

> Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.

## 2. 谁在哪个执行体里，拿着什么对象

| 执行体 | 位置 | 入口 | 输入对象 | 输出对象 | ZMQ 角色 |
|---|---|---|---|---|---|
| FastAPI 路由 + TokenizerManager | main 进程，同一个 uvloop | `entrypoints/http_server.py:925`；`managers/tokenizer_manager.py:493` | `GenerateReqInput` | `TokenizedGenerateReqInput` / SSE 行 | 两个 PULL/PUSH **bind**（`:714`、`:718`） |
| Scheduler | 子进程，TP×PP 各一个 | `managers/scheduler.py:486` | `TokenizedGenerateReqInput` | `BatchTokenIDOutput` | 全部 **connect**（`ipc_channels.py:37`、`:63`） |
| TpModelWorker | 与 Scheduler **同进程** | `managers/tp_worker.py:279`，实例化在 `scheduler.py:1091` | `ScheduleBatch` → `ForwardBatch` | logits / KV 写入 | 无，纯 GPU |
| DetokenizerManager | 子进程 | `managers/detokenizer_manager.py:75` | `BatchTokenIDOutput` | `BatchStrOutput` | PULL bind（`:97`）+ PUSH connect（`:104`） |

两个要点：

- **tokenizer 不是一个线程**。`TokenizerManager` 和 FastAPI handler 共享同一个 uvloop：`auto_create_handle_loop()`（`tokenizer_manager.py:2599`）只是往当前 loop 上挂了一个 `handle_loop` task，不是新线程。所以「HTTP 线程 → tokenizer 线程」这次交接根本不存在，是同一线程里的函数调用。只有 `--tokenizer-worker-num > 1` 时才有独立进程。
- **model runner 不是一个进程**。`Scheduler.init_tp_model_worker()`（`scheduler.py:1076`）直接在调度进程里 `new` 一个 `TpModelWorker`，KV 池（`init_memory_pools`，`scheduler.py:1156`）也在同一个对象图里。这就是为什么 SGLang 的调度器可以在一行代码里同时摸到「队列长度」「KV 余量」「radix 命中」——它们不是跨进程状态。

## 3. 进门：先登记，再 tokenize

`generate_request`（`tokenizer_manager.py:934`）的顺序很讲究：

1. `obj.normalize_batch_and_arguments()`（`managers/io_struct.py:1337`）：把单条 / batch 摊平，`rid` 为空时补 `uuid4().hex`（`:1396`、`:1402`）。
2. `_init_req_state()`（`:3907`）：给每个 rid 建一个 `ReqState`，里面是 `out_list=[]` 加一个 `asyncio.Event()`（`:3951`），塞进 `rid_to_state`（`:746`）。重复 rid 当场抛错（`:3945`）。
3. 过 pause gate 和 `model_update_lock.reader_lock`（`:979` 起）。
4. `_tokenize_one_request()`（`:1139`）→ `_create_tokenized_object()`（`:1598`）。
5. `_send_one_request()`（`:1835`）：`wrap_pickle_fields()` 之后 `_dispatch_to_scheduler`（`:734`）→ `sock_send`。

**登记必须先于 dispatch**，因为 scheduler 的输出可能比「登记」先到。`_handle_batch_output` 是按 rid 查表（`:2654`），查不到就打一条 warning 丢包（`:2660`）：

> Received output for {rid=} but the state was deleted in TokenizerManager.

反过来说，`rid` 是这四个对象之间唯一的关联键，跨进程没有任何「连接」「会话」的概念。所以客户端自己传 `rid` 时必须自己保证唯一，服务端只做本地去重（`:3945`）。

顺便清一个旧笔记的雷：**`self.mu` 已经不存在了**。`grep -n "self\.mu" tokenizer_manager.py` 零命中。现在没有锁，共享状态的串行化完全靠「单 event loop + 每请求一个 `asyncio.Event`」。你如果照老文章去找那把锁，会怀疑自己读错了文件。

## 4. ZMQ 拓扑：五条管道，一个 ROUTER 都没有

数据面三个端点、辅助面两个：

| 端点（`PortArgs`） | bind 方 | connect 方 | 类型 | 传什么 |
|---|---|---|---|---|
| `tokenizer_ipc_name` | TokenizerManager（`:714`，PULL） | Detokenizer（`:104`，PUSH）；Scheduler（`ipc_channels.py:52`，PUSH） | PULL/PUSH | `BatchStrOutput`；`skip_tokenizer_init` 时是 `BatchTokenIDOutput` |
| `scheduler_input_ipc_name` | TokenizerManager（`:718`，PUSH） | Scheduler（`ipc_channels.py:37`，PULL）；DP Controller（`data_parallel_controller.py:182`，PULL） | PUSH/PULL | `TokenizedGenerateReqInput`、`AbortReq` |
| `detokenizer_ipc_name` | Detokenizer（`:97`，PULL） | Scheduler（`ipc_channels.py:63`，PUSH） | PUSH/PULL | `BatchTokenIDOutput` |
| `rpc_ipc_name` | Engine（`engine.py:389`，DEALER） | Scheduler（`ipc_channels.py:48`，DEALER） | DEALER/DEALER | `flush_cache`、权重更新等控制请求 |
| `metrics_ipc_name` | 外部消费者 | Scheduler（`ipc_channels.py:76`，PUSH） | PUSH | 指标 |

单节点时全是 `ipc://` 临时文件（`server_args.py:795-800` 里五个 `tempfile.NamedTemporaryFile`）；只有 DP attention 才切成 TCP 派生端口（`:805` 起）。

三个观察：

**（1）数据面全是 PUSH/PULL，没有 ROUTER。** 如果你带着 vLLM 的记忆去找 ROUTER/DEALER，会一无所获：`grep -rn "zmq.ROUTER" python/sglang/srt` 只命中 `disaggregation/kv_events.py:577`（KV 事件回放），不在请求路径上。DEALER 只出现在 `rpc_ipc_name` 那一对。含义很直接：**PUSH/PULL 没有寻址能力**，前端没法说「这条请求给 2 号 rank」。第 7 节的 DP controller 就是这句话的直接后果。

**（2）只有 TokenizerManager 是 bind 方。** scheduler 和 detokenizer 全是 connect。因为 ZMQ 的 connect 可以先于 bind 成功（内部自动重连），启动顺序就自由了：`_launch_subprocesses` 先起 scheduler（`engine.py:1256`），再起 detokenizer（`:1366`），最后才构造 TokenizerManager（`:1376`）——而 bind 恰恰在最后这个人身上。反过来如果让 scheduler bind，就得先起 tokenizer，冷启动时间会串起来。

**（3）为什么不直接用 `mp.Queue`。** 三个理由，都能在代码里找到证据：

- 跨进程本来就要序列化。SGLang 用 msgspec/msgpack（`io_struct.py:2791` 的 `_USE_PICKLE_IPC` 开关决定是否退回 pickle，`:2831`/`:2839`），消息体是带 tag 的 `msgspec.Struct`（`BaseBatchReq`，`:107`），**一条消息装下这一步的所有行**。这是「按步批量」而不是「按请求」的传输契约。
- 反压方向要由自己定。`config_socket` 把 PUSH 的 `SNDHWM` 设成 0、`SNDBUF` 在内存够大时给 0.5 GB（`utils/network.py:300-311`）。也就是 scheduler 往 detokenizer 推时**永不阻塞**：GPU 的节奏不能被 detokenizer 拖住，宁可堆在 socket 缓冲里。`mp.Queue` 有 feeder 线程和锁，反压语义不归你管。
- 收端要能「一次 drain 干净再算」。scheduler 用 `sock_recv(..., zmq.NOBLOCK)` 循环取到空为止（`scheduler_components/request_receiver.py:145-152`），跟调度步对齐；`sock_recv` 是阻塞语义的 `get_nowait` 做不到非阻塞批量。

## 5. 出门：两次跨进程，一次增量 decode

`BatchTokenIDOutput`（`io_struct.py:1693`）不是「一个请求一条消息」，而是「一步一条消息，按行排」。输出 id 用扁平数组 + 偏移（`:1701-1704`）：

```text
row i 的 ids = output_ids_flat[output_ids_offsets[i] : output_ids_offsets[i+1]]
```

这是 main 分支的优化：不再为每行建一个 `List[int]`。`detokenize_headers` 只在某请求的**第一行**输出时非空（`:1706-1708`），detokenizer 靠它建增量状态。

detokenizer 的 `event_loop`（`detokenizer_manager.py:158`）是一个裸的阻塞循环：`sock_recv` → `_request_dispatcher` → `sock_send`。真正的活在 `handle_batch_token_id_out`（`:400`）→ `_update_detokenize_states`（`:270`）。

增量 decode 的状态就三个 offset（`managers/incremental_detokenize.py:79-93`，注释是源码里原样的 ASCII）：

```text
        |--------- read_ids ---------|
        |-- surr_ids --|
 xxxxx  |xxxxxxxxxxxxx|xxxxxxxxxxxxx|
        ^             ^             ^
    surr_offset   read_offset   last token
```

- `surr_offset → read_offset`：上一轮已经 decode 出去的前缀（surrogate）；
- `read_offset → end`：本轮新增的 token。

$$
\text{new\_text} = \text{decode}\big(\text{ids}[\text{surr\_offset}:]\big)\Big[\ \big|\text{ids}[\text{surr\_offset}:\text{read\_offset}]\big|\ :\ \Big]
$$

说人话：把「旧前缀 + 新 token」整段 decode 一遍，再按**字符数**把旧前缀那段切掉。

为什么不能只 decode 新 token？因为 BPE 的边界跨 token。`" world"` 还是 `"world"`，取决于前一个 token 是不是词尾。只 decode 新 token 会得到错误的空格和错位的多字节字符。所以增量 decode 的代价是 O(已生成长度) 的重复 decode，换来的是**文本正确**。

`step()`（`:164`）里还有一段容易漏的 UTF-8 处理：如果新文本以 `�` 结尾（说明这个 token 把一个多字节字符劈开了），就只发出可打印前缀 `find_printable_text(new_text)`，**不推进** `surr_offset`，下一轮拿到更多 token 再重试。流式输出偶尔「卡一下再吐一大段」，常常是这里。

回到 tokenizer 侧：`handle_loop`（`tokenizer_manager.py:2624`）是同一个 loop 上的 task，`_handle_batch_output`（`:2639`）按行把结果塞进 `state.out_list` 再 `state.event.set()`。注意它不是逐行 set，而是攒够 `batch_notify_size`（默认 16，`arg_groups/fields/serving.py:266`）一起 set 再 `await asyncio.sleep(0)`（`:2950-2957`）。原因就在第 2 节：这个 loop 同时跑 HTTP handler，逐行唤醒会把 loop 打满。

等待侧 `_stream_one_response`（`:2095`）是 `await state.event.wait()`，带 4 秒超时（`SGLANG_REQUEST_STATE_WAIT_TIMEOUT`，`environ.py:1860`；用点 `:2109`）。超时后拿 `await request.is_disconnected()`（`:2116`）判客户端是否已经跑了，跑了就 `abort_request`。

## 6. 一个请求到底几次跨进程

单位是**消息**，不是 token：

| 场景 | 去程 | 回程 | 合计 |
|---|---|---|---|
| 普通单机 `tp=1` | 1（A→B） | 2（B→C→A） | 3 |
| 流式 `stream_interval=1` | 1 | 每步 2 | `1 + 2 × 步数` |
| `skip_tokenizer_init` | 1 | 1（B 直推 A，`ipc_channels.py:55-60`） | 2 |
| DP（`num_dp > 1`） | 2（A→controller→B，`data_parallel_controller.py:182`） | 2 | 4 |

流式那一行的单位是「步」：一条消息装这一步**所有**并发请求的所有行。所以 detokenizer 的固定开销是 O(步数)，与并发数无关；一个 512 token 的流式请求产生的是 512 条消息 × 2 跳，而不是 512 × 并发 × 2。

顺手算个量级账（假设值，非实测）：decode 20 ms/步、单条消息编码 + IPC 约 100 µs 量级，512 步就是 `512 × 2 × 100 µs ≈ 100 ms`，对 `512 × 20 ms ≈ 10 s` 的生成时间约 1%。**跨进程不是流式的瓶颈**；真正贵的是 `stream_interval=1` 带来的 512 次 SSE 分块和 512 次前端 JSON 反序列化，而那部分在客户端侧。想省 CPU 就调大 `--stream-interval`（`arg_groups/fields/serving.py:262`，默认 1），代价是流式的「打字感」变差。

## 7. 为什么没有 EngineCore 那一层

vLLM 的三层是被一个矛盾逼出来的：**前端必须 async（等 HTTP、等 tokenizer、等多模态），GPU 循环必须 sync（busy loop 不能有 await）**。于是切出 `EngineCore`（纯同步 busy loop，`vllm/v1/engine/core.py:1744`），外面套 `EngineCoreClient`（`core_client.py:84`）的三个实现——`InprocClient`（`:376`，同进程）、`SyncMPClient`（`:967`）、`AsyncMPClient`（`:1161`）——再外面才是 `AsyncLLM`（`async_llm.py:94`）。

SGLang 把这三层压成了一层：**scheduler 进程就是 engine**。它同时是：

- 唯一碰 GPU 的地方（`TpModelWorker` 在同进程，`scheduler.py:1091`）；
- 唯一做调度的地方（`event_loop_normal` `:2107` / `event_loop_overlap` `:2143`）；
- 唯一持有 KV 池和 RadixCache 的地方（`init_memory_pools` `:1156`）。

也就是说，「engine」这个抽象在 SGLang 里**没有被命名**，它就是 `Scheduler` 这个类加它的进程。好处是零搬运：调度决策需要的状态全在一个 Python 对象图里，加一个字段就能读到，不需要在 core 与 client 之间定义请求/输出的传输结构（vLLM 那边是 `EngineCoreRequest` / `EngineCoreOutputs`）。

代价有三条，都很具体：

**（1）失败模型是「全死」而不是「降级」。** scheduler 里任何未捕获异常都会 `parent_process.send_signal(signal.SIGQUIT)`（`scheduler.py:6651`）；主进程的 handler 打一条 `SIGQUIT received. ... It usually means one child failed.`（`tokenizer_manager.py:4183`），dump 完在飞请求就 `kill_process_tree(os.getpid())`（`:4191`）。**没有「这条请求失败、别的继续」的路径**，也没有把错误作为一条消息回传给前端的通道（对比 vLLM 的 `ENGINE_CORE_DEAD`，`core.py:1284`）。所以 SGLang 的进程隔离只解决「GPU 崩了别把 uvicorn 带崩到无法收尸」，不解决「单请求失败隔离」。

**（2）控制面得自己造第二条管道。** `flush_cache`、`update_weights_from_disk` 这类不走 generate 路径，于是 Engine 侧 bind 一个 DEALER（`engine.py:389`）、scheduler 侧 connect 一个（`ipc_channels.py:48`），在主循环里和 tokenizer 的消息一起 drain（`request_receiver.py:154-161`）。vLLM 里这些是 EngineCore 的 utility 调用，走同一条输入队列，不需要额外 socket。

**（3）DP 必须多一个进程。** 见下。

### DP 时插进来的那个进程

第 4 节说过：PUSH/PULL 没有寻址能力。SGLang 因此无法「由前端指定第 3 个 DP rank」。解法是在中间插一个 `DataParallelController` 进程（`engine.py:997` 起的就是它），它连在 `scheduler_input_ipc_name` 上（`data_parallel_controller.py:182`），按 `--load-balance-method`（round_robin / total_requests / total_tokens / follow_bootstrap_room）选一个 rank 再 PUSH 过去。

请求因此多了一跳，controller 本身也是单点。细节（DP attention、EPLB、elastic）留到[第 11 篇]({{< relref "sglang-11-distributed" >}})。

## 8. 与 vLLM 对照

基线 vLLM `c7a7607`。

| 维度 | vLLM | SGLang |
|---|---|---|
| 分层 | `AsyncLLM` → `EngineCoreClient` → `EngineCore`/`EngineCoreProc`（`async_llm.py:94` / `core_client.py:84` / `core.py:1281`） | HTTP+Tokenizer → Scheduler（含 model runner）→ Detokenizer |
| 数据面 socket | 前端 bind ROUTER(入) + PULL(出)（`core_client.py:692`、`:697`）；core connect DEALER + PUSH（`core.py:2084`、`:2194`） | 五条 PUSH/PULL，请求路径上无 ROUTER |
| 寻址 | DEALER identity = engine_index（`core.py:1308`），前端可指定 rank | 无寻址，DP 靠 controller 转发 |
| ZMQ IO 与 GPU 的关系 | 两个 IO 线程 + 进程内 `queue.Queue` 解耦；注释明说为了 overlap ZMQ IO with GPU（`core.py:1367-1371`） | 在调度主循环里 NOBLOCK drain（`request_receiver.py:149`），就地 `sock_send`（`output_streamer.py:351`） |
| detokenize 在哪 | 前端进程内的对象（`output_processor.py:307` 的 `IncrementalDetokenizer`） | 独立进程，多一跳 |
| TP 间请求分发 | 每个 core 自己从 DEALER 收 | 只有 `attn_tp_rank 0` 收 ZMQ，再 NCCL `broadcast_pyobj`（`request_receiver.py:226-236`） |
| 单点故障 | core 死 → `ENGINE_CORE_DEAD` 帧，client 可重建 | SIGQUIT 父进程 → `kill_process_tree` |

两条更本质的差别：

- **SGLang 少一层，换来的是「调度器能直接读一切」；vLLM 多一层，换来的是「core 可以被不同前端驱动」**（同进程、单进程、多个 API server 共享一组 core，`core_client.py:660-680` 的 `client_addresses` 分支）。前者让 SGLang 在算法层迭代极快——改调度策略不用改任何传输结构；后者让 vLLM 能做成可组合的组件，代价是每加一个能力都要在 `EngineCoreRequest`/`EngineCoreOutputs` 上加字段。
- **SGLang 把序列化放在调度线程里，vLLM 明确不移**。`output_streamer.py:351` 的 `sock_send` 就在主循环上，msgpack 编码占的是调度步的 CPU 时间；vLLM 那条注释（`core.py:1367-1371`）说的正是「让线程处理 socket↔队列，busy loop 只碰队列」。SGLang 用另一条路抵这笔账：`event_loop_overlap`（`scheduler.py:2143`）把 CPU 工作藏到 GPU 后面。

## 9. 常见坑

- **以为 tokenizer 是独立线程**。单 tokenizer 模式下它和 FastAPI 抢同一个 loop，所以任何在 `handle_loop` 里做的重活（大 batch 的 msgpack 解码、logprob 重排）都会直接抬高 HTTP 延迟。
- **以为 `skip_tokenizer_init` 会省掉 detokenizer 进程**。它照样被拉起（`engine.py:1366`；只有 `SGLANG_RUST_SERVER` 那条分支会提前返回，注释在 `:1333-1337`），只是 scheduler 改成直推 tokenizer（`ipc_channels.py:55-60`），那个进程就闲着。想省内存得自己确认。
- **重复 `rid` 会被本地拒**。`_init_req_state` 直接抛（`:3945`），而且此时请求还没 dispatch，客户端看到的是 400 类错误而不是超时——别去 scheduler 侧找原因。
- **detokenizer 的状态表是有容量上限的 LRU**。`DETOKENIZER_MAX_STATES = 1 << 16`（`environ.py:721`），超了会淘汰最老的；被淘汰的 rid 再来输出就会打 `Dropped the text of an output for {rid=}`（`detokenizer_manager.py:284` 附近）。超长流式 + 极高并发时会看到「文本丢了但 token 数对」。
- **`rid_to_state` 的泄漏点在「校验失败」路径**。所以 `generate_request` 里有个 `except BaseException` 兜底（`:1001`），只为把没发出去的 state 删掉。自己写子类或插件时别绕过它。
- **`stream_interval=1` 不是免费的**。每步一次 SSE 分块 + 一次前端反序列化，高并发下 CPU 会先于 GPU 饱和。

## 10. 读代码时的三个检查点

1. **`ReqState` 是什么时候进的表、什么时候出的表？** 进在 `_init_req_state`（dispatch 之前），出在 `_handle_batch_output` 的 finished 分支（`:2945`）或 abort 路径（`_handle_abort_finish_reason`，`:2056`）。进出不平衡就是内存泄漏或「输出被丢」的那条 warning。
2. **这条消息是「一步一条」还是「一请求一条」？** 看 `BatchTokenIDOutput` 的构造点（`output_streamer.py`），它一批一行地攒；如果你按「一条消息 = 一个请求」去读，`output_ids_offsets` 会看不懂。
3. **谁 bind、谁 connect？** 记一句就够：**数据面上 TokenizerManager 是服务端，scheduler 全是客户端**。看到某个 `get_zmq_socket(..., bind=False)` 就想「它在连谁」，第三列自然对得上。

## 小结

- 一条请求只经过三个进程：main（HTTP + tokenize）→ scheduler（调度 + GPU）→ detokenizer（增量 decode）→ 回到 main。回程走另一条管道，不是返回值。
- 数据面三条 PUSH/PULL 管道 + 一条 DEALER RPC，报文面上没有一个 ROUTER；因此没有寻址能力，DP 才需要额外的 controller 进程。
- 只有 TokenizerManager 是 bind 方，所以进程启动顺序自由：先 scheduler、再 detokenizer、最后 tokenizer。
- `rid` 是四个对象之间唯一的键，且必须在 dispatch 之前登记；`self.mu` 已经不存在，串行化靠单 loop + 每请求 Event。
- 增量 detokenize 用三个 offset（`surr_offset` / `read_offset` / end）做「整段重 decode 再切前缀」，代价 O(长度)，换来跨 token 边界的文本正确。
- SGLang 没有 EngineCore 这一层：scheduler 就是 engine。换来零搬运和算法迭代速度，代价是全死式失败、控制面要另开管道、DP 多一跳。

下一篇回到数据结构的账本：[RadixAttention 的基数树到底怎么分裂、怎么锁、怎么淘汰]({{< relref "sglang-02-radix-attention" >}})。
