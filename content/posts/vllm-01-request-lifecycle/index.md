---
title: "vLLM 研读（一）请求的一生：从 HTTP JSON 到 token"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 2
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "EngineCore", "ZMQ", "请求生命周期"]
categories: ["推理框架"]
---

调 vLLM 的时候，最常见的困惑是"我明明只改了一个参数，为什么延迟变了"。这类问题往往不在 kernel 里，而在请求经过的那条链路上：tokenize 在哪个进程做、请求怎么跨进程、调度器什么时候看见它、输出又是怎么流回客户端的。链路上任何一段排队，都会变成你看到的 TTFT。

这一篇把链路走一遍。读完你应该能回答三个问题：

- 一个 `/v1/chat/completions` 请求，在 vLLM 内部被拆成了哪几种对象？
- 为什么 V1 要拆成 API server / EngineCore / Worker 三层进程？各层的边界在哪？
- 流式输出的每个 token，从采样完成到客户端收到，中间经过了什么？

## 1. 先看进程和链路

V1 的进程模型（官方架构文档里有一张完整的表）：

| 进程 | 数量 | 职责 | 代码入口 |
|---|---|---|---|
| API Server | 默认 = DP，可 `--api-server-count` 覆盖 | HTTP、chat template、tokenize、多模态预处理、流式回包 | `vllm/entrypoints/openai/api_server.py` |
| EngineCore | 每个 DP rank 一个 | 调度、KV 管理、驱动 worker | `vllm/v1/engine/core.py` |
| GPU Worker | 每卡一个 | 权重加载、forward、显存管理 | `vllm/v1/worker/gpu_worker.py` |
| DP Coordinator | DP>1 时 1 个 | DP 间负载均衡、MoE 同步 | `vllm/v1/engine/coordinator.py` |

四卡 `-tp=4` 是 6 个进程，八卡 `-tp=2 -dp=4` 是 17 个。这个数字不是八卦：每个 API server 进程有自己的 tokenizer 线程池（`renderer_num_workers`）和多模态处理线程，容器 CPU 配额不够时，TTFT 会先在 tokenize 这一段炸掉。

为什么不让一个进程全干了？三个原因：

1. **Python GIL**：EngineCore 是一个不停转的 busy loop，如果和 HTTP 服务同进程，tokenize 和 JSON 序列化会直接插进调度循环，把每一步的 CPU 时间撑大。
2. **故障隔离**：API server 崩了不该带走正在跑的请求；EngineCore 崩了，前端能明确地给所有在途请求回 500，而不是挂死。
3. **DP 扩展**：一个 API server 可以连多个 EngineCore（多对多），DP rank 增加时不用复制前端。

## 2. 前端：Renderer 与 InputProcessor

请求进 API server 后，第一站是 OpenAI 协议层（`vllm/entrypoints/openai/serving_chat.py` 等）。这一层只做协议解析：把 messages 转成模型输入、校验参数、决定用 chat 还是 completion 模板。

真正把"文本"变成"token"的是 **Renderer**（`vllm/renderers/base.py:106`）。这是 2026 年的一次重构：以前 chat template、tokenize、多模态处理散在 serving 层，现在统一收口到 Renderer：

- `render_chat()` / `render_cmpl()` 负责套模板、切多模态占位符、tokenize；
- tokenizer 调用被丢进线程池（`_tokenize_prompt_async`），因为 HuggingFace fast tokenizer 是 CPU 密集的，不能阻塞事件循环；
- 多模态预处理有独立的单 worker 线程池，避免和 tokenize 抢锁；图片/音频的 embedding 结果按内容哈希缓存。

Renderer 的产物是一个 `EngineInput`，里面已经是 token ids（或 prompt embeds）了。接下来 `InputProcessor.process_inputs()`（`vllm/v1/engine/input_processor.py:382`）做的是**校验和组装**：

- 校验 `SamplingParams`（top_p、temperature、logprobs 的组合是否合法）；
- 校验 LoRA、多模态输入长度、prompt + max_tokens 是否超过 `max_model_len`；
- 把 prompt、采样参数、到达时间、优先级、KV 传输参数打包成 `EngineCoreRequest`（`vllm/v1/engine/__init__.py:156`）。

`EngineCoreRequest` 是一个 `msgspec.Struct`，用 `array_like=True` 序列化成数组而不是字典——这是为了跨进程传输时少几个字节和几次字典查找。它包含的字段基本就是"调度器需要知道的一切"：token ids、mm_features、sampling_params、arrival_time、priority、cache_salt、data_parallel_rank。

> 一个容易忽略的点：`prompt_logprobs` 和前缀缓存是冲突的。前者要求重新计算 prompt 的 logits，后者会跳过已缓存的块，所以 `InputProcessor` 在检测到 `prompt_logprobs` 时会关掉这个请求的 prefix cache 查询（`vllm/v1/core/kv_cache_manager.py:293` 附近的注释）。这类"参数组合导致优化失效"的规则，在 vLLM 里到处都是。

## 3. 跨进程：ZMQ 与 msgpack

`EngineCoreRequest` 要送到另一个进程。V1 用 ZMQ，拓扑是**前端 bind、引擎 connect**：

- 前端 bind 一个 `ROUTER` 收请求、一个 `PULL` 收输出（`vllm/v1/engine/core_client.py:668` 起）；
- 每个 EngineCore 用 `DEALER` 连前端的 ROUTER，用 `PUSH` 连前端的 PULL（`vllm/v1/engine/core.py:2083`、`:2194`）。

这个方向是有意的：bind 方是"服务端"，进程重启后地址不变，EngineCore 可以重连；DEALER 的 identity 让前端能把请求定向到某个 EngineCore（多 DP rank 场景），而输出的回程路由靠 `EngineCoreOutputs` 里的 `client_index`，不是靠 socket identity。

消息本身用 msgspec 的 MessagePack 编码（`vllm/v1/serial_utils.py`）。控制消息很短，第一个字节就是请求类型（`EngineCoreRequestType`，`vllm/v1/engine/__init__.py:352`）：

| 类型 | 字节 | 用途 |
|---|---|---|
| `ADD` | `0x00` | 新请求 |
| `ABORT` | `0x01` | 取消请求 |
| `START_DP_WAVE` | `0x02` | DP 同步前向的 wave 开始 |
| `UTILITY` | `0x03` | 重置缓存、sleep/wake 等 |
| `EXECUTOR_FAILED` | `0x04` | 执行器故障哨兵 |
| `WAKEUP` | `0x05` | 空闲时唤醒 busy loop |

大对象（多模态张量、prompt embeds）不走 msgpack 拷贝，而是走共享内存旁路（`tensor_ipc.py`）：前端把张量放进共享内存，消息里只传句柄。否则一个 4K 图片的 embedding 会把 ZMQ 带宽打满。

## 4. EngineCore：一个不会停的 while 循环

EngineCore 的主循环只有三行（`vllm/v1/engine/core.py:1744`）：

```python
while self._handle_shutdown():
    self._process_input_queue()   # 没活就阻塞等消息
    self._maybe_publish_request_counts()
    self._process_engine_step()   # 有活就 schedule + execute
```

`_process_engine_step()` 调 `step()`（`:709`），后者是 vLLM 的心跳：

```python
scheduler_output = self.scheduler.schedule(...)
future = self.model_executor.execute_model(scheduler_output, non_block=True)
grammar_output = self.scheduler.get_grammar_bitmask(scheduler_output)
model_output = future.result()
if model_output is None:
    model_output = self.model_executor.sample_tokens(grammar_output)
engine_core_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
```

注意两个细节：

- `execute_model(..., non_block=True)` 先拿到 future，再在 `future.result()` 处等。中间这段空隙用来算 grammar bitmask（结构化输出的 token 掩码）——CPU 和 GPU 在这里重叠了一次。
- `model_output is None` 的分支是给"执行器只跑 forward、采样在别处"的路径准备的（例如某些分离式部署）。

请求进入调度器之前，`preprocess_add_request()`（`:1224`）会做一次预处理：多模态特征去重、watermark 注入、`Request.from_engine_core_request()` 构造调度器用的 `Request`、以及结构化输出的 grammar 初始化。这个函数可以放在独立的输入处理线程里跑，和模型 forward 并行——这是"请求还没进 batch，grammar 已经编译上了"的原因。

### 异步调度改变了什么

默认开启的 async scheduling（`vllm/config/vllm.py:1723` 起，无冲突时置 True）让 EngineCore 在 GPU 还在跑第 N 步时，就把第 N+1 步的输入准备好。调度器此时还没拿到第 N 步的输出 token，于是用 **placeholder token** 占位（`num_output_placeholders`），等输出回来再替换。

这带来一个反直觉的现象：`num_computed_tokens` 可能暂时大于真实的 token 数。所有"按 token 数做的判断"（是否到 max_tokens、是否该停）都要把 placeholder 扣掉，这也是调度器里那些 `+2 - num_output_placeholders` 的来源。

## 5. 输出回程：停在哪、怎么解码

`update_from_output()` 拿到 `ModelRunnerOutput` 后，做三件事：写回采样出的 token、判断是否停止、组装 `EngineCoreOutputs`。

停止条件集中在 `check_stop()`（`vllm/v1/core/sched/utils.py:98`），顺序很重要：

1. EOS token；
2. `stop_token_ids`；
3. 达到 `max_model_len` 或 `max_tokens`（`FINISHED_LENGTH_CAPPED`）；
4. `min_tokens` 未满足则继续；
5. 重复检测（repetition detection）触发 `FINISHED_REPETITION`。

请求的状态机定义在 `RequestStatus`（`vllm/v1/request.py:399`）：

| 状态 | 含义 |
|---|---|
| `WAITING` | 在等待队列，尚未进 batch |
| `WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR` | grammar 还没编译完 |
| `WAITING_FOR_REMOTE_KVS` | P/D 分离，等远端 KV |
| `WAITING_FOR_STREAMING_REQ` | 流式输入模式下等下一段输入 |
| `RUNNING` | 在 running 队列 |
| `PREEMPTED` | 被抢占，KV 已释放，回等待队列 |
| `FINISHED_*` | 各种终止原因 |

前端拿到 `EngineCoreOutputs` 后，`OutputProcessor.process_outputs()`（`vllm/v1/engine/output_processor.py:728`）逐个请求处理：更新 detokenizer、决定这一轮要不要发（`stream_interval`）、组装 `RequestOutput`。

### 增量 detokenize 为什么不是"每次 decode 整个序列"

如果每来一个 token 就把整个输出序列 `tokenizer.decode()` 一遍，复杂度是 $O(n^2)$，而且会遇到两个坑：

- **字节边界**：一个 UTF-8 字符可能由多个 token 拼成，单独 decode 最后一个 token 会得到替换字符；
- **空格规则**：SentencePiece 的 decode 会根据前后 token 决定要不要加空格，只看最后一个 token 会丢空格。

所以 vLLM 用 `detokenize_incrementally()`（`vllm/tokenizers/detokenizer_utils.py:176`）：保留一个"前缀窗口"（`prefix_offset`、`read_offset`），每次只 decode 窗口内的 token，再和上次结果做差。窗口通常只有几个 token，单 token 成本在几十微秒量级。fast tokenizer 有更快的路径（`IncrementalDetokenizer` 的子类），慢速 tokenizer 则退回 Python 实现。

## 6. 端到端时间账

把一次请求的 TTFT 拆开（8B 模型、8k prompt、H100、单请求）：

| 阶段 | 量级 | 备注 |
|---|---|---|
| 网络 + JSON 解析 | 0.1–1 ms | 取决于 body 大小 |
| chat template + tokenize | 1–5 ms | 8k token，fast tokenizer |
| ZMQ + msgpack | < 1 ms | 控制消息很小 |
| 排队 | 0 到数秒 | 取决于并发和 `max_num_seqs` |
| prefill | ~200–300 ms | 2·N·P ≈ 131 TFLOP |
| 首次 detokenize + SSE | < 1 ms | |

可以看到，**prefill 之外的部分加起来通常不到 10 ms**，但只要排队，TTFT 就完全不可控。这也是为什么调优的第一步永远是看 `waiting` 队列长度和 KV cache 使用率，而不是先怀疑 kernel。

ITL 的账更简单：一次 decode step（几毫秒到几十毫秒）+ detokenize（几十微秒）+ HTTP 写出（亚毫秒）。流式场景下，如果 ITL 出现规律性的尖刺，通常是 chunked prefill 的块太大，或者有请求在抢占/重算。

## 7. 常见坑

- **OpenAI 兼容层的字段会被静默忽略**：不在 schema 里的参数不会报错，而是直接丢掉。自定义 chat template 变量要走 `chat_template_kwargs`，别塞进顶层。
- **tokenize 与 detokenize 不对称**：特殊 token、byte-fallback、多字节 UTF-8 被切在两个 token 里，都会让"发出去的文字"和"模型看到的 token"对不上。调试时打印 token id，不要打印字符串。
- **断连不等于立刻释放**：客户端关闭流式连接后，请求要经过 abort 队列传到 EngineCore，KV 才会释放。高并发短连接场景下，这段延迟会体现在显存曲线上。
- **`--api-server-count` 不是越多越好**：每个 API server 是独立进程，有自己的 tokenizer 缓存和媒体加载线程；请求落到哪个进程取决于前端路由，粘性不好时前缀缓存会跨进程失效。
- **首 token 的时间不等于 prefill 时间**：TTFT 里还包含排队、tokenize、多模态加载和一次进程间往返。压测时把这几段分开计时，否则会误判优化效果。

## 8. 读代码时的三个检查点

1. **请求在哪一层被 tokenize？** 在 API server 的 Renderer，不在 EngineCore。`skip_tokenizer_init=True` 时传 token ids，可以完全跳过这一段。
2. **`EngineCoreRequest` 里有没有你要的字段？** 自定义请求属性（优先级、cache salt、KV hint）必须能序列化进这个 struct，否则跨进程就丢了。
3. **输出是 delta 还是 cumulative？** `RequestOutputKind` 决定 `RequestOutput` 的语义；`stream_interval` 决定攒几个 token 发一次。调 `stream_interval` 能降 CPU 开销，但会牺牲流式的"逐字感"。

## 小结

- V1 是"前端 bind、引擎 connect"的多进程架构：API server 负责协议/tokenize/多模态，EngineCore 负责调度，Worker 负责 forward。
- 请求对象换三次：`EngineInput`（Renderer 产物）→ `EngineCoreRequest`（跨进程）→ `Request`（调度器内部）。
- 控制面走 ZMQ + msgpack，大张量走共享内存；请求类型是一个字节的枚举。
- EngineCore 的 `step()` 是"调度 → 异步执行 → 采样 → 更新状态"四段，`future.result()` 前的空隙用来算 grammar bitmask。
- async scheduling 默认开启，用 placeholder token 解决"输出未回、输入先行"的竞态。
- 停止条件有严格的顺序（EOS → stop ids → 长度 → min_tokens → 重复检测），改顺序会改变行为。
- TTFT 的大头是排队和 prefill；tokenize/detokenize/ZMQ 加起来通常不到 10 ms。

下一篇进入显存：[PagedAttention 到底怎么把 KV 切成块、块表又怎么被 attention kernel 使用]({{< relref "vllm-02-paged-attention" >}})。

> 官方文档：<https://docs.vllm.ai/en/latest/design/arch_overview/>（进程模型）、<https://docs.vllm.ai/en/latest/usage/v1_guide/>（V1 行为变化）
