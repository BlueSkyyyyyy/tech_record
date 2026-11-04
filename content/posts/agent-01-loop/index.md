---
title: "Agent Loop 解剖：一条消息如何变成一次工具调用"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 2
series: ["agent"]
tags: ["AI Agent", "系列", "Agent Loop", "ReAct", "Function Calling", "Harness"]
categories: ["LLM算法"]
---

上一篇把 agent 拆成模型、harness、环境三个部件。这一篇只看最里面的那个齿轮：**循环**。

几乎每个 agent 教程都会给你这段代码：

```python
while True:
    response = model(messages, tools)
    if not response.tool_calls:
        break
    for call in response.tool_calls:
        messages.append(run_tool(call))
```

十行，能跑，也能做出 demo。但它藏着四个问题，每一个都会在生产里咬人：

1. **什么时候停？** "没有 tool call"就够吗？模型说"我做完了"但测试没过，算停吗？
2. **工具结果长什么样？** 是拼成一段文本，还是结构化地回填 `call_id`？并行调用怎么配对？
3. **工具报错怎么办？** 抛异常终止循环，还是把错误当成观察喂回去？
4. **循环有没有上限？** 步数、token、钱、墙钟时间，哪个先到？

这篇按"协议 → 三个真实实现 → 终止条件 → 失败模式"的顺序，把这些问题拆开。

## 1. 先记住一个画面

ReAct 那篇 2022 年的论文给了一个很朴素的画面：模型在"想"和"做"之间交替。

```text
Thought: 我需要先看 app.py 里 greet 函数的实现
Action:  read_file({"path": "app.py"})
Observation: def greet(name): return "hi " + name
Thought: 参数拼接少了一个逗号，改成 f-string
Action:  edit_file({"path": "app.py", "old": "...", "new": "..."})
Observation: ok
Thought: 跑测试
Action:  run_tests({})
Observation: 3 passed
Thought: 完成了
Final: 已修复
```

这个画面之所以重要，是因为它把"推理"和"行动"放进了同一个 token 流：模型不需要一次想完所有步骤，而是可以根据上一步的真实观察决定下一步。这也是 agent 和 workflow 的分界线——workflow 的路径是代码写死的，agent 的路径是模型每一步现选的。

但 ReAct 原始实现有一个工程上的硬伤：**它靠文本解析**。模型输出 `Action: read_file({"path": ...})`，harness 用正则去抠工具名和参数。模型多写一个换行、少一个引号、把 JSON 写成 Python dict，解析就失败；失败了还得让模型"重新按格式输出"，一来一回就是一次完整的模型调用。

> **类比的边界**：ReAct 像一个人在纸上写"我要去厨房拿杯子"，然后另一个人照着做。纸上写错了，执行的人只能猜。Function calling 相当于把这张纸换成了结构化的工单：字段、类型、必填项都由 schema 约束，执行的人不需要猜。

## 2. 从文本协议到结构化协议

2023 年之后，主流 API 陆续把工具调用做进了模型协议：模型不再输出一段"Action:"文本，而是输出一个结构化的 `tool_call` 对象，带 `id`、`name`、`arguments`。harness 执行完，用同一个 `id` 把结果回填。

以 OpenAI Responses 风格的 item 为例，一次往返大致是：

```json
{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "我先读文件"}]}
{"type": "function_call", "call_id": "call_1", "name": "read_file", "arguments": "{\"path\":\"app.py\"}"}
{"type": "function_call_output", "call_id": "call_1", "output": "def greet..."}
```

三个细节决定了这套协议能不能稳：

- **`call_id` 必须原样回传**。模型可能一次发起多个调用；harness 并行执行后，结果顺序可能乱，只有 id 能配对。配错一次，模型就会把 A 文件的输出当成 B 文件的。
- **参数是字符串，不是对象**。很多 API 把 `arguments` 当 JSON 字符串传，模型可能生成非法 JSON。harness 要在解析失败时把"解析错误 + 原始字符串"作为观察回传，而不是直接 crash。
- **工具结果是"用户侧"消息**。在多数协议里，`function_call_output` 的 role 是 user/tool，不是 assistant。它和普通用户消息共享同一个 token 预算，也一样会被压缩策略影响。

这就引出了第一个反直觉的结论：**agent loop 的核心数据结构不是 messages 数组，而是"未配对的调用"这个集合**。循环的每一步，本质上是在问：现在有没有 `function_call` 还没拿到 `function_call_output`？有就执行，没有就结束（或者继续让模型说）。

## 3. 三个真实实现

### 3.1 mini-swe-agent：190 行，把循环压到最小

`mini-swe-agent` 的整个 agent 在 `src/minisweagent/agents/default.py`，190 行。核心只有三个方法：

```python
def run(self, task=""):
    self.messages = []
    self.add_messages(system_message, user_message)
    while True:
        try:
            self.step()
            self.n_consecutive_format_errors = 0
        except FormatError as e:
            ...
        finally:
            self.save(self.config.output_path)
        if self.messages[-1].get("role") == "exit":
            break
    return self.messages[-1].get("extra", {})

def step(self):
    return self.execute_actions(self.query())

def query(self):
    if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
        raise LimitsExceeded(...)
    self.n_calls += 1
    message = self.model.query(self.messages)
    self.cost += message.get("extra", {}).get("cost", 0.0)
    self.add_messages(message)
    return message
```

（`default.py:88-152`，行号对应 `04d809ce`。）

三个设计非常值得抄：

- **终止是一个消息角色，不是一个返回值**。所有退出路径——模型提交、步数超限、成本超限、连续格式错误、未捕获异常——都往 `messages` 里追加一条 `role="exit"` 的消息，循环统一检查 `messages[-1].role == "exit"`。这样 trajectory 本身就是完整的审计日志，退出原因也在里面。
- **预算检查在调用之前**。`query()` 先看 `step_limit` 和 `cost_limit`，再发请求。所以 `step_limit=250` 的含义是"最多 250 次模型调用"，不是"最多 250 次工具调用"。
- **格式错误单独计数，并且补记成本**。模型返回了内容但解析不出动作，钱已经花了；`FormatError` 分支手动把 `extra.cost` 加回去，否则成本账会漏。连续 3 次格式错误就退出。

它的工具面也极小：只有一个 `bash` 工具，参数只有一个 `command`（`models/utils/actions_toolcall.py:11-27`）。任务完成的信号是 stdout 的**第一行**等于 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` 且返回码为 0，剩下的 stdout 就是 submission（`environments/local.py:45-56`）。这是一个很"Unix"的设计：不引入 finish 工具，不解析 JSON，用一条约定俗成的命令结束。

代价也很明显：没有编辑工具、没有 grep、没有持久 shell、没有压缩、没有子 agent。它把所有复杂度都推给了模型和提示词。

### 3.2 Codex CLI：三层循环，没有 max turns

Codex 的循环在 `codex-rs/core/src/session/turn.rs:182` 的 `run_turn` 里，循环头在 `:498`。它的结构和 mini 完全不同：

```rust
loop {
    // 1. 取 pending input（用户在模型运行中插的话）
    // 2. 刷新 step_context（工具、MCP、skills、环境）
    // 3. 组装 history，发一次 sampling request
    let result = run_sampling_request(...).await;
    // 4. 根据结果决定要不要继续
    let needs_follow_up = model_needs_follow_up || has_pending_input;
    // 5. 需要压缩就压缩，然后 continue；否则 break
    if !needs_follow_up { break; }
}
```

关键在 `needs_follow_up` 的来源：只要有工具调用，或者模型返回 `end_turn: false`，就继续（`session/turn.rs:644`、`stream_events_utils.rs:379`、`:3485`）。**代码里没有 `max_turns` 这个变量**。dossier 里把整个树搜了一遍，唯一的 `RECAP_HISTORY_MAX_TURNS` 是 TUI 的历史回看，不是循环上限。

为什么敢不设步数上限？代码里有一句注释（`turn.rs:715`）：

> "as long as compaction works well in getting us way below the token limit, we shouldn't worry about being in an infinite loop."

也就是说，Codex 把"循环会不会失控"这个问题，交给了 **token 预算 + 压缩**，而不是步数计数器。每轮采样后都会算一次 token status，超了就触发 compaction（见第 4 篇）。如果压缩失败、token 还是超，循环会因为请求失败而结束。

这是一个很重要的设计分歧：

- mini 用**外部预算**（步数、成本、墙钟）兜底，简单、可预测。
- Codex 用**内部预算**（context token）兜底，更贴近真实约束，但要求压缩足够可靠。

两者都没有错，错的是"什么都不设"。

Codex 还有两个和循环强相关的细节：

- **工具调用不等模型流结束**。`handle_output_item_done` 收到一个 `function_call` 就 `tokio::spawn` 执行，把 future 塞进 `FuturesOrdered`；等整条流结束，再 `drain_in_flight` 统一收口（`turn.rs:3239`、`:3684`）。所以工具执行和模型继续生成是重叠的。
- **工具并发靠 RwLock，不靠 API 的 `parallel_tool_calls`**。`build_prompt` 里确实把 `parallel_tool_calls` 设成了 `true`（`:1807`），但真正决定能不能并行的是每个工具的 `supports_parallel_tool_calls()`，默认 `false`（`codex-rs/tools/src/tool_executor.rs:122-124`）；运行时读锁并行、写锁独占（`tools/parallel.rs:287-299`）。也就是说，模型可以一次请求 5 个 `apply_patch`，它们仍然会串行执行。

### 3.3 Gemini CLI：递归 + 五道终止闸门

Gemini CLI 的循环又不一样：它不是 `while`，而是 **递归**。

`GeminiClient.processTurn`（`packages/core/src/core/client.ts:649`）处理一轮：压缩 → 估算 token → 溢出检查 → 循环检测 → 选模型 → `turn.run` 流式产出。流结束后，如果没有 pending tool call，它会做一次 "next speaker check"；如果判定模型还应该继续说，就递归调用 `sendMessageStream`，把 `boundedTurns - 1` 传下去（`:929-938`）。

工具调用由谁闭合？**不在 core 里**。`CoreToolScheduler` 执行完工具，把 `functionResponse` 放进 `CompletedToolCall.responseParts`；CLI 层的 `useGeminiStream.ts:2305-2356` 把这些 part 收拢，作为一条**新的 user turn** 重新 `submitQuery`。所以从 core 的视角看，工具结果就是用户又说了一句话。

Gemini 的终止条件有五个，叠在一起：

| 闸门 | 位置 | 触发后 |
|---|---|---|
| `MAX_TURNS = 100` | `client.ts:80`，`:995` 取 min | 返回 `MaxSessionTurns` |
| 会话级上限 | `getMaxSessionTurns()`，`:659-666` | 返回 |
| 循环检测 | `loopDetectionService.ts` | `LoopDetected`，或尝试恢复 |
| context 溢出 | `client.ts:744-750` | `ContextWindowWillOverflow`，直接返回 |
| next speaker 判定 | `client.ts:929-938` | 递归 "Please continue." |

循环检测的规则值得单说：同一工具调用（name + JSON 参数的 SHA-256）重复 5 次、或内容 chunk 重复 10 次、或一个 LLM judge 以 0.9 置信度判定"在绕圈"，都会触发（`loopDetectionService.ts:29-30,65,174-178`）。这是我在三个 harness 里看到的**唯一一个把"模型可能原地打转"当成一等公民**的实现。

## 4. 终止条件的分类学

把三个实现放在一起，终止条件可以分成六类。你写自己的 harness 时，最好六类都覆盖：

| 类型 | 例子 | 优点 | 风险 |
|---|---|---|---|
| **协议终止** | 没有 tool call；`end_turn: true` | 自然、模型自主 | 模型可能"说完了"但没做完 |
| **显式终止工具** | `finish` / `submit` | 可带结构化结果 | 模型可能忘记调用 |
| **哨兵终止** | stdout 第一行 `COMPLETE_...` | 极简、可脚本化 | 被 `tee`/前缀污染就失效 |
| **预算终止** | 步数、token、成本、墙钟 | 可预测、可计费 | 预算耗尽时留下半成品 |
| **安全终止** | 审批拒绝、沙箱拒绝、循环检测 | 防事故 | 误报会打断正常任务 |
| **错误终止** | API 错误、解析失败、取消 | 必须存在 | 错误信息要能回到模型才有救 |

一个成熟的 harness 不会只依赖一种。Codex 是协议 + 预算 + 安全；Gemini 是协议 + 预算 + 循环检测；mini 是显式 + 哨兵 + 预算 + 格式错误。

## 5. 失败模式：循环里的四个坑

### 坑一：工具错误被当成"循环终止"

最差的写法是 `run_tool` 抛异常，异常冒泡到 `while` 外面，整个 agent 死掉。正确的写法是**把异常变成观察**：

```python
try:
    out = fn(args)
except Exception as e:
    out = f"ERROR: {type(e).__name__}: {e}"
```

这不是"宽容"，而是把错误变成模型可以推理的信息。模型看到 `FileNotFoundError: missing.py`，下一步就可能去 `ls` 一下；看到 `Permission denied`，就可能换一个路径。当然，前提是错误信息足够具体——`ERROR: failed` 等于没说。

### 坑二：并行调用的结果配错

并行执行时，结果返回顺序是不确定的。如果 harness 按"完成顺序"往 messages 里塞，而不是按 `call_id` 配对，模型就会拿到错位的结果。这种 bug 在单工具、串行执行的 demo 里永远暴露不出来，一上并行就随机出错。

### 坑三：重复调用

模型可能连续三次调用同一个 `read_file`，参数完全一样。原因通常不是模型笨，而是**上一次的结果被截断或压缩掉了**，模型以为没读到。Gemini 的循环检测是在治标；治本的办法是让工具结果在压缩后仍然保留"我读过这个文件"的摘要，或者给工具调用加幂等缓存。

### 坑四：context 在循环里无声增长

每一轮都追加 assistant 消息、tool call、tool output。一个 20 轮的 SWE 任务，工具输出很容易到几十万 token。如果没有压缩，循环会在某一轮突然因为 context 超限而失败——而失败点往往在最关键的验证步骤。

这就是下一篇的主题：工具层怎么设计，才能让每次往返携带的信息更少、更准。

## 6. 最小可运行

`code/agent/loop_sim.py` 把这一篇的骨架跑给你看：一个剧本模型、三个工具、并行执行、错误观察、步数上限。

```bash
cd code/agent
python3 loop_sim.py
```

输出：

```text
status=submitted  steps=4  tool_calls=5
  model: 先读两个文件，再跑测试。 (calls=2)
  tool[c1] -> # contents of app.py
  tool[c2] -> # contents of test_app.py
  model: 读一个不存在的文件，看看错误怎么回来。 (calls=1)
  tool[c3] -> ERROR: FileNotFoundError: missing.py
  model: 改完了，跑测试。 (calls=1)
  tool[c4] -> 1 passed, 0 failed
  model: 提交。 (calls=1)
  tool[c5] -> SUBMITTED: patched app.py
```

四个断言值得注意（`--test` 里）：

- 并行和非并行必须产生**相同条数的消息**——否则配对逻辑有 bug。
- `max_steps=2` 必须真的在第 2 步停，且不能执行到 `finish`。
- 未知工具必须变成一条 `ERROR: unknown tool` 观察，而不是异常。
- `finish` 是终止工具，执行后仍然要把结果写回 transcript，再退出。

## 小结

- Agent loop 的骨架是"模型 → 工具 → 观察 → 模型"，但生产级的差异全在**终止条件、错误处理、状态管理**。
- 协议从文本走向结构化，核心收益是 `call_id` 配对和参数 schema；代价是模型仍然可能生成非法 JSON，harness 必须把解析错误喂回去。
- 三个 harness 代表三种哲学：mini 用外部预算兜底，Codex 用 token 预算兜底，Gemini 用多道闸门兜底。没有银弹，只有"你有没有为失控做准备"。
- 循环里最容易被忽略的失败模式是：错误被当成终止、并行配错、重复调用、context 无声增长。

下一篇往下走一层：模型能调用的那些工具，接口应该长什么样？为什么"工具描述"本身会吃掉几万 token，而 MCP 和 Code Mode 又分别在解决什么问题？
