---
title: "Harness 解剖（三）：OpenHands 的事件溯源、500 步熔断与可恢复会话"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 7
series: ["agent"]
tags: ["AI Agent", "系列", "OpenHands", "Harness", "Event Sourcing", "SDK", "沙箱", "源码分析"]
categories: ["训练框架"]
---

前两篇看了三个 harness：Codex 是本地优先的 Rust 运行时，Gemini CLI 是把循环放在 core 外面的 TypeScript 平台，mini-swe-agent 则用 190 行证明循环本身可以极简。

这一篇看第四个象限：**OpenHands**。它不是从 CLI 长出来的，而是从"一个可以被托管、被恢复、被多个客户端同时观察的 agent 会话"长出来的。源码固定在 `OpenHands/software-agent-sdk@2f2cd2bf7f19ca3bbf9f0c4a95c60f48094302a1`（2026-11-04），版本 `1.59.0`；行号以这个 checkout 为准。

它的核心选择只有一句话：**先把发生的事记成不可变事件，再决定给模型看什么。**

## 1. 先记住一个画面

把一次 agent 会话想成一家银行的**账本**，而不是一块白板。

柜员每做一笔业务，就在账本上记一行：谁、什么时候、做了什么、结果如何。下班时账本锁进金库。第二天换一个柜员，他先读账本，再决定下一步；白板可以擦，账本不能撕。

OpenHands 的三个对象正好对应这个画面：

- **EventLog** 是账本：每个事件一个小 JSON，追加写，永不修改。
- **View** 是上班前抄到白板上的那一段：它是账本的**投影**，只包含模型这次需要看的消息。
- **Condensation** 是账本里新加的一页："第 3 到第 187 笔我概括成了这段话"——它没有撕掉原来那几页，只是告诉后来的人"这部分可以看摘要"。

> **类比的边界**：银行账本只记录事实，不参与决策；agent 的 event log 会被投影成 prompt，投影本身会改变角色、顺序和权重。账本可以审计，投影不能自证。后面第 9 节会看到，OpenHands 的投影恰好有一个角色被压平的口子。

## 2. 四层，而不是一个 CLI

先把规模摆出来。这个 checkout 里：

| 层 | 目录 | 规模（Python） | 职责 |
|---|---|---|---|
| SDK | `openhands-sdk/` | 316 个文件，约 84,635 行 | Agent、Conversation、EventLog、View、工具契约、安全策略 |
| 工具 | `openhands-tools/` | 97 个文件，约 17,821 行 | terminal、file editor、browser、task tracker、subagent |
| Agent Server | `openhands-agent-server/` | 98 个文件，约 36,713 行 | FastAPI、会话注册、事件查询、租约、多客户端订阅 |
| Workspace | `openhands-workspace/` | local / Docker / Apptainer / cloud / remote API | 文件系统、shell、浏览器、沙箱边界 |

再加上 CLI、Canvas 前端和 ACP 客户端，就是完整的 OpenHands。

这和 Codex 的形态不同。Codex 的 session 活在进程里，进程退出，session 就结束了；Gemini CLI 的 chat 活在 CLI 进程里；OpenHands 的 `Conversation` 是一个可以被 server 托管、被多个客户端订阅、被 fork、被恢复的**第一类对象**。

顺带一个坐标：OpenHands 的平台论文（arXiv 2407.16741）说它有 15 个挑战性任务/benchmark、2.1K 次贡献、188 位贡献者，MIT 许可。它最初是 SWE-agent 那条 ACI（agent-computer interface）路线的工程化：SWE-agent 用接口设计把 SWE-bench pass@1 从非交互基线拉到 12.5%，HumanEvalFix 到 87.7%；OpenHands 把这条路线做成了一个平台。

## 3. 一次 run 的对象图

```text
Agent（LLM + tools + condenser + security analyzer）
   │  step()
   ▼
LocalConversation ──────► EventLog（append-only，事件树）
   │                         │
   │                         └──► View（派生，LLM messages）
   └── Workspace（local / Docker / remote）
```

`LocalConversation._run` 在 `conversation/impl/local_conversation.py:2028`，主循环是 `:2046` 的 `while True`，每一步调用 `agent.step` 在 `:2098`。

关键顺序是：**先有事件，再有视图**。模型调用、工具调用、工具结果、压缩、确认、状态变化，全都先变成 `Event` 追加到日志；`View` 是从日志派生出来的只读投影。这个顺序决定了后面所有的恢复、审计和 fork 能力。

## 4. Event 是不可变的，会话是一棵树

`Event` 的基类在 `event/base.py:21`，几个字段值得记住：

```python
model_config = ConfigDict(extra="forbid", frozen=True)  # :25
id: EventID                                             # :29
timestamp: str                                          # :30
source: SourceType                                      # :32  agent / user / environment
parent_id: EventID | None                               # :33
```

`frozen=True` 意味着事件创建后不能改；`parent_id` 意味着事件不是一条线，而是一棵树——**共享同一个 `parent_id` 的事件是兄弟分支**。`EventLog.path_to_root(leaf)`（`event_store.py:158`）从叶子往根走，返回 root-first 的链；`limit` 参数只保留最近 N 个事件，走的是 O(limit) 而不是 O(branch)。

持久化格式很朴素，一个事件一个文件（`persistence_const.py:10`）：

```text
conversations/<conversation-id>/
  base_state.json
  events/
    event-00000-<uuid>.json
    event-00001-<uuid>.json
    ...
    .eventlog.lock
```

追加逻辑在 `event_store.py:241`：拿文件锁（`flock`，超时 30 秒，`:24`、`:255`），校验事件 ID 不重复、`parent_id` 必须存在（`:265`、`:271`），写一个新文件，推进长度标记。它还有两个工程细节：

- **不重用磁盘序号**。即使中间丢了一个事件文件，新事件也会分配更大的序号，避免 reload 顺序依赖 UUID。
- **丢文件不遮住后面的事件**。扫描时把空洞记下来，把幸存事件重新编成稠密索引，日志里警告前 5 个空洞（`:392` 起）。一个文件坏了，不会让整条会话不可读。

这比"一个 JSONL 追加文件"重，但换来两件事：单事件可寻址、可分支。代价是文件数——按每步一对 action/observation 估算，跑到 500 步上限大约是上千个小文件。

## 5. 恢复：配置在 base_state，历史在 events

`ConversationState.create`（`conversation/state.py:488`）是 create-or-resume 的入口。它先读 `base_state.json`（`:552`），再挂载 `EventLog`（`:570`）。

两个文件分工明确：

| 文件 | 存什么 | 恢复时的角色 |
|---|---|---|
| `base_state.json` | agent 配置、workspace、max_iterations、预算、确认策略、统计 | **agent 的唯一事实来源** |
| `events/` | 所有发生过的事件 | 历史，用来重建 View 和 pending 状态 |

恢复时的兼容规则很有意思（`state.py:583-600`）：

- **工具不能删**。持久化的 agent 用过哪些工具，恢复时的 agent 必须至少包含这些工具；可以加，不能减。原因是历史里已经出现了这些工具调用，删掉会让模型看到无法解释的 tool call。
- **其他配置可以换**。LLM、system prompt、agent context、condenser 都可以在恢复时替换。这是有意的：模型升级、prompt 迭代不该被旧会话锁死。

### 5.1 确认是事件，不是内存里的回调

这是 OpenHands 最值得抄的一个设计。

`ConversationState.get_unmatched_actions`（`state.py:713`）扫描当前 active branch，找出**没有对应 observation 的 ActionEvent**：

- 有 `ObservationEvent` 或 `UserRejectObservation`，算已处理；
- 有 `AgentErrorEvent`，按 `tool_call_id` 匹配，也算已处理（这是崩溃恢复时错误事件可能晚到的兼容）；
- 剩下的 ActionEvent 就是 pending。

`Agent._step` 在采样新动作之前，先检查 pending（`agent/agent.py:747-757`）：

```python
pending_actions = ConversationState.get_unmatched_actions(state.active_branch())
if pending_actions:
    self._execute_actions(conversation, pending_actions, on_event)
    return
```

所以"等待用户确认"不是某个回调对象的状态，而是**日志里有一个 action、还没有 observation**。进程重启、页面刷新、server 换实例，只要日志还在，pending 就还在。这比把确认状态放在内存里可靠得多。

### 5.2 代价：至少一次，不是恰好一次

但事件日志只能证明"没有 observation"，**不能证明"动作没有发生"**。

如果工具已经执行、副作用已经产生，进程在写 observation 之前崩了，恢复时这个 action 仍然是 pending，会被再执行一次。也就是说，OpenHands 的恢复语义是 **at-least-once**，不是 exactly-once。

这对工具有真实要求：

- 读操作无所谓；
- 写文件要幂等，或者用内容哈希/版本号防重；
- 发邮件、提交 PR、转账这类外部副作用，必须带幂等键，或者在工具内部先写"已执行"标记再执行。

事件溯源把崩溃窗口**暴露**出来了，但没有替你消除它。任何 harness 都有同样的窗口；OpenHands 只是把 pending 状态显式写进了事件日志。

## 6. 循环：500 步、美元预算、stop hook

`_run` 的骨架（`local_conversation.py:2028-2150`）可以压缩成四件事：

```text
while True:
  if PAUSED or STUCK: break
  if FINISHED:
      run stop hooks
      if denied: inject feedback as user message; status = RUNNING; continue
      break
  stuck check / nudge
  if WAITING_FOR_CONFIRMATION: status = RUNNING
  agent.step()
  iteration += 1
  if WAITING_FOR_CONFIRMATION: break
  if budget exceeded: emit MaxBudgetReached; break
  if iteration >= max_iteration_per_run: emit MaxIterationsReached; break
```

几个容易看错的点：

1. **500 是 per run，不是 per conversation**。`iteration` 每次 `_run` 从 0 开始；用户再发一条消息，会开一次新的 run，额度重新算。所以一个长会话可以远超 500 步。
2. **500 数的是 step，不是 tool call**。一个 step 里模型可以并行调多个工具，实际工具调用数可能远大于 500。
3. **先执行，后检查上限**。到达上限的那一步已经执行完了；如果这一步刚好 `FINISHED`，状态会保留 `FINISHED`，不会被改写成 `ERROR`（`:2140-2145`）。
4. **预算比的是整个 conversation 的累计成本**。字段叫 `max_budget_per_run`，但 `_budget_exceeded_detail`（`:763`）读的是 `conversation_stats.get_combined_metrics().accumulated_cost`，包含 agent、condenser 等所有 LLM。跨 run 不会清零，除非新建会话。
5. **本地是 ERROR 事件，远程是异常**。本地 `run()` 把 `MaxIterationsReached` 写成 `ConversationErrorEvent`，状态置 `ERROR`；`RemoteConversation.run()` 则会抛 `ConversationRunError`。同一个语义，两种 API 表达。

### 6.1 谁决定"做完了"

OpenHands 把终止拆成了三层：

| 层 | 机制 | 谁说了算 |
|---|---|---|
| 模型 | `finish` 工具 / 自然结束 | 模型 |
| 策略 | stop hook 可以拒绝 `FINISHED` | 外部策略 |
| 硬边界 | 500 step、美元预算 | harness |

stop hook 的实现很直接（`:2052-2088`）：agent 说完成了，hook 返回的 `should_stop` 为 `False`（拒绝停止），就构造一条 `MessageEvent(source="environment", llm_message=Message(role="user", ...))` 的反馈消息，把状态改回 `RUNNING`，继续跑。

这条消息是 OpenHands 一个很实用的模式：**"完成"不是模型单方面宣布的，是可以被外部驳回的**。但它也埋下了第 9 节要讲的隐患：environment 的反馈在投影里就是一条普通 user 消息。

## 7. 卡住：先 nudge，再 STUCK

OpenHands 默认开 stuck detection（`state.py:119`），核心在 `conversation/stuck_detector.py`。

它只看**最近 20 个事件**（`MAX_EVENTS_TO_SCAN_FOR_STUCK_DETECTION = 20`），并且只看最后一条 user 消息之后的部分。五种模式：

| 模式 | 默认阈值 | 看什么 |
|---|---|---|
| 重复 action + observation | 4 | 同一个动作、同一个观察反复出现 |
| 重复 action + error | 3 | 同一个工具、同一个参数、同一个错误 |
| agent 独白 | 3 | 连续多条 agent message，没有 user 打断 |
| 交替模式 | 6 | A/B/A/B 的循环 |
| context window 错误循环 | — | 压缩/窗口错误反复出现 |

（阈值定义在 `conversation/types.py:136-161`。）

流程是**先 nudge，再判 STUCK**（`local_conversation.py:788`）：

- action-error 连续到阈值时，先注入一条环境消息，提醒模型"同样的调用已经错了 N 次，改参数或换方法"；每条 streak 只 nudge 一次（`stuck_detector.py:343`）。
- 如果 nudge 之后还是重复，`is_stuck()` 为真，状态置 `STUCK`，循环 break。

`STUCK` 不是终态：下次 `_run` 的起始状态允许 `STUCK`，用户补一条消息就能继续。这是"暂停并求助"，不是"杀死会话"。

这个设计解决的是**机械重复**。它抓不住另一种更常见的失败：模型每次都换一个做法，但每次都错。那种"语义卡住"需要任务级 verifier，而不是事件模式匹配。2026 年 arXiv 2606.29718 的测量也说明，agent 经常在窗口填满之前就给出"搜索没有结论"式的放弃；这类失败不是重复，nudge 和 stuck detector 都帮不上忙，只能靠 stop hook 或外部 verifier 把任务重新推回去。

## 8. 工具：默认很小，并发是 opt-in

默认工具集在 `openhands-tools/openhands/tools/preset/default.py:37`：

- `terminal`
- `file_editor`
- `task_tracker`
- `browser`（CLI 模式下默认关闭）
- 可选的 `task`（subagent 工具集，默认关闭）

加上默认内置的 `finish`；`think` 则要显式加进 `include_default_tools` 才会出现。这比 Codex 的 57 个 handler 小得多，比 mini 的单个 bash 大一点。它把"子 agent"和"浏览器"都做成了显式开关，而不是默认能力。

### 8.1 并发默认是 1

`Agent.tool_concurrency_limit` 默认 `1`（`agent/base.py:304`），`ParallelToolExecutor` 默认 `max_workers=1`（`agent/parallel_executor.py:63`）。也就是说，**默认串行**；要并行，得显式把 `tool_concurrency_limit` 调大。

调大之后，真正的安全阀不是线程池，而是**资源锁**（`conversation/resource_lock_manager.py`）：

- 每个工具通过 `declared_resources(action)` 声明自己碰哪些资源，key 形如 `file:/abs/path`、`terminal:session`；
- 执行器按资源 key 加锁，多个 key **按排序顺序获取**，避免死锁（`:100`）；
- 锁是 FIFO，避免饿死；每类资源有独立超时：file 30 秒、terminal 300 秒、browser 300 秒、mcp 300 秒、tool 60 秒（`:21-25`）。

两个实现细节很能说明设计者的取舍：

- `FileEditorTool` 连只读的 `view` 也锁目标文件（`file_editor/definition.py:198`），防止并发写时读到半截内容；
- `TerminalTool` 在 tmux 池化后端下**主动放弃**框架锁（`terminal/definition.py:275`），因为 pane 级隔离已经保证并发安全；只有单 session 的 subprocess 后端才声明 `terminal:session` 串行。

这比"API 说可以并行就并行"细，也比"全局一把大锁"快。但它有一个前提：**工具作者必须正确声明资源**。基类默认返回 `declared=False`（`tool/tool.py:529`），一个忘记声明的工具会在并发下裸奔。

| 维度 | Codex | Gemini CLI | OpenHands |
|---|---|---|---|
| 并发开关 | 每个工具 `supports_parallel_tool_calls()` | 模型声明 + 调度器分批 | agent 级 `tool_concurrency_limit` |
| 互斥粒度 | 一把 RwLock | 连续批 + all-ready 屏障 | 资源 key + 排序锁 |
| 默认 | 串行（只有少数工具 opt-in） | 模型说并行才并行 | 串行（默认 1） |

## 9. Context：condenser 是投影，不是删除

`View` 是 `context/view/view.py` 里的线性列表，`events_to_messages`（`event/base.py:99`）把事件投影成 LLM messages。它做两件合并：

- 同一个 `llm_response_id` 的多个 `ActionEvent` 合成一条 assistant 消息（`:110-125`），这是并行工具调用的正确表示；
- 相邻的"纯 user 消息"可以合并（`:143-150`），减少 turn 数。

压缩由 `LLMSummarizingCondenser` 负责（`context/condenser/llm_summarizing_condenser.py`）：

- 默认 `max_size=240` 个事件（`:64`），`keep_first=2`（`:67`）；
- 至少压缩 view 的 10%，否则视为失败（`:72`）；
- 目标是把事件数/ token 数压到阈值的一半左右；
- 摘要由一个**独立的 LLM** 生成（`:47-56`），和 agent 的 LLM 分开；摘要失败会做 hard context reset，最多重试 5 次（`:81`），每次把单事件字符串上限缩小到 0.8 倍再试。

关键在于：压缩产生的是一个 `Condensation` **事件**（`event/condenser.py:11`），它记录 `forgotten_event_ids`、`summary`、`summary_offset`。原始事件还在日志里；`View` 只是在投影时把它们去掉，并把摘要插到指定位置。

所以从审计角度，你事后可以回答："第 42 步到第 187 步被压缩了，摘要是什么，哪些事件被忘了。"这比"压缩后只剩一段不可追溯的文本"强。

### 9.1 但投影会把角色压平

2026 年 arXiv 2610.29869（*Compaction Laundering*）把 OpenHands 列进了"flattening"组：该组 5 个 harness 的伪造批准成功率是 **87%–100%**；另外 4 个"角色保真型" harness（Codex、Gemini CLI、Qwen Code、Continue）是 **0%**，即使窗口开到约 **908K token**。

这看起来和"事件日志保留一切"矛盾，其实不矛盾：**攻击发生在投影层，不在存储层**。

一个具体的口子：stop hook 的反馈和 stuck nudge 都构造 `MessageEvent(source="environment", llm_message=Message(role="user"))`（`local_conversation.py:2059-2080`、`:798-803`）。在 `Event` 里，`source` 还是 `environment`，审计日志能区分；但在 `to_llm_message()` 之后，模型看到的只是一条 `role="user"` 的消息，和真人用户没有区别。

这不是说 OpenHands 的设计错了——把环境反馈放进 user 角色，是让模型把它当指令执行的最直接方式。但它说明：**事件溯源解决了"事实是否留下痕迹"，没有解决"模型看到的消息是否保留来源"**。如果要做注入防御，投影必须带类型化的角色/来源，而不是只靠 event 的 `source` 字段。

## 10. 安全：确认是事件，分析器是可替换的

确认模式的两阶段在 agent 里：

- `_requires_user_confirmation`（`agent/agent.py:1164`）分析 pending actions；
- 任意一个 action 的风险超过策略阈值，状态置 `WAITING_FOR_CONFIRMATION`（`:1229`），循环 break；
- 用户批准后，下一次 `_step` 看到 pending，先执行，再继续采样。

这和 5.1 的 pending 机制是同一套：**确认状态可以跨进程恢复**。

但默认配置很值得注意（`state.py:128-131`）：

```python
confirmation_policy: ConfirmationPolicyBase = NeverConfirm()
security_analyzer: SecurityAnalyzerBase | None = None
```

默认是**不确认、不分析**。SDK 的示例里要手动装：

```python
conversation.set_security_analyzer(LLMSecurityAnalyzer())
conversation.set_confirmation_policy(ConfirmRisky(threshold=SecurityRisk.HIGH))
```

`LLMSecurityAnalyzer` 的实现只有一句（`security/llm_analyzer.py:20-30`）：返回 `action.security_risk`，也就是**模型在 tool call 里自己填的风险等级**。模型说 LOW，就不确认；模型漏填，就是 UNKNOWN，`ConfirmRisky` 默认会确认 UNKNOWN。

把模型的自评当安全边界，是很危险的。模型可能被注入、可能判断错、可能为了让任务继续而低报。OpenHands 也提供了另一条路：`security/defense_in_depth/` 里的**确定性分析器**，明确写着"no network calls, no model inference"：

- `PatternSecurityAnalyzer`：正则签名，双 corpus 扫描；
- `PolicyRailSecurityAnalyzer`：组合条件规则，v1 只有三条 rail——`fetch-to-exec`（`curl ... | sh`）、`raw-disk-op`、`catastrophic-delete`。

`EnsembleSecurityAnalyzer`（`security/ensemble.py:20`）把多个分析器按最严重结果融合；子分析器抛异常时**fail-closed 到 HIGH**；`propagate_unknown=True` 时任何一个 UNKNOWN 都会要求确认。这个方向是对的：**安全判断尽量放在模型外面，用确定性规则兜底**。

三条 rail 当然不够。它挡的是最常见的"下载即执行、写裸盘、删根目录"，挡不住业务逻辑层面的越权。真要做企业级，还得在这个框架上补自己的策略，而不是只装一个 LLM 分析器。

### 10.1 环境边界

Workspace 层提供了 local、Docker、Apptainer、cloud、remote API 几种实现。`DockerWorkspace`（`openhands-workspace/openhands/workspace/docker/workspace.py:52`）会：

- 启动 `ghcr.io/openhands/agent-server:latest-python` 镜像（`:85`）；
- 把容器的 8000 端口绑定到 `127.0.0.1:<随机端口>`（`:190-220`），默认不暴露到外网；
- 等健康检查，默认超时 120 秒（`:124`）；
- 工作目录默认 `/workspace`。

这是"环境"这一层的正确姿势：agent 的 shell 不在宿主上跑，而是在一个可替换的容器里跑。但它只保证**隔离**，不保证**最小权限**——容器里能不能访问网络、挂载了哪些 volume、以什么用户跑，仍然要单独配。

## 11. 事件溯源的代价

事件溯源不是免费的。把 OpenHands 的选择摊开，能看到五笔账：

1. **写放大**。一个事件一个文件，500 步通常就是上千个文件；每次 append 都要拿锁、写文件、更新长度标记。它适合"单会话几十到几百步"的产品形态，不适合每秒几千次工具调用的高吞吐场景。
2. **文件系统假设**。`EventLog` 的 docstring 自己写了：`flock` 在 NFS 或网络文件系统上不可靠，共享存储要换协调机制。这不是实现偷懒，是本地文件锁的边界。
3. **schema 兼容**。`Event` 是 `extra="forbid"` 的 Pydantic 模型，但没有显式的 `schema_version` 字段。事件日志是系统的**事实来源**，这意味着旧事件模型要长期保持可加载；一旦改字段名或语义，历史会话就可能读不出来。事件日志是一份数据合同，不是内部缓存。
4. **隐私**。日志里会有用户输入、文件内容、工具输出。`ConversationState.create` 的 `cipher` 参数可以加密 secrets，不传就**脱敏丢失**；但普通事件内容不会自动脱敏。生产环境要把事件日志当敏感数据管理，而不是当 debug 日志。
5. **分支的成本**。`fork` 会深拷贝事件（`local_conversation.py:841`），`from_event_id` 只拷贝到某个事件的 `path_to_root`。这让"从第 N 步开一个 what-if 分支"很容易，但每条分支都是一份独立的历史，存储和清理要单独设计。

还有一笔隐性账：**调试**。arXiv 2602.07150 跑了 60,000 条 SWE-bench Verified 轨迹，发现单次 pass@1 会摆动 **2.2–6.0 个百分点**，T=0 下标准差仍 **>1.5 pp**；arXiv 2609.26826 用 28,801 次 trial、**105,933 美元**审计 639 个 Terminal-Bench 任务。没有事件级 trace，你无法判断一个 2 个百分点的提升是模型变强，还是抽样噪声。事件日志在这里不是架构洁癖，是测量仪器。

所以它适合：长会话、有人工确认、多客户端、需要审计和恢复的产品。  
不适合：一次性的 benchmark harness、追求极致吞吐的批处理、或者团队根本没有运维事件存储的打算。

一句话：**事件溯源是一个数据库决策，不是一个 agent 技巧。**

## 12. 四套 harness 放在一起

| 维度 | Codex CLI | Gemini CLI | mini-swe-agent | OpenHands SDK |
|---|---|---|---|---|
| 形态 | 本地 Rust CLI | 平台型 TS CLI | 190 行 Python | SDK + server + workspace |
| 循环闭合 | `run_turn` 内 | CLI 层重新 submit | agent 内 `while` | `LocalConversation._run` |
| 终止 | `needs_follow_up` + token 预算 | 5 道闸门，`MAX_TURNS=100` | exit 消息 + 步数/成本/墙钟 | 500 step/run + 美元预算 + stop hook |
| 持久化 | rollout/JSONL | chat recording | 无（stateless） | 每事件一个 JSON + `base_state.json` |
| 确认 | 审批策略 + 沙箱 | 策略引擎 + 确认总线 | 无 | pending action + 可插拔策略 |
| 并发 | 每工具 opt-in + RwLock | 模型声明 + 连续批 | 单工具 | agent 级开关 + 资源锁 |
| 上下文 | 摘要/预算压缩 | 50% 触发压缩 | 无压缩 | View + 事件化 condenser |
| 沙箱 | 本地三级 sandbox | 策略 + 沙箱 | Docker/Singularity | Docker/Apptainer/remote |
| 最适合 | 本地 coding CLI | 平台型 CLI/IDE | benchmark/研究 | 可托管、可审计的 agent 产品 |

这张表里没有"谁更先进"。四套系统在四个不同的约束下做优化：Codex 优化本地交互和安全，Gemini 优化平台治理，mini 优化变量控制，OpenHands 优化**会话的可恢复性和可观察性**。

## 13. 最小可运行：30 行看懂事件树

这一篇不新增脚本，因为 event log 的核心是数据结构，不是算法。下面这段可以直接粘进 Python：

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class E:
    id: str
    parent: str | None
    kind: str
    tool: str | None = None
    action_id: str | None = None

log = [
    E("e1", None, "user"),
    E("e2", "e1", "action", "terminal"),
    E("e3", "e2", "observation", action_id="e2"),
    E("e4", "e3", "action", "file_editor"),  # 进程在这里崩了
]

by_id = {e.id: e for e in log}

def path_to_root(leaf: str) -> list[E]:
    out, cur = [], leaf
    while cur is not None:
        e = by_id[cur]
        out.append(e)
        cur = e.parent
    return out[::-1]

def pending(events: list[E]) -> list[E]:
    seen = {e.action_id for e in events if e.kind == "observation"}
    return [e for e in events if e.kind == "action" and e.id not in seen]

branch = path_to_root("e4")
print("branch:", " -> ".join(e.id for e in branch))
print("pending:", [e.id for e in pending(branch)])

# 从 e3 开一个 what-if 分支：e5 和 e4 共享同一个 parent
log.append(E("e5", "e3", "action", "terminal"))
by_id["e5"] = log[-1]
print("fork:", " -> ".join(e.id for e in path_to_root("e5")))
```

输出：

```text
branch: e1 -> e2 -> e3 -> e4
pending: ['e4']
fork: e1 -> e2 -> e3 -> e5
```

三个输出对应三件事：

1. `path_to_root` 返回的是 root-first，和 SDK 的 `path_to_root` 一致；
2. `e4` 没有 observation，所以是 pending——这就是 OpenHands 恢复时重放的依据；
3. `e5` 和 `e4` 共享 `e3`，所以从 `e3` 分叉时它们是兄弟；在 SDK 里 `fork(from_event_id="e3")` 会拷贝 `e3` 的 `path_to_root` 作为新会话的起点。

## 源码与论文

- 源码基线：[`OpenHands/software-agent-sdk@2f2cd2b`](https://github.com/OpenHands/software-agent-sdk/tree/2f2cd2bf7f19ca3bbf9f0c4a95c60f48094302a1)（v1.59.0，2026-11-04）
- OpenHands 平台论文：[arXiv 2407.16741](https://arxiv.org/abs/2407.16741)
- SWE-agent / ACI：[arXiv 2405.15793](https://arxiv.org/abs/2405.15793)
- Compaction Laundering：[arXiv 2610.29869](https://arxiv.org/abs/2610.29869)
- Agent 提前放弃的测量：[arXiv 2606.29718](https://arxiv.org/abs/2606.29718)
- 单次运行方差：[arXiv 2602.07150](https://arxiv.org/abs/2602.07150)
- Terminal-Bench 任务审计：[arXiv 2609.26826](https://arxiv.org/abs/2609.26826)

## 14. 哪些能抄，哪些不能

| 设计 | 能不能抄 | 说明 |
|---|---|---|
| 事件日志 + `base_state` 分离 | 能 | 配置和历史分开，恢复时可以换模型、换 prompt，不换历史 |
| pending action 事件化 | 能 | 确认状态跨进程恢复，比内存态可靠 |
| 每事件一个文件 | 谨慎 | 可寻址、可分支，但文件数和锁开销大；高吞吐换 segmented log/DB |
| 500 step/run + 美元预算 | 能 | 两个边界互补；注意预算是 conversation 累计，不是单 run |
| stop hook 驳回完成 | 能 | 把"完成"从模型手里拿走一部分，适合长任务 |
| 先 nudge 再 STUCK | 能 | 成本低、体验好；但只抓机械重复 |
| 资源级并行锁 | 能 | 比全局锁快，比裸并发安全；前提是工具声明正确 |
| `LLMSecurityAnalyzer` | 不能单独用 | 它读模型自评，不是安全边界；只能做 UX 提示 |
| 确定性 policy rails | 能 | no model/no network，适合做第一道闸；但要补业务规则 |
| 事件日志无 schema version | 不能 | 事实来源必须有版本和迁移策略 |

## 小结

- OpenHands 把会话做成了**事件日志 + 派生视图**：EventLog 是不可变事实，View 是给模型的投影，Condensation 是"我忘了哪些、摘要是什么"的又一条事件。
- 恢复靠 `base_state.json` + `events/`：工具不能删，模型/prompt/condenser 可以换；pending action 让确认状态跨进程存活。
- 循环有三层终止：模型的 `finish`、stop hook 的驳回、harness 的 500 step/run 与美元预算。500 是 per run，预算是 conversation 累计。
- stuck detector 先 nudge 再判 STUCK，阈值 4/3/3/6，只看最近 20 个事件；它能抓重复，抓不住"每次换一种错法"。
- 工具并发默认 1，调大后靠资源 key + 排序锁；文件只读也锁，终端池化后端主动放弃锁。
- 事件溯源解决"事实留下痕迹"，不解决"投影保留来源"：environment 反馈在 prompt 里就是 user 消息。Compaction Laundering 论文把 OpenHands 列进 flattening 组，正是这个口子。
- 默认安全是关闭的：`NeverConfirm` + 无 analyzer；示例里的 `LLMSecurityAnalyzer` 读模型自评，只能当 UX，不能当边界。

下一篇离开 harness，进入评测：SWE-bench、Terminal-Bench、$\tau$-bench 到底在测什么，pass@k 和 pass^k 为什么会在 k=10 时讲出相反的故事，以及"仪器错误"如何让 leaderboard 失真。
