---
title: "Harness 解剖（一）：Codex CLI 的三层循环、沙箱与 apply_patch"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 5
series: ["agent"]
tags: ["AI Agent", "系列", "Codex", "Harness", "Sandbox", "apply_patch", "源码分析"]
categories: ["训练框架"]
---

前三篇讲的是抽象机制。从这一篇开始，拆真实的 harness。

第一个是 OpenAI Codex CLI。选它有三个理由：它是开源的；它是 Rust 写的，类型和模块边界清楚；它在 2026 年长出了一整套子系统（code mode、guardian、hooks、memories、skills），刚好可以看一个 harness 如何从"一个循环"演化成"一个运行时"。

源码固定在这个 checkout：`openai/codex@f2b94ef8b52196f127d162c6db71e390b9bb265c`（2026-11-04）。所有行号以它为准。

> 一个前提：这个树里**没有 `codex.rs`**。老的 monolith 已经拆成 `core/src/{session,tasks,tools,context}` 加一堆独立 crate（`codex-tools`、`codex-sandboxing`、`codex-apply-patch`、`codex-code-mode*`、`codex-hooks`、`codex-guardian-*`）。你在网上看到的旧文章，路径大概率对不上。

## 1. 先记住一个画面

把 Codex 想成一家**有门禁的工厂**：

- **销售**（`RegularTask`）接单，把任务交给车间；
- **车间主任**（`run_turn`）循环派活：模型说下一步做什么，他就安排机器（工具）去做，再把结果贴回看板；
- **机器**（工具）在**安全围栏**（sandbox）里跑，围栏有三级：只读、可写工作区、完全放开；
- **质检**（guardian）是一个独立的 LLM，在危险动作放行前看一眼；
- **仓库管理员**（compaction）在原料堆满之前，把旧料打包成一张摘要，腾出地方。

这条产线的每个环节都有代码边界，也都有"出事了怎么办"的分支。这一篇沿着一次用户请求走一遍。

## 2. 一次 turn 的完整数据流

dossier 里把调用链梳理成了一条线，我把它压缩成七步：

```text
用户输入
  → Session::start_task::<RegularTask>()          tasks/mod.rs:291
  → RegularTask::run                              tasks/regular.rs:51
  → RegularTask::run_turns（外层重启循环）          regular.rs:136,150
  → run_turn（agent 主循环）                       session/turn.rs:182, loop :498
      ├─ 取 pending input / 刷新 step_context
      ├─ clone_history → build_prompt（工具定义在这里注入）
      ├─ run_sampling_request（传输重试循环）        turn.rs:1834, retry :1877
      │    └─ try_run_sampling_request（流消费）     turn.rs:2831, :3002
      │         → OutputItemDone
      │            → handle_output_item_done        stream_events_utils.rs:327
      │               → ToolRouter::build_tool_call
      │               → ToolCallRuntime::handle_tool_call（tokio::spawn）
      │         → Completed{end_turn}
      │            → needs_follow_up = !end_turn
      ├─ 需要压缩 → run_auto_compact → continue
      └─ 不需要 follow-up → stop hooks → break
```

这里有**三层嵌套循环**，顺序很重要：

1. **任务重启循环**（`run_turns`）：只有当用户在模型运行中插话、留下 pending input 时才重新进入 `run_turn`。
2. **agent 循环**（`run_turn`）：模型 ↔ 工具的迭代。
3. **传输重试循环**（`run_sampling_request`）：API 失败重试，不改变对话状态。

大部分 harness 只有第 2 层。第 1 层是"可打断"的实现方式：不取消当前 turn，而是把用户的新输入排队，等当前 turn 结束再开一个。第 3 层是"网络不可靠"的实现方式：重试不能重复执行工具。

## 3. 没有 max turns

这是 Codex 最值得讨论的设计。`run_turn` 的循环条件只有一个：`needs_follow_up`。

```rust
let needs_follow_up = model_needs_follow_up || has_pending_input;
...
if !needs_follow_up {
    break;
}
```

（`session/turn.rs:644`、`:758`、`:865-867`。）

`model_needs_follow_up` 来自两处：只要有工具调用就为真（`stream_events_utils.rs:379`），或者模型返回 `end_turn: false`（`:3485`）。

**没有 `max_turns` 变量。** 整个树里唯一的 `RECAP_HISTORY_MAX_TURNS = 8` 是 TUI 回看历史用的，和循环无关。

为什么敢？`turn.rs:715` 的注释是答案：

> "as long as compaction works well in getting us way below the token limit, we shouldn't worry about being in an infinite loop."

翻成工程语言：**循环的边界不是步数，是 token 预算**。每一轮采样后都会算一次 `context_window_token_status`（`session/context_window.rs:31`），超过 `auto_compact_scope_limit + fallback_buffer_tokens` 就触发压缩（`:98-105`）；压缩失败、请求还是超限，循环会因为 API 错误结束。

这个设计有优点也有风险：

- 优点：不会因为一个"步数上限"把正在收敛的长任务砍断；对 SWE 这种可能跑几十轮的任务更友好。
- 风险：如果压缩逻辑有 bug，或者模型陷入"调工具 → 压缩 → 再调工具"的循环，账单会一直涨，直到 token 上限或用户取消。

Codex 用 `Stop` hook 和 `guardian` 来补这个风险：Stop hook 可以在模型想结束时强制再跑一轮（`turn.rs:758-805`），guardian 可以在危险动作前插入一次独立审查。但**没有硬性的步数熔断**，这是它和 mini/Gemini 最大的哲学差异。

## 4. 工具：注册、并行、截断

### 4.1 两层结构

Codex 的工具分两层：

- `codex-rs/tools/src`：**契约层**，定义 `ToolExecutor`、`ToolExposure`（6 类可见性）、`supports_parallel_tool_calls()`。
- `core/src/tools/`：**宿主层**，`registry.rs`（注册/派发）、`router.rs`（解析 tool call）、`spec_plan.rs`（生成模型可见的 spec）、`parallel.rs`（并发控制）、`orchestrator.rs`（审批 + 沙箱 + 重试）、`handlers/`（57 个具体工具）。

模型看到的工具列表在 `build_tool_router`（`spec_plan.rs:122`）里冻结成 `model_visible_specs`，再由 `build_prompt` 注入（`turn.rs:1793-1808`）。"冻结"很重要：一次采样请求里的工具列表必须和这次请求的 tool call 解析用同一份，否则模型调用的工具可能在列表里已经不存在了。

### 4.2 并行：API 说可以，工具说不行

`build_prompt` 里 `parallel_tool_calls: true` 是**无条件**的（`turn.rs:1807`）。但真正决定并发的是每个工具的 `supports_parallel_tool_calls()`，默认 `false`（`tools/src/tool_executor.rs:122-124`）。

只有少数工具覆写成 `true`：`exec_command`、`write_stdin`、`view_image`、`tool_search`、`current_time`、`get_context_remaining`、`list_mcp_resources` 等。`apply_patch` 和 `shell` **没有**覆写。

运行时用一把读写锁（`tools/parallel.rs:287-299`）：

```rust
let guard = if supports_parallel {
    lock.read().await      // 多个读锁可以并发
} else {
    lock.write().await     // 写锁独占
};
```

所以：N 个 `exec_command` 可以并发；任何一个 `apply_patch` 会把整批工具串行化。这个设计很务实——文件修改天然有竞争，读命令没有。

还有一个容易忽略的细节：工具 handler 是**先 spawn 再等**的。`handle_output_item_done` 收到 `function_call` 就 `tokio::spawn`，future 塞进 `FuturesOrdered`（`turn.rs:3239`）；等模型流结束，再 `drain_in_flight` 统一收口（`:3684`）。所以工具执行和模型继续生成是重叠的，这也是 Codex 首 token 延迟比较低的原因之一。

### 4.3 截断：历史问题，不是执行问题

工具输出不是"执行完就截断"，而是"写进 history 时按预算截断"。预算跟着 tool call 走（`ToolCall::response_byte_budget`），在转成模型 payload 时应用（`core/src/tools/context.rs:246-256`）：

```rust
truncate_function_output_payload(
    &mut payload,
    with_serialization_allowance(self.truncation_policy),
    estimate_audio_token_count,
);
```

`with_serialization_allowance` 把预算乘 **1.2**，给序列化和 header 留余量（`utils/output-truncation/src/lib.rs:25-27`）。截断后会加一行 header：`Warning: truncated output (original token count: N)`（`:39-45`）。

配置项是 `tool_output_token_limit`（`core/src/config/mod.rs:977-978`）。exec 输出还有第二层预算：取模型 `max_output_tokens` 和 turn 截断策略的较小值（`context.rs:526-534`），然后循环收缩直到渲染后的文本装得下（`:568-590`）。

这套设计的关键点是：**截断发生在"给模型看"的边界，而不是"工具执行"的边界**。原始输出可以完整写进 session/rollout，模型只看到预算内的版本。这和第 4 篇讲的 "session ≠ context window" 是同一个思想。

## 5. 沙箱：三条轴，一个收口

Codex 的权限模型有三条正交的轴：

| 轴 | 取值 | 定义位置 |
|---|---|---|
| **沙箱模式** | `read-only` / `workspace-write` / `danger-full-access` | `protocol/src/config_types.rs:104-116` |
| **审批策略** | `untrusted` / `on-request` / `granular` / `never` | `protocol/src/protocol.rs:1163-1187` |
| **具体策略** | `ReadOnly{network}` / `WorkspaceWrite{writable_roots, network}` / `DangerFullAccess` | `protocol.rs:1250-1275` |

`granular` 把审批拆成五个独立开关：`sandbox_approval`、`rules`、`skill_approval`、`request_permissions`、`mcp_elicitations`（`protocol.rs:1192-1210`）。这是 2026 年才有的细化——早期只有"问/不问"。

后端按平台选（`sandboxing/src/manager.rs:48-62`）：

- macOS → Seatbelt（`sandboxing/src/seatbelt.rs`，配 `.sbpl` 策略文件）；
- Linux → bubblewrap + seccomp + Landlock（`linux-sandbox/src/{bwrap,seccomp,landlock}.rs`）；
- Windows → Restricted Token（opt-in）或 MXC。

**唯一的执行收口是 `ToolOrchestrator::run`**（`core/src/tools/orchestrator.rs:125`）。它的注释把流程写死了：

```text
// 1) Approval
// 2) First attempt under the selected sandbox
```

审批要求由 `default_exec_approval_requirement` 决定（`tools/sandboxing.rs:200-215`）：

```rust
let needs_approval = match policy {
    AskForApproval::Never => false,
    AskForApproval::OnRequest | AskForApproval::Granular(_) =>
        matches!(file_system_sandbox_policy.kind, FileSystemSandboxKind::Restricted),
    AskForApproval::UnlessTrusted => true,
};
```

注意 `OnRequest` 的语义：**只有受限文件系统才需要审批**。如果已经是 `danger-full-access`，就没有"再问一次"的意义。

沙箱拒绝后的升级路径很窄（`orchestrator.rs:464-470`）：只有工具声明 `wants_no_sandbox_approval`，或者 `OnRequest` + 有网络审批上下文，才允许重试；而且**如果策略里有被拒绝的读路径，禁止升级**（`tools/sandboxing.rs:294-298`）：

```rust
pub(crate) fn unsandboxed_execution_allowed(policy: &FileSystemSandboxPolicy) -> bool {
    !policy.has_denied_read_restrictions()
}
```

这句话的意思是：你可以把"写"放开，但不能把"不许读的秘密"放开。比如策略里写了 `~/.ssh` 只读禁止，那么即使升级到无沙箱，也不能执行——否则一次 `cat ~/.ssh/id_rsa` 就把边界破了。

### 5.1 execpolicy：Starlark 写的前缀规则

除了沙箱，Codex 还有一层**命令级策略**（`codex-rs/execpolicy`）。规则是 Starlark 语法：

```starlark
prefix_rule(pattern = ["git", "push"], decision = "prompt",
            justification = "pushing to a remote requires confirmation")
prefix_rule(pattern = ["rm", ["-rf", "-r"]], decision = "forbidden")
```

决策有四种：`Allow` / `Prompt` / `PromptUser` / `Forbidden`（`execpolicy/src/decision.rs:9-21`）。匹配按第一个 token，支持 `host_executable()` 的 basename 回退。绑定在 `ExecPolicyManager::create_exec_approval_requirement_for_command`（`core/src/exec_policy.rs:337`、`:391-399`）。

这层规则和沙箱是互补的：沙箱管"能碰到什么"，execpolicy 管"这条命令本身该不该跑"。`rm -rf /` 在 workspace-write 沙箱里也可能造成巨大破坏，所以需要命令级禁止。

## 6. apply_patch：为什么不是 shell 写文件

Codex 的文件修改只有一个工具：`apply_patch`。它是一个 **freeform/custom tool**，不是 JSON：

```text
*** Begin Patch
*** Update File: src/app.py
@@ def greet(name):
-    return "hi " + name
+    return f"hi {name}"
*** End Patch
```

工具定义里明确写了 "This is a FREEFORM tool, so do not wrap the patch in JSON"（`core/src/tools/handlers/apply_patch_spec.rs:5-28`），语法用 Lark 描述（`core/assets/tools/apply_patch.lark:1-18`）。

### 6.1 为什么不直接 `shell: echo > file`

三个理由，代码里都能找到证据：

1. **token 更省**。改一行只需要上下文 + 一行 `-` + 一行 `+`，不需要重写整个文件。
2. **权限更窄**。`handlers/apply_patch.rs:229-278` 的原生运行时只给"被修改文件的父目录"授写权限；如果走 `exec_command`，就是整个 shell 的写权限。代码里还留着 `LegacyApplyPatchExecCommandWarning`，说明这是从 shell 迁移过来的。
3. **可审计**。patch 是结构化的，能生成 diff、能回滚、能做 turn diff tracker。shell 重定向是一串副作用，很难还原"改了哪些文件"。

### 6.2 四段式模糊定位

补丁里的上下文不一定和文件完全一致：行尾多了空格、缩进不同、文件里用的是印刷体破折号。Codex 的 `seek_sequence` 做四轮匹配（`apply-patch/src/seek_sequence.rs:4-9`）：

1. 精确匹配；
2. 忽略行尾空白；
3. 忽略两端空白；
4. **Unicode 标点归一**：`—` → `-`、`“”` → `"`、NBSP → 空格（`:78-105`）。

第四轮是很多 patch 工具没有的，但它解决一个真实问题：模型生成的补丁用 ASCII，源文件里有排版用的 Unicode 标点。没有这一轮，补丁会莫名其妙地"找不到上下文"。

### 6.3 非原子

多个 hunk 是按文件顺序应用的。一个文件内的多个 chunk 先在内存里算出新内容再写；**跨文件不是原子的**：第一个文件写成功、第二个文件找不到上下文时，第一个文件的修改已经落盘。错误里会带 `AppliedPatchDelta` 和 `exact: bool`，告诉调用方"改到哪了、是否精确"（`apply-patch/src/lib.rs:457-467`）。

这个设计是故意的：宁可留下部分修改 + 明确报告，也不做"全有或全无"的事务回滚。因为 agent 的下一步通常是根据"哪里失败了"继续修，而不是从头再来。

## 7. 压缩：两种哲学

Codex 有两套压缩实现：

- **模型摘要**（`core/src/compact.rs`）：发一个 summarization prompt，把历史换成"最近用户消息 + 摘要"。
- **token 预算压缩**（`compact_token_budget.rs`）：不调模型，直接开一个新窗口。它仍然走 compaction 的生命周期（hooks、`ContextCompaction` item 都会触发），只是不做摘要。由 `Feature::TokenBudget` 选择（`turn.rs:212-213`）。

模型摘要的细节上一篇已经讲过：保留 ≤20K token 的真实用户消息、丢掉 assistant/tool 历史、放一条 summary。这里补一个容易忽略的点：**压缩本身也可能溢出**。`compact.rs:358-368` 的处理是"从最前面删一条历史再重试"，注释写着 "Trim from the beginning to preserve cache (prefix-based)"——压缩的时候也在照顾 prefix cache。

## 8. AGENTS.md：项目指令的加载顺序

Codex 的项目指令文件是 `AGENTS.md`，加载规则在 `core/src/agents_md.rs:1-21`：

1. 从 cwd 向上找，直到遇到 `project_root_markers`（默认 `.git`）；
2. 从项目根到 cwd，逐层收集 `AGENTS.md`，拼接；
3. **不越过项目根**。

文件名优先级：`AGENTS.override.md` > `AGENTS.md` > `project_doc_fallback_filenames`。拼接分隔符是 `\n\n--- project-doc ---\n\n`（`:43-49`）。总字节有预算（`max_total`），超了截断并警告（`:155-166`）。

线程级指令还有一个 10K token 的硬上限，而且**超了是拒绝，不是截断**（`agents_md_manager.rs:190-200`）。这个选择值得注意：项目指令是"规则"，截断规则可能比没有规则更危险，所以宁可报错让用户改。

## 9. 2026 年长出来的子系统

Codex 这一年的演化，很能说明"harness 会变成运行时"。挑五个：

### 9.1 code-mode：一个 `exec` 工具，一个 V8 isolate

前面工具篇讲过。这里补一个架构细节：V8 是**独立进程**（`code-mode-host` 走 gRPC），不是嵌在 core 里。`v8_init.rs:49-56` 里能看到 `v8::V8::initialize_platform` 和 `v8_enable_sandbox` feature。每个 cell 有 `CodeModeSession`，可以 `store`/`load` 跨 cell 持久化，输出默认 10,000 token。

### 9.2 guardian：让另一个 LLM 审查工具调用

`core/src/guardian` 是一个**独立的同步 reviewer session**，有自己的 prompt、输入预算、请求预算。它在危险动作放行前跑一次，输出"允许/拒绝/需要人确认"。`guardian-context` crate 负责给它挑证据（根对话、可信用户回答、保留的用户指令、发送者消息、对话 transcript）。

这是一个很值得关注的模式：**用 LLM 做权限系统的一部分**。它比纯规则灵活，但也引入了新的攻击面（reviewer 自己可能被注入）和成本（每次审查都是一次模型调用）。Codex 的处理是给 reviewer 单独的预算和隔离的 session source，不让它继承主 agent 的 skills 目录。

### 9.3 hooks：14 个生命周期事件

`hooks/src/lib.rs:29-44` 定义了 14 个事件：`PreModelRequest`、`PostModelRequest`、`PreToolUse`、`PermissionRequest`、`PostToolUse`、`PreCompact`、`PostCompact`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`SubagentStart`、`SubagentStop`、`Stop`、`Interrupt`。

每个 hook 可以是 `Sync` 或 `Async`，作用域是 `Thread` 或 `Turn`。`PostModelRequest` 的同步 hook 甚至可以**延迟工具派发**（`turn.rs:2885`）；`Stop` hook 可以阻止结束、强制再跑一轮（`:758-805`）。这让 Codex 可以被外部策略"接管"一部分控制流，而不需要改 core。

### 9.4 memories / skills / agent-roles

- **memories**：读写分离的 crate，session 启动时跑 Phase 1/Phase 2 管线，把记忆写成文件，再在后续 session 注入。门控在 `Feature::MemoryTool`，且非 root agent 不跑（`memories/write/src/start.rs:24-38`）。
- **skills**：`SKILL.md` + YAML frontmatter 的 loader，支持依赖声明和 policy 门控（`skills/src/parser.rs:22-33`、`skills/src/metadata.rs:11-16`）。
- **agent-roles**：TOML 角色文件 = 配置层 + 描述 + 昵称池（`agent-roles/src/agent_role_config.rs:22-41`）。

### 9.5 agent-message-board：agent 之间的"邮件"

`agent-message-board-client` 是一个 session 作用域的 typed client，turn 循环可以在采样中途被"邮件"抢占（`turn.rs:3248-3281`，受 `Feature::DeferMailboxPreemption` 控制）。这是多 agent 协作的底层传输，也是"主 agent 可以被 worker 打断"的实现。

## 10. 最小可运行：apply_patch

`code/agent/apply_patch.py` 实现了 Codex 补丁语法的子集：Add/Delete/Update、Move、多 chunk、四段模糊匹配、非原子失败。

```bash
cd code/agent
python3 apply_patch.py --demo
```

输出：

```text
changes: ['A hello.py', 'M hello.py']
def hello():
    return "hello"
```

`--test` 里覆盖了几个关键行为：

- 精确匹配、行尾空白、Unicode 破折号都能命中；
- 找不到上下文时，错误里带具体上下文；
- 第一个文件成功、第二个文件失败时，第一个文件的修改**已经落盘**——非原子是可见的。

## 11. 哪些能抄，哪些不能

| 设计 | 能不能抄 | 说明 |
|---|---|---|
| 三层循环 | 能 | 可打断、可重试、可恢复，是生产级 harness 的基本盘 |
| 无 max turns + token 预算 | 谨慎 | 需要可靠的压缩和监控；否则账单失控 |
| 工具并发用 RwLock | 能 | 比"API 说并行就并行"安全得多 |
| freeform apply_patch | 能 | 省 token、窄权限、可审计；但要接受解析复杂度 |
| 四段模糊匹配 | 能 | 尤其是 Unicode 归一，成本很低 |
| 沙箱 + execpolicy 双层 | 能 | 一个管"碰什么"，一个管"跑什么" |
| guardian LLM 审查 | 谨慎 | 有效但增加成本/攻击面，适合高风险动作 |
| code-mode V8 | 谨慎 | 省 token 明显，但沙箱和可观测性成本高 |

Codex 的很多设计是**为它自己的产品形态**服务的：本地 CLI、单用户、强交互、需要审批。把它照搬到云端多租户 agent，沙箱和审批的假设会变，guardian 的成本模型也会变。

下一篇看另外两个极端：Gemini CLI 用近 70 万行 TypeScript（含测试）做了一套事件驱动的 harness，mini-swe-agent 用 190 行 Python 证明"循环本身可以极简"。两者放在一起，能看清"复杂度守恒"这件事。
