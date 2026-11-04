---
title: "Harness 解剖（二）：Gemini CLI 的近 70 万行与 mini-swe-agent 的 190 行"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 6
series: ["agent"]
tags: ["AI Agent", "系列", "Gemini CLI", "mini-swe-agent", "Harness", "源码分析", "SWE-bench"]
categories: ["训练框架"]
---

上一篇的 Codex 是"重运行时"：Rust、沙箱、guardian、hooks、code mode。这一篇看两个极端：

- **Gemini CLI**：TypeScript，`packages` 下近 70 万行（含测试），`packages/core/src` 几百个文件，事件驱动、策略引擎、子 agent、浏览器 agent、A2A 远程 agent。
- **mini-swe-agent**：Python，核心循环 190 行，只有一个 bash 工具，没有压缩、没有子 agent、没有沙箱抽象。

把两者放在一起，是为了回答一个问题：**一个 agent harness 的复杂度，到底花在哪了？**

## 1. 先记住一个画面

Gemini CLI 像一家**有完整组织架构的公司**：前台（CLI）接需求，项目经理（client）排期，车间（scheduler）按工序执行，法务（policy engine）审合规，质检（loop detector）抓返工，还有外派团队（subagents）和远程分部（A2A）。

mini-swe-agent 像一个**只有一把螺丝刀的师傅**：你说问题，他 `bash` 一下，看结果，再 `bash` 一下。没有组织，没有流程，只有"命令 → 输出 → 下一条命令"。

两家都能修东西。区别是：公司能接大项目、能合规、能协作，但沟通成本高；师傅便宜、透明、好调试，但遇到需要审批、需要并行、需要长期记忆的任务就抓瞎。

## 2. Gemini CLI：循环在 core 外面

Gemini CLI 最反直觉的设计是：**`packages/core` 自己不闭合 agent loop**。

它分四层：

| 层 | 文件 | 职责 |
|---|---|---|
| **GeminiChat** | `core/geminiChat.ts` | 持有 `HistoryTurn[]`，强制角色交替、修复 tool call 配对，暴露 `sendMessageStream` |
| **Turn** | `core/turn.ts:270-414` | 把流式 chunk 转成 typed event（Content/Thought/ToolCallRequest/Finished） |
| **GeminiClient** | `core/client.ts:945` | 压缩、溢出检查、循环检测、模型路由、`turn.run` |
| **CoreToolScheduler** | `scheduler/scheduler.ts` | 校验 → 策略 → 确认 → 执行 → 生成 `functionResponse` |

工具执行完，`CompletedToolCall.responseParts` 里是 `functionResponse` part。谁来把它送回模型？**CLI 层**：`packages/cli/src/ui/hooks/useGeminiStream.ts:2305-2356` 把 parts 收拢，作为一条**新的 user turn** 重新 `submitQuery`。

所以从 core 的视角看，"工具结果"就是"用户又说了一句话"。这个设计的好处是 core 不需要知道工具怎么执行；坏处是"一次用户请求"在 CLI 层被拆成了多次 `submitQuery`，循环的终止条件散落在多层。

### 2.1 递归，而不是 while

`GeminiClient.processTurn`（`client.ts:649`）处理一轮。如果流结束后没有 pending tool call，它会做一次 "next speaker check"；判定模型还应该继续说，就递归调用 `sendMessageStream`，把 `boundedTurns - 1` 传下去（`:929-938`）。

`MAX_TURNS = 100`（`client.ts:80`），在 `:995` 取 `Math.min(turns, MAX_TURNS)`。递归深度受这个预算约束。除了它，还有四道闸门：

- 会话级 `getMaxSessionTurns()`（`:659-666`）；
- `LoopDetectionService`：同一工具调用（name + JSON 参数的 SHA-256）重复 5 次、内容 chunk 重复 10 次、或 LLM judge 0.9 置信度判定绕圈（`services/loopDetectionService.ts:29-30,65,174-178`）；
- `ContextWindowWillOverflow`：发请求前估算 token，超了直接返回（`client.ts:744-750`）；
- next speaker check 的递归。

五道闸门叠在一起，是"宁可误停，不可失控"的哲学。和 Codex 的"无 max turns、靠 token 预算"正好相反。

### 2.2 工具调度：并行是模型声明的

Gemini 的工具声明会被统一打一个补丁：每个工具加一个 `wait_for_previous` 参数（`tools/tools.ts:546-573`）。也就是说，**并行不是 harness 根据"只读/可写"推断的，而是模型显式声明的**。

`isParallelizable` 只硬性排除编辑类工具和 `update_topic`（`utils/tool-utils.ts:29-41`）。调度器只把**连续的**可并行调用打成一批（`scheduler.ts:519-529`），而且执行前有一个"所有 active call 都 ready"的屏障（`:554-566`）：

```typescript
const allReady = activeCalls.every(
  (c) => c.status === CoreToolCallStatus.Scheduled || this.isTerminal(c.status),
);
if (allReady && scheduledCalls.length > 0) {
  const execResults = await Promise.all(scheduledCalls.map((c) => this._execute(c, signal)));
}
```

后果是：**一个慢 shell 会挡住整批工具**，即使其他工具早就 ready 了。这不是 bug，是"确认流程必须串行"的代价——只要有一个工具在等用户确认，整批都不能往下走。

### 2.3 确认与策略：七种确认结果，三层规则

确认走 `MessageBus`（EventEmitter）的 request/response，用 `correlationId` 配对（`confirmation-bus/message-bus.ts:15`、`scheduler/confirmation.ts:60-90`）。确认结果有七种，包括 `ProceedAlways`（分 AndSave/Server/Tool 三种范围）和 `ModifyWithEditor`（`tools/tools.ts:1106-1114`）。

审批模式有四档：`PLAN` / `DEFAULT` / `AUTO_EDIT` / `YOLO`（`policy/types.ts:52-66`）。策略引擎是一个 TOML 规则引擎（`policy/policy-engine.ts:242`），决策三分：`allow` / `deny` / `ask_user`（`policy/types.ts:13-17`）。

两个细节很值得抄：

- **非交互式 session 默认 DENY**（`policy-engine.ts:299-301`）。CI 里跑 agent，没人能点"允许"，所以默认拒绝，而不是默认放行。
- **规则有来源层级**：default / user / admin，编码成 `tier + priority/1000`（`toml-loader.ts:47-50`）。admin 策略可以覆盖 user，user 可以覆盖 default。这是企业部署必须的。

还有一个容易忽略的坑：`isClientInitiated` 会把 `ASK_USER` **静默升级成 `ALLOW`**（`scheduler/policy.ts:82-92`）。也就是说，某些客户端发起的调用不会真的问用户。这个行为在写策略时必须知道，否则你以为"所有写操作都会问"，实际不会。

### 2.4 上下文：1M 窗口，50% 触发压缩

Gemini 的默认窗口是 **1,048,576 token**，Gemma 4 是 256K（`core/tokenLimits.ts:20-38`）。压缩在 **0.5×limit** 触发，保留最近 **3 个 turn**（`context/chatCompressionService.ts:45-50`）。文档说压缩"设计上对信息是无损的"——当然，这只是目标，不是保证。

2026 年的树里还有一个更新的 `ContextManager.renderHistory` 管线（`client.ts:678-713`），返回 `{history, apiHistory, baseUnits}`，支持 token ground truth 回传。这说明 Gemini 也在从"压缩"走向"上下文管理管线"，和 Anthropic 的方向一致。

项目记忆是 `GEMINI.md` + `MEMORY.md`，向上走到 `~/.gemini`（`tools/memoryTool.ts:11-12`、`utils/memoryDiscovery.ts:457-506`），支持 JIT 子目录加载（`:512`）。`@file` 引用在 CLI 层处理（`atCommandProcessor.ts:699`），不在 core。

### 2.5 子 agent 与浏览器 agent

子 agent 的默认限制是 **30 turns / 10 分钟**，没调 `complete_task` 就报 `ERROR_NO_COMPLETE_TASK_CALL`（`agents/types.ts:24-56`、`local-executor.ts:373-382`）。子 agent 对主 agent 来说就是一个同名工具，走同一套调度。

浏览器 agent 只能通过 `invoke_agent` 进入，有硬编码的域名白名单，页面内容被标记为不可信（`agents/browser/browserAgentDefinition.ts:11-13,58-76`）。这是一个很务实的设计：**浏览器是最大的注入面，所以给它单独的策略边界**，而不是让主 agent 直接开一个 browser 工具。

## 3. mini-swe-agent：190 行的反面

`src/minisweagent/agents/default.py` 全文 190 行。核心循环上一篇已经贴过，这里看它**故意不做什么**：

| 能力 | mini | Gemini |
|---|---|---|
| 多工具 schema | 无，只有 bash | 20+ 内置工具 |
| 持久 shell / cwd / env | 无，每条命令新 subshell | 有 session |
| 压缩 | 无 | 有，50% 触发 |
| 子 agent / MCP / 策略引擎 | 无 | 有 |
| 循环检测 | 无 | 有，三种规则 |
| 结构化事件日志 | 无，trajectory = messages | 有 |
| 沙箱抽象 | 无，local/docker/singularity 三选一 | 有 policy + sandbox |

它保留的只有四件事：

1. **模型调用**：`litellm.completion(model, messages, tools=[BASH_TOOL])`（`litellm_model.py:64-126`）。
2. **动作解析**：从 tool call 里取 `bash` 的 `command`；没有 tool call 就抛 `FormatError`。
3. **环境执行**：`subprocess.Popen(shell=True, start_new_session=True)`，超时 `killpg`（`environments/local.py:24-91`）。
4. **终止**：stdout 第一行 `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` + 返回码 0 → `Submitted`，其余 stdout 是 submission（`local.py:45-56`）。

### 3.1 一个被低估的细节：成本补记

`FormatError` 分支手动把 `extra.cost` 加回去（`default.py:100-102`）。原因是：模型调用已经计费，但解析失败时 `query()` 还没来得及把 cost 累加。这个 bug 在"按调用次数计费"的 API 上不明显，在"按 token 计费"的 API 上会让成本账偏低。

mini 的作者显然被这个坑咬过，所以注释写得很直白：

> "The call was billed before parsing failed, so query() never got to charge it."

### 3.2 SWE-bench 的 patch 是"提示词约定"出来的

mini 的 SWE-bench runner（`run/benchmarks/swebench.py`）不做 patch 提取。它要求模型自己跑：

```bash
git diff > patch.txt
```

然后用哨兵命令把内容 cat 出来，harness 只负责把 `preds.json` 写成 SWE-bench 格式（`:97-108`）。真正的评测在外部 `swebench.harness.run_evaluation` 里跑（`docs/usage/swebench.md:130`）。

这是一个很"mini"的选择：**不替模型做决定，把约定写进 prompt，让模型自己遵守**。代价是模型可能忘记写 patch，或者 patch 不完整；好处是 harness 极简，任何模型都能接。

### 3.3 Docker 环境的一个坑

SWE-bench 的官方镜像用 `bash -lc` 会激活错误的 conda 环境，mini 改成 `bash -c` + `BASH_ENV=/root/.bashrc`（`docker.py:74-124`）。这是一行改动，但会直接影响 benchmark 数字。它说明：**benchmark 分数不只取决于模型，还取决于环境怎么启动 shell**。

## 4. 复杂度守恒

把两个 harness 放在一起，会发现一个"复杂度守恒"：

```text
总复杂度 ≈ harness 代码 + prompt 约定 + 模型能力
```

- Gemini 把复杂度放在 **harness 代码**里：策略引擎、调度器、压缩、子 agent、确认总线。换来的是可治理、可审计、可扩展。
- mini 把复杂度放在 **prompt 约定 + 模型能力**里：一个 bash 工具、一段工作流提示、一个哨兵。换来的是 190 行、任何模型可接、任何环境可跑。

这不是"谁更先进"，而是"你把复杂度放在哪，决定了谁能维护它"。

- 平台团队做产品 → 复杂度放 harness，因为你要支持很多用户、很多策略、很多模型。
- 研究团队做 benchmark → 复杂度放 prompt，因为你要控制变量，harness 越薄，模型差异越干净。
- 企业做内部 agent → 复杂度放策略层，因为合规和安全不能靠模型自觉。

## 5. 一个可复用的对比表

| 维度 | Codex | Gemini CLI | mini-swe-agent |
|---|---|---|---|
| 语言/规模 | Rust，~100 crate | TypeScript，core 数百文件 | Python，190 行 |
| 循环闭合 | core 内 `run_turn` | CLI 层重新 submit | agent 内 `while` |
| 终止 | `needs_follow_up` + token 预算 | 5 道闸门（100 turns 等） | exit 消息 + 步数/成本/墙钟 |
| 工具并发 | 每工具 opt-in + RwLock | 模型声明 + 连续批 + 屏障 | 单工具，无并发 |
| 上下文 | 摘要/预算压缩，20K 用户消息 | 50% 触发，保留 3 turn | 无压缩 |
| 安全 | 沙箱 + execpolicy + guardian | policy 引擎 + 确认总线 + 沙箱 | 无（靠 docker/singularity） |
| 扩展 | hooks / skills / code mode | 子 agent / MCP / A2A | 无 |
| 适合 | 本地 coding CLI | 平台型 CLI/IDE | benchmark / 研究 |

## 6. 最小可运行

这一篇没有新脚本，但可以用 `code/agent/loop_sim.py` 做两个练习：

1. 把 `max_steps` 改成 100，把剧本改成"模型每步都调 `read_file`"，观察循环怎么被步数上限拦住——这是 Gemini 的 `MAX_TURNS` 模型。
2. 去掉 `max_steps`，改成"只有 `finish` 才退出"，观察没有上限时会发生什么——这是 Codex 的模型，区别是 Codex 还有 token 预算兜底。

```bash
cd code/agent
python3 loop_sim.py --max-steps 2     # 观察 max_steps
python3 loop_sim.py --test           # 看四个断言
```

## 小结

- Gemini CLI 的 core 不闭合循环，CLI 层把工具结果作为新 user turn 重新提交；终止靠五道闸门，最硬的是 100 turns 和循环检测。
- 它的并行是"模型声明 + 连续批 + 全 ready 屏障"，一个慢工具会挡住整批；确认走 MessageBus，策略是三层 TOML 规则。
- mini-swe-agent 用 190 行证明：循环、预算、错误处理可以极简；复杂度被推给了 prompt 和模型。
- 复杂度守恒：harness 代码、prompt 约定、模型能力，三者此消彼长。选型时先问"这个复杂度应该由谁维护"。
- benchmark 数字对环境细节极其敏感：`bash -lc` vs `bash -c` 就能改变 SWE-bench 结果。

下一篇离开 harness，进入评测：SWE-bench、Terminal-Bench、$\tau$-bench 到底在测什么，pass@k 和 pass^k 为什么会在 k=10 时讲出相反的故事，以及"仪器错误"如何让 leaderboard 失真。
