---
title: "AI Agent 系统工程（专题）：模型、Harness 与环境的三体问题"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 1
series: ["agent"]
tags: ["AI Agent", "系列", "Harness", "Context Engineering", "评测", "开源代码"]
categories: ["LLM算法"]
---

同一个模型，在三个 harness 里跑出三个分数——这不是模型能力的差异，而是系统工程的问题。

2026 年，agent 已经不再是一个"提示词技巧"。它是一条完整的流水线：模型在循环里调用工具，harness 负责攒上下文、执行工具、挡住危险操作，环境提供文件系统、浏览器、终端和测试。任何一段出问题，最终都表现为"模型不行"。

这个专题把这条流水线拆开，按 13 篇的篇幅讲清楚三件事：

- **loop 怎么闭合**：一条消息如何变成工具调用，工具结果如何回到模型，循环什么时候停。
- **context 怎么流动**：窗口、压缩、重置、记忆、缓存，以及信息在哪一步被丢掉。
- **正确性怎么保证**：评测仪器、沙箱、权限、注入防御，以及多智能体和 RL 的账本。

## 先记住一个画面

把 agent 想成一个 7×24 小时的工程团队，但每个工程师只上一班，**下班就失忆**。

下一班的人推门进来，只看到三样东西：上一班留下的代码、一份进度文件、和一条 git 历史。他必须靠这三样东西在 5 分钟内判断"现在做到哪了、下一步做什么、有没有留下烂摊子"。

Anthropic 在 2025 年底的 long-running harness 实验里，就是被这个画面卡住的：给 Claude 一个"做一个 claude.ai 克隆"的高层提示，它会试图一口气做完，做到一半窗口爆了，下一班接手时只看到半成品，于是花大量时间把项目修回能跑的状态；或者反过来，看到已经有一些功能，就宣布"做完了"。

解法不是更大的窗口，而是**换一种工作方式**：第一个 agent 负责初始化——写一份 200 多项的 feature list（全部标成 failing）、写 `init.sh`、写 `claude-progress.txt`、做第一次 git commit；之后的 agent 每班只做一个 feature，做完必须端到端验证、提交、更新进度文件。窗口还是会满，但每一班交接时，环境是干净的。

这个专题关心的就是这些"交接"发生在哪里：模型与工具之间、harness 与沙箱之间、这一次采样与下一次采样之间、这个 agent 与那个 agent 之间。

> **类比的边界**：把 agent 比作轮班工程师很好用，但有一处关键不同——人类工程师下班后记忆还在，agent 的记忆**只存在于你显式写下的东西里**。所以"进度文件"不是辅助文档，它是 agent 的一部分。凡是没写进 context 或文件系统的状态，下一班一律视为不存在。

## 三个部件

2026 年 4 月，Anthropic 在 Managed Agents 的工程博客里给了一个很干净的拆分：

| 部件 | 是什么 | 谁负责 |
|---|---|---|
| **模型** | 读 context、产出文本和工具调用 | 模型厂商 |
| **Harness** | 循环、上下文管理、工具路由、审批、压缩、重试 | 应用/框架 |
| **环境** | 文件系统、shell、浏览器、测试、网络、沙箱 | 基础设施 |

三者之间是**接口**，不是实现。同一个模型可以换 harness，同一个 harness 可以换沙箱，同一个 session 可以换模型。这个拆分最重要的推论是：

> **harness 里写下的每一条假设，都会随模型变强而过期。**

Anthropic 给了两个具体例子。Sonnet 4.5 有"context anxiety"——接近它以为的窗口上限时会提前收尾，所以 harness 需要"context reset"（清空窗口 + 结构化交接）。换到 Opus 4.5，这个行为消失了，reset 变成了纯粹的额外开销和延迟，于是他们把它删掉。另一个例子是 evaluator：在模型能力边界附近，独立的 QA agent 能抓出真问题；当任务落进模型舒适区，evaluator 就只剩成本。

这条"假设会过期"的规律，是理解过去两年所有 agent 框架演化的钥匙。也是这个专题反复要问的问题：**你这个 harness 里的哪一部分，是在补模型当前做不到的事？模型升级后，它还成立吗？**

## 四本账

Agent 系统的复杂度，最后都落在四本账上。后面每一篇都会回到这四本账。

### 账本一：context

窗口不是免费的。Anthropic 把"token 越多、召回越差"的现象叫 **context rot**：所有模型都有，只是程度不同；原因是 transformer 的 $n^2$ 注意力和训练分布里长序列更稀少。于是 context 不是"越大越好"，而是一份**会边际递减的注意力预算**。

2026 年 10 月的一篇论文把这个账算得更细。作者在一个真实工具型 agent 上跑了 720 次：最大的一次 prompt 只用了窗口的 **5.75%**，但 **11,189 条工具结果里有 3,827 条（约三分之一）在到达模型前被截断**，而且每一条被截断的结果都变成了坏 JSON。更刺眼的是同一个结果里的两个字段：一个字段只有 **4.7%** 的调用能到达模型，另一个字段是 **99.3%**——决定故障证据在哪个字段，纯属运气。

所以第一篇要建立的直觉不是"窗口有多大"，而是"**context 是一条流水线，每一站都可能丢东西**"。

### 账本二：时延

在线 agent 看两个数：TTFT（首 token 时间）和每一步的墙钟时间。Agent 的时延结构和 chat 不同：一次用户请求可能触发 20 次模型调用、50 次工具调用，每次工具调用又是一次网络或进程往返。

这解释了一个反直觉的现象：**更少的工具调用不一定更快**。如果一次工具调用能替代五次模型往返，它通常更快，哪怕这次工具本身慢一点。工具设计的第一目标不是"原子"，而是"减少往返"。

### 账本三：成本

Anthropic 的实测：普通 agent 的 token 消耗约为 chat 的 **4 倍**，多 agent 系统约为 **15 倍**。多 agent 的性能提升也是真的：Opus 4 做 lead、Sonnet 4 做 subagent，在他们内部的 research eval 上比单 agent Opus 4 高 **90.2%**；而 token 使用量单独解释了 BrowseComp 上 **80%** 的性能方差。

这两句话必须一起读：多 agent 很多时候是"用钱买 token，再用 token 买性能"。它划不划算，取决于任务价值，以及这些 token 是不是花在了有效探索上。

2026 年 7 月 Cursor 的 swarm 实验给了一个更细的账本：同样的 SQLite 复刻任务，planner 用 Opus 4.8、worker 用 Composer 2.5 的混合配置花了 **1,339 美元**，全部用 GPT-5.5 花了 **10,565 美元**，质量都在 4 小时内接近或达到 100%。差别不在并行度，而在**每个 token 花在哪个 role 上**。

### 账本四：可靠性

Agent 的评测不能只看一次成功率。$\tau$-bench 在 2024 年就提出了 **pass^k**：同样的任务跑 k 次，每一次都成功的概率。一个每次 75% 成功率的 agent，跑 3 次全对的概率是 $0.75^3 \approx 42\%$，跑 10 次是 **5.6%**。

这就是为什么"demo 很惊艳"和"能上线"之间隔着一整个可靠性工程。第 8 篇会讲，2026 年的一篇论文甚至发现：公开 leaderboard 上 13 个百分点的差距，来源不是模型，而是 grader 没有给一半的 episode 打分。

## 三个问题

整个专题围绕三个问题展开，每一篇都在回答其中一个或几个：

1. **循环怎么闭合？** 模型输出如何变成动作，动作结果如何变成观察，什么时候停，停了以后状态放在哪。
2. **context 怎么组织？** 什么进窗口、什么留在外面、什么压缩、什么重置、什么写进文件、什么进缓存。
3. **正确性怎么保证？** 工具调用是否真的执行了、评测是否真的测到了、权限是否真的挡住了、模型升级后 harness 是否还成立。

## 阅读地图

| # | 篇目 | 一句话 |
|---|---|---|
| 1 | [导读：模型、Harness 与环境的三体问题]({{< relref "agent-00-index" >}}) | 三个部件、四本账、十三篇的路线图 |
| 2 | [Agent Loop 解剖]({{< relref "agent-01-loop" >}}) | 从 ReAct 到 tool calling，循环的复杂度在终止条件而不在 `while` |
| 3 | [工具层：schema、MCP 与 Code Mode]({{< relref "agent-02-tools" >}}) | 工具接口是一份上下文压缩契约 |
| 4 | [Context Engineering]({{< relref "agent-03-context" >}}) | 预算、压缩、重置与信息衰减 |
| 5 | [Harness 解剖（一）：Codex CLI]({{< relref "agent-04-codex" >}}) | 三层循环、无 max turns、沙箱与 apply_patch |
| 6 | [Harness 解剖（二）：Gemini CLI 与 mini-swe-agent]({{< relref "agent-05-gemini-mini" >}}) | 近 70 万行与 190 行，复杂度守恒 |
| 7 | [Harness 解剖（三）：OpenHands 的事件溯源与可恢复会话]({{< relref "agent-06-openhands" >}}) | 事件日志、500 步熔断、资源锁与投影 |
| 8 | [评测：SWE-bench、Terminal-Bench 与 Harness 混淆]({{< relref "agent-06-eval" >}}) | pass@k、pass^k，以及仪器错误 |
| 9 | [Computer-use 与浏览器 Agent]({{< relref "agent-07-cua" >}}) | 动作空间、grounding 与恢复 |
| 10 | [多智能体：context 隔离、handoff 与 token 倍率]({{< relref "agent-08-multiagent" >}}) | 多 agent 是上下文隔离术，不是并行术 |
| 11 | [Agentic RL：环境、reward 与 GSPO]({{< relref "agent-09-rl" >}}) | 环境工程是瓶颈，verifier 质量就是 reward 质量 |
| 12 | [记忆与安全：从 MemGPT 到 CaMeL]({{< relref "agent-10-memory-security" >}}) | 安全边界必须在模型外面 |
| 13 | [2026 论文地图与十个开放问题]({{< relref "agent-11-trends" >}}) | 从哪里继续读，以及还没解决什么 |

## 代码与版本

配套代码在 [`code/agent/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/agent)，全部纯标准库、CPU 可跑、带 `--test`：

| 脚本 | 对应文章 | 做什么 |
|---|---|---|
| `loop_sim.py` | 第 2 篇 | 剧本模型 + 并行工具 + 错误观察 + 步数上限 |
| `mcp_rpc.py` | 第 3 篇 | JSON-RPC 2.0 的 MCP 握手、tools/list、tools/call |
| `token_budget.py` | 第 4 篇 | 三种截断策略下的证据保留率与 JSON 合法性 |
| `apply_patch.py` | 第 5 篇 | Codex 风格补丁语法、四段模糊定位、非原子失败 |
| `passk.py` | 第 8 篇 | pass@k / pass^k / Wilson 区间 / grader bug |
| `gui_ground.py` | 第 9 篇 | 0–1000 归一化坐标、缩放误差、点击命中率与 IoU |
| `multiagent_cost.py` | 第 10 篇 | 单 agent、多 agent、chat 三种 token 账 |
| `grpo.py` | 第 11 篇 | GRPO 的 token 比率 vs GSPO 的序列比率 |
| `taint.py` | 第 12 篇 | 逐动作授权 vs 全局 taint 检查 |

源码引用固定在四个 checkout 上（行号以这些版本为准）：

- OpenAI Codex CLI，`openai/codex@f2b94ef8b52196f127d162c6db71e390b9bb265c`（2026-11-04）
- Google Gemini CLI，`google-gemini/gemini-cli@89e9d25d67e7dc7cb0deb84ad2c3474b7a392d31`（2026-11-03）
- mini-swe-agent，`SWE-agent/mini-swe-agent@04d809ceab9df28f9adaed044884180159172930`（2026-09-03）
- OpenHands SDK，`OpenHands/software-agent-sdk@2f2cd2bf7f19ca3bbf9f0c4a95c60f48094302a1`（2026-11-04，v1.59.0）

论文和博客的日期都标在正文里。这个领域半年就换一批术语，读的时候请把日期一起看。

## 主要来源

大厂工程博客（都标了发布日期）：

- Anthropic, *Building effective agents*（2024-12）：<https://www.anthropic.com/research/building-effective-agents>
- Anthropic, *Effective context engineering for AI agents*（2025-09）：<https://www.anthropic.com/research/effective-context-engineering-for-ai-agents>
- Anthropic, *How we built our multi-agent research system*（2025-06）：<https://www.anthropic.com/research/multi-agent-research-system>
- Anthropic, *Code execution with MCP*（2025-11）：<https://www.anthropic.com/research/code-execution-with-mcp>
- Anthropic, *Advanced tool use*（2025-11）：<https://www.anthropic.com/research/advanced-tool-use>
- Anthropic, *Harness design for long-running agents*（2026-03）：<https://www.anthropic.com/research/harness-design-long-running>
- Anthropic, *Managed Agents*（2026-04）：<https://www.anthropic.com/research/managed-agents>
- Anthropic, *Claude Code sandboxing*（2025-10）：<https://www.anthropic.com/research/claude-code-sandboxing>
- Anthropic, *Demystifying evals for AI agents*（2026-01）：<https://www.anthropic.com/research/demystifying-evals-for-ai-agents>
- Anthropic, *Building a C compiler with a team of parallel Claudes*（2026-02）：<https://www.anthropic.com/research/a-c-compiler>

开源代码：

- OpenAI Codex CLI：<https://github.com/openai/codex>
- Google Gemini CLI：<https://github.com/google-gemini/gemini-cli>
- mini-swe-agent：<https://github.com/SWE-agent/mini-swe-agent>
- OpenHands SDK：<https://github.com/OpenHands/software-agent-sdk>
- MCP 规范 2025-06-18：<https://modelcontextprotocol.io/specification/2025-06-18>

Benchmark：

- SWE-bench：<https://www.swebench.com/>
- Terminal-Bench：<https://www.tbench.ai/>

关键论文：

- *OpenHands: An Open Platform for AI Software Developers as Generalist Agents*：<https://arxiv.org/abs/2407.16741>
- *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering*：<https://arxiv.org/abs/2405.15793>
- *Context Is a Pipeline, Not a Window*：<https://arxiv.org/abs/2610.36114>
- *Silent Instrument Errors Reorder Agent Leaderboards*：<https://arxiv.org/abs/2611.04399>
- *Compaction Laundering*：<https://arxiv.org/abs/2610.29869>
- *On Randomness in Agentic Evals*：<https://arxiv.org/abs/2602.07150>
- Agent 提前放弃的测量：<https://arxiv.org/abs/2606.29718>
- Terminal-Bench 任务审计：<https://arxiv.org/abs/2609.26826>
- *Local Allow, Global Deny*：<https://arxiv.org/abs/2610.27620>
- *Is Complex Agentic Memory Always Needed?*：<https://arxiv.org/abs/2610.30755>
- *Defeating Prompt Injections by Design*（CaMeL）：<https://arxiv.org/abs/2503.18813>
- *Group Sequence Policy Optimization*（GSPO）：<https://arxiv.org/abs/2507.18071>
- *SWE-bench Pro*：<https://arxiv.org/abs/2509.16941>
- *UI-TARS-2*：<https://arxiv.org/abs/2509.02544>

## 谁适合读

- 写过 ReAct / function calling demo，但不知道为什么上线后不稳的人。
- 在做 agent 框架、eval、RL 环境，想知道大厂 harness 到底怎么搭的人。
- 被"多 agent / 更长 context / 更强模型"三种叙事绕晕，想要一本账的人。

这个专题不提供"最佳框架"。它提供的是拆解方法：拿到任何一个 agent 系统，你能在三分钟内指出它的 loop 在哪、context 在哪丢、评测在测什么、安全边界在哪。

下一篇从最小的地方开始：一个 `while` 循环，怎么把一条消息变成一次工具调用。
