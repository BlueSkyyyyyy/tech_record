---
title: "2026 论文地图与十个开放问题"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 13
series: ["agent"]
tags: ["AI Agent", "系列", "论文地图", "开放问题", "趋势", "综述"]
categories: ["随笔"]
---

这个专题从 loop 讲到 context，从 Codex 讲到 Gemini，从评测讲到安全。最后一篇做三件事：给一张论文地图，列十个还没解决的问题，以及给不同读者一条继续往下读的路线。

先说结论：**2026 年的 agent 研究，重心已经从"能不能做"转向"能不能信"。**

- 2023–2024 年，大家在比 SWE-bench 分数、比 demo；
- 2025 年，开始讲 context engineering、multi-agent、agentic RL；
- 2026 年，最热的论文是评测仪器、权限组合、缓存成本、记忆覆盖度——全是"系统可靠性"问题。

这不是因为能力停止进步，而是因为**能力已经够用，但不可靠**。一个 80% 成功率的 agent，在 demo 里很惊艳，在生产里每天会错几次；要把它变成 99%，需要的工作量和从 0 到 80% 差不多。

## 先记住一个画面

2026 年底，三个团队在同一个会议上碰头：

- 做 harness 的团队说："我们把循环、压缩、沙箱都调好了，模型还是不稳。"
- 做 eval 的团队说："你们的分数不可比，因为 grader 把一半 episode 记成了失败。"
- 做安全的团队说："你们的 agent 每个动作都合规，连起来却把私钥发了出去。"

三个人说的是同一件事：**单点能力已经够用，系统可靠性还没到位。** 这一篇的地图，就是按"可靠性"这条线重新排的。

## 1. 一张论文地图

按主题分组，每篇一句话。arXiv ID 可以直接搜。

### 1.1 Context / Harness

| 论文 | ID | 一句话 |
|---|---|---|
| *Context Is a Pipeline, Not a Window* | 2610.36114 | 720 次真实 agent 运行：窗口只用 5.75%，三分之一工具结果被截断，坏 JSON |
| *Cache Engineering for Agents* | 2610.28577 | 压缩会破坏 KV cache 前缀，re-prefill 很贵；用 cache drop 替代文本删除 |
| *Do Coding Agents Follow AGENTS.md?* | 2611.04367 | 86 个任务、4 种措辞、4 个 harness-model 组合：条件句最不被遵守 |
| *Silent Instrument Errors Reorder Agent Leaderboards* | 2611.04399 | 13 个点的差距来自未评分 episode，不是模型 |
| *Local Allow, Global Deny* | 2610.27620 | 逐动作授权不能组合；关系型违规是 2-safety hyperproperty |
| *The Devil is in the Context* | 2611.04672 | Prompt injection 有九个维度，benchmark 覆盖不全，13% 组合没测 |
| *Compaction Laundering* | 2610.29869 | 压平 transcript 会把伪造的用户批准"洗"进记忆：5 个 harness 87–100%，角色保真的 4 个 0% |
| *On Randomness in Agentic Evals* | 2602.07150 | 60,000 条轨迹：单次 pass@1 摆动 2.2–6.0 个点，T=0 仍有 >1.5 个点标准差 |
| Terminal-Bench 任务审计 | 2609.26826 | 639 个任务、28,801 次 trial、10.6 万美元：125 个全失败任务里只有 78 个是真没解 |

### 1.2 评测

| 论文 | ID | 一句话 |
|---|---|---|
| SWE-bench | 2310.06770 | 2,294 个真实 GitHub issue，原始最佳 1.96% |
| SWE-bench Pro | 2509.16941 | 1,865 题、41 仓库，长程任务，防饱和 |
| $\tau$-bench | 2406.12045 | 对话 + 规则 + 数据库状态，提出 pass^k |
| $\tau^2$-bench | 2506.07982 | 双控制：用户也能改共享状态 |
| OSWorld | 2404.07972 | 369 个真实桌面任务，人类 72.36% vs 模型 12.24% |
| AgentBench | 2308.03688 | 8 个环境，揭示跨环境能力差异 |
| CrossSWE | 2611.01484 | 171 个跨仓库任务、165 仓库、13 语言；最佳 58.48% |
| *AI Agents That Matter* | 2407.01502 | 只看 accuracy 会选出又贵又脆的 agent |

### 1.3 记忆 / 安全

| 论文 | ID | 一句话 |
|---|---|---|
| *Is Complex Agentic Memory Always Needed?* | 2610.30755 | Raw-RAG 在 7 个设置里 6 个第一，覆盖度 > 利用率 |
| AgentDojo | 2406.13352 | 97 任务、629 安全用例，动态注入评测 |
| CaMeL | 2503.18813 | 可信 query 定控制流 + capability 控数据流，77% vs 84% 可证明安全 |
| *Local Allow, Global Deny* | 2610.27620 | 同上，安全监控需要跨执行视图 |

### 1.4 Agentic RL

| 论文 | ID | 一句话 |
|---|---|---|
| SWE-RL | 2502.18449 | 软件演化数据 + 规则 reward，Llama3-70B 41.0% Verified |
| GSPO | 2507.18071 | 序列级比率 + clip，稳 MoE，Qwen3 在用 |
| verl / HybridFlow | 2409.19256 | 单/多控制器混合，解耦 rollout 与 training |
| SWE-smith | 2504.21798 | 任意 Python 仓库自动造环境，50k 实例/128 仓库 |
| Kimi K2 | 2507.20534 | 1T/32B MoE，τ² 66.1，SWE Verified 65.8 |
| GLM-4.5 | 2508.06471 | 355B/32B MoE，TAU 70.1%，SWE Verified 64.2 |

### 1.5 Computer-use

| 论文 | ID | 一句话 |
|---|---|---|
| UI-TARS | 2501.12326 | 端到端 GUI agent，OSWorld 24.6（50 步）超 Claude 22.0 |
| UI-TARS-2 | 2509.02544 | 数据飞轮 + 多轮 RL，OSWorld 47.5，AndroidWorld 73.3 |
| WebArena | 2307.13854 | 812 任务、4 域，最佳 GPT-4 agent 14.41% vs 人类 78.24% |
| GAIA | 2311.12983 | 466 题，人类 92% vs GPT-4+插件 15% |
| BrowseComp | 2504.12516 | 1,266 题，持久浏览找难找的信息 |

## 2. 十个开放问题

### 2.1 评测仪器怎么才可信

**问题**：分数不只取决于模型，还取决于 scaffold、grader、环境、成本核算。2026 年的审计发现 13 个点的差距可以完全来自"未评分 = 失败"。

**为什么难**：agent 的评测是"执行式"的，环境和 grader 本身就是系统的一部分；换一个 grader，排名就变。

**现在到哪**：Terminal-Bench 把 agent/model/cost 分开列；SWE-bench 要求提交 scaffold；但大部分 leaderboard 仍然只报一个数。2026 年的两项审计更狠：639 个 Terminal-Bench 任务里，125 个"全失败"只有 78 个是真没解（2609.26826）；单次运行的 pass@1 方差有 2.2–6.0 个点（2602.07150）。

**怎么下手**：在自己的 eval 里同时报"原评/重评"、分开"未评分/失败"、记录 grader 版本、固定环境镜像。

### 2.2 长程任务怎么不崩

**问题**：OSWorld 任务几十步，真实工作流几百步；错误会累积，context 会满，模型会"提前收尾"或"宣布完成"。

**为什么难**：长程任务的 reward 稀疏、状态空间大、恢复策略复杂；而且"做到一半"的状态很难用文本完整表达。

**现在到哪**：Anthropic 的 initializer + progress file + feature list；Cursor 的 planner/worker/reviewer 分层；Codex 的 compaction + token 预算。

**怎么下手**：把"进度"外化成文件/JSON，不要依赖模型记忆；每班结束必须端到端验证；用 git 做检查点。

### 2.3 记忆：结构 vs 覆盖度

**问题**：复杂记忆系统（摘要、结构化、学习检索）不一定比 raw-history RAG 强；后者在 7 个设置里 6 个第一。

**为什么难**：记忆的价值取决于任务：事实召回、关系推理、时间演化，需要不同的结构；而"覆盖度"和"利用率"是 trade-off。

**现在到哪**：Raw-RAG baseline 被重新重视；provenance、置信度、遗忘策略还没有标准。

**怎么下手**：先做 raw-history dense retrieval + rerank 的 baseline，再逐步加结构；每次加结构都要证明它带来的收益 > 丢失的覆盖度。

### 2.4 多 agent 的协调成本

**问题**：多 agent 能提升广度任务，但协调成本、冲突、重复探索会吃掉收益；Cognition 说并行 subagent "非常脆弱"。

**为什么难**：agent 的决策是隐式的，只存在文件/代码状态里；共享摘要会丢决策，共享完整轨迹又太贵。

**现在到哪**：Cursor 用 planner/worker/reviewer + 多层 review，把冲突从 70,000 降到 <1,000；Anthropic 用"lead 只读摘要"控制 context。

**怎么下手**：worker 无状态、产物落盘、lead 只读摘要；先证明任务可切分、结果可合并、冲突可解决，再上多 agent。

### 2.5 Agentic RL 的环境与 reward

**问题**：环境要可复现、可重置、可验证、可并行、低成本；verifier 的偏差会被 RL 放大。

**为什么难**：真实环境（SWE-bench、网页、桌面）本身就不稳定；reward hacking 的手段比防御多。

**现在到哪**：SWE-smith 这类"自动造环境"的管线；GSPO 稳 MoE；Kimi K2/GLM-4.5 的联合 RL。

**怎么下手**：先把环境做稳（固定镜像、seed、超时），再谈算法；verifier 要对抗性验证（独立重评、人工抽查、held-out）。

### 2.6 安全的组合性

**问题**：逐动作授权不能组合成全局授权；关系型违规没有单轨迹责任点；LLM judge 在单动作视图下几乎全部放行。

**为什么难**：这是 2-safety hyperproperty，需要跨执行视图；而大多数 guardrail 只看单次调用。

**现在到哪**：CaMeL 用 capability + 控制流隔离，给出可证明安全；但牺牲灵活性，且需要重新写工具。

**怎么下手**：先做数据流标签（private/untrusted/public），再做 sink 策略；高风险动作加跨执行监控；不要把"审批弹窗"当成数据流控制。

### 2.7 GUI agent 的 grounding 与恢复

**问题**：4K 屏幕缩到 1280，1 像素误差放大 3 倍；点错了往往无声；恢复策略比规划更缺。

**为什么难**：界面动态、跨应用、延迟高；评测需要执行式环境，成本高。

**现在到哪**：UI-TARS-2 到 OSWorld 47.5；set-of-marks / 混合 DOM 提升 grounding；`recovery_rate` 还很少被报。

**怎么下手**：分层截图（全屏 → 区域 → 元素）；动作后验证状态；checkpoint/rollback；把 recovery rate 作为一等指标。

### 2.8 成本与延迟的经济学

**问题**：多 agent 约 15× chat token；Cursor 的混合配置 1,339 美元 vs 10,565 美元；延迟决定产品形态。

**为什么难**：成本取决于任务价值、模型价格、缓存命中、重试策略；没有通用的"每任务成本"模型。

**现在到哪**：Terminal-Bench 列 cost/tokens；Anthropic 报 4×/15×；Cursor 报 planner/worker 分工。

**怎么下手**：把成本按 role 拆（规划/执行/审查），把 token 按"有效探索/重复/协调"拆；优化 context efficiency，而不是单纯减 agent 数。

### 2.9 可观测性与审计

**问题**：agent 的动作、决策、数据流散在多个进程/服务里；出了问题很难还原"为什么这么做"。

**为什么难**：trajectory 不是传统日志，包含模型内部推理、工具调用、环境状态；隐私和体积都难处理。

**现在到哪**：append-only session log、OpenTelemetry 式的 trace、Codex 的 rollout、Claude Code 的 checkpoint。

**怎么下手**：每个动作记录 provenance（输入来自哪、输出到哪、谁批准的）；区分 agent action / grader action；保留原始工具输出，context 里只放精简版。

### 2.10 人机协作的接口

**问题**：agent 什么时候该问、什么时候该自己做？审批太多会烦，太少会出事。

**为什么难**：这是"信任校准"问题，因人、因任务、因风险而异；而且用户很难判断 agent 的置信度。

**现在到哪**：权限模式（plan/default/auto-edit/yolo）、sandbox 减少 84% 提示、guardian 审查。

**怎么下手**：按"可逆性 × 影响范围"分级；可逆的低风险动作自动执行，不可逆的高风险动作必须确认；给用户"一键回滚"。

## 3. 三条主线

如果只记三件事，记这三条：

### 3.1 从"模型能力"到"系统能力"

同一个模型，换 harness 可以差十几个点。Agent 的能力 = 模型 × harness × 环境。**优化 harness 的回报，往往比等下一个模型更高**。

### 3.2 从"更大窗口"到"更好的信息流"

窗口会继续变大，但 context rot 不会消失。真正的竞争力在"什么进窗口、什么留在外面、什么压缩、什么重置"。Context 是一条流水线，不是一个大桶。

### 3.3 从"能跑"到"可信"

评测仪器、权限组合、reward hacking、注入防御——2026 年最热的论文都在解决"怎么信"。一个 agent 要上线，需要的不是更高的 demo 分数，而是**可复现、可审计、可回滚、可证明的安全边界**。

## 4. 继续读的路线

### 4.1 如果你想做 agent 产品

1. 先读第 2、3、4 篇，把 loop、工具、context 的账算清楚；
2. 再读第 5、6、7 篇，看 Codex/Gemini/OpenHands 怎么处理终止、沙箱、审批；
3. 然后读第 8 篇，设计自己的 eval（pass^k、cost、grader）；
4. 最后读第 10、12 篇，决定要不要多 agent、怎么防注入。

### 4.2 如果你想做 agent 研究

1. 从 *AI Agents That Matter*（2407.01502）开始，建立"accuracy + cost"的评测观；
2. 读 $\tau$-bench（2406.12045）和 AgentDojo（2406.13352），理解 pass^k 和注入评测；
3. 读 GSPO（2507.18071）和 verl（2409.19256），理解 agentic RL 的算法与系统；
4. 读 2026 年的 "instrument errors"（2611.04399）和 "local allow, global deny"（2610.27620），看最前沿的问题。

### 4.3 如果你想做基础设施

1. 读 Codex 的 sandbox/orchestrator/execpolicy（第 5 篇）；
2. 读 Gemini 的 scheduler/policy engine（第 6 篇）；
3. 读 Anthropic 的 sandboxing、Managed Agents、code execution with MCP；
4. 读 Cache Engineering（2610.28577），理解压缩和 KV cache 的耦合。

## 5. 最后一句

Agent 不是"更聪明的聊天机器人"。它是一个**在真实环境里执行动作的系统**：会读文件、跑命令、点按钮、发邮件、改数据库。它的能力来自模型，但它的可靠性来自 harness、环境和评测。

这个专题拆了 13 篇，核心只有一句话：

> **先把循环闭合，再把 context 管好，最后把正确性证明给自己看。**

下一篇？没有下一篇了。但你可以从这里开始：挑一个你正在用的 agent，按第 8 篇的十问检查一遍它的评测，按第 12 篇的五层画一遍它的安全边界。你会发现，很多"模型不行"的问题，其实是系统工程的问题。

---

**专题目录**：[AI Agent 系统工程]({{< relref "agent-00-index" >}})

**配套代码**：[`code/agent/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/agent)
