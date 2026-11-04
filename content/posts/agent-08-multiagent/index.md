---
title: "多智能体：context 隔离、handoff 与 token 倍率"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 9
series: ["agent"]
tags: ["AI Agent", "系列", "Multi-Agent", "Orchestration", "Handoff", "上下文隔离", "成本"]
categories: ["LLM算法"]
---

多智能体是 agent 领域分歧最大的话题。2025–2026 年，三家给出了三种看起来互相矛盾的答案：

- **Anthropic**：多 agent 系统（Opus 4 lead + Sonnet 4 subagents）在内部 research eval 上比单 agent 高 **90.2%**，代价是约 **15 倍** token。
- **Cognition**：**不要构建多 agent**。并行 subagent 非常脆弱，"actions carry implicit decisions"，冲突的决策会导致坏结果。
- **Cursor**：swarm 能成立，但真正的杠杆不是并行，是 **context efficiency**；同一个 SQLite 任务，混合 planner/worker 花 **1,339 美元**，全用 GPT-5.5 花 **10,565 美元**，质量相当。

这三句话都成立，因为它们说的不是同一个东西。这一篇把它们拆开：多 agent 到底是什么、什么时候划算、账怎么算。

## 1. 先记住一个画面

一个研究任务："找出 S&P 500 信息技术板块所有公司的董事会成员。"

单 agent 的做法：搜一家、读页面、记下来、再搜下一家。它的 context 里会堆满 500 个页面的片段，越到后面越容易忘、越容易重复搜。

多 agent 的做法：lead 把任务拆成"按公司分组"，派 4 个 subagent 各查 100 家；每个 subagent 在自己的 context 里读页面、筛选、压缩，最后只回传一份"公司 → 董事名单"的表。lead 的 context 里只有 4 份表 + 自己的计划。

这就是多 agent 的本质：**用多个 context window 换一个更大的有效工作集**。它不是因为"多个 agent 更聪明"，而是因为"一个窗口装不下 500 个页面，但 4 个窗口可以"。

> **类比的边界**：多 agent 像 MapReduce，但有一个关键不同——MapReduce 的 map 函数是确定的，agent 的 map 是模型，可能跑偏、可能重复、可能给出互相冲突的结论。所以多 agent 的难点不在"怎么拆"，而在"怎么保证拆完还能合"。

## 2. 四种组织方式

### 2.1 Orchestrator-workers（编排者-工人）

一个 lead 负责规划、拆任务、汇总；N 个 worker 各自执行。Anthropic 的 research system 就是这种。

```text
lead: 我需要查 500 家公司的董事
  ├─ worker 1: 公司 1-100
  ├─ worker 2: 公司 101-200
  ├─ worker 3: 公司 201-300
  └─ worker 4: 公司 301-500
lead: 汇总、去重、校验
```

适合：任务可切分、子任务独立、结果可合并（研究、检索、批量分析）。
不适合：子任务之间有强依赖（改代码时 A 改了接口，B 还在用旧接口）。

### 2.2 Handoffs（交接）

不是"lead 派活"，而是"当前 agent 把控制权交给另一个 agent"。OpenAI Agents SDK 的实现很典型：handoff 被建模成一个工具，名字是 `transfer_to_<agent>`，带 `input_filter` 和 `on_handoff` 回调。

```text
triage agent → (handoff) → billing agent → (handoff) → refund agent
```

适合：客服分流、专家路由、多阶段流程。
关键设计：handoff 之后，**对话历史带不带过去**？带，新 agent 可能被旧 context 污染；不带，新 agent 可能缺少必要信息。OpenAI 的 `input_filter` 就是让你决定"传哪些消息"。

### 2.3 Agents-as-tools（agent 当工具）

一个主 agent 把另一个 agent 当成一个普通工具调用：给输入，拿输出，主 agent 的 context 不变。

这和 handoff 的区别是**控制权**：handoff 是"你接手"，agents-as-tools 是"你帮我算一下，算完还给我"。前者适合"换人"，后者适合"调专家"。

### 2.4 Swarm（群体）

没有中心 lead，大量 worker 在一个共享环境里工作，通过文件/消息板/任务队列协调。Cursor 的 swarm 是这种：planner 规划，worker 实现，reviewer 审查，通过 git 和冲突解决机制协调。

适合：大规模、可并行、有明确验证标准的任务（复刻 SQLite、批量修 bug、生成训练数据）。
难点：冲突、重复、错误累积。Cursor 的数据很能说明问题：旧 swarm 在 2 小时内产生 **68,000 次 commit**、**70,000+ 次冲突**，最热的文件被 **1,173 个 agent** 碰过、冲突 **7,771 次**；新 swarm 4 小时冲突不到 **1,000**，最热文件冲突 **47**。

## 3. 为什么多 agent 会"更聪明"：token 账

Anthropic 的分析很直接：在他们内部的 research eval 上，三个因素解释了 **95%** 的性能方差，其中 **token 使用量单独解释 80%**，另外两个是工具调用次数和模型选择。

换句话说：**多 agent 的提升，很大一部分是"花了更多 token 去探索"**。这不是贬义——如果任务价值足够高，多花 token 换覆盖率是理性的。但它意味着两件事：

1. 多 agent 的收益有上限：当 token 不再是瓶颈，继续加 agent 的边际收益会下降。
2. 多 agent 的失败模式常常是"token 花了，但花在重复/无效探索上"。

Anthropic 的实测倍数：

| 形态 | 相对 chat 的 token |
|---|---|
| 普通 agent | ~4× |
| 多 agent | ~15× |

15× 是一个平均值。具体到你的场景，可以用 `code/agent/multiagent_cost.py` 算：

```bash
cd code/agent
python3 multiagent_cost.py
```

```text
                        single         multi          chat
input tok              570,000       140,000        10,000
output tok               8,000        10,000         1,000
cost $                    1.83          0.57          0.04
multi/chat token 倍率: 13.64×
multi/single token 倍率: 0.26×
```

这个输出有两个反直觉的点：

- 和 chat 比，多 agent 是 **13.64×**，和 Anthropic 的 15× 对得上。
- 和"一个把所有材料都塞进 context 的单 agent"比，多 agent 反而只有 **0.26×**——因为单 agent 每一轮都要重新发送累积的 context，10 轮就是 57 万 token；多 agent 的每个 worker 只读自己那 3 万。

所以"多 agent 更贵"不是绝对的。**贵的参照系是 chat，不是单 agent**。如果你的单 agent 已经在靠"每轮重发 10 万 token"硬撑，拆成多 agent 可能更便宜，而且 lead 的峰值 context 从 10 万降到 3 万。

### 3.1 一个更细的成本模型

多 agent 的成本可以拆成：

$$
C_{\text{multi}} = C_{\text{lead}} + \sum_{i=1}^{N} C_{\text{worker}_i} + C_{\text{coord}}
$$

其中 $C_{\text{coord}}$ 是协调成本：lead 读摘要、写计划、解决冲突、重试失败 worker。Anthropic 说他们早期"给简单查询生成 50 个 subagent"，这就是 $C_{\text{coord}}$ 失控的典型。

一个经验法则（Anthropic 的 effort scaling）：

| 任务复杂度 | agent 数 | 每个 agent 的工具调用 |
|---|---|---|
| 简单事实查找 | 1 | 3–10 |
| 中等比较/研究 | 2–4 个 subagent | 每个 10–15 |
| 复杂广度优先 | >10 个 subagent | 按需 |

超过 10 个 subagent 时，协调成本开始占主导；除非任务可以完全独立切分，否则不建议。

## 4. Cognition 的反面：为什么并行 subagent 脆弱

Cognition 的文章（2025-06）讲了两条原则：

> **Share context, and share full agent traces, not just individual messages.**
>
> **Actions carry implicit decisions, and conflicting decisions carry bad results.**

翻成工程语言：

- 两个 subagent 各自做决策时，它们只看到自己那部分 context，不知道对方做了什么决定。
- 一个 agent 决定"把函数改成 async"，另一个 agent 还在按同步调用写测试；两边都"局部正确"，合起来是坏的。
- 这不是"沟通不够"，而是**决策的隐式性**：很多决定没有写进消息里，只体现在代码/文件状态里。

所以 Cognition 的建议是：**默认单线程线性 agent**；如果要多 agent，必须共享完整轨迹，而不只是摘要。

这和 Anthropic 的结论不矛盾，因为任务类型不同：

- 研究任务：子任务独立、结果可合并、冲突少 → 多 agent 划算。
- 编码任务：子任务强依赖、共享文件、决策隐式 → 单 agent 更稳。

Anthropic 自己也在博客里承认："大多数 coding 任务真正可并行的部分比研究少，而且 LLM agent 还不擅长实时协调和委派。"

## 5. Cursor 的第三条路：context efficiency > parallelism

Cursor 2026-07 的 swarm 实验是这一年最值得读的多 agent 工程报告。他们的结论不是"多 agent 好"或"不好"，而是：

> "We suspect the ability to scale the agent swarm comes from this context efficiency, more than from parallelism itself."

他们的设计：

- **planner 不实现**，所以它的 context 不会被低层细节填满；
- **worker 不规划**，所以它可以把全部 context 花在一个狭窄的工作单元上；
- 任务被组织成一棵树，swarm 的形状随问题轮廓生长，而不是固定拓扑；
- 用多层 review 来对抗错误累积（"review 比 work 便宜，所以花在 review 上的 compute 回报高"）。

结果（SQLite 复刻，Rust，4 小时）：

| 配置 | 4 小时通过率 | 成本 |
|---|---|---|
| 旧 swarm（Grok 4.5） | 第 2 小时前就被暂停 | — |
| 新 swarm（Grok 4.5） | 80% | — |
| Opus 4.8 planner + Composer 2.5 worker | ~100% | **1,339 美元** |
| GPT-5.5 全包 | ~100% | **10,565 美元** |

新 swarm 的代码量也小得多：Fable 5 混合配置下，旧 swarm 64,305 行、新 swarm 9,908 行；Opus 混合下，旧 19,013 行/97%、新 4,645 行/100%。

这组数字说明：**多 agent 的胜负手是"每个 token 花在哪个 role 上"，不是"开了多少个 agent"**。用便宜模型做执行、贵模型做规划，质量不降，成本降一个数量级。

## 6. 工程清单

如果你要做多 agent，先回答这七个问题：

1. **任务可切分吗？** 子任务之间有没有共享状态/文件/接口？有强依赖就别拆。
2. **结果可合并吗？** 合并规则是确定的，还是需要另一个模型判断？后者会引入新的错误源。
3. **冲突怎么解决？** 文件锁、git merge、任务队列、还是"重新派一个 agent"？
4. **context 怎么传？** 传摘要（省 token，丢信息）、传完整轨迹（贵，但决策可追溯）、还是传文件引用（折中）？
5. **失败怎么发现？** 每个 worker 有独立 verifier，还是 lead 统一验？
6. **成本上限是多少？** 按 token、按 agent 数、按墙钟，哪个先到？
7. **停止条件是什么？** lead 说"够了"、预算耗尽、还是所有子任务都有结论？

一个可落地的最小架构：

```text
lead（规划 + 汇总 + 冲突解决）
  ├─ worker pool（无状态，输入=子任务+文件引用，输出=摘要+产物路径）
  ├─ verifier（每个产物跑测试/检查）
  └─ session log（所有 agent 的轨迹，可回溯）
```

关键设计：**worker 无状态、产物落盘、lead 只读摘要**。这样 worker 可以重试/替换，lead 的 context 不会被 worker 的中间过程污染。

## 7. 什么时候不要用多 agent

- 任务是一条链（A 的输出是 B 的输入），并行没有收益。
- 子任务共享大量隐式决策（改同一个模块的代码）。
- 你没有 verifier，无法判断 worker 的输出对不对。
- 你的成本敏感，而任务价值不高。
- 你只是想让"多个 agent 讨论"显得更聪明——这是最贵的安慰剂。

## 8. 最小可运行

`code/agent/multiagent_cost.py` 把三种形态的 token 账打出来（见上文）。试着改参数：

```bash
python3 multiagent_cost.py --workers 10 --worker-in 50000 --lead-in 30000
```

你会看到 multi/chat 的倍数迅速上升，而 lead 的峰值 context 仍然只有 3 万。这就是"用钱换 context 隔离"的定量版本。

## 小结

- 多 agent 的本质是**用多个 context window 扩大有效工作集**，不是"多个模型更聪明"。
- Anthropic：research 类任务 +90.2%，代价 ~15× chat token；token 使用量解释 80% 的性能方差。
- Cognition：编码类任务默认单线程；并行 subagent 的隐式决策会冲突，"share full traces" 是底线。
- Cursor：swarm 的杠杆是 context efficiency；planner/worker 分工 + 多层 review，1,339 美元 vs 10,565 美元。
- 选型先问"任务可切分、结果可合并、冲突可解决、失败可验证"；四个都 yes 再上多 agent。
- 成本要和 chat 比，也要和"被 context 撑爆的单 agent"比；两个参照系给出不同结论。

下一篇进入训练侧：agent 的能力不是只靠 prompt 和 harness，agentic RL 正在把"多轮工具使用"直接训进模型。环境、reward、GSPO，以及为什么 verifier 的质量就是 reward 的质量。
