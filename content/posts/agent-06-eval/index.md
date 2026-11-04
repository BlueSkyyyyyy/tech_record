---
title: "评测：SWE-bench、Terminal-Bench 与 Harness 混淆"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 8
series: ["agent"]
tags: ["AI Agent", "系列", "评测", "SWE-bench", "Terminal-Bench", "pass@k", "Leaderboard"]
categories: ["LLM算法"]
---

前六篇讲 agent 怎么搭。这一篇讲一个更麻烦的问题：**你怎么知道它真的变好了？**

Agent 评测比模型评测难，因为分数不只取决于模型，还取决于 harness、工具、提示词、重试策略、环境、grader。同一个模型换一个 scaffold，分数可以差十几个点；同一个 scaffold 换一个 grader，排名可以反过来。

这一篇分三部分：先看几个主流 benchmark 到底在测什么，再看指标（pass@k / pass^k / cost），最后看 2026 年暴露出来的"仪器错误"。

## 1. 先记住一个画面

你参加一场考试，考完发现：

- 有一半的题，监考老师没看到你交卷，直接记了 0 分；
- 有一道题的答案是 `96.124991`，你写 `96.12`，被判错；
- 你的草稿纸被算进了"答题时间"；
- 隔壁考场用的是另一套评分标准。

这不是段子，是 2026 年一篇论文在公开 agent leaderboard 上审计出来的问题（*Silent Instrument Errors Reorder Agent Leaderboards*，arXiv 2611.04399）。它发现一个通用 agent 在某个客服 benchmark 上落后专用 scaffold **13 个点**，差距全部来自 grader：通用 agent 经常以"不触发评分"的方式结束对话，于是大约一半的 episode 被记成失败；把未评分的一半按 benchmark 自己的规则补评，差距就消失了——**没有重跑任何 agent**。

所以读 agent 分数时，第一个问题不是"多少分"，而是"**这个分是怎么算出来的**"。

## 2. 几个 benchmark 在测什么

### 2.1 SWE-bench：真实 GitHub issue

2023 年的 SWE-bench（arXiv 2310.06770）从 12 个 Python 仓库里收了 **2,294** 个 issue + 对应 PR。给模型一个仓库和一个 issue，让它生成 patch；评测时把 patch 应用到仓库，跑测试，看 fail-to-pass 和 pass-to-pass 是否都满足。

原始论文里最好的模型（Claude 2）只解决了 **1.96%**。这个数字当时很震撼，因为它说明"会写代码"和"会改真实仓库"是两件事。

后来 OpenAI 在 2024 年做了 **SWE-bench Verified**：从原集里人工验证出 **500** 道题，确保题目可解、测试可靠。今天大部分模型报的都是 Verified 分数，不是原始 2,294。

**SWE-bench Pro**（arXiv 2509.16941）把难度再推高一档：**1,865** 道题，来自 **41** 个活跃维护的仓库，分成 public（11 个仓库）、held-out（12 个）、commercial（18 个闭源）。它明确针对"长程任务"——人类工程师可能要几小时到几天，patch 跨多个文件。这解决的是 SWE-bench Verified 逐渐饱和的问题。

### 2.2 Terminal-Bench：终端里的长任务

Terminal-Bench 测的是"在命令行里完成一个真实任务"：装环境、跑命令、改配置、验证结果。它比 SWE-bench 更贴近"agent 操作一台机器"，也更依赖环境稳定性。

截至 2026-11，官网已经到 **Terminal-Bench 4.0**，榜单同时列 resolution rate、cost、tokens。这三个数必须一起看：一个 80% 但 50 美元/task 的 agent，和一个 70% 但 2 美元/task 的 agent，是不同的产品。

### 2.3 τ-bench / τ²-bench：和人对话，还要遵守规则

$\tau$-bench（arXiv 2406.12045）模拟"用户 ↔ agent"的多轮对话，agent 有领域 API 和业务规则。评测不看"回答像不像"，而是看**对话结束后的数据库状态**是否等于目标状态。它还提出了 **pass^k** 指标。

原始论文的结论很扎心：GPT-4o 这类 function calling agent 在 **<50%** 的任务上成功，而且很不稳定——retail 域的 **pass^8 < 25%**。

$\tau^2$-bench（arXiv 2506.07982）加了"双控制"：不只是 agent 能调工具，用户也能调工具改共享状态（比如客服让用户自己在手机上操作）。这更接近真实客服，也更难——论文说从单控制切到双控制，性能显著下降。

### 2.4 OSWorld / WebArena / GAIA / BrowseComp

| Benchmark | 规模 | 测什么 | 关键数字 |
|---|---|---|---|
| **OSWorld** | 369 个任务 | 真实桌面应用、文件 I/O、跨应用工作流 | 人类 72.36%，最佳模型 12.24%（2024 原论文） |
| **WebArena** | 812 个任务、4 个域 | 电商、论坛、GitLab、CMS 的真实网站 | 最佳 GPT-4 agent 14.41%，人类 78.24% |
| **GAIA** | 466 个问题 | 需要推理 + 多模态 + 浏览 + 工具的综合问答 | 人类 92%，GPT-4+插件 15% |
| **BrowseComp** | 1,266 个问题 | 持久浏览、找难以发现的信息 | 答案短、可自动验证 |

这些数字现在看已经过时（2026 年的 GUI agent 在 OSWorld 上到 47.5，见第 9 篇），但它们的**设计意图**没过时：OSWorld 测"操作电脑"，WebArena 测"操作网站"，GAIA 测"综合助理"，BrowseComp 测"持久搜索"。

### 2.5 AgentBench：8 个环境的多维评测

AgentBench（arXiv 2308.03688）用 8 个不同环境（OS、数据库、知识图谱、卡牌游戏、电商、网页等）测 LLM-as-agent。它的价值不在某个分数，而在"**同一个模型在不同环境里的能力差异很大**"这个结论：长程推理、决策、指令遵循是主要瓶颈；代码训练对不同 agent 任务的影响甚至是矛盾的。

## 3. 指标：pass@k 和 pass^k 不是一回事

这是 agent 评测里最容易混的一对指标。

### 3.1 定义

- **pass@k**：跑 k 次，**至少成功一次**的概率。用于"多给几次机会，能不能做出来"。
- **pass^k**：跑 k 次，**每一次都成功**的概率。用于"每次都要对"的场景。

如果单次成功率是 $p$，且每次独立：

$$
\text{pass@}k = 1 - (1-p)^k, \qquad \text{pass}^k = p^k
$$

$p=0.75$ 时：

| k | pass@k | pass^k |
|---|---|---|
| 1 | 75.0% | 75.0% |
| 3 | 98.4% | 42.2% |
| 10 | 100.0% | 5.6% |

k=1 时两者相同；k=10 时一个接近 100%，一个接近 0%。同一份 75% 的原始数据，可以讲出完全相反的故事。

### 3.2 为什么 agent 必须看 pass^k

客服、运维、自动改代码、自动转账——这些场景的用户期待是"每次都对"，不是"多试几次总有一次对"。一个 95% pass@1 的 agent，跑 20 次全对的概率只有 $0.95^{20} \approx 36\%$。这就是为什么 demo 很惊艳、上线就出事。

Anthropic 在 2026-01 的 evals 博客里给了一个更细的版本：75% 单次成功率跑 3 次，pass^3 是 $0.75^3 \approx 42\%$；"到 k=10，两个指标讲出相反的故事"。

### 3.3 成本也要进指标

*AI Agents That Matter*（arXiv 2407.01502）批评了 agent 评测的三个问题：

1. 只看准确率，不看成本，导致 SOTA agent 越来越复杂、越来越贵；
2. 模型开发者和下游开发者的评测需求被混在一起；
3. 很多 benchmark 没有像样的 holdout，agent 会过拟合/走捷径。

他们主张**联合优化 accuracy 和 cost**，并给出了一套控制成本的评测方法。这个观点在 2026 年变得更重要：Terminal-Bench 榜单已经把 cost 和 tokens 列出来了，Cursor 的 swarm 实验也在比"同样质量下 1,339 美元 vs 10,565 美元"。

一个可用的指标组合：

```text
resolve_rate        # 任务成功率
pass^k              # k 次全对率（k=3/5/8）
cost_per_task       # 含模型 + 工具 + 环境
tokens_per_task     # 输入/输出分开
p50/p95_latency     # 端到端墙钟
tool_calls_per_task # 效率
error_rate          # 工具错误/解析错误/超时
```

只看第一行，你会选出"最贵的最强"；七行一起看，才能选出"能上线"的。

## 4. 仪器错误：分数是怎么被算坏的

2026 年的 *Silent Instrument Errors* 审计了两个公开 leaderboard（HAL 和 τ²-bench），发现四类问题。每一类都值得你在自己的 eval 里自查。

### 4.1 未评分 = 失败

通用 agent 经常以"不触发 grader"的方式结束（比如没调用 `complete_task`，或者对话超时）。如果 harness 把"未评分"直接记成 0，分数就被系统性拉低。

论文的例子：13 个点的差距，全部来自这里。补评之后差距消失。

**自查**：你的 eval 里，"未评分"和"失败"是不是同一个桶？如果是，分开统计。

### 4.2 成本列重复计数

嵌套的模型调用（比如子 agent、guardian、judge）会被计两次，而且不同行的重复程度不同。结果成本列不可比。

**自查**：成本是按"最外层请求"算，还是按"每一次模型调用"算？两者不能混。

### 4.3 把 grader 的动作记成 agent 的动作

存档的 trajectory 里混进了 grader 自己写的操作（比如 grader 为了检查状态而点击、查询）。这些动作被算进 agent 的工具调用数、token、甚至成功率。

**自查**：trajectory 里能不能区分 `agent_action` 和 `grader_action`？不能区分，就不要用它算效率指标。

### 4.4 用新 evaluator 重评旧轨迹，本身不是中立的

用更新版本的 evaluator 去重评旧 trajectory，看起来是"统一标准"，但新 evaluator 可能对旧 agent 更苛刻/更宽松，排名会变。

**自查**：重评要同时报"原评"和"重评"，并说明 evaluator 版本。

### 4.5 还有一个更隐蔽的：AGENTS.md 的措辞

2026-11 的 *Do Coding Agents Follow AGENTS.md?*（arXiv 2611.04367）在 86 个 benchmark 任务、4 个 harness-model 组合上，把同一条策略用四种措辞写进 Agent Context File。结论：**条件句（conditional）比规定句（prescriptive）和解释句（explanatory）更不容易被遵守**，而"规定 vs 解释"哪个更好因组合而异。

这意味着：你测的"agent 能力"里，有一部分是"它对指令措辞的敏感度"。换一种写法，分数就变了。

### 4.6 任务本身可能是坏的

2026 年的另一篇论文（arXiv 2609.26826）做了一件很耗钱的事：审计 Terminal-Bench 的任务。**1,081** 个 PR、**639** 个被评分的任务、**28,801** 次 trial，花了 **105,933 美元**。

结论：在 **125** 个"所有模型都失败"的任务里，只有 **78** 个被认证为"真的没解出来"；其余是坏引用、缺上下文、或者 verifier 可以被绕过。

这意味着 leaderboard 上的"0% 解决率"不一定是难度，可能是**题目坏了**。一个 benchmark 如果没有做过"人类能不能做出来 / verifier 能不能被作弊"的审计，它的低分和满分都不可信。

### 4.7 单次运行的方差有多大

*On Randomness in Agentic Evals*（arXiv 2602.07150）跑了 **60,000** 条 SWE-bench Verified 轨迹，3 个模型 × 2 个 scaffold。结论：

- 单次运行的 pass@1 会摆动 **2.2–6.0 个百分点**；
- 即使温度设为 0，标准差仍然 **>1.5 个百分点**。

所以"我们比 baseline 高 2 个点"在 agent 评测里基本是噪音。要区分真实的提升，至少需要多次运行 + 置信区间，或者用配对设计（同一批任务、同一套 scaffold）。

## 5. Harness 混淆：同一个模型，不同分数

把影响分数的因素列出来，至少有七层：

| 层 | 例子 | 影响 |
|---|---|---|
| 模型 | 版本、thinking 预算 | 大 |
| 系统提示 | 角色、工作流、输出格式 | 大 |
| 工具集 | 有没有 edit/grep、schema 设计 | 大 |
| 上下文管理 | 压缩、reset、子 agent | 中-大 |
| 重试/循环 | max turns、错误重试、循环检测 | 中 |
| 环境 | 镜像、shell 启动方式、依赖版本 | 中-大 |
| Grader | 评分规则、未评分处理、evaluator 版本 | 大 |

所以"某模型在 SWE-bench 上 65%"这句话，信息量很低。完整的说法应该是："某模型 + 某 scaffold + 某工具集 + 某重试策略 + 某 grader，在 SWE-bench Verified 上 65%，成本 X 美元/task，pass^3 是 Y%。"

这也是为什么 SWE-bench 官方后来要求提交 scaffold 配置；也是为什么 Terminal-Bench 榜单把 agent 和 model 分开列。

## 6. 最小可运行：pass@k / pass^k / grader bug

`code/agent/passk.py` 把这一篇的三个核心计算跑出来：

```bash
cd code/agent
python3 passk.py
```

输出：

```text
per-trial p=75%, trials=10
  k    pass@k    pass^k
  1     75.0%     75.0%
  3     98.4%     42.2%
 10    100.0%      5.6%

grader bug: true=75%  observed=40%  95% CI=[31%,50%]
修复方式不是重跑模型，而是把未评分的 episode 按 benchmark 自己的规则补评。
```

`--test` 里还验证了 pass@k 的无偏估计公式：$1 - \binom{n-c}{k}/\binom{n}{k}$。小样本下这个估计和"$1-(1-p)^k$"不同，写 eval 报告时要注意用的是哪个。

## 7. 读 leaderboard 的检查清单

看到一个新的 agent 分数，按顺序问：

- [ ] benchmark 是哪个版本？SWE-bench 原集 / Verified / Pro 不是一回事。
- [ ] agent 和 model 分开列了吗？scaffold 是什么？
- [ ] 报的是 pass@1 还是 pass^k？k 是多少？
- [ ] 成本怎么算的？含工具/环境/judge 吗？
- [ ] 未评分的 episode 怎么处理？
- [ ] grader 版本是什么？有没有重评？
- [ ] holdout 怎么划分？有没有污染？
- [ ] 环境镜像、shell 启动方式固定了吗？
- [ ] 有没有报 p50/p95 延迟和错误率？
- [ ] 代码/轨迹开源吗？能不能复现？

十问里能答上六个，这个分数才有参考价值。

## 小结

- SWE-bench 测真实 issue（2,294 / Verified 500 / Pro 1,865），Terminal-Bench 测终端长任务，τ-bench 测对话 + 规则 + 数据库状态，OSWorld/WebArena 测 GUI/网页，GAIA/BrowseComp 测综合检索。
- pass@k 和 pass^k 回答不同问题；agent 上线要看 pass^k，75% 单次成功率在 k=3 时只有 42%。
- 成本、延迟、工具调用数、错误率必须和成功率一起报；只看 accuracy 会选出又贵又脆的系统。
- 2026 年的仪器错误审计说明：未评分、重复计费、grader 动作混入、evaluator 版本变化，都能单独改变排名。
- Harness 混淆有七层；"模型多少分"不是有效陈述，"模型 + scaffold + 工具 + grader 多少分"才是。

下一篇进入一个特殊的 agent 类型：computer-use / 浏览器 agent。它们的动作空间不是 JSON，而是像素、坐标、点击和输入；它们的失败模式也不是"工具报错"，而是"看错了屏幕"。
