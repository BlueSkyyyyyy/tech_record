---
title: "Context Engineering：预算、压缩与信息衰减"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 4
series: ["agent"]
tags: ["AI Agent", "系列", "Context Engineering", "压缩", "KV Cache", "记忆"]
categories: ["LLM算法"]
---

第 2 篇讲循环，第 3 篇讲工具。这一篇讲它们共享的那块地：**context window**。

先给一个反直觉的结论：**窗口没用满，信息也可能已经丢了。**

2026 年 10 月，一篇叫 *Context Is a Pipeline, Not a Window* 的论文在一个真实工具型 agent 上跑了 720 次。这个 agent 要读 Linux 内核 trace、定位注入的故障。作者统计了三件事：

- 最大的一次 prompt 只用了窗口的 **5.75%**；
- 11,189 条工具结果里，**3,827 条（约 34%）在到达模型前被截断**；
- 每一条被截断的结果都变成了**非法 JSON**。

更刺眼的是同一个结果里的两个字段：一个字段只有 **4.7%** 的调用能到达模型，另一个字段是 **99.3%**。哪个字段带故障证据，纯属运气。

这就是这一篇要建立的直觉：context 不是"一个能装多少 token 的桶"，而是一条**从工具到模型、经过截断/压缩/重排的流水线**。每一站都可能丢东西，而窗口大小只描述最后一站。

## 1. 先记住一个画面

想象一条快递分拣线。包裹从仓库（工具）出发，经过传送带（harness），最后放到收件人桌上（模型）。

窗口大小是桌子的面积。但包裹在传送带上可能被**拆箱**（结构化截断）、**压缩**（摘要）、**贴错标签**（字段重排），甚至**掉进缝隙**（token 预算不足）。桌子再大，也救不回已经丢在传送带上的东西。

Agent 的 context engineering，做的就是分拣线的设计：什么先拆、什么原样送、什么放仓库（文件/记忆）、什么直接扔掉。

## 2. 为什么"更大"不等于"更好"

Anthropic 在 2025-09 的 effective context engineering 博客里把这件事讲得很清楚：

- **context rot**：token 越多，模型从 context 里准确召回信息的能力越差。所有模型都有，只是程度不同。
- **attention budget**：模型解析长 context 时有一个"注意力预算"，每多一个 token 就消耗一点。
- **架构原因**：transformer 的注意力是 $n^2$ 对关系；训练分布里长序列更稀少，模型对长距离依赖的经验更少。

所以正确的目标不是"把窗口填满"，而是：

> 找到**最小的、高信号的 token 集合**，让模型最可能产生想要的行为。

这句话是 Anthropic 那篇博客的题眼，也是后面所有压缩策略的判据。压缩不是"删到能放下"，而是"删到只剩有用的"。

## 3. 先立预算：一次请求到底花在哪

把一个 agent 请求拆开，context 的组成大致是：

$$
T_{\text{total}} = T_{\text{system}} + T_{\text{tools}} + T_{\text{history}} + T_{\text{tool\_out}} + T_{\text{reserve}}
$$

其中：

| 项 | 典型量级 | 是否随轮次增长 |
|---|---|---|
| system prompt | 2K–10K | 否 |
| 工具定义 | 5K–100K+ | 否（除非做检索） |
| 对话/轨迹历史 | 每轮 1K–10K | 是 |
| 工具输出 | 每轮 1K–50K | 是 |
| 输出预留 + 安全垫 | 10%–20% 窗口 | 否 |

举个具体的：一个 200K 窗口的模型，system 5K、工具 30K、每轮历史和工具输出合计 8K、预留 20K。那么第 $n$ 轮的 context 是：

$$
T(n) = 5 + 30 + 8n + 20 \quad (\text{K token})
$$

$n=18$ 时就到 199K，必须压缩。如果工具输出偶尔来一个 50K 的大结果，$n=8$ 就可能爆。

这张账的意义在于：**压缩的触发点不应该等窗口满，而应该等"增长斜率"变陡**。Codex 的做法是每轮采样后算 token status，超过 `auto_compact_scope_limit + fallback_buffer` 就触发（`session/context_window.rs:98-105`、`session/turn.rs:699-702`）。它不是等 100%，而是留了一个 buffer。

## 4. 压缩的四种手法

### 4.1 工具结果清理（最轻）

把历史里已经用过的工具输出删掉，只留一句"已读过 X，关键结论是 Y"。

Anthropic 说这是"最安全的轻触式压缩"，并且已经在 Claude Developer Platform 上做成 feature。适合场景：工具输出很大、但结论已经被后续消息引用过。

风险：模型可能还需要原始细节（比如精确的行号、报错栈）。所以清理时要保留"指针"——文件路径、行号范围、可以用什么命令重新取回。

### 4.2 摘要压缩（compaction）

把整段历史交给模型，让它写一份交接摘要，然后用"摘要 + 最近几条消息"替换历史。

Codex 的实现很具体（`core/src/compact.rs`）：

- 保留**最近的真实用户消息**，最多 20K token（`COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000`，`:63`）；
- 从后往前装，装不下的第一条做截断（`:716`、`:806`）；
- 媒体内容直接丢弃（`:769-772`）；
- 最后放一条 summary item（`:818-825`）；
- assistant 消息、reasoning、tool call、tool output **全部丢弃**。

摘要 prompt 在 `codex-rs/prompts/templates/compact/prompt.md`，要求写清楚：当前进度和关键决策、重要的上下文/约束/用户偏好、剩下要做什么、关键数据/例子/引用。摘要前面还会加一段"另一个模型已经开始解决这个问题"的框架语（`summary_prefix.md`），让接手模型知道这是交接而不是用户新指令。

Claude Code 的压缩则保留了"**最近访问的 5 个文件**"（Anthropic 博客原话），因为代码任务里"我正在改哪几个文件"比"对话说过什么"更重要。

### 4.3 结构化笔记（agentic memory）

让 agent 自己维护一份 `NOTES.md` / `claude-progress.txt` / `feature_list.json`，把关键状态写在 context 外面，需要时再读回来。

Anthropic 的 long-running harness 实验里，最有效的几个产物是：

- `claude-progress.txt`：这一班做了什么、下一步做什么；
- `feature_list.json`：200 多项功能，每项有 `passes: false/true`；
- `init.sh`：怎么启动、怎么跑端到端测试；
- git history：可回滚的检查点。

一个很细但很重要的选择：**feature list 用 JSON 而不是 Markdown**。Anthropic 的原话是"模型不太可能不恰当地修改或覆盖 JSON 文件，相比 Markdown 文件"。Markdown 太容易被"重写一遍"，JSON 有结构约束，模型更倾向于只改 `passes` 字段。

### 4.4 子 agent（context 隔离）

主 agent 只拿计划，子 agent 各自有完整 context，最后只回传 1K–2K token 的摘要。

Anthropic 的数字：子 agent 可能探索"数万 token"，但只返回"**通常 1,000–2,000 token**"。这是上下文隔离，不是并行加速——第 9 篇会展开。

## 5. 压缩 vs 重置：一个被低估的区别

Compaction 是"在同一段对话里，把前面 Summarize 掉，让同一个 agent 继续"。Context reset 是"清空窗口，起一个新 agent，用一份结构化交接让它接手"。

Anthropic 在 2026-03 的 harness design 博客里讲了两者的差别：

- compaction 保留连续性，但**不给你一个干净的起点**；如果模型有"context anxiety"（接近它以为的上限就提前收尾），压缩后这种焦虑仍在。
- reset 给干净起点，但要求交接产物足够完整；代价是编排复杂度、token 开销、每轮延迟。

他们的实测：Sonnet 4.5 的 context anxiety 强到"仅靠 compaction 不够"，所以 reset 是必需的；换到 Opus 4.5，这个行为基本消失，reset 就变成了"dead weight"，被删掉。

这条演化说明：**压缩策略不是永久真理，它是模型能力的函数**。今天必须做的 reset，明天可能就是纯开销。

### 5.1 压缩是一个信任边界

2026 年还有两篇论文把"压缩"从性能问题推到了安全问题。

第一篇是 *Compaction Laundering*（arXiv 2610.29869）。它发现：当 harness 把 transcript **压平**成一段文本再摘要时，伪造的"用户批准"可以在压缩后继续存活，被下一班 agent 当成真实指令。在 5 个"压平型" harness（pi、opencode、Cline、Aider、OpenHands）里，成功率 **87%–100%**；在 4 个"角色保真型" harness（Codex、Gemini CLI、Qwen Code、Continue）里是 **0%**，即使窗口开到约 **908K token**。

差别不在压缩算法，而在**压缩时有没有保留消息的角色和来源**。一条 `role=user` 的伪造批准，和一条 `role=tool` 的网页内容，压平成一段文字后就分不清了；角色保真的 harness 会把它们分开摘要，伪造的批准无法"洗白"成用户指令。

第二篇是 2026 年的一项测量（arXiv 2606.29718）：agent 往往**在窗口填满之前就放弃**——它会说"搜索没有结论"，然后停止，而不是继续探索。4 个旗舰模型 × 3 个 benchmark 都出现了这种"inconclusive search failure"。

这两篇合起来说明：context 的问题不只是"装不下"，还有"装下了但分不清谁说的"和"没装满但模型先泄气"。压缩、角色、持久性，必须一起设计。

## 6. 缓存：压缩的隐藏成本

只要改 history 的前缀，KV cache 就失效。Anthropic 的 prompt caching、各家推理引擎的 prefix cache，都依赖"前缀稳定"。

2026 年 10 月的 *Cache Engineering for Agents* 把这件事讲透了：agent 轨迹越来越长，KV cache 的 I/O 和存储成为主要成本；而传统的 context engineering（trim/summarize）会**修改轨迹前缀，使后续缓存失效，触发对保留部分的昂贵 re-prefill**。保留太少又会影响准确率。

论文提出的方案不是"少压缩"，而是让 agent **直接控制 active KV cache**：用 `drop` 把选中的 cache 排除出后续计算，但不修改轨迹、不 re-prefill 保留部分。这相当于把"删历史"变成"删缓存"，历史还在，只是不再参与 attention。

这对 harness 设计有一个直接推论：**压缩策略要和缓存布局一起设计**。如果你每轮都把历史重写成"摘要 + 最近消息"，那缓存命中率会很低；更好的做法是保持前缀稳定，把要丢弃的内容放到尾部，或者用 cache drop 而不是文本删除。

## 7. 记忆：不要把"窗口"和"会话"混为一谈

2026 年 4 月，Anthropic 在 Managed Agents 里给了一个很清晰的区分：

> "The session is not Claude's context window."

Session 是**追加式的事件日志**，持久化在 context window 外面；context window 只是 harness 从日志里**取出来给模型看的那一段**。Compaction 会丢信息，但 session 不会；harness 可以随时从 session 里重新取、重新组织、重新压缩。

这个区分解决了一个长期混淆：我们总说"agent 的记忆不够"，其实有两件不同的事：

- **窗口不够**：一次采样能看的 token 有限；
- **会话不持久**：跨请求/跨进程的状态没存下来。

前者靠压缩/检索/子 agent；后者靠 session log、文件系统、数据库。把两者混在一起，就会得出"换个更大窗口的模型就好了"这种错误结论。

### 7.1 复杂记忆一定更好吗？

2026 年 10 月还有一篇很"反潮流"的论文：*Is Complex Agentic Memory Always Needed? Naive RAG Is Stronger Than We Thought*。

作者把 Raw-RAG（对原始历史做 dense retrieval + rerank）和一堆复杂的 agentic memory 系统比，在 LoCoMo、LongMemEval、MEME、MemoryAgentBench 上，Raw-RAG 在 **7 个 benchmark-model 设置里 6 个排第一**，比最强的 agentic memory 高 **0.13–5.53 个点**。

原因不是"复杂方法没用"，而是"**覆盖度比利用率更重要**"：复杂记忆会做摘要、结构化、改写，损失了证据的覆盖；原始历史检索虽然"笨"，但保留了答案值和推导所需的关系/依赖路径。

这对做 memory 的团队是一个很好的提醒：先做一个强的 raw-history baseline，再决定要不要上复杂的记忆结构。很多时候，你需要的不是更聪明的记忆，而是更好的检索和更大的原始证据覆盖面。

## 8. 一条可执行的 context pipeline

把上面的东西拼起来，一个可落地的 pipeline 是：

```text
工具输出
  → 结构化裁剪（按字段，不按字符串）
  → 单条结果预算（比如 25K token，Claude Code 的默认值）
  → 写入 session log（原始版本，可回溯）
  → 进入 context 的版本（精简 + 指针）
  → 每轮 token status
  → 超阈值：先清工具结果，再摘要，最后 reset
  → 跨会话：进度文件 + git + 任务列表
```

几个具体的数字，可以直接拿去当默认值：

| 参数 | 建议 | 来源 |
|---|---|---|
| 单条工具结果上限 | 25K token | Anthropic writing tools for agents |
| 压缩时保留的用户消息 | ≤20K token | Codex `COMPACT_USER_MESSAGE_MAX_TOKENS` |
| 压缩后保留的最近文件 | 5 个 | Claude Code |
| 触发压缩的阈值 | 窗口的 80%–90%，留 buffer | Codex token status |
| 子 agent 回传摘要 | 1K–2K token | Anthropic multi-agent |
| 进度文件 | 每班更新，JSON 优先 | Anthropic long-running harness |

## 9. 最小可运行

`code/agent/token_budget.py` 把三种截断策略放在同一批合成 trace 上比较：

```bash
cd code/agent
python3 token_budget.py --runs 50 --budget 800
```

输出：

```text
runs=50  budget=800 chars
policy       evidence kept    parseable JSON
raw                  0.0%             0.0%
middle             100.0%             0.0%
field              100.0%           100.0%
```

三条曲线对应三种 harness：

- `raw`：`payload[:budget]`，最常见的写法，证据在尾部时直接归零，JSON 也坏了。
- `middle`：保留头尾，证据能保住，但 JSON 仍然非法。
- `field`：先理解结构，删低价值字段，从最近事件往前装，证据和 JSON 都保住。

这个 demo 不是要证明"字段压缩一定更好"，而是要说明：**截断策略是一个需要设计的接口，不是一行切片**。你的工具返回什么结构、模型需要什么字段、哪些字段可以丢，这些信息只有工具作者知道，harness 不该用 `[:N]` 一刀切。

## 小结

- context 是一条流水线，不是窗口。窗口只描述最后一站，前面的截断/压缩/重排才是信息丢失的主战场。
- 目标不是填满窗口，而是"最小高信号 token 集合"；context rot 和 attention budget 是所有模型共有的约束。
- 四种压缩手法：工具结果清理、摘要压缩、结构化笔记、子 agent 隔离。它们可以叠加，但顺序很重要。
- compaction 保留连续性，reset 给干净起点；用哪个取决于模型有没有 context anxiety，而这是会随模型升级变化的。
- 压缩会破坏 KV cache 前缀；缓存布局要和压缩策略一起设计。
- session ≠ context window。持久化日志和"这次给模型看什么"是两件事。
- 复杂记忆不一定更好；先做 raw-history RAG baseline，再谈结构化。

下一篇开始拆真实 harness。第一个是 Codex CLI：三层循环、没有 max turns、沙箱、apply_patch，以及 2026 年长出来的 code mode / guardian / hooks。
