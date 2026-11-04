---
title: "记忆与安全：从 MemGPT 到 CaMeL"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 12
series: ["agent"]
tags: ["AI Agent", "系列", "Memory", "Prompt Injection", "CaMeL", "Sandbox", "安全"]
categories: ["LLM算法"]
---

这一篇把两个看起来不相干的话题放在一起：**记忆**和**安全**。

它们共享同一个底层问题：**数据从哪来、能流到哪去。**

- 记忆系统决定"哪些历史数据可以进入当前 context"；
- 安全系统决定"哪些数据可以离开 agent、流向外部"。

一个 agent 如果记忆很强但安全很弱，它会把私密信息记得很牢，然后准确地发给攻击者；如果安全很强但记忆很弱，它每次都要重新问用户，体验很差。两者必须一起设计。

## 1. 先记住一个画面

MemGPT 在 2023 年给了一个很漂亮的类比：**LLM 是一个操作系统，context window 是内存，外部存储是磁盘。**

- 主上下文（main context）= 内存：system prompt、最近消息、当前工作集；
- 外部上下文（external context）= 磁盘：历史对话、文档、数据库；
- agent 通过函数调用在两者之间换页：`archival_memory_insert`、`archival_memory_search`、`core_memory_append`、`core_memory_replace`。

这个类比解释了为什么 agent 需要"记忆管理"：窗口是有限的，历史是无限的，必须有一个换页策略。但它也埋了一个坑：**操作系统的换页是确定的，agent 的换页是模型决定的**。模型可能忘记写、写错、或者把不该记的东西记下来。

> **类比的边界**：OS 的页表由内核管理，agent 的"页表"由模型自己管理。所以 agent 记忆的可靠性，取决于模型的自我管理能力，而不是硬件的保证。这是为什么很多"复杂记忆系统"最后不如一个朴素的 RAG baseline。

## 2. 记忆的四个层次

一个完整的 agent 记忆栈，通常有四层：

| 层 | 存什么 | 生命周期 | 例子 |
|---|---|---|---|
| **工作 context** | 当前采样能看到的 token | 一次请求 | system + 最近消息 + 工具输出 |
| **会话日志** | 这次 session 的所有事件 | 一次会话 | append-only event log |
| **长期存储** | 跨会话的事实/偏好/经验 | 永久 | 文件、数据库、向量库 |
| **程序性记忆** | 怎么做事（技能/流程） | 永久 | skills、playbooks、代码 |

大部分"agent memory"讨论只关注第三层，但实际工程里，**第二层（session log）和第四层（skills）往往比第三层更重要**：

- session log 让 agent 可以回溯、重放、审计；compaction 丢了信息，但 log 还在。
- skills 让 agent 不用每次从零推导流程；它比"记住上次怎么做的"更可靠。

Anthropic 在 Managed Agents 里把这件事说得很直接：

> "The session is not Claude's context window."

Session 是追加式日志，持久化在窗口外面；窗口只是 harness 从日志里取出来给模型看的那一段。这个区分让"记忆"从"模型的能力"变成"系统的能力"。

## 3. 记忆系统的设计问题

### 3.1 存什么

不是所有东西都值得记。一个实用的分类：

- **事实**：用户偏好、项目约束、API 版本；
- **决策**：为什么选 A 不选 B（比"选了什么"更有价值）；
- **教训**：上次踩过的坑、失败的尝试；
- **指针**：文件路径、commit hash、URL（比内容更省 token）。

Anthropic 的 long-running harness 里，最有价值的记忆不是"对话摘要"，而是 `claude-progress.txt`（这一班做了什么、下一步做什么）、`feature_list.json`（哪些功能通过了）、git history（可回滚的检查点）。

### 3.2 什么时候召回

三种策略：

- **always-in-context**：把核心事实固定在 system prompt 里。简单，但占 token，而且会过期。
- **retrieval**：按当前 query 检索。省 token，但依赖检索质量。
- **agent-initiated**：让模型自己决定"我要查记忆"。最灵活，但多一次往返，而且模型可能不知道自己要查什么。

2026 年 10 月的 *Is Complex Agentic Memory Always Needed?* 给了一个很重要的 baseline：Raw-RAG（对原始历史做 dense retrieval + rerank）在 LoCoMo、LongMemEval、MEME、MemoryAgentBench 上，**7 个 benchmark-model 设置里 6 个排第一**，比最强的 agentic memory 高 **0.13–5.53 个点**。

原因不是"复杂方法没用"，而是"**覆盖度比利用率更重要**"：复杂记忆会摘要、结构化、改写，损失了证据的覆盖；原始历史检索保留了答案值和推导所需的关系/依赖路径。即使给 transformed memory 完整的访问权限，仍然和 raw evidence 有差距。

这条结论对做 memory 的团队很实用：**先做一个强的 raw-history RAG baseline，再决定要不要上复杂结构**。很多时候，你需要的不是更聪明的记忆，而是更好的检索和更大的原始证据覆盖面。

### 3.3 什么时候忘

遗忘不是"删掉"，而是"降低召回优先级"。三种机制：

- **TTL**：事实有保质期（比如"用户当前项目"）；
- **冲突检测**：新事实和旧事实冲突时，保留新的、标记旧的；
- **衰减**：长期不被召回的记忆降权，但不删除。

遗忘的反面是**记忆污染**：一条错误的记忆会被反复召回，影响后续所有决策。所以记忆写入要有"来源"和"置信度"，召回时要能区分"用户说的"、"模型推断的"、"工具返回的"。

### 3.4 怎么评估

常用的 benchmark：

- **LoCoMo**：长对话记忆；
- **LongMemEval**：长期记忆的多类型问题；
- **MEME**：多会话记忆；
- **MemoryAgentBench**：agent 记忆任务。

评估时要注意：记忆系统的分数对"检索器、reader 模型、history 长度"都很敏感；跨论文比较要看清配置。

## 4. 安全：三个真实威胁

### 4.1 Prompt injection

Agent 读到的每一个外部内容都可能是注入：网页、邮件、PDF、issue 评论、工具返回的 JSON、甚至文件名。

AgentDojo（arXiv 2406.13352）是专门评测这个的：**97** 个真实任务、**629** 个安全测试用例，覆盖邮件、银行、旅行等场景。它的结论是：现有攻击能破坏部分安全属性，但不是全部；现有防御能挡一部分，但不是全部。

2026 年 11 月的 *The Devil is in the Context*（arXiv 2611.04672）进一步指出：prompt injection 不是单一攻击，而是**九个维度**的谱系——注入的指令本身、它和用户任务的关系、它在目标系统里的表现形式。九个主流 benchmark 只覆盖了这个小空间的一角；即使把它们合起来，仍有 **13%** 的类别组合完全没被测到。

所以"我们的防御能挡 95% 的注入"这句话，必须问：**在哪个维度上、哪个 benchmark 上、哪种攻击分布上？**

### 4.2 数据外泄：lethal trifecta

Simon Willison 在 2025-06 提了一个很简洁的框架：**lethal trifecta**——当一个 agent 同时具备以下三件事，就存在致命风险：

1. 访问私密数据；
2. 处理不可信内容；
3. 能对外通信。

攻击者只需要在不可信内容里放一句指令，让 agent 把私密数据发到外部。三者缺一，风险都可控；三者齐全，任何"提示词防御"都不可靠。

这就是为什么浏览器 agent 特别危险：它天然同时具备三者（读网页、看 cookie、能提交表单）。

### 4.3 权限组合：local allow, global deny

2026 年 10 月的 *Local Allow, Global Deny*（arXiv 2610.27620）给了一个很硬的结果：

> 逐动作授权不能组合成全局授权。

他们定义了 **compositional authorization failure (CAF)**：一次本地授权的执行，或一对本地授权的执行，违反了跨 intent/purpose/provenance/consent/delegation 的全局策略。

核心定理：**关系型目的违规是一个 2-safety hyperproperty，没有单轨迹的"责任点"**。也就是说，一个"从不误报"的逐动作监控器，**检测不到任何**这类违规——不管底层模型多强、决策过程多好。

实验更扎心：九个 LLM judge 在"逐动作视图"下几乎一致地批准了合成的违规 witness，阻断率和 benign control 差不多；但当它们看到**两条执行放在一起**时，又能识别出来。

这说明：**安全监控需要跨执行的全局视图**，而不是"每次工具调用问一遍模型"。这也是为什么"给每个工具加一个审批弹窗"不能解决数据流问题——弹窗看到的是单个动作，看不到"这封邮件的内容来自刚才读的私密文件"。

## 5. CaMeL：把控制流和数据流分开

CaMeL（arXiv 2503.18813）是目前最成体系的"设计上防注入"方案。它的核心思想：

1. **从可信的用户 query 里提取控制流和数据流**。控制流（先做什么、再做什么、什么条件）只由可信 query 决定，**不可信数据永远不能影响程序流**。
2. **用 capability 做数据流控制**。每个值带一个 capability/taint 标签；工具调用时检查策略：私密数据不能流向未授权的外部 sink。
3. **模型只负责"填参数"和"读结果"**，不负责决定"下一步调用哪个工具"——后者由解释器根据可信 query 生成的程序决定。

效果：在 AgentDojo 上，CaMeL 解决了 **77%** 的任务，同时**可证明安全**；无防御系统是 **84%**。对 Gemini 2.5 Pro 的成功注入攻击从 **300 次降到 0**。也就是说，用 7 个点的效用换"可证明的安全边界"。

这个 trade-off 很值得注意：**安全不是免费的，但你可以选择把成本花在哪**。CaMeL 选择牺牲灵活性（不能动态决定工具链），换形式化的保证。

### 5.1 一个极简的 taint 模型

`code/agent/taint.py` 把 CaMeL 的核心思想压缩成了一个可运行的检查器：

```bash
cd code/agent
python3 taint.py
```

```text
per-action:
  read_email  (True, 'ok')
  send_email  (True, 'ok')

global:
  allowed=False  reason=send_email: private data reaches an external sink (['private', 'public', 'untrusted'])
```

两次逐动作检查都说 ok，全局检查在第二次调用拒绝——差别只在"数据从哪来"。

它实现的三条规则：

- `read_email` 产出 `private` 标签；
- `web_fetch` 产出 `untrusted` 标签；
- `send_email` / `transfer` 是外部 sink：`private` 数据不能进，`untrusted` 数据不能控制收件人/金额。

这不是生产级实现（真实系统要考虑字段级标签、隐式流、declassification、策略语言），但它演示了"**安全边界在 harness，不在模型**"这句话的具体形态。

## 6. 沙箱：最后一道物理边界

Prompt injection 防不住的时候，沙箱是最后一道防线。

Anthropic 的 Claude Code sandboxing（2025-10）给了两个数字：

- 沙箱让权限提示减少 **84%**；
- 有效的沙箱必须**同时**做文件系统和网络隔离：没有网络隔离，被攻陷的 agent 可以把 SSH key 外传；没有文件系统隔离，agent 可以逃出沙箱再联网。

它的实现是 OS 级原语：Linux 的 bubblewrap + seccomp，macOS 的 Seatbelt；网络只允许通过一个 Unix domain socket 连到沙箱外的代理，代理按域名做白名单和确认。

Cursor 的 agent sandboxing（2026-02）也报了一个数字：沙箱化的 agent "stop 40% less often"——因为不需要每步问用户，agent 可以更连续地工作。

Codex 的沙箱前面讲过：三级模式 + 审批策略 + execpolicy。Claude Code on the web 更进一步：**敏感凭证永远不在沙箱里**，git 操作走一个代理，代理持有真正的 token，沙箱里只有一个 scoped credential。

这条"凭证不进沙箱"的原则很重要：即使 agent 被完全攻陷，攻击者也只能拿到一个受限的、可撤销的 token，而不是用户的 GitHub 私钥。

## 7. 一个纵深防御的参考架构

把上面的东西拼起来：

```text
[输入层]
  外部内容打标签：trusted / untrusted / private
  注入检测（规则 + 模型）：只做信号，不做唯一防线

[决策层]
  可信 query → 控制流（不可信数据不能改变）
  模型只填参数、读结果
  高风险动作：独立 reviewer（guardian）+ 人工审批

[执行层]
  沙箱：文件系统 + 网络隔离
  最小权限：只给需要的路径/域名/命令
  egress 代理：所有外联走白名单

[数据层]
  taint/capability 追踪
  私密数据不进外部 sink
  凭证不进沙箱

[审计层]
  append-only session log
  每个动作的 provenance
  跨执行监控（不是只看单动作）
```

五层里，**模型只占决策层的一部分**。把安全寄托在"模型不会被骗"上，等于把五层防御压成一层。

## 8. 记忆与安全的交叉点

- **记忆写入**：不可信内容不能直接写进长期记忆（否则一次注入会污染所有后续会话）。
- **记忆召回**：召回时要带 provenance，模型要知道"这条记忆来自用户还是网页"。
- **记忆删除**：用户要求删除时，要能真的删掉（包括向量库、缓存、日志）。
- **跨会话**：一个会话里被注入的"事实"，不能自动变成另一个会话的"系统指令"。

这几条目前没有标准答案，是 2026 年 agent 安全最活跃的方向之一。

## 小结

- 记忆栈有四层：工作 context、session log、长期存储、程序性记忆；后两层常被忽略，但往往更重要。
- Session ≠ context window；持久化日志和"这次给模型看什么"是两件事。
- 复杂记忆不一定更好：Raw-RAG 在 7 个设置里 6 个排第一，覆盖度比利用率更重要。
- Prompt injection 是九维谱系，benchmark 覆盖不全；lethal trifecta 是判断风险的好框架。
- 逐动作授权不能组合成全局授权（2-safety hyperproperty）；安全监控需要跨执行视图。
- CaMeL 用"可信 query 定控制流 + capability 控数据流"换可证明安全，AgentDojo 77% vs 84%。
- 沙箱是最后一道物理边界：文件系统 + 网络同时隔离，凭证不进沙箱；Anthropic 报权限提示减少 84%。

最后一篇把整个专题收口：2026 年的论文地图、十个开放问题，以及如果你要继续往下读，应该按什么顺序读。
