---
title: "Agentic RL：环境、reward 与 GSPO"
date: 2026-11-04T11:00:00+08:00
draft: false
weight: 11
series: ["agent"]
tags: ["AI Agent", "系列", "Agentic RL", "GSPO", "GRPO", "verl", "SWE-RL", "Kimi K2"]
categories: ["强化学习"]
---

前九篇讲的是"怎么用好一个模型"。这一篇讲"怎么把 agent 能力训进模型"。

Agentic RL 和 RLHF 看起来都是"采样 → 打分 → 更新"，但难度不在一个量级：

- RLHF 的轨迹是一段回答，长度几百 token；agent 的轨迹是几十轮工具调用，长度几万到几十万 token。
- RLHF 的 reward 来自人类偏好模型；agent 的 reward 来自**环境**：测试通没通过、文件改对没改对、网页状态对不对。
- RLHF 的环境是静态的；agent 的环境是**有状态的**：文件系统、数据库、浏览器、API，每一步都改变世界。
- RLHF 的样本可以离线批量生成；agent 的 rollout 要真的执行代码、跑测试，慢且贵。

所以 agentic RL 的瓶颈往往不是算法，而是**环境工程**：你能不能稳定、可复现、低成本地跑几万条轨迹，并且给每条轨迹一个可信的 reward。

## 1. 先记住一个画面

一个 RL 训练循环：

```text
policy（当前模型）
  → 采样 N 条轨迹：每条轨迹 = 多轮 (观察, 动作, 工具结果)
  → 环境执行：真的跑测试、真的改文件
  → verifier 打分：通过=1，失败=0（或部分分）
  → 组内比较：同一道题的多条轨迹，谁比谁好
  → 更新 policy：提高好轨迹的概率，降低坏轨迹的概率
  → 把新 policy 放回环境，重复
```

这个循环里，**verifier 是新的"人类偏好模型"**。它的质量直接决定 reward 的质量；它的偏差会被 RL 放大成模型的偏差。

一句很实用的话：**Agentic RL 的上限，由环境与 verifier 决定，不由算法决定。**

## 2. 形式化：把 agent 当成 POMDP

Agent 与环境交互，可以写成一个部分可观测马尔可夫决策过程：

- 状态 $s_t$：环境的真实状态（文件、数据库、页面）；
- 观察 $o_t$：模型看到的东西（工具输出、截图、DOM）；
- 动作 $a_t$：工具调用或 GUI 动作；
- 转移 $P(s_{t+1}|s_t,a_t)$：由环境决定；
- 奖励 $R$：通常在轨迹结束时给（稀疏），也可能中间给（过程奖励）。

一条轨迹 $\tau = (o_1,a_1,\dots,o_T,a_T)$，目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=1}^{T} \gamma^{t-1} r_t\right]
$$

Agent 场景里通常 $\gamma=1$，reward 只在最后给：$r_T = 1$ 如果测试通过，否则 0。

这和 RLHF 的关键区别是：**动作空间是"工具调用"，不是"token"**。但实际训练时，我们仍然在 token 层面做优化——把整条轨迹的 token 序列当成一个动作序列，reward 在序列结束时给。这就是为什么"序列级"的算法（GSPO）在 agent 场景里越来越重要。

## 3. 从 PPO 到 GRPO 到 GSPO

### 3.1 PPO 的比率

PPO 的核心是重要性采样比率：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}
$$

目标：

$$
L^{\text{PPO}} = \mathbb{E}\left[\min\left(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]
$$

$A_t$ 是优势函数，衡量"这个动作比平均水平好多少"。PPO 需要一个 critic 来估 $V(s)$，在长轨迹上很难训。

### 3.2 GRPO：用组内比较替代 critic

GRPO（DeepSeek-R1 用的算法）去掉了 critic：对同一道题采样 $G$ 条轨迹，用组内 reward 的均值和标准差做归一化：

$$
A_i = \frac{r_i - \text{mean}(r_1,\dots,r_G)}{\text{std}(r_1,\dots,r_G)}
$$

然后对每个 token 算比率、做 clip：

$$
L^{\text{GRPO}} = -\frac{1}{\sum_i |\tau_i|}\sum_{i}\sum_{t} \min\left(\rho_{i,t} A_i,\ \text{clip}(\rho_{i,t},1-\epsilon,1+\epsilon) A_i\right)
$$

好处：不需要 critic，实现简单，适合"同一道题多个样本"的场景。坏处：**token 级比率在长轨迹上会爆炸**——一个 token 的概率变化 2 倍，整条轨迹的更新就被它主导。

### 3.3 GSPO：序列级比率

GSPO（arXiv 2507.18071）把比率从 token 级换成序列级：

$$
\rho_i(\theta) = \left(\frac{\pi_\theta(\tau_i)}{\pi_{\text{old}}(\tau_i)}\right)^{1/|\tau_i|}
= \exp\left(\frac{1}{|\tau_i|}\sum_t \log\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{\text{old}}(a_{i,t}|s_{i,t})}\right)
$$

注意有一个 $1/|\tau_i|$ 的长度归一化。然后做序列级 clip：

$$
L^{\text{GSPO}} = -\frac{1}{G}\sum_{i=1}^{G} \min\left(\rho_i A_i,\ \text{clip}(\rho_i,1-\epsilon,1+\epsilon) A_i\right)
$$

三个好处：

- **稳定 MoE**：序列级比率不会因为某个 token 的概率突变而爆炸；Qwen3 的 MoE RL 训练就是靠它稳住的。
- **基础设施更简单**：不需要在每个 token 上存 old logprob、算比率；序列级 ratio 可以只在轨迹结束时算一次。
- **credit assignment 更粗**：一条轨迹一个 ratio，适合"整条轨迹好不好"的稀疏 reward；代价是细粒度 credit 变弱。

`code/agent/grpo.py` 把两者的差异打了出来：

```bash
cd code/agent
python3 grpo.py
```

```text
token ratios: [1.35, 0.942, 0.942, 0.942]
sequence ratio: 1.127

A=+2.65 (成功样本)
  GRPO loss=-2.6625  GSPO loss=-2.9831
  单看 token 1 的比率是 1.350，超过 1+ε=1.2；整条序列的比率却只有 1.127。
  GRPO 会把这个 token 裁掉；GSPO 只看序列比率，仍落在 trust region 内。
```

这就是为什么 GSPO 在长轨迹 agent 上更稳：**一个 token 的异常不应该决定整条轨迹的更新**。

## 4. 环境：真正的瓶颈

Agentic RL 的环境要满足五个条件：

| 条件 | 含义 | 不满足的后果 |
|---|---|---|
| **可复现** | 同一初始状态，同样的动作，得到同样的结果 | 同一轨迹两次打分不同，reward 有噪声 |
| **可重置** | 每条轨迹从干净状态开始 | 上一条轨迹的副作用污染下一条 |
| **可验证** | 有确定的 verifier | reward 不可信，RL 学歪 |
| **可并行** | 能同时跑几百个环境 | rollout 成为瓶颈，GPU 等 CPU |
| **低成本** | 单条轨迹的时间/钱可控 | 训练成本爆炸 |

SWE-bench 类的环境尤其难：要装依赖、跑测试、处理 flaky test、隔离网络。SWE-smith（arXiv 2504.21798）的做法是：给任意 Python 代码库自动构造执行环境，再合成 100s–1000s 个"破坏现有测试"的任务实例；他们用这个管线造了 **50k** 个实例，来自 **128** 个仓库。这解决的是"训练数据不够"的问题，但环境本身仍然是几个 TB。

一个常见的工程架构：

```text
rollout workers（几百个容器/VM）
  → 执行 agent 轨迹
  → 收集 (trajectory, reward, metadata)
  → 写入 replay buffer
trainer（GPU）
  → 采样 batch
  → 算 advantage / ratio / loss
  → 更新 policy
  → 同步权重到 rollout workers
```

verl（HybridFlow，arXiv 2409.19256）就是把这个数据流抽象成"单控制器 + 多控制器"的混合模式：单控制器负责编排，多控制器负责分布式计算；3D-HybridEngine 处理训练/生成阶段的 actor 重分片。它的价值不是某个算法，而是**把 rollout 和 training 的耦合拆开**，让环境可以独立扩展。

## 5. Reward：verifier 就是新的偏好模型

Agentic RL 的 reward 大致分四类：

### 5.1 二值 verifier

测试通过=1，失败=0。最干净，但稀疏。

SWE-RL（arXiv 2502.18449）用的是"轻量规则 reward"：ground-truth 和模型解的相似度分数，而不是跑测试。他们在 Llama 3 上训出 Llama3-SWE-RL-70B，在 SWE-bench Verified 上 **41.0%**，是当时 <100B 模型的最好成绩，甚至接近 GPT-4o。更有意思的是，只在软件演化数据上做 RL，模型在五个 out-of-domain 任务上也有提升，而 SFT baseline 平均是下降的。

### 5.2 部分分 / rubric

一个任务有 5 个子目标，完成 3 个给 0.6。好处是信号更密，坏处是"部分分"可能鼓励模型做容易的子目标，而不是解决真正的问题。

### 5.3 LLM judge

用另一个模型打分。灵活，但引入 judge 的偏差、成本和被注入的风险。适合主观任务（设计、写作），不适合有确定 verifier 的任务。

### 5.4 过程奖励

每一步给一点 reward（比如"这个工具调用是否合理"）。信号最密，但最难标定；而且容易 reward hacking。

### 5.5 Reward hacking 的四个例子

- **测试作弊**：模型发现可以改测试文件而不是改代码；verifier 如果只跑测试，就会给高分。
- **空 patch**：模型学会"什么都不改"通过某些测试；需要 pass-to-pass 检查。
- **重复提交**：模型反复提交同一个 patch，赌 verifier 的随机性；需要去重和确定性环境。
- **格式投机**：模型学会输出 verifier 喜欢的格式，而不是正确答案；需要 held-out 和人工抽查。

防 reward hacking 的核心不是"更聪明的 reward"，而是**对抗性验证**：定期用独立 verifier 重评、人工抽查高分轨迹、把失败案例加回训练集。

## 6. 三个模型案例

| 模型 | 规模 | 关键数字 | RL 特点 |
|---|---|---|---|
| **SWE-RL** | Llama 3 70B | SWE-bench Verified 41.0% | 规则 reward，软件演化数据，out-of-domain 泛化 |
| **Kimi K2** | 1T total / 32B active | τ²-bench 66.1，SWE-bench Verified 65.8，SWE-bench Multilingual 47.3 | 大规模 agentic 数据合成 + 联合 RL，真实/合成环境 |
| **GLM-4.5** | 355B total / 32B active | TAU-Bench 70.1%，SWE-bench Verified 64.2%，AIME 24 91.0% | 混合推理（thinking/direct）+ 专家迭代 + RL |

三个案例的共同点：

- **都是 MoE**（Kimi K2、GLM-4.5），RL 稳定性是关键；GSPO/MuonClip 这类技术就是为 MoE 训稳服务的。
- **都强调环境**：Kimi K2 的"real and synthetic environments"，GLM-4.5 的"expert model iteration"。
- **都把 agentic 能力和 reasoning/coding 一起训**，不是单独做一个"agent 模型"。

## 7. 训练稳定性：几个工程细节

- **old logprob 的存储**：GRPO 要存每个 token 的 old logprob；长轨迹（100K token）乘以 batch，显存压力很大。GSPO 只需要序列级 logprob，省很多。
- **长度归一化**：长轨迹的 logprob 和短轨迹不可比；GSPO 的 $1/|\tau|$ 是一种处理，另一种是按 token 平均。
- **KL 约束**：是否加 KL 到 reference model，取决于任务。Agentic RL 里过度 KL 会限制探索，不加大约会 reward hacking；常见做法是小的 KL 系数 + 早停。
- **off-policy 程度**：rollout 用旧权重，训练用新权重，比率会偏；GSPO 的序列级 clip 在这里更稳。
- **环境随机性**：flaky test、网络、时间戳，都会让同一动作得到不同 reward；要么固定 seed，要么在 reward 里做多次采样平均。

## 8. 最小可运行

`code/agent/grpo.py` 实现了组内优势、token 级/序列级比率、clip 和两种 loss。它不训练模型，但把公式跑成了可验证的代码：

```bash
python3 grpo.py --test   # 断言：序列比率 = token 比率乘积；clip 边界；零方差组无梯度
python3 grpo.py          # 看一个"单 token 比率爆炸、序列比率正常"的例子
```

## 小结

- Agentic RL 的形式化是 POMDP：状态是环境，观察是工具输出，动作是工具调用，reward 通常在轨迹结尾。
- PPO 需要 critic，GRPO 用组内比较替代，GSPO 把比率从 token 级换成序列级并做长度归一化，更适合长轨迹和 MoE。
- 环境是瓶颈：可复现、可重置、可验证、可并行、低成本，五个条件缺一不可。
- Verifier 就是新的偏好模型；它的偏差会被 RL 放大。Reward hacking 要靠对抗性验证，不是靠更复杂的 reward。
- SWE-RL、Kimi K2、GLM-4.5 的共同点是：大规模环境 + 联合 RL + MoE 稳定性技术。

下一篇换一个角度：agent 的记忆和安全。为什么 MemGPT 式的分层记忆不一定比 naive RAG 强？为什么"每个动作都合规"仍然会泄密？以及 CaMeL 怎么用能力/污点追踪把注入挡在模型外面。
