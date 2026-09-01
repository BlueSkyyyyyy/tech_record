---
title: "Megatron 源码精读（十三）：强化学习"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "强化学习", "rlhf", "grpo"]
categories: ["训练框架"]
weight: 13
series: ["megatron-code"]
---

上一篇[《fused 算子》]({{< relref "megatron-code-12-fused-kernels" >}})讲了手写 CUDA 的融合。本篇话题切换到一个新的子系统——Megatron-RL（`megatron/rl/`），它给 Megatron-LM 的训练循环加进**强化学习后训练**（RLHF/GRPO 这类），让「推理采样 rollout → 打分 → 策略更新」这条链路跑在同一套框架里。

需要注意 Megatron-RL 的定位（`megatron/rl/README.md:3-4`）：截至 2025 年 8 月仍在积极开发，内部可用但**对外尚未完整发布**。所以本篇少谈「怎么跑通」，多谈它**解耦的架构设计**和**几个算法核心**，这比 API 细节更有价值。

---

## 1. 问题背景：RL 后训练和普通 SFT 训练有什么本质不同

普通训练是「读数据 → 前向 → 反向 → 更新」，是纯同步的、数据驱动的循环。RL 后训练多出两条性质完全不同的链路：

1. **推理采样（rollout）**：让策略模型**自回归生成**整段响应，这天然是异步、可变长、延迟敏感的（和异步推理引擎更亲）；
2. **组内打分 + advantage 归一化**：GRPO 要对「同一个 prompt 生成的一组 response」分别打分，再做**组内标准化**得到 advantage，然后才进入常规反向传播。

所以 Megatron-RL 的核心设计问题不是「怎么算梯度」（复用现有训练循环），而是：

- **怎么把「推理」和「训练」两套节奏不同的东西编排在一起**，又不把它们写死耦合；
- **怎么把「rollout 的并发、限流、组组装」这些 RL 特有的工程问题抽象干净**。

README（`megatron/rl/README.md:19-38`）给出的答案是三组件解耦。

---

## 2. 现象：三个角色各管一段

`megatron/rl/README.md` 把系统拆成三个概念组件（`megatron/rl/README.md:27-38`）：

- **Agent（环境+智能体）**：持有 `InferenceInterface` 的句柄，返回 `Rollout` / `EvaluationResponse`。它决定「采样什么参数、用什么生成参数（stop 条件、评估方式）」；
- **Trainer/Evaluator**：管 rollout 生成与评估的控制流，协调（或创建）`InferenceInterface` 和 Agent；
- **Inference Interface**：给环境提供 `.generate(prompt, **generation_args)` 的端点，有多种形态（Megatron / OpenAI / HF）。

这一层的代码就是 `megatron/rl/agent/api.py` 和 `megatron/rl/inference/api.py` 里的一堆类型定义。看几个关键数据结构：

```python
# megatron/rl/agent/api.py:55-79（节选）
class Rollout(AgentBaseModel):
    trajectory: list[str]
    reward: float = None
    env_id: str = ''

class TokenRollout(AgentBaseModel):
    trajectory: list[list[int]]      # token id 表示
    reward: list[float] | float
    logprobs: list[list[float]] | None
```

注意 `Rollout`（字符串轨迹）和 `TokenRollout`（token id + logprob 轨迹）是两套并行表示——前者面向「环境/评估」这种只关心文本的场合，后者面向「训练」这种需要 token 级 logprob 算梯度的场合。`RolloutGroup`（`agent/api.py:85-99`）则把「同一个 prompt 的多条 completion」打包成一个组，并带上 `batch_id` / `index_in_batch` 元数据——这就是 GRPO 里「一组 rollouts」的容器。

---

## 3. 根因一：grouped rollout 的异步流水线 + 背压

三组件解耦是「静态」的；「动态」的编排核心在 `_RolloutPipeline`（`megatron/rl/agent/api.py:286-487`）。它把一次 grouped rollout 拆成四个异步 stage：

- `stage_prepare`（`api.py:337`）：生成受控的推理工作项；
- `stage_infer`（`api.py:371`）：持久化的 inference worker 池；
- `stage_assemble`（`api.py:406`）：把推理结果拼回完整的 rollout 组；
- `stage_consume`（`api.py:462`）：按序 yield 给下游（训练循环）。

四个 stage 之间用 asyncio 队列串起来（`infer_queue` / `assemble_queue` / `output_queue`，`api.py:311-316`）。真正控制「能跑多远」的是 `_SubmissionGate`（`api.py:229-258`）——一个基于 `asyncio.Semaphore` 的限流闸：

```python
# megatron/rl/agent/api.py:246-258（节选）
async def acquire_for(self, granularity: SubmissionGranularity) -> None:
    if self._submission == granularity:
        await self._sem.acquire()
        self.held += 1

def release_after(self, state: ReleaseState) -> None:
    if self._release_on == state:
        self._sem.release()
        self.held -= 1
```

理解这段的关键是 `granularity` 这个抽象（`megatron/rl/rollout_granularity.py:7-16`）：

```python
# megatron/rl/rollout_granularity.py:7-16
SubmissionGranularity = Literal["R", "G", "B"]
ReleaseState = Literal["inferred", "assembled", "consumed"]
RELEASE_STATE_BY_SUBMISSION = {"R": "inferred", "G": "assembled", "B": "consumed"}
```

「提交粒度」R（单个 rollout）/ G（一个组）/ B（整个 batch）对应「闸门在哪个阶段释放」。例如 `submission=="G"` 时，闸在「组组装完成（assembled）」就释放一个槽位，而不是等整批消费完——这直接决定了「训练循环比推理引擎超前多少」（run-ahead，由 `--rl-generation-lag` 控制）。`get_rl_parallel_generation_tasks`（`rollout_granularity.py:19-26`）把这个 lag 换算成 generation 槽位数。一句话总结：**异步流水线把「采样快慢不一」和「训练要按序消费」靠闸门 + 粒度参数解耦。**

---

## 4. 根因二：GRPO 的 advantage 就是「组内标准化」

算法核心反而短。GRPO 不需要 critic，它的 advantage 是**组内 reward 归一化**，落在 `calculate_grpo_advantages`（`megatron/rl/rl_utils.py:853-877`）：

```python
# megatron/rl/rl_utils.py:862-877（节选）
rewards = np.array(rewards)
group_turns = num_turns.sum(axis=-1)
reward_means = rewards.mean(axis=1, keepdims=True).repeat(group_turns)
reward_stds = rewards.std(axis=1, keepdims=True).repeat(group_turns)
rewards = rewards.flatten().repeat(num_turns.flatten())
return ((rewards - reward_means) / (1e-4 + reward_stds)).tolist()
```

三行拆开看：

1. `rewards.mean(axis=1)` / `rewards.std(axis=1)`：`axis=1` 是「组内」维度（形状 `[g, group_size]`，g 是组数），所以这算出的是**每个组自己的均值和标准差**——组内标准化，不是全局标准化；
2. `repeat(group_turns)`：multi-turn（多轮对话）时，一个 rollout 会被拆成多个 turn，每个 turn 复用同一个组均值/方差；
3. `(rewards - mean) / (1e-4 + std)`：标准 z-score 形式，`1e-4` 是防除零的 epsilon。

这就是 GRPO 论文里 `A = (r - mean(r)) / std(r)` 的逐行实现。它把「每个 prompt 的一组采样」当成一个小 batch 做归一化，让「相对比组内其他人好多少」成为梯度信号，从而免去训练一个独立的 value 网络。

---

## 5. 小结

- **定位**：Megatron-RL 给训练循环加 RL 后训练，仍在开发、对外未完整发布（`megatron/rl/README.md:3-4`）。
- **三组件解耦**：Agent（环境+智能体）/ Trainer-Evaluator / InferenceInterface，各持边界（`megatron/rl/README.md:27-38`）。
- **双套数据表示**：`Rollout`（字符串，面向环境）与 `TokenRollout`（token id + logprob，面向训练），`RolloutGroup` 打包组（`agent/api.py:55-99`）。
- **异步四阶段 + 闸门背压**：`_RolloutPipeline` 的 prepare→infer→assemble→consume，靠 `_SubmissionGate` 和 `R/G/B` 提交粒度控制 run-ahead（`agent/api.py:229-258`、`rollout_granularity.py:7-26`）。
- **GRPO advantage = 组内标准化**：`(r - group_mean) / (1e-4 + group_std)`，无 critic（`rl_utils.py:853-877`）。

下一篇讲 **MoE 实现与优化**：专家并行的切分、`moe_layer` 的 token 路由（top-k + aux loss）、以及 expert capacity / dropless 这些把稀疏专家塞进密集硬件的工程手段。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/rl/README.md`、`megatron/rl/agent/api.py`、`megatron/rl/inference/api.py`、`megatron/rl/rollout_granularity.py`、`megatron/rl/rl_utils.py`。）
