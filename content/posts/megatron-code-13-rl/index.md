---
title: "Megatron 源码精读（十三）：强化学习"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "强化学习", "rlhf", "grpo"]
categories: ["训练框架"]
weight: 13
series: ["megatron-code"]
---

上一篇[《fused 算子》]({{< relref "megatron-code-12-fused-kernels" >}})讲的是手写 kernel。本篇换个子系统：`megatron/rl/`（Megatron-RL）怎么把 RL 后训练塞进既有的 Megatron 训练循环——哪些是算法，哪些是编排。

**一句话结论**：算法部分只有一行「组内标准化」；真正的设计量全在「两套节奏不同的系统怎么解耦」上——三个组件划分职责、四段异步流水线搬运数据、提交粒度 R/G/B 控制训练器能超前多少。

---

## 1. 问题：RL 后训练比 SFT 多出两条异步链路

SFT 是同步循环：读数据、前向、反向、更新。RL 后训练多出两条性质完全不同的链路。

第一条是 **rollout**：让策略模型自回归生成整段响应，天然异步、变长、延迟敏感，更适合丢给推理引擎。第二条是 **组内打分**：GRPO 必须拿到「同一 prompt 的一组 response」才能算出 advantage，单条样本进来时算不了。

所以 Megatron-RL 的核心问题不是「怎么算梯度」（复用现有循环即可），而是「怎么把推理和训练两种节奏编排在一起，又不写死耦合」。

## 2. 机制：三组件解耦 + 两套轨迹表示

### 2.1 三个角色各管一段

README 把系统拆成三个概念组件（`megatron/rl/README.md:27-38`）：

- **Agent**：持有 `InferenceInterface` 句柄，返回 `Rollout` 或 `EvaluationResponse`，负责采样参数、stop 条件、行内评估；
- **Trainer/Evaluator**：管理 rollout 生成与评估的控制流，协调（或创建）接口与 Agent；
- **Inference Interface**：提供生成端点，形态可以是 Megatron、OpenAI 或 HF。

这个划分回答的是「谁写环境、谁写调度」：环境侧只写 Agent，不需要知道训练怎么组 batch；推理侧的多样性则被接口吸收。

### 2.2 `Rollout` 与 `TokenRollout`：文本与 token 两种表示

`megatron/rl/agent/api.py` 里并排放着两种轨迹：

```python
class Rollout(AgentBaseModel):
    trajectory: list[str]           # 文本轨迹
    reward: float = None
    policy_epoch: list[list[tuple[int, int]]]

class TokenRollout(AgentBaseModel):
    trajectory: list[list[int]]     # token id
    reward: list[float] | float
    logprobs: list[list[float]] | None = None
    generation_mask: list[list[bool]] | None = None
```

前者面向「环境/评估」这种只关心文本的场合，后者面向「训练」这种需要 token 级 logprob 算梯度的场合（`agent/api.py:55-79`）。

两者都带 `policy_epoch` / `kv_cache_epoch` / `num_evictions`：按 token 段记录「这段是第几个训练迭代、哪份 KV cache 生成的」。`rl_utils` 用它算 staleness——`current_iteration - epoch`（`rl_utils.py:1018`）。这是后面敢让推理超前训练的前提：数据有多旧是可以量出来的。

`RolloutGroup`（`agent/api.py:85-99`）把同一 prompt 的多条 completion 打包，并带上 `batch_id` / `index_in_batch`，这就是 GRPO 里「一组 rollouts」的容器，也是下游恢复顺序的依据。

## 3. 代码落点：`_RolloutPipeline` 与 `calculate_grpo_advantages`

### 3.1 四个 stage 与三条队列

动态编排在 `_RolloutPipeline`（`agent/api.py:286`）里：

| stage | 位置 | 职责 |
| --- | --- | --- |
| `stage_prepare` | `api.py:337` | 过闸门、造推理工作项、塞进 `infer_queue` |
| `stage_infer` | `api.py:371` | 常驻 worker 池，取一项就跑一次生成 |
| `stage_assemble` | `api.py:406` | 按 `group_id` 攒够一组，`build_rollout` 成型 |
| `stage_consume` | `api.py:462` | 按 `batch_id` / `index_in_batch` 保序 yield |

三条队列串起它们：`infer_queue`、`assemble_queue`、`output_queue`。四个 stage 里三个是 task，`stage_consume` 是异步生成器——它就在调用方（`get_grouped_rollouts`）的 `async for` 里跑，产出一个交一个。

### 3.2 R/G/B 粒度决定什么时候放槽

真正的限流器是 `_SubmissionGate`（`api.py:229-258`），一个 `asyncio.Semaphore` 加两个「只在匹配时才动作」的判断：

```python
async def acquire_for(self, granularity):
    if self._submission == granularity:
        await self._sem.acquire()
        self.held += 1

def release_after(self, state):
    if self._release_on == state:
        self._sem.release()
```

映射关系写死在 `rollout_granularity.py:12-16`：R → `inferred`（每条 rollout 出结果就放槽）、G → `assembled`（整组拼齐才放）、B → `consumed`（整批消费完才放）。

`stage_prepare` 里三层嵌套调用 `acquire_for("B")` / `("G")` / `("R")`，看起来像重复申请，其实不匹配的层级是空操作——这样三种粒度共用同一套代码，而容量单位始终是「配置的那个粒度」。

### 3.3 `calculate_grpo_advantages`：组内标准化

算法核心反而最短（`rl_utils.py:853-877`）：

```python
group_turns = num_turns.sum(axis=-1)
reward_means = rewards.mean(axis=1, keepdims=True).repeat(group_turns)
reward_stds = rewards.std(axis=1, keepdims=True).repeat(group_turns)
rewards = rewards.flatten().repeat(num_turns.flatten())
return ((rewards - reward_means) / (1e-4 + reward_stds)).tolist()
```

也就是

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

`axis=1` 是组内维度（`rewards` 形状 `[组数, 组大小]`），所以 $\mu_g$、$\sigma_g$ 都是**每个 prompt 自己的**统计量，不是全局的；std 用 numpy 默认的总体标准差。

multi-turn 靠两处 `repeat` 处理：组统计按 `group_turns`（每组总 turn 数）展开，reward 也按每个 rollout 的 turn 数展开。因为组统计只基于「每条 rollout 一个标量 reward」，同一 rollout 切出的多个 turn 会共享同一个 advantage。

## 4. 工程细节与坑

**README 写的接口签名不是代码里的签名。** README 描述成 `.generate(prompt, **generation_args)`，实际是 `generate(request: InferenceRequest)`（`megatron/rl/inference/inference_interface.py:35-43`）：prompt 与生成参数要先由 `prepare_request` 打包。

接口能力也有要求：`Rollout` 要拿原始文本，所以请求的接口必须实现 `ReturnsRaw`。

**`output_queue` 是故意无界的。** 源码注释给了理由：给它加界等于引入第二重背压，会悄悄把 `--rl-generation-lag` 配出来的 run-ahead 削掉。

**worker 数不是常数。** 等于 `parallel_generation_tasks` 乘以「一个提交单位含几个 rollout」（R 是 1，G 是 `rollouts_per_group`，B 再乘每组 batch 的组数），非 streaming 时还要被总需求量截断（`api.py:301-310`）。

**`--rl-generation-lag` 的语义。** 它允许训练器落后几个 batch 的 rollout 生成，在飞 batch 数为该值加一，且大于 0 时要求同时开 `--rl-partial-rollouts`（训练更新权重时推理继续生成，即 off-policy）。换算成生成槽位是（`rollout_granularity.py:19-26`）：

```python
slots = args.rl_generation_lag + 1
if args.rl_submission_granularity != "B":
    slots *= args.grpo_prompts_per_step
if args.rl_submission_granularity == "R":
    slots *= args.grpo_group_size
```

**两条已知限制。** B 提交配 G 消费不被支持（直接 assert）；`filter_groups_with_same_reward` 目前是死代码——组被丢掉后不会重新生成，非流式调用方会少收组，所以构造期就 assert 掉了。

**epsilon 在分母，位置别写反。** `1e-4 + std` 是加在分母上防除零，不是「先加 1e-4 再除」。副作用是它轻微压小所有组的 advantage；全对或全错的组方差为 0，advantage 全 0、不产生梯度，而不是被放大成 inf。

## 5. 小结

- **定位**：Megatron-RL 给训练循环加 RL 后训练；README 自述截至 2025-08-27 仍在开发、对外未发布（路线图见 issue #1776）。
- **三组件解耦**：Agent（持推理句柄）/ Trainer-Evaluator（管控制流）/ InferenceInterface（提供端点）。
- **两套轨迹 + 版本戳**：`Rollout`（文本）与 `TokenRollout`（token + logprob），后者用 `policy_epoch` 量化 staleness。
- **四段流水线 + 粒度背压**：prepare → infer → assemble → consume；R/G/B 分别在某条 rollout / 整组 / 整批完成时放槽。
- **GRPO 无 critic**：组内 mean/std 标准化，epsilon 加在分母。

下一篇讲 **MoE 实现与优化**：四段流水线、路由的命名约定、aux loss 的系数，以及专家到底是 grouped GEMM 还是「一次大 GEMM」，也就是《[MoE 实现与优化]({{< relref "megatron-code-14-moe" >}})》。

---

**版本提示（2026-10）**：dev 把 `_RolloutPipeline` 与 `_SubmissionGate` 移到了 `megatron/rl/agent/rollout_pipeline.py`；`calculate_grpo_advantages` 重写为「一个 episode 一个 advantage」，零 turn 的占位 rollout 被 mask 掉并置 0（epsilon 仍在分母）。

训练主循环新增 callback 机制，`megatron/rl/callbacks.py` 的 `RLLoggingCallback` / `RLCheckpointCallback` 挂到 `on_train_start` / `on_train_step_end`；RL 侧可用 `grad_buffer_offload` 配合 `offload_grad_buffers()` 把梯度缓冲放进可替换的 NCCL mem pool。三组件解耦与 GRPO 公式未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
