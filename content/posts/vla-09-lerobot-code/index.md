---
title: "VLA 精读（九）LeRobot 代码走读：π0/π0.5 的 PyTorch 版"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 10
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "LeRobot", "pi0", "代码走读", "PyTorch"]
categories: ["LLM算法"]
---

openpi 是 JAX 写的，读起来门槛高；LeRobot 的 `pi0`/`pi05` 是 PyTorch 移植，数学一一对应，更适合作为代码走读的入口。

这一篇按数据流走：`配置 → 模型组装 → embed_prefix → embed_suffix → forward → sample_actions → RTC`。每一步都给出文件、行号和关键代码，最后给一个「怎么读」的检查清单。

> 代码版本：LeRobot `ae32126a`，openpi `215abfb2`。行号以这两个 commit 为准。

## 1. 配置：所有魔法数字都在这里

`configuration_pi0.py`：

```python
# lerobot/policies/pi0/configuration_pi0.py:33-50
paligemma_variant: str = "gemma_2b"
action_expert_variant: str = "gemma_300m"
chunk_size: int = 50          # openpi 里的 action_horizon
n_action_steps: int = 50      # 实际执行多少步
max_state_dim: int = 32
max_action_dim: int = 32
num_inference_steps: int = 10
time_sampling_beta_alpha: float = 1.5
time_sampling_beta_beta: float = 1.0
time_sampling_scale: float = 0.999
time_sampling_offset: float = 0.001
```

对照论文：

| 参数 | 代码 | 论文 |
|---|---|---|
| chunk | 50 | H=50 |
| 推理步数 | 10 | 10 |
| 动作维度 | 32（填充上界） | 18（最大机器人） |
| 时间采样 | Beta(1.5,1)×0.999+0.001 | $p(\tau)=\mathrm{Beta}((s-\tau)/s;1.5,1)$ |

`n_action_steps` 是「实际执行多少步」，默认等于 `chunk_size`。π0 论文里 50 Hz 机器人执行 25 步、20 Hz 执行 16 步，对应把 `n_action_steps` 设成 25/16。**模型输出 50 步，不代表控制器执行 50 步**——这是部署时最常调的参数。

## 2. 模型组装：两个 Gemma 共享注意力

`PaliGemmaWithExpertModel`（`modeling_pi0.py:185`）把两个 Gemma 拼在一起：

- `paligemma`：Gemma-2B + SigLIP，处理 prefix；
- `gemma_expert`：Gemma-300M，处理 suffix；
- 两者**共享 self-attention 的 K/V**：expert 的每一层 attention 会接收 paligemma 的 K/V，拼在自己的 K/V 前面。

这正是「两套权重、一个 transformer」的实现方式。`forward` 的签名：

```python
# modeling_pi0.py:311
def forward(self, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond)
```

`inputs_embeds=[prefix_embs, suffix_embs]`，返回 `(prefix_out, suffix_out)`。当 `use_cache=True` 且只传 suffix 时，`prefix_out` 是 `None`，只返回 suffix 的输出。

## 3. embed_prefix：图像 + 语言，全是 0 边界

```python
# modeling_pi0.py:512-547（简化）
for img_emb in self._embed_images(images):
    embs.append(img_emb)
    pad_masks.append(...)
    att_masks += [0] * num_img_embs

lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
embs.append(lang_emb)
att_masks += [0] * num_lang_embs
```

`att_masks` 全 0 意味着**整个 prefix 是一个双向块**：图像 token 可以看语言，语言可以看图像，没有因果 mask。

图像编码器是 SigLIP，每路 256 个 patch（224×224 / 14）。三路相机就是 768 个 token。

## 4. embed_suffix：state 单独一块

```python
# modeling_pi0.py:549-610（简化）
state_emb = self.state_proj(state)
embs.append(state_emb[:, None, :])
att_masks += [1]                      # state 是块边界

time_emb = create_sinusoidal_pos_embedding(timestep, ...)
action_emb = self.action_in_proj(noisy_actions)
action_time_emb = torch.cat([action_emb, time_emb], dim=2)
action_time_emb = self.action_time_mlp_out(F.silu(self.action_time_mlp_in(action_time_emb)))

embs.append(action_time_emb)
att_masks += [1] + ([0] * (self.config.chunk_size - 1))   # 动作块
```

注意 `att_masks += [1]`（state）和 `att_masks += [1] + [0]*(chunk-1)`（动作）是两个独立的块边界。cumsum 后：

- prefix：0
- state：1
- 动作：2

所以 mask 是三块：**prefix 看不到 state/动作；state 看不到动作；动作能看到所有**。这和上一篇的 `block_attn_mask.py` 输出一致。

时间注入是 π0 的方式：`concat(action_emb, time_emb)` 再过 MLP。π0.5 的 `embed_suffix` 不一样：它把 state 拿掉（state 已经在 prompt 里），时间走 `time_mlp_in/out` + adaRMS。

## 5. forward：训练时一次前向

```python
# modeling_pi0.py:612-660（简化）
x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)

prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)

att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
position_ids = torch.cumsum(pad_masks, dim=1) - 1

(_, suffix_out), _ = self.paligemma_with_expert.forward(...)
suffix_out = suffix_out[:, -self.config.chunk_size:]
v_t = self.action_out_proj(suffix_out)
return F.mse_loss(u_t, v_t, reduction="none")
```

几个细节：

- `make_flow_matching_inputs` 用 `NOISE_AT_ONE`，即 $x_t=t\varepsilon+(1-t)A$、$u=\varepsilon-A$，和 openpi 一致。
- `position_ids = cumsum(pad_masks) - 1`：padding token 不占位置，避免 position id 跳变。
- `suffix_out[:, -chunk_size:]`：只取最后 50 个 token 的输出，前面的 state token 不参与动作预测。
- loss 是 **MSE on velocity**，不是 L1，不是交叉熵。

## 6. sample_actions：prefix 缓存 + 10 步 Euler

```python
# modeling_pi0.py:662-723（简化）
prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(...)
_, past_key_values = self.paligemma_with_expert.forward(
    inputs_embeds=[prefix_embs, None], use_cache=True,
)

return euler_integrate(
    lambda x_t, t: self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, t),
    noise, num_steps,
    rtc_processor=self.rtc_processor,
    ...
)
```

`euler_integrate` 在 `common/flow_matching.py:256`，核心是：

```python
# 简化
time = 1.0
dt = -1.0 / num_steps
for _ in range(num_steps):
    v_t = velocity_fn(x_t, time)
    x_t = x_t + dt * v_t
    time = time + dt
```

从 $t=1$（噪声）积分到 $t=0$（动作），每步 $\Delta t=-0.1$。

`denoise_step`（`modeling_pi0.py:725`）做三件事：

1. `embed_suffix(state, x_t, timestep)` 重新编码 suffix；
2. 拼 mask：suffix 对 prefix 全可见，suffix 内部按 `att_masks`；
3. `clone_past_key_values(past_key_values)` 后前向，只算 suffix。

`clone_past_key_values` 是必须的：每步去噪都会往 cache 里写新的 K/V，如果不 clone，10 步会互相污染。这是 PyTorch 移植里最容易出的 bug 之一。

## 7. π0.5 的差异

`pi05/modeling_pi05.py` 和 π0 的结构几乎一样，差异集中在三处：

### 7.1 state 进 prompt，不进 suffix

π0.5 的 `embed_prefix` 会把 `Task: ..., State: ...;\nAction: ` tokenize 进去；`embed_suffix` 只处理动作和时间，没有 `state_proj`。

### 7.2 时间走 adaRMS

```python
# modeling_pi05.py:820-877（简化）
time_emb = self.time_mlp_in(time_emb)
time_emb = F.silu(time_emb)
time_emb = self.time_mlp_out(time_emb)
time_emb = F.silu(time_emb)
adarms_cond = time_emb
```

`adarms_cond` 传给 `PaliGemmaWithExpertModel.forward`，在 expert 的每一层做自适应 RMSNorm。π0 的 `adarms_cond` 是 `None`，时间只拼在输入里。

### 7.3 训练时带 prefix mask

```python
# modeling_pi05.py:885-887（简化）
x_t, u_t, model_time = make_flow_matching_inputs(
    actions, noise, time, convention=FlowConvention.NOISE_AT_ONE, prefix_mask=prefix_mask
)
```

`prefix_mask` 标记「哪些动作位置是干净的、不参与加噪」——这是 RTC 训练时的用法：把已经执行的前缀固定住，只训后面的部分。openpi 的 `make_flow_matching_inputs` 也有同样的参数。

## 8. RTC 怎么接进去

`euler_integrate` 的 `rtc_processor` 参数会把每一步的速度计算包一层：

```python
# rtc/modeling_rtc.py:122-160（简化）
def denoise_step(self, x_t, prev_chunk_left_over, inference_delay, time, original_denoise_step_partial, ...):
    v_t = original_denoise_step_partial(x_t)
    if prev_chunk_left_over is None:
        return v_t
    x1_t = x_t - time * v_t
    err = (prev_chunk_left_over - x1_t) * weights
    correction = torch.autograd.grad(x1_t, x_t, err)[0]
    return v_t - guidance_weight * correction
```

`weights` 来自 `get_prefix_weights(inference_delay, execution_horizon, chunk_size)`，四种 schedule 在上一篇讲过。

`inference_delay` 的单位是**步数**，不是毫秒。如果延迟 200 ms、控制 50 Hz，`inference_delay=10`。这个换算错了，guidance 会拉错位置。

## 9. 怎么读这份代码：五个检查点

如果你只想快速判断一个 LeRobot 策略的实现对不对，按这个顺序看：

1. **配置**：`chunk_size`、`n_action_steps`、`max_action_dim`、`num_inference_steps`。这四个数决定行为和延迟。
2. **embed_suffix 的 att_masks**：state 和动作是不是分开的块？如果写成 `[0]*chunk`，state 会被并进动作块，mask 就错了。
3. **forward 的 loss**：是 MSE on velocity 还是 L1？是只算动作 token 还是全序列？
4. **sample_actions 的 cache**：有没有 `clone_past_key_values`？没有的话 10 步去噪会互相污染。
5. **euler_integrate 的符号**：`dt=-1/N` 还是 `+1/N`？和训练约定是否一致？

这五点对上了，剩下的都是工程细节。

## 10. LeRobot 的局限

- **不是官方实现**：openpi 是 Physical Intelligence 的官方 JAX 代码，LeRobot 是社区移植。数值可能有细微差异（dtype、attention kernel、位置编码）。
- **π0.5 的支持不完整**：高层子任务预测、web 数据混训、完整的 knowledge insulation mask，LeRobot 只实现了推理/微调需要的部分。
- **RTC 是可选组件**：默认关闭，需要显式配置 `rtc_config`。
- **没有训练脚本的完整配方**：数据配比、归一化统计、增强策略，LeRobot 给的是接口，不是论文的复现。

所以读 LeRobot 的正确姿势是：**用它理解数学和数据流，不要用它复现论文数字**。

## 小结

- LeRobot 的 π0 是 openpi 的 PyTorch 移植，配置、mask、flow matching 一一对应。
- 三个块：prefix（图像+语言）、state、动作；state 看不到动作，动作能看到所有。
- 训练用 `NOISE_AT_ONE` 的 flow matching，MSE on velocity。
- 推理先算 prefix KV cache，再 10 步 Euler；每步 `clone_past_key_values`。
- π0.5 的差异：state 进 prompt、时间走 adaRMS、训练带 prefix mask（RTC）。
- 读代码的五个检查点：配置、att_masks、loss、cache、符号。

下一篇换一个代码库：[OpenVLA/OFT 的 binning、LoRA 与并行解码]({{< relref "vla-10-openvla-code" >}})——看离散动作和连续动作在工程上的分叉。
