---
title: "VLA 精读（四）π0：两个 Gemma、一条 mask，和 73 毫秒的账"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 5
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "pi0", "OpenPI", "Flow Matching", "模型架构"]
categories: ["LLM算法"]
---

π0 是第一个把「VLM + 连续动作 expert + flow matching」跑通到 50 Hz 机器人上的通用策略。它的论文写得很克制，没有堆砌新名词，但每一个设计决策背后都有一笔明确的账。

这一篇按「结构 → 前向 → 训练 → 推理 → 代码」的顺序拆，最后列出六个最常见的误引。

## 1. 结构：一个 transformer，两套权重

π0 的骨架是 PaliGemma，但拆成两部分（p.5, p.15）：

| 部分 | 配置 | 参数量 | 初始化 |
|---|---|---|---|
| VLM | Gemma-2B + SigLIP，width 2048、depth 18、18 heads、1 KV head、head_dim 256 | ~3B | PaliGemma 预训练 |
| action expert | width 1024、MLP 4096 | ~300M | **从零初始化** |
| 合计 | | **3.3B** | |

注意两个口径问题：

- 论文说 VLM 是 "3 billion parameter"（p.5），π0.5 论文说 "2B VLM initialized from PaliGemma weights"（p.19）。这不是矛盾，是**含不含视觉塔**的记账差异。引用时最好写「3.3B 总参数，其中 action expert 300M」，避免卷进 2B/3B 的口水。
- 论文里动作维度是 **18**（数据集里最大机器人的维度，小的零填充，p.6）；代码里是 `action_dim=32`。32 是填充上界，不是物理维度。

两套权重不是 MoE。MoE 是「每个 token 选几个 expert」，π0 是「prefix token 走 VLM 权重，suffix token 走 expert 权重」，两套权重在同一个 self-attention 里交互。论文原话是 "a mixture-of-experts-like architecture"，但实现上更接近**双塔共享注意力**。

## 2. 前向：prefix 和 suffix

一次前向分两段：

**prefix**：图像 token + 语言 token。3 路相机，每路 256 个 patch，共 768 个 token；语言指令几十个 token。prefix 内部是**双向**注意力。

**suffix**：1 个 state token + 50 个动作 token。state 先过线性层投到 expert 的 width，**单独成一块**；动作 token 是当前去噪步的 $x_t$，再过 `action_in_proj`；时间 $\tau$ 用正弦编码，和动作 token 拼接后过一个小 MLP：

$$
h = W_3\,\mathrm{swish}\left(W_2\,\mathrm{concat}(W_1 a^\tau,\ \phi(\tau))\right)
$$

这是 π0 的版本；π0.5 改成了 adaRMSNorm（下一篇讲）。

然后两段拼起来，用一条 mask 控制谁能看谁：

```python
# openpi/models/pi0.py:19-44（简化）
def make_attn_mask(input_mask, mask_ar):
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
```

`mask_ar` 是一个「块边界」标记：prefix 全 0，state 为 1，动作 token 的第一个为 1、其余为 0。cumsum 之后得到**三块**：

- prefix token 的 cumsum 都是 0，只能看 cumsum ≤ 0 的 token，也就是 prefix 自己；
- state token 的 cumsum 是 1，能看 prefix 和自己，**看不到动作**；
- 动作 token 的 cumsum 是 2，能看 prefix、state 和所有动作。

于是得到三个性质：

1. prefix 内部双向；
2. **prefix 和 state 都看不到动作**，VLM 表征不被动作梯度污染；
3. **动作 token 之间双向**，50 个动作并行去噪，不是自回归；state 作为条件被动作看到，但不反向看动作。

`code/vla/block_attn_mask.py` 把这条 mask 打印了出来：

```
            img0 img1 img2 img3 txt0 txt1 txt2 stat act0 act1 act2 act3 act4
      img0     #    #    #    #    #    #    #    .    .    .    .    .    .
      img1     #    #    #    #    #    #    #    .    .    .    .    .    .
      ...
     state     #    #    #    #    #    #    #    #    .    .    .    .    .
      act0     #    #    #    #    #    #    #    #    #    #    #    #    #
      act1     #    #    #    #    #    #    #    #    #    #    #    #    #
```

π0 的 state 在 suffix；π0.5 把 state 挪进了 prefix（作为字符串），这是两版 mask 的主要差异。

## 3. 训练：10,000 小时与 n^0.43

π0 的预训练数据（p.11）：

- **10,000 小时**自采灵巧操作数据；
- **7 种机器人**；
- **68 个任务**；
- 外加 OXE、Bridge v2、DROID；
- 按 timestep 计，开源数据占 **9.1%**，自采 **903M** timesteps（单臂 106M + 双臂 797M）。

混合权重不是按数据集平均，而是每个 task-robot 组合按 **$n^{0.43}$** 降权（p.6）。$n$ 是该组合的样本数：数据越多，权重增长越慢于线性。这是处理长尾的常用手法——不让几个大数据集淹没小任务。

训练目标就是 flow matching 的 MSE：

```python
# openpi/models/pi0.py:197-200, 212-215
time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
time_expanded = time[..., None, None]
x_t = time_expanded * noise + (1 - time_expanded) * actions
u_t = noise - actions
...
v_t = self.action_out_proj(suffix_out[:, -self.action_horizon:])
return jnp.mean(jnp.square(v_t - u_t), axis=-1)
```

时间采样是 `Beta(1.5, 1) * 0.999 + 0.001`——先抽 $\beta \sim \mathrm{Beta}(1.5,1)$，再映射到 $[0.001, 1]$。$\mathrm{Beta}(1.5,1)$ 的密度是 $1.5\sqrt{t}$，**偏向 1**，也就是偏向噪声端。论文写作 $p(\tau)=\mathrm{Beta}((s-\tau)/s;1.5,1)$、$s=0.999$（p.16），和代码等价，只是变量替换。

为什么偏向噪声端？因为高噪声区域的速度场最难学，多采样一些。这是 flow matching 训练的常见 trick，不是 π0 独有。

主模型训练 **700k steps**；作为算力对齐的 "compute parity" 版是 **160k steps**（p.7）。batch size 论文没给。

## 4. 推理：73 ms 是怎么来的

π0 在 RTX 4090 上的延迟分解（p.16, Table I）：

| 阶段 | 耗时 |
|---|---|
| 图像编码（3 路） | 14 ms |
| observation forward（prefix） | 32 ms |
| 10 次 action forward（suffix） | 27 ms |
| 网络延迟（off-board） | 13 ms |
| **on-board 合计** | **73 ms** |
| **off-board 合计** | **86 ms** |

三个数字要分开看：

- **73 ms** 是模型在机器人本地的计算时间，对应约 **13.7 Hz**；
- **86 ms** 是加上网络传输的远程推理；
- **50 Hz** 是机器人控制频率，不是模型推理频率。

π0 的执行策略是：50 Hz 机器人每 **0.5 s** 推理一次，一次执行 **25** 个动作；20 Hz 的 UR5e/Franka 每 **0.8 s** 推理一次，执行 **16** 个动作（p.16）。也就是说，模型只跑了 2 Hz 的决策，靠 chunk 把 50 Hz 的控制填满。

这就是上一篇说的「chunk 藏延迟」。73 ms 相对于 500 ms 的决策周期只占 15%，机器人不需要停；但如果 chunk 只有 5 步（100 ms），73 ms 就占 73%，每次推理都会造成明显停顿。

KV cache 是这个账能成立的关键：

```python
# openpi/models/pi0.py:234-237
prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
positions = jnp.cumsum(prefix_mask, axis=1) - 1
_, kv_cache = self.PaliGemma.llm([prefix_tokens, None], ...)
```

prefix 只前向一次，10 步去噪每步只算 51 个 suffix token。`block_attn_mask.py` 算过：P=768、S=51 时，有缓存比无缓存少 **6.7 倍**的注意力元素。

代码里还有一句很诚实的注释：

```python
# openpi/models/pi0.py:224-225
# note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
# distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
```

论文的 $\tau=0$ 是噪声、$\tau=1$ 是动作；代码的 $t=1$ 是噪声、$t=0$ 是动作。同一个路径，方向相反。读代码时如果不注意，会把速度场的符号搞反。

## 5. 训练配方与结果

π0 的后训练数据量差异极大：最简单任务 **5 小时**，最复杂 **100+ 小时**（p.6）。这解释了为什么同一个模型在不同任务上的成功率差别很大——不是模型不行，是后训练预算不同。

结果方面，论文的表述比较保守：

- 开箱任务（叠衣服、简单的收拾）"near perfect success rates"（p.7）；
- 复杂多阶段任务 "more than 50% of the maximum score"（p.11），每任务 10 trials；
- 没有给出一个统一的「平均成功率」数字。

这在一众「SOTA +X%」的论文里反而可信。π0 的贡献不是某个 benchmark 的分数，而是**证明了这条架构路线可以跑 50 Hz**。

## 6. 六个常见误引

1. **「π0 是 3B」**：总参数 3.3B，其中 VLM 3B、expert 300M。说 3B 会漏掉 expert。
2. **「50 Hz 是模型推理频率」**：50 Hz 是机器人控制频率；模型推理是 13.7 Hz（73 ms），靠 chunk 填满。
3. **「Beta(1.5,1) 直接采样 τ」**：实际是 `Beta(1.5,1)*0.999+0.001`，且论文的 τ 与代码的 t 方向相反。
4. **「π0 用了 temporal ensembling」**：论文明确说试过，**掉点**，所以改成开环执行整块（p.16）。
5. **「10,000 小时包括 OXE」**：10,000 小时是自采数据（7 机器人 / 68 任务），OXE/Bridge/DROID 是额外叠加；开源数据只占 9.1%。
6. **「action_dim=32 是物理维度」**：32 是填充上界；论文里最大机器人是 18 维。

## 7. 和 π0.5 的关系

π0.5 不是 π0 的放大版，而是**换了训练范式和数据**：

- π0：纯 flow matching，动作 expert 从头训；
- π0.5：预训练用 FAST 离散 token（α=0），后训练才加 flow expert（α=10）；
- π0.5 把 state 离散成字符串塞进 prompt，支持异构本体；
- π0.5 的数据从「10,000 小时自采」扩展到「400 小时移动双臂 + 100 个家庭 + web 数据」。

架构骨架没变，但「怎么用 VLM 的知识」变了。下一篇专门讲这个。

## 小结

- π0 = PaliGemma 3B + 300M action expert，总 3.3B；两套权重共享 self-attention。
- 一条 block-causal mask 实现三件事：prefix 双向、prefix 看不到 suffix、suffix 内部双向。
- 训练用 flow matching，$x_t=t\varepsilon+(1-t)A$、$u=\varepsilon-A$，时间采样偏向噪声端。
- 推理 73 ms（on-board），13.7 Hz；靠 50 步 chunk 填满 50 Hz 控制。
- KV cache 把 10 步去噪的成本压到只算 51 个 suffix token，是实时性的前提。
- 论文的 τ 和代码的 t 方向相反；这是读 openpi 时最容易踩的坑。

下一篇：[π0.5 如何把 state 写成字符串]({{< relref "vla-05-pi05" >}})，以及「知识绝缘」到底指什么。
