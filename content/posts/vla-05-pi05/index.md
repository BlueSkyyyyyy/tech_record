---
title: "VLA 精读（五）π0.5：把 state 写成字符串，和「知识绝缘」的真相"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 6
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "pi05", "知识绝缘", "OpenPI", "跨本体"]
categories: ["LLM算法"]
---

π0.5 的论文标题里有一个关键词：**Open-World Generalization**。它要解决的不是「在训练过的厨房里把杯子放进抽屉」，而是「在没见过的家里，听懂人话，找到抽屉，完成 10 分钟的多阶段任务」。

π0 已经能做到 50 Hz 的灵巧操作，但它有两个结构性限制：

1. **state 是连续向量**，不同本体的维度、单位、语义都不同，没法直接混训；
2. **动作 expert 从头训**，VLM 的语义知识没有被充分利用。

π0.5 的两个改动正好对应这两点：把 state 离散成字符串塞进 prompt，以及「先离散预训练、后 flow 后训练」的两阶段配方。这一篇讲清楚这两个改动，以及被中文社区传得很广的「知识绝缘」到底指什么。

## 1. 把 state 写成字符串

π0 的 state 走 action expert 的连续输入：一个线性层把 state 投到 expert 的 width，再拼上时间。π0.5 把 state 变成了**语言 prompt 的一部分**：

```python
# openpi/models/tokenizer.py:21-27
if state is not None:
    # This is the Pi05 format, where the state is part of the discrete language input.
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
```

几个细节：

- **256 档**：`np.linspace(-1, 1, 257)[:-1]` 是 256 条边界，`digitize - 1` 得到 0..255。这和 OpenVLA 动作的 255 档不同（上一篇讲过）。
- **state 必须先在 [-1, 1]**：tokenizer 假设输入已经按 q01/q99 归一化。
- **prompt 格式**：`Task: {instruction}, State: {numbers};\nAction: `。state 是一串十进制整数，比如 `128 64 200 ...`，不是特殊 token。
- **长度**：π0.5 的 `max_token_len=200`（π0 是 48），因为 state 字符串 + 长指令会占很多 token。

这个改动的意义不在「字符串比向量好」，而在**统一接口**：任何本体，只要能把 state 归一化到 [-1,1]、离散成 256 档，就能变成同一段文本。7 维单臂是 7 个数，14 维双臂是 14 个数，人形 132 维是 132 个数——VLM 的 tokenizer 不需要知道维度，只需要读数字。

代价也在那里：132 维的 state 很容易超过 200 个 token（数字通常被拆成多个 digit token），`max_token_len=200` 会被撑爆。所以 π0.5 主要面向 18–19 维的移动双臂，不是 132 维人形。GR00T 走的是另一条路（每本体一个 MLP），下一篇讲。

## 2. adaRMS：时间注入的第二种写法

π0 把时间 $\tau$ 和动作 token 拼接后过 MLP：

$$
h = W_3\,\mathrm{swish}\left(W_2\,\mathrm{concat}(W_1 a^\tau,\ \phi(\tau))\right)
$$

π0.5 改成**单独的时间 MLP + adaRMSNorm**：

$$
c = \mathrm{swish}\left(W_2\,\mathrm{swish}(W_1 \phi(\tau))\right)
$$

然后把 $c$ 作为自适应 RMSNorm 的调制参数，注入 action expert 的每一层。

区别在哪？

- π0：时间信息只进入**输入层**，后面靠 self-attention 传播；
- π0.5：时间信息进入**每一层的归一化**，类似 DiT 的 adaLN。

后者对去噪步更敏感，适合「同一个网络要在 10 个不同噪声水平上工作」的场景。代价是多了一组 per-layer 的调制参数，但相对 300M 的 expert 可以忽略。

代码里两条路径是并列的：

```python
# openpi/models/pi0.py:162-179
if self.pi05:
    time_emb = self.time_mlp_in(time_emb)
    time_emb = nnx.swish(time_emb)
    time_emb = self.time_mlp_out(time_emb)
    time_emb = nnx.swish(time_emb)
    action_expert_tokens = action_tokens
    adarms_cond = time_emb
else:
    time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
    action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
    ...
```

同一个 `Pi0` 类，`pi05=True` 就切到 adaRMS。这也是 openpi 代码复用得比较好的地方。

## 3. 两阶段训练：α=0 与 α=10

π0.5 的损失是联合的（p.5）：

$$
\mathbb{E}\left[\mathcal{H}\left(x_{1:M},\ f^\ell_\theta(o_t,\ell)\right) + \alpha\left\|\omega - a_{t:t+H} - f^a_\theta(a^{\tau,\omega},o_t,\ell)\right\|^2\right]
$$

第一项是文本 token（含 FAST 动作 token）的交叉熵，第二项是 flow matching 的速度回归。

训练分两阶段：

| 阶段 | α | 动作表示 | 训什么 |
|---|---|---|---|
| 预训练 | **0** | FAST 离散 token | 只训 VLM/离散 token，action expert 不参与 |
| 后训练 | **10.0** | flow matching 连续 | 加 action expert，联合训 |

预训练 280k 步，后训练 80k 步（p.7）。

这个设计的逻辑是：**先让 VLM 学会「动作也可以是一种语言」**，再把连续动作的 expert 接上去。如果一上来就训 flow，VLM 的语义表征会被连续动作的梯度带偏；先训离散 token，VLM 学到的是「任务 → 动作序列」的高层映射，后训练只需要在它上面加一个精细的连续头。

论文的分解（p.5）把这个意图写成了概率式：

$$
\pi_\theta(a_{t:t+H}, \hat\ell \mid o_t, \ell) = \pi_\theta(a_{t:t+H} \mid o_t, \hat\ell)\cdot\pi_\theta(\hat\ell \mid o_t,\ell)
$$

并且明确说："the action distribution does not depend on $\ell$, only on $\hat\ell$." 也就是说，低层动作只依赖**预测出来的子任务** $\hat\ell$，不直接依赖原始指令 $\ell$。这是**分层控制**：VLM 先决定「做什么」，action expert 再决定「怎么做」。

## 4. 「知识绝缘」到底指什么

中文社区常说 π0.5 用了「知识绝缘（knowledge insulation）」。需要澄清：**π0.5 论文里没有这个词**（全文 grep "insul" 零命中）。这个词来自后来的一篇论文 *Knowledge Insulating Vision-Language-Action Models*（arXiv 2505.23705），它把 π0.5 的机制总结成了「知识绝缘」。

π0.5 论文里实际写的是三个机制：

### 4.1 注意力单向性

p.19 的原话：

> "In effect, information flows unidirectionally from the VLM to the action expert; no VLM embedding attends to the action expert."

也就是上一篇讲的 block-causal mask：prefix（图像/语言/state）看不到 suffix（动作），suffix 可以看 prefix。

### 4.2 两种动作表征互不可见

p.19 的 Fig.18 里，FAST 动作 token 和 flow 动作 token 之间**互相不 attend**：

> "to avoid information leakage between the two representations of actions"

这条很容易被忽略。π0.5 同时有离散 token 和连续 token 两套动作表征，如果它们互相可见，离散 token 的预训练知识会污染连续动作的流场，反之亦然。mask 把它们隔开，只在 prefix 处共享 VLM 表征。

### 4.3 分阶段解耦 + 独立专家

预训练 α=0，VLM 不被动作目标扰动；后训练才随机初始化 action expert（p.7），两套权重只通过 self-attention 交换信息。

所以「知识绝缘」不是某一个模块，而是**三个机制的组合**：单向 mask + 表征互斥 + 分阶段训练。它的目的是让 VLM 的语义知识能流向动作，但动作梯度不回传污染 VLM。

> **类比的边界**：这像一个公司里「战略部门」和「执行部门」的关系。战略部门（VLM）看得到执行部门的报告，但执行部门不能反向改战略部门的决策逻辑；两边用不同的 KPI（离散 token 的 CE vs 连续动作的 MSE）训练。完全隔离会失去协同，完全不隔离会让执行细节污染战略判断。

## 5. 数据：400 小时移动双臂与 97.6%

π0.5 的预训练数据分五类（p.6）：

| 代号 | 内容 | 规模 |
|---|---|---|
| MM | 移动双臂家务 | **~400 小时、~100 个家庭** |
| ME | 非移动单/双臂多环境 | 环境多、本体不同 |
| CE | 实验室跨本体 + OXE | π0 数据的扩展 |
| HL | 人工标注子任务文本 | 先预测 bounding box，再预测子任务 |
| WD | web 数据（CapsFusion、COCO、Cambrian-7M、PixMo、VQAv2 等） | 视觉语义 |

一个关键数字：**第一阶段 97.6% 的训练样本不来自移动双臂家务数据**（p.2）。换句话说，只有 2.4% 是「目标域」数据，其余都是别的本体、别的环境、web 数据。这就是「开放世界泛化」的来源——不是靠 400 小时目标数据，而是靠 97.6% 的异构数据把视觉和语义先验打进去。

后训练用 MM + ME（过滤成功且短于阈值）+ WD + 部分 HL + **VI（语言示范）**，去掉 CE。VI 只占高层移动操作样本的 **~11%**，但去掉后显著变差（p.11）。

最终模型覆盖 **104 个训练地点**（p.9）。

## 6. 结果：10–15 分钟的长程任务

π0.5 的评测在 **3 个未见过的新家**（3 kitchen + 3 bedroom）做，任务是 items in drawer、laundry basket、dishes in sink，每任务每环境 **10 trials**（p.7–8）。

几个值得记的结论：

- 能执行 **10–15 分钟**的长程操作（p.2）；新家中单任务 2–5 分钟多阶段（p.8）。
- 全量 π0.5 显著优于 π0 和 π0-FAST+Flow；即使把 π0 训到 **300k steps** 仍成立（p.10）。
- 高层推理 ablation（p.11）：全量 π0.5 **优于 human HL "oracle"**；第二名是 implicit HL（运行时不做高层推理，但训练里有子任务数据）；no HL 显著更差；no VI 显著更差；no WD 显著更差；零样本 GPT-4 做高层最差。
- 组件 ablation：去掉 ME 或 CE 都显著掉；no WD 在整体任务上**不显著**，但对 **OOD 物体的语言跟随**显著变差（p.10）。
- 论文对 web 数据的确切主张："much of the benefit of web data (perhaps unsurprisingly) lies in improving the high-level policy."（p.11）

最后一条很重要：web 数据提升的是**高层子任务选择**，不是低层控制。这和 RT-2 的结论一致——VLM 的知识买的是语义，不是动作精度。

「优于人类 oracle」也要限定：只在**高层子任务选择**这个维度、且是 Figure 13 的 mock home 任务上（p.11）。不是「整个任务比人强」。

## 7. 代码：π0.5 在 LeRobot 里长什么样

LeRobot 的 `pi05` 实现和 openpi 基本一一对应：

- `embed_prefix` 里把 state 字符串 tokenize 进语言 prompt；
- `embed_suffix` 用 adaRMS 注入时间；
- 训练时 `make_flow_matching_inputs(..., convention=FlowConvention.NOISE_AT_ONE, prefix_mask=prefix_mask)`（`modeling_pi05.py:886`）；
- 推理时先自回归解码子任务文本（可选），再 10 步 flow 去噪。

一个容易踩的坑：**π0.5 论文写 "action horizon of 50, i.e. H = 49"**（p.19），但代码 `chunk_size=50`（`configuration_pi05.py:38`）。论文自己 off-by-one。引用时写 50 步、并注明论文写 H=49。

## 8. 局限

论文自己列的（p.11）：

- 不熟悉的抽屉把手、难开的柜子会失败；
- 部分可观测（手臂挡住污渍）；
- 高层子任务推理 "easily distracted"，会反复开关抽屉。

没列的但很明显的：

- 只验证了移动双臂（18–19 维），没验证高维人形；
- state 字符串会随维度线性增长，`max_token_len=200` 是硬上限；
- 依赖大量人工标注的子任务文本（HL）和语言示范（VI），数据成本高。

## 小结

- π0.5 的两个核心改动：state 离散成 256 档字符串塞进 prompt；adaRMS 注入时间。
- 两阶段训练：预训练 α=0 只训 FAST 离散 token，后训练 α=10 才加 flow expert。
- 「知识绝缘」不是 π0.5 论文的词，而是后来论文的总结；机制是单向 mask + 两种动作表征互斥 + 分阶段解耦。
- 97.6% 的训练样本不是目标域数据；开放世界泛化靠的是异构数据，不是 400 小时目标数据。
- web 数据提升高层子任务选择，不提升低层控制；「优于人类 oracle」只在这个维度上成立。
- 论文写 H=49，代码是 50；引用时注意。

下一篇换一条技术路线：[GR00T N1/N1.7 的双系统与本体条件化]({{< relref "vla-06-groot" >}})——不共享动作 expert，而是给每个本体一套 MLP。
