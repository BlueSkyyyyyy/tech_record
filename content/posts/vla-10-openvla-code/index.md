---
title: "VLA 精读（十）OpenVLA 代码走读：binning 与 OFT 的并行解码"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 11
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "OpenVLA", "OpenVLA-OFT", "代码走读", "LoRA"]
categories: ["LLM算法"]
---

OpenVLA 和 OpenVLA-OFT 是同一个代码库的两个分支，正好代表两条动作路线：

- **OpenVLA**：7B VLM 自回归输出 7 个离散 token，每步推理 0.33 s；
- **OFT**：同样的基座，换成并行解码 + L1 回归 + chunk，LIBERO 从 76.5% 提到 97.1%，吞吐 26×。

这一篇走读两条路线的关键代码，重点看三件事：binning 到底怎么做的、loss 怎么只算动作 token、OFT 的并行解码为什么能快 26 倍。

> 代码版本：OpenVLA `c8f03f48`，OFT `e4287e94`。

## 1. ActionTokenizer：256 条边界，255 个中心

OpenVLA 的动作 tokenizer 只有几十行，但注释把坑都写清楚了：

```python
# prismatic/vla/action_tokenizer.py:31-36
self.bins = np.linspace(min_action, max_action, self.n_bins)
self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
self.action_token_begin_idx: int = int(self.tokenizer.vocab_size - (self.n_bins + 1))
```

- `bins` 是 256 条**边界**；
- `bin_centers` 是 255 个**中心**；
- 动作 token 从词表倒数第 257 个开始，覆写最后 256 个 token。

编码：

```python
# action_tokenizer.py:41-46
action = np.clip(action, a_min=-1, a_max=1)
discretized_action = np.digitize(action, self.bins)
return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
```

`np.digitize` 返回 `[1, 256]` 的索引（因为 256 条边界），用 `vocab_size - index` 映射到词表末尾。

解码：

```python
# action_tokenizer.py:65-68
discretized_actions = self.tokenizer.vocab_size - action_token_ids
discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)
return self.bin_centers[discretized_actions]
```

注释里解释得很清楚：

> "Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the digitization returns bin indices between [1, # bins], inclusive, when there are actually only (# bins - 1) bin intervals."

所以「256 bins」在 OpenVLA 里实际是 **255 个区间**，第 256 个索引被 clip 到第 255 个中心。这是上一篇算的「有效档位 255」的代码来源。

## 2. Loss 只算动作 token

OpenVLA 的微调脚本里，labels 的构造是：

```python
# prismatic/vla/datasets/datasets.py:63
labels[:-(len(action_tokens) + 1)] = IGNORE_INDEX
```

- 前向：图像 + prompt + 7 个动作 token + EOS；
- labels：prompt 和图像位置全是 `-100`（`IGNORE_INDEX`），只有动作 token 和 EOS 参与 loss。

这和 RT-2 不同。RT-2 没有明确说只算动作 token，而 OpenVLA 明确写了 "evaluating the cross-entropy loss on the predicted action tokens only"（p.5）。这个细节对训练稳定性影响很大：如果 prompt 也参与 loss，模型会花容量去复述指令。

## 3. predict_action：自回归 7 个 token

```python
# modeling_prismatic.py:506-536（简化）
generated_ids = self.generate(input_ids, max_new_tokens=self.get_action_dim(unnorm_key))
predicted_action_token_ids = generated_ids[0, -action_dim:].cpu().numpy()
discretized_actions = self.vocab_size - predicted_action_token_ids
discretized_actions = np.clip(discretized_actions - 1, 0, self.bin_centers.shape[0] - 1)
normalized_actions = self.bin_centers[discretized_actions]

action_high, action_low = np.array(stats["q99"]), np.array(stats["q01"])
actions = 0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low
```

三个点：

1. `generate` 是标准的 HuggingFace 自回归解码，`max_new_tokens=7`；
2. 反归一化用 **q01/q99**，不是 mean/std；
3. 7 个 token 是**串行**生成的，每个 token 都要过一遍 7B 模型——这就是 0.33 s/步的来源。

即使有 KV cache，7 次前向的延迟也远大于一次。OFT 的 26× 主要就是把这 7 次串行变成 1 次并行。

## 4. OFT 的并行解码

OFT 的核心改动在 `_regression_or_discrete_prediction`：

```python
# modeling_prismatic.py:877-942（简化）
# 1. 把动作位置的输入 embedding 清零
all_actions_mask = all_actions_mask.unsqueeze(-1)
input_embeddings = input_embeddings * ~all_actions_mask

# 2. 一次前向
language_model_output = self.language_model(
    input_ids=None, attention_mask=multimodal_attention_mask,
    inputs_embeds=multimodal_embeddings, output_hidden_states=True,
)

# 3. 取动作位置的 hidden states
actions_hidden_states = last_hidden_states[
    :, NUM_PATCHES + NUM_PROMPT_TOKENS : NUM_PATCHES + NUM_PROMPT_TOKENS + ACTION_DIM * NUM_ACTIONS_CHUNK, :
]

# 4. L1 head 回归
normalized_actions = action_head.predict_action(actions_hidden_states)
```

三个关键设计：

### 4.1 动作位置输入置空

自回归模型在训练时，动作位置的输入是「上一个 token 的 embedding」；并行解码没有「上一个」，所以把动作位置的输入 embedding 全部置零。

模型只能靠 **position id + 前面的 prompt** 来区分「这是第几个动作」。这听起来信息不足，但实验证明够用——因为动作 token 的位置和 prompt 的语义已经决定了它该输出什么。

### 4.2 双向 attention

OFT 把 causal mask 换成 **bidirectional mask**，让 8 个动作位置可以互相看。这样一次前向就能出 8 个动作，而不是 8 次。

代价是：训练时不能用标准的 next-token 目标，必须用 L1 回归（或 diffusion）。OFT 选 L1。

### 4.3 L1 head 的输入是拼接的 hidden states

```python
# action_heads.py:84-107
class L1RegressionActionHead(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=4096, action_dim=7):
        self.model = MLPResNet(
            num_blocks=2, input_dim=input_dim * ACTION_DIM, hidden_dim=hidden_dim, output_dim=action_dim
        )

    def predict_action(self, actions_hidden_states):
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, NUM_ACTIONS_CHUNK, -1)
        return self.model(rearranged)
```

- 输入：`(B, chunk*7, 4096)` 的 hidden states；
- reshape 成 `(B, chunk, 4096*7)`；
- 2 个 MLPResNet block，输出 `(B, chunk, 7)`。

注意 `input_dim * ACTION_DIM` = 4096×7 = 28672，这是一个很大的输入层。OFT 的 LIBERO 版可训练参数 **279M**（111M LoRA + 151M action head + 17M proprio projector，p.17），其中 action head 是大头。

## 5. 常量：不同本体的 chunk 和维度

```python
# prismatic/vla/constants.py:26-44
LIBERO_CONSTANTS = {"NUM_ACTIONS_CHUNK": 8, "ACTION_DIM": 7, "PROPRIO_DIM": 8, ...}
ALOHA_CONSTANTS = {"NUM_ACTIONS_CHUNK": 25, "ACTION_DIM": 14, "PROPRIO_DIM": 14, ...}
BRIDGE_CONSTANTS = {"NUM_ACTIONS_CHUNK": 5, "ACTION_DIM": 7, "PROPRIO_DIM": 7, ...}
```

这些常量是**按启动脚本自动推断**的（看 `sys.argv` 里有没有 `libero`/`aloha`/`bridge`），推断不出来就默认 LIBERO。这是一个工程上的 hack，读代码时要知道。

## 6. 微调配方

OpenVLA 的微调有两种：

| 方式 | 可训练参数 | 硬件 | 时间 |
|---|---|---|---|
| 全参 | 7B | 8×A100 | 每任务 5–15 h |
| LoRA r=32 | 97.6M（1.4%） | 单卡 A100 | 10–15 h |

LoRA 的效果（Table 1, p.10）：全参 69.7%，LoRA **68.2%**，只掉 1.5 个点，算力降 8×。这是 OpenVLA 能被社区广泛微调的原因。

OFT 更进一步：**只训 279M**（LIBERO）或 **853M**（ALOHA OFT+），基座冻结。LIBERO 用 8×A100/H100，batch 64，50–150K 步，lr 5e-4（p.17）。

## 7. OFT 的 26× 和 43× 到底是什么

这是最容易误引的数字（Table II/III, p.6, p.10）：

| 场景 | 基线 | OFT | 倍数 |
|---|---|---|---|
| LIBERO，K=8，单图 224 | OpenVLA 4.2 Hz / 0.2396 s | **108.8 Hz / 0.0735 s** | 25.9× |
| ALOHA，K=25，三图 + 14 维 state | OpenVLA 1.8 Hz / 0.543 s | **77.9 Hz / 0.321 s** | 43.3× |

两个都是**推理吞吐**，不是训练加速；基准不同（K=8 vs K=25，单图 vs 三图），不能互换。

还有一个单位笔误：OFT 正文写 "much lower latency (**0.07 ms** … **0.321 ms** …)"（p.2），但 Table II 是 **0.0735 s**、Table III 是 **0.321 s**。是秒，不是毫秒。

## 8. OFT 的精度代价

OFT 的 97.1% 是 LIBERO 上的数字，配置是「每 suite 一个策略 + 过滤后数据 + 加 wrist/proprio」（p.6）。换配置会掉：

- 纯第三人称、未过滤数据：**94.5%**；
- 单策略跑 4 suite：**96.8%**；
- 从零微调（不用 OpenVLA 预训练）：**91.9%**，掉 5.2 点。

所以「97.1%」不是一个可以随便复现的数字，它依赖数据过滤和多视角输入。

## 9. 两条路线的工程分叉

| 维度 | OpenVLA | OFT |
|---|---|---|
| 动作表示 | 离散 256 bin | 连续 7 维 |
| 解码 | 自回归 7 token | 并行 1 次前向 |
| 动作头 | 复用 lm_head | MLPResNet + L1 |
| chunk | 1 | 8 / 25 |
| 训练 loss | CE on action tokens | L1 |
| 推理 | 0.33 s/步 | 0.0735 s/chunk（K=8） |
| 适合 | 复用 VLM 词表、研究离散表示 | 部署、低延迟、chunk 控制 |

OFT 论文自己承认 L1 的局限：它通过 "encouraging the policy to learn the median mode" 平滑噪声，但面对真正的多峰分布 "may struggle to accurately model truly multimodal action distributions"（p.10）。所以两条路线不是替代关系，是场景分工。

## 10. 读这份代码的四个检查点

1. **`action_token_begin_idx`**：是不是 `vocab_size - (n_bins+1)`？如果是 `-n_bins`，会覆盖到 EOS。
2. **labels 的 mask**：prompt 位置是不是 `IGNORE_INDEX`？
3. **并行解码的 attention mask**：动作位置是不是双向？如果还是 causal，8 个动作会互相看不到，L1 head 的输入就是错的。
4. **反归一化**：用的是 q01/q99 还是 min/max？OFT 的 constants 里 LIBERO 是 `BOUNDS_Q99`，ALOHA 是 `BOUNDS`，别混。

## 小结

- OpenVLA 的「256 bins」是 256 条边界、255 个中心；解码时 clip 到 255 个中心。
- loss 只算动作 token（`labels[:-(len+1)] = -100`），prompt 不参与。
- 自回归 7 token → 0.33 s/步；OFT 并行解码 + L1 head → 0.0735 s/chunk（K=8），26×。
- OFT 的加速是推理吞吐，不是训练；26× 和 43× 的基准不同。
- OFT 的 97.1% 依赖数据过滤 + wrist/proprio；纯第三人称是 94.5%。
- L1 在多峰任务上会学中位数，OFT 论文自己承认。

下一篇讲评测：[LIBERO 的 AUC 不是成功率，以及 10 次试验能说明什么]({{< relref "vla-11-evaluation" >}})。
