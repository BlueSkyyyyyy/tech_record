---
title: "VLA 精读（六）GR00T N1/N1.7：双系统与「每个本体一套 MLP」"
date: 2026-11-01T20:00:00+08:00
draft: false
weight: 7
series: ["vla"]
tags: ["VLA", "系列", "具身智能", "GR00T", "NVIDIA", "双系统", "跨本体"]
categories: ["LLM算法"]
---

π0/π0.5 的跨本体策略是「把 state 变成字符串，让 VLM 自己理解维度」。GR00T 走的是相反的路：**共享 VLM 和动作主干，但给每个本体一套独立的输入/输出 MLP**。

两条路线的分歧点很清楚：

- π0.5 假设「语义可以统一维度」——128 维的 state 和一串数字没有本质区别；
- GR00T 假设「本体差异是结构性的」——7 维单臂和 132 维人形的动作语义不能靠 pad 对齐，必须显式建模。

这一篇拆 GR00T N1 的双系统、本体条件化机制、数据金字塔，以及 N1.7 改了什么。

## 1. 双系统：10 Hz 的 VLM + 120 Hz 的 DiT

GR00T N1 的架构是「双系统」（p.2）：

```
图像 + 语言 ──► System 2: Eagle-2 VLM ──► VL token 特征
                                              │ cross-attention
机器人状态 q_t ──► State Encoder (每本体一个 MLP) ──┤
噪声动作 A_t^τ ──► Action Encoder (每本体一个 MLP, 含 τ) ──► DiT blocks × N ──► Action Decoder (每本体一个 MLP) ──► 动作 chunk
```

两个模块都是 Transformer，端到端联合训练：

- **System 2**：Eagle-2 VLM（SmolLM2 + SigLIP-2），**1.34B** 参数，取第 **12 层**的隐状态（middle-layer），标称 **10 Hz**；
- **System 1**：DiT（Diffusion Transformer），交替 cross-attention / self-attention，adaLN 注入去噪步，标称 **120 Hz**；
- 总参数 **2.2B**（p.3）；System 1 的参数量论文没单列，2.2B − 1.34B ≈ 0.86B 是推算。

三个容易误读的点：

1. **「双系统」不是两个独立模型**。它们端到端联合训练，System 1 通过 cross-attention 读 System 2 的特征，不是「VLM 输出文本，另一个模型执行」。
2. **10 Hz / 120 Hz 是设计目标，不是实测推理频率**。论文给的实测是 **63.9 ms / 16 动作 chunk**（L40、bf16），换算约 **15.6 Hz** 的 chunk 生成频率（p.3）。
3. **用第 12 层而不是最后一层**。论文说中间层「更快且成功率更高」（p.4）。这是「VLM 最后一层不适合动作」的一个可引用证据——最后一层已经过度特化到语言生成。

动作头是 **DiT + flow matching**，不是 Diffusion Policy 的 U-Net：

- 动作 chunk **H = 16**；
- 去噪步数 **K = 4**（"K = 4 inference steps to work well across all embodiments", p.5）；
- 时间采样沿用 π0：$\tau \sim \mathrm{Beta}((s-\tau)/s;1.5,1)$，$s=0.999$；
- 插值 $A^\tau = \tau A + (1-\tau)\varepsilon$，速度目标 $\varepsilon - A$；
- 前向 Euler：$A^{\tau+1/K} = A^\tau + (1/K)V_\theta$。

注意速度目标的符号：GR00T 是 $\varepsilon - A$，和 π0 的 $A-\varepsilon$ 相反，和 SmolVLA 一致。这不是抄错，是「噪声在 $\tau=0$ 还是 $\tau=1$」的约定差异。

## 2. 本体条件化：不是「每个 tag 一套网络」

GR00T 的跨本体机制，论文里的描述是（p.3–5）：

- **State Encoder**：每个本体一个 MLP，把不同维度的 state 投到共享 embedding；
- **Action Encoder**：每个本体一个 MLP，把**加噪动作和时间步**一起编码；
- **Action Decoder**：DiT 最后一个 block 之后，每个本体一个 MLP，输出 H 个动作 token；
- **VLM 和 DiT 主干完全共享**。

代码里这套机制叫 `CategorySpecificLinear`：

```python
# Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py:59-90
class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)
```

一个 `nn.Parameter` 里存 `[num_categories, in, out]` 的全部权重，forward 时按 `cat_ids` gather。所以「per-embodiment MLP」在实现上是**一个张量的切片**，不是 N 个独立模块。

几个工程细节：

- **tag → id**：`EmbodimentTag` 枚举（如 `oxe_droid_relative_eef_relative_joint`、`libero_sim`、`new_embodiment`）映射到整数 id，再查 `embodiment_id.json`。
- **槽位数 32**：`max_num_embodiments = 32`（`gr00t_n1d7.py:133`）。
- **相近变体共享槽位**：比如 R1 Pro Sharpa 的 4 个数据源变体共用一个 index。所以 **embodiment tag ≠ 每 tag 一套网络**，而是「每类动作空间一套投影，相近的合并」。
- **「新本体」可以凭空造**：人类视频和神经轨迹没有动作标签，用 VQ-VAE 提 latent action（LAPA），并 "treat it as a distinct 'LAPA' embodiment"（p.5）；神经轨迹上的 IDM 伪动作也被当作另一个本体。**embodiment 的边界是「数据来源 + 动作空间」，不是「机器人型号」。**

数据层还做了「尽力对齐」（p.24）：

- state 的末端旋转用 **6D rotation**，action 的旋转用 **axis-angle**；
- 关节/末端位置做 **min-max 归一化**；
- 向量顺序统一为「末端旋转 → 末端位置 → 夹爪，先左臂后右臂」；
- 每个数据集附 `modality.json`，逐维标注语义和旋转表示。

这一层是跨本体能训起来的前提。没有它，`CategorySpecificLinear` 只是在拟合一堆语义不一致的维度。

## 3. 数据金字塔：8,375.7 小时

GR00T N1 的数据分三层（p.2, Fig.1）：底层 web 数据 + 人类视频，中层合成数据（仿真 + 神经视频），顶层真机数据。越往上数量越少、本体相关性越强。

Table 7（p.28）是唯一给全量的表：

| 层 | 小时 | 帧 |
|---|---|---|
| 真机（GR-1 Teleop 等） | **3,288.8** | — |
| 人类视频 | **2,517.0** | — |
| 仿真（含 DexMimicGen 540k demos） | **1,742.6** | — |
| 神经轨迹 | **827.3** | — |
| **合计** | **8,375.7** | **592.9M** |

几个口径陷阱：

- 论文 p.7 还有一句「780k 仿真 trajectories = 6,500 小时」，包含后训练数据，和 Table 7 的 1,742.6 小时**对不上**。引用总量时用 Table 7。
- episode 数大多没给；只有 AgiBot **140k trajectories**（p.9）和 DexMimicGen **540k demos**（p.10）有明确数字。
- 「8,375.7 小时」是**多本体、多来源**的总和，不是「真机 8,000 小时」。真机只有 3,288.8 小时。

## 4. 结果：45.0% 仿真、76.8% 真机

GR00T N1 的评测分仿真和真机（p.15）：

| 设置 | GR00T N1 | Diffusion Policy |
|---|---|---|
| 仿真平均（100 demos/任务） | **45.0%** | 33.4% |
| 真机 GR-1（13 任务） | **76.8%** | 46.4% |

读这两个数字要注意：

- 仿真 45.0% 是**每 benchmark 单独后训练 100 demo** 得到的，不是零样本。零样本是另一组数（76.6% / 73.3%，只有 15 次试验）。
- 真机 13 个任务，**每任务 10 trials**，某些任务（Pack Machinery）只跑 5 trials，且是**部分计分**（抓住但没放进容器算 0.5）。
- 仿真取「最后 5 个 checkpoint 的最大值」，存在 checkpoint 选择偏差。

所以 76.8% 不是一个可以直接和 LIBERO 上的 97% 比较的数字。评测协议、试验数、计分方式都不同。

## 5. N1.7：3B、Cosmos-Reason2、horizon 40

N1.7 是 2026-04 的更新（GitHub tag `n1.7-release`），论文是后来的技术报告（arXiv 2609.18108）。关键变化：

| 项 | N1 | N1.7 |
|---|---|---|
| 总参数 | 2.2B | **3B** |
| VLM | Eagle-2（SmolLM2 + SigLIP-2） | **Cosmos-Reason2-2B / Qwen3-VL** |
| 动作 horizon | 16 | **40**（LeRobot 默认；LIBERO checkpoint 执行 8） |
| 最大维度 | 未单列 | **132** |
| DiT | 未披露层数/宽度 | 16 层、32 heads、head_dim 48、output 1024 |
| 延迟 | 63.9 ms / 16 动作 | ~70 ms 前向 → **~14 Hz**（A800） |

N1.7 的 DiT 配置在代码里是明确的：

```python
# gr00t/configs/model/gr00t_n1d7.py:104-112
"num_layers": 16,
"num_attention_heads": 32,
"attention_head_dim": 48,
"output_dim": 1024,
```

但**不能把这组数回填到 N1**。N1 论文的 Table 6 只有优化器超参（p.27），没有 DiT 层数/宽度。博客里写「N1 的 DiT 是 16 层」是把 N1.7 的数倒推回去，属于事实错误。

N1.7 还有一个值得记的取舍：**一步 drifting head**。把 4 步去噪换成一步生成，单次前向从 **45.3 ms 降到 5.0 ms**，但 LIBERO-Spatial 从 **97.6% 掉到 64.0%**（2609.18108）。9 倍加速换 33 个点精度——在需要精细空间推理的任务上不划算。这解释了为什么 2026 年的主流仍是 4–10 步。

## 6. 和 π0.5 的路线对比

| 维度 | π0.5 | GR00T N1/N1.7 |
|---|---|---|
| 跨本体机制 | state 离散成字符串，VLM 统一读 | 每本体一套 state/action MLP，主干共享 |
| 动作 expert | 300M，所有本体共享 | DiT ~0.86B（N1，2.2B−1.34B 推算），所有本体共享 |
| 时间注入 | adaRMS（每层） | adaLN（DiT 每层） |
| 动作表示 | flow matching，H=50 | flow matching，H=16（N1）/40（N1.7） |
| 去噪步数 | 10 | 4 |
| 控制频率 | 50 Hz 机器人 | System 2 10 Hz / System 1 120 Hz（设计） |
| 数据规模 | 10k 小时自采 + web | 8,375.7 小时多源 |
| 强项 | 开放世界语义、长程任务 | 多本体、人形、仿真到真机 |

两条路线没有绝对优劣：

- 本体重、维度差异大（人形 132 维）→ GR00T 的 per-embodiment MLP 更自然；
- 本体轻、语义差异大（不同家庭、不同物体）→ π0.5 的字符串 state + web 数据更自然。

2026 年出现了一个折中：**canonical action space**（LingBot-VLA 2.0 的 55 维统一空间），把所有本体映射到同一组语义槽位，既不用字符串也不用 per-embodiment MLP。这是下一篇「数据与跨本体」要展开的。

## 小结

- GR00T N1 = Eagle-2 VLM 1.34B + DiT ~0.86B（推算），总 2.2B；H=16、K=4、63.9 ms/chunk。
- 双系统是端到端联合训练，不是两个独立模型；10 Hz/120 Hz 是设计目标，实测约 15.6 Hz chunk。
- 跨本体靠 per-embodiment MLP（`CategorySpecificLinear`），32 个槽位，相近变体共享；embodiment 的边界是「数据来源 + 动作空间」。
- 数据总量 8,375.7 小时，其中真机只有 3,288.8 小时；780k sim/6,500 小时与 Table 7 冲突。
- N1.7 是 3B、Cosmos-Reason2、H=40、132 维；DiT 16 层/32 heads 是 N1.7 的数，不能回填 N1。
- 一步 drifting head 快 9 倍，但 Spatial 掉 33 个点，说明 4–10 步仍是必要的。

下一篇离开模型，进入数据：[OXE 的 1M 轨迹为什么不够，以及归一化的三个坑]({{< relref "vla-07-data" >}})。
