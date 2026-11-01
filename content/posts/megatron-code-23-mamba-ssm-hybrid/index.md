---
title: "Megatron 源码精读（二十三）：Mamba/SSM 与 Hybrid 模型"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "mamba", "ssm", "状态空间模型", "hybrid"]
categories: ["训练框架"]
weight: 23
series: ["megatron-code"]
---

注意力不是唯一的序列混合器。Mamba 用一个固定大小的状态递推地「吸收」整条序列，那套数学在代码里到底长什么样？训练、prefill、decode 为什么必须分成三条路径？它又怎么和 attention 层混排成 Hybrid 模型？

一句话结论：**Mamba 层的数学只有四行递推，工程复杂度全在另外三件事上——五个投影打包成一个 GEMM、三条执行路径、CP 下的序列↔head 布局交换。**

前面 22 篇的主线是 attention 与并行基础设施，本篇换到另一类序列混合器，代码落在 `megatron/core/ssm/` 和 `megatron/core/models/hybrid/`。

---

## 1. 数学：四行递推，两个容易写错的地方

结论：**选择性 SSM 就是「离散化 → 状态更新 → 读出」，但状态是每 head 一个矩阵，读出的收缩方向常被写反。**

每个 head 维护一个状态矩阵 $h_t \in \mathbb{R}^{P \times N}$（$P$ 为 headdim，$N$ 为 d_state），$A$ 是每 head 一个负标量（在 $P \times N$ 上广播），$B_t, C_t \in \mathbb{R}^{N}$，$x_t \in \mathbb{R}^{P}$：

$$
\begin{aligned}
\Delta_t &= \mathrm{softplus}(\delta_t + b_\delta) \\
\bar{A}_t &= \exp(\Delta_t A) \\
h_t &= \bar{A}_t\, h_{t-1} + \Delta_t\, B_t x_t^{\top} \\
y_t &= h_t C_t + D \odot x_t
\end{aligned}
$$

两处易错：状态增量是**外积** $\Delta_t B_t x_t^\top$（不是逐元素乘）；读出是 $h_t C_t$（$C$ 在右边收缩 $N$ 维），不是 $C_t^\top h_t$。

这组公式逐行落在 `_ssm_decode` 的纯 PyTorch 回退分支里（`mamba_mixer.py:1241`）：

```python
dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # Δ_t
dA = torch.exp(dt * A)                                 # A_bar
x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)       # Δ·B·x^T
ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)  # h·C
y = y + rearrange(self.D.to(dtype), "h -> h 1") * x    # + D⊙x
```

`einsum` 的下标就是维度说明书：`b`=batch、`h`=head、`p`=headdim、`n`=d_state。这段代码只在 `selective_state_update` 不可用时才跑，而且只覆盖 `ngroups_local_tp == 1` 这一支；`ngroups > 1` 时同样的数学要先把 B/C/dt 展开到 `(h p)`，写法是 `"bd,bdn,bd->bdn"`。

## 2. 初始化：三个反直觉的点

**第一，$A$ 是每 head 一个、落在 $(-16,-1)$ 的负数，不是 $(-1,0)$ 之间的小数。** 代码先采 $a \sim U(1,16)$，再存 `A_log = log(a)`（`mamba_mixer.py:390`），真正参与计算的是

$$
A = -\exp(A_\mathrm{log}) = -a
$$

恒为负才是关键：$\bar{A}_t = e^{\Delta_t A}$ 必落在 $(0,1)$，状态指数衰减；$a$ 的量级只决定衰减快慢（$(1,16)$ 对应相当快的衰减）。

**第二，`dt_bias` 存的是 softplus 的逆。** 初始化先采 $\Delta \in [10^{-3}, 10^{-1}]$，再用 $\delta = \Delta + \log(-e^{-\Delta}+1)$ 反解成 bias（`:361`），于是训练起点处 `softplus(dt_bias)` 正好落在该区间，而不是从 0 附近慢慢爬。

**第三，`A_log`/`dt_bias`/`D` 的形状都是 `(nheads_local_tp,)`**，`partition_dim=0`——它们按 **head** 切，不按 hidden 切。这也解释了为什么 decode 缓存可以预展开成 `(nheads, headdim, d_state)` 的 stride-0 视图。

## 3. 五个投影压进一个 `in_proj`

结论：**`z/x/B/C/dt` 五块拼成一次 column-parallel GEMM，宽度不同是为了让 CPU 侧只付一次 launch 和一次通信。**

```python
self.in_proj = build_module(..., self.d_model,
    self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads,  # z x B C dt
    ...)
```

| 块 | 作用 | 宽度 |
|---|---|---|
| `z` | 门控输入 | `d_inner` |
| `x` | SSM 输入 | `d_inner` |
| `B` | 输入权重 | `ngroups * d_state` |
| `C` | 输出权重 | `ngroups * d_state` |
| `dt` | 步长原始值 | `nheads` |

顺序就是 `z, x, B, C, dt`，`_ssm_decode` 里的 `torch.split` 用的也是这个顺序。因为五块宽度不同，TP 切分后不能做连续 concat，代码把每块的本地宽度记在 `in_proj.weight.partition_sizes` 上，供跨 TP 尺寸 reshard 用；FP8 下还额外要求总宽是 16 的倍数。

`conv1d` 只作用在 `x B C` 三块上——`z` 和 `dt` 不过卷积。卷积参数宽度用**本地**（已 TP 切过）的值：`d_inner_local_tp + 2*ngroups_local_tp*d_state`。

这里有个坑：`d_state`、`headdim`、`ngroups` 都不在构造函数签名里，而是从 `TransformerConfig` 取（`mamba_state_dim=128`、`mamba_head_dim=64`、`mamba_num_groups=8`）。`expand=2` 只决定 `d_inner`，一旦显式设了 `mamba_num_heads`，`d_inner` 会被改写成 `nheads * headdim`，`expand` 就失效了。

## 4. 一条 forward，三条路径

`forward` 的骨架很薄，分支全在三个 `_ssm_*` 里（`mamba_mixer.py:463`）：

```python
zxBCdt, _ = self.in_proj(hidden_states)
zxBCdt = self.cp.pre_conv_ssm(zxBCdt, packed_seq_params)   # CP 的 all-to-all
if in_inference_mode or not self.use_mem_eff_path:
    y = self._ssm_prefill(zxBCdt, conv_state=..., ssm_state=...)
else:
    y = self._ssm_training(zxBCdt, packed_seq_params)
out, out_bias = self.out_proj(y)
```

- **训练**走 `mamba_split_conv1d_scan_combined`：卷积、扫描、门控融合进一个 kernel，大幅缩小 forward 要为反向保存的激活——这就是 `use_mem_eff_path` 的含义。packed 序列通过 `seq_idx` 传入。
- **prefill** 分两支：动态 batching（varlen）走 `mamba_chunk_scan_combined_varlen` 并用 `cu_seqlens` 定位，静态 batching 走 `mamba_chunk_scan_combined` 且 `return_final_states=ssm_state is not None` 决定是否回写 cache。
- **decode** 走 `selective_state_update` + `causal_conv1d_update`，只更新状态不落中间结果；它支持 `1 + num_speculative_tokens` 个 token，而纯 PyTorch 回退只支持 1 个。

decode 阶段直接禁掉 CP：`assert self.cp.cp_size == 1`（`:715`）。单 token 摊不平一次 all-to-all，收益是负的。

prefill 里有个必须讲清的顺序问题：动态 batching 下要**先读出旧的 conv state，再写回新算出的 conv state**（`:864` 附近的长注释）。反过来做，被恢复的请求会看到自己刚算出的状态而不是上一轮的缓存，卷积输出直接错。

还有一个性能细节：decode 每步都用 `-exp(A_log.float())`，会额外 launch 三个小 elementwise kernel（cast、exp、neg），于是被缓存成 `_A_neg_exp_cache`（`:1091`），并靠 stride-0 的 `expand` 视图走 kernel 的 TIE_HDIM 快速路径；切回 `train()` 时置脏。

## 5. 并行：TP 切 head，CP 是 Ulysses 式换轴

TP 侧的约束链很硬（`mamba_mixer.py:246` 起）：`nheads % tp == 0`、`ngroups % tp == 0`、`nheads % ngroups == 0`。`in_proj` 列并行、`out_proj` 行并行，输入按 `d_inner` 切、输出按序列维切——代码注释直接写明这是按 sequence parallel 的约定假设的。

CP 侧的关键认知是：**Mamba 的 CP 也切 head**。`nheads_local_tpcp = nheads_local_tp / cp_size`、`d_inner_local_tpcp = d_inner_local_tp / cp_size`，然后靠 all-to-all 把「序列分片」换成「head 分片」——这正是 Ulysses 的思路。

`MambaContextParallel`（`mamba_context_parallel.py:30`）自己没有可训参数，注释说得很直白：同一 TP rank 下的所有 CP rank **存同一份参数，但只用各自的那一片切片**。切片在 forward 里做，梯度照样能反传回 `cp_size=1` 的原始参数。

`pre_conv_ssm` 对 `z/x/B/C/dt` 各做一次 `_all_to_all_cp2hp`，把 `[S/cp, b, h]` 变成 `[S, b, h/cp]`；`post_conv_ssm` 再做一次反向。五块分五次调用，代码里还留着 TODO 问能不能合并——这是真实的待优化点。

B/C 的复制条件值得记准（`mamba_context_parallel.py:122` 起）：

- `cp_size > ngroups_local_tp`：要求 `cp_size % ngroups_local_tp == 0`，`group_repeat_count = cp_size // ngroups_local_tp`，`ngroups_local_tpcp = 1`——每个 CP rank 只拿一组，组状态被复制；
- 否则：要求 `ngroups_local_tp % cp_size == 0`，`group_repeat_count = 1`，`ngroups_local_tpcp = ngroups_local_tp // cp_size`——一个组的全部 head 都在同一 rank 上。

复制的动机在注释里：把同一组状态连续复制，可以让「状态的副本」和「对应的 head」落在相邻的 CP rank 上。

最后一块拼图是 `_undo_attention_load_balancing` / `_redo_attention_load_balancing`：attention 层看到的序列是 zigzag 交错顺序（docstring 的例子：cp_size=3 时是 `162534`），而卷积和递推必须按物理顺序 `123456` 走。所以进 SSM 前 undo、出 SSM 后 redo。这是 hybrid 模型里最容易忽略的一处耦合。

## 6. Hybrid：一串字符决定排列

结论：**Hybrid 在 MCore 里被解耦成两层——字符串决定「怎么排」，module spec 决定「每层长什么样」。**

`Symbols`（`hybrid_layer_allocation.py:14`）用单字符表示层类型：

```python
MAMBA = "M"; GDN = "G"; KDA = "K"; ATTENTION = "*"
DS_ATTENTION = "D"; MLA = "+"; CSA = "C"; HCA = "H"; WINDOW = "W"
MLP = "-"; MOE = "E"; PIPE = "|"; MTP_SEPARATOR = "/"
```

所以 `"M*M*"` 就是 Mamba、Attention、Mamba、Attention 四层。`|` 切 pipeline 分段，`/` 切 MTP 分段。

三个函数的职责不要混：

- `pattern_from_ratios`（`:80`）把废弃的比例参数转成 pattern，做法是均匀插桩（`section_len = mamba_count / (attention_count + 1)`，累计值 `x < 0.5` 时翻成 attention），保证首尾都是 Mamba；
- `parse_hybrid_pattern`（`:206`）把 pattern 拆成 `main_pattern` + `mtp_pattern`，并校验各 MTP 段一致；
- `select_pipeline_segment`（`:342`）按 `|` 取当前 `pp_rank` / `vp_stage` 的层列表和层偏移。

落到实现上，`hybrid_stack_spec` 是个 ModuleSpec 常量（`hybrid_layer_specs.py:100`，注意不是函数）：`mamba_layer` 指向 `MambaLayer` + `MambaMixer(in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear)`，attention 层指向 `SelfAttention` 或其变体。`HybridStack`（`hybrid_block.py:653`）按 pattern 逐层 build。

`MambaLayer` 本身和 `TransformerLayer` 高度对称：norm → mixer → bias-dropout-add 残差。值得一提的是它的 `forward` 里 `attention_mask` 和 `rotary_pos_emb` 都标着 "Not used"——因果性由递推顺序天然保证，位置信息由卷积和状态顺序捕捉，所以它不需要这两样东西。这也正是它能和 attention 层在 hybrid 里无缝互换的原因。

## 7. 小结

- **数学四行**：$\Delta$ → $\bar{A}$ → 状态更新（外积）→ 读出 $h_t C_t$；decode 回退分支（`mamba_mixer.py:1241`）是最好的注释。
- **$A = -\exp(A_\mathrm{log})$ 恒负**（$A_\mathrm{log}=\log a,\,a\sim U(1,16)$），保证 $\bar A \in (0,1)$。
- **五块打包**：`z/x/B/C/dt` 一次 GEMM，conv1d 只管 `x/B/C`，`partition_sizes` 供 reshard。
- **三路径**：训练融合 kernel、prefill 分 varlen/static、decode 用 `selective_state_update` 且禁 CP。
- **CP 换轴**：`pre/post_conv_ssm` 两次 all-to-all 把序列分片换成 head 分片，B/C 按 `group_repeat_count` 复制，进出还要 undo/redo attention 的 zigzag 顺序。

---

**版本提示（2026-10）**：上游新增 `ssm/ops/gdp/`（Gated Delta Product 的 chunk 描述与 decode/prefill prepare，对应 `gated_delta_product.py` 的 `GatedDeltaProductMixer`），`linear_cp_mode` 未显式设置时 `gdn2`（GDP）走 headwise、`gdn`/`kda` 走 chunkwise；`ops/mamba2/batch_invariant_decode.py` 提供 batch-invariant 的 Mamba decode（#8221）；`wide_residual_mamba_layer.py` 让 Mamba 分支接上共享 residual 流（#8344）。`MambaMixer` 的数学没变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
