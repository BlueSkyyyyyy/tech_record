---
title: "Megatron 源码精读（二十三）：Mamba/SSM 与 Hybrid 模型"
date: 2026-09-02
draft: false
tags: ["megatron-lm", "系列", "训练框架", "mamba", "ssm", "状态空间模型", "hybrid"]
categories: ["训练框架"]
weight: 23
series: ["megatron-code"]
---

前 22 篇把注意力（attention）这条主线的并行、通信、checkpoint 都走完了。但 Megatron-LM（MCore）里还有一整套**完全不同的序列混合器**——状态空间模型（State Space Model, SSM），代表作是 Mamba（M2/Mamba-2 的 selective scan）。更关键的是它还能和注意力层**混排**成 Hybrid 模型（如 NVIDIA 的 Nemotron-H、DeepSeek-V4 这类 attention + SSM 交替的架构）。

本篇精读 `megatron/core/ssm/`（约 4600 行）加 `megatron/core/models/hybrid/`（约 2900 行），讲清三件事：

1. **数学层**：选择性状态空间（S6 / Mamba-2）的递推到底长什么样，代码里那几行 `einsum` 对应教科书里的哪一步。
2. **实现层**：`MambaMixer` 如何把 `z/x/B/C/dt` 五个投影压进一个 `in_proj`，训练/prefill/decode 三条路径为何要分开，TP/CP 怎么切（尤其 CP 的 all-to-all 与「组状态复制」）。
3. **编排层**：Hybrid 模型怎么用一串字符串 `"M*M*"` 描述「Mamba 层和 Attention 层的排列」，再落到 layer spec。

行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`。

---

## 1. 背景：Mamba 想解决什么

标准自注意力是 $O(L^2)$ 的显存/算力（显存随序列平方增长），长上下文下撑不住。状态空间模型走另一条路：**用一个固定大小的隐状态递归地「吸收」整条序列**，每个 token 只做 $O(1)$ 的状态更新，复杂度 $O(L)$ 且显存随序列**线性**增长。

经典 SSM（S4 族）的递推是**时不变**的——状态转移矩阵、输入/输出投影都是常数，表达能力受限。Mamba（S6）把它变成**时变/选择性**的：这三组参数变成输入的**函数**。这正是 Megatron 里 `MambaMixer` 的核心，也是「选择性状态空间（selective scan）」名字的由来。

后续又出了 Gated DeltaNet（GDN）、Kimi DeltaAttention（KDA）等变体，MCore 把它们归到 `ssm/gated_delta_net/` 下，本篇只点到为止（它们共享同一套 layer/hybrid 编排骨架）。

---

## 2. 数学核心：选择性 SSM 的递推（S6）

Mamba-2 的 selective scan 等价于一个线性注意力 / chunked 的矩阵形式，但理解代码最直接的是**离散化的状态递推**。给定输入 $x_t \in \mathbb{R}^d$，状态 $h_t \in \mathbb{R}^N$，输出 $y_t$：

$$
\Delta_t = \mathrm{softplus}(\delta_t + b_\delta),\qquad
\overline{A}_t = e^{\Delta_t A},
$$

$$
\overline{B}_t = \Delta_t B_t,\qquad
h_t = \overline{A}_t \odot h_{t-1} + \overline{B}_t \odot x_t,
$$

$$
y_t = C_t^\top h_t + D \odot x_t .
$$

其中 $A$ 是**每个 head 一个**的标量向量（Mamba-2 把对角化状态降到标量级），$B_t, C_t, \delta_t$ 都由输入线性投影而来（「选择性」就体现在这里），后面的 $\odot x_t$ 是 $D$ 残差跳接（skip connection），最后还乘一个门控 $z_t$（来自 RMSNorm gating）。

这组公式**逐字**落在 `MambaMixer._ssm_decode` 的「纯 PyTorch 回退分支」里，这是整个文件里最容易读懂的 SSM 实现（`mamba_mixer.py:1239-1250`）：

```python
# Discretize A and B (b (g n))
dt = F.softplus(dt + self.dt_bias.to(dtype=dt.dtype))  # (batch, nheads)
dA = torch.exp(dt * A)                                  # \bar{A} = e^{Δ A}
x = rearrange(x, "b (h p) -> b h p", p=self.headdim)
dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)        # \bar{B} ⊙ x
ssm_state.copy_(ssm_state * rearrange(dA, "b h -> b h 1 1") + dBx)
y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)  # C^T h
y = y + rearrange(self.D.to(dtype), "h -> h 1") * x      # + D ⊙ x
```

逐行对应：`dt = softplus(...)` 是 $\Delta_t$；`dA = exp(dt*A)` 是 $\overline{A}_t$（`A` 存的是 `A_log`，即 $A=\log(a)$，保证 $a\in(0,1)$）；`dBx = dt·B·x` 用 `einsum` 把 $B$ 和 $x$ 合并进状态增量；`ssm_state.copy_(...)` 是状态更新 $h_t$；最后两行读 $C_t^\top h_t$ 加残差 $D x$。真实训练/推理走的是下面的 Triton 融合 kernel（`selective_state_update` / `mamba_chunk_scan_combined`），但数学完全一致——看懂这几行就够了。

> `A` 用 `A_log` 存而不是直接存 $A$：`A_log` 是 `torch.log(A)`（`mamba_mixer.py:391`），且 `A` 初始化为 `uniform_(1, 16)` 后再取 log。这样 $-\exp(A_\mathrm{log})$ 恒在 $(-1, 0)$ 之间，保证离散化后状态指数衰减、数值稳定。实际用的时候取 `A = -torch.exp(self.cp.get_A_log().float())`（`mamba_mixer.py:745`）。

---

## 3. 五个投影压进一个 `in_proj`

MambaMixer 的 `__init__`（`mamba_mixer.py:161`）里最反直觉的是：`in_proj` 一个 `ColumnParallelLinear` 同时产出 5 块东西——`z`, `x`, `B`, `C`, `dt`（`mamba_mixer.py:268-294`）：

```python
self.in_proj = build_module(
    submodules.in_proj,
    self.d_model,
    self.d_inner * 2 + 2 * self.ngroups * self.d_state + self.nheads,  # z x B C dt
    ...
)

in_proj_partition_sizes = [
    self.d_inner_local_tp,   # z
    self.d_inner_local_tp,   # x
    self.ngroups_local_tp * self.d_state,  # B
    self.ngroups_local_tp * self.d_state,  # C
    self.nheads_local_tp,    # dt
]
```

五个输出块的宽度分别是：

| 块 | 含义 | 宽度 |
|---|---|---|
| `z` | 门控（RMSNorm gating 的输入） | `d_inner = expand·d_model` |
| `x` | SSM 的输入 | `d_inner` |
| `B` | 输入投影（时间戳上的输入权重） | `ngroups · d_state` |
| `C` | 输出投影 | `ngroups · d_state` |
| `dt` | 步长 $\Delta$ 的原始值 | `nheads` |

它们沿列方向**拼在一起**做一个大的 column-parallel GEMM，省掉多个 GEMM 的 launch 和通信开销。因为五块的宽度不同，TP 切分时不能做连续 concat，`partition_sizes` 属性记录了每个 rank 分到的各块宽度，供跨 TP 尺寸的 checkpoint reshard 用（`mamba_mixer.py:294`）。

`conv1d` 同理把 `x B C` 三块拼成一个 `conv_dim = d_inner + 2·ngroups·d_state` 的卷积参数（`mamba_mixer.py:306-335`），`conv_partition_sizes` 记录分块。

`d_inner = expand·d_model` 是 SSM 的「扩张系数」（Mamba 默认 `expand=2`），`d_state` 是状态维度（默认 128/16 级），`ngroups` 是「组数」——B/C 投影的通道在做组共享：`nheads` 个 head 共享 `ngroups` 组 $B/C$ 参数（`group_size = d_inner/ngroups`）。

---

## 4. 一条 forward，三条路径

`MambaMixer.forward`（`mamba_mixer.py:463-519`）的骨架很薄——真正的分支在 `_ssm_training` / `_ssm_prefill` / `_ssm_decode` 三条实现在里面：

```python
zxBCdt, _ = self.in_proj(hidden_states)                 # (1) 一次 GEMM 出 z x B C dt
zxBCdt = self.cp.pre_conv_ssm(zxBCdt, packed_seq_params)  # (2) CP 的 all-to-all
if in_inference_mode or not self.use_mem_eff_path:
    y = self._ssm_prefill(zxBCdt, conv_state=..., ssm_state=...)
else:
    y = self._ssm_training(zxBCdt, packed_seq_params)    # (3) 核心 SSM
out, out_bias = self.out_proj(y)                          # (4) row-parallel 输出投影
```

三条路径的区别是**kernel 选择**和**状态传不传**：

- **训练 `_ssm_training`**（`mamba_mixer.py:730-784`）：调 `mamba_split_conv1d_scan_combined`，这是**显存高效版**——forward 时不存完整激活，反向时重算，缓解长序列训练的显存压力（这正是 `use_mem_eff_path` 的含义）。
- **prefill `_ssm_prefill`**（`mamba_mixer.py:786-1089`）：分「动态 batching」（varlen，带 `cu_seqlens`/`batch_indices`，走 `mamba_chunk_scan_combined_varlen`）和「静态 batching」（走 `mamba_chunk_scan_combined`）。动态分支里有一大段注释专门强调**先读旧 conv state、再更新**——顺序错了恢复的请求会看到「自己刚算出来的新状态」，污染因果性（`mamba_mixer.py:859-876`）。prefill 还需要把 `ssm_state` 的最终值写回 cache（`return_final_states=True`）。
- **decode `_ssm_decode`**（`mamba_mixer.py:1117-1285`）：单 token 或含 speculative tokens，走 `selective_state_update` 或 `causal_conv1d_update`，只更新状态不上存储中间结果。注意 decode 特判 `assert self.cp.cp_size == 1`——**decode 阶段不支持 context parallel**（`mamba_mixer.py:715`），因为每个 token 都要读完整状态，all-to-all 反而不划算。

三步（训练/prefill/decode）共享同一套数学，但因「要不要存中间激活」「序列是不是变长」「状态从哪读、写回哪」而分流。这也解释了这个文件高达 1434 行的原因——一半在伺候这三条路径的参数铺排。

一个精巧细节：decode 里反复用到的 `-exp(A_log.float())` 被缓存成 `_A_neg_exp_cache`（`mamba_mixer.py:401-402`、`1091-1107`），避免每个 token 都 launch 三个小 elementwise kernel（float cast、exp、neg），且在 `train(mode)` 切到训练时置 dirty（`mamba_mixer.py:1109-1115`），因为权重又变了。

---

## 5. 并行：TP 切 head，CP 做 all-to-all

### 5.1 TP 切分

和 attention 一样，SSM 的并行单位是 **head**。`__init__` 里的约束链（`mamba_mixer.py:246-261`）：

```python
assert self.nheads % tp_size == 0, "nheads must be evenly divisible by tp_size"
self.nheads_local_tp = self.nheads // tp_size
self.d_inner_local_tp = self.d_inner // tp_size   # d_inner = nheads·headdim
assert self.ngroups % tp_size == 0, ...
```

`in_proj` 是 column-parallel（输入 `d_model` 不变、输出按 head 切到每个 rank `d_inner_local_tp`），`out_proj` 是 row-parallel（输入按 head、输出 `d_model` 全量，`gather_output=False` 配合 sequence parallel）。中间 SSM 的 `A_log`（`1×nheads`）、`dt_bias`、`D` 都是 `partition_dim=0` 的 TP 参数。

### 5.2 CP：`MambaContextParallel`

`MambaContextParallel`（`mamba_context_parallel.py:30-311`）**不是** `MegatronModule`，没有任何自己的可训参数——它只是「在当前 CP rank 上，取 TP 参数的正确切片 + 做 all-to-all」。注释明确：同一 TP rank 内的所有 CP rank **共享同一份参数，但各用各的切片**（`mamba_mixer.py:442-446`）。

CP 的核心是 `pre_conv_ssm` / `post_conv_ssm` 这一对（`mamba_context_parallel.py:148-219`）：

- **pre**（卷积和 SSM 之前）：把 `[seq/cp, b, hidden]` 布局的 `z x B C dt` 各做一次 `_all_to_all_cp2hp`，变成 `[seq, b, hidden/cp]`（`mamba_context_parallel.py:170-201`）。这样每 rank 拿到「全序列、但只有自己的 head 切片」。
- **post**（SSM 之后）：`_all_to_all_hp2cp` 再切回 `[seq/cp, b, hidden]`。

`_all_to_all_cp2hp`（`mamba_context_parallel.py:313-337`）底层调 `all_to_all_sp2hp`（sequence-parallel→hidden-parallel 的 all-to-all），跟 TP 通信原语同源（第 21 篇讲过）。

最需要理解的是 **B/C 的「组状态复制」**（`mamba_context_parallel.py:179-198`）：

```python
B = repeat(B, "l b (g n) -> l b (g r n)", g=self.ngroups_local_tp,
           n=self.d_state, r=self.group_repeat_count)
```

当 `cp_size > ngroups_local_tp` 时（CP rank 数比组数多），一个组的 B/C 状态**必须复制到多个 CP rank**，因为这一组的 head 被切到多个 rank 上，每个 rank 回读状态时都需要完整的 B/C。`group_repeat_count = cp_size // ngroups_local_tp`（`mamba_context_parallel.py:127`）就是复制份数。对应的参数切片逻辑在 `_slice_conv_param` / `_slice_vector_param`（`mamba_context_parallel.py:263-310`）里——它们按 `cp_rank` 切出当前 rank 该用的那一份参数切片，**切片在 forward 里做**，这样梯度能反传回 cp_size=1 的原始参数。

还有个 `_undo_attention_load_balancing` / `_redo_attention_load_balancing`（`mamba_context_parallel.py:367-405`）——hybrid 模型里 attention 层和 SSM 层的 CP 负载均衡不同，进 SSM 前要「undo」attention 那头做的负载均衡、出 SSM 再「redo」回去。注释里还留了 TODO：考虑把负载均衡隔离到 attention 层（`mamba_context_parallel.py:204`）。

---

## 6. MambaLayer：套壳

`MambaLayer`（`mamba_layer.py:60-165`）结构上和 `TransformerLayer` 高度对称，就是「norm → mixer → bias-dropout-add 残差」：

```python
hidden_states = apply_module(self.norm)(hidden_states)          # 输入 RMSNorm
mixer_out_with_bias = self.mixer(hidden_states, ...)            # MambaMixer
hidden_states = self.mamba_bda(...)(mixer_out_with_bias, residual, self.hidden_dropout)
```

注意 `MambaLayer.forward` 签名里 `attention_mask`、`rotary_pos_emb` 都是「Not used」——SSM 不需要因果 mask（因果性天然由递推顺序保证）也不需要 RoPE（位置信息由状态顺序 + 卷积捕捉）。这让它和 `TransformerLayer` 在 hybrid 里能**无缝互换**（`mamba_layer.py:120-145`）。

它继承 `GraphableMegatronModule`，支持 CUDA graph（`mamba_layer.py:103-113`、`203-225`）：Mamba 的 decode 状态更新很轻，CUDA graph 收益大。`sharded_state_dict`（`mamba_mixer.py:1333-1415`）则处理分布式 checkpoint——`_split_tensor_factory` 把 `in_proj.weight` 按 `[z,x,B,C,dt]` 五块拆开存，还做了 `conv1d_weight → conv1d.weight` 的 key 映射以兼容旧 checkpoint（`mamba_mixer.py:1364-1366`），与第 22 篇的 sharding 元数据机制严丝合缝。

---

## 7. Hybrid：一串字符串编排 attention + SSM

Hybrid 模型 = 在一串层里**交替**塞 attention 层和 Mamba 层（有的还掺 MLP/MoE/MLA）。编排的入口是一个**字符串 pattern**，由 `megatron/core/models/hybrid/hybrid_layer_allocation.py` 解析。

### 7.1 `Symbols`：一个字符 = 一种层

`Symbols`（`hybrid_layer_allocation.py:14-44`）定义了层类型字母：

```python
MAMBA = "M"; GDN = 'G'; KDA = 'K'; ATTENTION = "*"
DS_ATTENTION = "D"; MLA = "+"; CSA = "C"; HCA = "H"; WINDOW = "W"
MLP = "-"; MOE = 'E'; PIPE = '|'; MTP_SEPARATOR = "/"
```

所以 `"M*M*"` = Mamba、Attention、Mamba、Attention 四层交替。`|` 是 pipeline 分段符，`/` 是 MTP（multi-token prediction）分段符。

### 7.2 从比例生成 pattern（兼容旧参数）

早期的 `hybrid_attention_ratio` / `hybrid_mlp_ratio` 两个比例参数已被废弃，`pattern_from_ratios`（`hybrid_layer_allocation.py:80-131`）把它们转成 pattern：

```python
attention_count = round(num_layers * attention_ratio)
mamba_count = num_layers - attention_count
sections = attention_count + 1
section_len = mamba_count / sections
```

核心思想是**均匀插桩**：attention 层均匀散布在 mamba 层之间，首尾是 mamba（`layer_types = [Symbols.MAMBA]*num_layers` 起手，再按 `x < 0.5` 的累计判定把某些位置翻成 attention）。这样保证「M M M * M M M *」这类开头结尾都是 Mamba 的均匀分布，而不是头重脚轻。

### 7.3 解析与 pipeline 分段

`parse_hybrid_pattern`（`hybrid_layer_allocation.py:206-280`）把 pattern 拆成 `main_pattern` + `mtp_pattern`，校验所有 MTP 段必须一致。`select_pipeline_segment`（`hybrid_layer_allocation.py:342-505`）则按 `|` 切 pipeline 分段，把当前 `pp_rank` / `vp_stage` 对应的那段 `layer_type_list` 和层偏移 `layer_offset` 算出来，供后面逐层 build。

### 7.4 落到 layer spec

字符串只是「排列」，真正决定「M 层长什么样」的是 `hybrid_layer_specs.py` 的 `hybrid_stack_spec`（`hybrid_layer_specs.py:117-...`）：`mamba_layer` 这一支是 `MambaLayer → MambaMixer(in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear)`；attention 层则指向 `SelfAttention`（或 DSA/MLA 等实验变体）；`HybridStack`（`hybrid_block.py:653`）负责把一个 pattern 段按顺序 build 成一叠层。

也就是说，「Hybrid」在 MCore 里被解耦成两层：**排列（字符串 pattern）+ 每层的实现（module spec）**。改 mix 只改字符串，改 mix 里某一层的实现只改 spec，互不干扰——这和第 17 篇讲的 layer spec 机制是同一个设计哲学的延伸。

---

## 8. 小结

- **数学**：Mamba-2 的选择性 SSM 是一个三行的离散递推（$\Delta$ → $\overline A/\overline B$ → 状态更新 → 输出），`_ssm_decode` 的 PyTorch 回退分支（`mamba_mixer.py:1239-1250`）是最佳注释。
- **投影打包**：`z/x/B/C/dt` 五块拼一个 `in_proj`，`x/B/C` 拼一个 `conv1d`，用 `partition_sizes` 记录分块宽度以便 TP reshard（`mamba_mixer.py:287-335`）。
- **三路径**：训练走显存高效的 `mamba_split_conv1d_scan_combined`，prefill 走 varlen/static 的 `mamba_chunk_scan_combined`，decode 走 `selective_state_update` 且**禁用 CP**（`mamba_mixer.py:715`）。
- **并行**：TP 切 head；CP 靠 `pre/post_conv_ssm` 的 all-to-all 做 seq↔head 布局转换，`B/C` 组状态在 `cp_size > ngroups` 时按 `group_repeat_count` 复制（`mamba_context_parallel.py:127`、`179-198`）。
- **Hybrid**：`Symbols` 字符串（`"M*M*"`）+ `pattern_from_ratios` / `parse_hybrid_pattern` / `select_pipeline_segment` 决定排列，`hybrid_layer_specs.py` 决定每层的实现，二者解耦。

下一篇预告：Gated DeltaNet / Kimi DeltaAttention（`gated_delta_net/`）这一支更偏「线性注意力」的 SSM 变体，以及它在 CP/Triton 下的实现。
