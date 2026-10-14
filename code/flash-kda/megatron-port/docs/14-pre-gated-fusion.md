# pre-GDR 融合缺口：为什么 KDA 用不了 GDN 的 `fused_streamed_pre_gated_delta_rule`

> 本子专题（`megatron-port`）此前把 FLA KDA 算子移植进 Megatron-LM 的**数值**与**接口**
> 逐层锚定完毕（M0–M11，见 `docs/00`–`docs/13`）。本篇处理路线图里最后一条
> 明确点名的缺口：`gdn_pre_gated_delta_rule_fusion` 对 KDA 未实现
> （`kda.py:399-402`）。我们做三件事：**讲清结构差异（为什么不能直接复用 GDN 的融合核）**、
> **给出 KDA 专属 fused pre-GDR 的代数/布局契约并用纯 PyTorch 参考实现锚定**、
> **证明该参考可作为 `pre_gated_delta_rule` 的 drop-in 替换**（含真实前反向数字）。
>
> 本文所有行号按容器内 Megatron-LM（`~/proj/Megatron-LM`，2026-10）与
> `flash-linear-attention`（`fla` 0.5.x）核对；所有数字来自
> `harness/check_port_pre_gated_fusion.py`（`harness/check_port_pre_gated_fusion.out.txt`，
> 空闲卡单进程）。**本专题未改 Megatron-LM 一行**。

## 1. pre-GDR 是什么，融合的是什么

KDA 层的一次前向（`KimiDeltaAttention.forward`）中，核心 `chunk_kda` 之前的**全部张量准备**
统称 **pre-GDR**（pre gated-delta-rule）。对 KDA 而言就是
`kda.py:519` 的 `pre_gated_delta_rule` 及其调用的
`common.py:396` `_prepare_input_for_gated_delta_rule` + `kda.py:167` `_compute_gates`：

1. `qkvfg` 从 `in_proj` 出来 → 转置、按段切出 `qkv / raw_g / gate(Z)`；
2. 对 `qkv` 三个通道组做 **depthwise causal conv1d（SiLU 激活）**（`common.py:259-265` 构造，`kda.py:544-607` 调用）；
3. `qkv` 再切成 `query_key / value`，`query_key` reshape 后做 **q/k L2-norm**（`common.py:431-432`，FLA `l2norm`）；
4. 切出 `query/key`（Megatron KDA 强制 `num_value_heads==num_key_heads`，故 GVA `repeat_interleave` 因子为 1）；
5. `raw_g` reshape 成 `[B,T,Hk,KD]`，`beta` 走 `sigmoid`（`kda.py:180-184`）。

GDN 有一个**融合版**入口 `fused_streamed_pre_gated_delta_rule`
（`megatron/core/fusions/fused_pre_gated_delta_rule.py:2571`，一个 Triton 核族）把上面这些
逐 scope 融在一起，省掉多次 launch 和中间激活落盘；由模块属性
`self.gdn_pre_gated_delta_rule_fusion`（`kda.py:126` / `gdn.py:38`）控制开关。

## 2. 两道 guard：KDA 明确拒绝这条融合

在本机实测（`check_port_pre_gated_fusion.out.txt` Part A）两道 guard **都**触发：

- **配置期**（`TransformerConfig.__post_init__`）：
  ```python
  # transformer_config.py:1936-1938
  if self.gdn_pre_gated_delta_rule_fusion and self.experimental_attention_variant == "kda":
      raise NotImplementedError(
          "gdn_pre_gated_delta_rule_fusion is not implemented for KDA yet."
      )
  ```
- **前向期**（`kda.py:399-402`，即便绕过配置把模块属性手动置 True 也会命中）：
  ```python
  # kda.py:399-402
  if self.gdn_pre_gated_delta_rule_fusion:
      raise NotImplementedError(
          "gdn_pre_gated_delta_rule_fusion is not implemented for KDA yet."
      )
  ```

两道 guard 不是冗余防御，而是**同一句结构判断的两次落点**：GDN 的融合核与 KDA 的
pre-GDR 在**投影布局**、**beta 来源**、**门控数学**、**门控参数形状**四处都不同，直接复用会
静默算错。下面逐条给出源码证据。

## 3. GDN 融合核做了什么

`fused_streamed_pre_gated_delta_rule(qkvzba, conv1d_weight, conv1d_bias, A_log, dt_bias, ...)`
（`:2571`）接收**单个** `qkvzba` 投影，内部建了三条 streamed scope，最关键的两个是：

- `_conv_silu_project_kernel`（`:92`）——depthwise causal-conv + SiLU 直接吐 `q/k/v`；
- `_compute_g_and_beta_kernel`（`:434`）——**在核内**把 `alpha`/`beta` 两个通道段激活：

  ```python
  # fused_pre_gated_delta_rule.py:500-508
  pre = alpha + dt_bias[None, :]
  softplus_val = _softplus_with_torch_threshold(pre)
  g = -tl.exp(A_log)[None, :] * softplus_val      # g = -exp(A_log) * softplus(alpha + dt_bias)
  beta_sig = tl.sigmoid(beta)
  ```

  并把 `Z`（输出门）以 qkvzba 的**strided view** 返回、不 materialize。

返回 `(query, key, value, gate, beta, g)`（`:2698`），其中 **`g` 已是激活后的 log-decay**。

## 4. KDA 与 GDN 的四处结构差异

### 4.1 投影段不同：KDA 的 `g` 是 per-channel 且 `beta` 是独立投影

| | GDN | KDA |
|---|---|---|
| `in_proj_split_names` | `["query","key","value","z","beta","alpha"]`（`gdn.py:52`） | `["query","key","value","g","gate"]`（`kda.py:133`） |
| sections | `(qk, qk, v, v, Hv, Hv)`（`gdn.py:53-60`） | `(qk, qk, v, qk, v)`（`kda.py:134-140`） |
| `beta` 来源 | `in_proj` 里的独立段 | **独立的 `beta_proj`**（`kda.py:100-110`） |
| `in_proj_extra_dim` | `2*Hv`（`gdn.py:48`） | `qk_dim`（`kda.py:129`） |

KDA 的 `g` 段宽度是 **`qk_dim`（= `Hk·KD`，per-channel）**，不是 `Hv`（per-value-head）。
GDN 融合核的 `_compute_g_and_beta_kernel` 把 `alpha`/`beta` 都当 **`num_v_heads` 宽**的段来读
（`g_offs = pid_h*BLOCK_H + arange(BLOCK_H)`，`h_mask = h_offs < num_v_heads`，`:462-464`），
对 KDA 的 per-channel `g` 直接错位。

### 4.2 门控数学不同：KDA 把激活留给核心核

- GDN `_compute_gates`（`gdn.py:78-89`）：`g = -A_log.exp() * softplus(alpha + dt_bias)`，
  在 pre-GDR 里就算完。
- KDA `_compute_gates`（`kda.py:167-184`）：**原样返回 `raw_g`**，只把 `beta` sigmoid，
  并显式置 `self.use_gate_in_kernel = True`（`kda.py:121`），把 `A_log`/`dt_bias` 作为
  **kernel inputs** 一起交给 `chunk_kda`（`kda.py:437-444`）在核内做
  `-exp(A_log)*softplus(raw_g+dt_bias)`。

所以 KDA 的 fused pre-GDR **绝不能**像 GDN 那样在预处理里激活 `g`——否则会**双重激活**。
它必须返回 `raw_g` 与 `A_log/dt_bias` 本身（KDA 的 `pre_gated_delta_rule` 返回 8 元组，
比 GDN 多 `A_log/dt_bias`，见 `kda.py:629-638`）。

### 4.3 门控参数形状不同：KDA 的 `dt_bias` 是 per-channel

- GDN：`dt_bias_dim = a_log_dim = num_v_heads_local_tp`（`gdn.py:61-62`），都是 per-value-head 标量。
- KDA：`dt_bias_dim = qk_dim_local_tp`（**per-channel**），`a_log_dim = num_k_heads_local_tp`
  （per-key-head）（`kda.py:141-142`）。

这正是「**K**imi **D**elta **A**ttention 是 channel-wise 变体」的体现：每个 `(head, channel)`
有自己的 decay。GDN 融合核里 `A_log`/`dt_bias` 都按 `num_v_heads` 广播（`:504-506` 的
`[None, :]`），对 KDA 的 per-channel `dt_bias` 会广播错。

### 4.4 融合核的输入契约假定单投影 + causal-conv1d

`fused_streamed_pre_gated_delta_rule` 还硬性要求（`:2637-2656`）：`conv1d_bias is None`、
`use_qk_l2norm=True`、`num_value_heads % num_key_heads == 0`，且依赖
`causal_conv1d_bwd_function`。**本机容器未装外部 `causal-conv1d`**，故该入口在这里连
`ImportError` 都过不去（`fused_pre_gated_delta_rule.py:2625-2630`），而 KDA 走的是
`fla.modules.convolution.causal_conv1d`（`common.py:40`，见 §6 的实测归因）。

## 5. KDA 专属 fused pre-GDR 的契约

综合 §4，一个 KDA 专属融合入口需要：

- **输入**：`qkvfg`（`[S,B, 2qk+v+qk+v]`）、独立的 `beta`（来自 `beta_proj`）、
  `conv1d_weight/bias`、`A_log`、`dt_bias`；
- **conv scope**：对 `qkv`（前 `2qk+v` 个通道）做 depthwise causal-conv + SiLU，**不碰**
  `raw_g`/`gate`；
- **qk scope**：reshape + q/k L2-norm（fp32 内部累加，同 FLA `l2norm`）+ split；GVA
  `repeat`（KDA 中因子恒 1）；
- **g/beta scope**：`raw_g` reshape 到 `[B,S,Hk,KD]`；`beta` sigmoid；**不做** decay 激活；
- **输出**：`(q, k, v, gate, beta, raw_g, A_log, dt_bias)`，且 `gate` 尽量以
  `qkvfg` 的 strided view 返回（GDN 融合核的「streamed Z」卖点，见 `:10-12` docstring）。

`harness/check_port_pre_gated_fusion.py` 里 `kda_pre_gdr_fused_ref` 就是这个契约的
**纯 PyTorch 参考**（单函数、可 autograd 穿过），`conv_impl`/`l2norm_impl` 两个开关分别选
FLA 原语或 torch 原语。

## 6. 数值锚定（真实数字）

`T=64, B=2`，`hidden_size=128`、`Hk=Hv=4`、`KD=VD=32`、`conv_kernel_dim=4`，
`kda_safe_gate=True, lower_bound=-5`。数值全部取自 `check_port_pre_gated_fusion.out.txt`。

### Part B：参考 vs 真实层 eager `pre_gated_delta_rule`（逐张量 max_abs）

| tensor | fp32 (conv=FLA,l2=FLA) | fp32 (conv=torch,l2=torch) | bf16 (FLA,FLA) | bf16 (torch,torch) |
|---|---|---|---|---|
| q | 8.94e-08 | 1.19e-07 | **1.95e-03** | 3.91e-03 |
| k | 1.19e-07 | 1.19e-07 | **0** | 3.91e-03 |
| v | **0** | 5.96e-08 | **0** | 7.81e-03 |
| gate | **0** | **0** | **0** | **0** |
| beta | 5.96e-08 | 5.96e-08 | 1.19e-07 | 1.19e-07 |
| g (raw_g) | **0** | **0** | **0** | **0** |

两点结论：

1. **代数/布局完全等价**：用同一批 FLA 原语时，`v/gate/g` **逐位相同**，`k` 在 bf16 也逐位；
   `q`/`beta` 只剩 ≤1 ulp 的**原语归约序**残差（fp32 ~1e-7，bf16 `q` ~1.95e-3＝亚 ulp）。
   参考实现**run-to-run 自洽为 0**（Part B control），故这不是随机性，而是 FLA `l2norm`
   与 torch 版归约路径 / eager 调用点布局的微小差异。
2. **残差可归因到原语**：把 conv 换成 torch `F.conv1d`（自研融合核会自己实现 conv）后，
   `v` 从逐位 0 抬到 fp32 5.96e-08 / bf16 7.81e-03——即 **conv-backend 的舍入**；
   把 l2norm 换成 torch 版后 `q/k` 抬到 bf16 ~3.9e-03。这精确地把差异钉在
   「FLA causal_conv1d / l2norm 的 kernel 舍入」上，而非算子代数。

### Part C：把参考当 `pre_gated_delta_rule` 的 drop-in（整层前反向）

用逐位同权重的第二个模块，只把 `pre_gated_delta_rule` 换成 `kda_pre_gdr_fused_ref`
（`conv/l2=FLA`），跑完整前反向：

| dtype | out max_abs | hs.grad max_abs | param.grad max_abs | param.grad rel |
|---|---|---|---|---|
| fp32 | 6.48e-05 | 2.65e-04 | 1.38e-02（max\|g\|=75.7） | 1.82e-04 |
| bf16 | 9.77e-04 | 3.91e-03 | 2.50e-01（max\|g\|=75.5） | 3.31e-03 |

bf16 量级与 M10-1/M11-2 的「参考芯 vs FLA 芯」整模块锚定同阶（~3e-3），
说明**融合预处理对 `chunk_kda` 核心与整层梯度是数值透明的**；fp32 的 1e-4 级残差是
`l2norm` 原语舍入经 delta-rule 递推 + RMSNorm 放大后的结果，远低于部署底噪。

### Part D：streamed / 布局账

对 `T=64,B=2`：六个核心操作数（`q,k,v,gate,beta,g`）在 eager 路径里都**不是** `qkvfg` 的 view，
全部是新分配——fp32 共 **322.0 KiB**、bf16 共 **161.0 KiB**。其中 `gate` 的 eager
`.contiguous()` 会实打实 materialize 一份（fp32 64 KiB / bf16 32 KiB），这正是 GDN 融合核
「返回 strided Z、不物化 gate」能省掉的部分。诚实边界：本 SBHD cp=1 布局下其余操作数
（除 q/k 由 l2norm 核写出外）也已连续，所以融合的收益主要是**省 kernel launch + 省中间激活
读写**，而不是「省掉最终操作数的拷贝」——后者只在 `gate` 上成立。

## 7. 结论与边界

- **GDN 融合核不能复用**：四处结构差异（投影段、beta 来源、门控激活位置、门控参数形状）
  都是**会静默算错**的硬差异，两道 `NotImplementedError` guard 是必要的。
- **KDA 专属融合的契约是明确的**：一个单函数纯 PyTorch 参考
  （`kda_pre_gdr_fused_ref`）即可从 `qkvfg`/`beta` 复现 eager `pre_gated_delta_rule` 的核心
  操作数（用同一批 FLA 原语时 `v/gate/g` 逐位、`q/beta` ≤1 ulp），且作为
  `pre_gated_delta_rule` 的 drop-in 对整层前反向数值透明（bf16 param.grad rel 3.3e-3）。
  这就是未来一个 KDA Triton 融合核需要满足的代数/布局契约。
- **边界**：本实验为单卡 `tp=cp=1`、SBHD、`conv_impl/l2norm_impl` 只覆盖 FLA/torch 两档、
  `use_qk_l2norm=True`、GVA 因子为 1（Megatron KDA 硬约束）、未写 Triton 融合核、
  未做性能计时（仅报告布局/拷贝账）；容器未装外部 `causal-conv1d`，故 GDN 融合入口在本机
  也不可运行。
- **下一步（可选）**：若要真正关闭该缺口，需按 §5 契约实现一个 KDA 专属
  `fused_streamed_pre_gated_delta_rule` Triton 核族（conv/l2norm/raw_g/beta 三 scope）并接到
  `kda.py:399` 分支；或保持维护态。
