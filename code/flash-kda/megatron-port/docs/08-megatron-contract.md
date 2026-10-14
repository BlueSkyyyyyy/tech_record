# 08 · Megatron 调用契约逐项对齐（M9-1）

> 系列第 9 篇。前八篇把 FLA `fla/ops/kda` 的前反向、门控/布局边界、chunk/pad/varlen/CP
> 都锚定到了 `src/kda_ref.py`（M6–M8）。但那些锚定都是**直接调 FLA**，还没回答一个移植前
> 必须钉死的问题：**Megatron `megatron/core/ssm/gated_delta_net/kda.py` 到底怎么准备张量、
> 以什么参数、什么布局去调 `chunk_kda`？** 本篇按 `docs/02` §7 的移植检查表**逐项对齐**这套
> **调用契约**，并用 `harness/check_port_contract.py` 真机核验（含真实误差数字）。
>
> 行号按本机版核对：`Megatron-LM`（`kda.py`/`common.py`/`gdn.py`）与容器内 pip 包
> `fla==0.5.2`（`/usr/local/lib/python3.12/dist-packages/fla/ops/kda/chunk.py`）。

---

## 1. 契约全貌：`_forward_compute` 的一次 KDA 调用

Megatron `KimiDeltaAttention` 的前向骨干预备在一个函数里（`kda.py:359-490`
`_forward_compute`）：in_proj → conv1d → 切 q/k/v/raw_g/output-gate → 调 `chunk_kda` →
gated norm → out_proj。与本专题相关的**调用面**收敛到下面这些行：

| 契约点 | 代码位置 | 事实 |
|---|---|---|
| 布局 | `kda.py:533-535` | 先 `qkvfg.transpose(0,1)`、`beta.transpose(0,1)`，把 SBHD 的 `[S,B,...]` 转成 FLA 要的 `[B,S,...]`，再 `torch.split` 切段 |
| 输入投影 | `kda.py:376-377` | `in_proj` 出 `qkvfg`，**beta 单独走 `beta_proj`**（`kda.py:100-113`） |
| 切段尺寸 | `kda.py:134-140`（`_setup_variant_attrs`） | `in_proj_split_sections = (qk, qk, v, qk, v)`，即 `q,k,v,raw_g,output_gate` |
| beta 激活 | `kda.py:181` | `beta = beta.reshape(...).float().sigmoid()`——**sigmoid 在 kernel 外** |
| raw g | `kda.py:180` | `raw_g = raw_g.reshape(batch, seq_len, num_key_heads, key_head_dim)`，**未激活** |
| 门控进 kernel | `kda.py:120-121` | `self.use_gate_in_kernel = True`（把 $A_\text{log}$/`dt_bias` 激活留给内核） |
| 门控参数精度 | `kda.py:117-118` | `mark_keep_in_fp32(self.dt_bias)` / `mark_keep_in_fp32(self.A_log)` |
| l2norm | `kda.py:443` + `common.py:431-432` | `use_qk_l2norm_in_kernel=False`，q/k 的 L2 norm 在外层 `l2norm(...)` 做 |
| 调用 | `kda.py:439-449` | `self.gated_delta_rule(**kernel_inputs, initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, safe_gate=..., lower_bound=..., cu_seqlens=..., cp_context=...)` |
| 初/终态 | `kda.py:441-442` | 训练路径 `initial_state=None`、`output_final_state=False` |

`harness/check_port_contract.py` 把这些 transfer 逐字复刻成一个 `megatron_pre_gdr()`
shim（只做 tp/cp=1、SHBD→BHSD、切段、l2norm、GVA、beta sigmoid），再以
`use_gate_in_kernel=True` 喂 FLA，最后与 `src/kda_ref.py` 对比。

---

## 2. Part A：SBHD→BHSD 转置就是 kernel 要的布局（`kda.py:533-535`）

Megatron 内部张量是 **SBHD**（`[seq, batch, hidden]`），FLA `chunk_kda` 的公开 API 是
**BHSD**（`[batch, seq, head, dim]`）。`pre_gated_delta_rule` 在切片前先做两次 `transpose(0,1)`：

```python
# kda.py:533-535
qkvfg = qkvfg.transpose(0, 1)
beta = beta.transpose(0, 1)
qkv, raw_g, gate = torch.split(qkvfg, self._get_feat_dim_split(cp_size_headwise), dim=-1)
```

**为什么必须是转置而不是 reshape**：`in_proj` 的输出在 SBHD 下内存序是
`(s, b, section)`，而 FLA 期望 `(b, s, section)`。这两者不是简单地重排 section——若用
`reshape` 去"凑" `[B,S,...]`，会把 batch 与 seq 两个轴揉在一起、完全错位。`transpose(0,1)`
把 `s` 与 `b` 轴对调，**数学上等价于**先在 BHSD 布局下构出同样的张量。

**核验**（Part A）：把同一份随机 `qkvfg` 分别按 SBHD→transpose 与直接按 BHSD 构造，
两条路出来的 q/k（经 l2norm）**逐位相同，`max_abs=0`**：

```
[Part A] BHSD-built q/k vs SBHD-transposed q/k: max_abs=0.00e+00   OK
```

---

## 3. Part B/C：raw g + `use_gate_in_kernel=True`，beta 在核外 sigmoid

### 3.1 beta 是独立投影、sigmoid 在核外（`kda.py:100-113,181`）

KDA 把写入门强度拆到**单独的 `beta_proj`**（`kda.py:100`），且 `_compute_gates` 直接
`beta.reshape(...).float().sigmoid()`（`kda.py:181`）。这有两个契约后果：

1. 传给 `chunk_kda` 的 `beta` **必须是 sigmoid 之后**的 `(0,1)`，且 `chunk_kda` 的
   `use_beta_sigmoid_in_kernel` 保持默认 `False`（`chunk.py` 签名）——否则会被二次 sigmoid。
2. 参考实现 `src/kda_ref.py` 的 `beta` 入参约定正是"**post-sigmoid**"（`kda_ref.py:14-15`），
   与 Megatron 对齐；**不能**把 raw logits 直接喂参考。

**核验**（Part C）：`torch.sigmoid(raw_logits)` 与 shim 出来的 `beta` **`max_abs=0`**。

### 3.2 raw g：门控激活必须在 kernel 内（`kda.py:120-121`）

Megatron 把**未激活**的 `raw_g`（`kda.py:180`）连同 `A_log`/`dt_bias` 一起丢给 FLA，
由 `use_gate_in_kernel=True`（`kda.py:121`）让内核做
$g=-e^{A_\text{log}}\,\text{softplus}(g_\text{raw}+\text{dt\_bias})$。这与普通
`use_gate_in_kernel=False`（g 已是激活后的 log-decay）是**两种不同的入口契约**。

**核验**（Part B）：按 `kda.py:439-449` 的原样参数（`use_gate_in_kernel=True`、
`use_qk_l2norm_in_kernel=False`、`initial_state=None`）跑 FLA，与参考
（`kda_gate` 先算激活、再 `kda_chunked_fwd`）对比：

```
[Part B] FLA(kernel) vs ref: rel|do|=1.92e-03  rel|dh|=1.11e-03   OK
shape B=2 S=256 H=4 HV=4 K=64 V=64
```

1.9e-3 是 fp32 入口下 bf16/tf32 kernel 的底噪（与 M6 的 `o/S ≤6.4e-3` 同量级），
说明"raw g 进内核"这条契约在数值上成立。

---

## 4. Part D：`A_log`/`dt_bias` 必须 fp32（`kda.py:117-118`）

门控参数被 `mark_keep_in_fp32` 钉死（`kda.py:117-118`），全程 fp32。**为什么**：
$g=-e^{A_\text{log}}\text{softplus}(\cdot)$ 里 $e^{A_\text{log}}$ 的 A_log 取值在
$\ln[1,16]$（`A_init_range=(1,16)`，`kda.py:69`），指数对 A_log 的**相对误差**会被原样放大
进 $g$；而 $g$ 又是逐 token 连乘的衰减基底，长序列上误差累积。

**核验**（Part D）：把 `A_log`/`dt_bias` 人为降 bf16 再算门控，与 fp32 对比：

```
[Part D] gate bf16 params vs fp32: max_abs=1.27e-01 max_rel=4.96e-03
         (g range [-25.610, -0.000])
```

单个门控值的绝对偏差可达 **0.127**、相对 **~5e-3**——落进 $2^g$ 的指数后会放大成
远大于 bf16 表示底噪的衰减误差。这正是 `mark_keep_in_fp32` 的必要性证据：
**模型整体 bf16 时，`A_log`/`dt_bias` 仍须保持 fp32**。

---

## 5. Part E：l2norm 在核外做（`kda.py:443` + `common.py:431-432`）

Megatron 用 `use_qk_l2norm_in_kernel=False`（`kda.py:443`），q/k 的 L2 norm 在
`_prepare_input_for_gated_delta_rule` 里先行完成（`common.py:431-432`：
`query_key = l2norm(query_key.contiguous())`）。**为什么**：Megatron 把 q/k 的归一化与其它
投影后处理（reshape、GVA 展开、contiguous）融在一个 `jit_fuser` 里做，避免再走一趟 kernel；
内核里的 `use_qk_l2norm_in_kernel=True` 分支与之等价。

**核验**（Part E）：用未归一化的 q/k + `use_qk_l2norm_in_kernel=True` 跑 FLA，与
核外归一化 + `False` 的结果对比：

```
[Part E] l2norm(outside) vs l2norm(in-kernel): rel|do|=2.48e-04   OK
```

2.5e-4 是两种归一化在 fp32/bf16 边界的舍入差，等价性成立。

---

## 6. Part F（契约更正）：KDA-in-Megatron **不支持 GVA**

`src/kda_ref.py` 与 FLA 都支持 **GVA**（$H_V>H$，M7 已锚定）。但 Megatron KDA 的
`_compute_gates` 把 raw g reshape 成 **`num_key_heads`** 个 head（`kda.py:180`）：

```python
raw_g = raw_g.reshape(batch, seq_len, num_key_heads, self.key_head_dim)
```

而 FLA 硬断言 `g.shape == (B, T, HV, K)`（`chunk.py:411`）。当 `num_value_heads > num_key_heads`
时，Megatron 产出的 `g.shape=(B,T,H,K)` 会**直接撞断言**。`kda.py` 自身的 docstring 也写明
"supports training with equal query/key and value head layouts"（`kda.py:55-57`）。

**核验**（Part F）：取 `H=2, HV=4` 走一遍 shim，FLA 报：

```
Megatron _compute_gates -> g.shape=(2, 256, 2, 64)  v.shape=(2, 256, 4, 64)
FLA rejects g.shape (2, 256, 2, 64) for HV=4:
  g must have shape [B, T, HV, K]=[2, 256, 4, 64], got [2, 256, 2, 64]   OK
  => KDA-in-Megatron requires num_value_heads == num_key_heads
```

**移植结论**：`common.py:438-442` 的 `repeat_interleave` GVA 展开（q/k）只解决了 q/k 侧；
g/beta 侧仍按 key-head 数产出，**所以 Megatron 的 KDA 路径必须 `num_value_heads == num_key_heads`**。
若要启用 GVA，需要在 `_compute_gates` 里把 `raw_g`/`beta` 也按 value-head 展开
（`repeat_interleave`），并相应调整独立的 `beta_proj` 输出维度——这是**上游尚未实现**的缺口，
不是本参考实现的缺陷。

---

## 7. Part G：训练期初/终态契约与 `dht`

### 7.1 训练路径（`kda.py:441-442`）

`initial_state=None`、`output_final_state=False`。**核验**：传一个 fp32 全零
`initial_state` 与 `None` 结果**逐位相同**（`rel|do|=0`）。

### 7.2 FLA 要求初始状态 fp32（`chunk.py:383`）

`assert initial_state.dtype == torch.float32, "initial_state must be in float32."`。
**核验**：传 bf16 初始状态被拒。**为什么**：状态 $S\in\mathbb R^{K\times V}$ 是**跨 chunk
累积**的递推量，低精度会累积；FLA 只允许 fp32 状态入口。

### 7.3 反向 `dhu` 的 `dh` 初值用 `dht` 而非 `h0`（`docs/02` §2）

这是**反向 kernel 的契约**：反向状态伴随 $\bar S$ 的递推沿时间倒序进行，其"初值"
（即最后一块之后的状态伴随）来自上层传下来的 **`dht`**（对 `final_state` 的梯度），
而不是前向的初始状态 `h0`。FLA `chunk_bwd.py:442-447` 里 `initial_state` 与 `dht` 是
**两个独立入参**，分别喂给 `chunk_gated_delta_rule_bwd_dhu(h0=initial_state, dht=dht)`
（`chunk_bwd.py:546-547`）。移植时若把 `dht` 误当 `h0`，则最后一块的状态梯度整段错。
训练路径 `output_final_state=False` ⟹ `dht` 是**零**，这是最常见的一档；但只要用户用
`initial_state`/`output_final_state` 做状态透传（如分块推理、CP），`dht` 就必须显式接入。

### 7.4 CP 下初/终态被禁用（`chunk.py:362-364`）

`cp_context is not None` 时 `assert initial_state is None` 且 `output_final_state is False`。
CP 的跨 rank 状态由 `fwd_h_pre_process`/`bwd_dhu_pre_process` 的 `all_gather`+`merge`
重建（M8-1/M8-2 已锚定），**不走** `initial_state`/`dht` 入口。这与 `docs/07` §关键事实一致。

---

## 8. Part H：静态契约审计 + `gdn_pre_gated_delta_rule_fusion` 缺口

`harness/check_port_contract.py` 的 Part H 直接在源码里搜关键行，防止"行号漂移"：

```
OK  raw g handed to kernel     <- kda.py: 'self.use_gate_in_kernel = True'
OK  dt_bias fp32               <- kda.py: 'mark_keep_in_fp32(self.dt_bias)'
OK  A_log fp32                 <- kda.py: 'mark_keep_in_fp32(self.A_log)'
OK  l2norm outside kernel      <- kda.py: 'use_qk_l2norm_in_kernel=False'
OK  fusion gap                 <- kda.py: 'is not implemented for KDA yet'
OK  kernel entry               <- kda.py: 'self.gated_delta_rule = chunk_kda'
OK  GVA not supported          <- kda.py: 'supports training with equal query/key and value'
OK  l2norm applied             <- common.py: 'query_key = l2norm(query_key.contiguous())'
OK  GVA q / GVA k              <- common.py: 'query/key = ...repeat_interleave(repeat_factor, dim=2)'
OK  FLA fp32 state assert      <- chunk.py: 'initial_state must be in float32'
OK  FLA gate assert            <- chunk.py: 'A_log must be provided when use_gate_in_kernel=True'
GDN fusion path present in gdn.py: True
```

### 8.1 `gdn_pre_gated_delta_rule_fusion` 缺口（`kda.py:399-402`）

普通 GDN 有**融合流式前处理**路径（`gdn.py:318-350`
`_fused_streamed_pre_gated_delta_rule`：把 conv1d + 门控 + 布局整理融成一条 streamed kernel）。
但 KDA 在同一个开关下**直接抛 `NotImplementedError`**：

```python
# kda.py:399-402
if self.gdn_pre_gated_delta_rule_fusion:
    raise NotImplementedError(
        "gdn_pre_gated_delta_rule_fusion is not implemented for KDA yet."
    )
```

**为什么是缺口**：KDA 的门控比 GDN 复杂（**per-channel** raw g + `A_log`/`dt_bias` 激活 +
chunk 内 cumsum），而 GDN 的门是 per-head 标量，融合 kernel 的模板不同。因此把
`gdn_pre_gated_delta_rule_fusion` 用于 KDA 会走一条**尚未写出的融合实现**；
Megatron 选择显式报错而非静默走错误路径。移植时若要补，需要写 KDA 版的
streamed 融合前处理（conv1d 输出 → 切 q/k/v/raw_g/gate → l2norm → beta sigmoid 全流式），
**不能**复用 GDN 的 `_fused_streamed_pre_gated_delta_rule`。

### 8.2 `chunk_size` 只允许 32/64（`chunk.py:391`）

与 M8-1 的更正一致：`if chunk_size not in (32, 64): raise ValueError`（`chunk.py:391`）。
Megatron 未显式传 `chunk_size`，走 FLA 默认 64。

---

## 9. 小结

- **调用契约已逐项对齐**（`harness/check_port_contract.py`，真机 ALL PASS）：
  SBHD→BHSD 转置（`max_abs=0`）、raw g + `use_gate_in_kernel=True`
  （vs 参考 `rel|do|=1.9e-3`）、beta 核外 sigmoid（`max_abs=0`）、
  `A_log`/`dt_bias` fp32（bf16 化给门控带来 `max_abs=0.127`）、l2norm 核外
  （vs 核内 `rel|do|=2.5e-4`）、`initial_state` fp32 断言、`dht` vs `h0` 的分工。
- **一条契约更正**：FLA/参考支持 **GVA**，但 **Megatron KDA 不支持**
  （`num_value_heads == num_key_heads` 是硬约束，`kda.py:55-57,180` vs `chunk.py:411`）。
  移植时不要把 M7 的 GVA 锚定当成 Megatron 已具备的能力。
- **已知缺口**：`gdn_pre_gated_delta_rule_fusion` 对 KDA 未实现（`kda.py:399-402`），
  补它需要 KDA 专属的 streamed 融合前处理。

**下一篇**：M10-1（在 Megatron 路径上端到端前反向对拍）。
