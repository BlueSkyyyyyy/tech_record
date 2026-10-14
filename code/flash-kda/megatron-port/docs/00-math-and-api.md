# 00 · 数学原理与算子函数接口

> 「把 FLA KDA 前反向算子移植进 Megatron-LM `gated_delta_net`」系列第 1 篇。
> 目标：先用**自洽的数学**把 Kimi Delta Attention（KDA）讲清楚，再把 FLA
> `fla/ops/kda` 的**公开接口**与 Megatron `megatron/core/ssm/gated_delta_net` 的**调用面**
> 对齐，作为后续两篇（前向 / 反向）与优化篇的坐标原点。
>
> 参考坐标（本机 clone 于 `~/proj/`）：
> - FLA：`flash-linear-attention`（`fla/ops/kda/*`、`fla/ops/common/chunk_delta_h.py`）
> - Megatron：`Megatron-LM`（`megatron/core/ssm/gated_delta_net/{kda,gdn,common}.py`）
> - 本目录参考实现：`../src/kda_ref.py`（CPU 可跑，纯 PyTorch，自带 fp64 自测）

---

## 1. 记号与维度

单层、单头（多头复用同一套公式；GVA 见 §3.4）。所有张量布局一律 **`[B, T, H, *]`**
（FLA 的公开 API；Megatron 内部是 `[T, B, H, *]` 的 SBHD，只在调 kernel 前转置，
见 `kda.py:533-535`）。

| 符号 | 形状 | 含义 |
|---|---|---|
| $q,k$ | $[B,T,H,K]$ | query / key（KDA 里 $K$ 就是 key 维，常取 128） |
| $v$ | $[B,T,H,V]$ | value（$V$ 常取 128） |
| $g$ | $[B,T,H,K]$ | **逐通道**（per-key-channel）对数衰减；已激活后 $\le 0$ |
| $\beta$ | $[B,T,H]$ | 写入门（sigmoid 之后 $\in(0,1)$） |
| $S$ | $[B,H,K,V]$ | 递推状态（FLA 默认 $[K,V]$；`state_v_first=True` 时是 $[V,K]$） |
| $o$ | $[B,T,H,V]$ | 输出 |
| $A_\text{log}$ | $[H]$ | 每 head 的 log 步长（gate 用） |
| $\text{dt\_bias}$ | $[H\cdot K]$ | 每通道步长偏置（gate 用） |

$\text{scale}$ 默认 $K^{-1/2}$（`chunk.py:496-497`）。

---

## 2. 逐 token gated delta rule

KDA 是 **逐通道衰减 $\alpha$ + 写入门 $\beta$** 的 delta rule。令 $\alpha_t=2^{g_t}$
（FLA 内部以 **log2** 度量，见 §3.3；参考实现用自然对数，二者同构），则

$$
p_t=(\alpha_t\odot k_t)\,S_{t-1},\qquad
\delta_t=\beta_t\,(v_t-p_t),\qquad
S_t=\operatorname{diag}(\alpha_t)\,S_{t-1}+k_t\,\delta_t^{\top},\qquad
o_t=\text{scale}\cdot q_t\,S_t .
\tag{2.1}
$$

要点与对照：

- $\alpha_t$ 是**逐 key 通道**的行缩放（$\operatorname{diag}(\alpha_t)$ 作用在 $S$ 的 $K$ 维上），
  这是 KDA 与 GLA / DeltaNet 的关键差别。
- **DeltaNet**：有 $\beta$、无 $\alpha$；**GLA**：有 $\alpha$（每 head 标量）、无 $\beta$；
  **KDA**：两者兼有且 $\alpha$ 逐通道。
- 输出 $o_t$ 用**更新后**的状态 $S_t$（含当前 token 自身的写入）。这一点在分块实现里
  靠 $A_{qk}$ 的对角项还原，容易写错，见 §4。
- 状态递推沿 $T$ **严格串行**，块内可并行；这正是 chunk 算法的动机。

参考实现 `src/kda_ref.py::kda_token_recurrent`（`kda_ref.py:62-115`）逐式对应 (2.1)。

---

## 3. 门控（gate activation）

### 3.1 普通门

$$
g^{\text{act}}=-e^{A_\text{log}}\odot \operatorname{softplus}(g^{\text{raw}}+\text{dt\_bias}),\qquad
\alpha=e^{g^{\text{act}}}\in(0,1].
\tag{3.1}
$$

对应 `fla/ops/kda/gate.py::naive_kda_gate`（`gate.py:27-55`，kernel `kda_gate_fwd_kernel`
`gate.py:113-156`，其中 `b_yg = -exp(b_A) * softplus(b_g)` 在 `gate.py:147`）。

### 3.2 下界（safe）门

当 `lower_bound` 给定时（推荐 $-5$，即单步衰减 $e^{-5}\approx0.0067$）：

$$
g^{\text{act}}=\text{lower\_bound}\cdot\sigma\!\big(e^{A_\text{log}}\odot(g^{\text{raw}}+\text{dt\_bias})\big).
\tag{3.2}
$$

当 $A_\text{log}=\text{None}$ 时退化为 $\text{lower\_bound}\cdot\sigma(g^{\text{raw}}+\text{dt\_bias})$。
`safe_gate=True` 时单步门被夹在 $[\text{lower\_bound},0)$，可将块内 $M=16$ 的
TensorCore 加速打开（`chunk.py:288-291`）。对应
`naive_kda_lowerbound_gate`（`gate.py:58-93`）。

### 3.3 为什么是 log2

FLA 在 chunk 内核里把门控后的 $g$ 直接做**块内前缀和**，并用 `exp2` 取幂：
`chunk_fwd.py:61-68` 调 `chunk_local_cumsum(g, scale=RCP_LN2)`（$1/\ln 2$），
`chunk_fwd.py:49-59` 走 `kda_gate_chunk_cumsum`（`gate.py:479-523`），其内部
`kda_gate_chunk_cumsum_vector_kernel`（`gate.py:411-474`）同时完成
「激活 + 块内 cumsum + 乘 `scale`」。于是：

$$
g^{\log_2}_t = \frac{1}{\ln 2}\,g^{\text{act}}_t,\qquad
E_i = 2^{g^{\log_2,\text{cum}}_i}=\exp\!\Big(\sum_{j\le i}g^{\text{act}}_j\Big).
$$

用 log2 是为了让内核统一走 `exp2`（比 `exp` 少一次常数乘），**数学上等价**。
本目录 `src/kda_ref.py` 为可读性改用自然对数，并在 `kda_ref.py:16-19` 注明。

### 3.4 GVA（Grouped Value Attention）

$v,g,\beta$ 的 head 数记为 $H_V$，$q,k$ 的 head 数记为 $H$；当 $H_V>H$ 时，
每个 $qk$ head 服务 $H_V/H$ 个 value head（`chunk.py:249`、`chunk.py:489-492`）。
前向里 $q,k$ 用 $i_h=i_{hv}\,/\,(H_V/H)$ 寻址（`chunk_intra.py:81`），
反向最后把 $dq,dk$ 从 $[B,T,H_V,K]$ 归约回 $[B,T,H,K]$（`chunk_bwd.py:631-634`）。

---

## 4. 分块形式（chunk form）

把序列切成 $C$ 长的块（KDA 的 `chunk_size` 只允许 $32/64$，`chunk.py:473-475`；
块内再切 $B_C=16$ 的子块，`chunk_intra.py:826`）。块内局部下标 $i,j$，
$g^{\text{cum}}_i=\sum_{j\le i}g_j$，$E_i=2^{g^{\text{cum}}_i}$，$E_\text{tot}=E_{C-1}$。

定义（per value-head）：

$$
k^{\text{dec}}_i=k_i\odot E_i,\quad
k^{\text{rest}}_i=k_i\odot E_\text{tot}/E_i=k_i\odot 2^{\,g^{\text{cum}}_\text{tot}-g^{\text{cum}}_i},\quad
q^{\text{dec}}_i=q_i\odot E_i,
$$

$$
L_{ij}=\beta_i\,(k^{\text{dec}}_i\cdot k^{\text{inv}}_j)\ \ (i>j),\qquad
\mathcal A_{ij}=(\text{scale})\,(q^{\text{dec}}_i\cdot k^{\text{inv}}_j)\ \ (i\ge j),
$$

其中 $k^{\text{inv}}_j=k_j/E_j$。再令 $T=(I+L)^{-1}$，则

$$
\boxed{\;
U=T\,(\beta\odot v),\qquad
W=T\,(\beta\odot k^{\text{dec}}),\qquad
v_\text{new}=U-W\,S_\text{in},
\;}
$$

$$
S_\text{out}=\operatorname{diag}(E_\text{tot})\,S_\text{in}
+\sum_i k^{\text{rest}}_i\,(v_\text{new})_i^{\top},\qquad
o_i=\text{scale}\,(q^{\text{dec}}_i S_\text{in})+\sum_{j\le i}\mathcal A_{ij}\,(v_\text{new})_j .
\tag{4.1}
$$

**推导要点**（与 §2 逐 token 递推一致）：

1. 把 $p_i$ 里的初始态项提出：$\alpha_i\odot(\operatorname{diag}(E_{i-1})S_\text{in})
   =k^{\text{dec}}_i S_\text{in}$（$\alpha_i E_{i-1}=E_i$）。
2. 块内早期写入对 $p_i$ 的贡献系数为 $k_i\cdot k_j\,2^{g^{\text{cum}}_i-g^{\text{cum}}_j}$，
   写成矩阵即 $L$（含 $\beta_i$）。
3. 于是 $(\beta\odot v-k^{\text{dec}}S_\text{in})$ 经 $(I+L)^{-1}$ 得到修正后的写入
   $v_\text{new}$（FLA 叫 `v_new`，即「delta」）。
4. $S_\text{out}$ 与 $o$ 都只用 $S_\text{in}$、$v_\text{new}$ 表达；块间只传 $D\times D$ 状态。

**注意点**（都是实现里踩过的坑）：

- $o$ 用 $S_\text{in}$（块首状态）**而不是** $S_i$；块内贡献由 $\mathcal A$（含对角）与
  $v_\text{new}$ 补上。
- $S_\text{out}$ 的 $-\sum k^{\text{rest}}_i (W S_\text{in})_i^\top$ 项**不能省**，它刻画
  「初始态如何被块内 delta 修正」。FLA 内核的做法是先给状态乘 $E_\text{tot}$、再把
  $V_\text{new}=U-W S_\text{in}$ 累加回去（`chunk_delta_h.py:238-316`）。
- 直接实现 $k^{\text{inv}}=k/E$ 会在强衰减下**上溢**（$E$ 下溢到 0）。正确做法是用
  $2^{g^{\text{cum}}_i-g^{\text{cum}}_j}$（$i\ge j$ 时指数 $\le0$）逐对构造衰减矩阵，
  只对掩码内的元素取幂。本参考实现即如此（`kda_ref.py:196-224`），FLA 则用
  「减去块内参考点 $g_{n}$」的等价技巧（`chunk_intra.py:160-166`）。

参考实现 `kda_ref.py::kda_chunked_fwd`（`kda_ref.py:150-260`）逐式对应 (4.1)，
并在 fp64 下与逐 token 递推**逐元素相等**（见 §6）。

---

## 5. 算子函数接口

### 5.1 FLA 公开入口 `chunk_kda`

`fla/ops/kda/chunk.py:216` 定义，`dispatch` 装饰（可路由到 TileLang / flash_kda 后端）。
签名（省略 `**kwargs`）：

```python
chunk_kda(
    q, k, v, g, beta,
    scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    safe_gate=False, lower_bound=None, disable_recompute=False,
    return_intermediate_states=False, state_v_first=False,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None,
    use_graph=False, max_num_seqs=None,
) -> (o, final_state)
```

关键参数语义（逐条与 `chunk.py:241-358` 的 docstring 对齐）：

| 参数 | 作用 | 对应代码 |
|---|---|---|
| `g` | `use_gate_in_kernel=False` 时是**已激活**的 log 衰减；`True` 时是 raw logits，内核内部做 $-e^{A}\text{softplus}(g+\text{dt\_bias})$ + 块内 cumsum | `chunk_fwd.py:45-68` |
| `use_gate_in_kernel` | 见上；`True` 时必须给 `A_log`（或 `lower_bound`） | `chunk.py:467-471` |
| `beta` | `use_beta_sigmoid_in_kernel=True` 时是 raw logits，否则须已 sigmoid | `chunk.py:61-63` |
| `allow_neg_eigval` | 配合 beta sigmoid 把 $\beta$ 放到 $[0,2)$ | `chunk.py:206-207` |
| `safe_gate`/`lower_bound` | 下界门 + 块内 TensorCore 加速 | `chunk.py:477-481` |
| `state_v_first` | 状态存 $[V,K]$ 而非 $[K,V]$ | `chunk.py:306-307` |
| `disable_recompute` | 反向是否复用前向落盘的中间量（省算、费显存） | `chunk.py:298-301` |
| `cu_seqlens` | 变长（varlen）；`B` 必须为 1 | `chunk.py:453-463` |
| `use_graph` | CUDA/NPU graph 静态变长块表 | `chunk.py:318-341` |

`ChunkKDAFunction`（`chunk.py:24`）是 autograd wrapper：前向 `forward`
（`chunk.py:27-158`）调 `chunk_kda_fwd`，把中间量 `save_for_backward`
（`chunk.py:121-145`）；反向 `backward`（`chunk.py:162-211`）调 `chunk_kda_bwd`。

### 5.2 Megatron 调用面

Megatron 侧把 KDA 封成 `KimiDeltaAttention`（`kda.py:52`），
`_setup_variant_attrs` 里 `self.gated_delta_rule = chunk_kda`（`kda.py:144`）。
核心计算在 `_forward_compute`（`kda.py:359`）：

```python
# kda.py:376-377  in_proj -> qkvfg ; beta_proj -> beta
# kda.py:416-426  pre_gated_delta_rule(): conv1d + 切 q/k/v/raw_g/gate
# kda.py:439-449  调 FLA：
core_attn_out, _ = self.gated_delta_rule(
    q=query, k=key, v=value, g=raw_g, beta=beta,
    A_log=A_log, dt_bias=dt_bias,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=self.use_gate_in_kernel,   # kda.py:121 设为 True
    safe_gate=self.config.kda_safe_gate,
    lower_bound=self.config.kda_lower_bound,
    cu_seqlens=cu_seqlens_q, cp_context=chunkwise_cp_context,
)
```

要点：

- Megatron 把 **raw g** 直接丢给 FLA，让内核做 $A_\text{log}$/`dt_bias` 激活
  （`kda.py:120-121`）。因此移植时 §3 的门控必须在 kernel 内实现。
- `A_log`/`dt_bias` 被 `mark_keep_in_fp32`（`kda.py:117-118`），必须全程 fp32。
- `beta` 在 Megatron 侧已 `sigmoid`（`_compute_gates`，`kda.py:181`），所以
  `use_beta_sigmoid_in_kernel` 保持默认 `False`。
- GDN 变体（`gdn.py:33`）走 `chunk_gated_delta_rule`（来自 `fla.ops.gated_delta_rule`，
  `common.py:42`）；KDA 变体走 `chunk_kda`。二者共享 `fla/ops/common/chunk_delta_h.py`
  的状态递推内核。
- 确定性模式提供纯 torch 参考 `torch_chunk_gated_delta_rule`（`gdn.py:638-743`），
  它与 §4 的公式一一对应（`gdn.py:703-712` 即 decay_mask / attn / 前代）。

### 5.3 内部算子清单（移植时需要逐个对上的函数）

**前向**（详见 `01-forward.md`）：

| 阶段 | 函数 / kernel | 文件:行 |
|---|---|---|
| 编排 | `chunk_kda_fwd` | `chunk_fwd.py:20-142` |
| 门控+cumsum | `kda_gate_chunk_cumsum` | `gate.py:479-523` |
| 块内（token 并行对角块） | `chunk_kda_fwd_intra_token_parallel` | `chunk_intra_token_parallel.py:130-188` |
| 块内（子块对角线，safe gate） | `chunk_kda_fwd_kernel_intra_sub_chunk` | `chunk_intra.py:693-804` |
| 块内（非对角 + solve_tril） | `chunk_kda_fwd_kernel_inter_solve_fused` | `chunk_intra.py:44-384` |
| 块内编排 | `chunk_kda_fwd_intra` | `chunk_intra.py:807-912` |
| $W/U$ 重算 | `recompute_w_u_fwd` | `wy_fast.py:271-320` |
| 状态递推 | `chunk_gated_delta_rule_fwd_h` | `chunk_delta_h.py:688-747` |
| 输出 | `chunk_gla_fwd_o_gk` | `gla/chunk.py:990-1032` |

**反向**（详见 `02-backward.md`）：

| 阶段 | 函数 / kernel | 文件:行 |
|---|---|---|
| 编排 | `chunk_kda_bwd` | `chunk_bwd.py:461-654` |
| $dA_{qk},dv$ | `chunk_kda_bwd_dAv` | `chunk_bwd.py:324-379` |
| 状态伴随 | `chunk_gated_delta_rule_bwd_dhu` | `chunk_delta_h.py:751-822` |
| $dq,dk,dv,db,dg,dA_{kk}$ | `chunk_kda_bwd_wy_dqkg_fused` | `chunk_bwd.py:383-458` |
| 块内反向 | `chunk_kda_bwd_intra` | `chunk_intra.py:915-989` |
| 门控反向 | `kda_gate_bwd` | `gate.py:284-322` |
| 反向前缀和 | `chunk_local_cumsum(reverse=True)` | `chunk_bwd.py:637-644` |
| $W/U$ 前向重算 | `recompute_w_u_fwd` | `wy_fast.py:271-320` |
| l2norm / beta 反向 | `l2norm_bwd` / `fused_beta_sigmoid_bwd` | `chunk.py:203-207` |

---

## 6. 参考实现与自测

`../src/kda_ref.py` 把 §2、§4、§3 三套公式落成纯 PyTorch，并在 fp64、CPU 上自测：

```bash
python3 src/kda_ref.py        # 见 src/kda_ref.out.txt
```

实测（本机，2026-10，fp64）：

- **Part A**：逐 token 递推 vs chunked 前向，$o$ 与终态相对误差 $\le 1.1\times10^{-14}$
  （含 $T=130$、非整除块、$B=2$）。
- **Part B**：autograd 穿过 chunked 前向 vs 穿过逐 token，六项梯度
  （$dq,dk,dv,dg,d\beta,dh_0$）相对误差 $\le 7.2\times10^{-15}$。
- **Part C**：门控闭式梯度 vs autograd（softplus），相对误差 $\sim5\times10^{-8}$
  （因 `kda_gate` 内部转 fp32，与 FLA 一致）。

> 用途：这份参考实现既是「数学正确」的 golden，也是移植时逐 kernel 对拍的**可直接运行**底座
> （接口刻意对齐 §4 的中间量命名 `W/U/v_new`）。

---

## 7. 本篇结论

1. KDA = 逐通道衰减 $\alpha$ + 写入门 $\beta$ 的 delta rule；分块形式把块间状态压缩成
   $D\times D$，块内退化为 $(I+L)^{-1}$ 的小矩阵代数。
2. FLA 的公开入口是 `chunk_kda`；Megatron 用 `KimiDeltaAttention.gated_delta_rule`
   承载它，并把 **raw g** 交给内核做门控激活。
3. 门控在 log2 域里表达，与自然对数等价，仅是内核实现选择。
4. 强衰减下的数值安全点：**不要**直接算 $k/E$，用逐对 $2^{\Delta g}$ 或减参考点。

**下一篇** [01-forward.md](01-forward.md)：把 §5.3 的前向三段（gate → intra → fwd_h → o）
逐个对照到 Triton kernel 的 grid / block / 数式与落盘张量。
