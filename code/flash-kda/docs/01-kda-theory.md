# KDA 数学与分块（chunk）公式

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 1 篇。先把 KDA 的数学定义、
> 门控构造、以及 FlashKDA 采用的 **CHUNK=16 分块公式**写清楚，作为后续精读 CUDA kernel 的地图。
> 公式与 FlashKDA 的 PyTorch 参照实现 `~/github/FlashKDA/tests/torch_ref.py` 逐项对应。

## 1. KDA 是什么：带门控的 delta rule

线性注意力把注意力写成「状态 + 递推」的形式。**DeltaNet / 门控 Delta rule** 的状态是一个
$D\times D$ 矩阵 $S_t$（这里 $K=V=D=128$），逐 token 更新：

$$
S_t = S_{t-1}\big(\operatorname{diag}(\alpha_t) - \beta_t\,k_t k_t^\top\big) + \beta_t\,v_t k_t^\top,
\qquad o_t = q_t^\top S_t,
$$

其中 $k_t,v_t,q_t\in\mathbb{R}^D$ 是逐头的 key/value/query，$\beta_t\in(0,1)$ 是写入门（delta 强度），
$\alpha_t\in(0,1]^D$ 是**逐通道**的遗忘/衰减门。

**KDA（Kimi Delta Attention）** 与 GLA / DeltaNet 的关键差别，就在这个门如何取：
KDA 使用 **per-channel（逐通道）的 log-decay** $g_t\in\mathbb{R}^D$（而不是每个 head 一个标量），
$\alpha_t=\exp(g_t)$ 在 $D$ 个通道上各不相同，表达能力更强。

## 2. 输入、输出与门控构造

FlashKDA 前向接口（`flash_kda/__init__.py`）的输入：

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q,k,v` | `[B,T,H,K]` | bf16 | query / key / value（$K=V=128$） |
| `g` | `[B,T,H,K]` | bf16 | 门控 logits（**激活前**） |
| `beta` | `[B,T,H]` | bf16 | 写入门 logits（**激活前**，kernel 内做 sigmoid） |
| `A_log` | `[H]` | fp32 | log 门参数 |
| `dt_bias` | `[H,K]` | fp32 | 门偏置 |
| `lower_bound` | 标量 | float | 门下界（期望在 $[-5,0]$） |

门控在 kernel 里这样构造（对应 `torch_ref.py:158–169`）：

1. **L2 归一化**：对每行 $q,k$ 做 $\ell_2$ 归一化（kernel 用 warp shuffle 树归约 + FMA，故意
   与逐元素累加顺序一致，见 `l2_normalize_kernel_match`）。
2. **加门偏置**：$g \leftarrow g + \mathrm{dt\_bias}$（fp32）。
3. **缩放 + sigmoid**：令 $a=\exp(A\_log)$，
   $$
   g \leftarrow \text{lower\_bound}\cdot\log_2 e \;\cdot\; \sigma\big(a\cdot g\big),
   $$
   sigmoid 用 PTX `tanh.approx.f32` 实现（$\sigma(x)=\tfrac12\tanh(x/2)+\tfrac12$），既快又够准。
4. $\beta \leftarrow \sigma(\beta)$（同样用 `tanh.approx`）。

于是 $g_t\le 0$，$\exp(g_t)\in(0,1]$ 就是逐通道衰减。

## 3. 分块（chunk）前向公式

把序列按 $\mathrm{CHUNK}=16$ 切块。对某一块（长度 $C$，通常 $C=16$），记该块的
$q,k,v,g\in\mathbb{R}^{C\times D}$、$\beta\in\mathbb{R}^{C}$，块起始的递推状态为
$S_{\text{in}}\in\mathbb{R}^{D\times D}$（布局为 $[V,K]$，即 `torch_ref` 里的 `state_slice`）。按 `torch_ref.py:208–239`：

**① 累积门 / 衰减因子**

$$
g^{\text{cum}}_i=\sum_{j\le i} g_j,\qquad
g^{\text{total}}=g^{\text{cum}}_{C-1},\qquad
E_i = 2^{\,g^{\text{cum}}_i},\quad E^{\text{tot}}=2^{\,g^{\text{total}}} .
$$
（代码里 `fp32_ex2_ftz` 即 `ex2.approx.ftz.f32`；base-2 免去换底 FMA。）

**② 衰减后的 q/k 与「还原」用的 k**

$$
k^{\text{dec}}_i = k_i\odot E_i,\quad
q^{\text{dec}}_i = q_i\odot E_i\cdot\text{scale},\quad
k^{\text{inv}}_i = k_i\odot E_i^{-1},\quad
k^{\text{rest}}_i = k^{\text{inv}}_i\odot E^{\text{tot}} .
$$

**③ 块内注意力矩阵**

$$
L = \operatorname{tril}\!\big(k^{\text{dec}}\,{k^{\text{inv}}}^\top\big)\odot \beta \ (\text{严格下三角}),\qquad
M_{qk} = \operatorname{tril}\!\big(q^{\text{dec}}\,{k^{\text{inv}}}^\top\big).
$$

**④ 块内求逆** $P=(I+L)^{-1}$
（$L$ 严格下三角且已乘 $\beta$）。FlashKDA 对 $16\times16$ 用「两个 $8\times8$ 的 fp32 前代 + 一次
bf16-HMMA 的 off-diagonal 合并」精确求逆（`inv_fwd_subst_16` / `fwd_kernel1.cuh` 的
`inv` 段），比 Neumann 级数在 $|L|\to1$ 时更稳。

**⑤ 修正后的 value 与输出**

$$
\tilde v_i = \big(v_i - k^{\text{dec}}_i S_{\text{in}}^\top\big)\,\beta_i,\qquad
U = P\,\tilde v,
$$
$$
o_i = q^{\text{dec}}_i S_{\text{in}}^\top + (M_{qk}U)_i .
$$

**⑥ 状态更新**

$$
\Delta S = {k^{\text{rest}}}^\top U,\qquad
S_{\text{out}} = \big(\Delta S + S_{\text{in}}^\top \odot E^{\text{tot}}\big)^\top
= E^{\text{tot}}^\top\!\odot S_{\text{in}} + \Delta S^\top .
$$

直觉：$E^{\text{tot}}$ 把整块累积衰减作用到旧状态（块内所有 token 的衰减一次性结算），
$\Delta S$ 是本块「delta 写回」的增量。

> 说明：以上分块公式与逐 token 递推（式 1）在 fp32 下等价，差异只在数值与计算顺序。
> 反向的推导、以及 chunked backward 如何复用 $L$ / $P$ / $M_{qk}$，见后续篇目（`05-kda-bwd.md`）。

## 4. 为什么拆成两个 kernel（K1 / K2）

FlashKDA 把前向拆成两个 kernel（见 `docs/20260420-flashkda-v1-deep-dive.md` §2）：

- **K1（token 并行，grid = $N\times H\times\text{num\_chunks}$）**：门激活 → L2 归一化 → 衰减施加
  → 构造 $L$ / $M_{qk}$ → 块内求逆；把 $k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},g^{\text{total}},
  P,M_{qk}$ 写进 workspace。
- **K2（head 并行，grid = $N\times H$）**：逐块跑 delta-rule 递推 → 输出投影 → 状态累加。

单 kernel 版本里，K1 的 token 并行度会被 K2 很低的并行度拖住，大量 SM 空转；拆开后至少快 **15%**，
而且两段可各自调优。这里是后文精读 K1/K2 的动机。

## 5. 下一步

- 精读 K1（`docs/02-kda-fwd-k1.md`）：TMA 载入、L2 norm、cumsum、decay、$L/M_{qk}$、求逆。
- 精读 K2（`docs/03-kda-fwd-k2.md`）：warp specialization、八个 Phase、状态更新。
- 数值对拍（`docs/04`）与反向（`docs/05`）。
