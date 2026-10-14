# 01 · KDA 前向算子（FLA `chunk_kda` 前向）整理

> 系列第 2 篇。把 FLA `fla/ops/kda` 的**前向**按数据流拆成四段：
> **Gate → Intra（块内）→ 状态递推 → 输出**，逐 kernel 给出 grid / 数式 / 落盘张量，
> 并标注真实源码行号。数学公式见 [00-math-and-api.md](00-math-and-api.md) §4；
> 优化手段见 [03-optimization.md](03-optimization.md)。

---

## 0. 编排：`chunk_kda_fwd`

`fla/ops/kda/chunk_fwd.py:20-142` 是前向编排，数据流：

```
q,k,v,g,beta
  └─(1) gate激活 + 块内 cumsum ─────────► g_cumsum  [B,T,HV,K]   (chunk_fwd.py:45-68)
  └─(2) chunk_kda_fwd_intra ─────────────► w,u,qg,kg, Aqk, Akk   (chunk_fwd.py:71-84)
  └─(3) chunk_gated_delta_rule_fwd_h ────► h, v_new, final_state (chunk_fwd.py:100-113)
  └─(4) chunk_gla_fwd_o_gk ──────────────► o                    (chunk_fwd.py:122-134)
```

细节：

- `disable_recompute=False`（默认）时 kernel (1)(2)(3) 产生的 `w,u,qg,kg,v_new,h`
  在算完 $o$ 后**被释放**（`chunk_fwd.py:135-142`），反向靠**重算**恢复它们；
  `disable_recompute=True` 时全部保留（`qg` 才会真正落盘，见 `wy_fast.py:80-84`）。
- CP（context parallel）模式在 (2) 后、 (3) 前有 `_pre_process` 做跨 rank 状态聚合
  （`chunk_fwd.py:86-98`），本篇不展开。

---

## 1. 门控 + 块内前缀和

两条分支（`chunk_fwd.py:45-68`）：

- `use_gate_in_kernel=True`（**Megatron 走的路径**，`kda.py:121`）：调
  `kda_gate_chunk_cumsum`（`gate.py:479-523`）。内核
  `kda_gate_chunk_cumsum_vector_kernel`（`gate.py:411-474`）一次完成
  激活 + 块内 cumsum：

  $$
  \text{gate}_t=-e^{A_\text{log}}\odot\operatorname{softplus}(g^{\text{raw}}_t+\text{dt\_bias}),\qquad
  g^{\text{cum}}_i=\sum_{j\le i}\frac{\text{gate}_j}{\ln 2}.
  $$

  对应源码 `gate.py:459-470`（`b_gate = -exp(b_A)*softplus(b_s)`、
  `tl.cumsum(b_gate, axis=0)`），`scale=RCP_LN2` 由 `chunk_fwd.py:53` 传入。
- `use_gate_in_kernel=False`：`g` 已是激活后的值，只做
  `chunk_local_cumsum(g, scale=RCP_LN2)`（`chunk_fwd.py:61-68`）。

产出 `g_cumsum`（log2 域、块内前缀和），同时作为 (2)(3)(4) 的 `gk`/`g`。

---

## 2. 块内：`chunk_kda_fwd_intra`

`fla/ops/kda/chunk_intra.py:807-912`。块大小 $B_T\in\{32,64\}$，子块 $B_C=16$，
$N_C=B_T/B_C$。落盘三张表（`chunk_intra.py:832-836`）：

- `Aqk ∈ [B,T,HV,BT]`：下三角（含对角）的 $\mathcal A_{ij}$ **已乘 scale**；
- `Akk ∈ [B,T,HV,BT]`：**零初始化**，最后写满 $(I+L)^{-1}$ 的完整逆（`chunk_intra.py:834`）；
- `Akkd ∈ [B,T,HV,BC]` fp32：只存**对角 16×16 子块的** $L$（供 solve_tril 用高精度）。

分两步：

### 2.1 对角子块（token 并行）

`safe_gate=False` 走 `chunk_kda_fwd_intra_token_parallel`（`chunk_intra_token_parallel.py:130-188`）；
`safe_gate=True` 走 `chunk_kda_fwd_kernel_intra_sub_chunk`（`chunk_intra.py:693-804`）。
前者 grid 是 `(B*T, cdiv(HV,BH))`，**每个 token 一个 program**（`chunk_intra_token_parallel.py:168`），
直接在全局内存按行写 `Aqk/Akkd`，避免对 pad 行的浪费：

```python
# chunk_intra_token_parallel.py:115-127
b_k = b_k * beta            # β_i 折进 key（:115）
for j in range(i_ts, min(i_t+1, min(T, i_ts+BC))):
    b_kgj = b_kj * exp2(b_g - b_gj)          # k_j * 2^{g_i-g_j}   (:122)
    b_Aqk = (b_q * b_kgj).sum(-1) * scale    # A_ij = q_i·k_kgj      (:123)
    b_Akk = (b_k * b_kgj).sum(-1) * (j<i)    # 严格下三角            (:124)
```

即 §4 的 $L_{ij}=\beta_i k_i\cdot k_j 2^{g^{\text{cum}}_i-g^{\text{cum}}_j}$（$i>j$）
与 $\mathcal A_{ij}$。注意 **$g$ 的减法**等价于「减参考点」的数值稳定写法。

`safe_gate` 版本用 `gather` 或直接寻址取子块中点 `b_gn`（`chunk_intra.py:756-766`），
把 $g-g_n$ 限制在安全范围（`exp2` 指数不溢出），并顺带做 16×16 的 forward substitution
（`chunk_intra.py:793-804`）。

### 2.2 非对角子块 + `solve_tril`：`chunk_kda_fwd_kernel_inter_solve_fused`

`chunk_intra.py:44-384`，grid `(NT, B*HV)`（`chunk_intra.py:879`）。一个 program 处理
一个 $B_T$ 块的全部 $N_C$ 个子块，做三件事：

1. **算非对角 $\mathcal A$ 与 $L$**（`chunk_intra.py:140-223`）：

   ```
   b_gqn = exp2(g_i - g_n)          # 以子块首/末为锚，安全
   b_kgt = trans(k_j * exp2(g_n - g_j))
   Aqk_ij = dot(q_i*b_gqn, b_kgt) ;  Akk_ij = dot(k_i*b_gqn, b_kgt)
   ```

   对 $N_C=2/3/4$ 展开成 `Aqk10/Akk10`、`Aqk20/Akk21/Akk22`… 这批寄存器矩阵
   （`chunk_intra.py:122-135` 声明，`:150-223` 计算）。

2. **落 $A_{qk}$（乘 scale）、$L$ 乘 $\beta_i$**（`chunk_intra.py:228-257`）；
   从 `Akkd` 读回**对角 $L$ 块**（`chunk_intra.py:259-268`）。

3. **forward substitution 求 $(I+L_{ii})^{-1}$**（`chunk_intra.py:274-313`），再**块前代合并**
   成完整 $(I+L)^{-1}$（`chunk_intra.py:320-356`）。合并式（以 2 子块为例）：

   $$
   (I+L)^{-1}=\begin{pmatrix} P_{00} & 0\\ P_{10} & P_{11}\end{pmatrix},\qquad
   P_{10}=-P_{11}\,L_{10}\,P_{00},
   $$

   对应 `chunk_intra.py:320-324`（`b_Ai10 = -dot(dot(b_Ai11, b_Akk10), b_Ai00)`）。
   4 子块时逐层展开（`chunk_intra.py:326-356`）。合并用 **tf32** `tl.dot`
   （`SOLVE_TRIL_DOT_PRECISION`，`chunk_intra.py:21-24`、`:320-356`），
   因为逆矩阵精度对 tf32 不敏感。最后把完整逆写回 `Akk`（`chunk_intra.py:362-384`）。

> 一句话：`Aqk` 供输出用，`Akk=(I+L)^{-1}` 供下一步 `W/U` 用。

### 2.3 `W/U`（及可选 `qg/kg`）：`recompute_w_u_fwd`

`wy_fast.py:271-320`，kernel `recompute_w_u_fwd_kda_kernel`（`wy_fast.py:34-132`），
grid `(NT, B*HV)`。用 `Akk`（即 $T=(I+L)^{-1}$）算

$$
U=T\,(\beta\odot v)\quad(\text{wy\_fast.py:101-104}),\qquad
W=T\,(\beta\odot k^{\text{dec}})\quad(\text{wy\_fast.py:131-132}),
$$

其中 `b_kb = k*β`，再乘 `exp2(gk)=E` 得 $k^\text{dec}$（`wy_fast.py:113-117`）。
`disable_recompute=False` 时 `qg` 不落盘（`wy_fast.py:80-84` 的 `STORE_QG=False`），
`kg`（$k^\text{rest}$）**始终**落盘（`wy_fast.py:124-129`）：

$$
k^{\text{rest}}_i=k_i\odot 2^{\,g^{\text{cum}}_\text{tot}-g^{\text{cum}}_i},\qquad
g^{\text{cum}}_\text{tot}=g^{\text{cum}}_{\min(i B_T+B_T,T)-1}.
$$

---

## 3. 状态递推：`chunk_gated_delta_rule_fwd_h`

`fla/ops/common/chunk_delta_h.py:688-747`，kernel
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`（`chunk_delta_h.py:59-349`），
grid `(cdiv(V,BV) * N * HV,)`（`chunk_delta_h.py:726`）——沿 **$V$ 切块、沿 $T$ 串行**。

这是 KDA 前向**唯一沿 $T$ 串行**的 kernel。状态常驻寄存器，按 $K$ 分 4 个 64 列块
`b_h1..b_h4`（`chunk_delta_h.py:100-115`），逐 chunk 递推（`chunk_delta_h.py:489-316`）：

1. **落当前块首状态** `h[chunk]`（`chunk_delta_h.py:493-523`）。
2. **算 $v_\text{new}=U-W S_\text{in}$**：`b_v = W @ h`（`:211-237`），
   `b_v = u - b_v`（`:238-239`），落盘（`:241-243`）。
3. **状态衰减**：$S\mathrel{*}=E_\text{tot}$（`chunk_delta_h.py:260-287` 的 `USE_GK` 分支，
   逐 key 通道 `exp2(gk_last)`）。
4. **状态累加**：$S\mathrel{+}=k^{\text{rest}\top}v_\text{new}$（`chunk_delta_h.py:290-316`）。

即 §4 的 $S_\text{out}=\operatorname{diag}(E_\text{tot})S_\text{in}+k^{\text{rest}\top}v_\text{new}$。
`output_final_state=True` 时落终态（`chunk_delta_h.py:318-349`）。

> 传参约定：KDA 调它时 `k=kg`（restored keys）、`u=u`、`gk=g_cumsum`、`g=None`，
> 所以走 `USE_GK` 分支（`chunk_fwd.py:100-113`）。`g` 只给纯 GLA 用。

---

## 4. 输出：`chunk_gla_fwd_o_gk`

`fla/ops/gla/chunk.py:990-1032`，kernel `chunk_gla_fwd_kernel_o`
（`fla/ops/gla/chunk.py:363-455`），grid `(cdiv(V,BV)*NT, B*HV)`。逐 $K$ 块累加
（`gla/chunk.py:420-444`）：

$$
\tilde o_i=\text{scale}\sum_k q_i\,2^{g^{\text{cum}}_i}\,S_\text{in}\quad(\text{gla/chunk.py:434-445}),
$$

再补块内项（`gla/chunk.py:446-455`）：

$$
o_i=\tilde o_i+\sum_{j\le i}\mathcal A_{ij}\,(v_\text{new})_j \quad(\text{`b_A = Aqk` 掩下三角, gla/chunk.py:448-454}).
$$

注意传入的 `v` 是 `v_new`（`chunk_fwd.py:123`），`A` 是 `Aqk`（含 scale 与对角）。
`o` 初始化为 `torch.zeros_like(v)`（`gla/chunk.py:1011`），以兼容 padding。

---

## 5. 落盘张量表（`save_for_backward`）

`ChunkKDAFunction.forward` 保存（`chunk.py:121-145`）：

| 张量 | 来源 | 反向用途 |
|---|---|---|
| `q,k` | 输入（可能已 l2norm） | 块内反向、梯度 |
| `q_rstd,k_rstd` | `l2norm_fwd`（若开） | l2norm 反向 |
| `v` | 输入 | 块内反向 |
| `g_cumsum` | gate/ cumsum | 块内反向、门控反向的 `dyg` 载体 |
| `g_input` | raw g | `kda_gate_bwd` |
| `beta_raw,beta` | beta / sigmoid | beta 反向 |
| `A_log,dt_bias` | 参数 | 门控反向 |
| `Aqk,Akk` | intra | $dA_{qk},dA_{kk}$ 入口 |
| `w,u,qg,kg,v_new,h` | intra / fwd_h | **复用**（`disable_recompute=True`）或占位 |
| `initial_state` | 输入 | $dh_0$ |
| `cu_seqlens,chunk_indices,chunk_offsets` | 变长 | 反向索引 |

`disable_recompute=False`（默认）时 `w,u,qg,kg,v_new` 在返回前被置 `None`
（`chunk_fwd.py:135-142`），反向在 `chunk_kda_bwd` 里用 `recompute_w_u_fwd` + `fwd_h`
**重算**（`chunk_bwd.py:505-532`）。这是**用时间换显存**的选择，也是前向移植时
需要与 Megatron 的 `recompute_gdn` 对齐的点（`kda.py:321-337`）。

---

## 6. 参考实现对照

`../src/kda_ref.py::kda_chunked_fwd`（`kda_ref.py:150-260`）是 §2–§4 的可运行等价物：

| FLA 阶段 | 参考实现位置 | 说明 |
|---|---|---|
| gate + cumsum | `kda_gate` / `_chunk_cumsum` | `kda_ref.py:35-75,120-122` |
| $(I+L)^{-1}$ | `_invert_lower_i_plus` | `kda_ref.py:125-143`（O(C) 前代） |
| $U,W$ | `kda_ref.py:238-240` | einsum 版 |
| 状态/输出递推 | `kda_ref.py:245-260` | 逐 chunk 循环 |

fp64 下 `kda_chunked_fwd` 与 `kda_token_recurrent` 的相对误差 $\le1.1\times10^{-14}$
（`src/kda_ref.out.txt` Part A）。

---

## 7. 小结

前向四段：**门控（log2 + cumsum）→ 块内（$A_{qk}$ 与 $(I+L)^{-1}$）→ 状态递推（沿 $T$ 串行）→
输出（$S_\text{in}$ + 块内项）**。落盘的核心中间量是 `Aqk/Akk/w/u/kg/v_new/h`，
反向要么复用要么重算。移植到 Megatron 时，**门控必须在 kernel 内做**（Megatron 传 raw g），
且 `A_log/dt_bias` 全程 fp32。

**下一篇** [02-backward.md](02-backward.md)：`chunk_kda_bwd` 的五段拆解与逐 kernel 数式。
