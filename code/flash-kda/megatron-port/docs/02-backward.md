# 02 · KDA 反向算子（FLA `chunk_kda` 反向）整理

> 系列第 3 篇。把 FLA `fla/ops/kda` 的**反向**拆成五段：
> **重算 → dAv → 状态伴随(dhu) → Q/K/G/B/A 伴随(wy_dqkg) → 块内(intra) → 门控**，
> 逐 kernel 给出数式与真实行号。前向数学见 [00-math-and-api.md](00-math-and-api.md) §4；
> 优化见 [03-optimization.md](03-optimization.md)。

反向的数据流是前向的**逆序**，且块间只传 $D\times D$ 的**状态伴随** $dh$。

---

## 0. 编排：`chunk_kda_bwd`

`fla/ops/kda/chunk_bwd.py:461-654`。输入是前向保存/重算出来的
`q,k,v,beta,Aqk,Akk,w,u,qg,kg,v_new,h,initial_state` 加上上游 `do, dht`。步骤：

```
disable_recompute=False（默认）：用 recompute_w_u_fwd + fwd_h 重算 w,u,qg,kg,v_new,h
                                  (chunk_bwd.py:505-532)
(1) dAv:  dAqk, dv ← chunk_kda_bwd_dAv(q,k,v_new,do,Aqk,scale)      (chunk_bwd.py:542-553)
(2) dhu:  dh, dh0, dv2 ← chunk_gated_delta_rule_bwd_dhu(qg,...,do,dv) (chunk_bwd.py:575-590)
(3) dqkg: dq,dk,dv,db,dg,dAkk ← wy_dqkg_fused(q,k,v,v_new,g,beta,Akk,h,do,dh,dv)(:592-611)
(4) intra: dq,dk,db,dg ← chunk_kda_bwd_intra(q,k,g,beta,dAqk,dAkk,...) (:613-629)
(5) gate:  dg → 反向前缀和 → kda_gate_bwd → dg,dA,dbias                (:637-652)
```

GVA（$H_V>H$）时最后把 `dq,dk` 从 $[B,T,H_V,K]$ 归约回 $[B,T,H,K]$
（`chunk_bwd.py:631-634`）。若前向开了 `use_qk_l2norm_in_kernel` /
`use_beta_sigmoid_in_kernel`，在 `ChunkKDAFunction.backward` 里再补
`l2norm_bwd` / `fused_beta_sigmoid_bwd`（`chunk.py:203-207`）。

### 0.1 重算（`disable_recompute=False`）

`chunk_bwd.py:492-532`：先按需重跑门控 cumsum（`:493-504`）、
`recompute_w_u_fwd`（`:505-515`）、`chunk_gated_delta_rule_fwd_h`（`:520-532`）。
这就是前向 [01-forward.md](01-forward.md) §5 里置 `None` 的那些张量的来源。
`disable_recompute=True` 时直接从 `kwargs` 取回（`:534`）。

---

## 1. 输出支路的伴随：`chunk_kda_bwd_dAv`

`chunk_bwd.py:324-379`，kernel `chunk_kda_bwd_kernel_dAv`（`chunk_bwd.py:44-119`），
grid `(NT, B*HV)`。只处理输出支路 $o=\mathcal A\,v_\text{new}$（$\mathcal A=$`Aqk`），
产出两张表：

$$
d\mathcal A_{ij}=(\text{scale})\,(do_i\cdot v_{\text{new},j})\ \ (i\ge j),\qquad
dv_{\text{new}} = \mathcal A^{\top} do .
$$

源码：`b_A` 加载并掩下三角（`chunk_bwd.py:91-95`）；
`b_dA = tl.dot(b_do, b_v, b_dA)`（`:111`）累加 $do\,v^\top$；
`b_dv = tl.dot(b_A, b_do)`（`:113`）；最后 `b_dA = where(o_t[:,None]>=o_t, b_dA*scale, 0)` 落盘（`:118-119`）。

> **布局坑**：FLA 对 `Aqk` 用**转置存取**——`p_A = A + o_A[:,None] + o_t[None,:]*(HV*BT)`
> （`chunk_bwd.py:91`），即内存里按 `[within_chunk, token]`，与名字上的 $[i,j]$ 相反。
> 移植到 Megatron/CUDA 时必须先把这个 stride 约定对齐，否则 $dq$ 对、$dk$ 全错。

---

## 2. 状态递推的伴随：`chunk_gated_delta_rule_bwd_dhu`

`chunk_delta_h.py:751-822`，kernel `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`
（`chunk_delta_h.py:370-684`），grid `(cdiv(V,BV)*N*HV,)`。**沿 $T$ 逆序串行**，
状态伴随常驻寄存器 `b_dh1..b_dh4`（`chunk_delta_h.py:413-428`）。

对前向（[00](00-math-and-api.md) §4）

$$
v_\text{new}=U-W S_\text{in},\qquad
S_\text{out}=\operatorname{diag}(E_\text{tot})S_\text{in}+k^{\text{rest}\top}v_\text{new},
\qquad o_i=\text{scale}\,q^{\text{dec}}_i S_\text{in}+\dots
$$

令 $dh_c$ 为 $\partial L/\partial S_\text{in}^{(c)}$（即块首状态伴随），逆序递推：

$$
dv_\text{new}=k^{\text{rest}}\,dh_c+dv^{\text{(in)}}\quad(\text{chunk\_delta\_h.py:539-583}),
$$

$$
dh_{c-1}=E_\text{tot}\odot dh_c+\text{scale}\,q^{\text{dec}\top}do-W^{\top}dv_\text{new}
\quad(\text{chunk\_delta\_h.py:585-651}).
$$

逐项对应源码：

- 逆序循环 `for i_t in range(NT-1,-1,-1)`（`:489`），先**落盘** $dh_c$（`:493-523`），再更新到 $dh_{c-1}$。
- $E_\text{tot}\odot dh_c$：`:589-596`（`USE_GK`，`b_dh *= exp2(gk_last)`）。
- $dv_\text{new}=k^{\text{rest}}dh_c$：`:544-577`（`b_dv = dot(b_k, b_dh)`；$k$ 这里就是 `kg`）。
- `+dv`（来自 §1 的输出支路）：`:581`。
- $+\text{scale}\,q^{\text{dec}\top}do-W^\top dv_\text{new}$：`:598-600`
  （`b_dh1 += trans(dot(b_q,b_do))*scale - dot(b_w,b_dv)`）。
- $dh_0$ 落盘：`:653-684`。

> **坑（重要）**：$dh$ 的初值来自**终态梯度 `dht`**，不是初态 `h0`；`h0` 只在
> [00](00-math-and-api.md) 的 varlen/CP 场景经 `_pre_process` 进入（`chunk_bwd.py:555-573`）。
> 首版若误用 `h0`，指纹是「$dh_0$ 对、但仅最末块的 $dh/dv$ 全错」。

---

## 3. Q/K/G/B/A 伴随：`chunk_kda_bwd_wy_dqkg_fused`

`chunk_bwd.py:383-458`，kernel `chunk_kda_bwd_kernel_wy_dqkg_fused`
（`chunk_bwd.py:139-320`），grid `(NT, B*HV)`。这是反向最重的一支，把
$W=T(\beta\odot k^\text{dec})$、$U=T(\beta\odot v)$、$S_\text{out}=k^{\text{rest}\top}v_\text{new}$
三处的伴随一次算完，产出 `dq,dk,dv2,db,dg,dAkk`。

### 3.1 主循环（`i_k` 块 × `i_v` 块）

在 $K$ 方向按 `BK`、$V$ 方向按 `BV` 分块（`chunk_bwd.py:223-241`），累加：

$$
dq\mathrel{+}=do\,S_\text{in}^{\top}\quad(:264),\qquad
dk\mathrel{+}=v_\text{new}\,dh^{\top}\quad(:265),\qquad
dw\mathrel{+}=dv\,S_\text{in}^{\top}\quad(:266),
$$

并直接累加 $g$ 对状态衰减的偏置 $\sum_v S\,dh$（`:263`）。当 `i_k==0` 时顺带算
$dA_{qk}=dv\,v^\top$ 与 $dv_2=\beta\,\mathcal A\,dv$、$db\mathrel{+}=\sum(\beta\mathcal A dv\odot v)$
（`:268-280`）。

### 3.2 乘衰减、折回 $L$

出循环后把 $(d q,d k,d w)$ 从「$E$ 域」折回原变量（`:282-301`）：

$$
dq\mathrel{*}=2^{g^{\text{cum}}}\,\text{scale},\qquad
dk\mathrel{*}=2^{\,g^{\text{cum}}_\text{tot}-g^{\text{cum}}},
$$

$$
dL\mathrel{+}=(-dw)\,(k\odot 2^{g^{\text{cum}}})^{\top},\quad
dA_{kk}\mathrel{+}=(-dw)\,(k\odot 2^{g^{\text{cum}}})^{\top}\ \text{的相关项},
$$

$$
dg=q\,dq-\underbrace{k\,dk}_{\text{写入门支路}}+\mathbb 1_\text{last}\,dg_\text{state}
+(k\odot 2^{g^{\text{cum}}})\,dA_{kk}\,\beta \quad(:300),
$$

$$
dk\mathrel{+}=dA_{kk}\,(\beta\,2^{g^{\text{cum}}})\quad(:301).
$$

源码逐行：`b_gk_exp=exp2(b_g)`（`:282`），`b_dw=-b_dw`（`:290`），
`b_dA=tl.dot(b_dw, tl.trans(b_kg))`（`:291`），`b_dkgb=tl.dot(b_A, b_dw)`（`:293`），
`b_dg=...`（`:300`），`b_dk=...`（`:301`）；`b_gb=exp2(b_g)*beta`（`:283`）。

### 3.3 求逆的伴随

$T=(I+L)^{-1}$ 的导数 $dT=-T\,dL\,T$，故（`chunk_bwd.py:310-314`）

$$
dL=-\big(\mathbb 1_{i>j}\odot dA_{kk}\odot\beta\big)\ \text{两侧夹 }T:\quad
dL=-T^{\top}(M)\,T^{\top},\ \ M=\mathbb 1_{i>j}\odot dA_{kk}\odot\beta .
$$

源码 `b_dA = where(m_A, b_dA*b_beta[None,:], 0)`（`:311`）、
两次 `tl.dot` 与取负（`:312-314`）。这里同样有 §1 的转置布局约定。

> `:267` 的 `tl.debug_barrier()` 标注「**DO NOT REMOVE**」——`b_A` 在多个 `i_k`
> 迭代间被复用为只读，编译器可能把它与其他缓冲区做别名优化，需要显式屏障。
> 这是移植时最容易漏的一处（漏了会出现随机错误）。

---

## 4. 块内反向：`chunk_kda_bwd_intra`

`chunk_intra.py:915-989`，kernel `chunk_kda_bwd_kernel_intra`
（`chunk_intra.py:400-677`），grid `(NK*NC*NT, B*HV)`（`chunk_intra.py:954`）。
它把 $dA_{qk},dA_{kk}$ 反传回 $dq,dk,db,dg$，覆盖对角与非对角**所有**块内对。

非对角（$i>j$，`:487-506`）：

$$
dq_i\mathrel{+}=\sum_j dA_{qk,ij}\,(k_j\odot 2^{g_i-g_j}),\qquad
dk_i\mathrel{+}=\sum_j dA_{kk,ij}\,(k_j\odot 2^{g_i-g_j}),
$$

再整体乘 $2^{\,g_i-g_n}$（`:504-506`）。对角用逐 $j$ 的循环（`:544-556`）；
反向支路 `b_dkt`（当前 token 作为**被乘的 key**）在 `:574-661` 累加：

$$
dk_j\mathrel{+}=\sum_i dA_{qk,ij}\,(q_i\odot 2^{g_j-g_i})
+dA_{kk,ij}\,(\beta_i k_i\odot 2^{g_j-g_i}).
$$

最后组装（`:561-562, 667-677`）：

$$
db=\sum_i dk_2\odot k,\qquad
dg=q\,dq_2+(dk_2-dkt)\odot k,\qquad
dk=dk_2+dkt .
$$

`safe_gate=True` 时走 `:519-543` 与 `:617-647` 的 TensorCore 分支（16×16 用
`tl.dot` 展开对角），否则走标量循环。**注意 `:607` 的注释「important to not
use bf16 here」**：块内反向的衰减链在 bf16 下会明显掉精度。

> `dA_{qk}/dA_{kk}` 的**行下标是块内局部量**（`i_i*BC+o_i`），不是绝对 token
> （`chunk_intra.py:590-591`）。误用绝对下标会「只有第 0 块对、其余全错」。

---

## 5. 门控反向与反向前缀和

`chunk_bwd.py:637-652`：

1. 先对 `dg` 做**反向前缀和** `chunk_local_cumsum(dg, reverse=True)`（`:637-644`）——
   因为前向的 `g_cumsum` 是块内前缀和，其伴随是后缀和。
2. `use_gate_in_kernel=True` 时调 `kda_gate_bwd`（`gate.py:284-322`，kernel
   `kda_gate_bwd_kernel` `gate.py:175-248`），把对**激活后** $g$ 的梯度折回
   `g_raw, A_log, dt_bias`：

   $$
   dg^{\text{raw}}=-e^{A_\text{log}}\,\sigma(z)\odot dy_g,\qquad
   dA=\sum_{t,k}-e^{A_\text{log}}\operatorname{softplus}(z)\odot dy_g\ \ (z=g^{\text{raw}}+\text{dt\_bias}),
   $$

   $$
   d\text{bias}=\sum_t dg^{\text{raw}},\qquad dA\ \text{按 }(t,h)\ \text{落盘后求和}.
   $$

   源码：`b_dg = b_A*(b_dyg*sigmoid(b_g))`（`gate.py:221`，普通门），
   `b_dA=sum(sum(b_dyg*b_yg,1),0)`（`:222`）；下界门在 `:224-232`。
   返回后 `dA`/`dbias` 在 host 端 `.sum(0)`（`gate.py:319-320`）。

---

## 6. 参考实现对照

`ChunkKDAFunction.backward`（`chunk.py:162-211`）把 `chunk_kda_bwd` 的返回值按
输入顺序排好（`chunk.py:209-211`），非叶子参数（`scale`、flags 等）位置填 `None`。
`../src/kda_ref.py` 没有手写反向，而是用 **autograd 穿过 §4 的 chunked 前向**
当作反向 golden，并与 autograd 穿过逐 token 递推对拍：六项梯度相对误差
$\le7.2\times10^{-15}$（`src/kda_ref.out.txt` Part B）。移植时可用它做逐 kernel 对拍的
最终裁判。

---

## 7. 小结与移植检查表

反向五段，块间只传 $D\times D$ 的状态伴随 $dh$：

$$
\underbrace{dA_{qk},dv}_{\text{输出支路}}\to
\underbrace{dh,dh_0,dv_2}_{\text{状态伴随}}\to
\underbrace{dq,dk,db,dg,dA_{kk}}_{\text{Q/K/G/B/A 伴随(最重)}}\to
\underbrace{dq,dk,db,dg}_{\text{块内}}\to
\underbrace{dg,dA,d\text{bias}}_{\text{门控}} .
$$

移植检查表：

- [ ] `Aqk/Akk` 的**转置存取**约定（§1、§3.3）。
- [ ] `dhu` 的 $dh$ 初值用 `dht` 而非 `h0`（§2）。
- [ ] `wy_dqkg` 里的 `tl.debug_barrier()`（§3.3）必须保留。
- [ ] 块内 $dA$ 的**行下标是块内局部量**（§4）。
- [ ] 门控反向的**反向前缀和**（§5），以及 `A_log/dt_bias` 保持 fp32。
- [ ] GVA 下 `dq/dk` 的 head 维归约（`chunk_bwd.py:631-634`）。

**下一篇** [03-optimization.md](03-optimization.md)：分块尺寸 / 重算策略 / tf32 与
数值稳定 / 变长与 CP / graph 等优化手段。
