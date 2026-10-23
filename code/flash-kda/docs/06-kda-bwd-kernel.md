# KDA 反向 kernel：FLA Triton 分解精读与目标卡 CUDA 反向设计

> 这是 KDA 专题的第 6 篇。`05` 篇推清了分块反向的公式，本篇看**实现**：FLA `chunk_kda` 的
> 5 个 Triton 反向 kernel 各自回答什么问题、站在数据流哪一段；再给出复用 FlashKDA 前向
> workspace 的目标卡 CUDA 反向设计，以及已跑通的最小 kernel 与实测数字。

## 0. 先记住一个画面

把前向想成一条**从块 0 流向块 N 的产线**。

每个工位（一个 chunk）接过上游的半成品——那个 $D\times D$ 的状态 $S$——加工 16 个 token，
再把新状态交给下一站。

反向就是**质检员倒着走回原料端**。

他手里拿着成品端的抱怨：输出的梯度 $\bar o$、终态的梯度 $\bar S_T$。
每到一个工位，他要回答两件事：

1. 这一站该负多少责任？算出 $dq,dk,dv,d\beta,dg$；
2. 该把多少责任转给上游？算出并交出 $\bar S_{\text{in}}$。

好消息：工位旁留着一张**交接单**（前向 workspace），上面写着当时的
$q^{\text{dec}},k^{\text{dec}},k^{\text{rest}},E^{\text{tot}},P,M_{qk}$。
所以质检员**不必重做这一站的活**，照单子算伴随即可。

坏消息：有两样东西没进交接单——$\beta$ 和门的原始 logits。
得派一个轻量小工（B0）回原料端补算。

> **类比的边界**：质检员不是「重放」前向。梯度是函数的**线性化伴随**，走的是转置方向，
> 不是把前向再跑一遍；每个工位内部都是 GEMM，转置乘的顺序写反就是 $\mathcal{O}(1)$ 的错。
> 数值上还有 tf32 / bf16 底噪，这些后面逐条交代。

读完这一篇，你应该能回答：

1. FLA 的 5 个 Triton 反向 kernel 各自回答什么问题，站在数据流的哪一段？
2. 为什么目标卡反向可以「复用前向 workspace」而不重算？缺口是什么，代价多大？
3. 这条链的误差底噪由谁决定（tf32？bf16？），哪些环节必须保 fp32？

## 1. 为什么反向参照 FLA，而不是 FlashKDA

**FlashKDA 没有反向。** 它只提供前向（`csrc/smxx/fwd_kernel1.cuh` / `fwd_kernel2.cuh`），
仓库里没有 CUDA 反向。

`flash-linear-attention` 的 `fla.ops.kda.chunk_kda` 则自带一套完整的 Triton 反向。
更关键的是，它的文件头写着：

> `# Related files are modified and supported by the Moonshot AI Team`（`fla/ops/kda/chunk.py:8`）

也就是说，FLA 的 KDA 反向与前向出自同一算法谱系，是核对「目标卡反向该做什么」的最近参照。

所以路线是：**精读 FLA Triton 反向的 kernel 分解 → 对上 `05` §7 的公式 → 给出适配
FlashKDA 前向 workspace 的目标卡 CUDA 设计**。

### 1.1 记号对照：FLA 保存量 ↔ `05` §7 分块符号

FLA 的 `chunk_kda` 前向会保存一组中间量给反向用（`fla/ops/kda/chunk.py:103-107`）：
`q,k,v,g_cumsum,beta,Aqk,Akk,w,u,qg,kg,v_new,h`。

它们与 `05` §7.1 式 (7.1) 的分块符号（$q^{\text{dec}},k^{\text{dec}},k^{\text{inv}},k^{\text{rest}},L,M_{qk},P,U,S_{\text{in}}$）一一对应，但**命名不同**。先钉死，后面才读得懂：

| FLA 张量 | 形状 | `05` §7 符号 | 关系式 | 来源 |
|---|---|---|---|---|
| `qg`  | `[B,T,HV,K]` | $q^{\text{dec}}/\text{scale}=q\odot E$ | `qg = q * exp2(gk)` | `wy_fast.py:117-119` |
| `kg`  | `[B,T,HV,K]` | $k^{\text{rest}}$ | `kg = k * exp2(g_last - gk)` | `wy_fast.py:120-125` |
| `w`   | `[B,T,HV,K]` | $P\,\mathrm{diag}(\beta)\,k^{\text{dec}}$ | `w = A @ (beta*k*exp2(gk))` | `wy_fast.py:109-128` |
| `u`   | `[B,T,HV,V]` | $P\,\mathrm{diag}(\beta)\,v$ | `u = A @ (v*beta)` | `wy_fast.py:97-99` |
| `Akk` | `[B,T,HV,BT]` | $P=(I+L)^{-1}$ | `A = (I+L)^{-1}` | `chunk_intra.py:315-379` |
| `Aqk` | `[B,T,HV,BT]` | $M_{qk}=\operatorname{tril}(q^{\text{dec}}{k^{\text{inv}}}^\top)$ | 存**未 mask** 全阵，用时再 mask | `chunk_intra.py:224-246` |
| `v_new` | `[B,T,HV,V]` | $U=P\,u'$ | `v_new = u - w @ S` | `chunk_delta_h.py:227-232` |
| `h`   | `[B,NT,HV,K,V]` | $S_{\text{in}}$（每块起始态） | 每块循环前落盘 | `chunk_delta_h.py:168-198` |
| `g_cumsum` | `[B,T,HV,K]` | $g^{\text{cum}}$（log2 累计，**块内局部**） | `cumsum(g_fla)*RCP_LN2` | `chunk_fwd.py:58-64` |

`harness/check_bwd_kernel_map.py` Part A 用 fp32、$B{=}1,T{=}128,H{=}2,K{=}V{=}64,BT{=}32$
（q,k 已 L2 归一化）实测了这张表。

下面这段是原始输出，每一行是「一个 FLA 张量 vs 对应的 `05` 符号」：

```
  qg  vs qd/scale (=q*E)       max_abs=2.24e-08  max_rel=6.47e-08
  kg  vs kr (k_restored)       max_abs=2.09e-07  max_rel=4.79e-07
  tril(Aqk) vs Mqk             max_abs=2.77e-05  max_rel=8.51e-04
  Akk vs P                     max_abs=1.46e-04  max_rel=1.46e-04
  w   vs P*diag(beta)*kd       max_abs=1.22e-04  max_rel=7.06e-04
  v_new vs U (P*(w*beta))      max_abs=2.32e-03  max_rel=6.82e-04
  h[0] vs S_in[0]              max_abs=0.00e+00  max_rel=0.00e+00
  h[-1] vs S_in[-1]            max_abs=1.23e-03  max_rel=1.80e-03
```

读的时候注意两件事：

**(a)** `Akk` 就是 $P$（相对误差 `1.5e-4`），而不是 $P\,\mathrm{diag}(\beta)$
（后者相对误差 ≈ 1）。

**(b)** `W` 方向的映射是 `Akk` 先乘 `beta` 再乘 $k^{\text{dec}}$，即
$w=P\,\mathrm{diag}(\beta)\,k^{\text{dec}}$。

量级也对得上：$10^{-8}$ 项是逐元素复算，$10^{-4}\!\sim\!10^{-3}$ 项来自 FLA kernel 内部
`tl.dot` 的 tf32 精度。

> 踩坑（值得记）：验证 $P$ 时若用 `torch.linalg.solve(I+L, I)` 的 fp32 稠密解，
> 在**未归一化**的随机 k 上 $L$ 元素可达 $\mathcal{O}(\sqrt{K})$，单位下三角阵的逆逐行指数增长，
> 稠密 LU 会把连第一行 `[1,0,...]` 都解成 6.99（残差 $10^5$）。
> 改用**前代（forward substitution）**即恢复。这正是 kernel 里不用 `cuSOLVER`、
> 而用 8×8 前代 + HMMA 合并的原因（`02` §8）。

## 2. 五个 kernel 在整体数据流里的位置

入口 `chunk_kda_bwd`（`fla/ops/kda/chunk_bwd.py:434-615`）把反向拆成 5 个 Triton kernel
加 2 个逐元素/门控步骤。顺序如下（行号为真实核对）：

| 步骤 | 代码位置 | 这个 kernel 回答什么问题 | 对应 `05` §7 公式 |
|---|---|---|---|
| 0 重算前向中间量 | `463-506` | 前向存的东西不够用，缺的怎么补？`kda_gate_chunk_cumsum` / `recompute_w_u_fwd` / `fwd_h` 重算 `w,u,qg,kg,v_new,h` | 复现式 (7.1) |
| 1 `dAv` | `510-520` | 输出支路该往回传什么？`dAqk=do@U^\top`，`dv=M_{qk}^\top do` | 式 (7.2) 输出支路 |
| 2 `dhu` | `541-555` | 状态这一路的反向怎么沿块逆序推？逆序扫块传 `dh`，并补 `dv` | 式 (7.13) 跨块传递 |
| 3 `wy_dqkg` | `557-574` | 本块的 $dq,dk,dv,db,dg,dAkk$ 各是多少？逐块由 `dh,h` 算 | 式 (7.2)~(7.7)、(7.9)~(7.10) |
| 4 `intra` | `576-591` | 块内那张 $A$ 的反向怎么回传到 $q,k,\beta,g$？ | 式 (7.6)、(7.8) |
| 5 反向 cumsum | `599-605` | 衰减链的梯度怎么反向前缀和？`chunk_local_cumsum(reverse=True)` | 式 (7.12) 反向前缀和 |
| 6 门控反向 | `606-613` | 门的 logits / `A_log` / `dt_bias` 梯度怎么算？`kda_gate_bwd` | §3 式 (5.10) |

**注意步骤 0**：默认 `disable_recompute=False`，反向**重算**前向的
`w,u,qg,kg,v_new,h`（`463-506`），而不是全部保存。

这是一个重要的工程取舍。§5 会把它和 FlashKDA 的 workspace 设计对照。

## 3. 逐 kernel 精读

每个小节先给一句话问题，再对公式、再给代码位置。

### 3.1 `chunk_kda_bwd_kernel_dAv`——输出支路（式 7.2）

**它回答：输出 $o=q^{\text{dec}}S_{\text{in}}+M_{qk}U$ 这一路的伴随，第一跳是谁？**

kernel 体在 `chunk_bwd.py:43-114`。grid `(NT, B*HV)`，每 program 处理一个 chunk。

- 载入 `Aqk`，**用到时才施加因果 mask**：`m_A = (o_t[:,None] <= o_t[None,:])`，
  `b_A = tl.where(m_A, b_A, 0)`（`89-90`）。

  也就是说，前向把 $M_{qk}$ 存成了**未 mask 的全阵**（Part A 已证 `Aqk[0,0,1]≠0`，
  `tril(Aqk)` 才等于 $M_{qk}$），反向在这里补 mask。

- 对 $V$ 分块累加（`93-109`）：
  - `b_dA += tl.dot(b_do, b_v)`（`106`）→ $\bar M_{qk}=\bar o\,U^\top$；
  - `b_dv = tl.dot(b_A, b_do)`（`108`）→ $\bar U_{(\text{out})}=M_{qk}^\top\bar o$。
- 写回时把 `dAqk` 乘上 `scale`，只保留严格上三角以外（`111-114`，`o_t[:,None] >= o_t`）。

这正是式 (7.2) 的伴随：输出支路给 $\bar M_{qk}$ 与 $\bar U$ 一个初值；
$\bar q^{\text{dec}}$ 与 $\bar S_{\text{in}}$ 那两片，留给后面的 `wy_dqkg` / `dhu`。

所以它只做「分叉」，不做「汇总」。

### 3.2 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`——状态伴随（式 7.13）

**它回答：状态沿块正向推进，反向怎么沿块逆序把 $\bar S$ 传回去？**

kernel 体在 `fla/ops/common/chunk_delta_h.py:359-671`，wrapper `735-794`。
grid `(cdiv(V,BV), N*HV)`——注意它在 **V 维分块**，每个 CTA 持有一段 $[K,\ BV]$ 的状态伴随
`b_dh1..b_dh4`（K 按 64 分四段，`400-415`）。

主循环是 `for i_t in range(NT-1, -1, -1)`（**逆序**，`476`）。每步做三件事：

1. **落盘本块状态伴随** `dh[tile]`（`480-510`，对应 FLA 保存的 `dh`，形状 `[B,NT,HV,K,V]`）。
2. **更新 `dv`**（`519-570`）：
   - `b_do = load(do)`（`523`），`b_k = load(k)`（`526-527`）；
   - `b_dv = tl.dot(b_k, b_dh)`（`532-534`）——这就是式 (7.3) 的
     $\bar U\mathrel{+}=k^{\text{rest}}\bar S_{\text{out}}$（`k` 在此与衰减相乘）；
   - `b_dv *= exp2(bg_last - b_g)`（`566-567`）把 `k` 补成 $k^{\text{rest}}$（块末相对衰减）；
   - `b_dv += load(dv)`（`568`）叠加步骤 1 的 $\bar U$，写回 `dv2`（`570`）。
3. **更新状态伴随 `dh`**（`571-638`）：
   - 先 `b_dh *= bg_last_exp`（`576-577`）——即式 (7.3) 的
     $\bar S_{\text{in}}\mathrel{+}=E^{\text{tot}}\odot\bar S_{\text{out}}$；
   - `b_q = b_q * exp2(b_g)`（`578`）把 `q` 补成 $q^{\text{dec}}$；
   - `b_dh += tl.dot(b_q, b_do) * scale - tl.dot(b_w, b_dv)`（`585-587`）——即式 (7.13)
     $$\bar S_{\text{out}}^{(c-1)}={q^{\text{dec}}}^\top\bar o+E^{\text{tot}}\odot\bar S_{\text{out}}^{(c)}-{k^{\text{dec}}}^\top\bar w .$$

这里 `w` 就是 $P\,\mathrm{diag}(\beta)k^{\text{dec}}$，$\bar w$ 把预测支路吃进来。

4. 循环结束把最终 `b_dh` 写进 `dh0`（`640-671`）→ $\partial L/\partial S_0$。

**结构要点**：`dhu` 只做「$D\times D$ 状态伴随沿块反传」，与 `05` §7.3 完全一致。
它按 V 维切分并行（每 CTA 一段 V），状态伴随逐块顺序，但**块内**只有矩阵乘。

### 3.3 `chunk_kda_bwd_kernel_wy_dqkg_fused`——逐块 dq/dk/dv/db/dg（式 7.2~7.10）

**它回答：拿到本块的 $S_{\text{in}}$ 和 $\bar S$ 后，$dq,dk,dv,d\beta,dg$ 各是多少？**

kernel 体在 `chunk_bwd.py:133-310`。grid `(NT, B*HV)`，每 program 一个 chunk，
在 K/V 两个方向分块累加（`213`、`231`）。

输入含 `h`（本块 $S_{\text{in}}$）与 `dh`（本块 $\bar S$，来自 3.2），
输出 `dq,dk,dv,dv2,dg,db,dA`。核心几行（真实核对）：

- `b_dgk += tl.sum(b_h * b_dh, axis=0)`（`253`）——逐 key 通道的 $\bar E^{\text{tot}}$ 贡献；
- `b_dq += tl.dot(b_do, b_h)`（`254`）→ $\bar q^{\text{dec}}=\bar o\,S_{\text{in}}^\top$（式 7.2）；
- `b_dk += tl.dot(b_v_new, b_dh)`（`255`）→ 状态项 $\bar k^{\text{rest}}$（式 7.3）；
- `b_dw += tl.dot(b_dv, b_h)`（`256`）→ $\bar k^{\text{dec}}$ 的预测支路（式 7.7）；
- `b_dA += tl.dot(b_dv, tl.trans(b_v))`（`264`）、`b_dA += tl.dot(b_dw, tl.trans(b_kg))`（`281`）
  → 两条 $\bar L$ 的块内 GEMM（式 7.6、7.8）；
- 衰减链：`b_dq *= exp2(b_gk)*scale`、`b_dk *= exp2(b_gn-b_g)`、逐 token 的
  `b_dg` 合成（`272-298`），对应式 (7.9)~(7.12) 里 $E/E^{\text{tot}}$ 的反向；
- 求逆的伴随：`b_dA = tl.dot(b_dA, b_A); b_dA = tl.dot(b_A, b_dA)`（`302-303`）
  再取负（`304`）——这就是式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$
  （对称写法，$P$ 缩略见下）。

> 精度注释：FLA 在 `302-303` 用 `b_A.to(b_A.dtype)` 做的两次 `tl.dot`。
> **`b_A.dtype` 就是 $P$ 被存成的 dtype**（那个 `.to` 是恒等 no-op；生产时 FLA 前向把
> `Akk` 开成 `k.dtype`=bf16，`chunk_intra.py:819`）。另一侧 `chunk_intra.py:597` 有一条
> 作者注释 `# (SY 09/17) important to not use bf16 here to have a good precision.`
> ——那条针对的是**块内衰减链**，不是这两条求逆伴随 GEMM。
> $P$ 降 bf16 的实际代价已由 §6.8 实测为 ≤1.3e-4，可忽略。

### 3.4 `chunk_kda_bwd_kernel_intra`——块内 A 的反向（式 7.6、7.8）

**它回答：块内那张 $A$（$\bar M_{qk}$、$\bar L$）怎么回传到块内的 $q,k,\beta,g$？**

kernel 体在 `fla/ops/kda/chunk_intra.py:395-667`，wrapper `897-962`。
grid `(NK*NC, NT, B*HV)`：K 维 × 子块维 × (chunk, batch·head)，即**最细粒度**地并行
（`BC=16` 子块）。

它接收 `dAqk,dAkk`，回传 `dq,dk,db,dg`。

前向对应物是 `chunk_kda_fwd_kernel_inter_solve_fused`（`chunk_intra.py:43-379`）。
它把**块间（inter）$A_{qk}/A_{kk}$ + 块内 8×8 前代 + Schur 合并求逆**融成一个 kernel
（`265-379`）——这与 FlashKDA K1 的「8×8 前代 + 16×16 HMMA 合并」（`02` §8）是同一套代数。

反向里 `b_dq2/b_dk2` 由对 `kgi` 的循环累加（`477-496`、`571-600`），`b_db/b_dg` 由
`b_dk2*b_k` 等合成（`551-552`、`658-667`）。

特别地，SAFE_GATE 分支（`509-532`、`607-637`）用子块中点 `min(BC//2, ...)` 做数值稳定的
`exp2(g - g_n)` 归一——和 FlashKDA 门控 `lower_bound·log2e`（`02` §5）是同一防溢出思路。

### 3.5 门控链（式 5.10）

**它回答：衰减与门的梯度，最后怎么落到 logits、`A_log`、`dt_bias`？**

`dg` 得到后先做**反向前缀和**：`chunk_local_cumsum(dg, reverse=True)`（`chunk_bwd.py:599-605`），
对应式 (7.12) 的 $\bar g_i=\sum_{j\ge i}\hat g^{\text{cum}}_j$。

随后 `kda_gate_bwd`（`606-613`）完成式 (5.10) 的
$\bar g^{\text{logit}}/\overline{A\_log}/\overline{\text{dt\_bias}}$。

### 3.6 kernel 分解的实测证据

**它回答：这五个 kernel 在真实 run 里真的各跑一遍吗？各占多少时间？**

`harness/check_bwd_kernel_map.py` Part B 用 `torch.profiler` 抓了一次真实
$B{=}1,T{=}512,H{=}4,K{=}V{=}128$、chunk 64、bf16 的 `chunk_kda` 前向+反向，
CUDA kernel 级统计如下：

```
    1x  chunk_kda_fwd_kernel_inter_solve_fused       (0.028 ms)
    2x  chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (0.027 ms)
    2x  recompute_w_u_fwd_kda_kernel                 (0.013 ms)
    1x  chunk_kda_fwd_kernel_intra_token_parallel    (0.008 ms)
    1x  chunk_gla_fwd_kernel_o                       (0.004 ms)
    2x  chunk_local_cumsum_vector_kernel             (0.005 ms)
    ---- 反向 ----
    1x  chunk_kda_bwd_kernel_dAv                     (0.004 ms)
    1x  chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 (0.014 ms)
    1x  chunk_kda_bwd_kernel_intra                   (0.017 ms)
```

读的时候注意三点：

- `recompute_w_u_fwd_kda_kernel` 出现 2 次：前向一次、反向重算一次；
- `wy_dqkg` 因该 run 的 `tl.dot` 归并进相同名字统计而未单列，属正常；
- 反向 3 个专用 kernel 的耗时量级与前向同阶，印证「反向是前向同规模的一遍线性化」。

## 4. 与前向 K1/K2 的对称性

把 `03` 的前向结构和本节反向结构并排看，是一个很清晰的「转置 / 反序」关系：

| 前向（FlashKDA） | 反向（对应量） | 关系 |
|---|---|---|
| K1 逐 tile 算 $L,M_{qk},P,k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},E^{\text{tot}}$ | `intra` 由 $\bar L,\bar M_{qk}$ 回传 $q,k,\beta,g$ | 式 (7.6)/(7.8)，**复用前向的 $L,P$** |
| K2 单 CTA 沿 tile **正序**递推 `s_acc` | `dhu` 沿 tile **逆序**传状态伴随 | 式 (7.13) 是状态递推的转置 |
| K2 状态更新 $s\mathrel{+}=k^{\text{rest}\top}U$ | `dhu` 的 $\bar U\mathrel{+}=k^{\text{rest}}\bar S$、$\bar k^{\text{rest}}=U\bar S^\top$ | 式 (7.3) |
| K2 输出 $o=q^{\text{dec}}S+M_{qk}U$ | `dAv`/`wy_dqkg` 的 $\bar q^{\text{dec}},\bar M_{qk},\bar U$ | 式 (7.2) |

一句话：**前向把 $S$ 沿块正向推进，反向把 $\bar S$ 沿块反向推进；块内都是 GEMM，
块间只传 $D\times D$。**

这也是「块内并行、块间串行」在两侧都成立的原因。

## 5. 目标卡 CUDA 反向设计（自研）

FlashKDA 只开源前向。下面是本专题给出的、**复用前向 K1 workspace** 的 CUDA 反向方案。

### 5.1 前向已经落盘了什么（可直接复用）

K1 的 workspace 每 tile 六字段（`csrc/smxx/utils.cuh:70-76`，总量 13824 B）：

| 字段 | 字节 | 反向里对应 |
|---|---|---|
| `kKDecayed` ($k^{\text{dec}}$) | 4096 | 式 (7.6)/(7.7) 的 $k^{\text{dec}}$ |
| `kQDecayed` ($q^{\text{dec}}$) | 4096 | 式 (7.2) 的 $q^{\text{dec}}$ |
| `kKRestored` ($k^{\text{rest}}$) | 4096 | 式 (7.3) 的 $k^{\text{rest}}$ |
| `kGTotal` ($E^{\text{tot}}$) | 512 | 式 (7.3)/(7.11) 的 $E^{\text{tot}}$ |
| `kINV` ($P$) | 512 | 式 (7.5) 的 $P$（**不必再求逆**） |
| `kMqk` ($M_{qk}$) | 512 | 式 (7.2) 的 $M_{qk}$ |

外加分配时的 `ceil(T/16)+N` 上界与 `ws_tile_prefix`（`flash_kda.cpp:5-30`，`02`/`03` 已讲）。

前向 K1 把这些全部落盘（`fwd_kernel1.cuh:516-580`），所以反向**不需要**像 FLA 那样重算
`qg,kg,w,u,v_new,h`——只要 $k^{\text{inv}}$ 由 $k^{\text{rest}}/E^{\text{tot}}$ 现算即可。

**缺口只有一个：`beta` 与原始门 logits 没有落盘**（前向只在 K1 的 smem 里用过）。
因此反向需要一个轻量 **B0 重算 kernel**：读原始 `g`(logits)/`beta`/`A_log`/`dt_bias`，
重算 $\sigma(\beta)$ 与 $\alpha=2^{g}$、$g^{\text{cum}}$（后向也需要 $\beta$ 反传到 logits）。

这与 FLA 的 recompute 分支同构，但代价小得多：只重算门控，不重算 GEMM。

### 5.2 kernel 计划

沿用 K1/K2 的 CUTLASS + TMA + warp specialization 风格，反向拆 3 个 GEMM kernel + 1 个门控。

**B3 `bwd_intra`（对应 FLA `intra`，token-parallel，复用 K1 模板）**

- grid：每个前向 tile 一个 CTA（与 K1 相同，`H·total_tiles`）。
- 输入：workspace 的 $q^{\text{dec}},k^{\text{dec}},k^{\text{rest}},E^{\text{tot}},P,M_{qk}$
  加 B0 重算的 $\beta,g^{\text{cum}}$，加上游 $\bar o$、上一步得到的 $\bar U,\bar S$。
- 计算：按式 (7.2)/(7.6)~(7.10) 先算 $\bar M_{qk}=\bar o\,U^\top$、$\bar U$ 各支路，再用
  **已落盘的 $P$** 做式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$（3 次 16×16 GEMM），最后
  式 (7.6)/(7.8) 两个块内 GEMM 回传 $k^{\text{dec}},k^{\text{inv}},q^{\text{dec}}$。
- `CHUNK=16` 下块内矩阵极小，`intra` 可整体放进寄存器 + HMMA，几乎不耗 smem。
- 输出：`dq,dk,dv` 的块内部分 + `dbeta,dg` 的块内部分 + `dA`（给 K1-style 合并）。

**B2 `bwd_dqkg`（对应 FLA `wy_dqkg`，per-tile，需要 $\bar S$）**

- 由 B1 得到的每 tile $\bar S$（$\bar S_{\text{out}}$）和 B0 的 $E^{\text{tot}}$ 做：
  - $\bar U\mathrel{+}=k^{\text{rest}}\bar S_{\text{out}}$，$\bar k^{\text{rest}}=U\bar S_{\text{out}}^\top$；
  - $\bar E^{\text{tot}}=\sum_v\bar S_{\text{out}}\odot S_{\text{in}}$，$\bar S_{\text{in}}\mathrel{+}=E^{\text{tot}}\odot\bar S_{\text{out}}$；
  - 预测支路 $\bar w$、$\bar k^{\text{dec}}$、$\bar S_{\text{in}}\mathrel{+}=-{k^{\text{dec}}}^\top\bar w$。
- 输出 `dq,dk,dv,dbeta,dg`。

**B1 `bwd_state_adjoint`（对应 FLA `dhu`，head-parallel，是 K2 的镜像）**

- grid：每 head 一个 CTA（与 K2 相同），沿 tile **逆序**扫，维护 $\bar S$（$D\times D$，
  fp32；对上前向 K2 的 `s_acc`）。
- 每步按式 (7.13)：
  $$\bar S_{\text{out}}^{(c-1)}={q^{\text{dec}}}^\top\bar o+E^{\text{tot}}\odot\bar S_{\text{out}}^{(c)}-{k^{\text{dec}}}^\top\bar w,$$
  并把 $\bar S_{\text{out}}^{(c)}$ 落盘给 B2（bf16，$D\times D$/tile，量级同前向 `h`）。
- 终块初值 $\bar S_T^{\text{out}}$ 来自 `final_state` 上游（可为 0）。
- 数据流可**整体套用 K2 的八个 Phase 与 union/流水线**（`03`），只是把「正序累加」换成
  「逆序累加」，`MOVM_T` 的寄存器转置照用。

**B0 `gate_bwd`（门控链，逐元素 + 反向前缀和）**

- 反向前缀和用 K1 的列并行前缀和骨架反向跑一遍（式 7.12）；
- 门激活链式 (5.10) 与 `kda_gate` 一一对应，可与 B2/B3 融合或在单独小 kernel 里做。

### 5.3 显存与精度取舍

- **状态伴随 $\bar S$**：需要每 tile 一份 $D\times D$ 给 B2。参考 FLA，用 **bf16** 存即可
  （前向 K2 已验证有衰减 regime 下 bf16 状态存储对输出无可测量损失，`03` §18；
  反向的 $\bar S$ 与状态同阶，可用同一结论，**待验证**：需实测 bf16 $\bar S$ 对
  $dk,d\beta$ 的误差。这条在 §6.2 已关闭：~2.5e-3，可行）。
- **$P$ 的精度**：式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$ 对 $P$ 敏感。FLA 在 `302-303`
  的两次 `tl.dot` 精度由**上游把 $P$ 存成什么 dtype**决定（那个 `.to(b_A.dtype)` 是 no-op；
  生产时 FLA `Akk` 本就是 bf16，见 §6.8）。**已由 §6.8 实测关闭**：目标卡 K1 的 bf16 `kINV`
  足够——$P$ 降 bf16 对六项梯度只加 **≤1.3e-4**（比 bf16 操作数底噪低 2–3 个数量级），
  **无须增存 fp32 $P$**。真正要保 fp32/tf32 累加的是**块内衰减链**那一侧
  （`chunk_intra.py:597` 的注释）。
- **不复算 GEMM**：与 FLA 默认「重算 `w,u,qg,kg,v_new,h`」不同，目标卡方案直接吃 K1
  workspace，反向的额外开销只剩状态伴随 + 块内 GEMM。这正是 K1「把重活落盘」设计的回报。

### 5.4 与 FLA 的差异汇总

| 维度 | FLA Triton | 目标卡 CUDA（本设计） |
|---|---|---|
| chunk | 64（BC=16 子块） | 16 |
| 前向中间量 | 默认**重算** | K1 workspace **已落盘**，直接复用 |
| 状态伴随 | `dhu`，V 维分块、Triton | B1，head 并行、CUTLASS+TMA，镜像 K2 |
| 块内 A 反向 | `wy_dqkg` + `intra` 两个 kernel | B2+B3，`CHUNK=16` 下可合并 |
| 求逆伴随 | 用保存的 bf16 `Akk` | 用 workspace 的 bf16 `kINV`（§6.8 实测：bf16 够用，**无须** fp32 副本） |
| 精度敏感点 | `chunk_intra.py:597` 注释禁 bf16 | 保持 fp32/tf32 |

## 6. 把设计落成最小 kernel：七轮实测

§5 是纸面设计。这一节是把它一段段写成**能跑的最小 Triton kernel**，再逐段对拍。

为什么值得这么干：反向的错法太多（转置写反、局部/绝对下标混淆、单位换算差一个
$\log_2 e$），光看代码看不出来。最小 kernel 能把每一段的契约先钉死。

先给一张总表，后面按顺序展开：

| 段 | 对应 FLA kernel | 最小 kernel | 回答什么 |
|---|---|---|---|
| B1 | `dhu` | `harness/check_bwd_dhu_kernel.py` | 跨块状态伴随怎么逆序传 |
| B2 | `wy_dqkg` | `harness/check_bwd_dqkg_kernel.py` | 逐块 $dq/dk/dv/d\beta/dg/dA$ |
| B3 | `intra` | `harness/check_bwd_intra_kernel.py` | 块内 $A$ 的四个阶段回传 |
| B0 | 门控 | `harness/check_bwd_b0_gate.py` | 门控正向重算 + 反向链 |
| dAv | `dAv` | `harness/check_bwd_e2e.py` | 输出支路的两个伴随 |
| 组装 | `chunk_kda_bwd` | `harness/check_bwd_e2e.py` | 五段串起来对不对 |
| 真 workspace | — | `harness/check_bwd_ws_reuse.py` | 不借 FLA 中间量还能不能跑 |

### 6.1 B1 状态伴随：契约与三方对拍

**它回答：§5.2 的 B1 是整条反向里唯一跨块串行的部分，它的契约到底怎么写？**

先看契约。对一个 head、沿 block 逆序扫，维护状态伴随 `dS[K,V]`（fp32）。每块 `c`：

```
dh[c] = dS                                  # 落盘给 B2（= 式 7.11 的状态伴随）
dU    = k_rest @ dS + dU_in                  # = FLA dv2（输出支路 dU_in 来自 dAv）
dS   <- E_tot[:,None] ⊙ dS                  # 式 (7.13) 的 E^tot 支（`dhu:577`）
        + q_dec^T @ do · scale               # 输出支路（`dhu:585`）
        - w^T @ dU                           # 预测支路（w = P diag(β) k_dec，`dhu:587`）
dh0   = 完成 c=0 后的 dS                     # = ∂L/∂S_0
```

读的时候注意：这四行与 `dhu`（`fla/ops/common/chunk_delta_h.py:576-587`）**逐行同构**。
FLA 只是把 `q_dec = qg` 与 `k_rest = kg` 通过 `g`/`gk` 的 `exp2` 缩放现算，
并把 K 拆成 64、V 按 `BV` 分块。

本 kernel 用 `tl.dot` 直实现：`tl.dot(k_rest, dS)`、`tl.dot(tl.trans(q_dec), do)`、
`tl.dot(tl.trans(w), dU)`，grid `(cdiv(V,BV), B·HV)`（head 并行、V 分块，镜像 §5.2 的 B1）。

**踩坑（坐标约定）。**

试了什么：把初态 `h0` 当初始 `dS`。
看到什么：`dh0` 逐位对（1e-3），但**最后一个块**的 `dh`/`dv2` 全错，其余块全对。
结论：初始 `dS` 应来自**终态梯度 `dht`**（无上游时为零），`h0` 只进 B2 的 `detot`。
只有最末块的 `dS` 直接由初值决定，所以错得那么有规律。改成从 0 起即归位。

**三方对拍（fp32）。** `A` 是纯 torch 显式递推（`dhu_torch`，ieee fp32 matmul），
`B` 是 Triton kernel，`C` 是 FLA `chunk_gated_delta_rule_bwd_dhu`（真跑）。
两组配置，`do` 与 `dU_in` 取自 FLA `dAv` 的真实输出：

| 配置 | 对比 | `dh` max_rel | `dv2` max_rel | `dh0` max_rel |
|---|---|---|---|---|
| B1 T256 H2 D128 BT64 | torch vs FLA | 1.12e-3 | 8.68e-4 | 1.34e-3 |
| | **Triton vs FLA** | **4.60e-5** | **2.07e-5** | **1.82e-5** |
| | Triton vs torch | 1.13e-3 | — | — |
| B1 T512 H4 D128 BT32 | torch vs FLA | 1.16e-2 | 5.23e-3 | 6.73e-3 |
| | **Triton vs FLA** | **3.71e-5** | **3.88e-5** | **8.00e-6** |
| | Triton vs torch | 1.16e-2 | — | — |

**为什么 Triton 反而比 torch 更贴 FLA？**

FLA `dhu` 的 `tl.dot` 默认走 **tf32**，torch 显式递推走 ieee fp32，两者差 ~1e-3
（BT32 更长的块链放大到 ~1e-2）；而 Triton kernel 同样走 tf32、还原序也一致，
于是 **Triton↔FLA 落到 1e-5**。

这本身是一条结论：**`dhu`/B1 的误差底噪由 tf32 决定（~1e-3），不是 fp32 机器精度**。
目标卡 CUDA 若用 tf32 MMA，需接受同量级底噪；若需更准，应保留 fp32 累加
（对照 §3.3 的 $P$ 伴随精度注释）。

**与参考实现对齐。** 把同一份 `q,k,v,g,beta,do` 喂给
`src/kda_bwd_ref.py::kda_chunked_backward`（head 0），其 `dh0` 与 B1 的 `dh0` 差
**max_rel 3.3e-4（BT64）/ 5.2e-3（BT32）**，与上面 torch-vs-FLA 同量级
（对到 **~3e-4**）。这说明 B1 递推就是 `05` §7.3 那条跨块状态伴随链，没有引入额外近似。

### 6.2 B1 的两个边界：bf16 存储与非整除 T

**它回答两个问题：状态伴随能不能存 bf16？T 不能整除块长时还对不对？**

**（一）bf16 消融。** §5.3 标了「bf16 存 $\bar S$ 对 $dk,d\beta$ 的误差待验证」。
B1 里做两个受控消融（fp32 golden）：

- **(i) bf16 carry**（最坏情形：每块把运行中的 `dS` 量化成 bf16 再续传；FLA 实际保留 fp32）：
  `dh` max_rel **2.72e-3 / 2.53e-3**（两组配置），`dh0` 1.83e-3 / 3.08e-3。
- **(ii) bf16 storage**（FLA 的真实做法：`dS` 全程 fp32，只把**落盘的每块 `dh`** 存 bf16）：
  `dh` max_rel **2.72e-3 / 2.53e-3**（与 (i) 相同，因为落盘量化与逐步量化同阶）。

因为 B2（`wy_dqkg`）对 `dh` 是**线性**的（`b_dk += dot(v_new, dh)`、`b_dg += sum(h*dh)` 等，
§3.3），这个 **~2.5e-3 的相对误差会原样传给 `dk/dβ/dA`**。

结论：**bf16 存状态伴随可行**，误差落在 bf16 GEMM 底噪（前向 `o`/`S` 已 3e-3~1e-2）之内，
与 §5.3 的「同阶」判断一致。但若追求反向精度，`dh` 用 fp32 落盘（每 tile 多 64 KB @D128）
是廉价保险。

Triton 的 bf16-carry 变体与 torch bf16-carry 交叉验证到 5.9e-3 / 1.2e-2（含 tf32），自洽。

**（二）非整除 T。** 试了什么：直接跑 $T=250$、$BT=64$。
看到什么：wrapper 的 `.view(B, NT, BT, H, K)` 直接抛
`RuntimeError: shape '[1,4,64,2,128]' is invalid for input of size 64000`（尾巴 58 行）。

结论：**kernel 本体本来就是 T-general，只有 host 侧 `E_tot` 记账假设了整除**——
这是一处真实的、此前未被覆盖的边界 bug。

根因是尾块的 **$E^{\text{tot}}$ 根本没被寻址过**。而 FLA 的前向量是**不给 pad** 的
（`g_cum` 形状 $[B,T,H,K]$，$T=250$ 实测 size $=250\cdot H\cdot K$），
FLA 自己的 `dhu` 靠 mask 处理非整除 T，是合法参照。

修法：$E^{\text{tot}}_c = 2^{g\_cum[\min((c{+}1)BT,T)-1]}$——按块的**最后一个有效 token** 取
（`g_cum` 是块内局部 cumsum，最后有效行天然等于该块有效行的门控和），尾块 pad 行直接丢弃。
kernel 侧不用改：`m_t = o_t < T` 已把每个 token load 置零，pad 行对 `tl.dot` 贡献为 0。

实测（fp32，Triton↔FLA `max_rel`）：

| config | NT | tail | `dh` | `dv2` | `dh0` |
|---|---|---|---|---|---|
| T=256 BT=64（可整除，回归上一轮） | 4 | 0 | 4.60e-5 | 2.07e-5 | 1.82e-5 |
| T=250 BT=64 | 4 | 58 | 4.60e-5 | 2.07e-5 | 1.82e-5 |
| T=100 BT=64 | 2 | 36 | 9.12e-8 | 2.74e-5 | 7.17e-6 |
| T=250 BT=32 | 8 | 26 | 1.41e-5 | 3.78e-5 | 2.75e-5 |
| B2 T=130 H3 BT=64 | 3 | 2 | 8.56e-6 | 1.96e-5 | 1.46e-5 |
| T=513 BT=64 | 9 | 1 | 2.49e-5 | 2.48e-5 | 1.84e-5 |
| T=31 BT=32（单块） | 1 | 31 | 0 | **0** | 1.07e-7 |

三条读数：

- 可整除的 T=256 与上一轮数字**逐位相同**（4.60e-5/2.07e-5/1.82e-5）——修法是纯 no-op，
  不改变已有结论。
- 非整除各例 `dh/dv2/dh0` 都落在 **1e-8~4e-5**，与可整除同量级（tf32 底噪），
  说明尾块处理正确。
- 单块 `T=31`：`dh≡0`（$NT=1$ 时 `dh[0]=dS=0`，无跨块伴随），`dv2` **逐位等于** FLA
  （$dU=k^{rest}{\cdot}0+dv$，tf32 下 $0$ 加法精确）——这是尾块 mask 的最干净证据。

**关于「K 的 64 分段」。** 查 FLA `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 的
`grid=(cdiv(V,BV), N*HV)`（`fla/ops/common/chunk_delta_h.py` 的
`chunk_gated_delta_rule_bwd_dhu`），它**按 V 分块、K 不占 grid**，
`K=64/128/192/256` 是同 program 内 `o_k1..o_k4` 的**展开**（`K>64` 才多算一组），
属寄存器压力取舍而非正确性约束。

本 kernel 以 `K=128` 单块实现，与其数学等价，故「K 的 64 分段」不是待补的正确性缺口
（若做目标卡 CUDA B1，K 轴如何切是性能问题，见 §5.2 设计）。

**计时（朴素实现）**：`triton.testing.do_bench`（含 grid 启动）T256 H2 → **107 µs**，
T512 H4 → **191 µs**（`num_warps=4, num_stages=1, BV=32`）。

这是「一个 CTA 持整条 K 轴、V 维只分 32 一路」的朴素实现，远未优化
（FLA `dhu` 在 §3.6 的 profiler 里 ~14 µs）。它的价值是**先把契约和误差底噪钉死**，
为后续 CUTLASS/TMA 版 B1 提供可对拍的 golden。

局限：BT 需整除 `tl.arange` 且 `K=128` 固定（本 kernel 未做 K 的 64 分段，故不是 §5.2 的最终形态）。

### 6.3 B2 `wy_dqkg`：先钉转置约定，再逐位对拍

**它回答：逐块那六项梯度，怎么用一个 Triton kernel 全算出来？**

B2 的输入就是 B1 的输出加前向 workspace：`k,q,v,g(块内 log2 cumsum),beta,A(P),h(S_in)`、
`do`、`dh`（B1 落盘的每块 $\bar S$）、`dv`（`dAv` 的输出支路 $\bar U$）；
输出 `dq,dk,dv2,db,dg,dA`。与 `05` §7 公式一一对应：

| 代码（`chunk_bwd.py`） | 公式 |
|---|---|
| `b_dgk += tl.sum(b_h * b_dh, axis=0)`（`253`） | $\bar E^{\text{tot}}=\sum_v \bar S\odot S_{\text{in}}$ |
| `b_dq += tl.dot(b_do, b_h)`（`254`） | $\bar q^{\text{dec}}=\bar o\,S_{\text{in}}^\top$（式 7.2） |
| `b_dk += tl.dot(b_v_new, b_dh)`（`255`） | 状态项 $\bar k^{\text{rest}}=U\,\bar S$（式 7.3） |
| `b_dw += tl.dot(b_dv, b_h)`（`256`） | 预测支路 $\bar w$（式 7.7） |
| `b_dA += tl.dot(b_dv, tl.trans(b_v))`（`264`） | 块内 $\bar L$ 第一支 |
| `b_dw=-b_dw.to(A.dtype); b_dA+=tl.dot(b_dw, tl.trans(b_kg))`（`280-281`） | $\bar L$ 第二支（式 7.6/7.8） |
| `b_dA=dot(dA,A); b_dA=dot(A,b_dA); b_dA=-b_dA`（`302-304`） | $\bar L=-P^\top\bar P\,P^\top$（式 7.5） |

一个容易踩的细节：kernel 里**所有「矩阵」在载入寄存器时都被转置了**。

`b_A[i,j]=A^{\text{T}}[i,j]`（`p_A` 用 `o_A` 行 `o_t` 列，`chunk_bwd.py:205-208`）、
`b_h[v,k]=h[k,v]=S_{\text{in}}^\top`、`b_dh=dS^\top`。

原因是这些片段随后要当 `tl.dot` 的 B 操作数（列主序），等价于 host 侧的
`S.transpose(-1,-2)`。`harness` 里的 torch 参考（`wy_dqkg_torch`）用 `einsum`
把这些转置显式写出并与 kernel 对齐，A 张量按 `[B,T,H,BT]`（`A4[b,o_t,h,o_A]`）布局
——`b_A` 即其 chunk 内转置。

**三方对拍。** Part A 是纯 torch 逐块实现（ieee fp32），Part B 是本 kernel，参照是 FLA 的
Triton 内核（本机 FLA 默认会把 `@dispatch('kda')` 路由到 **TileLang** 后端，见 6.3 末尾；
这里用 `FLA_DISABLE_BACKEND_DISPATCH=1` 钉回文档所读的 Triton 实现）：

| config | 对比 | `dq` | `dk` | `dv` | `db` | `dg` | `dA` |
|---|---|---|---|---|---|---|---|
| B1 T256 H2 D64 | torch vs FLA（rel） | 8.3e-4 | 8.3e-4 | 6.7e-4 | 5.8e-4 | 8.0e-4 | 2.1e-3 |
| | **Triton vs FLA** | **0** | **0** | **0** | **0** | **0** | **0** |
| B2 T512 H4 D128 | torch vs FLA（rel） | 9.6e-4 | 8.4e-4 | 6.2e-4 | 7.0e-4 | 8.7e-4 | 1.9e-3 |
| | **Triton vs FLA** | **0** | **0** | **0** | **0** | **0** | **0** |
| B1 T1024 H8 D128 | torch vs FLA（rel） | 9.1e-4 | 9.8e-4 | 6.2e-4 | 5.7e-4 | 9.1e-4 | 1.7e-3 |
| | **Triton vs FLA** | **0** | **0** | **0** | **0** | **0** | **0** |
| B1 T250 H3 D64×128（非整除） | Triton vs FLA | 0 | 0 | 0 | 0 | 0 | 0 |
| B2 T513 H2 D64（非整除） | Triton vs FLA | 0 | 0 | 0 | 0 | 0 | 0 |

两条结论：

- **Part A 的 ~1e-3 就是 `tl.dot` 的 tf32 误差**：torch 参考走 ieee fp32、FLA 内核走 tf32，
  六项相对误差都落在 **5.8e-4~2.1e-3**，与 §6.1 里「`dhu` 的误差底噪是 tf32」完全同一量级。
  这说明 `wy_dqkg_torch` 与 kernel 的**代数完全一致**（若有某个转置写反，会出现 O(1) 的相对误差）。
- **Part B 逐位相同（max_abs=0）**：本 kernel 是 FLA `wy_dqkg` 的忠实移植（BK/BV 分块、
  `i_k==0` 分支、`tl.debug_barrier()`、求逆伴随的 bf16 cast 都照搬），因此 fp32 输入下与 FLA
  位级相同。这既验证了移植的正确性，也说明「B2 的算术路径 = FLA 的路径」。
  **非整除 T（尾块）也逐位相同**，即 `wy_dqkg` 是 per-chunk 独立的、不依赖 `T` 的分块
  （与 B1 的 host 侧非整除 bug 不同）。

**精度底噪：bf16 操作数。** 把全部输入量化到 bf16 再喂本 kernel（模拟目标卡 bf16 GEMM），
对 fp32 FLA 参照：

| config | `dq` | `dk` | `dv` | `db` | `dg` | `dA` |
|---|---|---|---|---|---|---|
| B1 T256 H2 D64 | 2.9e-3 | 5.0e-2 | 3.2e-3 | 2.5e-3 | 4.3e-2 | 5.5e-3 |
| B2 T512 H4 D128 | 2.8e-3 | 4.6e-2 | 3.6e-3 | 2.4e-3 | 3.3e-2 | 6.4e-3 |

`dq/dv/db/dA` 落在 **2.3e-3~6.4e-3**（bf16 GEMM 底噪），但 **`dk`/`dg` 明显更高
（3.3e-2~5.0e-2）**：因为 `dk` 的衰减链 $\exp_2(g^{\text{tot}}-g)$ 和 `dg` 的
`b_dg = q*dq - k*dk + ...` 都是**小量相减 / 指数放大**，对操作数量化更敏感。

这解释了为什么 FLA 在 `chunk_intra.py:597` 专门留了「这里不要用 bf16」的注释（§3.3）
——**`g`/`k` 相关通道应保 fp32/tf32**。目标卡 B2 的设计（§5.2）因此建议门控链与
`dk/dg` 的合成走 fp32。

**计时。** `triton.testing.do_bench`（naive，`num_warps=4, num_stages=1`，BK=BV=64）：
B1 T512 H4 D128 → **97.8 µs**（FLA 88.4 µs）；B2 T2048 H8 D128 → **496.8 µs**（FLA 452.8 µs）。

本 kernel 比 FLA 慢约 10%，符合「同结构、未调 autotune」的预期。它不是性能交付，
价值在把 **B2 的契约、转置约定、误差底噪**钉死，供目标卡 CUDA B2 对拍。

**踩坑：运行期 FLA 会路由到 TileLang 后端。**

本机 `fla` 0.5.x 里 `chunk_kda_bwd_wy_dqkg_fused` 被 `@dispatch('kda')` 装饰，
运行时会优先选中 **TileLang** 实现（`fla/ops/kda/backends/tilelang/chunk_bwd_dqkg.py`），
而它当前 **fp32 路径会编译失败**（无 fp32 MMA：`tl::mma_sync: unsupported configuration`）。

两条对策：

- 对拍时设 `FLA_DISABLE_BACKEND_DISPATCH=1`，回到文档所精读的 Triton 内核（本篇即如此）；
- 若必须用 TileLang 后端，用 bf16 输入（fp32 会撞上面的 unsupported 配置）。

这也提醒：**文档 `06` §3 精读的 Triton 代码，在运行期不一定是被执行的那一份**——
精读与实测要同时交代后端。

### 6.4 B3 `intra`：四个阶段与三个坑

**它回答：`dAqk`/`dAkk` 这两张块内矩阵，怎么回传到块内的 `dq,dk,dbeta,dg`？**

grid `(NK*NC, NT, B*HV)`（`chunk_intra.py:928`），`BC=16`、`BK=min(32,2^⌈log2 K⌉)`；
每个 program 处理**一个 query 子块**（`BC=16` 行）的一段 K。核心是四个阶段：

| 阶段 | 行号 | 作用 |
|---|---|---|
| A 前序整块 | `473-496` | $\bar q^{dec}{+}{=}\bar A_{qk}k^{inv}$、$\bar k^{dec}{+}{=}\bar A_L k^{inv}$（式 7.8） |
| B 对角块 | `534-546` | 同上，但对角子块按 `m_i=o_i≥j` 因果裁剪，逐 `j` 展开 |
| C 后继整块 | `571-599` | $\bar k^{inv}{+}{=}\bar A_{qk}^\top q^{dec}$（**转置读**，见下），再 $E$ 缩放 |
| D 对角块 | `639-660` | 同上，按 `m_i=o_i≤j` 反向裁剪 |

末尾合成（`662-667`）：`dg = \bar q^{dec}\odot q + (\beta\bar k^{dec}-\bar k^{inv})\odot k + dg_{in}`、
`dk = \beta\bar k^{dec}+\bar k^{inv}+dk_{in}`。

注意 `db` 由 `b_dk2*b_k` 行和得到（`551`），且这是**逐 token 标量在 `NK` 个 K-块间求和**
——FLA 用 `db2[NK,...].sum(0)`（`930/957`）避免写冲突。

**三个容易踩的点**（都靠对拍抓出来的）：

1. **漏了 `b_dg2 = b_q * b_dq2`（`chunk_intra.py:558`）。** 这是衰减链里
   $\bar g{+}{=}\bar q^{dec}\odot q$（式 7.10 的 $\bar E$ 项）。
   指纹非常干净：**`dq/dk/db` 全对、只有 `dg` 大面积错（`max_rel≈1.0`）**；补上后 `dg` 立刻归位。
   且它必须用**加回上游 `dq` 之前**的 `b_dq2`（`558` 在 `559` 之前），顺序写反会引入上游累积量的污染。
2. **Part C/D 对 `dAqk/dAkk` 是「转置读」。** A/B 段按下标 `dAqk[query, key_local]` 直读
   （`482`），而 C/D 段写成 `dAqk + (i_i*BC+o_i)[:,None] + o_j[None,:]*(HV*BT)`（`580`）——
   即行是**块内局部列**、列是**绝对 token**，等价于取 $\bar A_{qk}^\top$。
   这里的关键坑是：**行下标是 chunk-内局部量 `i_i*BC+o_i`，不是绝对量 `i_ti`**。
   第一版误用 `i_ti`（= `i_t*BT+i_i*BC`），结果**只有第 0 个 chunk 对、其余 chunk 全错**
   （`dk` 错到 $\mathcal{O}(1)$）——因为 `i_t=0` 时两者恰好相等。
   这再次说明「单 chunk 测试会漏跨块/绝对-局部混淆」的 bug。
3. **`db` 的跨 K-块归约。** 本 kernel 用 `tl.atomic_add` 落到一张 `db` 上（`intra_triton`），
   与 FLA 的 `[NK,...].sum(0)` 数学等价但**加法顺序不同**，故 `db` 不是逐位相同而是停在 ~1e-7（fp32 尾数）。

**三方对拍（fp32）。** Part A 是纯 torch 逐块实现（ieee fp32，`intra_torch`），
Part B 是本 kernel，参照 FLA 的 Triton 内核（`FLA_DISABLE_BACKEND_DISPATCH=1` 钉回，
同 §6.3）：

| config | 对比 | `dq` | `dk` | `db` | `dg` |
|---|---|---|---|---|---|
| B1 T256 H2 K64 | torch vs FLA（rel） | 5.5e-4 | 5.8e-4 | 6.0e-4 | 6.2e-4 |
| | **Triton vs FLA** | **0** | **0** | 1.3e-7 | 8.2e-8 |
| B2 T512 H4 K128 | torch vs FLA（rel） | 8.8e-4 | 7.0e-4 | 5.6e-4 | 6.2e-4 |
| | **Triton vs FLA** | **0** | **0** | 1.1e-7 | 1.1e-7 |
| B1 T1024 H8 K128 | torch vs FLA（rel） | 6.6e-4 | 7.8e-4 | 7.6e-4 | 6.9e-4 |
| | **Triton vs FLA** | **0** | **0** | 1.0e-7 | 6.3e-8 |
| B1 T64 H1 K32 | torch vs FLA（rel） | 5.3e-4 | 3.8e-4 | 3.0e-4 | 6.4e-4 |
| | **Triton vs FLA** | **0** | **0** | 3.9e-8 | 5.2e-8 |
| B1 T250 H3 K64（非整除）| **Triton vs FLA** | **0** | **0** | 1.1e-7 | 8.3e-8 |
| B2 T513 H2 K64（非整除）| **Triton vs FLA** | **0** | **0** | 7.1e-8 | 7.1e-8 |

两条结论：

- **Part A 的 ~5e-4~9e-4 就是 `tl.dot` 的 tf32 误差**（torch 走 ieee fp32、FLA 走 tf32），
  与 §6.1/§6.3 的底噪同量级，说明 `intra_torch` 与 kernel **代数完全一致**
  （任一转置/`m_i` 写反都会是 $\mathcal{O}(1)$ 误差）。
- **Part B 的 `dq/dk` 逐位相同（max_abs=0）**，`db/dg` 停在 ~1e-7；**非整除 T（250/513）也成立**，
  即 `intra` 与 `wy_dqkg` 一样是 **per-query-subblock 独立**的分块，不依赖 `T` 整除
  （与 B1 的 host 侧非整除 bug 不同，§6.2）。

至此三条 GEMM 支路（B1/B2/B3）全部有了位级可核验的最小 kernel。

**精度底噪：bf16 操作数。** 把全部输入量化到 bf16 再喂本 kernel（模拟目标卡 bf16 GEMM，
`tl.dot` 操作数 cast 到 bf16、累加 fp32），对 fp32 FLA 参照：

| config | `dq` | `dk` | `db` | `dg` |
|---|---|---|---|---|
| B1 T256 H2 K64 | 4.8e-2 | 3.0e-2 | 1.1e-2 | 2.0e-2 |
| B2 T512 H4 K128 | 2.8e-2 | 3.5e-2 | 2.1e-2 | 2.2e-2 |

全部落在 **1.1e-2~4.8e-2**——比 §6.3 的 B2（`dq/dv/db/dA` 2.3e-3~6.4e-3）高约一个量级，
因为 `intra` 的 A/B/C/D 四段全是**指数衰减因子直接乘在操作数上**（`exp2(g-g)`），
bf16 量化被指数放大。

这正是 `chunk_intra.py:597` 那条 `# important to not use bf16` 注释要防的。
**结论**：B3 必须保 fp32/tf32，目标卡的设计（§5.2）也据此把门控/衰减链留在 fp32。

**计时。** `triton.testing.do_bench`（naive，`num_warps=4, num_stages=1`，`BC=16`、`BK=32`）：
B1 T512 H4 K128 → **86.9 µs**（FLA 86.4 µs）；B2 T2048 H8 K128 → **591.1 µs**（FLA 592.0 µs）。

本 kernel 与 FLA 基本同速（差 <1%），符合「同结构」预期。价值在把 **B3 的契约、四个阶段、
转置/绝对-局部混淆的坑、bf16 误差底噪**钉死，供目标卡 CUDA B3 对拍。

### 6.5 B0 门控重算：把 K1 没落盘的门控补回来

**它回答：workspace 缺 `beta` 和门 logits，补算它们的最小 kernel 长什么样、准不准？**

前向 K1 的门控段（`fwd_kernel1.cuh:306-331`，128 个线程按 key 列并行）逐行是：

```
316  float sum = 0.0f;
321  g_val = bf16_to_f32(g_bf16_smem[row*D+col]) + dt;              // g + dt_bias
322  g_val = a_log_exp * g_val;                                      // * exp(A_log)   (a_log_exp @ :258)
323  g_val = gate_scale * sigmoid_tanh_approx_f32(g_val);            // * lower_bound   (gate_scale @ flash_kda.cpp:128)
327  sum += g_val;
328  g_smem[row*D+col] = sum;                                        // 块内前缀和（逐列）
330  shared_storage.g_total.begin()[col] = sum;                      // 块尾和
```

读的时候注意 `gate_scale = lower_bound * 1.4426950408889634`
（`flash_kda.cpp:128`，`1.4427 = log2(e)`），所以 `g_smem` 里存的累计值
**已经是 log2 单位**的 $g^{\text{cum}}$。

`g_total` 随后被 `ex2_approx_ftz_f32` 变成 $E^{\text{tot}}=2^{g^{\text{cum}}[\text{last}]}$
（`:356`）并写进 workspace（`:516` 的 `ws_idx`、`:550-558` 的 TMA store）。

`beta` 则根本没进 workspace——K1 只在算 $L$ 时临时 `sigmoid(beta)` 用了一下（`:502`）。

于是 B0 的**正向重算契约**（与 FLA `kda_gate_chunk_cumsum_vector_kernel`，
`fla/ops/kda/gate.py:375` 的 `:422/:425-431` 完全同构）就是：

$$
\text{inner}=e^{A_{\log}}\odot(g+\text{dt\_bias}),\quad
\text{gate}=\text{lower\_bound}\cdot\sigma(\text{inner}),\quad
g^{\text{cum}}=\operatorname{cumsum}_{\text{chunk}}(\text{gate})\cdot \log_2 e,\quad
\beta=\sigma(\beta_{\text{logits}}).
$$

一句话：**B0 不重算任何 GEMM，只重算 4 个逐元素算子 + 一个块内前缀和**——
这正是 K1「把重活落盘」的回报。

**前向重算结果。** 5 组 shape（含**非整除 T=250/513**、`g=0` 无衰减、B=2）实测：

| 量 | vs 纯 torch 参考 | vs FLA kernel |
|---|---|---|
| `g_cum` | max_rel **1.5e-7~3.9e-7** | max_rel **2.2e-7~2.8e-7** |
| `beta` | max_rel **6.3e-8~1.2e-7** | **逐位相同（max_abs=0）** |

即 $10^{-7}$ 量级 = fp32 机器精度：前向重算与 torch/FLA 对到 **~2e-7**，
B0 与公式、与 FLA 都一致，且**非整除 T 不破坏**
（B0 的 `m = o_t[:,None] < T` 把尾块 pad 行屏蔽，与 K1 的 `actual_len` 处理同理）。

**反向门控链。** 反向前缀和（B0 的 `APPLY_GATE=False, REVERSE=True`）vs FLA
`chunk_local_cumsum(reverse=True)` 实测 max_rel **1.2e-7~2.1e-7**。

门控反向 `dyg→(dg_logits, dA, dbias)` 则与 `torch.autograd`（对同一 `gate`/`g_cum` 公式求导）比：

| 量 | B0 vs autograd |
|---|---|
| `dg_logits` | max_rel **2.3e-7~3.9e-7** |
| `dA` | max_rel **2.3e-7~1.3e-6** |
| `dbias` | max_rel **1.2e-7~1.9e-7** |

反向门控链与 autograd 对到 **~3e-7**。

**踩坑（$\log_2 e$ 住在反向前缀和里）。**

试了什么：按 FLA `chunk_bwd.py:599` 那样用 `scale=1` 做反向前缀和。
看到什么：归一化误差稳定停在 **0.3069**。
结论：$g^{\text{cum}}=\log_2 e\cdot\operatorname{cumsum}(\text{gate})$，所以给定上游
$\bar g^{\text{cum}}=\partial L/\partial g^{\text{cum}}$，回传第一跳必须是
$\bar{\text{gate}}=\log_2 e\cdot\operatorname{cumsum}_{\text{reverse}}(\bar g^{\text{cum}})$，
**这个 $\log_2 e$ 要乘在反向前缀和上**，再进门控激活的链式。

而 $0.3069 = 1-1/\log_2 e = 1-\ln 2$，正好是一个缺掉的 $\log_2 e$ 因子。补上后立刻降到 $10^{-7}$。

**另一个坑（FLA 的门控反向输出是 bf16）。** FLA `kda_gate_bwd`（`gate.py:252`）末尾
`dg = dg.view_as(g).type_as(g)` 把 `dg` 转回**输入 dtype（bf16）**，于是它 vs fp32 autograd 的
`dg_logits` 出现 **~3e-3** 的相对差（= bf16 舍入 $2^{-9}$）。

把 autograd 也 round 到 bf16 后，残差掉到 **max 1.0e-4~5.6e-4、mean ~1e-7**
（剩个别大值来自 triton `exp/sigmoid` 的 fast-math）。

**B0 保 fp32 输出**，所以 B0↔autograd 是 1e-7——这解释了「B0 vs FLA 3e-3」并非代数差异
而是 **FLA 的 bf16 出口**，也提示目标卡反向应像 B2/B3 一样把门控链留在 fp32。

**交叉验证：对上 FlashKDA K1 当场算的 $E^{\text{tot}}$。**

**这是唯一能直接证明 B0 复现了前向 kernel 的地方。** 用 `flash_kda._fwd_raw` + 自备 workspace
调 K1，按 `fwd_launch.cu:64-71` 的 SoA 偏移读出 `kGTotal` 字段（head-major：
`ws_idx=head*total_tiles+tile`，`fwd_kernel1.cuh:516`），再和 B0 重算的
$\exp_2(g^{\text{cum}}[\text{last}])$ 比：

| config | `E_tot` B0 vs K1 workspace |
|---|---|
| B1 T128 H4 g=1.0 | max_rel **8.1e-5** |
| B1 T250 H2 g=1.0（非整除 T） | max_rel **2.0e-5** |
| B1 T512 H2 g=2.0 | max_rel **1.6e-5** |

残差 $10^{-5}$ 不是 bug：K1 用的是 **`sigmoid_tanh_approx_f32`**（`fwd_kernel1.cuh:323`，
`tanh.approx` 近似 sigmoid，`docs/02` §11(c) 实测 max_abs 3.9e-6），B0 用的是**精确 `tl.sigmoid`**。
把门控的单点近似沿 16 行累加、再被 `exp2` 放大，正好落在 $10^{-5}$——
**B0 复现了 K1 的门控到 approx 底噪**。

**计时。** event 对包 REPS=100 次取均值，含 host 派发：

| config | B0 | FLA `kda_gate_chunk_cumsum` | ratio |
|---|---|---|---|
| B1 T512 H4 D128 | 27.3 µs | 50.4 µs | 0.54× |
| B4 T2048 H8 D128 | 46.6 µs | 67.9 µs | 0.69× |
| B1 T2048 H64 D128 | 91.6 µs | 64.7 µs | 1.41× |

同量级、各有胜负（FLA 用 `BS` 切列 + autotune，B0 是整列 `[BT, D]` 单 program 的朴素实现，
`H=64` 时 grid 太小才被反超）。

这不是性能交付，价值在把 **B0 的契约（前向 $\beta/g^{\text{cum}}$、反向
$\bar g_{\text{logits}}/\bar A/\bar{\text{bias}}$）、$\log_2 e$ 的位置、FLA 的 bf16 出口陷阱、
以及「B0 与 K1 当场算的门控一致到 approx 底噪」**钉死，供目标卡 CUDA B0 对拍。

### 6.6 dAv 与端到端组装：五段串成一条链

**它回答：四段各自都对，接起来还对吗？反向 5 个 kernel 里最后没落地的 `dAv` 呢？**

前面几轮都是**逐段孤立**验证的：每段吃的中间量由 FLA 直接喂（例如 B2 的 `dh` 用 FLA 的
`dhu` 输出），**从没验证过「一段的输出接下一段的输入」这条链**。

而且反向 5 个 kernel 里还差一个 **B-dAv（输出支路）** 没落地
（`chunk_kda_bwd_kernel_dAv`，`chunk_bwd.py:44-114`，式 (7.2) 的
$\bar M_{qk}=\bar o\,U^\top$、$\bar U=M_{qk}^\top\bar o$）。

**dAv 最小 kernel 的契约。** `_dav_kernel`（grid `(NT, B·HV)`，非 varlen）逐式复刻 FLA：

- 载入 $M_{qk}$（FLA `Aqk`）：A 的布局是 `[B,T,HV,BT]`、末轴为 chunk-局部 key 下标，
  在 `[o_A,o_t]` 朝向里保 `o_t[:,None] <= o_t[None,:]`（上三角含对角，`chunk_bwd.py:83-90`）；
- $\bar M_{qk}\mathrel{+}=\bar o\,v_{\text{new}}^\top$（`b_v` 以 $v^\top$ 载入 `[BV,BT]`，
  `tl.dot(b_do[BT,BV], b_v[BV,BT])`），最后按 `o_t[:,None] >= o_t[None,:]` 保下三角含对角、乘 `scale`；
- $\bar U=M_{qk}\,\bar o$（`tl.dot(b_A, b_do)`）落盘为 `dv`。

**实测**：4 组配置（T=256/512/1024/250）下 `dAqk`、`dv` 与本 kernel 输出
**`max_abs=0`、`max_rel=0`（逐位相同）**——包括非整除 T=250。这与 §6.3/§6.4 里 B2/B3 的
「Triton↔FLA 逐位」同一性质：两者算术路径完全一致。

**端到端链条。** 输入构造沿用既有约定：`use_qk_l2norm_in_kernel/use_gate_in_kernel/
use_beta_sigmoid_in_kernel` 全 `False`，两侧吃**同一份 fp32 操作数**（归一化后 q/k、
`sigmoid(beta)`、自然对数衰减 `g_fla = g_dec·ln2`），损失 $L=\langle o,\bar o\rangle+\langle S_T,\bar S_T\rangle$，**因此 `dht=S_T` 的上游梯度非零**。

链条为：

```
dAv → B1(dhu) → B2(wy_dqkg) → B3(intra) → 反向 chunk-local cumsum(B0)
```

读的时候注意它和 `chunk_kda_bwd`（`chunk_bwd.py:434-615`）逐段同构：B2 的 `dv` 输入 =
B1 的 `dv2`；B2 的 `dA`（即 `dAkk`）与 dAv 的 `dA`（即 `dAqk`）一起喂 B3；
B3 的 `dq/dk/db/dg` 是**叠加**在 B2 之上（本实现分别跑后相加，与 kernel 内
`tl.load(p_dq)+...` / `atomic_add` 等价）。

**实测（链 vs `torch.autograd` 穿 `chunk_kda`，max_rel）**：

| config | dq | dk | dv | dg | dbeta | dh0 |
|---|---|---|---|---|---|---|
| B1 T256 H2 | **0** | 1.01e-5 | 1.19e-4 | 3.51e-6 | 3.07e-5 | 5.06e-6 |
| B2 T512 H4 | **0** | 1.51e-5 | 3.90e-4 | 1.10e-5 | 5.67e-5 | 1.84e-6 |
| B1 T1024 H8 | **0** | 8.12e-6 | 1.87e-4 | 1.64e-6 | 1.09e-4 | 7.58e-7 |
| B1 T250 H3（非整除） | **0** | 1.42e-5 | 2.72e-4 | 4.11e-6 | 6.38e-5 | 2.19e-6 |
| **worst** | **0** | 1.51e-5 | 3.90e-4 | 1.10e-5 | 1.09e-4 | 5.06e-6 |

六项全部落在 **tf32 底噪（`tl.dot`）**之内；`dq` 逐位相同（其两条支路都不经过 B1 的 `dh`）。
这就是把 §5.2 的四段**组装起来**后，对「真实前向中间量 + 真实上游梯度」的端到端核对。

**两条佐证。** 要把「链的残差」与「我们的 kernel 与 FLA 不一致」分开，脚本又加了两条对照：

- **Part C**：把 FLA 自己的子 kernel 按 `chunk_kda_bwd` 的顺序手动串起来，与 autograd 对拍——
  `dq/dk/dv/dg/db/dh0` **全部 `max_abs=0`（逐位相同）**。这证明「golden 就是 FLA 的分解」，
  Part B 的 $10^{-5}$ 残差不是 setup 偏差。
- **Part D（swap）**：把 FLA 的 B1 输出（`dh/dv2`）、FLA 的 dAv 输出（`dAqk`）、
  FLA 的 `dAkk` 喂给**我们的** B2/B3，再与 FLA 的 `wy_dqkg+intra` 比——`dk` 与反向 cumsum 后的
  `dg` 落到 **8.4e-8~1.2e-7**（数据集 4 组均 $<1.3\times10^{-7}$）。
  这说明 **B2/B3 在给定相同输入时与 FLA 代数完全一致**；Part B 的残差来自我们独立的 B1
  （tf32 累加序）经由衰减链放大，而非 kernel 语义差异。

  （例如 T=512 时 B1 的 `dh` 相对误差 9.9e-6，`dv2` 8.3e-5，落进 B2 的 `exp2` 衰减链后成为
  `dk` 1.5e-5、`dg` 1.1e-5。）

**两个必须记住的坑（都是这次组装抓出来的）。**

1. **B2 要吃的是 `v_new`（`fwd_h` 输出），不是 `u`（`fwd_intra` 输出）。**
   FLA 前向同时存了 `u` 与 `v_new`（`chunk.py` 的 `save_for_backward`），
   `chunk_kda_bwd` 传给 `wy_dqkg`/`dAv` 的是 **`v_new`**（$= u - w\,S$ 求解后的 $v$）。
   第一版误用 `u`，指纹是「**`dq` 逐位相同、`dk`/`dg` 大面积错（max_rel≈1e-2，且随 T 增大）**」
   ——因为 `dq=do\,S^\top` 不碰 `v_new`，而 `dk=v_{\text{new}}\,dh^\top`、`dg` 依赖 `dk`。
   修正后四项全落到 tf32 底噪。
2. **B0 反向前缀和不要额外乘 $\ln 2$。** FLA 的 `chunk_local_cumsum(..., scale=None)`
   **默认 `scale=RCP_LN2`**，故其 `reverse=True` 已把「对 log2 的 `g_cum` 的梯度」直接映回
   「对自然对数 `g` 输入的梯度」= `ga.grad`。
   第一版照搬上一轮的 `*LN2`（那里是为了换算到 fp64 参考的 log2 门控单位）而**多乘了一次**，
   指纹是「`dg` 与其余五项差一个**均匀常数因子**」（线性拟合 `gold/ours≈1.44=RCP_LN2`，
   `max_rel` 稳停 0.44）。
   这与 §6.5 的「$\log_2 e$ 必须落在反向前缀和上」是同一件事的两面：先确认**基准的 `g` 单位**，
   再决定是否补因子。

**小结。** 至此 §5.2 的四段 GEMM（dAv/B1/B2/B3）加 B0 尾接成**一条完整反向**，
用 `torch.autograd` 穿 `chunk_kda` 做端到端核对：六项梯度全部落 tf32 底噪（`dq` 逐位），
并用 Part C（FLA 自分解==autograd）与 Part D（swap 后 B2/B3==FLA 到 1e-7）把残差来源钉死到
「B1 独立 tf32 舍入」。

反向五段**不再只是各自最小可跑，而是能组装出一条与 FLA 数值一致的完整链**——
这是目标卡 CUDA 反向（§5.2）在落地前的最后一道组合正确性关卡。
B1 也顺带补上了 `dht`（终态上游梯度）入口（`check_bwd_dhu_kernel.py::_dhu_kernel` 的
`HAS_DHT` 分支），使其覆盖带 `final_state` 上游的损失。

### 6.7 架到真 workspace 上：不借 FLA 中间量

**它回答：前面所有对拍都吃 FLA 前向保存的中间量。换成 FlashKDA 自己的 workspace，整条链还成立吗？**

§5.1/§5.2 的整个设计前提是：目标卡反向**不重算前向**，而是**复用 K1 落盘的 workspace**
——六字段 `k_dec/q_dec/k_rest/E_tot/P/M_qk`（`csrc/smxx/utils.cuh:70–76`），只缺 `β` 与门控
logits（K1 只在 smem 里用过就丢，`fwd_kernel1.cuh:306–331`），补一个轻量 **B0**（§6.5）即可。

但前面几轮的对拍都**吃 FLA 前向保存的中间量**（`qg/kg/w/Akk/Aqk/v_new/h`），
§6.5 也只用 workspace 的 `kGTotal`（$E^{\text{tot}}$）做了点对点核验。
**没有人真正把整条反向架在 FlashKDA 自己的 workspace 上跑通。**

方法：跑 FlashKDA `fwd`（bf16，`CHUNK=16`，`D=128`，含初态/终态），把六个 SoA 字段按字节
偏移切出来。字段布局是 **head-major**：`n_ht = H·total_tiles`，
`ws_idx = head·total_tiles + tile`（`fwd_kernel1.cuh:516`），读出后 reshape 成
`[H, total_tiles, CHUNK, D]`。

有了 workspace 再加 B0 重算的 `β/g_cum`，就按 `kda_ref` 公式逐块重建：

- 状态递推（与 K2 同式）：$S_{c+1}=E^{\text{tot}}_c\odot S_c+k^{\text{rest}\top}_c U_c$；
- $U_c=P_c\big((v_c-k^{\text{dec}}_cS_c)\odot\beta_c\big)$，
  $w_c=P_c\,\text{diag}(\beta_c)\,k^{\text{dec}}_c$；
- 反向操作数：`qg=q_dec/scale`、`kg=k_rest`、`Akk=P`、`Aqk=Mqk`、`v_new=U`、`h=S_in`。

然后按 `CHUNK=16` 跑 `dAv→B1→B2→B3→反向 cumsum(B0)`。

**实测。** 六组 case（`T=128/256/250/512/513/192`、`H=2/3/4`、`B=1/2`、`g_scale=0/1/2`，
**含非整除 T**）。三部分：

| 部分 | 对照 | 结果 |
|---|---|---|
| **A** 用 workspace 重算前向 `o=q_dec S_in+M_qk U`、`S_out=E_tot S_in+k_rest^T U` | FlashKDA 自己的 `out`/`final_state` | `o` max_rel **4.5e-3~8.9e-3**、终态 **4.5e-3~8.0e-3** |
| **B** 七个反向操作数 | `kda_ref` 公式独立重算（fp32，CHUNK=16） | `qg/kg/w/Aqk/Akk/v_new/h` max_rel **1.3e-4~8.1e-3** |
| **C** 完整反向 `dAv→B1→B2→B3→B0` | `torch.autograd`（FLA，同激活操作数） | 六项梯度 max_rel 见下 |

六组 case 的**最坏** max_rel（workspace 驱动反向 vs FLA autograd）：

```
dq 3.42e-3   dk 2.98e-3   dv 6.34e-3   dbeta 5.97e-3   dg 7.05e-3   dh0 6.42e-3
```

全部落在 **bf16 输入舍入 / tf32 `tl.dot` 的底噪**，与上一轮（吃 FLA 中间量）同阶。
也就是说：**只要 workspace 六字段 + 一个 B0 重算，就足以驱动一条与 autograd 数值一致的
完整反向**——§5.2/§5.1 的「复用 K1 workspace」设计前提被端到端证实，而不只是逐字段点对点。

**四个只有「架到真 workspace 上」才会暴露的坑。**

1. **字段展平要按 `(NT, CHUNK, H)` 重排**：workspace 字段是 `[H, total_tiles, CHUNK, D]`
   （head-major），而 FLA 操作数布局是 token-major `[B, T, H, D]`（chunk-major、H 在里）。
   直接 `reshape(B,T,H,D)` 会把 head 轴和 chunk 轴**交错**，指纹是「`qg/kg` 相对误差 ≈ 1.0」
   （数值完全对不上但形状合法）。正确写法 `permute(0,1,3,2,4).reshape(...)`。
2. **`INV` 就是 $P=(I+L)^{-1}$，不需转置**：字段按 $P[o_t,o_A]$ 存（与 `Akk` 同布局）；
   前向重建用 $P_c$ 直接左乘 `uprime` 即正确（转置会把 `o` 误差从 8e-3 抬到 3e-2）。
3. **golden 必须吃「激活后」操作数**：我们的链输出的是对 `qn/kn/β_act/g_fla` 的梯度，
   而 `use_*_in_kernel=True` 的 FLA golden 输出的是对**原始 logits** 的梯度，二者差一个
   l2norm / sigmoid 的 Jacobian（直接比会看到 `dq/dbeta` 大面积 0.5+）。
   改用上一轮的 `use_qk_l2norm_in_kernel=False / use_gate_in_kernel=False /
   use_beta_sigmoid_in_kernel=False` 并喂同一份激活操作数即可。
   注意 `g_fla` 是**逐 token** 的自然对数门（`lower_bound·σ(...)`，因 `log2(e)·ln2=1`），
   **不是** cumsum 后的 `g_cum`。
4. **非整除 T 要按 K1 的方式补尾块**：K1 的 tile 恒为 `CHUNK=16` 宽，尾块用零填充；
   重建时把 `v/β/g` 尾部 pad 到 `NT·CHUNK`（pad 置零使行惰性）、结果再切回 `T`。
   与前面「last 有效 token 取 $E^{\text{tot}}$」（§6.2）配齐后，非整除 T 与整除 T 同精度。

### 6.8 $P$ 的存储精度：bf16 `kINV` 够用，无须增存 fp32

**它回答：式 (7.5) 的求逆伴随对 $P$ 敏感，那要把 $P$ 存成 fp32 吗？**

§5.3 留了最后一条精度**待验证**：而目标卡 K1 workspace 把 $P$（=`kINV`，即 FLA 的 `Akk`）
按 **bf16** 落盘（512 B/tile，`utils.cuh:70-76`），§5.3 猜测「可考虑随 workspace 增存一份
fp32 $P$（每 tile 多 1 KB）以换取精度」。

这一轮受控地只动「反向 B2 读到的 $P$ 的存储值」，量它到底值不值这 1 KB。

**Part A：先钉死「存的是什么」。** FLA 前向 `chunk_kda_fwd_intra` 把 `Akk` 开成 `k.dtype`
（`chunk_intra.py:817-819`：`Aqk = torch.empty(..., dtype=k.dtype)`、
`Akk = torch.zeros(..., dtype=k.dtype)`）。

实测：喂 **fp32** 操作数 → `Aqk/Akk/w/v_new/h` 全 fp32；喂 **bf16**（生产路径）→ 全 bf16。
即 **FLA 生产时 $P$ 本来就是 bf16 落盘的**，与目标卡 K1 的 bf16 `kINV` 一模一样。

> 顺带更正 §3.3 的一处措辞：那里写「FLA 在 `302-303` 用的是 `b_A.to(b_A.dtype)`（bf16）」——
> 这个 `.to(b_A.dtype)` 其实是**恒等 no-op**（转成自己），FLA 并没有「强制 bf16」；
> $P$ 的精度完全由**上游把它存成什么 dtype**决定。真正的 `# important to not use bf16`
> 警告在 `chunk_intra.py:597`，指的是**块内衰减链**那一侧，不是这两条求逆伴随 GEMM。
> 因此「增存 fp32 $P$」是对 FLA 的**改进**，而不是修 FLA 的 bug。

**Part B：孤立解析。** 取真实前向的 fp32 $P$（单位下三角），随机严格下三角 $\bar P$，
比 $-P^\top\bar P\,P^\top$ 在 $P$=fp32 vs $P$=bf16($P$).fp32 下的差：
$P$ 自身量化 `max_rel=2.1e-4`，伴随量 `max_rel=4.9e-4`。
即**该公式把 $P$ 的 bf16 量化放大不到 3×**，不存在量级放大。

**Part C：孤立 B2。** 只在 B2 输入里把 `A` 换成 `bf16(P).fp32`（其余操作数保持 fp32，
使伴随 `tl.dot` 仍走 fp32/tf32，隔离出**纯存储量化**这一轴），六项输出：

| | `dq` | `dk` | `dv` | `db` | `dg` | `dA` |
|---|---|---|---|---|---|---|
| `D_max_rel` | **0** | 4.9e-6 | 4.7e-5 | 6.5e-5 | 2.0e-6 | 4.9e-4 |

`dq` 逐位不变（B2 的 `dq=do@S_in` 不含 $P$），$P$ 只在求逆伴随处进链，故误差全落在
`dA`–`db` 一侧且 **≤5e-4**。

（注：真 bf16 MMA 路径还会把伴随的另一个点乘操作数量化，那是 §6.3 已量的
「全 bf16 操作数底噪」，与 $P$ 无关。）

**Part D：整链，$P$-归因增量。** 把 dAv→B1→B2→B3→反向 cumsum 的链跑两遍
（$P$=fp32 vs $P$=bf16 存储值），与 `torch.autograd` 比。
四组 case（T=256/512/1024 加非整除 T=250，H=2/4/8，含非整除）实测：

| 口径 | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|
| 链($P$ fp32) vs autograd 最坏 | 0 | 1.5e-5 | 3.9e-4 | 1.1e-5 | 1.1e-4 | 5.1e-6 |
| 链($P$ bf16) vs autograd 最坏 | 0 | 4.4e-5 | 3.9e-4 | 1.5e-5 | 1.1e-4 | 5.1e-6 |
| **$P$-归因增量（两链之差）** | **0** | **4.4e-5** | **1.3e-4** | **1.5e-5** | **1.0e-4** | **0** |

**结论（给出目标卡设计输入）**：把 $P$ 从 fp32 降到 bf16 存储，六项反向梯度的**归因增量
≤ ~1.3e-4**，比 bf16 操作数底噪（`dk/dg` 3e-2~5e-2，§6.3）低 **2–3 个数量级**，
甚至低于链本身对 autograd 的残差（~1e-4~4e-4，tf32/tf32 链）。

**故目标卡 CUDA 反向无须为 $P$ 增存 fp32 副本**——直接复用 K1 已落盘的 bf16 `kINV` 即可，
省下每 tile 1 KB、且与 FLA 生产行为一致。

§5.3 的「可考虑增存 fp32」与 §5.4 对照表里的对应建议据此**撤回**；若将来要收紧 `dk`，
该投入应优先花在**衰减链的操作数精度**（§3.3 的 `not bf16` 注释所指）而非 $P$ 的存储。

## 7. 三句话小结

- FLA `chunk_kda` 反向的 5 个 Triton kernel 与 `05` §7 的分块反向**逐式对应**：
  `dAv` 给输出支路初始伴随、`dhu` 逆序传状态伴随、`wy_dqkg` 算逐块六项梯度、
  `intra` 回传块内 $A$、反向 `cumsum` 加 `kda_gate_bwd` 收尾门控链；记号映射由实测表钉死
  （`Akk`=$P$、`kg`=$k^{\text{rest}}$、`qg`=$q^{\text{dec}}$、`h`=$S_{\text{in}}$）。
- 目标卡反向的可行方案是**复用 K1 workspace**（六字段已含 $P,M_{qk},k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},E^{\text{tot}}$），只补一个轻量门控重算 B0，
  再按「B3 intra / B2 dqkg / B1 state-adjoint」三个 GEMM kernel 落地，B1 直接镜像前向 K2。
- 这条链已经**逐段跑通并端到端核验**：B1↔FLA 1e-5、B2/B3 逐位、B0 到 $10^{-7}$ 且与 K1 的
  $E^{\text{tot}}$ 对到 $10^{-5}$、整链对 autograd 落 tf32 底噪；架到真 workspace 上也只到
  bf16/tf32 底噪（最坏 6.4e-3）。误差底噪由 tf32 与 bf16 操作数决定，与算法无关。

下一篇（性能篇）进入 FlashKDA 前向的 CUPTI 计时与 `ncu` 剖析，并回答「这么多最小 kernel
工程化成 CUTLASS/TMA 之后，到底还剩多少性能空间」。

## 实验与出处索引

正文里的「上一轮 / 这一轮」在此对账。证据原始输出都在 `code/flash-kda/harness/*.out.txt`。

| 本文位置 | 编号 | 证据 |
|---|---|---|
| §1.1 记号对照 | — | `harness/check_bwd_kernel_map.py`（Part A） |
| §3.6 kernel 分解 | — | `harness/check_bwd_kernel_map.py`（Part B） |
| §6.1 B1 三方对拍 | P6-2 | `harness/check_bwd_dhu_kernel.py` |
| §6.1 与参考实现对齐 | P5-4 | `src/kda_bwd_ref.py` |
| §6.2 bf16 状态伴随 | P6-2 | `harness/check_bwd_dhu_kernel.py` |
| §6.2 非整除 T | P6-3 | `harness/check_bwd_dhu_boundary.py` |
| §6.3 B2 `wy_dqkg` | P6-4 | `harness/check_bwd_dqkg_kernel.py` |
| §6.4 B3 `intra` | P6-5 | `harness/check_bwd_intra_kernel.py` |
| §6.5 B0 门控 | P6-6 | `harness/check_bwd_b0_gate.py` |
| §6.6 dAv 与端到端 | P6-7 | `harness/check_bwd_e2e.py` |
| §6.7 真 workspace | P6-8 | `harness/check_bwd_ws_reuse.py` |
| §6.8 $P$ 的存储精度 | P6-9 | `harness/check_bwd_p_dtype.py` |
| §6.2 前向底噪引用 | P4-2 / P5-3 | `docs/04`、`docs/05` |
