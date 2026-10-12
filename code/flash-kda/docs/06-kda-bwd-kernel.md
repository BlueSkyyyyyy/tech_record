# KDA 反向 kernel：FLA Triton 分解精读与目标卡 CUDA 反向设计

> 本文是「Kimi Delta Attention（KDA）前向 / 反向算子实现解析」的第 6 篇。`05` 篇把逐 token 的
> adjoint 与**分块反向**（chunked backward）推导清楚，并用纯 PyTorch 与 FLA `chunk_kda` 对拍。
> 本篇进入**实现层**：先逐 kernel 精读 FLA `chunk_kda` 的 Triton 反向（它是目前唯一公开的
> KDA 分块反向实现），把它和 `05` §7 的公式逐行对上；再给出**目标卡风格**（CUTLASS / TMA /
> warp specialization、`CHUNK=16`、`D=128`）的 CUDA 反向 kernel 设计——FlashKDA 只开源前向，
> 反向的落地方案属于本专题的原创设计。

## 1. 背景：为什么反向参照 FLA 而不是 FlashKDA

FlashKDA（`~/github/FlashKDA`）只提供**前向**（`csrc/smxx/fwd_kernel1.cuh` /
`fwd_kernel2.cuh`），仓库没有 CUDA 反向。而 `flash-linear-attention` 的 `fla.ops.kda.chunk_kda`
自带一套完整的 Triton 反向；更关键的是，该实现的文件头明确写着

> `# Related files are modified and supported by the Moonshot AI Team`（`fla/ops/kda/chunk.py:8`）

也就是说 FLA 的 KDA 反向与前向出自同一 KDA 算法谱系，是可以用来核对「目标卡会在反向里
做什么」的最近参照。因此 P6 的路线是：**精读 FLA Triton 反向的 kernel 分解 → 抽象出与
`05` §7 公式的对应 → 给出适配 FlashKDA 前向 workspace 的目标卡 CUDA 设计**。

### 1.1 记号对照：FLA 保存量 ↔ `05` §7 分块符号

FLA 的 `chunk_kda` 前向会保存一组中间量给反向用（`fla/ops/kda/chunk.py:103-107`）：
`q,k,v,g_cumsum,beta,Aqk,Akk,w,u,qg,kg,v_new,h`。它们与 `05` §7.1 式 (7.1) 的分块符号
（$q^{\text{dec}},k^{\text{dec}},k^{\text{inv}},k^{\text{rest}},L,M_{qk},P,U,S_{\text{in}}$）
一一对应，但**命名不同**，先钉死，后面才读得懂 kernel：

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
（q,k 已 L2 归一化）实测了这张表（原始输出 `harness/check_bwd_kernel_map.out.txt`）：

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

两个关键结论：**(a)** `Akk` 就是 $P$（相对误差 `1.5e-4`），而不是 $P\,\mathrm{diag}(\beta)$
（后者相对误差 ≈ 1）；**(b)** `W` 方向映射是 `Akk` 先乘 `beta` 再乘 $k^{\text{dec}}$，即
$w=P\,\mathrm{diag}(\beta)\,k^{\text{dec}}$。$10^{-8}$ 项是逐元素复算，$10^{-4}\!\sim\!10^{-3}$
项来自 FLA kernel 内部 `tl.dot` 的 tf32 精度（见下）。

> 踩坑（值得记）：验证 P 时若用 `torch.linalg.solve(I+L, I)` 的 fp32 稠密解，在**未归一化**的
> 随机 k 上 $L$ 元素可达 $\mathcal{O}(\sqrt{K})$，单位下三角阵的逆逐行指数增长，稠密 LU 会
> 把连第一行 `[1,0,...]` 都解成 6.99（残差 $10^5$）。改用**前代（forward substitution）**即
> 恢复：这正是 kernel 里不用 `cuSOLVER` 而用 8×8 前代 + HMMA 合并的原因（`02` §8）。

## 2. FLA `chunk_kda` 反向的整体数据流

入口 `chunk_kda_bwd`（`fla/ops/kda/chunk_bwd.py:434-615`）把反向拆成 5 个 Triton kernel +
2 个逐元素/门控步骤，顺序如下（行号为真实核对）：

| 步骤 | 代码位置 | 作用 | 对应 `05` §7 公式 |
|---|---|---|---|
| 0 重算前向中间量 | `463-506` | `kda_gate_chunk_cumsum` / `recompute_w_u_fwd` / `fwd_h` 得到 `w,u,qg,kg,v_new,h` | 复现式 (7.1) |
| 1 `dAv` | `510-520` | `dAqk=do@U^\top`，`dv=M_{qk}^\top do` | 式 (7.2) 输出支路 |
| 2 `dhu` | `541-555` | 逆序扫块传状态伴随 `dh`，并补 `dv` | 式 (7.13) 跨块传递 |
| 3 `wy_dqkg` | `557-574` | 逐块由 `dh,h` 算 `dq,dk,dv,db,dg,dAkk` | 式 (7.2)~(7.7)、(7.9)~(7.10) |
| 4 `intra` | `576-591` | 由 `dAqk,dAkk` 回传到 `q,k,beta,g` | 式 (7.6)、(7.8) |
| 5 反向 cumsum | `599-605` | `chunk_local_cumsum(reverse=True)` | 式 (7.12) 反向前缀和 |
| 6 门控反向 | `606-613` | `kda_gate_bwd` | §3 式 (5.10) |

注意步骤 0：默认 `disable_recompute=False`，反向**重算**前向的 `w,u,qg,kg,v_new,h`
（`463-506`），而不是全部保存——这是一个重要的工程取舍，§5 会把它和 FlashKDA 的
workspace 设计对照。

## 3. 逐 kernel 精读

### 3.1 `chunk_kda_bwd_kernel_dAv`——输出支路（式 7.2）

kernel 体：`chunk_bwd.py:43-114`。grid `(NT, B*HV)`，每 program 处理一个 chunk。

- 载入 `Aqk` 并**在做用时才施加因果 mask**：`m_A = (o_t[:,None] <= o_t[None,:])`，
  `b_A = tl.where(m_A, b_A, 0)`（`89-90`）。也就是说前向把 $M_{qk}$ 存成了**未 mask 的全阵**
  （Part A 已证 `Aqk[0,0,1]≠0`，`tril(Aqk)` 才等于 $M_{qk}$），反向在这里补 mask。
- 对 $V$ 分块累加（`93-109`）：
  - `b_dA += tl.dot(b_do, b_v)`（`106`）→ $\bar M_{qk}=\bar o\,U^\top$；
  - `b_dv = tl.dot(b_A, b_do)`（`108`）→ $\bar U_{(\text{out})}=M_{qk}^\top\bar o$。
- 写回时把 `dAqk` 乘上 `scale` 并只保留严格上三角以外（`111-114`，`o_t[:,None] >= o_t`）。

这正对应式 (7.2)：$o=q^{\text{dec}}S_{\text{in}}+M_{qk}U$ 的伴随——输出支路给 $\bar M_{qk}$
与 $\bar U$ 一个初值，$\bar q^{\text{dec}}$ 与 $\bar S_{\text{in}}$ 那两片留到 `wy_dqkg` / `dhu`。

### 3.2 `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64`——状态伴随（式 7.13）

kernel 体在 `fla/ops/common/chunk_delta_h.py:359-671`，wrapper `735-794`。grid
`(cdiv(V,BV), N*HV)`——注意它在 **V 维分块**，每个 CTA 持有一段 $[K,\ BV]$ 的状态伴随
`b_dh1..b_dh4`（K 按 64 分四段，`400-415`）。

主循环是 `for i_t in range(NT-1, -1, -1)`（**逆序**，`476`），每步：

1. **落盘本块状态伴随** `dh[tile]`（`480-510`，对应 FLA 保存的 `dh`，形状 `[B,NT,HV,K,V]`）。
2. **更新 `dv`**（`519-570`）：
   - `b_do = load(do)`（`523`），`b_k = load(k)`（`526-527`）；
   - `b_dv = tl.dot(b_k, b_dh)`（`532-534`）——这就是式 (7.3) 的
     $\bar U\mathrel{+}=k^{\text{rest}}\bar S_{\text{out}}$（`k` 在此与衰减相乘）；
   - `b_dv *= exp2(bg_last - b_g)`（`566-567`）把 `k` 补成 $k^{\text{rest}}$（块末相对衰减）；
   - `b_dv += load(dv)`（`568`）叠加步骤 1 的 $\bar U$，写回 `dv2`（`570`）。
3. **更新状态伴随 `dh`**（`571-638`）：
   - 先把 `b_dh *= bg_last_exp`（`576-577`）——即式 (7.3) 的
     $\bar S_{\text{in}}\mathrel{+}=E^{\text{tot}}\odot\bar S_{\text{out}}$；
   - `b_q = b_q * exp2(b_g)`（`578`）把 `q` 补成 $q^{\text{dec}}$；
   - `b_dh += tl.dot(b_q, b_do) * scale - tl.dot(b_w, b_dv)`（`585-587`）——即式 (7.13)
     $$\bar S_{\text{out}}^{(c-1)}={q^{\text{dec}}}^\top\bar o+E^{\text{tot}}\odot\bar S_{\text{out}}^{(c)}-{k^{\text{dec}}}^\top\bar w .$$
     这里 `w` 就是 $P\,\mathrm{diag}(\beta)k^{\text{dec}}$，$\bar w$ 把预测支路吃进来。
4. 循环结束把最终 `b_dh` 写进 `dh0`（`640-671`）→ $\partial L/\partial S_0$。

**结构要点**：`dhu` 只做「$D\times D$ 状态伴随沿块反传」，与 `05` §7.3 完全一致；
它按 V 维切分并行（每 CTA 一段 V），状态伴随是逐块顺序的，但**块内**只有矩阵乘。

### 3.3 `chunk_kda_bwd_kernel_wy_dqkg_fused`——逐块 dq/dk/dv/db/dg（式 7.2~7.10）

kernel 体：`chunk_bwd.py:133-310`。grid `(NT, B*HV)`，每 program 一个 chunk，在 K/V 两个
方向分块累加（`213`、`231`）。输入含 `h`（本块 $S_{\text{in}}$）与 `dh`（本块 $\bar S$，
来自 3.2），输出 `dq,dk,dv,dv2,dg,db,dA`。核心几行（真实核对）：

- `b_dgk += tl.sum(b_h * b_dh, axis=0)`（`253`）——逐 key 通道的 $\bar E^{\text{tot}}$ 贡献；
- `b_dq += tl.dot(b_do, b_h)`（`254`）→ $\bar q^{\text{dec}}=\bar o\,S_{\text{in}}^\top$（式 7.2）；
- `b_dk += tl.dot(b_v_new, b_dh)`（`255`）→ 状态项 $\bar k^{\text{rest}}$（式 7.3）；
- `b_dw += tl.dot(b_dv, b_h)`（`256`）→ $\bar k^{\text{dec}}$ 的预测支路（式 7.7）；
- `b_dA += tl.dot(b_dv, tl.trans(b_v))`（`264`）、`b_dA += tl.dot(b_dw, tl.trans(b_kg))`（`281`）
  → 两条 $\bar L$ 的块内 GEMM（式 7.6、7.8）；
- 衰减链：`b_dq *= exp2(b_gk)*scale`、`b_dk *= exp2(b_gn-b_g)`、逐 token 的
  `b_dg` 合成（`272-298`），对应式 (7.9)~(7.12) 里 $E/E^{\text{tot}}$ 的反向；
- 求逆的伴随：`b_dA = tl.dot(b_dA, b_A); b_dA = tl.dot(b_A, b_dA)`（`302-303`）再取负
  （`304`）——这就是式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$（对称写法，$P$ 缩略见下）。

> 精度注释：FLA 在 `302-303` 用的是 `b_A.to(b_A.dtype)`（bf16）做的两次 `tl.dot`，
> 而 `chunk_intra.py:597` 有一条作者注释
> `# (SY 09/17) important to not use bf16 here to have a good precision.`
> ——说明 L 的伴随对精度敏感，目标卡 CUDA 版应保持 fp32（或 tf32）累加。

### 3.4 `chunk_kda_bwd_kernel_intra`——块内 A 的反向（式 7.6、7.8）

kernel 体：`fla/ops/kda/chunk_intra.py:395-667`，wrapper `897-962`。grid
`(NK*NC, NT, B*HV)`：K 维 × 子块维 × (chunk, batch·head)，即**最细粒度**地并行
（`BC=16` 子块）。它接收 `dAqk,dAkk`，回传 `dq,dk,db,dg`。

前向对应物是 `chunk_kda_fwd_kernel_inter_solve_fused`（`chunk_intra.py:43-379`），它把
**块间（inter）$A_{qk}/A_{kk}$ + 块内 8×8 前代 + Schur 合并求逆**融成一个 kernel
（`265-379`），这与 FlashKDA K1 的「8×8 前代 + 16×16 HMMA 合并」（`02` §8）是同一套代数。

反向里 `b_dq2/b_dk2` 由对 `kgi` 的循环累加（`477-496`、`571-600`），`b_db/b_dg` 由
`b_dk2*b_k` 等合成（`551-552`、`658-667`）。特别地，SAFE_GATE 分支（`509-532`、`607-637`）
用子块中点 `min(BC//2, ...)` 做数值稳定的 `exp2(g - g_n)` 归一——和 FlashKDA 门控
`lower_bound·log2e`（`02` §5）是同一防溢出思路。

### 3.5 门控链（式 5.10）

`dg` 得到后先做**反向前缀和**：`chunk_local_cumsum(dg, reverse=True)`（`chunk_bwd.py:599-605`），
对应式 (7.12) 的 $\bar g_i=\sum_{j\ge i}\hat g^{\text{cum}}_j$；随后 `kda_gate_bwd`
（`606-613`）完成式 (5.10) 的 $\bar g^{\text{logit}}/\overline{A\_log}/\overline{\text{dt\_bias}}$。

### 3.6 kernel 分解的实测证据

`harness/check_bwd_kernel_map.py` Part B 用 `torch.profiler` 抓了一次真实
$B{=}1,T{=}512,H{=}4,K{=}V{=}128$、chunk 64、bf16 的 `chunk_kda` 前向+反向，
CUDA kernel 级统计（`harness/check_bwd_kernel_map.out.txt`）：

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

（`recompute_w_u_fwd_kda_kernel` 出现 2 次：前向一次、反向重算一次；
`wy_dqkg` 因该 run 的 `tl.dot` 归并进相同名字统计而未单列，属正常。）
反向 3 个专用 kernel 的耗时量级与前向同阶，印证「反向是前向同规模的一遍线性化」。

## 4. 与前向 K1/K2 的对称性

把 `03` 的前向结构和本节反向结构并排看，是一个很清晰的「转置 / 反序」关系：

| 前向（FlashKDA） | 反向（对应量） | 关系 |
|---|---|---|
| K1 逐 tile 算 $L,M_{qk},P,k^{\text{dec}},q^{\text{dec}},k^{\text{rest}},E^{\text{tot}}$ | `intra` 由 $\bar L,\bar M_{qk}$ 回传 $q,k,\beta,g$ | 式 (7.6)/(7.8)，**复用前向的 $L,P$** |
| K2 单 CTA 沿 tile **正序**递推 `s_acc` | `dhu` 沿 tile **逆序**传状态伴随 | 式 (7.13) 是状态递推的转置 |
| K2 状态更新 $s\mathrel{+}=k^{\text{rest}\top}U$ | `dhu` 的 $\bar U\mathrel{+}=k^{\text{rest}}\bar S$、$\bar k^{\text{rest}}=U\bar S^\top$ | 式 (7.3) |
| K2 输出 $o=q^{\text{dec}}S+M_{qk}U$ | `dAv`/`wy_dqkg` 的 $\bar q^{\text{dec}},\bar M_{qk},\bar U$ | 式 (7.2) |

一句话：**前向把 $S$ 沿块正向推进，反向把 $\bar S$ 沿块反向推进，块内都是 GEMM，块间只传
$D\times D$。** 这也是「块内并行、块间串行」在两侧都成立的原因。

## 5. 目标卡 CUDA 反向设计（自研）

FlashKDA 只开源前向，下面是本专题给出的、**复用前向 K1 workspace** 的 CUDA 反向方案。

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

**缺口只有一个：`beta` 与原始门 logits 没有落盘**（前向只在 K1 的 smem 里用过）。因此反向
需要一个轻量 **B0 重算 kernel**：读原始 `g`(logits)/`beta`/`A_log`/`dt_bias`，重算
$\sigma(\beta)$ 与 $\alpha=2^{g}$、$g^{\text{cum}}$（后向也需要 $\beta$ 反传到 logits）。
这与 FLA 的 recompute 分支同构，但代价小得多（只重算门控，不重算 GEMM）。

### 5.2 kernel 计划

沿用 K1/K2 的 CUTLASS + TMA + warp specialization 风格，反向拆 3 个 GEMM kernel + 1 个门控：

**B3 `bwd_intra`（对应 FLA `intra`，token-parallel，复用 K1 模板）**
- grid：每个前向 tile 一个 CTA（与 K1 相同，`H·total_tiles`）。
- 输入：workspace 的 $q^{\text{dec}},k^{\text{dec}},k^{\text{rest}},E^{\text{tot}},P,M_{qk}$
  + B0 重算的 $\beta,g^{\text{cum}}$ + 上游 $\bar o$、上一步得到的 $\bar U,\bar S$。
- 计算：按式 (7.2)/(7.6)~(7.10) 先算 $\bar M_{qk}=\bar o\,U^\top$、$\bar U$ 各支路，再用
  **已落盘的 $P$** 做式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$（3 次 16×16 GEMM），最后
  式 (7.6)/(7.8) 两个块内 GEMM 回传 $k^{\text{dec}},k^{\text{inv}},q^{\text{dec}}$。
- `CHUNK=16` 下块内矩阵极小，`intra` 可整体放进寄存器 + HMMA，几乎不耗 smem。
- 输出：`dq,dk,dv` 的块内部分 + `dbeta,dg` 的块内部分 + `dA`（给 `K1`-style 合并）。

**B2 `bwd_dqkg`（对应 FLA `wy_dqkg`，per-tile，需要 $\bar S$）**
- 由 B1 得到的每 tile $\bar S$（$\bar S_{\text{out}}$）和 B0 的 $E^{\text{tot}}$ 做：
  - $\bar U\mathrel{+}=k^{\text{rest}}\bar S_{\text{out}}$，$\bar k^{\text{rest}}=U\bar S_{\text{out}}^\top$；
  - $\bar E^{\text{tot}}=\sum_v\bar S_{\text{out}}\odot S_{\text{in}}$，$\bar S_{\text{in}}\mathrel{+}=E^{\text{tot}}\odot\bar S_{\text{out}}$；
  - 预测支路 $\bar w$、$\bar k^{\text{dec}}$、$\bar S_{\text{in}}\mathrel{+}=-{k^{\text{dec}}}^\top\bar w$。
- 输出 `dq,dk,dv,dbeta,dg`。

**B1 `bwd_state_adjoint`（对应 FLA `dhu`，head-parallel，是 K2 的镜像）**
- grid：每 head 一个 CTA（与 K2 相同），沿 tile **逆序**扫，维护 $\bar S$（$D\times D$，
  fp32；对上前向 K2 的 `s_acc`）。
- 每步按式 (7.13)：$\bar S_{\text{out}}^{(c-1)}={q^{\text{dec}}}^\top\bar o+E^{\text{tot}}\odot\bar S_{\text{out}}^{(c)}-{k^{\text{dec}}}^\top\bar w$，
  并把 $\bar S_{\text{out}}^{(c)}$ 落盘给 B2（bf16，$D\times D$/tile，量级同前向 `h`）。
- 终块初值 $\bar S_T^{\text{out}}$ 来自 `final_state` 上游（可为 0）。
- 数据流可**整体套用 K2 的八个 Phase 与 union/流水线**（`03`），只是把「正序累加」换成
  「逆序累加」，`MOVM_T` 的寄存器转置照用。

**B0 `gate_bwd`（门控链，逐元素 + 反向前缀和）**
- 反向前缀和用 K1 的列并行前缀和骨架反向跑一遍（式 7.12）；
- 门激活链式 (5.10) 与 `kda_gate` 一一对应，可与 B2/B3 融合或在单独小 kernel 里做。

### 5.3 显存与精度取舍

- **状态伴随 $\bar S$**：需要每 tile 一份 $D\times D$ 给 B2。参考 FLA，用 **bf16** 存即可
  （前向 K2 已经验证有衰减 regime 下 bf16 状态存储对输出无可测量损失，`03` §18；
  反向的 $\bar S$ 与状态同阶，可用同一结论，**待验证**：需实测 bf16 $\bar S$ 对
  $dk,d\beta$ 的误差）。
- **$P$ 的精度**：式 (7.5) 的 $\bar L=-P^\top\bar P\,P^\top$ 对 $P$ 敏感。FLA 在这里用了一次
  bf16 `tl.dot` 并留了「不要用 bf16」的注释（§3.3）。目标卡应保持 **fp32/tf32** 累加，$P$ 本身
  已在 K1 以 bf16 落盘（`kINV` 512 B），可考虑随 workspace 增存一份 fp32 $P$（每 tile 多 1 KB）
  以换取精度。
- **不复算 GEMM**：与 FLA 默认「重算 `w,u,qg,kg,v_new,h`」不同，目标卡方案直接吃 K1 workspace，
  反向的额外开销只剩状态伴随 + 块内 GEMM；这正是 K1「把重活落盘」设计的回报。

### 5.4 与 FLA 的差异汇总

| 维度 | FLA Triton | 目标卡 CUDA（本设计） |
|---|---|---|
| chunk | 64（BC=16 子块） | 16 |
| 前向中间量 | 默认**重算** | K1 workspace **已落盘**，直接复用 |
| 状态伴随 | `dhu`，V 维分块、Triton | B1，head 并行、CUTLASS+TMA，镜像 K2 |
| 块内 A 反向 | `wy_dqkg` + `intra` 两个 kernel | B2+B3，`CHUNK=16` 下可合并 |
| 求逆伴随 | 用保存的 bf16 `Akk` | 用 workspace 的 bf16 `kINV`（建议增存 fp32 版） |
| 精度敏感点 | `chunk_intra.py:597` 注释禁 bf16 | 保持 fp32/tf32 |

## 6. P6-2：最小可运行反向 kernel——B1 状态伴随实测

§5 的 B1 是整个反向里唯一**跨块串行**的部分（其余 B2/B3 都是逐块并行），也是最值得先落地
验证的一块。本节在 `kernel_lab`（H100）上写了一个**最小可运行的 Triton 状态伴随 kernel**，
复刻 FLA `dhu`（§3.2）与式 (7.13)，并与三方对拍。证据：
`harness/check_bwd_dhu_kernel.py` / `harness/check_bwd_dhu_kernel.out.txt`。

### 6.1 kernel 契约（与 `dhu` 逐式对应）

对一个 head、沿 block 逆序扫，维护状态伴随 `dS[K,V]`（fp32）。每块 `c`：

```
dh[c] = dS                                  # 落盘给 B2（= 式 7.11 的状态伴随）
dU    = k_rest @ dS + dU_in                  # = FLA dv2（输出支路 dU_in 来自 dAv）
dS   <- E_tot[:,None] ⊙ dS                  # 式 (7.13) 的 E^tot 支（`dhu:577`）
        + q_dec^T @ do · scale               # 输出支路（`dhu:585`）
        - w^T @ dU                           # 预测支路（w = P diag(β) k_dec，`dhu:587`）
dh0   = 完成 c=0 后的 dS                     # = ∂L/∂S_0
```

这四行与 `dhu`（`fla/ops/common/chunk_delta_h.py:576-587`）**逐行同构**：FLA 只是把
`q_dec = qg` 与 `k_rest = kg` 通过 `g`/`gk` 的 `exp2` 缩放现算，并把 K 拆成 64、V 按 `BV` 分块。
本 kernel 用 `tl.dot` 直实现：`tl.dot(k_rest, dS)`、`tl.dot(tl.trans(q_dec), do)`、
`tl.dot(tl.trans(w), dU)`，grid `(cdiv(V,BV), B·HV)`（head 并行、V 分块，镜像 §5.2 的 B1）。

**踩坑（坐标约定）**：`dhu` 的初始 `dS` 来自**终态梯度 `dht`**（无上游时为零），
**不是**初态 `h0`——`h0` 只进 B2 的 `detot`。第一版误把 `h0` 当初始 `dS`，指纹是
「`dh0` 逐位对（1e-3）、但**最后一个块**的 `dh`/`dv2` 全错而其余块全对」——因为只有最末块
的 `dS` 直接由初值决定。改成从 0 起即归位。

### 6.2 三方对拍（fp32）

`A` 用**纯 torch 显式递推**（`dhu_torch`，ieee fp32 matmul）；`B` 是 Triton kernel；
`C` 是 FLA `chunk_gated_delta_rule_bwd_dhu`（真跑）。两组配置，`do` 与 `dU_in` 取自
FLA `dAv` 的真实输出：

| 配置 | 对比 | `dh` max_rel | `dv2` max_rel | `dh0` max_rel |
|---|---|---|---|---|
| B1 T256 H2 D128 BT64 | torch vs FLA | 1.12e-3 | 8.68e-4 | 1.34e-3 |
| | **Triton vs FLA** | **4.60e-5** | **2.07e-5** | **1.82e-5** |
| | Triton vs torch | 1.13e-3 | — | — |
| B1 T512 H4 D128 BT32 | torch vs FLA | 1.16e-2 | 5.23e-3 | 6.73e-3 |
| | **Triton vs FLA** | **3.71e-5** | **3.88e-5** | **8.00e-6** |
| | Triton vs torch | 1.16e-2 | — | — |

**为什么 Triton 反而比 torch 更贴 FLA**：FLA `dhu` 的 `tl.dot` 默认走 **tf32**，torch 显式
递推走 ieee fp32，两者差 ~1e-3（BT32 更长的块链放大到 ~1e-2）；而 Triton kernel 同样走
tf32、还原序也一致，于是 **Triton↔FLA 落到 1e-5**。这本身是一条结论：**`dhu`/B1 的误差底噪
由 tf32 决定（~1e-3），不是 fp32 机器精度**；目标卡 CUDA 若用 tf32 MMA，需接受同量级底噪，
若需更准应保留 fp32 累加（对照 §3.3 的 `P` 伴随精度注释）。

**与 P5-4 参考实现对齐**（Part A2）：把同一份 `q,k,v,g,beta,do` 喂给
`src/kda_bwd_ref.py::kda_chunked_backward`（head 0），其 `dh0` 与 B1 的 `dh0` 差
**max_rel 3.3e-4（BT64）/ 5.2e-3（BT32）**，与上面 torch-vs-FLA 同量级——说明 B1 递推就是
`05` §7.3 分块反向里那条跨块状态伴随链，没有引入额外近似。

### 6.3 精度消融：状态伴随存 bf16（回答 §5.3 的「待验证」）

§5.3 标了「bf16 存 $\bar S$ 对 $dk,d\beta$ 的误差待验证」。B1 里做两个受控消融（fp32 golden）：

- **(i) bf16 carry**（最坏情形：每块把运行中的 `dS` 量化成 bf16 再续传；FLA 实际保留 fp32）：
  `dh` max_rel **2.72e-3 / 2.53e-3**（两组配置），`dh0` 1.83e-3 / 3.08e-3。
- **(ii) bf16 storage**（FLA 的真实做法：`dS` 全程 fp32，只把**落盘的每块 `dh`** 存 bf16）：
  `dh` max_rel **2.72e-3 / 2.53e-3**（与 (i) 相同，因为落盘量化与逐步量化同阶）。

由于 B2（`wy_dqkg`）对 `dh` 是**线性**的（`b_dk += dot(v_new, dh)`、`b_dg += sum(h*dh)` 等，
§3.3），这个 **~2.5e-3 的相对误差会原样传给 `dk/dβ/dA`**。结论：**bf16 存状态伴随可行**，
其误差落在 bf16 GEMM 底噪（前向 `o`/`S` 已 3e-3~1e-2，P4-2/P5-3）之内，与 §5.3 的「同阶」
判断一致；但若追求反向精度，`dh` 用 fp32 落盘（每 tile 多 64 KB @D128）是廉价保险。
Triton 的 bf16-carry 变体与 torch bf16-carry 交叉验证到 5.9e-3 / 1.2e-2（含 tf32），自洽。

### 6.4 计时与边界

单 kernel 计时（`triton.testing.do_bench`，含 grid 启动）：T256 H2 → **107 µs**，
T512 H4 → **191 µs**（`num_warps=4, num_stages=1, BV=32`）。这是「一个 CTA 持整条 K 轴、
V 维只分 32 一路」的朴素实现，远未优化（FLA `dhu` 在 §3.6 的 profiler 里 ~14 µs）；
它的价值是**先把契约和误差底噪钉死**，为后续 CUTLASS/TMA 版 B1（镜像前向 K2 的八个 Phase）
提供可对拍的 golden。局限：BT 需整除 `tl.arange` 且 `K=128` 固定（本 kernel 未做 K 的 64 分段，
故不是 §5.2 的最终形态）。

### 6.5 非整除 T 的边界：P6-2 的「待验证」已关闭（`harness/check_bwd_dhu_boundary.py`）

§6.4 末尾留了「非整除 T 待验证」。补测后结论：**kernel 本体本来就是 T-general，只有 host 侧
`E_tot` 记账假设了整除**——这是一处真实的、此前未被覆盖的边界 bug。

**根因**：P6-2 的两个 wrapper 用 `.view(B, NT, BT, H, K)` 把逐 token 张量折叠成按块视图。
当 $T\not\equiv 0\pmod{BT}$ 时这个 reshape 直接抛
`RuntimeError: shape '[1,4,64,2,128]' is invalid for input of size 64000`（$T=250$：尾巴 58 行）。
也就是说尾块的 **$E^{\text{tot}}$ 根本没被寻址过**。而 FLA 的前向量是**不给 pad** 的
（`g_cum` 形状 $[B,T,H,K]$，$T=250$ 实测 size $=250\cdot H\cdot K$），FLA 自己的 `dhu`
靠 mask 处理非整除 T，是合法参照。

**修法**：$E^{\text{tot}}_c = 2^{g\_cum[\min((c{+}1)BT,T)-1]}$——按块的**最后一个有效 token** 取
（`g_cum` 是块内局部 cumsum，最后有效行天然等于该块有效行的门控和），尾块 pad 行直接丢弃。
kernel 侧不用改：`m_t = o_t < T` 已把每个 token load 置零，pad 行对 `tl.dot` 贡献为 0。

**实测**（fp32，Triton↔FLA `max_rel`；`harness/check_bwd_dhu_boundary.out.txt`）：

| config | NT | tail | `dh` | `dv2` | `dh0` |
|---|---|---|---|---|---|
| T=256 BT=64（可整除，回归 P6-2） | 4 | 0 | 4.60e-5 | 2.07e-5 | 1.82e-5 |
| T=250 BT=64 | 4 | 58 | 4.60e-5 | 2.07e-5 | 1.82e-5 |
| T=100 BT=64 | 2 | 36 | 9.12e-8 | 2.74e-5 | 7.17e-6 |
| T=250 BT=32 | 8 | 26 | 1.41e-5 | 3.78e-5 | 2.75e-5 |
| B2 T=130 H3 BT=64 | 3 | 2 | 8.56e-6 | 1.96e-5 | 1.46e-5 |
| T=513 BT=64 | 9 | 1 | 2.49e-5 | 2.48e-5 | 1.84e-5 |
| T=31 BT=32（单块） | 1 | 31 | 0 | **0** | 1.07e-7 |

- 可整除的 T=256 与 P6-2 数字**逐位相同**（4.60e-5/2.07e-5/1.82e-5）——修法是纯 no-op，不改变
  已有结论。
- 非整除各例 `dh/dv2/dh0` 都落在 **1e-8~4e-5**，与可整除同量级（tf32 底噪），说明尾块处理正确。
- 单块 `T=31`：`dh≡0`（$NT=1$ 时 `dh[0]=dS=0`，无跨块伴随），`dv2` **逐位等于** FLA（$dU=k^{rest}{\cdot}0+dv$，
  tf32 下 $0$ 加法精确）——这是尾块 mask 的最干净证据。

**关于「K 的 64 分段」**：查 FLA `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` 的
`grid=(cdiv(V,BV), N*HV)`（`fla/ops/common/chunk_delta_h.py` 的 `chunk_gated_delta_rule_bwd_dhu`），
它**按 V 分块、K 不占 grid**，`K=64/128/192/256` 是同 program 内 `o_k1..o_k4` 的**展开**（`K>64` 才
多算一组），属寄存器压力取舍而非正确性约束。本 kernel 以 `K=128` 单块实现，与其数学等价，故
「K 的 64 分段」不是待补的正确性缺口（若做目标卡 CUDA B1，K 轴如何切是性能问题，见 §5.2 设计）。

### 6.6 P6-4：最小可运行 B2（`wy_dqkg`）kernel——`dq/dk/dv/dβ/dg/dA` 实测

§6 落地了 B1（跨块状态伴随）。反向里第二块「逐块并行、又要吃状态」的是 **B2 `bwd_dqkg`**
（对应 FLA `chunk_kda_bwd_kernel_wy_dqkg_fused`，`chunk_bwd.py:133-310`，即 `05` §7.2~7.5、
7.7~7.10 的逐块公式）。本节把它做成最小可运行 Triton kernel 并三方对拍。
证据：`harness/check_bwd_dqkg_kernel.py` / `harness/check_bwd_dqkg_kernel.out.txt`。

#### 6.6.1 kernel 契约（寄存器的转置约定是关键）

B2 的输入就是 B1 的输出加前向 workspace：`k,q,v,g(块内 log2 cumsum),beta,A(P),h(S_in)`、
`do`、`dh`（B1 落盘的每块 $\bar S$）、`dv`（`dAv` 的输出支路 $\bar U$）；输出
`dq,dk,dv2,db,dg,dA`。与 `05` §7 公式一一对应（`chunk_bwd.py` 真实行号）：

| 代码（`chunk_bwd.py`） | 公式 |
|---|---|
| `b_dgk += tl.sum(b_h * b_dh, axis=0)`（`253`） | $\bar E^{\text{tot}}=\sum_v \bar S\odot S_{\text{in}}$ |
| `b_dq += tl.dot(b_do, b_h)`（`254`） | $\bar q^{\text{dec}}=\bar o\,S_{\text{in}}^\top$（式 7.2） |
| `b_dk += tl.dot(b_v_new, b_dh)`（`255`） | 状态项 $\bar k^{\text{rest}}=U\,\bar S$（式 7.3） |
| `b_dw += tl.dot(b_dv, b_h)`（`256`） | 预测支路 $\bar w$（式 7.7） |
| `b_dA += tl.dot(b_dv, tl.trans(b_v))`（`264`） | 块内 $\bar L$ 第一支 |
| `b_dw=-b_dw.to(A.dtype); b_dA+=tl.dot(b_dw, tl.trans(b_kg))`（`280-281`） | $\bar L$ 第二支（式 7.6/7.8） |
| `b_dA=dot(dA,A); b_dA=dot(A,b_dA); b_dA=-b_dA`（`302-304`） | $\bar L=-P^\top\bar P\,P^\top$（式 7.5） |

一个容易踩的细节：kernel 里**所有「矩阵」在载入寄存器时都被转置了**——`b_A[i,j]=A^{\text{T}}[i,j]`
（`p_A` 用 `o_A` 行 `o_t` 列，`chunk_bwd.py:205-208`）、`b_h[v,k]=h[k,v]=S_{\text{in}}^\top`、
`b_dh=dS^\top`。原因是这些片段随后要当 `tl.dot` 的 B 操作数（列主序），等价于 host 侧的
`S.transpose(-1,-2)`。`harness` 里的 torch 参考（`wy_dqkg_torch`）用 `einsum` 把这些转置显式写出并与
kernel 对齐，A 张量按 `[B,T,H,BT]`（`A4[b,o_t,h,o_A]`）布局——`b_A` 即其 chunk 内转置。

#### 6.6.2 三方对拍：Part A（torch，ieee fp32）与 Part B（Triton，tf32）

`Part A` 是纯 torch 逐块实现（ieee fp32），`Part B` 是本 kernel，参照是 FLA 的 Triton 内核
（本机 FLA 默认会把 `@dispatch('kda')` 路由到 **TileLang** 后端，见 6.6.5；这里用
`FLA_DISABLE_BACKEND_DISPATCH=1` 钉回文档所读的 Triton 实现）。实测
（`check_bwd_dqkg_kernel.out.txt`）：

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

- **Part A 的 ~1e-3 就是 `tl.dot` 的 tf32 误差**：torch 参考走 ieee fp32、FLA 内核走 tf32，六项相对
  误差都落在 **5.8e-4~2.1e-3**，与 `05`/§6.2 里「`dhu` 的误差底噪是 tf32」完全同一量级。这说明
  `wy_dqkg_torch` 与 kernel 的**代数完全一致**（若有某个转置写反，会出现 O(1) 的相对误差）。
- **Part B 逐位相同（max_abs=0）**：本 kernel 是 FLA `wy_dqkg` 的忠实移植（BK/BV 分块、`i_k==0`
  分支、`tl.debug_barrier()`、求逆伴随的 bf16 cast 都照搬），因此 fp32 输入下与 FLA 位级相同。
  这既验证了移植的正确性，也说明「B2 的算术路径 = FLA 的路径」。**非整除 T（尾块）也逐位相同**，
  即 `wy_dqkg` 是 per-chunk 独立的、不依赖 `T` 的分块（与 B1 的 host 侧非整除 bug 不同）。

#### 6.6.3 精度底噪：bf16 操作数（Part C）

把全部输入量化到 bf16 再喂本 kernel（模拟目标卡 bf16 GEMM），对 fp32 FLA 参照：

| config | `dq` | `dk` | `dv` | `db` | `dg` | `dA` |
|---|---|---|---|---|---|---|
| B1 T256 H2 D64 | 2.9e-3 | 5.0e-2 | 3.2e-3 | 2.5e-3 | 4.3e-2 | 5.5e-3 |
| B2 T512 H4 D128 | 2.8e-3 | 4.6e-2 | 3.6e-3 | 2.4e-3 | 3.3e-2 | 6.4e-3 |

`dq/dv/db/dA` 落在 **2.3e-3~6.4e-3**（bf16 GEMM 底噪），但 **`dk`/`dg` 明显更高
（3.3e-2~5.0e-2）**：因为 `dk` 的衰减链 $\exp_2(g^{\text{tot}}-g)$ 和 `dg` 的
`b_dg = q*dq - k*dk + ...` 都是**小量相减 / 指数放大**，对操作数量化更敏感。这解释了为什么 FLA
在 `chunk_intra.py:597` 专门留了「这里不要用 bf16」的注释（§3.3）——**`g`/`k` 相关通道应保
fp32/tf32**。目标卡 B2 的设计（§5.2）因此建议门控链与 `dk/dg` 的合成走 fp32。

#### 6.6.4 计时（Part D，朴素实现）

`triton.testing.do_bench`（naive，`num_warps=4, num_stages=1`，BK=BV=64）：
B1 T512 H4 D128 → **97.8 µs**（FLA 88.4 µs）；B2 T2048 H8 D128 → **496.8 µs**（FLA 452.8 µs）。
本 kernel 比 FLA 慢约 10%，符合「同结构、未调 autotune」的预期；它不是性能交付，价值在把
**B2 的契约、转置约定、误差底噪**钉死，供目标卡 CUDA B2（CHUNK=16，复用 K1 workspace 的
$P/M_{qk}/k^{\text{dec}}/q^{\text{dec}}/k^{\text{rest}}/E^{\text{tot}}$）对拍。

#### 6.6.5 踩坑：运行期 FLA 会路由到 TileLang 后端

本机 `fla` 0.5.x 里 `chunk_kda_bwd_wy_dqkg_fused` 被 `@dispatch('kda')` 装饰，运行时会优先选中
**TileLang** 实现（`fla/ops/kda/backends/tilelang/chunk_bwd_dqkg.py`），而它当前 **fp32 路径会编译
失败**（无 fp32 MMA：`tl::mma_sync: unsupported configuration`）。两条对策：(a) 对拍时设
`FLA_DISABLE_BACKEND_DISPATCH=1` 回到文档所精读的 Triton 内核（本篇即如此）；(b) 若必须用
TileLang 后端，用 bf16 输入（fp32 会撞上面的 unsupported 配置）。这也提醒：**文档 `06` §3 精读的
Triton 代码在运行期不一定是被执行的那一份**——精读与实测要同时交代后端。

### 6.7 P6-5：最小可运行 B3（`intra`）kernel——`dq/dk/dβ/dg` 的块内回传实测

P6-4 之后，§5.2 的三段 GEMM 里只剩 **B3 `bwd_intra`**（FLA `chunk_kda_bwd_kernel_intra`，
`chunk_intra.py:395-667`）没做成最小 kernel。它吃 `dAv`+`wy_dqkg` 产出的两张块内矩阵
`dAqk`（$=\operatorname{tril}(\bar M_{qk}\cdot\text{scale})$）、`dAkk`（$A_L/\bar L$ 链的伴随），
把它们回传到块内的 `dq, dk, dbeta, dg`（`05` §7 式 (7.6)、(7.8)、(7.10) 的 $q/k$ 链）。本节把它
做成最小可运行 Triton kernel 并三方对拍。证据：`harness/check_bwd_intra_kernel.py` /
`harness/check_bwd_intra_kernel.out.txt`。

#### 6.7.1 kernel 契约与四个阶段（真实行号）

grid `(NK*NC, NT, B*HV)`（`chunk_intra.py:928`），`BC=16`、`BK=min(32,2^⌈log2 K⌉)`；
每个 program 处理**一个 query 子块**（`BC=16` 行）的一段 K。核心是四个阶段：

| 阶段 | 行号 | 作用 |
|---|---|---|
| A 前序整块 | `473-496` | $\bar q^{dec}{+}{=}\bar A_{qk}k^{inv}$、$\bar k^{dec}{+}{=}\bar A_L k^{inv}$（式 7.8） |
| B 对角块 | `534-546` | 同上，但对角子块按 `m_i=o_i≥j` 因果裁剪，逐 `j` 展开 |
| C 后继整块 | `571-599` | $\bar k^{inv}{+}{=}\bar A_{qk}^\top q^{dec}$（**转置读**，见下），再 $E$ 缩放 |
| D 对角块 | `639-660` | 同上，按 `m_i=o_i≤j` 反向裁剪 |

末尾合成（`662-667`）：`dg = \bar q^{dec}\odot q + (\beta\bar k^{dec}-\bar k^{inv})\odot k + dg_{in}`、
`dk = \beta\bar k^{dec}+\bar k^{inv}+dk_{in}`。注意 `db` 由 `b_dk2*b_k` 行和得到（`551`），且这是
**逐 token 标量在 `NK` 个 K-块间求和**——FLA 用 `db2[NK,...].sum(0)`（`930/957`）避免写冲突。

#### 6.7.2 三个容易踩的点（都靠对拍抓出来的）

1. **漏了 `b_dg2 = b_q * b_dq2`（`chunk_intra.py:558`）。** 这是衰减链里 $\bar g{+}{=}\bar q^{dec}\odot q$
   （式 7.10 的 $\bar E$ 项）。第一版漏掉它，指纹非常干净：**`dq/dk/db` 全对、只有 `dg` 大面积错
   （`max_rel≈1.0`）**；补上后 `dg` 立刻归位。且它必须用**加回上游 `dq` 之前**的 `b_dq2`（`558` 在
   `559` 之前），顺序写反会引入上游累积量的污染。
2. **`Part C/D` 对 `dAqk/dAkk` 是「转置读」。** A/B 段按下标 `dAqk[query, key_local]` 直读
   （`482`），而 C/D 段写成 `dAqk + (i_i*BC+o_i)[:,None] + o_j[None,:]*(HV*BT)`（`580`）——即
   行是**块内局部列**、列是**绝对 token**，等价于取 $\bar A_{qk}^\top$。这里的关键坑是：**行下标
   是 chunk-内局部量 `i_i*BC+o_i`，不是绝对量 `i_ti`**。第一版误用 `i_ti`（= `i_t*BT+i_i*BC`），
   结果**只有第 0 个 chunk 对、其余 chunk 全错**（`dk` 错到 $\mathcal{O}(1)$）——因为 `i_t=0`
   时两者恰好相等。这再次说明「单 chunk 测试会漏跨块/绝对-局部混淆」的 bug。
3. **`db` 的跨 K-块归约。** 本 kernel 用 `tl.atomic_add` 落到一张 `db` 上（`intra_triton`），
   与 FLA 的 `[NK,...].sum(0)` 数学等价但**加法顺序不同**，故 `db` 不是逐位相同而是停在
   ~1e-7（fp32 尾数）。

#### 6.7.3 三方对拍（fp32）

`Part A` 是纯 torch 逐块实现（ieee fp32，`intra_torch`），`Part B` 是本 kernel，参照 FLA 的 Triton
内核（`FLA_DISABLE_BACKEND_DISPATCH=1` 钉回文档所读实现，同 §6.6）。实测
（`check_bwd_intra_kernel.out.txt`）：

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

- **Part A 的 ~5e-4~9e-4 就是 `tl.dot` 的 tf32 误差**（torch 走 ieee fp32、FLA 走 tf32），与
  `05`/§6.2/§6.6 的底噪同量级，说明 `intra_torch` 与 kernel **代数完全一致**（任一转置/`m_i`
  写反都会是 $\mathcal{O}(1)$ 误差）。
- **Part B 的 `dq/dk` 逐位相同（max_abs=0）**，`db/dg` 停在 ~1e-7；**非整除 T（250/513）也成立**，
  即 `intra` 与 `wy_dqkg` 一样是 **per-query-subblock 独立**的分块，不依赖 `T` 整除（与 B1 的 host
  侧非整除 bug 不同，§6.5）。至此三条 GEMM 支路（B1/B2/B3）全部有了位级可核验的最小 kernel。

#### 6.7.4 精度底噪：bf16 操作数（Part C）

把全部输入量化到 bf16 再喂本 kernel（模拟目标卡 bf16 GEMM，`tl.dot` 操作数 cast 到 bf16、
累加 fp32），对 fp32 FLA 参照：

| config | `dq` | `dk` | `db` | `dg` |
|---|---|---|---|---|
| B1 T256 H2 K64 | 4.8e-2 | 3.0e-2 | 1.1e-2 | 2.0e-2 |
| B2 T512 H4 K128 | 2.8e-2 | 3.5e-2 | 2.1e-2 | 2.2e-2 |

全部落在 **1.1e-2~4.8e-2**——比 §6.6 的 B2（`dq/dv/db/dA` 2.3e-3~6.4e-3）高约一个量级，因为
`intra` 的 A/B/C/D 四段全是**指数衰减因子直接乘在操作数上**（`exp2(g-g)`），bf16 量化被指数放大，
正是 `chunk_intra.py:597` 那条 `# important to not use bf16` 注释要防的。**结论**：B3 必须保
fp32/tf32，目标卡的设计（§5.2）也据此把门控/衰减链留在 fp32。

#### 6.7.5 计时（Part D，朴素实现）

`triton.testing.do_bench`（naive，`num_warps=4, num_stages=1`，`BC=16`、`BK=32`）：
B1 T512 H4 K128 → **86.9 µs**（FLA 86.4 µs）；B2 T2048 H8 K128 → **591.1 µs**（FLA 592.0 µs）。
本 kernel 与 FLA 基本同速（差 <1%），符合「同结构」预期；它不是性能交付，价值在把 **B3 的
契约、四个阶段、转置/绝对-局部混淆的坑、bf16 误差底噪**钉死，供目标卡 CUDA B3（`CHUNK=16`、
复用 K1 workspace 的 $P/M_{qk}/k^{dec}/q^{dec}/k^{rest}/E^{tot}$）对拍。

## 7. 小结与下一步

- FLA `chunk_kda` 反向的 5 个 Triton kernel 与 `05` §7 的分块反向**逐式对应**：`dAv`=输出支路、
  `dhu`=跨块状态伴随、`wy_dqkg`=逐块 dq/dk/dv/db/dg、`intra`=块内 $A$ 反向、`cumsum(reverse)`+
  `kda_gate_bwd`=衰减/门控链；记号映射由 `harness/check_bwd_kernel_map.py` 的实测表钉死
  （`Akk`=$P$、`kg`=$k^{\text{rest}}$、`qg`=$q^{\text{dec}}$、`h`=$S_{\text{in}}$）。
- 目标卡 CUDA 反向的可行方案：**复用 K1 workspace**（11 个字段已含 $P,M_{qk},k^{\text{dec}},
  q^{\text{dec}},k^{\text{rest}},E^{\text{tot}}$），只补一个门控重算 kernel B0，再按
  「B3 intra / B2 dqkg / B1 state-adjoint」三个 GEMM kernel 落地，B1 直接镜像前向 K2。
- §6 已把 **B1 状态伴随**做成最小可运行 Triton kernel：与 FLA `dhu` 逐行同构，
  Triton↔FLA 实测 **1e-5**（tf32 底噪），并与 `src/kda_bwd_ref.py` 的 `dh0` 对到 ~3e-4；
  状态伴随 bf16 存储的受控消融给出 **~2.5e-3** 的相对误差（回答了 §5.3 的待验证项）；
  §6.5 关闭「非整除 T」边界（尾块 `E_tot` 取最后有效 token，Triton↔FLA 仍 1e-8~4e-5）。
- §6.6（P6-4）已把 **B2 `wy_dqkg`** 做成最小可运行 Triton kernel：torch 独立参考 vs FLA 内核
  六项梯度 **5.8e-4~2.1e-3**（=tf32 底噪，证明代数一致），本 kernel vs FLA **逐位相同（含非整除
  T）**；bf16 操作数下 `dq/dv/db/dA` **2.3e-3~6.4e-3** 而 `dk/dg` 抬到 **3.3e-2~5.0e-2**
  （门控/衰减链对量化敏感，印证 §3.3 的「不要 bf16」注释）；并记录运行期 FLA 路由到 TileLang
  后端（fp32 路径编译失败）这一实测陷阱。
- §6.7（P6-5）已把 **B3 `intra`** 做成最小可运行 Triton kernel：torch 独立参考 vs FLA 内核
  `dq/dk/db/dg` **3.0e-4~8.8e-4**（=tf32 底噪，证明代数一致），本 kernel vs FLA 的 **`dq/dk`
  逐位相同、`db/dg` ~1e-7（atomic-add 顺序）**，**含非整除 T**；bf16 操作数下四项抬到
  **1.1e-2~4.8e-2**（指数衰减因子直接作用于操作数，印证 `chunk_intra.py:597` 的「不要 bf16」），
  朴素计时与 FLA 同速。三个踩坑点：漏 `b_dg2=b_q*b_dq2`（指纹=dq/dk/db 对、仅 dg 错）、
  Part C/D 转置读时把 chunk-局部行下标误写成绝对 `i_ti`（指纹=只有第 0 chunk 对）、
  `db` 跨 K-块的 atomic 归约。至此 §5.2 的 **B1/B2/B3 三条 GEMM 支路全部有最小可运行 kernel**。

下一步：

- P7：FlashKDA 前向 CUPTI 计时与 `ncu` 剖析。**P6 的三条 GEMM 支路（B1 状态伴随 / B2 `wy_dqkg`
  / B3 `intra`）均已做成最小可运行 Triton kernel 并与 FLA 位级核验**；§5.2 剩余的是把这些
  最小 kernel **工程化**为 CUTLASS/TMA + warp-specialization 的目标卡 CUDA（复用 K1 workspace、
  `CHUNK=16`），以及 B0 门控重算 kernel——属更大工程，留待有明确需求时再做。
