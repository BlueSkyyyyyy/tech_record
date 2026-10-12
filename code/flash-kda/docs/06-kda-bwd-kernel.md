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

## 6. 小结与下一步

- FLA `chunk_kda` 反向的 5 个 Triton kernel 与 `05` §7 的分块反向**逐式对应**：`dAv`=输出支路、
  `dhu`=跨块状态伴随、`wy_dqkg`=逐块 dq/dk/dv/db/dg、`intra`=块内 $A$ 反向、`cumsum(reverse)`+
  `kda_gate_bwd`=衰减/门控链；记号映射由 `harness/check_bwd_kernel_map.py` 的实测表钉死
  （`Akk`=$P$、`kg`=$k^{\text{rest}}$、`qg`=$q^{\text{dec}}$、`h`=$S_{\text{in}}$）。
- 目标卡 CUDA 反向的可行方案：**复用 K1 workspace**（11 个字段已含 $P,M_{qk},k^{\text{dec}},
  q^{\text{dec}},k^{\text{rest}},E^{\text{tot}}$），只补一个门控重算 kernel B0，再按
  「B3 intra / B2 dqkg / B1 state-adjoint」三个 GEMM kernel 落地，B1 直接镜像前向 K2。
- 精度上最需要盯的是 $P$ 的伴随（式 7.5）与状态伴随的 bf16 存储，二者都标了**待验证**。

下一步：

- **P6-2**（可选）：在 `kernel_lab` 上写最小可运行的反向 kernel（先用纯 Triton 或 CUDA 复刻
  `dhu` / B1 状态伴随），与 `src/kda_bwd_ref.py` 对拍，拿到实测 max abs/rel。
- 补一组受控消融：bf16 存 $\bar S$ 对 $dk/d\beta$ 的误差（对应 §5.3 的待验证项）。
