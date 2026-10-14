# 03 · KDA 算子优化手段

> 系列第 4 篇。汇总把 KDA 前反向算子「做快、做稳、做省」的手段：
> **算法层**（分块 / 融合 / 重算）→ **kernel 层**（tiling / autotune / 精度 / 布局）
> → **系统层**（变长 / CP / graph / 后端 dispatch），最后给移植到目标卡的取舍建议。
> 前向数式见 [01-forward.md](01-forward.md)，反向见 [02-backward.md](02-backward.md)。
>
> 说明：本篇聚焦 FLA 算法与 Triton kernel 提供的优化**选项**；FlashKDA 手写 CUDA 的
> 微观优化（TMA / warp-specialization / 寄存器常驻等）见本仓库 `flash-kda` 主系列的
> `docs/07-perf.md`，本篇不重复。

---

## 1. 算法层：分块化本身是最大的优化

逐 token 递推沿 $T$ **严格串行**、每步 $O(KV)$，总时间 $O(TKV)$ 且无法并行。
分块把序列切成 $C$，块间只传 $D\times D$ 状态，块内变成 $(I+L)^{-1}$ 的小矩阵代数：

| 维度 | 逐 token | 分块 |
|---|---|---|
| 串行深度 | $T$ | $\lceil T/C\rceil$ |
| 块间状态搬运 | 每 token $O(KV)$ | 每块 $O(KV)$ |
| 块内并行 | 无 | 沿 token / head 全并行 |
| 计算量 | $O(TKV)$ | $O(TKV)+O(T/K\cdot C^2 V)$ |

KDA 取 $C\in\{32,64\}$（`chunk.py:473-475`），块内再切 $B_C=16$ 子块用于
`forward substitution`/`solve_tril`（`chunk_intra.py:826`）。$C=64$ 时
$C^2$ 的额外项开始显著，$C=32$ 省算但块间串行深度加倍——这是移植时要 sweep 的第一旋钮。

**两段式块内**：先「token 并行」算对角块（`chunk_intra_token_parallel.py:130-188`），
再「chunk 并行」算非对角块并 `solve_tril`（`chunk_intra.py:44-384`）。这样对角块
的计算与 pad 行无关，且非对角块可以并到 `(I+L)^{-1}` 的块前代里一次做完。

---

## 2. 融合：能一次算完的绝不落盘

FLA 的融合分三处：

1. **门控 + 块内 cumsum**：`kda_gate_chunk_cumsum`（`gate.py:479-523`）把
   $-e^A\text{softplus}(g+\text{bias})$ 与 `cumsum` 融进一个 kernel，省一次
   $[B,T,H,K]$ 的全局往返，也是 Megatron 传 raw g 的原因。
2. **非对角块 + `solve_tril` + 块前代**：`chunk_kda_fwd_kernel_inter_solve_fused`
   （`chunk_intra.py:44-384`）把「算 $A_{kk}$ 非对角块」「读回对角块」「前代合并」
   放进同一个 program（`chunk_intra.py:137-356`），中间量全在寄存器，只落最终逆。
3. **反向 Q/K/G/B/A 五路伴随**：`chunk_kda_bwd_wy_dqkg_fused`（`chunk_bwd.py:383-458`）
   把 $W/U/v_\text{new}/S_\text{out}$ 四处的伴随一次算完，避免多次读写
   $[B,T,H,K]$ 级张量。

对照：`safe_gate=True` 时块内**对角**走 `chunk_kda_fwd_kernel_intra_sub_chunk`
（`chunk_intra.py:693-804`）而非 token 并行版，因为它要顺带做 16×16 的前代。

---

## 3. 重算 vs 保存：`disable_recompute`

默认 `disable_recompute=False`：前向算完 $o$ 就把 `w,u,qg,kg,v_new,h` 释放
（`chunk_fwd.py:135-142`），反向重算（`chunk_bwd.py:492-532`）。

| 模式 | 显存 | 时间 | 适用 |
|---|---|---|---|
| `disable_recompute=False` | 省（不存中间量） | 多一次前向 kernel (2)(3) | 默认、大模型 |
| `disable_recompute=True` | 费（存全部） | 快 | 小模型、训练吞吐优先 |

Megatron 侧对应 `recompute_gdn` 与 `tensor_parallel.checkpoint`（`kda.py:321-337`），
即**层级**重算；两者粒度不同，移植时应避免「层级 + 算子级」双重存/算叠加。

---

## 4. 数值稳定与精度

### 4.1 log2 域 + `exp2`

内核统一用 `exp2`（$g$ 与 cumsum 都乘 `RCP_LN2=1/\ln2`），少一次常数乘，
且 `exp2` 在硬件上更快（`chunk_fwd.py:53`、`gate.py:470`）。

### 4.2 强衰减下的溢出

直接算 $k^{\text{inv}}=k/E$ 会在 $E$ 下溢时上溢。FLA 的两条防护：

- **减参考点**：块内以子块端点 $g_n$ 为锚，$2^{g_i-g_n}$、$2^{g_n-g_j}$
  （`chunk_intra.py:160-166`、`chunk_intra.py:756-769`），指数落在安全区。
- **逐对差**：$k^{\text{rest}}_i=k_i 2^{\,g_\text{tot}-g_i}$ 的指数 $\le0$
  （`wy_fast.py:124-127`），$2^{g_i-g_j}\,(i\ge j)\le1$。

参考实现 `src/kda_ref.py:196-224` 即用逐对差，并在注释里点明这一点。

### 4.3 `solve_tril` 用 tf32

求 $(I+L)^{-1}$ 的块前代用 **tf32** `tl.dot`（`SOLVE_TRIL_DOT_PRECISION`，
`chunk_intra.py:21-24`、`:320-356`）：逆矩阵对 tf32 不敏感，而 tf32 快得多。

### 4.4 哪些地方**不能**降精度

- 块内**反向**的衰减链：`chunk_intra.py:607` 明确注释「不要用 bf16」，
  否则 $dq/dk/dg$ 会掉一个量级精度。
- 门控的 $A_\text{log}/\text{dt\_bias}$：Megatron 用 `mark_keep_in_fp32`
  强制 fp32（`kda.py:117-118`），FLA 侧 `dA/dbias` 也以 fp32 累加
  （`gate.py:297-298`）。
- `dg` 的反向前缀和与门控反向在 fp32（`chunk_bwd.py:637-652`）。

### 4.5 `safe_gate`：精度换 TensorCore

`safe_gate=True` 把单步门夹到 $[\text{lower\_bound},0)$，使块内 $M=16$ 的
`tl.dot` 可用（`chunk.py:288-291`），显著提速；代价是门被截断（推荐
`lower_bound=-5`，衰减 $\approx0.0067$）。

---

## 5. Kernel tiling 与 autotune

FLA 用 `fla_cache_autotune`（磁盘缓存 autotune 结果）对各 kernel 扫参。主要旋钮：

| kernel | 扫的空间 | 源码 |
|---|---|---|
| `chunk_kda_bwd_kernel_wy_dqkg_fused` | `BK∈{32,64}`、`BV∈{64,128}`(ampere)、`num_warps∈{2,4}`(Hopper)/{2,4,8}、`num_stages∈{2,3,4}` | `chunk_bwd.py:26-28,125-137` |
| `chunk_kda_bwd_kernel_dAv` | `num_warps`、`num_stages` | `chunk_bwd.py:31-42` |
| `chunk_kda_fwd_kernel_inter_solve_fused` | `BK∈{32,64}`、`num_warps∈{1,2,4}` | `chunk_intra.py:34-42` |
| `chunk_kda_bwd_kernel_intra` | `num_warps∈{1,2,4,8}`、`num_stages∈{2,3,4}` | `chunk_intra.py:390-398` |
| `kda_gate_*` | `BT∈{32,64,128}`、`num_warps∈{4,8,16,32}`、`num_stages∈{2,3}` | `gate.py:22-24,102-111` |
| `chunk_gated_delta_rule_fwd/bwd_h` | `BV∈{32,64}`、`num_warps`、`num_stages` | `chunk_delta_h.py:48-57,359-368` |
| `chunk_kda_fwd_kernel_intra_token_parallel` | `BH∈{1,2,4,8}`、`num_warps∈{1,2,4,8}` | `chunk_intra_token_parallel.py:23-31` |

**平台相关的硬约束**（移植时的坑）：

- SM100（Blackwell）上 `chunk_gated_delta_rule_fwd_h` 的 `num_warps` **锁死为 2**
  （`chunk_delta_h.py:27-33`）：Triton 3.8 前 `tl.dot` 递推存在 race，放宽 config 会错。
- Hopper 上 `wy_dqkg` 禁用 `(BK=32, num_warps=4)`，SM100 上 `BK=32` 只允许
  `num_warps=2`（`chunk_bwd.py:132-133`）。
- `BK/BV/BS` 的上限由 `check_shared_mem` 决定（`chunk_bwd.py:26-27`、`gate.py:22`）。

---

## 6. 布局与并行结构

- **状态布局 `state_v_first`**：默认 $[K,V]$；某些 kernel 在 $[V,K]$ 下能省掉
  `tl.trans`（`chunk_bwd.py:247-252`、`chunk_delta_h.py:544-547`）。Megatron 的
  持久化状态/终态可以据此选布局，减少反向的转置。
- **状态按 $K$ 分 4 个 64 列块常驻寄存器**：`b_h1..b_h4`（`chunk_delta_h.py:100-115`），
  使 $K\le256$ 的单 head 状态不必反复落盘；$K>64$ 时逐块累加。
- **沿 $V$ 切 grid、沿 $T$ 串行**：`fwd_h`/`dhu` 的 grid 是 `cdiv(V,BV)*N*HV`
  （`chunk_delta_h.py:726,798`）。$V$ 越大并行度越高；若 $V$ 小则 grid 小、
  串行延迟暴露（这是 KDA 前向的主要延迟来源，详见 `flash-kda/docs/07-perf.md`）。
- **反向 $dq/dk$ 在 $H_V$ 维并行**：`wy_dqkg` 输出的 `dq,dk` 是 $[B,T,H_V,K]$，
  最后归约（`chunk_bwd.py:410,631-634`），即 GVA 下每个 value head 独立反传。
- **块内反向按 $K$ 切 `NK`**：`db2` 存成 `[NK, *beta.shape]` 再 `sum(0)`
  （`chunk_intra.py:952,986`）——把跨 $K$ 块的归约放在 host，避免内核内原子。

---

## 7. 系统层：变长 / CP / graph / 后端

- **变长（varlen）**：`cu_seqlens` + `chunk_indices`（每块的 `(seq_id, chunk_id)`）
  + `chunk_offsets`（每序列的块起始）。所有 kernel 都有 `IS_VARLEN` 分支
  （如 `chunk_bwd.py:69-77`），用 `chunk_indices` 二次寻址，避免 pad 浪费。
- **上下文并行（CP）**：`fla/ops/cp/chunk_delta_h.py` 的 `_pre_process` /
  `_bwd_dhu_pre_process` 做跨 rank 状态 all-gather（`chunk_fwd.py:86-98`、
  `chunk_bwd.py:555-573`）。Megatron 用 `chunkwise`/`headwise` 两种 CP
  （`kda.py:214-231`），headwise 要求 zigzag 布局（`kda.py:256-270`）。
- **CUDA/NPU graph**：`use_graph=True` 用 `prepare_chunk_indices_static`
  生成**固定形状**块表（`chunk.py:65-86`），配合 `get_static_buffer` 复用持久缓冲
  （如 `chunk_bwd.py:352-353`），支持 graph capture / replay。
- **后端 dispatch**：`@dispatch('kda')` 把算子路由到不同实现——Triton、TileLang
  （`fla/ops/kda/backends/tilelang/`）、FlashKDA（`fla/ops/kda/backends/flash_kda.py`）。
  调试/对拍时用 `FLA_DISABLE_BACKEND_DISPATCH=1` 钉回 Triton（否则本机 0.5.x 会
  路由到 TileLang，fp32 路径可能编译失败）。

---

## 8. Megatron 侧的工程点

- **门控在 kernel 内**：Megatron 传 raw g（`kda.py:120-121`），移植时 §3.1/§3.2
  的激活与 cumsum 必须在 kernel 里。
- **`A_log`/`dt_bias` 保持 fp32**（`kda.py:117-118`），且 `dt_bias` 用
  逆-softplus 空间初始化（`kda.py:155-164`）。
- **`gdn_pre_gated_delta_rule_fusion` 对 KDA 未实现**（`kda.py:399-402`）——
  这是一处移植缺口：GDN 能把 pre-gate 融合，KDA 目前不能。
- **conv1d 用 FLA `causal_conv1d`**（`kda.py:598-607`），需 SiLU；
  确定性模式退回 `F.conv1d`（`kda.py:561-573`）。
- **确定性模式提供纯 torch 路径** `torch_chunk_gated_delta_rule`
  （`gdn.py:638-743`）——因为 FLA 的 Triton/atomic 路径不保证确定性。移植后
  若目标卡要求 bitwise 可复现，需要类似 fallback。
- **CP 的 padding 约束**：`gdn_conv_pad_alignment` 与 chunkwise CP 冲突
  （`kda.py:409-413`）。

---

## 9. 移植到目标卡的取舍建议

按「先正确、后快」排序：

1. **对齐契约**：先把 [02-backward.md](02-backward.md) §7 的检查表逐项过一遍
   （转置布局、`dht` 初值、`debug_barrier`、块内局部下标、反向前缀和、GVA 归约）。
2. **第一档（免费）**：`chunk_size` sweep（32/64）；`safe_gate`（开 TensorCore）；
   选择 `state_v_first` 省转置；把 `A_log/dt_bias` 保持 fp32。
3. **第二档（结构性）**：手写 kernel 时优先保证 **fwd_h/dhu 沿 $T$ 串行**这一支的
   访存效率（状态常驻寄存器、$V$ 方向并行度），因为它是前向延迟瓶颈；
   反向瓶颈在 `wy_dqkg`（占反向大头，见 `flash-kda/docs/07-perf.md` 的链级归因）。
4. **第三档（融合）**：`inter_solve` 融合、`wy_dqkg` 融合、门控融合已在 FLA 里做好，
   移植时**照抄其融合边界**即可，不要拆开。
5. **避免双重重算**：Megatron 层级 `recompute_gdn` 与 FLA 算子级
   `disable_recompute` 二选一，别叠加。

---

## 10. 一句话总结

KDA 优化的主线是 **「分块化把 $O(T)$ 串行压成 $O(T/C)$ 串行 + 块内小矩阵并行」**，
其余手段（融合、重算、log2/exp2、tf32 solve、safe_gate、autotune、varlen/CP/graph）
都是在不同维度上为这条主线服务；而数值稳定的红线只有一条——
**强衰减下不要直接算 $k/E$，要减参考点或用逐对 $2^{\Delta g}$**。
