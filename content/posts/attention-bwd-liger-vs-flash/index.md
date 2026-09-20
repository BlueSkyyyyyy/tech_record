---
title: "注意力反向的两条路线：Liger-Kernel 与 FlashAttention 反向源码对照"
date: 2026-09-20
draft: false
weight: 1
series: ["attention-kernel-compare"]
tags: ["attention", "backward", "triton", "cuda", "flash-attention", "liger-kernel", "算子开发"]
categories: ["算子开发"]
---

这里读的是两个开源仓库（下文所有路径均为仓库内相对路径）：

- **Liger-Kernel** 的 `src/liger_kernel/ops/fused_neighborhood_attention.py`（下称 **FNA**）。
- **FlashAttention** 的两代反向：
  - FA2（sm80）：`csrc/flash_attn/src/flash_bwd_kernel.h`、`flash_bwd_preprocess_kernel.h`、`flash_bwd_launch_template.h`
  - FA3（sm90/Hopper）：`hopper/flash_bwd_kernel_sm90.h`、`hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`、`hopper/flash_bwd_preprocess_kernel.h`、`hopper/flash_bwd_launch_template.h`

一个最直观的结论先摆在这里：

> **Liger FNA 是"教科书式可微分注意力算子"**——把 $N\times N$ 的中间矩阵算出来、存下来，逐条套用反向公式，每个公式一个 kernel。
> **FlashAttention 是"为 GPU 而生的执行引擎"**——前向只留一本每行的总账（$O$ 与 LSE），反向靠重计算把 $P$ 现场拼回来，再用流水线/张量核/TMA 把它压进片上存储。

本篇不重新推导数学（推导见[第 1 篇]({{< relref "flash-attention-01-theory" >}})与[第 4 篇]({{< relref "flash-attention-04-bwd" >}})），而是聚焦：**代码逻辑、调用关系、关键代码、以及两者各自用了哪些优化手段、为什么这么写**。

---

## 0. 先看全景：同一条链，两种走法

设（省略 batch/head 下标）

$$
S=\sigma QK^\top,\qquad P=\mathrm{softmax}_{\text{row}}(S),\qquad O=PV,\qquad dO=\partial L/\partial O,\qquad \sigma=1/\sqrt d .
$$

反向的五条主干公式是：

$$
\begin{aligned}
dV &= P^\top dO \\
dP &= dO\,V^\top \\
dS &= P\circ\big(dP-\Delta\,\mathbf{1}^\top\big),\qquad \Delta_i=\sum_k P_{ik}dP_{ik} \\
dQ &= \sigma\, dS\,K \\
dK &= \sigma\, dS^\top Q
\end{aligned}
$$

两条路线对这五条公式的执行方式，可以用一个"审计"的比喻说清楚：

- **Liger FNA** 像一个**留有底稿的会计**：前向把整本 $P$（`attn_weights`）完完整整存档；反向时把底稿摊开，一条条翻查。省心，但底稿占满整张桌子（显存 $O(N^2)$）。
- **FlashAttention** 像一个**只记日报的会计**：前向只留下每行的汇总（$O$ 和 LSE），把 $P$ 碎纸机处理掉；反向时靠 $O$、LSE、$Q$、$K$、$dO$ **现场重算** $P$。省地方，但要多算一遍。

---

## 1. Liger FNA：调用关系与代码逻辑

### 1.1 模块层调用链

```
LigerFusedNeighborhoodAttention(nn.Module)          # transformers/fused_neighborhood_attention.py
        │  q/k/v_proj → reshape [B,H,N,D]
        ▼
LigerFusedNeighborhoodAttentionFunction.apply        # ops/fused_neighborhood_attention.py
        │      torch.autograd.Function
        ├── forward(ctx, q,k,v,kernel_size,dilation,scale)
        │        └── fused_neighborhood_attention_forward(...)
        │              ├── _neighborhood_mask_kernel                生成 [N,N] 邻域掩码
        │              ├── _fused_neighborhood_attention_qk_kernel  QK^T * scale + mask
        │              ├── _softmax_forward                         softmax.py  行 softmax
        │              └── _fused_neighborhood_attention_av_kernel  Attn @ V
        │        ctx.save_for_backward(query, key, value, attn_weights)   ★ 存下 N×N 的 P
        │
        └── backward(ctx, grad_output)
                 ├── _fused_neighborhood_attention_grad_attn_kernel  dP = dO @ V^T
                 ├── _softmax_backward                              softmax.py  dS = P∘(dP−Δ)
                 ├── _fused_neighborhood_attention_grad_qk_kernel    dQ = scale * dS @ K
                 ├── _fused_neighborhood_attention_grad_k_kernel     dK = scale * dS^T @ Q
                 └── _fused_neighborhood_attention_grad_v_kernel     dV = P^T @ dO
```

### 1.2 前向的数据流（4 个 kernel，2 张 $N\times N$ 大图）

```
Q,K [B,H,N,D] ──► mask kernel ──► mask [N,N]
                       │
Q,K ──► QK kernel ─────┴──► qk_scores [B,H,N,N]   (dtype = 输入 dtype)
                                   │
                              softmax_forward
                                   ▼
                            attn_weights [B,H,N,N]  ← ctx.save_for_backward ★
                                   │
V ──► AV kernel ───────────────────┴──► output [B,H,N,D]
```

两个关键事实：

1. **`qk_scores`、`attn_weights` 都是 `[B,H,N,N]` 的稠密张量**。邻域窗口（`kernel_size`）只是把窗口外的 score 置成 `-inf`，**并没有让计算或存储变稀疏**——它虽然叫 neighborhood attention，实际仍是 $O(N^2)$ 的算力与显存。
2. **反向要用到的 $P$ 被完整保存**，这是反向内存开销的根源：

```python
output, attn_weights, softmax_params = fused_neighborhood_attention_forward(
    query, key, value, kernel_size, dilation, scale)
ctx.save_for_backward(query, key, value, attn_weights)   # ★ [B,H,N,N] 的 P
```

### 1.3 反向的数据流（5 个 kernel，又多了 2 张 $N\times N$ 大图）

```
dO [B,H,N,D] , V
      │
      ▼  grad_attn kernel
 grad_attn [B,H,N,N]   (= dP)
      │            , attn_weights (P)
      ▼  softmax_backward
 grad_qk   [B,H,N,N]   (= dS)
      │
      ├──► grad_qk kernel + K ──► grad_Q [B,H,N,D]
      ├──► grad_k  kernel + Q ──► grad_K [B,H,N,D]
      └──► grad_v  kernel + dO ─► grad_V [B,H,N,D]
```

每个反向 kernel 独立 launch，且每个都要从 HBM 完整读回一张 $N\times N$ 矩阵。以 $B{=}1,H{=}16,N{=}4096$ 为例，一张 `[B,H,N,N]` 的 fp16 矩阵约 $16\times4096^2\times2\approx512$ MB。前向写、反向读写反复往返，量级是几十 GB。

### 1.4 关键代码：反向就是"公式直译"

`softmax_backward` 是整个反向的枢纽，两行就是 $dS=P\circ(dP-\Delta)$：

```python
@triton.jit
def _softmax_single_block_backward_kernel(dy_ptr, dy_stride, y_ptr, y_stride,
                                          dx_ptr, dx_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_cols
    dy = tl.load(dy_ptr + row_id * dy_stride + offs, mask=mask, other=0.0)
    y  = tl.load(y_ptr  + row_id * y_stride  + offs, mask=mask, other=0.0)
    dot = tl.sum(dy * y, axis=0)      # Δ_i = Σ_j P_ij·dP_ij
    dx  = y * (dy - dot)              # dS  = P ∘ (dP − Δ)
    tl.store(dx_ptr + row_id * dx_stride + offs, dx, mask=mask)
```

而三个梯度 kernel 也都是单条 GEMM 的直译，例如 $dV=P^\top dO$：

```python
# _fused_neighborhood_attention_grad_v_kernel
attn_chunk     = tl.load(attn_ptrs,     mask=attn_mask,     other=0.0)      # P
grad_out_chunk = tl.load(grad_out_ptrs, mask=grad_out_mask, other=0.0)      # dO
acc += tl.dot(tl.trans(attn_chunk), grad_out_chunk)                          # dV += Pᵀ dO
```

### 1.5 澄清：Liger 有 FlashAttention 反向吗？

**目前 Liger-Kernel 主干没有标准稠密注意力的 FlashAttention 反向。** 遍历 `src/liger_kernel/ops/` 与全部 remote 分支，注意力系（attention-family）算子只有三类：

| 算子 | 文件 | 反向形态 |
|---|---|---|
| Fused Neighborhood Attention | `ops/fused_neighborhood_attention.py` | 物化 $P,dP$，逐公式 kernel |
| Multi-Token Attention | `ops/multi_token_attention.py` | softmax/sparsemax + conv2d 反向 |
| Attention Residuals（按深度做 softmax） | `ops/attn_res.py` | 单 kernel 融合，$N\le16$ 在寄存器 |

它们的反向无一使用 online-softmax 重计算。以 Multi-Token Attention 为例，其 softmax 反向和 FNA 是同一个模式（只是外圈换成 conv）：

```python
# LigerMultiTokenAttentionFunction.backward
grad_probs = F.conv_transpose2d(grad_conv, weight, None, ...)
dot = (grad_probs * activation_output).sum(dim=-1, keepdim=True)
grad_scores_inf = activation_output * (grad_probs - dot)   # 同样是 P∘(dP−Δ)
grad_scores = _mask_inf_backward(grad_scores_inf)
```

所以下文的"FlashAttention 反向"特指 FlashAttention 仓库的实现；Liger 侧的代表就是 FNA。

---

## 2. FlashAttention：调用关系与代码逻辑

### 2.1 FA2（sm80）调用关系

```
run_mha_bwd_hdim{32..256}                            # flash_bwd_launch_template.h
        └── run_flash_bwd
              └── run_flash_bwd_seqk_parallel
                    ├── flash_bwd_dot_do_o_kernel    → compute_dot_do_o
                    │        · dPsum = rowsum(dO ⊙ O)         ★ Δ 恒等式
                    │        · 清零 dQaccum
                    │
                    ├── flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel
                    │        └── compute_dq_dk_dv_seqk_parallel
                    │              └── compute_dq_dk_dv_1colblock   ← 核心循环
                    │                   · 每 CTA 认领一个 n_block
                    │                   · for m_block: max-1 → min
                    │                       重算 S → P → dP → dS → dQ/dK/dV
                    │
                    └── flash_bwd_convert_dq_kernel  → convert_dQ
                             · dQaccum(fp32) 多 split 求和 → dQ(fp16/bf16)
```

FA2 的 grid 是 `(num_n_block, batch, head)`：**一个 CTA 负责一个 K/V 块，然后沿 Q 方向倒着扫**。这个方向选择不是随意的，见 §3。

### 2.2 FA3（sm90）调用关系

```
run_flash_bwd
        ├── PreprocessKernel (FlashAttnBwdPreprocess)
        │        · dPsum = rowsum(dO ⊙ O)  →  gmem
        │        · LSE_log2 = LSE * log2(e) → gmem
        │        · 清零 dQaccum(fp32)；初始化 dq_semaphore (deterministic)
        │
        ├── AttnKernel (FlashAttnBwdSm90)
        │        ├── Producer warpgroup (1 WG)
        │        │      · TMA 预取 Q / dO / LSE / dPsum（多 stage pipeline）
        │        │      · TMA 载入 K / V + barrier_KV
        │        │      · store_dq：smem 里的 dQacc 用 TMA bulk reduce-add 写回 gmem
        │        └── Consumer warpgroups (2–3 WG)
        │               └── mainloop.mma()
        │                     └── bwd_step()   ← 核心循环
        │                         · 重算 S=QK^T、P=exp2(...)
        │                         · dP = dO V^T；dS = P∘(dP−Δ)
        │                         · dV += Pᵀ dO, dK += dSᵀ Q, dQ += dS K
        │               └── epilogue.store()：dK/dV (fp32) → TMA store / GQA 累加
        │
        └── PostprocessKernel (FlashAttnBwdPostprocessConvertdQ)
                 · dQaccum(fp32) → dQ(fp16/bf16)
                 · GQA 时：dK_accum/dV_accum → dK/dV
```

FA3 是**持久化内核 + producer/consumer warp specialization**：一组 warp 专门搬数据（TMA），另一/两组专门做 MMA（WGMMA）。

### 2.3 与前向的配合：FA 到底"存"了什么

| | 前向保存 | 反向需要 |
|---|---|---|
| Liger FNA | $Q,K,V,P$（$P$ 是 $O(N^2)$） | $P,dO$ |
| FA2 | $Q,K,V,O,\text{LSE}$（都是 $O(N)$） | $O,\text{LSE}$ 用来重算 $P$ |
| FA3 | $Q,K,V,O,\text{LSE}_{\log 2},dP_{sum}$ | 同上，LSE 已预先乘 `log2(e)` |

---

## 3. 反向优化手段逐项拆解（附关键代码）

### 3.1 优化一：不存 $P$，改用"总账 + 重算"

只存 LSE，反向用 $P_{ij}=\exp(S_{ij}-\text{LSE}_i)$ 现场重算。为避开 `exp`，前向把 LSE 定义在 base-2 上，`softmax_scale_log2 = scale * log2(e)` 提前折进去：

```cpp
// FA2 flash_bwd_kernel.h
FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(
    scores, lse, params.scale_softmax_log2);          // P = exp2(S*scale_log2 − LSE)
```

```cpp
// FA3 mainloop_bwd_sm90_tma_gmma_ws.hpp
scores(mi, ni) = exp2f(scores(mi, ni) * params.softmax_scale_log2 - lse_scaled);
```

### 3.2 优化二：$\Delta$ 的"恒等式"——连 $P$ 都不用就能算 $\Delta$

$$
\Delta_i=\sum_k P_{ik}dP_{ik}=\sum_k P_{ik}\Big(\sum_d dO_{id}V_{kd}\Big)=\sum_d dO_{id}O_{id}=\text{rowsum}(dO\odot O).
$$

所以 $\Delta$ 只需要 $dO$ 和 $O$（都是 $O(N)$ 量级）。FA2 的独立 preprocess kernel：

```cpp
// flash_bwd_preprocess_kernel.h : dot_do_o
float dP_sum_cur = do_fp32(mi, 0) * o_fp32(mi, 0);
for (int ni = 1; ni < size<1>(do_reshaped); ni++)
    dP_sum_cur += do_fp32(mi, ni) * o_fp32(mi, ni);
dP_sum_cur = Allreduce<THREADS_PER_ROW>::run(dP_sum_cur, sum_op) * scale;
```

FA3 的 preprocess 顺带把 LSE 换成 base-2：

```cpp
// hopper/flash_bwd_preprocess_kernel.h
dP_sum(mi) = Allreduce<kGmemThreadsPerRow>::run(dP_sum_cur, sum_op);
...
gLSElog2(thread_idx) = lse == -INFINITY ? 0.f : lse * float(M_LOG2E);
```

**Liger 对照**：它用 $P$ 算 —— `dot = tl.sum(dy * y)`（见 §1.4），因为有 $P$，这条恒等式对它没必要。

### 3.3 优化三：dQ 的跨块归约——"多本账"怎么合并

三条梯度的求和方向是：

$$
dV_j=\sum_i P_{ij}dO_i,\qquad dK_j=\sigma\sum_i dS_{ij}Q_i,\qquad dQ_i=\sigma\sum_j dS_{ij}K_j .
$$

$dV_j,dK_j$ 沿 **Q 的行维 $i$** 归约，是 CTA 内累加；$dQ_i$ 沿 **KV 的列维 $j$** 归约，要跨 K/V 块。FlashAttention 选择"固定 K/V 块、循环 Q"，把跨块归约全甩给 **dQ**。

FA2 把 `acc_dq` 以 **fp32 的 `dQaccum`** 写回全局，非 deterministic 用 `atomicAdd`，deterministic 写独立 split：

```cpp
// flash_bwd_kernel.h
if (Is_first || Seq_parallel) {
    clear(acc_dq);
} else {
    // 先读回本块已累加的 dQaccum（fp32）
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, acc_dq_reshaped);
}
...
if (!Seq_parallel) {
    cute::copy(gmem_tiled_copy_dQaccum, acc_dq_reshaped, tdQgdQaccum);   // 覆盖写
} else {
    for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
}
```

`convert_dQ` 再做 split 求和与 fp32→fp16：

```cpp
// flash_bwd_preprocess_kernel.h
for (int s = 0; s < nsplits; ++s) {
    cute::copy(gmem_tiled_copy_dQaccum, tdQgdQaccum, tdQrdQaccum);
    for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) += tdQrdQaccum(i); }
    tdQgdQaccum.data() += params.dq_accum_split_stride;
}
for (int i = 0; i < size(acc_dq); ++i) { acc_dq(i) *= params.scale_softmax_rp_dropout; }
Tensor rdQ = convert_type<Element>(acc_dq);   // fp32 → fp16
```

FA3 更硬核：MMA 把 dQ 先写 **shared memory**，再用 **TMA `SM90_BULK_REDUCE_ADD`** 硬件归约写回全局，省掉 atomic 往返：

```cpp
// mainloop_bwd_sm90_tma_gmma_ws.hpp : store_dq
SM90_BULK_REDUCE_ADD::copy(raw_pointer_cast(sdQ(_, warpgroup_idx).data()),
                           raw_pointer_cast(gdQaccum(_, warpgroup_idx, m_block).data()),
                           dQ_TMA_num_bytes, static_cast<uint64_t>(TMA::CacheHintSm90::EVICT_LAST));
```

### 3.4 优化四：循环方向与"谁能留在寄存器里"

FA2 的 m 循环**倒着**走，注释说能省 1 个寄存器；而 `acc_dk, acc_dv` 整个 m 循环都驻留寄存器，循环结束才写回。核心循环骨架（删减版）：

```cpp
// flash_bwd_kernel.h : compute_dq_dk_dv_1colblock
clear(acc_dv); clear(acc_dk);
for (; m_block >= m_block_min; --m_block) {
    clear(acc_s);
    gemm(acc_s, tSrQ, tSrK, ...);                                  // S = Q K^T
    // ... 掩码 ...
    scale_apply_exp2(scores, lse, params.scale_softmax_log2);      // P = exp2(S·s − LSE)

    clear(acc_dp);
    gemm(acc_dp, tdPrdO, tdPrV, ...);                              // dP = dO V^T
    for (mi,ni) dS(mi,ni) = scores(mi,ni) * (dS(mi,ni) - dP_sum(mi));   // dS = P∘(dP−Δ)

    // dQ：跨块归约的累加器，写 dQaccum
    clear/load acc_dq;
    gemm(acc_dq, tdQrdS, tdQrK, ...);                              // dQ += dS K
    write dQaccum (copy 或 atomicAdd);

    // dV、dK：整段 m 循环在寄存器里累加
    gemm(acc_dv, tdVrPt, tdVrdO, ...);                             // dV += P^T dO
    gemm(acc_dk, tdKrdSt, tdKrQt, ...);                            // dK += dS^T Q
}
// Epilogue：dK,dV (fp32) → fp16，写回 gmem
```

FA3 更进一步，当 `Mma_dKV_is_RS` 成立时，$P$/$dS$ **直接从寄存器喂给 dV/dK 的 WGMMA**，连 shared memory 都省了：

```cpp
// mainloop_bwd_sm90_tma_gmma_ws.hpp : bwd_step
if constexpr (Mma_dKV_is_RS) {
    Tensor tdVrP  = make_tensor(rP.data(),  convert_layout_acc_Aregs<TiledMmadKV>(tSrS.layout()));
    flash::gemm</*zero_init=*/false>(tiled_mma_dKV, tdVrP, tdVrdO(...), tdVrdV);
    ...
    Tensor tdKrdS = make_tensor(rdS.data(), convert_layout_acc_Aregs<TiledMmadKV>(tdPrdP.layout()));
    flash::gemm</*zero_init=*/false>(tiled_mma_dKV, tdKrdS, tdKrQ(...), tdKrdK);
}
```

### 3.5 优化五：访存与流水线

**FA2（cp.async + double buffer）**：`Q` 双缓冲随 m_block 奇偶切换；`dO` 用 `cp_async_fence/cp_async_wait` 异步预取；`V in regs` 选项把 V 提到寄存器；并用 `make_tiled_copy_B_warpcontiguousN` / `make_tiled_copy_C_warpcontiguousN` 重排 copy layout，让 `P`/`dS` 的写入方向也能合并访存。

**FA3（TMA + mbarrier + swizzle）**：Q/dO/K/V/LSE/dPsum 全部走 `SM90_TMA_LOAD`，地址与 swizzle 交给硬件；Q 与 dO 用两条独立 pipeline；同一块 smem 用 `is_position_independent_swizzle_tensor` 按不同 layout 视图复用——**转置是"零成本"的，只是换个 layout 解释同一片内存**：

```cpp
Tensor sQ  = make_tensor(make_smem_ptr(smem_q),  SmemLayoutQ{});
Tensor sQt = make_tensor(make_smem_ptr(smem_q),  SmemLayoutQt{});   // 同一片内存的转置视图
...
Tensor tdVrdO = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sdOt);  // 供 dV = PᵀdO
Tensor tdKrQ  = mma_partition_fragment_AB</*A=*/dKV_swapAB>(wg_mma_dKV, sQt);   // 供 dK = dSᵀQ
```

### 3.6 优化六：warp specialization 与寄存器重分配

FA3 把 CTA 拆成 1 个 producer warpgroup（搬数据）和 2–3 个 consumer warpgroups（算 MMA），并用动态寄存器调配绕开寄存器墙：

```cpp
// flash_bwd_kernel_sm90.h
using NumMmaWarpGroups = ...;
static constexpr uint32_t LoadRegisterRequirement = NumMmaWarpGroups == 2 ? 24 : 32;
static constexpr uint32_t MmaRegisterRequirement  = NumMmaWarpGroups == 2 ? 240 : 160;

if (warp_group_idx == 0) {                       // Producer
    cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    ... TMA load ...
} else {                                          // Consumer
    cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();
    ... mainloop.mma(...) ...
}
```

另有三处寄存器压力优化：`ShuffleLSE`/`ShuffledPsum`（LSE 分散到 8 线程 + `__shfl_sync`）、`SeparateMaskingIterations`（hdim≤64 把掩码迭代单独拆出）、`Slice_dQKV_Mma`（hdim=256 把 dQ 的 MMA 切两半）。

### 3.7 优化七：数值稳定性与精度

| 技巧 | FlashAttention | Liger FNA |
|---|---|---|
| exp 用 base-2 | `exp2f(x*scale_log2 - lse)` | 普通 `tl.exp` |
| LSE/Δ 预计算 | preprocess kernel | 在 softmax kernel 里算 |
| 累加类型 | fp32（`dQaccum`, `acc_*`） | `tl.zeros(..., fp32)` 累加，但**存回是输入 dtype** |
| dropout | 用符号位编码避免额外乘法 | 不支持 |
| softcap | `dtanh` 前向/反向各算一次 | 不支持 |
| 确定性 | 独立 split + semaphore / TMA reduce | 单 kernel 串行，天然确定但慢 |

一个值得强调的差异：Liger 把 $P$ 以输入精度（fp16/bf16）存进显存，反向 $dS$ 也在低精度下计算；FA 在寄存器里用 `exp2f` 得到 fp32 score 再转 fp16 供 MMA，$\Delta$ 用 fp32 累加。长序列下 Liger 的中间精度损失更明显。

### 3.8 优化八：针对特殊形态的分支

- **GQA/MQA**：FA3 的 `CollectiveEpilogueBwdGQA` 把多个 q-head 的 dK/dV 累加到 `dK_accum/dV_accum`，再统一 postprocess。
- **varlen**：变长序列用 `cu_seqlens` 定位，避免 padding。
- **local/sliding window**：掩码迭代单独处理（`apply_mask_local`）。
- **split-K / seqlen 并行**：FA2 的 `run_flash_bwd_seqk_parallel` 把 K 维切到多个 CTA，提升"小 Q 大 K"场景的并行度。

---

## 4. 一张总表：同一条链，两种工程哲学

| 维度 | Liger FNA | FA2 (sm80) | FA3 (sm90) |
|---|---|---|---|
| 反向 kernel 数 | 5 个（各管一条公式/一段） | 3 类（preprocess / main / convert） | 3 类（preprocess / main / postprocess） |
| $P$ 是否保存 | ✅ 存 $N\times N$ | ❌ 重算 | ❌ 重算 |
| 显存复杂度 | $O(N^2)$ | $O(N)$ | $O(N)$ |
| $\Delta$ 来源 | $P\cdot dP$（用 $P$） | $\text{rowsum}(dO\odot O)$ | $\text{rowsum}(dO\odot O)$ |
| dQ 归约 | 不需要（$dS$ 全存） | fp32 `dQaccum` + atomic / split | smem `dQacc` + TMA bulk reduce-add / atomic |
| 数据搬运 | 每 kernel 全量 $N^2$ HBM 往返 | cp.async 双缓冲 | TMA 多 stage 流水线 |
| 并行组织 | 简单 grid，独立 launch | 每 CTA 一个 K/V 块 | 持久化 + producer/consumer WG |
| 张量核 | Triton `tl.dot` | mma / wgmma | WGMMA + rs/ss |
| 转置 | 物化 + `tl.trans` | smem layout | 同 smem 换 layout 视图 |
| 寄存器优化 | 交给编译器 | V-in-regs | 寄存器重分配 + ShuffleLSE + RS-MMA |
| 可读性 | ★★★★★ | ★★ | ★ |
| 适用规模 | 小 $N$ / 教学 | 生产（A100 及以下） | 生产（H100） |

---

## 5. 为什么 Liger 要这么写？什么时候该用哪种？

**Liger FNA 的写法不是"错"，而是定位不同**：

1. **可微分性清晰**：每个公式一个 kernel，`torch.autograd.Function` 的 `forward/backward` 一目了然，便于和参考实现做数值对齐。
2. **对任意 mask 友好**：预生成 mask + 全量 softmax，可轻松表达邻域、dilation 等结构化稀疏，无需为每种 mask 写特化内核。
3. **Triton 声明式**：不碰 TMA/WGMMA/mbarrier，开发者门槛低。

但它的天花板很清楚：**$O(N^2)$ 的显存与 HBM 带宽会先于算力成为瓶颈**，长序列直接 OOM。

**FlashAttention 的写法是"把每一分带宽都当成敌人"**：前向只用 $O(N)$ 的账本，反向靠重算换显存；把 dQ 的跨块归约当头号问题，用 fp32 accum / split / TMA reduce-add 三套方案解决；用 TMA、warp specialization、RS-MMA、寄存器重分配把 Hopper 的硬件能力榨干。代价是代码极度复杂，且每种架构、每种 head dim、每种 mask 都要特化。

**一句话选择指南**：

> 序列长度几千以内、想快速验证一个新注意力变体的梯度，用 FNA 这种"物化 + 公式化"的写法最快最稳；
> 一旦序列上到万级、要进生产训练，就必须走 FlashAttention 这条"重算 + 片上归约 + 流水线"的路。

---

## 6. 附：关键代码位置速查（仓库内相对路径）

**Liger-Kernel**

| 功能 | 位置 |
|---|---|
| autograd Function | `src/liger_kernel/ops/fused_neighborhood_attention.py` |
| softmax fwd / bwd | `src/liger_kernel/ops/softmax.py` |
| nn.Module 封装 | `src/liger_kernel/transformers/fused_neighborhood_attention.py` |
| Multi-Token Attention 反向 | `src/liger_kernel/ops/multi_token_attention.py` |

**FlashAttention FA2 (sm80)**

| 功能 | 位置 |
|---|---|
| 反向主循环 | `csrc/flash_attn/src/flash_bwd_kernel.h` |
| dPsum 预计算 / dQaccum→dQ | `csrc/flash_attn/src/flash_bwd_preprocess_kernel.h` |
| host 调度 | `csrc/flash_attn/src/flash_bwd_launch_template.h` |

**FlashAttention FA3 (sm90)**

| 功能 | 位置 |
|---|---|
| 反向主循环（bwd_step） | `hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp` |
| kernel 入口（warp 角色） | `hopper/flash_bwd_kernel_sm90.h` |
| preprocess | `hopper/flash_bwd_preprocess_kernel.h` |
| host 调度 | `hopper/flash_bwd_launch_template.h` |

> 注：Liger-Kernel 目前没有标准稠密注意力的 FlashAttention 反向；其"注意力反向"代表是 FNA（见 §1.5）。若需要在 Liger 中做长序列注意力训练，通常直接调用 FlashAttention / SDPA，而非 Liger 自有 kernel。
