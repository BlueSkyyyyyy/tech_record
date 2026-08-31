---
title: "Flash Attention 精读（四）：反向梯度推导与 recompute 实现精读"
date: 2026-08-28
draft: false
weight: 4
tags: ["flash-attention", "triton", "cuda", "attention", "autograd", "系列"]
categories: ["算子开发"]
---

前向消灭了 $N \times N$ 的中间矩阵，代价落在反向：**梯度计算需要 $P$，而 $P$ 没有被保存**。本篇先完整推导 $dQ, dK, dV$（三个实现共享同一套数学），再逐一精读 Triton 教程版（`06-fused-attention.py` L410–748）和 FlashAttention-2 CUDA 版（`csrc/flash_attn/src/flash_bwd_kernel.h`）的反向实现，最后对照三种不同的 kernel 组织策略。

前置阅读：[第 1 篇]({{< relref "flash-attention-01-theory" >}})的 LSE 定义、[第 2 篇]({{< relref "flash-attention-02-triton-fwd" >}})前向的 epilogue。

## 1. 完整的梯度推导

记 $S = \sigma QK^\top$（$\sigma = 1/\sqrt d$），$P = \mathrm{softmax}_{\mathrm{row}}(S)$，$O = PV$，上游梯度 $dO \equiv \partial L/\partial O$。

### 1.1 dV：一个普通 GEMM

对 $O = PV$ 的 $V$ 求导不经过 softmax：

$$
\boxed{\ dV = P^\top dO\ }
$$

### 1.2 softmax 的雅可比：$dS = P \circ (dP - \Delta \mathbf{1}^\top)$

$dP = dO\,V^\top$ 之后，要穿过行 softmax。对第 $i$ 行，$P_{ij} = e^{S_{ij}} / \sum_k e^{S_{ik}}$，求雅可比：

$$
\frac{\partial P_{ij}}{\partial S_{ik}} = P_{ij}(\delta_{jk} - P_{ik})
$$

于是

$$
dS_{ij} = \sum_k \frac{\partial L}{\partial P_{ik}} \frac{\partial P_{ik}}{\partial S_{ij}}
= \sum_k dP_{ik} P_{ik}(\delta_{jk} - P_{ij})
= P_{ij}\left(dP_{ij} - \underbrace{\sum_k P_{ik}\, dP_{ik}}_{\Delta_i}\right)
$$

$$
\boxed{\ dS = P \circ (dP - \Delta \mathbf{1}^\top), \qquad \Delta_i = \sum_k P_{ik} dP_{ik}\ }
$$

直觉读法：$dS_{ij}$ 的第一项 $P_{ij} dP_{ij}$ 是"这条路自己"的贡献，第二项 $-P_{ij}\Delta_i$ 是"整行分母被拉动"的贡献——softmax 把概率质量在行内搬运，拉高任何一个 $S_{ij}$ 都会压低整行其他位置。

### 1.3 $\Delta$ 恒等式：整个反向最划算的一笔交易

$\Delta_i = \sum_k P_{ik} dP_{ik}$ 看起来需要完整的 $P$ 行（$O(N)$ 每行、$O(N^2)$ 总量）。但代入 $dP_{ik} = dO_i \cdot V_k$：

$$
\Delta_i = \sum_k P_{ik} (dO_i \cdot V_k) = dO_i \cdot \underbrace{\sum_k P_{ik} V_k}_{= O_i} = \mathrm{rowsum}(dO \circ O)
$$

$$
\boxed{\ \Delta_i = dO_i \cdot O_i\ }
$$

**$O(N \cdot d)$ 的逐行点积替代了 $O(N^2)$ 的重算**。$O$ 和 $dO$ 都是前向/上游本来就有的矩阵，$\Delta$ 可以在反向开始前用一个极廉价的 kernel 一次性算好。这就是所有实现里那个 "preprocess kernel" 的全部数学内涵。

### 1.4 dQ 与 dK

穿过 $S = \sigma QK^\top$（$\sigma$ 是标量）：

$$
\boxed{\ dQ = \sigma\, dS\,K, \qquad dK = \sigma\, dS^\top Q\ }
$$

五个公式集齐。注意它们的**求和方向**：$dV_j = \sum_i P_{ij}\, dO_i$（对 Q 求和）、$dK_j = \sigma\sum_i dS_{ij} Q_i$（对 Q 求和）、$dQ_i = \sigma \sum_j dS_{ij} K_j$（对 KV 求和）。**dK/dV 天然沿 Q 维归约，dQ 天然沿 KV 维归约**——这个方向差异决定了反向 kernel 的组织方式，是本篇的主线。

### 1.5 P 的重算：LSE 一行搞定

反向需要 $P$，但前向只存了每行的 $\mathrm{LSE}_i = m_i + \log \ell_i$（第 1 篇埋的钩子）。重算 $S_{ij}$ 后：

$$
P_{ij} = e^{S_{ij} - \mathrm{LSE}_i}
$$

分子分母一次到位——**反向不需要重跑 online softmax，也不需要单独存 $\ell$**。重算多付出的只是又一次 $QK^\top$ GEMM + 一次 exp；换来 $O(N)$ 显存和数值上更干净的 $P$（没有 online rescale 的中间舍入）。所有实现都把 $1/\ln 2$ 折进 scale，让这一步直接用硬件 exp2 指令。

## 2. Triton 教程版反向：零原子的优雅拆分

### 2.1 `_attn_bwd_preprocess`（L410–433）：Δ 预处理

```python
o = tl.load(O_ptrs)
do = tl.load(DO_ptrs).to(tl.float32)
delta = tl.sum(o * do, axis=1)     # Δ = rowsum(dO ∘ O)
```

每个 program 处理 `[BLOCK_M=128, HEAD_DIM]` 的行块，$\Delta$ 存为 `[Z, H, N_CTX]`。必须独立成 kernel 的原因：$\Delta$ 被 dK/dV 和 dQ 两个方向**按行反复消费**，预先算好一个 $[N]$ 向量，比在任何 inner loop 里现算便宜一个数量级。

### 2.2 base-2 域与 K 的预缩放（L862–868）

```python
RCP_LN2 = 1.4426950408889634
arg_k = k * (ctx.sm_scale * RCP_LN2)     # σ/ln2 预乘进 K
```

前向存的 $M_i$ 是以 2 为底的 LSE。反向把 $\sigma/\ln 2$ **预乘进 K 本身**，任何 `tl.dot(q, k)` 的结果天然就是 $S' = \frac{\sigma}{\ln 2} QK^\top$，于是循环体内一行：

```python
p = tl.math.exp2(qk - m)        # L569 / L492：精确 softmax 概率
```

循环体内看不到任何 sm_scale 乘法——全部被折叠进 K 的预缩放和出口处的一次性修正（§2.5）。每块逐元素乘法 × 迭代次数，换成每 tile 出口一次乘法。

### 2.3 `_attn_bwd_dkdv`（L440–517）：输出所有权决定遍历方向

**设计原则：谁的输出，谁独占 tile。** $dK_j = \sigma\sum_i dS_{ij} Q_i$ 沿 Q 归约，因此每个 program 独占一个 K/V 块（`start_n = pid * BLOCK_N1`，L641），沿 M 维内层循环累加，dk/dv 是寄存器累加器，最后一次性写回：

```python
qkT = tl.dot(k, qT)                        # L491 (σ/ln2)·K Qᵀ
pT  = tl.math.exp2(qkT - m[None, :])       # L492 Pᵀ = exp2(S' − M)：LSE 一步恢复
if MASK: pT = tl.where(mask, pT, 0.0)
dv  += tl.dot(pT.to(tl.float16), do)       # L503 dV += Pᵀ @ dO
dpT = tl.dot(v, tl.trans(do)).to(tl.float32)   # L507 dPᵀ = V dOᵀ
dsT = pT * (dpT - Di[None, :])             # L508 dSᵀ = Pᵀ ∘ (dPᵀ − Δ)   ← §1.2 的公式
dk  += tl.dot(dsT.to(tl.float16), tl.trans(qT)) # L510 dK += dSᵀ @ Q
```

L508 就是 $dS = P \circ (dP - \Delta)$ 的转置形态；注意 `qT` 是**未缩放**的原始 Q（scale 的账在出口算）。`Di`（$\Delta$）和 `m`（LSE）按 Q 行加载，随行块推进步进（L487, L513–515）。

### 2.4 `_attn_bwd_dq`（L524–588）：对偶结构

对偶地，$dQ_i = \sigma \sum_j dS_{ij} K_j$ 沿 KV 归约——program 独占一个 Q 块，沿 N 维循环：

```python
qk = tl.dot(q, kT)                     # L568 kT 来自预缩放的 arg_k
p  = tl.math.exp2(qk - m)              # L569
dp = tl.dot(do, vT).to(tl.float32)     # L578 dP = dO Vᵀ
ds = p * (dp - Di[:, None])            # L579 dS = P ∘ (dP − Δ)
dq += tl.dot(ds.to(tl.float16), tl.trans(kT))   # L581 dQ += dS @ K（K 含 σ/ln2！）
```

causal 调度沿用前向的 off-band/on-band 思想但方向更讲究：从右边界 `end_n = start_m + BLOCK_M2` **反向界定**，先用 `MASK_BLOCK_N2`（对角带）带 mask 地走完，再大步无 mask 扫剩余部分（L715–741）。

### 2.5 出口处的 scale 修正与"一鱼两吃"

三个出口，两笔账（L690, L747）：

- **dK 乘 `sm_scale`**：dkdv 里 L510 用的是未缩放的 Q，结果 = $\sum_i dS_{ij} Q_i$，乘 $\sigma$ 修正；
- **dQ 乘 `LN2`**：L581 用的是**预乘了 $\sigma/\ln 2$ 的 K**，结果 = $\frac{\sigma}{\ln 2}\sum_j dS_{ij} K_j$，乘 $\ln 2$ 消掉多余的 $1/\ln 2$；
- **dV 不修正**：$P^\top dO$ 不过 $S$ 的链式导。

还有一个 grid 复用技巧（L610–613 注释）：`program_id(0)` 同时作为 dK/dV 的 `start_n` 和 dQ 的 `start_m`——这要求 `BLOCK_N1 == BLOCK_M2 == 128`（L860），grid 对两部分同时合法。一个 kernel 双产出，省一次 launch，代价是寄存器压力叠加（num_warps=4, num_stages=5）。

**零原子操作**是这套设计的招牌：每个输出 tile 唯一属主、寄存器累加、一次写回（L686–692, L746–748），确定性好、无 atomic 流量。代价是反向要求 `N_CTX % 128 == 0`（L872 assert），变长序列需另行处理。

### 2.6 反向的精度纪律

梯度是大量正负项相消的结果，比前向敏感得多。教程版的策略一句话：**"边界 fp32、tensor core 操作数 fp16、硬件累加 fp32"**——

- dk/dv/dq 三个累加器全程 fp32；$P$ 在 fp32 域算出（`exp2`），只在进 dot 前降精；
- `dp = tl.dot(...).to(tl.float32)`：$dP - \Delta$ 是两个大数相消，fp16 会直接损失有效位，所以 dP 显式转 fp32 后才做减法；
- preprocess 里 dO 强制 `.to(tl.float32)`，$\Delta$ 全程 fp32。

降精只发生在 tensor core 强制要求的"操作数"位置。测试容差 `atol=1e-2`（fp32 参考实现、softmax 强制 fp32 计算）验证了这套纪律够用；FP8 反向干脆不支持。

## 3. FlashAttention-2 CUDA 版：另一种组织方式

CUDA 版（`flash_bwd_kernel.h`，841 行）的数学与上文完全一致，值得读的是工程取舍的三个不同答案。

### 3.1 五 GEMM 交错的主循环

`compute_dq_dk_dv_1colblock` 处理一个 KV tile，内层沿 Q 块倒序遍历（m_block 从高到低，L317/L457），每个迭代依次发射 5 个 GEMM：

$$
S \;(L474) \to P\;(\text{exp2, L536}) \to dP\;(L577) \to dV\;(L635) \to dQ\;(L655) \to dK\;(L689)
$$

dK 的 GEMM 刻意排在最后、紧挨下一轮 Q tile 的 cp.async 加载（L695–701）——**WGMMA 与异步拷贝重叠**的指令级调度。与前向的对称性：外层 n_block、内层 m_block，`acc_dv/acc_dk` 对应前向的 `acc_o`，$(P^\top, dO) \to dV$ 与前向 $(P, V^\top) \to O$ 是同构的累加结构；tile 尺寸转置（bwd 用 kBlockM=64 / kBlockN=128，launcher 的注释直言 M=128 时 dQaccum 读写翻倍、"quite slow"）。

$P$ 的重算同样是 `scale_apply_exp2<scale_max=false>`（L536）——不减 max，因为 LSE 已含 max；L411 对 OOB 行设 `INFINITY` 让 $P = 0$（注释解释了 ALiBi 场景下的 NaN 坑）。

### 3.2 dQ 的三种模式：问题与三个答案

Triton 教程用"独占 Q 块"解决了 dQ，CUDA 版没这么做——它的主 kernel 按 KV tile 并行（dK/dV 的所有权），此时 dQ 天然跨 block 归约，三种处理模式（L598–687）：

**① gmem read-modify-write**（单 kernel 串行）：每 (b,h) 一个 block 串行处理全部 KV 块，dQ 部分和从 `dq_accum`（fp32 gmem 缓冲）读进寄存器、累加、写回；最后一块负责 fp32→fp16 转换写出到真正的 dQ。

**② seq-parallel + atomicAdd**（当前主干，launcher L128–133 无条件启用）：grid `(num_n_block, b, h)`，每个 block 只处理一个 KV 块，dQ 部和直接对 `dq_accum` 逐元素 `atomicAdd`（L672–679）。grid 更大、长序列负载均衡更好，代价是 atomic 流量与浮点归约的不确定性。最后由独立的 `flash_bwd_convert_dq_kernel` 做格式转换——**"dq 转换 kernel"分出来的是累加结果的收尾，不是 dq 的计算**（dQ 的 GEMM 就在主循环里）。

**③ deterministic 分片**：atomicAdd 的浮点加法不可复现。确定性模式把 `dq_accum` 分成 `{nsplits, ...}` 切片，每个 block 写自己的切片（归约顺序固定），再由转换 kernel 求和。

### 3.3 寄存器压力：反向真正的敌人

前向每 warp 持有 `acc_s + acc_o`；反向同一线程要同时持有 $P$、$dP/dS$、$dQ$（M×d）、$dK$（N×d）、$dV$（N×d）、LSE、$\Delta$——约 2.5–3 倍。CUDA 版的应对是一组组合拳：tile 缩小（M=64）；`Is_V_in_regs` 按显存预算切换；`sP` 与 `sdQ` 复用同一块 smem（L176–177 注释警告竞态）；dK/dV 的转换与写回尽量推迟。launcher 里一堆被注释掉的历史配置（L173–201）就是调寄存器/spill 的化石层。

转置 GEMM（$P^\top dO$、$dS^\top Q$ 要求列主序喂 MMA）靠 smem 双视图（`sP/sPt`、`sdS/sdSt`）+ 寄存器 fragment 重排解决，其中 L29–53 的 `make_tiled_copy_B_warpcontiguousN` 是作者自己注释了 "This gives the correct layout, idk why" 的手写分块拷贝——工业级代码也有认怂的时刻。

### 3.4 preprocess kernel 与 dropout 的坑

独立路径的 `compute_dot_do_o`（preprocess L57–140）按行块起 grid 算 $\Delta$：fragment 重排 + warp shuffle Allreduce，**零共享内存**，顺手把 `dq_accum` 清零（省一次 memset kernel）。FA3 的 preprocess 一个 kernel 干四件事：$\Delta$、LSE×log₂e、清零 dQaccum、清零信号量。

Dropout 值得一提：前向实际用 $\tilde P = M \circ P/q$，反向把 $1/q$ 推迟到 epilogue 统一乘，$\Delta$ 相应缩放回未缩放量纲（preprocess L46, L128–132 的注释）；dropout mask 编码进 $P$ 的**符号位**（负值 = 被丢弃），`pointwise_mult` 对负值分支改用 $d$——恰好给出 softmax 分母路径的正确梯度。把信息藏进符号位这种"便宜存储"的用法，是性能代码里常见的黑话。

### 3.5 FA3 的反向：把两个瓶颈逐个拆掉

FA3（`hopper/flash_bwd_kernel_sm90.h`）针对 §3.2/§3.3 的两个瓶颈重写：

- **寄存器**：CUTLASS 3 式 warp specialization——1 个 load warp group + 2–3 个 MMA warp group，`warpgroup_reg_alloc` 把配额粗分为 24:240，load 线程几乎不占寄存器；
- **原子加**：hdim<256 时 dQ 部分和经 smem 走 **SM90_BULK_REDUCE_ADD（TMA 硬件归约加）** 直接累加到 gmem（带 `EVICT_LAST` hint），NamedBarrier 双缓冲；hdim=256 放不下才回退 atomicAdd 并把 dQ GEMM 切两半交错发射压寄存器。确定性模式用 `dq_semaphore` 的跨 block 顺序同步取代分片缓冲；
- GEMM 全部换 WGMMA 的 swapAB 变体（直接以转置布局喂 wgmma，省掉 §3.3 那套手写拷贝）、persistent tile scheduler 按 causal 负载均衡调度。

## 4. 三种 kernel 组织的对照

同一个数学，三种截然不同的反向组织，各自回答"沿哪个维度归约、谁独占输出、跨 block 归约怎么合"：

| | Triton 教程 | FA2 CUDA | TileLang（第 5 篇） |
|---|---|---|---|
| kernel 数 | 2（dkdv + dq） | 2~3（dkdv+dq 内联 + convert_dq） | 1（三梯度内联） |
| 主 kernel 并行维度 | 两个 kernel 分别按 KV / Q | 按 KV tile | 按 KV tile |
| dQ 跨块归约 | 不存在（独占 Q 块） | dq_accum: RMW / atomic / 分片 | atomic_add + swizzle layout 向量化 |
| QK^T 重算次数 | dkdv、dq 各一次 | 一次（三梯度共享） | 一次（三梯度共享） |
| 确定性 | 完全确定 | 默认非确定（atomic），可选分片 | 非确定（atomic） |
| 约束 | N_CTX % 128 == 0 | tile 启发式 + smem 预算 | 静态 shape 特化 |

没有唯一正确的答案：Triton 版选确定性、牺牲一次重算；FA2 选负载均衡、接受 atomic；TileLang 选最小访存、用 layout 工程把 atomic 的代价压下来。**kernel 组织 = 归约方向与并行度的拓扑设计**——这与第 2 篇 FA2 前向"换个循环顺序提速 1.8 倍"是同一条规律在反向的再次显形。

## 5. 小结

- 反向的数学骨架五条公式，其中 $\Delta_i = dO_i \cdot O_i$ 是最重要的恒等式（$O(Nd)$ 换 $O(N^2)$）；LSE 让 $P$ 的重算一行完成。
- recompute 不是妥协而是设计：省下 $O(N^2)$ 显存，重算的 $P$ 反而数值更干净。
- "输出所有权决定遍历方向"是反向 kernel 的第一设计原则；跨块归约（dQ）的三种解法（独占重算 / gmem 累加 / atomic）各有其代价函数。
- 反向的精度纪律比前向严格：相消运算（$dP - \Delta$）和跨块累加必须 fp32，降精只允许发生在 tensor core 操作数上。
- FA3 用 warp specialization（寄存器再分配）和 TMA reduce（硬件归约）分别拆掉了反向的两个历史瓶颈。

最后一篇[（五）]({{< relref "flash-attention-05-dsl-zoo" >}})离开单实现视角，看 TileLang、Gluon、Liger 如何在更高/更低的抽象层级上重写同一算法。
