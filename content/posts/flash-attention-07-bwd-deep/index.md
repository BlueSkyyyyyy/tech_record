---
title: "Flash Attention 精读（七）：反向传播深潜 —— 从四套实现到 MLA 反向"
date: 2026-09-14
draft: false
weight: 7
series: ["flash-attention"]
tags: ["flash-attention", "cuda", "triton", "cute", "mla", "autograd", "系列"]
categories: ["算子开发"]
---

[第 4 篇]({{< relref "flash-attention-04-bwd" >}})推导了反向的五条公式，并精读了 Triton 教程版和 FA2 CUDA 版。但反向是 Flash Attention 里最容易被低估的一半：**前向的 online softmax 是一套优雅的单向递推，反向却要在"不保存 $P$"的约束下，同时处理跨块归约、GQA 累加、dropout、softcap、确定性、以及 DeepSeek MLA 这种改造过的注意力**。

如果说前向是"一笔算清"，反向就是"**查清每一笔账从哪来、到哪去**"。这一篇要把反向里那些看起来各不相同的工程难题，收敛到同一条主线：**dQ 天生要跨块归约，每一代硬件都在回答"这个归约在哪里做"**——寄存器里、显存里、还是交给硬件。

本篇把反向讲透，分三层：

1. **数学层**（§1）：完整推导五条公式，补上 dropout / softcap / GQA 的反向，以及那个最划算的 $\Delta$ 恒等式。
2. **实现层**（§2–§3）：Triton / FA2(sm80) / FA3(sm90) / FA4(sm100) 四个抽象层级如何组织同一个反向，重点看 **dQ 跨块归约**这条主线。
3. **特例层**（§4–§5）：DeepSeek MLA 的反向（三个 kernel、topk 稀疏 scatter）与 sm100 hd256 的 2CTA 反向。

前置阅读：[第 1 篇]({{< relref "flash-attention-01-theory" >}})的 LSE、[第 4 篇]({{< relref "flash-attention-04-bwd" >}})的五公式、[第 6 篇]({{< relref "flash-attention-06-official-code" >}})的仓库地图。行号以官方仓库 commit `8d3a3b8` 为准，Triton 教程行号以带注译的 `06-fused-attention.py` 为准。

## 1. 完整的反向数学（含 dropout / softcap / GQA）

记 $S = \sigma QK^\top$（$\sigma = 1/\sqrt d$），$P = \mathrm{softmax}_{\text{row}}(S)$，$O = PV$，上游梯度 $dO \equiv \partial L/\partial O$。所有量省略 batch/head 下标。

### 1.1 五条主干公式

从 $O = PV$ 出发，逐层往回走：

$$
\boxed{\ dV = P^\top dO\ } \qquad
dP = dO\,V^\top
$$

穿过行 softmax 的雅可比 $\partial P_{ij}/\partial S_{ik} = P_{ij}(\delta_{jk} - P_{ik})$，得到

$$
dS_{ij} = P_{ij}\Big(dP_{ij} - \underbrace{\textstyle\sum_k P_{ik}\,dP_{ik}}_{\Delta_i}\Big)
\quad\Longrightarrow\quad
\boxed{\ dS = P \circ (dP - \Delta\,\mathbf{1}^\top)\ }
$$

穿过 $S = \sigma QK^\top$：

$$
\boxed{\ dQ = \sigma\, dS\, K\ } \qquad
\boxed{\ dK = \sigma\, dS^\top Q\ }
$$

**为什么要盯着"求和方向"？** 把它们写成逐元素形式：

$$
dV_j = \sum_i P_{ij}\,dO_i, \qquad
dK_j = \sigma\sum_i dS_{ij}\,Q_i, \qquad
dQ_i = \sigma\sum_j dS_{ij}\,K_j
$$

$dV_j$ 和 $dK_j$ 沿 **Q 的行维 $i$** 求和，$dQ_i$ 沿 **KV 的列维 $j$** 求和。这个方向差异就是整个反向 kernel 组织的主线——**谁沿哪个维度归约，谁就决定循环方向**。第 4 篇里 Triton 版把 dK/dV 和 dQ 拆成两个 kernel，正是因为它们的最优遍历方向相反。

### 1.2 $\Delta$ 恒等式：$O(Nd)$ 换 $O(N^2)$

$\Delta_i = \sum_k P_{ik} dP_{ik}$ 形式上需要整行 $P$（$O(N^2)$）。但代入 $dP_{ik} = dO_i \cdot V_k$：

$$
\Delta_i = \sum_k P_{ik}\,(dO_i \cdot V_k)
= dO_i \cdot \Big(\sum_k P_{ik} V_k\Big)
= dO_i \cdot O_i
= \mathrm{rowsum}(dO \circ O)
$$

$$
\boxed{\ \Delta_i = dO_i \cdot O_i\ }
$$

**$P$ 消失了。** $\Delta$ 只需要前向输出 $O$ 和上游梯度 $dO$ 的逐行点积——两者本来就有，代价 $O(Nd)$。

这就像**想核对某笔开销的分摊，其实不用翻全部明细，只要知道"这行总共花了多少"就够了**。于是所有实现都会在反向最开始跑一个极廉价的 **preprocess kernel** 把 $\Delta$ 算好（第 4 篇 §2.1 的 `_attn_bwd_preprocess`）。这是整个反向工程里最划算的一笔交易。

### 1.3 $P$ 的重算：LSE 一行恢复

反向需要 $P$，但前向只存了每行 LSE（第 1 篇的钩子）。重算 $S_{ij}$ 后：

$$
P_{ij} = e^{S_{ij} - \mathrm{LSE}_i}
$$

一步到位，**不需要重跑 online softmax、也不单独存 $\ell$**。所有实现都把 $1/\ln 2$ 折进 scale，让这一步走硬件 `exp2`。重算多付一次 $QK^\top$ GEMM + 一次 exp，换来 $O(N)$ 显存和数值上更干净的 $P$（没有 online rescale 的中间舍入）。

### 1.4 GQA / MQA 反向：dK/dV 要沿 Q head 求和

GQA 下 $h_q$ 个 Q head 共享 $h_{kv}$ 个 KV head（$g = h_q/h_{kv}$，MQA 是 $g = h_q$ 的极端）。前向里每个 Q head 各自对同一份 KV 做 attention；反向里，**同一 KV head 的 dK/dV 是所有共享它的 Q head 的梯度之和**：

$$
dV^{(kv)} = \sum_{g} P_g^\top dO_g, \qquad
dK^{(kv)} = \sigma \sum_{g} dS_g^\top Q_g
$$

$\Delta$ 和 $dS$ 仍然是每个 Q head 独立的（它们不跨 head 求和）。这个求和是**跨 CTA 的**：如果 $g$ 个 Q head 分给不同 block 计算，dK/dV 就要用 `atomicAdd` 或 TMA reduce 累加（§2.3、§4.3）。这是 GQA 反向最主要的额外复杂度来源。

FA2 里这一步在 `flash_api.cpp:1002-1005`（MQA/GQA 的 dK/dV 归约）；FA3 里在 `epilogue_bwd.hpp` 的 `CollectiveEpilogueBwdGQA`；FA4 的 `flash_bwd_sm90.py:1651` 的 `epilogue_dKV`。

### 1.5 dropout 反向：把 $1/q$ 推迟

带 dropout 的前向是 $P_{\text{drop}} = M \circ P / (1-p)$（$M$ 是 0/1 mask，$q = 1-p$ 是 keep 概率）。反向要同时穿过 mask 和 $1/q$。**$\Delta$ 恒等式依然成立**：此时 $dP_{ik} = \frac{M_{ik}}{q}(dO_i\cdot V_k)$，于是

$$
\Delta_i = \sum_k P_{ik}\,dP_{ik}
= dO_i \cdot \Big(\sum_k \frac{M_{ik}}{q}P_{ik}V_k\Big)
= dO_i \cdot O_i
$$

工程上**不逐元素乘 $1/q$，而是推迟到最后统一乘**：FA2 的 preprocess 反而把 $\Delta$ 乘上 keep 概率 $q$（`flash_bwd_preprocess_kernel.h:44` 的 `scale`，`:129-131` 的注释解释："dP 没有放大 $1/q$，只想在最后缩放 dQ/dK，所以 dP_sum 也要缩小 $q$ 倍，好让 $dP - \Delta$ 同量纲"），最后出口处再对 dQ/dK 乘 `scale_softmax_rp_dropout`（$=1/q$，`flash_bwd_kernel.h:686`）。

FA2 还把 dropout mask 编码进 $P$ 的**符号位**（负值 = 被丢弃），`pointwise_mult` 对负值分支改用 $d$——一个便宜的位级技巧（第 4 篇 §3.4）。Triton 教程版则直接用 `MASK` 分支 `tl.where` 处理。

### 1.6 softcap 反向：$\tanh$ 的导数

Gemma 2 / Grok 这类模型用 softcap 限制 attention logits：$S' = c\cdot\tanh(S/c)$。反向要多穿一层：

$$
\frac{dS'}{dS} = 1 - \tanh^2(S/c) = 1 - \Big(\frac{S'}{c}\Big)^2
\quad\Longrightarrow\quad
dS = dS' \circ \Big(1 - (S'/c)^2\Big)
$$

由于 $S' = c\tanh(S/c)$ 在算 $dS'$ 时已经存在（就是重算的 logits），这一步是零额外开销的逐元素乘。FA2 的 `apply_softcap` / `flash_bwd_kernel.h` 里对应 `ds *= (1 - (s' / softcap)^2)` 形式；FA3 的 `softcap` 走同一模式。

### 1.7 精度纪律

反向的数值比前向敏感得多，因为 $dS = P\circ(dP - \Delta)$ 是**两个相近大数的相减**：$dP$ 和 $\Delta$ 量级相当时会严重损失有效位。所有实现遵循同一条纪律：

> **累加器全程 fp32；相消运算前的中间量显式转 fp32；降精只允许发生在 tensor core 强制要求的操作数位置。**

具体：$dP$ 用 `tl.dot(...).to(tl.float32)` / CUDA 的 `to_float` 后再减 $\Delta$；$dS$ 在 fp32 算完、进 MMA 前才 cast 回 fp16；$P$ 本身也是 fp32 算出再降精。测试容差 `atol=1e-2`（参考实现 softmax 强制 fp32）验证这套纪律够用。FP8 反向多数实现干脆不支持。

## 2. Triton 教程版：零原子 + 确定性

第 4 篇已逐行读过，这里只把它的设计原则提炼成可复用的三条，作为后面三套 CUDA 实现的对照组。

### 2.1 输出所有权决定遍历方向

$dK_j, dV_j$ 沿 Q 归约 → 每个 program **独占一个 KV 块**，沿 M 维内层循环累加（`_attn_bwd_dkdv`，L441）；$dQ_i$ 沿 KV 归约 → 每个 program **独占一个 Q 块**（`_attn_bwd_dq`，L525）。输出 tile 有唯一属主、寄存器累加、最后一次性写回，**全程零原子操作**。

### 2.2 LSE 一步恢复 P，K 预乘 $\sigma/\ln 2$

`arg_k = k * (sm_scale * RCP_LN2)`（L862 附近）把 $\sigma/\ln2$ 折进 K 本身，于是 `tl.dot(k, qT)` 直接得到 $S' = \frac{\sigma}{\ln2}KQ^\top$，`p = tl.math.exp2(qk - m)`（L492 / L569）一行就是精确概率。出口处再乘 $\ln2$ 消掉多余的 $1/\ln2$（dK 乘 $\sigma$、dV 不修正）。

### 2.3 代价与收益

- **收益**：完全确定（无 atomic）、无需 gmem 累加缓冲、精度路径干净。
- **代价**：$QK^\top$ 重算两次（dkdv 和 dq 各一次）；两个 kernel 的输出 tile 独立，grid 利用率不如合并版；要求 `N_CTX % 128 == 0`（L872 assert）。

这套"拆分 + 确定性"策略在变长/小 batch 场景下不够灵活，生产实现必须换路子——于是有了 FA2 的三种 dQ 模式。

## 3. FA2 / FA3 / FA4：dQ 跨块归约的三代答案

这是本篇的主线。dK/dV 的方向（沿 Q 归约）在所有实现里都自然对应"按 KV tile 并行"；麻烦的是 dQ——它沿 KV 归约，而主 kernel 的并行维度就是 KV。**dQ 必须跨 block 归约**。

还是用"全班合抄一份作业"来打比方：dQ 就是那个**被拆给好几组、最后必须合并的公共答案**。三代硬件给的三种答案，其实是三种"合并策略"——第一代把各自的答案先写进一个公共草稿本再汇总（gmem 累加器），第二代让同桌两人当面核对（smem + 信号量），第三代干脆定制了一张专用汇总桌（TMEM + TMA 硬件归约）。

### 3.1 FA2（sm80）：gmem 累加器 + 三种模式

FA2 的主 kernel 按 KV tile 并行（`compute_dq_dk_dv_1colblock`，`flash_bwd_kernel.h:81`），内层沿 Q 块倒序遍历（`m_block = m_block_max - 1`，L317；循环 L457）。每个 KV tile 内发射五个 GEMM：

$$
S\,(L474) \to P\,(\text{exp2}, L536) \to dP\,(L577) \to dV\,(L635) \to dQ\,(L655) \to dK\,(L689)
$$

dK 的 GEMM 排最后，紧挨下一轮 Q tile 的 cp.async 预取，让计算与拷贝重叠。dQ 的结果写进一个 fp32 的 `dq_accum` gmem 缓冲，历史上有三种模式（第 4 篇已澄清**当前只有后两种在 launch**）：

| 模式 | 做法 | 复现性 | 代码 |
|---|---|---|---|
| ① gmem RMW（历史） | 单 block 串行处理全部 KV，读-改-写 dq_accum | 确定 | `flash_bwd_kernel.h:672-673`（已不 launch） |
| ② seq-parallel + atomicAdd（当前主干） | grid `(num_n_block,b,h)`，每 block 一个 KV 块，逐元素 `atomicAdd` | **不确定** | `:674-679` |
| ③ deterministic 分片 | `dq_accum` 切成 `{nsplits,...}`，每 block 写自己的切片 | 确定 | `:122-125` + `flash_bwd_preprocess_kernel.h:243-248` |

③ 的 `nsplits = ceil(num_SM / (b*h))`（`flash_api.cpp:928`），把并发 block 数摊到 SM 数上。最后用 `convert_dQ`（`flash_bwd_preprocess_kernel.h:184-268`）把 fp32 累加器转成 fp16/bf16：非确定时 `nsplits=1`、确定时跨 split 求和后前移指针（`:243-248`）。

preprocess kernel（`compute_dot_do_o`，`:57-140`）负责算 $\Delta$ 并顺手清零 `dq_accum`，省一次 memset。反向的 dK/dV 转换走 `convert_dKV`（`:274-383`）。

> FA2 的 tile 在反向被转置：launcher 用 `kBlockM=64 / kBlockN=128`，注释直言 M=128 时 dQaccum 读写翻倍、"quite slow"（`flash_bwd_launch_template.h:146-306`）。反向真正的敌人是寄存器压力：同一线程要同时持有 $P, dP/dS, dQ, dK, dV$，约为前向的 2.5–3 倍。

### 3.2 FA3（sm90）：dQ 走 TMA 硬件归约

FA3 反向（`hopper/flash_bwd_kernel_sm90.h`）沿用 warp specialization：**1 个 load WG（内部分两个 warp：一个 load Q/dO/K/V，一个 `store_dq`）+ 2~3 个 MMA WG**，寄存器配额粗分 24:240（`:212` dealloc、`:242` alloc）。

三个关键升级：

**（1）dQ 用 SM90 的 TMA 硬件归约加。** 只要 `head_dim < 256`，`dQacc_use_TMA = true`，dQ 部分和经共享内存直接由 `SM90_BULK_REDUCE_ADD` 发 `cp.reduce.async.bulk...add.f32` 累加到全局（`mainloop_bwd_sm90_tma_gmma_ws.hpp:647`，PTX 封装 `copy_sm90_bulk_reduce.hpp:22`）。这是硬件原生的"reduce to gmem"，绕过软件 atomicAdd 的流量。hdim=256 放不下才回退到 `atomicAdd`（`:960-964`）。

**（2）确定性用信号量而非分片。** `dq_semaphore` 按 `(bidb, bidh)` 计数，每个 KV block 完成后 `Barrier::arrive_inc`（`:657`），下一个 block `Barrier::wait_eq`（`:638/640`）等前序完成——**用跨 block 顺序同步取代了 FA2 的分片缓冲**，显存省一个数量级。信号量在 preprocess kernel 里清零（`flash_bwd_preprocess_kernel.h:242-246`）。

**（3）WGMMA 的 swapAB。** `SdP_swapAB/dKV_swapAB/dQ_swapAB`（`:53-55`）让 WGMMA 直接以转置布局喂操作数，省掉 FA2 那套 `make_tiled_copy_B_warpcontiguousN` 手写拷贝（第 4 篇 §3.3 里作者自嘲 "idk why" 的那个）。

FA3 的 launch 分三步：**preprocess → main → dQ postprocess**，GQA 时再加 dK/dV postprocess（`flash_bwd_launch_template.h:225-289`）。preprocess（`flash_bwd_preprocess_kernel.h:142-248`）一个 kernel 干四件事：$\Delta$（`:201-212`）、$LSE_{\log2} = LSE\cdot\log_2 e$（`:224-229`）、清零 dQaccum（`:231-240`）、复位 dq_semaphore（`:242-246`）。postprocess（`flash_bwd_postprocess_kernel.h`）把 fp32 dQaccum 反量化并按 `softmax_scale` 缩放写出。

### 3.3 FA4（cute, sm90）：把公式编号写进源码

FA4 的 `flash_bwd_sm90.py` 里，`mma_one_m_block`（`:1491`）的注释直接标了公式序号，逐条对应 §1：

| 公式 | 代码位置 | MMA |
|---|---|---|
| $S = QK^\top$ | `flash_bwd_sm90.py:1523-1525` | `mma_qk_fn` |
| $dP = dO\,V^\top$ | `:1528-1532` | `mma_dov_fn` |
| $P = e^{S-\mathrm{LSE}}$（exp2） | `:1541-1551` | — |
| $dS = P\circ(dP-\Delta)$ | `:1563-1569` | — |
| $dV \mathrel{+}= P^\top dO$ | `:1590-1596` | `mma_pdo_fn` |
| $dQ = dS\,K$ | `:1602-1604` | `mma_dsk_fn` |
| $dK \mathrel{+}= dS^\top Q$ | `:1607-1613` | `mma_dsq_fn` |

dQ 的归约：每 tile 的 dQ 先写进共享内存 `sdQaccum`，由独立 warp 用 TMA reduce-add 累加到全局 fp32 缓冲（`dQaccum_store:1802` → `copy_utils.cpasync_reduce_bulk_add_f32:1903`）；确定性模式下用 `mdQ_semaphore` 保序（`:1839/1889`）。dK/dV 的 GQA 累加在 `epilogue_dKV:1651`，确定性用 `mdK_semaphore/mdV_semaphore`。

### 3.4 FA4（cute, sm100）：dQ 累加器搬进 TMEM

Blackwell 的 tcgen05 MMA 把累加器放在 **Tensor Memory（TMEM）** 而不是寄存器。`flash_bwd_sm100.py`（4345 行，全仓库最大）的 dQ 累加器 `tdQtdQ` 就在 TMEM 里（`:1462`）：

- **MMA 阶段**：5 个 tiler 直接对应 §1 的五公式（`:117-127` 的注释）。
- **归约阶段**：`dQacc_reduce`（`:3707`）里 reduce warp 把 TMEM→RMEM（`:3862-3864`），经 SMEM 后由 TMA warp 用 `cpasync_reduce_bulk_add_f32`（`:3912`）累加到全局。
- **2-CTA**：按 CTA rank 分走一半（`:3734-3743`）；确定性用 `mdQ_semaphore`（`:3652`）。

warp 角色（`:175-180`）：reduce `(0..3)`、compute `(4..11)`、mma=12、load=13、relay=14、empty=15。TMEM 偏移在 `__init__` 里精打细算——`tmem_dS_offset` 与 `tmem_dP_offset` 重叠、`tmem_dQ_offset` 与 dP 重叠（`:211-244`），hdim64 甚至有专门的 `split_P_dS` 让 P/dS 独占 TMEM slot 以重叠 `QK_{t+1}` 与上一 tile 的 softmax（`:48-64` 的 NOTE）。

后处理统一走 `flash_bwd_postprocess.py`（`:135`，kernel `:409`）把 fp32 accumulator 缩放转 dtype；learnable-sink 的 `dsink` 由独立的 `DSinkReduceKernel`（`:81`）算。preprocess（`flash_bwd_preprocess.py:38`）算 $\Delta$（`:137-141`）和 $LSE_{\log2}$（`:262-263`）。

### 3.5 四代对照

| | Triton 教程 | FA2 (sm80) | FA3 (sm90) | FA4 (sm100) |
|---|---|---|---|---|
| kernel 数 | 2（dkdv + dq） | 2~3（主 + preprocess + convert） | 3（preprocess + main + postprocess） | preprocess + main + postprocess |
| 主 kernel 并行 | 两 kernel 分别按 KV / Q | 按 KV tile | 按 KV tile | 按 KV tile |
| dQ 累加 | 不存在（独占 Q 块） | gmem + atomicAdd / 分片 | smem + TMA reduce-add | TMEM → smem + TMA reduce-add |
| 确定性 | 天然确定 | 可选分片 | 信号量保序 | 信号量保序 |
| MMA | Triton 自动 | HMMA + 手写布局重排 | WGMMA + swapAB | tcgen05/TMEM |
| 主要瓶颈 | 重算两次 | 寄存器 / atomic 流量 | 寄存器 | TMEM 容量 / 同步 |

一条清晰的演进线：**dQ 的累加器从寄存器 → gmem → smem+TMA → TMEM，每换一代硬件，"跨块归约"就多一层硬件原语可用**。但确定性这个需求始终靠"显式顺序同步"解决，只是实现从分片缓冲变成了信号量。

## 4. DeepSeek MLA 反向：三个 kernel 的稀疏拼图

MLA（Multi-head Latent Attention）是 DeepSeek-V2/V3 的核心，它的反向在官方仓库里只有 FA4 sm100 实现（FA2/FA3 都没有）。理解它需要先理解前向的"吸收式"写法。

先建立直觉：**MLA 相当于把完整的 K/V 先"压缩打包"成一个低维 cache**，用的时候再按公式"就地解压"参与计算——显存省了，但前向的表达式变了，反向自然也要跟着变。第 4 篇那套五公式，在这里不够用了。

### 4.1 MLA 的吸收式前向

标准 MLA 先把 KV 压缩到低维 latent $c_t^{KV}$，再投影回 $k_t = W^{UK}c_t^{KV}$、$v_t = W^{UV}c_t^{KV}$。把两个投影"吸收"进 attention 计算后，前向可以写成（`interface.py:3532-3534`）：

$$
O = \mathrm{softmax}\Big(\text{scale}\cdot(Q\,K^\top + Q_v\,V^\top)\Big)\,V
$$

其中 $Q = q_{\text{pe}}$（RoPE 部分）、$K = \text{pe\_cache}$，$Q_v = q_{\text{nope}}$、$V = \text{kv\_cache}$。**K 和 V 共享同一份压缩 KV cache**（代码里 `k is v`），所以 `flash_bwd_mla_sm100.py` 里 `hdim=64`、`hdimv=512`，且是 MQA（`qhead_per_kvhead` 为 64 或 128）。

这里比标准 attention 多出一项 $Q_v V^\top$，反向要多算一个 $dQ_v$，这是 MLA 独有的：

$$
dQ_v = \text{scale}\cdot dS\,V, \qquad
dV \mathrel{+}= \text{scale}\cdot dS^\top Q_v
$$

### 4.2 三个 kernel 的分工

MLA 反向被拆成三个 kernel（`_flash_attn_bwd_sparse_mla`，`interface.py:2706`）：

**① 主 kernel `flash_bwd_mla_sm100.py`**（2135 行）：算 $dP, dS, dV$。公式注释在 `:163-165`：$dP^\top = V\,dO^\top$、$dV \mathrel{+}= P^\top dO$、$dV \mathrel{+}= dS^\top Q_v$。`num_hdimv_splits=2` 把 $d_{v}=512$ 对半切（`:154`）。`mma`（`:1499`）发三组 GEMM（`gemm_VdO` / `gemm_PtdOt` / `gemm_dStQvt`）；`compute_loop`（`:1738`）算出 $P$ 和 $dS = P\circ(dP - \Delta)$（`:1910-1916`），并把 $dS$ 写回全局供后两个 kernel 用（`:1924-1936`）。

**② `flash_bwd_mla_dk_sm100.py`**：算 $dK = dS^\top Q$。它是个 GEMM（`dKGemmKernel:66`，`A=dS^\top, B=Q`），但 epilogue 要做 **topk scatter-reduce**——因为 MLA 用的是 top-k 稀疏注意力，dK 要按 gather 索引散回真实 `seqlen_k` 位置（`epilogue_scatter_reduce:1002`，`atomic_add:1125`，无效槽 `-1` 跳过）。

**③ `flash_bwd_mla_dq_dqv_sm100.py`**：把 $dQ = dS\,K$ 和 $dQ_v = dS\,V$ 合并进一个 cluster `(1,2)` 的 kernel（`dQdQvGemmKernel:51`，cluster 覆盖整个 dQv，CTA0 额外算 dQ），K/V 按 topk 索引 gather。

三者的 $dV$/$dK$ 都用 `atomic_add_fp32x4` 散列累加（主 kernel `:2122`、dk kernel `:1125`），所以 **MLA 反向目前只有非确定模式**（`interface.py` 里断言 `deterministic is False`）。

### 4.3 为什么这么复杂

标准 attention 的 dK/dV 是稠密的（每个 KV 位置都被所有 Q 看到）；MLA + topk 稀疏下，某个 KV 位置可能只被少数 Q 选中，且选中关系由运行时索引决定。**这迫使 dK/dV 从"块级累加"退化成"按索引 scatter-add"**——这是第 5 篇"变体的性能来源对应循环结构的哪处改动"的又一个例子：稀疏 → 索引 gather/scatter。

## 5. sm100 hd256 2CTA 反向：另一种拆分

`head_dim = 256` 时 TMEM 和寄存器都吃紧，FA4 给它单独写了三个文件（`interface.py:2502` 分派）：

- `sm100_hd256_2cta_fmha_backward.py`：主入口 `BlackwellFusedMultiHeadAttentionBackward:28`，`__call__:115` 先跑 **dQ kernel**（`:199`）再跑 **dKdV kernel**（`:212`），dQ 直接写进 `dQ_accum` 槽。
- `..._dqkernel.py`：`compute_step:2067` 算 $P = e^{S\cdot\text{scale} - \text{LSE}}$ 和 $dS = P\circ(dP - dPsum)$，`dQ_epilogue:2199` 在 TMEM→SMEM→TMA 写出 dQ。
- `..._dkdvkernel.py`：`compute:2463` 算 $P/dS$，`mma_2cta:2101` 用 2-CTA MMA 算 $dP = V dO$、$dV = P dO$、$dK = dS\,Q^\top$（`:2165-2295`），`epilogue:2871` TMA 写出 dK/dV。

注意它把 **dQ 和 dKdV 彻底拆成两个顺序执行的 kernel**，而不是像通用 sm100 kernel 那样在一个 kernel 里交错。这是"hdim 太大、一个 kernel 装不下"时的必然取舍——第 4 篇讲的"寄存器压力是反向真正的敌人"，在 hdim256 上直接决定了 kernel 的个数。

## 6. 反向的工程要点清单

把所有实现反复出现的决策浓缩成一张检查表：

- **$\Delta$ 恒等式**：任何反向的第一件事，独立 preprocess kernel，$O(Nd)$。
- **LSE 恢复 $P$**：前向存 log2 域 LSE，反向 `exp2(S - LSE)` 一行。
- **REcompute vs 保存**：省 $O(N^2)$ 显存，重算的 $P$ 数值更干净。
- **遍历方向 = 归约方向**：输出去谁，谁独占 tile。
- **dQ 跨块归约**：独占重算（Triton）/ atomic（FA2）/ TMA reduce（FA3）/ TMEM+TMA（FA4）。
- **GQA 的 dK/dV**：跨 Q head 求和，走 atomic 或 TMA reduce。
- **dropout**：$1/q$ 推迟到 epilogue；mask 可藏进符号位。
- **softcap**：$dS \mathrel{*}= 1 - (S'/c)^2$，零额外开销。
- **确定性**：信号量保序优于分片缓冲。
- **精度**：$dP-\Delta$ 相消前转 fp32；降精只在 MMA 操作数。
- **MLA/稀疏**：topk 索引让 dK/dV 退化为 scatter-add，只能非确定。

## 7. 小结

- 反向的五条公式里，$\Delta_i = dO_i\cdot O_i$ 是最重要的恒等式；LSE 让 $P$ 的重算一行完成。dropout、softcap、GQA 都是在主干上各加一个逐元素因子或一次跨 head 求和。
- **dQ 跨块归约是反向工程的主线**：四代实现分别用独占重算、gmem atomic、TMA reduce、TMEM+TMA 解决，确定性则从分片缓冲演化到信号量保序。
- FA2 反向当前只有 seq-parallel + atomic 与 deterministic 分片两种模式；FA3 把 dQ 归约交给 TMA 硬件；FA4 把它搬进 TMEM，并把架构差异收敛成 Python 类。
- MLA 反向是三个 kernel 的拼图（dP/dS/dV、dK、dQ/dQv），topk 稀疏使 dK/dV 变成 scatter-add，目前只有非确定模式。
- 反向的精度纪律比前向严格：相消运算和跨块累加必须 fp32，降精只发生在 tensor core 操作数上。

本系列到这里收束：从 softmax 的溢出（[一]({{< relref "flash-attention-01-theory" >}})）讲到 MLA 反向（本篇）。回头看，Flash Attention 的故事始终是两层——**数学层**是"softmax 是可流式计算的可交换归约"，**工程层**是每一代硬件上"谁在等谁、谁归约谁"的重新编排。这十一个字的公式，值得用七篇文章去拆。

配套代码 [`code/flash-attention/bwd_variants_ref.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/flash-attention) 把 §1 的 dropout / softcap / GQA 反向写成纯 PyTorch 参考实现，并与 autograd 逐一对拍（`python bwd_variants_ref.py`，CPU 可跑）。
