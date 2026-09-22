# FlashAttention 反向优化手段梳理（FA / TE 对照）

> 目标：为「在目标 AI 卡上开发 flash-attention-backward」做技术摸底。本文梳理 FA2/FA3 反向的
> 实现结构与优化手段，以及 FP8 反向（TE 做法）的关键难点。**行号引用基于本地
> `~/github/flash-attention` commit `edb5c76`**，标注为 `文件:行`，需在写作/实现时回源码复核。
> 后续实现与实测见 `../ROADMAP.md`。

---

## 1. 反向的数学与计算量

设 `S = scale·QKᵀ`，`P = softmax(S)`（causal），`O = PV`，损失对 `dO` 已知。反向：

$$
\begin{aligned}
D &= \operatorname{rowsum}(dO \odot O) \quad(\text{即 }\sum_j dO_{ij}O_{ij})\\
dP &= dO\,V^{\mathsf T}\\
dS &= P \odot (dP - D)\\
dV &= P^{\mathsf T} dO\\
dQ &= \text{scale}\cdot dS\,K\\
dK &= \text{scale}\cdot dS^{\mathsf T} Q
\end{aligned}
$$

要点：
- **`D`（delta / softmax_d）可以预计算**，与 `P` 无关（只需 `dO,O`）。FA 用独立 preprocess kernel 算它，
  这样主 kernel 不必同时持有 O 和 dO（省 smem/寄存器）。
- `dQ` 需要对**所有 K 块**求和（沿列归约），`dK,dV` 对**所有 Q 块**求和（沿行归约）。
- 反向 FLOPs ≈ 前向的 2×：约 `4·b·s²·h·(qk_dim+v_dim)`。
- 反向要读 `Q,K,V,O,dO` 写 `dQ,dK,dV`；若像前向一样把 `P` 存下来，显存会是 `O(N²)`——所以必须
  **recompute**。

---

## 2. FA2（SM80）反向结构

核心文件：`csrc/flash_attn/src/flash_bwd_kernel.h`、`flash_bwd_preprocess_kernel.h`、
`flash_bwd_launch_template.h`。

### 2.1 三段式

1. **preprocess**（`flash_bwd_preprocess_kernel.h`）
   - `compute_dot_do_o`（`:58`）：逐元素 `dot(dO,O)` 行和 → `softmax_d`（即 `D`），并做 NaN 检查。
   - `clear_dKVaccum`（`:145`）：清空 `dK/dV` 的累加缓冲。
   - `convert_dQ`（`:185`）/ `convert_dKV`（`:275`）：把 fp32 累加缓冲转回目标精度写回。
2. **main kernel** `compute_dq_dk_dv`（`flash_bwd_kernel.h:799`）
   - 调用 `compute_dq_dk_dv_1colblock`（`:81`）逐 K/V 列块处理。
   - causal 用 `Is_first/Is_last` 模板区分「对角块（需 mask）」与「全块（无需 mask）」（`:813-820`）。
   - 另有 `compute_dq_dk_dv_seqk_parallel`（`:826`）做序列并行（带 `Seq_parallel`）。

### 2.2 数据流（每个 K/V 列块，`1colblock`）

对 Q 块 `[BLK_M, d]` 与 K/V 块 `[BLK_N, d]`：
1. 从 smem 取 Q,K，算 `S = QKᵀ`（`P` 用 online-softmax 的前向 `m` 重新 exp 得到，**recompute**）；
2. `dV += Pᵀ dO`（对 N 累加）；
3. `dP = dO Vᵀ`；
4. `dS = P∘(dP − D)`；
5. `dQ += dS K`（Q 方向在寄存器/smem 累加，最终写 `dQ_accum`）；
6. `dK += dSᵀ Q`。

### 2.3 优化手段清单（FA2）

| 手段 | 作用 | 证据 |
|---|---|---|
| **recompute P** | 不存 `O(N²)` 的 P，反向用 `m` 重新 exp | `flash_bwd_kernel.h` 主循环 |
| **预计算 D=rowsum(dO∘O)** | 主 kernel 不必同时驻留 O/dO | `flash_bwd_preprocess_kernel.h:58` |
| **dQ 累加缓冲 `dQ_accum`** | dQ 跨 K 块累加，主 kernel 用 `atomicAdd` 写缓冲 | `flash_bwd_kernel.h:124-125, :678` |
| **dK/dV 原子累加** | 跨 Q 块累加；非确定性模式直接 `atomicAdd` | `:678` 附近 |
| **deterministic 模式** | 每 Q 块写独立 `dQ_accum` split，再 convert，结果可复现 | `:124-125`, `:826` |
| **causal 模板特化** | 对角块 mask、其余全块，省无效计算 | `:813-820` |
| **smem 复用 Q/K/V 分块** | 减少全局访存 | Kernel_traits 分块 |
| **warp 划分（N 方向切分）** | 4 warps 各算 S 的一段，dK/dV 后合并 | Kernel_traits |
| **fp32 累加** | S/dS/dQ/dK/dV 均 fp32 累加，降低误差 | acc_dq/acc_dkv |

---

## 3. FA3（SM90 Hopper）反向结构

核心文件：`hopper/flash_bwd_kernel_sm90.h`、`hopper/mainloop_bwd_sm90_tma_gmma_ws.hpp`、
`hopper/epilogue_bwd.hpp`、`hopper/tile_scheduler.hpp`。

在 FA2 基础上叠加 Hopper 特性：
- **TMA**（`cp.async.bulk.tensor`）搬 Q/K/V/dO 分块，配 `mbarrier`；
- **warpgroup MMA（wgmma）** + **warp specialization**（producer 搬数、consumer 算）；
- **ping-pong / 多级流水**（`sm90_pipeline_no_cluster.hpp`）；
- `TiledMmadKV`、`dKV_swapAB`（`flash_bwd_kernel_sm90.h:39-44`）——dK/dV 的 MMA 布局交换以适配 wgmma；
- epilogue 单独在 `epilogue_bwd.hpp` 里 store dK/dV（`:268`）。

> 论文结论（需实测复核）：FA3 反向相对 FA2 主要赢在 **TMA + wgmma + 更深的流水**，
> 而不是算法变化。

---

## 4. FP8 反向：为什么最难点，TE 怎么做

FA 仓库的**反向没有 FP8**（`csrc/flash_attn/src` 只有 fp16/bf16 的 `flash_bwd_hdim*`；
`grep -ril fp8` 命中的是前向/interface）。FP8 反向要借鉴 **TransformerEngine fused_attn_bwd**。

### 4.1 TE 的 FP8 反向口径（来自 `te-perf/bench_te.py` 与 TE 文档）

- 前向：Q/K/V/S/O 用 **E4M3**；
- 反向：`dO`、`dP`（以及 `dQ/dK/dV`）用 **E5M2**（动态范围大，适合梯度）；
- 输入张量在计时区外**预先量化**（rowwise，`Float8Quantizer(rowwise=True)`），
  所以测到的是纯 FP8 kernel；kernel 内部做 rowwise 动态缩放。
- 反向支持受 head_dim 限制：训练口径 `qk==v` 最大 256，`qk!=v` 仅在 `qk≤192,v≤128` 附近可用
  （见 `te-perf/results_summary.md`）。

### 4.2 关键难点

1. **量化对象**：`dO` 与 `P`/`dS` 都要进张量核。`P=softmax` 在 `[0,1]`，动态范围小，直接 E4M3；
   `dO` 无界，需 E5M2 + rowwise scale。`dS = P∘(dP−D)` 的量化会放大误差。
2. **缩放因子（scale）**：FP8 张量核要 `scale_a*scale_b` 乘回 fp32 累加器。rowwise（每行一个 amax）
   够用但精度一般；需与 TE 的 scaling 布局严格对齐才能数值可比。
3. **累加精度**：所有 MMA 累加器保持 **fp32**；只在输入侧量化。`D`、`m`、`l` 等统计量用 fp32。
4. **量化误差对 dQ/dK/dV 的影响**：反向对 S 的误差敏感（softmax 的 `dS` 含 `P` 因子，小 P 处噪声相对大），
   需要对拍确定容差（预期比 fp16 松 1~2 个数量级）。
5. **确定性**：FP8 kernel 一般非确定性（原子累加）；对拍用容差而非位相等。

### 4.3 我们的 FP8 反向实现路线（计划）

- 以 FA2 的 `1colblock` 结构为骨架，把 `dO`/`V`/`P`/`Q`/`K` 按 rowwise 量化到 FP8，
  用 `mma` FP8（`m16n8k32` e5m2/e4m3）做 `dP=dO·Vᵀ`、`dV=Pᵀ·dO`、`dQ=dS·K`、`dK=dSᵀ·Q`；
- `D`/`dS` 计算和 `dS` 量化在 fp32 完成后再量化；
- 先做 **正确性**（对 fp32 ref 的容差），再对标 TE fp8 的时间。

---

## 5. 目标 AI 卡的可移植性注意点

（当前开发机为 H100 sm90，反编译/运行均在此验证；目标卡待定，故代码尽量抽象。）

- 把「张量核 MMA 指令」「异步拷贝」「warp 调度」抽成可替换层，便于换卡；
- 先写**功能正确的标量/CUDA-core 版本**作为 golden，再逐层上张量核；
- 数值对拍统一走 `harness/fa_bwd_bench.py` dump 的 CPU npy，跨卡可比。

---

## 6. 待办

见 `../ROADMAP.md`：按 fp16→bf16→fp8 的顺序，各做「单文件 / 两文件」实现 + 编译 + ncu + 对拍 + 对标 TE，
并持续沉淀到本目录 `docs/`。
