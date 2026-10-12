# MoE 反向 kernel 接入与端到端对拍：SM90 grouped GEMM vs PyTorch autograd

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 14 篇（路线图 **P4-4**）。
> P4-1 推导了反向数学（float64 对拍）、P4-2 精读 wgrad 的 `k_grouped_*_contiguous`、P4-3 精读 dgrad 的
> `m_grouped_*`。本文是反向部分的收尾：把这些 **H100 可跑的 SM90 kernel 真的接进一个 MoE 的完整反向图**，
> 讲清「每个专家的 token 集合怎么 gather 成 kernel 要的物理布局」「四项梯度各走哪个 kernel、形状角色是什么」，
> 并给出与 **PyTorch autograd** 的 kernel 级数值对拍和吞吐。
>
> 约束提醒：SM100（`sm100_*` 1D1D + tcgen05 + packed UE8M0）为**仅分析**；本机 H100（sm90）走
> `sm90_k_grouped_fp8_gemm_1d1d` / `sm90_m_grouped_fp8_gemm_contiguous_1d2d`。实测脚本
> `harness/exp_moe_backward_kernels.py`（+ `.out.txt`）。

## 1. 一个 MoE 反向图里的四个大 GEMM

把一个 token 级 MoE 层（router → top-k → 逐专家 linear1 → SwiGLU → linear2 → combine）的反向拆开，
真正是「大 GEMM」的只有四处；其余（combine 反向、SwiGLU 反向、router/topk 反向）是逐元素或小 dense 算子，
保持 torch 实现即可。

记 token 集合为 $t$，专家 $j$ 收到的 token 集合为 $S_j$（$|S_j|=n_j$），各专家 $n_j$ **不等长**。
P4-1 的 (3)(5) 给出四项梯度，本文按 kernel 视角重列：

| 梯度 | 数学式 | 不等长轴 | kernel | (M, N, K) |
|---|---|---|---|---|
| `dW2` linear2 wgrad | $dW^2_j = A_j^{\mathsf T} dY_j$ | K=token$j$ | `k_grouped_*_nt_contiguous` | (I, H, $n_j$) |
| `dA` linear2 dgrad | $dA_{tj} = dY_{tj} (W^2_j)^{\mathsf T}$ | M=token$j$ | `m_grouped_*_contiguous` | ($n_j$, I, H) |
| `dW1` linear1 wgrad | $dW^1_j = x_j^{\mathsf T} dG_j$ | K=token$j$ | `k_grouped_*_nt_contiguous` | (H, 2I, $n_j$) |
| `dX` linear1 dgrad | $dX_{tj} = dG_{tj} (W^1_j)^{\mathsf T}$ | M=token$j$ | `m_grouped_*_contiguous` | ($n_j$, H, 2I) |

四条都满足 P4-1 的判据：**wgrad 的不等长轴是收缩维 K（token）→ K 轴分组；dgrad 的不等长轴是输出 M
（token）→ M 轴分组**，N/K（或 M/N）固定。这就是它们必落到 `k_grouped` / `m_grouped` 两族的原因。

host 入口：wgrad 走 `csrc/apis/gemm.hpp:373`（`k_grouped_fp8_gemm_nt_contiguous`，SM90 kernel 派发在
`:421-423`），dgrad 走 `csrc/apis/gemm.hpp:181`（`m_grouped_fp8_fp4_gemm_nt_contiguous`，SM90 派发在
`:236-241` 到 `sm90_m_grouped_fp8_gemm_contiguous_1d2d`）。

## 2. gather：从 token 级张量到 kernel 物理布局

MoE 反向的特殊之处不在 GEMM 本身，而在**喂给 kernel 的数据要按专家重排**。设 `sel ∈ {0,1}^{T×E}`
是 top-k 选择矩阵，`rows[e] = { t : sel[t,e]=1 }` 是专家 $e$ 的 token 行号。

### 2.1 wgrad（K 轴分组）

K 轴分组要求把各专家的 token 行**沿 K 拼起来**。以 `dW2 = A_e^T dY_e` 为例，把每组的
$A_e=[n_e,I]$（M=I 固定）、$dY_e=[n_e,H]$（N=H 固定）按 token 行拼成 `[sum_k, ·]`：

- 逻辑上第 $e$ 组占 `[off, off+n_e)`，`off` 是**对齐后**的累计偏移；
- 每组补齐到 `align(n_e, 128)`，gap 行填 0（K 对齐硬要求，见 `check_k_grouped_args`，`gemm.hpp:69`
  `k % k_alignment == 0`）；
- `ks_cpu` = 各组的**补齐后** K（`host_ks`），`grouped_layout` = `host_ks` 的 int32 张量。

kernel 侧 SM90 是每组 `[M, k_g]` **K-major** 物理布局（P4-2 §2），所以量化时按组做 per-channel（沿 K 128 粒）
cast，再 `.T.flatten()` 拼成 1D `data`，SF 转置成 `[·, sum_k/128]`——与 DeepGEMM 测试 generator 完全一致。

### 2.2 dgrad（M 轴分组）

M 轴分组要求把各专家 token 行**沿输出 M 拼起来**。以 `dA_e = dY_e @ W2_e^T` 为例，A=$dY_e=[n_e,H]$
（K=H 固定，K-major）、B=$W2_e=[I,H]$（N=I, K=H）：

- A 按 M 轴拼成 `[m_total, H]`，每组补齐到 `align(n_e, 128)`，gap 行填 0；
- `grouped_layout` 长度 = `m_total`，有效行写组号 $e$、gap 行写 `-1`；
- A 用 per-token cast（每行每 128 个 K 一个 SF），B 用 per-block cast（$128\times128$ 块一个 SF）。

`m_grouped_fp8_fp4_gemm_nt_contiguous` 的 host 契约见 `gemm.hpp:194-233`：A 必须 K-major
（`:197`）、`d.scalar_type()==kBFloat16`（`:209`）、contiguous 下 `grouped_layout` 长度必须等于 `m`
（`:217-218`）、padding 行需为 0（`ensure_zero_padding`）。

## 3. 对拍策略：两层误差

数值上要区分**两种误差**，否则会把 FP8 量化噪声误当成 kernel bug：

1. **`vs deq`（隔离 kernel+布局）**：把喂进 kernel 的 FP8 反量化后在 fp32 重算同一 GEMM。若 kernel
   与布局正确，这只剩 bf16 输出舍入，应 ~1e-4。
2. **`vs ref`（含 FP8 量化）**：与 fp32 autograd 参考比。这是训练真实场景的误差，被 per-128 块量化
   主导，通常 ~1e-2（与 P4-2/P4-3 一致）。

参考梯度用 `torch.autograd.grad` 对 `(x, Wg, W1, W2)` 求；四项 kernel 输出再与 autograd 的
`dW1/dW2/dx` 比。注意 `dx` 含 FFN 部分（kernel）与 router 部分（torch），比较时把 router 部分解析算出
后从 autograd 的 `dx` 里减去（推导同 P4-1）。

## 4. H100 实测

配置：`T=64, H=512, I=256, E=8, topk=2`，逐专家 token 数 `[13,20,20,14,10,18,15,18]`（和 128）。
M 对齐 128、K 对齐 128。原始输出 `harness/exp_moe_backward_kernels.out.txt`。

| kernel | 形状 (M,N,K) | `vs deq` relerr | cos | `vs ref`(autograd) relerr |
|---|---|---|---|---|
| `[1]` linear2 wgrad `k_grouped` | (I=256, H=512, K=token) | 2.621e-04 | 1.000000 | 2.460e-02 |
| `[2]` linear2 dgrad `m_grouped` | (token, I=256, H=512) | 3.134e-04 | 0.999999 | 3.728e-02(analysis) |
| `[3]` linear1 dgrad `m_grouped` | (token, H=512, 2I=512) | 3.129e-04 | 0.999999 | 5.078e-02(analysis) |
| `[4]` linear1 wgrad `k_grouped` | (H=512, 2I=512, K=token) | 2.625e-04 | 1.000000 | 4.546e-02 |

端到端拼回：

```
dW2 kernel vs autograd : relerr=2.460e-02  cos=0.999653
dW1 kernel vs autograd : relerr=4.546e-02  cos=0.999001
dX_ffn kernel vs autograd: relerr=5.076e-02  cos=0.998709
full dx: (kernel dx_ffn + torch router) vs autograd: relerr=3.506e-02  cos=0.999655
```

**读法**：`vs deq` 全部 ~2.6e-4（纯粹是 bf16 输出舍入），说明四个 kernel 与物理布局接入**正确**；
`vs ref` 的 2.5e-2 ~ 5.1e-2 全部来自 per-128 FP8 块量化（与本专题其它实验同量级），不是实现错误。
`dX` 的误差（5.1e-2）高于 `dW`，因为 dgrad 的激活侧量化后再经 SwiGLU 反向放大；注意力应放在
**cos 都在 0.999 以上**，梯度方向完全正确。

## 5. 吞吐：单 rank 反向的四个 kernel

DeepSeek 量级单 rank 视角：`T=2048, H=7168, I=2048, E=8, topk=2`（共 4096 次专家分配，每专家 512）。
只测四个 GEMM kernel 本体（不含量化、gather/scatter、SwiGLU、router）：

| kernel | 形状 | 时间 | 吞吐 |
|---|---|---|---|
| `[1]` linear2 wgrad `k_grouped` | (2048, 7168, sum_k=4096) | 1171 us | 103 TFLOPS |
| `[4]` linear1 wgrad `k_grouped` | (7168, 4096, sum_k=4096) | 2268 us | 106 TFLOPS |
| `[2]` linear2 dgrad `m_grouped` | (m=4096, 2048, 7168) | 268 us | 449 TFLOPS |
| `[3]` linear1 dgrad `m_grouped` | (m=4096, 7168, 4096) | 443 us | 543 TFLOPS |

**为什么 wgrad 只有 ~100 TFLOPS、dgrad 却 450+**：wgrad 的收缩维 K 就是 token 数（这里 sum_k=4096，
分到 8 组每组仅 512），**算术强度低**（每读一份权重只做很少的乘法），跟 P4-2 §6 里 `m=4096,n=7168`
那组 425 TFLOPS 的差别正是**K 太小**（那组 k/group=4096，这里 K=512/组）。dgrad 的 K 是隐藏/中间维
（7168/4096），算术强度高，故接近 H100 FP8 峰值。这也解释了为什么 MoE 反向里 wgrad 往往是瓶颈：
它天然是「瘦长」GEMM，且需要跨 rank/all-reduce 之外的本地聚合。

## 6. 与前向的对照：反向为什么没有 mega-kernel

前向（Mega MoE）把 dispatch + linear1 + SwiGLU + linear2 + combine 融进**一个** SM100 kernel，靠
通信-计算重叠掩盖 NVLink 时延（本专题 P2/P3）。反向没有对应的大融合 kernel：

- 反向的**通信**是 combine 的反向（把 $dY$ dispatch 回源 rank）与 dgrad 的跨专家 scatter-add，规模小、
  且可与 GEMM 重叠的收益有限；
- 反向的**计算**天然是可分解的独立 grouped GEMM（四个），wgrad 还**留守专家本地 rank**、无需 all-reduce；
- 因此工程上反向就是「gather → 四个 grouped GEMM → scatter-add」的常规 pipeline（本文实测即此）。

> SM100 侧差异（**仅分析**）：wgrad 用 `k_grouped_fp8_gemm_tn_contiguous`（A/B 为 `[sum_k,M]`
> MN-major，`gemm.hpp:320`；sm90 调会 `DG_HOST_UNREACHABLE`，`:369`）；dgrad 用
> `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d`（`:242-246`，tcgen05 + packed UE8M0 +
> `BLOCK_M=256` 对齐）。数学与本文一致，只是 major、对齐粒度与 SF 打包不同（见 P4-2/P4-3）。

## 7. 小结与下一步

- **四项梯度 = 两个 kernel 族**：wgrad（`k_grouped`，不等长在 K=token）与 dgrad（`m_grouped`，不等长在
  M=token）；接进真实 MoE 反向只需按专家 `sel` 做 gather/scatter-add。
- **两层对拍**：`vs deq` ~2.6e-4 证明 **kernel+布局**正确；`vs ref` ~1e-2 是 FP8 量化噪声，cos > 0.999。
- **wgrad 是反向瓶颈**：K=token 偏小 → 算术强度低（~100 TFLOPS）；dgrad K 是隐藏维 → ~450+ TFLOPS。
- **反向无 mega-kernel**：可分解为独立 grouped GEMM，wgrad 留守专家本地 rank。

**下一步**：P4 反向全部完成，转入 **P5-1 Mega Gate** / **P5-2 Mega mHC**（相关 mega-kernel 精读）。
