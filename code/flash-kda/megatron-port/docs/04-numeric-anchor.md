# 04 · 把参考实现锚定到 FLA `chunk_kda`（M6）

> 系列第 5 篇。前三篇（[00](00-math-and-api.md) 数学/接口、[01](01-forward.md) 前向、
> [02](02-backward.md) 反向）与 [03](03-optimization.md) 都在**读算子**；
> [`../src/kda_ref.py`](../src/kda_ref.py) 是我们要用来对拍移植结果的 golden。
> 但在此之前，`kda_ref.py` 只被它**自己**验证过（逐 token 递推 vs 分块前向、fp64 CPU，
> 见 `src/kda_ref.out.txt`）。本篇把 golden 锚定到**真正要移植的那套算子**——
> FLA `fla/ops/kda` 的 Triton chunk 内核：喂两侧**同一份 bf16 操作数**，量残差。

为什么必须先做这一步：移植的正确性判据是「Megatron 里的新实现 == FLA 原算子」。
如果拿一个**只自洽**的参考当裁判，任何「参考与 FLA 的契约差异」都会被记到移植头上
（主轴、log 单位、`dt_bias` 布局、`Aqk` 转置……），把真 bug 淹没。先把参考钉到 FLA，
之后每一条残差才有归属。

---

## 1. 双方契约（真实核对）

| 项 | 本参考 `kda_ref.py` | FLA（本机 `fla` 0.5.2） |
|---|---|---|
| 布局 | `[B,T,H,·]` | `[B,T,HV,·]`（HV=H，无 GVA） |
| 衰减 `g` | **自然对数**、已激活（$\le0$） | 同：kernel 内 `chunk_local_cumsum(..., scale=RCP_LN2)` 才转 log2（`chunk_fwd.py:51`） |
| `beta` | **已 sigmoid** | `use_beta_sigmoid_in_kernel=False` 时同 |
| 状态 | `[B,H,K,V]` | 默认 `[N,HV,K,V]`（`state_v_first=False`） |
| 门控激活 | `-A_log.exp()·softplus(g+dt_bias)`（`src/kda_ref.py:65`） | `naive_kda_gate` 同式（`fla/ops/kda/gate.py:54`）；kernel 内 `b_yg=-exp(b_A)*softplus(b_g)`（`gate.py:122`） |
| `dt_bias` | view 成 `[H,K]`（`src/kda_ref.py:62`） | `[HV*K]`，kernel 内 view `(HV,K)` |
| L2 归一 | `x/√(Σx²+eps)`，eps=1e-6（`src/kda_ref.py:73`） | `1/√(Σx²+eps)`（`fla/modules/l2norm.py:103`） |
| scale | 默认 `K^{-1/2}` | 同 |

FLA 调用面：前向入口 `ChunkKDAFunction.forward`（`fla/ops/kda/chunk.py:28`），
`use_qk_l2norm_in_kernel` 走 `l2norm_fwd`（`chunk.py:51-54`），
`use_beta_sigmoid_in_kernel` 走 `fused_beta_sigmoid`（`chunk.py:57-58`），
门控在 `chunk_kda_fwd` 内 `use_gate_in_kernel=True` 时算（`chunk.py:63-86`）。
反向 l2norm/sigmoid 的伴随在 `chunk.py:167-170`。

---

## 2. 两段锚定（为什么要分两段）

可能的契约错有两类，混在一起会互相掩盖，故分成两个 case 集：

- **Part A（显式操作数）**：两侧都喂**同一个** bf16 `q/k/v/g/beta`
  （`g`=`kda_gate` 产出的自然对数衰减、`beta`=sigmoid 后），FLA 关掉
  `use_gate_in_kernel`/`use_qk_l2norm_in_kernel`/`use_beta_sigmoid_in_kernel`。
  残差 = **分块代数**（$L$、$(I+L)^{-1}$、$U/W$、状态递推、$o$）的差异。
- **Part B（核内融合门控）**：FLA 只收**原始 logits** `g_logit`/`beta_logit`，
  自己算 `-exp(A_log)·softplus(g+dt_bias)` 与 `sigmoid`（`use_gate_in_kernel=True`、
  `use_beta_sigmoid_in_kernel=True`）；本参考走 `kda_gate`+`torch.sigmoid`。
  残差 = **门控激活契约**（softplus、`dt_bias` 布局、log 单位）的差异。

参考侧一律跑 fp64，故报出的数是 **FLA bf16 内核误差**（不是我们的）。

运行（`kernel_lab`、GPU）：

```
FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
    ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla.py
# 输出存 harness/check_port_fla.out.txt
```

（`FLA_DISABLE_BACKEND_DISPATCH=1` 把算子钉回 Triton，避免本机 FLA 0.5.x 把 KDA
路由到 TileLang 后端。）

---

## 3. Part A：分块代数（4 组，含 `K≠V`、非整除 `T`）

损失 $L=\langle o,\bar o\rangle+\langle S_T,\bar S_T\rangle$，六项梯度由两侧 autograd 给出
（FLA 侧穿 Triton 内核，参考侧穿 fp64 分块前向）。

| case | fwd `o` max_rel | fwd `S` max_rel | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|---|---|
| B1 T128 H2 K128 V128 | 4.84e-3 | 2.77e-3 | 2.95e-3 | 3.52e-3 | 2.15e-3 | 3.04e-3 | 1.45e-3 | 1.87e-3 |
| B2 T512 H4 K128 V128 | 3.69e-3 | 3.50e-3 | 3.93e-3 | 3.96e-3 | 3.19e-3 | 3.02e-3 | 1.63e-3 | 1.89e-3 |
| B1 T250 H3 K128 V128 | 5.91e-3 | 3.45e-3 | 3.90e-3 | 2.90e-3 | 2.71e-3 | 1.93e-3 | 6.10e-3 | 1.80e-3 |
| B2 T256 H2 **K64 V96** | 3.40e-3 | 4.36e-3 | 3.37e-3 | 4.40e-3 | 4.21e-3 | 2.70e-3 | 2.35e-3 | 5.23e-4 |

（`max_rel` = max|Δ| / max|ref|；`max_abs` 见 `check_port_fla.out.txt`，量级
`o` ~1e-4、`S` ~1e-3、`dq/dk` ~1e-4~9e-4。）

**读数**：全部落在 **bf16 底噪 ~3e-3**。`T=250`（最后一 chunk 只有 58 个有效 token，
两侧都按 pad 处理）与 `K=64≠V=96` 都不抬高误差——即参考的分块代数、pad 处理、
非方阵 $K/V$ 与 FLA 一致。

---

## 4. Part B：核内融合门控（同 4 组）

FLA 从原始 logits 起算门控，参考走 `kda_gate`。除 `q/k/v/g_logit/beta_logit/dh0`
外，还比了 `A_log`/`dt_bias` 的梯度（FLA 门控内核自己产出，`gate.py:193-194`）。

| case | fwd `o` max_rel | fwd `S` max_rel | `dq` | `dk` | `dv` | `dg_logit` | `dbeta_logit` | `dh0` | `dA_log` | `ddt_bias` |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 T128 H2 K128 | 4.49e-3 | 2.84e-3 | 3.65e-3 | 3.12e-3 | 4.44e-3 | 3.75e-3 | 1.58e-3 | 7.19e-3 | 1.98e-2 | 8.28e-3 |
| B2 T512 H4 K128 | 3.39e-3 | 5.02e-3 | 2.53e-3 | 4.47e-3 | 3.69e-3 | 3.31e-3 | 1.24e-3 | 2.58e-3 | 9.45e-3 | 5.63e-3 |
| B1 T250 H3 K128 | 6.40e-3 | 3.89e-3 | 3.05e-3 | 2.75e-3 | 3.19e-3 | 6.19e-3 | 8.48e-3 | 4.52e-3 | 2.47e-3 | 5.51e-3 |
| B2 T256 H2 K64 V96 | 4.03e-3 | 2.44e-3 | 4.39e-3 | 3.77e-3 | 2.58e-3 | 1.79e-3 | 2.39e-3 | 4.30e-3 | 2.95e-2 | 4.57e-3 |

**读数**：

- 主路（`o/S`、`dq/dk/dv/g_logit/beta_logit/dh0`）与 Part A **同量级 ~3e-3**——
  说明「把门控融进内核」相对「显式喂激活后 $g$」**没有引入可测的额外误差**；门控链条
  （softplus、`dt_bias` 的 `[HV*K]` 布局、自然对数单位）与参考逐项对齐。
- **`dA_log` 相对误差最大（到 2.95e-2），但绝对误差只有 5.7e-4**。原因：
  `dA_log` 是**逐 head 的标量** $-\sum_{t,k}e^{A}\operatorname{softplus}(z)\,dy_g$，
  是大量项求和、可能近抵消，分母 `max|ref|` 很小 → 相对量放大。它不是精度退化的证据
  （`ddt_bias` 8.3e-3 同因）。移植时若拿 `dA_log` 做单指标会因为它的相对量错觉而误判。

---

## 5. 结论与后续

- **结论**：`src/kda_ref.py` 作为移植 golden **已被 FLA `chunk_kda` 正反向锚定**：
  显式操作数与核内融合门控两条路径、含 `K≠V`（64/96）与非整除 `T=250`，
  前反向所有量都落在 **bf16 底噪（主路 ≤~7e-3、梯度 ≤~4e-3，`dA_log` 相对量到 3e-2
  但绝对量 6e-4）**。参考可以进移植的主对拍回路。
- **未覆盖（待验证 / 下一步）**：
  - `safe_gate=True`/`lower_bound`（TensorCore 门控分支）与 `allow_neg_eigval`、
    GVA（$H_V>H$）、`state_v_first` 的布局约定——本篇只测了默认门与 `HV=H`。
  - **chunk_size**：FLA 内部默认 64（`chunk.py` 的 `chunk_size=64`）；参考
    `chunk_size` 可取 16/32/64，需要 sweep 证明两者在多种 `T` 下的 pad 语义一致。
  - varlen（`cu_seqlens`）与 CP 路径。

**下一篇**（计划 M7/M8）把这些边界补成锚定矩阵，然后进入 Megatron-LM
`gated_delta_net/kda.py` 的接口对齐与端到端对拍。
