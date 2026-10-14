# 05 · 把锚定扩到门控/布局边界（M7）

> 系列第 6 篇。前一篇 [04](04-numeric-anchor.md) 把 [`../src/kda_ref.py`](../src/kda_ref.py)
> 锚定到 FLA `chunk_kda` 的正反向，但**只测了默认门控与 `HV=H`**。本篇把锚定扩到四个
> 边界：`safe_gate`/`lower_bound`、`allow_neg_eigval`、**GVA**（$H_V>H$）、`state_v_first`，
> 并顺手在参考实现里把后两者补齐。

---

## 1. 为什么这四个是「边界」而不是普通 case

M6 的锚定矩阵里，两侧的契约是「同一份 bf16 操作数 + 默认门 + 同 head 数」。
但 FLA `chunk_kda` 的公开签名（`fla/ops/kda/chunk.py:178`）还有四个开关，它们各自
改写**一个独立环节**，任一错都会把移植的残差归错地方：

| 开关 | 改写的环节 | 核对到的源码 |
|---|---|---|
| `safe_gate=True` + `lower_bound` | 门控激活式换成有界版 $\text{lb}\cdot\sigma(e^{A}(g+b))$，且 intra 走 TensorCore 分支 | `chunk.py:394-398`（校验）、`gate.py:124` / `gate.py:422`（公式）、`chunk_intra.py:825/881`（`USE_SAFE_GATE`） |
| `allow_neg_eigval=True` | `beta` 的 sigmoid 前乘 2 → $\beta=2\sigma(\text{logit})\in(0,2)$ | `chunk.py:62`、`chunk.py:400-401`（要求 `use_beta_sigmoid_in_kernel=True`） |
| **GVA**（$H_V>H$） | 值头 $hv$ 复用 q/k 头 $hv\,//\,(H_V//H)$；`v/g/beta/state` 用 $H_V$ 头 | `chunk_intra.py:79`（前向）、`:426`/`:705`（反向/其他 kernel） |
| `state_v_first=True` | `initial_state`/`final_state` 用 V-first `[·,V,K]`，而不是默认 `[·,K,V]` | `chunk_delta_h.py:702`、`:763` |

两点值得先说清：

- **`allow_neg_eigval` 只改 $\beta$ 的入口缩放**。它不进 `chunk_kda_fwd`/`chunk_kda_bwd`
  的实参（`chunk.py:74-96`/`135-165` 里没有它），在 `ChunkKDAFunction` 里只出现在
  `fused_beta_sigmoid(..., scale=2.0 ...)`（`chunk.py:62`）和反向 `fused_beta_sigmoid_bwd`
  （`chunk.py:170`）。所以「分块代数」一字未动，参考侧只需把 `sigmoid` 换成 `2*sigmoid`。
- **`safe_gate` 与 `lower_bound` 是两个东西**：`safe_gate` 决定 intra 用哪条 kernel
  （`chunk_intra.py:825` 的 `sub_chunk` MMA 版 vs `:848` 的 token-parallel 版），
  `lower_bound` 决定门控公式。二者由 `chunk.py:394` 绑在一起：`safe_gate=True` 且
  `use_gate_in_kernel=True` 时必须给 `lower_bound∈[-5,0)`。有界门控使相邻 token 的
  累计衰减差有界，正是那条 TensorCore 分支能安全使用的前提。

---

## 2. 参考实现补的两件事

M6 的 `kda_chunked_fwd` 假定 $H_V=H$（`v.shape[2]==q.shape[2]`）、状态恒 `[K,V]`。
M7 把它泛化（`src/kda_ref.py:144-` ）：

- **GVA**：若 $H_V\ne H$，对 q/k 做 `repeat_interleave(H_V//H, dim=2)`，之后按 $H_V$ 头
  的普通分块前向算。这一步是**精确**的：值头 $hv$ 取到 q/k 头 $hv//(H_V//H)$，与
  `chunk_intra.py:79` 的 `i_h = i_hv // (HV // H)` 一致；而 `repeat_interleave` 的 autograd
  正好是**按组求和**，所以 $\bar q/\bar k$ 自动回到原始 $H$ 头，无需手写归约。
- **state_v_first**：`initial_state` 传进来先 `.transpose(-1,-2)` 再进递推，`final_state`
  返回前再转置；`o` 不变。

为让这一点**自我成立**（不依赖 FLA），harness 的 **Part 0** 在 fp64 CPU 上验证：

```
GVA  chunked vs token-recurrence(expanded): rel|do|=7.39e-15 rel|dS|=2.22e-14
svf  o unchanged: |do|max=0.00e+00   S_svf == S^T: |d|max=0.00e+00
```

即 GVA 的分块前向 == 把 q/k 展开后逐 token 递推（$10^{-15}$），state_v_first 只转置状态、
`o` 逐位不变。

---

## 3. Part A：GVA（显式操作数）

喂同一份 bf16 `q/k`（$H$ 头）与 `v/g/beta`（$H_V$ 头），FLA 关掉核内门控；
参考侧把 q/k 展开成 $H_V$ 头跑 fp64。损失 $L=\langle o,\bar o\rangle+\langle S,\bar S\rangle$。

| case（$H_V{=}G{\cdot}H$） | fwd `o` max_rel | fwd `S` max_rel | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|---|---|
| B2 T256 H2 **HV4** K128 V128 | 3.61e-3 | 4.21e-3 | 2.92e-3 | 2.63e-3 | 3.18e-3 | 2.48e-3 | 2.11e-3 | 1.94e-3 |
| B1 T250 H2 **HV8** K64 V96 | 6.54e-3 | 4.37e-3 | 1.92e-3 | 2.90e-3 | 3.21e-3 | 2.71e-3 | 1.56e-3 | 1.68e-3 |
| B1 T512 H4 **HV8** K128 V128 | 4.02e-3 | 3.60e-3 | 3.45e-3 | 3.77e-3 | 3.39e-3 | 3.26e-3 | 3.20e-3 | 2.22e-3 |

**读数**：G=2/4、含非整除 `T=250`（HV8）都落在 **bf16 底噪 ~3–7e-3**，与 M6 的 `HV=H`
同量级——**GVA 的头映射与参考的展开式一致**，且 $\bar q/\bar k$ 的组求和正确
（若映射错，`dq/dk` 会是 O(1) 级）。

---

## 4. Part B：`state_v_first`（显式操作数）

`initial_state` 传 V-first `[B,HV,V,K]`、`final_state` 也按 `[B,HV,V,K]` 比。
参考侧传 `state_v_first=True`（内部转置）。

| case | fwd `o` max_rel | fwd `S`(V,K) max_rel | `dq` | `dk` | `dv` | `dg` | `dbeta` | `dh0` |
|---|---|---|---|---|---|---|---|---|
| B2 T256 H2 K128 V128 | 4.67e-3 | 2.48e-3 | 2.28e-3 | 3.64e-3 | 4.99e-3 | 5.65e-3 | 2.92e-3 | 2.50e-3 |
| B1 T250 H2 **HV4** K64 V96 | 7.95e-3 | 3.87e-3 | 2.48e-3 | 3.27e-3 | 3.82e-3 | 2.30e-3 | 4.77e-3 | 1.43e-3 |

**读数**：`o` 与 `S` 同落 bf16 底噪；`dh0`（对 V-first 初态的梯度）也正确。
**移植要点**：`state_v_first=True` 时，**输入初态也要转置**——本 harness 先按默认 `[K,V]`
生成 `h0` 再 `.transpose` 喂入（`check_port_fla_m7.py:run`），否则 `o` 会错到 O(1)
（早期探针 `probe_fla_flags2` 实测：喂未转置的 `h0` 时 `o` 与默认路径不一致）。

---

## 5. Part C：`safe_gate` + `lower_bound=-5`（核内融合门控）

FLA 从**原始 logits** 起算有界门 $\text{lb}\cdot\sigma(e^{A}(g+b))$，参考侧用
`gate_fp64` 同式但跑 fp64。除 `q/k/v` 外还比 `g_logit/beta_logit/dh0` 与逐 head 标量
`A_log/dt_bias` 的梯度。为免有界门把无衰减通道放大到溢出，门控参数取温和区间
（`A_log~log U(1,4)`、`g_logit~0.4·N(0,1)`，`check_port_fla_m7.py:make_inputs(tame=True)`）。

| case（lb=-5） | fwd `o` | fwd `S` | `dq` | `dk` | `dv` | `dg_logit` | `dbeta_logit` | `dh0` | `dA_log`(abs) | `ddt_bias`(abs) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 T256 H2 K128 | 6.13e-3 | 2.54e-3 | 5.19e-3 | 4.04e-3 | 4.88e-3 | 1.83e-3 | 1.66e-3 | 2.71e-3 | 5.65e-4 | 7.99e-4 |
| B1 T250 H2 K64 V96 | 4.58e-3 | 3.32e-3 | 1.76e-3 | 2.77e-3 | 3.64e-3 | 3.63e-3 | 4.49e-3 | 2.08e-3 | 7.49e-4 | 1.17e-3 |
| B1 T128 H4 K128 | 4.72e-3 | 2.74e-3 | 3.20e-3 | 4.42e-3 | 2.75e-3 | 1.71e-3 | 1.96e-3 | 2.44e-3 | 3.95e-4 | 8.99e-4 |
| B2 T256 H2 **HV4** K128 | 7.16e-3 | 2.78e-3 | 3.01e-3 | 3.01e-3 | 4.14e-3 | 2.61e-3 | 2.08e-3 | 2.74e-3 | 2.42e-4 | 9.90e-4 |

（表中 `dA_log`/`ddt_bias` 列给 **max_abs**；其 max_rel 到 2.7e-2 / 3.6e-2，原因见下。）

**读数**：
- 主路（`o/S`、五项元素梯度）与 M6 的默认门控**同量级 ~3–7e-3**——**有界门公式
  $\text{lb}\cdot\sigma(e^{A}z)$ 与参考一致**，`safe_gate` 走的 intra MMA 分支
  （`chunk_intra.py:825`）与 token-parallel 分支代数等价。
- `dA_log`/`ddt_bias` 的**相对**量到 2.7e-2–3.6e-2，但**绝对只有 4e-4–1.2e-3**：
  它们是逐 head 标量（$\partial A=\sum_{t,k}\partial g\cdot(g+b)$ 之类）大量项求和、
  近抵消，分母 `max|ref|` 极小。判据同 M6 §4：**这两项看绝对、不看相对**。

---

## 6. Part D：`allow_neg_eigval`（$\beta=2\sigma$）

FLA 从原始 `beta_logit` 起算 $\beta=2\sigma$（`chunk.py:62`），参考侧同样乘 2。

| case | fwd `o` | fwd `S` | `dq` | `dk` | `dv` | `dg_logit` | `dbeta_logit` | `dh0` | `dA_log`(abs) | `ddt_bias`(abs) |
|---|---|---|---|---|---|---|---|---|---|---|
| B2 T256 H2 K128 | 4.98e-3 | 3.30e-3 | 3.23e-3 | 2.93e-3 | 6.04e-3 | 4.83e-3 | 3.71e-3 | 2.04e-3 | 4.16e-4 | 1.12e-4 |
| B1 T250 H2 K64 V96 | 5.68e-3 | 3.61e-3 | 3.51e-3 | 4.69e-3 | 3.42e-3 | 4.49e-3 | 1.67e-3 | 2.66e-3 | 3.10e-4 | 1.03e-4 |
| B1 T250 H2 **HV4** K64 V96 | 4.82e-3 | 4.21e-3 | 3.82e-3 | 3.66e-3 | 4.10e-3 | 3.08e-3 | 2.04e-3 | 2.72e-3 | 1.01e-3 | 1.76e-4 |

**读数**：$\beta\in(0,2)$ 时 $|S|$ 抬到 ~1–1.6（门控无衰减时状态累积更久），但残差仍
落 bf16 底噪 ≤6e-3，`dA_log`/`ddt_bias` 这次连绝对都小。**$\beta$ 的 `×2` 入口缩放
与参考的 `2*sigmoid` 一致**。

---

## 7. 结论与后续

- **结论**：M7 把参考实现从「只测默认门 + `HV=H`」扩到四个边界，全部锚定到
  **FLA bf16 底噪**（主路 ≤~8e-3；`dA_log`/`ddt_bias` 相对量到 3.6e-2 但绝对 ≤1.2e-3）。
  参考新增的 GVA（`repeat_interleave` 组映射）与 `state_v_first`（转置布局）在 fp64 CPU 上
  与「展开后逐 token 递推」自洽到 $10^{-15}$。**M6 默认路径的输出逐位不变**
  （重跑 `check_port_fla.py` 与旧 `check_port_fla.out.txt` diff 为空）。
- **移植要点（可写进 `docs/02` §7 检查表）**：
  1. GVA 用「值头 $hv$ 复用 q/k 头 $hv//(H_V//H)$」的**连续分组**；反向对 $\bar q/\bar k$
     要**按组求和**（组内多个值头共享一个 q/k 头）。
  2. `state_v_first` 是**双端**的：`initial_state` 入参也要是 `[·,V,K]`。
  3. `safe_gate`/`lower_bound` 是「内核分支 + 有界门公式」两件事，移植时别只做其一。
  4. `allow_neg_eigval` 只改 $\beta$ 入口（`×2`），门控与分块代数不变。
- **未覆盖（下一步 M8/M9/M10）**：`chunk_size∈{16,32,64}` × `T` 的 pad 语义 sweep、
  varlen（`cu_seqlens`）、CP；Megatron `kda.py` 调用契约逐项对齐；端到端对拍。

运行：

```
FLA_DISABLE_BACKEND_DISPATCH=1 python3 \
    ~/proj/tech_record/code/flash-kda/megatron-port/harness/check_port_fla_m7.py
# 输出存 harness/check_port_fla_m7.out.txt（Part 0 为 fp64 CPU，A–D 为 GPU）
```
