# MoE 反向：dgrad / wgrad 数学与 grouped GEMM 映射

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 9 篇。前八篇把前向的
> dispatch → linear1 → SwiGLU → linear2 → combine（以及它们的融合实现）讲完了；从这篇开始进入**反向**。
> 篇幅安排上，本文（路线图 **P4-1**）先把**回传的数学**逐层推导清楚、并说清它为什么天然对应
> DeepGEMM 的两种 grouped GEMM（wgrad 走 `k_grouped`、dgrad 走 `m_grouped`），并用 autograd
> 对拍给出真实误差；后续 P4-2/P4-3 再逐行精读对应 API 与 kernel，P4-4 做 kernel 级数值实验。
>
> 本文为**纯数学 + 数值对拍**，不依赖 SM100；实验 `harness/exp_moe_backward_math.py` 在 H100
> 容器里跑（float64，CPU 即可复现）。

## 1. 记号与前向回顾

固定一个 MoE 层：hidden 维 $H$、专家中间维 $I$（SwiGLU 门 + 值两条线性层，故 up 矩阵宽 $2I$）、
专家数 $E$、每 token 选 $k$ 个专家（top-k）。对某个 token $x\in\mathbb{R}^{H}$，

$$
\text{router:}\quad \ell = xW_g\in\mathbb{R}^{E},\qquad p=\mathrm{softmax}(\ell),\qquad
\mathcal{S}=\operatorname{topk}_k(p),
$$

$$
y \;=\; \sum_{j\in\mathcal{S}} \tilde p_j\,\mathrm{FFN}_j(x),\qquad
\tilde p_j = \frac{p_j}{\sum_{i\in\mathcal{S}} p_i},\qquad
\mathrm{FFN}_j(x)=W^{2}_j\,\big(\mathrm{SwiGLU}(xW^{1}_j)\big).
$$

把 experts 的权重堆叠起来：$W^1\in\mathbb{R}^{E\times H\times 2I}$、$W^2\in\mathbb{R}^{E\times I\times H}$。
对单个专家 $j$，记

$$
G_j = xW^{1}_j\in\mathbb{R}^{2I},\quad
[G_G;G_U]=G_j\ (\text{gate/up 各 }I),\quad
A_j=\mathrm{silu}(G_G)\odot G_U\in\mathbb{R}^{I},\quad
Y_j=A_jW^{2}_j\in\mathbb{R}^{H}.
$$

$\mathrm{silu}(z)=z\,\sigma(z)$，$\sigma$ 为 sigmoid。

**训练前向**（DeepGEMM 的 `m_grouped_fp8_fp4_gemm_nt_contiguous`，`csrc/apis/gemm.hpp:181`）把
所有 token 按专家分组，M 轴 = token、N/K 固定：linear1 是 `[tokens,H] @ [H,2I]`，linear2 是
`[tokens,I] @ [I,H]`，两组都只分组 token（M）轴。Mega MoE 把这五段塞进一个 kernel（SM100 专属），
其中 SwiGLU 在 L1 epilogue 里 **in-place** 做（`sm100_fp8_fp4_mega_moe.cuh:1108-1134`，
`silu(gate)*up` 后乘 topk 权重、并在量化前 clamp），细节见 `docs/10`。

## 2. 反向全景

设上游梯度 $dz = \partial L/\partial y$（形状 `[T,H]`）。反向沿前向逆序、逐层回传：

```
dz [T,H]
  │  combine 反向：dY_(t,j) = w_tj · dz_t，并回传 topk 权重 q_tj = <Y_(t,j), dz_t>
  ▼
dY [T,E,H]
  │  linear2 反向
  ├─► dW2[e] = A_e^T @ dY_e        （wgrad：K=token 轴分组，M=I, N=H 固定）→ k_grouped
  └─► dA    = dY @ W2^T            （dgrad：M=token 轴分组，N=I, K=H 固定）→ m_grouped
  ▼
dA [T,E,I]
  │  SwiGLU 反向
  ▼
dG [T,E,2I]  = concat(dG_G, dG_U)
  │  linear1 反向
  ├─► dW1[e] = X_e^T @ dG_e        （wgrad：K=token 轴分组，M=H, N=2I 固定）→ k_grouped
  └─► dX    = dG @ W1^T + dX_router（dgrad）→ m_grouped
  ▼
dX [T,H]  +  router 反向 → dWg
```

关键观察（贯穿全专题）：**wgrad 在 token 轴（每个专家处理的样本数）上收缩，这个轴各专家不等长，
所以叫 K 轴分组（`k_grouped`）；dgrad 在 hidden/中间维上收缩（各专家固定），但输出的 token 行按专家
不等长分组，所以是 M 轴分组（`m_grouped`）。** 这正好解释了 DeepGEMM README 第 80 行那句
「K-axis-grouped API for MoE weight backward (with M and N must remain fixed)」。

## 3. 逐算子反向推导

下面所有式子用「逐元素 + 每 token 只对其被选中的专家有贡献」的口径。为了可读，按 (token, expert)
对展开成 `[T,E,...]`；未选中的 $j\notin\mathcal{S}(t)$ 上所有量取 0。

### 3.1 combine 反向

前向 $y_t=\sum_{j\in\mathcal{S}} \tilde p_{tj} Y_{tj}$。直接得

$$
\frac{\partial L}{\partial Y_{tj}} = w_{tj}\, dz_t,\quad w_{tj}=\begin{cases}\tilde p_{tj} & j\in\mathcal{S}(t)\\0&\text{else}\end{cases}
\tag{1}
$$

$$
q_{tj} := \frac{\partial L}{\partial \tilde p_{tj}} = \langle Y_{tj},\, dz_t\rangle.
\tag{2}
$$

（$q$ 只在被选中的 $j$ 上非零，用于 §3.5 的 router 反向。）

### 3.2 linear2 反向

$Y_{tj}=A_{tj}W^2_j$。对整批 token 按专家分组：

$$
\boxed{\;dA_{tj}=dY_{tj}\,(W^2_j)^{\mathsf T}=\frac{\partial L}{\partial A_{tj}}\;}
\qquad
\boxed{\;dW^2_j=\sum_{t} A_{tj}^{\mathsf T}\,dY_{tj}\;}
\tag{3}
$$

- $dA$：`[T,E,I]`。分组看是 **m_grouped**——M=该专家收到的 token 数（不等长）、N=$I$、K=$H$ 固定。
- $dW^2$：`[E,I,H]`。分组看是 **k_grouped**——第 $j$ 组是「所有把 $j$ 选进 topk 的 token」，
  把这些 token 的 $A_{(\cdot)j}$ 行拼成 $[n_j,I]$、$dY_{(\cdot)j}$ 行拼成 $[n_j,H]$，
  收缩维 $n_j$（token 数）随专家变化 → K 轴分组，M=$I$、N=$H$ 固定。

### 3.3 SwiGLU 反向

$A_G=\mathrm{silu}(G_G)$、$A_U=G_U$、$A=A_G\odot A_U$。逐元素：

$$
\frac{\partial A}{\partial G_G}=\frac{\partial A}{\partial A_G}\cdot\mathrm{silu}'(G_G)=dA\odot G_U\odot \mathrm{silu}'(G_G),\qquad
\frac{\partial A}{\partial G_U}=\frac{\partial A}{\partial A_U}\cdot\mathrm{silu}(G_G)=dA\odot\mathrm{silu}(G_G),
$$

$$
\mathrm{silu}'(z)=\sigma(z)\bigl(1+z\,(1-\sigma(z))\bigr).
\tag{4}
$$

于是 $dG=[dG_G;dG_U]$，两半拼回 `[T,E,2I]`。前向里 SwiGLU 是 **in-place**、带 `activation_clamp`：

```
bf16_gate = min(gate, clamp); bf16_up = clamp(up);  // kernel:1116-1120
activation_values = silu(gate) * up * weight;      // kernel:1122-1134
```

注意前向在 **bf16** 上做 gate/up 取整再算 SwiGLU（`docs/10` §SwiGLU），反向若要严格对齐，
SwiGLU 的局部导数应在同一 bf16 取整后的值上算；作为数学参考本文在 float 上推导。

### 3.4 linear1 反向

$G_{tj}=x_t W^1_j$：

$$
\boxed{\;dX_{tj}=dG_{tj}\,(W^1_j)^{\mathsf T}\;}\qquad
\boxed{\;dW^1_j=\sum_{t} x_t^{\mathsf T}\,dG_{tj}\;}
\tag{5}
$$

- $dX$：`[T,E,H]`，对 $j$ 求和得 `[T,H]`；分组看是 **m_grouped**（M=token 不等长，N=$H$，K=$2I$）。
- $dW^1$：`[E,H,2I]`，**k_grouped**（K=token 数，M=$H$，N=$2I$ 固定）。

### 3.5 router / topk 反向

这是最容易写错的一块。前向：$\ell_t=x_tW_g$、$p_t=\mathrm{softmax}(\ell_t)$、硬 topk 选 $\mathcal{S}$、
再归一化 $\tilde p_{tj}=p_{tj}/P_t$，其中 $P_t=\sum_{i\in\mathcal{S}}p_{ti}$。topk **选择**不可导
（straight-through：索引当常数），梯度只经**数值**路径回传。

记 $q_{tj}$ 如 (2)。$\tilde p_{ti}=p_{ti}/P_t$ 对选中的 $p_{tj}$ 的偏导：

$$
\frac{\partial \tilde p_{ti}}{\partial p_{tj}}=\frac{\delta_{ij}P_t-p_{ti}}{P_t^{2}}\quad(i,j\in\mathcal{S}),
$$

故

$$
\frac{\partial L}{\partial p_{tj}}=\frac{q_{tj}P_t-\sum_{i\in\mathcal{S}}q_{ti}\,p_{ti}}{P_t^{2}}
=\frac{q_{tj}P_t-A_t}{P_t^{2}},\qquad A_t:=\sum_{i\in\mathcal{S}}q_{ti}p_{ti},
\qquad j\in\mathcal{S};\quad 0\ \text{else}.
\tag{6}
$$

再做 **softmax 的 Jacobian**（对全部 $E$ 个 logits，未选中的 $p$ 也参与 softmax）：

$$
\frac{\partial L}{\partial \ell_{tl}}=p_{tl}\Bigl(\frac{\partial L}{\partial p_{tl}}-\sum_{m}p_{tm}\frac{\partial L}{\partial p_{tm}}\Bigr).
\tag{7}
$$

最后

$$
dW_g=\sum_t x_t^{\mathsf T}\,d\ell_t,\qquad dx^{\text{router}}_t=d\ell_t\,W_g^{\mathsf T}.
\tag{8}
$$

> **为什么容易错**：把 (6) 的分母 $P_t$ 误写成「归一化后的 topk 权重的和」（恒为 1）。分母必须是
> **原始 softmax 概率在选中集上的和** $P_t=\sum_{i\in\mathcal{S}}p_{ti}$——这正是推导里最容易踩的坑，
> 本文实验第一版就因此让 `dx`/`dWg` 偏到 $10^{-2}$ 量级（见 §5 注）。
>
> DeepSeek 的**共享专家**（shared expert）不参与 topk 竞争，其输出直接加到 $y$；
> 反向时它是一条**稠密**路径（无 (6)(7)，直接走 §3.2–§3.4），在实现里通常单独一个 dense GEMM。

## 4. 为什么 wgrad 是 `k_grouped`、dgrad 是 `m_grouped`

把 (3)(5) 写成 GEMM 的 $(M,N,K)$ 角色，就能直接对上 DeepGEMM 的两族 API：

| 反向 | 式子 | M | N | K（收缩维） | 分组轴 | DeepGEMM API |
|---|---|---|---|---|---|---|
| linear1 wgrad | $dW^1_j=\sum_t x_t^{\mathsf T}dG_{tj}$ | $H$ | $2I$ | token 数 $n_j$（**不等长**） | K | `k_grouped_*` |
| linear2 wgrad | $dW^2_j=\sum_t A_{tj}^{\mathsf T}dY_{tj}$ | $I$ | $H$ | token 数 $n_j$（**不等长**） | K | `k_grouped_*` |
| linear1 dgrad | $dX_{tj}=dG_{tj}(W^1_j)^{\mathsf T}$ | token 数 $n_j$（**不等长**） | $H$ | $2I$（固定） | M | `m_grouped_*` |
| linear2 dgrad | $dA_{tj}=dY_{tj}(W^2_j)^{\mathsf T}$ | token 数 $n_j$（**不等长**） | $I$ | $H$（固定） | M | `m_grouped_*` |

前向与 dgrad 都把「不等长」放在 **M（token）轴**，所以共用 `m_grouped` 族（contiguous/masked/PSUM
三种布局，见 `docs/01` §1）；wgrad 把「不等长」放在 **收缩维 K（token）轴**，于是需要专门的
`k_grouped` 族。DeepGEMM 里对应的入口是
`csrc/apis/gemm.hpp:320` 的 `k_grouped_fp8_gemm_tn_contiguous`（SM100）与 `:373` 的
`k_grouped_fp8_gemm_nt_contiguous`（SM90，本机 H100 可跑）。二者在 host 侧只做形状/对齐校验：

```
// csrc/apis/gemm.hpp:340-349
// A/B use MN-major layouts `[sum_k, M]` and `[sum_k, N]`
const auto [num_groups, m, n] = get_shape<3>(d);
const auto [sum_k_ , m_] = get_shape<2>(a.first);
const auto [sum_k__, n_] = get_shape<2>(b.first);
const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups, ...);
DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);
```

即：**N/M 固定、K=sum_k 是各组 token 数按 `k_alignment` 补齐后的和**。`ks_cpu` 就是每组的
（逻辑）K 大小，`check_k_grouped_args`（`gemm.hpp:55-76`）校验每个 $k$ 是 `k_alignment` 的倍数。
DeepGEMM 测试里对 K-grouped GEMM 的参考实现正是逐组 $A_e^{\mathsf T}B_e$：

```python
# tests/generators.py:583-587
for i, (group_k, end) in enumerate(zip(logical_ks_cpu, logical_group_ends)):
    start = end - group_k
    a[start:end] = torch.randn((group_k, m), ...)   # a: [sum_k, m]
    b[start:end] = torch.randn((group_k, n), ...)   # b: [sum_k, n]
    ref_d[i] = (a[start:end].float().T @ b[start:end].float() + ...)  # D[e] = A_e^T @ B_e
```

这与我们 (3)(5) 的 wgrad 完全一致。

## 5. H100 数值对拍（真实数字）

实验 `harness/exp_moe_backward_math.py`（输出 `harness/exp_moe_backward_math.out.txt`）用
**float64**、$T{=}11,H{=}16,I{=}8,E{=}5,k{=}2$ 的小 MoE：前向用可微 torch 算子写，autograd 给参考梯度，
手写反向（(1)–(8)）逐层对比；再独立做中心差分抽查。

**手写反向 vs autograd（float64）**：

| 梯度 | 形状 | relerr | max_abs |
|---|---|---|---|
| `dx`（含 router 路径） | (11,16) | 2.89e-16 | 1.71e-13 |
| `dWg`（router 权重） | (16,5) | 2.52e-16 | 1.14e-13 |
| `dW1` | (5,16,16) | 1.30e-16 | 1.42e-14 |
| `dW2` | (5,8,16) | 0.00e+00 | 0.00e+00 |

四个梯度全部对齐到 **机器精度**（~1e-16），说明 (1)–(8) 的推导（含 §3.5 的归一化 topk Jacobian）正确。
**中心差分抽查**（每张量 3 个元素）误差 1.9e-8 ~ 5.7e-8，与 $O(\epsilon^2)$ 一致。

**grouped 视角**：把每个专家的 token 行拼成 $[n_j,\cdot]$ 并按 `k_alignment=128` 对齐，
逐组算 $A_e^{\mathsf T}B_e$，得到的 `dW1`/`dW2` 与 autograd 的相对误差分别是
**1.30e-16 / 0.00e+00**——即 §4 的 `k_grouped` 映射在数值上成立。本组 22 个 (token,expert) 槽、
5 个专家各补到 128，故物理 `sum_k=640`、gap 占 $1-22/640\approx96.6\%$（小形状下的极端值；
真实 MoE 每专家数百 token 时 gap 比例才会降到个位数）。

> **踩坑记录（第一版）**：(6) 的分母写成「归一化权重之和」后，`dW1/dW2` 仍精确（它们不依赖 router），
> 但 `dx`/`dWg` 的 relerr 掉到 1.2e-2 / 9.4e-3、中心差分差 0.38。改成 $P_t=\sum_{i\in\mathcal{S}}p_{ti}$
> 后归零。这正是 §3.5 标注的那个坑的实测复现。

## 6. 与融合 kernel 的对应关系

- **反向没有 mega-kernel**：Mega MoE 的融合价值主要在前向「通信-计算重叠」。反向目前由一串
  独立 grouped GEMM 组成（DeepGEMM PR [#432](https://github.com/deepseek-ai/DeepGEMM/pull/432)
  称反向 GEMM 约提速 20%；最早的 wgrad kernel 见 PR
  [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95)）。wgrad/dgrad 的 kernel 级实现、
  以及 SM90 上 `k_grouped` 切组改 tensormap 的竞态修复（PR
  [#343](https://github.com/deepseek-ai/DeepGEMM/pull/343)）留到 P4-2/P4-3。
- **EP 下的反向通信**：前向 dispatch 把 token 送到专家所在 rank、combine 把结果送回；
  反向对称——`dy` 需沿 combine 的反向送回专家 rank（形状/路由同前向 dispatch），算出
  $dA/dW$ 后 $dA$ 再沿 dispatch 的反向 combine 回 token rank 得 $dX$。**wgrad 留在专家本地**：
  每个专家只被其 owner rank 处理，故 $dW$ 天然完整、无需在 EP 组内 all-reduce（数据并行时才需要）。
  DeepEP 的 `EPBuffer.lb_reduce_grads` 处理**冗余专家**的 FP32 梯度累加
  （`docs/01` §5.4、`docs/07`）。

## 7. 下一步

- **P4-2**：`k_grouped_fp8_gemm_tn_contiguous`（`gemm.hpp:320-371`）与 SM90 `nt` 版精读——
  `tn`/`nt` 的 major 差异、`ks_cpu`/`grouped_layout`/PSUM、SF 的 `transform_k_grouped_*` 打包。
- **P4-3**：dgrad `m_grouped_*` 与 contiguous/masked 布局在训练中的用法。
- **P4-4**：H100 上用 SM90 `k_grouped_fp8_gemm_nt_contiguous` 做 wgrad 的 kernel 级数值对拍 vs PyTorch。
