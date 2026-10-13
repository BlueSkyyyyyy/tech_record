# Mega Gate 路由的反向：sqrt-softplus 打分、带 bias 排名与 top-k 归一化权重的梯度

> 第 4 篇（P4-1）把 MoE 反向的数学逐层推导清楚了，其中 §3.5 处理的是 **softmax + 硬 top-k** 这一路
> router 反向；第 5/13 篇（P5-1）又把 **Mega Gate** 的前向拆透了——但那是 `sqrt(softplus(·))` 打分、
> **带 bias 排名、不带 bias 出权重**的另一套路由语义，它**只有前向、没有反向**（DeepGEMM 不实现
> gate 反向，且 kernel 为 SM100 专属）。于是「这套打分的梯度应该长什么样」一直是个空白。
>
> 本篇补上：把 Mega Gate 的打分与权重语义逐行钉死后，**推导它的反向**，并在 H100(sm90) 上用
> **真实设备代码 + torch autograd** 双向对拍。所有数字来自
> `harness/exp_mega_gate_backward.{sh,py,out.txt}` 与 `src/mega_gate_backward.cu`（DeepGEMM 2.8.1）。
> 打分/权重语义架构无关、可真机验证；主 kernel（SM100 UMMA）**仅分析**。

---

## 1. 前向语义（逐行钉死）

对 token $x\in\mathbb{R}^{H}$、路由权重矩阵 $W_g\in\mathbb{R}^{H\times E}$（`weight` 形状 `[E,H]`，
`mega_gate.hpp:93`），Mega Gate 一次算出 `(topk_idx, topk_weights)`。三个易混的量：

**(a) 打分 logits 与打分函数**

$$
z_e=\langle x, W_{g,e}\rangle,\qquad s_e=\operatorname{ssp}(z_e):=\sqrt{\operatorname{softplus}(z_e)} .
$$

实现里 $\operatorname{ssp}$ 带一个**阈值分支**（`epilogue/sm100_mega_gate.cuh:19-24`）：

```
softplus_precise = log1pf(expf(value));
softplus = value > 20.0f ? value : softplus_precise;   // :21
return sqrtf(softplus);                                 // :23
```

即 $z>20$ 时直接返回 $\sqrt{z}$（此时 $\operatorname{softplus}(z)\approx z$）。这是**实现所定义**的
分段函数——反向求导必须按这条分段求，而不是"真正的" $\sqrt{\operatorname{softplus}}$。

**(b) 排名用 `score + bias`，出权重用"不带 bias 的 score"**

这是 Mega Gate 最反直觉的一点，源码注释写得非常直白（`mega_gate.hpp:56-57`）：

> The kernel ranks experts on **score + bias**, while the emitted weights are the **unbiased scores**
> normalized over the top-k sum and scaled by `routed_scaling_factor`.

落到代码：排名时 `scores_local[..] = unbiased_score + bias_local[..]`（`impls/sm100_bf16_mega_gate.cuh:397-398`），
但选出的槽只回读 **unbiased** 的 `selected_unbiased_score`（`:412-419`），**bias 不参与权重**。
无 bias 时连打分函数都可以省——因为 $\operatorname{ssp}$ 单调（`:420-423` 注释）：

> Without a bias the ranking is on raw scores (sqrt-softplus is monotonic), only the selected ones get activated.

即**无 bias 时按 $z$ 排名 ≡ 按 $s$ 排名**；$\operatorname{ssp}$ 只作用在最终选中的 $k$ 个槽上。

**(c) top-k 权重：选中 score 的归一化**

`store_topk_token`（`epilogue:187-216`）：

```
topk_sum = warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;   // :212
selected_weight = lane_idx < kNumTopk
    ? selected_unbiased_score / topk_sum * args.routed_scaling_factor      // :213
    : 1.0f;                                                                // 共享专家槽
```

于是对 token 的选中集 $\mathcal{T}$（$|\mathcal{T}|=k$）：

$$
w_i=g\cdot\frac{s_i}{S},\qquad S=\Big(\sum_{j\in\mathcal{T}} s_j\Big)+\epsilon,\quad \epsilon=10^{-20},\quad g=\text{routed-scaling}.
$$

注意三点：① 分母是**未归一化的 score 和** $S$（不是 softmax 概率在选中集上的和——那正是第 4 篇
§3.5 的"最容易踩的坑"，这里换成了另一个"原始量"）；② $1{k}$ 里叠加 $\epsilon=10^{-20}$ 防 0/0；
③ `use_shared_as_routed` 时，共享专家占据最后 `num_shared_experts` 个槽、它们的权重恒为 **1.0**、
索引固定（`epilogue:194-195`），不参与归一化也不参与排名。

---

## 2. 反向：三处非平凡

### 2.1 排名是硬选择（straight-through）→ bias 对梯度"透明"

top-k **索引**当常数（与第 4 篇一致）。选中的集合 $\mathcal{T}$ 由 $r_e=s_e+b_e$ 的排名决定，
但 $r$ 的**比较**不可导，所以：

- 排名与出权重**用的是两个不同的量**（$s_e+b_e$ vs $s_e$），bias 只改"谁入选"，不进权重表达式；
- 因此 **bias 对梯度路径完全透明**——无论 `with bias` 还是 `no bias`，只要**选中集相同**，$dz$ 逐位相同
  （见 §3 的 D 组：两组独立随机选中集，`dz` relerr 都 ~1.1e-7，正是 fp32 地板）；
- 若 bias 是可训练参数，它的梯度只经 `w_i` 的 $s_i$ **间接**回流，而 **不**经排名比较回流（比较不可导）；
- 无 bias 时，因 $\operatorname{ssp}$ 单调，**按 $z$ 选 == 按 $s$ 选**（§3 B 组：0/4096 不符）。

### 2.2 归一化 Jacobian（$k$ 个槽互相耦合）

$w_i=g\,s_i/S$ 里 $S$ 同时依赖**所有**选中槽，所以 $\partial w_i/\partial s_j=g(\delta_{ij}S-s_i)/S^2$。
给定上游 $\mathrm{d}y_i:=\partial L/\partial w_i$：

$$
\boxed{\;\mathrm{d}s_i=\frac{g}{S^2}\Big(\mathrm{d}y_i\,S-\sum_{j\in\mathcal{T}}\mathrm{d}y_j\,s_j\Big),
\qquad i\in\mathcal{T};\quad \mathrm{d}s_i=0\ (i\notin\mathcal{T}).\;}
\tag{1}
$$

这是第 4 篇式 (6) 的**同构**：那里分母是被 softmax 概率在选中集上的和、并用 $\sum q_{ti}p_{ti}$ 的
修正项表达；这里分母换成 score 和、修正项是 $\sum \mathrm{d}y_j s_j$。**形式不同、结构相同**：
"归一化把 $k$ 个槽拧在一起"。

### 2.3 sqrt-softplus 的分段导数

由 §1(a) 的实现：

$$
\operatorname{ssp}'(z)=\begin{cases}
\dfrac{1}{2\sqrt{z}}, & z>20,\\[6pt]
\dfrac{\operatorname{sigmoid}(z)}{2\sqrt{\operatorname{softplus}(z)}}, & z\le 20,
\end{cases}
\qquad \operatorname{sigmoid}(z)=\dfrac{e^{z}}{1+e^{z}} .
\tag{2}
$$

在 $z=20$ 处两段**连续**（$\operatorname{sigmoid}(20)\approx 1$、$\operatorname{softplus}(20)\approx 20$），
§3 探针坐实：`d(ssp)/dz` 在 19.999/20.000/20.001 上为 0.111806/0.111803/0.111801。

### 2.4 合起来

$$
\boxed{\;\mathrm{d}z_i=\mathrm{d}s_i\cdot\operatorname{ssp}'(z_i)\ (i\in\mathcal{T}),\quad
\mathrm{d}W_g=\sum_t x_t^{\mathsf T}\mathrm{d}z_t,\quad
\mathrm{d}x_t=\mathrm{d}z_t\,W_g^{\mathsf T}.\;}
\tag{3}
$$

（$\mathrm{d}W_g$ 形状 `[H,E]`；未选中专家 $\mathrm{d}z_e=0$，因为它们的 $s_e$ 不进任何 $w_i$。）

**与完整 MoE 反向的接口**：在训练图里 $\mathrm{d}y$ 不是随便给的——Mega MoE 把 top-k 权重**提前乘进
L1 输出**（`sm100_fp8_fp4_mega_moe.cuh:1134`，见第 111 篇 §1），于是 combine 反向退化成 $\mathrm{d}Y$ 直传、
而 $\mathrm{d}y_i=\langle Y_i,\mathrm{d}z_t\rangle$（combine 反向对 $\tilde p$ 的偏导，第 4 篇式 (2)）。
**本篇验证的是从 $\mathrm{d}y$ 到 $\mathrm{d}z/\mathrm{d}W_g/\mathrm{d}x$ 这一段**（路由 scorer 的反向），
与第 4 篇的 linear/SwiGLU 反向拼起来就是完整一层。

---

## 3. H100 真机实测

`harness/exp_mega_gate_backward.sh` 在 kernel_lab(H100) 上编译运行 `src/mega_gate_backward.cu`
（**逐行移植** `sqrt_softplus`(:19-24) 与"归一化 + bias 排名"语义）并跑
`harness/exp_mega_gate_backward.py`（手写反向 vs torch autograd，float64）。原始输出见
`harness/exp_mega_gate_backward.out.txt`。

**A. 前向与解析导数 vs double**（$n=2^{22}$，$z\sim U[-60,60]$）：

```
forward  max_abs_err = 2.5244e-07     (fp32 舍入地板)
grad     max_abs_err = 6.4290e-08
阈值 20 探针: z=19.999→d=0.111806 ; z=20.000→0.111803 ; z=20.001→0.111801   (连续)
```

**B. 排名单调等价性**（$T=4096,E=128,\text{topk}=8$）：无 bias 时"按 $z$ 选 vs 按 $s$ 选"
**0/4096 不符**；有 bias 时"按 $z$ 选 vs 按 $s+b$ 选"**4096/4096 不符**——坐实 §2.1 的两个论断。

**C. 归一化 Jacobian**（$k=8,g=1.5,\epsilon=10^{-20}$）：(1) 式 vs double 有限差分
`max_abs_err = 5.13e-11`。

**D. 整 token gate 反向**（$T=4096,H=64,E=128,\text{topk}=6,g=1.5$）：device(fp32) 产出 $\mathrm{d}z$，
host 用 double 参考算 $\mathrm{d}W_g/\mathrm{d}x$ 对拍：

| 配置 | $\mathrm{d}z$ relerr | $\mathrm{d}W_g$ relerr | $\mathrm{d}x$ relerr |
|---|---|---|---|
| no bias | 1.1413e-07 | 1.1387e-07 | 1.1987e-07 |
| with bias | 1.1173e-07 | 1.1172e-07 | 1.1359e-07 |

两组都落在 fp32 舍入地板，且 **with/no bias 的差异只是各自随机的选中集**——印证 §2.1 "bias 对梯度透明"。

**E. $\epsilon=10^{-20}$ 的影响**：$\mathrm{d}z$ 与 $\epsilon=0$ 时 relerr **0.0e+00**（在 score 量级下
$10^{-20}$ 远小于 fp32 可辨精度，纯防 0/0）。

**F. 手写反向 vs torch autograd**（float64，6 组形状，含/不含 bias、不同 scaling）：

```
T=  64 H= 32 E= 64 K=4 scaling=1.5 bias=0  dWg 2.35e-16  dx 2.25e-16  dz 1.97e-16
T=  64 H= 32 E= 64 K=4 scaling=1.5 bias=1  dWg 2.72e-16  dx 2.22e-16  dz 1.97e-16
T= 256 H= 64 E=128 K=6 scaling=1.5 bias=0  dWg 2.37e-16  dx 2.24e-16  dz 1.95e-16
T= 256 H= 64 E=128 K=6 scaling=2.0 bias=1  dWg 2.64e-16  dx 2.12e-16  dz 1.78e-16
T= 512 H=128 E=256 K=8 scaling=1.5 bias=1  dWg 2.88e-16  dx 2.27e-16  dz 1.91e-16
T=2048 H=128 E=256 K=6 scaling=1.0 bias=0  dWg 2.47e-16  dx 2.13e-16  dz 1.80e-16
```

六组全部对齐到**机器精度**（~2e-16），说明 (1)–(3) 的推导正确。autograd 里 top-k 索引保持
`detached`（straight-through），因此 autograd 也只经数值路径回传——与本篇语义一致。
另做 $T=8,E=32,K=5$ 的**中心差分**抽查：解析 Jacobian vs 中心差分 `max_abs_err = 3.54e-09`。

---

## 4. 与第 4 篇（softmax router）的对照

| | 第 4 篇 softmax top-k | 本篇 Mega Gate（sqrt-softplus） |
|---|---|---|
| 打分 | $p=\mathrm{softmax}(xW_g)$ | $s=\sqrt{\operatorname{softplus}(xW_g)}$（分段） |
| 排名 | 按 $p$ | 按 $s+b$（有 bias）/ 按 $z$（无 bias，单调等价） |
| 选中集权重 | $\tilde p_i=p_i/\sum_{j\in\mathcal{T}}p_j$ | $w_i=g\,s_i/(\sum s_j+\epsilon)$ |
| 归一化反向 | 分母 $P_t=\sum_{\mathcal{T}}p$，修正项 $\sum q p$ | 分母 $S=\sum_{\mathcal{T}}s+\epsilon$，修正项 $\sum \mathrm{d}y\,s$ |
| 额外链 | softmax Jacobian（含未选中 $p$） | $\operatorname{ssp}'$ 分段导数 + bias 透明 |
| 有无实现 | 训练 router（`docs/04`） | Mega Gate 前向（`docs/05`）；**反向为本篇推导** |

**一句话**：两套路由的"权重归一化"都是把 $k$ 个槽拧在一起的 Jacobian（结构相同），差别在
**打分函数**（softmax 还要额外过 softmax Jacobian、且未选中的 $p$ 也参与；sqrt-softplus 只在选中的
槽上过一条分段链）和 **排名与出权重是否同源**（softmax 同源；Mega Gate 排名带 bias、权重不带）。

---

## 5. 边界

- **Mega Gate 是 SM100 专属、且只有前向**（`mega_gate.hpp:173-174` 派发 `sm100_bf16_mega_gate`、
  `:99` 断言 `scoring_func == "sqrtsoftplus"`）。本篇的"反向"是**按它的语义推导**的——DeepGEMM 本身
  不提供 gate 反向，训练侧若要复现必须自己实现这条链。
- 本篇验证的数值语义（打分函数、排名、归一化、$\operatorname{ssp}'$）**架构无关**，故 H100 上可跑；
  主 kernel 的 UMMA/tcgen05/TMEM/split-K 排序网络**仅分析**（第 5/13 篇已拆）。
- split-K 时前向是"先累加 partial logits、再过 $\operatorname{ssp}$"（`impls:284-286` 的
  `kScoreOnStore`）；反向的 $z$ 应取**累加后**的 logit，本篇的 device 复刻即按此（`z` = 完整 dot）。
- 目标卡代号敏感，一律写「目标卡」。

---

### 源码坐标（`~/github/DeepGEMM@057ca59`，行号已核对）

- 打分函数：`deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh:19-24`（标量）、`:26-39`（向量）；
  阈值 20 在 `:21`/`:38`。
- 排名用 `score+bias`：`deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh:397-398`；
  无 bias 单调等价：`:420-423`；split-K 的 `kScoreOnStore`：`:284-286`。
- tie-break（最小专家号）：`epilogue/sm100_mega_gate.cuh:80-86`、`:88-95`；top-k 游标归并 `:107-152`。
- 权重归一化：`epilogue/sm100_mega_gate.cuh:210-213`（`+1e-20f` 在 `:212`）；共享槽权重 1.0、固定索引 `:194-195`。
- host 契约：`csrc/apis/mega_gate.hpp:56-57`（排名带 bias、权重不带）、`:99`（`"sqrtsoftplus"`）、`:173-174`（SM100 派发）。

### 实验

- 驱动：`harness/exp_mega_gate_backward.sh` → `src/mega_gate_backward.cu` + `harness/exp_mega_gate_backward.py`；
  原始输出 `harness/exp_mega_gate_backward.out.txt`。
- 约束：Mega Gate 主 kernel `sm100_bf16_mega_gate.cuh` 本机不可跑（`mega_gate.hpp:173-174` 派发 SM100）；
  故主 kernel 仅分析，打分/排名/权重/反向的数值语义用裸 CUDA + torch 复刻。

### 线上资料

- DeepGEMM 官方仓库：https://github.com/deepseek-ai/DeepGEMM （Mega MoE / Mega Gate API 见 `csrc/apis/`）。
- DeepSeek-V3 技术报告（MoE 路由与 noaux_tc 打分）：https://arxiv.org/abs/2412.19437 。
- Mega MoE 融合 kernel 的官方 PR：https://github.com/deepseek-ai/DeepGEMM/pull/316 （Mega Gate 的实现随公版发布一并进入仓库）。
