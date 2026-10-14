# KDA 反向：逐 token adjoint 推导与参考实现

> 本文是「Kimi Delta Attention（KDA）前向 / 反向算子实现解析」的第 5 篇。前四篇把前向的数学
> （`01`）、K1/K2 kernel（`02`/`03`）、数值对拍（`04`）讲完。从本篇起进入**反向**：先给出
> 逐 token gated delta rule 的 **adjoint（伴随/反向递推）** 完整推导，再用纯 PyTorch 写一个
> **不依赖 autograd** 的显式反向参考 `src/kda_bwd_ref.py`，先与 `torch.autograd` 和中心差分
> 三方对齐（P5-1 / P5-2），再与 FLA `chunk_kda` 的 chunked 反向在真实 bf16 规模上对拍（P5-3）。
>
> 说明：FlashKDA 目前只开源**前向**（`csrc/smxx/fwd_kernel*.cuh`），仓库内没有 CUDA 反向；
> 因此反向的「真值」以逐 token 数学 + autograd 为准。§7 给出**分块化反向（chunked backward）**
> 的完整推导与参考实现（P5-4），kernel 设计与 CUDA 实现留到 P6。

## 1. 记号与前向回顾

沿用 `01` 篇的记号，单头、状态 $S\in\mathbb{R}^{K\times V}$（布局 $[K,V]$）。前向逐 token
（对应 `src/kda_ref.py::kda_token_recurrent`，`kda_ref.py:66–72`）：

$$
\begin{aligned}
\alpha_t &= 2^{\,g_t}\ \in\mathbb{R}^{K}\quad(\text{逐 key 通道衰减}),\\
p_t &= (\alpha_t\odot k_t)\,S_{t-1}\ \in\mathbb{R}^{V},\\
\delta_t &= \beta_t\,(v_t-p_t)\ \in\mathbb{R}^{V},\\
S_t &= \operatorname{diag}(\alpha_t)\,S_{t-1} + k_t\,\delta_t^{\top},\\
o_t &= \text{scale}\cdot q_t\,S_t\ \in\mathbb{R}^{V}.
\end{aligned}
\tag{5.1}
$$

其中 $S_0 = h_0$（初态，可为 0），$\beta_t\in(0,1)$ 已做 sigmoid，$g_t\le 0$ 是 **log2** 衰减。
需要求的梯度有六个：$dq,dk,dv,d\beta,dg$ 以及初态 $dh_0$。

> 布局提示：这里 $S$ 是 $[K,V]$。FlashKDA kernel 内部的状态缓冲是 $S^\top$（见 `03` 篇
> §8、`04` 篇布局坑），反向推导只关心数学，布局差异在实现里统一。

## 2. 反向递推（adjoint）推导

设标量损失 $L$，记输出上游梯度 $\bar o_t := \partial L/\partial o_t$，终态上游梯度
$\bar S_T^{\text{out}} := \partial L/\partial S_T$（来自 `final_state` 那条支路，若不需要则为 0）。
反向从 $t=T-1$ 扫到 $0$，维护状态伴随 $\bar S_t := \partial L/\partial S_t$（含来自 $t+1$ 的
递推支路与 $o_t$ 的直接支路）。

**(a) 状态伴随的合成。** $S_t$ 出现在两处：输出 $o_t$ 与下一时刻递推 $S_{t+1}$。因此在处理
token $t$ 之前，先把两条支路合起来：

$$
\bar S_t^{\text{full}}
= \underbrace{\bar S_t^{\text{rec}}}_{\text{来自 }t+1\text{ 的递推}}
+\ \text{scale}\cdot q_t^{\top}\bar o_t ,
\tag{5.2}
$$

其中 $q_t^\top\bar o_t$ 是外积 $[K,V]$。初始 $\bar S_T^{\text{rec}}=\bar S_T^{\text{out}}$。

**(b) 输出支路：**

$$
\bar q_t = \text{scale}\cdot S_t\,\bar o_t\ \in\mathbb{R}^{K}.
\tag{5.3}
$$

**(c) 状态更新支路。** 对 $S_t=\operatorname{diag}(\alpha_t)S_{t-1}+k_t\delta_t^\top$ 求伴随
（记 $G:=\bar S_t^{\text{full}}$）：

$$
\boxed{
\begin{aligned}
\bar S_{t-1} &\mathrel{+}= \operatorname{diag}(\alpha_t)\,G = \alpha_t\odot_{\text{row}} G,\\
\bar k_t &\mathrel{+}= G\,\delta_t \ \ (\in\mathbb{R}^{K}),\\
\bar\delta_t &= G^{\top}k_t \ \ (\in\mathbb{R}^{V}),\\
\bar\alpha_t &\mathrel{+}= \sum_{v} G_{:v}\odot (S_{t-1})_{:v}\ \ (\text{逐行点积}).
\end{aligned}}
\tag{5.4}
$$

（$\alpha_t$ 出现在 $\operatorname{diag}(\alpha_t)S_{t-1}$ 里，逐 key 行缩放；其伴随是
$\sum_v G_{k,v}(S_{t-1})_{k,v}$。）

**(d) delta 支路。** 对 $\delta_t=\beta_t(v_t-p_t)$：

$$
\bar\beta_t = \big\langle \bar\delta_t,\ v_t-p_t\big\rangle,\qquad
\bar v_t = \beta_t\,\bar\delta_t,\qquad
\bar p_t = -\,\beta_t\,\bar\delta_t.
\tag{5.5}
$$

**(e) pred 支路。** 对 $p_t=(\alpha_t\odot k_t)S_{t-1}$：

$$
\bar S_{t-1} \mathrel{+}= (\alpha_t\odot k_t)^{\top}\bar p_t,\qquad
\overline{(\alpha_t\odot k_t)} = \bar p_t\,S_{t-1}^{\top}\ (\in\mathbb{R}^{K}),
\tag{5.6}
$$

再由逐元素乘积拆到 $k_t$ 与 $\alpha_t$：

$$
\bar k_t \mathrel{+}= \alpha_t\odot \overline{(\alpha_t\odot k_t)},\qquad
\bar\alpha_t \mathrel{+}= k_t\odot \overline{(\alpha_t\odot k_t)}.
\tag{5.7}
$$

**(f) 衰减链。** $\alpha_t = 2^{g_t}$，且 $g$ 逐 token 独立（不跨时间耦合），所以

$$
\bar g_t = \bar\alpha_t\odot (\ln 2)\cdot\alpha_t .
\tag{5.8}
$$

扫完后 $\bar S_{-1}$ 即 $dh_0$。整段反向在 $t$ 上是**串行**的（因为状态伴随要向前传），
每步只有矩阵-向量级别的运算。

## 3. 从 decay 到原始门 logits（gate 链）

上面的 $\bar g$ 是「log2 衰减」层的梯度。真实模型里 $g$ 由门激活函数产生
（`kda_ref.py::kda_gate`，`kda_ref.py:32–42`），设 $C=\text{lower\_bound}\cdot\log_2 e$、
$a_h=\exp(A\_log_h)$（逐 head）、$u_t = a_h\,(g^{\text{logit}}_t + \text{dt\_bias})$：

$$
g_t = C\cdot\sigma(u_t),\qquad \sigma(u)=\frac{1}{1+e^{-u}}.
\tag{5.9}
$$

把 $\bar g$ 继续往前穿（记 $\bar u = \bar g\cdot C\,\sigma(u)(1-\sigma(u))$）：

$$
\bar g^{\text{logit}}_t = \bar u_t\cdot a_h,\qquad
\overline{\text{dt\_bias}}_h = \sum_t \bar u_t\,a_h,\qquad
\overline{A\_log}_h = \sum_{t,d} \bar u_{t,d}\,(g^{\text{logit}}_{t,d}+\text{dt\_bias}_{h,d})\cdot a_h.
\tag{5.10}
$$

这一层是逐元素 + 按 head 归约，和 kernel 里门激活（`02` 篇 §5.2 第 4 步、`fwd_kernel1.cuh`
门激活段）一一对应。`src/kda_bwd_ref.py::kda_gate_backward` 实现了它。

## 4. 参考实现（P5-1/P5-2 交付）

`src/kda_bwd_ref.py` 给出两个函数：

- **`kda_token_backward(q,k,v,g,beta,scale,dout,h0=None,dfinal_state=None)`**：先重跑前向缓存
  $S_{t-1},S_t,p_t,\delta_t,\alpha_t$，再按 §2 的 (a)–(f) 逐 token 反向，返回
  `{dq,dk,dv,dg,dbeta,dh0}`。核心循环在 `kda_bwd_ref.py:75–106`，与式 (5.2)–(5.8) 逐行对应：
  - `kda_bwd_ref.py:78`：$G=\bar S_t^{\text{rec}}+\text{scale}\,q_t^\top\bar o_t$（式 5.2）；
  - `kda_bwd_ref.py:81`：$\bar q_t$（式 5.3）；
  - `kda_bwd_ref.py:84–86`：$\bar S_{t-1}, \bar k_t^{\text{rec}}, \bar\delta_t$（式 5.4）；
  - `kda_bwd_ref.py:89–92`：$\bar\beta_t,\bar v_t,\bar p_t$（式 5.5）；
  - `kda_bwd_ref.py:95–98`：pred 支路的 $\bar S_{t-1},\bar k_t,\bar\alpha_t$（式 5.6–5.7）；
  - `kda_bwd_ref.py:100`：diag 支路的 $\bar\alpha_t$（式 5.4 最后一行）；
  - `kda_bwd_ref.py:104`：$\bar g_t$（式 5.8）。
- **`kda_gate_backward(dg_decay,g_logits,A_log,dt_bias,lower_bound)`**：§3 的门链，返回
  `(dg_logits, dA_log, ddt_bias)`。

全程只用 `torch` 的矩阵乘与逐元素运算，**不调用 autograd**，所以它与 `torch.autograd` 的结果
一致，验证的是「推导」本身，而不只是「算得对」。

## 5. 数值对拍（P5-2 结果）

`harness/check_bwd_ref.py` 把显式反向与 `torch.autograd`（对该前向）对拍，并做中心差分独立
核验。原始输出 `harness/check_bwd_ref.out.txt`。

### 5.1 显式 adjoint vs torch.autograd

被驱动的前向是 `kda_token_recurrent`（单头 `[T,D]`），损失 $L=\langle o,\bar o\rangle+\langle S_T,\bar S_T\rangle$。
覆盖 6 个输入梯度 $dq,dk,dv,dg,d\beta$ 与 $dh_0$，16 组 case（$T=1\to512$、$D=16\to64$、
$g$ 尺度 $\{0,1,2,4\}$、多 seed、fp64/fp32）：

| dtype | 指标 | dq | dk | dv | dg | dbeta | dh0 |
|---|---|---|---|---|---|---|---|
| fp64 | max_abs | **0** | **0** | **0** | **0** | **0** | **0** |
| fp32 | max_abs | 0 | 5.7e-6 | 3.6e-7 | 1.2e-7 | 2.9e-6 | 1.1e-8 |
| fp32 | max_rel | 0 | 2.0e-7 | 1.7e-7 | 8.6e-8 | 6.4e-7 | 1.6e-7 |

**fp64 下逐元素完全相等（max_abs=0，含全部 16 组 case）**——链式法则与 autograd 在实数域严格
一致。fp32 下差异 $\sim10^{-7}$，就是机器精度量级的舍入（两种计算顺序不同），无任何系统性偏差。
注意 `dq` 在 fp32 也为 0：$\bar q_t=\text{scale}\,S_t\bar o_t$ 与 autograd 走同一乘法顺序，
逐位相同。

### 5.2 中心差分独立核验

为排除「autograd 自己也错了」的可能，对 fp64、$T=6,D=8$ 的若干元素做中心差分
（$\epsilon=10^{-6}$）：

| 量 | 索引 | 有限差分 | 解析 adjoint | 绝对差 |
|---|---|---|---|---|
| dq | (0,1) | +0.45479900 | +0.45479900 | 1.3e-10 |
| dk | (2,3) | +0.00474793 | +0.00474793 | 6.9e-10 |
| dv | (4,0) | +0.62645724 | +0.62645724 | 9.9e-11 |
| dg | (1,2) | +0.02727861 | +0.02727861 | 2.0e-10 |
| dbeta | (3,) | −1.39270046 | −1.39270046 | 4.0e-10 |
| dh0 | (0,5) | +0.03325671 | +0.03325671 | 2.5e-11 |

数值一致到 $\sim10^{-10}$（差分本身的截断误差量级），三方（解析式、autograd、有限差分）互相印证。

### 5.3 gate 链核验

`kda_gate_backward` 对 `kda_gate` 的 autograd：

| 形状 | dg_logits max_abs/rel | dA_log max_abs/rel | ddt_bias max_abs/rel |
|---|---|---|---|
| T=8 H=2 D=16 lb=−5 | 4.8e-7 / 1.1e-7 | 1.2e-7 / 6.2e-9 | 4.8e-7 / 6.6e-8 |
| T=64 H=4 D=64 lb=−3 | 4.8e-7 / 5.6e-8 | 3.8e-6 / 6.2e-8 | 1.9e-6 / 6.2e-8 |
| T=200 H=3 D=128 lb=−5 | 4.8e-7 / 6.2e-8 | 3.1e-5 / 1.1e-7 | 3.8e-6 / 5.9e-8 |

（`kda_gate` 内部 `.float()`，autograd 也只到 fp32，故 max_abs $\sim10^{-7}$ 即机器精度。）
至此，**从 $\bar o$ 一路到原始 `g` logits / `A_log` / `dt_bias` 的完整反向链**数值闭环。

## 6. 与 FLA `chunk_kda` 反向对拍（P5-3 结果）

§5 已经把「显式 adjoint ↔ `torch.autograd`（同一前向）」对齐到逐位相等。这一步换**独立实现**
的真值：FLA 的 Triton `chunk_kda` 前向 + 它自带的 chunked 反向（`torch.autograd` 穿过该前向），
在真实的 $D=128$/多头 **bf16** 上对拍，量化「把逐 token adjoint 搬到 chunk 算法并降到 bf16」的误差。
原始输出 `harness/check_bwd_fla.out.txt`。

### 6.1 对拍设置（让两侧作用于**同一份**数值）

为把「算法差异」与「输入/门控/l2norm 的约定差异」分开，两侧吃完全相同的操作数：

- `use_qk_l2norm_in_kernel=False`：先在 fp32 里做 `l2_normalize`，再把**归一化后的 bf16**
  同时喂给 FLA 与参考实现（`kda_ref.l2_normalize`）。
- `use_gate_in_kernel=False`：直接传**自然对数衰减** `g_fla = g_log2 · ln2`（FLA 的状态衰减
  为 `exp(g_fla) = 2^{g_log2}`，与参考的 log2 衰减逐位同义）。门激活链（`g_logits/A_log/dt_bias`）
  已在 P5-2 §5.3 单独核验，这里不混入。
- `use_beta_sigmoid_in_kernel=False`：直接传 `sigmoid(beta_logits)`。
- `initial_state` 用默认布局 $[N,H,K,V]$，即参考里的 $S$（$[K,V]$）。

损失 $L=\langle o,\bar o\rangle+\langle S_T,\bar S_T\rangle$。参考实现跑 **fp64**，所以报告的是
**FLA bf16 反向**的误差而不是我们的；同时并列一个 fp32 显式反向，用来证明「fp32/fp64 参考彼此一致、
误差全部来自 FLA 的 bf16」。

### 6.2 结果

5 组 case（$T=128/512/1024/1023$、$H=2/4/8$、$g\sim N(0,1/2)$、含 $g=0$），逐 head 对拍
`dq,dk,dv,dg,dbeta,dh0`，取各 case 的 `max_rel`（vs fp64 参考）：

| case | dq | dk | dv | dg | dbeta | dh0 |
|---|---|---|---|---|---|---|
| T=128 H=2 g=N(0,1) | 3.9e-3 | 5.9e-3 | 8.2e-3 | 3.4e-3 | 5.7e-3 | 2.5e-3 |
| T=512 H=4 g=N(0,2) | 4.0e-3 | 3.9e-3 | 5.6e-3 | 6.1e-3 | 4.2e-3 | 2.4e-3 |
| T=1024 H=2 g=N(0,2) | 3.6e-3 | 4.7e-3 | 4.3e-3 | 3.9e-3 | 3.5e-3 | 1.9e-3 |
| T=1023 H=8 g=N(0,1) | 3.6e-3 | 3.8e-3 | 5.1e-3 | 6.8e-3 | 7.6e-3 | 7.2e-3 |
| T=512 g=0 | 4.1e-3 | 3.4e-3 | 3.2e-3 | 3.5e-3 | 4.8e-3 | 2.8e-3 |
| **worst** | **4.1e-3** | **5.9e-3** | **8.2e-3** | **6.8e-3** | **7.6e-3** | **7.2e-3** |

关键观察：

- **六项全部落在 bf16 底噪**（$\sim3\times10^{-3}\sim8\times10^{-3}$），与 §5、`04` 篇前向
  bf16 误差（$o$ 4e-3$\sim$7e-3、状态 5e-3$\sim$1.4e-2）同量级；**没有哪一项出现算法级偏差**
  （若 chunked backward 公式或状态伴随传递写错，会出现 10%+ 或随 $T$ 增长的漂移，实测没有）。
- 同一 case 里 **fp64 参考与 fp32 参考给出到打印精度相同的结果**（差异 $<10^{-6}$），说明对拍误差
  由 FLA 的 bf16 主导，参考实现本身的数值精度不是瓶颈。
- 前向 sanity（FLA bf16 vs 同输入 fp64 token-recurrent）$o$ `max_rel` 5e-3$\sim$8.5e-3、$S$
  3.4e-3$\sim$6e-3，和反向同量级——**反向误差 ≈ 前向误差**，符合「反向是前向的线性伴随、
  误差同阶传递」的预期，而非反向单独放大。
- $g=0$（无衰减）时六项依然 $\sim3\sim5\times10^{-3}$，没有像前向状态那样因累加而抬升——因为 $g=0$
  时衰减链 $\bar g=\bar\alpha\cdot\ln2\cdot 2^{g}$ 是个**局部**（逐 token、逐通道）缩放，不沿时间累积。

### 6.3 踩坑：log2 vs 自然对数的 $\ln 2$ 缩放

FLA 的 `g` 是**自然对数**衰减（`alpha=exp(g)`），参考是 **log2** 衰减（`alpha=2**g`），换算
`g_fla = g_log2·ln2`，于是 $\partial L/\partial g_{\log2} = (\partial L/\partial g_{\text{fla}})\cdot\ln2$。
第一次对拍**误写成除以** $\ln2$，`dg` 的 `max_rel` 立刻稳定停在 **0.52**（$\approx 1-1/\ln^2 2$
的比值差），而其余五项正常——这类「整体缩放错一个常数」的 bug 只体现在**一项**上，很容易被当成
「反向算法有问题」。改回乘 $\ln2$ 后 `dg` 立即降到 3e-3$\sim$7e-3（见 §6.2），五项其余项不变。
教训：跨实现比梯度时，先把**换元常数**（此处 $\ln2$）写进注释并单独断言一项。

## 7. 反向分块化（chunked backward）推导（P5-4）

§2 的 adjoint 是逐 token 串行的；真实训练里要与分块前向（`01` §3、`03`）配套，就需要一个
**块并行**的反向：每个 chunk 内部用矩阵运算做完，只把状态的伴随沿 chunk 从右往左传。本节把
式 (5.2)–(5.8) 的逐 token adjoint 折叠成 chunk 形式，并说明它能**复用前向已经算好的
$L,\ P=(I+L)^{-1},\ M_{qk},\ k^{dec},q^{dec},k^{inv},k^{rest},E,E^{tot}$**。

### 7.1 单个 chunk 的前向（回顾，$S$ 采 $[K,V]$ 布局）

沿用 `01` §3、`src/kda_ref.py::kda_chunked_fwd`。对块长 $C$、块起始状态 $S_{\text{in}}$，
记 $g^{\text{cum}}_i=\sum_{j\le i}g_j$、$g^{\text{tot}}=g^{\text{cum}}_{C-1}$：

$$
\begin{aligned}
E_i&=2^{\,g^{\text{cum}}_i},\quad E^{\text{tot}}=2^{\,g^{\text{tot}}},\\
k^{\text{dec}}_i&=k_i\odot E_i,\quad q^{\text{dec}}_i=q_i\odot E_i\cdot\text{scale},\quad
k^{\text{inv}}_i=k_i\odot E_i^{-1},\quad k^{\text{rest}}_i=k^{\text{inv}}_i\odot E^{\text{tot}},\\
L&=\operatorname{tril}\!\big(k^{\text{dec}}{k^{\text{inv}}}^\top,-1\big)\odot\beta,\qquad
M_{qk}=\operatorname{tril}\!\big(q^{\text{dec}}{k^{\text{inv}}}^\top\big),\qquad P=(I+L)^{-1},\\
w&=v-k^{\text{dec}}S_{\text{in}},\qquad u'=w\odot\beta,\qquad U=P\,u',\\
o&=q^{\text{dec}}S_{\text{in}}+M_{qk}U,\qquad
S_{\text{out}}={k^{\text{rest}}}^\top U+E^{\text{tot}}\odot_{\text{row}}S_{\text{in}}.
\end{aligned}
\tag{7.1}
$$

（$\odot_{\text{row}}$ 表示 $E^{\text{tot}}\in\mathbb{R}^{K}$ 逐行缩放 $S$。）

### 7.2 反向链式法则（单块）

反向输入是输出上游 $\bar o\in\mathbb{R}^{C\times V}$ 和**本块输出状态**的上游
$\bar S_{\text{out}}\in\mathbb{R}^{K\times V}$（= 下一块传回的 $S_{\text{in}}$ 伴随 + 终态支路）。
按式 (7.1) 自外向内求伴随，全部是矩阵乘 + 逐元素：

**(a) 输出支路** $o=q^{\text{dec}}S_{\text{in}}+M_{qk}U$：

$$
\bar q^{\text{dec}}=\bar o\,S_{\text{in}}^\top,\quad
\bar M_{qk}=\bar o\,U^\top,\quad
\bar U \mathrel{+}= M_{qk}^\top\bar o,\quad
\boxed{\bar S_{\text{in}}\mathrel{+}= {q^{\text{dec}}}^\top\bar o}.
\tag{7.2}
$$

**(b) 状态更新支路** $S_{\text{out}}={k^{\text{rest}}}^\top U+E^{\text{tot}}\odot_{\text{row}}S_{\text{in}}$：

$$
\bar k^{\text{rest}}=U\,\bar S_{\text{out}}^\top,\quad
\bar U\mathrel{+}=k^{\text{rest}}\,\bar S_{\text{out}},\quad
\bar E^{\text{tot}}=\sum_v \bar S_{\text{out}}\odot S_{\text{in}}\ (\text{逐行}),\quad
\boxed{\bar S_{\text{in}}\mathrel{+}=E^{\text{tot}}\odot_{\text{row}}\bar S_{\text{out}}}.
\tag{7.3}
$$

**(c) $U=P\,u'$：**

$$
\bar P=\bar U\,u'^\top,\qquad \bar u'=P^\top\bar U.
\tag{7.4}
$$

**(d) 求逆 $P=(I+L)^{-1}$：** 因 $dP=-P\,dL\,P$，得 **复用前向 $P$ 即可**（只多两次矩阵乘，无需再求逆）：

$$
\boxed{\bar L=-P^\top\bar P\,P^\top}.
\tag{7.5}
$$

**(e) 两个块内 GEMM** $L=\operatorname{tril}(A_L,-1)\odot\beta$（$A_L=k^{\text{dec}}{k^{\text{inv}}}^\top$）、
$M_{qk}=\operatorname{tril}(A_{qk})$（$A_{qk}=q^{\text{dec}}{k^{\text{inv}}}^\top$）：

$$
\bar A_L=\operatorname{tril}\!\big(\bar L\odot\beta,\,-1\big),\qquad
\bar A_{qk}=\operatorname{tril}\!\big(\bar M_{qk}\big),\qquad
\bar\beta\mathrel{+}=\operatorname{rowsum}\!\big(\bar L\odot A_L\big).
\tag{7.6}
$$

**(f) delta/pred 支路** $u'=w\odot\beta$、$w=v-k^{\text{dec}}S_{\text{in}}$：

$$
\bar w=\bar u'\odot\beta,\quad
\bar\beta\mathrel{+}=\sum_v\bar u'\odot w,\quad
\bar v=\bar w,\quad
\bar k^{\text{dec}}\mathrel{+}=-\bar w\,S_{\text{in}}^\top,\quad
\boxed{\bar S_{\text{in}}\mathrel{+}=-\,{k^{\text{dec}}}^\top\bar w}.
\tag{7.7}
$$

**(g) 四个共享操作数的 GEMM**（注意 $k^{\text{inv}}$ 出现在两个 GEMM 里，要合并）：

$$
\bar q^{\text{dec}}\mathrel{+}=\bar A_{qk}k^{\text{inv}},\quad
\bar k^{\text{inv}}\mathrel{+}=\bar A_{qk}^\top q^{\text{dec}},\quad
\bar k^{\text{dec}}\mathrel{+}=\bar A_L k^{\text{inv}},\quad
\bar k^{\text{inv}}\mathrel{+}=\bar A_L^\top k^{\text{dec}}.
\tag{7.8}
$$

**(h) 衰减三兄弟** $k^{\text{rest}}=k^{\text{inv}}\odot E^{\text{tot}}$、$k^{\text{dec}}=k\odot E$、
$q^{\text{dec}}=q\odot E\cdot\text{scale}$、$k^{\text{inv}}=k\odot E^{-1}$：

$$
\bar k^{\text{inv}}\mathrel{+}=\bar k^{\text{rest}}\odot E^{\text{tot}},\qquad
\bar E^{\text{tot}}\mathrel{+}=\sum_i\bar k^{\text{rest}}\odot k^{\text{inv}},
\tag{7.9}
$$

$$
\bar k=\bar k^{\text{dec}}\odot E+\bar k^{\text{inv}}\odot E^{-1},\quad
\bar q=\bar q^{\text{dec}}\odot E\cdot\text{scale},\quad
\bar E=\bar k^{\text{dec}}\odot k+\bar q^{\text{dec}}\odot q\cdot\text{scale}
-\bar k^{\text{inv}}\odot k\odot E^{-2}.
\tag{7.10}
$$

**(i) 衰减与累积和** $E=2^{g^{\text{cum}}}$、$E^{\text{tot}}=2^{g^{\text{tot}}}$、
$g^{\text{cum}}_i=\sum_{j\le i}g_j$：

$$
\bar g^{\text{cum}}=\bar E\odot(\ln 2)\,E,\qquad
\bar g^{\text{tot}}=\bar E^{\text{tot}}\odot(\ln 2)\,E^{\text{tot}},
\tag{7.11}
$$

$$
\hat g^{\text{cum}}=\bar g^{\text{cum}}\ \text{且}\ \hat g^{\text{cum}}_{C-1}\mathrel{+}=\bar g^{\text{tot}},
\qquad
\boxed{\bar g_i=\sum_{j\ge i}\hat g^{\text{cum}}_j}\ (\text{反向前缀和}).
\tag{7.12}
$$

### 7.3 跨块状态伴随传递

把 $S_{\text{in}}$ 的三条直接支路（式 7.2/7.3/7.7 的方框）合成一个累加器 $\bar S_{\text{in}}$，
它就是**上一块的 $\bar S_{\text{out}}$**：

$$
\bar S_{\text{out}}^{(c-1)}=\bar S_{\text{in}}^{(c)}
=\underbrace{{q^{\text{dec}}}^\top\bar o}_{\text{输出}}
+\underbrace{E^{\text{tot}}\odot_{\text{row}}\bar S_{\text{out}}^{(c)}}_{\text{递推}}
-\underbrace{{k^{\text{dec}}}^\top\bar w}_{\text{pred}} .
\tag{7.13}
$$

最右块的初始 $\bar S_{\text{out}}^{(L-1)}=\bar S_T^{\text{out}}$（终态上游，可为零）。反向按
$c=L-1\to0$ 扫块，块**内**全部是 GEMM，块**间**只有 $D\times D$ 的状态伴随传递——这正是
chunked backward 能 block-parallel 的原因，也解释了 `03`/`05` §6 观察到的「反向误差 ≈ 前向误差」。

### 7.4 参考实现与对拍（P5-4 结果）

`src/kda_bwd_ref.py::kda_chunked_backward` 按式 (7.2)–(7.13) 逐行实现（块长默认 16），
`harness/check_bwd_chunked.py` 把它与三个独立真值对拍：

1. **`kda_token_backward`**：P5-2 已证与 autograd 逐位相等的逐 token 显式 adjoint；
2. **autograd 穿过 `kda_chunked_fwd` 本体**（含 $(I+L)^{-1}$，直接校验对分块图求导是否写对）；
3. **autograd 穿过 `kda_token_recurrent`**（不碰 $P/L$，独立对照）。

13 组 case（$T=16\to512$、$D=12\to128$、$g$ 尺度 $\{0,1,2,4\}$、chunk $\{15,16\}$、含初态与
终态上游），指标为 `max_rel`（vs 逐 token adjoint）。原始输出 `harness/check_bwd_chunked.out.txt`：

| dtype | dq | dk | dv | dg | dbeta | dh0 |
|---|---|---|---|---|---|---|
| **fp64** | 4.2e-16 | 6.4e-16 | 3.9e-16 | 7.3e-16 | 7.5e-16 | 6.1e-16 |
| fp32 (GPU, **ieee**) | 2.5e-7 | 8.9e-7 | 5.2e-7 | 1.0e-6 | 5.0e-7 | 3.7e-7 |
| fp32 (GPU, tf32 默认) | 6.5e-4 | 8.5e-4 | 5.7e-4 | 5.3e-4 | 1.0e-3 | 4.8e-4 |
| bf16 (GPU) | 1.8e-2 | 6.8e-2 | 2.4e-2 | 4.4e-2 | 2.3e-2 | 1.5e-2 |

> **口径（P7-114 / `07` §118）**：`harness/check_bwd_chunked.py` 自本轮起在模块顶部显式
> `allow_tf32=False`，故 `check_bwd_chunked.out.txt` 现报的是 **ieee fp32** 行（$\sim10^{-7}$）。
> 上表「tf32 默认」行是修复前那份 artifact 的读数（本 NGC 容器默认 `allow_tf32=True`，把纯 torch
> 参考的 fp32 `@` 全走 tf32）；tf32-ON / OFF 两档的完整审计见 `harness/check_tf32_refs.py`。

关键结论：

- **fp64 下 13 组 case 全部到机器精度**（`max_rel` $\sim10^{-15}$，含 $D=128,T=512$ 的真实规模）。
  因为分块公式与逐 token 递推在实数域严格等价（`01` §6），其 adjoint 也必然等价；这确认
  「把逐 token adjoint 折叠成分块形式」没有丢项、没有符号错误。
- fp32 的 $\sim10^{-3}$ 不是推导误差。~~而是**参考实现里块内求逆走的是 fp32 `torch.linalg.solve`**：
  同样输入下 `kda_chunked_fwd` 在 **GPU 上** 与逐 token 前向就差 $\sim3\times10^{-4}$（CPU 上仅
  $\sim2\times10^{-7}$，见 `harness/check_bwd_chunked.out.txt` 附注），梯度再放大到 $\sim10^{-3}$。~~
  **——经 `07` §117 / P7-113 更正**：那 $\sim3\times10^{-4}$ 与本表「tf32 默认」行的 $\sim10^{-3}$ **不是**求逆
  （本容器 NGC torch 默认 `allow_tf32=True`，分块前向的 fp32 matmul 全走 tf32）；关 tf32 后
  `kda_chunked_fwd` vs 逐 token 只剩 $\sim10^{-7}$、`kda_chunked_backward` vs 逐 token 只剩 $\sim10^{-6}$
  （`harness/check_fwd_gpu_cpu.py`/`.out.txt`）。cuSOLVER fp32 的 $P$ 在 CPU/GPU **逐位相同**（$\sim10^{-8}$）。
  同一 case 里「本实现 vs autograd(chunked)」只有 $\sim5\times10^{-4}$（两者都吃同一个 fp32 $P$）。~~**待验证**：真正的 kernel 用 8×8 前代 + HMMA 合并
  （`02` §8）在 bf16 下是否比 cuSOLVER fp32 更稳，留到 P6。~~——**已解决**（`07` §116 / P7-112）：
  在真实 KDA 块矩阵上，cuSOLVER fp32 的 $P$ 就到 $\sim10^{-8}$（且 GPU==CPU），kernel 路径
  （8×8 前代 + bf16 HMMA 合并）$\sim10^{-4}\!\sim\!10^{-3}$——**不比 cuSOLVER 更准**，但**稳定**
  （bf16 量化地板、非失稳），且隔离到 fp64 反向脚手架后对六项梯度的贡献 **$\le10^{-4}$**（远低于
  bf16 操作数底噪），故「真 kernel 会不会拖坏反向精度」＝**不会**，无须增存 fp32 $P$（与 `06` §6.11
  P6-9 一致）。
- bf16 下 $\sim10^{-2}\sim3\times10^{-1}$，与 `04`/`05` §6 的 bf16 底噪一致。

### 7.5 两个容易写错的点（踩坑）

- **$S_{\text{in}}$ 的伴随不能被 $E^{\text{tot}}$ 缩放。** $S_{\text{in}}$ 出现在三处：输出
  $q^{\text{dec}}S_{\text{in}}$、状态更新 $E^{\text{tot}}\odot S_{\text{in}}$、pred
  $k^{\text{dec}}S_{\text{in}}$。只有**状态更新**那支带 $E^{\text{tot}}$。第一版把输出支路的
  ${q^{\text{dec}}}^\top\bar o$ 也并进了「状态伴随」再整体乘 $E^{\text{tot}}$，结果 `dq` 逐位正确
  （$dq$ 不依赖状态伴随）而 `dk/dv/dg/dbeta/dh0` 全错、`max_rel≈0.3`——这种「一半对一半错」的
  指纹正好指向**状态伴随的错误复合**。修法是把 $\bar S_{\text{in}}$ 独立累加（式 7.13），
  不要与 $\bar S_{\text{out}}$ 混在一个变量里。
- **式 (7.5) 的转置顺序。** $\bar L=-P^\top\bar P\,P^\top$（不是 $-\bar P^\top P\,\bar P^\top$ 之类）。
  可以用「$\bar L$ 与 $L$ 同形状、$P$ 非对称时顺序敏感」单测一项来钉死。

## 8. 小结与下一步

- 推导层面：式 (5.2)–(5.8) 给出逐 token gated delta rule 的完整 adjoint，六个梯度全解析；
  式 (5.9)–(5.10) 补齐门激活链；式 (7.2)–(7.13) 把它折叠成 chunked backward，只复用前向的
  $L,P,M_{qk}$ 与衰减因子，状态伴随按块反传。
- 实现层面：`src/kda_bwd_ref.py` 不依赖 autograd，逐 token 版 fp64 与 `torch.autograd` **逐位相等**，
  中心差分独立确认到 $\sim10^{-10}$；分块版（`kda_chunked_backward`）fp64 与逐 token 版
  **机器精度一致**（$max\_rel\sim10^{-15}$，13 组 case 含 $D=128$）。
- 跨实现层面（P5-3）：与 FLA `chunk_kda` 的 chunked 反向（bf16、$D=128$、多头）六项梯度全部
  落在 bf16 底噪 $3\times10^{-3}\sim8\times10^{-3}$，与前向误差同阶，确认推导在真实规模下成立。
- 反向的结构特征：**块内并行（全 GEMM）、块间串行（只传 $D\times D$ 状态伴随）**——这正是
  chunked backward 的设计空间。

下一步：

- **P6-1**：反向 kernel 设计/解析。FlashKDA 无开源反向，基于 §7 的分块反向给出自研 kernel
  的数据流（复用 K1 落盘的 $L/P/M_{qk}$，块内 GEMM + 块间状态伴随；或先看 FLA Triton
  `chunk_kda` 的 `chunk_bwd_d*` 结构对照）。
