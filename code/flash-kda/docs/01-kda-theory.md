# 从 delta rule 到 16-token 分块：KDA 前向的数学地图

> 这是 KDA 专题的第 1 篇。它只做一件事：把 **KDA 到底在算什么**、**分块公式是怎么一步步推出来的**、
> 以及 **FlashKDA 为什么把它切成 16 个 token 一块、又拆成 K1/K2 两颗 kernel** 讲清楚。
> 后面几篇精读 CUDA 代码时，所有符号都从这里来。

读这一篇不需要预先了解 KDA，只需要知道矩阵乘法。推导会尽量慢，每一步都交代「为什么要做这个变形」；
如果你只想先抓结论，可以看每节开头加粗的那句话，再回到推导。

**全篇只用一个状态布局**：单头、状态 $S\in\mathbb{R}^{K\times V}$（$K=V=128$）。
约定 $q_t,v_t,o_t,p_t,\delta_t$ 是**行向量**，$k_t$ 是**列向量**；这样 $q_tS_t$ 和
$k_t^\top S_t$ 都是行向量，外积直接写成 $k_tv_t$，不用在公式里到处补转置。
块级矩阵 $K_{\text{dec}},Q_{\text{dec}}$ 的每一行是相应的行向量，
$K_{\text{inv}},K_{\text{rest}}$ 的每一行是相应列向量的转置。
FlashKDA 的 CUDA 实现把状态存成转置 $S^\top\in\mathbb{R}^{V\times K}$，这只是存储习惯，
数学上和本篇互为转置（§4.7 会把这个转换写清楚）。先把一套记号用到底，可以避免后面
「同一个公式一会儿转置、一会儿不转置」的混乱。

读完这一篇，你应该能回答：

1. DeltaNet、GLA、KDA 三者的状态更新，差的到底是哪一行公式？
2. 块内那个 $16\times16$ 的 $P=(I+L)^{-1}$ 是怎么来的，它到底在解什么？
3. 为什么前向要拆成 K1、K2 两颗 kernel，而不是一颗大 kernel？

## 0. 先记住一个画面

把状态 $S$ 想成一块**可以逐格擦除的白板**。每来一个 token，做三件事：

1. **擦**：按每个通道各自的比例，把旧记忆擦淡一点（比例由 $\alpha_t=2^{g_t}$ 决定）；
2. **读**：用这个 token 的 key 去白板上读一下：「以我这条 key 的方向看，白板现在写的是什么？」
3. **补差**：把「想要的 value」减去「读到的值」，只把差额写上去（差额乘一个写入门 $\beta_t$）。

GLA 只会「擦 + 直接写 value」；DeltaNet 多了第 2、3 步的「先读再补差」；
KDA 则把两件事同时做了：**既保留 delta 的「只补差值」，又保留 GLA 的「逐通道擦除」**。

所以一句话：

> **KDA = 带「逐通道遗忘」的 delta rule。**

用一个最小的数字例子把这三种动作串起来。设 $K=V=2$，当前状态

$$
S=\begin{bmatrix}1 & 1\\ 0 & 0\end{bmatrix},
\qquad k=\begin{bmatrix}1\\0\end{bmatrix},
\qquad v=\begin{bmatrix}3 & 4\end{bmatrix},
\qquad \beta=0.5 .
$$

$S$ 的第一行对应 key 通道 1，第二行对应 key 通道 2。用 $k$ 去读：$k^\top S=(1,1)$，
也就是「白板认为这个 key 对应的 value 是 $(1,1)$」。但这次想要的 value 是 $(3,4)$，
差额是 $(2,3)$。写入门 $0.5$ 表示只补一半：

$$
S\leftarrow S+0.5\,k\,(2,3)
=\begin{bmatrix}1&1\\0&0\end{bmatrix}
+\begin{bmatrix}1&1.5\\0&0\end{bmatrix}
=\begin{bmatrix}2&2.5\\0&0\end{bmatrix}.
$$

再读一次，$k^\top S=(2,2.5)$，正好落在旧值 $(1,1)$ 和目标 $(3,4)$ 的中点。
这就是「delta」：**不重复写已经记住的部分，只补差的那一部分**。

> **类比的边界**：白板不会互相干扰，但 $S$ 是一个 $128\times128$ 的矩阵，一次更新是
> **秩一（rank-1）修正**；$k$ 的不同通道之间会通过 $S$ 耦合。更重要的是，真实 kernel 里
> $S$ 常驻 SMEM、只有 bf16 精度，「先擦后写」和「先写后擦」在实数域就不等价，在浮点下还会
> 多一层舍入差异。这些细节后文都会逐条交代。

## 1. 从线性注意力到 delta rule

### 1.1 线性注意力：把整段历史压进一个矩阵

标准的 softmax 注意力要保存每个 token 的 key/value，序列越长，KV cache 越大。线性注意力
换了一个思路：不保存历史，而是把历史**累加**进一个状态矩阵。

对单个 head，设 $q_t,v_t$ 为行向量、$k_t$ 为列向量（维度分别为 $K,V$）。因果注意力输出是

$$
o_t=\sum_{j\le t}(q_t\cdot k_j)\,v_j .
$$

把求和拆开：

$$
o_t=\sum_{j\le t}\left(q_tk_j\right)v_j
=q_t\left(\sum_{j\le t}k_jv_j\right)
=q_tS_t,
\qquad
S_t=\sum_{j\le t}k_jv_j .
$$

括号里的 $S_t\in\mathbb{R}^{K\times V}$ 就是「历史」。它满足一个非常简单的递推：

$$
S_t=S_{t-1}+k_tv_t,
\qquad o_t=q_tS_t .
$$

这就是最朴素的线性注意力：**来一个 token，就往状态上加一个秩一矩阵 $k_tv_t$**。
如果没有遗忘，$S_t$ 就是所有 $k_jv_j$ 的和。

### 1.2 纯累加的两个毛病

纯累加有两个问题，而且是同一个问题的两面。

**第一，它不会忘。** 序列走到一万步，$S_t$ 仍带着第一步写入的内容，范数随 $T$ 增长。
**第二，它不会改。** 如果同一个 key 先出现一次、后又出现一次，两次 value 不同，状态里会把
两个 value **加在一起**，而不是用新的覆盖旧的。

第二个问题可以用一行数字看清。设 $k=(1,0)^\top$，先写 $v_1=(1,0)$，再写 $v_2=(3,4)$：

$$
S=k v_1+k v_2
=\begin{bmatrix}1&0\\0&0\end{bmatrix}
+\begin{bmatrix}3&4\\0&0\end{bmatrix}
=\begin{bmatrix}4&4\\0&0\end{bmatrix}.
$$

再用 $k$ 去读，得到 $(4,4)$——既不是 $v_1$，也不是 $v_2$。如果模型想表达「这个 key 现在对应
$v_2$」，纯累加做不到。

### 1.3 delta rule：把「写」变成「先预测，再补差」

DeltaNet 的做法是：每次写入前，先看旧状态对当前 key 预测了什么，然后**只写预测误差**。

把「预测」写成行向量 $p_t=k_t^\top S_{t-1}\in\mathbb{R}^{V}$，把「误差」写成
$v_t-p_t$，更新就是

$$
S_t=S_{t-1}+\beta_t\,k_t\,(v_t-p_t),
\qquad \beta_t\in(0,1).
$$

这就是 **delta rule**（也叫 Widrow–Hoff 规则）。$\beta_t=1$ 时，它有一个很漂亮的性质。
把上式两边左乘 $k_t^\top$：

$$
k_t^\top S_t
=k_t^\top S_{t-1}+\beta_t\,k_t^\top k_t\,(v_t-p_t)
=p_t+\beta_t\|k_t\|^2\,(v_t-p_t).
$$

如果 $k_t$ 已经 L2 归一化（$\|k_t\|=1$）且 $\beta_t=1$，上式就是
$p_t+(v_t-p_t)=v_t$：**这个 key 方向上的值被精确改写成 $v_t$**。
而在与 $k_t$ 正交的方向 $u$ 上，$u^\top k_t=0$，所以 $u^\top S_t=u^\top S_{t-1}$，
旧内容原封不动。这就是「只改这个 key 管的那一维，不碰别的方向」。

$\beta_t<1$ 时，只改一部分，相当于在新旧 value 之间做插值。$\beta_t$ 由模型学出来，
所以它既可以「小心地修正」，也可以「几乎完全覆盖」。

> **为什么 $k$ 要 L2 归一化？** 不归一化时，实际步长是 $\beta_t\|k_t\|^2$，同一个 $\beta_t$
> 在不同长度的 key 上效果不同；归一化后，$k_tk_t^\top$ 才是一个真正的正交投影，
> $\beta_t$ 的含义才稳定。FlashKDA 对 $q$ 也做了归一化，主要是为了让输出尺度稳定。

delta rule 还可以从「最小化预测误差」推出来。对当前 token 定义损失

$$
\mathcal{L}_t(S)=\tfrac12\|v_t-k_t^\top S\|^2 .
$$

它对 $S$ 的梯度是 $-k_t(v_t-k_t^\top S)$（一个 $[K,V]$ 矩阵）。沿负梯度走一步、
步长取 $\beta_t$（并利用 $\|k_t\|=1$），得到

$$
S_t=S_{t-1}+\beta_t\,k_t\,(v_t-k_t^\top S_{t-1}),
$$

正是上面的 delta rule。所以它不是拍脑袋的公式，而是**对当前 token 做一步在线最小二乘**。

把括号展开，可以得到它的矩阵形式：

$$
S_t=\bigl(I-\beta_t\,k_tk_t^\top\bigr)S_{t-1}+\beta_t\,k_tv_t .
$$

这个形式后面会反复出现：**$I-\beta_tk_tk_t^\top$ 就是「擦掉这个 key 方向上的旧预测」，
$\beta_tk_tv_t$ 就是「写入新 value」**。

### 1.4 加上遗忘：先擦后写

GLA（Gated Linear Attention）在纯累加前面加了一个逐通道的衰减：每个 token 先把整个状态
按通道乘一个小于 1 的因子，再写入新的 $kv$。设逐通道衰减向量
$\alpha_t\in(0,1]^{K}$：

$$
S_t=\operatorname{diag}(\alpha_t)\,S_{t-1}+k_tv_t .
$$

把它展开，就看得出「遗忘」的累积效应：

$$
S_t=\sum_{j\le t}
\operatorname{diag}\!\left(\prod_{l=j+1}^{t}\alpha_l\right)k_jv_j
\;+\;\operatorname{diag}\!\left(\prod_{l=1}^{t}\alpha_l\right)S_0 .
$$

第 $j$ 步写入的内容，此后每一步都被再擦一次，所以到第 $t$ 步只剩
$\prod_{l=j+1}^{t}\alpha_l$ 的比例。这个「乘积形式的衰减」是后面分块推导的种子：
**块内所有衰减都可以先记成前缀积，最后一次性结算。**

KDA 把「遗忘」和「delta」合在一起。关键在顺序：**先擦，再在擦过的状态上做 delta**。

设

$$
S'=\operatorname{diag}(\alpha_t)\,S_{t-1},
\qquad
p_t=k_t^\top S',
\qquad
\delta_t=\beta_t\,(v_t-p_t).
$$

注意预测 $p_t$ 用的是**擦过之后**的状态 $S'$，不是 $S_{t-1}$。然后

$$
S_t=S'+k_t\,\delta_t
=\operatorname{diag}(\alpha_t)S_{t-1}
+\beta_t\,k_t\left(v_t-k_t^\top\operatorname{diag}(\alpha_t)S_{t-1}\right).
$$

整理成矩阵形式：

$$
\boxed{\;
S_t=\bigl(I-\beta_t\,k_tk_t^\top\bigr)\operatorname{diag}(\alpha_t)\,S_{t-1}
+\beta_t\,k_tv_t
\;}
\tag{1.1}
$$

这个式子要读三遍：

- $\operatorname{diag}(\alpha_t)S_{t-1}$：先把旧状态按通道擦一遍；
- $k_t^\top\bigl(\operatorname{diag}(\alpha_t)S_{t-1}\bigr)$：在擦过的状态上做预测；
- $(I-\beta_tk_tk_t^\top)$：把预测里「已经记住的部分」扣掉，只留下差额。

> **一个必须记住的坑：顺序不能换。** 有人会把 (1.1) 写成
> $\bigl(\operatorname{diag}(\alpha_t)-\beta_tk_tk_t^\top\bigr)S_{t-1}+\beta_tk_tv_t$，
> 这等于先做 delta、再做衰减，预测用的是**没擦过**的状态，和 KDA 的定义不是同一个模型。
> 两者在 $\alpha_t=1$ 时重合，所以这个错误在 DeltaNet 上不会暴露，一开逐通道门就会错。
> 本篇后面所有推导都基于 (1.1)。

### 1.5 三种模型，差在一行

把三个模型并排写，区别一目了然（都按 $S\in\mathbb{R}^{K\times V}$ 的布局）：

| 模型 | 状态更新 | 遗忘门 $\alpha$ | 写入门 $\beta$ |
|---|---|---|---|
| **DeltaNet** | $S_t=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t$ | 没有（$\alpha_t\equiv1$） | 逐 token 标量 |
| **GLA** | $S_t=\operatorname{diag}(\alpha_t)S_{t-1}+k_tv_t$ | 逐通道（$\alpha_t=2^{g_t}$ 或 $e^{g_t}$） | 没有（等价于 $\beta_t=1$） |
| **KDA** | $S_t=(I-\beta_tk_tk_t^\top)\operatorname{diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t$ | 每个 key 通道一个 $2^{g_t}$ | 逐 token 标量 |

三点说明：

- DeltaNet 的「delta」是主角，但它没有遗忘，历史会一直累积；
- GLA 的「遗忘」是主角，但它只会直接写 $v_t$，不会扣掉旧预测；
- KDA 两者都要，而且 $\alpha$ 必须逐 token、逐 key 通道地参与「擦」；门控要逐 key 通道算 128 个数，
  还要和 delta 的秩一修正正确排序。这就是为什么后面的 kernel 里，门控要单独占一个流水阶段。

> 逐 token 的参考实现是 `src/kda_ref.py::kda_token_recurrent`（`kda_ref.py:66–72`）。
> 它和后文的 16-token 分块实现在实数域严格等价（§4.10 给出证明，§7 给出数值验证）。

## 2. 输入、输出，以及门是怎么造出来的

**数学上只认两个量：逐通道衰减 $\alpha_t$ 和写入门 $\beta_t$。** 但模型传进来的是「激活前」的
logits，门是 kernel 里现造的。这一节把接口和造门过程对齐，后面看代码时就不会把
`g`（logits）、`g_cum`（前缀和）、`E`（指数后的衰减）混在一起。

FlashKDA 的前向接口（`flash_kda/__init__.py`）收这些东西：

| 名称 | 形状 | dtype | 含义 |
|---|---|---|---|
| `q,k,v` | `[B,T,H,K]` | bf16 | query / key / value（$K=V=128$） |
| `g` | `[B,T,H,K]` | bf16 | 门控 logits（**激活前**） |
| `beta` | `[B,T,H]` | bf16 | 写入门 logits（**激活前**，kernel 内 sigmoid） |
| `A_log` | `[H]` | fp32 | 每 head 的 log 门参数 |
| `dt_bias` | `[H,K]` | fp32 | 门偏置 |
| `lower_bound` | 标量 | float | 门下界参数（FlashKDA 取 $-5$） |

门在 kernel 里分四步造出来（对应 `tests/torch_ref.py:158–169`）：

1. **L2 归一化**：对每行 $q,k$ 做 $\ell_2$ 归一化。kernel 用 warp shuffle 树归约，
   且**故意与逐元素累加同序**，保证和 torch 参考逐位一致（`l2_normalize_kernel_match`）。
2. **加偏置**：$g\leftarrow g+\mathrm{dt\_bias}$（fp32 算）。
3. **缩放 + sigmoid**：令 $a=\exp(A\_log)$，
   $$
   g\leftarrow \underbrace{\text{lower\_bound}\cdot\log_2 e}_{\text{gate\_scale}}
   \cdot\sigma\bigl(a\cdot g\bigr).
   $$
   sigmoid 用 PTX 的 `tanh.approx.f32` 实现（$\sigma(x)=\tfrac12\tanh(x/2)+\tfrac12$），
   快且精度够用。
4. **写入门**：$\beta\leftarrow\sigma(\beta)$，同样用 `tanh.approx`。

这里有两个容易看漏的细节。

**第一，为什么是 $\log_2 e$？** 因为后面要算 $2^{g}$，而 GPU 有现成的 `ex2.approx`。
把换底因子 $\log_2 e$ 提前乘进 `gate_scale`，kernel 里就只剩一次 `ex2`，省掉一次乘法。
代价是 $g$ 的数值范围被放大了 $\log_2 e\approx1.4427$ 倍：`lower_bound=-5` 时，
$g\in(-7.22,0)$，而 $\alpha=2^g\in(0,1)$。**「$g\le0$」是衰减的前提，不是巧合**：
sigmoid 恒正，`gate_scale` 为负，所以 $g$ 恒负。

**第二，$\beta$ 和 $\alpha$ 是两套独立的门。** $\alpha$ 逐 key 通道，控制「旧的记忆擦多少」；
$\beta$ 逐 token 一个标量，控制「这次补差补多狠」。两者都过 sigmoid，所以都在 $(0,1)$。
后面的公式里，$\alpha$ 永远出现在「擦」的位置，$\beta$ 永远出现在「补差」的位置，不要混。

## 3. 为什么要分块：把「串行递推」换成「矩阵乘」

### 3.1 逐 token 递推的账

逐 token 更新有一个致命问题：**完全串行**。第 $t$ 个 token 必须等第 $t-1$ 个算完，
因为 $S_t$ 依赖 $S_{t-1}$。GPU 上有一万个线程也没用，能并行的只有 $B\times H$ 个 head。

举个具体例子：$B=1$、$H=8$、$T=8192$。前向只有 **8 个 CTA** 在沿序列跑，
每个 CTA 要串 8192 步；而一张 H100 有 132 个 SM。大部分 SM 从第一秒空到最后一秒。

单步的计算量其实不大：一次 $k^\top S$（$K\times V$ 矩阵乘向量）、一次 rank-1 更新、
一次 $qS$。但每一步都有**严格的数据依赖**，延迟被串行链锁死。这就是「算得快没用，
料要等上一站」的典型场景。

### 3.2 分块的两个目标

分块（chunk）把序列切成 $C=16$ 个 token 一块，追求两件事：

1. **降低串行深度**：块间仍然串行，但串行步数从 $T$ 降到 $T/C$；
2. **把块内 16 步的代数展开成矩阵运算**：让 Tensor Core 有活干，让块内 token 并行。

第二件事是数学上的关键。块内 16 步并不是「近似」，而是可以**严格展开**成一个
$16\times16$ 的下三角线性系统；解这个系统只需要一次小规模前代/求逆。下一节就做这个展开。

### 3.3 先看依赖关系：哪些量不依赖块入口状态

在动手推公式之前，先看一张依赖图。它决定了后面为什么要拆 K1/K2。

```
q,k,g,beta ──► 归一化 / 门 / 前缀和 / 三种 key 变形 ──► K_dec, Q_dec, K_inv, K_rest
                                                      │
                                                      ├──► L = tril(K_dec K_invᵀ, -1) ⊙ β
                                                      ├──► M = tril(Q_dec K_invᵀ)
                                                      └──► P = (I+L)⁻¹        ← 都不依赖 S_in
S_in ─────────────────────────────────────────────────┐
                                                      ├──► W = V - K_dec S_in
                                                      ├──► U = P (β ⊙ W)
                                                      ├──► O = Q_dec S_in + M U
                                                      └──► S_out = diag(E_tot) S_in + K_restᵀ U
```

上半部分是**块内自足的**：只用到 $q,k,g,\beta$，不碰状态，所以不同块之间可以完全并行。
下半部分才依赖 $S_{\text{in}}$，必须沿序列串行。

这张图就是 K1/K2 拆分的全部理由：**K1 做上半部分，K2 做下半部分**。§5 会展开讲。

## 4. 分块推导：16 步怎么变成一个下三角系统

这一节是全篇的核心。目标只有一个：

> 把「块内 16 次串行更新」改写成「只依赖 $S_{\text{in}}$ 和块内已知量的矩阵公式」。

推导分六步，每一步只做一件事。先立记号。

### 4.1 记号：把一块 token 竖着摞成矩阵

设块长 $C=16$，块内 token 下标 $i,j\in\{0,\dots,C-1\}$。块入口状态记 $S_{\text{in}}$，
第 $i$ 步之后的状态记 $S_i$（所以 $S_{-1}=S_{\text{in}}$）。块内输入为

$$
q_i,k_i\in\mathbb{R}^{K},\qquad v_i\in\mathbb{R}^{V},\qquad
g_i\in\mathbb{R}^{K},\qquad \beta_i\in\mathbb{R}.
$$

把逐 token 的量竖着摞成矩阵：

| 记号 | 形状 | 含义 |
|---|---|---|
| $G_i$ | $[K]$ | 块内 log2 前缀和：$G_i=\sum_{j\le i}g_j$ |
| $E_i$ | $[K]$ | $2^{G_i}$，从块首到第 $i$ 个 token 的累计衰减 |
| $E^{\text{tot}}$ | $[K]$ | $E_{C-1}=2^{G_{C-1}}$，整块累计衰减 |
| $K_{\text{dec}},K_{\text{inv}},K_{\text{rest}}$ | $[C,K]$ | 三种 key 变形，每行一个 token |
| $Q_{\text{dec}}$ | $[C,K]$ | 带输出 scale 的 query 变形 |
| $V,W,U$ | $[C,V]$ | value、残差、实际写入量 |
| $L,M$ | $[C,C]$ | 块内下三角矩阵 |
| $P$ | $[C,C]$ | $(I+L)^{-1}$ |

逐 token 的四个变形定义如下（$\odot$ 是逐元素乘，$E_i^{-1}$ 是逐元素取倒数）：

$$
k^{\text{dec}}_i=(k_i\odot E_i)^\top,\qquad
k^{\text{inv}}_i=k_i\odot E_i^{-1},\qquad
k^{\text{rest}}_i=k^{\text{inv}}_i\odot E^{\text{tot}},\qquad
q^{\text{dec}}_i=\text{scale}\cdot(q_i\odot E_i)^\top .
\tag{4.1}
$$

先不用记为什么是这四个，后面几步它们会自己长出来。这里只需要确认一件事：
**它们都只依赖块内的 $q,k,g$，不依赖 $S_{\text{in}}$**。

这里 $k^{\text{dec}}_i$ 和 $q^{\text{dec}}_i$ 是**行向量**（要左乘 $S$），
$k^{\text{inv}}_i$ 和 $k^{\text{rest}}_i$ 是**列向量**（要做外积）。
大写矩阵 $K_{\text{dec}},Q_{\text{dec}}$ 的每一行就是对应的行向量；
$K_{\text{inv}},K_{\text{rest}}$ 的每一行是相应列向量的转置。
$U,W,O$ 的第 $i$ 行分别是 $\delta_i,w_i,o_i$。
这样 §4.8 的矩阵乘可以直接写，不用在公式里到处补转置。

### 4.2 第一步：把「衰减」从状态里提出来

从逐 token 更新 (1.1) 出发：

$$
S_i=\operatorname{diag}(\alpha_i)S_{i-1}+k_i\delta_i,
\qquad \alpha_i=2^{g_i}.
\tag{4.2}
$$

把它展开两层，找规律：

$$
\begin{aligned}
S_0&=\operatorname{diag}(\alpha_0)S_{\text{in}}+k_0\delta_0,\\
S_1&=\operatorname{diag}(\alpha_1)S_0+k_1\delta_1\\
&=\operatorname{diag}(\alpha_1\alpha_0)S_{\text{in}}
+\operatorname{diag}(\alpha_1)k_0\delta_0
+k_1\delta_1 .
\end{aligned}
$$

第 0 步写入的 $k_0\delta_0$，到第 1 步被 $\alpha_1$ 擦了一次；
第 1 步写入的还没被擦。继续展开到第 $i$ 步，得到

$$
S_i=\operatorname{diag}(E_i)S_{\text{in}}
+\sum_{j=0}^{i}\operatorname{diag}\!\left(2^{G_i-G_j}\right)k_j\delta_j .
\tag{4.3}
$$

这里 $E_i=2^{G_i}$，而 $2^{G_i-G_j}=\prod_{l=j+1}^{i}\alpha_l$，正是「第 $j$ 步写入、
到第 $i$ 步还剩多少」。可以用归纳法验证 (4.3)：$i=0$ 时两项退化为
$\operatorname{diag}(\alpha_0)S_{\text{in}}+k_0\delta_0$，成立；从 $i-1$ 到 $i$ 时，
左边乘 $\operatorname{diag}(\alpha_i)$，正好把旧项的 $2^{G_{i-1}-G_j}$ 变成
$2^{G_i-G_j}$，再加上新项。

接下来做第一个关键变形：**把公共的 $\operatorname{diag}(E_i)$ 提到求和号外面**。
利用 $\operatorname{diag}(E_i)k_j=k_j\odot E_i$ 和 $E_i/E_j=2^{G_i-G_j}$：

$$
\begin{aligned}
S_i
&=\operatorname{diag}(E_i)\left[
S_{\text{in}}+\sum_{j=0}^{i}
\left(k_j\odot E_j^{-1}\right)\delta_j
\right]\\
&=\operatorname{diag}(E_i)\left[
S_{\text{in}}+\sum_{j=0}^{i}k^{\text{inv}}_j\delta_j
\right].
\end{aligned}
\tag{4.4}
$$

这一步的直觉很重要：

> **不要把「每个时刻的衰减」分散在 16 个项里，而是统一记到块首或块末。**
> $k^{\text{inv}}_j=k_j\odot E_j^{-1}$ 就是把第 $j$ 步的 key「还原到块首的坐标系」；
> 乘上外面的 $\operatorname{diag}(E_i)$，就一次性补上了从块首到第 $i$ 步的总衰减。

这也解释了 $k^{\text{inv}}$ 的名字：它是 $E_j$ 的逆，用来抵消第 $j$ 步之前已经发生的衰减。

**换个角度，这一步就是一个变量替换。** 令

$$
\hat S_i=\operatorname{diag}(E_i^{-1})S_i
$$

表示「把状态归一化到块首」。因为 $E_i=\alpha_i\odot E_{i-1}$，代入 (4.2) 得

$$
\hat S_i
=\hat S_{i-1}+\beta_i\,k^{\text{inv}}_i
\left(v_i-k^{\text{dec}}_i\hat S_{i-1}\right).
\tag{4.4b}
$$

**衰减项完全消失了。** 在块首坐标系里，每步只剩「读一次、补一次差、写一次」；
读用 $k^{\text{dec}}_i=k_i\odot E_i$，写用 $k^{\text{inv}}_i=k_i\odot E_i^{-1}$。
真实状态 $S_i=\operatorname{diag}(E_i)\hat S_i$ 的衰减，被挪到了「进入/离开块首坐标系」的两次缩放里。
把 (4.4b) 从 $0$ 展开再乘回 $\operatorname{diag}(E_i)$，就是 (4.4)。这也是后面
$K_{\text{dec}}$、$K_{\text{inv}}$ 两个矩阵最自然的解释：它们不是凭空造的，
而是同一个 key 在「块首坐标系」里的读法与写法。

### 4.3 第二步：预测只依赖块首状态和「前面的写入」

逐 token 的预测（行向量）是 $p_i=k_i^\top\operatorname{diag}(\alpha_i)S_{i-1}$。
把 (4.4) 里的 $i$ 换成 $i-1$ 代进去：

$$
\begin{aligned}
p_i
&=k_i^\top\operatorname{diag}(\alpha_i)S_{i-1}\\
&=k_i^\top\operatorname{diag}(\alpha_i)\operatorname{diag}(E_{i-1})
\left[S_{\text{in}}+\sum_{j=0}^{i-1}k^{\text{inv}}_j\delta_j\right]\\
&=(k_i\odot E_i)^\top
\left[S_{\text{in}}+\sum_{j=0}^{i-1}k^{\text{inv}}_j\delta_j\right]\\
&=k^{\text{dec}}_iS_{\text{in}}
+\sum_{j=0}^{i-1}\underbrace{\left(k^{\text{dec}}_i k^{\text{inv}}_j\right)}_{A_{ij}}\delta_j .
\end{aligned}
\tag{4.5}
$$

第二行用了 $\alpha_i\odot E_{i-1}=2^{g_i}\odot2^{G_{i-1}}=2^{G_i}=E_i$。
$A_{ij}$ 的定义是

$$
A_{ij}=k^{\text{dec}}_i k^{\text{inv}}_j=\left(k_i\odot E_i\right)^\top\left(k_j\odot E_j^{-1}\right).
\tag{4.6}
$$

它的指数也很值得看一眼：

$$
A_{ij}=(k_i^\top k_j)\,2^{G_i-G_j}
=(k_i^\top k_j)\prod_{l=j+1}^{i}\alpha_l
\qquad(j\lt i).
$$

它精确地等于「第 $i$ 个 token 的 query-key 相似度，乘上第 $j$ 到第 $i$ 步之间的衰减」。
**块内注意力的衰减，已经被吸收进 $k^{\text{dec}}$ 和 $k^{\text{inv}}$ 里了**，
所以后面算 $L$ 和 $M$ 时只需要普通的矩阵乘，不用再逐项乘衰减。

### 4.4 第三步：写成一个下三角线性系统

把 (4.5) 代回 $\delta_i=\beta_i(v_i-p_i)$：

$$
\delta_i
=\beta_i\left(v_i-k^{\text{dec}}_iS_{\text{in}}\right)
-\beta_i\sum_{j=0}^{i-1}A_{ij}\delta_j .
\tag{4.7}
$$

定义**残差** $w_i$ 和**块内系数** $L_{ij}$：

$$
w_i=v_i-k^{\text{dec}}_iS_{\text{in}},
\qquad
L_{ij}=\beta_iA_{ij}\quad(j\lt i),
\tag{4.8}
$$

并记 $U_i=\delta_i$（它就是第 $i$ 步真正写进状态的那一行）。于是 (4.7) 变成

$$
U_i+\sum_{j=0}^{i-1}L_{ij}U_j=\beta_iw_i .
\tag{4.9}
$$

把 $i=0,\dots,C-1$ 竖起来，就是一个矩阵方程：

$$
\boxed{\;(I+L)\,U=\beta\odot W\;}
\tag{4.10}
$$

其中 $W$ 的每一行是 $w_i$，$\beta\odot W$ 表示第 $i$ 行乘 $\beta_i$。
$L$ 的定义用矩阵写就是

$$
L=\operatorname{tril}\!\left(K_{\text{dec}}K_{\text{inv}}^\top,\,-1\right)\odot\beta,
\tag{4.11}
$$

`tril(..., -1)` 表示保留下三角、且不含对角线，所以 $L$ 是**严格下三角**：
$L_{ij}=0$ 当 $j\ge i$。

这一步是整个分块推导的枢纽，值得停一下：

> **块内 16 步递推，被改写成了一个单位下三角线性系统。**
> 未知量是 $U_0,\dots,U_{15}$；第 $i$ 个方程只用到 $U_0,\dots,U_{i-1}$，
> 因为因果性——第 $i$ 个 token 不可能看到后面的 token。

为什么 $I+L$ 一定可逆？因为 $L$ 严格下三角，$I+L$ 是**单位下三角矩阵**，
行列式等于 1（对角线全是 1）。代数上它不需要任何数值条件就可逆，这比「$\|L\|<1$ 才能用
Neumann 级数」强得多。FlashKDA 也不用 Neumann 级数，而是直接做**精确前代**：
从 $U_0$ 开始，一行一行往下解。

### 4.5 第四步：解出 $U$

记

$$
P=(I+L)^{-1}.
\tag{4.12}
$$

于是

$$
U=P\,(\beta\odot W).
\tag{4.13}
$$

数学上 $P$ 是一个逆矩阵；但解 $U$ 时并不需要先求逆再乘。把 (4.9) 写成前代就是

$$
U_i=\beta_iw_i-\sum_{j=0}^{i-1}L_{ij}U_j,
\qquad i=0,\dots,C-1.
\tag{4.14}
$$

从 $U_0=\beta_0w_0$ 开始，每一步只用到已经算出的 $U_j$，一共 $C$ 步。
FlashKDA 的 K1 确实把 $P$ 显式算出来并写进 workspace（供 K2 和反向复用），
但它的算法本质是解 $C$ 个右端项的前代：两个 $8\times8$ 的 fp32 前代，
再用一次 bf16 HMMA 合并 off-diagonal 块（`inv_fwd_subst_16`）。
选精确前代而不是 Neumann 级数，是因为 $\|L\|$ 接近 1 时级数会失稳，而前代没有这个问题。

> **为什么 $U$ 就是「真正写进状态的增量」？** 因为 $U_i=\delta_i=\beta_i(v_i-p_i)$，
> 正是 §1.3 里那个「补差值」。$P$ 的作用不是改变写入的含义，而是把「第 $i$ 步之前
> 所有写入对第 $i$ 步预测的影响」一次性解出来——块内前面的 token 会改变后面 token 看到的
> 状态，这个耦合就藏在 $L$ 里。

### 4.6 第五步：输出

输出是 $o_i=\text{scale}\cdot q_iS_i$。把 (4.4) 代进去：

$$
\begin{aligned}
o_i
&=\text{scale}\cdot q_i\operatorname{diag}(E_i)
\left[S_{\text{in}}+\sum_{j=0}^{i}k^{\text{inv}}_jU_j\right]\\
&=q^{\text{dec}}_iS_{\text{in}}
+\sum_{j=0}^{i}
\underbrace{\left(q^{\text{dec}}_i k^{\text{inv}}_j\right)}_{M_{ij}}U_j .
\end{aligned}
\tag{4.15}
$$

第二个等号用了 $q^{\text{dec}}_i=\text{scale}\cdot(q_i\odot E_i)^\top$（行向量）。矩阵形式是

$$
O=Q_{\text{dec}}S_{\text{in}}+MU,
\qquad
M=\operatorname{tril}\!\left(Q_{\text{dec}}K_{\text{inv}}^\top\right).
\tag{4.16}
$$

注意 $M$ 的 `tril` **包含对角线**，因为第 $i$ 个 token 能看到自己刚写进去的 $U_i$。
对角线上
$M_{ii}=q^{\text{dec}}_i k^{\text{inv}}_i=\text{scale}\cdot q_i k_i$，
衰减因子正好抵消——自己看自己，不需要衰减。

$O$ 的两项各有清楚的物理含义：

- $Q_{\text{dec}}S_{\text{in}}$：这个 token 从**块之前的历史**里读到的内容；
- $MU$：这个 token 从**块内前面的 token**（包括自己）通过 delta 传过来的内容。

### 4.7 第六步：状态更新

最后把 $i=C-1$ 的 (4.4) 写出来：

$$
\begin{aligned}
S_{\text{out}}
&=S_{C-1}\\
&=\operatorname{diag}(E^{\text{tot}})
\left[S_{\text{in}}+\sum_{j=0}^{C-1}k^{\text{inv}}_jU_j\right]\\
&=\operatorname{diag}(E^{\text{tot}})S_{\text{in}}
+\sum_{j=0}^{C-1}
\underbrace{\left(E^{\text{tot}}\odot k^{\text{inv}}_j\right)}_{k^{\text{rest}}_j}U_j .
\end{aligned}
\tag{4.17}
$$

也就是

$$
\boxed{\;
S_{\text{out}}
=\operatorname{diag}(E^{\text{tot}})\,S_{\text{in}}
+K_{\text{rest}}^\top U
\;}
\tag{4.18}
$$

其中 $K_{\text{rest}}$ 的每一行是 $(k^{\text{rest}}_j)^\top$，
$k^{\text{rest}}_j=k^{\text{inv}}_j\odot E^{\text{tot}}$。
**旧状态先整块衰减一次，再加上本块的 delta 增量**——这就是状态更新的全部。

$k^{\text{rest}}_j$ 的含义也清楚了：$k^{\text{inv}}_j$ 把第 $j$ 步的 key 还原到块首，
再乘 $E^{\text{tot}}$ 把它推到块末。所以 $k^{\text{rest}}_j=k_j\odot2^{G_{\text{tot}}-G_j}$，
正好是「第 $j$ 步写入、到块末还剩的衰减」。这样每个 token 的贡献都按它**剩余的生命周期**
加权，块末一次性结算。

> **FlashKDA 存的是 $S^\top$。** 把 (4.18) 两边转置，得到
> $S_{\text{out}}^\top=S_{\text{in}}^\top\operatorname{diag}(E^{\text{tot}})+U^\top K_{\text{rest}}$。
> 在 $[V,K]$ 布局下，$\operatorname{diag}(E^{\text{tot}})$ 作用在**列**（key 通道）上，
> 表现为逐列缩放；K2 的 `s_acc` 就是这么存的（`fwd_kernel2.cuh:82,457`，详见第 3 篇 §8.1）。
> 数学上两者没有区别，只是读写时少一次转置。

### 4.8 六步汇总：一张表对上代码

把上面的推导收拢。每块（$C=16$）的计算是：

| 步 | 公式 | 形状 | 说明 |
|---|---|---|---|
| ① 累积门 | $G_i=\sum_{j\le i}g_j,\ E_i=2^{G_i},\ E^{\text{tot}}=E_{C-1}$ | $[C,K]$ | 块内前缀和，逐 key 通道 |
| ② 四种变形 | $K_{\text{dec}}=K\odot E,\ Q_{\text{dec}}=\text{scale}\cdot Q\odot E$ | $[C,K]$ | 都只依赖块内输入 |
|  | $K_{\text{inv}}=K\odot E^{-1},\ K_{\text{rest}}=K_{\text{inv}}\odot E^{\text{tot}}$ | $[C,K]$ | 还原到块首 / 推到块末 |
| ③ 两个矩阵 | $L=\operatorname{tril}(K_{\text{dec}}K_{\text{inv}}^\top,-1)\odot\beta$ | $[C,C]$ | 严格下三角，块内 delta 系数 |
|  | $M=\operatorname{tril}(Q_{\text{dec}}K_{\text{inv}}^\top)$ | $[C,C]$ | 含对角，块内注意力 |
| ④ 解线性系统 | $P=(I+L)^{-1},\ U=P(\beta\odot W)$ | $[C,C],[C,V]$ | $W=V-K_{\text{dec}}S_{\text{in}}$ |
| ⑤ 输出 | $O=Q_{\text{dec}}S_{\text{in}}+MU$ | $[C,V]$ | 历史项 + 块内项 |
| ⑥ 状态更新 | $S_{\text{out}}=\operatorname{diag}(E^{\text{tot}})S_{\text{in}}+K_{\text{rest}}^\top U$ | $[K,V]$ | 块末一次性结算 |

把 (4.13) 和 (4.18) 再合一步，可以得到整块的**仿射形式**：

$$
S_{\text{out}}
=\underbrace{\left[\operatorname{diag}(E^{\text{tot}})
-K_{\text{rest}}^\top P\operatorname{diag}(\beta)K_{\text{dec}}\right]}_{\text{对角}+\text{秩 }C}
S_{\text{in}}
+K_{\text{rest}}^\top P\operatorname{diag}(\beta)V .
\tag{4.18b}
$$

这个式子把「块」看成一个线性映射：旧状态先被 $\operatorname{diag}(E^{\text{tot}})$ 整体衰减，
再被 $C$ 个秩一写入修正。修正项 $K_{\text{rest}}^\top P\operatorname{diag}(\beta)K_{\text{dec}}$
的秩最多是 $C$，所以**一个 $128\times128$ 的状态更新，本质上是「对角缩放 + 秩 $C$ 修正」**。
这也解释了为什么块内必须出现一个 $C\times C$ 的 $P$：$C$ 个秩一更新叠加，便宜的只有对角部分，
耦合项必然要一个 $C\times C$ 的矩阵来表示；这正是 Sherman–Morrison–Woodbury 的思路：$C$ 个秩一修正不必逐个展开，合起来解一个 $C\times C$ 系统。
$C=16$ 时是秩 16 修正，$C=64$ 时是秩 64 修正——块越大，这个「小矩阵」越大，
但相对 $K\times V$ 仍然很小。

这张表和 `tests/torch_ref.py:208–239` 的实现逐行对应；和 `src/kda_ref.py::kda_chunked_fwd`
也逐行对应。后面第 2、3 篇精读 K1/K2 时，每个 Phase 都能回到这张表找到出处。

> **推导到此完成。** 我们没有做任何近似：$L$、$M$、$P$、$U$、$O$、$S_{\text{out}}$ 全是从
> 逐 token 递推**恒等变形**出来的。§4.9 给一个手算例子，§4.10 给归纳法证明。

### 4.9 手算一遍：$C=2$ 的标量例子

公式再多，不如亲手算一遍。取 $K=V=1$（全部退化成标量）、$C=2$、$q_0=q_1=1$、
$k_0=k_1=1$、$v_0=2$、$v_1=3$、$g_0=g_1=-1$（所以 $\alpha=0.5$）、$\beta_0=\beta_1=1$、
$S_{\text{in}}=0$、$\text{scale}=1$。

**逐 token 递推：**

$$
S_0=0.5\times0+1\times(2-0)=2,
\qquad
S_1=0.5\times2+1\times(3-0.5\times2)=3.
$$

输出 $o_0=2$、$o_1=3$，终态 $S_{\text{out}}=3$。

**分块公式：**

| 量 | 值 |
|---|---|
| $G_0,G_1$ | $-1,\,-2$ |
| $E_0,E_1,E^{\text{tot}}$ | $0.5,\,0.25,\,0.25$ |
| $k^{\text{dec}}$ | $(0.5,\,0.25)$ |
| $k^{\text{inv}}$ | $(2,\,4)$ |
| $k^{\text{rest}}$ | $(0.5,\,1)$ |
| $q^{\text{dec}}$ | $(0.5,\,0.25)$ |
| $W=V-K_{\text{dec}}S_{\text{in}}$ | $(2,\,3)$ |
| $L_{10}=\beta_1k^{\text{dec}}_1k^{\text{inv}}_0$ | $1\times0.25\times2=0.5$ |
| $U_0=\beta_0W_0$ | $2$ |
| $U_1=\beta_1W_1-L_{10}U_0$ | $3-1=2$ |
| $M_{00},M_{10},M_{11}$ | $1,\,0.5,\,1$ |
| $o_0=M_{00}U_0$ | $2$ |
| $o_1=M_{10}U_0+M_{11}U_1$ | $0.5\times2+1\times2=3$ |
| $S_{\text{out}}=E^{\text{tot}}S_{\text{in}}+k^{\text{rest}}\cdot U$ | $0+0.5\times2+1\times2=3$ |

两种算法逐项对上。注意 $U_1=2$ 而不是 $3$：因为第 0 步已经写过 $2$，第 1 步的预测
被衰减到 $0.5\times2=1$，所以只需补 $3-1=2$。**delta 的「只补差」在块内也成立**，
$L_{10}$ 就是把这个耦合算进去的系数。

### 4.10 为什么这是严格等价，而不是近似

用归纳法把 §4.2–§4.7 串起来。

**命题**：对 $i=0,\dots,C-1$，逐 token 递推产生的 $S_i$ 满足

$$
S_i=\operatorname{diag}(E_i)\left[S_{\text{in}}+\sum_{j=0}^{i}k^{\text{inv}}_jU_j\right],
\qquad U_j=\delta_j .
\tag{4.19}
$$

**基例** $i=0$：$E_0=\alpha_0$，$k^{\text{inv}}_0=k_0\odot\alpha_0^{-1}$，
所以 $\operatorname{diag}(E_0)k^{\text{inv}}_0=k_0$，右边等于
$\operatorname{diag}(\alpha_0)S_{\text{in}}+k_0\delta_0=S_0$。

**归纳步**：设 (4.19) 对 $i-1$ 成立。由 (4.2)，

$$
S_i=\operatorname{diag}(\alpha_i)\operatorname{diag}(E_{i-1})
\left[S_{\text{in}}+\sum_{j=0}^{i-1}k^{\text{inv}}_jU_j\right]
+k_iU_i .
$$

前两项合并成 $\operatorname{diag}(E_i)$，因为 $\alpha_iE_{i-1}=E_i$；
括号里的 $k^{\text{inv}}_j$ 乘上 $\alpha_i$ 后变成
$k_j\odot E_i/E_j=k^{\text{inv}}_j\odot E_i$，正好是 $\operatorname{diag}(E_i)k^{\text{inv}}_j$；
新项 $k_iU_i=\operatorname{diag}(E_i)(k^{\text{inv}}_iU_i)$，因为
$E_i\odot k^{\text{inv}}_i=k_i$。于是 (4.19) 对 $i$ 也成立。

有了 (4.19)，$p_i$、$U$ 的线性系统 (4.10)、输出 (4.16)、状态更新 (4.18) 都是代数代入，
没有近似。**所以六步公式与逐 token 递推在实数域严格等价。**

那为什么数值上还会差？因为浮点加法不满足结合律：分块公式把 16 项求和换了一个顺序，
fp32/fp64 下会差几个 ulp。§7 给实测数字。

## 5. 为什么是 16，为什么拆 K1/K2

### 5.1 块长 16 是工程取舍，不是数学定理

分块公式对任何 $C$ 都成立（$C=1$ 就退化成逐 token）。选 16 是四个因素的交汇：

1. **Tensor Core 的形状**：$16\times16$ 正好是一个 `m16n16` 的 MMA 输出 tile（底层是两条 `m16n8k16` 指令），
   $L$、$M$、$P$ 都是 $16\times16$，不会出现「半块」浪费。
2. **块内额外计算随 $C^2$ 增长**：$L$、$M$、$P$ 的规模都是 $C^2$ 或 $C^3$。
   从 $C=16$ 到 $C=64$，$C^2$ 涨 16 倍，$C^3/6$ 从 683 涨到约 43691（64 倍）。
   块内每 token 的额外 FLOPs 约 $4CK+2CV$（$L$、$M$ 两个 $C^2K$ GEMM，
   加上 $P$ 的前代和两次 $C^2V$ 乘，摊到每个 token）：$C=16$ 时约 1.2 万，
   $C=64$ 时约 4.9 万。块间状态更新每块只有 $O(CKV)$，两者之比约
   $C(K+V)/(2KV)=C/128$（$K=V=128$）；$C=64$ 时块内开销已接近状态更新的一半，
   再大就不划算了。
3. **求逆/前代的成本与寄存器压力**：$C=16$ 的 $P$ 可以一个 warp 用寄存器 + 少量 smem 算完；
   $C=64$ 的前代需要更大的工作集，smem 和寄存器都会吃紧。
4. **数值范围**：`lower_bound=-5` 在 log2 单位下是 $-5\log_2 e=-7.213$。
   16 个 token 最坏累计 $G=-115.4$，于是 $E=2^{G}$ 最小约 $1.8\times10^{-35}$，
   $1/E$ 最大约 $5.5\times10^{34}$。这仍在 fp32/bf16 的 $2^{-126}\sim2^{128}$ 内，
   但只剩约 12 个 log2 单位余量；$C=18$ 就会分别跌破 $2^{-126}$、超过 $2^{128}$。
   所以 16 不只是「Tensor Core 好看」，也是 $1/E$ 不溢出的安全块长。

作为对照，FLA 的 Triton `chunk_kda` 用 `chunk_size=64`：它用更大的块换更少的串行步数，
代价是块内矩阵更大。**两种选择都合法，只是优化的目标函数不同。** FlashKDA 的 K2 把
$128\times128$ 的状态常驻 SMEM，块长再大，smem 就放不下输入/输出流水了，所以 16 是它的
一个自洽设计点。

### 5.2 一颗 kernel 为什么不行

回到 §3.3 的依赖图。K1 那半部分（门、归一化、$L/M/P$）**不依赖 $S_{\text{in}}$**，
所以不同块之间完全独立；K2 那半部分（$W/U/O/S_{\text{out}}$）才依赖状态，必须串行。

如果合成一颗 kernel，让每个 CTA 负责一个 head、沿序列逐块做「K1 的活 + K2 的活」，
会发生什么？还是 $B=1,H=8$ 的例子：只有 **8 个 CTA**。K1 的活（归一化、两个 GEMM、
$16\times16$ 求逆）并不少，但只有 8 个 CTA 在干，132 个 SM 里 124 个在看。
这就是「串行部分拖着并行部分一起挨饿」。

拆开之后：

| | K1（prepare） | K2（recurrence） |
|---|---|---|
| 并行维度 | **token 并行**：grid $=N\times H\times\text{num\_chunks}$ | **head 并行**：grid $=N\times H$ |
| 每个 CTA 干什么 | 一个块：归一化、门、$L/M$、求逆 | 一个 head 的整条序列：逐块递推 |
| 写什么 | workspace（$K_{\text{dec}},Q_{\text{dec}},K_{\text{rest}},E^{\text{tot}},P,M$） | 输出 $O$、终态 $S$ |
| 特征 | 短、无 pipeline、并行度极高 | 长、warp specialization、TMA 流水 |

同样是 $T=8192$：K2 还是 8 个 CTA，但它只做状态相关的几次 GEMM；K1 的 grid 是
$8\times512=4096$ 个 CTA，可以把所有 SM 喂满。实测拆开至少快 **15%**（见 `docs/07`），
而且两段可以各自调优。

代价是多写多读一份 workspace。每 tile 的六个字段是

$$
\underbrace{3\cdot C\cdot D\cdot2}_{K_{\text{dec}},Q_{\text{dec}},K_{\text{rest}}}
+\underbrace{D\cdot4}_{E^{\text{tot}}}
+\underbrace{2\cdot C^2\cdot2}_{P,\,M}
=12288+512+1024=13824\ \text{B}.
$$

按 $T=8192,H=8$ 算，约 4096 个 tile，写入约 56.6 MB，K2 再读一遍，合计约 113 MB。
这笔账在 `docs/03` 和 `docs/07` 里有完整的 L2/DRAM 量化；结论是：**用一点带宽换并行度，
在 FlashKDA 的 shape 上是划算的**。

## 6. 源码地图：先把每个文件钉死

后面精读时不再迷路。所有行号都在 `~/github/FlashKDA` 上核对过。

| 文件 | 职责 | 关键入口 |
|---|---|---|
| `csrc/flash_kda.cpp` | host API：校验、workspace 大小、形状规整、分派 | `get_workspace_size`（L5）、`fwd`（L28） |
| `csrc/fwd.h` | launch 模板声明（$D$ / 状态 / varlen 作编译期参数） | `launch_fwd`（L7） |
| `csrc/smxx/fwd_launch.cu` | 建 TMA descriptor、切 workspace 指针、两次 `<<<>>>` | `launch_fwd`（L7） |
| `csrc/smxx/fwd_kernel1.cuh` | **K1 prepare**：门/归一化/衰减/$L,M$ /求逆 → workspace | `_flash_kda_fwd_prepare`（L120） |
| `csrc/smxx/fwd_kernel2.cuh` | **K2 recurrence**：逐块递推 + 输出 + 状态更新 | `_flash_kda_fwd_recurrence`（L133） |
| `csrc/smxx/utils.cuh` | PTX 近似函数、workspace 尺寸、warp 角色、pipeline、求逆例程 | 见下 |

三层是分开的：`flash_kda.cpp` 只做 tensor 检查和 reshape，不碰 cute/TMA；descriptor 全在
`fwd_launch.cu` 里建；kernel 模板只收 descriptor 常量。

**K1 的九段流水**（`fwd_kernel1.cuh:120`，单次 TMA load、无 pipeline）：

1. 定位 tile（L168–199）：非 varlen 直接整除；varlen 用 `tile_prefix` 二分。
2. TMA 载入（L200–263）：`threadIdx.x==0` 一次发 5 个 copy；beta 做 `& ~7` 对齐。
3. 归一化（L265–303）：每行 16 线程、每线程 8 元素，4 步 `__shfl_xor_sync` 树归约。
4. 门激活 + cumsum（L306–338）：128 线程按列并行，天然串行，不需要跨行同步。
5. decay_apply（L353–476）：算出四种 q/k 变体，用 union 复用 smem。
6. 两个 GEMM（L478–487）：warp 0 算 $L$（fp32 累加），warp 1 算 $M$（bf16）。
7. tril 掩码 + 乘 $\beta$（L489–508）。
8. 求逆（L510–511）：`inv_fwd_subst_fused_1warp`（`utils.cuh:213–369`）。
9. 写 workspace（L513–585）：6 个 TMA store，`tma_store_wait<0>`。

**K2 的角色分工**（`fwd_kernel2.cuh:133`，一个 CTA 串完整个序列）：

- 4 个 MMA warp + 1 个 LOAD warp + 1 个 STORE warp，共 192 线程（`fwd_launch.cu:186`）。
- load 用 3-stage TMA pipeline，store 用 2-stage async pipeline。
- 六个 Phase：双 GEMM → cast/载入 $v$ → $u=(v-u)\beta$ 与 $Pu$ → $MU+\text{out}$ →
  写回 smem → 状态更新。
- `MOVM_T` 让 MMA 的输出片段直接当下一步的 B 操作数，省掉一次 smem 往返。

**workspace 布局**（`flash_kda.cpp:5` 与 `fwd_launch.cu:62–71` 必须一致）：

$$
\text{per\_tile}=
\underbrace{3\cdot C\cdot D\cdot2}_{K_{\text{dec}},Q_{\text{dec}},K_{\text{rest}}}
+\underbrace{D\cdot4}_{E^{\text{tot}}}
+\underbrace{2\cdot C^2\cdot2}_{P,\,M}.
$$

六段按 `kKDecayed → kQDecayed → kKRestored → kGTotal → kINV → Mqk` **分开排**（SoA，不是结构体数组），
最后留 varlen 的 `tile_prefix`。这样每个字段能独立配 TMA descriptor，且都满足 128B 对齐
（L16–18 的 `static_assert`）。

## 7. 怎么验证推导没推错

公式推完了，但「我觉得对」不算数。`src/kda_ref.py` 里有**两份独立实现**：

- `kda_token_recurrent`：逐 token 递推（§1 的公式），当 golden；
- `kda_chunked_fwd`：16-token 分块（§4 的六步）。

两者计算顺序完全不同。如果分块代数推错了，它们不可能对上。实测结果：

| 对照 | 口径 | 结果 |
|---|---|---|
| 分块 vs 逐 token | fp64 原始差（不落 fp32） | $\le2\times10^{-16}$，几个 ulp |
| 分块 vs 逐 token | 落回 fp32 后 `max_abs` | $0$（$T=1$）到 $7.2\times10^{-7}$ |
| FLA `chunk_kda` vs 逐 token | bf16，$o$ 的 `max_rel` | $3.7\times10^{-3}\sim4.9\times10^{-3}$ |
| FLA `chunk_kda` vs 逐 token | bf16，终态 $S$ 的 `max_rel` | $3.1\times10^{-3}\sim5.9\times10^{-3}$ |

关于第一行，有一个**口径**必须说清楚：`kda_ref.py` 自测的 **fp64 段**把结果
用 `.float()` 写回 fp32 张量再比较，所以终端打印 `max_abs=0.000e+00`
（`harness/kda_ref_selftest.out.txt` 开头；同一文件的 fp32 段打印的是
$10^{-8}\sim7\times10^{-7}$）。这表示「差异低于 fp32 分辨率」，不表示 fp64 下逐位相同。
真正用 fp64 直接比时，差约 $10^{-16}$，也就是几个 ulp——这正是
「实数域严格等价、浮点下只差舍入顺序」的量化版。
§4.10 的归纳法证明的是实数域的恒等，数值对拍验证的是实现没有把公式抄错，两者互补。

第二、三行说明：分块与逐 token 的差异只来自 fp32 舍入；而和 FLA 的 bf16 差异
（$3.7\times10^{-3}\sim5.9\times10^{-3}$）与 bf16 输入本身的量化误差
（$2^{-8}\approx3.9\times10^{-3}$）同量级。**没有看到「算法级」的误差。**

### 一个必须知道的坑：状态是 $S^\top$，不是 $S$

FlashKDA 的 `initial_state` / `final_state` 用的是 **$S^\top$ 布局**（`[N,H,V,K]`，
`torch_ref.py` 里 `state_slice` 存的就是 $S^\top$），和 K2 内部的 `s_acc`（$[V,K]$）一致。

FLA 的 `chunk_kda` 则默认返回 $[N,H,K,V]$。两者互为转置：直接比 `max_abs=0.511`，
转置后再比只剩 `0.00398`（`harness/probe_state_layout.py`）。

还有一个更隐蔽的坑：FLA 的 `transpose_state_layout=True`（新版叫 `state_v_first`）不只是改
**返回**布局，还会把传入的 `initial_state` 也按 V-first 重新解释。最初误用该开关后，
除 $t=0$（唯一直接吃初态的 token）外一切正常，$t=0$ 的误差却高达 $0.1$——看起来像
「随机小误差」，实际是布局错了。默认 `False` 时两边布局一致，对拍正常。

> 为什么布局错误只让 $t=0$ 误差大？因为 $t=0$ 的输出直接读初态，布局一转置就全错；
> 后面的 token 会不断用新的 $kv$ 覆盖旧状态，错误被「冲淡」了。这个现象反过来
> 可以当判据：**如果只有序列开头误差大，先查初态布局，不要先怀疑 kernel 的数学。**

## 8. 三句话小结

- KDA 的更新是「**先按通道擦、再在擦过的状态上做 delta**」，即
  $S_t=(I-\beta_tk_tk_t^\top)\operatorname{diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t$。
  顺序不能换，换成 $(\operatorname{diag}(\alpha_t)-\beta_tk_tk_t^\top)S_{t-1}$ 就是另一个模型。
- 分块把块内 16 步展开成「四种 q/k 变形 + 两个 $16\times16$ 矩阵 + 一个单位下三角系统
  $(I+L)U=\beta\odot W$ + 一次状态更新」。$P=(I+L)^{-1}$ 不是近似，精确前代即可解；
  因为 $L$ 严格下三角，$I+L$ 永远可逆。
- 前向拆 K1（token 并行 prepare）与 K2（head 并行 recurrence），是因为 K1 那半部分不依赖
  块入口状态、可以吃满所有 SM；K2 只做必须串行的状态相关计算。16 是 Tensor Core 形状、
  块内 $C^2$ 开销和求逆成本之间的工程取舍，不是数学要求。

如果三个开场问题你都能自己答出来，这一篇就可以翻过去了：

1. **三种模型差在哪一行？** DeltaNet 没有 $\alpha$，GLA 没有 $\beta$，KDA 两者都有，
   且 $\alpha$ 是逐通道的；KDA 的矩阵形式是 $(I-\beta kk^\top)\operatorname{diag}(\alpha)$。
2. **$16\times16$ 的求逆在干什么？** 在解块内 delta 的耦合：第 $i$ 个 token 的写入量
   $U_i$ 依赖前面所有 $U_j$，$L_{ij}$ 就是「第 $j$ 步对第 $i$ 步预测的影响」。
3. **为什么拆 K1/K2？** 因为块内 prepare 与状态无关、可以 token 并行；状态递推必须串行。
   拆开后并行部分不再被串行部分拖住。

下一篇进入 K1 的代码：看这六步公式如何在一个 CTA 里落地，尤其是那个 $16\times16$ 的求逆
到底是怎么用一个 warp 写出来的。
