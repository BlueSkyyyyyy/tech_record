# MoE 路由的负载均衡：aux-loss、router z-loss 与 aux-loss-free bias（含反向）

> 本专题到第 138 篇为止，已经把 DeepSeek Mega MoE 的前向（第 1~11 篇）、反向（第 4/12/13/14 篇）、
> Mega Gate / Mega mHC（第 5/13/130/131/132/135 篇）和三条门控反向（第 4 篇 router、第 127 篇 Mega Gate、
> 第 130 篇 mHC）逐个拆开。但有一个**训练侧、直接决定 MoE 能不能收敛**的算子一直没有正面讲过：
> **路由的负载均衡（load balancing）**。
>
> 为什么这是缺口：第 4 篇推导 router 反向时、第 127 篇推导 Mega Gate 反向时，那个 `bias` 都被当作
> **给定常量**（stop-grad）；而 DeepGEMM 的 Mega Gate 源码里确实只消费 `bias`、不产生它
> （`csrc/apis/mega_gate.hpp:41-43`）。也就是说，模块内「加 bias 排名、不带 bias 出权重」的**用法**已经
> 在第 5/13 篇拆透了（并在 H100 上逐位验证过），但**「这个 bias 从哪来、怎么更新」、「传统 aux-loss /
> z-loss 的算子与梯度长什么样」**从没被写过——而这正是 DeepSeek-V2 的 aux-loss 与 DeepSeek-V3 的
> **aux-loss-free** 策略的分别所在。
>
> 本篇补上这一层：把三种负载均衡机制（**aux load-balancing loss**、**router z-loss**、
> **aux-loss-free bias**）的**前向公式、梯度、以及与 Mega Gate 语义的对接**讲清，并在 H100(sm90) 上
> 用纯 PyTorch（float64）把解析梯度对拍 autograd / 中心差分，量出三种机制的**硬负载杠杆强度**。
> 全部数字来自 `harness/exp_moe_load_balance.{py,sh,out.txt}`（DeepGEMM 2.8.1，公开版 commit `057ca59` 附近）。
> 本层纯 host / Python、架构无关，H100 真机 **21 PASS / 0 FAIL**。

---

## 1. 问题：路由会「坍缩」到少数专家

MoE 的路由是「每个 token 从 $E$ 个专家里选 top-$k$」，门控概率
$p_{t,i}=\operatorname{softmax}(z_t)_i$，$z_t=W_g x_t$。若不加约束，训练早期几个专家的 logits 偶然占优，
就会被**正反馈**越选越多——被选中的专家梯度更多、变得更强、被选得更多；其余专家逐渐「饿死」。
Sinkhorn 式的负载均衡（第 5/130 篇的 Mega mHC）是**单 token 内**的归一化，管不了**跨 token / 跨 step**
的专家利用不均。

本篇实验里造一个「前 4 个专家被系统性偏好」的路由（$z$ 的这几个通道 +2.5），不加任何均衡机制时
**top-k 硬负载的 max/min 达 28.04**（§5 Part C）。这就是要解决的问题。

三种机制按「作用对象」分两类：

| 机制 | 作用对象 | 需要梯度？ | 代表 |
|---|---|---|---|
| aux load-balancing loss | **软概率 $P_i$**（经 softmax） | 是（反传到 logits） | Switch / DeepSeek-V2 |
| router z-loss | **logits 的尺度**（$\log\sum\exp$） | 是 | ST-MoE |
| aux-loss-free bias | **硬排名的偏置**（阶跃选择） | **否**（符号规则） | DeepSeek-V3 |

---

## 2. 三种机制的前向与梯度

约定：$T$ 个 token、$E$ 个专家、top-$k$；$p_{t,i}$ 为门控概率；
$\mathrm{mask}_{t,i}=\mathbb{1}[i\in\operatorname{topk}(t)]$。

### 2.1 aux load-balancing loss（Switch / DeepSeek-V2）

Switch Transformer（arXiv 2101.03961 式 (4)）的辅助损失：

$$
\mathcal{L}_{\text{aux}}=\alpha\cdot E\cdot\sum_{i=1}^{E} f_i\,P_i,
\qquad
f_i=\frac{1}{T}\sum_t \mathrm{mask}_{t,i},
\qquad
P_i=\frac{1}{T}\sum_t p_{t,i}.
$$

两个因子里：
- $f_i$ 是**硬计数**（top-k 选了谁），**视为常量**（stop-grad）——它不可微；
- $P_i$ 是**平均门控概率**，可微。

**关键洞察**：梯度只穿过 $P_i$。因为 $f_i$ 被当常量，

$$
\frac{\partial \mathcal{L}_{\text{aux}}}{\partial z_{t,j}}
=\frac{\alpha E}{T}\sum_i f_i\,\frac{\partial p_{t,i}}{\partial z_{t,j}}
=\frac{\alpha E}{T}\, p_{t,j}\Big(f_j-g_t\Big),
\qquad
g_t:=\sum_i f_i\,p_{t,i},
$$

其中用到 $\partial p_{t,i}/\partial z_{t,j}=p_{t,i}(\delta_{ij}-p_{t,j})$。
直觉：$f_j>g_t$ 的专家（被选得比「概率加权平均」多）会被**压制** $z_{t,j}$，$f_j<g_t$ 的被抬升。

本机实测（§5 Part A）：解析梯度 vs `autograd` relerr **2.20e-15**、vs 中心差分 **7.97e-06**。

一个漂亮的不变量：任意 $\mathrm{mask}$ 都有 $\sum_i f_i=k$（每 token 恰选 $k$ 个），
故当 $p$ 均匀（$P_i=1/E$）时

$$
\mathcal{L}_{\text{aux}}=\alpha E\sum_i f_i\cdot\frac1E=\alpha\cdot k,
$$

**与 $f$ 的具体分布无关**——实测 $z=0$ 时 $L=\alpha k=0.04$（$\alpha=0.01,k=4$），与公式差 $<10^{-12}$。

DeepSeek-V2（arXiv 2405.04434 §3.3）用的是同一族形式（加上序列级归一化），
DeepSeek-V3 则只在**前若干步 warmup** 用它防止「序列内极端不均」。

### 2.2 router z-loss（ST-MoE）

ST-MoE（arXiv 2202.08906 §10）的 router z-loss 惩罚 logits 尺度过大，改善数值稳定性：

$$
\mathcal{L}_{z}=\beta\cdot\frac1T\sum_t\Big(\log\!\sum_i e^{z_{t,i}}\Big)^2.
$$

记 $\mathrm{lse}_t=\log\sum_i e^{z_{t,i}}$、$p_t=\operatorname{softmax}(z_t)$，
用 $\partial\,\mathrm{lse}_t/\partial z_{t,j}=p_{t,j}$：

$$
\frac{\partial \mathcal{L}_{z}}{\partial z_{t,j}}=\frac{\beta}{T}\cdot 2\,\mathrm{lse}_t\cdot p_{t,j}.
$$

本机实测解析梯度 vs `autograd` relerr **4.95e-16**、vs 中心差分 **5.33e-07**，
且把 logits 尺度放大 4 倍会把 $L_z$ 从 **2.54e-2** 抬到 **2.89e-1**（§5 Part B）。

### 2.3 aux-loss-free bias（DeepSeek-V3）

DeepSeek-V3（arXiv 2412.19437 §5.2）提出**无辅助损失**的均衡：给每个专家一个偏置 $b_i$，
**只**加在**排名**分数上、**不**加在门控权重上：

$$
s'_{i,t}=s_{i,t}+b_i \quad\text{(用于 top-k 选择)},
\qquad
w_{t,i}\propto s_{i,t}\;\text{仅对选中者}\quad\text{(权重用无偏分)}.
$$

其中 $s=f_{\text{score}}(z)$ 是打分函数（DeepSeek/Mega Gate 用 $\sqrt{\operatorname{softplus}}$）。
每个训练 step 后按**符号**更新：

$$
b_i \leftarrow b_i + \gamma\cdot\operatorname{sign}\!\big(\text{target}_i-\text{load}_i\big),
\qquad \text{target}_i=1/E,
$$

`load_i` 是专家 $i$ 本 step 被选中的占比。**这个规则不需要梯度**（$\operatorname{sign}$ 非光滑），
所以 bias 完全不进反向图——它是「控制回路」，不是「可微参数」。

在 Mega Gate 语义下复刻（§5 Part C）：起始硬负载 max/min **28.04**，跑 800 步
（$T=4096,E=32,k=4,\gamma=10^{-3}$）后 **1.0178**，各专家占比与 $1/E$ 的偏差 $<3.05\times10^{-4}$；
过载专家 bias 均值 $+0.1745$ **低于**欠载专家 $+0.3589$（方向正确）。$\gamma$ 加到 $10^{-2}$，40 步即达
max/min **1.1054**。

---

## 3. 与 Mega Gate 的对接（真实行号）

DeepGEMM 的 Mega Gate 就是上面 2.3 那套语义（但 bias 由外部传入、kernel 不更新它）：

- **接口**：`csrc/apis/mega_gate.hpp:41-43` 的 `bias [E] FP32 | None`、`image_bias [E] FP32 | None`、
  `image_token_mask [T] bool | None`；文档注释 `:56-57` 明写
  「ranks experts on **score + bias**, while the emitted weights are the **unbiased scores**」。
  `bias` 与 `image_bias`/`image_token_mask` 必须**成对**出现（`:116`）。
- **排名用带 bias 分**：`impls/sm100_bf16_mega_gate.cuh:380-383` 取 `smem_bias`（图像 token 换
  `smem_image_bias`），`:396-398` 把 `scores_local` 就地改成
  `unbiased_score + bias_local[value_idx]`（非法专家置 `-inf`）；`:404` 在这个**有偏分**上
  `select_warp_topk` 选 top-k。`kHasScoringBias = kHasBias or kHasImageTokenMask`（`:283`）。
- **权重用无偏分**：`epilogue/sm100_mega_gate.cuh:188` 的 `store_topk_token`，`:212-213`
  用 `selected_unbiased_score` 求和归一化、乘 `routed_scaling_factor`——**bias 不出现在这里**。
- **参考实现**：`tests/test_mega_gate.py:130-135` 的 `ranking = scores + ranking_bias`
  （含 `image_bias` 的 `torch.where` 选择）；`:137-139` 的 `reference_weights` 用 `scores`（无偏）
  gather 后归一化；`:141-143` 断言权重差 $<10^{-11}$。

也就是说，**Mega Gate 只实现 2.3 的「消费端」**：给定 $b$，它做「带 bias 排名 + 无偏权重」。
`b` 的产生与更新在框架的训练循环里（本篇 §2.1/§2.3 的算子），不在 DeepGEMM 内。
第 127 篇推导 Mega Gate 反向时把 $b$ 当常量正是对的——**bias 本来就该 detach**。

---

## 4. 反向 / autograd 侧的三条要点

1. **aux-loss 的梯度只走软概率**：`f_i`（top-k 计数）必须 stop-grad；若误把它当可微，梯度会多出
   一条穿过「阶跃选择」的伪路径（top-k 的次梯度几乎处处为 0，反而破坏梯度信号）。
   实测：`f` 当常量时解析梯度与 autograd 位级一致（2.2e-15）。
2. **z-loss 的梯度只走 lse 与 softmax**：$\partial L_z/\partial z_{t,j}=\frac{2\beta}{T}\mathrm{lse}_t\,p_{t,j}$，
   与 autograd 一致（4.95e-16）。
3. **bias 不参与反向**：`bias.grad is None`（§5 Part D，即使 `requires_grad=True` 也只是常量输入），
   更新是纯符号规则 $\gamma\operatorname{sign}(\cdot)$，与 SGD 无关。

**总损失**（DeepSeek-V3 训练）：

$$
\mathcal{L}=\mathcal{L}_{\text{task}}+\alpha\mathcal{L}_{\text{aux}}+\beta\mathcal{L}_z,
$$

其中 aux/z 只在 warmup 阶段生效，之后靠 $b$ 的符号更新维持均衡（$\mathcal{L}_{\text{aux}}$ 的
$\alpha$ 撤掉后 $b$ 继续跑）。

---

## 5. H100 实测

脚本 `harness/exp_moe_load_balance.py`（driver `harness/exp_moe_load_balance.sh`，
输出 `harness/exp_moe_load_balance.out.txt`），H100 sm90、`torch 2.12`、float64，**21 PASS / 0 FAIL**：

```
=== Part A: aux load-balancing loss 的梯度 ===
  [PASS] 解析梯度 vs autograd           relerr=2.20e-15
  [PASS] 解析梯度 vs 中心差分             relerr=7.97e-06
  [PASS] 均匀 P 时 L=alpha*k（与 f 分布无关）  L_eq=4.000000e-02 (alpha*k=4.000000e-02)
  [PASS] 偏斜路由 L 显著大于均衡            L_skew=3.1703e-01 vs alpha=0.01
  [PASS] f_i 之和 = k                    sum f=4.000000 (k=4)

=== Part B: router z-loss 的梯度 ===
  [PASS] 解析梯度 vs autograd           relerr=4.95e-16
  [PASS] 解析梯度 vs 中心差分             relerr=5.33e-07
  [PASS] 放大 logits 尺度抬升 z-loss       L(z)=2.5364e-02 < L(4z)=2.8852e-01

=== Part C: aux-loss-free bias 的负载收敛 ===
  [PASS] 无 bias：负载明显不均衡           max/min=28.04
  [PASS] 加 bias 更新后：负载趋衡          max/min=1.0178 (起始 28.04)
  [PASS] 负载单调改善                     step0=28.04 -> step50=19.67 -> step800=1.0178
  [PASS] 收敛后各专家占比接近 1/E          max|load-1/E|=3.05e-04 (1/E=0.0312)
  [PASS] gate 权重用无偏分（bias 只进排名）  relerr=0.00e+00
  [PASS] bias 改变排名                  变化 token 占比=1.000
  [PASS] 过载专家 bias 更低 / 欠载更高（方向正确）  over_bias=+0.1745 under_bias=+0.3589
  [PASS] 更大 gamma 收敛更快             gamma=0.01,40 步 max/min=1.1054

=== Part D: aux-loss 与 bias 的关系 ===
  [PASS] aux-loss 不含 bias             bias.grad=None
  [PASS] bias 更新 = 纯符号规则           |db| in {0.0010, 0.0010}
  [PASS] aux-loss 负梯度一步使 L 下降      L0=1.123291e-01 -> L1=1.123290e-01
  [PASS] aux-loss 梯度对硬负载是弱杠杆      max/min: 8.50 -> 5.44（2000 步, alpha=1）
  [PASS] bias 符号更新对硬负载是强杠杆      bias max/min=1.0278  vs aux-loss=5.44
```

---

## 6. 最重要的结论：两个杠杆的强度不同

Part D 量到一个**很实际**的对比，也解释了 DeepSeek-V3 为什么改用 aux-loss-free：

- **aux-loss 的梯度是「软杠杆」**。它只能通过 $P_i$（softmax 概率）间接影响路由。但硬负载 $f_i$
  是**排名的阶跃函数**：概率稍微一变，只要没跨过 top-k 阈值，$f_i$ 就完全不变。实测即使把
  $\alpha$ 放大到 1.0、跑 2000 步梯度下降，硬负载 max/min 也只从 **8.50 降到 5.44**——改善有限、
  且需要精细调 $\alpha$ 与学习率。
- **bias 的符号更新是「硬杠杆」**。它**直接改排名**，每步把过载专家的排名分压低、欠载专家抬高，
  是作用在阶跃函数**自变量**上的控制回路。同样问题，800 步把它压到 **1.03**（强杠杆）。

这正是 DeepSeek-V3「aux-loss-free」的动机：用 $b$ 的符号更新**直接、稳定**地控制**硬负载**，
把昂贵的、难调的 aux-loss 降级成只在前若干步做 warmup（防止序列内极端不均）。

---

## 7. 线上资料链接

- Switch Transformer（aux load-balancing loss 式 (4)）：<https://arxiv.org/abs/2101.03961>
- ST-MoE（router z-loss §10）：<https://arxiv.org/abs/2202.08906>
- DeepSeek-V2（aux-loss 与专家均衡，§3.3）：<https://arxiv.org/abs/2405.04434>
- DeepSeek-V3（aux-loss-free 策略与 `e_score_correction_bias`，§5.2）：<https://arxiv.org/abs/2412.19437>
- DeepGEMM（Mega Gate 源码与测试）：<https://github.com/deepseek-ai/DeepGEMM>

---

## 8. 小结

- **三种负载均衡**：aux load-balancing loss（$\alpha E\sum f_iP_i$，$f$ stop-grad）、router z-loss
  （$\beta\,\overline{\mathrm{lse}^2}$）、aux-loss-free bias（$b_i\mathrel{+}=\gamma\operatorname{sign}(\text{target}-\text{load})$）。
- **反向**：aux/z 的梯度都只走软概率（解析梯度与 autograd 位级一致，2.2e-15 / 5.0e-16）；
  **bias 完全 detach**，是零梯度的控制回路。
- **与 Mega Gate 的对接**：DeepGEMM 只做「带 bias 排名、不带 bias 出权重」的**消费端**
  （`mega_gate.hpp:56-57`、`impls:396-398`、`epilogue:212-213`、`test_mega_gate.py:130-139`），
  第 127 篇把 bias 当常量推导反向是对的。
- **H100 实测**：aux-loss 是**弱杠杆**（2000 步把硬负载 max/min 8.50→5.44）、bias 是**强杠杆**
  （800 步 28.04→1.02）——坐实 DeepSeek-V3 aux-loss-free 的工程动机。
