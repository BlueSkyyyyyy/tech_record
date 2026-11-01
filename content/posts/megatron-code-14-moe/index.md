---
title: "Megatron 源码精读（十四）：MoE 实现与优化"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "moe", "专家并行"]
categories: ["训练框架"]
weight: 14
series: ["megatron-code"]
---

上一篇[《强化学习》]({{< relref "megatron-code-13-rl" >}})讲了 RL 后训练。本篇回到模型结构本身：`megatron/core/transformer/moe/` 里，MoE 靠什么支撑「稀疏激活」——路由、搬运 token、均衡负载、专家计算，各自落在哪个符号上。

**一句话结论**：MoE 的复杂度几乎不在 GEMM，而在「搬运 + 排序」；默认配置下连 all-to-all 都不用（all-gather + reduce-scatter），aux loss 也默认根本不开。

---

## 1. 问题：稀疏激活撞上密集硬件

密集 transformer 里每个 token 过完整 FFN；MoE 把它换成 E 个专家，每个 token 只进 top-k 个（`moe_router_topk` 默认 2）。总参数量能放大到 E 倍，单 token 计算量基本不变。

代价是两件新麻烦：路由是数据依赖的，**每层、每步的 token 分布都不一样**，需要额外机制防止专家饿死或撑死；token 又散落在不同 rank 的专家上，需要跨设备搬运。

所以代码骨架是「四段流水线 + 一套路由/均衡机制」。

## 2. 机制：四段流水线，但通信与均衡各有前提

### 2.1 docstring 四步与真实方法名

`MoELayer.forward` 的 docstring 直接给了四步（`moe_layer.py:678-682`）：

```python
# 1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
# 2. Dispatch: Tokens are sent to the expert devices using communication collectives.
# 3. Expert Computation: Experts process the dispatched tokens.
# 4. Combine: The outputs from the experts are combined and returned.
```

对应的方法：`route`（`moe_layer.py:451`）、`preprocess`（`:469`）、`dispatch`（`:520`）、`routed_experts_compute`（`:593`）、`combine`（`:627`）、`postprocess`（`:636`）。第四步被拆成两个方法，共享专家的相加就在 `postprocess` 里。

还有一处顺序容易看漏：`forward` 里最先调用的是 `shared_experts_compute`（`:532`），然后才 `route`。共享专家不看路由结果，所以可以提前算，和后面的路由、通信在时间上重叠。

### 2.2 dispatch 与 combine 只是转发，默认不是 all-to-all

两段方法体都只有一行（`moe_layer.py:527-529`、`633`）：`return self.token_dispatcher.token_dispatch(...)`，combine 同理。

真正的搬运在 dispatcher 里，它有六个钩子：`dispatch_preprocess` / `token_dispatch` / `dispatch_postprocess` / `combine_preprocess` / `token_combine` / `combine_postprocess`。**只有 `token_dispatch` 和 `token_combine` 含通信**，另外四个是单卡内的 reshape 与 permute。

`moe_token_dispatcher_type` 默认是 `allgather`（`transformer_config.py:965`，可选 `allgather` / `alltoall` / `flex`）。默认路径做的是：在 TP×EP 域上 all-gather 全部 token 和 routing_map，每个 rank 只算自己的本地专家，combine 阶段用 reduce-scatter 把结果还回原 rank。

所以「MoE 必然 all-to-all」是个常见误解。all-gather 换来了实现简单、对 dropless 友好，代价是通信量随 EP 增长；要按 token 精确发送，才需要 `alltoall` 或 `flex`。

## 3. 代码落点：`TopKRouter` 与 `switch_load_balancing_loss_func`

### 3.1 logits / scores / probs / routing_map

`TopKRouter` 的 docstring 把命名约定交代得很清楚（`router.py:159-170`，下面意译）：

```python
# logits:       router gating 网络的原始输出
# scores:       score function 之后的分数，用来选专家、算 aux loss
# probs:        top-k 权重，用来加权专家输出
# routing_map:  token 到 expert 的布尔掩码
```

`score_function` 有三个分支：`softmax`、`sigmoid`、`sqrtsoftplus`（`moe_utils.py:817-835`）。选择与归一化的实现在 `topk_routing_with_score_function`（`moe_utils.py:696`）。

### 3.2 归一化规则与 expert_bias

规则比想象中细，三种情况要分开记：

- `softmax` + `moe_router_pre_softmax=False`（默认）：先对原始 logits 取 top-k，再对选中的 k 个值做 softmax → 权重在 k 内和为 1；
- `softmax` + `pre_softmax=True`：先对全部 E 个专家 softmax，再取 top-k → 选中权重之和小于 1，不重新归一；
- `sigmoid` / `sqrtsoftplus`：`topk > 1` 时用选中分数之和重新归一（分母加 `1e-20`），`topk == 1` 直接取原始分数。

所以「sigmoid 的路由权重不归一」只在 topk=1 时成立。另外 `moe_router_topk_scaling_factor` 是在归一化之后无条件乘上去的（配置注释说它只在 pre-softmax 下有效，代码里并没有这个检查）。

`expert_bias` 的位置值得记：它只加在「用来选专家」的分数上，之后再用 `gather` 取回不带 bias 的分数当权重（`moe_utils.py:830-835`）。也就是说 bias 只影响谁被选中，不改权重。

这条路径叫 aux-loss-free 均衡：`moe_router_enable_expert_bias` 默认关闭，开启时强制 score_function 是 `sigmoid` 或 `sqrtsoftplus`（`transformer_config.py:2953-2960`），偏置按全局 batch 内各专家分到的 token 数增减。

### 3.3 aux loss 的 f、P 与跨 rank 聚合

公式在 docstring 里（`moe_utils.py:74-82`），实现在同一函数的末尾（`moe_utils.py:154-157`）。记 k 为 topk、m(x,i) 为路由掩码、s(x,i) 为 aux loss 用的稠密分数：

$$
f_i = \frac{1}{T k} \sum_{x} m(x, i)
$$

$$
P_i = \frac{1}{T} \sum_{x} s(x, i)
$$

$$
L_{\text{aux}} = \alpha E \sum_{i=1}^{E} f_i P_i
$$

`f_i` 是专家 i 实际吃到的 token 比例，`P_i` 是被期望分到的概率。均衡时两者一致、内积最小；某个专家被「饿」（`f_i` 低但 `P_i` 高）时乘积抬高损失，反推路由走向均衡。

两个容易讲错的点：`P_i` 用的是**稠密分数**，对全部 E 归一化（`compute_routing_scores_for_aux_loss`，`moe_utils.py:866`），不是 top-k 之后那个稀疏的 `probs`；跨 rank 时 `f_i` 是 all-reduce 后的**全局**计数，`P_i` 只在本地算，每个 rank 贡献「全局 f × 本地 P_j」，梯度累加后正好等于全局公式（docstring `moe_utils.py:84-97` 有推导）。

所以别把 `probs` 也 all-reduce 一遍：`P_i` 的跨 rank 求和本来就由「各 rank 的 loss 相加」完成，提前聚合等于重复计算。

## 4. 工程细节与坑

**专家是「对本地专家做 grouped GEMM」，不是拼成一个大 GEMM。** `TEGroupedMLP`（`experts.py:173`）拿到的是 `num_local_experts`，等于 `num_moe_experts // ep_size`（`moe_layer.py:190`），并把 `tokens_per_expert` 一起传给 TE 的 `GroupedLinear`。

准确的语义是：每个本地专家保留自己的权重矩阵，grouped GEMM 把「M 各不相同的一组 GEMM」合成一次 launch。说成「把 E 个专家权重拼成一个大矩阵、做一次大 GEMM」是错的——那算出的是稠密乘积，数学上根本不是 MoE。

**路由概率乘在激活上，不是乘在专家输出上。** `permuted_probs` 在 `bias_act_func` 里就乘到 fc1 输出上（`experts.py:874`）；因为 fc2 是线性的，先乘后乘等价，于是 combine 阶段只剩求和。`moe_apply_probs_on_input` 走另一条路（概率乘在输入上），且要求 `moe_router_topk == 1`。

**共享专家的输出在 `postprocess` 相加。** `output = output + shared_expert_output`（`moe_layer.py:649`）。开 `moe_shared_expert_overlap` 时，共享专家在旁路 stream 上与 dispatch 通信重叠，收尾仍在 `postprocess`。

**aux loss 默认不加。** `moe_aux_loss_coeff` 默认 `0.0`（`transformer_config.py:949`），要显式给系数（注释建议从 `1e-2` 起）。它的挂载方式也别忽略：`MoEAuxLossAutoScaler`（`moe_utils.py:270`）forward 原样返回输入，backward 才注入 aux loss 的梯度——这样 aux loss 不占用 logits 的前向计算图，只在反向多一项。

## 5. 小结

- **四段流水线**：route(+preprocess) → dispatch → routed_experts_compute → combine(+postprocess)，docstring 在 `moe_layer.py:678-682`。
- **通信被 dispatcher 收走**：`dispatch` / `combine` 只是转发，六个钩子里只有两个含通信。
- **默认不是 all-to-all**：`allgather` dispatcher 在 TP×EP 上 all-gather + reduce-scatter。
- **四个名字**：logits / scores / probs / routing_map；sigmoid 在 topk>1 时仍会归一，`expert_bias` 只影响选择。
- **aux loss 默认关闭**：`f_i` 全局、`P_i` 本地且用稠密分数，梯度靠 `MoEAuxLossAutoScaler` 注入。
- **专家是本地 grouped GEMM**：每个专家独立权重，概率乘在激活上，共享专家输出在 `postprocess` 相加。

下一篇讲 **多模态（LLaVA）实现**：视觉编码器怎么把图像 patch 变成 token、图文怎么拼进同一序列，也就是《[多模态（LLaVA）实现]({{< relref "megatron-code-15-multimodal" >}})》。

---

**版本提示（2026-10）**：dev 新增 megakernel/MOK 后端，用 `config.moe_megakernel_backend="mok"` 整体替换 dispatch + 专家计算 + combine。

另新增 MoonEP 式虚拟专家复制（`megatron/core/transformer/moe/virtual_expert_load_balancer.py`，docstring 自述算法取自 MoonEP，开关 `config.moe_virtual_expert_load_balance`）。四段式骨架与四个命名约定未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
