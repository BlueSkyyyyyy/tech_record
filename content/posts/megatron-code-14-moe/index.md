---
title: "Megatron 源码精读（十四）：MoE 实现与优化"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "moe", "专家并行"]
categories: ["训练框架"]
weight: 14
series: ["megatron-code"]
---

上一篇[《强化学习》]({{< relref "megatron-code-13-rl" >}})讲了 RL 后训练。本篇回到模型结构本身，精读 MoE（Mixture-of-Experts）的实现——`megatron/core/transformer/moe/` 目录，核心是 `moe_layer.py`（MoELayer，901 行）、`router.py`（路由，1061 行）、`experts.py`（专家实现）、`moe_utils.py`（aux loss）。

MoE 一句话：每个 token 只走少数几个专家，用「稀疏激活」换「更大参数总量」。它给训练框架带来的三个新难题正是本篇主线：

1. **token 怎么路由**：每个 token 选哪几个专家、权重多少；
2. **跨设备怎么搬运 token**：专家分散在不同 rank，token 要先 all-to-all 发过去、算完再 all-to-all 回来；
3. **怎么防止专家饿死/撑死**：负载不均衡时怎么办（aux loss + capacity/dropless）。

---

## 1. 问题背景：稀疏激活 vs 密集硬件

密集 transformer 里，每个 token 都要过完整的 FFN；MoE 把 FFN 换成 E 个专家，每个 token 只进 top-k 个（k 通常 1~8，E 通常 8~256），其余专家对这个 token 是「关着」的。好处是总参数量能放大 E 倍而单 token 计算量几乎不变；代价是：

- 路由本身引入了「软性」的稀疏性，需要额外机制保证专家负载均衡（否则路由可能把 token 全塞给少数专家）；
- token 的分布是动态的、数据依赖的，**每一层、每个 step 都不一样**，这跟前一篇 checkpoint 篇里「静态切分」截然不同。

所以 MoE 的代码骨架是四段式流水线 + 一套路由/均衡机制。

---

## 2. 现象：MoELayer 的「route → dispatch → compute → combine」四段

`MoELayer.forward` 的 docstring 直接点明四步（`megatron/core/transformer/moe/moe_layer.py:676-681`）：

```python
# megatron/core/transformer/moe/moe_layer.py:678-681（docstring）
# 1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
# 2. Dispatch: Tokens are sent to the expert devices using communication collectives.
# 3. Expert Computation: Experts process the dispatched tokens.
# 4. Combine: Expert outputs are sent back to their original devices.
```

对应的方法分别是 `route`/`preprocess`（`moe_layer.py:451-467`）、`dispatch`（`moe_layer.py:520-529`）、`routed_experts_compute`（`moe_layer.py:593`）、`combine`（`moe_layer.py:627-634`）+ `postprocess`（`moe_layer.py:636`）。关键看 dispatch 和 combine 两段是怎么依赖通信的：

```python
# megatron/core/transformer/moe/moe_layer.py:520-529
def dispatch(self, hidden_states, probs):
    if self.config.overlap_dispatch_backward_with_experts_wgrad:
        hidden_states = _RegisterDelayedWgradForExperts.apply(self, hidden_states)
    return self.token_dispatcher.token_dispatch(hidden_states, probs)

# megatron/core/transformer/moe/moe_layer.py:627-634
def combine(self, output):
    output = self.token_dispatcher.token_combine(output)
    return output
```

注意 `dispatch` 和 `combine` 都**只是把活转发给 `token_dispatcher`**（`moe_layer.py:529`、`633`）。这个 dispatcher（`token_dispatcher.py`）才是真正做 all-to-all 的地方：dispatch 阶段把每个 token 连同它的路由概率一起 all-to-all 到「它选中专家所在的 rank」，combine 阶段再把专家输出 all-to-all 回原始 token 位置。**「路由是算的、通信是 dispatcher 做的、专家计算是 batched GEMM」三者彻底解耦**——这是 MoE 代码最值得记住的分工。

---

## 3. 根因一：TopKRouter 的「scores / probs / routing_map」三级概念

路由的核心是 `TopKRouter`（`router.py:157`），docstring 把 workflow 和命名约定交代得很清楚（`router.py:158-171`）：

```python
# megatron/core/transformer/moe/router.py:158-170（docstring）
# (1) Calculate the logits by the router gating network.
# (2) Calculate the routing probabilities and map for top-k selection with score function.
# (3) [Optional] Apply token dropping to top-k expert selection.
# (4) [Optional] Apply the auxiliary load balancing loss.

# 命名约定：
#   logits:       gating 网络输出的原始 logits
#   scores:       score function 之后的分数（用于选专家、算 aux loss）
#   probs:        topk 权重（用于组合专家输出）
#   routing_map:  token 与 expert 之间的掩码映射
```

**这段命名约定是理解全文件的钥匙**。`scores` 和 `probs` 的区别很微妙：`score_function` 决定「用什么激活函数把 logits 变成可比较的分」（`router.py:738-741`）：

```python
# megatron/core/transformer/moe/router.py:738-741（节选）
if self.score_function == "softmax":
    scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
elif self.score_function == "sigmoid":
    scores = torch.sigmoid(logits.float()).type_as(logits)
```

- **softmax**：各专家分数和为 1，top-k 选完后权重是「归一化后的软权重」（Switch 风格）；
- **sigmoid**：各专家独立打分，top-k 权重不再强行归一（DeepSeek 风格，配合 expert bias）。

选完 top-k 得到 `routing_map`（硬掩码：token 命中某专家的位置为 1 其余为 0），再据此把 `scores` 里非选中的位置清零、按需归一化，才是最终用来加权专家输出的 `probs`。

---

## 4. 根因二：aux loss 就是「专家负载 × 专家概率」的内积

负载均衡靠辅助损失，实现在 `switch_load_balancing_loss_func`（`moe_utils.py:58-152`）。docstring 把公式写全了（`moe_utils.py:74-82`）：

```python
# megatron/core/transformer/moe/moe_utils.py:74-82（docstring）
# loss = E * Σ_{i=1}^{E} (f_i * P_i)
#   f_i = 1/(T*topk) * Σ_{x∈B} routing_map(x, i)   # 派给专家 i 的 token 比例
#   P_i = 1/T * Σ_{x∈B} probs(x, i)                  # 专家 i 的平均路由概率
#   E = 专家数，T = batch 内 token 总数
```

直觉：`f_i` 是「专家 i 实际吃到的 token 占比」（由硬路由决定），`P_i` 是「专家 i 被期望分配的概率」（由软概率决定）。若负载均衡，两者应一致；`f_i * P_i` 求和会最小。当某些专家被「饿」（`f_i` 低但路由倾向它 `P_i` 高）时，乘积拉高损失，反向传播会把路由往均衡方向推。

Leaky 细节都在 docstring 的分布式部分（`moe_utils.py:84-97`）：序列/上下文并行下每个 rank 只算本地子 batch 的 `P_ij`，再跨 rank 聚合。这是「aux loss 的正确性依赖于多 rank 协作」的一个具体体现。

---

## 5. 解法：专家 = 一次大 GEMM（batched expert）

最后补「专家到底怎么算」这一环。`experts.py` 里 `TEGroupedMLP`（`experts.py:173`）把 E 个专家的 FFN 拼成一次大矩阵乘：不再逐个专家调 `Linear`，而是把 E 个专家的权重矩阵堆叠、把分给同一 rank 的所有 token 的隐状态拼接，做**一个 batched GEMM**（实际通过 TE 的 grouped gemm）。这正是「稀疏激活」落到硬件上的关键：虽然 token 稀疏地散在 E 个专家里，但同一 rank 上「被分到专家 i 的那批 token」可以一起算，GPU 的矩阵乘单元依然满载。

与之配套的还有 `shared_experts.py` / `shared_experts_compute`（`moe_layer.py:532-557`）：DeepSeek 风格里，除了 top-k 路由专家，还有「所有 token 都过的共享专家」，共享专家的输出在 `postprocess` 里直接加到路由输出上（`moe_layer.py:648-649` 的 `output = output + shared_expert_output`）。

---

## 6. 小结

- **四段流水线**：route → dispatch(all-to-all) → expert compute → combine(all-to-all)，`moe_layer.py:676-681` 是总纲。
- **dispatcher 解耦通信**：`dispatch`/`combine` 只是转发给 `token_dispatcher`，真正的 all-to-all 在其中（`moe_layer.py:520-529`、`627-634`）。
- **三级概念 logits/scores/probs/routing_map**：score_function 选 softmax 还是 sigmoid，决定了 top-k 权重是否归一（`router.py:158-171`、`738-741`）。
- **aux loss = E·Σ(fᵢ·Pᵢ)**：专家实际负载 × 期望概率的内积，跨 rank 聚合（`moe_utils.py:58-97`）。
- **专家 = batched GEMM + shared experts**：`TEGroupedMLP` 把 E 个专家拼成一次大矩阵乘（`experts.py:173`），shared expert 输出后加（`moe_layer.py:648-649`）。

下一篇讲 **多模态（LLaVA）实现**：视觉编码器如何把图像 patch 变成 token、image token 和文本 token 怎么拼进同一个序列、以及多模态的数据/训练循环和纯文本有哪些不同。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/transformer/moe/moe_layer.py`、`megatron/core/transformer/moe/router.py`、`megatron/core/transformer/moe/experts.py`、`megatron/core/transformer/moe/moe_utils.py`。）
