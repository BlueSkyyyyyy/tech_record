---
title: "Megatron 源码精读（十二）：fused 算子"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "cuda", "算子融合"]
categories: ["训练框架"]
weight: 12
series: ["megatron-code"]
---

上一篇[《优化器》]({{< relref "megatron-code-11-optimizer" >}})收束在「下一步看 fused 算子」。本篇就进入 MCore 的 `megatron/core/fusions/` 目录——一窝把多个 PyTorch 原语合并成单个 CUDA kernel 的算子，包括 `fused_layer_norm.py`、`fused_softmax.py`、`fused_cross_entropy.py`、`fused_bias_gelu.py` 等。

它们共享同一个动机和同一套工程范式，本篇不逐个展开，只抓三点：

1. **为什么要把算子写进 CUDA**——GPU 的瓶颈在哪，fuse 到底省了什么；
2. **一个典型例子**：`FusedLayerNorm` 的两档 kernel 与「固定 shape 白名单」；
3. **优雅降级范式**：`FusedScaleMaskSoftmax` 的「能 fuse 就 fuse、不能就退回 torch」，以及 fused cross entropy 的 TP 协同。

---

## 1. 问题背景：为什么不是「写 PyTorch 让编译器帮你并」

直觉上「把 layer norm、softmax 写成几个 PyTorch 调用」最省事。但 attention 里 `scale → mask → softmax` 是三个分离的 kernel，产量再高也要在中间 tensor 上做一次完整的写 + 读。而 transformer 的这些逐元素/归一化算子，**瓶颈几乎都在 memory bandwidth，不在算力**——数据每多一次 global memory 往返就多一次浪费。

fuse 的价值就是这两点：

- **省 launch 开销**：一次 kernel 取代三次 kernel 的启动与调度；
- **省访存往返**：中间结果（缩放后的 logits、减均值后的残差）不再写回 global memory，直接在寄存器/共享内存里流转。

但一旦写死一个 CUDA kernel，就丢掉了 PyTorch 的灵活性和自动求导，于是 MCore 的 fused 算子几乎都遵循同一条铁律：**用 `torch.autograd.Function` 把 CUDA kernel 接进自动求导，并保留「不满足约束就降级到纯 torch」的后路。** 本章核心就是拆这条铁律的三处落地。

---

## 2. 现象：`FusedLayerNorm` 的「两档 + 白名单」

`FusedLayerNorm`（`megatron/core/fusions/fused_layer_norm.py:30-169`）把「减均值、除标准差、乘 gamma 加 beta」压成一个 kernel。看它的 forward（`fused_layer_norm.py:131-168`）：

```python
# megatron/core/fusions/fused_layer_norm.py:135-149（节选）
if self.persist_layer_norm:
    output = FastLayerNormFN.apply(input, weight, self.bias, self.eps, ...)
    output = make_viewless_tensor(inp=output, requires_grad=input.requires_grad, keep_graph=True)
else:
    return FusedLayerNormAffineFunction.apply(input, weight, self.bias, self.hidden_size, self.eps, ...)
```

两个关键观察：

1. **两档 kernel**：`persist_layer_norm`（apex 的 `FastLayerNormFN`，persistent kernel，常驻复用）和普通 `FusedLayerNormAffineFunction`（apex 的 fused layer norm）。persistent kernel 更快，但只支持一组固定的 `hidden_size`（`fused_layer_norm.py:73-98` 那个白名单：1024/2048/.../65536），不在列表里或没装 apex 就退回非 persistent 档（`fused_layer_norm.py:100-101`）。
2. **viewless tensor 的坑**（`fused_layer_norm.py:143-149`）：`FastLayerNormFN` 返回的 tensor 带 `_base` 字段（是个 view），会触发 `schedule.py` 里 `deallocate_output_tensor()` 的报错，所以用 `make_viewless_tensor` 把它「洗白」成非 view。

还有 `zero_centered_gamma`（`fused_layer_norm.py:133`）：`weight = self.weight + 1 if zero_centered_gamma else self.weight`，把 gamma 存成「围绕 0 的偏移」以改善数值稳定性。这些都是「为了快/稳，代价是接口变得挑剔」的具体例证。

---

## 3. 根因：`FusedScaleMaskSoftmax` 的「条件融合」

softmax 更有代表性，因为它是「scale + mask + softmax」三个操作，且 mask 有 causal / pad 之分。核心在 `FusedScaleMaskSoftmax`（`megatron/core/fusions/fused_softmax.py:179-297`）的 `forward`：

```python
# megatron/core/fusions/fused_softmax.py:233-236
if self.is_kernel_available(mask, *input.size()) and softmax_offset is None:
    return self.forward_fused_softmax(input, mask)
else:
    return self.forward_torch_softmax(input, mask, softmax_offset)
```

**这是整个 fused 算子模块最值得背下来的一段**：「先问 kernel 支不支持，支持走融合，不支持退回 torch」。`is_kernel_available`（`fused_softmax.py:238-270`）列的约束非常具体——这也是「写死 kernel」的代价的直观体现：

```python
# megatron/core/fusions/fused_softmax.py:253-259（节选）
self.scaled_masked_softmax_fusion   # 用户开了融合
and self.input_in_float16           # 输入必须是 fp16/bf16
and 16 < sk <= 4096                 # key 长度 16~4096
and sq % 4 == 0                     # sq 是 4 的倍数
and sk % 4 == 0                     # sk 是 4 的倍数
and attn_batches % 4 == 0           # (b*np) 是 4 的倍数
```

任何一条不满足（例如序列长度不是 4 的倍数、超长上下文 sk>4096、或者用了自定义 `softmax_offset`），就整段退回 `forward_torch_softmax`（`fused_softmax.py:299`），用 `torch.nn.functional.softmax` 老老实实算。真正的融合 kernel 藏在几个 `torch.autograd.Function` 子类里：

- `ScaledUpperTriangMaskedSoftmax`（`fused_softmax.py:11-57`）：causal 自注意力的「scale + 上三角 mask + softmax」，forward 里 `import scaled_upper_triang_masked_softmax_cuda` 调手写 CUDA（`fused_softmax.py:31-34`），backward 同样调 CUDA（`fused_softmax.py:50-55`）；
- `ScaledMaskedSoftmax`（`fused_softmax.py:60`）：带显式 mask 的版本；
- `ScaledSoftmax`（`fused_softmax.py:108`）：纯缩放 softmax。

**注意三者的共性**：都是 `torch.autograd.Function`，forward 里 `import xxx_cuda` 调手写 kernel，同时把结果 `save_for_backward` 存起来给 backward 复用——这样手写 CUDA 就无缝接进了自动求导图，上层调用方无感知。

---

## 4. 解法：fused cross entropy 里的 TP 协同

最后一类值得讲的是 `fused_cross_entropy.py` 的 `_VocabParallelCrossEntropy`（`megatron/core/fusions/fused_cross_entropy.py:87`）。它融合的是「logits 求 softmax + 取 target 位置的 -log」这个 cross entropy，但加了一个分布式维度——vocab 被 TP 切分，每个 rank 只有一部分 logits：

```python
# megatron/core/fusions/fused_cross_entropy.py:110-124（节选，docstring）
# In the fused case, tensors are batches to invoke a single
# kernel ...
def backward(ctx, grad_output):
    ...
```

`fused_vocab_parallel_cross_entropy`（`fused_cross_entropy.py:136`）这个入口，把「每个 TP rank 各自算局部 softmax 的 max/sum（`calculate_logits_max`，`fused_cross_entropy.py:13`）→ all-reduce 出全局 max/sum → 各自算局部 loss → all-reduce 累加」这一串，用几个小函数（`calculate_predicted_logits` / `calculate_cross_entropy_loss` / `calculate_gradients`，`fused_cross_entropy.py:26-85`）拆开。它说明 fused 不只是「单卡算得快」，还包括**把 TP 通信（all-reduce）编排进融合流程，减少中间越界张量的物化**。

这一节不必深究每个函数行号，重点是补全图景：`megatron/core/fusions/` 里剩余那些 `fused_bias_gelu.py`、`fused_bias_swiglu.py`、`fused_bias_dropout.py`、`fused_rotary`、`fused_mrope`、`fused_mla_yarn_rope_apply`，走的全是同一条路——`autograd.Function` + 手写 CUDA + 条件降级。

---

## 5. 小结

- **fuse 省 bandwidth 和 launch，不省计算**：逐元素/归一化算子是 memory-bound，融合是为了少读写 global memory、少启动 kernel。
- **铁律是 `torch.autograd.Function` + 条件降级**：手写 CUDA 接进自动求导，`is_kernel_available` 一条条对约束，不满足就 `forward_torch_softmax` 退回纯 torch（`fused_softmax.py:233-270`）。
- **`FusedLayerNorm` 两档 kernel**：persistent（固定 hidden_size 白名单）+ 非 persistent（apex fallback），外加 `zero_centered_gamma` 和 `make_viewless_tensor` 洗 view 的坑（`fused_layer_norm.py:100-149`）。
- **mask 三态对应三个 autograd 函数**：`ScaledUpperTriangMaskedSoftmax`（causal）/ `ScaledMaskedSoftmax`（pad）/ `ScaledSoftmax`（无 mask）（`fused_softmax.py:11-108`）。
- **fused 也管 TP 编排**：`fused_vocab_parallel_cross_entropy` 把「TP 局部 softmax + all-reduce」融合进 CE 计算，减少稠密中间量（`fused_cross_entropy.py:13-136`）。

下一篇讲 **强化学习**：MCore 里 RLHF/GRPO 的 rollout、reward 模型打分、和策略/价值更新，怎么和「训练框架 + 数据并行」这套既有基础设施拼起来，也就是《[强化学习]({{< relref "megatron-code-13-rl" >}})》。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/fusions/fused_layer_norm.py`、`megatron/core/fusions/fused_softmax.py`、`megatron/core/fusions/fused_cross_entropy.py`。）
