---
title: "Megatron 源码精读（四）：激活重计算——recompute.py 精读"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "显存优化"]
categories: ["训练框架"]
weight: 4
series: ["megatron-code"]
---

全量重计算到底多花多少算力？`recompute_method` 和 `recompute_granularity` 各管什么？这是读 `recompute.py` 时最容易含糊的两件事。

结论：**全量重计算只是把前向多跑一遍，训练算力从 $6P$ 涨到 $8P$（约 +33%），不是翻倍；`granularity` 决定重算发生在哪一层（整层还是层内子模块），`method` 只在前者里有定义。**

上一篇[并行拓扑]({{< relref "megatron-code-03-parallel-topology" >}})只说了 PP 靠 1F1B 省激活，那管的是「同时在飞的 microbatch 数」；本篇补上单个 microbatch 内部的那一刀。

## 1. 问题：激活是显存第一大头，重算是唯一不切模型的解法

反向要用层内中间量：$\partial L/\partial W = X^{\top}\,dY$ 里的 $X$ 就是前向激活。全存下来，显存随层数线性膨胀；可这些量都能由「输入 + 权重」重算出来，代价是再算一遍前向。

$$
\begin{aligned}
T_{\text{fwd}} &= 2P \\
T_{\text{bwd}} &= 4P \\
T_{\text{extra}} &= 2P \cdot \frac{N}{L}
\end{aligned}
$$

$P$ 是参数量（按每 token 前向 $2P$、反向 $4P$ 的通行估算），$N/L$ 是被重算的层占比。全量重算因此是 $6P \to 8P$，约 +33%；`block` 只重算 $N$ 层就按比例摊薄；selective 只重算 attention 核心，增量更小。

省下来的是「层数 × 中间量」，留下的是「层数 × 输入量」。中间量通常比输入大几倍，attention 的 $QK^{\top}$ 更是 $s^{2}$ 量级，这才是它值得单独做机制的原因。

## 2. 两个维度：先选粒度，再选层

结论：`recompute_granularity` 是总开关，`recompute_method` 只是它下面 full 分支的细分参数，两者不能并列着讲。

- `full`：整段层的前向交给 checkpoint 包装，此时才需要 `recompute_method`（`uniform`/`block`）与 `recompute_num_layers`。
- `selective`：不走整层 checkpoint，而是让 `recompute_modules` 里的子模块各自包一层，默认 `["core_attn"]`。此时 `recompute_num_layers` 必须为 `None`（`transformer_config.py:2297-2301`）；`recompute_method` 则不会被读取——校验不强制它为空，但代码路径根本不看它。
- `None`：不重算。

配置校验把这条关系写死了：`granularity != 'selective'` 却没给 `method` 会直接报错（`transformer_config.py:2283-2287`）。所以「`method=block` 只重算前 N 层」这句话只在 full 下成立。

selective 里「重算哪些」由 `recompute_modules` 回答，候选是 `core_attn`/`mlp`/`layernorm`/`moe`/`moe_act`/`shared_experts`/`mla_up_proj`/`mhc`/`gdn`/`gdn_norm_out`（`transformer_config.py:2311-2327`）。

## 3. `uniform` 按块切，`block` 按层数算

`checkpointed_forward` 主体只是一个 if/elif（`recompute.py:147-173`）。

### 3.1 uniform：层切成 `num_layers` 一组，每组只存块输入的激活

```python
layer_idx = 0
while layer_idx < self.num_layers_per_pipeline_rank:
    chunk_end = min(layer_idx + self.config.recompute_num_layers, ...)
    chunk_runner(layer_idx, chunk_end, True)
    layer_idx += self.config.recompute_num_layers
```

块内各层正常前向、正常存激活，只有块边界那一份输入被 checkpoint 记住。`recompute_num_layers=1` 是最细的一档：每层一个 checkpoint，省得最多、重算也最多。

### 3.2 block：只 checkpoint 前 `num_layers` 层，其余层完全不重算

```python
recompute_skip_num_layers = 0
for layer_idx in range(self.num_layers_per_pipeline_rank):
    if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
        recompute_skip_num_layers += 1
    use_checkpoint = (layer_idx >= recompute_skip_num_layers
        and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers)
    chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
```

两个 `recompute_num_layers` 意思不同：uniform 里是「块大小」，block 里是「重算多少层」。

docstring 写得很直白——block 只重算每个 pipeline stage 里固定数量的层，其余层不做任何重算（`transformer_config.py:612-617`）。

## 4. 代码落点：`chunk_runner` 只按精度分流

`chunk_runner(start, end, use_checkpoint)`（`recompute.py:116-145`）里，决定用哪套 checkpoint 的分支只有精度一条：

```python
cf = custom(start, end)
args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
if self.config.fp8 or self.config.fp4:
    hidden_states, context = te_checkpoint(cf, self.config.distribute_saved_activations,
        tensor_parallel.random.get_cuda_rng_tracker, self.pg_collection.tp, *args)
else:
    hidden_states, context = tensor_parallel.checkpoint(
        cf, self.config.distribute_saved_activations, *args)
```

`custom(start, end)` 返回闭包 `custom_forward`（`recompute.py:54-114`），它遍历 `self.layers[start:end]` 逐层前向——这才是被 checkpoint 包住的「可重放片段」。

注意它在每层前重建量化 context：`use_inner_quantization_context` 为真时按 `layer.layer_number - 1` 取 `get_fp8_context`/`get_fp4_context`（`recompute.py:64-77`），否则重放会拿着上一层的 amax/scale 去算下一层。

签名里还有两个与显存无关的旁路参数：`extract_layer_indices` 和 `layer_offset`。前者要求从指定全局层号抽中间特征，后者把本 rank 的局部层号换算成全局层号。

命中时把该层输出攒进 `intermediate_hidden_states`，返回值也从单张量变成二元组（`recompute.py:140-145`、`176-177`）。

一个容易讲错的地方：`recompute.py` 不是 dense GPT 的必经之路。`TransformerBlock` 里有一份同构内联实现 `_checkpointed_forward`（`transformer_block.py:581-750`），uniform/block 分支、精度分流、skip 逻辑、逐层量化 context 与它一一对应。

`recompute.checkpointed_forward` 是抽出来的公共版，当前由 `HybridBlock` 调用（`models/hybrid/hybrid_block.py:1106`）；读 dense 路径时别在 `recompute.py` 里找调用点。

## 5. 工程细节与坑

### 5.1 fp8/fp4 下重算窗口会整体后移

TE 的 checkpoint 要求重放时至少有一个输入带梯度，否则 re-entrant autograd 立不起来。所以 block 分支里，凡是当前 `hidden_states` 不带梯度的层一律不 checkpoint。

`recompute_skip_num_layers` 记下跳过了几层，窗口整体后移同样层数（`recompute.py:157-171`）。

典型场景是流水线中段：跨 stage 收到的激活是通信产物，本身不带梯度，于是第 0 层被跳过、窗口变成 $[1, N+1)$。bf16 路径没有这个 skip——这不是精度 bug，是 TE 路径的必要让步。

### 5.2 `CheckpointFunction` 额外管一套 RNG

bf16 路径走 `random.py` 的 `CheckpointFunction`（`random.py:634`），相对 `torch.utils.checkpoint` 只改了两处：`_set_cuda_rng_state` 取代 `torch.cuda.set_rng_state`；model-parallel tracker 也一起快照/复位。

forward 用 `ctx.rng_states = _get_all_rng_states()` 存全量状态、`torch.no_grad()` 跑一遍、只 `save_for_backward` 输入。

backward 用 `with _fork_rng(): _set_all_rng_states(*ctx.rng_states)` 回到前向那一刻，再 `torch.enable_grad()` 重放（`random.py:677-715`）。

`_fork_rng` 退出时恢复原状态，重放不污染训练主随机流——这部分留到[第五篇]({{< relref "megatron-code-05-rng-seeds" >}})展开。

CUDA graph 有个特判：`is_graph_warmup()` 或 `is_graph_capturing()` 时，`checkpoint()` 直接 `return function(*args)`（`random.py:718-730`），因为捕获好的图里跑不了重算。

backward 里还有一道守卫：`torch.autograd._is_checkpoint_valid()` 为假就抛「不兼容 `.grad()`」，只有正在做 graph capture 时放行（`random.py:682-686`）。

### 5.3 `distribute_saved_activations` 与 sequence parallel 互斥

checkpoint 已把「存中间量」压成「存输入」，但输入在大 TP 下也不小。这个开关再切一刀。

forward 用 `split_tensor_into_1d_equal_chunks` 只留本 rank 的 $1/\mathrm{tp}$，backward 用 `gather_split_1d_tensor` 拼回原形状再重放（`random.py:664-668`、`690-693`）。

约束是配置校验直接抛 `ValueError`，不是 assert（`transformer_config.py:2304-2308`）：SP 已沿 seq 维切过激活，再沿 TP 切一遍，两套切分语义会打架。

另一个副作用：切分是 `numel // tp_size` 且不做整除检查（`tensor_parallel/utils.py:63`），所以第一个输入的元素数必须能被 TP size 整除；不能被整除时不会在切分处报错，而是回退时 `view(ctx.input_0_shape)` 才炸。

## 6. 小结

- 全量重算 = 多跑一遍前向：$6P \to 8P$，约 +33%；block 按层数摊薄，selective 更小。
- `granularity` 先决定 full 还是 selective；`method` 只在 full 下被读取，selective 下即使填了也不生效。
- uniform 的 `num_layers` 是块大小，block 的 `num_layers` 是重算层数，两个名字一个含义陷阱。
- `chunk_runner` 只有一处分流：fp8/fp4 走 `te_checkpoint`，其余走 `tensor_parallel.checkpoint`，后者自己管 RNG 快照与重放。
- fp8/fp4 下 block 的重算窗口会后移；`distribute_saved_activations` 与 SP 互斥。

下一篇：[随机种子的设置]({{< relref "megatron-code-05-rng-seeds" >}})把本篇反复出现的 `_get_all_rng_states` / `_fork_rng` 讲透。

---

**版本提示（2026-10，对外配置语义未变）**：

- `CheckpointWithoutOutput` 内部重构（#8334/#8390）：`checkpoint()` / `discard_output_and_register_recompute()` 签名未变，内部换成 `_RecomputeRegion` + `CheckpointWithoutOutputManager`。构造时不传 `fp8`（默认 `False`）现在也会走 TE 的 recompute context（判断写成 `fp8 is not None`），依赖这个默认值的 selective 模块（`layernorm`/`moe_act`/`gdn_norm_out`）行为与基线不同。
- `tensor_parallel.checkpoint` 现在会在 forward 记录 FP8 autocast 状态、backward 重进（#7538）；基线没有这一步，而 selective 的 `core_attn`/`mlp`/`moe_act` 重算正是走它（`attention.py:559`、`transformer_layer.py:994`），fp8 下重放会掉出 autocast。
- GDN/KDA 重算丢 FP8/FP4 context 已修（#8203）；基线 `recompute.py` 的 fp4 分支还留着 TODO（`recompute.py:69`）。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已到 2026-10-29，行号会漂移，定位请以符号名为准。
