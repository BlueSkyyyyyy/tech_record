---
title: "Megatron 源码精读（四）：激活重计算——recompute.py 精读"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "显存优化"]
categories: ["训练框架"]
weight: 4
series: ["megatron-code"]
---

这是「Megatron-LM 源码精读」系列的第四篇。前三篇搭好了骨架（[结构]({{< relref "megatron-code-01-structure" >}}) → [并行原理]({{< relref "megatron-code-02-parallel-principles" >}}) → [并行拓扑]({{< relref "megatron-code-03-parallel-topology" >}})），本篇进入第一个「显存工程」主题：**激活重计算（activation recomputation / checkpointing）**。核心文件只有一个——`megatron/core/recompute.py`（分析基准 `f713506ce`，179 行），但它背后牵着 PyTorch 的 checkpoint 机制、TP 的 RNG tracker、以及 Transformer Engine 的 fp8 重算，值得单独一篇。

---

## 1. 问题背景：激活显存是训练的第一大头

前一篇讲 PP 时提到 1F1B 靠交错调度省激活，但那只解决「流水线各 stage 同时在飞的 microbatch 数」这一层。真正的大头在**单个 microbatch 内部**：一个 Transformer 层前向时，为反向存下来的中间激活（QKV 投影结果、softmax 前的 logits、attention 得分矩阵、MLP 的 hidden 等）体积远超参数本身。

核心矛盾是：

- **反向需要层中间量的值**（比如 $\partial L/\partial W = X^T \cdot dY$，需要 $X$）；直接存下所有这些中间量，显存爆炸。
- **中间量可以由输入 + 权重重新算出来**，代价是**多花一倍的 flops**。

于是有了激活重计算：**前向时只存输入（或少量关键中间量），反向时重放（replay）一遍前向把中间量重新算出来**。用「计算」换「显存」。

第 2 篇里那张并行表还没回答「省的是什么」的这一维，这篇补上：重计算不切模型不作通信，单纯把「存激活」换成「重算激活」。

## 2. 入口：`checkpointed_forward` 挂在 `TransformerBlock` 前向里

`transformer_block.py` 的前向在 `self.config.recompute_granularity` 非空时，不走普通的逐层循环，而是把整段层的前向委托给 `recompute.checkpointed_forward`（`transformer_block.py:312-313` 附近决定是否开 selective）。`checkpointed_forward` 的签名（`recompute.py:21-35`）很直白：吃进 `hidden_states / attention_mask / context / rotary_pos_emb` 等一整套 layer kwargs，返回最终的 `hidden_states`（或带上 `intermediate_hidden_states`，用于特征抽取）。

它只干两件事，分别对应两个配置维度：

1. **「重算哪些层」** —— 由 `recompute_method`（`uniform` / `block`）决定，见第 3 节；
2. **「每层重算多细」** —— 由 `recompute_granularity`（`full` / `selective` / `None`）决定；selective 时只在层内某个子模块（默认 `core_attn`）上做 checkpoint，见第 5 节。

## 3. `recompute_method`：均匀切块 vs. 只重算前若干层

`checkpointed_forward` 主体是一个 `if/elif`（`recompute.py:147-173`），按 `recompute_method` 分路：

### 3.1 uniform：把层均匀分组，每组 checkpoint 一次

```python
if self.config.recompute_method == 'uniform':
    layer_idx = 0
    while layer_idx < self.num_layers_per_pipeline_rank:
        chunk_end = min(layer_idx + self.config.recompute_num_layers, ...)
        chunk_runner(layer_idx, chunk_end, True)   # use_checkpoint=True
        layer_idx += self.config.recompute_num_layers
```

把 pipeline 上本 rank 的全部层，按 `recompute_num_layers` 一组切块，**每块只 checkpoint 一次输入激活**，块内各层正常前向、正常存激活。`recompute_num_layers=1` 就是最极端的「每层都 checkpoint」。`recompute_num_layers` 的含义见 `transformer_config.py:620-624`。

### 3.2 block：只重算前 `recompute_num_layers` 层

```python
elif self.config.recompute_method == 'block':
    recompute_skip_num_layers = 0
    for layer_idx in range(self.num_layers_per_pipeline_rank):
        if (self.config.fp8 or self.config.fp4) and not hidden_states.requires_grad:
            recompute_skip_num_layers += 1
        use_checkpoint = (
            layer_idx >= recompute_skip_num_layers
            and layer_idx < self.config.recompute_num_layers + recompute_skip_num_layers
        )
        chunk_runner(layer_idx, layer_idx + 1, use_checkpoint)
```

语义是：**只有前 `recompute_num_layers` 层做重计算，其余层正常存激活**（`transformer_config.py:615-617` 的 docstring 说得很清楚）。这里有个 FP8/FP4 的坑（`recompute.py:163-166`）：因为 fp8 路径用 Transformer Engine 的 checkpoint（见第 4 节），它要求**重放时输入张量 `requires_grad=True`** 才撑得住 re-entrant autograd。前几层里某些 `hidden_states.requires_grad == False` 的 slot 会被「推过重算窗口」，即把这些层挪到「不重算」的那一段里，用 `recompute_skip_num_layers` 计数补偿。

### 3.3 `chunk_runner`：真正决定「checkpoint 还是直跑」

两条路最终都落到内嵌函数 `chunk_runner(start, end, use_checkpoint)`（`recompute.py:116-145`）：

```python
cf = custom(start, end)
args = (hidden_states, attention_mask, context, context_mask, rotary_pos_emb, padding_mask)
if use_checkpoint:
    if self.config.fp8 or self.config.fp4:
        hidden_states, context = te_checkpoint(cf, self.config.distribute_saved_activations,
            tensor_parallel.random.get_cuda_rng_tracker, self.pg_collection.tp, *args)
    else:
        hidden_states, context = tensor_parallel.checkpoint(
            cf, self.config.distribute_saved_activations, *args)
else:
    hidden_states, context = cf(*args)
```

三条关键信息：

1. `custom(start, end)` 返回一个闭包 `custom_forward`（`recompute.py:54-114`），它遍历 `self.layers[index]`（`start`..`end`）逐层前向——**这就是被 checkpoint 包裹的「可重放片段」**。
2. **精度分流**：FP8/FP4 走 TE 的 `te_checkpoint`，BF16/FP16/FP32 走 `tensor_parallel.checkpoint`。原因在第 4/6 节展开。
3. `distribute_saved_activations`：是否把 checkpoint 存下来的输入激活按 TP 组切分，只留本 rank 那一份（第 7 节）。

## 4. `tensor_parallel.checkpoint`：TP 版的自定义 autograd.Function

BF16/FP16/FP32 路径用 `megatron/core/tensor_parallel/random.py` 里的 `checkpoint`。注释（`random.py:637-640`）点明了它相对 `torch.utils.checkpoint` 的两处改动：

```python
class CheckpointFunction(torch.autograd.Function):
    """... adapted from torch.utils.checkpoint with two main changes:
    1) torch.cuda.set_rng_state is replaced with `_set_cuda_rng_state`
    2) the states in the model parallel tracker are also properly tracked/set/reset.
    """
```

这两处改动正是 TP 场景下的关键，分别对应两个 RNG 系统：

### 4.1 forward：存全量 RNG 状态 + 关梯度跑一遍

`CheckpointFunction.forward`（`random.py:643-674`）：

- `_set_checkpointing()` 置位全局 `IS_CHECKPOINTING`（`random.py:609`），供下游（如 dropout）判断当前是否在 checkpoint 内。
- `ctx.rng_states = _get_all_rng_states()`：**快照所有 RNG 状态**（CUDA RNG + TP 的 tracker）。
- `with torch.no_grad(): outputs = run_function(*args)`：前向在无梯度下跑，只产出输出，不建反向图。
- 若 `distribute_saved_activations`，把第一个输入 `args[0]` 沿 TP 组切成 1D 等分，只留本 rank 段（`random.py:664-668`）。
- `ctx.save_for_backward(*args)` 只存**输入**，中间激活全部丢弃。

### 4.2 backward：回退 RNG → 开梯度重放 → 反向传播

`CheckpointFunction.backward`（`random.py:677-715`）：

```python
with _fork_rng():
    _set_all_rng_states(*ctx.rng_states)      # 回退到前向那一刻的 RNG
    detached_inputs = detach_variable(inputs)
    with torch.enable_grad():
        outputs = ctx.run_function(*detached_inputs)   # 重放前向
...
torch.autograd.backward(outputs, args)         # 对重放的图做反向
grads = tuple(inp.grad ... for inp in detached_inputs)
```

两个要点：

1. **`_fork_rng` + `_set_all_rng_states`**：重放前向里如果有 dropout 等随机算子，必须让它**复现前向时相同的 mask**，否则梯度算错。`_fork_rng` 是一个 contextmanager（`random.py:596-606` 附近），在退出时把 RNG 状态恢复到进入前，避免重放污染训练全局 RNG。
2. **`torch.enable_grad()` 重放**：重放产生的是一张全新的图，再对这张图做 `torch.autograd.backward`，得到输入梯度。

`checkpoint` 包装函数（`random.py:718-730`）里还有一个 CUDA graph 特判：**graph warmup / 捕获期间直接 `return function(*args)` 跳过 checkpoint**——因为重算不能在捕获好的图里运行。

## 5. selective granularity：只重算 attention 核心

上面的 `recompute_method` 回答「重算哪些层」，`recompute_granularity` 回答「层内重算多细」。`transformer_config.py:601-610` 的 docstring 引用了论文 *Reducing Activation Recomputation in Large Transformer Models*（arXiv:2205.05198）：

- **`full`**：整层 checkpoint（`checkpointed_forward` 这整套就是 full 的载体）；
- **`selective`**：只 checkpoint 某个子模块，默认 `["core_attn"]`（attention 里最占显存、却又相对不费算力的部分）；还有 `mlp`、`layernorm`、`moe_act`、`moe`、`shared_experts`、`mla_up_proj`、`mhc` 等可选（`transformer_config.py:629-644` 枚举）；
- **`None`**：不重算。

selective 的实现不在 `recompute.py`，而在 `transformer_block.py` 的层内（`transformer_block.py:312-313` 判断 `recompute_granularity == 'selective'` 且 `"core_attn" in recompute_modules` 时，把 attention 子模块单独包上 checkpoint）。它的动机很经济学：attention 那一步 flops 相对便宜、但中间激活（得分矩阵 $QK^T$）巨大，是「重算性价比最高」的一环。

## 6. fp8/FP4 为什么单独走 `te_checkpoint`

`chunk_runner` 里精度分流（`recompute.py:123-130`）不是随便写的。TE 的 `te_checkpoint`（`megatron/core/extensions/transformer_engine` 里）配合 `activation_recompute_forward`（`random.py:82` 导入的那条）：

- fp8 前向里，TE 的 scale 计算、fp8 cast 都有 side effect（比如 amax 的更新、scale 的取反）。如果用朴素 PyTorch checkpoint 重放，这些 fp8 meta 状态会和反向期望的不一致。
- TE 的 checkpoint 能从 `ctx` 里把 fp8 的 context 也一并保存/恢复（`random.py:800-809` 那段 `fp8: ... activation_recompute_forward(recompute_phase=False)` 的保存逻辑），保证重放时 scale/metadata 一致。

所以这条分流本质是：**「精度无关」用 TP 自己的轻量 checkpoint，「精度相关（fp8/fp4）」交给 TE 管元数据**。

## 7. `distribute_saved_activations`：再把存下来的激活切一刀

checkpoint 已经把「存中间激活」压缩到「只存输入激活」，但**输入激活本身**在大 TP 下也不小（$s \times b \times h$）。`distribute_saved_activations`（`transformer_config.py:626-627`）再补一刀：把 checkpoint 存下的输入激活沿 TP 组切分，每个 rank 只留 $1/p$ 份。

实现就在 `forward`/`backward` 那一对切分/合并函数里：

- forward 存时 `split_tensor_into_1d_equal_chunks(args[0].data, new_buffer=True)`（`random.py:666-668`）；
- backward 取回时 `gather_split_1d_tensor(inputs[0].data).view(ctx.input_0_shape)`（`random.py:690-693`）先拼回完整形状再重放。

一个约束：`distribute_saved_activations` 与 `sequence_parallel` 互斥（`transformer_config.py:2304-2306` 会 assert）——因为 SP 已经把激活沿 seq 切了，两个切分语义会打架。

## 8. 小结

- **重计算 = 用计算换显存**：前向不存中间激活，反向重放前向重算；flops 约 +100%，但显存从「层数 × 中间量」降到「层数 × 输入量」。
- **两个正交维度**：`recompute_method` 决定「重算哪些层」（`uniform` 均匀切块 / `block` 只重算前 N 层）；`recompute_granularity` 决定「层内重算多细」（`full` / `selective` 只重算 `core_attn` / `None`）。
- **实现看 `recompute.py:147-173` 一个 if/elif**，统一收口到 `chunk_runner`，按精度分流到 `tensor_parallel.checkpoint` 或 `te_checkpoint`。
- **`CheckpointFunction`（`random.py:634`）是 PyTorch checkpoint 的 TP 扩展**：多管了一套 TP RNG tracker，保证重放时 dropout mask 与 forward 一致。
- **两处工程细节**：fp8/FP4 走 TE checkpoint 管 fp8 元数据；`distribute_saved_activations` 把存下的激活再沿 TP 切一刀，但与 SP 互斥。

## 9. 下一篇预告

下一篇《随机种子的设置》进入 `random.py` 完整精读——重计算里反复调用的 `_get_all_rng_states` / `_fork_rng`、TP 的 RNG tracker，以及「同一模型副本必须拿到相同 dropout mask」这一整套种子管理是怎么设计的。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/recompute.py`、`megatron/core/tensor_parallel/random.py`、`megatron/core/transformer/transformer_config.py`。）
