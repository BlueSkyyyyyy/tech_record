---
title: "Megatron 源码精读（五）：随机种子的设置"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "随机种子"]
categories: ["训练框架"]
weight: 5
series: ["megatron-code"]
---

并行训练里的「随机种子」为什么不是一个数，而是一张表？

结论：**同一个 seed 在不同并行维度上的要求正好相反——DP 副本要同一组随机数，TP 组内要各不相同的。**

`CudaRNGStatesTracker` 就是把这两套相反要求做成「命名态 + fork」的一张表。

上一篇[激活重计算]({{< relref "megatron-code-04-recompute" >}})里 `CheckpointFunction.backward` 反复调用的 `_get_all_rng_states` / `_fork_rng` 都出自 `megatron/core/tensor_parallel/random.py`。本篇只讲这个文件里的种子与状态管理。

## 1. 问题：一份 seed 要同时满足四个相反的要求

- **DP 副本要一致**：同一 pipeline stage 的参数是复制关系，初值必须相同，否则 all-reduce 的是几套不同模型的梯度。
- **TP 组内要不同**：同一张权重被切成 tp 块，各 rank 只初始化自己那块；同一个 seed 会让每块生成同一组数。
- **重算要能回退**：上一篇的重放必须精确回到前向那一刻的随机状态。
- **CUDA graph 捕获时**，host 侧读 RNG 状态不是 capture-safe 的。

单卡一个 `torch.manual_seed` 解决不了这四条。Megatron 的答案是「一组命名状态槽 + fork」，而不是「一个全局 seed」。

## 2. `CudaRNGStatesTracker`：一张「名字 → 状态」的表

类在 `random.py:216-333`，内部只有三个字段（`reset`，`random.py:242-256`）。

`states_` 是 `name -> cuda rng state`；`seeds_` 纯记账，防止同一个 seed 被 add 两次；`_current_state_name` 记录 generator 当前用的是哪股流，默认 `"default-rng"` 且不入表。

### 2.1 `add`：从 seed 造出一个命名态

两种实现殊途同归，都是「造出 seed 对应的状态并存进表」（`random.py:272-295`）：

```python
if self.use_cudagraphable_rng:
    new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
    new_state.manual_seed(seed)
    self.states_[name] = new_state
else:
    orig_rng_state = torch.cuda.get_rng_state()
    torch.cuda.manual_seed(seed)
    self.states_[name] = torch.cuda.get_rng_state()
    _set_cuda_rng_state(orig_rng_state)
```

区别只在 graph-safe 路径克隆一个 generator 出来单独播种，**完全不碰当前 generator**；普通路径则要「临时改、读完、改回」。

### 2.2 `fork`：进入切流，退出时把状态存回去

这是全篇最核心的 contextmanager（`random.py:297-333`）。进入时先把当前状态写回它所属的命名态（default 除外），再切到 `states_[name]`；退出时把 `states_[name]` 更新成最新值，然后还原进入前的状态。

所以每个命名态都是一条**有记忆的流**：多次 `fork` 之间连续推进，而不是每次从种子重画。这正是 dropout 在多层之间不重复、同时重放又能精确复现的原因。

`finally` 里还有两道防御：CPU RNG 被改动会告警（`random.py:319-320`）；`_current_state_name` 与 `name` 对不上直接抛异常，防止嵌套 fork 错配。

## 3. 三个命名态与种子算式

入口是 `model_parallel_cuda_manual_seed`（`random.py:433-484`），算式只有四行：

```python
offset = seed + 2718                       # 2718 只是随手取的正数
tensor_model_parallel_seed = offset + tp_rank
data_parallel_seed = seed
expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
```

随后 `torch.cuda.manual_seed(data_parallel_seed)`，再依次 `add` 三个态：data-parallel、model-parallel、expert-parallel（`random.py:477-484`）。

`manual_seed` 在 `add` 之前，所以 generator 的 live 状态停在 DP 态——块外不在任何 `fork` 里的随机算子，吃的就是它。

开头还有一句 `_CUDA_RNG_STATE_TRACKER.reset()`，把 `seeds_` 清空；否则 `add` 会因为「seed 已存在」抛异常。也正因如此，重复调用这个函数是幂等的。

### 3.1 为什么 DP 必须相同、TP 必须不同

权重初始化是最硬的证据。`_initialize_affine_weight_gpu` 在 `get_cuda_rng_tracker().fork()` 里对本 rank 的那块权重跑 init（`layers.py:151-155`）：

- DP 副本是同一份参数的复制，种子必须一致，否则副本从不同初值出发，聚合出来的东西没有意义；
- TP 分片是同一张矩阵的不同块，种子若相同、形状又相同，每块会生成**同一组数**，整张矩阵退化成 tp 份周期重复。

dropout 同理：TP 区域里 mask 是按 head 或按 seq 切开的，共用状态会让 mask 以分片长度为周期重复，噪声在分片间完全相关，等于把有效噪声自由度除以 tp。

反过来，**复制张量**的随机量必须各 rank 相同——`RandomSTEShared` 要一个全 rank 一致的随机 bias，就显式 fork 到 DP 态（`moe_utils.py:1305-1306`）。

代码里 fork 的位置正好对着这条判据：不开 SP 时 block 内不 fork，留在 DP 态；开了 SP 才把整个 block forward 包进 `fork()`（`transformer_block.py:928-932`），因为激活已沿 seq 切开。

attention dropout 则在无 SP 时单独 fork 到 MP 态、有 SP 时直接调用（`dot_product_attention.py:235-239`）。

### 3.2 真正的 seed 在进 tracker 之前已经被改过

训练入口先做一次派生（`initialize.py:424-429`）：`seed = seed_ + 100 * pp_rank`，只有开 `--data-parallel-random-init` 才再加 `10 * dp_rank`。

所以「DP 副本种子相同」不是 tracker 给的，是入口根本没给 DP 偏移；PP stage 之间反而天然不同，因为它们算的是不同的层。

docstring 那句「default 态在同一组模型并行 GPU 上相同、在不同模型并行组上不同」，说的就是这个 pp 偏移。

## 4. `initialize_rng_tracker`：Megatron 版、TE 版，和推理的 no-op

三条分支（`random.py:341-408`）：装了 TE 且 `use_te_rng_tracker=True` 时用 `TECudaRNGStatesTracker`，但先卡版本，`is_te_min_version("1.5.0")` 不满足直接 `RuntimeError`。

没装 TE 则**静默回退**到自带实现，不报错；自带实现可以带 `use_cudagraphable_rng`。

`inference_rng_tracker=True` 时再套一个匿名子类，把 `add` / `set_states` 改成 `pass`、`fork` 返回 `nullcontext()`（`random.py:376-393`）——推理用不到这些状态，直接省掉开销。

## 5. `_get_all_rng_states`：为什么要分 `cloned` 和 `live`

graph-safe tracker 的 `get_states` 返回的是**共享底层 generator 的句柄**，等重放时 generator 早已前进，拿句柄恢复等于 no-op；而捕获期间 host 侧读状态又不安全，只能保留句柄。

于是快照带一个显式标签（`random.py:499-543`）：graph-safe 且不在捕获中时 `kind="cloned"`，逐项 `clone_state()` 拷内容；否则 `kind="live"` 直接拿句柄。

恢复端按标签走两条路（`random.py:546-594`）：`cloned` 就地 `set_state` 写回 live generator，并校验名字集合没变；`live` 则整表替换。

恢复时不靠 `is_current_stream_capturing()` 反推类型，因为快照和恢复可能落在捕获边界两侧。

## 6. `fork` 与 `_fork_rng` 是两个层次

`fork` 在命名态之间切换，管「我用哪股流」；`_fork_rng`（`random.py:597-606`）是整表快照 + 退出还原，管「能不能回到这之前」，`CheckpointFunction.backward` 就用它把重放整段包住。

## 7. 小结

- 种子是一张表：`states_` / `seeds_` / `_current_state_name` 三个字段撑起全部语义。
- 算式只有四行：`offset = seed + 2718`、`tp = offset + tp_rank`、`dp = seed`、`ep = seed + 1024 + 100*ep_rank + etp_rank`。
- TP 组内不同是为了让同一权重的各分片、同一激活的各切片画出不同的随机数；DP 相同是因为它们是复制关系。
- `fork` 退出时把最新状态存回命名态，所以每条流有记忆；`_fork_rng` 才是整表回退。
- `_get_all_rng_states` 用 `cloned`/`live` 双通道，保证重算回退在 CUDA graph 下不失效。

下一篇：[与 Transformer Engine 的关系]({{< relref "megatron-code-06-transformer-engine" >}})接上 `TECudaRNGStatesTracker` 与 `te_checkpoint`。

---

**版本提示（2026-10）**：上游新增了 GTP 的两个权重初始化 tracker：`gtp_remat` 按 `tensor_model_parallel_seed + 65536 * (1 + gtp_remat_rank)` 取值、`egtp_remat` 按 `expert_parallel_seed + 32768 + 65536 * (1 + egtp_remat_rank)`，且只在对应 `gtp_remat_world_size > 1` 时注册。这样做是因为 GTP_remat 各分片必须初始化为不同值，而其余三个种子的语义要保持不变。非 GTP 运行的 tracker 集合与基线完全一致，本篇结论不受影响。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已到 2026-10-29，行号会漂移，定位请以符号名为准。
