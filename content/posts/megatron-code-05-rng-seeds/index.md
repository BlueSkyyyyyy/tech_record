---
title: "Megatron 源码精读（五）：随机种子的设置"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "随机种子"]
categories: ["训练框架"]
weight: 5
series: ["megatron-code"]
---

上一篇[《激活重计算》]({{< relref "megatron-code-04-recompute" >}})讲到 `CheckpointFunction.backward` 重放前向时必须复现相同的 dropout mask，反复调用的 `_get_all_rng_states` / `_fork_rng` 都来自 `megatron/core/tensor_parallel/random.py`。本篇把这个文件里**种子与 RNG 状态管理**的部分完整讲清：为什么并行训练里「种子」不是一个数而是一组状态、TP 组内为什么需要不同的种子、`CudaRNGStatesTracker` 怎么用 fork 机制让每个并行域拿到自己的独立随机流。

---

## 1. 问题背景：并行训练里的「随机」比单卡复杂得多

单卡训练时，`torch.manual_seed(seed)` 一个数就搞定。但到了模型并行 + 数据并行，同一个问题被拆成了八块：

1. **数据并行（DP）副本之间要一致**：同一个 batch 在不同的 DP rank 上，前向/反向里 dropout 这些随机算子的 mask 必须一致，否则梯度聚合时对不上。
2. **张量并行（TP）组内要不同**：TP 把一层劈到多张卡上，各 rank 若是同样的 dropout mask，会破坏统计独立性（相当于每张卡拿同一份噪声，「扩容」失去了意义）。
3. **重计算要回退**：上一篇讲的激活重计算，重放前向时 RNG 必须精确回到「前向那一刻」。
4. **MoE 的专家层又是一种切法**：expert 并行的独立随机流也要单独管。
5. **CUDA graph 捕获**时 host 侧读 RNG 状态是 capture-unsafe 的，需要专门处理。

Megatron 的答案是**一个全局的 `CudaRNGStatesTracker` + 一组命名好的状态槽**，用「命名 + fork」代替「一个全局 seed」。本节先看核心数据结构。

## 2. `CudaRNGStatesTracker`：一张「名字 → 状态」的表

`random.py:216-333` 定义了这个类。它内部就三样东西（`random.py:242-256` 的 `reset` 方法）：

- `states_`：dict，`name -> cuda rng state`，保存各个并行域当前的状态；
- `seeds_`：set，纯记账用，防止同一个 seed 被 `add` 两次（`random.py:252`）；
- `_current_state_name`：当前 generator 正在用哪个名字，默认 `"default-rng"`，且这个默认态**不**塞进 `states_`（`random.py:254-256`）。

### 2.1 `add(name, seed)`：从 seed 造出一个命名态

`random.py:272-295`：

```python
def add(self, name, seed):
    self._is_initialized = True
    if seed in self.seeds_:
        raise Exception('seed {} already exists'.format(seed))
    self.seeds_.add(seed)
    if name in self.states_:
        raise Exception('cuda rng state {} already exists'.format(name))
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

两种写法殊途同归：**把 GPU generator 临时设成 `seed`、读出它的状态、存进 `states_[name]`、再把 generator 恢复原状**。区别只在是否走 graph-safe 路径（`clone_state` vs `get_rng_state`）。

### 2.2 `fork(name)`：在一个命名态里做操作、退出即还原

这是整篇最核心的一个 contextmanager（`random.py:297-333`）：

```python
orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
orig_state_name = self._current_state_name
if orig_state_name != "default-rng":
    self.states_[orig_state_name] = orig_cuda_rng_state
_set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
self._current_state_name = name
cpu_rng_state = torch.get_rng_state()
try:
    yield
finally:
    ...
    self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
    _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)
    self._current_state_name = orig_state_name
```

语义是：**进入时把 generator 切到 `states_[name]`，退出时（1）把这个命名态的最新状态存回 `states_[name]`、（2）把 generator 还原成进入前的状态**。这样每次 `with tracker.fork("model-parallel-rng")` 都是一个「有记忆、可续跑」的随机流。

注意 `finally` 里还检查了 CPU RNG 是否被改（`random.py:319-320`），以及 `_current_state_name` 是否还等于 `name`（防止嵌套 fork 错配），防御性很足。

## 3. 三个命名态是种子管理的骨架

`model_parallel_cuda_manual_seed`（`random.py:433-484`）是把 seed 拆成各并行域种子的入口，docstring（`random.py:449-458`）已经把三种状态的含义写得很清楚：

- **default state（data-parallel）**：DP 副本间相同、不同 DP 组之间不同——dropout 在非 TP 区域用它；
- **tensor-model-parallel state**：TP 组内不同、DP 组之间相同——dropout 在 TP 区域用它；
- **expert-parallel seed**：MoE 专家层专用，在 expert-tensor/expert-model 并行的 GPU 间不同。

拆分的算式很直白（`random.py:466-484`）：

```python
offset = seed + 2718                          # 2718 随便取的，任何正数都行
tensor_model_parallel_seed = offset + tp_rank
data_parallel_seed = seed                     # DP 直接用原始 seed
torch.cuda.manual_seed(data_parallel_seed)
_CUDA_RNG_STATE_TRACKER.add(_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)
_CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, tensor_model_parallel_seed)
expert_parallel_seed = seed + 1024 + 100 * ep_rank + etp_rank
_CUDA_RNG_STATE_TRACKER.add(_EXPERT_PARALLEL_RNG_TRACKER_NAME, expert_parallel_seed)
```

核心思想：**TP 的 seed 要带上 `tp_rank` 才能让组内各 rank 不同；DP 共用同一个 seed 才能保证副本一致**；expert 并行则同时带上 `ep_rank` 和 `etp_rank`。

## 4. `initialize_rng_tracker`：Megatron tracker 还是 TE tracker？

`random.py:341-408` 里 `initialize_rng_tracker` 决定用哪套实现：

- 若装了 TE 且 `use_te_rng_tracker=True`（且 TE >= 1.5.0），用 `TECudaRNGStatesTracker`，它天然 cudagraphable、支持 FP8（`random.py:362-368`）；
- 否则用 Megatron 自己的 `CudaRNGStatesTracker`，可传 `use_cudagraphable_rng`（`random.py:369-374`）；
- 若 `inference_rng_tracker=True`，则包一层 `InferenceCudaRNGStatesTracker`，把 `add`/`set_states`/`fork` 全部变成空操作（`random.py:376-393`）——推理根本用不到这些 RNG fork，直接 no-op 提速。

`get_cuda_rng_tracker`（`random.py:401-408`）就是「惰性初始化 + 返回全局单例」的封装，`initialize_rng_tracker` 里用 `_CUDA_RNG_STATE_TRACKER_INITIALIZED` 全局标志避免重复初始化（`random.py:351-358`）。

## 5. `_get_all_rng_states` / `_set_all_rng_states`：快照与回退

上一篇引用的这对函数在 `random.py:499-594`，这里补上它们相对朴素实现的精髓——**graph-safe 与普通模式的对称处理**。

`_get_all_rng_states`（`random.py:499-543`）返回一个四元组 `(cpu_rng_state, cuda_rng_state, cuda_rng_state_tracker, kind)`，其中 `kind` 是关键：在 graph-safe 且**不在 CUDA graph 捕获中**时，用 `clone_state()` clone 出状态**内容**的副本（`kind="cloned"`）；否则直接拿 generator **句柄**（`kind="live"`）。

为什么要分两种？docstring（`random.py:500-529`）讲得很透：graph-safe 的 `get_states` 返回的是**共享同一底层 generator 句柄**，之后 generator 又前进了，`live` 句柄回退等于 no-op；而捕获期间 host 侧读状态 capture-unsafe，又只能保留句柄语义。所以用一个显式的 `cloned`/`live` 标签，把「快照」和「恢复」严格绑在同一类语义上，避免跨 capture 边界时悄悄失效。

对应的 `_set_all_rng_states`（`random.py:546-594`）里，`cloned` 分支是把内容 `set_state` 回 live generator（并检查名字集合没变），`live` 分支则是整表替换。

## 6. 串起来：`_fork_rng` 与 checkpoint 的关系

`_fork_rng`（`random.py:597-606`）就是 `_get_all_rng_states` + `_set_all_rng_states` 包成的 contextmanager：

```python
current_states = _get_all_rng_states()
try:
    yield
finally:
    _set_all_rng_states(*current_states)
```

上一篇的 `CheckpointFunction.backward` 就是用它包住「重放前向」这一段（`random.py:695-702`），保证重放里的 dropout 复现 forward 的 mask，退出后又把训练 RNG 恢复到重放前的状态——**这样重计算既不影响后续层的随机流，又能得到正确的梯度**。

对比一下，`fork`（第 2.2 节）和 `_fork_rng`（本节）是两个层次：`fork` 是在**命名态之间切换**（TP ↔ DP ↔ expert），`_fork_rng` 是**整张状态表的快照/回退**（跨一个重放区段）。一个管「我用哪股随机流」，一个管「我能不能回到这之前」。

## 7. 小结

- **并行训练里种子不是单个数，而是一组命名态**：`CudaRNGStatesTracker` 用 `states_`/`seeds_`/`_current_state_name` 管理「名字 → 状态」的映射。
- **TP 种子带 `tp_rank`、DP 共用 seed、expert 兼顾 `ep_rank`/`etp_rank`**，这是 `model_parallel_cuda_manual_seed`（`random.py:466-484`）一行算式决定的。
- **`fork(name)` 在命名态间切换并在退出时还原**，给每个并行域一条有记忆、可续跑的独立随机流（`random.py:297-333`）。
- **`_get_all_rng_states`/`_set_all_rng_states` 用 `cloned`/`live` 双通道**处理 graph-safe 与普通模式，保证重计算的 RNG 回退在 CUDA graph 下也不翻车。
- **两套 tracker 可选**：TE 的 cudagraphable 版本 vs Megatron 自带实现，由 `initialize_rng_tracker` 按 `use_te_rng_tracker` 分流。

## 8. 下一篇预告

下一篇《与 Transformer Engine 的关系》把本文埋下的伏笔——`te_checkpoint`、`TECudaRNGStatesTracker`、FP8 元数据管理——展开：Megatron 的 transformer 层里哪些模块走 TE、`transformer_engine` 扩展包怎么桥接两边、FP8 的训练路径是怎么接管前向/反向的。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/tensor_parallel/random.py`。）
