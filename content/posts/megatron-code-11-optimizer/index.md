---
title: "Megatron 源码精读（十一）：优化器"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "优化器", "数据并行"]
categories: ["训练框架"]
weight: 11
series: ["megatron-code"]
---

上一篇[《checkpoint 处理》]({{< relref "megatron-code-10-checkpoint" >}})结尾卖了个关子：分布式优化器的 state 是「每个 DP rank 只 hold 一段参数切片」，还说「`sharded_state_dict` 怎么产生惰性张量」下一篇讲。本篇就兑现这个承诺，精读 MCore 的分布式优化器 `DistributedOptimizer`（`megatron/core/optimizer/distrib_optimizer.py`，3283 行）。

它解决的核心问题一句话：**Adam 的优化器 state（一阶矩、二阶矩、fp32 master weight）是参数体积的数倍，朴素 DDP 里每个 rank 各存一份完整的，显存爆炸；能不能像切参数一样，把「优化器 state」也按 DP 维切碎，每个 rank 只维护自己那一小段？** 答案是能，而答案的关键是一个特别朴素的数据结构——`Range`。

---

## 1. 问题背景：ZeRO-1 想省掉的那份「数倍冗余」

先回顾第 8 篇《ZeRO-1 / FSDP 实现》的结论：`use_distributed_optimizer=true` 时，梯度用 reduce-scatter 聚合、参数副本被销毁。但那时没展开「优化器 state 本身怎么切」。本篇补上。

优化器 state 的冗余来自 Adam 这类自适应优化器：每个参数除了 `fp32 master weight`（1×），还有一阶矩 `exp_avg`（1×）、二阶矩 `exp_avg_sq`（1×），总计 2~3 倍参数体积（fp32 下更多）。朴素 DDP 里每个 DP rank 都算一遍、存一遍，DP=8 就是 8 份冗余。ZeRO-1 的做法是：**把这 2~3 倍的 state，连同参数本身，按 DP 维切成 1/dp 份，每个 rank 只存自己那一份的量级**。

于是整个问题坍缩成一个几何问题：**在一个连续的内存 buffer 上，怎么精确地算出「哪个 rank 拥有哪一段区间」，以及「某个参数落在哪个 rank 的哪一段」。** MCore 用一个 `Range` 类加几层 range map 把它解了出来。

---

## 2. 现象：一切靠 `Range` 这个区间对象

`distrib_optimizer.py` 一开篇就定义了 `Range`（`distrib_optimizer.py:78-110`）：

```python
# megatron/core/optimizer/distrib_optimizer.py:88-101
def __init__(self, start: int, end: int):
    self.start = start
    self.end = end
    self.size = end - start

def normalize(self, start: int = 0):
    return Range(start, start + self.size)
```

就三个字段：start / end / size。它唯一重要的方法 `normalize` 把区间「平移到新的起点、保持长度不变」。别小看这十行——**整个分布式优化器的切分逻辑，就是把「参数、梯度 buffer、优化器 state」三者的索引关系，全部表达成 `Range` 之间的区间运算**（交集、平移、换算到不同坐标系）。

「现象」层要抓住的一点：这个文件里几乎没有出现「显式的张量切分」，到处是 `Range` 的构造和 `normalize`。真正的张量切片，只是最后 `param.view(-1)[param_range.start : param_range.end]` 这一下。

---

## 3. 根因：三个坐标系，靠区间运算对齐

`_build_model_gbuf_param_range_map`（`distrib_optimizer.py:134-197`）是切分的核心。docstring 说得很直白：grad buffer 被概念性地分成 `dp_world_size` 个连续 region，每个 DP rank 「拥有」一段，负责 reduce 这段梯度、更新这段参数。它给每个参数算出**四个 range**：

```python
# megatron/core/optimizer/distrib_optimizer.py:190-194
param_range_map[param] = {
    "gbuf_world": param_world_range,          # 参数在「全局 grad buffer」里的区间
    "gbuf_world_in_bucket": param_world_range_in_bucket,  # 在 bucket 内坐标
    "gbuf_local": param_local_range,          # 参数在本 rank 本地 buffer 里的区间
    "param": sub_param_range,                 # 参数在「自己」内部的区间（即它的 shard）
}
```

这四个 range 就是「三个坐标系」的对齐结果：

- **world 坐标**：整个 DP 组共享的全局 grad buffer 索引；
- **local 坐标**：本 rank 拥有的那段（`gbuf_world_range`）里的相对索引；
- **param 坐标**：某个参数自己内部的索引（0 到 `param.numel()`）。

关键的三行是 `param_local_range` 的计算（`distrib_optimizer.py:176-177`）：

```python
param_local_start = max(0, param_world_start - gbuf_world_range.start)
param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)
```

这就是「参数的全局区间 ∩ 本 rank 拥有的区间」的求交，取 `[max(0, 参数起点-我区间起点), min(我区间长度, 参数终点-我区间起点)]`。**一个参数如果横跨两个 rank 的边界，会被切成两段，分别归两个 rank**——这正是「切分不尊重参数边界」的含义（docstring 明确说了这一点，`distrib_optimizer.py:151-156`）。

那「每个 rank 拥有哪一段」是怎么定的？在 `_build_model_gbuf_range`（`distrib_optimizer.py:200-244`）：

```python
# megatron/core/optimizer/distrib_optimizer.py:216-234（节选）
gbuf_size = bucket.grad_data.numel()
assert gbuf_size % data_parallel_world_size == 0
max_gbuf_range_size = gbuf_size // data_parallel_world_size
...
for r in range(data_parallel_world_size):
    gbuf_world_start = r * max_gbuf_range_size
    gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
    gbuf_world_range = Range(gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset)
    gbuf_world_all_ranges.append(gbuf_world_range)
gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]
```

**第 216-218 行的断言是整个切分的前提**：每个 bucket 的 buffer 元素数必须能被 DP world size 整除。然后 rank r 拥有 `[r*size, (r+1)*size)` 这段连续区间（`size = gbuf_size / world_size`）。这跟第 8 篇里 `param_and_grad_buffer.py` 那个「开 DistOpt 后 buffer 元素数必须被 dp_world_size 整除」的断言是同一条约束，两边咬合。

---

## 4. 解法一：shard 参数就是一次 `view(-1)` 切片

有了 range map，真正「切出」参数 shard 和 main param 的代码在 `_build_model_and_main_param_groups`（`distrib_optimizer.py:355`）。核心一行：

```python
# megatron/core/optimizer/distrib_optimizer.py:420-422（节选）
shard_model_param = model_param.detach().view(-1)[
    param_range.start : param_range.end
]
```

`param_range` 就是上面 `param_range_map[model_param]["param"]`。所以：

- **shard_model_param**：`model_param.view(-1)[start:end]` —— 是原参数的一个「view 切片」，不是副本。它就是这个 rank 负责更新的那一小段；
- **main param**（fp32）：shard 对应的 fp32 主副本，优化器实际 step 的对象。

这里有个漂亮的细节：**每个 rank 上 shard 参数的 `requires_grad`、张量属性（TP 元数据、量化标记）都被 `copy_*` 显式带过来**（`distrib_optimizer.py:423-426`），所以后续 `view(-1)` 切片出来的 shard 能无缝参与 mc 的其余逻辑。

而优化器本身（Adam 等）调用的仍是标准 `torch.optim` —— 只不过喂给它的 `param_groups` 不再是「完整参数」，而是「本 rank 拥有的一堆 shard」。于是每个 rank 上的优化器 state 体积自动缩水为 `1/dp`。这就是 ZeRO-1 省显存的落地点。

---

## 5. 解法二：param layout，「反向遍历 + pad 对齐」

切分前必须先确定参数在连续 buffer 里的排布，这就是 `_compute_per_buffer_param_layout`（`distrib_optimizer.py:511-592`）。它有三个值得记住的点：

1. **反向遍历参数**（`distrib_optimizer.py:557` 的 `for param in params[::-1]`）：按 backprop 的反向顺序排布，让 reduce-scatter 的通信顺序大致对齐梯度产生的顺序；
2. **64-byte 对齐**与 **bucket 结尾 pad 到 DP 整除**：`_finalize_bucket` 调 `pad_bucket_end(param_end_index, data_parallel_world_size, ...)`（`distrib_optimizer.py:547-553`）——这正是 §3 那个「buffer 必须被 dp 整除」断言的来源，pad 就是为了凑整除；
3. **shared embedding 单独拆 bucket**（`distrib_optimizer.py:561-565`）：带 `shared_embedding` 标记的参数（输入/输出 embedding 共享）强制拉开一个新 bucket，避免共享参数在切分时被拆散。

`compute_full_param_layout`（`distrib_optimizer.py:595-639`）再把参数按 `(param_dtype, grad_dtype, is_expert_parallel)` 分组，每组各算一份 layout——**expert 并行的那组用 `expert_data_parallel_world_size` 做 pad**，这是 MoE 场景下 EP 和 DP 切分维度不同的体现。

---

## 6. 解法三：`sharded_state_dict` 的六种切分格式

最后回到第 10 篇的悬念：`sharded_state_dict`（`distrib_optimizer.py:1514-1642`）怎么产生惰性张量。它不像 legacy 那样返回真实张量，而是根据 `metadata['distrib_optim_sharding_type']` 六选一，返回**带 sharding 元数据的惰性引用**：

- `dp_reshardable`：把每个不连续 buffer 拆成一个 `ShardedTensor`，可全并行保存/加载，但只支持 DP 维 reshard（`distrib_optimizer.py:1617-1620`）；
- `fully_reshardable`：保存时在 DP rank 0 gather 所有 buffer 转成「canonical 表示」，每个模型参数对应同形状的 state 张量（`distrib_optimizer.py:1627-1630`）；
- `fsdp_dtensor`：每个参数是一个 PyTorch `DTensor`，Megatron-FSDP 的默认（`distrib_optimizer.py:1580-1583`）;
- 三个 deprecated 格式：`dp_zero_gather_scatter` / `fully_sharded_model_space` / `fs_model_space`（`distrib_optimizer.py:1621-1636`）。

非「fully reshardable」格式走的是「DP rank 0 存、其余 rank 空」的老路子（`distrib_optimizer.py:1595-1610`，用 `ShardedObject` 包装）。这段和 `save_checkpoint` 里 `generate_state_dict` 调 `optimizer.sharded_state_dict(...)`（`checkpointing.py:1330-1331`）严丝合缝地对上了第 10 篇的叙述。

---

## 7. 小结

- **`Range` 是灵魂**：start/end/size + `normalize`，十几行撑起整个切分逻辑（`distrib_optimizer.py:78-110`）。
- **切分 = 区间求交**：四个 range（world / world_in_bucket / local / param）对齐三个坐标系，`param_local` 就是「参数全局区间 ∩ 本 rank 区间」（`distrib_optimizer.py:134-197`）。
- **整除是硬前提**：每个 bucket 的 buffer 元素数必须被 `dp_world_size` 整除，rank r 拥有 `[r*size, (r+1)*size)`（`distrib_optimizer.py:216-234`）。
- **shard 就是一次 view 切片**：`model_param.detach().view(-1)[start:end]`，main param 是它的 fp32 副本（`distrib_optimizer.py:420-422`）。
- **layout 三细节**：反向遍历、bucket 结尾 pad 到 DP 整除、shared embedding 拆 bucket（`distrib_optimizer.py:547-565`）。
- **六种 sharding 格式**：`sharded_state_dict` 按 `distrib_optim_sharding_type` 分派，返回惰性 `ShardedTensor`/`DTensor`，与 checkpoint 篇的 `save_checkpoint` 咬合（`distrib_optimizer.py:1514-1642`）。

下一篇讲 **fused 算子**：MCore 里那些手写的 CUDA/融合 kernel（softmax、layernorm、rotary、moe 聚合等），说到底为什么要跟编译器抢活干、以及它们怎么和 `torch.autocast` 的调度协同，也就是《[fused 算子]({{< relref "megatron-code-12-fused-kernels" >}})》。

（本文所有行号基于 commit `f713506cea2e7705dd2ebb00c5c58a046ff974fe`，对应文件 `megatron/core/optimizer/distrib_optimizer.py`。）
