---
title: "Megatron 源码精读（十一）：优化器"
date: 2026-09-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "优化器", "数据并行"]
categories: ["训练框架"]
weight: 11
series: ["megatron-code"]
---

上一篇[《checkpoint 处理》]({{< relref "megatron-code-10-checkpoint" >}})留了个悬念：分布式优化器的 state 是「每个 DP rank 只存一段」，那 `sharded_state_dict` 又是怎么生成惰性张量的。

本篇精读 `DistributedOptimizer`（`megatron/core/optimizer/distrib_optimizer.py`，3283 行）。**一句话结论：它做的是「索引切分」——模型参数本身没被切开，被切开的是优化器看到的那份 view，以及与之配套的 state。**

先把这个层次分清，后面所有几何问题才有意义。

---

## 1. 问题：切的是 state，但索引必须先对齐

Adam 的 state 是参数体积的 2~3 倍：fp32 master、一阶矩、二阶矩。DP=8 就是 8 份完整冗余。ZeRO-1 的办法是让每个 rank 只对「参数的一段」负责，于是它的 state 也只剩 1/dp。

注意这不是把参数张量切碎分发。`param.data` 仍是完整参数，优化器拿到的是 `model_param.view(-1)[start:end]` 这样一个 view。所以真正要解的是几何问题：**在一条连续 buffer 上，某个参数落在哪个 rank 的第几段到第几段？**

---

## 2. `Range`：整个切分只有一个数据结构

`Range`（`distrib_optimizer.py:78`）只有三个字段，加一个平移方法：

```python
def __init__(self, start: int, end: int):
    self.start = start
    self.end = end
    self.size = end - start

def normalize(self, start: int = 0):
    return Range(start, start + self.size)   # 换坐标系，长度不变
```

整篇文件几乎不出现显式张量切分，到处是 `Range` 的构造与 `normalize`。真正的切片留到最后一句 `view(-1)[start:end]`——**先把区间算对，再动手切**，这是它读起来清爽的原因。

---

## 3. 四个 range：三个坐标系的对齐结果

`_build_model_gbuf_param_range_map`（`distrib_optimizer.py:134`）为每个参数算四个区间（`:190`）：

- `gbuf_world`：在整条 grad buffer 里的全局区间，用来拼 reduce-scatter / all-gather 的偏移；
- `gbuf_world_in_bucket`：换到 bucket 坐标，用来在单个 bucket 内定位；
- `gbuf_local`：在本 rank 那段里的相对区间，用来索引本地 buffer；
- `param`：在参数自己内部的区间，这才是切模型参数的刀口。

核心是求交：

$$
\begin{aligned}
s_{\text{local}} &= \max(0,\; s_{\text{world}} - b) \\
e_{\text{local}} &= \min(n,\; e_{\text{world}} - b)
\end{aligned}
$$

其中 $b$ 是本 rank 区间的起点、$n$ 是它的长度。**参数横跨两个 rank 时会被切成两段，各归一边**——docstring 明说这个划分「不尊重参数边界」，所以每个 range 指向的往往是参数的**一部分**。

那「本 rank 拥有哪段」怎么定？`_build_model_gbuf_range` 里断言 bucket 的 `grad_data.numel()` 能被 dp 整除（`:217`），然后 rank $r$ 拥有 $[r \cdot m, (r+1) \cdot m)$，$m$ 为整除后的商。这条断言和第 8 篇 DistOpt 的整除约束是同一条，两边咬合。

用的是 pad 之后的 `grad_data.numel()`，所以 pad 出来的尾巴必然落在最后一个 rank 的区间里，那段区间没有参数与之对应——最后一个 rank 的 shard 会「空」一点。

---

## 4. shard 就是一次 view 切片

`_build_model_and_main_param_groups`（`:355`）把它落地：

```python
shard_model_param = model_param.detach().view(-1)[
    param_range.start : param_range.end
]
tensor_parallel.copy_tensor_model_parallel_attributes(shard_model_param, model_param)
```

两件事值得记住：

- 它是 **view 而非副本**，写它就等于写原参数对应位置；
- TP 元数据和量化标记要用 `copy_*` 显式带过去，否则这片 view 在后续逻辑里会「失忆」。

MXFP8、NVFP4、grouped 量化参数不支持 `view(-1)`，走的是存储重映射的另一条路，此时 `shard_model_param` 直接置 `None`。

接着为每个 shard 配一份 fp32 主副本（main param），喂给标准 `torch.optim`。**优化器自始至终只看见一堆 shard**，所以它的 state 自动缩到 1/dp——这就是 ZeRO-1 省显存的落地点，也是全篇最容易被误读成「参数被切了」的地方。

---

## 5. layout 的三个细节，其中一个常被写错

排布由 `_compute_per_buffer_param_layout`（`:511`）决定：

1. **反向遍历**：`for param in params[::-1]`，按反传顺序排，让 reduce-scatter 的发射顺序大致贴合梯度产生顺序；
2. **起点 64 元素对齐**：每个参数前调 `pad_param_start`。注意是 64 个**元素**，不是 64 字节（`param_layout.py` 的 docstring 写得很明确）——bf16 下这恰好是 128 字节，fp32 下是 256 字节，写成「64 字节」就错了；
3. **bucket 尾 pad 到 `lcm(dp, 128)`**：`pad_bucket_end` 的除数不是 dp，而是 dp 与 128 的最小公倍数（开启高 NCCL 带宽优化时再乘上 $2^{16}$）。整除断言之所以成立，就是这里 pad 出来的。

另外 shared embedding 会被强制拆进新 bucket，避免共享参数在切分时被拆散。`compute_full_param_layout`（`:595`）再按 `(param_dtype, grad_dtype, is_expert_parallel)` 分组，expert 组用 expert DP 的 world size 做 pad。

---

## 6. `sharded_state_dict`：不是六选一，是 4 条分派

`sharded_state_dict`（`:1514`）按 `metadata['distrib_optim_sharding_type']` 分派，**默认值是 `fully_sharded_model_space`**。真正走分支的只有四种：

| 格式 | 形态 | 备注 |
|---|---|---|
| `fully_sharded_model_space` | 每参数一个 `ShardedTensor`，canonical state 的扁平子集 | 默认，可全并行存取 |
| `fully_reshardable` | 保存时在 DP rank 0 gather 成 canonical 表示 | 可全并行，但要 gather |
| `dp_reshardable` | 每个不连续 buffer 一个 `ShardedTensor` | 只能在 DP 维 reshard |
| `dp_zero_gather_scatter` | 复用 legacy gather/scatter，存 `ShardedObject` | 已 deprecated |

`fsdp_dtensor`（每参数一个 `DTensor`，Megatron-FSDP 用）在函数开头就单独 return，不进这四分支，读侧也不经过 `load_state_dict` 的那四条。

还有一个隐蔽的分界：`checkpoint_fully_reshardable_formats` 只包含 `fully_reshardable`、`fully_sharded_model_space`、`fsdp_dtensor` 三个。不在此列的格式，`state_dict` 会被包成 `ShardedObject`，「DP rank 0 存、其余 rank 空」。

这两种「全并行」格式的取舍很直白：`fully_reshardable` 存的时候要在 rank 0 收齐所有 buffer、再拆成每参数一份 canonical state，加载时每个 rank 读一份超集、自己 flatten 再切片；`dp_reshardable` 一个通信都不用，代价是格式绑定在 DistOpt 的内部 buffer 结构上，只能沿 DP 维重新分片。

---

## 7. 小结

- **切分的是索引不是张量**：参数完整，优化器只持有 `view(-1)[start:end]` 的 shard，state 因此缩到 1/dp。
- **`Range` 是全部几何**：start/end/size 加 `normalize`；四个 range（world / bucket / local / param）对齐三个坐标系，求交式是 `max(0, s-b)` 与 `min(n, e-b)`。
- **整除是硬前提**：bucket 元素数必须被 dp 整除，rank $r$ 拥有第 $r$ 段。
- **64 元素对齐**：不是 64 字节；bucket 尾 pad 到 `lcm(dp, 128)`。
- **四种 sharding 分派 + 一个特例**：默认 `fully_sharded_model_space`，`fsdp_dtensor` 单独走。

下一篇讲 [fused 算子]({{< relref "megatron-code-12-fused-kernels" >}})：MCore 为什么要在 softmax、layernorm、rotary 这些地方跟编译器抢活干。

---

**版本提示（2026-10）**：上游优化器侧的增量集中在 Muon——per-head QKV 正交化（#7085）、`LayerShardedMuon` 的流式 Newton–Schulz（#8379），以及 chained optimizer 与 MFSDP v2 对 Muon 的支持。本文梳理的 Adam 路径未变。

**行号基线**：commit f713506ce（2026-08-31）。上游 dev 已前进到 2026-10-29（+1208 commits），行号会漂移，定位请以符号名为准。
