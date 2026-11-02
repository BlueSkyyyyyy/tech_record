---
title: "通算并行研读（四）：数据并行——bucket、hook 与 reduce-scatter 的时差"
date: 2026-11-02
draft: false
weight: 5
series: ["comm-overlap"]
tags: ["通算并行", "系列", "数据并行", "DDP", "megatron-lm", "reduce-scatter", "分布式优化器"]
categories: ["训练框架"]
---

数据并行的通信最"老实"：每张卡算完梯度，把所有卡的梯度求平均。它没有流水线那么复杂的依赖，也没有 MoE 的 all-to-all 那么夸张的带宽需求，但它有一个非常固定的暴露点：

> **反向算完了，梯度还没同步完，optimizer 就不能开始。**

一个 70B 模型、BF16、DP=8，梯度总量约 140 GB。哪怕 NVLink 跑到 400 GB/s，一次全量 all-reduce 也要几百毫秒。如果每次都等，训练吞吐会被砍掉一大截。Megatron 的解法是：**把梯度切成 bucket，算完一个 bucket 就发一个，wait 推到 optimizer 之前。**

这一篇拆 Megatron 的 `overlap_grad_reduce` / `overlap_param_gather` 实现，并用 `bucket_sim.py` 算清"桶多大才划算"。

## 1. 三个暴露点

DP 的通信不止一处：

### 1.1 反向的梯度 reduce

标准 DDP：所有参数的梯度算完，一次 all-reduce。暴露 = 整个反向之后的同步时间。

bucket 化：参数按反向到达顺序分成若干桶，桶满就发 reduce-scatter（或 all-reduce），最后一个桶的 wait 留到 optimizer 前。暴露 ≈ 最后一个桶的通信时间 + 前面没被计算盖住的累积。

### 1.2 前向的参数 all-gather

用分布式优化器（ZeRO-1/FSDP 风格）时，每张卡只存 $1/DP$ 的参数分片。前向前必须把整桶参数 all-gather 回来。

- 不重叠：每个 microbatch 前硬等一次 all-gather；
- 重叠：用 forward pre-hook，在当前层计算时提前 gather 下一层。

### 1.3 optimizer 的 all-gather

optimizer step 更新完参数分片后，下一轮前向又要把参数 gather 回来。`overlap_param_gather_with_optimizer_step` 让第一个桶的 all-gather 和 optimizer step 重叠。

## 2. 源码：三件套

Megatron 的实现可以概括为：**hook 登记 → 计数判满 → dispatch/wait 分离**。

### 2.1 开关

`DistributedDataParallelConfig`（`megatron/core/distributed/distributed_data_parallel_config.py`）：

```python
overlap_grad_reduce: bool = False   # :18
overlap_param_gather: bool = False  # :21
delay_wgrad_compute: bool = False   # :202
```

对应 CLI：

- `--overlap-grad-reduce`（`arguments.py:4131`）
- `--overlap-param-gather`（`arguments.py:4194`）
- `--overlap-param-gather-with-optimizer-step`（`arguments.py:4200`）
- 桶大小：新版用 `--ddp-num-buckets` / `--ddp-bucket-size`，对应 `bucket_size` / `num_buckets`（config `:60/:65`）

注意 `--overlap-param-gather` 的约束：只支持分布式优化器 / Megatron-FSDP，且必须和 `--overlap-grad-reduce` 一起开（`arguments.py:1082-1085`）。

### 2.2 反向 hook

DDP 初始化时给每个参数的 `grad_acc` 注册 post hook：

```python
# megatron/core/distributed/distributed_data_parallel.py:436
def enable_forward_pre_hook(self):
    ...
# :468
def _make_forward_pre_hook(self):
    ...
# :500
def _make_backward_post_hook(self, param):
    ...
```

hook 体做三件事：

1. 把 `param.grad` 累进 `param.main_grad`（fp32 主梯度）；
2. 把 `param.grad` 置 None，省显存；
3. 调用 `register_grad_ready`（`distributed_data_parallel.py:525`）。

CUDA graph 捕获期间 hook 直接返回，因为 capture 时不能触发真实通信。

### 2.3 计数判满：为什么不是"参数齐全"

`register_grad_ready`（`param_and_grad_buffer.py:913`）维护一个 per-param 计数：

```python
# 简化
self.per_param_grad_ready_counts[param] += 1
if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
    self.start_grad_sync()
```

`golden_per_param_grad_ready_counts` 是第一个 batch 结束时录下的"每个参数应该到达几次"（`:307`、`:339`）。

为什么用"次数相等"而不是"参数集合齐全"？因为同一个参数可能在一个 batch 里被前向图经过多次：

- tied embedding（输入/输出共享权重）；
- weight sharing；
- 某些 MoE 结构里的共享专家。

如果只检查"参数是否出现过"，会在第一次到达时就误判整桶就绪，导致梯度不完整。**这是读这段代码最容易错的地方。**

### 2.4 dispatch 与 wait 分离

```python
# megatron/core/distributed/param_and_grad_buffer.py:651
def start_grad_sync(self, force_all_reduce=False):
    ...
    if self.ddp_config.overlap_grad_reduce:
        handle = torch.distributed.all_reduce(
            self.grad_data, op=..., async_op=True
        )
    ...

# :833
def finish_grad_sync(self, force_all_reduce=False):
    if self.grad_reduce_finished:   # :856 幂等
        return
    ...
    self.grad_reduce_finished = True  # :868/:878
```

- `start_grad_sync` 只发 `async_op=True` 的通信，不 wait；
- `finish_grad_sync` 在真正需要梯度时（optimizer 前）才 wait；
- `grad_reduce_finished` 保证幂等：后继 bucket group 可能在 dispatch 时提前 drain 前驱，末尾 finalize 又会再调一遍，不能重复 wait。

`previous_grad_reduce_bucket_group` 只在 `reduce_scatter_with_fp32_accumulation` 路径下串联，用来保证前一个 bucket 的 reduce 完成后，后一个 bucket 才能安全复用 buffer。

### 2.5 参数 all-gather 的 hook

前向 pre-hook（`:468`）在模块执行前触发 `start_param_sync`（`:562`）：

```python
def start_param_sync(self, *unused, force_sync=False, force_dispatch=False):
    ...
```

它把当前 bucket 的 all-gather 发出去，但不 wait；真正 wait 在 `finish_param_sync`（`param_and_grad_buffer.py:611`）。`skip_next_bucket_dispatch` 用来避免重复 dispatch。

### 2.6 delay_wgrad_compute

`delay_wgrad_compute`（config `:202`）把 wgrad 计算本身往后挪，让 wgrad 更靠近它的 reduce。它只在 `transformer_impl=transformer_engine` 下可用，且要求 TE ≥ 2.8.0（`arguments.py:2099-2106`）。

直觉：wgrad 算得越晚，它到 reduce 的"在途时间"越短，越容易被后面的计算盖住。但也不能太晚，否则 reduce 会拖到 optimizer 之后。

## 3. 桶多大才划算

`bucket_sim.py` 的模型：

- $N$ 个桶，单桶计算 $c$，单桶通信 $t$（含 launch）；
- 第 $i$ 个桶在 $(i+1)c$ 就绪；
- 单通道 FIFO，通信完成时间递推；
- 暴露闭式：

$$
T_{\text{exposed}} = t + (N-1)\max(0, t-c)
$$

固定总计算 100、总通信 60、每桶 launch 2：

```text
 N     c      t   makespan   exposed
 1 100.00  62.00    162.00     62.00
 4  25.00  17.00    117.00     17.00
 8  12.50   9.50    109.50      9.50
16   6.25   5.75    105.75      5.75
32   3.12   3.88    127.12     27.12
64   1.56   2.94    189.56     89.56
```

三个区间：

- **$N$ 太小**（1–4）：最后一个桶的通信直接暴露，$t$ 很大；
- **$N$ 刚好**（8–16）：$t < c$，尾桶被计算盖住，暴露趋近 $t$；
- **$N$ 太大**（32–64）：launch 把 $t$ 推过 $c$，通信开始堆积，暴露反而上升。

最优桶数由"尾延迟 vs launch/带宽"共同决定。真实系统里还要加两个因素：

1. **NCCL 带宽利用率**：桶太小，消息太小，ring/tree 算法打不满带宽；Megatron 的 `pad_buckets_for_high_nccl_busbw`（config `:70`）就是为此；
2. **reduce-scatter vs all-reduce**：分布式优化器用 reduce-scatter（只传 $1/DP$ 的梯度），通信量比 all-reduce 小，但多一次 all-gather。

## 4. reduce-scatter + all-gather：分布式优化器的账

标准 DDP：

$$
\text{comm} = 2 \cdot \frac{DP-1}{DP} \cdot G \quad \text{(all-reduce)}
$$

分布式优化器（ZeRO-1）：

$$
\text{comm} = \frac{DP-1}{DP} \cdot G \quad \text{(reduce-scatter)} + \frac{DP-1}{DP} \cdot P \quad \text{(all-gather)}
$$

其中 $G$ 是梯度量，$P$ 是参数量。因为 $G \approx P$（BF16 下梯度也是 2 字节/参数），总量和 all-reduce 差不多，但**方向不同**：

- reduce-scatter 在反向，和计算同向，容易重叠；
- all-gather 在前向，和计算反向，需要提前预取。

Megatron 的 `overlap_param_gather` 就是给 all-gather 做预取：在当前层计算时，提前 gather 下一层。`overlap_param_gather_with_optimizer_step` 则把第一个桶的 all-gather 和 optimizer step 重叠，进一步缩短前向开始前的空窗。

## 5. 工程实践：什么时候开、怎么调

### 5.1 先开 grad reduce，再开 param gather

- `--overlap-grad-reduce` 的收益通常更稳：反向计算量大，桶容易藏住；
- `--overlap-param-gather` 依赖分布式优化器，且对桶大小/顺序更敏感；先确认 `--overlap-grad-reduce` 已经有效。

### 5.2 桶大小：从"一个 Transformer 层"开始

经验起点：

- 桶太小（< 1 MB）：launch 和 NCCL 延迟占主导；
- 桶太大（> 100 MB）：尾延迟暴露，且显存峰值上升；
- 一个合理的起点是**一个 Transformer 层的参数量**（例如 hidden=4096、FFN=16384 时约 100M 参数，BF16 约 200 MB——偏大，通常要再切）。

新版 Megatron 用 `--ddp-num-buckets` 直接指定桶数，比自己算 `bucket_size` 更直观。可以先用 `bucket_sim.py` 估一个范围，再在真实 trace 上扫。

### 5.3 不要忽略 launch 开销

每个桶一次 NCCL launch，GPU 侧约 2–5 μs；CPU 侧如果 bucket 太多，还可能被 Python/hook 开销卡住。**桶数不是越多越好**，`bucket_sim.py` 的凹陷就是这件事的定量版。

### 5.4 和 PP/TP 的交互

- PP 的 stage 间 P2P 和 DP 的 reduce-scatter 共享网络，可能互相争抢；
- TP 的 all-reduce 在层内，和 DP 的 reduce-scatter 在不同时间窗口，但会争 SM；
- 如果同时开 `overlap_grad_reduce` 和 `overlap_p2p_comm`，要确认通信流/优先级不会互相阻塞。

## 6. 一个容易忽略的点：梯度累积

梯度累积（gradient accumulation）下，`register_grad_ready` 的计数是按 microbatch 累加的。`is_last_microbatch` 才触发 reduce，所以：

- 累积步数越多，桶就绪越晚，但每个桶的通信量不变；
- 如果最后一个 microbatch 的桶很少，尾延迟会更明显；
- `num_microbatches` 和桶大小的乘积决定了在途通信量，太大可能打满网络。

## 结论

- DP 的三个暴露点：反向梯度 reduce、前向参数 all-gather、optimizer 后的 all-gather。
- Megatron 的三件套：hook 登记 → per-param 计数判满（处理 tied weights）→ `start_*` 发异步、`finish_*` 才 wait。
- 桶不是越细越好：$T_{\text{exposed}} = t + (N-1)\max(0,t-c)$，最优在 $t \approx c$ 附近；launch 和 NCCL 小消息效率决定了下界。
- 分布式优化器把 all-reduce 拆成 reduce-scatter + all-gather，方向不同，重叠策略也不同。
- 先开 `--overlap-grad-reduce`，再开 `--overlap-param-gather`；桶大小从一层参数量的量级开始扫。

下一篇进入张量并行：TP 的 all-reduce 在层内、和 GEMM 紧密耦合，不能像 DP 那样简单挪到 side stream。Flux 的做法是把通信拆到 tile 级、融进 GEMM kernel——这也是"切"和"融"结合得最漂亮的一个案例。

---

**系列导航**：[三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}}) · [五：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}})

> 配套代码：`code/comm-overlap/bucket_sim.py`、`overlap_model.py`
> 自测：`python3 bucket_sim.py --test && python3 overlap_model.py --test`
