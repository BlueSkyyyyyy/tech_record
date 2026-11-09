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

先说清楚：**ZeRO-1 不切参数**，所以它没有这一步。三种常被混为一谈的"省显存"方案要分开看：

| 方案 | 切什么 | 前向要不要 gather 参数 | 粒度 |
|---|---|---|---|
| ZeRO-1 | 只切优化器状态（fp32 master、Adam $m/v$） | **不要**（参数全复制） | — |
| ZeRO-2 | + 梯度切分（reduce-scatter 后不再 gather 梯度） | **不要** | — |
| ZeRO-3 / FSDP | + 参数切分 | 要 | 每层、每 microbatch，用完即弃 |
| Megatron 分布式优化器 | fp32 主权重 + 优化器状态；bf16 计算副本仍是**全量** | 要（gather bf16 副本） | **每 iteration、每 bucket 一次** |

最后一行是本篇的主题。Megatron 的做法是把 fp32 master weights 和优化器状态切成 $1/DP$，但前向/反向用的 bf16 参数缓冲区是**全量**的，每个 iteration 由 all-gather 刷新一遍（每个 bucket 一次，不是每个 microbatch 一次，更不是每层一次）。

- 不重叠：`start_param_sync` 必须等 all-gather 完成，第一个 microbatch 之前干等；
- 重叠：forward pre-hook 按层 wait，同时预取下一个 bucket，于是 gather 藏在当前层的计算里。

### 1.3 optimizer 的 all-gather

optimizer step 更新完参数分片后，下一轮前向又要把参数 gather 回来。`overlap_param_gather_with_optimizer_step` 让第一个桶的 all-gather 和 optimizer step 重叠。

## 2. 源码：三件套

Megatron 的实现可以概括为：**hook 登记 → 计数判满 → dispatch/wait 分离**。

### 2.1 开关

`DistributedDataParallelConfig`（`megatron/core/distributed/distributed_data_parallel_config.py`）：

```python
overlap_grad_reduce: bool = False
overlap_param_gather: bool = False
delay_wgrad_compute: bool = False
```

对应 CLI：

- `--overlap-grad-reduce`（`arguments.py:4131`）
- `--overlap-param-gather`（`arguments.py:4194`）
- `--overlap-param-gather-with-optimizer-step`（`arguments.py:4200`）
- 桶大小：新版用 `--ddp-num-buckets` / `--ddp-bucket-size`，对应 config 里的 `num_buckets` / `bucket_size`

注意 `--overlap-param-gather` 的约束（`arguments.py` 里 `_validate_args` 附近）：

- 只支持分布式优化器 / Megatron-FSDP；
- 必须和 `--overlap-grad-reduce` 一起开。

还有一条容易踩的：`--overlap-param-gather-with-optimizer-step` 额外要求**开启 interleaved pipeline**（`virtual_pipeline_model_parallel_size` 非空）、且**不能用分布式 checkpoint**。这两条不满足时会直接报错，不是静默降级。

### 2.2 反向 hook

DDP 初始化时给每个参数的 `grad_acc` 注册 post hook：

```python
# megatron/core/distributed/distributed_data_parallel.py
def enable_forward_pre_hook(self): ...
def _make_forward_pre_hook(self): ...
def _make_backward_post_hook(self, param): ...
```

> **版本与定位**：本文的行号对应 Megatron-LM `main`（2026 年中）那份树，一律用"函数名 + 行号"双索引——行号只作同版本内的快速跳转，跨版本请以函数名为准。这段代码 2025–2026 被重写过一轮：在本地留存的 `megatron-core 0.16.0rc0` 快照里，这三个函数分别在 `distributed_data_parallel.py:376 / :405 / :441`，而 §2.3 的 per-param 计数**还不存在**——那时判满用的是 `params_with_grad` 集合 + `is_last_microbatch`。

hook 体做三件事：

1. 把 `param.grad` 累进 `param.main_grad`（fp32 主梯度）；
2. 把 `param.grad` 置 None，省显存；
3. 调用 `register_grad_ready`（实现在 `param_and_grad_buffer.py` 的 `_ParamAndGradBuffer` 里）。

CUDA graph 捕获期间 hook 直接返回，因为 capture 时不能触发真实通信。

### 2.3 计数判满：为什么不是"参数齐全"

`register_grad_ready`（`param_and_grad_buffer.py`）维护一个 per-param 计数：

```python
# 简化
self.per_param_grad_ready_counts[param] += 1
if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
    self.start_grad_sync()
```

`golden_per_param_grad_ready_counts` 是第一个 iteration 播下的种子："每个参数在这个 iteration 里应该到达几次"（见 `_ParamAndGradBuffer.reset` 与 `_playback_golden_counts` 一类的逻辑）。

为什么用"次数相等"而不是"参数集合齐全"？因为同一个参数可能在一个 batch 里被前向图经过多次：

- tied embedding（输入/输出共享权重）；
- weight sharing；
- 某些 MoE 结构里的共享专家。

如果只检查"参数是否出现过"，会在第一次到达时就误判整桶就绪，导致梯度不完整。**这是读这段代码最容易错的地方。**

（老版本正是这么做的：只用 `params_with_grad` 集合，`register_grad_ready` 里 `if len(self.params_with_grad) == len(self.params): start_grad_sync()`。它靠 `is_last_microbatch` 挡住梯度累积的误判，但挡不住"同一 microbatch 内 tied weight 少到一次"的情况——这就是后来换成 per-param 计数的原因。）

### 2.4 dispatch 与 wait 分离

```python
# megatron/core/distributed/param_and_grad_buffer.py
def start_grad_sync(self, force_all_reduce=False):
    ...
    if self.ddp_config.overlap_grad_reduce:
        handle = torch.distributed.all_reduce(
            self.grad_data, op=..., async_op=True
        )
    ...

def finish_grad_sync(self, force_all_reduce=False):
    if self.grad_reduce_finished:      # 幂等
        return
    ...
    self.grad_reduce_finished = True
```

- `start_grad_sync` 只发 `async_op=True` 的通信，不 wait；
- `finish_grad_sync` 在真正需要梯度时（optimizer 前）才 wait；
- `grad_reduce_finished` 保证幂等：后继 bucket group 可能在 dispatch 时提前 drain 前驱，末尾 finalize 又会再调一遍，不能重复 wait。

`previous_grad_reduce_bucket_group` 只在 `reduce_scatter_with_fp32_accumulation` 路径下串联，用来保证前一个 bucket 的 reduce 完成后，后一个 bucket 才能安全复用 buffer。

### 2.5 参数 all-gather：谁发、谁等

这一段最容易读错，因为 `start_param_sync` 和 forward pre-hook 不是同一个环节：

| 环节 | 谁做 | 做什么 |
|---|---|---|
| dispatch | `start_param_sync`（`async_op=True`） | 把某个 bucket 的 all-gather **发出去**，不等 |
| 发起者 | schedule 或 optimizer | interleaved 调度在 `forward_step` 前调用一次（`config.param_sync_func`），`--overlap-param-gather-with-optimizer-step` 时由 optimizer step 发起 |
| wait | `finish_param_sync`（被 forward pre-hook 调用） | 等当前模块所属 bucket 的 handle；**等完立刻 dispatch 下一个 bucket**（`skip_next_bucket_dispatch` 可关掉） |

所以正确的说法是：**pre-hook 负责"等到我需要的那桶"，等完立刻把下一桶的 gather 推出去**（于是下一桶的传输藏在本层计算里）；而不是"pre-hook 触发 all-gather"。源码里这句写得很直白：

```python
# param_and_grad_buffer.py::finish_param_sync（docstring 原话）
# Waits for asynchronous communication call to complete
# (and dispatches one if one is not already outstanding).
if self.param_gather_handle is not None:
    self.param_gather_handle.wait()
    ...
    if self.next_param_gather_bucket_group is not None and not skip_next_bucket_dispatch:
        self.next_param_gather_bucket_group.start_param_sync()
```

这也解释了为什么"每个 microbatch 硬等一次 all-gather"是错的——一个 iteration 里每桶只 gather 一次，后面几个 microbatch 直接命中已完成的 handle。

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
2. **reduce-scatter vs all-reduce**：分布式优化器用 reduce-scatter（梯度段只传 $1/DP$），但多一次参数 all-gather。**净量≈all-reduce**，省的是显存和带宽峰值，不是总字节。

## 4. reduce-scatter + all-gather：分布式优化器的账

标准 DDP：

$$
\text{comm} = 2 \cdot \frac{DP-1}{DP} \cdot G \quad \text{(all-reduce)}
$$

分布式优化器（Megatron 的实现；注意 ZeRO-1/2 不切参数，没有后面这项 all-gather）：

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

梯度累积（gradient accumulation）下，同一份梯度要跨 microbatch 攒够才能发。Megatron 用"计数判满"处理这件事：

- 每个 microbatch 的反向都会让 `per_param_grad_ready_counts[param]` 加一；
- `golden_per_param_grad_ready_counts` 记的是"这个 iteration 里每个参数应该到达几次"（含累积倍数）；
- 两者相等才 dispatch 该桶的 reduce-scatter。

所以重叠窗口是"**最后一个 microbatch 的反向还在算，前面 microbatch 攒好的桶已经在传**"，而不是"全部反向算完再统一发"。三个连带后果：

- 累积步数越多，桶就绪越晚（要等最后一个 microbatch），但每桶的通信量不变；
- `is_last_microbatch` 管的是 iteration 末尾那次 `finish_grad_sync`（optimizer 前必须等齐），和"桶什么时候 dispatch"是两件事，别混；
- 一个 iteration 里每个桶都会发一次，所以在途流量上限就是梯度总量——真正要调的是桶数（launch 次数）与桶大小（尾延迟），也就是 §3 那条曲线。

## 结论

- DP 的三个暴露点：反向梯度 reduce、前向参数 all-gather、optimizer 后的 all-gather。
- Megatron 的三件套：hook 登记 → per-param 计数判满（处理 tied weights）→ `start_*` 发异步、`finish_*` 才 wait。
- 桶不是越细越好：$T_{\text{exposed}} = t + (N-1)\max(0,t-c)$，最优在 $t \approx c$ 附近；launch 和 NCCL 小消息效率决定了下界。
- 分布式优化器把 all-reduce 拆成 reduce-scatter + all-gather（多出来的那次 gather 只有 Megatron 分布式优化器 / ZeRO-3 / FSDP 才有，ZeRO-1/2 不切参数，没有），方向不同，重叠策略也不同。
- 先开 `--overlap-grad-reduce`，再开 `--overlap-param-gather`；桶大小从一层参数量的量级开始扫。

下一篇进入张量并行：TP 的 all-reduce 在层内、和 GEMM 紧密耦合，不能像 DP 那样简单挪到 side stream。Flux 的做法是把通信拆到 tile 级、融进 GEMM kernel——这也是"切"和"融"结合得最漂亮的一个案例。

---

**系列导航**：[三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}}) · [五：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}})

> 配套代码：`code/comm-overlap/bucket_sim.py`、`overlap_model.py`
> 自测：`python3 bucket_sim.py --test && python3 overlap_model.py --test`
