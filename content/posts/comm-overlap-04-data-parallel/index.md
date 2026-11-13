---
title: "通算并行研读（四）：数据并行——算完一桶，先发一桶"
date: 2026-11-02
draft: false
weight: 5
series: ["comm-overlap"]
tags: ["通算并行", "系列", "数据并行", "DDP", "megatron-lm", "reduce-scatter", "分布式优化器"]
categories: ["训练框架"]
---

数据并行只有一处硬等：反向算完了，梯度还没同步完，optimizer 就不能开始。

**数据并行把梯度同步从全算完再发改成了算完一桶发一桶**，代价是多几次通信启动，尾巴还会露一点。

读完你会知道：

- 梯度和参数两组通信，各自暴露在迭代的哪一段；
- 为什么判满要数"每个参数到了几次"，而不是看"参数齐没齐"；
- 桶多大才不亏，先开哪个开关。

## 1. 暴露点只有一段，就在 optimizer 之前

DP 的通信最老实：每张卡算完梯度，把各卡的平均一下。它没有流水线的依赖链，也没有 MoE 的带宽压力，所以重叠起来相对简单——但暴露点非常固定。

70B 模型、BF16、DP=8，梯度总量约 140 GB。就算 NVLink 跑到 400 GB/s，一次全量 all-reduce 也要几百毫秒；每步都等，吞吐会被砍掉一大截。

Megatron 的做法是把梯度切成桶：桶按反向到达顺序装，装满就发 reduce-scatter，wait 推到 optimizer 之前。DP 的通信其实有三处，但只有一处是硬等：

```text
反向：桶满即发 reduce-scatter，藏在后续反向里，这是唯一能盖住的一段
尾桶：wait 推到 optimizer 之前，硬暴露，约等于最后一桶的通信
step 后：桶 1 的 all-gather 和 optimizer step 重叠
前向：逐层 all-gather，pre-hook 只等本层，等完推下一桶
```

前向这一项最容易读错，因为"切优化器状态"和"切参数"是两件事：

| 方案 | 切什么 | 前向要再 gather 吗 |
|---|---|---|
| ZeRO-1 | 只切优化器状态（fp32 master、Adam $m/v$） | 不要，参数全复制 |
| ZeRO-2 | 再加梯度切分 | 不要 |
| ZeRO-3 / FSDP | 再加参数切分 | 要，每层用完即弃 |
| Megatron 分布式优化器 | fp32 主权重与优化器状态切 $1/DP$，bf16 计算副本仍是全量 | 要，gather bf16 副本，每 iteration 每桶一次 |

最后一行才是本篇的主角：前向反向用的 bf16 参数缓冲区是全量的，每个 iteration 每桶 gather 一次，不是每 microbatch 一次，更不是每层一次。

## 2. 两个巧劲：数次数，顺手预取

实现可以压成两句：判满靠 per-param 计数，wait 与 dispatch 分成两个函数。

### 2.1 判满数的是"到达几次"

DDP 初始化时给每个参数的 `grad_acc` 注册 post hook（`distributed_data_parallel.py`）。hook 里做三件事：把 `param.grad` 累进 fp32 的 `main_grad`，把 `param.grad` 置 None 省显存，再调 `register_grad_ready`。

CUDA graph 捕获期间 hook 直接返回，因为 capture 时不能触发真实通信。

`register_grad_ready`（`param_and_grad_buffer.py`）维护一个 per-param 计数：

```python
self.per_param_grad_ready_counts[param] += 1
if self.per_param_grad_ready_counts == self.golden_per_param_grad_ready_counts:
    self.start_grad_sync()
```

`golden_per_param_grad_ready_counts` 是第一个 iteration 播下的种子：每个参数在这一轮应该到达几次。梯度累积的倍数也算在里面，所以桶在最后一个 microbatch 的反向里就能发出去。

为什么不看"参数齐没齐"？同一个参数可能在一轮里被前向图经过多次：tied embedding、weight sharing、某些 MoE 的共享专家。只检查"出现过"，第一次到达就会把整桶判成就绪，梯度不完整。**读这段代码最容易错的就是这里**。

老版本只看 `params_with_grad` 集合的大小，用 `is_last_microbatch` 挡住跨 microbatch 的累积误判，但挡不住同一 microbatch 内 tied weight 少到一次的情况。

这段代码 2025–2026 重写过一轮，本地留存的 `megatron-core 0.16.0rc0` 快照里还是老写法。所以跨版本请按函数名找，别按行号。

### 2.2 pre-hook 只 wait，等完顺手推下一桶

wait 和 dispatch 是两个函数：`start_*` 只发 `async_op=True` 的通信，`finish_*` 才等。

反向侧由 `finish_grad_sync` 在 optimizer 前等齐最后一桶；`grad_reduce_finished` 保证幂等——后继桶可能在 dispatch 时提前 drain 前驱，末尾 finalize 还会再调一遍。

`previous_grad_reduce_bucket_group` 只在 fp32 累积那条路径上串联：前一桶 reduce 完，后一桶才能安全复用 buffer。

参数侧最容易读错：pre-hook 不是"触发 all-gather"的那个环节。

| 环节 | 谁做 | 做什么 |
|---|---|---|
| 发起 | `start_param_sync`；由 schedule 的 `param_sync_func` 或 optimizer step 调用 | 把某桶的 all-gather 发出去，不等 |
| 等待 | forward pre-hook 调 `finish_param_sync` | 等当前模块那一桶，等完立刻 dispatch 下一桶 |

不开 overlap 时，`start_param_sync` 必须等 gather 完成，第一个 microbatch 之前只能干等。

开了之后，pre-hook 负责等到本层要用的那桶，并顺手把下一桶推出去，下一桶的传输就藏在本层计算里；`skip_next_bucket_dispatch` 能关掉这个预取。

这也解释了为什么"每个 microbatch 硬等一次 all-gather"是错的：一轮里每桶只 gather 一次，后面的 microbatch 直接命中已完成的 handle。

还有个旋钮：`delay_wgrad_compute` 把 wgrad 计算往后挪，让它离自己的 reduce 更近。它只在 `transformer_impl=transformer_engine` 下可用，且要求 TE ≥ 2.8.0。

## 3. 桶不是越细越好

`bucket_sim.py` 的模型很直白：$N$ 个桶，单桶计算 $c$、单桶通信 $t$（含 launch），第 $i$ 个桶在 $(i+1)c$ 就绪，通信走单通道 FIFO。

$$
T_{\text{exposed}} = t + (N-1)\max(0, t-c)
$$

一句话：尾桶那 $t$ 必然暴露；前面每多一个"传得比算得慢"的桶，就多欠一份 $t-c$。

固定总计算 100、总通信 60、每桶 launch 2（单位是模型时间，不是秒）；端到端减掉总计算，剩下的就是暴露量：

| $N$（桶数） | 单桶计算 $c$ | 单桶通信 $t$ | 端到端 |
|---|---|---|---|
| 1 | 100.00 | 62.00 | 162.00 |
| 4 | 25.00 | 17.00 | 117.00 |
| 8 | 12.50 | 9.50 | 109.50 |
| 16 | 6.25 | 5.75 | 105.75 |
| 32 | 3.12 | 3.88 | 127.12 |
| 64 | 1.56 | 2.94 | 189.56 |

三段区间，对应三种病：

- $N$ 太小（1–4）：最后一桶的通信直接暴露，$t$ 很大；
- $N$ 刚好（8–16）：$t<c$，尾桶被计算盖住，暴露贴着 $t$；
- $N$ 太大（32–64）：launch 把 $t$ 推过 $c$，通信开始堆积，暴露反弹到 27.12，再往上就是 89.56。

最优桶数就在 $t \approx c$ 附近，由"尾延迟 vs launch/带宽"共同决定。真实系统还要再打一个折：桶太小，消息太小，ring/tree 算法打不满带宽，`pad_buckets_for_high_nccl_busbw` 就是为此加的。

一轮里每个桶都会发一次，所以在途流量上限就是梯度总量。真正能调的只有两个：桶数（launch 次数）和桶大小（尾延迟），也就是上表这条 U 形。

经验起点是"一层 Transformer 的参数量"：hidden=4096、FFN=16384 时约 100M 参数，BF16 下折成 200 MB——偏大，通常还要再切。

新版 Megatron 直接用 `--ddp-num-buckets` 指定桶数，比自己算 `bucket_size` 直观。

下界则由 launch 和 NCCL 延迟决定：桶小于 1 MB 时它们占主导，大于 100 MB 又轮到尾延迟和显存峰值。所以桶数不是越多越好，模拟里那个凹陷就是这件事的定量版。

## 4. 拆成 RS + AG：净量不变，方向变了

标准 DDP 是一次 all-reduce：

$$
\text{comm} = 2\cdot\frac{DP-1}{DP}\cdot G
$$

分布式优化器拆成两段（ZeRO-1/2 不切参数，没有第二项）：

$$
\text{comm} = \frac{DP-1}{DP}\cdot G \;(\text{reduce-scatter}) + \frac{DP-1}{DP}\cdot P \;(\text{all-gather})
$$

$G$ 是梯度量，$P$ 是参数量。BF16 下 $G \approx P$，所以总字节和 all-reduce 差不多——省的是显存和带宽峰值，不是通信量。

但方向不同：reduce-scatter 在反向，和计算同向，容易重叠；all-gather 在前向，和计算反向，必须提前预取。`overlap_param_gather` 干的就是这件事：当前层在算，提前 gather 下一层。

## 5. 先开 grad reduce，再开 param gather

两个开关的收益不稳，因为它们对桶大小、桶顺序和并行度的敏感程度不一样。按这个顺序开：

| 开关或决定 | 怎么定 | 代价与前提 |
|---|---|---|
| `--overlap-grad-reduce` | 先开这个 | 反向计算量大，桶容易藏住，收益更稳 |
| `--overlap-param-gather` | 确认上一条有效后再开 | 只支持分布式优化器 / Megatron-FSDP，必须和上一条同开 |
| `--overlap-param-gather-with-optimizer-step` | 想在 optimizer 处再抢一段 | 要求 interleaved pipeline、不能用分布式 checkpoint，否则直接报错 |
| launch 开销 | 每桶一次 NCCL launch，GPU 侧约 2–5 μs | 桶太多，CPU 侧可能先被 Python 和 hook 开销卡住 |
| 和 PP/TP 一起开 | 同时开 `overlap_grad_reduce` 和 `overlap_p2p_comm` 前，先确认通信流与优先级 | PP 的 P2P 和 DP 的 reduce-scatter 抢网络，TP 的 all-reduce 抢 SM |

## 如果只记一件事

**能藏多少由最后一个桶决定，不由平均值决定**。桶切得再细，尾桶该暴露还是暴露；桶切得太大，前面全在干等。

下一篇进张量并行：TP 的 all-reduce 在层内、和 GEMM 紧耦合，不能像 DP 这样挪到 side stream。Flux 把通信拆到 tile 级、融进 GEMM kernel，是"切"和"融"结合得最漂亮的一个案例。

---

**系列导航**：[三：DualPipe]({{< relref "comm-overlap-03-dualpipe" >}}) · [五：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}})

> 配套代码：`code/comm-overlap/bucket_sim.py`、`overlap_model.py`
> 自测：`python3 bucket_sim.py --test && python3 overlap_model.py --test`
