---
title: "通算并行研读（三）：DualPipe——双向流水如何把 bubble 压到 (PP/2-1)"
date: 2026-11-02
draft: false
weight: 4
series: ["comm-overlap"]
tags: ["通算并行", "系列", "流水线并行", "DualPipe", "DeepSeek-V3", "Zero Bubble"]
categories: ["训练框架"]
---

Zero Bubble 的思路是"用 W 填 B 的空隙"，但它仍然是一个**单向**流水：microbatch 从 stage 0 走到 stage $p-1$，再反向走回来。DualPipe 换了一个更激进的排法：

> 把流水拆成**两个方向**，每个设备同时持有两个方向的 chunk；一个方向的前向，和另一个方向的反向，正好可以同时做。

这样做的效果是：**F 和 B 互相填充**，气泡公式从 $(PP-1)(F+B)$ 变成 $(PP/2-1)(F\&B+B-3W)$。代价也很直接：每设备参数 **2×**，激活 $PP+1$ 份。

这一篇拆 DualPipe 的 8 步调度、源码实现，以及 DeepSeek-V3 里它和通信 SM 预算的配合。

## 1. 先看它到底长什么样

普通 1F1B 是单向的：

```text
stage 0 → stage 1 → ... → stage 7
        ←        ←        ←
```

DualPipe 把 8 个 PP rank 分成两半，前 4 个负责方向 0（正向），后 4 个负责方向 1（反向）。每张卡上放**两个 chunk**：一个属于方向 0，一个属于方向 1。

于是同一时刻，一张卡可以：

- 做方向 0 的 F（它的下游在 rank+1）；
- 同时做方向 1 的 B（它的上游在 rank-1，但方向 1 的数据从另一头来）。

```text
rank 0        rank 1        rank 2        rank 3
dir0 F →      dir0 F →      dir0 F →      dir0 F →
dir1 B ←      dir1 B ←      dir1 B ←      dir1 B ←
```

两个方向的 F/B 在时间上错开，就形成了"F&B"重叠对。

### 1.1 气泡公式

DualPipe README 给的表（$PP$ 为偶数）：

| 方法 | 气泡 | 每设备参数 | 每设备激活 | 设备数 |
|---|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | 1× | $PP$ | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | 1× | $PP$ | $PP$ |
| **DualPipe** | $(PP/2-1)(F\&B+B-3W)$ | **2×** | $PP+1$ | $PP$ |
| **DualPipeV** | $(PP/2-1)(F\&B+B-3W)$ | **2×** | $PP+1$ | $PP/2$ |

$F\&B$ 是"一对前向/反向重叠后的等效时间"，它小于 $F+B$（因为两者并行）。所以 DualPipe 的气泡不仅系数从 $PP-1$ 降到 $PP/2-1$，括号里的项也被 $F\&B$ 压小了。

举例：$PP=8$，$F=B=W=1$，假设 $F\&B=1.5$：

- 1F1B：$7 \times 3 = 21$；
- DualPipe：$(4-1) \times (1.5+1-3) = 3 \times (-0.5) = -1.5$ → 截断为 0。

当然这是理想化数字；真实系统里 $F\&B$ 受通信、SM 争用影响，不会这么低。但公式说明了方向：**DualPipe 用 2 倍参数，把气泡的系数减半，再用 F/B 重叠把括号里的时间压下去。**

## 2. 8 步调度：源码里的 `step()`

`dualpipe/dualpipe.py` 的 `step()`（从第 330 行开始）把一次迭代拆成 8 步。设 `num_ranks = PP`，`half_rank = min(rank, PP-1-rank)`，`num_half_ranks = PP/2`：

| 步骤 | 代码行 | 做什么 | 作用 |
|---|---|---|---|
| 1 | `:358` | `nF0`：方向 0 连续前向 | 填充方向 0 的流水 |
| 2 | `:363` | `nF0F1`：方向 0/1 交替前向 | 把两个方向都填起来 |
| 3 | `:373` | `nB1W1F1`：方向 1 反向 + W + 方向 1 前向 | 用 W 填 B 的空隙（zero bubble） |
| 4 | `:381` | **主步** `nF0B1F1B0` | 两个方向的 F/B 交叉，形成 F&B 重叠 |
| 5 | `:398` | `nB1F1B0` | 继续交叉，逐步收尾 |
| 6 | `:404` | `nB1B0` | 两个方向的反向 |
| 7 | `:415` | `nWB0` | W 填方向 0 的尾部气泡 |
| 8 | `:421` | `nW` | 清空 WeightGradStore |

主步的代码（简化）：

```python
# dualpipe/dualpipe.py:381
step_4 = half_num_chunks - num_ranks + half_rank + 1
for i in range(step_4):
    if i == 0:
        ...
        self._forward_backward_chunk(0, 1, recv0=False)
    else:
        self._forward_backward_chunk(0, 1)
    self._forward_backward_chunk(1, 0)
```

`_forward_backward_chunk(phase0, phase1)` 做四件事：

1. `_recv_forward(phase0)` / `_recv_backward(phase1)`：把两个方向的 P2P 收包都挂上；
2. `_commit_and_wait_comm()`：`batch_isend_irecv` 一起提交，**等所有包完成**；
3. `_forward_backward_compute_chunk(phase0, phase1)`：调用模块的 `overlapped_forward_backward`，同时算方向 0 的 F 和方向 1 的 B；
4. `_send_forward(phase0)` / `_send_backward(phase1)`：把两个方向的发包包好。

关键在第 3 步。`_forward_backward_compute_chunk`（`:121`）会：

```python
# dualpipe/dualpipe.py:168
outputs0, loss0 = type(module0).overlapped_forward_backward(
    module0, inputs0, criterion0, labels0,
    module1, loss1, outputs1, output_grads1,
)
```

**`overlapped_forward_backward` 不是框架自带的，要模型自己实现。** README 明确说："For real-world applications, you will need to implement a custom `overlapped_forward_backward` method tailored to your specific module." 这是 DualPipe 落地的最大工程门槛：它要求你把"一个 chunk 的前向"和"另一个 chunk 的反向"写进同一个函数，共享输入/输出，甚至共享一些中间 buffer。

## 3. 通信是怎么被藏住的

DualPipe 的通信是 stage 间 P2P（激活/梯度），不是 collective。它的重叠靠两点：

### 3.1 批量提交，一次 wait

`_commit_and_wait_comm()`（`:285`）：

```python
reqs = dist.batch_isend_irecv(self.comm_ops)
for req in reqs:
    req.wait()
```

它把当前 step 攒下的所有 send/recv 一起提交，然后**等全部完成**再进计算。粗看这是"同步等待"，但注意：这些 P2P 是在**上一个计算 chunk 之后**提交的，而下一个计算 chunk 需要的数据正好是这些包。所以等待被放在"确实需要数据"的那一刻，前面的计算已经盖住了传输时间。

更准确地说：DualPipe 不是"发完就算"，而是**把通信和计算交错成固定的 8 步节奏**，让每个 step 的通信窗口刚好等于另一个方向的计算窗口。这是"错"（stagger）的极致——不是动态调度，而是把调度写死成周期。

### 3.2 W 的延迟执行

`WeightGradStore`（`dualpipe/utils.py:8`）是一个全局队列：

- `_backward_compute_chunk(enable_zb=True)` 时，不立刻算 W，而是把 W 的 closure 塞进 `funcs_queue`；
- `_weight_chunk()`（`dualpipe.py:216`）在需要填气泡时 `WeightGradStore.pop()`，执行一个 W；
- 第 8 步 `nW` 把剩下的 W 全部清空。

这就是 Zero Bubble 的 W 填充在 DualPipe 里的实现：**W 被推迟到真正有空隙的时候**，而不是跟着 B 立刻算。

## 4. 代价：2× 参数、PP+1 激活、写死的调度

### 4.1 显存

每张卡放两个 chunk（方向 0 和方向 1），所以：

- 参数：2×（两个 chunk 的权重）；
- 激活：$PP+1$ 份（比 1F1B 的 $PP$ 多一份，用于跨方向流水）。

DeepSeek-V3 的配置里，DualPipe 每个 rank 还要留 **20 个 SM 给通信**（132 个 SM 中的 20 个），计算和通信的 SM 比约 **1:1**。这意味着：**DualPipe 不是"通信不占资源"，而是"把资源预算也写进调度"**。

### 4.2 调度是静态的

8 步的步长由 `half_rank`、`num_half_ranks`、`num_chunks` 算出来，**不随运行时负载变化**。这带来两个问题：

- 如果某个 chunk 的计算时间偏离预期（routing skew、变长序列），固定节奏就会出现气泡；
- `num_chunks` 必须是偶数且 $\ge 2 \times num\_ranks$（`dualpipe.py:333` 的 assert），小 batch 用不了。

### 4.3 需要模型配合

`overlapped_forward_backward` 要自己写。对 Transformer，通常要把 attention 和 MLP 的前向/反向拆成可组合的单元，才能让"方向 0 的 F"和"方向 1 的 B"共享一个 kernel 启动。这也是为什么 DualPipe 的代码只有 440 行，但真正落地要改模型。

## 5. DualPipeV：把设备数减半

DualPipeV 是 Sea AI Lab 的 "cut-in-half" 变形：

- 设备数从 $PP$ 降到 $PP/2$；
- 每个设备放两个 stage（V 形）；
- 气泡公式不变，参数 2×，激活 $PP+1$。

直觉：既然每张卡已经放了两个 chunk（两个方向），不如把这两个 chunk 变成**两个相邻的 stage**，用 V 形数据流让它们首尾相接。这样同样的模型规模，设备数减半，但单卡要算两个 stage 的活。

`dualpipe/dualpipev.py` 是它的实现；README 里的示例是 4 个 PP rank（8 个 PP stage）、10 个 microbatch。

## 6. 和 Zero Bubble 的关系

| 维度 | Zero Bubble | DualPipe |
|---|---|---|
| 方向 | 单向 | 双向 |
| 气泡 | $(p-1)(F+B-2W)$（ZB-H2） | $(PP/2-1)(F\&B+B-3W)$ |
| 参数 | 1× | 2× |
| 激活 | $(2p-1)M_B$（ZB-H2） | $PP+1$ |
| 调度 | 自动搜索（启发式/ILP） | 8 步写死 |
| 适用 | 通用 PP | 大规模、跨节点、有 NVLink+IB 混合拓扑 |

DualPipe 可以看作"把 Zero Bubble 的 W 填充，和双向流水的 F/B 重叠结合起来"。它牺牲了通用性（要改模型、要偶数 chunk、要 2× 参数），换来的是在大规模训练里更极致的 bubble 压缩。

DeepSeek-V3 的报告里，DualPipe 让训练在 2,788K H800 GPU hours 内完成；V3 的通信预算是每 rank 20/132 SM、1:1 计算通信比、最多 4 节点、每节点 3.2 个专家、最大 13 个专家跨节点。这些数字说明：**DualPipe 的收益不是"算法 alone"，而是算法 + 拓扑 + SM 预算一起调出来的。**

## 7. 什么时候不该用 DualPipe

- **模型小、PP 小**：$PP=2$ 时气泡系数 $PP/2-1=0$，DualPipe 的公式优势消失，2× 参数纯亏。
- **单机、NVLink 很快**：P2P 通信本来就不贵，双向流水带来的复杂度不划算。
- **batch 小、chunk 数不够**：`num_chunks >= 2*num_ranks` 的约束直接卡死。
- **不想改模型**：没有 `overlapped_forward_backward`，DualPipe 退化成普通双向流水，收益大打折扣。

## 结论

- DualPipe 把流水拆成两个方向，每设备两个 chunk；一个方向的 F 和另一个方向的 B 重叠，气泡从 $(PP-1)(F+B)$ 降到 $(PP/2-1)(F\&B+B-3W)$。
- 代价是参数 2×、激活 $PP+1$、调度写死、需要模型实现 `overlapped_forward_backward`。
- 源码的 8 步调度（`dualpipe.py:358-425`）是"静态错峰"的典型：通信批量提交、一次 wait，W 用 `WeightGradStore` 延迟到空隙。
- DualPipeV 用 V 形把设备数减半，气泡公式不变。
- 它不是"通用优化"，而是**大规模、跨节点、计算通信比可调**场景下的专用方案。

下一篇回到更常见的场景：数据并行。看 Megatron 怎么用 bucket + backward hook，把梯度 reduce-scatter 藏进反向计算里。

---

**系列导航**：[二：流水线气泡]({{< relref "comm-overlap-02-pipeline" >}}) · [四：数据并行]({{< relref "comm-overlap-04-data-parallel" >}})

> 配套代码：`code/comm-overlap/overlap_model.py`（含 `pipeline_dualpipe` 公式）
> 自测：`python3 overlap_model.py --test`
