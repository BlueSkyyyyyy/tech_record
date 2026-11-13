---
title: "通算并行研读（三）：DualPipe——两个方向对开，气泡压到 (PP/2-1)"
date: 2026-11-02
draft: false
weight: 4
series: ["comm-overlap"]
tags: ["通算并行", "系列", "流水线并行", "DualPipe", "DeepSeek-V3", "Zero Bubble"]
categories: ["训练框架"]
---

**DualPipe 把单向流水换成两个方向对开：一张卡同时算一路的前向和另一路的反向，气泡从跟段数成正比降到跟段数的一半成正比；代价是参数翻倍，外加一个模型必须自己写的融合函数**。

读完你会知道：

- 为什么「两个方向」不是把卡分成两拨；
- 八步调度里真正有用的只有三步；
- 什么规模的模型不该上它。

先用同一组参数算两笔账。$PP=8$、单段前向 $F=1$、整段反向 $B=1.5$（其中 $W=0.5$）、一对前反向重叠后 $F\&B\approx1.2$：

- 1F1B：$7\times(1+1.5)=17.5$；
- DualPipe：$(4-1)\times(1.2+1.5-3\times0.5)=3\times1.2=3.6$，约 1F1B 的 1/5。

少掉的是空转，不是算力。它把另一个方向的活，塞进了这一方向的等待时间里。

## 1. 两个方向是镜像的，不是把卡劈成两半

普通 1F1B 单向：microbatch 从第一级走到最后一级，再反向走回来，每张卡都有一长段只能干等。

DualPipe 最容易被讲错的就是「两个方向」怎么落到设备上。**不是**把 8 个 rank 分成两半、前 4 个跑方向 0、后 4 个跑方向 1——那样每个方向只剩 4 级流水，系数减半就对不上了。

DualPipe 是每个 rank 都同时参与两个方向：

- rank $r$ 在方向 0 上负责第 $r$ 个 stage；
- rank $r$ 在方向 1 上负责第 $PP-1-r$ 个 stage，正好是镜像。

```text
dir0:  rank0 ─F→ rank1 ─F→ rank2 ─F→ ... ─F→ rank7
dir1:  rank7 ←B─ rank6 ←B─ rank5 ←B─ ... ←B─ rank0
       （同 8 个 rank，每个 rank 同时持有一个 dir0 chunk 和一个 dir1 chunk）
```

于是同一时刻，方向 0 的前向正好填方向 1 的反向。

## 2. 系数减半，括号里的项也一起变小

DualPipe README 给了一张对照表（$PP$ 为偶数）：

| 方法 | 气泡 | 每设备参数 | 每设备激活 |
|---|---|---|---|
| 1F1B | $(PP-1)(F+B)$ | 1× | $PP$ |
| ZB1P | $(PP-1)(F+B-2W)$ | 1× | $PP$ |
| DualPipe | $(PP/2-1)(F\&B+B-3W)$ | 2× | $PP+1$ |

这张表要拆开读两处。

第一，这里的 $B$ 是整段反向（README 口径），不是[第二篇]({{< relref "comm-overlap-02-pipeline" >}})里只含 dX 的 $B$。所以 ZB1P 的 $(PP-1)(F+B-2W)$ 和 ZB-H2 的 $(p-1)(T_F+T_B-2T_W)$ 数值一样，只是记号不同。但 README 的 ZB1P 就是 H2 的形状，和论文里那个「显存受限的自动调度 ZB-1p」不是一回事。

第二，$F\&B$ 是一对前向/反向重叠后的等效时间，比 $F+B$ 小。所以 DualPipe 不只是把系数从 $PP-1$ 降到 $PP/2-1$，括号里的项也被一起压小了。

公式还有有效域。把 $W$ 取到和 $F$ 一样大（$F=B=W=1$）时，$F\&B+B-3W$ 会算成负数。这不是「负气泡」，是这组参数已经出了模型假设：$W$ 不可能和整段反向同量级。

## 3. 三步就能讲完：先填、再对开、最后收尾

源码里 `step()` 把一次迭代切成八步，符号名是 `nF0`、`nB1W1F1` 这种，没读过源码等于零信息。按「哪张卡在干什么」重讲，只有三段：

1. **填流水**：每张卡先只朝方向 0 连续发前向，把这条链灌满；再切成方向 0 和方向 1 交替前向，让两条链都热起来（源码里从 `nF0` 走到 `nF0F1`）。此时还没有反向可算。
2. **主步**：轮到成对调用 `_forward_backward_chunk(0, 1)` 和 `(1, 0)`。一次调用里同时算方向 0 的前向和方向 1 的反向，这就是那个重叠对。
3. **收尾**：两个方向把剩下的反向做完，W 从 `WeightGradStore` 里逐个放出来填尾部空隙，最后一步清空队列。反向时它并不算 W，只把 W 的 closure 塞进 `funcs_queue`，有空隙再 `pop()` 一个。

主步的骨架：

```python
# dualpipe/dualpipe.py
step_4 = half_num_chunks - num_ranks + half_rank + 1
for i in range(step_4):
    self._forward_backward_chunk(0, 1)   # 方向 0 前向 + 方向 1 反向
    self._forward_backward_chunk(1, 0)
```

一次 `_forward_backward_chunk` 做四件事：挂两路收包、用 `batch_isend_irecv` 一次提交并等全部完成、调 `overlapped_forward_backward` 把两路一起算掉、挂两路发包。

**DualPipe 不是发完就算，而是把通信和计算交错成一段固定的节奏**：等待发生在「确实要数据」的那一刻，每个 step 的通信窗口刚好对上另一个方向的计算窗口。这是「错峰」的极致——调度不是运行时动态排的，是写死成周期的。

## 4. 代价：一倍参数、一条写死的节奏、一个要自己写的函数

**显存**：每张卡放两个 chunk（方向 0 和方向 1），参数就是 2×；激活 $PP+1$ 份，比 1F1B 的 $PP$ 多一份，用于跨方向流水。

**节奏写死**：八步的步长由 `half_rank`、`num_half_ranks`、`num_chunks` 算出来，不随负载变。哪个 chunk 因为 routing skew 或变长序列慢下来，固定节奏立刻变成气泡。另外 `num_chunks` 必须是偶数且 $\ge 2\times num\_ranks$（`step()` 开头的 assert），小 batch 直接卡死。

**要改模型**：`overlapped_forward_backward` 框架不自带，README 明确说要按自己的 module 写一个定制版。对 Transformer，得把 attention 和 MLP 的前向/反向拆成可组合的单元，才能让方向 0 的 F 和方向 1 的 B 共享一次 kernel 启动。DualPipe 本身只有 440 行，真正落地要改的是模型。

还有一笔容易看漏的账。V3 的配置里每个 rank 留 20 个 SM 给通信——132 个里的 20 个，剩下 112 个给计算。

别把 V3 那句「计算通信比约 1:1」当成 SM 分配比例。它说的是 MoE 跨节点 EP 的负载特征：未优化时通信耗时和计算耗时同量级。20:112 才是 SM 预算。合起来就是：DualPipe 不是通信不占资源，是把资源预算也写进了调度。

## 5. 变体与边界：设备减半，或者干脆别用

DualPipeV 是 Sea AI Lab 的 cut-in-half 变形：设备数从 $PP$ 降到 $PP/2$，每个设备放两个相邻 stage（V 形），气泡公式不变，参数还是 2×、激活还是 $PP+1$。直觉是每张卡的两个 chunk 干脆变成首尾相接的两个 stage：同样模型规模设备减半，单卡算两个 stage 的活。实现在 `dualpipe/dualpipev.py`，README 示例是 4 个 PP rank（8 个 stage）、10 个 microbatch。

它和 Zero Bubble 的关系一句话能说清：DualPipe 就是「ZB 的 W 延迟执行」加「双向的 F/B 重叠」。换来更小的气泡，代价是通用性——ZB 的调度是自动搜的（启发式/ILP），DualPipe 八步写死；ZB 是 1× 参数配 $(2p-1)M_B$ 激活，DualPipe 是 2× 配 $PP+1$；适用面从「通用 PP」缩到「大规模、跨节点、NVLink+IB 混合拓扑」。

四条明确的劝退条件：

- **模型小、PP 小**：$PP=2$ 时气泡系数 $PP/2-1=0$，公式优势消失，2× 参数纯亏；
- **单机、NVLink 很快**：P2P 本来就不贵，双向流水的复杂度不划算；
- **batch 小、chunk 数不够**：`num_chunks >= 2*num_ranks` 的约束直接卡死；
- **不想改模型**：没有 `overlapped_forward_backward`，DualPipe 退化成普通双向流水，收益大打折扣。

反过来，V3 正是它想吃的场景：报告里 DualPipe 让训练落在 2,788K H800 GPU hours。

> **口径**：每 rank 20/132 SM、1:1 计算通信比、最多 4 节点、每节点 3.2 个专家、最大 13 个专家跨节点。收益不是算法单独挣来的，是算法、拓扑、预算一起调出来的。

## 结论

**如果只记一件事**：DualPipe 用一倍参数和一条写死的节奏，换掉了流水线里最难消的那段空转——一个方向的空等，正好是另一个方向在干活的时间。

下一篇回到更常见的场景：数据并行。看 Megatron 怎么用 bucket 加 backward hook，把梯度的 reduce-scatter 藏进反向计算里。

---

**系列导航**：[二：流水线气泡]({{< relref "comm-overlap-02-pipeline" >}}) · [四：数据并行]({{< relref "comm-overlap-04-data-parallel" >}})

> 配套代码：`code/comm-overlap/overlap_model.py`（含 `pipeline_dualpipe` 公式）
> 自测：`python3 overlap_model.py --test`
