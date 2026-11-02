---
title: "通算并行研读（一）：先算清三笔账——关键路径、暴露通信与 overlap tax"
date: 2026-11-02
draft: false
weight: 2
series: ["comm-overlap"]
tags: ["通算并行", "系列", "性能模型", "compute-comm-overlap", "流水线并行", "roofline"]
categories: ["训练框架"]
---

先看一个真实会发生的场景：

- nsys 上通信和计算的重叠率 **95%**，团队很开心；
- 端到端却比串行版本只快 **8%**，甚至某些 shape 更慢；
- 把通信全关掉（假装没有 TP/DP），反而快了 **12%**。

这不是测量错了，而是**指标选错了**。重叠率只回答"通信有没有和计算同时在跑"，不回答"计算有没有被拖慢、关键路径有没有缩短、显存多用了多少"。这一篇把评价通算并行的账本搭起来，后面所有文章都复用这套语言。

## 1. 先把任务画成 DAG，不要看时间轴截图

nsys 的 timeline 很直观，但它把"并发"画成了"并行"：两条 stream 叠在一起，看起来就像两份工作都全速在跑。实际要看的是**依赖图**：

- 节点是任务：GEMM、all-reduce、all-to-all、norm、optimizer step；
- 边是依赖：数据依赖（A 的输出是 B 的输入）、资源依赖（同一个 SM / 同一条通信通道一次只能跑一个）。

一条**关键路径**是从源点到汇点的最长路径。只有关键路径上的时间才决定端到端；不在关键路径上的通信，哪怕很慢，也可能被完全藏住。

```text
        ┌─────────┐      ┌─────────┐
input ──►  GEMM   ├──────►  GEMM   ├──► ...
        └────┬────┘      └────┬────┘
             │ grad           │ grad
             ▼                ▼
        ┌─────────┐      ┌─────────┐
        │ reduce  │      │ reduce  │   ← 不在关键路径上就能被藏
        └─────────┘      └─────────┘
```

所以第一个原则：**先找关键路径，再谈 overlap。** 关键路径上没有通信，overlap 做得再漂亮也不涨吞吐；关键路径上有一长段通信，才值得动刀。

## 2. 三笔账：compute、exposed、bubble

对一次迭代（或一层），记：

$$
T = T_{\text{compute}} + T_{\text{exposed}} + T_{\text{bubble}}
$$

- $T_{\text{compute}}$：真正做浮点运算的时间；
- $T_{\text{exposed}}$：通信没被藏住、压在关键路径上的部分；
- $T_{\text{bubble}}$：依赖没满足导致的空转（流水线气泡、等 all-to-all 的 expert 空转）。

通信总量记为 $T_{\text{comm}}$。被藏住的比例：

$$
\eta = 1 - \frac{T_{\text{exposed}}}{T_{\text{comm}}}
$$

$\eta$ 就是常说的 **overlap efficiency / overlap ratio**。注意它的分母是**通信总量**，不是端到端时间。这个定义本身没错，但它有两个盲区：

1. 它不区分"通信被藏住"和"计算被拖慢"；
2. 它不给上界：通信量越大，$\eta$ 越容易做得高，但端到端可能更慢。

### 2.1 一个手算例子

假设一层：

- 计算 $C=100$ ms；
- 通信 $M=80$ ms；
- 理想重叠（通信完全被藏住，且不拖慢计算）：$T=\max(100,80)=100$ ms，$\eta=100\%$。
- 串行：$T=100+80=180$ ms。
- 重叠后计算被拖慢 20%：$C'=120$ ms，$T=\max(120,80)=120$ ms，$\eta$ 仍然是 **100%**，但只比串行快 **50%**，而不是 **80%**。

如果计算被拖慢 80%（$C'=180$），重叠就**没有任何收益**；再慢一点就是负收益。这就是 **overlap tax**。

## 3. overlap tax：通信占的不只是时间

通信在 GPU 上不是"另一个维度的免费操作"。它要：

- **SM**：NCCL 的 reduction kernel、NVSHMEM 的拷贝 warp、TMA 的发起线程，都要占 SM 或 SM 的 issue 槽；
- **HBM/L2 带宽**：通信要读写数据，和 GEMM 的 operand/activation 流量抢带宽；
- **功耗/频率**：计算单元和内存系统同时忙，功耗上升，可能触发降频；
- **cache**：通信数据会污染 L2，把本来能命中的 GEMM 数据挤出去。

把这些统一记成**计算减速因子** $\gamma \ge 1$：

$$
T_{\text{compute}}^{\text{overlap}} = \gamma \cdot T_{\text{compute}}
$$

则重叠后的下界是：

$$
T \ge \max(\gamma C, M)
$$

只要 $\gamma C > M$，overlap 的收益就完全被"计算变慢"吃掉；此时**串行反而可能更优**，因为串行时 $\gamma=1$。

Characterizing（[2507.03114](https://arxiv.org/abs/2507.03114)）在 4×A100/H100/MI250/MI210 上测得的 $\gamma$ 是：

- 平均 **1.189**（计算慢 18.9%），最高 **1.40**；
- 串行只比 overlap 平均慢 **10.2%**，最高 26.6%；
- overlap 峰值功耗高 **25%**；H100 平均 38% TDP、峰值 140% TDP。

换句话说：**在很多真实负载上，overlap 的净收益只有 10% 量级，而不是论文标题里的 1.5×–2×。** 那些 1.5×–2× 通常来自"通信本来占 50% 以上"的场景（MoE、长序列 TP），或者来自把通信量本身砍掉（量化、融合）。

> 这不是说 overlap 没用。而是说：**先测 $\gamma$，再决定要不要 overlap。** 如果一层本来通信只占 10%，把 SM 让给通信、把计算拖慢 15%，就是负收益。

## 4. 上界：$\max(C,M)$ 不是总能达到

即使 $\gamma=1$，重叠也有硬上界：

$$
T \ge \max(C, M) + T_{\text{startup}}
$$

- 如果 $M > C$，再怎么 overlap，端到端也不会低于 $M$；此时 $\eta \le C/M$。
- $T_{\text{startup}}$ 是流水线填充/排空、第一个 chunk 的通信、kernel launch 等无法重叠的部分。

这条式子解释了很多"为什么 overlap 到 90% 就不动了"：

- TP 的 all-reduce 比 GEMM 慢时（跨节点、小 batch），$\eta$ 的上界就是 $C/M$，可能只有 40%；
- 流水线的第一个 microbatch 必须等上游，最后一个必须等下游，单步气泡至少 $(p-1)T_F$；
- MoE 的 dispatch 没完成，expert 一个 token 都算不了——第一个 chunk 的 $d$ 必然暴露。

### 4.1 分块能改变什么，不能改变什么

把通信和计算都切成 $n$ 块，理想两级流水：

$$
T = \max(nc, ng) + \min(c,g)
$$

其中 $c$ 是每块计算，$g$ 是每块通信。当 $n \to \infty$：

- $c \ge g$：$T \to nc$，通信被完全藏住，$\eta \to 1$；
- $g > c$：$T \to ng$，通信是瓶颈，$\eta \to c/g < 1$。

**分块只能把"启动/尾部"摊薄，不能把"通信比计算慢"这件事变没。** 这就是为什么跨节点 TP 光靠切 chunk 救不了，必须砍通信量（量化、层次化 all-reduce）或换拓扑。

## 5. 用三个模拟器把账算出来

### 5.1 流水线：1F1B 的 22.58% 从哪来

`schedule_sim.py` 把每个 microbatch 的前向 F、输入梯度 B、权重梯度 W 建成 DAG，按 stage 串行 + 跨 stage 依赖做 list scheduling。取 $p=8,m=24,F=B=W=1$：

```text
$ python3 schedule_sim.py --schedule 1f1b
makespan=93 per_step=93.00 ideal=72 bubble=21 bubble_rate=22.58%
idle per stage: p0=21 p1=21 ... p7=21
```

- 理想工作量：$m(F+B+W)=72$；
- 实际 makespan：$(m+p-1)(F+B+W)=93$；
- 气泡：$21/93=22.58\%$。

这个数和 Zero Bubble 论文 Table 2 的 **24.31%** 很接近（差异来自论文的 $T_F/T_B/T_W$ 不相等）。它说明：**1F1B 的气泡不是"调度没排好"，而是结构性的 $(p-1)$ 个 stage 填充。**

同尺寸下，把 B 拆成 B/W、用 W 填气泡（本系列的 `zbh2` 简化版）：

```text
$ python3 schedule_sim.py --schedule zbh2
makespan=86 bubble=14 bubble_rate=16.28%
```

气泡从 22.58% 降到 16.28%。论文里的完整 ZB-H2 能到 **0.29%**，因为它还会自动重排每个 stage 的 F/B/W 顺序、并把 optimizer 的同步也藏掉；我们的模拟器只做了"W 填 B 的空隙"，所以只拿到一部分收益。这个差距本身就是第二篇的主题。

### 5.2 DP bucket：桶不是越细越好

`bucket_sim.py` 固定总计算 100、总通信 60，扫描 bucket 数 $N$，每桶加 2 个单位的 launch 开销：

```text
 N     c      t   makespan   exposed
 1 100.00  62.00    162.00     62.00
 4  25.00  17.00    117.00     17.00
 8  12.50   9.50    109.50      9.50
16   6.25   5.75    105.75      5.75
32   3.12   3.88    127.12     27.12
64   1.56   2.94    189.56     89.56
```

- $N$ 太小：最后一个桶的通信 $t$ 直接暴露（$N=1$ 时暴露 62）；
- $N$ 太大：launch 开销把 $t$ 推过 $c$，通信开始堆积（$N=32$ 时暴露 27）；
- 最优在 $N=16$ 附近：$t=5.75 < c=6.25$，尾桶刚好被计算盖住。

单通道闭式：

$$
T_{\text{exposed}} = t + (N-1)\max(0, t-c)
$$

这就是 Megatron 的 `bucket_cap_mb` 背后的取舍。第四篇会把它落到 `param_and_grad_buffer.py` 的 `start_grad_sync` / `finish_grad_sync`。

### 5.3 TP 分块：通信比计算慢时，切块也救不了

`overlap_model.py` 的两级流水，取 $c=1.0, g=1.5$（通信比计算慢 50%）：

```text
chunks    total   exposed
     1     2.50      1.50
     2     4.00      2.00
     4     7.00      3.00
     8    13.00      5.00
    16    25.00      9.00
```

切得越细，暴露的**绝对时间**反而越大（因为总通信量不变、通信是瓶颈），$\eta$ 趋近 $c/g=66.7\%$。此时正确的方向不是"继续切"，而是：

- 降低通信量（FP8/INT6 量化、低秩压缩）；
- 换更快的链路/拓扑（NVLink、NVSHARP、in-network collective）；
- 把通信挪到别的层（TP 换成 PP/EP，改变并行策略）。

## 6. 该记哪几个指标

建议每次优化至少记五个数：

| 指标 | 定义 | 回答什么问题 |
|---|---|---|
| `exposed` | $T_{\text{comm}} - T_{\text{overlap}}$ | 还有多少通信在关键路径上 |
| `overlap_ratio` | $T_{\text{overlap}} / T_{\text{comm}}$ | 通信被藏住的比例（分母是通信量） |
| `compute_slowdown` | $T_{\text{compute}}^{\text{overlap}} / T_{\text{compute}}^{\text{serial}}$ | overlap tax 有多大 |
| `bubble_rate` | $T_{\text{bubble}} / T$ | 依赖造成的空转 |
| `goodput` | 有效 token/样本数 / 端到端时间 | 最终用户关心的吞吐 |

只看 `overlap_ratio` 会漏掉后三个；只看端到端 speedup 又不知道收益从哪来。**五个一起看，才能判断一个优化是不是"真收益"。**

`overlap_audit.py` 可以从 trace CSV 直接算前三个：

```text
$ python3 overlap_audit.py --demo
span=160.0 compute=150.0 comm=90.0
overlap=80.0 exposed=10.0 overlap_ratio=88.9% compute_share=93.8%
```

它把区间并集、交集算清楚，避免"两条 stream 有重叠"被误读成"通信全藏住了"。

## 7. 一个决策流程

把上面的东西串成一个可执行的判断顺序：

1. **画 DAG，找关键路径。** 关键路径上没有通信 → 不要做 overlap，去做计算优化/并行策略。
2. **算 $M/C$。** 如果 $M \ll C$，overlap 的收益上限就是 $M$；先确认这个上限值不值得投入。
3. **测 $\gamma$。** 在小规模上开/关 overlap，量计算 kernel 本身的时间；如果 $\gamma C > M$，考虑串行或减少通信量。
4. **选粒度。** 用 `bucket_sim.py` / `overlap_model.py` 估最优 chunk 数；再到真实 trace 上验证。
5. **看尾部。** 第一个 chunk 的通信、最后一个 chunk 的等待、kernel launch，往往决定实际收益。
6. **记录负结果。** 哪个 shape 变慢了、慢多少、为什么；这比只报最好 case 有用。

## 结论

- 通算并行的账本：$T = C + E + B$；重叠效率 $\eta = 1 - E/M$ 只是其中一个数。
- **overlap 有 tax**：通信抢 SM/带宽/功耗，会让计算变慢；Characterizing 测得平均 18.9%。
- **上界是 $\max(\gamma C, M)$**：通信比计算慢时，切块只能摊薄启动/尾部，不能突破 $M$。
- **桶/块不是越细越好**：尾延迟和 launch 开销决定内部最优；DP 的例子最优在 16 个桶。
- 评价一个方案至少看五个数：exposed、overlap_ratio、compute_slowdown、bubble_rate、goodput。

下一篇从最有历史感的地方开始：流水线气泡。GPipe 为什么必须全 F 再全 B，1F1B 怎么把内存从 $m$ 降到 $p$，Zero Bubble 又怎么用 W 把气泡填到接近 0。

---

**系列导航**：[专题首页]({{< relref "comm-overlap-00-index" >}}) · [二：流水线气泡]({{< relref "comm-overlap-02-pipeline" >}})

> 配套代码：`code/comm-overlap/overlap_model.py`、`schedule_sim.py`、`bucket_sim.py`、`overlap_audit.py`
> 自测：`python3 overlap_model.py --test && python3 schedule_sim.py --test && python3 bucket_sim.py --test && python3 overlap_audit.py --test`
