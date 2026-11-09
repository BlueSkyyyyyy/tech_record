---
title: "通算并行研读（十）：实测手册——用 nsys/ncu 把 overlap 账算出来"
date: 2026-11-02
draft: false
weight: 11
series: ["comm-overlap"]
tags: ["通算并行", "系列", "性能分析", "nsys", "ncu", "overlap", "profiling"]
categories: ["训练框架"]
---

第一篇开头留了一个悬念：nsys 上通信和计算叠得很好看，overlap ratio 95%，端到端只快 8%。当时我们说这是"指标选错了"。但还有一个更尴尬的可能：**指标没错，是数字根本没测对**。

把这两个数拆开，差别就出来了。设一层串行时 $C=100$、$M=60$：

- nsys 报告 overlap ratio 95%，即通信只暴露 3；
- 如果 overlap 期间计算被拖慢 45%（$\gamma=1.45$），端到端是 $145+3=148$；
- 串行是 $100+60=160$，所以只省了 **7.5%** 的时间（约 **1.08×**）。

同样一个 95%，可以对应两种完全不同的真相：

1. 通信确实藏住了，但 **overlap tax** 把收益吃光——下一步该去减 SM/带宽争用；
2. 通信藏住了、计算也几乎没受影响，但通信量本来就小——下一步该去砍通信量或换并行策略。

只报 ratio 和 speedup，这两种情况长得一模一样。要把它们分开，必须在**同一份 trace 上同时量出通信暴露了多少、计算被拖慢了多少**。（上面这组数字是算术示例，不是实测；本篇讲的就是怎么在真机上把它量出来。）

这篇是系列的第十篇，做三件事：给出三层测量各测什么、一套能跑的 nsys/ncu 流程、以及一个把 trace 变成三个数的脚本 `overlap_audit.py`。最后是五类 overlap 各自的观测点、七个坑，和一张可以贴在显示器边上的 checklist。

## 1. 三层测量：端到端、时间轴、kernel

同一个"变快了没有"的问题，在三个尺度上有三种问法。

### 1.1 第一层：系统/端到端

指标只有四个，但每个都有坑：

| 指标 | 定义 | 坑 |
|---|---|---|
| `step_time` | 一步训练/一次 decode 的墙钟 | 只报均值会被第一步和 GC 拉偏；报**中位数 + p90** |
| `tokens/s` | 全局 token 数 / step_time | 要写清是 global batch 还是 per-GPU；含不含重计算 |
| `MFU` | $6 \cdot N_{\text{param}} \cdot \text{tokens} / (\text{step\_time} \cdot \text{peak\_flops})$ | 是上界估计，忽略了 attention 的二次项和通信；只适合纵向对比自己 |
| `goodput` | 有效 token（扣掉 padding、重算、失败的 step）/ 墙钟 | 最接近"用户拿到多少"，但需要框架侧打点 |

这一层回答"值不值得做"，不回答"哪里没藏住"。**它最大的价值是给出分母**：后面所有比例都要拿它当基准。

测量纪律：固定 shape / batch / 并行度，关掉 checkpoint、日志、eval，至少跑 20 步取中位数。对比 serial 和 overlap 时，除被测变量外一个都不许动。而且**端到端至少要跑三种配置**：serial、overlap、以及构造出来的 ideal（§5 详述）。只报 overlap 一个数，你无法判断收益是从"藏住通信"来的，还是从"计算本来就快"来的。

### 1.2 第二层：时间轴（nsys）

nsys 给的是 stream × 时间的二维图。它有三个东西是别的工具替代不了的：

- **stream 归属**：哪个 kernel 在计算 stream、哪个在 NCCL 的内部 stream、哪个在你的 side stream；
- **NVTX range**：把 Python 侧的逻辑边界（step、forward、backward、某个 bucket、某个 chunk）钉到 GPU 时间轴上；
- **通信 kernel 的真实形状**：NCCL 的 `AllReduce` / `AllGather` / `ReduceScatter` / `SendRecv`，DeepEP 的 dispatch / combine，以及它们的 grid 大小和持续时长。

但看时间轴最常见的错误，是**把"两条 bar 叠在一起"当成 overlap 成功**。真正的判据是反过来的：

> 找 **compute stream 上没有任何 kernel 的时间段**，再看这段时间里 comm stream 在不在跑。两个条件同时成立，才是通信暴露。

```text
时间 ───────────────────────────────────────────────────────────►
compute  [===GEMM===][===GEMM===]        [===GEMM===][===GEMM===]
comm          [=AR1=]        [====AR2====]
                                  ▲
                                  └── AR2 的尾段：compute 已经空了，
                                      这段才是 exposed；AR2 的前半段是 overlap
```

注意图的读法：AR2 整条都"在跑"，但它只有前半段落在 GEMM 窗口里。**"通信在跑"和"通信被藏住"是两件事**，这也是为什么 ratio 高不等于快。

### 1.3 第三层：kernel（ncu）

ncu 回答的是"这个 kernel 为什么慢"：PC 采样（程序计数器采样）加硬件计数器，逐 kernel replay。分两类看。

对**计算 kernel**，看六个量：

- `sm__throughput.avg.pct_of_peak_sustained_elapsed`：SM 流水线整体利用率，先看它判断是 compute-bound 还是 latency-bound；
- `sm__warps_active.avg.pct_of_peak_sustained_active`：achieved occupancy，和理论 occupancy 的差距说明寄存器/共享内存还是调度的问题；
- warp stall 原因（`smsp__average_warps_issue_stalled_*_per_issue_active.ratio`）：`long_scoreboard` 是等访存，`barrier` 是同步，`mio_throttle` 是共享内存/特殊功能单元排队——**overlap 期间这几项会一起变差**，这是 tax 的微观证据；
- `dram__throughput` 与 `lts__t_sector_hit_rate`：通信数据流过 L2 会把 GEMM 的命中率打下去；
- tensor pipe 利用率（`sm__pipe_tensor_cycles_active`）：判断 Tensor Core 到底有没有喂饱；
- TMA 吞吐：Hopper 之后在 Memory Workload Analysis 里有专门视图，字段名各版本不同，以 `ncu --query-metrics` 为准。

对**通信 kernel**，ncu 的用处有限——replay 会破坏并发关系，你测到的是"单独跑这个 collective"的时间，不是 overlap 时的时间。通信侧真正要的四个量，用 nsys 加算术就够了：

- **传输字节**：从算法和 dtype 算，别从 profiler 猜；
- **实测带宽** = bytes / duration，和链路峰值比（NVLink / IB 各算各的）；
- **活跃 SM 数**：用 nsys 的 active SM 口径，不要用 grid size（grid 只是 CTA 数），这是"通信抢计算资源"的直接证据；
- **launch / 同步开销**：小消息时 duration 远大于 bytes/带宽，差值就是固定开销。

三层的关系可以这样记：

| 层 | 工具 | 回答 | 不回答 |
|---|---|---|---|
| L1 端到端 | 计时器、训练日志 | 值不值得做、有没有回归 | 慢在哪一段 |
| L2 时间轴 | nsys | 哪段暴露、谁在等谁、哪个 stream 空着 | kernel 内部为什么慢 |
| L3 kernel | ncu | 计算被谁拖慢、occupancy/带宽/停顿 | 跨 rank 的时序关系 |

**顺序不能反**：先在 L1 确认有收益空间，再到 L2 找暴露点，最后才用 ncu 钻进那一个 kernel。反过来做，你会花一下午优化一个不在关键路径上的 GEMM。

## 2. 一套能跑的 nsys 流程

### 2.1 抓 trace

```bash
nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o trace python train_step.py
```

每个 flag 的作用：

- `cuda` 必须有（kernel、memcpy、stream、API），`nvtx` 用来标逻辑边界，`osrt` 看线程和系统调用，`cublas`/`cudnn` 给库层标注；
- `--sample=none --cpuctxsw=none` 关掉 CPU 采样和上下文切换记录，显著降低扰动——**默认是开的**，很多人不知道自己的 trace 被拖慢了几个点；
- `--capture-range=cudaProfilerApi` 只抓 `torch.cuda.profiler.start()` 到 `stop()` 之间，天然排除 warmup；
- **flag 名和取值以你的 nsys 版本为准**（`nsys --version`、`nsys profile --help`）。这里给的是模板，不是圣旨。不同年份的 nsys 对 `--trace` 的取值域、`--capture-range-end` 的行为都改过。

### 2.2 先看聚合，再要区间

聚合报表：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum trace.nsys-rep
```

- `cuda_gpu_kern_sum`：每个 kernel 的总时长、次数、平均——找 top 10；
- `cuda_gpu_mem_time_sum`：H2D/D2H 的时间，顺便确认有没有意外的同步拷贝；
- `nvtx_sum`：每个 range 的总时长，用来拿到 step / forward / backward 的边界。

**但这三张表都是聚合值，没有 start/end，算不了 overlap。** 要区间：

```bash
# 每个 kernel 一行，带 Start / Duration / Stream
nsys stats --report cuda_gpu_trace --format csv -o trace_kernels trace.nsys-rep

# 需要和 NVTX 做 join 时，导出 sqlite 用 SQL 关联
nsys export --type sqlite -o trace.sqlite trace.nsys-rep
```

`cuda_gpu_trace` 的列名随版本变化（`Start (ns)` / `Duration (ns)` / `Stream` / `Name` 是常见的四个），**先 `head` 看一眼表头再写映射**。转成 `overlap_audit.py` 需要的四列：

```python
import csv

COMM = ("nccl", "allreduce", "allgather", "reducescatter",
        "alltoall", "sendrecv", "dispatch", "combine")
rows = []
for r in csv.DictReader(open("trace_kernels.csv", newline="", encoding="utf-8")):
    start, dur = float(r["Start (ns)"]), float(r["Duration (ns)"])
    name = r["Name"]
    rows.append({
        "name": name, "stream": r["Stream"],
        "type": "comm" if any(k in name.lower() for k in COMM) else "compute",
        "start": start, "end": start + dur,
    })
```

这个 `type` 判定是**启发式**：按 kernel 名匹配，一定会漏。两个典型例外要手工覆盖：

- fused kernel（比如 AG+GEMM、GEMM+RS）：它既是计算又是通信，最诚实的做法是单独归成 `other` 并在结论里写清"这部分没有计入 comm"；
- 框架自己起的 copy kernel（DeepEP 的 combine 里可能混着 permute）：按名字可能被当成 compute。

**归类错了，后面所有比例都是错的。** 这是整个流程里最需要人工核对的一步，别指望脚本。

### 2.3 一个辅助：给 bucket / chunk 打 NVTX

如果你要测的是 DP 的桶或 MoE 的 chunk，光有 kernel 名不够——同一个 `AllReduce` 会出现几十次，你不知道哪次属于哪个桶。加两行：

```python
torch.cuda.nvtx.range_push(f"bucket_{i}_start_sync")
# start_grad_sync(...)
torch.cuda.nvtx.range_pop()
```

然后用 `nvtx_sum` 拿到每个 range 的时长，用 sqlite join 把 range 和落在里面的 kernel 关联起来。**这一步做了，后面所有"per bucket / per chunk"的指标才有意义。**

### 2.4 跨 rank 的时间基准

每台机器有自己的时钟，nsys 的时间戳是本机的单调时钟。同节点多进程基本可比；**跨节点会有毫秒级偏移，而 overlap 的时间尺度常常就是几十微秒**。直接把两份 trace 贴在一起看，会得出"rank 0 比 rank 7 早发起 3ms"这种不存在的结论。

没法统一时钟的时候，用锚点对齐：同一个 collective 在所有 rank 上几乎同时发起，取它在本端 trace 里的起点当锚，把各自的 trace 平移。这也是为什么 §2.3 的 NVTX 值得打——range 名里带上 step 号，锚点就好找了。真要跨节点细对，得靠 PTP/NTP 把机器时钟压到亚毫秒，并在报告里写明对齐方法。

## 3. `overlap_audit.py`：把截图变成三个数

有了带 start/end 的 CSV，剩下的就是区间运算。`code/comm-overlap/overlap_audit.py` 做的事很简单：

- 输入：`name,stream,type,start,end`，`type ∈ {compute, comm, other}`，时间单位任意但必须一致；
- 把 compute 区间、comm 区间各自求**并集**，再求**交集**；
- 输出 span、compute、comm、overlap、exposed、overlap_ratio、compute_share、每条 stream 的忙时间，以及一行 ASCII 时间轴。

实际运行（`--demo` 内置了 6 个区间）：

```text
$ python3 overlap_audit.py --demo
span=160.0 compute=150.0 comm=90.0
overlap=80.0 exposed=10.0 overlap_ratio=88.9% compute_share=93.8%
streams: comm=90.0 compute=150.0
timeline: CCCCCCCCCCCCCCBBBBBBBBBBBBBCCCCCCCBBBBBBBBBBBBCCCCCCCCCCCCCBBBBBBBBBBBBBBBMMMMM.
          C=compute  M=comm  B=both  .=idle
```

逐个数读一遍，这比看十张截图有用：

- `span=160`：最早 kernel 开始到最晚 kernel 结束。**注意它不是 step time**——step time 还要加上没有 kernel 的 host 空转、Python 循环、优化器同步；
- `compute=150`：三个 GEMM 区间的**并集**，不是 duration 相加；
- `comm=90`：同理，是并集；
- `overlap=80`：两个并集的交集；
- `exposed=10`：`comm - overlap`，通信没被计算盖住的部分。timeline 末尾那 5 个 `M` 大致对应它；但**不要在字符数上做精确对账**——80 个字符铺满 span，是缩放渲染（末尾的 `.` 也是取整），数字以 `span/compute/comm` 为准；
- `overlap_ratio=88.9%`：$80/90$，**分母是 comm**；
- `compute_share=93.8%`：$150/160$，说明时间轴上有 6.2% 的时间没有任何计算 kernel。这个比例由三个数算出，和 timeline 的字符数无关。

一个必须知道的边界：脚本里 `type=other` 的行会进 `streams` 统计，但**不进 span，也不进 compute/comm**。所以如果你把 fused kernel 归成 `other`（§2.2 建议的做法），span 会被低估。要么把这类 kernel 单独算一段再手动加回；要么在报告里注明"span 只覆盖 compute + comm"。**别让一个分类约定悄悄变成结论里的偏差。**

### 为什么必须用并集/交集，而不是把 kernel 时间相加

假设有人直接累加：compute 侧 150，comm 侧 90，"总时间" 240。但 span 只有 160。

| 做法 | compute | comm | 合计 | 误差来源 |
|---|---|---|---|---|
| 直接累加 duration | 150 | 90 | **240** | 重叠的 80 被数了两遍 |
| 并集 / 交集 | 150 | 90 | **span=160** | 无 |

`240 - 160 = 80`，正好等于 overlap。这不是巧合：**跨 stream 的加法会把重叠区间重复计数**，多出来的部分恰好就是重叠量。所以"我把所有 NCCL kernel 加起来，发现通信占了 step 的 60%"这种说法，几乎总是高估——真实的通信时间要么是并集长度，要么是落在关键路径上的那一段。

还有一个容易忽略的点：**`exposed` 不等于"关键路径上的通信"**。它只说明"这段时间没有计算 kernel 在跑"，不说明端到端一定得等它。如果这段通信和后面的计算没有依赖，它可能完全不影响 makespan。要判断关键路径，得回到第一篇的 DAG 视角（[先算清三笔账]({{< relref "comm-overlap-01-performance-model" >}})）：`exposed` 是上界，关键路径上的通信是下界，真实收益在两者之间。

## 4. 五类 overlap，各自该看什么

不同并行策略的暴露点完全不同，用同一套指标会看错。

### 4.1 PP：气泡率和 send/recv 等待

- **stage idle**：把每个 microbatch 的 F/B 用 NVTX 圈住，对每个 stage 算"最后一个 kernel 结束 − 第一个 kernel 开始"再减去 kernel 时间之和，就是 idle；
- **bubble rate**：$\text{bubble rate} = T_{\text{bubble}}/T$。1F1B 的气泡时间是 $(p-1)(T_F+T_B+T_W)$，$p=8,m=24$ 时占 22.58%（第一篇里推过）；实测对不上，说明 $T_F/T_B/T_W$ 不相等，或者调度没排满；
- **send/recv 等待**：nsys 里 P2P 表现为 `ncclDevKernel_SendRecv` 一类的 kernel，它的 duration 通常远大于"字节/带宽"，差值就是同步开销。**要看等待，不要看传输**；
- 一个负结果值得记住：Characterizing 测到 PP 的 slowdown 明显低于 FSDP（A100 上从 0 到 5.4%），因为 send/recv 不像 all-gather/reduce-scatter 那样大量占用 SM。**PP 的通信量小、tax 也小，但可藏的空间同样小。**

### 4.2 DP：桶的就绪延迟和暴露尾

- **bucket ready-to-finish**：在 `start_grad_sync` / `finish_grad_sync` 上打 NVTX，量每个桶从发起到完成的墙钟。这个值应该 ≈ 桶字节 / 有效带宽 + launch，偏差大说明在排队（多个桶挤同一个 NCCL 通道）；
- **exposed tail**：backward 结束后还在跑的 NCCL kernel。这段没人盖，是纯暴露。第二个暴露点是 optimizer 之后的 param all-gather（FSDP/ZeRO）；
- **NCCL duration vs bucket size**：扫 `bucket_cap_mb`，画 duration-vs-size 曲线。小桶由 launch + 同步主导（曲线平），大桶由带宽主导（曲线线性）。**拐点就是 `bucket_sim.py --sweep` 找的那个 $t \approx c$ 附近**，模型和实测应该能对上；
- 扫桶数时只改一个变量，每次至少 20 步取中位数。桶大小会影响显存和通信量，别一次改三个。

具体到 Megatron 的 hook 和计数逻辑，见[数据并行那一篇]({{< relref "comm-overlap-04-data-parallel" >}})。

### 4.3 TP：all-reduce 时间、GEMM 时间、以及"被拖慢了多少"

TP 是五类里最需要"三个时间一起看"的：

- `T_ar`：all-reduce kernel 的 duration；
- `T_gemm`：GEMM kernel 的 duration；
- `T_gemm_overlap`：**同一个 GEMM 在有并发 AR 时的 duration**。

第三个除以第二个再减一（$T_{\text{gemm\_overlap}}/T_{\text{gemm}} - 1$），就是 compute slowdown。**这一步必须在 trace 里挑同名同 shape 的 kernel 比**，不能拿不同 tile 配置的 GEMM 比——那是两个 kernel，比出来的是实现差异不是 tax。

SM 占用要看 nsys 的 **active SM** 口径，不是 grid size：grid 只是 CTA 数，一个 CTA 不一定占满一个 SM（第七篇讲过，分区的粒度其实是 CTA 槽位）。T-CCL 测到 NCCL 一个 128 KB 的 all-gather 平均占 **3.30 个 active SM**，T-CCL 自己的实现只占 **1.32 个**，这些 SM 直接和 GEMM 抢 L2。**再用 ncu 看同一段 GEMM 在 overlap 期间的 `sm__throughput`**：如果它比单跑时掉了一截，说明 SM 是真被抢走了，而不是被调度器挡住了。

Flux 给了这套账一个更干净的定义。它不直接量"通信被藏住多少"，而是先定义**有效通信时间**：

$$
ECT = \text{OverallTime} - GEMM_{\text{non-split}}
$$

再定义 overlap efficiency：

$$
E_{\text{overlap}} = 1 - \frac{ECT_{\text{overlap}}}{ECT_{\text{non-overlap}}}
$$

好处是：切碎 GEMM 带来的计算变慢、没藏住的通信、launch 开销，**全部合并进 $ECT$ 这一项**，正好是"我们希望 overlap 消掉的东西"；坏处是 $GEMM_{\text{non-split}}$ 必须用同一套最快的 GEMM kernel 测，否则基线不可比。这也是它坚持所有方法都用同一份 GEMM 实现的原因。

一个容易漏的观察：**AG 和 RS 的可重叠性不一样**。Flux 发现切碎后的 GEMM 能通过 GPU multiplexing 和 all-gather 并发，但 reduce-scatter 不行（它自带 reduction，吃更多 SM）。所以同一个方案在 AG 上好看，在 RS 上可能完全没效果——**测 TP 时 AG 和 RS 要分开测**。

### 4.4 MoE：三段、chunk、routing skew

MoE 的账是三段流水：

- 单 chunk：$T = d + c + b$（dispatch、expert 计算、combine）；
- 多 chunk：$T \approx \max(\sum c, \sum d + \sum b) + T_{\text{startup}}$。

要测的：

- **每段的时长**：按 chunk 分别计时，DeepEP 的 `EventOverlap` 或者自己打 NVTX。三段比例决定了理论上限——如果 $\sum d + \sum b > \sum c$，再怎么排也藏不完；
- **通信占的 SM 数**：dispatch/combine 的 active SM（nsys 口径；grid size 只能当 CTA 数参考）。Comet 论文里 MoE 层通信占 47%，靠的就是 CTA specialization 把通信压到固定几个 SM；
- **routing skew**：每个 expert 分到的 token 数的 `max/mean`。**这个 profiler 给不了**，要在 router 之后打点。skew 大时最热的 expert 变成关键路径，其它 expert 的 SM 空转，此时"通信被藏住"没有任何意义——瓶颈在计算的不均衡；
- **tail chunk**：最后一个 chunk 的 combine 必然暴露，没有任何计算能盖住它。chunk 越多 tail 越小，但 startup 越大；Syncopate 扫到的最优在 2–3 个 chunk，这个数很值得在自己的 workload 上重扫一遍。

三段式和 chunk 粒度的细节在 [MoE 那一篇]({{< relref "comm-overlap-06-moe" >}})。

### 4.5 推理：TTFT、TPOT 和 wave 数

- **TTFT**：prefill 的首 token 时间，通信主要在 TP 的 all-reduce 上；
- **TPOT**：单请求两个输出 token 之间的间隔（inter-token latency），**这是最容易骗人的指标**——batch 增大不会把它"摊薄"：并发请求只会争抢 HBM 带宽和 SM，TPOT 通常持平或变差。所以报 TPOT 必须同时报 batch size 和并发数；
- **tokens/s**：吞吐，和 TPOT 一起看才能判断是延迟优化还是吞吐优化；
- **wave 数**：decode 的 GEMM 常常一个 wave 就结束，SM 本来就不满。此时"让 SM 给通信"代价小，但"计算能藏住通信"的空间也小；
- **decode 的 overlap 阈值**：batch 越小，GEMM 越瘦，通信相对越贵，overlap 的收益阈值随之上移。`decode_overlap.py` 可以扫这条曲线。

抓法上有一条硬规矩：**decode 的 trace 只抓几十个 step**，用 `--capture-range` 精确框住。decode 的单步很短，抓上千步会让 trace 大到没法分析，时间戳精度也会变差。推理侧的完整讨论在[推理那一篇]({{< relref "comm-overlap-09-inference" >}})。

## 5. overlap tax 怎么量：必须跑三个配置

tax 是"通信在跑的时候，计算变慢了多少"。它不能从一份 trace 里读出来，必须对比：

1. **serial**：通信后立即 wait（同一个 stream，或显式 `synchronize`），确保不重叠；
2. **overlap**：正常配置；
3. **ideal**：**不是跑出来的，是算出来的**。

Characterizing 的构造是：

$$
E2E_{\text{ideal}} = E2E_{\text{overlap}} - (C_{\text{overlap}} - C_{\text{serial}})
$$

也就是把"计算被拖慢的那部分"从 overlap 的时间里减掉，得到"通信存在但不干扰计算"的假想时间。然后

$$
\gamma = \frac{C_{\text{overlap}}}{C_{\text{serial}}}
$$

$\gamma$ 用**同名同 shape 的 kernel duration 中位数**算，别用总时间——总时间里混着气泡和调度噪声。

也可以补一个"假通信"版本：把 collective 换成等长的空转 kernel，看计算是否还慢。但要注意这已经不是同一个负载（空转不占带宽、不吃 L2），只能用来分离"SM 争用"和"带宽争用"。

### 5.1 功耗和频率

tax 的物理来源是资源和功耗，所以必须同时记：

- 功耗：`nvidia-smi dmon -s pucvmet -d 1`，一秒一条，看趋势够用。NVML（NVIDIA 管理库）的采样粒度是 100ms 量级（Characterizing 用 NVML 100ms、AMD-SMI 20ms），**抓不到微秒级尖峰**——想看尖峰只能靠 nsys 的高分辨率采样或专门的功率计；
- 频率：`nvidia-smi -q -d CLOCK`，或从 ncu 的 `sm__cycles_elapsed.avg.per_second` 换算 SM 实际频率；
- 锁不锁频：锁了（`nvidia-smi -lgc`）结论可复现但不代表生产；不锁则同一份代码两次跑可能差几个点。**报告里写清锁没锁，比锁了更重要。**

### 5.2 该记住的几个数

Characterizing（[2507.03114](https://arxiv.org/abs/2507.03114)，4×A100/H100/MI250/MI210，GPT/LLaMA，25 次平均）：

- overlap 让计算平均慢 **18.9%**，最高 **40.0%**（相对"通信完全不干扰计算"的假想基线）；
- 串行只比 overlap 慢 **10.2%**，最高 26.6%；
- overlap 的峰值功耗最多高 **25%**；H100 平均 38% TDP、峰值 140% TDP；
- 严格功耗帽下，overlap 的执行时间最多增加 **107%**；在 4×A100 上把帽压到 100W 时，计算 slowdown 达到 100%——**也就是翻倍**。

最后一条最值得琢磨：它说明 overlap 的收益依赖"有电可用"。当 GPU 已经贴着 TDP 跑，通信和计算抢的是同一个功耗预算，overlap 就从"藏时间"变成了"抢功率"。这也解释了为什么同一份代码在实验室和机房跑出来的收益不一样。

## 6. 七个坑

1. **分母搞错。** `overlap_ratio` 的分母是**通信量**，不是端到端。100% 只说明"通信全落在计算窗口里"，不说明变快——计算被拖慢 40% 时，100% 也可以更慢。
2. **把 kernel 时间相加。** 见 §3：跨 stream 相加会把重叠区间数两遍，多出来的正好是 overlap。
3. **nsys 自身的开销和时钟漂移。** 默认的 CPU 采样和 ctxsw 会扰动；不锁频时 GPU 会 boost 或降频；trace 越长扰动越大。要么锁频，要么两种配置各跑两次报区间。
4. **CUDA Graph 会藏 launch。** capture 之后 CPU 侧 launch 消失，nsys 里只剩 `cudaGraphLaunch` 和它的内部节点；通信可能被 capture 到不同的 stream/节点上，按名字匹配容易错位。用 graph 的推理框架测到的"launch 开销"会系统性偏低。
5. **NCCL 的 stream 归属。** NCCL 用自己创建的内部 stream，不是 PyTorch 的 current stream；nsys 里表现为另一个 stream id。P2P 的 send/recv 只看到本端那一半，跨 rank 的"对端"要另抓一份 trace 对时间。
6. **warmup。** 第一步包含 context 创建、cuBLAS autotune、显存池扩张。至少丢前 5–10 步；最好用 `--capture-range` 精确框住稳态。
7. **"ideal" 是构造的。** 任何 ideal 基线都要写清公式和假设：减掉了什么、没减什么、$GEMM_{\text{non-split}}$ 用哪份实现。**把构造出来的 ideal 当实测报，是最常见的夸大来源。**

## 7. 一页 checklist

下表是经验值，**不是标准**。不同模型、不同互联、不同 batch 都会移动这些线，先用它们判断"要不要继续查"，不要用来判断"合不合格"。

| 指标 | 工具 | 怎么取 | 经验阈值（rule of thumb） |
|---|---|---|---|
| `exposed / e2e` | `overlap_audit.py` | 并集/交集 | < 5% 好；> 15% 值得动手 |
| `overlap_ratio` | 同上 | $overlap/comm$ | > 80% 常见，但**单独看没有意义** |
| `compute_slowdown` | nsys 同名 kernel 对比 | $C_{\text{ov}}/C_{\text{serial}} - 1$ | < 5% 好；> 15% 先查 SM/带宽争用 |
| overlap 期间 SM active | ncu | `sm__throughput` | 比无通信时掉 > 10% 要警觉 |
| NCCL 实测带宽 | nsys | bytes / duration | < 链路峰值 70% 查拓扑/消息大小 |
| `bubble_rate` | NVTX + stage 时间 | $T_{\text{bubble}}/T$ | PP 下 < 10% 好 |
| tail exposed | 时间轴 | 最后一个 bucket/chunk | 越小越好；扫桶数/chunk 数看凹陷 |
| 峰值功耗 / 频率 | `nvidia-smi dmon` | 1s 采样 | 贴 TDP 或掉频 → tax 会被放大 |
| TTFT / TPOT | 框架日志 | 中位数 + p90 | 必须同时报 batch size |

判读顺序固定：**先 L1 确认有空间 → 再 `exposed/e2e` 找暴露 → 再 $\gamma$ 确认 tax → 最后才进 ncu 看单个 kernel。** 三个都小的时候，别再调 overlap 了，去优化计算或换并行策略——这时候 overlap 已经不是瓶颈。

具体到一次排查，可以压成这样一条流水线：

1. 跑 serial 和 overlap 各 20 步，记 step time 中位数、tokens/s、峰值功耗；
2. 用 `--capture-range` 抓 overlap 的 3–5 个稳态 step，导出 `cuda_gpu_trace` 的 CSV；
3. 跑一次 `overlap_audit.py`，拿到 span / exposed / ratio / compute_share；
4. 在时间轴上找最长的 compute 空档：如果那段是通信，进第 5 步；如果不是，去查依赖和气泡；
5. 用同名 GEMM 的 duration 算 $\gamma$，超过 1.15 就先查 SM 占用和 L2 命中率；
6. 只改一个变量（桶数、chunk 数、SM 预算、batch），回到第 1 步。

六步里最容易被跳过的是第 1 步和第 5 步，而它们恰好决定结论对不对：跳过第 1 步就不知道分母，跳过第 5 步就分不清"通信没藏住"和"计算被拖慢"。

## 结论

- **overlap 好声称、难测量**：ratio 高只说明"通信落在计算窗口里"，不说明关键路径变短。必须同时量 exposed 和 compute slowdown。
- **三层分工**：端到端回答值不值得，nsys 回答哪里暴露，ncu 回答 kernel 为什么慢。顺序不能反。
- **区间必须用并集/交集**：跨 stream 累加 duration 会把重叠部分数两遍，误差恰好等于 overlap。
- **tax 要单独测**：serial / overlap / ideal 三个配置，$\gamma = C_{\text{ov}}/C_{\text{serial}}$，同时记功耗和频率；Characterizing 的量级是计算慢 18.9%、串行只慢 10.2%、100W 帽下时间翻倍。
- **五类 overlap 各有各的账**：PP 看气泡和 P2P 等待，DP 看桶尾和 launch，TP 看 AG/RS 分开的 $ECT$，MoE 看 skew 和 tail chunk，推理看 batch 和 wave。
- **七个坑里最致命的是分类**：fused kernel、CUDA Graph、NCCL 内部 stream，任何一个归错类，后面所有比例都是错的。

下一篇是最后一篇：把十二篇的结论收成一张 2026 年的地图，列出五个还没解决的问题——通信能不能再便宜一个数量级、SM 分区要不要动态、overlap 的收益怎么在集群层面复现、编译器能不能自动选出粒度、以及当互联变成瓶颈时这套账本还成不成立（[总结篇]({{< relref "comm-overlap-11-summary" >}})）。

---

**系列导航**：[九：推理侧]({{< relref "comm-overlap-09-inference" >}}) · [十一：总结]({{< relref "comm-overlap-11-summary" >}})

> 配套代码：`code/comm-overlap/overlap_audit.py`
> 自测：`python3 overlap_audit.py --test && python3 overlap_audit.py --demo`
