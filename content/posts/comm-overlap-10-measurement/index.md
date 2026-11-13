---
title: "通算并行研读（十）：重叠率能算，省下的和被拖慢的得分开量"
date: 2026-11-02
draft: false
weight: 11
series: ["comm-overlap"]
tags: ["通算并行", "系列", "性能分析", "nsys", "ncu", "overlap", "profiling"]
categories: ["训练框架"]
---

**把重叠率这一个数换成暴露时长和拖慢倍数两个数；代价是多跑 serial 和 ideal 两个配置，而且顺序不能反：先端到端，再时间轴，最后 kernel**。

读完你会知道：

- 三层工具各答什么问题，什么时候不该往下钻；
- 一份 trace 怎么压成 span、compute、comm 三个数，为什么必须用并集而不是相加；
- 拖慢倍数怎么用三个配置量出来，五类 overlap 各自盯哪个数。

## 1. 时间轴叠得好看，只换来一点点提速

某个团队在 nsys 上看到通信和计算叠得很好看：overlap ratio 95%，端到端只快 8%。两种解释：指标选错了，或者指标没错、数字根本没测对。

拆开算。设一层串行时计算 $C=100$、通信 $M=60$：

- ratio 95% 意味着通信只暴露 3；
- 若 overlap 期间计算被拖慢 45%（$\gamma=1.45$），端到端是 $145+3=148$；
- 串行是 $100+60=160$，只省 7.5%，约 1.08×。

同一个 95% 可以对应两种完全不同的真相：

1. 通信确实藏住了，但 overlap tax 把收益吃光——下一步该去减 SM 和带宽争用；
2. 计算几乎没受影响，但通信量本来就小——下一步该去砍通信量或换并行策略。

只报 ratio 和 speedup，这两种长得一模一样。要把它们分开，必须在**同一份 trace** 上同时量出通信暴露了多少、计算被拖慢了多少。

## 2. 三层工具：先在 L1 确认有空间，再往下钻

顺序不能反：先在 L1 确认有收益空间，再到 L2 找暴露点，最后才用 ncu 钻进那一个 kernel。反过来做，你会花一下午优化一个不在关键路径上的 GEMM。

| 层 | 工具 | 回答什么 | 什么时候用 |
|---|---|---|---|
| L1 端到端 | 计时器、训练日志 | 值不值得做、有没有回归 | 每次改动后 |
| L2 时间轴 | nsys | 哪段暴露、谁在等谁、哪个 stream 空着 | L1 看到有空间 |
| L3 kernel | ncu | 计算被谁拖慢、occupancy 和停顿 | 已定位到单个 kernel |

L1 只有四个指标，每个都有坑：

| 指标 | 怎么取 | 坑 |
|---|---|---|
| `step_time` | 一步训练或一次 decode 的墙钟 | 只报均值会被第一步和 GC 拉偏，报中位数加 p90 |
| `tokens/s` | 全局 token 数除 step_time | 写清 global 还是 per-GPU、含不含重计算 |
| `MFU` | 6 乘参数量乘 token 数，再除以 step 墙钟和峰值算力 | 上界估计，只适合自己跟自己纵向比 |
| `goodput` | 有效 token（扣掉 padding、重算、失败 step）除墙钟 | 最接近用户拿到多少，但要框架侧打点 |

这一层回答"值不值得做"，不回答"哪里没藏住"，它最大的价值是给出分母——后面所有比例都除它。测量纪律：固定 shape、batch、并行度，关掉 checkpoint、日志、eval，至少跑 20 步取中位数；对比 serial 和 overlap 时，除被测变量外一个都不许动。

L2 给的是 stream × 时间的二维图，三样东西别的工具替代不了：stream 归属（计算 stream、NCCL 内部 stream、你的 side stream）、NVTX 逻辑边界（step、forward、backward、某个桶、某个 chunk）、通信 kernel 的真实形状（NCCL 的 AllReduce/AllGather/ReduceScatter/SendRecv，DeepEP 的 dispatch/combine，连同 grid 和时长）。

看时间轴最常见的错，是把两条 bar 叠在一起当成 overlap 成功。判据是反过来的：**找 compute stream 上没有任何 kernel 的时间段，再看那段 comm stream 在不在跑**；两个条件同时成立，才是通信暴露。

```text
compute  [===GEMM===][===GEMM===]        [===GEMM===][===GEMM===]
comm          [=AR1=]        [====AR2====]
                                  ▲
                                  └── AR2 的尾段：compute 已经空了，这段才是 exposed
```

AR2 整条都"在跑"，但只有前半段落在 GEMM 窗口里。**通信在跑和通信被藏住是两件事**，这就是 ratio 高不等于快的原因。

L3 的 ncu 回答"这个 kernel 为什么慢"：PC 采样加硬件计数器，逐 kernel replay。先看两个利用率——SM 流水线（`sm__throughput`）判断是 compute-bound 还是 latency-bound，achieved occupancy（`sm__warps_active`）和理论值的差距说明问题在寄存器、共享内存还是调度。

再看三处旁证：warp stall 原因（`long_scoreboard` 等访存、`barrier` 等同步、`mio_throttle` 排队，overlap 期间这几项一起变差就是 tax 的微观证据）、`dram__throughput` 与 L2 命中率（通信数据流过 L2 会把 GEMM 的命中率打下去）、tensor pipe 与 TMA 吞吐（字段名以 `ncu --query-metrics` 为准）。

对通信 kernel，ncu 用处有限：replay 会破坏并发关系，你测到的是单独跑这个 collective 的时间，不是 overlap 时的时间。通信侧真正要的四个量，用 nsys 加算术就够：传输字节（从算法和 dtype 算，别问 profiler）、实测带宽（bytes 除 duration，和链路峰值比）、活跃 SM 数（用 nsys 的 active SM 口径，不是 grid——grid 只是 CTA 数）、launch 和同步开销（小消息时 duration 远大于 bytes 除带宽，差值就是固定开销）。

## 3. 一条能跑的流程：抓 trace、聚合、要区间、归类

先抓 trace：

```bash
nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn \
    --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop \
    -o trace python train_step.py
```

- `cuda` 必须有（kernel、memcpy、stream、API），`nvtx` 标逻辑边界，`osrt` 看线程和系统调用，`cublas`/`cudnn` 给库层标注；
- `--sample=none --cpuctxsw=none` 关掉 CPU 采样和上下文切换记录，显著降低扰动——这两个默认是开的，很多人不知道自己的 trace 被拖慢了几个点；
- `--capture-range=cudaProfilerApi` 只抓 `torch.cuda.profiler.start()` 到 `stop()` 之间，天然排除 warmup；手工抓就得至少丢掉前 5–10 步，第一步里还混着 context 创建和 cuBLAS autotune；
- flag 名和取值以你的 nsys 版本为准（`nsys --version`、`nsys profile --help`），不同年份对 `--trace` 的取值域、`--capture-range-end` 的行为都改过。

聚合报表三张：

```bash
nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum trace.nsys-rep
```

`cuda_gpu_kern_sum` 给每个 kernel 的总时长、次数、平均（找 top 10），`cuda_gpu_mem_time_sum` 给 H2D/D2H 时间（顺便确认有没有意外的同步拷贝），`nvtx_sum` 给每个 range 的总时长（拿到 step 和 forward/backward 的边界）。**但三张都是聚合值，没有 start/end，算不了 overlap**。要区间：

```bash
nsys stats --report cuda_gpu_trace --format csv -o trace_kernels trace.nsys-rep
nsys export --type sqlite -o trace.sqlite trace.nsys-rep
```

`cuda_gpu_trace` 的列名随版本变化（`Start (ns)`、`Duration (ns)`、`Stream`、`Name` 是常见的四个），先 `head` 看一眼表头再写映射；要和 NVTX 做 join 就导出 sqlite，用 SQL 关联。

**归类是整条流程里唯一必须人工核对的一步**。按 kernel 名匹配 comm 一定会漏，两个典型例外：fused kernel（AG+GEMM、GEMM+RS 这类既是计算又是通信的），和框架自己起的 copy kernel（DeepEP 的 combine 里可能混着 permute）。归错了，后面所有比例都是错的。

要测 DP 的桶或 MoE 的 chunk，光有 kernel 名不够——同一个 `AllReduce` 会出现几十次。在 `start_grad_sync` 前后压两行 NVTX（`range_push(f"bucket_{i}_start_sync")` 和 `range_pop`），用 `nvtx_sum` 拿 range 时长，再用 sqlite join 把 range 和落在里面的 kernel 关联起来；不然 per bucket、per chunk 的指标没意义。

跨 rank 还有一条：nsys 的时间戳是本机单调时钟，同节点多进程基本可比，**跨节点会有毫秒级偏移，而 overlap 的时间尺度常常就是几十微秒**。两份 trace 直接贴在一起，会得出"rank 0 比 rank 7 早发起 3ms"这种不存在的结论。没法统一时钟就用锚点对齐：同一个 collective 在所有 rank 上几乎同时发起，取它在本端 trace 的起点当锚平移；真要细对，靠 PTP/NTP 把机器时钟压到亚毫秒，并在报告里写明对齐方法。

## 4. 并集和交集都要算，相加一定错

拿到带 start/end 的 CSV，剩下的就是区间运算。`overlap_audit.py` 只做一件事：输入 `name,stream,type,start,end`（`type` 只能是 compute、comm、other，时间单位任意但要一致），把 compute 区间、comm 区间各自求并集再求交集，输出 span、compute、comm、overlap、exposed、overlap_ratio、compute_share 和一行 ASCII 时间轴。

内置 `--demo` 有 6 个区间，它给出的就是 span、compute、comm 这三个数，以及由它们推出来的比例。几个输出这样读：

- `span`：最早 kernel 开始到最晚 kernel 结束。**它不是 step time**，step time 还要加上没有 kernel 的 host 空转、Python 循环、优化器同步；
- `compute`：compute 区间的**并集**，不是 duration 相加；
- `comm`：同理，是并集；
- `overlap` 是两者交集，`exposed = comm - overlap` 是通信没被计算盖住的部分（demo 上是 10），`overlap_ratio = overlap / comm`（**分母是 comm**），`compute_share = compute / span`。

有人图省事直接累加：compute 150、comm 90，合计 240，可 span 只有 160。

| 做法 | 合计 | 误差来源 |
|---|---|---|
| 直接累加 duration | 240 | 重叠的 80 被数了两遍 |
| 并集 / 交集 | 160 | 无 |

240 减 160 等于 80，正好是 overlap。这不是巧合：**跨 stream 的加法会把重叠区间重复计数，多出来的部分恰好就是重叠量**。所以"我把所有 NCCL kernel 加起来，发现通信占了 step 的 60%"这种说法，几乎总是高估——真实的通信时间要么是并集长度，要么是落在关键路径上的那一段。

同一个 demo 上 ratio 是 80/90 = 88.9%，compute_share 是 150/160 = 93.8%，也就是说 6.2% 的时间里一个计算 kernel 都没有。ratio 高只说明通信落进了计算窗口：**ratio 100% 也可以更慢，计算被拖慢 40% 时就是更慢**。

有两个边界要记住。一是 `exposed` 不等于"关键路径上的通信"：它只说明这段时间没有计算 kernel 在跑，如果这段通信和后面的计算没有依赖，它可能完全不影响 makespan。回到[第一篇]({{< relref "comm-overlap-01-performance-model" >}})的 DAG 视角，`exposed` 是上界，关键路径上的通信是下界，真实收益在两者之间。

二是 `type=other` 的行会进 `streams` 统计，但**不进 span，也不进 compute 和 comm**。按推荐做法把 fused kernel 归成 `other`，span 就会低估：要么把这类 kernel 单独算一段手动加回，要么在报告里注明"span 只覆盖 compute 加 comm"。别让一个分类约定悄悄变成结论里的偏差。

## 5. 拖慢倍数要对比三个配置才量得出来

tax 是"通信在跑的时候计算变慢了多少"。它不能从一份 trace 里读出来，必须对比三个配置：

1. **serial**：通信后立即 wait（同一个 stream，或显式 synchronize），确保不重叠；
2. **overlap**：正常配置；
3. **ideal**：**不是跑出来的，是算出来的**。

Characterizing 的构造，是把"计算被拖慢的那部分"从 overlap 时间里减掉：

$$
E2E_{\text{ideal}} = E2E_{\text{overlap}} - (C_{\text{overlap}} - C_{\text{serial}})
$$

减掉的正是计算被拖慢的那部分，剩下的是"通信存在但不干扰计算"的假想时间。拖慢倍数就是

$$
\gamma = \frac{C_{\text{overlap}}}{C_{\text{serial}}}
$$

$\gamma$ 用**同名同 shape 的 kernel duration 中位数**算，别用总时间——总时间里混着气泡和调度噪声。也可以补一个"假通信"版本：把 collective 换成等长的空转 kernel，看计算是否还慢；但空转不占带宽、不吃 L2，这已经不是同一个负载，只能用来分离 SM 争用和带宽争用。

tax 的物理来源是资源和功耗，所以功耗和频率必须同时记：

- 功耗：`nvidia-smi dmon -s pucvmet -d 1`，一秒一条看趋势够用；NVML 的采样粒度是 100ms 量级（Characterizing 用 NVML 100ms、AMD-SMI 20ms），抓不到微秒级尖峰，要尖峰只能靠 nsys 的高分辨率采样或专门的功率计；
- 频率：`nvidia-smi -q -d CLOCK`，或从 `sm__cycles_elapsed.avg.per_second` 换算 SM 实际频率；
- 锁不锁频（`nvidia-smi -lgc`）：锁了结论可复现但不代表生产，不锁则同一份代码两次跑可能差几个点。**报告里写清锁没锁，比锁了更重要**。

Characterizing（[2507.03114](https://arxiv.org/abs/2507.03114)，4×A100/H100/MI250/MI210，GPT/LLaMA，25 次平均）的量级：

- overlap 让计算平均慢 **18.9%**，最高 **40.0%**（相对"通信完全不干扰计算"的假想基线）；
- 串行只比 overlap 慢 **10.2%**，最高 **26.6%**；
- overlap 的峰值功耗最多高 **25%**；H100 平均 38% TDP、峰值 140% TDP；
- 严格功耗帽下，overlap 的执行时间最多增加 **107%**；4×A100 上把帽压到 100W，计算 slowdown 达到 100%——也就是翻倍。

最后一条最值得琢磨：overlap 的收益依赖"有电可用"。GPU 已经贴着 TDP 跑时，通信和计算抢的是同一个功耗预算，overlap 就从"藏时间"变成了"抢功率"——同一份代码在实验室和机房里收益不一样，原因常常在这里。

## 6. 五类 overlap，各自盯一个数

不同并行策略的暴露点完全不同，用同一套指标会看错。

| 类型 | 怎么量 | 一个能对照的数 |
|---|---|---|
| PP | NVTX 圈住每个 microbatch 的 F/B，算 stage idle；P2P 看等待不看传输 | 1F1B 气泡 22.58%（p=8、m=24） |
| DP | `start_grad_sync` 和 `finish_grad_sync` 上打 NVTX 量每个桶；扫 `bucket_cap_mb` 画 duration 对 size | 拐点在 t≈c 附近 |
| TP | 挑同名同 shape 的 GEMM 比 duration；active SM 用 nsys 口径 | 128 KB 的 all-gather 占 3.30 个 active SM |
| MoE | 按 chunk 分别计时三段；router 之后打点算 skew；扫 chunk 数 | Comet 里 MoE 通信占 47% |
| 推理 | TTFT、TPOT 取中位数加 p90（[推理那篇]({{< relref "comm-overlap-09-inference" >}})）；`--capture-range` 只框几十个 step | TPOT 必须同时报 batch 和并发数 |

四件跨类的事，表里放不下：

- **PP 和 DP 的账都容易看歪**：PP 的负结果是 slowdown 明显低于 FSDP（A100 上从 0 到 5.4%），因为 send/recv 不像 all-gather 那样大量占 SM——通信量小、tax 小，可藏的空间同样小。DP 则是小桶由 launch 和同步主导（曲线平）、大桶由带宽主导（曲线线性）；扫桶时只改一个变量，每次至少 20 步取中位数，hook 和计数逻辑见[数据并行那篇]({{< relref "comm-overlap-04-data-parallel" >}})。
- **TP 要看三个时间**：$T_{\text{ar}}$、$T_{\text{gemm}}$、以及同一个 GEMM 在有并发 AR 时的 $T_{\text{gemm\_overlap}}$，第三个除第二个再减一就是 compute slowdown，必须同名同 shape 比。再用 ncu 看同一段 GEMM 的 `sm__throughput` 掉没掉——掉了说明 SM 真被抢走，不是被调度器挡住。
- **AG 和 RS 要分开测**：Flux 发现切碎后的 GEMM 能和 all-gather 并发，和 reduce-scatter 不行（后者自带归约，吃更多 SM）。T-CCL 自己的实现只占 1.32 个 active SM，这个差就是能抢回来的空间。
- **MoE 的账在三段流水上**：单 chunk 是 $d+c+b$，多 chunk 约是 $\max(\sum c, \sum d + \sum b)$ 再加一个 startup；若 $\sum d+\sum b > \sum c$，再怎么排也藏不完。routing skew（每个 expert 分到 token 数的 max/mean）profiler 给不了，要在 router 之后自己打点——skew 大时最热 expert 就是关键路径，这时"通信被藏住"没有任何意义。最后一个 chunk 的 combine 也必然暴露，chunk 越多 tail 越小、startup 越大，Syncopate 扫到的最优在 2–3 个 chunk（三段式的推导见 [MoE 那篇]({{< relref "comm-overlap-06-moe" >}})）。

一个可以借用的合并口径：Flux 不直接量藏住多少，先定义有效通信时间 $ECT = \text{OverallTime} - GEMM_{\text{non-split}}$，再算 $E_{\text{overlap}} = 1 - ECT_{\text{overlap}}/ECT_{\text{non-overlap}}$。切碎 GEMM 的变慢、没藏住的通信、launch 开销全被合并进 $ECT$ 一项，正好是"我们希望 overlap 消掉的东西"；代价是 $GEMM_{\text{non-split}}$ 必须用同一套最快的 GEMM 测，否则基线不可比。

## 7. 四个坑，每一个都能让结论反向

1. **nsys 自身的开销和时钟漂移**。默认的 CPU 采样和 ctxsw 会扰动；不锁频时 GPU 会 boost 或降频；trace 越长扰动越大。要么锁频，要么两种配置各跑两次报区间。
2. **CUDA Graph 会藏 launch**。capture 之后 CPU 侧 launch 消失，nsys 里只剩 `cudaGraphLaunch` 和它的内部节点；通信可能被 capture 到不同的 stream，按名字匹配容易错位，用 graph 的推理框架测到的 launch 开销会系统性偏低。
3. **NCCL 的 stream 归属**。NCCL 用自己创建的内部 stream，不是 PyTorch 的 current stream，nsys 里表现为另一个 stream id；P2P 的 send/recv 只看到本端那一半，跨 rank 的对端要另抓一份 trace 对时间。
4. **ideal 是构造的**。任何 ideal 基线都要写清公式和假设：减掉了什么、没减什么、$GEMM_{\text{non-split}}$ 用哪份实现。把构造出来的 ideal 当实测报，是最常见的夸大来源。

## 8. 一页 checklist：六步，每步一个门槛

1. **跑三个配置**：serial、overlap、ideal。各 20 步取中位数，记 step time、tokens/s、峰值功耗和频率。跳过这步就不知道分母。
2. **抓 3–5 个稳态 step**，导出 `cuda_gpu_trace` 的 CSV，把 kernel 归成 compute、comm、other，人工核对 fused kernel。
3. **算 `exposed/e2e`**：小于 5% 算好，大于 15% 值得动手；`overlap_ratio` 大于 80% 很常见，但单看它没有意义。
4. **找最长的 compute 空档**：那段是通信就进第 5 步；不是就去查依赖和气泡（PP 下 `bubble_rate` 小于 10% 算好）。
5. **算 $\gamma$**：同名 GEMM 的 $C_{\text{ov}}/C_{\text{serial}}$，慢不到 5% 算好，超过 15%（$\gamma$ 大于 1.15）先查 SM 占用（`sm__throughput` 比无通信时掉超过 10%）和 NCCL 带宽（低于链路峰值 70% 就去查拓扑和消息大小）。跳过这步就分不清"通信没藏住"和"计算被拖慢"。
6. **只改一个变量**（桶数、chunk 数、SM 预算、batch），回第 1 步。

判读顺序和三层工具是同一个：L1 确认有空间 → `exposed/e2e` 找暴露 → $\gamma$ 确认 tax → 最后才进 ncu 看单个 kernel。三个都小的时候别再调 overlap 了，去优化计算或换并行策略——这时 overlap 已经不是瓶颈。

> 上面这些线是经验值，不是标准。不同模型、互联、batch 都会移动它们，用来判断"要不要继续查"，不要用来判断"合不合格"。

## 只记一件事

**overlap 好声称、难测量**：ratio 高只说明通信落在计算窗口里，不说明关键路径变短。真正要报的是两个数——`exposed` 告诉你亏在哪，$\gamma$ 告诉你赢在哪。

最后一篇收口：把全系列的结论收成一张 2026 年的地图，列出五个还没解决的问题——什么时候不该 overlap、跨节点还差多远、功耗算不算第四条约束、静态分区碰上动态 routing、手写 kernel 与编译器路线什么时候合得上（[总结篇]({{< relref "comm-overlap-11-summary" >}})）。

---

**系列导航**：[九：推理侧]({{< relref "comm-overlap-09-inference" >}}) · [十一：总结]({{< relref "comm-overlap-11-summary" >}})

> 配套代码：`code/comm-overlap/overlap_audit.py`
> 自测：`python3 overlap_audit.py --test && python3 overlap_audit.py --demo`

> 口径：开头那组拆账的数是算术示例，不是实测；`overlap_audit.py` 只处理单机导出的四列 kernel 区间。
> 跨节点要先做时钟对齐（§3 的锚点法），`type=other` 造成的 span 缺口要手工补。
