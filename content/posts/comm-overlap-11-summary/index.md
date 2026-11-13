---
title: "通算并行研读（总结）：一张按粒度分层的地图"
date: 2026-11-02
draft: false
weight: 16
series: ["comm-overlap"]
tags: ["通算并行", "系列", "总结", "分布式训练", "推理", "开放问题"]
categories: ["训练框架"]
---

**通算并行没有银弹，只有一张按粒度分层的地图：每层能藏住的通信量、要付的代价、要改的工程量都不同，选错层再努力也拿不到收益**。

读完你会知道：

- 五个层次各自藏住什么、代价是什么，为什么这些数字不能横着比；
- 三条主线分别在往前推什么；
- 三个还没答案的问题，以及你该先读哪两篇。

## 1. 五个层次，一张地图

五层对应五处不同的等待：stage 之间、梯度同步、层内 all-reduce、专家路由、decode 的启动开销。

| 层次 | 能藏住什么 | 一个代表数字 | 代价 |
|---|---|---|---|
| PP | stage 之间的 P2P 空等 | 气泡率 4.33% | 显存约 2× |
| DP | 反向里梯度的 all-reduce | 暴露时长≈尾桶的 $t$，不是百分比 | 桶大小与 launch 的权衡 |
| TP | 层内 all-reduce 的时延 | overlap efficiency 96% | 与 shape 强绑定 |
| EP/MoE | dispatch/combine 的 all-to-all | overlap ratio 47.1% | routing skew 与 SM 争用 |
| 推理 | TP all-reduce 与 KV 搬运 | 吞吐 1.19–1.91× | 小 batch 阈值 |

这五行的数字口径不同，不要横着比。

PP 那一行是 1.5B、$m=24$ 上的 ZB-2p：气泡率从 24.31% 压到 4.33%，代价是显存从 30 GB 涨到 59 GB。同一篇里 6.2B、$m=128$ 能到 0.29%，两个数都别套到别的模型上。

EP/MoE 那一行是 Weave 的 overlap ratio 口径：Comet 按同一口径只有 14.3%，它另报的「藏住 86.5% 通信」是另一个指标，代价是 NVSHMEM 常驻 buffer，每卡 2MN 字节。

TP 的 96% 是 Flux 在 A100 NVLink 上的数，换到 H800 上 decode 没有加速，小 m 时只有 0.95×；推理那行是各自 baseline 的端到端吞吐倍率，TokenWeave 单看延迟能到 1.28×，但 Mixtral-8x22B 在 1K/2K 下是净开销。

DP 那行没有百分比：暴露出来的就是尾桶那一段 $t$，桶切多大得用 `bucket_sim.py` 估，切太小 launch 反而反噬。

**选型不是选最快的论文，是选代价你能接受的那一档**。

## 2. 三条主线

把 2020–2026 的研究按三条线看，比按年份排更清楚。

### 2.1 切得对不对：从 layer 一路切到 warp

早期以 layer 为单位，通信是「层间」的事；1F1B/ZB 切到 microbatch；Flux/Comet 切到 tile，让通信和 GEMM 的 tile 级依赖对齐；Weave/T-CCL 切到 warp/CTA，通信只占几个 SM 甚至一个 warp。

粒度越细，可利用的空隙越多，但要付三笔：同步和 launch 变贵，Syncopate 的最优 chunk 只有 2–3 份；对 shape 变敏感，TokenWeave 在小 batch 下切了反而更慢；编程模型变复杂，TileLink 要引入 tile-centric primitive。

2026 年的前沿不是「无限细」，是「找到当前 shape 下的最优粒度」——Lagom、Analytical RM、Syncopate 都在把这件事自动化。

### 2.2 资源分得准不准：通信到底占几个 SM

「通信走网络、不占 GPU 资源」这个假设在 NVLink 时代就破了：128 KB 的 all-gather 在 NCCL 下占 3.30 个 SM，T-CCL 只占 1.32 个；Comet 把 132 个 CTA（正好是 Hopper 的 SM 数）分给通信和 GEMM；Weave 用 cost model 选通信 SM 数 $c$ 和 chunk 数 $K$，即使最优也仍有 mid bubble；DeepEP 直接说 zero-SM 的 RDMA EP 不支持。

于是问题变成分配。这一笔账在[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})里推过，这里只记结论：`sm_split_model.py` 的最优解在 $s=16$（$S=132$、$W_c=10000$、$W_m=3000$、$s_0=16$），再多 SM 不会让通信更快，只会偷走计算。

这就是 overlap tax 的根源：通信不是不占资源，是把资源从计算那里借走。Characterizing 测到的 18.9% 计算减速，就是这笔借债的利息。

### 2.3 语义保不保：overlap 改的不只是执行顺序

ZB 为了让 W 接上下一轮 F，得把 optimizer 的同步从关键路径上拿掉，再用参数/梯度版本握手保住同步训练的数学等价；DualPipe 把通信和计算写死成 8 步周期，靠错峰而不是动态调度；CO2 的 local-SGD 干脆允许 $\tau$ 步陈旧参数，用 staleness penalty 换 100% 扩展性；MoE 的 capacity factor 和 token drop 还会和 chunk 切分一起改掉路由结果。

**能不能 overlap 是工程问题，overlap 后还是不是同一个模型是数学问题**。后者常被忽略，直到 loss 不收敛。

## 3. 三个还没有答案的问题

### 3.1 什么时候不该 overlap

负结果比正结果有用。Flux 在 H800 上 decode 没加速，小 m 时 0.95×；TokenWeave 的 simple fusion 在 512–8K 是 0.92–0.98×，比不融合还慢；Lagom 的 AutoCCL 在某个 pattern 上 0.87×，比 NCCL 差；T-CCL 在 TP=4 的最差 shape 低于串行 baseline；tile-level signaling（[2607.19539](https://arxiv.org/abs/2607.19539)）里 cCTA=2 反而更慢，因为通信 SM 太少、消费者跟不上生产者。

收益 = 藏住的通信 − 计算减速 − 额外显存和复杂度。Characterizing 测到 overlap 平均让计算慢 18.9%，不少场景净收益只剩 10%；通信占比低、或计算对争用敏感时，这个差就是负的。可到现在也没人给出「给定 $(C,M,\gamma,\text{shape})$ 该不该 overlap」的判据，多数工作仍是先做、再在评估里挑好 shape 报。

**谁能把何时不该 overlap 形式化，谁就补上了这张地图最大的空白**——这个判据比又一个 1.2× 更有价值。

### 3.2 跨节点：大部分工作还停在单机

Flux 的实验主要是单机 8 GPU，Comet/Weave 是 8×H800 / 4×H100，TokenWeave 是 8×H100 DGX，T-CCL 明确只支持 intra-node、≤4 GPU，Syncopate 把多节点和跨 NIC 列为 future work。

真实训练不长这样：DeepSeek-V3 的 DualPipe 要跨最多 4 节点，NVLink 160 GB/s 对 IB 50 GB/s；DisagMoE 在 8 节点时 all-to-all 占比从 22% 涨到 78%；MegaScale 在 12,288 GPU 上 MFU 55.2%，可 0.5% 的慢机就能拖垮整体。

差的也不只是带宽：IB 的带宽和延迟比 NVLink 差一个量级，通信更难藏；多 NIC、rail-optimized 拓扑让「走哪条链路」变成新的调度维度；全局集合通信的 tail 还会被最慢节点放大。单机的 tile 级技巧搬过去，往往直接不成立。

### 3.3 功耗：被忽略的第四条约束

Characterizing 的测量很刺眼：overlap 的峰值功耗高 25%，H100 平均 38% TDP、峰值 140% TDP，100 W 的功耗帽下慢化可达 100%——论文引言写的是 107%，两处不一致。

在功耗受限的机房里，overlap 可能因为降频变成净负收益；多卡同时 overlap 会让整机功耗墙更早触发。频率和功耗本该是 overlap 评估的标配，现在大多数论文只报时间。

另外两个问题也只留一句：静态分区（Comet 的 $n_p/n_c$、Weave 的 cost model、Analytical RM 的 wave-quantized 模型都在 launch 前定死）碰上动态 routing 和动态序列长度必然错配，而动态调度的原子和信号开销又可能吃掉收益；手写 fused kernel 拿得到最好的数字，但离不开 Hopper TMA、NVSHMEM、NVSHARP，编译器路线能移植却只能打平（Syncopate 的 AG-GEMM），两边还没合上。

## 4. 先读哪两篇，看你手上是什么

- 做训练框架：先把 DP 的 `overlap_grad_reduce` / `overlap_param_gather` 和 PP 的 1F1B + `overlap_p2p_comm` 开起来，桶大小用 `bucket_sim.py` 估；TP 跨节点先砍通信量（FP8/INT6、层次化 all-reduce），别硬 overlap；MoE 先上 DeepEP 的 `async_with_compute_stream`，再考虑 fused kernel。读[第四篇]({{< relref "comm-overlap-04-data-parallel" >}})和[第二篇]({{< relref "comm-overlap-02-pipeline" >}})。
- 做 kernel：先量 SM active time 和 warp stall，分清是「通信占 SM」还是「计算本身慢」，并且记住 `sm_split_model.py` 的阶梯就是 wave 量化的结果。读[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})和[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})。
- 做推理：先算 decode 的 arithmetic intensity，小 batch 别切分（TokenWeave 的 1K 阈值），优先砍字节（FlashComm 的 INT6、AR+RMSNorm 融合）而不是硬 overlap，用 CUDA Graph 时还要注意通信捕获的兼容性。读[第九篇]({{< relref "comm-overlap-09-inference" >}})和[第十篇]({{< relref "comm-overlap-10-measurement" >}})。

## 5. 如果只记一件事

**不要问这个方案 overlap 了多少，要问它把哪部分资源、从哪种等待里、挪去干了什么活，代价是谁付**。

完整目录见[专题首页]({{< relref "comm-overlap-00-index" >}})。

---

**系列导航**：[补遗（三）：Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})

> 配套代码：`code/comm-overlap/`（7 个 CPU 模型 + 自测）
> 系列索引：[通算并行研读（专题）]({{< relref "comm-overlap-00-index" >}})
