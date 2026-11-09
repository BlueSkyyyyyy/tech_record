---
title: "通算并行研读（总结）：2026 年的地图与五个开放问题"
date: 2026-11-02
draft: false
weight: 16
series: ["comm-overlap"]
tags: ["通算并行", "系列", "总结", "分布式训练", "推理", "开放问题"]
categories: ["训练框架"]
---

写到这里，我们已经从流水线气泡走到 MoE 的 all-to-all，从 Megatron 的 bucket hook 走到 Flux 的 tile 级融合，再到 T-CCL 的 1.32 个 SM、Weave 的 persistent megakernel（一个 kernel 常驻跑完整层）。是时候把地图摊开，看看哪些路已经修好、哪些还是断头路。

先说结论：**通算并行没有银弹，只有一张按粒度分层的地图。** 每一层能藏住的通信量、要付的代价、需要的工程改造都不同；选错层，再努力也拿不到收益。

## 1. 一张地图：五个层次

这一节只回答一件事：五个层次各自把通信藏到什么程度，代价是什么。

先给读法：横着看是五个层次，竖着看每行是「手段 + 指标 + 代价」。**不要跨行比第五列的数字**——每行的分母、baseline、模型规模都不同，口径写在第四列里。

| 层次 | 通信 | 主流手段 | 指标 / 口径 | 报告值（baseline / 规模） | 主要代价 |
|---|---|---|---|---|---|
| **PP** | stage 间 P2P | GPipe → 1F1B → ZB → DualPipe | 气泡率（气泡时间 / 总时间，baseline 1F1B） | 1.5B、$m=24$：24.31% → 4.33%（ZB-2p），显存 30 → 59 GB；6.2B、$m=128$：ZB-2p **0.29%** | 显存 ~2×、调度复杂、要改模型 |
| **DP** | 梯度 all-reduce / reduce-scatter | bucket + backward hook + 异步 | 暴露时间 $t$（尾桶） | 约等于尾桶的 $t$，不是百分比 | 桶大小/launch 权衡；显存 |
| **TP** | 层内 all-reduce / all-gather | Flux tile 融合、T-CCL 低 SM、TileLink | overlap efficiency（Flux 自己的定义） | 最高 **96%**（Flux，A100 NVLink） | 与 shape/硬件强绑定；kernel 复杂 |
| **EP/MoE** | dispatch/combine all-to-all | DeepEP、Comet、Weave、Lancet | overlap ratio（Weave 口径） | 同口径 **14.3%（Comet）→ 47.1%（Weave）**；Comet 另报「藏住 86.5% 通信」，不是同一指标 | routing skew（专家负载不均）、SM 争用、显存 |
| **推理** | TP all-reduce、KV cache | NanoFlow、TokenWeave、FlashComm | 端到端吞吐倍率（各自 baseline） | **1.19–1.91×** 吞吐 | 小 batch 阈值、精度、CUDA Graph |

这张表最重要的信息不是"谁的数字大"，而是**每一层的"能藏住多少"和"代价"是绑定的**：

- PP 的 ZB-2p 在 1.5B 上把气泡压到 4.33%，显存从 30 GB 涨到 59 GB；0.29% 是 6.2B（$m=128$）的数，别套到 1.5B 上；
- TP 的 Flux 能到 96% overlap efficiency，但 H800 上 decode 没有加速，小 m 时甚至 0.95×；
- MoE 的 Comet 能藏 86.5% 的通信（这是「藏住比例」，不是上表的 overlap ratio），但需要 NVSHMEM 常驻 buffer，每卡 2MN 字节；
- 推理的 TokenWeave 能到 1.28× 延迟，但 Mixtral-8x22B 在 1K/2K 下是净开销。

**所以"选型"不是选最快的论文，而是选代价你能接受的那一档。**

## 2. 三条主线

这一节把 2020–2026 的工作压成三条主线。把论文按时间排，会看到它们反复在同一个方向上推进。

### 2.1 粒度下推：layer → chunk → tile → warp

- 早期：以 **layer** 为单位做 PP/DP，通信是"层间"的事；
- 1F1B/ZB：以 **microbatch** 为单位，通信和计算在 microbatch 之间交错；
- Flux/Comet：以 **tile** 为单位，通信和 GEMM 的 tile 级依赖对齐；
- Weave/T-CCL：以 **warp/CTA** 为单位，通信只占几个 SM，甚至只占一个 warp。

粒度越细，能利用的空隙越多，但：

- 同步/launch 开销越大（Syncopate 最优 chunk 只有 2–3 份）；
- 对 shape 越敏感（TokenWeave 在小 batch 下切分反而更慢）；
- 编程模型越复杂（TileLink 要引入 tile-centric primitive）。

**2026 年的前沿不是"无限细"，而是"找到当前 shape 下的最优粒度"。** 这也是为什么 Lagom、Analytical RM、Syncopate 这些"自动选粒度/资源"的工作在 2026 年集中出现。

### 2.2 资源显式化：从"通信免费"到"通信占几个 SM"

早期 overlap 的隐含假设是"通信走网络，不占 GPU 资源"。这个假设在 NVLink 时代就破了：

- NCCL 的 reduction kernel 要占 SM；T-CCL 测得 128 KB all-gather 在 NCCL 下占 **3.30 个 SM**，它自己只占 **1.32 个**；
- MoE 的 dispatch/combine 要占 SM，DeepEP 明确说"zero-SM RDMA EP is not supported"；
- Comet 把 **132 个 CTA**（正好是 Hopper 的 SM 数）分给通信和 GEMM；
- Weave 用 cost model 选通信 SM 数 $c$ 和 chunk 数 $K$，即使最优仍有 mid bubble。

一旦承认"通信占资源"，问题就变成**资源分配问题**：

$$
T = \max\left(\frac{W_c}{S-s},\ \frac{W_m}{\min(s,s_0)}\right)
$$

$s$ 是分给通信的 SM 数，$s_0$ 是带宽饱和点。`sm_split_model.py` 的最优解在 $s=16$（$S=132$，$W_c=10000$，$W_m=3000$，$s_0=16$）：再多的 SM 不会让通信更快，只会偷走计算。

**这也是 overlap tax（通信从计算那里借资源要付的税）的根源**：通信不是不占资源，而是把资源从计算那里借走。Characterizing 测得的 18.9% 计算减速，就是这笔借债的利息。

### 2.3 语义与调度协同：overlap 不是免费的午餐

最深的一条线是"overlap 改的不只是执行顺序，还可能改语义"：

- ZB 的 optimizer 同步 bypass：为了让 W 接上下一轮 F，必须把 optimizer step 的同步从关键路径上拿掉，但又要保持同步训练的数学等价（论文 §4 用参数/梯度版本握手）；
- DualPipe 的 8 步静态调度：把通信和计算写死成周期，靠"错峰"而不是"动态调度"；
- CO2 的 local-SGD（本地 SGD，允许参数陈旧）：直接允许 $\tau$ 步陈旧参数，用 staleness penalty 换 100% 扩展性；
- MoE 的 capacity factor（每个专家最多收多少 token）/ token drop：overlap 的 chunk 切分可能改变路由结果，需要保证语义一致。

**"能不能 overlap"是工程问题；"overlap 后还是不是同一个模型"是数学问题。** 后者往往被忽略，直到 loss 不收敛才被发现。

## 3. 五个开放问题

这一节列五个我认为还没有答案的问题。

### 3.1 什么时候不该 overlap？

这是整个领域最缺的一条判据。

已知的负结果：

- Flux 在 H800 NVLink 上 decode **没有加速**，小 m 时 0.95×；
- TokenWeave 的 simple fusion 在 512–8K 下 **0.92–0.98×**（比不融合慢）；
- Lagom 的 AutoCCL 在某个 pattern 下 **0.87×**（比 NCCL 还差）；
- Tile-level signaling（[2607.19539](https://arxiv.org/abs/2607.19539)）在它报告过的 shape 上，cCTA=2 反而比不 overlap 慢：通信 SM 太少，消费者跟不上生产者；
- T-CCL 在 TP=4 的最差 shape 低于串行 baseline；
- Characterizing 的 overlap 平均让计算慢 18.9%，很多场景净收益只有 10%。

这些结果说明：**overlap 的收益 = 藏住的通信 − 计算减速 − 额外显存/复杂度**。当通信占比低、或者计算对资源争用敏感时，这个差值可能是负的。

但目前没有一篇论文给出"给定 $(C,M,\gamma,\text{shape})$，是否该 overlap"的通用判据。大多数工作还是"先做，再在评估里挑好的 shape 报"。

**谁能把"何时不 overlap"形式化，谁就补上了这块地图最大的空白。**

### 3.2 静态分区 vs 动态调度

Comet 的 $n_p/n_c$ 是"按层形状自适应"，但仍是**launch 前**决定的；Weave 的 cost model 也是 prologue 里算一次；Analytical RM 用 wave-quantized 模型在 launch 前选 $C$。

问题是：

- routing 是动态的（MoE 的 hot expert 随 batch 变）；
- 序列长度是动态的（推理的请求长度分布）；
- 多租户/弹性训练的资源是动态的。

静态分区在这些变化下必然错配。动态调度（work stealing、persistent kernel 里偷 tile）是一个方向，但：

- 动态调度的开销（原子操作、信号、负载均衡）可能吃掉收益；
- 动态调度对 kernel 的编程模型要求更高（Weave 的 megakernel 要自己管 tile worklist）；
- 动态调度更难做编译器/可移植。

**"多细的动态"和"多准的静态"之间的边界，还没有人画清楚。**

### 3.3 多节点/跨 NIC：大部分工作还在单机

这是最明显的缺口：

- Flux 的实验主要是单机 8 GPU（A100/H800 NVLink）；
- Comet/Weave 是 8×H800 / 4×H100 单机；
- T-CCL 明确说只支持 intra-node、≤4 GPU；
- Syncopate 的多节点/跨 NIC 是 future work；
- TokenWeave 是 8×H100 DGX 单机。

但真实训练是跨节点的：

- DeepSeek-V3 的 DualPipe 要跨最多 4 节点，NVLink 160 GB/s vs IB 50 GB/s；
- DisagMoE 在 8 节点时 all-to-all 占比从 22% 涨到 **78%**；
- MegaScale 在 12,288 GPU 上 MFU（模型 FLOPs 利用率）55.2%，但 0.5% 的慢机就能拖垮整体。

跨节点的 overlap 难点不同：

- IB 的带宽/延迟比 NVLink 差一个量级，通信更难藏；
- 多 NIC、 rail-optimized 拓扑让"哪条链路"成为新的调度维度；
- 全局 barrier/集合通信的 tail 会被最慢的节点放大。

**单机的 tile 级技巧，搬到跨节点往往不成立。** 这是 2026 年最值得投入的方向之一。

### 3.4 可移植性：手写 kernel vs 编译器

手写 fused kernel（Flux/Comet/Weave）能拿到最好的数字，但：

- 依赖特定硬件（Hopper TMA、NVSHMEM、NVSHARP）；
- 对 shape 敏感，换模型要重调；
- 不支持 CUDA Graph / 动态 shape（T-CCL、TokenWeave 都提到）；
- 开发成本高，只有少数团队能维护。

编译器路线（TileLink/Syncopate/Triton-distributed）试图把"通信原语 + 调度"抽象出来，但：

- 目前只支持 NVIDIA；
- 多节点/动态 shape 还不成熟；
- 在已经高度优化的场景，只能和手写打平（Syncopate 的 AG-GEMM）。

**"手写的性能 + 编译器的可移植"还没有同时实现。** 短期的现实是：核心 kernel 手写，外围用编译器生成；长期看，需要一套跨节点的通信原语抽象。

### 3.5 功耗、频率与热：被忽略的第四条约束

Characterizing 的测量很刺眼：

- overlap 峰值功耗高 **25%**；
- H100 平均 38% TDP（热设计功耗）、峰值 **140% TDP**；
- 100 W 帽下，overlap 的慢化可达 **100%**（论文引言写 107%，两处不一致）。

这意味着：

- 在功耗受限的数据中心，overlap 可能因为降频而**净负收益**；
- 多卡同时 overlap 会让整机功耗墙更早触发；
- 频率/功耗的测量应该成为 overlap 评估的标配，但大多数论文只报时间。

**"性能 per watt"而不是"性能 per second"，可能是下一代 overlap 论文的分水岭。**

## 4. 按角色的建议

这一节按角色给建议，每条都尽量落到可执行的动作。

### 4.1 做训练框架

- 先把 DP 的 `overlap_grad_reduce` / `overlap_param_gather` 开起来，桶大小用 `bucket_sim.py` 估；
- PP 先上 1F1B + `overlap_p2p_comm`，再考虑 ZB/DualPipe；
- TP 如果跨节点，优先考虑砍通信量（FP8/INT6、层次化 all-reduce），而不是硬 overlap；
- MoE 先上 DeepEP 的 `async_with_compute_stream`，再考虑 Comet/Weave 的 fused kernel。

### 4.2 做 kernel

- 先测 SM active time 和 warp stall，确认瓶颈是"通信占 SM"还是"计算本身慢"；
- 如果通信占 SM 多，先试低 SM 的 collective（T-CCL 风格），再考虑 fused kernel；
- 如果做 fused kernel，先把 tile 级依赖画清楚，再决定 producer/consumer 的 CTA 比例；
- 不要忽略 wave 边界：`sm_split_model.py` 的阶梯就是 wave 量化（按 SM 波次取整）的结果。

### 4.3 做推理

- 先算 decode 的 arithmetic intensity，确认通信占比；
- 小 batch 不要切分（TokenWeave 的 1K 阈值）；
- 优先做通信量削减（FlashComm 的 INT6）和算子融合（AR+RMSNorm），再考虑 overlap；
- 用 CUDA Graph 时注意通信捕获的兼容性。

### 4.4 做研究

- 负结果和正结果一样重要；Characterizing 的价值就在于它敢报"overlap 更慢"；
- 跨节点、动态 shape、功耗是三个明显的空白；
- "何时不 overlap"的判据，比又一个 1.2× 更有价值。

## 5. 回看这个系列

十二篇正文（含导读），再加 2026-11 的一组补遗（总览 + 三篇深读），其实只讲了一个故事：

> **通信和计算争的是同一套资源。overlap 的本质，是在依赖允许的范围内，把资源从"等待"重新分配给"干活"；而每一次重新分配，都有代价。**

- 第 1 篇：账本（$T=C+E+B$，overlap tax）；
- 第 2–3 篇：PP 的错峰（GPipe → 1F1B → ZB → DualPipe）；
- 第 4 篇：DP 的切桶 + hook；
- 第 5 篇：TP 的 tile 融合（Flux/T-CCL/TileLink）；
- 第 6 篇：MoE 的三段式（DeepEP/Comet/Weave/Lancet/DisagMoE）；
- 第 7 篇：kernel/SM 的资源分配；
- 第 8 篇：编译器/自动调优（Syncopate/Lagom）；
- 第 9 篇：推理的阈值（NanoFlow/TokenWeave/FlashComm）；
- 第 10 篇：测量（nsys/ncu/overlap_audit）；
- 第 11 篇：地图与开放问题；
- 补遗：总览 + UniEP、MegaFlux、Entwine/mKernel 三篇深读。

如果只记一句话，我希望是：

**不要问"这个方案 overlap 了多少"，要问"它把哪部分资源、从哪种等待里、挪去干了什么活，代价是谁付"。**

## 结论

- 通算并行的五个层次：PP、DP、TP、EP/MoE、推理；每层的收益和代价强绑定。
- 三条主线：粒度下推、资源显式化、语义/调度协同。
- 五个开放问题：何时不 overlap、静态 vs 动态、跨节点、可移植性、功耗。
- 2026 年的前沿：不是"能不能 overlap"，而是"在正确的粒度上、用正确的资源、对正确的 shape 做 overlap"。
- 负结果和正结果一起看，才能判断一个方案是不是真收益。

前面三篇补遗补上了正文没覆盖的 2026 年工作：UniEP（persistent worker + 确定性）、MegaFlux（routing skew）、Entwine/mKernel（tile 级节奏）。其中 **UniEP 把 bitwise 可复现做成 overlap 的硬约束**，是最值得回头看的一篇——它提醒你：overlap 的收益要能被复现才有意义。

到这里整个专题就结束了。如果要动手，建议的顺序是：先按第一篇把三笔账算出来，再用实测手册确认瓶颈在哪一层，最后才挑这一层的方案——**不要反过来，先挑方案再找场景。**

---

**系列导航**：[十四：补遗三 Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})

> 配套代码：`code/comm-overlap/`（7 个 CPU 模型 + 自测）
> 系列索引：[通算并行研读（专题）]({{< relref "comm-overlap-00-index" >}})
