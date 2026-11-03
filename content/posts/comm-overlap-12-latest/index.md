---
title: "通算并行研读（补遗）：UniEP 与 2026 下半年的 MoE overlap 新论文"
date: 2026-11-03
draft: false
weight: 13
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "UniEP", "MegaFlux", "Entwine", "mKernel", "论文补遗"]
categories: ["训练框架"]
---

先承认一个缺口：正文第六篇和第十一篇的论文地图，停在 Comet、Weave、Lancet、DisagMoE、INC-MoE 这一批，**漏掉了 2026 年 4 月的 UniEP**，也漏掉了 2026 下半年的 MegaFlux、Entwine、mKernel、Mixture-of-Kittens、NCCL EP 等。这不是“再加一篇引用”那么简单，因为 UniEP 正好回答了前面留下的一个硬问题：

> overlap 把执行顺序改了，**数值还能不能和串行执行逐位一致？**

Comet 的 shared-tensor 切分、Weave 的 chunk pipeline、DualPipe 的双向调度，都在改 token 的到达顺序；对训练来说，这会影响 top-k 归约和 weight gradient 的累加顺序。大多数论文用“精度不掉”带过，UniEP 把它做成了第一等约束。这一篇补上 UniEP，并把 2026 下半年值得跟的 MoE overlap 工作串成一张新地图。

> 调研口径：2026-11-03 用四组 arXiv 查询（`expert parallelism + overlap`、`computation-communication + overlap`、`MoE + overlap`、`cs.DC + communication + overlap`）按提交时间倒序扫描，得到 2025-10 之后的 117 条候选，再人工筛到 15 篇。只覆盖 arXiv，不保证覆盖会议论文和工业界未公开工作。

## 1. UniEP：把“可复现”变成 overlap 的约束

UniEP（[2604.19241](https://arxiv.org/abs/2604.19241)，ByteDance Seed + 清华大学）的出发点很明确：EP 的优化手段已经很多——通信压缩、计算通信 overlap、SM 分区、GroupGEMM 融合——但生产框架如 Megatron-LM 的采用仍然保守。原因不是“论文不够快”，而是这些方案往往是 ad-hoc kernel，换一个 shape/拓扑就不适配，而且**不保证数值稳定性**。

UniEP 的做法可以拆成三层。

### 1.1 结构：两个 MegaKernel，而不是一条长流水

它没有把整个 MoE 层塞进一个 kernel，而是按依赖切成两个：

```text
Dispatch + GroupGEMM  →  GroupGEMM + Combine
   （第一个 MegaKernel）      （第二个 MegaKernel）
```

每个 MegaKernel 都是 **persistent threadblock**：launch 的 block 数等于物理 SM 数，一个 block 占住一个 SM 跑完整段，避免 context switch，也避免 block 数超过硬件容量导致的跨 block 同步死锁。

block 在运行时有四种角色：

| 角色 | 做什么 |
|---|---|
| **Comm-Worker** | 发 token、发跨 GPU 事务 |
| **Comp-Worker** | 算 GroupGEMM tile |
| **Relay-Worker** | 盯 token 到达、做同 rank 内的 multicast |
| **Reduce-Worker** | 在 combine 阶段等齐 top-k 贡献，再做归约 |

角色不是 launch 时写死的。UniEP 在 device 上放一个全局 atomic counter，把任务空间线性化成 `[0, N_comm)` 的通信任务和 `[N_comm, N_comm+N_comp)` 的计算任务；SM 每次原子加一取任务，取到什么就临时当什么。于是时间线自然呈现“先通信、再混合、最后计算”的三段，而不是靠 host 侧多 stream 硬排。

这套设计和 Comet/Weave 的差别很关键：

- Comet 是 **CTA specialization + 双 stream**：producer CTA 和 consumer CTA 的角色在 launch 时定死，靠 stream/event 排依赖；
- Weave 是 **每层重算 SM 分区**：cost model 在 device 上选 $c$ 和 $K$，但角色仍然按分区静态落位；
- UniEP 是 **任务级动态角色**：SM 不预设“我是通信还是计算”，由任务队列决定。它把“让”从“层粒度/launch 粒度”推进到了“任务粒度”。

代价是调度器更复杂：任务队列、scoreboard、角色切换都要在 kernel 内完成；换来的好处是没有 host 侧多 stream 的 alignment bubble，也没有 micro-batch 切分带来的顺序变化。

### 1.2 语义：为什么 token 顺序会影响数值

MoE 的 forward 里，token 被路由到不同 rank 的专家；combine 时要把同一个 token 的 top-k 个专家输出加起来。浮点加法不满足结合律，**不同的到达/累加顺序会给出不同的低位**。反向更敏感：weight gradient 的累加顺序也会变。

常见 overlap 方案为了制造并行窗口，会把 token 切成 micro-batch，或者让不同 chunk 乱序到达。这在“精度看起来没掉”的层面往往能过，但做不到 **bitwise identical**。UniEP 的约束是：不管 overlap 怎么排，最终结果要和“不 overlap 的串行执行”逐位一致。

它做了两件事：

1. **确定性全局 token 映射**。为每个本地 token 预计算 `(target_rank, expert_id, destination_offset)`，用全局前缀和算出每个专家在每个 rank 上的固定偏移；Comm-Worker 按这张表写，不依赖运行时的原子竞争来决定顺序。
2. **不切 micro-batch**。UniEP 明确把“避免 micro-batch splitting”作为保持顺序的手段；并行窗口靠 SM 角色动态切换和 tile 级流水，而不是靠把 batch 切开。

这带来一个反直觉的结论：**overlap 不一定要靠“切 batch”**。前面几篇里，切 chunk 是制造并行的常用手段；UniEP 证明在 MoE 层内，靠 persistent worker + scoreboard + 任务队列，可以在不切 batch 的情况下把通信藏进计算。代价是 kernel 更复杂，收益是语义更干净。

论文在 12 个 MoE 配置上对比了 PyTorch+NCCL 的无 overlap 参考，forward/backward 都做到 bitwise identical；作为对照，Comet 不保证内部 token 顺序与无 overlap 参考一致。这是 UniEP 最值得记的一条：**“精度没掉”与“逐位一致”是两个不同的验收标准。**

### 1.3 数字：1.03×–1.38× 是下限到上限，不是平均值

UniEP 在两个 Hopper 集群上评估：

- Cluster 1：8×Hopper，NVLink 200 GB/s/dir，带宽受限；
- Cluster 2：8×Hopper，NVLink 400 GB/s/dir，带宽更充裕；
- 12 个生产 MoE 配置，专家数 64–512，序列长度 8k/32k/128k。

论文摘要的总体口径是 **1.03×–1.38× over state-of-the-art**。分场景拆开看：

- Kernel 级：Dispatch+GroupGEMM 在 Cluster 1 8k 上相对 Comet **1.32×**，32k 上 **1.30×**；Cluster 2 8k 上 **1.57×**，32k 上 **1.45×**；
- GroupGEMM+Combine：Cluster 1 8k **1.31×**、32k **1.87×**；Cluster 2 8k **1.21×**、32k **1.07×**；
- Layer 级：Cluster 1 8k forward **1.08×**、backward **1.03×**；32k forward **1.22×**；Cluster 2 32k forward **1.19×**、backward **1.09×**；
- 长上下文 128k：Cluster 1 forward **1.28×**、Cluster 2 **1.33×** 相对 Comet；
- 生产训练：128 GPU（16 节点 × 8）、512k 序列，吞吐从 **127 B tokens/day 提到 138 B tokens/day**，**1.09×**，且保持 bitwise 可复现。

所以 **1.03× 是 layer backward 的下限，1.38× 是上层汇总的上限**；不要把它当成“平均 1.2×”。真正稳定的是 kernel 级的 1.3×–1.6×，到了 layer/e2e 会被 attention、gating、非 MoE 计算稀释。

### 1.4 代价与边界

- **实现成本**：论文说 MegaKernel 约 **21k 行 Python**（Triton-distributed），不是几百行能改出来的；
- **自动调参**：SM 分区、warp 数、Relay/Reduce 数量都要搜；论文用解析性能模型把 $10^5$ 量级的配置空间压下来，但调参本身仍有成本；
- **硬件**：只在 Hopper 上验证，且是单节点 8 卡；跨节点扩展是下一步；
- **代码**：论文称 MegaKernel 实现已开源在 [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)。我检查该仓库 `main` 分支时没有看到独立的 `uniep` 目录，可能还在分支/PR 或尚未完全合入；引用代码时以论文和仓库实际状态为准；
- **baseline 限制**：Comet 只支持特定 $H_{dim}$ 和 top-k，MoE-2/MoE-12 这类配置跑不了，所以部分对比是“UniEP vs Serial”，不是“UniEP vs Comet”。

## 2. MegaFlux：routing skew 下的“复制专家”

UniEP 解决的是“顺序 + 调度”，MegaFlux（[2610.00671](https://arxiv.org/abs/2610.00671)）解决的是另一个老问题：**routing skew**。

固定专家放置下，热点专家所在的 GPU 会成为 straggler，其他 GPU 空转；层延迟由最慢的 GPU 决定。复制热点专家可以把工作挪到空闲 GPU，但复制不是免费的：

- forward：副本要先收到专家权重，才能开始算；
- backward：副本产生的 partial weight gradient 要送回 owner 做归约。

MegaFlux 把“复制哪些专家、放到哪、每个 GPU 复制多少”做成运行时决策，并把复制引入的通信**塞进 persistent MoE 执行里**：

- forward：副本权重一到就开始算，权重传输和计算流水；
- backward：副本梯度归约和后续专家计算重叠；
- planner 在 device 上选副本位置和 tile-aligned token block，不改 router 输出。

数字（8×B200，147 个配置/方向）：

- forward 几何平均 **1.45×**、backward **1.28×**，相对“同样 megakernel 但固定放置”；
- 峰值 **2.14×** / **2.64×**；
- 复制权重传输成本被隐藏 **56%–76%**（forward）；权重传输 + 梯度归约合计隐藏 **91%–100%**（backward）；
- 接进 vLLM 跑 DeepSeek-V4-Pro prefill，端到端中位数 **1.13×–1.26×**。

它的边界也很清楚：**它优化的是“放置 + 复制通信”，不是“减少总计算”**；副本带来额外权重流量和显存占用，routing 极度均匀时收益会变小。和 UniEP 是正交的：UniEP 管单层内的顺序/调度，MegaFlux 管跨 GPU 的负载分布。

## 3. Entwine 与 mKernel：tile 级 overlap 往两头走

### 3.1 Entwine：让 tile 的生产节奏可预测

Entwine（[2609.11562](https://arxiv.org/abs/2609.11562)）的观察是：tile 计算完成得比整个 tensor 早，所以有 overlap 机会；但**计算和通信的进度不匹配**——通信可能没数据可发，也可能在数据突发到达时滞后；通信还会抢 SM，反过来拖慢计算。

它做三件事：

1. 重排 tile 计算顺序，让数据以更均匀的节奏产出，而不是“前 90% 时间沉默、最后 10% 爆发”；
2. 用 fine-grained SM-based communication 处理 tile 结果，低延迟、低开销；
3. 协调通信和计算的 SM 分配，平衡“通信进度”和“计算被拖慢”。

在 TP LLM workload 上：相对 cuBLAS+NCCL 几何平均 **1.232×**、最高 **1.433×**；相对 SOTA overlap baseline 高 **3.1%–9.8%**。代码“will open-source upon publication”，目前还没看到公开仓库。

这篇和第五篇的 Flux/TileLink 是一条线：**tile 级 overlap 的瓶颈从“能不能融”变成“tile 的产出节奏和 SM 配额是否匹配”**。Flux 用 swizzling 对齐通信顺序，Entwine 把“节奏”本身作为优化对象。

### 3.2 mKernel：把 tile 级融合推到多机

mKernel（[2609.13585](https://arxiv.org/abs/2609.13585)）指出：现有 fused kernel 大多局限在单 NVLink domain；跨节点时，RDMA 的延迟和带宽又变成新约束。

它的做法：

- persistent kernel 把 SM 分成 compute/communication 两类角色；
- on-GPU controller 在运行时调 SM 分区（不同 kernel/shape 最优分区不同）；
- 数据移动分层：先 intra-node NVLink，再 inter-node RDMA，尽量让跨节点流量最小；
- 用 GPU 侧的轻量 command queue + host proxy 驱动 RDMA verbs，支持 IB 和 AWS EFA；
- 一个反直觉观察：GPUDirect Async（IBGDA）相对 host-assisted GPU-initiated communication 没有明显额外收益。

结果：2 个 16-GPU H200 集群上，GEMM+AllReduce 最高 **1.72×**，Ring Attention 最高 **1.88×**。代码：[uccl-project/mKernel](https://github.com/uccl-project/mKernel)。

把 Entwine 和 mKernel 放在一起看：**tile 级 overlap 正在从“单机单 NVLink”扩展到“单机多级互联 + 多机 RDMA”**。第七篇的 SM 分区模型在单机成立，到了 mKernel 里，“分区”还要和网络层级一起调。

## 4. 2026 下半年雷达：scale-up、标准 API 与异构硬件

下面这些不一定都要写进正文，但值得跟进。按“对前面结论的冲击程度”排序：

| 时间 | 工作 | 一句话 | 代码/artifact |
|---|---|---|---|
| 2026-09 | **Mixture-of-Kittens**（[2609.36070](https://arxiv.org/abs/2609.36070)） | NVL72 scale-up 上的 MoE 训练 megakernel；按算子选 push/pull、重排 overlap、去掉 CPU-GPU 同步；最高 **2.37×**，512 GPU 生产 **1.41×** | [cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens) |
| 2026-03 | **NCCL EP**（[2603.13606](https://arxiv.org/abs/2603.13606)） | 把 EP dispatch/combine 做成 NCCL Device API 的标准原语：`ncclEpDispatch` / `ncclEpCombine`，LL 模式 1–128 token，HT 模式 4096+ token | 基于 [NVIDIA/nccl](https://github.com/NVIDIA/nccl) Device API，论文未给独立仓库 |
| 2026-09 | **ThunderEP**（[2609.40093](https://arxiv.org/abs/2609.40093)） | PCIe 消费级 GPU 的 EP 通信：去掉 ring relay、用 DMA engine、降低 completion flag 轮询；dispatch **2.00×**、combine **1.53×**、e2e **1.66×** | 论文未给公开仓库 |
| 2025-11 | **FarSkip-Collective**（[2511.11505](https://arxiv.org/abs/2511.11505)） | 改模型结构（skip connection）来让 blocking communication 可 overlap；DeepSeek-V3 prefill **97.3%** overlap、TTFT **32.6%**；训练 A2A **88.9%** | 未确认公开代码 |
| 2026-05 | **HyperParallel-MoE**（[2605.23764](https://arxiv.org/abs/2605.23764)） | Ascend NPU 的 AIC/AIV 异构 tile 调度；Dispatch-to-Combine 延迟最高 **1.58×** | [gitcode](https://gitcode.com/mindspore/hyper-parallel/tree/master/hyper_parallel/core/multicore) |
| 2026-05 | **Piper**（[2605.05049](https://arxiv.org/abs/2605.05049)） | HPC 上的 MoE 资源建模 + pipeline hybrid parallelism；MFU 比 X-MoE 高 **2–3.5×**，all-to-all 带宽 **1.2–9×** | 未给公开仓库 |
| 2025-10 | **HybridEP**（[2510.19470](https://arxiv.org/abs/2510.19470)） | 跨数据中心 EP：低带宽下不硬 overlap，而是动态改专家放置/传输比例；受限带宽最高 **5.6×** | 未给公开仓库 |
| 2025-09 | **FlowMoE**（[2510.00207](https://arxiv.org/abs/2510.00207)） | 把 MHA、gating、expert、A2A、all-reduce 放进统一 pipeline；训练时间 **−13%–−57%**，能耗 **−10%–−39%** | 未给公开仓库 |
| 2026-06 | **Resource-aware overlap**（[2606.09200](https://arxiv.org/abs/2606.09200)） | 不改库/kernel，只调 occupancy 和通信 stream 优先级；A40/A100/H100/MI250X 上最多 **−25.5%** 时间 | 未给公开仓库 |
| 2025-11 | **ParallelKittens / Iris**（[2511.13940](https://arxiv.org/abs/2511.13940)、[2511.12500](https://arxiv.org/abs/2511.12500)） | 多 GPU kernel 的“最小原语集”：PK 用 8 个原语 + <50 行 device code，DP/TP **2.33×**、SP **4.08×**、EP **1.22×**；Iris 用 Triton 做 tile 级通信，GEMM+AllScatter **1.79×** | 论文/项目页，未逐一确认仓库 |

这张表里，真正会改变前面结论的是三条：

1. **NCCL EP 把“EP 通信”标准化**。DeepEP 是事实标准，但它是“库”；NCCL EP 是“API”。一旦 `ncclEpDispatch/Combine` 进入 NCCL 主线，第六篇里“DeepEP 是唯一今天就能用的基线”这句话就要改成“DeepEP 是高性能基线，NCCL EP 是生态基线”。
2. **scale-up 改变最优策略**。MoK 在 NVL72 上发现现有 MoE 训练系统“经常比 PyTorch+NCCL 还慢”，因为它们是按 scale-out 网络设计的；在单跳大带宽域里，push/pull 的选择、overlap 结构、CPU-GPU 同步的权重都变了。第十一篇的“跨节点”开放问题要拆成“scale-out”和“scale-up”两个问题。
3. **硬件异构化**。HyperParallel-MoE 在 Ascend 上把 AIC/AIV 当成两类 SM 来排；ThunderEP 在 PCIe 消费卡上绕开 NVLink 假设。前面以 NVIDIA Hopper + NVLink 为默认前提的账本，需要按硬件重新标定。

## 5. 对前面几篇的修正

补完这批论文，第十一篇的地图要改三处。

### 5.1 “语义”不只是“梯度对不对”，还包括“逐位可复现”

前面把语义约束写成“DP 梯度不能省、PP 激活不能错位、MoE token 不能丢”。UniEP 把标准抬高了：**overlap 之后的执行顺序必须和串行参考逐位一致**。这对生产训练很重要——bitwise 不一致会让“换一个 overlap 配置”变成一次不可归因的实验变更；对调试、复现、合规都是负担。

所以第六篇的选型表应该加一列“是否保证 deterministic/bitwise”；Comet 是“不保证”，UniEP 是“保证”，Weave/DeepEP 需要看具体模式。不是所有场景都需要 bitwise，但至少要知道自己站在哪一边。

### 5.2 “单 stream 也能做细粒度 overlap”

前面默认“双 stream / 多 stream 才有 overlap”。UniEP 证明：**persistent kernel + SM 角色动态切换 + scoreboard** 可以在单 CUDA stream 内做 tile/token 级 overlap，而且没有 host 侧 stream 管理的 alignment bubble；mKernel 也把 SM 分区和跨节点传输放进了 persistent kernel。代价是 kernel 复杂度、调试难度、以及和 CUDA Graph/框架调度的兼容性。

这不是说双 stream 过时，而是说“overlap 的实现载体”多了一种：从“stream 级”下推到“kernel 内任务级”。

### 5.3 “自动调参”从加分项变成必选项

Syncopate 是“编译 + 实测搜索”，Lagom 是“资源参数搜索”，UniEP 是“解析模型 + 在线选 SM/warp”。共同点是：**形状、拓扑、routing 一变，手写常数就失效**。2026 年的系统如果不带 autotuning，很难跨模型/跨集群复用。

但自动调参也有代价：UniEP 的配置空间 $10^5$，Lagom 的搜索轮数约 2× AutoCCL，Syncopate 要枚举 backend × chunk × tile order。**“搜得起”本身是工程能力**，不是论文里一句“we autotune”就能带过。

## 6. 结论

- **UniEP 是这次补遗里最该补的一篇**：它把 MoE overlap 从“切 chunk/双 stream”推进到“persistent worker + 动态角色 + deterministic token mapping”，并用 1.03×–1.38× 的总体收益证明“可复现”不必以牺牲 overlap 为代价。
- **MegaFlux 补上了 routing skew 的另一半**：不是把热点专家算得更快，而是把副本和副本通信一起流水化；B200 上 forward/backward 1.45×/1.28×，且和 UniEP 正交。
- **Entwine/mKernel 把 tile 级 overlap 推到“节奏”和“多机”**：前者调 tile 产出节奏 + SM 配额，后者调 SM 分区 + NVLink/RDMA 分层；单机模型不够用了。
- **MoK/NCCL EP/ThunderEP/HyperParallel 说明硬件在分叉**：NVL72 scale-up、PCIe 消费卡、Ascend AIC/AIV，各自的最优解不同；不要拿一套 Hopper NVLink 的结论套所有集群。
- **补遗不是“再加几篇引用”，而是修正地图**：语义约束要加“逐位可复现”，实现载体要加“单 stream 内任务级”，自动调参要从加分项变成必选项。

> 配套阅读：第六篇（MoE 三段式）、第七篇（SM 分区）、第十一篇（地图）；UniEP 的 deterministic token mapping 可以和第四篇的 DP bucket 顺序、第二篇的 1F1B 调度对照看——它们都在回答同一个问题：**为了 overlap 改顺序，改到什么程度还可以接受？**

---

**系列导航**：[十一：总结]({{< relref "comm-overlap-11-summary" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
