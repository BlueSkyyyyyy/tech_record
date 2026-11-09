---
title: "通算并行研读（补遗三）：Entwine、mKernel 与 2026 生态——tile 级 overlap 的下一站"
date: 2026-11-03
draft: false
weight: 15
series: ["comm-overlap"]
tags: ["通算并行", "系列", "Entwine", "mKernel", "tile", "NCCL EP", "Mixture-of-Kittens", "异构硬件"]
categories: ["训练框架"]
---

前两篇补遗分别讲了“层内顺序”（UniEP）和“跨卡放置”（MegaFlux）。这一篇讲第三、第四个方向：

- **Entwine**：单机内，tile 的**生产节奏**和通信/计算的 **SM 配额**怎么协调；
- **mKernel**：把 tile 级融合推到**多机**，NVLink 和 RDMA 怎么分层。

最后用一节把 MoK、NCCL EP、ThunderEP、FarSkip、HyperParallel、Piper 等放进同一张生态图。

## 1. Entwine：让 tile 的产出节奏可预测

这一节回答一个问题：tile 早算完，为什么通信不一定能早开始。

### 1.1 问题：tile 早完成，不等于通信能早开始

Entwine（[2609.11562](https://arxiv.org/abs/2609.11562)）研究的是 GEMM–ReduceScatter（TP 里的 row-parallel GEMM + RS）。现代 GEMM kernel 把输出切成 tile，block 独立调度；一个 tile 算完，理论上就可以发给需要的 rank。但有两个 mismatch：

1. **生产节奏不均**：如果按输出 partition 连续遍历，一个 producer 会先连续算“自己保留”的 tile，这段时间没有跨 GPU 数据可发；等它切换到远程 partition 时，数据又突发到达。通信单元要么闲着，要么滞后。
2. **资源竞争**：通信 block 也占 SM、占 HBM 带宽；通信开得越多，计算越慢。overlap 的收益可能被“计算 slowdown”吃掉。

论文测得：在它的 workload 上，串行执行会留下 **11.0%–40.2%** 的通信暴露时间，中位数 **23.3%**。这就是 Entwine 要吃掉的空间。

### 1.2 设计：交错生产 + 独立 tile 通信 + 共享 SM 池

Entwine 的设计可以拆成三件事：生产节奏、通信粒度、SM 配额。

**（1）交错生产。**

Entwine 不按 partition 连续遍历，而是按逻辑 block 顺序轮转输出 partition。对 $W$ 个等宽、block 对齐的 partition，producer block 的逻辑位置 $p$ 映射到：

$$
d = p \bmod W, \quad \ell = \lfloor p / W \rfloor
$$

其中 $d$ 是目标 rank，$\ell$ 是 partition 内 block 索引。所有 rank 用同一映射，保证同一个 tile 的贡献在对应逻辑位置。这样每个 producer 的本地保留 tile 被均匀地插在“需要远程传输”的 tile 之间，通信数据以更平稳的节奏产出。

代价是**数据 reuse（复用）可能下降**：连续访问同一 partition 对 cache 更友好，交错会打散局部性。Entwine 明确把这一点算进 end-to-end，而不是只看 GEMM 吞吐。

**（2）独立 tile 通信。**

每个通信任务只处理**一个输出 tile**。一个 tile 的 ReduceScatter 只依赖各 rank 对它的贡献，不依赖其他 tile。所以一个 tile 的贡献到齐就能归约，不用等一个 wave/group 的所有 tile。

这和 FlashOverlap 的“按 wave/group 触发 collective”不同：group 能形成更大消息、提高带宽利用率，但组内一个 tile 可能被无关输出拖住。Entwine 选择“tile 级独立”，用更细的粒度换更短的等待。

**（3）共享 SM 池 + communication budget。**

Entwine 让一个 compute kernel 和一个 communication kernel 并发执行，不按 tile 逐个 launch。block 退休后，释放的 SM 资源可以被任一 kernel 的 pending work 使用。

它用一个 **communication budget（同时允许跑几个通信 block）** 限制“同时允许的通信 block 数”：

- budget 太小：通信尾巴长，暴露多；
- budget 太大：通信抢 SM/带宽，GEMM 变慢；
- 最优 budget 由 **end-to-end latency** 决定，不是“通信尾巴最短”。

论文 Figure 2 展示了两种 $K$：$K=2048$ 时，增大 budget 先快速降延迟，然后进入一个较宽的平台区；$K=512$ 时，延迟有一个内部极小值，再增大 budget 不再收益。这说明**没有通用常数**，必须按 workload 选。

### 1.3 结果与边界

- 相对 cuBLAS+NCCL：几何平均 **1.232×**，最高 **1.433×**；
- 相对 SOTA overlap baseline：几何平均高 **3.1%–9.8%**；
- 代码：论文说“will open-source upon publication”，目前没看到公开仓库；
- 适用边界：主要验证在 GEMM–ReduceScatter / TP LLM workload；换到 MoE all-to-all 需要重新设计依赖和 budget；
- 代价：交错顺序可能降低 reuse，通信 block 的 budget 需要离线 profile。

Entwine 和第五篇的 Flux/TileLink 是一条线：**tile 级 overlap 的瓶颈从“能不能融”变成“tile 的产出节奏和 SM 配额是否匹配”**。Flux 用 swizzling 对齐通信顺序，Entwine 把“节奏”本身作为优化对象。

## 2. mKernel：把 tile 级融合推到多机

这一节把同样的 tile 级思路推到多机。

### 2.1 问题：单 NVLink domain 的假设失效

mKernel（[2609.13585](https://arxiv.org/abs/2609.13585)）指出：现有 fused kernel 大多局限在单 NVLink domain；跨节点时，RDMA 的延迟和带宽变成新约束。它给了一个简单的 fused kernel 时间模型：

$$
T_{\text{fused}} \approx \max\left(T_{\text{compute}} \cdot \frac{S}{S-S_c},\; T_{\text{NVLink}},\; T_{\text{network}}\right) + T_{\text{fill/drain}} + T_{\text{sync}}
$$

其中 $S$ 是总 SM 数，$S_c$ 是分给通信的 SM 数。这个式子把 trade-off 说清楚了：

- 多给通信 SM，$T_{\text{NVLink}}$/$T_{\text{network}}$ 下降，但 $T_{\text{compute}} \cdot S/(S-S_c)$ 上升；
- 跨节点时 $T_{\text{network}}$ 可能成为主项，因为 IB/EFA 每 GPU 每方向只有 **50 GB/s**，而节点内 NVLink 是 **450 GB/s**，HBM 是 **4.8 TB/s**。

### 2.2 五个挑战

论文列了 C1–C5，值得记：

1. **C1 节点内/间带宽不对称**：节点内 NVLink 快一个数量级，应该先做本地聚合/复制，减少重复跨节点传输；在 rail-optimized（按 rail 对齐）拓扑里，优先和同 local index 的 rail peer 交换。
2. **C2 传输粒度不对称**：节点内是 memory semantics（peer memory load/store），跨节点是 message semantics（RDMA request）；要把细粒度本地 tile 聚成网络 chunk，摊销 per-message 成本，同时尽早暴露 ready work。
3. **C3 传输顺序语义不同**：IB reliable connection 保序，EFA SRD 不保序；arrival flag 必须在 payload 之后可见，不能假设所有平台一样。
4. **C4 跨设备同步**：网络完成和 GPU 内存可见性有开销；persistent kernel 需要 readiness check + memory ordering，不能每次传输后全局同步。
5. **C5 SM 分配要自适应**：节点内 sweep 的最优通信 SM 数在 **2–64** 之间；同一个 kernel 内，compute/communication 的平衡也会变。

### 2.3 设计：角色分工 + 分层传输 + 自适应控制器

mKernel 的 persistent kernel 把 thread block 分成：

- **Compute**：算 tile，写结果到 GPU memory，设 readiness flag；
- **Intra-node comm**：通过 NVLink/NVSwitch 和同节点其他 7 个 GPU 交换 tile，做 broadcast/reduction；
- **Inter-node send**：把传输命令写入 host memory 的 command queue；
- **Receive**：观察 arrival flag，消费远端数据；
- **Controller**（可选）：读 progress counter，发布目标通信 block 数。

跨节点路径是：

```text
Compute → GPU buffer + ready flag
        → inter-node send block → host command queue
        → host proxy thread → NIC RDMA
        → rail peer GPU memory + arrival flag
        → receive block → 远端 compute
```

几个关键点：

- **payload 留在 GPU memory**，host 只转发命令，避免额外拷贝；
- **host proxy 批量提交**命令，摊销 doorbell 成本；
- **rail peer**：跨节点优先同 local index，减少网络跳数；
- **自适应 SM 分区**：controller 根据 progress counter 调通信 block 数，而不是 launch 时定死；
- 一个反直觉观察：**GPUDirect Async（IBGDA，GPU 直接发起 RDMA）相对 host-assisted GPU-initiated communication 没有明显额外收益**。这和“GPU 发起一定更快”的直觉相反，说明瓶颈在别处（消息粒度、flag、SM 争用）。

### 2.4 结果与边界

- 2 个 16-GPU H200 集群（一个 IB、一个 AWS EFA）；
- GEMM+AllReduce 最高 **1.72×**；
- Ring Attention 最高 **1.88×**；
- 代码：[uccl-project/mKernel](https://github.com/uccl-project/mKernel)；
- 边界：需要多机环境；对单机 workload，收益不如 Entwine/Flux 直接；RDMA 平台差异（IB/EFA）要分别适配。

把 Entwine 和 mKernel 放在一起看：**tile 级 overlap 正在从“单机单 NVLink”扩展到“单机多级互联 + 多机 RDMA”**。第七篇的 SM 分区模型在单机成立，到了 mKernel 里，“分区”还要和网络层级一起调。

## 3. 生态雷达：scale-up、标准 API 与异构硬件

这一节把其余工作放进同一张生态图。

### 3.1 Mixture-of-Kittens：NVL72 上的“单 kernel 全融合”

MoK（[2609.36070](https://arxiv.org/abs/2609.36070)）的目标是 NVL72 这种 **scale-up 域（NVLink 连成的一个大节点）**：72 个 GPU 通过 NVLink 交换机连成一个单跳大带宽域。它的观察是：现有 MoE 训练系统很多是按 scale-out 网络设计的，在 NVL72 上“经常比 PyTorch+NCCL 还慢”。

MoK 的做法：

- 把 MoE 的所有计算和通信融成一个 **deterministic megakernel**；
- 按算子选 **push 或 pull**：不是所有通信都用同一种语义；
- 重排 overlap，让通信和计算在单 kernel 内交错；
- 去掉不必要的 CPU-GPU 同步；
- 支持 MXFP8、FSDP 的 RDMA overlap、fused router weight gradient。

结果：最高 **2.37×**，512 GPU 生产 **1.41×**；代码 [cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens)。

它和 UniEP 的区别：UniEP 是单节点 8 卡、EP MoE、强调 bitwise；MoK 是 NVL72 scale-up、训练全栈、强调“单 kernel + 确定性”。两者都说明“单 kernel 全融合”在特定硬件上可以赢过“多 kernel + 多 stream”。

### 3.2 NCCL EP：把 EP 通信标准化

NCCL EP（[2603.13606](https://arxiv.org/abs/2603.13606)）把 EP 的 dispatch/combine 做成 NCCL Device API 的原语：

- `ncclEpDispatch` / `ncclEpCombine`；
- **LL 模式**：1–128 token，低延迟；
- **HT 模式**：4096+ token，高吞吐；
- 基于 NCCL 的 Device API，H100 多节点验证。

它的意义不是“又一个更快的库”，而是**API 标准化**：DeepEP 是事实标准，但它是“库”；NCCL EP 是“生态基线”。一旦进入 NCCL 主线，框架可以直接调用，不用自己维护一套 EP 通信栈。

### 3.3 ThunderEP：PCIe 消费卡的 EP

ThunderEP（[2609.40093](https://arxiv.org/abs/2609.40093)）针对的是没有 NVLink 的 PCIe 系统（RTX 4090/5090）。传统 NCCL 的 CPU-staged 通信会多一次 PCIe 拷贝，还会和专家计算抢 GPU 资源。ThunderEP：

- 去掉 ring 的 relay hop；
- 用 DMA engine 搬数据，避免占 compute SM；
- 降低 completion flag 轮询开销；
- 集成进 vLLM。

结果：dispatch **2.00×**、combine **1.53×** vs NCCL，e2e 最高 **1.66×**。这说明“EP overlap”不是数据中心 GPU 的专利；消费卡/PCIe 场景需要另一套设计。

### 3.4 FarSkip：改模型结构，而不是改调度

FarSkip-Collective（[2511.11505](https://arxiv.org/abs/2511.11505)）走了一条更激进的路：**改模型结构**（skip connection）来让 blocking communication 可 overlap。

- 把 16B–109B 的模型转成“可 overlap”的版本；
- Llama 4 Scout（109B）用 self-distillation 转换，下游平均精度在原始 instruction-tuned 版本的 **1%** 以内；
- DeepSeek-V3 架构 + EP + SGLang：TTFT **32.6%** 加速，prefill 阶段 **97.3%** 通信-计算 overlap；
- 训练 DeepSeek-V3 MoE 层：all-to-all **88.9%** overlap。

这条路的代价是“要改模型、要蒸馏”，不是所有团队都能接受；但它说明**架构和系统不是分离的**：如果调度层怎么调都藏不住，可以考虑让模型结构本身更适合 overlap。

### 3.5 异构硬件：Ascend、HPC、跨 DC

- **HyperParallel-MoE**（[2605.23764](https://arxiv.org/abs/2605.23764)）：Ascend NPU 的 AIC/AIV 异构 tile 调度，Dispatch-to-Combine 延迟最高 **1.58×**；代码在 gitcode。
- **Piper**（[2605.05049](https://arxiv.org/abs/2605.05049)）：HPC 上的 MoE 资源建模 + pipeline hybrid parallelism；MFU（模型 FLOPs 利用率）比 X-MoE 高 **2–3.5×**，all-to-all 带宽 **1.2–9×**。
- **HybridEP**（[2510.19470](https://arxiv.org/abs/2510.19470)）：跨数据中心 EP，低带宽下不硬 overlap，而是动态改专家放置/传输比例；受限带宽最高 **5.6×**。
- **FlowMoE**（[2510.00207](https://arxiv.org/abs/2510.00207)）：把 MHA、gating、expert、A2A、all-reduce 放进统一 pipeline；训练时间 **−13%–−57%**，能耗 **−10%–−39%**。
- **Resource-aware overlap**（[2606.09200](https://arxiv.org/abs/2606.09200)）：不改库/kernel，只调 occupancy 和通信 stream 优先级；A40/A100/H100/MI250X 上最多 **−25.5%** 时间。

这些工作的共同信号：**“一套 Hopper NVLink 的最优解”不再通用**。scale-up（NVL72）、scale-out（IB）、PCIe、Ascend、跨 DC，各自需要不同的通信原语、SM 配额和放置策略。

## 4. 把四条线合起来

读法：下面四行是四个可以组合的旋钮，不是四个竞品；“优化对象”和“主要约束”比“适用场景”更值得看。

| 线 | 代表 | 优化对象 | 主要约束 | 适用场景 |
|---|---|---|---|---|
| 语义/调度 | UniEP | 层内 token 顺序、SM 角色 | bitwise、shape、Hopper | 单节点 EP 训练 |
| 负载分布 | MegaFlux | 专家副本、token block | replica budget、显存 | routing skew 明显 |
| tile 节奏 | Entwine | tile 生产顺序、SM 配额 | reuse、budget、TP | GEMM–RS / TP |
| 多机传输 | mKernel | SM 分区、NVLink/RDMA 分层 | 网络语义、平台差异 | 多机 fused kernel |

它们不是四个“竞品”，而是四个可以组合的旋钮：

- UniEP 决定“token 按什么顺序进出”；
- MegaFlux 决定“token block 去哪个 GPU”；
- Entwine 决定“tile 什么时候产、通信占多少 SM”；
- mKernel 决定“跨节点怎么搬、用多少 SM 协调网络”。

短期最现实的组合是“UniEP 管单机层内 + MegaFlux 管跨卡放置 + mKernel 管跨节点”，Entwine 的 tile 节奏用于 TP/GEMM–RS 路径。四个都上、硬融成一个 kernel，目前还没有论文证明可行。

## 5. 结论

- **Entwine** 把 tile 级 overlap 的焦点从“融合”转到“节奏 + 配额”：交错生产、独立 tile 通信、共享 SM 池 + communication budget；1.232× geo、最高 1.433×。
- **mKernel** 把同一思路推到多机：SM 角色分工、command queue + host proxy、rail peer、自适应 controller；GEMM+AR 1.72×、Ring Attention 1.88×。
- **MoK/NCCL EP/ThunderEP/FarSkip/HyperParallel** 说明 2026 的生态在分叉：scale-up、标准 API、PCIe、架构改造、Ascend，各自有不同答案。
- 读这一篇的重点不是记数字，而是记住四个旋钮：**顺序、放置、节奏、分层**。前面的系列主要讲“顺序”和“资源”，这一篇补上了“节奏”和“多机分层”。

---

**系列导航**：[补遗·导读]({{< relref "comm-overlap-12-latest" >}}) · [补遗（二）：MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) · [总结：2026 年的地图与五个开放问题]({{< relref "comm-overlap-11-summary" >}})
