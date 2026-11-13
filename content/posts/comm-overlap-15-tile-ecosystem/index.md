---
title: "通算并行研读（补遗三）：tile 早完成 ≠ 通信能早开始——Entwine、mKernel 与 2026 生态"
date: 2026-11-03
draft: false
weight: 15
series: ["comm-overlap"]
tags: ["通算并行", "系列", "Entwine", "mKernel", "tile", "NCCL EP", "Mixture-of-Kittens", "异构硬件"]
categories: ["训练框架"]
---

**把「让 tile 尽早算完」换成「让 tile 按消费节奏产出」，通信窗口才真的前移；代价是要重排整个产出调度，还要让 SM 资源在通信与计算之间动态让步，让步多少得按 workload 试出来**。

**tile 早完成 ≠ 通信能早开始**。消费者什么时候才真的需要这份数据，决定了通信窗口从哪一刻开始；生产者提前算完，只是把结果先堆在 smem 和 buffer 里。

读完你会知道：为什么算得快不等于发得早；让节奏对齐的三个动作各要付什么；同一思路推到多机后哪条假设先断；生态里剩下的方向凭什么各自成立。

## 1. 算得快，不等于发得早

一个 TP 里的 row-parallel GEMM 接 ReduceScatter：输出按 tile 切、block 独立调度，一个 tile 算完理论上就能发给需要它的 rank。Entwine（[2609.11562](https://arxiv.org/abs/2609.11562)）在它的 workload 上串行跑一遍，通信仍然暴露 **11.0%–40.2%** 的时间，中位数 **23.3%**。这是串行 baseline 的暴露量，不是 overlap 之后的残余。

反例很具体：A 卡算完一个 tile，确实立刻能发；可 B 卡此刻还在算别的 tile，收到也只能先在 buffer 里排着。「提前算完」改的是生产者的完成时刻，不改消费者的需求时刻。

第一个错位在产出顺序。producer 按输出 partition 连续遍历，会先一路算「自己保留」的 tile，这段时间没有跨 GPU 数据可发；等它切到远程 partition，数据又一次性突发。通信单元要么闲着，要么滞后。

第二个错位在资源。通信 block 也占 SM、占 HBM 带宽，通信开得越多，GEMM 越慢。所以「把通信提前」不是免费的，先得回答通信能占多少 SM。

两个错位合起来就是这一篇的主线：**需求时刻由消费者决定，不由生产者决定**。

## 2. 三个动作，把产出顺序对齐消费顺序

Entwine 同时动三处：产出顺序、通信粒度、SM 配额。

### 重排产出节奏，让产出顺序对上消费顺序

不按 partition 连续遍历，而是按逻辑 block 轮转输出 partition。把输出分成 $W$ 个等宽、block 对齐的 partition，producer block 的逻辑位置 $p$ 映射到

$$
d = p \bmod W, \quad \ell = \lfloor p / W \rfloor
$$

$d$ 是目标 rank，$\ell$ 是 partition 内 block 索引。所有 rank 共用同一映射，同一个 tile 的贡献就落在同一逻辑位置。

于是「本地保留」的 tile 被均匀插进「需要远程传输」的 tile 之间，任何时刻手里都有新东西可发，数据不再一阵一阵。

代价是 reuse 可能下降：连续访问同一 partition 对 cache 更友好，交错会打散局部性。Entwine 把这一项算进 end-to-end，而不是只看 GEMM 吞吐。

### 把通信切到一个 tile，而不是一个 wave

每个通信任务只处理一个输出 tile。一个 tile 的 ReduceScatter 只依赖各 rank 对它的贡献，不依赖别的 tile，所以贡献到齐就能归约，不用等一个 wave/group 的所有 tile。

这和 FlashOverlap 按 wave 触发 collective 的路子相反：group 能凑出更大消息、带宽利用率更高，但组内一个 tile 会被无关输出拖住。Entwine 用更细的粒度换更短的等待，代价是消息切小后带宽利用率下降。

### 让 SM 在通信与计算之间动态让步

一个 compute kernel 和一个 communication kernel 并发跑，不按 tile 逐个 launch；block 退休后释放的 SM，谁的 pending work 就归谁。资源不预留、按需抢，计算才不会为了「等通信」空转。

communication budget 限制「同时跑几个通信 block」：太小，通信尾巴长、暴露多；太大，通信抢 SM 和带宽、GEMM 变慢。最优值由 end-to-end latency 决定，不是「通信尾巴最短」。

结果：相对 cuBLAS+NCCL 几何平均 **1.232×**、最高 **1.433×**；相对 SOTA overlap baseline 几何平均高 **3.1%–9.8%**。口径是 GEMM–ReduceScatter / TP LLM，不能外推到 MoE all-to-all，那里的依赖和 budget 都得重设计。

那个最优 budget 也随形状变：$K=2048$ 时增大 budget 先快速降延迟，随后进入一段很宽的平台区；$K=512$ 时延迟有一个内部极小值，再加大不再收益。没有通用常数，只能按 workload 离线 profile。

代码目前还没有公开仓库，论文的原话是「will open-source upon publication」。

Entwine 和第五篇的 Flux/TileLink 是一条线（[为什么只能切 tile]({{< relref "comm-overlap-05-tensor-parallel" >}})）：Flux 用 swizzling 把通信顺序对齐，Entwine 干脆把「节奏」本身当优化对象。瓶颈从「能不能融」变成「产出节奏和 SM 配额匹不匹配」。

## 3. mKernel：单机的 SM 分区假设，跨节点就断了

- **SM 分区**：persistent kernel 里 compute、intra-node comm、send、receive 四类 block 常驻，通信 SM 数由 controller 按 progress counter 调，和同节点其余 7 个 GPU 交换；节点内 sweep 的最优值落在 **2–64** 之间，同一 kernel 里平衡还会变，定死没用。
- **按互联分层**：节点内 NVLink **450 GB/s**、HBM **4.8 TB/s**，跨节点 IB/EFA 每 GPU 每方向只有 **50 GB/s**，差一个数量级；所以先在本地聚合、按 rail 对齐 peer，再把细粒度本地 tile 聚成网络 chunk 摊销 per-message 成本。
- **GPU 侧 command queue**：send block 只把命令写进 host 侧队列，payload 留在 GPU memory，host proxy 批量提交摊掉 doorbell；2 个 16-GPU H200 集群（一个 IB、一个 AWS EFA）上 GEMM+AllReduce **1.72×**、Ring Attention **1.88×**，代码在 [uccl-project/mKernel](https://github.com/uccl-project/mKernel)，论文见 [2609.13585](https://arxiv.org/abs/2609.13585)。反直觉的是 GPU 直接发起 RDMA（IBGDA）没有额外收益，瓶颈在消息粒度、flag 和 SM 争用。

## 4. 生态雷达：五条分支，各换掉一个前提

| 方向 | 代表 | 换了什么，换到多少 |
| --- | --- | --- |
| 往上堆规模 | **Mixture-of-Kittens**（[2609.36070](https://arxiv.org/abs/2609.36070)） | 72 个 GPU 用 NVLink 交换机连成一个单跳域（NVL72），把 MoE 全融成一个 deterministic megakernel，按算子选 push 或 pull；最高 2.37×，512 GPU 生产 1.41×，[代码](https://github.com/cursor/mixture-of-kittens) |
| 往下立标准 | **NCCL EP**（[2603.13606](https://arxiv.org/abs/2603.13606)） | dispatch/combine 变成 Device API 原语，框架直接调、不用自养一套栈；1–128 token 走 LL、4096+ 走 HT，H100 多节点验证 |
| 换硬件：PCIe | **ThunderEP**（[2609.40093](https://arxiv.org/abs/2609.40093)） | 没有 NVLink 的 RTX 4090/5090 上去掉 ring relay、用 DMA 搬数据不占 compute SM；dispatch 2.00×、combine 1.53× vs NCCL、e2e 最高 1.66× |
| 换硬件：Ascend、HPC、跨 DC | **HyperParallel-MoE**（[2605.23764](https://arxiv.org/abs/2605.23764)）、**Piper**（[2605.05049](https://arxiv.org/abs/2605.05049)）、**HybridEP**（[2510.19470](https://arxiv.org/abs/2510.19470)） | AIC/AIV 异构 tile 调度，Dispatch-to-Combine 1.58×；HPC 上 MFU 比 X-MoE 高 2–3.5×、A2A 带宽 1.2–9×；跨 DC 不硬 overlap、改专家放置，受限带宽 5.6× |
| 改模型，或只改参数 | **FarSkip**（[2511.11505](https://arxiv.org/abs/2511.11505)）、**Resource-aware**（[2606.09200](https://arxiv.org/abs/2606.09200)）、**FlowMoE**（[2510.00207](https://arxiv.org/abs/2510.00207)） | 改 skip connection 让 blocking 通信可藏：把 16B–109B 的模型转成可 overlap 版本，下游平均精度在原始 instruction-tuned 版本的 1% 以内，DeepSeek-V3 prefill overlap 97.3%、TTFT 32.6%、A2A 88.9%；只调 occupancy 和通信 stream 优先级，A40/A100/H100/MI250X 上最多省 25.5% 时间；MHA、gating、expert 进同一条 pipeline，训练时间 −13%–−57%、能耗 −10%–−39% |

共同信号：**一套 Hopper NVLink 的最优解不再通用**。scale-up、scale-out、PCIe、Ascend、跨 DC，各自要不同的通信原语、SM 配额和放置策略。

## 5. 把四条线合起来：它们在拧同一个变量

顺序、放置、节奏、分层，看起来是四件事，动的都是「数据什么时候被谁真的需要」。

- UniEP 管单节点内 token 的进出次序（8 卡 EP、bitwise 可复现），决定谁先到、谁后到；MegaFlux 管 token block 去哪个卡，决定要不要等最慢那张。
- Entwine 管 tile 什么时候产、通信占多少 SM，把「产出时刻」排进消费时间表；mKernel 管跨节点分几层搬、留多少 SM 协调网络，把「到达时刻」压到网络最快的那一段上。
- 所以能叠，不能互替：任何一层没对齐，暴露就从那一层漏回来。短期最现实的是三层各管一段——UniEP 层内、MegaFlux 放置、mKernel 跨节点，Entwine 的节奏只走 TP；四个全上融成一个 kernel，目前还没有论文证明可行。

**如果只记一件事**：问「通信为什么没提前」之前，先问「这份数据什么时候才真的被需要」。

---

**系列导航**：[补遗·导读]({{< relref "comm-overlap-12-latest" >}}) · [补遗（二）：MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) · [总结：一张按粒度分层的地图]({{< relref "comm-overlap-11-summary" >}})
