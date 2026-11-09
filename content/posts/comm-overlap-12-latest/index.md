---
title: "通算并行研读（补遗·导读）：2026 最新论文雷达与阅读路线"
date: 2026-11-03
draft: false
weight: 12
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "论文雷达", "UniEP", "MegaFlux", "Entwine", "mKernel"]
categories: ["训练框架"]
---

本系列原先只有一篇补遗，现在拆成**总览 + 三篇深读**。原因很直接：一篇三五千字的文章，把 UniEP 的确定性映射、MegaFlux 的复制规划、Entwine 的 tile 节奏、mKernel 的多机 RDMA 全塞进去，结果只能是“每篇一段摘要”，不是研读。

这一篇是**总览/雷达**：只回答“有哪些、为什么值得看、先看哪篇”；真正的拆解放到三篇深读里。

> 调研口径：2026-11-03 用四组 arXiv 查询（`expert parallelism + overlap`、`computation-communication + overlap`、`MoE + overlap`、`cs.DC + communication + overlap`）按提交时间倒序扫描，得到 2025-10 之后的 117 条候选，再人工筛到 15 篇。只覆盖 arXiv，不保证覆盖会议论文和工业界未公开工作。
>
> 再补一句边界：这是截至 2026-10 提交的 arXiv 手工检索，非系统综述，没有严格的检索式和纳入/排除标准；117 和 15 都只是这一次扫描的结果。

## 1. 先看这四篇

这四篇是入口，按你手上的问题挑，不用按顺序读。

| 论文 | 一句话 | 深读 |
|---|---|---|
| **UniEP**（[2604.19241](https://arxiv.org/abs/2604.19241)） | 两个 MegaKernel + persistent worker + 动态角色 + 确定性 token mapping；1.03×–1.38×，bitwise（逐位一致）可复现 | [补遗（一）]({{< relref "comm-overlap-13-uniep" >}}) |
| **MegaFlux**（[2610.00671](https://arxiv.org/abs/2610.00671)） | routing skew（专家负载不均）下复制热点专家，把副本权重传输/梯度归约塞进前后向流水；8×B200 fwd 1.45×、bwd 1.28× | [补遗（二）]({{< relref "comm-overlap-14-megaflux" >}}) |
| **Entwine**（[2609.11562](https://arxiv.org/abs/2609.11562)） | 重排 tile 产出节奏 + fine-grained SM 通信 + 资源协同；相对 cuBLAS+NCCL 1.232× | [补遗（三）]({{< relref "comm-overlap-15-tile-ecosystem" >}}) |
| **mKernel**（[2609.13585](https://arxiv.org/abs/2609.13585)） | 把 fused kernel 推到多机：SM 分区 + NVLink/RDMA 分层 + GPU 侧 command queue；GEMM+AR 1.72× | [补遗（三）]({{< relref "comm-overlap-15-tile-ecosystem" >}}) |

这四篇代表四个不同方向：**语义/调度**（UniEP）、**负载分布**（MegaFlux）、**tile 节奏**（Entwine）、**多机数据移动**（mKernel）。它们不是互相替代，而是可以叠：UniEP 管层内顺序，MegaFlux 管跨 GPU 放置，Entwine 管 tile 生产节奏，mKernel 管跨节点传输。

## 2. 雷达：其余 11 篇

下面按 arXiv 首次提交时间倒序排（新 → 旧）；同一个月内的顺序不代表重要性。

| 时间 | 工作 | 一句话 | 代码/artifact |
|---|---|---|---|
| 2026-09 | **Mixture-of-Kittens**（[2609.36070](https://arxiv.org/abs/2609.36070)） | NVL72 scale-up 上的 MoE 训练 megakernel；按算子选 push/pull、重排 overlap、去掉 CPU-GPU 同步；最高 **2.37×**，512 GPU 生产 **1.41×** | [cursor/mixture-of-kittens](https://github.com/cursor/mixture-of-kittens) |
| 2026-09 | **ThunderEP**（[2609.40093](https://arxiv.org/abs/2609.40093)） | PCIe 消费级 GPU 的 EP 通信：去掉 ring relay、用 DMA engine、降低 completion flag 轮询；dispatch **2.00×**、combine **1.53×**、e2e **1.66×** | 论文未给公开仓库 |
| 2026-06 | **Resource-aware overlap**（[2606.09200](https://arxiv.org/abs/2606.09200)） | 不改库/kernel，只调 occupancy 和通信 stream 优先级；A40/A100/H100/MI250X 上最多 **−25.5%** 时间 | 未给公开仓库 |
| 2026-05 | **HyperParallel-MoE**（[2605.23764](https://arxiv.org/abs/2605.23764)） | Ascend NPU 的 AIC/AIV 异构 tile 调度；Dispatch-to-Combine 延迟最高 **1.58×** | [gitcode](https://gitcode.com/mindspore/hyper-parallel/tree/master/hyper_parallel/core/multicore) |
| 2026-05 | **Piper**（[2605.05049](https://arxiv.org/abs/2605.05049)） | HPC 上的 MoE 资源建模 + pipeline hybrid parallelism；MFU 比 X-MoE 高 **2–3.5×**，all-to-all 带宽 **1.2–9×** | 未给公开仓库 |
| 2026-03 | **NCCL EP**（[2603.13606](https://arxiv.org/abs/2603.13606)） | 把 EP dispatch/combine 做成 NCCL Device API 的标准原语：`ncclEpDispatch` / `ncclEpCombine`，LL 1–128 token，HT 4096+ token | 基于 [NVIDIA/nccl](https://github.com/NVIDIA/nccl) Device API，论文未给独立仓库 |
| 2025-11 | **FarSkip-Collective**（[2511.11505](https://arxiv.org/abs/2511.11505)） | 改模型结构（skip connection）让 blocking communication 可 overlap；DeepSeek-V3 prefill **97.3%** overlap、TTFT **32.6%**；训练 A2A **88.9%** | 未确认公开代码 |
| 2025-11 | **ParallelKittens / Iris**（[2511.13940](https://arxiv.org/abs/2511.13940)、[2511.12500](https://arxiv.org/abs/2511.12500)） | 多 GPU kernel 的“最小原语集”：PK 用 8 个原语 + <50 行 device code，DP/TP **2.33×**、SP **4.08×**、EP **1.22×**；Iris 用 Triton 做 tile 级通信，GEMM+AllScatter **1.79×** | 论文/项目页，未逐一确认仓库 |
| 2025-10 | **HybridEP**（[2510.19470](https://arxiv.org/abs/2510.19470)） | 跨数据中心 EP：低带宽下不硬 overlap，而是动态改专家放置/传输比例；受限带宽最高 **5.6×** | 未给公开仓库 |
| 2025-10 | **FlowMoE**（[2510.00207](https://arxiv.org/abs/2510.00207)） | 把 MHA、gating、expert、A2A、all-reduce 放进统一 pipeline；训练时间 **−13%–−57%**，能耗 **−10%–−39%** | 未给公开仓库 |

这张表只用来查“哪篇讲过什么”。数字不要横向比：每篇的 baseline、硬件、模型规模都不同。

## 3. 三条会改地图的结论

这一节只留三条我认为会改变选型的结论。

### 3.1 语义：从“精度不掉”到“逐位一致”

UniEP 把 **bitwise identical** 作为 overlap 的硬约束。论文实测：Comet 的 forward/backward 输出最多有 **0.25** 的绝对误差，**22%–29%** 的元素不满足 bitwise 相等；UniEP 在 12 个配置上误差为 0。

这意味着第六篇的选型表要加一列“是否保证 deterministic/bitwise”。不是所有场景都需要，但生产训练里，“换一个 overlap 配置”不应该变成一次不可归因的实验变更。

### 3.2 载体：单 stream 内也能做细粒度 overlap

UniEP 用 persistent kernel + 动态 SM 角色 + scoreboard，在**单 CUDA stream** 内做 token/tile 级 overlap；mKernel 也把 SM 分区和跨节点传输放进 persistent kernel。前面默认的“双 stream / 多 stream 才有 overlap”不再是唯一答案。

代价是 kernel 复杂度、调试难度、和 CUDA Graph/框架调度的兼容性；收益是没有 host 侧 stream 管理的 alignment bubble。

### 3.3 自动调参：从加分项变成必选项

UniEP 的搜索空间约 $10^5$，Python 版遍历要 **~100 s**，改成 C++/OpenMP 后 **~144 ms**；再用 4096-token bucketing 做 memoization。Syncopate 枚举 backend × chunk × tile order，Lagom 搜资源参数——**形状/拓扑/routing 一变，手写常数就失效**。

“搜得起”本身是工程能力，不是一句“we autotune”能带过。

## 4. 阅读路线

这一节按“你想解决什么问题”给路线，不给必读顺序。

- 想先看**为什么生产框架不敢用**：读 [UniEP 深读]({{< relref "comm-overlap-13-uniep" >}})，重点是确定性映射和 Comet 的 22%–29% 非 bitwise；
- 想解决**热点专家拖慢整层**：读 [MegaFlux 深读]({{< relref "comm-overlap-14-megaflux" >}})，重点是 block 粒度复制规划和前后向 readiness；
- 想先看**tile 级 overlap 的下一步**：读 [Entwine/mKernel 与生态]({{< relref "comm-overlap-15-tile-ecosystem" >}})，重点是节奏、SM 配额、NVLink/RDMA 分层；
- 只想要一张表：这篇就是。

---

**系列导航**：[十一：总结]({{< relref "comm-overlap-11-summary" >}}) · [补遗（一）：UniEP]({{< relref "comm-overlap-13-uniep" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
