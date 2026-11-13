---
title: "通算并行研读（补遗·导读）：四篇会改判断，其余当雷达扫"
date: 2026-11-03
draft: false
weight: 12
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "论文雷达", "UniEP", "MegaFlux", "Entwine", "mKernel"]
categories: ["训练框架"]
---

**2025–2026 这批论文里只有四篇会改你对「通算并行该怎么做」的判断，其余十一篇当雷达按方向扫就行**。

读完你会知道：哪四篇值得花一小时、各自换掉了哪条默认假设、剩下十一篇又在哪个方向报了数。

## 先读这四篇：每篇换掉一条默认假设

第三列是先补哪篇：

| 论文 | 它改了什么 | 先读谁 |
| --- | --- | --- |
| **UniEP**（[2604.19241](https://arxiv.org/abs/2604.19241)）· [深读]({{< relref "comm-overlap-13-uniep" >}}) | 把确定性做成硬约束：两个 MegaKernel + persistent worker + 动态角色，1.03×–1.38× 且 bitwise 可复现 | [MoE 的 A2A]({{< relref "comm-overlap-06-moe" >}}) |
| **MegaFlux**（[2610.00671](https://arxiv.org/abs/2610.00671)）· [深读]({{< relref "comm-overlap-14-megaflux" >}}) | routing skew 下复制热点专家，把副本权重传输与梯度归约塞进前后向流水；8×B200 上 fwd 1.45×、bwd 1.28× | [MoE 的 A2A]({{< relref "comm-overlap-06-moe" >}}) |
| **Entwine**（[2609.11562](https://arxiv.org/abs/2609.11562)）· [深读]({{< relref "comm-overlap-15-tile-ecosystem" >}}) | 重排 tile 产出节奏 + fine-grained SM 通信 + 资源协同；相对 cuBLAS+NCCL 1.232× | [为什么只能切 tile]({{< relref "comm-overlap-05-tensor-parallel" >}}) |
| **mKernel**（[2609.13585](https://arxiv.org/abs/2609.13585)）· [深读]({{< relref "comm-overlap-15-tile-ecosystem" >}}) | 把 fused kernel 推到多机：SM 分区 + NVLink/RDMA 分层 + GPU 侧 command queue；GEMM+AR 1.72× | [SM 到底归谁]({{< relref "comm-overlap-07-kernel-sm" >}}) |

四行按表序各回答一个问题：换配置会不会变成不可归因的实验变更、热点专家为什么拖慢整层、tile 早完成为什么通信不早开始、单机 SM 分区跨节点还成不成立。

## 其余 11 篇：按方向扫

时间顺序没信息量，同方向的论文才是对照组。

### MoE 侧：all-to-all 还是最长的一段

- **Mixture-of-Kittens**（[2609.36070](https://arxiv.org/abs/2609.36070)、[代码](https://github.com/cursor/mixture-of-kittens)）：NVL72 上按算子选 push/pull，最高 2.37×、512 GPU 生产 1.41×。
- **ThunderEP**（[2609.40093](https://arxiv.org/abs/2609.40093)）：PCIe 卡上去掉 ring relay、少轮询 completion flag，端到端 1.66×。
- **FlowMoE**（[2510.00207](https://arxiv.org/abs/2510.00207)）：MHA、gating、expert、A2A 放进同一条 pipeline，训练时间 −13%–−57%。
- **NCCL EP**（[2603.13606](https://arxiv.org/abs/2603.13606)、[基于 nccl](https://github.com/NVIDIA/nccl)）：dispatch/combine 成了 Device API 的正式原语，低延迟档 1–128 token、高吞吐档 4096+。

### TP 与 tile 侧：通信被压到 tile 粒度

- **ParallelKittens**（[2511.13940](https://arxiv.org/abs/2511.13940)）：8 个原语 + 不到 50 行 device code，DP/TP 2.33×、SP 4.08×。
- **Iris**（[2511.12500](https://arxiv.org/abs/2511.12500)）：用 Triton 在 tile 级发通信，GEMM+AllScatter 1.79×。

### 编译器侧：不改 kernel，改参数或改图

- **Resource-aware overlap**（[2606.09200](https://arxiv.org/abs/2606.09200)）：只调 occupancy 和通信 stream 优先级，最多省 25.5% 时间。
- **FarSkip-Collective**（[2511.11505](https://arxiv.org/abs/2511.11505)）：改 skip connection 让 blocking 通信可 overlap，DeepSeek-V3 prefill overlap 97.3%。

### 异构与多机侧：换硬件或拓扑

- **HyperParallel-MoE**（[2605.23764](https://arxiv.org/abs/2605.23764)、[代码](https://gitcode.com/mindspore/hyper-parallel/tree/master/hyper_parallel/core/multicore)）：Ascend NPU 上的 AIC/AIV 异构 tile 调度，Dispatch-to-Combine 延迟 1.58×。
- **HybridEP**（[2510.19470](https://arxiv.org/abs/2510.19470)）：跨数据中心不硬 overlap，动态改专家放置，受限带宽 5.6×。
- **Piper**（[2605.05049](https://arxiv.org/abs/2605.05049)）：HPC 上做 pipeline hybrid，MFU 比 X-MoE 高 2–3.5×。

## 三条会改地图的结论

**确定性从加分项变成选型列**：UniEP 把 bitwise 当硬约束，位不一致就不再算「精度看起来没掉」。代价在个位数到两位数百分比这一档，细节见 [UniEP 那篇]({{< relref "comm-overlap-13-uniep" >}})。

**overlap 不必依赖多 stream**：persistent kernel 加动态 SM 角色和 scoreboard，token/tile 级 overlap 在单 CUDA stream 内就能做完，mKernel 把同一套思路推到多机。代价是 kernel 复杂度与 CUDA Graph 兼容性。

**手写常数正在失效**：形状、拓扑、routing 一变，调好的 chunk 和 SM 配额就不对；Syncopate 枚举 backend × chunk × tile order，Lagom 搜资源参数。搜得起本身就是工程能力。

## 读哪篇：看你手上有多少时间

只够一篇就读 UniEP：它改的是语义层判断，另外三篇改的是局部手段。

真要动手，先抄 Entwine 的 tile 节奏；mKernel 那步跨节点，得先有 RDMA 侧的配合。

> 口径与边界：2026-11-03 用四组 arXiv 查询按提交时间倒序扫描，得到 2025-10 之后的 117 条候选、人工筛到 15 篇；只覆盖 arXiv，不是系统综述、没有纳入标准，117 与 15 都只是这一次扫描的结果。

---

**系列导航**：[十一：总结]({{< relref "comm-overlap-11-summary" >}}) · [补遗（一）：UniEP]({{< relref "comm-overlap-13-uniep" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
