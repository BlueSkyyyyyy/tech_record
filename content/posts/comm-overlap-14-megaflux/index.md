---
title: "通算并行研读（补遗二）：MegaFlux 深读——routing skew、专家复制与前后向流水"
date: 2026-11-03
draft: false
weight: 15
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "MegaFlux", "routing skew", "专家复制", "B200"]
categories: ["训练框架"]
---

UniEP 解决的是“单层内的顺序和调度”，MegaFlux（[2610.00671](https://arxiv.org/abs/2610.00671)）解决的是另一个老问题：

> **routing 不均匀时，固定专家放置会让最慢的 GPU 决定整层延迟。**

这两件事正交：UniEP 管 token 怎么排、SM 怎么切；MegaFlux 管专家副本放哪、token block 分给谁。把两者叠起来，才是“顺序 + 负载”都顾到的 MoE overlap。

这一篇拆 MegaFlux 的四个部分：怎么量化 skew、怎么规划复制、怎么把复制通信塞进前后向、以及它的收益和边界。

## 1. 问题：固定放置下，skew 直接变成 straggler

MoE 的 expert 通常按固定规则放在 EP rank 上。热 expert 所在的 GPU 会收到更多 token，计算更久；冷 expert 的 GPU 早早算完，等 all-to-all/combine。层延迟由最慢的 GPU 决定，其他 GPU 的算力被浪费。

MegaFlux 用 DA-MoE 的 skew 指标量化：

$$
\kappa = E \sum_{e=1}^{E}\left(\frac{c_e}{\sum_j c_j}\right)^2
$$

其中 $c_e$ 是专家 $e$ 收到的 token 数，$E$ 是专家数。均匀路由 $\kappa=1$，越大越不均。

论文在 8×B200（EP8）上回放 OLMoE / Qwen3 等 workload 的 P95-skew 路由，和“只改 expert ID、保持 token 数不变”的 balanced 路由对比：

- P95 路由比 balanced 路由延迟高 **7.3%–41.0%**；
- 即使底座已经是 CuTeDSL MegaMoE（计算和通信已经 pipeline），固定放置仍然把 skew 暴露在关键路径上。

这说明：**只做 kernel 内 overlap 不够，放置本身要动态化。**

## 2. Planner：以 token block 为单位的复制规划

### 2.1 为什么按 block，不按 token

如果把一个专家的 token 均分给多个副本，每个副本都可能拿到不满一个 tile 的尾部，产生 partial tile，浪费 Tensor Core。MegaFlux 的做法是：

- 把每个专家的 routed rows 切成 **最多 $\beta$ 行**的 token block，$\beta$ 与 megakernel 的 FC1/FC2 tile 的 token 维对齐；
- 每个专家最多只有一个 partial block，不管复制多少份；
- 复制和分配都以 block 为单位。

这样计算负载可以用 block 数近似，规划问题变成整数规划。

### 2.2 形式化：min-max 负载 + 复制预算

设：

- $c_e$：专家 $e$ 的总 token 数；
- $h(e)$：专家 $e$ 的 owner GPU；
- $b_e = \lceil c_e / \beta \rceil$：block 数；
- $q_{er}$：分给 GPU $r$ 的专家 $e$ 的 block 数；
- $s$：每个 GPU 的 replica budget（最多复制多少个远程专家）。

目标：

$$
\min_{q \in \mathbb{Z}_{\ge 0}^{E \times P}} \max_r \ell_r,
\quad \ell_r = \sum_e q_{er}
$$

约束：

$$
\sum_r q_{er} = b_e \quad \forall e
$$

$$
\sum_{e: h(e) \ne r} \mathbf{1}[q_{er} > 0] \le s \quad \forall r
$$

第一条约束保证每个专家的 block 都分完；第二条限制每个 GPU 最多接 $s$ 个远程专家的副本。$q_{er}>0$ 且 $r \ne h(e)$ 就表示“在 $r$ 上开一个副本”。

这是一个 min-max 负载均衡 + 复制容量约束的问题。精确解很贵，MegaFlux 用**有界贪心搜索**。

### 2.3 四个贪心搜索

MegaFlux 并行跑四个贪心：

- 起点：no-replica 分配；
- 目标水位：$\tau = \lceil \sum_e b_e / P \rceil$；
- 每轮把 block 从过载 owner 往 $\tau$ 搬；
- 优先用已有副本，再在最空的合格 GPU 上开新副本；
- donor 选择有两种优先级：减少总过载，或减少最大 home-expert 负载；
- fan-out 有两种：每轮每个专家最多开一个新副本，或一次开多个；
- 每轮每个过载 owner 访问一次；owner 负载降到 $\tau$ 以下或专家扫完就结束；
- 搜索在 $s+2$ 轮后停止，或某一轮没有 block 移动时停止；
- 先最小化最大负载，再按副本数打破平局，最后剪掉冗余副本、做不增加最大负载的微调。

四个候选由四个 warp 在 device 上并行评估；选最优后，planner 协作生成 `block → GPU` 表。每个专家内部用 stable ordering 给 routed assignment 一个 block index，这个 index 直接选目标 GPU。

### 2.4 backward 的 reduction-aware 变体

forward 的复制只增加“权重传输”；backward 还会增加“副本梯度归约”：

- 副本算出的 partial WGrad 要送回 owner；
- owner 要等所有参与 GPU 的 partial 到齐才能归约；
- 这会影响 owner 的负载。

所以 backward 用 reduction-aware 规划，在剪枝时用：

$$
\max_r (\ell_r + \lambda n_r)
$$

其中 $n_r$ 是“owner 为 $r$ 的专家在远程有多少副本”，$\lambda=12$（论文对所有报告的 backward 配置固定这个值）。这相当于给“远程副本多的 owner”加惩罚，避免规划出一个“计算均衡、归约爆炸”的方案。

## 3. Executor：把复制通信塞进 megakernel

规划只是“放哪”，执行才是“怎么藏”。MegaFlux 的 executor 有两个关键设计。

### 3.1 Forward：W1/W2 分离 readiness

一个副本要算 FC1 + FC2，但两份权重的到达时间可以不同：

- FC1 只需要 $W_1$ 和输入 token block；
- FC2 需要 $W_2$ 和 FC1 的输出 activation。

所以 MegaFlux 不要求“副本权重全部到齐”才开始：

- $W_1$ 到 → 开始 FC1；
- $W_1$ 计算期间，$W_2$ 传输；
- FC1 完成 → 等 $W_2$ → 开始 FC2；
- owned expert 的计算也能和副本传输重叠。

论文 Figure 2 的 forward 时间线就是：`W1 transfer → FC1 || W2 transfer → FC2`。

实现上，副本权重传输**复用通信 warp**，不单独保留一组 worker：

- 一小部分 warp 从 kernel 入口就服务 replica-weight queue；
- 其余 input-transfer warp 先排空自己负责的 token block，再加入同一个队列；
- 这样副本传输资源随 token 传输减少而“逐渐变多”，不需要副本级 barrier。

### 3.2 Backward：DGrad 与 WGrad 的流水

Backward 的依赖更绕：

- DGrad 先消费 $W_2$，再消费 $W_1$；
- 所以副本权重传输按 $W_2 \to W_1$ 的顺序 readiness；
- WGrad 需要 token 轴上的 $D_e$ 和 $dZ_e$，MXFP8 下还要做 requantization（RQ）；
- RQ 被安排在通信空闲区间；
- 每个 worker 完成自己的 DGrad 后独立转入 WGrad，不需要 device-wide phase barrier。

最关键的是**副本梯度归约的同步**：owner 只有等所有参与 GPU 的 matching partials 都到了，才能归约一个 gradient tile。MegaFlux 的做法是：

- 在所有参与者上优先准备“被复制专家”的 WGrad；
- 优先生产这些专家的 gradient tile；
- 让 owner 的归约尽早开始，和剩余计算重叠；
- 用固定顺序做 FP32 累加，保证结果可复现。

这和第 3.1 节的“分离 readiness”是同一个思想：**把“等全部”拆成“等一个 tile/一个投影”**。

## 4. 收益：1.45×/1.28× 与 56%–100% 隐藏率

实验设置：

- 8×B200，单 NVLink domain，EP8；
- hidden/intermediate 7168/2048，SwiGLU，top-k 8；
- 147 个配置/方向，覆盖不同 routing 分布和 workload 大小；
- 基线是“同样 megakernel、固定放置”的 MegaMoE，不是朴素 PyTorch。

结果：

| 指标 | Forward | Backward |
|---|---:|---:|
| 几何平均加速 | **1.45×** | **1.28×** |
| 峰值加速 | **2.14×** | **2.64×** |
| 复制权重传输隐藏率 | **56%–76%** | — |
| 权重传输 + 梯度归约隐藏率 | — | **91%–100%** |
| vLLM DeepSeek-V4-Pro prefill e2e | **1.13×–1.26×**（中位数） | — |

几个读法：

- **forward 的 1.45× 是几何平均，不是峰值**；峰值 2.14× 出现在 skew 很重的配置；
- **backward 的隐藏率更高（91%–100%）**，因为 backward 本来就有更多可重叠的 DGrad/WGrad 工作；
- **vLLM e2e 只有 1.13×–1.26×**，因为 prefill 里还有 attention、KV cache、非 MoE 层；这和 UniEP 的 e2e 稀释是同一个规律。

## 5. 边界：复制不是免费的

MegaFlux 的收益有明确前提：

1. **routing 必须够 skew**。如果路由接近均匀，复制带来的额外权重流量和显存占用可能超过收益；论文的 workload 是 P95-skew，不是均匀分布。
2. **replica budget $s$ 是硬约束**。$s$ 越大，负载越均衡，但副本权重流量、显存、归约参与者都增加；论文没有给“最优 $s$”的通用公式，而是让 planner 在预算内搜。
3. **planner 有开销**。四个贪心 + 四个 warp 并行，虽然 on-device，但仍要占 SM/寄存器；论文没有单独报 planner 的延迟占比。
4. **它不改 router**。复制只改“谁算”，不改“选哪些专家”；所以它和 capacity factor、drop policy 是正交的，不能解决“路由本身负载不均”的问题。
5. **代码未公开**。论文没有给仓库链接；复现需要自己实现 planner + megakernel 集成。

## 6. 和 UniEP 的关系：一个管“顺序”，一个管“放置”

| 维度 | UniEP | MegaFlux |
|---|---|---|
| 优化对象 | 层内 token 顺序、SM 角色、通信/计算 overlap | 跨 GPU 的专家副本放置、token block 分配 |
| 关键约束 | bitwise 可复现 | replica budget、负载均衡 |
| 并行载体 | 单 stream persistent kernel + 动态任务队列 | 同 megakernel 内的复制传输/归约流水 |
| 主要收益 | kernel 1.3×–1.6×，e2e 1.09× | fwd 1.45×、bwd 1.28×（vs 固定放置） |
| 失败模式 | 跨节点、非 Hopper、shape 太动态 | routing 均匀、副本预算太小、显存不够 |

两者可以叠：UniEP 决定“token 按什么顺序进/出”，MegaFlux 决定“token block 去哪个 GPU 算”。但叠加也有交互：

- UniEP 的确定性映射假设“目标专家固定”；MegaFlux 把 block 分给副本，会改变 token 在哪个 GPU 上被算，确定性映射要扩展到“副本级”；
- MegaFlux 的副本梯度归约需要固定累加顺序，UniEP 的 scoreboard 需要感知“副本 partial 是否到齐”；
- 两者都依赖 on-device 调度，SM 预算会竞争。

所以短期更现实的组合是“UniEP 管单机层内，MegaFlux 管跨卡放置”，而不是把两个 megakernel 硬融成一个。

## 7. 结论

MegaFlux 的核心不是“复制专家”这个点子——MoE 负载均衡里早有 expert replication——而是：

- 把复制做成 **block 粒度**的整数规划，避免 partial tile；
- 把复制权重传输、副本梯度归约 **拆成 tile/投影级 readiness**，塞进现有 megakernel；
- 用 **reduction-aware** 目标避免“计算均衡但归约爆炸”；
- 在 8×B200 上把 forward/backward 几何平均做到 **1.45×/1.28×**，峰值 **2.14×/2.64×**。

它的边界也很清楚：routing 越 skew 越有用，副本预算和显存是硬约束，planner 本身也有成本。读它的时候，重点看 **planner 的目标函数和 executor 的 readiness 拆分**，而不是只记 1.45×。

---

**系列导航**：[总览]({{< relref "comm-overlap-12-latest" >}}) · [补遗（一）：UniEP]({{< relref "comm-overlap-13-uniep" >}}) · [补遗（三）：Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
