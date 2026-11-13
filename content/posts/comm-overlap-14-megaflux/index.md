---
title: "通算并行研读（补遗二）：MegaFlux 深读——复制热点专家，再把副本通信塞进流水"
date: 2026-11-03
draft: false
weight: 14
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "MegaFlux", "routing skew", "专家复制", "B200"]
categories: ["训练框架"]
---

**把「固定专家放置」换成「按 token 块复制热点专家」，让最慢那张卡不再决定整层时间；代价是副本权重要传、梯度要额外归约，必须塞进前后向流水才不亏**。

读完你会知道：

- skew 到多重，复制才划得来；
- 「复制几个」怎么变成一条水位线；
- 副本权重和梯度归约，怎么塞进 megakernel 而不互相等。

## 1. skew 一重，最慢那张卡就是整层的时间

MoE 的专家按固定规则放在 EP rank 上。热门专家所在的 GPU 收到更多 token，算得更久；冷门专家那张卡早早算完，等 all-to-all。整层时间由最慢的卡决定，其余算力空转。

有多重，用 DA-MoE 的 skew 指标量：

$$
\kappa = E \sum_{e=1}^{E}\left(\frac{c_e}{\sum_j c_j}\right)^2
$$

$c_e$ 是专家 $e$ 的 token 数，$E$ 是专家数；均匀路由 $\kappa=1$，越大越不均。

MegaFlux（[2610.00671](https://arxiv.org/abs/2610.00671)）在 8×B200（EP8）上回放 OLMoE / Qwen3 的 P95-skew 路由，和「只改 expert ID、token 数不变」的 balanced 路由对比：延迟高 **7.3%–41.0%**。

底座已经是 CuTeDSL MegaMoE，计算和通信已经 pipeline，固定放置仍把 skew 留在关键路径上。

所以只做 kernel 内 overlap 不够，放置本身要动态化。

## 2. 复制几个：把最大负载压到一条水位线

复制不能按 token 均分。把一个专家的 token 劈给几个副本，每个副本都可能拿到不满一个 tile 的尾巴（partial tile），Tensor Core 白算。

MegaFlux 以 token block（一个专家的一段连续 token 行）为单位：每个专家按最多 $\beta$ 行切块，$\beta$ 对齐 FC1/FC2 tile 的 token 维，每个专家最多一个 partial block，复制和分配都按块走。负载因此近似等于 token block 数，规划变成整数规划。

设专家 $e$ 有 $c_e$ 个 token、$b_e=\lceil c_e/\beta\rceil$ 块，常驻卡（owner）记作 $h(e)$，GPU $r$（共 $P$ 张）分到 $q_{er}$ 块，每张卡最多接 $s$ 个远程专家的副本。目标是把最重的卡压下去：

$$
\min_{q}\ \max_r \ell_r,\qquad \ell_r=\sum_e q_{er}
\quad\text{s.t.}\quad
\sum_r q_{er}=b_e,\quad
\sum_{e:\,h(e)\ne r}\mathbf{1}[q_{er}>0]\le s
$$

判据就一句话：水位线 $\tau=\lceil\sum_e b_e/P\rceil$，把 token block 从超过 $\tau$ 的 owner 搬到 $\tau$ 以下；搬不动就开副本，但每张卡最多 $s$ 个远程专家的副本。

精确解很贵，跑的是四个有界贪心。它们不是四套判据，只是两处选择：donor（往外搬 token block 的卡）先减总过载，还是先减 owner 的最大负载；fan-out 一轮开一个，还是一次开几个。搬的时候优先用已有副本，再在最空的合格 GPU 上开新副本。

四个 warp 在 device 上并行评估，$s+2$ 轮后停，或某一轮没有 token block 移动就停。先压最大负载，再按副本数打破平局，最后剪掉冗余副本。

backward 多一笔账：副本算出的 partial WGrad（对权重的梯度）要回 owner，owner 等齐才能归约。所以 backward 的规划在目标里加 $\lambda n_r$，$n_r$ 是 owner 名下的专家在远程有多少副本，$\lambda=12$；否则会规划出「计算均衡、归约爆炸」的方案。

## 3. 执行：FC2 只等 W2，归约只等一个 tile

```text
forward    W1 ──▶ FC1 ──┐
           W2 传输 ──────┴──▶ FC2      W1 在算时，W2 在传
backward   W2 ──▶ DGrad ──▶ W1
           WGrad：owner 等齐同一 tile 的所有副本 partial ──▶ 归约
```

谁等谁：FC2 等 W2，不等整个副本权重；owner 归约等一个 gradient tile 的所有 partial，不等一整层；本地专家不等副本。

forward 把两份权重的就绪状态（readiness）拆开。W1 到就开 FC1，FC1 算的时候 W2 在传，FC1 完再等 W2 进 FC2。

副本权重传输复用通信 warp：少数 warp 从 kernel 入口就服务 replica-weight 队列，其余 input-transfer warp 排空自己的 token block 后加入同一队列。副本传输资源随 token 传输变少而变多，不需要副本级 barrier。

backward 的依赖反过来：DGrad（对输入的梯度）先吃 W2 再吃 W1。

WGrad 要 token 轴上的 $D_e$ 和 $dZ_e$，MXFP8（一种 FP8 训练格式）下还要一次 requantization（RQ），RQ 被塞进通信空闲区间；每个 worker 做完自己的 DGrad 就转 WGrad，没有 device-wide barrier。

归约那头，被复制专家的 WGrad 优先生产，owner 的归约就能尽早开始、和剩余计算重叠；累加用 FP32 固定顺序，保证结果可复现。两处是同一个手法：把「等全部」拆成「等一个 tile 或一个投影」。

## 4. 几何平均 1.45×/1.28×，副本通信藏住 56%–100%

口径先摆清：8×B200、单 NVLink domain、EP8；hidden/intermediate 7168/2048，SwiGLU，top-k 8；147 个配置/方向，覆盖不同 routing 分布和 workload 大小；基线是同样 megakernel、固定放置的 MegaMoE，不是朴素 PyTorch。

| 指标 | Forward | Backward |
|---|---:|---:|
| 几何平均 | **1.45×** | **1.28×** |
| 峰值 | **2.14×** | **2.64×** |
| 权重传输隐藏率 | **56%–76%** | — |
| 权重传输+梯度归约隐藏率 | — | **91%–100%** |
| vLLM prefill e2e | **1.13×–1.26×** | — |

vLLM 上 DeepSeek-V4-Pro 的 prefill 端到端只有 1.13×–1.26×（中位数）：attention、KV cache、非 MoE 层把收益稀释了，和 UniEP 的 e2e 稀释是同一个规律。

两个数别混：1.45× 是几何平均，不是峰值；2.14× 出现在 skew 很重的配置。backward 隐藏率更高（91%–100%），因为 DGrad/WGrad 本来就有更多可叠的活。

论文只报了这套配置，换 routing 或换互联都不作数。

## 5. 边界：routing 不够 skew，复制就是纯开销

1. routing 必须够 skew。接近均匀时，多传的副本权重和显存可能超过收益；论文的 workload 是 P95-skew，不是均匀分布。最薄那档只比 balanced 高个位数百分比，副本权重传输很容易把这部分收益吃回去。
2. 副本预算 $s$ 是硬约束。$s$ 越大越均衡，但权重流量、显存、归约参与者一起涨；论文没给最优 $s$ 的公式，只让 planner 在预算内搜。
3. planner 自己也有成本。四个贪心各占一个 warp，在 device 上并行跑，占 SM 和寄存器；论文没单独报 planner 的延迟占比。
4. 它不改 router。复制只改「谁算」，不改「选哪些专家」，所以和 capacity factor、drop policy 正交。
5. 代码没公开，复现要自己写 planner 加 megakernel 集成；所有收益都在 8×B200 单 NVLink domain 上。

## 6. 一个管顺序，一个管放置

- UniEP 管顺序：token 按什么顺序进/出、SM 怎么切；MegaFlux 管放置：副本放哪、token block 给谁。
- 两者正交，可以叠：一个决定同一批 token 的先后，一个决定每块 token 去哪算。
- 叠加的接口在确定性：映射要从专家级扩到副本级，scoreboard 要能看见副本 partial，归约要固定累加顺序。
- 两个都是 on-device 调度，SM 预算会抢；短期现实的是分开用，不是硬融成一个 megakernel。

## 7. 只记一件事

复制专家这个点子不新，MoE 负载均衡里早有 expert replication。MegaFlux 真正做的是两件事：把复制做成 token block 粒度的放置规划，避免 partial tile；再把副本权重传输和梯度归约拆成 tile/投影级 readiness，塞进现有 megakernel。

routing 越 skew 越值；均匀路由下，它只是多传一份权重。下一篇把视角从「专家放哪」挪到「tile 什么时候能开工」：Entwine 与 mKernel（[补遗三]({{< relref "comm-overlap-15-tile-ecosystem" >}})）。

---

**系列导航**：[总览]({{< relref "comm-overlap-12-latest" >}}) · [补遗（一）：UniEP]({{< relref "comm-overlap-13-uniep" >}}) · [补遗（三）：Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
