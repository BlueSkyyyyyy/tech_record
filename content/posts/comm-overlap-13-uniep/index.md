---
title: "通算并行研读（补遗一）：UniEP 深读——persistent worker、确定性 token mapping 与 1.03×–1.38×"
date: 2026-11-03
draft: false
weight: 13
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "UniEP", "Expert Parallelism", "MegaKernel", "确定性"]
categories: ["训练框架"]
---

UniEP（[2604.19241](https://arxiv.org/abs/2604.19241)，ByteDance Seed + 清华大学，2026-04）是我认为这批补遗里最该单独读的一篇。原因不是它比 Comet 快多少，而是它把一个问题摆到了台面上：

> **overlap 改了执行顺序，数值还能不能和串行执行逐位一致？**

在生产训练里，这个问题比“再快 5%”更硬。Comet 的 shared-tensor 切分、Weave 的 chunk pipeline、DualPipe 的双向调度，都会改变 token 的到达顺序；MoE 的 top-k 归约和 weight gradient 累加又都不满足结合律。大多数论文用“精度没掉”带过，UniEP 直接把它做成约束，并用 12 个配置证明可以做到误差为 0。

这一篇按“问题 → 结构 → 确定性 → 调度/调参 → 数字 → 边界”拆开。

## 1. 问题：为什么生产框架不敢直接上 overlap

这一节先解释：EP 的优化已经很多，生产框架为什么还是不放心。

UniEP 的引言给了一个很准确的观察：EP 的优化手段已经很多——通信压缩、计算通信 overlap、SM 分区、GroupGEMM 融合——但 Megatron-LM 这类生产框架的采用仍然保守。原因不是“论文不够快”，而是：

1. **不自适应**：ad-hoc kernel 往往为某个 $H_{dim}$、top-k、专家数写死，换一个模型就崩；
2. **数值不稳定**：为了制造并行窗口切 micro-batch 或乱序到达，结果和串行参考对不上；
3. **工程复杂**：host 侧多 stream、event、chunk 调度叠在一起，调试和复现都困难。

UniEP 的目标是“既要有 Comet 级/更好的 overlap，又要保持串行执行的数值语义”。它选择的路不是“把 Comet 改得更快”，而是**把 MoE 的两个关键子图各自融成一个 MegaKernel（整段常驻 kernel），把调度从 host 下推到 SM**。

## 2. 结构：两个 MegaKernel，四个角色

这一节拆结构：为什么是两个 MegaKernel，四类角色怎么分工。

### 2.1 为什么是两个，不是一个大 kernel

MoE 的 forward 可以粗分成：

```text
gate → dispatch → GroupGEMM(FC1) → SwiGLU → GroupGEMM(FC2) → combine
```

UniEP 没有把整层塞进一个 kernel，而是按依赖切成两段：

```text
MegaKernel 1: Dispatch + GroupGEMM
MegaKernel 2: GroupGEMM + Combine
```

这样做的好处是模块化：每一段只需要处理“通信 → 计算”或“计算 → 通信”的依赖，scoreboard 和角色集合可以分别设计；代价是两段之间仍然有一次 kernel 边界，但这比“每个 expert 一次 kernel”或“每个 chunk 一次 launch”要轻得多。

### 2.2 persistent worker：一个 block 占一个 SM

每个 MegaKernel 都是 **persistent threadblock（常驻线程块）**：

- launch 的 block 数等于物理 SM 数；
- 一个 block 占住一个 SM 跑完整段，避免 context switch；
- block 数不超过硬件容量，避免跨 block 同步死锁。

block 在运行时有四种角色：

| 角色 | 做什么 | 关键资源 |
|---|---|---|
| **Comm-Worker** | 跨 GPU 发 token：论文 Triton 版 push，仓库 CuTeDSL 版 TMA pull | NVLink、warp ILP |
| **Comp-Worker** | 算 GroupGEMM tile | Tensor Core、HBM |
| **Relay-Worker** | 盯 token 到达、同 rank 内 multicast | HBM、scoreboard |
| **Reduce-Worker** | combine 阶段等齐 top-k 贡献再归约 | HBM、归约单元 |

「发 token」在两套实现里方向相反：论文的 Triton 实现用 `putmem_warp` 把 token push 到对端；仓库里的 CuTeDSL 副本走 TMA，从对端的 symmetric buffer 把需要的行 pull 过来。读代码前先确认自己在看哪一套。

角色不是 launch 时写死的。UniEP 在 device 上放一个全局 atomic counter，把任务空间线性化成：

```text
[0, N_comm)                  → 通信任务
[N_comm, N_comm + N_comp)    → 计算任务
```

每个 SM 循环执行：

```python
task_id = atomic_add(counter, 1)
while task_id < TASK_COUNT:
    task_type, args = decode_task(task_id)
    if task_type == GROUPGEMM: compute_tile(args)
    elif task_type == DISPATCH: dispatch_tokens(args)
    elif task_type == RELAY:    relay_signal(args)
    task_id = atomic_add(counter, 1)
```

这段伪代码来自论文 Listing 1，信息量很大：

- **没有 host 侧 role 分配**：SM 取到什么任务就干什么；
- **没有 kernel relaunch**：角色切换在同一个 kernel 内完成；
- **没有全局 barrier**：任务队列本身是负载均衡器，先通信、再混合、最后计算。

时间线因此自然呈现三段：一开始所有 SM 都在 dispatch；数据到达后，一部分 SM 转成 Comp；最后通信收尾、计算占满。UniEP 论文 Figure 2 画的就是这个“overlapped region → mixed phase”的过程。

### 2.3 和 Comet/Weave 的角色模型对比

| 系统 | 角色怎么定 | 依赖怎么排 | 并行载体 |
|---|---|---|---|
| **Comet** | CTA specialization，producer/consumer 在 launch 时定死 | 双 stream + event | stream 级 |
| **Weave** | 每层用 cost model 选 $c$、$K$，角色按分区静态落位 | chunk pipeline + 信号 | 层/chunk 级 |
| **UniEP** | 全局任务队列，SM 运行时取任务 | scoreboard（token/tile） | kernel 内任务级 |

UniEP 把“让”从“层粒度/launch 粒度”推进到了“任务粒度”。代价是调度器复杂：任务队列、scoreboard、角色切换都要在 kernel 内完成；收益是没有 host 侧多 stream 的 alignment bubble，也没有 micro-batch 切分带来的顺序变化。

## 3. 确定性：UniEP 的核心贡献

这一节讲核心贡献：确定性从哪来，为什么它不是免费的。

### 3.1 为什么 token 顺序会影响数值

MoE 的 combine 要把同一个 token 的 top-k 个专家输出加起来：

$$
y_t = \sum_{j \in \text{top-}k(t)} w_{t,j} \cdot E_j(x_t)
$$

浮点加法不满足结合律，不同的累加顺序会给出不同的低位。反向更敏感：weight gradient 的累加顺序也会变。常见 overlap 方案为了制造并行窗口，会把 token 切成 micro-batch，或者让不同 chunk 乱序到达；这在“精度看起来没掉”的层面往往能过，但做不到 **bitwise identical**。

UniEP 的约束是：不管 overlap 怎么排，最终结果要和“不 overlap 的串行执行”逐位一致。

### 3.2 确定性全局 token 映射

UniEP 为每个本地 token 预计算一个三元组：

```text
(target_rank, expert_id, destination_offset)
```

目标地址不是运行时抢出来的，而是由全局前缀和算出来的。论文 Algorithm 1 的核心步骤：

1. 统计每个专家被选中的次数 $C_{exp}$（bin count，`num_bins = N_exp`）；
2. 对 $C_{exp}$ 做 cumsum 得到每个专家在本 rank 内的偏移 $O_{exp}$；
3. 对所有 rank 的 $C_{exp}$ 做 AllGather，得到 $C_{all}[r,e]$；
4. 对 rank 维做 exclusive prefix sum：

$$
O_{all}[r,e] = \sum_{k=0}^{r-1} C_{all}[k,e]
$$

5. 把 $O_{all}[r,e]$ 和本地 stable sort 的索引相加，得到每个 token 的全局唯一偏移。

这样每个 Comm-Worker 都能独立算出目标地址，不需要原子竞争决定“谁先写”。最后写入顺序由**预计算的映射**决定，而不是由执行时序决定。

论文特别强调 **stable sort（相等元素保持原序）**：同一个 token 的多个专家副本、同一个专家内的多个 token，都保持原始顺序。这是 bitwise 的前提。

### 3.3 scoreboard：token 级 + tile 级

UniEP 在 global memory 放一个结构化 scoreboard（显存里的进度板）：

```text
Scoreboard = [ S_token | S_tile ]
```

- `S_token`：每个 token 是否到达；
- `S_tile`：一个完整 GroupGEMM tile（例如 128 个 token）是否 ready。

不同角色的用法：

- **Comm-Worker** 写完一个 token，更新 `S_token`；
- **Relay-Worker** 轮询 `S_token`，计数到 tile size 后原子设置 `S_tile`；
- **Comp-Worker** 轮询 `S_tile`，ready 就执行对应 tile 的 GroupGEMM。

这比“整层 barrier”细得多：一个 tile 的 128 个 token 到齐就能算，不用等整批 token。它也比“每个 token 一个 event”省：event 的创建/销毁/轮询在 GPU 侧很贵，scoreboard 只是一段 global memory + acquire/release 语义。

### 3.4 不切 micro-batch

UniEP 明确把 **avoiding micro-batch splitting** 作为保持顺序的手段。并行窗口靠：

- SM 角色动态切换；
- tile 级流水；
- scoreboard 的细粒度 ready；

而不是靠把 batch 切开。这个选择很关键：切 micro-batch 会让每个 chunk 的 token 顺序、梯度累加顺序都变；不切就没有这个问题，但要求 kernel 能在“整批 token 还没到齐”时就开始算——这正是 scoreboard 的作用。

### 3.5 数值证据：Comet 的 22%–29% 非 bitwise

论文 Table 6 对比了“无 overlap 的 PyTorch+NCCL 参考”：

| 系统 | 最大绝对误差 | 非 bitwise 元素比例 |
|---|---:|---:|
| **UniEP** | **0** | **0%** |
| **Comet** | 最高 **0.25** | **22%–29%** |

这不是“Comet 精度差”，而是“Comet 不保证和串行参考逐位一致”。对大多数训练任务，22%–29% 的元素低位不同未必影响收敛；但对“换 overlap 配置后 loss 曲线对不上”的排查，这是实打实的成本。UniEP 的价值是把这类不确定性消掉。

## 4. 调度与自动调参：让“完美 overlap”假设成立

### 4.1 优先级 token 调度：解决队头阻塞

即使映射正确，发送顺序也会影响 overlap。朴素实现按自然顺序发（Expert 0、Expert 1……），但远端 Comp-Worker 是按专家升序处理 tile 的。如果 Expert 0 的数据晚到，Comp-Worker 就会停等，哪怕 Expert 1 的数据已经在 HBM 里——这就是队头阻塞（head-of-line blocking）。

UniEP 的做法是**让生产顺序和消费顺序对齐**：

- Comp-Worker 按专家升序处理；
- Comm-Worker 也按专家升序发送；
- 由于全局偏移是前缀和预计算的，按专家偏移数组迭代本身就实现了优先级，**零额外排序开销**。

这条优化看起来简单，但它依赖前面的确定性映射：只有目标地址预先算好，才能“按专家顺序发”而不破坏 token 顺序。

### 4.2 四个可调原语

UniEP 把每个阶段的配置抽象成：

| 参数 | 含义 | 典型取值 |
|---|---|---|
| $w_{disp/comb}$ | 每个 worker 的 warp 数 | 8 / 16 / 32 |
| $N_{disp/comb}$ | 通信 worker 的 SM 数 | 搜索 |
| $N_{relay/red}$ | Relay/Reduce worker 的 SM 数 | 搜索 |

warp 数的权衡很实际：

- Comp-Worker 8 warps 就够喂满 Tensor Core；开 32 warps 会增加功耗密度，触发降频，反而降低 FLOPS；
- Comm-Worker 是延迟受限的，32 warps 提供更多 ILP（指令级并行），才能压满 NVLink。

因为 MegaKernel 要求统一 grid 配置，$w$ 必须折中。这就是“为什么需要模型，而不是拍脑袋”。

### 4.3 解析模型 + 10^5 搜索 + 4096 桶

UniEP 的搜索空间约 $10^5$ 个配置。它没有用实测搜索，而是：

1. 用解析性能模型预测每个配置的 stage latency；
2. 枚举所有配置，选最小；
3. Python 版遍历约 **100 秒**，不可接受；
4. 改成 C++/OpenMP 并行，降到 **~144 ms**；
5. 用 **4096-token bucket** 做 memoization（缓存查过的结果），只有跨桶才重新查模型。

这套组合让 online autotuning 的摊销成本可以忽略。它也解释了为什么 UniEP 能跨 12 个模型配置：不是每个模型手调，而是模型 + 搜索。

## 5. 数字：1.03×–1.38× 怎么读

这一节读数字：1.03×–1.38× 是哪个口径下的，别拿 kernel 级数字去对标 e2e。

### 5.1 实验设置

- Cluster 1：8×Hopper，NVLink **200 GB/s/dir**，带宽受限；
- Cluster 2：8×Hopper，NVLink **400 GB/s/dir**，带宽更充裕；
- 12 个生产 MoE 配置：DeepSeek-MoE-16B、DeepSeek-OCR-2、DeepSeek-V2-Lite/Chat、DeepSeek-R1、Qwen3-30B-A3B、Qwen3-235B-A22B、Qwen3-Coder-480B、Qwen3-Next-80B、Qwen3-Omni-30B、Kimi-K2、Kimi-Linear-48B；
- 专家数 64–512，top-k 6–10，序列长度 8k/32k/128k。

### 5.2 分场景数字

读法：先看“对比”列，全部是 vs Comet 的加速比；跨行比之前先看集群和序列长度。

| 场景 | 对比 | 结果 |
|---|---|---|
| Dispatch+GroupGEMM，Cluster 1，8k | vs Comet | **1.32×** |
| Dispatch+GroupGEMM，Cluster 1，32k | vs Comet | **1.30×** |
| Dispatch+GroupGEMM，Cluster 2，8k | vs Comet | **1.57×** |
| Dispatch+GroupGEMM，Cluster 2，32k | vs Comet | **1.45×** |
| GroupGEMM+Combine，Cluster 1，8k | vs Comet | **1.31×** |
| GroupGEMM+Combine，Cluster 1，32k | vs Comet | **1.87×** |
| GroupGEMM+Combine，Cluster 2，8k | vs Comet | **1.21×** |
| GroupGEMM+Combine，Cluster 2，32k | vs Comet | **1.07×** |
| Layer，Cluster 1，8k，forward | vs Comet | **1.08×** |
| Layer，Cluster 1，8k，backward | vs Comet | **1.03×** |
| Layer，Cluster 1，32k，forward | vs Comet | **1.22×** |
| Layer，Cluster 2，32k，forward | vs Comet | **1.19×** |
| Layer，Cluster 2，32k，backward | vs Comet | **1.09×** |
| 128k，Cluster 1，forward | vs Comet | **1.28×** |
| 128k，Cluster 2，forward | vs Comet | **1.33×** |
| 生产训练，128 GPU，512k | 吞吐 | **127 → 138 B tokens/day（1.09×）** |

论文摘要的总体口径是 **1.03×–1.38× over state-of-the-art**。从表里能看到：

- **1.03× 是 layer backward 的下限**；
- kernel 级多数落在 1.3×–1.6×，两端是 1.07×（Cluster 2 / 32k / Combine）和 1.87×（Cluster 1 / 32k / Combine）；
- 到 layer/e2e 会被 attention、gating、非 MoE 计算稀释；
- 所以不要把它当成“平均 1.2×”，更不要拿 kernel 级的 1.57× 去对标 e2e。

### 5.3 生产训练的意义

128 GPU（16 节点 × 8）、512k 序列，吞吐 127 → 138 B tokens/day，**1.09×**，且 bitwise 可复现。这个数字比 kernel 级的 1.5× 小很多，但它更接近真实训练：MoE 层只占整个 step 的一部分，Amdahl 定律（串行部分限制总加速）会把收益压下来。论文的论点是“10% 的 MFU 提升在千卡集群上等于数百万美元”，这个账是成立的。

## 6. 边界与代价

这一节列边界：哪些场景这套东西会失效，或者要重做。

### 6.1 实现成本

论文说 MegaKernel 约 **21k 行 Python**（Triton-distributed），并估计如果写 raw CUDA 会超过 10 万行、不可维护。这不是“改个 kernel”能复现的工作量。

### 6.2 代码状态

论文称实现已开源在 [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)。我检查该仓库 `main` 分支时没有看到独立的 `uniep` 目录，可能还在分支/PR 或尚未完全合入；引用代码时以论文和仓库实际状态为准。

### 6.3 硬件与规模

- 只在 **Hopper** 上验证；
- 实验是**单节点 8 卡**；生产训练是 128 GPU/16 节点，但论文没有展开跨节点 kernel 的细节；
- 跨节点、NVL72 scale-up、PCIe 等场景需要重新验证。

### 6.4 baseline 限制

Comet 只支持特定 $H_{dim}$ 和 top-k。MoE-2（$H_{dim}=1280$，top-k=6）是 top-k=6 不在支持范围；MoE-12（$H_{dim}=2304$，top-k=8）也跑不了——注意 Table 4 里它的 top-k 是 8，不是 6。所以部分对比是“UniEP vs Serial”，不是“UniEP vs Comet”。读数字时要注意这一点。

### 6.5 确定性的适用边界

UniEP 保证的是“和它自己的无 overlap 串行参考逐位一致”。这不等于“和任意框架的参考一致”：如果 baseline 的 stable sort、padding、top-k 顺序不同，bitwise 的定义也会变。跨框架复现时，要先对齐参考实现。

## 7. 和系列前文的关系

- **对第六篇（MoE）**：Comet 是“CTA specialization + 双 stream”，UniEP 是“persistent worker + 单 stream 内任务级调度”。前者把角色定死，后者让角色流动；两者都能 overlap，但后者多了一个“确定性”约束。
- **对第七篇（SM 分区）**：Weave 是“每层重算 $c$、$K$”，UniEP 是“SM 取任务时动态决定角色”。分区从“层粒度静态”变成“任务粒度动态”。
- **对第一篇（账本）**：UniEP 的 1.03×–1.38× 正好说明 $T = C + E + B$ 里的 $E$ 不可能全部消掉；kernel 级 overlap 效率高，到了 layer/e2e 仍会被非 MoE 部分稀释。
- **对“切/错/融/让”**：UniEP 主要是“融”（两个 MegaKernel）+“让”（SM 角色动态让渡）+“错”（按专家优先级错峰发送）；它刻意**不切** micro-batch，以换确定性。

## 8. 结论

UniEP 最值得记的不是某个百分比，而是它把“可复现”变成了 overlap 的一等约束，并把代价压到个位数百分比：bitwise 版本比非 bitwise 版本多数慢 2%–8%（MoE-10/11 例外），相对 Comet 的端到端加速仍在 1.03×–1.38× 之内。

- **结构**：两个 MegaKernel，persistent worker，四类角色，任务队列动态调度；
- **语义**：确定性 token mapping + stable sort + 不切 micro-batch，做到 bitwise identical；
- **调度**：scoreboard 细粒度 ready + 按专家优先级发送，解决队头阻塞；
- **调参**：解析模型 + $10^5$ 搜索 + 4096 桶 memoization，144 ms 完成；
- **代价**：21k 行 Python、Hopper 单节点验证、Comet 部分配置不支持、代码尚未在 main 明确可见。

如果你的集群是 Hopper + NVLink、模型是 MoE、且训练需要严格复现，UniEP 是这批论文里最值得先读的；如果你更关心 routing skew 或跨节点，继续读 [MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) 和 [Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}})。

---

**系列导航**：[总览]({{< relref "comm-overlap-12-latest" >}}) · [补遗（二）：MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
