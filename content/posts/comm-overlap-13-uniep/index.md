---
title: "通算并行研读（补遗一）：UniEP——把「顺序不定」换成「kernel 内保序」"
date: 2026-11-03
draft: false
weight: 13
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "UniEP", "Expert Parallelism", "MegaKernel", "确定性"]
categories: ["训练框架"]
---

**把「并行下顺序不定」换成「kernel 内保序」，overlap 之后就能逐位复现；代价是个位数百分比的性能，换来的加速比只有 1.03×–1.38×**。

读完你会知道：

- 哪三个算子在并行下顺序不定，UniEP 怎么把它们钉死；
- 两个 MegaKernel、四种角色，调度怎么从 host 挪进 SM；
- 1.03×–1.38× 里哪一端是 kernel 级、哪一端是端到端。

## 1. 生产框架不是不想上 overlap，是不敢

EP 的优化手段已经很多：通信压缩、计算通信 overlap、SM 分区、GroupGEMM 融合。但 Megatron-LM 这类框架的采用一直保守，原因不是「论文不够快」：

1. 不自适应：ad-hoc kernel 常把 $H_{dim}$、top-k、专家数写死，换个模型就崩；
2. 数值不稳：切 micro-batch、让 chunk 乱序到达来造窗口，结果和串行参考对不上；
3. 工程复杂：host 侧多 stream、event、chunk 叠在一起，调试和复现都难。

UniEP（[2604.19241](https://arxiv.org/abs/2604.19241)，ByteDance Seed + 清华大学，2026-04）要的是「Comet 级或更好的 overlap + 串行执行的数值语义」：不把 Comet 改得更快，而是把 MoE 的两个子图各融成一个 MegaKernel（整段常驻），把调度从 host 下推到 SM。

## 2. 换成两个 MegaKernel 和一张任务队列

MoE 的 forward 链是 `gate → dispatch → GroupGEMM(FC1) → SwiGLU → GroupGEMM(FC2) → combine`。UniEP 没把整层塞进一个 kernel，而是按依赖切成两段：

```text
MegaKernel 1: Dispatch + GroupGEMM
MegaKernel 2: GroupGEMM + Combine
```

好处是模块化：一段管「通信 → 计算」，一段管「计算 → 通信」，scoreboard 和角色集合能分开设计；代价是两段之间还有一次 kernel 边界，但比「每个 expert 一次 kernel」轻得多。

每个 MegaKernel 都是 persistent threadblock（常驻线程块）：block 数等于 SM 数，一块占一个 SM 跑完整段、不做 context switch；也不超过硬件容量，否则跨 block 同步死锁。

block 在运行时有四种角色：

| 角色 | 干什么 | 关键资源 |
|---|---|---|
| Comm-Worker | 跨 GPU 发 token | NVLink、warp ILP |
| Comp-Worker | 算 GroupGEMM tile | Tensor Core、HBM |
| Relay-Worker | 盯 token 到达、同 rank 内 multicast | HBM、scoreboard |
| Reduce-Worker | combine 阶段等齐 top-k 贡献再归约 | HBM、归约单元 |

角色不是 launch 时写死的：device 上一个全局 atomic counter 把任务空间线性化成 `[0, N_comm)` 通信、`[N_comm, N_comm + N_comp)` 计算，每个 SM 循环取号：

```python
task_id = atomic_add(counter, 1)
while task_id < TASK_COUNT:
    task_type, args = decode_task(task_id)
    if task_type == GROUPGEMM: compute_tile(args)
    elif task_type == DISPATCH: dispatch_tokens(args)
    elif task_type == RELAY:    relay_signal(args)
    task_id = atomic_add(counter, 1)
```

伪代码（论文 Listing 1）说了三件事：没有 host 侧角色分配，SM 取到什么干什么；没有 kernel relaunch，角色切换在同一 kernel 内完成；没有全局 barrier，队列本身就是负载均衡器。

时间线因此分三段：全在 dispatch → 部分转 Comp → 通信收尾、计算占满——论文 Figure 2 画的就是这个「overlapped region → mixed phase」。

「发 token」两套实现方向相反，读代码先确认哪一套：论文的 Triton 版用 `putmem_warp` push 到对端，仓库的 CuTeDSL 副本走 TMA，从对端 symmetric buffer 把行 pull 过来。

## 3. 三个算子顺序不定，UniEP 把顺序写进映射

顺序为什么会脏？combine 要把同一个 token 的 top-k 个专家输出加起来：

$$
y_t = \sum_{j \in \text{top-}k(t)} w_{t,j} \cdot E_j(x_t)
$$

浮点加法不满足结合律，累加顺序一变低位就变，反向的 weight gradient 更敏感。常见 overlap 方案为造窗口切 micro-batch、或让 chunk 乱序到达，「精度看起来没掉」能过，但做不到 bitwise identical。

UniEP 把约束提到前面：不管 overlap 怎么排，结果要和「不 overlap 的串行执行」逐位一致。判定口径就是这个——和它自己的串行参考逐位一致，不是「和任意框架一致」，也不是「loss 曲线好看」。

哪三个算子、怎么改：

| 算子 | 并行下会怎样 | UniEP 怎么改 | 代价 |
|---|---|---|---|
| top-k 加权求和 | 归约顺序随 token 到达顺序变，低位不同 | 等齐 top-k 贡献再归约，累加序固定 | 归约前必须等齐 |
| histogram（每专家计数 $C_{exp}$） | 现算靠原子竞争占位，谁先到谁先占 offset | 规划期算好，stable sort 保原序 | 多一次 AllGather |
| 累积和（跨 rank 偏移 $O_{all}$） | 前缀和是串行链，各 rank 独立算会不一致 | 一次 exclusive prefix sum 定死地址 | 前置阶段不能和通信重叠 |

Algorithm 1 五步：统计专家被选中次数 $C_{exp}$（`num_bins = N_exp`）→ cumsum 得本 rank 内偏移 $O_{exp}$ → 所有 rank 的 $C_{exp}$ AllGather 成 $C_{all}[r,e]$ → 对 rank 维做 prefix sum：

$$
O_{all}[r,e] = \sum_{k=0}^{r-1} C_{all}[k,e]
$$

最后和本地 stable sort 的索引相加，得到全局唯一偏移。于是每个 Comm-Worker 都能独立算目标地址，不用原子竞争决定谁先写；写序由预计算的映射定，不由时序定。

stable sort（相等元素保持原序）是 bitwise 的前提：同一 token 的多个专家副本、同一专家内的多个 token 都保原序。

进度板在 global memory，分 `S_token`（token 到没到）和 `S_tile`（一个 GroupGEMM tile，例如 128 个 token，是否 ready）。Comm-Worker 写完更新 `S_token`，Relay-Worker 数够 tile size 就原子置 `S_tile`，Comp-Worker 轮询 `S_tile`。它比整层 barrier 细（128 个 token 到齐就算），也比每 token 一个 event 省：event 的创建/销毁/轮询在 GPU 侧很贵，scoreboard 只是一段 global memory 加 acquire/release 语义。

代价不是零：bitwise 版本比非 bitwise 版本多数慢 2%–8%，MoE-10/11 是例外——论文只给结论、没给方向，别当漏写。它换来的不是更快，是能复现。

论文 Table 6 拿「无 overlap 的 PyTorch+NCCL 参考」做基准：

| 系统 | 最大绝对误差 | 非 bitwise 元素比例 |
|---|---:|---:|
| UniEP | 0 | 0% |
| Comet | 最高 0.25 | 22%–29% |

这不是 Comet 精度差，是它不保证和串行参考逐位一致：22%–29% 的元素低位不同，未必影响收敛，但排查「换配置 loss 对不上」时是实打实的成本。

## 4. 完美 overlap 能成立，因为顺序在发车之前就定了

窗口不靠切 micro-batch，靠 SM 角色切换、tile 级流水和 scoreboard 的细粒度 ready——代价是 kernel 得在整批 token 没到齐时开工。

映射对了，发送顺序仍然会拖慢 overlap。朴素实现按自然顺序发（Expert 0、Expert 1……），远端 Comp-Worker 却按专家升序处理 tile：Expert 0 晚到就停等，哪怕 Expert 1 已在 HBM 里——队头阻塞。

UniEP 让生产顺序和消费顺序对齐：Comp-Worker 按专家升序算，Comm-Worker 也按专家升序发。全局偏移是前缀和预计算的，按专家偏移数组迭代本身就成优先级，零额外排序开销。但它依赖前面的映射：地址预先算好，才能「按专家顺序发」而不破坏 token 顺序。

配置抽象成几个旋钮：

| 参数 | 含义 | 典型取值 |
|---|---|---|
| $w_{disp/comb}$ | 每个 worker 的 warp 数 | 8 / 16 / 32 |
| $N_{disp/comb}$ | 通信 worker 的 SM 数 | 搜索 |
| $N_{relay/red}$ | Relay/Reduce worker 的 SM 数 | 搜索 |

权衡很实际：Comp-Worker 8 warps 就够喂满 Tensor Core，开 32 warps 反而增加功耗密度、触发降频，FLOPS 更低；Comm-Worker 是延迟受限的，要 32 warps 提供更多 ILP（指令级并行）才压得满 NVLink。但 MegaKernel 要求统一 grid 配置，$w$ 必须折中——所以要模型，不能拍脑袋。

搜索空间约 $10^5$ 个配置，不走实测，而是三步：

1. 解析模型预测每个配置的 stage latency，枚举取最小；
2. Python 版遍历约 100 秒，不可接受；
3. 改 C++/OpenMP 并行降到约 144 ms，再用 4096-token 的 bucket 做 memoization，只有跨桶才查模型。

这套组合让 online autotuning 的摊销成本可忽略，也解释了它为什么能跨 12 个模型配置：不是每个模型手调，而是模型 + 搜索。

## 5. 1.03×–1.38× 里，kernel 级和端到端不是一回事

两个集群都是 8×Hopper：Cluster 1 的 NVLink 200 GB/s/dir（带宽受限），Cluster 2 是 400 GB/s/dir。12 个生产 MoE 配置，规模从 16B 的 DeepSeek-MoE 到 480B 的 Qwen3-Coder（含 30B/235B 两档），专家数 64–512，top-k 6–10，序列长度 8k/32k/128k。

表里全是 vs Comet 的加速比，跨行比之前先确认集群和序列长度。

| kernel 级 | Cluster 1 | Cluster 2 |
|---|---:|---:|
| Dispatch+GroupGEMM，8k | 1.32× | 1.57× |
| Dispatch+GroupGEMM，32k | 1.30× | 1.45× |
| GroupGEMM+Combine，8k | 1.31× | 1.21× |
| GroupGEMM+Combine，32k | 1.87× | 1.07× |

| layer 级 | Cluster 1 | Cluster 2 |
|---|---:|---:|
| Layer，8k，forward | 1.08× | — |
| Layer，8k，backward | 1.03× | — |
| Layer，32k，forward | 1.22× | 1.19× |
| Layer，32k，backward | — | 1.09× |
| 128k，forward | 1.28× | 1.33× |

摘要给的总体口径是 1.03×–1.38×，落到格子上差得很远：

- 1.03× 是 layer backward 的下限；
- kernel 级多数落在 1.3×–1.6×，两端是 1.07×（Cluster 2 / 32k / Combine）和 1.87×（Cluster 1 / 32k / Combine）；
- 到 layer/e2e 会被 attention、gating、非 MoE 计算稀释，别当「平均 1.2×」，更别拿 kernel 级的 1.57× 对标 e2e。

生产训练是另一层稀释：128 GPU（16 节点 × 8）、512k 序列，吞吐 127 → 138 B tokens/day，1.09×，且 bitwise 可复现。

它比 kernel 级的 1.5× 小得多，但更接近真实训练：MoE 层只占 step 的一部分，Amdahl 定律（串行部分限制总加速）会压下来。论文说「10% 的 MFU 提升在千卡集群上等于数百万美元」——账成立，前提是先接受这层层稀释。

这些数字不能外推：两个集群都是单节点 8 卡 Hopper，跨节点、NVL72、PCIe 都要重测。

## 6. 这些前提下它才划算，破一条就别抄

- 硬件：只在 Hopper、单节点 8 卡上验证；生产训练跑到 128 GPU/16 节点，但论文没展开跨节点 kernel。
- 复现门槛：MegaKernel 约 21k 行 Python（Triton-distributed），论文估计写 raw CUDA 会超过 10 万行、不可维护；论文称已开源在 [ByteDance-Seed/Triton-distributed](https://github.com/ByteDance-Seed/Triton-distributed)，但我查 `main` 没看到 `uniep` 目录，可能还在分支/PR。
- baseline 不齐：Comet 只支持特定 $H_{dim}$ 和 top-k。MoE-2（$H_{dim}=1280$，top-k 6）的 top-k 不支持，MoE-12（$H_{dim}=2304$，top-k 8）也跑不了——注意 Table 4 里它是 8 不是 6。所以部分对比是「UniEP vs Serial」而非 vs Comet，读表前先确认。
- 确定性有边界：它只保证和自家的无 overlap 串行参考逐位一致。baseline 的 stable sort、padding、top-k 顺序一变，bitwise 的定义就变，跨框架复现先对齐参考实现。

## 7. 前文讲的是粒度，它加的是顺序

- 对 [第六篇]({{< relref "comm-overlap-06-moe" >}})：Comet 在 launch 定死角色，UniEP 让角色在 kernel 内流动；两者都能 overlap，后者多一条确定性约束。
- 对 [第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})：分区从「层粒度静态」变成「任务粒度动态」，账还是 $T = C + E + B$ 那笔，只是多了一条顺序约束。
- 对「切错融让」：它做融、做让、做错，唯独不切 micro-batch——切了就没有 bitwise。

## 8. 如果只记一件事

**确定性不是精度指标，是一个可以工程化的执行顺序问题**：把顺序从执行时序挪到预计算的映射，bitwise 就从祈祷变成可验证的约束。

代价那一半别丢：1.03×–1.38× 是相对 Comet 的收益，保序本身的开销另算，两个数必须一起看。

想看 routing skew 和跨节点的下一步，读 [MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) 和 [Entwine/mKernel]({{< relref "comm-overlap-15-tile-ecosystem" >}})。

---

**系列导航**：[总览]({{< relref "comm-overlap-12-latest" >}}) · [补遗（二）：MegaFlux]({{< relref "comm-overlap-14-megaflux" >}}) · [专题首页]({{< relref "comm-overlap-00-index" >}})
