---
title: "通算并行研读（六）：MoE 的通信比计算长，只能藏住一段"
date: 2026-11-02
draft: false
weight: 7
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "专家并行", "DeepEP", "Comet", "all-to-all"]
categories: ["训练框架"]
---

**把「藏住一整段 all-to-all」换成「只藏计算那么长的一段」，代价是剩下的启动和收尾只能靠切得更细来摊薄，而切得越细 GEMM 越碎**。

读完你会知道：

- 一层 MoE 的通信为什么经常比 expert 计算还长；
- 切成 chunk 之后，哪三笔时间一定藏不掉；
- 六个方案各在哪一层动手，以及你的场景该先问什么。

## 1. 先把两个数对上

Comet（[2502.19811](https://arxiv.org/abs/2502.19811)）在 8×H800 上把一层 MoE 拆开看，通信占了 **47%** 的时间；Lancet（[2404.19429](https://arxiv.org/abs/2404.19429)）在另一套集群上测到，all-to-all 最高是 expert compute 的 **3.36 倍**。

这两个数不能直接比：一个的分母是整层时间，一个的分母是计算时间，节点数和 EP/TP 配置也不一样。它们只说明同一件事——**通信不是边角料，它经常比计算长**。

而"把通信藏进计算"的上限就是计算那一头：通信比计算长时，最多藏住计算那么长的一段，多出来的必然暴露。

通信为什么偏偏这么长，三个原因：

- top-k 复制：$k=2$ 就是两倍流量，$k=8$ 就是八倍（DeepSeek-V3 这类模型每 token 激活 8 个专家）；
- routing skew：token 往热点专家挤，还要按 capacity 补 padding，而 padding 也占 GEMM 的行；
- combine 卡在层的末端：dispatch 没到，expert 一个 token 都算不了；combine 没回，下一层 attention 就没有输入。它不像 DP 的梯度能挪到 optimizer 前，也不像 TP 有相邻 GEMM 可以错位。

反过来说，**expert GEMM 很短**：单卡上每个专家分到的 token 不多，grouped GEMM 的 shape 又碎，算力利用率本来就不高；all-to-all 却要跨节点走 IB，带宽比 NVLink 低一个数量级。3.36 倍就是这么来的——通信比计算长，只藏计算等于没藏。

## 2. 三段式：dispatch → compute → combine

一层 MoE 是三个算子串起来的：

- **D**（dispatch）：按 routing 把 token 发到专家所在的 rank，一次 all-to-all；
- **C**（expert compute）：本地专家跑 grouped GEMM（gate/up → SiLU → down），按专家分组；
- **B**（combine）：把专家输出送回 token 原 rank，做 top-k 加权归约。这里的 $B$ 指 combine，和[流水线篇]({{< relref "comm-overlap-02-pipeline" >}})里那个 $B$（dX 反向）**不是一回事**。

D 的输出一般按本地专家重排成 `[num_local_experts, capacity, H]`，capacity 取该卡最热专家的 token 数（把每个专家的 token 补齐到最热的那个），或按 grouped GEMM 的 M 维对齐。所以 padding 的 token 也占 GEMM 的行：routing 越偏，padding 越多，算得越慢。

DeepEP 用 `expert_alignment` 对齐这个 M 维；Comet 干脆不按固定 capacity 切，按实际收到的 token 重排 tile。

依赖是硬链：$D_i \to C_i \to B_i$。不分块时一层的时间就是三段直接相加：

$$
T = d + c + b
$$

所以每段通信都压在关键路径上，没有任何藏身处。

dispatch 和 combine 也合不成一次 all-to-all：中间隔着整个 expert 计算，数据依赖不允许。能合的只有 combine 和下一段 P2P——DisagMoE 的 M2N 就是这么把两次搬运压成一次的。

分块之后 chunk 之间互相独立，才有流水。图中 D/B 各占 1 个时间单位、C 占 4 个：

```text
comm  [D0][D1][D2][D3]    [B0]            [B1]            [B2]            [B3]
comp      [======C0======][======C1======][======C2======][======C3======]
      ↑ startup=d0                                                        ↑ tail=b3
```

两条通道各只有一条：$C_1$ 得等 $C_0$ 算完，$B_i$ 又得等 $C_i$。所以能重叠的只有"通信 vs 计算"，不是"计算 vs 计算"；两次 combine 之间，通信通道还是空着的。

**分块省掉的是中间那些 $D_i$ 和 $B_i$，省不掉头尾**。

## 3. 切成块，只剩三笔藏不掉

通信能全藏住时，计算是长边：

$$
T = \sum_i c_i + d_0 + b_{\text{last}}
$$

通信成瓶颈时，长边换成通信，计算只在头部露一块：

$$
T = \sum_i d_i + \sum_i b_i + c_0
$$

所以长边是谁，决定了藏住谁：理想情形藏住中间的 $D_i$ 和 $B_i$，只露 $d_0$ 和末块的 $b_{\text{last}}$；通信瓶颈时藏住中间的 $C_i$，只露第一块 $c_0$。不管哪一边，都剩下三笔：

1. **startup**：第一个 chunk 的 $d_0$ 没回来，expert 一个 token 都算不了。这就是[第一篇]({{< relref "comm-overlap-01-performance-model" >}})说的那笔藏不住的启动开销：只能靠更小的 chunk 摊薄，不能靠重叠消除。
2. **tail**：最后一个 chunk 的 $b_{\text{last}}$ 永远在关键路径上。CommFuse 的做法是把 collective 拆成 ring P2P、再重排顺序把它消掉，数字和长序列上的反例都在[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})。
3. **mid bubble**：$B_i$ 等 $C_i$，通信通道在两次 combine 之间空一段。$K=1$ 时这段就是整个 GEMM；切成 $K$ 块后变成每个 chunk 边界一小段，总长取决于 $c$ 和 $b$ 的比例，**降不到 0**。

所以 chunk 不是越细越好：切得越细，头尾越短，但每块 GEMM 的行数变少、grouped GEMM 效率下降、launch 和同步次数上升。Weave 实测 $K=8$ 时 GEMM 吞吐比 $K=1$ 低约 **27%**，它选出来的最优 $K$ 通常只有 2–4。

**分块只能摊薄启动和尾部，不能把"通信比计算慢"这件事变没**——和 TP 的结论一样。

## 4. DeepEP 只做一件事：把 wait 往后推

DeepEP 是现在 MoE 训练/推理最常用的 EP 通信库，本地仓库 `~/github/DeepEP`（commit `b5a33eef`），接口在 `deep_ep/buffers/ep.py`：dispatch 在 `:562`，combine 在 `:770`。

`dispatch` 把本卡 token 发出去，返回接收缓冲、routing 信息和 `handle`；`combine` 拿着同一个 `handle` 把专家输出送回来。`handle` 记着这次 dispatch 的布局和 SM 数，所以 combine 传 `num_sms=0` 就是"复用 dispatch 的选择"。

它真正给的不是"异步 API"，而是**把 wait 推到最后一个可能的时刻**。五个参数各管一段：

| 参数 | 把 wait 推到哪 | 什么时候用 |
|---|---|---|
| `previous_event` | 执行前先等这个 event | dispatch 等 gate/routing，或等上一块 buffer 释放 |
| `async_with_compute_stream` | 当前 stream 不等通信 kernel 完成 | 双 stream 重叠的关键 |
| `allocate_on_comm_stream` | 输出 tensor 的归属记在通信 stream 上 | 配合 `previous_event`，避免跨 stream 隐式同步 |
| `defer_epilogue` | CPU 侧的接收计数等待推迟到 wait 时 | 想让 launch 尽早返回 |
| `num_sms` | 通信 kernel 占几个 SM（0 表示按流量自动估） | 手动调 SM 预算 |

最自然的写法是先发射、真要用了再等：

```python
recv_x, recv_idx, recv_w, handle, ev = buf.dispatch(
    x, topk_idx, topk_weights, num_sms=comm_sms, async_with_compute_stream=True)

do_independent_work()   # shared expert GEMM、上一层残余、下一个 microbatch 的 attention

ev.current_stream_wait()   # 只有 expert GEMM 真要读 recv_x 时才等
y = grouped_gemm(recv_x, recv_idx, ...)

ev2 = buf.combine(y, handle, topk_weights,
                  async_with_compute_stream=True, defer_epilogue=True)
ev2.current_stream_wait()
```

`EventOverlap`（`deep_ep/utils/event.py`）是对 CUDA event 的薄封装，支持 `with event_overlap:`——进入时什么都不做，退出时才让当前 stream 等这个 event。于是"发通信"和"用结果"能拆成两段代码，中间插任意独立计算。`defer_epilogue` 在 CUDA graph 或多 microbatch 场景尤其重要：host 线程不会被计数等待阻塞。

### 4.1 num_sms 是从 GEMM 嘴里抢 SM

all-to-all 不是"网卡自己搬"，它要在 GPU 上跑 copy/notify kernel，占 SM 来发指令、轮询、做 reduction。README 写得很直白：

> EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported.

这一版**没有零 SM 选项**。`num_sms` 就是你从 GEMM 手里抢出来给通信的 SM 数，它是个纯权衡：

- 通信带宽随 SM 数增加，但会饱和——SM 够多之后瓶颈回到 NIC/NVLink，再加没有收益；
- GEMM 算力随可用 SM 数线性下降，`total_sms - num_sms` 就是计算能用的部分；
- 最优点在"通信边际带宽增益 = GEMM 边际算力损失"的位置。

`get_theoretical_num_sms`（`ep.py:431`）给的是解析估计：算出来的值乘 1.3 倍、对齐到 4 的倍数、至少 4；不开 `prefer_overlap_with_compute` 时下限抬到 64；最后封顶在设备 SM 数的一半。它假设 gate 负载均衡，真实 skew 大时要手动加。

`num_qps` 是另一个旋钮：直连模式自动取 `min(num_sms, 9)`，混合模式取 `num_sms * 16 + 1`。QP 少了打不满 RDMA，多了会有 doorbell 和缓存开销——但那是跨节点才需要关心的参数，单节点 NVLink 上影响很小。

Weave 测过：单层最优的通信 SM 数在不同 GPU 上从 **12 到 36** 不等。在它自己的配置下，DeepEP 取 20，Comet 取 18。这两个数都是**通信 CTA** 的预算，和 Comet 的 $n_c$ 同一口径——20 是 DeepEP 自己按解析估计算出来的，18 是 Comet 按 iteration 从预编译变体里挑的，所以两者都会错过某些卡的最优值。

## 5. 五个层级，同一件事

DeepEP 只是积木：Comet 在层内搭、Weave 在整层搭、Lancet 在整图搭、DisagMoE 在模块之间搭、INC-MoE 把积木搬进网络。越往下越通用，越往上收益越大、依赖越强。

### Comet：让上游按 tile 的节奏到货

Comet 解决的是粒度错位：通信以 token 为单位，GEMM 以 tile 为单位，一个 128×128 tile 可能要来自多张卡的 128 个 token，缺一个就整块等。

它把生产者/消费者之间的缓冲区叫 shared tensor，按 consumer 的访问模式决定沿哪一维切：dispatch → GEMM1 沿 token 维切（token 之间独立），并按来源 rank 排序让本地 token 先算；GEMM2 → combine 要做 top-k 归约、$M$ 维不能切，就沿 hidden 维切，GroupGEMM 改成按列执行，前 $T_N$ 列算完 combine 就能开始。

切完把两段各做成一个融合 kernel（dispatch+GEMM1、GEMM2+combine），用 CTA 特化隔离通信和计算，每个 SM 一个 CTA、总数 132；compute CTA 内部还是 CUTLASS 那套分工，producer warp 用 TMA 搬数据、consumer warp 发 MMA。

kernel 内再按当前 shape 自适应分 $n_c$ 个通信 CTA 和 $n_p$ 个计算 CTA——TP=8、$M$ 从 4096 涨到 16384 时 $n_c$ 从 18 变到 26，TP 降到 4 时又从 26 涨到 46。

数字（EP=8、TP=1）：单层 **1.96×**、端到端 **1.71×**（Mixtral-8x7B、Qwen2-MoE、Phi3.5-MoE 三个模型平均），藏住 **86.5%** 的通信（FasterMoE 29.2%、Tutel 68.6%）；随输入长度变化，单层 1.28×–2.37×。

边界：每卡要一块约 $2MN$ 字节的 NVSHMEM buffer（$M=16384$、$N=4096$ 时是 128 MB 量级）；通信和计算不能塞进同一个 block（warp/thread 上限会让通信打不满带宽，还会干扰计算 warp）；GEMM 侧绑 Hopper 的 CUTLASS；$n_c/n_p$ 每个 iteration 重选，但层与层之间不重分。

### Weave：整层一个 kernel，动态分 SM

Comet 留了两个洞：空间上 SM 划分按 iteration 调、层间卡间不一致；时间上 GEMM 中段还有 mid bubble，通信 SM 整段空着。

Weave（[2609.21483](https://arxiv.org/abs/2609.21483)）把 dispatch、GEMM1、SiLU、GEMM2、combine 全融进一个 persistent megakernel，grid 就是 SM 数。routing 一结束，这层的通信量和计算量就完全确定，于是 prologue 跑一个轻量 cost model，联合选通信 SM 数 $c$、chunk 数 $K$、以及通信 SM 在空闲窗口里偷多少 GEMM tile。

时间调度做两件事：dispatch 不切、其余切 $K$ 块，让 chunk $i$ 的 combine 落在 chunk $i+1$ 的 GEMM 中间；通信 SM 偷的那段 tile 是连续的，不散落在每个 chunk 之间，免得角色反复切换。最后一个 chunk 的 combine 让所有 SM 参与，消掉尾部。

数字（4×H100 SXM5、EP=4、每卡 132 SM / BF16 峰值 989 TFLOPS / NVLink 单向 450 GB/s，6 个模型、序列 2k/4k/8k）：单层几何平均 **2.89×**、端到端 **1.33×**，overlap ratio **47.1%**（Comet 14.3%、DeepEP 9.2%）；cost model 选出的 $(c,K)$ 与穷举最优平均差 8.2%，在线开销 0.54 μs；消融里空间调度（S）平均降 10.1%、时间调度（T）再降 14.1%，合计 22.8%。

边界：单节点、无跨节点 RDMA；端到端被 attention 稀释，序列越长越明显；$K$ 有上限，往上加就撞前面那 27%。

### Lancet：拿层外的计算去填

3.36 倍说明只藏 expert compute 不够，Lancet 的答案是：用 MoE 层之外的计算来填。

它在编译期拿到整张训练图的 IR，把 attention 的 QKV/MLP 沿 batch 维切成若干 partition，和 all-to-all 做流水；切多深、切几段用动态规划搜，目标是最小化迭代时间，约束是 partition 开销和显存。反向更硬——下游是 dX、上游是 dY——但 **dW 不依赖 A2A 的完成**，只需要本层输入激活和 dY，于是用 best-fit 贪心把 dW 塞进各个 A2A 的空隙。

数字（8 节点×8 卡 A100/V100、另有 L20 集群、GPT2-S/L-MoE）：非重叠通信最多降 **77%**（A100 上 66%）；端到端最高 1.3×（V100）、1.21×（A100），平均 1.22×/1.17×；cost model 误差 3.83%。

边界：切分本身增加计算（overlapped 和 non-overlapped 加起来可能比不切还高）；最优 partition 数从不超过 4；batch-prioritized gating 会把流水范围限在 MoE 层之后；它是编译期方案，图变了要重跑优化，论文里大多数模型的优化时间在 20 分钟以内。

### DisagMoE：让 attention 和 FFN 各占一组 GPU

attention 和 FFN 的 compute/comm 比不同，凭什么让它们抢同一批 SM？

DisagMoE（[2605.11005](https://arxiv.org/abs/2605.11005)）的 A-Worker 放 attention（组内 DP），F-Worker 放 FFN/MoE 专家（组内 EP），两组之间用 M2N/N2M 交换，而不是 all-to-all + P2P。AF-Pipe 把这次交换当成流水线的一级，combine 融进 M2N——省掉"先 combine 再 P2P"的第二次搬运，通信成本大约降到 $1/k$。三个 stream（forward/backward/communication）异步跑，稳态下两组都在算。

roofline 的依据是 attention 的算术强度随序列长度按 $O(s^2)$ 涨、FFN 和通信只按 $O(s)$ 涨，所以最优比例随 $s$ 变；它同时定两组各多少 GPU、NIC 带宽怎么分。

数字（128×H800、EP=16、序列 4K–32K）：16K 下最优 **A:F = 16:10**（26 卡），比 baseline 快 **1.56×**、比同构 16:16（32 卡）快 1.29×；对 Megatron-1F1B 1.59×–1.81×，对 Tutel/Comet 这类 SOTA overlap 1.2×–1.5×、对 DualPipe 1.05×–1.13×；非重叠通信比 Tutel 少 88%、比 Comet 少 75%、比 DualPipe 少 45%。

边界：静态分配——比例和带宽按已知的序列长度选好，workload 一变要重新规划；warmup/cool-down 气泡仍在，约为基线的 1/4；4K 下则接近 16:16；适合形状稳定的预训练，不适合请求长度剧烈波动的 serving。

### INC-MoE：把 all-to-all 卸进交换机

推理 prefill、跨节点 scale-out 时，dispatch/combine 可以写成 AllGather + ReduceScatter，而交换机天生会 multicast 和 reduction。

INC-MoE（[2610.36339](https://arxiv.org/abs/2610.36339)）的做法是：dispatch = 交换机内 AG（multicast 给其余 $N-1$ 个 rank），combine = 交换机内 RS（交换机先把 $N-1$ 份非 root 贡献 reduce 掉，root 只做一次本地加法）。妙处是方向：AG 主要吃接收，RS 主要吃发送，同时跑就吃满全双工。

带宽受限时间从 ring 的 $\frac{2(N-1)}{N}\cdot\frac{S}{BW}$ 降到 $\frac{S}{BW}$，提升为 $\frac{N-2}{2(N-1)}$——$N=4$ 时 33.3%，$N=8$ 时 42.9%，$N=2$ 时为零。kernel 侧是 persistent kernel，token 切成 band，gather/compute/reduce 三类 CTA 各司其职，稳态下三个 band 同时在飞，drain 时 gather 还会去帮 reduce。

数字（OLMoE-1B-7B prefill、4-way DP attention + 4-way EP、4 节点模拟）：TTFT 降 **20.2%**（24K token）到 **33.1%**（126K token），吞吐涨 25.3%–49.5%；并发 AG+RS 的等效带宽 163–172 Gbps，ring 约 130 Gbps，达到理论上界的 82%–86%。

边界：$N=2$ 没有收益；实测只有理论上界的 82–86%；交换机是拿 accelerator 模拟的，真机上小消息的固定开销会小很多，但也还没在真交换机上验证；inference-only，训练侧不能直接套。

## 6. 本地模型：$(c,K)$ 是联合的，底部很平

`code/comm-overlap/moe_overlap.py` 是个纯 CPU 解析模型：一条通信通道（D 优先于 B）、一条计算通道（可用 SM 是 `total_sms - comm_sms`），每个 chunk 的 $D_i \to C_i \to B_i$ 用 list scheduling 求 makespan，`--sweep` 把两者交叉扫一遍。

按默认参数（`tokens=16384`、`hidden=4096`、`topk=2`、`ep=8`、`total_sms=132`、`bw_gbps=400`、`tflops=700`、`sm_sat=16`），每卡 2048 个 token、乘 top-k 得 4096 行，dispatch 总量约 33.6 MB，GEMM 总量约 1.44 TFLOP。

$c=4$ 时通信效率是 $4/16=25\%$、有效带宽 100 GB/s，切 16 块共约 671 μs，GEMM 约 2126 μs——**通信总量只有计算的三分之一，4 个 SM 就够**。

| 结论 | 数值 | 为什么 |
|---|---|---|
| 通信 SM 的最优点 | `comm_sms=4` | 通信比计算短，加 SM 只减通信 |
| 模型里的 chunk 数 | 越多越好 | 没有 per-chunk 开销 |
| 真实系统能用的 chunk 数 | 2–4 | 每块一次 launch，GEMM 还变碎 |

$c=8$ 时通信减半，但 GEMM 少了 4 个 SM（2126 → 2195），净结果反而慢 48 μs；$c=2$ 时暴露的 startup 从 42 μs 翻到 84 μs——底部很平，$c=4$ 是这张网格上的最优点（DeepEP 的估计器也是往 4 的倍数上取，算是巧合的呼应）。

$C$ 从 1 到 16 能降 600 多 μs，全靠"没有 per-chunk 开销"这个假设；真实系统里每块一次 kernel launch + event 同步，chunk 边界还要对齐 expert 的 token 布局，所以 $K$ 落在 2–4（Weave 图里是 $K=2$，Lancet 的最优 partition 不超过 4）。

> 这是解析模型，不是实测：带宽、算力、SM 饱和点都是假设值，算不出 routing skew、L2 争用、NVSHMEM 的延迟。它的价值是告诉你 $(c,K)$ 是联合的、底部很平；具体数值要用真实 trace 校准。

## 7. 怎么选：一张表

| 层级 / 方案 | 它把什么藏进什么 | 主数字 | 先确认什么 |
|---|---|---|---|
| 只调 API · DeepEP | 把 wait 推后，SM 预算可调 | 工业基线 | 今天就想上线、不改 kernel |
| 层内 · Comet | 按 tile 重排，本地 token 先算 | 单层 1.96× | 单节点、敢改 kernel 和 GEMM 后端 |
| 整层 · Weave | 整层一个 megakernel，routing 后选 $(c,K)$ | 单层 2.89× | 单节点规模、形状会变 |
| 整图 · Lancet | 拿层外的 attention 和 dW 填 A2A | 非重叠通信 −77% | 有编译器、图稳定、多节点 |
| 模块间 · DisagMoE | attention 和 FFN 分两组，combine 融进 M2N | 1.81× vs 1F1B；1.34× vs SOTA | 形状稳定的大规模预训练 |
| 网络内 · INC-MoE | A2A 卸进交换机，AG 与 RS 同时跑 | TTFT −33.1% | 有 INC 交换机、只做推理 |

换个切法看这张表：六个方案其实只做三件事——减字节（FP8 dispatch、不传 padding token）、减跳数（尽量在 NVLink domain 内解决、把 collective 卸进交换机）、减暴露（切分、错位、SM 分区）。先判断你的瓶颈在哪一类，比对着 speedup 数字选更可靠。

判断顺序只有一个：先量**通信占这层多少**（一成左右就别上复杂 overlap，调 `num_sms` 和 chunk 就够；接近一半才值得动 kernel），再量**A2A 是计算的几倍**（超过 1 就说明只藏计算不够，该往整图或分组走）。

然后看**单节点还是跨节点**：跨节点时 all-to-all 占比能从单节点的 22% 涨到 8 节点的 78%，这时"减通信"比"藏通信"更值。最后两个过滤器是形状稳不稳、愿意付多少工程代价。

三个容易踩的坑：

- **只报 overlap ratio**。Weave 的 47.1% 和 Comet 的 14.3% 是同一个 MoE forward 窗口、同一指标下测的，可以直接比；换别的论文的 ratio 之前要先对齐定义，真指标还是 exposed time 和端到端。
- **忽略 routing skew**。模型里 `--skew 1.0` 是均衡假设，真实热点专家会让某个 chunk 明显慢于平均，收益被吃掉。
- **忽略 tail**。分块能把 tail 降到 $b/K$，降不到 0。

## 只记一件事

**MoE 的重叠上限就是计算那一头**：切块只能摊薄 startup 和 tail，摊不薄"通信比计算长"。所以顺序是：先量通信占比和 A2A 倍数，再决定是调 API、切 tile，还是换掉搬运本身。

SM 怎么分、persistent kernel 怎么写、warp specialization 怎么摆，是下一层的问题——[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})会把 Comet 和 Weave 的 CTA 分区拆到 kernel 级。

推理侧的阈值可以直接跳[第九篇]({{< relref "comm-overlap-09-inference" >}})，整张地图在[总结篇]({{< relref "comm-overlap-11-summary" >}})，而[2026-11 补遗]({{< relref "comm-overlap-13-uniep" >}})里的 UniEP 把"逐位可复现"做成了 overlap 的硬约束。

---

**系列导航**：[五：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}}) · [七：Kernel 与 SM]({{< relref "comm-overlap-07-kernel-sm" >}})

> 配套代码：`code/comm-overlap/moe_overlap.py`
> 自测：`python3 moe_overlap.py --test`
> 扫描：`python3 moe_overlap.py --sweep`
