---
title: "通算并行研读（五）：张量并行——通信挪不走，就把 GEMM 切开"
date: 2026-11-02
draft: false
weight: 6
series: ["comm-overlap"]
tags: ["通算并行", "系列", "张量并行", "Flux", "T-CCL", "TileLink", "kernel fusion"]
categories: ["训练框架"]
---

**张量并行的通信长在层里、挪不到旁边，只能把 GEMM 按 tile 切开，让算到的行和还在传的行同时跑；代价是四条路线各自拿代码量、SM 或通用性去换**。

读完你会知道：

- 为什么 all-gather 还没做完，GEMM 就能先算其中几行；
- 四个方案各自把什么融进了什么，招牌数字和边界在哪；
- 该先看通信占比，还是先挑实现路线。

## 1. 通信长在层里，挪不走

TP 的经典形态是在第二个 GEMM 的输出上做一次 all-reduce；换成序列并行（sequence parallel，SP）之后，它变成两个 GEMM 外侧的一次 AG（all-gather，把别人的分片收过来）和一次 RS（reduce-scatter，把各卡的部分和加起来、每张卡只留自己那段）。位置挪了，性质没挪。

它卡在关键路径上，因为输出 $Z$ 的每一个元素都要这次通信的**全部**元素。

DP 的通信等的是整个反向算完，PP 的通信等的是另一个 microbatch，TP 的下一次通信等的是它自己——层内没有第二件活可以填。

一层 MLP 是 $Y=\mathrm{GeLU}(XA)$、$Z=YB$：权重 $A$ 按列切，$B$ 按行切，切几份就是 TP 度 $N$。第一个 GEMM 的输出只有列方向的一段，第二个 GEMM 算出的 $Z_r=YB_r$ 只是 $N$ 个部分和里的一个。所以不用 SP 时前向只有一次 all-reduce、反向两次；用上 SP，通信搬到两个 GEMM 外侧，前向 AG + GEMM、GEMM + RS，反向对称，一共四次。

attention 的形态不同、结论一样：QKV 投影按 head 切，output projection 在 head 维上求和，两次通信仍然都在层内。区别在比例——decode 时 $M=1$，GEMM 变瘦而通信量不减，$g/c$ 很容易超过 1。

### 1.1 反向多一个窗口：dgrad 不必等 wgrad

一个 Linear 的反向有两个 GEMM，dgrad（对输入求导）和 wgrad（对权重求导）。它们共享输入、彼此不依赖——dgrad 在等 reduce-scatter 的时候，wgrad 可以先算。

所以反向的 TP 通信往往比前向好藏，不是因为通信变快了，而是旁边多了一件活。Megatron 的 `delay_wgrad_compute` 把这个窗口显式化，前置条件（[第四篇]({{< relref "comm-overlap-04-data-parallel" >}})提过）是只在 TransformerEngine 实现下可用。

代价是显存：wgrad 往后挪，输入激活就要活得更久。**overlap 的收益从来不是白拿的，要么用显存换，要么用代码复杂度换，要么用计算效率换**。

## 2. 能融，是因为 GEMM 一行一行地算

答案不在通信库里，在 GEMM 自己的数据依赖上。

SP 的 AG 补的是序列维（M 维），每张卡持有一段 M tile。沿 M 维看，第一个 GEMM 的输出和输入逐行对应：

$$
G[m,:] = X[m,:]\, A
$$

第 $j$ 段序列一到，各卡就能算这几行的本地列；GeLU 逐元素，第二个 GEMM 的每个输出行也只由 $Y$ 的对应行决定。于是一段的 $Z$ 部分和可以接着算，算完立刻参与 reduce-scatter。

```text
      rank 0          rank 1          rank 2
    ┌────────┐      ┌────────┐      ┌────────┐
    │  X0..  │      │  X1..  │      │  X2..  │   ← 各持一段 M tile（序列切片）
    └───┬────┘      └───┬────┘      └───┬────┘
        │   tile 级 all-gather（逐段到）
        ▼               ▼               ▼
    ┌──────────────────────────────────────────┐
    │  G[m] = X[m] · A                         │
    └──────────────────────────────────────────┘
        ▲               ▲               ▲
        │               │               └── t2 到了，算 t2 这几行
        │               └────────────────── t1 到了，算 t1 这几行
        └────────────────────────────────── t0 在本地，先算
```

看最后三行箭头：本地那段先算，远端两段谁先到谁先算，三段之间不需要互相等。这不是把通信藏起来，而是把"等整份数据"换成"等下一段数据"。

输出方向同理。第二个 GEMM 的某个输出 tile 一旦算完，这块部分和就不再变了，可以直接参与 reduce-scatter——所以 RS 能融进 epilogue（收尾阶段），写出时直接写到该去的地方。

另一种实现切的是 K 维，AG 送来的就是 K tile。这时 GEMM 的 K 循环本身就是求和，$G=\sum_j X_jA_j$ 里第 $j$ 块没到只影响属于它的那几次累加，其它块照跑。AG 补哪一维，就按哪一维的 tile 推进 GEMM，道理是同一个。

## 3. 按设备数切 chunk，只能拿到一半

TransformerEngine 的 UserBuffers 路线把 GEMM 和 collective 都切成 $N$ 份（或 $2N$ 份用上双向链路），靠 stream/event 让第 $i$ 块的通信和第 $i+1$ 块的计算并行。TPU 上很好用，GPU 上有三个硬伤：

- 执行时机排不准，生产环境里 stream 一多就不确定；
- 一个大 GEMM 变成 $N$ 个小 GEMM，每个都填不满 SM；
- NCCL 的 wait 是 kernel 级的，产它的 GEMM 只要还有一个 tile 没写完，collective 就发不出去。

这就是中粒度和细粒度的分界：切的对象不是算子，而是 tile。下面四个方案都站在这条线上。

## 4. Flux：把 wait 塞进 GEMM kernel

Flux（[2406.06858](https://arxiv.org/abs/2406.06858)）一句话：overdecompose 到 tile，把有依赖的通信和 wait 融进那个 GEMM kernel，最后只 launch 一个 kernel。

一个 tile 对应一个 thread block（视实现也可以是 warp 或 cluster），tile 的划分与 TP 度解耦——不必等于设备数，K 循环和 tile 形状按 GEMM 自己的最优来选。AG 摆在 prologue，RS 摆在 epilogue，wait 长在消费者那一侧：哪个 tile 等哪个信号，编译期就定好。

它不静态分 SM。整个 grid 跑同一份 GEMM，每个 block 自己决定这一轮是算还是搬——用的是"融"和"切"，不是"让"。

tile coordinate swizzling 把 block index 到 tile 坐标的映射，和信号到达顺序对齐；通信顺序在 host 侧按拓扑选（节点内 NVLink 用从 `local_rank+1` 开始的环，PCIe 也用环）。消融显示开了 swizzling 总是更好，矩阵越大差距越大。搬运走节点内 P2P 或 TMA、跨节点 NVSHMEM，没有额外的通信 kernel。

GEMM + collective 的算子级对比，基线 cuBLAS + NCCL，对手是 TransformerEngine：

| 集群 | 相对 TransformerEngine | overlap efficiency |
|---|---|---|
| A100 PCIe（8 卡/节点，节点内 PCIe） | 1.20×–3.25× | 41%–57% |
| A100 NVLink（8 卡/节点） | 1.01×–1.33× | 36%–96% |
| H800 NVLink（8 卡/节点） | 1.10×–1.51× | 37%–93% |

同一套融合 kernel，在 PCIe 上 overlap efficiency 只有 41%–57%，到 NVLink 能到 93%–96%——链路越快，能藏住的份额越高。

端到端是 GPT-3 175B 与 Llama-2 70B，128 GPU（2 路 DP × 8 路 PP × 8 路 TP），对比 Megatron-LM 与 vLLM：训练最高 1.24×、prefill 最高 1.66×、decode 最高 1.30×。这三个数是跨 A100 PCIe / A100 NVLink / H800 NVLink 三个集群取的最大值；分集群看是 A100 PCIe 的 1.24× / 1.46× / 1.28×、A100 NVLink 的 1.05× / 1.45× / 1.30×，H800 NVLink 是 1.10× / 1.66×、decode 没有加速。

负结果紧跟主数字。论文原话是 "no decoding speedups"，同一段里还有 5 个配置比不重叠的 vLLM 基线更慢。H800 NVLink 上 $m \le 1024$ 时相对 TransformerEngine 只有 0.95×–1.03×（下限那端是变慢），$m=64$ 的 ReduceScatter 是唯一比它更慢的点，因为 8 路 TP 之后沿 m 维的 TMA store size 只剩 8。

还有一条更根本的：通信占比低的时候，overlap 做得再好也没用。A100 NVLink 训练时通信只占 8%–11%，即使平均 overlap efficiency 有 63%，端到端也只有 1.04×–1.05×。

顺带钉一个定义。Flux 论文说得很直接：overlap efficiency 为负，意味着这个方法比不重叠的基线还慢。所以 $\eta = 1 - T_{\text{exposed}} / T_{\text{comm}}$ 允许负值——切碎、抢带宽、多一次同步，都可能让总时间超过老实串行。**看到负值不要当成测量错误，那是结论**。

融合之后 nsys 上**看不到独立的通信 kernel**，拿"有没有 NCCL kernel"判断 overlap 有没有生效会得到假阴性：通信退化成了 GEMM kernel 内部的 load/store 和 barrier wait，只能从 SM 活跃度、HBM 流量和有效算力反推。[第十篇]({{< relref "comm-overlap-10-measurement" >}})讲怎么量。

## 5. T-CCL：不融合，只让通信少占 SM

Flux 是把通信塞进计算 kernel。T-CCL（[2610.07098](https://arxiv.org/abs/2610.07098)）走另一条：不融合，让通信几乎不占资源，计算独享剩下的 SM。

它用 TMA 同时卸载"搬"和"reduce"。TMA 是 Hopper 起的异步搬运单元，一次 bulk transfer 由 block 里一个被选出的线程发起，之后走专用硬件；它还支持异步的 element-wise reduction（add/min/max），所以 all-reduce 的归约也能交给它，不用一堆线程做加法。

每个 collective 被拆成 shard-in 和 shard-out 两段——拥有某个 shard 的 rank 先把别人的贡献收进来在本地归约，再把结果推出去——再 lowering 成一组 TMA 的 load/store/reduce 任务，按流水线发射。一个线程就能发一整串，SM 侧只剩"发指令 + 等 barrier"的成本。

于是活跃 SM 只要 1–6 个：AllGather 从 128 KB 的 1.32 个到 128 MB 的 6.60 个。相对 NCCL，不限通信资源时最高 2.4×；把通信限制到 8 个 CTA（H100）/ 9 个 CTA（GH200）时最高 3.42×——后者才是它想证明的场景，资源受限时谁更抗压。

| 消息大小 | T-CCL | NCCL | Symmetric NCCL |
|---|---|---|---|
| 128 KB | 1.32 | 3.30 | 1.98 |
| 32 MB | 6.60 | 23.43 | 7.92 |
| 128 MB | 6.60 | 23.76 | 9.24 |

消息越大，NCCL 吃的 SM 越多，T-CCL 到 6.60 就基本封顶；128 KB 那一行 NCCL 要 3.30 个 SM 才干同样的活。

换进 FlashOverlap 当 backend（GH200，24 个 GEMM 形状平均，基线是串行 cuBLAS + NCCL）：TP=2 从 1.12× 提到 1.25×（最高 1.51×），TP=4 从 1.04× 提到 1.14×（最高 1.34×）。收益来自"通信少占的 SM 还给了 GEMM"，而不是通信本身变快。

接进 vLLM（Qwen2.5-72B-Instruct，TP=4，GH200）：对话负载几何平均 1.10×、最高 1.22×，decode-heavy 负载 1.15× / 1.31×。

边界四条。最要命的两条是不能上 CUDA Graph、H100 小消息有回归：vLLM 实验跑在 eager 模式（planning 和 launch 有 host 侧工作），AllReduce 只有 0.94×–2.74×、ReduceScatter 0.95×–3.42×，最小值都小于 1。

另外两条是范围限制：只做单节点（靠 P2P 访问对端显存），评测规模 TP=2 / TP=4；TP=4 的最差形状仍低于串行基线——小 GEMM 没多少计算可以藏通信，这不是 T-CCL 独有的问题。

## 6. TileLink：把这套融合交给编译器

Flux 是手写几千行 CUDA 换来的。TileLink（[2503.20313](https://arxiv.org/abs/2503.20313)，ByteDance Seed，后端是 Triton-distributed）问的是：这套融合能不能自动化？

前端用 tile-centric primitives 把两个设计空间解耦：计算侧按 GEMM 的最优选 tile，通信侧按链路的最优选 tile，两边形状可以不一样，依赖用细粒度的 barrier/notify 维护（`producer_tile_notify`、`peer_tile_wait`）。后端把原语 lower 成 TMA / NVSHMEM 指令，并负责 shape、rank、channel 的映射——编译期仿射变换或运行期查表（MoE 的动态路由要后者）。

资源绑定于是成了编译器的参数：绑 DMA、绑 SM、或混合（GEMM+RS 用 DMA 做 scatter、SM 做归约）。这是"让"第一次变成可调项。论文的 MoE 例子里，一个 AllGather + MoE kernel 用 20 个 SM 做通信，其余算 expert。

数字（8×H800），相对非重叠基线 1.17×–20.76×，20.76× 出现在 MoE 对 cuBLAS+NCCL 的对比上：

- AG+GEMM **1.27×**、达到 Flux 的 94.5%（Flux 自己 1.34×），这里把 AllGather 绑到 DMA；
- GEMM+RS **1.25×** 于 cuBLAS+NCCL、1.28× 于 Flux、2.22× 于 Async-TP PyTorch，用混合绑定；
- 整个 MLP 是 Flux 的 **101.4%**，MoE 层平均 1.14× 于 vLLM；
- 端到端 8 个模型平均 **1.32×**（dense 1.20×，MoE 1.54×），两节点 16 GPU 1.29×。

代价换到了代码量上：几百行 Python，对几千行 CUDA。

边界三条。AG+GEMM 仍落后 Flux 5.5%，手写的 prologue 融合（尤其 TMA + swizzling）还是更狠，它赢在 GEMM+RS 和 MoE（Flux 和 Async-TP 都不支持 MoE 的 overlap）。

另外两条是前提：20 个 SM 是论文代码示例里手选的例子，不是搜出来的最优，真实部署要自己扫这个预算、而它和形状强相关；只支持 NVIDIA，换硬件这套 lowering 要重写。

## 7. CommFuse：拆成 P2P，让最后一块不用通信

前三个都在 kernel 层动手。CommFuse（[2604.24013](https://arxiv.org/abs/2604.24013)）回到算法层问：tail 为什么一定存在？

切成 $N$ 块，前 $N-1$ 块的通信可以藏在计算里，最后一块的通信没人藏——它后面没有计算了。切得越细，tail 的绝对时间越小，算术强度也越差。

它把 RS / AG 拆成 ring 顺序的 P2P（FuseRS / FuseAG），再按 rank 重排计算顺序：本 rank 最后要留下的那片输出安排到最后算，算完就结束，不需要再通信；FuseAG 反过来，本地已有的那片先算，别人送来的后算，等数据的时间被本地计算盖住。

于是"最后一块必须通信"变成"不必通信"，tail 被结构性地消掉，而不是靠把 chunk 切得更细来摊薄。

代价是通信从一次 collective 变成 $N-1$ 步 P2P，同步点变多，每步都要自己保证顺序。论文的立场是这类同步点对点、可控，不像"拆了 collective 但总通信量上升"的方案那样容易把省下的 tail 吃回去；它也不依赖 batch size，不会像按 batch 切那样在小 batch 下退化。

数字来自论文的 MLP 实验（TP=4）：

| 方案 | 通信开销 (ms) | 端到端 (ms) |
|---|---|---|
| Vanilla 基线 | 43.8 | 122.3 |
| 数据切分 | 7.1（−83.9%） | 83.9（−31.4%） |
| CommFuse | 0.1（−99.8%） | 77.1（−36.9%） |

通信开销从 43.8 ms 压到 0.1 ms，端到端只从 122.3 ms 降到 77.1 ms——通信省下 99.8%，总时间只省 36.9%，因为剩下的大头是计算。

边界两条。论文评测段**没有写明加速器型号和互联**（正文只说 accelerator / device），这组数不能直接和 Flux 的 A100 NVLink / H800 NVLink 对齐。

长序列注意力上，数据切分方案自己会翻车：$B=4, S=8192$ 时切分方案 279.7 ms，比 baseline 的 195.6 ms 还慢（CommFuse 自己是 145.1 ms）——切分不是免费的，序列维切太碎会把 attention 打成 memory-bound。

它的机制和前面三个**正交**：不要求 fused kernel，只是改调度顺序，实现层还能再叠融合 kernel。MoE 场景下的版本见[第六篇]({{< relref "comm-overlap-06-moe" >}})。

## 8. 一个 all-reduce，四种代价

| 系统 | 把什么融进了什么 | 主数字 | 边界 |
|---|---|---|---|
| Flux | AG 进 prologue、RS 进 epilogue，wait 长在 tile 消费者侧，单 kernel | overlap efficiency 最高 96%（A100 NVLink） | H800 上 decode 无加速；小 m 掉到 0.95× |
| T-CCL | TMA 同时卸搬运和归约，collective 拆 shard-in / shard-out | 限 8/9 CTA 时相对 NCCL 3.42× | 无 CUDA Graph；只做单节点 |
| TileLink | 原语 + 编译器 lower，资源绑定变成参数 | 非重叠基线 1.17×–20.76× | AG+GEMM 落后 Flux 5.5%；仅 NVIDIA |
| CommFuse | collective 拆 ring P2P，重排顺序消掉 tail | 通信开销 43.8 → 0.1 ms | 没写 GPU 型号；长序列切分更慢 |

四个方案正好对应[导读]({{< relref "comm-overlap-00-index" >}})里的四个动作：Flux 是"切 + 融"，T-CCL 是"让"（让出 SM），TileLink 是"融 + 让"的自动化，CommFuse 是"切 + 错"（重排顺序）。**同一个 all-reduce，四种解法，代价完全不同**。

## 9. 怎么选：先算账，再看占比，最后才挑路线

先算一笔账。切 $n$ 块、每块通信 $g$、计算 $c$，两级流水的 makespan 是：

$$
T = \max(nc,\; ng) + \min(c,\; g)
$$

$\max$ 是两条流水并行跑、总长由慢的那条定，$\min$ 是填充排空。随 $n$ 翻倍的那组数这里不展开，只记结论：$g > c$ 时切分不改变瓶颈，只是把同一个瓶颈摊到更多块上，顺带多付 $n$ 次 launch。

1. **先看通信占比**。层内通信低于 10% 时，任何 overlap 的天花板就是那 10%（Flux 自己在 8%–11% 的场景下也只有 1.04×–1.05×）。这时候该减少通信量，不是换 kernel。
2. **单节点、形状稳定、有人力写 kernel → 融合 kernel**。收益最大（A100 PCIe 上 1.20×–3.25×），代价是几千行 CUDA 和跟硬件强绑定。
3. **不想改模型代码、推理服务想立刻拿收益 → 换 collective backend**。前提是单节点、能接受没有 CUDA Graph、消息不太小；好处是正交，不碰模型，vLLM / TE UserBuffers 都能接。
4. **形状多、要覆盖 MLP / attention / MoE → 让编译器生成**。接受比手写低几个点的性能（AG+GEMM 差 5.5%），换几百行 Python 覆盖全部形态。
5. **跨节点 + 小 batch → 先别想 overlap**。$M$ 压过 $\gamma C$ 时上界已经钉死，切得越细、暴露的绝对时间越大；正确动作是降 TP 度、改 PP / EP，或者量化通信。

三件事指向同一个原因：Flux 的 H800 decode 无加速、CommFuse 的长序列切分翻车、T-CCL 在 TP=4 上的最差形状低于串行。**一块计算本身已经小到藏不住一次通信时，任何 overlap 都是负收益**。判断标准很朴素：通信时间除以计算时间，大于 1 就先去减通信量。

## 结论

如果只记一件事：**TP 的 overlap 不是把通信挪走，是把等待从整份数据换成下一段数据**。这个窗口是 GEMM 的数据结构给的，四个方案只是用四种代价把它兑现。

下一篇进 MoE：TP 的通信是"一个张量要所有人都有"，MoE 的 all-to-all 是"每个 token 去不同的地方"——依赖更弱（能按 token 切），模式更乱（routing 决定流量），于是切、错、让三件事必须一起做。见[第六篇：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}})。

---

**系列导航**：[四：数据并行]({{< relref "comm-overlap-04-data-parallel" >}}) · [六：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}})

> 配套代码：`code/comm-overlap/overlap_model.py`（`tp_exposed` / `tp_ledger`；TP 的账本和 PP/DP/MoE 共用一个文件）
> 自测：`python3 overlap_model.py --test`
