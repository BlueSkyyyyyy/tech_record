---
title: "通算并行研读（七）：不是 stream 叠不叠，是 SM 归谁"
date: 2026-11-02
draft: false
weight: 8
series: ["comm-overlap"]
tags: ["通算并行", "系列", "CUDA", "persistent kernel", "warp specialization", "SM 分区", "TMA"]
categories: ["算子开发"]
---

**kernel 层的 overlap 不是把两条 stream 叠起来，而是决定谁占哪个 SM、哪个 warp、哪条发射槽；最优点是一段平台，不是某个点，要选平台左缘**。

读完你会知道：

- 七种资源里，为什么只有 SM 数能「分」；
- 一个 warp 为什么不能既等通信又算 GEMM；
- 分区为什么该选平台左缘，而不是最小值点。

## 1. nsys 说叠得很好，ncu 说 GPU 在空转

nsys 上通信和计算叠得很漂亮，overlap ratio 90% 以上；换 ncu 看同一个 GEMM，Tensor Core 的忙碌比例掉了，warp 大量停在 `stall_long_scoreboard`，SM 的 active cycles 却没降。

两个工具都没错，量的是不同的东西：nsys 量「时间轴上有没有重叠」，ncu 量「重叠的时候，硬件资源被谁拿着」。前者是软件抽象（stream），后者是硬件实体（SM）。两条 stream 能被同时调度，不等于两份工作能同时拿到发射机会。

## 2. 七种资源，只有一种能「分」

Hopper 的一个 SM 是一笔固定资产：4 个 warp scheduler、256 KB 寄存器堆（64K 个 32-bit 寄存器）、228 KB 可配置 shared memory（单 CTA 最多 227 KB）、一个 TMA 单元，外加它到 L2/HBM 的口子；整卡还有一份功耗帽。

通信要用的和计算要用的，是同一批资产：

| 资源 | 通信怎么吃它 | 计算怎么吃它 | 冲突时的表现 |
|---|---|---|---|
| SM / CTA 槽位 | NCCL、NVSHMEM 的 copy + reduce CTA 常驻 | GEMM 的 tile CTA | 计算 wave 装不满，尾部变长 |
| warp scheduler | 通信 warp 轮询 flag、等 mbarrier | MMA warp 需要连续发射 | issue slot 被占，MMA 发射间隔变大 |
| 寄存器 | 通信用量低，但按 CTA 整块分配 | wgmma 的 accumulator 吃满 | occupancy 下降，双 buffer 放不下 |
| shared memory | 通信的 staging buffer（TMA 目标） | GEMM 的 A/B tile + epilogue | 一个 CTA 占满 smem，别的 CTA 进不来 |
| TMA / copy 引擎 | bulk 拷贝、multicast | operand 加载 | TMA 队列排队，计算等 operand |
| L2 / HBM 带宽 | 读写对端数据、做 reduction | operand、activation 流 | L2 hit rate 掉，HBM 拥塞 |
| 功耗 / 频率 | 与计算同时忙，抬高峰值功耗 | 同上 | 撞功耗墙降频，两边一起慢 |

只有第一行是**数量**问题：分几个 SM。其余六行分不了，只能决定同一批 SM 内部的份额归谁——哪个 warp 拿 issue slot、哪个 CTA 拿寄存器、哪个 stage 拿 smem、谁排队等 TMA、带宽和功耗预算归谁。

所以 SM 分区只是第一层。第二层的粒度不是 SM，而是 CTA 槽位 × 每个 SM 能常驻几个 CTA：通信 CTA 要 100 KB smem、计算 CTA 要 200 KB，就永远不可能共居。一个 CTA 独占一个 SM 时，「分几个 CTA」才等于「分几个 SM」；一旦两个 CTA 能挤进同一个 SM，账要按 residency 重算。

## 3. 三种写法，区别在谁做决定

**双流两 kernel**（launch 前，driver 决定）。NCCL 走一条 stream，cuBLAS/CUTLASS 的 GEMM 走另一条，中间用 event 连依赖。省事的地方是不改 kernel：NCCL 的拓扑探测、channel 选择、协议都是现成的，Megatron、DeepSpeed 也有成熟 hook。

代价四条：

- **每个 chunk 一次 launch**：微秒量级，chunk 切到几十个，launch 本身就成了固定税；
- **依赖只能落在 kernel 边界**：做不到「第 3 个 tile 算完就发第 3 个 tile」；
- **SM 争用不可控**：NCCL 自己决定开几个 channel、每个 channel 几个 CTA，你只能靠 `NCCL_MAX_NCHANNELS` 间接影响；
- **两套 occupancy 配置打架**：两个 kernel 各按自己的寄存器/smem 编译，staging buffer 和 operand tile 没法复用同一块 smem。

Comet 对这条路的批评很直接：把通信和计算封进不同 stream 的不同 kernel，"restricts control over hardware resources and results in non-deterministic kernel performance"。注意后半句——不只是慢，是不可预测：你不知道这一轮 NCCL 拿了几个 SM，也就不知道 GEMM 被拖慢多少。

**Fused kernel + CTA specialization**（launch 时，程序员决定）。一个 kernel，grid 里 CTA 分两类：producer 负责通信，consumer 负责 GEMM；grid 大小取 SM 数（Hopper 上是 132），于是「分几个 CTA 给通信」近似等于「分几个 SM 给通信」。Flux 不预设比例，整个 grid 跑同一份 GEMM，每个 CTA 在自己的 K-loop 里决定这一轮是算还是搬（见[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})）。

关键是 tile 级信号：consumer 算完一个 tile 就把 ready 标志打上，producer 看到就发，不等整个 GEMM 结束；全程没有 global barrier，不同 tile 处在流水线的不同阶段，谁先好谁先走。代价是 kernel 变成手写、和 shape 强绑定，producer/consumer 比例成了新超参，某个角色先干完只能空转（Flux 的自调度缓解了这一点，换成了 tile 顺序和信号的开销）。

**Persistent megakernel**（运行时，cost model 决定）。Weave 把 dispatch、GEMM1（上投影）、SiLU、GEMM2（下投影）、combine 五个操作塞进一个 persistent kernel，grid 仍是 132，一次 launch 跑完整层。

它多了两个能力。角色可变：每层 routing 结束后，这一层的通信量和计算量就定了，kernel 里跑一个轻量 cost model，算出该层的通信 SM 数 $c^*$ 和 chunk 数 $K^*$，分区从「每个模型一次」变成「每层每卡一次」。

工作可偷：通信 SM 干完自己的活，如果 GEMM 还有 tile 没人算，就去偷一个（bubble stealing），「通信 SM 在 GEMM 中段空闲」的那段气泡就这样被填上。

代价也清楚：整层重写、绑定特定模型结构、cost model 得在 device 上跑（Weave 测到 0.54 μs，占 MoE 层时间不到 0.021%）。这不是能塞进现有框架的改动，是一个新 runtime。

不动 kernel 的第四种：FlashOverlap（2504.19519）把「什么时候发」从 kernel 边界挪到一个 wave 的边界。GEMM 还是原来的 GEMM，只在一个 wave 的 tile 算完时打一个 signal，通信线程收到就去调 NCCL。粒度不能太细——4×RTX 4090 上，一个 192 KB 的 tile 单独做 AllReduce 只能跑到 13% 的带宽利用率，所以按 wave 分组（G1/G2/G3）。

代价落在两处 reorder。发之前要把 block swizzling 打散的 tile 重排成连续地址，融进 epilogue，代价 0.07% / 0.35%（A800 / 4090）；通信完再改回顺序，加在 RMSNorm 上，代价 7.46% / 7.93% / 8.45%（tile / subtile / subtoken，A800），4090 上是 8.80% / 8.78% / 9.63%——大约 7–10%。

整体 1.02–1.65×。它的价值不在数字，而在示范了一条低侵入路线；代价是 reorder 的带宽税，以及一个事实：signal 太细会被消息碎片反噬，太粗就退化成 kernel 边界。

四种写法回答的是「谁来做分区决策」；定下来之后，同一批 SM 上还有两个旋钮：warp 怎么分工（下一节）、SM 之间怎么分（再下一节）。

## 4. 一个 warp 不能既等通信又算 GEMM

既然通信只是搬数据，为什么不让同一个 warp 顺手发一下？「顺手」在 GPU 上要付三笔账。

**发射是串行的**。TMA 发起只要一条 `cp.async.bulk.tensor`，但「发起 → 等 mbarrier → 消费」是一条长延迟依赖链。warp 在链上等的时候已经不是 eligible warp，scheduler 不会让它去发 wgmma；而 Hopper 的 wgmma 是 warpgroup 级（128 线程）的异步 MMA，Tensor Core 的吞吐靠连续发射维持，中间插一次等 barrier 就出空隙。

**寄存器要按两条角色取峰值**。producer 要描述符、地址、循环变量，consumer 要 accumulator 和 operand fragment；一个 warp 兼两职，寄存器按两条角色的峰值活跃集（近似取和）分配，occupancy 就下来了——occupancy 一掉，双 buffer 都放不下。

**wait 是 warp 粒度的**。只要这个 warp 还在等 barrier 的 phase，它就不会发下一条 MMA。把 wait 放到专门的 producer warp 上，consumer 只在真缺 tile 时才等，不会被「下一块还没搬完」拖住。

所以标准做法是角色分离：producer warp 只搬，consumer warpgroup 只算，中间用 mbarrier 握手。不是为了架构好看，是为了让 Tensor Core 的 issue 流不断。

TMA 让这件事变便宜。Ampere 的 `cp.async` 是 per-thread 异步拷贝，每个线程自己发、自己 `cp.async.wait_group`；Hopper 的 TMA 是单线程发起、硬件搬运，SM 这边几乎不占线程。搬 1 MB，`cp.async` 要一堆线程算地址，TMA 只要一个线程发一条指令。T-CCL 把整条 collective 拆成一串 TMA load / store / reduce，128 KB 的 all-gather 平均只用 1.32 个活跃 SM，NCCL 要 3.30 个——同一份通信，SM 占用差 2.5 倍。

握手的六个步骤放在文末附录。主线只要记住一句：tile 级 overlap 能成立，靠的是同步粒度等于 buffer 粒度，而不是层粒度。

## 5. 分区的最优点是一段平台，不是一个点

现在只剩一个问题：给通信 $s$ 个 SM，makespan 是多少。

把一层看成两份工作：计算 $W_c$（单位 SM·μs）只用剩下的 $S-s$ 个 SM，通信 $W_m$ 只用分到的 $s$ 个，且有效带宽在 $s_0$ 个 SM 之后饱和。

$$
T_{\text{compute}} = \frac{W_c}{S-s}, \qquad
T_{\text{comm}} = \frac{W_m}{\min(s, s_0)}, \qquad
T = \max\left(T_{\text{compute}}, T_{\text{comm}}\right)
$$

所以 makespan 由慢的那一头决定。真实 GPU 上计算按 wave 走，再套一层量化：$T_{\text{compute}} = \lceil (W_c/(S-s))/u \rceil \cdot u$，$u$ 是一个 wave 的时间。参数取 $S=132$、$W_c=10000$、$W_m=3000$、$s_0=16$、$u=50$，扫一遍：

| $s_{\text{comm}}$ | $T_{\text{compute}}$ | $T_{\text{comm}}$ | $T$ |
|---|---|---|---|
| 0 | 100.0 | inf | inf |
| 4 | 100.0 | 750.0 | 750.0 |
| 8 | 100.0 | 375.0 | 375.0 |
| 12 | 100.0 | 250.0 | 250.0 |
| 16 | 100.0 | 187.5 | 187.5 |
| 20 | 100.0 | 187.5 | 187.5 |
| 32 | 100.0 | 187.5 | 187.5 |
| 33 | 150.0 | 187.5 | 187.5 |
| 36 | 150.0 | 187.5 | 187.5 |
| 65 | 150.0 | 187.5 | 187.5 |
| 66 | 200.0 | 187.5 | 200.0 |
| 132 | inf | 187.5 | inf |

$s=0$ 和 $s=132$ 是两行退化：没有通信 SM，或没有计算 SM。

```text
 200 ┤                         └─────  T_compute：跨过一个 wave 边界就抬一级
     ..........................│......  T_comm：16 个 SM 之后钉在 187.5
 150 ┤                ┌────────┘
 100 ┤────────────────┘
     └──┬──────┬──────┬──────┬──►  s_comm
        0     16     33     65
```

图里只有两条线：`T_comm` 在 16 之后是平的，`T_compute` 每跨一个 wave 边界抬一级；它越过那条虚线之后，makespan 就跟着涨。

- $s < 16$：通信瓶颈。$T_{\text{comm}} = W_m/s$，每多给一个 SM 线性下降：750 → 375 → 250 → 187.5，每一分投入都直接兑现。
- $16 \le s \le 65$：平台。通信在 $s_0=16$ 处饱和、钉死在 187.5；计算因为 SM 变少而变慢（未量化的原始值从 86.2 涨到 149.3），但一直没超过 187.5，所以 makespan 一动不动。多出来的 49 个 SM 在关键路径上是隐形的——不是「多给没关系」，是「多给的这些根本没产生收益」。
- $s \ge 66$：计算反超。此处计算原始值 151.5，被 wave 量化抬到 200，超过 187.5，makespan 变成 200。从这里开始，每多给通信一个 SM，都是拿计算时间换的。

wave 量化把计算时间切成台阶：同一级台阶内，多给几个 SM 不涨 makespan；跨过边界就翻脸。所以「多给一点没关系」只在台阶内部成立。

模型给出的最优是 $s=16$，但工程上该选平台左缘，理由不是「它最小」：同样 187.5，$s=16$ 留了 116 个 SM 给计算，$s=65$ 只剩 67 个。workload 一变（$W_c$ 变大，或 wave 边界移动），平台右缘会立刻塌到关键路径上，左缘的余量大得多。这就是「从 sweep 里选一个常数」和「选一个稳健的常数」的区别。

串行基线 263.3 是 $10000/132 + 3000/16$。注意第二项：即使 132 个 SM 全给通信，有效带宽也只按 16 个算，所以「串行」不是「通信也变快」，而是「通信还是那么慢，只是排在计算后面」。最优 187.5 对 263.3，约 1.40×。

模型有三处失真：带宽在 $s_0$ 处硬饱和（真实是渐进曲线）；假设通信与计算零干扰，忽略了[第一篇]({{< relref "comm-overlap-01-performance-model" >}})里的 $\gamma$；wave 量化只用一个 $u$ 粗粒度模拟，真实 kernel 的尾 wave 更碎。所以它**用来选区间，不用来报绝对数**。

## 6. 六篇论文，六个能对上的数

| 论文 | 它说明的事 | 一个数 | 口径 |
|---|---|---|---|
| Comet | fused kernel 里 producer/consumer 比例随 shape 变，没有常数 | 132 | grid 大小 = Hopper 的 SM 数 |
| Weave | 静态分区已最优，GEMM 中段仍有 mid bubble；单层 2.89×、端到端 1.33× | 47.1% | 4×H100、EP=4、六个 MoE 模型的 overlap ratio（Comet 14.3%、DeepEP 9.2%） |
| Analytical RM（2609.07536） | 分区可以在 launch 前解出来，不用查表 | 3.22% | 4×A100（评估用到 108 个 SM）、15 个 p90 workload 相对实测 oracle 的平均 regret，最差 10.21%；相对 Comet 1.771×（最高 2.584×） |
| T-CCL（2610.07098） | 把 collective 占的 SM 数本身压下来 | 3.5 倍 | 4×GH200 上 32 MB all-gather 的活跃 SM 数之比；128 KB 时 T-CCL 1.32、NCCL 3.30、NCCL 对称内存 1.98 |
| TokenWeave（2505.11329） | 2–8 个 SM 跑满带宽（此前要 16–20+ 个），但朴素融合更慢 | 2–8 | 8×H100 上跑满带宽所需的 SM 数 |
| Characterizing（2507.03114） | 分区模型漏掉了 $\gamma$ | 18.9% | 4×A100/H100/MI250/MI210 上 overlap 时计算平均变慢 |

Comet 的最优划分点 "is influenced by the shape of input and specific model configurations"，所以它预先 profile 一批不同划分点的 kernel 变体、把最优配置存成 metadata，运行时按配置选——本质是离线 sweep 加查表。这也解释了为什么「写死一个 `s_comm=20`」在生产里迟早出事（CTA 数与 $n_c$ 的具体扫描见[第六篇]({{< relref "comm-overlap-06-moe" >}})）。

Weave 更值钱的是它对「分区解决不了什么」的描述：即使 SM 划分已经最优，通信 SM 在 GEMM 中段仍然整段空闲，形成 mid bubble——dispatch 集中在头部、combine 集中在尾部，中间那一大段只有 GEMM 在跑。它的解法分两层：空间上用 cost model 定 $c^*$，时间上用 chunk pipeline 把 combine 摊进 GEMM 中段，再用 bubble stealing 让通信 SM 去偷 GEMM tile；cost model 选出来的配置和穷举最优平均差 8.2%。所以**分区只解决分多少，不解决什么时候用**。

解析模型那条要带两个 caveat：Qwen 的平均 regret 是 7.14%，明显高于另外两个模型（Granite 0.91%、DeepSeek-V2-Lite 1.60%），说明它对某些 routing 分布就是不准；实验只在 A100、prefill 上做，H100 的 TMA 和更大的 SM 数没有覆盖。算子级（GEMM2+GatherRS）相对 Comet 是 2.528×（最高 4.218×）。求解耗时 0.157 μs，方向是对的：分区应该在 launch 前用一个可解释的模型算出来，而不是查表——这是[第八篇]({{< relref "comm-overlap-08-compiler" >}})要展开的自动化线索。

T-CCL 在 32 MB 上只要 6.60 个 SM，NCCL 要 23.43 个。在分区模型里的含义很直接：$W_m$ 不变，但达到同样带宽所需的 $s_0$ 变小，平台左缘左移，留给计算的 SM 更多。它的 GEMM–collective overlap 案例里，把后端从 NCCL 换成 T-CCL，四卡上的算子级加速从 1.04× 提到 1.14×，原因就是通信少占了 SM。

TokenWeave 给的是负结果：朴素融合（把 AllReduce 拆成 RS + AG，再和 RMSNorm 拼在一起）在 512 / 1K / 2K / 4K / 8K token 上比串行还慢，相对性能 0.98× / 0.96× / 0.93× / 0.92× / 0.96×。原因是拆成 RS+AG 多了一轮读写，而 RMSNorm 本身是 memory-bound，和通信抢同一条 HBM 通道。真正有效的做法是让融合 kernel 只占 2–8 个 SM、把其余 SM 全留给计算——「把两个算子塞进一个 kernel」本身不等于 overlap。

Characterizing 补的是模型里被漏掉的那一项：overlap 时计算最高慢 40%，串行只比 overlap 慢 10.2%；峰值功耗高 25%，H100 平均 38% TDP、峰值 140% TDP。140% TDP 说明 overlap 不是免费并行，它会把 GPU 推到功耗墙外面然后降频，所以 $\max(T_{\text{compute}}, T_{\text{comm}})$ 只是乐观下界，真实值要把 $\gamma$ 乘进去。

## 7. 动手前先量五个数

| 指标 | ncu 里看什么 | 回答什么问题 |
|---|---|---|
| SM active | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | 计算 kernel 在 overlap 时掉了多少 |
| achieved occupancy | `smsp__warps_active.avg.pct_of_peak_sustained_active` | 和理论 occupancy 差多少，被谁占了寄存器/smem |
| warp stall | `stall_long_scoreboard` / `stall_barrier` / `stall_mio_throttle` | 等数据、等 barrier，还是 smem/TMA 队列堵 |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | 通信数据有没有把 GEMM 的 operand 挤出 L2 |
| DRAM 吞吐 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 是不是撞了 HBM 墙 |

再加一路 `nvidia-smi dmon` 或 DCGM 看功耗和时钟：overlap 时 clock 掉了，前面五个数都会变差，而你会以为是 kernel 写得不好。完整的 trace 方法留给[第十篇]({{< relref "comm-overlap-10-measurement" >}})。

四条经验：

1. **从 sweep 里选 `s_comm`，别写常量**：至少扫 0 到 $S$，步长 4；取平台左缘，不取最小值点。
2. **盯住 wave 边界**：换 shape、换 TP/EP、换 chunk 数，都会让计算的 wave 数变，原来的 $s$ 可能正好跨在台阶边上。
3. **先问「通信能不能少占 SM」，再问「该分几个 SM」**：T-CCL 的 3.5 倍、TokenWeave 的 2–8 个 SM，都是把 $s_0$ 打下来，比调分区更根本。
4. **把负结果记下来**：哪个 shape 下 overlap 反而慢、慢多少、是不是跨了 wave 边界——这比只报最好 case 有用，也是第十篇要做的事。

## 8. 如果只记一件事

**kernel 层的 overlap 是资源归属问题，不是 stream 问题**；能不能赢，看你在一段平台上有没有给计算留够 SM。

下一篇换一个角度：既然分区要随 shape 变、又要算得准，那就把「选分区」交给编译器。Syncopate 的 chunk 抽象、Lagom 的联合调参和 TileLink 的 tile 级编程，都是这条路上的尝试。

## 附：一次 TMA 握手的六步

1. producer warp 里选一个 elected thread，用 host 预先编码好的 TMA descriptor（形状、stride、swizzle）发指令；
2. 先 `mbarrier.arrive.expect_tx`，告诉 barrier 这一轮要等 N 字节；
3. 再发 `cp.async.bulk.tensor`，把 global（或 peer）内存的一块搬到 shared memory，完成通知记在 mbarrier 的 tx-count 上；
4. TMA 硬件每搬完一块，tx-count 自动递减，减到 0，barrier 的 phase 翻转；
5. consumer warpgroup 用 `mbarrier.try_wait.parity` 轮询 phase，等到就消费 smem 里的 tile，发 wgmma；
6. consumer 用完这块 buffer，arrive 一个 buffer-free barrier 还给 producer，producer 复用这个 stage。

这一串里没有一次全局同步：每个 stage 有自己的 barrier，生产者和消费者各自推进。

---

**系列导航**：[六：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}}) · [八：编译与自动调度]({{< relref "comm-overlap-08-compiler" >}})

> 配套代码：`code/comm-overlap/sm_split_model.py`
> 自测：`python3 sm_split_model.py --test`
