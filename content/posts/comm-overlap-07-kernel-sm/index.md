---
title: "通算并行研读（七）：Kernel 与 SM——persistent kernel、warp specialization 与分区"
date: 2026-11-02
draft: false
weight: 8
series: ["comm-overlap"]
tags: ["通算并行", "系列", "CUDA", "persistent kernel", "warp specialization", "SM 分区", "TMA"]
categories: ["算子开发"]
---

上一篇把 MoE 一层拆成了 dispatch、expert GEMM、combine 三段，再用 chunk 让它们错开。但拆完之后，一个更硬的问题冒出来：**这三段在 GPU 上各占几个 SM？谁来定？**

这一篇往下走一层，进到 kernel 里面。

先说结论：在 kernel 这一层，"overlap" 不再是"两条 stream 叠在一起"，而是**谁拥有哪个 SM、哪个 warp、哪个 issue slot**。stream 是软件抽象，SM 是硬件实体；两条 stream 能被同时调度，不等于两份工作能同时拿到发射机会。

一个常见的对账现场：nsys 上通信和计算叠得很漂亮，overlap ratio 90% 以上；换 ncu 看同一个 GEMM，Tensor Core 的忙碌比例掉了，warp 大量停在 `stall_long_scoreboard`，SM 的 active cycles 却没降。两个工具都没错，它们量的是不同的东西——nsys 量"时间轴上有没有重叠"，ncu 量"重叠的时候硬件资源被谁拿着"。

本篇沿着这个视角走：先把 kernel 眼里的资源列清楚，再看三种实现风格，然后钻进 warp specialization 的握手细节，最后用一个几十行的模型把"通信该拿几个 SM"算出来，并用六篇论文的数字校准它。前置阅读是[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})的 AG+GEMM 融合和[第六篇]({{< relref "comm-overlap-06-moe" >}})的 MoE 三段式。

## 1. Kernel 眼里的七种资源

Hopper 的一个 SM 大概是这样一笔资产：4 个 warp scheduler、256 KB 寄存器堆（64K 个 32-bit 寄存器）、228 KB 可配置 shared memory（单个 CTA 最多寻址 227 KB）、一个 TMA 单元，外加它到 L2/HBM 的口子。整卡还有一份功耗帽。

通信要用的和计算要用的，是同一批资产：

| 资源 | 通信怎么吃它 | 计算怎么吃它 | 冲突时的表现 |
|---|---|---|---|
| SM / CTA 槽位 | NCCL、NVSHMEM 的 copy + reduce CTA 常驻 | GEMM 的 tile CTA | 计算 wave 装不满，尾部变长 |
| warp scheduler | 通信 warp 轮询 flag、等 mbarrier | MMA warp 需要连续发射 | issue slot 被占，MMA 发射间隔变大 |
| 寄存器 | 通信 kernel 用量低，但按 CTA 整块分配 | wgmma 的 accumulator 吃满 | occupancy 下降，双 buffer 放不下 |
| shared memory | 通信的 staging buffer（TMA 目标） | GEMM 的 A/B tile + epilogue | 一个 CTA 占满 smem，别的 CTA 进不来 |
| TMA / copy 引擎 | bulk 拷贝、multicast | operand 加载 | TMA 队列排队，计算等 operand |
| L2 / HBM 带宽 | 读写对端数据、做 reduction | operand、activation 流 | L2 hit rate 掉，HBM 拥塞 |
| 功耗 / 频率 | 与计算同时忙，抬高峰值功耗 | 同上 | 撞功耗墙降频，两边一起慢 |

这张表里只有第一行是**数量**问题（分几个 SM），其余六行都是同一批 SM 内部的**份额**问题。所以"SM 分区"只是 overlap 的第一层；第二层是同一个 SM 内部，warp 之间怎么分工——这就是 warp specialization 要解决的事。

还有一个容易被忽略的点：分区的粒度不是 SM，而是 **CTA 槽位 × 每 SM 能常驻几个 CTA**。一个 SM 能不能同时放下一个通信 CTA 和一个计算 CTA，取决于两者的寄存器、shared memory 和线程数能否凑进同一个 SM。如果通信 CTA 要 100 KB smem、计算 CTA 要 200 KB，它们就永远不可能共居——所谓"分区"其实是"整 SM 归属"，没有任何共享余地。

## 2. 三种实现风格

### 2.1 双流两 kernel

最朴素的做法：NCCL 的 all-gather / all-to-all 走一个 stream，cuBLAS/CUTLASS 的 GEMM 走另一个 stream，中间用 event 连依赖。

好处是工程上最省事：不改 kernel，NCCL 的拓扑探测、channel 选择、协议都是现成的；Megatron、DeepSpeed 这些框架已经有成熟的 hook 体系。

代价有四条：

1. **启动延迟**。每次通信、每个 chunk 都要过一次 launch，微秒量级；chunk 切到几十个，launch 本身就成了固定税。
2. **依赖只能落到 kernel 边界**。GEMM 必须整个 kernel 结束，NCCL 才能拿到"数据已就绪"；做不到"第 3 个 tile 算完就发第 3 个 tile"。
3. **SM 争用不可控**。NCCL 自己决定开几个 channel、每个 channel 几个 CTA，你只能通过 `NCCL_MAX_NCHANNELS` 这类环境变量间接影响；它不会因为你的 GEMM 快没 SM 了就自动让路。
4. **两套 occupancy 配置打架**。两个 kernel 各自按自己的寄存器/smem 需求编译，没法共享 buffer，也没法让通信的 staging buffer 和 GEMM 的 operand tile 复用同一块 smem。

Comet 论文对这条路线的批评很直接：把通信和计算封进不同 stream 的不同 kernel，"restricts control over hardware resources and results in non-deterministic kernel performance"。注意后半句——不只是慢，是**不可预测**：你不知道这一轮 NCCL 拿了几个 SM，所以也不知道 GEMM 会被拖慢多少。

### 2.2 Fused kernel + CTA specialization

Comet 的做法：一个 kernel，grid 里 CTA 分两类。producer CTA 负责通信（收、发、reduce），consumer CTA 负责 GEMM。grid 大小取 SM 数（Hopper 上是 132），于是"分几个 CTA 给通信"就近似等于"分几个 SM 给通信"。

Flux 稍有不同：它**不预设** producer/consumer 的比例，整个 grid 跑同一份 GEMM，每个 CTA 在自己的 K-loop 里决定这一轮是算还是搬（见[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})）。但两者都把依赖粒度降到了 tile，也都把"谁等谁"写进了 kernel。

这个"近似"有个前提：每个 CTA 都能独占一个 SM。只有当单个 CTA 的资源占用大到一块 SM 只能放一个（比如 smem 用到 100 KB 以上、或寄存器吃满 64K），1:1 才成立；一旦两个 CTA 能挤进一个 SM，CTA 数和 SM 数就脱钩了，分区的账要按 residency 重算。

关键设计是 **tile 级信号**：consumer 算完一个 tile，就把这个 tile 的 ready 标志打上（flag 或 mbarrier arrive）；producer 看到就发，不等整个 GEMM 结束。反过来，producer 收到一个 tile 的数据就置 ready，consumer 取走继续算。全程没有 global barrier——不同 tile 处在流水线的不同阶段，谁先好谁先走。

这条路线把"依赖粒度"从 kernel 降到了 tile，代价是：kernel 变成手写、和 shape 强绑定；对 Comet 来说，producer/consumer 的比例成了新的超参；一旦某个角色先干完，它只能空转——因为同一个 kernel 里，CTA 的角色是 launch 时写死的（Flux 的自调度缓解了这一点，但换成了 tile 顺序和信号的开销）。

### 2.3 Persistent megakernel

Weave 再往前一步：整个 MoE 层（dispatch、GEMM1、GEMM2、combine 等五个操作）塞进**一个** persistent kernel，grid 仍是 132，一次 launch 跑完整层。

它多了两个能力：

- **角色可变**。每层 routing 结束后，这一层的通信量和计算量就已知了；kernel 里跑一个轻量 cost model，算出该层的通信 SM 数 $c^*$ 和 chunk 数 $K^*$，然后按新分区干活。分区从"每个模型一次"变成"每层每卡一次"。
- **工作可偷**。通信 SM 干完自己的活，如果 GEMM 还有 tile 没人算，它就去偷一个来算（bubble stealing）。"通信 SM 在 GEMM 中段空闲"的那段气泡就这样被填上了。

代价也很清楚：整层重写、和特定模型结构绑定、cost model 得在 device 上跑（Weave 测到 0.54 μs，占 MoE 层时间不到 0.021%，可忽略）。这不是能塞进现有框架的改动，而是一个新 runtime。

三条路线放在一条时间轴上，差别一目了然：

```text
时间 ────────────────────────────────────────────────────────►

(a) 双流两 kernel
  stream A   [ AG chunk1 ][ AG chunk2 ]              [ AG chunk3 ]
  stream B               [ GEMM chunk1 ][ GEMM chunk2 ]
              ↑ 每个 chunk 一次 launch，依赖只能落在 kernel 边界

(b) CTA specialization（Comet / Flux）
  producer   ████ AG ████        ████ AG ████        ████ AG ████
  consumer        ████ GEMM ████     ████ GEMM ████
              ↑ per-tile flag / mbarrier，无 global barrier

(c) persistent megakernel（Weave）
  comm SM    ██ dispatch ██ ····· 偷 GEMM tile ····· ██ combine ██
  comp SM        ██████ GEMM1 ██████ ██████ GEMM2 ██████
              ↑ 角色每层重算，中段气泡被 bubble stealing 填掉
```

三条路线的分别，不是"谁 overlap 得更好"，而是**把决策点放在了哪里**：launch 前（双流，driver 决定）、launch 时（Comet，程序员决定）、运行时（Weave，cost model 决定）。

## 3. Warp specialization：一个 warp 为什么不能又发 TMA 又喂 Tensor Core

### 3.1 四个理由

既然通信只需要搬数据，为什么不让同一个 warp 顺手发一下？因为"顺手"在 GPU 上很贵：

1. **发射是串行的**。TMA 发起本身只要一条指令（`cp.async.bulk.tensor`），但"发起 → 等 mbarrier → 消费"是一条长延迟依赖链。同一个 warp 在这条链上等的时候，warp scheduler 不会让它发 wgmma——它已经不是 eligible warp 了。
2. **wgmma 要连续喂**。Hopper 的 wgmma 是 warpgroup 级（128 线程）的异步 MMA，accumulator 在寄存器里。Tensor Core 的吞吐靠"连续发射"维持；这 128 个线程如果还要分神算地址、发拷贝、等 barrier，MMA 的发射就会出现空隙。
3. **寄存器预算翻倍**。producer 要描述符、地址、循环变量；consumer 要 accumulator 和 operand fragment。一个 warp 兼两职，寄存器按两者的最大值分配，occupancy 就下来了——occupancy 一掉，双 buffer 都放不下。
4. **wait 是 warp 粒度的**。只要这个 warp 还在等 barrier 的 phase，它就不会去发下一条 MMA。把 wait 放在专门的 producer warp 上，consumer warp 就只在自己真正缺 tile 时才等，不会被"下一块还没搬完"拖住。

所以标准做法是**角色分离**：producer warp 只负责搬，consumer warpgroup 只负责算，中间用 mbarrier 握手。这不是为了架构好看，是为了让 Tensor Core 的 issue 流不断。

### 3.2 握手长什么样

Hopper 的 TMA 路径大致是：

1. producer warp 里选一个 elected thread，用 host 预先编码好的 TMA descriptor（形状、stride、swizzle）发指令；
2. 先 `mbarrier.arrive.expect_tx`，告诉 barrier "这一轮要等 N 字节"；
3. 再发 `cp.async.bulk.tensor`，把 global（或 peer）内存的一块搬到 shared memory，完成通知记在 mbarrier 的 tx-count 上；
4. TMA 硬件每搬完一块，tx-count 自动递减；减到 0，barrier 的 phase 翻转；
5. consumer warpgroup 用 `mbarrier.try_wait.parity` 轮询 phase，等到就消费 smem 里的 tile，发 wgmma；
6. consumer 用完这块 buffer，arrive 一个 "buffer free" barrier 还给 producer，producer 复用这个 stage。

```text
producer warp（1 个 elected thread）              consumer warpgroup（128 线程）
        │                                                    │
        │ ① mbarrier.arrive.expect_tx(bytes)                  │
        │ ② cp.async.bulk.tensor ──┐                          │
        │                          │  TMA 硬件搬运            │
        │                          ▼                          │
        │                   [ HBM → SMEM ]                    │
        │                          │  tx-count 递减           │
        │                          ▼                          │
        │                   phase 翻转 ───────────────────────►│ ③ try_wait(parity)
        │                                                    │ ④ wgmma(acc, smem)
        │◄────────────────── ⑤ buffer-free arrive ────────────┤
        │ ⑥ 复用这个 stage                                    │
```

这套机制里没有一次全局同步：每个 stage 有自己的 barrier，生产者和消费者各自推进。**tile 级 overlap 能成立，靠的就是"同步粒度 = buffer 粒度"，而不是"层粒度"。**

对比 Ampere 的 `cp.async`：那是 per-thread 的异步拷贝，每个线程自己发、自己 `cp.async.wait_group`；Hopper 的 TMA 是**单线程发起、硬件搬运**，SM 这边几乎不占线程。这个差别直接决定了通信的 SM 占用：用 `cp.async` 搬 1 MB，要一堆线程算地址；用 TMA 搬 1 MB，只要一个线程发一条指令。

T-CCL 就是把这个性质用到底：整个 collective 拆成一串 TMA load / store / reduce，单线程发起。它测到 128 KB 的 all-gather 平均只用 1.32 个活跃 SM，而 NCCL 要 3.30 个——同一份通信，SM 占用差 2.5 倍。

## 4. SM 分区：一个能算的模型

### 4.1 模型

`code/comm-overlap/sm_split_model.py` 把一层看成两份工作：

- 计算工作量 $W_c$（单位 SM·μs），只用分到的 $S-s$ 个 SM；
- 通信工作量 $W_m$（单位 SM·μs），只用分到的 $s$ 个 SM，且有效带宽在 $s_0$ 个 SM 后饱和。

$$
T_{\text{compute}} = \frac{W_c}{S-s}, \qquad
T_{\text{comm}} = \frac{W_m}{\min(s, s_0)}, \qquad
T = \max\left(T_{\text{compute}}, T_{\text{comm}}\right)
$$

真实 GPU 上计算是按 wave 走的，所以再套一层量化：$T_{\text{compute}} = \left\lceil \frac{W_c/(S-s)}{u} \right\rceil \cdot u$，$u$ 是一个 wave 的时间（代码里 `wave_unit=50`）。

参数取 $S=132$、$W_c=10000$、$W_m=3000$、$s_0=16$、$u=50$。

### 4.2 输出

```text
$ python3 sm_split_model.py
S=132 Wc=10000.0 Wm=3000.0 sat=16 wave=50.0
 s_comm    T_comp    T_comm         T
      0     100.0       inf       inf
      4     100.0     750.0     750.0
      8     100.0     375.0     375.0
     12     100.0     250.0     250.0
     16     100.0     187.5     187.5
     20     100.0     187.5     187.5
     24     100.0     187.5     187.5
   ...  （每 4 行打印一次，中间略）
    124    1250.0     187.5    1250.0
    128    2500.0     187.5    2500.0
    132       inf     187.5       inf
best: s_comm=16 makespan=187.5 (serial=263.3)
```

关键点单独列出来：

| s_comm | T_compute | T_comm | T | 区间 |
|---|---|---|---|---|
| 0 | 100.0 | inf | inf | 退化：没有通信 SM |
| 4 | 100.0 | 750.0 | 750.0 | 通信瓶颈 |
| 8 | 100.0 | 375.0 | 375.0 | 通信瓶颈 |
| 12 | 100.0 | 250.0 | 250.0 | 通信瓶颈 |
| 16 | 100.0 | 187.5 | 187.5 | 饱和点，最优 |
| 20 | 100.0 | 187.5 | 187.5 | 平台 |
| 32 | 100.0 | 187.5 | 187.5 | 平台（计算原始值刚好 100.0） |
| 33 | 150.0 | 187.5 | 187.5 | 计算掉一个 wave |
| 36 | 150.0 | 187.5 | 187.5 | 平台 |
| 65 | 150.0 | 187.5 | 187.5 | 平台右缘 |
| 66 | 200.0 | 187.5 | 200.0 | 计算反超，makespan 抬高 |
| 132 | inf | 187.5 | inf | 退化：没有计算 SM |

### 4.3 三个区域，一条平台，两道台阶

1. **$s < 16$：通信是瓶颈。** $T_{\text{comm}} = W_m/s$，每多给一个 SM，通信时间线性下降：750 → 375 → 250 → 187.5。这一段每一分投入都直接兑现。
2. **$16 \le s \le 65$：平台。** 通信在 $s_0=16$ 处饱和，$T_{\text{comm}}$ 钉死在 187.5；计算虽然因为 SM 变少而变慢（未量化的原始值从 86.2 涨到 149.3），但一直没超过 187.5，所以 makespan 一动不动。多出来的 49 个 SM 在关键路径上是**隐形**的——不是"多给没关系"，而是"多给的这些根本没产生收益"。
3. **$s \ge 66$：计算反超。** $s=66$ 时计算原始值 151.5，被 wave 量化抬到 200，超过 187.5，makespan 变成 200。从这里开始，每多给通信一个 SM，都是拿计算时间换的。

wave 量化在这里制造了两道台阶：$s$ 从 32 加到 33，计算从 100 跳到 150（多了一个 wave）；从 65 加到 66，从 150 跳到 200。**计算时间不是随 $s$ 平滑上升，而是台阶式上升**——"多给一点没关系"这种直觉只在台阶内部成立，跨过台阶就翻脸。

串行基线 263.3 是 $10000/132 + 3000/16$。注意第二项：即使 132 个 SM 全给通信，有效带宽也只按 16 个算，所以"串行"不是"通信也变快"，而是"通信还是那么慢，只是排在计算后面"。最优 187.5 对 263.3，约 1.40×。

模型给出的最优是 $s=16$，但**平台左缘才是工程上该选的**：同样 187.5，$s=16$ 留了 116 个 SM 给计算，$s=65$ 只剩 67 个。workload 一变（$W_c$ 变大，或 wave 边界移动），平台右缘会立刻塌到关键路径上；左缘的余量大得多。这就是"从 sweep 里选一个常数"和"选一个稳健的常数"的区别。

模型有三处失真也要说清楚：(1) 带宽在 $s_0$ 处硬饱和，真实是渐进曲线；(2) 假设通信与计算零干扰，忽略了第一篇里的 $\gamma$；(3) wave 量化只用一个 $u$ 粗粒度模拟，真实 kernel 的尾 wave 更碎。**这个模型用来选区间，不用来报绝对数。**

## 5. 论文里的证据

### 5.1 Comet：132 个 CTA，划分点随 shape 走

Comet 的 fused kernel 总 CTA 数等于 Hopper 的 SM 数（132），其中 $n_p$ 个做 producer、$n_c$ 个做 consumer。论文的原话是最优划分点 "is influenced by the shape of input and specific model configurations"。

具体数字（论文 Figure 8）：TP=8 时，输入长度 $M$ 从 4096 加到 16384，最优 $n_c$ 从 18 变到 26；把 TP 从 8 换成 4（$M=16384$），$n_c$ 又从 26 跳到 46。

也就是说：**同一个模型，换个序列长度、换个并行策略，最优分区就变了，而且幅度不小（26→46 接近翻倍）。** Comet 的应对是预先 profile 一批不同划分点的 kernel 变体、把最优配置存成 metadata，运行时按配置选——本质是"离线 sweep + 查表"。

这也解释了为什么"写死一个 `s_comm=20`"在生产里迟早出事。

### 5.2 Weave：静态最优之后，还有 mid bubble

Weave 在 4×H100、EP=4 上测六个 MoE 模型，overlap ratio 47.1%（Comet 14.3%、DeepEP 9.2%），单层 2.89×、端到端 1.33×。

但更有价值的是它对"分区解决不了什么"的描述：**即使 SM 划分已经最优，通信 SM 在 GEMM 中段仍然整段空闲**，形成一个 mid bubble。原因是依赖结构——dispatch 集中在头部、combine 集中在尾部，中间那一大段只有 GEMM 在跑。

Weave 的解法分两层：空间上用 cost model 定 $c^*$，时间上用 chunk pipeline 把 combine 摊进 GEMM 中段，再用 bubble stealing 让通信 SM 去偷 GEMM tile。它的 cost model 选出来的配置和穷举最优相比，平均差 8.2%。

**这条证据的意义：SM 分区只解决"分多少"，不解决"什么时候用"。** 只做静态分区的系统，天花板就在这里。

### 5.3 Analytical RM：把 wave 量化写进解析式

arXiv 2609.07536 做的事和 `sm_split_model.py` 很像，但严谨得多：它把"CTA 常驻"当约束，用 routed tile 数、kernel occupancy、GPU residency、split 级就绪依赖，在 launch 前直接解出通信 CTA 数 $C$ 和对应的资源预留 $R(C)$。

结果（4×A100，评估用到 108 个 SM）：$C$ 的扫描范围 1–16，15 个真实 p90 workload 上，相对实测 oracle 的平均 regret 3.22%、最大 10.21%；分模型看，Granite 0.91%、Qwen 7.14%、DeepSeek-V2-Lite 1.60%。相对 COMET，完整 post-router MoE 层几何平均 1.771×（最高 2.584×），GEMM2+GatherRS 算子 2.528×（最高 4.218×）。求解耗时 0.157 μs。

两个 caveat 必须带上：**Qwen 的平均 regret 是 7.14%，明显高于另外两个模型**，说明解析模型对某些 routing 分布就是不准；而且实验只在 **A100、prefill** 上做，H100 的 TMA 和更大 SM 数没有覆盖。

即便如此，它的方向是对的：**分区应该在 launch 前用一个可解释的模型算出来，而不是查表。** 这和 5.1 的"离线 profile"是两种哲学，也是[第八篇]({{< relref "comm-overlap-08-compiler" >}})要展开的自动化线索。

### 5.4 T-CCL：先把 collective 的 SM 占用压下来

前面所有讨论都默认"通信要占 $s$ 个 SM"。T-CCL（2610.07098）提醒：这个 $s$ 本身是可以改的。

它把 collective 的数据搬运和 reduction 都卸载到 TMA 上，用单线程发起一串异步 TMA 操作。在 4×GH200（TP=4）上用 nsys 统计 1000 次的平均活跃 SM 数：

| AllGather | T-CCL | NCCL | NCCL Symmetric |
|---|---|---|---|
| 128 KB | 1.32 | 3.30 | 1.98 |
| 32 MB | 6.60 | 23.43 | 7.92 |

32 MB 上 NCCL 要 23.43 个 SM，T-CCL 只要 6.60——**同样一份通信，SM 占用差 3.5 倍**。NCCL 的对称内存内核对小消息很有效（128 KB 上 1.98，比 NCCL 的 3.30 少四成），但到 128 MB 又回到 9.24，仍然比 T-CCL 的 6.60 多。这在分区模型里的含义很直接：$W_m$ 不变，但达到同样带宽所需的 $s_0$ 变小了，平台左缘左移，留给计算的 SM 更多。T-CCL 的 GEMM–collective overlap 案例里，把后端从 NCCL 换成 T-CCL，四卡上的算子级加速从 1.04× 提到 1.14×，原因就是"通信少占 SM"。

### 5.5 TokenWeave：2–8 个 SM 就够，但简单融合会更慢

TokenWeave（2505.11329）把 AllReduce 和 RMSNorm 融成一个 kernel，靠 NVSHARP/Multimem，在 8×H100 上只用 2–8 个 SM；此前的工作做同样的通信要 16–20+ 个 SM。这是把 $s_0$ 打到极低的另一个例子——只占 H100 的 1.5%–6%，却能跑满带宽。

但它同时给了一个负结果：**朴素融合（把 AllReduce 拆成 RS + AG，再和 RMSNorm 拼在一起）在 512–8K token 区间比串行还慢**，相对性能 0.98× / 0.96× / 0.93× / 0.92× / 0.96×（对应 512 / 1K / 2K / 4K / 8K）。原因是拆成 RS+AG 多了一轮读写，而 RMSNorm 本身是 memory-bound，和通信抢同一条 HBM 通道。真正有效的做法是让融合 kernel 只占 2–8 个 SM、把其余 SM 全留给计算，而不是"把两个算子塞进一个 kernel"就完事。

### 5.6 Characterizing：分区模型里那个被忽略的 γ

第一篇里的 overlap tax，在 kernel 层就是"同一批 SM 里，通信和计算互相拖慢"。Characterizing（2507.03114）在 4×A100/H100/MI250/MI210 上量到：overlap 时计算平均慢 18.9%、最高 40%；串行只比 overlap 慢 10.2%；峰值功耗高 25%，H100 平均 38% TDP、峰值 140% TDP。

140% TDP 这个数说明：overlap 不是"免费并行"，它会把 GPU 推到功耗墙外面，然后降频。**所以分区模型里的 $\max(T_{\text{compute}}, T_{\text{comm}})$ 只是一个乐观下界，真实值要把 $\gamma$ 乘进去。**

## 6. Tile 级信号：FlashOverlap 怎么做"不打断 GEMM 的通信"

前面两种方案都要动 kernel。FlashOverlap（2504.19519）给了一条低侵入路线：**GEMM 还是原来的 GEMM，只在一个 wave 的 tile 算完时打一个 signal；通信线程收到 signal 就去调 NCCL。**

三个部件：

1. **signal 时机**。不是每个 tile 一发，而是按 wave 分组（论文里分成 G1/G2/G3 几组）。原因很硬：4×RTX 4090 上，一个 192 KB 的 tile 单独做 AllReduce，只能跑到 13% 的带宽利用率——消息太碎，固定开销吃掉一切。
2. **pre-communication reorder**。GEMM 的 block swizzling 让同一个 wave 完成的 tile 在地址上不连续，直接交给 NCCL 会变成很多小消息。所以发之前先把这一组 tile 重排成连续地址，NCCL 就能当一个大消息发。这一步融进 GEMM 的 epilogue，代价只有 0.07% / 0.35%（A800 / 4090）。
3. **post-communication reorder**。通信完成后要把顺序改回去。这一步加在 RMSNorm 上，代价是 7.46% / 7.93% / 8.45%（tile / subtile / subtoken，A800），4090 上是 8.80% / 8.78% / 9.63%——**大约 7–10%**。

整体收益 1.02–1.65×。它的价值不在数字，而在于示范了一条"低侵入"路线：不改 GEMM 的数学、不重写 collective，只在"什么时候发"上做文章。代价是 reorder 的带宽税，以及一个事实：**signal 太细会被消息碎片反噬，太粗就退化成 kernel 边界。**

## 7. 实践清单

### 7.1 先量五个数

| 指标 | ncu 里看什么 | 回答什么问题 |
|---|---|---|
| SM active | `sm__cycles_active.avg.pct_of_peak_sustained_elapsed` | 计算 kernel 在 overlap 时掉了多少 |
| achieved occupancy | `smsp__warps_active.avg.pct_of_peak_sustained_active` | 和理论 occupancy 差多少，被谁占了寄存器/smem |
| warp stall | Warp State Statistics 里的 `stall_long_scoreboard` / `stall_barrier` / `stall_mio_throttle` | 等数据、等 barrier，还是 smem/TMA 队列堵 |
| L2 hit rate | `lts__t_sector_hit_rate.pct` | 通信数据有没有把 GEMM 的 operand 挤出 L2 |
| DRAM 吞吐 | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 是不是撞了 HBM 墙 |

再加一路 `nvidia-smi dmon` 或 DCGM 看功耗和时钟：如果 overlap 时 clock 掉了，前面五个数都会变差，而你会以为是 kernel 写得不好。完整的 trace 方法留给[第十篇]({{< relref "comm-overlap-10-measurement" >}})。

### 7.2 四条经验

1. **`s_comm` 从 sweep 选，不写常量。** 至少扫 0 到 $S$，步长 4；取平台左缘，不取最小值点。
2. **盯 wave 边界。** 换 shape、换 TP/EP、换 chunk 数，都会让计算的 wave 数变，原来的 $s$ 可能正好跨在台阶边上。5.1 的 26→46 就是这个现象。
3. **先问"通信能不能少占 SM"，再问"该分几个 SM"。** T-CCL 的 3.5 倍、TokenWeave 的 2–8 个 SM，都是把 $s_0$ 打下来，比调分区更根本。
4. **把负结果记下来。** 哪个 shape 下 overlap 反而慢、慢多少、是不是跨了 wave 边界——这比只报最好 case 有用，也是第十篇要做的事。

## 8. 结论

- kernel 层的 overlap 是**资源归属**问题，不是 stream 问题：七种资源里只有 SM 数量能"分"，其余都是同一批 SM 内部的份额。
- 三种风格的分野在**决策点**：双流（driver 决定）、CTA specialization（launch 时决定）、persistent megakernel（运行时决定）。
- warp specialization 的本质是**保护 issue 流**：把长延迟的搬运和等待从算 MMA 的 warpgroup 上挪走。
- SM 分区的最优点通常是一个**平台**，不是一个点；wave 边界让计算时间台阶式跳，所以选平台左缘、并随 shape 重扫。
- 分区之外还有三件事同样重要：把 collective 的 SM 占用压下来（T-CCL）、把气泡从时间维度填掉（Weave 的 bubble stealing）、把 $\gamma$ 算进去（Characterizing）。

往回看，[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})讲的是"把 all-gather 融进 GEMM"，那篇里的 tile 级 overlap，在 kernel 里就是本篇的 producer/consumer；[第六篇]({{< relref "comm-overlap-06-moe" >}})讲三段式，本篇讲这三段在 SM 上怎么分家。下一篇换一个角度：既然分区要随 shape 变、又要算得准，那就把"选分区"交给编译器——Syncopate 的 chunk 抽象、Lagom 的联合调参和 TileLink 的 tile 级编程，都是这条路上的尝试。

---

**系列导航**：[六：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}}) · [八：编译与自动调度]({{< relref "comm-overlap-08-compiler" >}})

> 配套代码：`code/comm-overlap/sm_split_model.py`
> 自测：`python3 sm_split_model.py --test`
