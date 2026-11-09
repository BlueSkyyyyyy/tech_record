---
title: "通算并行研读（五）：张量并行——从 all-reduce 到 AG+GEMM 融合"
date: 2026-11-02
draft: false
weight: 6
series: ["comm-overlap"]
tags: ["通算并行", "系列", "张量并行", "Flux", "T-CCL", "TileLink", "kernel fusion"]
categories: ["训练框架"]
---

数据并行的通信在反向末尾，流水线的通信在 stage 边界。这两处的通信都能问一句"能不能挪到旁边去"。gradient bucket 可以异步发出去，策略只是"晚一点再 wait"；PP 的 send/recv 可以塞进另一个 microbatch 的计算窗口，因为它等的是**另一份数据**。

张量并行没有这个空间。

TP 的通信长在层里面。经典形态是在第二个（row-parallel）GEMM 的输出上做一次 all-reduce；换成 sequence parallel（序列并行，SP）之后，它变成两个 GEMM 外侧的一次 AG 和一次 RS。位置挪了，性质没挪——**它仍然在这一层内部，仍然卡在关键路径上**。它不是"算完一批活顺手发出去"，而是下一段计算的输入依赖：输出 $Z$ 的每一个元素都需要这次通信的**全部**元素。层内也没有第二件活可以拿来填，因为下一个算子本身就在等这份数据。

所以 TP 的 overlap 只有一条路：**把 GEMM 切开，让"算已经到的部分"和"传还没到的部分"同时发生。** 这一篇讲这条路上的四个代表：Flux（融进 kernel）、T-CCL（把 collective 压到 1–6 个 SM）、TileLink（编译器生成）、CommFuse（算法级拆成 P2P）。每个方案后面都跟着它自己的负结果。

## 1. TP 的数据流：两个 GEMM、四次通信

### 1.1 Megatron 式 MLP

固定记号。一层 MLP：

$$
Y = \mathrm{GeLU}(X A), \qquad Z = Y B
$$

权重 $A$ 按列切（column-parallel），$B$ 按行切（row-parallel），切分度就是 TP 度 $N$。

- **第一个 GEMM**：$A = [A_0, \dots, A_{N-1}]$，第 $r$ 张卡算 $Y_r = X A_r$，输出只有列方向上的一段。
- **第二个 GEMM**：$B = [B_0; \dots; B_{N-1}]$，第 $r$ 张卡算 $Z_r = Y B_r$，这**不是**完整的 $Z$，只是 $N$ 个部分和中的一个。

不做 sequence parallel 时，一个 MLP 块的前向只有一次 all-reduce：row-parallel GEMM（第二个 GEMM）之后，把 $N$ 个部分和加起来；反向有两次。配上 sequence parallel 之后，通信被搬到两个 GEMM 的外侧，变成"前面 AG、后面 RS"，反向再对称来一遍，一共四次。这就是大家熟悉的 **AG + GEMM，GEMM + RS**。

AG 补的是"被切掉的那一维"。Megatron SP 切的是序列维，AG 送来一段段 M tile；也有些实现在 K 维上切（上一层按 K 分片），AG 送来的就是 K tile。两种形态都成立，融合的道理相同，区别只在切哪一维。下文主要沿 M 维推导，K 维放在 §1.2 末尾作 remark。

**一次也好、四次也好，通信全都落在层内。** 这就是 TP 和 DP/PP 的结构性差别：DP 的通信等的是"整个反向算完"，PP 的通信等的是"另一个 microbatch"，而 TP 的下一次通信等的是"这一次通信自己"。

attention 的 TP 形态略有不同，但结论一样：QKV 投影按 head 切（column-parallel），output projection 在 head 维上求和（row-parallel）。所以"两次通信都长在层内"不变。区别在于比例——短序列、小 batch（decode，$M=1$）时 attention 的 GEMM 变瘦，通信量却不减，$g/c$ 很容易超过 1；这也是 TP 在 decode 阶段格外难受的原因之一。

### 1.2 为什么 AG 可以和 GEMM 融

这一节回答一个问题：AG 还没做完，GEMM 为什么能先算？答案不在通信库，而在 GEMM 自己的数据依赖上。

Megatron SP 的 AG 补的是**序列维（第一维）**：每张卡持有一段 M tile，AG 把别的卡那几段送过来。沿 M 维看，第一个 GEMM 的输出和输入逐行对应（记 $G = X A$）：

$$
G = X A, \qquad G[m,:] = X[m,:] A
$$

第 $j$ 段序列一到，各卡就能算出这几行的本地列；GeLU 逐元素，$Y = \mathrm{GeLU}(G)$，第二个 GEMM 的每个输出行也只由 $Y$ 的对应行决定。于是这一段的 $Z$ 部分和可以接着算，算完立刻参与 reduce-scatter。

```text
      rank 0          rank 1          rank 2
    ┌────────┐      ┌────────┐      ┌────────┐
    │  X0..  │      │  X1..  │      │  X2..  │   ← 各持一段 M tile（序列切片）
    └───┬────┘      └───┬────┘      └───┬────┘
        │               │               │
        │   tile 级 all-gather（逐段到）
        ▼               ▼               ▼
    ┌──────────────────────────────────────────┐
    │  G[m] = X[m] * A                         │   ← 每个 M tile 到了，各卡算这几行
    └──────────────────────────────────────────┘
        ▲               ▲               ▲
        │               │               └── t2 到了，算 t2 的输出行
        │               └────────────────── t1 到了，算 t1 的输出行
        └────────────────────────────────── t0 在本地，先算
```

看最后三行箭头：本地那段先算，远端两段谁先到谁先算，三段之间不需要互相等。这不是把通信藏起来，而是把"等整份数据"换成"等下一段数据"。

反过来，第二个 GEMM 的 **输出 tile** 一旦算完，也可以立刻参与 reduce-scatter：这一块的部分和已经不需要再改了。所以 RS 能融进 GEMM 的 epilogue（收尾阶段）——写出的时候直接写到"该去的地方"，顺便把 reduction 做了。

> 一句话：AG 融在"被切的那一维"（M 或 K），RS 融在输出方向。两个方向都可切，所以 TP 的两类通信都有融合点。

**另一种形态：沿 K 维切分时。** 有些实现在 K 维上切（上一层按 K 分片），AG 送来的就是 K tile。这时关键在于 **GEMM 的 K 维本身就是一个求和循环**：

$$
G = X A = \sum_{j=0}^{N-1} X_j A_j
$$

$X_j$ 是第 $j$ 张卡持有的那段 K 切片，$A_j$ 是权重对应的行块。第 $j$ 块到达之前，K 循环里属于它的那几次累加不能做；**其它块完全不受影响**。依赖的粒度从"整张 $X$"变成了"一个 K tile"。这不是 Megatron SP 在 MLP 上的形态（它的 AG 补的是 M 维），但融合的道理相同，区别只在切哪一维。

看这张图：每段 K tile 到了就累加，没到就跳过；三段的加法互不阻塞。

```text
   rank 0        rank 1        rank 2
   ┌──────┐      ┌──────┐      ┌──────┐
   │  X0  │      │  X1  │      │  X2  │   ← 各持一段 K 切片
   └──┬───┘      └──┬───┘      └──┬───┘
      │             │             │
      │   tile 级 all-gather（逐个到）
      ▼             ▼             ▼
   ┌──────────────────────────────────┐
   │  G = X0·A0 + X1·A1 + X2·A2       │   K 循环按 tile 累加
   └──────────────────────────────────┘
      ▲             ▲             ▲
      │             │             └── t2 到了，累加 t2
      │             └──────────────── t1 到了，累加 t1
      └────────────────────────────── t0 在本地，先算
```

所以 K 维形态不是另一套机制，而是同一套依赖分析换了一个切分方向：AG 补哪一维，就按哪一维的 tile 推进 GEMM。

### 1.3 为什么"按设备数切 chunk"只能拿一半

TransformerEngine 的 UserBuffers 路线是把 GEMM 和 collective 都切成 $N$ 份（或 $2N$ 份以利用双向链路），然后靠 stream/event 让第 $i$ 个 chunk 的通信和第 $i+1$ 个 chunk 的计算并行。这在 TPU 上很好用，在 GPU 上有三个硬伤，Flux 论文把这三点讲得很清楚：

1. **时间不可控**：多 stream + event 的执行顺序可以排，但精确的执行时机排不了；生产环境里 stream 一多就不确定。
2. **GEMM 被切碎**：一个大 GEMM 变成 $N$ 个小 GEMM，每个都填不满 SM，TP 度越大越明显。
3. **wait 是 kernel 级的**：NCCL 的 all-reduce 吃的是整块 buffer，产它的 GEMM 只要还有一个 tile 没写完，collective 就发不出去。

这就是"中粒度"（medium-grained）和"细粒度"（fine-grained）的分界：切的对象不是算子，而是 **tile**。

## 2. Flux：把通信和 wait 塞进同一个 GEMM kernel

Flux（[2406.06858](https://arxiv.org/abs/2406.06858)）的做法可以概括成一句话：**overdecompose 到 tile，再把有依赖的通信/wait 融进那个 GEMM kernel，最后只 launch 一个 kernel。**

### 2.1 机制

- **粒度**：一个 tile 对应一个 thread block（论文脚注说，视 GEMM 实现也可以是 warp 或 thread block cluster）。tile 的划分**与 TP 度解耦**——不必等于设备数，也不必是它的两倍，因此 K 循环和 tile 形状都可以按 GEMM 自己的最优来选。
- **融合方式**：AG+GEMM 把通信放在 prologue（tile 到了就往下算），GEMM+RS 把 reduce/scatter 放在 epilogue。wait 逻辑不是另起一个 kernel 等，而是长在 tile 的消费者那一侧——**哪个 tile 等哪个信号，是编译期/模板期就定好的**。
- **tile coordinate swizzling**：把 thread block index 到 tile 坐标的映射，和"信号到达顺序"对齐。通信顺序在 host 侧按拓扑选（节点内 NVLink 用从 `local_rank+1` 开始的环，PCIe 也用环），两者一起调，目的是让尽量少的 thread block 卡在等上。论文的消融显示，开了 swizzling 总是更好，且矩阵越大差距越大。
- **数据搬运**：节点内走 P2P 的直接 load/store，或 `cp.async.bulk` / TMA；跨节点走 NVSHMEM 的 put/get。**没有额外的通信 kernel**。
- **不静态分 SM**：这一点常被误解。Flux 里没有"拿 $k$ 个 CTA 专门做通信"的概念，整个 grid 跑的是同一份 GEMM，每个 thread block 自己决定这一轮是算还是搬。它用的是"融"和"切"，不是"让"。

这里还有一个观测上的坑：融合之后，nsys 上**看不到独立的通信 kernel** 了。如果拿"有没有 NCCL kernel"来判断 overlap 有没有生效，会得到假阴性。通信退化成了 GEMM kernel 内部的 load/store 和 barrier wait，只能从 SM 活跃度、HBM 流量和 GEMM 自身的有效算力上反推。测量方法在第十篇（实测手册）里展开。

### 2.2 数字（都带硬件条件）

GEMM + collective 的算子级对比，基线是 cuBLAS + NCCL，对手是 TransformerEngine：

| 集群 | 相对 TransformerEngine 的加速 | overlap efficiency（重叠效率） |
|---|---|---|
| A100 PCIe（8 卡/节点，PCIe 节点内） | 1.20×–3.25× | 41%–57% |
| A100 NVLink（8 卡/节点，NVLink 节点内） | 1.01×–1.33× | 36%–**96%** |
| H800 NVLink（8 卡/节点，NVLink 节点内） | 1.10×–1.51× | 37%–93% |

看第 3 列：同一套融合 kernel，在 PCIe 上 overlap efficiency 只有 41%–57%，到 NVLink 能到 93%–96%。链路越快，融合能藏住的份额越高。

端到端：GPT-3 175B 与 Llama-2 70B，128 GPU（2 路 DP × 8 路 PP × 8 路 TP），对比 Megatron-LM 与 vLLM：

- 训练最高 **1.24×**、prefill 最高 **1.66×**、decode 最高 **1.30×**；
- 这三个数是**跨 A100 PCIe / A100 NVLink / H800 NVLink 三个集群取的最大值**，不是单集群结论。分集群看：A100 PCIe 是 1.24× / 1.46× / 1.28×，A100 NVLink 是 1.05× / 1.45× / 1.30×，H800 NVLink 是 1.10× / 1.66× / **decode 没有加速**。

### 2.3 负结果

- **H800 NVLink 上 decode 没有加速。** 论文的措辞是 "no decoding speedups"，同一段里还提到 decode 有 5 个配置比非重叠的 vLLM 基线更慢。
- **小 m 会翻车。** 在 H800 NVLink 上，$m \le 1024$ 时相对 TransformerEngine 的加速范围是 **0.95×（变慢）到 1.03×**；$m=64$ 的 ReduceScatter 是唯一一个比 TransformerEngine 还慢的点，原因是 8 路 TP 之后沿 m 维的 TMA store size 只剩 8，TMA 效率被打掉。
- 论文自己也解释了机制：m 极小时 GEMM 的 warp 数很少，latency hiding 本来就弱。
- 还有一条更根本的：**通信占比低的时候，overlap 做得再好也没用。** A100 NVLink 训练时通信只占 8%–11%，即使平均 overlap efficiency 有 63%，端到端也只有 1.04×–1.05×。

### 2.4 一个必须记住的定义

Flux §2.3 定义了 overlap efficiency，并顺手把负值的情况说破：

> An overlapping method with a negative overlap efficiency implies that the method runs slower than the non-overlapping baseline.

（一个 overlap efficiency 为负的方法，意味着它比不重叠的基线跑得还慢。）

前面几篇反复出现的 $\eta = 1 - T_{\text{exposed}} / T_{\text{comm}}$ 之所以要允许负值，就是因为"重叠"本身有代价：切碎、抢带宽、多一次同步，都可能让总时间超过老老实实串行。**看到负值不要当成测量错误，那是结论。**

## 3. T-CCL：换一个方向，把 collective 压到 1–6 个 SM

Flux 的思路是"把通信塞进计算 kernel"。T-CCL（[2610.07098](https://arxiv.org/abs/2610.07098)）走的是另一条：**不融合，但让通信几乎不占资源**，这样计算可以独享剩下的 SM。

### 3.1 机制

- **用 TMA 同时卸载"搬"和"reduce"**。TMA 是 Hopper 起就有的异步搬运单元，一次 bulk transfer 由 thread block 里**一个**被选出的线程发起，之后走专用硬件。更关键的是它支持异步的 element-wise reduction（add/min/max），所以 all-reduce 的归约也可以交给 TMA，而不是让一堆线程去做加法。
- **collective 变成一个任务图**：每个 collective 被拆成 shard-in 和 shard-out 两个阶段——拥有某个 shard 的 rank 先把别人的贡献收进来在本地归约（shard-in），再把结果推给别人（shard-out）。这两个阶段再 lowering 成一组 TMA 的 load/store/reduce 任务，按流水线发射。
- **单线程发起**：一个 CTA 里一个线程就能把一整串 TMA 指令发出去，SM 侧只剩"发指令 + 等 barrier"的成本。
- **1–6 个 SM 就够**：论文 profile 的活跃 SM 数，AllGather 从 128 KB 的 1.32 个到 128 MB 的 6.60 个；这也是它敢说"少占资源"的底气。

### 3.2 数字

- 相对 NCCL：不限通信资源时最高 **2.4×**；把通信限制到 8 个 CTA（H100）/ 9 个 CTA（GH200）时最高 **3.42×**。后者才是它真正想证明的场景——**资源受限时谁更抗压**。
- 活跃 SM 数（AllGather，GH200）。看 T-CCL 和 NCCL 两列：消息越大，NCCL 吃的 SM 越多，T-CCL 到 6.60 就基本封顶。

| 消息大小 | T-CCL | NCCL | Symmetric NCCL |
|---|---|---|---|
| 128 KB | **1.32** | 3.30 | 1.98 |
| 32 MB | **6.60** | 23.43 | 7.92 |
| 128 MB | **6.60** | 23.76 | 9.24 |

（32 MB 这一行 Symmetric NCCL 是 7.92，到 128 MB 才升到 9.24。注意 128 KB 这一行：NCCL 要 3.30 个 SM 才干同样的活。）

- **作为 backend 换进 FlashOverlap**（GH200，24 个 GEMM 形状平均，基线是串行 cuBLAS + NCCL）：TP=2 从 1.12× 提到 **1.25×**（最高 1.51×）；TP=4 从 1.04× 提到 **1.14×**（最高 1.34×）。收益来自"通信少占的 SM 还给了 GEMM"，而不是通信本身变快。
- **接进 vLLM**（Qwen2.5-72B-Instruct，TP=4，GH200）：对话负载相对 vLLM Auto 的几何平均 **1.10×**、最高 1.22×；decode-heavy 负载几何平均 **1.15×**、最高 1.31×。

### 3.3 边界

- **不支持 CUDA Graph**。论文的 vLLM 实验是在 eager 模式下跑的（因为 planning 和 launch 有 host 侧工作），作者也明说上 Graph 之后两边都会受益。
- **H100 小消息有回归**：相对 NCCL，H100 上 AllReduce 的范围是 0.94×–2.74×、ReduceScatter 是 0.95×–3.42×，最小值都小于 1，且集中在 H100 的小消息。
- **只做单节点**（intra-node，靠 P2P 访问对端显存），评测规模是 TP=2 / TP=4。
- **TP=4 的最差形状仍然低于串行基线**：不是 T-CCL 独有的问题，而是小 GEMM 没多少计算可以拿来藏通信。FlashOverlap 的搜索选出来的配置，可能还不如老实串行。TP=2 时 T-CCL 变体在全部 24 个形状上都高于基线，NCCL 变体则有一个低于基线的最差 case。

## 4. TileLink：让编译器生成融合 kernel

Flux 是手写几千行 CUDA 换来的性能。TileLink（[2503.20313](https://arxiv.org/abs/2503.20313)，ByteDance Seed，后端就是 Triton-distributed）问的是另一个问题：**这套融合能不能自动化？**

### 4.1 机制

- **前端：用 tile-centric primitives 把通信和计算的设计空间解耦。** 计算侧按 GEMM 的最优选 tile，通信侧按链路的最优选 tile，两边 tile 形状可以不一样；producer-consumer 依赖用细粒度的 barrier/notify 维护（`producer_tile_notify`、`peer_tile_wait` 这类原语）。
- **后端：把原语 lower 成 TMA / NVSHMEM 指令**，并负责 shape mapping、rank mapping、channel mapping（可以是编译期仿射变换的静态映射，也可以是运行期查表的动态映射——MoE 的动态路由就需要后者）。
- **资源绑定是一个显式选择**：通信可以绑到 DMA（copy engine）、绑到 SM、或者混合——比如 GEMM+RS 用 DMA 做 scatter、用 SM 做 reduction。这是"让"这个动作第一次变成编译器的参数。
- **MoE 的例子**：一个 AllGather + MoE 的 kernel 里，用 **20 个 SM** 做通信，其余算 expert。

### 4.2 数字（8×H800）

- 相对非重叠基线：**1.17×–20.76×**（20.76× 出现在 MoE 的 cuBLAS+NCCL 对比上）。
- AG + GEMM：**1.27×** 于 cuBLAS+NCCL，达到 Flux 的 **94.5%**（Flux 自己是 1.34×）。这里 TileLink 把 AllGather 绑到 DMA engine。
- GEMM + RS：**1.25×** 于 cuBLAS+NCCL、**1.28×** 于 Flux、**2.22×** 于 Async-TP PyTorch。这里是混合绑定：scatter 走 DMA、reduction 走 SM。
- 整个 MLP 层（两段 GEMM + 激活）：Flux 的 **101.4%**，即基本打平。
- MoE 层：平均 **1.14×** 于 vLLM。
- 端到端 8 个模型、8×H800：平均 **1.32×**（dense 1.20×，MoE 1.54×）；两节点 16 GPU 平均 1.29×。

代价也很清楚：**代码量**。论文特意对比了一句——TileLink 只要几百行 Python，Flux 要几千行 CUDA。

### 4.3 边界

- AG+GEMM 仍然**落后 Flux 5.5%**：手写的 prologue 融合（尤其 TMA + swizzling 那一套）还是更狠。TileLink 赢在 GEMM+RS 和 MoE（Flux 和 Async-TP 都不支持 MoE 的 overlap）。
- **20 个 SM 是手选的例子**，出现在论文的代码示例里，不是自动搜索出来的最优——真实部署要自己扫这个预算，而它和形状强相关。
- **只支持 NVIDIA**：后端压在 TMA 和 NVSHMEM 上，换硬件这套 lowering 要重写。

## 5. CommFuse：把 collective 拆成 P2P，让最后一块不需要通信

前三个都在 kernel 层动手。CommFuse（[2604.24013](https://arxiv.org/abs/2604.24013)）回到算法层，问的是：**tail 为什么一定存在？**

数据切分方案的 tail 来自结构：切成 $N$ 块，前 $N-1$ 块的通信可以藏在计算里，**最后一块的通信没人藏**——它后面没有计算了。切得越细，tail 的绝对时间越小，但算术强度也越差。

CommFuse 的办法是把 RS / AG 拆成 ring 顺序的 P2P（FuseRS / FuseAG），然后 **rank-adaptive 地重排计算顺序**：

- **FuseRS**：对本 rank 来说最后要留下的那片输出，安排在最后算——它算完就结束了，**不需要再通信**；
- **FuseAG**：反过来，本地已有的那片先算，从别人那里收到的后算，这样"等数据"的时间被本地计算盖住。

于是"最后一块"从"必须通信"变成"不必通信"，tail 被结构性地消掉，而不是靠把 chunk 切得更细来摊薄。

代价是通信从一次 collective 变成 $N-1$ 步 P2P：同步点变多，每一步都要自己保证顺序正确。论文的立场是，这类同步是点对点的、可控的，不像"拆解 collective 但总通信量上升"的方案那样容易把省下的 tail 又吃回去。它还强调这套做法不依赖 batch size，不会像按 batch 切那样在小 batch 下退化。

效果（论文的 MLP 实验，TP=4）。看第二列：通信开销从 43.8 ms 压到 0.1 ms，但端到端只从 122.3 ms 降到 77.1 ms——通信省掉 99.8%，总时间只省 36.9%，因为剩下的大头是计算。

| 方案 | 通信开销 (ms) | 端到端 (ms) |
|---|---|---|
| Vanilla 基线 | 43.8 | 122.3 |
| 数据切分 | 7.1（−83.9%） | 83.9（−31.4%） |
| CommFuse | **0.1（−99.8%）** | **77.1（−36.9%）** |

注意它的机制和前面三个是**正交**的：CommFuse 不要求 fused kernel，它就是改调度顺序；论文也说实现层还可以再叠融合 kernel。

边界也要记：

- 论文的评测段**没有写明加速器型号和互联**（正文只说 accelerator / device），所以这组数字不能直接和 Flux 的"A100 NVLink / H800 NVLink"对齐。
- 长序列注意力上，数据切分方案本身会翻车：$B=4, S=8192$ 时切分方案 **279.7 ms**，比 baseline 的 **195.6 ms** 还慢（CommFuse 自己是 145.1 ms）。这既是它的动机，也提醒一件事——**切分不是免费的，序列维切太碎会把 attention 打成 memory-bound。**

## 6. 四个方案放在一起

先看"粒度"和"资源策略"两列：粒度决定 overlap 能做多细，资源策略决定代价落在哪。

| 系统 | 机制 | 粒度 | 资源策略 | 头条数字 | 主要边界 |
|---|---|---|---|---|---|
| **Flux** | 通信 + wait 融进同一个 GEMM kernel，单 kernel | tile（与 TP 度解耦） | 不静态分 SM，靠 tile swizzling 对齐到达顺序 | overlap efficiency 最高 96%（A100 NVLink）；GEMM 级 1.20–3.25×（A100 PCIe） | H800 NVLink 上 decode 无加速；小 m 0.95×–1.03× |
| **T-CCL** | TMA 卸载搬运 + reduction，collective 拆成 shard-in/shard-out 任务 | 任务（TMA 指令流水） | 少占 SM：1.32–6.60 个活跃 SM | vs NCCL 最高 2.4×，限 8/9 CTA 时 3.42× | 无 CUDA Graph；H100 小消息回归；只做单节点 |
| **TileLink** | tile-centric primitives + 编译器 lower 到 TMA/NVSHMEM | tile（通信/计算可不同形状） | 显式绑定 DMA / SM / 混合 | 非重叠基线 1.17×–20.76×；AG+GEMM 1.27× | AG+GEMM 落后 Flux 5.5%；20 SM 手选；仅 NVIDIA |
| **CommFuse** | RS/AG 拆 ring P2P + rank-adaptive 排序消 tail | chunk（按 TP 度） | 不改资源分配，只改顺序 | 通信开销 43.8 → 0.1 ms，端到端 −36.9% | 未写明 GPU 型号；长序列切分可能更慢 |

四个方案对应了本系列开头的四个动作：Flux 是"切 + 融"，T-CCL 是"让"（让出 SM），TileLink 是"融 + 让"的自动化，CommFuse 是"切 + 错"（重排顺序）。**同一个 all-reduce，四种解法，代价完全不同。**

## 7. 一个常被忽略的窗口：反向

前向的 TP 只有一条依赖链，反向不是。一个 Linear 的反向有两个 GEMM——dgrad（对输入求导）与 wgrad（对权重求导）——它们共享输入，但彼此不依赖。也就是说，dgrad 在等那次 reduce-scatter 的时候，wgrad 可以先算。

这解释了一个经验观察：**反向的 TP 通信往往比前向好藏。** 不是因为通信变快了，而是因为旁边多了一件活。Megatron 的 `delay_wgrad_compute` 就是把这个窗口显式化的开关。第四篇里提过它的前置条件：只在 TransformerEngine 实现下可用。它的本质是"把 wgrad 挪到更靠近它的 reduce 的位置"。

代价是显存：wgrad 往后挪，输入激活就要活得更久。这个交换很典型——**overlap 的收益从来不是白拿的，要么用显存换，要么用代码复杂度换，要么用计算效率换。**

## 8. 什么时候选哪个

给一个可执行的判断顺序，前提是先算清第一篇那笔账：

1. **先看通信占比。** 层内通信占比低于 10% 时，任何 overlap 的天花板都是那 10%；Flux 自己在 8%–11% 的场景下也只有 1.04×–1.05×。这时候该做的是减少通信量（量化、低秩、换并行策略），不是换 kernel。
2. **单节点、形状稳定、有人力写 kernel → 融合 kernel（Flux 路线）。** 收益最大（A100 PCIe 上 1.20×–3.25×），代价是几千行 CUDA 和跟硬件强绑定。
3. **不想改模型代码、推理服务想立刻拿收益 → 换 collective backend（T-CCL 路线）。** 前提是单节点、能接受没有 CUDA Graph、消息不太小。这条路的好处是**正交**：它不碰模型，只换 backend，vLLM / TE UserBuffers 都能接。
4. **形状多、要覆盖 MLP/attention/MoE → 让编译器生成（TileLink 路线）。** 代价是接受比手写低几个点的性能（AG+GEMM 差 5.5%），换来几百行 Python 覆盖全部形态。
5. **跨节点 + 小 batch → 先别想 overlap。** 通信比计算慢时，第一篇的 $\max(nc, ng)$ 已经把上界钉死了：切得越细，暴露的绝对时间越大。这时候正确的动作是降低 TP 度、改用 PP/EP、或者量化通信。

关于小 batch 跨节点，多说一句。Flux 的 H800 decode 无加速、CommFuse 的长序列切分翻车、T-CCL 在 TP=4 上的最差形状低于串行。这三件事指向同一个原因：**当一块计算本身已经小到藏不住一次通信时，任何 overlap 都是负收益。** 判断标准很朴素：把这一层的通信时间除以计算时间，如果大于 1，先去减通信量。

## 9. 用模型把账算一遍

TP 的解析账本在 `code/comm-overlap/overlap_model.py` 里，两个入口：`tp_exposed(n_chunks, c, g)` 和 `tp_ledger(n_chunks, c, g)`。

一段 all-gather + GEMM 切成 $n$ 块，每块通信 $g$、计算 $c$，两级流水的 makespan 是：

$$
T = \max(nc,\; ng) + \min(c,\; g)
$$

为什么是这两项？$\max(nc, ng)$ 是"谁慢谁决定总时间"——两条流水线并行跑，总长度由长的那条定；$\min(c,g)$ 是流水线的填充/排空：第一块通信必须等，最后一块计算必须等，加起来正好是一块的 $\min$。

取 $c = 1$、$g = 1.5$（通信比计算慢 50%）。看第 2 列：$n$ 每翻一倍，$T$ 就多一个 $ng$；$\min(c,g)$ 是固定的填充项，不随 $n$ 变。

| $n$ | $T = \max(nc,ng)+\min(c,g)$ | exposed $= T - nc$ |
|---|---|---|
| 1 | 2.50 | 1.50 |
| 2 | 4.00 | 2.00 |
| 4 | 7.00 | 3.00 |
| 8 | 13.00 | 5.00 |

这组数第一篇已经算过，这里换成 TP 的语境再看一遍：

- $n=1$ 时 $T = c + g$，完全没有重叠，暴露就是整个 $g$；
- $n$ 变大，总时间按 $ng$ 涨——因为**通信比计算慢，瓶颈在通信**；
- 暴露的绝对时间从 1.50 涨到 5.00，$\eta = 1 - \text{exposed}/(ng)$ 从 0 趋近 $c/g = 66.7\%$，永远到不了 100%。

所以"切得更细"只在 $c \ge g$ 时有用。$g > c$ 时，切分只是把同一个瓶颈摊到更多块上，顺带多付 $n$ 次 launch。

> 这些是**模型值，不是实测**。`--test` 里断言了这些恒等式（比如 $n=1$ 时暴露等于 $g$、$n$ 很大时总时间等于 $ng + c$），跑一遍能确认公式没写错；真实数字以论文和你自己的 `ncu` 为准。

## 结论

- TP 的通信在层内、在关键路径上，**不能挪到 side stream**；唯一的空间是把 GEMM 按 tile（M 或 K）切开、把输出按 tile 散出，让"算已到的"和"传未到的"并发。
- AG 能融进 GEMM 的 prologue、RS 能融进 epilogue，是因为 GEMM 的输入行（M 维）互不依赖、K 维本来就是求和、输出 tile 本来就是终值。**这是数学结构给的窗口，不是工程技巧。**
- 四条路线的取舍：Flux 用几千行 CUDA 换 tile 级融合（A100 NVLink 上 overlap efficiency 最高 96%，但 H800 上 decode 无加速、小 m 会掉到 0.95×）；T-CCL 用 TMA 把 collective 压到 1–6 个 SM（vs NCCL 最高 2.4×，但只做单节点、无 CUDA Graph）；TileLink 把融合交给编译器（1.17×–20.76×，AG+GEMM 差 Flux 5.5%）；CommFuse 在算法层消 tail（通信开销 43.8 → 0.1 ms，但没写明硬件）。
- 判断顺序永远是：**先看通信占比 → 再看 $c$ 和 $g$ 谁大 → 最后才选实现路线。** 通信比计算慢时，切分救不了，只能减通信量。

前面几篇的脉络在这里收了一下：[导读]({{< relref "comm-overlap-00-index" >}})那篇的"切、错、融、让"四个动作，在 TP 这一层全部出现了；[性能账本]({{< relref "comm-overlap-01-performance-model" >}})给出了 $T = \max(nc,ng)+\min(c,g)$ 这个上界；[数据并行]({{< relref "comm-overlap-04-data-parallel" >}})是"通信在关键路径之外"的正面例子，正好和 TP 对照。

下一篇进 MoE：TP 的通信是"一个张量要所有人都有"，MoE 的 all-to-all 是"每个 token 去不同的地方"。后者的依赖更弱（可以按 token 切）、但通信模式更乱（routing 决定流量），于是"切 + 错 + 让"三件事必须一起做——见[第六篇：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}})。

---

**系列导航**：[四：数据并行]({{< relref "comm-overlap-04-data-parallel" >}}) · [六：MoE 专家并行]({{< relref "comm-overlap-06-moe" >}})

> 配套代码：`code/comm-overlap/overlap_model.py`（`tp_exposed` / `tp_ledger`；TP 的账本和 PP/DP/MoE 共用一个文件）
> 自测：`python3 overlap_model.py --test`
