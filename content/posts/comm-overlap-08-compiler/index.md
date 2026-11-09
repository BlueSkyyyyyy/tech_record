---
title: "通算并行研读（八）：编译与自动调度——从 Syncopate 到 Lagom"
date: 2026-11-02
draft: false
weight: 9
series: ["comm-overlap"]
tags: ["通算并行", "系列", "编译器", "自动调优", "Syncopate", "Lagom", "TileLink", "Triton"]
categories: ["训练框架"]
---

手写 fused kernel 的人，迟早会撞上同一堵墙。

第一个 shape 调通了：all-gather 融进 GEMM，通信切 4 块，留 16 个 SM 发 TMA，tile 按 chunk 到达顺序扫。换一个模型——hidden 4096→5120、TP 8→4、序列 8K→32K——这套参数全部作废。同一份代码换一代卡，最优参数也不一样；把它写成常量，等于把某张卡的答案焊死。更麻烦的是没法"按比例"推：chunk 从 4 改到 6，最优 backend 可能从 TMA 变成 copy engine；SM 从 16 加到 24，计算侧的 wave 可能正好掉一档。

[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})里的 Flux、[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})里的 persistent kernel 与 SM 分区，都是"人把答案写死"：人决定切几块、谁来发、留多少 SM。这一篇换一个问题：**这些决定能不能不靠人？**

答案是"能，但有限"。2025–2026 年的四条路线——Syncopate、Lagom、TileLink、CommFuse——代表了四种自动化：编译加实测、在线调参、原语加 lowering、算法重排。它们都没有解决通算并行本身，但把"换一个 shape 就要重写一遍"变成了"换一个 shape 就再搜一遍"。

## 1. 设计空间：五个旋钮，千级候选

先把旋钮摊开。以一次 AG+GEMM 为例，至少有五个维度：

| 旋钮 | 取什么值 | 影响 | 谁在调 |
|---|---|---|---|
| 通信粒度 | chunk / split factor：1、2、3、4、8… | 太小：launch 与同步占比高、带宽打不满；太大：尾块暴露 | Syncopate、CommFuse |
| tile 顺序 | M/N/K 轴序、chunk 内与 chunk 间的 swizzle | 数据到达时计算是否正好要用它；L2 局部性 | Syncopate、TileLink |
| 通信后端 | copy engine / TMA / load-store | 带宽、延迟、资源曲线完全不同 | Syncopate、TileLink |
| SM 配额 | 给通信 4/8/16/24/32… 个 SM | 少了发不满带宽，多了饿死计算 | Lagom、SM 分区 |
| 流与优先级 | 几条流、谁优先、是否跨 stream 同步 | 重叠窗口与尾延迟 | 框架层 |

五个维度都不是单调的，而且互相耦合：

- chunk 变小，单次传输变小，copy engine 的有效带宽下降——每次 host launch 要 2–3 μs，传输本身可能还没 launch 久；
- chunk 变大，尾块变大，最后一个 chunk 的通信暴露；
- backend 换掉，需要的 SM 数变了：TMA 要 16 个左右的 SM 才跑到 300+ GB/s，copy engine 则完全不占 SM；
- tile 顺序换掉，chunk 的"就绪时刻"变了，同一个 chunk size 的最优值也跟着变。

**这就是手调失灵的根本原因：不是参数多，而是参数之间不独立。** 贪心地一个一个调，会停在某个局部点，而这个点可能比最优差一倍以上。Syncopate 的消融里，仅仅换 tile 顺序就能造成 **>2×** 的差异；Lagom 的测量里，NC=16 与 NC=32 的通信时间几乎一样，计算时间却差 **30.2%**。

给个数量级。把上面五个维度各取几档：split 取 6 档、tile order 4 种、backend 5 种、SM 配额 6 档、流优先级 2 种，一共 **1440 个候选**。每个候选要编译加跑一遍。按每次 30 秒算，一个算子就是 12 小时。**穷举不可行，于是"怎么剪枝"变成了核心问题**——这也正是四篇论文的分野。

## 2. Syncopate：把 chunk 变成编译器的第一等公民

### 2.1 kernel 级 overlap 的三处浪费

kernel 级 overlap 的做法是：把 GEMM 切成几个小 kernel，把 all-gather 放另一条流，让它们重叠。Syncopate 指出这有三处结构性浪费：

```text
kernel 级 overlap：
  [GEMM k1] [GEMM k2] [GEMM k3]      ← 每个边界一次 device-wide 同步
      [AG 1]     [AG 2]     [AG 尾块暴露]
  ↑ 每个小 kernel 的最后一个 wave 装不满；尾块没人盖

chunk 级 overlap（Syncopate）：
  [========== 一个 fused GEMM kernel ==========]
     tile wave 按 chunk 就绪顺序重排
     [AG c1]  [AG c2]  [AG c3]      ← 通信从 kernel 内直接发
  ↑ 没有 kernel 边界；chunk 一到就算
```

1. 每个 kernel 边界都要一次全设备同步，还要多一次 launch；
2. 小 kernel 的 tile 数少，最后一个 wave 装不满，SM 利用率下降（wave quantization，尾批凑不满一波的浪费）；
3. 通信的尾巴落在最后，没有计算可以盖它。

这三条其实是同一个病：**粒度的单位被 kernel 锁死了**。要更细的重叠，就得多切 kernel；多切 kernel，就多付 launch 和 wave 量化的代价。

### 2.2 chunk 抽象：粒度、结构、后端三者解耦

Syncopate 的做法是把"通信的单位"从 kernel 里拿出来，变成 **communication chunk**：

- 一个 chunk 是一块逻辑数据，加上生产或消费它的那些 tile；
- chunk 的大小是**逻辑传输大小**，不等于 tile 大小，也不绑定任何 backend；
- 一份 communication schedule 是"每个 rank 上，chunk 级操作（P2P push/pull 或 collective）加依赖"的序列。

于是同一份 schedule 可以换 backend、换 SM 数、换 tile 顺序，而不用重写通信逻辑。**这三件事以前是焊死的：选了 TMA，就得按 TMA 的粒度切；选了 copy engine，就得接受它只传连续内存。**

### 2.3 编译流水线

可以概括成四步：

1. 建依赖图：chunk 乘 tile，谁生产、谁消费、谁等谁；
2. 插 wait：只在必要的地方插，wait 的实现由 backend 决定；
3. 重排 tile：chunk 间按就绪顺序排，chunk 内保持原来的遍历顺序，以保住寄存器、共享内存、L2 的局部性；
4. lower 加 autotune：五种 backend 实现（copy engine、专用 SM 的 TMA、同 SM 的 TMA、专用 SM 的 load/store、同 SM 的 load/store），乘上 chunk size/shape/split、SM 配额、tile 配置，枚举加实测。

注意第 3 步：它**不搬数据**。很多融合方案为了对齐通信和计算的 layout，要额外做一次重排，多付一次全局读写；Syncopate 让 chunk 原地不动，只改 tile 的访问顺序。这是"用调度换搬移"。

顺带说一句为什么必须是源到源。每个候选都要真编译一遍、真跑一遍，编译开销直接决定搜索能不能收敛。Syncopate 在 Triton 上做轻量改写、复用 Triton 的 JIT，才敢枚举上千个候选。如果每个候选都要走一遍完整的 CUTLASS 模板实例化，搜索本身就成了瓶颈。

### 2.4 数字与边界

硬件是 8×H100、NVLink 聚合 900 GB/s。结果：

- 端到端平均 **1.3×**，最高 **4.7×**；
- 在已经被手写优化过的 AG-GEMM / GEMM-RS 上，4 GPU 达到最佳基线的 **99.8%**，8 GPU 达到 **104%**——编译器基本复现了手写，偶尔略好（chunk 把多个 tile 的就绪检查合并了，少了同步点）；
- chunk 的 split factor 有明显最优：GEMM-AR 上 **2–3 份（约 128 MB）** 最好，再细就开始亏；
- 只换 tile 顺序：**>2×** 差异；
- backend 的物理限制：copy engine 在 H100 上 **400 GB/s/方向**，但每次传输要 host launch，约 **2–3 μs**，且只支持连续内存；TMA 用 **16 个左右的 SM** 能到 **300+ GB/s**，但只支持 intra-node P2P，也不支持交换机内 reduction。

这几个数放在一起，能看出甜区：**chunk 不能太小（copy engine 的 launch 税），不能太大（尾块暴露），backend 不能瞎选（差一倍），tile 顺序不能默认（差两倍）。**

边界也要说清楚：

- 只做了单节点；多节点、跨 NIC 是 future work，作者说要给 communication plan 加一个 channel 维度，把 NVLink 和 NIC 分开调度；
- 只支持固定 shape 或少量 shape bucket，动态 shape 要重新 instantiate；
- 它不打算替代专家手写 kernel：在专家已经调好的算子上，目标是打平，不是超过。

**所以 Syncopate 的贡献不是"更快"，而是"可复现"**：把"某个 shape 下 chunk=3、tile 按 chunk 顺序、TMA 16 SM"这种知识，从某个人的脑子里搬进一份可搜索的 schedule。

## 3. Lagom：给通信"少分一点资源"

### 3.1 反命题

如果说 Syncopate 问的是"怎么把通信藏得更好"，Lagom 问的是：**通信已经藏得差不多了，为什么还是慢？**

因为藏住的通信还在抢资源。Lagom 的测量：在计算瓶颈的场景下，通信争用可以让计算慢 **35%**。机制不复杂：

- NC 个 channel 在实现上是 NC 个 threadblock，直接占掉 NC 个 SM 的槽位；计算可用的 SM 少了，wave 数就多了；
- C（chunk size）越大，单次传输的并发访存越猛，和 GEMM 抢 HBM/L2 带宽越凶，每个 wave 的有效带宽下降、wave 变长；
- 两者叠加，计算被拖慢的幅度可能超过通信省下来的时间。

所以 Lagom 的结论很反直觉：**在计算瓶颈下，给通信更多资源，端到端会更慢。** 它的调参目标不是"通信最快"，而是"通信刚好够用"。

### 3.2 三个旋钮与代价模型

Lagom 只调三个资源相关参数：NC（channel 数）、NT（线程数）、C（chunk size）。它沿用 AutoCCL 的分治，先定 Algorithm/Protocol/Transport，再在这三个里搜。测量结论是：NT 影响很小（不改变 SM 占用，也不明显改变全局争用），**NC 和 C 是主因**。

代价模型极简：

$$
Z = \max(X, Y), \quad X = \sum_{j=1}^{N} x_j(s_j), \quad Y = \sum_{i=1}^{M} y_i
$$

其中 $X$ 是通信总时间，$Y$ 是计算总时间，$Z$ 是 makespan。计算时间再拆成 wave 数和每 wave 时间：

$$
g_{ij} = \left\lceil \frac{\mu_i}{(\lambda - NC_j) \cdot TB_i} \right\rceil, \quad
f_{ij} = \theta_{ij} + \frac{(\lambda - NC_j) \cdot TB_i \cdot D_i}{\overline{B} - V(NC_j, C_j)}
$$

$\lambda$ 是总 SM 数，$\mu_i$ 是第 $i$ 个计算需要的 threadblock 数，$TB_i$ 是每 SM 能放的 block 数，$D_i$ 是每 block 的数据量，$\overline{B}$ 是峰值带宽，$V$ 是通信占掉的带宽。两式的意思很直白：**NC 越大，能用的 SM 越少，wave 越多；C 越大，$V$ 越大，每 wave 越慢。** 这就是"少给通信资源反而更快"的数学形式。

### 3.3 从 $k^N$ 到线性

$N$ 个通信、每个 $k$ 种配置，联合空间是 $k^N$（论文说实际超过 $1{,}000{,}000^N$）。Lagom 用一个优先级指标 $H$ 剪枝。$H_j = \Delta Y / \Delta x$（论文 Eq.7），是**代价与收益之比**：分子是调完这个通信后计算变慢了多少，分母是通信省下了多少。换句话说，它是"每省一单位通信时间，要付多少计算代价"。

于是算法每一轮挑 $H$ **最小**（性价比最高）的通信来调，把 NC/NT/C 按学习率往上加，直到通信时间小于计算时间、或通信自身到最优。

代价是它不再保证全局最优，只保证"按性价比排序的贪心"；收益是复杂度从指数降到线性。

### 3.4 两个 pattern

硬件是 2 节点，每节点 8 张 A40（cluster A：节点内 NVLink 400 Gbps，节点间 2×400 Gbps IB；cluster B：PCIe 4.0 加 100 Gbps IB）。在 cluster A 单节点的 Phi-2-2B FSDP 上，两个主要 pattern 覆盖了约 90% 的端到端时间：

- **Pattern 1**（计算瓶颈的单次 AllGather）：NCCL 默认 NC=8、C=2 MB；AutoCCL 把 NC 拉到 61，通信是快了，但争用爆炸，端到端掉到 **0.87×**（比 NCCL 还差）；Lagom 选 NC=2、C=684 KB，通信慢一点，端到端 **1.35×**。
- **Pattern 2**（多次通信）：Lagom 按 $H$ 先调 ReduceScatter，从 (NC=8, C=2 MB) 改成 (NC=4, C=1366 KB)，拿到 **1.43×**。

总体：FSDP **1.10–1.33×**、TP **1.08–1.16×**、EP **1.07–1.08×**（对 NCCL）；全部场景 **1.07–1.33×** 对 NCCL、**1.03–1.27×** 对 AutoCCL。cluster A 的收益更大，因为 NVLink 上 NCCL 默认就把 NC 开得很大。

### 3.5 边界

- 只在 A40 上测（PCIe 和 NVLink 两种集群），没有 H100/H800；
- EP 只有 1.07–1.08×，说明 all-to-all 场景的争用结构不同；
- 收敛比 AutoCCL 慢约 2×（两通信的场景：AutoCCL 16 轮，Lagom 33 轮），线性复杂度的常数不低；
- 前提是计算瓶颈；如果通信本来就是瓶颈，AutoCCL 那套"把通信调快"仍然是对的。

最后一条很重要：**Lagom 和 Syncopate 不是替代关系，而是分工。** Lagom 管"通信占多少资源"，Syncopate 管"kernel 内部怎么排"；一个在 kernel 外，一个在 kernel 内。

## 4. TileLink：原语层，而不是搜索器

### 4.1 前后端分离

TileLink 不搜。它做的事是：**给"手写 tile 级通信"提供一套原语，让编译器负责正确性和 lowering。** 前后端是这样切的：

```text
        用户写的 tile 程序（Triton + TileLink primitives）
                          │
        ┌─────────────────▼─────────────────┐
        │ frontend：解耦通信/计算的设计空间   │
        │  shape / rank / channel mapping    │
        │  静态仿射（编译期）/ 动态查表（运行期）│
        └─────────────────┬─────────────────┘
                          │ lower
        ┌─────────────────▼─────────────────┐
        │ backend：TMA / NVSHMEM / ld.acquire │
        │  资源绑定：DMA engine / SM / 混合    │
        └───────────────────────────────────┘
```

前端解耦的意思是：通信的 tile 顺序和计算的 tile 顺序可以不一样。通信可以按 ring 顺序传，计算可以从任意 rank 的 tile 开始算；两者用 signal/wait 对齐，而不是被迫用同一个 tile size。这一条直接对应第五篇里 Flux 反复处理的"通信 chunk 和计算 tile 对不齐"问题。

### 4.2 原语

TileLink 的原语分两组：

| 组 | 原语 | 作用 |
|---|---|---|
| signal | `producer_tile_notify` / `consumer_tile_wait` | 生产者 tile 完成即通知消费者；消费者阻塞到依赖满足 |
| signal | `peer_tile_notify` / `peer_tile_wait` | 跨 rank 的同一个 tile 之间通知/等待，用来构造 tile 顺序 |
| signal | `rank_notify` / `rank_wait` | rank 级 barrier，主要给 copy engine 用 |
| data | `tile_push_data` / `tile_pull_data` | 设备侧发或拉一个 tile（P2P 或 broadcast） |
| data | `rank_copy_data` | host 侧触发 copy engine 搬数据 |

信号带 release/acquire 语义：notify 是 release，wait 是 acquire。这样编译器做多级流水重排时，不会把访存挪到 notify 之后、wait 之前。

### 4.3 映射与资源绑定

后端把 tile_id 映射到三件事：shape（哪一段张量）、rank（哪台设备）、channel（哪个 barrier）。静态映射是编译期仿射变换；动态映射（MoE 的 routing）变成运行期查表，表项在 launch 前填好。

然后 lower 到具体指令：TMA、NVSHMEM、`ld.global.acquire` / `red.release`。资源绑定是它的一个亮点：

- AG+GEMM：AllGather 走 DMA engine（copy engine），GEMM 走 SM；
- GEMM+RS：混合——scatter 用 DMA，reduction 用 SM；
- MoE：AG 走 DMA，TopK reduce 走 SM，RS 又是 DMA 加 SM 混合。

论文例子里给通信 **20 个 SM**——注意这个 20 是手写的常量，不是搜出来的。这既是 TileLink 的灵活之处，也是它的边界：它给你旋钮，但转旋钮的还是人。

### 4.4 数字

8×H800：

- 对非重叠基线：**1.17×–20.76×**（20.76× 是 MoE 对 cuBLAS+NCCL 的最大值）；
- AG+GEMM：**1.27×** 对 cuBLAS+NCCL，达到 Flux 的 **94.5%**；
- GEMM+RS：**1.28×** 对 Flux（Flux 的紧耦合融合在这个 case 上反而次优，TileLink 的混合映射赢了）；
- MLP 整层：Flux 的 **101.4%**；
- MoE 层：**1.14×** 对 vLLM；
- 端到端 8 卡平均 **1.32×**（dense 1.20×、MoE 1.54×）；两节点 16 卡 **1.29×**。

边界：

- AG+GEMM 上仍比 Flux 慢 **5.5%**——手写在它最擅长的 pattern 上还是最强；
- 20 SM 是手选的，没有自动搜索；
- 只支持 NVIDIA（后端是 Triton 加 PTX）；
- 不支持 pipeline、model 级并行，论文明确留作 future work；
- 它的卖点其实是**开发成本**：几百行 Python 对 Flux 的几千行 CUDA。性能上是"接近手写"，不是"超过手写"。

## 5. 三条背景路线

在进入这四篇之前，还有三条背景值得知道。

**Centauri（ASPLOS'24）** 把"通信分区加调度"形式化：把 collective 切成可独立调度的分区，再把分区和计算算子一起放进一个带依赖和资源约束的调度问题里，求一个更短的 makespan。它的思路是"用求解器替代手写规则"，但调度粒度还在分区或算子级，形状或拓扑一变就要重解。它和 TileLink 的差别也在这里：Centauri 把"切多细"交给求解器；TileLink 把"切多细"留给人，只保证切完之后语义正确。

**TE 的 UserBuffers（TE-CCL）** 是工业界的"手工编译器"：在 TransformerEngine 里用 `ub_cfgs` 配置，`ub_overlap_ag` 决定 all-gather 走 UserBuffers 还是 NCCL，`fill_userbuffers_buffer_for_all_gather` 负责把 all-gather 的目标 buffer 准备好。它不生成新 kernel，而是在框架层把 TP 的 AG/RS 和 GEMM 用 NVSHMEM 拼成固定模式。好处是开箱即用；坏处是只覆盖少数几种模式。

**Triton-distributed（[2504.19442](https://arxiv.org/abs/2504.19442)）** 是开源的原语层：把 tile 级通信原语做进 Triton，TileLink 的实现就落在这个仓库里。它让 tile-centric primitive 从论文变成了可以 import 的东西——这也是这批工作能快速复现的原因之一。

## 6. CommFuse：不搜索，改顺序

### 6.1 让尾块不存在

CommFuse 的出发点和前三篇都不一样：**它不搜任何参数。** 它把 RS/AG 拆成 ring 顺序的 P2P 步骤，然后用一个 rank-adaptive 的排序，让"最后一个 chunk"根本不需要通信：

- **RS**：每个 rank 最后要留下的那块输出，放到最后算。算完就是本地的最终结果，不用再发给任何人——尾块的通信被"定义"掉了。
- **AG**：本 rank 已经有的那块先算，把等数据的时间用来算；收到的块按到达顺序算，P2P 的接收和上一块的计算重叠。

背后是一个很漂亮的观察：**data slicing 的尾块是结构性的**——只要最后一块要通信，它就一定会暴露。你当然可以把块切得更小，但块太小算术强度下降、变成 memory-bound。CommFuse 的做法是改顺序，让尾块不存在。

注意它没有减少通信量：ring 的步数和每步的数据量都没变，变的只是"算哪一块"和"什么时候算"——省下来的是等待，不是字节。

### 6.2 数字

MLP 层，$d_{model}=4096$、TP=4：

| 方案 | 计算 (ms) | 通信 (ms) | 端到端 (ms) |
|---|---|---|---|
| vanilla | 78.5 | 43.8 | 122.3 |
| data slicing | 76.9 | 7.1 | 83.9 |
| CommFuse | 77.0 | **0.1** | **77.1** |

通信开销 43.8 → 7.1 → 0.1 ms，端到端延迟降 **36.9%**；比 data slicing 再快 8.1%。0.1 ms 是"几乎全藏"，不是数学上的 0——作者说完全融合的 kernel 才可能到 0。

### 6.3 一个反例

同一篇论文的 attention 表里有一个负结果：$B=4, S=8192$ 时，data slicing 的 attention 要 **279.7 ms**，比 baseline 的 **195.6 ms** 还慢。原因是这个 shape 下切分后的 attention 计算太碎，算术强度掉下去了。CommFuse 自己的 query-split 版本是 145.1 ms，但它靠的是"切 query 加把输出投影也纳入重叠"，不是单纯把数据切小。

**这个反例比正结果更有信息量：粒度不是越小越好，切法要匹配算子。**

### 6.4 边界

- 论文正文没有写 GPU 型号和互联（至少 arXiv 版没有），所以 43.8 → 0.1 只能当趋势看，不能当某张卡上的绝对性能；
- 论文只在 MLP 与 attention 层上验证（Mamba 只是方法上的映射，没有实验），也没有端到端训练；
- ring 顺序假设，每步 P2P 的同步点增加，跨节点时 ring 的跳数会放大延迟；
- 它和 fused kernel 正交：论文自己说，进一步收益要靠算子级融合。

## 7. 四张脸放一起

先给读法：第二列是"自动化发生在哪一层"，第三列是"怎么剪枝"，第四列是"有没有显式代价模型"，最后两列才是成绩和边界。**横向比之前先看第二列——四行自动化的根本不是同一个东西。**

| | 自动化的是什么 | 搜索空间与方法 | 代价模型 | 头条 | 主要 caveat |
|---|---|---|---|---|---|
| **Syncopate** | chunk 计划到融合 kernel：tile 顺序、backend、SM 配额、chunk size | 枚举加实测（5 种 backend 乘 split 乘 tile order 乘 SM） | 无显式模型，直接测 | 平均 1.3×、最高 4.7×；4 卡 99.8%、8 卡 104% 最佳基线 | 单节点；固定 shape/少量 bucket；多节点 future work |
| **Lagom** | 通信参数 NC/NT/C | 按 H 排序的贪心，$k^N$ 降到线性 | $Z=\max(X,Y)$ 加 wave 与带宽两式 | 1.07–1.33× 对 NCCL；Pattern 1 1.35× | 仅 A40；EP 1.07–1.08×；迭代约 2×；假设 compute-bound |
| **TileLink** | 不自动搜；自动的是原语到指令的 lowering 与正确性 | 人写 tile 程序，旋钮（tile size/order、SM、backend）人手选 | 无 | 1.17–20.76×；AG+GEMM 1.27×、94.5% Flux；e2e 1.32× | AG+GEMM 仍慢 Flux 5.5%；20 SM 手选；仅 NVIDIA；无 pipeline 级 |
| **CommFuse** | 不搜；自动的是分解加排序的算法 | O(1)：一条确定的 rank-adaptive 顺序 | 无（解析地消掉尾块） | 通信 43.8 到 0.1 ms；延迟 −36.9% | 未标 GPU；仅层内；B=4,S=8192 切分反例 |

四行放在一起，能看到一条清晰的光谱：**从左到右，"搜索"越来越少，"改问题结构"越来越多。** Syncopate 在给定结构里搜参数；Lagom 在给定 kernel 外搜资源；TileLink 把结构暴露给人；CommFuse 直接改结构。而真正靠"改结构或改资源"拿到的两个数字（Lagom 的 1.35×、CommFuse 的 −36.9%），都不是"把某个 kernel 生成得更快"。

## 8. 没有编译器时，怎么用这些想法

不是每个人都能上 Syncopate。但上面四篇的结论可以直接搬进手写 kernel 或训练框架，成本很低。

**(a) 把 chunk size 变成可调参数。** 不要在代码里写死 4 块。把它提成 config，至少扫 1、2、3、4、8。`bucket_sim.py --sweep` 能先给你一个数量级：固定总算力和总通信、每桶加 launch 开销，会看到一条 U 形曲线——桶太少尾延迟暴露，桶太多 launch 反噬。Syncopate 的实测也印证了这点：GEMM-AR 的最优是 2–3 份，不是越多越好。

**(b) 扫 tile 顺序。** 这是最便宜、最容易被忽略的一项。同一个 chunk 计划，只改 tile 的访问顺序（M-major、N-major、按 chunk 就绪、swizzle），Syncopate 测到 **>2×** 差异。手写 kernel 里，这通常只是改一个循环嵌套或一个 `blockIdx` 到 `(m,n)` 的映射；先用一个 shape 扫三五种，再固化。

**(c) 把通信后端从 kernel 里拆出来。** 至少把 all-gather 做成三个实现：copy engine（适合大块连续、不想占 SM）、TMA（适合 intra-node 结构化张量、愿意给 16 个左右的 SM）、load-store（适合需要 in-network reduction 的小块）。同一份 schedule 换 backend，Syncopate 的消融里差距可以到"一半性能"。接口上就是一个 enum，不值得为它改 kernel 结构。

**(d) 量 wave 边界，再定 SM 配额。** `sm_split_model.py` 是个粗模型：计算工作量 $W_c$、通信工作量 $W_m$、带宽饱和点 $s_0$、wave 粒度。默认参数下它的输出是阶梯：

```text
 s_comm  T_comp  T_comm       T
     12   100.0   250.0   250.0
     16   100.0   187.5   187.5   ← 通信 SM 到饱和点
     32   100.0   187.5   187.5
     36   150.0   187.5   187.5   ← 计算掉一档 wave
     68   200.0   187.5   200.0
best: s_comm=16 makespan=187.5 (serial=263.3)
```

两件事：通信 SM 过了饱和点（$s_0=16$）再加没用；计算 SM 少到跨过 wave 边界（36）时，计算时间从 100 跳到 150。**所以 SM 配额要落在 wave 边界上，不要落在中间。** 这个模型不是实测，但它能告诉你该扫哪几个点，比盲扫 0 到 132 省时间。

如果只做一件事：先量出暴露时间最大的那个 shape，再决定动哪个旋钮。粒度、顺序、backend、SM 四件事里，通常只有一件是当前瓶颈；同时动四个，测出来的差异分不清是谁的贡献。

最后一条不是旋钮：**先写一个 20 行的代价模型，再决定要不要搜。** Lagom 的模型只有 $Z=\max(X,Y)$ 加两个式子，却足以把 $k^N$ 降成线性；很多团队一上来就上网格搜索或贝叶斯优化，其实缺的是"哪个方向值得搜"的判断。

## 结论

- 编译器路线的产品不是"更快的 kernel"，而是**把一次性的手写变成可复现的搜索**：Syncopate 把 chunk 计划变成可 lower 的 IR，TileLink 把 tile 级通信变成原语，Lagom 把资源分配变成代价模型。
- 四条路线的自动化程度与性能上限大致反向：**越靠近改结构（顺序、分解、资源配额），收益越大；越靠近生成实现（backend、tile size），越容易只打平手写。**
- 负结果要一起记：Lagom 的对比里 AutoCCL 掉到 0.87×，Syncopate 只追平手写，TileLink 差 Flux 5.5%，CommFuse 在 $B=4, S=8192$ 上切分反而比 baseline 慢 43%。
- 共同缺口：多节点与跨 NIC（Syncopate、TileLink 都留作 future work）、动态 shape、以及"什么时候不该 overlap"的判据。

下一篇转向推理侧：decode 是 memory-bound，通信占比随 batch 变小，很多在训练里成立的结论到推理会翻转。[第十篇]({{< relref "comm-overlap-10-measurement" >}})会把这些数字真正测出来，[总结篇]({{< relref "comm-overlap-11-summary" >}})再收口成地图。

---

**系列导航**：[七：Kernel 与 SM]({{< relref "comm-overlap-07-kernel-sm" >}}) · [九：推理侧]({{< relref "comm-overlap-09-inference" >}})

> 配套代码：`code/comm-overlap/sm_split_model.py`、`bucket_sim.py`
> 自测：`python3 sm_split_model.py --test && python3 bucket_sim.py --test`
