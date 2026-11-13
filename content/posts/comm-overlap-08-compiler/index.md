---
title: "通算并行研读（八）：参数之间不独立，所以才要编译器"
date: 2026-11-02
draft: false
weight: 9
series: ["comm-overlap"]
tags: ["通算并行", "系列", "编译器", "自动调优", "Syncopate", "Lagom", "TileLink", "Triton"]
categories: ["训练框架"]
---

**手调失灵的根因不是参数多，是参数之间不独立；四条路线分别在 kernel 内、kernel 外、语言层、算法层做自动化，代价是换个 shape 就得重来一遍**。

读完你会知道：

- 五个旋钮为什么不能一个一个调；
- 四条路线各自动了什么、各付什么代价；
- 没有编译器时，今天就能抄的四件事。

## 1. 换一个 shape，同一份 kernel 的答案全部作废

手写 fused kernel 的人，迟早会撞上同一堵墙。第一个 shape 调通了：all-gather 融进 GEMM，通信切 4 块，留 16 个 SM 发 TMA，tile 按 chunk 到达顺序扫。换一个模型，hidden 4096→5120、TP 8→4、序列 8K→32K，这套参数全部作废。

不是百分之几那种作废。Syncopate 的消融里，chunk 计划和 backend 都不动、只换 tile 顺序，就有 >2× 的差异；Lagom 的测量里，NC=16 与 NC=32 的通信时间几乎一样，计算时间差 30.2%。同一份代码换一代卡，最优参数又是另一套；写成常量，等于把某张卡的答案焊死。

也没法"按比例"推：chunk 从 4 改到 6，最优 backend 可能从 TMA 变成 copy engine；SM 从 16 加到 24，计算侧的 wave 可能正好掉一档。

[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})里的 Flux、[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})里的 persistent kernel 与 SM 分区，都是"人把答案写死"。这一篇换一个问题：这些决定能不能不靠人？可以，但有限——四条路线把"换一个 shape 就要重写一遍"变成了"换一个 shape 就再搜一遍"。

## 2. 五个旋钮，一张表看完，但它们互相耦合

| 旋钮 | 取什么值 | 调坏了会怎样 | 谁在调 |
|---|---|---|---|
| 通信粒度 | chunk / split factor：1、2、3、4、8 | 太小 launch 与同步占比高，太大尾块暴露 | Syncopate、CommFuse |
| tile 顺序 | M/N/K 轴序、chunk 内与 chunk 间 swizzle | 数据到了计算没正好要用它，L2 局部性差 | Syncopate、TileLink |
| 通信后端 | copy engine / TMA / load-store | 带宽、延迟、资源曲线完全不同 | Syncopate、TileLink |
| SM 配额 | 给通信 4/8/16/24/32 个 SM | 少了发不满带宽，多了饿死计算 | Lagom、SM 分区 |
| 流与优先级 | 几条流、谁优先、是否跨 stream 同步 | 重叠窗口与尾延迟 | 框架层 |

五个维度都不是单调的，而且互相耦合。chunk 变小，copy engine 每次传输要 host launch，约 2–3 μs，传输本身可能还没 launch 久；chunk 变大，尾块暴露；backend 一换，需要的 SM 数就跟着变——TMA 要 16 个左右的 SM 才跑到 300+ GB/s，copy engine 则完全不占 SM。

**所以手调会停在局部点**：贪心地一个一个调，结果可能比最优差一倍以上。把五档各取几档——split 6 档、tile order 4 种、backend 5 种、SM 配额 6 档、流优先级 2 种——一共 **1440 个候选**。每个候选编译加跑一遍，按每次 30 秒算，一个算子 12 小时。穷举不可行，**怎么剪枝成了四篇论文的分野**。

## 3. 四条路线：动的是四个不同的层

一句话定位：Syncopate 在 kernel 内做编译加实测，Lagom 在 kernel 外做在线调参，TileLink 在语言层做原语加 lowering，CommFuse 在算法层做重排。每一条只讲两件事：自动化了什么，代价是什么。

### 3.1 Syncopate：把 chunk 变成编译器的第一等公民

**自动化了什么**：chunk 计划——通信粒度、tile 顺序、通信后端、SM 配额这四件事一起定。动机一句话：kernel 级 overlap 把粒度锁死在 kernel 边界上，每切一刀就多一次全设备同步、多一次 launch、多一个小 kernel 的最后一个 wave 装不满。

做法是把"通信的单位"从 kernel 里拿出来，变成 communication chunk：一块逻辑数据，加上生产或消费它的那些 tile；大小是逻辑传输大小，不绑定任何 backend。于是一份 schedule 可以换 backend、换 SM 数、换 tile 顺序，而不用重写通信逻辑。

编译四步：建依赖图（chunk 乘 tile）、只插必要的 wait、按 chunk 就绪顺序重排 tile、lower 到五种 backend 再 autotune。第三步只改 tile 的访问顺序、**不搬数据**——很多融合方案为了对齐 layout 要多付一次全局读写，这里用调度换掉了搬移。

**代价是什么**：每个候选都要真编译、真跑一遍，所以必须做在 Triton 上（轻量改写、复用 JIT）；换成 CUTLASS 的全量模板实例化，搜索本身就成了瓶颈。只做单节点，跨 NIC 留作 future work；只支持固定 shape 或少量 bucket，动态 shape 要重新 instantiate。

**数字**（8×H100、NVLink 聚合 900 GB/s）：端到端平均 1.3×、最高 4.7×。

- 在已被手写优化过的 AG-GEMM / GEMM-RS 上：4 GPU 达到最佳基线 99.8%、8 GPU 104%——编译器基本复现手写，偶尔略好。
- 单一维度的差异比总分更大：tile 顺序 >2×，split factor 最优在 2–3 份（约 128 MB）。
- backend 的物理限制：copy engine 在 H100 上 400 GB/s/方向，但每次传输要 host launch（2–3 μs），且只支持连续内存；TMA 用 16 个左右的 SM 到 300+ GB/s，只支持 intra-node P2P。

### 3.2 Lagom：少给通信资源，端到端反而更快

**自动化了什么**：通信占多少资源。命题反着问：通信已经藏得差不多了，为什么还是慢？因为藏住的通信还在抢资源。NC 个 channel 在实现上是 NC 个 threadblock，直接占掉 NC 个 SM 的槽位；C 越大，单次传输的并发访存越猛，和 GEMM 抢 HBM/L2 越凶。Lagom 的测量：计算瓶颈场景下，通信争用能让计算慢 35%。

三个参数是 NC（channel 数）、NT（线程数）、C（chunk size），NT 影响很小，NC 和 C 是主因。代价模型只有一句人话——端到端时间 $Z$ 等于通信总时间 $X$ 与计算总时间 $Y$ 里的较大者：

$$
Z = \max(X, Y)
$$

所以谁大谁决定结果。NC 越大，能分给计算的 SM 越少、wave 越多；C 越大，被通信占走的带宽越多、每个 wave 越慢。这也是"少给通信资源反而更快"的数学形式。

搜索用优先级：每轮挑"计算变慢的代价 / 通信省下的时间"最小的一条通信，把 NC/NT/C 按学习率往上加，直到通信时间小于计算时间。联合空间从 $k^N$ 降到线性，代价是不再保证全局最优。

**数字**（2 节点、每节点 8 张 A40；cluster A 节点内 NVLink 400 Gbps、节点间 2×400 Gbps IB）：两个 pattern 覆盖约 90% 端到端时间。

- Pattern 1，计算瓶颈的单次 AllGather：NCCL 默认 NC=8、C=2 MB；AutoCCL 把 NC 拉到 61，通信快了但争用爆炸，端到端 **0.87×**（比 NCCL 还差）；Lagom 选 NC=2、C=684 KB，端到端 1.35×。
- Pattern 2，多次通信：按优先级先调 ReduceScatter，从 (NC=8, C=2 MB) 改成 (NC=4, C=1366 KB)，拿到 1.43×。
- 总体：FSDP 1.10–1.33×、TP 1.08–1.16×、EP 1.07–1.08×（对 NCCL）；全部场景 1.07–1.33× 对 NCCL、1.03–1.27× 对 AutoCCL。

**代价是什么**：只在 A40 上测（PCIe 与 NVLink 两种集群），没有 H100/H800；EP 只有 1.07–1.08×，all-to-all 的争用结构不同；收敛比 AutoCCL 慢约 2×（两通信场景 16 轮对 33 轮）；前提是计算瓶颈，通信本来就是瓶颈时，AutoCCL 那套把通信调快仍然对。

### 3.3 TileLink：不搜，只保证"切完之后语义还对"

**自动化了什么**：正确性和 lowering，不是性能。用户用 Triton 加原语写 tile 级通信：前端解耦设计空间，通信的 tile 顺序和计算可以不一样（通信按 ring 传，计算从任意 rank 的 tile 开始，用 signal/wait 对齐，而不是被迫同一个 tile size）；后端 lower 到 TMA / NVSHMEM / `ld.acquire`，并把资源绑到 DMA engine、SM 或混合。

| 组 | 原语 | 作用 |
|---|---|---|
| signal | producer / consumer notify-wait | 生产者的 tile 完成即通知消费者，消费者阻塞到依赖满足 |
| signal | peer / rank notify-wait | 跨 rank 同一 tile 的通知；rank 级 barrier 主要给 copy engine 用 |
| data | tile_push / tile_pull / rank_copy | 设备侧发或拉一个 tile；host 侧触发 copy engine |

信号带 release/acquire 语义，编译器做多级流水重排时不会把访存挪到 notify 之后、wait 之前。资源绑定是它的亮点：AG+GEMM 让 AllGather 走 DMA、GEMM 走 SM；GEMM+RS 让 scatter 走 DMA、归约走 SM。

**数字**（8×H800）：对非重叠基线 1.17×–20.76×（20.76× 是 MoE 对 cuBLAS+NCCL 的最大值）。

- AG+GEMM：1.27× 对 cuBLAS+NCCL，达到 Flux 的 94.5%。
- GEMM+RS：1.28× 对 Flux，Flux 的紧耦合融合在这个 case 上反而次优。
- MLP 整层是 Flux 的 101.4%；MoE 层 1.14× 对 vLLM。
- 端到端 8 卡平均 1.32×（dense 1.20×、MoE 1.54×）、两节点 16 卡 1.29×。

**代价是什么**：AG+GEMM 上仍比 Flux 慢 **5.5%**，手写在它最擅长的 pattern 上还是最强。论文例子里给通信 20 个 SM，这个 20 是手写常量、不是搜出来的——它给你旋钮，转旋钮的还是人。只支持 NVIDIA，不支持 pipeline 与 model 级并行（future work）。卖点其实是开发成本：几百行 Python 对 Flux 的几千行 CUDA。

### 3.4 CommFuse：不搜参数，改"算哪一块"的顺序

**自动化了什么**：算法层的重排。尾块是结构性的——只要最后一块要通信，它就一定暴露；把每个 rank 最后要留下的那块输出放到最后算，算完就是本地最终结果，尾块被定义掉了。AG 同理：先算本 rank 已有的块，收到的块按到达顺序算。通信量一点没减，省的只是等待。

**数字**（MLP 层，$d_{model}=4096$、TP=4）：同一份 shape、三种排法。

| 排法 | 计算 (ms) | 通信 (ms) | 端到端 (ms) |
|---|---|---|---|
| vanilla | 78.5 | 43.8 | 122.3 |
| data slicing | 76.9 | 7.1 | 83.9 |
| CommFuse | 77.0 | **0.1** | **77.1** |

通信开销 43.8 → 7.1 → 0.1 ms，端到端延迟降 36.9%——计算列几乎没动，省的全是等待。0.1 是"几乎全藏"，不是数学上的 0。

**代价是什么**：论文正文没写 GPU 型号和互联，这个数只能当趋势看；只在 MLP 与 attention 层验证，没有端到端训练。

粒度也不是越小越好：同一篇的 attention 表里，$B=4, S=8192$ 时 data slicing 要 279.7 ms，比 baseline 的 195.6 ms 还慢（CommFuse 的 query-split 版本是 145.1 ms）。

## 4. 没有编译器时，能直接抄的四件事

不是每个人都能上 Syncopate，但上面四篇的结论可以搬进手写 kernel 或训练框架，成本很低。

- **把 chunk size 提成 config**：别在代码里写死 4 块，至少扫 1、2、3、4、8。`bucket_sim.py --sweep` 先给个数量级——固定总算力和总通信、每桶加 launch 开销，会看到一条 U 形曲线：桶太少尾延迟暴露，桶太多 launch 反噬。Syncopate 的实测印证了：GEMM-AR 的最优是 2–3 份，不是越多越好。
- **扫 tile 顺序**：最便宜、最容易被忽略的一项。同一个 chunk 计划，只改访问顺序（M-major、N-major、按 chunk 就绪、swizzle），差 >2×。手写 kernel 里这通常只是改一个循环嵌套，或改 `blockIdx` 到 `(m,n)` 的映射。
- **把 backend 从 kernel 里拆出来**：至少三个实现——copy engine（大块连续、不占 SM）、TMA（intra-node 结构化张量、愿意给 16 个左右的 SM）、load-store（需要 in-network reduction 的小块）。接口上就是一个 enum，不值得为它改 kernel 结构。
- **量 wave 边界，再定 SM 配额**：`sm_split_model.py` 是个粗模型，输入是计算工作量、通信工作量、带宽饱和点、wave 粒度。默认参数下的输出是阶梯：

```text
 s_comm  T_comp  T_comm       T
     12   100.0   250.0   250.0
     16   100.0   187.5   187.5   ← 通信 SM 到饱和点
     32   100.0   187.5   187.5
     36   150.0   187.5   187.5   ← 计算掉一档 wave
     68   200.0   187.5   200.0
best: s_comm=16 makespan=187.5 (serial=263.3)
```

两件事：通信 SM 过了饱和点（16）再加没用；计算 SM 少到跨过 wave 边界（36）时，计算时间从 100 跳到 150。**所以配额要落在 wave 边界上**，不要落在中间。

这个模型不是实测，但它能告诉你该扫哪几个点，比盲扫 0 到 132 省时间。

粒度、顺序、backend、SM 里通常只有一个瓶颈；同时动四个，测出来的差异分不清是谁的贡献。

最后一条不是旋钮：**先写一个 20 行的代价模型**，再决定要不要搜。Lagom 的模型只有一个 max，却足以把 $k^N$ 降成线性；很多团队一上来就网格搜索，其实缺的是"哪个方向值得搜"。

## 5. 四张脸放一起，横向比之前先看第二列

第二列是"自动化发生在哪一层"，最后两列才是成绩和代价。**四行自动化的根本不是同一个东西**。

| | 自动化的是什么 | 主数字 | 代价 |
|---|---|---|---|
| Syncopate | kernel 内：chunk 计划（粒度、tile 顺序、backend、SM 配额） | 平均 1.3×、最高 4.7×；4 卡 99.8%、8 卡 104% 手写基线 | 单节点；固定 shape 或少量 bucket；每候选真编真跑 |
| Lagom | kernel 外：通信参数 NC/NT/C | 1.35×（Pattern 1），总体 1.07–1.33× 对 NCCL | 仅 A40；EP 1.07–1.08×；迭代慢约 2× |
| TileLink | 语言层：原语到指令的 lowering 与正确性，不搜 | AG+GEMM 1.27×、达 Flux 94.5% | 20 SM 手选；AG+GEMM 慢 Flux 5.5% |
| CommFuse | 算法层：分解与排序，不搜 | 通信 43.8 → 0.1 ms，延迟降 36.9% | 未标 GPU；仅层内验证；长序列切分反例 |

四行是一条光谱：**从左到右，搜索越来越少、改问题结构越来越多**。Syncopate 在给定结构里搜参数，Lagom 在 kernel 外搜资源，TileLink 把结构暴露给人，CommFuse 直接改结构。真正靠改结构或改资源拿到的两个数字（Lagom 的 1.35×、CommFuse 的 −36.9%），都不是"把某个 kernel 生成得更快"。

## 6. 越靠近改结构越赚，但别指望它替你调

- 自动化程度与性能上限大致反向：越靠近改结构（顺序、分解、资源配额），收益越大；越靠近生成实现（backend、tile size），越容易只打平手写。
- 编译器路线的产品不是"更快的 kernel"，是把一次性的手写变成**可复现的搜索**：Syncopate 把 chunk 计划变成可 lower 的 IR，TileLink 把 tile 级通信变成原语，Lagom 把资源分配变成代价模型。
- 负结果也都在前面：Lagom 的对比里 AutoCCL 掉到 0.87×，TileLink 差 Flux 5.5%，CommFuse 在 B=4、S=8192 上切分反而比 baseline 慢 43%。
- 共同缺口：多节点与跨 NIC、动态 shape、以及"什么时候不该 overlap"的判据。

如果只记一件事：先量暴露时长最大的那个 shape，再决定动哪个旋钮。

下一篇转推理侧：decode 是 memory-bound，通信占比随 batch 变小，很多在训练里成立的结论到推理会翻转。[第十篇]({{< relref "comm-overlap-10-measurement" >}})会把数字真正测出来，[总结篇]({{< relref "comm-overlap-11-summary" >}})再收口成地图。

## 附：三条背景路线

- **Centauri**（ASPLOS'24）把"通信分区加调度"形式化：把 collective 切成可独立调度的分区，和计算算子一起放进带资源约束的调度问题，求更短的 makespan。差别在"切多细"归谁：Centauri 交给求解器，TileLink 留给人。
- **TE 的 UserBuffers** 是工业界的手工编译器：在 TransformerEngine 里用 `ub_cfgs` 配置 AG/RS 走 UserBuffers 还是 NCCL，不生成新 kernel。好处开箱即用，坏处只覆盖少数几种模式。
- **Triton-distributed**（[2504.19442](https://arxiv.org/abs/2504.19442)）是开源原语层，TileLink 的实现就落在里面，让 tile-centric primitive 从论文变成可以 import 的东西。

---

**系列导航**：[七：Kernel 与 SM]({{< relref "comm-overlap-07-kernel-sm" >}}) · [九：推理侧]({{< relref "comm-overlap-09-inference" >}})

> 配套代码：`code/comm-overlap/sm_split_model.py`、`bucket_sim.py`
> 自测：`python3 sm_split_model.py --test && python3 bucket_sim.py --test`
