---
title: "通算并行研读（九）：推理侧——decode 的通算并行与它的阈值"
date: 2026-11-02
draft: false
weight: 10
series: ["comm-overlap"]
tags: ["通算并行", "系列", "推理", "decode", "NanoFlow", "TokenWeave", "Flash Communication"]
categories: ["推理框架"]
---

训练侧的 overlap 有一个很舒服的前提：**GEMM 足够大**。一个 $8192 \times 8192 \times 8192$ 的矩阵乘要跑几百微秒，通信慢一点也躲得进去；切 microbatch、切 bucket、切 tile，本质上都是在"大块计算"里找缝隙。

decode 没有这种缝隙。

batch=1、输出长度 1 的时候，一个 token 的计算量是 $2P$ 次浮点运算（$P$ 是参数量），而它要读的权重是 $2P$ 字节（BF16）。70B 模型：算 140 GFLOP，搬 140 GB。**算术强度约 1 FLOP/byte**，比 A100 的 roofline 转折点低了两个数量级。SM 的绝大多数时间不在做 FMA，而在等 HBM 把权重喂进来。

这时候再把 TP 的 all-reduce 塞进去，问题就变了：通信抢的不是"多余的算力"，而是本来就已经被权重加载占满的 HBM 带宽、以及 SM 上那点本来就快见底的 issue 槽。**训练侧 overlap 是"用空闲算力换时间"，decode 侧 overlap 是"在更紧的带宽预算里换时间"。** 前者几乎总能赢一点，后者有盈亏平衡点。

所以这一篇要回答三个问题：

1. decode 的通信占比为什么会有 **20%**，而不是像训练那样被轻松藏到个位数？
2. NanoFlow、TokenWeave 分别在哪个粒度上做 overlap，各自付出什么代价？
3. 为什么存在一个"batch 小于某个值就不要切"的阈值，这个阈值由什么决定？

前置是[第一篇的账本]({{< relref "comm-overlap-01-performance-model" >}})（$T = C + E + B$、$\eta$、overlap tax）；如果关心 SM 怎么分给通信，可以配合[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})一起看。

## 1. 一个 token 的账：算 0.7 微秒，搬 109 微秒

先把 decode 的 roofline 摊开。设模型 $P$ 个参数、BF16 权重、batch 为 $B$：

- 计算量：每 token $2P$ FLOP，一个 step $2PB$；
- 权重读取：$2P$ 字节，与 $B$ 无关（每个 step 都要把权重完整读一遍）；
- KV cache 读取：每 token 每层 $2 \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot L \cdot 2$ 字节，随上下文长度线性增长；
- TP 通信：每层 2 次 all-reduce（attention 后、FFN 后），每次 $B \cdot H \cdot 2$ 字节。

算术强度（忽略 KV，取 $b=2$ 字节）：

$$
\text{AI} \approx \frac{2PB}{2P} = B \quad (\text{FLOP/byte})
$$

A100 80GB 的转折点是 $312\ \text{TFLOPS} / 2.0\ \text{TB/s} \approx 156$ FLOP/byte。也就是说，**70B 模型在 BF16 下要 batch 到 150 以上才算进 compute-bound 区**；加上 KV cache 的读取，实际还要更高。在此之前，decode 一直是 memory-bound。

手算一层（70B、80 层、TP=8、单卡）：

```text
每层每卡权重   = 140 GB / (80 × 8)      = 219 MB
每层每卡计算   = 2×70e9 / (80 × 8)      = 219 MFLOP
搬这些权重     = 219 MB / 2.0 TB/s      = 109 us
算这些 FLOP    = 219 MFLOP / 312 TFLOPS = 0.7 us
```

**搬 109 微秒，算 0.7 微秒，差 156 倍。** 这就是 decode 的全部困境：SM 有 99% 的时间在等内存，MFU 停在 1% 量级不是"没优化好"，而是 roofline 画在那里。

### 1.1 通信为什么是 20%，而不是 5%

单看字节数，TP 的 all-reduce 在 decode 里小得可怜。$B=1$、$H=8192$ 时一次 all-reduce 的 payload 是 16 KB；8 卡 ring 一圈，每卡实际搬运约 28 KB，按 400 GB/s 的 NVLink 算是 **0.07 微秒**——相对 109 微秒的权重加载可以忽略。

但实测不是这样。TokenWeave 的 Fig.1 在 H100 上测到：**TP 通信占 decode 总时间的约 20%，最高 23%。**

差在哪里？差在 **16 KB 的消息是延迟受限的，不是带宽受限的**。一次 collective 的固定成本包括 kernel launch、rank 间同步、NVSHARP/Multimem 的建链与归约指令下发——这些开销跟消息多大基本无关。

80 层、每层 2 次，一个 token 就是 **160 次 all-reduce**。粗估每次 collective 的固定开销 5–10 微秒（这个量级来自 launch 与 rank 同步，和消息大小无关），160 次就是 0.8–1.6 毫秒；而权重加载是 8.75 毫秒（17.5 GB / 2.0 TB/s），固定开销占其中的 9%–18%——正好落在 TokenWeave 测到的量级上。

这个观察决定了推理侧 overlap 的技术路线，和训练侧完全不同：

| | 训练（大 GEMM） | decode（小 batch） |
|---|---|---|
| 通信的消息大小 | MB 级 | KB 级 |
| 限制因素 | 带宽 | **延迟 / 固定开销** |
| overlap 的收益来源 | 把通信藏进计算 | 减少 launch、减少 SM 占用、减少字节 |
| 典型优化 | chunk、bucket、tile 融合 | 融合 kernel、量化、切 batch |

读法：先把"限制因素"和"收益来源"两行对齐看，最后一行是它们的推论。

**在 decode 里，省一次 launch 往往比省一半带宽更值钱。** 这句话是理解后面三篇论文的钥匙。

顺带回答一个很自然的反问：**那把 batch 拉大不就行了？** 拉到 150 以上就进 compute-bound 区，通信占比自然下降。现实是拉不上去——KV cache 的显存占用随 batch 和上下文长度**线性增长**，而权重是固定的。70B 模型在长上下文下，KV cache 往往比权重还大。decode 的 batch 上限通常由显存决定，而不是由算力或带宽决定。**所以"阈值"不是一个可以靠调 batch 绕过去的东西，它是 serving 系统必须在原地解决的问题。** 这也是为什么下面三篇论文全都在"小 batch 且不能变大"这个约束下做文章。

## 2. NanoFlow：复制算子，让三种瓶颈同时忙

NanoFlow（[2408.12757](https://arxiv.org/abs/2408.12757)）的出发点是论文里的一句观察，值得原样引下来：

> 单个算子能用掉它瓶颈资源的约 **80%**，但串行执行下来，总的计算利用率只有 **40%**。

原因是资源不是一个。GEMM 吃 SM 的 FMA 单元，decode attention 吃 HBM 带宽，TP all-reduce 吃 NVLink。三者串行执行时，每一段都只压满一种资源，另外两种在空转：

```text
串行（单流）：
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │  GEMM    │ │ decode   │ │AllReduce │
  │ 算力 ~80%│ │ HBM  ~80%│ │ NVLink80%│
  │ HBM ~20% │ │ 算力 ~5% │ │ 算力 ~5% │
  └──────────┘ └──────────┘ └──────────┘
   ↑ 每一段都只有一种资源在满负荷，另外两种在等

NanoFlow（nano-batch × 复制算子）：
  stream 0: [GEMM n0][GEMM n1][GEMM n2] ...
  stream 1:     [attn n0][attn n1][attn n2] ...
  stream 2:         [AR n0]  [AR n1]  ...
                   └─ 同一时刻：算力/HBM/NVLink 三种资源同时在忙
```

做法叫 **intra-device parallelism（IDP）**：把 request batch 切成若干 **nano-batch**，然后**把算子复制一份**——不是把一个大 kernel 切小，而是同时跑两个 GEMM kernel、两个 attention kernel、两个通信 kernel，让它们分别压不同的资源。

这里有个容易忽略的细节：**"复制算子"和"切 batch"不是一回事。** 切 batch 是把一个 kernel 的 grid 变小；复制算子是在两个 stream 上各起一个完整 kernel。前者不改变"一次只有一个 kernel 在跑"，后者才制造了并发。代价也很直接：两个 GEMM kernel 各自要读一遍权重，**权重加载的 I/O 翻倍**——这正好戳在 decode 最疼的地方。

### 2.1 MILP 搜什么

nano-batch 怎么切、切几个、按什么顺序排、每个 stream 分几个 SM——NanoFlow 把这四个决策丢给一个 **MILP**（混合整数线性规划）求解器自动搜：

- nano-batch 的**个数**与每个的**大小**（决定每个 kernel 的 grid 尺寸）；
- 算子的**执行顺序**（谁先发，决定重叠窗口）；
- 每个 stream 的 **SM quota**（决定 kernel 之间怎么抢 SM）。

MILP 的目标函数里必须包含**kernel 干扰模型**：两个 kernel 同时跑，各自的耗时不等于单独跑。NanoFlow 用一个归一化的性能代理 $R_A$ 描述 A 被拖慢了多少：A 只跑出单跑最好性能的 40%，就记 $R_A=0.4$；B 则假设拿到剩下的 $R_B = 1 - R_A$。注意 $R$ 不是"SM 占比"，而是以 GEMM 为中心的相对性能；B 的归一化性能 $P_B$（它对内存/网络带宽的利用程度）不是算出来的，而是一张 $R \to P$ 的实测表——来自两两 kernel 的离线 profiling。

这个假设显然不精确（还有 L2、HBM 带宽、功耗的耦合），所以它要做两层离线 profiling：每个 kernel 的"耗时 vs SM 数"曲线，以及每一对 kernel 的干扰表。

### 2.2 数字与代价

配置：LLaMA-2-70B，8×A100 80GB SXM，dense batch 2048。

- 对比 TensorRT-LLM **1.91×**，对比 vLLM **4.18×**，对比 DeepSpeed-FastGen **3.45×**；
- 达到理论最优的 **68.5%**（最好 case）；不同模型、不同配置在 **50–72%** 之间；
- 低负载下 SLO 内的请求率提升 **1.64×**。

需要照写的三个代价：

1. **离理论最优还差 31.5%。** 这个差距不是调参能补的，而是"复制算子 + 静态 SM 配额"这套抽象的天花板——真实干扰不是线性的，配额也不能随 kernel 阶段动态变。
2. **权重加载 I/O 翻倍。** 在 memory-bound 的 decode 里，这是硬成本。它能被"计算与通信重叠"的收益盖住，只在 batch 大到权重加载已经被摊薄之后才成立。
3. **干扰模型假设 $R_B = 1 - R_A$。** 这个假设忽略了共享 L2 和 HBM 控制器带来的次线性/超线性效应，所以在带宽争用剧烈的形状上，MILP 解出来的"最优"可能不是最优。

还有一条不写在论文摘要里、但一定会撞上的约束：**nano-batch 不能无限切小。** 每个 nano-batch 都要独立跑一个 kernel，切得太细时，每个 kernel 的 grid 都掉到 wave 下限以下——一个 wave 能装下的活被拆成三份，三份各占一个 wave，总时间反而翻倍。**这和第五节算出的 4K 阈值是同一个机制**：粒度小到一定程度，wave 的固定开销就变成主项。论文的模型里没有显式的 wave 项，它靠 profile 出来的 kernel 时间去吸收这部分；但这不影响"阈值存在"这个结论。

## 3. TokenWeave：wave-aware 切分与 AR-RMSNorm 融合

TokenWeave（[2505.11329](https://arxiv.org/abs/2505.11329)）走的是另一条路：**不复制算子，只切 token，然后把切分本身的开销消掉。**

### 3.1 为什么"切一半"可以不额外花钱

GPU 上的 kernel 是按 **wave** 调度的：一个 wave 是能同时驻留在所有 SM 上的一批 CTA（H100 是 132 个）。**朴素地把 token 对半分，并不免费。** 论文的反例是 300 个 CTA、正好 3 个 wave；对半分 150+150，每半都要 2 个 wave，总共 4 个——切分自己多付了一个 wave。

所以 TokenWeave 要的是 "**wave-aware**" 切分：切点选在 wave 边界上，让两半的 wave 数加起来等于原来的 wave 数。同样 300 个 CTA，smart split 切成 132+168，1 个 wave 加 2 个 wave，总数还是 3 个，计算侧才真正免费。切歪了就退化成 150+150，总 wave 从 3 涨到 4，反而慢约 33%。这个现象在第五节的模型里会精确地重现。

切好之后，两个 split 走两个 CUDA stream：

```text
单流（不切）：
  [  attn A+B  ][   FFN A+B   ][ AR A+B ][ RMS A+B ]
                              └── 通信与 norm 全部暴露 ──┘

TokenWeave（双流 + wave-aware 切分）：
  split A: [ attn A ][  FFN A  ][ AR A ]
  split B:             [ attn B ][  FFN B  ][ AR B ]
                       └──── A 的 AR 落进 B 的计算窗口 ────┘
  总 wave 数不变，两个 AR 各藏掉一半窗口
```

### 3.2 RMSNorm 为什么能跟着 all-reduce 一起做

这一步是全文最巧的地方。标准 TP 的 row-parallel linear 之后是 all-reduce，然后接 RMSNorm。RMSNorm 要沿 hidden 维求平方和，所以它**必须在 all-reduce 之后**、且需要完整的 hidden 向量。

TokenWeave 把 all-reduce 拆成 reduce-scatter + all-gather，顺序改成：**先 reduce-scatter，再 RMSNorm，最后按需 all-gather**。关键是它沿 **token 边界**切，不是沿 hidden 切：每张卡分到 $1/N$ 的 token，但每个 token 的 hidden 维是完整的。所以 RMSNorm 要的平方和可以在本地算完，**不需要跨卡再做一次标量归约**；RMSNorm 就这样被塞进了 reduce-scatter 和 all-gather 之间。

结果是两个收益叠加：

- RMSNorm 从"一次独立的 kernel launch + 一次完整的 HBM 往返"变成"ReduceScatter、RMSNorm、AllGather 融进同一个 kernel，中间结果尽量留在片上"；
- 这个融合 kernel 基于 Multimem/NVSHARP，活跃 SM 从 **16–20+ 降到 2–8**。

回到第一节的结论：decode 里省的是**固定开销**，不是带宽。少一次 launch、少占十几个 SM，正好打在这个点上。

### 3.3 数字与负结果

配置：8×H100 DGX（另有 4×H100、8×B200 的结果）。

- TP 通信占比 **20%**（Fig.1 最高 **23%**），RMSNorm 本身再加 **4–9%**；
- fused kernel 只占 **2–8 个 SM**（此前 16–20+）；
- 延迟最高 **1.28×**，1K 以上普遍 **1.16–1.28×**；吞吐最高 **1.19×**（ShareGPT 1.19×，arXiv 1.15×）；
- 单算子层面，fused AR-RMSNorm 在 H100 上 **1.34–1.39×**，B200 上 **1.24–1.38×**；
- 单层对比 TileLink 最高 **1.35×**——TileLink 在 2K 上就已经是负收益，要到 8K 以上才拿到最大 **1.2×**。

必须照写的负结果和代价：

1. **朴素融合在 512–8K 全程更慢（0.92–0.98×）。** 原因很干脆：拆开 all-reduce 意味着中间结果要在 HBM 里多读写一遍，省下的计算时间抵不过新增的访存。**"融合"本身不产生收益，"融合 + 消掉切分开销"才产生收益。**
2. **Mixtral-8x22B 在 1K/2K 上，完整方案是净开销。** 因为它的 FFN 在 decode 阶段已经彻底 memory-bound，切成两半后 FFN 的权重加载无法被摊薄，重叠窗口里没有"计算"可藏；只有 fuse-only（不切 batch、只融 AR-RMSNorm）才有效。
3. **没有 CUDA Graph。** 双流 + 动态切分让 graph capture 变复杂，实际部署时这部分开销要自己补。
4. **4 卡的收益低于 8 卡。** TP 越小，通信总量越小，能藏的东西越少——这符合第一节的判断：收益上限是通信占比。

## 4. Flash Communication：不减时间，减体积

Flash Communication（[2412.04964](https://arxiv.org/abs/2412.04964)）严格说**不是 overlap**，它做的是**减少通信量**。放在这一篇里，是因为它和 overlap 是两条正交的轴，而且在 decode 场景下往往比 overlap 更划算。

### 4.1 低比特 + 两步 all-reduce

两个改动：

**一是量化激活。** TP 的 all-reduce 传的是激活。这里有个反直觉的点：**激活比权重更难量化**——论文引 LLM.int8()/SmoothQuant 的结论，激活里存在 outlier，动态范围大；权重的分布反而温和。训练侧的通信量化只压权重和梯度，正好躲开了这个坑；推理侧要压的恰恰是激活。

那为什么仍选 INT6？因为 decode 的 TP 链路上只有激活在传，不压它就没有别的可减；误差只能靠混合精度兜：ReduceSum 用 **INT4**，AllGather 用 **INT8**，两者组合的方案被作者叫作 **"INT6"**（等效 6 bit）。

**二是换算法。** ring all-reduce 要 $2(N-1)$ 步；改成两步之后，每卡只做**一次 All2All reduce + 一次 AllGather**：

```text
ring all-reduce（N 卡）:
  reduce-scatter: N-1 步  →  all-gather: N-1 步     共 2(N-1) 步
  量化/反量化：每步一次                                共 N 次

Flash Communication:
  All2All(reduce) 1 步  →  AllGather 1 步             共 2 步
  量化/反量化：入口一次、出口一次                       共 2 次
```

量化和反量化的次数从 $N$ 降到 **2**，reduce/gather 的步数从 $N-1$ 降到 **1**——**这又是"省固定开销"，不是"省带宽"**，和 TokenWeave 的思路同源。整个流程被塞进一个 fused **Flash All-Reduce kernel**。

### 4.2 数字与代价

- 节点内通信加速 **>3×**；
- TTFT：L40 上 TP=4 为 **2.06×**，A100 上 TP=8 只有 **1.19×**；
- LLaMA-3-70B 平均分（1024 输入 / 64 输出，batch 8）：FP16 **80.00** → INT6 **79.29** → INT4 **78.29**。

三个必须一起读的限定：

1. **A100 只有 1.19×，不是方法不行，是带宽太高。** A100 的 NVLink 带宽比 L40 高得多，通信本来就不那么疼，砍掉 3/4 的字节也换不来 2×。**收益和"通信占比"成正比，这条规律在量化上同样成立。**
2. **权重也量化到 INT8 时掉得更多**（79.65 → 78.75）。这说明权重压到 8 bit 之后，通信再压到 6 bit 的误差更显眼——论文因此把通信精度卡在 INT8/INT4 的混合上，而不是一路压到 INT4。
3. **它减的是体积，不是重叠。** 通信该暴露还是暴露；只是暴露的那段时间变短了。**在 batch 很小、通信纯延迟受限的场景，量化帮不上忙**（消息已经从 16 KB 变成 4 KB，固定开销一分没少）。

## 5. 把阈值算出来：`decode_overlap.py`

前三节讲了三种手段，但都没回答一个问题：**什么时候根本不该切？** 用 `decode_overlap.py` 把这个阈值算出来。

模型是两级流水的理想化版本：

$$
c(T) = \max\left(\left\lceil \frac{T \cdot w / S}{u} \right\rceil u,\ u\right), \qquad m(T) = \frac{T \cdot v}{\text{BW}}
$$

其中 $T$ 是 token 数、$w$ 是每 token 的计算工作量、$S$ 是可用 SM 数、$u$ 是 wave 的时间粒度（wave quantization 的最小单位）、$v$ 是每 token 要传的字节、$\text{BW}$ 是有效带宽。$c$ 里有 $\lceil \cdot \rceil$——**这就是 wave 量化的来源**：算不完一个 wave 也要占一个 wave。

于是：

$$
T_{\text{unsplit}} = c(T) + m(T)
$$

$$
T_{\text{split}} = \max\left(2c(T/2),\ 2m(T/2)\right) + \min\left(c(T/2),\ m(T/2)\right)
$$

第二项的含义：切两半之后，两个 stream 各跑一半，取"计算通道"和"通信通道"里更慢的那个当主时间线，再加上另一条通道的**启动尾巴**（一个半 batch 的量）。这就是第一篇那个 $\max(nc, ng) + \min(c,g)$ 在 $n=2$ 时的样子。

实际跑（`--scan`，默认参数：$w=0.5$、$S=132$、$u=10$、$v=8192$ B、$\text{BW}=400$ GB/s）：

```text
$ python3 decode_overlap.py --scan
  tokens   unsplit     split   ratio
     128      12.6      21.3   1.69x
     256      15.2      22.6   1.48x
     512      20.5      25.2   1.23x
    1024      31.0      31.0   1.00x
    2048      51.9      51.9   1.00x
    4096     103.9      93.9   0.90x
    8192     207.8     187.8   0.90x
   16384     405.5     375.5   0.93x
   32768     801.1     741.1   0.93x
   65536    1592.2    1472.2   0.92x
threshold≈4096 tokens
```

（`ratio` 是 split/unsplit，**小于 1 才是收益**。）

### 5.1 曲线为什么长这样

三段，对应三种机制（下面记 $c = c(T)$、$m = m(T)$、$c_h = c(T/2)$、$m_h = m(T/2)$）：

**128–512：切分净亏，最差慢 1.69×。** 因为此时的 $c(T)$ 已经被 $u$ 这个下限顶住了——128 个 token 的活不够一个 wave，`max(..., wave_unit)` 直接给一个 wave。切成两半后每半**还是**一个 wave，$2c(T/2) = 2c(T)$，计算时间凭空翻倍。这就是 TokenWeave 说的"wave 数涨了"：**小 batch 下，切分的第一笔开销是把 wave 数从 1 抬到 2。**

**1024–2048：刚好打平（1.00×）。** 单个 batch 仍占 1 个 wave，切两半各占 1 个 wave。这里可以把账拆得很干净：wave 惩罚是 $c_h - c/2 = 10 - 5 = 5$，而不付惩罚时能省下的是 $\min(c,m)/2 = 5$——**两个 5 刚好抵消，净收益为零。** 也就是说，1024–2048 这个区间里，重叠的收益还没有超过"多开一个 wave"的成本。

**4096 及以上：稳定收益（0.90–0.93×）。** 关键转折在 4096：此时 $c(4096)$ 是 2 个 wave，$c(2048)$ 是 1 个 wave，$2c_h = c$——**切分不再多付 wave**。剩下的就是纯赚：$\max(c, m) + \min(c/2, m/2)$ 相对 $c+m$ 省下 $\min(c,m)/2$。

把 4096 这一行拆开对一遍账：

```text
c(4096) = 20.0   m(4096) = 83.9   unsplit = 20.0 + 83.9 = 103.9
c(2048) = 10.0   m(2048) = 41.9
split = max(2×10.0, 2×41.9) + min(10.0, 41.9)
      = max(20.0, 83.9) + 10.0 = 93.9          → 93.9 / 103.9 = 0.90x
```

$2c(2048)=20$ 和 $c(4096)=20$ 一模一样，切分在计算侧完全免费；省下的正是"最后一个半 batch 的通信尾巴" $c(2048)=10$。再往上一档（16384）为什么回落到 0.93×？因为 $c(16384)=70$（7 个 wave）而 $2c(8192)=80$（$2 \times 4$ 个 wave），**又多付了一个 wave**——只要 $T$ 落在"波数不是偶数"的位置，切分就要多付一次。

所以 **threshold≈4096 不是拍出来的，是 wave 边界对齐的位置**。这个数只由 $w$、$S$、$u$ 决定，换硬件、换模型都会变；模型的意义是告诉你"阈值存在、由什么决定"，而不是"阈值是 4096"。

### 5.2 TokenWeave 为什么能到 1K

模型给的阈值是 4K，TokenWeave 却在 **1K 以上**就有 1.16–1.28× 的延迟收益。这不矛盾——**模型缺了两项，而 TokenWeave 正好补的就是这两项**：

1. **RMSNorm 的融合。** 模型里没有 RMSNorm 这一项。真实系统里它是"一次 launch + 一次完整 HBM 往返"，在小 batch 下占比不小（论文测到 4–9%）。融进 AR 之后，串行时间里的这个固定项直接消失，阈值随之前移。
2. **通信 kernel 的 SM 占用从 16–20+ 降到 2–8。** 模型假设两个 stream 用满 $S$ 个 SM；实际通信 kernel 占着 SM 时，计算 kernel 的 wave 会变形。省下十几个 SM，等于把 $c(T)$ 的分母做大。

这两项都属于"**消掉切分本身的开销**"，而不是"把切分做得更聪明"。这也解释了为什么 TokenWeave 的**朴素融合版本在 512–8K 反而慢 0.92–0.98×**：只切不融，模型里那个"多付一个 wave"的开销没有人替它买单。

> **模型的边界**：这是解析模型，不是实测。它没有 kernel launch、没有 L2 命中率、没有 HBM 的 bank 冲突、没有 CUDA Graph 的省法。它的价值是**把"为什么有阈值"讲清楚**，具体的数必须以你自己卡上的 trace 为准。

## 6. PD 分离：跨机器的粗粒度 overlap

前面所有手段都在**一张卡内部**腾挪资源。还有一条更粗的路：**把 prefill 和 decode 拆到不同机器上**（prefill-decode disaggregation，PD 分离）。

动机来自第一节的 roofline：prefill 一次处理上千 token，算术强度高，是 **compute-bound**；decode 一次一个 token，是 **memory-bound**。放在同一张卡上跑，两者互相干扰——prefill 的长 GEMM 会把 decode 的 TPOT 顶上去，decode 的驻留又让 prefill 的 batch 上不去。拆开之后，两类资源各归各的机器，各自按最优 batch 跑，中间只传一次 KV cache。

代价也很清楚：多一次 KV cache 的网络传输（长上下文时不小）、需要额外的调度器做请求路由、以及**两个集群的利用率都不再平滑**（prefill 集群和 decode 集群的配比要调）。

用第一篇的语言说，PD 分离其实是一次**"用跨机器的错峰换掉卡内的 overlap"**：它不试图在同一张卡上把通信藏进计算，而是承认"prefill 的计算"和"decode 的带宽"根本不该共享同一份 SM 和 HBM。代价从"overlap tax"变成了"一次 KV 传输 + 一个更难调的集群配比"。业界更轻量的折中是 **chunked prefill**（把长 prefill 切成块，和 decode 混在一个 batch 里跑），它保留了单机部署的简单性，但也保留了干扰——只是把干扰的粒度从"一次 prefill"缩小到"一个 chunk"。

### 6.1 一个 prefill 侧的 EP overlap 例子

prefill 侧的通算并行仍然值得做，尤其 MoE。**INC-MoE**（[2610.36339](https://arxiv.org/abs/2610.36339)）把 all-gather / reduce-scatter 挪进**交换机**（in-network），让双向链路并发，省掉 GPU 侧的收集与转发：

- prefill 场景 TTFT **−33.1%**，吞吐 **+49.5%**；
- 但 **N=2 时收益为零**：按论文的加速比公式 $(N-2)/(2(N-1))$，$N=2$ 时分子为 0——ring 已经走到下界，没有可再省的那一跳。

这个负结果值得记：**所有依赖"拓扑里的空闲维度"的优化，在最小规模上都会失效**。N=2 就是它的下界。

## 7. 决策表

把训练、prefill、decode 三种场景的判据整理成一张表。注意每一行的"首选动作"不同，**用错行的代价是负收益**（表中通信占比是经验区间，不是某一篇的实测）。

读法：先看第二列找瓶颈，再看最后两列的动作和禁忌；同一行的动作和禁忌不要同时做。

| 场景 | 瓶颈资源 | 通信占比 | 首选动作 | 不要做什么 |
|---|---|---|---|---|
| 训练（大 GEMM） | 算力 | 10–50% | **切 + 错**：bucket、microbatch、1F1B、tile 融合 | 通信 <10% 时别动 SM 配额，overlap tax 会吃掉收益 |
| prefill（长序列） | 算力 | 20–47%（MoE 尤甚） | **融 + 让**：AG+GEMM、in-network collective、SM 分区 | batch 太小时 EP 的 all-to-all 无处可藏，先合 batch |
| decode（小 batch） | HBM 带宽 + 延迟 | ~20% | **减体积 + 省 launch**：低比特量化、AR-Norm 融合、少占 SM | **不要切 batch**：wave 数会翻倍（模型阈值 4K） |
| decode（大 batch） | 从带宽转向算力 | 随 batch 下降 | **切 batch + 双流**：wave-aware split，权重被摊薄后再考虑复制算子 | 权重 I/O 还没摊薄时别复制算子——翻倍的成本会抵消收益 |

再给一个更短的决策顺序，遇到新 workload 按这个顺序问：

1. **通信占比是多少？** 低于 10%，先别做 overlap，去做计算优化或减通信量。
2. **消息有多大？** KB 级 → 延迟受限，优先省 launch、省 SM；MB 级 → 带宽受限，优先省字节、切 chunk。
3. **batch 在阈值哪一侧？** 小于阈值 → 融合（fuse-only）；大于阈值 → 才考虑切 batch。
4. **能不能减体积？** 量化、低秩、层次化——这条和 overlap 正交，通常先做。
5. **记录负结果。** 哪个 batch、哪个模型变慢了；TokenWeave 在 Mixtral 上、TileLink 在 <4K 上、朴素融合在 512–8K 上，都是这么被发现的。

## 结论

- **decode 的 roofline 决定了一切**：batch=1 时算术强度约 1 FLOP/byte，比 A100 的转折点低两个数量级；一层搬 109 微秒、算 0.7 微秒。SM 大部分时间在等 HBM，不是在算。
- **decode 的通信是延迟受限，不是带宽受限**：16 KB 的消息传输只要 0.07 微秒，但一次 collective 的固定开销是微秒级，而一个 token 有 160 次 all-reduce。所以 TP 通信占到 20%（最高 23%）。**省 launch 比省带宽值钱。**
- **NanoFlow 用"复制算子"制造并发**，让算力/HBM/NVLink 三种资源同时忙，MILP 搜 nano-batch 与 SM 配额；1.91× vs TensorRT-LLM，达理论最优 68.5%。代价是权重 I/O 翻倍、干扰模型线性化、还差 31.5%。
- **TokenWeave 用"wave-aware 切分 + 融合消开销"**，把阈值从 4K 推到 1K：延迟 1.28×、吞吐 1.19×、fused kernel 只占 2–8 个 SM。代价是没有 CUDA Graph、Mixtral 上净亏、朴素融合在 512–8K 全程更慢。
- **Flash Communication 换的是体积不是时间**：INT6 量化 + 两步 all-reduce，节点内通信加速 >3×，但 A100 只有 1.19×（带宽越高收益越小）。
- **阈值是结构性的**：`decode_overlap.py` 的 threshold≈4096 来自 wave 边界对齐——小 batch 下切分的第一个代价是把 wave 数从 1 抬到 2。4K 是这个模型（以及 NanoFlow 那类 nano-batch 方案）的阈值；TokenWeave 在 dense 模型上把它推到 1K（Mixtral 仍要 4K），TileLink 则到 8K 以上才拿到最大 1.2×。

**一句话**：训练侧的 overlap 在"抢空闲算力"，decode 侧的 overlap 在"抢已经很紧的固定开销"；前者几乎稳赚，后者必须先算阈值。

推理侧还有一条没展开的线：把 collective 本身做得更轻（低 SM、TMA 发起、persistent kernel），这属于 [第七篇]({{< relref "comm-overlap-07-kernel-sm" >}}) 的 SM 分区话题；MoE 在 prefill 侧的 all-to-all 三段式在[第六篇]({{< relref "comm-overlap-06-moe" >}})。下一篇[第十篇]({{< relref "comm-overlap-10-measurement" >}})回到方法论：**怎么用 nsys/ncu 把上面这些数真正测出来**，以及那些看起来很美但会骗人的指标。

---

**系列导航**：[八：编译与自动调度]({{< relref "comm-overlap-08-compiler" >}}) · [十：实测手册]({{< relref "comm-overlap-10-measurement" >}})

> 配套代码：`code/comm-overlap/decode_overlap.py`
> 自测：`python3 decode_overlap.py --test && python3 decode_overlap.py --scan`
