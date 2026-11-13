---
title: "通算并行研读（九）：decode 的通信贵在次数，不贵在字节"
date: 2026-11-02
draft: false
weight: 10
series: ["comm-overlap"]
tags: ["通算并行", "系列", "推理", "decode", "NanoFlow", "TokenWeave", "Flash Communication"]
categories: ["推理框架"]
---

**decode 的通信贵在次数、不贵在字节；所以要把省带宽的心思换成省启动开销，代价是小 batch 下切分自己要先多付一个 wave——不到阈值，切了反而更慢**。

读完你会知道：

- decode 的通信为什么能占到 20%，而不是 5%；
- 三条路线各自省的是字节、launch 还是 SM；
- 阈值由什么决定，什么时候根本不该切。

## 1. 一个 token 的账：搬 109 微秒，算 0.7 微秒

decode 的 roofline 很干净。设模型 $P$ 个参数、BF16 权重、batch 为 $B$：

- 计算量：每 token $2P$ FLOP，一个 step $2PB$；
- 权重读取：$2P$ 字节，与 $B$ 无关，每个 step 都要把权重完整读一遍；
- KV cache：每 token 每层 $2 \cdot n_{\text{kv}} \cdot d_{\text{head}} \cdot L \cdot 2$ 字节，随上下文长度线性增长；
- TP 通信：每层 2 次 all-reduce（attention 后、FFN 后），每次 $B \cdot H \cdot 2$ 字节。

忽略 KV、每个元素按 2 字节算，算术强度是：

$$
\text{AI} \approx \frac{2PB}{2P} = B \quad (\text{FLOP/byte})
$$

计算量是 $2P$、要读的也是 $2P$ 字节，算术强度就等于 batch 本身。70B 模型每 token 算 140 GFLOP、搬 140 GB，约 1 FLOP/byte，比 A100 80GB 的转折点低两个数量级——那个转折点是 $312\ \text{TFLOPS} / 2.0\ \text{TB/s} \approx 156$ FLOP/byte。所以 70B 在 BF16 下要 batch 到 150 以上才算进 compute-bound 区，算上 KV cache 的读取还要更高。

手算一层（70B、80 层、TP=8、单卡）：

```text
每层每卡权重   = 140 GB / (80 × 8)      = 219 MB
每层每卡计算   = 2×70e9 / (80 × 8)      = 219 MFLOP
搬这些权重     = 219 MB / 2.0 TB/s      = 109 us
算这些 FLOP    = 219 MFLOP / 312 TFLOPS = 0.7 us
```

**搬 109 微秒，算 0.7 微秒，差 156 倍**。SM 有 99% 的时间在等内存，MFU 停在 1% 量级不是没优化好，而是 roofline 画在那里。

## 2. 通信占 20%：160 次 launch，一次省一点才够

训练侧一个 $8192 \times 8192 \times 8192$ 的矩阵乘要跑几百微秒，通信慢一点也躲得进去；decode 没有这种缝隙。

先按字节算，TP 的 all-reduce 小得可怜。$B=1$、$H=8192$ 时一次 all-reduce 的 payload 是 16 KB；8 卡 ring 一圈，每卡实际搬运约 28 KB，按 400 GB/s 的 NVLink 算是 **0.07 微秒**，相对 109 微秒的权重加载可以忽略。

但 TokenWeave 的 Fig.1 在 H100 上测到，**TP 通信占 decode 总时间的约 20%，最高 23%**。

差在 16 KB 的消息是延迟受限，不是带宽受限：一次 collective 的固定成本跟消息多大基本无关，都是 kernel launch、rank 间同步、NVSHARP/Multimem 的建链与归约指令下发。

80 层、每层 2 次，一个 token 就是 **160 次 all-reduce**。按一次固定开销 5–10 微秒粗估，160 次是 0.8–1.6 毫秒；而权重加载是 8.75 毫秒（17.5 GB / 2.0 TB/s），固定开销占 9%–18%，正好落在 TokenWeave 测到的量级上。

**在 decode 里，省一次 launch 往往比省一半带宽更值钱**。 训练侧的消息是 MB 级、卡在带宽上，overlap 的收益来自把通信藏进计算；decode 的消息是 KB 级、卡在固定开销上，收益来自少 launch、少占 SM、少传字节。

那能不能把 batch 拉到 150 以上，直接进 compute-bound？拉不上去：KV cache 的显存随 batch 和上下文长度线性增长，权重是固定的，70B 模型在长上下文下 KV 往往比权重还大。batch 上限由显存决定，所以阈值不是调 batch 能绕过去的，三条路线都只能在"小 batch 且不能变大"里做文章。

## 3. 三条路线：复制算子、切 token、压体积

### 3.1 NanoFlow：复制一份算子，让三种资源同时忙

NanoFlow（[2408.12757](https://arxiv.org/abs/2408.12757)）的出发点是论文里的两个数：单个算子能用掉它瓶颈资源的约 **80%**，但串行执行下来，总的计算利用率只有 **40%**。

资源不是一个：GEMM 吃 SM 的 FMA 单元，decode attention 吃 HBM 带宽，TP all-reduce 吃 NVLink。串行时每一段都只压满一种，另外两种在空转：

```text
串行： [GEMM 算力 80%]  [attn HBM 80%]  [AR NVLink 80%]      每段只满一种，另两种在等
IDP ： stream0 [GEMM n0][GEMM n1]   stream1 [attn n0][attn n1]   stream2 [AR n0][AR n1]
```

做法叫 intra-device parallelism（IDP）：把 request batch 切成若干 nano-batch，然后**把算子复制一份**——不是把一个大 kernel 切小，而是在两个 stream 上各起一个完整 kernel。前者不改变"一次只有一个 kernel 在跑"，后者才制造并发。

代价直接戳在 decode 最疼的地方：两个 GEMM kernel 各读一遍权重，**权重加载的 I/O 翻倍**。这笔成本能被"计算与通信重叠"的收益盖住，但只在 batch 大到权重加载已经被摊薄之后才成立。

切几个、按什么顺序排、每个 stream 分几个 SM，交给一个 MILP 自动搜。目标函数里必须有 kernel 干扰模型：A 只跑出单跑最好性能的 40% 就记 $R_A=0.4$，B 假设拿到剩下的 $R_B = 1 - R_A$。这个线性假设不精确，所以要两层离线 profiling：每个 kernel 的耗时 vs SM 数曲线，以及每一对 kernel 的干扰表。

配置 LLaMA-2-70B、8×A100 80GB SXM、dense batch 2048：对比 TensorRT-LLM **1.91×**、vLLM **4.18×**、DeepSpeed-FastGen **3.45×**；最好 case 达到理论最优的 **68.5%**，不同配置落在 **50–72%**；低负载下 SLO 内的请求率提升 **1.64×**。

负结果也直白：**离理论最优还差 31.5%**，差距来自"复制算子 + 静态 SM 配额"这套抽象的天花板，不是调参能补的。另外 nano-batch 不能无限切小，切到每个 kernel 的 grid 掉到 wave 下限以下，一个 wave 能装下的活被拆成三份、三份各占一个 wave，总时间反而翻倍——这就是第四节那个阈值机制。论文的模型里没有显式的 wave 项，它靠 profile 出来的 kernel 时间去吸收这部分，但阈值存在这个结论不受影响。

### 3.2 TokenWeave：切 token 不额外花钱，顺手把 norm 融进去

TokenWeave（[2505.11329](https://arxiv.org/abs/2505.11329)）不复制算子，只切 token，然后把切分本身的开销消掉。

GPU 的 kernel 按 wave 调度，一个 wave 是能同时驻留在所有 SM 上的一批 CTA（H100 是 132 个）。**朴素地对半分并不免费**：论文的反例是 300 个 CTA、正好 3 个 wave，对半 150+150 每半要 2 个 wave、总共 4 个。所以切点要落在 wave 边界上，300 个 CTA 切成 132+168，1 个 wave 加 2 个 wave 还是 3 个，计算侧才真正免费；切歪了就退化成 150+150，反而慢约 **33%**。这笔账第四节会用模型精确重现一次。

切好之后两个 split 各走一个 CUDA stream：

```text
单流：   [ attn A+B ][  FFN A+B  ][ AR A+B ][ RMS A+B ]     通信与 norm 全部暴露
双流：   split A: [ attn A ][ FFN A ][ AR A ]
         split B:          [ attn B ][ FFN B ][ AR B ]     两个 AR 各藏掉一半窗口
```

更巧的一步是 RMSNorm。标准 TP 里它在 all-reduce 之后，要完整的 hidden 向量才能求平方和。TokenWeave 把 all-reduce 拆成 reduce-scatter + all-gather，并且沿 **token 边界**切：每张卡分到 $1/N$ 的 token，但每个 token 的 hidden 维是完整的，平方和本地就能算完，不需要跨卡再归约一次标量。

于是 RMSNorm 被塞进 reduce-scatter 和 all-gather 中间，三件事融进一个基于 Multimem/NVSHARP 的 kernel，中间结果尽量留在片上，活跃 SM 从 **16–20+ 降到 2–8**。

配置 8×H100 DGX（另有 4×H100、8×B200 的结果）：TP 通信占 **20%**（Fig.1 最高 **23%**），RMSNorm 自己再加 **4–9%**；延迟最高 **1.28×**，1K 以上普遍 **1.16–1.28×**；吞吐最高 **1.19×**（ShareGPT 1.19×、arXiv 1.15×）；单算子的 fused AR-RMSNorm 在 H100 上 **1.34–1.39×**、B200 上 **1.24–1.38×**；单层对比 TileLink 最高 **1.35×**——TileLink 在 2K 上就已经是负收益，要到 8K 以上才拿到最大 **1.2×**。

负结果：**朴素融合在 512–8K 全程更慢**（0.92–0.98×）——拆开 all-reduce 意味着中间结果要在 HBM 里多读写一遍，省下的计算抵不过新增的访存，融合本身不产生收益。**Mixtral-8x22B 在 1K/2K 上完整方案是净开销**：它的 FFN 在 decode 阶段已经彻底 memory-bound，切成两半后权重加载摊不薄，窗口里没有计算可藏，只有 fuse-only 有效。另外没有 CUDA Graph（双流加动态切分让 capture 变复杂），4 卡的收益也低于 8 卡——TP 越小通信总量越小，能藏的上限就是通信占比。

### 3.3 Flash Communication：不减时间，减体积

Flash Communication（[2412.04964](https://arxiv.org/abs/2412.04964)）严格说不是 overlap，它减的是通信量：量化激活到 INT6，再把 ring all-reduce 的 $2(N-1)$ 步换成一次 All2All reduce 加一次 AllGather。

```text
ring all-reduce：reduce-scatter N-1 步 + all-gather N-1 步   量化/反量化 N 次
Flash Comm.    ：All2All(reduce) 1 步 + AllGather 1 步        量化/反量化 2 次
```

量化和反量化的次数从 $N$ 次降到 2 次——又是省固定开销，不是省带宽，整个流程被塞进一个 fused Flash All-Reduce kernel。为什么选 INT6？decode 的 TP 链路上只有激活在传，不压它就没有别的可减；而激活恰恰比权重难量化，动态范围大、有 outlier，误差只能靠混合精度兜：ReduceSum 用 INT4、AllGather 用 INT8，合起来等效 6 bit。

数字：节点内通信加速 **>3×**；TTFT 在 L40 上 TP=4 是 **2.06×**，A100 上 TP=8 只有 **1.19×**——A100 的 NVLink 带宽高，通信本来就不那么疼，砍掉 3/4 的字节也换不来 2×。收益和通信占比成正比，这条规律在量化上同样成立。质量上 LLaMA-3-70B 平均分（1024 输入 / 64 输出、batch 8）从 FP16 的 **80.00** 掉到 INT6 的 **79.29**、INT4 的 **78.29**；权重也压到 INT8 时掉得更多（**79.65** → **78.75**）。

它减的是体积不是重叠：通信该暴露还是暴露，只是暴露的那段时间变短了。batch 很小时消息已经从 16 KB 变成 4 KB，固定开销一分没少，量化帮不上忙。

## 4. 阈值不是猜的：一个式子，三段曲线

三条路线都在动通信，但都没回答一个问题：什么时候根本不该切。`decode_overlap.py` 把这个阈值算出来。

模型是两级流水的理想化版本：

$$
c(T) = \max\left(\left\lceil \frac{T \cdot w / S}{u} \right\rceil u,\ u\right), \qquad m(T) = \frac{T \cdot v}{\text{BW}}
$$

其中 $T$ 是 token 数、$w$ 是每 token 的计算工作量、$S$ 是可用 SM 数、$u$ 是 wave 的时间粒度、$v$ 是每 token 要传的字节、$\text{BW}$ 是有效带宽。$c$ 里那个 $\lceil \cdot \rceil$ 就是 wave 量化的来源：算不完一个 wave 也要占一个 wave。模型其实只做两件事——把计算按 wave 向上取整，把通信按字节除以带宽。

不切时 $T_{\text{unsplit}} = c(T) + m(T)$；切两半：

$$
T_{\text{split}} = \max\left(2c(T/2),\ 2m(T/2)\right) + \min\left(c(T/2),\ m(T/2)\right)
$$

人话：两个 stream 各跑一半，取计算通道和通信通道里更慢的那个当主时间线，再加上另一条通道的启动尾巴。这就是[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})里 $T=\max(nc,ng)+\min(c,g)$ 在 $n=2$ 时的样子。

实际跑（`--scan`，默认 $w=0.5$、$S=132$、$u=10$、$v=8192$ B、$\text{BW}=400$ GB/s）：

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

（`ratio` 是 split/unsplit，小于 1 才是收益。）

**128–512：净亏，最差慢 1.69×**。128 个 token 的活不够一个 wave，`max(..., wave_unit)` 直接给一个 wave；切成两半，每半**还是**一个 wave，$2c(T/2) = 2c(T)$，计算时间凭空翻倍。**小 batch 下切分的第一笔开销，就是把 wave 数从 1 抬到 2**。

**1024–2048：刚好打平**（1.00×）。单个 batch 占 1 个 wave，切两半各占 1 个 wave。记 $c_h = c(T/2)$、$m_h = m(T/2)$，沿这一档把账拆开：wave 惩罚是 $c_h - c/2 = 10 - 5 = 5$，不付这笔惩罚时，能省下的是 $\min(c,m)/2 = 5$，两个 5 刚好抵消，净收益为零。

**4096 及以上：稳定收益**（0.90–0.93×）。转折点在这一档：$c(4096)$ 是 2 个 wave，$c(2048)$ 是 1 个 wave，$2c_h = c$，切分不再多付 wave，剩下的是纯赚。

把 4096 那一行拆开对账：

```text
c(4096) = 20.0   m(4096) = 83.9   unsplit = 20.0 + 83.9 = 103.9
c(2048) = 10.0   m(2048) = 41.9
split = max(2×10.0, 2×41.9) + min(10.0, 41.9)
      = max(20.0, 83.9) + 10.0 = 93.9          → 93.9 / 103.9 = 0.90x
```

$2c(2048)=20$ 和 $c(4096)=20$ 一模一样，计算侧完全免费；省下的正是"最后一个半 batch 的通信尾巴" $c(2048)=10$。再上一档 16384 为什么回落到 0.93×？因为 $c(16384)=70$（7 个 wave）而 $2c(8192)=80$（$2 \times 4$ 个 wave），又多付了一个 wave——只要 $T$ 落在波数不是偶数的位置，切分就要多付一次。

所以 threshold≈4096 不是拍出来的，是 wave 边界对齐的位置，只由 $w$、$S$、$u$ 决定，换硬件、换模型都会变。模型的价值是告诉你阈值存在、由什么决定，而不是阈值是 4096。

> **模型的边界**：这是解析模型，不是实测。它没有 kernel launch、没有 L2 命中率、没有 HBM 的 bank 冲突、没有 CUDA Graph 的省法。具体的数必须以你自己卡上的 trace 为准。

那 TokenWeave 为什么在 **1K 以上**就有 1.16–1.28× 的收益？模型缺两项，而它补的正好是这两项：一是 RMSNorm 那 **4–9%**，模型里没这一项，真实系统里它是一次 launch 加一次完整 HBM 往返，融进 AR 之后这个固定项直接消失；二是通信 kernel 的 SM 占用从 **16–20+ 降到 2–8**，模型假设两个 stream 用满 $S$ 个 SM，实际通信 kernel 占着 SM 时计算 kernel 的 wave 会变形，省下十几个 SM 等于把 $c(T)$ 的分母做大。

两项都属于"消掉切分自身的开销"，不是"把切分做得更聪明"——这也顺带解释了只切不融的朴素版本在 512–8K 反而更慢：模型里"多付一个 wave"的开销，没有人替它买单。

## 5. PD 分离：用跨机器的错峰换掉卡内 overlap

prefill 一次处理上千 token，算术强度高，是 compute-bound；decode 一次一个 token，是 memory-bound。放在同一张卡上，prefill 的长 GEMM 会把 decode 的 TPOT 顶上去，decode 的驻留又让 prefill 的 batch 上不去。

拆到不同机器，两类资源各归各的机器，各自按最优 batch 跑，中间只传一次 KV cache。代价是多一次 KV 的网络传输（长上下文时不小）、多一个请求路由的调度器、以及两个集群的配比要调——用[第一篇]({{< relref "comm-overlap-01-performance-model" >}})的话说，这不是把通信藏进计算，而是承认 prefill 的计算和 decode 的带宽不该共享同一份 SM 和 HBM。轻量的折中是 chunked prefill：长 prefill 切成块和 decode 混在一个 batch 里，保住单机部署的简单性，只把干扰的粒度从一次 prefill 缩到一个 chunk。

prefill 侧的通算并行仍然值得做，尤其 MoE。**INC-MoE**（[2610.36339](https://arxiv.org/abs/2610.36339)）把 all-gather / reduce-scatter 挪进交换机，让双向链路并发，省掉 GPU 侧的收集与转发——prefill 场景 TTFT **−33.1%**、吞吐 **+49.5%**，但 **$N=2$ 时收益为零**，按论文的 $(N-2)/(2(N-1))$ 分子为 0，ring 已经走到下界。细节归[第六篇]({{< relref "comm-overlap-06-moe" >}})。

## 6. 一张表，四种场景别用错行

| 场景 | 瓶颈资源 | 通信占比 | 首选动作与禁忌 |
|---|---|---|---|
| 训练（大 GEMM） | 算力 | 10–50% | 切 + 错：bucket、microbatch、1F1B、tile 融合；通信 <10% 时别动 SM 配额，overlap tax 会吃掉收益 |
| prefill（长序列） | 算力 | 20–47%（MoE 尤甚） | 融 + 让：AG+GEMM、in-network collective、SM 分区；batch 太小时 EP 的 all-to-all 无处可藏，先合 batch |
| decode（小 batch） | HBM 带宽 + 固定开销 | ~20% | 减体积 + 省 launch：低比特量化、AR-Norm 融合、少占 SM；别切 batch，wave 数会翻倍（阈值 4K） |
| decode（大 batch） | 从带宽转向算力 | 随 batch 下降 | 切 batch + 双流：wave-aware split；权重 I/O 没摊薄前别复制算子，翻倍成本会抵消收益 |

表里的通信占比是经验区间，不是某一篇的实测；用错行的代价是负收益。先看瓶颈资源，再决定动不动切分，同一行的动作和禁忌不要同时做。通信占比很低时先别碰 overlap——天花板就是那个占比，该做的是减通信量或者做计算优化。

## 只记一件事

训练侧的 overlap 在抢空闲算力，decode 侧的 overlap 在抢已经很紧的固定开销；前者几乎稳赚，后者必须先算阈值。

推理侧还有一条没展开的线：把 collective 本身做得更轻（低 SM、TMA 发起、persistent kernel），属于[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}})的 SM 分区话题。下一篇[第十篇]({{< relref "comm-overlap-10-measurement" >}})回到方法论：怎么用 nsys/ncu 把上面这些数真正测出来，以及那些看起来很美但会骗人的指标。

---

**系列导航**：[八：编译与自动调度]({{< relref "comm-overlap-08-compiler" >}}) · [十：实测手册]({{< relref "comm-overlap-10-measurement" >}})

> 配套代码：`code/comm-overlap/decode_overlap.py`
> 自测：`python3 decode_overlap.py --test && python3 decode_overlap.py --scan`
