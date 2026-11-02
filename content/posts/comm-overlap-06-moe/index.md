---
title: "通算并行研读（六）：MoE 专家并行——dispatch、expert、combine 的三段式"
date: 2026-11-02
draft: false
weight: 7
series: ["comm-overlap"]
tags: ["通算并行", "系列", "MoE", "专家并行", "DeepEP", "Comet", "all-to-all"]
categories: ["训练框架"]
---

MoE 层的通信不是"GEMM 中间夹了两段 all-to-all"这么简单。Comet（[2502.19811](https://arxiv.org/abs/2502.19811)）在 8×H800 上把一层 MoE 拆开看，通信占了 **47%** 的时间；Lancet（[2404.19429](https://arxiv.org/abs/2404.19429)）在另一套集群上测到，all-to-all 最高可达 expert compute 的 **3.36 倍**。

这两个数放在一起，指向同一个结论：**如果只把 all-to-all 和 expert GEMM 做重叠，你最多只能藏住"计算那么长"的一段通信；多出来的部分一定暴露。** 而 MoE 的通信偏偏经常比计算长。

这一篇按"三段式"往下拆：先讲清 dispatch → expert GEMM → combine 的依赖和分块，再看 DeepEP 的 API 怎么把 wait 推后，然后依次读 Comet、Weave、Lancet、DisagMoE 和 INC-MoE——它们分别代表层内、整层、整图、模块间、网络内五个不同的重叠层级。最后用 `moe_overlap.py` 把"通信 SM 预算 × chunk 数"这笔账算一遍。

## 1. 为什么 MoE 的通信不是边角料

先看一层 MoE 在 EP 下到底传了什么。每个 token 经 gate 选 top-k 个专家，token 要被送到专家所在的卡上：

$$
\text{dispatch bytes} \approx M \cdot k \cdot H \cdot 2 \quad \text{(BF16, one-way)}
$$

$M$ 是本卡 token 数，$k$ 是 top-k，$H$ 是 hidden。combine 再把专家输出按原路送回，量级相同。所以一层至少两次全互连交换，每次都被 top-k 放大。

三个因素让这件事比公式更糟：

1. **top-k 复制**。$k=2$ 就是两倍流量，$k=8$ 就是八倍。DeepSeek-V3 这类模型每 token 激活 8 个专家，dispatch 的有效字节数远大于"一份 hidden state"。
2. **routing skew**。专家热度不均，某张卡可能收到远超平均的 token。GEMM 可以按 expert 分组算，但一个 chunk 的完成时间由**最慢的那个专家**决定；为了对齐，系统还要按 capacity factor 补 padding，进一步放大流量和计算。
3. **combine 在关键路径的末端**。dispatch 没到，expert 一个 token 都算不了；combine 没回，下一层 attention 就没有输入。它不像 DP 的梯度 all-reduce 那样可以挪到 optimizer 前，也不像 TP 的 all-reduce 那样有相邻 GEMM 可以错位。

还有一个常被忽略的对比：**expert GEMM 其实很短**。单卡上每个专家分到的 token 不多，grouped GEMM 的 shape 又碎，算力利用率本来就不高；而 all-to-all 要跨节点走 IB，带宽比 NVLink 低一个数量级。Lancet 的 3.36 倍就是这么来的——**通信比计算长，只藏计算等于没藏**。

这也解释了后面几篇论文的演化路线：Comet 在层内把 token 和 tile 的粒度对齐；Weave 把整层融成一个 persistent kernel、动态分 SM；Lancet 把重叠范围扩大到整张训练图；DisagMoE 干脆把 attention 和 FFN 拆到两组 GPU 上；INC-MoE 则把 all-to-all 卸到交换机里。**它们解决的是同一个问题的不同层次。**

## 2. 三段式：D → C → B

把一层写成三个算子：

- **D（dispatch）**：按 routing 把 token 发到专家所在 rank，写成 all-to-all；
- **C（expert compute）**：本地专家做 grouped GEMM（gate/up → SiLU → down），按专家分组；
- **B（combine）**：把专家输出送回 token 原 rank，做 top-k 加权归约。

在实现上，dispatch 的输出通常按本地专家重新分组，形状是 `[num_local_experts, capacity, H]`；capacity 取该卡最热专家的 token 数，或按 grouped GEMM 的 M 维对齐。这意味着 **padding 的 token 也会占 GEMM 的行**：routing 越偏，padding 越多，计算越慢。DeepEP 的 `expert_alignment` 就是用来对齐这个 M 维的；Comet 则干脆不按固定 capacity 切，而是按实际收到的 token 重排 tile。

依赖是严格的链：$D_i \to C_i \to B_i$。不分块时，一层就是 $T = d + c + b$，通信完全没有藏身之处。

顺带回答一个常见疑问：**dispatch 和 combine 能不能合成一次 all-to-all？** 不能。dispatch 的输入是 token，combine 的输入是专家 GEMM 的输出，中间隔着整个 expert 计算，数据依赖不允许合并。能合并的是 combine 和"下一段 P2P"——DisagMoE 的 M2N 就是这么做的，把两次搬运压成一次。

分块之后，不同 chunk 之间相互独立，才有流水：

```text
comm  [D0][D1][D2][D3]    [B0]            [B1]            [B2]            [B3]
comp      [======C0======][======C1======][======C2======][======C3======]
      ↑ startup=d0                                                        ↑ tail=b3
```

图中 D/B 各占 1 个时间单位、C 占 4 个，$d=b=1$、$c=4$。读图注意五件事：

1. **C 是串行的**：同一条计算通道，$C_1$ 必须等 $C_0$ 算完，哪怕 $D_1$ 早就到了。所以 chunk 之间能重叠的只有"通信 vs 计算"，不是"计算 vs 计算"。
2. **$B_i$ 要等 $C_i$**：$B_0$ 在 $C_0$ 结束才发，$B_1$ 又晚 4 个单位；通信通道在两次 combine 之间是空的。
3. **空白就是 mid bubble**。分块把它从"整段 GEMM 都空着"（$K=1$）拆成"每个 chunk 边界空一小段"；空隙的总长度取决于 $c$ 和 $b$ 的比例，不可能降到 0。
4. **头部 $d_0$ 和尾部 $b_3$ 消不掉**。它们分别是一层的启动成本和收尾成本，只能靠更小的 chunk 摊薄，不能靠重叠消除——这就是 [第一篇]({{< relref "comm-overlap-01-performance-model" >}}) 里的 $T_{\text{startup}}$。
5. **chunk 不是越细越好**。chunk 越小，$d_0$ 和 $b_3$ 的尾巴越短；但每个 chunk 的 GEMM 行数变少，grouped GEMM 的效率下降，launch 和同步次数上升。Weave 实测 $K=8$ 时 GEMM 吞吐比 $K=1$ 低约 **27%**，所以它的最优 $K$ 通常只有 2–4。

在"通信被完全藏住"的理想情况下：

$$
T \approx \max\left(\sum_i c_i,\ \sum_i d_i + \sum_i b_i\right) + d_0 + b_{\text{last}}
$$

如果 $\sum d + \sum b > \sum c$，通信就是瓶颈，$T \approx \sum d + \sum b + c_0$，再怎么切也降不下去——这和 TP 的结论一样：**分块只能摊薄启动和尾部，不能把"通信比计算慢"这件事变没。**

## 3. DeepEP：把 wait 推后的 API

DeepEP 是目前 MoE 训练/推理最常用的 EP 通信库。本地仓库 `~/github/DeepEP` 在 commit `b5a33eef`，核心接口在 `deep_ep/buffers/ep.py`：**dispatch 在 `:562`，combine 在 `:770`**。

### 3.1 两段式调用

`dispatch` 把本卡 token 发出去，返回接收缓冲、routing 信息和 `handle`；`combine` 拿着同一个 `handle` 把专家输出送回来。`handle` 里记了这次 dispatch 的布局和 SM 数，所以 `combine` 的 `num_sms=0` 就是"复用 dispatch 的选择"。两者都有一组共同的调度参数：

```python
# deep_ep/buffers/ep.py:562（签名节选）
def dispatch(self, x, topk_idx=None, topk_weights=None, ...,
             num_sms: int = 0, num_qps: int = 0,
             previous_event: Optional[EventHandle] = None,
             async_with_compute_stream: bool = False,
             allocate_on_comm_stream: bool = False,
             handle: Optional[EPHandle] = None,
             defer_epilogue: bool = False):
    ...
```

这些参数各自解决一个"什么时候等"的问题：

| 参数 | 作用 | 什么时候用 |
|---|---|---|
| `num_sms` | 通信 kernel 占多少个 SM；0 表示用解析估计 | 想手动调 SM 预算时 |
| `num_qps` | RDMA queue pair 数；0 表示自动 | 跨节点、QP 成为瓶颈时 |
| `previous_event` | 执行前先等这个 event | 让 dispatch 等 gate/routing，或等上一块 buffer 释放 |
| `async_with_compute_stream` | 当前 stream **不等**通信 kernel 完成 | 双 stream 重叠的关键 |
| `allocate_on_comm_stream` | 输出 tensor 的归属记在通信 stream 上 | 配合 `previous_event`，避免跨 stream 隐式同步 |
| `defer_epilogue` | 把 CPU 侧的接收计数等待和拷贝 epilogue 推迟到 wait 时 | 想让 launch 尽早返回 |

`EventOverlap`（`deep_ep/utils/event.py`）是对 CUDA event 的薄封装。它提供 `current_stream_wait()` / `wait()`，也支持 `with event_overlap:` 语法：进入时什么都不做，退出时才让当前 stream 等这个 event。这样你可以把"发通信"和"用通信结果"拆成两段代码，中间插入任意独立计算。

### 3.2 重叠模式：先发射，需要时再等

最自然的写法是：

```python
# 伪代码：省略返回值解包和错误处理
recv_x, recv_idx, recv_w, handle, ev = buf.dispatch(
    x, topk_idx, topk_weights,
    num_sms=comm_sms,
    async_with_compute_stream=True,
)

# 这里可以干任何不依赖 recv_x 的活：
# shared expert 的 GEMM、上一层的残余计算、下一个 microbatch 的 attention……
do_independent_work()

ev.current_stream_wait()          # 只有 expert GEMM 真要读 recv_x 时才等
y = grouped_gemm(recv_x, recv_idx, ...)

ev2 = buf.combine(y, handle, topk_weights,
                  async_with_compute_stream=True,
                  defer_epilogue=True)
# combine 在通信 stream 上跑，当前 stream 可以继续做下一层的 norm/gate
ev2.current_stream_wait()
```

关键不是"用了异步 API"，而是**把 wait 放到最后一个可能的时刻**。`defer_epilogue=True` 进一步把 CPU 侧的计数等待也推迟，避免 host 线程被阻塞——这一点在 CUDA graph 或多 microbatch 场景下很重要。

### 3.3 `num_sms`：通信的 SM 预算

MoE 的 all-to-all 不是"网卡自己搬"，它要在 GPU 上跑 copy/notify kernel，占 SM 来发指令、轮询、做 reduction。README 写得很直白：

> EP dispatch and combine require GPU SMs; zero-SM RDMA EP is not supported.

（`~/github/DeepEP/README.md`）

也就是说，**MoE 的通信没有"零 SM"选项**（至少这一版没有）。`num_sms` 就是你从 GEMM 嘴里抢出来给通信的 SM 数。

`get_theoretical_num_sms`（`ep.py:431`）给了一个解析估计：结果对齐到 4 的倍数、至少 4 个；按流量放大 1.3 倍；如果不开 `prefer_overlap_with_compute`，下限抬到 64；最后封顶在设备 SM 数的一半。这个估计假设 gate 负载均衡，所以真实 skew 大时要手动加。

为什么它是一个**权衡旋钮**：

- 通信带宽随 SM 数增加，但会饱和。DeepEP 的 copy kernel 在 SM 足够多之后，瓶颈回到 NIC/NVLink，再加 SM 没有收益；
- GEMM 的算力随可用 SM 数**线性下降**。`total_sms - num_sms` 就是计算能用的部分；
- 所以最优的 $c$ 在"通信边际带宽增益 = GEMM 边际算力损失"的位置。Weave 测量过，单层的最优 $c$ 在不同 GPU 上从 **12 到 36** 不等，DeepEP 固定 20、Comet 按 iteration 选 18，都会错过某些卡的最优值。

`num_qps` 是另一个维度：直连模式下自动取 `min(num_sms, 9)`，混合模式取 `num_sms * 16 + 1`。QP 少了打不满 RDMA，多了会有 doorbell 和缓存开销——但那是跨节点才需要关心的参数，单节点 NVLink 上影响很小。

## 4. Comet：把 token 和 tile 的粒度对齐

Comet 要解决的是一个很具体的错位：**通信以 token 为单位，GEMM 以 tile 为单位**。一个 128×128 的 GEMM tile 可能需要来自多张卡的 128 个 token；只要有一个 token 没到，整个 tile 就得等。

### 4.1 shared tensor 的依赖分析

Comet 把 producer 和 consumer 之间的缓冲区叫 **shared tensor**，然后按 consumer 的访问模式决定能沿哪个维度切：

- **layer0（dispatch → GEMM0）**：shared tensor 形状是 $(M \times topk, N)$，consumer 是 GEMM。token 之间沿 $M$ 维独立，所以沿 **token 维**切；切完按来源 rank 排序，让本地 token 的 tile 先算，远程 token 的传输同时进行。
- **layer1（GEMM1 → combine）**：consumer 里有一次 top-k 归约，沿 $M$ 维把 token 加在一起，所以 $M$ 维不能切；只能沿 **$N$（hidden）维**切。于是 Comet 把 GroupGEMM 改成**按列执行**：前 $T_N$ 列一算完，combine 就可以开始，而不必等所有专家算完。

这就是"重排算子内执行顺序"的含义：不是简单地流水线化两个算子，而是**改变算子内部的 tile 顺序**，让下游能更早启动。

### 4.2 一个融合 kernel + CTA 特化

切完之后，Comet 把通信和计算放进**融合 kernel**，并用 **thread block specialization** 隔离两类工作：

- producer CTA：负责通信，读/写 shared tensor、发 NVSHMEM 请求；
- consumer CTA：跑 CUTLASS 的 GEMM，warp 内部分工不变（producer warp 用 TMA 搬数据，consumer warp 发 MMA）。

整层按两段 pipeline 各一个 fused kernel（dispatch+GEMM0、GEMM1+combine），每个 SM 放一个 CTA，**总 CTA 数等于 Hopper 的 132 个 SM**。然后在 kernel 内按当前 shape 自适应地分 $n_p$ 个通信 CTA 和 $n_c$ 个计算 CTA：TP=8、$M$ 从 4096 涨到 16384 时，最优 $n_c$ 从 18 变到 26；TP 从 8 降到 4 时，$n_c$ 又从 26 变到 46。

效果（论文口径）：

- MoE 层通信占 **47%**；
- 单层加速 **1.96×**，端到端 **1.71×**（Mixtral-8x7B、Qwen2-MoE、Phi3.5-MoE 平均）；
- 藏住 **86.5%** 的通信，FasterMoE 是 **29.2%**、Tutel 是 **68.6%**；
- EP=8、TP=1 下随输入长度变化，单层加速 **1.28×–2.37×**；测试平台是 8×H800，另有 L20 集群。

### 4.3 代价

- 每个 GPU 要一块 NVSHMEM persistent buffer，大小约 **$2MN$ 字节**（BF16 下的 $M \times N$ 个元素）。$M=16384$、$N=4096$ 时是 128 MB 量级，不算小。
- 通信 warp 和计算 warp **不能塞进同一个 block**：warp/thread 数量上限会限制通信打满带宽，而且会干扰计算 warp。所以必须 CTA 级隔离，这也意味着 SM 一旦分给通信，就不能同时跑 GEMM。
- GEMM 侧依赖 Hopper 的 CUTLASS 实现；换架构要换 compute CTA 的代码。
- 每个 iteration 调一次 $n_p/n_c$，但**层与层之间不重分**，所以它只是"粗粒度动态"。

## 5. Weave：整层一个 megakernel

Weave（[2609.21483](https://arxiv.org/abs/2609.21483)）的出发点是 Comet 留下的两个洞：**空间上**，SM 划分按 iteration 调，层与层、卡与卡之间不一致；**时间上**，GEMM 中段仍有 mid bubble，通信 SM 整段空闲。

### 5.1 一个 kernel 装下五个算子

Weave 把 **dispatch、GEMM0、SiLU、GEMM1、combine** 全部融进一个 persistent megakernel，grid 大小就是 SM 数（H100 上 132）。每个 block 在启动时按 `blockIdx.x` 决定角色：前 $c$ 个是 comm SM，后面的是 comp SM。

关键在于**调度时机**：routing 一结束，这一层的通信量和计算量就完全确定了。Weave 在 megakernel 的 prologue 里跑一个轻量 cost model，联合选择：

- 通信 SM 数 $c$；
- chunk 数 $K$；
- 通信 SM 在空闲窗口里"偷"多少 GEMM tile（$n_{\text{steal}}$）。

### 5.2 chunk pipeline 与 bubble stealing

时间调度做两件事：

1. **chunk pipeline**：把 GEMM0、SiLU、GEMM1、combine 切成 $K$ 块（dispatch 不切），让 chunk $i$ 的 combine 落在 chunk $i+1$ 的 GEMM 中间。这样通信窗口从层的两头扩展到整个计算段。
2. **bubble stealing**：dispatch 结束后、chunked combine 开始前，comm SM 还有一段空闲；让它们去偷 GEMM tile，把空闲算力换成有效吞吐。偷 tile 的位置是**连续**的一段，而不是散落在每个 chunk 之间——否则角色切换会把 GEMM 和 combine 反复打断。

最后一个 chunk 的 combine 让**所有 SM** 参与，消掉尾部。

### 5.3 数字与边界

测试平台是 **4×H100 SXM5**：每卡 132 SM、BF16 峰值 **989 TFLOPS**、NVLink 单向 **450 GB/s**；EP=4，6 个主流 MoE 模型，序列长度 2k/4k/8k。

- 单层几何平均 **2.89×**，端到端 **1.33×**；
- overlap ratio **47.1%**，Comet **14.3%**、DeepEP **9.2%**、PK **5.8%**、TD **5.1%**；
- cost model 选出的 $(c,K)$ 与穷举最优的平均差距 **8.2%**，在线开销 **0.54 μs**；
- 消融：空间调度（S）平均降 **10.1%**，时间调度（T）再降 **14.1%**，合计 **22.8%**。

边界也很清楚：**单节点 4×H100、EP=4**，没有跨节点 RDMA；端到端加速被 attention 稀释，序列越长越明显（Figure 10 第二行里灰色 attention 占比很大）。另外 $K$ 不能太大：$K=8$ 时 GEMM 吞吐掉约 27%，cost model 必须把这项修正算进去。

## 6. Lancet：把重叠扩大到整张图

Lancet 的问题意识来自前面那个 3.36 倍：**只藏 expert compute 不够，因为 all-to-all 比它长。** 那能不能用 MoE 层之外的计算来填？

### 6.1 前向：切非 MoE 计算

Lancet 在编译期拿到整张训练图的 IR，把 MoE 层之外的算子（attention 的 QKV/MLP 等）沿 **batch 维**切成若干 partition，再和 all-to-all 做流水。切多深、切几段，用动态规划搜：目标是最小化迭代时间，约束是 partition 开销和显存。论文里观察到**最优 partition 数从不超过 4**。

### 6.2 反向：用 dW 填 A2A

反向的 all-to-all 更难藏，因为它的下游是 dX（要继续往更前面传），而上游是 dY。Lancet 发现 **dW（weight gradient）不依赖 A2A 的完成**：它只需要本层的输入激活和 dY，可以先算。于是用一个 best-fit 贪心，把 dW 算子分配到各个 all-to-all 的空隙里，让 dW 的总时长尽量贴近 A2A 的时长。

代价是切分会增加总计算量（Figure 13：partition 有额外开销，overlapped + non-overlapped 的计算时间加起来可能比不切还高），但通信降得更多，净收益仍为正。

### 6.3 数字与边界

- 非重叠通信最多降 **77%**（V100 对 Tutel；A100 上 66%）；
- 端到端最高 **1.3×**（V100 + Switch gate）、**1.21×**（A100），平均 **1.22×/1.17×**；
- cost model 误差 **3.83%**；
- 测试：GPT2-S/L-MoE，8 节点 × 8 张 A100 或 V100，每 GPU 2 个专家。

边界：**最优 partition 数不超过 4**，说明切分本身有不可忽略的固定成本；**batch-prioritized gating 会把流水范围限制在 MoE 层之后**，重叠窗口变小（虽然论文说加速仍接近）；它是编译期方案，图变了要重新优化，论文里大多数模型的优化时间在 20 分钟以内。

## 7. DisagMoE：把 attention 和 FFN 拆到两组 GPU

前面几个方案都在"同一组 GPU 上，让通信和计算错开"。DisagMoE（[2605.11005](https://arxiv.org/abs/2605.11005)）换了个思路：**既然 attention 和 FFN 的 compute/communication 比不同，为什么让它们抢同一批 SM？**

### 7.1 A/F 分组与 AF-Pipe

- **A-Worker**：放 attention，组内用 **DP** 复制；
- **F-Worker**：放 FFN/MoE 专家，组内用 **EP** 切分；
- 两组之间用 **M2N/N2M** 的 many-to-many 交换，而不是传统的 all-to-all + P2P。

AF-Pipe 把这次 M2N 交换当成**流水线的一级**，和 attention、FFN 的计算平级排布。传统做法里，FFN 要先 combine 再发 P2P，数据被搬两遍；AF-Pipe 把 combine 融进 M2N，通信成本大约降到 $1/k$。三个 stream（forward/backward/communication）异步跑，稳态下 A 组和 F 组都在算。

### 7.2 roofline 分配资源

DisagMoE 用一个 compute-communication roofline 同时决定：

- A 组和 F 组各多少 GPU；
- NIC 带宽怎么在两组之间分。

因为 attention 的算术强度随序列长度增长（$O(s^2)$），FFN 和通信是线性的（$O(s)$），所以最优比例随 $s$ 变。16K 序列下最优是 **A:F = 16:10**，比同构的 16:16 快 **1.56×**；4K 下则接近 16:16。

### 7.3 数字与边界

- 相比 Megatron-1F1B，**1.59×–1.81×**；
- 相比 SOTA MoE overlap（Tutel/Comet），**1.2×–1.5×**；相比 DualPipe，**1.05×–1.13×**；
- 非重叠通信：比 Tutel 少 **88%**、比 Comet 少 **75%**、比 DualPipe 少 **45%**；
- 128×H800（16 节点 × 8），序列 4K–32K，EP=16；all-to-all 占比从单节点 **22%** 涨到 8 节点 **78%**。

边界：**静态分配**——A:F 比例和 NIC 划分是按已知的序列长度、batch、EP 选好的，workload 一变就要重新规划；warmup/cool-down 的气泡仍然存在，约为基线的 **1/4**（AF-Pipe 的公式里 $(PP-1)$ 项还在）。它适合"形状稳定的大规模预训练"，不适合请求长度剧烈波动的 serving。

## 8. INC-MoE：把 all-to-all 卸进交换机

INC-MoE（[2610.36339](https://arxiv.org/abs/2610.36339)）针对的是**推理 prefill、跨节点 scale-out**：这时 dispatch/combine 可以写成 AllGather + ReduceScatter，而交换机天生就会 multicast 和 reduction。

- **dispatch = in-network AllGather**：每个 rank 把自己的 token 发给交换机，交换机 multicast 给其余 $N-1$ 个 rank；
- **combine = in-network ReduceScatter**：交换机先把 $N-1$ 份非 root 贡献 reduce 掉，root 只做一次本地加法。

妙处在于**方向**：AG 主要吃接收方向，RS 主要吃发送方向，两者同时跑就能把全双工链路吃满。带宽受限时间从 ring 的 $\frac{2(N-1)}{N}\cdot\frac{S}{BW}$ 降到 $\frac{S}{BW}$，提升为 $\frac{N-2}{2(N-1)}$——$N=4$ 时 33.3%，$N=8$ 时 42.9%，$N=2$ 时**为零**。

kernel 侧是一个 persistent kernel，把 token 切成 band，三类 CTA 各司其职：gather block 发 AG、compute block 跑 expert FFN、reduce block 发 RS 并做本地归约。稳态下 band $j+1$ 的 AG、band $j$ 的 FFN、band $j-1$ 的 RS 同时在飞；drain 阶段 gather block 还会去帮 reduce（tail assistance）。

数字（OLMoE-1B-7B prefill，4-way DP attention + 4-way EP，4 节点模拟）：

- TTFT 降 **20.2%**（24K token）到 **33.1%**（126K token）；
- 吞吐涨 **25.3%–49.5%**；
- 并发 AG+RS 的等效带宽 **163–172 Gbps**，ring 约 **130 Gbps**，达到理论上界的 **82%–86%**。

边界：$N=2$ 没有收益；只有理论上界的 82–86%；**交换机是拿一块 accelerator 模拟的**，真实硬件上小消息的固定开销会小很多，但也还没在真交换机上验证。它是 inference-only 的方案，训练侧不能直接套。

## 9. `moe_overlap.py`：把 SM 预算和 chunk 数一起扫

`code/comm-overlap/moe_overlap.py` 是一个纯 CPU 的解析模型。它把一层 MoE 建成两台"机器"：

- **一条通信通道**：dispatch 和 combine 共用，D 优先于 B；
- **一条计算通道**：grouped GEMM，可用 SM 数是 `total_sms - comm_sms`；
- 每个 chunk 的依赖是 $D_i \to C_i \to B_i$，用 list scheduling 求 makespan。

命令行用 `--comm-sms` 指定通信 SM 数、`--chunks` 指定块数；`--sweep` 会把两者交叉扫一遍。默认参数：`tokens=16384`、`hidden=4096`、`topk=2`、`inter=14336`、`ep=8`、`total_sms=132`、`bw_gbps=400`、`tflops=700`、`sm_sat=16`。通信时间按 $min(1, \text{comm\_sms}/\text{sm\_sat})$ 打折，GEMM 时间按可用 SM 数线性缩放。

```text
$ python3 moe_overlap.py --sweep
comm_sms C=1      C=2      C=4      C=8      C=16
       4  2797.1  2461.6  2293.8  2209.9  2168.0
       8  2530.1  2362.4  2278.5  2236.5  2215.6
      12  2491.4  2379.6  2323.7  2295.7  2281.7
      16  2513.7  2429.8  2387.9  2366.9  2356.4
      20  2597.5  2513.6  2471.7  2450.7  2440.2
      24  2687.5  2603.6  2561.7  2540.7  2530.2
      32  2889.1  2805.2  2763.2  2742.3  2731.8
      48  3407.4  3323.5  3281.6  3260.6  3250.1
      64  4169.7  4085.8  4043.8  4022.9  4012.4
```

### 9.1 为什么最优在 `comm_sms=4` 附近

在这组参数下：

- 每卡 token 数 $16384/8=2048$，乘 top-k 2 得 4096 行；
- 每行 $4096 \times 2 = 8192$ 字节，dispatch 总量约 33.6 MB；
- GEMM 总量约 $4096 \times 6 \times 4096 \times 14336 \approx 1.44$ TFLOP。

$c=4$ 时通信效率是 $4/16=25\%$，有效带宽 100 GB/s；16 个 chunk 下每块 dispatch 约 21 μs，32 次传输共约 **671 μs**。同一组参数下 GEMM 总计约 **2126 μs**。**通信总量只有计算的三分之一，4 个 SM 就够了**——再多的 SM 只是从 GEMM 嘴里抢人。

这就是表里的形状：

- $c=8$：通信时间减半，但 GEMM 少了 4 个 SM，计算总量从 2126 涨到 2195，净结果反而慢 48 μs；
- $c=2$：通信更慢，暴露的 startup 从 42 μs 翻到 84 μs，也变慢；
- 底部很平，$c=4$ 是这张偶数网格上的最优点（DeepEP 的估计器同样把 `num_sms` 对齐到 4 的倍数、下限 4，算是巧合的呼应）。

### 9.2 为什么 chunk 越多越好——以及为什么现实中不是

表里每一行都是 $C$ 越大越好，$C=1 \to 16$ 能降 600 多 μs。原因很直接：**模型里没有 per-chunk 开销**。切得越细，头部 $d_0$ 和尾部 $b_{\text{last}}$ 越小，而 GEMM 总量不变。

真实系统不是这样：

- 每个 chunk 一次 kernel launch + event 同步，GPU 侧几 μs、CPU 侧可能更多；
- chunk 变小后 grouped GEMM 的行数变少，tile 利用率下降（Weave 测到 $K=8$ 掉 27%）；
- chunk 边界要对齐 expert 的 token 布局，太细会引入额外的 permute/padding。

所以真实的最优 $K$ 通常在 **2–4**（Weave 图示里 $K=2$、Lancet 的最优 partition 不超过 4），而不是表里的 16。**这张表的价值是告诉你 $(c,K)$ 是联合的、底部很平；具体数值必须用真实 trace 校准。**

> 必须说清楚：这是**解析模型，不是实测**。带宽、算力、SM 饱和点都是假设值；它算不出 routing skew、L2 争用、NVSHMEM 的延迟。论文里的数字才是有硬件口径的实测。

## 10. 六个方案放在一起

| 系统 | 并行/场景 | 粒度 | 关键手法 | 公开数字 | 主要代价 |
|---|---|---|---|---|---|
| **DeepEP** | 训练/推理，EP | chunk + kernel | GPU-initiated A2A；`num_sms`/`num_qps` 可配；`EventOverlap` | 工业基线；SM 可配 | 必须占 SM；chunk 与 wait 要手工排 |
| **Comet** | 单节点，EP+TP | token/tile | shared tensor 切分 + 算子内重排 + CTA 特化 | 通信 47%；层 1.96×；e2e 1.71×；藏 86.5% | NVSHMEM buffer $2MN$；绑 CUTLASS/Hopper；两段 kernel |
| **Weave** | 单节点，EP=4 | tile + chunk | 整层 megakernel；routing 后选 $(c,K)$；bubble stealing | 层 2.89×；e2e 1.33×；overlap 47.1% | 单节点 4 卡；e2e 被 attention 稀释；$K$ 有上限 |
| **Lancet** | 多节点训练 | 整图/partition | 非 MoE 计算沿 batch 切分流水；反向 dW 填 A2A | 非重叠通信 −77%；e2e 1.3× | 切分增计算；partition ≤4；编译期优化 |
| **DisagMoE** | 多节点训练 | 模块间 | A/F 分组；M2N 作为流水级；roofline 分 GPU/NIC | 1.81× vs 1F1B；1.34× vs SOTA | 静态分配；warmup 气泡 ~1/4 |
| **INC-MoE** | 推理 prefill | band | 交换机内 AG/RS；persistent kernel；三类 CTA | TTFT −33.1%；吞吐 +49.5% | 需 INC 交换机；$N=2$ 无收益；模拟硬件 |

一张表读下来，抽象层级很清楚：**DeepEP 给你积木，Comet/Weave 在层内搭，Lancet 在整图搭，DisagMoE 在模块之间搭，INC-MoE 把积木搬到网络里。** 越往下（层内）越通用，越往上（网络内）收益越大但依赖越强。

如果按"改动发生在哪"来分，这六个方案其实只做了三件事：**减字节**（FP8 dispatch、不传 padding token）、**减跳数**（尽量在 NVLink domain 内解决、把 collective 卸进交换机）、**减暴露**（切分、错位、SM 分区）。DeepEP 和 Comet/Weave 主要在第三类；DisagMoE 和 INC-MoE 同时动了前两类；Lancet 用"别处的计算"换暴露，是第三类里最激进的一种。**先判断你的瓶颈在哪一类，再选方案**，比对着 speedup 数字选更可靠。

## 11. 怎么选：五个问题

1. **通信占这层多少时间？** 只占一成左右，先别做复杂 overlap，调 `num_sms` 和 chunk 就够；接近一半（Comet 的 47%），才值得上融合/分组。
2. **A2A 是 expert compute 的几倍？** 如果超过 1（Lancet 测到 3.36×），只藏计算一定不够，要考虑整图（Lancet）或分组（DisagMoE）。
3. **单节点还是跨节点？** 单节点 NVLink 上 Comet/Weave 的收益最确定；跨节点 IB 上，通信占比随节点数从 22% 涨到 78%，这时 DisagMoE 或 INC-MoE 的"减少通信本身"比"藏通信"更有效。
4. **形状稳定吗？** 稳定（预训练）可以用静态分配；波动（serving）要用 Weave 这种按 routing 动态决策的方案。
5. **愿意付多少工程代价？** Comet/Weave 要改 kernel 和 GEMM 后端；Lancet 要编译器；DisagMoE 要改并行策略；INC-MoE 要交换机。DeepEP 是唯一"今天就能用"的基线。

几个容易踩的坑：

- **只报 overlap ratio**。Weave 的 47.1% 和 Comet 的 14.3% 口径不同（前者把整个 MoE 窗口算进去），跨论文比较要先对齐定义；真正的指标还是 exposed time 和端到端。
- **忽略 routing skew**。模型里 `--skew 1.0` 是均衡假设；真实训练里热点专家会让某个 chunk 明显慢于平均，overlap 的收益会被吃掉。
- **忽略 combine 的 tail**。分块能把 tail 降到 $b/K$，但降不到 0；最后一个 chunk 的 combine 永远在关键路径上。

## 结论

- MoE 层的骨架是 **dispatch → expert GEMM → combine** 的三段式；通信占 47%（Comet），A2A 可达 expert compute 的 3.36×（Lancet），所以"只藏计算"经常不够。
- **分块是重叠的前提**：$D_i \to C_i \to B_i$ 的链只有切开才能让 $D_{i+1}$ 与 $C_i$ 并发；但 $d_0$ 和 $b_{\text{last}}$ 必然暴露，chunk 太细还会降低 GEMM 效率。
- **DeepEP 提供的是"把 wait 推后"的积木**：`async_with_compute_stream`、`previous_event`、`defer_epilogue`、`EventOverlap`，加上 `num_sms` 这个 SM 预算旋钮；它要求通信占 SM，零 SM 的 RDMA EP 在这一版不支持。
- **Comet 对齐 token/tile 粒度**（shared tensor 切分 + CTA 特化），**Weave 把整层融成一个 megakernel 并动态选 $(c,K)$**，**Lancet 把重叠扩到整图**，**DisagMoE 把 A/F 拆开**，**INC-MoE 把 A2A 卸进交换机**——收益依次变大，适用条件依次变窄。
- 本地模型显示：在这组参数下 `comm_sms=4` 最优、chunk 越多越好；真实系统有 launch/同步/GEMM 效率开销，最优的 $K$ 会明显变小，$c$ 也要用真实带宽/算力曲线重扫。**模型用来建立直觉，数字以论文和实测为准。**

SM 怎么分、persistent kernel 怎么写、warp specialization 怎么摆，是下一层的问题。[第七篇]({{< relref "comm-overlap-07-kernel-sm" >}}) 会把 Comet 和 Weave 的 CTA 分区拆到 kernel 级；如果你更关心推理侧的阈值，可以直接跳到[第九篇]({{< relref "comm-overlap-09-inference" >}})；整张地图和五个开放问题在[第十一篇]({{< relref "comm-overlap-11-summary" >}}) 收尾。TP 侧把 all-gather 融进 GEMM 的做法，可以和这里的 dispatch+GEMM0 对照着看：[第五篇]({{< relref "comm-overlap-05-tensor-parallel" >}})。

---

**系列导航**：[五：张量并行]({{< relref "comm-overlap-05-tensor-parallel" >}}) · [七：Kernel 与 SM]({{< relref "comm-overlap-07-kernel-sm" >}})

> 配套代码：`code/comm-overlap/moe_overlap.py`
> 自测：`python3 moe_overlap.py --test`
> 扫描：`python3 moe_overlap.py --sweep`
