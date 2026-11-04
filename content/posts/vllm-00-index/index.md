---
title: "vLLM 研读（专题）：推理引擎的四个账本"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 1
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "PagedAttention", "调度", "KV Cache"]
categories: ["推理框架"]
---

一个 8 卡 H100 的推理服务，200 个请求同时在线：有的 prompt 是 200 token 的问候，有的是 30k token 的代码库；有的已经生成到第 500 个 token，有的刚做完 tokenize；客户端全部开着流式，每 20 ms 就期待一个新 token。

框架在每个 step 都要回答同一个问题：**这一步，用哪张卡、算哪些请求、各算多少 token、KV 放哪、算完的 token 什么时候发回去？**

这个问题没有一劳永逸的答案，因为约束之间互相打架：把 batch 拼大能提吞吐，但会抬高尾延迟；把长 prompt 一次算完 TTFT 好看，但会卡住所有人的 decode；把 KV 缓存全留下来做前缀复用，显存就不够接新请求。所谓"推理框架"，本质上是在这四个约束之间做**在线、每步、毫秒级**的再平衡。

vLLM 是这套再平衡做得最彻底的一个实现。这个专题把它拆开看。

## 先记住一个画面

想象一块 80 GB 的 H800，上面跑 Llama-3-8B（bf16 权重约 16 GB），`gpu_memory_utilization=0.9`。剩下约 54 GB 给 KV cache。

Llama-3-8B 的 KV 是 32 层、8 个 KV head、head_dim 128、bf16：

$$
2 \times 32 \times 8 \times 128 \times 2 = 131072\ \text{bytes} = 128\ \text{KiB/token}
$$

也就是说，**一个 8k 上下文的请求，KV 就要 1 GiB**。如果按"每个请求预留 max_model_len 大小的连续显存"来分配，54 GB 只够 **54 个请求**——哪怕它们实际只用了 500 token。

vLLM 的做法是把 KV 切成固定大小的 block（默认 16 token 一块，2 MiB），按需分配、用块表（block table）把逻辑位置映射到物理块。54 GB 变成 **约 2.7 万个块**，平均 500 token 的请求只占 32 块；同样的显存能装下几百个请求。这不是"省一点内存"，是把并发数提高了一个数量级。

而一旦 KV 变成非连续的块，attention kernel 就不能再假设"第 $i$ 个 token 在基址 + $i \times \text{stride}$"，必须查表。这条链路上的每一个设计——块表、slot mapping、prefix cache 的哈希、抢占时的重算——都是从这一个决定长出来的。

> **类比的边界**：vLLM 常被比作操作系统——block 是页、block table 是页表、scheduler 是进程调度器。但有一处关键不同：操作系统缺页可以把页换到磁盘，vLLM V1 的抢占是**直接丢弃 KV、之后重算**，没有 swap。所以"内存不够"在 vLLM 里表现为"重复计算"，而不是"变慢一点"。

## 三个问题

这个专题围绕三个问题展开：

1. **显存怎么不被碎片吃掉？** 块怎么分、块表怎么建、前缀缓存怎么在不改变输出的前提下复用块、混合模型（sliding window / Mamba）的块大小怎么统一。
2. **batch 怎么拼？** 一个 step 的 token budget 怎么分给 prefill 和 decode、长 prompt 怎么切块、KV 不够时抢占谁、异步调度如何让 CPU 准备和 GPU 计算重叠。
3. **正确性怎么保证？** 前缀缓存命中不能改变 logits、投机解码的输出分布必须与目标模型一致、量化后的误差要可解释。性能优化只要碰了这三条，就必须证明"等价"。

## 四个账本

### 账本一：显存

推理服务的显存分四块：权重、KV cache、激活/workspace、CUDA Graph 与通信缓冲。权重是常数，KV 是唯一随负载线性增长的部分，也是调参的主战场。

以 8B 模型、8k 上下文为例：

| 项目 | 量级 | 是否随负载变 |
|---|---|---|
| 权重（bf16） | 16 GB | 否 |
| KV cache（54 GB） | 128 KiB/token，约 44 万 token | 是，随并发 × 长度 |
| 激活 / workspace | 数 GB（随 max_num_batched_tokens 变） | 弱相关 |
| CUDA Graph | 数 GB（随 capture size 数量变） | 否 |
| NCCL buffer | 数百 MB | 否 |

这张表解释了为什么 `max_model_len` 设得越大，能同时服务的请求反而越少：它不直接占显存，但它决定了单请求 KV 的上限和调度器的准入判断。

### 账本二：时延

在线服务看两个数：

- **TTFT**（time to first token）：排队 + prefill。8k prompt 在 8B 模型上约 0.2–0.3 s（按 2·N·P FLOPs 估，约 131 TFLOP，H100 有效算力几百 TFLOPS）。
- **ITL**（inter-token latency）：一次 decode step 的墙钟时间。batch=1 时被权重带宽卡住（16 GB / 3 TB/s ≈ 5 ms），batch 大了变成算力/带宽混合约束。

两者由同一个 token budget 分配。chunked prefill 的本质是**把 prefill 切成小块混进 decode 步**，用 TTFT 换 ITL 的稳定性；`max_num_batched_tokens` 就是这场交易的价码。

### 账本三：吞吐

吞吐 = 每秒 token 数，约等于 `batch 内 token 数 / step 时间`。continuous batching 让 batch 随请求到达/结束动态变化，调度器每一步重新装箱。

装箱的约束不止 token 数：还有 `max_num_seqs`（请求数上限）、KV 块数、以及每个请求"这一步最多能推进多少 token"（受 max_model_len 和投机 token 数限制）。vLLM 的调度器不是简单的 FIFO，而是一个带预算的在线装箱器，末尾还要处理"装不下时抢占谁"。

### 账本四：正确性

前三个账本都是性能，第四个是底线：

- prefix cache 用 block 哈希复用 KV，命中与否必须不改变输出（只缓存完整块、哈希包含父块与额外信息，就是为了这个）。
- 投机解码用拒绝采样保证输出分布与目标模型一致，而不是"看起来差不多"。
- 量化改变数值，但必须明确哪些算子被量化、误差如何累积；fp8 KV cache 不是"免费减半"。

这个专题会在每篇里标出：这项优化**改变了什么、没改变什么**。

## vLLM 的答案：V1 架构地图

当前主线（v0.33 之后的 main）里 V0 已被完全移除，V1 是唯一引擎。它的进程模型是理解一切代码的起点：

| 进程 | 数量 | 职责 |
|---|---|---|
| API Server | 随 LB 模式：内部 LB = DP，多端口/外部 LB = 1（`--api-server-count`） | HTTP、tokenize、多模态加载、流式回包 |
| EngineCore | 每个 DP rank 一个 | 调度、KV 管理、驱动 worker |
| GPU Worker | 每卡一个（DP×PP×TP） | 加载权重、执行 forward、管理显存 |
| DP Coordinator | DP>1 时 1 个 | DP 间负载均衡、MoE 同步前向 |

举例：单机 4 卡 `-tp=4` → 1 + 1 + 4 = **6 个进程**；8 卡 `-tp=2 -dp=4` → 4 + 4 + 8 + 1 = **17 个进程**。这个表直接决定了你要给容器留多少 CPU 核（每个 API server 还有媒体加载线程）。

进程之间用 ZMQ 通信：前端 bind 一个 ROUTER（收请求）和一个 PULL（收输出），EngineCore 用 DEALER/PUSH 连上来，消息用 msgspec 的 msgpack 编码，多模态张量走共享内存旁路。这条链路在[请求的一生]({{< relref "vllm-01-request-lifecycle" >}})这篇会逐段走。

## 2026 年的 vLLM 与 2024 年那版有什么不同

如果你读的是两年前的 vLLM 源码笔记，下面这些已经变了：

- **V1 成为唯一引擎**，调度器不再区分 prefill/decode 阶段，只看 `num_computed_tokens` 追没追上 `num_tokens`。
- **异步调度默认开启**：scheduler 准备第 N+1 步时，GPU 还在跑第 N 步，用 placeholder token 处理"输出还没回来"的空窗。
- **CUDA Graph 与编译解耦**：`cudagraph_mode` 有五种，默认 `FULL_AND_PIECEWISE`（uniform decode 走 full graph，prefill/mixed 走 piecewise）。
- **实验性的 Model Runner V2**（`vllm/v1/worker/gpu/model_runner.py`）：把持久状态与每步输入解耦、去掉 async barrier，是下一代执行层的方向。
- **MoE 的 DBO**（Dual Batch Overlap）：DP+EP 下把 batch 切两半，用两个线程让 all-to-all 与计算重叠。
- **P/D 分离与 KV connector** 成为一等公民：NIXL、Mooncake、LMCache、offload connector 都在主仓里。
- **量化**：fp8 成为常规选项，mxfp4/nvfp4 进入主线，KV cache 也能 fp8。

所以这个专题以**当前 main 的代码**为准，行号会随版本漂移，但机制和接口是稳定的。

## 三条阅读路线

- **系统线（推荐先读）**：01 请求的一生 → 02 PagedAttention → 03 前缀缓存 → 04 调度器 → 05 执行侧。读完能回答"一个请求从进到出，经过了哪些数据结构"。
- **算子/编译线**：05 执行侧 → 06 Attention 后端 → 07 torch.compile。读完能判断"为什么我的模型在这个后端上慢"。
- **能力/部署线**：08 采样 → 09 投机解码 → 10 分布式 → 11 KV 传输 → 12 量化与调优。读完能做部署选型和参数调优。

## 系列目录

| # | 文章 | 一句话 |
|---|---|---|
| 0 | 本篇 | 四个账本与架构地图 |
| 1 | [请求的一生：从 HTTP JSON 到 token]({{< relref "vllm-01-request-lifecycle" >}}) | 进程、ZMQ、输入/输出处理 |
| 2 | [PagedAttention：块、块表与 KV 显存账本]({{< relref "vllm-02-paged-attention" >}}) | 分页 KV 的数据结构 |
| 3 | [前缀缓存：一条哈希链省下多少 prefill]({{< relref "vllm-03-prefix-caching" >}}) | 哈希、引用计数、淘汰 |
| 4 | [调度器：token budget、chunked prefill 与抢占]({{< relref "vllm-04-scheduler" >}}) | 每步怎么装箱 |
| 5 | [执行侧：InputBatch、attention metadata 与 CUDA Graph]({{< relref "vllm-05-execution" >}}) | 调度结果怎么变成 GPU 输入 |
| 6 | [Attention 后端：FlashAttention、FlashInfer 与 MLA 怎么选]({{< relref "vllm-06-attention-backends" >}}) | backend 选择与 MLA |
| 7 | [torch.compile：切分、融合与编译预算]({{< relref "vllm-07-torch-compile" >}}) | 编译级别与融合 |
| 8 | [采样：logits 处理、logprobs 与结构化输出]({{< relref "vllm-08-sampling" >}}) | 从 logits 到 token |
| 9 | [投机解码：n-gram、EAGLE 与拒绝采样]({{< relref "vllm-09-spec-decode" >}}) | 用 draft 换吞吐 |
| 10 | [分布式：TP/PP/DP/EP 与 Executor 的进程账]({{< relref "vllm-10-distributed" >}}) | 多卡怎么切 |
| 11 | [KV 传输：connector、P/D 分离与 offload]({{< relref "vllm-11-kv-transfer" >}}) | 把 KV 搬来搬去 |
| 12 | [量化、显存与调优：把模型塞进去并跑满]({{< relref "vllm-12-quantization-tuning" >}}) | 显存账本与调参手册 |

配套代码在 [`code/vllm/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/vllm)：四个纯 CPU 脚本，分别模拟块分配、前缀缓存、调度装箱和显存预算，每个都能 `--test` 自测。

## 怎么读这个专题

- 如果你只想调参：读 01、04、12，跳过源码细节。
- 如果你想改 vLLM：按系统线读，每篇末尾的"读代码检查点"是入口。
- 如果你在选型（vLLM vs SGLang vs TensorRT-LLM）：重点看 02、03、04、09、11，这五篇决定了框架的能力边界。
  姊妹专题 [SGLang 研读]({{< relref "sglang-00-index" >}}) 第 12 篇是逐维度的总对比（含两边的参数映射表与选型决策树）。

下一篇从最外面一层开始：[一个 HTTP 请求进来，vLLM 内部到底发生了什么]({{< relref "vllm-01-request-lifecycle" >}})。

> 代码基线：`vllm-project/vllm` main @ `c7a7607`（2026-11-03），最新稳定 tag `v0.33.0`。
> 官方文档：<https://docs.vllm.ai/en/latest/design/arch_overview/>、<https://docs.vllm.ai/en/latest/usage/v1_guide/>
