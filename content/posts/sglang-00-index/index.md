---
title: "SGLang 研读（专题）：把前缀树当调度器用"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 1
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "RadixAttention", "前缀缓存", "调度"]
categories: ["推理框架"]
---

一个客服机器人上线第一天：200 个会话同时在线，每个会话前面都顶着同一段 4,000 token 的**系统提示**（人设、工具清单、业务规则），后面各自接 20 token 的用户问题。到第二天，产品经理把系统提示改了 30 个 token，再发布一次——缓存全废。

第三天，他们又加了「每个租户一段 skill 定义」：A 租户的 skill 和 B 租户的 skill 前 800 token 完全一样，从第 801 个 token 开始分叉。

这时候你会发现，问题已经不是「缓存命中率有多高」，而是**「共享结构本身就是一棵树，你用什么数据结构去装它」**。

vLLM 的答案是**块**：把 KV 切成 16 token 的定长块，块做哈希、哈希串成链，命中以块为单位。SGLang 的答案是**树**：token 级基数树（radix tree），节点可以分裂，命中可以落在任意 token 边界。（SGLang 侧也有哈希，但它是 KV 事件与 HiCache 的键，不参与索引结构。）

这一个选择，决定了后面所有的东西：KV 池怎么分、调度器怎么排序、attention 怎么拿索引、以及「一次能验证几棵草稿树」。

## 先记住一个画面

把 200 个请求的 prompt 全部倒进一个桶里，按 token 逐位对齐，相同的部分叠起来：

```text
root
├── [system prompt 4,000 tok]           ← 200 个请求共享
│   ├── [tenant A skill 800 tok]
│   │   ├── [tool schema 300 tok]       ← A 的 120 个请求共享
│   │   │   ├── 用户问题 1 …
│   │   └── [另一种 tool schema 280 tok]
│   └── [tenant B skill 800 tok]
└── [另一个 system prompt 长短不一]
```

这棵树上的每一个节点，对应一段连续 token 的 KV，存在 `token_to_kv_pool` 里的一段 slot 上。一条根到叶的路径，就是一个请求的完整 KV。**树的形状就是负载的共享结构**，而不是被人为切成 16 token 的块。

一旦接受这个画面，「前缀缓存」就不再是「把算过的块存起来」，而是「把 prompt 插进树里、匹配最长前缀、按引用计数保护正在被用的路径、按 LRU 淘汰没人用的叶子」。

而 SGLang 的调度器更进一步：在缓存感知档（`lpm` / `dfs-weight` 等）下它**先查树、再排序**，因为「插入这棵树能复用多少 KV」本身就是一个极好的装箱依据。默认档仍是 `fcfs`，等待队列超过 128 条也会降级回 `fcfs`。

> **类比的边界**：vLLM 常被比作操作系统（块=页、块表=页表），SGLang 则像文件系统的 **inode + 按内容寻址**：同一段字节只存一份，多个路径可以指向它。但有一个关键不同——文件系统的引用是持久的，SGLang 的引用是**按请求生命周期**的（`lock_ref`），请求一结束路径就退回可淘汰池。所以「树很大」不等于「KV 很多」，树里同时混着活路径和死路径。

## 三个账本

| 账本 | 记什么 | 对应文章 |
|---|---|---|
| **复用** | 一次请求能省下多少 prefill token；树查找的 CPU 成本 | 02、03 |
| **装箱** | 每一步跑谁、跑多少 token；KV 不够时丢谁 | 04 |
| **时间线** | CPU 调度、GPU 计算、采样三者怎么重叠 | 05 |

三个账本互相咬：树复用越激进 → 命中前缀越长 → 单步 prefill 越短（装箱变容易），但树越大 → 淘汰越频繁 → 抓到手的路径可能刚好被淘汰（复用失效）。SGLang 用 `lock_ref` 把正在跑的路径钉住，用 `evict()` 只淘汰叶子，才让这两件事不打架。

## 一个可以现在就跑的账：粒度到底值多少钱

配套脚本 [`code/sglang/radix_tree_sim.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang) 把两种结构放在**同一个 token 池、同一容量、同一 LRU 语义**下对拍，只在索引粒度上不同（树 = 1 token，块 = 16 token）：

```text
负载           请求  prompt tok     树命中     块命中   差(pp)     重算(树)     重算(块)      多算       查找(树/块)
multiturn    36      16416  89.3%   87.7%     1.5      1764      2016     252     198/1944
toolcall     80      22720  93.7%   92.4%     1.3      1440      1728     288     760/2752
system      120      96879  74.1%   73.4%     0.7     25080     25743     663     117/10560
agent       200      94163  79.8%   78.2%     1.6     18999     20499    1500     192/10590
rag         200     106502   0.0%    0.0%     0.0    106502    106502       0       0/6766
```

三行结论，比任何 benchmark 都可靠：

1. **粒度红利是「尾巴」级的**：块命中永远 ≤ 树命中，差值就是每块最多 15 个 token 的浪费。摊到全量 prompt 上只有 1–2 个百分点；但在「每轮只追加 8 token」的工具调用循环里（`toolcall` 组），块要多重算 288 token，相对树的 1440 是 **+20%**——增量越小，粒度越贵。
2. **查找成本反过来差近两个数量级**：`agent` 组里树只访问了 **192 个节点**，块算了 **10,590 次哈希**（55 倍）。原因很直白——树每下行一层就吃掉一整段共享前缀，而块必须把整条链的每一块都哈希一遍。前缀共享越深，树越便宜。
3. **没有共享前缀时（`rag` 组）树的收益是 0**，只剩内存和查找开销。这解释了为什么 SGLang 要在调度层再放一层 `CacheAwarePolicy`（lpm / dfs-weight 等，默认仍是 `fcfs`）：树只负责「记得住」，「把同前缀的请求排到一起、让它们真的命中」得靠调度器。树的价值完全来自负载的共享结构。

所以这个专题不会说「树比块好」——它说的是：**树把「共享结构」当成一等公民，代价是更复杂的生命周期管理；块把「均匀性」当成一等公民，代价是粒度损失。** 选哪个，取决于你的负载长什么样。

## SGLang 的进程模型

读完 vLLM 那 13 篇的话，这里是第一处结构性差异。SGLang 没有 vLLM 的 EngineCore 层：

| 进程 | 数量 | 职责 |
|---|---|---|
| 主进程（HTTP + TokenizerManager） | 1 | FastAPI、tokenize、多模态预处理、流式回包、请求状态表 |
| Scheduler | `TP × PP`（每卡一个） | 调度、KV 决策、驱动 model runner、采样 |
| Detokenizer | 1（可配多 worker + router 进程） | 增量 detokenize、停用词、输出整形 |
| DataParallelController | DP>1 时 1 个 | DP 组间的请求分发与负载均衡 |

举例：单机 4 卡 `--tp 4` → 1 + 4 + 1 = **6 个进程**；8 卡 `--tp 2 --dp 4` → 主进程 + 4 组 scheduler（每组 2 个）+ 1 detokenizer + 1 controller。

调度和执行的边界，SGLang 划在**进程内**（scheduler 直接持有 `TpModelWorker`，调 `forward_batch`），vLLM V1 划在**进程间**（前端 ↔ EngineCore 走 ZMQ，EngineCore ↔ Worker 是 mp 队列）。这条边界的取舍会在[架构篇]({{< relref "sglang-01-architecture" >}})和第 12 篇对比里展开。

## 2026 年的 SGLang 变了什么

如果你读的是 2024 年 RadixAttention 论文时期的笔记，下面这些已经变了：

- **`page_size` 不再恒为 1**：CUDA 上默认仍是 1，但 MLA 后端强制 64、FA4 强制 128，AMD/MUSA 另有默认——「token 级树」在 kernel 侧会被对齐成页。
- **`mem_fraction_static` 不再手填**：默认从 `chunked_prefill_size`、CUDA Graph 的 `max_bs` 反推，公式在 `arg_groups/memory_hook.py`，而不是「0.9 一把梭」。
- **Radix cache 正在收敛到 Unified Radix Cache**：Full/SWA/Mamba 三类组件共用一套树（`mem_cache/unified_radix_cache.py`），旧的 `radix_cache.py` 仍是 MHA 主线。
- **两级 overlap**：除批间流水（`event_loop_overlap`）外还有批内 TBO（`batch_overlap/two_batch_overlap.py`），把通信和计算在一条 batch 内部切开。
- **前后端分家**：Rust 写的 `sgl-model-gateway` / `sgl-router` 成为生产入口，Python 侧保留 HTTP server 与 Engine。
- **PD 分离与 HiCache** 成为一等公民：`disaggregation/` 有 NIXL/Mooncake/MORI 后端，`mem_cache/` 有 device/host/storage 三级。

所以本专题以 **main @ `1ef3335`（2026-11-03）** 为准。行号会漂，机制和数据结构稳定。

## 三条阅读路线

- **系统线（推荐先读）**：01 架构 → 02 基数树 → 03 KV 池 → 04 调度器 → 05 overlap。读完能回答「一个请求从进到出，KV 在树上怎么走」。
- **能力线**：06 Attention 后端 → 07 结构化输出 → 08 投机解码。读完能判断「为什么 SGLang 在 JSON/agent 场景更快，以及代价」。
- **部署线**：09 前端与路由 → 10 PD 分离与 HiCache → 11 分布式 → 12 与 vLLM 总对比。读完能做选型。

## 系列目录

| # | 文章 | 一句话 |
|---|---|---|
| 0 | 本篇 | 三个账本与架构地图 |
| 1 | [进程与请求的一生]({{< relref "sglang-01-architecture" >}}) | ZMQ、三进程、一轮请求的三次跨进程消息 |
| 2 | [RadixAttention：基数树前缀缓存]({{< relref "sglang-02-radix-attention" >}}) | 匹配、分裂、lock_ref、LRU 叶子淘汰 |
| 3 | [KV 池、slot 分配器与显存账本]({{< relref "sglang-03-memory-pool" >}}) | mem_fraction_static 自动推导与每 token 字节 |
| 4 | [调度器：PrefillAdder、chunked prefill 与 retract]({{< relref "sglang-04-scheduler" >}}) | 每步怎么装箱、装不下丢谁 |
| 5 | [两级 overlap 与 CUDA Graph]({{< relref "sglang-05-overlap" >}}) | 批间流水与批内 TBO |
| 6 | [Attention 后端：extend/decode 两条路径]({{< relref "sglang-06-attention-backends" >}}) | metadata、page table 与 MLA |
| 7 | [结构化输出：压缩 FSM 与跳词解码]({{< relref "sglang-07-constrained" >}}) | 掩码 vs 跳词，谁在低熵输出上更快 |
| 8 | [投机解码：EAGLE 草稿树与一次验证]({{< relref "sglang-08-speculative" >}}) | topk 树、验证、接受长度 |
| 9 | [前端 DSL、Router 与 cache-aware 负载均衡]({{< relref "sglang-09-frontend" >}}) | 从 `sgl.gen` 到多机路由（默认仍是 round_robin） |
| 10 | [PD 分离与 HiCache 三级缓存]({{< relref "sglang-10-pd-disagg" >}}) | 把 KV 搬到别的机器/下一层 |
| 11 | [TP/EP/DP-attention 与 EPLB]({{< relref "sglang-11-distributed" >}}) | 多卡怎么切、专家怎么放 |
| 12 | [总对比：同一道题，两种解法]({{< relref "sglang-12-vs-vllm" >}}) | 架构、复用、调度、能力、调参对照 |

配套代码在 [`code/sglang/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang)：纯 CPU 脚本，覆盖基数树 vs 块哈希、显存预算、调度装箱、overlap 时间线、跳词收益，每个都能 `--test` 自测。

## 怎么读这个专题

- 只想选型：读本篇、02、12，再翻 07（结构化输出）和 10（PD 分离）。
- 想改 SGLang：走系统线，每篇末尾的「读代码检查点」是入口。
- 已经读过 vLLM 系列：可以倒着读——先看 12 的总对比，再回来补 02/04 的机制细节。

下一篇从最外面一层开始：[一个请求进来，SGLang 内部到底发生了什么]({{< relref "sglang-01-architecture" >}})。

> 代码基线：`sgl-project/sglang` main @ `1ef3335`（2026-11-03，tarball 快照，无 `.git`）。
> 对比基线：`vllm-project/vllm` main @ `c7a7607`（2026-11-03）。
> 论文与文档：SGLang 论文 <https://arxiv.org/abs/2312.07104>、官方文档 <https://docs.sglang.ai/>、RadixAttention 设计说明见仓库 `docs/`。
