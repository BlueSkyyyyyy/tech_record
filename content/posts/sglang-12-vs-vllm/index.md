---
title: "SGLang 研读（十二）总对比：同一道题，两种解法"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 13
series: ["sglang"]
tags: ["SGLang", "vLLM", "系列", "推理框架", "对比", "选型"]
categories: ["推理框架"]
---

前十一篇把 SGLang 拆完了。这一篇换个问法：**给定同一台 8 卡 H100、同一个 Llama-3-8B、同一份 200 并发的负载，这两个框架会在哪一步做出不同的决定？**

先把结论摆出来，再逐条论证：

> 2026 年的 SGLang 和 vLLM **不是「有和没有」的关系，是「重心不同」的关系**。
> vLLM 的重心是**调度器的状态机**：把「每个请求算到第几个 token」抽象成唯一的状态，其他一切（chunked prefill、投机、抢占、异步）都是这个状态的推论。
> SGLang 的重心是**负载的共享结构**：把 prompt 的公共前缀当一棵树来管理，其他一切（调度排序、KV 池、路由、前端 fork）都是围着这棵树转。

这个差别在 2024 年非常显著（那时 vLLM 还没有 prefix caching 的块链、SGLang 还没有 TBO），到 2026 年已经收窄成「同一组机制的两种实现顺序」。所以一篇诚实的对比，重点不该是功能清单打勾，而是**每个机制背后的取舍，以及它在什么负载下会翻盘**。

## 1. 六个维度的一张表

| 维度 | vLLM（main @ c7a7607） | SGLang（main @ 1ef3335） | 差异在什么时候变成性能差 |
|---|---|---|---|
| 调度边界 | API Server / **EngineCore 进程** / Worker，前端 ↔ core 走 ZMQ | 主进程（tokenizer）+ **Scheduler 进程（每卡）** + Detokenizer 进程，scheduler 直接持有 `TpModelWorker` | CPU 核数紧张、或需要单独重启引擎时 |
| KV 组织 | 16-token **块** + 链式哈希 + 块表 | token 级**基数树** + `lock_ref` + LRU 叶子淘汰 | 前缀共享深、且长度不是块对齐时 |
| batch 形态 | 统一 `num_computed_tokens` 追 `num_tokens` | **extend / decode 两态**，两套 metadata | 长 prefill 与 decode 混跑时 |
| 装箱策略 | 单一 token budget，running 优先 | token + 请求数 + KV **三约束**，decode 优先 + `retract` | KV 紧张、批量异构时 |
| 重叠 | `AsyncScheduler` + placeholder 做「提前一步」 | `event_loop_overlap` 延后处理结果 + **TBO** 批内切分 | CPU 调度开销占比高、或通信占比高时 |
| 前端 | OpenAI 兼容层 + `LLM` 类 | **DSL（fork/join）+ Rust prefix-aware router** | 多分支生成、多机路由时 |

注意最后一行不是「谁功能多」——vLLM 也能多机部署，只是它把路由交给了外部（llm-d、Gateway API Inference Extension 之类），SGLang 把它做进了仓库。

## 2. 调度边界：进程内 vs 进程间

vLLM V1 的引擎是一个**独立进程**，通过 ZMQ 收请求、回输出，worker 是它下面的子进程。好处是：引擎可以单独重启（不带着 HTTP 层）、前端可以多副本、DP 时每个 rank 一个 EngineCore 天然隔离。

SGLang 把 scheduler 和 model runner 放在**同一个进程**里（`Scheduler` 直接 `self.tp_worker = TpModelWorker(...)`，`scheduler.py:1091`），tokenizer 在主进程、detokenizer 单独一个进程。logits 和采样都在 scheduler 进程里跑完，只有 token id 出进程。

代价对比很具体：

- **CPU 核数**：vLLM `-tp=4` 是 1（API Server）+ 1（EngineCore）+ 4（worker）= 6 个进程；SGLang 是 1（主进程）+ 4（scheduler）+ 1（detokenizer）= 6 个。数量一样，但 vLLM 的 EngineCore 是一个**自旋的 busy loop 进程**，请求要经它转发给 worker；SGLang 的 scheduler 进程自己就是那个 loop，少一层转发。
- **故障隔离**：vLLM 的 worker 崩了，EngineCore 能感知并报错；SGLang 的 model runner 崩了，就是 scheduler 进程崩了——但 scheduler 进程本来也只服务它那一个 rank。
- **扩展**：vLLM 的 EngineCore 可以搬到别的节点（Ray 后端）；SGLang 也有 Ray 后端，但主线是 `mp.Process` 起本机进程。

这条差异**几乎不影响吞吐**，只影响运维形态和「每步跨进程跳几次」。

## 3. KV 组织：树与块的真正差别

这是两套设计里最容易被讲错的地方。两边都有「哈希」、都有「引用计数」、都有「LRU」，差别在**索引的粒度与形状**：

| | vLLM 块链 | SGLang 基数树 |
|---|---|---|
| 索引单位 | 16 token 的块（`block_size` 可配） | 1 token（`page_size=1` 时） |
| 索引结构 | `dict[block_hash -> block_id]`，哈希含父哈希 | 树，节点=一段 token，按首 token 分叉 |
| 命中边界 | 块边界（除非设 `prefix_match_unit`） | 任意 token 边界（`_split_node` 在命中处切开） |
| 共享方式 | 相同哈希的块被多个请求 `ref_cnt+1` | 相同路径被多个请求 `lock_ref+1` |
| 淘汰 | 空闲块 LRU 队列 | 只淘汰**叶子**，按 `last_access_time` |

一个容易被忽略的细节：vLLM main 已经加了 `prefix_match_unit`（`vllm/config/cache.py:204`），允许命中的粒度细于物理块（例如 32 vs 1024），也就是说「块粒度」在 vLLM 里正在变成**可配置项**而不是铁的约束。反过来，SGLang 的 `page_size` 在 MLA/FA4 后端上会被强制成 64/128——树是 token 级的，但 kernel 侧的 KV 是按页读的，两者必须对齐。

所以真实的结论是：

- **两边的索引都在向「更细粒度」收敛**，只是 SGLang 的默认更细、vLLM 的默认更粗。
- 粒度带来的收益有**明确上限**。配套脚本 [`radix_tree_sim.py`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang) 在同一容量、同一 LRU 语义下对拍：

```text
负载           请求  prompt tok     树命中     块命中   差(pp)     重算(树)     重算(块)      多算       查找(树/块)
toolcall     80      22720  93.7%   92.4%     1.3      1440      1728     288     760/2752
agent       200      94163  79.8%   78.2%     1.6     18999     20499    1500     192/10590
rag         200     106502   0.0%    0.0%     0.0    106502    106502       0       0/6766
```

  - 摊到全量 prompt，块比树多算 **1–2%**；
  - 但在「每轮只追加 8 token」的工具调用循环里，块要多算 **+20%**（尾巴被反复重算）；
  - 而树的查找成本反而**低一到两个数量级**（`agent` 组 192 次节点访问 vs 10,590 次哈希），因为每下行一层就吃掉一整段共享前缀。

- **没有共享前缀的负载（RAG、单轮问答）上，树的收益是 0**，只剩内存与查找开销。这就是为什么 SGLang 要配 `lpm` 调度——树的价值必须靠「把同前缀的请求排到一起」来兑现。

vLLM 系列的第 3 篇（[前缀缓存]({{< relref "vllm-03-prefix-caching" >}})）与本系列第 2 篇（[基数树]({{< relref "sglang-02-radix-attention" >}})）是同一道题的两种解法，建议对照读。

## 4. 调度器：一个状态 vs 两个形态

vLLM 的抽象更「数学」：每个请求只有一个数 `num_computed_tokens`，所有阶段都变成「让它去追 `num_tokens_with_spec`」。chunked prefill 是「这一步只追一部分」，投机是「一次追 1+k 个」，抢占是「清零重追」。

SGLang 的抽象更「工程」：`ScheduleBatch` 有 `forward_mode.is_extend()` 和 `is_decode()` 两个真实分支，走不同的 attention 路径、不同的 CUDA Graph、不同的 metadata。`PrefillAdder` 专门负责「这个 prefill 能不能进来」，三重约束（token 预算、请求数、KV）逐条扣。

两者的后果：

- **统一状态的好处**是正确性容易论证（一个不变量管住所有阶段），代价是每步要做一次「阶段推断」，而且 prefill/decode 混跑时的 metadata 会变复杂。
- **两态的好处**是每态可以极致优化（extend 用 varlen + 无 graph，decode 用 full graph），代价是状态机分支多、`retract` 与 `prefill` 的交互需要额外不变量。

在「长 prompt 与短 decode 混跑」这个最常见也最难的场景下，两边的实际行为其实很像：都是 decode 优先、都把长 prefill 切块、都在 KV 紧张时牺牲新请求。差别在于**牺牲谁**：

- vLLM 的抢占挑「最后进来的」或「优先级最低的」请求，把它的 KV **全部丢掉**，之后从头重算。
- SGLang 的 `retract_decode` 按 `(已生成长度, −prompt 长度)` 排序后从尾部挑（先丢生成最少、同分 prompt 最长的），丢掉它的 KV 后**下一轮重新 prefill**，同时保留 `Req` 的状态（已经生成的 token 不丢）。

两者都不是 swap。「内存不够 → 表现为重复计算」这条物理事实，两个框架一模一样。

## 5. 重叠：提前一步 vs 延后一步

这是最容易被 benchmark 数字掩盖的一节。

- vLLM 的做法是**提前**：第 N 步 GPU 还在算，scheduler 就把第 N+1 步排好，输出没回来就用 placeholder 顶着，等结果回来再修正。难点在于「在途未确认」的计数（`num_stale_output_tokens` 之类）。
- SGLang 的做法是**延后**：`event_loop_overlap` 先把第 N 步的 forward 发起，再去调度第 N+1 步，然后才 `process_batch_result(第 N 步)`。采样必须等上一步结果（grammar 的 FSM 要推进），所以顺序被写死成「forward → 调度 → 处理上一批 → 采样」。

两者都把 CPU 调度藏进了 GPU 时间里，区别是**语义复杂度**：placeholder 方案要维护「假 token」，SGLang 方案要维护「延迟一步的 batch 引用」。

批内重叠上两边也趋同了：vLLM 有 `ubatch_size` / `--enable-dbo`（`vllm/config/parallel.py:277`、`:275`），SGLang 有 TBO（`enable_two_batch_overlap`，`arg_groups/fields/exec_.py:1087`）。思路都是「把一条 batch 切成两个 micro-batch，让通信和算力交错」。这个优化只对**通信占比高**的模型（MoE 的 all-to-all、TP 的 all-reduce）有意义，Dense 小模型上收益接近零。

## 6. 能力面：谁的钱花在哪

| 能力 | vLLM | SGLang | 备注 |
|---|---|---|---|
| 结构化输出 | xgrammar / outlines / guidance，**掩码**路径 | xgrammar 为主，掩码 + 跳词**接口** | SGLang 的 `try_jump_forward` 在 4 个 grammar backend 里都有（3 个真实现 + reasoner 透传）（`base_grammar_backend.py:119`），但 `srt/managers/` 下**当前没有调用点**；vLLM 只在 `backend_xgrammar.py:253` 留了注释。两边生产路径实际都是掩码 |
| 投机解码 | 统一 `RejectionSampler`，n-gram / EAGLE / draft，**无树状 topk>1** | EAGLE 为主，**支持 topk>1 树状 verify** | SGLang 的树路径是 target-only 近似（kernel 里还留着 `// FIXME: leverage draft probs`），精确拒绝采样只在 topk=1 的链路径、且默认关；vLLM 链上的拒绝采样是精确的 |
| 前缀感知路由 | 交给外部（llm-d / Gateway API） | 内置 Rust router + `data_parallel_controller` | 多机部署时这条差别很实在 |
| KV 分层 | KV connector（NIXL/Mooncake/LMCache）+ offload | **HiCache** 三级（device/host/storage）+ 同样的连接器 | 两边都在往「L2/L3 缓存」走 |
| 专家并行负载均衡 | `vllm/distributed/eplb/` + elastic EP | `srt/eplb/` + LPLB 求解器 | 都有一等公民实现，别信「只有某家有」 |
| PD 分离 | `vllm/distributed/kv_transfer/` + NIXL | `srt/disaggregation/` + NIXL/Mooncake/MORI | 机制同源，SGLang 的 PD 与 radix cache 有专门的交互（decode 侧也能命中） |
| 前端 DSL | 无 | `sglang/lang`（fork/join/select） | 一次编程表达内多分支，实际是发 k 个 rid |

这张表最重要的信息是**下半部分全是勾**：2026 年这两家在能力面上已经高度重合，选型不该再靠功能清单。

## 7. 参数对照：调参时最容易混的几对

| 你想控制的事 | vLLM | SGLang |
|---|---|---|
| 显存给 KV 多少 | `--gpu-memory-utilization`（默认 None → 0.92；开 extensible KV cache 时 1.0） | `--mem-fraction-static`（默认 None → 按 chunked_prefill/graph bs 反推） |
| 一步最多算多少 token | `--max-num-batched-tokens` | `--chunked-prefill-size` + `--max-prefill-tokens`（默认 16384） |
| 同时在跑的请求数 | `--max-num-seqs` | `--max-running-requests`（默认按 KV 容量推） |
| 前缀缓存开关 | `--enable-prefix-caching`（V1 默认开） | `--disable-radix-cache`（反向开关） |
| 前缀匹配粒度 | `--prefix-match-unit` / `--block-size` | `--page-size`（默认 1；MLA 强制 64，FA4 非 MLA 在 SM100 上 128） |
| 投机 | `--speculative-config` | `--speculative-algorithm eagle --speculative-num-steps …` |
| 批内重叠 | `--enable-dbo` / `--ubatch-size` | `--enable-two-batch-overlap` |
| 结构化输出后端 | `--guided-decoding-backend` | `--grammar-backend` |

注意「默认值」这一列两边都在变：vLLM 的 `gpu_memory_utilization` 从 0.9 改成了 None + 解析，SGLang 的 `mem_fraction_static` 从手填的 0.9 一类常数改成自动推导。**照抄两年前的调参博客，两个框架都会踩坑。**

## 8. 选型：一棵决策树

```text
负载里有大量共享前缀（系统提示 / 多轮 / agent tool loop）？
├── 是 → 单机：两者差距 1–2%，看你的调度是否喜欢 lpm（把同前缀排一起）
│        多机：SGLang 的内置 prefix-aware router 省一个组件（默认 policy 是 round_robin，要显式开）
└── 否 → 纯 RAG / 单轮问答：树的收益归零，选你团队更熟的那个

需要一次请求内多分支生成（fork / select / 并行采样）？
├── 是 → SGLang 的 DSL 是唯一现成答案（否则没有一层能保证「前缀逐字相同 + 同时在途」，复用会碎）
└── 否 → 跳过

模型是 MoE + EP，通信占比高？
├── 是 → 两者都有 DBO/TBO + EPLB，比的是「谁的 kernel 生态对这个模型更熟」
└── 否 → 批内重叠不用开

只有 JSON / 工具调用，输出低熵？
├── 是 → 两边都是掩码路径；真正省时间的是「少采样」，不是框架
└── 否 → —

运维约束：进程模型 / 灰度 / 多租户隔离？
├── 需要引擎独立重启、前端多副本 → vLLM 的 EngineCore 边界更干净
└── 想少一层转发、少一个进程 → SGLang 更直接
```

## 9. 三个「别指望」

写对比文章最容易犯的错，是让人以为换个框架就能改物理。这三条对两个框架都成立：

1. **别指望框架改变 decode 的带宽下界**。batch=1 时一次 decode 要把权重读一遍，8B bf16 在 3 TB/s 上是 ~5 ms，谁来都一样；框架只能通过「把 batch 做大」来摊薄，而 batch 能做多大又回到 KV 容量。
2. **别指望前缀缓存救没有共享的负载**。`rag` 组那个 0% 命中率不是实现问题，是负载没有结构。
3. **别指望重叠能掩盖串行依赖**。采样必须等 forward 结果、grammar 必须等采样结果——这条链上能藏的只有 CPU 的调度工作，藏不了 GPU 的等待。

## 小结

- **重心不同**：vLLM 是「一个状态机管所有阶段」，SGLang 是「一棵树管所有复用」。前者正确性好论证，后者在共享结构深的负载上更省。
- **粒度红利有限但真实**：全量摊薄 1–2%，工具调用循环里 +20%；而树的查找成本低 1–2 个数量级。
- **调度差异体现在「牺牲谁」**：vLLM 抢占丢 KV 重算，SGLang retract 后重新 prefill，都逃不开「内存不够 = 重复计算」。
- **重叠已趋同**：提前一步（placeholder）vs 延后一步（result_queue），批内重叠两边都有（DBO/ubatch vs TBO）。
- **能力面高度重合**：EPLB、PD 分离、KV 分层、量化格式，两边都有一等公民实现；真正的差异在**前端 DSL** 与**内置 prefix-aware router**。
- **调参别抄旧文**：两个框架的显存参数默认值都在 2026 年改成了「自动推导」。

这个专题到这里结束。如果只记一句话：

> **SGLang 把「prompt 的公共结构」当成一等公民，vLLM 把「请求的计算进度」当成一等公民。**
> 你的负载里哪一种结构更强，就用哪一个。

> 代码基线：`sgl-project/sglang` main @ `1ef3335`、`vllm-project/vllm` main @ `c7a7607`（均为 2026-11-03）。
> 配套代码：[`code/sglang/`](https://github.com/BlueSkyyyyyy/tech_record/tree/main/code/sglang)（树 vs 块对拍、显存预算、调度装箱、overlap 时间线、跳词收益）。
> 交叉阅读：vLLM 专题的[导读]({{< relref "vllm-00-index" >}})、[调度器]({{< relref "vllm-04-scheduler" >}})、[KV 传输]({{< relref "vllm-11-kv-transfer" >}})。
