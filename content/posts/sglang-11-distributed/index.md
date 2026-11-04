---
title: "SGLang 研读（十一）分布式：TP/EP/DP-attention 与 EPLB"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 12
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "分布式", "EP", "EPLB", "张量并行", "数据并行"]
categories: ["推理框架"]
---

[第 1 篇]({{< relref "sglang-01-architecture" >}})画过进程图，[第 5 篇]({{< relref "sglang-05-overlap" >}})拆过 TBO。两篇都留了一个口子没堵：`tp=8` 时到底有几个 scheduler？为什么 SGLang 里 `tp` 和 `ep` 经常是同一个数？为什么它敢在 `attn_tp=1` 的情况下还自称「张量并行」？

先看一个画面。`DeepSeek-V3 + --tp 8 --attn-dp-size 8 --ep 8 --moe-a2a-backend deepep` 这个配置下，8 张卡上有 **8 个 Scheduler 进程**（不是一个），每个只认得自己那几十条请求的 KV；但它们每一步都要在同一个 NCCL 组里碰头，因为每层的专家权重被切成 8 份，token 必须去别人家算。同一组 8 个 rank，在 attention 眼里是 8 个互不相干的副本，在 MoE 眼里是一个整体。**一个 world，两套坐标系**——这是本篇的主线。

三个问题：四种并行切下去每卡持有什么、DP attention 省的是什么、EPLB 什么时候动以及动了要不要停推理。

## 1. 四种切法：切下去之后每卡有什么

| 并行 | 切什么 | 每卡持有什么 | 每步的跨卡动作 | 主要代价 |
|---|---|---|---|---|
| TP | 层内权重（列切 + 行切）、head 维 | 每层 1/tp 的权重；KV 按 head 切 | 每层 2 次 all-reduce | 通信随层数线性，必须全员到齐 |
| PP | 按层切 stage | 只有自己那几层的权重与该层 KV | stage 间 p2p 一次 hidden | 并发流水，气泡 |
| DP（`--dp-size`） | 请求 | 完整模型副本 + 自己的 KV | 几乎不通信 | 显存按副本数翻倍 |
| DP attention（`--attn-dp-size`） | TP 组内的「注意力」这一维 | attention 权重整份、KV 只存自己的请求；MLP/MoE 仍按 tp 切 | MLP 前 gather、后 scatter | batch 必须跨 rank 对齐 |
| EP（`--ep-size`） | MoE 专家 | 1/ep 的专家权重 | dispatch/combine 的 all-to-all | 流量按 topk 计，热点专家排队 |

三句话把表串起来：

- **TP 与 PP 切的是「同一批 token 的模型」**，必须同步执行，所以 `world_size = tp × pp`（`python/sglang/srt/distributed/parallel_state.py:2640-2648` 那一句不等就抛错）。
- **DP 切的是「哪些 token」**，副本之间没有 NCCL 依赖，`num_dp_ranks = dp_size × attn_dp_size` 个 scheduler（`python/sglang/srt/runtime_context.py:202-210`）。
- **DP attention 是第三种东西**：它切的是 TP 组内部的注意力维，属于「在 TP 里面做 DP」。所以它既不是 `--dp-size`，也不能和它同开——两者都大于 1 直接抛 `ValueError`（`python/sglang/srt/arg_groups/parallel_hook.py:290-295`）。

> 顺手拆个雷：`--enable-dp-attention` 在当前 main 上**已经不是命令行开关**。`enable_dp_attention` 声明成 `no_cli=True`（`python/sglang/srt/arg_groups/fields/parallel.py:203-205`），老写法 `--dp-size N --enable-dp-attention` 会被兼容钩子翻译成 `--attn-dp-size N --dp-size 1`（`parallel_hook.py:41-64`），计划 2026-12-31 后删。照老文章找 `enable_dp_attention` 的分支会白找。

## 2. DP attention 省的是什么

结论先给：**省的是 KV 的 tp 倍复制，以及 attention 侧那一串「每层一次、必须全员到齐」的 all-reduce；代价是每步一次跨 DP 组的 batch 对齐。**

### 2.1 KV 的账

KV 按 head 切，所以复制倍数只取决于 head 数：

$$
\text{复制倍数} = \max\left(1,\ \frac{\text{tp}}{\text{kv\_heads}}\right)
$$

MLA 只有 1 个「kv 头」（512 维 latent + 64 维 rope），`tp=8` 就是 8 份完整复制。按 `(512+64) B/token/layer × 61 层 = 35 KiB/token`、每卡给 KV 40 GB 估：单卡约 1.1M token，8 卡 TP 部署的**唯一** token 容量还是 1.1M——多出来的 7 张卡在存同样的东西。换成 `attn_tp=1` 的 DP attention，8 卡各存各的，容量 9.1M。这是 8 倍，也是官方文档把 MLA 列为 DPA 头号受益者的原因（`docs/docs/advanced_features/dp_dpa_smg_guide.mdx:23-29`）。

### 2.2 通信的账：每层每卡 3.5S → 1.75S

设全局 batch 为 $B$ 个 token、hidden 为 $h$、bf16，单次 collective 的 payload：

$$
S = B \times h \times 2\ \text{bytes}
$$

ring all-reduce 每卡搬 $2(N-1)/N \cdot S$，$N=8$ 时约 $1.75S$。两种切法每层的账：

| 每层每卡通信 | 纯 TP（`attn_tp=8`） | DP attention（`attn_tp=1`，MLP 仍 TP） |
|---|---|---|
| attention 之后 | 1 次 all-reduce = 1.75S | **0**（组大小 1，直接返回） |
| MLP 前 + 后 | 入口不需要搬（每卡本来就有全部 token）、出口 1 次 all-reduce = 1.75S | 入口 gather、出口 scatter，合计 1.75S |
| 合计 | **3.5S** | **1.75S** |

$B=256$、$h=7168$ 时 $S \approx 3.5$ MiB，即每层每卡 12.3 MiB 对 6.1 MiB；61 层是 750 MiB 对 375 MiB，按 NVLink 450 GB/s 折 1.7 ms 对 0.85 ms。**减半的来源很具体**：attention 那次 all-reduce 整个消失，MLP 侧只是把「all-reduce」换成「gather + scatter」，字节打平。

那个「直接返回」不是比喻：`GroupCoordinator.all_reduce` 第一句就是 `if self.world_size == 1: return input_`（`parallel_state.py:707-709`）。而 `attn_tp_size = tp // attn_dp_size // attn_cp_size`（`runtime_context.py:73-77`），所以 `attn_dp_size = tp_size` 时 attention 一个 collective 都没有。MoE 层要补一句：MLP 那半被 EP 的 dispatch/combine 取代，两种切法都要付，DP attention 在 MoE 层省下的是完整的 1.75S，只是相对 all-to-all 的流量不算大。

### 2.3 代价

- **batch 必须对齐**（第 4 节）：不是「顺手 sync 一下」，而是 forward mode、CUDA Graph 能力、TBO 切分点都要投票一致。
- **LM head 变贵**：TP 下 lm_head 按 vocab 切在 8 卡上，采样只要在 `attn_tp` 内 all-gather；DPA 下这个 gather 要跨整个 tp 组（`python/sglang/srt/layers/logits_processor.py:539-541` 的 `do_tensor_parallel_all_gather_dp_attn`），除非开 `--enable-dp-lm-head` 把词表并行收回 attention TP 组（代价是每卡存整份 lm_head），或 `--enable-tp-lm-head-all-to-all` 改成 all-to-all。

## 3. group 怎么建：一个 world，两套坐标系

`initialize_model_parallel`（`parallel_state.py:2563`）读一串宽度，然后**从同一个 tp 组里切出两套组**：

```python
# python/sglang/srt/runtime_context.py:73-101  去掉 assert 的骨架
attn_tp_size = tp_size // attn_dp_size // attn_cp_size
attn_tp_rank = tp_rank % attn_tp_size
attn_dp_rank = tp_rank // (attn_tp_size * attn_cp_size)   # 布局 (dp, cp, tp)，tp 变得最快
```

- TP 组是连续块 `[g0..g7]`（`parallel_state.py:2682`）。
- attention TP 组是 TP 组里按 `attn_tp_size` 切出的连续子块，组名 `attention_tp`（`parallel_state.py:2790-2801`）；`attn_tp_size == tp_size` 时它**直接等于 TP 组**，连新组都不建（`:2773-2774`）。
- MoE EP 组是同一段 rank 上按 `moe_tp_size` 步长取的跨步块，组名 `moe_ep`（`parallel_state.py:2886-2892`）；`moe_tp_size = tp_size // ep_size // moe_dp_size`（`runtime_context.py:137-155`）。

8 卡跑第 1 节那个 DeepSeek 配置，两套坐标系的差别一目了然：

| 量 | 值 | 怎么来的 |
|---|---|---|
| `attn_tp_size` | 1 | `8 // 8 // 1` |
| `moe_ep_size` / `moe_tp_size` | 8 / 1 | 配置 `--ep 8`；`8 // 8 // 1` |
| `num_dp_ranks`（scheduler 数） | 8 | `dp_size × attn_dp_size` |
| 每卡 attention 组 / MoE 组 | `[自己]` / 全部 8 卡 | `attn_tp_size = 1`；`moe_ep_size = 8` |

**`--ep-size` 其实不是自由旋钮。** 只要 `--moe-a2a-backend` 落在 `_A2A_EP_SPANNING_BACKENDS` 里（deepep / deepep_v2 / mooncake / nixl / flashinfer / megamoe / mori / pplx / ascend_fuseep …，`python/sglang/srt/arg_groups/overrides.py:1656-1669`），`ep_size` 会被静默改成 `tp_size`（`overrides.py:1690-1699`）。文档那句是同一个意思：这些后端只支持 `ep_size = tp_size`，混合 EP/TP 只能用 `none` 后端（`docs/docs/advanced_features/expert_parallelism.mdx:75`）。所以在 SGLang 里「EP 度」和「TP 度」是**一个数字**，要改 EP 只能改 TP。

## 4. require_mlp_sync：batch 为什么必须跨 rank 对齐

`require_mlp_sync` 只有一行：`attn_dp_enabled or require_gathered_buffer()`（`python/sglang/srt/utils/common.py:4205-4208`）。名字起得不好，它管的远不止 MLP：只要开了 DPA，**每步 forward 之前所有 DP rank 都要交换一份 11 列的元数据**。

```python
# python/sglang/srt/managers/scheduler_components/dp_attn.py:176-193  每 rank 一行
[num_tokens, num_tokens_for_logprob, can_run_decode_cuda_graph, is_extend_in_batch,
 local_can_run_tbo, local_forward_mode, can_run_prefill_cuda_graph,
 prefill_cuda_graph_max_prefix_len, can_run_draft_cuda_graph, is_health_check,
 attention_pairs]
```

收集在 `MLPSyncBatchInfo.all_gather`（`dp_attn.py:226`），归约规则分三种，每种都有理由：`global_num_tokens` 直接收集（gather/scatter 的偏移表，MoE 的 all-to-all buffer 也按它定形状，`dp_attn.py:288-290`）；`is_extend_in_batch` 取 **max**（任一 rank 有 prefill，全场按 extend 走，`:291`）；`can_run_*_cuda_graph` 取 **min**（有一家 replay 不了，全场退回 eager）。TBO 的 `local_can_run_tbo` 也在这里投票，切分点必须一致，见[第 5 篇]({{< relref "sglang-05-overlap" >}})。

然后 `prepare_mlp_sync_batch_raw`（`dp_attn.py:518`）做三件事：**造 idle batch**（自己一步没活也得拿空 batch 陪跑，否则别人的 collective 等不到它，`dp_attn.py:660-672`）；**统一 forward mode**（`maybe_convert_decode_to_extend` 在「本 rank 只有 decode、对端在 prefill」时把 decode 批次转成 extend 视图，让 8 个 rank 一起 replay prefill 图，`dp_attn.py:783-812`）；**回填给调度器**（`python/sglang/srt/managers/scheduler.py:4179-4192`）。

**这就是 DP attention 真正的代价**。`scheduler.py:1489-1491` 的注释写得最直白：prefill/decode 的交错节奏由 `is_extend_in_batch` 决定，「即使只有一个 rank 有本地 prefill 工作，所有 rank 也跟着走同一条节奏」。8 个副本本来可以各跑各的，现在一个 rank 的 8K prefill 会把另外 7 个 rank 的 decode 一起拽进 extend 步。硬币的另一面是 `scheduler.py:2225-2226`：「Under DP, `is_extend_in_batch` is synced, so all ranks decide the same (no deadlock)」——正因为同步了，「先处理哪一批结果」这类顺序决策才不会死锁。下游还有一处：CUDA Graph 的 batch size 对齐粒度变成 `attn_tp_size`（`common.py:4210-4231` 的 `get_cuda_graph_batch_size_alignment`），DPA 下 decode 必须 pad 到整数倍，否则各 rank 的图对不上。

## 5. EP：dispatch/combine，以及「不用 a2a 的 EP」

`--moe-a2a-backend none`（默认）时，SGLang 的 EP 是**假的 all-to-all**：`StandardDispatcher.dispatch` 先把各 DP 组的 token 拼到全组（`python/sglang/srt/layers/moe/token_dispatcher/standard.py:169-177` 的 `all_gatherv`），每个 rank 只算本地专家，`combine` 再 `reduce_scatterv` 回去（`standard.py:288-299`）。好处是不装 DeepEP 也能跑，坏处是**每个 rank 都要收下全量 token**。真 backend 才走 dispatch/combine，DeepEP 是主力，且有一条硬约束：`num_max_dispatch_tokens_per_rank` 断言 `<= 1024`（`token_dispatcher/deepep.py:402-409`，注释解释了 INTERNODE_LL 的 `FINISHED_SUM_TAG=1024`），这个值同时决定 NVSHMEM 的 QP 深度（`deepep.py:82-83`）和显存预留。

**EP 的显存账**（沿用 [vLLM 那篇]({{< relref "vllm-10-distributed" >}})的算法）：DeepSeek-V3 类配置每个专家 3 个 7168×2048 矩阵，fp8 下约 44 MB；256 专家 × 58 个 MoE 层 ≈ 653 GB，`ep=8` 摊到 82 GB/卡。这是「671B 塞进 8 张 H200」的前提，也是 SGLang 宁可让 `ep_size = tp_size` 的原因。

## 6. EPLB：统计、求解、搬迁

EPLB 解决一个很朴素的问题：256 个专家里总有几个被 route 得特别多，切到 8 张卡上就是某几张卡单独排队。三段，每段的开关和代价都不一样。

**统计**：`--enable-eplb` 把 recorder 模式自动设成 `stat`（`parallel_hook.py:752-759`）。`_StatAccumulator` 每个 forward 往 `_CircularBuffer` 追加一行 `[num_layers, num_physical_experts]` 计数（`python/sglang/srt/eplb/expert_distribution.py:957-987`、`:1076-1092`），窗口默认等于 `eplb_rebalance_num_iterations`（`parallel_hook.py:805-814`），并有断言保证窗口不比重排间隔长（`eplb/eplb_manager.py:55-61`）——否则 buffer 里会残留上一轮的旧热度。dump 时跨 rank `all_reduce` 合成全局计数（`expert_distribution.py:1001-1003`），判据是平均利用率：

$$
u = \frac{\text{mean}_g(\text{count}_g)}{\text{max}_g(\text{count}_g)}
$$

见 `compute_utilization_rate`（`expert_distribution.py:1164-1179`）。但默认 `eplb_min_rebalancing_utilization_threshold = 1.0`（`arg_groups/fields/exec_.py:957-960`），而 $u$ 恒小于等于 1——于是 `_check_rebalance_needed`（`eplb_manager.py:225-238`）的「跳过」分支永远不命中：**默认行为是「到点就重排」，不是「不均衡才重排」**。想按需，把阈值调到 0.5 量级；注意只有阈值不为 1.0 时才记窗口历史（`expert_distribution.py:843-847`）。

**求解**：算法在 `eplb/eplb_algorithms/deepseek.py`，文件头第一行就写着抄自 DeepSeek 官方 EPLB 仓库。两个贪心：`balanced_packing`（`deepseek.py:7-53`）把带权对象装进 m 个桶，每桶数量固定，按权重降序放进当前最轻且没满的桶；`replicate_experts`（`:55-83`）按 `weight / logcnt` 的最大值反复复制，热点专家被复制多份，摊薄单卡负载。层次版 `rebalance_experts_hierarchical`（`:86-169`）三步：专家组装到节点 → 节点内造冗余副本 → 物理专家装到卡。非层次版就是把参数拍平成 `num_groups=1, num_nodes=1` 再调同一个函数（`deepseek.py:202-205`）——**不是两套算法，是一套算法的两种装箱粒度**。`auto` 的选择在 `eplb_algorithms/__init__.py:75-87`：专家组数能被节点数整除就用层次版。节点数取 `get_parallel().nnodes`（`eplb/expert_location.py:199`），所以 `--nnodes` 填错，装箱也会错——它会把两张卡当成同一台机器。

**搬迁**：`EPLBManager` 是个生成器，每 `eplb_rebalance_num_iterations` 次 forward 开始一轮（`eplb_manager.py:72-94`），挂在 `ModelRunner.forward` 尾部（`python/sglang/srt/model_executor/model_runner.py:1946-1948`）。**没有独立线程，也没有 stop-the-world**，重排就发生在调度步里：按 `--eplb-rebalance-layers-per-chunk` 切层，每块之间 `yield` 一次（`eplb_manager.py:152-155`），于是「搬一块、跑一步、再搬下一块」。块内是同步 P2P——把权重从源 rank `irecv` 到临时 buffer 再拷回张量（`eplb/expert_location_updater.py:234-300`、`:544-566`），`batch_isend_irecv` 按 `SGLANG_EPLB_P2P_BATCH_CHUNK_SIZE`（默认 32，`environ.py:1425-1427`）分批提交，注释写明是为了避免大规模重排时 NCCL/RCCL 在 GPU 侧堆积挂死。三种情况省掉真实传输：位置没变、同卡换槽（memcpy）、目标 rank 已有副本（free-rider，`expert_location_updater.py:332-372`）；搬不到的（弹性 EP 下掉过 rank）回落成从磁盘/DRAM 重载整份权重（`eplb_manager.py:321-340`）。

**代价能算**：61 层、256 逻辑专家 = 32 槽/卡，一轮挪 10% 的槽位，$61 \times 3.2 \times 44\ \text{MB} \approx 8.6$ GB；NVLink（450 GB/s）约 19 ms，跨机 IB（50 GB/s）约 172 ms。默认 `--eplb-rebalance-layers-per-chunk` 是 `None` → `chunk_size = 1000000`（`eplb_manager.py:243`）→ **只有一块，全部层一步搬完**，也就是默认配置下这 19–172 ms 是一次性卡住的。想摊平就显式给 4 或 8。

还有一条更细的：`--ep-dispatch-algorithm lp` 启用 LPLB（线性规划）。它不是「重排权重」，而是**每个 batch 现解一次 LP**，决定一个逻辑专家的多份副本这一批往哪份发：`LPLBSolver.solve` 先本地计数、再 EP 组 all_reduce 出全局计数、然后跑 LP（`eplb/lplb_solver.py:208-254`，kernel 三步在 `:256-270`）。因为带 all-reduce，**所有 EP rank 每层都必须调它，包括这一层一个 token 都没有的 rank**（`lplb_solver.py:30-42` 解释了为什么只支持 7 个模型家族：其它家族的 MoE 有空 token 提前返回，会死锁）。这是「在线重平衡」，和 EPLB 的「离线搬权重」是两件事。

## 7. DP 的请求分发：controller 与四种策略

[第 1 篇]({{< relref "sglang-01-architecture" >}})说过 PUSH/PULL 没有寻址能力，所以 DP 一定要有个中间进程。四种策略在 `LoadBalanceMethod`（`python/sglang/srt/managers/data_parallel_controller.py:105-112`），分派表在 `:189-195`：

| 策略 | 依据 | 适用 |
|---|---|---|
| `round_robin` | 轮转下标 | 默认（非 PD）；elastic EP 强制只用它（`parallel_hook.py:674-679`） |
| `follow_bootstrap_room` | `bootstrap_room % num_workers` | PD 分离的 prefill（同一 room 必须落同一 rank） |
| `total_requests` | 各 rank 的 running + waiting 请求数 | 请求长度均匀 |
| `total_tokens` | 各 rank 的 token 总量 | 长度方差大；只有 DPA 下有真实 token 账 |

`--load-balance-method` 默认 `auto`（`fields/parallel.py:95-107`），解析很短：PD prefill → `follow_bootstrap_room`，其余 → `round_robin`（`arg_groups/serving_hook.py:214-232`）。

**为什么不能简单 round-robin？** 三层原因，都能在代码里找到：

1. round-robin 均分的是「请求个数」，不是「工作量」。一条 8K prefill 和一条 1 token decode 在账本上都是 1。`total_tokens` 就是为这个存在的，按 `len(input_ids)` 预估（`data_parallel_controller.py:899-909`）。
2. **账本是异步的**。controller 从 `/dev/shm` 的 slot 读各 scheduler 发布的快照（`managers/load_snapshot.py`），`refresh_load_budget` 被节流到 20 ms（`data_parallel_controller.py:335-349`）。注释把坑写全了：突发到来时每个请求都刷新预算，会把刚加的推测 +1 覆盖掉，**整批请求全落到同一个 rank**；所以 20 ms 窗口内只用推测计数。
3. **它解决不了「位」**。DP 副本各有自己的 radix cache，一条请求的前缀在哪个 rank，controller 根本不知道——那是[前端 Router]({{< relref "sglang-09-frontend" >}})的 cache-aware 负载均衡要管的。controller 只能均「量」。

多机时第 2 条会变形：`/dev/shm` 不跨机，多机 DPA 走 ZMQ 把各 rank 快照 PUSH 到 node 0 再落本地 shm（`load_snapshot.py:75-91`，`should_use_zmq` 的条件正是 `attn_dp_enabled and nnodes > 1`）。

## 8. 多机：`--nnodes` 与「别把 TP 跨机」

`--nnodes` / `--node-rank` / `--dist-init-addr`（`fields/parallel.py:36-48`）里只有第一个会流进拓扑决策：

- `(tp_size × pp_size) % nnodes == 0`（`arg_groups/validation_hook.py:182-184`）。`--tp 16 --nnodes 2` 合法，`--tp 12 --nnodes 8` 当场炸。
- **`--dp-size > 1` 且 `nnodes != 1` 直接被拒**，报错文案就是建议：「use attention DP (`--attn-dp-size`) across nodes」（`validation_hook.py:202-205`）。跨机的「数据并行」在 SGLang 里只有一个正规写法：把 `tp_size` 撑到跨机，再让 `attn_dp_size = tp_size` 把 `attn_tp_size` 压到 1，这样跨机链路上只剩 MoE 的 all-to-all。
- 反过来，`attn_tp_size > 1` 又跨机，就是在 IB 上做每层一次的 all-reduce。第 2 节那 6.1 MiB/层，NVLink 上约 14 µs，单张 400 Gb/s IB（约 50 GB/s）上约 128 µs，61 层就是 7.8 ms/步，比一步的计算还长。**`TP ≤ 单机卡数`** 的量纲就在这里。
- EPLB 也跟着 `nnodes` 变：层次装箱的 `num_nodes` 直接取它（`expert_location.py:199`）。

## 9. 与 vLLM 对照

基线 vLLM `c7a7607`。

| 维度 | vLLM | SGLang |
|---|---|---|
| DP 是什么 | 一整个引擎副本，`data_parallel_size` 个 EngineCore，外部 LB 或 `DPCoordinator` 波次 | `--dp-size`（副本）或 `--attn-dp-size`（TP 组内的注意力维），后者是主推路径 |
| 「attention 按 DP、专家按 EP」 | 支持，形态就是 DP+EP：每副本跑自己的 attention | 支持，但 DPA 是**同一条 TP 组内**的坐标系，不是独立引擎 |
| 谁来对齐 batch | **worker**：`coordinate_batch_across_dp`，一次 all-reduce 交换 4 列（原 token 数、padding 后 token 数、是否 ubatch、cudagraph mode），再 pad 到 max（`vllm/v1/worker/dp_utils.py:57-76`、`:190-263`） | **scheduler**：`prepare_mlp_sync_batch_raw` 交换 11 列，造 idle batch、强制 mode 一致、必要时 decode→extend（`dp_attn.py:518`） |
| 对齐粒度 | token 数（pad 到 max）；没有 forward mode 这一列 | token 数 **+ forward mode + CUDA Graph 能力 + TBO 切分点** |
| 批内重叠 | `--enable-dbo` / `--ubatch-size`（`vllm/config/parallel.py:275-277`），两个 CPU 线程 | TBO，stage 列表 + `delta_stage`，DP 一致性靠 `local_can_run_tbo` 取 min |
| EPLB | `vllm/distributed/eplb/`：`migration_scheduler.py` 把迁移排成「每 rank 每批只跟一个对端说话」的批次（`:20-46`），`eplb_utils.py` 的 `CpuGpuEvent` 给**异步 EPLB 线程**用 | `srt/eplb/`：同步 P2P，按层分块，卡在调度步里；另有 LPLB 在线解 LP |
| 冗余副本数 | `(逻辑专家 + 冗余) % ep_size == 0` 自动推（`vllm/config/parallel.py:660-679`） | `--ep-num-redundant-experts` 手填（`fields/exec_.py:940-942`） |

**两边都有 EPLB，别写成「只有某家有」。** 真正的差别两条：

1. **DP attention 是不是默认路径**。SGLang 把它做成 TP 组内的一个维度，配 `--attn-dp-size` 就生效，代价是全组 batch 耦合；vLLM 没有「DPA」这个开关，等价效果靠 DP+EP 得到，代价是每副本一个完整引擎进程和一套跨进程协调。
2. **谁来同步 batch**。vLLM 在 worker 的 `execute_model` 里 pad token 数，调度器不知情；SGLang 在 scheduler 里同步，因此能把「forward mode 一致」一起投票——这既让 CUDA Graph 更容易命中，也让一个 rank 的 prefill 拖住所有 rank。

## 10. 常见坑

- **`--dp-size 8 --attn-dp-size 8` 起不来**：两者互斥（`parallel_hook.py:290-295`）。要 8 组 attention DP，正确写法是 `--tp 8 --attn-dp-size 8`。
- **`--ep-size` 填了没用**：deepep/mooncake/nixl/flashinfer 这些后端下它被强制等于 `tp_size`（`overrides.py:1690-1699`）；只有 `--moe-a2a-backend none` 允许 `ep < tp`。
- **`--chunked-prefill-size` 开 DPA 后变小了**：代码主动除以 `attn_dp_size`（`parallel_hook.py:332-338`），这是设计不是 bug；但按全局 batch 调参的人会看到吞吐和预期不符。
- **EPLB 一开就卡一下**：默认 `--eplb-rebalance-layers-per-chunk` 是 `None`，等价「所有层一次搬完」（`eplb_manager.py:240-245`）。给 4 或 8 把停顿摊开。
- **EPLB 到点就重排**：默认阈值 1.0 让 `_check_rebalance_needed` 恒真（`eplb_manager.py:225-238`）。
- **`--ep-dispatch-algorithm lp` 只认 7 个模型家族**（`lplb_solver.py:33-42`）：别的模型不是「效果差一点」，是直接 `NotImplementedError`，因为空 token 提前返回会让 EP 组的 all-reduce 死锁。
- **`--nnodes` 填错不会报错，只会让 EPLB 装错箱**：它就是 `num_nodes`（`expert_location.py:199`）。

## 11. 读代码时的三个检查点

1. **这个 collective 属于哪套坐标系？** 看到 `attn_tp_all_reduce` 就问「组多大」——`attn_dp_size = tp_size` 时它是 1，`parallel_state.py:708-709` 直接返回，profile 里没有 kernel。看到 `moe_ep_group` 就反过来：它常常就是整个 tp 组。
2. **这一步的 batch 是「本 rank 的」还是「全组的」？** `batch.global_num_tokens` 是列表（每 DP rank 一个），`batch.batch_size()` 是本 rank 的。DPA 下所有 `dp_gather`/`dp_scatter` 的偏移都由前者算，混用会读到别人的 KV。
3. **这次重排是同步还是异步？** SGLang 的 EPLB 全在调度步里（`eplb_manager.py:96` 是普通生成器，`srt/eplb/` 下没有任何线程）；vLLM 有独立的异步迁移线程。判断标准很简单：`grep thread python/sglang/srt/eplb/` 零命中。

## 小结

- **四种切法**：TP 切层内权重、PP 切层、DP 切请求、EP 切专家；DP attention 是「在 TP 组内切注意力维」，不能与 `--dp-size` 同开。
- **一个 world，两套坐标系**：`attn_tp_size = tp // attn_dp // attn_cp`，`moe_tp_size = tp // ep // moe_dp`；`--ep-size` 在真 a2a 后端下被强制等于 `tp_size`。
- **DP attention 省的是 KV 的 tp 倍复制**（MLA 只有 1 个 kv 头，`tp=8` 就是 8 份），外加每层每卡 3.5S → 1.75S 的通信（attention 的 all-reduce 消失，MLP 换成 gather+scatter，字节打平）；代价是每步一次 11 列跨组对齐，以及「一个 rank 的 prefill 拖住所有 rank」。
- **EPLB 三段**：环形窗口统计（窗口 = 重排间隔）、层次贪心装箱（非层次只是 `num_groups=1,num_nodes=1` 的退化）、同步 P2P 分块搬迁（默认一块，一步搬完）。
- **DP 分发的四种策略都解决不了「前缀在哪」**：round-robin 均个数、`total_tokens` 均 token 量，缓存局部性归前端 Router。
- **跨机只有一种正规写法**：`--dp-size > 1` 不允许跨机（`validation_hook.py:202-205`），要跨机就把 `tp` 撑到跨机、`attn_dp_size = tp_size`，让跨机链路上只剩 MoE 的 all-to-all。

下一篇收尾，把整个系列的账本并排放在同一套硬件上逐条对 vLLM：[与 vLLM 的总对比]({{< relref "sglang-12-vs-vllm" >}})。进程模型见[第 1 篇]({{< relref "sglang-01-architecture" >}})，TBO 与 CUDA Graph 见[第 5 篇]({{< relref "sglang-05-overlap" >}})，PD 分离见[第 10 篇]({{< relref "sglang-10-pd-disagg" >}})。

---

参考：仓库内 `docs/docs/advanced_features/expert_parallelism.mdx`、`docs/docs/advanced_features/dp_dpa_smg_guide.mdx`；DeepSeek 官方 [EPLB](https://github.com/deepseek-ai/EPLB)；vLLM [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/)。
