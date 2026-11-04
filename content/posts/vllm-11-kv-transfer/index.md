---
title: "vLLM 研读（十一）KV 传输：connector、P/D 分离与 offload"
date: 2026-11-03T19:00:00+08:00
draft: false
weight: 12
series: ["vllm"]
tags: ["vLLM", "系列", "推理框架", "P/D 分离", "KV Connector", "NIXL", "offload"]
categories: ["推理框架"]
---

一个 8k prompt 的 prefill，在 70B、TP=4、H100 上大约 645 ms；同一个模型的一次 decode step 大约 15 ms。混跑时这 645 ms 会整块压在某个 decode step 上：那一步的 ITL 从 15 ms 变成 660 ms。chunked prefill 能把它切成 2048 token 一块（约 161 ms），但块大小与 SLO 的对应关系只能靠调，而且尖峰永远存在。

先记住一个画面：**P 实例算完 prefill 不 decode，它把 2.5 GiB 的 KV 留在显存里等 D 来读；D 实例把这块 KV 通过 RDMA 拉进自己预分配的 block，然后接着 decode。** 这条路径上的全部代码，都挂在 `KVConnectorBase_V1` 一个抽象类上。

本篇回答三个问题：prefill 与 decode 的最优点差在哪；connector 的钩子分别在哪个进程、什么时候被调用；以及这笔传输什么时候划算、什么时候纯亏。

## 1. 为什么要拆：两种 bound 抢同一张卡

prefill 一次处理几千个 token，矩阵乘的算术强度高，瓶颈是算力；decode 每步只处理一个 token，却要把全部权重读一遍，瓶颈是显存带宽。两者的最优点几乎正交：

| | prefill | decode |
|---|---|---|
| 瓶颈 | 算力（GEMM） | 显存带宽（读权重 + 读 KV） |
| 8k / 70B | 645 ms（TP=4，MFU 45%） | 15 ms per step（batch 64） |
| 最优并行 | 大 TP/PP，把单请求摊开 | 大 batch，把权重读摊薄 |
| 敏感指标 | TTFT | ITL（p99） |
| 对网络 | 只吃 TP all-reduce | 只吃 TP all-reduce，但对抖动零容忍 |

账算在 ITL 上：8k prefill 相当于 43 个 decode step 的时间。同一个 batch 里只要进来一个长 prompt，其余请求这一步就得等 645 ms。V1 默认开 chunked prefill（`vllm/config/scheduler.py:134`），把一次 prefill 切成 `max_num_batched_tokens` 大小的块，但块大小本身是个两难：块大则 ITL 尖峰大，块小则 prefill 的 MFU 掉、TTFT 涨。官方文档把这层意思写得很直白——P/D 分离是为了**分别调 TTFT 和 ITL**、控制尾延迟，并且明确写着"不提升吞吐"（`docs/features/disagg_prefill.md:12-16`）。

类比的边界：这不是"把计算搬到另一台机器"那种拆分，KV 仍然是完整的 KV，只是换了个存放地点；一旦传输比重算还慢，拆分就只剩成本。

## 2. 接口：一个基类，两个进程角色

`KVConnectorBase_V1`（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:181`）被显式拆成两个角色（`base.py:139` 的 `KVConnectorRole`：`SCHEDULER=0`、`WORKER=1`）。factory 为同一个 engine 建两个实例，一个住在 scheduler 进程，一个住在 worker 进程（`factory.py:67-75` 的注释直接说明了这个约定）。所以**任何 connector 都要写两套逻辑，且不能共享内存**。

| 侧 | 方法（行号） | 作用 |
|---|---|---|
| scheduler | `get_num_new_matched_tokens`（`base.py:523`） | 这一步能从外部拿多少 token；返回 `(n, async)` |
| scheduler | `update_state_after_alloc`（`base.py:576`） | block 分配完后，把 block id 记进 metadata |
| scheduler | `build_connector_meta`（`base.py:602`） | 把本步的传输计划打包成 `KVConnectorMetadata` |
| scheduler | `request_finished`（`base.py:638`） | 请求结束，决定 block 是立刻释放还是等发送完成 |
| worker | `start_load_kv`（`base.py:340`） | 发起本步的加载（可异步、可放到 forward 之后） |
| worker | `wait_for_layer_load`（`base.py:357`） | 在某一层 attention 前等这一层的 KV 到位 |
| worker | `save_kv_layer`（`base.py:371`） | 一层算完就把这层 KV 发出去 |
| worker | `wait_for_save`（`base.py:393`） | 每步末尾提交/同步所有 save |
| worker | `get_transfer_results`（`base.py:423`）、`take_events`（`base.py:672`） | 汇报完成/失败、吐出 KV 事件（scheduler 消费） |

`scheduler` 侧的方法都是"算术"：只看 request 和 block，不看 GPU。`worker` 侧的方法才是"搬运"：拿到的是 `ForwardContext` 和张量。

## 3. 调用点：一次 P→D 传输的时序

scheduler 每步在给请求分配 slot 时问一次 connector（`vllm/v1/core/sched/scheduler.py:1138`）：

```python
ext_tokens, load_kv_async = self.connector.get_num_new_matched_tokens(
    request, block_aligned_local
)
```

返回的 `ext_tokens` 就是 `num_external_computed_tokens`——**"这些 token 的 KV 不用我算，别人有"**。它随即被送进 `allocate_slots`（`scheduler.py:1434-1441`），关键参数是 `delay_cache_blocks=load_kv_async`：异步加载的 block 先不登记进 prefix cache，等加载成功再登记（`vllm/v1/core/kv_cache_manager.py:610-613` 的注释就是 "P/D: delay caching blocks if we have to recv from remote"）。若加载失败，这些 block 从未进过哈希表，重算不会命中脏数据。

异步路径上，请求被塞进 `WAITING_FOR_REMOTE_KVS` 状态（`scheduler.py:1522`，状态定义在 `vllm/v1/request.py:404`），本步不调度它，只保留已分配的 block。worker 侧完成回传后，`_update_from_kv_xfer_finished`（`scheduler.py:3557`）把请求记进 `finished_recving_kv_req_ids`，下一次调度前 `_update_waiting_for_remote_kv`（`scheduler.py:3468`）才把 block 真正 cache 起来、让请求退回 `WAITING`（入口在 `scheduler.py:3526-3530`）。

worker 侧的生命周期被一个 context manager 包住（`vllm/v1/worker/kv_connector_model_runner_mixin.py:67`）：

```text
bind_connector_metadata(meta)                  # :77  这一步的传输计划
[同步] start_load_kv(fwd_ctx)                  # :85  forward 之前，KV 必须已就位
forward()                                      # 每层 wait_for_layer_load / save_kv_layer
[异步] start_load_kv(fwd_ctx)                  # :90  forward 提交之后再发，避开关键路径
wait_for_save()                                # :92
get_transfer_results(...) -> finished_recving  # :94
```

同步加载与异步加载的分叉在 `scheduler_output.has_sync_kv_loads`（`scheduler.py:1551`）：这一步的 forward 要用到外部 KV，就必须先 load；只是"顺手预热"，就把提交推迟到 forward 之后，省掉主机侧的提交延迟（`kv_connector_model_runner_mixin.py:79-83`）。

层级的钩子由装饰器注入，不侵入各 attention 后端：

```python
# vllm/model_executor/layers/attention/kv_transfer_utils.py:51,57
connector.wait_for_layer_load(layer_name)   # 进入 attention 前
result = func(*args, **kwargs)
connector.save_kv_layer(layer_name, kv_cache, attn_metadata)   # 退出后
```

这就是"逐层流水"的全部机制：P 侧第 1 层算完就能发第 1 层，D 侧第 1 层到位就能开始算第 1 层，传输和前向在同一时间轴上交错。没有这个钩子，就只能整块 KV 传完再动，P 的尾部和 D 的头部会各自空转一次。

拉还是推，是两条不同的时序。默认的 `NixlConnector` 是 D 主动读（`NIXL READ`）；`NixlPushConnector` 反过来，由 P 直接写进 D 预分配的显存（`NIXL WRITE`），D 先把 block id 注册给 P（`docs/design/nixl_kv_push_connector.md:3-7`）。push 的额外代价是一个专用线程 `nixl-push-writer` 和两侧的匹配表（同文件 `:82-84`）。

P 侧的 block 不能无限期挂着。传完之后 P 需要知道"可以释放了"，D 崩了 P 也不能把几 GB 显存锁死。当前机制是**租约 + 心跳**：prefill 结束先给 30 s 租约，D 在排队或执行期间周期性发心跳，每次续 `lease_duration * 2/3`；心跳停了，租约到期 P 自己回收（`docs/design/nixl_kv_cache_lease.md:17-27`）。心跳搭在 NIXL 的通知通道上，不另开连接。

失败回退由两处合起来完成：worker 在 `KVConnectorOutput` 里报 `failed_recving` / `invalid_block_ids`（`vllm/v1/outputs.py:222-227`），scheduler 的 `_handle_failed_recving`（`scheduler.py:3689`）按 `kv_load_failure_policy` 决定行为。默认是 `"fail"`（直接失败），配成 `"recompute"` 才会把 `num_computed_tokens` 清零重算（`vllm/config/kv_transfer.py:101`，判定在 `scheduler.py:185-186`）。**这是最容易踩的一个默认值**：想要"传输失败就本地重算"的韧性，必须显式配 `recompute`。另一条更轻的路径是 NIXL 自己的 `kv_recompute_threshold`（默认 64 token）：远端 token 太少时干脆不拉，直接在本地重算，因为握手和通知的固定开销摊不平（`docs/features/nixl_connector_usage.md:250`）。

## 4. 连接器生态与 factory

factory 是一张"名字 → 惰性 import"的表（`factory.py:28-40`），注册集中在文件末尾（`factory.py:158` 起）。查找顺序是：`kv_connector_module_path` 优先（外部模块，`factory.py:106-116`），其次内置注册表，`NixlConnector` 会按 `transfer_mode` 改写成 `NixlPullConnector` 或 `NixlPushConnector`（`factory.py:126-130`）。

| 连接器 | 传输介质 | 适用场景 |
|---|---|---|
| `NixlConnector`（pull/push） | GPU↔GPU，RDMA（UCX / GDS / LIBFABRIC） | 跨机 P/D，主流选择 |
| `MooncakeConnector` / `MooncakeStoreConnector` | RDMA + 分布式 KV store | P/D + 跨实例池化缓存 |
| `LMCacheConnectorV1` / `LMCacheMPConnector` | CPU/磁盘，可外挂 lmcache server | 多级缓存、多实例共享 |
| `OffloadingConnector` | 本机 pinned CPU（+ 二级存储） | 单实例扩容，不改拓扑 |
| `SimpleCPUOffloadConnector` | 本机 CPU / 本地磁盘 | 最简 offload，默认 8 GiB |
| `FlexKVConnectorV1` | 分布式 KV store | 超大规模、独立缓存集群 |
| `HF3FSKVConnector` | 3FS 文件系统 | 存算分离部署 |
| `MoRIIOConnector` | ROCm 上的 RDMA | AMD 平台 |
| `MultiConnector` | 组合上述任意多个 | 同时挂远端 + 本地 offload |

`--kv-transfer-config` 吃一个 JSON 字符串，字段就是 `KVTransferConfig` 的 dataclass 字段（`vllm/config/kv_transfer.py:72-104`）：`kv_connector`、`kv_role`（`kv_producer` / `kv_consumer` / `kv_both`）、`engine_id`、`kv_buffer_device`（`cuda`/`cpu`/`xpu`，默认取当前平台）、`kv_port`（默认 14579）、`kv_connector_extra_config`。只给 `kv_connector` 不给 `kv_role` 会在 `__post_init__` 里直接报错（`vllm/config/kv_transfer.py:133-137`）。

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both",
  "kv_buffer_device":"cuda","kv_connector_extra_config":{"backends":["UCX","GDS"]}}'
```

`kv_connector_extra_config` 是各家自己的方言，没有统一 schema：NIXL 放 `backends`、`kv_lease_duration`、`transfer_mode`；OffloadingConnector 放 `block_size`（每块的 token 数）和 `cpu_bytes_to_use`（`docs/features/disagg_prefill.md:38-44`）。

## 5. KV events：为什么需要一个全局目录

P/D 分离之后，一个请求的前缀散落在两个进程的显存里，还有一个"哪个实例有哪个前缀"的问题——路由层不知道 P 上有没有这段前缀，就只能盲发，命中率归零。

vLLM 的答案是**事件流**：scheduler 每步把 `kv_cache_manager.take_events()` 和 connector 的 `take_events()` 合起来，打成 `KVEventBatch` 发布（`vllm/v1/core/sched/scheduler.py:2627-2641`）。事件类型里最重要的是 `BlockStored`，它带 `block_hashes`、`parent_block_hash`、`medium`、`locality`、`ownership`（`vllm/distributed/kv_events.py:56-88`）——也就是"哪个块、挂在谁的后面、存在哪种介质、是本地的还是远端的"。有 `parent_block_hash`，消费方才能重建出哈希链（也就是前缀树），而不只是一堆散块。

发布端默认是 ZMQ PUB，带重放缓冲：`enable_kv_cache_events` 打开、`publisher` 选 `zmq`、`endpoint` 默认 `tcp://*:5557`、`hwm` 默认 100000（`vllm/config/kv_events.py:14-41`，实现见 `vllm/distributed/kv_events.py:313` 的 `ZmqEventPublisher`）。订阅方维护的正是那份"缓存目录"：全局记录每个副本上有哪些前缀。llm-d 的文档把用途写得很清楚——按 KV 事件做 prefix-aware 路由、用全局索引记录块在哪个副本上（`docs/deployment/integrations/llm-d.md:11-12`）。

为什么必须"全局"：单机 prefix cache 是进程内的哈希表，命中与否是本地判断；P/D 之后，"有没有"和"在哪"分开了。路由错一次，代价就是一次完整的 prefill（8k/70B 约 645 ms）；路由对一次，省下的也是 645 ms。这套事件流的价值全在这个差值上。

## 6. 账本一：KV 到底要传多少字节

vLLM 里的页大小公式在 `vllm/v1/kv_cache_interface.py:579-587`：

```python
# 每页字节 = head 数 × 状态数 × (head_size + head_size_v) × dtype 字节
return self.num_heads * self.num_states * self.state_content_size_bytes
```

把"每页"换成"每 token"就得到常用的手算式：

$$
\text{KV/token} = L \times H_{kv} \times (d + d_v) \times b
$$

$L$ 是层数，$H_{kv}$ 是 KV head 数，$d+d_v$ 对 MHA 是 $2 \times$ head_dim（K 和 V 各一份），MLA 则退化成单个 latent：`head_size_v` 默认 0，宽度是 `kv_lora_rank + qk_rope_head_dim`（`vllm/v1/kv_cache_interface.py:736-737`、`vllm/model_executor/models/deepseek_v2.py:520`）。

| 模型 | KV/token | 8k prompt 的 KV | 400 Gb/s（50 GB/s） | PCIe Gen5 x16（约 50 GB/s 实测） |
|---|---|---|---|---|
| Llama-3.1-8B（32L, 8kv, 128, fp16） | 128 KiB | 1 GiB | 21 ms | 21 ms |
| Llama-3.1-70B（80L, 8kv, 128, fp16） | 320 KiB | 2.5 GiB | 54 ms | 54 ms |
| DeepSeek-V3（61L MLA, 576, fp8） | 34 KiB | 275 MiB | 6 ms | 6 ms |

400 Gb/s 是线速，IB 实际 goodput 按 85% 算是 63 ms；再叠上握手（NIXL side channel，默认端口 5600）和完成通知，8k/70B 的一次传输按 **60–70 ms** 估比较实在。

这 60 ms 花得值不值，看它和重算的比较：8k prefill 645 ms 摊到每个 token 是 78.7 μs（TP=4 的墙钟时间），而传输一个 token 只要 6.55 μs。**搬比算便宜 12 倍**——这是 P/D 分离能成立的物理基础。MLA 把这个比值再拉大一个数量级：KV 小了 9 倍，传输几乎免费，代价是 attention 的计算形态变了（见[第六篇]({{< relref "vllm-06-attention-backends" >}})）。

## 7. 账本二：收益门槛

把两笔账并排：

- 混跑：TTFT = 排队 + 645 ms；其他请求的 ITL 在这一步被顶到 660 ms（43 倍）。
- 拆分：TTFT = 645 ms + 握手/传输 60–70 ms + D 侧首个 step 15 ms ≈ 725 ms（+12%）；ITL 稳定在 15 ms。

也就是说，P/D 分离是**用 12% 的 TTFT 换掉 43 倍的 ITL 尖峰**。这笔交易只在"ITL 有硬 SLO"时成立；如果你的 SLA 只写吞吐，那它纯属亏损——两套实例各自有启动、握手、心跳、租约的开销，缓存还被劈成两半。

配比也要算。按上面的速率：P 侧每 4 卡每秒吃 8192/0.645 ≈ 12.7k prompt token；D 侧每 4 卡在 batch 64 下每秒出 64/0.015 ≈ 4.3k token。于是

$$
P:D \;=\; \frac{\text{输入 token 数} / 12.7\text{k}}{ \text{输出 token 数} / 4.3\text{k}}
$$

8k 输入 / 512 输出 → 约 5:1（P 多）；512 输入 / 2k 输出 → 约 1:12（D 多）。**"P:D 该配几比几"没有通用答案，它是流量形状的函数**，这也解释了为什么各家的部署比例差得那么远。

什么时候拆分反而更慢：

1. **prompt 太短**。512 token 的 prefill 在 8B 上只有 18 ms，而握手、建连、完成通知是毫秒级的固定成本，占比可以到 20% 以上。64 token 的 prefill 在 70B 上是 5 ms——`kv_recompute_threshold` 的默认值 64 隐含的正是这个量级的固定开销。
2. **prefix 命中率高**。多轮对话、agent 场景下前缀高度复用，本地 prefix cache 几乎免费；拆开之后两个池子各自变小，命中率下降，省下的 prefill 又被重算吐回去。
3. **decode 的 batch 起不来**。decode 是带宽 bound，权重读一遍的 15 ms 与 batch 无关；batch 从 64 掉到 16，每 token 的成本就涨 4 倍。总量不变而拆成两池，D 池的 batch 必然变小。
4. **网络和 TP/EP 抢带宽**。跨机 TP 的 all-reduce 和 KV 传输走同一张 400 Gb/s 网卡，KV 一开，TP 的 all-reduce 延迟就抖，而 decode 对抖动零容忍。
5. **只想要吞吐**。官方文档明确写着 P/D 不提升吞吐（`docs/features/disagg_prefill.md:16`）；它做的是把两种 bound 的资源分开，总卡数不变时总算力也没变。

## 8. offload：拿 CPU 内存换显存

如果问题只是"显存装不下更多前缀"，那不必引入第二个实例——`OffloadingConnector` 把算完的 KV 块沉到主机内存，命中时再提回 GPU。官方文档的定位一句话说清：**它扩展的是 prefix cache**，把完成的块卸载到更慢但更大的层级（`docs/features/kv_offloading_usage.md:3`）。

底座在 `vllm/v1/kv_offload/`：`Medium` 只有 `CPU` 和 `STORAGE` 两种（`vllm/v1/kv_offload/base.py:48`），`OffloadingManager` 定义了 `lookup` / `prepare_load` / `prepare_store` / `complete_*` 这套异步语义（`vllm/v1/kv_offload/base.py:305` 起）。`vllm/v1/simple_kv_offload/` 是简化实现，`SimpleCPUOffloadConnector` 默认给 8 GiB、每个 rank 均分，可用 `cpu_bytes_to_use` / `cpu_bytes_to_use_per_rank` 改（`vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py:55,108-116`）。

边界就在带宽上：HBM 3.35 TB/s，PCIe Gen5 x16 约 50 GB/s，差 67 倍。所以 offload 只对**"命中一次就省一次 prefill"**的场景有意义——从 CPU 提回 8k/70B 的 2.5 GiB 要 54 ms，而重算要 645 ms，还是划算 12 倍；但如果命中率低、块被反复换入换出，PCIe 就会成为新的瓶颈，且和 KV 传输、权重加载抢同一条总线。

容量账：8 GiB 主机内存按 320 KiB/token 只装 26k token，约 3 个 8k prompt——聊胜于无；真正有意义的是 512 GiB 级的主机内存，能放 1.6M token；同一个 70B 实例的显存 KV 池按 40 GiB 算只有 131k token，主机侧是它的 12 倍。**offload 的收益与主机内存大小成正比，与命中率成正比，与 PCIe 带宽成反比。**

## 9. 常见坑

1. **`kv_load_failure_policy` 默认是 `fail`**（`vllm/config/kv_transfer.py:101`）。传输失败默认让请求直接报错，不是重算。要韧性就显式配 `recompute`，并知道重算路径会清零 `num_computed_tokens`。
2. **只配 `kv_connector` 不配 `kv_role`**，`__post_init__` 直接抛错（`vllm/config/kv_transfer.py:133-137`）。P 侧和 D 侧的 `kv_role` 可以都是 `kv_both`，但 `engine_id` 必须能对应上。
3. **忘了 `delay_cache_blocks`**。自己写 connector 时若把异步加载的 block 提前登记进 prefix cache，加载失败后这些 block 会带着未写入的 KV 被别的请求命中——脏读，且极难复现。
4. **同步/异步加载放错位置**。`has_sync_kv_loads` 为真时必须先 `start_load_kv` 再 forward（`kv_connector_model_runner_mixin.py:83-85`）；放反了就是拿未初始化的 KV 做 attention，输出是乱的但不报错。
5. **P 侧租约当超时用**。租约是 30 s + 心跳续期（`docs/design/nixl_kv_cache_lease.md:17`），不是"30 s 后一定释放"。D 排队很久仍会心跳，P 的显存就一直被占着——容量规划要按"在途请求数 × 每请求 KV"算，不能按租约时长算。
6. **多 KV group 的模型（混合注意力）上 `invalid_block_ids` 不可用**。block id 在 group 内唯一、跨 group 不唯一，scheduler 遇到多 group 会直接抛错，要求 connector 改用 `failed_recving` 汇报（`scheduler.py:3713-3716`）。

## 小结

- prefill 是算力 bound、decode 是带宽 bound；8k/70B 的 prefill（645 ms）等于 43 个 decode step（15 ms），混跑就是把 15 ms 的 ITL 顶成 660 ms。
- P/D 分离买到的是"TTFT 与 ITL 解耦"，价格是 12% 的 TTFT（60–70 ms 传输 + 15 ms 首步）；官方文档明说不提升吞吐。
- `KVConnectorBase_V1` 是双角色接口：scheduler 侧 `get_num_new_matched_tokens` / `update_state_after_alloc` / `build_connector_meta` 只做算术，worker 侧 `start_load_kv` / `wait_for_layer_load` / `save_kv_layer` 才碰张量，两侧不能共享状态。
- `num_external_computed_tokens` 是"别人算好的 token 数"，配 `delay_cache_blocks` 把 block 的缓存登记推迟到加载成功之后；异步路径上请求停在 `WAITING_FOR_REMOTE_KVS`。
- KV 传输量的手算式是 $L \times H_{kv} \times (d+d_v) \times b$：70B/8k 是 2.5 GiB、400 Gb/s 下约 60 ms；搬一个 token 6.55 μs，算一个 token 78.7 μs，比值 12 倍。
- 传输失败的默认行为是失败而不是重算（`kv_load_failure_policy="fail"`）；远端 token 少于 64 个时 NIXL 干脆不拉。
- offload 扩展的是 prefix cache（`Medium` 只有 CPU 和 STORAGE），收益与主机内存、命中率成正比，与 PCIe 带宽成反比：8 GiB 只装 3 个 8k prompt，512 GiB 才够看。

下一篇进入量化与调优：[把模型塞进去并跑满]({{< relref "vllm-12-quantization-tuning" >}})——KV cache 的 dtype 是那里最直接的一根旋钮，本篇所有字节数都会随它按比例缩放。

参考文档：

- [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/)
- [KV Offloading Usage Guide](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
