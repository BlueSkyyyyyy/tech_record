---
title: "SGLang 研读（十）PD 分离与 HiCache：把 KV 搬到别的机器、下一层"
date: 2026-11-03T20:00:00+08:00
draft: false
weight: 11
series: ["sglang"]
tags: ["SGLang", "系列", "推理框架", "PD 分离", "HiCache", "KV 传输"]
categories: ["推理框架"]
---

decode 实例上跑着 300 个请求，下一步 decode 要再问池子要 300 个 slot。就在这一步之前，一个 70B 的 8k prompt 在 prefill 实例上算完了，2.5 GiB 的 KV 正沿着 RDMA 朝这边飞——落点是 decode **在两百毫秒前就划好、并且这两百毫秒里一直不许别人碰**的那批页。

这句话里藏着 PD 分离全部的实现难点：**KV 的落点必须先于数据存在，而且要在数据到达之前被冻结**。单机调度器只在分配那一刻看一眼池子；分布式调度器要和几百毫秒之外的另一台机器对同一批 slot 达成一致。

前置阅读：[RadixAttention]({{< relref "sglang-02-radix-attention" >}}) 讲树怎么找前缀，[KV 池与 slot 分配器]({{< relref "sglang-03-memory-pool" >}}) 讲 slot 从哪来。本篇只讲两件这两篇没讲的事：**同一个池子的 KV 怎么跨机交接**，以及**池子之外还有两层（host、storage）怎么接上去**。对位篇是 [vLLM 的 KV 传输]({{< relref "vllm-11-kv-transfer" >}})，本篇会在最后逐条对照。

## 1. 角色不是插件，是调度器的两种循环

SGLang 的 `DisaggregationMode` 只有三个值：`NULL` / `PREFILL` / `DECODE`（`disaggregation/mode.py:8`）。但 `Scheduler` 只有一个类，它同时继承两个混入（`managers/scheduler.py:486`）：

```python
class Scheduler(
    SchedulerDisaggregationDecodeMixin,
    SchedulerDisaggregationPrefillMixin,
    ...
```

启动时按模式挑一条事件循环（`managers/scheduler.py:6510` 起）：prefill 走 `event_loop_overlap_disagg_prefill`，decode 走 `event_loop_overlap_disagg_decode`，两者都是把 `get_next_batch_to_run` / `run_batch` / `process_batch_result` 这一套换成 PD 版本。**KV 搬运不是一个 hook，而是循环里的一个 phase**：`process_disagg_prefill_transfer_queue`（`disaggregation/prefill.py:1223`）和 `pop_transferred_with_outcomes`（`disaggregation/decode.py:3190`）就在调度步里被调用，poll 结果直接决定这一步能不能出 batch。

| 角色 | 入队 | 每步多做的事 | 出队 |
|---|---|---|---|
| `prefill` | `PrefillBootstrapQueue`（bootstrap 完成才进 waiting） | 逐 chunk 送 KV；末尾送 aux | `disagg_prefill_transfer_queue` poll 到 Success 才回包 |
| `decode` | `DecodePreallocQueue`（预分配 + 发元数据） | poll 传输；HiCache 本地恢复 | `DecodeTransferQueue` commit 进 running |

真正的插件是**搬运实现**：`TransferBackend` 五个值（`disaggregation/utils.py:764`），CLI 暴露六个选项（`arg_groups/choices.py:123`，多出来的是 `mooncake_tcp`——把 mooncake 强制成 TCP 传输，见 `arg_groups/pd_disaggregation_hook.py:30`）。换后端只换 `Manager`/`Sender`/`Receiver`/`BootstrapServer` 四个类（`KVArgs` 共用），调度器一行不改。

还有一个副产品：池子不随角色重建。`disaggregation/role_switch.py:3` 写得很清楚——token KV 池是角色无关的，翻转时只拆装 disaggregation 结构。所以 `--enable-pd-role-switch`（`arg_groups/fields/disagg.py:121`）能在运行时把 P 改成 D，同一个池子里刚算完的 KV 直接变成可 decode 的。

## 2. 三段协议：谁先动，谁等谁

一次 P→D 交接有三个阶段（外加一次收尾），控制面和数据面是两条独立的路：

```text
router   同一个请求发两份：bootstrap_host=P、bootstrap_room=R（同一个 R）
   │
   ├─> D  receiver 连 P:8998 登记自己的控制端点            ← 阶段 1 bootstrap
   │      P 侧 poll Bootstrapping → WaitingForInput
   │
   ├─> D  _pre_alloc 先占 slot，再发元数据                  ← 阶段 2 metadata（ZMQ）
   │      dst_kv_indices（D 的页号）+ dst_aux_index + decode_prefix_len
   │
   └─> P  逐 chunk prefill，NIXL WRITE 进 D 的页            ← 阶段 3 data（RDMA）
          最后一块追加 send_aux：首 token + logprobs
```

对应 `KVPoll` 五个状态（`disaggregation/base/conn.py:123`）：`Failed=0`、`Bootstrapping=1`、`WaitingForInput=2`、`Transferring=3`、`Success=4`；P 的 sender 和 D 的 receiver 各自维护一份，由 ZMQ 状态消息对齐。

**阶段 1**：P 为每条请求建一个 sender，在 `DeferredBootstrap`（`disaggregation/common/bootstrap.py:27`）里开一个「房间」并把自己登记成 owner。D 的 receiver 连同一个房间 `register`，把自己的控制端点交给 P。P 的 `pop_bootstrapped`（`disaggregation/prefill.py:519`）poll 到 `WaitingForInput` 后才调 `finalize_bootstrap`（`disaggregation/prefill.py:440`），从 sender 里取出 `decode_prefix_len`，写成 `req.start_send_idx`。**注意这一句：要发多少，是 D 在 bootstrap 时告诉 P 的，不是 P 自己算的。** 后面第 5 节会说明为什么这决定了前缀树在 PD 下的全部语义。

**阶段 2**：D 先分配，再通知。`pop_preallocated`（`disaggregation/decode.py:1440`）挑出能放下的请求，`_pre_alloc`（`disaggregation/decode.py:2509`）拿 req 行 + KV 页，然后 `_send_kv_metadata`（`disaggregation/decode.py:2049`）把 `TransferInfo`（`disaggregation/nixl/conn.py:285`）发给 P。这个结构体里最重要的是 `dst_kv_indices`——**D 自己的页号**，以及 `dst_aux_index`——D 上某个 metadata buffer 槽位。

**阶段 3**：P 只写，不分配。`_send_kv_chunk`（`disaggregation/prefill.py:1620`）把 `[start_send_idx, end_idx)` 的 slot 翻成页号，交给 sender。NIXL 路径上这一步是 `agent.make_prepped_xfer("WRITE", ...)`（`disaggregation/nixl/conn.py:2160`）；mooncake 路径上是 `engine.batch_transfer_sync(session_id, src, dst, lens)`（`disaggregation/mooncake/conn.py:835`）。两者都是 **P 主动写 D**，SGLang 没有 pull 模式。

**元数据有两路，不能混为一谈**：

| 路 | 载体 | 方向 | 内容 | 什么时候到 |
|---|---|---|---|---|
| 控制面 | ZMQ 短消息 | D → P | 页号、aux 槽位、`decode_prefix_len`、端点 | 传输前（P 靠它才知道往哪写） |
| 数据面 | DRAM→DRAM RDMA WRITE | P → D | 首个输出 token、logprobs、`bootstrap_room` | 传输末尾，和 KV 同一个 batch |

第二路容易被忽略，但它必须存在：prefill 算完最后一个 token 会采样出**第一个输出 token**，decode 要从这个 token 接着生成，而这个 token 不在 KV 里。所以 `MetadataBuffers` 的注释写得很直白——「我们把首个输出 token 的元数据传给 decode，RDMA 最小 64 字节，所以每行 pad 到 16 个 int32」（`disaggregation/utils.py:396` 起的类）。

它同时兼任一个**到达门**：`_apply_metadata_gate`（`disaggregation/utils.py:238`）在 poll 汇总前把 `Success` 降级成 `Transferring`，判据是 metadata buffer 里的 `bootstrap_room` 字段是否非 0——也就是 P 到底有没有把这行写过。没有这个门，KV 先落地、首 token 后到，decode 会拿一个空 token 去跑第一步。

## 3. decode 侧的「预留」：分布式版的准入判断

单机调度器问的是「这一步的 token 预算够不够」。PD 的 decode 侧问的是**同一件事的另一个版本**：「这批正在飞的 KV，落点够不够，而且够到它们落地之后还能不能被接着用完」。

三个数字先摆出来：`--num-reserved-decode-tokens` 默认 **512**（`arg_groups/fields/disagg.py:109`），`CLIP_MAX_NEW_TOKEN` 默认 **4096**（`disaggregation/decode.py:180`），单请求准入要的是

$$
\text{need} = \max\left(\text{alloc} + R,\ \text{fill} - \text{prefix} + \min(\text{maxNew}, 4096) - \text{retractable}\right)
$$

`_required_admission_tokens`（`disaggregation/decode.py:580`）就是这一行。第一项是「已经给的页 + 每请求 512 token 的增长余量」，第二项是「填满上下文还要多少，减去可以被 retract 的部分」。池子侧的对应量是

```python
allocatable_tokens = available_size - max(reserved_tokens, need_space_for_single_req)
```

在 `_allocatable_token_budgets`（`disaggregation/decode.py:2386`），其中 `reserved_tokens = 512 × n_active`，`n_active` 把四类请求都算上：running batch、transfer queue、waiting queue、本轮已预分配的（`_active_req_count`，`disaggregation/decode.py:2314`）。

**为什么不能「等传输完再看有没有地方」**。`pop_preallocated` 里有一段注释把这个风险写死了（`disaggregation/decode.py:1463` 起）：必须保证「在途 token + 可分配 token > 每个在途请求的 input+output 长度」，否则会出现「一个请求在 decode 时 OOM，而其他请求全卡在 transfer queue 里、**不可 retract**」的死局。retract 需要能重算或能备份，而 transfer queue 里的请求连 KV 都还没有。

这就把单机的两道闸（token 闸、行闸）升级成了四本账：

| 账 | 单机 | PD decode |
|---|---|---|
| 已占 | running batch | running + transfer queue + waiting + 本轮预分配 |
| 预留 | 无（retract 兜底） | `512 × n_active` |
| 可回收 | `evictable_size` | 同上 + decode 侧 radix 的 evictable（开了才有） |
| 硬约束 | `available_size` | `available_size`，但**在途的页已从池子里扣掉** |

最后一行是关键：在途请求的页已经在 `available_size` 之外了，所以「看起来够」和「真的够」之间差着全部在途 KV。

## 4. HiCache：把池子往下接两层

`mem_cache/README.md:23` 一张图说完了分层：`pool/`（device，L1）→ `pool_host/`（host，L2）→ `storage/`（L3）。PD 和 HiCache 能拼在一起，是因为 decode 侧把「L1 命中」「L2 命中」「L3 命中」折成同一个数（后面第 5 节）。

| | 介质 | 谁可见 | 容量默认 | 命中省什么 |
|---|---|---|---|---|
| L1 | HBM，`KVCache` | 本实例 | `mem_fraction_static` 算出来的 | 什么都不用做 |
| L2 | pinned host，`HostKVCache`（`mem_cache/pool_host/base.py:204`） | **本进程**，跨实例不可见 | `--hicache-ratio` 默认 2.0（cache 模式） | 一次 H2D，省一次 prefill |
| L3 | `HiCacheStorage`（`mem_cache/hicache_storage.py:333`） | 看后端：file 是本机盘，mooncake/hf3fs/nixl 是集群 | `--hicache-size` 或后端自己的 | 一次 L3→L2 + L2→L1，省一次 prefill |

L2 那一列的「本进程」值得单独说：`HostKVCache` 的 buffer 就是这个进程分配、再用 `cudaHostRegister` 钉住的（`mem_cache/pool_host/common.py:434` 的 `alloc_with_host_register`），两个实例即使在同一个节点上也读不到对方的 L2。**PD 拆开之后，P 的 L2 对 D 是不存在的**——跨实例复用只能走 L3。这是拆分的一笔隐性成本，也是 HiCache 在 PD 场景里突然变重要的原因。

各层的代价是纯物理的：

| 链路 | 单向带宽（量级） | 搬 1 GiB |
|---|---|---|
| HBM 内部 | ~3.3 TB/s | 0.3 ms |
| NVLink 4（同机跨卡） | ~450 GB/s | 2.4 ms |
| PCIe Gen5 x16（L1↔L2） | ~50 GB/s | 21 ms |
| 400 Gb/s IB（goodput 85%） | ~42 GB/s | 25 ms |
| 100 Gb/s 以太 | ~10.6 GB/s | 101 ms |
| 本地 NVMe | ~7 GB/s | 153 ms |

三个模型的 8k prompt 分别是 1 GiB（8B，128 KiB/token）、2.5 GiB（70B，320 KiB/token）、549 MiB（61 层 MLA，68.6 KiB/token，数字来自[第三篇]({{< relref "sglang-03-memory-pool" >}})）。于是 70B 的一次跨机传输约 63 ms、一次 L2 hit 的 H2D 约 54 ms。

这笔搬运值不值，和重算比：

$$
\frac{t_{\text{move}}}{t_{\text{prefill}}} = \frac{\text{KV/token} \div B_{\text{link}}}{2 N_{\text{param}} \div (\text{MFU} \cdot F_{\text{peak}})}
$$

8B、128 KiB/token、42 GB/s：搬一个 token 3.1 μs。同样一个 token 的 prefill 是 $2 \times 8\times10^9$ FLOP，H100 按 45% MFU（约 445 TFLOP/s）算是 36 μs。**搬比算便宜约 11 倍**，和 70B 上量出来的比值（6.55 μs vs 78.7 μs，见对位篇）是同一个数量级。HiCache 的收益全在这个比值上，命中率只是把它乘一遍。

几个默认值决定了它好不好用：`--hicache-write-policy` 默认 `write_through`（每算出一页就写一页 L2，`arg_groups/fields/memory.py:123`），`--hicache-io-backend` 默认 `kernel`（用 kernel 搬，不是 `cudaMemcpy`，`arg_groups/fields/memory.py:130`），`--hicache-mem-layout` 默认 `page_first`。写入是后台线程（`managers/cache_controller.py:1552` 的 `backup_thread_func`），读取也是（`:1438` 的 `prefetch_thread_func`），但**它们和 forward 抢同一条 PCIe**——`start_loading`（`managers/cache_controller.py:1125`）里专门有一句 `wait_stream(load_fence_stream)`，就是为了不让回填读到一个还在被 forward 写的页。

回填不是整块等：`submit_host_to_device`（`mem_cache/l2_transfer.py:196`）按 `for layer_id in range(...)` 逐层下一次拷贝，每层结束回调 `on_layer_done(layer_id)`；forward 侧对应的是 `wait_for_layer_loads`（`model_executor/runner_backend_utils/hicache_graph_events.py:46`）。第 i 层落地，第 i 层就能算——这是 HiCache 能藏在关键路径后面的唯一原因。

L3 的 key 是另一套坐标系。它不是 token 序列，而是**按页切块后的链式哈希**：`get_hash_str(token_ids, prior_hash, page_size)`（`mem_cache/utils.py:91`），`prior_hash` 把前一页的摘要串进来；命名空间用 `extra_key` / `cache_salt` 做种子（`storage_namespace_seed`），所以和 radix 的命名空间是同一套。unified 方案把 key 写成 `{page_hash}_{digest}_L{start}-{end}[_H{head_group}]`（`mem_cache/hicache_key_scheme.py:20`），坐标只描述「这份字节省了什么」，不描述「谁写的」——TP2 的 rank 存的对象正好是两个 TP4 rank 对象的并集。

代价是**页号即对象数**。`--page-size 1` 时 8k prompt 就是 8192 个对象、8192 次 EXISTS，官方 HiCache 示例统一配 `--page-size 64`（`docs/docs/advanced_features/hicache_best_practices.mdx:20`），对象数降到 128。L3 的元数据也不缓存——树只在访问时实时向后端查；查询命中不足 256 个 token 就只查不搬，不触发 prefetch（`prefetch_threshold` 默认 256，`mem_cache/unified_radix_cache.py:530`）。

## 5. 前缀树在 PD 下：三条，每条都能被误解

**第一条：P 侧命中 radix，省的是算，不是传。** 传输区间在 `finalize_bootstrap` 里就被 `decode_prefix_len` 定死了（`disaggregation/prefill.py:440`），和 P 的树里有什么无关。P 命中了前缀，只是不用重算那段；那段 KV 还得一页不少地写进 D。`maybe_send_cached_prefix_chunk`（`disaggregation/prefill.py:1552`）做的是**提前发**——把已经躺在 device 上的前缀在本步 forward 之前就发出去，和 forward 并行，而不是省掉。

**第二条：D 侧命中 radix，才真的省传。** 打开 `--disaggregation-decode-enable-radix-cache`（还要同时开 `--enable-hierarchical-cache`，见 `managers/scheduler.py:560` 的 `enable_decode_hicache`），decode 就会在预分配时做一次 `match_kv_cache`，得到 `DecodePrefixMatch`（`disaggregation/decode_hicache_mixin.py:28`）：

```python
@property
def decode_prefix_len(self) -> int:
    return self.l1_prefix_len + self.l2_host_hit_length + self.l3_storage_hit_length
```

这个和（`disaggregation/decode_hicache_mixin.py:41`）就是发给 P 的承诺：「这 3000 个 token 我本地有（或者马上能从 L2/L3 恢复），你别发」。P 于是只发 `[3000, 8192)`。全命中时 `dst_kv_indices` 是空数组——但**这不等于空传输**，NIXL 里专门有一段注释说这种情况仍然是「非 dummy」，因为 aux/state 还得送（`disaggregation/nixl/conn.py:316`）。

**第三条：树节点的 `hash_value` 是给 L3 用的。** `TreeNode` 除了 `key`/`value`，还有一个惰性填的 `hash_value: Optional[List[str]]`（`mem_cache/radix_cache.py:265`），切分时跟着 `key` 一起切。树负责回答「最长前缀在哪、在 L1 还是 L2」，哈希链负责回答「这段字节在 L3 的哪个 key 下」。同一份 token 序列，两种编码，缺一个就连不起来。

L2/L3 命中还有一条时间线：D 承诺的时候，设备页其实还没填。所以这批 token 要预留（`_hicache_pending_restore_tokens`，`disaggregation/decode_hicache_mixin.py:166`），恢复是异步的，恢复失败就是 `KVPoll.Failed`——`HiCacheRestoreGatedKVReceiver`（`disaggregation/decode_hicache_mixin.py:177`）把本地恢复的状态塞进 KV 的 poll 里，`Success` 必须等两边都成立。这就是为什么第 3 节的准入要额外减掉 `hicache_reserved_tokens`（`disaggregation/decode.py:2405`）。

## 6. 失败模式：没有重试，只有「谁先发现」

先把结论说了：**SGLang 的 PD 传输失败一律整条请求失败，没有「传输失败就本地重算」的开关**。唯一的 `recompute` 路径是 retraction 之后的 rebootstrap（换了权重才需要重算），不是传输失败的回退。

| 场景 | 谁先发现 | 动作 | 请求结局 |
|---|---|---|---|
| bootstrap 超时 / 对端死 | P 的 `pop_bootstrapped` poll 到 `Failed`；超时 300 s（`SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT`） | `handle_bootstrap_failure`（`disaggregation/prefill.py:1440`）：abort sender、还 metadata buffer、`release_kv_cache(checkpoint=False)`、500 | 失败，KV 也丢 |
| P 算完、传输失败 | P 的 `process_disagg_prefill_transfer_queue`（`disaggregation/prefill.py:1223`） | `handle_inflight_transfer_failure`（`disaggregation/prefill.py:1369`）：`release_kv_cache(checkpoint=True)`、500 | 失败，**但算好的 KV 进了树**，下一个同前缀请求还能命中 |
| D 侧接收失败 | `pop_transferred_with_outcomes`（`disaggregation/decode.py:3190`） | `_abort_failed_transfer`；后端支持 deferred release 时先 hold 住页 30 s，等 P 的 abort ack 再放 | 失败；hold 是为了不让 P 还在写的页被复用 |
| 元数据没到但 KV 到了 | `_apply_metadata_gate`（`disaggregation/utils.py:238`） | 把 `Success` 降级成 `Transferring` | 等下一轮 poll |
| D 侧 OOM | `retract_decode`（`managers/schedule_batch.py:2884`） | 按 `--disaggregation-decode-retraction-backup` 备份：`cpu_tensor` 存 CPU 张量、`host_pool` 用预留的 HiCache host 池、`none` 不备份 | `host_pool` 耗尽→500；`none`→503 让客户端重试 |
| 权重更新后 retract | decode 需要在新权重下重建前缀 | rebootstrap：D 反向调原 P 的 `/generate` 重算（`disaggregation/common/conn.py:1226`） | 可恢复 |

表里有两处设计值得注意。一是**失败时 checkpoint 的取舍不同**。bootstrap 失败切 `checkpoint=False`（什么都没算，没得存）；传输失败切 `checkpoint=True`（算完了，KV 是有效资产，留在树里）。同一个函数名、一个布尔值，含义是「这次失败有没有留下值钱的东西」。

二是**deferred release 是一个反直觉的 30 秒**。D 已经判定失败，但不能立刻把页还给池子——P 可能还有 sibling rank 的 RDMA 在飞，页一被复用就会写到别人的 KV 上。所以 NIXL 要等 P 把 abort 排空并回一个 ack（`SGLANG_DISAGGREGATION_DEFERRED_DECODE_KV_RELEASE_TIMEOUT` 默认 30 s）。没有这个 ack 的后端只能靠超时兜底。

## 7. 与 vLLM 对照

两边都把「KV 搬运」抽象成了一个可替换的层，但**抽象的位置不一样**：

| 维度 | vLLM @ `c7a7607` | SGLang @ `1ef3335` |
|---|---|---|
| 抽象 | `KVConnectorBase_V1` 双角色插件（`vllm/distributed/kv_transfer/kv_connector/v1/base.py:181`） | `Scheduler` 的两个 mixin（`managers/scheduler.py:486`）+ 五个可替换的 KV 类 |
| 搬运动作 | connector 是 hook，调度器只调 `get_num_new_matched_tokens`（同文件 `:523`） | poll 就在调度步里，结果决定本步出不出 batch |
| 方向 | 默认 pull（D 读 P，`vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py:153` 的 `transfer_mode="pull"`），push 可选 | 只有 push（P 写 D，`disaggregation/nixl/conn.py:2160` 的 `"WRITE"`） |
| 元数据 | `kv_transfer_params` 随请求体走，输出再带回来（`vllm/outputs.py:208`） | ZMQ 控制面 `TransferInfo` + DRAM 上的 metadata buffer |
| 传输失败默认 | `kv_load_failure_policy="fail"`（`vllm/config/kv_transfer.py:101`），可改 `recompute` | 一律 abort；`recompute` 只用于 retraction |
| 缓存分级 | offload 的 `Medium` 两级：`CPU` / `STORAGE`（`vllm/v1/kv_offload/base.py:48`） | L1/L2/L3 三级，且是 `mem_cache` 的一等公民：`pool/` → `pool_host/` → `storage/` |

- **「谁把搬运做成调度器的一部分」**：SGLang 是。`process_disagg_prefill_transfer_queue` 在调度循环里同步 poll 一批请求，`Transferring` 就留在队列里，`Success` 才回包。vLLM 是「调度器问 connector 要一个数字（`num_external_computed_tokens`），然后照常调度」，传输的异步性被 `WAITING_FOR_REMOTE_KVS` 状态吸收掉。前者把延迟摊在每一步，后者把延迟藏进状态机。
- **push 和 pull 的代价不一样**。pull 是 D 主动读，D 可以在自己想要的时候发起，但 P 必须把自己的页钉住等 D 来读（vLLM 用租约 + 心跳解决）。push 是 P 主动写，P 不用等 D，代价是 **D 必须提前把页交出去并冻结**——这就是第 3 节那 512 个 token 存在的理由。SGLang 选了 push，所以它的准入判断必须比 vLLM 更保守。
- **失败语义的差别是设计选择，不是遗漏**。vLLM 给了一个全局开关，默认还是「失败就失败」，说明作者认为重算的代价通常不划算。SGLang 干脆不提供这个开关，但它在 P 侧保留了已算好的 KV（`checkpoint=True`），把「回滚」变成「这次失败，下次命中」。
- **PD 和分层缓存在 SGLang 里是同一套抽象**。decode 侧的 HiCache 就是 `match_kv_cache` 多返回两个长度（`disaggregation/decode_hicache_mixin.py:41`），而 `--disaggregation-decode-enable-offload-kvcache` 用的是同一个 `HiCacheController`。vLLM 那边，PD 走 `kv_connector/`，分层缓存走 `kv_offload/`，是两套接口，靠 `OffloadingConnector` 桥接。
- **两边在事件目录上已经收敛**。SGLang 的 `BlockStored`（`disaggregation/kv_events.py:279`）和 vLLM 的 `KVCacheEvent` 用同一种 tag-msgpack 编码，`KVCacheEvent` 的 docstring 直接写了「Dynamo 这类消费者可以一套代码解两家」（`disaggregation/kv_events.py:243`）。`StorageMedium` 还比三级多一级：`GPU` / `CPU_PINNED` / `DISK` / `EXTERNAL`（`disaggregation/kv_events.py:261`）——最后那个是 Mooncake 这类共享池，代码里叫 L4。

一句话：**vLLM 把「KV 从哪来」做成插件，SGLang 把「我现在是 P 还是 D」做成调度器**。前者的边界更干净，后者的调度信息更全（它知道在途的页、知道设备页和 host 页的对应关系），代价是 P/D 两条循环各自长出自己的状态机。

## 8. 常见坑

1. **只配 `--disaggregation-mode` 不配后端**。默认是 `mooncake`（`arg_groups/fields/disagg.py:65`），在只装了 NIXL 的环境里会在建 manager 时炸。显式写 `--disaggregation-transfer-backend nixl`。
2. **以为 P 的 radix 命中会省传输**。不会。省传输的只有 D 的命中。看到「P 上命中率 90% 但网络还是满的」，不是 bug。
3. **把 `--num-reserved-decode-tokens` 调小以多塞请求**。512 是给 running batch 的增长留的，调小之后 transfer queue 会把预算吃光，然后 running 的请求 decode 时无处可去——而它们不可 retract。
4. **HiCache 不配 `--page-size`**。L3 的对象数等于 token 数除以 page_size；`page_size=1` 时 8k prompt 是 8192 个对象。
5. **以为 L2 能跨实例共享**。L2 是进程私有的 pinned 内存；两实例同机也不行，跨实例只能走 L3。
6. **开了 HiCache 又抱怨 TTFT 变长**。`write_through` 默认会把每一页都往 L2 写，PCIe 上行和 PD 的 KV 传输抢同一条总线。读多写少的负载可以换 `write_through_selective`。

## 读代码时的检查点

1. `disaggregation/prefill.py:1620` 的 `_send_kv_chunk`，开头几行里 `start_idx = req.start_send_idx`，往上追到 `finalize_bootstrap`（`disaggregation/prefill.py:440`）的 `req.start_send_idx = decode_prefix_len`。**传多少由 D 说了算**，这条链是本篇所有结论的地基。
2. `disaggregation/decode.py:2386` 那一行减法，把 `_active_req_count`（`disaggregation/decode.py:2314`）的四个加数数一遍。这是单机 `PrefillAdder` 的分布式版本，也是「为什么不能等传输完再看」的答案。
3. `mem_cache/l2_transfer.py:196` 的 `for layer_id in range(transfer_layer_id_max)` 和 `on_layer_done(layer_id)`，再对上 `model_executor/runner_backend_utils/hicache_graph_events.py:46` 的 `wait_for_layer_loads`。层流水两端接上，HiCache 才不是串行的。

## 小结

- PD 的「角色」编在调度器里：一个 `Scheduler` 类、两个 mixin、两套 event loop（`managers/scheduler.py:486`、`:6510`）；可插拔的是搬运实现（五个钩子 × 六个 CLI 选项）。
- 三段协议：bootstrap 建房间 → D 预分配并把 `dst_kv_indices` 发给 P → P 用 RDMA **写**进 D 的页，末尾追送首 token。SGLang 只有 push，没有 pull。
- 元数据两路：ZMQ 走控制面（页号），DRAM WRITE 走数据面（首 token + logprobs），后者兼任「P 到底写没写」的到达门（`disaggregation/utils.py:238`）。
- decode 侧预留 512 token/请求，是因为在途 KV 的落点必须先冻结；`allocatable = available − max(512 × n_active, 单个请求的缺口)` 就是分布式版的准入。
- HiCache 三级：L1 HBM / L2 进程私有 pinned / L3 共享存储。搬 1 GiB 分别是 0.3 / 21 / 25 ms 量级；8B 上搬一个 token 3.1 μs，算一个 token 36 μs，比值约 11 倍。
- 前缀树在 PD 下：**P 命中省算不省传，D 命中才省传**；树节点的 `hash_value` 是通往 L3 key 的桥。
- 失败没有重试：bootstrap 失败丢 KV，传输失败留 KV（`checkpoint=True`），D 侧失败要 hold 页等 P 排空（30 s），OOM 走 retraction 备份。

下一篇回到节点内部：[TP/EP/DP-attention 与 EPLB]({{< relref "sglang-11-distributed" >}})——本篇一直在说「跨机」，下一篇说「同机多卡怎么把权重和 KV 切开」，两者的带宽账是同一本。

> 代码基线：`sgl-project/sglang` main @ `1ef3335`（2026-11-03）。对比基线：`vllm-project/vllm` main @ `c7a7607`。
