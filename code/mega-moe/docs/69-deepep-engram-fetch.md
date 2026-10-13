# DeepEP 的 Engram 远程抓取原语 `engram_fetch`

前面第 7/58/61 篇拆了 DeepEP 的 **EP all-to-all**（dispatch/combine），第 64 篇拆了 **bucket 集合通信**
（all-gather/reduce-scatter/all-reduce），第 68 篇拆了 **流水线并行的相邻 rank send/recv**。本篇补上
DeepEP V2.5 里**第四类通信设施**——**Engram**：一套「把远端 rank 上按 key 存放的 entry 表（KV / 记忆库）
按索引**用 RDMA 拉回本地**」的原语，`deep_ep.EngramBuffer` / `deep_ep::engram::engram_fetch_impl`
/ `engram_fetch_wait_impl`（`csrc/buffers/engram.hpp`、`csrc/kernels/engram/engram.hpp`、
`deep_ep/include/deep_ep/impls/engram/engram_fetch.cuh`、`.../layout/engram/workspace.cuh`）。

它与前三类的区别在**语义形态**：all-to-all 是「每个 rank 的数据散给所有 rank」、bucket 是「所有 rank
同步一份数据」、PP 是「环上邻居传张量」；而 Engram 是**带索引的随机远程读（indexed RDMA gather）**：
每个 rank 只在本地存「自己那部分 entry」，调用方给一个 `[num_layers, num_tokens, num_entries_per_token]`
的**全局 entry 索引**，Engram 把每个索引对应的 `hidden * elem_size` 字节从**拥有它的那个 rank** 拉回本地
接收缓冲，**每层一个完成钩子**。这是把「外部键值存储 / 记忆库 / 长上下文缓存」放进训练循环时的通信原语，
所以 README 把它归为 experimental 的 *remote memory access (Engram)*。

> **可跑性**：Engram 的**布局层**（`layout::EngramLayout` 的字节偏移 / SF 行号与分片号、
> `layout::EngramWorkspace` 的字段偏移）与**发射面**（`EngramIssueWarpLayout` 的 warp↔(peer,
> gin_context,tile) 映射，以及 issue 循环里「哪些 entry 归本 peer、落到哪个 window/源偏移/目的偏移」）
> 是纯整数映射与 `__ballot_sync`/`__popc`，**架构无关**。只有真正的 RDMA 动作
> ——`gin.get` 拉取、`gin.flush_async` 门铃、`gin.wait` 等待、以及 SF 的 `gin.get_sym_ptr<ncclTeamTagLsa>`
> ——需要 `nccl_device.h` 的 **NCCL GIN 设备 API**（本机 NCCL 2.29.7 不提供）。这部分**仅源码分析**；
> 本篇在 **H100（sm90）真机**把前者逐行抽取跑成真实设备代码。

---

## 1. 它解决什么问题：带索引的远程 gather

设想一个「记忆库」：每层有 `num_entries_per_layer[l]` 个 entry，每个 entry 是 `hidden` 维的表征；
**所有 rank 的 entry 拼成一个全局表**，第 `r` 个 rank 拥有全局下标区间
`[r * num_entries_per_layer[l], (r+1) * num_entries_per_layer[l])`。

一次 `fetch(indices)` 要做的，就是给 `num_tokens` 个 token 各取 `num_entries_per_token` 个 entry，
按 token 拼起来（每层返回 `[num_tokens, hidden * num_entries_per_token]`）。朴素做法是「全量 all-gather
所有 entry 再本地 gather」，但那样要搬 `num_ranks` 倍数据——Engram 只搬**被索引命中**的那些。

三个设计决定：

1. **数据面用 RDMA、scale-factor 面走 NVLink**。entry 本体只在**拥有它的 rank** 上，所以必须走
   RDMA（`gin.get`，跨节点也成立）；而 FP8 的 block-scale（SF）在测试里是**每个 rank 都保留一份副本**
   （`tests/engram/test_engram.py:47-52`，`all_gather_single` 在 `:50`），所以它可以用**对称指针在 NVLink
   域内直接读**（`gin.get_sym_ptr<ncclTeamTagLsa>`），比再走一趟 RDMA 便宜。
2. **发射与等待分离**。`fetch()` 一次把**所有层**的 RDMA 请求都发出去（`engram_fetch_impl`），
   返回每层一个 `hook`；调用方在**真正要用某一层**时才调 `hook()`（`engram_fetch_wait_impl`）等它到。
   这样多层的拉取在网络上**并发**，而等待被推迟到使用点。
3. **请求要合并成批、合并成门铃（doorbell）**。RDMA 的网络瓶颈在**消息数**（MPPS）而不是字节数，
   所以 issue warp 把命中同一 peer 的 entry 攒成 32 个一批发 `ncclGinOptFlagsAggregateRequests`，
   攒够 `flush_depth` 才敲一次门铃（`gin.flush_async`）。

---

## 2. host 面：`EngramBuffer` 的生命周期与两段显存

`EngramBuffer`（`deep_ep/buffers/engram.py:31`，C++ 侧 `csrc/buffers/engram.hpp:26`）与其它 buffer 一样
继承 `BufferBase`、内部持有一个 `comm::Context`（`engram.hpp:56-60`）。它比 EP/Bucket 多两段显存：

- **GPU 主段**（`num_gpu_bytes`）：接收缓冲 + 本 rank 的 SF 分片。接收区按层切成
  `[num_layers][num_tokens, hidden*num_entries_per_token]`（`engram.hpp:168-176`）；
  SF 分片放在**尾部的 `num_sf_shard_bytes`**（`:136-137`、`:198-199`）。
- **RDMA storage 段**（`num_rdma_storage_bytes`）：每 rank 自己的 entry 表，**跨 rank 对称**。

配置三步（`test_engram.py:19-42`）：

1. `get_storage_size_hint(...)`（`engram.py:112-127`）→ C++ `calculate_storage_size`
   （`engram.hpp:251-271`）按公式算出 `(num_gpu_bytes, num_rdma_storage_bytes)`；
2. `get_theoretical_config(...)`（`engram.py:129-144`）→ C++ `get_theoretical_config`
   （`engram.hpp:273-299`）解析式算出 `(num_qps, qp_depth, restrict_rd_atomic)`；
3. 构造 buffer 后 `set_config(...)`（`engram.py:146-151` / `engram.hpp:79-107`）把层配置写进去。

**写入**（`write`，`engram.hpp:109-155`）：把每层的本地存储**首尾相接**拷进本 rank 的 storage 分片：
`storage_ptr` 起点是 `local_rdma_storage_ptr`，逐层 `advance_ptr(storage_ptr, storage.nbytes())`
（`:118-131`）；所以 storage 分片布局是 `[layer 0 的 entries][layer 1 的 entries]...`。SF 则写到
GPU 段尾部的分片（`:136-137`），并且只拷**本 NVL rank 的那一段**（`:144-148`，`nvl_rank_idx *
num_shard_bytes`）。

为什么每 rank 的 storage 要跨 rank 对称、而 GPU 段不必：storage 是**被别人 RDMA 读**的
（必须用对称窗口 `ncclCommWindowRegister` 注册，`context.cpp:208-217` 为 `gin_min_stride` 个 rank
各注册一个窗口），GPU 接收缓冲是本 rank 自己的落地区（虽然也对称注册，但只被本地索引）。

---

## 3. 布局层：`EngramLayout` 与 `EngramWorkspace`

### 3.1 `EngramLayout`（`layout/engram/workspace.cuh:28-74`）

它把**全局 entry 下标**翻译成「哪个 shard 的哪一行 / 哪个字节偏移」，全是 constexpr：

- `get_num_entries(l)` / `get_num_prefix_entries(l)`（`:36-47`）：第 `l` 层的 entry 数与层前缀和；
- `get_storage_byte_offset(l, i)`（`:50-52`）：`(前缀 + i) * kNumHiddenBytes`——**storage 分片内**
  第 `l` 层第 `i` 个 entry 的字节偏移（印证第 2 节的「层首尾相接」布局）；
- `get_recv_byte_offset(l, entry_idx, num_max_tokens)`（`:55-57`）：
  `(l * kNumEntriesPerToken * num_max_tokens + entry_idx) * kNumHiddenBytes`——**GPU 接收缓冲内**的目的偏移，
  注意它用 `num_max_tokens`（容量）而不是实际 token 数，所以接收区是**按最大容量预留**的；
- SF 三件套（`:60-73`）：`get_num_sf_rows_per_shard(l) = kNumSFOwnersPerShard * entries(l)`、
  `get_sf_row_idx(l, g) = owners*prefix(l) + g % rows_per_shard`、
  `get_sf_shard_idx(l, g) = g / rows_per_shard`，其中 `kNumSFOwnersPerShard = kNumRanks / kNumSFShards`
  （`:31`）。SF 表按 NVL 分片数 `kNumSFShards` 切成若干 shard，每个 shard 覆盖 `owners` 个 rank 的行；
  行号里的 `g % rows_per_shard` 正是「在 shard 内的行」。

### 3.2 `EngramWorkspace`（`layout/engram/workspace.cuh:9-21`）

```
struct alignas(64) EngramWorkspace {
    BarrierSignals barrier;                                        // @0,  64B
    uint32_t issue_counters[kNumMaxLayers=128][kNumMaxPeerContexts=2048];  // @64
};
```

`get_issue_counter_ptr(layer, peer_context)`（`:16-19`）给出每个 `(层, peer-context)` 的**到达计数器**，
用在第 4 节的「多 warp 汇合后再敲门铃」协议里。H100 实测 `sizeof = 64 + 128*2048*4 = 1048640`，
偏移 `barrier@0 / issue_counters@64`、`ctr(3,5)@64+(3*2048+5)*4 = 24660`（见实测）。

---

## 4. 发射面：issue warp 的布局与请求生成

`engram_fetch_impl`（`engram_fetch.cuh:41-255`）每个 SM 起 `kNumSFWarpsPerSM + kNumIssueWarpsPerSM` 个
warp（host 里 4 个 SF warp + 28 个 issue warp，`csrc/kernels/engram/engram.hpp:36-42`）：**issue warp** 只算索引、发 RDMA；
**SF warp** 走 NVLink 拷 scale factor。每个 issue warp 有一块
`__shared__ int64_t pending_requests[kNumIssueWarpsPerSM][2*kWarpSize]`（`:78`）作为**环形待发队列**。

### 4.1 `EngramIssueWarpLayout`（`engram_fetch.cuh:13-39`）

把 1 维的 `issue_warp_idx = sm_idx * kNumIssueWarpsPerSM + warp_idx` 分解成：

```
peer_context_idx = idx % (kNumRDMAPeers * kNumGinContexts)
rdma_peer_idx    = peer_context_idx % kNumRDMAPeers       // 管哪个 RDMA peer 组
gin_context_idx  = peer_context_idx / kNumRDMAPeers       // 用哪条 QP
peer_warp_idx    = idx / kNumRDMAPeers                    // 本 peer 内的 warp 序号（token tile 起点）
num_peer_warps   = ceil_div(kNumIssueWarps - rdma_peer_idx, kNumRDMAPeers)
```

**为什么这样切**：RDMA 的额度是按「(peer, QP)」组织的，所以外层用 `peer_context`；token 维按
32 个一条 tile 横向铺开，**同 peer 的 warp 交错着走 tile**（`peer_warp_idx, +num_peer_warps`，
`:133-135`），既并行又保证覆盖。`get_num_active_peer_context_warps`（`:34-38`）只在 `gin_context <
num_token_tiles` 时才活跃——token 少的时候不必把所有 QP 都开动。

### 4.2 请求生成（`engram_fetch.cuh:99-171`）

对每一层，warp 只处理**属于自己 peer** 的 entry：

```
peer_global_idx_begin = rdma_peer_idx * kNumRanksPerRDMAPeer * num_entries_per_rank   // :106
```

每个 token tile、每个 round，lane 读 `indices[layer, entry_idx]` 得 `global_idx`，
用 `belongs_to_peer = global_idx ∈ [begin, end)` 判定（`:143-144`），然后
`__ballot_sync` 出 mask、用 `__popc(mask & lower_lanes)` 把命中项**保序压进**环形队列
（`:147-155`）。`issue_batch`（`:109-131`）从队头取最多 32 项，把每个 `(global_idx, entry_idx)`
翻译成 RDMA 参数：

```
peer_local_idx      = global_idx - peer_global_idx_begin
intra_peer_rank_idx = peer_local_idx / num_entries_per_rank     // 组内第几个 rank（=storage window）
local_entry_idx     = peer_local_idx % num_entries_per_rank
src_byte_offset     = get_storage_byte_offset(layer, local_entry_idx)
dst_byte_offset     = get_recv_byte_offset(layer, entry_idx, kNumMaxTokens)
```

即：源 = 组内第 `intra_peer_rank_idx` 个 rank 的 storage window 上的
`src_byte_offset`；目的 = 本地 `fetched + dst_byte_offset`；长度 = `kNumHiddenBytes`；
源 rank = `rdma_peer_idx * kNumRanksPerRDMAPeer`（`:124-129`）。**为什么源用 window 数组**：
`rdma_storage_windows_gpu` 是 `gin_min_stride` 个窗口的数组（`context.cpp:211-217`），
组内每个 rank 一个，`intra_peer_rank_idx` 选具体是哪一个。

队列的**环形设计**很巧：缓冲是 `2*kWarpSize=64`，`pending_head` 只在 `0` 和 `32` 之间翻
（`:162` 的 `pending_head ^= kWarpSize`）。当一轮追加越过 32、或者攒满 33 项触发一次 `issue_batch(32)` 时，
**多出来的那项恰好绕回到翻转后的新队头**——所以队列永不需要 memmove。

### 4.3 门铃 + 多 warp 汇合（`engram_fetch.cuh:173-205`）

每层发完后，warp 用 `elect_one_sync` 选出 leader 敲**一次门铃** `gin.flush_async`（把本 QP 的
聚合请求一并投递，并把 `last_gin_requests[layer, peer_context]` 填好供 `wait` 用）。如果同一个
`peer_context` 有**多个活跃 issue warp**（`num_active_peer_context_warps > 1`），要先用
`issue_counters[layer][peer_context]` 做**到达计数**：每个 warp `fetch_add(1, release)`，**最后一个**
`fence_acquire_gpu()` 后敲门铃并把计数清零（`:185-190`）；其余 warp 在两件事里二选一——若是**非最后一层**，
自旋等计数归零（`:192-202`，带 `timeout_while` 看门狗），否则直接走。这样既保证「所有 warp 的请求都
攒进同一 QP 之后才投递」，又让下一层能复用同一个计数器而**无需复位**。

### 4.4 SF warp（`engram_fetch.cuh:208-254`）

SF warp（`warp_idx >= kNumIssueWarpsPerSM`）逐 entry 拷 scale factor：用
`get_sf_row_idx`/`get_sf_shard_idx` 定位源（**本地 NVL 分片桩**，真实代码是
`gin.get_sym_ptr<ncclTeamTagLsa>(sf_shards, shard)`，`:224`），按 `uint64` 向量读，再按
`sf_token_stride`/`sf_hidden_stride` 写到目的（`:232-240`）。这两个 stride 由 host 按
`use_tma_aligned_col_major_sf` 决定（`engram.hpp:183-191`）：**TMA 列主序对齐**时可被下游 GEMM 的
TMA 直接吃（token stride=1、hidden stride=align(tokens,4)），否则走行主序（token stride=ept*packs、
hidden stride=1）。

---

## 5. 等待：`engram_fetch_wait_impl`

`engram_fetch_wait_impl`（`engram_fetch_wait.cuh:9-28`）单调而清晰：一个 1024 线程的 block，把
`num_active_peer_contexts = num_active_gin_contexts * kNumRDMAPeers` 个 `last_gin_requests`
分给线程，每线程对该 request 调 `gin.wait(...)`（`:19-27`）。`num_active_gin_contexts = min(num_qps,
ceil_div(num_tokens, 32))`（`engram.hpp:205-206`）——等待的 context 数与发射侧「有多少 token tile」
一致。

---

## 6. host 公式：两段显存与 QP 配置

### 6.1 `calculate_storage_size`（`csrc/buffers/engram.hpp:251-271`）

```
num_bytes_per_entry = hidden * elem_size              // 必须 64B 对齐（:260）
num_recv_bytes      = align(num_layers * num_max_tokens * ept * num_bytes_per_entry, 2MiB)  // :261-263
num_rdma_storage    = align(Σ_l entries(l) * num_bytes_per_entry, 2MiB)                      // :264-265
num_sf_bytes        = align(Σ_l entries(l) * num_rdma_ranks * num_sf_packs * 4, 2MiB)        // :267-269
返回 (num_recv_bytes + num_sf_bytes, num_rdma_storage)
```

注意 SF 段乘的是 **`num_rdma_ranks`**（每个 RDMA rank 一份副本），而 storage 只存本 rank 的。

### 6.2 `get_theoretical_config`（`csrc/buffers/engram.hpp:273-299`）

解析式估 QP 数与深度，不做 auto-tune：

```
kReadRTTSeconds   = 10e-6                                             // :279
num_qps           = ceil(nic_read_mpps * 1e6 * RTT / max_read_req_per_qp / num_rdma_peers)   // :286-287
num_requests      = num_layers * num_max_tokens * ept
num_required_slots = ceil_div(num_requests, num_qps * num_rdma_peers)
if num_required_slots > 32768: 回退 (num_min_qps, 32768, restrict_rd_atomic=True)            // :291-294
else: qp_depth 从 1024 翻倍到覆盖 num_required_slots（上限 32768）                            // :296-298
```

`nic_read_mpps` 由 Python 侧给（`check_fast_rdma_atomic_support()` 为真取 90、否则 45，
`engram.py:140`）——即「网卡每秒能发多少百万次读」。**为什么**：Engram 是**读受限**，
吞吐 = `MPPS / 平均每次读请求的 RTT 内能挂多少在途`，所以要按「每秒读次数 × RTT」估 QP 数。

---

## 7. H100（sm90）实测

`harness/exp_deepep_engram.sh` 把 `EngramIssueWarpLayout`（`engram_fetch.cuh:13-39`）与 issue 循环
（`:99-171`）、SF 主体（`:208-254`）**逐行原样抽取**，`gin.get` 换成只记录 `(layer, peer, entry_idx,
window, src, dst)` 的 recorder，直接 include 真实 `layout/engram/workspace.cuh`，跑真实设备代码；
host 侧布局 / 发射映射 / 请求索引 / SF 目的布局 / 两个 host 公式由 Python **独立复刻**对拍。
（片段 sha256：issue warp 布局 `32673de73ceeb9ce`、issue 循环 `06b24f4417c29feb`、SF 主体
`6f80791f169522ad`，与 `~/github/DeepEP@93eb6eb` 一致。）

配置：4 ranks、2 SF shards、hidden 64B、每 token 2 entry、层 entry `{4,6}`、`kNumRDMAPeers=2`、
`gin_min_stride=2`、`kNumQPs=2`、每 SM 4 个 issue warp。

- **布局层**（Part A/B）：`storage/recv_byte_offset`、`get_sf_row_idx/get_sf_shard_idx` 逐项 **0 不符**；
  `EngramWorkspace` `sizeof=1048640`、`barrier@0 / issue_counters@64`、
  `get_issue_counter_ptr(3,5)@24660` 与公式一致。
- **发射映射**（Part C）：4 个 issue warp 的 7 个导出字段逐字段 **0 不符**；**每个 RDMA peer 的 2 个
  token tile 恰好覆盖一次**（无重无漏）；`gin_context >= num_token_tiles` 的 warp `active=0`。
- **请求生成**（Part D）：256 条索引（含 43 条 `-1` 的 OOB）产生 **213 条请求**，**miss=extra=值不符=0**
  ——每条有效 entry **恰好一条请求**，且 `window/src_off/dst_off` 逐字段一致。
- **SF 拷贝**（Part E）：TMA 列主序与行主序两种目的布局各 1024 个元素 **mismatch=0**（含 `-1` 留零）；
  token/hidden stride 与 `engram.hpp:183-191` 完全一致。
- **host 公式**（Python）：三组 `calculate_storage_size` 用例（BF16/FP8、不同规模）两段大小均 2MiB 对齐、
  `storage=align(Σentries*hidden*elem,2MiB)`；`get_theoretical_config` 五组输入 QP 数与深度合理，
  **请求量溢出**（`64*512*32 > 32768*29`）时正确返回 `(qps=33, 32768, restrict_rd_atomic=True)`。

**设备 PASS=5 / Python PASS=32 / FAIL=0**（`harness/exp_deepep_engram.out.txt`）。

### 仅分析的部分

真正的数据面——`gin.get`（RDMA 拉取）、`gin.flush_async`（门铃与 `last_gin_requests`）、
`engram_fetch_wait_impl` 的 `gin.wait`、SF 的 `gin.get_sym_ptr<ncclTeamTagLsa>`——需要 `nccl_device.h`
的 GIN 设备 API（本机 NCCL 2.29.7 不提供；对齐 `docs/08` 的「阻塞」）。README 也点名
**Engram requires a NCCL build providing `ncclGinOptFlagsWarpGet`**。因此真实的端到端 fetch（含
多 rank RDMA、MPPS 与带宽）需连接 NCCL ≥ 2.32.3 的环境才能量。

---

## 8. 与前三类通信的对照

| 维度 | EP all-to-all（第 7/58/61） | bucket（第 64） | PP send/recv（第 68） | **Engram（本篇）** |
| --- | --- | --- | --- | --- |
| 语义 | 每个 rank 散给所有 rank | 所有 rank 同步一份 | 环上邻居传张量 | **按全局索引随机远程读** |
| 数据面 | NVLink/RDMA dispatch/combine | copy engine / NVLink / RDMA | TMA + RDMA `put` | **RDMA `get`（带索引 gather）** |
| 计数/前缀和 | 有（分布式前缀和） | 无 | 无（在途配额背压） | 无（按 index 直接寻址） |
| 特殊点 | 通信-计算重叠 | 零 SM（copy engine） | 两级 TMA 流水 | **发射/等待分离 + 请求聚合** |

至此 DeepEP 的**四类通信设施**（EP all-to-all、bucket 集合、PP 点对点、Engram 远程抓取）都已在
H100 上把架构无关的部分拆成可复现实验；余下的运行期空档（EP dispatch/combine 端到端、bucket 的
all_reduce/reduce_scatter 内核、Engram 的真实 RDMA）统一卡在 NCCL GIN 设备 API 上，见「阻塞」。
