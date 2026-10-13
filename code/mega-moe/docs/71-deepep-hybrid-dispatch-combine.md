# DeepEP 的两级 EP 通信：hybrid dispatch / combine（scale-out RDMA × scale-up NVLink）

> 专题第 71 篇。第 7 篇（`docs/07-deepep-dispatch-combine.md`）讲了 DeepEP 的
> `impls/ep/dispatch.cuh` / `combine.cuh` 主 kernel 数据流，第 48/61 篇拆了它们的两个 epilogue。
> 但那两篇讲的其实是**单级**（`num_scaleout_ranks == 1`）路径。真实的跨节点 EP —— 也就是
> **Mega MoE 在 EP8/EP16 上真正跑的那条路** —— 走的是 `impls/ep/hybrid_dispatch.cuh`（674 行）与
> `impls/ep/hybrid_combine.cuh`（627 行）：**同一 kernel 内同时编排两级域**
> —— **scale-out（跨节点 RDMA，`ncclTeamTagRail`）** 与 **scale-up（同节点 NVLink，`ncclTeamTagLsa`）**。
> 本篇把这两条一直缺席的 hybrid kernel 补上。
>
> 约束同前：它们的数据面（TMA 搬运、`gin.put`/`get_sym_ptr`、跨 rank barrier）走 **NCCL 设备 API**，
> 本机 PyTorch 链的 NCCL 2.29.7 不提供（见 ROADMAP「阻塞」）→ **数据面仅源码分析**；
> 但它们的**计数汇聚、就绪哨兵、前缀和、tail 信号、round-robin 分片调度**只踩公共设备层
> （`common/{math,ptx}.cuh` + `layout/ep/token.cuh`），是**架构无关**的，本机 H100（sm90）
> **直接 include 真实头文件、跑真实设备代码**——`src/deepep_hybrid.cu` +
> `harness/exp_deepep_hybrid.{sh,py,out.txt}`，设备 **7 PASS** / Python oracle **6 PASS** / 0 FAIL。

## 1. 为什么要「两级」：拓扑与 host 分派

DeepEP 的 rank 空间被拆成两个正交域（`comm/context.cpp` 的 `num_scaleout_ranks × num_scaleup_ranks`，
第 53 篇拆过）：**scale-out** 是跨节点的 RDMA 子团队（同一 `scaleout_rank_idx` 的各节点），
**scale-up** 是节点内 NVLink 子团队（同一 `scaleup_rank_idx` 的各 GPU）。全局 rank 用
**混合进制**编码（`hybrid_dispatch.cuh:60`）：

```
rank_idx = scaleout_rank_idx * kNumScaleupRanks + scaleup_rank_idx
```

host 侧按 `num_scaleout_ranks` 决定编译哪个 kernel（`csrc/kernels/ep/dispatch.hpp:89-114`、
`combine.hpp:64-90`）：`== 1` 走单级 `dispatch_impl`/`combine_impl`，否则走 hybrid，并把 warp 角色
编进模板。hybrid 的 warp 划分是**三个角色**（`dispatch.hpp:80-85`）：

- **notify warps**（`kNumNotifyWarps = 4`，`dispatch.hpp:20`）：只做计数与全局前缀和（唯一只在
  hybrid dispatch 出现、combine 没有的角色）；
- **scaleout warps**：每 warp 一个 channel，负责跨节点 RDMA 收发（`kNumScaleoutWarps = num_channels_per_sm`）；
- **forward warps**：把 scale-out 收到（或本地）的 token **转发**给 scale-up 目的 rank（`kNumForwardWarps = num_channels_per_sm`）。

combine 只有两个角色（`combine.hpp:48-58`）：**scaleup warps**（把本 rank 专家算完的结果按链式列表
送回 scale-up 目的 rank）与 **forward warps**（把 scale-up 汇聚后的结果转发到 scale-out 目的 rank）。
发射都是 `grid = num_sms`、`cluster_dim = 2 - (num_sms % 2)`、`cooperative = true`（`dispatch.hpp:124-131`、
`combine.hpp:100-107`）——**因为 kernel 里要跨 SM 做 `gpu_barrier`/grid sync**。

一句话数据流：**dispatch**：源 token →(scale-out RDMA)→ 目标节点 →(scale-up NVLink)→ 目标 GPU 的专家；
**combine** 恰好反过来。hybrid 的价值就是把「本节点内」这一段从 RDMA 降级成 NVLink 直写（延迟 ~1/3、
带宽 ~8×，第 8/11 篇 H100 实测），只在跨节点时才用 RDMA。

## 2. hybrid dispatch：三个角色

### 2.1 三层缓冲与几何（`:87-97`）

```
token_layout      = TokenLayout(kNumHiddenBytes, kNumSFPacks*sizeof(sf_pack_t), kNumTopk, true)  // :87，带 metadata
scaleup_buffer    = BufferLayout<false>(token_layout, kNumScaleupRanks,
                                        kNumScaleoutRanks*kNumMaxTokensPerRank, buffer)            // :92-93
scaleout_send_buffer = BufferLayout<false>(token_layout, 1, kNumMaxTokensPerRank, ...)            // :94-95
scaleout_recv_buffer = BufferLayout<false>(token_layout, kNumScaleoutRanks,
                                           kNumChannels*kNumMaxTokensPerChannel, ...)             // :96-97
```

三层缓冲**链式拼在同一个 `buffer`**（`scaleup_buffer.get_buffer_end_ptr()` 再起下一层），布局细节
见第 7 篇 §3。`TokenLayout` 的「隐藏 / SF / metadata + mbarrier」四段字节几何 + 每槽 32B(TMA)/64B(RDMA)
对齐由真实设备代码验证（本篇 Part 5，H100 实测：`hidden=7168, sf=56, topk=8, with_metadata` →
`bytes<false>=7360`、`bytes<true>=7392`）。

每个 TMA warp 一个共享内存 token 槽（`tma_buffer`，`:88-89`），与外层 role 的 `smem` 划分
（notify 段 `kNumSmemBytesForNotify`，`:68-70`）分离。

### 2.2 notify warps：算清「谁收多少」（`:107-328`）

这是整个 dispatch 最精巧、且**完全架构无关**的一段，也是 hybrid 独有的。它要回答的问题：
**每个 scale-up rank、每个专家各将收到多少 token？**（用于后面的专家区分组对齐分配与展开）。

1. **本地计数**（`:114-135`）：清共享内存计数后，每个 warp 扫自己那份 token，对每个 top-k 命中
   `atomicAdd_block(expert_count + dst_expert, 1)`；**rank 计数要先去重**——同一 token 的多个 top-k
   落在同一 rank 只算一次（`ptx::deduplicate(dst_rank_idx, lane_idx)`，`:133`）。这点与 combine 的
   `combine_utils.cuh` 去重是同一思想（第 48 篇 §4）。
2. **全局归约**（`:140-143`）：把「本 SM 的本地计数」与「到达的 SM 数」**打包进一个 64 位量**
   `red_add` 到 workspace：`counter = (1ll<<32) | rank_expert_count[i]` —— **高 32 位是 SM 到达数、
   低 32 位是 token 计数**。用 `red_add`（不用 `atomicAdd` 返回值）因为只关心和。
3. **SM0 汇聚 + 就绪哨兵**（`:146-171`）：SM0 轮询 `ld_volatile<int64>`，等 `status>>32 == kNumSMs`
   （所有 SM 都到齐），然后把低 32 位**编码后**写进 send buffer：`get_scaleout_rank_expert_count_ptr<true>()[i]
   = math::encode_decode_positive<int>(status & 0xffffffff)`（`:155-156`），并清零以便下轮复用。
   编码的意义见下（§2.3）。
4. **scale-out 通知 peers**（`:177-188`）：SM0 的前 `kNumScaleoutRanks` 个线程用 `gin.put<ncclTeamTagRail>`
   把 rank 计数与 expert 计数发给每个 scale-out peer。
5. **收齐并二次归约**（`:193-219` 的 `recv_and_reduce`）：对每个 scale-out peer 读回计数（解码 +
   就绪判定），累加后清零。**这一步把「所有 scale-out 节点」的计数合成「全局计数」**。
6. **写回 scale-up peers**（`:221-252`）：rank 级计数用 `gin.put_value<ncclTeamTagLsa>` 直写远端
   （`counter = (kNumScaleupRanks<<32)|count`，高 32 位编码「预期到达的 scale-up peer 数」）；
   expert 级计数用 `gin.red_add_rel<ncclTeamTagLsa>` **累加**到远端（`:249-251`，因为多个 scale-out
   节点都会往同一个规模小的 expert 区加）。
7. **等本地计数器就绪 + 对齐**（`:263-300`）：等每个 scale-up 槽 `status>>32 == kNumScaleupRanks`，
   把 rank 计数（粒度 1）与 expert 计数（粒度 `kExpertAlignment`）**向上对齐**（`:269-270`），
   写进 host workspace（`kDoCPUSync` 时）、写统计与 unaligned 记录。
8. **前缀和**（`:302-327`）：`warp0` 对 rank 计数做**含端点**前缀和写
   `psum_num_recv_tokens_per_scaleup_rank`；`warp1` 对 expert 计数做**排他**前缀和写
   `psum_num_recv_tokens_per_expert`（排他 = `out[0]=0, out[i+1]=Σ_{<i}`，为展开铺位）。算法是
   `do_psum`（`:304-320`），用 `ptx::warp_inclusive_sum` 逐 32 段扫、`ptx::exchange(sum, 31)` 把上一段
   末值作为 `psum` 进位传下去。

### 2.3 「就绪哨兵」`encode_decode_positive`（`common/math.cuh:26-33`）

上面第 3/5 步全靠它：远端 peer 读计数时，怎么区分「还没写」与「写了计数 0」？对称缓冲初值是 0。
DeepEP 的做法是把**任何非负计数 c 编码成负值 `-c-1`** 再写：

```
is_decoded_positive_ready(v) = (v >= 0)      // math.cuh:26-28
encode_decode_positive(v)    = -v - 1        // math.cuh:31-33（对合：做两次还原）
```

于是：buffer 初值 `0` → 解码 `-1`（<0，**未就绪**）；写过的计数 `c`（编码 `-c-1 < 0`）→ 解码 `c`
（≥0，**就绪**）——**含 c=0 也能区分**（编码 `-1`）。读者先 `encode_decode_positive(raw)` 解码、
再 `is_decoded_positive_ready(decoded)` 判定（`hybrid_dispatch.cuh:200-201`）。

**H100 实测**（`src/deepep_hybrid.cu` Part 1，直接调真实设备函数）：8 个计数
（含 0、`2^20`、`INT32_MAX`）编码/解码对合、解码后就绪、初值 0 解码后未就绪，全部一致；
host 侧 4096 个计数逐点对合。

### 2.4 scaleout warps：把 token 发出去（`:329-460`）

每 warp 一个 channel。循环里（`:388-457`）：预取下一个 token（TMA 拉 hidden + `cp.async` 拉 SF，
`preload_next_token`，`:354-385`）→ 读 top-k 得到目的 scale-out rank → **按 rank 去重分槽**
（`ptx::deduplicate` + `ptx::exchange` 取槽号，`:408-417`）→ 等 TMA 到达后把 token 写进
`scaleout_send_buffer`（只给要走 RDMA 的 rank 写，`:424-428`）→ **本地目的 rank 直接写 recv buffer
绕过 RDMA**（`:432-436`）→ 对远端 rank 发 `gin.put<ncclTeamTagRail>`（`:445-452`）。
每 `kScaleoutUpdateInterval=6` 个 token 用 `update_scaleout_tail`（`:338-351`）把「已发 tail」
**按增量** `red_add_rel` 到对端（低 32 位=tail、高 32 位=finish 标志；见 §2.6）。

### 2.5 forward warps：转发 + 写元数据（`:461-656`）

forward warp 轮询每个 scale-out peer 的 tail（`:497-523`），用 **round-robin 位图**选「上一个处理过的
rank 之后、第一个有数据的 rank」：

```
offset = (recv_scaleout_rank_idx + 1) % kNumScaleoutRanks     // :492
hi_mask = (wip_mask >> offset) << offset                       // :493
recv_scaleout_rank_idx = hi_mask ? ffs(hi_mask) : ffs(wip_mask) // :494
```

每次只处理一个 chunk（`kNumSlotsPerForwardChunk = kScaleoutUpdateInterval = 6` 个 slot，
`:527-530`），避免在一个 rank 上饿死其它 rank。对每个 slot：TMA 拉进 smem → 读 top-k 得到
**scale-up 目的 rank** → 算链式列表下标（`transform_linked_list_idx`，`:475-479`）→ scale-up 去重分槽
（`atomicAdd(get_scaleup_atomic_sender_counter()+rank, 1)`，`:584-585`）→ **`gin.get_sym_ptr<ncclTeamTagLsa>`
取远端对称地址后 `tma_store_1d` 直写**（`:590-596`）→ 记录 `token_metadata_at_forward`（源 token 全局
索引 + 是否 chunk 末 + top-k 目的 rank/slot，`:609-627`）+ `dst_buffer_slot_idx`。扫完后写链式列表
尾巴（`:638-649`）并清 tail（`:653-654`）。最后退到 `gpu_barrier`（scale-up 子团队，`:660-662`）
+ `cudaTriggerProgrammaticLaunchCompletion()`（PDL，触发下游 copy epilogue，`:665`）。

### 2.6 tail 信号与 round-robin 调度的 H100 验证

**(a) tail 信号 `pack2/unpack2`**（`common/math.cuh:45-68`）：把 `(finish, tail)` 打成一个 `int64`
（**`finish` 在低 32 位、`tail` 在高 32 位**——`int64` 重解释成 `int[2]`，`ptr[0]=x, ptr[1]=y`），
写端只发「新旧差」的增量，读端累加即得最新 tail（`hybrid_dispatch.cuh:341-347`、`519`；
combine 用 `expected = pack2<int64>(1, 0)` 一次 `red_add_rel` 后比等，`hybrid_combine.cuh:590-596`）。
H100 实测（Part 2）：位段逐位一致；「增量 {2→5→9} 累加 = 9 且 finish=1」成立。

**(b) round-robin 分片调度**：本篇把 `:490-532` 的位图选择逻辑逐字搬到 H100 跑（Part 4），
覆盖 4 组（`R∈{3,4,5,8}`、`chunk∈{1,6,100}`、含 0-token rank）：**每个 rank 的 `[0, tail)` 恰好被
处理一次**，且选择序列与独立 host 贪心 oracle **逐步相等**（`seq_match=Y`）。

## 3. hybrid combine：两个角色

### 3.1 三层缓冲与模式（`:49-79`）

combine 的 token 布局**不带 SF、不带 metadata**（`:63`：`TokenLayout(kNumHiddenBytes, 0, kNumTopk, false)`）。
三层缓冲：`scaleup_buffer`（scale-up 目的 rank 收到的贡献）、`scaleout_recv_buffer`（本 rank 最终该收的）、
`scaleout_send_buffer`（要走 RDMA 的），`:71-79`。两个模式开关：`kUseExpandedLayout` 与
`kAllowMultipleReduction`（`:12`），后者决定「同一 token 的多个 top-k 落在同一 rank 时是就地归约还是
逐个转发」，直接改缓冲外层维（`use_rank_layout`/`get_num_tokens_in_layout`，第 48 篇 §3 已拆）。

### 3.2 scaleup warps：按链式列表把结果送回（`:107-351`）

沿 `channel_linked_list` 迭代每个 scale-up peer 的 token（`:158-348`），对每个 token：
读源 metadata（源全局 token 索引 → 拆出源 scale-out rank 与 token，`:202-205`）→ 定位远端
`scaleup_buffer` 对称槽（`gin.get_sym_ptr<ncclTeamTagLsa>`，`:217`）→ **三种情形**（`:230-303`）：
① 无需本地归约（非 expand 或 expand+单贡献）→ TMA 直接 load 后 store 到远端；②
`kAllowMultipleReduction` → 先把本 rank 的多个贡献用 `combine_reduce` 就地归约进 smem
（`combine_utils.cuh`，第 48 篇）；③ expand 但不允许归约 → 逐个 top-k 各发一份。再写 top-k 权重
（`:306-316`）、`tma_store_1d` 到远端 scale-up 槽（`:319-333`）、按 `kNumScaleupUpdateInterval=3`
发 `st.release.sys` 的 tail 增量（`update_tails`，`:122-149`——注释 `:123` 明确 `st.release.sys` 很慢，
所以间隔发）。

### 3.3 forward warps：replay + 转发到 scale-out（`:352-622`）

forward warp **重放** dispatch 阶段记录的 `token_metadata_at_forward`（`:390-403`）——这正是
「通信-计算重叠」里 forward 与 combine 共享的元数据。对每个 token 等所有需要的 scale-up tail
到位（`:405-447`，用 `stored_num_tokens_recv[j] < stored_cached_scaleup_tail[j]` 的水位判定），
然后：

- **不允许多次归约**（`:454-495`）：把每个 scale-up 贡献逐个 TMA load→store 到 `scaleout_send_buffer`
  （本地目的 rank 直接写 recv buffer 绕过 RDMA），并逐个 `gin.put<ncclTeamTagRail>` 发远端；
- **允许多次归约**（`:496-572`）：先把来自各 scale-up peer 的贡献 `combine_reduce` 成一个，
  再转发——用 `flush_last_tma_and_issue_rdma`（`:372-388`）把上一条的 TMA store 与 RDMA put
  **与下一条的重叠**，只在 chunk 末（`is_token_last_in_chunk`）才去掉 `AggregateRequests` 标志。

收尾清 scale-up tail（`:579-586`），然后 `gin.flush<ncclCoopWarp>()` + 对每个 scale-out peer
`red_add_rel` 一个 `pack2<int64>(1, 0)` 到位线（`:590-597`），自旋等它变成 `expected_signal` 再清零
（`:600-620`）——即 **scale-out 侧的「我发完了」到达计数**。

## 4. 与单级 kernel 的对照

| 维度 | 单级（`dispatch.cuh`/`combine.cuh`） | 两级 hybrid |
|---|---|---|
| 触发条件 | `num_scaleout_ranks == 1`（`dispatch.hpp:89`） | 否则 |
| 域 | 只有 scale-up NVLink | scale-out RDMA × scale-up NVLink |
| warp 角色 | notify/dispatch（dispatch）、scaleup/forward（combine） | **+scaleout +forward**（dispatch 3 角色） |
| 缓冲层数 | 1 层 | 3 层（scaleup / scaleout_recv / scaleout_send） |
| 元数据 | 不需 `token_metadata_at_forward` | forward warp 记录 + combine replay |
| 计数 | `psum_num_recv_tokens_per_*` | 同上，但先跨 scale-out 聚合再跨 scale-up 传播（§2.2） |
| 同步原语 | `red_add`/`put`/`get_sym_ptr` | 同 + `put_value<ncclTeamTagLsa>`/`red_add_rel`/`st.release.sys` |

一句话：hybrid 没有新的数学，而是把「跨 rank 计数聚合」与「两跳转发」显式编排进同一 kernel，
让**节点内那跳走 NVLink（`get_sym_ptr` 直写 + TMA）**，只有节点间才用 RDMA（`gin.put`）。

## 5. H100 实测与边界

**可跑、已跑**（`harness/exp_deepep_hybrid.out.txt`，H100 sm90，真实 include DeepEP `93eb6eb` 头文件）：

```
Part 1 计数就绪哨兵 encode_decode_positive        8 计数 + host 4096        PASS
Part 2 tail 信号 pack2/unpack2 + 增量累加                                   PASS
Part 3 do_psum 含端点/排他前缀和（逐字搬运 :304-320）  8 组                 PASS
Part 4 forward round-robin 分片调度（逐字搬运 :490-532）4 组               PASS
Part 5 TokenLayout 每槽字节 vs host oracle                                  PASS
设备 7 PASS / Python oracle 6 PASS / 0 FAIL
```

**仅分析（数据面，NCCL 设备 API）**：`gin.put`/`gin.put_value`/`gin.red_add_rel`/`gin.get_sym_ptr`/
`gin.flush`、`gpu_barrier`、TMA 跨 rank 搬运、PDL 衔接。这些需要 **NCCL ≥ 2.32.3 的 GIN 设备 API**
（本机 2.29.7 不提供，见 ROADMAP「阻塞」）；待环境就绪可在 8×H100 上做端到端 dispatch/combine
与 compute overlap 实测。

## 6. 小结

- hybrid dispatch/combine 是 DeepEP 在**两级拓扑**（跨节点 RDMA × 节点内 NVLink）上的 EP all-to-all
  实现，也是 Mega MoE 在 EP8/EP16 上真正跑的那条路；单级 kernel 只是它在 `num_scaleout_ranks==1`
  的退化。
- **dispatch 三角色**：notify（计数 + 全局前缀和）、scaleout（RDMA 收发）、forward（NVLink 转发 +
  元数据）；**combine 两角色**：scaleup（按链式列表送回）、forward（重放元数据 + 转发到 scaleout）。
- 四个「架构无关但很讲究」的机制：**就绪哨兵** `encode_decode_positive`（区分「未写」与「写了 0」）、
  **tail 增量信号** `pack2/unpack2`、**含端点/排他前缀和** `do_psum`、**round-robin 位图分片调度**。
- 它们在 H100 上以**真实设备代码**逐字验证通过；数据面因 NCCL GIN 设备 API 本机不可用而仅分析。
