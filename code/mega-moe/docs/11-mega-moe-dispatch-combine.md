# `sm100_fp8_fp4_mega_moe.cuh` 逐段精读（四）：dispatch / combine 与 NVLink、locality domain

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 8 篇，对应路线图 **P3-4**。
> P3 拆四段：模板/block config（`docs/03`）→ warp specialization & mbarrier（`docs/09`）→
> tcgen05 MMA 与 L1/L2 epilogue（`docs/10`）→ **dispatch & combine 段（本文）**。
>
> **重要**：本文讨论的 kernel 是 **SM100（Blackwell）专属**（tcgen05、2-CTA cluster、symmetric
> memory、device locality domain），本机 H100（sm90）**不跑端到端**，因此以**源码精读为主线**；
> 唯一能上 H100 实测的是「远端显存 vs 本地显存」的**访存成本**（§6 给出真实数字），它解释了
> dispatch/combine 为何要分 chunk、重叠，以及 SM100 为何要做 locality-domain 亲和。
>
> 源码坐标（`~/github/DeepGEMM`，2026-09-30 公开版 commit `057ca59`）：
> - 主体 kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）
> - 调度器：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
> - 对称显存与 buffer 布局：`deep_gemm/include/deep_gemm/layout/sym_buffer.cuh`、`.../layout/mega_moe.cuh`
> - 跨 SM / 跨 rank 同步：`deep_gemm/include/deep_gemm/comm/barrier.cuh`
> - locality domain：`csrc/runtime/locality_domain.hpp`、`csrc/apis/locality_domain.hpp`、
>   `deep_gemm/include/deep_gemm/impls/sm100_locality_domain.cuh`
> - host 侧入口：`csrc/apis/mega_moe.hpp`
> - 端到端测试：`tests/test_mega_moe.py`（baseline 用 DeepEP 做 dispatch/combine）
>
> 以下行号均逐条核对过。

## 1. 两段通信在整体 kernel 里的位置

Mega MoE 一个 kernel 里有三类 warp（`docs/03`）：`kNumDispatchThreads` 个 dispatch warp、
4 个 non-epilogue warp（TMA-A / TMA-B / MMA / 调度）、`kNumEpilogueThreads` 个 epilogue+combine
warp。通信由两类 warp 分担：

| 段 | 谁做 | kernel 行 | 数据方向 |
|---|---|---|---|
| **dispatch** | dispatch warps（`warp_idx < kNumDispatchWarps`） | `:342-689` | 远端 rank 的 token 激活/SF → 本地 `l1_token_buffer` 环 |
| **combine 写入** | epilogue warps 的 **L2 分支** | `:1254-1357` | 本地 L2 BF16 结果 → 远端 rank 的 `combine_token_buffer` |
| **combine 归约** | epilogue warps（全部） | `:1365-1520` | 本地 `combine_token_buffer` 的 top-k 贡献 → fp32 累加 → 写 `y` |

dispatch 与 combine **不是先后两段**，而是**同一次 launch 内的生产者/消费者**：dispatch 把
token 灌进 L1 环触发计算，计算完的 L2 结果又经 NVLink 写回源 rank，最后 combine 归约。整条链
都在跑，靠 workspace 里的原子计数（非阻塞轮询）串起来——这就是「通信-计算重叠」的落地方式
（宏观机制见 `docs/02` §4-§5，mbarrier/流水见 `docs/09`）。

## 2. dispatch 段（`:342-689`）

dispatch 的目的：**每个 rank 只算自己负责的专家**，因此要把「topk 选中本 rank 专家」的那些
token 从它所在的源 rank 搬到本地，并按「专家 → 池块（pool block）」排布，喂给后面的 GEMM。

### 2.1 先数 token，再跨 rank 汇总（`:349-418`）

每个 dispatch warp 先扫一遍本地 `input_topk_idx_buffer`，对每个 `(token, topk)` 用
`atomicAdd_block` 累加本 rank 各专家的到达数（`read_topk_idx`，`:349-366`；`atomicAdd_block`，`:370`）。

接着做**分布式前缀和**：每个线程分到一个专家，把自己的计数**高 32 位当加数、低 32 位当有效负载**
一次原子加进全局 workspace（`:374-381`）：

```cpp
const uint64_t send_value = (1ull << 32) | shared_storage.expert_token_count[i];
shared_storage.expert_token_count[i] = static_cast<uint32_t>(
    ptx::atomic_add(workspace.get_expert_send_count_ptr(i), send_value));
```

一次 `atomicAdd` 同时完成「返回值 = 之前所有 rank 之和（低 32 位）+1（高 32 位计数）」，省掉一次
独立计数。这一步注释标注约 **6.5 us**（`:374`）。随后把「该 topk 的源坐标」写到**目标 rank** 的
workspace（`*sym_buffer.map(dst_ptr, dst_rank_idx) = token_topk_idx`，`:384-390`），供远端 pull
时反查。

`grid_sync`（`:393-396`）后，由 **SM 0** 把「本 launch 的 grid index（`get_grid_idx()+1`）」push
给每个 peer 的 `peer_grid_idx`（`:399-404`），并原子累加各专家的全局接收数到
`expert_recv_count_sum`（`:406-417`）。`peer_grid_idx` 是后面 combine readiness 的标签
（§3.2）——用 grid index 而非复位 flag，天然避免了上一轮的残留。

### 2.2 跨 rank barrier 与「拉取」主循环（`:421-619`）

`nvlink_barrier<...kBeforeDispatchPullBarrierTag>`（`:422-428`）保证所有 rank 的源索引都已写好，
之后每个 warp 从 `scheduler.fetch_expert_recv_count()`（`:446`）拿到各专家的到达数，进入 pull 循环。

**关键约束：一个 token 只能被一个远端线程拉取一次。** 由于各 rank 的到达数不同，这里用
**round-robin min-peeling** 把「本 rank 的第 `token_idx_in_expert` 个 token」映射到某个源 rank
（`:480-528`）：每轮找所有 rank 的 `min(remaining)`，让活跃 rank 轮流各出 `min` 个 token。这样
全局 token 顺序是「各 rank 轮转交错」，且**无锁**——每个 `(expert, slot)` 由唯一线程负责。
注释也点出这是非合并访问（`:474`）但只发生在每个专家第一次碰到时。

拿到源坐标后（`:531-534`），执行：

- **环空等待**：`pool_block_idx` 对应的 ring slot 必须等上一代消费者把 N 个块都算完
  （`l1_empty_count`，`:544-550`）。环容量由 `get_num_max_live_pool_blocks`（`scheduler:48-80`）
  给出，保证不会覆盖在飞数据。
- **远端 pull token**（`:552-576`）：`src_base_ptr = sym_buffer.map(远端 input_token_buffer, src_rank)`，
  每个 token 沿 hidden 切成 `kNumChunks = kHidden / kNumBytesPerPull` 块，逐块 `tma_load_1d`
  远端 → smem（`pull_buffer`）→ `tma_store_1d` 本地（`:555-575`）。这是**异步 TMA + arrive.expect_tx**
  的典型用法，注释注明远端延迟靠「最后一块与下面 SF/weight 拷贝重叠」来掩盖（`:578`、`:583`）。
- **远端 topk weight**：`*sym_buffer.map(input_topk_weights_buffer + src_token_topk_idx, src_rank)`
  一次 4B 读（`:579-581`），它决定了这个 topk 对最终输出的加权。
- **远端 SF**：按 32 元素一块从远端逐块拷进本地 `l1_sf_buffer`，并**用 `transform_sf_token_idx`
  做 UTCCP 4x32 转置写位**（`:584-599`；转置映射见 `docs/02` §3.4、`docs/10`）。
- **落元数据**：写本地 `l1_topk_weights_buffer`、写 `get_token_src_metadata_ptr`（供 combine 反查
  目的 rank/token/topk，`:607-608`），最后 `red_add_rel` 递增 `l1_full_count`（`:613-616`）通知
  下游 GEMM：「这个池块凑齐了」。

### 2.3 清理 workspace（`:621-689`）

dispatch 的最后一段把 workspace 复位供下一轮用：SM 0 清 `expert_send_count` 和四个 task 计数器
（`:626-641`），其余 SM 分摊清各专家的接收计数、L1/L2 的 full/empty 计数与 l2_full_mask
（`:642-679`），顺带用 `red_add` 把本轮的接收数累加进 `cumulative_local_expert_recv_stats`
（`:661-662`，全丢给 SM 0/1 的统计）。注释说明这段**与 combine 归约重叠**（`:621-622`），
`sync_unaligned(kDispatchThreads + kEpilogueThreads, kDispatchWithEpilogueBarrierIdx)`（`:623`）
就是这条重叠的汇合点：combine 段复用 `smem_buffer` 头部那块（`kNumReusableSmemBytes =
offsetof(SharedStorage, dispatch_barriers)`，`:219`，即 send buffer 区域），必须等 dispatch
清理完再用。最后 `nvlink_barrier<...kAfterWorkspaceCleanBarrierTag>`（`:683-689`）保证所有 rank
都清完，整个 kernel 才算收尾。

## 3. combine 段（`:1254-1520`）

combine 有两半：epilogue 的 **L2 分支**把结果写回远端，随后所有 epilogue warp 一起做 **top-k 归约**。

### 3.1 L2 epilogue：经 NVLink 直写远端 combine buffer（`:1254-1357`）

L1 分支产出的是 FP8 激活（作为 L2 的输入，走 TMA 上屏 + `l2_full_mask` 通知），而 L2 分支才是
**combine 的写入端**。每一个 store block，先读回本行 token 的源元数据（`cached_src_metadata`，
`:1281-1290`）——分布式地告诉每个 rank「这一行该回到哪个 rank 的哪个 token 的哪个 topk 槽」。然后：

```cpp
const auto dst_token = buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx)
                       .get_data_buffer(dst_token_idx);
const auto dst_ptr = math::advance_ptr<float4>(dst_token.get_base_ptr(), n_idx * 2 + (lane_idx % 16) * 16);
*sym_buffer.map(dst_ptr, dst_rank_idx) = packed;   // :1346-1351
```

即：把本地 smem 里的 BF16 片段用 `sym_buffer.map` 换成**远端地址**直接 store。注意 `get_rank_buffer`
本身也是把 base 指针按 rank 偏移（`layout/mega_moe.cuh:326-332`），两层偏移叠加后落到目标 rank
的 `combine_token_buffer[topk][token]`。这就是 DeepEP combine 里「send」那一步，被融进了 epilogue。

### 3.2 combine readiness：用 grid index 当「水位线」（`:1405-1441`）

combine 归约必须等**所有可能给本 rank 写结果的 peer** 都写完。kernel 的做法很巧：

1. 每个 rank 在 dispatch 时把自己的 grid index push 给所有 peer（`:403`），存进该 peer 本地的
   `peer_grid_idx`。
2. combine 归约前，每个 rank 在本地 grid_sync 之后把「我从各 peer 收到的 grid index」用
   `st_rel_sys` 写到 peer 的 `combine_ready_grid_idx[own_rank]`（`:1417-1419`）。
3. 归约时，对每个选中的专家所属 rank `r`，自旋等
   `combine_ready_grid_idx[r] == own_grid_idx`（`:1436-1441`）。

用**每 launch 唯一的 grid index** 当标签，就不需要任何复位（signals 初始化一次即可），也天然
区分了「上一轮的完成」与「本轮还没开始」——这是 `docs/10` 里 `l2_full_mask` 用 generation 区分
环代际的同一思路。

### 3.3 top-k 归约（`:1424-1519`）

归约按 **token × chunk** 二维展开（`:1426`），前台是每个 warp 处理一个 token 的一个 chunk：

- 读本 token 的 topk 槽索引（每 lane 一个，`:1431-1434`）；共享专家用第 `kNumTopk` 个槽
  （`:1433`），掩码 `total_mask` 统一处理 routed/shared（`:1434`）。
- `move_mask_and_load`（`:1447-1466`）：按槽位从本地 `combine_token_buffer` 用 `tma_load_1d`
  拉进 smem，**双缓冲**：一个在累加、一个在预取下一个 topk（`load_stage_idx ^ 1`，`:1475`）。
- 累加用 **fp32**（`ptx::accumulate`，`:1485`），这解释了为何 combine 是数值上的**最后一次
  精确归约**（BF16 写回，但不累加丢失）。
- 最后 cast 回 BF16（`:1492-1500`），经 smem `tma_store_1d` 写 `y`（`:1512-1518`）。

## 4. NVLink 读写原语汇总

跨 rank 全靠 **symmetric memory**。`SymBuffer`（`layout/sym_buffer.cuh:9-42`）存了本 rank 的
base 指针和每个 rank 相对本 rank 的偏移；`map(ptr, dst_rank)` 就是把本地指针改成远端指针
（`:33-40`）。kernel 里用到的远端访存原语：

| 原语 | 用途 | 位置 |
|---|---|---|
| `*sym_buffer.map(...) = v`（普通 st） | 写远端源索引 / combine 结果 | `:389, :1351` |
| `*sym_buffer.map(...)`（普通 ld） | 读远端 topk weight | `:579-581` |
| `tma_load_1d` on mapped ptr | 远端 token / SF 批量拉取 | `:567, :1459` |
| `ptx::atomic_add_sys` / `red_add_rel_sys` | 跨 rank 原子加（grid index、barrier 信号） | `:414, :75`(barrier) |
| `ptx::st_rel_sys` / `ld_acq_sys` | 跨 rank release/acquire（combine readiness） | `:1419, :1440` |
| `comm::grid_sync` | 单 rank 内全 SM barrier | `barrier.cuh:33-51` |
| `comm::nvlink_barrier` | 跨 rank barrier（内部 = grid_sync + NVLink 信号 + grid_sync） | `barrier.cuh:53-92` |

`nvlink_barrier` 的实现值得记一笔：只有 **SM 0 每 rank 一个线程**参与 NVLink 信号，用
`nvl_barrier_counter` 的 2 bit 编码出「phase + sign」，让信号是**加减**式的（偶数轮 +1、奇数轮 -1），
从而复用同一对 `nvl_barrier_signals[2]` 而不必清零（`barrier.cuh:66-87`）。

## 5. locality domain：SM 与显存的「亲和」

### 5.1 概念

SM100 上 `kNumDeviceLocalityDomains = 2`（`common/types.cuh:54`）：目标卡的显存被硬件分成两个
locality domain（可理解为两个内存分区/die），**从某个 SM 访问「本域」显存比「跨域」显存延迟低**。
DeepGEMM 据此把 N 轴任务按域切分，让每个 SM 尽量处理本域的数据，减少跨域访存。

### 5.2 调度器怎么用（`scheduler/mega_moe.cuh:344-370`）

任务计数器 `l1_task_count[kNumDeviceLocalityDomains]`（`layout/mega_moe.cuh:67-70`）**每域一个**：

```cpp
get_next_task_idx<kNumClusters>(task_count_ptr, locality_domain_idx, ...):
    task_idx = atomic_add(task_count_ptr + locality_domain_idx, 1)
    n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters
```

即域 `d` 的 SM 优先从计数器 `d` 抢任务，抢到的 N-cluster 也落在域 `d`。当本域队列抽干，且任务量
还够（`num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs/2)`，`kNumMinStealWaves=2`，
`:225`）才去偷另一个域（`sm_locality_domain_idx ^ 1`，`:369`）——**任务少时宁可空转也不跨域**，
避免跨域延迟。每个 SM 属于哪个域，由 `sm_locality_domains[get_sm_idx()]` 查表（`:440`）。

启用条件（`impls/sm100_fp8_fp4_mega_moe.cuh:85-87`）：只有当 `L1_SHAPE_N` 和 `L2_SHAPE_N` 都能被
`BLOCK_N * 2 * 2`（两个 domain × 2-CTA）整除时 `kNumLocalityDomains=2`，否则退化为 1（关掉分域），
保证每域拿到完整的 cluster。L2 调度还会等「该 M 块所需的全部 L1 任务被各域计数器领走」才发
（`:398-403`），避免 L1→L2 死锁——这套 warmup/交错的推导见 `get_num_l1_warmup_waves`
（`scheduler:17-45`）。

### 5.3 域标签是怎么测出来的（`csrc/apis/locality_domain.hpp:74-142`）

不是由驱动直给，而是**用延迟探针测出来的**：

- 每个域分配一块 buffer，构造一条**指针追逐链**（word 0 of line `l` 指向 line `(l+7)%n`，
  `get_probe_chain`，`:74-79`）。
- `sm100_locality_domain_probe_chase`（`impls/sm100_locality_domain.cuh:7-20`）让每个 SM 的
  一个线程 chase 整条链，用 `clock64` 量 **cycles/hop**；每 hop 都跨 128B 行，确保落到 HBM。
- 比较同一个 SM 对两个域 buffer 的中位延迟：若「近/远」有明显差异（`kMinFarNearRatio=1.25`）
  且同一 TPC 的两个 SM 一致，则采纳该映射；否则回退到「偶数均分」`get_even_sm_locality_domains`
  （`:103-117`）。最后 `balance_sm_locality_domains` 让两域 SM 数相等（`:126-142`），
  保证 2-CTA cluster 的配对语义正确。

## 6. H100 实测：远端访存比本地贵多少

本机 H100 是**单 die 显存**，没有目标卡的域划分，但可以用 **8 卡 NVLink 拓扑**做最近的类比：
从 GPU0 访问 GPU0 / GPU1（同 NUMA socket）/ GPU4（跨 socket）的显存。实验源码
`src/nvlink_access.cu`，driver `harness/exp_nvlink_access.sh`，原始输出
`harness/exp_nvlink_access.out.txt`。方法完全对齐 §5.3 的失真探针：**128 MiB 指针追逐链
（`.cv` 绕过 cache）量 cycles/hop**，另测流式读/写带宽。

```
[dependent-load latency] chain = 1048576 lines x 128B = 128 MiB, .cv
  local  (GPU0->GPU0)                    ~350.5 ns  (x1.00)
  peer   (GPU0->GPU1, same socket)       ~990.9 ns  (x2.83)
  peer   (GPU0->GPU4, cross socket)      ~991.9 ns  (x2.83)

[streaming read  bandwidth]  local 2824 GB/s | peer ~368 GB/s
[streaming write bandwidth]  local 2915 GB/s | peer ~369 GB/s
```

结论与对 kernel 的解释：

1. **远端裸延迟 ≈ 3× 本地**（~350ns → ~991ns）。dispatch 每个 token 要跨 rank 读 topk weight、
   分 chunk 拉 hidden、逐块读 SF（§2.2），如果**串行**做，光延迟就压不住。这就是为什么 kernel
   用异步 `tma_load_1d` + `arrive.expect_tx`，并把「最后一块 pull」「weight 读」「SF 拷贝」
   **三者互相重叠**（`:555-599`）——把 N 次可见延迟压成约 1 次。
2. **远端带宽 ≈ 本地 1/8**（~2.8 TB/s → ~0.37 TB/s）。这直接对应 `tests/test_mega_moe.py:397-401`
   的 NVLink 流量模型（dispatch pull + combine 写回），也说明 combine 直写远端必须**逐 token
   小片段**做、和计算重叠，而不是等算完再批量拷。
3. **同 socket / 跨 socket 几乎无差异**（991 vs 992ns，带宽也相同）：H100 的 NV18 是**全互联平
   面**，主机 NUMA 距离不影响 GPU-GPU 路径。反过来看，目标卡的 locality domain 是**芯片内部**
   的内存分区（不是主机 NUMA），所以它必须用 §5.3 那种**片上延迟探针**来标定——这也解释了
   DeepGEMM 为什么不用拓扑/驱动 API，而是自己 chase 指针测。

> H100 数字只是「远端 vs 本地」的**量级类比**，不能搬去当 SM100 的实际延迟；SM100 的具体域
> 延迟需在目标卡上用同一套探针实测。此实验的定位是：让「为什么要重叠/分域」有真机数字支撑。

## 7. 小结

- **dispatch**（`:342-689`）= 计数（分布式前缀和）→ 写远端源索引 → NVLink barrier → 无锁
  round-robin 认领 → 分 chunk TMA 远端拉 token/SF + 读远端 weight → 写本地元数据 → 清 workspace。
- **combine**（`:1254-1357` 写 + `:1365-1520` 归约）= L2 epilogue 经 `sym_buffer.map` 直写远端
  combine buffer；用 grid index 当 readiness 水位线；再分 chunk 拉回、fp32 累加、BF16 写 `y`。
- **NVLink 原语**统一走 symmetric memory（`SymBuffer::map` + TMA/原子/rel-acq），跨 rank barrier
  用加减式信号免复位（`comm/barrier.cuh`）。
- **locality domain** 把 N 轴任务按「SM 与显存的亲和」分域调度（`scheduler:344-370`），域标签
  用片上指针追逐延迟探针标定（`csrc/apis/locality_domain.hpp`）；H100 实测「远端 ≈ 3× 延迟、
  1/8 带宽」给出了这套亲和机制与重叠策略的**量级动机**。
