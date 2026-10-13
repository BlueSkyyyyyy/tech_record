# DeepEP bucket all_gather 的两个 SM kernel 后端：pure-RDMA 与 hybrid

> 本文对应路线图 **P102-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 112 篇。
>
> 第 64 篇（P54-1）拆了 DeepEP 与 EP all-to-all **并列的通信设施** `BucketBuffer`，但只把
> **纯 NVLink all-gather**（不用 SM kernel、交给硬件 copy engine）讲透；第 72 篇（P62-1）补上了
> `all_reduce` / `reduce_scatter` 的**三种后端**。唯独 `all_gather` 的另外两条后端
> ——`rdma_all_gather` 与 `hybrid_all_gather`——在两篇里都只被 4 行带过。本篇把它们补全，
> 让「三类算子 × 三物理域」的矩阵真正拼完。
>
> **可跑性**：两个 kernel 的**数据面**（`gin.put` / `ncclGin_WeakVASignalInc`）走 `ncclDevComm` /
> `ncclGin`（NCCL GIN 设备 API），本机 NCCL 2.29.7 不具备 → **仅源码分析**；但它们脚下的
> **信号布局、dst-rank 分解、每 chunk 按 QP 均分的整数寻址、完成协议（GEQ 信号）** 全部架构无关，
> 在 H100(sm90a) 上**直接 include 真实头文件、编译运行真实设备代码 + 真实驱动 mem-op**
> （`src/bucket_allgather_backends.cu` + `harness/exp_bucket_allgather_backends.{sh,py,out.txt}`），
> 设备 4 部分全部 OK、Python host 算术层 PASS。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - host 编排：`csrc/buffers/bucket.hpp:245-330`（`BucketBuffer::all_gather`）、
>   `csrc/kernels/bucket/all_gather.hpp:24-230`（`launch_all_gather`）
> - 设备：`deep_ep/include/deep_ep/impls/bucket/all_gather/{rdma,hybrid}.cuh`
> - 布局：`deep_ep/include/deep_ep/layout/bucket/{all_gather,copy_engine,workspace,chunk}.cuh`、
>   `deep_ep/include/deep_ep/common/compiled.cuh`
> - 驱动原语：`csrc/kernels/driver/driver.cpp:14-87`
> - GIN 句柄：`deep_ep/include/deep_ep/comm/handle.cuh:11-225`
> - Python / 语义：`deep_ep/buffers/bucket.py:361-392`、`tests/bucket/test_all_gather.py`

## 1. 一段话总览

一次 `all_gather` 的目标是：每个 rank 只拥有完整张量的第 `rank` 片，收齐后每个 rank 都拿到完整张量。
和 `all_reduce` / `reduce_scatter` 不同，它**不需要求和**，所以理论上用纯数据搬运就行——第 64 篇
讲的「copy engine 路径」正是这一点。但那是**只有 NVLink、没有 RDMA**（`num_rdma_ranks == 1`）的
情形：硬件 copy engine 用 `cudaMemcpyBatchAsync` 把本地 shard 直接推到每个 peer 的对称地址。

一旦**跨节点**，就需要 `gin.put` 走网卡，这必须有 SM kernel。于是 `launch_all_gather` 按物理域
三分支：

| 物理域 | 分支 | 手段 | kernel |
|---|---|---|---|
| `num_rdma_ranks > 1`、`num_nvl_ranks == 1` | pure RDMA | 每 warp 负责一个目标 rank，`gin.put` 直发 | `rdma_all_gather` |
| `num_rdma_ranks == 1` | pure NVLink | 硬件 copy engine（第 64 篇） | 无 |
| 两者都 `> 1` | hybrid | SM 发 RDMA，copy engine 域内转发 | `hybrid_all_gather` |

关键设计：**RDMA 只在 rail 域内点对点发同一份 shard**，域内（NVLink）的复制交给 copy engine，
从而 RDMA 流量与 NVLink 流量各走各的硬件，SM 只发一次网卡、不搬 NVLink。

## 2. host 编排

**入口 `BucketBuffer::all_gather`**（`csrc/buffers/bucket.hpp:245-330`）先把张量解包、校验，再把每个
bucket 建出来——注意 `buckets[bucket_idx] = {offset, num_bytes = src.nbytes()}`（`:286-290`）里的
`num_bytes` 是**单 rank 的 shard 字节数**，不是收齐后的全量。`dst` 的整体输出是 `src.numel()*num_ranks`
的 1-D 视图（`:301-304`）。`num_sms` 恒被断言为 `0`（`:257`）——all_gather 不需要用户指定 SM 数；
host 发射后，`num_rdma_ranks > 1` 时在**计算流**上收一个 barrier epilogue（`:315-321`，注释即
「hybrid 路径在 barrier 前已由 tail 信号确认每一次 put，无需 flush」）。

**核心 `launch_all_gather`**（`csrc/kernels/bucket/all_gather.hpp:24-230`）：
`assert num_buckets ∈ (0, kNumMaxBuckets]`（`:32`）、`assert`「越界输入只支持纯 NVLink」（`:33-34`），
然后三分支。

**pure RDMA**（`:36-61`）：512 线程、`num_sms = device.get_num_sms()`、`grid_dim = num_sms`、
`cooperative=true`，JIT 实例化 `rdma_all_gather<R, num_sms, timeout, context_idx>`。

**pure NVLink**（`:62-108`）：构造 copy-engine 描述符——对每个 peer（`rank_offset 1..nvl`）把
`dst = get_sym_ptr(buffer + shard_offset, peer)`、`src = buffer + src_offset` 攒起来（`:75-91`），
越界输入额外补一次**本地**拷贝（`:93-103`）。三次 driver 调用构成握手：
`copy_engine_sync`（等对端到达）→ `copy_engine_push`（推）→ `copy_engine_sync`（等对端也推完），
`seq_idx` 单调递增（`:106-108`）。

**hybrid**（`:109-229`）：256 线程、`grid_dim = num_rdma_ranks`（**一个 SM 对一个 RDMA 目标**），
`assert num_rdma_ranks <= num_sms`（`:113`）、`assert kNumQPsPerChunk <= num_allocated_qps`（`:114`）。
SM kernel 只负责把本地 shard 用 RDMA 发出去（`hybrid_all_gather`，`:151-171`）；host 随后自己
调度 copy engine 做域内转发（`:173-226`）：先把本地 shard 推给所有 NVLink peer（`:198-203`），
再逐 chunk 等所有 RDMA peer 到齐（`copy_engine_wait`，`:213-215`）后把该 chunk 转发给 NVLink peer
（`:217-224`）。chunk 尺寸不是拍脑袋，见 §4。

## 3. pure RDMA：`rdma_all_gather` 逐行

`deep_ep/include/deep_ep/impls/bucket/all_gather/rdma.cuh:11-49`：

- 模板 `kNumRanks / kNumSMs / kNumTimeoutCycles / kContextIdx`，默认 `kNumThreads=512`、
  `kNumWarps=kNumThreads/32=16`、`kNumGlobalWarps=kNumSMs*kNumWarps`（`:11-14`）。
- `rank_idx = ncclTeamRail(nccl_dev_comm).rank`（`:26`）——用 **rail 域**（跨节点同一位置的 rank 集合）。
- **入口屏障**（`:33-36`）：`comm::gpu_barrier<true, kNumRanks, 1, kNumSMs, kNumThreads, 1, ...>`
  保证所有 rank 的本地 shard 已就位、才允许往对端写。
- **任务分解**（`:38-48`）：`global_warp_idx` 沿着 `kNumGlobalWarps` 步进地认领目标 rank；跳过自己；
  每个 warp 内 32 个 lane 再沿着 bucket 列表步进（`bucket_idx = lane; += 32`）。因为「每 warp 一个
  目标 rank」，一个 rank 要收全量就得由**它自己的所有 peer** 各推一份——本 kernel 里每个
  `global_warp_idx` 恰好负责一个 `dst_rank`，bucket 数 ≤ 32 时 lane 0..n 各管一个桶。
- 发送：`gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, dst_rank_idx)`（`:45`），`ptr = buffer +
  offset + rank_idx*num_bytes`（**自己那一份 shard**），对称指针使收发同址。

`num_sms` 取满整卡（`all_gather.hpp:39`），所以小 rank 数时大量 warp 无事可做（`dst_rank_idx` 步进
很快越过 `kNumRanks`），但保证带宽打满。

## 4. hybrid：`hybrid_all_gather` 与 chunk 自选择

### 4.1 kernel（`impls/bucket/all_gather/hybrid.cuh:11-80`）

- 模板 `kNumRDMARanks / kNumNVLRanks / kNumThreads / kNumTimeoutCycles / kContextIdx /
  kNumQPsPerChunk=AllGatherSignals::kNumQPsPerChunk=8`（`:11-13`）。
- **SM = 目标 RDMA rank、warp = QP**（`:20-27`）：`sm_idx = blockIdx.x`，
  `NCCLGin(..., qp_idx = warp_idx % kNumQPsPerChunk, NCCL_GIN_RESOURCE_SHARING_CTA)`（`:30`）——
  同一 CTA 的 8 个 warp 各占一个 RDMA QP。
- **清 tails**（`:34-39`）：grid-stride 把 `signals.tails[0..7][chunk]` 全部置 0，供本轮用。
- **入口屏障**（`:43-47`）：清完 tails 后、任何 peer 发信号之前必须可见（注释 `:42`：不需要 flush，
  上一轮的 put 已由 tail 信号确认）。
- **退避**（`:50-51`）：`sm_idx == rdma_rank_idx`（自己就是目标，无需经 RDMA 发给自己）的 SM、
  以及 `warp_idx >= kNumQPsPerChunk` 的 warp 直接返回。
- **发射**（`:54-78`）：`elect_one_sync` 后单 lane 执行，按 **bucket → chunk → QP** 顺序：
  - 每 chunk 先算 `num_bytes_in_chunk = min(bucket余量, num_chunk_bytes)`（`:61`）；
  - `num_bytes_per_qp = align(ceil_div(chunk, QPs), kNumRDMAAlignmentBytes=64)`（`:62`）；
  - `qp_offset = min(warp_idx*num_bytes_per_qp, chunk)`、`num_bytes = min(per_qp, chunk - qp_offset)`（`:63-64`）；
  - 远端动作 `ncclGin_WeakVASignalInc` 落在 **目标 rank（=sm_idx）** 的
    `signals.tails[warp_idx] + chunk_idx`（`:66-70`）；
  - `num_bytes > 0` 就 `gin.put<ncclTeamTagRail>(ptr, ptr, num_bytes, sm_idx, 0, remote_action)`（`:71-73`），
    否则**只发 signal 不发数据**（`:74-76`）——注释 `:66` 点明：**即使该 QP 没数据也照发 signal**，
    好让 host 侧 copy engine 的 `copy_engine_wait` 等待次数在 NVL/弱网下保持统一（每个 (chunk,QP) 槽
    恒定收 `num_rdma_ranks-1` 次）。

注意 `num_bytes_per_qp` 的「均分」是**向上取整到 64B** 后再乘 QP 数，因此前 `QPs-1` 段都 64B 对齐，
只有尾块的最后一段可能不是 64 的倍数（真机验证见 §5）。这满足 `gin.put` 的 RDMA 对齐要求（64B）。

### 4.2 chunk 尺寸自选择（`all_gather.hpp:116-148`）

host 侧一段显式代价模型：设 shard 总字节 `B`、`P = num_nvl_ranks - 1`、`Q = num_rdma_ranks - 1`，
`nvl_bw`/`rdma_bw` 为带宽（第 63 篇的 `get_nvlink_gbs`/`get_rdma_gbs`）：

```
push_t    = B * P / nvl_bw            # 本地 shard 推给 NVLink peer
rdma_t    = B * Q / rdma_bw           # RDMA 收齐所有 peer
forward_t = push_t * Q                # CE 把每个 RDMA peer 的 chunk 转发给 NVLink peer
for i in 1 .. min(64, B / 4MiB):      # i = chunk 波数
    t_i = max( max(push_t, rdma_t/i) + forward_t + i*10us,   # CE 受限：本地推 + 每波转发 + 每波 launch 间隔
               rdma_t + forward_t/i )                        # RDMA 受限：收齐 + 每波转发
num_best_chunks = argmin t_i
chunk_bytes = clamp( align(ceil_div(B, num_best_chunks), QPs*64), 4KiB, 256MiB )
num_chunks  = Σ_bucket ceil_div(bucket.num_bytes, chunk_bytes)     # assert <= kNumMaxChunks=8192
```

`kNumMinChunkBytes = 4MiB`（`:123`）就是「**每个 QP 至少发 512KiB**」（4MiB/8）这条经验常数：NIC 在
单 QP 512KiB 以上效率才稳定。`10e-6`（`:131`）是 CE 每波一次的固定开销。带宽模型与 all_gather 的
Python/`get_theoretical_num_sms('all_gather') == 0`（第 63 篇）一致：**all_gather 不占 SM 预算**。

### 4.3 完成协议（真实可跑）

peer 侧每 `(chunk, QP)` 对远端 `tails[qp][chunk]` 做**一次加 1**（`WeakVASignalInc`）；host 侧
`copy_engine_wait`（`driver.cpp:82-87`）对该 chunk 的 8 个 QP 槽各等 `WAIT_VALUE_GEQ，
target = num_rdma_ranks-1`（`all_gather.hpp:213-215`）。所以：

- 每个槽最终恰好 `num_rdma_ranks-1`（每个 RDMA peer 一次）；
- **按 chunk 独立**：只发了 chunk 0 时，chunk 1 的槽仍为 0，等待不会被 chunk 0 的进展提前满足；
- **GEQ 语义**：跨 launch 若槽值已 ≥ 目标（过计/陈旧），等待立即通过——`seq_idx` 单调递增（`:14-19`）
  负责清 tails，所以正常不会陈旧；
- 与 pure NVLink 的 `CopyEngineSignals.base[rank]` 不同，hybrid 用 `AllGatherSignals.tails[8][8192]`
  （512KiB）承载细粒度完成。

## 5. H100(sm90a) 实测

`harness/exp_bucket_allgather_backends.sh` 里 `nvcc -arch=sm_90a` 直接 include
`layout/bucket/{all_gather,copy_engine,workspace}.cuh`、`common/{compiled,math}.cuh`；驱动 mem-op
与 `driver.cpp:14-87` 逐行等价。结果（原始输出 `harness/exp_bucket_allgather_backends.out.txt`）：

- **Part A 布局**：`AllGatherSignals` sizeof=`524288`（=8×8192×8，512KiB）、align=64；
  `CopyEngineSignals` sizeof=`8192`（=1024×8）；`BucketSignals` sizeof=`794688`，字段偏移
  `barrier=0 / copy_engine=64 / all_gather=8256 / reduce_scatter=532544 / all_reduce=663616`；
  `BucketWorkspace` sizeof=`274792960`（`signals[8]` + `ChunkStorage(256MiB)`）。
- **Part B pure-RDMA 分解**：设备端枚举 `(R, SMs) ∈ {(8,8),(8,132),(16,8),(2,64)}`——每个非己 rank
  恰好被 **1 个 global warp** 认领、每个 bucket 恰好被覆盖一次（`bad=0`）。
- **Part C hybrid QP 均分**：10 组 chunk 长度（含 `0/1/63/64/65/511/512/513/8191/4MiB`）下，
  8 个 QP 的 `[offset, offset+len)` **恰好覆盖 `[0,chunk)` 一次**、无重叠、除尾段外 offset 均 64B
  对齐（`bad=0`）。例如 `chunk=513` → `[0+128][128+128][256+128][384+128][512+1][513+0]...`，
  `chunk=8191` → `[0+1024]...[7168+1023]`（最后一段 1023 非 64 倍数）。
- **Part D 完成协议**（真实 device `atomicAdd` + 真实 `cuStreamBatchMemOp`）：
  场景1 R=4 全体 peer → 每槽恰好 `R-1=3`、host 等待通过；场景2 只发 chunk0 → chunk1 的 8 个 QP
  槽仍为 0（per-chunk 独立）；场景3 过计（槽值 4 ≥ 3）→ GEQ 立即通过；场景4 仅 1 个 peer（< 3）→
  等待不会通过。
- **Python host 算术层**：逐行复刻 §4.2 代价模型，与独立复刻（显式枚举）在
  `RDMA ∈ {100Gb/s,400Gb/s} × NVLink 430.3GB/s × EP{2x8,4x8,8x8,2x4,4x4} × shard{1MiB..64GiB}`
  上 `chunk` 完全一致；`chunk` 恒 512B 对齐、落在 [4KiB,256MiB]；`shard ≥ 4MiB` 时每 QP ≥512KiB；
  边界 `shard=0 → chunk=4096`（夹取下限）。

## 6. 三类算子 × 三物理域（拼完的矩阵）

| | all_gather | reduce_scatter | all_reduce |
|---|---|---|---|
| 纯 NVLink | copy engine，**无 SM kernel**（第 64 篇） | `multimem.ld_reduce.add`（第 72 篇） | `multimem` 归约 + 多播存（第 72 篇） |
| 纯 RDMA | `rdma_all_gather`：每 warp 一个 dst，`gin.put`（本篇） | 反向环 + `cp.reduce.async.bulk.add`（第 72 篇） | 环 + `scale`（第 72 篇） |
| hybrid | `hybrid_all_gather`：SM 发 RDMA + CE 域内转发（本篇） | `multimem→RDMA→broadcast`（第 72 篇） | 三级流水（第 72 篇） |
| 完成信号 | `AllGatherSignals.tails[8][8192]` | `ReduceScatterSignals` 信用环 | `AllReduceSignals` 三组信号 |
| 分块映射 | 每 chunk 按 QP 均分（64B 对齐） | slot-major | rank-major |

**为什么 all_gather 是唯一能用 copy engine 做 hybrid 的**：它不做归约，域内「把整块复制给每个
NVLink peer」正好是 CE 的强项；而 reduce-scatter/all_reduce 需要求和，域内腿必须用 `multimem`
硬件归约，只能由 SM kernel 发。

## 7. 与前面各篇的关系

- 第 7 / 48 / 51 / 61 篇：EP all-to-all（dispatch/combine）是「每 token 一条消息」的稀疏通信；
  bucket 是「每张量一整块」的稠密集合通信，本篇补的是后者里 all_gather 的跨节点腿。
- 第 54 篇：pure NVLink all-gather 的 copy engine + `ChunkIterator` 分块；本篇是它的**跨节点对偶**。
- 第 62 篇：all_reduce/reduce_scatter 的三后端；本篇补齐同一矩阵的 all_gather 两格。
- 第 43 篇（EPLB `reduce_grads`）、第 9/11 篇（DeepGEMM combine 直写）：都是「专家侧集合通信」的
  不同抽象层。
- 第 63 篇：SM/QP 自动选择模型——all_gather 的 `get_theoretical_num_sms` 恒 0。

**一句话**：all_gather 的 pure-RDMA 用「每 warp 一个目标 rank、`gin.put` 自己的 shard」把跨节点
复制摊到 512 warp 上；hybrid 用「SM 发 RDMA、copy engine 域内转发」把两条网络各走各的硬件，并用
一份显式代价模型自选 chunk 尺寸（每 QP ≥512KiB）、用 `(chunk,QP)` 细粒度 tail 信号收尾。
