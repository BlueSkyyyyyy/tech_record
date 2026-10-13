# DeepEP 的 bucket 集合通信：copy-engine 路径、分块布局与算法选型

> 前面第 7 篇讲了 dispatch/combine、第 48 篇讲 combine 归约 epilogue、第 51 篇讲 dispatch 落盘 epilogue、
> 第 53 篇讲 EPLB 反向后端、第 58/59 篇讲 ptx 公共层、第 63 篇讲 barrier——这些全在 **EP 的 all-to-all**
> 语义下。本篇补上 DeepEP 里**另一套、与 EP 并列的通信设施**：`deep_ep.buffers.BucketBuffer` 的
> **bucket 集合通信**（`all_gather` / `reduce_scatter` / `all_reduce`）。
>
> 它和 MoE 前向的关系：Mega MoE / MoE 训练里的同步点（例如把各 rank 的专家梯度、或优化器状态做
> all-reduce/reduce-scatter）走的正是这套 bucket 原语；而它的**纯 NVLink all-gather 甚至不用 SM kernel**，
> 完全交给硬件 **copy engine** 驱动——这跟第 52 篇（PDL 让 kernel 通信-计算重叠）、第 9/11 篇
> （DeepGEMM 自研 combine 用 NVLink 直写）是同一主题「让通信离开计算单元」的不同实现。
>
> **本机可跑的部分**：`copy_engine_*`（`cudaMemcpyBatchAsync` + `cuStreamBatchMemOp`）是**驱动/运行时 API**，
> 与 SM 架构无关；`layout::ChunkIterator` 只依赖整数运算；带宽模型/chunk 自选是纯 host 算术。三者
> 都在 H100 真机验证。真正跨 rank 的 all_reduce/reduce_scatter 内核用 `ncclDevComm`/`ncclGin`/`multimem`
> ——本机 NCCL 2.29.7 不提供 GIN 设备 API，**仅源码分析**（沿用第 53/63 篇的处理）。

---

## 1. bucket 是什么：与 EP all-to-all 并列的一套集合通信

`BucketBuffer`（`deep_ep/buffers/bucket.py:184`，host 实现在 `csrc/buffers/bucket.hpp:36`）维护一段
**注册进 NCCL 对称窗口的连续存储**，用户往里 `allocate` 若干张量（`BucketSession.allocate:90-101`），
再对这些张量发起集合同步。三类算子：

| 算子 | 语义 | 分片方式 | host 入口 |
|---|---|---|---|
| `all_gather` | 每 rank 拥有全量的第 `rank` 片，收齐成完整张量 | 源按 rank 切片（`num_shards=(R,1)`） | `bucket.py:361-392` |
| `reduce_scatter` | 全量求和后每 rank 留自己那一片 | 源整份、目的切片（`(1,R)`） | `bucket.py:313-332` |
| `all_reduce` | 全量求和、原地写回 | 整份（`(1,1)`） | `bucket.py:334-359` |

分片维度的统一表示在 `BucketSession._collective_func`（`bucket.py:114-118`）：
```python
num_shards = {'all_gather': (num_ranks, 1),
              'reduce_scatter': (1, num_ranks),
              'all_reduce': (1, 1)}[name]
```
于是 `src_buffer = buffer.view(num_shards[0], -1)[rank % num_shards[0]]`、
`dst_buffer = buffer.view(num_shards[1], -1)[rank % num_shards[1]]`（`bucket.py:156-157`）。

一次调用在 host 侧被压成 **`comm::BucketList`**（`comm/bucket.cuh:29-39`，最多 `kNumMaxBuckets=64` 个
`Bucket{offset, num_bytes}`，`sizeof(Bucket)==16`）。每个 `Bucket` 的 `num_bytes` 是
**shard 大小或整份大小**（reduce-scatter/all-gather 用 shard 视图、all-reduce 用整份，见
`comm/bucket.cuh:12`）——`csrc/buffers/bucket.hpp` 里三种算子分别按此构造：
`reduce_scatter` 用 `dst.nbytes()`（`:161-164`）、`all_reduce` 用 `src.nbytes()`（`:225-228`）、
`all_gather` 用 `src.nbytes()`（`:287-290`）。

**「域」决定算法**。一次调用按物理拓扑三选一：`num_rdma_ranks`/`num_nvl_ranks`
（`context.get_physical_domain_size()`，`bucket.py:239`）分出：

- **纯 NVLink**（`num_rdma_ranks==1`）：见 §3，all-gather 走 copy engine、all-reduce/reduce-scatter 走
  `multimem`（NVLink 域内多播归约）。
- **纯 RDMA**（`num_nvl_ranks==1`）：三段都走 `ncclGin` 的 rail 域 put 到远端。
- **hybrid**（两者都 `>1`）：RDMA 跨节点 + NVLink 域内；all-gather 由 RDMA 内核拉入 + copy engine 域内转发。

**SM 预算**由解析式带宽模型 `get_theoretical_num_sms`（`bucket.py:236-293`）给出，用户在
`all_reduce`/`reduce_scatter` 传 `num_sms=None` 时自动选（`bucket.py:324/352`）；`all_gather` 恒 `0`
（`bucket.py:291`、`bucket.hpp:257` 断言 `num_sms==0`，因为 copy engine 不需要 SM）。

---

## 2. copy-engine 原语：`driver.cpp` 的四个函数

bucket 的「0-SM」能力来自 `csrc/kernels/driver/driver.cpp` 里四个薄封装（host 驱动 API）：

```cpp
// driver.cpp:50-66  —— push：一次批量 D2D 拷贝，偏好与计算重叠
cudaMemcpyAttributes attrs = {};
attrs.srcAccessOrder = cudaMemcpySrcAccessOrderStream;      // 源随 stream 顺序访问
attrs.flags          = cudaMemcpyFlagPreferOverlapWithCompute;
cudaMemcpyBatchAsync(dsts, srcs, sizes, count, &attrs, &attr_idx, 1, stream);

// driver.cpp:67-79  —— sync：写 seq 到每个 write_signal，再等待每个 wait_signal ≥ seq
create_mem_op(write_signals[i], seq_idx, CU_STREAM_MEM_OP_WRITE_VALUE_64);
create_mem_op(wait_signals[i],  seq_idx, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);

// driver.cpp:82-86  —— wait：等一组 signal ≥ target
create_mem_op(signals[i], target, CU_STREAM_MEM_OP_WAIT_VALUE_64, CU_STREAM_WAIT_VALUE_GEQ);

// driver.cpp:41-49  —— 批 mem-op 上限：CUDA 一次最多 255 个，超过要切分
static constexpr int kNumMaxBatchMemOps = 255;
```

三个关键事实（H100 真机实测，见 `src/deepep_bucket_copy_engine.cu`）：

1. **`cudaMemcpyBatchAsync` 是纯 D2D 指针批拷贝**，段长任意、对齐不要求（实测 `{1,2,3,7,4096,100}`
   逐字节一致）；**空列表是 no-op**（`driver.cpp:57-58`）。它**必须用非 legacy 流**——传 `0`（legacy
   null stream）会直接返回 `invalid argument`，且每个拷贝都必须带合法的
   `srcAccessOrder`（否则也是 `invalid argument`）。DeepEP 在 `comm_stream` 上调用天然满足。
2. **`cuStreamBatchMemOp` 的 write/wait 是 GPU 侧（非 host）阻塞**：把 `WRITE_VALUE_64` 与
   `WAIT_VALUE_GEQ` 编进同一条 stream，wait 会一直等到写入值到达。实测两条 stream 交替 write/wait
   能正确握手，形成**跨 stream 的流水屏障**。
3. **255 上限**：`launch_batched_mem_ops` 按 255 切分（`driver.cpp:43-45`）；实测一次提交 600 个写
   全部落地。
4. `increase_seq_idx`（`driver.cpp:14-19`）拒绝 CUDA graph capture 下使用（`cudaStreamCaptureStatusNone`），
   因为序列号语义与 capture 冲突。

> 与第 58/59 篇的对照：那两篇的 barrier / combine 是**设备端**用 `red.release.sys` / mbarrier / TMA
> 在 SM 上自旋等待；这里的 copy engine 是**把等待交给 GPU 的 stream 命令处理器 + 把搬运交给 DMA 引擎**，
> 一条 SM 都不占。

---

## 3. 纯 NVLink all-gather：协议 + 描述符构造

`launch_all_gather` 的纯 NVLink 分支（`csrc/kernels/bucket/all_gather.hpp:62-108`）是整篇最干净的一段：

```cpp
for (int rank_offset = 1; rank_offset < num_nvl_ranks; ++ rank_offset) {
    const auto peer_rank_idx = (nvl_rank_idx + rank_offset) % num_nvl_ranks;
    // 我在 peer 上的信号槽（peer 的 signals.base 的第 nvl_rank_idx 个）
    write_signals.push_back(context.get_sym_ptr(signals.base, peer_rank_idx) + nvl_rank_idx);
    // peer 在我上的信号槽
    wait_signals.push_back(signals.base + peer_rank_idx);
    for (int bucket_idx = 0; bucket_idx < num_buckets; ++ bucket_idx) {
        const auto shard_offset = offset + nvl_rank_idx * num_bytes;   // 我的 shard
        dst_ptrs.push_back(context.get_sym_ptr(buffer + shard_offset, peer_rank_idx));
        src_ptrs.push_back(buffer + shard_offset);
        sizes.push_back(num_bytes);
    }
}
// 到达屏障 → push 我的 shard 给每个 peer → 完成屏障
driver::copy_engine_sync(stream, increase_seq_idx(stream, seq_idx), write_signals, wait_signals);
driver::copy_engine_push(stream, dst_ptrs, src_ptrs, sizes);
driver::copy_engine_sync(stream, increase_seq_idx(stream, seq_idx), write_signals, wait_signals);
```

**为什么两个 `copy_engine_sync` 就够**：

- 信号布局是 `CopyEngineSignals{ uint64_t base[kNumMaxNVLRanks]; }`（`layout/bucket/copy_engine.cuh:10-13`）
  ——**每个 rank 一个 slot**，`base[peer]` 是 peer 的缓冲里的 slot，写入时用 `+ nvl_rank_idx`
  定位「我写进 peer 的第 islot」。于是「我在所有 peer 上的第 l 槽」+「所有 peer 在我上的槽」正好
  组成一个**全互连屏障**。
- `seq_idx` 是 **per-context 单调递增计数器**（`bucket.hpp:46` 的 `seq_indices[kNumMaxContexts]`），
  每次 `increase_seq_idx` 自增。屏障用 `GEQ`（`driver.cpp:76-77`），所以**不要求双方用同一个值**——
  我先到就写较大的 seq，后到的只要 ≥ 就行。第一次 sync 保证「所有 rank 的本地 shard 已就位」，
  第二次 sync 保证「所有 peer 都把我需要的 shard 推过来了」，于是读回即完整。

**H100 真机结果**：

- 单 GPU 用 R 条 `cudaStreamNonBlocking` + R 组对称缓冲/信号**模拟多 rank**：R=2/4 通过；R≥8 会因
  **单 GPU 无法真正并发所有流**而互锁（互锁是模拟的产物，不是 DeepEP 的问题）。
- 因此进一步做 **真·多 GPU**（每 rank 一块 H100，`cudaDeviceEnablePeerAccess` 全互联，跨 NVLink peer
  做 `cudaMemcpyBatchAsync`、`cuStreamBatchMemOp` 写 peer 内存）：**R=2/4/8 全部逐字节一致**。
  这坐实了第 63 篇里 `NCCLGin::get_sym_ptr` 的「对称偏移」语义在 copy-engine 路径同样成立。

---

## 4. 分块层：`ChunkIterator` / `ChunkStorage` / 信号布局

RDMA/hybrid 路径对**物理 chunk** 有严格的负载均衡切块要求：一组 worker 各自以固定 stride 遍历
「把所有 bucket 首尾拼成的一条 chunk 序列」，互不重叠。这个迭代器就是
`deep_ep/include/deep_ep/layout/bucket/chunk.cuh`：

```cpp
// chunk.cuh:58-102
template <int kNumBytesPerChunk, int kStride>
struct ChunkIterator {
    const int num_buckets;  const comm::BucketList& buckets;
    int num_total_chunks = 0;   // 跨所有 bucket（含每个 bucket 的尾块）
    // next()：chunk_idx += kStride；在 bucket 序列上定位（跨过边界）
    //   只有最后一个 chunk 可能小于 kNumBytesPerChunk
};
```

要点：

- **构造时** `num_total_chunks += ceil_div(buckets[i].num_bytes, kNumBytesPerChunk)`（`:73-74`）——
  注意注释 `:70`：「chunk 一般至少几 KB，覆盖到几 TB，`int` 足够」。
- **`next()`**（`:77-98`）先 `chunk_idx += kStride`，再用 `bucket_idx/bucket_start_chunk_idx` 两个
  私有游标跨 bucket 边界前进；`num_chunk_bytes` 仅在 `is_last_chunk` 时收窄
  （`:88-89`，「只有末块可能不满，余数放进 `int` 不需要 64 位 `min`」）。
- **覆盖性保证**（注释 `:56-57`）：`kStride` 个起点在 `[0, kStride)` 的迭代器**恰好覆盖每个 chunk 一次**。
  这是 hybrid all-reduce 里三类 warp（multimem 归约 / RS issue / reduce / broadcast）用不同 stride
  分头工作的正确性前提（`all_reduce/hybrid.cuh:151-152/175-176/216-217/282-283`）。

物理存储是 `ChunkStorage`（`chunk.cuh:12-17`，**256 MiB**，64B 对齐），视图 `ChunkView1D/2D`
（`:19-52`）把「第 rank 的第 chunk 块」映射成字节偏移。信号区布局（`layout/bucket/*.cuh`）：

| 结构 | 定义 | 大小 | 用途 |
|---|---|---|---|
| `BarrierSignals` | `barrier.cuh:11-25` | 64 B | 计数 + 双相位槽（第 63 篇详） |
| `CopyEngineSignals` | `copy_engine.cuh:10-13` | 8192 B | all-gather NVLink 的每 rank 信号槽 |
| `AllGatherSignals` | `all_gather.cuh:11-15` | 512 KiB | `tails[8 QPs][8192 chunks]` |
| `AllReduceSignals` | `all_reduce.cuh:11-15` | 128 KiB | `counters[16384]` |
| `ReduceScatterSignals` | `reduce_scatter.cuh:12-17` | 128 KiB | `heads/tails[8192]` |

`BucketSignals` 顺序拼这五个（`workspace.cuh:13-19`），`BucketWorkspace = signals[kNumMaxContexts=8] +
ChunkStorage`（`:24-27`）。`BucketSignals=794688 B`、`BucketWorkspace=274792960 B`（H100 真机
`sizeof` 实测，见实验 Part D）。约束 `sizeof(BucketSignals)*8 ≤ kNumMaxSignalBytes=16MiB`
（`workspace.cuh:19`）满足。

**H100 真机验证 `ChunkIterator`**（直接 include 真实头文件跑设备代码）：`kStride∈{1,2,4,8,128}` ×
多 bucket（含空组、非整块尾）——**每个 chunk 恰好访问一次、字节和=Σbucket、部分尾块数正确**。

---

## 5. hybrid all-gather：chunk 尺寸自选与带宽模型

hybrid 分支（`all_gather.hpp:109-229`）把 RDMA 与 NVLink 串起来：`grid_dim=num_rdma_ranks`（一个 SM 一个
RDMA peer），每个 SM 发 `kNumQPsPerChunk=8` 个 QP 把本地 shard 的 chunk 发到各 RDMA peer；副本到达后，
**copy engine** 负责把先进来的 chunk 域内转发给 NVLink peer（`:205-225`），并用 `all_gather.tail`
信号（`:213-215`，`copy_engine_wait`）等 chunk 到齐。

**chunk 尺寸怎么选**（`all_gather.hpp:120-142`）是一段显式代价模型：

```cpp
const auto push_t    = num_shard_bytes * (num_nvl_ranks - 1) / nvl_bandwidth;   // 域内推送
const auto rdma_t    = num_shard_bytes * (num_rdma_ranks - 1) / rdma_bandwidth; // 跨节点
const auto forward_t = push_t * (num_rdma_ranks - 1);                           // CE 转发
for (int i = 1; i <= min(64, num_shard_bytes / 4MiB); ++i) {
    // CE-bound：本地推送或首波之后开始转发所有波；RDMA-bound：只转发最后一波
    const auto t = max(max(push_t, rdma_t / i) + forward_t + i * 10e-6f,   // 10us = CE 启动开销
                       rdma_t + forward_t / i);
    if (t < best_t) best_t = t, num_best_chunks = i;
}
// 再对齐到 kNumQPsPerChunk*64B、clamp 到 [4KiB, 256MiB]
```

它的动机（注释 `:120-123`）：「CE 先推本地 shard，之后每落一个 chunk 波（每个 RDMA peer 一个）就转发
一波，每波付一次启动间隔；chunk 地板定成让每个 QP 至少 512KiB，此时 NIC 效率稳定」。
`num_chunks = Σ ceil_div(bucket_bytes, num_chunk_bytes)` 必须 ≤ `kNumMaxChunks=8192`（`:148`）。

**H100 host 算术对拍**（Python，见 `harness/exp_deepep_bucket.py`）：全枚举取最小 + 后处理对齐/clamp
逐条满足；典型 `shard=256MiB R=8 N=8 → num_best_chunks=2, num_chunk_bytes=128MiB`。

**SM 预算模型** `get_theoretical_num_sms`（`bucket.py:236-293`）逐分支：

- NVLink reduce-scatter（`:241-246`）：`num_sms = align(max(6, ceil(max(d/180, d/45))), 2)`，
  `d = nvlink/N`。H100 实测 `nvlink=430.3 GB/s`（= Σlinks 478.1 × 0.9，`envs.py:231`），
  N=8 → **6 SMs**。
- NVLink all-reduce（`:275-278`）：`align(max(4, ceil(...)), 2)`，N=8 → **4 SMs**。
- hybrid 各分支则 `min(nvlink/N, rdma·R/(R-1))` 再乘读写系数（reduce-scatter ×2、all-reduce ×3）、
  乘 1.2 安全系数。
- 每 SM 的 HBM 读/写带宽写死 `180/45 GB/s`（`envs.py:194-204`，注释即 TODO）。
- `num_allocated_qps = num_rdma_ranks==1 ? 1 : kNumMaxQPs/num_rdma_ranks`（`bucket.hpp:86`）。

Python 用 `ast` **逐行抽取真实 `get_theoretical_num_sms`** 在 9 个域配置 × 3 算子下与独立公式**逐点一致**。

---

## 6. SM 驱动的三段算法（NCCL GIN，仅分析）

非 all-gather 路径需要 SM 参与归约，三段算法各有一份 `nvlink/rdma/hybrid` 内核：

**reduce-scatter**：

- NVLink（`reduce_scatter/nvlink.cuh`）：`multimem_ld_reduce_add_f32`（跨 NVLink 域**一次读出所有 rank 的
  和**，`ptx.cuh` 的 `multimem.*`，第 59 篇）+ 写本地 shard；`bucket.get_num_elems<float4>()`。
- hybrid（`reduce_scatter/hybrid.cuh`，host `reduce_scatter.hpp:37-82`）：`8 multimem + 2 issue + 2 reduce`
  warp，64KiB/减 warp 的 TMA 缓冲。
- RDMA（`reduce_scatter/rdma.cuh`，637 行；host `:83-149`）：分三种模板实例，`num_batch_chunks =
  min(8, R)`；bf16 精度只在纯 RDMA 支持（`reduce_scatter.hpp:34-35` 断言）。

**all-reduce**：

- NVLink（`all_reduce/nvlink.cuh:36-51`）最简：对本地 shard 做
  `multimem_ld_reduce_add_f32 → 乘 scale → multimem_st_f32`（**fused reduce-scatter + multicast-store
  all-gather**，一行注释 `:35`）。
- hybrid（`all_reduce/hybrid.cuh`，340 行）是 showcase：`multimem 归约（域内）→ RS issue warp 用
  `gin.put` 把 chunk 发到 owner RDMA rank（带 `StrongVASignalAdd`）→ reduce warp 收齐 R 份做
  TMA reduce-add 归约、再 `gin.put` 广播回去 → broadcast warp 用 `hybrid_ar_broadcast`（`:12-44`，
  `multimem_cp_async_bulk`）域内多播。整套用 `ChunkIterator` 分头遍历、`counters/rs_tail/ag_tail`
  三组信号 + `bcst_head` 信用流控（`:123-126`）；信号容量 `2·R·slots + reduce_warps·R ≤ 16384`
  （`:64-72` static_assert）。host 侧 warp 数由 `rdma/nvl` 与 SMEM 预算算出
  （`all_reduce.hpp:32-47`）。
- RDMA（`all_reduce/rdma.cuh`，261 行；host `:103-141`）：`8 issue + 8 reduce` warp。

**all-gather 的 RDMA / hybrid 内核**（`all_gather/rdma.cuh`、`all_gather/hybrid.cuh`）：
RDMA 分支每个 `global_warp_idx` 负责一个目标 rank、32 lane 平分 bucket 发 `gin.put`（`rdma.cuh:39-49`）；
hybrid 分支「SM=目标 RDMA rank、warp=QP」，每 chunk 均分到 QP（对齐 64B）后发，**即使该 QP 没数据也
照发 signal**（`hybrid.cuh:72-78`，让 CE 的 `copy_engine_wait` 等待次数统一）。

> 这些内核都带 `comm::gpu_barrier`（`:26-33` 进场 / 结束）、`cooperative=true`、`cluster_dim=2-(num_sms%2)`。
> 因为用 `ncclDevComm`/`ncclGin`/`multimem`，本机 NCCL 不具备，**仅分析**。

---

## 7. 与前面篇章的关系

| 主题 | 本篇 | 关联篇 |
|---|---|---|
| 对称偏移 / 窗口句柄 | `get_sym_ptr` 用在 copy-engine 描述符里（§3） | 第 63 篇 `NCCLGin` |
| 设备端搬运原语 | copy engine（DMA + stream mem-op） | 第 59 篇 `ptx.cuh`（TMA/multimem） |
| PDL / 重叠 | all-gather 靠 CE 天然不占 SM | 第 52 篇 PDL；第 9/11 篇 combine 直写 |
| 归约语义 | `multimem_ld_reduce_add` / TMA reduce-add | 第 53 篇 EPLB `tma_store_reduce_add` |
| 分块/负载均衡 | `ChunkIterator` 固定 stride 覆盖 | 第 24 篇 MQA 调度器、第 53 篇任务窃取 |

**一句话**：EP 的 all-to-all 是**每个 token 一条消息**的稀疏通信，bucket 是**每张量一整块**的稠密集合通信；
bucket 用「copy engine + 流内存操作」把 all-gather 做到 0 SM，用「multimem + TMA reduce-add + GIN」
把 all-reduce/reduce-scatter 做成带宽最优的域内/跨域流水。

---

## 8. H100 实测证据台账

实验：`src/deepep_bucket_copy_engine.cu` + `harness/exp_deepep_bucket.{sh,py}` + `.out.txt`。

| 项 | 结果 |
|---|---|
| `cudaMemcpyBatchAsync` 64 段 D2D | 逐字节一致；任意长度/对齐也一致；空列表 no-op |
| 必须非 legacy 流 | 传 `0` → `invalid argument`（真机复现） |
| `cuStreamBatchMemOp` write/wait | 顺序正确、GEQ 语义成立、跨 stream 握手成立（GPU 侧阻塞） |
| 255 批拆分 | 600 个 mem-op 全部落地 |
| 布局 `sizeof`（真实编译） | `Bucket=16`、`BarrierSignals=64`、`CopyEngineSignals=8192`、`AllGatherSignals=512KiB`、`AllReduceSignals/ReduceScatterSignals=128KiB`、`BucketSignals=794688`、`BucketWorkspace=274792960`、`ChunkStorage=256MiB` |
| `ChunkIterator`（真实设备代码） | 5 组 stride×bucket：每 chunk 恰好一次、字节和、部分尾块全对 |
| 纯 NVLink all-gather（真·多 GPU） | **R=2/4/8 每块 GPU 收齐全部 shard，逐字节一致** |
| 单 GPU 模拟多 rank | R=2/4 通过；R≥8 互锁（单卡并发限制，非算法问题） |
| `get_theoretical_num_sms` | 9 域配置×3 算子 与独立公式逐点一致；H100 NVLink-only 8 rank：RS=6 / AR=4 |
| 带宽探测 | `nvidia-smi nvlink -s` Σlinks=478.1 → `get_nvlink_gbs()=430.3`（×0.9）；`ibstat` 缺失 → rdma=0 |
| chunk 自选择 + 后处理 | 全枚举最小 + 512B 对齐 + `[4K,256MiB]` clamp 全部满足 |
| 设备程序总判定 | **PASS=21 / FAIL=0**（Python）+ **PASS=20 / FAIL=0**（Parts 1/2/4）+ Part3/5 全通过 |

**边界**：单 GPU 上 R≥8 的互锁提醒我们——**「通信-计算重叠」的正交性来自真实并发的执行单元**；
用一条 GPU 的多 stream 模拟多 rank 在等待语义上不等价于多张独立 GPU（真机 R=8 正是靠这个才成立）。

---

## 参考

- 仓库：DeepEP（commit `93eb6eb`，v2.5.0）——`csrc/kernels/driver/driver.cpp`、
  `csrc/kernels/bucket/*.hpp`、`csrc/buffers/bucket.hpp`、`deep_ep/buffers/bucket.py`、
  `deep_ep/include/deep_ep/{layout/bucket,impls/bucket,comm}/*.cuh`。
- CUDA 驱动/运行时：`cudaMemcpyBatchAsync`（CUDA ≥ 12.8，本机 13.2）、
  `cuStreamBatchMemOp`（`CU_STREAM_MEM_OP_WRITE_VALUE_64` / `WAIT_VALUE_GEQ`）。
- 本专题：第 52 篇（PDL）、第 53 篇（EPLB）、第 59 篇（ptx 原语）、第 63 篇（barrier/handle）。
