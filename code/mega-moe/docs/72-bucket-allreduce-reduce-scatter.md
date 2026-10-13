# DeepEP bucket 的 all_reduce / reduce_scatter：三种后端、信用环协议与 bf16 随机舍入归约

> 本文对应路线图 **P62-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 72 篇。
>
> 第 64 篇（P54-1）拆了 DeepEP 与 EP all-to-all **并列的通信设施** `BucketBuffer`，但当时只把
> **纯 NVLink all-gather**（不用 SM kernel、交给硬件 copy engine）讲透，另外两个算子
> `all_reduce` / `reduce_scatter` 因为「用 `ncclDevComm`/`ncclGin`/`multimem`，本机 NCCL 2.29.7
> 不提供 GIN 设备 API」一笔带过为「仅源码分析」。本篇补上这两个算子的**完整实现解析**。
>
> **与专题的关系**：MoE 训练里「把各 rank 的专家梯度 / 优化器状态做 all-reduce 或 reduce-scatter」
> 走的正是这套原语；它与第 43 篇（EPLB 副本梯度 `reduce_grads`）、第 9/11 篇（DeepGEMM 的 combine
> 用 NVLink 直写）、第 52 篇（PDL 让通信离开计算单元）是同一主题的不同实现。
>
> **可跑性**：六个 kernel（`all_reduce/{nvlink,rdma,hybrid}` 与 `reduce_scatter/{nvlink,rdma,hybrid}`）
> 的**数据面**确实依赖 NCCL 设备 API 与 SM100 的 `cvt.rs` → **仅源码分析**；但它们脚下
> **信号/工作区布局、`ChunkIterator` 覆盖、两种分块映射、归约算子（`fadd4`/`fmul4`）、bf16 随机
> 舍入哈希、host 发射配置算术**全部架构无关，在 H100(sm90a) 上**直接 include 真实头文件、编译运行
> 真实设备代码**（`src/deepep_bucket_collectives.cu` + `harness/exp_deepep_bucket_collectives.{sh,py,out.txt}`），
> **30 PASS / 0 FAIL**。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - host 发射：`csrc/kernels/bucket/all_reduce.hpp`（144 行）、`reduce_scatter.hpp`（180 行）
> - Python：`deep_ep/buffers/bucket.py`（`_collective_func:103`、`reduce_scatter:313`、`all_reduce:334`、`all_gather:361`）
> - 布局：`deep_ep/include/deep_ep/layout/bucket/{all_reduce,reduce_scatter,workspace,chunk}.cuh`、`comm/bucket.cuh`
> - 设备：`deep_ep/include/deep_ep/impls/bucket/{all_reduce,reduce_scatter}/{nvlink,rdma,hybrid}.cuh`
> - 公共层：`common/{ptx,utils,math,compiled}.cuh`
> - 语义参考：`tests/bucket/test_all_reduce.py`、`test_reduce_scatter.py`；`third-party/tilelang_ops/stochastic_rounding.py`
> - README `§Bucket collectives (experimental)`（`:291-328`）

## 1. 一段话总览

`BucketBuffer` 把三类集合通信（`all_gather` / `reduce_scatter` / `all_reduce`）做成**对一组张量的
批量调用**（`deep_ep/buffers/bucket.py:313-392`），并按**物理域**自动三选一后端：纯 NVLink（同一节点，
`num_rdma_ranks==1`）、纯 RDMA（跨节点，`num_nvl_ranks==1`）、hybrid（两者都有）。分片语义由
`_collective_func:114-118` 统一表示：

```
num_shards = {'all_gather': (R, 1), 'reduce_scatter': (1, R), 'all_reduce': (1, 1)}[name]
```

即 all_gather 把**源**按 rank 切片（每 rank 只拥有第 `rank` 片）、reduce_scatter 把**目的**按 rank
切片（每 rank 只留第 `rank` 片）、all_reduce 全量原地。默认语义是 **FP32 求和**，`scale` 缩放结果
（README `:316`）。

**为什么这三个算子不能都交给 copy engine**（像 all-gather 那样）：all-gather 是纯数据搬运，
`cudaMemcpyBatchAsync` 就够；而 reduce-scatter / all_reduce 需要**求和**——要么用 NVLink 硬件多播
归约 `multimem.ld_reduce.add`（NVLink-only/hybrid 的域内腿），要么用 SM kernel 做 RDMA 收发 +
`cp.reduce.async.bulk.add.f32`（TMA 归约写）。所以这两个算子**必须有 SM kernel**。

## 2. host 发射：三后端与解析式配置

两个 host 入口都先做参数检查，再按 `num_rdma_ranks`/`num_nvl_ranks` 三分支：

**`launch_all_reduce`**（`all_reduce.hpp:17-142`）：

| 分支 | 条件 | 线程/角色 | 信号预算断言 |
|---|---|---|---|
| hybrid | `R_rdma>1 ∧ R_nvl>1`（`:27`） | multimem + issue + reduce + broadcast（`:33-43`） | `2·R·S + SM·warps·R ≤ 16384`（`:46-47`） |
| NVLink-only | `R_rdma==1`（`:73`） | 1024 线程，纯 `multimem` | — |
| RDMA-only | else（`:103`） | 8 issue + 8 reduce warp（`:105-108`） | smem ≤ 上限（`:110-111`） |

hybrid 的派生配置（`:32-44`）值得一读：`mm_warps = rdma<=32 ? 8 : 4`、`num_chunk_bytes =
rdma<=4 ? 256KB : 64KB`、`num_slots = ChunkStorage::kNumBytes / num_chunk_bytes / rdma`。**为了
省 smem，broadcast warp 越多单 warp smem 越小**（`:40-41` 的 `broadcast_warps<=2 ? 64:32 / 32:8 KB`）。

**`launch_reduce_scatter`**（`reduce_scatter.hpp:19-178`）：hybrid（`:37-82`）、
RDMA+`bf16`（`:83-117`，**只有 RDMA-only 支持 bf16**，`:34-35` 断言）、RDMA-only（`:118-149`）、
NVLink-only（`:150-176`）。bf16 分支还会按 `R_rdma` 调 warp 配比（`:84-91`）。

## 3. 信号布局与预算（真实 `sizeof`）

两个算子各用一块信号区，都从 `BucketWorkspace` 里切（`layout/bucket/workspace.cuh:14-26`）：

```
struct alignas(64) BucketSignals {
    BarrierSignals barrier;            // 64B：计数 + 双相位槽
    CopyEngineSignals copy_engine;     // 8×1024 = 8192B（all-gather 用）
    AllGatherSignals all_gather;       // 8×8192×8 = 524288B
    ReduceScatterSignals reduce_scatter;
    AllReduceSignals all_reduce;
};
struct alignas(64) BucketWorkspace { BucketSignals signals[kNumMaxContexts=8]; ChunkStorage chunk_storage; };
```

- `AllReduceSignals`（`all_reduce.cuh:10-13`）：`int64_t counters[16384]`，128 KiB，**只一块**。
- `ReduceScatterSignals`（`reduce_scatter.cuh:10-24`）：`heads[8192]` + `tails[8192]`，128 KiB；
  寻址由 `get_head_ptr<kSlotsPerRank>(rank,slot) = heads + rank*kSlotsPerRank + slot`（尾指针同式，
  但在 `tails` 里）。
- `ChunkStorage`（`chunk.cuh:12-17`）：一块 256 MiB 的暂存，**RDMA 收发的中转槽**就切自这里。

H100 真机 `sizeof` 实测（`harness/exp_deepep_bucket_collectives.out.txt`）：`BucketSignals=794688`、
`BucketWorkspace=274792960`（=8×794688 + 256 MiB）。`BucketSignals*8 = 6357504 ≤ kNumMaxSignalBytes
= 16 MiB`（`compiled.cuh:82`），正是 `workspace.cuh:21` 那条 `static_assert`。

**信号预算**是关键设计约束（每个 launch 的槽数必须放得下）：

- all_reduce（`all_reduce/rdma.cuh:31-37`）：`S = ChunkStorage/num_chunk_bytes/R`，断言
  `2·R·S ≤ 16384`（`rs_head` 与 `rs_tail` 各占一半，`:62-63`）。
- reduce_scatter（`reduce_scatter/rdma.cuh:32-35`）：`S = ChunkStorage/32KB/R`，断言 `R·S ≤ 8192`。

H100 实测：all_reduce hybrid 在 `R_rdma∈{2..64}` 全部满足预算；reduce_scatter `S = 8192/R`
（`R=2→4096, R=8→1024`），`R·S ≤ 8192` 恒成立。

## 4. 分块：`ChunkIterator` 与**两种不同的映射**

RDMA/hybrid 路径都用 `ChunkIterator<kChunkBytes,kStride>`（`chunk.cuh:58-102`）把「所有 bucket」当成
一条**扁平的 chunk 序列**，每个迭代器从 `start` 出发按 `kStride` 步进——`kStride` 个不同起点
**恰好覆盖每个 chunk 一次**（`:54-57` 注释）。第 64 篇已把它逐行验证过，这里只补**它怎么被两个算子
用**。

关键差异在于 **chunk → (对端 rank, slot, generation) 的映射**：all_reduce 和 reduce_scatter 用到
**完全不同**的映射。

**all_reduce（rank-major）**（`all_reduce/rdma.cuh:96-104`）——每个 chunk 发给**某个 rank**，同一个
slot 被 R 个 peer 轮流使用：

```
dst_rank       = chunk_idx % R
chunk_idx_in_shard = chunk_idx / R
slot           = chunk_idx_in_shard % S
head_target    = chunk_idx_in_shard / S     # generation
```

**reduce_scatter（slot-major）**（`reduce_scatter/rdma.cuh:100-101`）——每个 chunk 由**所有 rank**
各贡献一片，slot 直接由 chunk 序号定：

```
slot      = chunk_idx % S
head_target = chunk_idx / S
```

H100 实测：R=4 时 8191/8192 个 chunk 的 `(slot,gen)` 在两种映射下不同——**它们的 slot 粒度根本不是
一回事**。另实测 `ChunkIterator` 在 `[200000,65536,131072]` 三桶、chunk=64KB、`kStride=4` 下产出
恰好 7 块（4+1+2），各 stride 不重叠不遗漏，每块 `(offset_in_bucket, num_chunk_bytes)` 与手算一致
（仅末块可不满 64KB）。

## 5. 归约语义：三种实现，一个数学

数学上两个算子都是「对 R 个 rank 的贡献求和」：
- all_reduce：`out = scale · Σ_p x_p`（结果写回**每个** rank，原地）；
- reduce_scatter：`out[r] = scale · Σ_p x_p[shard r]`。

但**求和顺序**在 fp32 下会影响逐位结果——这正是测试 `rtol=0, atol=0` 能过的原因。

### 5.1 NVLink-only：`multimem.*` 硬件归约

`all_reduce/nvlink.cuh:36-51` 里，每个 rank 只处理**自己那一片 shard slot**，用
`multimem_ld_reduce_add_f32_with_gt_pred`（`ptx.cuh:162`，即 `multimem.ld_reduce…add.v4.f32`）
在 NVLink 域内**一次读出所有 rank 的求和**，再 `multimem_st_f32`（`ptx.cuh:177`）多播写回。
这就是注释 `:35` 说的 **"Fused reduce-scatter + multicast-store all-gather"**：把每 rank 的
NVLink 流量从 O(N) 降到 O(1)。`reduce_scatter/nvlink.cuh:36-50` 是它的对偶——`ld_reduce` 求和后
用**普通** `st` 写自己那一片（不广播）。

### 5.2 RDMA-only：反向环 + TMA 归约写

**all_reduce**（`all_reduce/rdma.cuh`）：
- issue warp（`:89-132`）：`elect_one` 单 lane 用 `gin.put` 把 chunk 发到 `dst_rank`，并
  `StrongVASignalAdd` 把对端的 `rs_tail` +1（`:123-130`）。
- reduce warp（`:133-252`）：本地先有自己那份，然后**按环序** `src = (rank + i) % R, i=1..R-1`
  逐个等对端 `rs_tail`（`:158-172`），用 `tma_store_reduce_add_f32`（`ptx.cuh:148`，
  `cp.reduce.async.bulk…add.f32`）累加进输出（`:183-184`）。**`kWithScale` 时最后一个 peer 留给
  寄存器**做 `scale·(x+y)`（`:153` 的 `kNumTMAPeers = R-2`、`:199-242`），使缩放只乘一次。
  最后把归约好的 chunk **广播**给每个 peer（`:246-249`）。

**reduce_scatter**（`reduce_scatter/rdma.cuh:130-226`）：
- issue warp（`:94-129`）向每个 peer 发**自己那一片的 chunk 副本**，`StrongVASignalAdd` 对端 tail。
- reduce warp 按**反序环** `src = (rank - peer_offset - j) % R`（`:175-176`）累加，`fadd4`
  （`ptx.cuh:562`）在暂存槽里的 float4；`kWithScale` 只在**最后一批**乘 `scale`（`:195-201`）。
  释放信用用 `gin.put_value(head, tail_target, src)`（`:217-222`）。

### 5.3 求和顺序 = 测试参考顺序（逐位可复现的根因）

`test_reduce_scatter.py:78-87` 的参考实现：`all_to_all_single` 后对
`peer = (rank - i) % R, i=0..R-1` **依次** `add_`，再 `mul_(scale)`。即
**本地 rank 先行，然后 rank-1、rank-2、…**。

这与 kernel 的顺序**逐字一致**：`reduce_scatter/rdma.cuh:145` 先读本地 dst，`:175` 的
`(rank - peer_offset - j) % R` 即 `rank-1, rank-2, …`；hybrid 版 `reduce_scatter/hybrid.cuh:223-246`
也是 `i=0` 本地（只释放信用、不归约）、`i≥1` 才累加 `(rank - i) % R`。**正因为顺序一致，测试才能
用 `rtol=0, atol=0` 逐位判等。**

H100 实测：把 `fadd4` + `fmul4` 的反序环累加逐行搬进设备内核，与独立 `np.float32` 同序实现对拍
**逐位 mismatch=0/16384**；而**升序**朴素求和在 4096 个元素里就有 **922 个 bit 不同**——坐实
「顺序重要，且 kernel/参考选的是同一个反序环」。

### 5.4 hybrid：两级域的三级归约

hybrid 把两级域编进同一个 kernel（`all_reduce/hybrid.cuh:12-51`）：

1. **multimem warp**（`:145-168`）在 **NVLink 域内** 先做 `ld_reduce`，得到「本 RDMA 组内各 rank 的
   和」，`iterate_ld_st` 分块（`utils.cuh:19`）；
2. **issue warp**（`:171-208`）按 `rail_seq` 折算 `owner/slot/round`（`:178-182`），跨 RDMA 发 chunk；
3. **reduce warp**（`:209-271`）按 `source = (rank + step) % R`（`:241`）收齐后 TMA 归约，`:250-253`
   区分 step==0（TMA copy）与 step≥1（TMA reduce-add），再 `red_add_rel_gpu(ag_tail)`；
4. **broadcast warp**（`:272-332`）按 peer 分区轮询 `ag_tail` 水位（`:294-302`），用
   `hybrid_ar_broadcast`（`:11-41`，`multimem_cp_async_bulk`，`ptx.cuh:188`）把结果多播回 NVLink 域。

reduce_scatter 的 hybrid（`reduce_scatter/hybrid.cuh`）是对偶：multimem（`:102-138`）先在域内归约、
RDMA issue（`:142-200`）、reduce（`:201-278`）累加远端、再本地 multimem 广播。

## 6. 信用环协议：head / tail

RDMA 路径的收发靠 `ReduceScatterSignals` 的一对计数器做**背压**，这是「不丢数据、不复用未消费槽」
的根：

- **发送方**在发一个 chunk 前等 `head[对端][slot] >= generation`（all_reduce `:107-119`；
  reduce_scatter `:109-112`），即「对端已消费上一代」；
- **接收方**收到数据后 `tail` 由发送方的 `StrongVASignalAdd` +1，接收方等 `tail[源][slot] >= gen+1`
  （all_reduce `:160-171`；reduce_scatter `:154-156`）；
- **接收方消费完**把 `head[自己][slot] = gen+1` 回给源（reduce_scatter `:217-222`），**但最后一个
  用到该槽的 chunk 不需要信用**（`:137` 的 `needs_credit = chunk_idx + S < num_total_chunks`；
  all_reduce 侧在 hybrid `:266` 有同款判断）。

每个 launch 开始前都会把整块信号清零（`reduce_scatter/rdma.cuh:83-87` 用
`get_head_ptr/get_tail_ptr` 逐槽写 0），再用一次 `gpu_barrier` 让清零对所有 rank 可见（`:88-91`）。

H100 实测：`get_head_ptr<64>/get_tail_ptr<64>` 的字节偏移 = `(rank*64+slot)*8` 与
`(8192 + rank*64+slot)*8`——与手算一致（`heads` 是 8192 个 int64）。

## 7. bf16 reduce-scatter：随机舍入哈希

RDMA-only 的 `bf16` 路径（`reduce_scatter/rdma.cuh:235-635`，只在 `R_rdma>1 ∧ R_nvl==1` 可用，
`reduce_scatter.hpp:34-35`）为省一半带宽，把发送侧 FP32 **转成 BF16** 再传，接收侧累加回 FP32。
**朴素截断会引入系统性偏差**，所以发送侧用**随机舍入**（stochastic rounding）。

三段角色：cast warp（`:348-441`）把每个目的地 chunk 的 FP32 就地压成 BF16（写在自己那半）；
issue warp（`:442-528`）发 BF16；reduce warp（`:529-628`）用 `ptx::accumulate`（`ptx.cuh:593`）
把 BF16 累加进 FP32。

舍入字来自一个**四元组哈希**（`:401-412`），用同一个 float4 的四个 32 位模式 + 该 float4 在
bucket 里的**全局下标**，生成两个独立的随机字 `h1`/`h2`（分别给 float4 的两对）：

```
h  = x·0x5671D42B + y·0x9995E499 + z·0xACE1B8A5 + w·0xE153538D + global_float4_index
h ^= h>>23;  h *= 0x7FEB352D;  h ^= h>>16;          // murmur3 的 fmix 收尾
h1 = h·0x846CA68B; h1 ^= h1>>11;                    // 给 (x,y)
h2 = h·0xD35A2D97; h2 ^= h2>>11;                    // 给 (z,w)
```

真实的舍入由 SM100 指令 `cvt.rs.bf16x2.f32`（`ptx.cuh:604`）用这 16 个随机位完成；
`accumulate` 在 SM100 上走融合 `add.rn.f32.bf16`、sm90 回退为「BF16→FP32 精确转换再相加」
（BF16 尾数只有 8 位，放进 FP32 是精确的）。测试用
`third-party/tilelang_ops/stochastic_rounding.py`（`:53-65`）作**逐位参考**——预编译 SM100 时
`rtol=0, atol=0`，否则只做松检查（`test_reduce_scatter.py:91-94`）。

**跨篇联系**：两个终端乘子 `0x846CA68B` / `0xD35A2D97` 与 **DeepGEMM 的
`StochasticRoundToBF16`**（`epilogue/operators.cuh:35-48`，见第 21/43 篇）**完全相同**——DeepEP 的
bf16 集合通信复用了 DeepGEMM 的随机舍入方案，只是把「四元组哈希」的预混合从 DeepGEMM 的部分哈希
换成了「四个模式字各乘常数 + 全局下标」，以便在张量上处处去相关。

H100 实测（架构无关）：把这段哈希逐行搬进设备内核，与 Python 复刻的 tilelang 参考对拍
**逐位 mismatch=0/4096**；4096/4096 个样本的低 16 位既非全 0 也非全 1（随机位有熵，非退化）。
`cvt.rs` 本身在 `sm_90a` ptxas 被拒（第 59 篇已实证），故**该指令仅分析**。

## 8. H100 实测汇总

`bash harness/exp_deepep_bucket_collectives.sh > harness/exp_deepep_bucket_collectives.out.txt`：
直接 include 真实头文件（`comm/bucket.cuh` + `common/{math,ptx,utils}.cuh` + `layout/bucket/*.cuh`）、
`-arch=sm_90a` 编译运行真实设备代码，再与独立 host oracle 逐项对拍，**PASS=30 / FAIL=0**：

| 组 | 内容 | 结果 |
|---|---|---|
| A | 8 个结构的 `sizeof`/`offsetof` + `BucketSignals`/`BucketWorkspace` 链式大小 + 16 MiB 预算 | 全对 |
| B1 | `ChunkIterator` 3 桶 chunk=64KB `kStride=4`：7 块恰好覆盖一次、每块 offset/bytes 一致 | 全对 |
| B2 | all_reduce rank-major 映射 `dst=ci%R, slot=(ci/R)%S, gen=ci/R/S` | 0 不符 |
| B3 | reduce_scatter slot-major 映射 `slot=ci%S, gen=ci/S`；与 B2 有 8191/8192 个不同 | 0 不符 |
| B4 | `get_head_ptr/get_tail_ptr` 字节偏移公式 | 0 不符 |
| C1 | 反序环累加×scale vs `np.float32` 同序 **逐位 0/16384**；升序朴素 922/4096 bit 不同 | 全对 |
| C2 | all_reduce 环求和×scale vs 同序逐位 0/4096；vs fp64 精确和 relerr 1.40e-07 | 全对 |
| D | bf16 哈希 vs tilelang 参考逐位 0/4096；随机位有熵；乘子与 DeepGEMM 同 | 全对 |
| E | host 配置算术（mm/broadcast warps、chunk KB、slots）与信号预算上限 | 全对 |

**未能测**（NCCL GIN 设备 API / `cvt.rs` 专属，仅源码分析）：`gin.put` / `red_add_rel_gpu` /
`multimem_ld_reduce_add` 的**真实跨 rank 行为**、`cvt.rs` 的真实舍入结果、以及 8 卡端到端性能
（需链接 NCCL≥2.32.3 的 PyTorch 环境，见路线图「阻塞」）。

## 9. 结论与边界

- `all_reduce` / `reduce_scatter` 与 all-gather 不同，**必须有 SM kernel**（要归约），但三者都按
  物理域三选一后端；纯 NVLink 腿用 `multimem.*` 硬件归约把 NVLink 流量降到 O(1)。
- 两个算子的 chunk→(rank,slot,generation) 映射**完全不同**：all_reduce 是 rank-major（R 个 peer 轮用
  一个 slot），reduce_scatter 是 slot-major。
- RDMA 路径的收发靠 head/tail **信用环**背压，且**最后一个用槽的 chunk 免信用**。
- 归约的 fp32 **求和顺序**（本地先行 + 反序环）与测试参考逐字一致，这正是 `rtol=0` 能过的原因。
- bf16 路径用**随机舍入**消除截断偏差，其哈希终端乘子与 DeepGEMM 的随机舍入**同源**。
- 本机（H100/sm90、NCCL 2.29.7）只能验证架构无关部分；真实多 rank 端到端待环境就绪。

## 10. 参考

- DeepEP 仓库：<https://github.com/deepseek-ai/DeepEP>（本地 `93eb6eb`，v2.5.0）
- DeepEP README `§Bucket collectives (experimental)`：`README.md:291-328`
- 本专题第 64 篇（P54-1，copy-engine 与 `ChunkIterator`）、第 59 篇（P49-1，`ptx.cuh` 公共层）、
  第 63 篇（P53-1，屏障与对称窗口句柄）、第 43 篇（P43-1，EPLB 梯度归约）
- 第 21/43 篇（DeepGEMM 随机舍入 `StochasticRoundToBF16` 与 `tests/utils.py` 的 oracle）
