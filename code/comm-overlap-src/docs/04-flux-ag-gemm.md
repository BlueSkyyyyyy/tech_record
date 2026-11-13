# （四）Flux 的 AllGather + GEMM：AG 不在这个 kernel 里，只在等一个 flag

**把 AllGather 挪出 GEMM kernel，交给另一条 stream；每个 tile 只等自己那几行数据的 flag，代价是等待粒度被锁死在拷贝单位上**。

读完你会知道：
- AG 不在这个 kernel 里，那它在等什么；
- `world_size × SPLIT` 份 flag 是谁置的、谁等的；
- 它和 UniEP 差在哪，哪三处静默出错。

副本：[kernel](../flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp)、[scheduler](../flux/src/sm90_all_gather_gemm_tile_scheduler.hpp)、[`system_barrier.hpp`](../flux/include/system_barrier.hpp)。

## 1. AG 不在这个 kernel 里，那这个 kernel 在等什么

`Sm90AGGemmUniversal` 只做 GEMM；AllGather 是 host 侧在另一条 stream（`cp_streams[0]`）上用 `cudaMemcpyAsync` 搬完的，两条 stream 之间没有 event 依赖。

```text
input_buffer = [rank0 的 chunk | rank1 的 chunk | ... | rank_{n-1} 的 chunk]   ← 对称内存
                 ↑ 每个 rank 把自己的那份拷到自己的位置（local copy）
                 ↑ 然后 All2All / ring：从对端读回别人的 chunk（P2P read）
```

看什么：每搬完一个 `(segment, split)`，往自己的 barrier 写 1。

```cpp
CUStreamWriteValue(stream, barrier_ptrs[rank] + (segment * SPLIT + split), 1, ...);
```

所以呢：【结构】拷贝引擎在搬、GEMM 的 SM 在算，两者只通过内存里的 flag 耦合，kernel 内没有协作。

`CUStreamWriteValue` 是 stream 操作，天然排在 `cudaMemcpyAsync` 之后，不需要事件同步；写的是**本 rank** 的 barrier，即"我这份到了"。

## 2. 每个 tile 只等自己那几行数据

看什么：`data_chunk_id_start` 那几行整数运算。

```cpp
const int data_chunk_id_start = work_tile_info.M_idx * params.TILE_SIZE_M / params.m_per_data_chunk;
int data_chunk_id_end = (work_tile_info.M_idx * params.TILE_SIZE_M + params.TILE_SIZE_M - 1) / params.m_per_data_chunk;
data_chunk_id_end = (data_chunk_id_end < params.n_data_chunks - 1) ? data_chunk_id_end : (params.n_data_chunks - 1);   // 最后一块要 clamp
```

所以呢：【粒度】chunk 号就是 flag 下标：`n_data_chunks = world_size × SPLIT`（`SPLIT` 是编译期宏，默认 1），`m_per_data_chunk = M / n_data_chunks`。tile 覆盖 `TILE_SIZE_M` 行，可能跨 chunk，所以等的是 `[start, end]` 区间。

看什么：入口和主循环各等一次，原语不同。

```cpp
// ① 入口：整 CTA 等一次（块级）
for (int id = data_chunk_id_start; id <= data_chunk_id_end; ++id)
    SystemBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);
// ② 换 tile：warp 级补等
for (int id = new_data_chunk_id_start; id <= new_data_chunk_id_end; ++id)
    WarpBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);
```

所以呢：循环内不能回块级同步——consumer 的两个 warp group 走另一个 `if` 分支，`__syncthreads` 会让 producer 等一个永不到达的线程组。

另外，循环里比的是入口算出的区间（两个变量从不更新）：只有同区间的 tile 走快路径，其余每次重等。flag 单调，重等只是一次 acquire load，不会错，想省该滚动更新。

### 调度器把第一个 tile 换成你本地那份

看什么：`Sm90AGKernelTileScheduler::get_current_work` 会重映射 `M_idx`，单机分支只有一句。

```cpp
new_M_idx = (M_idx + problem_blocks_m_offset) % problem_blocks_m;
```

所以呢：`problem_blocks_m_offset` 是本地段的起始 tile 号，`M_idx=0` 因此落在本地 chunk 上，flag 在 `copy_local()` 后已置 1，第一个 tile 不用等。跨机按 node 打散（`nnodes > 1` 查 `nodes_wizzle`），免得所有 node 同抢一段远端数据。

【藏通信】这是把 $t^{need}$ 往后推：数据晚到不要紧，先算已到的。

## 3. 一个 chunk 一个 flag：谁置、谁等

`GenericSystemBarrier` 有四个原语（`arrive_inc` / `wait_eq` / `wait_lt` / `wait_eq_reset`），GEMM 只用 `wait_eq`；置的一方走 host 的 stream 写。

看什么：两个方向的内联 PTX。

```cpp
ld.global.acquire.sys.b32                        // 等的一方：acquire 读
fence.acq_rel.sys + red.relaxed.sys.global.add   // 到的一方：release 累加
```

所以呢：【原语】跨卡可见性全靠 `.sys`，`.gpu` 在多卡 P2P 上不保证；代价是慢，所以 UniEP 在 intranode 用 `.gpu`，Flux 必须 `.sys`。

【份数】为什么不是一份：一份只能表示"整轮 AG 到齐"，GEMM 得等所有人；拆开后 flag 下标 = M 行号 ÷ `m_per_data_chunk`，每个 tile 只碰自己那几个格子。

两条细节：只让 thread 0 自旋，其余在 `Sync::sync()`（`__syncthreads` / `bar.sync`）上等——32 个线程打同一地址只制造 L2 热点；`wait_eq` 精确等于、`wait_lt` 至少到，前者配一次性置 1 的 flag，后者配计数型 barrier。

## 4. 同一问题，两种切法

- 谁搬、谁置谁等：UniEP 是同 kernel 的 dispatch warp 发 TMA、`release` 原子置，GEMM 的 TMA warp 等；Flux 是另一条 stream 的 DMA / P2P memcpy，host 侧 `CUStreamWriteValue` 置，producer warp 等。
- 等多大：UniEP 一个 expert 一次；Flux 一个 M-chunk，入口一次加每 tile 一次轻量检查。
- 为什么不能更细：UniEP 更细没有调度价值；Flux 更细是多一堆 flag 和 `CUStreamWriteValue`，小 chunk 的 DMA 效率也崩。
- 【切法】前者按模型结构分组，后者按拷贝引擎的搬运单位分片。

## 5. 这样切丢掉了什么

1. 确定性：tile 顺序随 SM 数和调度变化，浮点归约顺序跟着变。AG-GEMM 的 K 方向完整，每个元素只写一次；开 stream-K / split-K 就不保证。
2. 耦合度低：拷贝引擎在搬、GEMM 全时在算，不抢 SM；但抢 HBM/L2，且 kernel 内无法精细调度。
3. 小消息不友好：`cudaMemcpyAsync` 有启动开销，chunk 切太细更慢，`SPLIT` 默认为 1 就为此。【代价】flag 数量 ∝ `world_size × SPLIT`。

## 6. 三个会静默出错的坑

- 重置时机：`forward` 结尾要 `reset_signals()`（等 `cp_event` + 跨 rank 同步 + 清零），漏掉就第二次调用读到上次的 1，静默错。
- chunk 编号必须与 M 顺序对齐：`barrier[segment*SPLIT + split]` 和 GEMM 侧 `M_idx*TILE_SIZE_M/m_per_data_chunk` 都按 M 顺序，一边按 (rank, split) 排就错位，错但不报错。
- `m_per_data_chunk` 是整数除法，没有整除检查（`gather()` 只查 `chunk_size`）：M 不能被 `world_size × SPLIT` 整除时边界对不齐，只能靠多等一个 chunk 兜住。

## 7. 如果只记一件事

Flux 的 overlap 是两条 stream 的错位，不是 kernel 内分工；GEMM 侧只做一件事：把等整轮换成等自己这个 chunk。

下一篇反过来：GEMM 算完不落地，直接在 epilogue 里归约散射，[Flux 的 GEMM-RS](05-flux-gemm-rs.md)。
