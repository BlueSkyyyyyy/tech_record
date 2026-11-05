# （四）Flux 的 AllGather + GEMM：两条 stream + 按 chunk 等 flag

对应副本：[`flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp`](../flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp)、
[`flux/src/sm90_all_gather_gemm_tile_scheduler.hpp`](../flux/src/sm90_all_gather_gemm_tile_scheduler.hpp)、
[`flux/include/system_barrier.hpp`](../flux/include/system_barrier.hpp)。

前三篇是 UniEP（EP、SM 内角色分工）。从这篇开始换到 Flux（TP、多 stream）。
两份代码解决的问题不同，但"把等待切小"的手法可以直接对照。

## 1. Flux 的 AG 根本不在这个 kernel 里

这是读 Flux 时第一个反直觉的点。`Sm90AGGemmUniversal` 这个 kernel 只做 GEMM，
AllGather 是 host 侧在**另一条 stream**（`cp_streams[0]`）上用 `cudaMemcpyAsync` 做完的：

```text
input_buffer = [rank0 的 chunk | rank1 的 chunk | ... | rank_{n-1} 的 chunk]   ← 对称内存
                 ↑ 每个 rank 把自己的那份拷到自己的位置（local copy）
                 ↑ 然后 All2All / ring：从对端读回别人的 chunk（P2P read）
```

每完成一个 `(segment, split)`，就往**自己的** `barrier_buffer[segment*SPLIT + split]` 写 1：

```cpp
CUStreamWriteValue(stream, barrier_ptrs[rank] + (segment * SPLIT + split), 1, ...);
```

注意两点：

- `barrier_ptrs[rank]` 是**本 rank** 的 barrier（不是对端的）；写的是"我本地这份数据到了"。
- 写 flag 用的是 `CUStreamWriteValue`——**把一次内存写变成一个 stream 操作**，
  于是它天然排在前面那些 `cudaMemcpyAsync` 之后，不需要额外的事件同步。

所以 Flux 的 overlap 是：**拷贝引擎/别的 SM 在搬数据，GEMM 的 SM 在算；两者只通过内存里的 flag 耦合。**

## 2. GEMM 侧：只等自己这个 tile 需要的 chunk

搜 `data_chunk_id_start`：

```cpp
const int data_chunk_id_start = work_tile_info.M_idx * params.TILE_SIZE_M / params.m_per_data_chunk;
int data_chunk_id_end = (work_tile_info.M_idx * params.TILE_SIZE_M + params.TILE_SIZE_M - 1)
                        / params.m_per_data_chunk;
data_chunk_id_end = (data_chunk_id_end < params.n_data_chunks - 1)
                        ? data_chunk_id_end : (params.n_data_chunks - 1);   // 最后一块要 clamp
```

- `n_data_chunks = world_size × SPLIT`（`SPLIT` 是编译期宏，默认为 1）；
- `m_per_data_chunk = M / n_data_chunks`；
- 一个 GEMM tile 覆盖 `TILE_SIZE_M` 行，可能跨 chunk 边界，所以取 `[start, end]` 区间。

然后 producer warp 在发第一批 TMA load **之前**等这些 flag。注意这里有**两处**等待，用的不是同一个原语：

```cpp
// ① 入口：整 CTA 等一次（SystemBarrier = 块级同步，自旋 + __syncthreads）
for (int id = data_chunk_id_start; id <= data_chunk_id_end; ++id)
    SystemBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);

// ② 循环内：每换一个 tile，用 warp 级 barrier 补等（搜 WarpBarrier）
for (int id = new_data_chunk_id_start; id <= new_data_chunk_id_end; ++id)
    WarpBarrier::wait_eq(params.ptr_barrier, thread_idx, id, 1);
```

为什么循环内不能继续用块级同步？因为 **consumer 的两个 warp group 走的是另一个 `if` 分支**，
它们根本没进这段代码——块级的 `__syncthreads` 会把 producer 永远卡在等一个不会到达的线程组上（死锁）。
warp 级 barrier 只同步这一个 warp，恰好够用。

还有一个容易看漏的细节：循环内比较的对象是 **入口时算出的区间**（`data_chunk_id_start/end` 在循环里从不更新），
所以"快路径"（完全跳过等待）只对**与入口 tile 同区间**的 tile 生效；其余 tile 每次都要重等一遍。
这不会错——flag 是单调的（置 1 后不复位），重复 `wait_eq` 只是一次 acquire load；
但它也不是"最优"：真想省，应该在循环里滚动更新 `data_chunk_id_start/end`。

### 为什么这样就能"藏住"通信

两个原因叠在一起：

1. **调度器把第一个 tile 换成本地数据**。`Sm90AGKernelTileScheduler::get_current_work` 里：
   ```cpp
   new_M_idx = (M_idx + problem_blocks_m_offset) % problem_blocks_m;
   ```
   `problem_blocks_m_offset` 是「本 rank 那份数据所在 M 块」的 tile 编号，于是 `M_idx=0` 的 tile
   正好落在本地 chunk 上。本地 chunk 的 flag 在 `copy_local()` 之后立刻被置 1（在我们的 kernel 启动之前），
   所以**第一个 tile 不用等**。
2. **跨机时按 node 打散**。`nnodes > 1` 走 `nodes_wizzle` 表，让不同 node 不要同时去抢同一段远端数据
   （否则所有 node 的第一批 tile 都是同一个 chunk，网络瞬间打满再空转）。

这就是"把 $t^{need}$ 往后推"的教科书做法：数据到得晚不要紧，只要**先算已经到的那部分**。

## 3. flag 协议（`system_barrier.hpp`）

`GenericSystemBarrier` 提供了四个原语，GEMM 只用 `wait_eq`，AG 侧用 stream 写（不经过 device 代码）：

```cpp
ld.global.acquire.sys.b32      // 等的一方：acquire 读
fence.acq_rel.sys + red.relaxed.sys.global.add   // 到的一方：release 累加
```

三个关键点：

1. **必须是 `.sys` scope**：跨卡（甚至跨机）可见性，`.gpu` scope 在多卡 P2P 上不保证。
   代价是比 `.gpu` 慢——所以 UniEP 在 intranode 场景用 `.gpu`，Flux 这里必须 `.sys`。
2. **只让 thread 0 自旋**，其余线程在 `Sync::sync()`（`__syncthreads` 或 `bar.sync`）上等。
   32 个线程打同一个地址只会制造 L2 热点。
3. `wait_eq` 是"精确等于"，`wait_lt` 是"至少到"。前者适合「一次性置 1 的 flag」，
   后者适合「计数型 barrier」（多个 rank 各 arrive 一次，等计数到齐）。

## 4. 与 UniEP 对照：同一个问题的两种切法

| | UniEP（dispatch + GEMM） | Flux（AG + GEMM） |
|---|---|---|
| 通信谁做 | 同 kernel 的 dispatch warp（TMA） | 另一条 stream 的 DMA / P2P memcpy |
| flag 写者 | dispatch warp（device 代码，`release` 原子） | host 侧 `CUStreamWriteValue`（stream 操作） |
| 等的人 | GEMM 的 TMA warp | GEMM 的 producer warp |
| 粒度 | 一个 expert | 一个 M-chunk |
| 等几次 | 每个 expert 一次（M-contiguous 保证单调） | 入口等一次 + 每 tile 一次轻量检查（快路径只对与入口同区间的 tile 生效） |
| 为什么不能更细 | 更细没有调度价值（GEMM 必须按 expert 整块算） | 更细 = 更多 flag + 更多 `CUStreamWriteValue`，小 chunk 的 DMA 效率也崩 |

一句话：**UniEP 用"结构性分组"（expert）做粒度，Flux 用"数据分片"（M-chunk）做粒度。**
前者由模型结构决定，后者由拷贝引擎的搬运单位决定。

## 5. 这样做丢了什么

1. **确定性**：tile 的执行顺序随 SM 数、调度变化，浮点归约顺序跟着变。AG-GEMM 里 K 方向是完整的，
   所以每个输出元素仍只写一次；但如果开了 stream-K / split-K，就不保证了。
2. **通信与计算的耦合度低**：AG 用的是拷贝引擎或另一批 SM，GEMM 的 SM 全时在算——
   这既是优点（不抢 SM），也是缺点（拷贝和 GEMM 抢 HBM/L2 带宽，且无法在 kernel 内做精细调度）。
3. **小消息不友好**：`cudaMemcpyAsync` 每次都有启动开销，chunk 切太细反而更慢——这就是 `SPLIT` 默认为 1 的原因。

## 6. 通算并行要点

- 【结构】通信在另一条 stream、计算在 GEMM kernel，两者只通过内存 flag 耦合，没有 kernel 内协作。
- 【粒度】chunk = `M/(world_size×SPLIT)` 行；一个 tile 可能跨多个 chunk，所以等的是**区间**。
- 【藏通信】靠调度器把第一个 tile 换成本地数据、跨机再按 node 打散；不是靠"通信更快"。
- 【原语】`CUStreamWriteValue` 把内存写变成 stream 序；`ld.global.acquire.sys` 保证跨卡可见。
- 【代价】flag 数量 ∝ `world_size×SPLIT`；chunk 太细会输给 DMA 启动开销。

## 7. 坑

- **chunk 编号必须与 M 顺序对齐**：`barrier[segment*SPLIT + split]` 的下标是"M 顺序"的，
  GEMM 侧用 `M_idx*TILE_SIZE_M/m_per_data_chunk` 算出来的也是 M 顺序。两边一旦有一边按 (rank, split) 排就错位。
- **最后一个 chunk 要 clamp**：`data_chunk_id_end` 不减到 `n_data_chunks-1` 会越界读 flag。
- **重置时机**：Flux 在每次 `forward` 结尾 `reset_signals()`（等 `cp_event` + 跨 rank 同步 + 清零）；
  漏掉就会第二次调用直接读上一次的 1。
- **`m_per_data_chunk` 是整数除法且没有显式整除检查**（`gather()` 只检查了 `chunk_size`）。
  M 不能被 `world_size×SPLIT` 整除时，chunk 边界和 tile 边界对不齐，只能靠"多等一个 chunk"兜住。

## 8. 行号表（副本 `flux/src/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp` / `..._tile_scheduler.hpp`，已核对）

| 内容 | 行号 |
|---|---|
| chunk 区间计算（`M_idx` → `[start, end]`，end 要 clamp） | kernel 668–674 |
| 入口等 flag（`SystemBarrier`，块级，全 CTA 一次） | kernel 715–721 |
| 循环内等 flag（`WarpBarrier`，warp 级） | kernel 767–781 |
| 被作者注释掉的旧粗粒度写法 | kernel 746–747 |
| `LoadWarpOrderBarrier`：声明 / `group_id` / `arrive` / `wait` | kernel 226 / 601 / 814 / 840 |
| scheduler：`M_start` / `problem_blocks_m_offset` | scheduler 156–157 |
| scheduler：`get_current_work` 的三分支 M 重映射 | scheduler 200–232 |
| kernel 取起始 tile / 每次 `fetch_next_work` 都重映射 | kernel 643 / 978–990 |
