# （五）Flux 的 GEMM + ReduceScatter：把 RS 塞进 epilogue

对应副本：[`flux/src/epilogue_evt_nvshmem.hpp`](../flux/src/epilogue_evt_nvshmem.hpp)、
[`flux/src/reduce_scatter_barrier_struct.hpp`](../flux/src/reduce_scatter_barrier_struct.hpp)、
[`flux/include/system_barrier.hpp`](../flux/include/system_barrier.hpp)。

上一篇是 AG-GEMM（数据进场时 overlap），这一篇是反方向：**GEMM 算完的结果要 reduce-scatter（RS）出去**。
Flux 的做法不是"GEMM 完了一个 kernel、再起一个 RS kernel"，而是**在 GEMM 的 epilogue 里直接做 RS**。

## 1. 为什么 RS 特别适合塞进 epilogue

RS 的形状是「同 shape 的 N 份数据按元素求和，再按 rank 切分，每 rank 只留 1/N」。
它和 GEMM 的关系是**逐 tile 的**：

```text
GEMM 产出 tile (m, n) → 这个 tile 需要和所有 peer 的 (m, n) 相加 → 相加后归属某个 rank
```

也就是说，**GEMM 每算完一个 tile，就有一份"可以立刻发走的部分和"**。
如果不这么做，就得等整个 GEMM 结束、把全部结果落盘，再启动 RS kernel 从头读一遍——
多了一次全量 HBM 往返，而且通信只能发生在最后。

Flux 的选择是：epilogue 一算完 tile，立刻把这份部分和推到目标 rank 的累积 buffer，
并用一个 **per-tile 计数**告诉对方"我这个 rank 的这份到了"。

## 2. 数据怎么走（读 `end_step` / `end_epilogue`）

这个 epilogue 是 CUTLASS 的 **EVT（epilogue visitor tree）**写法：一个 tile 的累加器被切成若干 `step`，
每一步调 `visit()` 做类型转换、`end_step()` 把这一步的结果写出去。Flux 在 `end_step` 里做了三件很"不 epilogue"的事：

```text
end_step:   bf16 结果 → 转成 fp32 → red.global.add.f32 累加到「本地目标 rank」的 fp32 buffer
end_epilogue:
    fence.acq_rel.sys
    atomicAdd_system(目标 rank 的 barrier[tile_idx], 1)     ← 告诉 owner：我这份加完了
    if (我是这个 tile 的本地 owner):
        wait_eq(barrier[tile_idx], local_world_size)         ← 同机所有 rank 都加完了
        fp32 buffer → bf16（recast_backto_bf16）
        nvshmemx_putmem_nbi_block(远端 owner 的 tile, 本地 bf16 结果)  ← 跨机只发一份
```

逐条解释为什么这么设计：

1. **同机用 `red.global.add.f32` 而不是先发再算**：同机是 NVLink 域，直接对 owner 的 buffer 做全局原子加，
   省掉一次"发过去 + owner 再读回来"的往返。代价是 buffer 要开成 fp32（注释里写着 "this need to double the size of the output buffer"）。
2. **跨机只在 owner 上发一次**：同机 N 个 rank 的贡献先合成一份，跨机只传这一份 —— 跨机带宽是最贵的资源，
   能少传 `local_world_size` 倍。
3. **`fence.acq_rel.sys` + `atomicAdd_system`**：`red.global` 是松弛的，必须用一条 sys-scope 的 fence 把它
   和后面那次计数绑在一起（release），owner 侧用 `wait_eq`（acquire）才算真的"看见了"那些加法。
4. **只有 owner 等**：其他 rank 加完 1 就走人，立刻回去算下一个 tile。等的人只有一个，而且它等的是
   "同机所有 rank"，不是整个 world。

## 3. 三段 flag：这个 RS 有层次

`reduce_scatter_barrier_struct.hpp` 里的 `PerTileFlags` 有三组：

```cpp
int epilogue;         // 本 rank 的 epilogue 写完了这个 tile
int reduce;           // 子世界（同机）内的 reduce 完成了
int reduce_sub_node;  // 跨子世界（跨机）的那一份也到齐了
```

对应上面那条链路的三步：

```text
epilogue  →  reduce（同机 global_red + owner 归约）  →  reduce_sub_node（跨机 put 到齐）
```

另外两个字段 `epilogue_queue` / `reduce_queue`（注释写着 "only valid for use_barrier = true"）
是给「按完成顺序排队」的工作队列用的——先完成的 tile 先进队，消费者按队列顺序处理，
避免"等某个慢 tile"把后面的全堵住。

### 为什么要 128 B 对齐 + padding

```cpp
char padding[128 - sizeof(ShapeOnly)];   // don't delete. for m < kM
```

每个 flag 独占 128 B（两倍 cacheline）。注释里 "for m < kM" 说的是：
当 M 小到 tile 很密时，相邻 tile 的 flag 会落在同一条 cacheline 上，
每次 `atomicAdd` 都会把整条 line 作废——**false sharing 会让这个"细粒度"设计直接退化**。
所以宁可浪费内存，也要一条 flag 一条 line。

同一个文件里还给了 SoA / AoS 两种访问器：

- `PerTileFlagsWrapper = PerTileFlagsAoSWrapper`：tile 连续，每个 tile 的结构体连着放；
- `PerRankFlagsWrapper = PerRankFlagsSoAWrapper`：同一个字段在所有 rank 上连着放。

AoS 适合"处理某个 tile 的全部字段"（epilogue），SoA 适合"所有 rank 的同一个字段一起扫"（跨 rank 归约）。
宏 `DEFINE_SOA_MEMBER` / `DEFINE_AOS_MEMBER` 就是同一份字段表的两种排布。

## 4. 与上一篇（AG-GEMM）对照

| | AG-GEMM | GEMM-RS |
|---|---|---|
| 通信方向 | 进场（我需要的数据） | 出场（我的结果） |
| 谁发起 | 另一条 stream 的 host 侧拷贝 | kernel 内的 epilogue（device 代码） |
| 同步原语 | `CUStreamWriteValue` + `wait_eq` | `nvshmem putmem` + `atomicAdd_system` + `wait_eq` |
| 粒度 | 一个 M-chunk | **一个 tile**（比 chunk 细，因为 epilogue 天然按 tile 产出） |
| 重叠对象 | 本 tile 的 GEMM 与后续 chunk 的拷贝 | 本 tile 的 RS 与后续 tile 的 MMA |

粒度更细的原因很直观：epilogue 的产出单位就是 tile，没有比它更自然的切法。

## 5. 代价与边界

1. **flag 数量 ∝ tile 数**，而且一条 flag 一条 cacheline —— 显存开销和 L2 压力都不小，
   所以它只适合"tile 数不太多"的常规 GEMM，不适合极小 tile。
2. **只能处理整 tile**：如果 GEMM 开了 split-K / stream-K，一个输出 tile 会被多个 CTA 写，
   需要额外的 `fixup` 归约（`gemm_v3_reduce_scatter.hpp` 里 `TileScheduler::fixup` 就是干这个的），
   复杂度立刻上一个台阶。
3. **RS 的目标 tile 必须已经分配**：这套设计假设"输出 buffer 一开始就是全尺寸"，
   如果要做动态 shape，flag 数组也得跟着动态分配。
4. **最后一次 `wait_eq` 仍是串行点**：最慢的 peer 决定这个 tile 何时能归约完；
   只是它被摊到了整个 GEMM 生命周期里，而不是集中在末尾。

## 6. 通算并行要点

- 【方向】RS 是"结果出场"，最适合放 epilogue；AG 是"数据进场"，最适合放独立 stream/producer。
- 【粒度】RS 的天然粒度是 **tile**（epilogue 产出单位），不是 chunk，也不是整层。
- 【两级】同机先 reduce、跨机再交换子世界部分和；三段 flag 分别守住这三步。
- 【防抖】flag 独占 cacheline，否则 `atomicAdd` 的 false sharing 会把细粒度设计打回原形。
- 【边界】split-K 会破坏"一个 tile 一个 owner"的假设，必须额外 fixup。

## 7. 坑

- `PerTileFlags` 里那些 `padding_*[8]` 是**逻辑的一部分**，不是格式对齐的装饰；删掉会静默变慢（甚至因
  false sharing 的伪共享导致计数错误可见性，取决于架构）。
- `atomicAdd_system` 是 sys-scope，比 `atomicAdd` 慢一个数量级；用它换的是跨卡可见性，
  不能图省事换成普通原子。
- `wait_eq` 的期望值是 **`local_world_size`（同机 rank 数）**，不是整个 world；
  把两级 RS 的语义写错成 `world_size` 就是直接死等。
- 这套设计里没有"用完复位"的机制（和 Flux 的 AG 一样），flag 的复位在 kernel 之外，
  跨调用复用 buffer 时必须显式清零。

## 8. 行号表（以当前副本为准，已核对）

`flux/include/system_barrier.hpp`：

| 内容 | 行号 |
|---|---|
| `SingleThreadSync`（不做同步的 Sync 策略） | 52 |
| `GenericSystemBarrier`（派生类重新声明 sys-scope 的原子） | 59 |
| `ld_acquire`：`ld.global.acquire.sys.b32` | 72 |
| `red_release`：`fence.acq_rel.sys` + `red.relaxed.sys.global.add` | 95 |

`flux/src/reduce_scatter_barrier_struct.hpp`：

| 内容 | 行号 |
|---|---|
| `PerTileFlags`（三段 per-tile flag + 128 B padding） | 68 |
| `PerRankFlags`（gemm_done / buffer_ready / copy_done / counter / remote_copy_done） | 118 |
| `BarrierWorkQeueuFlags`（按完成顺序排队的工作队列） | 138 |
| `PerTileFlagsWrapper = AoS` | 219 |
| `PerRankFlagsWrapper = SoA` | 263 |

`flux/src/epilogue_evt_nvshmem.hpp`：

| 内容 | 行号 |
|---|---|
| `recast_backto_bf16`（fp32 累加 → bf16 的原地压缩） | 323 |
| `end_step`：half 走 `global_red<half>` | 359 |
| `end_step`：bf16 先转 fp32 再 `global_red<float>`（所以 buffer 要双倍） | 374 |
| `copy_remote`：`nvshmemx_putmem_nbi_block` | 400 |
| `end_epilogue`：fence + `atomicAdd_system` | 413、426 |
| owner 侧 `wait_eq(..., local_world_size)`（全 kernel 唯一显式串行点） | 437 |
| owner 侧 recast + 跨机 put | 446 |

## 9. 照抄前该自己确认的四处

这几处不是"错"，是**证据不足**——上游行为依赖文件之外的上下文，或者看起来像是 历史遗留。移植/引用时逐条确认：

1. **`wait_eq` 计的是 `local_world_size`，不是 `world_size`**。上游这行自己带着 `// double check here`。
   写错不会崩，只会让 owner 在"还差几份没到"的情况下开始转发 —— 读到没归约完的数据。
2. **`after_inc` 取了就丢**：用的是 `atomicAdd_system` 而不是 `red`，白等一次返回值。
   看起来像是从"需要判断我是不是最后一名"的版本改过来的；真要省，换成 red 更快。
3. **`fence.acq_rel.sys` 在 `atomicAdd_system` 之前，但之前没有 `__syncthreads`**。
   也就是说「所有线程的 `red.global` 都已完成」这件事，靠的是 epilogue 框架在 `end_step` 与 `end_epilogue`
   之间插过同步（本文件看不到）。照抄这段时，这是第一处要确认的地方。
4. **`nnodes`、`coord_per_rank_local` 传进来（或算出来）却没用到**：读的时候不要为它们编一个"必有深意"的解释。

> 这类"看起来多余"的代码，在通信-计算 overlap 的实现里很常见：版本演进的残留 + 为另一条路径预留的参数。
> 判断标准只有一个——**它是否在当前的同步链上承担了某个不变量**。承担不了，就当注释里的疑点记下来。
