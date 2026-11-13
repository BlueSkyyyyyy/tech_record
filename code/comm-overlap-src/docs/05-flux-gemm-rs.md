# （五）Flux 的 GEMM + ReduceScatter：把 RS 塞进 epilogue

**Flux 把 ReduceScatter 从「GEMM 完再起一个 kernel」换成「epilogue 里按 tile 就地归约」，代价是每个 tile 一条 flag 加一次 sys 级原子**。

读完你会知道：

- 为什么 RS 的等待能按 tile 摊开，不必堆在 GEMM 末尾
- 三段 flag 各自守住哪一步
- 一个 tile 从累加器到远端 owner 的完整路径

## 1. RS 的产物要等整段 K 才算完，所以它只能藏在 epilogue 里

一句人话：AG 是「我要的数据还没到」，RS 是「我算出的结果还没人要」。前者能提前开工，后者只能等算完。

但不必等整个 GEMM 算完：一个输出 tile 的 K 方向累加结束，这一块就可以发走了。

方向图三行：

```text
GEMM 算 tile(m,n) → 部分和 red 进「这段输出 owner」的 fp32 buffer（同机所有 rank 都往这加）
                 → owner 等同机到齐 → 压回 bf16
                 → owner put 一份到远端节点
```

对照做法是等 GEMM 全部结束、结果落 HBM，再起一个 RS kernel 从头读一遍：多一次全量 HBM 往返，通信还被压到最后一刻。

## 2. 什么时候能发：`end_step` 攒，`end_epilogue` 才发

这个 epilogue 是 CUTLASS 的 EVT（epilogue visitor tree）：一个 tile 的累加器切成若干 step，`visit()` 做类型转换，`end_step()` 把这一步写出去。看两个钩子各做什么：

```text
end_step:  bf16 结果 → fp32 → red.global.add.f32 累加到本地目标 rank 的 fp32 buffer
end_epilogue:
    fence.acq_rel.sys
    atomicAdd_system(owner 的 barrier[tile_idx], 1)    ← 我这份加完了
    if (我是这个 tile 的本地 owner):
        wait_eq(barrier[tile_idx], local_world_size)    ← 同机所有 rank 都加完了
        fp32 buffer → bf16（recast_backto_bf16）
        nvshmemx_putmem_nbi_block(远端 owner 的 tile)    ← 跨机只发一份
```

「什么时候能发」的答案就在这里：每个 step 结束都 red 一次，但跨机 put 要等整个 tile 的 step 都结束。发得早等于发多次、多份流量。

所以呢：同机是 NVLink 域，用 `red.global.add.f32` 直接加进 owner 的 buffer，省掉「发过去再读回来」的往返。代价是 buffer 得开成 fp32（源码注释：this need to double the size of the output buffer）。

跨机那一跳只在 owner 上发一次：同机 `local_world_size` 份先合成一份，跨机流量只剩 1/`local_world_size`。

为什么「计数到了」就等于「数据到了」：`red.global` 是松弛的，计数前那条 sys 域 `fence.acq_rel.sys` 把数据一起 release 出去，owner 侧的 `wait_eq` 是 acquire 读。少了 fence，peer 会先看到计数加一，再去读还没到的部分和。

非 owner 加完 1 就走，回去算下一个 tile；等的人只有一个，且它等的是同机，不是整个 world。

## 3. 三段 flag：这个 RS 是分层的

`PerTileFlags` 一个 tile 一组，三组字段对应链路三步：

```cpp
int epilogue;         // 本 rank 的 epilogue 写完了这个 tile
int reduce;           // 子世界（同机）内的 reduce 完成了
int reduce_sub_node;  // 跨子世界（跨机）的那一份也到齐了
```

三层各在等什么：

1. `epilogue` 等本 rank 的这份部分和写完，每个 tile 各记一笔，允许乱序。
2. `reduce` 等同机所有 rank 都加进 owner 的 buffer，这是 owner 唯一要等的。
3. `reduce_sub_node` 等跨机的那一份到齐，也就是下一跳的输入。

`epilogue_queue` / `reduce_queue` 只在 use_barrier = true 时有效，作用是按完成顺序排队：先完成的 tile 先进队，消费者照队列顺序取，避免被一个慢 tile 全堵住。

### 一条 flag 一条 cacheline，这是逻辑不是对齐

```cpp
char padding[128 - sizeof(ShapeOnly)];   // don't delete. for m < kM
```

每个 flag 独占 128 B（两倍 cacheline）。M 小到 tile 很密时，相邻 tile 的 flag 会落到同一条 cacheline，每次 atomicAdd 都把整条 line 作废——false sharing 会让这套细粒度设计直接退化。宁可浪费内存，也要一条 flag 一条 line。

同一个文件还给了 AoS / SoA 两种访问器：`PerTileFlagsWrapper` 是 AoS，一个 tile 的全部字段连着放；`PerRankFlagsWrapper` 是 SoA，同一字段在所有 rank 上连着放。前者服务「处理某个 tile 的全部字段」，后者服务「所有 rank 的同一个字段一起扫」。

## 4. 同一套切法，方向反过来、粒度细一档

两个方案都在等「更小的一份东西」：

| | AG-GEMM | GEMM-RS |
|---|---|---|
| 方向 | 进场：我需要的数据 | 出场：我的结果 |
| 谁发起 | 另一条 stream 的 host 拷贝 | kernel 里的 epilogue |
| 粒度 | 一个 M-chunk | 一个 tile |
| 同步 | `CUStreamWriteValue` + `wait_eq` | `putmem` + `atomicAdd_system` + `wait_eq` |

粒度细不是选的，是 epilogue 的产出单位本来就是 tile。于是重叠的对象成了本 tile 的 RS 与后续 tile 的 MMA，进场那半见 [AG-GEMM](04-flux-ag-gemm.md)。

## 5. 代价卡在 flag 数量和 tile 粒度上

- flag 数量 ∝ tile 数，且一条 flag 一条 cacheline：显存和 L2 压力都不小，只适合 tile 数不多的常规 GEMM。
- 只能处理整 tile：开了 split-K / stream-K，一个输出 tile 会被多个 CTA 写，得额外做 `fixup` 归约（`TileScheduler::fixup`），复杂度上一个台阶。
- 目标 tile 必须已分配：这套假设输出 buffer 一开始就是全尺寸，动态 shape 得连 flag 数组一起动态分配。
- 最后一次 `wait_eq` 仍是串行点：最慢的 peer 决定这个 tile 何时归约完，只是它被摊到整个 GEMM 生命周期里。
- owner 侧 `atomicAdd_system` 的返回值取了就丢，白等一次响应；不需要「我是不是最后一名」时用 `red` 更便宜。

## 6. 坑：只留会静默出错的

- 三组 `padding_*[8]` 是逻辑的一部分，不是格式装饰。删掉后相邻 flag 共享 cacheline，false sharing 会静默吃掉细粒度换来的收益，计数可见性还取决于架构。
- `wait_eq` 的期望值是 `local_world_size`，不是 `world_size`（上游这行自己标了 double check here）。写错不崩：owner 在还差几份没到时就开始转发，读到没归约完的数；真按 `world_size` 等就是死等。
- 两处同步不在这个文件里：`fence.acq_rel.sys` 之前没有 `__syncthreads`，靠的是 epilogue 框架在 `end_step` 与 `end_epilogue` 之间插过同步；flag 的清零也在 kernel 之外，跨调用复用 buffer 不显式清零就会读到上一次的值。

副本（带注释）：[`epilogue_evt_nvshmem.hpp`](../flux/src/epilogue_evt_nvshmem.hpp)、
[`reduce_scatter_barrier_struct.hpp`](../flux/src/reduce_scatter_barrier_struct.hpp)、
[`system_barrier.hpp`](../flux/include/system_barrier.hpp)。
