# KDA 前向 K2（recurrence）精读：上半场看谁在干活，下半场看每个 Phase 算什么

> 这是 KDA 专题的第 3 篇。K2 是前向里真正串行的那一半：一个 CTA 从序列头走到尾，逐块更新
> $128\times128$ 的状态。**上半场**看谁在干活（3 种 warp 角色 + 两级流水线），
> **下半场**看每个 Phase 算什么（六步数据流，以及 `MOVM_T` 省掉的每一次 smem 往返）。
> 最后收在两个坑上：状态为什么敢用 bf16 存，$E^{\text{tot}}$ 为什么必须逐块搬进流水线。

## 0. 先记住一个画面

把 K2 想成一条**只能单车道走的装配线**。

整条序列是传送带上一盒一盒的料，一盒 16 个 token，必须按顺序过；盒子之间不能超车，
因为每一盒都要用到上一盒改过的那块公共黑板。

车间里有三个工种：

- **搬运工（LOAD）**：提前把后面 3 个工位的零件摆好，摆满就等；
- **4 个技工（MMA）**：并排站着，每人只管成品的 32 列（两个 16 列的块），算完一盒，
  在公共黑板上改自己那 32 列；
- **发货员（STORE）**：把做好的成品装车，落后 2 个工位慢慢发。

每加工完一盒，4 个技工必须**对一次表**。因为下一盒一开工，每个人都要读整块黑板的全部 128 列，
而黑板刚刚被四个人各改了一部分——谁先跑，谁就会读到别人还没写完的旧值。

> **类比的边界**：传送带上的盒子不会被覆盖，GPU 上的 stage 是**环形缓冲**，第 `t%3` 号 stage
> 会被第 `t+3` 个 tile 盖掉，所以"提前"必须靠硬件 barrier 计数兜底，不能靠人记。黑板也不是
> "抄一份"：`s_acc` 是 $128\times128$ 的 **bf16**，就地读写，跨 warp 可见性靠 smem + fence。
> 最后一处差别最重要：技工算出的结果**不是一个盒子**，它热在寄存器里；想让它当下一步的输入，
> 还得在寄存器内做一次转置（§11 的 `MOVM_T`）。

读完这一篇，你应该能回答：

1. 192 个线程分成 4+1+1，谁在等谁、谁把数据递给谁？
2. 六个 Phase 里，哪些中间结果可以一直留在寄存器里，哪一步非写 smem 不可？
3. 状态为什么敢用 bf16 存、更新又为什么必须 fp32 FMA？每个 chunk 的 $E^{\text{tot}}$
   为什么不能只算一次？

## 1. 一页看懂：谁在干活、每个 Phase 算什么

### 1.1 谁在干活

一个 CTA 192 个线程，按下标**硬切**成三段：0–127 是 4 个 MMA warp，128–159 是 LOAD warp，
160–191 是 STORE warp（`fwd_launch.cu:186` 的 `kK2Threads = 32*2 + 128`）。

| 角色 | warp | 线程 | 职责 | 出处 |
|---|---|---|---|---|
| **MMA** | 0–3 | 128 | 六个 Phase：块内递推、输出投影、状态更新；输入流水的消费者、输出流水的生产者 | `fwd_kernel2.cuh:191–192,426` |
| **LOAD_QKG** | 4 | 32 | 初态载入 + 每 tile 8 个 TMA load；输入流水的生产者 | `fwd_kernel2.cuh:193–194,240–323` |
| **STORE** | 5 | 32 | 每 tile 把 out TMA 出去 + 末态写回；输出流水的消费者 | `fwd_kernel2.cuh:195–196,746–802` |

一句话记：**4 个算、1 个喂、1 个发货。**

### 1.2 六个 Phase

MMA warp 的主循环体里，代码用注释标了 `Phase 1..6`（`fwd_kernel2.cuh:533,570,589,625,652,659`）。
它们的依赖基本是一条直线，只有两处是"留给下一块用"的：

| Phase | 公式 | 代码位置 | 依赖的上一步 |
|---|---|---|---|
| 1 双 GEMM | $u=k^{\text{dec}}S,\ o=q^{\text{dec}}S$ | `533–568` | 本 tile 的 $k^{\text{dec}},q^{\text{dec}}$ + 上一块 Phase 6 留下的 `s_acc` |
| 2 降精度 + 备料 | $\text{out}\leftarrow\text{bf16}(\text{out})$，载入 $v,P,\beta$ | `570–587` | Phase 1 的 `out_acc` |
| 3 修正并求 $U$ | $\tilde v=(v-u)\beta,\ U=P\tilde v$ | `589–623` | Phase 2 的 `v_bf16`、$P$、$\beta$ |
| 4 输出投影 | $o\leftarrow o+M_{qk}U$ | `625–650` | Phase 3 的 $U$ |
| 5 写回 | `out_bf16` → smem | `652–657` | Phase 4 的 `out_bf16` |
| 6 状态更新 | $S^\top\leftarrow S^\top\odot E^{\text{tot}}+U^\top k^{\text{rest}}$ | `659–731` | Phase 4 转置好的 $U$ + 本块 $k^{\text{rest}}$ |

（代码位置都在 `fwd_kernel2.cuh`。）

三条一眼能看出来的事实：

- **只有 Phase 6 会改 `s_acc`**，所以它是唯一需要跨 warp 会合的地方（§5）。
- **只有 Phase 5 写 smem 给别的角色看**，其余中间量都在寄存器里转（§15）。
- $U$ 一个量服务两路下游 GEMM（Phase 4 的 $M_{qk}U$、Phase 6 的 $k^{\text{rest}\top}U$），
  **一次 smem 都没落**。

## 上半场：谁在干活

## 2. K2 的边界：一个 CTA 串完一个 head

**K2 只做有串行依赖的那部分。**

K1（第 2 篇）已经把每块 16 个 token 预处理成 workspace 中间件；K2 拿着一块一块的中间件，
沿序列做 gated delta-rule 递推，用跑动状态 $S\in\mathbb{R}^{K\times V}$ 把输出 $o_t=q_tS_t$
投影出来，并在每块末更新状态。

状态在块与块之间串行传递，所以 K2 只能沿 **head 轴**并行：一个 CTA 负责一个 `(seq, head)`
的全部 tile。

| 项 | 值 | 出处 |
|---|---|---|
| kernel 入口 | `_flash_kda_fwd_recurrence` | `fwd_kernel2.cuh:133` |
| grid | `dim3(N, H)`（seq × head，一个 CTA 串一个序列的一个 head） | `fwd_launch.cu:203` |
| block | `192` 线程 = 4 MMA + 1 LOAD + 1 STORE | `fwd_launch.cu:186,204` |
| 输入流水 | `InputStages = 3`（v/beta/workspace 共 8 个 TMA descriptor） | `fwd_launch.cu:29` |
| 输出流水 | `OutputStages = 2`（out） | `fwd_launch.cu:30` |
| 动态 smem | `sizeof(SharedStorageK2<K2L,3,2>)` = **98432 B（96.1 KB）** | §7.1 |
| 模板参数 | `<…, CHUNK, D, InputStages, OutputStages, NumThreads, HasStateIn, HasStateOut, StateFP32, IsVarlen>` | `fwd_kernel2.cuh:115–132` |

**为什么非要拆角色？** 和 K1 对比一下就清楚了。

K1 是"单次 TMA load、无 pipeline"的短 kernel（`docs/02` §0.1），一个 CTA 只干一块活，
等 TMA 的时候切走别的 CTA 就行——靠**高占用率**（8 CTA/SM）掩盖延迟。

K2 反过来：一个 CTA 要串完 `t_tiles` 个 tile。如果还是"所有线程一起 load、一起算、一起存"，
每个 tile 都会被 TMA 延迟和 `__syncthreads()` 的全局栅栏卡住。

于是把角色拆开：LOAD 专职喂、STORE 专职发、4 个 MMA 专职算，三者用硬件 barrier 解耦。
算力不和访存抢线程，也不被全局栅栏打断。

| 维度 | K1 `_flash_kda_fwd_prepare` | K2 `_flash_kda_fwd_recurrence` |
|---|---|---|
| 并行轴 | token（`grid=(tiles,H)`） | head（`grid=(N,H)`，CTA 内串 tile） |
| CTA | 256 线程，全部同角色 | 192 线程，**3 种角色** |
| TMA | 单次 load、无 pipeline | 输入 3-stage / 输出 2-stage |
| 状态 | 无（不读状态） | 读初态、写末态（独立 barrier） |
| 同步 | `__syncthreads` | pipeline barrier + `NamedBarrier(128)` |
| 寄存器（实测） | `REG:32`（压满 8 CTA/SM 上限） | `REG:74–80`（无 `__launch_bounds__` 下限） |
| 设计目标 | **延迟受限**→高占用 | **吞吐受限**→专职分工、深流水 |

K1 的取向是"省 smem、压寄存器、堆占用"（`docs/02` §11.2）；K2 正好相反：
**寄存器放宽到 74–80**、smem 用到 96 KB，换来 4 个 MMA warp 全速算 + 两级流水掩盖访存。

**所以**：看到 K2 的 smem 和寄存器都不"省"，别以为是没调优——它在拿资源换流水线。

## 3. 三种角色怎么切：4 + 1 + 1

角色枚举在 `utils.cuh:79–84`：`WarpRole{MMA, LOAD_QKG, STORE, NonParticipant}`。
K2 在 kernel 开头按下标切分（`fwd_kernel2.cuh:188–197`）：

```cpp
constexpr int kComputeThreads = 128;                 // 4 个 MMA warp
int warp_id = threadIdx.x / 32;
if      (warp_id < 128/32)         warp_role = MMA;        // warps 0..3  (tid 0..127)
else if (warp_id < 128/32 + 1)     warp_role = LOAD_QKG;   // warp 4     (tid 128..159)
else if (warp_id < 128/32 + 2)     warp_role = STORE;      // warp 5     (tid 160..191)
```

这段在干什么：**纯按下标分段**。分支的边界（`128/32`、`+1`、`+2`）都是编译期常量，
每个线程跑一次、算出一个角色，后面整个 kernel 都靠这个枚举分派。

读的时候注意什么：判断用的是 `kComputeThreads / kWarpSize`，不是写死的 4/5——
以后改 MMA warp 数量时，LOAD/STORE 会自动顺延。

三点值得注意：

- **LOAD / STORE 各只需 1 个 warp，而且实际只有 `lane_predicate` 那一个 lane 干活。**
  TMA 是单线程发起的异步拷贝（`fwd_kernel2.cuh:237` 的 `cute::elect_one_sync()`），
  所以 `fwd_kernel2.cuh:324` 的 `if (warp_role == LOAD_QKG && lane_predicate)` 里只有一个线程
  跑整个 producer 循环。多出来的 lane 不做事，但保留了 warp 粒度，便于 barrier 计数。
- **MMA 用 4 个 warp，不多也不少。** `D=128`，每个 warp 负责 `N=128/4=32` 列 = 两个
  $16\times16$ 块（`fwd_kernel2.cuh:461–462` 的注释、`warp_id * 2` 的 tile 坐标）。
  4 个 warp 正好把输出列铺满。
- **角色名 `LOAD_QKG` 是历史遗留。** 在 K1 里它确实搬 q/k/g；在 K2 里搬的是 v/beta/workspace
  （`fwd_kernel2.cuh:325–334`），功能上就是"load warp"。

> 为何不把 LOAD / STORE 合成一个 warp？两者在同一时刻可能都在等（load 等 stage 空、
> store 等 stage 满），合并会互相阻塞。拆开后 LOAD 可以提前 3 个 tile 预取，STORE 可以滞后
> 2 个 tile 排空，正好对应两个流水线的深度。

**所以**：192 这个数字不是拍的，是 `4×32 + 32 + 32`。

## 4. 两级流水线：输入 3 级、输出 2 级

**输入用 `PipelineTmaAsync`（带 transaction-byte 计数），输出用 `PipelineAsync`（纯 arrive 计数）。**

两者的共享存储都嵌在 `SharedStorageK2` 里（`fwd_kernel2.cuh:109–111`），
在 `fwd_kernel2.cuh:202–212` 用两个 device 辅助函数（`utils.cuh:86–143`）构造：

```cpp
// 输入流水：LOAD 是 producer，MMA 是 consumer
auto load_pipeline = make_load_pipeline<InputStages>(
    shared_storage.load_pipeline, kTmaTransactionBytes, warp_role, /*num_producers=*/1, /*num_consumers=*/kComputeThreads);

// 输出流水：MMA 是 producer，STORE 是 consumer
auto store_pipeline = make_store_pipeline<OutputStages>(
    shared_storage.store_pipeline, warp_role, /*num_producers=*/kComputeThreads, /*num_consumers=*/1);
```

这段在干什么：把"谁生产、谁消费、几个线程参与"一次性写进 pipeline 的 `Params`。

读的时候注意什么：输入流水的 `num_producers=1` 是关键——TMA 只需要一个 leader lane 发。

角色 → `ThreadCategory` 的映射在辅助函数内部（`utils.cuh:98–105`、`129–134`）：

- **输入流水**：`LOAD_QKG → Producer`（且 `is_leader = elect_one_sync()`）、`MMA → Consumer`、
  其余 `NonParticipant`。生产者计数 1（只有那个 elected lane 真发 TMA），消费者计数 128
  （4 个 MMA warp 都要 `consumer_wait`）。
- **输出流水**：`MMA → Producer`（128 线程）、`STORE → Consumer`（1）。

两者都调了 `cutlass::pipeline_init_wait(1)`（`utils.cuh:114,141`），等 barrier 初始化对全 CTA 可见。

`kTmaTransactionBytes`（`fwd_kernel2.cuh:173–181`）是**每个输入 stage 的字节数**。
producer 用 `arrive_and_expect_tx` 语义，让 barrier 等够这么多字节才放行：

$$
\underbrace{4096}_{v}+\underbrace{64}_{\beta}+\underbrace{3\times4096}_{k^{\text{dec}},q^{\text{dec}},k^{\text{rest}}}
+\underbrace{512}_{g^{\text{total}}}+\underbrace{2\times512}_{\text{INV},M_{qk}}=17984\ \text{B}.
$$

这个值正好等于 `InputStorage` 每 stage 的大小（§7.1），两处对得上——不是各算各的。

### 4.1 输入 producer：一次喂 8 个 descriptor（`fwd_kernel2.cuh:323–422`）

LOAD warp 先把 8 个 TMA descriptor 切好（`fwd_kernel2.cuh:337–344`），然后逐 tile：

```
for t in 0..t_tiles:                                 // 346
    load_pipeline.producer_acquire(load_write);       // 347 等 stage[t%3] 空出来
    tma_barrier = load_pipeline.producer_get_barrier(load_write);
    stage  = load_write.index();                      // 350
    ws_idx = head_idx * total_tiles + tile_base + t;  // 351
    TMA v / beta / kd / qd / kr / gtotal / INV / Mqk → input[stage]   // 354–417
    ++load_write;                                     // 419
load_pipeline.producer_tail(load_write);              // 421 等最后几级被消费完
```

- `producer_acquire`（`347`）是**背压**：若第 `t%3` 号 stage 还没被 MMA 释放，LOAD 就阻塞在这里，
  不会覆盖正在被读的数据。
- 8 个 copy 都挂在**同一个** `tma_barrier` 上（`358–416` 的 `.with(*tma_barrier)`），
  所以 barrier 的 phase 翻转要等**全部 17984 B** 到齐——这就是 `kTmaTransactionBytes` 的用途。
- `beta` 是 1D 数组，只能按 8 个 bf16 对齐取值（`362–363` 的 `beta_linear & ~7`）；真正的 token
  起点偏移 `(head*T_total + bos + t*16) & 7` 记在 `beta_smem_offset`（`447`），与 K1 里同一套处理
  （`docs/02` §2）。

**所以**："8 个 descriptor、1 个 barrier、3 个 stage"是这一节的全部含义。

### 4.2 输入 consumer / 输出 producer：MMA warp 主循环（`fwd_kernel2.cuh:425–743`）

```
for t in 0..t_tiles:                                  // 434
    store_pipeline.producer_acquire(out_write);        // 436 等输出 stage[t%2] 空
    load_pipeline.consumer_wait(load_read);            // 437 等输入 stage[t%3] 到齐
    load_stage = load_read.index(); out_stage = out_write.index();   // 438–439
    ... 六个 Phase 在 input[load_stage] / s_acc / output[out_stage] 上跑 ...   // 445–731
    compute_barrier.arrive_and_wait();                 // 733 4 个 MMA warp 会合
    fence_view_async_shared();                         // 736 generic→async proxy 可见
    store_pipeline.producer_commit(out_write);         // 737 告诉 STORE 可读 out[stage]
    load_pipeline.consumer_release(load_read);         // 738 归还 input[load_stage]
    ++load_read; ++out_write;                          // 739–740
```

**每个 buffer 都只是从 `input[load_stage]` 取一个 Tensor 视图**（`445–458`）：
`v/beta/kd/qd/kr/g_total/INV/Mqk` 各一行 `make_tensor(make_smem_ptr(...), Layout{})`，
再加上 `s_acc`（`457`）与它的转置视图 `s_acc_T`（`458`）。

**没有一次拷贝。** 视图只是给同一块 smem 换了个坐标解释。

### 4.3 输出 consumer：STORE warp（`fwd_kernel2.cuh:746–802`）

```
for t in 0..t_tiles:                                  // 750
    store_pipeline.consumer_wait(out_read);            // 751 等 out[stage] 就绪
    actual_len = min(16, seq_len - t*16);              // 753
    if actual_len < 16:  逐元素手动写 gmem（避免尾块越界覆盖下一序列）   // 757–766
    else:                TMA store 整块                                // 767–779
    tma_store_wait<0>();                               // 781 等 TMA 读走 smem
    store_pipeline.consumer_release(out_read);         // 782 归还 out[stage]
    ++out_read;                                        // 783
```

尾块特判（`757–766`）是个必要细节：TMA store 按**整 tile 16 行**发。若序列长不是 16 的倍数，
最后一块只有 `actual_len<16` 行有效；直接 TMA 会把 padding 行写进下一序列。
所以改用单线程逐元素写 `out_raw_ptr`（`761–765`）——`lane_predicate` 保证只有一个 lane 跑，
不会重复写。

**所以**：STORE 是唯一碰输出 TMA 的角色；MMA 把 `out` 写进 smem 就算交差。

## 5. 每块末尾必须会合一次：`compute_barrier`

**这是理解 K2 的钥匙。**

`compute_barrier` 是 `NamedBarrier(128, 0)`（`fwd_kernel2.cuh:427`），只同步 4 个 MMA warp
（恰好 128 线程），不牵扯 LOAD/STORE。

为什么每 tile 末必须会合？因为跑动状态 `s_acc` 是**跨 warp 共享**的：

- Phase 1（`533–568`）里，每个 warp 都要读**整列 K 方向**的 `s_acc` 做两个 GEMM
  （`540,549–550` 的 `local_tile(s_acc, …, (warp_id*2, k))` 遍历所有 K 块）；
- Phase 6（`659–731`）里，每个 warp 只**写自己那 32 列**。

如果某个 warp 跑得快、先进入下一 tile 的 Phase 1，它读到的可能是别人还没更新完的旧 `s_acc`。

`compute_barrier.arrive_and_wait()`（`733`）保证"本 tile 的状态更新全部落 smem 之后，
大家才一起进下一 tile"。

**这不是性能优化，是递推正确性的唯一同步点。**

## 6. 初始状态与最终状态：不走流水线的两次搬运

状态不走输入/输出流水线，而是用**单独的** `state_acc_tma_barrier`（`fwd_kernel2.cuh:111`）。
理由很直接：它在流水线循环**之前/之后**各发生一次，不需要多级 stage。

按 `HasStateIn / HasStateOut / StateFP32` 的编译期分支，进入时有三条路径。

**(a) 有初态、bf16（默认，`fwd_kernel2.cuh:241–266`）。**
LOAD warp 的 elected lane 先 `arrive_and_expect_tx(32768)`（$=128\times128\times2$），
再 TMA 载入 `state_acc`（`243–263`）；随后全 CTA `__syncthreads()` + `wait(0)` +
`fence_view_async_shared()`（`264–266`）。
**状态直接落到它最终被使用的位置**，省掉一次转换。

**(b) 有初态、fp32（`fwd_kernel2.cuh:267–304`）。**
fp32 先 TMA 到 `state_fp32_buf`（`284–293`），再由全线程调 `smem_cvt_fp32_to_bf16`
（`300–303`；实现在 `utils.cuh:379–407`），做 fp32→bf16 的**同 smem 布局转换**，写进 `state_acc`。

这条路径解释了 `SharedStorageK2` 里那个 union（`fwd_kernel2.cuh:99–107`）：

```cpp
union {
    struct { InputStorage input[InputStages]; OutputStorage output[OutputStages]; };
    alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];
};
```

这段在干什么：让流水线缓冲和 fp32 状态转换缓冲**共用同一块字节**。

读的时候注意什么：能共用，靠的是生命周期不重叠——见下面那行注释。

**fp32 状态缓冲 65536 B，比整个流水线缓冲（`3×17984 + 2×4096 = 62144 B`）还大**，
所以 union 取两者最大 = 65536 B（§7.1）。注释（`99–100`）写明了安全性：
fp32 状态的 load/store 发生在流水线循环前后，"不会与流水线缓冲同时活着"。

**(c) 无初态（`fwd_kernel2.cuh:305–317`）。**
全线程把 `state_acc` 清零，再 `fence_view_async_shared()` + `__syncthreads()`。

**末态对称**：

- bf16：STORE warp 直接把 `state_acc` TMA 出去（`786–801`）；
- fp32：全线程 `__syncthreads()` → `smem_cvt_bf16_to_fp32` → `fence` → `__syncthreads()`，
  再由 STORE warp TMA 出去（`804–835`）。

注意这里**先全 CTA 同步再转换**（`809` 的注释）：要等流水线的 smem 空出来，
才能复用 union 里的 `state_fp32_buf`。

**所以**：初态/末态各搬一次，不占流水线的 stage，也不用 `compute_barrier`。

## 7. 上半场的账：96.1 KB smem 与 74–80 个寄存器

探针 `harness/smem_k2_probe.cu`（编译期 `cosize`/`sizeof`）与对已编译 `.so` 的
`cuobjdump -res-usage`；原始输出见同目录 `smem_k2_probe.out.txt`、`k2_res_usage.out.txt`。

### 7.1 smem 逐 buffer 账

| 缓冲 | 布局 | cosize | 字节 | 出处 |
|---|---|---|---|---|
| `v`/`k_decayed`/`q_decayed`/`k_restored` | `MMALayout` bf16 16×128 | 2048 | 4096 各 | `fwd_kernel2.cuh:11–15,85–89` |
| `beta` | `BetaSmemLayout` bf16 32 | 32 | 64 | `fwd_kernel2.cuh:23,86` |
| `g_total` | `GTotalLayout` fp32 128 | 128 | 512 | `fwd_kernel2.cuh:34,90` |
| `INV`,`Mqk` | `LMLayout` bf16 16×16 | 256 | 512 各 | `fwd_kernel2.cuh:35–39,91–92` |
| `out` | `VOLayout` bf16 16×128 | 2048 | 4096 | `fwd_kernel2.cuh:21,96` |
| `state_acc` | `StateSmemLayout` bf16 128×128 | 16384 | 32768 | `fwd_kernel2.cuh:24–28,82` |
| `state_fp32_buf` | fp32 128×128 | 16384 | 65536 | `fwd_kernel2.cuh:106` |

由此：

- 每输入 stage = `4×4096 + 64 + 512 + 2×512` = **17984 B**；
- `Input[3] + Output[2]` = `3×17984 + 2×4096` = **62144 B**；
- union 取 `max(62144, 65536)` = **65536 B**（fp32 状态缓冲胜出）；
- 再加 `state_acc` 32768 B 与 barrier，`sizeof(SharedStorageK2<128,16;3,2>)` = **98432 B（96.1 KB）**。

### 7.2 占用率

H100（sm90）每 SM：线程 2048、寄存器 65536、动态 smem ~228 KB（233472 B）。
以实测 `sizeof = 98432 B`、`REG≤80`、192 线程/CTA 计：

| 约束 | 上限 | 每 CTA 需求 | CTA/SM |
|---|---|---|---|
| smem | 233472 B | 98432 B | $\lfloor 233472/98432\rfloor = 2$ |
| 寄存器 | 65536 | $80\times192 = 15360$ | 4 |
| 线程 | 2048 | 192 | 10 |

**瓶颈是 smem，2 CTA/SM**（2×98432 = 196864 B ≤ 233472）。

这看似低，但 K2 的并行度本来就只在 head 轴（`grid=N×H`）；3 级输入 stage 已经把单 CTA 的
TMA 延迟盖住，不需要靠多 CTA 切换来隐藏访存——这正是 warp specialization 的目的。

### 7.3 寄存器

对已编译的 `flash_kda_C.*.so` 扫 `_flash_kda_fwd_recurrence` 的全部实例
（7 种 `HasStateIn/HasStateOut/StateFP32` × varlen/非 varlen = 14 个变体）：

| 项 | 结果 |
|---|---|
| 寄存器 | **74–80** |
| 栈 / 本地内存 | `STACK:0`、`LOCAL:0` |

14 个变体无一 spill（`harness/k2_res_usage.out.txt`）。

K2 **没有**像 K1 那样写 `__launch_bounds__(…, N)` 的占用下限
（`fwd_kernel2.cuh:133` 只有 `__launch_bounds__(NumThreads)`），编译器按需分 74–80 个寄存器。
因为它是吞吐受限，不是延迟受限（§2）。

**所以**：上半场三张账（smem 96.1 KB、2 CTA/SM、REG 74–80）是同一个设计取舍的三个侧面。

## 下半场：每个 Phase 算什么

## 8. 先立两个约定

不先把这两个约定说清，后面每个 Phase 都会看晕。

### 8.1 状态存的是 S 的转置（V×K 布局）

第 1 篇的 `src/kda_ref.py` 用 $S\in\mathbb{R}^{K\times V}$（`kda_ref.py:52,64`）。
K2 的 CUDA 侧**把状态存成转置** $S^\top\in\mathbb{R}^{V\times K}$，记作 `s_acc`
（`fwd_kernel2.cuh:82,457`）。

对照 FlashKDA 官方 torch 参考 `tests/torch_ref.py` 一眼就能看出来，它在 `227–239` 行对
`state_slice` 处处用 `.t()`：

```python
state_slice = work_state[seq_idx, h]                          # [V,K] 布局的 Sᵀ
v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())  # kd @ S
_out    = torch.matmul(q_decayed, state_slice.t())            # qd @ S
work_state[...] = fp32_fma(delta_s, state_slice.to(float).t(), g_total_exp).to(bf16).t()
```

这段在干什么：把 $[V,K]$ 的 `s_acc` 转回 $[K,V]$ 再做普通矩阵乘。

读的时候注意什么：**四个 `.t()` 不是装饰**——它们就是 kernel 里 transposed smem 视图的出处（§14）。

gmem 布局是 `(N·H, D, D) LayoutRight`（`fwd_launch.cu:53`），所以 `s_acc[v, k]` 就是
"value 行 × key 列"。记住这一点，§17.3 的"逐 key 缩放"才不会看晕。

### 8.2 片段约定：C 是 (M,N)，B 是 (N,K)

K2 全程用 SM80 的 `SM80_16x8x16_F32BF16BF16F32_TN`（`fwd_kernel2.cuh:469`），
并用 `make_tiled_mma(..., Tile<16,16,16>)` 把一个 warp 的 16×16 tile 铺成
**2 个 `m16n8k16` atom**（沿 N 方向）。

对 TN atom（`cutlass/include/cute/atom/mma_traits_sm80.hpp:53` 的 `SM80_16x8_Row`）：

- **C 累加器**：每个 lane 持 `(m,n) = (lane/4 + 8·b, 2·(lane%4) + a + 8·d)`，
  其中 $a,b,d\in\{0,1\}$ 是片段里的三个小坐标。
  即 **`m` 由 `lane/4` 决定、`n` 由 `lane%4` 决定**。
- **B 操作数**：B 的逻辑形状是 $(N,K)$，片段的 **`n` 由 `lane/4` 决定、`k` 由 `lane%4` 决定**。

两者都是"一个下标跟 `lane/4`、另一个跟 `lane%4`"，但**哪个跟哪个正好相反**。

所以 C 片段想当 B 用，必须**交换 `lane/4` 与 `lane%4` 的角色**——也就是对每个 lane 的数据
做一次 8×8 转置。

这正是 `movmatrix` 的用武之地：`SM75_U32x1_MOVM_T`
（`cutlass/include/cute/arch/copy_sm75.hpp:201–218`）封装
`movmatrix.sync.aligned.m8n8.trans.b16`，把一个 8×8 b16 矩阵（每 lane 一个 u32 = 2×b16）
**在寄存器内转置**。`utils.cuh:304–309` 的 `transpose_u32x4`（4 次 `MOVM_T`）是同一 idiom 的独立佐证。

### 8.3 六个代码 Phase ↔ 八个逻辑步

`fwd_kernel2.cuh` 里只标了 Phase 1..6。把 Phase 1 的两个 GEMM、Phase 3 的两次运算各自展开，
就是**八个逻辑步**：

| 逻辑步 | 代码位置 | 一句话 |
|---|---|---|
| ① $k^{\text{dec}}S$ | Phase 1 `533–568` | → `u_acc`（state 读取项） |
| ② $q^{\text{dec}}S$ | Phase 1 `533–568` | → `out_acc`（输出 state 项） |
| ③ cast + 载入 | Phase 2 `570–587` | `out_acc→bf16`，载 v/`INV`/β |
| ④ $u=(v-u)\beta$ | Phase 3 `589–623` | 得到 $\tilde v$ |
| ⑤ $u=\text{INV}\,u$ | Phase 3 `589–623` | $U=P\tilde v$（`MOVM_T` 转置） |
| ⑥ $M_{qk}U+$out | Phase 4 `625–650` | 输出投影 |
| ⑦ store | Phase 5 `652–657` | 写回 out smem |
| ⑧ `s_acc` 更新 | Phase 6 `659–731` | $S^\top\leftarrow S^\top\!\odot E^{\text{tot}}+U^\top k^{\text{rest}}$ |

一进循环：`store_pipeline.producer_acquire` / `load_pipeline.consumer_wait`（`436–437`），
每个 buffer 从 `input[load_stage]` 取 Tensor 视图（`445–458`）；出循环前 `compute_barrier`
会合一次（`733`），再 `commit`/`release`（`737–738`）。

下面逐 Phase。

## 9. Phase 1：两个 GEMM 一起打（`fwd_kernel2.cuh:533–568`）

**目标**：算出 state 读取项 $k^{\text{dec}}S$（存进 `u_acc`）和输出 state 项 $q^{\text{dec}}S$
（存进 `out_acc`）。

- 每个 warp 负责**两个 16 宽的 value 块**：`warp_id*2` 与 `warp_id*2+1`（`461–462,541,550`），
  4 个 warp 铺满 128 列。
- 收缩维是 key（$D=128$），切成 `K_BLOCKS = size<1>(k_decayed)/16 = 8` 块（`534`）。
- 循环体（`544–568`）是一段**手动软件流水**：先把上一轮预取的 A（`k_decayed`/`q_decayed`）和
  B（`s_acc`）从 `tCrAi_*` 搬进计算用的 `tCrA/tCrB`（`545–547`），再**预取下一块**（`549–564`），
  最后对两个 N-块各发两发 gemm（`552–553,566–567`）。`#pragma unroll`（`543`）让 8 次迭代
  完全展开、寄存器命名消除。
- A 和 B 都用 `SM75_U32x4_LDSM_N`：`k_decayed/q_decayed` 存成 `MMALayout`（K_INTER，
  `11–15,450–451`），`s_acc` 存成 `StateSmemLayout`（K_INTER，`24–28`）；两者作为 A/B 都是
  "列主序 → LDSM_N"直读（`481,489`），不需要转置。

**结果**：`u_acc[i]` $=k^{\text{dec}}S$、`out_acc[i]` $=q^{\text{dec}}S$，
都是 16×16 的 fp32 累加器片段（`527–531`）。

**所以**：Phase 1 只读 smem、不写 smem，两个 GEMM 共享同一条 A 预取流水。

## 10. Phase 2：把输出降到 bf16，顺手备料（`fwd_kernel2.cuh:570–587`）

四件事，全是从 smem 读进寄存器，没有写：

- **`out_acc → out_bf16`**（`571–574`）：$q^{\text{dec}}S$ 先降到 bf16 留在寄存器，
  Phase 4 再累加 $M_{qk}U$。**立即降精度是刻意的**——反正输出最终存 bf16。
- **载入 `v`**：`local_tile(v_tile, ..., (0, warp_id*2+i))` 经 `LDSM_N` 读成 C 片段
  `v_bf16[i]`（`576–581`）。它已经是 C 片段的坐标，所以下一步能直接和 `u_acc` 逐元素相减。
- **载入 `INV`**（就是 $P$）到 A 片段 `tCrA_k`（`583–584`），留给 Phase 3 的 $P\tilde v$。
- **$\beta$**：从 `beta_tile` 取两个值，各做一次 `sigmoid_tanh_approx_f32`（`utils.cuh:50`）。
  `beta0 = β[group_id]`、`beta1 = β[group_id+8]`，其中 `group_id = (lane_id/4)%8`（`476`）。
  由 §8.2，`lane/4` 正是 C 片段的**行（token）**，所以 β 逐 token 作用，位置完全正确。

**所以**：Phase 2 结束时，$v$、$P$、$\beta$、$q^{\text{dec}}S$ 全在寄存器里了，
smem 里只剩 `s_acc` 还要用。

## 11. Phase 3：`MOVM_T` 登场（先修 v，再乘 P）

**这一步是 K2 "消掉 smem 往返"的核心。**（`fwd_kernel2.cuh:589–623`）

**(a) 先做逐元素运算**（`593–606`）：

```
u_bf16[i] = bf16(u_acc[i])                                 // u_acc = k_dec@S
u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0   // c0: b=0 → 行 lane/4
u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1   // c1: b=1 → 行 lane/4+8
```

这段在干什么：$\tilde v_i=(v_i-k^{\text{dec}}_iS)\beta_i$（同 `torch_ref.py:228–229`）。

读的时候注意什么：`c0/c1` 的差别只在 `b`（§8.2），正是行 0–7 / 8–15，所以两个 β 就够。

**(b) 把结果转成 B 操作数**（`608–617`）：把 `u_bf16[i]` 的 4 个 u32 逐个
`SM75_U32x1_MOVM_T::copy` 转置，写进一个新的 B 片段 `tCrB_u_tmp`（`614`）。

**这一步的直觉一句话：刚算完的 16×16 结果还热在寄存器里，直接拿去当下一步的输入，不走 smem。**
代价是 4 条寄存器指令，而不是一次 STSM + 一次 LDSM。

**(c) $U=P\tilde v$**（`619–622`）：`gemm(A=INV, B=tCrB_u_tmp, C=u_acc)`。

注意 B 片段里装的其实是 $\tilde v^\top$，所以

$$
C[c,v]=\sum_{c'}P[c,c']\,\tilde v^\top[v,c']=\sum_{c'}P[c,c']\,\tilde v[c',v]=(P\tilde v)[c,v].
$$

**转置在这里被自动吃掉了。** 这正是 `MOVM_T` 的代数意义：它不是纯优化，
而是把 C→B 的片段语义差补上。

**(d)** 结果 `u_acc` 再 cast 回 `u_bf16`（`622`），供 Phase 4 / 6 复用。

> **没有 `MOVM_T` 会怎样？** 只能把 $\tilde v$ 用 `STSM` 写回一块 smem（或写进 `out` 缓冲），
> 再用 `LDSM_N` 按 B 布局读回来。那需要一块额外的 $16\times128$ bf16 缓冲、两次 smem 往返、
> 一次 producer/consumer 之外的手工同步。FlashKDA 用 4 条 `movmatrix` 全部省掉，实测见 §15。

**所以**：Phase 3 之后，$U$ 只活在寄存器里，而且已经是"能直接喂 GEMM 的 B 片段"。

## 12. Phase 4：把 Mqk·U 加到输出上（`fwd_kernel2.cuh:625–650`）

- 载入 `Mqk` 到 A 片段（`626–627`）。
- 对两个 N-块：**再**用 `MOVM_T` 把同一份 `u_bf16[i]` 转成 `tCrB_u_arr[i]`（`632–642`），
  `gemm(Mqk, tCrB_u_arr[i], out_acc)`（`645`），再 `out_bf16[i] += bf16(out_acc[i])`（`647–649`）。

于是 $o=q^{\text{dec}}S+M_{qk}U$（同 `torch_ref.py:232–233`）。

**$U$ 这份寄存器数据，喂了两路下游 GEMM**：Phase 4 算 $M_{qk}U$、Phase 6 算
$k^{\text{rest}\top}U$。Phase 4 转置好的 `tCrB_u_arr[i]` 会被 Phase 6 直接拿走当 B 操作数（`698`）。

所以 **$\tilde v$ 与 $U$ 各转置一次（共两次 `MOVM_T`），而 $U$ 被两路 GEMM 消费，一次 smem 都没落**。

## 13. Phase 5：把 out 写回 smem（`fwd_kernel2.cuh:652–657`）

`out_bf16[i]` 经 `SM90_U32x4_STSM_N`（`495`）写进 `output[out_stage].out` 的对应 16 列块
（`654–656`）。

之后到主循环尾（`733–740`）才 `fence_view_async_shared` + `producer_commit`，
把这块交给 STORE warp 走 TMA（§4.3）。

**MMA warp 自己不碰 TMA store，写 smem 即止。**

**所以**：`out` 是整个 K2 里唯一"为了给别人看"而写 smem 的中间量。

## 14. Phase 6：状态更新，两个转置视图（`fwd_kernel2.cuh:659–731`）

**这是六个 Phase 里唯一改 `s_acc` 的一步**，也是 transposed smem 视图的用武之地。

目标（对照 `torch_ref.py:235–239`）：$\Delta S=k^{\text{rest}\top}U$，
$S_{\text{new}}=S\odot E^{\text{tot}}+\Delta S$；存成 `s_acc`（$S^\top$ 布局）就是

$$
S^\top_{\text{new}}=S^\top\odot E^{\text{tot}}+U^\top k^{\text{rest}}.
$$

**两个 transposed 视图**：

- **`s_acc_T`**（`457–458`）：**同一块** `state_acc` 字节，用 `TransposedStateSmemLayout`
  （MN_INTER，`29–33`）读成 $[K,V]$。它和 `StateSmemLayout`（K_INTER）互为转置视图——
  Phase 1 按 $[V,K]$ 读、Phase 6 按 $[K,V]$ 读，**同一份数据、零拷贝**。
- **`k_restored_t`**（`464`）：`k_restored` 存成 `MMALayout`（$[C,D]$），
  用 `TransposedMMALayout` 读成 $[D,C]=[K,C]$，正好是 $k^{\text{rest}\top}$，
  供 `SM75_U16x8_LDSM_T` 直读（`485–486`）。

**循环**（`688–731`，`S_M_BLOCKS=8`）：

```
ring_A_kr[i]      = LDSM_T(k_restored_t[块 i])                  // 672–676  预取 k_rest^T 的 M 块
ring_S_acc[bi][i] = LDSM_T(s_acc_T[块 i, 列块 warp_id*2+bi])    // 678–682
ring_g0[slot]     = g_total(i*16 + group_id)                    // 684  逐 key 的衰减因子
ring_g1[slot]     = g_total(i*16 + group_id + 8)                // 685
for m in 0..8:
    gemm(ring_A_kr, tCrB_u_arr[bi], u_acc[bi])                  // 698  Δ 块 = k_rest^T @ U
    ...预取下一 M 块的 k_rest^T / g_total...                    // 701–707
    ring_S_acc = bf16(bf16→f32(ring_S_acc)*g0 + u_acc)          // 718–719  S*E + Δ
    STSM_T(ring_S_acc → s_acc_T[m, warp_id*2+bi])               // 723–724
    ...预取下一 M 块的 s_acc_T...                               // 726–728
```

- `g0/g1` 用 `group_id`（行内偏移）。这里 `s_acc_T` 的行是 **key**，所以 $E^{\text{tot}}$
  **逐 key 通道**缩放旧状态（`684–685,718–719`）。这与"decay 作用在 key 上"一致（§17.3）。
- $U$ 直接复用 Phase 4 的 `tCrB_u_arr`（`698`），**再一次没有任何 $U$ 的 smem 往返**。
- 累加在 fp32、只在写回 smem 时降到 bf16（`718–719` 的 `bf16→f32` 再算）：
  与 deep-dive §3 的"状态更新用 fp32 FMA、存储用 bf16"一致
  （`utils.cuh:55` 的 `bf16_to_f32` 就是 `cvt.f32.bf16`）。

**为什么用转置视图而不是直接算**：$k^{\text{rest}\top}$ 若用普通 `MMALayout` 读，
A 操作数的 M/N 就被锁死；转置视图 + `LDSM_T` 让"$k^{\text{rest}\top}$ 是 A（M=key）、
$U$ 是 B（N=value）"这个几何直接成立，输出 $[K,V]$ 又正好是 `s_acc_T` 的块索引，
`STSM_T` 原样写回。

**没有一次显式 transpose 指令，也没有一块额外缓冲。**

**所以**：Phase 6 = 一次普通 GEMM + 一条逐元素 FMA + 一次写回，结构上和别的 Phase 一样平。

## 15. 搬运预算与实测：`MOVM_T` 到底省了什么

### 15.1 六个 Phase 的 smem 归属

| Phase | 读 smem | 写 smem | 关键寄存器操作 | 行号 |
|---|---|---|---|---|
| 1 双 GEMM | kd/qd/s_acc（LDSM_N） | 无 | 8 次 k-iter 软件流水 | `533–568` |
| 2 cast+载入 | v（LDSM_N）、INV（LDSM_N）、β | 无 | `out_acc→bf16`、sigmoid | `570–587` |
| 3 $\tilde v$ & $P\tilde v$ | INV 已在寄存器 | 无 | `(v-u)*β`，**4× MOVM_T** | `589–623` |
| 4 $M_{qk}U$ | Mqk（LDSM_N） | 无 | **4× MOVM_T**（复用 u） | `625–650` |
| 5 store out | 无 | out（STSM_N） | — | `652–657` |
| 6 state update | s_acc_T/k_rest_t（LDSM_T）、g_total | s_acc_T（STSM_T） | 8×8 ring 预取 | `659–731` |

**$U$ 和 $k^{\text{rest}\top}$ 全程零 smem**：$\tilde v\to U\to$ 下一步 B 的链路全在寄存器里，
靠 `MOVM_T` 完成。`out` 是唯一为"给 STORE warp"写 smem 的中间量。

### 15.2 实测：正确性 + 收益

探针 `harness/movm_probe.cu`（H100/sm90a 实测，原始输出 `harness/movm_probe.out.txt`）
用与 K2 完全相同的 cute 片段（`SM80_16x8x16_F32BF16BF16F32_TN` + `Tile<16,16,16>`）做两件事。

**(1) 正确性**：对随机 $P$、$\tilde v$（bf16，16×16，值域 ±0.2）算 $U=P\tilde v$，
与 host 端 fp32 参考对比：

| 路径 | 说明 | U max_abs | U max_rel |
|---|---|---|---|
| **R（kernel 实际）** | C 片段 → **4× `MOVM_T`** → B 片段 → gemm | **1.5e-08** | **8.7e-08** |
| S（naive） | C 片段 → `STSM_N` 写 smem → `LDSM_N` 读成 B（**不转置**） | 2.1e-01 | 1.26 |

路径 R 与参考的差只有 fp32 累加顺序量级（~1e-8），**证明 `MOVM_T` 的寄存器转置是精确的**。

路径 S 的 B 操作数与 R **逐位不同（128/128 个 u32）**，$U$ 全错。这正说明
**C→B 的转置是数学上必需的**：不转置得到的是 $P\tilde v^\top$，不是 $P\tilde v$。

`MOVM_T` 并非"省一次 smem"那么简单，它**在寄存器里补齐了片段语义差**；
否则必须额外发转置指令，或者走 `LDSM_T`/`STSM_T`。

**(2) 收益**：1 warp、`clock64()` 计时、20 万次迭代的**延迟**微基准：

| 操作 | 每迭代 cycles |
|---|---|
| 4× `MOVM_T`（寄存器内转置） | **18.0** |
| `STSM_N` + `__syncwarp` + `LDSM_N` + `__syncwarp` | **21.8** |
| 加速比 | **1.21×** |

单看延迟差不大（smem 本身就快），**更实际的收益在别处**：

- 免去一块 $16\times128$ bf16 = **4096 B** 的 $U$ 中转缓冲，而 K2 的 smem 正卡在 2 CTA/SM
  （§7.2）；
- 免去 producer/consumer 之外的手工 `__syncwarp`/屏障，让 Phase 3→4→6 对 $U$ 的复用
  **全程无同步**。

deep-dive §4 说的 "eliminate every intermediate shared-memory round trip" 正指这条链。

**所以**：`MOVM_T` 的账要算两笔——延迟只赢 1.21×，但省下的 4096 B smem 和一路同步，
在 2 CTA/SM 的预算里更值钱。

## 16. 六个 Phase 的行号速查

| 内容 | 位置 | 行号 |
|---|---|---|
| 每 tile 取 Tensor 视图（v/β/kd/qd/kr/g/INV/Mqk/s_acc/s_acc_T） | `fwd_kernel2.cuh` | `445–464` |
| `make_tiled_mma` / copy atoms（LDSM_N/LDSM_T/STSM_N/STSM_T） | 同上 | `468–502` |
| `u_acc/out_acc` 分配、`K_BLOCKS` | 同上 | `527–534` |
| Phase 1 双 GEMM | 同上 | `533–568` |
| Phase 2 cast + 载 v/INV/β | 同上 | `570–587` |
| Phase 3 $\tilde v$ + `MOVM_T` + $P\tilde v$ | 同上 | `589–623` |
| Phase 4 $M_{qk}U+$out（`MOVM_T` 复用） | 同上 | `625–650` |
| Phase 5 store out | 同上 | `652–657` |
| Phase 6 state update（ring 预取/`LDSM_T`/`STSM_T`） | 同上 | `659–731` |
| `MOVM_T` 指令封装 / `transpose_u32x4` 佐证 | `cutlass.../copy_sm75.hpp`、`utils.cuh` | `201–218`、`304–309` |

## 17. 状态更新在算什么

Phase 6 那一行 `s_acc = s_acc⊙g_total + k_restoredᵀ U` 只写了半行代码，但它是整个 KDA
"逐通道遗忘"落地的地方。这一节把它拆开。

### 17.1 从逐 token 递推到块末

逐 token 的 gated delta rule（$S\in\mathbb{R}^{K\times V}$，$\alpha_t[k]=2^{g_t[k]}$）是
（`src/kda_ref.py:66–71`）：

$$
S_t=\operatorname{diag}(\alpha_t)\,S_{t-1}+k_t\,\delta_t^\top,\qquad
\delta_t=\beta_t\bigl(v_t-(k_t\odot\alpha_t)S_{t-1}\bigr).
$$

把块内 $C=16$ 步展开（块初态 $S_0$）：第 $j$ 步写入的 rank-1 项，此后每一步都要再乘一次逐通道
衰减 $\prod_{i=j+1}^{C}\alpha_i[k]$，所以

$$
S_C=\underbrace{\operatorname{diag}\Bigl(\prod_{i=1}^{C}\alpha_i\Bigr)}_{D_C}\,S_0
+\sum_{j=1}^{C}\operatorname{diag}\Bigl(\prod_{i=j+1}^{C}\alpha_i\Bigr)\,k_j\,\delta_j^\top .
\tag{17.1}
$$

记块内 log2 前缀和 $g^{\text{cum}}_{t}[k]=\sum_{i\le t}g_i[k]$、整块累计
$g^{\text{tot}}[k]=\sum_{i=1}^{C}g_i[k]$，则 $\prod_{i=1}^{C}\alpha_i=2^{g^{\text{tot}}}=:E^{\text{tot}}$、$\prod_{i=j+1}^{C}\alpha_i=2^{g^{\text{tot}}-g^{\text{cum}}_j}$。

于是 (17.1) 就是 kernel 的那一行：

$$
\boxed{\,S_C=\operatorname{diag}(E^{\text{tot}})\,S_0+k^{\text{rest}\top}U\,},\qquad
k^{\text{rest}}_j:=k_j\odot 2^{\,g^{\text{tot}}-g^{\text{cum}}_j}.
\tag{17.2}
$$

### 17.2 `k_restored` 的来历

`k_restored` 恰好实现 (17.2) 里的衰减因子 $2^{g^{\text{tot}}-g^{\text{cum}}}$。

K1 的 `decay_apply` 一次性算出四个派生量，全部逐 key 通道、逐 token：

| 张量 | 公式 | 代码行 |
|---|---|---|
| `k_decayed` | $k\odot 2^{g^{\text{cum}}}$ | `fwd_kernel1.cuh:453,455` |
| `q_decayed` | $q\odot 2^{g^{\text{cum}}}\cdot\text{scale}$ | `fwd_kernel1.cuh:454` |
| `k_inv`（$E^{-1}$ 后再规整为 $\beta L$ 的种子） | $k\odot 2^{-g^{\text{cum}}}$ | `fwd_kernel1.cuh:466,467` |
| `k_restored` | $k\odot 2^{-g^{\text{cum}}}\odot E^{\text{tot}}$ | `fwd_kernel1.cuh:468` |

其中 $E^{\text{tot}}$ 从 smem `g_total` 读进 `reg_gt`（`fwd_kernel1.cuh:396,411,418`）。

`g_total` 在 K1 的 gate 阶段先写下 log2 累计和（`fwd_kernel1.cuh:330`），再就地 `exp2`
（`353–356`）。**所以 kernel 里的 `g_total` 就是 $E^{\text{tot}}$ 本身（已取指数），
不是 log2 的累计和**——名字有点误导。K2 Phase 6 直接把它当乘子用
（`fwd_kernel2.cuh:718–719`）。

直觉：块内每个 token 的贡献，要以"它被写入之后、到块末还剩多少衰减"加权，
才能在块末一次性结算。`k_restored` 把这个权重预先吸收进 $k$，
于是状态更新退化成一次**无时刻依赖**的普通 GEMM $k^{\text{rest}\top}U$——
这正是 K2 能把状态更新塞进最后一个 Phase 的原因。

### 17.3 衰减因子是逐 **key** 通道的

注意 $g\in\mathbb{R}^{T\times D}$，这里的 $D$ 是 **key 通道**。

K1 的前缀和是**沿 token、按列（key）独立**做的（`fwd_kernel1.cuh:311–330`：
`compute_tid<128` 固定 `col=` key，循环 `row` 累加）。

所以 $g^{\text{tot}}$ 是长度 $D$ 的向量，$E^{\text{tot}}[k]$ 逐 key 通道不同。
这就是 KDA 相对 GLA / DeltaNet 的 **per-channel decay** 在状态更新上的落点。

K2 里 `s_acc = S^\top`（$[V,K]$，§8.1），所以"逐 key 缩放"表现为**逐列**缩放：

```cpp
// fwd_kernel2.cuh:684-685（预取）、706-707（滑窗预取）、718-719（应用）
ring_g0[i] = g_total(i * 16 + group_id);        // E_tot[key]
ring_g1[i] = g_total(i * 16 + group_id + 8);
ring_S_acc[bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c0)) * g0 + u_acc[bi](c0));
ring_S_acc[bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c1)) * g1 + u_acc[bi](c1));
```

这段在干什么：把旧状态读成 fp32、乘该 key 通道的 $E^{\text{tot}}$、加本块增量，再压回 bf16。

读的时候注意什么：`g0` / `g1` 对应的是**两个不同的 key 行**，不是两半 value。

Phase 6 的外层循环 `m` 扫的是 **key 维**的 8 个 16-块，`warp_id*2+bi` 扫的是 **value 维**；
`group_id=(lane/4)%8`（`476`）给出块内 key 行偏移，于是 `g_total(m*16+group_id)`
精确取到该行（key）的 $E^{\text{tot}}$（`684–685,706–707`）——
**逐 key 通道缩放旧状态**，与 §17.1 的 $\operatorname{diag}(E^{\text{tot}})S_0$ 完全一致。

对照 `src/kda_ref.py:140`（$S$ 存 $[K,V]$，所以是行缩放）：

```python
S = kr.transpose(0, 1) @ U + etot[:, None] * S   # (17.2)，行缩放 = 逐 key
```

两者是同一式的转置写法：K2 的列缩放 `*g_total` ⇔ 参考的行缩放 `etot[:,None]*`。

### 17.4 这一步不用再验证

`src/kda_ref.py` 的自测已经证明**分块式与逐 token 式在 fp64 下逐元素完全相等**
（`harness/kda_ref_selftest.out.txt`，max_abs=0），而 `kda_chunked_fwd` 的那一行就是 (17.2)。

所以 (17.2) 的分块含义不用再验证——它被恒等对拍兜住了。真正要回答的是：
**当状态在块与块之间以 bf16 落盘，这个恒等关系还保多少精度**（§18）。

### 17.5 坑之一：衰减因子为什么必须逐 chunk 搬进流水

**因为它每块都不一样，而且它不是标量。** 三件事叠在一起，决定了它只能跟着数据走：

1. **逐块不同**：$E^{\text{tot}}=2^{g^{\text{tot}}}$，$g^{\text{tot}}$ 是这一块 16 个 token 的
   $g$ 之和（`fwd_kernel1.cuh:311–330,353–356`）。换一块就是新值，不能循环外算一次。
2. **逐 key 通道**：长度 $D=128$ 的 fp32 向量，不是标量（`GTotalLayout`，
   `fwd_kernel2.cuh:34`）。顺带一提，它存 fp32 而不是 bf16：乘的对象是 fp32 累加器
   （`718–719`）。
3. **每块都要用**：Phase 6 拿它缩放整块旧状态。

所以它被放进 **per-tile workspace**，再由 LOAD warp 随每个 chunk 一起搬进输入 stage：
`InputStorage.g_total` 是 `GTotalLayout` 的 fp32 128 个 = **512 B**，是 17984 B 里的一块
（§4、§7.1）；MMA 侧只是取个 Tensor 视图（`fwd_kernel2.cuh:453`）。

它和这一块的其他数据挂在**同一个 barrier** 上（§4.1），到齐才放行——所以 Phase 6 读它时，
不会另有一次 global 访问在等。

搬进来之后，Phase 6 也不整块读，而是**按需取两行**：外层循环第 `m` 步只取
`g_total(m*16+group_id)` 与 `+8`（`684–685`），正好对上该 warp 这一轮要缩放的那两行 key。
两个 float 走 `ring_g0/ring_g1`（`670`）滑窗，与 `k_rest^T`、`s_acc_T` 的预取同拍
（`701–707`）。

换句话说，这 512 B 是"逐块衰减"的固定开销：K2 的输入流水预算，就是按"每个 chunk 自带一份
衰减"算出来的。

**所以**：$E^{\text{tot}}$ 不是"一个全局常数"，而是"每个 chunk、每个 key 通道一个数"，
它必须坐在输入 stage 里、和别的中间件同批到达。

## 18. 坑之二：`s_acc` 为什么敢用 bf16 存

deep-dive §3 的原话是：FlashKDA 把 on-chip 递推状态存 **bf16**，只要**更新本身用 fp32 FMA**，
"在推理基准上不引入可测量的精度损失"。

这一节用纯 PyTorch 的**受控消融**把它量化。

### 18.1 实验怎么设计（`harness/check_state_bf16.py`）

固定一份 fp64 输入，用 5 个变体跑分块前向（工作精度统一 **fp32**，bf16 舍入由探针显式插入），
全部对照 **fp64 逐 token golden**：

| 变体 | 状态存储 | `@S` GEMM 读 | 其它算符(kd/qd/ki/kr/U/P/Mqk) | 更新算术 |
|---|---|---|---|---|
| `fp32` | fp32 | fp32 | fp32 | fp32 |
| `store_bf16` | **bf16** | fp32 | fp32 | fp32 |
| `gemm_bf16(st_fp32)` | fp32 | bf16 | bf16 | fp32 |
| `kernel_bf16` | **bf16** | bf16 | bf16 | fp32 FMA |
| `kernel_arith_bf16` | bf16 | bf16 | bf16 | **bf16 FMA** |

`store_bf16`（只把块间状态降 bf16）隔离"状态存储"这一项；
`gemm_bf16(st_fp32)` 是"GEMM 操作数是 bf16、但状态保 fp32"的反事实。
两者的差，就是"bf16 存储的边际代价"。

### 18.2 结果（D=128；`max_rel = max_abs / max|golden|`）

**T=1024、g~log2 衰减（衰减强，KDA 的实际工作区）**

| 变体 | o max_abs | o max_rel | S max_abs | S max_rel |
|---|---|---|---|---|
| `fp32` | 7.5e-08 | 3.8e-07 | 1.0e-07 | 2.8e-07 |
| `store_bf16` | 6.8e-05 | 3.5e-04 | 9.6e-04 | 2.6e-03 |
| `gemm_bf16(st_fp32)` | 3.4e-04 | 1.7e-03 | 1.2e-03 | 3.1e-03 |
| `kernel_bf16` | 3.4e-04 | 1.7e-03 | 2.0e-03 | 5.4e-03 |

- 纯 fp32 分块 ≈ 机器精度（1e-7），确认分块代数本身无误差。
- **单独把状态降成 bf16**，只带来 o 3.5e-4 / S 2.6e-3 的 max_rel，比"bf16 GEMM 操作数"
  那一项（o 1.7e-3）**小约 5×**。
- 在 GEMM 已是 bf16 的前提下，再把状态从 fp32 换成 bf16（`gemm_bf16(st_fp32)` →
  `kernel_bf16`），**输出只变 o 2.8e-05（max_abs）**、末态变 9.5e-04（max_abs）。
  即 deep-dive 说的 "no measurable loss" 在**输出**上成立；末态本身有 ~2.5e-3 的漂移，
  但它不进入下一步计算。

**T=1024、g=0（无衰减，病态：状态单调累加到 |S|max≈2.8）**

| 变体 | o max_abs | o max_rel | S max_abs | S max_rel |
|---|---|---|---|---|
| `fp32` | 1.2e-07 | 4.2e-07 | 6.3e-07 | 2.2e-07 |
| `store_bf16` | 1.7e-03 | 6.0e-03 | 2.7e-02 | 9.5e-03 |
| `gemm_bf16(st_fp32)` | 1.3e-03 | 4.4e-03 | 9.9e-03 | 3.5e-03 |
| `kernel_bf16` | 1.7e-03 | 6.1e-03 | 2.5e-02 | 8.7e-03 |

无衰减时误差不再被 decay 冲刷、逐块累积，bf16 存储的边际代价升到 o ~2e-3（max_abs）。

但 KDA 实际用 `lower_bound=-5` 让衰减足够强（`flash_kda.cpp:128` 的 `gate_scale`），
落在上一张表的 regime，不会一直贴着 g=0。

### 18.3 fp32 FMA 更新值不值

固定 `kernel_bf16`，只把"$E^{\text{tot}}\cdot S+\Delta$"这一步从 fp32 累加改成 bf16
（`kernel_arith_bf16`）：

- T=1024、g=1 时，输出差 **1.2e-04（max_abs）**、状态差 2.0e-3；
- g=4（强衰减）时，两者**逐位相同**（差 ≤1e-6，旧状态已被衰减到可忽略）。

可见 fp32 FMA 更新在强衰减区几乎免费，在长序列/弱衰减区把误差压低约一个量级。
这支持 deep-dive §3 的选择。

### 18.4 和真实实现对得上

`harness/probe_refs.out.txt`（T=512, H=4, D=128, bf16）里，FLA `chunk_kda` 的末态对 fp64
golden max_rel≈6.3e-3、输出≈4.0e-3，FlashKDA 内核分别是 9.0e-3 / 5.4e-3——
和上表 `kernel_bf16` 的状态误差量级一致。

也就是说，端到端误差**主要由 bf16 操作数决定，而非状态存储**；
把状态升级到 fp32 最多改善 ~0.1%，却要付出 2× smem 和一次额外 cast（§19）。

### 18.5 结论与边界

- **支持 deep-dive §3**：在 KDA 实际的有衰减 regime，bf16 状态存储对**输出**的边际影响
  ≤ ~1e-4（max_abs），远小于 bf16 操作数误差本身；fp32 FMA 更新让这个代价可忽略。
- **边界**：以上是纯 PyTorch 消融，没有逐位复现 kernel 的 smem/MMA 累加顺序；
  g→0 的病态长序列误差会升到 ~0.5–1%，属已知 trade-off，而 KDA 的 gate 设计避开了这片区域。
  （补充：FlashKDA 自带的 `tests/torch_ref.py` 已作为 kernel 逐位 oracle 引入 `docs/04` §7——
  需要"kernel 累加序"的对照物时可以直接用它，它的 `o`/`S^T` 与 kernel `torch.equal`。
  本轮消融的 bf16 状态项与端到端结果的一致性未受影响。）
- 未验证"Neumann 级数 vs 前代求逆"等其它精度项——那是 `docs/02` §11.4 的主题，这里不重复。

**所以**：bf16 状态不是"省显存省出来的"，它有实测背书；真正的误差主项是 bf16 操作数。

## 19. 落盘：bf16 默认，fp32 可选

末状态写回有两条编译期分支：

- **bf16（默认，`StateFP32=false`）**：状态在 smem 里本来就是 bf16，STORE warp 直接把
  `state_acc` 用 TMA 一次发走（`fwd_kernel2.cuh:786–801`），**无转换、无额外同步**。
- **fp32（`StateFP32=true`）**：全 block `__syncthreads()` 后走 `smem_cvt_bf16_to_fp32`，
  把 bf16 状态转成 fp32 到 `state_fp32_buf`，再 `fence_view_async_shared()` +
  `__syncthreads()`，最后 STORE warp TMA（`fwd_kernel2.cuh:804–835`）。
  这块 fp32 缓冲与流水线输入/输出共用同一个 union（`99–107`），因为转换发生在流水线循环
  **之后**，生命周期不重叠。

这解释了 §18 的工程动机：状态存 bf16 不只是"省一半 smem"（`state_acc` 64 KB → 32 KB，§7.1），
更省掉了每次喂 bf16 GEMM 前都要做一次 fp32→bf16 cast——
`smem_cvt_bf16_to_fp32` 只在最终落盘做一次，而不是每块做一次。

**所以**：fp32 状态是"可选外壳"，代价是 2× smem 加一次全量转换，收益只有 ~0.1%（§18.4）。

## 20. 三句话小结

- K2 用 **4 + 1 + 1** 三种角色把"喂数据、算、发货"解耦，再用 `compute_barrier` 每块把
  4 个 MMA warp 拉齐一次——这是递推正确性的唯一同步点。
- 六个 Phase 里，$\tilde v\to U\to(M_{qk}U,\ k^{\text{rest}\top}U)$ 这条链**全程在寄存器**，
  靠 `MOVM_T` 在寄存器内转置、补齐 C→B 的片段语义差；只有 `out` 为了交给 STORE warp 才写 smem。
- 状态敢用 bf16，是因为误差主项是 bf16 操作数（存储的边际 ≤ ~1e-4），且更新用 fp32 FMA 补回精度；
  $E^{\text{tot}}$ 则必须按 chunk、按 key 通道、以 fp32 随流水搬进来。

下一篇（`docs/04`）进入数值对拍：把 kernel 的 `o` 与 $S^\top$ 和 fp64 golden、
以及 `tests/torch_ref.py` 这个 kernel 逐位 oracle 三方对齐（bf16 侧可由
`harness/check_state_bf16.py` 补充）；再往后是反向（`docs/05`），用已装好的 FLA `chunk_kda`
autograd 对拍。

## 实验与出处索引

正文里已经不提任务编号。这张表用于对账——每一行对应一次独立实验，脚本与原始输出都可复跑。

| 编号 | 做了什么 | 脚本 / 原始输出 | 正文位置 |
|---|---|---|---|
| P3-1 | 执行结构：warp 角色、两级流水线、状态搬运、`compute_barrier`；smem 与寄存器实测 | `harness/smem_k2_probe.cu`、`smem_k2_probe.out.txt`、`k2_res_usage.out.txt` | §2–§7 |
| P3-2 | 六个 Phase 的数据流、`MOVM_T` 的寄存器转置与收益 | `harness/movm_probe.cu`、`movm_probe.out.txt` | §8–§16 |
| P3-3 | 状态更新的分块含义；bf16 状态 + fp32 FMA 的精度消融 | `harness/check_state_bf16.py`、`check_state_bf16.out.txt` | §17–§19 |
| P1-4 | 分块式 vs 逐 token 的 fp64 恒等自检（(17.2) 的兜底） | `src/kda_ref.py`、`harness/kda_ref_selftest.out.txt` | §17.4 |
| P2-2(d) | 求逆途径（Neumann 级数 vs 前代）的精度讨论，本文不重复 | `docs/02` §11.4 | §18.5 |
| P4-6 | 把 `tests/torch_ref.py` 作为 kernel 逐位 oracle 引入 | `docs/04` §7 | §18.5 |

### 代码位置索引（结构 / 状态 / K1 对照）

| 结构 | 位置 | 行号 |
|---|---|---|
| 布局 `K2Layouts` | `fwd_kernel2.cuh` | `9–70` |
| smem `SharedStorageK2`（union/pipeline/barrier） | 同上 | `72–112`（union `99–107`，pipeline `109–111`） |
| `WarpRole` 枚举 | `utils.cuh` | `79–84` |
| 角色切分（4 MMA + 1 LOAD + 1 STORE） | `fwd_kernel2.cuh` | `188–197` |
| 流水线构造封装 | `utils.cuh` | `86–143` |
| 初始状态（bf16/fp32/零） | `fwd_kernel2.cuh` | `240–317` |
| LOAD warp producer 循环 | 同上 | `323–422`（`producer_tail` `421`） |
| MMA warp 主循环（acquire/wait/release/commit） | 同上 | `425–743`（`733,736–740`） |
| `compute_barrier`（4 MMA rendezvous） | 同上 | `427,733` |
| STORE warp consumer 循环 + 尾块特判 | 同上 | `746–802`（尾块 `757–766`） |
| 末状态写回（bf16/fp32） | 同上 | `786–835` |
| K2 launch（stages/线程/grid） | `fwd_launch.cu` | `29–30,186,201–214` |
| K1 逐 key 前缀和 / `g_total` 写入 | `fwd_kernel1.cuh` | `311–330` |
| K1 `g_total` 就地 `exp2`（$E^{\text{tot}}$） | 同上 | `353–356` |
| K1 `decay_apply`（kd/qd/ki/kr 四个派生量） | 同上 | `360–475`（应用 `449–471`） |
| K2 `g_total` smem 字段 / Tensor 视图 | `fwd_kernel2.cuh` | `90`、`453` |
| K2 Phase 6 更新（逐 key 缩放 + fp32 FMA + bf16 落盘） | 同上 | `659–731`（应用 `718–719`，g 取 `684–685,706–707`） |
| `gate_scale = lower_bound·log2(e)` | `flash_kda.cpp` | `128` |

所有行号均在 `~/github/FlashKDA` 下核对过。
