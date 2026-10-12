# KDA 前向 K2（recurrence）精读 I：warp specialization 与 TMA pipeline

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 3 篇，对应路线图 **P3-1**。
> 聚焦 FlashKDA 前向第二个 kernel **K2 = `_flash_kda_fwd_recurrence`**（`csrc/smxx/fwd_kernel2.cuh`）
> 的**执行结构**：warp specialization（4 个 MMA warp + 1 个 LOAD warp + 1 个 STORE warp）
> 与两级 TMA 流水线（输入 3-stage / 输出 2-stage）、初始/最终状态的搬运，以及相位的 rendezvous。
> K2 内部八个计算 Phase 的数据流留到 **P3-2**；本文给出足够上下文但不展开。
> 公式记号沿用第 1 篇 `01-kda-theory.md` §3；行号均为 `~/github/FlashKDA` 下**核对过**的真实行号。

## 0. K2 的职责、形状与 launch 配置

K1（第 2 篇）把每块 16 个 token 预处理成 workspace 中间件后，**K2 负责真正有串行依赖的部分**：
沿序列一块一块地做 gated delta-rule 递推，用**跑动状态** $S\in\mathbb{R}^{D\times D}$ 把输出
$o_t=q_t^\top S_t$ 投影出来，并在每块末更新状态。因为状态在块与块之间串行传递，K2 只能沿
**head 维度**并行：一个 CTA 串完一个 `(seq, head)` 的全部 tile。

| 项 | 值 | 出处 |
|---|---|---|
| kernel 入口 | `_flash_kda_fwd_recurrence` | `fwd_kernel2.cuh:133` |
| grid | `dim3(N, H)`（seq × head，一个 CTA 串一个序列的一个 head） | `fwd_launch.cu:203` |
| block | `192` 线程 = 4 MMA warp + 1 LOAD warp + 1 STORE warp | `fwd_launch.cu:186,204` |
| 输入流水 | `InputStages = 3`（v/beta/workspace 共 8 个 TMA descriptor） | `fwd_launch.cu:29` |
| 输出流水 | `OutputStages = 2`（out） | `fwd_launch.cu:30` |
| 动态 smem | `sizeof(SharedStorageK2<K2L,3,2>)` = **98432 B（96.1 KB）** | §5.1 |
| 模板参数 | `<…, CHUNK, D, InputStages, OutputStages, NumThreads, HasStateIn, HasStateOut, StateFP32, IsVarlen>` | `fwd_kernel2.cuh:115–132` |

**为什么 K2 要搞 warp specialization？** K1 是「单次 TMA load、无 pipeline」的短 kernel
（`docs/02` §0），一个 CTA 只干一块活，靠**高占用率**（8 CTA/SM）在等 TMA 时切走别的 CTA。
K2 恰恰相反：一个 CTA 要串完整条序列，**t_tiles 个 tile 顺序执行**，如果还用「所有线程一起
load、一起算、一起存」的 bloated 结构，每个 tile 都会被 TMA 延迟和同步卡住。于是把角色拆开：
**LOAD warp 专职喂数据、STORE warp 专职写结果、4 个 MMA warp 专职算**，三者通过硬件 barrier
解耦成流水线。这样 MMA warp 的算力不和访存争线程、也不被 `__syncthreads()` 全局栅栏打断。

## 1. Warp specialization：谁干什么

角色枚举定义在 `utils.cuh:79–84`（`WarpRole{MMA, LOAD_QKG, STORE, NonParticipant}`）。
K2 在 kernel 开头按下标切分（`fwd_kernel2.cuh:169–197`）：

```cpp
constexpr int kComputeThreads = 128;                 // 4 个 MMA warp
int warp_id = threadIdx.x / 32;
if      (warp_id < 128/32)         warp_role = MMA;        // warps 0..3  (tid 0..127)
else if (warp_id < 128/32 + 1)     warp_role = LOAD_QKG;   // warp 4     (tid 128..159)
else if (warp_id < 128/32 + 2)     warp_role = STORE;      // warp 5     (tid 160..191)
```

| 角色 | warp | 线程 | 职责 | 出处 |
|---|---|---|---|---|
| **MMA** | 0–3 | 128 | 块内递推 + 输出投影 + 状态更新；唯一的流水线**消费者**（输入）/ **生产者**（输出） | `fwd_kernel2.cuh:191–192,426` |
| **LOAD_QKG** | 4 | 32 | 初始状态载入 + 每 tile 8 个 TMA load；输入流水线**生产者** | `fwd_kernel2.cuh:193–194,240–323` |
| **STORE** | 5 | 32 | 每 tile TMA store 输出 + 最终状态写回；输出流水线**消费者** | `fwd_kernel2.cuh:195–196,746–802` |

三点值得注意：

- **LOGO/STORE 各只需 1 个 warp，且实际只有 `lane_predicate` 那一个 lane 干活。**
  TMA 是单线程发起的异步拷贝（`fwd_kernel2.cuh:237` 的 `cute::elect_one_sync()`），
  所以 `if (warp_role == LOAD_QKG && lane_predicate)`（`fwd_kernel2.cuh:324`）里只有一个线程
  跑整个 producer 循环。多出来的 lane 不做事，但保留了 warp 粒度便于 barrier 计数。
- **MMA 用 4 个 warp，而非更多。** `D=128`，每个 warp 负责 `N=128/4=32` 列 = 两个 $16\times16$
  块（`fwd_kernel2.cuh:461–462` 注释、`warp_id * 2` 的 tile 坐标）。4 warp 正好把输出列铺满。
- **角色名 `LOAD_QKG` 是历史遗留。** 在 K1 里它确实搬 q/k/g；在 K2 里它搬的是 v/beta/workspace
  （`fwd_kernel2.cuh:325–334`），功能上就是「load warp」。

> 为何不把 LOAD/STORE 合成一个 warp？两者在同一时刻可能都在等（load 等 stage 空、store 等
> stage 满），合并会互相阻塞；拆开后 LOAD 可以提前 3 个 tile 预取，STORE 可以滞后 2 个 tile 排空，
> 正好对应两个流水线的深度。

## 2. 两级流水线是怎么搭起来的

K2 用 cutlass 的 `PipelineTmaAsync`（输入，带 transaction-byte 计数）与 `PipelineAsync`
（输出，纯 arrive 计数），共享存储嵌在 `SharedStorageK2`（`fwd_kernel2.cuh:109–111`）。
构造走两个 device 辅助函数（`utils.cuh:86–143`）：

```cpp
// 输入流水：LOAD 是 producer，MMA 是 consumer
auto load_pipeline = make_load_pipeline<InputStages>(
    shared_storage.load_pipeline, kTmaTransactionBytes, warp_role, /*num_producers=*/1, /*num_consumers=*/kComputeThreads);

// 输出流水：MMA 是 producer，STORE 是 consumer
auto store_pipeline = make_store_pipeline<OutputStages>(
    shared_storage.store_pipeline, warp_role, /*num_producers=*/kComputeThreads, /*num_consumers=*/1);
```

角色→`ThreadCategory` 的映射在辅助函数内部（`utils.cuh:98–105`、`129–134`）：

- **输入流水**（`utils.cuh:100–105`）：`LOAD_QKG → Producer`（且 `is_leader = elect_one_sync()`）、
  `MMA → Consumer`、其余 `NonParticipant`。生产者计数 `num_producers=1`（只有那个 elected lane
  真正发 TMA），消费者计数 `num_consumers=128`（4 个 MMA warp 都要 `consumer_wait`）。
- **输出流水**（`utils.cuh:130–134`）：`MMA → Producer`（128 线程）、`STORE → Consumer`（1）。

两者都调 `cutlass::pipeline_init_wait(1)`（`utils.cuh:114,141`）等 barrier 初始化对所有线程可见。

`kTmaTransactionBytes`（`fwd_kernel2.cuh:172–181`）是**每个输入 stage 的字节数**，producer
用 `arrive_and_expect_tx` 语义让 barrier 等够这么多字节才放开：

$$
\underbrace{4096}_{v}+\underbrace{64}_{\beta}+\underbrace{3\times4096}_{k^{\text{dec}},q^{\text{dec}},k^{\text{rest}}}
+\underbrace{512}_{g^{\text{total}}}+\underbrace{2\times512}_{\text{INV},M_{qk}}=17984\ \text{B}.
$$

实测该值正好等于 `InputStorage` 每 stage 的大小（§5.1），二者一致。

### 2.1 输入 producer：LOAD warp 一次喂 8 个 descriptor（`fwd_kernel2.cuh:323–422`）

LOAD warp 先把 8 个 TMA descriptor 切好（`fwd_kernel2.cuh:337–344`），然后逐 tile：

```
for t in 0..t_tiles:                          // 346
    load_pipeline.producer_acquire(load_write);      // 347 等 stage[t%3] 空出来
    barrier = load_pipeline.producer_get_barrier(load_write);
    stage   = load_write.index();                     // 350
    ws_idx  = head_idx * total_tiles + tile_base + t; // 351
    copy TMA v / beta / kd / qd / kr / gtotal / INV / Mqk  → input[stage]  // 354–417
    ++load_write;                                     // 419
load_pipeline.producer_tail(load_write);              // 421 等最后几级被消费完
```

- `producer_acquire`（`fwd_kernel2.cuh:347`）是**背压**：若第 `t%3` 号 stage 还没被 MMA 释放，
  LOAD 就阻塞在这里，避免覆盖正在被读的数据。
- 8 个 copy 都挂在**同一个** `tma_barrier` 上（`fwd_kernel2.cuh:358–416` 的 `.with(*tma_barrier)`），
  所以 barrier 的 phase 翻转要等**全部 17984 B** 到齐——这就是 `kTmaTransactionBytes` 的用途。
- `beta` 是 1D 数组、按 8 bf16 对齐取值（`fwd_kernel2.cuh:362–363`），真正的 token 起点偏移
  `(head*T_total + bos + t*16) & 7` 记在 `beta_smem_offset`（`fwd_kernel2.cuh:447`），
  与 K1 里同一套处理（`docs/02` §2）。

### 2.2 输入 consumer & 输出 producer：MMA warp（`fwd_kernel2.cuh:425–743`）

```
for t in 0..t_tiles:                                 // 434
    store_pipeline.producer_acquire(out_write);       // 436 等输出 stage[t%2] 空
    load_pipeline.consumer_wait(load_read);           // 437 等输入 stage[t%3] 到齐
    load_stage = load_read.index(); out_stage = out_write.index();  // 438–439
    ... 八个 Phase 在 input[load_stage] / state_acc / output[out_stage] 上执行 ...  // 445–731
    compute_barrier.arrive_and_wait();                // 733 4 个 MMA warp 的 rendezvous
    fence_view_async_shared();                        // 736 generic→async proxy 可见
    store_pipeline.producer_commit(out_write);        // 737 通知 STORE 可读 out[out_stage]
    load_pipeline.consumer_release(load_read);        // 738 归还 input[load_stage]
    ++load_read; ++out_write;                         // 739–740
```

**`compute_barrier`（`fwd_kernel2.cuh:427`，`NamedBarrier(128, 0)`）是理解 K2 的关键。**
它是 4 个 MMA warp（恰好 128 线程）的**命名栅栏**，只同步 MMA 组、不牵扯 LOAD/STORE。
为什么每 tile 末必须让 4 个 MMA warp 会合？因为跑动状态 `s_acc` 是**跨 warp 共享**的：

- Phase 1（`fwd_kernel2.cuh:533–568`）里，每个 warp 都要读**整列 K 方向**的 `s_acc` 做
  $k^{\text{dec}} S$、$q^{\text{dec}} S$ 两个 GEMM（`fwd_kernel2.cuh:540,549–550` 的
  `local_tile(s_acc, …, (warp_id*2, k))` 遍历所有 K 块）；
- Phase 6（`fwd_kernel2.cuh:659–731`）里，每个 warp 只**写自己那 32 列**的 `s_acc`。

若某个 warp 跑得快、先进入下一 tile 的 Phase 1，它读到的可能是别人还没更新完的旧 `s_acc`。
`compute_barrier.arrive_and_wait()`（733）保证「本 tile 的状态更新全部落 smem 之后，大家才一起
进下一 tile」，是维持递推正确性的同步点。

### 2.3 输出 consumer：STORE warp（`fwd_kernel2.cuh:746–802`）

```
for t in 0..t_tiles:                                 // 750
    store_pipeline.consumer_wait(out_read);           // 751 等 out[stage] 就绪
    actual_len = min(16, seq_len - t*16);             // 753
    if actual_len < 16:  逐元素手动写 gmem（避免尾块越界覆盖下一序列）  // 757–766
    else:                TMA store 整块                               // 767–779
    tma_store_wait<0>();                              // 781 等 TMA 读走 smem
    store_pipeline.consumer_release(out_read);        // 782 归还 out[stage]
    ++out_read;                                       // 783
```

尾块特判（`fwd_kernel2.cuh:757–766`）是一个必要细节：TMA store 是**整 tile 16 行**的，
若序列长不是 16 的倍数，最后一块只有 `actual_len<16` 行有效；直接 TMA 会把 padding 行
写进下一序列，所以改用单线程逐元素写 `out_raw_ptr`（`fwd_kernel2.cuh:761–765`，`lane_predicate`
保证只有一个 lane 跑，不会重复写）。

## 3. 初始 / 最终状态：独立的 cluster transaction barrier

状态不走输入/输出流水线，而是用**单独的** `state_acc_tma_barrier`（`fwd_kernel2.cuh:111`），
因为它在流水线循环**之前/之后**各发生一次，不需要多级 stage。根据 `HasStateIn/HasStateOut/
StateFP32` 编译期分支，共有三条路径：

**(a) 有初始状态、bf16（默认路径，`fwd_kernel2.cuh:241–266`）。**
LOAD warp 的 elected lane `arrive_and_expect_tx(32768)`（= $128\times128\times2$）后直接 TMA
载入 `state_acc`（`fwd_kernel2.cuh:243–263`），全 CTA `__syncthreads()` + `wait(0)` +
`fence_view_async_shared()`（`fwd_kernel2.cuh:264–266`）。**状态直接落到它最终被使用的位置**，
省掉一次转换。

**(b) 有初始状态、fp32（`fwd_kernel2.cuh:267–304`）。**
fp32 状态先 TMA 到 `state_fp32_buf`（`fwd_kernel2.cuh:284–293`），再全线程调
`smem_cvt_fp32_to_bf16`（`fwd_kernel2.cuh:300–303`；实现在 `utils.cuh:378–407`）
做 fp32→bf16 的**同 smem 布局转换**，写进 `state_acc`。
这条路径解释了 `SharedStorageK2` 里那个 union（`fwd_kernel2.cuh:99–107`）：

```cpp
union {
    struct { InputStorage input[InputStages]; OutputStorage output[OutputStages]; };
    alignas(128) char state_fp32_buf[cute::cosize_v<StateSmemLayout> * sizeof(float)];
};
```

**fp32 状态缓冲（65536 B）比整个流水线缓冲（`3×17984 + 2×4096 = 62144 B`）还大**，
所以 union 取两者最大 = 65536 B（§5.1 实测）。注释（`fwd_kernel2.cuh:99–100`）指出：
fp32 状态 load/store 发生在流水线循环前后，「不会与流水线缓冲同时活着」，故可安全共用。

**(c) 无初始状态（`fwd_kernel2.cuh:305–317`）。** 全线程把 `state_acc` 清零，
`fence_view_async_shared()` + `__syncthreads()`。

**最终状态**同理对称：
- bf16：STORE warp 直接把 `state_acc` TMA 出去（`fwd_kernel2.cuh:786–801`）；
- fp32：全线程 `__syncthreads()` → `smem_cvt_bf16_to_fp32` → `fence` → `__syncthreads()`，
  再由 STORE warp TMA 出去（`fwd_kernel2.cuh:804–835`）——**注意这里先全 CTA 同步再转换**，
  因为要等流水线 smem 空出来才能复用 union 里的 `state_fp32_buf`（`fwd_kernel2.cuh:809` 注释）。

## 4. 与 K1 的结构对比

| 维度 | K1 `_flash_kda_fwd_prepare` | K2 `_flash_kda_fwd_recurrence` |
|---|---|---|
| 并行轴 | token（`grid=(tiles,H)`） | head（`grid=(N,H)`，CTA 内串 tile） |
| CTA | 256 线程，全部线程同角色 | 192 线程，**3 种角色** |
| TMA | 单次 load、无 pipeline | 输入 3-stage / 输出 2-stage |
| 状态 | 无（不读状态） | 读初态、写末态（独立 barrier） |
| 同步 | `__syncthreads` | pipeline barrier + `NamedBarrier(128)` |
| 寄存器（实测） | `REG:32`（压满 8 CTA/SM 上限） | `REG:74–80`（无 `__launch_bounds__` 下限） |
| 设计目标 | **延迟受限**→高占用 | **吞吐受限**→专职分工、深流水 |

K1 的典型结构（`docs/02` §11.2）是「省 smem、压寄存器、堆占用」；K2 反过来：**寄存器放宽到
74–80**、smem 用到 96 KB，换来 4 个 MMA warp 全速算 + 两级流水掩盖访存。

## 5. 实测：smem 预算、transaction 与寄存器

探针 `harness/smem_k2_probe.cu`（编译期 `cosize`/`sizeof`）与对已编译 `.so` 的
`cuobjdump -res-usage`；原始输出见同目录 `smem_k2_probe.out.txt`、`k2_res_usage.out.txt`。

### 5.1 smem 逐 buffer 账

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
- 加 `state_acc` 32768 B 与 barrier，**`sizeof(SharedStorageK2<128,16;3,2>)` = 98432 B（96.1 KB）**。

### 5.2 占用率（由实测 sizeof + 寄存器推得）

H100（sm90）每 SM：线程 2048、寄存器 65536、可动态 smem ~228 KB（233472 B）。以实测
`sizeof = 98432 B` 与 `REG≤80`、192 线程/CTA 计：

| 约束 | 上限 | 每 CTA 需求 | CTA/SM |
|---|---|---|---|
| smem | 233472 B | 98432 B | $\lfloor 233472/98432\rfloor = 2$ |
| 寄存器 | 65536 | $80\times192 = 15360$ | 4 |
| 线程 | 2048 | 192 | 10 |

**瓶颈是 smem，2 CTA/SM**（2×98432 = 196864 B ≤ 233472）。这看似低，但 K2 的并行度本来就
只在 head 轴（`grid=N×H`）；深流水（3 输入 stage 预取）已把单 CTA 的 TMA 延迟掩盖住，
不需要靠多 CTA 切换来隐藏访存——这正是 warp specialization 的目的。寄存器实测（§5.3）
`REG:74–80`、`STACK:0`、`LOCAL:0`，无 spill，符合「算力密集、放宽寄存器」的取向。

### 5.3 寄存器实测（`cuobjdump -res-usage`）

对已编译的 `flash_kda_C.*.so` 扫 `_flash_kda_fwd_recurrence` 的全部实例（7 种
`HasStateIn/HasStateOut/StateFP32` × varlen/非 varlen = 14 个变体）：

| 变体（`HasStateIn,HasStateOut,StateFP32`） | varlen | REG | STACK | LOCAL |
|---|---|---|---|---|
| 有态/有态/bf16、无态/有态/bf16、无态/无态/bf16 等 | 是/否 | **74–80** | 0 | 0 |

即全部变体 `REG:74–80`、`STACK:0`、`LOCAL:0`（`harness/k2_res_usage.out.txt`）。K2 **没有**
像 K1 那样写 `__launch_bounds__(…, N)` 的占用下限（`fwd_kernel2.cuh:133` 只有
`__launch_bounds__(NumThreads)`），编译器按需分配 74–80 个寄存器、不 spill——因为它是吞吐
受限而非延迟受限（§4）。

## 6. 行号速查

| 结构 | 位置 | 行号 |
|---|---|---|
| 布局 `K2Layouts` | `fwd_kernel2.cuh` | `9–70` |
| smem `SharedStorageK2`（union/pipeline/barrier） | 同上 | `72–112`（union `99–107`，pipeline `109–111`） |
| `WarpRole` 枚举 | `utils.cuh` | `79–84` |
| 角色切分（4 MMA + 1 LOAD + 1 STORE） | `fwd_kernel2.cuh` | `169–197` |
| 流水线构造封装 | `utils.cuh` | `86–143` |
| 初始状态（bf16/fp32/零） | `fwd_kernel2.cuh` | `240–317` |
| LOAD warp producer 循环 | 同上 | `323–422`（`producer_tail` `421`） |
| MMA warp 主循环（acquire/wait/release/commit） | 同上 | `425–743`（`733,736–740`） |
| `compute_barrier`（4 MMA warp rendezvous） | 同上 | `427,733` |
| STORE warp consumer 循环 + 尾块特判 | 同上 | `746–802`（尾块 `757–766`） |
| 最终状态写回（bf16/fp32） | 同上 | `786–835` |
| K2 launch（stages/线程/grid） | `fwd_launch.cu` | `29–30,186,201–214` |

## 7. 下一步

- **P3-2**：K2 八个 Phase 的逐条数据流拆解（dual GEMM $k^{\text{dec}}S/q^{\text{dec}}S$ →
  cast → $u=(v-u)\beta$ → $u=\text{INV}\,u$ → $M_{qk}U+$out → store → $S\leftarrow S\,g^{\text{tot}}
  +k^{\text{rest}\top}U$），重点讲 `SM75_U32x1_MOVM_T` 寄存器内转置消 smem 往返。
- **P3-3**：状态更新 `s_acc = s_acc*g_total + k_restored^T @ U` 的分块含义与 bf16 状态精度验证。
