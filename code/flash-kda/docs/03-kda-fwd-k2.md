# KDA 前向 K2（recurrence）精读：执行结构与八相数据流

> 本文是「Kimi Delta Attention 前向 / 反向算子实现解析」的第 3 篇，对应路线图 **P3-1 + P3-2**。
> 聚焦 FlashKDA 前向第二个 kernel **K2 = `_flash_kda_fwd_recurrence`**（`csrc/smxx/fwd_kernel2.cuh`）。
> **Part I（§0–§7，P3-1）**讲**执行结构**：warp specialization（4 个 MMA warp + 1 个 LOAD warp +
> 1 个 STORE warp）、两级 TMA 流水线（输入 3-stage / 输出 2-stage）、初始/最终状态搬运、相位 rendezvous。
> **Part II（§8–§15，P3-2）**讲 **MMA warp 内部六个代码 Phase**（展开为路线图的「八步」）的**逐条数据流**：
> 从 dual GEMM、`u=(v-u)β`、`u=INV u`、`Mqk U+out` 到状态更新 `s_acc = s_acc*g_total + k_restoredᵀ U`，
> 重点讲清 `SM75_U32x1_MOVM_T`（`movmatrix`）如何把 C 累加器片段**在寄存器内转置**成下一步的 B 操作数、
> **消掉每一次中间 smem 往返**，以及 transposed smem 视图（`s_acc_T`/`k_restored_t`）如何在状态更新里
> 复用同一块字节。公式记号沿用第 1 篇 `01-kda-theory.md`；行号均为 `~/github/FlashKDA` 下**核对过**的真实行号。

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

## 7. Part I 小结

Part I 把 K2 的**外部骨架**固定下来：3 种 warp 角色（`fwd_kernel2.cuh:169–197`）、两级流水线
（`utils.cuh:86–143`）、每 tile 末的 `compute_barrier`（`fwd_kernel2.cuh:427,733`）、状态搬运的独立
`state_acc_tma_barrier`（`fwd_kernel2.cuh:111`）。下面 **Part II** 钻进 MMA warp 主循环体
（`fwd_kernel2.cuh:445–732`）的**六个 Phase**，把每一步的数据流、寄存器/smem 归属、以及为什么这样写讲清。

---

# Part II：MMA warp 的六个 Phase 数据流（P3-2）

## 8. 记号与 MMA 骨架：为什么「输出片段」能直接当「B 操作数」

### 8.1 状态布局：K2 里的 `s_acc = Sᵀ`（$[V,K]$）

第 1 篇 `src/kda_ref.py` 用 $S\in\mathbb{R}^{K\times V}$（`kda_ref.py:52,70,135`）。K2 的 CUDA 侧
**把状态存成转置** $S^\top\in\mathbb{R}^{V\times K}$，记作 `s_acc`（`fwd_kernel2.cuh:82,457`）。
对照 FlashKDA 官方 torch 参考 `tests/torch_ref.py` 一眼可见：它在第 `227–239` 行对 `state_slice`
处处用 `.t()`：

```python
state_slice = work_state[seq_idx, h]                       # [V,K] 布局的 Sᵀ
v_chunk = v_chunk - torch.matmul(k_decayed, state_slice.t())  # kd @ S
_out    = torch.matmul(q_decayed, state_slice.t())            # qd @ S
work_state[...] = fp32_fma(delta_s, state_slice.to(float).t(), g_total_exp).to(bf16).t()
```

第 2 篇 already 指出初始状态直接 TMA 进 `state_acc`（`fwd_kernel2.cuh:241–266`），gmem 布局是
`(N·H, D, D) LayoutRight`（`fwd_launch.cu:53`），所以 `s_acc[v, k]` 就是「value 行 × key 列」。

### 8.2 MMA 片段约定：`C=(M,N)`，`B=(N,K)`（TN）

K2 全程用 SM80 的 `SM80_16x8x16_F32BF16BF16F32_TN`（`fwd_kernel2.cuh:469`），并以
`make_tiled_mma(..., Tile<16,16,16>)` 把一个 warp 的 16×16 tile 铺成 **2 个 `m16n8k16` atom**
（沿 N 方向）。对 TN atom（`cutlass/include/cute/atom/mma_traits_sm80.hpp:53` 的 `SM80_16x8_Row`）：

- **C 累加器**：每个 lane 持有 `(m,n) = (lane/4 + 8·b, 2·(lane%4) + a + 8·d)`，其中 $a,b,d\in\{0,1\}$
  是片段里的三个小坐标；即 **`m` 由 `lane/4` 决定、`n` 由 `lane%4` 决定**。
- **B 操作数**：B 的逻辑形状是 $(N,K)$，每个 lane 的 b 片段让 **`n` 由 `lane/4` 决定、`k` 由 `lane%4` 决定**。
- 两者都是「一个下标跟 `lane/4`、另一个跟 `lane%4`」，但**哪个跟哪个正好相反**——所以 C 片段想当 B 用，
  必须**交换 `lane/4` 与 `lane%4` 的角色**，也就是对 per-lane 数据做一次 8×8 转置。

这正是 `movmatrix` 的用途：`SM75_U32x1_MOVM_T`（`cutlass/include/cute/arch/copy_sm75.hpp:201–218`）
封装 `movmatrix.sync.aligned.m8n8.trans.b16`，把一个 8×8 b16 矩阵（warp 里每 lane 一个 u32=2×b16）
**在寄存器内转置**。`utils.cuh:304–309` 的 `transpose_u32x4`（4 次 `MOVM_T`）就是同一 idiom 的独立佐证。

### 8.3 代码注释里的「六 Phase」vs 路线图的「八步」

`fwd_kernel2.cuh` 里只标了 `Phase 1..6`（注释行 `533,570,589,625,652,659`）。把 Phase 1 的两个
GEMM、Phase 3 的两次运算各自展开，就得到路线图列的**八个逻辑步**：

| 逻辑步 | 代码位置 | 一句话 |
|---|---|---|
| ① $k^{\text{dec}}S$ | Phase 1 `533–568` | `k_decayed @ s_acc` → `u_acc`（state 读取项） |
| ② $q^{\text{dec}}S$ | Phase 1 `533–568` | `q_decayed @ s_acc` → `out_acc`（输出 state 项） |
| ③ cast + 载入 | Phase 2 `570–587` | `out_acc→bf16`、载 v/`INV`/β |
| ④ $u=(v-u)\beta$ | Phase 3 `589–623` | 得到 $\tilde v$ |
| ⑤ $u=\text{INV}\,u$ | Phase 3 `589–623` | $U=P\tilde v$（`MOVM_T` 转置） |
| ⑥ $M_{qk}U+$out | Phase 4 `625–650` | 输出投影 |
| ⑦ store | Phase 5 `652–657` | 写回 out smem |
| ⑧ $s\_acc$ 更新 | Phase 6 `659–731` | $S^\top \leftarrow S^\top\!\odot E^{\text{tot}} + U^\top k^{\text{rest}}$ |

一进循环：`store_pipeline.producer_acquire` / `load_pipeline.consumer_wait`（`436–437`）、
每个 buffer 从 `input[load_stage]` 取 Tensor 视图（`445–458`）；出循环前一齐 `compute_barrier.
arrive_and_wait()`（`733`）再 `commit`/`release`（`737–738`）。下面逐 Phase。

## 9. Phase 1：dual GEMM $k^{\text{dec}}S$ 与 $q^{\text{dec}}S$（`fwd_kernel2.cuh:533–568`）

**目标**：算出 state 读取项 $k^{\text{dec}}S$（存 `u_acc`）和输出 state 项 $q^{\text{dec}}S$（存 `out_acc`）。

- 每个 warp 负责 **两个 16 宽的 value 块**：`warp_id*2` 与 `warp_id*2+1`（`461–462, 541, 550`），
  4 个 warp 铺满 128 列 value。
- 收缩维是 key（$D=128$），切成 `K_BLOCKS = size<1>(k_decayed)/16 = 8` 块（`534`）。
- 循环体（`544–568`）是一段**手动软件流水**：先把上一轮预取的 A(`k_decayed`/`q_decayed`)、B(`s_acc`)
  从 `tCrA*i` 搬到计算用的 `tCrA/tCrB`（`545–547`），再**预取下一块**（`549–564`），最后对两个
  N-块各发两发 gemm（`552–553, 566–567`）。`#pragma unroll`（`543`）让 8 次迭代完全展开、寄存器
  命名消除。

**为什么 A 用 `SM75_U32x4_LDSM_N` 而 B 也 `LDSM_N`**：`k_decayed/q_decayed` 存成 `MMALayout`
（K_INTER，`fwd_kernel2.cuh:11–15,450–451`），`s_acc` 存成 `StateSmemLayout`（K_INTER，`24–28`）；
两者作为 A/B 都是「列主序 → LDSM_N」直读（`481,489`），无需转置。

**结果**：`u_acc[i]` $=k^{\text{dec}}S$、`out_acc[i]` $=q^{\text{dec}}S$，都是 16×16 fp32 累加器片段（`527–531`）。

## 10. Phase 2：cast 输出、载入 v/INV/β（`fwd_kernel2.cuh:570–587`）

- `out_acc→out_bf16`（`571–574`）：$q^{\text{dec}}S$ 先降到 bf16 留在寄存器，Phase 4 再累加 $M_{qk}U$。
  立即降精度是刻意的——反正输出最终存 bf16。
- 载入 `v`：`local_tile(v_tile, ..., (0, warp_id*2+i))` 经 `LDSM_N` 读成 C 片段 `v_bf16[i]`（`576–581`）。
- 载入 `INV`（$P$）到 A 片段 `tCrA_k`（`583–584`），供 Phase 3 的 $P\tilde v$ 用。
- β：从 `beta_tile` 取两个值并做 `sigmoid_tanh_approx_f32`（`utils.cuh:50`），
  `beta0 = β[group_id]`、`beta1 = β[group_id+8]`，`group_id = (lane_id/4)%8`（`476`）。
  由 §8.2，`group_id = lane/4` 正是 C 片段的**行（token）**，所以 β 按 token 逐行作用，完全正确。

## 11. Phase 3：$u=(v-u)\beta$ 与 $u=\text{INV}\,u$，`MOVM_T` 在此登场（`fwd_kernel2.cuh:589–623`）

这一步是 K2「消 smem 往返」的核心。

**(a) 元素运算得到 $\tilde v$**（`593–606`）：

```
u_bf16[i] = bf16(u_acc[i])                       // u_acc = k_dec@S
u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0   // c0: b=0 -> 行 lane/4
u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1   // c1: b=1 -> 行 lane/4+8
```

即 $\tilde v_i=(v_i-k^{\text{dec}}_iS)\beta_i$（同 `torch_ref.py:228–229`）。坐标 `c0/c1` 的差别仅在
`b`（§8.2），正是行 0–7 / 8–15。

**(b) `MOVM_T` 把 $\tilde v$ 从 C 片段转成 B 操作数**（`608–617`）：把 `u_bf16[i]` 的 4 个 u32
逐个 `SM75_U32x1_MOVM_T::copy` 转置，写进一个新的 B 片段 `tCrB_u_tmp`（`614`）。

**(c) $U=P\tilde v$**（`619–622`）：`gemm(A=INV, B=tCrB_u_tmp, C=u_acc)`。由 §8.2，B 片段里存的其实是
$\tilde v^\top$，所以 `C[c,v]=Σ_{c'} P[c,c'] \tilde v^\top[v,c'] = Σ P[c,c']\tilde v[c',v] = (P\tilde v)[c,v]` ——
**转置在这里被自动「吃掉」**。这正是 `MOVM_T` 的代数意义：不是纯优化，而是把 C→B 的片段语义差补上。

**(d)** 结果 `u_acc` 再 cast 回 `u_bf16`（`622`），供 Phase 4/6 复用。

> **没有 `MOVM_T` 会怎样？** 只能把 $\tilde v$ 用 `STSM` 写回一块 smem（或写 `out` 缓冲），再用
> `LDSM_N` 按 B 布局读回来。那需要一块额外的 16×128 bf16 缓冲、两次 smem 往返、一次
> producer/consumer 之外的手工同步。FlashKDA 用 4 条 `movmatrix`（寄存器指令）全部省掉。见 §15 实测。

## 12. Phase 4：$M_{qk}U+\text{out}$（`fwd_kernel2.cuh:625–650`）

- 载入 `Mqk` 到 A 片段（`626–627`）。
- 对两个 N-块：再次用 `MOVM_T` 把同一个 `u_bf16[i]` 转成 `tCrB_u_arr[i]`（`632–642`），
  `gemm(Mqk, tCrB_u_arr[i], out_acc)`（`645`），再 `out_bf16[i] += bf16(out_acc[i])`（`647–649`）。

于是 $o=q^{\text{dec}}S+M_{qk}U$（同 `torch_ref.py:232–233`）。注意 $\tilde v$ 的 C 片段只转置一次
概念上被用了**两次**（Phase 4 和 Phase 6）——§11 的 `u_bf16` 是 bf16（8 个 bf16），每次消费都重新
`MOVM_T` 一次，于是**同一份寄存器数据服务两个下游 GEMM**，全程不落 smem。

## 13. Phase 5：写回 out smem（`fwd_kernel2.cuh:652–657`）

`out_bf16[i]` 经 `SM90_U32x4_STSM_N`（`495`）写进 `output[out_stage].out` 的对应 16 列块（`654–656`）。
之后主循环尾（`733–740`）才 `fence_view_async_shared` + `producer_commit`，交给 STORE warp 走 TMA
（Part I §2.3）。**MMA warp 自己不碰 TMA store**，写 smem 即止。

## 14. Phase 6：状态更新 $S^\top\leftarrow S^\top\!\odot E^{\text{tot}}+U^\top k^{\text{rest}}$（`fwd_kernel2.cuh:659–731`）

这是六个 Phase 里唯一改 `s_acc` 的一步，也是 transposed smem 视图的用武之地。

**目标（对照 `torch_ref.py:235–239`）**：$\Delta S=k^{\text{rest}\top}U$，$S_{\text{new}}=S\!\odot\!E^{\text{tot}}+\Delta S$；
存成 `s_acc`（$S^\top$ 布局）即 $S^\top_{\text{new}}=S^\top\!\odot E^{\text{tot}}+U^\top k^{\text{rest}}$。

**两个 transposed 视图**：

- `s_acc_T`（`457–458`）：同一块 `state_acc` 字节，用 `TransposedStateSmemLayout`（MN_INTER，
  `fwd_kernel2.cuh:29–33`）读成 $[K,V]$。它与 `StateSmemLayout`（K_INTER）**互为转置视图**，
  所以 Phase 1 读 $[V,K]$、Phase 6 读 $[K,V]$，**同一份数据、零拷贝**。
- `k_restored_t`（`464`）：`k_restored` 存成 `MMALayout`（`[C,D]`），用 `TransposedMMALayout`
  读成 $[D,C]=[K,C]$，正好是 $k^{\text{rest}\top}$，供 `SM75_U16x8_LDSM_T` 直读（`485–486`）。

**循环**（`688–731`，`S_M_BLOCKS=8`）：

```
ring_A_kr[i]  = LDSM_T(k_restored_t[块 i])            // 672–676  预取 k_rest^T 的 M 块
ring_S_acc[bi][i] = LDSM_T(s_acc_T[块 i, 列块 warp_id*2+bi])  // 678–682
ring_g0[slot] = g_total(i*16 + group_id)              // 684  逐 key 的累计衰减
ring_g1[slot] = g_total(i*16 + group_id + 8)          // 685
for m in 0..8:
    gemm(ring_A_kr, tCrB_u_arr[bi], u_acc[bi])         // 698  Δ 块 = k_rest^T @ U
    ...预取下一 M 块的 k_rest^T / g_total...            // 701–707
    ring_S_acc = bf16(bf16→f32(ring_S_acc)*g0 + u_acc) // 718–719  S*E + Δ
    STSM_T(ring_S_acc -> s_acc_T[m, warp_id*2+bi])      // 723–724
    ...预取下一 M 块的 s_acc_T...                       // 726–728
```

- `g0/g1` 用 `group_id`（行的列内偏移）——这里 `s_acc_T` 的行是 **key**，所以 $E^{\text{tot}}$
  **逐 key 通道**缩放旧状态（`684–685, 718–719`），与「decay 作用在 key 上」一致。
- `U` 复用 Phase 4 已转置好的 `tCrB_u_arr`（`698`），**再一次没有任何 $U$ 的 smem 往返**。
- 累加在 fp32、只在写回 smem 时降到 bf16（`718–719` 的 `bf16→f32` 再算）：与 deep-dive §3 的
  「状态更新用 fp32 FMA、存储用 bf16」一致（`utils.cuh:55` 的 `bf16_to_f32` 就是 `cvt.f32.bf16`）。

**为什么用 transposed 视图而非直接算**：$k^{\text{rest}\top}$ 若用普通 `MMALayout` 读，A 操作数的
M/N 就被锁死；转置视图 + `LDSM_T` 让「$k^{\text{rest}\top}$ 是 A（M=key）、$U$ 是 B（N=value）」这一
几何直接成立，输出 $[K,V]$ 又正好是 `s_acc_T` 的块索引，`STSM_T` 原样写回。**没有一次显式 transpose
指令，也没有一块额外缓冲。**

## 15. 数据搬运预算与实测：`MOVM_T` 到底省了什么

### 15.1 六个 Phase 的 smem/寄存器归属（由源码逐行核对）

| Phase | 读 smem | 写 smem | 关键寄存器操作 | 行号 |
|---|---|---|---|---|
| 1 dual GEMM | kd/qd/s_acc（LDSM_N） | 无 | 8 次 k-iter 软件流水 | `533–568` |
| 2 cast+载入 | v（LDSM_N）、INV（LDSM_N）、β | 无 | `out_acc→bf16`、sigmoid | `570–587` |
| 3 $\tilde v$ & $P\tilde v$ | INV 已在寄存器 | 无 | `(v-u)*β`，**4× MOVM_T** | `589–623` |
| 4 $M_{qk}U$ | Mqk（LDSM_N） | 无 | **4× MOVM_T**（复用 u） | `625–650` |
| 5 store out | 无 | out（STSM_N） | — | `652–657` |
| 6 state update | s_acc_T/k_rest_t（LDSM_T）、g_total | s_acc_T（STSM_T） | 8×8 ring 预取 | `659–731` |

**U/$k^{\text{rest}\top}$ 全程零 smem**：$\tilde v$→$U$→下一步 B 的链路全在寄存器里靠 `MOVM_T` 完成。
`out` 是唯一为「给 STORE warp」写 smem 的中间量。

### 15.2 GPU 实测：`movmatrix` 正确性 + 收益

探针 `harness/movm_probe.cu`（编译/运行见 `harness/movm_probe.out.txt`，H100/sm90a 实测）用与 K2 完全
相同的 cute 片段（`SM80_16x8x16_F32BF16BF16F32_TN` + `Tile<16,16,16>`）做两件事：

**(1) 正确性**：对随机 `P`、$\tilde v$（bf16，16×16，值域 ±0.2）算 $U=P\tilde v$，与 host 端 fp32
参考对比：

| 路径 | 说明 | U max_abs | U max_rel |
|---|---|---|---|
| **R（kernel 实际）** | C 片段 → **4× `MOVM_T`** → B 片段 → gemm | **1.5e-08** | **8.7e-08** |
| S（naive） | C 片段 → `STSM_N` 写 smem → `LDSM_N` 读成 B（**不转置**） | 2.1e-01 | 1.26 |

路径 R 与参考的差仅 fp32 累加顺序量级（~1e-8），**证明 `MOVM_T` 的寄存器转置是精确的**。路径 S
的 B 操作数与 R 逐位不同（128/128 个 u32）、U 全错——这正说明 **C→B 的转置是数学上必需的**：
不转置得到的是 $P\tilde v^\top$，不是 $P\tilde v$。`MOVM_T` 并非「省一次 smem」那么简单，
它**在寄存器里补齐了片段语义差**，否则必须额外发转置指令或走 `LDSM_T`/`STSM_T`。

**(2) 收益**：1 warp、`clock64()` 计时、20 万次迭代的**延迟**微基准：

| 操作 | 每迭代 cycles |
|---|---|
| 4× `MOVM_T`（寄存器内转置） | **18.0**（≈4.5 cyc/条） |
| `STSM_N` + `__syncwarp` + `LDSM_N` + `__syncwarp` | **21.8** |
| 加速比 | **1.21×** |

单看延迟差不大（smem 本身很快），**更实际的收益在于**：(a) 免去一块 16×128 bf16=**4096 B** 的 U 中转
缓冲，K2 的 smem 本就卡在 2 CTA/SM（Part I §5.2）；(b) 免去 producer/consumer 之外的手工
`__syncwarp`/屏障，让 Phase 3→4→6 三段对 $U$ 的复用**全程无同步**。deep-dive §4 说的「eliminate
every intermediate shared-memory round trip」正指这条链。

## 16. 行号速查（Phase）

| 内容 | 位置 | 行号 |
|---|---|---|
| 每 tile 取 Tensor 视图（v/β/kd/qd/kr/g/INV/Mqk/s_acc/s_acc_T） | `fwd_kernel2.cuh` | `445–464` |
| `make_tiled_mma` / copy atoms（LDSM_N/LDSM_T/STSM_N/STSM_T） | 同上 | `468–502` |
| `u_acc/out_acc` 分配、`K_BLOCKS` | 同上 | `527–534` |
| Phase 1 dual GEMM | 同上 | `533–568` |
| Phase 2 cast + 载 v/INV/β | 同上 | `570–587` |
| Phase 3 $\tilde v$ + `MOVM_T` + $P\tilde v$ | 同上 | `589–623` |
| Phase 4 $M_{qk}U+$out（`MOVM_T` 复用） | 同上 | `625–650` |
| Phase 5 store out | 同上 | `652–657` |
| Phase 6 state update（ring 预取/`LDSM_T`/`STSM_T`） | 同上 | `659–731` |
| `MOVM_T` 指令封装 / `transpose_u32x4` 佐证 | `cutlass.../copy_sm75.hpp`、`utils.cuh` | `201–218`、`304–309` |

## 17. 下一步

- **P3-3**：状态更新 $s\_acc = s\_acc*g^{\text{total}} + k^{\text{rest}\top}U$ 的分块含义与
  **bf16 状态精度**验证（对照 deep-dive §3 的「state 存 bf16、更新用 fp32 FMA」给出实测误差）。
