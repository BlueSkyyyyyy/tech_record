# `sm100_fp8_fp4_mega_moe.cuh` 逐段精读（二）：warp specialization 与 mbarrier 同步

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 6 篇，对应路线图 **P3-2**。
> 上一篇（P3-1）讲了**静态骨架**：grid/cluster/线程数、BLOCK_M 启发式、smem 预算→`num_stages`。
> 本篇讲**动态骨架**：这一堆 warp 之间靠什么 barrier 建立生产者-消费者关系，相位怎么翻转，
> TMA 载入激活/权重/scale 的坐标与 2-CTA multicast，以及调度器 task 槽的双缓冲。
>
> **重要**：主体 kernel 是 **SM100（Blackwell）专属**（tcgen05、2-CTA UMMA、TMEM、FP8xFP4），
> 本机 H100（sm90）**不跑端到端**，本篇以源码精读为主。但「mbarrier 多级流水的相位语义 /
> warp specialization 的解耦」是 SM90/SM100 通用的，本文用 H100 真机实验验证了相位约定（见 §6）。
>
> 源码坐标（`~/github/DeepGEMM`，2026-09-30 公开版 commit `057ca59`）：
> - 主体 kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）
> - 调度器：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（481 行）
> - barrier 原语：`ptx/tma.cuh`、`ptx/tcgen05.cuh`、`comm/barrier.cuh`
> - cutlass：`ClusterTransactionBarrier`（`third-party/cutlass/include/cutlass/arch/barrier.h`）
>
> 行号均逐条核对过。

## 1. 五类 warp 与两个「解耦层」

kernel 里 `warp_idx` 一分派，16 个 warp 落在五类角色上（见 P3-1 §4）。同步的目标是让这 5 类角色
**互不等待地各跑各的**，只在该交换数据的地方用 mbarrier 点对点握手。核心是**两层解耦**：

```
   全局内存/NVLink              shared memory              TMEM             寄存器/smem
  ┌───────────────┐   TMA    ┌──────────────┐  UTCCP+UMMA  ┌──────┐  tmem_load  ┌────────┐
  │ acts / weights│ ───────▶ │ smem_a/b + SF│ ───────────▶ │ acc  │ ──────────▶ │ epilogue│
  └───────────────┘          └──────────────┘              └──────┘             └────────┘
        ▲                          ▲          full/empty        ▲         tmem_full/empty
        │ dispatch pulls           │  (TMA↔MMA 解耦)            │         (MMA↔epilogue 解耦)
```

- **层 1（full/empty）**：TMA loader warp（生产者，`warp 4/5`）与 MMA warp（消费者，`warp 6`）
  之间，用 `full_barriers[stage]`（TMA 完成）和 `empty_barriers[stage]`（MMA 用完 stage）握手，
  实现 **K 方向的多级 TMA 流水**（`kNumStages` 级）。
- **层 2（tmem_full/tmem_empty）**：MMA warp 与 epilogue warpgroup（`warp 8-15`）之间，
  因为累加器放在 **TMEM**，用 `tmem_full_barriers`（UMMA 提交完成）和 `tmem_empty_barriers`
  （epilogue 读完该 accumulator）握手，实现 **accumulator 的双缓冲**（`kNumEpilogueStages=2`）。

两层用**不同的 stage 深度**，这一点很关键：TMA 流水通常 5–10 级（省 smem），TMEM 累加器只 2 级
（TMEM 列数有限，`kNumAccumTmemCols = UMMA_N * kNumEpilogueStages`，`:229`）。于是 epilogue
可以滞后 MMA **两个 CTA-block**，TMA 又可以领先 MMA 好几级——三段各自打满自己的吞吐。

调度器（`warp 7`）与所有消费者之间还有第三组：`task_info_full/empty_barriers`（§4）。

## 2. 全部 mbarrier 清单与初始化

所有 barrier 都在 `SharedStorage`（`:196-218`）：

| barrier | 个数 | init 计数 | 生产者 arrive | 消费者 wait | 作用 |
|---------|------|----------|--------------|------------|------|
| `dispatch_barriers` | 4 | 1（`:254`） | 远端 TMA pull 的 `complete_tx`（`:572`） | dispatch warp（`:556`） | dispatch 从 NVLink 拉 token 的 1D TMA |
| `full_barriers` | `kNumStages` | 2×2=4（`:262`） | A/B 两个 loader（各 2 CTA 的 arrive） | MMA warp（`:886`） | `smem_a/b/sf` 就绪 |
| `empty_barriers` | `kNumStages` | 1（`:263`） | MMA 的 `umma_arrive_multicast`（`:874`） | A/B loader（`:729`/`:783`） | stage 可被 TMA 重写 |
| `tmem_full_barriers` | `kNumEpilogueStages` | 1（`:268`） | UMMA commit（`:878`） | epilogue（`:986`） | accumulator 就绪 |
| `tmem_empty_barriers` | `kNumEpilogueStages` | `2*kNumEpilogueThreads`=512（`:270`） | 所有 epilogue 线程（`:1008`） | MMA warp（`:865`） | accumulator 已读空 |
| `combine_barriers` | `kNumEpilogueWarps*2` | 1（`:274`） | combine 的 1D TMA（`:1460`） | combine warp（`:1478`） | combine 分块加载 |
| `task_info_full_barriers` | 2 | 1（`:277`） | 调度器`publish_task`（`:411`） | 所有消费者（`:249`） | task 槽就绪 |
| `task_info_empty_barriers` | 2 | `2*kNumEpilogueThreads`=512（`:278`） | **仅 epilogue**（`:258`） | 调度器（`:428`/`:462`） | task 槽可复用 |

几个 init 计数的含义：

- **`full_barriers.init(2 * 2)`（`:262`）**：注释 "Arrive at 2 CTAs, A + B"。A loader 与 B loader
  各自让**两个 CTA**都到齐才算满：leader CTA 用 `arrive_and_expect_tx`（`:748`/`:798`/`:810`，
  自己算一个 arrive），非 leader CTA 用 `arrive(0u)`（`:750`/`:800`/`:812`）。注意
  `ClusterTransactionBarrier::arrive(cta_id)` 的语义是**替 `cta_id` 那个 CTA 的 barrier arrive**
  （`barrier.h:382`→`ClusterBarrier::arrive`，`mapa.shared::cluster` 映射到远端），而 2-CTA TMA
  只给 leader 发信号，所以非 leader 显式 `arrive(0u)` 把票投到 leader 的 barrier 上。
- **`empty_barriers.init(1)`**：一次 UMMA 的 `tcgen05.commit` 用 `umma_arrive_multicast_2x1SM`
  （`:869-874`）同时给两个 CTA 的 empty barrier 各发一次 arrive，故每 CTA 计数 1。
- **`tmem_empty_barriers.init(2 * kNumEpilogueThreads)`（`:270`）**：TMEM 累加器跨 2 个 CTA
  （2-SM UMMA），leader 的 MMA warp 要等**两个 CTA 的全部 epilogue 线程**都读完（各 256 线程，
  每线程 `arrive(0u)` 投到 leader，`:1008`），故 512。`kNumScheduleConsumerThreads = 2*kNumEpilogueThreads`
  （`:189`）同理。

初始化在 `:241-285` 由 warp 0-3 分工完成后 `cute::cluster_sync()`（`:286`）；TMEM 由 warp 3 分配
（`Allocator().allocate(...)`，`:284`）。

## 3. 层 1：TMA ↔ MMA 的 K 方向流水

### 3.1 相位翻转：`advance_pipeline`

整个非 epilogue 组共享一份流水指针（`:307-314`）：

```
uint32_t stage_idx = 0, phase = 0;
auto advance_pipeline = [&](uint32_t& k_block_idx) {
    ++ k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;      // 回绕到 stage 0 时才翻转相位
};
```

即：stage 在 `[0, kNumStages)` 上循环，**每走完一轮才翻一次 phase**。这与 cutlass
`PipelineState::operator++` 完全同构（`sm90_pipeline.hpp:205-211`：`index_==Stages` 时 `index_=0; phase_^=1`）。

### 3.2 相位约定：生产者从 1 开始（H100 实测）

微妙之处在**初值相位**：loader 等 empty 用 `wait(phase ^ 1)`（`:729`/`:783`），MMA 等 full 用
`wait(phase)`（`:886`）。`phase` 初值都是 0，所以：

- **MMA 第一次 `full.wait(0)`** → 等第一轮 TMA 完成的相位；
- **loader 第一次 `empty.wait(1)`** → **立即成功**（buffer 初始就是空的）。

这正是 cutlass 的生产者约定：`make_producer_start_state()` 给生产者初值 phase=1，
注释 "Producer starts with an opposite phase as the buffers are initially empty"
（`sm90_pipeline.hpp:253-258`）。也就是说 `mbarrier.try_wait.parity(p)` 的语义是：
**全新 mbarrier 上 `p=1` 会立即返回成功，`p=0` 会阻塞；一次 arrive 完成后反过来**。

这个约定很容易记错，所以本文在 H100 上用真机 probe 验证（§6 实验第一部分，`probe_phase_kernel`）：
输出 `fresh mbarrier: try_wait(parity=0)=0 try_wait(parity=1)=1`，
`after 1 arrive: try_wait(parity=0)=1 try_wait(parity=1)=0`——与上面的推断一致。

### 3.3 A loader：激活 + SFA

`warp == kNumDispatchWarps`（=4，`:690`）是「载激活」的 TMA warp。每个 task 的 K 循环里：

- 先等 `empty_barriers[stage].wait(phase ^ 1)`（`:729`），保证本 stage 上一轮的 MMA 已用完；
- 算坐标（`:732-739`）：`m_idx = block_idx * BLOCK_M`、`k_idx = k_block_idx * BLOCK_K`；
  **非 leader CTA 再加半个 UMMA M 偏移** `task_info.get_umma_aligned_valid_m()/2`（`:738-739`），
  因为 2-CTA cluster 里 A 被切成两半、每 CTA 只载 `LOAD_BLOCK_M = BLOCK_M/2`（`:165`，P3-1 §3）；
- 两个 `tma::copy`：激活（`:743`，`kSwizzleAMode=128`，multicast=2）与 SFA（`:745`，
  `SF_BLOCK_M x 1`，无 swizzle）；
- leader CTA `arrive_and_expect_tx(smem_a*2 + smem_sfa*2)`（`:748`，`*2` 因为 2-CTA 各一份），
  非 leader `arrive(0u)`（`:750`）。

在此之前还要等数据的**到达条件**（这两条是 dispatch→GEMM 侧的握手，不是逐 stage 的）：
`Linear1` 等 `l1_full_count == BLOCK_M*(generation+1)`（`:713-716`，dispatch 攒够一整个 block 的 token），
`SharedLinear2` 等 shared 的 full count（`:717-720`）。`Linear2` 则用
`L2KBlockDependency`（`:723-726`）等 L1 各 N block 就绪（§5.3）。

### 3.4 B loader：权重 + SFB（含 FP4 unpack）

`warp == kNumDispatchWarps + 1`（=5，`:756`）。同样先等 empty（`:783`），然后：

- 权重偏移（`:786-788`）：`k_idx = k_block_idx*BLOCK_K`；SFB 的 K 索引
  `sfb_k_idx = local_expert_idx * shape_sfb_k + k_block_idx*(BLOCK_K/128)`——**每专家一段 SF**，
  这就是 P2-1 讲的「按专家拼好的 SF 布局」；
- `n_block_idx = n_cluster_idx*2 + (is_leader_cta?0:1)`（`:775`，2-CTA 各占一个 BLOCK_N）；
- `half_n`/`half_idx`（`:778`）：N 方向分成上下两半（gate/up），交给 `tma::copy_nd<...,4>`
  的 5D 坐标（`:793`/`:803`）——这也是 L1 权重 gate/up 交错后的物理布局；
- **FP4 unpack**：routed 权重可能是 e2m1（`:155-156`），`expect_tx` 里
  `kNumWeightBytes = sizeof(smem_b)*2 / (kIsWeightFP8 ? 1 : 2)`（`:808-809`）——FP4 两个元素打一字节，
  从全局搬进 smem 的字节数是 FP8 的一半（重量比以 smem 为准）；
- shared 专家走 `copy_nd` 的 shared 版本（`:792-801`）与 routed（`:802-814`）分开。

两个 loader 的 `expect_tx` 字节数都按 **2-CTA** 计（`smem_a*2`、`smem_b*2`），因为它们共同喂
leader 上那个 init(4) 的 full barrier。

## 4. 调度器 task 槽的双缓冲

调度器在 `scheduler/mega_moe.cuh`。它把「下一批 task」写进一个 2 级环形槽，消费者来取。

- `get_next_task`（`:248-253`）：`task_info_full_barriers[sched_stage_idx].wait(sched_phase)`，
  读 `task_infos[sched_stage_idx]`，再 `advance_sched_pipeline()`（`:242-246`，同样是回绕翻相位）；
- `publish_task`（`:409-419`）：lane 0/1 各替一个 CTA 写 `task_infos`（`st_async_cluster`），
  并 `arrive_and_expect_tx(sizeof(task_info_t), cta_id)`；
- `release_task_info`（`:255-259`）：`fence_acq_rel_cta()` 后 `task_info_empty_barriers[^1].arrive(0u)`。

**为什么只有 epilogue 调 `release_task_info`（`:990`）？** task 槽 init 计数是
`2*kNumEpilogueThreads=512`（`:278`），即「两个 CTA 的所有 epilogue 线程读完」才允许调度器覆盖。
A/B loader 和 MMA warp 也调 `get_next_task` 消费同一个 `task_info`，但它们**不 arrive empty**。
这不是漏写，而是依赖链保证的安全：

```
loader 读 task_info(m/n/k 坐标) ──▶ TMA 满 ──▶ MMA 读 task_info(shape) ──▶ tmem_full ──▶ epilogue 读 task_info 后才 release
```

即：任何 task 的 loader/MMA 一定在对应 `tmem_full` **之前**用完 task_info，而 epilogue 在
`tmem_full.wait`（`:986`）之后才 `release_task_info`（`:990`）。所以「最后一个用完元数据的消费者」
必然是 epilogue，由它一家 arrive empty 就已足够。这是一个用**执行顺序**省掉冗余 arrive 的设计。

调度器 `mainloop`（`:438-476`）还负责：先派 shared 专家的 L1/L2（`:447-451`），等 dispatch 的
`fetch_expert_recv_count`（`:455`），再按 `get_next_task()`（`:372-407`）产 routed task。
其中 L1/L2 交替调度有个**防死锁的 warmup**（`get_num_l1_warmup_waves`，`:18-45`）：先发够
`num_sched_l1_waves` 轮 L1 task，L2 才有资格被取（`:375-404`），避免 L2 等 L1 数据、而 L1
又被 L2 占了 CTA 的环形等待。`get_next_task()` 里还有一个显式等待「本 M block 所需的全部 L1 task
都已被各调度队列取走」（`:399-402`）。

## 5. 层 2 与跨 SM / 跨 rank 的同步

### 5.1 MMA → epilogue（TMEM 双缓冲）

MMA warp（`warp 6`，仅 leader CTA，`:819-938`）：

- 进每个 task 前 `tmem_empty_barriers[accum_stage].wait(accum_phase ^ 1)`（`:863-865`），
  确保这块 TMEM 上一轮被 epilogue 读空；`accum_stage = iter % 2`，`accum_phase = (iter/2)&1`（`:863-864`）；
- K 循环里 `full_barriers[stage].wait(phase)`（`:886`）等 TMA；然后 UTCCP 把 SFA/SFB
  搬进 TMEM（`:895-907`），发 `SM100_MMA_MXF8F6F4_2x1SM_SS::fma`（`:918`），
  累加地址是 `accum_stage * UMMA_N`（`:919`，两个累加器槽）；
- 每个 K block 结束时 `empty_barrier_arrive`（`:929`）：用 `umma_arrive_multicast_2x1SM`
  同时 arrive `empty_barriers[stage]`（释放 smem stage，`:874`）和——**最后一个 K block 时**——
  `tmem_full_barriers[accum_stage]`（`:877-878`）。注意 `tcgen05.commit` 天然带
  `before_thread_sync` 语义，所以注释（`:928`）说不需要额外的 `tcgen05.fence::before_thread_sync`。

epilogue（`warp >= 8`）在每个 task 开头 `tmem_full_barriers[accum_stage].wait(accum_phase)`
（`:986`）等 UMMA，`tcgen05_after_thread_sync()`（`:987`）后 `release_task_info`（`:990`）；
读完本 warpgroup 最后一个 store block 后 `tcgen05_before_thread_sync()` + `tmem_empty_barriers.arrive`
（`:1006-1008`）释放 TMEM 槽。

### 5.2 跨 SM 的 grid sync / 跨 rank 的 NVLink barrier

dispatch 组（warp 0-3）和 epilogue 组各自有独立的 grid sync 计数器（`kDispatchGridSyncIndex=0` /
`kEpilogueGridSyncIndex=1`，`:338-339`），避免互相冲突。`comm::grid_sync`（`comm/barrier.cuh:33-51`）
用 `atomic_add` + 一个 `0x80000000` 哨兵位做整卡 rendezvous；`grid_sync` 里的 `sync_scope`
回调是 `ptx::sync_aligned(kNumDispatchThreads, kDispatchBarrierIdx)`（`:393-396`）等
组内 named barrier。

真正跨 rank 的是 `comm::nvlink_barrier`（`comm/barrier.cuh:53-92`）：只有 SM 0
参与，通过对端 symmetric memory 上的信号计数器做 `red_rel_sys`，前后各套一次 `grid_sync`。
kernel 里在两处用它：pulling 前（`:422-428`，tag 1）与 workspace 清理后（`:683-689`，tag 2）。
dispatch 拉 token 用的 `pull_mbarrier`（`:434-436`）是本地 mbarrier，由 1D-TMA 的
`complete_tx` 触发（`:570-572`），与跨 rank 的 NVLink barrier 是两回事。

### 5.3 组内 named barrier

同一个 warp 组内（如 4 个 dispatch warp）用 `ptx::sync_aligned(nThreads, barrier_idx)`
做命名 barrier：dispatch 用 `kDispatchBarrierIdx=0`（`:317`），还有跨组的一次
`kDispatchWithEpilogueBarrierIdx=1`（`:318`，`sync_unaligned(kNumDispatchThreads+kNumEpilogueThreads, 1)`，
`:431`/`:623`/`:977`/`:1422`）——用于「dispatch 拉数据」与「epilogue combine / 清 workspace」
之间互斥。epilogue 内部用 `kEpilogueFullBarrierIdx=2`（`sync_aligned(kNumEpilogueThreads, 2)`，
`:1240`）与每个 warpgroup 的 `kEpilogueWGBarrierStartIdx + wg`（`:320`，`:1161`）。

### 5.4 L1→L2 的数据依赖：bitmask

`L2KBlockDependency`（`scheduler/mega_moe.cuh:140-166`）不是 mbarrier，而是 workspace 上的一个
`uint64_t` 位图：每个 L1 N block 完成时 `red_xor` 翻转一位（`:156-158`），L2 的某个 K block
轮询自己需要的位（`:161-165`）。每个 K block 用到的 L1 N block 子集由
`kNumL1BlockNsPerL2KBlock = BLOCK_K/(BLOCK_N/2)` 决定（`:145`，`BLOCK_N/2` 是 SwiGLU 后的 N）。
kernel 里 L2 loader 等它（`:723-726`）、L1 epilogue 翻转它（`:1246`）。这实现了「L1 的某个
N block 一算完，依赖它的 L2 K block 立刻能起步」的细粒度重叠。

## 6. H100 实测：mbarrier 多级流水的相位与重叠

端到端跑不了，但「mbarrier + `cp.async.bulk` 1D + warp specialization」这套同步机制是
SM90/SM100 通用的。实验 `src/mbarrier_pipeline.cu` 在 H100 上搭一条与 kernel §3 同构的流水：

- **probe**：全新 mbarrier 上探测 `try_wait.parity(0)/(1)`，验证 §3.2 的相位约定；
- **pipeline**：1 个 producer warp（`wait empty(phase^1)` → `cp.async.bulk` 1D → `arrive_expect_tx`
  full）与 1 个 consumer warp（`wait full(phase)` → 校验 → `arrive empty`）搭 `NSTAGES` 级流水，
  用 `advance_pipeline` 同款相位翻转；对比 `NSTAGES=1`（串行化，无重叠）与 `NSTAGES=4`（流水）。

在 `kernel_lab`（H100 sm90）运行，原始输出见 `harness/exp_mbarrier_pipeline.out.txt`：

```
NVIDIA H100 80GB HBM3, 9.0, 570.211.01
[probe] fresh mbarrier: try_wait(parity=0)=0 try_wait(parity=1)=1
[probe] after 1 arrive: try_wait(parity=0)=1 try_wait(parity=1)=0
[setup] per-stage=48 KiB iters=2048 total=96.0 MiB
[pipe NSTAGES=1 verify=1] best=5.580 ms  eff_bw=16.8 GiB/s  checksum_ok=yes
[pipe NSTAGES=4 verify=1] best=3.439 ms  eff_bw=27.3 GiB/s  checksum_ok=yes
[pipe NSTAGES=1 verify=0] best=2.287 ms  eff_bw=41.0 GiB/s  checksum_ok=yes
[pipe NSTAGES=4 verify=0] best=0.794 ms  eff_bw=118.1 GiB/s  checksum_ok=yes
```

几点可读的结论：

- **相位约定被实测坐实**：全新 mbarrier `try_wait(1)` 立即成功、`try_wait(0)` 阻塞 —— 正是
  loader 用 `empty.wait(phase^1)`、`phase` 从 0 开始（首次 = `wait(1)`，立即过）的原因；
  一次 arrive 后两位反转，与 cutlass 的生产者初值 phase=1 一致。
- **重叠有效**：全部拷贝 96 MiB。带校验（consumer 逐个元素比对，耗时在 consumer 侧）时
  串行 5.58 ms → 流水 3.44 ms（**1.62×**）；去掉校验只看 DMA 时 2.29 ms → 0.79 ms
  （**2.88×**，有效带宽 41 → 118 GiB/s）。校验版绝对带宽低是因为被 consumer 的逐元素
  比较主导；去校验版才是纯 DMA 延迟。单 warp 只发 4 个在途的 48 KiB 1D bulk，118 GiB/s
  是被**延迟**而非带宽卡住（每笔 ~0.39 µs），这正是要**多级 + 双缓冲**来填满在途请求的原因。
- checksum 全部 `yes`，说明这套 full/empty 相位在 producer 先行的多级场景下**数据正确**。

> 说明：真实 kernel 的 `kNumStages` 达 5–10 级（P3-1 §3），且 A/B 用 2-CTA multicast、
> 累加器另有 TMEM 双缓冲；本实验只是把**同步骨架**抽出来在 H100 上量到真实数字。

## 7. 为什么这样设计（小结）

- **两层解耦、深度不同**：TMA↔MMA 用深流水（`kNumStages`）扛 TMA 延迟；MMA↔epilogue 用浅
  双缓冲（`kNumEpilogueStages=2`）省 TMEM。段与段之间只靠 mbarrier 点对点握手，互不阻塞。
- **生产者相位从 1 开始**：buffer 初始为空，省掉一次「预 arm」；这与 cutlass 约定一致，实测确认。
- **task 槽由 epilogue 单点释放**：利用 `loader→full→MMA→tmem_full→epilogue` 的执行顺序，
  只需一个消费者 arrive，省掉三组各发一次的冗余同步。
- **细粒度依赖用位图而非 barrier**：L1→L2 用 `uint64_t` 掩码，按 N/K block 粒度唤醒，比粗粒度
  barrier 更早释放重叠机会。
- **寄存器按角色再平衡**（P3-1 §4.1）：dispatch/non-epilogue/epilogue 三档寄存器两两顶满 64512，
  让「同步密集」的 dispatch 与「寄存器密集」的 epilogue 各取所需。

下一篇（P3-3）进入计算本体：**tcgen05 MMA、UTCCP 的 SFA/SFB 搬入、SwiGLU 的 TMEM 原位处理**，
以及 FP8xFP4 的 scale 索引；之后 P3-4 讲 dispatch/combine 的 NVLink 读写与 locality domain。

配套实验：`src/mbarrier_pipeline.cu`、driver `harness/exp_mbarrier_pipeline.sh`、
输出 `harness/exp_mbarrier_pipeline.out.txt`。
