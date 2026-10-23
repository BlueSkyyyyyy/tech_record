# MPK 单 task device 实现走读（Hopper）

> 本文是 `docs/06-mpk-source.md`（运行层走读）的**下一层**：`docs/06` 讲清了
> 「一个常驻 kernel 内的 worker/scheduler 如何用原子队列 + 事件计数器把整图跑起来」，
> 但 worker 真正执行的是 `_execute_task()`（`persistent_kernel.cuh:1145`）——**每个 task 内部
> 具体是一段怎样的 device 代码**，`docs/06` 只到「调用生成的 `_execute_task`」为止。
> 本文补上这一层：以 **Hopper** 的 `tasks/hopper/*.cuh` 为例，走读「单 task = 一个 CTA」的
> 设备实现（warp specialization / mbarrier 流水 / TMA / wgmma / 融合残差），并说明它如何接入
> 编译层生成的分发代码。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06` 同一快照，所有行号都据此核对。
>
> 定位：`docs/01` §3 讲「编译器怎么把算子切成 task」，`docs/06` 讲「运行层怎么调度 task」，
> 本文讲「**task 内部怎么写**」。论文侧证据用 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`。
>
> 本文为**纯源码走读（资料分析）**：不重新构建/运行 mirage，所有结论标注文件与行号。

## 先记住一个画面

一个 task，就是**一支包工队承包一间厂房**。

- 厂房（一个 CTA）里有固定的工位：进料口（TMA）、机床（wgmma）、包装台（epilogue）；
- 包工队按图纸（编译期常量：`TILE_SIZE=128`、`Kstages`、swizzle）把设备摆好，然后开始流水：
  进料不停、机床不停、包装不停，三拨人靠**信号灯**（mbarrier）交接；
- 图纸不是现场画的——编译层在 `task_register.cc` 里按张量形状选好模板参数，生成一句
  `kernel::linear_kernel_hopper<...>(...)` 的调用，运行时只是照着调用。

> **类比的边界**：厂房的面积是**硬上限**——一个 task 的 smem 必须塞进 worker 那一份动态 smem
> 预算（sm90 离线模式约 207KB 减去保留区），塞不下就编译失败，不会「挤一挤」。包工队也不会
> 跨厂房协作：task 之间只通过显存里的数据与 event 打交道，不共享寄存器或 smem。
> 最后，这套「一个 task 一个 CTA」的写法是 Hopper 实现的惯例，不是编译器的硬性规定。

读完这一篇，你应该能回答：

1. 编译层是怎么把 `(TaskType, variant)` 变成一个具体模板调用的？哪些量必须在编译期定死？
2. 一个 `linear_kernel_hopper` 里，进料 / 计算 / 包装三拨人各自由谁驱动、用什么同步？
3. 换一种 task（attention、MoE、norm）时，哪些结构是共用的、哪些必须重写？

## 0. 从 tGraph task 到 device function：编译层如何生成调用

在 `docs/06` §1.1 里，编译层为每个 `(TaskType, variant_id)` 生成一个
`if (task_type == TASK_xxx ...) { <device 实现> }` 分支。这些 device 实现**不是**编译器凭空生成的
PTX，而是**对 `tasks/` 下预置 CUDA 模板函数的实例化调用**。以 Hopper 的线性层（`TASK_LINEAR_HOPPER`，
枚举值 `152`，`runtime_header.h:115`）为例，生成逻辑在 `src/kernel/task_register.cc` 的
`register_linear_hopper_task`：

1. **读取形状**（`task_register.cc:1181-1189`）：从输出张量取 `batch_size`/`output_size`，从输入
   张量取 `reduction_size`，从 `KNInputOp` 取 `output_stride`。
2. **选定编译期常量**（`:1194-1207`）：`TMA_CP_ASYNC_SIZE = 64`、`TILE_SIZE = 128`、
   **`Kstages = output_size >= 256 ? 3 : 6`**（K 方向 SMEM 流水级数，见 §2.3）、
   `SMEM_M_SIZE = batch_size`；`output_atom_size` 按 `output_size` 分档取 `16/32/64/128/256`
   （`:1203-1207`），`output_tma_cp_size = min(output_size,64)`（`:1202`）。
3. **生成 TMA 类型别名**：`TMA_A`（`:1208-1223`）、`TMA_B`（`:1225-1240`）、
   `TMA_OUT`（`:1262-1277`）、可选 `TMA_RESIDUAL`（`:1243-1259`），都是
   `kernel::tma::tma_2d<bfloat16, B, M, S, ...>` 的实例，把 gmem 形状、smem 形状、步长、
   swizzle（`B=M=S=3`）全部编进类型。
4. **生成调用**（`:1295-1311`）：`kernel::linear_kernel_hopper<bfloat16, batch_size, output_size,
   reduction_size, Kstages, TMA_A, TMA_B, TMA_OUT, TMA_RESIDUAL|void, output_stride>(tma_a, tma_b,
   tma_out, [&tma_residual])`。`tma_*` 构造函数从 `task_desc->input_tma_desc_ptrs[i][0]` /
   `output_tma_desc_ptrs[0][0]` 取出 `CUtensorMap*`（`:1279-1292`）——**这解释了为什么
   `TaskDesc` 要携带 TMA descriptor 指针**（见 `docs/06` §2.2）。
5. **注册 variant**：`with_residual` 决定注册为 `TASK_LINEAR_WITH_RESIDUAL_HOPPER`（`151`）或
   `TASK_LINEAR_HOPPER`（`152`）（`:1313-1318`）。

**粒度结论（对上论文 §4.1 的算子分解）**：一个 task = **一个 CTA**，计算
`BATCH_SIZE × OUTPUT_SIZE`（K 方向为 `REDUCTION_SIZE`）的一块输出；worker 在
`_execute_task` 分支里调用上面这个模板函数（`persistent_kernel.cuh:1126-1145`）。Hopper 上
`WORKER_NUM_THREADS = 256`（`persistent_kernel.cuh:75-77`），而该 task 自身只用
`THREADS_PER_WARPGROUP(128) × NUM_WARPGROUPS(2) = 256` 线程（`linear_hopper.cuh:53-56`），恰好铺满一个
worker block。SMEM 上限 `MAX_DYNAMIC_SHARED_MEMORY_SIZE` 在 sm90 离线模式为
`207*1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE`（`runtime_header.h:54-58`），
模板里有 `static_assert(TOTAL_SHARED_MEMORY <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)`
（`linear_hopper.cuh:130-131`）——**每个 task 必须把 SMEM 用量压在 worker 的同一份动态 SMEM 预算内**，
这正是论文 §5.3「paged shared memory」的动机（多 task 时分页复用，见 §5）。

> 实现差异：论文 §4.2（快照行 1126–1146）说 MPK 用 **Mirage superoptimizer 在 thread block 粒度
> 自动搜索**任务实现；线上 Hopper 实现里既有**手写模板**（`tasks/hopper/linear_hopper.cuh`），
> 也有 **CUTLASS 封装**（`tasks/cute/hopper/gemm_ws_mpk.cuh`，562 行）注册为
> `TASK_LINEAR_CUTLASS_HOPPER`（`runtime_header.h:120`）。可见线上实现已把「自动搜索」与
> 「手写/CUTLASS 模板」并行使用，`TaskRegister` 按 variant 选择——这是对论文叙述的工程化扩展。

## 1. 单 task 的设备实现：`linear_kernel_hopper`

`tasks/hopper/linear_hopper.cuh`（364 行）是 Hopper 线性层的核心设备函数。模板参数
（`:33-47`）：`T`、`BATCH_SIZE`、`OUTPUT_SIZE`、`REDUCTION_SIZE`、`Kstages`，以及四个 TMA 类型
（A/B/OUT/可选 RESIDUAL）和 `OUTPUT_STRIDE`。整体是一个**经典 warp-specialized producer-consumer
流水线**（与 DeepGEMM/Spector 的手写 GEMM 同源）：

- **两个 warpgroup**（`:53-56`）：`CONSUMER_WARPGROUPS = 1`（做 MMA）、`PRODUCER_WARPGROUPS = 1`
  （发 TMA），`NUM_WARPGROUPS = 2`，`warpgroup_id = warp_idx / 4`（`:85`）。Hopper 上一共 8 个
  warp（256 线程），按 `warpgroup_id == NUM_WARPGROUPS - 1`（`:215`）分出最后一个 warpgroup 当
  生产者，其余做消费者。
- **编译期常量**（`:49-77`）：`TILE_SIZE = min(REDUCTION_SIZE, SMEM_COL*SMEM_REPEAT_COL)`
  （`:49-52`，通常 128）；`INPUT_TMA_TILE_SIZE = 64`（`:60`，swizzle<3,3,3> 下每条 TMA 指令
  搬 64 列）；`OUTPUT_ATOM_SIZE = min(OUTPUT_SIZE,256)`（`:63`）；`SMEM_M_SIZE = BATCH_SIZE`
  （`:67`）；`NUM_ITER_N = ceil(OUTPUT_SIZE/OUTPUT_ATOM_SIZE)`（`:75-76`）、
  `NUM_ITER_K = ceil(REDUCTION_SIZE/TILE_SIZE)`（`:77`）。**注意：一个 task 的 N 方向可以有多
  个 `output_atom`（`NUM_ITER_N>1`），K 方向有 `NUM_ITER_K` 个 tile——这些都在这一个 CTA 内循环完成**。

### 1.1 手写 SMEM 布局与 swizzle

SMEM 用**手工预算的偏移量**切成若干缓冲（`:90-128`）：`shared_input`（A，`Kstages × M × TILE_SIZE`）、
`shared_weight`（B，`Kstages × TILE_SIZE × OUTPUT_ATOM_SIZE`）、可选 `shared_residual`、
`mm_output`（累加结果回写区），再把 mbarrier 数组（`input_barrier`/`weight_barrier`/
`residual_barrier`/`compute_done`/`residual_done`，各 `Kstages` 个）接在后面（`:184-197`）。
每个缓冲的地址在运行时通过 `set_ptr()` 在 `Kstages` 个槽位间轮转（`:248-249,254-255,281-283`）。

访问 SMEM 的是 `smem_tma<T,B,M,S,ROW,INNER_COL,OUTER_COL>`（`smem_layout_tma.cuh:26`），
它把逻辑索引映射到 **128B swizzle** 后的物理地址：`get_swizzled_offset`（`:51-80`）用
`B=M=S=3` 做 `icol ^= irow` 的 bank 冲突消除，`operator()/at()`（`:82-119`）提供 3D/2D/1D 访问。
`mma_descriptor`（`wgmma.cuh:47-91`）再把 `smem_tma` 的裸指针编码成 **wgmma 的 matrix descriptor**
（含 swizzle mode 位，`:52-70`），供 `wgmma.mma_async` 直接读 SMEM。**这套「类型携带 swizzle」的
写法和 CUTLASS 的 `GMMA::Layout` 思路一致**。

### 1.2 生产者 warpgroup（TMA 预取）

生产者只由 **一个 warp 的 lane 0** 驱动（`:218`，`lane_id()==0 && warp_idx == NUM_WARPGROUPS*4-4`），
按 `(output_atom_idx, i)` 双层循环下发 TMA：

- **残差预取**（`:224-237`）：每轮先 `wait(residual_done[slot], phase^1)`，再
  `set_barrier_transaction_bytes(...)` + `tma_residual->tma_cp_async(...)` 把残差 tile 搬进 SMEM。
- **A/B 预取**（`:239-260`）：对每个 K tile，先 `wait(compute_done[slot], phase^1)`——**等消费者用完
  该槽位**再复用，然后 `tma_a.tma_cp_async`（A，坐标 `{i*TILE_SIZE,0}`）与 `tma_b.tma_cp_async`
  （B，坐标 `{i*TILE_SIZE, output_atom_idx*OUTPUT_ATOM_SIZE}`）。

`set_barrier_transaction_bytes` 用 `mbarrier.arrive.expect_tx` 告诉 mbarrier「本槽位会收到 N 字节」
（`barrier.cuh:49-66`），TMA 搬运完成的字节数到达后自动满足该 barrier。这就是 **TMA + mbarrier
的 transaction-count 同步**，生产者无需逐个数据确认。**关键：流水跨 `output_atom_idx` 也保持**——
预取一直跑到填满 `Kstages` 深度的槽位，实现 K 方向与 N 方向的双重软件流水。

### 1.3 消费者 warpgroup（wgmma + 回写）

消费者 warpgroup（`:263-358`）对每个 `output_atom_idx`：

1. **清零累加器** `s_frag[OUTPUT_ATOM_SIZE/2]`（`:266-272`，`clear_8_floats`）。
2. **K 循环**（`:274-308`）：`wait(input_barrier[slot], phase)` + `wait(weight_barrier[slot], phase)`
   等 TMA（`:278-279`）；构造 `A_DESC a_desc(input_smem(0,0))` / `B_DESC`（`:284-285`）；
   `warpgroup_fence_fragment` → `warpgroup_arrive`（`wgmma.fence.sync`，`:287-288`）→
   `wgmma::mma<bfloat16,64,OUTPUT_ATOM_SIZE,16,...>`（`:290-299`，底层按 N 分档调用
   `wgmma.mma_async.m64n{N}k16.f32.bf16.bf16`，`wgmma.cuh:798-849`）→ `mma_commit_group` +
   `mma_async_wait`（`wgmma.commit_group` / `wgmma.wait_group 0`，`:300-301`）→
   `warpgroup_fence_fragment`（`:302`）。**注意这里用了 `mma_async_wait()`（wait_group 0）**，
   即每个 K tile 都同步等 WGMMA 完成——实现简单但流水深度受限（见 §6 观察）。
3. **翻转 compute_done**（`:305-307`）：`if (idx_in_warp==0 && warp_idx%4==0) arrive(compute_done[slot],1)`
   ——通知生产者该 A/B 槽位已释放。
4. **融合残差 + 写回**（`:310-340`）：若 `HAS_RESIDUAL`，先 `wait(residual_barrier[...])`，再把
   `s_frag` 转 bf16 并**在寄存器里加上 `residual_smem.at(row,col)`**（`:331-339`），否则直接写。
   逐元素写入 `mm_output_smem.at(row,col)`——用的是 WGMMA m64n16 的**寄存器 fragment 布局**
   （`:329-330` 的 `row/col` 映射，注释链到 PTX 文档，`:321`）。
5. **TMA 回写**（`:342-357`）：`async_proxy_fence()`（`fence.proxy.async.shared::cta`，保证 generic
   proxy 的 SMEM 写对 TMA async proxy 可见，`tma.cuh:25-27`）→ `wg_sync<128*1>(1)`（warpgroup 内
   bar.sync，`utils.cuh:30-36`）→ `tma_out.tma_store_async(...)` + `store_commit_group()`。

**残差融合是 task 内融合的示例**：`linear + residual add` 两个「算子级」动作被编进**同一个
device 函数**（`HAS_RESIDUAL` 编译期开关），中间结果 `s_frag` **只在寄存器**，没有落 HBM——
这正是 `docs/00` §1「损失三：中间张量 HBM 往返」在单 task 内的消除，也是 `TASK_LINEAR_WITH_RESIDUAL_HOPPER`
被单列为一个 task 类型的原因。

## 2. 支撑原语：barrier / TMA / wgmma / register realloc

- **mbarrier**（`tasks/hopper/barrier.cuh`）：`struct Barrier` 只是一个 8 字节 SMEM 槽（`:20-28`）；
  `initialize_barrier`（`mbarrier.init.shared::cta.b64`，`:33-47`）、
  `set_barrier_transaction_bytes`（`mbarrier.arrive.expect_tx`，`:49-66`）、
  `wait`（`mbarrier.try_wait.parity` 自旋，`:68-85`）、`arrive`（`:87-96`）、
  `try_wait_barrier`（非阻塞版，`:99-114`）。全部包在
  `#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)` 里，其它架构编译成
  `brkpt`（`:44-46`）——**task 代码是架构特化的**，sm80 走 `tasks/ampere/`。
- **TMA 辅助**（`tasks/hopper/tma.cuh`）：`async_proxy_fence`（`:25-27`）、
  `store_commit_group`（`cp.async.bulk.commit_group`，`:29-31`）、
  `store_async_wait`（`cp.async.bulk.wait_group`，`:33-36`）、
  `prefetch_tma_descriptor`（`prefetch.tensormap`，`:38-41`）。
- **wgmma 封装**（`tasks/hopper/wgmma.cuh`）：`mma_descriptor`（`:47-91`）、`wgmma_m64n8/16/32/64k16` 的
  PTX 内联（`:102` 起）、`mma` 模板按 `SMEM_A::COL/K` 展开（`:795-849`）、
  `warpgroup_arrive`/`mma_commit_group`/`mma_async_wait`/`warpgroup_fence_fragment`
  （`:1559-1577`）。
- **寄存器重分配**（`tasks/hopper/utils.cuh`）：`wg_decrease_regs<RegCount>` /
  `wg_increase_regs<RegCount>` 包 `setmaxnreg.dec/inc.sync.aligned.u32`（`:19-46`），
  用于让生产者 warpgroup 让出寄存器给消费者。**但在 `linear_hopper.cuh` 里这两处调用被注释掉了**
  （`:217` `wg_decrease_regs<32>()`、`:265` `wg_increase_regs<160>()`）——说明默认路径没有做
  warpgroup 级寄存器再平衡，与本机实测「线性层对寄存器需求」的取舍有关（见 §6）。
- **warpgroup barrier**：`wg_sync<GROUP_THREADS>(barrier_id)` 包 `bar.sync`（`utils.cuh:30-36`），
  用于消费者内部回写与 TMA store 之间的跨 warp 同步（`:347`）。

配置常量在 `tasks/common/worker_config.h`：`NUM_THREADS_PER_WARP = 32`（`:19`）、
`WARPGROUP_WARPS = 4`（`:21`）、`CONSUMER_NUM_THREADS = 128`（`:28`）。**一个 warpgroup = 4 warp =
128 线程**是贯穿全部 task 的基本单位。

## 3. 其它 task 形态：同一套 task 框架下的异构实现

`tasks/` 下并非所有 task 都长成 GEMM 那样。对比三种典型形态，可以看清「task 粒度 = 一个 CTA」
下各算子各写各的：

### 3.1 逐元素/归约类（`rmsnorm_hopper.cuh`、`silu_mul_hopper.cuh`）

- **RMSNorm**（`rmsnorm_hopper.cuh`，162 行）：`rms_norm_hopper_impl<T,BATCH,HIDDEN,NUM_THREADS=256>`
  （`:19-23`）。用 **cp.async 双缓冲**搬输入/权重（`:81-108`，`cp_async_fence`/`cp_async_wait<1>`），
  然后每 tile 求平方和（`:112-115`）→ warp shuffle + SMEM 两级归约（`:118-134`）→
  `rsqrt`（`:136`）→ 逐元素缩放写回（`:138-158`）。**没有 TMA、没有 wgmma、没有 producer/consumer
  分工**，就是全线程参与的向量流水。
- **SwiGLU**（`silu_mul_hopper.cuh`，45 行）：`silu_mul_task_impl_hopper`（`:25-43`）是最简形态——
  `if (threadIdx.x >= CONSUMER_NUM_THREADS) return;`（`:27-29`）把超过 128 的线程挡掉，其余线程
  按 `num_active_tokens*OUTPUT_SIZE` 网格 stride 循环逐元素算 `x*sigmoid(x)*mul`（`:33-42`）。
  **它连 SMEM 都不用**，直接 gmem→gmem，`num_active_tokens` 作为运行期参数（对上 `docs/06`
  的 `TaskDesc` 运行期语义）。

对照结论：**逐元素算子是「纯数据并行标量循环」，重算子（GEMM/attention）才是 warp-specialized
流水线**。task 框架对两者的唯一约束是「一个 CTA、共享同一份动态 SMEM 预算、以 `_execute_task`
被调用」。

### 3.2 通信类（`allreduce.cuh`）

`nvshmem_tile_allreduce<T,BATCH,OUTPUT_SIZE,OUTPUT_STRIDE>`（`allreduce.cuh:29-63`）把
AllReduce 写成**一个 task**：用 NVSHMEM 的 `nvshmemx::tile_sum_reduce_block<..., NVLS_ONE_SHOT_PULL_NBI>`
（`:52-57`）在 team 上做 tile 归约，输入/输出张量用 `nvshmemx::Tensor`（`:44-47`）包成带 layout 的
视图。**这正是论文 §5.1「计算 task 与通信 task 用同一套 event 驱动表示、无需拓扑画像」的落地**：
通信 task 和计算 task 一样被 worker 取出执行、完成后触发事件（`docs/06` §3.4）。

### 3.3 融合类（`norm_linear_hopper.cuh`）

`norm_linear_kernel_hopper`（`norm_linear_hopper.cuh:44-49`）把 **RMSNorm + Linear** 融合进一个
task：SMEM 里同时放 `input`/`norm_weight`/`linear_weight`/`mul_output`/`elementary_unary_output`/
`mm_output`/`reduction_output` 多个缓冲（`:79-105`），一个 CTA 内先归一化再 GEMM。这与
`linear_hopper` 的残差融合是同一思路的更大范围版本（**段内多算子融合**），对应 `docs/00` §4
谱系的 T2（段级融合）。

## 4. 与运行层的接缝

把本文与 `docs/06` 拼起来，`task` 的完整生命周期是：

1. **编译期**：算子分解 → 每块输出一个 task → `TaskRegister` 生成
   `kernel::linear_kernel_hopper<...>(...)` 调用（`task_register.cc:1295-1311`），编进
   `_execute_task` 的 `if` 分支。
2. **初始化**：`init_persistent_kernel` 把 task/event 表、TMA descriptor 指针 `cudaMemcpy` 到显存
   （`docs/06` §5.1）。
3. **运行期**：worker 拉取 `TaskDesc`（含 `input_tma_desc_ptrs`）→ `_execute_task` 实例化 TMA 对象
   → 进入本文 §1 的 producer/consumer 流水线 → 完成 → worker 触发 trigger_event
   （`persistent_kernel.cuh:1155` 起，`docs/06` §3.4）。
4. **task 内同步**（mbarrier / `bar.sync`）与 **task 间同步**（设备内原子环形队列 + 事件计数器，
   `mpk_atoms.cuh`）**是两套完全不同的机制**：前者在 CTA 内部，后者跨 CTA。**跨 task / 跨算子的
   软件流水由运行层负责**（论文 §5.3 cross-task software pipelining，快照行 1326–1363），
   task 内部只做 CTA 内的 K/N 维流水。

## 5. 与论文逐项对照

| 论文机制（快照章节） | 源码落点 | 备注 |
|---|---|---|
| thread-block 级任务实现生成（§4.2，行 1126–1146） | `task_register.cc` 生成模板调用；`tasks/hopper/*.cuh` 手写实现 | 线上混合「手写模板 + CUTLASS 封装」，超出纯 superoptimization |
| intra-SM 软件流水 / 寄存器复用 / 消除 bank conflict（§4.2，行 1146） | `linear_hopper.cuh` 的 K/N 双重流水（`:219-260`）；`smem_tma` swizzle（`smem_layout_tma.cuh:51-80`） | 一致 |
| paged shared memory（§5.3，行 1312–1325） | 每个 task 手工按 `Kstages` 分页 + `set_ptr` 轮转（`:90-128,248-255`）；`static_assert` 压预算（`:130-131`） | task 内已是「槽位轮转」，运行层可分页跨 task 复用 |
| cross-task software pipelining（§5.3，行 1326–1363） | 由运行层 worker 负责，task 内部不可见 | 本文只覆盖 task 内流水 |
| 计算/通信统一表示（§5.1） | 通信 task `allreduce.cuh` 与计算 task 同构 | 一致 |

**观察到的实现延伸**：

- `linear_hopper.cuh` 的消费者在每个 K tile 后都 `mma_async_wait()`（`:301`，wait_group 0），
  **没有把多个 WGMMA group 在途**，因此 task 内的 MMA 级流水深度较浅；这是「简单正确优先」的实现，
  与 CUTLASS 手写 GEMM 常做的多 group 在途有差距。真正的跨 task 重叠交给运行层，而非 task 内堆深度。
- 生产者只用一个 warp 的 lane 0 发 TMA（`:218`），**其余 3 个 warp 空转**——生产者 warpgroup 是
  「TMA 发射器」而非计算单元；这与 DeepGEMM mega 家族「静态 warp 角色」的思路一致（`docs/02` §1）。
- 寄存器重分配被注释（`:217,265`），说明默认配置下没有做 warpgroup 间寄存器再平衡（口径留待
  后续在有构建环境时验证）。

## 6. 判据映射（沿用 `docs/00` §3.2）

单看 task 层：每个 task 是**一个 CTA 的一段 device 代码**，本身只贡献 **C3（覆盖多算子）**——
不同 `TaskType` 对应不同 device 实现。**C1/C2（单次 launch / 常驻）由运行层给**，
**C4/C5（设备内协调 / SM 级调度）由 worker/scheduler 给**（`docs/06` §7）。task 内部那套
mbarrier/TMA/wgmma 只是**单个 CTA 内的经典 warp specialization**，不是 megakernel 的判据；
megakernel 的「SM 级调度」体现在**运行层把一个 task 指派给一个 SM（worker block）**、以及
`split_worker_scheduler` 下 worker 占物理 SM 的划分上（`utils.py:25-29`，`docs/06` §1.3）。

## 7. 本轮小结

- `docs/06` 停在「worker 调用 `_execute_task`」；本文补上 task 内部：**编译层把每个
  `(TaskType, variant)` 生成为一个对 `tasks/<arch>/*.cuh` 模板函数的实例化调用**
  （`task_register.cc:1295-1311`），**一个 task = 一个 CTA（Hopper 256 线程）**计算一块输出。
- 重算子 task 是 **warp-specialized producer-consumer 流水线**：生产者 warpgroup 用 **TMA +
  mbarrier transaction-count** 预取 A/B/残差（`linear_hopper.cuh:219-260`），消费者用
  **wgmma + swizzled SMEM descriptor** 做 MMA（`:274-308`），**残差在寄存器内融合**
  （`:331-339`），最后 TMA store 回写（`:342-357`）。
- 逐元素/通信/融合 task 形态各异（`rmsnorm`/`silu_mul`/`allreduce`/`norm_linear`），但都遵守
  「一个 CTA、共享动态 SMEM 预算、以 `_execute_task` 被调用」的统一契约——**task 层是异构实现的
  公共抽象面**。
- **task 内同步（mbarrier）与 task 间同步（原子队列 + 事件计数器）是两套机制**；
  论文 §5.3 的 cross-task software pipelining 由**运行层**负责，不是 task 内部完成的。
- 论文 §4.2 的「纯 superoptimization 生成 task」在线上已演化为**手写模板 + CUTLASS 封装 +
  TaskRegister 按 variant 选择**（`docs/06` 的双 kernel 开关同理）。
- 与生产案例对照：DeepGEMM mega 家族是**手工静态角色 + 设备内异步 barrier + SMEM 环形流水**的
  段级 T4（`docs/02` §1），本文的 `linear_hopper` 是其**单 task 尺度**的对应物；MPK 把这些 task
  用通用 event queue 串成整图 T4。

## 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
- task 目录：`include/mirage/persistent_kernel/tasks/{hopper,ampere,blackwell,cute,common}/`；
  编译层 `src/kernel/task_register.cc`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.2 行 1126–1146、
  §5.3 行 1312–1368）。
- 专题内交叉引用：`docs/00` §1/§3/§4（损失/判据/谱系）、`docs/01` §3/§4（编译器/运行层精读）、
  `docs/03` §3（in-kernel 调度）、`docs/06`（运行层源码走读）。
- 生产案例：`docs/02` §1（DeepGEMM 段级 mega-kernel）。
