# MPK 自动路径（Blackwell 版）源码走读：Mirage transpiler 如何用 2SM MMA + TMEM + cluster 生成 B200 kernel

> 本文是 MPK 源码走读链条里 **`docs/18`（transpiler 通用/Hopper 骨架）** 的下一代硬件补篇。
> `docs/18` §8 明留了一条尾巴：
>
> > **Hopper/Blackwell 只给骨架**：本文以通用/Ampere 版为主线；Hopper 的
> > `hopper_async_pipeline` / TMA / `Hopper_Matmul`、Blackwell 的 `2sm_mma` / `launch_kernel_on_cluster`
> > 只标出调用点与行号，未逐行走读其内部。
>
> 其中 Hopper 的一半已由 `docs/07`（单 task device 实现，TMA + wgmma warp specialization + `Kstages`
> 双重流水）覆盖；**本文补的是 Blackwell 的一半**——`transpiler_tb_blackwell.cc`（1575 行）+
> `plan_tb_swizzle_blackwell.cc`（177 行）+ 运行期 `Blackwell_Matmul` / `BlackwellAsyncPipeline` /
> `InputTMAAsyncCopy_Blackwell` 三个 `tb::` 模板。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`–`docs/20` 同一快照，所有行号据此核对。
> CuTe/CUTLASS 侧引用快照自带的 `deps/cutlass`（本机 `f3fde583`）。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（「快照论文」，§4.2 行 1126–1146）。
>
> 本文为**纯源码走读（资料分析）**：Blackwell（SM100）在本机（目标卡 H100/sm90）**不可执行**，故不构建、
> 不运行、不引用任何性能数——与 `docs/17`（多卡，需 2 张 NVLink 卡）同一口径。相关机制的 sm90 投影实测
> 见 `docs/04` §18–§23（Thread Block Cluster / DSM / TMA multicast / 2D cluster / 2SM 的 sm90 投影）。

## 0. 定位：Blackwell 后端在编译栈里的坐标

`docs/18` §3 给出 `transpile_ugraph` 按 `config.target_cc` 三分派：

```
KN_CUSTOMIZED_OP
  ├─ transpile_kn_custom_op_hopper    (H100)   transpiler_tb_hopper.cc:303
  ├─ transpile_kn_custom_op_blackwell (B200)   transpiler_tb_blackwell.cc:350   ← 本文
  └─ transpile_kn_custom_op           (其它)   transpiler_tb.cc:253
```

Blackwell 版与 Hopper/通用版**共享同一套前端**（`Transpiler` ctor 的图归一化、五个 resolve、
`get_threadblock_schedule`、融合链），差异**全在「一个 customized op 生成什么 kernel」**这一段。
一句话概括本文的结论：

> Hopper 路径是**单 CTA 的 warp-specialized wgmma 流水**（producer warpgroup 发 TMA、consumer
> warpgroup 跑 GMMA，`PipelineTmaAsync`）；Blackwell 路径把它整体升格为**一对 CTA 协作的 2SM UMMA**：
> 两个 SM 绑成一个 cluster 计算单元，累加器放 **TMEM**，TMA 载入带 **multicast**，跨 CTA 用
> `tcgen05.commit.cta_group::2 + multicast` 的 **mbarrier 会合**收尾。

## 1. 入口与硬约束：`transpile_kn_custom_op_blackwell`（`:350`）

生成一开始就钉死三个前提（`:368-377` 的断言）：

1. **`config.target_cc == GPU_CC::B200`**——这段代码只对 Blackwell 生效；
2. `config.num_consumer_wgs + config.num_producer_wgs <= config::MAX_NUM_WARP_GROUPS`；
3. `num_threads == (num_consumer_wgs + num_producer_wgs) * 128`（当 `forloop_range > 1`）——
   即**每 warpgroup 128 线程**，producer 与 consumer 各占若干 warpgroup。

入口随后照 `docs/18` §4 的三件套走（行号对齐 `docs/18` §4）：

```
TBSched sched = get_threadblock_schedule(g);                  // :385
get_threadblock_swizzle_plan_blackwell(g, sched);             // :387
TBMemoryPlan mem_plan = get_threadblock_memory_plan(g,sched,true,true); // :390
```

注意与通用版的差别：内存计划显式带 `(true,true)`（Hopper/Blackwell 两个标志），因为 Blackwell 多出
**TMEM 基址**与 **mbarrier 缓冲**两种需要预留的 SMEM 区域（见 §3）。

## 2. 集群与 2SM：编译期把「一对 CTA」设成计算单元

### 2.1 `cluster_shape` 与 `elect_one_cta`

生成代码开头声明集群形状（`:410-416`）：

```c
auto cluster_shape = make_shape(Int<cx>{}, Int<cy>{}, Int<cz>{});  // g.cluster_dim
uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
int cta_rank = cute::block_rank_in_cluster();
```

`g.cluster_dim` 的默认值是 `{4,4,1}`（`include/mirage/threadblock/graph.h:200`）。随后在 matmul 定义处
构造集群布局并选出「leader CTA」（`:519-526`）：

```c
Layout cluster_layout_vmnk = tiled_divide(make_layout(cluster_shape),
                             make_tile(typename decltype(tiled_mma)::AtomThrID{}));
auto cta_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank);
auto elect_one_cta = get<0>(cta_in_cluster_coord_vmnk) == Int<0>{};
```

`AtomThrID` 是 2SM MMA 原子的线程 ID 形状（`Shape<_2,_1,_1>`，见 §2.2 的 `mdspan`），把它 tile 进
cluster 后，`cta_in_cluster_coord_vmnk` 的第 0 模（`V`，即 pair 内位置）为 0 的 CTA 就是**每对里的 leader**。
`elect_one_cta` 即「我是 leader」。`get_cluster_layout` / `get_mma_coord_vmnk` / `get_cta_mma` 三个设备侧
辅助函数在 `include/mirage/transpiler/runtime/threadblock/utils.h:94-118`（用 `blockIdx.x % cluster_size`
等把全局 block 坐标折回 cluster 内坐标）。

### 2.2 `TiledMMA`：2SM 的选型

真正的关键在 `:504-512`：

```c
auto tiled_mma =
  cutlass::gemm::collective::detail::sm100_make_2sm_trivial_tiled_mma<
      A, B, C, Shape<Int<2*m>, Int<n>>, decltype(cluster_shape),
      UMMA::Major::K, UMMA::Major::MN>();
```

这个 CUTLASS 工厂（`deps/cutlass/include/cutlass/gemm/collective/builders/sm100_common.inl:373-429`）
按数据类型选**2SM MMA 原子**——TF32 → `SM100_MMA_TF32_2x1SM_SS`（`:385`）、fp16/bf16 →
`SM100_MMA_F16BF16_2x1SM_SS`（`:391`）、int8 → `SM100_MMA_S8_2x1SM_SS`（`:396`）、fp8/fp6/fp4 →
2x1SM 的 `SM100_MMA_F8F6F4_2x1SM_SS`（`:409-422`）。它的硬约束写在 `:377`：

```cpp
static_assert(M == 128 || M == 256, "Invalid TileShape_M.");
```

即 **2SM MMA 的 M 必须是 128 或 256**（一对 SM 共同吃 256 行 M，各持一半）。该函数在 mirage 里被调了
**两处**：生成 kernel 内（设备侧，`:504`）与 host 侧声明 TMA（`transpiler_kn.cc:747-761`）——因为
TMA descriptor 的构造依赖 `TiledMMA`，必须在 host 端也定义一份（见 §9）。

`mma_tiler` 的 K 维取 `k/16` 个 16 元素 block（`:514-517`），供 CuTe 的 `partition_shape_A/B` 用。

## 3. TMEM 分配 + MMA mbarrier：2SM 的两种「新内存」

Blackwell 比 Hopper 多两类资源，都在 kernel prologue 里初始化（`generate_Tmem_mbarrier_init_code`，
定义 `:319-345`，调用 `:443-452`）：

```c
using TmemAllocator = cute::TMEM::Allocator2Sm;
TmemAllocator tmem_allocator{};
if (elect_one_warp) {
  tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, tmem_base_ptr);
}
__syncthreads();
// MMA mbarrier Initialization
if (elect_one_warp && cute::elect_one_sync()) {
  cute::initialize_barrier(*mma_barrier_ptr, arrive_cnt);
}
uint16_t consumer_sync_mask = static_cast<uint16_t>((1u << size(cluster_shape)) - 1u);
```

三件事：

- **TMEM（Tensor Memory）**：Blackwell 的累加器住在每 SM 的一块专用 tensor memory 里（不像 Hopper 在
  寄存器 / SMEM）。`Allocator2Sm` 是**成对 SM 的分配器**，按列分配 `Sm100TmemCapacityColumns`；
- **MMA mbarrier**：`arrive_cnt = forloop_range * cluster_dim.x*y*z / 2`（`:446-450`）——即「每对 CTA
  每次 K 迭代一次 MMA 提交」的总次数，`/2` 正是 2SM 一提交顶两 CTA；
- **`consumer_sync_mask`**：cluster 内所有 CTA 的位掩码（`:340-341`），即 multicast 的投递目标集合。

SMEM 布局按 guid 区分三类（`:428-441`）：`tmem_base_ptr_guid` → `uint32_t*`（TMEM 基址）、
`>= mbarrier_buf_guid_offset` → `uint64_t*`（mbarrier 数组）、其余 → 普通 stensor 指针。TMA 的 mbarrier
空间在计划末尾按 `tmaParamsList.size() * pipeline_stages * 16` 追加（`:883`）。

## 4. 输入：TMA multicast 载入 + `BlackwellAsyncPipeline`

### 4.1 G→S 拷贝的三档（与通用/Hopper 同构）

`docs/18` §5.3 的三档在这里同样存在（`:649-808`）：非 chunked 同步拷贝
（`InputNonChunkedSyncCopy`）、chunked 同步（`InputChunkedSyncCopy`）、以及**chunked + async → TMA
流水**（`InputTMAAsyncCopy_Blackwell`）。判据来自 `op_meta.is_chunked_input` /
`is_pipelined_input`（`:649-651`），与通用版一致。

### 4.2 `BlackwellAsyncPipeline`（`blackwell_pipeline.h`）

每个 pipelined 输入建一个流水对象（生成点 `:731-746`，模板定义
`include/mirage/transpiler/runtime/threadblock/blackwell_pipeline.h:11-92`）：

```cpp
using MainloopPipeline = cutlass::PipelineTmaUmmaAsync<Stage, ClusterShape, AtomThrShape_MNK>;
```

注意流水类型是 **`PipelineTmaUmmaAsync`**（CUTLASS `pipeline/sm100_pipeline.hpp:537`）——比 Hopper 用的
`PipelineTmaAsync`（`pipeline.h:24`）多了 **UMMA** 二字，因为 Blackwell 的消费者是 `tcgen05` MMA，它的
**到达/放行由 `tcgen05.commit` 驱动**（见 §7）。构造参数 `(transactionBytes, producer?, consumer?,
leader?=wg0 且 leader CTA, num_consumer_threads)` 见 `blackwell_pipeline.h:38-46`，构造尾部
`pipeline_init_arrive_relaxed / pipeline_init_wait` 用 cluster 大小做初始会合（`:54-55`）。对外接口
`producer_acquire/producer_advance/producer_commit` 与 `consumer_wait/consumer_release`（`:67-91`）。

### 4.3 `InputTMAAsyncCopy_Blackwell`（`input.h:256-395`）

单线程（`lane_id()==0`）发 TMA（`:312`）。值得单看的是 **multicast mask 的投影**（`:352-360`）：

```cpp
if constexpr (MInput) tma_mcast_mask = create_tma_multicast_mask<2>(cluster_layout_vmnk, cta_coord);
else                 tma_mcast_mask = create_tma_multicast_mask<1>(cluster_layout_vmnk, cta_coord);
```

即 **A 矩阵沿 N 模广播、B 矩阵沿 M 模广播**（2SM 的 A 被同 N 的所有 CTA 共享、B 被同 M 的所有 CTA
共享）。TMA 分区用 `tma_partition` 按 cluster 内的 N（A）或 M（B）坐标切（`:362-384`），最后
`copy(tma.with(*tma_barrier, tma_mcast_mask), ...)`（`:388-390`）。这正是 `docs/04` §19/§20「TMA
multicast 一次读、C 个 SMEM 收」在编译器里的生成点，也是 DeepGEMM Mega MoE「Always multicast on A」
选择「被多 CTA 共享、最省源带宽」那一路的同类机制（`docs/02` §1.5）。

## 5. 数据布局：`partition_shape` + `sm100_smem_selector` + `tile_to_mma_shape`

`Blackwell_Matmul`（`blackwell_matmul.h:40`）比 `Hopper_Matmul` 多出一整套「UMMA 形状」推导：

- 主序 `UMMA::Major::K`（A）/ `UMMA::Major::MN`（B）（`:50-51`）；
- SMEM 原子用 `sm100_smem_selector`（`:56-76`）按 `mma_tiler` 的 `(M,K)` / `(N,K)` 选；
- `partition_shape_A/B`（`:62-64`、`:77-79`）把 MMA 原子形状投影到 SMEM 切片；
- `UMMA::tile_to_mma_shape`（`:66-69`、`:81-84`）把它 tile 成带 `PIPELINE_STAGE` 的流水布局
  （A 用 `Step<_1,_2,_3>`、B 用 `Step<_2,_1,_3>`）；
- 输入侧 `InputTMAAsyncCopy_Blackwell` 复用同一套 `DstPipeLayout`（`input.h:294-297`），保证 producer 写
  与 consumer 读命中同一片。

全局 N 是 `size<1>(ClusterShape) * N`（`:94`）——即 **cluster 沿 N 维把输出拼宽**。

## 6. swizzle：`plan_tb_swizzle_blackwell.cc`

与通用版（`docs/18` §4.2）结构相同（先算 `innermost_min_chunk_size`，再 XOR/shift），但两处
Blackwell 特征：

- **chunk 下界提到 16B**：`num_16B_elems = get_num_elems_in_16B(dtype)`（`:54/62/71`），并断言
  `chunk_size_bytes <= 16`（`:114-115`）——因为 UMMA 的 SMEM 访存粒度以 16B 为单位；
- **`num_chunks_in_128B` 固定用 128B**（`:122-123`）——Hopper 的 `swizzle<B,M,S>` 与 UMMA 的
  `sm100_smem_selector` 对齐到同一 bank 宽度。

XOR 分支（`:125-140`）产出 `meta.is_xor_swizzled / xor_swizzle_{b,m,s}`，shift 分支（`:141-172`）重排
stride，逻辑与通用版逐行一致（`docs/18` §4.2 已详述，本文不重复）。

## 7. 主循环：producer 发 TMA / consumer 跑 2SM UMMA

### 7.1 warp specialization（`:1036-1078` + `:1465-1508`）

```c
if (warpgroup_id == num_consumer_wgs) {          // 最后一个 warpgroup = producer
  if (tb::warp_id_in_wg() == 0) {
    for (for_idx = 0; for_idx < forloop_range; ++for_idx)
      STensor<..>InputAtom::run(tma_<..>, stensor<..>_ptr, tiled_mma, mma_tiler,
                                for_idx, blackwell_async_pipeline_<..>);
  }
} else {                                          // 其余 = consumer
  for (for_idx ...) { ...sched.loop_nodes... }
}
```

producer 只有每个 warpgroup 的 warp0 发 TMA（`:1053`）；consumer 跑主循环（`:1465-1508`）。这与
Hopper 版（`docs/18` §5.4 / `docs/07` §1.2）结构相同，差别在 consumer 的「算」换成了 UMMA。

### 7.2 2SM 的 `elect_one_cta && elect_one_warp` 门

循环内每个节点被 `add_cta_warp_selector_if_need(code, is_in_loop, true, true)` 包一层（`:1143-1146`，
定义 `:274-288`）：

```c
bool use_2sm_mma = true;                     // TODO(zy): support other cases such as 1sm mma
if (use_2sm_mma && is_in_loop) {
  if (elect_one_cta && elect_one_warp) { ... }
}
```

即循环体内的算子**只由 leader CTA 的 warp0 执行**——这与「UMMA 单线程发指令」一致（§7.3），也是 2SM
M 被一对 CTA 分摊时「只在一处提交、另一处被动收」的必然形态。**注**：循环里若出现非 MMA 的逐元素/reduce
算子，也会落进这个单 warp 门内，是读码可见的一个隐含约束（§13 局限）；实际的 matmul 融合链把逐元素
算子编成 epilogue 模板（§8），并不作为独立循环节点。

### 7.3 MMA 提交与跨 CTA 到达（`:1180-1205`）

matmul 节点先等输入流水（`add_loop_node_consumer_wait_if_need`，`:290-317`），再：

```c
Matmul<..>Kernel::run(matmul_<..>_accum, stensor<..>_ptr, stensor<..>_ptr,
                      for_idx, tiled_mma, read_idx_<..>, pipe_A, pipe_B);
cutlass::arch::umma_arrive_multicast_2x1SM(mma_barrier_ptr, consumer_sync_mask);
```

`Blackwell_Matmul::run`（`blackwell_matmul.h:198-233`）内部由 **`warp_id()==0`** 单 warp 发 `gemm`（CuTe
`tiled_mma` → `tcgen05.mma.cta_group::2`），`k_iter==0` 时 `accumulate_ = UMMA::ScaleOut::Zero` 清零
（`:208-210`），之后 `One`（`:230`）。`umma_arrive_multicast_2x1SM`
（`deps/cutlass/include/cutlass/arch/barrier.h:813-828`）展开为：

```
tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;
```

——**一次 `tcgen05.commit` 把「MMA 干完」这个到达信号 multicast 给 cluster 内所有 CTA 的同一 mbarrier**
（mask 即 §3 的 `consumer_sync_mask`）。这就是 2SM 的「一个屏障、一对 SM」的落点，也是 `docs/04` §22
（P33-1）从 CUTLASS 资料分析里读到的 `umma_arrive_2x1SM` 的**上层用法**：mirage 直接用 multicast 版本，
把 entire cluster 当同步域。循环尾 consumer 放行输入环（`:1500-1506`）。

## 8. 累加器：TMEM → SMEM → （融合 epilogue）

2SM MMA 的累加器进 **TMEM**，不是寄存器，也不是 SMEM。两步：

1. `get_mma_tC(blockIdx.x, blockIdx.y, tmem_base_ptr)`（`:571-574`，定义 `blackwell_matmul.h:125-137`）
   造一个「指向 TMEM 的 C 张量」：`cta_mma.make_fragment_C(...).data() = tmem_base_ptr`；
2. 循环后按 accum 是否 in-reg 决定写回：`write_tC_to_sC`（`blackwell_matmul.h:158-192`）用
   `TMemLoadOp`（`sm100_get_tmem_load_op`，`:102-108`）把 TMEM→寄存器→SMEM，`thread_idx < 128`
   即只用**一个 warpgroup**搬运（`:161-164`）。

是否 in-reg 由 `sched_tb_graph.cc:239` 的 192 寄存器预算决定（`docs/18` §4.1）。in-reg 的在 `:1510-1538`
写回，且**必须先 `cute::wait_barrier(*mma_barrier_ptr, 0)`**（`:1536`）确保 2SM MMA 全部落地；in-SMEM 的
（`is_store_accum`）直接在 epilogue 里用 `EpilogueStoreAccum` 消费。融合链的逐元素算子编成
`EpilogueExp/SILU/GELU/...` 模板链（`transpile_fusion_epilogue`，`:1083-1121`）——与通用/Hopper 版一致
（`docs/18` §5.5），即「中间结果不落 SMEM、在寄存器/SMEM 写出时被下游 epilogue 就地消费」。

## 9. host 侧：cluster launch

`transpiler_kn.cc` 对 Blackwell 生成一段专门的 host 代码：

- 定义 `dim3 cluster_dim` 与 `cluster_shape`（`:734-744`）；
- 在 host 上**再定义一份 `TiledMMA`**（`:747-778`，与设备侧 `:504` 同款工厂）——因为 TMA descriptor
  的 host 构造依赖它；
- `generate_tma_code_blackwell(...)` 生成 TMA descriptor（`:786-788`）；
- `kernel_ptr` + `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)`（`:799-804`）；
- 最后用 **CUTLASS 的集群启动器**（`:824-831`）：

```c
cutlass::ClusterLaunchParams params = {grid_dim, block_dim, cluster_dim, smem_size};
cutlass::launch_kernel_on_cluster(params, (void const*)kernel_ptr, tma_..., dtensor_...);
```

`launch_kernel_on_cluster` 定义在 `deps/cutlass/include/cutlass/cluster_launch.hpp:364`。对照 Hopper 分支
（`:819-823`）用的是朴素 `<<<grid, block, smem, stream>>>`——**Hopper 不启动 cluster**（`cluster_dim`
虽在 `transpiler_tb_hopper.cc:559` 有断言，却不用于 launch）。这是两条后端最外层、也最本质的分野。

## 10. Hopper vs Blackwell：逐维对照

| 维度 | Hopper（`transpiler_tb_hopper.cc` / `docs/07`） | Blackwell（`transpiler_tb_blackwell.cc`，本文） |
|---|---|---|
| MMA 原子 | `Hopper_Matmul`，CuTe `GMMA::ss_op_selector`（wgmma，`hopper_matmul.h:95`） | `Blackwell_Matmul`，`SM100_MMA_*_2x1SM_SS`（tcgen05 2SM UMMA，`sm100_common.inl:385+`） |
| 计算单元 | 单 CTA（1 个 warpgroup 发 MMA） | **一对 CTA**（2SM，M=128/256，`AtomThrID=Shape<_2,_1,_1>`） |
| 累加器 | 寄存器 / SMEM（192-reg 预算） | **TMEM**（`Allocator2Sm`，`blackwell_matmul.h:125-192`） |
| 输入流水 | `HopperAsyncPipeline` = `PipelineTmaAsync`（`pipeline.h:22-24`） | `BlackwellAsyncPipeline` = `PipelineTmaUmmaAsync`（`blackwell_pipeline.h:15-16`） |
| TMA 载入 | `InputTMAAsyncCopy`（单 CTA，`transpiler_tb_hopper.cc:520`） | `InputTMAAsyncCopy_Blackwell`（**multicast** A 沿 N / B 沿 M，`input.h:352-360`） |
| 跨 CTA 同步 | — | `umma_arrive_multicast_2x1SM` → `tcgen05.commit...multicast::cluster`（`:1202`） |
| 启动 | 朴素 `<<<>>>`（`transpiler_kn.cc:819-823`） | `cutlass::launch_kernel_on_cluster`（`:824-831`） |
| swizzle chunk | 通用 `innermost_min_chunk_size` | 16B 下界 + 128B bank 对齐（`plan_tb_swizzle_blackwell.cc:114/122`） |
| 寄存器再平衡 | `wg_decrease/increase_regs` 被注释（`transpiler_tb_hopper.cc:907/1441`） | 同样被注释（`transpiler_tb_blackwell.cc:1052/1471`） |

一句话：**Blackwell 后端 = Hopper 后端 + 「把计算单元从 1 CTA 扩到 2 CTA」的一整条链**（2SM TiledMMA →
cluster 布局 → multicast TMA → TMEM 累加 → `tcgen05.commit` 会合 → cluster launch）。硬件换代把
`docs/04` §22 里「只存在于资料里的 2SM 原语」变成了编译器的默认生成目标。

## 11. 与论文 §4.2 的对照

`docs/18` §6 已把 Mirage transpiler 对论文 §4.2（快照行 1126–1146）的四项 intra-SM 优化逐条对上。
Blackwell 版**语义一致、实现升级**：

| 论文 §4.2（快照行） | Blackwell 版落点 |
|---|---|
| software pipelining（行 1138–1142） | producer warpgroup 发 TMA + `BlackwellAsyncPipeline`（`PipelineTmaUmmaAsync`），consumer 跑 UMMA（`:1036-1508`） |
| register reuse（行 1140） | 融合 epilogue 模板链（`:1083-1121`）+ accum-in-reg/TMEM 决策（`sched_tb_graph.cc:239`） |
| layout/bank-conflict（行 1140–1142） | `plan_tb_swizzle_blackwell.cc` + `UMMA::tile_to_mma_shape` 的 A/B 流水布局（`blackwell_matmul.h:66-84`） |
| thread-block 级超优化（行 1132–1134） | 一个 customized op = 一个 `custom_kernel_N`（`:350`），2SM 把粒度推到**跨 CTA 的 thread-block 对** |

论文正文（MPK）以 A100/H100 为实验平台，**未描述 Blackwell/SM100 的 2SM 路径**——这套是 Mirage 上游在
`mpk` 分支之上、面向 SM100 的继续实现，属 `docs/17`（多卡路径）同类的「论文未提、快照新增」。

## 12. 判据 C1–C5 映射

与 `docs/14`/`docs/15`/`docs/18` 一致，transpiler 层**只贡献 C3**：

| 判据 | Blackwell 后端 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | 每个 customized op 各一次 `launch_kernel_on_cluster`；C1 由 MPK 运行层（`docs/06`）给 |
| C2 常驻 | 否 | 生成的 kernel 非常驻 |
| C3 多算子范围 | **是（粒度来源）** | 一个多算子 bGraph → 一个 `custom_kernel_N`（内含 2SM UMMA + 融合 epilogue） |
| C4 设备内协调 | 否（但含 cluster 内会合） | 跨 kernel 仍靠 host 多次 launch；cluster 内 `mbarrier` 会合只覆盖一个 op 内部 |
| C5 SM 级调度 | 否 | 无 worker/scheduler、无去中心化调度 |

> 与 `docs/04` §18–§23 的关系：那里在 H100 上用 Thread Block Cluster / DSM / TMA multicast / 2D cluster
> 把「跨 SM 协作原语」实测出来（`cluster.sync`、multicast 投递、2SM 的 sm90 投影）。本文给出的是
> **SM100 上这套原语由编译器自动生成的形态**——`cluster` 从「手写实验」变成「codegen 默认」，
> `cta_group::2` 从「资料里的 PTX」变成「一个 `tiled_mma` 类型别名」。

## 13. 局限

- **纯源码走读，未构建/运行**：Blackwell 在本机（目标卡 H100/sm90）不可执行，故不采信任何性能数；
  所有结论基于 `f9eb70c`（mirage）与 `f3fde583`（`deps/cutlass`）快照行号。
- **未逐行展开 CuTe 模板装配**：`sm100_smem_selector` / `partition_shape_A/B` / `UMMA::tile_to_mma_shape`
  / `sm100_get_tmem_load_op` 只给代表性命中与作用，未逐句证明其 tile/layout 正确性（同 `docs/19` §8）。
- **未跑 tcgen05/TMEM 本体**：`tcgen05.mma.cta_group::2` 的语义、TMEM 分配冲突、`tcgen05.commit` 与
  mbarrier 的精确到达计数，本文只读生成代码与 CUTLASS 封装，未在 SM100 上验证；sm90 上的等价投影见
  `docs/04` §22（且那里的结论是「sm90 缺席 `.cta_group::2`、软件投影反而慢 ~13%」）。
- **`g.cluster_dim` 默认 `{4,4,1}` 与 2SM 的适配未深挖**：本文指出 cluster 沿 N 维拼宽、A 沿 N 广播、
  B 沿 M 广播（`input.h:352-360`），但 `cluster_dim` 的具体取值来源（`graph.h:200` 默认值 vs 搜索器/
  前端设置）未逐条追（`docs/14`/`docs/15` 覆盖了 bGraph 的构建，未含 cluster_dim）。
- **循环内 `elect_one_cta && elect_one_warp` 门的影响未验证**：`:1143-1146` 把循环内**所有**算子收进
  单 CTA 单 warp；对「循环里出现非 MMA 算子」的正确性/效率影响本文只作标注，非结论。
- **`docs/18` §8 的其余尾巴未覆盖**：SASS 级生成、`benchmark/` 性能数、`docs/18` 提到的
  `docs/transpiler/transpiler.md`（快照未提供）不在本文范围。

## 参考坐标

- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.2 行 1126–1146）。
- 源码快照：`github.com/mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。
  主文件：`src/transpiler/transpiler_tb_blackwell.cc`（1575 行）、`plan_tb_swizzle_blackwell.cc`（177 行）；
  运行期头 `include/mirage/transpiler/runtime/threadblock/blackwell_matmul.h`（236）、
  `blackwell_pipeline.h`（93）、`input.h:256-395`、`utils.h:94-118`；host 侧 `src/transpiler/transpiler_kn.cc:700-839`。
  CuTe/CUTLASS：快照 `deps/cutlass`（`f3fde583`）的
  `include/cutlass/gemm/collective/builders/sm100_common.inl:373-429`、
  `include/cutlass/arch/barrier.h:813-828`、`include/cutlass/pipeline/sm100_pipeline.hpp:537`、
  `include/cutlass/cluster_launch.hpp:364`。同专题相关篇：`docs/07`（Hopper device task）、
  `docs/18`（transpiler 通用/Hopper）、`docs/19`（runtime 模板库）、`docs/20`（Triton 后端）、
  `docs/04` §18–§23（跨 SM 协作原语实测）。
