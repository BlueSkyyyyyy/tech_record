# MPK SM100 2SM MLA attention 走读：整颗 CUTLASS FMHA kernel 与 2SM 协作（P122-1）

> 定位：本文补掉 `docs/35-mpk-sm100-mla-family.md` **§9「局限」明留的头号尾巴**（原话）——
>
> > **`mla_sm100_2sm.cuh`（3165 行，2SM 协作 attention——`task_header.cuh:53` 有 include、standalone
> > 单测有 caller，但本文未逐行）**……（留待后续）
>
> `docs/35` 走完了 MLA attention 家族的 KV gather / decode / reduce / prefill 各**device task 身体**；本文打开
> 其中**唯一一颗「整颗 CUTLASS kernel」形态**的 `mla_sm100_2sm.cuh`——它不是 MPK 惯用的「一个 task =
> 一段 `__device__ __noinline__` 身体」，而是**一整颗由 CUTLASS `device::MLA` 设备层用 cluster launch
> 直接发射的 `__global__` kernel**，且**只在 2SM 模式下编译**（`kIs2Sm = true` 硬编码）。快照
> `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/35` 同快照）。
> **纯源码走读（资料分析），无新实验**——SM100（`sm_100f`/`sm_100a`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **它是「整颗 kernel」，不是「一个 task body」**：文件头（`:1-20`）自陈 *"Self-contained MLA Device Kernel
   for Blackwell (SM100) … All flashinfer dependencies are inlined - only CUTLASS/CuTe headers are needed"*，
   内容是完整的 6 节（Pow2 助手 / CustomStride-Gather / tile scheduler / split-KV reduction kernel /
   主 MLA kernel / device 层）。它的落地形态是 CUTLASS 惯用法——`device::MLA<Kernel>::run()` 用
   `ClusterLauncher::launch`（`:3108-3109`）自己发一个 `(2,1,1)` cluster grid、`can_implement`/`initialize`/
   `update`/`get_workspace_size` 一整套宿主 API（`:2968-3158`），**不经过 MPK 运行层的 `_execute_task`**。
   这解释了它为何在 `task_header.cuh:53` 里被 include、却**没有任何 `register_*_task` / `TaskType` /
   Python layer**：它走的是 `docs/29`/`docs/30` 记的「一个 device task = 一整颗 kernel，由 host 手写/
   CUTLASS 直接 launch」的**单 kernel 单测层**（wrapper `runtime_kernel_wrapper_sm100_mla.cu`，对外
   pybind `mla_attention`），**不是生产 task**。
2. **2SM 协作是全文件的第一性设定**：`static bool const kIs2Sm = true`（`:610`），由此派生
   `ClusterShape = Shape<_2,_1,_1>`（`:617-618`）、`TmemAllocator = cute::TMEM::Allocator2Sm`（`:631-632`）、
   `kWarpsInN = 2`（`:635`）、GEMM 调度 `KernelTmaWarpSpecialized2SmSm100`（`:689-692`）、
   `AtomThrShapeMNK`（`:740`）。**一对 CTA**（`cluster_dim=(2,1,1)`）共同算一次 MMA：只有 **CTA rank 0 是
   `mma_leader`**（`:1102-1104`）去发 `tcgen05`/UMMA 与 TMEM 分配（`:1365`）；TMEM 按 `kWarpsInN=2` 对半
   切（`kSizeS=64`、`kSizeO=256`、`kSizeAccO=128`，`:816-829`），softmax 的 `row_max`/`row_sum` 要靠
   **SMEM 交换 + 命名屏障 + `peer_index=(threadIdx.x+64)%128`** 在 2×2 的数据并行维上归约（`:2447-2457`、
   `:2847-2856`），LSE 只由 `threadIdx.x < 64` 写（`:2657-2659`、`:2713-2717`）。这与 `docs/04` §22
   （P33 `docs/04-experiments.md` §22：`cta_group::2` TMA load + tcgen05 2-SM MMA 的 sm90 投影）和
   `docs/21`（Blackwell 自动 transpiler 的 2SM UMMA codegen）是**同一套 SM100 2SM 原语**在 MLA 上的手工果实。
3. **它把「一整个 FlashInfer 风格查表注意力」压进一条五级流水线**：7 个 warp 分 `kCompute(0-3)` /
   `kMma(4)` / `kLoad(5,6)` / `kLoadPageTable(7)` 四种角色（`:640-675`），用 5 条 pipeline
   （`PipelineLoadQK`/`PipelineS`/`PipelineP`/`PipelineO`/`PipelinePT`，`:749-761`）串起
   QK→softmax→PV→online-rescale→epilogue；分页 KV 用 `CustomStride`/`Gather`（`:100-265`）在 CuTe 层
   表达（page table → 物理页号），`load_cpasync` 里靠 `copy_split` **先做完所有 page-table 查找再发
   `cp.async`**（`:1606-1622`），`load_tma` 走 TMA 描述符。这与 `docs/35` §2 的手写 KV gather 是**同一
   问题的两种表达**：那里是「显式 flatten + 4 路 ILP 的 `__device__` 函数」，这里是「CuTe composed
   layout + gather functor 的标准 CUTLASS 写法」。

判据口径同 `docs/07`/`docs/29`–`docs/35`：**这项工作只贡献 C3**（单 kernel 可表达、数值可被单测校验），
C1/C2/C4/C5 由运行层拿。而且它连「进常驻内核」都还没接（§7）。

---

## 1. 定位：它在编译栈里的坐标

| 层 | 本文覆盖 | 上游/下游 |
|---|---|---|
| 前端（`docs/14`/`docs/25`/`docs/26`） | **无**（没有对应 Python layer） | DeepSeek-V3 builder 的 MLA 段 |
| task 目录（`docs/24`） | **无**（无 `TaskType`） | — |
| 生产布线（`docs/31`） | **无**（无 `register_*`） | — |
| **单 kernel 单测层（本文）** | **`mla_sm100_2sm.cuh` + `runtime_kernel_wrapper_sm100_mla.cu`（pybind `mla_attention`）** | `docs/29`/`docs/30`（同层惯例） |
| device 身体（`docs/32`/`docs/33`/`docs/34`/`docs/35`） | 本文是其中**唯一的「整颗 CUTLASS kernel」形态** | `docs/07`（Hopper task）/`docs/21`（Blackwell 2SM codegen） |
| 运行层（`docs/06`/`docs/22` §8） | **不经过**（自持 launch） | C1/C2/C4/C5 只在此拿 |

**接线事实（静态 `grep`）**：全仓引用 `mla_sm100_2sm.cuh` 的只有两处（除自身外）——
`include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh:53`（聚合 include，便于整仓库一处编译
所有 SM100 task）与 `tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_sm100_mla.cu:23`
（standalone 单测 wrapper）。`Sm100FmhaMlaKernelTmaWarpspecialized` / `device::MLA` 在生产侧
（`src/`/`python/`/`graph.cc`/`task_register.cc`）**零引用**——即它与 `docs/35` §3/§4 的旧 decode/reduce
属于**同一族「有实现、非生产路径」**，但更彻底：连 `TaskType` 都没有，只在单测层。

> 与 `docs/35` §3/§4 的对照：旧 decode/reduce 是「进了生产 registrations、但被后继 MTP 版顶替」；
> 本文件是「从未进生产注册表、只在单测层」。读代码时不能把「`task_header.cuh` 里有 include」当成
> 「生产会用」（本专题「已知坑」同条：「有实现 ≠ 已启用」）。

---

## 2. 文件骨架：六个 Section（`:1-20` 自陈 + 逐行核对）

| Section | 行 | 内容 |
|---|---|---|
| 1 Pow2 助手 | `:44-98` | `struct Pow2`（把「除以 2 的幂」编译成移位/掩码，`:50-91`）+ `cute::is_integral` 特化（`:95-98`） |
| 2 CustomStride/Gather | `:100-265` | `example::NoGather`/`IndexedGather`/`StridedGather`（`:108-148`）、`CustomStride`（`:150-201`）、分页 gather functor（`:202-265`）——把 page table 折进 CuTe stride |
| 3 Tile Scheduler | `:267-381` | `Sm100MlaIndividualTileScheduler`（`:272-310`，grid=`(cluster_m, batch, split_kv)`）+ `Sm100MlaPersistentTileScheduler`（`:312-379`，`block_idx += gridDim.x`，用 `FastDivmod` 解码 `(m, b, split)`） |
| 4 Reduction Kernel | `:383-580` | `Sm100FmhaMlaReductionKernel`（`:396`）：split-KV 的部分 O/LSE 合并，128 线程、`grid=(kNumHeads,1,batches)`（`:444-446`） |
| 5 **主 MLA Kernel** | `:583-2887` | `Sm100FmhaMlaKernelTmaWarpspecialized`（`:602`）——本文主体（§3–§6） |
| 6 Device 层 | `:2889-3165` | `device::MLA<Kernel>`（`:2898`）：`run()` 用 `ClusterLauncher::launch`（`:3102-3109`）+ 可选 reduction kernel（`:3122-3137`） |

`TileShape = (H=128, S=128, D=(L=512, R=64))`（wrapper `:38-40`），`Element = bfloat16`、
`ElementAcc = float`、`TileScheduler = Individual`（wrapper `:41-45`）。

---

## 3. 2SM 机制的逐点落位（headline）

1. **集群形状与调度器**：`kIs2Sm=true` → `ClusterShape=Shape<_2,_1,_1>`（`:617-618`）→
   `Schedule=KernelTmaWarpSpecialized2SmSm100`（`:689-692`）→ `CollectiveMmaQK/PV` 以此为 `ClusterShape`
   `:704`/`:731`。`AtomThrShapeMNK = CollectiveMmaQK::AtomThrShapeMNK`（`:740`）且强制与 PV 版相等
   （`:741-743`，*"schedule must match"*）——QK 与 PV 两条 GEMM 必须共用同一 2SM atom。
2. **一对 CTA 一体的 TMEM**：`TmemAllocator = Allocator2Sm`（`:631-632`），MMA warp（role 只在 **leader
   CTA**）`tmem_allocator.allocate(Sm100TmemCapacityColumns, &tmem_base_ptr)`（`:1365-1366`），退出前
   `cluster_sync()`（`:1441`）+ 命名屏障 `kNamedBarrierTmemDealloc` 到齐后再 `free`（`:1442-1449`）——
   **TMEM 生命周期与 `docs/34` §3.5「常驻 kernel 内 TMEM alloc/dealloc 必须同一 warp」同族，但这里是
   「同一 WarpRole + 跨 CTA cluster 会合」**。
3. **只有 leader CTA 发 MMA**：`cta_rank_in_cluster % size<0>(AtomThrShapeMNK{})` 定 `cta_coord_v`，
   `is_mma_leader_cta = (cta_coord_v == 0)`（`:1102-1104`）；`kMma` 角色的主循环包在
   `if (is_mma_leader_cta) { mma(...) }`（`:1369-1399`）里，非 leader CTA 的 MMA warp **不发 GEMM**。
   TMA load 侧的 `is_leader` 也要求 `is_mma_leader_cta`（`:1134-1135`）——TMA 只由 leader 发。
4. **2×2 数据并行的归约 = SMEM 交换 + 命名屏障**：softmax 求 `row_max` 时，`kWarpsInN>1` 分支把
   本线程 `row_max_new` 写进 `smem_exchange[threadIdx.x]`，`NamedBarrier(kNumComputeWarps*NumThreadsPerWarp,
   kNamedBarrierExchange).sync()`（= 4 warp×32 = 128 线程），再用 `peer_index=(threadIdx.x+64)%128`
   取对端并 `max`（`:2448-2457`）；compute 末尾对 `row_sum` 做同构 `+`（`:2847-2856`）。**这就是 2SM 把
   S 维切成两半（`kSizeS=TileShapeS/kWarpsInN=64`）后必须补的那一步**。
5. **LSE 写只由半 CTA 做**：epilogue 里 `if (!kIs2Sm || (threadIdx.x < 64)) gLSE(threadIdx.x) = lse;`
   （`:2657-2659` 与 `:2713-2717`，注释 *"for 2x2 dp, this must be conditional and the index is wrong"*）
   ——2SM 下 `gLSE` 的索引只对前半 64 线程有效，故用 `threadIdx.x < 64` 守卫，**这是代码里自带的一处
   「2SM 适配未做全」的自陈注释**（索引语义在 2×2 下未完全重排）。
6. **基数约定**：`kWarpsInN = kIs2Sm ? 2 : 1`（`:635`）只从 2SM 派生，其他 1SM 分支（`Allocator1Sm`/
   `KernelTmaWarpSpecialized1SmSm100`）在文件里**存在但恒不实例化**（`conditional_t` 的 dead 分支）——
   文件明确声明 *"only 2Sm mode is supported"*（`:609`）。

> **与 P33 的关系**：P33 在 sm90 上**软件投影** `cta_group::2`（两 CTA 一个 cluster、rank0 发 TMA/MMA），
> 并实测「分区投递不省源流量、2SM 的价值在硬件层（一次发、一个屏障、一对 SM 分持 TMEM）」。本文件正是
> 那条硬件行为的**真实 SM100 消费者**：`Allocator2Sm` + `2SmSm100` 调度 + leader-only 发 MMA 就是
> P33 资料分析里 CUTLASS `SM100_MMA_*_2x1SM_SS` / `umma_arrive_2x1SM` 的上层使用者。

---

## 4. warp 角色与五级流水线

**warp 分工**（`:640-675`，`kWarpAssignment = 0x4221'3333`@cp.async / `0x0021'3333`@TMA）：

| warp | cp.async 模式 | TMA 模式 | role |
|---|---|---|---|
| 0–3 | `kCompute` | `kCompute` | softmax / PV-rescale / epilogue（4 warp） |
| 4 | `kMma` | `kMma` | QK + PV 的 `tcgen05`/UMMA（leader CTA） |
| 5 | `kLoad` | `kLoad` | Q/K/V 载入（TMA 或 `cp.async`） |
| 6 | `kLoad` | `kEmpty` | cp.async 时多一个载入 warp |
| 7 | `kLoadPageTable` | `kEmpty` | 分页 page table 的 `cp.async_zfill` 生产者 |

`MaxThreadsPerBlock = 256`（`:611`），无 thread guard（与 `docs/35` 的 128 线程守卫不同——因为它是独立
kernel、block 恰好 256 线程）。

**五条 pipeline**（`:749-761`）：

- `PipelineLoadQK`：load→mma。cp.async 用 `PipelineUmmaConsumerAsync`、TMA 用
  `PipelineTmaUmmaAsync<StagesQK, ClusterShape, AtomThrShapeMNK>`（`:749-752`）。`StagesQK =
  24/sizeof(Element)`（bf16 → **12**，`:682`）。`PipelineLoadPV = PipelineLoadQK`（`:753`，**K/V 共用同一
  段 SMEM staging**，见 `TensorStorage` 的 `union { smem_kc; smem_vc; }`，`:841-844`；静态断言
  `kBytesLoadKC == kBytesLoadVC`，`:799`）。
- `PipelineS`：mma(QK)→softmax，`PipelineUmmaAsync<TotalSNum=2>`（`:755`），TMEM 里 S 双缓冲
  `kS0/kS1`（`:816-829`）。
- `PipelineP`：softmax(P)→mma(PV)，`PipelineUmmaConsumerAsync<TotalPNum=2>`（`:757`）。
- `PipelineO`：mma→softmax（供 online rescale），`PipelineUmmaAsync<1>`（`:759`，单 stage）。
- `PipelinePT`：page table 生产者→消费者，`PipelineAsync<StagesPageTable>`（`:761`）。

`smem_exchange`（`:837-838`，2×4 warp×32 个 `ElementAcc`）就是 §3.4 的 2SM 归约缓冲。

---

## 5. 主循环逐段（`compute`/`mma`/`softmax`/`rescale`/`epilogue`）

**载入两条路**（二选一，`kIsCpAsync` 模板参，wrapper 编译未定义 `CPASYNC`，故默认 **TMA**）：

- `load_cpasync`（`:1532-1834`）：用 `SM80_CP_ASYNC_CACHEALWAYS<uint128_t>`（`:1600`，16 B `cp.async`）
  把 Q/KC/VC 搬进 SMEM；page table 用 `cp_async_zfill`（`:1504`）预取到 `smem_page_table`（`:839`）；
  分页地址靠 `example::CustomStride`+`Gather` 折进 composed layout（`:1650-1705`）；`copy_split`
  （`:1606-1622`）**先把所有 page-table 查找算完再发拷贝**，避免边查边发的依赖链。
- `load_tma`（`:1835-2184`）：TMA 描述符版（`prefetch_tma_descriptor`，`:1107-1117`），
  `pipeline_load_qk_params.is_leader = lane_predicate && role==kLoad && is_mma_leader_cta`（`:1134-1135`）。

**MMA**（`:2185-2388`，只在 leader CTA）：QK 循环 `IterationsQK` 个 k_block 累加到 `tStS`（TMEM `kS0/kS1`，
`:2259-2278`），`producer_commit` 给 softmax；主 `while(k_tile_count>0)`（`:2286-2351`）里交错发
`PipelineS`/`PipelineP`/`PipelineO`：QK→softmax 后，PV 从 `tOrP`（softmax 产出的 P）× `tOrVC`（V）
累加到 TMEM `kO0 + j*kSizeAccO`（`:2316-2343`）。**online softmax 顺序在注释里画成全图**（`:2253-2256`）。

**Softmax**（`:2390-2524`）：`SM100_TMEM_LOAD_32dp32b32x` 从 TMEM 取 S（`:2401`），mask 最后一块
（`is_last_tile`，`:2431-2438`），`row_max_new`（含 §3.4 的 2SM peer 归约，`:2447-2457`），
`correction_factor = exp2(scale_log2*(row_max - row_max_new))`（`:2463`），`exp2` 求 P 并
`NumericArrayConverter` 量化成 bf16 写回 `smem_p`（`:2476-2496`），`row_sum *= correction_factor`
再 `+= Σp`（`:2499-2523`）；`B2B` 宏可关掉 rescale（`:2459`/`:2528`）。

**Rescale**（`:2526-2575`）：`O *= correction_factor`（TMEM load→乘→store，`SM100_TMEM_LOAD_32dp32b32x`
+ `tmem_load_to_store`）。

**Epilogue**（`:2577-2720`）：从 TMEM 取 O（`load_op`）、`LinearCombination` 按 `output_scale/row_sum`
缩放写 `ptr_o`（`:2684-2696`）；若有 `ptr_o_acc`（split-KV）则写 `o_accum` 并算 LSE（`:2601-2660`）；
LSE 按 §3.5 的 `threadIdx.x < 64` 守卫写（`:2657-2659`/`:2713-2717`）。

`compute`（`:2722-2882`）串起 softmax→rescale(每 `IterationsPV_N` 个 TMEM O 块)→epilogue，末尾做 2SM
`row_sum` peer 归约（`:2847-2856`）后逐 `j` 调 `epilogue`（`:2864-2877`）。

---

## 6. Device 层：自持 launch 与 split-KV

`device::MLA<Kernel>`（`:2898`）是一个标准 CUTLASS device adapter：

- **`set_split_kv`**（`:2953-2966`，host 自动 split 启发式）：`max_splits=ceil_div(K,128)`、
  `sms_per_batch=max(1, sm_count/B)`、`split_heur=min(max_splits, sms_per_batch)`、
  wave-aware 再折一层——把 KV 维按可见 SM 数与 batch 数切给多 CTA，余下由 reduction kernel 合并。
- **`run`**（`:3095-3137`）：`ArchTag::kMinComputeCapability >= 90` 走 `ClusterLauncher::launch`
  （`:3102-3109`，cluster=`(2,1,1)`）；`split_kv>1` 时再发 `Sm100FmhaMlaReductionKernel`
  （`:3122-3137`）。**即它自己发 1~2 个 kernel，完全绕开 MPK 运行层**。
- **Reduction kernel**（`:396-578`）：`grid=(kNumHeads,1,batches)`、128 线程（`:444-450`），
  warp0 用 `__shfl_xor_sync` 在 32 split 上求 `lse_max`/`sum_lse`/`global_lse`（`:490-545`），
  然后每线程按 `sLseScale[split]` 合并 `o_accum` 写 bf16 O（`:548-576`）。

wrapper `runtime_kernel_wrapper_sm100_mla.cu`（314 行）把 PyTorch 张量 pack 成 `Arguments`
（`problem_shape`/`mainloop`/`epilogue`，`:121-193`），`split_kv=1`、`ptr_lse=nullptr`（`:192`/`:182`），
`can_implement`→`get_workspace_size`→`run`（`:196-221`），并强制 `num_heads==128`、`head_dim==576`
（`:89-91`）。单测 `test_mla_attention.py` 头注（`:1-12`）自陈**对标 FlashInfer 参考实现**
（`import flashinfer.mla`，`:19-24`）+ 纯 PyTorch 参考（`attention_ref_pytorch`）。

---

## 7. 判据与生产位置

- **判据**：本文件**只贡献 C3**（单 kernel 可表达、数值由单测对标 FlashInfer/PyTorch 校验）。它是
  一颗**独立 kernel**（自己 launch、自己 `cluster_sync`、`alloc/free` TMEM），所以 C1（单次 launch 整图）
  / C2（常驻）/ C4（跨算子设备内协调）/ C5（去中心化调度）**都不满足**——与 `docs/29`–`docs/35` 同口径。
- **生产位置**：**不在生产路径**。无 `TaskType`、无 `graph.cc` op、无 `task_register.cc` 注册、无 Python
  layer；全仓 caller 只有 standalone 单测 wrapper（§1）。它代表 MPK 仓库里「把一整颗 Blackwell FMHA
  kernel 作为参考/能力储备收进来」的一档——**与 `docs/30`/`docs/31` 记的 `cute/hopper/gemm.cu`
  （CUTLASS standalone）同族**，只是更完整（连 device adapter 一起）。
- **对 megakernel 专题的意义**：它是「**整颗 kernel**」的形态上限参照——把 `docs/35` 的「一个 task
  = 一条 `__device__` 身体」与本文的「一个 task = 一整颗 cluster kernel」摆在同一条 MLA 家族里，可看清
  MPK 生产为何选前者：常驻常驻 kernel 里 worker 的 256 线程要能**反复执行不同 task**，而 CUTLASS device
  层的 `run()` 是「发一次、跑完退」，与 C2 常驻根本冲突。这也从反面印证 `docs/34` §3.5 / `docs/33`
  那些「TMEM 不 relinquish / alloc 同 warp」的让步——生产 task 必须按常驻内核的约束重写，而不能直接
  搬一颗 CUTLASS kernel 进来。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建/未运行**：SM100 在目标卡 H100/sm90 不可执行（推荐 `-gencode=arch=compute_100a,
  code=sm_100a`，见 setup `:50-56`），未取任何 device 计数；「无生产 caller」是**静态 `grep`** 结论，
  不构成运行时判定，也**不代表上游 bug**。
- **「整颗 CUTLASS kernel」的细节未逐行**：QK/PV 的 `CollectiveMmaQK/PV` 内部（TMA 描述符布局、
  `AtomThrShapeMNK` 的 CuTe atom、UMMA 指令序列）来自 CUTLASS `collective_builder`，本文只核到「本文件
  如何实例化与驱动它们」，未展开 CUTLASS 侧（`deps/cutlass`）实现。
- **2SM 的「索引未完全重排」是源码自陈**：epilogue 的 `gLSE` 注释（`:2656`/`:2713`，*"the index is
  wrong"*）说明 2×2 数据并行下 LSE 索引语义**源码作者自认未做全**；本文只转述，不判对错。
- **`B2B` 宏路径未细究**：`B2B`（back-to-back）在 softmax/rescale 里关掉 rescale（`:2459`/`:2528`），
  但未在本文逐行追其触发条件与消费方。
- **相邻尾巴（留待后续）**：`docs/35` §9 的其余条目——注意力之外的 SM100 task 家族
  （`attention_sm100`/`dflash_*`/`inkling_*`/`glm_moe_router_sm100`/`prob_scatter_sm100`/
  `softmax_gather_sm100`/`topk_*`/`sampling_sm100`/`rotary_embedding_sm100` 等）、`mla_prefill_host.cu`
  原型来源——本文未覆盖。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/35` 同快照）。
- **主文件**：`include/mirage/persistent_kernel/tasks/blackwell/mla_sm100_2sm.cuh`（3165 行；文件头
  六节说明 `:1-20`；Pow2 `:44-98`；Gather/CustomStride `:100-265`；tile scheduler `:267-381`；
  reduction `:383-580`；主 kernel `:583-2887`；device 层 `:2889-3165`）。
- **2SM**：`kIs2Sm :610`、`ClusterShape :617-618`、`TmemAllocator(Allocator2Sm) :631-632`、
  `kWarpsInN :635`、`Schedule(2SmSm100) :689-692`、`AtomThrShapeMNK :740-743`、
  `is_mma_leader_cta :1102-1104`、TMEM alloc/free `:1365-1366`/`:1441-1449`、
  softmax peer 归约 `:2447-2457`、compute peer 归约 `:2847-2856`、LSE 半 CTA 守卫 `:2657-2659`/`:2713-2717`。
- **主循环**：warp 角色 `:640-675`、`kWarpAssignment :648-649`、五 pipeline `:749-761`、
  `TmemAllocation :816-829`、`TensorStorage :835-846`、`operator() :1095-1450`、
  `load_page_table :1452-1531`、`load_cpasync :1532-1834`（`copy_split :1606-1622`）、
  `load_tma :1835-2184`、`mma :2185-2388`、`softmax :2390-2524`、`rescale :2526-2575`、
  `epilogue :2577-2720`、`compute :2722-2882`。
- **Device 层**：`device::MLA :2898`、`set_split_kv :2953-2966`、`run/ClusterLauncher :3095-3137`、
  reduction kernel `:396-578`。
- **接线**：`task_header.cuh:53`（include）；standalone 单测 wrapper
  `tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_sm100_mla.cu`（314 行；主 kernel
  类型 `:47-53`、packing `:121-193`、run `:196-221`、pybind `:291-314`），
  `test_mla_attention.py`（对标 FlashInfer，`:1-24`）；setup `setup.py`（`compute_100a`，`:50-56`）。
