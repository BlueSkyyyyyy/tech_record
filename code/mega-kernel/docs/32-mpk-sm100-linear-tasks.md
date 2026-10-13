# MPK SM100 手工线性 task 的 device 内部：`linear_fp8_sm100` 家族与生产布线走读（`docs/31` §10 尾巴，P118-1）

> 定位：本文补掉 `docs/31-mpk-fp8-wrappers.md` **§10「局限」明留的两条尾巴**——
>
> > **未展开的更深层**：`fp8_group_gemm_sm100.cuh` 只走读了 warp 角色与 scale 路径，**UMMA 主循环 /
> > UTCCP 的具体 smem→TMEM 搬运序列 / epilogue scatter** 未逐行；`linear_fp8_sm100` 的 11 头家族里
> > `_scheduler`/`_epilogue_utils`/`_cute_tie` 未逐行走；TP 变体的 `mla_mtp_tpN_main` 主体只核了常量。
> > **未与生产路径对跑**：`docs/24` 的 `TaskType` 目录里这些 SM100 task 的登记/接线未复核。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），快照 **commit `f9eb70c`**，
> 与 `docs/06`–`docs/31` 同一快照，所有行号据此核对。CuTe/CUTLASS 侧引用 m1 仓库自带的
> `deps/cutlass`（本机 `f3fde583`）。
>
> 本文为**纯源码走读（资料分析）**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，故不构建、
> 不运行、不引用任何性能数——与 `docs/17`/`docs/21`/`docs/29`/`docs/30`/`docs/31` 同一口径；
> 相关机制的 sm90 投影实测见 `docs/04` §18–§23、§46。

---

## 0. 一句话：把「一个 SM100 线性 task 的身体」从一个头文件扩到整族 + 生产布线

`docs/31` §4 只列了目录、§3 走了 wrapper 的 host 侧；**没有打开 device 端**。本文打开两端：

1. **device 端**：`linear_fp8_sm100.cuh`（828 行）及其 8 个 helper（`_common`/`_types`/`_utils`/
   `_sm100_utils`/`_tma_utils`/`_scheduler`/`_epilogue_utils`/`_cute_tie`）逐个走通——这是一个
   **手工 warp 特化的 SM100 block-scaled FP8 linear CTA**（TMA 生产者 / UMMA leader / UTCCP scale
   转置 / TMEM→SMEM epilogue 四组角色）。
2. **生产布线**：把 unit-test wrapper 里那个同名的 device 模板，与**生产路径**
   `task_register.cc` 的登记逐参数对照——同一份 device 模板被**两种调用方**以**不同模板实参**实例化，
   且生产侧为「持久化内核」做了两处明确的适配（`kNumSMs=1` 的 `blockIdx.x` HACK、`kNumStages=25`
   压 207 KB SMEM 预算）。

两条尾巴各给一个 headline：

- **headline A（device 内部）**：`linear_fp8_sm100` 家族在仓库里其实是**两套互不相干的 device 身体**——
  一套是**手写 SMEM 描述符 + 裸 PTX `tcgen05.mma…block_scale`**（`linear_fp8_sm100.cuh`），另一套是
  **CuTe `make_tiled_mma` / `local_tile` / `tile_to_mma_shape`**（`linear_fp8_1d2d_sm100.cuh`）。**同名
  「1d2d」的公开 pybind 实际走的是第一套**；第二套在快照里**只有 `#include`、没有任何调用者**。
- **headline B（生产布线）**：生产前端 `linear_fp8_sm100` op 生成的是 `linear_fp8_sm100_task_impl<…>`
  的**单 CTA（`kNumSMs=1`）`GemmType::Normal`** 实例，`kNumMulticast=1`（**不用** `cta_group::2`/
  cluster）；而 unit-test 里保留的同模板可以 `Allocator2Sm`/multicast（`docs/21` 的 2SM 路线）。
  ⇒ **同一份「one task = one CTA」device 模板，在产线是单 SM 顺序跑完所有 tile、在单测里才展开 cluster。**

---

## 1. 家族清单（先修正 `docs/31` §4 的计数）

`docs/31` §4 的标题写「**11 个头**」，但逐目录实点为 **10 个** `tasks/blackwell/` 下匹配 `linear_fp8`
的文件（`docs/31` 的列表本身也只列了 10 个名字，属笔误）：

| 文件 | 行数 | 角色 |
|---|---|---|
| `linear_fp8_sm100.cuh` | 828 | **身体 A**：`__device__` task_impl + `__global__` wrapper |
| `linear_fp8_sm100_common.cuh` | 8 | 聚合 include（`_epilogue_utils`/`_scheduler`/`_sm100_utils`/`_tma_utils`/`_types`/`_utils`） |
| `linear_fp8_sm100_types.hpp` | 48 | `MmaKind`/`GemmType`/`KernelType` 三枚举 |
| `linear_fp8_sm100_utils.cuh` | 143 | `PatternVisitor`、`ceil_div`/`align`、`ld/st_shared`、BF16 打包/残差加 |
| `linear_fp8_sm100_sm100_utils.cuh` | 232 | SMEM descriptor、`make_instr_desc_block_scaled`、TMEM 列对齐、`tcgen05` fence、**两个 MMA 结构体** |
| `linear_fp8_sm100_tma_utils.cuh` | 118 | `tma_copy`（SM90 vs SM100 2SM load 分派）、tensormap fence/replace |
| `linear_fp8_sm100_scheduler.cuh` | 199 | `Scheduler<GemmType,…>`：tile 分块、swizzle、全局索引 |
| `linear_fp8_sm100_epilogue_utils.cuh` | 30 | `EpilogueIdentity` / `EpilogueHeadSplits`（N 维重映射） |
| `linear_fp8_sm100_cute_tie.cuh` | 60 | `cute::ignore_t` + `CUTE_TIE` 宏 |
| `linear_fp8_1d2d_sm100.cuh` | 727 | **身体 B**：独立 CuTe 实现（`linear_fp8_1d2d_sm100_task_impl`） |

另有三个**同族近亲**（不属 `linear_fp8` 前缀，但同属 SM100 线性 task，本文按需交叉引用）：
`linear_sm100_mpk.cuh`（**BF16** 版，生产登记 `TASK_LINEAR_SM100`，§5.4）、`fp8_group_gemm_sm100.cuh`
（grouped FP8 MoE 版，`docs/31` §5 已走）、`linear_swapAB_hopper.cuh`（Hopper 版，`docs/07` 已走）。

---

## 2. 身体 A：`linear_fp8_sm100.cuh` 的四个角色与三条流水

### 2.1 入口签名与编译期契约（`:45-171`）

`linear_fp8_sm100_task_impl<...>`（`:45`）带**一长串编译期模板参数**（`:18-44`）：`kMajorA/B`、
`kGranKA/B`、`SHAPE_M/N/K`、`BLOCK_M/N/K`、`kNumGroups`、三个 swizzle、`kNumStages`、
`kNumNonEpilogueThreads`/`kNumEpilogueThreads`、`kNumMulticast`/`kIsMulticastOnA`、`kNumSMs`、
`kWithResidual`、`GemmType`、`kWithAccumulation` 与 4 个 dtype（`a/b/cd/epilogue`）；运行时参数是
**6 个 TMA descriptor**（`tensor_map_a/b` + `sfa/sfb` + `residual` + `cd`，`:46-55`）。

编译期契约（`:58-171`）：

- **SM100 专属**：整个 body 包在 `#if (__CUDA_ARCH__ >= 1000) || __CLION_IDE__`（`:58-59`）里；
  否则只在 block0/thread0 触发 `DG_DEVICE_ASSERT(false && "This kernel only support sm_100f")`（`:742-744`）。
- **累加/残差 dtype 断言**：`kWithAccumulation ⇒ cd_dtype=float`（`:65-68`）；`kWithResidual ⇒
  cd_dtype=bfloat16`（`:69-72`）。
- **形状常量**：`LAYOUT_AD_M=128`、`WAVE_BLOCK_M=min(BLOCK_M,128)`、`kNumMWaves=BLOCK_M/128`（`:74-81`）；
  `BLOCK_K==128` 硬断言（`:79`）；scale 粒度 `kGranK∈{32,128}`（`:85-88`）。
- **scale 载入批次**：`kNumSFAStagesPerLoad = (kGranKA==32)?1:4`（`:83`）——128 粒度时一次载 4 个
  K-tile 的 scale。
- **SMEM/TMEM 预算编排**（`:100-171`）：`extern __shared__ __align__(1024)`（`:100`）；SMEM 依次排
  `CD → residual → A → B → SFA → SFB → barrier 数组 → TMEM 指针`（`:184-243`）；TMEM 列
  `kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols` 经 `get_num_aligned_tmem_cols` 对齐到
  {32,64,128,256,512}（`:165-172`，helper `_sm100_utils.cuh:156-172`）。
- **TMA box 尺寸按全局维 clamp**（`:119-130`），且注释明确「`arrive_and_expect_tx` 必须等于真实 TMA
  传输字节而非 SMEM 大小」（`:116-118`）——这是 SM100 TMA 的常见零碎。

### 2.2 初始化与 TMEM 分配（`:247-281`）

- `kNumMulticast>1` 时先 `cute::cluster_sync()`（`:247-249`）。
- **warp1 的 elect_one 线程**初始化所有 barrier（`:251-267`）：`full/empty_barriers[i].init(1)`、
  `with_sf_full_barriers[i].init(kNumMulticast*32)`（scale 转置由 32 通道 warp 完成）、
  `tmem_full/empty_barriers` 分别 init `1` / `kNumMulticast*kNumUMMAStoreThreads`，末尾
  `fence_barrier_init()`。
- **warp2** 用 `Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem)` 分配 TMEM（`:268-270`），
  分配器按 `kNumMulticast` 选 `Allocator1Sm`/`Allocator2Sm`（`:61-63`）。
- 收尾 `cluster_sync()` 或 `__syncthreads()`（`:271`）。

### 2.3 Scheduler：把 GEMM 分组形态折平进一个模板（`:273-281` + `_scheduler.cuh`）

`Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>`
（`_scheduler.cuh:38-52`）用 `if constexpr` 把 6 种 `GemmType` 的 tile 遍历折平：

- `get_next_block`（`:168-196`）里有一句**关键注释**：
  「In persistent kernel: use block_offset=0 (each CTA handles its own output slice, **kNumSMs=1**,
  processes all tiles sequentially)」（`:170-172`），`next_block_idx = (++current_iter)*kNumSMs + 0`。
  ⇒ 这就是 §5.3 生产登记必须传 `kNumSMs=1` 的**同源设计**。
- `get_swizzled_block_idx`（`:105-129`）按 `kIsMulticastOnA` 决定主/次维、用
  `get_num_1d_blocks_per_group`（`:14-34`，在 {8,16} 里选 SMEM 用量最小的分组）做 Z 序邻近分组。
- `get_global_idx`（`:131-166`）用 `if constexpr` 把 `Normal/MGroupedContiguous/MGroupedMasked/
  KGroupedContiguous/…WithPsumLayout` 的「组偏移」折平（`IndexType::{MN,K,SF_K}`）。

### 2.4 三条流水（warp0/1/2 + epilogue warps）

**warp0 = TMA 生产者**（`:290-402`）：对每个 tile、每个 K-block，先等 `empty_barriers[stage_idx]`
（`:315`），再发 A/B（`tma_copy`，`:341-365`；major==K 时）与按 `kNumSFAStagesPerLoad` 节流的
SFA/SFB（`:372-398`），最后 `arrive_and_expect_tx(num_arrival_bytes)`（`:400`）。`num_arrival_bytes`
用 clamp 后的 box 维与 B 的实际 dtype 宽度（`float_e4m3` 1 B 否则减半，`:367-370`）。

**warp1 = UMMA leader**（`:403-534`）：仅 `is_leader_cta` 的 warp1 执行。

- `make_instr_desc_block_scaled<a_dtype,b_dtype,float,float_ue8m0,UMMA_M,UMMA_N,kMajorA,kMajorB>`
  构造 block-scaled 指令描述符（`:408-416`），`mma_t = …::sm100::SM100_MMA_MXF8F6F4_SS`（`:486-487`）。
- A/B 的 SMEM descriptor 用 `lane_idx` 预展开各 stage 的 `desc.lo`（`:420-429`），循环里用
  `__shfl_sync` 取当前 stage 的 base（`:488-491`）。
- 主循环（`:456-527`）：等 `with_sf_full_barriers[stage_idx]`（scale 已由 warp2 转置好，`:460`）→
  每 `kNumSFAStagesPerLoad` 个 K-block 由 `elect_one` 线程发 **UTCCP**
  （`cute::SM100_UTCCP_4x32dp128bit_1cta::copy(sf_desc, tmem_col)`，`:463-483`）→ 对 `BLOCK_K/UMMA_K=4`
  个 UMMA_K=32 子块、`kNumMWaves` 个 M wave 发 `mma_t::fma(a_desc,b_desc,tmem_col,scale_c,desc,sfa,sfb)`
  （`:492-525`），`scale_c = (k_block_idx>0 || k>0)` 即首块不累加（`:519`）。
- 每个 K-block 结束用 `umma_arrive`（`cutlass::arch::umma_arrive`）对 `empty_barriers`（可还给 warp0）
  与末块对 `tmem_full_barriers` 到达（`:445-454`、`:526`）。

**warp2 = UTCCP scale 转置**（`:535-575`）：等 `full_barriers`（数据到）→ 对 SFA/SFB 每 128 元素
块做 `utccp_required_smem_warp_transpose`（`:536-547` 的 lane-swizzled 4×32 转置）→
`fence_view_async_shared` → `with_sf_full_barriers[stage_idx]->arrive(0u)`（`:573`）。

**epilogue warps**（`:576-734`，`kNumNonEpilogueThreads/32 ≤ warp < (nonEpi+kNumUMMAStoreThreads)/32`）：

- 等 `tmem_full_barriers`（UMMA 累加完成，`:597`）；若残差融合再等 `residual_full_barrier`（`:598-601`）。
- 逐 M wave、逐 `BLOCK_N/STORE_BLOCK_N` 子块：`SM100_TMEM_LOAD_32dp32b4x`（float，`:653`）或
  `32dp32b8x`（bf16，`:662`）把 TMEM 搬进寄存器；bf16 + 残差时用
  `add_packed_bf16x2_into_fp32_bits`（`:680-687`，helper `_utils.cuh:122-141`）在寄存器里加残差，
  再 `cast_into_bf16_and_pack`（`:689-693`）。
- 写回：`st_shared` 到 `smem_cd`（含 TMA swizzle 地址计算 `:626-648`）→ `tma_store_fence` +
  `NamedBarrier::sync` → warp0 的 lead 线程发 `SM90_TMA_STORE_2D/3D`（`kWithAccumulation` 时换成
  `SM90_TMA_REDUCE_ADD_2D/3D`，`:704-724`）→ `tma_store_arrive`。

### 2.5 `__global__` wrapper（`:748-824`）

外层 `linear_fp8_sm100_wrapper` 用 `__launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads, 1)`
（`:775`），6 个 descriptor 以 `__grid_constant__` 传入（`:782-787`），再调 `task_impl`。单测与
生产都按这个签名实例化（§5）。

### 2.6 helper 的要点

- **`_sm100_utils.cuh`**：`make_smem_desc`（`:13-28`，version_/lbo_mode_/layout_type_/start_address_>>4…）、
  `make_sf_desc`（`:30-34`，`SWIZZLE_NONE` + stride `8*16`）、`replace_smem_desc_addr`（`:36-41`）、
  `to_umma_layout_type`（`:48-79`）、`make_umma_desc`（`:105-145`，K-major vs MN-major 的
  stride/leading 偏移）、`make_runtime_instr_desc_with_sf_id`（`:147-154`，把 sfa/sfb id 塞进 desc 高 32 位）、
  `get_num_aligned_tmem_cols`（`:156-172`）、`tcgen05_before/after_thread_sync`（`:174-180`）。
- **两个 MMA 结构体**（`:182-230`）：`SM100_MMA_MXF8F6F4_SS`（`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale`）
  与 `SM100_MMA_MXF8F6F4_2x1SM_SS`（`cta_group::2`）。**身体 A 只实例化了前者**（`:487`）——
  单测里可选的 2SM 路径未在这个身体里被激活（生产登记也是 `kNumMulticast=1`，§5.3）。
- **`_tma_utils.cuh`**：`tma_copy`（`:16-91`）按 `num_tma_multicast==1` 分派 `SM90_TMA_LOAD_2D/3D`，
  否则 `SM100_TMA_2SM_LOAD_2D/3D`（`:49-60`、`:76-88`，均在 `__CUDA_ARCH__>=1000` 保护下）；
  另有 tensormap 的 `release/acquire.cta`（`:93-104`）与 `tensormap.replace.tile.global_address`（`:106-116`）。
- **`_epilogue_utils.cuh`**：`EpilogueIdentity`（`:8-14`）与 `EpilogueHeadSplits<kLeft,kMid,kRight>`
  （`:16-26`，把 head 维三段的分裂在 N 索引上重映射 `n_idx + (n_idx+kRight)/(kLeft+kRight)*kMid`）——
  对应「QKV 融合」类 head 切分（`docs/25` 前端 fqkv）。
- **`_types.hpp`**：`MmaKind{BF16,MXFP8FP4}`（`:5-8`）、`GemmType{Normal,MGroupedContiguous,MGroupedMasked,
  KGroupedContiguous,Batched,MGroupedContiguousWithPsumLayout}`（`:21-28`）、`KernelType{Kernel1D1D,
  Kernel1D2D,KernelNoSF}`（`:42-46`）。
- **`_cute_tie.cuh`**：补 CuTe 缺失的 `cute::ignore_t`（`:5-14`）与 1–5 元 `CUTE_TIE` 宏（`:46-60`）——
  一个纯便利层，`task_impl` 未直接使用。

---

## 3. 身体 B：`linear_fp8_1d2d_sm100.cuh` 为什么是「另一套身体」

身体 B（727 行）与被身体 A 完全不同：

- **include 不同**：拉的是 **hopper 的基础设施**（`../hopper/barrier.cuh`、`../hopper/tma.cuh`、
  `../hopper/smem_layout_tma.cuh`、`storage.cuh`，`:35-40`），而不是身体 A 的 `_common.cuh` 家族。
- **风格不同**：用 **CuTe 高层**——`cute::make_tiled_mma`、
  `cute::SM100_MMA_MXF8F6F4_SS<T,T,float,float_ue8m0,MMA_M,MMA_N,K,K>`（`:117-124`）、
  `cute::local_tile`（`:146/149`）、`cute::partition_shape_A/B`、`cute::UMMA::tile_to_mma_shape` +
  `Layout_K_SW128_Atom`（`:174-188`）；身体 A 则是手写 SMEM descriptor + 裸 PTX。
- **有自己的 scale 转置拷贝**：`detail::utccp_required_smem_warp_transpose`（`:46-58`）与身体 A 的
  warp2 版本逐行同构（`kNumUTCCPAlignedElems=128`，`:42`）。
- **签名/阶段不同**：`linear_fp8_1d2d_sm100_task_impl<T_,TMA_A,TMA_B,BiasTensor,TMA_OUT,…,
  NUM_AB_STAGE=8,NUM_ACC_STAGE=2,NUM_C_STAGE=4>`（`:71-90`）——带 `NOBIAS`/`SplitK` 两个 bool，
  无 `GemmType` 模板参数。

**关键静态事实**：全仓 `grep` 显示 `linear_fp8_1d2d_sm100_task_impl` **只有定义、只在
`task_header.cuh:37` 被 `#include`，没有任何调用者**；而公开 pybind 符号 `linear_fp8_1d2d_sm100`
（`runtime_kernel_wrapper_sm100.cu:1100-1106`）的实际 dispatch 走的是
`launch_linear_fp8_1d1d_sm100` / `launch_linear_fp8_1d1d_sm100_splitk`（`:962/977`），
即**身体 A**。⇒ 「1d2d」这个公开名字对应的是身体 A 的 `1d1d` 调用，身体 B 在快照里是**未被接线的
第二实现**（同 `docs/16`/`docs/20`/`docs/22` 反复出现的「已实现未启用」模式）。**仅记录静态事实、
不判对错**。

---

## 4. 与 `docs/07`（Hopper task）/ `docs/21`（Blackwell transpiler）的坐标关系

三篇刚好覆盖「SM100 手工线性 task」与「自动 codegen」两条路：

| 维度 | `docs/07`（Hopper task） | `docs/21`（Blackwell transpiler） | 本文（SM100 手工 task） |
|---|---|---|---|
| 谁生成 | 手写（frontend `register_task`） | 自动（`superoptimize` → transpiler） | 手写（frontend `register_task`） |
| 计算单元 | 单 CTA + `wgmma` | **一对 CTA 的 2SM UMMA** | 身体 A **单 CTA `cta_group::1`**；身体 B CuTe |
| 累加器 | 寄存器 | **TMEM**（`Allocator2Sm`） | **TMEM**（`Allocator1Sm`/`2Sm` 可选，生产选 1Sm） |
| block scale | 无（BF16） | 无（gemm） | **有**——`tcgen05.mma…block_scale` + UTCCP |
| 生产态 | 是（Hopper） | 生成代码，非手写 | 是（SM100 产线 `linear_fp8_sm100` op） |

一句话：**SM100 的手工 `linear_fp8_sm100` 是「Hopper 单 CTA 特化」升级到「TMEM + block-scaled UMMA」，
但并没有采用 `docs/21` 自动路径的 2SM cluster**——2SM 结构体在 `_sm100_utils.cuh` 里保留但产线
不激活。

---

## 5. 生产布线：同一 device 模板的两种实例化（补齐「未与生产路径对跑」）

### 5.1 op 名 → `TaskType` → device 模板

前端 `customized` op 名到 `TaskType` 的映射（`src/kernel/graph.cc`）：

- `"linear_fp8_sm100"` → `register_linear_fp8_sm100_task(..., with_residual=false)` →
  `TASK_LINEAR_FP8_SM100`（`graph.cc:852-853`）。
- `"linear_fp8_with_residual_sm100"` → `register_linear_fp8_sm100_task(..., true)` →
  `TASK_LINEAR_FP8_WITH_RESIDUAL_SM100`（`graph.cc:854-858`）。

`TaskType` 数值落在 `docs/24` 记的 **SM100 ID 块**（230–299）内：
`TASK_LINEAR_FP8_SM100=276`、`TASK_LINEAR_FP8_WITH_RESIDUAL_SM100=277`
（`include/mirage/persistent_kernel/runtime_header.h:163-164`）。

运行时侧：`runtime.cc:1190-1194` 对这两个 `TaskType` 调 `create_tma_desc_by_task(task_desc)`
（注释「FP8 linear tasks need TMA (outside SM100_TMA range)」）；`tma.cuh:784-800` 是
`TASK_LINEAR_FP8_SM100 ({,WITH_RESIDUAL})` 的 TMA descriptor 构造分支（`BLOCK_M=32,BLOCK_N=16,
BLOCK_K=128`，按 A/SFA/SFB/CD 各自编码 tiled map）。

### 5.2 生产生成的 device 调用（`task_register.cc:4817-4868`）

`register_linear_fp8_sm100_task`（`:4772`）生成的模板实参（**逐行核对**）：

- `kMajorA=kMajorB=cute::UMMA::Major::K`、`kGranKA=kGranKB=128`（`:4827-4828`）；
- `BLOCK_M/BLOCK_N/BLOCK_K = 32/16/128`（`:4833`）；
- `kNumGroups=1`（`:4834`）；swizzle `128/128/32`（`:4835`）；
- **`kNumStages=25`**（`:4836`），注释 *"fit persistent kernel 207KB smem budget"*；
- `kNumNonEpilogueThreads=kNumEpilogueThreads=128`（`:4837`）⇒ **256 线程 = 8 warp**；
- **`kNumMulticast=1, kIsMulticastOnA=false`**（`:4838`）；
- **`kNumSMs=1`**（`:4839-4840`），注释 *"persistent kernel: 1 CTA per task, processes all tiles"*；
- `kWithResidual` 由 `params[0]` 决定（`:4841-4843`）；
- `GemmType::Normal`（`:4844`）、`kWithAccumulation=false`（`:4845`）、
  dtype `float_e4m3_t/float_e4m3_t/bfloat16_t`（`:4846-4847`）、**`EpilogueIdentity`**（`:4848`）；
- 6 个 TMA 指针取自 `task_desc->input_tma_desc_ptrs[0/1/2/3/4]` 与 `output_tma_desc_ptrs[0]`
  （`:4852-4868`，A←in[0]、SFA←in[1]、B←in[2]、SFB←in[3]、residual←in[4] 或 CD 顶替）。

### 5.3 `kNumSMs=1` 的 HACK 与「one task = one CTA」的冲突

`register_linear_fp8_sm100_task` 顶部有一段**明确标注 HACK** 的注释（`:4817-4825`）：

> `blockIdx.x` 在持久化内核里是 **worker 索引**（0..num_workers-1），不是 task 的 grid block 索引；
> 由于不能覆盖 `blockIdx.x`，就把 `kNumSMs` 设成「≥ num_workers」的值（取编译期 `MPK_MAX_NUM_WORKERS`），
> 使首块 `block_idx = blockIdx.x` 恒 `< kNumSMs`、该 CTA 一趟处理完自己的所有 tile。

这与 `_scheduler.cuh:170-172` 的注释是同一条设计：**手写 `linear_fp8_sm100` 本来是「多维 grid 的
多 CTA GEMM」**（`blockIdx.x` 当 tile scheduler），而 MPK 的**持久化 worker 语义要求「一个 task =
一个 CTA 跑完 6 维 grid」**——于是生产侧用 `kNumSMs=1` 把它退化成「单 CTA 顺序遍历所有 tile」。
这正是 `docs/07`「一个 task = 一个 CTA」在 SM100 block-scaled 路径上的**具体适配痕迹**，也解释了
为什么 `kNumStages` 要压到 25 塞进 207 KB（单 CTA 要同时装下整条深流水）。

### 5.4 同族：BF16 版 `linear_sm100_mpk`

生产 SM100 的 **BF16** 线性 task 是另一份 device 身体 `linear_sm100_mpk.cuh`（737 行）的
`linear_sm100_mpk_task_impl`（`:43`），由 `register_linear_sm100_task`（`task_register.cc:2013`）与
`register_splitk_linear_sm100_task`（`:2178`）生成调用（`:2151`/`:2304`，模板实参
`MMA_M=128, MMA_N=16, SplitK=false/true`），登记为 `TASK_LINEAR_SM100=253` /
`TASK_LINEAR_WITH_RESIDUAL_SM100=252`（`runtime_header.h:139-140`）；op 名 `"linear_sm100"` /
`"splitk_linear_sm100"` / `"linear_with_residual_sm100"`（`graph.cc:626-641`）。它与 FP8 版**同为
「单 CTA + CuTe 风格 + TMEM」**，但**无 block scale**——可与 `fp8_group_gemm_sm100`（MoE 版，
`docs/31` §5）三足对照。unit-test 侧 `tests/runtime_python/blackwell/sm100_linear/runtime_kernel_wrapper_sm100.cu`
也实例化同一 `linear_sm100_mpk_task_impl`（`:129/438`），pybind 名 `linear_sm100_mpk`（`:662`）。

### 5.5 unit-test wrapper 侧（与生产同模板、异实参）

`tests/runtime_python/blackwell/sm100_linear_fp8/runtime_kernel_wrapper_sm100.cu`：

- `launch_linear_fp8_1d1d_sm100`（`:419`）的 `kernel_ptr` 就是
  `kernel::linear_fp8_sm100_wrapper<…GemmType::Normal, EpilogueIdentity>`（`:606-633`），grid
  `(kNumSMs,1,1)`、block `256`、`cluster (1,1,1)`（`:643-645`）——与生产的模板实参**同构**。
- splitk 路径（`:674`）的 `kernel_ptr` 是 `kernel::linear_fp8_sm100_splitk_wrapper`（`:811-812`），
  其头 `blackwell/linear_fp8_sm100_splitk.cuh` **在快照里不存在**（`include :19` 无条件包含）——
  `docs/31` §6 的「缺头文件」headline 在此**得到别名确认**：它对应的正是 splitk 分支的 device 身体。
- pybind 只暴露 `linear_fp8_1d2d_sm100`（`:1101`）与 `supported_dense_gemm_shapes`（`:1104`）；
  `test_linear_1d2d_fp8.py` / `test_runtime_quantize_linear_fp8.py` 以 `linear_fp8_1d2d_sm100`
  名字调它，实际执行身体 A（§3）。

---

## 6. 判据映射与与 `docs/31` 的关系

- **补的尾巴**：`docs/31` §10 两条（`linear_fp8_sm100` 家族内部 / 与生产路径对跑）。
- **修正/加强 `docs/31`**：§4 的「11 个头」实为 **10 个**文件；§6 的「缺头文件」是 splitk device
  身体的头（`linear_fp8_sm100_splitk.cuh`），且本文记下**身体 B（`linear_fp8_1d2d_sm100.cuh`）未被接线**。
- **新增证据**：生产 `register_linear_fp8_sm100_task` 的 `kNumSMs=1` HACK 与 `kNumStages=25`
  SMEM 预算（`:4817-4840`）——把「one task = one CTA」从 `docs/07` 的 Hopper 形态推进到 SM100
  block-scaled 形态。
- **判据映射**：与 `docs/29` §6 / `docs/30` §7 / `docs/31` §9 **完全一致**——本层把「一个 device task
  当成一整颗 kernel」由 `register_task` 生成/由 host 直接 launch，**它贡献的只是 C3（融合范围）里
  「单 device task 数值正确/可被表达」的一环，不拿 C1/C2/C4/C5**。唯一与 C1–C5 有关的事实：
  `kNumSMs=1` 的适配正是「把多 CTA grid 收进单 CTA」以适配常驻 worker——**C1/C2（单次 launch/常驻）
  由运行层（`docs/06`/`docs/22` §8）拿到，task 身体本身不产生**。

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，未取任何 device
  计数；对 `linear_fp8_1d2d_sm100.cuh` 的「无调用者」是**静态 `grep`** 结论，不构成运行时判定。
- **未展开**：`fp8_group_gemm_sm100.cuh` 的 UMMA 主循环 / UTCCP 序列 / epilogue scatter 仍未逐行
  （本文把重心放在 `linear_fp8_sm100` 家族与生产布线）；身体 B 的 CuTe 主循环只核了结构与惯用法，
  未逐行走完 727 行；`mla_mtp_tpN_main` 主体仍未走（留待后续）。
- **生产对跑未做**：DeepGEMM SM100 mega 路径（SM100 专属）本机不具备，未对照；`docs/24` 的 SM100
  `TaskType` 登记只复核了 `linear_fp8`/`linear` 两支，未全表复核。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/31` 同快照）。
- **身体 A**：`include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100.cuh`
  （`task_impl :45-746`、`#if __CUDA_ARCH__>=1000 :58`、`Allocator1Sm/2Sm :61-63`、TMEM 列
  `:159-171`、TMA prefetch `:173-182`、SMEM 布局 `:184-243`、barrier 初始化 `:247-271`、
  `Scheduler` 构造 `:273-281`、warp0 TMA `:290-402`、warp1 UMMA `:403-534`（`mma_t :486-487`）、
  warp2 UTCCP `:535-575`、epilogue `:576-734`、TMEM free `:736-739`、`__global__ wrapper :775-824`）。
- **helpers**：`_scheduler.cuh`（`get_next_block :168-196`，`kNumSMs=1` 注释 `:170-172`）、
  `_epilogue_utils.cuh :8-26`、`_sm100_utils.cuh`（`make_smem_desc :13-28`、`make_sf_desc :30-34`、
  `make_umma_desc :105-145`、`make_runtime_instr_desc_with_sf_id :147-154`、`get_num_aligned_tmem_cols
  :156-172`、`SM100_MMA_MXF8F6F4_SS :182-205`、`_2x1SM_SS :207-230`）、`_tma_utils.cuh :16-116`、
  `_utils.cuh :37-141`、`_cute_tie.cuh :5-60`、`_types.hpp :5-46`、`_common.cuh :1-8`。
- **身体 B**：`…/linear_fp8_1d2d_sm100.cuh`（`detail::utccp_required_smem_warp_transpose :46-58`、
  `task_impl :71-90`、CuTe `make_tiled_mma`/`local_tile`/`tile_to_mma_shape :117-188`）。
- **生产**：`src/kernel/task_register.cc`（`register_linear_sm100_task :2013`、
  `linear_sm100_mpk_task_impl 调用 :2151/2304`、`register_splitk_linear_sm100_task :2178`、
  `register_linear_fp8_sm100_task :4772`、**HACK 注释 :4817-4825**、模板实参 `:4826-4868`）；
  `src/kernel/graph.cc`（`linear_sm100`/`splitk_linear_sm100`/`linear_with_residual_sm100 :626-641`、
  `linear_fp8_sm100`/`linear_fp8_with_residual_sm100 :852-858`）；
  `include/mirage/persistent_kernel/runtime_header.h:139-140,163-164`；
  `src/kernel/runtime.cc:1188-1194`；`include/mirage/persistent_kernel/tma.cuh:784-800,1487-1494`。
- **unit test**：`tests/runtime_python/blackwell/sm100_linear_fp8/runtime_kernel_wrapper_sm100.cu`
  （`include :19`、`launch_linear_fp8_1d1d_sm100 :419`、`kernel_ptr :606-633`、splitk `:674/811-812`、
  pybind `:1100-1106`）；`tests/runtime_python/blackwell/sm100_linear/runtime_kernel_wrapper_sm100.cu`
  （`:129/438/662`）；`tests/runtime_python/blackwell/sm100_linear_fp8/test_linear_1d2d_fp8.py`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量 / 单测层，同 `docs/28`–`docs/31`）；论及「手工 kernel 可包成 task」见 `docs/05`。
- 同专题相关篇：`docs/31`（本文补其 §10 尾巴）、`docs/07`（Hopper 单 task device）、`docs/21`
  （Blackwell transpiler 2SM codegen）、`docs/24`（`TaskType` 目录）、`docs/25`（前端 fqkv/head
  split）、`docs/27`（FP8 UE8M0 权重预处理）、`docs/22` §8（C 判据总映射，本层不贡献）。
- 交叉专题：DeepGEMM mega 家族前向/反向验证见 `../mega-moe/`（本专题不重复）。
