# MPK SM100 2SM MLA 的 CUTLASS collective_builder 侧走读：QK/PV atom、TMA 描述符与 UMMA 序列（P138-1）

> 定位：本文补掉 `docs/36-mpk-sm100-mla-2sm.md` **§8「局限」明留的尾巴**（原话）——
>
> > **「整颗 CUTLASS kernel」的细节未逐行**：QK/PV 的 `CollectiveMmaQK/PV` 内部（TMA 描述符布局、
> > `AtomThrShapeMNK` 的 CuTe atom、UMMA 指令序列）来自 CUTLASS `collective_builder`，本文只核到
> > 「本文件如何实例化与驱动它们」，未展开 CUTLASS 侧（`deps/cutlass`）实现。
>
> `docs/36` 走完了 `mla_sm100_2sm.cuh` **本体**的六个 Section（Pow2 助手 / CustomStride-Gather / tile
> scheduler / reduction kernel / 主 MLA kernel / device 层）；本文把它的**依赖方**打开——即
> `CollectiveBuilder<...>::CollectiveOp` 究竟展开成了什么：哪一颗 UMMA atom、哪一条 TMA atom、
> SMEM 布局如何选、`AtomThrShapeMNK` 从哪来、以及 MPK 到底用了这颗 CUTLASS 的**哪些部分**。
>
> 快照：MPK `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/51` 同）；
> CUTLASS 侧 `deps/cutlass` 由该 commit 的 submodule 钉死在 **`f3fde583`**（`git submodule status`：
> `-f3fde58372d33e9a5650ba7b80fc48b3b49d40c8 deps/cutlass`，快照里可核对）。
> **纯源码走读（资料分析），无新实验**——SM100（`sm_100f`/`sm_100a`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **MPK 把 `CollectiveBuilder` 当「类型工厂 + 参数打包器」，而不是拿来就跑的 mainloop**：
   `mla_sm100_2sm.cuh` 从 `CollectiveMmaQK/PV` 里**只取类型**（`TiledMma`/`CtaShape_MNK`/`SmemLayout*`/
   `Params::TMA_*`/`AtomThrShapeMNK`，`:707-784`、`:903-907`）与**一个 host 侧函数**
   `to_underlying_arguments`（`:954-1000`）；真正的设备侧主循环是它自己手写的 `load_tma`/`mma`/`softmax`
   （`docs/36` §5），`cute::gemm(tiled_mma_qk/pv, ...)`（`:2269/2299/2329/2369`）由它自己的 fragment
   `tSrQ/tSrKC/tOrP/tOrVC`（`TiledMmaQK::make_fragment_A/B`，`:2221-2224`）驱动。
   **NVIDIA 的 `CollectiveMma::load`/`CollectiveMma::mma` 这两个成员函数在本文件里从未被调用**。
   这与 `docs/33`/`docs/34` 记的「SM100 task 手拼 TMA+UMMA」是同一路数，也和 DeepGEMM 用 CUTLASS
   当「atom 库」的选择同构。

2. **QK 与 PV 两条 GEMM 的 2SM 原子是同一颗 UMMA 指令的两种 K/N 分块**：二者都用
   `Schedule = KernelTmaWarpSpecialized2SmSm100`（`:689-692`）→ `is_2sm=true` →
   `TiledMma = SM100_MMA_F16BF16_2x1SM_SS<bf16,bf16,f32, M=128, N, K-major>`（builder
   `sm100_umma_builder.inl:204/218`、`sm100_common.inl:374-395`），设备侧落到一条 PTX
   `tcgen05.mma.cta_group::2.kind::f16`（`cute/arch/mma_sm100_umma.hpp:446`）。因为 `M=128`、
   `ThrID=Layout<_2>`（traits `:1695`），`AtomThrShapeMNK` 恒为 `Shape<2,_1,_1>`，`CtaShape_MNK =
   TileShape/2`——这就是 MPK 文件里 `static_assert(QK.AtomThrShapeMNK == PV.AtomThrShapeMNK, "schedule
   must match")`（`:740-743`）在 CUTLASS 侧的根。

3. **TMA 走 `SM100_TMA_2SM_LOAD`（`cta_group::2` 的 bulk tensor 拷贝）**：builder 的
   `sm100_cluster_shape_to_tma_atom_A/B`（`sm100_common.inl:199/235`）在 `size(AtomThrID)==2` 且
   cluster 的对应维为 1 时返回 `SM100_TMA_2SM_LOAD{}`（`:207`/`:243`），其 `copy()` 落到
   `cp.async.bulk.tensor.Nd.cta_group::2.shared::cluster.global...`（`cute/arch/copy_sm100_tma.hpp:92` 等）。
   `TmaTransactionBytes` 里显式乘 `size(AtomThrShapeMNK)`（mainloop `:235-237`），因为
   **一对 CTA 只有 leader 发 TMA、但要为两个 CTA 的 SMEM 计账**。这与 `docs/36` §3.3 观察到的
   `is_mma_leader_cta` 门控、以及 `PipelineTmaUmmaAsync` 里 `producer_arv_cnt = 1` 是同一件事。

判据口径同 `docs/29`–`docs/36`：**本工作只贡献 C3**（单 kernel 可表达）；C1/C2/C4/C5 由运行层拿，
且本 task 连「进常驻内核」都还没接（`docs/36` §7）。

---

## 1. 定位：本文覆盖的是哪一层

`docs/36` 的地图停在这里：

| 层 | `docs/36` | 本文 |
|---|---|---|
| MPK 文件本体（骨架/主循环/device 层） | ✅ 全走 | 只引其**实例化点**做坐标 |
| **CUTLASS `CollectiveBuilder` 分派链** | ❌（§8 明留） | ✅ §2 |
| **CUTLASS `TiledMma`（UMMA atom / `AtomThrShapeMNK`）** | ❌ | ✅ §3 |
| **CUTLASS SMEM 布局选择（swizzle atom）** | ❌ | ✅ §4 |
| **CUTLASS TMA atom（`SM100_TMA_2SM_LOAD`）** | ❌ | ✅ §5 |
| **CUTLASS mainloop（`CollectiveMma` 的 load/mma/pipeline）** | ❌ | ✅ §6（并说明 MPK **没用**它） |

依赖方路径前缀：CUTLASS 侧均在 `deps/cutlass/`，下称 `$C`。

**接线事实（静态 `grep`，可复现）**：`mla_sm100_2sm.cuh` 对 `CollectiveMmaQK/PV` 的全部引用
（`grep -n 'CollectiveMmaQK::\|CollectiveMmaPV::'`）只有 21 处，全部是 `using`/`static_assert`/
`to_underlying_arguments`/`Params::TMA_*`；**没有一处是 `CollectiveMmaQK::load(...)` / `::mma(...)` /
`::prefetch_tma_descriptors()` / `::load_init(...)`**。这坐实 headline 1。

---

## 2. 从 `CollectiveBuilder` 到 `CollectiveMma`：分派链逐段

MPK 文件 (`:693-742`) 的写法是标准 CUTLASS 用户写法：

```
CollectiveMmaQK = CollectiveBuilder<
    arch::Sm100, OpClassTensorOp,
    Element, TensorStride, Alignment,        // A: bf16, (int64,_1,int64), 8
    Element, TensorStride, Alignment,        // B: 同上
    ElementAcc,                              // f32
    TileShapeQK, ClusterShape, StageCount<StagesQK>, Schedule>::CollectiveOp;
```

`Element = bfloat16`、`TensorStride = Stride<int64_t,_1,int64_t>`、`Alignment = 128/16 = 8`、
`TileShapeQK = Shape<128,128,64>`、`ClusterShape = Shape<_2,_1,_1>`、
`StagesQK = 24/sizeof(bf16) = 12`、`Schedule = KernelTmaWarpSpecialized2SmSm100`（`docs/36` §2/§4）。

### 2.1 特化点

`$C/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl:168` 起的
`CollectiveBuilder<arch::Sm100, OpClassTensorOp, ...>` 特化负责稠密、非 block-scaled 的 SM100 GEMM。
其中关键三行：

- `:204` `is_2sm = is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag> || ...`。`Schedule` 的定义在
  `$C/include/cutlass/gemm/dispatch_policy.hpp`：`KernelTmaWarpSpecialized2SmSm100 final :
  KernelSchedule2Sm, KernelScheduleSm100DenseGemm{}`（`:715`，基类 `KernelSchedule2Sm` 在 `:706`）。
  故 `is_2sm = true`。
- `:218-221` `TiledMma = sm100_make_trivial_tiled_mma<ElementAMma, ElementBMma, ElementAcc,
  product_each(TileShape_MNK), ClusterShape_MNK, UmmaMajorA, UmmaMajorB, BuilderScheduleTag>()`——
  见 §3。
- `:227` `AtomThrShapeMNK = Shape<decltype(shape<0>(TiledMma::ThrLayoutVMNK{})), _1, _1>`——
  「V 维」= `ThrID` 的 size = 2（§3.2）。
- `:307-320` `CollectiveOp = CollectiveMma<DispatchPolicy, TileShape_MNK, ElementA, StrideA, ElementB,
  StrideB, TiledMma, GmemTiledCopyA, SmemLayoutAtomA, void, identity, GmemTiledCopyB, SmemLayoutAtomB,
  void, identity>`，其中 `DispatchPolicy = MainloopSm100TmaUmmaWarpSpecialized<PipelineStages,
  SchedulerPipelineStageCount, AccumulatorPipelineStageCount, ClusterShape_MNK>`（`:291-305`，
  定义在 `dispatch_policy.hpp:1005`）。

> 注意 `SmemCopyAtomA/B = void`：SM100 的 UMMA 从 SMEM 直接吃描述符，不需要 `ldmatrix` 式的
> SMEM→寄存器 copy atom（mainloop `:159` 的 `static_assert(is_void_v<SmemCopyAtomA>)` 会钉住）。

### 2.2 落到哪个 mainloop 实现

`CollectiveMma` 的 `DispatchPolicy = MainloopSm100TmaUmmaWarpSpecialized<...>` 对应特化文件
`$C/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:80`。它是一个标准
`CollectiveMma`：`AtomThrShapeMNK`（`:105`）、`CtaShape_MNK = shape_div(TileShape, AtomThrShapeMNK)`
（`:110`，即 `(64,128,64)` 的 per-CTA tile）、`MainloopPipeline = PipelineTmaUmmaAsync<Stages,
ClusterShape, AtomThrShapeMNK>`（`:150-153`）、`SharedStorage`（`:220-233`）、`TmaTransactionBytes`
（`:235-237`）都在这里。

---

## 3. `TiledMma`：2×1SM UMMA atom 与 `AtomThrShapeMNK`

### 3.1 `sm100_make_trivial_tiled_mma` 的分派

`$C/include/cutlass/gemm/collective/builders/sm100_common.inl:448`：

- `is_base_of_v<KernelSchedule2Sm, BuilderScheduleTag>` 为真 → 调
  `sm100_make_2sm_trivial_tiled_mma`（`:374-455`）。
- 该函数 `static_assert(M == 128 || M == 256)`（`:377`）。本例 `M = size<0>(TileShapeQK) = 128`
  → 合法。
- `ElementAMma = bf16` 分支（`:388-395`）返回
  `make_tiled_mma(SM100_MMA_F16BF16_2x1SM_SS<bf16,bf16,f32, M=128, N=128, UmmaMajorA, UmmaMajorB,
  ScaleIn::One, ScaleIn::One>{})`。

`UmmaMajorA/B` 由 `tag_to_umma_major_A/B<GmemLayoutATag>()` 推出（builder `:201-202`）；本例
`TensorStride = Stride<int64_t,_1,int64_t>` 是 K-major，故 A/B 均 `UMMA::Major::K`（`N` 维 stride 为 1
时才是 MN-major；PV 的 B 用 `TransposeTensorStride = select<1,0,2>(TensorStride)`，`:711-713`，故 PV 的
B 是 MN-major）。

### 3.2 atom 的 2SM 语义：`ThrID = Layout<_2>`

`$C/include/cute/atom/mma_traits_sm100.hpp:1677` 起是 `MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS<...>>`：

- `:1689` `FrgTypeC = UMMA::tmem_frg_2sm<c_type>`——累加器是**一对 CTA 共享的 TMEM**（这正是
  `docs/36` §3.2 `Allocator2Sm` + 对半切 TMEM 的 CuTe 表达）。
- `:1695-1698` `ThrID = Layout<_2>`；`ALayout = Layout<Shape<_2, Shape<Int<M/2>,Int<K>>>,
  Stride<Int<M/2>, Stride<_1, Int<M>>>>`——**第一个 mode 就是「2 个 CTA」**，每个 CTA 拿 `M/2` 行
  A；`BLayout`/`CLayout` 同构（B 切 N、C 切 M）。这就是 `AtomThrShapeMNK = Shape<2,_1,_1>` 的
  CuTe 出处，也是「一对 CTA 一体」的第一性设定。
- `:1709-1735` `mma_unpack` 把描述符寄存器 `desc_a/desc_b`、TMEM 指针 `tmem_c`、运行时 idesc
  交给 `SM100_MMA_F16BF16_2x1SM_SS::fma`。

`$C/include/cute/arch/mma_sm100_umma.hpp:422` 是 atom 的 PTX 层：

```
:426  static_assert(M == 128 || M == 256, ... "2 CTA cluster MMA");
:446  "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {%5..%12}, p;"
```

注意 `cta_group::2`（对照 1SM 版 `SM100_MMA_F16BF16_SS` 的 `cta_group::1`，同文件 `:100`），且 `fma`
外层包 `if (cute::elect_one_sync())`（`:444`）——**一条 2SM UMMA 由单线程发射、作用于一对 CTA 的
TMEM**。这与 MPK `docs/36` §3.3「只有 `mma_leader` CTA 发 UMMA」在指令层完全对应：device 侧
`is_mma_leader_cta` 门控是 CUTLASS 惯用法的显式版本。

---

## 4. SMEM 布局选择：`sm100_smem_selector`

`$C/include/cutlass/gemm/collective/builders/sm100_common.inl:82` 的 `sm100_smem_selector<major,
ElementType, BLK_MN, BLK_K>()` 按 (MN,K) 块形状挑一颗 swizzle atom：

- `major == K` 分支（`:120-138`）：优先 `Layout_K_SW128_Atom`（BLK_K0 能被其 K 维整除），否则 SW64/SW32/
  INTER 逐级降。
- `major == MN` 分支（`:92-119`）：非 4 字节元素走 `Layout_MN_SW128_Atom` 起降；`sizeof==4`（tf32）
  特判只允许 `Layout_MN_SW128_32B_Atom`。

本例 QK 的 A/B 都是 K-major、bf16、`SmemShape_M/K = 128/64`（M 被 2SM 对半成 `SmemShape_M`，
见 builder `:251-254` 的 `SmemShape_M/N/K`），PV 的 B 是 MN-major。

> **这与 `docs/51`（`smem_layout.cuh`）是同一枚「128 B swizzle」的两个世界**：MPK 生产 task 手写
> `smem_row` 位切片 swizzle（`docs/51` §1），而 CUTLASS 走 `UMMA::Layout_K_SW128_Atom` 这种
> **布局类型即 ABI** 的表达（`docs/51` §4 正好记了 `SwizzleOffsetCalculator` 显式指 `cute/swizzle.hpp`）。
> 两条路径算的是同一个 XOR swizzle，写法完全不同——`docs/51` 的「三套写法」在这里出现第四、五种
> （`Layout_K_SW128_Atom` / `Layout_MN_SW128_Atom`）。
> `sm100_smem_selector` 依赖 `BLK_MN0/BLK_K0 % 8 == 0`（`:84-85`），本例 64/128 皆满足。

---

## 5. TMA atom：`SM100_TMA_2SM_LOAD` 与 `cta_group::2`

### 5.1 builder 选 atom

`sm100_cluster_shape_to_tma_atom_A`（`sm100_common.inl:199`）/`_B`（`:235`）：当
`size(atom_thr_id) == 2`（2SM）且静态 cluster 的对应维为 1 时，返回 `cute::SM100_TMA_2SM_LOAD{}`
（A 看 `size<1>(cluster)==1`，`:207`；B 看 `size<0>(cluster)==2`，`:243`）；否则返回 `_MULTICAST`
（A 用于 N 维 >1，B 用于 M 维 >2）。本例 `ClusterShape = (2,1,1)`，故 **A 与 B 都是
`SM100_TMA_2SM_LOAD`（非 multicast）**。

### 5.2 atom 的 PTX 层

`$C/include/cute/arch/copy_sm100_tma.hpp:182` 的 `SM100_TMA_2SM_LOAD` 按维数转发到
`SM100_TMA_2SM_LOAD_{1..5}D`，底层 asm 是
`cp.async.bulk.tensor.Nd.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`
（1D `:66`、2D `:92`、…、5D `:170`）。对照 1SM 的 `SM90_TMA_LOAD`（`copy_sm90_tma.hpp`）——差别是
`cta_group::2` 与 `shared::cluster`：**TMA 硬件一次为「一对 CTA」把数据放进 cluster 内共享的 SMEM
窗口**，与 §6.3 的 producer 计账一致。

### 5.3 atom 的 host 构造：`make_tma_atom_A/B_sm100`

`$C/include/cute/atom/copy_traits_sm100_tma.hpp:404`（A）/`:463`（B）的 `make_tma_atom_A_sm100_
sm100` 是新的（`#if 0` 调试打印仍留在源里）：它把 `gtensor(M,K,..)`、分片后的 `slayout`、
`mma_tiler`、`TiledMMA`、`cluster_shape(VMNK)` 折成 TMA descriptor；multicast 维度由
`size<2>(cluster)`（A）/`size<1>(cluster)`（B）决定；`cta_v_tile` 取 `mma.thrfrg_A/B(g_tile)`
的 value 布局。mainloop 的 `Params` 里 `TMA_A = decltype(make_tma_atom_A_sm100<TmaInternalElementA>(
GmemTiledCopyA{}, ...))`（`sm100_mma_warpspecialized.hpp:305-316`）。
`TmaInternalElement` 有 `float→tfloat32_t` 特判（`:208-209`），本例 bf16 无此转换。

> MPK 文件把这些具体化为 5 个 TMA 别名（`:903-907`）：`TmaLoadQLatent/QRope =
> CollectiveMmaQK::Params::TMA_A`、`TmaLoadCLatent/KRope = ...TMA_B`、
> `TmaLoadCLatentTranspose = CollectiveMmaPV::Params::TMA_B`——即 **QK 与 PV 共用 A-atom 类型、
> 用 K-major/mn-major 两种 B-atom**，与 `docs/36` §4 的「K/V 共用一段 SMEM staging」相互印证。

---

## 6. mainloop `CollectiveMma`：MPK 取用了什么、没取用什么

### 6.1 CUTLASS 侧提供了什么

`sm100_mma_warpspecialized.hpp` 的 `CollectiveMma` 提供一整套 mainloop API：

- `load_init`（`:496-546`）：切 `gA_mkl/gB_nkl`、`cta_mma.get_slice(blockIdx.x % size(AtomThrID))`
  （`:519`，**2SM 下每个 CTA 取自己的 slice**）、`tma_partition`、multicast mask。
- `load`（`:586-627`）：`while(k_tile_count)` 循环，`producer_try_acquire`/`producer_acquire`，
  然后 `if (cute::elect_one_sync())` 里**单线程发两条 TMA**（A 和 B，`:614-618`）。
- `load_tail`（`:629-645`）：`producer_tail` 防 cluster 内 CTA 提前退出。
- `mma`（`:649-716`）：`consumer_wait` → `for k_block: cute::gemm(tiled_mma, tCrA, tCrB,
  accumulators)`（`:699`）→ `consumer_release`；`accumulate_` 首拍 `ScaleOut::Zero` 后转
  `ScaleOut::One`（`:686/706`）——即首 k_block 覆写累加器、其后累加。
- `mma_init`（`:585`）：`TiledMma::make_fragment_A/B(sA/sB)` 生成描述符片段，`static_assert` 管线
  阶段数匹配（`:586-590`）。
- `TmaTransactionBytes`（`:235-237`）：`bits_to_bytes(size(AtomThrShapeMNK) * cosize(take<0,3>
  (SmemLayoutA)) * sizeof_bits_v<ElementA>) + ...B`——**乘 `size(AtomThrShapeMNK)=2`**，因为一对 CTA
  共享一次 TMA 的到达字节。

### 6.2 MPK 侧实际用法

对照 `docs/36` §5：MPK **没有**调 `CollectiveMma::load/mma/load_init/mma_init`，而是：

- 自建 `load_tma`（`:1835-2184`）发自己分片的 TMA（`TmaLoad*` atom 类型来自 CUTLASS，§5.3）；
- 自建 `mma`（`:2185-2388`），用 `TiledMmaQK/PV::make_fragment_A/B`（`:2221-2224`）造
  `tSrQ/tSrKC/tOrP/tOrVC`，再在**自己的 online-softmax 结构里**逐 k_block 调
  `cute::gemm(tiled_mma_qk/pv, ...)`（`:2269/2299/2329/2369`），并自己管 TMEM 偏移
  （`tStS.data()=kS0/kS1`、`tOtO.data()=kO0+j*kSizeAccO`，`:2263`/`:2321`）。
- 自建五条 pipeline 的驱动（`PipelineLoadQK/S/P/O` 类型里 `PipelineTmaUmmaAsync` 来自 CUTLASS，
  `docs/36` §4）。

**即：CUTLASS 提供「指令/描述符/布局/管线原语」，MPK 提供「查表注意力的数据流编排」。**
这正是 `docs/36` §7 判据段说的「整颗 kernel 形态」为何不能直接搬进常驻内核的另一面——MPK 只借了
CUTLASS 的积木，避免了 `device::MLA::run()` 那种「发一次、跑完退」的整颗 executor。

### 6.3 2SM 的「单 leader 发 TMA/UMMA」

`PipelineTmaUmmaAsync`（`$C/include/cutlass/pipeline/sm100_pipeline.hpp:537`）：

- `:562-564` `multicast_consumer_arrival_count = size<0>(cluster)/size<0>(atom_thr_shape) +
  size<1>(cluster)/size<1>(atom_thr_shape) - 1`——2SM 下 `(2/2 + 1/1) - 1 = 1`，即**消费者的到达计数
  按「一对 CTA 算一个」**。
- `:566` `producer_arv_cnt = 1`——**只有 1 个线程（leader CTA 的 elect_one）为整对 CTA 计数**。
- `:597-617` `init_masks` 用 `AtomThrShape_MNK` 算 multicast mask（`calculate_multicast_mask<kRowCol>`）。

对应 MPK：`docs/36` §3.3 的 `pipeline_load_qk_params.is_leader = lane_predicate && role==kLoad &&
is_mma_leader_cta`（`:1134-1135`）——CUTLASS 的 `producer_arv_cnt=1` 是编译期写死的「只有一个生产者」，
MPK 的 `is_mma_leader_cta` 是同一语义的运行时门控。**两者在 2SM 下必须一致，否则 mbarrier 的到达
计数会多算一倍**。

---

## 7. 逐条对照：MPK 实例化点 ↔ CUTLASS 展开

| MPK 文件（`mla_sm100_2sm.cuh`） | CUTLASS 侧（`deps/cutlass@f3fde583`） | 说明 |
|---|---|---|
| `Schedule = KernelTmaWarpSpecialized2SmSm100` `:689-692` | `dispatch_policy.hpp:715`（`:KernelSchedule2Sm` `:706`） | `is_2sm=true` 的判据 |
| `CollectiveBuilder<Sm100,OpClassTensorOp,...>` `:693-711`/`:720-742` | `sm100_umma_builder.inl:168` 特化 | 稠密非 block-scaled SM100 |
| `TiledMmaQK/PV` `:707`/`:738` | `sm100_common.inl:374-395` → `SM100_MMA_F16BF16_2x1SM_SS` | M=128 → 2×1SM |
| `AtomThrShapeMNK` `:740` | `sm100_umma_builder.inl:227` ← `mma_traits_sm100.hpp:1695` `ThrID=Layout<_2>` | 恒 `Shape<2,_1,_1>` |
| `static_assert(QK==PV AtomThrShapeMNK)` `:740-743` | 两条 builder 调用同 `Schedule`/`ClusterShape` | 「schedule must match」 |
| `CtaShapeQK/PV` `:708`/`:734` | `sm100_mma_warpspecialized.hpp:110` `shape_div(TileShape,AtomThr)` | `(64,128,64)` / `(64,256,32)` |
| `SmemLayoutA/KC/VC` `:778-784` | `sm100_common.inl:82` `sm100_smem_selector` | K-major/mn-major swizzle |
| `TmaLoad*` (5 别名) `:903-907` | `copy_traits_sm100_tma.hpp:404/457` `make_tma_atom_{A,B}_sm100` | A 共用、B 两态 |
| `to_underlying_arguments(...)` `:954-1000` | mainloop `CollectiveMma` 的 host API | 唯一用到的成员函数 |
| `TiledMmaQK/PV::make_fragment_A/B` `:2221-2224` | `sm100_mma_warpspecialized.hpp:585-600` 同款调法（MPK 自建） | 描述符片段 |
| `cute::gemm(tiled_mma_*, ...)` `:2269/2299/2329/2369` | `sm100_mma_warpspecialized.hpp:699` 同款调法（MPK 自建） | → `tcgen05.mma.cta_group::2.kind::f16` `mma_sm100_umma.hpp:446` |
| `is_mma_leader_cta` 门控 `:1102-1104`/`:1134-1135` | `PipelineTmaUmmaAsync` `producer_arv_cnt=1` `sm100_pipeline.hpp:566` | 2SM 单 leader 发 TMA/UMMA |
| `PipelineLoadQK = PipelineTmaUmmaAsync<...>` `:749-752` | `sm100_pipeline.hpp:537` | 含 `AtomThrShapeMNK` |

---

## 8. 判据与意义

- **判据**：本工作**只贡献 C3**（一颗可编译、可被单测对标 FlashInfer/PyTorch 的单 kernel）。
  CUTLASS 侧完全是编译期类型展开，**不引入任何运行期协调**——它既不满足 C1（单次 launch 整图）/
  C2（常驻）/ C4（跨算子设备内协调）/ C5（去中心化调度），与 `docs/36` §7 一致。
- **对 megakernel 专题的意义**：这一层回答了「MPK 生产 task 为什么能『手拼 TMA+UMMA』」——
  因为 CUTLASS 的 `CollectiveBuilder` 已经把「SM100 2SM atom 选择 + SMEM swizzle + TMA 描述符 +
  mbarrier 到达计数」封装成**可在编译期展开的类型**，MPK 只需 `using` + `to_underlying_arguments`，
  就可以在自己的常驻内核里复用这些原语，而不必搬 `device::MLA::run()` 那种整颗 executor。
  这与 `docs/22`（Blackwell transpiler 的 2SM UMMA codegen）、`docs/33`/`docs/34`（手写 SM100 task
  的 UTCCP/UMMA/MXF8F6F4）合起来，构成「SM100 2SM 原语」在 MPK 仓库里的**三条来源**：自动 transpiler、
  手工 device task、CUTLASS atom 库。
- **对 DeepGEMM 的旁证**：DeepGEMM 的 SM100 kernel 同样以 CUTLASS/CuTe 为「atom 库」而非整颗
  executor（见 `docs/02` §1 与 `../mega-moe/`）。这与本文看到的 MPK 用法一致——**生产级
  megakernel 的 task 身体普遍是「CUTLASS 原语 + 手工数据流编排」**，而非「直接调用某个 CUTLASS GEMM」。

---

## 9. 局限（诚实标注）

- **纯源码走读，未构建/未运行**：SM100 在目标卡 H100/sm90 不可执行；CUTLASS 侧全部结论来自
  `deps/cutlass@f3fde583` 静态阅读，未做模板实例化打印（如 `print(SmemLayoutA{})`）核对 TMA box
  shape，也未取 device 计数。
- **只追到「atom 与描述符的产生」，未追 TMA descriptor 的位编码**：`make_instr_desc`/
  `make_tma_copy_atom` 内部（`UMMA::InstrDescriptor` 的 32/64 位字段布局、TMA descriptor 的
  swizzle/box 编码）本文未逐位展开；`docs/22` 的 Blackwell transpiler 有另一套等价手写，可互为对照。
- **`B2B` 宏路径仍未细究**：`docs/36` §8 记的 `B2B`（softmax/rescale 关 rescale）与 CUTLASS 侧无关，
  本文也未追。
- **「同一颗 atom」的数值等价未验证**：MPK 自建 mainloop 与 CUTLASS `CollectiveMma::mma` 都调同一
  `cute::gemm(..., SM100_MMA_F16BF16_2x1SM_SS)`，但累加器 staging（`tStS` 双缓冲）与
  `accumulate_` 切换点是否完全等价，本文只做**源码对照**，未做数值比对。
- **相邻尾巴（仍留待后续）**：`docs/35` §9 的其余条目——注意力之外的 SM100 task 家族
  （`dflash_*`/`inkling_*`/`glm_moe_router_sm100`/`prob_scatter_sm100`/`softmax_gather_sm100`/
  `topk_*`/`sampling_sm100` 等）已由 `docs/38` 部分覆盖；`mla_prefill_host.cu` 原型来源仍未走。

---

## 10. 参考坐标（可核对）

- 快照：MPK `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/51` 同）。
  CUTLASS：`deps/cutlass` submodule 钉死 `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8`。
- **MPK 主文件**：`include/mirage/persistent_kernel/tasks/blackwell/mla_sm100_2sm.cuh`（3165 行）：
  `kIs2Sm :610`、`ClusterShape :617-618`、`Schedule :689-692`、`CollectiveBuilder QK :693-711`、
  `TiledMmaQK :707`/`CtaShapeQK :708`、`TransposeTensorStride :711-713`、`CollectiveBuilder PV :720-742`、
  `AtomThrShapeMNK :740-743`、`PipelineLoadQK :749-752`、`SmemLayoutA/KC/VC :778-784`、
  `TmaLoad* :903-907`、`to_underlying_arguments :954-1000`、`is_mma_leader_cta :1102-1104`/
  `:1134-1135`、`make_fragment_A/B :2221-2224`、`cute::gemm :2269/2299/2329/2369`、
  TMEM offset `:2263`/`:2321`。
- **CUTLASS 侧**（`$C = deps/cutlass`）：
  - `$C/include/cutlass/gemm/collective/builders/sm100_umma_builder.inl`：`CollectiveBuilder` 特化
    `:168`、`is_2sm :204`、`TiledMma :218-221`、`AtomThrShapeMNK :227`、`SmemShape_M/N/K :251-254`、
    `DispatchPolicy :291-305`、`CollectiveOp :307-320`。
  - `$C/include/cutlass/gemm/dispatch_policy.hpp`：`KernelSchedule1Sm/2Sm :705-706`、
    `KernelTmaWarpSpecialized{1Sm,2Sm}Sm100 :714-715`、`MainloopSm100TmaUmmaWarpSpecialized :1005-1011`。
  - `$C/include/cutlass/gemm/collective/builders/sm100_common.inl`：`sm100_smem_selector :82-138`、
    `sm100_cluster_shape_to_tma_atom_A :199-226`/`_B :235-252`、
    `sm100_make_1sm_trivial_tiled_mma :306`、`sm100_make_2sm_trivial_tiled_mma :374-455`、
    `sm100_make_trivial_tiled_mma :448`。
  - `$C/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp`：`CollectiveMma :80`、
    `AtomThrShapeMNK :105`、`CtaShape_MNK :110`、`MainloopPipeline :150-153`、`SharedStorage :220-233`、
    `TmaTransactionBytes :235-237`、`prefetch_tma_descriptors :443`、`load_init :496-546`、`load :586-627`、
    `load_tail :629-645`、`mma :649-716`（`cute::gemm :699`）。
  - `$C/include/cute/atom/mma_traits_sm100.hpp`：`MMA_Traits<SM100_MMA_F16BF16_2x1SM_SS> :1677`、
    `FrgTypeC(tmem_frg_2sm) :1689`、`ThrID=Layout<_2> :1695`、`ALayout :1696-1698`、
    `mma_unpack :1709-1735`。
  - `$C/include/cute/arch/mma_sm100_umma.hpp`：`SM100_MMA_F16BF16_SS :86`（`cta_group::1` asm `:100`）、
    `SM100_MMA_F16BF16_2x1SM_SS :422`（`cta_group::2.kind::f16` asm `:446`）。
  - `$C/include/cute/atom/copy_traits_sm100_tma.hpp`：`make_tma_atom_A_sm100 :395-446`、
    `make_tma_atom_B_sm100 :448-497`。
  - `$C/include/cute/arch/copy_sm100_tma.hpp`：`SM100_TMA_2SM_LOAD :182-222`、
    `cta_group::2` 的 `cp.async.bulk.tensor.Nd` asm `:66/92/118/144/170`（1D–5D）、multicast `:243+`。
  - `$C/include/cutlass/pipeline/sm100_pipeline.hpp`：`PipelineTmaUmmaAsync :537`、
    `multicast_consumer_arrival_count :562-564`、`producer_arv_cnt :566`、`init_masks :597-617`。
