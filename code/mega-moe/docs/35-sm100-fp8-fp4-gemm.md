# 35. DeepGEMM 的 SM100 FP8/FP4 稠密 GEMM 主 kernel：`sm100_fp8_fp4_gemm_1d1d`

> 本篇补上本专题一直绕开的一块：**SM100（目标卡）上所有 FP8/FP4 GEMM 的统一主 kernel**。
> 第 3/10 篇反复把它当「黑盒后端」引用（Mega MoE 的 `SM100_MMA_MXF8F6F4_2x1SM_SS`、
> wgrad 的 `k_grouped tn` 就落在这里），第 21 篇只拆了它的 batched 分支、第 31 篇只拆了它的
> epilogue；而它的**主体**（block-scaled UMMA + TMEM + UTCCP + 五类 warp）从没系统讲过。
> 与第 22 篇（SM90 的 1D1D/1D2D 两个主 kernel）正好构成「同一算法在 SM90/SM100 的两种落地」。
>
> **重要**：这是 **SM100 专属 kernel**，本机 H100（sm90）**跑不了**（`sm100_fp8_fp4_gemm_1d1d.cuh:47`
> 整段 `#if __CUDA_ARCH__ >= 1000`；UMMA/UTCCP/TMEM 全是 SM100 硬件）。因此本文主体是
> **源码精读 + 线上资料，明确标注「仅分析」**；只把 kernel 里**架构无关的几何/描述符设备代码**
> 抽到 H100 上真机跑（`harness/exp_sm100_fp8_fp4_geometry.{sh,out.txt}`），给可复现数字。
>
> 目标卡代号敏感：全文用「目标卡」指代。

---

## 1. 一句话与全景

`sm100_fp8_fp4_gemm_1d1d` 是 SM100 架构上**唯一**的 GEMM 后端：稠密（Normal）、m-grouped
（dgrad：contiguous / masked / PSUM）、k-grouped（wgrad）、batched（MLA einsum）**全部**走同一个
kernel 模板，只靠模板参数区分（`GemmType`、major、`num_groups`、`swap_ab`…）。一句话概括它与
SM90 的差别：

- **SM90（第 22 篇）**：WGMMA 把累加器放在**寄存器**，A/B 从 SMEM 经 `wgmma` 读；SF 是
  1D1D（per-token）或 1D2D（per-128×128）两种**独立 kernel**；block-scale 在 epilogue/主循环里
  手动乘。
- **SM100（本篇）**：UMMA（`tcgen05.mma`）把累加器放在 **TMEM**（tensor memory），由**单个 warp**
  发射（`tcgen05.mma...block_scale`）；A/B/SF 都在 SMEM，SF 经 **UTCCP** 从 SMEM **直接搬进 TMEM**，
  由硬件在每个 `gran_k` 上应用；MXF8F6F4（unpacked，`UMMA_K=32`）与 MXF4（packed 4-bit，`UMMA_K=64`）
  两种 MMA 共用一套骨架；layout 用 **2-CTA cluster / multicast** 扩到 `UMMA_M=256`。

对应的 host 入口在 `csrc/apis/gemm.hpp` / `einsum.hpp` 里按 `arch_major==10 and sfa.dtype==torch::kInt`
（packed UE8M0）派发（`gemm.hpp:126-129`、`:243-246`、`:365-368`、`:481-484`、`einsum.hpp:240`），
与 SM90 的 `sfa.dtype==torch::kFloat` 分支互斥。默认 recipe：SM100 packed SF 走 `(1,1,128)`
（`utils/layout.hpp:91-95`，即 `gran_k=128`；FP4 的 `k_grouped_fp4` 硬编码 `gran_k=32`，见 §1.5）。

---

## 2. host 包装：六个入口 + 契约

文件：`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`（563 行）。

### 2.1 统一的 `compile_and_launch`

`SM100FP8FP4Gemm1D1DRuntime::compile_and_launch`（`:35-90`）负责 JIT 微源码 + `launch`：

- `Args`（`:18-33`）装 `GemmDesc`、`GemmConfig`、`deep_jit::cuda::LaunchOptions`、
  `EpilogueClass`、三个 SF/对齐标量（`gran_k_a`/`gran_k_b`/`k_alignment`）与五张 TMA 描述符。
- `:36-39` **MXF4 vs MXF8F6F4 只差 dtype 字符串**：`to_string(a_dtype, not is_mxf4_mma())` ——
  FP4×FP4 pair 把 `a/b_dtype` 打成「packed 4-bit」字符串走 `MXF4`，其余走 unpacked 的
  `MXF8F6F4`。这解释了 kernel 里 `float_e2m1_t` 与 `float_e2m1_unpacksmem_t` 两个类型的区别（§3.2）。
- `:40-79` 就是第 33 篇讲的 JIT 微源码：`#include <deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh>`
  + `__instantiate_kernel` 里显式实例化 `sm100_fp8_fp4_gemm_1d1d_impl<...>`，模板实参按
  `GemmDesc`/`GemmConfig` 拼出（注意 `:66-68` 的 `get_compiled_dim(...,'m'/'n'/'k',compiled_dims)`）。
- `:82-89` `jit->launch(...)`：参数顺序 `grouped_layout, m, n, k, epilogue_operator_args,
  tensor_map_a/b/sfa/sfb/cd`。

### 2.2 六个 shape 入口

| 入口 | 行号 | `GemmType` | 角色 |
|---|---|---|---|
| `sm100_fp8_fp4_gemm_1d1d` | `:101-167` | `Normal` | 普通稠密 |
| `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d` | `:169-252` | `MGroupedContiguous` / `...WithPsumLayout` | dgrad（不等长在 M） |
| `sm100_m_grouped_fp8_fp4_gemm_masked_1d1d` | `:254-323` | `MGroupedMasked` | dgrad（`[G,max_m,·]` 稠密 + device `masked_m`） |
| `sm100_k_grouped_fp8_gemm_1d1d` | `:325-403` | `KGroupedContiguous` / `...WithPsumLayout` | wgrad（FP8，major 恒 **MN**） |
| `sm100_k_grouped_fp4_gemm_1d1d` | `:483-561` | `KGroupedContiguous` / `...WithPsumLayout` | wgrad（FP4，`gran_k=32`、major 恒 **K**） |
| `sm100_fp8_bmm` | `:405-481` | `Batched` | MLA einsum 按头 batched |

几个关键契约（host 侧就断言，早于编译）：

- **wgrad FP8 的 major 被钉死为 MN**：`k_grouped_fp8` 第 `:337` 行
  `DG_HOST_ASSERT(major_a == MN and major_b == MN)` —— 因为 wgrad 的 A/B 是 `[sum_k, M]`/`[sum_k, N]`
  **MN-major**（第 12 篇）。而 **wgrad FP4 被钉死为 K-major**：`k_grouped_fp4` `:497`
  `DG_HOST_ASSERT(major_a == K and major_b == K)`，配 `gran_k=32`（`:496`）—— 与 kernel 里
  MXF4「only supports K-major operands and 32-element SF granularity」的静态断言（kernel `:88-89`）呼应。
- **`k_alignment % block_k == 0`**：`:357`/`:515` 断言；对应 kernel `:103-104` 的静态断言
  「K alignment must be divisible by block K」。这是第 26 篇 K-grouped SF 打包的 host 契约落点。
- **psum layout**：m-grouped contiguous 里 `use_psum_layout` 决定 `GemmType`（`:181-182`），
  且 `expected_m_for_psum_layout` 只在 psum 下允许（`:185-186`）；`expected_m/n/k/num_groups`
  用于启发式估算（`:202-204`、`:512`）。
- **m-grouped masked 把 `masked_m.data_ptr()` 当 `grouped_layout`**（`:316`）；contiguous 传的
  `grouped_layout` 是每 M block 首行的组号（第 13 篇）。
- **wgrad 的 CD 是 3D**：`make_tma_3d_desc(d, n, m, num_groups, ..., d.stride(1), d.stride(0))`
  （`:370-375`），每组的 `[n, m]` 输出。
- **k-grouped 的 SF 描述符**用 `sfa.size(0)*gran_k*4` 当逻辑 K（`:377-380`、`:535-538`），
  把「按组独立对齐补零后的 packing」还原成 TMA 可寻址形状（第 16/26 篇）。
- **bmm** 用 `make_tma_3d_desc` 给 A/B/D 都加 batch 维（`:431-451`），SF 用 batch 当 group（`:454-457`）。

### 2.3 `get_sf_block_config`

`:93-99`：`get_sf_uttcp_aligned_block_sizes(block_m, block_n, mma_kind)`（
`heuristics/sm100.hpp`）把 block_m/block_n **上取整到 128 的倍数**，再返回 `SF_BLOCK_K = block_k/128`。
这三个量决定了 SMEM 里 SF 块的 TMA box 与 TMEM 里 SF 的列数（§3.4）。**BF16 的 MmaKind 返回 `{0,0}`**
（无 SF 通路），这也是同一套 host 能服务 `sm100_bf16_gemm` 的原因。

---

## 3. kernel 模板：类型 / 尺寸约束

文件：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`（579 行）。

### 3.1 模板参数与 arch 守卫

`:24-37` 是 24 个模板参数：`kMajorA/B`、`kGranKA/B`、`kKAlignment`、`SHAPE_M/N/K`（可 0 =
运行时）、`BLOCK_M/N/K`、`kNumGroups`、三档 swizzle、`kNumStages`、`kNumTMAStoreStages`、
`kNumNonEpilogueThreads`/`kNumEpilogueThreads`、`kNumMulticast`、`kIsMulticastOnA`、`kNumSMs`、
`kSwapAB`、`kEnsureZeroPadding`、`GemmType`、`kWithAccumulation`、三个 dtype、`epilogue_operator_t`。

`:47` 是硬守卫：`#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)`，
`#else` 分支（`:571-574`）只留一句 `DG_DEVICE_ASSERT(false and "This kernel only support sm_100f")`。
**这就是本机 H100 不可跑的根因。**

### 3.2 MXF4 vs MXF8F6F4

`:51-54` 的注释一句话点破：

```cpp
// an FP4xFP4 pair uses the dedicated MXF4 MMA (packed 4-bit operands, K-major only, UMMA_K=64);
// all other combinations use the MXF8F6F4 MMA (unpacked operands, UMMA_K=32)
constexpr bool kIsMXF4 = is_same_v<a_dtype_t, cutlass::float_e2m1_t> and
                         is_same_v<b_dtype_t, cutlass::float_e2m1_t>;
```

- `UMMA_K = kIsMXF4 ? 64 : 32`（`:72`）。
- `kPackFactor = get_smem_pack_factor<a_dtype_t>()`（`:75`；`packing.cuh:7-11`：FP4 packed → 2，
  否则 1），且要求 A/B 同 pack factor（`:76`）。
- `kIsFP4A/kIsFP4B`（`:77-80`）同时认 `float_e2m1_t`（packed）与 `float_e2m1_unpacksmem_t`（unpacked）
  —— 后者是 host 把 `is_mxf4_mma()` 之外的情况转成整数类型字符串的结果（§2.1）。
- ARG check：`:81` 强制 `(MXF4 且 BLOCK_K==256) 或 (非 MXF4 且 BLOCK_K==128)`；`:82` `BLOCK_K % UMMA_K == 0`；
  `:88-89` MXF4 只允许 K-major 且 `gran_k=32`。

### 3.3 C/D 与输出 SF

`:60-66`：`kWithOutputSF = is_same_v<epilogue_operator_t, QuantizeToFP8>`；只有当输出 SF 时
`cd_dtype` 必须是 E4M3 且 `GemmType==Batched and not kWithAccumulation`；否则 `cd_dtype ∈ {float, bf16}`。
这把第 31 篇的 `QuantizeToFP8` 算子和「FP8 输出对仅 batched」硬约束对上了。

### 3.4 尺寸与 TMEM 列预算（为什么是 512）

`:68-85`：

- `LAYOUT_AD_M = 128`、`UMMA_M = LAYOUT_AD_M * kNumMulticast`（2-CTA → 256）、
  `UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N`。**swap-AB** 是 m-grouped 的默认（`heuristics/sm100.hpp`
  对三种 m-grouped `swap_ab=true`），此时「M（token）在 N 轴、被动态缩」。
- `LOAD_BLOCK_M/N`：按 multicast 在 A 还是 B 上把 block 切成两半。
- block size 断言 `:84-85`：swap-AB 要求 `BLOCK_N==128`；非 swap 要求 `BLOCK_M ∈ {32,64,128}`。

TMEM 预算 `:120-132` 是这一篇最值得讲的「为什么」：

```cpp
constexpr uint32_t kNumAccumTmemCols = UMMA_N * kNumEpilogueStages;   // 2 级累加
constexpr uint32_t kNumSFATmemCols   = SF_BLOCK_M * SF_BLOCK_K / 32;  // 每 32 个 K 一列
constexpr uint32_t kNumSFBTmemCols   = SF_BLOCK_N * SF_BLOCK_K / 32;
constexpr uint32_t kNumSFTmemCols    = kNumSFATmemCols + kNumSFBTmemCols;
constexpr uint32_t kNumOverlappedTmemCols = max(kNumAccumTmemCols + kNumSFTmemCols, 512) - 512;
constexpr uint32_t kNumTmemCols = get_num_aligned_tmem_cols<accum + sf - overlap>();
```

SM100 每个 SM 的 TMEM 只有 **512 列**。累加用**双缓冲**（`kNumEpilogueStages=2`），所以
`accum = 2*UMMA_N`；再加 SFA/SFB。当 `accum+SF > 512` 时，`overlap = accum+SF-512`，
最终 `kNumTmemCols = align(accum+SF-overlap) = 512` ——**正好卡满**：多出来的 SF 与累加器的
尾部「借用」同一批列（`:399-408` 的 `tmem_overlap_barriers` 在 SF UTCCP 之后、首条 UMMA 之前
才等待，保证重叠只在累加器已经被上一个 epilogue 读完的窗口里发生）。这也解释了为什么
`heuristics/sm100.hpp` 的合法性检查只写 `umma_n + tmem_sf_cols > 512`（用的是**单级**累加
`umma_n`，比 kernel 的 2 级松）——重叠机制兜住了双缓冲的超额。

### 3.5 epilogue 配置

`:106-118`：`kNumEpilogueStages=2`、`kNumTMAStoreStages ∈ {1,2}`；`STORE_BLOCK_M/N` 在 swap-AB 下
是 `{16, BLOCK_N}`、非 swap 下是 `{min(BLOCK_M,128), swizzle_cd/sizeof(cd)}`，且
`kNumUMMAStoreThreads` 相应取值。`UMMA_A_SIZE_PER_STAGE`（`:118`）把 LOAD_BLOCK_M 上取整到 128
防止 UMMA padding 越界。

---

## 4. 五类 warp：与 Mega MoE 同构的骨架

`warp_idx` 分工（`:137-562`），与第 3/9 篇的 Mega MoE 五类 warp 一一对应：

1. **初始化**（`:167-192`）：warp1 初始化 `sf_full`/`empty`/`full`/`tmem_full`/`tmem_empty`/
   `tmem_overlap` barriers（`:175` 的 `full.init(kNumMulticast*(1+32*SF_BLOCK_K))` 把 SF warp 转置
   的到达也算进去）；warp2 `Allocator().allocate(kNumTmemCols, &smem.tmem_ptr)`（`:190`）；
   2-CTA 用 `cluster_sync_with_relaxed_arrive`（`:192`）。随后 `cudaGridDependencySynchronize`
   （`:195`，PDL）。
2. **block 调度器**（`:197-210`）：`sched::Scheduler<...>`，`advance_pipeline` 翻 stage/phase
   （与第 9 篇同一个模式）。
3. **TMA load warp**（`:213-300`）：`warp_idx==0 and elect_one`。`while(get_next_block(...))` →
   `empty_barriers.wait` → 用 `get_global_idx<kWithGroupOffset, IndexType::MN/K/SF_K>` 算 m/n/k
   坐标（grouped 偏移由调度器给，`:231-249`）→ **先发 SFA/SFB TMA**（`:260-277`，注释明说
   「让转置和 A/B 搬运算重叠」）→ 再发 A/B TMA（按 major 选 2D/3D、带 multicast 与 batch_idx，
   `:280-291`）→ leader CTA `arrive_and_expect_tx`、非 leader `arrive(0u)`（`:294-298`）。
4. **MMA issue warp**（`:301-459`）：`warp_idx==1 and is_leader_cta`，**只有 leader CTA 的一个
   warp 发 UMMA**。构造 `instr_desc = make_instr_desc_block_scaled<...>`（`:305-308`）与
   `sf_desc = make_sf_desc(nullptr)`（`:309`）；把每 stage 的 `a_desc.lo/b_desc.lo` 预存进 lane
   （`:314-315`）；`tmem_empty_barriers.wait`（`:331`）后进入 K 循环：
   - `full_barriers.wait(phase)`（`:366`）；
   - `elect_one` 里按 `kNumSFAStagesPerLoad/kNumSFBStagesPerLoad` 发 **UTCCP**：把 SMEM 里
     `SF_BLOCK_K*SF_BLOCK_M/128` 个 128 元素块用 `SM100_UTCCP_4x32dp128bit_{1,2}cta::copy(sf_desc, col)`
     搬进 `kTmemStartColOfSFA + i*4`（`:371-393`）；
   - 重叠屏障 `tmem_overlap_barriers` 在 SF UTCCP 之后、首条 UMMA 之前等待（`:399-408`）；
   - `elect_one` 里按 `BLOCK_K/UMMA_K` 发 UMMA：每条算 `subblock_idx`、`sf_id_in_subblock`、
     `tmem_col_sfa/sfb`、`sfa_id/sfb_id`（`:416-431`，§5），用
     `SM100_MMA_MXF8F6F4_2x1SM_SS::fma(a_desc,b_desc,tmem_c,scale_c,runtime_instr_desc,tmem_col_sfa,tmem_col_sfb)`
     （`ptx/tcgen05.cuh:67-86`）发射，`umma_k_idx>0 or k_block_idx>0` 作为 accumulate 位（首条清零）；
   - `empty_barrier_arrive(k_block_idx==last)`：`umma_arrive` 到 `empty_barriers`，末 K block 时
     同时 `umma_arrive` 到 `tmem_full_barriers`（`:335-350`）。
5. **UTCCP transposer warp**（`:460-496`）：`warp_idx==2`（或 `SF_BLOCK_K==2` 时的 warp3），
   `while(get_next_block)` → `sf_full_barriers.wait` → 对每个 128 元素 SF 块做
   `utccp_required_smem_warp_transpose`（§5.2）→ `fence_view_async_shared` → `full_barriers.arrive(0u)`。
6. **epilogue warp groups**（`:497-561`）：`tmem_full_barriers.wait` → 按 `kSwapAB` 调
   `epilogue::sm100_store_cd_swap_ab` / `sm100_store_cd`（第 31 篇），并传 `tmem_overlap_barriers`/
   `tmem_empty_barriers` 做双缓冲交接。

最后 `:564-569` 再 `cluster_sync`/`__syncthreads` 后 `Allocator().free(0, kNumTmemCols)`。

---

## 5. SF 通路：TMA → warp 转置 → UTCCP → TMEM

这是 SM100 与 SM90 最本质的差异：**SF 不再在寄存器里乘，而是搬进 TMEM 交给 `block_scale` UMMA 硬件**。

### 5.1 gran32 vs gran128 的 stage 数与 id

`:96-97`：

```cpp
constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
```

- **gran=32**：一个 `BLOCK_K=128` 里有 4 个 32 元素的 SF，各自独立，一次 TMA 载入就够
  （`kNumSFAStagesPerLoad=1`）。
- **gran=128**：SF 粒度是 128，一个 128 的 UTCCP 子块对应 4 个「stage」（`=4`），需要跨 4 个
  `BLOCK_K` 才凑齐一整段 SF —— 所以只在 `k_block_idx % 4 == 0` 时载入/转置/UTCCP（kernel
  `:263-276`、`:376-391`、`:481-490`），并通过 `sfa_stage_in_group_idx = k_block_idx % 4`
  选当前的 `sfa_id`（`:369-370`、`:427-428`）。

每发一条 UMMA 时（`:416-431`）：`subblock_idx = offset/128`、
`sf_id_in_subblock = (offset%128)/32`、`tmem_col_sfa = kTmemStartColOfSFA + subblock_idx*SF_BLOCK_M/32`；
`gran32` 用 `sf_id_in_subblock`（0..3），`gran128` 用 `sfa_stage_in_group_idx`（0..3）。
最后 `make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id)` 把这两个 2-bit id
写进指令描述符（swap-AB 下 A/B 互换，`:429-431`）。

### 5.2 为什么 TMA 落下的 SF 要先「warp 内转置」

TMA 按 `[SF_BLOCK_M, SF_BLOCK_K]`（MN-major 的 SF 布局）把一个块摊进 SMEM；而 UTCCP
`4x32dp128bit` 期望的 SMEM 布局是「每个 lane 连续持有 4 个 32-bit SF」。kernel 用一段
**共享内存内原地转置**桥接：

```cpp
// sm100_fp8_fp4_gemm_1d1d.cuh:464-472
auto utccp_required_smem_warp_transpose = [&](const uint32_t* smem_ptr) {
    DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128, "Invalid aligned elements");
    uint32_t values[4];
    #pragma unroll
    for (uint32_t i = 0; i < 4; ++ i)
        values[i] = ptx::ld_shared(smem_ptr + i * 32 + lane_idx);
    __syncwarp();
    ptx::st_shared(smem_ptr + lane_idx * 4, values[0], values[1], values[2], values[3]);
};
```

即把 `in[i*32 + lane]`（4 行 × 32 列）转成 `out[lane*4 + i]`（32 行 × 4 列，每 lane 连续 4 个）。
**这段是纯 `ld_shared`/`st_shared`，与 UMMA 硬件无关，因此可在 H100 真机运行**（§7 A）。

---

## 6. 与 SM90 / Mega MoE 对照

### 6.1 与 SM90 FP8 主 kernel（第 22 篇）

| 维度 | SM90 1D1D / 1D2D | SM100 1D1D（本篇） |
|---|---|---|
| MMA | WGMMA（`wgmma.mma_async`），累加器在**寄存器** | UMMA（`tcgen05.mma.block_scale`），累加器在 **TMEM** |
| 发射者 | 每 warpgroup 各自发 | **仅 leader CTA 单个 warp** 发（`warp_idx==1 and is_leader_cta`） |
| `UMMA/WGMMA_K` | 32（FP8） | 32（MXF8F6F4）/ 64（MXF4） |
| M 方向 | `BLOCK_M` | `UMMA_M = 128 * kNumMulticast`（2-CTA → 256） |
| SF 应用 | 主循环/epilogue 手动乘，SF 1D1D/1D2D 两个 kernel | **UTCCP 搬进 TMEM**，硬件 per-`gran_k` 应用；一个 kernel 通吃 gran 32/128 |
| SF 通路 | host `transform_sf_into_required_layout` → SMEM | 同上 + **SMEM 内 warp 转置** + UTCCP |
| K-grouped | 原地改 tensormap（`sm90_fp8_gemm_1d1d.cuh:193-215`） | MN-major 直接按组偏移 TMA（`get_global_idx<...,K/SF_K>`） |
| epilogue | STSM + TMA store / TMA reduce-add | TMEM 读回 → `sm100_store_cd(_swap_ab)`（第 31 篇） |

一句话：SM90 的两种 SF 粒度分裂成两个 kernel，SM100 合并为一个，代价是把 SF 的「形状适配」
从 host/寄存器挪到了 **SMEM 转置 + UTCCP + TMEM**。

### 6.2 与 Mega MoE（第 3/10 篇）

Mega MoE 的线性层用的就是同一个 `SM100_MMA_MXF8F6F4_2x1SM_SS`（本篇 §4.4），但多两处差异：
Mega MoE 有 **shared expert 的独立 instruction descriptor**（一个 kernel 内两套 block-scaled
描述符）与**按有效 token 数动态缩 `UMMA_N`**（`update_instr_desc_with_umma_n`）。本篇 kernel 里
`swap-AB` 分支也用了同一个 `update_instr_desc_with_umma_n`（`:352-356`），把 `umma_n` 设为
`get_aligned_effective_m_in_block(m_block_idx)`（`:354`）。可以说 **Mega MoE = 本篇 kernel 的
dispatch/combine + ring buffer 外衣**。

---

## 7. H100 实测：架构无关的几何 / 描述符设备代码

因为 kernel 主体是 SM100 专属，我们把 kernel 里**不依赖 UMMA 硬件**的几块抽到 H100 跑真机
（`harness/exp_sm100_fp8_fp4_geometry.sh`，`nvcc -arch=sm_90 --expt-relaxed-constexpr`）：

- **A) SF warp 内原地转置**（`cuh:465-471` 逐行抽取，sha256 `ae0401ba…`）：在真实设备代码上跑
  128 元素块，`out[lane*4+i] == in[i*32+lane]` **128/128 全对**。
- **B/C) smem 描述符**（真实头文件 `deep_gemm/mma/sm100.cuh`，非手抄）：对
  `make_umma_desc` 的位布局与 `advance_umma_desc_lo` 的字节偏移，用独立 host 模型逐项对拍，4 例
  全对：

| 例 | layout_type | SBO | LBO | advΔ(0,1) | advΔ(32,0) |
|---|---|---|---|---|---|
| K-major BM128 BK128 SW128 FP8 | 2 | 64 | 0 | 0 | 2 |
| MN-major BM128 BK128 SW128 FP8 | 2 | 64 | 1024 | 8 | 2 |
| K-major BM128 BK128 SW64 FP4 | 4 | 32 | 0 | 0 | 1 |
| MN-major BM256 BK128 SW64 FP8 | 4 | 32 | 512 | 4 | 2 |

  这坐实了几条「为什么」：**K-major 的 stride_k = 1**（沿 K 逐个元素推进），**MN-major 的
  stride_k = swizzle atom 大小**（SW128→128、SW64→64）；**FP4 的 pack factor=2 让所有字节偏移
  减半**（K-major FP4 的 advΔ(32,0)=1 而非 FP8 的 2）；SBO/LBO 存进描述符时都 `>>4`。
- **D) 指令描述符**（真实 cutlass `make_instr_desc_block_scaled` + DeepGEMM
  `make_runtime_instr_desc_with_sf_id` / `update_instr_desc_with_umma_n`）：5 例全对，逐位核对了
  `scale_format_=1`（E8M0）、`n_dim_=N>>3`、`m_dim_=M>>4`、`a_sf_id_`/`b_sf_id_`（2-bit，写入
  `[29,31)`/`[4,6)`）、`a_format_`/`b_format_`（E4M3=0；packed `float_e2m1_t` 走
  `MXF4Format::E2M1=1`，**不是** `MXF8F6F4Format` 的 5）、低 32 位恒 0、以及
  `update_instr_desc_with_umma_n(...,96)` 把 `n_dim_` 置成 `96>>3=12`。

**总判定 ALL PASS**（原始输出见 `harness/exp_sm100_fp8_fp4_geometry.out.txt`）。

---

## 8. 结论与边界

- `sm100_fp8_fp4_gemm_1d1d` 是目标卡上**唯一**的 FP8/FP4 GEMM 后端：稠密/m-grouped/k-grouped/
  batched 共用一套模板，只靠模板参数区分；MXF8F6F4（K=32）与 MXF4（K=64，packed、K-major、
  gran32）两种 MMA 共用骨架。
- 与 SM90 的本质差异是 **UMMA + TMEM + UTCCP**：累加器与 SF 都进 TMEM，由单个 warp 发
  `tcgen05.mma.block_scale`，硬件按 `gran_k` 应用 SF；SM90 的两种 SF 粒度被合并成一个 kernel。
- **512 列 TMEM 的预算与「overlap」技巧**是 host `umma_n + sf <= 512` 与 device 双缓冲之间的
  缝补点，是本 kernel 最值得记住的「为什么」。
- **边界**：kernel 主体为 **SM100 专属，仅分析**；H100 上只验证了架构无关的 SF 转置、smem/UMMA
  描述符与指令描述符位布局（ALL PASS）。真正的 UMMA 数值与性能需目标卡。

---

## 9. 引用

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
  - kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`
  - host：`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp`、`csrc/apis/gemm.hpp`、`csrc/apis/einsum.hpp`
  - 描述符：`deep_gemm/include/deep_gemm/mma/sm100.cuh`、`cute/arch/mma_sm100_desc.hpp`
  - 启发式：`csrc/jit_kernels/heuristics/sm100.hpp`、`csrc/utils/layout.hpp`
  - 调度器：`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`
- 版本：DeepGEMM `057ca59`（Public release 26/09/30）。
- 相关 PR：Mega MoE（#316）、MoE backward（#95）——见第 1/2 篇。
- 本专题对照篇：第 3/10 篇（Mega MoE 主体与 MMA/epilogue）、第 19 篇（SM90 BF16）、
  第 21 篇（FP8 einsum batched）、第 22 篇（SM90 FP8 1D1D/1D2D）、第 31 篇（SM100 epilogue）、
  第 33 篇（DeepJIT）。
