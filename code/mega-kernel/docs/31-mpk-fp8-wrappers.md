# MPK FP8 单测 wrapper 与 TP standalone 走读（`docs/30` §9 尾巴，P117-1）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit **`f9eb70c`**（与 `docs/06`–`docs/30` 同一快照，行号已核对）。
> 本文**纯源码走读（资料分析），本轮无新实验**；目标卡 H100/sm90 不能执行 SM100 路径，所有 SM100 形态均为**静态走读**。
>
> 定位：补掉 `docs/30-mpk-unit-wrappers.md` **§9「局限」明留的两条尾巴**——
> （原话）「**未逐行走读 `sm100_linear_fp8`（1107 行）/`sm100_fp8_moe`（810 行）等大 wrapper 的 fp8
> scale 重排与 UMMA 内部**，只核了其注册符号与构建宏；`test_tp*_wrapper.cu` 的内部（`shim` 之后的
> TP 切分）**未展开**」。
>
> 与既有篇的关系：`docs/29`/`docs/30` 是「一个 task = 一颗 kernel」的单测层对照；本文把其中**两个最重的
> 目录**（fp8 家族 + TP standalone）打开，重点是 **fp8 scale 的 host 布局层 / device UTCCP+UMMA 路径**与
> **TP 的机械 port 形态**——它同时是「同一套硬件惯用法在仓库里实现了 N 遍」母题（`docs/19` §5、
> `docs/30` §5）的又一肉身。

---

## 0. 一句话：本文把两个「重目录」打开

- **fp8 家族**（`sm100_linear_fp8` / `sm100_fp8_moe` / `sm100_quantize_fp8`）：host 侧有一个被 `docs/30`
  漏掉的**共享层** `tests/runtime_python/blackwell/common/`——**UE8M0 packed scale 的布局探测/重排**
  （`sm100_fp8_scale_layout.{h,py}` + `sm100_fp8_runtime_registry.h`），device 侧落到
  `tasks/blackwell/linear_fp8_sm100*.cuh`（11 个头）与 `fp8_group_gemm_sm100.cuh`（1567 行）。
- **TP standalone**（`sm100_mla_mtp_decode/`）：`test_tp{2,4,8}_wrapper.cu` 是**含 `int main`、无
  pybind** 的独立程序（`docs/30` 已按「standalone」归类），本文补其内部——它们是**对外部 standalone
  内核（`v037_tp*`）的机械 port 验证器**，把 MLA MTP decode 按 **head 维切成 TP=2/4/8**。

两条尾巴各给一个 headline：

1. **fp8 家族 host 侧有一层真实的「scale 布局契约」**：同时接受 **row-major** 与 **DeepGEMM column-major**
   两种 packed UE8M0 布局，`detect_scale_layout` 在 C++ 与 Python 里**逐行同构**（§2）。
2. **`sm100_linear_fp8` 在 `f9eb70c` 里引用了不存在的头文件与未定义符号**（§6）——`docs/30` 报的是
   `sm100_mla_prefill_tp8` 的「缺源文件」，本文报一处**更靠前、连非 split-K 路径也编译不过**的「缺头文件」。

---

## 1. fp8 家族全景：三道扩展 + 一层共享 common

`docs/30` §1 统计 blackwell 14 目录 / 18 `.cu` / 13 `setup*.py`。fp8 相关的是三处**大 wrapper**，
且都**不在**「同名扩展撞名」之列（扩展名互异）：

| 目录 | 扩展名 | 源文件 | 行数 | gencode |
|---|---|---|---|---|
| `sm100_linear_fp8/` | `runtime_kernel_blackwell_linear_fp8` | `runtime_kernel_wrapper_sm100.cu` | 1107 | `compute_100a,sm_100a` |
| `sm100_fp8_moe/` | `runtime_kernel_fp8_moe`（+ `runtime_kernel_bf16_moe` 对照） | `runtime_kernel_wrapper.cu`（+ `_bf16.cu`） | 810（+363） | `compute_100a,sm_100a` |
| `sm100_quantize_fp8/` | `runtime_kernel_blackwell_quantize_fp8` | `runtime_kernel_wrapper_sm100.cu` | — | `compute_100a,sm_100a` |

- 三个 `setup.py` 的 nvcc 参数一致：`-O3 -gencode=arch=compute_100a,code=sm_100a -DMIRAGE_GRACE_BLACKWELL
  -DMPK_ENABLE_TMA`（`sm100_linear_fp8/setup.py:88-93`、`sm100_fp8_moe/setup.py:35-37`、
  `sm100_quantize_fp8/setup.py:90`）。**fp8 家族带 `MPK_ENABLE_TMA`**（`docs/30` §1 已记「多数带」）。
- **`sm100_fp8_moe/setup.py` 一个 `setup()` 装两个扩展**（`runtime_kernel_fp8_moe` `:45` +
  `runtime_kernel_bf16_moe` `:67`，`name='runtime_kernels_fp8_moe'` `:84`）——同一目录内 fp8 与 bf16
  对照两份 wrapper，这是 `docs/30` §5「同一套惯用法实现两遍」在**同一目录内**的形态。
- `sm100_linear_fp8/` 与 `sm100_quantize_fp8/` 的 wrapper **第 18 行左右都 `#include
  "../common/sm100_fp8_scale_layout.h"`**——`common/` 这层是 `docs/30` 逐目录核注册表时**没有覆盖**的。

`common/` 三个文件：`sm100_fp8_runtime_registry.h`（192 行）、`sm100_fp8_scale_layout.h`（57 行）、
`sm100_fp8_scale_layout.py`（152 行，Python 参考实现）。

---

## 2. headline：UE8M0 packed scale 的「布局契约」（host）

### 2.1 常量与形状白名单（`sm100_fp8_runtime_registry.h`）

`namespace mirage::blackwell::sm100_fp8_runtime` 里把 FP8 的 `group_size` / `batch` / `N` / `K` 白名单
写死：

- `kBlockK = 128`（`:11`）、`kScalePackSize = 4`（`:12`）——scale block = 128K，**4 个 UE8M0 打包进一个
  uint32**；
- `kSupportedGroupSizes = {128}`（`:13`）、`kSupportedBatchSizes = {1,2,4,8,16}`（`:14`）、
  `kSupportedOutputSizes = {128}`（`:15`）、`kSupportedReductionSizes = {128,256,384,512,768,1024,1536,
  2048,4096,7168}`（`:16-17`）。

三个关键推导（`:41-64`）：`round_up_to_multiple`、`ceil_div`、`logical_scale_k_for_reduction_size(K) =
ceil(K/128)`、`packed_scale_k_for_reduction_size(K) = ceil(logical/4)`、
`aligned_scale_outer_dim(outer) = round_up_to_multiple(outer, 16/sizeof(uint32_t)) = ceil16(outer)`。

### 2.2 两种合法布局 + 探测（`:66-110`）

`enum class PackedScaleLayout { Invalid, RowMajor, DeepGemmColumnMajor }`（`:66-70`），两种布局的判据是
**shape 都是** `[outer_dim, packed_scale_k]`、只差 **stride**：

- **RowMajor**：`stride == [packed_scale_k, 1]`（`:72-82`）；
- **DeepGEMM column-major**：`stride == [1, aligned_scale_outer_dim(outer_dim)]`（`:84-93`）——
  即 **DeepGEMM 的列主序 packed scale**，outer 维按 16 对齐后作为列 stride。

`detect_scale_layout(...)`（`:95-110`）先试 row-major、再试 DeepGemm、否则 `Invalid`。C++ 侧的
`sm100_fp8_scale_layout.h:10-21` 只是把 `torch::Tensor` 的 `size/stride` 转发进去；
`check_scale_tensor`（`:23-55`）要求 `dim==2`、dtype 是 `kInt/kUInt32`、布局非 `Invalid`，并给出
一长串期望 shape/stride 的报错文本。

### 2.3 Python 参考实现**逐行同构**（`sm100_fp8_scale_layout.py`）

Python 侧把同一套公式重写了一遍（`BLOCK_K=128` `:4`、`SCALE_PACK_SIZE=4` `:5`、
`packed_scale_k_for_reduction_size` `:21-22`、`aligned_scale_outer_dim(outer)` 用 element size 4
`:29-30`），并**镜像**了两种布局：

- `allocate_packed_ue8m0_scale`（row-major，`:33-38`）/ `allocate_packed_ue8m0_scale_deepgemm_style`
  （`empty_strided((outer,pk),(1,aligned_outer))`，`:41-49`）；
- `detect_packed_scale_layout`（`:52-62`）与 C++ `detect_scale_layout` 判据**逐条相同**；
- `encode_ue8m0 / decode_ue8m0`（`:65-72`）：`ue8m0 = clamp(ceil(log2(clamp(scale,min=1e-30)))+127,
  0,255)`；
- `quantize_to_fp8_packed_ue8m0`（`:75-119`）：按 128K 块取 `amax/448`、编码 UE8M0、**4 个 scale 按
  `(scale_k%4)*8` 位移打包**进 `packed_scales[outer, scale_k//4]`（`:109-113`）；
- `deepgemm_col_major` 版本（`:122-123`）与反量化（`:126-152`）对称。

> **要点**：`docs/27`（权重预处理）讲的是 **HF checkpoint → builder** 的 FP8 重定标；本文这层是
> **单测输入张量 → kernel** 的 scale 布局契约。两者都锚在「**128K 块 + UE8M0 + DeepGEMM 列主序**」
> 这同一个约定上——`aligned_scale_outer_dim` 的 16 对齐正是 TMA 对 scale 张量的要求。

### 2.4 host 侧重排：`materialize_internal_scale_tensor`（wrapper `:174-200`）

`sm100_linear_fp8` wrapper 对**每个** scale 张量都过一遍：

- `detect_scale_tensor_layout` 非 `Invalid` 才继续（`:182-183`）；
- 若已是 **DeepGemmColumnMajor** → **原样返回**（`:184-185`）；
- 否则（row-major）→ 分配/复用一块 `allocate_legacy_col_major_scale_buffer`（`empty_strided` 成
  **列主序** `[outer, packed_k]` stride `[1, aligned_outer]`，`:164-172`）并 `copy_`（`:186-200`）。

即 **host 把非 DeepGEMM 布局统一「物化」成 DeepGEMM column-major 供 kernel 读**，缓冲带在
`LinearFp8DescriptorCache` 里（含 `internal_input_scale_buffer` / `internal_weight_scale_buffer`、
`last_*_ptr`，`:72-84`）。scale 的 TMA descriptor 由 `make_tma_sf_desc`（`:318-328`）按
`[outer, packed_k]` 建。

---

## 3. `sm100_linear_fp8` wrapper 的 host 侧（1107 行）

### 3.1 描述符缓存 + 地址热替换

`LinearFp8RuntimeKey{batch_size, output_size, reduction_size, split_k, has_residual}`（`:46-62`）做 key，
`static std::map<...>` + `std::mutex` 缓存一组 `LinearFp8DescriptorCache`（`:96-103`）。第一次建 TMA
descriptor；之后每次调用**只比较指针、变了就 `cuTensorMapReplaceAddress`**（`:466-560` 的
`update_desc`/`last_*_ptr` 逻辑）——这是单测 wrapper 为了不被 TMA descriptor 构建开销主导而做的
**host 侧复用**（`docs/29` §3 的「host 按形状硬编码派发」在这里更重）。

### 3.2 编译期 smem 预算（`linear_fp8_dynamic_smem_bytes`，`:330-375`）

一个 `constexpr` 函数把 SMEM 预算算成**编译期常量**：CD 输出 stage（`kNumTmaStoreStages = kWithResidual
? 1 : 2`）、residual、A/B（`kNumStages * {BLOCK_M,BLOCK_N} * BLOCK_K * sizeof(e4m3)`）、
**SFA/SFB**（`kNumStages * kSfBlock{M,N} * sizeof(uint32_t)` `:363-364`，block 对齐到 128 `:348-353`）、
barrier、TMEM 指针。`launch_linear_fp8_1d1d_sm100` 里用 `static_assert` **钉死**两个数：
`203468`（无 residual）/`203484`（有 residual）字节（`:449-464`），并断言
`kDynamicSmemBytes + kSchedulerSmemReservationBytes(6KB) <= 228KB`（`:465-467`）——**给 scheduler 预留
SMEM 头寸**，与 `docs/07`/`docs/06` 的「worker = CTA + 全 SMEM」一致。

主要编译期参数（`:426-441`）：`BLOCK_M=32, BLOCK_N=16, BLOCK_K=128, kNumStages=28`、非 epilogue/epilogue
线程各 128、`kNumSMs=8`。**一个 task = 一个 CTA**（`docs/29` §3）：`grid = kNumOutputTiles(*SPLIT_K)`，
`block = 256`，`cluster = (1,1,1)`（`:645-658`），`launch_kernel_ex_cluster`（`:377-411`）在
cluster=(1,1,1) 时**退化回 `cudaLaunchKernel`**、否则走 `cudaLaunchKernelExC` + cluster 属性。

### 3.3 split-K 启发式与 dispatch

`choose_split_k(batch, K, has_residual)`（`:148-162`）：有 residual → 1；否则读环境变量
`MIRAGE_FORCE_SM100_FP8_SPLIT_K`（`{1,2,4,8}`，`:129-146`）覆盖；否则 `K<7168 → 1`，
`batch<=4 → 4`，否则 `8`。

`linear_fp8_1d2d_sm100_kernel`（host 入口，`:874-1094`）做完整 shape/dtype 校验（`:880-947`），然后
用宏展开三层 dispatch：`with_residual` → `split_k==1` → `else {2,4,8}`，再 `batch ∈ {1,2,4,8,16}` ×
`N=128` × `K ∈ 10 档`（`:1000-1087`）。PYBIND 只暴露两个符号（`:1100-1106`）：
`linear_fp8_1d2d_sm100` 与 `supported_dense_gemm_shapes`。

---

## 4. `linear_fp8_sm100` device 家族（11 个头）

`sm100_linear_fp8` wrapper 的 device 端在 `tasks/blackwell/` 有**一整个家族**（`docs/30` §1 只列了目录，
没打开）：

`linear_fp8_sm100.cuh`(828) + `linear_fp8_sm100_common.cuh`(8, 聚合)
+ `_types.hpp` / `_utils` / `_sm100_utils` / `_tma_utils` / `_scheduler` / `_epilogue_utils` /
`_cute_tie` / `linear_fp8_1d2d_sm100.cuh`。

- **`_types.hpp`**：`enum MmaKind {BF16, MXFP8FP4}`、`enum GemmType {Normal, MGroupedContiguous,
  MGroupedMasked, KGroupedContiguous, Batched, MGroupedContiguousWithPsumLayout}`、
  `enum KernelType {Kernel1D1D, Kernel1D2D, KernelNoSF}`——**同一个线性算子 task 按 GEMM 分组形态/是否
  带 scale factor 分档**。
- **`linear_fp8_sm100.cuh`** 的入口 `linear_fp8_sm100_task_impl<...>`（`:45-55`）有**一长串编译期模板
  参数**（`kMajorA/B`、`kGranKA/B`、`SHAPE_*`、`BLOCK_*`、`kNumGroups`、swizzle、`kNumStages`、
  线程数、`kNumMulticast`/`kIsMulticastOnA`、`kNumSMs`、`kWithResidual`、`GemmType`、ILP 模板参数
  `a/b/cd/epilogue dtype`），参数里带**6 个 TMA descriptor**：`tensor_map_a/b` +
  `tensor_map_sfa/sfb`（scale）+ `tensor_map_residual` + `tensor_map_cd`（若无 SF 则 `KernelNoSF`）。
  TMEM 分配器按 `kNumMulticast` 选 `Allocator1Sm`/`Allocator2Sm`（`:61-63`）——2SM 路径即 `docs/21` 的
  Blackwell codegen 目标，单测里也保留。

---

## 5. `sm100_fp8_moe`：grouped FP8 MoE（wrapper 810 + device 1567）

### 5.1 固定 DeepSeek-V3 配置 + **fp32 scale**（与 linear 的 UE8M0 对比）

`runtime_kernel_wrapper.cu` 顶部写死 DSV3 配置（`:43-53`）：`MMA_M=128, MMA_N=16, BATCH=16,
OUTPUT=4096, REDUCTION=7168, NUM_EXPERTS=256, NUM_TOPK=8`，`K_SCALE=7168/128=56`。

**关键差异**：fp8 MoE 的 scale 是 **`float const*`**（`:119-120`、`:757-759`），**不是** linear 的
UE8M0 packed `uint32`。device task 头注释（`fp8_group_gemm_sm100.cuh:76-87`）说得明确：
「expects **per-block float32 scale factors**，**converted to UE8M0 in-kernel**」——
SFA = weight scale `[E, N, K/128]`、SFB = activation scale `[B, topk, K/128]`（W2）或 `[B, K/128]`（W13）。

> 即：**同一份「128K 块 + UE8M0」语义，在仓库里有两种载体**——linear 单测把 UE8M0 **打包在 host**（uint32，
> §2），MoE 单测把 fp32 scale **传到 device 现场转 UE8M0**（`fp8_tcgen05_after_thread_sync` `:153`）。
> 这正对上 `docs/27` 记的「FP8 UE8M0 重定标」在两条链上的两种落地口径。

### 5.2 warp specialization：7 个 warp（`fp8_group_gemm_sm100.cuh:89-108`）

256 线程 = 8 warp，角色固定：

- **warps 0-3（0-127）EPILOGUE**：`TMEM_LOAD` 把 FP32 累加器搬进寄存器 → 转 BF16 → 按 routing
  indices **scatter** 写；
- **warp 4 MMA**：等 A/B smem tile（`a_full/b_full`）与 **scale 转置**（`sf_ready`）→ 发
  `UTCCP`（`SM100_UTCCP_4x32dp128bit_1cta`，include `:32`）把转置后的 scale 拷进 TMEM → 每个 K-tile 发
  **4 条 `SM100_MMA_MXF8F6F4_SS`**（block-scaled FP8 UMMA，include `:34`；`bK/UMMA_K = 128/32 = 4`）；
- **warp 5 DMA**：TMA 载 weight tile（SWIZZLE_128B）、`cp.async` 载 input tile（带 per-token routing 检查）；
- **warp 6 SCALE**：读 fp32 scale → 转 UE8M0、**4 份打包进 uint32**、**warp 转置**成 UTCCP 列主序；
- **warp 7 IDLE**。

SMEM 里 SFA/SFB 各 `kNumStages × 128` 个 `uint32_t`（`:204-207`）；scale buffer 用 `SWIZZLE_NONE`
（`:129`，UTCCP 不需要 swizzle）。这是**与 `docs/07`（Hopper linear task）同一类 warp 特化**、但把
「scale transpose + UTCCP」加进来的 **SM100 block-scaled** 版本。

### 5.3 2D grid + 「emulating MPK runtime」（`runtime_kernel_wrapper.cu:103-143`）

`dsv3_fp8_moe_2d_kernel`（`__launch_bounds__(256,1)`，`:115-116`）用**二维 grid**
`(expert_stride, n_splits, 1)`：

- `blockIdx.x → expert_offset`（`gridDim.x = EXPERT_STRIDE`，`docs/24`/`docs/07` 说 `task_register` 把
  `EXPERT_STRIDE` 设为 `grid_dim.x`，`fp8_group_gemm_sm100.cuh:115`）；
- `blockIdx.y → N-slice`：**每个 CTA 自己按 `blockIdx.y` 平移 weight_scale / output 指针**
  （`:142-150`）。

**源码注释原话**：「Each CTA adjusts its own weight/scale/output pointers based on `blockIdx.y`,
**emulating the MPK runtime's per-CTA pointer offset logic from `runtime.cc`**」（`:13-14`），且逐行注明
「This mirrors MPK runtime's offset logic: `offset = block_size * bid.y * stride[mapped_dim]`」
（`:128-129`）、「from `runtime.cc` lines 1026-1042」（`:112-113`）。

> 这与 `docs/29` §3 记的 `MIRAGE_UNIT_TEST` 用 `blockIdx.y` **手推 expert 指针**同源——单测层为了在
> **没有 `PersistentKernel`** 的情况下复现 runtime 的 `input_map/output_map` 偏移，只能**手写一遍**。
> 即 `docs/28`（test_mode 端到端）bypass 掉的那套 host→device 元数据传递，在单测层被**内联**进 kernel。

device 侧 `fp8_moe_group_gemm_sm100_task_impl`（`:727-753` / `:191-200`）用 `W13_LINEAR` 一个 bool
切换 W13/W2 两种线性、`EXPERT_STRIDE` 做专家分组。PYBIND 暴露 6 个符号（`:791-810`）：W13/W2 单 CTA
正确性测试 + 2D grid + 一组 benchmark setup/launch/cleanup。bf16 对照扩展 `runtime_kernel_wrapper_bf16.cu`
（363 行）是「同结构、换 dtype」的第二份。

---

## 6. headline：`sm100_linear_fp8` 引用了**不存在的头文件**

`runtime_kernel_wrapper_sm100.cu` 顶部**无条件**包含两行（`:18-19`）：

```
#include "blackwell/linear_fp8_sm100.cuh"
#include "blackwell/linear_fp8_sm100_splitk.cuh"   // ← 快照里不存在
```

- include_dirs 含 `tasks/`（`setup.py:79`），故 `blackwell/linear_fp8_sm100.cuh` 解析到
  `tasks/blackwell/linear_fp8_sm100.cuh`（存在）；
- 但 **`tasks/blackwell/linear_fp8_sm100_splitk.cuh` 在整个 `f9eb70c` 快照里不存在**
  （全仓 `grep -r` 只命中 wrapper 自己的这一行 include）；且 split-K 路径用到的
  **`kernel::linear_fp8_sm100_splitk_wrapper`（wrapper `:812`）在全仓也无定义**（只此一处提及）。

⇒ **该扩展在快照里无法编译**；且因 include 在文件顶部、**连 `split_k==1` 的非 split-K 路径也编译不过**。
这是 `docs/30` §8/§9 报的 `sm100_mla_prefill_tp8`「setup.py 指向不存在的源文件」的**同族、且更靠前**
的一处——`docs/30` 只核了注册符号与 `sources=`，没读 wrapper 的 `#include` 头。

> **口径**：这是**静态事实**（缺头 + 缺符号），**不构成「该测试当前一定失败」的运行时判定**——与
> `docs/30` §9 同口径（是否显现取决于该目录是否被构建 / `sys.path` 隔离）。本文只记录「快照里这个
> include 与符号没有对应定义」。[^1]

[^1]: 可能的解释（不判对错）：split-K 实现被迁移/合并进 `linear_fp8_sm100.cuh` 后**忘了删 include 与
    改符号名**，或在某次重构里整个 `_splitk.cuh` 被删。快照 commit 标题是
    「Decouple SM100 attention per-CTA SMEM …; MoE group-GEMM bug fix (#788)」，本次未做构建验证。

---

## 7. `sm100_mla_mtp_decode`：TP standalone wrapper（`docs/30` 尾巴②）

### 7.1 定位：**对外部 standalone 内核的机械 port 验证器**

`test_tp2_wrapper.cu` 顶部注释（`:1-8`）自陈：

> 「Standalone test for the **device-function port** of MLA MTP TP=2 (no PDL). … Goal: verify the port
> (`mla_mtp_tp2_main` / `mla_mtp_tp2_reduce`) is **correctness-equivalent and perf-equivalent** to the
> original standalone `v037_nopdl`, so we know the **mechanical port is clean before doing MPK
> plumbing**.」

device 侧 `mla_mtp_decode_tp{2,4,8}_sm100.cuh` 的头部注释（tp2 `:1-13`）列出**机械改动清单**：
`__global__ → __device__ __noinline__`、`blockIdx.{x,y,z} → block_x/block_y` 参数、
`__grid_constant__ CUtensorMap` 按值 → 按指针、**`cudaTriggerProgrammaticLaunchCompletion()` 删除
（PDL not supported by MPK）**、加 thread guard。并注明「Kernel body（SMEM layout, mbarriers, MMA,
softmax, PV, epilogue）is **byte-identical**」。源头分别是 `v037_tp2_swapab_unrolled_reduce.cu`（tp2
`:3-4`）/ `v007_tp4_swapab.cu`（tp4 `:3`）等。

⇒ 这一层是 **`docs/05` §「手工 kernel 可包成 task」（论文 §7）** 的一个**真实落地片段**：把外部手写
standalone **机械地**改成 device 函数，先验证「port 干净」，再谈接进 MPK。

### 7.2 TP = head 维切分（`NUM_HEADS = 128 / TP`）

| 变体 | `NUM_HEADS` | 位置 | RD_TB |
|---|---|---|---|
| base `mla_mtp_decode` | 128 | `mla_mtp_decode_sm100.cuh:38` | 512 |
| TP2 | 64 | `mla_mtp_decode_tp2_sm100.cuh:40` | 256 |
| TP4 | 32 | `mla_mtp_decode_tp4_sm100.cuh:29` | 256 |
| TP8 | 16 | `mla_mtp_decode_tp8_sm100.cuh:26` | 256 |

其余常量四者相同（`D_K=576, D_V=512, TILE_S=128, BK=64, MMA_K=16, K_ITERS=9, V_CHUNKS=8, TB=128,
NUM_QK_STAGES=5, NUM_PV_STAGES=3`）——**TP 只动 head 维并行度**，`D_K/D_V` 不变（MLA 的 head dim 固定）。

### 7.3 「PTX 惯用法」每个变体重写一份

base 用共享的 `kernel::sm100_ptx`；而 **tp2/tp4/tp8 各自在 `kernel::mla_mtp_tpN` 下重定义一份
`namespace ptx`**（tp2 `:65-128`），含 `elect_sync / mbar_init / mbar_wait / mbar_tx / desc_enc /
make_desc / tcgen05_mma / tcgen05_commit`——`tcgen05.mma.cta_group::1.kind::f16`（tp2 `:109-114`）。
三个变体各自 700+ 行**几乎逐字重复**这份 `ptx` 与流水骨架。

这是 `docs/19` §5 / `docs/30` §5「**同一套硬件惯用法在仓库里实现了 N 遍**」母题的**又一肉身**——
而且是在**同一目录内**为了 TP 2/4/8 三个度数复制三份（`docs/30` §5 报的第三肉身是 `cute/hopper/gemm.cu`）。
另有微差：tp4 在 `:149` 引入了 `PV_CHUNKS = V_CHUNKS/2`（每半 4 chunk），其他变体没这行——
即便号称「byte-identical」，TP=4 的 PV 分工仍与 tp2/tp8 不同。

### 7.4 standalone 形态与参考

`test_tp2_wrapper.cu`（360 行，含 `int main:145`）内含一个**逐张量 naive 参考** `ref_k`（`:69-128`：
bf16 QK → softmax（`causal_lim` 因果掩码）→ PV），与 `docs/30` §4 记的 `moe_linear` 单测同构——
**「调 device 函数 → 跟朴素参考比数值」**，只是入口是 `main` 而非 pybind。`MLA_TP_SYNC_ACTIVE()`
= `asm volatile("bar.sync 12, 128;")`（tp2 `:38`），用**命名 barrier 12**（不用 barrier 0 / `__syncthreads`）
——为的是与 MPK 更大的 launch 配置共存（同「mechanical changes」里加的 thread guard）。

---

## 8. 三方对照（本文新打开的层）

| 维度 | `sm100_linear_fp8` | `sm100_fp8_moe` | `sm100_mla_mtp_decode/tp*` |
|---|---|---|---|
| 入口 | pybind `linear_fp8_1d2d_sm100` | pybind（6 符号） | **`int main`**（standalone，无 pybind） |
| scale 载体 | **UE8M0 packed uint32**（host 打包/重排） | **fp32**（device 转 UE8M0） | 无（bf16 attention） |
| scale 布局层 | `common/sm100_fp8_scale_layout.{h,py}` | 无（直接 fp32） | — |
| device task | `linear_fp8_sm100_task_impl`（11 头家族） | `fp8_moe_group_gemm_sm100_task_impl`（1567 行） | `mla_mtp_tpN_main/reduce`（重定义 ptx） |
| 一个 task | 一 CTA（grid=output tiles×split_k） | 一 CTA（grid=(expert_stride,n_splits)） | device 函数（shim 转发 blockIdx） |
| 来源 | 库内手写 | 库内手写 | **外部 standalone（`v037_tp*`）机械 port** |
| 可编译 | **引用了不存在的头文件**（§6） | 静态看齐 | 静态看齐 |

---

## 9. 与 `docs/29`/`docs/30` 的关系 + 判据映射

- **补的尾巴**：`docs/30` §9 两条（fp8 大 wrapper 内部 / `test_tp*_wrapper.cu` 内部）。
- **修正/加强 `docs/30`**：`docs/30` §5 说「同一套惯用法实现了两遍」的第三肉身是 `cute/hopper/gemm.cu`；
  本文补两处——**fp8 scale 语义的两种载体**（§2 vs §5.1）、**TP 变体各自重定义 ptx**（§7.3）。
- **新增证据**：`docs/30` §2/§8 的「缺源文件」清单再加一条**缺头文件**（§6）。
- **判据映射**：与 `docs/29` §6 / `docs/30` §7 **完全一致**——本层把「一个 device task 当成一整颗 kernel」
  由 host 直接 launch，不经 `PersistentKernel`/event/tGraph/worker-scheduler，**不贡献任何 C 判据**；
  仅作 C3（融合范围）范围内「单 device task 数值正确」的回归网。**唯一与 C1–C5 有关的事实**是：
  TP port 明确**删掉了 `cudaTriggerProgrammaticLaunchCompletion`（PDL）**，说明这些外部 standalone 曾用
  PDL（P4-2/P5-3 的降级手段），而 MPK 的运行时**不支持 PDL**（它要常驻单 kernel，见 `docs/03` §2.5）。

---

## 10. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：§6 的「缺头文件」是 `#include` + 全仓 `grep` 的**静态对照**，未
  `pip install`、未 `nvcc` 编该扩展，**不构成「该测试当前一定失败」的运行时判定**（同 `docs/30` §9）。
- **SM100 不可执行**：`sm_100a` 形态在目标卡 H100/sm90 不能跑（同 `docs/17`/`docs/21`/`docs/29`/`docs/30`），
  本文所有 UMMA/UTCCP/TMEM 描述均为**静态走读**，未取任何 device 计数。
- **未展开的更深层**：`fp8_group_gemm_sm100.cuh`（1567 行）只走读了 warp 角色与 scale 路径，
  **UMMA 主循环 / UTCCP 的具体 smem→TMEM 搬运指令序列 / epilogue scatter** 未逐行；`linear_fp8_sm100`
  的 11 头家族里 `_scheduler`/`_epilogue_utils`/`_cute_tie` 未逐行走；TP 变体的 `mla_mtp_tpN_main`
  主体（SMEM layout / mbarrier / softmax / PV）只核了常量与 `ptx` 命名空间。
- **未与生产路径对跑**：`docs/24` 的 `TaskType` 目录里这些 SM100 task 的登记/接线未复核；DeepGEMM
  SM100 mega 路径（SM100 专属）本机不具备，未对照。

---

## 11. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/30` 同快照）。
- **共享 scale 布局层**：
  `tests/runtime_python/blackwell/common/sm100_fp8_runtime_registry.h`
  （`kBlockK:11`、`kScalePackSize:12`、白名单 `:13-17`、`packed_scale_k*:53-56`、
  `aligned_scale_outer_dim:62-64`、`enum PackedScaleLayout:66-70`、`detect_scale_layout:95-110`）；
  `.../common/sm100_fp8_scale_layout.h`（`detect_scale_tensor_layout:10-21`、`check_scale_tensor:23-55`）；
  `.../common/sm100_fp8_scale_layout.py`（`detect_packed_scale_layout:52-62`、`encode/decode_ue8m0:65-72`、
  `quantize_to_fp8_packed_ue8m0:75-119`、deepgemm 版 `:122-152`）。
- **`sm100_linear_fp8`**：`setup.py`（`name:68/71`、nvcc `:88-93`）；
  `runtime_kernel_wrapper_sm100.cu`（include `:18-19`、`choose_split_k:148-162`、
  `materialize_internal_scale_tensor:174-200`、`make_tma_sf_desc:318-328`、
  `linear_fp8_dynamic_smem_bytes:330-375`、`launch_kernel_ex_cluster:377-411`、
  `launch_linear_fp8_1d1d_sm100:415`、`kDynamicSmemBytes` 断言 `:449-464`、
  `launch_linear_fp8_1d1d_sm100_splitk:674`、splitk `kernel_ptr:811-812`、
  `linear_fp8_1d2d_sm100_kernel:874-1094`、dispatch `:1000-1087`、`PYBIND11_MODULE:1100-1106`）。
  device：`tasks/blackwell/linear_fp8_sm100_types.hpp`（`MmaKind/GemmType/KernelType`）、
  `tasks/blackwell/linear_fp8_sm100.cuh:18-55`（模板参数 + 6 个 TMA map）、`:61-63`（TMEM Allocator1Sm/2Sm）。
- **`sm100_fp8_moe`**：`setup.py`（两个扩展 `:45/67`、`name:84`、nvcc `:35-37`）；
  `runtime_kernel_wrapper.cu`（注释 `:1-14`、DSV3 常量 `:43-56`、`dsv3_fp8_moe_2d_kernel:115-200`、
  指针平移 `:142-150`、`task_impl` 调用 `:727-753`/`:191-200`、`PYBIND11_MODULE:791-810`）；
  device `tasks/blackwell/fp8_group_gemm_sm100.cuh`（scale 语义 `:76-87`、warp 角色 `:89-108`、
  2D 分布 `:110-118`、`UTCCP` include `:32`、`MXF8F6F4` include `:34`、
  `fp8_tcgen05_after_thread_sync:153`、SFA/SFB smem `:204-207`）。
- **`sm100_mla_mtp_decode`**：`test_tp2_wrapper.cu`（注释 `:1-8`、`shim_main:25-47`、`shim_reduce:49-66`、
  `ref_k:69-128`、`int main:145`）；device `tasks/blackwell/mla_mtp_decode_tp2_sm100.cuh`
  （头部 `:1-13`、`MLA_TP_SYNC_ACTIVE:38`、`NUM_HEADS:40`、`ptx` 命名空间 `:65-128`）、
  `_tp4_sm100.cuh`（`:3` 源头 `v007_tp4`、`NUM_HEADS:29`、`PV_CHUNKS:149`）、
  `_tp8_sm100.cuh`（`NUM_HEADS:26`）、base `mla_mtp_decode_sm100.cuh`（`NUM_HEADS:38`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量 / 单测层，同 `docs/28`/`docs/29`/`docs/30` 判断）；论及「手工 kernel 可包成 task」见 `docs/05`。
- 同专题相关篇：`docs/29`/`docs/30`（本文补其 §9 尾巴）、`docs/27`（FP8 UE8M0 权重预处理，
  与本文 §2 同一约定）、`docs/24`（`register_task`/`EXPERT_STRIDE`）、`docs/07`（Hopper linear task 的
  warp 特化，与本文 §5 同族）、`docs/21`（Blackwell UMMA/TMEM/cluster codegen）、`docs/19` §5、
  `docs/28`（test_mode 端到端层，与单测层相对）、`docs/22` §8（C 判据总映射，本层不贡献）。
- 交叉专题：DeepGEMM mega 家族前向/反向验证见 `../mega-moe/`（本专题不重复）。
