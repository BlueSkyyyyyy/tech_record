# MPK SM100 block-scaled FP8 linear 支撑层走读：`_utils` / `_sm100_utils` / `_tma_utils`（+ 描述符位编码 host 探针）

> 定位：本文补掉两处**明留的尾巴**：
>
> - `docs/64`（P150-1）§12「未走的相邻层」——
>   > `linear_fp8_sm100_tma_utils.cuh` / `linear_fp8_sm100_sm100_utils.cuh` 只借用了行号
>   > （`docs/32` 已给 `:16-116` 摘要），未逐行。
> - `docs/57`（P143-1）§8「局限」——
>   > 「committed 描述符的位编码未逐位展开」……`make_smem_desc` 的 `version_`/`lbo_mode_` 语义
>   > 依赖 GPU 架构文档，本文只记「写死值」。
>
> `docs/32`（P119-1）§1 列过 `linear_fp8_sm100` 家族的 10 个文件，`docs/60`（P146-1）走完了
> `_cute_tie`、`docs/64` 走完了 `_scheduler`/`_epilogue_utils`，但**支撑层的三支**
> （`linear_fp8_sm100_utils.cuh` 143 行、`linear_fp8_sm100_sm100_utils.cuh` 232 行、
> `linear_fp8_sm100_tma_utils.cuh` 118 行）此前只有 `docs/32` §2.6 的一页摘要。本文把三支逐行打开，
> 并把其中**纯算术**（`SmemDescriptor` / `make_umma_desc` / `InstrDescriptorBlockScaled` /
> `get_num_aligned_tmem_cols`）在 host 上**逐字重放、打印真实描述符位模式**。
>
> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `/ssd/mirage_mpk_build`，与
> `docs/06`–`docs/67` 同快照）；CUTLASS 侧走 `deps/cutlass` submodule，commit `f3fde583`。
> **纯源码走读 + 1 个 host 侧描述符位探针**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，
> 但 `SmemDescriptor`/`InstrDescriptorBlockScaled` 是 union+bitfield（`CUTE_HOST_DEVICE`），
> `make_instr_desc_block_scaled` 是 `CUTE_HOST_DEVICE`，故它们的**位编码**在 host 上可精确复现。
> 所有行号为快照静态事实。相关机制实测交叉引用 `docs/04`。

---

## 0. 三条 headline

**(A)（headline）支撑层其实是「一份完整版 + 一份 K-major 子集副本」——同一批 helper 的两处分叉，
不是简单复制。** `linear_fp8_sm100_sm100_utils.cuh`（232 行，身体 A `linear_fp8_sm100.cuh` 经
`linear_fp8_sm100_common.cuh:5` 用）含 `to_umma_layout_type`（`_sm100_utils.cuh:48-79`）、
`get_atom_base`（`:43-46`）、`make_umma_desc` 的 **K-major 与 MN-major 两个分支**（`:105-145`）；
而 `sm100_utils.cuh`（152 行，身体 B `linear_fp8_1d2d_sm100.cuh:41` 与 MoE
`fp8_group_gemm_sm100.cuh:18` 用）是它的**前身/子集**——`advance_umma_desc_lo` 里一行
`static_assert(kMajorMode == cute::UMMA::Major::K, "Current Mirage helper only supports K-major
path.")`（`sm100_utils.cuh:79-80`）把它锁在 K-major；`make_sf_desc` 还多了一段「为什么 SWIZZLE_NONE」
的解释性注释（`sm100_utils.cuh:59-66`）。两处**同名函数各定义一遍**（consumer audit 见 §5）。

**(B)（headline）描述符位编码逐位钉死。** host 探针重放 `make_smem_desc`（`_sm100_utils.cuh:13-28`）
得到：三个字节偏移字段（`start_address_`/`leading_byte_offset_`/`stride_byte_offset_`）都**右移 4 位**
（16 B 粒度）存，`version_` 落 bit[46,48)、`base_offset_` 落 bit[49,52)、`lbo_mode_` 落 bit[52]、
`layout_type_` 落 bit[61,64)（host 探针 §A）。`make_sf_desc` = `SWIZZLE_NONE` + `stride=8*16=128 B` +
`leading=0`（`_sm100_utils.cuh:30-34`）。指令描述符 `InstrDescriptorBlockScaled`（CUTLASS
`mma_sm100_desc.hpp:441-468`）对 M=128、N=16/128/256 实测 `desc32 = 0x08840000 / 0x08a00000 /
0x08c00000`（`n_dim_ = N>>3`、`m_dim_ = M>>4`）；MPK 的 `make_runtime_instr_desc_with_sf_id`
（`_sm100_utils.cuh:147-154`）= 填入 `a_sf_id_`/`b_sf_id_` 后 `<<32`，与 CUTLASS 同名 constexpr
**逐位 MATCH**（§B/§D）。

**(C)（headline）`_tma_utils.cuh` 三项 tensormap helper 全仓零调用者；`tma_copy` 只服务身体 A；
2SM MMA 结构体零消费者。** `tensor_map_release_cta`（`:93-95`）、`tensor_map_acquire_cta`
（`:97-104`）、`tensor_map_replace_global_addr_in_smem`（`:106-116`）在整棵 MPK 树
（`include`+`src`）**定义处之外 0 引用**；`tma_copy`（`:16-91`）只被 `linear_fp8_sm100.cuh`
（身体 A）调用，身体 B/MoE（走 CuTe 原生 TMA）一次不用；`SM100_MMA_MXF8F6F4_2x1SM_SS`
（`:207-230`，`cta_group::2`）也只有定义。这是「有实现、零调用」族（`docs/16`/`docs/20`/
`docs/32`/`docs/45`）在支撑层的又一实例。

---

## 1. 定位与 include 链

支撑层三支互为依赖，最终被 `linear_fp8_sm100_common.cuh` 汇进身体 A：

```
linear_fp8_sm100_utils.cuh        (143)  PatternVisitor / 算术 / SMEM ld·st / bf16 打包
   ^
linear_fp8_sm100_tma_utils.cuh    (118)  tma_copy 分派 + 三个 tensormap helper
   ^
linear_fp8_sm100_sm100_utils.cuh  (232)  SmemDescriptor / make_umma_desc / InstrDescriptorBlockScaled / 两个 MMA 结构体
   ^
linear_fp8_sm100_common.cuh       (8)    聚合 include（body A 的入口）
```

`linear_fp8_sm100_sm100_utils.cuh:8-9` include 前两支；`linear_fp8_sm100_tma_utils.cuh:7`
include `_utils`；`linear_fp8_sm100_common.cuh:5-8` 把三支与 `_types`/`_scheduler`/`_epilogue_utils`
聚合。**身体 B/MoE 走的是另一支 `sm100_utils.cuh`**（见 §5），不 include 这三支。

---

## 2. `linear_fp8_sm100_utils.cuh`（143 行）逐行

### 2.1 断言宏（`:10-33`）

`DG_DEVICE_ASSERT`（`:10-21`，失败时 `printf` 文件名/行号/条件再 `asm("trap;")`）、
`DG_TRAP_ONLY_DEVICE_ASSERT`（`:23-29`）、`DG_STATIC_ASSERT`（`:31-33`）——三个宏都带 `#ifndef`
守卫，即「其它头若已定义就复用」。这是 `DG_*` 家族（DeepGEMM 命名习惯，对应 `docs/33` 里的
`DG_STATIC_ASSERT`）在 MPK 侧的定义处。

### 2.2 `PatternVisitor`：把「第 i 个 stage 的地址/屏障」变成可索引的 functor（`:37-47`）

```cpp
template <typename FuncT> struct PatternVisitor {
  FuncT func;
  __device__ __host__ explicit PatternVisitor(FuncT &&func) : func(std::forward<FuncT>(func)) {}
  __device__ __host__ auto operator[](uint32_t const &i) { return func(i); }
};
```

它把「一个 lambda(算 i 号 stage 的 SMEM 指针 / mbarrier 指针)」包成 `foo[i]` 的可索引对象。
身体 A 用它造 10 个视图（`linear_fp8_sm100.cuh`：`smem_cd:184`、`smem_a:195`、`smem_b:200`、
`smem_sfa:209`、`smem_sfb:213`、`full_barriers:223`、`empty_barriers:225`、
`with_sf_full_barriers:227` 等），把「stage 数」这一维**从模板参数摊成运行期下标**。身体 B/MoE
不用它（consumer audit：`PatternVisitor` 只出现在 `_utils.cuh` 与身体 A）。

### 2.3 算术工具（`:49-72`）

`ceil_div`（`:49-52`）、`constexpr_ceil_div`（`:54-57`）、`align`（`:59-62`）、`constexpr_align`
（`:64-67`）、`constexpr_gcd`（`:69-72`）——`__device__ __host__`，可 host 求值。`ceil_div`
被身体 A、`_scheduler.cuh`、`mla_sm100_2sm.cuh` 各自 include 后复用（故它同时是「同名多份」族，
仓库里 `attention_sm100.cuh` 也有自己的 `align`）。`get_sm_idx`（`:81-85`，`mov.u32 %smid`）
与 `constexpr_gcd`（`:69-72`）在快照里**定义处之外零引用**。

### 2.4 `ld_shared` / `st_shared`：绕过编译器的 SMEM 访存（`:93-113`）

```cpp
ld_shared   :93-99   asm("ld.shared.u32 %0, [%1];")
st_shared   :101-105 asm("st.shared.u32 [%0], %1;")
st_shared   :107-113 asm("st.shared.v4.u32 [%0], {%1,%2,%3,%4};")   // 128-bit 存
```

三个函数都把指针经 `__cvta_generic_to_shared` 转成 shared 地址后用**裸 PTX** 访存
（`ld.shared`/`st.shared`），即「显式控制 SMEM 访问」。身体 A 用它们读写 SMEM（如
`smem_cd` 的 swizzle 地址落位）。这是与 `docs/51`（SMEM 布局即 ABI）同层的机制。

### 2.5 bf16 打包与「寄存器内加残差」（`:115-141`）

```cpp
cast_into_bf16_and_pack(old_t &x, old_t &y)              // :115-120  float2 -> bf16x2(打包成 int)
add_packed_bf16x2_into_fp32_bits(packed_residual, x_bits, y_bits)  // :122-141
```

`cast_into_bf16_and_pack` 用 `__float22bfloat162_rn` 把两个 fp32 的 bits 打包成一个 bf16x2
（返回 int）。`add_packed_bf16x2_into_fp32_bits` 用 union 把「一个打包 bf16x2 残差」拆成 `float2`
再加到两个 fp32 的 bits 上——**残差在寄存器里以 fp32 精度加、再统一转 bf16**。身体 A 的 epilogue
（`linear_fp8_sm100.cuh:680-693`）正是这样把残差融合进 TMEM→寄存器的输出再写回。这与 `docs/07`
§1.2「寄存器内融合残差」同构，是「段级融合」把残差加从独立 kernel 折进 GEMM epilogue 的落点。

---

## 3. `linear_fp8_sm100_sm100_utils.cuh`（232 行）逐行

### 3.1 `make_smem_desc` / `make_sf_desc` / `replace_smem_desc_addr`（`:13-41`）

`make_smem_desc(layout, smem_ptr, stride_byte_offset, leading_byte_offset)`（`:13-28`）构造
UMMA 的 64 位 SMEM 描述符：

```cpp
desc.version_ = 1;  desc.lbo_mode_ = 0;  desc.layout_type_ = (uint8_t)layout;
desc.start_address_ = (uint16_t)(uint_ptr >> 4);   // 16B 粒度
desc.base_offset_  = 0;
desc.stride_byte_offset_  = stride_byte_offset  >> 4;
desc.leading_byte_offset_ = leading_byte_offset >> 4;
```

`make_sf_desc(smem_ptr)`（`:30-34`）= `make_smem_desc(SWIZZLE_NONE, smem_ptr, 8*16, 0)`，即
**scale-factor 的 SMEM 暂存用「非 swizzle + 128 B 步长」**（与 `docs/59`/`docs/60` 的 UTCCP
scale 路径一致）。`replace_smem_desc_addr`（`:36-41`）只改 `start_address_`，用于运行期换地址。

**host 探针实算**（`src/fp8_linear_support.cu` §A，假 SMEM 基址 `0x400`）：`make_smem_desc`
对 `SWIZZLE_NONE stride=128 leading=0` 得 `desc_ = 0x0000400800000040`（`stride_byte_offset_=8`、
`layout_type_=0`）；对 `SWIZZLE_128B stride=1024 leading=0` 得 `0x4000404000000040`
（`stride_byte_offset_=64`、`layout_type_=2`）；对 `SWIZZLE_128B_BASE32B stride=512 leading=2048`
得 `0x2000402000800040`（`leading=128`、`stride=32`、`layout_type_=1`）。**这直接补上了 `docs/57`
§8 那条「位编码未逐位展开」的尾巴**：三个偏移都做 `>>4`，`version_`/`lbo_mode_`/`layout_type_`
落在文档位域（CUTLASS `mma_sm100_desc.hpp:98-123`）声明的位置。

### 3.2 `to_umma_layout_type` / `get_atom_base`（`:43-79`）

`get_atom_base`（`:43-46`）= `SWIZZLE_128B_BASE32B ? 32 : 16`。`to_umma_layout_type<kMajorMode,
kSwizzleMode,kUseBase32,dtype>`（`:48-79`）把 MPK 的「swizzle 字节数」映射到 CUTLASS
`UMMA::LayoutType`，且有**两条强制 BASE32 规则**：

- `dtype==float && kMajorMode==MN` ⇒ 必须 `kUseBase32`，否则 `DG_STATIC_ASSERT(kUseBase32,
  "Invalid swizzling base")`（`:57-62`）；
- `kUseBase32` 为真时恒返回 `SWIZZLE_128B_BASE32B`。

host 探针 §B1 实测映射表：`fp8 K, swizzle 0/16/32/64/128 -> SWIZZLE_NONE/NONE/32B/64B/128B`；
`bf16 MN, 0 -> SWIZZLE_NONE`；`float MN (base32) -> SWIZZLE_128B_BASE32B`。即
**「float + MN-major」是唯一强制 BASE32 的组合**（对应 C 累加器/float 操作数的特殊布局）。

### 3.3 `get_umma_desc_stride_k` / `advance_umma_desc_lo` / `make_umma_desc`（`:81-145`）

`get_umma_desc_stride_k<kMajorMode,BLOCK_MN,kSwizzleMode,dtype>`（`:81-89`）= K-major 时
`1`、MN-major 时 `get_inner_block_atom_size<BLOCK_MN,kSwizzleMode,dtype>`（即
`swizzle==0 ? BLOCK_MN : swizzle/sizeof(dtype)`，定义在 `_tma_utils.cuh:11-14`）。

`make_umma_desc`（`:105-145`）分两支：

- **K-major**（`:118-127`）：`DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K*sizeof(dtype))`；
  `stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype)`（`num_non_contiguous =
  128/get_atom_base`，即 8 或 4）；`leading_byte_offset = 0`；地址 `base + mn_idx*BLOCK_K +
  k_idx*stride_k`。
- **MN-major**（`:128-144`）：`stride = num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype)`、
  `leading = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype)`；**`kSwizzleMode==16` 时把 stride/leading
  对调**（`:137-139`）。另有一句 `DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0)`。

**host 探针实算**（§B2/B3/B4）：K-major、fp8、`BLOCK_MN=128 BLOCK_K=128 swizzle=128`：
`stride_byte_offset = 8*128*1 = 1024 (>>4=64)`、`leading=0`，`mn_idx=0/1/2` 的 `start_address_`
= `4096/4104/4112`（每加 1 个 mn 前进 `BLOCK_K=128` 元素 = 128 B = `>>4` 8 个单位）。MN-major、
bf16、`BLOCK_MN=64 BLOCK_K=64 swizzle=64`：`BLOCK_MN_ATOM = 64/2 = 32`、
`stride = 8*32*2 = 512 (>>4=32)`、`leading = 64*32*2 = 4096 (>>4=256)`、`stride_k = 32` 元素。
swizzle=16 时 `stride/leading` 被对调到 `32/8`，且 `layout_type_` 落到 `SWIZZLE_NONE`。

### 3.4 `make_runtime_instr_desc_with_sf_id`（`:147-154`）

```cpp
desc.a_sf_id_ = sfa_id;  desc.b_sf_id_ = sfb_id;
return (uint64_t)(uint32_t)desc << 32;   // idescE 的高 32 位
```

它把 block-scaled 指令描述符塞进 `idescE` 的高 32 位（低 32 位对 dense 无意义）。这与 CUTLASS
`make_runtime_instr_desc_block_scaled`（`mma_sm100_desc.hpp:588-621`）同构，但**来源相反**：
CUTLASS 从 TMEM 地址 `bit[30,32)` 抽 `sf_id`（`(tmem_sf_addr & 0xC0000000) >> 30`），MPK 直接按
参数填（`docs/60` §6 已记这对「填同两位域、来源相反」）。`a_sf_id_` 落 bit[29,31)、`b_sf_id_`
落 bit[4,6)（`InstrDescriptorBlockScaled` 位域，`mma_sm100_desc.hpp:450/461`）。

host 探针 §D2：`desc32(a_sf_id=1,b_sf_id=2)=0x08a00000`、`idescE=0x28a0002000000000`（低 32 位 = 0）。

### 3.5 `get_num_aligned_tmem_cols`（`:156-172`）

按列数向上取整到 `{32,64,128,256,512}`，`DG_STATIC_ASSERT(kNumCols <= 512)`。host 探针 §C
实测：`1..32 -> 32`、`33..64 -> 64`、`65..128 -> 128`、`129..256 -> 256`、`257..512 -> 512`。
（`docs/59` §8 已记 `linear_fp8_sm100.cuh` 里 `get_num_aligned_tmem_cols` 有两套公式各算一遍。）

### 3.6 `tcgen05` fence 与两个 MMA 结构体（`:174-230`）

`tcgen05_before_thread_sync`（`:174-176`，`tcgen05.fence::before_thread_sync`）与
`tcgen05_after_thread_sync`（`:178-180`）是「TMEM 异步指令与线程同步之间的栅栏」，被身体 A 在
`__syncthreads` 前后使用。`SM100_MMA_MXF8F6F4_SS`（`:182-205`）发
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%5], [%6], p;`
（`%3` = `desc>>32`、`p = scale_c != 0`）；`SM100_MMA_MXF8F6F4_2x1SM_SS`（`:207-230`）只有
`cta_group::1 -> ::2` 一处差别。**身体 A 只实例化前者**（`linear_fp8_sm100.cuh:486-487`），
后者在快照里**零消费者**（consumer audit §5）——同 `docs/32` §2.6「单测里可选的 2SM 路径未在
这个身体里被激活」。

---

## 4. `linear_fp8_sm100_tma_utils.cuh`（118 行）逐行

### 4.1 `get_inner_block_atom_size` 与 `tma_copy`（`:11-91`）

`get_inner_block_atom_size<BLOCK_INNER,kSwizzleMode,dtype>`（`:11-14`）=
`kSwizzleMode==0 ? BLOCK_INNER : kSwizzleMode/sizeof(dtype)`——**同一 swizzle 下「一个原子覆盖多少
内层元素」由元素宽度决定**（fp8 与 bf16 不同）。

`tma_copy`（`:16-91`）是一个**按 `<kIs3DTMA, num_tma_multicast>` 分四路的一维 tile 拷贝**：
它把 `BLOCK_INNER` 拆成 `BLOCK_INNER/BLOCK_INNER_ATOM` 个原子，逐个发 TMA：

| 条件 | 指令 |
|---|---|
| `!kIs3DTMA && num_tma_multicast==1` | `cute::SM90_TMA_LOAD_2D::copy`（`:40`） |
| `!kIs3DTMA && num_tma_multicast!=1` | `cute::SM100_TMA_2SM_LOAD_2D::copy`（`:52`，`__CUDA_ARCH__>=1000` 保护） |
| `kIs3DTMA && num_tma_multicast==1` | `cute::SM90_TMA_LOAD_3D::copy`（`:66`） |
| `kIs3DTMA && num_tma_multicast!=1` | `cute::SM100_TMA_2SM_LOAD_3D::copy`（`:79`） |

开头有一条 `DG_STATIC_ASSERT`（`:29-32`）校验 `CacheHintSm90::EVICT_NORMAL ==
CacheHintSm100::EVICT_NORMAL`。**`num_tma_multicast` 是运行期参数**：`==1` 走 SM90 单 CTA 载入，
`!=1` 走 SM100 的 `cta_group::2` 2SM 载入（与 `docs/04` §18–§20 的 cluster/multicast 实测同轴；
生产登记 `kNumMulticast=1`，见 `docs/32` §5.3）。身体 B/MoE 不用 `tma_copy`（走 CuTe 的
`make_tma_copy`/`copy`），故 `tma_copy` 实为**身体 A 专用**。

### 4.2 三个 tensormap helper —— 零调用者（`:93-116`）

```cpp
tensor_map_release_cta()                    :93-95   fence.proxy.tensormap::generic.release.cta;
tensor_map_acquire_cta(gmem_desc_ptr)       :97-104  fence.proxy.tensormap::generic.acquire.cta [ptr], 128;
tensor_map_replace_global_addr_in_smem()    :106-116 tensormap.replace.tile.global_address.shared::cta.b1024.b64
```

这三个是「运行期改 TMA tensormap 的 global address 并做 proxy fence」（用于同一常驻 kernel 里
动态换 tensor 基址的场景）。但 consumer audit（§5）显示它们在 **整棵 MPK 树（`include`+`src`）里
定义处之外 0 引用**——一枚典型的「有实现、零调用者」（`docs/16`/`docs/20`/`docs/32`§3/`docs/45`）。

---

## 5. consumer audit：两份副本、零调用者（`harness/fp8_linear_support.out.txt`）

**(a) 同名 helper 各定义两遍**——`linear_fp8_sm100_sm100_utils.cuh` vs `sm100_utils.cuh`：

| symbol | `linear_fp8_sm100_sm100_utils.cuh` | `sm100_utils.cuh` |
|---|---|---|
| `make_smem_desc` | ✓ (`:13-28`) | ✓ (`:33-57`) |
| `make_sf_desc` | ✓ (`:30-34`) | ✓ (`:59-66`，**多一段注释**) |
| `replace_smem_desc_addr` | ✓ | ✓ |
| `make_umma_desc` | ✓（**K + MN 双分支**） | ✓（**K-only**，`advance_umma_desc_lo` 有 `static_assert("Current Mirage helper only supports K-major path.")`） |
| `make_runtime_instr_desc_with_sf_id` | ✓ | ✓ |
| `advance_umma_desc_lo` | ✓（K+MN） | ✓（K-only） |
| `SM100_MMA_MXF8F6F4_SS` | ✓ | ✓ |
| `to_umma_layout_type` / `get_atom_base` / `get_umma_desc_stride_k` | ✓ | ✗（无） |
| `get_num_aligned_tmem_cols` / `tcgen05_*_thread_sync` | ✓ | ✗（无） |
| `SM100_MMA_MXF8F6F4_2x1SM_SS` | ✓ | ✗（无） |

即 `sm100_utils.cuh`（152 行）是 `linear_fp8_sm100_sm100_utils.cuh`（232 行）的**K-major 子集**
——身体 A 用完整版（要 MN-major / float-base32 / swizzle-16 对调 / 2SM 结构体），身体 B/MoE 用子集
（只走 K-major）。两处 include 分工明确：`linear_fp8_sm100_common.cuh:5` 拉完整版（→ 身体 A）、
`linear_fp8_1d2d_sm100.cuh:41` 与 `fp8_group_gemm_sm100.cuh:18` 拉子集。

**(b) `tma_copy` 只服务身体 A**：`tma_copy` 仅出现在 `_tma_utils.cuh`（定义）与
`linear_fp8_sm100.cuh`（使用，5 处）；身体 B / MoE 均 0。`PatternVisitor` 同样只在 `_utils.cuh`
与身体 A（10 处）。

**(c) 零调用者**：`tensor_map_release_cta` / `tensor_map_acquire_cta` /
`tensor_map_replace_global_addr_in_smem` 全仓定义外 0 引用；`SM100_MMA_MXF8F6F4_2x1SM_SS` 0 引用；
`get_sm_idx` / `constexpr_gcd` 0 引用。

---

## 6. 判据映射（C1–C5）与 megakernel 含义

- 本文是**单 task device 身体内部的支撑层**——描述符/指令编码、TMA 分派、SMEM 访存与 bf16 打包。
  它**不改变 task 的提交方式**，故对 C1「单次 launch」/C2「常驻」无贡献；对 C4「设备内协调」/
  C5「SM 级调度」也无贡献。
- 它贡献 **C3（多算子范围 / 设备体内布局与操作数契约）**：正是「把 SM100 block-scaled FP8 GEMM
  编成一个可被常驻 worker 调用的 device 函数」所需的**操作数与指令契约**这一层，与 `docs/07`
  （Hopper task 内部）/`docs/32`（SM100 linear 身体）/`docs/57`（共享原语）同档。
- **megakernel 视角**：`tma_copy` 的 `num_tma_multicast` 分支、`SM100_MMA_*_2x1SM_SS`、
  三个 tensormap helper 都是「为**段级 megakernel**（一个 task 覆盖一整个 GEMM/MoE 段）预备的
  运行期灵活性」。生产登记 `kNumMulticast=1`（`docs/32` §5.3）说明**身体 A 在产线走单 CTA 路径**，
  multicast/2SM/运行期换基址这些能力**已备但未启用**——与 `docs/16`/`docs/20`/`docs/28` 记的
  「有实现未启用」是同型现象（这里只是未接线，不是死码分支）。

---

## 7. 局限（诚实标注）

- **SM100 指令/硬件本体未执行**：本文的位模式是**在 host 上对真实 CUTLASS 头文件（`f3fde583`）
  求值/重放**得到的，属「trait 与源码规定的编码」，**不是** SM100 上观测到的硬件行为；`tcgen05.mma`
  的实际描述符解析、TMA `cta_group::2` 的行为均**未验证**。
- **重放的是 MPK 源码的算术，不是其 `__device__` 函数的直接执行**：`make_smem_desc`/`make_umma_desc`
  是 `__device__`（用 `cute::cast_smem_ptr_to_uint`，host 不可调），故 `src/fp8_linear_support.cu`
  按**同一 bitfield 赋值与同一公式**在 host 复现，并（§D3）与 CUTLASS 同名 constexpr 逐位比对
  （MATCH）。这保证了「编码算术」正确，但不等于「MPK 在 SM100 上跑出来就是这个值」。
- **未构建/未跑 MPK 的 FP8 线性 task**：SM100 专属，本机不可执行；本文是纯源码走读 + host 位探针。
- **consumer audit 是 `grep` 级**：计数可能受宏/注释影响；`make_sf_desc`/`make_umma_desc` 等
  的「两处定义」是静态事实，「哪支更新」本文**不判**（同 `docs/32` §3 的「按名字推断会错」族）。
- **`_utils.cuh` 的三个位探针只覆盖「文档化的公开位域」**：CUTLASS 位域注释里若有个别保留位/语义
  未在其头文件说明（如 `base_offset_` 的 3 位具体用途），本文只按位域声明记录，未深究硬件语义。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `/ssd/mirage_mpk_build`）。
- **支撑层三支**（`include/mirage/persistent_kernel/tasks/blackwell/`）：
  - `linear_fp8_sm100_utils.cuh`（143 行）：断言宏 `:10-33`、`PatternVisitor :37-47`、
    `ceil_div :49-52`、`constexpr_ceil_div :54-57`、`align :59-62`、`constexpr_align :64-67`、
    `constexpr_gcd :69-72`、`swap :74-79`、`get_sm_idx :81-85`、`get_lane_idx :87-91`、
    `ld_shared :93-99`、`st_shared :101-105`/`:107-113`、`cast_into_bf16_and_pack :115-120`、
    `add_packed_bf16x2_into_fp32_bits :122-141`。
  - `linear_fp8_sm100_sm100_utils.cuh`（232 行）：`make_smem_desc :13-28`、`make_sf_desc :30-34`、
    `replace_smem_desc_addr :36-41`、`get_atom_base :43-46`、`to_umma_layout_type :48-79`、
    `get_umma_desc_stride_k :81-89`、`advance_umma_desc_lo :91-103`、`make_umma_desc :105-145`、
    `make_runtime_instr_desc_with_sf_id :147-154`、`get_num_aligned_tmem_cols :156-172`、
    `tcgen05_before/after_thread_sync :174-180`、`SM100_MMA_MXF8F6F4_SS :182-205`、
    `SM100_MMA_MXF8F6F4_2x1SM_SS :207-230`。
  - `linear_fp8_sm100_tma_utils.cuh`（118 行）：`get_inner_block_atom_size :11-14`、
    `tma_copy :16-91`、`tensor_map_release_cta :93-95`、`tensor_map_acquire_cta :97-104`、
    `tensor_map_replace_global_addr_in_smem :106-116`。
  - `linear_fp8_sm100_common.cuh`（聚合 include `:5-8`）；另一支 `sm100_utils.cuh`（152 行，
    K-major 子集，`make_smem_desc :33-57`、`make_sf_desc :59-66`、`advance_umma_desc_lo :79-93`、
    `make_umma_desc :95-115`、`make_runtime_instr_desc_with_sf_id :116-126`、
    `SM100_MMA_MXF8F6F4_SS :127-152`）。
- **消费者**：身体 A `linear_fp8_sm100.cuh`（`PatternVisitor` 10 处、`tma_copy` 5 处、
  `make_instr_desc_block_scaled` 调用 `:408-416`、MMA 结构体 `:486-487`）；身体 B
  `linear_fp8_1d2d_sm100.cuh:41`；MoE `fp8_group_gemm_sm100.cuh:18`。
- **CUTLASS**（`deps/cutlass` `f3fde583`）：`include/cute/arch/mma_sm100_desc.hpp`
  （`SmemDescriptor :98-123`、`InstrDescriptor :412-439`、`InstrDescriptorBlockScaled :441-468`、
  `make_instr_desc_block_scaled :552-586`、`make_runtime_instr_desc_block_scaled :588-648`）。
- **本专题实验**：`src/fp8_linear_support.cu` + `harness/fp8_linear_support.sh` +
  `harness/fp8_linear_support.out.txt`（目标卡 H100/sm90a、nvcc 13.2、CUDA_VISIBLE_DEVICES=3、
  host 侧位探针 + grep audit，未 launch kernel）。
- 相关篇：`docs/32`（本文补其 §2.6 摘要）、`docs/57`（本文补其 §8 位编码尾巴）、`docs/59`/`docs/60`
  （UTCCP / scale-factor TMEM trait、`make_sf_desc` 的 SF 契约）、`docs/64`（本文补其 §12 尾巴）、
  `docs/33`（MoE 走 `fp8_group_gemm_sm100.cuh`）、`docs/07`（Hopper 单 task device 流水）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （device task 内部实现质量，同 `docs/28`–`docs/67`）。
- 交叉专题：DeepGEMM MoE 前向/反向正确性验证见 `../mega-moe/`（本专题不重复）。
