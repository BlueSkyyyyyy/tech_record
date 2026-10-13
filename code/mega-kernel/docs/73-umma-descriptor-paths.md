# UMMA SMEM 描述符构造的多路并置：DeepGEMM 自建母本 vs MPK 两份派生 vs MPK 手写 `make_desc`

> 快照与坐标：MPK = `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（完整 checkout
> `/ssd/mirage_mpk_build`）；CUTLASS = 其 submodule `deps/cutlass` commit `f3fde583`；DeepGEMM =
> `~/github/DeepGEMM` commit `057ca59`（`Public release 26/09/30`）。本文补 `docs/72`（P158-1）
> §7 明留、且 ROADMAP「下一步候选」第一条点名的**「描述符构造并置」**——`docs/69`（P154-1）只走了
> MPK 侧两份副本、`docs/72` 只覆盖了 `tcgen05.mma` **指令串**层，从无一篇把「**SMEM 描述符怎么算出来**」
> 的四条路径（DeepGEMM 自建母本 / MPK 全量副本 / MPK K-major 子集 / MPK 手写在 `sm100_ptx` 的裸描述符）
> 摆在一起。**纯源码走读 + 1 个 host 侧描述符算术探针**：SM100 的 `tcgen05`/TMEM 本体在目标卡
> H100/sm90 不可执行，但 `cute::UMMA::SmemDescriptor` 是 union+bitfield（`CUTE_HOST_DEVICE`），
> 各层 `make_smem_desc`/`make_umma_desc`/`advance_umma_desc_lo` 是 `__device__`，故其**构造算术**
> 在 host 上可逐字重放、逐位对比（`src/umma_desc_paths.cu` + `harness/umma_desc_paths.sh` +
> `harness/umma_desc_paths.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）UMMA 的 SMEM 描述符层是一条「母本 → 全量副本 → K-major 子集 → 手写裸描述符」的
抄写链，不是一个共享库。** 母本是 **DeepGEMM 自建** `deep_gemm/mma/sm100.cuh`（160 行，10 个 SM100
impl 共享 include）；MPK 的 **全量副本** `linear_fp8_sm100_sm100_utils.cuh`（232 行）几乎逐行相同
（含 `to_umma_layout_type`/`get_atom_base`/MN-major 分支/BASE32）；MPK 的 **K-major 子集**
`sm100_utils.cuh`（152 行）自陈 *"adapted from the DeepGEMM SM100 Blackwell path"*（`:26-28`）、
用两行 `static_assert("Current Mirage helper only supports K-major path.")`（`:81-82`/`:96-97`）
砍掉 MN-major；MPK 的 **手写裸描述符** `sm100_ptx.cuh::make_desc`（`:63-67`）连通用构造都绕开，
直接把 SBO=8×128、`version=1`、`layout=SWIZZLE_128B` 焊死成一个 64 位常数表达式。host 探针证明：
对生产实际用到的 fp8/bf16（pack=1）配置，**四路算出的 64 位描述符逐位相同**（§A/§B/§C/§F）。

**(B)（headline）四路唯一实质性分歧是「sub-byte packed 类型」——母本支持、MPK 三份都不支持。**
DeepGEMM 母本对 `float_e2m1_t`（FP4，两个逻辑元素压一字节）在 `advance_umma_desc_lo`
（`:89-96`）与 `make_umma_desc` K-major 分支（`:98-142`）里三处除以 `get_smem_pack_factor`（=2）；
MPK 的全量副本与子集**完全没有任何 pack factor**（`_sm100_utils.cuh:91-145`、`sm100_utils.cuh:79-115`）。
host 探针 §E：同一 K-major FP4 SW128 配置，母本得 `sbo=32`、MPK 得 `sbo=64`（描述符 `DIFF`）。
即 MPK 抄过来时**只保留了 FP8/BF16 路径**——这与 `docs/72` 的能力包含链（DeepGEMM 自建 11 opcode ⊃
MPK 4 opcode、MPK 无 `mxf4nvf4`）在**描述符算术层**再次吻合：**没抄那类指令，也就没抄那类指令的描述符**。

**(C)（headline）MPK 手写的 `make_desc` 不是「另一套算术」，而是通用 K-major f16 SW128 描述符的
「编译期常量特化」。** `sm100_ptx::make_desc`（`sm100_ptx.cuh:63-67`）把
`desc_enc(x)=(x&0x3FFFF)>>4`、`SBO=8*128=1024`、`version=1<<46`、`layout=2<<61` 拼起来；host 探针
§F 证明它与 `make_umma_desc<Major::K, BLOCK_MN, BLOCK_K=64, swizzle=128, half_t>`（f16：128 B swizzle
÷2 B/元素 = BLOCK_K 64）**逐位相同**（`0x4000404000000040`），与 MPK 子集公式也相同。它之所以「手写」，
是因为 MLA decode/MTP 只需要**一种**固定形态（f16、128 B swizzle、无 pack），把它折成常数比走模板少一层；
但这份「折成常数」的写法在 MPK 里**被抄了四遍**（`sm100_ptx.cuh:63` + 三个 TP 兄弟各自的私有 `ptx`
命名空间 `mla_mtp_decode_tp{2,4,8}_sm100.cuh:103/92/89`，逐字相同），是 `docs/67` §8「PTX 助手复制四份」
在描述符层的同型现象。

---

## 1. 定位：本文补哪一层、与谁分工

| 篇 | 覆盖 |
|---|---|
| `docs/57`（P143-1） | MPK `tasks/blackwell/` 的**共享原语层**（`sm100_ptx`/`sm100_utils`/`storage`/`tensor_init`），headline「原语实现不止一遍」|
| `docs/69`（P154-1） | MPK **两份副本**（`_sm100_utils` 全量 vs `sm100_utils` 子集）逐行 + `make_smem_desc` **位域编码** + `InstrDescriptorBlockScaled` **指令描述符** host 重放 |
| `docs/72`（P158-1） | `tcgen05.mma` **指令串**的多路并置（CUTLASS 母本 ⊃ DeepGEMM 自建 ⊃ MPK，40→11→4 opcode）|
| **本文** | **SMEM 描述符的构造算术**：把 **DeepGEMM 自建母本**拉进来，与 MPK 全量/子集/**手写**四路并置、host 逐位对比 |

`docs/69` 已经把 MPK 两份副本走透，`docs/72` 已经把 `make_smem_desc` 的位域与 `mxf8f6f4` 指令串
钉死；本文**不重复**这两篇的结论，只补三件 `docs/69`/`docs/72` 未做的事：**(1)** 引入 **DeepGEMM
自建母本** `deep_gemm/mma/sm100.cuh`（`docs/72` 只走 `ptx/tcgen05.cuh`，从未走 `mma/sm100.cuh`）；
**(2)** 把「**描述符构造**」的每个 helper（`to_umma_layout_type`/`get_atom_base`/`get_umma_desc_stride_k`/
`advance_umma_desc_lo`/`make_umma_desc`）跨四路**逐函数对照**（`docs/69` 只对了 MPK 内部两份）；
**(3)** 把 **手写 `make_desc`** 与通用路径**逐位**对上（`docs/72` §7 明说未做）。

---

## 2. 四路 helper 存在性（`harness/umma_desc_paths.out.txt`）

| helper | DeepGEMM 母本 `mma/sm100.cuh` | MPK 全量 `_sm100_utils.cuh` | MPK 子集 `sm100_utils.cuh` |
|---|---|---|---|
| `make_smem_desc` | `:15` | `:14` | `:33` |
| `make_sf_desc` | `:43` | `:31` | `:59`（多一段注释）|
| `replace_smem_desc_addr` | `:52` | `:37` | `:69` |
| `get_atom_base` | `:58` | `:44` | — |
| `to_umma_layout_type` | `:65` | `:52` | — |
| `get_umma_desc_stride_k` | `:85` | `:85` | — |
| `advance_umma_desc_lo` | `:91` | `:95` | `:79` |
| `make_umma_desc` | `:101` | `:112` | `:95` |
| `make_runtime_instr_desc_with_sf_id` | `:144` | `:147` | `:116` |
| `update_instr_desc_with_umma_n` | `:150` | — | — |

母本共 160 行、被 **10 个 SM100 impl 共享 include**（`sm100_bf16_gemm` / `sm100_bf16_mega_gate` /
`sm100_bf16_mega_moe` / `sm100_bmk_bnk_mn` / `sm100_fp8_fp4_gemm_1d1d` / `sm100_fp8_fp4_mega_moe` /
`sm100_mega_mhc` / `sm100_mqa_logits` / `sm100_sparse_mqa_logits` / `sm100_tf32_hc_prenorm_gemm`，
见 `harness` 末段）——即 **DeepGEMM 的 mega 家族与 SM100 GEMM 全用这一份描述符层**。

---

## 3. [母本] DeepGEMM `deep_gemm/mma/sm100.cuh` 逐函数

`namespace deep_gemm::mma::sm100`（`:11`）。头 include `cute/atom/mma_traits_sm100.hpp` /
`cute/arch/mma_sm100_umma.hpp` 与自己的 `common/{exception,math,packing,tma_copy}.cuh`。

### 3.1 `make_smem_desc`（`:14-40`）——与 MPK 副本逐行相同

```cpp
cute::UMMA::SmemDescriptor desc;
desc.version_ = 1;                    // SM100
desc.lbo_mode_ = 0;                   // legacy mode
desc.layout_type_ = (uint8_t)layout;
desc.start_address_ = (uint16_t)(cast_smem_ptr_to_uint(smem_ptr) >> 4);  // 16B 粒度
desc.base_offset_ = 0;
desc.stride_byte_offset_  = stride_byte_offset  >> 4;
desc.leading_byte_offset_ = leading_byte_offset >> 4;
```

与 MPK 全量副本 `_sm100_utils.cuh:13-28`、子集 `sm100_utils.cuh:33-57` **逐字段相同**（host §A：
三份对 `SWIZZLE_128B sbo=1024 lbo=0` 都给出 `0x4000404000000040`）。这就是 `docs/69` §3.1 已钉死的
位编码（三个偏移 `>>4`、`version_` bit[46,48)、`lbo_mode_` bit[52]、`layout_type_` bit[61,64)）的**出处**。

### 3.2 `make_sf_desc` / `replace_smem_desc_addr`（`:42-55`）

`make_sf_desc`（`:42-49`）= `make_smem_desc(SWIZZLE_NONE, smem_ptr, 8*16, 0)`，注释解释「UTCCP 布局
默认 K-major、原子 8×128 bit、LBO 可零」——MPK 两份副本同值（`docs/59`/`docs/60` 的 SF 契约）。
`replace_smem_desc_addr`（`:51-55`）只改 `start_address_`。

### 3.3 `get_atom_base` / `to_umma_layout_type`（`:57-81`）——MPK 全量副本抄了、子集没有

`get_atom_base`（`:57-60`）= `SWIZZLE_128B_BASE32B ? 32 : 16`。`to_umma_layout_type<Major,kSwizzle,
kUseBase32,dtype>`（`:64-81`）把「swizzle 字节数」映射到 `UMMA::LayoutType`，含**两条强制 BASE32 规则**
（`float + MN-major`、或 `kUseBase32` 显式置位）。MPK 全量副本 `:43-79` 与它**逐分支相同**；MPK
**子集无此函数**（直接 `constexpr auto layout_type = SWIZZLE_128B;` 硬编码，`sm100_utils.cuh:101`）。

### 3.4 `get_umma_desc_stride_k` / `advance_umma_desc_lo`（`:83-96`）——**pack factor 分叉处**

```cpp
// :83-87  DeepGEMM 母本: 4 个模板参, 显式带 kPackFactor
template <Major kMajorMode, uint32_t BLOCK_MN, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_umma_desc_stride_k() {
  return kMajorMode == Major::K ? 1
       : tma::get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, /*kPackFactor*/1, dtype_t>();
}
// :89-96  母本显式除以 kPackFactor
uint32_t advance_umma_desc_lo(const uint32_t& base, const uint32_t& offset, const uint32_t& k_idx) {
  constexpr uint32_t kPackFactor = get_smem_pack_factor<dtype_t>();
  const uint32_t byte_offset =
      (offset + k_idx * get_umma_desc_stride_k<...>()) * sizeof(dtype_t) / kPackFactor;
  return base + (byte_offset >> 4u);
}
```

三处差异溯源：**(a)** 母本调 `deep_gemm::tma::get_inner_block_atom_size<BLOCK_MN,kSwizzleMode,1,dtype>`
（**4 个模板参**，`common/tma_copy.cuh:12-15`：`kSwizzleMode==0 ? BLOCK_INNER/kPackFactor :
kSwizzleMode/sizeof(dtype)`）；MPK 全量副本把它改名成局部的 `get_inner_block_atom_size<BLOCK_MN,
kSwizzleMode,dtype>`（**3 个模板参**，`_tma_utils.cuh:11-14`，**丢掉 kPackFactor**）。**(b)** 母本
`advance_umma_desc_lo` 有 `/kPackFactor`；MPK 两份都没有。**(c)** `get_smem_pack_factor`
（`common/packing.cuh:8-12`）= `float_e2m1_t ? 2 : 1`——这是 FP4 专用，MPK 没抄。

### 3.5 `make_umma_desc`（`:98-142`）——K-major / MN-major 两支

K-major 支（`:108-120`）：`stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(dtype) /
kPackFactor`、`leading = 0`、地址 `byte_ptr = base + (mn*BLOCK_K + k*stride_k)*sizeof/kPackFactor`。
MN-major 支（`:121-141`）：`BLOCK_MN_ATOM = tma::get_inner_block_atom_size<BLOCK_MN,kSwizzleMode,1,
dtype>`、`stride = num_non_contiguous*BLOCK_MN_ATOM*sizeof`、`leading = BLOCK_K*BLOCK_MN_ATOM*sizeof`、
`kSwizzleMode==16` 时对调（`math::swap`）。MPK 全量副本 `:105-145` **只差 pack factor**（K-major 支
与 `tma::`→局部、`math::swap`→`swap`）；MPK 子集 `:95-114` 只有 **K-major**、且 `stride_byte_offset`
硬编码 `8*BLOCK_K*sizeof`。

### 3.6 `make_runtime_instr_desc_with_sf_id` / `update_instr_desc_with_umma_n`（`:144-158`）

`make_runtime_instr_desc_with_sf_id`（`:144-148`）= 填 `a_sf_id_/b_sf_id_` 后 `<<32`——与 MPK 两份
副本逐字相同（`docs/69` §3.4 已 host 核对与 CUTLASS constexpr 逐位 MATCH）。`update_instr_desc_with_umma_n`
（`:150-158`，两个重载改 `n_dim_ = umma_n>>3`）**MPK 两份副本都没有**——DeepGEMM 用它做「运行期改 N」
的动态 tile（MPK 生产用编译期 `BLOCK_N`，不需要）。

---

## 4. [全量]/[子集] MPK 两份副本的派生关系

**母本 → MPK 全量副本**（`linear_fp8_sm100_sm100_utils.cuh`）的**唯一实质差异**：删掉全部 sub-byte
pack factor（`get_umma_desc_stride_k` 少一个模板参、`advance_umma_desc_lo`/`make_umma_desc` 少除法）、
`tma::` → 局部函数名、`math::swap` → `swap`、少 `update_instr_desc_with_umma_n`；并**多出**
`get_num_aligned_tmem_cols`（`:156-172`，TMEM 列数 round 到 2 的幂，`docs/59` §8）与
`tcgen05_before/after_thread_sync`（`:174-180`）——后两者是 MPK 为「常驻 kernel 内 TMEM 生命周期」
补的（DeepGEMM 把它们放在别处）。

**MPK 全量 → 子集**（`sm100_utils.cuh`）的差异（`docs/69` §5 已列表，此处只补「相对母本」的坐标）：
子集 = 全量去掉 `to_umma_layout_type`/`get_atom_base`/`get_umma_desc_stride_k`/MN-major 分支/
`get_num_aligned_tmem_cols`/`tcgen05_*`/2SM 结构体，且用两行 `static_assert("Current Mirage helper
only supports K-major path.")`（`:81-82`/`:96-97`）把 K-major 锁死、`layout_type` 硬编码
`SWIZZLE_128B`（`:101`）。即**子集是从母本再砍一刀（连 MN-major 都不要）的结果**。

---

## 5. [手写] MPK `sm100_ptx::make_desc`：把通用构造成常数

```cpp
// sm100_ptx.cuh:59-67
__device__ constexpr uint64_t desc_enc(uint64_t x) { return (x & 0x3FFFFULL) >> 4; }
__device__ uint64_t make_desc(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) | (2ULL << 61);
}
```

它把「K-major、SWIZZLE_128B、LBO=0、version=1、base_offset=0、lbo_mode=0」全焊成常数，只留
`start_address`（`(smem_addr & 0x3FFFF) >> 4`）随 SMEM 地址变——正是
`make_umma_desc<Major::K, *, BLOCK_K, 128, *, dtype>` 在 f16（`128 B swizzle ÷ 2 B/元素 = BLOCK_K
64`）时的编译期常量特化。**host 探针 §F 逐位证实**：`make_desc(0x400) = 0x4000404000000040`
== `dg_make_umma_desc<Major::K,128,64,128,half_t>(0x400,0,0)`
== `sub_make_umma_desc<128,64,128,half_t>(0x400,0,0)`。

**消费者与复制**：`sm100_ptx::make_desc` 被基版 MLA decode/MTP 用（`mla_decode_sm100.cuh:67`
`using namespace ::kernel::sm100_ptx` → `:195/196/383/384`；`mla_mtp_decode_sm100.cuh:93` → `:268/269/546/547`）；
但 TP 三兄弟**各在自己的私有 `namespace ptx` 里重抄一份**（`mla_mtp_decode_tp2_sm100.cuh:103-107`、
`tp4:92-96`、`tp8:89-93`，逐字相同，`using` 处 `tp2:326`/`tp4:312`/`tp8:305` 用 `ptx::make_desc`）。
即这份「常数描述符」在 MPK 里有 **4 份相同拷贝**（1 共享 + 3 私有），与 `docs/67` §8 记的
「MTP attention 一族 PTX 助手复制四份」同源。

---

## 6. host 逐位并置（`harness/umma_desc_paths.out.txt`）

**§A `make_smem_desc`**（`SWIZZLE_128B sbo=1024 lbo=0 ptr=0x400`）：DeepGEMM = MPK 全量 =
`0x4000404000000040`（`MATCH`）。

**§B `make_umma_desc` K-major fp8**（`BLOCK_MN=128 BLOCK_K=128 swizzle=128 mn=1 k=0`）：
DeepGEMM = MPK 全量 = MPK 子集 = `0x4000404000000048`（`start=72`、`sbo=64`、`layout=2`，
`MATCH`，三方各 `MATCH`）。即生产配置下**子集的硬编码 `8*BLOCK_K*sizeof` 与母本的
`num_non_contiguous*BLOCK_K*sizeof` 同值**（fp8 SW128：`128/16*128 = 8*128 = 1024`）。

**§C `make_umma_desc` MN-major bf16**（`BLOCK_MN=64 BLOCK_K=64 swizzle=64`）：DeepGEMM = MPK 全量
= `0x8000402001000040`（`lbo=256 sbo=32 layout=4`=SWIZZLE_64B，`MATCH`）；`k=1` 时 `start` +4
（`stride_k=BLOCK_MN_ATOM=32` 元素 × 2 B ÷ 16）。**MPK 子集不参与**（无 MN-major 分支）。

**§D `advance_umma_desc_lo` 步进**（K-major fp8、`base=0x40 offset=0 k=0..3`）：DeepGEMM/全量/子集
**逐点一致**（`k_idx 0..3` 全是 `64`——因为 K-major 的 `stride_k=1`、fp8 每元素 1 B、除以 16 后
每步 `(1*1)>>4=0` 不足 16 B 不动）。

**§E pack-factor 分叉**（K-major **FP4** e2m1、`BLOCK_MN=128 BLOCK_K=128 swizzle=128`）：母本
（÷pack=2）`0x4000402000000040`（`sbo=32`）vs MPK（不除）`0x4000404000000040`（`sbo=64`）→
**`DIFF`**。这是四路唯一的实质分歧，且**只在 sub-byte packed 类型上出现**——生产 fp8/bf16（pack=1）
无差异。

**§F 手写 vs 通用**（K-major f16 `BLOCK_K=64 swizzle=128`）：`hand` = 通用 DeepGEMM = 通用 MPK 子集
= `0x4000404000000040`（两个 `MATCH`）；直接解手写描述符得 `start=64 sbo=64 ver=1 layout=2`，
与 `0x400>>4=64`、`1024>>4=64` 吻合。

---

## 7. 对 megakernel 的含义

- **对 `docs/72` 的补强**：`docs/72` 把**指令层**数成「CUTLASS ⊃ DeepGEMM 自建 ⊃ MPK」的包含链；
  本文在**描述符层**得到同构结论——**母本是 DeepGEMM 自建 `mma/sm100.cuh`，MPK 的两份副本 + 手写
  常数是它的逐级裁剪**。即「生产实现各自只抄自己用到的形态」不只发生在 `tcgen05.mma` 指令上，也发生
  在「描述符怎么拼」这件更基础的事上——**连描述符构造都不是共享库，而是团队复制**。这与 `docs/22`
  §5「同一套惯用法实现两遍」的母题在**最底层的编码算术**处再次显现。
- **对 `docs/69` 的补强**：`docs/69` 只对 MPK 内部两份副本 + 位域；本文给出**母本坐标**
  （`deep_gemm/mma/sm100.cuh`，10 个 SM100 impl 共享）与**跨库逐函数差异表**（§2/§3/§4），并指出
  MPK 抄写链丢掉的是 **FP4/sub-byte pack 支持**——与 `docs/72` 的「MPK 无 `mxf4nvf4` opcode」
  在算术层互为表里。
- **对「少而重 vs 多而杂」设计选择的呼应**：手写 `make_desc`（§5）说明，当一个 task 类型只需**一种**
  固定描述符形态时，把它折成常数是**比模板更轻**的封装——这解释了为什么 DeepGEMM 的**极重**段级
  mega-kernel（Mega MoE/Gate/mHC）走通用 `make_umma_desc`（要 K/MN-major、多 swizzle），而 MPK 的
  **薄** attention task 只留常数版。与 `docs/02`（DeepGEMM 段级）/`docs/07`（task 粒度）的选择一致。
- **判据映射**：本文只贡献 **C3（多算子范围 / 设备体内指令与操作数契约）**。它讲的是「一个 task 的
  device body 发 UMMA 前怎么拼 SMEM 描述符、那份拼法来自哪份拷贝」，不涉及 launch/常驻（C1/C2）、
  不涉及设备内调度（C4/C5）。与 `docs/57`/`docs/69`/`docs/72` 同层。

---

## 8. 局限（诚实标注）

- **纯 host 算术重放，SM100 本体未执行**：本机（目标卡 H100/sm90）无 `tcgen05`/TMEM；本文的描述符
  值是对真实 CUTLASS 头（`f3fde583`）的 bitfield + 各层源码**同一算术**在 host 上重放得到的，属
  「源码规定的编码」，不是 SM100 上观测到的硬件解析行为。
- **重放的是源码算术，不是其 `__device__` 函数的直接执行**：各 `make_smem_desc`/`make_umma_desc` 是
  `__device__`（用 `cute::cast_smem_ptr_to_uint`，host 不可调），探针按**同 bitfield 赋值 + 同公式**
  复现；「编码算术正确」≠「MPK 在 SM100 上跑出来就是这个值」。
- **`mpk_make_umma_desc` 的 host 重放里复用了 `dg_atom_base`**（与 MPK 自身 `get_atom_base` 逐字段相同），
  不改变数值；已逐分支核对。
- **pack-factor 分叉是静态事实、非 bug 判定**：MPK 不抄 FP4 pack factor 可能只是因为**它不在 MPK 用**，
  不是因为 DeepGEMM 那条路径有问题；本文只记「有/无」与由此产生的描述符位差，不判意图。
- **「谁是母本」按公开发布顺序与 MPK 自陈注释**（`sm100_utils.cuh:26-28` *"adapted from the DeepGEMM
  SM100 Blackwell path"*）+ DeepGEMM 早于 MPK 该 commit 公开（`057ca59`）；不排除两者共享更早的内部
  版本，但证据指向 DeepGEMM 侧。
- **consumer audit 是 `grep` 级**：`make_desc` 的 4 份拷贝计数受宏/命名空间影响，本文只记静态事实；
  运行期是否真走到某分支未在编译期验证。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` @ `f9eb70c`（本地 `/ssd/mirage_mpk_build`）；`deps/cutlass` @ `f3fde583`；
  DeepGEMM @ `057ca59`（`~/github/DeepGEMM`，容器内 `~/github/DeepGEMM`）。
- **DeepGEMM 母本**（`deep_gemm/include/deep_gemm/mma/sm100.cuh`，160 行）：`make_smem_desc:14-40`、
  `make_sf_desc:42-49`、`replace_smem_desc_addr:51-55`、`get_atom_base:57-60`、
  `to_umma_layout_type:64-81`、`get_umma_desc_stride_k:83-87`、`advance_umma_desc_lo:89-96`、
  `make_umma_desc:98-142`、`make_runtime_instr_desc_with_sf_id:144-148`、
  `update_instr_desc_with_umma_n:150-158`；支持层 `common/tma_copy.cuh:12-15`（`get_inner_block_atom_size`）、
  `common/packing.cuh:8-12`（`get_smem_pack_factor`）；10 个 SM100 impl 共享 include（§2 清单）。
- **MPK 全量副本**（`tasks/blackwell/linear_fp8_sm100_sm100_utils.cuh`，232 行）：`make_smem_desc:13-28`、
  `make_sf_desc:30-34`、`replace_smem_desc_addr:36-41`、`get_atom_base:43-46`、`to_umma_layout_type:48-79`、
  `get_umma_desc_stride_k:81-89`、`advance_umma_desc_lo:91-103`、`make_umma_desc:105-145`、
  `make_runtime_instr_desc_with_sf_id:147-154`、`get_num_aligned_tmem_cols:156-172`、
  `tcgen05_before/after_thread_sync:174-180`；`get_inner_block_atom_size` 在
  `linear_fp8_sm100_tma_utils.cuh:11-14`。
- **MPK K-major 子集**（`tasks/blackwell/sm100_utils.cuh`，152 行，自陈 adapted from DeepGEMM `:26-28`）：
  `make_smem_desc:33-57`、`make_sf_desc:59-66`、`replace_smem_desc_addr:69-73`、
  `advance_umma_desc_lo:79-86`（K-only `static_assert:81-82`）、`make_umma_desc:95-114`
  （K-only `static_assert:96-97`、硬编码 `SWIZZLE_128B:101`）、`make_runtime_instr_desc_with_sf_id:116-123`。
- **MPK 手写 `make_desc`**（`tasks/blackwell/sm100_ptx.cuh`，88 行）：`desc_enc:59-61`、`make_desc:63-67`；
  共享消费者 `mla_decode_sm100.cuh:49/67/195-196/383-384`、`mla_mtp_decode_sm100.cuh:66/93/268-269/546-547`；
  私有副本 `mla_mtp_decode_tp2_sm100.cuh:103-107`、`tp4:92-96`、`tp8:89-93`（`using` `:326/312/305`）。
- CUTLASS 位域：`deps/cutlass/include/cute/arch/mma_sm100_desc.hpp`——`SmemDescriptor:98-123`
  （`version_ [46,48)`、`base_offset_ [49,52)`、`lbo_mode_ [52)`、`layout_type_ [61,64)`）、
  `LayoutType:79-85`（`SWIZZLE_NONE=0/BASE32B=1/128B=2/64B=4/32B=6`）、`InstrDescriptorBlockScaled:441-468`。
- 本文实验：`src/umma_desc_paths.cu` + `harness/umma_desc_paths.sh` +
  `harness/umma_desc_paths.out.txt`（目标卡 H100/sm90a、nvcc 13.2、`CUDA_VISIBLE_DEVICES=3`，
  host 侧描述符算术重放 + grep audit，未 launch kernel；SM100 本体不可执行）。
- 前置/相关走读：`docs/57`（MPK SM100 原语层）、`docs/59`/`docs/60`（TMEM/UTCCP/scale trait）、
  `docs/67`（MLA-MTP TP 三兄弟的私有 `ptx` 副本）、`docs/69`（MPK 两份副本 + 位编码）、
  `docs/72`（UMMA 指令多实现并置）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及描述符层**
  （device task 内部实现质量，同 `docs/28`–`docs/72`）。
- 交叉专题：DeepGEMM MoE 前向/反向正确性验证见 `../mega-moe/`（本专题不重复）。
