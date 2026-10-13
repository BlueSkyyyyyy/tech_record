# SM100 block-scaled scale-factor 的 TMEM 布局 trait 走读：CUTLASS `tmem_sf_frg` ↔ MPK `sf_id`/UTCCP 消费层

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/59` 同快照）；
> CUTLASS 侧走的是 MPK 该快照钉死的 submodule `deps/cutlass`，commit `f3fde583`
> （`/ssd/mirage_mpk_build/deps/cutlass`，这条 checkout 有完整 submodule；MPK 的
> `include/mirage/persistent_kernel/tasks/blackwell/sm100_utils.cuh:20` 等直接
> `#include <cute/atom/mma_traits_sm100.hpp>`）。
> `docs/51`/`docs/58` 走 SMEM 软件/TMA 布局、`docs/59` 走 TMEM 列与 UTCCP 重排；本文补上
> `docs/59` §8 **明确留的最直接下一步**——**CUTLASS 侧 `mma_traits_sm100.hpp` 里 block-scaled
> scale-factor 的 TMEM 布局 trait**，以及 MPK 手工路径如何把同一个 scale 槽位填出来。
> **纯源码走读 + 1 个 host 侧微实验**：SM100 指令/TMEM/UTCCP 本体在本机（目标卡 H100/sm90）不可执行，
> 但 `cute::UMMA::tmem_sf_frg`/`tmem_frg` 是 `CUTE_HOST_DEVICE`，整套布局代数可在 **host** 上实例化/
> 求值——本文用快照钉死的 **真实 CUTLASS 头文件**把 trait 的形状钉死（`src/sf_tmem_trait.cu` +
> `harness/sf_tmem_trait.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）`tmem_sf_frg` 把「scale 因子在 TMEM 里长什么样」写成一个 **CuTe Layout**，本文用
真实 CUTLASS 头文件在 host 上把它求出来了。** 对 `float, VS=32, 1SM, SFA`、M=128、K 分 4 个子块，
实测映射为：**逻辑 M 行 `r` → (dp = `r mod 32`, col = `4 * (r div 32)`)**；每个 K 子块把列 **+1**；
`MMA_VS`（一个 scale 覆盖的 32 个值）**stride 0**（完全复制）；`REP=_4` 把同一份数据**复制到 4 个子分区**
（`dp += 32*rep`）。这与 `docs/59` 讲的「UTCCP 一次搬 4 列 × 32 dp」是同一段 TMEM 列空间的两种描述。

**(B)（headline）`sf_id` 是这次布局在**指令描述符**里的「地址高位」——CUTLASS 直接
`(tmem_sf_addr & 0xC0000000) >> 30` 取，MPK 则**显式按 K 子块号填**。** 这是本文最实际的一条：
同一个 `tcgen05.mma...block_scale` 指令，CUTLASS 高层路径假设 SF 地址的高 2 位编码了 id，
而 MPK 手工路径把纯列偏移当地址、用 `make_runtime_instr_desc_with_sf_id` 显式写 `a_sf_id_`/`b_sf_id_`。
两条路径填的是 `InstrDescriptorBlockScaled` 的同两个位域，但**来源完全不同**。

**(C)（headline）`docs/59` §8 点名的「CuTe tie 层」其实是一个 60 行、**全仓零调用点**的宏工具。**
`linear_fp8_sm100_cute_tie.cuh`（注意：`docs/59` §8 写成 `linear_fp8_1d2d_sm100_cute_tie.cuh`，快照里
**没有这个名字**）定义了 `cute::ignore_t`/`cute::ignore` 与 `CUTE_TIE`/`CUTE_TIE_DECL` 宏，却被
`linear_fp8_sm100_utils.cuh:8` 无条件 `#include` 后**一次都不用**（全仓 `CUTE_TIE(` 只有定义处 1 处）。
这与本专题反复出现的「有实现未接线」族同向；它本意是「解构 CUTLASS 返回的 tuple、对 `cute::ignore`
占位符容错」，但这条 SM100 线性路径最终没有用它。

---

## 1. 定位：从「TMEM 列空间」到「trait 规定的列布局」

`docs/59` 讲清了 MPK 的 **累加器 + SFA + SFB 三段塞进同一段 TMEM 列空间**、靠**列偏移**区分
（`kTmemStartColOfSFA/SFB`），也讲了未 round 的 `// TODO`。但它停在「MPK 自己算列数」这一层，没回答
**「这些列里的数据按什么形状摆放」**——这正是 CUTLASS `mma_traits_sm100.hpp` 里一组 `tmem_*_frg`
**fragment 布局 trait** 负责的事。它们不是「某段代码怎么算」，而是**「TMEM 的列/数据通路（DP）该如何被
这个 MMA 指令解释」的类型级契约**——与 `docs/51`「布局类型即 SMEM 的 ABI」同构，只是换成 TMEM。

文件很短但密（`deps/cutlass/include/cute/atom/mma_traits_sm100.hpp`，4032 行里的 455–1016 一段）：

| trait | 行 | 职责 |
|---|---|---|
| `UMMA::TmemAllocMode` | `:426-455` | 5 种分配模式枚举（见 §2.1） |
| `UMMA::tmem_frg_base` | `:457` | 所有 TMEM fragment 的基类（tag） |
| `UMMA::tmem_frg` | `:461-583` | **累加器 / A 值**的 TMEM 布局（1SM/2SM） |
| `UMMA::tmem_e_frg` / `_ws` | `:593-669` / `:671-811` | 稀疏 MMA 的**元数据 E** 布局 |
| `UMMA::tmem_sf_frg` | `:813-873` | **block-scaled 的 scale-factor** 布局（本文主角） |
| `UMMA::tmem_frg_ws` | `:876-959` | weight-stationary MMA 的 C/D 布局 |
| `MakeTensor<...>` 定制点 | `:967-1016` | 把上述 trait 接到 `make_tensor` |

---

## 2. `tmem_frg`：累加器 / A 值如何落到 TMEM

### 2.1 `TmemAllocMode` 五模式（`:426-455`）

枚举值（`:426-455`）：`Interleaved=0`（默认，半子分区可**交织**上/下半子分区以占满容量）、
`NonInterleaved=1`（不交织，`UMMA_TS` 下 TMEM-A/C 地址等价性需要）、`Duplicated=2`
（2SM 下把分配**跨子分区复制**，2x2 DP 布局）、**`ScaleFactorDuplicated4by1=3`**、
**`ScaleFactorDuplicated2by2=4`**（后两个是 scale-factor 专用，见 §3.3）。后两者是本文关键：
scale-factor 的复制模式**独立于**数据矩阵的复制模式。

### 2.2 1SM / 2SM 的 atom（`:498-579`）

`tmem_frg::make` 先做一个「虚拟寻址」的 `tmem_restride`（DP stride、COL stride），把一个
`tmem_atom` 铺开成逻辑布局，再 `composition` 回「逻辑 TMEM 地址」（`:490-495`）。atom 随
`(N_SM, M_MMA, TmemAllocMode)` 分支：

- **1SM**：只接受 `Interleaved`/`NonInterleaved`（`:500`）。`M_MMA==64` 用半子分区 atom
  `Shape<Shape<_16,_4>, N>` / `Stride<Stride<_1,_32>, _128>`（`:507-508`），且 `tile_stride` 由
  I/NI 决定（`Interleaved→1`、`NonInterleaved→2`，`:510`）——**这是 I/NI 唯一的差别**（外层 tiling
  的步长）；`M_MMA==128` 用全子分区 `Shape<_128,N>` / `Stride<_1,_128>`（`:521-522`），I/NI 无差别。
- **2SM**：只接受 `Interleaved`/`Duplicated`（`:532`）；`M_MMA=32` →「1x4」、`M_MMA=64` →
  「2x2」或复制版、`M_MMA=128` →「4x1」（`:536-578`）。

### 2.3 实验 C 证据（`harness/sf_tmem_trait.out.txt`）

```
[acc M=128 N=128]  ((_128,_128),_1,_1):((_65536,_1),_0,_0)
[acc M=64  N=128 I ] (((_16,_4),_128),_1,_1):(((_65536,_2097152),_1),_0,_0)
[acc M=64  N=128 NI] (((_16,_4),_128),_1,_1):(((_65536,_2097152),_1),_0,_0)
```

- M=128：地址 = `dp*65536 + col`，**每个 N 元素一个列（col stride=1）**、**每行一个 DP（dp stride=65536）**。
  即 128×128 累加器占 **128 列**（正是 `docs/59` 说的「累加器段列数 = BLOCK_N（× epilogue stages）」）。
- M=64：M 轴拆成 `(16,4)`，stride `(65536, 2097152)`——`2097152 = 32<<16` 即「隔 32 个 DP 复制一份」，
  这就是**半子分区交织**；且如 §2.2 所预言，**单 tile 时 I 与 NI 打印完全一致**（差别只体现在外层
  tiling 的 `tile_stride`，本例只有 1 个 tile，故看不出）。

---

## 3. `tmem_sf_frg`：scale-factor 的 TMEM 布局

### 3.1 模板参数与商定契约（`:813-842`）

```
template <class ValueType, int SFVecSize, int N_SM, bool Is_SFA,
          UMMA::TmemAllocMode TmemAlloc = UMMA::TmemAllocMode::ScaleFactorDuplicated4by1>
struct tmem_sf_frg : tmem_frg_base
```

- `ValueType`：scale 的数据类型（本文用例是 UE8M0，`docs/59` 里是 `uint32` 存一行一个）；
- `SFVecSize`：**一个 scale 覆盖多少个值**，源码注释「Valid values are (16, 32)」（`:821-822`）；实际
  由指令 trait 给出：dense `K=32 → SFVecSize=32`（`mma_traits_sm100.hpp:2713-2714`），
  sparse `K=64 → 64`（`:2804-2805`）；
- `N_SM`、`Is_SFA`：A 还是 B；
- `TmemAlloc`：默认 4by1。

**输入的 `tmem_shape` 不是「实际分配大小」**，而是一个 **MMA-SF 分区形状**
`((MMA_MN,(VecSize,NSF)), num_MMA_MN, num_MMA_K, …)`，注释明写
「`size<0,1>(tmem_shape)` 给的是 K 方向的 MMA 元素数而不是 scale 数」（`:824-826`），并要求
`R>=3`、`rank(get<0,1>)==2`（`:841-842`）。

### 3.2 atom：REP 复制 + VS 复制（`:843-858`）

```
using REP = _4;                 // Replication factor（数据恒跨子分区复制）
constexpr int SUBPART_DPs = 32; // 一个子分区的 DP 数
…
auto tmem_atom = Layout<
  Shape< Shape< Shape<32, MMA_MN/32>, REP>,   Shape<MMA_VS, MMA_NSF> >,
  Stride<Stride<Stride< 1,        512>,  32>, Stride<    0,      128> > >;
```

三条一眼可见的语义：**(i)** `MMA_VS` 的 stride 是 **0**——一个 scale 值被它覆盖的 `VS` 个元素**共享**，
所以布局在这一维完全复制；**(ii)** `REP=4` 的 stride 是 `32`（即 32 个 DP）——同一份 scale 被复制到
4 个子分区；**(iii)** `MMA_NSF` stride `128`（虚拟列宽），即 **NSF 维每步进一列**。

### 3.3 SFA 恒 4by1，SFB 4by1 / 2by2（`:850-871`）

分支条件 `Is_SFA || (!Is_SFA && TmemAlloc == ScaleFactorDuplicated4by1)`（`:850`）——
**SFA 无论传什么 `TmemAlloc` 都走 4by1 分支**；只有 SFB 且显式 `ScaleFactorDuplicated2by2`
时走 2x2（`:860-871`，要求 `N_SM==2`）。2x2 的 atom 把 M 轴拆成
`Shape<Shape<32, MMA_MN/2/32>, _2, _2>` / `Stride<Stride<_1,_512>, _64, _32>`（`:865-866`）。

### 3.4 实验 A/B 证据（`harness/sf_tmem_trait.out.txt`）

实验 A 对 `float, VS=32, 1SM, SFA`、M=128、nK=4 逐点求 `(dp,col)`：

```
layout = ((((_32,_4),_4),(_32,_1)),_1,_4):((((_65536,_4),_2097152),(_0,_1)),_0,_1)
  r=  0 -> dp= 0 col= 0
  r= 32 -> dp= 0 col= 4
  r= 96 -> dp= 0 col=12
  r=127 -> dp=31 col=12
  K 子块 k0..k3 -> col 0,1,2,3          （每步 +1）
  REP rep0..rep3 -> dp 0,32,64,96        （跨 4 个子分区复制）
  MMA_VS vs0,vs1,vs31 -> 地址恒为 0      （stride 0，完全复制）
```

这与源码 strides 逐条吻合，且把「**M 行 `r` → col = `4*(r div 32)`**」这条（§7 会用到）钉死。
实验 B 验证变体：`VS=64`（sparse）形状 `((_128,(_64,_1)),_1,_2)`、2SM 2by2 的 atom 出现
`((_32,_2),_2,_2):((_65536,_4),_4194304,_2097152)`（M 轴 2x2 拆分，stride `{64,32}` 个 DP）。

---

## 4. `sf_id`：这个布局在指令描述符里的「地址高位」

scale 在 TMEM 里躺好了，但**指令怎么知道去哪一列读**？答案在指令描述符
`UMMA::InstrDescriptorBlockScaled`（`deps/cutlass/include/cute/arch/mma_sm100_desc.hpp:441-468`）：

```
a_sf_id_ : 2,  // bit [29,31) : Matrix A Scale Factor ID
b_sf_id_ : 2,  // bit [ 4, 6) : Matrix B Scale Factor ID
```

CUTLASS 的填充函数 `make_runtime_instr_desc_block_scaled`（`:588-621`，另有同语义重载 `:623-649`）
**从 TMEM 地址的高 2 位反推**（`:604-605` / `:631-632`）：

```cpp
// The first 2-bits of TMEM address includes byte address.
desc_i.a_sf_id_ = (tmem_sfa_addr & 0xC0000000) >> 30;
desc_i.b_sf_id_ = (tmem_sfb_addr & 0xC0000000) >> 30;
```

即 CUTLASS 假设 **SF 的 TMEM 地址高 2 位编码了 id**（`0xC0000000` 是 bit 30、31）。位域注释说
`a_sf_id_` 在 `[29,31)`、`b_sf_id_` 在 `[4,6)`——注意这里不是「直接拷贝位」，而是「取高 2 位当 2-bit
数值、再按位域落位」。最终 idesc 被 `<< 32` 放进 64 位指令描述符的高 32 位（`:618`）。

在 `MMA_Traits<SM100_MMA_MXF8F6F4_SS>` 里，SFA/SFB 的各自由
`UMMA::tmem_sf_frg<sf_type, SFVecSize, 1, true/false>` 给出（`:2719-2720`），`mma_unpack` 调
`make_runtime_instr_desc_block_scaled<>(traits.idesc_, traits.tsfa_addr_, traits.tsfb_addr_)`（`:2766`）
——**地址 → id → 指令**，闭环在 CUTLASS 内部完成。

---

## 5. MPK 手工路径：自己算列、自己填 id

MPK 的 SM100 线性/组 GEMM **不用** `SM100_MMA_MXF8F6F4_SS` 的 CUTLASS trait（`MMA_Atom`），而是
自备 `struct SM100_MMA_MXF8F6F4_SS` 裸 PTX 包装（`docs/57` §2 的「实现不止一遍」）。在本文语境下，
有两件事值得记：

### 5.1 「三份封装 + 两份 id 函数」又各来一份

- `kernel::sm100`（`tasks/blackwell/sm100_utils.cuh`，152 行，DeepGEMM 派生）：
  `make_sf_desc`（`:51-63`）、`make_runtime_instr_desc_with_sf_id`（`:116-123`）、
  `SM100_MMA_MXF8F6F4_SS::fma`（`:127-150`）；
- `mirage::blackwell::linear_fp8_sm100::sm100`（`tasks/blackwell/linear_fp8_sm100_sm100_utils.cuh`，
  232 行）：**同名同语义再来一遍**——`make_sf_desc`（`:30-34`）、
  `make_runtime_instr_desc_with_sf_id`（`:147-154`）、`SM100_MMA_MXF8F6F4_SS`（`:182-205`）外加
  **`SM100_MMA_MXF8F6F4_2x1SM_SS`**（`:207-230`，`cta_group::2`）；
- 再加 CUTLASS 的 `cute::` 模板——这就是 `docs/57` §2「两遍变三遍」在 SF 路径上的具体落点。

### 5.2 地址与 id 的分工相反

MPK 的 `make_runtime_instr_desc_with_sf_id`（两处）**不做任何掩码**，直接
`desc.a_sf_id_ = sfa_id; desc.b_sf_id_ = sfb_id;`——把 id 当**入参**；而地址操作数
`tmem_sfa`/`tmem_sfb` 传的是**纯列偏移**。看 `linear_fp8_sm100.cuh` 的实参（`:494-522`）：

```cpp
uint32_t const sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);  // :495-496
uint32_t const sfb_id = (kGranKB == 32 ? k : sfb_stage_in_group_idx);  // :497-498
auto const &runtime_instr_desc =
    make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);    // :499-500
…
mma_t::fma(a_desc, b_desc,
           accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,       // tmem_c
           k_block_idx > 0 || k > 0,                                   // scale_c(accumulate)
           runtime_instr_desc,
           kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),      // tmem_sfa :521
           kTmemStartColOfSFB);                                        // tmem_sfb :522
```

要点：**`sfa_id` 是 K 子块号 `k`（`UMMA_K=32`、`BLOCK_K=128` ⇒ `k∈[0,4)`），而不是从地址高位取的**；
`tmem_sfa` 传的是列偏移 `kTmemStartColOfSFA + w*4`（`128/32 = 4`，每 M-wave 走 4 列）。也就是说
**MPK 把「哪个 K 子块的 scale」放进指令的 id 位，把「M 波/段」放进地址列**——恰好与 CUTLASS 反过来
分配这两件事。

### 5.3 列预算与 UTCCP 目标列（与 `docs/59` 对齐）

- `kNumUTCCPAlignedElems = 128`（`:78`）、`SF_BLOCK_M = align(BLOCK_M,128)`（`:140-141`）、
  `kNumSFATmemCols = SF_BLOCK_M/32`、`kNumSFBTmemCols = SF_BLOCK_N/32`（`:159-160`）；
- `kNumTmemCols = get_num_aligned_tmem_cols<累加 + SFA + SFB>()`（`:167-169`），而
  `get_num_aligned_tmem_cols` 就是 **round 到 2 的幂、下限 32、上限 512** 的阶梯
  （`linear_fp8_sm100_sm100_utils.cuh:156-172`；实验 D 逐点验证：`33→64`、`129→256`、`257→512`），
  与 CUTLASS 分配器契约一致（`docs/59` §2）。**注意 `docs/59` §8 说 `linear_fp8_1d2d_sm100.cuh:118`
  没 round**；本文这一支（`linear_fp8_sm100.cuh`）**是 round 的**——「两份实现」在 round 这件事上也
  不一致；
- `kTmemStartColOfSFA = 累加列数`、`kTmemStartColOfSFB = 累加 + SFA 列数`（`:170-171`），
  UTCCP 把 scale 拷进 `kTmemStartColOfSFA + i*4` / `kTmemStartColOfSFB + i*4`（`:463-483`）。

### 5.4 一条重要事实：MPK 不实例化 `tmem_sf_frg`

全仓 `grep tmem_sf_frg` 在 MPK 源码里**零命中**（只在 `deps/cutlass` 里）。所以 §2/§3 的 trait
对 MPK 是**「类型与惯例的源头」而非调用点**——MPK 用「手算列偏移 + 显式 sf_id + 手写 warp 转置 +
UTCCP」把同一段 TMEM 列空间填出来。这正是本专题「SM100 同一件事被实现多遍」在 scale-factor 上的
一次具体展开。

---

## 6. 附：`linear_fp8_sm100_cute_tie.cuh` 是零调用点的宏工具

`docs/59` §8 把「CuTe tie 层」写成了 `linear_fp8_1d2d_sm100_cute_tie.cuh`——快照里**没有**这个文件名，
真实文件是 `tasks/blackwell/linear_fp8_sm100_cute_tie.cuh`（60 行）。它提供：

- `cute::ignore_t` / `cute::ignore`（`:5-14`）：一个「吞掉任何赋值」的占位符对象；
- `CUTE_TIE_DECL` / `CUTE_TIE` 宏族（`:16-60`）：把 `cute::get<I>(tuple)` 自动解构到若干变量，
  支持 1–5 个变量、用 `__LINE__` 做临时变量名、`CUTE_TIE` 把赋值包在 `do{…}while(0)` 里。

用途显然是「解构 CUTLASS 返回的 tuple、对 `cute::ignore` 占位符容错」（配合 `ignore_t::operator=`
吞掉不需要的返回值）。但它在快照里**只有定义**：`linear_fp8_sm100_utils.cuh:8` 无条件 `#include`
它，全仓 `CUTE_TIE(` 仅 1 处（定义处）、`cute::ignore` 仅 `cute_tie.cuh` 自身出现。故它是一个
**被传递 include 却零使用**的 utility——记入「有实现未接线」族，并顺带修正 `docs/59` §8 的文件名。

---

## 7. 两种「4 列足迹」约定的对照（只记不判）

本文有一处**不敢下判**的对照，如实记录：对 M=128 的 SFA，**二者都用 4 列足迹**，但 M 轴的列分配不同：

- **CUTLASS `tmem_sf_frg`**（实验 A 实测）：M 行 `r` → `col = 4 * (r div 32)`，即 4 组各占
  **col 0、4、8、12**（组间列 stride 4）；K 子块 → 列 stride **1**。
- **MPK UTCCP 路径**（`docs/59` §3 的 SMEM 重排参考 `out[l*4+k]=in[k*32+l]` + `kTmemStartColOfSFA+i*4`）：
  行 `r` → (`dp = r mod 32`, `col = r div 32`)，即 4 组各占 **col 0、1、2、3**（组间列 stride 1）；
  K 子块则由 `sf_id` 选，不再占列（§5.2）。

两者对「4 列」的**组间 stride**（4 vs 1）与「K 放哪」（列 vs id）都不同。**谁与
`tcgen05.mma...block_scale` 的硬件 TMEM SF 寻址契约一致，本机（H100/sm90）不可执行 SM100 指令、
无法核验**，故**只记不判**。要收敛此问，需在 SM100 上跑一个「同一组 A/SF、CUTLASS 路径 vs MPK 路径
输出逐位比对」的实验——列为后续 `docs/60` 之外的一条尾巴。

---

## 8. 与相邻文档的关系 + 判据映射

- `docs/57`（SM100 共享原语层）：本文补上它 §3 只给摘要的「scale 进 TMEM」的**类型侧规范**，
  以及它没点名的 `tmem_sf_frg`/`TmemAllocMode`。
- `docs/59`（TMEM 列布局与 UTCCP）：本文补上它 §8 **点名的最直接下一步**（`mma_traits_sm100.hpp`
  的 block-scaled scale TMEM trait）并修正其文件名笔误；§7 的对照是它「UTCCP 精确映射未验证
  （只记不判）」那条边界的延伸。
- `docs/33`/`docs/34`（`linear_fp8_sm100` / `fp8_group_gemm_sm100` device 身体）：本文是它们
  §「scale 转置 + UTCCP」那段所依赖的 **CUTLASS trait 侧根**。
- **判据映射**：本文只贡献 **C3（多算子范围/设备体内布局契约）**——它是「一个 task 的 device body
  内部、SMEM→TMEM 的布局/寻址契约」，不涉及 launch/常驻（C1/C2）、不涉及跨算子（C3 之外的协调）、
  与去中心化调度（C5）无关。

---

## 9. 局限（诚实标注）

- **SM100 指令/TMEM/UTCCP 本体未执行**：本文的 `tmem_sf_frg`/`tmem_frg` 布局是**在 host 上对真实
  CUTLASS 头文件求值**得到的（`src/sf_tmem_trait.cu`），属于「trait 规定的形状」，**不是**在 SM100 上
  观测到的硬件行为；`sf_id` 与地址高位的硬件语义、§7 的两种约定谁对，均**未验证**。
- **未构建/未跑 MPK 的 FP8 线性 task**：SM100 专属，本机不可执行；本文是纯源码走读 + host 侧布局求值。
- **MPK 侧只走 `linear_fp8_sm100.cuh` 一支**：`linear_fp8_1d2d_sm100.cuh`（另一支、未 round 列数）
  与 `linear_fp8_sm100_cute_tie` 想表达的 CuTe 解构路径未展开；`fp8_group_gemm_sm100.cuh` 的同段
  只做了行号级交叉引用，未逐行重走（`docs/33` 已覆盖其 device 身体）。
- **CUTLASS 只走 `f3fde583` 快照**：更高版本的 trait 命名/实现可能变化。

---

## 10. 参考坐标（可核对）

- MPK（`f9eb70c`）：
  - `include/mirage/persistent_kernel/tasks/blackwell/sm100_utils.cuh`（`make_sf_desc:51-63`、
    `make_runtime_instr_desc_with_sf_id:116-123`、`SM100_MMA_MXF8F6F4_SS:127-150`）；
  - `include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_sm100_utils.cuh`
    （`make_sf_desc:30-34`、`make_runtime_instr_desc_with_sf_id:147-154`、
    `get_num_aligned_tmem_cols:156-172`、`SM100_MMA_MXF8F6F4_SS:182-205`、`..._2x1SM_SS:207-230`）；
  - `include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100.cuh`
    （`kNumUTCCPAlignedElems:78`、`SF_BLOCK_M/N:140-143`、`kNumSFATmemCols:159-160`、
    `kNumTmemCols:167-169`、`kTmemStartColOfSFA/SFB:170-171`、`instr_desc:408-416`、
    UTCCP copy `:463-483`、`sf_id:495-500`、`fma:508-523`、warp 转置 `:536-547`）；
  - `include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_cute_tie.cuh`（全 60 行；
    `linear_fp8_sm100_utils.cuh:8` 传递 include）。
- CUTLASS（`deps/cutlass` `f3fde583`）：
  - `include/cute/atom/mma_traits_sm100.hpp`
    （`TmemAllocMode:426-455`、`tmem_frg:461-583`、`tmem_e_frg:593-669`、
    `tmem_sf_frg:813-873`、`tmem_frg_ws:876-959`、`MakeTensor:967-1016`、
    `MMA_Traits<SM100_MMA_MXF8F6F4_SS>:2700-2784`）；
  - `include/cute/arch/mma_sm100_desc.hpp`
    （`InstrDescriptorBlockScaled:441-468`、`make_instr_desc_block_scaled:552-586`、
    `make_runtime_instr_desc_block_scaled:588-621 / 623-649`）；
  - `include/cute/arch/tmem_allocator_sm100.hpp:45-53`（`DP<T> = (1<<16) << OffsetShift`）。
- 本文实验：`src/sf_tmem_trait.cu` + `harness/sf_tmem_trait.sh` + `harness/sf_tmem_trait.out.txt`
  （目标卡 H100/sm90a，host 侧，未 launch kernel）。
