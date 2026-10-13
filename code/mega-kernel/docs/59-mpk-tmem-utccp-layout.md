# MPK SM100 TMEM 列布局与 UTCCP 重排走读：block-scaled FP8 的 scale 从 SMEM 到 TMEM

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/58` 同快照）。
> `docs/51` 走 Ampere 软件 `smem_row`/`dmem_row` 布局、`docs/58` 走 Hopper/Blackwell TMA 版
> `smem_tma` 布局，两者都是 **SMEM 侧**的布局契约；本文补上**第三个存储层级**——SM100 的
> **TMEM（Tensor Memory）列布局**与 block-scaled FP8 把 scale 因子从 SMEM 搬进 TMEM 的
> **UTCCP 重排**。这是 `docs/57` §3 只给摘要（「scale 先 stage 到 SMEM，再经 UTCCP 拷进 TMEM
> scale 列」）的下一层细节，也是本专题「布局即 ABI」链条的最后一段。
> **纯源码走读 + 1 个目标卡 H100/sm90a 微实验**：UTCCP 指令/TMEM/描述符是 SM100 专属（本机不可执行），
> 但 **UTCCP 前置的 warp-transpose 是 arch-agnostic SMEM 代码**，可在 H100 上真跑——本文用它把
> 「三份实现是否等价」「重排到底排成什么」两问钉死。所有行号为快照 `f9eb70c` 静态事实。

---

## 0. 三条 headline

**(A)（headline）TMEM 是第三种「布局即 ABI」，且它比 SMEM 更贵——它是 per-SM 的显式资源、必须
`alloc/free` 且列数须是 2 的幂。** 一次 block-scaled FP8 GEMM 把 **累加器 + SFA + SFB 三段**塞进同一段
TMEM 列空间（用列偏移区分），MPK 用三套公式各算一遍、又用两份 `get_num_aligned_tmem_cols`（一份模板、
一份内联）各 round 一遍——与 `docs/57`「SM100 张量核原语实现不止一遍」同族。CUTLASS 的分配器契约写死
「`32 ≤ num_columns ≤ 512` 且 2 的幂」，而 `linear_fp8_1d2d_sm100.cuh` 的分配列数**没 round**（源码留
`// TODO`），只因该身体无调用者才没出事。

**(B)（headline）scale 走「SMEM 行主序 → warp 转置 → UTCCP → TMEM 列」三段，UTCCP 一次搬
128 个 uint32 = 4 列 × 32 dp × 128 bit。** `SM100_UTCCP_4x32dp128bit_1cta` 本质是
`tcgen05.cp.cta_group::1.32x128b.warpx4`（CUTLASS `copy_sm100.hpp:482-498`），源是 SWIZZLE_NONE 的
SMEM 描述符（`sm100_utils.cuh:51-63` 的 `make_sf_desc`，SBO=128 B）。因为 scale 加载先写成
「一行一个 uint32」的行主序，而 UTCCP 要求 4×32 列块，故中间必须插一次 **warp 内转置**。

**(C)（headline，H100 实测）那一次 warp 转置在快照里有三份写法、全部逐位等价；其中两份带一个
`(i ^ (lane>>3))` 的额外置换，注释自称「identical to DeepGEMM」，但 DeepGEMM 自身用的是**无 XOR**
版本。** 本文在 H100/sm90a 上用真实 `ld.shared`/`st.shared` 复刻三份实现，对 host 参考
`out[l*4+k]=in[k*32+l]` **逐位相等**（mismatches=0），并在隔离微基准里量到带 XOR 的两份比无 XOR 的
纯标量版慢约 1.5×——**XOR 是功能中性的迭代序置换**（读集/写集不变），不影响结果、只多算地址；在真实
kernel 里它藏在一次 warp 同步后的 SMEM 小操作里（每次 SF block 一次），代价可忽略，但这坐实了
「按注释推断代码行为会错」这条本专题反复出现的族。

---

## 1. 定位：三种存储层级的布局 ABI

| 层级 | 谁管 | MPK 走读 | 契约形态 |
|---|---|---|---|
| SMEM（软件） | 软件独占解释 | `docs/51` `smem_row`/`dmem_row` | 行主序 + swizzle 位切片，写侧 `cp.async` / 读侧 `ldsm` |
| SMEM（TMA 描述符） | 软硬共管、以硬件为准 | `docs/58` `smem_tma`（Hopper/Blackwell） | 一个 `(B,M,S)` 喂 TMA map + wgmma/UMMA 描述符 + 软件偏移 |
| **TMEM（本文）** | **硬件显式资源** | **本文** | **列号 + 数据通路 lane（dp），须 alloc/free、列数为 2 的幂** |

`docs/57` §3 讲了「谁提供哪套 `tcgen05` 原语、TMEM 生命周期归谁管」，但把「scale 怎么排进 TMEM 列」
留给了本文。下面先把 TMEM 列预算讲清，再讲 UTCCP 重排。

---

## 2. TMEM 列预算：累加器 + SFA + SFB 共用一段列空间

### 2.1 硬件契约（CUTLASS `deps/cutlass@f3fde583`）

`cute/arch/tmem_allocator_sm100.hpp`：TMEM 容量 `504`/`512` 列（`Sm100TmemCapacityColumns = 512`）、
每列 128 个数据通路 lane。`Allocator1Sm::allocate` 的文档注释写死**列数约束**：

> `@param num_columns Number of columns being freed. Must be 32 <= num_columns <= 512 and power of 2.`

`Allocator1Sm`/`Allocator2Sm` 都通过 `tcgen05.alloc.cta_group::{1,2}...b32 [dst], n`（`dst` 是 SMEM 地址，
结果写回该 SMEM 再广播）；且 `@pre For repeated allocations, the same warp must be used`——这正是
`docs/57` §4 记录的「常驻 kernel 内 TMEM alloc 必须同一 warp」契约的 CUTLASS 侧原文。

### 2.2 三份列预算公式（同一件事写三遍）

`SM100 block-scaled FP8` 的 TMEM 排布是 **`[ACC | SFA | SFB]`**，用列偏移切三段：

- **`linear_fp8_sm100.cuh:159-171`**（手写 UMMA 路径，生产注册的名）：
  ```
  kNumSFATmemCols = SF_BLOCK_M / 32;              // SF_BLOCK_M = align(BLOCK_M,128)
  kNumSFBTmemCols = SF_BLOCK_N / 32;
  kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N;   // 累加器（多 M-wave 各一份）
  kNumTmemCols = get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols + kNumSFBTmemCols>();
  kTmemStartColOfSFA = kNumAccumTmemCols;
  kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;
  ```
  注意 `kNumEpilogueStages`（`:161-163`）由「`2*kNumMWaves*BLOCK_N + SF 列 > 512 ? 1 : 2`」决定——
  **列预算是编译期按 512 上限反推 epilogue 级数的**。
- **`fp8_group_gemm_sm100.cuh:486-508`**（MoE group-GEMM，三角色 warp）：`num_tmem_acc_cols =
  MMA_N * NUM_ACC_STAGE`，`kNumSFATmemCols = kNumSFBTmemCols = 4`（写死，注释指「4 列 = 4 个 K 子块」），
  并**内联**同一套 2 的幂 round-up（`<=64/128/256/512`）。
- **`linear_fp8_1d2d_sm100.cuh:115-119`**（CuTe 高层版，无调用者）：`num_tmem_columns = acc + SFA +
  SFB; // TODO`——**没有 round 到 2 的幂**。它调用 `tmem_allocator.allocate(num_tmem_columns, ...)`
  （`:615`），若该值非 2 的幂则违反 CUTLASS 注释契约；因该身体全仓无调用者（`docs/32` §3），静态上
  不触发，属「有实现未接线」族里又一处 TODO（**只记不判**：CUTLASS 实现是否容忍非 2 的幂未核）。

`get_num_aligned_tmem_cols` 本身也有**两份**：`linear_fp8_sm100_sm100_utils.cuh:156-172`（模板、供
`linear_fp8_sm100`）与 `sm100_utils.cuh` 之外的内联版（供 `fp8_group_gemm`）。逻辑一致：col ≤ 32/64/
128/256 → 32/64/128/256，否则 512。

### 2.3 SFA/SFB 的列数与 UMMA 的 `sf_id`

UMMA block-scaled 指令 `tcgen05.mma...kind::mxf8f6f4.block_scale [tmem_c], desc_a, desc_b, idesc,
[tmem_sfa], [tmem_sfb], p`（`sm100_utils.cuh:127-150` 的 `SM100_MMA_MXF8F6F4_SS`）。scale 的**列选择**
由 `idesc` 里的 `a_sf_id_`/`b_sf_id_` 字段给出（`make_runtime_instr_desc_with_sf_id`，
`sm100_utils.cuh:116-123`：把 `sfa_id`/`sfb_id` 填进 `InstrDescriptorBlockScaled` 后整体左移 32 位）。
MMA warp 每个 32-element K 子块传一个 `sf_id`（`linear_fp8_sm100.cuh:515-523`）：

```
sfa_id = (kGranKA == 32 ? k : sfa_stage_in_group_idx);   // k 是 BLOCK_K/32 子块号
...
mma_t::fma(a_desc, b_desc,
           accum_stage_idx*kNumMWaves*BLOCK_N + w*BLOCK_N,     // 累加器列
           k_block_idx>0 || k>0,                               // 是否累加
           runtime_instr_desc,
           kTmemStartColOfSFA + w*(kNumUTCCPAlignedElems/32),  // SFA 列（每个 M-wave +4）
           kTmemStartColOfSFB);                                // SFB 列
```

即 **`kNumMWaves` > 1 时每个 M-wave 的 SFA 占 4 列**（`+w*4`），与 `kNumAccumTmemCols` 里
`kNumMWaves * BLOCK_N` 的累加器切分对应；`fp8_group_gemm` 单 M-wave 故直接 `kTmemStartColOfSFA`
（`:584-585` 的 `linear_fp8_1d2d` 亦然）。DeepGEMM 同字段的算法在其
`sm100_fp8_fp4_gemm_1d1d.cuh:418-426`：`tmem_col_sfa = kTmemStartColOfSFA + subblock_idx*SF_BLOCK_M/32`，
`sf_id_in_subblock = (offset % 128)/32`——与 MPK 逐项同构。

---

## 3. UTCCP：把 scale 从 SMEM 列块搬进 TMEM 列

### 3.1 指令本体

`cute::SM100_UTCCP_4x32dp128bit_1cta`（`cp/arch/copy_sm100.hpp:482-498`）唯一一条 PTX：

```
tcgen05.cp.cta_group::1.32x128b.warpx4 [dst_tmem], src_smem_desc;
```

「4x32dp128bit」= **4 列 × 32 dp（数据通路 lane）× 128 bit**；`.warpx4` 指 4 个 warp 的数据通路粒度。
源用 SMEM 描述符：`make_sf_desc`（`sm100_utils.cuh:51-63`）→ `make_smem_desc(SWIZZLE_NONE, ptr,
8*16, 0)`，即 `layout_type_ = SWIZZLE_NONE`、`stride_byte_offset_ = 8`（128 B = 32 个 uint32）。
`fp8_group_gemm_sm100.cuh:283-296` 的本地 `fp8_make_sf_smem_desc` 是同一件事的第二份（`desc_=0`、
`desc.stride_byte_offset_=8`），又一次「实现不止一遍」。**scale 不需要 swizzle**（注释
`fp8_group_gemm_sm100.cuh:129`），因为 UTCCP 的数据通路本身就是 128-bit 粒度的列拷贝。

### 3.2 一次 UTCCP = 128 个 uint32 = 4 列 × 32 dp

发 UTCCP 的循环（`linear_fp8_sm100.cuh:462-481`、`linear_fp8_1d2d_sm100.cuh:529-548`、
`fp8_group_gemm_sm100.cuh:1178-1199`）都是：

```
using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;
for (i = 0; i < SF_BLOCK_{M,N} / 128; ++i) {
  auto smem_ptr = smem_sf[stage] + i * 128;
  replace_smem_desc_addr(sf_desc, smem_ptr);          // 只改描述符地址字段
  cute_utccp_t::copy(sf_desc, kTmemStartColOfSF + i*4); // 目标列 +4/块
}
```

即**每 128 个 scale 元素 → 4 个 TMEM 列**（`i*4`），与 `kNumSF_TmemCols = SF_BLOCK/32` 一致
（128/32 = 4）。`fp8_group_gemm` 里必须 `elect_one_sync()` 单 lane 发（硬件要求，`:1181-1184`），
MPK `linear_fp8_sm100` 亦用 `cute::elect_one_sync()` 包住（`:465/474`）。

### 3.3 为什么必须先 warp 转置

`fp8_group_gemm_sm100.cuh:298-322` 的注释给了完整理由（本文核对的第一手依据）：
scale 加载代码按 **`sfa_smem[stage][128]`**（`alignas(128)` 的 128 个 uint32，一行一个）**行主序**写入；
而 UTCCP 期望的是「**4 列 × 32 行的列块**」。故在 `smem` 与 UTCCP 之间插一次 warp 内转置。这一层
SMEM 里的「行主序 → 4×32 列块」正是 `docs/51`（软件布局）、`docs/58`（TMA 描述符布局）之外的
**第三处 SMEM 布局约定**——只是它服务的消费者是 UTCCP 而非 `ldsm`/wgmma。

---

## 4. warp 转置的三份实现与索引数学

### 4.1 数学：`init[4][32] → out[32][4]` 的转置

128 个 uint32 可看作 `init[k][l]`（k=0..3，l=0..31；行主序即 `init[k*32+l]`）。目标
`out[l][k] = init[k][l]`，flat 即

```
out[l*4 + k] = in[k*32 + l]
```

本文 H100 实测用 host 端按此式算参考，三份实现全部 `mismatches=0`。

### 4.2 三份实现（快照里真实并存）

| 位置 | 读索引 | 写索引 | 备注 |
|---|---|---|---|
| `fp8_group_gemm_sm100.cuh:322-336` | `(i^(lane>>3))*32 + lane` | `lane*4 + (i^(lane>>3))` | 本地 `fp8_ld_shared`/`fp8_st_shared`，注释称「identical to DeepGEMM」 |
| `linear_fp8_sm100.cuh:536-546` | `(i^(lane>>3))*32 + lane` | `lane*4 + (i^(lane>>3))` | 同一 XOR 版（用 `ld_shared`/`st_shared` 单元素） |
| `linear_fp8_1d2d_sm100.cuh:50-65` | `i*32 + lane` | `lane*4 + i` | `kernel::detail::utccp_required_smem_warp_transpose`，无 XOR |

而 **DeepGEMM 自身**（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:464-473` 与
`sm100_mqa_logits.cuh:288-296`）用的是**无 XOR** 版：读 `smem_ptr + i*32 + lane`、写
`st.shared.v4` 一次四连写 `smem_ptr + lane*4`。所以 MPK 注释「identical to DeepGEMM」只在**结果**意义上
成立——**代码序列并不相同**（多一个 XOR 置换）。

### 4.3 为什么 XOR 是功能中性的

令 `k = i ^ (lane>>3)`。因 `i` 遍历 0..3 时 `k` 也遍历 0..3（XOR 只按 lane 重排迭代顺序），每个 lane
的**读集** `{k*32+lane}` 与**写集** `{lane*4+k}` 都不变，且映射 `(k,lane) → (lane,k)` 与无 XOR 版一致。
即 XOR 只是「同一转置换一个迭代序」，对最终 SMEM 内容**逐位无影响**。H100 实测证实（§5）。

---

## 5. 目标卡 H100/sm90a 实测（`src/tmem_utccp_layout.cu`）

脚本 `harness/tmem_utccp_layout.sh`、原始输出 `harness/tmem_utccp_layout.out.txt`（device 3、nvcc 13.2、
`sm_90a`、单 warp；**重排本身 arch-agnostic 故可真跑；UTCCP/TMEM 指令本体 SM100 专属、只建模不执行**）。

### 5.1 §A 正确性：三实现逐位等价

| 实现 | mismatches vs host 参考 `out[l*4+k]=in[k*32+l]` |
|---|---|
| `plain`（MPK 元素级，无 XOR） | 0 → OK |
| `xor`（`linear_fp8_sm100`/`fp8_group_gemm`） | 0 → OK |
| `v4`（DeepGEMM/`1d2d` 风格） | 0 → OK |

⇒ **三份写法输出逐位相同**，XOR 与 v4 都不改变转置结果。

### 5.2 §B UTCCP 列块视图：是原始 128 行的纯置换

把转置后的 SMEM 按 UTCCP「4 组各 32 个连续 uint32 = 4 列」读，得到「原始行号 → (列, 深度)」表：

```
col0 :   0  32  64  96   1  33  65  97 ...
col1 :   8  40  72 104   9  41  73 105 ...
col2 :  16  48  80 112  17  49  81 113 ...
col3 :  24  56  88 120  25  57  89 121 ...
```

即 **第 g 列收 `{8g+j, 32+8g+j, 64+8g+j, 96+8g+j : j=0..7}`**（32 行），四列合起来恰好是 0..127
的一个置换（实测 `YES`）。这把「转置把 128 行重排成 4×32 列块」讲成了可核对的表。
**注**：UTCCP 经描述符 SBO=128 B 从 SMEM 取数的**精确 dp↔深度映射**由硬件定义，本文未在 SM100 上
用 cuda-gdb/ТMEM dump 验证（本机不具备），故「原始行最终落在 TMEM 第几列/第几 dp」**只记不判**
（表给出的是「按连续 4×32 读 SMEM」的视图）。

### 5.3 §C 计时：XOR/v4 在隔离微基准里更慢

`ITERS=2e6` 次 × 2 遍转置，min-of-5：

| 实现 | cycles / 2×ITERS | cycles / 单次转置 |
|---|---|---|
| `plain`（无 XOR） | 238,000,140 | **59.5** |
| `xor`（MPK 两份） | 364,000,140 | 91.0 |
| `v4`（DeepGEMM/`1d2d`） | 348,000,140 | 87.0 |

⇒ 带 XOR 的两份比无 XOR **慢约 1.5×**（XOR 版每元素多算一次 lane 相关地址、v4 版的 128-bit SMEM
store 有分体代价）。这坐实 §4.3「XOR 功能中性但有实现代价」。**但要诚实标注**：这是**隔离单 warp 微基准**
（含 `__syncwarp` 与循环开销、`clock64` 单 SM 口径），而在真实 kernel 里该转置**每个 SF block 只做一次**
且发生在 `full_barriers` 之后的旁支 warp 上，~30 cycle 的差被隐藏——**不可据此判 MPK 生产路径有性能问题**。

---

## 6. 与 megakernel 的关系

- **TMEM 是常驻 kernel 里最「不共享」的显式资源**：SMEM 可跨 task 复用（`docs/51`/`docs/58`），TMEM 却
  由**单 warp alloc/free**、列数受 2 的幂约束（§2.1），且 `docs/57` §4 已证「共驻的 SM100 task 必须同
  warp 管 TMEM」。故 SM100 megakernel 的 **task 共驻安全集**被 TMEM 生命周期进一步切分——一个 task
  用 warp0 alloc，邻居若用 warp1 就会挂（`mla_mtp_decode_tp2_sm100.cuh:190-195` 的 `NOTE`）。
- **scale 的 SMEM→TMEM 是一次「布局转换必须显式做」的例子**：软件写的是行主序，硬件要 4×32 列块，中间
  没有自动 swizzle（不像 TMA/TMA descriptor 那样软硬共管），必须由 warp 显式转置 + `fence_view_async_shared`
  后再交给 UTCCP（`linear_fp8_sm100.cuh:536-546` 转置后紧接 `fence_view_async_shared`）。这是「布局即
  ABI」在 **跨存储层级**（SMEM→TMEM）版本：转换责任在软件，且只在特定的异步拷贝原语（UTCCP）前。
- **同步点**：scale warp 写 `sfa/sfb_smem` 前必须等 `ab_empty`（由 MMA 的 `umma_arrive` = `tcgen05.commit`
  触发，会等**所有** tcgen05 含 UTCCP 完成）——`fp8_group_gemm_sm100.cuh:1355-1368` 的 `FIX 2026-04-22`
  注释即此。即 **UTCCP 读 SMEM 的生命周期也被纳入 pipeline barrier**，与 `docs/57` §5 的 commit→mbarrier
  契约闭环。

---

## 7. 判据映射

| 判据 | 本段是否贡献 |
|---|---|
| C1 单次 launch | 否 |
| C2 常驻 | 间接（TMEM 生命周期约束「哪些 task 能共驻同一常驻 worker」） |
| C3 多算子范围 | **是**——本文是 block-scaled FP8 task 内 **scale 数据流契约**（SMEM 行主序 → 4×32 列块 → UTCCP → TMEM 列）的逐段清单 |
| C4 设备内协调 | 间接（转置后 `fence_view_async_shared`、UTCCP 目标列、`ab_empty` 门控） |
| C5 SM 级调度 | 否 |

结论：本文贡献 **C3 的「device body 内跨存储层级（SMEM→TMEM）的布局/数据流契约」**，与 `docs/51`
（SMEM 软件布局）、`docs/58`（SMEM TMA 布局）、`docs/57`（TMEM 运行期契约）互补，补齐布局三部曲。

---

## 8. 局限（诚实标注）

- **UTCCP/TMEM 本体未在 SM100 上执行**：本文的 §2/§3 是静态源码事实；§5 只跑了**前置 warp 转置**
  （arch-agnostic SMEM 代码）。UTCCP 经 SBO 从 SMEM 取数的精确 dp↔深度映射、以及 TMEM 列的实际落位
  **未验证**（§5.2 的表是「连续 4×32 读 SMEM」视图，**只记不判**）。
- **计时是隔离微基准**：单 warp、含同步与循环开销，不能映射到真实 kernel 的 SF-block 频率；XOR 的
  ~1.5× 只说明「有实现代价」，不等于生产路径问题（§5.3）。
- **`linear_fp8_1d2d_sm100.cuh` 未 round 列数**只作静态记录（`// TODO`），CUTLASS 实现是否容忍非 2 的幂
  **未核**（该身体全仓无调用者，`docs/32` §3）。
- **未构建/未跑 MPK 本体**；SM100 专属、本机（H100/sm90）不可执行，与 `docs/32`–`docs/58` 同档。
- 未逐行走读 SM100 其它 block-scaled 变体（`linear_fp8_1d2d_sm100_cute_tie.cuh` 的 CuTe 路径、
  `linear_sm100_mpk.cuh` 的 tie 层）里的 UTCCP 用法；未读 `mma_traits_sm100.hpp` 里 block-scaled
  scale-factor 的 TMEM 布局 trait（最直接的下一步）。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`。
- 主线源码：`include/mirage/persistent_kernel/tasks/blackwell/`——
  `sm100_utils.cuh`（`make_sf_desc :51-63`、`make_runtime_instr_desc_with_sf_id :116-123`、
  `SM100_MMA_MXF8F6F4_SS :127-150`）；
  `linear_fp8_sm100_sm100_utils.cuh:156-172`（`get_num_aligned_tmem_cols`）；
  `linear_fp8_sm100.cuh`（列预算 `:159-171`、UTCCP `:462-481`、MMA `:508-537`、warp2 转置 `:536-546`）；
  `fp8_group_gemm_sm100.cuh`（SF 缓冲 `:194-207`、`fp8_make_sf_smem_desc :283-296`、转置 `:298-336`、
  列预算 `:486-508`、MMA warp `:990-1060`、UTCCP `:1178-1199`、scale warp `:1330-1410`）；
  `linear_fp8_1d2d_sm100.cuh`（转置 `:50-65`、列预算 `:115-119`、UTCCP `:529-548`、MMA `:574-596`）。
- CUTLASS 侧（`deps/cutlass` submodule 钉死 `f3fde583`）：`cute/arch/copy_sm100.hpp:482-498`
  （`SM100_UTCCP_4x32dp128bit_1cta`）、`cute/arch/tmem_allocator_sm100.hpp`（`Allocator1Sm/2Sm`
  及 2 的幂/32–512 列契约）。
- 生产案例：`DeepGEMM` `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`
  （`kNumSFATmemCols :122`、UTCCP `:374-389`、`sf_id`/列 `:418-426`、转置 `:464-473`）、
  `sm100_mqa_logits.cuh`（转置 `:288-296`、UTCCP `:309-330`）。
- 本专题前置：`docs/51`（SMEM 软件布局）、`docs/57`（SM100 原语/TMEM 运行期契约）、`docs/58`
  （TMA 版 SMEM 布局）、`docs/32`/`docs/33`（手写 SM100 线性/group-GEMM 身体）。
- 本轮实验：`src/tmem_utccp_layout.cu` + `harness/tmem_utccp_layout.sh` +
  `harness/tmem_utccp_layout.out.txt`。
