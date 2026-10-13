# MPK SM100 共享原语层走读：`sm100_ptx.cuh` / `sm100_utils.cuh` / `storage.cuh` / `tensor_init.cuh`（P143-1）

> 定位：本文补掉 `docs/22-mpk-compiler-manual.md` §3.2 只给摘要（「**TMEM** 累加（`Allocator2Sm`）→
> `tcgen05.commit.cta_group::2...multicast::cluster` 会合」）与 `docs/38-mpk-common-primitives.md`
> **只走 `tasks/common/*` + `ampere/mma.cuh`** 留下的缺口——`docs/38` 走的是 **arch-agnostic/Ampere**
> 的建筑地基，**完全没有打开 `tasks/blackwell/` 里那层「SM100 共享原语」**：
> `blackwell/sm100_ptx.cuh`（89 行）、`blackwell/sm100_utils.cuh`（152 行）、`blackwell/storage.cuh`
> （167 行）、`blackwell/tensor_init.cuh`（30 行）。它们被 `docs/32`–`docs/37`/`docs/51`/`docs/52`
> **作为依赖反复引用**（`docs/33` §9.4、`docs/34` §3.5/§? 都列过坐标），却从未有哪一篇把它们
> **当作一层**逐行打开、说清「谁提供哪套 `tcgen05` 原语、TMEM 生命周期归谁管、commit 会合到哪个
> mbarrier」。本文补上这层。快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`
> （与 `docs/06`–`docs/56` 同快照）。**纯源码走读（资料分析），无新实验**——这些原语是
> SM100/Blackwell 专属，本机（目标卡 H100/sm90）不可编译更不可运行，只核静态事实。

---

## 0. 三条 headline

1. **SM100 的同一组张量核原语，在 `blackwell/` 目录里「实现了不止一遍」，且互不共享代码**。
   两条主脉：(a) `sm100_ptx.cuh`（`namespace kernel::sm100_ptx`）——**裸内联 PTX**，为手写 MLA
   decode/MTP（`docs/35`/`docs/36`）提供 `elect_sync`/`mbar_*`/`make_desc`/`tcgen05_mma`/
   `tcgen05_commit`；(b) `sm100_utils.cuh`（`namespace kernel::sm100`）——源码首行自陈
   *"adapted from the DeepGEMM SM100 Blackwell path"*（`:26-28`），为 block-scaled FP8 linear 家族
   （`docs/32`/`docs/33`）提供 UMMA 描述符/UTCCP scale 描述符/block-scaled MMA。更细地，
   **同一个 block-scaled MMA 指令 `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale` 竟有三份封装**：
   `kernel::sm100::SM100_MMA_MXF8F6F4_SS`（`sm100_utils.cuh:127-150`）、
   `mirage::blackwell::linear_fp8_sm100::sm100::SM100_MMA_MXF8F6F4_SS`（`linear_fp8_sm100_sm100_utils.cuh:182-205`，
   与前者近乎逐行相同）、以及 CUTLASS 模板 `cute::SM100_MMA_MXF8F6F4_SS<T,...>`；
   `linear_fp8_1d2d_sm100.cuh` 甚至**同时**用两份（`:132` 用 `kernel::sm100::…` 当 `mma_issue_t`、
   `:134` 用 `cute::…<>` 当 `mma_plumbing_t`）。这是 `docs/22` §5「同一套惯用法在仓库里实现了两遍」
   在 SM100 原语层上的**「两遍变三遍」**。

2. **`tcgen05.commit` 与 TMEM 分配构成一条明确的「运行期契约」，且它是常驻 megakernel 里
   「一个 task = 一个 CTA、单 warp 持有 TMEM」这条硬约束的根**。commit 把异步 MMA 的完成事件
   `arrive` 到**一个 mbarrier**（`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`，
   `sm100_ptx.cuh:82-86`；per-stage 一个、末尾一个 `mainloop_bar`，`mla_decode_sm100.cuh:200/202`），
   消费者用 `mbar_wait` 等；TMEM 则由**一个 warp** 用 `tcgen05.alloc/dealloc`（手写路径，把结果
   写进共享内存再 `__syncthreads` 广播，`mla_decode_sm100.cuh:109-116/452-457`）或 CuTe
   `TmemAllocator`（FP8 家族，`linear_fp8_1d2d_sm100.cuh:615-617/723`）分配释放。**同一 worker 上
   共驻的所有 SM100 task 必须用同一个 warp 做 TMEM 生命周期**——`mla_mtp_decode_tp2_sm100.cuh:190-195`
   有显式 `NOTE` 解释这条（FP8 CUTLASS 家族全用 warp 0，故 MLA 也必须 warp 0）。而**旧版 base
   `mla_mtp_decode_sm100.cuh`（生产注册路径，`docs/24` P121-1）仍用 warp 1 alloc**（`:146-152`）——
   一条**静态不对称**（只记不判，本机无法证伪）。

3. **`storage.cuh` 是 block-scaled FP8 线性家族的「SMEM ABI」，`tensor_init.cuh` 是唯一纯
   arch-agnostic 的薄身体**。`storage.cuh` 定义四个 pipeline storage 结构体
   （`PipedSharedStorage` `:25-55`、`PipedSharedStorageWithSF` `:57-100`、`GateTopKSharedStorage`
   `:102-130`、`MoESharedStorage` `:132-165`），把「A/B/C（+SFA/SFB）数据缓冲 + 四组
   `ab/acc_full/empty` mbarrier 数组 + `tmem_base_ptr`」**按固定对齐（数据 `alignas(128)`、
   mbarrier `alignas(16)`）打包成一个 struct**，四种任务共用同一布局——与 `docs/51` 的「布局类型
   即 SMEM 的 ABI」是同一件事的数据侧镜像。`tensor_init.cuh`（30 行）则是全仓最短的 device 身体
   （纯逐元素填 `init_val`、无 TMA/UMMA/SMEM），登记为 `TASK_TENSOR_INIT=262`（`docs/37` §?）。

---

## 1. 这层库在目录里的位置与包含关系

`tasks/` 按代际分目录（`docs/38` §1 已列）。本文只取 `blackwell/` 下**跨 task 共享**的四件：

| 文件 | 行数 | 角色 | 生产闭包？ |
|---|---|---|---|
| `blackwell/sm100_ptx.cuh` | 89 | 裸 PTX：`elect_sync`/`mbar_*`/`make_desc`/`tcgen05_mma`/`tcgen05_commit` | 是（`task_header.cuh:59` 直接 include） |
| `blackwell/sm100_utils.cuh` | 152 | DeepGEMM 派生：UMMA/SF 描述符 + block-scaled MMA 封装 | 是（经 FP8 linear 家族） |
| `blackwell/storage.cuh` | 167 | 四个 pipeline SMEM storage 结构体 | 是（经 FP8/MoE linear 家族） |
| `blackwell/tensor_init.cuh` | 30 | 纯填值身体（`TASK_TENSOR_INIT`） | 是（`task_header.cuh:65`） |

各文件的**直接消费者**（`grep` 快照）：

- `sm100_ptx.cuh` ← `task_header.cuh:59`、`mla_decode_sm100.cuh:49`、`mla_mtp_decode_sm100.cuh:66`。
  另有 tp2/4/8 **各自在私有 `namespace …::ptx` 里重定义一份**（docs/31 已记；`mla_mtp_decode_tp2_sm100.cuh:64-127`），
  是本文 headline 1「实现多遍」的第四份。
- `sm100_utils.cuh` ← `fp8_group_gemm_sm100.cuh:18`、`linear_fp8_1d2d_sm100.cuh:41`。
- `storage.cuh` ← `linear_fp8_1d2d_sm100.cuh:42`、`linear_sm100_mpk.cuh:27`、`moe_linear_sm100.cuh:27`、
  `fp8_group_gemm_sm100.cuh:45`。
- `tensor_init.cuh` ← `task_header.cuh:65`。

一个值得记的**注释 vs 现实落差**（`docs/34` 已点过一次）：`task_header.cuh:43-44` 注释写
*"sm100_ptx.cuh must be included BEFORE mla_mtp_decode_sm100.cuh at top level so kernel::sm100_ptx
is defined in the correct namespace"*，但直接 `#include "sm100_ptx.cuh"` 实际在 `:59`（`mla_mtp_decode_sm100.cuh`
在 `:46`，更早）——靠 `mla_mtp_decode_sm100.cuh:66` 自己 include + `#pragma once` 兜住。**该注释已滞后于
include 顺序**。

---

## 2. `sm100_ptx.cuh`：手写路径的裸 PTX 原语（89 行）

`namespace kernel::sm100_ptx`（`:24-25`），文件头注释自陈用途（`:16-18`）：*"tcgen05 MMA, TMA mbarrier,
TMEM, warp election. Shared by any SM100 kernel using tensor cores (MLA decode, MTP, future kernels)"*。
八个函数：

- **`elect_sync()`（`:27-34`）**：`elect.sync _|%px, 0xFFFFFFFF` 选出一个 lane，返回值 1/0。所有
  「单线程代表」`if (wid == k && elect_sync())` 分支（`mla_decode_sm100.cuh:100/119/154/164/184/211`）
  的基元。
- **`mbar_init(addr,count)`（`:36-39`）**：`mbarrier.init.shared::cta.b64`，初始化一个 CTA 内 mbarrier。
- **`mbar_wait(addr,phase)`（`:41-49`）**：`mbarrier.try_wait.parity.acquire.cta` + 自旋（超时常数
  `0x989680`=10,000,000 ns≈10 ms）；**parity 语义**——同一个 mbarrier 反复用要翻转 `phase`（消费侧
  `mla_decode_sm100.cuh:168/188` 每绕满一轮 `phase ^= 1`）。
- **`mbar_tx(addr,bytes)`（`:51-57`）**：`mbarrier.arrive.expect_tx.release.cta`——给 TMA 事务登记字节数，
  TMA 完成时自动 `complete_tx`（`mla_decode_sm100.cuh:120/173` 在发 `cp.async.bulk.tensor` 前登记）。
- **`desc_enc`/`make_desc`（`:59-67`）**：把一个 SMEM 地址编成 **128 B swizzle、f16** 的 UMMA 描述符——
  `SBO = 8*128` 硬编码（`:64`）、bit 46/61 写死（`:65-66`）。注意这是**固定布局**描述符，等价于
  `sm100_utils.cuh` 的 `make_umma_desc<Major::K, …, BLOCK_K*sizeof==128>`，但**不复用同一函数**。
- **`tcgen05_mma(taddr,a_desc,b_desc,idesc,acc)`（`:69-80`）**：`tcgen05.mma.cta_group::1.kind::f16`，
  `acc` 谓词决定累加/覆盖（`mla_decode_sm100.cuh:198` 用 `(ki==0 && k2==0)?0:1` 控制首轮清零）。
- **`tcgen05_commit(mbar_addr)`（`:82-86`）**：`tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64`——
  **把此前所有异步 tcgen05 的完成事件 arrival 到一个 mbarrier**（`shared::cluster` 作用域，尽管
  `cta_group::1`）。

**配套的「fence 三件套」不在本文件**、散在消费者里：`tcgen05.fence::after_thread_sync`
（`mla_decode_sm100.cuh:189/209`，在 `mbar_wait` 与 `tcgen05_mma`/`tcgen05.ld` 之间强制时序）、
`tcgen05.wait::ld.sync.aligned`（`:255`，等 TMEM 读落地）、`fence.mbarrier_init.release.cluster`
（`:107`，mbarrier 初始化后发布）。**即「原语层只给算子砖块，成套的时序契约由 task 身体自己拼」**。

---

## 3. `sm100_utils.cuh`：DeepGEMM 派生的 UMMA/SF 描述符（152 行）

`namespace kernel::sm100`（`:25`），首行注释（`:26-28`）*"adapted from the DeepGEMM SM100 Blackwell
path: provide the minimum descriptor, TMEM, block-scaled MMA helpers needed by Mirage's block-scaled
FP8 linear kernel"*。逐函数：

- **`make_smem_desc(layout, smem_ptr, sbo, lbo)`（`:30-49`）**：通用 UMMA SMEM 描述符——`version_=1`、
  `lbo_mode_=0`、`layout_type_` 直接存、`start_address_ = smem_addr >> 4`、`stride_byte_offset_ = sbo >> 4`、
  `leading_byte_offset_ = lbo >> 4`（GPU 描述符就是**地址/偏移按 16 B 压缩**）。
- **`make_sf_desc(smem_ptr)`（`:51-63`）**：UTCCP scale 路径的描述符——`SWIZZLE_NONE`、`SBO=8*16`、`LBO=0`；
  注释 `:54-57` 说明「scale 先 stage 到 SMEM，再经 UTCCP 拷进 TMEM scale 列，故用非 swizzle 布局、
  重排在 UTCCP 前显式做」，与 `docs/33` 的 scale warp 逐行对上。
- **`replace_smem_desc_addr(desc, smem_ptr)`（`:65-73`）**：只改描述符的地址字段（保留 layout/SBO/LBO），
  用于 pipe 里换 stage buffer。
- **`advance_umma_desc_lo<Major,K,BLOCK_MN,kSwizzle,dtype>(base,offset,k_idx)`（`:75-86`）**：K-major 下
  **描述符沿 K 线性推进**——`base + ((offset+k_idx)*sizeof(dtype)) >> 4`；`static_assert` 只支持 K-major
  （`:81-82`）。
- **`make_umma_desc<Major,K,BLOCK_MN,BLOCK_K,kSwizzle,kUseBase32,dtype>(ptr,mn_idx,k_idx)`（`:88-114`）**：
  K-major SW128——`static_assert(kSwizzleMode == BLOCK_K*sizeof(dtype))`（`:98-100`）、`SBO=8*BLOCK_K*sizeof`、
  `LBO=0`（注释 `:103-106`「atom = K 轴 8×128B，SBO=跨 MN 走一个 atom，LBO=0」）。
- **`make_runtime_instr_desc_with_sf_id(desc,sfa_id,sfb_id)`（`:116-123`）**：给 block-scaled 指令描述符
  填 `a_sf_id_`/`b_sf_id_` 并整体左移 32 位——**SF 的 TMEM 列号就藏在这条 idesc 里**。
- **`struct SM100_MMA_MXF8F6F4_SS`（`:127-150`）**：`fma(...)` 发
  `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p`——
  **block-scaled MXFP8/6/4 UMMA**，`scale_c` 谓词决定是否覆盖 scale。

**这一层与 `linear_fp8_sm100_sm100_utils.cuh`（232 行，`namespace mirage::blackwell::linear_fp8_sm100::sm100`）
近乎逐行重复**：后者的 `make_smem_desc` 实现与前者**逐字符相同**（对比 `sm100_utils.cuh:30-49` 与
`linear_fp8_sm100_sm100_utils.cuh:13-31`），另有 `get_num_aligned_tmem_cols`（`:156-172`）、
`tcgen05_before/after_thread_sync`（`:174-180`）、`SM100_MMA_MXF8F6F4_SS`（`:182-205`）与
`SM100_MMA_MXF8F6F4_2x1SM_SS`（`:207-230`，`cta_group::2`）。**「同一段代码在 `kernel::sm100` 和
`mirage::blackwell::linear_fp8_sm100::sm100` 各存一份」是 `docs/22` §5 清单的一条新实例**。

---

## 4. TMEM 运行期契约：alloc / dealloc / 单 warp 约束

TMEM（Tensor Memory）在 SM100 上是 per-SM 的显式资源，**必须显式 alloc/dealloc**，这正是常驻
megakernel 里最脆的一环。两条分配路线并存：

### 4.1 手写路径：`tcgen05.alloc/dealloc` + 共享内存广播

形状是「**某个 warp 发 alloc 指令、结果写进一块 `__shared__ int`，再 `__syncthreads` 让全 CTA 读到
`taddr`**」（`mla_decode_sm100.cuh:108-116`）：

```
} else if (wid == 1) {                                        // alloc 由 warp 1 发
  int addr_smem = __cvta_generic_to_shared(tmem_addr_buf);
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
               "r"(addr_smem), "r"(MLA_D_V));                 // 分配 MLA_D_V(=512) 列
}
__syncthreads();
int const taddr = tmem_addr_buf[0];                           // 全 CTA 读同一个 taddr
```

释放对称（`mla_decode_sm100.cuh:452-457`）：`__syncthreads(); if (wid == 0) tcgen05.dealloc...taddr, ncols`。

**关键静态事实——alloc 的 warp 与 dealloc 的 warp 在旧文件里不一致**：

| 文件 | alloc warp | dealloc warp | 备注 |
|---|---|---|---|
| `mla_decode_sm100.cuh`（standalone、`docs/36` 记「被 MTP 版顶替」） | **1**（`:108`） | **0**（`:453`） | 无 CuTe 约定注释 |
| `mla_mtp_decode_sm100.cuh`（base tp1，**生产注册路径**） | **1**（`:146`） | **0**（`:666`） | 无 CuTe 约定注释 |
| `mla_mtp_decode_tp2_sm100.cuh` | **0**（`:196`） | 0（`:724`） | `:190-195` 有 `NOTE` |
| `mla_mtp_decode_tp4_sm100.cuh` | 0（`:184`） | 0（`:717`） | `:181` 注释「matches CuTe convention」 |
| `mla_mtp_decode_tp8_sm100.cuh` | 0（`:177`） | 0（`:703`） | `:176` 注释「matches CuTe convention」 |

`mla_mtp_decode_tp2_sm100.cuh:190-195` 的 `NOTE` 就是契约原文：*"CuTe's TMEM allocator requires
tcgen05.alloc/dealloc to be issued by the SAME warp across all kernels co-scheduled in the persistent
kernel. All FP8 CUTLASS kernels (linear_fp8, moe_linear, fp8_group_gemm) use warp 0 for both. If MLA
kernels use a different warp (e.g. warp 1), subsequent FP8 kernels on the same worker hang inside their
tmem_allocator.allocate(). So we must also use warp 0 here."*

即：**tp2/4/8 的较新作者把 alloc 从 warp1 改到 warp0 来满足 CuTe 约定，但 base tp1 与旧 decode
仍是 warp1**。这是 `docs/34` §3.5「已知坑」记录的「TMEM-warp 一致」约束的**又一处静态不对称**，
只记不判（本机无 SM100，无法证伪生产上是否真并发 FP8）。同时注意：**alloc 与 dealloc 用不同 warp
在 PTX 语义下未必非法**（`taddr` 是 CTA 级、`sync.aligned` 只要整 warp 参与即可），真正的风险点是
**跨 task 的 CuTe 分配器锁**——这正解释了为何「同一 task 内 alloc/dealloc 用不同 warp」的老代码能
存活，而「跨 task 不一致」才会挂。

### 4.2 CuTe 路径：`TmemAllocator` + NamedBarrier 广播

FP8 线性家族（`docs/32`/`docs/33`）走 CUTLASS 的 `cute::TMEM::Allocator1Sm`：一个 warp 调
`tmem_allocator.allocate(cols, &shared_storage.tmem_base_ptr)`（`linear_fp8_1d2d_sm100.cuh:615-617`、
`linear_sm100_mpk.cuh:570-572`、`fp8_group_gemm_sm100.cuh:1446`），配一个 `NamedBarrier
tmem_allocation_result_barrier`（`:282`/`:267`/`:661`）把结果广播给全 CTA
（`arrive_and_wait` `:452`/`:487`/`:1020`/`:1450`），结束时 `tmem_allocator.free(tmem_base_ptr, cols)`
（`:723`/`:732`/`:1560`）。**列数须对齐**：`get_num_aligned_tmem_cols<kNumCols>()`
（`linear_fp8_sm100_sm100_utils.cuh:156-172`）把请求向上取整到 {32,64,128,256,512}，并
`DG_STATIC_ASSERT(kNumCols <= 512)`——**TMEM 一共 512 列**，`linear_fp8_sm100.cuh:168` 把
accum+SF 三段列数相加再对齐。

### 4.3 为什么它是 megakernel 的硬约束

在 kernel-per-operator 世界里，每颗 kernel 起止自带 TMEM 生命周期，CSV 到 driver 都帮你隔离。
但 megakernel **把所有 SM100 task 编进同一个常驻 kernel、同一小组 worker SM 上轮流跑**——于是
TMEM 变成一份**跨 task 复用的、必须约定归属的全局资源**：谁 alloc（哪个 warp）、列数怎么对齐、
释放时机谁来管，全部从「由运行时保证」退化成「由 device task 之间的约定保证」。这与
`docs/14`/`docs/17` 反复出现的主题（**megakernel 把「框架帮你做的事」下沉成「task 之间自己约定」**）
同构，也是「一个 task = 一个 CTA、单 warp 持有 TMEM」这条粒度约束的根。

---

## 5. `tcgen05.commit` / mbarrier 运行期契约

异步 `tcgen05.mma` 不阻塞；要知道「这一批 MMA 何时算完、TMEM 结果何时可读」，靠 **commit + mbarrier**。
契约形状（`mla_decode_sm100.cuh` QK 相）：

- 建立阶段：warp0 的单线程 `mbar_init` 每个 stage 一个 `tma_bar`/`mma_bar` + 一个 `mainloop_bar`，
  然后 `fence.mbarrier_init.release.cluster`（`:154-161`）。
- 生产阶段（warp1 单线程）：`mbar_wait(tma_bar+stage*8, phase)`（等 TMA 把 K tile 送到）→
  `tcgen05.fence::after_thread_sync` → 若干 `tcgen05_mma` → **`tcgen05_commit(mma_bar+stage*8)`**
  （`:184-201`）：**把这一 stage 的 MMA 完成事件 arrival 到该 stage 的 mma_bar**；循环跑完再
  `tcgen05_commit(mainloop_bar)`（`:202`）。
- 消费阶段：全 CTA `__syncthreads(); mbar_wait(mainloop_bar, 0);`（`:205-206`）——**mainloop_bar 是
  「整个 K 循环都做完」的会合点**；随后 `tcgen05.fence::after_thread_sync`（`:209`）再进入 softmax，
  用 `tcgen05.ld.sync.aligned.32x32b.x16` 从 TMEM 读回（`:236-254`）+ `tcgen05.wait::ld`（`:255`）。

要点：(1) **commit 是「一次性 arrival」**（`arrive::one`）——一个 mbarrier 收一次 commit；per-stage
轮转靠 parity `phase` 翻转（`:169-171`）；(2) **commit 作用域是 `shared::cluster`**，即便
`cta_group::1`（`sm100_ptx.cuh:83-84`），为 2SM 路径留了口子（`docs/37`/`docs/52` 的
`cta_group::2 + multicast::cluster`）；(3) **fence 必须成对出现**在 `mbar_wait` 与 `tcgen05` 读写之间，
否则异步指令可能越过同步点。**这套「commit→mainloop_bar→fence→ld→wait::ld」是 SM100 手写 attention
每个 tile 都要重打一遍的样板**，也是 `docs/35`/`docs/36` 走读里「消费者/生产者两个 elected warp」
时序的底层。

---

## 6. `storage.cuh`（SMEM ABI）与 `tensor_init.cuh`

### 6.1 四个 pipeline storage 结构体

`storage.cuh`（`namespace kernel`）把「一个流水 task 在 SMEM 上需要的全部东西」打包成 struct，
**四种任务共用同一布局约定**：

| struct | 行 | 数据缓冲 | mbarrier | 特异字段 |
|---|---|---|---|---|
| `PipedSharedStorage` | `:25-55` | A/B/C | `ab_full/empty[Num_AB_Stage]` + `acc_full/empty[Num_ACC_Stage]` | `tmem_base_ptr` |
| `PipedSharedStorageWithSF` | `:57-100` | A/B/C/SFA/SFB | 同上 | `tmem_base_ptr` + `tmem_columns` |
| `GateTopKSharedStorage` | `:102-130` | A/B | 同上 | `tmem_base_ptr` + `reduce_values_buffer[32]` |
| `MoESharedStorage` | `:132-165` | A/B | `a_full/b_full/ab_empty`（A、B 分开）+ acc | `expert_mask[Num_Experts]` + `tmem_base_ptr` |

对齐是契约的一部分：数据缓冲 `alignas(128)`（`cute::ArrayEngine`，配 SW128/TMA），mbarrier 数组
`alignas(16)`。每个 struct 都给 `tensor_sA()/tensor_sB()/…` 便捷方法（`make_tensor(make_smem_ptr(A.begin()), Layout{})`）
——**布局类型由模板参数 `ASmemLayout` 等传入**，即 `docs/51`「布局类型即 SMEM 的 ABI」在**数据侧**
的镜像：`storage.cuh` 定「有哪些 buffer、怎么对齐」，`smem_layout.cuh` 定「每个 buffer 内部怎么
swizzle」。`PipedSharedStorageWithSF` 相对 `PipedSharedStorage` 多出的 `tmem_columns` 字段，
正是 `docs/34` 说的「身体 B 在身体 A 基础上多两个 scale 缓冲和 `tmem_columns`」。

### 6.2 `tensor_init.cuh`：全仓最短的「薄身体」

30 行、一个模板函数 `tensor_init_sm100_task_impl<T,BATCH_SIZE,OUTPUT_SIZE,OUTPUT_STRIDE>(input_ptr, init_val=0.0f)`
（`:19-28`），双层循环逐元素写 `T(init_val)`——**无 TMA、无 UMMA、无 mbarrier、无 SMEM**，
是纯 arch-agnostic 的标量身体（名字带 `_sm100` 但内容与 SM100 无关，与 `docs/56` headline「`_sm100`
后缀 ≠ SM100 专属」同族）。登记为 `TASK_TENSOR_INIT=262`（`docs/37` §?），前端 `tensor_init_layer`
用于把 `moe_output` 清零（`docs/26`）。它出现在本文里，是因为它被 `task_header.cuh:65` 直接 include、
是「SM100 目录下但完全不依赖 SM100 原语」的反例——**判一个 `blackwell/` task 能否在目标卡执行，
须看它是否真用到这层原语，而非它躺在哪个目录**。

---

## 7. 与相邻走读/论文的对照，与判据映射

- **与 `docs/38`（`tasks/common/*`）互补**：`docs/38` 走 Ampere/arch-agnostic 地基（`cp.async`/
  `ldmatrix`/bf16/SMEM 助手 + 手写 `m16n16k16`），本文走 **SM100 张量核原语**（`tcgen05`/TMEM/UMMA
  描述符）。二者关系：Ampere 地基是**所有** device task 的公共依赖，SM100 原语只被**部分** task
  消费（MLA decode/MTP + FP8 linear 家族 + MoE linear/group-GEMM）；**通用 attention（`docs/44`）
  仍在用前者、不用本文这层**——正是「Blackwell 目录里也有 Ampere 手写 MMA」的 `docs/38` headline 1。
- **与 `docs/32`/`docs/33`/`docs/34`**：这三篇把 `sm100_utils.cuh` 的**每个函数**作为具体 task 的依赖
  引用过（`docs/33` §9.4 列 `fp8_make_sf_smem_desc` 等「有定义无调用者」的本地副本），本文把
  **这层本身**收口，并补出「三份 `SM100_MMA_MXF8F6F4_SS`」与「两份 `make_smem_desc`」的全貌。
- **与 `docs/35`/`docs/36`**：那两篇的手写 MLA decode/MTP 时序，其 `elect_sync`/`mbar_*`/`tcgen05_*`
  全部来自 `sm100_ptx.cuh`；本文补上 alloc/dealloc 的 warp 契约与 commit 的 mbarrier 契约。
- **与 `docs/52`**：`docs/52` 走 CUTLASS `CollectiveBuilder`（QK/PV atom、`SM100_TMA_2SM_LOAD`、
  `AtomThrShapeMNK`），是**2SM 的 CUTLASS 侧**；本文走的是 **MPK 自造的 `cta_group::1` 手写原语层**，
  二者是「同一硬件、两条来源」。
- **与 MPK 论文**：论文 §4.2 只提「每 task 用 superoptimizer 生成 CUDA」、§5.3 讲单 kernel 内
  task 级流水与 paged SMEM；**SM100 的 `tcgen05`/TMEM 原语与三条 TMEM-warp 契约论文均未描述**
  （同 `docs/22` §7「论文未提 Blackwell/2SM」）。故本层是**快照新增、生产专用**的一层。
- **判据映射**：**只贡献 C3（多算子范围/粒度）**。这些原语是 device body **内部**的算子砖块，
  不引入 host 侧提交、常驻、设备内协调或去中心化调度（C1/C2/C4/C5 全部在运行层 `docs/06`）。
  **但** TMEM-warp 契约间接**约束 C2**：它把「哪些 task 能安全共驻在同一常驻 worker 上」变成
  一个**必须跨 task 约定的资源归属**问题——是 megakernel 把框架职责下沉给 task 之间的又一例证。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100 原语在目标卡 H100/sm90 上**不可编译**（无 `tcgen05`/
  TMEM），本文只读代码、无任何 device 计数；所有行号为快照 `f9eb70c` 静态事实。
- **「warp 不对称」只记不判**：§4.1 表里 base tp1/旧 decode 的 alloc=warp1、dealloc=warp0，
  与 tp2/4/8 的 warp0 不一致——**不判上游是否有意、也不判生产上是否真触发 CuTe 分配器挂起**
  （需 SM100 + 真实共驻负载才能验证）。
- **CuTe/CUTLASS 内部未展开**：`TmemAllocator::allocate/free`、`cute::SM100_MMA_*` 模板、`UMMA::SmemDescriptor`
  的位字段定义属 `deps/cutlass`（`docs/52` 已给 `f3fde583` 锚点），本文只核 MPK 侧的**调用契约**。
- **「三份 MMA 封装」的等价性未证**：只核到三处的 PTX 串逐字相同/近乎相同，未做模板实例化打印或
  数值对跑。
- **committed 描述符的位编码未逐位展开**：`make_desc` 的 bit 46/61、`make_smem_desc` 的
  `version_/lbo_mode_` 语义依赖 GPU 架构文档，本文只记「写死值」。
- **`storage.cuh` 的 `cosize`/对齐未实例化验证**：`alignas(128)` 与各 `ArrayEngine` 实际占用
  未在 SM100 上 `static_assert` 核过。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- `include/mirage/persistent_kernel/tasks/blackwell/`：`sm100_ptx.cuh`（89 行）、`sm100_utils.cuh`（152）、
  `storage.cuh`（167）、`tensor_init.cuh`（30）、`linear_fp8_sm100_utils.cuh`（143）、
  `linear_fp8_sm100_sm100_utils.cuh`（232）、`linear_fp8_sm100_tma_utils.cuh`（118）。
- 生产 include：`blackwell/task_header.cuh:43-44/46/59/65`。
- 手写路径消费者：`mla_decode_sm100.cuh:49/100-116/154-210/236-255/452-457`、
  `mla_mtp_decode_sm100.cuh:66/140-154/665-670`、`mla_mtp_decode_tp2_sm100.cuh:64-127/190-212/724`、
  `mla_mtp_decode_tp4_sm100.cuh:181-195/717`、`mla_mtp_decode_tp8_sm100.cuh:176-188/703`。
- CuTe 路径消费者：`linear_fp8_1d2d_sm100.cuh:132/134/282/342/452/615-617/723`、
  `linear_sm100_mpk.cuh:267/410/487/570-572/732`、`fp8_group_gemm_sm100.cuh:18/45/661/778/1446/1560`、
  `linear_fp8_sm100_sm100_utils.cuh:13-31/156-172/182-230`。
- 生产布线：`src/kernel/task_register.cc`（`register_mla_decode_sm100_task:4102`、
  `mla_mtp_decode_sm100_task_impl<false>` 发射 `:4168/4443-4445`、tp2/4/8 注册 `:5192/5287/5384`）、
  `runtime_header.h`（`TASK_TENSOR_INIT`）。
- 前置走读：`docs/22`（编译栈手册 §3.2 TMEM/tcgen05 commit）、`docs/32`（SM100 手写线性）、
  `docs/33`（FP8 group-GEMM）、`docs/34`（CuTe 线性 + MTP §3.5 TMEM-warp）、
  `docs/35`/`docs/36`（MLA family）、`docs/38`（`tasks/common/*` 地基）、`docs/51`（SMEM/DMEM 布局）、
  `docs/52`（CUTLASS collective_builder 侧）。
- 相关坑（`ROADMAP.md`「已知坑」）：常驻 kernel 内 TMEM alloc/dealloc 同 warp、SM100 共驻 device
  适配（warp 守卫/命名屏障 ID）、`_sm100` 后缀 ≠ SM100 专属、`extern __shared__` 全 TU 共享基址。
