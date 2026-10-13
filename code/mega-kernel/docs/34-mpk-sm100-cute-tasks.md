# MPK SM100 第二套 device 身体：CuTe 线性 task 与 MLA-MTP decode 走读（P120-1）

> 定位：本文补掉 `docs/32-mpk-sm100-linear-tasks.md` **§7「局限」明留的两条尾巴**（原话）——
>
> > **未展开**：……身体 B 的 CuTe 主循环只核了结构与惯用法，**未逐行走完 727 行**；
> > `mla_mtp_tpN_main` 主体仍未走（留待后续）。
>
> `docs/33`（P119-1）只处理了 `fp8_group_gemm_sm100.cuh` 这一支，**两条本文该走的尾巴它明确标注「未处理」**
> （`docs/33` §11）。本文把两条一起走完：
>
> 1. **身体 B** `linear_fp8_1d2d_sm100.cuh`（**727 行**）——SM100 block-scaled FP8 线性 task 的**CuTe 高层
>    实现**（与 `docs/32` 走的身体 A「手写裸 PTX」相对）；
> 2. **MLA-MTP decode 家族** `mla_mtp_decode_sm100.cuh`（748 行）+ `mla_mtp_decode_tp{2,4,8}_sm100.cuh`
>    （796/787/775 行）——DeepSeek-V3 **多 token 预测（MTP）投机解码的 MLA attention task**，是目前走读到的
>    **最“非 GEMM”也最吃“常驻 kernel 内跨 task 复用”工程约束**的一份 device 身体。
>
> 快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/33` 同快照）。
> **纯源码走读（资料分析），无新实验**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **同一块 SM100 FP8 线性算子在快照里有两套 device 身体，生产接线只用了「手写裸 PTX」那套**。身体 B
   `linear_fp8_1d2d_sm100.cuh` 用 `cute::make_tiled_mma` / `local_tile` / `partition_shape_*` /
   `UMMA::tile_to_mma_shape` 一整套 CuTe 高阶惯用法写（§2），`grep` 全仓**只有定义、无任何调用者**
   （`task_header.cuh:37` 只 `#include`）；公开 pybind 名字 `linear_fp8_1d2d_sm100`
   （`runtime_kernel_wrapper_sm100.cu:1101`）实际 dispatch 到身体 A 的 `launch_linear_fp8_1d1d_sm100`
   （`:962`）。这与本专题反复出现的「已实现未启用」模式同族（`docs/16`/`docs/20`/`docs/22`/`docs/32` 等）。
2. **MLA-MTP decode 把一份完整 flash-attention（QK→softmax→PV→online-rescale→归一化）压进一个 device
   task**，并且**为「跑在 256 线程 MPK worker CTA 里」做了三类明确适配**：① 线程守卫（`tid<128`，其余线程
   返回但不缺席 `__syncthreads`）；② 命名屏障 `bar.sync 12,128`（只同步活跃半 CTA，避开 `bugfix.md` Bug 14）；
   ③ **TMEM 分配必须用 warp 0**（与同 CTA 里共驻的 FP8 CUTLASS task 保持一致，否则后续 FP8 task 的
   `tmem_allocator.allocate()` 挂起）——`docs/32` §5.3 的 `kNumSMs=1` 与 `docs/33` §6.3 的「TMEM 不
   relinquish」是同一族**为 C2 常驻让路的 device 侧事实**，本文再添两条（TMEM alloc warp 一致、named barrier
   ID 保留区）。
3. **生产布线上，MLA-MTP 是一整族（1 单卡 + 3 TP）8 个 `TaskType`**：`TASK_MLA_MTP_DECODE_SM100=269` /
   `TASK_MLA_MTP_REDUCE_SM100=270` / `TP2={287,288}` / `TP4={289,290}` / `TP8={291,292}`
   （`runtime_header.h:156-157,177-182`），由 `graph.cc` 的 op 名 dispatch 到 `register_mla_mtp_*_sm100_task`，
   前端由 `persistent_kernel.py` 的 `mla_mtp_decode_layer`（`:1778`）/`mla_mtp_reduce_layer`（`:1812`）
   与 DeepSeek-V3 builder（`models/deepseek_v3/builder.py:744/748`）生成——**TP 变体是「对 ferret standalone
   内核的机械 port」**（文件头 `:3-13`），不是 MPK 原创。

---

## 1. 定位：两条尾巴在编译栈里的坐标

| 层 | 本文覆盖 | 上游/下游 |
|---|---|---|
| 前端（`docs/14`/`docs/25`） | `PersistentKernel.mla_mtp_decode_layer` / `mla_mtp_reduce_layer` | 谁把 MTP 图拼进来 |
| task 目录（`docs/24`） | `TaskType 269/270/287–292` 家族 | 代际 ID 块 |
| 生产布线（`docs/31`/`docs/32`） | `graph.cc` op→register→`register_task_variant` | `kNumSMs=1` HACK（`docs/32` §5.3） |
| **device 身体（本文）** | **`linear_fp8_1d2d_sm100.cuh` / `mla_mtp_decode*_sm100.cuh`** | `docs/07`（Hopper）/`docs/21`（Blackwell 自动）/`docs/32`/`docs/33` |
| 运行层（`docs/06`/`docs/22` §8） | 常驻 worker 调 `_execute_task` 后进这些身体 | C1/C2/C4/C5 只在此拿 |

判据口径同 `docs/07`/`docs/29`–`docs/33`：**一个 device task 身体只贡献 C3**（单 task 可表达 / 数值正确），
C1/C2/C4/C5 由运行层拿（本文 §5 重申）。

---

## 2. 身体 B：`linear_fp8_1d2d_sm100.cuh` 的 CuTe 主循环

### 2.1 与身体 A 的差异（`docs/32` §3 的结构核对 → 逐行）

- **include 是 hopper 基础设施**（`:36-42`）：`../common/dmem_layout.cuh`、`../hopper/barrier.cuh`、
  `../hopper/smem_layout_tma.cuh`、`../hopper/tma.cuh`、`sm100_utils.cuh`、`storage.cuh`——而身体 A 用
  `_common.cuh` 家族（`docs/32` §3）。
- **签名**（`:69-90`）：`linear_fp8_1d2d_sm100_task_impl<T_, TMA_A, TMA_B, BiasTensor, TMA_OUT, MMA_M,
  MMA_N, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, bool NOBIAS, bool SplitK, NUM_AB_STAGE=8,
  NUM_ACC_STAGE=2, NUM_C_STAGE=4>`——带 `NOBIAS`/`SplitK` 两个 bool、**无 `GemmType` 模板参数**（`docs/32` §3）。
- **风格是 CuTe 高层**：`cute::make_tiled_mma`（`:143`）、`cute::SM100_MMA_MXF8F6F4_SS<T_,T_,float,
  cutlass::float_ue8m0_t,MMA_M,MMA_N,K,K>`（`:134-141`）、`cute::local_tile`（`:158/161`）、
  `cute::partition_shape_A/B`（`:164-174`）、`cute::UMMA::tile_to_mma_shape` + `Layout_K_SW128_Atom`
  （`:182-188`）；身体 A 是手写 SMEM descriptor + 裸 PTX。

### 2.2 编译期常量与 TMEM 列布局（`:96-119`）

- `TILE_K=128`、`UMMA_K=32`、`NUM_K_SUBTILES = TILE_K/UMMA_K = 4`（`:97-99`）。
- `SCALE_K = REDUCTION_SIZE / TILE_K`、`PADDED_SCALE_K = ceil4(SCALE_K)`（`:100-101`）——**per-128-K 打包
  block scale**：一个 `uint32` = 4 个 UE8M0 字节（`:103-104`）。
- `SF_BLOCK_M`/`SF_BLOCK_N`（`:105-111`）按 `kNumUTCCPAlignedElems=128`（`:49`）向上取整。
- **TMEM 列预算**（`:113-119`）：`kNumAccumTmemCols = MMA_N*NUM_ACC_STAGE`，其后依次排
  `kTmemStartColOfSFA = kNumAccumTmemCols` 与 `kTmemStartColOfSFB`，`num_tmem_columns` 三者之和。
  即 **累加器、SFA、SFB 共用同一段 TMEM 列空间**，用列偏移区分（与身体 A / `docs/33` 同构）。

### 2.3 CuTe MMA plumbing 与 SMEM layout（`:121-231`）

- `warp_idx = cutlass::canonical_warp_idx_sync()`（`:121`）、`lane_idx = kernel::lane_id()`（`:122`）。
- `mA`/`mB` 是**坐标张量**（`make_coord_tensor`），形状 `(OUTPUT_SIZE, REDUCTION_SIZE)` 与
  `(BATCH_SIZE, REDUCTION_SIZE)`（`:150-156`）；`gA`/`gB` 用 `local_tile` + `Step<1,X,1>`/`Step<X,1,1>`
  切出本 CTA 的 tile（`:158-162`）。
- `sA/sB` 用 `UMMA::tile_to_mma_shape(Layout_K_SW128_Atom<T_>, mma_shape_{A,B})`（`:182-185`）——**K-major
  SW128 swizzle**，与身体 A 的 `make_umma_desc<Major::K,...,128>` 指向同一物理布局。
- `sSFA/sSFB` layout 是 `(SF_BLOCK_{M,N}, NUM_AB_STAGE)` 的行主序（`:206-212`）。
- `SharedStorage = PipedSharedStorageWithSF<...>`（`:214-224`），定义在 `storage.cuh:69-100`——在身体 A 的
  `PipedSharedStorage`（`:33-55`）基础上**多两个 scale 缓冲 `SFA`/`SFB` 和 `tmem_columns` 字段**。
- SMEM 128 B 对齐后重解释为 `SharedStorage`（`:226-231`）；`smem_sfa(stage)`/`smem_sfb(stage)`
  是 addresses 偏移 lambda（`:234-240`）。

### 2.4 warp5：loader（`:350-446`）

- 外层 `m_tile × n_tile` 遍历 `size<3>(tCgA)`/`size<3>(tCgB)`，内层 `k_tile`（`:352-366`）。
- **skip-lookahead 模式**：`try_wait_barrier(ab_empty…)` peek，只在 peek 失败时真正 wait（`:362-377`）；
  每轮预 peek 下一 stage（`:434-438`）——与身体 A/`docs/33` 的乐观 peek 同惯用法。
- `load_packed_scale_tile`（lambda，`:242-255`）逐 lane strided 读 global 的 packed scale
  （`src[global_row*PADDED_SCALE_K + k_tile]`）到 SMEM。
- **UTCCP 前置 warp 转置**：`detail::utccp_required_smem_warp_transpose`（`:53-65`，按 `128` 元素块
  调用 `:397-407`），实现与 `docs/33` 的 `fp8_utccp_warp_transpose` **逐行同构**（`i ^ (lane>>3)` 的
  4×32↔32×4 重排）。
- `fence_view_async_shared()` 后 `elect_one_sync` 发 **两条 `tma_a.tma_cp_async` / `tma_b.tma_cp_async`**
  （`:426-431`），transaction bytes 由 `set_barrier_transaction_bytes(ab_full…)` 预置（`:422-424`）。

### 2.5 warp4：MMA（`:450-609`）

- `tmem_allocation_result_barrier.arrive_and_wait()` 后取 `tmem_base_ptr`（`:452-453`）。
- `cute::UMMA::make_instr_desc_block_scaled<...>`（`:457-465`）生成 idesc；`make_sf_desc(nullptr)`
  （`:467`）、`make_umma_desc<Major::K, OUTPUT_ATOM_SIZE/MMA_N, TILE_K, 128>`（`:469-475`）生成 A/B 描述符。
- **描述符按 stage 预存到 lane 再 `__shfl_sync` 广播**：`a_desc_lo/b_desc_lo` 只在前 `NUM_AB_STAGE` 个 lane
  持有（`:477-485`），循环里 `__shfl_sync(0xffffffff, …, smem_rd_buffer)` 取用（`:554-557`）——身体 A
  /`docs/33` 的同款优化。
- **每 K-tile 三件事**：① `SM100_UTCCP_4x32dp128bit_1cta::copy` 把 SFA/SFB 从 SMEM 搬进 TMEM 列
  （`:530-550`，用 `replace_smem_desc_addr` 改地址）；② 每 K-tile 发 **4 条** `mma_issue_t::fma`
  （`NUM_K_SUBTILES=4`，`:559-588`），`accumulate` 只在每 tile 首条为 false（`:514/586`），sfa/sfb id 随
  k_sub 走（`make_runtime_instr_desc_with_sf_id`，`:564-566`）；③ `cutlass::arch::umma_arrive(&ab_empty…)`
  释放 A/B stage（`:590-591`）。tile 收尾 `umma_arrive(&acc_full…)`（`:604-605`）。

### 2.6 epilogue warps（`warp_idx<4`，`:613-718`）

- warp0 里 `tmem_allocator.allocate(num_tmem_columns, …)`（`:614-616`）——**用 warp0**（与 §3.5 的约束一致）。
- `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, …)`（`:626-628`）+ `partition_S/partition_D`（`:630-637`）
  把 TMEM 累加器搬进寄存器。
- 每 tile：wait `acc_full` → `copy` 到寄存器 → `epilogue_wg_barrier.arrive_and_wait()` → `elect_one`
  `arrive(acc_empty)`（`:651-661`）→ `NumericConverter` 转 BF16 → 写 SMEM `sC_epi`（`:663-670`）。
- **`!NOBIAS` 时逐元素加 bias**（`:672-691`）：从 SMEM 读回 FP32，加 `mBias(global_row, global_col)`
  再转换写回，夹两次 `epilogue_wg_barrier`。
- `tma_store_fence` 后按 `SplitK` 选 **`tma_reduce_add_async`** 或 `tma_store_async`（`:696-709`），
  `tma_store_wait<NUM_C_STAGE-1>` 做 C 环深度；函数尾 `tma_store_wait<0>`（`:715-717`）。
- 最后 `__syncthreads()` + warp0 `tmem_allocator.free`（`:720-724`）——注意**这里 free 且带 `__syncthreads`**，
  与 `docs/33` §6.3 的「megakernel 里不 relinquish」是**不同文件的不同选择**（身体 B 未被生产接线，故未做该适配）。

### 2.7 静态事实：无调用者

全仓 `grep linear_fp8_1d2d_sm100_task_impl` **只命中定义处**（`:85`）；`task_header.cuh:37` 只
`#include "linear_fp8_1d2d_sm100.cuh"`。⇒ 身体 B 在快照里是**未被接线的第二实现**（同 `docs/16`/`docs/20`/
`docs/22`/`docs/32`）。**仅记录静态事实、不判对错**。

（附带一条「注释 vs 现实」的观察：`task_header.cuh:43-44` 注释称 `sm100_ptx.cuh` 「必须在
`mla_mtp_decode_sm100.cuh` 之前包含」，但直接 `#include "sm100_ptx.cuh"` 实际在 `:59`（`mla_mtp` 之后）；
靠 `mla_mtp_decode_sm100.cuh:66` 自己 `#include "sm100_ptx.cuh"` + `#pragma once` 兜住。**注释已滞后**。）

---

## 3. MLA-MTP decode 家族：投机解码的 MLA attention task

### 3.1 文件清单与常量

| 文件 | 行数 | `NUM_HEADS` | 主函数 | reduce |
|---|---|---|---|---|
| `mla_mtp_decode_sm100.cuh` | 748 | **128** | `mla_mtp_decode_sm100_task_impl`（`:77-671`） | `mla_mtp_reduce_sm100_task_impl`（`:676-746`） |
| `mla_mtp_decode_tp2_sm100.cuh` | 796 | 64 | `mla_mtp_tp2_main`（`:132`） | `mla_mtp_tp2_reduce`（`:733`） |
| `mla_mtp_decode_tp4_sm100.cuh` | 787 | 32 | `mla_mtp_tp4_main`（`:121`） | `mla_mtp_tp4_reduce`（`:726`） |
| `mla_mtp_decode_tp8_sm100.cuh` | 775 | 16 | `mla_mtp_tp8_main`（`:118`） | `mla_mtp_tp8_reduce`（`:712`） |

公共常量（`mla_mtp_decode_sm100.cuh:36-62`）：`D_K=576`、`D_V=512`、`TILE_S=128`、`BK=64`、`MMA_K=16`、
`K_ITERS=D_K/BK=9`、`V_CHUNKS=D_V/BK=8`、`TB=128`（活跃线程数）、`NUM_QK_STAGES=5`、`NUM_PV_STAGES=3`、
`TILE_BYTES=128*64*2=16 KiB`、`MTP_SMEM_SIZE=NUM_QK_STAGES*2*TILE_BYTES=160 KiB`（`:60`）；
reduce 侧 `RD_DV=4`、`RD_TB=512`、`MAX_SK=32`（`:54-57`，单卡版）。

### 3.2 单卡主函数 `mla_mtp_decode_sm100_task_impl`（`:77-671`）

数学：一个 CTA 处理 `(gi=head_group, si=split_idx, bi=batch)`（`blockIdx.x` 拆成 `gi/si`、`blockIdx.y`
是 `bi`，由参数传入 `:88-90`），把 `M=128` 行映射为 **`Q_LEN` 个 query × `128/Q_LEN` 个头**
（文件头 `:17-19`）。KV 一次从 SMEM 载入、被所有 query 共享。

- **三类早退（都是全程统一、不与 `__syncthreads` 冲突）**：`gi>=num_head_groups`（`:103`）、
  **dual-dispatch gate `Q_LEN>8` 直接返回**（`:110-112`，与 `mla_prefill` 分工）、空 split（`:118-120`）。
- **线程守卫**：`active = tid < TB(128)`；MPK worker 有 256 线程，**剩余线程不能提前 `return`**（会缺席
  `__syncthreads`，`:96-98` 注释明确）。
- TMEM 分配在 **`wid==1`**（`:146-152` 用 `tcgen05.alloc.cta_group::1`）——**注意单卡版用 warp1，而
  TP 变体改成 warp0**（见 §3.5，这是常驻复用踩坑后的修正）。
- **QK 阶段**（`:203-279`）：每 tile 先重 init QK barrier（`:205-212`）。producer `wid0` 逐 `ki=0..8`：
  等 `mma_bar`、`mbar_tx(TILE_BYTES + hpb_bytes*Q_LEN)`、发 **3D TMA 载 K tile**（`:228-236`）+ 每
  query 一条 TMA 载 Q 行（`:237-252`）；consumer `wid1` 逐 `ki`：等 `tma_bar`、
  `tcgen05.fence::after_thread_sync`、对 `k2=0..BK/MMA_K-1` 发 `tcgen05_mma`（`idesc_qk`，`:267-272`），
  `tcgen05_commit` 到 `mma_bar`；末尾 `tcgen05_commit(mainloop_bar)`（`:275`）。
- **Softmax 阶段**（`:281-502`）：`mbar_wait(mainloop_bar)` 等 QK 完事。
  - Pass1（`:327-358`）：按 `c<effective_len` 从 TMEM `tcgen05.ld` 读 S 分数，`t16[i]*ss` 求 `tile_max`；
    `effective_len` 处理 **causal（Q_LEN>1 时 `causal_limit=kv_len-Q_LEN+q_idx+1`，`:304-315`）**。
  - Pass2（`:360-438`）：算 `exp(t*ss - tile_max)` 累加 `tile_sum`，**手工打包 2×bf16 成 `uint32` 并用
    `st.shared.v4.b32` + swizzle 写进 P0/P1 SMEM**（`:398-436`，`swizzled = (byte_off & ~0xF) ^
    ((tid&7)<<4) | (byte_off & 0xF)`）。
  - **Online rescale**（`:441-443`）：`nm=max(row_max,tile_max)`、`corr=exp(row_max-nm)`、
    `ts=tile_sum*exp(tile_max-nm)`；`tile>t0` 时把 TMEM 里已累积的 `O[128:511]` **统一乘 `corr`**
    （`:446-498`，`tcgen05.ld`→乘→`tcgen05.st`）；最后更新 `row_max/row_sum`（`:500-501`）。
- **PV 阶段**（`:503-618`）：`pv_acc_base` 标记本 tile 是否首 tile（决定 PV 是否从 0 开始累加）。
  producer `wid0` 逐 `vc=1..7` 发 V-chunk TMA（`:509-528`）；consumer `wid1` 逐 `vc=0..7` 对
  `k1=0..1`、`k2=0..3` 发 `tcgen05_mma`（`idesc_pv`，`:529-556`）；末尾把**保存的 `O[0:127]` 用 `corr`
  合并回 TMEM**（`:563-618`，即把上一个 tile 的 `o_save` 重新加权加进新 PV 结果）。
- **Epilogue**（`:621-663`）：`inv=1/row_sum`，逐 `vc` 从 TMEM 读 O、乘 `inv` 转 bf16 写到 `Oout`
  （`:622-658`）；`La[block_linear*128+tid] = log(max(row_sum,1e-30)) + row_max`（`:660-662`，log-sum-exp）。
  最后 `__syncthreads` + `wid0` `tcgen05.dealloc`（`:665-670`）。

### 3.3 reduce（`:676-746`）

`mla_mtp_reduce_sm100_task_impl<NUM_THREADS_=512>`：把 `sk` 个 split 的 `Oa`/`La` 合并。读 `La` 到 SMEM
（`:710-716`），对 `sk` 个 split 做 **log-sum-exp 归并**（`:726-734`），再按 split 加权累加 `Oa` 写 `O`
（`:736-745`）。`NUM_THREADS_` 在 MPK 里实例化为 **256**（`register_mla_mtp_reduce_sm100_task` 发
`mla_mtp_reduce_sm100_task_impl<256>`，见 §4），standalone 用 512。

### 3.4 TP 变体差异（tp2/tp4/tp8）

三份是**同骨架**（QK/softmax/PV/epilogue 四段一致），差异集中在：

- **`NUM_HEADS` = 64/32/16**（tp2 `:40` / tp4 `:29` / tp8 `:26`），即每 rank 的本地头数；Q 索引/边界检查
  用 `NUM_HEADS`（tp2 `:299`）而非单卡版的运行期 `local_num_heads`。
- **TP 版把 `bar.sync 12,128` 抽成宏 `MLA_TP_SYNC_ACTIVE()`**（tp2 `:38` / tp4 `:27` / tp8 `:24`），
  **替换掉单卡版的 `__syncthreads()`**——原因见 §3.5。
- **query 按 `qpg` 分组**：`qpg`（queries per group）参数，`num_groups = ceil(Q_LEN/qpg)`（tp2 `:160`），
  tp8 因 `Q_LEN` 已 pad 成 `qpg` 的倍数用 `Q_LEN/qpg`（tp8 `:146`）。
- **PV 分半流水**：tp4 用 `PV_CHUNKS = V_CHUNKS/2 = 4` 分两个 half 处理（tp4 `:149`），tp2/tp8 用整
  `V_CHUNKS=8`。
- **reduce 的 `RD_DV` 变小**：TP 版 `RD_DV=2`（tp2 `:61`）、`RD_TB=256`（tp2 `:62`），单卡版 `RD_DV=4`/
  `RD_TB=512`。
- 文件头都注明是**对 ferret standalone 内核的机械 port**（`__global__`→`__device__`、`blockIdx`→参数、
  `__grid_constant__` CUtensorMap→指针、**删掉 `cudaTriggerProgrammaticLaunchCompletion()`（MPK 不支持
  PDL）**、加线程守卫；`tp2:3-13`）。

### 3.5 「跑在常驻 256 线程 worker CTA 里」的三类工程约束（headline 2 的展开）

1. **线程守卫 + active-half 屏障**。MPK worker CTA 是 256 线程，MLA 内核只用 128。单卡版靠
   `active=tid<128` + 全程参与 `__syncthreads`（`:96-98`）；**TP 版更狠**——文件头 `tp2:23-38` 说明：
   在 SM100 的 Independent Thread Scheduling 下，**提前 `return` 的那半 CTA 会在 worker 外层循环里
   「漂移领先」，使后续 `__syncthreads()`（barrier 0）在 warps 4–7 缺席时触发，导致它们静默跳过下一个
   task 的身体、进而挂起（`bugfix.md` Bug 14）**。故 TP 版改用 **`bar.sync 12, 128`**——只同步活跃的
   128 线程（bar 12 落在 CUTLASS 用户屏障区 `FirstUserBarrier=8` 之后，prefill 用 8–11；ID 1–7 是 CUTLASS
   保留区，误用会留脏状态毒死同 CTA 后续 FP8 线性 task，`bugfix.md` Bug 19，用 compute-sanitizer
   `synccheck` 定位）。
2. **TMEM 分配必须用 warp 0**。TP 版 `:190-195` 注释明确：CuTe 的 TMEM allocator 要求
   **`tcgen05.alloc/dealloc` 由同一 warp 在所有共驻于持久化 kernel 的 task 里发出**；FP8 CUTLASS 家族
   （`linear_fp8`/`moe_linear`/`fp8_group_gemm`）都用 warp0，**若 MLA 用别的 warp（如 warp1），同 worker
   上后续 FP8 task 会挂在其 `tmem_allocator.allocate()` 里**。故 TP 版统一用 warp0（单卡版此处仍是
   warp1，见 §3.2——这是被踩坑后修正的痕迹）。
3. **dual-dispatch gate**。单卡版 `:110-112` 与 TP reduce `tp2:746` 都有 `Q_LEN>8 return`——因为
   同一 attention 步可能同时登记了 `mla_prefill` 与 `mla_mtp_decode`，**预填充态（Q_LEN>8）交给 prefill
   内核，decode 内核跳过**，避免用陈旧/部分 `Oa`/`La` 覆盖 `attn_out`（`tp2:146-152` 注释）。

---

## 4. 生产布线

- **op→task 名 dispatch**（`src/kernel/graph.cc`）：`mla_mtp_decode_sm100`→`register_mla_mtp_decode_sm100_task`
  →`task_config=(2,2,TASK_MLA_MTP_DECODE_SM100,…)`（`:792-796`）；`mla_mtp_reduce_sm100`
  →`(2,1,TASK_MLA_MTP_REDUCE_SM100)`（`:797-802`）；TP2/4/8 的 decode/reduce 各一对（`:805-840`）。
- **`TaskType`**（`runtime_header.h`）：`TASK_MLA_MTP_DECODE_SM100=269`/`REDUCE=270`（`:156-157`）；
  `TP2={287,288}`/`TP4={289,290}`/`TP8={291,292}`（`:177-182`）。同族还有
  `TASK_MLA_DECODE_SM100=266`/`REDUCE=267`/`PREFILL=268`（`:153-155`）与 `PREFILL_TP8=295`（`:189`）。
- **register 生成**（`src/kernel/task_register.cc`）：`register_mla_mtp_decode_sm100_task`（`:4405-4471`）
  按 `single_tile=(tps==1)` 选 `<true>/<false>` 实例（`:4418-4446`），填 **softmax scale `ss`** 为
  DeepSeek-V3 YARN 调整值 `1/sqrt(192)*mscale^2 ≈ 0.13525`（`:4453-4460`，**注意不是 `1/sqrt(576)`**，
  注释 `:4454-4455` 解释 `d_k=576` 是吸收后的 latent dim、点积缩放维是 192）；`gi/si/bi` 从
  `task_metadata.request_id`/`kv_idx`/`merge_task_offset` 取（`:4466-4468`）。reduce 发
  `mla_mtp_reduce_sm100_task_impl<256>`（`:4248/4495` 同族）。
- **前端**（`python/mirage/mpk/persistent_kernel.py`）：`mla_mtp_decode_layer`（`:1778`）派
  `hpb=128//q_len`、`num_head_groups`、`num_splits=ceil(kv_len/128)`、`grid=(num_splits,num_head_groups,
  max_batched_requests)`、`block_dim=(128,1,1)`；`mla_mtp_reduce_layer`（`:1812`）派
  `rd_dv=2`（注释 `:1789-1793` 说明 `rd_dv=4` 会减半 block 数但实测更慢）。TP 变体
  `mla_mtp_decode_tp2/4/8_layer`（`:1892` 起）。DeepSeek-V3 builder 在 `models/deepseek_v3/builder.py`
  `:744/748`（及 `:1389/1393`）调用。
- **单测**：`tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_mla_mtp.cu:29/50`（单卡
  decode/reduce）与 `sm100_mla_mtp_decode/test_tp{2,4,8}_wrapper.cu`（TP standalone port 验证器，与
  `docs/31` §9 的 `test_tp*_wrapper` 同族——**是「机械 port 验证器」，不在 pytest 收集内**）。

---

## 5. 判据映射（C1–C5）

同 `docs/07`/`docs/29`–`docs/33`：**一个 device task 身体只贡献 C3**。

- **贡献 C3**：把「SM100 block-scaled FP8 线性（身体 B）」与「DeepSeek-V3 MTP 投机解码的 MLA
  attention + reduce」编成可被 `register_task` 生成、可被常驻 worker 调度的 device 函数。
- **不贡献 C1/C2**：单次 launch / 常驻由运行层拿（`docs/06`/`docs/22` §8）。
- **不贡献 C4/C5**：task 内 `mbarrier`/`NamedBarrier`/`bar.sync 12,128` 是 **CTA 内跨 warp / 跨活跃半
  CTA 同步**，不是跨 task/跨 SM 的去中心化调度。
- **与常驻有关的 device 侧事实（本文新增两条，与 `docs/32` §5.3、`docs/33` §6.3 同族）**：
  ① **TMEM alloc/dealloc 必须与同 CTA 其它 task 用同一 warp**（TP 版统一 warp0）；② **命名屏障 ID 必须
  落在 CUTLASS 用户区**（`bar 12`），否则脏状态毒死后续 task。二者都是「多个 task 共驻一个常驻 CTA」
  逼出来的适配。

---

## 6. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  「无调用者」是**静态 `grep`** 结论，不构成运行时判定。
- **`bugfix.md` 不在快照里**：TP 版注释反复引用的 `bugfix.md` Bug 14/Bug 19/Bug 20 在 `f9eb70c` 快照中
  **不存在**（只 `grep` 到引用），故本文只转述**源码注释自陈的因果**，未核对原始 bug 报告内容。
- **单卡版 vs TP 版的差异归因是「凭注释 + 代码推断」**：单卡版用 `wid1` 分配 TMEM、用 `__syncthreads`，
  TP 版改成 `wid0` + `bar.sync 12,128`；本文据源码注释把它们归到「常驻复用踩坑后的修正」，但**没有
  commit 历史/测试佐证**。
- **未与生产端到端对跑**：SM100 路径本机不具备；TP 变体是对 ferret standalone 的机械 port，本文只核了
  port 后的结构，未对照原始 ferret 源码逐行。
- **未走读的相邻层**：`mla_prefill_sm100.cuh`/`mla_prefill_tp8_sm100.cuh`、`mla_decode_sm100.cuh`、
  `mla_kv_cache_gather*`、`mla_sm100_2sm.cuh`、注意力之外的 SM100 task 家族仍未逐行（留待后续）。

---

## 7. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/33` 同快照）。
- **身体 B**：`include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_1d2d_sm100.cuh`（727 行）——
  `detail::utccp_required_smem_warp_transpose :53-65`；签名 `:69-90`；常量/TMEM 布局 `:96-119`；
  CuTe plumbing `:121-231`；warp5 loader `:350-446`；warp4 MMA `:450-609`；epilogue `:613-718`；
  `tmem free :720-724`。存储 `storage.cuh:69-100`（`PipedSharedStorageWithSF`）。
- **身体 B 用的 SM100 原语**：`blackwell/sm100_utils.cuh` —— `make_smem_desc :30-49`、`make_sf_desc :51-63`、
  `replace_smem_desc_addr :65-73`、`advance_umma_desc_lo :75-86`、`make_umma_desc :88-114`、
  `make_runtime_instr_desc_with_sf_id :116-123`、`SM100_MMA_MXF8F6F4_SS::fma :127-150`（与 `docs/33` 同文件）。
- **身体 B 的「未接线」证据**：`grep linear_fp8_1d2d_sm100_task_impl` 仅命中定义；`task_header.cuh:37`
  只 `#include`；公开 pybind `linear_fp8_1d2d_sm100`（`tests/runtime_python/blackwell/sm100_linear_fp8/
  runtime_kernel_wrapper_sm100.cu:1101`）实走 `launch_linear_fp8_1d1d_sm100`（`:962`）/`_splitk`（`:977`）。
- **MLA-MTP**：`blackwell/mla_mtp_decode_sm100.cuh`（748 行；常量 `:36-62`；主函数 `:77-671`；reduce
  `:676-746`）；`mla_mtp_decode_tp2_sm100.cuh`（`NUM_HEADS=64` `:40`；`MLA_TP_SYNC_ACTIVE` `:38`；
  main `:132`；TMEM warp 约束注释 `:190-195`）；`mla_mtp_decode_tp4_sm100.cuh`（`NUM_HEADS=32` `:29`；
  `PV_CHUNKS` `:149`）；`mla_mtp_decode_tp8_sm100.cuh`（`NUM_HEADS=16` `:26`）；PTX 助手
  `blackwell/sm100_ptx.cuh`（`elect_sync :27`、`mbar_init :36`、`mbar_wait :41`、`mbar_tx :51`、
  `make_desc :63`、`tcgen05_mma :69`、`tcgen05_commit :82`）。
- **生产**：`src/kernel/graph.cc`（dispatch `:791-840`）；`src/kernel/task_register.cc`
  （`register_mla_mtp_decode_sm100_task :4405-4471`、`..._reduce :4473-4508`、TP2 `:5192/5252`、
  TP4 `:5287/5349`、TP8 `:5384/5447`）；`include/mirage/persistent_kernel/runtime_header.h`
  （`TaskType` `:153-157,177-182,189`）；`python/mirage/mpk/persistent_kernel.py`
  （`mla_mtp_decode_layer :1778`、`mla_mtp_reduce_layer :1812`、TP `:1892` 起）；
  `python/mirage/mpk/models/deepseek_v3/builder.py:744/748/1389/1393`。
- **单测**：`tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_mla_mtp.cu:29/50`；
  `tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_tp{2,4,8}_wrapper.cu`（standalone port）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （device task 内部实现质量，同 `docs/28`–`docs/33`）。
- 同专题相关篇：`docs/32`（本文补其 §7 两条尾巴）、`docs/33`（补其 §11「身体 B 未处理」）、
  `docs/07`（Hopper 单 task device 流水）、`docs/21`（Blackwell transpiler）、`docs/31`（wrapper 家族）、
  `docs/24`（`TaskType` 目录）、`docs/22` §8（C 判据总映射）。
- 交叉专题：DeepGEMM MoE 前向/反向正确性验证见 `../mega-moe/`（本专题不重复）。
