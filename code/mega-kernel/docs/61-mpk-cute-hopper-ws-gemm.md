# MPK Hopper CuTe 权重驻留 GEMM device 家族走读：`tasks/cute/hopper/` 的「三份身体、一份活」

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/60` 同快照）；
> CUTLASS 侧走 MPK 钉死的 submodule `deps/cutlass` commit `f3fde583`
> （`~/mirage_mpk_build/deps/cutlass`，这条 checkout 有完整 submodule）。
> `docs/07` 只在一行旁注里提过它（`gemm_ws_mpk.cuh` 注册为 `TASK_LINEAR_CUTLASS_HOPPER`）、
> `docs/30` §5 走的是它的**单测 wrapper**（`tests/runtime_python/cute/hopper/gemm.cu`）、
> `docs/58` 只借它一行确认「`smem_tma` 被 `cute/hopper/gemm_ws_mpk.cuh` 共用」——
> **这条 device 家族本身（6 个文件、2346 行）从未被逐行走读**。本文把它当作一层打开。
> **纯源码走读 + 1 个真机实验**：它是 **Hopper/sm90a**（不是 SM100 专属），故其 standalone 单测
> 入口 `linear_mpk` 在目标卡 H100 上**能真跑**——本文补齐快照自身缺失的编译配置后把它跑通
> （新 `harness/mpk_cute_hopper_ws.py` + `harness/mpk_cute_hopper_ws.sh` + 输出
> `harness/mpk_cute_hopper_ws.out.txt`）。所有行号为快照静态事实。

---

## 0. 三条 headline

**(A)（headline）这是一条「后端已接、前端未发」的支线。** `TASK_LINEAR_CUTLASS_HOPPER=157`（无残差）
与 `TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER=158`（含残差）在**后端全链路打通**——`runtime_header.h:120-121`
的枚举、`task_register.cc:1754` 的 `register_linear_cutlass_hopper_task`（**JIT 生成**调用串）、
`graph.cc:585-595` 的 `name=="linear_cutlass_hopper"` 分支、`tma.cuh:659`/`:1531` 的 TMA 描述符构造、
两份 `task_header.cuh` 的 `#include`、`profiler_persistent.py:157-158` 的 task 名表**全都在**。
但 **MPK 前端 `persistent_kernel.py` 里两处 `register_task(tb_graph, "linear_cutlass_hopper")` /
`"linear_cutlass_with_residual_hopper"` 都被注释**（`:2441`/`:2479`，同位置改发
`linear_swapAB_hopper`）——**没有任何 builder 会发出这两个 task 名**（全仓 grep 只有后端与注释）。
因此这条支线的**唯一活入口是 standalone 单测** `tests/runtime_python/cute/hopper/gemm.cu` 注册的
`m.def("linear_mpk", ...)`。这是本专题反复出现的「有实现未接线」族（`docs/17`/`docs/49`/`docs/60` §C）
在 Hopper GEMM 上的又一实例，且**比通常的「死代码」更进一步：它同时被前端注释、却仍被两份 task_header
无条件 include**。

**(B)（headline）家族里**三份 kernel 身体**，只有一份是活的；而那份活身体**又把 CUTLASS 的流水抄了一遍**。**
目录 `include/mirage/persistent_kernel/tasks/cute/hopper/` 含三段并列的 warp-specialized GEMM 主体：

| 身体 | 文件 | 谁调用 | 状态 |
|---|---|---|---|
| 非协同版 | `gemm_ws.cuh`（`gemm_kernel_tma_warp_specialized`，274 行） | 单测中**被注释**的 `linear_kernel_hopper_cute_wrapper`（`gemm.cu:49-58`） | 死 |
| 协同版 | `gemm_ws_cooperative.cuh`（`..._cooperative`，428 行） | **无任何调用点**（只被两份 `task_header.cuh` include） | 死 |
| **MPK 版** | `gemm_ws_mpk.cuh`（`linear_cutlass_ws_hopper`，562 行） | **活**：单测 `linear_mpk` + 后端 JIT 注册 | 活 |

前两份是**近乎逐字照搬的 CUTLASS `sm90_gemm_tma_warpspecialized` 内核**（`gemm_ws.cuh` 调
`collective_mainloop.load/mma` + `collective_epilogue.store`，`gemm_ws.cuh:210/231/252`；协同版还带 CLC
调度与 warpgroup 寄存器重分配）。**第三份 `gemm_ws_mpk.cuh` 却不再调用 collectives 的方法**——它从
`CollectiveMainloop` 只借**类型别名与常量**（`SmemLayoutA/B`、`TiledMma`、`MainloopPipeline`、
`TensorStorage`、`NUM_STAGES`、`K_PIPE_MMAS`、`ClusterShape`、`TileShape`），把 TMA 取数、wgmma 主循环、
epilogue **全部在 kernel 体内又手写了一遍**（`:264-290` / `:299-424` / `:452-552`，与
`mma_tma_ws_mainloop.cuh:284-311`/`:434-466`、`epilogue.cuh:83+` 是同一段流水的**第三份拷贝**）。
⇒ 母题「同一套硬件惯用法在仓库里实现两遍」（`docs/20` §5、`docs/57`）在这里是**「实现三遍、第三遍抄」**。

**(C)（headline）在目标卡 H100/sm90a 上把 `linear_mpk` 真跑通了；顺带暴露「快照自身坏了两处」。**
本机（目标卡 H100/sm90 `sm_90a`）能编译并运行该单测：`B∈{8,16}`、`O=64`、`K=4096` **全部 `allclose`**
（`max_rel≈2.4e-3/2.0e-3`）。真机实测（新 harness，事件计时 200 次）：**单 CTA** 的 `linear_mpk`
`~0.043 ms/次`、读带宽 **~14–15 GB/s**（`B=8` 0.0946 TFLOPs / `B=16` 0.1950 TFLOPs）——这是
「一 task = 一 CTA」的**能力/正确性**验证，**不是**性能优化版（单 SM 读 ~0.5 MB 权重 + 激活，带宽必然低，
对上 `docs/07` 的「薄 task」定位）。实验同时钉死两个「快照坏了」的事实：
**(1)** 单测的 `setup.py` 的 `include_dirs` **缺 `include/mirage/persistent_kernel`**（只有
`tasks/cute/hopper`、`include`、cutlass），故 `common_header.cuh:20` 的 `#include "runtime_header.h"`
**找不到头文件、as-shipped 直接编译失败**——本文不修改快照，改用 `torch.utils.cpp_extension.load`
显式补上 include 路径与 `-lcuda` 才构建成功（`harness/mpk_cute_hopper_ws.py:36-56`）；
**(2)** 单测自带的 benchmark（`gemm.cu:423-424`）把 `WARMUP_RUNS=0`、`BENCHMARK_RUNS=1`，**单次 launch
无 warmup 的 event 计时**（打印 `Average: 0.012 ms`）与本文的稳态计时（~0.043 ms）差了 3.5×，
即**快照自报数字不可作数**（同 `docs/09` 对 MPK 自报延迟的保留态度）。

---

## 1. 定位：这个家族在仓库里的五层坐标

`tasks/cute/hopper/` 是本专题走读到的**第一个「CUTLASS/CuTe 包装」的 GEMM device 家族**
（`docs/07` 走的 `tasks/hopper/linear_hopper.cuh` 是**手写** wgmma/TMA；`docs/32`/`docs/34` 走的
`tasks/blackwell/linear_fp8_*` 是**手写裸 PTX 的 SM100**）。它的定位是「MPK 把 CUTLASS 的
warp-specialized GEMM **当模板搬进来**、再用 MPK 自己的 TMA 封装喂数」。

| 层 | 位置 | 事实 |
|---|---|---|
| 文件 | `include/mirage/persistent_kernel/tasks/cute/hopper/{kernel_traits,gemm_ws,gemm_ws_cooperative,gemm_ws_mpk,mma_tma_ws_mainloop,epilogue}.cuh` | 203/274/428/562/492/387 行，共 2346 行 |
| include | `tasks/hopper/task_header.cuh:13-15`、`tasks/blackwell/task_header.cuh:11-13` | **三份身体都被 Hopper 与 Blackwell 两份 task_header 无条件 include**（名字带 `cute/hopper` 却被 blackwell include，又一例「目录/后缀 ≠ 专属」，同 `docs/55`/`docs/58`） |
| `TaskType` | `runtime_header.h:120-121` | `TASK_LINEAR_CUTLASS_HOPPER=157`、`TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER=158` |
| register | `task_register.cc:1754-1922` | `register_linear_cutlass_hopper_task`：从 bgraph 取 `batch_size/output_size/reduction_size/output_stride`，**JIT 生成** `linear_cutlass_ws_hopper<...>` 调用串（`:1903-1915`） |
| op 映射 | `graph.cc:585-595` | `name=="linear_cutlass_hopper"` → `(2,1,TASK_LINEAR_CUTLASS_HOPPER,variant)`；含残差版 `(3,1,…WITH_RESIDUAL…)` |
| TMA 构造 | `tma.cuh:659-...` / `tma.cuh:1531-...` | 为 `157/158` 构造 A、B 两个 `tma_2d` 描述符（`SMEM_COL_=64`、`B=M=S=3`） |
| 前端 | `persistent_kernel.py:2441/2479` | **两处 `register_task(..., "linear_cutlass_hopper")` 被注释**，平行分支改发 `linear_swapAB_hopper` |
| 单测 | `tests/runtime_python/cute/hopper/gemm.cu` + `setup.py` + `test_matmul_ws_mpk.py` | 唯一活入口；注册 `m.def("linear_mpk", ...)`（`:554-556`） |
| profiler | `python/mirage/mpk/profiler_persistent.py:157-158` | task 号 → 名表 |

关键结论已在 §0(A)：**后端六层全通、前端一层注释**。所以本文的实验只能走单测。

---

## 2. 三份身体逐行

### 2.1 `kernel_traits.cuh`：一份 traits 同时喂三份身体

`MMAKernelTraits`（`:56-201`）是家族唯一的「配置中心」，模板参数为
`<DataType, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, GmemLayout A/B/C/D, NUM_WARPS, M, N, K, ProblemShape,
O_STRIDE, NUM_STAGES=3, IS_SWAPAB>`。三份身体都用它。要点：

- **tile/ISA**：`TileShape_MNK = Shape<M,N,K>`（`:77`），`TiledMma` 走 CUTLASS 的
  `GMMA::ss_op_selector`（`:139-146`）——即 Hopper 的 **wgmma**（Hopper 上 `ss_op_selector` 落
  `SM90_64xNx16_F32BF16BF16_SS` 一族），不是 SM100 的 tcgen05。
- **`SwapAB` 与「协同判据」**：`IS_SWAPAB`（`:89`/`:200`）决定 A/B 语义互换；
  `IsCooperative = size<0>(TileShape)!=Int<64>`（`:119`）、`AtomLayoutMNK` 随之在
  `_2,_1,_1`/`_1,_1,_1` 间切（`:121-123`）——即**同一份 traits 用 tile-M 是否 64 区分
  「非协同单 wgmma」与「协同双 wgmma」**。
- **scheduler/pipeline**：`TileScheduler = PersistentTileSchedulerSm90`（`:125-126`，
  协同版用；注意旁边注释掉的 `enable_if_t<IsCooperative,...>` 说明它本意只给协同版）、
  `MainloopPipeline = PipelineTmaAsync<NUM_STAGES>`（`:191`）。
- `SharedStorage`（`:44-54`）只放 `smem_a/smem_b` 两个 stage 数组 + pipeline storage。

### 2.2 `gemm_ws.cuh`：CUTLASS 非协同 warp-specialized 内核（死）

`gemm_kernel_tma_warp_specialized`（`:41-269`）是 CUTLASS `sm90_gemm_tma_warpspecialized` 的**直系拷贝**：
`WarpGroupRole{Producer,Consumer}` + `ProducerWarpRole{MainloopEpilogue,Warp1..3}`（`:67-76`）、一个生产者
warpgroup 发 TMA、一个消费者 warpgroup 做 wgmma。三分支各调 collective：
producer 段 `collective_mainloop.load(...)`（`:210`）→ `load_tail`（`:224`）；
consumer 段 `collective_mainloop.mma(...)`（`:231`）→ `mma_tail`（`:241`）→
`collective_epilogue.store(...)`（`:252`）。**它只被单测里被注释的 wrapper（`gemm.cu:49-58`）引用**。

### 2.3 `gemm_ws_cooperative.cuh`：CUTLASS 协同内核（死）

`gemm_kernel_tma_warp_specialized_cooperative`（`:39-427`）是 CUTLASS 协同版拷贝，比非协同版多三件「协同」：
**(1)** `PersistentTileSchedulerSm90` 的 `work_tile_info` 循环（`:248-250`/`:277-332`），producer 与
consumer 各自 `fetch_next_work`；
**(2)** **寄存器重分配**——producer warpgroup `warpgroup_reg_dealloc<LoadRegisterRequirement>`
（`:269`）、consumer `warpgroup_reg_alloc<MmaRegisterRequirement>`（`:341`），阈值按
`HeavyRegisterPressure` 在 `(40,232)`/`(24,240)` 间切（`:116-126`），且 `static_assert` 要求
`NumMMAThreads==256` 且 `TileShape-M≥128`（`:128-133`）；
**(3)** CLC（cluster launch control）throttle pipeline（`:301-307`）与跨 split 的
`TileScheduler::fixup`（`:385`）。
**全仓无任何调用点**——只被两份 `task_header.cuh` include，是一份「编译进去但从不执行」的协同 GEMM。

### 2.4 `gemm_ws_mpk.cuh`：活的 MPK 版（主角）

`linear_cutlass_ws_hopper<...,TMAOnHost,T,BATCH,OUTPUT,REDUCTION,TMA_A,TMA_B,OUTPUT_STRIDE,WITH_RESIDUAL>`
（`:48-559`）是唯一被生产/单测使用的身体。与 §2.2/§2.3 的关键差异都指向「**它把流水从 collectives
里搬回 kernel 体内**」：

- **SMEM 基址对齐**：直接用 `aligned_smem = (smem+1023)/1024*1024`（`:98-99`）——即 `docs/58` §C 那条
  「TMA swizzle 须 1024 B 基址对齐」的**同一处生产兜底**（本文家族亦记一遍）。
- **TMA 用 MPK 自己的封装**：`smem_tma<bf16,3,3,3,...>` 建 `InputSmem`（batch 维）与
  `WeightSmem`（output 维，`:132-148`），k 循环里手动 `tma_a.tma_cp_async`/`tma_b.tma_cp_async`
  （`:284-286`）——**不经过** `CollectiveMainloop::load`。tile 常量：`INPUT/WEIGHT_TMA_TILE_SIZE=64`、
  `OUTPUT_ATOM_SIZE=64`、`TILE_SIZE=128`（`:121-125`）。
- **wgmma 主循环手写**：Prologue GMMA + 稳态 `cute::gemm`（`:336-424`）——与
  `mma_tma_ws_mainloop.cuh:376-468` 是**逐行同构**的另一份拷贝。
- **epilogue 内联且带 swapAB 转置**：`:452-552` 手写 OOB 谓词 + `ThreadEpilogueOp`，
  且仅当 `Has_SwapAB_v<CollectiveMainloop>` 为真时对输出做转置（注释 `// NOTE(Yu): stride_c and
  stride_d are global view ... tranpose stride_c and stride_d`，`:460-463`）。这正是
  `docs/20`「`linear_*_hopper` 的 swapAB 是一等公民」在 CUTLASS 路径上的对应。
- **`problem_shape` 的 A/B 交换**：前端固定 `(OUTPUT_SIZE, BATCH_SIZE, REDUCTION_SIZE, 1)`
  （`:241-244`）——即 **M=output、N=batch**（对上 §4 的「output_size/batch_size swapped」注释）。

### 2.5 `mma_tma_ws_mainloop.cuh`：`CollectiveMainloop`（被 2.2/2.3 用、被 2.4 不用）

`CollectiveMainloop`（`:90-490`）与 `MainloopParamsImpl<OnHost>`（`:38-88`）也是 CUTLASS 的
`sm90_mma_array_tma` collective 直系拷贝。**唯一值得单列的是 `OnHost` 特化**：

- `Params<true>`（`:39-77`）带**真 TMA 描述符** `tma_load_a/tma_load_b`（由 CuTe
  `make_tma_copy_A_sm90/B_sm90` 在 host 构造）——供单测/非持久路径；
- `Params<false>`（`:80-88`）只留 `tma_transaction_bytes{,_mk,_nk}` + `problem_shape`，**不带描述符**——
  供 **MPK device 路径**（描述符由 `task_desc->input_tma_desc_ptrs` 在设备侧给，见 §4 的 JIT 串）。

这与 `docs/60`/`docs/34` 反复出现的「同一套逻辑在 host 版/device 版各写一份」同源。
其 `load`（`:239-311`）、`mma`（`:326-469`）、`mma_tail`（`:472-489`）**只被 §2.2/§2.3 调用**；
`gemm_ws_mpk.cuh` 一次都不调（已 grep 确认）。

### 2.6 `epilogue.cuh`：`CollectiveEpilogue`

`CollectiveEpilogue`（`:28-386`）提供 `store_op`（accumulator → gmem，含 swapAB 转置分支
`if constexpr(Has_SwapAB_v<Ktraits>)`，`:114`）与 `store`（`:340+`）。**同样被 §2.2/§2.3 调用、
被 §2.4 只当类型来源**（`gemm_ws_mpk.cuh:480-481/518` 借 `DataTypeC`/`ThreadEpilogueOp` 自己实现内联 epilogue）。

---

## 3. 「三份实现」的对照

| | `gemm_ws.cuh` | `gemm_ws_cooperative.cuh` | `gemm_ws_mpk.cuh` |
|---|---|---|---|
| 角色 | Producer + Consumer（各 1 wg） | Producer + Consumer0/1（1 + 2 wg） | Producer + Consumer（各 1 wg） |
| 调度 | 单 tile（示 `blk_coord`，`gemm_ws.cuh:196`） | `PersistentTileSchedulerSm90` + CLC + 寄存器重分配 | **固定单 CTA**（`blk_coord=(0,0,_,0)`，`gemm_ws_mpk.cuh:246`） |
| TMA | `collective_mainloop.load` | 同 | **手写 `tma_cp_async`（MPK `tma_2d`）** |
| wgmma | `collective_mainloop.mma` | 同 | **手写 `cute::gemm` 主循环** |
| epilogue | `collective_epilogue.store` | 同 | **手写 store + swapAB 转置** |
| 调用者 | 单测**注释**的 wrapper | **无** | 单测 `linear_mpk` + 后端 JIT |
| 状态 | 死 | 死 | **活** |

即：**CUTLASS 骨架 / CUTLASS 协同骨架 / MPK 单 CTA 版** 三份并存，只有第三份活着；第三份又**不复用**
前两份所依赖的 collectives 方法。这与 `docs/20` §5「同一套惯用法实现两遍」的分母不同——这里连
「非协同/协同」两个 CUTLASS 变体都一并搬了进来。

---

## 4. 生产布线：JIT 生成串（`task_register.cc:1754-1922`）

后端不为这个 task 写死一个 kernel，而是**拼出源码串交给 MPK 的 JIT**（同 `docs/19`/`docs/22` 的
`CodeKeeper` 模式）。关键静态事实：

- 三个尺寸从 bgraph 取：`batch_size=output_tensors[0].dim[0]`、`output_size=...dim[1]`、
  `reduction_size=input_ops[0].dtensor.dim[1]`、`output_stride=kn_input_op->input_strides[0]`
  （`:1775-1783`）。
- **注意源码注释两处 `// NOTE: output_size and batch_size are swapped here`**（`:1788`/`:1794`）：
  生成的 `problem_shape = Shape<output_size, batch_size, reduction_size>`、`MMAKernelTraits<...,
  output_size, batch_size, reduction_size, ..., 8, 64, batch_size, TILE_SIZE, ..., output_stride,
  KSTAGES>`——即 `M=64`、`N=batch_size`、`K=TILE_SIZE=128`，与 §2.4 的 `(OUTPUT,BATCH,REDUCTION)`
  交换一致（swapAB 约定）。
- **一处自相矛盾的常量**：`:1764` 定义 `constexpr int KSTAGES = 4;` 传给 traits 的 `NUM_STAGES`；
  但紧接 TMA 段 `:1853` 又 `constexpr int Kstages = 5;`（大小写不同、**未被使用**）。**只记不判**。
- 生成的调用串 `kernel::linear_cutlass_ws_hopper<Mainloop,Epilogue,false,cutlass::bfloat16_t,
  $batch,$output,$reduction,TMA_A,TMA_B,$output_stride,$with_residual>(tma_a,tma_b,
  task_desc->output_ptrs[0], task_desc->input_ptrs[2])`（`:1903-1915`）；`TMA_A/B` 描述符由
  `task_desc->input_tma_desc_ptrs[1][0]`/`[0][0]` 就地 `CUtensorMap*` 转入（`:1899-1902`）。
- 返回 `with_residual? 158 : 157`（`:1918-1921`）。`task_config` 为 `(2,1,…)`/`(3,1,…)`（`graph.cc`）。

TMA 描述符侧：`tma.cuh:659` 处 `case 157/158` 用 `cp_async_size=64`、`B=M=S=3` 构造 A（output 行）；
`:1531` 处「只有 A、B 各有 1 个 tma_desc」。

---

## 5. 一个「swapped」设计：为什么 M=output、N=batch

该家族把**权重当 A（M×K）、激活当 B（N×K）但交换 A/B 语义**（`IS_SWAPAB` / `Has_SwapAB_v`）。
后果贯穿：
- `MMAKernelTraits::SwapAB`（`:200`）在多处被读；
- 单测注释 `// NOTE(Yu): Assume batch size is smaller than 16, and padding the batch size to 16`
  （`gemm_ws_mpk.cuh:127-130`）+ `static_assert(BATCH_SIZE<=16, "...in swapAB")`；
- `gemm_ws_mpk.cuh:466-552` 与 `epilogue.cuh:114-150` 都对输出做转置。
这与生产实现的动机一致（对上 `docs/07`/`docs/20`）：**推理的 M 维是 batch（小）、N 维是 output（大），
把「output 大维」放进 wgmma 的 M 轴能提高 tile 利用率，但代价是 epilogue 要转置**。这是
「同一数学、两种 A/B 摆放」在 CUTLASS 路径上的具体化。

---

## 6. 目标卡 H100/sm90a 真机实验：`linear_mpk`

该家族是 **Hopper**，故单测入口 `linear_mpk` 在本机（目标卡 H100/sm90a）可执行——本文把它跑通。
脚本 `harness/mpk_cute_hopper_ws.sh`（`lab.sh exec`，device 4，输出 `harness/mpk_cute_hopper_ws.out.txt`），
驱动 `harness/mpk_cute_hopper_ws.py`（用 `torch.utils.cpp_extension.load` 显式补 include 路径，见 §0(C)）。

**观测配置**：单测 wrapper 只实例化 `OUTPUT_SIZE=64`、`BATCH_SIZE∈{8,16}`、`REDUCTION_SIZE=4096`
（`gemm.cu:485-546`，其余 case 被注释）；kernel 恒为 `grid(1)`、`block(256)`、动态 SMEM 100000 B
（`gemm.cu:401-404`），即**单 CTA、一个 SM**。

**结果**：

| case | correctness | 稳态 ms/次 | read GB/s |
|---|---|---|---|
| B=8, O=64, K=4096 | `allclose`（max_rel 2.4e-3） | ~0.0435 | ~13.6 |
| B=16, O=64, K=4096 | `allclose`（max_rel 2.0e-3） | ~0.0430 | ~15.2 |

**读法（诚实）**：
- **正确性通过**：证明这条 CUTLASS-wrapped MPK 身体在 Hopper 上功能正确（数值误差来自 bf16 累加前的
  fp32 主循环与 torch fp32 参考之差，`rel≈2e-3` 符合 bf16）。
- **性能数字不宜外推**：`grid(1)` 只占 1 个 SM，权重（64×4096×2 B=0.5 MB）每 CTA 全读一遍，
  ~14–15 GB/s 是**单 SM 的 SMEM/TMA 上限**、不是机器带宽；这不代表 MPK 的 GEMM 能力，只代表
  「它作为一条独立 task 能跑对」。真正的多 CTA 路径是手写 `linear_hopper`/`linear_swapAB_hopper`
  （前端实际发的就是后者）。
- **快照自报数字不可信**：同一 kernel 的单测自带 benchmark 打印 `Average: 0.012 ms`
  （`WARMUP_RUNS=0`、`BENCHMARK_RUNS=1`、单次 event），与稳态 0.043 ms 差 3.5×——又一次「按脚本自报
  推断性能会错」。

---

## 7. 判据映射

- **C1（单次 launch）**：该 device 身体**并入**常驻 megakernel（由 worker 调用、不自 launch）；
  单测里是 standalone launch，但那不是生产形态。**贡献 C1（生产形态）**。
- **C2（常驻）／C4（设备内协调）／C5（SM 级调度）**：task 身体不构成新机制——调度仍由 `docs/06`
  的 worker/scheduler 事件环给。**不贡献**（同全部 device-task 走读）。但**协同版 `gemm_ws_cooperative`
  内含一个 CUTLASS 自带的 tile scheduler + CLC**，与本专题的「设备内调度应下沉给 worker」相反——
  它正是**被弃用**的那一支（无调用点），可作为「为什么 MPK 不把 per-task 调度留给 CUTLASS」的反证。
- **C3（多算子范围）**：它把「TMA 取权重/激活 + wgmma 主循环 + swapAB epilogue + 可选残差」压进
  单个 task 的 device body，是 C3 边界在 CUTLASS GEMM 上的形态。**贡献 C3**。

---

## 8. 局限（诚实标注）

- **纯 Hopper/sm90a 路径**：本文的 `gemm_ws_mpk` 是 wgmma 版；SM100 的对应物是 `docs/32`/`docs/33`/`docs/34`
  走的 `tasks/blackwell/linear_fp8_*`（不同家族，不重复）。
- **前端注释是静态事实，不等于上游决策**：`persistent_kernel.py:2441/2479` 被注释只说明**当前快照**
  不发这条 task；不判其被弃用/未启用的原因（可能只是「`linear_swapAB_hopper` 更快」）。**只记不判**。
- **`KSTAGES=4` vs `Kstages=5`**（`:1764`/`:1853`）：只记不判是否为笔误。
- **实验只覆盖单测实例化的两档**（O=64、B∈{8,16}、K=4096、单 CTA）；`O=128`、`B∈{32,64}`、`K=12288`
  的分派 case 被注释，未测；未测多 CTA/多 SM 形态（生产形态不可直接跑，需完整 MPK 运行层）。
- **未跑生产 JIT 路径**：需要完整 `persistent_kernel` + 前端发出被注释的 task 名，本机不具备
  （且因前端注释，构造该图需改前端，超出本轮范围）。
- **未逐行走完 `gemm_ws_cooperative.cuh` 的 CLC/寄存器重分配的每个分支**（只核了骨架与关键常量）。

---

## 9. 参考坐标（可核对）

- MPK（`f9eb70c`）：
  - `include/mirage/persistent_kernel/tasks/cute/hopper/kernel_traits.cuh`（全 203 行；`MMAKernelTraits:56-201`、
    `IsCooperative:119`、`AtomLayoutMNK:121-123`、`TiledMma:139-146`、`MainloopPipeline:191`、`SwapAB:200`）；
  - `.../cute/hopper/gemm_ws.cuh`（全 274 行；`gemm_kernel_tma_warp_specialized:41-269`、
    `collective_mainloop.load:210`、`mma:231`、`epilogue.store:252`）；
  - `.../cute/hopper/gemm_ws_cooperative.cuh`（全 428 行；`..._cooperative:39-427`、
    寄存器重分配 `:116-126/269/341`、CLC `:301-307`、`fixup:385`）；
  - `.../cute/hopper/gemm_ws_mpk.cuh`（全 562 行；`linear_cutlass_ws_hopper:48-559`、
    `aligned_smem:98-99`、`smem_tma:132-148`、`problem_shape:241-244`、producer `:255-298`、
    `tma_cp_async:284-286`、内联 wgmma `:299-424`、内联 epilogue+swapAB `:452-552`）；
  - `.../cute/hopper/mma_tma_ws_mainloop.cuh`（全 492 行；`MainloopParamsImpl<OnHost>:38-88`、
    `CollectiveMainloop:90-490`、`load:239-311`、`mma:326-469`、`mma_tail:472-489`）；
  - `.../cute/hopper/epilogue.cuh`（全 387 行；`CollectiveEpilogue:28-386`、`store_op:83+`、`store:340+`）；
  - `include/mirage/persistent_kernel/tasks/{hopper,blackwell}/task_header.cuh:13-15/11-13`（三份 include）；
  - `include/mirage/persistent_kernel/runtime_header.h:120-121`；
  - `src/kernel/task_register.cc:1754-1922`（`register_linear_cutlass_hopper_task`；`KSTAGES=4:1764`、
    `Kstages=5:1853`、调用串 `:1903-1915`）；
  - `src/kernel/graph.cc:585-595`；`include/mirage/persistent_kernel/tma.cuh:659/1531`；
  - `python/mirage/mpk/persistent_kernel.py:2438-2444/2475-2481`（两处注释）；
    `python/mirage/mpk/profiler_persistent.py:157-158`；
  - `tests/runtime_python/cute/hopper/gemm.cu`（全 557 行；注释 wrapper `:49-58`、
    活 wrapper `linear_kernel_hopper_cute_mpk_wrapper:256-281`、`linear_mpk`/dispatch `:485-556`）、
    `setup.py`（include_dirs `:30-36`，**缺 `include/mirage/persistent_kernel`**）、
    `test_matmul_ws_mpk.py`。
- 本文实验：`harness/mpk_cute_hopper_ws.py` + `harness/mpk_cute_hopper_ws.sh` +
  `harness/mpk_cute_hopper_ws.out.txt`（目标卡 H100/sm90a、device 4、nvcc 13.2、`-gencode=arch=compute_90a,
  code=sm_90a`；单 CTA、事件计时 200 次、min 前 20 次 warmup）。
- **交叉引用**：`docs/07`（`TASK_LINEAR_CUTLASS_HOPPER` 一行旁注，本文补全）、`docs/30` §5（单测 wrapper，
  从 wrapper 视角；本文从其**被包装的 device 家族**视角）、`docs/20` §5（「惯用法实现两遍」，本文是
  CUTLASS 路径的第三遍）、`docs/58` §C/`docs/51`（`smem_tma` 与 1024 B 对齐）、`docs/19`/`docs/22`
  （`CodeKeeper` JIT 生成模式）、`docs/06`（worker/scheduler 运行层，C2/C4/C5 的来源）。
