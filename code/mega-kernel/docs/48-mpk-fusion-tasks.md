# MPK 段内融合 task 家族走读：`norm_linear` / `silu_mul_linear` 的跨代身体与「已实现未编入」接线审计

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/47` 同快照）。
> `docs/07` §3.3 把 `norm_linear_hopper.cuh` 当作「段内多算子融合（T2）」的样例一笔带过；此前从没有
> 一篇把**整个融合 task 家族**（Ampere `norm_linear` / `norm_linear_new` / `silu_mul_linear` + Hopper
> `norm_linear_hopper`）逐行打开，也没有核过它们**到底有没有被编进任何生产/测试编译单元**。本文补这
> 一段，并顺手修正 `docs/07` §3.3 的「它是个活的融合 task」这一默认读法。
> **纯源码走读（资料分析）**：不构建、不运行；Hopper 版按 `docs/09` 的 `target_cc>=90` 分支本可在目标卡
> 编译，但**它不在任何 `task_header` 的 include 集里**（下文 §4），故**无新实验**。

---

## 0. 三条 headline

**(A) 融合边界确实存在，而且是「跨代各写一遍」**：`norm_linear`（RMSNorm+Linear）与
`silu_mul_linear`（SiLU×mul+Linear，可选 +residual）两个「段级融合算子」在仓库里有**四份独立 device
身体**——Ampere `norm_linear.cuh`（461 行）、Ampere `norm_linear_new.cuh`（465 行，同函数名的第二版）、
Ampere `silu_mul_linear.cuh`（371 行）、Hopper `norm_linear_hopper.cuh`（367 行，TMA+wgmma warp 特化）。
Ampere 三件是「SMEM 内先归一化/SiLU、再走同一份手写 `m16n16k16` MMA」，Hopper 一件是「TMA 三段流
（input / norm_weight / linear_weight）+ wgmma」的完整流水。

**(B) 但整个家族在 `f9eb70c` 是「已实现、未编入」**：`TaskType`（`TASK_RMS_NORM_LINEAR=102`、
`TASK_SILU_MUL_LINEAR_WITH_RESIDUAL=105`）、`graph.cc` 分支、`task_register` 发射器、前端
`rmsnorm_linear_layer`/`silu_mul_linear_with_residual_layer` **四层全通**，但——
**没有任何生产 `task_header.cuh`（ampere/hopper/blackwell）include 这四个头**；唯一引用它们的单测
`runtime_kernel_wrapper.cu` 把调用整段包在**从未定义的 `#ifdef DEPRECATED_TESTS`** 里、`using` 别名也
被注释；`norm_linear_hopper.cuh` 仅在 `runtime_kernel_wrapper_hopper.cu:21` 有一行**被注释掉的
include**。更硬的一条：这四个头自己的 `#include "common.h"`（Ampere）/`#include "../common.h"`（Hopper）
指向的文件在快照里**都不存在**（`tasks/common/` 只有 `common_header.cuh`）——**即使有人把它们加进
`task_header` 也编译不过**。这是本专题「已实现未接线」（`docs/16`/`docs/20`/`docs/32`/`docs/43` 等）
在**融合算子**层的一个整族实例。

**(C) 生产实际走的是「拆开的薄 task」而不是融合版**：Qwen3 builder 的 MLP 段是
`rmsnorm_layer`（`:505`）→ `linear_layer`（`:512`）→ `silu_mul_layer`（`:520`）→
`linear_with_residual_layer`（`:540`）**四次注册**，`rmsnorm_linear_layer`/`silu_mul_linear_with_residual_layer`
在整个 `models/` 里**零调用者**。也就是说：「把 norm 折进 GEMM」这条 T2 融合路，MPK 的**自动/生产前端
都主动没走**——与 `docs/23` §5「少而重 vs 多而杂」、`docs/40`「把写回从读里拆出去是 task 可复用/
可调度的先决条件」母题一致，只是这次被拆掉的是**归一化**而不是 KV 写回。本文把「已实现未编入」与
「生产为何选拆分」两侧证据一起给出。

---

## 1. 定位：融合 task 家族在编译栈里的坐标

四层布线（全部核对 `f9eb70c`）：

| 层 | `norm_linear`（RMSNorm+Linear） | `silu_mul_linear_with_residual` |
|---|---|---|
| 前端 layer | `persistent_kernel.py:883` `rmsnorm_linear_layer` | `:2570` `silu_mul_linear_with_residual_layer` |
| register 名 | `register_task(tb_graph, "rmsnorm_linear")`（`:902`） | `"silu_mul_linear_with_residual"`（`:2589`） |
| graph 分支 | `graph.cc:453-455` | `graph.cc:522-524` |
| register 实现 | `task_register.cc:489` `register_rmsnorm_linear_task` | `:826` `register_silu_mul_linear_with_residual_task` |
| 发射的 device 符号 | `kernel::norm_linear_task_impl<bfloat16,…>(…)`（`:520`） | `kernel::silu_mul_linear_task_impl<bfloat16,…>(…)`（`:857`） |
| `TaskType` | `TASK_RMS_NORM_LINEAR = 102`（`runtime_header.h:92`） | `TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105`（`:95`） |
| device 身体（名字） | `ampere/norm_linear.cuh:37`、`ampere/norm_linear_new.cuh:37`（同名两版） | `ampere/silu_mul_linear.cuh:38` |
| Hopper 身体（不同名） | `hopper/norm_linear_hopper.cuh:45` `norm_linear_kernel_hopper` | 无（Hopper 没有 silu_mul_linear） |

两处**注册即发射的语义**值得记下（读这层容易忽略）：

- `register_rmsnorm_linear_task` 断言 `num_inputs=3`（input / norm_weight / linear_weight）、
  `num_outputs=1`，并把 `runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` 与常数
  `1e-6f` 作为路由/eps 实参（`:520-526`）。
- `register_silu_mul_linear_with_residual_task` 断言 `num_inputs=3`（**input 是 gate/up 拼在一起的
  2×intermediate**，`reduction_size = input_ops[0]->dtensor.dim[1] / 2`，`:846`），并把
  **`runtime_config.my_gpu_id == 0`** 当作 `residual` 布尔实参传给 device（`:868`）——即 TP 下
  **只有 rank0 加残差**（对上 `docs/17` §7 的 residual 约定）。

> 注意这里有一处「名字会骗人」：`norm_linear_task_impl` / `silu_mul_linear_task_impl` 的**发射名是
> arch-agnostic 的**（task_register 不带 arch 分支），真正「哪份身体被编进本次 kernel」完全由
> `tasks/<arch>/task_header.cuh` 的 include 集决定（对上 `docs/07` §0、`docs/14` §4）。本文 §4 正是
> 从这个 include 集读出「谁也没编进去」。

---

## 2. Ampere 三件套：同一份手写 MMA 上的三种融合

### 2.1 `norm_linear.cuh`（RMSNorm + Linear，461 行）

一个 CTA 内把 RMSNorm 与 Linear 融成一次执行，全部用 Ampere 手写原语（`cp.async` + `ldsm` +
`mma_m16n16k16_bf16bf16bf32`），**没有 TMA、没有 wgmma、没有 producer/consumer 分工**。

- **SMEM 预算先行**：`MAX_OUTPUT_ATOM_SIZE = max_power_of_two_le((MAX_DYNAMIC_SHARED_MEMORY_SIZE -
  16 - 16898*BATCH_SIZE) / (4*(128+BATCH_SIZE)))`（`:45-48`），据此切出 `OUTPUT_ATOM_SIZE ≤ 128`
  与 `NUM_OUTPUT_ATOMS`（`:50-57`）。即「一批 output 原子」共享同一份 norm 结果，避免对每个原子重算
  归一化——这是融合的第一个结构性收益。
- **K 方向流水**：`TILE_SIZE=128`、`FORLOOP_RANGE=REDUCTION_SIZE/TILE_SIZE`、`K_PIPE_MAX=2`
  （`:36`/`:59-60`）；`input`（`BATCH_SIZE×TILE_SIZE`）与 `norm_weight`（一行 `TILE_SIZE`）打在
  同一段 SWIZZLE 为 0 的 `smem_row` 里，linear weight 走 `smem_col<T,3,3,3,…>`（`:178-192`）。
- **归一化现算**：每轮 `mul_broadcast_row(mul_output_smem, input_smem, norm_weight_smem)`
  （`:340`）先把 `x*norm_weight` 广播成 `BATCH_SIZE×TILE_SIZE`；同时 `perform_element_unary_chain_kernel`
  只对 `input` 这一半做 `SQUARE → MULSCALAR(1/REDUCTION_SIZE)`（`:371-378`），累进 `element_unary_smem`
  作为「x² 的部分和」。K 维跑完后 `reduction_sum_col` 用 `ADDSCALAR(eps) → SQRT`（`:412-421`）得到
  `rms = sqrt(eps + mean(x²))`；最后 `div_col == 归一化缩放`（`:423-427`）把 MMA 输出逐列除以 rms。
  **注意**：被平方的是**原始 x**，被乘进 GEMM 的是**已乘 norm_weight 的 x**——这正是 `x/rms * norm_weight`
  的正确重排（先乘权重不改变均值结构）。
- **MM 与写回**：`s_frag[1][NUM_ITERS_N][8]`（`NUM_ITERS_M=1`，注释自陈 *"Batch size more than 16
  will cause error"* `:89-90`），`ldsm` 取 `mul_output`(A) 与 `weight`(B)，`mma_m16n16k16` 累加；K 方向
  多 warp 时用 `reduction_sum_row(mm_output, mm_intermediate)` 跨 warp 归约（`:405-410`），写回
  `output_dmem`（`:430-436`）。**没有残差、没有 bias、没有量化**——纯 `y = (x*norm_weight)/rms @ Wᵀ`。

### 2.2 `norm_linear_new.cuh`：同函数名的第二版（465 行）

与上一版**函数签名/名字相同**，只有若干实现差异（`diff` 核对）：

- 多一个 `int num_active_tokens` 形参（`:40`），有效性检查从 `m_row < BATCH_SIZE` 改成
  `m_row < num_active_tokens`（`:348`）、写回也从 `< BATCH_SIZE` 改成 `< num_active_tokens`
  （`:392`）——即支持运行期「本 batch 只有前若干 token 有效」（对上 `docs/06` 的 `TaskDesc`
  运行期 active-token 语义，与 `silu_mul_hopper` 同型）。
- input 改用**带 swizzle 的** `smem_row<T,3,3,3,…>`（`:178`/`:181`，原版是 `0,0,0`），并按新的
  SMEM 公式重算 `MAX_OUTPUT_ATOM_SIZE = max_power_of_two_le((SMEM/sizeof(T) - ((REDUCTION_SIZE +
  2*TILE_SIZE+1)*BATCH_SIZE + REDUCTION_SIZE + 8)) / (K_PIPE_MAX*TILE_SIZE + 5*BATCH_SIZE))`
  （`:45-50`）。
- 结构上把 `process_atom` 的 lambda 位置/缩进重整，并显式 `clear_smem_buffer<NUM_WARPS_K*BATCH_SIZE*
  OUTPUT_ATOM_SIZE>(mm_intermediate)`（`:194-195`）。

**等价结论**：`norm_linear.cuh` 与 `norm_linear_new.cuh` 是「同一融合算子的两版」，靠 **include 选择**
而非 `#if` 切换——因为都定义同名 `norm_linear_task_impl`，同时 include 会重定义。它们谁生效完全取决于
谁被 `task_header` 引入；而 §4 显示**两个都没被引入**。

### 2.3 `silu_mul_linear.cuh`（SiLU×mul + Linear [+ residual]，371 行）

把「SwiGLU 门控 + 输出投影（+ 残差）」融成一个 task：

- **输入布局**：`d_input = input_ptr`、`d_mul = input_ptr + REDUCTION_SIZE`（`:79-81`）——gate 与 up
  已经在同一块 buffer 里首尾拼接（builder 侧的 `shuffle_tensors`/`inplace_shuffle_tensors` 负责交织，
  对上 `docs/28`/`docs/27`）；`InputDmem` 的 stride 是 `REDUCTION_SIZE*2`（`:88`），即一行读 2×inter。
- **SiLU 在 SMEM 现算**：每轮 K-tile 载入 input/mul/weight（`K_PIPE_MAX=3`，`:36`，三缓冲），
  `silu_mul_output = (x * sigmoid(x)) * mul`（`:294-301`，直接 `expf`），再 `ldsm` 进 MMA。
- **残差**：`bool residual` 形参（`:42`，由 §1 的 `my_gpu_id==0` 传入）；真值时多搬一份
  `residual_smem` 并在写回时 `val + residual_smem.at(row,i)`（`:356-363`）。**这是仓库里「残差加在哪
  、由哪个 rank 加」这条 TP 约定的又一处落点**（交叉引用 `docs/17` §7、`docs/28`）。
- `OUTPUT_ATOM_SIZE <= 128`、`NUM_WARPS_N`/`NUM_WARPS_K = 4/NUM_WARPS_N`（`:63-65`）——与 §2.1 同一
  套 warp 切分。

---

## 3. Hopper 融合版：`norm_linear_hopper.cuh`（TMA + wgmma warp 特化，367 行）

`docs/07` §3.3 只说了「一个 CTA 内先归一化再 GEMM」，实际是一套完整的 **Hopper warp-specialized 流水**，
比 Ampere 版高一代：

- **模板**：`norm_linear_kernel_hopper<T,BATCH_SIZE,OUTPUT_SIZE,REDUCTION_SIZE,Kstages,
  TMA_INPUT,TMA_NORM_WEIGHT,TMA_LINEAR_WEIGHT,TMA_OUT,OUTPUT_STRIDE>`（`:34-49`），四个 TMA 描述符
  显式传入（不像 Ampere 版直接收裸指针）。
- **角色**：`CONSUMER_WARPGROUPS=1`、`PRODUCER_WARPGROUPS=1`（`:54-56`）。`warpgroup_id ==
  NUM_WARPGROUPS-1` 是数据搬运组（`:193`），其余是计算组（`:266`）；两者通过
  `input_barrier`/`norm_weight_barrier`/`linear_weight_barrier`/`compute_done` 四组 `Barrier`
  数组（各 `Kstages` 个，`:171-188`）做 K-stage 双/多缓冲。
- **搬运组**：只有 `warp_idx == NUM_WARPGROUPS*WARPGROUP_WARPS-4` 的 `lane_id==0` 单线程发 TMA
  （`:196`），先 prefetch `min(Kstages,num_k)` 个 slot，之后每轮 `wait(compute_done[slot],phase^1)`
  再发下一段（`:198-264`）；三段 TMA（input `BATCH_SIZE×TILE_SIZE`、norm_weight 同尺寸、
  linear_weight `TILE_SIZE×OUTPUT_SIZE`，`:58-62`）各自 `set_barrier_transaction_bytes`。
- **计算组**：`wait(input_barrier,phase)`→`wait(norm_weight_barrier,phase)`→`mul(mul_output, input,
  norm_weight)`（`:286`，仓库通用 `mul`）→ `wait(linear_weight_barrier,phase)` → 用 `A_DESC`/`B_DESC`
  （`wgmma::mma_descriptor`，`:151-152`）发一条 `wgmma::mma<bfloat16,64,OUTPUT_SIZE,16,…>`（`:299-309`）；
  同时把 **原始 input** 做 `SQUARE→MULSCALAR` 累进 `element_unary`（`:311-317`）；`mma_async_wait`
  后由 warp0 每 slot `arrive(compute_done[slot])`（`:325-327`）。K 循环后 `reduction_sum_col`（ADDSCALAR
  eps→SQRT，`:332-338`）、寄存器 `s_frag` 写回 `mm_output_smem`（`:341-347`）、`div_col`
  （`:352`），最后 `async_proxy_fence()` → 单线程 `tma_out.tma_store_async`（`:359-363`）。
- **注意**：`wg_decrease_regs<32>()`/`wg_increase_regs<160>()` 两行**都被注释**（`:195`/`:268`）——与
  `docs/07` §2、`docs/18` §5、`docs/19` §1 反复出现的「setmaxnreg 被注释」是同一事实。
- **TILE_SIZE=64**（`:52`）且 `num_m = BATCH_SIZE/64`（`:66`）——即 **Hopper 版要求 BATCH_SIZE 是 64
  的倍数**，与 Ampere 版「batch≤16、NUM_ITERS_M=1」完全不同口径。

一句话：Hopper 版才是「真正的融合流水」（类 `linear_hopper` 的骨架 + 前方插入归一化），Ampere 版是
「同 CTA 内先归一再 GEMM 的两段拼接」。两者都**没进生产**。

---

## 4. 接线审计（headline）：四层全通，但一个编译单元都没进去

这是本文最值得记的一节。逐项核对（`f9eb70c`）：

1. **`task_header` 不含它们**：`ampere/task_header.cuh`、`hopper/task_header.cuh`、`blackwell/task_header.cuh`
   的 include 集里**都没有** `norm_linear*.cuh` 或 `silu_mul_linear.cuh`。Hopper header 引了
   `rmsnorm_hopper`/`silu_mul_hopper`/`linear_hopper`，**却没有 `norm_linear_hopper`**。
2. **唯一引用在从未定义的宏里**：`tests/runtime_python/runtime_kernel_wrapper.cu` 里
   `using kernel::norm_linear_task_impl;`（`:17`）与 `using kernel::silu_mul_linear_task_impl;`
   （`:23`）**均被注释**；真正调用它们的两段 `*_kernel_wrapper`/`launch_*` 全包在
   `#ifdef DEPRECATED_TESTS`（`:800`/`:1149`）里。全仓 `grep` **没有任何地方 `#define DEPRECATED_TESTS`**
   （与 `docs/40` 对退役 `paged_attention.cuh` 的结论同族，只是这里连 `#include` 都没有）。
3. **Hopper wrapper 只有注释里提过一次**：`tests/runtime_python/hopper/runtime_kernel_wrapper_hopper.cu:21`
   是 `// #include "tasks/hopper/norm_linear_hopper.cuh"`、`:28` 是 `// using kernel::norm_linear_kernel_hopper;`
   ——**全被注释**。
4. **自身 include 陈旧、指向不存在的文件**：四个头都 `#include "common.h"`（Ampere，相对同目录）或
   `#include "../common.h"`（Hopper）；`tasks/ampere/common.h`、`tasks/hopper/common.h`、
   `tasks/common.h` **在快照里都不存在**（`tasks/common/` 只有 `common_header.cuh`）。同族写法见
   `docs/40` §5、`docs/39` §1（`target_verify.cuh` 的 `#include "../common.h"`）。⇒ 即便把任一头加进
   `task_header`，**第一步预处理就会失败**。

结合起来：`TASK_RMS_NORM_LINEAR(102)`/`TASK_SILU_MUL_LINEAR_WITH_RESIDUAL(105)` 这两个 `TaskType`
**在快照里不可能被任何模型真正点亮**——没有 builder 调它们（§5），即使调了，注册发射的
`kernel::norm_linear_task_impl`/`silu_mul_linear_task_impl` 也**不在编译单元里**。这是「按名字/grep
到注册/前端就以为它能跑」的又一反例（方法论同 `docs/32` §3、`docs/45` §3.3、`docs/24` §4）。

> 与 `docs/37`/`docs/38` 的薄 task「每个都有真实 builder 调用」正好相反：融合 task 家族是**有注册、
> 无生产者、无编译单元**的三无状态。

---

## 5. 生产实际走的路：拆开的薄 task（对照）

以 Qwen3 MLP 段为例（`models/qwen3/builder.py`），实际注册的是**四个独立 task**：

- `self.mpk.rmsnorm_layer(...)`（`:505`）——独立 RMSNorm（`rmsnorm`/`rmsnorm_hopper` 按
  `target_cc>=90` 选，见 `persistent_kernel.py:881`）。
- `self.mpk.linear_layer(...)`（`:512`）——独立 GEMM（`linear` / `linear_swapAB_hopper` /
  `linear_sm100`）。
- `self.mpk.silu_mul_layer(...)`（`:520`）。
- `self.mpk.linear_with_residual_layer(...)`（`:540`）。

而融合版 `rmsnorm_linear_layer`/`silu_mul_linear_with_residual_layer` **在整个 `models/` 里零调用者**
（`grep` 只命中定义处）。一个「历史的指纹」是 `models/utils.py:3` 的辅助
`grid_for_rmsnorm_linear_layer(size)`：它名字里带 `rmsnorm_linear`，却**被拆开后的 `linear_layer`
与 `rmsnorm_layer` 用来算 grid**（`builder.py:452`/`:466`/`:499`/`:642`）——名字残留自融合时代，行为只
是「给一个 output size 挑 96/64/256 的 block 数」。

**为什么拆开**？`docs/23` §5、`docs/40` §0 给的工程理由是「task 可复用/可调度」：一个融合 task 把
「归一化的依赖」与「GEMM 的 tile 并行」绑死，就不能被不同 grid 的后续算子复用、也不能在 event 图里
单独作为依赖节点被重排；而拆开后 `rmsnorm`（grid=`max_num_batched_tokens`）与 `linear`
（grid=`grid_for_rmsnorm_linear_layer(...)`）各自可独立布线、独立通信-计算重叠。本文**只给静态事实**
（融合版没被调用、拆开版被调用）与这条与其它篇一致的机制解释，**不判哪支更新**。

---

## 6. 与论文 / 本专题的对照

- **论文 §4.1「operator decomposition」（快照行 462–468）**：论文设想的自动分解会把复合算子拆成
  细粒度 task 再由 linearization 排序。生产前端在这里做的正是**反向**——把一个融合算子主动拆回
  薄 task（`docs/14`/`docs/15` 已述「生产前端是手工任务图、不调 superoptimizer」）。融合 task 家族的
  「已实现未编入」正是这条反向选择的**化石证据**。
- **`docs/07` §3.3 修正**：该节把 `norm_linear_hopper.cuh` 当作活样例；实际它（以及 Ampere 三件）
  都不在任何编译单元里。§3.3 的**机理描述仍然正确**（它确实融合了 RMSNorm+Linear），但「它是个在跑
  的融合 task」的默认读法应改为「一套未编入的融合实现」。
- **`docs/45`（norm/rope 家族）**：那篇把「独立 norm task」与「折进 attention 的 SMEM norm」分开；
  本文补上第三种边界——「折进 GEMM 的 norm」（`norm_linear`），并指出它同样**没被采用**。
- **`docs/02`（DeepGEMM Mega Gate/mHC）**：生产级 mega 家族确实大量做**段内融合**（大写 GEMM +
  epilogue + barrier 融进单 kernel）。MPK 侧却把 norm+GEMM 拆开——两者并不矛盾：DeepGEMM 融的是
  「同一算子的 epilogue / 相邻通信」，而 `norm_linear` 融的是「两个**结构不同**的算子」（归约密集的
  norm + Tensor-Core 密集的 GEMM）。**融合哪些边界**是设计选择，不总是「越多越好」。
- **判据**：这些融合 task 只贡献 **C3 的范围**（段级多算子），C1/C2/C4/C5 由运行层保证（同
  `docs/22` §8）。本文不改变任何 C 判据。

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。未验证「把它们加进
  `task_header` 并修 `common.h` 后能否编译/跑对」——只核出「当前状态编译不过」。
- **`norm_linear` vs `norm_linear_new` 的取舍系静态推断**：二者同名、互斥，本文只列 diff；谁更「新」、
  谁本意替换谁，源码无注释，**不判**。
- **「生产为何拆分」的机制解释**引 `docs/23`/`docs/40`，属工程推断，非源码结论。
- **`my_gpu_id==0` 作 `residual` 的 TP 语义**：从实参静态读出「只有 rank0 加残差」，但**未追
  `runtime_config.my_gpu_id` 在多卡下的赋值链**（交叉引用 `docs/17` §7 有相关但未逐行对齐本处）。
- **Hopper 版与 Ampere 版的性能差异未实测**（本文不做实验；且它们当前不可编译）。
- **未覆盖**：`graph.cc`/`task_register` 里是否还残留其它「无生产者」的融合算子（本文只查了这两个）；
  以及 `docs/44` §10 仍余的 `smem_row`/`ldsm`/`dmem_row` 布局内部（**留作下一步**）。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/47` 同快照）。
- Ampere device：
  - `tasks/ampere/norm_linear.cuh`（`norm_linear_task_impl:37`；`MAX_OUTPUT_ATOM_SIZE:45-48`；
    SMEM 偏移表 `:114-154`；swizzle 定义 `:176-208`；K 流水 `:240-274`/`:286-338`；
    `mul_broadcast_row:340`；`SQUARE/MULSCALAR:371-378`；`reduction_sum_row:406`；
    `reduction_sum_col ADDSCALAR/SQRT:412-421`；`div_col:423-427`；写回 `:430-436`；`NUM_ITERS_M=1` 注释 `:89-90`）。
  - `tasks/ampere/norm_linear_new.cuh`（`num_active_tokens:40`；新 SMEM 公式 `:45-50`；
    swizzle `3,3,3` `:178`/`:181`；`m_row<num_active_tokens:348`；写回 `<num_active_tokens:392`）。
  - `tasks/ampere/silu_mul_linear.cuh`（`silu_mul_linear_task_impl:38`；`d_mul=input+REDUCTION_SIZE:80-81`；
    `InputDmem` stride `REDUCTION_SIZE*2:88`；`K_PIPE_MAX=3:36`；SiLU 融合 `:294-301`；
    `reduction_sum_row:351`；残差写回 `:356-363`）。
- Hopper device：`tasks/hopper/norm_linear_hopper.cuh`（模板 `:34-49`；`TILE_SIZE=64:52`；
  four-barrier 数组 `:171-188`；搬运组 `:193-265`；计算组 `:266-364`；`mul:286`；`wgmma:299-309`；
  `SQUARE/MULSCALAR:311-317`；`reduction_sum_col:332-338`；`div_col:352`；`tma_store_async:359-363`；
  被注释的 `wg_decrease/increase_regs:195/268`）。
- 注册/发射：`src/kernel/task_register.cc:489-527`（rmsnorm_linear）、`:826-871`（silu_mul_linear，
  `my_gpu_id==0:868`）；`include/mirage/persistent_kernel/runtime_header.h:92/95`（`TaskType 102/105`）；
  `src/kernel/graph.cc:453-455`/`:522-524`。
- 前端：`python/mirage/mpk/persistent_kernel.py:883-903`（`rmsnorm_linear_layer`）、
  `:2570-2590`（`silu_mul_linear_with_residual_layer`）；对照 `rmsnorm_layer` 注册名 `:881`、
  `linear_layer` 注册名 `:2445`、`silu_mul_layer:2527`、`linear_with_residual_layer:2484`。
- 生产 builder：`python/mirage/mpk/models/qwen3/builder.py:505/512/520/540` 与
  `grid_for_rmsnorm_linear_layer` 调用 `:452/466/499/642`；辅助 `models/utils.py:3-14`。
- 接线审计：无 `task_header` include；`tests/runtime_python/runtime_kernel_wrapper.cu:17/23`（注释 using）、
  `:800`/`:1149`（`#ifdef DEPRECATED_TESTS`）、`:814`/`:1153`（调用点）；
  `tests/runtime_python/hopper/runtime_kernel_wrapper_hopper.cu:21/28`（注释 include/using）。
- 本专题交叉引用：`docs/07` §3.3、`docs/14` §4、`docs/15` §7、`docs/17` §7、`docs/22` §8、
  `docs/23` §5、`docs/24` §4、`docs/40` §5、`docs/45`、`docs/02`。
