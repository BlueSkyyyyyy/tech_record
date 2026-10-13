# MPK 自动路径（下）：Mirage transpiler 源码走读——从 thread-block 图到可编译的 CUDA

> 本文是 MPK 源码走读链条里、**`docs/15`（superoptimizer）与 `docs/16`（搜索器辅助模块）
> 之后的下半段**。`docs/15` §5「生成：从等价图到 CUDA」只把 `src/transpiler/*` 拆成
> resolve / schedule / codegen 三段与真实行号，并在 §9 末句明留一条尾巴：
>
> > **`transpiler` 只给管线结构**：`src/transpiler/*` 共约 8 千行（`transpiler_tb{,_hopper,_blackwell}.cc`
> > 各 1.5 千行），本文把它拆成 resolve/schedule/codegen 三段与真实行号，**未逐条读 SASS 级生成逻辑**。
>
> 本文把这条尾巴补上：**一次 `KNGraph.compile()` 到底怎样把一张 thread-block 图变成一段能被 `nvcc`
> 编译、能在 GPU 上 launch 的 CUDA**。聚焦「编译器生成什么代码」，不重复 `docs/15` 已给的搜索/验证。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`/`07`/`08`/`14`/`15`/`16`/`17` 同一快照，
> 所有行号都据此核对。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（以下简称「快照论文」）。
>
> 本文为**纯源码走读（资料分析）**：不重新构建/运行 mirage。`docs/15` §9 已说明原因——
> 自动路径需在 GPU 上反复 compile+profile 选优，本机不具备该跑批条件。

## 0. 本文在链条里的坐标

先给出**谁调谁**（`docs/15` §1 走到 `transpile()` 就停了，本文从这里往下）：

```
python  KNGraph.compile()                       kernel.py:396
          └─ generate_cuda_program(...)         kernel.py:431
                └─ transpile(...)               core.pyx:1420   (C++)
                      └─ Transpiler ctor         transpile.cc:461 (图重写/归一化)
                      └─ generate_code()        transpiler.h:128
                            ├─ resolve_distributed_config   transpiler.h:52
                            ├─ resolve_dtensor_meta          resolve_dtensor_meta.cc:27
                            ├─ resolve_tb_fusion             resolve_tb_fusion.cc:11
                            ├─ resolve_tensor_layout         resolve_tensor_layout.cc:121
                            ├─ plan_dtensor_memory           plan_dtensor_memory.cc:22
                            └─ transpile_ugraph()            transpiler_kn.cc:332
                                  └─（per KN op 或 KNCustomizedOp）
                                     transpile_kn_custom_op{,_hopper,_blackwell}
                                        ├─ get_threadblock_schedule     sched_tb_graph.cc:144
                                        ├─ get_threadblock_swizzle_plan plan_tb_swizzle*.cc
                                        └─ get_threadblock_memory_plan  plan_stensor_memory.cc:256
                                          → __global__ void custom_kernel_N(...)
          └─ f.write(code + HARD_CODE)           kernel.py:469      (拼 pybind 入口)
          └─ nvcc 编译成 test.cpython-*.so       kernel.py:476+
          └─ module.launch → execute_mugraph     runtime.h:31
```

被生成的 `.cu` 顶部 `#include "runtime.h"`，而 `runtime.h`（`include/mirage/transpiler/runtime/runtime.h`）
**前向声明**了两个由 transpiler 生成的函数——`_init()`（`:14`）与
`_execute_mugraph(inputs, outputs, buf, stream, profiler_buffer)`（`:15`），并提供两个 C/C++/Python 入口：
`extern "C" void execute_mugraph(...)`（`:31`，首次调用时 `_init()` 一次）与
`void execute_mugraph_wrapper(...)`（`:47`）。所以 transiler 产出的代码是**两个可编译单元**：
「一堆 `custom_kernel_N` + `_init` + `_execute_mugraph`」拼上「固定 runtime 头 + `HARD_CODE` 的
PyMethodDef」。

一个关键事实（与 `docs/08` 的运行层路径对照）：**这条 transpiler 路径产出的是
`gate convolve / torch::` 式的一次性 `.so`，不是 MPK 的常驻 worker/scheduler megakernel**。
两条路共用上游的图 IR，但下游完全不同（`docs/15` §0 的表）。本文走的是「Mirage 通用编译器」这半条。

## 1. 入口与图归一化：ctor 里发生的两段「图重写」

`Transpiler` 的构造函数（`transpile.cc:461`）在真正生成代码前先做两件**图重写**——这正是论文 §4.1
「operator decomposition」在 Mirage 编译器侧的真实落点（对照 `docs/08` 里 MPK 运行层用的
`annotated_graph.cc`，那是另一套）：

1. **online softmax 重写**（ctor 调用点 `transpile.cc:473-475`，仅当 `config.enable_online_softmax`）：
   `rewrite_graph_for_online_softmax`（定义在 `transpile.cc:101`）扫描「`TB_EXP_OP` 的输入是
   `TB_MATMUL_OP`、且被 `TB_FORLOOP_ACCUM_RED_LD_SUM_OP` + 另一个 `TB_MATMUL_OP` 消费」的
   FlashAttention 型子图（`get_tensors_replace_for_online_softmax`，`transpile.cc:35-100`，恒替换 4 个
   张量），改写成 **rescale 形态**（`reduction_max` → `sub` → `exp` → `forloop_accum_rescale(..._RESCALE_OP)`），
   见 `transpile.cc:276-301`。不做则原图返回。
2. **算子分解/归一化**（`transpile.cc:482-767` 的第二个逐 op 循环）：重建一张**新图**，把「复合 TB
   算子」拆成原语写入新图——
   - `TB_FORLOOP_ACCUM_RED_LD_SUM_OP` → `forloop_accum(NO_RED)` + `reduction`（`transpile.cc:669-681`）；
   - `TB_FORLOOP_ACCUM_RED_LD_RMS_OP` → `square` + `mul_scalar(1/(dim*forloop))` + `forloop_accum` +
     `reduction` + `sqrt`（`transpile.cc:682-695`）；
   - `TB_RMS_NORM_OP` → `square` + `mul_scalar(1/dim)` + `reduction` + `sqrt` + `div`
     （`transpile.cc:734-741`）。
   归一化后有一段**断言收尾**（`transpile.cc:769-789`）：新图里不允许出现非默认的 forloop-accum
   （只允许 `NO_RED`/`NO_RED_RESCALE`/`MAX`），也不允许残留 `TB_RMS_NORM_OP`——即「复合算子必须在
   这里被拆干净，后面 schedule/codegen 只认原语」。这一步让下游的调度/融合/流水都只需处理少量原语，
   是「把复杂度挡在编译前端」的典型做法。

一句话：`ctor` 是**算子级归一化**（semantics-preserving rewrite），`generate_code()` 才是**布局/融合/调度**。

## 2. 五个 resolve 阶段：`generate_code()`（`transpiler.h:128-135`）

`generate_code()` 是一串固定顺序的 resolve，各自写各自的元数据容器：

| 阶段 | 入口 | 产物 |
|---|---|---|
| `resolve_distributed_config` | `transpiler.h:52-55` | `num_gpus`/`use_nvshmem`（`gpu_dim` 乘积）；多卡分支见 `docs/17` |
| `resolve_dtensor_meta` | `resolve_dtensor_meta.cc:27` | 每个 DTensor 的 `DTensorMeta`（`structs.h:173`）：`innermost_dim`/`strides`/`num_phy_elems`/`is_input`/`is_output` |
| `resolve_tb_fusion` | `resolve_tb_fusion.cc:11` | `is_fused_with_prev/next`、`fusion_chain`（`structs.h` 之外，`transpiler.h:68-75`） |
| `resolve_tensor_layout` | `resolve_tensor_layout.cc:121` | 每个 STensor 的 `STensorMeta`（`structs.h:207`）：SMEM 布局 + `swizzled_dim` |
| `plan_dtensor_memory` | `plan_dtensor_memory.cc:22` | 中间 DTensor 的 `addr` 与全局 `d_buf_size` |

其中两处值得单看：

- **`resolve_tb_fusion`（`resolve_tb_fusion.cc:11-73`）**：只在「**elementwise unary 或
  `TB_FORLOOP_ACCUM_NO_RED_OP`，且其唯一输入张量的消费者恰为 1**」时才与**前驱**算子融合；并显式排除
  「前驱是 `FORLOOP_ACCUM_*` / `TB_INPUT_OP` / reduction-max（两输出）」（`resolve_tb_fusion.cc:38-57`）。
  融合后沿输入链回溯出 `fusion_chain`（tail→leading，`resolve_tb_fusion.cc:58-72`），供 codegen 生成
  **融合 epilogue**（§4）。这是「threadblock-level data reuse」的最小规则集，比论文 §4.2 的
  「software pipelining / register reuse / layout optimization」中的 register reuse 更保守。
- **`plan_dtensor_memory`（`plan_dtensor_memory.cc:22-61`）**：一个极简 `MemoryPlanner`——**只分配不复用**，
  **128 B 对齐**（`ALIGN = 128`，`:29`），非输入非输出的中间 DTensor 依次分配地址，得到 `d_buf_size`。
  源注释自陈「未来可引入 memory reuse 等更高级策略」（`:28-30`）——即**当前无 liveness 分析**。

## 3. `transpile_ugraph()`：把整图逐 op 展开成一个 `.cu`（`transpiler_kn.cc:332`）

`transpile_ugraph()` 生成四个 `CodeKeeper` 并拼接（`transpiler_kn.cc:857-862`）：`header` +
`custom_kernels` + `init` + `exec`（Hopper 另有 `hopper_tma` 段）。关键内容：

- **header**（`:339-348`）：`#define NUM_GPUS`/`USE_NVSHMEM`、按 `target_cc` 定义
  `MIRAGE_GRACE_HOPPER` / `MIRAGE_BLACKWELL`、`#include "runtime.h"`、`using namespace cute;`。
- **`_execute_mugraph`（`:350-355`）** 顺序遍历 `g->operators`，对每个 op 起一个 `{...}` 作用域并
  打注释 `// OP type: <json>`。
- **非 customized 算子直接落到 runtime 模板**：
  - `KN_MATMUL_OP` → `kn::gemm<...>`（cuBLAS，`:366-425`，含 batch stride 推导）；
  - unary（exp/silu/gelu/relu/clamp/square/sqrt，`:427-493`）→ `kn::ElementUnaryKernel<...>::run`；
  - binary（add/mul/div/pow，`:495-550`）→ `kn::ElementBinaryKernel<...>::run`；
  - reduction（`:552-612`）→ `kn::ReductionKernel<...>`。
  这些 op 的**布局用 CuTe `Layout<Shape,Stride>` 在 host 侧拼成模板参数**，内层维度被搬到最前以合并
  全局访问（`mov_last_and_get_layout`，`transpiler_kn.cc:80-85`）。
- **指针获取统一走 `get_dtensor_ptr`（`transpiler_kn.cc:92-118`）**：输入 → `input_tensors.at(idx)`、
  输出 → `output_tensors.at(idx)`、中间张量 → `(char*)buf + meta.addr`。这解释了为什么
  `_execute_mugraph` 的签名里既有 `input_tensors`/`output_tensors` 也有 `buf`（中间张量池）。
- **`KN_CUSTOMIZED_OP`（`:614-845`）** 是本文重点：先按「**输出在前、输入在后**」的约定把该 kernel 的
  所有实参指针对齐（`:621-634`，源注释原文 "the arguments are in the order of
  output_tensors, input_tensors"），再按 `config.target_cc` 分派到三个后端：
  `transpile_kn_custom_op_hopper`（H100）/ `_blackwell`（B200）/ `transpile_kn_custom_op`（其它）——见
  `:642-655`。随后生成 `dim3 grid_dim/block_dim`、`smem_size`（`:686-694`）、Hopper/Blackwell 的
  TMA 声明与 `TiledMMA`（`:700-790`）、`cudaFuncSetAttribute(...MaxDynamicSharedMemorySize...)`
  与 launch（Hopper 用 `$<<<grid_dim, block_dim, smem_size, stream>>>(...)`，B200 用
  `cutlass::launch_kernel_on_cluster(...)`，H100 分支 `:804-813`）。
- **返回值 `TranspileResult`（`structs.h:63`）** 携带 code、`buf_size`（`d_buf_size`）、
  `max_smem_size`（所有 custom kernel 的最大 SMEM，`:657-659`）、`profiler_buf_size`、
  `output_directives`（`:879-893`）。`generate_cuda_program` 把它转成 Python dict（`core.pyx:1430-1444`）。
  `kernel.py:437-449` 用它做 SMEM 上限检查：超限则置 `_valid_cuda_kernels = False`。

## 4. 单个 customized op：schedule + swizzle + memory plan 三件套

三个后端（`transpiler_tb.cc:253`、`transpiler_tb_hopper.cc:303`、`transpiler_tb_blackwell.cc:351`）
的骨架一致，都是先算三件套、再 emit kernel：

```
TBSched sched = get_threadblock_schedule(g);              // sched_tb_graph.cc:144
get_threadblock_swizzle_plan{,_hopper,_blackwell}(g,sched);
TBMemoryPlan mem_plan = get_threadblock_memory_plan(g,sched{,hopper}{,blackwell});
```

——Hopper 版调用点 `transpiler_tb_hopper.cc:338/340/343`，Blackwell 版
`transpiler_tb_blackwell.cc:385/387/390`。

### 4.1 schedule：`get_threadblock_schedule`（`sched_tb_graph.cc:144-402`）

产出 `TBSched`（`sched_tb_graph.h:44-51`）：三段 `pre_loop_nodes` / `loop_nodes` / `post_loop_nodes`，
每段是 `TBSchedNode` 序列（`OPERATOR` 或 `SYNCTHREADS`，`sched_tb_graph.h:10/33`）。核心步骤：

1. **逐 op 定 `TBSchedOpMeta`（`sched_tb_graph.cc:148-264`）**：
   - `TB_INPUT_OP`：判是否 **chunked copy**（`can_perform_chunked_copy`，`:19-70`：要求 DTensor 所有
     stride 是 16 B 对齐或为 1、且 DTensor 与 STensor 的「real innermost dim」相同）与是否
     **async/pipelined**（chunked 且 `forloop_dim != -1` 且 `target_cc >= A100`，`:200-203`）。
   - `TB_OUTPUT_OP`：判 chunked output；并断言「若某维被多 TB 切分则必须 `div_dim >= 0`」（`:220-228`，
     源注释 "as stated in the paper"）。
   - `TB_FORLOOP_ACCUM_NO_RED_OP`：**用 192 寄存器预算决定 accum 是否放进寄存器**
     （`:239-260`，源注释引 CUDA guide 的 255 寄存器上限、留 192 给累加器）。这是 `docs/07`/`docs/14`
     §2 里那条「reg realloc 被注释掉」的**编译器侧等价物**——此处是**编译期**决定 accum 在寄存器还是 SMEM。
2. **算 level 与 fusion-chain 分组**（`:271-370`）：level = max(前驱 level)+1；融合链内 op 沿用同一
   `fuse_chain_idx`、`pos_in_fuse_chain+1`。
3. **`ops2sched`（`:91-141`）**：按 `(level, fuse_chain_idx, pos_in_fuse_chain)` 排序，**同 level 与
   跨 level 之间插入 `SYNCTHREADS` 节点**（`:117-121`），把同链 op 打成一个 `OPERATOR` 节点，并调用
   `refine_opmeta_on_chain`（`:77-88`）：**只有「首 op 是 MATMUL 且链长恰为 2」时 accum 才留寄存器**。
4. 三个分区各有过滤谓词：pre 只收 `TB_INPUT_OP` 且 `forloop_dim == -1`（`:277`）、loop 收
   `forloop_dim != -1` 的 input 与其余算子（`:316-323`）、post 收 `after_accum` 之后的算子（`:368-375`）。
   收尾断言三段算子总数 == 原图算子数（`:377-390`）。

### 4.2 swizzle：`get_threadblock_swizzle_plan`（`plan_tb_swizzle.cc:9-177`）

目的：给需要 swizzle 的 STensor 消除 SMEM bank conflict（论文 §4.2 的 "layout optimizations to
reduce shared-memory bank conflicts"）。分两步——先对全图 STensor 求
`innermost_min_chunk_size`（chunked input/output 与 matmul 都要求至少 `num_16B_elems` 个连续元素，
`:38-88`），再决定 swizzle 方式（`:89-176`）：当内维 chunk 数是 2 的幂时用 **XOR swizzle**
（`meta.is_xor_swizzled = true` + `xor_swizzle_{b,m,s}`），否则退化为 **shift-based**（重排 stride，
`:143-171`）。cute 布局侧在 `get_stensor_layout`（`transpiler_tb.cc:80-95`）把 XOR 参数编成
`Swizzle<b,m,s>`。Hopper/Blackwell 各有自己的版本（`plan_tb_swizzle_hopper.cc:52` /
`plan_tb_swizzle_blackwell.cc:9`），因为 TMA/wgmma 对 SMEM 布局有额外约束。

### 4.3 memory plan：`get_threadblock_memory_plan`（`plan_stensor_memory.cc:256`）

给每个 STensor 分配 SMEM 地址（`TBMemoryPlan`，`structs.h:250`），并预留
`pipelined_input_buf_guid_offset`——即「双缓冲的第二份 buffer」的地址偏移，供软件流水
`SWAP(stensor$_ptr, stensor$_async_copy_buf)` 使用（见 §5）。

## 5. codegen：被生成的 kernel 长什么样

以通用/Ampere 版 `transpiler_tb.cc` 的 `transpile_kn_custom_op`（`:253`）为主线（Hopper/Blackwell
在其上加 TMA/wgmma/cluster）：

1. **函数签名**（`:280-320`）：`__global__ void __launch_bounds__(num_threads) custom_kernel_N(
   out_tensors..., in_tensors...)`（顺序与 `transpile_ugraph` 的实参约定一致）。`num_threads =
   block_dim.x*y*z`（`:256`）。
2. **SMEM 视图**（`:322-336`）：`extern __shared__ char buf[];`，按 `mem_plan.addrs` 从 `buf + addr`
   起造 `stensor$_ptr` 指针；并把 `buf` 头 16 B 清零（给 GEMM 用）。
3. **G→S 拷贝原子**（`:342-445`）：对每个 input op 选拷贝原子——
   `InputNonChunkedSyncCopy` / `InputChunkedSyncCopy` / `InputChunkedAsyncCopy`
   （按 §4.1 的 chunked/async 决策）；Hopper 版换成 TMA load（`STensor$InputAtom::run(tma_$, ...)`，
   `transpiler_tb_hopper.cc:915-925`）。
4. **软件流水**（`transpiler_tb.cc:836-855` + `:1416-1478`）：
   - prologue：对所有 pipelined input 发 `InputAtom::run` + `cute::cp_async_fence()`（`:836-855`）；
   - 主循环 `for (int for_idx = 0; for_idx < forloop_range; for_idx++)`（`:1421`）：**先发下一轮**
     `for_idx+1` 的 async copy、`cp_async_fence()`、`cp_async_wait<1>()`，再 `SWAP` 双缓冲
     （`:1424-1478`），然后跑本轮 `loop_nodes`。
   - Hopper/Blackwell 版把这条**升格为 warp specialization**：`warpgroup_id == num_consumer_wgs`
   的 warpgroup 当 producer 发 TMA（`transpiler_tb_hopper.cc:900-1000`），其余 consumer warpgroup 跑
   wgmma（`:1426-1470` 的 "Consumer main loop"），用 `hopper_async_pipeline_N.consumer_wait/release()`
   做环上同步（`:283`/`:1472`）。**producer/consumer 的 warp 组数与流水级数是编译期旋钮**，来自
   `core.pyx:1406-1408`（`num_warp_groups`/`pipeline_stages` 展开成
   `config.num_producer_wgs = 1`、`num_consumer_wgs = num_warp_groups - 1`）。这正是 `docs/15` §5 记的
   「扫 `{2,3,4}×{2,3,4}`」两个旋钮的**代码落点**。
5. **融合 epilogue**（`transpiler_tb.cc:861-899`）：把 fusion_chain 里 tail 之后的 elementwise
   算子（exp/silu/gelu/relu/clamp/square/sqrt/mul_scalar）在写出时**串联进一条 CuTe epilogue 模板链**
   （`EpilogueStore<...EpilogueExp<...>>`），即论文 §4.2 的 "register reuse"——**中间结果不落 SMEM、
   直接在寄存器里被下游 epilogue 消费**。
6. **写回**（`:1423-1490` 附近）：在寄存器里的 accum 在循环外写回（`Write back in-register
   accumulators`），再跑 `post_loop_nodes`。

Hopper 版另有一处与 `docs/07` §2 呼应的细节：`transpiler_tb_hopper.cc:902-906` 处**本应**调
`tb::wg_decrease_regs`/`wg_increase_regs`（wgmma 的寄存器再平衡）却被注释掉，只算了 `tma_reg`/
`mma_reg` 数值——与 `docs/07` §2 发现 `utils.cuh` 的 `setmaxnreg` 被注释是同一工程事实。

## 6. 与论文 §4.1/§4.2 的逐条对照

| 论文（快照论文行） | 源码落点 | 一致性 |
|---|---|---|
| §4.2「在 thread-block 级而非 kernel 级超优化」（行 1132-1134） | 搜索在 `LV_THREADBLOCK`（`docs/15` §3），编译器侧以**一个 customized op = 一个 `custom_kernel_N`** 为单位生成（`transpiler_tb.cc:280`） | 一致 |
| §4.2「每 task 生成 CUDA，含 software pipelining」（行 1138-1142） | `sched_tb_graph.cc:144` 的三段调度 + `transpiler_tb.cc:1416` 的主循环 + `cp_async_fence/wait<1>` + 双缓冲 `SWAP`（Hopper 升格为 producer/consumer warpgroup） | 一致 |
| §4.2「register reuse」（行 1140） | `resolve_tb_fusion`（`resolve_tb_fusion.cc:11`）+ fused epilogue（`transpiler_tb.cc:861`）+ accum-in-reg 预算（`sched_tb_graph.cc:239`） | 一致 |
| §4.2「layout optimizations to reduce shared-memory bank conflicts」（行 1140-1142） | `resolve_tensor_layout.cc:121` + `get_threadblock_swizzle_plan`（`plan_tb_swizzle.cc:9`，XOR/shift swizzle） | 一致 |
| §4.1「按输出张量分区成 task」（行 462-468） | 分区在**搜索器**里做（`docs/15` §3）；编译器侧的「分解」是**图归一化**（`transpile.cc:484-767`，复合算子拆原语） | 语义一致、分工不同：论文的「decomposition」在实现里分成「搜索器分区（高层）+ 编译器归一化（低层）」两处 |

一句话：**论文 §4.2 的四项 intra-SM 优化在源码里都有明确落点**——software pipelining、register reuse、
layout/bank-conflict 优化三者在 `src/transpiler/` 一一对应；而论文 §4.1 的「自动分区」落在搜索器
（`docs/15`），编译器只负责「把分区后的图归一化 + 生成代码」。

## 7. 判据 C1–C5 映射

与 `docs/14`/`docs/15` 一样，transpiler 层**只贡献 C3（融合的范围/粒度）**，不碰 C1/C2/C4/C5：

| 判据 | transpiler 是否贡献 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | `_execute_mugraph` 里每个 customized op 各自 `<<<...>>>` 一次 launch；**不是**单次提交。C1 由 MPK 运行层（`docs/06`）给 |
| C2 常驻 | 否 | 生成的 kernel 非常驻；常驻由 `docs/06` 的 worker/scheduler 给 |
| C3 多算子范围 | **是（粒度来源）** | 一个 customized op（多算子 bGraph）→ 一个 `custom_kernel_N`；融合范围 = bGraph 里被 search 选中的算子数 |
| C4 设备内协调 | 否 | 跨 kernel 依赖由 host 侧多次 launch 的顺序保证；无设备内事件/原子队列 |
| C5 SM 级调度 | 否 | 无 worker/scheduler、无去中心化调度 |

**结论**：Mirage transpiler 是「**把搜索出的 thread-block 图编译成高效单 kernel**」的编译器，它把
**单个 customized op 内部**做到接近论文 §4.2（warp specialization + 软件流水 + swizzle + register
reuse），但**跨 customized op 仍是 kernel-per-operator**（C1/C4/C5 缺位）。这恰好解释了
`docs/15` §7 的工程落差：MPK 要真正成为 megakernel，缺的不是这个编译器（它很完整），而是
MPK 在生产前端换了**手写 task + 运行层去中心化调度**（`docs/06`/`docs/14`）——编出一堆好 kernel
容易，把它们**编成一张图、常驻着调度**才是难的一半。

## 8. 局限

- **纯源码走读，未构建/运行**：不采信任何性能数；所有结论基于 `f9eb70c` 快照行号。
- **未逐条读 SASS 级生成逻辑**：本文覆盖到「生成什么 CUDA 源码」（模板名、循环结构、warp 角色、
  布局），但 `transpiler_tb{,_hopper,_blackwell}.cc` 里大量 CuTe 模板参数的**逐行装配**（约 4.5 千行）
  只给了代表性命中，未逐句展开。`docs/07` 从 device 模板侧覆盖了其中 Hopper wgmma 一半。
- **Hopper/Blackwell 只给骨架**：本文以通用/Ampere 版为主线；Hopper 的
  `hopper_async_pipeline` / TMA / `Hopper_Matmul`、Blackwell 的 `2sm_mma` / `launch_kernel_on_cluster`
  只标出调用点与行号，未逐行走读其内部。
- **未覆盖 runtime 模板内部**：`include/mirage/transpiler/runtime/**`（`kn::gemm` / `ElementUnaryKernel` /
  `ReductionKernel` / `InputAtom` / `Hopper_Matmul`）是**被生成代码调用的库**，本文只当作黑盒引用。
- **未覆盖 benchmark 数字**：`benchmark/`、`demo/` 里超级化器相对手写的加速（论文表）未在本机复现，
  故本文不引用任何性能数（与 `docs/15` §9 一致）。
- **`src/transpiler/*` 里的自引用文档**：`sched_tb_graph.cc:19/144` 等注释指向
  `docs/transpiler/transpiler.md`——该文件在 `f9eb70c` 快照中**未随仓库提供**（与 `docs/08` §9
  记的 `parallel_path_design.md` 缺失类似），故本文不引用其内容。

## 参考坐标

- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462-468、
  §4.2 行 1126-1146）。
- 源码快照：`github.com/mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。
  主文件：`src/transpiler/transpile.cc`（803 行）、`transpiler_kn.cc`（883 行）、
  `transpiler_tb.cc`（1544 行）、`transpiler_tb_hopper.cc`（1539）、`transpiler_tb_blackwell.cc`（1575）、
  `sched_tb_graph.cc`（402）、`plan_tb_swizzle{,_hopper,_blackwell}.cc`、`plan_stensor_memory.cc`（549）、
  `plan_dtensor_memory.cc`（62）、`resolve_tb_fusion.cc`（94）、`resolve_tensor_layout.cc`（818）、
  `resolve_dtensor_meta.cc`（63）；头 `include/mirage/transpiler/transpiler.h`（139）、
  `sched_tb_graph.h`（54）、`structs.h`；runtime 头 `include/mirage/transpiler/runtime/runtime.h`（60）；
  Python 侧 `python/mirage/kernel.py:396-530`、`python/mirage/_cython/core.pyx:1398-1444`。
- 同专题相关篇：`docs/07`（单 task device 实现，覆盖 device 模板一半）、
  `docs/08`（运行层依赖分析器，另一条下游）、`docs/14`（生产前端）、`docs/15`（搜索/验证）、
  `docs/16`（搜索器辅助模块）、`docs/17`（多卡/NVSHMEM 分支）。
