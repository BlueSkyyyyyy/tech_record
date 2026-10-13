# MPK 自动路径（runtime 库）：被生成 CUDA 调用的 device / thread-block 模板库走读

> 本文补齐 `docs/18`（P104-1）**§8 明留的尾巴**：
>
> > **未覆盖 runtime 模板内部**：`include/mirage/transpiler/runtime/**`（`kn::gemm` /
> > `ElementUnaryKernel` / `ReductionKernel` / `InputAtom` / `Hopper_Matmul`）是**被生成代码调用的库**，
> > 本文只当作黑盒引用。
>
> `docs/18` 走到「transpiler 生成什么 CUDA 源码」（模板名、循环结构、warp 角色、布局）就停了；
> 本文往下走**被生成源码调用/实例化的那层库**——`include/mirage/transpiler/runtime/**`（共 **23 个头、
> 3017 行**），即「一段能跑的快 kernel 里，元素级算子、G→S/S→R/S→S 拷贝、mbarrier 流水、wgmma/UMMA
> 究竟长什么样」。**纯源码走读（资料分析）**，不重新构建/运行 mirage。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`/`07`/`08`/`14`/`15`/`16`/`17`/`18`
> 同一快照，所有行号都据此核对（路径相对 `include/mirage/transpiler/runtime/`）。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（「快照论文」）。

## 0. 本文在链条里的坐标

`docs/18` §0 已给出下游调用链：`KNGraph.compile()` → `transpile()` → `generate_code()` →
`transpile_ugraph()` → 每个 customized op 生成一个 `__global__ void custom_kernel_N(...)`。
生成的 `.cu` **顶部 `#include "runtime.h"`**，本文就是走这个 `runtime.h` 及其 include 树。

关键分野是这套库的**两层 namespace**（对应两档「算子融合粒度」）：

```
runtime.h                                  // 两档 API  + 入口
├── kernel/**        namespace kn::         // 一整个张量算子 = 一次 <<<>>> launch
│   ├── element_unary.h / element_binary.h  // 逐元素（非融合 fallback）
│   ├── reduction.h                         // 整张量归约
│   └── matmul.h                            // 裸 GEMM → 直接调 cuBLAS
├── threadblock/**   namespace tb::         // 在一个 custom_kernel_N 内部，按 thread 调用的 run()
│   ├── input.h / output.h                  // G↔S 拷贝原子（sync/chunked/cp.async/TMA）
│   ├── element_unary.h / element_binary.h  // 逐元素（可带 epilogue 链）
│   ├── epilogues.h                         // 寄存器内 elementwise 链（register reuse）
│   ├── forloop_accum.h                     // 循环累加 / 清零 / max 归约
│   ├── reduction.h                         // 归约 + online-softmax 的 max/diff
│   ├── matmul.h / hopper_matmul.h / blackwell_matmul.h   // S2R→MMA→R2S
│   ├── pipeline.h / blackwell_pipeline.h   // mbarrier 流水（TMA / UMMA）
│   ├── profiler.h / utils.h                // 计时标签 / warpgroup 工具
│   └── threadblock.h                       // 聚合头
└── config.h / utils.h / nvshmem.h          // 宏配置 / 工具 / NVSHMEM 开关
```

**一句话**：`kn::` 是「没被搜索器选进融合 kernel 的算子的兜底实现（也可能是一次 cuBLAS launch）」；
`tb::` 是「真正被编进 `custom_kernel_N`、由线程在 SMEM/寄存器上反复调用的算子实现」。后者才是
论文 §4.2「thread-block 级超优化」的**运行期肉身**。

## 1. 入口层：`runtime.h`（60 行）与配置

`runtime.h` 只做三件事：

1. **前向声明两个由 transpiler 生成、不在库里定义的函数**——`_init()`（`:14`）与
   `_execute_mugraph(inputs, outputs, buf, stream, profiler_buffer)`（`:15-19`）。它们由生成的 `.cu`
   提供实现（`docs/18` §0）。
2. **include 全部 runtime 库**（`:22-28`：`config.h` → 四个 `kernel/*` → `threadblock/threadblock.h`
   → `utils.h`）。
3. **给外部两个入口**：`extern "C" void execute_mugraph(...)`（`:31-43`）用 `static bool inited`
   做**首调用一次 `_init()`**（`:37-41`）；`execute_mugraph_wrapper(...)`（`:47-60`）把 C 数组
   包装成 `std::vector` 供 Python 侧（`docs/18` §0 的 `HARD_CODE` pybind）调用。

配置层 `config.h`（16 行）是**生成器必须填的编译期宏**：`USE_PLACEHOLDER` 下给出默认
`USE_NVSHMEM 0` / `NUM_GPUS 1`（`:4-8`），否则 `static_assert(0)`（`:10-15`）——与 `docs/17` §2/§6
的多卡路径呼应（该库本身只认「是否 NVSHMEM / 几张卡」两个开关，多卡逻辑在别处）。

> 注意：`threadblock/utils.h:3` 里 `#include "../../../config.h"` 指的是**另一个** `config.h`
> （`include/mirage/config.h`，非本文目录的 runtime `config.h`），它定义 `NUM_THREADS_PER_WARP=32`、
> `NUM_WARPS_PER_GROUP=4`、`NUM_THREADS_PER_GROUP=128`（`include/mirage/config.h:36-38`）——即
> warpgroup 的常量。两个同名 `config.h` 是这套库第一个容易踩的坐标坑。

`utils.h`（68 行）是纯 host 工具：`CHECK_CUDA`（`:7-14`）、`ceil_div` / `ceil_div_cute`
（`:16-24`）、`round_to_multiple`（`:26-35`）、**`constexpr_for`**（`:37-43`，`template<...Start,End,Inc>`
的编译期展开，`reduction.h` 用它做秩检查）、`PRINT_RUNTIME_WARNING_ONCE`（`:54-61`）、
`SWAP`（`:63-68`，`docs/18` §5 记的双缓冲 `SWAP` 就来自这里）。

## 2. `kn::` 层：整张量算子的兜底实现

这一层的四个算子各是「**一整个张量 = 一次 grid launch**」，是没进入融合 kernel 的算子实现。

- **`kernel/element_unary.h`**（87 行）：`ElementUnaryOpType {EXP,SILU,GELU,RELU,CLAMP,SQUARE,SQRT}`
  （`:12`）与 `perform_element_unary_op`（`:14-38`，`if constexpr` 分派，fp 都先转 `float`）。
  `element_unary_kernel_fwd`（`:47-61`）**每线程一个元素**（`idx = blockIdx.x*blockDim.x+threadIdx.x`），
  经 `src_layout(idx)`/`dst_layout(idx)` 做物理下标重排；`ElementUnaryKernel::run`（`:81-84`）
  用固定 `BLOCK_SIZE=512`（`:77`）`<<<grid_shape, block_shape>>>` launch。
- **`kernel/element_binary.h`**（104 行）：广播机制在这里——`ReplacementFunctionClass`（`:52-59`）
  把源张量**广播维的 stride 替换成 0**，从而 `Src0InDstLayout = make_layout(shape(Dst),
  transform(shape/stride(Src0), ReplacementFunctionClass{}))`（`:80-89`）让「按 dst 逻辑下标索引 src」
  自动落到正确的广播元素。
- **`kernel/reduction.h`**（91 行）：`reduction_kernel_fwd`（`:16-34`）每线程产出一个输出元素、
  内部 `CUTE_UNROLL` 累加 `REDUCTION_FACTOR` 个（`:28-31`）；`DstInSrcLayout`（`:66-70`）由
  `replace<REDUCTION_DIM>(stride(Src), 起始偏移)` 构造。
- **`kernel/matmul.h`**（151 行）：最值得注意——**`kn::gemm` 直接包 `cublasGemmStridedBatchedEx`**
  （`:117-140`），带一个全局 `CublasHandleKeeper`（`:48-62`）与 `data_t2cuda_data_type_t`（`:8-21`），
  并用 `trans_C` 判断做行/列主序的 `m,n` 与 A/B 交换（`:99-110`）。也就是说：**没被搜索器编进融合
  kernel 的 GEMM，运行期走的是一条 cuBLAS launch，而不是 CuTe 模板**——这与 §3 的 `tb::Matmul`
  是「同一种算子、两条实现路径」。

## 3. `tb::` 层：一个 `custom_kernel_N` 内部的运行期肉身

`threadblock/threadblock.h`（16 行）把所有 `tb::*` 头聚合起来。这一层的每个算子都提供
`static __device__ run(...)`，**由生成代码在 kernel 主体里按 thread 调用**（`docs/18` §5 的循环体）。

### 3.1 G↔S 拷贝原子：`input.h`（397 行）/ `output.h`（33 行）

`input.h` 给出四类拷贝原子（头注释 `:3-13` 说明挑选依据是张量布局 + 硬件）：

- `InputNonChunkedSyncCopy`（`:46-63`）：逐元素 `dst[dst_layout(i)] = src[src_layout(i)]`。
- `GetChunkedCoord2Coord`（`:74-88`）+ `InputChunkedSyncCopy`（`:92-119`）：按 `CHUNK_SIZE=16/sizeof(T)`
  把最内维切成 16 B chunk（`uint128_t`，`:112-116`）。
- `InputChunkedAsyncCopy`（`:122-155`）：同 chunk 化，但用 **`cp.async.cg.shared.global.L2::128B`**
  一条内联汇编发 16 B 异步拷贝（`:150-152`），并做 `cast_smem_ptr_to_uint`（`:140`）。
- `InputTMAAsyncCopy`（Hopper，`:166-254`）：`prefetch_tma_descriptor`（`:190-192`）、
  `make_coord_runtime<6|7>`（`:194-216`）把 `imapx/y/z_a` 映射成 `blockIdx`、由 lane0 发 TMA
  （`:225`），经 `pipeline.producer_acquire()` 拿 `(tma_barrier, write_stage)` 再
  `copy(tma_a.with(*tma_barrier), ...)`（`:247-251`）。
- `InputTMAAsyncCopy_Blackwell`（`:257-395`）：换成 `UMMA::Major` 的 `sm100_smem_selector`
  （`:276-283`），并多出 **TMA multicast mask**（`:353-360` 的 `create_tma_multicast_mask<1|2>`）
  与 `tma_partition` 的 cluster 协作（`:362-384`）——对应 `docs/17`/生产 Blackwell 路径的 cluster 形态。

`output.h` 把前两类**直接 alias 回 input**（`:22-29`），TMA 输出只留注释占位（`:31`）。

### 3.2 逐元素 + epilogue 链：寄存器复用（register reuse）的落点

- `tb::element_unary.h`（85 行）：`ElementUnaryKernel::run`（`:67-82`）按
  `elem_idx += NUM_THREADS` 跨步扫元素，算完 `perform_element_unary_op` 后**立刻交给
  `Epilogue::run(res, dst, dst_phy_pos, epilogue_scalars)`**（`:80`）。
- `tb::element_binary.h`（99 行）：`DstCoord2SrcCoordGetter`（`:37-45`）算出「dst 逻辑坐标 → src
  逻辑坐标」，再 `composition(SrcLayout, ...)` 成物理下标（`:67-72`）；run 里对每个输入做
  `elem_idx % srcN_numel` 实现广播（`:89-90`）。
- `tb::epilogues.h`（154 行）就是 `docs/18` §5 所说的「CuTe epilogue 模板链」：所有尾算子
  `Epilogue{Exp,SILU,GELU,RELU,Clamp,Square,Sqrt,MulScalar}` 都以 `NextEpilogue::run(x, ...)` 串联
  （`:41-152`），末端是 `EpilogueStore` / `EpilogueStoreAccum`（`:35-39`，`IS_ACCUM` 决定 `=` 还是
  `+=`，`:27-31`）。每个中间算子把 `epilogue_scalars` **指针前移一格**（如 `:51` 的
  `++epilogue_scalars`），给 `MulScalar` 之类取到自己的标量。这就是论文 §4.2「register reuse 让中间
  结果不落 SMEM」的**运行期实现**——整条链是编译期展开的模板调用，中间值只在寄存器里流动。

### 3.3 循环累加 / 归约：`forloop_accum.h`（115 行）/ `reduction.h`（152 行）

`forloop_accum.h` 是「跨 K/forloop 轮次的累加器」五种：`ClearAccumlatorKernel`（`:12-28`，按
`uint128_t` 清零）、`InitMaxAccumulatorKernel`（`:32-42`，填 `lowest()`）、`ForloopAccumKernel`
（`:44-60`）、`ForloopAccumRescaleKernel`（`:62-93`，`accum = accum*rescale + src`）、
`ForloopAccumMaxKernel`（`:95-113`）。

`reduction.h` 除通用 `ReductionKernel`（`:13-67`，同样带 `Epilogue::run`，`:64`）外，还有一对
**online-softmax 专属**：`InitReductionMaxKernel`（`:70-80`）与 `ReductionMaxKernel`（`:82-150`），
后者一次扫过归约段同时更新 `updated_max` 与 `diff = old_max - new_max`（`:129-147`）——正是 flash-attention
在线 softmax 里「运行 max + 重标定残差」的两个量。这说明 runtime 库把 attention 的 fused 路径也抽象进去了。

### 3.4 S2R→MMA→R2S：`matmul.h`（464 行）/ `hopper_matmul.h`（229 行）/ `blackwell_matmul.h`（236 行）

三档算子的骨架一致，但**累加器所在存储**不同：

- 通用 **`tb::Matmul`**（`matmul.h:256-462`）：CuTe 三件套——`TiledMMA`（`:296-306`）、
  **S2R copy atom 选择器 `S2RTiledCopySelector`**（`:49-98`：按 K 维是否连续 + ldmatrix 可用性选
  `SM75_U32x4_LDSM_N` / `LDSM_T` / `UniversalCopy`）、**R2S 选择器 `R2STiledCopySelector`**
  （`:102-112`：stmatrix **尚未支持**，恒 `UniversalCopy<uint16_t>`）。`s2r_copy_with_oob_protection`
  （`:114-171`）在 tile 不整除时用「全零 SMEM 张量」替换越界片（`:156-168`）；`r2s_copy_with_oob_protection`
  （`:173-254`）在写回时顺手把 `NUM_EXPS_BEFORE_STORE` 个 `exp` 融进寄存器链（`:200-205`）、并按
  `IS_STORE_ACCUM` 选 `+=`/`=`（`:206-210`）。主循环 `run`（`:367-461`）用宏 `S2RCOPY(k_idx)`
  （`:435-447`）把 **S→R 拷贝与 MMA 软件流水两级**（`:449-458`），结尾 `__syncthreads()`（`:460`）。
- **`tb::Hopper_Matmul`**（`hopper_matmul.h:28-227`）：MMA 换成 `GMMA::ss_op_selector`（`:94-103`），
  `IS_COORPERATIVE` 时 `AtomLayoutMNK = Shape<_2,_1,_1>` 让 **2 个 CTA 协作**（`:90-92`、
  TILED_MMA_NUM_THREADS = NUM_THREADS*2，`:106-107`）。`run`（`:153-226`）用
  `warpgroup_arrive / gemm / warpgroup_commit_batch / warpgroup_wait<0>`（`:222-225`）发 wgmma，
  并支持 `IS_PIPELINE_A/B` 的读 stage（`:191-192`）。注意 `:163` 的
  `cutlass::arch::warpgroup_reg_alloc<192>()` 被注释——与 `docs/07`/`docs/18` §5 记的
  「`setmaxnreg` 被注释」是同一工程事实。
- **`tb::Blackwell_Matmul`**（`blackwell_matmul.h:21-234`）：UMMA + **张量内存（TMEM）累加器**。
  `get_mma_tC`（`:125-137`）把累加器 `data() = tmem_base_ptr`；`write_tC_to_sC`（`:157-192`）走
  「**TMEM→RMEM→SMEM**」两跳（`sm100_get_tmem_load_op` / `sm100_get_smem_store_op`，`:102-115`），
  只用一个 warpgroup（`:162`）。`run`（`:195-233`）里 `warp_id()==0` 单 warp 发 tcgen05 MMA，且
  `k_iter==0` 时 `accumulate_ = UMMA::ScaleOut::Zero`、之后置 `One`（`:208-230`）——用 MMA 指令本身
  完成「首轮清零」，省掉独立清零 kernel。

### 3.5 流水与工具：`pipeline.h` / `blackwell_pipeline.h` / `profiler.h` / `utils.h`

- `pipeline.h`（88 行）：`HopperAsyncPipeline<Stage>` 包 `cutlass::PipelineTmaAsync<Stage>`
  （`:21-86`），对外只暴露 `producer_acquire/advance`、`consumer_wait/release`；`PipelineStorage`
  （`:8-19`）把 `SharedStorage` 落在 caller 给的 SMEM 偏移上。
- `blackwell_pipeline.h`（94 行）：`BlackwellAsyncPipeline` 换 `PipelineTmaUmmaAsync<Stage,
  ClusterShape, AtomThrShape_MNK>`，构造时 `pipeline_init_arrive_relaxed/wait(size(ClusterShape))`
  （`:53-56`）——**cluster 级 barrier 初始化**，是 Blackwell cluster 协作的落点。
- `utils.h`（120 行）：`lane_id/warp_id/warpgroup_id`（`:8-31`）、**`wg_decrease_regs/wg_increase_regs`**
  （`:33-51`：仅 `MIRAGE_GRACE_HOPPER` 下发 `setmaxnreg.dec/inc`，否则 `brkpt`）、
  `wg_sync/wg_arrive`（`:53-72`）、`wait_barrier`（`:74-92`，`mbarrier.try_wait.parity` 自旋）、
  以及 cluster 坐标辅助 `get_cluster_layout`/`get_mma_coord_vmnk`/`get_cta_mma`（`:94-118`）。
- `profiler.h`（156 行，FlashInfer 版权头 `:1-15`）：`encode_tag`（`:42-47`）把
  `block_group_idx|event_idx|event_type` 位打包进 u32，`get_timestamp` 读 `%globaltimer_lo`
  （`:70-74`），`PROFILER_EVENT_START/END/INSTANT` 宏（`:90-154`）受 `MIRAGE_ENABLE_PROFILER` 门控。

## 4. 生成代码与模板的接口约定

把 `docs/18` §5 的「生成什么」与本文的「库长什么样」对上，接口约定可以收敛成四条：

1. **入口**：生成的 `.cu` 定义 `_init()`/`_execute_mugraph()`，并 `#include "runtime.h"`（§1）。
2. **每个 customized op = 一个 `custom_kernel_N`**：kernel 主体从 `extern __shared__ char buf[]`
   切出 `stensor$_ptr`，对输入调 `tb::*Input*Copy::run(...)`、对算子调 `tb::*Kernel::run(...)`，
   `docs/18` §5 列的 `InputAtom::run` / `EpilogueStore<...>` / `hopper_async_pipeline_N.consumer_wait()`
   都能在本文找到定义（§3.1 / §3.2 / §3.5）。
3. **流水结构由编译期模板参数给定**：`HopperAsyncPipeline<Stage>`（§3.5）、`IS_PIPELINE_A/B` 与
   `PIPELINE_STAGES`（`hopper_matmul.h:83-84`）——正是 `docs/18` §5 记的
   `pipeline_stages × num_warp_groups` 两个编译期旋钮的落点。
4. **兜底**：没被融合的算子走 `kn::` 层（§2），其中 GEMM 甚至落到 cuBLAS（`kernel/matmul.h`）。

## 5. 与 `docs/07`（生产 task）的关系：同一套硬件惯用法、实现了两遍

这是本文最有价值的一条观察。`docs/07` 走读的是 **MPK 生产前端**的 task——`tasks/hopper/*.cuh`
里的手写模板（`linear_kernel_hopper` / `rmsnorm_hopper` / `silu_mul_hopper`），由
`task_register.cc:1181-1318` 实例化、在常驻 kernel 的 `_execute_task` 里调用。本文走读的是
**Mirage 通用编译器**的 runtime 库。两者是**同一套硬件惯用法（warp specialization + TMA +
wgmma/UMMA + mbarrier 流水 + 寄存器内 epilogue）在同一个仓库里的两份实现**，差别在：

| 维度 | MPK 生产 task（`docs/07`） | transpiler runtime 库（本文） |
|---|---|---|
| 何时被谁调用 | 常驻 kernel 的 worker，按 event 派发（`docs/06`） | 生成的 `custom_kernel_N`，kernel-per-op 各 launch 一次 |
| 张量形状怎么进来 | 编译期常量 + `TaskDesc` 的 `CUtensorMap*`（`task_register.cc:1279-1292`） | 模板参数（`Config`/`DstLayout`/`SmemLayoutA_`…），`kn::run` 直接从实参取 |
| TMA 流水级 | `Kstages = output_size>=256?3:6`（每 task 固定） | `HopperAsyncPipeline<Stage>` / `PIPELINE_STAGES` 模板参数 |
| `setmaxnreg` | `utils.cuh` 里被注释（`docs/07` §2） | `wg_decrease/increase_regs` 在，但 `hopper_matmul.h:163` 的 `warpgroup_reg_alloc` 被注释 |
| 跨算子协调 | 有（in-kernel 事件队列 / scheduler，`docs/06`） | 无（跨 op 由 host 多次 launch 顺序保证） |

**结论**：`docs/15` §7 记的「工程落差」在这里有了运行时证据——要写一个很快的**单算子 kernel**，
Mirage 仓库里现成有**两套**成熟实现（手写 MPK task + transpiler runtime 模板）；缺的不是 device
代码的功力，而是把这些好算子**编成一张图、常驻着去中心化调度**的那一半（`docs/06`/`docs/15`）。

## 6. 与论文 §4.1/§4.2 的对照

| 论文（快照论文行） | runtime 库落点 | 一致性 |
|---|---|---|
| §4.2「每 task 生成 CUDA，含 software pipelining」（行 1138-1142） | `matmul.h:449-458` 的 S2R/MMA 两级 + `pipeline.h` 的 TMA mbarrier 环 + `blackwell_pipeline.h` 的 UMMA cluster barrier | 一致 |
| §4.2「register reuse」（行 1140） | `epilogues.h` 的 `Epilogue*→EpilogueStore` 寄存器链 + `matmul.h:200-205` 的 `NUM_EXPS_BEFORE_STORE` | 一致 |
| §4.2「layout optimizations to reduce shared-memory bank conflicts」（行 1140-1142） | `input.h` 的 `*_smem_selector`（`InputTMAAsyncCopy:173-178`）、`matmul.h` 的 `Dim01Swapper`（`:35-47`）、`S2R/R2S` copy atom 选择 | 一致（swizzle 由 CUTLASS selector 给出，`docs/18` §4 的 plan 决定参数） |
| §4.2「thread-block 级而非 kernel 级超优化」（行 1132-1134） | 整个 `tb::` 层以 **thread 索引 + SMEM 指针** 为界（`run(..., int thread_idx)`） | 一致 |
| §4.1「按输出张量分区成 task」（行 462-468） | 不在本库：分区在搜索器（`docs/15`），本库只被生成代码实例化 | 语义一致、分工不同（同 `docs/18` §6） |

## 7. 判据 C1–C5 映射

与 `docs/14`/`15`/`18` 一样，runtime 库**只贡献 C3（融合的范围/粒度）**：

| 判据 | 本库是否贡献 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | `kn::*Kernel::run` 各自 `<<<>>>`；`tb::` 只是被 `custom_kernel_N` 内联调用 |
| C2 常驻 | 否 | 全是「跑完即退」的算子体，无常驻循环 |
| C3 多算子范围 | **是（实现载体）** | `tb::` 的拷贝/逐元素/epilogue/归约/matmul 原子决定单个 customized op 能融合到哪一步 |
| C4 设备内协调 | 否 | `mbarrier`/cluster barrier 只在**单 kernel 内**同步，无跨 kernel 事件队列 |
| C5 SM 级调度 | 否 | 无 worker/scheduler、无去中心化调度 |

## 8. 局限

- **纯源码走读，未构建/运行**：不采信任何性能数；所有结论基于 `f9eb70c` 快照行号。
- **未逐行展开 CuTe 模板装配**：`matmul.h`/`hopper_matmul.h`/`blackwell_matmul.h` 的 `mma_atom`/
  `copy_atom`/`partition_*` 只给代表性命中，未逐句证明其 tile/layout 正确性。
- **未覆盖 `triton_kernels.py`**：本目录还有一个 Python 侧 `triton_kernels.py`（非 C++ 头），本文未走。
- **未覆盖 `src/transpiler/*` 对模板参数的生成逻辑**：本文只讲「库提供什么」，`docs/18` 讲「生成器
  传什么」，两者的**参数装配逐行对应**仍未做。
- **未覆盖 benchmark 数字**：与 `docs/15`/`docs/18` 一致，不引用未复现的性能数。
- **Hopper/Blackwell 只给骨架**：`Hopper_Matmul` 的 cooperative 2-CTA、`Blackwell_Matmul` 的
  tcgen05/TMEM 路径只标出调用点与行号，未展开内部。

## 参考坐标

- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462-468、
  §4.2 行 1126-1146）。
- 源码快照：`github.com/mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。
  本目录：`include/mirage/transpiler/runtime/`（23 头、3017 行）——`runtime.h`（60）、
  `config.h`（16）、`utils.h`（68）、`nvshmem.h`（9）；`kernel/`：`element_unary.h`（87）、
  `element_binary.h`（104）、`matmul.h`（151）、`reduction.h`（91）；`threadblock/`：
  `matmul.h`（464）、`input.h`（397）、`blackwell_matmul.h`（236）、`hopper_matmul.h`（229）、
  `profiler.h`（156）、`epilogues.h`（154）、`reduction.h`（152）、`forloop_accum.h`（115）、
  `blackwell_pipeline.h`（93）、`pipeline.h`（88）、`element_binary.h`（98）、`element_unary.h`（84）、
  `utils.h`（120）、`output.h`（33）、`threadblock.h`（16）。相关常量 `include/mirage/config.h:36-38`。
- 同专题相关篇：`docs/07`（MPK 生产 task 的同类硬件惯用法）、`docs/15`（superoptimizer）、
  `docs/16`（搜索器辅助）、`docs/18`（transpiler codegen，本文的上游）。
