# MPK 单测 wrapper 家族走读：blackwell 家族与 cute/hopper 的两类落地

> 补齐 `docs/29-mpk-unit-tests.md`（P115-1）**§7 明留的尾巴**——「§3 只精读 `moe_linear`
> （并旁及 `sampling_from_logits`/`multitoken_paged_attention` 的 launch 形态），**未逐行走读 12 个
> blackwell wrapper、`cute/hopper/gemm.cu` 与 `attention_tail_wrapper.cu` 的内部**」。
>
> 快照：`mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/29` 同快照）。
> **纯源码走读（资料分析），本轮无新实验**——target SM100 在目标卡 H100/sm90 不可执行
> （与 `docs/17`/`docs/21` 同口径），本层也只是「一个 device task = 一整颗 kernel」的 host 侧回归网，
> 不经 `PersistentKernel`。

本文把 `docs/29` §7 那条「只抽样解剖」的尾巴补完：§1 给 blackwell 家族单测的**全景计数与构建宏**，
§2 给**扩展名/符号/测试调用三方对照**（headline：与 `docs/29` §2 的「当前代际对齐」结论有一处重要修正），
§3 解剖「一个 task = 一颗 kernel」在 SM100 上的**两种落地形态**，§4 走 MoE wrapper 的 CuTe/TMEM 内部，
§5 走 `cute/hopper/gemm.cu`（CUTLASS 路线），§6 走 `hopper/attention_tail_wrapper.cu`（`extern "C"` 路线），
§7 复核 `hopper-moe`，§8 判据与定位，§9 局限。

## 0. 定位：本文补的是 `docs/29` 的哪一块

`docs/29` 已经把 `tests/runtime_python/` 这一层**总览**过：五套彼此独立的 pybind extension、主 / hopper
wrapper 的注册失配、`MIRAGE_UNIT_TEST` 宏、单测 = 「一个 device task 当整颗 kernel `<<<>>>` launch」。
它**没**打开的是：**当前代际（SM100/blackwell）**那 14 个目录里到底有几套 extension、各自注册什么、
测试怎么调、以及除 pybind 外的第二条路线（`extern "C"` 动态库 / CUTLASS standalone）。本文走这三块。

坐标（`docs/29` §1 已给目录树，本文只补规模）：`tests/runtime_python/blackwell/` 下——
**14 个目录、18 个 `*.cu`（含 3 个 standalone host 程序）、13 个 `setup*.py`，共 6876 行**
（`find blackwell -name '*.cu' -o -name 'setup*.py' | xargs wc -l`）。

## 1. blackwell 家族单测全景：13 套扩展 + 3 个 standalone 程序

按「构建单元」归类（`setup.py` 的 `name=` 即 Python 模块名，`CUDAExtension(sources=...)` 即编译源）：

| 目录 | 扩展名 / 产物 | 源文件 | gencode | 备注 |
|---|---|---|---|---|
| `sm100_linear` | `runtime_kernel_blackwell` | `runtime_kernel_wrapper_sm100.cu`（665） | `compute_100a,sm_100a` | **与 `sm100_moe` 撞名**（§2） |
| `sm100_moe` | `runtime_kernel_blackwell` | `runtime_kernel_wrapper_sm100.cu`（710） | `compute_100a,sm_100a` | **同名**、符号不同 |
| `sm100_moe_sigmoid` | `runtime_kernel_moe_sigmoid` | `runtime_kernel_wrapper_sm100.cu`（212） | `compute_100a,sm_100a` | 另有 testmode 用例 |
| `sm100_linear_fp8` | `runtime_kernel_blackwell_linear_fp8` | `runtime_kernel_wrapper_sm100.cu`（1107） | `compute_100a,sm_100a` | 配 `sm100_fp8_scale_layout` |
| `sm100_quantize_fp8` | `runtime_kernel_blackwell_quantize_fp8` | `runtime_kernel_wrapper_sm100.cu`（182） | `compute_100a,sm_100a` | ditto |
| `sm100_paged_attention` | `runtime_kernel_paged_attention` | `runtime_kernel_wrapper_sm100.cu`（166） | `compute_100a,sm_100a` | |
| `sm100_fp8_moe` | `runtime_kernel_fp8_moe` + `runtime_kernel_bf16_moe` | `runtime_kernel_wrapper.cu`（810）+ `_bf16.cu`（363） | `compute_100a,sm_100a` | **一个 setup 两个 ext** |
| `sm100_mla` | `runtime_kernel_blackwell_mla` | `runtime_kernel_wrapper_sm100_mla.cu`（314） | `compute_100a,sm_100a` | 另有 3 个独立 setup |
| `sm100_mla` | `runtime_kernel_mla_decode` | `runtime_kernel_wrapper_mla_decode.cu`（176） | `-arch=sm_100a` | `setup_mla_decode.py` |
| `sm100_mla` | `runtime_kernel_mla_prefill` | `runtime_kernel_wrapper_mla_prefill.cu`（122） | `-arch=sm_100a` | `setup_mla_prefill.py` |
| `sm100_mla` | `runtime_kernel_mla_mtp` | `runtime_kernel_wrapper_mla_mtp.cu`（212） | `-arch=sm_100a` | `setup_mla_mtp.py` |
| `sm100_mla_prefill_tp8` | `runtime_kernel_mla_prefill_tp8` | `runtime_kernel_wrapper_mla_prefill_tp8.cu`（**缺失**） | `-arch=sm_100a` | **源文件不存在 → 构建必失败**（§2） |
| `sm100_mla_mtp_decode` | （无 setup、无 pybind） | `test_tp{2,4,8}_wrapper.cu`（360/362/369） | — | **standalone host 程序（含 `int main`）** |
| `sm100_dflash_attention` / `sm100_glm4_moe` / `sm100_inkling` | （无扩展） | 无 `.cu` | — | 纯 testmode 用例，走 `PersistentKernel` |

**构建宏**：blackwell 家族普遍带 `-DMIRAGE_GRACE_BLACKWELL`（`cxx` 与 `nvcc` 两处），且 `linear`/`moe`/
`moe_sigmoid`/`paged_attention`/`fp8_moe`/`linear_fp8`/`quantize_fp8` 都带 `-DMPK_ENABLE_TMA`
（`sm100_linear/setup.py:47-49`、`sm100_moe/setup.py:52-54`、`sm100_moe_sigmoid/setup.py:50-52`、
`sm100_fp8_moe/setup.py:35-38`、`sm100_linear_fp8/setup.py:90-92`、`sm100_quantize_fp8/setup.py:90-92`、
`sm100_paged_attention/setup.py:47-49`）。这与 `docs/21` 走读的 Blackwell codegen 目标一致：
**SM100 单测的 device 代码本就是 TMA + tcgen05 形态**（§4）。MLA 系列只给 `-arch=sm_100a`（不带
`GRACE_BLACKWELL`，因为 `mla_sm100_2sm.cuh` 等自带头文件门控，`sm100_mla/setup.py:31-37` 用 `depends=`
显式声明对 `tasks/blackwell/mla_sm100_2sm.cuh` 的依赖）。

## 2. 扩展名 / 符号 / 测试调用：三方对照（headline）

`docs/29` §2 的结论是「**主 / hopper 通用历史层大面积失配、hopper-moe 与 blackwell 家族逐条对齐**」。
逐目录核过 blackwell 后，这条要**修正**：**blackwell 家族当前代际也有失配，而且是一种更隐蔽的形态**。

### 2.1 同名扩展撞名：`runtime_kernel_blackwell`（`sm100_linear` vs `sm100_moe`）

- `blackwell/sm100_linear/setup.py:22/25` 与 `blackwell/sm100_moe/setup.py:25/28` **都** `name=
  'runtime_kernel_blackwell'`；
- 但两个源文件注册的符号完全不同：
  - `sm100_linear/runtime_kernel_wrapper_sm100.cu:660-665` 注册 `linear_sm100_mpk`、`linear_splitk_sm100`；
  - `sm100_moe/runtime_kernel_wrapper_sm100.cu:697-711` 注册 `topk_softmax_sm100`、`moe_w13_linear_sm100`、
    `moe_w2_linear_sm100`、`mul_sum_add_sm100`、`silu_mul`；
- 两侧测试都**裸 `import runtime_kernel_blackwell`**并直接点符号：
  `sm100_linear/test_matmul_mpk.py:2,30` 调 `linear_sm100_mpk`、`test_matmul_splitk.py:2,30` 调
  `linear_splitk_sm100`；`sm100_moe/test_w13_linear.py:2,72` 调 `moe_w13_linear_sm100`、
  `test_gate_topk.py:32` 调 `topk_softmax_sm100`、`test_silu_mul.py:25` 调 `silu_mul`。

两个同名 `.so` 落在同一 `sys.path` 里，**只能存在一个**——最后安装/编译的那个覆盖另一个，于是另一组的
测试会 `ImportError`（模块来自对方）或 `AttributeError`（符号缺失）。这不是「注册少几个」式的失效，而是
**两个不同职责的扩展共用一个名字**，是比 `docs/29` §2 主 wrapper 那种「注册表裁剪」更难在 CI 里定位的
问题（症状随安装顺序而变，且不是缺符号而是**符号来自另一个扩展**）。本文只记录这个静态冲突事实，
**不判对错**（可能是两个目录从不同 build 目录分别 `sys.path.insert` 隔离运行的）。

### 2.2 源文件不存在的扩展：`runtime_kernel_mla_prefill_tp8`

`blackwell/sm100_mla_prefill_tp8/setup.py:17` 的 `sources=[... 'runtime_kernel_wrapper_mla_prefill_tp8.cu']`，
但该目录下**只有 `setup.py` 与 `test_mla_prefill_tp8.py`**（`find` 无此 `.cu`）。→ `pip install`
/ `setup.py build` 在该目录**必失败**（找不到源文件）。而测试 `test_mla_prefill_tp8.py:17` 仍
`import runtime_kernel_mla_prefill_tp8 as ext`。这是与 `docs/29` §2「导入即失败」同型、但**更早一步**
（连编译都起不来）的条目。

### 2.3 无 setup / 无 pybind 的 standalone host 程序：`sm100_mla_mtp_decode`

`blackwell/sm100_mla_mtp_decode/` 下只有 `test_tp{2,4,8}_wrapper.cu`，**没有 `setup.py`、没有
`PYBIND11_MODULE`**。三个文件都以 `int main` 结尾（`test_tp2_wrapper.cu:145`），是**独立可执行测试**——
注释自陈（`test_tp2_wrapper.cu:1-7`）：*"Standalone test for the device-function port of MLA MTP TP=2
(no PDL) … verify the port … is correctness-equivalent and perf-equivalent … before doing MPK
plumbing."* 它们用 `thin __global__ shim`（`:24-67`，把 `blockIdx` 转发给 `__device__` 的
`mla_mtp_tp2_main`/`mla_mtp_tp2_reduce`）直接 `nvcc` 编成可执行，与 pybind 路线并行。→ 这一层
**不在 pytest 收集范围内**（无 `test_*` 由 pytest 采？——文件名是 `test_*_wrapper.cu`，非 `.py`，
pytest 不会收集 `.cu`），是「开发者手动 `nvcc && ./a.out`」的调试产物。

### 2.4 逐条对齐的部分

除去 §2.1–§2.3，其余目录**扩展名 ↔ 模块名 ↔ 测试导入 ↔ 注册符号**一致：

| 目录 | setup `name` | 测试导入 | 注册符号（行号） |
|---|---|---|---|
| `sm100_moe_sigmoid` | `runtime_kernel_moe_sigmoid` | `test_gate_topk_sigmoid.py:2` | `topk_sigmoid_sm100`/`topk_softmax_sm100`（`:205-212`） |
| `sm100_quantize_fp8` | `runtime_kernel_blackwell_quantize_fp8` | `test_quantize_fp8.py:5` | `quantize_fp8_sm100`/`supported_hidden_sizes`/`supported_group_sizes`（`:168-182`） |
| `sm100_linear_fp8` | `runtime_kernel_blackwell_linear_fp8` | `test_linear_1d2d_fp8.py:5` | `linear_fp8_1d2d_sm100`/`supported_dense_gemm_shapes`（`:1100-1106`） |
| `sm100_paged_attention` | `runtime_kernel_paged_attention` | `test_window_skip.py:21` | `paged_attention_sm100`（`:162`） |
| `sm100_fp8_moe` | `runtime_kernel_fp8_moe` | `test_fp8_moe_gemm.py:25` | `fp8_moe_gemm_test`/`fp8_moe_w2_gemm_test`/`fp8_moe_gemm_2d`/`*_bench_{setup,launch,cleanup}`（`:791-810`） |
| `sm100_mla`（主） | `runtime_kernel_blackwell_mla` | `test_mla_attention.py:18` | `mla_attention`/`get_workspace_size`/`is_mla_available`/`get_mla_info`（`:291-314`） |
| `sm100_mla`（decode） | `runtime_kernel_mla_decode` | `test_mla_decode.py:22` | `mla_init`/`mla_run`/`mla_decode_test`（`:172-175`） |
| `sm100_mla`（prefill） | `runtime_kernel_mla_prefill` | `test_mla_prefill.py:17` | `mla_prefill_init`/`_run`/`_test`（`:118-121`） |
| `sm100_mla`（mtp） | `runtime_kernel_mla_mtp` | `test_mla_mtp.py:23` | `mla_mtp_init`/`_run`/`_test`（`:208-211`） |

统计口径：`blackwell/` 下 **26 个 `test_*.py`**——**8 个**走 `PersistentKernel`（`grep -rl
"PersistentKernel" ... --include=test_*.py`，全在 `sm100_{moe_sigmoid,glm4_moe,inkling,dflash_attention}`，
即 `docs/28` 的 testmode 层），**18 个**裸 import 扩展（本节四张表覆盖的 wrapper 层）。两层的分界与
`docs/29` §4 一致：testmode = 整图端到端，wrapper = 单 task 当整核。

## 3. 「一个 task = 一颗 kernel」在 SM100 上的两种落地

### 3.1 形态 A：`__global__` shim + 单 CTA + cluster launch（主流，MoE/linear/attention）

黑/白代际的单测 wrapper 统一长这样（以 `sm100_linear` 为例）：

- 一个薄 `__global__ __launch_bounds__(256, 1) linear_sm100_mpk_wrapper`（`:54-56`），函数体只做
  「转发 `blockIdx` + 构造 CuTe tensor + 调 `__device__` task impl」；
- host 侧 `launch_linear_sm100_mpk`（`:147`）用 **`dim3 grid_dim(1,1,1)`、`block_dim(256,1,1)`、
  `cluster_dim(1,1,1)`、`smemBytes = 224*1024`**（`:263-266`），经
  `cutlass::launch_kernel_on_cluster`（`:282/307`）发射。

即：**grid 只有 1 个 CTA、占 224 KB 动态 SMEM 上限、走 cluster launch API（哪怕 cluster 维是 1）**。
`sm100_moe/runtime_kernel_wrapper_sm100.cu` 同形（`:295-297`、`smemBytes=224*1024`、四个 `if constexpr
(W13_LINEAR)` × `residual` 分支各一次 `launch_kernel_on_cluster`，`:326-449`）；`mul_sum_add_sm100`
（`:623-625`）、`silu_mul`（`:678-680`）亦同。**一个 tile / 一次子问题 = 一颗只含 1 CTA 的 kernel**，
正是 `docs/29` §0 headline 说的「task = 一整颗 kernel」——只是换到 SM100 后，launch 走 cluster API。

### 3.2 形态 B：`extern "C" launch` 动态库（`hopper/attention_tail_wrapper.cu`）

`hopper/attention_tail_wrapper.cu` **没有** `PYBIND11_MODULE`，导出的是 C ABI：`extern "C" int launch(...)`
（`:74-91`）。它把生产 `multitoken_paged_attention_hopper_impl`（`:7` include）包成一个 `__global__
run_attention<MAX_TOKENS>`（`:13`），并**先用 `poison` 把动态 SMEM 填成 `0x7fffffff`**（`:20-24`，
注释 *"NaN in both bf16 and float32 interpretations"*）再调 task——这是 `docs/29` §3 那类单测
「隔离 + 污染」做法的显式版。`launch_attention`（`:60-73`）设 `cudaFuncAttributeMaxDynamicSharedMemorySize`
= `SMEM_BYTES = 205824`（`:10`）后 `<<<1, 256, SMEM_BYTES>>>`（`:69`）并 `cudaDeviceSynchronize`。
调用方 `hopper/test_attention_tail_hopper.py:54-56` 直接用 `nvcc` 把它编成 `attention_tail.so` 再
`ctypes`/子进程调用——**第二条「非 pybind」的 device task 单测路线**（`docs/29` §2 只覆盖 pybind 那条）。
`MAX_TOKENS` 只在 `1` / `8` 两个编译期实例间分派（`:84-90`）。

## 4. MoE wrapper 内部：一个 SM100 task 的 CuTe/TMEM 形态

打开 `sm100_moe/runtime_kernel_wrapper_sm100.cu:150-200`：`moe_linear_sm100_wrapper` 在 `__device__`
里构造 `TMA_A = kernel::tma::tma_2d<bfloat16, ...>`（`:170-184`，64 B cp 粒度、`swizzle 128`），再调
`kernel::moe_linear_sm100_task_impl<T, TMA_A, ..., MMA_M, MMA_N, BATCH_SIZE, OUTPUT_SIZE, ..., NUM_EXPERTS>`
（`:186-200`）。task impl 实体在 `include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh`：

- 签名 `:50-55` 是纯 `__device__ __forceinline__`，**不含 `grid`/`worker` 概念**（`docs/07`/`docs/29`
  「task 被 worker 在 CTA 内调用」的同一约定）；
- `:63` 起用 CuTe 组 `SM100_MMA_F16BF16_SS`（`TiledMMA`，`:66-73`）——即 **tcgen05 UMMA**（`docs/21`
  §2–§3 走读的 2SM 链的算子级形态）；
- `:63-65` 的 `mma_coord_vmnk = make_coord(0, _, _, _)`：**peer-CTA 维固定 0、MMA-M/N/K 用通配**
  ——coordinate tensor 化的 tile 调度（`docs/21` §9 的 `TiledMMA` 语义）。

一句话：单测 wrapper 把一个 **tcgen05/TMEM 的 SM100 GEMM task** 按 1 CTA（`grid_dim(1,1,1)`）跑一遍，
只对数值/标称延迟负责——与 `docs/29` §4 对 hopper 版 `moe_linear` 的描述同构，只是硬件代际从
`mma.cuh` 升到 `SM100_MMA_*` + UMMA + TMEM。

## 5. `cute/hopper/gemm.cu`：第三类路线——CUTLASS/CuTe standalone GEMM 单测

`cute/hopper/gemm.cu`（556 行）不含任何 MPK 运行层，是一台**独立的 CUTLASS/CuTe warp-specialized
GEMM 微基准**（include `cutlass/gemm/*`、`cute/tensor.hpp`、本地 `gemm_ws.cuh`/`gemm_ws_mpk.cuh`/
`kernel_traits.cuh`/`mma_tma_ws_mainloop.cuh`，`:14-38`）。它只注册**一个**符号
`m.def("linear_mpk", &linear_mpk_kernel, ...)`（`:554-556`），而旁边的 `m.def("linear", ...)` **被注释**
（`:555`），另一个 `linear_kernel_hopper_cute_wrapper` 也被整块注释（`:51-59`）——即 `docs/29` §2
「注册表裁剪」的又一实例（只保留 MPK 变体）。

- `linear_mpk_kernel`（`:531-552`）用宏 `DISPATCH_LINEAR_CUTE_MPK_{OUTPUT_SIZE,BATCH_SIZE,
  REDUCTION_SIZE}`（`:485-530`）分派，**只实例化了 `OUTPUT_SIZE=64` / `BATCH_SIZE∈{8,16}` /
  `REDUCTION_SIZE=4096`**（`:499-524`；`12288`、`32`、`64` 的 case 被注释）。
- 每档 launch `dim3 grid(1); dim3 block(256); shared_mem_size = 100000`（`:401-404`）——**又是单 CTA**，
  但 SMEM 只要 100 KB（比 SM100 的 224 KB 小），走朴素 `<<<grid, block, shared_mem_size>>>`（`:429`），
  **不用 cluster API**（Hopper 单 CTA 版）。

即 `cute/hopper/gemm.cu` 是「同专题母题『同一套惯用法实现了两遍』（`docs/19` §5）在单测层的第三个肉身」：
`docs/07` 手写 task / `docs/19` runtime 模板库 / 本文件的 CUTLASS standalone，都是一套 warp-specialized
TMA+wgmma GEMM，只是**装配方式与调用入口不同**。

## 6. `hopper-moe` 复核：与 `docs/29` §2 一致

`hopper-moe/runtime_kernel_wrapper_moe_hopper.cu`（492 行）只注册两个符号
`moe_w13_linear_sm90` / `moe_w2_linear_sm90`（`:486-492`），与其目录下测试的调用一致——**复核了
`docs/29` §2「hopper-moe 与当前代际对齐」的判断**。它证明上文的 §2.1 撞名 / §2.2 缺源不是「SM90 才有」，
而是**当前代际的局部卫生问题**。

## 7. 判据 C1–C5 映射（与 `docs/29` §6 完全一致）

本文走读的 blackwell/cute/hopper 单测 wrapper **不贡献任何 C 判据**：每次 host 一次 launch（且常是
`(1,1,1)` 单 CTA / 单 cluster），非常驻（无 C1/C2），只是 C3 范围内「**单个 device task 数值正确**」的
回归网（无 C4/C5）。它们存在的意义是**在 device task（`docs/07`/`docs/24`）与整图端到端（`docs/28`）
之间插一层最细粒度的数值闸门**——`docs/29` §6 的那句话在补完 blackwell 家族后依然成立。

## 8. 与 `docs/29` 的差异小结（本次新增的核对结论）

1. **修正 §2 的「当前代际对齐」**：blackwell 家族有 (a) **同名扩展撞名**（`sm100_linear` 与
   `sm100_moe` 都叫 `runtime_kernel_blackwell`、符号互异）、(b) **源文件缺失**（`sm100_mla_prefill_tp8`）、
   (c) **无 setup/pybind 的 standalone 程序**（`sm100_mla_mtp_decode/`）三类失配，均比「注册表裁剪」更隐蔽。
2. **补齐「非 pybind」的第二条单测路线**：`hopper/attention_tail_wrapper.cu` 的 `extern "C" launch`
   （nvcc → `.so`）与 `cute/hopper/gemm.cu` 的 standalone CUTLASS（`linear_mpk`），
   `docs/29` 只覆盖了 pybind 那条。
3. **坐实「一个 task = 一颗 kernel」在 SM100 的形态**：`grid_dim(1,1,1)` + `block 256` + `cluster launch`
   + 224 KB SMEM + `SM100_MMA_*`/tcgen05/TMEM task impl——与 `docs/21` 的 Blackwell codegen 目标同源。

## 9. 局限

- **纯源码走读，未构建 / 未运行**：§2 的撞名 / 缺源 / standalone 判断均基于 `setup.py` 的
  `name=`/`sources=` 与 `PYBIND11_MODULE` 注册表、测试 import 语句的**静态对照**；未 `pip install`
  各扩展、未跑 `pytest`、未 `nvcc` 编 `attention_tail.so` 或 `test_tp*_wrapper.cu`。
  **不构成「这些测试当前一定失败」的运行时判定**——撞名冲突是否显现取决于各目录的 build/`sys.path`
  隔离方式，本文只记录「同一环境里两个同名模块」这一静态事实，**不判对错**。
- **失配口径**：无法从快照区分「有意裁剪」与「代际腐化」（同 `docs/29` §7）。
- **只逐目录核注册表**：未逐行走读 `sm100_linear_fp8`（1107 行）/`sm100_fp8_moe`（810 行）等大 wrapper
  的 fp8 scale 重排与 UMMA 内部，只核了其注册符号与构建宏；`test_tp*_wrapper.cu` 的内部（`shim` 之后
  的 TP 切分）未展开。
- **SM100 不可执行**：target SM100 在目标卡 H100/sm90 不能跑（与 `docs/17`/`docs/21`/`docs/29` 同口径），
  本文所有 SM100 形态均为**静态走读**。

## 10. 参考坐标

- 快照：`mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。
  `tests/runtime_python/blackwell/sm100_linear/{setup.py,runtime_kernel_wrapper_sm100.cu}`（665 行；
  `name:22/25`、wrapper `:54-56`、`launch_linear_sm100_mpk:147`、`grid/block/cluster/smem:263-266`、
  `launch_kernel_on_cluster:282/307`、`linear_sm100_mpk_kernel:321`、splitk `:364/456/572-574/630`、
  `PYBIND11_MODULE:660-665`）。
  `tests/runtime_python/blackwell/sm100_moe/{setup.py,runtime_kernel_wrapper_sm100.cu}`（710 行；
  `name:25/28`、`topk_softmax_kernel:44`、`moe_linear_sm100_wrapper:150`、task_impl 调用 `:186-200`、
  `grid:295-297`、`moe_w13_linear_sm100_kernel:463`、`PYBIND11_MODULE:697-711`）。
  `include/mirage/persistent_kernel/tasks/blackwell/moe_linear_sm100.cuh`（`moe_linear_sm100_task_impl`
  `:50-55`、`mma_coord_vmnk:63`、`SM100_MMA_F16BF16_SS TiledMMA:66-73`）。
  `tests/runtime_python/blackwell/sm100_mla_prefill_tp8/setup.py`（`sources:17` 指向缺失的
  `runtime_kernel_wrapper_mla_prefill_tp8.cu`；测试 `test_mla_prefill_tp8.py:17`）。
  `tests/runtime_python/blackwell/sm100_mla_mtp_decode/test_tp{2,4,8}_wrapper.cu`
  （`int main:145`、`thin shim:24-67`；无 setup/pybind）。
  `tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_sm100_mla.cu`（`PYBIND11_MODULE:291-314`）、
  `..._mla_decode.cu:172-175`、`..._mla_prefill.cu:118-121`、`..._mla_mtp.cu:208-211`。
  `tests/runtime_python/blackwell/sm100_moe_sigmoid/runtime_kernel_wrapper_sm100.cu:205-212`、
  `sm100_quantize_fp8/...:168-182`、`sm100_linear_fp8/...:1100-1106`、
  `sm100_paged_attention/...:162`、`sm100_fp8_moe/runtime_kernel_wrapper.cu:791-810` 与 `_bf16.cu:350-360`。
  `tests/runtime_python/cute/hopper/gemm.cu`（556 行；`include:14-38`、注释掉的 `linear:51-59/555`、
  `linear_mpk_kernel:531-552`、dispatch 宏 `:485-530`、`grid/block/smem:401-404`、`PYBIND11_MODULE:554-556`）。
  `tests/runtime_python/hopper/attention_tail_wrapper.cu`（91 行；`SMEM_BYTES:10`、`poison:20-24`、
  `run_attention:13`、`launch_attention:60-73`、`extern "C" launch:74-91`；
  调用方 `hopper/test_attention_tail_hopper.py:54-56`）。
  `tests/runtime_python/hopper-moe/runtime_kernel_wrapper_moe_hopper.cu:486-492`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量层，同 `docs/28`/`docs/29` 判断）。
- 同专题相关篇：`docs/29`（本文补其 §7 尾巴）、`docs/07`/`docs/24`（wrapper 调用的 device task）、
  `docs/21`（Blackwell codegen：UMMA/TMEM/cluster，本文 §3/§4 的算子级对照）、`docs/19` §5
  （「同一套惯用法实现了两遍」，本文 §5 的第三个肉身）、`docs/28`（testmode 端到端层，与本文 wrapper
  层相对）、`docs/22` §8（C 判据总映射，本层不贡献）。
- 交叉专题：DeepGEMM mega 家族的前向/反向验证见 `../mega-moe/`（本专题不重复）。
