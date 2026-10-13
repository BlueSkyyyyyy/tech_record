# MPK 的另一条 codegen 后端：Triton transpiler 与 Triton 路径走读

> 本文补齐 `docs/19`（P105-1）**§8 明留的尾巴**：
>
> > **未覆盖 `triton_kernels.py`**：本目录还有一个 Python 侧 `triton_kernels.py`（非 C++ 头），本文未走。
>
> 顺着这条尾巴往上看，会发现它**不是孤立的运行时小文件**，而是一整条**平行于 NVCC/CUDA transpiler 的
> 第二条 codegen 后端**：`superoptimize(backend="triton")` → `triton_profiler.py`（子进程 profile）→
> Cython `generate_triton_program` → C++ `TritonTranspiler`（`src/triton_transpiler/*`）→ 生成的
> Python/Triton 模块 → 运行期 `triton_kernels.py`。`docs/18` 走的是 `src/transpiler/*` + NVCC 这条
> （把 thread-block 图编成 CUDA `<<<>>>`），`docs/19` 走它调用的运行期模板库；本文走**完全并行的另一条**。
>
> 另外，快照里还有一条**更早的、文件级的 Triton codegen**（`src/kernel/triton_code_gen.cc` 的
> `Graph::generate_triton_program`），本文一并对照——即「Triton codegen」在仓库里同样**实现了两遍**，
> 与 `docs/19` §5「同一套硬件惯用法实现了两遍」同构。
>
> **纯源码走读（资料分析）**，不重新构建/运行 mirage。主参考仓库
> `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`–`docs/19` 同一快照，所有行号据此核对
> （路径相对仓库根）。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（「快照论文」）。

## 0. 本文在链条里的坐标

`docs/18` 已给出「**cuda** 后端」的调用链：`KNGraph.compile()` → `transpile()` →
`generate_code()` → `transpile_ugraph()` → 每个 customized op 生成一个
`__global__ void custom_kernel_N(...)`，再交 NVCC JIT。本文走的是同一层的**兄弟分支**：把 `backend`
从 `"cuda"` 换成 `"triton"`，产出的是**一个 Python 模块**（内含 `@triton.jit` 内核 + `execute_mugraph`），
由 `importlib` 加载后直接 `torch`/`triton` 跑，**不经过 NVCC**。

整条链（四个坐标）：

```
python/mirage/kernel.py        superoptimize(backend="triton")            :543,:553,:706-745
        │                      __call__ → triton_call                      :300-315
        ▼
python/mirage/triton_profiler.py  TritonProfiler（每个候选图起一个子进程 profile）  :12,:187-259
        │                      调模块级 generate_triton_program(graph.cygraph, target_cc)
        ▼
python/mirage/_cython/core.pyx    generate_triton_program(CyKNGraph,*,target_cc)->dict  :1461-1469
        │                      （另有一个遗留方法 CyKNGraph.generate_triton_program(filepath) :901-906）
        ▼
src/triton_transpiler/transpile.cc    TritonTranspiler::transpile_ugraph()        :299-565
src/triton_transpiler/transpile_tb.cc transpile_kn_custom_op()                     :209-643
include/mirage/triton_transpiler/runtime/triton_kernels.py  运行期 3 个 @triton.jit  :1-66
```

`backend` 这个旋钮在两端出现：Python 侧 `superoptimize(..., backend=...)`，C++ 侧
`search_c.cc:16` 也接了一个 `backend` 参数——但**它只出现在函数签名里，函数体从不引用**
（`grep backend src/search/*.cc` 只有这一行）。所以 `backend` 实际只在「搜完之后选哪条编译支路」
（`kernel.py:596` 的 `if backend == "cuda"` / `:706` 的 `elif backend == "triton"`）上生效，
**不影响搜索本身产出的候选图**。

## 1. 前端选择与调用：`backend="triton"` 走什么

`KNGraph.superoptimize` 的默认后端是 CUDA（`kernel.py:553  backend: str = "cuda"`）。三条支路：

- `backend == "cuda"`（`kernel.py:596` 起）：把每个候选图再扫 `pipeline_stages × num_warp_groups`
  （sm90 上 `[2,3,4]×[2,3,4]`，`:606-608`），交给 NVCC transpiler（`docs/18`）。
- `backend == "triton"`（`kernel.py:706-745`）：
  1. 设 `KERNELS_PATH` 环境变量指向 `INCLUDE_PATH/mirage/triton_transpiler/runtime`（`:711`，
     即 `triton_kernels.py` 所在目录——生成的代码靠 `sys.path.append(os.environ['KERNELS_PATH'])` 找它）；
  2. `profile_and_select_best_graph(all_graphs, target_cc=cc, ...)`（`:714`）；
  3. 拿到 `best_file_path`（一个 `.py` 临时文件），`importlib` 动态加载，把模块里的
     `execute_mugraph` 绑到 `best_graph.run`（`:733-740`）；缺 `execute_mugraph` 直接抛错；
  4. `best_graph._cached_results = {"output_shapes": ...}`、`best_graph.backend = "triton"`。
- `KNGraph.__call__` 按 `self.backend` 分派（`:300-303`）：`"triton"` → `triton_call`（`:305-...`），
  它按 `output_shapes` 预建 `torch.zeros` 输出张量再调 `self.run`。

最小可运行示例就在仓库里：`demo/triton_rms_norm.py`——`superoptimize(config="mlp",
backend="triton", warmup_iters=2, profile_iters=6)`（`:17`），然后
`mi.generate_triton_program(optimized_graph.cygraph, target_cc=10)["code"]` 把生成的 Python 写盘
（`:20`），再 `optimized_graph(inputs=input_tensors)` 直接跑。注意它内部**又调了一次**
`superoptimize` 之外的 `generate_triton_program`（模块级、`target_cc`），只是为**导出代码文本**，
真正跑的是 `superoptimize` 里已选中的那份。

### 1.1 profile 是「子进程」做的：`triton_profiler.py`

`TritonProfiler`（`triton_profiler.py:12`）与 CUDA 路径最大的结构差别：**每个候选图在独立 Python
子进程里计时**，而不是在宿主进程内用 CUDA event。三步：

1. `_generate_profile_code_file`（`:28-87`）：把 C++ 生成的整段代码按 `@triton.jit` / 
   `if __name__ == "__main__":` 切成 header/kernel/main 三段，把 main 重新缩进后拼进一个
   `profile_kernel()`，里面 `warmup_iters` 次热身 + `profile_iters` 次 CUDA event 计时并打印
   `Average time per iteration: X ms`；`save_codes` 时另存 `generated_codes/generated_code_N.py`。
2. `_run_profile`（`:160-185`）：`subprocess.run([sys.executable, module_path], capture_output=True)`，
   从 stdout 里 grep 那行 `Average time per iteration` 解析出 `ms`；子进程非零退出记 `fail_num` 返回 `-1`。
3. `profile_graphs`（`:187-243`）：遍历所有候选图，取 `perf>0` 且最小者为 best，返回
   `(best_graph, best_file_path, best_output_shapes)`。`_generate_debug_code_file`（`:89-158`）是
   `debug=True` 的变体：包一层 try/except + `traceback.print_exc`，便于定位某个候选图编译失败。

子进程计时意味着每次测量都带一次解释器/CUDA 上下文冷启动，但好处是**候选图互不污染**
（triton JIT 缓存、显存碎片都在独立进程里）。这与 CUDA 路径在宿主进程内用 `torch`/graph 计时不同。

## 2. C++ 侧 IR 重建：`TritonTranspiler` 构造函数

`TritonTranspiler`（`transpile.h:58-85`）不直接改原图，而是**克隆出一个新 `kernel::Graph`**
（`transpile.cc:41-298` 的 ctor，`g = std::make_shared<kernel::Graph>()` `:46`），逐 op 用
`dtensor_mapping`（guid→新张量）重建：

- `KN_INPUT_OP`（`:57-70`）：`g->new_input(dims, input_strides, data_type, layout)`。
- `KN_OUTPUT_OP`（`:72-84`）：`g->mark_output` 并把对应 dtensor 收进 `mugraph_output_tensors`
  （`:80-82`，后面用来产 `output_shapes`）。
- `KN_MATMUL_OP`（`:86-92`）：`g->matmul`。
- 逐元素 unary/binary（`:93-115`）：`KN_EXP/SQUARE/SQRT/SILU/RELU/CLAMP` → `g->elementunary`；
  `KN_ADD/MUL/DIV/POW` → `g->elementbinary`。
- `KN_REDUCTION_0/1/2_OP`、`KN_RMS_NORM_OP`（`:116-125`）：**`assert(false && "To be implemented")`**。
- `KN_CUSTOMIZED_OP`（`:127-273`）：新建一个 `threadblock::Graph`，把 `customized_op->bgraph` 的
  TB op 逐个用 `stensor_mapping` 重建——`TB_INPUT/OUTPUT`（`:146-167`，带 `input_map`/`output_map`/
  `forloop_dim`/`epilogue`）、`TB_MATMUL_OP`（`:168-173`）、TB 逐元素（`:174-206`，含 `TB_MUL_SCALAR_OP`
  的 `scalar`）、`TB_FORLOOP_ACCUM_*`（`:207-250`）。注意这里把**复合 op 手工拆开**：
  `TB_FORLOOP_ACCUM_RED_LD_SUM_OP` 拆成 `forloop_accum + reduction`（`:215-223`）、
  `TB_FORLOOP_ACCUM_RED_LD_RMS_OP` 拆成 `square→mul_scalar(1/…)→forloop_accum→reduction→sqrt`
  （`:228-241`，`normalization_factor = dim * forloop_range`）、`TB_RMS_NORM_OP` 拆成
  `square→mul_scalar(1/dim)→sqrt→div`（`:251-261`，注意 `:256` 的 `reduction` 被注释掉，
  说明它偷懒没做跨维归约）；最后 `g->customized(dtensor_inputs, *tbg)`（`:267`）。

ctor 末尾（`:280-297`）做一次**后置校验**：遍历所有 customized op，要求 (1) `TB_FORLOOP_ACCUM_*`
只允许 `TB_FORLOOP_ACCUM_NO_RED_OP`（其余归约形态必须已被前一步拆掉）、(2) 不允许残留
`TB_RMS_NORM_OP`。所以「KN 级 reduction/rms_norm 直接 `assert(false)`」和「uGraph 代码生成里却还有
reduction 分支」（见 §3.4）之间的口径差异，靠这层校验和上游 search 的分解来弥合——**读这段时不能
只看单个 switch**。

## 3. 生成 Python 模块：`transpile_ugraph()`

`transpile_ugraph`（`transpile.cc:299-565`）把重建好的图编成**一个 Python 模块字符串**，四段拼接
（`:553-558`）：header + custom_kernels + entrance_func + exec。

### 3.1 header 与张量初始化

- header（`:301-309`）：`import triton / triton.language as tl / triton.ops as ops / torch`，
  `sys.path.append(os.environ['KERNELS_PATH'])`（`:308`），`from triton_kernels import *`（`:309`）。
- 遍历算子建张量：输入用 `torch.randn(shape, dtype=torch.float16)`（`:332`）、中间/输出用
  `torch.zeros(...)`（`:346`）——**dtype 硬编码 float16**，形状直接读 `dtensor.dim[]`。输入/输出张量
  名分开收集，并从中间张量列表里剔除（`:354-362`，防止把输出又当中间量分配）。
- `execute_mugraph(inputs, outputs)`（`:378-379`）：函数体内再为中间张量分配 `torch.zeros`（`:382-387`），
  然后逐 op emit 调用。**输出张量由调用方（`triton_call`）预建并传入**，函数只写不建。

### 3.2 非 customized 算子：调用运行时内核或 `ops.matmul`

对图里的每个 op，`transpile_ugraph` 生成一行调用（同时写进 `exec` 顶层自测和 `entrance_func` 函数体）：

- `KN_MATMUL_OP`（`:426-450`）：`ops.matmul(a, b)`（triton.ops）；若该输出同时是图的输出，
  改用 `output.copy_(ops.matmul(...))` 把结果拷回预建的输出张量（`:437-448`）。
- `KN_ADD/MUL/DIV/POW_OP`（`:452-476`）：`elementwise_binary_kernel[grid](x, y, out, M, N, OP)`，
  grid 形状 `(ceil(dim0/1), ceil(dim1/128))`（`BLOCK_SIZE_X=128`、`BLOCK_SIZE_Y=1`，`transpile.cc:21-22`），
  `OP` 直接传 `op->op_type`（KN 枚举值）。
- `KN_EXP/SQRT/SQUARE_OP`（`:478-498`）：`elementwise_unary_kernel[grid](x, out, M, N, OP)`。
- `KN_RELU_OP/KN_CLAMP_OP`（`:500-507`）：`assert(false && "To be implemented")`（注释说 relu 是 clamp
  的特例，未做）。
- `KN_REDUCTION_0/1/2_OP`（`:509-529`）：`reduce_sum_kernel[grid](x, out, M, N, dim)`。
- `KN_CUSTOMIZED_OP`（`:397-424`）：调 `transpile_kn_custom_op(cur_op)` 拿到
  `@triton.jit` 内核定义（见 §4），把内核源码塞进 `custom_kernels`，再 emit 一行
  `custom_kernel_N[grid](dtensors...)`，grid 取 `bgraph.grid_dim.xyz`（`:415-420`）。

### 3.3 输出形状

末尾按 `mugraph_output_tensors` 把每个输出 dtensor 的 `dim[]` 收成 `vector<vector<int>>`
（`:559-563`），随 `TritonTranspileResult` 返回（`transpile.h:47-51`），经 Cython 变成 Python 字典的
`"output_shapes"`——这正是 `triton_call` 预建输出张量用的。`generate_code()`（`:567-570`）只是
`return transpile_ugraph()`，模块级 `transpile(...)`（`:572-576`）构造后调它。

## 4. 单个 customized op 的 Triton 内核：`transpile_kn_custom_op`

`transpile_tb.cc:209-643` 生成一个 `@triton.jit def custom_kernel_N(dtensor_out…: tl.tensor,
dtensor_in…: tl.tensor)`（`:217-229`）。结构与 `docs/07`/`docs/19` 的 CUDA 版**不是**同一套
（没有 TMA/mbarrier/wgmma/warp specialization），而是**纯 tl 表达式的直译**。关键机制：

- **形状对齐到 2 的幂**：`round_up_to_power_of_2`（`:24-35`，下界 16）+ `adjust_tensor_dims`
  （`:61-67`）；用 `generate_mask_expr`（`:95-144`）对**被 padding 的维**生成
  `tl.arange(0, pad)[:, None] < dim` 掩码（`:107-119`），2 维以上用 `&` 串起来。
- **warp specialization 缺席，全部单程序**：`tl.arange` + `tl.program_id(0/1/2)` 直接算指针
  （`get_input_dim_expr` `:147-179`、`get_output_dim_expr` `:181-207`），`imap/omap` 哪一维映射到哪个
  `program_id` 就乘哪个 block 维。
- **累加器**：对每个 `TB_FORLOOP_ACCUM_NO_RED_OP` 输出预置
  `stensor$ = tl.zeros((padded_shape), dtype=tl.float32)`（`:232-249`）——**累加恒 fp32**，注释里附
  原始形状（`:247`）。
- **两遍遍历**：算子被 `input.after_accum` 分成两组——第一遍（`:257-446`）处理累积前的
  load/matmul/逐元素/累加，第二遍（`:452-640`）处理累积后的逐元素/归约/store。`forloop_range>1`
  时第一遍外面套 `for i in range(forloop_range):`（`:251-254`），累加在循环内（`+=` `:336-339`），
  循环在第二遍前 `dec_indent`（`:447-449`）。
- **GEMM 恒走 `tl.dot`**：`tl.dot(input0.to(tl.float32), input1.to(tl.float32))`（`:326-329`、
  `:629-632`），**无 `out_dtype`**（对比旧路径 §5.3 显式 `out_dtype=tl.float16`）。注意 matmul 分支算了
  `mask0/mask1`（`:321-324`）：**算了但没用**（只 `code.e` 出 `tl.dot`），与 `docs/16` 里
  「已实现未接线」同型的痕迹。
- **逐元素/归约**：`operator_type_to_triton`（`:69-93`）把枚举映射到 `tl.exp/tl.sqrt/`运算符
  （`TB_SILU_OP` 映射到 `tl.sigmoid`——是 silu 的近似/错误？见局限）；`TB_SQUARE_OP` 特意展开成
  `x*x`（`:357-364`）；归约 `tl.sum(x, axis=dim, keep_dims=True)`，带掩码时先乘掩码再 sum
  （`:417-441`）。
- **store/load**：`TB_INPUT_OP` → 带掩码的 `tl.load`（`:268-307`）；`TB_OUTPUT_OP` 只允许在累积后，
  `tl.store`（`:469-514`）。`TB_OUTPUT` 出现在累积前直接 `assert(false)`（`:309-312`）。

与 CUDA 版的另一处结构性差别：**TB 图里不再有 SMEM 定址**——`stensor$` 只是 Python 局部变量
（SSA 名的来源是 `stensor.guid`），没有 `smem_offset`、没有显式 barrier，Triton 编译器负责寄存器/
共享内存分配与同步。

## 5. 运行期库 `triton_kernels.py`（补齐 `docs/19` §8 的尾巴）

`include/mirage/triton_transpiler/runtime/triton_kernels.py`（66 行；另有
`include/mirage/transpiler/runtime/triton_kernels.py` 与之**逐字节相同**、只差一个行尾换行，
两份是同一文件的复制）。它只有三个 `@triton.jit`：

- `elementwise_binary_kernel(X_ptr, Y_ptr, Out_ptr, M, N, OP)`（`:4-26`）：`row_idx =
  tl.program_id(0)`、`col_idx = tl.program_id(1)*128 + tl.arange(0,128)`（`:7`，**128 硬编码 + TODO**）；
  `OP==1200→x+y`、`1201→x*y`、`1202→x/y`，否则 `tl.device_print("Unsupported OP code ...")`。
- `elementwise_unary_kernel(X_ptr, Out_ptr, M, N, OP)`（`:28-48`）：`OP==1100→tl.exp`、
  `1102→tl.sqrt`、`1101→x*x`，否则 device_print。
- `reduce_sum_kernel(X_ptr, Out_ptr, M, N, dim)`（`:50-66`）：`dim==1` 时对 M 行循环累加到 `[N]`
  再只在 `row_idx==0` 写；`dim==0` 时 `tl.sum` 后仅 `col_idx[0]==0` 写——**每个输出元素只由单个
  program 写**，没有跨 block 归约（grid 的第 0 维被当输出行）。

**关键耦合**：这些 `OP` 魔数不是随便定的——它们**就是 `include/mirage/type.h` 里 C++ 枚举的数值**：
`KN_EXP_OP=1100 / KN_SQUARE_OP=1101 / KN_SQRT_OP=1102`（`type.h:86-88`）、
`KN_ADD_OP=1200 / KN_MUL_OP=1201 / KN_DIV_OP=1202 / KN_POW_OP=1203`（`:98-101`），而 `transpile.cc:472`
正是把 `op->op_type` 原样传进去。于是：**(a)** Python 运行期的 `if OP==…` 与 C++ 枚举值形成**隐式
ABI**，任一改动都会静默错配；(b) `KN_POW_OP=1203` 与 `KN_SILU_OP=1104` **没有对应分支**，会落到
`device_print("Unsupported OP code")`——尽管 `transpile.cc` 的 KN_POW 分支会照常 emit 调用（POW 因此
是「能生成、但运行期不干活」）；同理 `transpile_ugraph` 不 emit SILU。

## 6. 旧路径对照：「Triton codegen」也实现了两遍

快照里还有一条**更早的、文件级**的 Triton codegen：`src/kernel/triton_code_gen.cc`
（整体包在 `#ifdef MIRAGE_BACKEND_USE_CUDA` 里，`:16`）。它和 §3/§4 那条**结构完全不同**：

| 维度 | 旧路径 `Graph::generate_triton_program`（`triton_code_gen.cc`） | 新路径 `TritonTranspiler`（`triton_transpiler/*`） |
|---|---|---|
| 入口 | `CyKNGraph.generate_triton_program(filepath)`（`core.pyx:901-906`）写**文件** | `generate_triton_program(graph, target_cc)`（`core.pyx:1461-1469`）返回**字典** |
| Python 用处 | **无调用者**（`grep .generate_triton_program(` 只剩定义与生成的 `core.cpp`） | `triton_profiler.py` + `demo/triton_rms_norm.py` |
| 描述符来源 | `bgraph.get_new_kernel_params(false)` 的**序列化 `NewKernelParams`**（`:455-456`），逐字段 `deserialize_*` | 直接读 `bgraph.operators` / `STensor` |
| 访存 | `tl.make_block_ptr` + `tl.advance`（`:146-182`、`:210-233`，含 `strides`/`order`） | `tl.arange` + `program_id` 手算指针（`transpile_tb.cc`） |
| GEMM | `tl.dot(a, b, out_dtype=tl.float16)`（`:254-256`） | `tl.dot(a.to(fp32), b.to(fp32))`（无 out_dtype） |
| 归约 | `tl.reshape` 到 3D + `tl.sum(axis=…)`，依赖 `reduction_dimx`（`:306-385`） | `tl.sum(x, axis=dim, keep_dims=True)`（`:417-441`） |
| concat | 支持 `tl.cat`（`:386-401`） | 新路径无 concat 分支 |
| 自测模板 | 生成 `kernel_launcher` + `triton.testing.do_bench(warmup=1000,rep=1000)`（`:489-499`） | 由 `triton_profiler.py` 在外层包 warmup/profile |

即：**旧的 kernel-级 Triton codegen 依赖「序列化描述符」、走 block_ptr；新的 transpiler 级直接读图、
走 arange**，两者对 GEMM/归约/访存各有各的写法，且旧路径已成**事实死代码**（无 Python 调用者）。
这与 `docs/19` §5「同一套硬件惯用法在仓库里实现了两遍」是同一现象的 Triton 版本——只不过这里
是**软件（codegen）实现了两遍**，且新路径显然是超优化器 `backend="triton"` 的产物。

另外，`TritonTranspilerConfig` 只有一个字段 `target_cc`（`transpile.h:38-41`），
`core.pyx:1462-1463` 也把它赋了值，但 **`grep target_cc src/triton_transpiler` 只在头文件里出现**
（`transpile.cc` 从不读 `config`）——即 `target_cc` 是**已接线但未使用**的旋钮，与 `docs/16` 记的
`range_propagation` 等「已实现未启用」同型。

## 7. 与论文、生产路径的关系与 C1–C5 判据

- **论文对照**：论文 §4.2（快照行 1126-1146）讲的是「以 thread block 为单位复用 Mirage
  superoptimizer 生成 **CUDA**」；本文这条是**同一台 superoptimizer 的另一个 codegen 目标**
  （Triton 而非 CUDA），论文正文并未把 Triton 后端作为 MPK 的组成部分描述——它是 mirage 上游实验
  分支的能力，`mpk` 分支保留但**生产前端不经它**（见下）。
- **与 MPK 生产路径的关系**：`docs/15` §7 已证 `mpk` 生产前端（`models/`、`mpk/`）**不调
  `superoptimize`**，只用 `mpk` 的手工 task API + CUDA transpiler。所以 Triton 后端与
  `docs/15`/`docs/16`/`docs/18`/`docs/19` 一样，属于「**自动/实验路径**」这一侧，**不参与 `mpk`
  常驻 mega-kernel 的构建**。它的价值是理解 Mirage 自动路径的**codegen 可替换性**。
- **C1–C5（本专题定义的五判据）**：
  - C1 单次 launch：**不满足**。Triton 模块是「每个 customized op / 每个 KN 算子一次 kernel launch」
    的 **kernel-per-operator**，没有持久化 CTA、没有设备内 event 驱动。
  - C2 常驻：不满足（每次调用就是普通 `[grid](...)` 启动）。
  - C3 多算子范围：**部分**——一个 customized op 内的多个 TB 算子会被编进**同一个
    `custom_kernel_N`**（图内同 kernel），这是 docs 一贯的「段级」粒度；但段与段之间仍是逐 launch。
  - C4 设备内协调：不满足（同步交给 Triton 编译器/流序，无显式设备内 event）。
  - C5 SM 级去中心化调度：不满足。
  - 结论：Triton 后端**只贡献 C3**，与 `docs/07`/`docs/18`/`docs/19` 同档——再次印证
    「**megakernel 的 C1/C2/C4/C5 不是靠『换 codegen 目标语言』能拿到的**，必须改执行模型」。
- **工程观察**：本文把「同仓库里平行存在两套 codegen（CUDA/NVCC vs Triton）、且 Triton 自身又存在
  新旧两套实现」摆出来，给 `docs/15` §7「自动路径 vs 生产路径的工程落差」补了第三个坐标
  （语言维度）。生产 mega-kernel 选择手写 CUDA + 手工 SM 角色，一部分原因正在于此：**要控制
  persistent/scheduler/事件协议，就必须落到 CUDA 层**，而 Triton 路径（不论新旧）都停在
  「一批独立 kernel + 编译器管的同步」这一表达层级。

## 8. 局限

- **纯源码走读，未构建/运行**：不采信任何性能数（`docs/15`/`docs/18`/`docs/19` 同口径）；所有结论
  基于 `f9eb70c` 快照行号。未在容器里跑 `demo/triton_rms_norm.py`、未验证 Triton 生成代码能编译。
- **`backend="triton"` 与 `search()` 的关系未实证**：本文只据源码判断 `search_c.cc` 的 `backend`
  参数未使用，未追进 `search()` 里被 `backend` 影响的下游（若有）。
- **`target_cc` 未被使用**：结论来自 `grep`，未排除它经宏/模板间接生效。
- **旧路径是否真死**：结论「`Graph::generate_triton_program` 无 Python 调用者」来自 `grep`；未排查
  CTest/benchmark 的 C++ 直接调用。
- **未逐行验证语义正确性**：§4 的 `TB_SILU_OP→tl.sigmoid`、`TB_RMS_NORM_OP` 里被注释掉的
  `reduction`（`transpile.cc:256`）、`reduce_sum_kernel` 的单 program 写、`elementwise_*` 的 128
  硬编码等，都是「读出来的实现形态」，未与数值参考对拍，不能据此判定上游正确或错误。
- **未展开 Triton 编译器内部**：`tl.dot`/`tl.load` 如何 lower 到 MMA/TMA 不在本文范围。

## 参考坐标

- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462-468、
  §4.2 行 1126-1146）。
- 源码快照：`github.com/mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。相关文件与行数：
  - 前端/驱动：`python/mirage/kernel.py`（`superoptimize` `:543`、backend 分支 `:596`/`:706-745`、
    `__call__`/`triton_call` `:300-315`）、`python/mirage/triton_profiler.py`（259 行）、
    `python/mirage/_cython/core.pyx`（`generate_triton_program` `:901-906` 与 `:1461-1469`）、
    `demo/triton_rms_norm.py`。
  - C++ 转译器：`include/mirage/triton_transpiler/transpile.h`（87 行）；
    `src/triton_transpiler/transpile.cc`（579 行）、`src/triton_transpiler/transpile_tb.cc`（645 行）；
    旧路径 `src/kernel/triton_code_gen.cc`（513 行）、声明 `include/mirage/kernel/graph.h:195`。
  - 运行期：`include/mirage/triton_transpiler/runtime/triton_kernels.py`（66 行；与
    `include/mirage/transpiler/runtime/triton_kernels.py` 逐字节相同）。
  - 枚举耦合：`include/mirage/type.h`（KN `:82-101`、TB `:174-276`）。
- 同专题相关篇：`docs/15`（superoptimizer，本文的上游「自动路径」）、`docs/16`（搜索器辅助，含
  「已实现未启用」证据）、`docs/18`（CUDA transpiler codegen，本文的兄弟分支）、
  `docs/19`（runtime 模板库；本文补其 §8 的 `triton_kernels.py` 尾巴）。
