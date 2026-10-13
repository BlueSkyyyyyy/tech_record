# MPK 的自动路径：Mirage superoptimizer 如何在 thread-block 级搜索并生成 task

> 本文是 MPK 源码走读链条的**另一条上游支线**。`docs/14` 走通了 `mpk` 分支的**生产前端**——一套
> 「手工 task API」（每层手写 bGraph + `register_task`），并在 §7 明留了一条尾巴：
>
> > **未覆盖自动（superoptimizer）路径**：本文只说明 `mpk` 前端不走它；Mirage 主分支的超优化器
> > 如何从高层算子搜出 threadblock 实现（`src/search/`、`src/transpiler/`）未在本文范围。
>
> 论文 §4.1/§4.2 描述的算子分解与 task 实现生成是**自动**的（「MPK decompiles each operator into a set
> of tasks by partitioning the operator's output tensors」「MPK performs superoptimization at the thread
> block level ... uses the Mirage superoptimizer [40]」）。本文把这条**论文里的自动路径**在源码里走通，
> 并把「论文自动 vs `mpk` 生产前端手工」的落差钉到具体函数上。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`/`docs/07`/`docs/08`/`docs/14` 同一快照，
> 所有行号都据此核对。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（以下简称「快照论文」）。
>
> 本文为**纯源码走读（资料分析）**：不重新构建/运行 mirage。自动搜索需要在 GPU 上反复 compile+profile
> 选优（下一步见 §5、§8），本机不具备该跑批条件。

## 0. 自动路径在两条链条里的位置

`mpk` 分支里其实有**两条**「高层算子 → 可 launch 的 kernel」的路：

| | 生产前端（`docs/14`） | 自动超优化（本文） |
|---|---|---|
| 入口 | `models/*/builder.py` 手工调 `*_layer()` | `KNGraph.superoptimize()`（`kernel.py:543`） |
| 分解 | 人把算子切成 task，写死 bGraph | 搜索器把算子输出**分区**成 task（论文 §4.1） |
| task 实现 | `task_register.cc` 映射到**手写模板**（`docs/07` §0） | 搜索 thread-block 图 → transpiler 生成 CUDA（论文 §4.2） |
| 适用 | 95 个 TaskType，生产 LLM serving | tests/demo/benchmark（`qwen2.5`、`demo_llama3-8b` 等） |
| 论文口径 | 论文 §4.1 说「自动分解」——与生产前端有落差 | 与论文 §4.1/§4.2 一致 |

两条路共用**下游同一套**依赖分析（`docs/08`）+ 运行层（`docs/06`）：自动路径产出的是同样的
`KNCustomizedOp` + bGraph，故 `annotated_graph.cc` 之后完全接得上。**本文只走自动路径。**

## 1. 入口链路：一次 `superoptimize` 的五段

从 Python API 到设备 kernel，自动路径是一条清晰的主线调用链：

| 段 | 位置 | 做什么 |
|---|---|---|
| ① 缓存查询 | `python/mirage/kernel.py:559-571`（`graph_dataset.find`） | 同一输入图 + 同一组 `imaps/omaps/griddims/...` 命中则直接返回 |
| ② 搜索 | `kernel.py:580` `cygraphs = search(...)` → `_cython/core.pyx:1314` `search()` → `search_c.cc:21` `cython_search` | C++ 超优化器搜出若干等价图形 |
| ③ 生成器 | `search_c.cc:41-89` 建 `GeneratorConfig` → `search::KernelGraphGenerator gen(...)`（`search.cc:20`）`gen.generate_kernel_graphs()` | DFS 展开搜索空间，验证，产 `json` 图 |
| ④ 反序列化 | `search_c.cc:89-95` `from_json` → `KNGraph(g)`（`kernel.py:590` `all_graphs = [KNGraph(g) for g in cygraphs]`） | 每个候选图变回 Python 对象 |
| ⑤ 编译 + profile 选优 | `kernel.py:598-692` | 每个候选图 `compile()`（→ §5 的 transpiler），扫 `pipeline_stages × num_warp_groups`，profiling 选最快 |

要点：**「搜索」与「生成」是解耦的两步**。搜索器（§2–§4）只负责在**图空间**里找功能等价的算子图；
真正的 CUDA 代码是到了 ⑤ 才由 transpiler（§5）按**具体硬件配置**（`target_cc`、流水级数、warp group 数）
生成并实测的。论文 §4 把这两步叫「tGraph generation（§4.1）」与「task implementation generation（§4.2）」——
源码里它们分别对应 `src/search/` 与 `src/transpiler/`。

`cython_search`（`search_c.cc:21`）的签名用一组 `std::vector<MInt3>/MDim3/int` 参数把 Python 侧的
「要探索哪些并行度」透传进来：`imap_to_explore`/`omap_to_explore`/`grid_dim_to_explore`/
`block_dim_to_explore`/`fmap_to_explore`/`frange_to_explore`（`include/mirage/search/search_c.h:20-31`）。
它们非空就覆盖默认候选集（`search_c.cc:47-79`）——这就是「用户可指定自定义分区策略」的入口
（论文 §4.1 末句）。

## 2. 搜索空间：`GeneratorConfig` 与候选生成

搜索器的一切边界由 `GeneratorConfig`（`include/mirage/search/config.h:23-56`）给定，默认值在
`config.cc:6-66` `get_default_config()`：

| 字段（默认值） | 含义 | 论文对应 |
|---|---|---|
| `max_num_threadblock_graph_op = 9` | 一张 bGraph（一个 task 组）里最多 9 个 TB 算子 | task 内的指令图规模 |
| `max_num_kernel_graph_op = 5` | 整张 KN 图最多 5 个算子 | 融合范围 |
| `max_num_threadblock_graphs = 1` | 最多 1 个 `KNCustomizedOp`（=1 张 bGraph） | 默认只融成**一个** thread-block 图（attention 优化会抬到 2） |
| `max_num_threadblock_graph_inputs/outputs = 3/2` | 一张 bGraph 的输入/输出张量数上限 | bGraph 的接口宽度 |
| `knop_to_explore` | KN 级算子候选：`MATMUL/EXP/SQUARE/SQRT/SILU/GELU/RELU/CLAMP/ADD/MUL/DIV/POW/CUSTOMIZED` | 算子分解可用的算子集 |
| `tbop_to_explore` | TB 级算子候选：`MATMUL` + 逐元素 + `RMS_NORM` + 三种 `FORLOOP_ACCUM_*`（归约） | thread-block 图可用的指令 |
| `frange_to_explore = {4, 16, 64}` | for-loop 行程候选 | tile 内循环次数 |
| `reduction_dimx = 64` | 归约 tile 的 x 维 | — |

分区策略（论文 §4.1「按输出张量的维度分区」）落在 `DimStrategy`（`dim_strategy.h`，实现在
`dim_strategy.cc`）里，四个候选生成函数对应分区的四个自由度：

- **grid dim**（= task 数）：`get_grid_dim_cand(tensors)`（`dim_strategy.cc:30-148`）。它枚举
  `x = 8,16,...,256` 使 `dim % x == 0` 的 1D 网格、以及由 `{1024,2048,4096}` 组合的 2D 网格，
  再**过滤到 `32 ≤ num_threadblocks ≤ config::MAX_NUM_THREADBLOCKS_PER_KERNEL`**
  （`config.h:31`，`=4096`）。这正是论文 §4.1「**默认生成与 SM 数成正比的 task 数**以促进负载均衡」
  的机制——用「最小 32 / 上限 4096」卡出与目标卡 SM 数同量级的网格。
- **block dim**（tile 内线程布局）：`get_block_dim_cand(tensors, grid_dim)`（`dim_strategy.cc:149-244`）。
- **input map**（把输入张量切到 grid×block 上）：`get_input_map_cand(...)`（`dim_strategy.cc:245-277`），
  每输入一个 `int3`（三轴映射）。
- **output map**（输出如何切分拼回）：`get_output_map_cand(...)`（`dim_strategy.cc:278-348`）。
- **for-loop dim/range**（tile 内维度的循环）：`get_forloop_dim_cand`/`get_forloop_range_cand`
  （`dim_strategy.cc:349-438`）。

论文 §4.1 说「选择**最小化 device→SMEM 数据加载**的分区策略」——源码里这条不显式出现在候选生成
（候选是全枚举），而是**隐含在 ⑤ 的 profile 选优**里：搜索器把有希望的图都交出去，由真实硬件计时决定
谁最快（`kernel.py:686-689`）。这是「搜索器枚举 + 运行期实测」两段式，而非在搜索器内跑代价模型。

## 3. 搜索算法：`generate_next_operator` 的三层 case

真正干活的是 `KernelGraphGenerator`（`search.h:23-98`，实现 `search.cc`）。入口
`generate_kernel_graphs()`（`search.cc:394-445`）建一张空 KN 图、开 OpenMP 并行（`num_thread`，
`search.cc:28-36` 取 `hardware_concurrency()`），调 `generate_next_operator(...)` 做 DFS。

`generate_next_operator`（`search.cc:129-392`）是核心递归，按 `SearchContext` 的层级
（`search_context.h:14-18`：`LV_KERNEL` / `LV_THREADBLOCK`）分两段：

**内核级（`LV_KERNEL`，`search.cc:166-281`）** 三类 case：
- **K1**（`search.cc:168-170`）：算子数到 `max_num_kernel_graph_op` 就停，回到外层验证；
- **K2**（`search.cc:172-205`）：加一个**预定义** KN 算子（逐元素/matmul），对每个合法输入组合
  `create_op` 后递归、回溯时 `pop_back`；
- **K3**（`search.cc:206-280`）：加一个 `KN_CUSTOMIZED_OP`——即**开一张新的 thread-block 图**。
  这里就是分区的发生点：对每个 `grid_dim × block_dim × input_map × forloop_dim × forloop_range`
  组合，建 `threadblock::Graph`、`create_input_op` 把输入切进 SMEM，然后把层级降到 `LV_THREADBLOCK`
  继续搜 tile 内部的算子图。

**thread-block 级（`LV_THREADBLOCK`，`search.cc:282-392`）** 两类 case：
- **B1**（`search.cc:320-349`）：tile 图达到「有输出」状态时，为刚算出的 STensor 加
  `TB_OUTPUT_OP`（写回 DTensor），然后造 `KNCustomizedOp` 挂回 KN 图、层级升回 `LV_KERNEL` 递归
  ——完成「一层 tile 图」的封装；
- **B2**（`search.cc:357-391`）：在 tile 图里继续加预定义 TB 算子（matmul、逐元素、归约），递归展开。

三个横切机制贯穿 DFS：
- **剪枝一：abstract_expr**（`search.cc:154-163` + `check_abstract_expr` `search.cc:504-519`）。
  每个张量带一个代数表达式，新算子先算 `get_abstract_expr`，只有表达式能
  `subexpr_to_final_expr`（是某个最终输出的子式）才继续——**避免搜出与输入图功能不等价的图**。
- **剪枝二：range 约束**（`init_ranges`/`target_ranges`，`preprocess` `search.cc:482-483`；
  `check_range` 在 `search.cc:338`）。`get_init_ranges` 从输入图取每个张量的取值范围，
  `get_interact_ranges` 推出目标范围，只在 range 自洽时才递归。
- **剪枝三：order 约束**（`check_order` `search.cc:82-102`）。用算子的「输入索引序 + 算子类型」
  构成一个 `Order`，只允许**严格递增**，从而消除同构图的重复枚举——对应论文 §4.1 依赖分析里
  「线性化以紧凑编码」的思想（`docs/08` §5）。

并行策略也在 `generate_next_operator` 顶部（`search.cc:140-152`）：深度 ≤ `multithread_threshold_depth`
（构造时 `=5`，`search.cc:27`）的分支用 `#pragma omp task` 复制 `SearchContext` 后并行展开，
更深的分支串行——浅层扇出大、深层状态小，这个阈值是吞吐与内存的折中。

## 4. 验证：概率指纹 v.s. 形式证明

每个候选图在 `LV_KERNEL` 完成时交给 `verify`（`search.cc:521-562`，实际由 `generate_kernel_graphs`
的 lambda `search.cc:419-424` 触发）。验证分两种（`preprocess` `search.cc:546-551` 按
`config.verifier_type` 选）：

- **`ProbabilisticVerifier`**（默认，`config.cc:14`）：对输入图每个张量跑 `op->fingerprint()`
  （`probabilistic_verifier.cc:8-19`），再用**同一套指纹**验证候选图的输出张量
  （`probabilistic_verifier.cc:21-60`）。指纹是一组固定随机输入下的**低精度数值摘要**——若候选与
  参考图在全部随机点上的指纹**逐位相同**则判等价。快、但是概率性的（理论上可能漏判）。
- **`FormalVerifier`**（`is_formal_verified=True`）：做形式化等价证明，慢但确定。

`verify` 命中后 `OutputMatch`（`output_match.h:10-27`）还要解决「输出顺序」——候选图的输出张量可能
是参考输出集的一个排列，`mark_outputs`（`search.cc:526-532`）按 match 把输出标记对齐，再
`save_graph`（`search.cc:540-547`）把 `json(g)` 收集进 `generated_graphs`，最后由
`save_results()`（`search.cc:564-567`）落盘成 checkpoint。**注意验证只查「功能等价」，不查性能**——
性能留给 ⑤ 的实测。

## 5. 生成：从等价图到 CUDA

`generated_graphs` 被 `search_c.cc:89-95` 逐个 `from_json` 成 `KNGraph`，Python 侧
（`kernel.py:590`）拿到 `all_graphs` 后进入**编译+选优**（`kernel.py:596-692`）：

1. **展开硬件配置**（`kernel.py:606-635`，仅 `target_cc ≥ 90`）：对每个候选图，扫
   `pipeline_stages ∈ {2,3,4}` × `num_warp_groups ∈ {2,3,4}`，各 `compile(async_=True, ...)`
   一次——即在 Hopper 上对**流水深度**与 **warp 组数**两个编译期旋钮做网格搜索（12 个变体/图）。
   低端卡（<90）只编译默认配置（`kernel.py:637-657`）。`compile` 会检查
   `max_smem_size ≤ get_shared_memory_capacity(target_cc)`（`kernel.py:440-443`）——不满足该配置直接
   失败。
2. **profile 选优**（`kernel.py:660-692`）：每个成功编译的图先 warmup 再 `profile_iters` 次计时
   （CUDA event），取 `perf < best_perf` 的图返回（`kernel.py:689`）。

CUDA 代码由 `generate_cuda_program`（`core.pyx:1398-1444`）落到 C++ `transpile()`
（`transpile.cc:792-800`）。`Transpiler::generate_code()`（`transpiler.h:127-134`）是一串固定的
resolve 阶段：

```
resolve_distributed_config()   // num_gpus>1 则用 NVSHMEM（多卡）
resolve_dtensor_meta()         // 张量元数据
resolve_tb_fusion()            // 沿 fusion_chain 把相邻 TB 算子融合
resolve_tensor_layout()        // 布局/stride
plan_dtensor_memory()          // 中间 DTensor 显存分配
transpile_ugraph()             // 出 CUDA 代码（transpile.cc:332）
```

每个 `KNCustomizedOp`（一张 bGraph）由 `transpile_kn_custom_op`（`transpiler_tb.cc:253`）处理，
它会先 `get_threadblock_schedule`（`sched_tb_graph.cc:144`）——按 `TBSched`（`sched_tb_graph.h:44-52`，
分 `loop_nodes`/`post_loop_nodes`）给 tile 图排出**生产者/消费者 warp 特化 + 软件流水**的执行计划，
再配 `get_threadblock_swizzle_plan{,_hopper,_blackwell}` 与 `get_threadblock_memory_plan`
（`transpiler.h:88-116`，Hopper/Blackwell 各一版）。生成的 CUDA 就包含论文 §4.2 说的
「intra-SM optimizations such as software pipelining, register reuse, and layout optimizations
to reduce shared-memory bank conflicts」。

论文原文「MPK 用 Mirage superoptimizer [40] 搜出**优化的 thread-block 图**，交给 Mirage 编译器生成
CUDA」在源码里逐字对上：**搜索器（`src/search/`）产出 thread-block 图，transpiler（`src/transpiler/`）
把它编译成 CUDA**。

## 6. 与论文 §4.1/§4.2 逐条对照

| 论文（快照论文行） | 源码 | 一致性 |
|---|---|---|
| §4.1「按输出张量分区成 task，task 计算不相交子集」（行 462-468） | `generate_next_operator` 的 K3（`search.cc:206-280`）按 `grid/block/imap/omap` 建 bGraph | 一致 |
| §4.1「多输出维分区；默认生成与 SM 数成正比的 task 数」（行 466-468） | `get_grid_dim_cand` 枚举多轴网格并卡 `[32, MAX_NUM_THREADBLOCKS_PER_KERNEL=4096]`（`dim_strategy.cc:30-148`） | 一致 |
| §4.1「用户可指定每维并行度」 | `cython_search` 的 `*_to_explore` 参数覆盖默认候选（`search_c.cc:47-79`） | 一致 |
| §4.1「选最小化 device→SMEM 加载的分区」 | 搜索器全枚举，**由 ⑤ 的硬件 profile 选优**（`kernel.py:686-689`） | 语义一致、机制不同（实测代替代价模型） |
| §4.2「在 thread-block 级而非 kernel 级超优化」（行 1132-1134） | 搜索在 `LV_THREADBLOCK`（`search.cc:282-392`）搜 TB 算子图；`max_num_threadblock_graph_op=9` | 一致 |
| §4.2「每 task 关联参考 PyTorch 实现，用 Mirage superoptimizer 搜 thread-block 图」（行 1134-1138） | `ProbabilisticVerifier` 用参考图指纹验证（`probabilistic_verifier.cc`）；搜索产出 bGraph | 一致 |
| §4.2「交给 Mirage 编译器生成 CUDA，含软件流水/寄存器复用/布局优化」（行 1138-1142） | `transpile()` → `resolve_tb_fusion`/`get_threadblock_schedule`/swizzle/memory plan（`transpiler.h:88-134`） | 一致 |
| §4「superoptimization 比手写/CUTLASS 更系统但更贵」（行 1835-1836） | 自动路径需 compile+profile 大批候选（`kernel.py:606-692`），故 `mpk` 生产前端改用手写 task | 与落差一致 |

## 7. 与两条链条的关系（回填 `docs/07`/`docs/14` 的尾巴）

- **`docs/07` §0** 发现 `task_register.cc:1181-1318` 把形状映射到**手写** `linear_kernel_hopper<...>`，
  并注明「superoptimizer vs 手写/CUTLASS 并存」。本文给出了并存的两半：自动路径产 thread-block 图 →
  transpiler 生成 CUDA（本节 §5）；生产前端的 `.cuh` 模板是**另一条**手工路（`docs/07`）。
- **`docs/14` §7 末句**「未覆盖自动（superoptimizer）路径」——本文即是那条尾巴。`mpk` 生产前端
  **不调** `superoptimize`（grep `models/`、`mpk/` 无命中）；只有 `tests/`、`demo/`、`benchmark/`
  调它（如 `tests/ci-tests/qwen2.5/models/modeling_qwen2.py:152`、`demo/demo_llama3-8b.py:22`）。
  这与论文「MPK 自动分解」的叙述形成**明确的工程落差**——论文描述的是研究原型，`mpk` 分支的生产
  前端用手工 task 换取了「少数极重、可手调」的确定性（对标 DeepGEMM mega 家族，`docs/02`）。
- **`docs/08`** 的依赖分析器（`annotated_graph.cc`）只认 `KN_CUSTOMIZED_OP`（`docs/14` §4 核对
  `annotated_graph.cc:278-283`）——**两条路最终都汇到同一个 `KNCustomizedOp`**，故自动路径产出的图
  与手工路径产出的图在下游**完全等价处理**。这是「前端可替换、后端统一」的分层证据。

## 8. 判据 C1–C5 映射

自动路径与生产前端一样，**只贡献 C3 的范围**，不碰 C1/C2/C4/C5：

| 判据 | 自动路径是否贡献 | 说明 |
|---|---|---|
| C1 单次 launch | 否 | 搜索/生成都在编译期；launch 由运行层一次做（`docs/06`） |
| C2 常驻 | 否 | 由 worker/scheduler 保证 |
| C3 多算子范围 | **是（范围来源）** | 搜索器决定把多少算子融进几张 bGraph（`max_num_kernel_graph_op=5`、`max_num_threadblock_graphs=1`） |
| C4 设备内协调 | 否 | 由事件/原子队列保证 |
| C5 SM 级调度 | 否 | 由去中心化调度保证 |

一句话：**自动路径把论文 §4.1/§4.2「怎么自动切、怎么自动生成」补齐；它和手工前端一样，都只是把
「要融什么、融成什么形状」定下来（C3 的范围），常驻与调度永远在运行层。**

## 9. 局限

- **纯源码走读，未构建/运行**：搜索器需在 GPU 上 compile+profile 候选图（`kernel.py:606-692`），
  本机不具备该跑批条件；所有结论基于 `f9eb70c` 快照行号。
- **只精读主搜索循环**：`src/search/` 另有 `abstract_expr/`（代数表达式等价类）、
  `range_propagation/`（range 推导）、`symbolic_graph/`（符号化搜索 `generate_kernel_graphs_symbolic`
  `search.cc:446-464`，走 `generate_next_symbolic_operator` `search.cc:584-835`）三块，本文只给了
  它们与主循环的接口（剪枝/约束/召回），未逐行走读其内部算法。
- **transpiler 只给管线结构**：`src/transpiler/*` 共约 8 千行（`transpiler_tb{,_hopper,_blackwell}.cc`
  各 1.5 千行），本文把它拆成 resolve/schedule/codegen 三段与真实行号，未逐条读 SASS 级生成逻辑
  （`docs/07` 从 device 模板侧覆盖了一半）。
- **未覆盖 benchmark 数字**：`benchmark/`、`demo/` 里超级化器相对手写的加速（论文表）未在本机复现，
  故本文不引用任何性能数。
- **多 GPU/NVSHMEM 路径未展开**：`resolve_distributed_config`（`transpiler.h:53-58`）的
  `gpu_dim`/NVSHMEM 分支与 `multigpu.py` 的张量并行切分未走读。
