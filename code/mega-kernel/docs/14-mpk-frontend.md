# MPK 前端与算子层走读：从 mirage 图 API 到 `task_configs` / bGraph

> 本文是 MPK 线上实现源码走读链条的**最上游**一节，补齐 `docs/06`/`docs/07`/`docs/08` 三篇共同的
> 上游空白：那三篇分别讲「运行层 worker/scheduler」「单 task device 实现」「依赖分析器」，但都**从
> 已经存在的 bGraph / `task_configs` 讲起**——而「一张算子图到底是**谁**、**怎么**建出来的」一直没走。
> `docs/08` §对照表里明确留了这条尾巴：算子分解「不在 `annotated_graph.cc`；由 `task_configs` +
> bgraph `grid_dim` 提供」，而「分解策略本身在 `task_register.cc`/前端」。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`/`docs/07`/`docs/08` 同一快照，
> 所有行号都据此核对。
>
> 定位：`docs/01` §3 讲「论文里的 tGraph 表示」，`docs/08` 讲「tGraph 的下游：依赖分析 → 事件/任务
> emit」，本文讲**上游**：「Python 前端如何用一个 `customized()` + `register_task()` 调用把每个算子
> 固化成一张 bGraph + 一条 `task_config`，从而**手工**拼出整图」。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`。
>
> 本文为**纯源码走读（资料分析）**：不重新构建/运行 mirage。读者若想先看运行结果，见 `docs/09`
> （构建 + 端到端延迟）与 `docs/04`（本专题的原生 CUDA 微基准）。

## 0. 编译栈四层与本文的位置

MPK 把「模型 → 可 launch 的持久化 kernel」这条路切成四段，前四篇文档各走一段：

| 层 | 产物 | 主要文件 | 已走读于 |
|---|---|---|---|
| **① 前端/算子层** | 一张 `KNGraph`：一串 `KNCustomizedOp`，每个带一张 bGraph；`task_config` 表 | `python/mirage/mpk/*.py`、`python/mirage/{kernel,threadblock}.py`、`src/kernel/{operator,customized,graph}.cc` | **本文（docs/14）** |
| ② 依赖分析/构图 | `AnnotatedGraph`：layer / edge / fork-join group / `task_view` | `src/kernel/annotated_graph.cc` | `docs/08` |
| ③ 运行层 | `persistent_kernel.cuh`：worker/scheduler 设备循环 | `include/mirage/persistent_kernel/*` | `docs/06` |
| ④ 单 task device | `tasks/hopper/*.cuh` 等模板实例化 | `src/kernel/task_register.cc`、`tasks/` | `docs/07` |

一句话结论（对上游 total 的最关键洞察）：**MPK 的线上前端不是「从高层算子自动分解出 task」的编译器，
而是一套「手工作者」API**——用户/模型 builder 对每一层显式地写一个 `*_layer()`，在里边**手写**
threadblock 级计算体（bGraph），再声明它属于哪种 `TaskType`。论文 §4.1 的「算子分解」在自动路径
（Mirage superoptimizer）里是一种搜索，但在 `mpk` 分支的**生产前端**里被替换成了**手工切分**。
这与 DeepGEMM mega 家族「手工角色、少数极重 task 类型」的做法是同一路数（交叉引用 `docs/02`）。

## 1. Python 前端的三层对象

前端由三层 Python 对象搭起来，都薄薄地包在 Cython/C++ 之上：

- `KNGraph`（`python/mirage/kernel.py:208`）：**kernel 级**图。持有算子序列，暴露
  `new_input`（`:220`）、`matmul`/`rms_norm`/…（`:240-282`，高层算子 API）、`customized`（`:283`）、
  `register_task`（`:802`）、`generate_task_graph`（`:805`）。
- `TBGraph`（`python/mirage/threadblock.py:4`）：**threadblock 级**图，描述「一个 task 内部怎么算」。
  暴露 `new_input`（`:8-15`）、`new_output`（`:17`）、`matmul`/`reduction`/`rms_norm`/`forloop_accum`
  等（`:20-77`）。
- `DTensor` / `STensor`：kernel 级 / threadblock 级张量句柄（由 Cython 指针包成，见
  `_cython/core.pyx`）。

`KNGraph` 其实同时提供**两套用法**：

1. **高层算子 API**（`self.cygraph.matmul(A,B)`、`rms_norm(A,shape)`…）：面向 Mirage 的「超优化器」
   路线——先用高层算子写模型，由后端搜索出 threadblock 实现。**`mpk` 生产前端基本不用它**（只用了
   `view`/`narrow`/`split`/`fuse_tensors`/`shuffle_tensors` 这类**虚拟张量**算术，见 §4）。
2. **手工 task API**（`customized(inputs, tb_graph)` + `register_task(tb_graph, name, params)`）：
   面向 MPK 持久化 kernel 路线——**每个算子的实现由调用者手写在 `tb_graph` 里**。

用户真正面对的是 `PersistentKernel`（`python/mirage/mpk/persistent_kernel.py:413`），它在 `:501`
构造 `self.kn_graph = KNGraph(CyKNGraph(disable_fingerprint=True))`，然后把上面第 2 套用法封装成一堆
语义化的 `*_layer()` 方法。上层 `MPK`（`python/mirage/mpk/mpk.py:132`）负责模型组装
（`build()`，`:471`）、张量分配、请求生命周期，最终在 `compile()`（`mpk.py:491`）里调
`PersistentKernel.compile()`。

## 2. 一个 layer 到底做了什么（三个真实例子）

`*_layer()` 的骨架高度一致，以 `rmsnorm_layer`（`persistent_kernel.py:865-881`）为例：

```python
tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
tb_graph.new_input(input,  (0, -1, -1), 1, True)
tb_graph.new_input(weight, (-1, -1, -1), 0, True)
tb_graph.new_input(output, (0, -1, -1), 1, True)
self.kn_graph.customized([input, weight, output], tb_graph)
self.kn_graph.register_task(tb_graph, "rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm")
```

三步语义：

1. **建 bGraph**：`CyTBGraph(grid_dim, block_dim, 1, 64)` = `(grid_dim, block_dim, forloop_range=1,
   reduction_dimx=64)`。`grid_dim` 就是「这一层要切出多少个 task」——它之后变成
   `KNCustomizedOp::bgraph.grid_dim`，并在运行层展开成 `grid.x*y*z` 个 task（§4）。
2. **声明 grid→tensor 映射**：`tb_graph.new_input(dtensor, input_map, forloop_dim, store_in_dmem)`。
   `input_map` 是一个 `int3`（`_cython/core.pyx:1167-1180`，`assert len==3`，转 `int3`），逐维声明
   「threadblock grid 的哪一轴对应张量的哪一维、或该维被复制（-1）」。例如 rmsnorm 的
   `input=(0,-1,-1)` 表示「grid 的 x 轴切张量第 0 维，其余维整块看到」，`weight=(-1,-1,-1)` 表示
   **权重被每个 task 完整复制**（复制轴的流量会被重复读，这正是依赖分析器 `derive_last3` 要处理的
   情况，见 `docs/08` §2）。
   **这两个 `input_map` / `output_map` 就是 `docs/08` 里 `EdgeInfo::input_map` / `output_map`、
   `TaskView::axis_map` 的来源。** 换言之，前端 `new_input` 的第二个参数，是整条事件/分块分析的输入。
3. **固化**：`customized(inputs, tb_graph)` 生成一个 `KN_CUSTOMIZED_OP`；`register_task(...)` 给它
   贴上 `TaskType` + `variant_id`。

再看两个稍复杂的：

- `embed_layer`（`:842-863`）：embedding 是「查表」，它把 `input`/`weight`/`output` 用不同的
  `input_map`（`(0,1,-1)` / `(1,-1,-1)` / `(1,0,-1)`）映射到同一 grid，然后
  `register_task(tb_graph, "embedding", [input_source])`——注意 `params` 是「变体参数」，一路传到
  `task_register` 的注册函数里用来挑编译期常量（`docs/07` §0）。
- `linear_layer`（`:2390-2456`）：除了 `new_input`/`customized`，还在 `register_task` **之前**做了一堆
  **合法性断言**（输出列数能被 `grid_dim[0]` 整除、SM100 下每 task 列片 16 B 对齐…），并**按
  `target_cc` 和权重形状选不同 task 名**（`linear_sm100` / `linear_swapAB_hopper` /
  `linear_cutlass_hopper` / `linear_with_residual_hopper`…）。**这说明「算子分解」的决策（切多少、
  用哪种实现）在前端就是一堆手写规则**，与论文 §4.1 的自动分解形成对照。
- `allreduce_layer`（`:2488-2512`）：把实现选择完全委托给 `auto_select_allreduce_implementation(
  world_size, rank)`，后者返回一个对象，再 `best_implementation.register_tasks(self, tensors,
  grid_dim, block_dim, params)`——**一个 layer 甚至可以注册多个 task**（allreduce 的分段/留 buffer）。

`grid_dim` 的具体数值也由前端 helper 算：例如 `grid_for_rmsnorm_linear_layer(size)`
（`models/utils.py:3-14`，按 `size` 取 96 / 64 / `size//256`）、`grid_for_splitk_linear_layer`
（`:17-35`，SM100 split-K 的 `(output//128, splits, 1)`，并保证 `reduction//splits` 是 64 的倍数）。
模型 builder 就在这些 helper 与 `*_layer` 之间穿梭：Qwen3 的 `build_layers`
（`models/qwen3/builder.py:288`）在 `:354-551` 里顺序调用 `rmsnorm_layer` / `linear_layer` /
`paged_attention_layer` / `splitk_linear_layer` / `allreduce_layer` / `silu_mul_layer`，
**一个 transformer 层就是一串 `*_layer` 调用**。

## 3. C++ 侧：一个 `customized()` 生成什么，`register_task()` 记什么

### 3.1 算子模型

kernel 级算子的基类是 `KNOperator`（`include/mirage/kernel/operator.h:29-53`）：`op_type` +
`input_tensors` + `output_tensors`。它**不含 bGraph**——bGraph 只在 `KNCustomizedOp`
（`include/mirage/kernel/customized.h:40`：`threadblock::Graph bgraph;`）里。`KNInputOp` /
`KNOutputOp`（`operator.h:56-90`）分别是图的源/汇（`input_map` / `output_map` 也在它们身上）。

`Graph::customized`（`src/kernel/customized.cc:35-41`）→ `create_customized_op`（`:44-105`）：

- **校验对齐**（`:49-62`）：遍历 `_graph.operators` 统计 `TB_INPUT_OP` 个数，要求它等于传入
  `inputs.size()`，且**按位置一一相等**（`assert(inputs[num_inputs] == input_op->dtensor)`）——
  所以 `customized(inputs, tb_graph)` 的 `inputs` **顺序必须与 `tb_graph.new_input` 的调用顺序一致**。
- **算输出大小**（`:64-72`）：累加各 `TB_OUTPUT_OP` 的 `dtensor.data_size()` 做显存分配检查。
- **构造 `KNCustomizedOp`**（`:72-105`）：把 bGraph 的 `(grid_dim, block_dim, forloop_range,
  reduction_dimx)` 拷进 op 自己的 `bgraph`；输出 `DTensor` 由 `TB_OUTPUT_OP` 反推。

于是：**Python 调用 `customized(...)` 返回的 `[output]` 就是这一层的输出张量**（上例里 rmsnorm
的输出），后继 layer 拿它当输入，串成 DAG。这就是 `PersistentKernel` 累积整图的方式。

### 3.2 `register_task` 是一个 95 分支的「name → TaskType」大表

`Graph::register_task`（`src/kernel/graph.cc:439` 起到函数尾）是整条前端里最「笨」也最关键的一段：

```cpp
void Graph::register_task(char const *task_type, std::vector<int> params) {
  std::string name = std::string(task_type);
  KNOperator const *op = operators.back();               // ← 取最后压入的 op
  assert(op->op_type == type::KN_CUSTOMIZED_OP);
  KNCustomizedOp const *customized = static_cast<KNCustomizedOp const *>(op);
  TaskRegister *task_register = TaskRegister::get_instance();
  if (name == "embedding") {
    int variant_id = task_register->register_embedding_task(customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 1, TASK_EMBEDDING, variant_id);
  } else if (name == "rmsnorm") {
    ...
```

- **`op = operators.back()`**：`register_task` 不接 op 参数，直接认「刚 `customized()` 进去的那个
  op」。这就是为什么 Python 侧必须**先 `customized` 再 `register_task`**，也解释了 `core.pyx:1118-1129`
  里 `register_task(bgraph, task_type, params)` 的 `bgraph` 形参**在绑定层被直接丢弃**
  （只把 `cname`/`cparams` 传进 C++）——C++ 根本不用它。
- 每个分支做两件事：①调 `TaskRegister::register_*_task(bgraph, params)` 得到 **`variant_id`**
  （这一步会**读 bGraph 的形状/块大小**、挑编译期常量、注册 device 模板——已在 `docs/07` §0 走读）；
  ②写 **`task_config[op] = (num_inputs, num_outputs, TaskType, variant_id)`**。
- **`task_config`** 是 `Graph` 的成员（`include/mirage/kernel/graph.h:230-233`），类型
  `unordered_map<KNOperator const*, tuple<int,int,TaskType,int>>`——正是 `docs/08` 里的
  `TaskConfigMap`（`include/mirage/kernel/annotated_graph.h:158-160`）。
- 大小：**95 处 `task_config[op] = ...`**（`grep -c`）。也就是说 `mpk` 分支一共手工登记了 95 个
  算子/变体的任务类型，从 `TASK_EMBEDDING` 到 `TASK_MOE_W2_FP8_SM100`。
- **`num_inputs` 也在前端硬编码**：如 `rmsnorm`=`(2,1,…)`（input+weight→output）、
  `attention`=`(7,1,…)`、`glm_moe_router`=`(2,3,…)`。它必须与 `tb_graph.new_input` 调用数一致，
  否则 `annotated_graph.cc:298-302` 会抛 `bgraph inputs/outputs count mismatch`。

> 对照 `docs/07` §0：`register_*_task` 负责「一个 task 内部怎么写」（device 模板实例化）；本文补的是
> 它的**调用者**——前端 `register_task` 负责「这个 task 叫什么、几张输入几张输出、属于哪个 TaskType」。

## 4. `task_configs` + bGraph 如何被消费（接 `docs/08`）

整张 `KNGraph` 连同 `task_config` 表被交给依赖分析器，入口在
`src/kernel/runtime.cc:222-243`（`register_mugraph(..., task_configs)` → `build_annotated_graph(
graph, task_configs)`，`:243`）。三个只在本文视角下才明显的点：

1. **依赖分析器只认 `KN_CUSTOMIZED_OP`**：`src/kernel/annotated_graph.cc:278-283` 遍历
   `kn_graph.operators` 时，非 `KN_CUSTOMIZED_OP` 直接 `continue`。于是「`KNGraph` 里
   `matmul`/`rms_norm` 这类高层算子」**根本不会进入 MPK 的 tGraph**——它们只在 Mirage 的超优化器
   路线里存在。MPK 图的「节点」= 前端每一次 `customized()`。「这张图里每个节点带一张手写 bGraph」
   就是本文标题里 bGraph 的由来。
2. **`grid_dim` = task 数**：运行层在 `runtime.cc:1395-1420` 用
   `bgraph.grid_dim.x/y/z` 三重循环为每层展开 `grid.x*grid.y*grid.z` 个 `FullTaskDesc`
   （`op_total_subtasks`，`:1415-1416`），并用 `TaskView` 的 `last3` 把「每个 event 放行哪些 task」
   算出来。前端 `CyTBGraph(grid_dim, …)` 的 `grid_dim` 因此**直接决定这一层切多少个 task**——前端
   选 96/64/`size//256`，运行层就展开 96/64/`size//256` 个。
3. **虚拟张量不进 operators**：`view`/`narrow`/`split`（`persistent_kernel.py:826-840`）**不创建
   `KNOperator`**（`graph.h:150-160` 注释：「他们不创建 KNOperator、不出现在 `operators` 里」），
   只让下游把同一段 storage 当新形状看到。这对应 `docs/08` §2 的**虚拟张量 → barrier 边**
   （`annotated_graph.cc:391-400` 的 `is_barrier_edge`）——前端只做 reshape 语义，依赖分析器把它
   坍缩成一条粗粒度 barrier event。

## 5. 与论文对照

| 论文（快照行） | 实现 | 差异 |
|---|---|---|
| §4.1 算子分解 operator decomposition（行 462–468） | **前端 `*_layer()` 手工切分** + `grid_for_*` helper；95 个手工 `task_config` 分支 | 论文叙述是编译器/超优化器自动把算子切成 task；`mpk` 生产前端是**手工**任务图（自动路径 `docs/07` §0 也提到，线上二者并存） |
| §4.2 每 task 的 CUDA 实现生成（行 1126–1146） | `TaskRegister::register_*_task(bgraph, params)` 返回 `variant_id`（`docs/07` §0） | 一致 |
| §3 SM 级图表示 tGraph（行 345–448） | `annotated_graph.cc` 从 bGraph + `task_configs` 构造 `AnnotatedGraph`（`docs/08`） | 一致（论文的 task/event 交替 = 实现的 layer/edge/event） |
| §1「三好处」（行 90–101） | 前端产出**单张静态图**，无运行时重组；`PersistentKernel.compile()` 一次固化（`persistent_kernel.py:3207-3224`） | 一致：前端构图只在编译期发生 |

论文把「图」当作**编译器的输入/输出**；实现里它是**前端 API 的直接产物**，且前端是手工的。这条
「论文描述自动、线上实现手工」的落差，与 `docs/07` §0 发现的「superoptimizer vs 手写/CUTLASS 模板
并行」是同一个工程化事实的两面。

## 6. 判据 C1–C5 映射

前端层本身**不贡献任何判据**——它只是「把整图固化成串行 API 调用」的构图阶段，既不 launch、也不常驻、
更不做设备内调度。但它决定了 megakernel 的**范围**（C3「多算子」）：

| 判据 | 前端是否贡献 | 说明 |
|---|---|---|
| C1 单次 launch | 否（间接） | 整图被 `compile()` 固化成一份图；launch 由 `docs/06` 的运行层做一次 |
| C2 常驻 | 否 | 由运行层 worker/scheduler 保证 |
| C3 多算子范围 | **是（范围来源）** | 前端每 `customized()` 一个算子，串成整图；范围 = 图里 customized op 数 |
| C4 设备内协调 | 否 | 由运行层事件/原子队列保证 |
| C5 SM 级调度 | 否 | 由运行层去中心化调度保证 |

一句话：**前端把「要融什么」定下来（C3 的范围），运行层把「怎么常驻着调度」做出来（C1/C2/C4/C5）。**
这也解释了为什么 `mpk` 前端的粒度是「一个算子 = 一个 `customized` = 一种 TaskType」，而不是
「一个元素级算子 = 一个 task 实例」——后者由 `grid_dim` 在运行层展开。

## 7. 局限

- **纯源码走读，未构建/运行**：所有结论基于 `f9eb70c` 快照的行号，未在本机重跑 mirage；端到端行为
  见 `docs/09`（含 P25-1 的 launch 顺序根因）。
- **未逐行走读全部 95 个任务分支**：本文只精读 `embedding`/`rmsnorm`/`rmsnorm_linear`/`linear`/
  `allreduce` 等代表性 layer，其余（MoE 系列、MLA/paged-attention 系列、speculative 系列）只统计了
  机制，未逐一展开。
- **未走读模型 builder 的完整组装逻辑**：`models/{qwen3,glm4_moe,deepseek_v3,dflash,eagle3,gpt_oss,
  inkling}/builder.py` 只以 Qwen3 的 `build_layers` 为例，未覆盖分布式/张量并行切分
  （`multigpu.py`）与请求/元数据管理（`mpk.py` 的 `MPKMetadata`）。
- **未覆盖自动（superoptimizer）路径**：本文只说明 `mpk` 前端不走它；Mirage 主分支的超优化器
  如何从高层算子搜出 threadblock 实现（`src/search/`、`src/transpiler/`）未在本文范围。
- **虚拟张量的语义边界未深挖**：`view`/`narrow`/`split`/`fuse_tensors`/`shuffle_tensors` 的
  layout 变换与 `annotated_graph` 的窗口分析如何精确对应，只给了 `is_barrier_edge` 一处锚点
  （更细的走读在 `docs/08` §2）。
