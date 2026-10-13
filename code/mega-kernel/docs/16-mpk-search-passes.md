# MPK 搜索器的三块辅助模块：e-graph 等价类、区间推导、符号化图

> 本文补齐 `docs/15`（P101-1）**§9 明留的尾巴**：
>
> > **只精读主搜索循环**：`src/search/` 另有 `abstract_expr/`（代数表达式等价类）、
> > `range_propagation/`（range 推导）、`symbolic_graph/`（符号化搜索）三块，本文只给了
> > 它们与主循环的接口（剪枝/约束/召回），未逐行走读其内部算法。
>
> `docs/15` 走通了主搜索循环 `generate_next_operator` / `generate_next_symbolic_operator`
> 的 **K1/K2/K3 + B1/B2 三层 DFS**、三横切剪枝、`ProbabilisticVerifier` 与 `transpile()`。
> 但那个主循环**自己不做「候选是否值得试」的判断**——它把判断委托给三块辅助模块：能不能用代数
> 等价关系证明候选是最终输出的一部分（`abstract_expr`）、候选的访问范围是否落在目标范围内
> （`range_propagation`）、以及整个搜索能否在**维度符号化**的图空间里一次枚举多个具体形状
> （`symbolic_graph`）。本文把这三块的**内部算法**逐行走通。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，与 `docs/06`/`docs/07`/`docs/08`/`docs/14`/`docs/15`
> 同一快照，所有行号都据此核对。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（以下简称「快照论文」）。
>
> 本文为**纯源码走读（资料分析）**：不重新构建/运行 mirage，不产生新实验。

## 0. 定位：主循环之外的三个「判断器」

`docs/15` 已给出 `generate_next_operator`（`src/search/search.cc:129-392`，非符号路径）与
`generate_next_symbolic_operator`（`search.cc:584-834`，符号路径）的骨架。主循环每到一个候选算子，
都会调用下列三类判定；它们各自是一个**独立、可单测**的子系统：

| 模块 | 目录 | 它回答的问题 | 在主循环的调用点 |
|---|---|---|---|
| **代数的** | `src/search/abstract_expr/` | 候选输出张量的表达式是不是最终输出的**子表达式**（同构/等价类）？ | `check_abstract_expr`→`subexpr_to_final_expr`（`search.cc:190,373,504-519,654,816`） |
| **几何的** | `src/search/range_propagation/` | 候选图里某输入的一个点，经过整张图正/反向传播后，落在目标输入范围内的哪些点？ | `preprocess` 建 `init/target_ranges`（`search.cc:482-484`），`check_range`（`search.cc:338`） |
| **符号的** | `src/search/symbolic_graph/` | 能否**不枚举具体形状**、而用维度变量 + 约束把一大类图一次性表示/拒绝？ | `generate_next_symbolic_operator`（`search.cc:584-834`）全程 |

三块是**正交**的：`abstract_expr` 管「算的是什么」、`range_propagation` 管「读/写的是哪一段」、
`symbolic_graph` 管「形状是变量还是常量」。下面 §1–§3 分别走读，§4 回收调用点、§5 对照论文、
§6 给局限。

## 1. `abstract_expr/`：用 e-graph 做代数等价类剪枝

### 1.1 两套表示：C++ 表达式树 + egg 字符串

C++ 侧（`include/mirage/search/abstract_expr/abstract_expr.h`）是一棵**只含算子语义、不含形状**的
表达式树，虚基类 `AbstractExpr`（`:28-35`）只有两个纯虚函数：

- `to_string()`：给人读的（如 `(x+y)`、`rms(128, x)`）；
- `to_egg()`：给 **egg**（Rust e-graph 库）读的 S 表达式（如 `(+ x y)`、`(rms 128 x)`）。

节点类型共 15 个（`abstract_expr.h:42-231`）：`Var`、`Add`、`Mul`、`Div`、`Pow`、`Exp`、`Square`、
`Sqrt`、`Silu`、`Gelu`、`Relu`、`Clamp`、`RMS`（带 `reduction_degree` 的「均方根」节点，`:188-206`）、
`Red`（带 `reduction_degree` 的「求和」节点，`:208-228`）。构造一律走 `abstract_expr_make_*` 工厂
（`abstract_expr.cc:243-343`），例如 `abstract_expr_make_red(1, x)` 直接返回 `x`（`abstract_expr.cc:328-330`，
「reduce degree 1 等于不归约」）。

关键设计：**形状被折叠进节点自己的字段**，而不是留在树的形状上。`Red`/`RMS` 的 `reduction_degree`
是一个 `TensorDimExpr`（可能是编译期常量，也可能是符号维度——见 §3），所以 `sum_128(x)` 与
`sum_256(x)` 是**两个不同的节点**（在 egg 里是 `(sum 128 x)` vs `(sum 256 x)`）。这解释了为什么
`get_abstract_expr` 对每个算子都要读 `tensors[0].dim[...]`（`abstract_expr_for_ops.cc:17,43,52,101,...`）——
归约长度是这个节点的**身份**。

### 1.2 egg：把表达式丢进 e-graph 做饱和重写

Rust 侧（`src/search/abstract_expr/abstract_subexpr/src/lib.rs`，用 `egg` crate，`lib.rs:1`）定义了
一套语言 `Expr`（`lib.rs:11-32`）与**重写规则集** `rules()`（`lib.rs:36-110`）：

- **代数定律**（`lib.rs:38-53`）：加/乘交换律、结合律、分配律、除法合并（`div-add`/`mul-div`/`div-div`）；
- **`sum` 归约定律**（`lib.rs:55-62`）：`sum` 与 `+`/`*`/`/` 的换序（如 `sum i (a+b) = sum i a + sum i b`）、
  以及**两个 `sum` 的交换**（`lib.rs:55`）；
- **动态因式分解规则**（`lib.rs:72-105`）：从表达式中正则提出所有 `(sum N ...)` 的常量 `N`
  （`lib.rs:127-131`），对每个 `N` 做质因数分解——若 `N = j*k`，就注入两条规则
  `(sum N a) ⇄ (sum j (sum k a))`（`lib.rs:84-98`）。这正是「把一次大归约拆成两级小归约」的代数依据。

FFI 只有两个入口（`lib.rs:121-146,148-183`，对应 `abstract_expr.h:14-23` 的 `extern "C"` 声明）：

- `get_egraph(expr)`：把**最终输出的表达式**解析进一个新的 e-graph，用
  `Runner::default().with_iter_limit(100_000).with_node_limit(100_000).with_time_limit(10s).run(...)`
  跑到饱和（`lib.rs:133-144`），把饱和后的 e-graph 追加进全局 `Graphs`（`lib.rs:119,142-144`）；
- `egg_equiv(subexprs, len)`：对每个候选子表达式，`graph.lookup_expr(&subexpr)`（`lib.rs:171`），
  只要在**任一** final e-graph 里能查到等价的 e-class，就返回 `true`（`lib.rs:170-176`）。

C++ 侧把二者包成两个函数（`abstract_expr.cc:16-49`）：

- `initialize_final_expr(expr)` = `get_egraph(expr->to_egg())`（`abstract_expr.cc:16-18`）；
- `subexpr_to_final_expr(exprs)` = 批量 `to_egg()` → `egg_equiv()`（`abstract_expr.cc:25-49`）。

### 1.3 剪枝语义与调用点

`KernelGraphGenerator::preprocess` 对**每个计算图输出**调 `initialize_final_expr`
（`search.cc:487-495`）；随后主循环对每个候选算子算出输出表达式 `expr = get_abstract_expr(...)`，
调 `check_abstract_expr(expr)`，其实现是「只要 `expr` 是某个 final 表达式的子表达式就放行」
（`search.cc:504-519`）。于是**搜索被限制在「最终输出所依赖的代数闭包」内**——任何与最终结果
在代数上无关的算子组合在生成前就被砍掉。

一个**重要的口径 caveat`：`check_abstract_expr` 的签名带第二个参数
`TensorDimConstraints const &constraints = {}`（`search.h:90-91`），但函数体**完全没用它**
（`search.cc:504-519`）。也就是说符号路径下「子表达式 + 维度约束」的联合判定**只实现了前半**。

`get_abstract_expr` 有四个重载（`abstract_expr_for_ops.h:11-30`）：非符号路径两个（`kernel::DTensor` /
`threadblock::STensor`，`abstract_expr_for_ops.cc:6-181`），符号路径两个（`SymbolicDTensor` /
`SymbolicSTensor` 带图上下文，`abstract_expr_for_ops.cc:183-310`）。非符号版里对不支持的算子是
`assert(false)`（`:60-63,178-180`），符号版则多覆盖了几个 `TB_FORLOOP_ACCUM_*` 形态，且归约长度用
`dim_expr_make_mul(g.forloop_range.dim_expr, ...)` 拼成**符号表达式**（`:234-259`）——这正是符号路径
把「形状」也抽象掉的落点。

`abstract_expr_eval`（`abstract_expr_eval.cc`）是把整张图**自底向上求表达式**的驱动器：非符号版
用 `guid`→expr 的 map（`:9-96`），符号版用**按算子下标的 vector**（`:98-157`，因为符号图里张量本来就按
加入顺序编号）。特别地，`TB_CONCAT_1 + TB_CONCAT_0 + TB_MATMUL` 被识别为**一个**语义算子
`TB_CONCAT_THEN_MATMUL_OP`（`:26-46`），表达为「两段归约再相加」
（`abstract_expr_for_ops.cc:160-175`）——这是 lora / attention 里常用的 concat 融合。

## 2. `range_propagation/`：区间代数与网格映射

### 2.1 数据结构

- `Range`（`include/mirage/search/range_propagation/range.h`）：一个 `[lower, upper)` 的**多维超矩形**，
  带 `valid` 位（`range.h:10-56`）。`is_subrange`/`is_empty`/`truncate`/`offset`/`transpose`/
  `extend_dim` 是基本运算（`range.cc:15-96`）。`all_range` 用 `INF=1e9` 表示「整维」（`range.h:51,58-62`），
  `empty_range` 是 `lower>=upper`（`range.cc:34-44`）。
- `RangeSet<R,T>`（`range_set.h:10-119`）：**一组带路径**的 Range。`PropagationPath<T>`
  （`propagation_path.h:8-27`）记录访问过的节点 id，`RangeSet::extend_path`（`range_set.h:41-51`）
  在节点已被访问过时返回 `false`——这是**防在环图上无限传播**的关键。`combine` 拼接两组 range
  并 `simplify`（`range_set.h:75-114`，删掉被别的 range 包含的冗余项）。
- `IKNRange` = `RangeSet<KNRange, size_t>`、`ITBRange` = `RangeSet<TBRange, size_t>`
  （`irange.h:23-39,68-81`）。`TBRange`（`tbrange.h:9-34`）多两维：`block_range`（落在哪个 CTA）
  与 `forloop_range`（落在流水第几迭代），即一个 STensor 的访问被记为 `(tensor | block | forloop)`。

### 2.2 传播规则：forward / backward / multiplicative_interact

每个算子对「操作数 i 的 range」有三类规则（`irange.h:41-52` 的单算子版 + `:54-64` 的向量版）：

- `forward_propagate`（`irange.cc:291-361`）：**输入 range → 输出 range**。
  - 逐元素算子（exp/square/sqrt/silu/gelu/relu/clamp）走 `EXP_AS_IDENTITY`
    （`irange.cc:296-305`，常量 `EXP_AS_IDENTITY=true` 在 `irange.h:20`）——即「按元素不变」；
  - add/mul 直接透传（`:306-310`）；
  - div/pow/matmul 对**被广播/被归约的那一维** `extend_dim` 成整维（`:311-331`）；
  - reduction 把被归约维 `extend_dim`（`:332-339`）；
  - rms_norm 按 `normalized_size` 反推要扩几维再截断（`:340-351`）。
  - 结束时 `extend_path(output_tensor.guid)` 做环检测（`:356-360`）。
- `backward_propagate`（`irange.cc:363-429`）：**输出 range → 输入 range**，规则对称。
- `multiplicative_interact`（`irange.cc:431-464`）：**一个操作数的 range 影响另一个操作数**——
  即乘法/矩阵乘里「一个操作数的非零范围会限制另一个操作数的范围」的交叉推理
  （mul：`opd_from!=opd_to` 时透传；matmul：转置 + 扩维，`:443-452`）。

向量版（`:466-559`）把多操作数合并：forward 对每个操作数分别算再 `combine`（`:489-496`）；
遇 `KN_CUSTOMIZED_OP` 则**下钻到 bGraph**，用 `range_propagate_forward/backward(pos?, tb?)` 递归
（`:472-488,505-521`），所以 range 推理是**跨 KN/TB 两层**的。

### 2.3 网格映射：d2s / s2d

`propagate_from_dtensor_to_stensor`（`irange.cc:55-147`）与反向的 `propagate_from_stensor_to_dtensor`
（`:169-268`）是 range 在 **device 张量空间 ↔ SMEM 张量空间**之间的换算，依据 `input_map`/`output_map`
（`int3`）与 `grid_dim`、`forloop_dim`、`forloop_range`。核心逻辑：对映射到某个网格轴的张量维
（`dim_map.{x,y,z} != -1`），按 `dim_size = dtensor.dim[dim_idx] / dim_deg` 把全局下标切成
「第几个 CTA（block）+ CTA 内偏移（tensor dim）+ 第几迭代（forloop）」三元组（`:87-125`）；
未映射的轴则 block range 取满（`:126-128`）。有文件内注释明说**只处理范围落在单个 STensor 内的情形，
跨多个 STensor 时返回子集**（`:88-90`）——一个已知的近似。

### 2.4 驱动器与调用点（以及「已禁用」）

`range_propagate_forward/backward`（`irange.cc:852-888,908-950`）按算子顺序/逆序把整图推一遍，
每步把结果 `combine` 进 range map；`interact_range_propagate`（`:890-906,952-968`）只做交叉推理。
三者组合成 `range_propagate`（`:1050-1062`）：先 forward，再（若给定 bGraph）跨层 interact，
再整体 interact，最后 backward——返回的是**从最终张量反向投影到每个输入张量的 range**。

调用入口：`preprocess` 建 `init_ranges = get_init_ranges(computation_graph)`
（`irange.cc:1095-1125`：取每个输入张量的一个**全零点** point range，`rand()` 那条被注释掉了
`:1106`）、`target_ranges = get_interact_ranges(init_ranges, ...)`（`:1147-1155`）；主循环在「回 kernel 级」
那一刻调 `check_range(init_ranges, target_ranges, *c.kn_graph)`（`search.cc:338`）。

**但 `check_range` 目前被显式禁用**：两个重载都在函数体第一行 `return true;`
（`irange.cc:1068-1069` 和 `1085-1086`），后面 1100 行推理代码**在当前快照里根本不会被执行**。
这和 `docs/15` §9 观察到的「`instantiate_symbolic_graph` 直接 `return false`」是同一类
「算法已写完但尚未接入主流程」的状态（见 §3.4）。因此 range 剪枝在当前 `mpk` 快照里
**不贡献任何剪枝**；它是一块**已实现、待接线**的基础设施。

## 3. `symbolic_graph/`：维度变量与约束下的符号化搜索

### 3.1 维度表达式：`TensorDimExpr`

`tensor_dim_expr.h` 定义一棵**整数维度表达式树**：`TensorDimVar`（一个 `uint32_t index` + INT/BOOL 类型，
`tensor_dim_expr.h:47-64`）、`TensorDimConst`、`TensorDimAdd/Mul/Div`、`TensorDimPow`
（仅 log-scaled，`:155-175`）、`TensorDimIte`（`:177-197`）。每个节点有两种求值（`tensor_dim_expr.cc`）：

- `get_value(assignments)`：给一组具体赋值后**整数求值**（`:73-110`）；
- `to_z3(c, assign, log_scaled)`：转成 z3 表达式（`:312-390`）。**log-scaled 模式**把乘法变加法、
  除法变减法、幂变乘（`:355-382`），把非线性的维度关系线性化以便 SMT 求解——
  这是 `docs/08`/`docs/15` 里 `tensor_dim_var_index_t`/`DimVarAssignments` 的底层算术。

`SymbolicTensorDim`（`symbolic_tensor_dim.h:8-21`）只是 `TensorDimExpr` 的包装，并重载 `+ * / ^`
（`symbolic_tensor_dim.cc:18-36`），使维度表达式可以像普通整数一样写。相等性/哈希都定义在
**结构**上（`symbolic_tensor_dim.cc:14-16,43-46` → `same_expr_as`/`hash`），所以两个结构相同的
表达式会被当作同一个维度。

### 3.2 一张符号图的组成

`SymbolicKNGraph` / `SymbolicTBGraph`（`symbolic_graph.h:18-91`）用**与具体图同构、但维度是表达式**
的表示：

- `SymbolicKNGraph`：`operators`（`SymbolicKNOp`，带 `OpArgs`，`symbolic_op.h:10-36`）、`tensors`
  （`SymbolicDTensor`，一串 `SymbolicTensorDim`）、`input_indices`/`output_indices`（**按下标的边**，
  与 `docs/08` §1 的 IR 呼应）、以及一组**维度约束** `conds`（`:83-84`）。
- `SymbolicTBGraph`：多 `grid_dim`（3 个 `SymbolicTensorDim`）、`block_dim`（**固定 128×1×1**，
  `symbolic_graph.cc:17-19`）、`forloop_range`、`reduction_dimx`、以及 `conds`（`symbolic_graph.h:42-53`）。
- `SymbolicMap`（`symbolic_map.h:12-28`）：张量维 → 网格轴的**映射矩阵** `map_mat`。默认构造给
  每个 `(grid轴, 张量维)` 对分配一个**独立的布尔变量**（`symbolic_map.cc:6-16`），于是 `add_input`
  用 `dim_expr_make_ite(map_var, device_dim, 1)` 把「该张量维是否切到该网格轴」写成符号表达式
  （`symbolic_graph.cc:336-350`）。

### 3.3 约束系统（两套，机制不同）

约束类型 `TensorDimConstraint` 有 `EQUAL`/`EQUAL_OR_ONE`/`GEQ`/`LEQ`（`tensor_dim_constraint.h:11-16`），
`to_z3` 直接映射（`tensor_dim_constraint.cc:24-47`）。`add_operator` 对每个算子**按语义**生成约束，
例如：

- matmul：前 `num_dims-2` 维必须为 1（批量维广播），最后一维等于另一操作数的倒数第二维
  （`symbolic_graph.cc:209-224`）；
- add/mul/div：逐维 `EQUAL_OR_ONE`（允许广播，`:152-161`）；
- concat：非拼接维必须相等（`:97-107`）；
- `add_input`：映射矩阵的每一行/列都是 0/1 且行和为 1（`make_sum_geq_zero`/`make_sum_leq_one`，
  `:308-330`）。

**关键落差**：约束的**可满足性判定有两套完全不同的引擎**——

- `SymbolicTBGraph` 用 `TensorDimConstraints`（`tensor_dim_constraints.h:55-72`），它是一个
  **可回退集合** `RevertableSet`（`:11-53`，`revert()` 实现搜索回溯），`satisfiable()` 用
  **z3 求解器** `s.check() != z3::unsat`（`tensor_dim_constraints.cc:38-47`）——真正的 SMT 判定；
- `SymbolicKNGraph` 的 `conds` 只是 `std::unordered_set<TensorDimConstraint>`（`symbolic_graph.h:84`），
  它的 `add_conds_for_new_op` 调 `check_satisfiability`（`tensor_dim_constraint.cc:88-148`），
  而后者**根本没用 z3**：注释写着 `// rule-based checking for now`（`:125`），只对常量相等和
  `const / var` 形式的除法做一条**启发式** `probably_equal`，其余**一律返回 `true`**（`:148`）。

也就是说：**TB 级约束是精确（z3）的，KN 级约束是启发式（几乎不拒绝）的**。这是一个「上层精确、
下层宽松」的不对称实现，值得在复现时注意。（另注：`TensorDimConstraints::satisfiable()` 每加一条
约束就把整组约束 `std::cerr` 打印一遍，`tensor_dim_constraints.cc:43`——调试残留，会污染 stderr。）

### 3.4 符号路径的搜索与「未接线」

`generate_next_symbolic_operator`（`search.cc:584-834`）与 `docs/15` 的主循环**同构**：
Case K1 完成并实例化、K2 产生预定义算子、K3 展开一张 `SymbolicTBGraph`、B1 回 kernel 级、B2 产生
TB 算子。区别在与「实例化」：

- 非符号路径 `verify(g)` 会真正跑 `ProbabilisticVerifier`/`FormalVerifier`（`search.cc:521-562`）；
- 符号路径的 Case K1 调 `instantiate_symbolic_graph`（`search.cc:618-621`），而该函数**第一件事就是
  `++num_symbolic_graphs; return false;`**（`search.cc:836-842`）——即**只统计、不实例化**。它后面
  `:844-917` 那套「枚举 grid/forloop 候选 → `DimVarAssignments` 笛卡尔积（`combine_candidate_assignments`
  `:855-871`）→ `to_kernel_graph` → `verify`」的完整实现，**在当前快照里永远走不到**。
- 更彻底的是：**`generate_kernel_graphs_symbolic()` 根本没有任何调用者**。`search_c.cc` 的
  `cython_search` 只调 `gen.generate_kernel_graphs()`（`search_c.cc:106`），全仓库
  `grep generate_kernel_graphs_symbolic` 只命中声明与定义本身
  （`include/mirage/search/search.h:29`、`src/search/search.cc:446`）。

结论：符号化搜索在 `f9eb70c` 快照里是一套**完整写出但未接入**的子系统——与 §2.4 的
`check_range` 禁用、§1.3 的 `check_abstract_expr` 忽略 constraints 一起，构成「研究原型里
备好、生产前端没用上」的又一证据（呼应 `docs/15` §7 的「`mpk` 前端不调 `superoptimize`」）。

## 4. 三块在主搜索里的调用点汇总

| 判定 | 实现 | 非符号路径调用点 | 符号路径调用点 | 当前是否真剪枝 |
|---|---|---|---|---|
| 代数子表达式 | `check_abstract_expr`→`subexpr_to_final_expr`（egg） | `search.cc:190,373` | `search.cc:654,816` | **是**（有 egg 查询） |
| 维度约束联合 | `check_abstract_expr` 的 `constraints` 参数 | — | 声明有、**体未用**（`search.cc:504-519`） | **否** |
| 区间范围 | `check_range`→`range_propagate` | `search.cc:338` | 未调用 | **否**（`return true`，`irange.cc:1068`） |
| 算子序（order）| `check_order`/`get_max_op_order`（`search.cc:66-102`） | `search.cc:185,216,368` | `search.cc:638-641,680-683,798-800` | **是**（单调序剪枝） |
| 形状约束（TB）| `TensorDimConstraints`（z3） | 仅 via `SELECTED` 常量路径 | `SymbolicTBGraph::add_*` | 符号路径（但整条路径未接线） |

补充：`get_init_ranges` 的随机点用 `rand()` 但对应行被注释（`irange.cc:1101-1107`），当前只用全零点；
`Order` 剪枝要求每步生成的操作数下标严格递增（`search.cc:185-187`），这在 `docs/15` §4 已述，
与本文三块正交。

## 5. 与论文的对照

| 论文（快照论文） | 源码落点 | 一致性 |
|---|---|---|
| §4.1「按输出张量分区成 task，task 计算不相交子集」 | 主循环 K3/B1（`docs/15` §3）+ 本节 §3 的符号图 `add_output` 等价 | 一致 |
| §4.1 依赖分析「输出 tile 重叠即插 event」 | 非本文范围（`docs/08`） | — |
| §4.2「thread-block 级超优化，用 Mirage superoptimizer 搜 thread-block 图」 | 主循环 + `transpile`（`docs/15` §4/§5）；本文补其**搜索判定层** | 一致（判定层即本文三块） |
| 「expressions 的代数等价类用于剪枝」 | §1 `abstract_expr` + egg（`abstract_expr.h`/`lib.rs`） | 与论文精神一致，实现用现成 e-graph 库 |
| 「符号化搜索一次枚举多个形状」 | §3 `symbolic_graph`（含 z3 约束） | 论文未细述；源码已有但**未接线**（§3.4） |

**判据映射（沿用 `docs/00` §3.2）**：本文三块与 `docs/14`/`docs/15` 一样，**只贡献 C3（多算子范围
来源）**——它们决定「融哪些算子、融成什么形状」，不碰 C1（单次 launch）/C2（常驻）/C4（设备内协调）/
C5（SM 级调度），后四者由运行层保证（`docs/06`/`docs/07`）。

## 6. 局限

- **纯源码走读，未构建/运行**：e-graph 饱和、z3 求解、符号搜索都需要在 GPU/编译上实跑才出数；
  所有结论基于 `f9eb70c` 快照行号，未在本机执行。
- **未量化剪枝收益**：本文只说「哪一步会剪、当前哪些被禁用」，未测「开了 range 剪枝后搜索状态数降多少」
  （论文 §6.7 只报了 event fusion / linearization 的消融，未单独报这三块）。
- **egg 规则集未穷举**：`lib.rs:36-110` 的规则集本文按组归纳，未逐条证明其**可靠性/完备性**
  （如 `sum-mul` 与 `sum-mul-2` 是否覆盖所有换序）。
- **z3 与 rule-based 的等价性未证**：§3.3 的两套约束引擎（TB z3 / KN 启发式）本文只指出差异，
  未构造「启发式漏判导致非法图通过」的具体反例。
- **`docs/15` 的其它尾巴未覆盖**：transpiler 的 SASS 级生成、`benchmark/` 性能数、多 GPU/NVSHMEM
  路径仍留待后续（见 `docs/15` §9）。

## 参考坐标

- 仓库：`https://github.com/mirage-project/mirage`（分支 `mpk`，commit `f9eb70c`）。
- `src/search/search.cc`（主循环与调用点）、`src/search/search_c.cc`（唯一入口 `cython_search`）。
- `src/search/abstract_expr/`（`abstract_expr.{h,cc}`、`abstract_expr_eval.cc`、`abstract_expr_for_ops.cc`、
  `abstract_subexpr/src/lib.rs`）。
- `src/search/range_propagation/`（`irange.{h,cc}`、`range.{h,cc}`、`range_set.h`、`tbrange.{h,cc}`、
  `propagation_path.h`）。
- `src/search/symbolic_graph/`（`symbolic_graph.{h,cc}`、`tensor_dim_expr.{h,cc}`、
  `tensor_dim_constraint.{h,cc}`、`tensor_dim_constraints.{h,cc}`、`dim_var_assignments.{h,cc}`、
  `symbolic_map.{h,cc}`、`symbolic_tensor_dim.{h,cc}`、`symbolic_op/tensor`、`op_args.{h,cc}`、`types.h`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1/§4.2）。
- 交叉引用：`docs/15`（主搜索循环与 transpile，本文的上游）、`docs/14`（前端，为何不调 superoptimize）、
  `docs/08`（依赖分析器，非本文范围）、`docs/06`/`docs/07`（运行层与单 task device）。
