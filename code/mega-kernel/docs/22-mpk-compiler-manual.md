# MPK 源码走读手册：从模型到常驻 mega-kernel 的完整编译栈

> 本文是 MPK 线上实现源码走读链条的**收官汇总**，把散落在 `docs/06`（运行层）、`docs/07`（单 task
> device）、`docs/08`（依赖分析器）、`docs/14`（前端）、`docs/15`（superoptimizer）、`docs/16`（搜索器
> 辅助模块）、`docs/17`（多卡）、`docs/18`（transpiler codegen）、`docs/19`（runtime 模板库）、
> `docs/20`（Triton 后端）、`docs/21`（Blackwell 后端）**十一篇**走读收口成一页「编译栈总图 + 决策表
> + 落差清单」。
>
> 定位与 `docs/11`（in-kernel scheduler 设计手册）、`docs/12`（共驻成本手册）、`docs/13`（端到端数据面
> 手册）并列：那三本收口的是 `docs/04` 的**实验**，本文收口的是**源码走读**。**纯资料汇总，无新实验**；
> 所有行号与文件均引自上述各篇（同一快照）。读者若想按需深挖，本文每节都给出对应分篇。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，CuTe/CUTLASS 用快照自带 `deps/cutlass`（本机
> `f3fde583`）。论文侧证据用 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
> （「快照论文」）。**全部为纯源码走读，未构建/运行**（多卡/B200/Triton 路径本机不具备），故本文
> 不引用任何性能数。

## 0. 一句话：MPK 源码里其实有「两条链」，共用一张图 IR，下游完全不同

走读十一篇后最该记住的一条结构性事实：**同一个仓库里同时跑着两条「模型 → kernel」的路**，它们在
`KNGraph`（一张 `KNCustomizedOp` 序列，每个 op 带一张 bGraph）处**分叉**，之后各走各的：

```
                                     ┌─ 生产路径（mpk 分支 LLM serving 用） ─────────────────────────┐
模型 ──► KNGraph ──►（分叉点）──────┤  build_annotated_graph → tGraph → runtime.cc 生成 host 代码    │
         （前端的                                                      → persistent_kernel.cuh    │
          两种建图法）          │                                        （单次常驻 mega-kernel，C1–C5）│
                                 └─ 自动路径（研究/测试/benchmark 用） ──────────────────────────┤
                                    superoptimize 搜索 → transpiler 生成 custom_kernel_N           │
                                    → NVCC 或 Triton（kernel-per-operator，只到 C3）              │
                                                                                                  └
```

- **生产路径**：`docs/14` §0 的核心洞察——`mpk` 前端**不是**「高层算子自动分解」的编译器，而是一套
  **手工作者 API**（每层手写 bGraph + `register_task`）。它经 `docs/08` 的依赖分析器变成 tGraph，
  再由 `docs/06` 的运行层编成**唯一一次 launch 的常驻 kernel**（worker 执行 + scheduler 派发）。
- **自动路径**：`docs/15` 的 Mirage superoptimizer 搜出等价 thread-block 图，`docs/18` 的 transpiler
  把它编成一段可 NVCC 编译的 CUDA（`docs/19` 是被这段代码调用的运行期模板库；`docs/20` 是同一台
  superoptimizer 的 Triton 后端；`docs/21` 是它的 Blackwell 版）。**这条路口产出的是 kernel-per-operator
  的 `.so`，不是常驻 megakernel。**

两条路**在 `KNGraph` 之后共用同一套 `KNCustomizedOp`**（`docs/15` §7：`annotated_graph.cc:278-283` 只认
`KN_CUSTOMIZED_OP`），所以「前端可替换、下游分叉」——这是全篇最重要的一张分层证据。

## 1. 编译栈总图：一张坐标表

| 层 | 产物 | 主文件（`f9eb70c`） | 生产路径 | 自动路径 | 分篇 |
|---|---|---|---|---|---|
| ① 前端/算子层 | `KNGraph`（`KNCustomizedOp` + bGraph + `task_config`） | `python/mirage/mpk/*.py`、`python/mirage/{kernel,threadblock}.py` | 手工 `*_layer()` | `superoptimize()` | `docs/14` |
| ①′ 搜索/判定层 | 等价 thread-block 图 | `src/search/`（`search.cc` + `abstract_expr/`+`range_propagation/`+`symbolic_graph/`） | 不用 | DFS 搜索 + 三判断器 | `docs/15`/`docs/16` |
| ② 图 IR 构建 | `AnnotatedGraph`（layer/edge/fork-join/`ordered_layers`） | `src/kernel/annotated_graph.cc`（1014 行） | **用** | 不用（走 transpiler） | `docs/08` |
| ②′ codegen | `custom_kernel_N` + `_execute_mugraph` 的 `.cu`/`.py` | `src/transpiler/*`；`src/triton_transpiler/*` | 不用 | **用** | `docs/18`/`docs/20`/`docs/21` |
| ③ 运行层 | `persistent_kernel.cuh`：worker/scheduler 设备循环 | `include/mirage/persistent_kernel/*`（主体 2115 行） | **用** | 不用 | `docs/06` |
| ④ 单 task device | `tasks/<arch>/*.cuh` 模板实例化（一个 task = 一个 CTA） | `src/kernel/task_register.cc`、`tasks/` | **用** | runtime 模板库（`docs/19`） | `docs/07` |
| ⑤ 多卡 | 跨卡 event + NVSHMEM task | `python/mirage/mpk/multigpu.py`、`persistent_kernel.cuh` | **用**（每 rank 一套） | 不涉及 | `docs/17` |

论文口径：① 对应 §4.1 operator decomposition、②′ 对应 §4.2 task implementation generation、②③④ 共同
对应论文的两层结构（tGraph + in-kernel 运行时，`docs/01` §3/§4）。**判据 C1–C5 只在 ③ 运行层落地**，
其余各层都只贡献 C3 的「范围/粒度」（见 §8）。

## 2. 生产路径逐层（这篇才是 megakernel 的所在）

### 2.1 前端（`docs/14`）：手工作者 API 定下「要融什么」

- 三层 Python 对象：`KNGraph`（`python/mirage/kernel.py:208`）、`TBGraph`（`python/mirage/threadblock.py:4`）、
  `DTensor`/`STensor`。`KNGraph` 同时提供**高层算子 API**（供 superoptimizer 用）与**手工 task API**
  （`customized()` `:283` + `register_task()` `:802`，供生产用）。
- 用户面对 `PersistentKernel`（`python/mirage/mpk/persistent_kernel.py:413`），每个算子一个 `*_layer()`
  **手写 bGraph** 并登记 `TaskType`；95 个手工分支登记 95 个 TaskType（`docs/14` §3）。
- 粒度结论：**一个算子 = 一个 `customized` = 一种 TaskType**（不是「一个元素级算子 = 一个 task 实例」，
  后者由 `grid_dim` 在运行层展开）。前端把 C3 的**范围**定下来。

### 2.2 依赖分析器（`docs/08`）：论文 §4.1 的五步在此变成一个 1014 行函数

`build_annotated_graph(graph, task_configs)`（`runtime.cc:243`）的十步 (a)–(j)，与论文五步的映射：

| 论文 §4.1 步骤 | 实现（`annotated_graph.cc`） | 与论文的差异 |
|---|---|---|
| 算子分解 | 不在本文件：由 `task_configs` + bGraph `grid_dim` 提供 | 分解在**前端**（手工）或**搜索器**（自动），不在依赖分析器 |
| 依赖分析 | Step (a) **最近写者 SSA** + bbox `intersect/subtract` **阴影覆盖**（`:119-425`） | 额外解决 **buffer 复用/view**（论文未提；reshape 回退整父张量，宁多报不漏报） |
| event fusion | Step (g) 逐边 **GCD** + (h) **fork LCM** + (i) **join LCM**（`:747-958`） | 论文用集合论 successor/predecessor-set fusion；实现用**分块因子 GCD/LCM**，目标一致 |
| normalization | Step (e) **case 2/3 拒绝** + barrier 边 `event_dim=1` 坍缩（`:583-662`） | 论文用 Figure 6 **空-task 重写**；实现是「**拒绝非法角色组合 + LCM 降级**」，不是显式重写 |
| linearization | Step (f) **两趟 Kahn**（`depth` 小者优先、同 `depth` 按插入序）（`:664-745`） | 论文 Algorithm 1 的落地；「每 event 的 task 连续」靠 emit 顺序 + fork 交错 |

另有论文未列的防御步：Step (b) 环检测、(c) **残余边剥离 = 传递规约**（把残差跳跃连接规约成纯链）。
硬约束：`FullTaskDesc` **只有一个 `trigger_event` 槽 + 一个 `dependent_event` 槽**，故 case 2/3 非法
组合**编译期报错**——这是写新模型的人的真实边界。

### 2.3 运行层（`docs/06`）：唯一一次 launch 的常驻 kernel（C1–C5 全在此）

结构：`compile()` 把图固化成宿主初始化代码 + `_execute_task` 派发代码 → `init_persistent_kernel()` 把
队列/事件/task 全部 `cudaMalloc` 到显存 → `launch_persistent_kernel()` 起**唯一一次**驻留 kernel。

- **worker**（`execute_worker`，`persistent_kernel.cuh:957-1252`）：一 worker = 一 block，`while(true)`
  循环——拉取 task（128b `cp.async` 批量搬 `TaskDesc` 进 SMEM）→ 自旋等事件计数达阈值
  （`all_event_num_triggers[event]*iteration`，含**迭代代次**防跨代串扰）→ `_execute_task` → 触发后继
  （`atom.add.release` 计数 + CAS-release 推 scheduler 队列游标）。
- **scheduler**（`execute_scheduler`，`:1254-1523`）：一 scheduler = 一 warp，一 block 最多 4 warp。按事件
  三档派发 task 到 worker 队列：`EVENT_LAUNCH_TASKS`（普通，按 `get_rand_sched_id`）/`_MASSIVE_`（跨
  local scheduler 切区间）/`_DEPENDENT_`（单 scheduler 内 round-robin）；`EVENT_END_OF_TASK_GRAPH` 触发
  `prepare_next_batch` 连续服务下一 batch（不重启 kernel）。
- **SM 划分**：`python/mirage/utils.py:25-29` —— `scheduler = 4*(sm_cnt - worker)`，即每 scheduler SM 放
  4 个 scheduler warp；目标卡 H100 132 SM → **128 worker + 16 scheduler（4 SM）**，对上论文 §5.1。
- **内存序**（`mpk_atoms.cuh`，97 行）：手写 PTX release/acquire 原语；队列统一协议
  「relaxed 写数据 + release 推游标 / acquire 读游标 + relaxed 读数据」（本专题 `docs/04` §36–§37 在
  H100 上把这条协议单独实测过）。
- 可选 `split_worker_scheduler`（`:1977`）：既支持「单 kernel 合并」也支持「worker/scheduler 双 kernel
  分 stream」（后者更接近 NVSHMEM 路径）。

### 2.4 单 task device 实现（`docs/07`）：一个 task = 一个 CTA

编译层 `task_register.cc:1181-1318` 把形状映射成对 `tasks/<arch>/*.cuh` **模板函数**的实例化调用
（不是凭空生成 PTX）。以 `linear_kernel_hopper`（`linear_hopper.cuh`，364 行）为例：

- **warp specialization**：consumer warpgroup 跑 wgmma、producer warpgroup（仅 warp0 lane0）发 TMA；
  `Kstages = output_size>=256?3:6`；SMEM 手工预算 + `smem_tma` 128B swizzle + `mma_descriptor`；
  **残差在寄存器内融合**（`HAS_RESIDUAL` 编译期开关）——损失三（中间张量 HBM 往返）在单 task 内被消掉。
- **异构 task 形态**：`rmsnorm_hopper`（cp.async 双缓冲 + 两级归约）、`silu_mul_hopper`（纯逐元素、
  连 SMEM 都不用）、`allreduce.cuh`（NVSHMEM 通信 task）、`norm_linear_hopper`（段内多算子融合）。
  它们共享同一个契约：**一个 CTA、共享同一份动态 SMEM 预算、以 `_execute_task` 被调用**。
- **两套同步机制**：task 内同步 = mbarrier/`bar.sync`；task 间同步 = 设备内原子环形队列 + 事件计数器。
  论文 §5.3 的 cross-task software pipelining 由**运行层**负责，不是 task 内部完成的。

## 3. 自动路径逐层

### 3.1 搜索（`docs/15`/`docs/16`）：论文 §4.1 的「自动分解」

- 入口 `KNGraph.superoptimize()`（`kernel.py:543`）→ `cython_search`（`search_c.cc:21`）→
  `KernelGraphGenerator::generate_next_operator`（`search.cc:129-392`，K1/K2/K3 + B1/B2 三层 DFS）→
  `ProbabilisticVerifier` 指纹验证 → `transpile()`；最后扫 `pipeline_stages × num_warp_groups` 并
  profile 选优（`kernel.py:606-692`）。**「搜索」与「生成」解耦**：搜索只产出等价图，CUDA 到 codegen 才生成。
- 主循环把「候选是否值得试」委托给三个**正交判断器**（`docs/16`）：`abstract_expr/`（egg e-graph 代数
  等价类，**真剪枝**）、`range_propagation/`（区间推导）、`symbolic_graph/`（维度符号化搜索）。

### 3.2 codegen（`docs/18`/`docs/19`/`docs/20`/`docs/21`）：一台 superoptimizer 的四个后端

- **CUDA/NVCC**（`docs/18`）：`KNGraph.compile()` → `transpile()` → `Transpiler` ctor（两段图重写：
  online-softmax 重写 + 复合 TB 算子拆原语）→ 五个 resolve（元数据/融合/布局/显存，显存**只分配不复用**）
  → `transpile_ugraph()` 逐 op 展开 → 每个 customized op 一个 `custom_kernel_N`（含 schedule/swizzle/
  memory plan 三件套、192 寄存器预算决定 accum 是否进寄存器、Hopper warp-specialized producer-TMA/
  consumer-wgmma 流水）。
- **runtime 模板库**（`docs/19`）：被生成 CUDA 调用的那层库（`include/mirage/transpiler/runtime/**`，
  **23 头、3017 行**），两层 namespace——`kn::`（整算子兜底，GEMM 直接调 **cuBLAS**）/`tb::`（在
  `custom_kernel_N` 内按 thread 调用的拷贝原子/epilogue 链/mbarrier 流水/wgmma·UMMA）。
- **Triton 后端**（`docs/20`）：`backend="triton"` → 每候选图独立子进程 profile → `TritonTranspiler`
  （`src/triton_transpiler/transpile.cc` 579 + `transpile_tb.cc` 645）→ 生成 Python 模块（`@triton.jit`
  + `execute_mugraph`），运行期 `triton_kernels.py`（66 行）。
- **Blackwell 后端**（`docs/21`）：= Hopper 后端 + 「把计算单元从 1 CTA 扩到 2 CTA」的一整条链——
  2SM TiledMMA → cluster 布局 + leader CTA（`elect_one_cta`）→ multicast TMA（A 沿 N / B 沿 M）→
  **TMEM** 累加（`Allocator2Sm`）→ `tcgen05.commit.cta_group::2...multicast::cluster` 会合 →
  `cutlass::launch_kernel_on_cluster`。

## 4. 多卡（`docs/17`）：C1 从优化变硬约束

MPK **没有为多卡另造一套 runtime**，只把两件东西塞进原有机制：(a) 给跨卡 task 打 `EVENT_NVSHMEM_TAG`
（event 由别的 GPU 触发，event id = `nvshmem_tag:16 | owner:16 | idx:32`）；(b) 把 NVSHMEM collective
当**普通 task** 编进同一张 tGraph，用 `nvshmemx_collective_launch` 启动同一个常驻 kernel。关键：**该
collective launch 不支持 split worker/scheduler**（`persistent_kernel.cuh:1988-1990`）——多卡下
「单次 launch」从可选优化变成**唯一可行路径**。AllReduce 有两种落法：策略 A `AllGather+Reduce`
（论文 §6.5）、策略 B `NVLS tile` one-shot pull（Blackwell 手写 `multimem.ld_reduce` 保共驻，**论文未提**）。

## 5. 「同一套惯用法实现了两遍」清单

走读系列反复撞见同一现象——**一个硬件惯用法在同一仓库里有两份独立实现**，是理解 MPK 工程状态的关键：

| 惯用法 | 实现之一 | 实现之二 | 分篇 |
|---|---|---|---|
| warp-specialized TMA+wgmma+epilogue | MPK 生产 task（`tasks/hopper/*.cuh`，常驻 worker 按 event 调用） | transpiler runtime 模板库（`tb::`，生成的 `custom_kernel_N` 内联调用） | `docs/07` vs `docs/19` §5 |
| Triton codegen | 新 path（`src/triton_transpiler/*`，直读图、`tl.arange`+`program_id`） | 旧 path（`src/kernel/triton_code_gen.cc`，513 行，已成**事实死代码**） | `docs/20` §6 |
| 「自动 vs 手工」task 生成 | superoptimizer 搜索生成 | `task_register.cc` 映射到**手写模板**（含 CUTLASS 封装 variant） | `docs/07` §0 / `docs/15` §7 |

结论（`docs/15` §7 / `docs/18` §7 / `docs/19` §5 一脉）：**编出一堆很快的单算子 kernel 从来不是难点**
（仓库里有两套现成实现）；难的是「把它们**编成一张图、常驻着去中心化调度**」——这后一半只在生产路径
（`docs/06`/`docs/08`）里存在。

## 6. 「已实现但未启用 / 未接线」清单（跨篇汇合）

读代码时最易踩的坑：**「有实现」≠「已启用」**（本专题「已知坑」同条）。集中列一次：

| 位置 | 现象 | 分篇 |
|---|---|---|
| `range_propagation` 两个 `check_range` | 首行 `return true;`（`irange.cc:1068-1069,1085-1086`），后面 ~1100 行区间推理**当前不执行** | `docs/16` |
| 符号搜索 `instantiate_symbolic_graph` | 首行 `++num_symbolic_graphs; return false;`（`search.cc:836-842`），候选枚举/`to_kernel_graph`/`verify` 走不到 | `docs/16` |
| `generate_kernel_graphs_symbolic` | `search.cc:446` **无任何调用者** | `docs/16` |
| `check_abstract_expr` 的 `constraints` 参数 | 签名带 `TensorDimConstraints`，函数体**完全没用**它（`search.cc:504-519`） | `docs/16` |
| `SymbolicKNGraph::conds` 约束引擎 | rule-based 启发式、**其余一律返回 `true`**（`tensor_dim_constraint.cc:88-148`），几乎不拒绝 | `docs/16` |
| `search_c.cc:16` 的 `backend` 参数 | 只在签名里，**函数体从不引用** | `docs/20` |
| `TritonTranspilerConfig.target_cc` | **已接线未使用** | `docs/20` |
| 旧 Triton codegen `Graph::generate_triton_program` | 无 Python 调用者（**事实死代码**） | `docs/20` |
| `parallel_path_design.md`（`docs/08` 注释引用） | 快照里**不存在**（`docs/mpk/` 只有 `tma.md`） | `docs/08` §9 |
| `docs/transpiler/transpiler.md`（`sched_tb_graph.cc` 注释引用） | 快照里**未随仓库提供** | `docs/18` §8 |
| `wg_decrease/increase_regs`（`setmaxnreg`） | Hopper/Blackwell task 与 runtime 里**均被注释** | `docs/07` §2 / `docs/19` §3 / `docs/21` §1 |
| `mpk` 生产前端**不调** `superoptimize` | grep `models/`/`mpk/` 无命中，只有 `tests/`/`demo/`/`benchmark/` 调 | `docs/15` §7 / `docs/20` §7 |

## 7. 论文描述 vs 线上实现的落差清单

| 论文口径 | 线上实现 | 分篇 |
|---|---|---|
| §4.1「算子分解自动」 | `mpk` 生产前端是**手工** `*_layer()`；自动路径存在但**生产不用** | `docs/14` §0/§5、`docs/15` §7 |
| §4.2「每 task 用 superoptimizer 生成」 | 生产 task 是**手写模板 + CUTLASS 封装**，`TaskRegister` 按 variant 选 | `docs/07` §0 |
| §4.1 normalization＝Figure 6 空-task 重写 | 实现是「**拒绝 case 2/3 + fork/join LCM 降级**」 | `docs/08` §4/§8 |
| §4.1 event fusion＝集合论 successor/predecessor-set | 实现用**分块因子 GCD/LCM**，目标一致、表述不同 | `docs/08` §6 |
| §4.2 只提 CUDA codegen | 仓库里还有**平行的 Triton 后端**（且 Triton 自身两套） | `docs/20` |
| 未描述 Blackwell/SM100 2SM | `docs/21`：**论文未提、快照新增**（同多卡路径） | `docs/17`/`docs/21` |
| §5.2 离线固定 batch | `RuntimeConfig` 内联大量 serving 元数据（step/tokens/paged-KV、`MODE_ONLINE_PINNED`） | `docs/06` §6 |

## 8. 判据 C1–C5 总映射（哪一层贡献哪条）

| 层 | C1 单次 launch | C2 常驻 | C3 多算子范围 | C4 设备内协调 | C5 SM 级调度 |
|---|---|---|---|---|---|
| 前端（`docs/14`/`docs/15`/`docs/16`） | — | — | **是（范围来源）** | — | — |
| 依赖分析（`docs/08`） | — | — | **是（event 图）** | 部分（生成 event，执行在运行层） | — |
| codegen（`docs/18`/`docs/19`/`docs/20`/`docs/21`） | — | — | **是（粒度来源）** | —（cluster 内会合只覆盖单 op） | — |
| 单 task device（`docs/07`） | — | — | **是（task 类型）** | —（mbarrier 只在 CTA 内） | — |
| **运行层（`docs/06`）** | **是** | **是** | 是（`_execute_task` 覆盖全算子） | **是**（原子队列+事件计数器） | **是**（worker/scheduler 去中心化） |
| 多卡（`docs/17`） | **强约束** | 是 | **扩展**（跨卡算子+通信） | **跨卡延伸**（远端触发/ NVLS barrier） | 是（每 rank 独立） |

一句话：**megakernel 的 C1/C2/C4/C5 只在运行层拿到**；前端、搜索、codegen、device task、多卡全部只
在 C3 的「范围/粒度」上加码——**换 codegen 目标语言（CUDA↔Triton）或换硬件后端（Hopper→Blackwell）
都拿不到 C1/C2/C4/C5**（`docs/20` §7 明说）。

## 9. 源码坐标总表（快照 `f9eb70c`）

| 文件/目录 | 行数 | 角色 | 分篇 |
|---|---|---|---|
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | 2115 | worker/scheduler 设备循环、宿主 init/launch | `docs/06` |
| `include/mirage/persistent_kernel/runtime_header.h` | 462 | `TaskDesc`/`EventDesc`/`RuntimeConfig` | `docs/06` |
| `include/mirage/persistent_kernel/mpk_atoms.cuh` | 97 | release/acquire 内存序原语 | `docs/06` |
| `src/kernel/runtime.cc` | 1972 | tGraph → 生成 `_init_persistent_kernel`/`_execute_task` | `docs/06`/`docs/08` |
| `src/kernel/annotated_graph.cc` + 头 | 1014 + 198 | 依赖分析器（十步 (a)–(j)） | `docs/08` |
| `src/kernel/task_register.cc` | — | `(TaskType,variant)` → `tasks/*.cuh` 实例化 | `docs/07` |
| `include/mirage/persistent_kernel/tasks/{hopper,ampere,blackwell,cute}/` | — | 单 task device 实现 | `docs/07` |
| `python/mirage/mpk/persistent_kernel.py` | 3618 | Python 前端：构图/`compile()`/生成 launcher | `docs/14` |
| `python/mirage/mpk/multigpu.py` | — | 多卡能力探测、team 分配、两种 AllReduce | `docs/17` |
| `python/mirage/utils.py` | 53 | SM → worker/scheduler 划分 | `docs/06` |
| `src/search/search.cc` | — | superoptimizer 主循环 | `docs/15` |
| `src/search/{abstract_expr,range_propagation,symbolic_graph}/` | — | 三个正交判断器 | `docs/16` |
| `src/transpiler/transpile.cc` / `transpiler_kn.cc` / `transpiler_tb.cc` | 803 / 883 / 1544 | CUDA codegen | `docs/18` |
| `src/transpiler/transpiler_tb_{hopper,blackwell}.cc` | 1539 / 1575 | Hopper / Blackwell codegen | `docs/18`/`docs/21` |
| `src/transpiler/plan_tb_swizzle_blackwell.cc` | 177 | Blackwell swizzle 计划 | `docs/21` |
| `include/mirage/transpiler/runtime/**` | 23 头 3017 | 被生成代码调用的模板库 | `docs/19` |
| `src/triton_transpiler/{transpile,transpile_tb}.cc` | 579 / 645 | Triton codegen | `docs/20` |
| `include/mirage/triton_transpiler/runtime/triton_kernels.py` | 66 | Triton 运行期 | `docs/20` |
| `src/kernel/triton_code_gen.cc` | 513 | 旧 Triton codegen（事实死代码） | `docs/20` |

## 10. 局限

- **纯资料汇总，无新实验**：本文所有结论均引自 `docs/06`–`docs/21` 各篇（同一快照 `f9eb70c`），未做
  任何本机构建/运行；不引用任何性能数。数字锚点见 `docs/09`（端到端延迟）与 `docs/04`（原生 CUDA
  微基准）。
- **汇总口径随分篇**：各分篇本身即「纯源码走读」，故本文继承它们的共同局限——未构建 mirage、未逐行
  展开 CuTe 模板装配、多卡/B200/Triton 路径本机不可执行、`docs/14` 未逐一走读全部 95 个任务分支、
  `docs/17` 未量化 `allocate_nvshmem_teams` 代价等。
- **不构成对上游的 bug 判定**：§6 的「已实现未启用」仅记录读码事实（多来自 `grep` + 函数体阅读），
  不排除宏/模板间接生效，也不代表上游缺陷。
- **两条链的分叉点判断基于源码**：生产路径（annotated_graph/runtime）与自动路径（transpiler）在
  `KNGraph` 处分叉，依据是 `annotated_graph.cc:278-283` 只认 `KN_CUSTOMIZED_OP` 与 `docs/15` §7 的
  前端调用统计；若上游后续改动分叉点，本文需同步更新。

## 11. 关键参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）；
  CuTe/CUTLASS 用快照自带 `deps/cutlass`（本机 `f3fde583`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
  （§3 行 345–448、§4.1 行 449–1146、§4.2 行 1126–1146、§5 行 1147–1368）。
- 分篇：`docs/06`（运行层）、`docs/07`（单 task device）、`docs/08`（依赖分析器）、`docs/14`（前端）、
  `docs/15`（superoptimizer）、`docs/16`（搜索器辅助模块）、`docs/17`（多卡）、`docs/18`（transpiler）、
  `docs/19`（runtime 模板库）、`docs/20`（Triton 后端）、`docs/21`（Blackwell 后端）。
- 实验侧配套手册：`docs/11`（in-kernel scheduler 设计）、`docs/12`（共驻成本）、`docs/13`（端到端数据面）。
- 原则总结：`docs/05`（一页纸 + 设计 checklist + 适用边界）。
