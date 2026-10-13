# MPK 多卡残差/输出契约走读：`my_gpu_id` 与「TP 下残差只能加一次」（P136-1）

> 定位：本文补掉 `docs/48-mpk-fusion-tasks.md` **§7「局限/下一步」明留的尾巴**（原话）——
>
> > **`my_gpu_id==0` 作 `residual` 的 TP 语义**：从实参静态读出「只有 rank0 加残差」，但**未追
> > `runtime_config.my_gpu_id` 在多卡下的赋值链**（交叉引用 `docs/17` §7 有相关但未逐行对齐本处）。
>
> `docs/17-mpk-multigpu.md` §7 已从**前端**角度讲了「TP 下残差只能加一次」的动机（`enable_residual`
> 参数在 rank≠0 置 0）。本文把同一件事**从设备层的 `runtime_config.my_gpu_id` 往上、往下各追一层**：
> 它**从哪来**（`nvshmem_my_pe()` → launch 期写入）、它在 device **怎么用**、以及「只有 rank0 加残差」
> 这条语义在**三层用三种不同机制**各实现了一遍（前端参数 / 编译期裁剪 / 设备硬编码）。
>
> 快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/49` 同快照），所有行号
> 据此核对。**纯源码走读（资料分析），无新实验**——多卡路径需 ≥2 张 NVLink 卡 + NVSHMEM + MPI，本机
> 不具备（与 `docs/17` 同）；本文与 `docs/17` 互补、不重复其通信机制。

---

## 0. 三条 headline

1. **`my_gpu_id` 有两个互不相通的来源，别混用**：(a) **运行期** `runtime_config.my_gpu_id`，在
   `launch_persistent_kernel` 里由 `mype = nvshmem_my_pe()` 写入（`persistent_kernel.cuh:1727/1793`），
   device task 读到它；(b) **编译期** `params`，由前端 `self.mpi_rank` 烘进生成的 CUDA 文本
   （AllReduce 的 `register_reduction_task`，`task_register.cc:1009-1010/1054`）。**残差走的是 (a)、
   AllReduce 的 rank 走的是 (b)**——两套来源理论上应一致（MPI world == NVSHMEM world），但代码里没有
   断言把它们钉在一起。
2. **「只有 rank0 加残差」在三层用三种机制实现**：前端 `enable_residual`（`persistent_kernel.py:2472-2474`，
   SM100/Hopper 发射器读 `params[0]`，如 `task_register.cc:1596/2025/3556`）；Ampere
   `linear_with_residual`/`silu_mul_linear_with_residual` **丢弃前端参数**（`assert(params.size()==0)`）
   改**硬编码** `runtime_config.my_gpu_id == 0`（`task_register.cc:866/918`）；MoE 输出用
   `add_residual_once` 在前端**选**「只加一次」还是「每 rank 加自己那份」（`:2360`）。
3. **两种残余量约定的差别不在 device、在前端实参**：`add_residual_once=True`（skip connection，
   每 rank 同值）→ 只有 rank0 加；`add_residual_once=False`（row-parallel shared-expert 部分和）→
   **每个 rank 都加自己那份**（DeepSeek-V3 MoE，`models/deepseek_v3/builder.py:1158/1646`）。device 只认
   「指针为空就跳过残差」这一条机械约定（Ampere `linear.cuh:102-103`、Hopper `linear_hopper.cuh:47`、
   SM100 `mul_sum_add_sm100.cuh:36`），约定本身完全落在前端。

判据口径同 `docs/17`：本文只贡献 **C3（多算子范围里「残差/输出契约」这一格）与 C4（跨卡 event 的
owner 约束）**，不改变 C1/C2/C5。

---

## 1. `my_gpu_id` 的赋值链（运行期）

### 1.1 声明与来源

`RuntimeConfig` 里只有一处 `my_gpu_id`（`runtime_header.h:340`）：

```cpp
int num_gpus, my_gpu_id;
```

它**不是**前端传进来的 rank，而是在 `launch_persistent_kernel` 里由 NVSHMEM 的 PE 号得到
（`persistent_kernel.cuh`）：

- `:1724-1727`：`#ifdef USE_NVSHMEM` 下 `nvshmemx_init_attr(...WITH_MPI_COMM...)`、
  `int mype = nvshmem_my_pe(); int npes = nvshmem_n_pes();`；
- `:1766-1768`：`#else`（单卡）分支里 `int mype = 0; int npes = 1;`；
- `:1791-1793`：`global_runtime_config.num_gpus = npes; global_runtime_config.my_gpu_id = mype;`。

即**单卡自洽**：不用 NVSHMEM 时 `my_gpu_id` 恒 0，于是 `my_gpu_id == 0` 恒真——这正是 Ampere 硬编码
路径在单卡下仍正确的原因（`docs/09` 在 H100 单卡上跑的 test_mode/MLP 图都走这一档）。

### 1.2 device 侧怎么用它

device 对 `config.my_gpu_id` 的引用分两类（逐处核对 `persistent_kernel.cuh`）：

- **纯 assert / verbose printf**：`:1048/1171/1243/1415/1458/1509` 都是 `[%d][FTCH]/[DONE]/[SCHD]`
  打印的 rank 前缀（`#ifdef MPK_ENABLE_VERBOSE` 内），或 `:1088/1161` 的
  `assert(get_event_gpu_id(event_id) == config.my_gpu_id)`——**本地（非 NVSHMEM）event 的 owner 必须是
  本 rank**，这是 event 编码 `(nvshmem_tag:16 | owner:16 | idx:32)`（`runtime.cc:30-31` 的
  `get_event_id` 把 `my_gpu_id << 32`）在设备侧的自我校验。NVSHMEM event 走另一条分支（`is_nvshmem_event`
  为真时不做该 assert），因为它的 owner 是远端 GPU。
- **残差门控**：只有一处真正影响数值——在 `task_register.cc` 里被代码生成器写进 device 文本的
  `runtime_config.my_gpu_id == 0`（§3.2）。

---

## 2. 为什么 TP 下残差「只能加一次」

TP 的一次层输出（如 attention 的 o_proj、MLP 的 down_proj）在 row-parallel 下是**各 rank 的部分和**，
后面接一次 AllReduce（AllGather + Reduce，`task_register.cc:1009` 的 `register_reduction_task`；策略 B 的
NVLS tile 见 `docs/17` §6）把它**求和**成全量输出。于是：

- 若残差（skip connection）在**每个 rank 上持有相同值**，而**每个 rank 都在自己的部分和上加它**，则
  AllReduce 求和后这段残差被计了 `world_size` 次 → **残差被放大 world_size 倍**；
- 反之若残差本身是**每 rank 的部分和**（row-parallel shared-expert 输出），则必须**每个 rank 都加自己
  那份**，否则其它 rank 的贡献丢失。

`persistent_kernel.py:2350-2358` 的原注释把这两种情形写成一个二选一（`add_residual_once` True/False）。
本文 §3 说的是：这个二选一在**前端**决定，而 device 只提供「加 / 不加（指针空）」这一个机械开关。

---

## 3. 同一语义、三层、三种机制

### 3.1 前端参数 `enable_residual`（SM100/Hopper）

`linear_with_residual_layer`（`persistent_kernel.py:2449`）构造 `params`（`:2471-2474`）：

```python
enable_residual = 1
if self.world_size > 1 and self.mpi_rank != 0:
    enable_residual = 0
params.append(enable_residual)
```

再按 `target_cc` 分派（`:2477-2487`）：`>=100` 发 `linear_with_residual_sm100`、`>=90` 发
`linear_swapAB_with_residual_hopper`（**两者都带 `params`**）、`80..90` 发
`linear_with_residual`（**不带 `params`**，见 §3.2）。发射器读 `params[0]`：

- Hopper：`register_linear_swapAB_hopper_task`，`task_register.cc:1593-1596`
  `rank_with_residual = (params[0] == 1);`
- SM100 线性：`register_linear_sm100_task`，`:2021-2025` 同式（并被 `:2149/2160` 用来在
  `rank_with_residual` 为假时把残差指针写成 `nullptr`、把模板的 `skip_residual` 置 `true`）。

`moe_mul_sum_add_layer`（`:2325`）也走这条参数路：`register_moe_mul_sum_add_sm100_task`
（`task_register.cc:3554-3559`）`rank_with_residual = (params[0] == 1)`，`:3601` 把假值发射成 `nullptr`。
它的 `add_residual_once` 默认 `True`；DeepSeek-V3 的 shared-expert 路显式传 `False`
（`models/deepseek_v3/builder.py:1158/1646`），于是 `enable_residual` 在**所有 rank** 都为 1 →
**每个 rank 都加自己的部分和**。这就是「两约定」在前端的唯一实现点。

### 3.2 设备硬编码 `runtime_config.my_gpu_id == 0`（Ampere）

`linear_with_residual`（frontend `:2484` 不带 params）→ `graph.cc:509-513` → `register_linear_task(...,
with_residual=true)`（`task_register.cc:871`）。该函数**开头就 `assert(params.size() == 0)`**，然后直接
发射（`:918`）：

```cpp
code.e("    runtime_config.my_gpu_id == 0);");
```

`silu_mul_linear_with_residual`（`graph.cc:522-525`）同构：`register_silu_mul_linear_with_residual_task`
（`:826`）`assert(params.size() == 0)`，`:866` 发 `runtime_config.my_gpu_id == 0);`。

**后果（值得记的设计事实）**：Ampere 这两个融合/线性 task **只能表达 `add_residual_once=True` 那一种
约定**——它没有 `params` 通道，rank 门控被焊死成「只有 `my_gpu_id==0` 加」。因此需要「每 rank 加部分和」
的 MoE 输出在 `target_cc<100` 上**不能**落到这两个 task（DeepSeek-V3 的 MoE 走 SM100 专属的
`moe_mul_sum_add_sm100`，见 `docs/24`/`docs/34`）。（本专题只记这条静态事实与原因，不判它是否为 bug。）

### 3.3 三层的分工表

| 层 | 机制 | 表达「只加一次」的手段 | 能否表达「每 rank 加部分和」 |
|---|---|---|---|
| 前端 | `enable_residual`（`params[0]`） | rank≠0 时置 0 | **能**（保持 1，如 `add_residual_once=False`） |
| 前端 | `add_residual_once` | True → 上面那行生效 | False → 跳过置 0 |
| 编译期 | `register_*_task` 读 `params[0]` | `rank_with_residual` 决定发 `nullptr` 还是指针 | 能（SM100/Hopper） |
| 编译期 | `register_linear_task`/`register_silu_mul_linear_with_residual_task` | **硬编码** `my_gpu_id==0` | **不能**（无 params 通道） |
| 设备 | `residual_ptr == nullptr` / 模板 `skip_residual` | 机械地「空则跳过」 | 由指针是否为空决定 |

---

## 4. 各代 device 残差契约（指针语义 + 门控）

device 端不读 rank，只认「残差指针是否为 null」和（Ampere/Hopper 的）一个运行期 `bool residual`：

- **Ampere `linear.cuh:44-49`**：签名末位 `bool residual`；`:99-103`
  `d_residual = static_cast<T const*>(residual_ptr); if (residual_ptr == nullptr) assert(!residual);`
  ——**契约**：指针为空时 `residual` 必须为假（`task_register.cc:918` 在 rank≠0 时正是「指针非空 +
  `residual=false`」，故 assert 成立，残差被跳过）。`:114` 起 `ResidualDmem` 用 `d_residual`，
  `:194-195` 在 `residual` 为真时预载进 output SMEM。
- **Ampere `silu_mul_linear.cuh:40-42`**：`void const *residual_ptr, ... bool residual = true`；
  `:83-84/149-150/184` 同构（`residual` 为假时 `shared_residual = nullptr`）。
- **Hopper `linear_hopper.cuh:47`**：残差走 TMA，形参是 `const TMA_RESIDUAL *tma_residual = nullptr`；
  `:188-207` 建 `residual_barrier`/`residual_done` mbarrier，`:221-222` 在 `residual` 路径里发
  `tma_residual`。rank≠0 时发射器给的是 `nullptr`（§3.1），故 TMA 残差分支不进。
- **SM100 `mul_sum_add_sm100.cuh:27-37`**：`void const *residual_ptr`，`d_residual` 为 null 时
  `sum_val` 直接取当前值不加残差——`register_moe_mul_sum_add_sm100_task`（`:3601`）在
  `rank_with_residual=false` 时发射 `nullptr`，即**用空指针表达「这个 rank 不加」**（对上 `docs/17` §7
  「SM100 kernel 指针为空就跳过残差加」）。SM100 线性 task 同理（`:2149` 发 `nullptr`、`:2160` 发
  `skip_residual=true` 模板参数）。

**统一口径**：三代 device 的残差契约都是「**空指针 = 不加**」，与 `docs/17` §7 的描述一致；差别只在
「谁来置空」——Hopper/SM100 由前端参数置空，Ampere 由硬编码 `my_gpu_id==0` 决定 `residual` 布尔。

---

## 5. 编译期 `my_gpu_id`：AllReduce 的 rank 走另一条通道

与残差不同，AllReduce 的 rank 不是运行期读 `runtime_config`，而是**编译期烘进 kernel 文本**：

- 前端 `allreduce_layer`（`persistent_kernel.py:2488`）在 `:2500-2508` 注释明写
  `params[0]: num_gpus / params[1]: my_gpu_id`，`params = [self.world_size, self.mpi_rank]`；
- 发射器 `register_reduction_task`（`task_register.cc:1006-1010`）`assert(params.size()==2)`，把
  `params[0]`/`params[1]` 作为**模板实参**写给 `kernel::reduction_kernel<bfloat16, $, $, ...>`
  （`:1050-1054`）。

**含义**：残差的 rank 门控在**运行期**（`runtime_config.my_gpu_id`，由 NVSHMEM PE 决定），AllReduce 的
rank 路由在**编译期**（`self.mpi_rank`）。二者在正常 MPI+NVSHMEM 同 world 下一致，但代码里**没有断言**
绑定 `nvshmem_my_pe() == self.mpi_rank`——若某天 world 不一致（如只对子集建 NVSHMEM team），残差门控与
通信路由会各按各的 rank。这是 `docs/17` §7 未覆盖的一处口径分离。

此外，`my_gpu_id` 还出现在 NVSHMEM 张量的 rank 切片偏移（`runtime.cc:1643`）：
`TASK_NVSHMEM_ALLGATHER_STRIDED_PUT` 的输出偏移 `offset = my_gpu_id * input_ops[0]->dtensor.num_elements()`
——把 `(world_size, batch, hidden)` 的 buffer 按 rank 取第 `my_gpu_id` 片。这里用的是**编译期** `my_gpu_id`
（`Graph::generate_task_graph(num_gpus, my_gpu_id)`，`graph.h:186` → `runtime.cc:1915-1949` 的 `my_gpu_id`
成员），与运行期 `runtime_config.my_gpu_id` **同名不同物**。

---

## 6. 判据映射

| 判据 | 本文贡献 |
|---|---|
| C1 单次 launch | 否（本文不涉及 launch 机制，`docs/06`/`docs/17` 已述） |
| C2 常驻 | 否 |
| C3 多算子范围 | **是**——把「残差/输出契约」这一格讲全：一张 TP 图里，线性/融合 task 的残差、MoE 输出、AllReduce 的 rank 路由各自怎么编，边界在哪 |
| C4 设备内协调 | **间接**——`get_event_gpu_id(eid)==config.my_gpu_id` 的 owner 约束（`persistent_kernel.cuh:1088/1161`）是跨卡 event 协调在设备侧的自我校验 |
| C5 SM 级调度 | 否 |

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建、未运行多卡**：多卡需 ≥2 张 NVLink 卡 + NVSHMEM + MPI，本机不具备（同
  `docs/17`）；所有结论为快照 `f9eb70c` 静态事实。
- **`nvshmem_my_pe() == self.mpi_rank` 的一致性只作「应成立」推断**：未在任何多卡环境核；本文只记
  「代码里没有断言钉住它们」这一静态事实。
- **`runtime_config.my_gpu_id` 在 `-G`/offline/test_mode 各构建下的取值未逐一核**：单卡下恒 0 已由
  `:1766-1768` 的 `#else` 分支支撑，多卡路径未跑。
- **Ampere 硬编码路径「不能表达每 rank 加部分和」是静态推断**：未构造「Ampere + row-parallel 部分和
  残差」的用例去验它是否真的在别处以别的 task 绕过（本文只指出这两个 task 无 params 通道）。
- **未追 `enable_residual` 与 `add_residual_once` 之外是否还有第三、第四种残差约定**（如 bias-as-residual
  `linear_with_bias_sm100`，`graph.cc:635-640` 把 bias 当 row stride=0 的残差——本文只在 §3.3 表里点到）。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/49` 同快照）。
- 运行期赋值：`include/mirage/persistent_kernel/persistent_kernel.cuh:1724-1727`（`nvshmem_my_pe`）、
  `:1766-1768`（单卡 `mype=0`）、`:1791-1793`（`num_gpus/my_gpu_id=`）；owner assert `:1088/1161`；
  verbose printf `:1048/1171/1243/1415/1458/1509`。
- 结构声明：`include/mirage/persistent_kernel/runtime_header.h:340`（`int num_gpus, my_gpu_id;`）。
- event 编码：`src/kernel/runtime.cc:30-31`（`get_event_id` 的 `my_gpu_id<<32`）；NVSHMEM 张量切片
  `:1643`；`Graph::generate_task_graph` `:1915-1949`、声明 `include/mirage/kernel/graph.h:186`。
- 前端：`python/mirage/mpk/persistent_kernel.py`
  （`add_residual_once:2333`、注释 `:2350-2358`、`:2360` rank 门控、`:2361`；
  `linear_with_residual_layer:2449`、`enable_residual:2471-2474`、cc 分派 `:2477-2487`（`:2484` 无 params）；
  `allreduce_layer:2488`、`params=[world_size,mpi_rank]:2500-2508`；
  `silu_mul_linear_with_residual_layer:2570`、`:2589` 注册（无 params））。
- 编译层：`src/kernel/graph.cc:505-513`（`linear`/`linear_with_residual`）、`:522-525`
  （`silu_mul_linear_with_residual`）、`:635-640`（`linear_with_bias_sm100`）；
  `src/kernel/task_register.cc`：`register_silu_mul_linear_with_residual_task:826`（`:866` 硬编码）、
  `register_linear_task:871`（`:918` 硬编码）、`register_linear_swapAB_hopper_task:1588`（`:1596` 参数）、
  `register_linear_sm100_task:2013`（`:2021-2026`、`:2149/2160` nullptr/skip）、
  `register_reduction_task:1006`（`:1009-1010/1054` 编译期 params）、
  `register_moe_mul_sum_add_sm100_task:3554`（`:3556-3559/3601`）。
- device：`tasks/ampere/linear.cuh:44-49/99-103/114/194-195`；
  `tasks/ampere/silu_mul_linear.cuh:40-42/83-84/149-150/184`；
  `tasks/hopper/linear_hopper.cuh:47/188-207/221-222`；
  `tasks/blackwell/mul_sum_add_sm100.cuh:27-37`。
- builder：`models/deepseek_v3/builder.py:1158/1646`（`add_residual_once=False`）、
  `models/qwen3/builder.py:425/540`、`models/glm4_moe/builder.py:215/242/269`、
  `models/gpt_oss/builder.py:317`、`models/eagle3/builder.py:385/407`（均走默认 `add_residual_once=True`）。
- 本专题交叉引用：`docs/17` §7（多卡残差/张量正确性，前端视角）、`docs/48` §7（本文补的尾巴）、
  `docs/49`（四层接线审计——`linear_with_residual`/`silu_mul_linear_with_residual` 是「编不进」孤儿）、
  `docs/24`（`TaskType` 目录）、`docs/34`（`moe_mul_sum_add_sm100` 的 device 布线）。
