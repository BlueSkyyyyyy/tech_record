# MPK 运行层「可执行规范」走读：`tests/runtime_python/test_mode/` 与 `PersistentKernel` API

> 本文补齐 `docs/25`（P111-1）**§9 明留的尾巴**：
>
> > **`MirageModelConfig` 的调用方（如 `demo/`/`tests/`）未走读**：本文只看库侧契约。
>
> 前面 `docs/06`–`docs/27` 全部是「往库里看」——编译器、运行层、device task、前端、builder、权重预处理。
> 但**库里这些契约到底对不对、谁来验**？答案在仓库的 **`tests/runtime_python/test_mode/`**：它用
> **同一套 `PersistentKernel` API** 把每个前端/依赖分析器/运行层特性编成一张小图、`pk()` 一次、
> 再对 PyTorch 参考逐位比。**这套测试就是 megakernel 编译栈的「可执行规范」（executable spec）**。
>
> headline——**`test_mode` 把一张常驻 megakernel 图「跑一次就退」**：`-DMPK_TEST_MODE` 同时改
> **编译期**（缺的 meta tensor 允许传空指针）与**设备端终止协议**（请求一迭代即标 `done`，不再等
> `max_seq_length`/EOS），于是每个测试都是一次「单次 launch → 常驻 kernel 自终止 → 拿结果比参考」的
> 最小端到端。这与 `docs/09`（P25-1）用的正是**同一批用例**——P25-1 的「不收敛」根因诊断（launch 顺序）
> 也发生在这一层（见 `docs/09` §7–§11）。
>
> 本文是**纯源码走读（资料分析）**，不构建、不运行 mirage，不引用任何新性能数。快照
> `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/27` 同一快照）。

## 0. 定位：测试套件是编译栈的「可执行规范」

```
                       ┌── docs/06 运行层（worker/scheduler）──┐
docs/08 依赖分析器 ────┤                                      ├── test_mode 用例
docs/14 前端 task API ─┤── docs/24/25/26/27 device / builder ─┘   （同一 PersistentKernel API）
                       └──────────────────────────────────────┘
```

- **库侧契约**（`docs/25`）说「builder 登记了什么 task、把哪些中间量标成 `cuda`/`nvshmem`」；
- **可执行规范**（本文）说「登记出的图**真的跑出正确输出**」——把 `fork`/`join`/`case 2/3`/
  `residual stripping`/`grid` 标度这些**只写在编译器里的规则**，各配一个能对 PyTorch 参考的用例。
- 这条路只有一条 API：`mirage.mpk.persistent_kernel.PersistentKernel`（`docs/25` §2 的 `MPK` 是它的
  生产包装/元数据层；`test_mode` 用例直接用它）。

**一张图怎么被驱动**（所有 test_mode 用例的共同骨架，以 `test_rmsnorm_testmode.py` 为例）：

```
1. num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)   # utils.py:33
2. params = PersistentKernel.get_default_init_parameters()               # :669
   params["test_mode"]=True; params["num_workers"]=…; params["world_size"]=1
3. pk = PersistentKernel(**params)                                       # :433
4. x_dt = pk.attach_input(x, name="x"); …                                # :746
5. pk.rmsnorm_layer(input=x_dt, weight=w_dt, output=o_dt, grid_dim=…, block_dim=…)  # :865
6. pk.compile(output_dir=…)                                              # :3207
7. pk()                                                                  # __call__ :3574 → launch_func
8. torch.cuda.synchronize(); max|out-ref| < tol
9. pk.finalize()                                                         # :3614
```

与生产路径（`demo/qwen3/demo.py` 走 `MPK`/builder，`docs/25`）的区别**只在第 4/5 步**：测试手工逐层
调 `*_layer()`，生产让 builder 按家族拓扑自动调——**下游的 `compile`/launch/终止协议完全相同**。

## 1. `PersistentKernel` 公开 API：从测试反推的契约

测试用例只用到这张面（`python/mirage/mpk/persistent_kernel.py`，3618 行）：

| API | 行号 | 作用 |
|---|---|---|
| `get_default_init_parameters()`（classmethod） | `:669-688` | 返回 16 个默认：`mode="offline"`、`world_size=1`、`num_workers=1`、`num_local_schedulers=4`、`max_seq_length=1`、`max_num_batched_*`=1、`page_size=1`… |
| `__init__(**params)` | `:433` | 建 `PersistentKernel`；`target_cc = major*10+minor`（`:536`）；`test_mode` 分支 `:538-545` |
| `attach_input(torch_tensor, name)` → `DTensor` | `:746` | 登记一个模型张量（名字→指针在 `compile` 时打包给运行层） |
| `*_layer(...)` 手工层 | `:842-3128` | 82 个前端登记函数（`rmsnorm_layer:865`、`linear_layer:2390`、`linear_with_residual_layer:2449`、`paged_attention_layer:1303`、`moe_topk_softmax_routing_layer:1972`、`sampling_partial_layer:2685`…） |
| `compile(output_dir=…)` | `:3207` | 生成/编译 device 代码；`test_mode` 允许缺 meta tensor（`:3413-3415`） |
| `__call__(**kwargs)` | `:3574-3604` | `stream_ptr = self.launch_func(stream_ptr)`；有 profiler 时导出 perfetto/csv |
| `wait()` | `:3610` | 阻塞等两条常驻 stream 退出 |
| `finalize()` | `:3614-3618` | 调 `finalize_func()`（`__del__` 兜底） |
| `target_cc` | `:536` | 决定 `block_dim=(256,1,1)`（≥90）还是 `(128,1,1)`——**测试里唯一的硬件分支** |

**worker/scheduler 划分**由 `python/mirage/utils.py` 给（`docs/06` §SM 划分的同一张表）：

```
get_scheduler(sm_cnt, worker) = 4 * (sm_cnt - worker)     # utils.py:25-29
get_configurations_from_gpu(rank):                        # utils.py:33-53
    sm_cnt>=160→worker=144; >=144→128; >=132→128; >=108→96; >=68→64; >=40→30; else 20
```

即**测试用的 worker/scheduler 数 = 目标卡实际 SM 数**（目标卡 H100/sm90 132 SM → **128 worker / 16
scheduler**），与 `docs/06`/`docs/07` 走读的 128/16 完全对上——**测试没有为小用例缩小 grid**，
这也是 `docs/09` §7 里 plain 构建 3/3 挂起、而加一处无害自旋计数即收敛的物理背景。

## 2. `test_mode`：把「常驻 kernel」变成「跑一次就退」的测试装置

`test_mode` 不是运行期开关，而是**贯穿编译期与设备端的编译宏**：

1. **编译期**：`flags += ["-DMPK_TEST_MODE"]`（`:318-319`，仅当 `test_mode=True`）。
2. **元数据默认**：`_apply_test_mode_meta_defaults()`（`:622-666`）为未提供的 `tokens`/`step`/
   `prompt_lengths`/`input_tokens`/`output_tokens`/`num_new_tokens`/`qo_indptr_buffer`/`paged_kv_*` 自动
   分配 GPU 张量（形状由 `max_num_*` 推），注释明说是「镜像 `demo/qwen3/demo.py` 的生产接线」（`:624`）。
3. **meta 指针允许空**：`compile` 打包 `meta_tensors_ptr` 时，测试模式下缺的 meta tensor 传 `0`
   而非报错（`:3413-3415`）。
4. **设备端终止协议**（关键）：`include/mirage/persistent_kernel/persistent_kernel.cuh:328`
   ```cpp
   #if defined(MPK_ENABLE_PROFILING) || defined(MPK_TEST_MODE)
         if (true)                    // ← 无条件把请求标 done
   #else
         if ((step+step_advance+1 >= max_seq_length) ||  (… == eos_token_id …))
   #endif
   ```
   即**生产里「请求跑到 `max_seq_length` 或 EOS 才 done」的条件，在测试模式下被替换成 `if(true)`**：
   迭代一次就把 `request_ids[i]=-1`、回收 KV 页——于是常驻 kernel 自己收敛、`pk()` 返回。

**这解释了为什么 `docs/09` 的 P25-1 能直接用官方 `test_rmsnorm_testmode.py` 复现挂起**：它走的就是
上面第 4 条测试终止协议，与生产 termination（`max_seq_length`/EOS）不同；`docs/09` §9–§11 把「挂起」
最终归因到 **worker/scheduler 的 launch 顺序**（scheduler 先 launch 即收敛），与本文第 4 条正交——
**终止协议只决定「跑几轮」，不决定「scheduler 会不会被派发」**。

## 3. 依赖分析（`docs/08`）的可执行规范：fork / join / case 2·3 / residual stripping

`docs/08` 走读的 `annotated_graph.cc` 十步里，最抽象的是 **fork/join 分类 + case 2/3 拒绝 + 残余边剥离**。
`test_mode/` 恰好为每一步配了一个用例——**这些用例的 docstring 就是那一步的自然语言规范**：

| 用例 | 拓扑 | 验证的编译器规则（对应 `docs/08`） |
|---|---|---|
| `test_fork_point_testmode.py` | `x → A → x_a →{B,C}` | **fork 产生**：A 的输出被两个下游消费 → 1 个 fork group、2 条出边（docstring `:8-11`）；两分支都得对（`tol=0.1`） |
| `test_join_point_testmode.py` | `x1→A→x_a`，`x2→B→x_b`，`C=linear_with_residual(x_a, w_c, x_b)` | **join 产生**：C 有两个互不相连的生产者 → 1 个 join group、2 条入边（`:16-18`）；无 residual stripping 适用 |
| `test_diamond_fork_join_testmode.py` | `A → {B,C} → D`（GEMM 分支） | **case 4（允许）**：B/C 同是 fork-consumer 与 join-producer → `trigger_event` 来自 fork 事件、`dependent_event` 指向 join 事件，**两个不同 slot 不冲突**（`:19-23`）；并说明「任一分支以全复制方式读生产者的输出时，event 塌到 1/轴」（`:32-39`） |
| `test_residual_stripping_testmode.py` | `x→A→x_a→B→x_b`，`C=linear_with_residual(x_b,…,residual=x_a)` | **残余边剥离**：A→C 直连与 A→B→C 长路径并存时，直连被当作「**假并行路径**」剥掉（`:10-18`），A 不再 fork、C 不再 join，图塌成链 `A→B→C`；但 `x_a` 运行期仍被 C 读 |
| `test_case2_case3_negative_testmode.py` | 两个非法拓扑 | **case 2 拒绝**（join-consumer + fork-consumer 需两个 `trigger_event`）与 **case 3 拒绝**（fork-producer + join-producer 需两个 `dependent_event`）（`:10-13`）；因编译错误是 `std::runtime_error` 会 `std::terminate()`，测试**在子进程里跑**、查 stderr 里有没有 `"case 2"`/`"case 3"` 且返回码非 0（`:146-176`） |

要点：

- **这五个用例把 `docs/08` §4「case 2/3 拒绝」与「残余边剥离=传递规约」从源码规则变成了可复现断言**：
  `test_case2_case3_negative` 甚至接受「两者同现」——因为**任何被拒拓扑里，case 2 与 case 3 数学上必然
  耦合**（`:148-153`：X 同时是 join-consumer 与 fork-consumer 时，必有一个生产者同时是 fork-producer
  与 join-producer）。
- **`diamond` 用例验证的是 `docs/08` §6 的 fork/join LCM**：两条分支用 **2D grid**（linear 的
  `hidden-tiling × batch`），与 rmsnorm 的 1D batch grid 不同，故「fork/join LCM 数学跑在非平凡 grid
  形状上」（`:28-30`）——直接呼应 `docs/08` §6（GCD 逐边 + fork/join LCM）。
- **`residual` 用例是 `docs/08` §2「阴影覆盖/残余边剥离」的行为见证**：多输入消费者（`linear_with_residual`
  同时读 `input` 与 `residual`）**不应**被误判成 join。

## 4. `grid_dim_sweep`：event 数随 grid 的标度（`docs/08` §5/§6 的量化见证）

`test_grid_dim_sweep_testmode.py` 把同一个 fork 图 `A→{B,C}` 在四档 `(batch, hidden)` 下重编：

```
(batch=4,  h=4096) → 4 fork events,  12 tasks (3 layers × 4)
(batch=8,  h=4096) → 8 fork events,  24 tasks
(batch=16, h=4096) → 16 fork events, 48 tasks
(batch=8,  h=2048) → 8 fork events,  更窄的 hidden
```

docstring 明说：「**改变 `batch_size` 直接改变 fork 事件数（`grid_dim.x=batch_size` 时每个 batch item
一个）**，故这验证 fork emission / prelaunch conversion / JSON task flush 在**不同 task-id 布局**上
都成立」（`:5-8`）。这正是 `docs/08` §5「linearization」+ §6「fork LCM」在**任务粒度 ≠ 1**时的落点：
**event 数 = 被切开的 task 数**（每 batch item 一个 task，故每 item 一个 fork event），而
`hidden=2048`（更窄）只改 task 的**形状**、不改拓扑。

## 5. `test_mode/` 覆盖的其它前端/运行契约（目录）

除 `docs/08` 的依赖分析外，`test_mode/` 还覆盖了大量**前端与 device task 契约**（各文件 docstring 写明
其精确边界，值得作为「哪些行为有回归保护」的索引）：

| 用例 | 覆盖点 |
|---|---|
| `test_rmsnorm_testmode.py` | 最小 PoC：`rmsnorm_layer` 单层、对 PyTorch 参考（`:3-6`） |
| `test_rmsnorm_uneven_testmode.py` | hidden size **不是线程 tile 整数倍**的 RMSNorm（`HIDDEN_DIM` 不再须整除 `NUM_THREADS`） |
| `test_linear_bias_testmode.py` | SM100 linear 的**广播 bias**（= 零行 stride 的 residual epilogue） |
| `test_join_point`/`test_fork_point`/`test_diamond…`/`test_residual_stripping`/`test_case2_case3` | 依赖分析规则（§3） |
| `test_qwen3_mlp_testmode.py` | Qwen3 dense MLP 全链（RMSNorm+linear+SwiGLU…） |
| `test_moe_router_batch_testmode.py` | MoE top-k+softmax 路由在 **batch>1** |
| `test_moe_w13_linear_testmode.py` | BF16 MoE `w13` linear（对应 `docs/24` 的 `moe_w13` task） |
| `test_moe_pipeline_batch_testmode.py` | 完整 BF16 Qwen3 MoE block @ batch=8 |
| `test_moe_gptoss_testmode.py` | GPT-OSS 的 **clamped-alpha SwiGLU** + 两个专家 GEMM 的 per-expert bias（对应 `docs/26` §4 GPT-OSS builder） |
| `test_fp8_moe_pipeline_testmode.py` | **FP8** MoE 流水（对应 `docs/26` §3 UE8M0 重定标） |
| `test_moe_residual_rank_guard_testmode.py` | TP 下 `moe_mul_sum_add_layer` 的 residual 处理 |
| `test_windowed_attention_testmode.py` / `test_attention_sinks_testmode.py` | paged attention 的**滑窗掩码** / **attention sink**（SM100） |
| `test_sampling_topk_topp_testmode.py` / `test_sampling_edges_testmode.py` | temperature/top-k/top-p 采样、`top_k=1`==argmax、**top-k 预算拒绝** |
| `test_argmax_vocab_bound_testmode.py` | lm_head **padding 到超过真实 vocab** 后的 argmax（Qwen3 从 151936 pad 到 153600） |
| `test_kv_planner.py` | `plan_kv_groups` 的单元测试（对应 `docs/25` §7 `kv_planner`） |

**读法**：这张表就是「megakernel 编译栈的回归矩阵」——**每个前端 task / device kernel / 规划器都有
一个能对 PyTorch 参考的最小端到端**。`docs/24`（task 目录）说「95 个 `TaskType` 由谁登记」，本表说
「其中哪些被端到端验证过」。

## 6. `demo/` 入口与测试入口的对照：自动 / 生产 / 实验性三条驱动路

| 驱动路 | 代表文件 | 用哪条编译链 | 目的 |
|---|---|---|---|
| **自动 superoptimizer** | `demo/demo_rms_norm.py`（42 行） | `mi.new_kernel_graph()` → `graph.rms_norm(X)` → `graph.matmul(D,W)` → `graph.mark_output` → **`graph.superoptimize(config="mlp")`**（`:18`）→ `optimized_graph(inputs=…)` | 走 `docs/15/16` 的**自动路径**（`docs/22` §3）；`splitk_linear`/`silu_mul`/`rmsnorm` 图 API，无 `PersistentKernel`、无 `test_mode` |
| **生产手工图** | `demo/qwen3/demo.py`、`demo/deepseek_v3/demo.py`（`docs/25/26/27`） | `MPK`/builder → `PersistentKernel.compile` → `pk()` | 真正部署（online/offline 六种 `MODE_*`） |
| **可执行规范** | `tests/runtime_python/test_mode/*.py` | 直接 `PersistentKernel` + `test_mode=True` + 手工 `*_layer()` | 回归/规范；`docs/09` 也用它 |
| **实验性 TTS/量化注入** | `demo/demo_hopper/main.py`（72 行） | `torchao.quantization.quantize_(model, float8_weight_only())` + `fx.symbolic_trace` + `MugraphInjector`（**注释掉、未接线**，`:38-40`） | 探索「把 mega-kernel matmul 注入 fx 图/FP8量化」——**已实现未启用**（与 `docs/22` §6 同型） |

要点：**同一个 `PersistentKernel` 契约支撑自动路径与生产路径两条上游**（`docs/22` §0 的双链分叉在
「谁调 `compile`」处：自动路径 `KNGraph.compile`、生产/测试路径 `PersistentKernel.compile`）。而
`demo_rms_norm.py` 展示自动路径的**用户面**：`superoptimize` 是 demonstrably 一行——**测试套件却完全
不走它**（`test_mode/` 无一处 `superoptimize`），这从调用面再次佐证 `docs/15` §7 的结论：
**`mpk` 生产前端不调 superoptimizer**。

## 7. 判据 C1–C5 映射

| 判据 | 测试套件是否见证 | 说明 |
|---|---|---|
| **C1 单次 launch** | **是（端到端见证）** | 每个用例只 `pk()` 一次（`__call__`→`launch_func`），一张图一个常驻 kernel；这是 `docs/22` §8「C1 只在运行层拿到」的执行见证 |
| **C2 常驻** | **是（同上）** | `test_mode` 的终止协议（§2）只在常驻 kernel 内生效；`wait()`/`finalize()` 管生命周期 |
| **C3 多算子范围** | **是（逐用例划范围）** | 用例的层序列就是「融合进同图的算子范围」：`rmsnorm` 单算子 → `qwen3_mlp`/`moe_pipeline` 整块 |
| **C4 设备内协调** | **是（fork/join/case 用例）** | fork/join 用例跑的正是「设备内 event 依赖」；case 2/3 负例证明「一张 task 只能一个 trigger/dependent event」的协调器约束 |
| **C5 SM 级调度** | **是（间接）** | worker/scheduler 由 `get_configurations_from_gpu` 按 SM 数定（128/16），测试与生产同一调度器（`docs/06`） |

一句话：**`test_mode/` 是本专题唯一能在「运行层」同时踩到 C1–C5 的源码面的地方**——它把
`docs/22` §8 的判据总映射从「源码位置表」补成「可运行断言表」。

## 8. 局限与坐标

- **纯源码走读，未构建/未运行**：所有结论基于 `f9eb70c` 快照行号；不采信性能数、不独立复现数值
  （端到端行为与 launch 顺序根因见 `docs/09` §7–§12）。
- **只精读依赖分析相关的 5 个用例 + `grid_dim_sweep`**（§3/§4）与其 API 骨架（§1/§2）；§5 的其余用例
  只按 docstring 定位、**未逐行走读其内部断言**（各自对应 device task 在 `docs/07`/`docs/24`）。
- **`test_kv_planner.py` 只是单元测试**（`plan_kv_groups`），不经 `PersistentKernel`/device，与 §1–§4
  不同层。
- **`demo_hopper/main.py` 的 fx/mugraph 注入被注释掉**（`:38-40`），本文只记录「已实现未启用」、未评估
  其可行性。
- **未覆盖 `tests/runtime_python/` 的非 test_mode 部分**（`test_rmsnorm.py` 等走 `runtime_kernel`
  pybind 直接调 device kernel，不经 `PersistentKernel`——那是「单 kernel 单测」，与 megakernel 端到端
  是两层，其 wrapper `runtime_kernel_wrapper.cu` 未走读）。
- **行号已核对**：`persistent_kernel.py`(3618)、`utils.py`(53)、`persistent_kernel.cuh`(MPK_TEST_MODE
  仅 `:328` 一处)、`test_rmsnorm_testmode.py`(91)、`test_fork_point_testmode.py`(105)、
  `test_join_point_testmode.py`(119)、`test_diamond_fork_join_testmode.py`(149)、
  `test_residual_stripping_testmode.py`(117)、`test_case2_case3_negative_testmode.py`(195)、
  `test_grid_dim_sweep_testmode.py`(133)、`demo/demo_rms_norm.py`(42)、`demo/demo_hopper/main.py`(72)。

## 9. 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
  主文件：`python/mirage/mpk/persistent_kernel.py`（`get_default_init_parameters:669`、`attach_input:746`、
  `rmsnorm_layer:865`、`rmsnorm_linear_layer:883`、`linear_layer:2390`、`linear_with_residual_layer:2449`、
  `compile:3207`、`__call__:3574`、`wait:3610`、`finalize:3614`、`target_cc:536`、`test_mode:268/318/538/622/3413`）；
  `python/mirage/utils.py`（`get_scheduler:25-29`、`get_configurations_from_gpu:33-53`）；
  `include/mirage/persistent_kernel/persistent_kernel.cuh:328`（`MPK_TEST_MODE` 终止协议）。
  测试：`tests/runtime_python/test_mode/`（23 个用例）——`test_{fork_point,join_point,diamond_fork_join,
  residual_stripping,case2_case3_negative,grid_dim_sweep,rmsnorm,linear_bias,moe_router_batch,…}_testmode.py`
  与 `test_kv_planner.py`。Demo：`demo/demo_rms_norm.py`、`demo/demo_hopper/main.py`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462–468 操作符
  分解、§4.2 行 1126–1146 每 task 生成、§5.8 局限）。测试套件论文未提，属**实现质量/回归**层面。
- 同专题相关篇：`docs/08`（依赖分析器，本文给它配可执行规范）、`docs/25`（本文补其 §9 尾巴——
  `MirageModelConfig` 调用方）、`docs/06`（运行层 worker/scheduler）、`docs/07`/`docs/24`（device task，
  §5 用例的 device 侧）、`docs/09`（P25-1 用同一批 test_mode 用例诊断不收敛）、`docs/22` §8（判据总映射，
  本文补「运行层可达」的可执行见证）、`docs/15` §7（本文从调用面再证「生产前端不调 superoptimizer」）。
- 交叉专题：DeepGEMM mega 家族的反向/前向端到端验证见 `../mega-moe/`（本专题不重复）。
