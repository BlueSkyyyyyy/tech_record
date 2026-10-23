# MPK 线上实现源码走读

> 本文是 `docs/01-mpk-paper.md`（论文精读）的**实现侧**补充：把 MPK 论文里描述的两层结构
> （编译层 tGraph + 运行层 in-kernel 调度）落到**真实可核对的源码**上。主参考仓库为
> `https://github.com/mirage-project/mirage`（默认分支即 `mpk`），本机快照
> **commit `f9eb70c`（2026-10-05，分支 `mpk`）**，所有行号都对着这次快照核对过。
>
> 定位：`docs/01` §2–§5 解决「论文说了什么」，本文解决「代码怎么做到」。焦点放在**运行层**
> （持久化 kernel、worker/scheduler、事件驱动、设备内原子队列），编译层只走通「tGraph → 生成的
> C++ 源码」这一条路径，不重复 `docs/01` §3 的算法推导。
>
> 本文为**纯源码走读（资料分析）**：mirage 是一套 4.4 万行 C++ + 4.2 万行 CUDA 的完整编译器/运行时，
> 在目标卡 H100/sm90 容器里从零编译成本过高，本轮不做构建/运行实验，所有结论均标注到文件与行号。
> 论文侧证据见 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`。

## 先记住一个画面

一颗常驻 megakernel，像一间**没有工头的车间**。

- 地上摆着一排**料箱**（worker queue），箱口写着「下一个该干的活是几号」；
- **工人**（worker，128 个 block）不等人派活，自己盯着箱口：号一变，就取活、等齐配件、干完在墙上划一笔；
- **记工员**（scheduler，16 个 warp）不看谁闲谁忙，只盯着「配件齐了」的清单，把下一批活写进箱子。

没有工头，就是**没有 host 侧的 launch 循环**；箱口那个号，是一个 64 位原子游标；墙上那一笔，
是一次 `atom.add.release`。

> **类比的边界**：车间里人看得见彼此，GPU 上不能——「看到箱口的号变了」必须靠 `acquire` 语义
> 保证后面的数据也可见，否则会取到一个还没写完的 task 描述符。工人也不会累、不会被换班：
> 常驻 kernel 一旦起来就占着 SM 直到收工。最后一处差别：料箱是**环形**的，写满就得等，
> 所以队列深度本身是一个需要调的参数。

读完这一篇，你应该能回答：

1. 一份 tGraph，是怎么变成 `.cu` 里那串 `if (task_type == ...)` 分支的？
2. worker 取一个 task，要等哪两样东西？各自用什么内存序？
3. 线上实现比论文多做了哪三件事？

## 0. 源码地图与关键坐标

仓库默认分支 `mpk`。运行层（megakernel 范式所在）集中在 `include/mirage/persistent_kernel/`：

| 文件 | 行数 | 角色 |
|---|---|---|
| `include/mirage/persistent_kernel/persistent_kernel.cuh` | 2115 | **运行时主体**：worker/scheduler 设备循环、宿主侧 init/launch |
| `include/mirage/persistent_kernel/runtime_header.h` | 462 | 数据结构：`TaskDesc` / `EventDesc` / `RuntimeConfig` 等 |
| `include/mirage/persistent_kernel/mpk_atoms.cuh` | 97 | 内存序原语（`atom.add.release.gpu.*` / `ld.acquire.*`） |
| `include/mirage/persistent_kernel/profiler.h` | 160 | profiling 埋点 |
| `include/mirage/persistent_kernel/tasks/` | — | 每种 task 的 device 实现（`ampere/` `hopper/` `blackwell/`） |
| `src/kernel/runtime.cc` | 1972 | 编译层：tGraph → 生成 `_init_persistent_kernel` / `_execute_task` C++ |
| `python/mirage/mpk/persistent_kernel.py` | 3618 | Python 前端：构图、`compile()`、生成 host/launcher |
| `python/mirage/utils.py` | 53 | SM 探测 → worker/scheduler 数量 |

一句话结构：**Python 构图（`PersistentKernel`）→ C++ 编译成一张 task/event 图（tGraph）→
`compile()` 把图固化成宿主初始化代码 + `_execute_task` 派发代码 → `init_persistent_kernel()`
把队列/事件/task 全部 `cudaMalloc` 到显存 → `launch_persistent_kernel()` 起**唯一一次**驻留 kernel,
其内部 worker 执行 task、scheduler 派发 task，靠设备内原子队列闭环。**整图只有 1 次 launch（判据 C1）、
kernel 常驻（C2）、跨算子（C3/C4）、SM 级调度（C5）。**

## 1. 编译层：tGraph 如何变成一份可 launch 的运行时

### 1.1 task/event 图的 C++ 固化石

`src/kernel/runtime.cc:1915` 的 `Graph::generate_task_graph(_num_gpus, _my_gpu_id)` 是入口：它把
KNGraph 编译成 `(all_tasks, all_events, first_tasks)` 三元组，最后调用
`print_task_graph(...)`（`runtime.cc:1939`）生成 C++。生成的产物有两块：

1. **图加载器**：`_init_persistent_kernel(vector<FullTaskDesc>&, vector<EventDesc>&, vector<TaskId>&, ...)`
   （声明见 `persistent_kernel.cuh:1566-1572`）。它把每个 task/event 以
   `all_tasks.push_back(FullTaskDesc(...))`、`all_events.push_back(EventDesc(...))` 的形式**硬编码**
   进生成的 C++（`runtime.cc:1730-1746`）。
2. **task 派发器**：`_execute_task(TaskDesc const*, RuntimeConfig const&)` 由
   `runtime.cc:1885-1903` 生成——对 `TaskRegister` 里注册的每个 `(TaskType, variant_id)` 产出一个
   `if (task_desc->task_type == TASK_xxx && task_desc->variant_id == v) { <该算子的 device 实现> }`
   分支。`persistent_kernel.cuh:1145` 的 worker 循环调用它，**这就是「每个 task 的 CUDA 实现生成」**
   （论文 §4.2，见 `docs/01` §3）。

图里有两个特殊 task 作为前导（prelude），不参与 JSON，由生成器处理
（`runtime.cc:1722-1724`）：

- `task[0] = TASK_TERMINATE`（终止哨兵，`runtime_header.h:88`）；
- `task[1] = TASK_BEGIN_TASK_GRAPH`，其 `trigger_event = event 1`（`runtime.cc:1344-1360`）——
  它是**每轮推理图的第一颗 task**，触发后把后继算子的 event 推上调度队列。

`generate_task_graph` 里还会对整图做一次 `sanity_check`（`runtime.cc:1938`），并额外
**追加一个 `EVENT_TERMINATION` 事件 + `TASK_TERMINATE` task**（`runtime.cc:1925-1929`）作为收尾。

### 1.2 Python 前端：构图 + 生成 host 侧派发

`python/mirage/mpk/persistent_kernel.py:413` 的 `PersistentKernel` 是用户 API。它以
`self.kn_graph = KNGraph(...)`（`:501`）累积算子（`embed_layer` / `rmsnorm_layer` /
`attention_layer` / `moe_router` 等），`attach_input`（`:746`）把 PyTorch tensor 的**裸 GPU 指针**
烘焙进生成的代码（`:773-776` 注释解释了为何要保活 tensor 引用）。`compile()`（`:3207`）调用
`self.kn_graph.generate_task_graph(...)`（`:3223`），把返回的 CUDA 代码 + `task_graph.json` 写到产物目录。

生成的 host 侧是一个 Python 扩展模块 `__mirage_launcher`（硬编码模板
`persistent_kernel.py:22-226`），暴露四个入口：`init_func` → `init_persistent_kernel(...)`（`:121`）、
`launch_func` → `launch_persistent_kernel(stream)`（`:141`）、`wait_func`、`finalize_func`。
**这正是「编译一次、之后每次推理只做 launch + wait」的边界**：图是静态的，运行时不再重组。

### 1.3 SM 数 → worker/scheduler 划分（对上论文 §5.1）

`python/mirage/utils.py:25-29` 给出硬关系：`scheduler = 4 * (sm_cnt - worker)`。即
**每个 scheduler 块放 4 个 scheduler warp、恰好占 1 个 SM**，其余 SM 全部做 worker。
`get_configurations_from_gpu`（`:33-53`）按 SM 数选 worker：目标卡 H100 有 132 SM →
`sm_cnt>=132` 分支取 **worker = 128**，于是 scheduler = `4*(132-128)` = **16 个 scheduler（= 4 个 SM）**。
这与论文 §5.1「4 SM / 16 warp 做 scheduler，其余 SM 做 worker」完全吻合
（论文侧见 `docs/01` §4）。

## 2. 运行时数据结构（`runtime_header.h`）

### 2.1 ID 编码

`TaskId` 与 `EventId` 都是 64 位打包值（`runtime_header.h:71-80`）：

- **TaskId**：高 32 位 = **iteration（迭代号）**，低 32 位 = **task index**（`:73-74`）。
  iteration 让同一张静态图在多轮解码里复用同一 index。
- **EventId**：`nvshmem_tag:16 | owner_node:16 | event_idx:32`（`:76-77`）。跨 GPU 时 `owner_node`
  与 `nvshmem_tag` 用于路由（本文只关注单 GPU：`owner_node == my_gpu_id`）。
- 哨兵：`TASK_INVALID_ID = 0x7fff...`、`EVENT_INVALID_ID = 0x7ffe...`（`:72,79`）。

### 2.2 task 描述符

- `FullTaskDesc`（`:260-290`）：编译期完整描述，含 `trigger_event`（本 task 完成后触发谁）、
  `dependent_event`（本 task 等谁）、输入/输出 `TensorDesc`，以及一个 8 字节 `TaskMetadata`
  union（`expert_offset` / paged-attention 的 `request_id,kv_idx,merge_task_offset` / nvshmem
  `task_offset`，`:276-289`）。
- `TaskDesc`（`:296-336`，`alignas(16)`）：**运行期**描述符，把 `TensorDesc` 压成裸 `void*` 指针数组
  （`input_ptrs[8]` / `output_ptrs[3]`，`:327-328`），并按 16 字节对齐以便用 **128-bit `cp.async`**
  批量拷进 shared memory（`:1064-1075` 的 128b 拷贝）。

### 2.3 事件与运行时配置

- `EventDesc`（`:249-258`）：`event_type` + `num_triggers`（需要被触发多少次才算「就绪」）+
  `first_task_id`/`last_task_id`（该事件一次性放行的 task 区间）。事件类型枚举
  `EventType`（`:228-236`）：`EVENT_EMPTY(900)`、`EVENT_LAUNCH_TASKS(901)`、
  `EVENT_LAUNCH_MASSIVE_TASKS(902)`、`EVENT_LAUNCH_DEPENDENT_TASKS(903)`、
  `EVENT_END_OF_TASK_GRAPH(910)`、`EVENT_TERMINATION(911)`。
- `RuntimeConfig`（`:338-459`）：把**所有队列、计数器、task/event 表、以及 LLM serving 的元数据
  （step/tokens/page 表…）** 打包成一个按值传给 kernel 的 POD。关键字段：
  `worker_queue_last_ready_task_id`（worker 队列的生产者游标）、
  `sched_queue_last_ready_event_id` / `sched_queue_next_free_event_id`（scheduler 队列游标）、
  `all_event_counters` / `all_event_num_triggers`（事件计数器与其阈值）、
  `worker_queues` / `sched_queues`（**指针数组**，每个 worker/scheduler 一条独立环形队列）。

## 3. 设备侧 worker 循环（`execute_worker`，`persistent_kernel.cuh:957-1252`）

一个 worker = 一个 CUDA block（`blockIdx.x < num_workers`），常驻执行一个 `while(true)` 循环：

1. **拉取 task**（`:1009-1081`）。worker 队列为空时，`threadIdx.x==0` 轮询本队列的
   `worker_queue_last_ready_task_id`（`ld.acquire.gpu`，`:1016`），`__nanosleep(10)` 自旋等待
   （`:1025`）；每 GPU 有 local + remote 两条队列（`:985-993`，单卡时只 1 条）。拿到新区间后，
   用 `ld.relaxed.gpu` 读 task id（`:1036`），再用 **128b `cp.async`** 把 `TaskDesc` 搬进
   shared memory 缓冲（`:1063-1078`，`cp_async_fence`/`cp_async_wait`）。这就是论文 §5.3 的
   **task descriptor 预取 + 批量加载**的实现雏形。

2. **等依赖**（`:1084-1111`）。若 `task_desc->dependent_event != EVENT_INVALID_ID`，则
   **自旋等待事件计数器达到阈值**：`needed_counts = all_event_num_triggers[event_index] *
   get_task_iteration_num(task_id)`（`:1090-1093`，即「本迭代内该事件的触发总次数」），
   用 `ld.acquire.sys` 读 `all_event_counters`（`:1106`）直到达标。跨 GPU 的 nvshmem 事件走
   `nvshmem_signal_wait_until`（`:1097-1101`）。**注意此处是「数据依赖的精确同代（iteration）匹配」**
   ——正是 `docs/00` §3.2 判据里 C4 的核心。

3. **执行**（`:1121-1146`）。`TASK_TERMINATE` 直接 `return`；`TASK_BEGIN_TASK_GRAPH` 空操作；
   其余调用生成的 `_execute_task(task_desc, config)`。

4. **触发后继**（`:1155-1249`）。`threadIdx.x==0` 对 `all_event_counters[event_index]` 做
   `atom.add.release.gpu.u64(..., 1)`（`:1164`）；当 `(count+1) == num_triggers * iteration`
   （`:1185`）说明事件「满足次数」，就把该 event 索引推入某条 **scheduler 队列**：
   - 选择队列（`:1201-1212`）：`EVENT_LAUNCH_MASSIVE_TASKS` / `EVENT_LAUNCH_DEPENDENT_TASKS`
     走**广播队列**（最后一个 sched_id，所有 local scheduler 共享，`:1291`），普通的 `EVENT_LAUNCH_TASKS`
     则按 `get_rand_sched_id` 随机分派给某 local scheduler；
   - 入队用 `atom.add.release.gpu.u64(sched_queue_next_free_event_id)` 取槽
     （`:1213-1214`），`st.relaxed.gpu` 写事件索引（`:1215-1218`），再用
     **CAS-release 推进 `sched_queue_last_ready_event_id`**（`:1222-1228`）——先写数据、后放行读游标，
     用 release 语义保证 scheduler CTA 看到完整事件。

## 4. 设备侧 scheduler 循环（`execute_scheduler`，`persistent_kernel.cuh:1254-1523`）

**一个 scheduler = 一个 warp**，一个 block 里最多 `SCHEDULERS_PER_BLOCK = 4` 个 warp
（`:1258-1260,1267`），`sched_id = blockIdx.x*4 + warp_id + offset`（`:1268`）。local scheduler
除了自己的队列，还**集体消费一条全局广播队列**（`:1288-1298`）。循环：

1. **等事件**（`:1334-1354`）：`ld.acquire.gpu` 读 `sched_queue_last_ready_event_id`（`:1340`），
   `__nanosleep(10)` 自旋；`assert` 保证队列不溢出（`:1353`）。
2. **读事件**（`:1357-1360`）：`ld.relaxed.gpu` 取 event_id，取 `EventDesc e`。
3. **按事件类型派发 task 到 worker 队列**：
   - `EVENT_TERMINATION`（`:1361-1376`）：向自己负责的所有 worker 队列写终止哨兵（task id 0）并放行；
   - `EVENT_END_OF_TASK_GRAPH`（`:1378-1423`）：调用 **`prepare_next_batch(config)`** 组织下一个 batch
     ——若返回 false（无更多请求）则 `terminate_schedulers`（`:1394`）；否则给下一个 worker 下发
     **下一迭代的 `TASK_BEGIN_TASK_GRAPH`**（`compute_task_id(iteration_num+1, 1)`，`:1400-1409`），
     这是**连续服务（offline/online batched serving）不重启 kernel 的关键**；
   - `EVENT_LAUNCH_DEPENDENT_TASKS`（`:1424-1474`）：`iteration_num += 1`，把 `[first_task_id,
     last_task_id)` 区间内的 task **round-robin 分给 `my_first_worker..my_last_worker`**；
   - 其余（`EVENT_LAUNCH_TASKS` / `EVENT_LAUNCH_MASSIVE_TASKS`，`:1475-1519`）：普通事件按
     `sched_id` 顺序派发；MASSIVE 事件先用 `get_first_last_ids` **在 local scheduler 间切分区间**
     （`:1477-1487`）。
   - 每次派发都是「`st.relaxed.gpu` 写 worker 队列槽 → `atom.add.release.gpu` 放行
     `worker_queue_last_ready_task_id`」（如 `:1495-1502`），与 worker 侧的 acquire 读配对。

这就是论文 §5 描述的**去中心化 in-kernel 调度**：没有全局 dispatch 表，也没有 host 参与；
worker 与 scheduler 只通过**设备内原子环形队列 + 事件计数器**交换信息，调度决策由**固定角色 +
本地状态**完成（`docs/01` §4、`docs/03` §3.3）。

## 5. 宿主侧：init / prepare / launch

### 5.1 `init_persistent_kernel`（`:1610-1951`）——一次性资源分配

`meta_tensors` 是 serving 元数据（step/tokens/page 表，`:1634-1665`）；随后用 `gpu_malloc`
分配并填充：

- 每个 event 的计数器 `all_event_counters`、阈值 `all_event_num_triggers`（`:1848-1859`）；
- 全部 task `all_tasks`、全部 event `all_events`（`:1868-1881`，`cudaMemcpy` H2D）；
- **worker 队列**：`num_workers * 2` 条环形缓冲（local+remote，`:1885-1891`）；
- **scheduler 队列**：`num_schedulers + 1` 条（多一条全局广播，`:1900-1906`）；
- `first_tasks`（`:1912-1919`）。

最后 `cudaFuncSetAttribute` 给 `worker_kernel`/`persistent_kernel` 设置动态 shared memory 上限
（`:1923-1930`），并创建 worker/scheduler 两条非阻塞 stream 与事件（`:1931-1944`）。

### 5.2 `launch_persistent_kernel`（`:1955-…`）——唯一的每轮提交

每次推理调用 `prepare_kernel<<<num_workers,128>>>`（`:1963-1967`）**重置所有队列游标、清零事件
计数器**，并**把 `EVENT_END_OF_TASK_GRAPH` 事件预置到 scheduler[0] 的队列**（`prepare_kernel`
`:226-233`）——这就是整图的**启动种子**：scheduler 读到它 → `prepare_next_batch` → 下发
`TASK_BEGIN_TASK_GRAPH` → 首发事件级联放行整图。随后有两条路径（`RuntimeConfig.split_worker_scheduler`，`:1977`）：

- **分开启动**（`:1977-2005`）：`worker_kernel`（grid = num_workers）与 `scheduler_kernel`
  （grid = `num_local_schedulers/4`，block = `32*4`）在**两条独立 stream** 上并发常驻；
- **合并启动**：直接起 `persistent_kernel`（`:1525-1533`），一个 grid 里
  `blockIdx.x < num_workers` 跑 worker、其余 block 跑 scheduler（`:1528-1532`）。

两者都是**一次 launch、kernel 常驻到整轮推理结束**——判据 C1/C2 的落地。

### 5.3 设备内同步原语（`mpk_atoms.cuh`）

调度正确性依赖一组手写 PTX 内存序原语（`:17-96`）：`atom.add.release.gpu.u64`（`:26`）、
`atom.cas.release.gpu.b64`（`:37`）、`ld.acquire.gpu.u64`（`:49`）、`ld.acquire.sys.u64`（`:56`，
用于事件计数器、保证跨 SM 可见）、`ld.relaxed.gpu.u64`（`:66`）、`st.relaxed.gpu.u64`（`:73`）。
队列协议统一是**「relaxed 写数据 + release 推游标 / acquire 读游标 + relaxed 读数据」**，
是典型 SPSC/MPSC 无锁环形的配对写法。

## 6. 与论文的逐项对照与差异

| 论文机制（快照章节） | 源码落点 | 备注 |
|---|---|---|
| 事件驱动、start event（§5.1） | `prepare_kernel` 预置 `END_OF_TASK_GRAPH`（`:226-233`）+ `task[1]` 触发（`runtime.cc:1344`） | 一致 |
| worker/scheduler 角色划分（§5.1） | `persistent_kernel`（`:1525`）、`utils.py:25-29`（4 warp/SM） | 一致；H100 → 128 worker/16 scheduler |
| 依赖精确触发（§5.2） | `all_event_counters` 阈值 `num_triggers*iteration`（`:1090-1093`） | 含 iteration 代次，防跨代串扰 |
| 去中心化调度（§5.1/§6.1） | worker↔scheduler 原子环形队列（`:1213-1228` / `:1495-1502`） | 无 host、无全局表 |
| task descriptor 预取（§5.3） | smem 缓冲 + 128b `cp.async` 批量加载（`:965-969,1063-1078`） | 一致 |
| 连续服务 / prepare next batch（§5.2） | `EVENT_END_OF_TASK_GRAPH` → `prepare_next_batch`（`:1378-1423`） | offline/online 各一份实现 |
| 「一个 task 一段生成 CUDA」（§4.2） | `_execute_task` 生成（`runtime.cc:1885-1903`） | 一致 |

**观察到的实现差异/延伸**（论文未展开、源码里可见）：

- 源码把事件分了三档 `EVENT_LAUNCH_TASKS` / `_MASSIVE_`（跨 local scheduler 切区间）/
  `_DEPENDENT_`（单 scheduler 内 round-robin），是论文「同一 event 扇出多 task」的具体工程化。
- `RuntimeConfig` 内联了大量 **LLM serving 元数据**（step/tokens/paged-KV 表、`MODE_ONLINE_PINNED`
  的 CPU↔GPU 无锁环形，`:408-450`），说明线上实现早已超出「离线固定 batch」的论文设定
  ——对应 `docs/01` §5.8 列出的论文局限。
- 提供了 `split_worker_scheduler` 开关（`:1977`），既支持「单 kernel 合并」也支持「worker/scheduler
  双 kernel 分 stream」——后者更接近 `docs/03` 的 persistent+调度器拆分形态，用于调试/兼容 NVSHMEM。

## 7. 本轮小结

- MPK 线上实现与论文**两层结构完全对应**：编译层把算子图去中心化成 task/event tGraph
  （`runtime.cc` 生成 `_init_persistent_kernel` + `_execute_task`），运行层用一个常驻 kernel
  内的 worker（执行）+ scheduler（派发）闭环完成整图，**单次 launch、跨迭代复用**。
- **判据映射**（沿用 `docs/00` §3.2）：C1 单次 `launch_persistent_kernel`；C2 `while(true)` 常驻；
  C3 `_execute_task` 覆盖全部算子类型（`TaskType` 枚举 `runtime_header.h:87-226`）；
  C4 设备内原子队列 + 事件计数器（`mpk_atoms.cuh` + `execute_worker/scheduler`）；
  C5 scheduler 逐 task round-robin 到 worker、MASSIVE 事件跨 SM 切分。**C1–C5 全满足**。
- 与生产案例对照：DeepGEMM mega 家族是**手工静态角色 + 少量 barrier**的段级 T4
  （`docs/02` §1），MPK 是**自动生成 + 通用 event queue** 的整图 T4（`docs/00` §4）——两者在
  「常驻 CTA / 设备内协调 / SM 级调度」上同构，差别在**融合范围**（段 vs 整图）与
  **调度器是否通用**（手写角色 vs 生成式 event-driven）。
- 待办（不属现有 P0–P6 清单，作为后续可选增量）：可尝试在容器内 `pip install -e .` 构建 mirage
  的 sm90 路径，跑 `demo/qwen3/demo.py --use-mirage` 拿到本机 MPK 端到端数字；或走读 `tasks/hopper/`
  下某个具体 task（如 `linear.cuh`）的 device 实现，补上「单 task 内部」如何做 software pipelining。

## 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（见 `docs/01`）。
- 专题内交叉引用：`docs/00` §3（判据/谱系）、`docs/01` §2–§5（论文精读）、`docs/03` §3（in-kernel 调度）。
- 生产案例：`docs/02` §1（DeepGEMM 段级 mega-kernel）。
