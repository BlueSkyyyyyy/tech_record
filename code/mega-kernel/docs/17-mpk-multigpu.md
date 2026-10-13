# MPK 多卡/NVSHMEM 路径源码走读：把通信编进同一个常驻 mega-kernel

> 本文补齐 `docs/15`（P101-1）**§9 明留的尾巴** 与 `docs/16`（P102-1）**§6 明确转引的同一条**：
>
> > **多 GPU/NVSHMEM 路径未展开**：`resolve_distributed_config`（`transpiler.h:53-58`）的
> > `gpu_dim`/NVSHMEM 分支与 `multigpu.py` 的张量并行切分未走读。
>
> 前面六篇源码走读（`docs/06` 运行层、`docs/07` 单 task device、`docs/08` 依赖分析器、`docs/14`
> 前端、`docs/15` 自动搜索、`docs/16` 搜索辅助模块）**全是单卡视角**：它们讲清楚了「一张算子图怎么
> 变成 tGraph、每个 task 怎么生成 device 代码、worker/scheduler 怎么跑」。但 MPK 论文标题与摘要
> 反复强调的恰恰是 **multi-GPU**（快照论文摘要行 39：*automatically transforms multi-GPU model
> inference into a single high-performance mega-kernel*）。本文把这条多卡路径逐段走通：跨卡依赖是
> **怎么进同一个 tGraph 的**、跨卡完成信号**怎么触发 event**、AllReduce 在实现里**有哪两种落法**、
> 以及为什么多卡把「单次 launch 的常驻 kernel」从**可选优化**变成**唯一可行路径**。
>
> 主参考仓库 `https://github.com/mirage-project/mirage`（分支 `mpk`），本机快照
> **commit `f9eb70c`**，与 `docs/06`/`docs/07`/`docs/08`/`docs/14`/`docs/15`/`docs/16` 同一快照，
> 所有行号都据此核对。论文侧证据用
> `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（以下简称「快照论文」）。
>
> 本文为**纯源码走读（资料分析）**：多卡运行需要 ≥2 张 NVLink 互连卡 + NVSHMEM + MPI，本机不具备，
> 故**不构建/运行多卡**、不产生新实验；跨卡单机验证过的部分（`docs/06`/`docs/10`）另行标注。
> 通信-计算重叠的**单机可测替身**见 `docs/04` §14（P20-1）与本专题 `harness/cross_op_overlap.out.txt`。

## 0. 定位：多卡路径在 MPK 编译栈里的坐标

MPK 的编译栈（`docs/14` §0 给过四层）在多卡下**每层都加了一个「GPU 维」参数**，且只有一处是真正
多卡特有的：

| 层 | 单卡入口 | 多卡入口/变化 | 源码坐标 |
|---|---|---|---|
| 前端 | `PersistentKernel` 建图 | `MPK.init_mpi()` 取 `world_size/rank`；`allreduce_layer` 按能力选实现 | `mpk.py:298-312`、`persistent_kernel.py:2488-2510` |
| tGraph 生成 | `generate_task_graph()` | `generate_task_graph(num_gpus, my_gpu_id)` | `graph.cc:186`、`runtime.cc:1915-1949` |
| 依赖/event | `register_mugraph`（单卡段） | 同一函数，多一个 `num_gpus`/`my_gpu_id`，**只对一种 task 打 NVSHMEM 事件标签** | `runtime.cc:223-243`、`:657-659` |
| 运行层 | `launch_persistent_kernel` | NVSHMEM 初始化 + `nvshmemx_collective_launch` + 跨卡 barrier | `persistent_kernel.cuh:1720-1755`、`:1955-2043` |
| device task | 本地 task | 两种 AllReduce device 实现 | `tasks/{ampere,hopper,blackwell}/allreduce.cuh` |

关键结论先行：**MPK 并没有为多卡另造一套 runtime**。它只把两件东西塞进原有机制——(a) 给「跨卡的
数据传输 task」打一个 `EVENT_NVSHMEM_TAG`，让它的**事件由别的 GPU 触发**；(b) 把 NVSHMEM 的
collective 当**普通 task** 编进同一张 tGraph，用 `nvshmemx_collective_launch` 启动同一个常驻 kernel。
下面 §1–§8 逐层走通，§9 对照论文，§10 映射 C1–C5，§11 给局限。

## 1. 启动栈：MPI 定 world/rank，每 rank 编自己的 launcher

多卡的**进程模型是 MPI**。`MPK.init_mpi()`（`python/mirage/mpk/mpk.py:298-312`）：

```python
def init_mpi(self):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        self.world_size = comm.Get_size()
        self.rank = comm.Get_rank()
        os.environ["RANK"] = str(self.rank); os.environ["WORLD_SIZE"] = str(self.world_size)
        ...
    except ImportError:
        self.world_size = 1; self.rank = 0
    if self.world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
```

即 **MPI 定拓扑、NCCL 只用其 process group 做初始化握手**（数据面是 NVSHMEM，不是 NCCL）。
`persistent_kernel.py:523` 把「是否走 NVSHMEM」等同于「>1 卡」：`self.use_nvshmem = world_size > 1`。
每一 rank **各自生成任务图、各自 nvcc 编译一个 launcher `.so`**（`persistent_kernel.py:3223` 调
`generate_task_graph(num_gpus=self.world_size, my_gpu_id=self.mpi_rank)`，`:3239-3241` 按
`test_rank{mpi_rank}.cu` / `task_graph_rank{mpi_rank}.json` / `mpk_launcher_rank{mpi_rank}...so` 落盘）。
也就是说 **tGraph 不是全局共享的，而是每个 rank 按「我是谁」生成自己的一份**——这正是后面
「同一张算子图里，每个 rank 的同一 event 由不同 GPU 触发」的编译期基础。

device 侧初始化在 `init_persistent_kernel`（`persistent_kernel.cuh:1610-1755`）：`cudaSetDevice(my_rank)`
（`:1716`）→ `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)` + `nvshmem_barrier_all()`
（`:1721-1725`）。**launch 与 finalize 各有一道跨卡 barrier**：`launch_persistent_kernel` 在
`prepare_kernel` 之后 `nvshmem_barrier_all()`（`:1972`），`init_request_resources` 末尾再一道
（`:1949`）；`finalize_persistent_kernel` 在释放资源后 `nvshmem_barrier_all(); nvshmem_finalize();`
（`:2104-2105`）。host 侧对应 `mpk.py:513-516` 的 `MPK.__del__` → `dist.destroy_process_group()`。
含义：**所有 rank 必须在 programmed 的同一批「相位」里齐步**——这正是「一个常驻 kernel 跨卡并行」
能成立的运行期前提。

## 2. 多卡为什么**被迫**单 kernel：`nvshmemx_collective_launch`

这是多卡路径里**对 megakernel 范式最重要的一条**。`docs/06` 讲过 MPK 运行层有「融合单 kernel」
与「worker/scheduler 分两个 kernel」两种形态（`persistent_kernel.cuh:1977` 的
`split_worker_scheduler`）。多卡选了哪一个？

看 `launch_persistent_kernel`（`persistent_kernel.cuh:1955-2043`）：

```cpp
if (global_runtime_config.split_worker_scheduler) {
    // ...
    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<...>>>(...);
    scheduler_kernel<<<...>>>(...);
} else {
    // a single persistent kernel
    #ifdef USE_NVSHMEM
    void *args[] = {&global_runtime_config};
    nvshmemx_collective_launch((void const *)persistent_kernel, dim3(num_sms_to_use,1,1), ...);
    #else
    persistent_kernel<<<...>>>(...);
    #endif
}
```

注释（`:1988-1990`）说得很直白：**split 路径不支持 NVSHMEM，因为 `nvshmemx_collective_launch`
是顺序启动 kernel 的，会阻断 worker kernel 与 scheduler kernel 的交互**。也就是说：

- 单卡时，「融合单 kernel」vs「worker/scheduler 双 kernel」是一个**性能/实现取舍**（`docs/06` §4）；
- **多卡时，NVSHMEM 只提供 `nvshmemx_collective_launch` 这一种启动方式，它要求整个 grid 是一个
  协同 kernel，于是 split 路径被排他性地排除**——多卡**只能**走单融合 kernel。

这给 megakernel 的「C1 单次 launch」在**多卡**下补了一个新理由：**不仅是省 launch 开销，而是
NVSHMEM 的启动原语根本不支持把常驻 kernel 拆成两个**。论文把「单 kernel」当优化讲，实现里在多卡
下它是**被互连库逼出来的唯一形态**。

## 3. event id 的多卡编码：`nvshmem_tag : 16 | owner : 16 | idx : 32`

跨卡依赖的载体仍是 event，但 event id 多了一位「owner GPU」。`runtime_header.h:75-80`：

```cpp
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
```

编译期 `get_event_id(my_gpu_id, event_pos, nvshmem_event)`（`runtime.cc:30-36`）把
`(my_gpu_id << 32) | event_pos`，再按需 `| EVENT_NVSHMEM_TAG`；device 侧
`get_event_gpu_id`（`persistent_kernel.cuh:134-136`）取高 16 位 owner、`get_event_position_index`
（`:138-140`）取低 32 位 index、`is_nvshmem_event`（`:130-132`）测 tag 位。注意 `RuntimeConfig` 的
event 数组是**本 GPU 私有的**（`runtime_header.h:340-347` 的 `num_gpus, my_gpu_id` /
`all_event_counters` / `all_event_num_triggers`）——跨卡事件不是共享内存，而是**别的 GPU 往我的
计数器里打信号**（§5）。

关键语义差：**普通 event 由「本 GPU 上完成生产者 task 的 worker」触发；NVSHMEM event 由
「别的 GPU」触发**。`runtime.cc:154-157` 有明文注释：

```cpp
if (nvshmem_event) {
    // NVSHMEM events need to be triggered by all other GPUs
    nvshmem_events_idx.insert(all_events.size());
}
```

## 4. 编译期：谁被标成「多卡 task」

### 4.1 只有一种 task 会打 NVSHMEM 标签

在 `register_mugraph` 的主循环里（`runtime.cc:644-659`）：

```cpp
int cur_num_subtasks = get_num_subtasks(num_gpus, task_type);
bool cur_is_multigpu = (task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
layer_is_multigpu[layer_idx] = cur_is_multigpu;
```

`get_num_subtasks`（`runtime.cc:66-74`）对 `TASK_NVSHMEM_ALLGATHER_STRIDED_PUT` 返回
`num_gpus - 1`，其余一律 1——即 **AllGather 的每个 bid 展开成「给每个别的 GPU 各一个 subtask」**
（注释 `:62-64`）。`get_num_subtasks` / `nfshmem_task` 的这套展开在 DFS 事件生成
（`dfs_create_events_add_tasks`，`runtime.cc:76-221`）里被逐 bid 实现（`:108-118` 建
`num_gpus-1` 个 subtask），并在设生产者触发者时把每个 subtask 的 `trigger_event` 改写成
**目标 GPU 的 event id**（`:129-150`）：

```cpp
for (int tgt_gpu_id = 0; tgt_gpu_id < num_gpus; tgt_gpu_id++) {
    if (tgt_gpu_id == my_gpu_id) continue;
    size_t idx = tgt_gpu_id < my_gpu_id ? tgt_gpu_id : tgt_gpu_id - 1;
    all_tasks[task_ids[idx]].trigger_event =
        get_event_id(tgt_gpu_id, all_events.size(), nvshmem_event);
    event_desc.num_triggers++;
}
```

同样的逻辑在分叉/合并路径的 `set_producer_triggers`（`runtime.cc:604-639`）里复刻一遍。含义：
**「我给 GPU j 发了数据」这个动作的完成事件，记在 GPU j 的 event 表里**——consumer 在 j 上等的是
自己的 event，等到了就说明「i 的数据已到」。这是把「通信完成」表示成「数据可用」的编译期落点，
对应快照论文 §5.1 行 1220-1224 的「*reacting to data availability rather than relying on a static
topology model*」。

### 4.2 NVSHMEM Tile AllReduce **不打**标签

注意 `runtime.cc:442-446`：`TASK_NVSHMEM_TILE_ALLREDUCE` 只设置
`task_metadata.task_offset = bid.x + bid.y*grid.x + bid.z*grid.x*grid.y`（线性 bid），
**没有**进 `layer_is_multigpu`。也就是说 **Tile AllReduce 在 tGraph 里是一个「普通本地 task」**：
它的 dependent/trigger event 都是本 GPU 的普通 event，跨卡同步发生在 **task 内部**（§6）。这两种
AllReduce 走的是两条完全不同的路。

## 5. 设备侧：换一种方式等事件

worker 等 dependent event 的分支在 `persistent_kernel.cuh:1084-1110`：对普通 event 用
acquire 自旋（`ld_acquire_sys_u64` + `__nanosleep(10)`），对 NVSHMEM event 换成
**`nvshmem_signal_wait_until`**：

```cpp
if (is_nvshmem_event(event_id)) {
    nvshmem_signal_wait_until(
        reinterpret_cast<uint64_t *>(&config.all_event_counters[event_index]),
        NVSHMEM_CMP_EQ, needed_counts);
} else {
    while (actual_counts < needed_counts) {
        actual_counts = ld_acquire_sys_u64(&config.all_event_counters[event_index]);
        __nanosleep(10);
    }
}
```

`needed_counts = all_event_num_triggers[event_index] * task_iteration_num`——即「**所有别的 GPU 都
各打过一次信号**」才算就绪。反过来，当 worker 完成一个**生产者** task 要触发事件时，若
`trigger_event` 是 NVSHMEM event，则 `:1234-1248` 的分支**什么都不做**，注释如下：

```cpp
// Case 2: trigger a nvshmem event
// TODO(Zepeng): This branch is no longer used. Sanitize later.
assert(task_desc->task_type == TASK_NVSHMEM_ALLGATHER_STRIDED_PUT);
// Note that nvshmem copy task signal counter during data copy
// we don't need to do anything here is the task type is NVSHMEM_COPY
```

理由：**信号是在数据传输 task 内部发生的**（§6.1 的 `nvshmemx_signal_op`），所以 worker 层不需要
再补一次。含义：**跨卡完成信号与本地 event 计数走的是同一个 `all_event_counters` 数组，只是
「谁写它」不同**——这正是「统一用 event 表示计算与通信」落到实现的样子。

## 6. 两种 AllReduce 实现

前端 `allreduce_layer`（`persistent_kernel.py:2488-2510`）从三个张量
（`input [batch, hidden]`、`buffer [world_size, batch, hidden]`、`output [batch, hidden]`）和
`params=[world_size, my_gpu_id]` 出发，调用
`auto_select_allreduce_implementation(self.world_size, self.mpi_rank)` 选实现。选择逻辑
（`python/mirage/mpk/multigpu.py:224-247`）：**SM≥90 且 VMM + multicast + full peer-access**
→ Tile；否则回退 Allgather+Reduce。

### 6.1 策略 A：AllGather + Reduce（论文描述的那条路）

`AllReduceStrategy_AllgatherReduce`（`multigpu.py:156-184`）注册**两个** bGraph/task：

```python
# allgather
allgather_tb_graph.new_input(input_tensor,  (1, -1, -1), -1, True)
allgather_tb_graph.new_input(buffer_tensor,(2, -1, -1), -1, True)
mpk.kn_graph.register_task(allgather_tb_graph, "nvshmem_allgather_strided_put", params)
# reduction
reduction_tb_graph.new_input(... input, buffer, output ...)
mpk.kn_graph.register_task(reduction_tb_graph, "reduction", params)
```

device 实现 `nvshmem_allgather_strided_put`（`tasks/ampere/allreduce.cuh:31-58`）：逐行
`nvshmemx_putmem_nbi_block` 把本 rank 的行推到**目标 GPU** 的 `buffer` 对应槽，`nvshmem_quiet()`
后 `threadIdx.x==0` 发一条**信号**：

```cpp
nvshmemx_signal_op(reinterpret_cast<uint64_t *>(sig_addr),   // &all_event_counters[event_index]
                   1, NVSHMEM_SIGNAL_ADD, target_gpu_id);
```

`reduction` task（`tasks/ampere/allreduce.cuh:67-92` 的 `reduction_kernel`）在本 GPU 上把
`world_size` 份（本地的 `input` + 收到的 `buffer[0..world_size)`）求和到 `output`。注册代码
（`task_register.cc:4601-4660` / `:1007-1063`）把 host 侧绑定的关键量烘进模板/调用：
`event_index`/`target_gpu_id` 取自 `task_desc->trigger_event` 的解码
（`task_register.cc:4642-4646`），`sig_addr = &runtime_config.all_event_counters[event_index]`。

**时序**：GPU i 的 allgather subtask（target gpu=j）完成时，push 数据到 j 并用
`nvshmemx_signal_op` 给 j 的 `all_event_counters[event_index]` 加 1；j 上的 reduction task 的
dependent_event 正是这个 NVSHMEM event，`needed_counts = (num_gpus-1)*iter`，等到「所有别的 GPU
都送完」才开跑。这就是 `docs/16`/`docs/15` 未展开的**跨卡 event 全链路**。它对应快照论文 §6.5
行 1563-1567 的原话（*compiles these collective operators into two types of tasks: inter-GPU
data-transfer tasks, implemented using NVSHMEM's `nvshmem_signal_wait_until`, and local reduction
tasks*）。

### 6.2 策略 B：NVSHMEM Tile（NVLS one-shot，快照新增）

`AllReduceStrategy_NvshmemTile`（`multigpu.py:187-208`）只注册**一个** task：
`register_task(tb_graph, "nvshmem_tile_allreduce", params)`，并调
`allocate_nvshmem_teams(mpk, grid_dim[0]*grid_dim[1]*grid_dim[2])`——**每个 CTA 一个 team**
（`multigpu.py:74-96` 顺带设 `NVSHMEM_MAX_TEAMS` / 未文档化的 `NVSHMEM_MAX_CTAS=1`）。
host 侧 team 在 `init_persistent_kernel` 里用 `nvshmem_team_split_strided(WORLD, 0, 1, npes, ...)`
创建（`persistent_kernel.cuh:1731-1755`），拷进 `runtime_config.nvshmem_teams`；device 侧用
`task_metadata.task_offset` 索引（`task_register.cc:4703-4704`）。

两个 device 版本：

- **Hopper**（`tasks/hopper/allreduce.cuh:29-63`）：一行 `nvshmemx::tile_sum_reduce_block<...,
  NVLS_ONE_SHOT_PULL_NBI>(teams[task_offset], src, dst, ...)`——直接调 NVSHMEM 的 NVLS tile
  collective（从 multicast 地址 `ld_reduce` 拉回本 rank）。
- **Blackwell**（`tasks/blackwell/allreduce.cuh:461-514`）：**手写** NVLS——先
  `__threadfence(); mpkar_sync_block(team)`（散布式 barrier，`:312-355`），再对
  `mpkar_mc_ptr` 出的 multicast 地址发 `multimem.ld_reduce.global.add.{v4.bf16x2|v4.f16x2|v4.f32}`
  （`:380-450`），逐 `int4` 写回本地 HBM，末尾 `__threadfence()`。文件头注释（`:22-32`）解释动机：
  完整 `<nvshmem.h>` 会拖进几百个 `__noinline__` transfer 函数，`-rdc=true` 下把寄存器从 166 顶到
  255，**破坏 B200 上的 persistent-kernel 共驻**，故只 include 两个轻量类型头 + 内联那 ~30 行。

**Tile 路径对 MPK event 系统是透明的**：它是「普通本地 task」，跨卡同步全在 `mpkar_sync_block` +
`ld_reduce` 里完成，tGraph 看不到任何 NVSHMEM event。这解释了 §4.2 为什么它不打标签、也解释了
`runtime.cc:1236` 为什么说 `ALLGATHER_STRIDED_PUT` 那条 trigger 分支「*no longer used*」——
**现代（SM≥90）多卡走的是 Tile，AllGather+Reduce 更像保留的回退/兼容路径**。

### 6.3 两种实现的对照

| 维度 | A. AllGather+Reduce | B. NVSHMEM Tile (NVLS) |
|---|---|---|
| tGraph 里的 task 数 | 2 个算子（allgather + reduction） | 1 个算子 |
| 每个 bid 的 subtask | `num_gpus-1` 个（逐目标 GPU） | 1 个（整 team 内 collective） |
| 跨卡依赖表示 | **NVSHMEM event**（远端触发） | **task 内**（不进 tGraph） |
| 通信原语 | `putmem_nbi` + `nvshmemx_signal_op` | `multimem.ld_reduce`（one-shot pull NVLS） |
| 数据流向 | push 到对端 buffer，再本地求和 | 从 multicast 地址直接 pull-reduce |
| 适用 | 回退 / 非 NVLS / SM<90 | SM≥90 + VMM + multicast + peer |
| 论文是否描述 | 是（§6.5 行 1563-1567） | **否**（快照新增） |

## 7. 张量与 residual 的多卡正确性

多卡张量按 `io_category="nvshmem_tensor"` 走 `attach_nvshmem_tensor`
（`persistent_kernel.py:795-800`；`graph.cc:351-355` 记 `IODesc::NVSHMEMMallocTensor`），
`buffer` 形状是 `(world_size, batch, hidden)`（qwen3 builder `models/qwen3/builder.py:143-147`、
`:214-226`；allreduce 调用在 `:435-444`/`:550-553`）。

一个容易踩的正确性点：**TP 下残差只能加一次**。`persistent_kernel.py:2347-2361` 注释解释，
若某残差在**每个 rank 上持有相同值**（skip connection），则只能有一个 rank 加它，否则 allreduce
会把它**重复计 world_size 次**；反之若是「每 rank 的部分和」（row-parallel shared-expert 输出），
则每个 rank 都要加自己那份。实现用 `enable_residual`（`params[0]`，`:2359-2363` 与
`:2470-2474`）在 rank≠0 时置 0，靠「SM100 kernel 指针为空就跳过残差加」实现。这条对
`mega-moe` 专题的 MoE 反向/TP 语义是同一类坑（交叉引用 `../mega-moe/docs/`）。

## 8. host 侧 launcher：per-rank 编译与流

每个 rank 的 launcher 是一段 Cython 里嵌的 C 代码（`persistent_kernel.py:38-141`）：
`init_func` 把 `my_mpi_rank` 等透传给 `init_persistent_kernel`（`:121`），`launch_func` 调
`launch_persistent_kernel(stream)`（`:141`），`wait_func`/`finalize_func` 对应
`wait_persistent_kernel`/`finalize_persistent_kernel`。**每 rank 一份 `.so`、同一个 MPI job 内各跑
各的**。`__call__`（`:3574-3592`）只把 stream 指针交给 launcher——**host 侧没有任何 per-token
的通信调度**，跨卡通信完全在常驻 kernel 内由 event 驱动。这正是论文 §6.5 行 1575（*eliminating
CPU-side dispatch overheads*）的多卡版本。

## 9. 与论文对照

| 论文点 | 快照行号 | 实现落点 |
|---|---|---|
| 摘要：把 multi-GPU inference 编成单个 mega-kernel | 39、135 | `nvshmemx_collective_launch`（`persistent_kernel.cuh:2025`）+ 单网格 `persistent_kernel` |
| 通信与计算统一进同一 tGraph、同一 scheduler 派发 | 1219-1224 | `register_mugraph` 用同一 `all_events`/`all_tasks` 编 NVSHMEM task（`runtime.cc:644-659`） |
| 事件驱动、不依赖拓扑画像 | 1220-1224 | NVSHMEM event 由「数据到达」触发（`runtime.cc:129-150`）、`nvshmem_signal_wait_until`（`persistent_kernel.cuh:1097`） |
| §6.5：AllReduce → 数据传输 task + 本地 reduction | 1563-1567 | 策略 A（`multigpu.py:156-184`、`tasks/*/allreduce.cuh`） |
| §6.5：TP 8×H100 相对 PyTorch 最高 10×、相对 SGLang/vLLM 1.1–1.4× | 1569-1578 | 需多卡机器，本文不实测（`docs/01` §5 已归档该数） |
| §6.6：compute–communication overlap 1.1× | 1617-1619 | 机制 = 细粒度 event（`runtime.cc:371-386`）；单机替身实测见 `docs/04` §14 |
| §5.1：4 scheduler / 128 worker SM（单机） | 1157-1164 | 多卡每 rank 各一套（`docs/06` §4）；多卡未改划分 |

**最大落差**：论文只把 AllReduce 描述成「数据传输 + 本地归约」（策略 A），而快照里 SM≥90 的
主路径已经是 **NVLS tile one-shot**（策略 B，甚至 Blackwell 版手写内联以保共驻）——论文未提。
这与 `docs/15`/`docs/16` 反复出现的「实现领先于论文描述」是同一现象。

## 10. 判据 C1–C5 映射

`docs/00` §3 给定的五条判据下，多卡路径的贡献：

| 判据 | 多卡路径的贡献 | 说明 |
|---|---|---|
| C1 单次 launch | **强约束** | 多卡**只能**单 kernel（NVSHMEM collective launch 不支持 split，§2） |
| C2 常驻 | 是 | 同单卡，`persistent_kernel` 跨 token 常驻 |
| C3 多算子范围 | **扩展** | 通信 task（allgather/reduction 或 tile）与计算 task 编进**同一张图** |
| C4 设备内协调 | **跨卡延伸** | event 由远端 GPU 触发（策略 A）/ task 内 NVLS barrier（策略 B） |
| C5 SM 级调度 | 是（每 rank 独立） | worker/scheduler 机制不变；跨卡事件让「依赖通信的 task 尽早跑」 |

一句话：**多卡把 C3 的范围从「一张卡上的算子」扩到「跨卡的算子 + 通信」，把 C4 从「设备内」扩到
「设备间」，并且让 C1 从优化变成硬约束。**

## 11. 局限

- **纯源码走读、未跑多卡**：多卡需 ≥2 张 NVLink 卡 + NVSHMEM + MPI（`docs/10` 记录的容器只做过
  单卡 smoke，见 `docs/06`/`docs/09`），本机不具备；所有结论基于 `f9eb70c` 快照行号。
- **未走 `nvshmemx_collective_launch` 内部**：只确认它「顺序启动、要求协同 grid」，未逐行走 NVSHMEM
  的 launch 实现。
- **策略 A 的 device 代码有布局假设**：`nvshmem_allgather_strided_put` 逐行 put 依赖
  `input_map.x==1`、buffer 多一个 `world_size` 维（`task_register.cc:4625-4638`），换布局会失效
  （源码已 assert）。
- **`allocate_nvshmem_teams` 的代价未测**：Tile 每 CTA 一个 team，team 数随 grid 线性增长
  （`multigpu.py:74-96` 明说「GPU 越多越慢」），本文只描述机制、未量化。
- **`docs/15` 另两条尾巴（transpiler SASS 级生成、`benchmark/` 性能数）仍未覆盖**。
- **多卡通信-计算重叠只有单机替身**：`docs/04` §14（P20-1）用「producer/consumer 分块并发」模拟
  跨算子细粒度重叠，量出静态切 block 的重叠**反而更慢**、载体必须是 SM 内共驻——那对「跨卡通信
  与计算重叠」是**机制类比而非同一实验**。

## 参考坐标

- 仓库：`https://github.com/mirage-project/mirage`（分支 `mpk`，commit `f9eb70c`）。
- host：`python/mirage/mpk/mpk.py`（`init_mpi:298-312`、`generate_task_graph:482-489`、`__del__:513-516`）、
  `python/mirage/mpk/multigpu.py`（能力探测 `:24-72`、team 分配 `:74-96`、策略 A `:156-184`、
  策略 B `:187-208`、选择 `:224-247`）、`python/mirage/mpk/persistent_kernel.py`
  （`use_nvshmem:523`、`attach_nvshmem_tensor:795-800`、`reduction:1007-1063`、`allreduce_layer:2488-2510`、
  `compile/generate_task_graph:3207-3241`、launcher C 代码 `:38-141`）。
- C++：`src/kernel/runtime.cc`（`get_event_id:30-36`、`get_num_subtasks:66-74`、
  `dfs_create_events_add_tasks:76-221`、`set_producer_triggers:604-639`、`layer_is_multigpu:657-659`、
  `task_offset:442-446`、`generate_task_graph:1915-1949`）、`src/kernel/task_register.cc`
  （allgather `:4601-4660`、tile `:4662-4707`、reduction `:1007-1063`）、
  `src/kernel/graph.cc:351-355`。
- device：`include/mirage/persistent_kernel/persistent_kernel.cuh`（event 编解码 `:130-148`、
  worker 等事件 `:1084-1110`、触发 `:1156-1249`、NVSHMEM init `:1720-1755`、launch `:1955-2043`、
  finalize `:2056-2110`）、`include/mirage/persistent_kernel/runtime_header.h:75-80`、
  `tasks/ampere/allreduce.cuh`（`nvshmem_allgather_strided_put:31-58`、`reduction_kernel:67-92`）、
  `tasks/hopper/allreduce.cuh:29-63`、`tasks/blackwell/allreduce.cuh`（NVLS `:377-514`、barrier `:145-355`）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
  （摘要行 39；§5.1 行 1195-1224；§6.5 行 1555-1578；§6.6 行 1617-1619）。
- 交叉引用：`docs/06`（运行层与双 kernel 形态）、`docs/07`（单 task device）、`docs/08`（依赖分析器
  的 event 生成，多卡事件是它的一个分支）、`docs/04` §14（跨算子重叠单机实测）、`../mega-moe/`
  （DeepGEMM MoE 的 TP/residual 语义）。
