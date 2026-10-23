# 生成的 megakernel 怎么跑起来：worker、scheduler 与事件队列

> 这是「Mirage 项目分析」的第 3 篇。`compile()` 之后我们拿到一颗常驻 kernel，但它内部到底谁在
> 干活、谁在派活、依赖怎么保证？这一篇把 `persistent_kernel.cuh` 的骨架拆开讲。

## 0. 先记住一个画面

把 GPU 想成一个**物流仓库**。

- **worker** 是搬运工：每人守一条传送带，从上面取包裹（task），等齐配件（依赖事件），拆包干活
  （执行 kernel），干完在单子上盖章（触发后继事件）。
- **scheduler** 是调度员：他不搬货，只盯着「哪些单子盖满了章」，然后把下一批包裹塞到搬运工的
  传送带上。

整个仓库**没有中央控制室**（host）：所有指令都写在地上的一块白板（显存里的环形队列 + 计数器），
搬运工和调度员靠**看一眼白板 + 内存屏障**来协作。

> **类比的边界**：现实里「看一眼」是原子的，GPU 上不是——白板读写要靠 `release`/`acquire` 语义
> 保证顺序，否则调度员可能看到「单子盖满了章」却还没看到包裹内容。这套内存序协议是本篇后半的
> 重点，也是 MPK 源码里最容易读错的地方。

读完这一篇，你应该能回答：

1. 一颗 grid 里，哪些 block 是 worker、哪些是 scheduler？H100 上各多少？
2. worker 取一个 task 要经过哪四步？每一步用哪条内存序指令？
3. 一个事件「就绪」的判定条件是什么？为什么必须带 iteration 号？

## 1. 角色与拓扑：谁在干活

一颗 kernel 里有两类 block：

| 角色 | 粒度 | 干什么 | 数量（H100，132 SM） |
|---|---|---|---|
| **worker** | 一个 CUDA block | 执行 task | **128** |
| **scheduler** | 一个 warp（4 warp 一个 block） | 派发 task | **16**（= 4 个 SM） |

比例是写死的：`python/mirage/utils.py:25-31` 给出 `scheduler = 4 * (sm_cnt - worker)`，
即**每个 scheduler 占 4 个 warp、正好 1 个 SM**，剩下的 SM 全做 worker。H100 的 132 个 SM
于是分成 128 个 worker + 4 个 scheduler SM（16 个 scheduler warp）。

> 这也解释了 §1 里那个 API 约束：`num_workers + schedulers/4 = SM 数`。填错就会出现
> 「有的 SM 没角色、有的角色没 SM」的错配。

## 2. 三个数据结构，先认识再读循环

运行时只有三样东西在流动：**task 描述符**、**event 描述符**、**队列游标**。

### 2.1 ID 是 64 位打包的

`runtime_header.h:71-80`：

- `TaskId`：高 32 位 = **iteration（第几轮解码）**，低 32 位 = task 序号。
  同一张静态图会被多轮解码复用，靠 iteration 区分「这一轮的 task 3」和「上一轮的 task 3」。
- `EventId`：`nvshmem_tag:16 | owner_node:16 | event_idx:32`。单卡时只看低 32 位。

### 2.2 task 描述符有两个版本

| 结构 | 何时用 | 关键字段 |
|---|---|---|
| `FullTaskDesc`（`:260-290`） | 编译期，信息全 | `trigger_event`（完成后触发谁）、`dependent_event`（等谁）、输入输出张量、8 字节 metadata union |
| `TaskDesc`（`:296-336`） | 运行期，要小要快 | 张量压成裸指针 `input_ptrs[8]` / `output_ptrs[3]`，`alignas(16)` 以便 128-bit `cp.async` 整块搬 |

为什么运行期要另一个结构？因为 worker 每次取 task 都要把描述符搬进 shared memory，
**16 字节对齐 + 固定 8/3 个指针槽**能让这次搬运是一条 `cp.async` 128b 指令，而不是逐字段解析。

### 2.3 事件：等到「够次数」才算就绪

`EventDesc`（`:249-258`）只有四个字段：

```cuda
struct EventDesc {
  EventType event_type;
  int num_triggers;    // 需要被触发多少次
  int first_task_id;   // 就绪后放行哪个区间的 task
  int last_task_id;
};
```

关键在 `num_triggers`：一个事件可能被**多个**上游 task 触发（比如 8 个分块 GEMM 都写同一块
输出），必须集齐 `num_triggers` 次才算就绪。这就是「join」的语义。

事件类型（`:228-236`）分五档：`EVENT_LAUNCH_TASKS`（普通）、`EVENT_LAUNCH_MASSIVE_TASKS`
（大规模，跨 scheduler 切区间）、`EVENT_LAUNCH_DEPENDENT_TASKS`（有依赖，单 scheduler 内轮转）、
`EVENT_END_OF_TASK_GRAPH`（一轮结束）、`EVENT_TERMINATION`（收工）。

## 3. worker 循环：取货 → 等配件 → 干活 → 盖章

`execute_worker`（`persistent_kernel.cuh:957-1252`）是一个 `while(true)`，四步。

### 第一步：取 task（`:1009-1081`）

worker 先看自己队列的游标 `worker_queue_last_ready_task_id`：

```cuda
// 伪代码，对应 :1016-1025
while (ld.acquire.gpu(worker_queue_last_ready_task_id) == my_pos) __nanosleep(10);
```

游标前进了，说明有新 task。然后：

1. 用 `ld.relaxed.gpu` 读 task id（`:1036`）——**为什么不是 acquire？** 因为后面紧跟一条
   `cp.async` 把整块描述符搬进 smem，搬运本身有 fence（`:1063-1078`）；
2. 用 **128-bit `cp.async`** 把 `TaskDesc` 拷进 shared memory（`:1063-1078`）。

这就是论文 §5.3 的「task descriptor 预取 + 批量加载」：**一次搬一条 cache line，而不是逐字段读**。

### 第二步：等依赖（`:1084-1111`）

如果这个 task 有 `dependent_event`，worker 就自旋等计数器涨到阈值：

```cuda
needed = all_event_num_triggers[event_index] * get_task_iteration_num(task_id);  // :1090-1093
while (ld.acquire.sys(all_event_counters[event_index]) < needed) __nanosleep(10);  // :1106
```

两个细节：

- **`acquire.sys` 而不是 `acquire.gpu`**：事件计数器可能被别的 GPU 写（多卡），要用 system scope。
- **乘 iteration**：阈值 = 本事件每轮触发次数 × 当前迭代号。这样「第 3 轮的 task」不会因为
  「第 1 轮已经触发过」而误判就绪——**跨代隔离**。

### 第三步：执行（`:1121-1146`）

- `TASK_TERMINATE`：直接 `return`，worker 退出；
- `TASK_BEGIN_TASK_GRAPH`：空操作（它只是个「起跑信号」）；
- 其余：调 `_execute_task(task_desc, config)`——就是上一篇里那个巨大的 if-else 派发器。

### 第四步：盖章，触发后继（`:1155-1249`）

执行完，`threadIdx.x==0` 对 `all_event_counters[event_index]` 做一次
`atom.add.release.gpu.u64(..., 1)`（`:1164`）。如果加完后正好等于
`num_triggers * iteration`（`:1185`），说明**这个事件集齐了**，于是：

1. 选一条 scheduler 队列（`:1201-1212`）：普通事件随机分给某个 local scheduler；
   `MASSIVE`/`DEPENDENT` 走共享的广播队列（`:1291`）；
2. `atom.add.release.gpu` 取一个槽位写 event id（`:1213-1218`）；
3. 用 **CAS-release** 推进 `sched_queue_last_ready_event_id`（`:1222-1228`）。

顺序很重要：**先写数据，再放行读游标**。这就是 release 语义的作用。

## 4. scheduler 循环：等单子 → 按类型派活

`execute_scheduler`（`:1254-1523`）是一个 warp 的循环：

1. **等事件**（`:1334-1354`）：`ld.acquire.gpu` 读 `sched_queue_last_ready_event_id`，
   `__nanosleep(10)` 自旋；
2. **读事件**（`:1357-1360`）：`ld.relaxed.gpu` 取 event id（因为游标已经 acquire 过，
   数据一定可见）；
3. **按类型派发**：
   - `EVENT_TERMINATION`（`:1361-1376`）：给所有 worker 队列写终止哨兵；
   - `EVENT_END_OF_TASK_GRAPH`（`:1378-1423`）：调 `prepare_next_batch(config)` 组下一批请求；
     有活就下发**下一迭代**的 `TASK_BEGIN_TASK_GRAPH`（`:1400-1409`），没活就收工；
   - `EVENT_LAUNCH_DEPENDENT_TASKS`（`:1424-1474`）：把 `[first, last)` 区间的 task
     **轮流**塞给 `my_first_worker..my_last_worker`；
   - 其余（`:1475-1519`）：普通事件按 `sched_id` 顺序派发；MASSIVE 事件先在 local scheduler
     之间切分区间（`:1477-1487`）。

每次派发都是同一种写法：`st.relaxed.gpu` 写槽位 → `atom.add.release.gpu` 放行
`worker_queue_last_ready_task_id`（`:1495-1502`）。

> 所以「调度」在这里没有全局表：scheduler 只根据事件类型和本地游标做决定，worker 只根据
> 自己的队列做事。这就是论文说的**去中心化 in-kernel 调度**。

## 5. 启动种子：一颗 kernel 怎么「点火」

`launch_persistent_kernel` 每次会先跑一个 `prepare_kernel<<<num_workers,128>>>`（`:1963-1967`），
它做三件事：**重置所有队列游标、清零事件计数器、把 `EVENT_END_OF_TASK_GRAPH` 预置到
scheduler[0] 的队列**（`prepare_kernel` 的 `:226-233`）。

这一颗预置事件就是**点火器**：

```
END_OF_TASK_GRAPH 就绪
   → scheduler 调 prepare_next_batch，下发 TASK_BEGIN_TASK_GRAPH（第 1 轮）
   → 它触发后继事件 → scheduler 把第一批 task 派给 worker
   → ... 整图跑起来 ...
   → 本轮结束又产生 END_OF_TASK_GRAPH → 下一轮
```

两种启动方式（`RuntimeConfig.split_worker_scheduler`，`:1977`）：

- **分开**：`worker_kernel` 和 `scheduler_kernel` 在两条 stream 上并发常驻；
- **合并**：一颗 `persistent_kernel`，`blockIdx.x < num_workers` 的跑 worker，其余跑 scheduler。

## 6. 内存序：为什么到处是 `relaxed` + `release/acquire`

`mpk_atoms.cuh:17-96` 手写了六条 PTX 原语：`atom.add.release.gpu`、
`atom.cas.release.gpu`、`ld.acquire.gpu`、`ld.acquire.sys`、`ld.relaxed.gpu`、`st.relaxed.gpu`。

协议只有一条：

> **写数据用 relaxed，推游标用 release；读游标用 acquire，读数据用 relaxed。**

直觉：游标是「门」，数据是「门后的货」。release 保证「货先放好，门再开」；acquire 保证
「看到门开了，门后的货一定看得见」。如果反过来（先推游标再写数据），worker 可能拿到一个
还没写完的 `TaskDesc`——这种 bug 在 GPU 上极难复现。

## 7. 与论文的三处差异

源码比论文多走了三步：

1. **事件分三档**（普通 / MASSIVE / DEPENDENT），对应不同的扇出策略：论文只说「一个事件放行
   一批 task」，工程上必须决定「这批 task 怎么分给多核」。
2. **`RuntimeConfig` 里塞满了 serving 元数据**（step/tokens/paged-KV 表，
   `runtime_header.h:338-459`），说明线上实现早就超出论文「离线固定 batch」的设定。
3. **`split_worker_scheduler` 开关**：既能单 kernel 合并，也能 worker/scheduler 双 kernel 分
   stream 跑——后者更接近传统 persistent kernel + 独立调度器的形态，便于调试。

## 8. 三句话小结

- 一颗 kernel 里，worker（128 个 block）干活、scheduler（16 个 warp）派活，靠显存里的环形队列通信。
- worker 四步：取 task（128b `cp.async`）→ 等事件（带 iteration 的阈值）→ 执行 → release 盖章。
- 正确性靠「relaxed 写数据 + release 推游标 / acquire 读游标」这一条内存序协议。

下一篇换一个角度：不再讲「应该怎样」，而是记录我们真的把它编出来、跑起来时遇到的坑。
