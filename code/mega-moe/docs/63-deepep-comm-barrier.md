# DeepEP 的多 rank 屏障与 symmetric-memory 句柄层：`gpu_barrier` / `NCCLGin`（H100 实测协议）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P53-1** 增量（第 63 篇）。
>
> 前面第 7 篇（`docs/07`）拆了 dispatch / combine 的**数据流**，第 43 篇拆了 EPLB 反向通信，
> 第 48 篇拆了 combine 的归约 epilogue，第 49 篇拆了 DeepEP 的**设备端公共层**（`common/ptx.cuh` 等），
> 第 51/52 篇拆了 dispatch 的拷贝 epilogue 与确定性排序。但这些 kernel 在每次运行的**开始与结束**
> 都调用同一个东西——`deep_ep::comm::barrier`（`deep_ep/comm/barrier.py` → `_C.barrier` →
> `csrc/comm/barrier.hpp` → device `impls/comm/barrier.cuh`）。它把**一整个 EP 通信组**（可跨节点）
> 拉到同一个同步点，是 dispatch / combine / bucket collective 正确性的地基。
>
> 这一层由三部分构成：
> 1. **symmetric-memory 句柄** `NCCLGin`（`comm/handle.cuh`）——把「本 rank 的某个 offset」翻译成
>    「目标 rank 的同一 offset」的对称指针，并封装 RDMA（GIN）的 put/get/signal/flush；
> 2. **屏障协议** `comm/barrier.cuh` —— 两级域（scaleup NVLink / scaleout RDMA）屏障 + `get_qp_mode` 的
>    SM↔QP 映射 + `timeout_while` 看门狗；
> 3. **host 编排** `csrc/kernels/comm/{context.cpp,barrier.hpp}` —— 域分解、GIN 需求、JIT 发射。
>
> **可跑性**：NVLink 腿 `nvlink_barrier_wo_local_sync` 本身只由架构无关内联 asm 构成
> （`red.release.sys` / `ld.acquire.sys` / `atomicAdd`），与 NCCL 相关的只有 `NCCLGin::get_sym_ptr`
> 这个**纯地址翻译**。故本篇把它**逐行原样抽取**、配一个语义等价的对称指针桩，在 H100（sm90）上
> 用多 block 模拟多 rank 跑**真实设备代码**；`get_qp_mode` 直接为 host/device 双态，逐字段对拍。
> scaleout 腿与 `gpu_barrier` 的 RDMA 编排用 `ncclGin` / `ncclDevComm`（本机 NCCL 2.29.7 头不提供
> GIN 设备 API）→ **仅源码分析**（同 ROADMAP「阻塞」）。
>
> 源码坐标：`~/github/DeepEP`，公开版 commit `93eb6eb`（v1.2.1-41）。仓库：
> [github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP)。

## 1. `NCCLGin`：一个对称窗口句柄

`comm/handle.cuh:11-35` 的 `NCCLGin` 是一次 `ncclCommWindowRegister` 注册出来的
**对称窗口**（`ncclWindow_t`）的封装。它构造时取三样东西：

- `team_world / team_lsa / team_rail`（`:32`）——NCCL 的三个通信域：
  `world`=全部 rank、`lsa`（Local SHARP Aggregation / 节点内 NVLink 域）、`rail`（跨节点同号卡）；
- `lsa_base_ptr = ncclGetLsaPointer(nccl_window, 0, team_lsa.rank)`（`:34`）——**本 rank** 在
  NVLink 域内的窗口基址；
- 一个 `ncclGin gin`（`:31`，QP index + sharing mode）。

**核心是 `get_sym_ptr`（`:64-92`）**：给定本地指针与目标 rank，返回目标 rank 上同一 offset 的地址。

```
handle.cuh:59-61   get_sym_offset(ptr) = reinterpret_cast<uint64_t>(ptr) - lsa_base_ptr
handle.cuh:88-90   ncclGetLsaPointer(nccl_window, get_sym_offset(ptr), dst_nvl_rank_idx)
```

也就是说**「对称」= 每个 rank 的窗口布局完全一致，于是「同一个相对 offset」在所有 rank 上代表
同一逻辑位置**。`get_sym_ptr` 对 `team_t` 分三类（`:67-92`）：

- `ncclTeamTagRail`：只有「正好是本 rail rank」才返回本地指针，否则 `nullptr`（`:67-69`）；
- `ncclTeamTagWorld`：先用 `is_nvlink_accessible`（`:37-54`）判断目标是否落在本 rank 所在的 NVLink
  域里；不可达返回 `nullptr`，可达则把 world rank 折算成 NVLink rank（`:79-80`）后查表；
- `ncclTeamTagLsa`：直接查表（`:84-90`）。

`red_add_rel`（`:96-120`）演示了这套设计如何**统一「本地/远端」两条路径**：拿到对称指针后，
**本地或同 NVLink 域**用 `ptx::red_add_rel_sys/gpu` 直接打到对称内存；指针为 `nullptr`（跨节点）
才 fallback 到 `gin.signal(...)`（RDMA 原子）。同一套 API `put`/`put_value`/`get`/`signal`/`wait`
都如此——**「能走 NVLink 就走 NVLink，否则走 RDMA」**。

### 1.1 我们用到的等价桩

实验里只需 `get_sym_ptr`。桩的定义（`src/deepep_comm_barrier.cu`）与真实语义一一对应：

```
offset = ptr - local;                 // 相对本 rank signals 的偏移
return &rank_signals[dst] + offset;   // 目标 rank 的同一 offset
```

这与 `ncclGetLsaPointer(window, offset, dst)` 完全同义，只是把「对称窗口」换成了一个显式的
per-rank 数组。

## 2. 屏障协议 `comm/barrier.cuh`

### 2.1 `BarrierSignals`：计数 + 双相位，免复位

`layout/common/barrier.cuh:11-25`：

```
struct alignas(kNumRDMAAlignmentBytes) BarrierSignals {   // 64B 对齐
    unsigned long long nvl_barrier_counter;               // 8B，仅用低 2 位
    int nvl_barrier_signals[2];                           // 两个相位槽
};
```

`alignas(64)` 把 16 字节的有效载荷补齐到 **64B**（= `kNumRDMAAlignmentBytes`，`compiled.cuh:68`），
保证一次系统域原子访问落在一个 cache line 内。**只用两个 `int` 槽 + 一个计数器**，就把一个可
无限复用的多 rank 屏障编码完了。

### 2.2 NVLink 腿 `nvlink_barrier_wo_local_sync`（`:88-131`）

这是整个屏障里唯一**不需要 RDMA** 的路径（同节点内 rank）。逐段看：

```
:94-95    if (kNumSMs > 1 and sm_idx > 0) return;     // 只用 1 个 SM
:98-100   status = (*counter) & 3; phase = status & 1; sign = status >> 1;
:101      signal_ptr = signals.get_nvl_barrier_signal_ptr(phase);   // 选一个槽
:103-106  for i in ranks: red_add_rel_sys(signal_ptr@rank_i, sign ? -1 : +1);
:107      __syncthreads();
:111-113  if (thread_idx==0) atomicAdd(counter, 1); __syncwarp();
:116-129  target = sign ? 0 : kNumRanks;
          timeout_while<...>(thread_idx==0, 轮询 ld_acquire_sys(signal_ptr) == target);
```

**「免复位」的根因就是 `sign ? -1 : +1` + 双槽**。设 rank 数 $N$，把每次调用读到的
`counter` 记为 $k$（$\text{status}=k\bmod 4$），则：

| 调用 $k\bmod 4$ | phase | sign | 对 slot 的操作 | 等待目标 |
|---|---|---|---|---|
| 0 | 0 | 0 | slot0 += 1（每 rank 一次，共 $+N$） | $N$ |
| 1 | 1 | 0 | slot1 += 1（共 $+N$） | $N$ |
| 2 | 0 | 1 | slot0 += −1（共 $−N$，回到 0） | 0 |
| 3 | 1 | 1 | slot1 += −1（共 $−N$，回到 0） | 0 |

所以**四个调用构成一个循环，两个槽在循环末尾恰好回到 0——永远不需要显式清零**。这比
「每轮把计数器重置」省一次全局同步，也避免「复位本身需要同步」的鸡生蛋问题。计数器高位不受
影响：`:109` 的注释算出要 $2^{64}/(10^6\cdot3600\cdot24\cdot365)\approx 57$ 万年才会溢出。

等待用 `ptx::ld_acquire_sys`（系统域 acquire 读）自旋，配合发送侧的 `red.release.sys`
（系统域 release 原子加）——这是**跨卡**可见性最小充分语义（见第 49 篇内存序阶梯）。

### 2.3 看门狗 `timeout_while`（`:30-54`）

`timeout_while<kNumTimeoutCycles>(condition, func, start_clock)`：条件为真时循环调用 `func(timeout)`，
`func` 返回 `true` 就退出；若已超时，先空转 1 秒（`kNumOneSecCycles = 2e9`，`:13`）让所有线程打完
诊断，再 `ptx::trap()`（`:46`）。屏障传的 `condition` 是 `thread_idx==0`，`func` 里轮询信号——
于是**只有 leader 线程自旋**，其余线程直接落过（后面靠 `gpu_barrier` 的 grid sync 收拢）。

### 2.4 `get_qp_mode`：SM↔QP 的确定映射（`:56-86`）

指向 `ncclGin` 的 QP（queue pair）资源是全 rank 共享的稀缺资源，`get_qp_mode<kNumSMs, kNumQPs,
kNumChannelsPerSM, kWithNotifyWarps>` 把它按两种策略映射：

- **`kNumSMs <= 可用 QP 数`**（每个 SM 独占若干 QP，`:73-80`）：
  `qp = kQPStartIdx + sm_idx + (channel % num_qps_in_sm) * kNumSMs`，sharing = **CTA**。
  注释的例（3 SM / 10 QP）：`SM0:0,3,6,9  SM1:1,4,7  SM2:2,5,8`——**round-robin 摊开**。
- **QP 不够分**（所有 SM 共享全部 QP，`:81-85`）：`qp = kQPStartIdx + (global_channel % 可用数)`，
  sharing = **GPU**。
- 特例：`kNumQPs==1` 恒 QP0（`:63-64`）；notify warp 恒 QP0 + CTA（`:67-68`，`:71` 的
  `kQPStartIdx = kWithNotifyWarps` 把 QP0 留给它）。

## 3. 两级域编排 `gpu_barrier`（`:196-252`）

真实集群同时有两条腿：**scaleup（节点内 NVLink，`ncclTeamTagLsa` 相同域）** 与
**scaleout（跨节点 RDMA，`ncclTeamTagRail/World`）**。`gpu_barrier` 把两者拼起来：

```
:207-211  if kFlushStores: ptx::tma_store_commit(); ptx::tma_store_wait();  // 先冲干净 TMA store
:214-215  if kSyncAtStart: cooperative_groups::this_grid().sync();          // 全 grid 对齐
:220-221  do_scaleout &= kNumScaleoutRanks > 1;  do_scaleup &= kNumScaleupRanks > 1;
:222-238  两条腿都做：SM0 做 scaleup，其余 SM 做 scaleout（并行），再补一次 grid sync
:239-247  只做一条腿
:250-251  if kSyncAtEnd: cooperative_groups::this_grid().sync();
```

两条关键工程细节：

1. **先 flush TMA store**（`:207-211`）。屏障发布「我到了」之前，必须让**之前发出的 TMA store
   对远端可见**，否则对端看到信号却读不到数据。这就是 `kFlushStores` 模板参数的作用。
2. **scaleup/scaleout 用不同 SM 并行**（`:225-238`）：`kNumSMs >= 2` 时 SM0 专做 NVLink 腿、其余
   SM 做 RDMA 腿，避免两级串行拖慢；代价是中间多一次 `this_grid().sync()`（`:232-233` 注释自认
   "kind of hacky"）。当 `kFlushStores && 混合` 时，还额外要求 `kSyncAtEnd`。

三级腿函数只是薄封装：`scaleup_barrier_wo_local_sync`（`:171-184`）按 `kIsScaleupNVLink` 选
NVLink 腿或 world RDMA 腿；`scaleout_barrier_wo_local_sync`（`:186-194`）走 rail RDMA 腿。

### 3.1 RDMA 腿 `gin_barrier_wo_local_sync`（`:133-169`）——**仅分析**

- **flush QP**（`:147-159`）：`kFlushStores` 时每个 warp 分摊 `nccl_dev_comm.ginContextCount` 个 QP
  调 `gin.flush()`；world 团队还要一次 `ptx::fence_acq_rel_sys()`（`:154-155`，因为 world 屏障会把
  直接 NVLink 写和 GIN 信号混在一起）；随后 `grid sync`。
- **signal/wait**（`:161-168`）：只用 QP0，在**每个 rank 的 slot `rank_idx`** 上 `signal(SignalInc)`，
  再等**每个 rank 的 slot** 到达。这是经典的「all-to-all 信号 + 等待」——$O(N)$ 信号但每 rank 一个
  计数器槽，无需树形归约。

### 3.2 `barrier_impl` 与顺序模式（`impls/comm/barrier.cuh`）

设备入口 `barrier_impl`（`:11-18`）把模板参数打包成 `gpu_barrier` 调用。`kSequential` 时
**scaleout 先、scaleup 后**（`:23-34`），后者需要 flush 前者发出的 RDMA 请求（`:31` 注释）；
非 sequential 则走并行混合（`:35-40`）。

## 4. host 编排：域分解与 GIN 需求

### 4.1 域分解（`context.cpp:52-63 / 174-183`）

`Context` 构造时从 NCCL 拿到物理域大小，再按 `allow_hybrid_mode` 定义逻辑域：

```
:52-57   get_physical_domain_size: (num_ranks / num_nvl_ranks, num_nvl_ranks)   // (RDMA, NVLink)
:59-63   get_logical_domain_size : hybrid ? (RDMA, NVLink) : (1, all)
:174-183 if hybrid: num_scaleout=num_rdma, num_scaleup=num_nvl
         else      : num_scaleout=1,        num_scaleup=num_ranks
         is_scaleup_nvlink = (num_scaleup == num_nvl)     // scaleup 腿是否走 NVLink
```

要紧的是 `is_scaleup_nvlink`：**只有在 hybrid 模式且 scaleup 域正好等于 NVLink 域时才走 NVLink**；
否则 scaleup 腿被放大到「所有 rank」，从而**只能走 RDMA**（`barrier.cuh:177-183` 的 else 分支
`gin_barrier_wo_local_sync<..., ncclTeamTagWorld, ...>`）。这解释了为什么「关掉 hybrid」会把节点内
通信也变成 RDMA。

### 4.2 GIN 需求与信号预算（`context.cpp:124-155`）

```
:140  reqs.ginSignalCount = num_ranks + 2 * 2;
```

这个公式直接对应屏障的设计：**RDMA 腿需要 `num_ranks` 个 signal 槽**（每 rank 一个），
**NVLink 腿的 2 个相位槽各需要一个「影子」GIN 信号**（`2 * 2`）。`:125-146` 还设定了
`ginType=GDAKI`、`ginContextCount=num_allocated_qps`、`ginExclusiveContexts`、以及
`CUSTOM_STRIDE / RAIL / FULL` 三种连接类型。`:156-164` 处理 NCCL 版本分叉：≥2.31 用 runtime
version 分配 `devCommRuntimeVersionSize` 字节，否则要求编译期与运行期版本一致（**正是本机
NCCL 2.29.7 撞上的那堵墙**，见 ROADMAP 阻塞）。`:118-119` 在 `enable_lsa_multimem` 时断言
`props.multimemSupport`。

### 4.3 Python → C++ 的最短路径

- `deep_ep/comm/barrier.py:6-11`：`barrier(buffer, wait_comm_stream, with_cpu_sync, sequential,
  context_idx)` → `_C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)`；
- `csrc/comm/handle.hpp:16-22`：pybind 注册，断言 `context->barrier_signals != nullptr`，用当前
  CUDA stream 调 `barrier(...)`；
- `csrc/comm/barrier.hpp:9-35`：`barrier()` 先按 `with_cpu_sync` / `wait_comm_stream` 与
  comm stream 对齐，再 `launch_barrier`，之后再对齐一次（把屏障夹在正确的 stream 顺序里）；
- `csrc/kernels/comm/barrier.hpp`：`launch_barrier` 里 `kNumThreads=512`、
  `num_sms = (not sequential and num_scaleout_ranks > 1) ? 2 : 1`（**只有并行混合才需要 2 个 SM**），
  JIT 编译微源码后 **`cooperative = true`** 发射（对应 `this_grid().sync()`）。

用在哪里：`csrc/buffers/ep.hpp:96-106`（buffer 析构时 `comm::barrier(..., true, true)` 把
「工作全部结束」同步掉）、`bucket.hpp:317-331`、`engram.hpp`、`pp.hpp`。注意 `ep.hpp:93-95`
的注释——**buffer 构造结束时不能用自家 barrier**（workspace 还没就绪），要用户自己调 Python
`dist.barrier`。

## 5. H100 实测（协议层真实设备代码）

`src/deepep_comm_barrier.cu`（`harness/exp_deepep_comm_barrier.sh`）在 H100 上：

- **Part 1 NVLink 协议（真实设备代码 + 对称指针桩）**：把 `nvlink_barrier_wo_local_sync` 与
  `timeout_while` 逐行抽取，用 `kNumRanks` 个 block 模拟 `kNumRanks` 个 rank（每 block 一个
  rank），块间靠对称指针桩互写信号槽。测 `ranks ∈ {2,4,8,16,32,64,128}` × `8` 次调用，
  再测 `sms ∈ {2,4}` 的 **SM gating**（只有 `sm_idx==0` 参与），外加奇数次调用
  （5/7/1 次）验证信号槽**留下非零残值**、与 host 四相位模型**逐字段 0 不符**。
- **Part 2 `get_qp_mode`**：device 与 host 双态函数，5 组配置（含 `kNumQPs==1`、notify warp、
  两种 sharing 分支）打印/核对映射。
- **Part 3 `BarrierSignals`**：`sizeof==64 && alignof==64`。

Python 端（`harness/exp_deepep_comm_barrier.py`）独立复刻 §4.1 的域分解（含 `is_scaleup_nvlink`）、
`ginSignalCount = num_ranks + 4`、`num_gpu_timeout_cycles`、`get_qp_mode`、四相位模型，并**读设备
写出的 dump 逐字段交叉核对**。

实测结果（`harness/exp_deepep_comm_barrier.out.txt`）：

```
设备： PASS=19 FAIL=0
  Part1  ranks=128 sms=1：8 次后 counter=8 slot0=0 slot1=0；ranks=64 iters=7 -> slot1=64；
         ranks=8 iters=5 -> slot0=8；ranks=16 iters=1 -> slot0=16（全部 = 模型值）
  Part2  sms=3 qps=10 ch=4 -> SM0 QP[0,3,6,9]；sms=4 qps=2 ch=8 -> QP[0,1,0,1]（GPU sharing）；
         sms=8 qps=8 notify=1 -> notify 恒 QP0/CTA
  Part3  sizeof=64 alignof=64
Python：PASS=17 FAIL=0
  ranks=64 nvl=8 hybrid=0 -> logical=(1,64) scaleout/up=(1,64) is_scaleup_nvlink=0 ginSignals=68
  get_qp_mode 5 组逐字段与设备一致；barrier 12 个用例 counter/双相位槽与模型逐字段一致
```

**结论**：屏障协议「计数低 2 位选相位/符号 + ±1 写双槽」是一条**架构无关、免复位、O(1) 状态**的
多 rank 同步；`NCCLGin` 用「对称窗口 + 同 offset」把 NVLink 与 RDMA 两条腿统一在一套指针语义下。
协议层（NVLink 腿 + QP 映射 + 信号布局）已由真实设备代码逐字段验证；RDMA 腿与 `gpu_barrier` 的
并行混合编排仍是 **NCCL GIN 专属，仅源码分析**。

## 6. 与 DeepGEMM 自研屏障的对照

| | DeepGEMM `comm/barrier.cuh`（第 4/9/49 篇） | DeepEP `comm/barrier.cuh`（本篇） |
|---|---|---|
| 触发 | `grid_sync` + `nvlink_barrier`（Mega MoE 内部） | 每个 buffer 操作**开始/结束**各一次 |
| 域 | 单 rank 内 grid + NVLink 域 | **scaleup NVLink + scaleout RDMA 两级** |
| 信号 | 加/减式相位信号（免复位，第 4 篇） | 同思想：计数低 2 位 + ±1 双槽（免复位） |
| 传输 | DeepGEMM `SymBuffer` 直接 NVLink 写 | `NCCLGin` 对称窗口 + GIN RDMA |
| QP | 无（不碰 NIC） | `get_qp_mode` 显式把 SM/通道映射到 QP |
| 看门狗 | `clock64` 超时 | `timeout_while` + `trap()`（同一套思想） |

两者都把「跨 rank/跨 SM 的到达计数」压进**极少量状态 + 相位翻转**里，这正是第 4/49 篇反复出现的
「免复位相位」思想在 DeepEP 侧的完整版。

## 7. 小结与下一步

- `NCCLGin`（`handle.cuh:11-255`）是「对称窗口 + 同 offset 翻译 + NVLink/RDMA 双路径」的句柄层，
  `get_sym_ptr`（`:64-92`）是它的原子单元，`red_add_rel`（`:96-120`）展示「能 NVLink 就不走 RDMA」。
- `gpu_barrier`（`barrier.cuh:196-252`）先 flush TMA store、再并行做两级域屏障；
  `nvlink_barrier_wo_local_sync`（`:88-131`）是免复位双槽协议，`get_qp_mode`（`:56-86`）把 QP
  按 SM/通道 round-robin（独占）或取模（共享）分配，`timeout_while`（`:30-54`）是看门狗。
- host 侧 `context.cpp` 的域分解（`:174-183`）决定哪条腿走 NVLink，`ginSignalCount=num_ranks+4`
  （`:140`）直接对应协议状态预算。
- **H100 真机**：协议层（NVLink 腿 + QP 映射 + 信号布局）跑真实设备代码 19+17 项全 PASS；
  RDMA 腿与并行混合编排为 NCCL GIN 专属，仅分析。

**下一步**：DeepEP 侧余下的运行期空档（bucket 集合通信 `all_gather/all_reduce/reduce_scatter`
的混合/RDMA 路径、`pp_send_recv`、`engram`）都要 NCCL 设备 API，待「阻塞」中的 NCCL≥2.32.3 环境；
在环境就绪前，可继续拆这些路径里**与 GIN 无关**的 host 编排或归约语义。
