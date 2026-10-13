# DeepEP 屏障的验收基准与「通信 kernel 精确测时」方法论（`tests/comm/test_barrier.py`）

> 目标源码：`~/github/DeepEP@93eb6eb`（本机核对）。参考实现：DeepEP <https://github.com/deepseek-ai/DeepEP>。
> 本篇是 DeepEP 系列里**唯一一份没被单独拆过的验收测试** `tests/comm/test_barrier.py`：它不看正确性，
> 只看**一个屏障有多快**——但这恰好把「DeepEP 怎么测量它自己的通信 kernel」这套方法论钉死了。
> 屏障 kernel 本体走 NCCL GIN 设备 API（本机 NCCL 2.29.7 不提供）→**仅分析**；它的域语义、host 启动、
> Python 转发和 `bench_kineto(barrier_comm_profiling=...)` 测时路径**全在纯 host/Python + torch profiler**
> 里，在 8×H100 上真机复刻（`harness/exp_deepep_barrier_profiling.{py,sh,out.txt}`）。

---

## 0. 为什么补这一篇

前面拆 DeepEP 屏障的有两篇：

- **第 53 篇（P53-1）**：`gpu_barrier` 的两级协议、`NCCLGin` 句柄、`BarrierSignals` 布局、`nvlink_barrier`
  的免复位双相位——**设备协议层**。
- **第 86 篇（P76-1）**：`csrc/comm/barrier.hpp` / `kernels/comm/barrier.hpp` 的 host 启动、
  `Context` 构造、Python 引导层——**host 编排层**。

但**「什么叫这个屏障是对的/够快的」的唯一规范**——`tests/comm/test_barrier.py`——从没被读过。它是
DeepEP 测试套件里最后一份未覆盖的文件（前 100 篇把 `tests/{ep,bucket,pp,engram,utils,buffer,comm}` 的其余
测试都做成了「验收测试即正确性规范」）。本篇补上，并在复刻它的过程中发现一个**关于测时本身**的非平凡现象。

---

## 1. 验收基准 `tests/comm/test_barrier.py`（66 行）

它不验证数值，只**测量屏障耗时**。两个顶层函数：

### 1.1 `test_barrier(buffer, args)`（`:10-26`）

```python
num_scaleout_ranks, num_scaleup_ranks = buffer.get_logical_domain_size()   # :12
...
def loop_barrier(num_tests=1000):                                          # :19-21
    for i in range(num_tests):
        buffer.barrier()
t = bench_kineto(lambda: loop_barrier(), 'barrier',                        # :23
                 barrier_comm_profiling=True, barrier=buffer.barrier)
```

三个非平凡点：

1. **被测对象是「反复调用同一个屏障」**（`loop_barrier` 连打 1000 次），`bench_kineto` 用 kernel 名子串
   `'barrier'` 从 profiler 表里挑出 `barrier_impl` 的**平均单次时长**（`:23`）。屏障是一次一 kernel
   的 kernel，测「单次」才有意义。
2. **`barrier=buffer.barrier`**：把 profiler 的「迭代间削峰屏障」**换成被测对象自己**（见 §5）——
   这样既做了削峰、又不引入一次额外的集合通信。`dist.all_reduce` 在 ring 实现下会有额外 RTT
   （`testing.py:169` 的注释原话），所以这里用本 rank 的 `buffer.barrier()`。
3. `num_allocated_qps`（`:15`）被打印出来——屏障的开销与 QP 数强相关，这是它报告的第一个配置项。

### 1.2 `test_loop(local_rank, num_local_ranks, args)`（`:30-52`）

```python
rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)        # :32
num_rdma_ranks, num_nvl_ranks = deep_ep.comm.get_physical_domain_size(group)   # :33
num_scaleout_ranks, num_scaleup_ranks = deep_ep.comm.get_logical_domain_size(group, args.allow_hybrid_mode)  # :34
assert num_rdma_ranks * num_nvl_ranks == num_ranks                         # :35
assert num_scaleout_ranks * num_scaleup_ranks == num_ranks                 # :36
for i in range(int(1e9) if do_pressure_test else 1):                       # :39
    buffer = deep_ep.EPBuffer(group, num_bytes=2**30, allow_hybrid_mode=...,  # :40-45
                              num_allocated_qps=..., explicitly_destroy=True)
    test_barrier(buffer, args)                                             # :48
    buffer.destroy()                                                       # :51
```

- **两条域恒等式断言**（`:35-36`）：物理域 `rdma×nvl==num_ranks`、逻辑域 `scaleout×scaleup==num_ranks`。
  它们是「域划分自洽」的可执行定义——两个乘积都必须**恰好**还原世界大小。
- **`--do-pressure-test`**（`:61`）把单次变成 `int(1e9)` 次：每轮**重新构造并销毁** EPBuffer
  （`explicitly_destroy=True`，`:44`），即验证**「反复建/毁运行时 + 屏障」不泄漏、不炸**。默认（不传）
  是 `int(1e9)`… 注意 `int(1e9)` 只在 `do_pressure_test` 为真时取（`range(int(1e9) if do_pressure_test else 1)`）。
- `num_bytes=2**30`：屏障本身只需要信号区，但测试故意分配 1 GiB 通信缓冲，贴近真实规模。

### 1.3 参数面（`__main__`，`:55-66`）

| 参数 | 默认 | 作用 |
|---|---|---|
| `--num-processes` | 8 | `mp.spawn` 进程数（`:66`） |
| `--allow-hybrid-mode` | 1 | 传给逻辑域求解（`:34`）→ 见 §2 |
| `--num-allocated-qps` | 8 | 预分配 QP 数（`:44`） |
| `--do-pressure-test` | `store_true` | 压力测试开关（`:61`） |

H100 复刻（Part B）原样 `exec` 真实文件、把 `bench_kineto` 换成记录器，确认调用形态：
`bench_kineto(kernel_names='barrier', barrier_comm_profiling=True, barrier=buffer.barrier)`；
并用 stub 的 `init_dist`/`EPBuffer` 驱动真实 `test_loop`：单次构造参数为 `(2**30, 1, 8, True)`、
`destroy` 恰好一次；伪造 `get_physical_domain_size=(3,4)`（12≠8）时 `:35` 断言按预期抛出。

---

## 2. 「域」的语义：`allow_hybrid_mode` 到底切了什么

`deep_ep/comm/handle.py` 的两个 helper（`:80-85` / `:88-95`）优先问 buffer 的 `context`，否则落到 `_C`；
`_C` 侧实现在 `csrc/kernels/comm/context.cpp`：

```cpp
std::tuple<int,int> get_physical_domain_size(const int64_t& nccl_comm) {          // :52-57
    const auto comm = reinterpret_cast<ncclComm_t>(nccl_comm);
    const int num_ranks = ncclTeamWorld(comm).nRanks, num_nvl_ranks = ncclTeamLsa(comm).nRanks;
    EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0);
    return {num_ranks / num_nvl_ranks, num_nvl_ranks};                            // (rdma, nvl)
}
std::tuple<int,int> get_logical_domain_size(const int64_t& nccl_comm, const bool& allow_hybrid_mode) {  // :59-63
    const auto [num_rdma_ranks, num_nvl_ranks] = get_physical_domain_size(nccl_comm);
    return {allow_hybrid_mode ? num_rdma_ranks : 1,
            allow_hybrid_mode ? num_nvl_ranks : num_rdma_ranks * num_nvl_ranks};
}
```

- **物理域**直接来自 NCCL 拓扑：`ncclTeamWorld`（世界大小）与 `ncclTeamLsa`（同一 NVLink 域，LSA = local
  shared address）。`rdma = world/lsa`，`nvl = lsa`。
- **逻辑域**是唯一可调的一层：
  - `allow_hybrid_mode=True` → `(rdma, nvl)`：**两级混合**，屏障先跨节点（scaleout/RDMA）再域内（scaleup/NVLink）。
  - `allow_hybrid_mode=False` → `(1, num_ranks)`：**一整级**，`scaleout=1` 表示「不跨节点」，
    屏障退化成一个平铺的 domain。

这解释了 `test_barrier` 里两条断言为什么都要写（两种模式乘积都得还原 `num_ranks`），也直接决定
§3 的 `num_sms` 与 §4 的 `kSequential` 走哪条路：

| `allow_hybrid_mode` | 逻辑域 | `launch_barrier` 的 `num_scaleout_ranks` | 效果 |
|---|---|---|---|
| 1（默认） | `(rdma, nvl)` | `rdma` | 两级屏障（`kNumScaleoutRanks>1` 才做 scaleout 腿） |
| 0 | `(1, num_ranks)` | `1` | 单级屏障（`scaleout` 腿被 `if constexpr` 编译掉） |

H100 复刻（Part C）对 5 组 `(num_ranks, nvl)` 复刻 `context.cpp:52-63`，两种模式乘积恒等于 `num_ranks`、
`flat` 恒为 `(1, num_ranks)`：6/6 PASS。

---

## 3. host 链路：Python → `_C.barrier` → `barrier()` → `launch_barrier`

一次 `buffer.barrier()` 的完整下钻：

### 3.1 `deep_ep/buffers/ep.py:237`

```python
barrier = comm.barrier                     # EPBuffer.barrier 就是 comm.barrier 的别名
get_physical_domain_size = comm.get_physical_domain_size
get_logical_domain_size = comm.get_logical_domain_size
```

### 3.2 `deep_ep/comm/barrier.py:6-11`

```python
def barrier(buffer, wait_comm_stream=True, with_cpu_sync=False, sequential=True, context_idx=None):
    context = buffer.context if context_idx is None else buffer.contexts[context_idx]   # :10
    _C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)                    # :11
```

- `context_idx` 选择多通信域里的哪一个（`buffer.context` vs `buffer.contexts[i]`）。
- 注意**参数重排**：Python 是 `(wait_comm_stream, with_cpu_sync, sequential)`，转发给 `_C.barrier` 是
  `(with_cpu_sync, sequential, wait_comm_stream)`。H100 复刻（Part E）用记录型 `_C` 桩验证两条路径的转发
  逐参正确：4/4 PASS。

### 3.3 `csrc/comm/handle.hpp:16-22`（pybind 入口）

```cpp
m.def("barrier", [](const std::shared_ptr<Context>& context,
                    const bool& with_cpu_sync, const bool& sequential, const bool& wait_comm_stream) {
    EP_HOST_ASSERT(context != nullptr);
    EP_HOST_ASSERT(context->barrier_signals != nullptr);      // 信号区必须先 set_barrier_signals
    barrier(*context, context->barrier_signals, at::cuda::getCurrentCUDAStream(),
            context->num_gpu_timeout_cycles, with_cpu_sync, sequential, /*flush_stores=*/true, wait_comm_stream);
});
```

**`flush_stores` 被 hardcode 成 `true`**（`:20`）——屏障前把该 rank 之前发出的 TMA store 都刷出去，
否则远端可能看不到数据。`barrier_signals != nullptr` 的断言把「先 `set_barrier_signals` 再 barrier」
的先后顺序钉死。

### 3.4 `csrc/comm/barrier.hpp:9-42`

```cpp
if (with_cpu_sync)              CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());   // :18-19
else if (wait_comm_stream)      stream_wait(stream, get_comm_stream());        // :20-21
launch_barrier(...);                                                           // :25-33
if (with_cpu_sync)              CUDA_RUNTIME_CHECK(cudaDeviceSynchronize());   // :36-37
else if (wait_comm_stream)      stream_wait(get_comm_stream(), stream);        // :38-40
```

**屏障必须被夹在正确的 stream 顺序里**：要么 CPU 全同步，要么与 comm stream 双向对齐（进入前等 comm
stream、出来后让 comm stream 等本 stream）。这保证「屏障之前计算的通信」与「屏障之后的计算」不会乱序。

### 3.5 `csrc/kernels/comm/barrier.hpp:17-54`（启动）

```cpp
constexpr auto kNumThreads = 512;                                           // :26
const auto num_sms = (not sequential and num_scaleout_ranks > 1) ? 2 : 1;   // :30
```

- **线程数恒为 512**，注释（`:25`）说「等于 rank 数」——每个线程负责一个 rank 的跨 rank 通信。
- **SM 数**：`sequential` 时 1 个 SM 够（先 scaleout 再 scaleup，串行）；只有**并行混合**且真跨节点
  （`num_scaleout_ranks > 1`）才用 2 个 SM（一条 SM 管一级域，并行）。`num_scaleout_ranks==1` 时
  退化成单级，2 个 SM 没意义。
- JIT 现编 `barrier_impl`，8 个模板实参顺序（`:36-41`）：`<is_scaleup_nvlink, num_sms, kNumThreads,
  num_scaleout_ranks, num_scaleup_ranks, num_timeout_cycles, flush_stores, sequential>`；
  发射时 `.cooperative = true`（`:49`）——kernel 内有 grid 协作。

H100 复刻（Part D）逐行复刻 `num_sms` 与模板实参，覆盖 `sequential × scaleout` 的 5 种组合：5/5 PASS。

---

## 4. 设备 kernel：sequential 两级 vs 并行一颗（`impls/comm/barrier.cuh:11-41`）

```cpp
__global__ void barrier_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
                             void* signals, const int scaleout_rank_idx, const int scaleup_rank_idx) {
    auto& barrier_signals = *static_cast<layout::BarrierSignals*>(signals);   // :21
    const auto gin = NCCLGin(nccl_dev_comm, nccl_window, 0);                  // :22
    if constexpr (kSequential) {
        if constexpr (kNumScaleoutRanks > 1)                                  // :25  scaleout 腿
            gpu_barrier<..., kFlushStores, true, false>(gin, barrier_signals, ..., true, false);
        gpu_barrier<..., kFlushStores, true, false>(gin, barrier_signals, ..., false, true);  // :32 scaleup 腿
    } else {
        gpu_barrier<...>(gin, barrier_signals, ...);                          // :36  并行：单次 grid 协作
    }
}
```

- **`kSequential`（默认）**：先做 scaleout 屏障（只有 `kNumScaleoutRanks>1` 才需要，`:25`），再做 scaleup
  屏障；**scaleup 那次要 flush 掉 scaleout 发出的 RDMA 请求**（`:31` 注释）。一个 SM 串行跑两级。
- **并行（`kSequential=false`）**：`num_sms=2` 时两个 SM 各跑一遍 `gpu_barrier`（`grid_sync` 把它们收拢），
  两级域同时进行。但**只有当 `scaleout>1` 时才配 2 个 SM**（§3.5）。

信号结构（`deep_ep/include/deep_ep/layout/common/barrier.cuh:11-23`）：

```cpp
struct alignas(kNumRDMAAlignmentBytes) BarrierSignals {
    unsigned long long nvl_barrier_counter;   // :12  计数器
    int nvl_barrier_signals[2];               // :13  两个相位槽（双相位免复位）
    int* get_nvl_barrier_signal_ptr(const int& phase) { return nvl_barrier_signals + phase; }  // :19-21
};
EP_STATIC_ASSERT(sizeof(BarrierSignals) <= kNumMaxSignalBytes, "Too many signals");             // :23
```

`nvl_barrier_signals[2]` 两槽 = 第 53 篇讲的「加减式相位信号、免复位」的落点；整个结构按 RDMA 对齐字节
对齐（`alignas`），保证跨 rank 的原子读写落在同一缓存行语义下。

`gpu_barrier` 的两级协议、`NCCLGin` 句柄、`get_sym_ptr` 同 offset 翻译、timeout 自旋等**细节已在第 53 / 86
篇展开**，本篇不重复——设备 kernel 的**端到端运行仍受 NCCL GIN 阻塞**（本机 NCCL 2.29.7 无 `ncclGin`），
故 §4 为**仅分析**。

---

## 5. 真正的主角：`bench_kineto(barrier_comm_profiling=...)` 的精确测时

`tests/comm/test_barrier.py:23` 之所以要 `barrier_comm_profiling=True`，全在
`deep_ep/utils/testing.py:113-221` 的 `bench_kineto` 里：

```python
barrier_comm_profiling &= int(os.environ.get('EP_DISABLE_BARRIER_PROFILING', 0)) == 0   # :154
...
for i in range(2):
    for _ in range(num_tests):
        flush_l2_cache(flush_l2)                       # :163
        if barrier_comm_profiling:
            torch.cuda._sleep(int(2e7)) # ~10ms        # :167
            if barrier is None: dist.all_reduce(dummy) # :171
            else:               barrier()              # :173
        fn()                                           # :174
    torch.cuda.synchronize(); profiler.step()          # :175-176
```

- `barrier_comm_profiling` 的含义（`:132-133` docstring）：**每轮被测 kernel 之前插一个「大 kernel + 屏障」**，
  注释原话「用大 kernel 和一个屏障消除**不均衡的 CPU 发射开销**」。
- `torch.cuda._sleep(int(2e7))` 实测**墙钟 13.47 ms**（`:167` 注释标 "~10ms"，H100 上 2e7 cycle 约 13.5ms）。
- `barrier=None`（默认）走 `dist.all_reduce`；测试传 `barrier=buffer.barrier` 用自己的屏障（`:173`）。
- `EP_DISABLE_BARRIER_PROFILING=1` 会把 `barrier_comm_profiling` 就地置 0（`:154`）——**调试用旁路**。

### 5.1 H100 实测：三种模式

用**真实** `deep_ep.utils.testing.bench_kineto` + 真实 DeepGEMM `sm90_fp8_gemm_1d2d`（M=N=K=4096）
在 8×H100 NCCL 上各测 3 次：

```
模式                     dur[us] 三次                            min      max   spread
no-barrier             [99.8, 99.7, 99.8]                   99.7     99.8     0.1%
barrier=all_reduce     [101.1, 101.1, 100.9]               100.9    101.1     0.2%
barrier=custom         [101.5, 101.5, 101.5]               101.5    101.5     0.1%
disabled(env)          [99.0, 98.7, 98.6]                   98.6     99.0     0.4%
```

`disabled(env)` 与 `no-barrier` 同区间——坐实 `EP_DISABLE_BARRIER_PROFILING` 把插桩关掉。
event 计时参考（`flush` 在 start event 之前、不计入）**min=104.7 / med=106.0 us**，与 profiler 的快档一致。

### 5.2 发现：无 sleep 时 `bench_kineto` 呈**双峰**（快档 ~100us / 慢档 ~417us）

单看上面这一轮，四种模式都在 ~100us，好像 `barrier_comm_profiling` 无所谓。但把**所有「无 sleep」
的测量**（no-barrier + disabled(env) + 裸 profiler 的 `sleep=0` 扫描）合并看，样本横跨 **98–419us**、
明显**双峰**；而三种 barrier 模式每轮都落在快档、spread < 0.3%。多次重跑都能观察到慢档（417us ≈ 4×）：

```
裸 profiler 扫 sleep 长度（其余同 bench_kineto）：
     sleep=        0 cycles → profiler 报告的 kernel 时长 = 98.345us
     sleep=  1000000 cycles → 100.384us
     sleep=  4000000 cycles → 418.927us      ← 慢档
     sleep= 10000000 cycles → 101.346us
     sleep= 20000000 cycles → 101.451us
```

**慢档是真实的 GPU 行为，不是 profiler 假象**。直接量持续负载（背靠背、无间隔）：

```
持续负载（背靠背，无间隔）：纯 GEMM 0.338 ms/iter、flush+GEMM 0.659 ms/iter
                            （对照隔离 event 的 0.106 ms/iter）
```

即同一个 4096³ FP8 GEMM，**隔离时 ~0.106 ms、持续背靠背时 ~0.34–0.66 ms**（3–6×）。这与第 15 篇（反向
profiling 里 ncu 观察到「单次 vs 持续降频 ≈2×」）和第 36 篇（`set_tc_util` 在 H100 上「降载并不抬频」）
是同一类现象：**H100 在持续满载下会降频**，而一段足够长的 GPU 空闲（`_sleep(2e7)` ≈ 13.5ms）让频率
恢复、每轮都是快档。

于是 `barrier_comm_profiling` 的真实作用比 docstring 说的更微妙：它名义上是「削 CPU 发射抖动」，实际
效果里**还包含「给 GPU 一段空闲以避开降频」**。这也解释了为什么 `test_barrier` 用它——**通信 kernel
的单次时长对频率极敏感，不插空闲测出来的数字会忽快忽慢**。

> 说明：本机 GPU 被多个自驱任务共享，慢档出现的具体轮次不可复现（有时 no-barrier 三次全快、有时全慢），
> 与第 30 / 40 / 65 篇记录过的共享 GPU 抖动一致。**结论是「双峰 + barrier 稳定」这个定性事实**，不是某一个
> 具体倍数。

---

## 6. H100 实测证据（`harness/exp_deepep_barrier_profiling.{py,sh,out.txt}`）

**真实源码路径加载**（`sys.modules` 注入合成包 + stub `deep_ep._C`/`envs`，按文件路径 exec），sha256：

```
testing.py        sha256=9a6f140744f53dc8…     handle.py     sha256=24f555020247b358…
barrier.py        sha256=706bef5123370813…     test_barrier.py sha256=307f062297d26958…
DeepEP 源码：93eb6eb
```

**Part B —— 验收基准驱动逻辑**（原样 exec `tests/comm/test_barrier.py`）：argparse 面 4 个参数；
`test_barrier` 的 `bench_kineto(..., 'barrier', barrier_comm_profiling=True, barrier=buffer.barrier)` 形态；
`test_loop` 单次构造 `EPBuffer((2**30, 1, 8, True))` + `destroy×1`；伪造域被 `:35` 断言拒绝。**4/4 PASS**。

**Part C —— 域尺寸语义**（复刻 `context.cpp:52-63`）：5 组 `(ranks,nvl)` × 两模式，乘积恒等 `num_ranks`、
`flat=(1,num_ranks)`。**6/6 PASS**。

**Part D —— `launch_barrier` 启动公式**（复刻 `kernels/comm/barrier.hpp:25-41`）：`kNumThreads=512`、
`num_sms` 在 5 种 `sequential×scaleout` 组合下与期望一致 + 模板实参逐项。**5/5 PASS**。

**Part E —— Python 参数转发**（stub `_C`）：`barrier(buffer)` / `barrier(context_idx=2, ...)` 的
`_C.barrier` 逐参正确；域 helper「有 context 问 context、无 context 落 `_C`」。**4/4 PASS**。

**Part F —— 真实 `bench_kineto` 测时**：见 §5.1/§5.2（真实 DeepGEMM kernel、8×H100 NCCL）。

---

## 7. 边界与链接

- **设备 kernel 端到端仅分析**：`barrier_impl` / `gpu_barrier` 走 NCCL GIN 设备 API，本机 NCCL 2.29.7
  不提供；协议层细节见第 53 篇，host 编排见第 86 篇。
- 本篇补齐的是**「验收测试即正确性规范」家族的最后一块**（`tests/comm/`），以及它顺带暴露的
  **通信 kernel 精确测时方法论**。
- **参考链接**：DeepEP 仓库 <https://github.com/deepseek-ai/DeepEP>（本机核对 commit `93eb6eb`、
  v2.5）；相关文件 `tests/comm/test_barrier.py`、`deep_ep/utils/testing.py`、`deep_ep/comm/{barrier,handle}.py`、
  `deep_ep/buffers/ep.py`、`csrc/comm/{barrier,handle,stream}.hpp`、
  `csrc/kernels/comm/{barrier.hpp,context.cpp,api.hpp}`、
  `deep_ep/include/deep_ep/impls/comm/barrier.cuh`、`deep_ep/include/deep_ep/layout/common/barrier.cuh`。
