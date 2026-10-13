# DeepEP 的通信—计算重叠「Python 调度地基」：通信流、`EventOverlap` 与 buffer 生命周期

> 本文对应路线图 **P65-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 75 篇。
>
> 专题一路从 kernel 侧讲「通信—计算重叠」：第 2 / 9 篇讲 mbarrier / ring buffer 让 TMA 与 UMMA 在
> **SM 内部**重叠，第 11 篇讲 dispatch / combine 的 NVLink 读写，第 63 篇讲「每次 dispatch 开几个 SM」
> 的解析式预算，第 71 / 73 篇讲两级域与 hybrid。但所有这些 kernel 都是被**某一股 CUDA stream** 发射的
> ——**谁把通信 kernel 发到一条独立 stream、谁让计算流只在该等的地方等**，这一层（Python / host 调度）
> 从没系统讲过。本篇补上：DeepEP 的 `EventOverlap` + `get_comm_stream` + `BufferBase`。
>
> **与专题的关系**：Mega MoE 的溢价来自「dispatch 的 NVLink 搬运」与「linear1/2 的 Tensor Core」在
> 时间上叠起来。叠不叠得起来，取决于两层：硬件层（通信只许占一部分 SM，第 63 篇的「半卡上限」）
> 与**软件层（通信在 comm stream、计算在 compute stream、用 event 只在必要处串起来）**。本篇讲后者。
>
> **可跑性**：`EventOverlap` / `get_comm_stream` / `BufferBase` 全是**纯 host / Python**，底层就是
> `torch::Event` / `torch::Stream`（即 `torch.cuda.Event` / `torch.cuda.Stream`），**不依赖 SM100**。
> `import deep_ep` 会触发 `deep_ep/__init__.py:check_nccl_so()`（本机 NCCL 2.29.7 与编译期不匹配 →
> abort），故实验用 `sys.modules` 注入合成 `deep_ep._C` 后**按文件路径 exec 真实文件**，再用
> `TorchEventHandle`（逐行镜像 `csrc/utils/event.hpp` 的 `EventHandle`）驱动**真实 CUDA 事件/流**；
> 另编译一个 CUDA 程序镜像 `csrc/comm/stream.hpp` 的 `get_comm_stream` / `stream_wait` 语义。
> **H100 真机 21 PASS / 0 FAIL**（`harness/exp_deepep_stream_event_overlap.{sh,py,out.txt}`、
> `src/deepep_stream_event.cu`）。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - Python：`deep_ep/utils/event.py:8-106`（`EventOverlap`）、`deep_ep/comm/stream.py:9-12`
>   （`get_comm_stream`）、`deep_ep/buffers/base.py:4-27`（`BufferBase`）
> - C++ 孪生：`csrc/utils/event.hpp:13-36`（`EventHandle`）、`csrc/comm/stream.hpp:9-27`
>   （通信流 / `stream_wait`）、`csrc/buffers/base.hpp:11-61`（`BufferBase`）
> - 真正的「交接点」：`csrc/buffers/ep.hpp:122-181`（`stream_control_prologue` / `_epilogue`）、
>   `csrc/buffers/bucket.hpp:48-58`
> - 消费点：`deep_ep/buffers/ep.py:700/741/750-757`、`deep_ep/buffers/bucket.py:72-87/318-390`

## 1. 一段话总览

DeepEP 里每次 `dispatch()` / `combine()` 都在**两条 CUDA stream** 上做文章：

* **compute stream**：调用者当前流（`torch.cuda.current_stream()`，上面跑 attention / GEMM / 反向）；
* **comm stream**：DeepEP 内部**唯一一条、进程级 static** 的通信流（`csrc/comm/stream.hpp:9-12` 的
  `getStreamFromPool(true)`，即高优先级 + 从流池取）。通信 kernel 全部发在它上面。

两条流各自推进，用 **CUDA event** 在「必须看见对方结果」的点上串起来：

```
compute stream:  ──(前序计算)──┐                            ┌──(后续计算/反向)──>
                              │                            │
comm stream:     (wait 前序) ─┴─ dispatch/combine kernel ──┴─ (record event) ...
                              ↑                            ↑
                    stream_wait(comm, compute)     EventHandle(comm) 交给 Python
```

Python 侧把这个事件包成 `EventOverlap`：它支持 `with event_overlap:` 语法（在 `__exit__` 里让当前流
等事件），也支持 **`hook_after_wait`**（等完之后在当前流跑一段收尾，例如确定性 dispatch 的排序）。
`async_with_compute_stream=True` 时，通信 kernel 一发射就返回事件、**当前流先不等**，调用者可以立刻把
计算塞进两条流之间的空隙——这就是「重叠」的软件开关。

## 2. 通信流：为什么是「池里取 + static 单例 + 高优先级」

`csrc/comm/stream.hpp:9-12`：

```cpp
static at::cuda::CUDAStream get_comm_stream() {
    static const auto comm_stream = at::cuda::getStreamFromPool(true);   // true = 高优先级
    return comm_stream;
}
```

三个「为什么」：

1. **`static`**：整进程只有一条通信流。这样所有 rank / 所有 Layer 的通信 kernel 都**串行**在同一条
   流上——NVLink / RDMA 的带宽本就被这些 kernel 共享，再开多条流只会互相抢 SM 与 QP，
   还要多一堆跨流 event；单例换来最简的依赖图。
2. **`getStreamFromPool(true)`**：从 CUDA 流池（`torch::cuda::StreamPool`）取、优先级取高。池化的
   流与非默认流一样支持跨流 event 同步，但**不占用「默认流会隐式同步」的特殊性**；高优先级让通信
   kernel 在调度上先于同卡的普通计算 kernel 抢 SM（对本机 H100，优先级区间是 `[0 .. -5]`，
   实测通信流拿到 `-5`，见实验 Part B）。
3. Python 侧只做一层**薄包装**（`deep_ep/comm/stream.py:9-12`）：把 C++ 返回的 `torch::Stream`
   重新包成 `torch.cuda.Stream`（`stream_id` / `device_index` / `device_type` 三个字段照抄），
   以便 Python 能 `with torch.cuda.stream(get_comm_stream())`。**底层仍是同一个流**——实验 B2/B3
   坐实：两次 `get_comm_stream()` 拿到的是**不同的 Python 对象、相同的 `stream_id`**。

`csrc/comm/handle.hpp:29` 把它绑到 Python：`m.def("get_comm_stream", ...)`。

### 2.1 `stream_wait`：两种「让某条流等」

`csrc/comm/stream.hpp:14-27` 有两个重载：

```cpp
static torch::Event create_event(const CUDAStream& stream) {       // :14-18
    auto event = torch::Event(torch::kCUDA);
    event.record(stream);
    return event;
}
static void stream_wait(const CUDAStream& stream, const CUDAStream& event_stream) {  // :20-23
    EP_HOST_ASSERT(stream.id() != event_stream.id());              // 自等待 = 死锁
    stream.unwrap().wait(create_event(event_stream));
}
static void stream_wait(const CUDAStream& stream, const EventHandle& event) {        // :25-27
    stream.unwrap().wait(*event.event);
}
```

* 第一种（流等流）：**现场在 `event_stream` 上记一个临时 event，再让 `stream` 等它**。断言
  `stream.id() != event_stream.id()` 是防呆——`cudaStreamWaitEvent` 让一条流等它自己刚记录的事件，
  在语义上是「等自己把当前队列跑完」以外没有意义的自环，容易写成死锁。**实验 E1/E2** 在真机上验证：
  两条不同的流 id 不同、同一流的自等待被这条断言拒绝。
* 第二种（流等 `EventHandle`）：等一个**已经在别处 record 过**的事件，这正是 `previous_event`
  参数（跨 `dispatch` / `combine` / `bucket` 传递已发射通信的句柄）走的路。

## 3. `EventHandle`：C++ 的事件句柄，Python 只暴露两件事

`csrc/utils/event.hpp:13-30`：

```cpp
struct EventHandle {
    std::shared_ptr<torch::Event> event;
    tensor_list_t tensors_to_record;                       // :15

    EventHandle() {                                         // :17-20 默认：记在当前流
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(at::cuda::getCurrentCUDAStream());
    }
    explicit EventHandle(const at::cuda::CUDAStream& stream) {  // :22-25 记在给定流
        event = std::make_shared<torch::Event>(torch::kCUDA);
        event->record(stream);
    }
    EventHandle(const EventHandle& other) = default;       // :27 拷贝共享同一 event

    void current_stream_wait() const {                     // :29
        at::cuda::getCurrentCUDAStream().unwrap().wait(*event);
    }
};
```

绑到 Python 时（`:32-36`）**只暴露默认构造 + `current_stream_wait`**：Python 不能指定「记在哪条流」，
因此 Python 造出的 `EventHandle` 总是记在**当前流**上。真正需要「记在 comm stream」的句柄，是 C++
侧 `stream_control_epilogue` 在通信 kernel 发完后造的（见 §5）。

`tensors_to_record`（`:15`）是一个 Optional<torch::Tensor> 列表，用来**替代 `record_stream`**：
当环境变量 `EP_AVOID_RECORD_STREAM=1` 时把张量挂到事件上（`ep.hpp:163-164`），否则对每个张量显式
`record_stream(compute_stream)` + `record_stream(comm_stream)`（`ep.hpp:166-169`，bucket 侧同款
`bucket.hpp:51-56`）。**为什么必须做这件事**：PyTorch caching allocator 以「流」为单位记账；一个张量
如果只在 compute stream 上登记，通信流还在读它时 allocator 可能已经把它回收给别的分配 → use-after-free。
把两条流都 `record_stream`（或把引用挂进事件）就告诉 allocator「这块内存在这两条流都跑完前不许回收」。
这是重叠**正确性**的隐藏前提，不是性能优化。

## 4. `EventOverlap`：Python 侧的事件包装（真实代码逐条验证）

`deep_ep/utils/event.py:8-106` 是这层的中枢。它的字段与语义：

```python
class EventOverlap:
    def __init__(self, event=None, extra_tensors=None):          # :17-37
        self.event = event
        self.extra_tensors = extra_tensors                       # :30（EPv2 已不用，:29 注释）
        self._release_handle_by_call = False                     # :33
        self.hook_after_wait = None                              # :37

    def current_stream_wait(self, release_handle=False):         # :39-60
        self.event.current_stream_wait()                         # :47 转发给 EventHandle
        result = None
        if self.hook_after_wait is not None:                     # :51-53 等待后触发一次 hook
            result = self.hook_after_wait()
            self.hook_after_wait = None
        if release_handle:
            self.event = None                                    # :58-59 释放底层句柄
        return result                                            # :60 透传 hook 结果

    def wait(self): return self.current_stream_wait()            # :62-64
    def register_hook_after_wait(self, hook):                    # :66-71
        assert self.hook_after_wait is None, "..."               # :70 只允许注册一次
        self.hook_after_wait = hook

    def __call__(self, release_handle=False):                    # :73-82
        self._release_handle_by_call = release_handle
        return self                                              # :82 返回 self，不新建包装

    def __enter__(self): return self                             # :84-96
    def __exit__(self, *exc):                                    # :98-106
        if self.event is not None:                               # :104
            self.current_stream_wait(release_handle=self._release_handle_by_call)  # :105
        self._release_handle_by_call = False                     # :106
```

**实验 Part A** 把这段真代码（`exec` 真实文件，`sha256=db7e5b40…`）逐条硬断言，**8/8 PASS**：

| 断言 | 结论 |
|---|---|
| A1 `current_stream_wait` 把等待转发给 `event`，无 hook 时返回 `None` | 与 `:47/:60` 一致 |
| A2 `hook_after_wait` **只触发一次**、返回值被透传、随后清空 | `:51-53` 的「一次性」 |
| A3 重复 `register_hook_after_wait` 抛 `AssertionError("already registered")` | `:70` 文案核对 |
| A4 普通 `with event_overlap:` 退出时等待、但**保留** `event`（可被多流再等） | `:104-105` |
| A5 `with event_overlap(release_handle=True):` 退出后 `event=None`、标志复位 | `:58-59/:105-106` |
| A6 `event_overlap(...)` 返回**同一个对象**（不新建，引用计数不变） | `:82` |
| A7 `event=None` 时 `with` 退出**不等待**（已释放的句柄无害） | `:104` 的 `is not None` |
| A8 异常穿过 `with` 仍执行等待/释放（`__exit__` 无视异常） | `:98-106` |

三个「为什么」：

1. **`hook_after_wait` 为什么存在**：`:35-37` 注释点名用途——**确定性 dispatch** 需要在
   「当前流等到通信完成」之后、**在同一股当前流上**跑一次排序（把 `atomicAdd` 分配的乱序槽位排回
   源 token 序，见第 62 篇 `EPHandle.deterministic_sort`）。它必须挂在「等待之后」这个精确时点，
   且只跑一次。`ep.py:733-743` 正是这么用：`deterministic=True` 且 `async_with_compute_stream=True`
   时，`event_overlap.register_hook_after_wait(deterministic_epilogue)`（`:741`）。
2. **`__call__` 返回 self** 是为了 `with event_overlap(release_handle=True):` 语法（`:77`）：
   在 `with` 前把「退出时要不要释放句柄」写进对象，避免每层调用新建包装对象、打乱 `Config`
   里底层事件的引用计数（`:79` 注释明确）。
3. **`release_handle` 默认 False**：句柄可被**多条流**等待（例如同一份 dispatch 结果被前向与
   反向先后使用），所以默认不释放；只有调用者明确知道「没人再等了」才 `True`。

## 5. 真正的交接点：`stream_control_prologue` / `_epilogue`

Python 的 `EventOverlap` 只是壳，真正决定「谁等谁、事件记在哪条流」的是 C++
`csrc/buffers/ep.hpp:122-181`（bucket 侧是 `bucket.hpp:48-58` 的简化版，因为 bucket 集合通信
是同步发起，不需要 prologue）。

**Prologue（`:122-150`）**：

```cpp
const auto compute_stream = at::cuda::getCurrentCUDAStream();
const auto comm_stream = comm::get_comm_stream();
if (previous_event.has_value())
    comm::stream_wait(comm_stream, previous_event.value());   // :128 通信流等前一个通信事件
else
    comm::stream_wait(comm_stream, compute_stream);           // :130 或等当前计算流

if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(comm::get_comm_stream());  // :148 让分配记在通信流账上
```

* **`previous_event` 的语义**（`:138-142` 注释）：当调用者**先发了计算 kernel（要重叠）再调 dispatch**
  时，通信流必须先等那条计算；此时 `allocate_on_comm_stream` 必须为 `True`，因为「不能再用计算流上的
  内存」（那块内存正被计算 kernel 用着）。这就是 `dispatch(...)` 文档里
  `async_with_compute_stream` / `allocate_on_comm_stream` 两个开关的由来（`ep.py:607-612`）。

**Epilogue（`:152-181`）**：

```cpp
if (async_with_compute_stream) {
    event = EventHandle(comm_stream);                        // :160 事件记在通信流完成点
    for (auto& t: tensors) { t->record_stream(compute_stream); t->record_stream(comm_stream); }  // :166-169
} else {
    comm::stream_wait(compute_stream, comm_stream);          // :172 不同步：当前流直接等通信
}
if (allocate_on_comm_stream)
    at::cuda::setCurrentCUDAStream(compute_stream);          // :177 切回计算流
return event;                                                // :180
```

这就是「重叠开关」的落点：

* `async_with_compute_stream=True` → 只造一个 `EventHandle`（记在 comm stream）返回给 Python，
  **当前流先不等**。Python 拿到 `EventOverlap` 后可以选择「先做点别的、再 `with` / `wait()`」——
  通信与计算的重叠窗口由此打开。
* `False` → 当前流立刻 `stream_wait` 通信流，等价于串行。

Python 侧（`ep.py:700-757`）把返回的事件包成 `EventOverlap(event)`（`:700`），并在
**`defer_epilogue=True`**（`:623-624` 文档）时把整个收尾（拿 recv 张量、做 deterministic 排序、
返回 handle）也挂进 `hook_after_wait`（`:750-753`），只把 `EventOverlap` 直接返回给调用者。
调用者 `event.current_stream_wait()` 的那一刻，收尾才真正跑。**bucket 的 `_wrap_handle`
（`bucket.py:72-87`）也是同一招**：把已有 hook 与新收尾串起来，保证 hook 链只注册一次。

## 6. `BufferBase`：通信 buffer 的生命周期契约

`deep_ep/buffers/base.py:4-27` 与 `csrc/buffers/base.hpp:11-61` 是一对。它管两件事：

1. **`main_context` 的转发与失效**：Python 的 `BufferBase.main_context`（`base.py:11-14`）在
   `runtime is None`（已销毁）时返回 `None`；C++ 侧 `main_context` 是 `shared_ptr<comm::Context>`
   （`base.hpp:32`）。所有 `EPBuffer` / `BucketBuffer` / `EngramBuffer` / `PPBuffer` 都继承它。
2. **`destroy` 契约**（`base.py:21-27` / `base.hpp:34-35`）：
   `explicitly_destroy=False`（默认）时析构函数自动 `destroy()`；`True` 时**必须**显式
   `destroy()`，否则析构时会 printf 一行 `destroy() is not called ... which can leak resources`
   （`base.hpp:19-29`）。Python 的 `destroy()` 在 `explicitly_destroy=False` 时被 `assert` 拦下
   （`base.py:24`）。

**实验 Part C** 对真实 `base.py` 逐条断言，**4/4 PASS**：`runtime=None → main_context=None`、
`main_context` 正确转发、`destroy()` 后 `runtime=None` 且底层只销毁一次、非 explicit 模式 `destroy`
被 assert 拦下。

`print_memory_usage`（`base.hpp:37-49`）把 workspace / GPU buffer / RDMA storage / CPU storage
四段显存按 GiB 打印——这是排查「EP buffer 吃掉多少显存」的第一手工具。

## 7. H100 真机实测：重叠的「软件层」与「硬件层」缺一不可

两个实验在 H100（sm90，132 SM）上跑出真实数字。

### 7.1 `EventHandle` 的 memory-ordering（实验 Part A / D3）

* C++ 镜像：通信流上 kernel 自旋后写 `7.25`，记录事件；默认流 `current_stream_wait` 后读——
  **读到 `7.25`，`bad=0`**。证 `cudaEventRecord` + `cudaStreamWaitEvent` 提供**跨流 memory ordering**：
  等待之后的访问一定看得见生产者流的写（这是 `EventOverlap` 能替代「插一个 `torch.cuda.synchronize`」
  的根据）。
* Python：用**真实 `EventOverlap` + `TorchEventHandle`**（记在 comm 流）做同样的事，`with` 退出后
  当前流读到 `3.14159`（D3 PASS）。
* D4：在 `TorchEventHandle` 上挂 `hook_after_wait`，等待后 hook 触发、返回值 `'R'` 透传——
  与 §4 的语义在真实事件上也成立。

### 7.2 重叠收益：全卡 vs 让出 SM（实验 Part D，`src/deepep_stream_event.cu`）

计算腿 = 全卡 FFMA（约 7.79 ms），通信腿 = device copy。两种配置：

| 通信腿配置 | 通信腿耗时 | 串行 (copy→wait→FFMA) | 重叠 (copy∥FFMA，末尾 wait) | 加速比 |
|---|---|---|---|---|
| **全卡**（1024 blocks，512 MB copy） | 0.392 ms | 8.173 ms | **8.263 ms** | **0.989×（不重叠）** |
| **1/4 SM**（33 blocks，512 MB copy） | 2.025 ms | 14.534 ms | **8.186 ms** | **1.776×（真重叠）** |

同一份 CUDA 事件/流代码，**只改通信腿的 block 数**，结论就从「不重叠」翻成「1.78×」。这就是
**重叠的两个必要条件**：

1. **软件层**：通信必须发在**另一条流**上、且当前流**不在发射点等**（`async_with_compute_stream`）。
   两行的代码差别，实验里只体现在「末尾 wait」还是「发射后立刻 wait」。
2. **硬件层**：通信 kernel 的 grid **必须留出 SM 给计算**。当两个 kernel 的 block 数都远超 132 个 SM
   时，GPU 把它们**近乎串行**地填满 SM（第一个 kernel 的 block 排满、第二个只能等 block 退役），
   墙钟 ≈ 两者之和——**事件/流语义上「可以并行」并不等于「真的并行」**。

这正面解释了第 63 篇那个「**半卡硬上限**」：`get_theoretical_num_sms` 把通信的 SM 数钳在
`device_sms // 2 * 2`，就是要给计算留出另一半 SM，重叠才不只是纸面上的。表里 33 个 block（1/4 SM）
的 1.776× 正是这个原理的最小复现。

## 8. 小结与索引

本篇补上了「通信—计算重叠」在 **Python / host 调度层**的地基，与专题其它篇互补：

| 层 | 篇目 | 关键机制 |
|---|---|---|
| SM 内部 | 第 2 / 9 篇 | mbarrier、ring buffer、TMA↔UMMA 流水 |
| kernel 本体 | 第 11 / 7 篇 | dispatch / combine 的 NVLink 读写 |
| **重叠预算** | **第 63 篇** | `get_theoretical_num_sms`（半卡上限） |
| **重叠软件层** | **本篇** | **comm stream + `EventOverlap` + `stream_wait`** |
| 两级域 | 第 71 / 73 篇 | hybrid dispatch/combine、SM/QP 模型 |

**一句话**：通信流是「池里取的高优先级 static 单例」，`EventHandle` 是记事簿，`EventOverlap` 是
「等不等 / 等完顺带做什么 / 要不要释放」的三个开关，`stream_control_epilogue` 决定事件记在哪条流，
而**重叠能不能兑现，最终取决于通信 kernel 是否给计算让出了 SM**——H100 上，改一个 block 数就从
0.989× 变成 1.776×。

## 9. 线上资料

* DeepEP 仓库（`deep_ep` Python 层与 `csrc` host 层）：<https://github.com/deepseek-ai/DeepEP>
* PyTorch CUDA semantics（stream / event 的语义与 caching allocator `record_stream`）：
  <https://pytorch.org/docs/stable/notes/cuda.html>
* DeepGEMM（Mega MoE / 反向 kernels，本专题主线）：
  <https://github.com/deepseek-ai/DeepGEMM>
