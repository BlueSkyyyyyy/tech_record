# DeepEP 的 `EPHandle` 句柄契约与 `dispatch`/`combine` 的 host 编排层（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P73 增量（第 83 篇）**。
> 前面的 DeepEP 篇目把每条**设备侧**路径都拆透了：dispatch/combine（第 7/11 篇）、combine 归约
> （第 48 篇）、dispatch 拷贝 epilogue（第 51 篇）、确定性排序（第 52 篇）、多 rank 屏障（第 53 篇）、
> 两级 hybrid（第 61 篇）、SM/QP 自动选择模型（第 63 篇）、通信-计算重叠的 Python 调度地基
> （第 65 篇）。但把这些拼起来的那条「**host Python 编排层**」从没被单独讲清：`EPBuffer.dispatch`
> /`combine` 怎么归一化参数、怎么自动选 SM/QP、怎么复用上一步的 `EPHandle`、deferred epilogue
> 怎么接线——以及 **`EPHandle` 本身作为「dispatch→GEMM→combine→反向」的路由账本，到底存了什么、
> 谁消费什么**。
>
> 源码坐标：`~/github/DeepEP`，公开版 commit `93eb6eb`（v2.5.0，2026-09-30）。
> 本篇全部是**纯 host / Python**，H100 真机可跑（`harness/exp_deepep_epbuffer_handle.{py,sh,out.txt}`）。

## 1. 为什么需要 `EPHandle`：把「一次 dispatch 的路由账本」物化成对象

Mega MoE 的前向是一条 `dispatch → linear1 → SwiGLU → linear2 → combine` 的流水线，
其中 dispatch 把 token 按 top-k 专家发给各 rank、combine 再把结果按原路收回来。
这两步之间的**一切路由信息**（谁被发到哪、每个本地专家收了多少 token、前缀和、源 token 索引）
如果每次都重算，就要再读一遍 `topk_idx`、再跑一遍计数内核。

DeepEP 的做法是：**dispatch 顺手产出这些账本，打包进 `EPHandle`**，让下游直接拿来用。
最典型的消费者就在 DeepGEMM 的 Mega MoE 端到端测试里（`~/github/DeepGEMM/tests/test_mega_moe.py`）：

```
# test_mega_moe.py:307-333
recv_x, _, recv_topk_weights, handle, _ = ep_buffer.dispatch(..., do_expand=True)
gemm_fn(recv_x, l1_weights, l1_y, handle.psum_num_recv_tokens_per_expert, ...)   # grouped GEMM 的 PSUM 布局
...
gemm_fn(l1_y, l2_weights, l2_y, handle.psum_num_recv_tokens_per_expert, ...)
...
ep_buffer.combine(l2_y, handle=handle, bias=...)                                  # 按 handle 原路收回
```

也就是说 `handle.psum_num_recv_tokens_per_expert` 就是 grouped GEMM 的 **PSUM 布局**（第 1 篇
§1 讲过的三布局之一，见 `docs/01`）。`EPHandle` 把「通信产出的分组前缀和」直接喂给「计算侧的
grouped GEMM」，这就是 DeepEP 与 DeepGEMM 之间的**接口**。

而 `EPHandle` 的另一个身份是**反向的缓存**。`dispatch` 的 docstring 明说：

```
# ep.py:613-615（节选）
handle: an optional cached `EPHandle` from a previous dispatch, if set, the CPU will reuse the layout
    information to save some time. `topk_idx` must be `None` (reused from handle).
    `topk_weights` can be optionally provided (e.g. for backward pass with cached expand).
```

前向 dispatch 缓存下的 handle，反向（autograd 重放 dispatch）可以只换 `topk_weights` 就复用整套
expand 布局——**通信布局在反向不必重算**。这正是「MoE 反向无 mega-kernel、但可以复用前向路由」
（第 4/14 篇结论）在通信侧的对应物。

> 结论先行：`EPHandle` = 「一次 EP all-to-all 的路由账本」，字段分三类——
> ① **身份/配置**（`do_expand`/`num_experts`/`expert_alignment`/`num_max_tokens_per_rank`/`num_sms`）；
> ② **分组前缀和**（两种 psum + 未对齐计数，给 grouped GEMM 当 PSUM 布局）；
> ③ **元数据**（`recv_src_metadata`/`dst_buffer_slot_idx`/hybrid 的 `token_metadata_at_forward`/
> `channel_linked_list`）。

## 2. `EPHandle` 的字段契约（`ep.py:27-207`）

### 2.1 `__init__` 与身份字段（`:64-105`）

构造签名（`:64-79`）按固定顺序收 16 个字段，`:80` 先 `assert topk_idx is not None`（handle 必须
记住路由，没有 top-k 索引的 handle 无意义）。身份字段直接存：`do_expand`、`num_experts`、
`expert_alignment`、`num_max_tokens_per_rank`、`num_sms`（combine 会复用这个 SM 数，见 §4）。

### 2.2 两种 psum 的语义差异（`:43-53`，**最容易读错的地方**）

`EPHandle` 的 docstring 把两种前缀和讲得很细，这里翻译成可执行的定义：

- **`psum_num_recv_tokens_per_scaleup_rank`**（`:43-46`）：去重后「每个 scaleup rank 收到的 token 数」
  的**含端点前缀和**。一个 token 的多个 top-k 专家落在**同一 rank** 时只计一次。**最后一个元素 =
  收到的 token 总数**——所以 `deterministic_sort` 在 `do_cpu_sync=False` 时用它当「有效行数」
  （`ep.py:141`）。

- **`psum_num_recv_tokens_per_expert`**（`:47-53`）：每个本地专家收到的、**按 `expert_alignment`
  向上取整后**的 token 数的前缀和。两种模式语义不同：
  - **非 expand**：就是普通的含端点前缀和。
  - **expand**：`psum[i] = align(psum[i-1], expert_alignment) + 专家 i 的未对齐真实计数`。
    于是：
    - 专家 `i` 的真实计数 = `psum[i] - align(psum[i-1], expert_alignment)`（`:51-52`）；
    - 专家 `i+1` 的起始偏移 = `align(psum[i], expert_alignment)`（`:52-53`）。

  这个「PSUM 末尾是**未对齐**真实计数、前一专家末尾才补齐」的设计，让 grouped GEMM 既能拿到
  每个专家的对齐块起点、又能知道块里有多少是真 token（对齐填充槽要输出 0 / 跳过，见第 13 篇
  的 PSUM 布局与 `assert_psum_zero_padding`）。

- **`num_unaligned_recv_tokens_per_expert`**（`:55-56`）：真实（未对齐）per-expert 计数，**仅 expand
  模式填充**。它和上面的 psum 一起，才能从 `psum` 反推出每个专家的真实计数（`deterministic_sort`
  的 expand 分支正是用 `psum - unaligned` 当「每专家起始的真实 token 位置」，`ep.py:187`）。

### 2.3 `topk_idx` 的版本守卫（`:103-111`）

`handle.topk_idx` 是一个 **property**，不是裸字段：

```
# ep.py:103-111
self._topk_idx = topk_idx
self._topk_idx_version = None if topk_idx.is_inference() else topk_idx._version

@property
def topk_idx(self) -> torch.Tensor:
    assert self._topk_idx_version is None or self._topk_idx._version == self._topk_idx_version, \
        '`topk_idx` must not be modified while the EP handle is in use'
    return self._topk_idx
```

**为什么**：`topk_idx` 是**按引用**存进 handle 的（`:39-42` docstring：「Stored by reference in the
handle; must not be modified until all uses of the handle complete」）。它可以被缓存 handle 的后续
dispatch 复用、被 combine 消费，若中途被原地改写，路由就错了。PyTorch 的 `_version` 计数器能抓到
**普通原地修改（含经 view）**，但抓不到 `.data` / 裸指针写入；**inference 模式创建的张量没有版本
计数器**（`is_inference()`），此时守卫退化成「不检查」——这是文档明确写下的能力边界（`:40-42`）。

### 2.4 未显示的元数据字段

`recv_src_metadata`、`dst_buffer_slot_idx`、`token_metadata_at_forward`、`channel_linked_list`
分别是「源 token 索引 + 缓冲槽位」、「目的缓冲槽位」、「hybrid 转发的 per-channel 元数据」、
「hybrid 每 scaleup peer 的链式列表」。它们的**产生**在设备侧（第 11/61 篇），本篇只关心它们在
**host 编排里怎么被搬运**（§3.4、§4）。

## 3. `EPBuffer.dispatch` 的 host 编排（`ep.py:562-757`）

`dispatch` 有 20 多个参数，但结构很清晰：**归一化 → 自动选 SM/QP → 句柄复用 → 调 runtime →
finalize**。逐段看。

### 3.1 前置断言与守卫（`:638-639`）

```
assert not do_handle_copy, '`do_handle_copy` must be False; handle copying is no longer supported'
check_torch_deterministic()
```

`do_handle_copy` 是历史包袱，现已禁用（`:616`）。`check_torch_deterministic()`（`utils/envs.py:184-189`）
断言「**不要同时**开启 `torch.use_deterministic_algorithms(True)` 与
`torch.utils.deterministic.fill_uninitialized_memory`」——因为二者同时开时，`torch.empty()`
会发射一个初始化内核，可能与通信流重叠而出错（docstring 原话）。这是 host 编排的**第一道守卫**。

### 3.2 自动选 SM / QP（`:641-645`）

```
num_topk = (handle.topk_idx if topk_idx is None else topk_idx).shape[1]
num_sms = self.get_theoretical_num_sms(num_experts, num_topk) if num_sms == 0 else align(num_sms, 2)
num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps
assert num_qps <= self.num_allocated_qps, f'Allocated QPs are not enough'
```

- `num_topk` **不需要显式传**：从 `topk_idx`（或缓存 handle 里的）形状第 2 维推出来。
- `num_sms==0` → 调 `get_theoretical_num_sms`（`ep.py:430-541` 的解析式 roofline，**第 63 篇已拆**）；
  否则 `align(num_sms, 2)`（SM 数必须偶数，第 63 篇讲过为何）。**注意 auto 分支不额外 align**——
  它内部已经 `align(..., 4)` 且硬上限半卡（`:530-532`）。
- `num_qps==0` → `get_theoretical_num_qps`（`:543-560`）：direct 模式 `min(num_sms, 9)`、
  hybrid 模式 `min(num_sms*16 + 1, num_allocated_qps)`（`:554-558`）。direct 压到 9 是为了减少
  DB ring 开销；hybrid 给每个 channel（外加 notify）一条独立 QP。
- `assert num_qps <= self.num_allocated_qps`：QP 是**构造期预分配**的（`:327-335`：direct 17、
  hybrid 65 或 129，取决于 NIC 是否支持 fast atomic），运行期不能超。

### 3.3 SF 解包与句柄复用（`:647-671`）

```
x, sf = x if isinstance(x, tuple) else (x, None)     # :648 FP8 模式 = (x_fp8, sf) 的二元组

if handle is not None:                                # :652 缓存句柄分支
    assert topk_idx is None                           # :653
    assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'   # :654
    topk_idx = handle.topk_idx                        # :655
    num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, handle.num_max_tokens_per_rank)
    num_experts = value_or(num_experts, handle.num_experts)
    expert_alignment = value_or(expert_alignment, handle.expert_alignment)
    do_cpu_sync = False                               # :659
    assert (num_experts, expert_alignment, num_max_tokens_per_rank) == \
           (handle.num_experts, handle.expert_alignment, handle.num_max_tokens_per_rank)  # :662
(cached_...) = self._unpack_handle(handle)            # :664-671
```

三个关键「为什么」：

1. **缓存句柄要求 `topk_idx is None`**（`:653`）：路由从 handle 里复用，**不允许再传**——否则新旧的
   top-k 可能不一致，句柄语义就崩了。
2. **缓存句柄禁止 CPU 同步**（`:654`）：`do_cpu_sync` 会等 CPU 拿到精确接收计数；缓存路径直接拿
   handle 里的旧计数，跳过同步。这是「缓存 handle 是为了省时间」的核心：**省掉的正是那次 CPU/GPM 同步**。
3. **上下文一致性断言**（`:661-663`）：复用 handle 时，本次传的（或默认的）`num_experts` /
   `expert_alignment` / `num_max_tokens_per_rank` 必须与 handle 里一致——因为这三者决定了 buffer
   的物理布局，改了就不匹配。这是**裸断言**（无消息），`align` 后的默认值回填靠 `value_or`（`:656-658`）。
4. `_unpack_handle(handle)`（`:401-418`）把 handle 里 10 个缓存字段拆成位置参数；`handle=None` 时
   返回 10 个 `None`（`:407-408`）——统一了「首跑」与「缓存」两条路径的调用形状。

### 3.4 默认值与 runtime 调用（`:673-699`）

```
num_max_tokens_per_rank = value_or(num_max_tokens_per_rank, self.num_max_tokens_per_rank)  # :674
expert_alignment = value_or(expert_alignment, 1)                                          # :675
do_cpu_sync = value_or(do_cpu_sync, True)                                                 # :676

result, event, deferred_epilogue = self.runtime.dispatch(x, sf, topk_idx, topk_weights,
    cumulative_local_expert_recv_stats, <10 个 cached_...>, num_max_tokens_per_rank,
    num_experts, expert_alignment, num_sms, num_qps, previous_event,
    async_with_compute_stream, allocate_on_comm_stream, do_cpu_sync, do_expand,
    do_zero_padding, use_tma_aligned_col_major_sf, defer_epilogue)                        # :679-699
```

`num_max_tokens_per_rank` 默认取 buffer 构造时的值（`:674`），`expert_alignment` 默认 1（不对齐），
`do_cpu_sync` 默认 `True`（`handle` 分支里被强制 `False`，`:659`）。`self.runtime.dispatch` 是
**C++ 侧**（`csrc/buffers/ep.hpp` 的 `EPBuffer::dispatch`），它再按 `do_expand`/`num_scaleout_ranks`
等分派到具体设备 kernel（第 11/51/61 篇）。host 只负责把归一化好的参数排成固定位置。

### 3.5 `finalize_dispatch`：句柄装配与确定性接线（`:702-747`）

`runtime.dispatch` 返回 `(result, event, deferred_epilogue)`。result 是一个 14 元组（`:703-713`）。
`finalize_dispatch(dispatch_result, deterministic_by_hook)`（`:702`）：

1. **构造或复用 handle**（`:716-731`）：`is_cached_dispatch = handle is not None`；首跑则
   `EPHandle(do_expand, num_experts, expert_alignment, num_max_tokens_per_rank, num_sms, topk_idx,
   num_recv_tokens, ..., channel_linked_list)`；缓存路径**原样返回旧 handle**（`else handle`）。
2. **确定性排序接线**（`:734-743`）：
   ```
   if self.deterministic:
       deterministic_epilogue = functools.partial(handle.deterministic_sort, do_cpu_sync,
           is_cached_dispatch, recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list)
       if deterministic_by_hook:
           event_overlap.register_hook_after_wait(deterministic_epilogue)   # 延迟到 wait 后跑
       else:
           deterministic_epilogue()                                          # 立即跑
   ```
   即：**开启确定性时，排序必须在 `current_stream_wait()` 之后**才能在正确的 stream 上读回数据；
   若调用方是 `async_with_compute_stream=True`（hook 模式）就注册成 hook，否则立即执行。
   `deterministic_sort` 的两键/单键排序细节见**第 52 篇**，本篇只讲「它由谁在何时触发」。
3. **返回形状归一化**（`:746-747`）：`recv_x = (recv_x, recv_sf) if recv_sf is not None else recv_x`
   ——输入是 `(fp8, sf)` 二元组，输出也回一个二元组；否则裸张量。

### 3.6 deferred epilogue：把收尾推迟到 wait（`:749-757`）

```
if deferred_epilogue is not None:
    event_overlap.register_hook_after_wait(lambda: finalize_dispatch(deferred_epilogue(), False))
    return event_overlap                      # 直接返回事件，不是 5 元组
assert result is not None
return *finalize_dispatch(result, async_with_compute_stream), event_overlap
```

`defer_epilogue=True`（要求 `async_with_compute_stream=True`，见 `:623-624` 的 docstring）时，设备侧把
「等 CPU 接收计数 + 拷贝 epilogue」也推迟；host 收到一个 **callable** `deferred_epilogue`，
把它注册成「等事件后再跑」。于是调用方拿到的直接是 `EventOverlap`，`event.wait()` /
`event.current_stream_wait()` 时才真正拉取结果。**为什么**：这让「通信发射」和「等通信完成 +
落盘」解耦，前者可以立刻与计算重叠，后者在需要数据时才付同步代价。

## 4. `EPBuffer.combine` 的 host 编排（`ep.py:770-843`）

`combine` 只接受 `handle`（必须来自 dispatch），逻辑比 dispatch 短：

```
check_torch_deterministic()                                                # :813
num_sms = handle.num_sms if num_sms == 0 else align(num_sms, 2)            # :816 复用 dispatch 的 SM 数
num_qps = self.get_theoretical_num_qps(num_sms) if num_qps == 0 else num_qps  # :817
assert num_qps <= self.num_allocated_qps                                   # :818
bias_0, bias_1 = EPBuffer._unpack_bias(bias)                              # :820
result, event, deferred_epilogue = self.runtime.combine(x, topk_weights, bias_0, bias_1,
    handle.recv_src_metadata, handle.topk_idx, handle.psum_num_recv_tokens_per_scaleup_rank,
    handle.token_metadata_at_forward, handle.channel_linked_list,
    handle.num_experts, handle.num_max_tokens_per_rank, num_sms, num_qps, previous_event,
    async_with_compute_stream, allocate_on_comm_stream, handle.do_expand, defer_epilogue)  # :821-835
```

两个要点：

- **SM 数默认从 handle 复用**（`:816`）：`num_sms==0` 时直接用 dispatch 时定的 `handle.num_sms`，
  保证 dispatch/combine 用同一套 SM 分区（否则重叠预算会算错）。这与 dispatch 的 auto 分支不同
  （dispatch 是 `get_theoretical_num_sms` 现算）。
- **combine 直接消费 handle 的 6 个字段**（`:823-829`）：`recv_src_metadata`（回到哪个源 rank/槽）、
  `topk_idx`（权重路由）、`psum_num_recv_tokens_per_scaleup_rank`（每 rank 分块）、hybrid 两个
  元数据、`num_experts` / `num_max_tokens_per_rank`。**这就是「combine 按 dispatch 原路返回」的实现
  形态**：路由不在 combine 里重算，全从 handle 读。

`_unpack_bias`（`:759-768`）：`bias` 允许是 0/1/2 个 `[num_combined_tokens, hidden]` 的 bf16：
- `None` → `(None, None)`；
- 单个 Tensor → `(bias, None)`；
- 2 元组 → `(b0, b1)`；
- **其它长度的 tuple 被 `assert len(bias) == 2` 拒绝**（`:766`）。

`defer_epilogue` 的处理与 dispatch 相同（`:837-839`），返回形状是 `(combined_x,
combined_topk_weights, EventOverlap)`（`:841-843`）。

## 5. 支撑这套编排的辅助基座

| 组件 | 位置 | 作用 |
|---|---|---|
| `value_or` | `utils/semantic.py:5-6` | 「None 才用默认」——`value_or(0, d)==0`、`value_or(False, d) is False`，与「`or` 短路」不同（0/False 是合法值） |
| `weak_lru` | `utils/semantic.py:9-26` | 对 `get_theoretical_num_sms`/`lb_get_theoretical_num_sms` 的缓存：**只弱引用 self**，避免 buffer 被缓存永久钉住（`functools.lru_cache` 直接装饰方法会泄漏） |
| `align`/`ceil_div` | `utils/math.py:22-27` | SM 数偶数对齐、缓冲对齐 |
| `check_torch_deterministic` | `utils/envs.py:184-189` | dispatch/combine 首行的守卫（§3.1） |
| `EventOverlap` | `utils/event.py:8-106` | deferred epilogue / 确定性 hook 的载体（`register_hook_after_wait`:66-71、`current_stream_wait`:39-60）。第 65 篇已拆其状态机 |
| `EPBuffer.__init__` 的 QP 预分配 | `ep.py:327-335` | direct 17 / hybrid 65（fast atomic）或 129，运行期不可超（§3.2 断言的来源） |

## 6. H100 真机实测（`harness/exp_deepep_epbuffer_handle.{py,sh,out.txt}`）

**跑法**：`import deep_ep` 会触发 `deep_ep/__init__.py:check_nccl_so()`（本机 NCCL 2.29.7 与编译期
不匹配 → abort），所以脚本用 `sys.modules` 注入合成 `deep_ep._C`（`EventHandle` 桩）与 `deep_ep.comm`
桩，再**按文件路径 exec 真实文件**（`semantic/math/event/envs/allocator/base/ep`），对
**未修改的真实方法体**做断言。编排层不碰 NCCL，故这是等价验证；`deterministic_sort` 用**真实 CUDA
张量**在 H100 上跑。

**结果：51 PASS / 0 FAIL**（GPU H100 80GB HBM3, cc 9.0；torch 2.12.0a0；源码 sha256 见 `.out.txt`）。

分层看：

- **Part A 句柄契约（7/7）**：字段存取；`topk_idx` 是 property；**原地改 `topk_idx` 后访问抛
  `` `topk_idx` must not be modified while the EP handle is in use ``（实测 `_version` 0→1）**；
  inference 张量 `_topk_idx_version is None`（守卫关闭）；`topk_idx=None` 被 `assert` 拒。
- **Part B 静态辅助（6/6）**：`_unpack_handle(None)` → 10 个 `None`；`_unpack_handle(h)` 按固定顺序
  取回 10 个缓存字段（逐个 `is` 核对）；`_unpack_bias` 的 None/Tensor/2-tuple 三态 + 长度≠2 被拒。
- **Part C dispatch 编排（20/20）**：`num_sms=0` → 调 `get_theoretical_num_sms(num_experts=8,
  num_topk=2)`（干净地从 `topk_idx.shape[1]` 推 `num_topk`）；显式 `num_sms=7` → align 到 8；
  direct auto QP = `min(sms,9)=9`、hybrid = `min(sms*16+1, allocated)`；`x=(fp8,sf)` 被拆成位置参数
  且返回重新打包；无 SF 时返回裸张量；默认 `expert_alignment=1`/`num_max_tokens_per_rank=buffer 默认`/
  `do_cpu_sync=True`；**缓存 handle：`topk_idx` 来自 handle、返回同一对象、强制 `do_cpu_sync=False`**；
  `num_experts` 与 handle 不一致被 `:662` 裸断言拦下；`handle` 与 `topk_idx` 互斥；`do_handle_copy=True`
  被拒；**defer_epilogue 直接返回 `EventOverlap`，`event.wait()` 才产出 4 元组**；
  **deterministic + async → 注册 hook、非 async → 立即排序**（钩子接线分支）。
- **Part D combine 编排（6/6）**：`num_sms=0` → 复用 `handle.num_sms=48`；runtime 收到的 6 个
  handle 字段逐个 `is` 核对 + `do_expand`；返回 3 元组；`bias=(b0,b1)` 正确透传；deferred 分支。
- **Part E 真机 `deterministic_sort`（3/3）**：`recv_src_metadata[:,0]=[3,1,0,2,7,5,4,6]` 乱序时，
  真实设备代码把 `recv_x` 重排成源 token 升序（标记值 `[0,10,...,70]`），`recv_src_metadata` 变升序，
  且非缓存时缓存了 pre-sort 副本（`:135-136`）。
- **Part F 辅助（6/6）**：`value_or` 对 `0`/`False` 不误判；`weak_lru` 命中缓存、不同参数各算一次、
  不强引用 self；`align`/`ceil_div` 边界。
- **Part G 守卫（3/3）**：确定性算法 ∧ `fill_uninitialized_memory` 同开 → `check_torch_deterministic`
  失败，且 `dispatch` 首行就抛出。

**一次小的源码行为确认**：C7 观察到的断言消息是**空字符串**——`:662` 的上下文一致性断言是**裸
`assert`**（无消息），与 `:638` 的带消息断言形成对照。这类「哪条断言带消息」的细节只有真跑才看得到。

## 7. 与其它篇章的关系、小结

- **第 63 篇**拆了 `get_theoretical_num_sms`/`get_theoretical_num_qps` 的数学；本篇讲它们在 dispatch/
  combine 里的**调用点**（dispatch auto、combine 从 handle 复用）。
- **第 52 篇**拆了 `deterministic_sort` 的排序算法；本篇讲它在 `finalize_dispatch` 里**被谁、何时**
  触发（hook vs 立即）。
- **第 65 篇**拆了 `EventOverlap` 的状态机；本篇把它当成 deferred epilogue / 确定性钩子的**载体**。
- **第 1/13 篇**的 PSUM grouped 布局，其「通信侧来源」正是 `handle.psum_num_recv_tokens_per_expert`
  （expand 模式的「末尾未对齐、前段补齐」语义，§2.2）。

**一句话**：`EPHandle` 是 DeepEP 把「一次 EP all-to-all 的路由账本」物化成的对象——dispatch 顺带
产出它，combine 按它原路收回，DeepGEMM 的 grouped GEMM 拿它的 PSUM 当布局，反向缓存 expand 布局
省一次重算。`EPBuffer.dispatch`/`combine` 的 host 层则负责把 20 多个参数归一化、自动选 SM/QP、
复用句柄、接线 deferred/确定性钩子——**设备 kernel 前的最后一道「人话」翻译层**。

**源文件清单**：`deep_ep/buffers/ep.py`（`EPHandle`:27-207、`EPBuffer`:208-906）、
`deep_ep/utils/semantic.py`、`deep_ep/utils/math.py`、`deep_ep/utils/event.py`、
`deep_ep/utils/envs.py`；消费方 `~/github/DeepGEMM/tests/test_mega_moe.py:307-333`。
