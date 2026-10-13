# 93 · DeepEP 的 BucketBuffer 会话机制：临时张量注册与生命周期守卫

> 前面把 DeepEP 的**通信原语**拆得很细：第 7 篇 dispatch/combine、第 64 篇 bucket 集合通信
> （`all_gather`/`reduce_scatter`/`all_reduce` 三类算子 × 三个物理域）、第 62 篇 bucket 的
> `all_reduce`/`reduce_scatter` 三后端、第 63/58 篇 combine 归约。但有一条一直被当作细节略过的
> 通路：**常规 `all_gather`/`reduce_scatter`/`all_reduce` 要求张量注册在 BucketBuffer 的对称
> 存储里**，而训练脚本经常需要「临时 scratch」——本篇补上把它包起来的 `BucketSession`：
> 一个上下文管理器，在会话内让这三个集合算子**接受任意（未注册）张量**，并负责把它们暂存进
> buffer、再把结果回拷到外部 dst。

本篇目标源码（`~/github/DeepEP@93eb6eb`）：

- `deep_ep/buffers/bucket.py`（398 行）：`BucketSession`（`:19-181`）+ `BucketBuffer`（`:184-398`）。
- `deep_ep/buffers/allocator.py`：`BufferAllocator`（meta 规划 + `materialize`）。
- `deep_ep/buffers/base.py`：`BufferBase`（生命周期基类）。
- `deep_ep/utils/event.py`：`EventOverlap` / `hook_after_wait`。
- `csrc/buffers/bucket.hpp`（374 行）：`stream_control_epilogue` 生命周期守卫 + kernel 侧契约。

**可跑边界**：`BucketSession` / `BufferAllocator` 是**纯 host Python**，不碰任何 CUDA device API，
H100(sm90) 真机可逐字段验证；真正需要 NCCL GIN / 对称窗口的是 `_C.BucketBuffer` 的 kernel 侧
（本机 NCCL 2.29.7 不匹配，见第 86 篇「阻塞」），故实验里用一个**同名方法签名**的 `FakeBucket`
承接 `_call_original`，把会话的注册/暂存/越界/组一致性/结果回拷语义逐条核对（`harness/
exp_deepep_bucket_session.{sh,py,out.txt}`，**31 PASS / 0 FAIL**）。

---

## 1. 为什么需要「会话」：常规集合算子的注册约束

`BucketBuffer` 是「一块注册进 NCCL 对称窗口的连续存储 + 若干通信组」（`bucket.py:184-185`
的 docstring）。常规路径下，`all_reduce`/`reduce_scatter` 的输入**必须**是这块 storage 的视图——
C++ 侧在 `csrc/buffers/bucket.hpp` 里用裸断言把守：

- `reduce_scatter` / `all_reduce`：`EP_HOST_ASSERT(src_begin >= storage_begin and src_end <= storage_end
  and "Input tensors must belong to this BucketBuffer")`（`bucket.hpp:150` / `:213`）。
- `all_gather`：只有 **dst** 必须在 storage 内（`bucket.hpp:279` 的 `bucket_offset` 断言），
  src 允许越界——但**仅限纯 NVLink 路径**（`num_rdma_ranks == 1`）。

这条约束本身合理（对称寻址要求「同 offset 即同逻辑位置」），但它把**使用体验**逼到一个死角：
想做一个临时 reduce、想 all-gather 一块现算出来的激活，都得先手工在 buffer 里 carve 出一段、
再自己拷进拷出。`BucketSession` 就是把这套模板化操作封起来的薄壳。

---

## 2. 会话的开关：用实例属性遮蔽类方法

`BucketSession.__enter__`（`bucket.py:29-39`）做的事很小，但很关键：

```python
def __enter__(self) -> 'BucketSession':
    assert not any(name in self.buffer.__dict__ for name in self.collective_methods), \
        'Nested sessions are not supported'
    self.num_allocated_bytes = 0
    self.context_idx = None
    for name in self.collective_methods:            # ('all_gather','reduce_scatter','all_reduce')
        method = partial(self._collective_func, name)
        setattr(self.buffer, name, method)          # 实例属性 -> 遮蔽类的同名方法
    return self
```

- `self.collective_methods = ('all_gather', 'reduce_scatter', 'all_reduce')`（`bucket.py:27`）；
- **遮蔽的是实例属性**：`setattr(buffer, name, ...)` 在 `buffer.__dict__` 里放一个
  `functools.partial(self._collective_func, name)`。之后 `buffer.all_gather(...)` 命中的是
  这个 partial，而不是 `BucketBuffer.all_gather`；
- **嵌套检测就是检测这个副作用**：`assert not any(name in self.buffer.__dict__ ...)` 一旦发现
  实例属性里已有集合算子名就报 `Nested sessions are not supported`；
- `__exit__`（`:41-43`）逐个 `delattr`，把实例属性抹掉，类方法自动恢复。

> 这也是为什么 `BucketSession._call_original`（`:45-46`）必须写
> `getattr(type(self.buffer), name)(self.buffer, *args, **kwargs)`——因为在会话里
> `getattr(self.buffer, name)` 已经指向了会话自己的 partial（会无限递归），必须显式从**类**
> 上取原始方法。

`_parse_args`（`:48-53`）用 `inspect.signature(...).bind(...)` 把位置/关键字参数统一成 dict，
于是 `_collective_func` 不必关心调用方怎么写。这是「用 Python 元编程做 ABI 适配」的典型手法。

---

## 3. 会话内的临时分配：`allocate` / `_allocate`

`BucketSession.allocate`（`:90-101`）接受单形状或形状列表：

```python
is_batched = isinstance(shapes[0], (tuple, torch.Size, list))
shapes = shapes if is_batched else (shapes, )
tensors = [self._allocate(math.prod(shape) * dtype.itemsize, dtype).view(shape) for shape in shapes]
return tuple(tensors) if is_batched else tensors[0]
```

底层 `_allocate`（`:55-62`）是**单调 bump 分配器**，对齐粒度是 **RDMA 对齐 64B**
（`self.alignment = _C.get_num_rdma_alignment()`，`:24`；对照 `compiled.cuh` 的
`kNumRDMAAlignmentBytes = 64`）：

```python
allocated_bytes = align(num_bytes, self.alignment)   # 64B 上取整
offset = self.num_allocated_bytes
if offset + allocated_bytes > self.buffer.num_bytes:
    raise RuntimeError(f'BucketBuffer session out of storage, which has only {self.buffer.num_bytes}')
self.num_allocated_bytes += allocated_bytes
return self.buffer.storage.narrow(0, offset, num_bytes).view(dtype)
```

三个要点：① **对齐的是记账字节数、不是返回张量长度**（返回 `narrow(0, offset, num_bytes)`
仍是 `num_bytes` 长）；② 越界抛 `RuntimeError`（不是断言）；③ 分配出来的是 `storage` 的**别名
视图**，写它即写 buffer。H100 实测 A1/A2：`_allocate(100)` 两次推进到 `256`（各 128B）、
越界 4096B buffer 抛 `out of storage`、batched `[(2,2) fp16, (4,) fp16]` 占 192B。

> 与 `BufferAllocator` 的区别：`BufferAllocator`（`allocator.py:10-35`）是**规划期**工具——
> 先用 `torch.empty(..., device='meta')` 占位、`num_bytes` 按 **2 MiB**
> （`kNumAllocationAlignmentBytes`）累积，最后 `materialize(storage)` 用
> `torch.utils.swap_tensors` 把 meta 占位**原地换成** storage 的窄视图。H100 实测 C2：
> `(3,5)fp32 + (7,)bf16` 规划成 4 MiB（2MiB 对齐）、`materialize` 后占位 `is_cuda`、
> `data_ptr()` 恰在 storage 起点、可读写；二次 `materialize` / materialize 后再规划都被断言拒绝。
> 会话分配则是**运行期**从 buffer 尾部（或已被规划占用的部分之后）bump 走的另一块空间。

---

## 4. `_collective_func`：把任意张量翻译成 buffer 上的操作

这是整个会话的核心（`bucket.py:103-181`）。它把「用户传进来的 src/dst」翻译成「runtime 能接受
的、落在 storage 内的 src/dst」，分三种情形。

### 4.1 分片数表

```python
num_shards = {
    'all_gather':     (num_ranks, 1),
    'reduce_scatter': (1, num_ranks),
    'all_reduce':     (1, 1),
}[name]
```

`num_shards[0]` 是 src 的「本 rank 分片数」、`num_shards[1]` 是 dst 的。all-gather 的 src 是
1/R 分片、dst 是整块；reduce-scatter 相反；all-reduce 都是整块（in-place）。

### 4.2 「纯 NVLink all-gather」的例外

```python
num_rdma_ranks = self.buffer.contexts[self.context_idx].get_physical_domain_size()[0]
accept_unregistered_src = name == 'all_gather' and num_rdma_ranks == 1
```

只有**纯 NVLink（`num_rdma_ranks==1`）的 all-gather** 允许未注册的 src 直接透传给 runtime
（copy-engine 从任意连续 CUDA 地址拉取，不需要它落在对称窗口里）。其余情形（reduce-scatter /
all-reduce / 跨节点的 all-gather）都必须先暂存。

### 4.3 逐 bucket 的三分支

对每个 `(src, dst)`（`:141-171`）：

```python
is_src_registered = storage_begin <= src.data_ptr() < storage_end
is_dst_registered = storage_begin <= dst.data_ptr() < storage_end if dst_exists else False
num_buffer_bytes = src.nbytes * num_shards[0]
if is_src_registered:
    offset = src.data_ptr() - storage_begin - (rank_idx % num_shards[0]) * src.nbytes
    buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(src.dtype)
elif is_dst_registered:
    offset = dst.data_ptr() - storage_begin - (rank_idx % num_shards[1]) * dst.nbytes
    buffer = self.buffer.storage.narrow(0, offset, num_buffer_bytes).view(dst.dtype)
else:
    buffer = self._allocate(num_buffer_bytes, src.dtype)      # 会话临时分配
```

- **已注册 src**：从 src 指针反推出「这个 bucket 的整块」起点（减去 `rank % num_shards[0]`
  个分片），得到一个覆盖多 rank 分片的 view；all-reduce 下 `num_shards[0]==1`，view 就是 src 本身。
- **已注册 dst**：同理从 dst 反推。
- **都不在**：bump 分配一块。

然后按 `num_shards` 切出本 rank 的 `src_buffer` / `dst_buffer`（`:156-160`）。

### 4.4 暂存与发射

```python
if not is_src_registered and not accept_unregistered_src:
    copy_tasks.append((src_buffer, src))     # 未注册 src 先拷进 buffer
...
if copy_tasks:
    torch._foreach_copy_(*zip(*copy_tasks))  # 一次性批量拷贝
if accept_unregistered_src:
    handle = self._call_original(name, srcs, dsts=dst_buffers, **kwargs)   # 外部 src 透传
else:
    handle = self._call_original(name, src_buffers, **kwargs)              # 只喂 buffer 视图
```

H100 实测（FakeBucket 记录 `_call_original` 收到的实参）：

- **A4**（已注册 all-reduce）：runtime 收到的 src 视图 `data_ptr()` 与用户 src 相同、
  落在 storage 内、`num_allocated_bytes==0`——**零暂存、零分配**。
- **A5**（未注册 reduce-scatter）：`_foreach_copy` 把外部 tensor 拷进 storage、runtime 只看到
  storage 视图、`num_allocated_bytes==align(nbytes,64)`。
- **A6**（纯 NVLink all-gather，`num_rdma==1`）：外部 src 原样透传（runtime 收到的 src 不在
  storage 内、且 `data_ptr` 就是用户张量），dst 落在 storage。
- **A7**（跨节点 all-gather，`num_rdma>1`）：同一个越界 src **被暂存**再喂 runtime、dst 仍落
  storage。这条正是 `accept_unregistered_src` 判据的实测分界。

### 4.5 结果回拷：`_wrap_handle`

`_wrap_handle`（`:72-88`）静/动结合地改写 handle 的 `hook_after_wait`：

```python
hook = handle.hook_after_wait
handle.hook_after_wait = None
def hook_after_wait():
    dst_buffers = hook()                                  # runtime 原本的 epilogue（返回 buffer 视图）
    copy_pairs = [(dst, dst_buffer) for dst_buffer, dst in zip(dst_buffers, dsts, strict=True)
                  if dst.data_ptr() != dst_buffer.data_ptr()]
    if copy_pairs:
        torch._foreach_copy_([dst.view(-1) for dst, _ in copy_pairs], ...)
    return result
handle.register_hook_after_wait(hook_after_wait)
```

注意三个「为什么」：

1. **先取旧 hook 再置空**：`EventOverlap.register_hook_after_wait` 有
   `assert self.hook_after_wait is None`（`event.py:73`），所以必须先把 `hook_after_wait`
   拿出来、置 `None`，再注册新的。
2. **用 data_ptr 判断是否需要回拷**：常规调用里用户传的 dst 就是 buffer 视图（in-place），
   `dst.data_ptr() == dst_buffer.data_ptr()`，`copy_pairs` 为空——**不产生任何多余拷贝**；
   只有用户传了外部 dst 时才回拷。
3. **返回 `result`**：`handle.wait()` 的返回值仍是 runtime 的 epilogue 结果（gathered tensor），
   回拷只是副作用。H100 实测 B：外部 dst 被写对、返回值就是传入的 `result`、
   `hook_after_wait` 事后为 `None`；同 `data_ptr` 的 dst 不重复拷贝。

---

## 5. 生命周期守卫：`stream_control_epilogue`

会话允许「外部 src」（纯 NVLink all-gather）和「外部 dst」，意味着 **runtime 在通信流上异步
读写这些张量**——用户完全可能在 `wait()` 之前就把 src 引用丢了。DeepEP 的守卫在 C++ 侧
`BucketBuffer::stream_control_epilogue`（`csrc/buffers/bucket.hpp:48-56`）：

```cpp
static EventHandle stream_control_epilogue(const std::vector<torch::Tensor>& tensors = {}) {
    const auto comm_stream = comm::get_comm_stream();
    auto event = EventHandle(comm_stream);
    if (get_env<int>("EP_AVOID_RECORD_STREAM", 0)) {
        event.tensors_to_record.assign(tensors.begin(), tensors.end());
    } else {
        for (const auto& tensor: tensors)
            tensor.record_stream(comm_stream);
    }
    return event;
}
```

- 默认路径：对每个 src `tensor.record_stream(comm_stream)`——告诉 CUDA 缓存分配器「这块显存
  的 free 必须等通信流也过了这个点」，于是即使 Python 侧 `del srcs`，底层也不会被回收。
- `EP_AVOID_RECORD_STREAM=1`：改成把张量存进 `EventHandle::tensors_to_record`，由事件生命周期
  托管（注释见 `event.py:14-16`：`record_stream` 与 CUDA Graph 不兼容，故留了这条替代）。
- **调用点**：只有 `all_gather` 把 `srcs` 交给它（`bucket.hpp:314`
  `auto event = stream_control_epilogue(srcs);`），因为只有 all-gather 接受越界 src；
  reduce_scatter/all_reduce 调 `stream_control_epilogue()`（无参，`:180`/`:240`），因为它们的
  输入必在 buffer 内、buffer 本身由 `BucketBuffer` 持有。

上游测试 `tests/bucket/test_all_gather.py:74-80` 正是这条守卫的验收：

```python
handle = buffer.all_gather(srcs, dsts=buckets, group=group, num_sms=num_sms)
del srcs                    # 立刻丢掉外部 src
dsts = handle.wait()        # 只有守卫生效，结果才可能正确
```

H100 实测 D：`stream_control_epilogue` 在 `bucket.hpp:48`、`record_stream` 在 `:55`、
`EP_AVOID_RECORD_STREAM` 在 `:51`、`all_gather` 的 srcs 调用点在 `:314`；
`test_all_gather.py` 的 `del srcs` 在 `:76`——行号逐一核对。

---

## 6. 一个容易读错的点：会话内 `allocate` 的 group 语义

`BucketSession.allocate`（`:90-96`）与 `_check_group`（`:64-69`）保证**一次会话只服务一个组**：

```python
def _check_group(self, group):
    context_idx = self.buffer.get_context_idx(group)
    assert self.context_idx in (None, context_idx), \
        'All collectives in a session must use the same group'
    self.context_idx = context_idx
```

`BucketBuffer` 可以同时持有多个通信组（`groups` 列表，`:204`），但一块缓冲在每个 context 里的
对称布局独立，跨组混用同一段 scratch 语义不清，故会话直接在 `_check_group` 拒绝
（H100 实测 A3：先 `allocate(..., group=groups[0])` 再 `group=groups[1]` → 断言命中）。

---

## 7. 测试即正确性规范

三份官方测试正好定义了会话与分配器的**对外行为契约**：

- `tests/bucket/test_session.py`（101 行）：用 `with buffer.session():` 调三种集合算子，
  再在会话里用 `session.allocate(...)` 就地调用（`:83-86`）。它与 `tests/bucket/test_all_gather.py`
  一起把「会话内可传外部张量、结束后类方法恢复」钉死。
- `tests/buffer/test_allocation.py`（54 行）：`BufferAllocator` 规划 → `BucketBuffer(plan)` 物化，
  断言 `bucket_0.is_meta` → `is_cuda`、`buffer.storage.numel() == plan.num_bytes`、
  `bucket_0.data_ptr() == buffer.storage.data_ptr()`（`:25-33`）——即本篇 §3 的
  meta→CUDA 原地替换语义。
- `tests/bucket/test_all_gather.py`：§5 的生命周期守卫验收（`del srcs` 再 `wait`）。

H100 实测 D2 核对了这三处源码片段存在。

---

## 8. 小结

| 机制 | 落点 | 一句话 |
| --- | --- | --- |
| 会话开关 | `bucket.py:29-43` | 实例属性遮蔽类方法，`__enter__` 装、`__exit__` 拆、嵌套靠属性检测拒绝 |
| 临时分配 | `bucket.py:55-62` | bump 分配器，64B(RDMA) 对齐记账，越界抛 RuntimeError |
| 规划器 | `allocator.py:10-35` | meta 占位 + 2MiB 对齐 + `swap_tensors` 原地物化 |
| 越界 src 分界 | `bucket.py:126` | 仅**纯 NVLink all-gather**（`num_rdma==1`）接受未注册 src 透传 |
| 暂存 | `bucket.py:162-174` | 未注册 src `_foreach_copy` 进 buffer 视图再喂 runtime |
| 结果回拷 | `bucket.py:72-88` | 改写 `hook_after_wait`，只对 `data_ptr` 不同的外部 dst 回拷，返回原 result |
| 生命周期守卫 | `bucket.hpp:48-56` / `:314` | `record_stream`（或 `tensors_to_record`）让外部 src 活到通信流读完 |
| 组一致性 | `bucket.py:64-69` | 一次会话只服务一个通信组 |

一句话：**常规集合算子要求「张量在对称 buffer 里」，`BucketSession` 用「属性遮蔽 + bump 分配 +
暂存/回拷 + record_stream 守卫」把这条约束藏起来**，让训练脚本能像调普通 collective 一样调
bucket collective。H100(sm90) 真机（真实 `bucket.py`/`allocator.py`/`event.py` + `FakeBucket`
承接 runtime）**31 PASS / 0 FAIL**；需要 NCCL GIN / 对称窗口的 kernel 侧仍为「仅分析」。

---

## 附：实验运行

```text
bash harness/exp_deepep_bucket_session.sh > harness/exp_deepep_bucket_session.out.txt
```

输出见 `harness/exp_deepep_bucket_session.out.txt`（含四个源文件的 sha256、DeepEP commit
`93eb6eb`、以及 31 条 PASS）。
