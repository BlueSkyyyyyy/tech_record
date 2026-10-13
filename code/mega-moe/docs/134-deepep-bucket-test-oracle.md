# DeepEP bucket / buffer 家族最后三份验收测试：少见的 `rtol=0` 逐位判据、会话内分配与多样式分配计划

> 本专题已把 DeepEP 的测试套件逐份拆成「测试即正确性规范」（EP `tests/ep/test_ep.py` 第 75/85 篇、
> bucket 的 `test_mixed.py`/`test_perf.py` 第 84 篇、EPLB 第 90 篇、barrier 第 91 篇、pp/engram 第 112 篇、
> `test_gate.py` 第 106 篇）。扫完整个 `tests/` 目录，只剩**三份**从没被单独读过：
> `tests/bucket/test_all_gather.py`（117 行）、`tests/bucket/test_session.py`（101 行）、
> `tests/buffer/test_allocation.py`（54 行）。本篇把它们收口，也就给整条 DeepEP 验收测试栈画上句号。
>
> 这三份测试覆盖了 DeepEP bucket 家族里**三个此前没在测试层讲过的机制**：
> ① all-gather 的**纯 NVLink copy-engine 语义**（无 SM kernel、`num_sms=0`）与「越界输入」特例；
> ② `BucketSession` 会话内 **bump 分配 + 在缓冲区内就地调用**；
> ③ `BufferAllocator` 的 **meta→cuda 物化计划**与**多 group 注册**。
> 其中 `test_session.py:62/75` 与 `test_all_gather.py:55/80` 用的**逐位判据（`rtol=0`/`torch.equal`）**
> 是整个测试套件里最严的一档——本篇解释它为什么成立、边界在哪。
>
> 参考版本：`~/github/DeepEP` 公开版 commit `93eb6eb`（v2.5.0）。本机 `kernel_lab`（H100 sm90，8 卡）。
> 实测：`harness/exp_deepep_bucket_tests.{py,sh}` + `.out.txt`。
> 目标卡代号敏感：全文一律用「目标卡」指代。
>
> **重要约束**：bucket 的数据面（`all_gather`/`reduce_scatter`/`all_reduce` 的设备 kernel）走 NCCL GIN /
> copy engine，本机 NCCL 2.29.7 不提供（见 ROADMAP「阻塞」），**端到端仅分析**；但测试侧全是
> **纯 PyTorch / NCCL 主机 API**，参考语义、逐位判据、缓冲区算术全部可在 H100 真机复现。

---

## 1. 三份测试的共同装置

三份测试用同一套样板（与第 75/84/112 篇一致）：

- `init_dist(local_rank, num_local_ranks)`（`deep_ep/utils/envs.py:89-133`，第 108 篇）起 `mp.spawn` 的
  每进程 NCCL 子组；
- `deep_ep.comm.get_physical_domain_size(group)`（`deep_ep/comm/handle.py:80-95`）拿 `(num_rdma_ranks,
  num_nvl_ranks)` 物理域大小——单机 8 卡是 `(1, 8)`；
- `deep_ep.BufferAllocator`（`deep_ep/buffers/allocator.py:11`）+ `deep_ep.BucketBuffer`（
  `deep_ep/buffers/bucket.py:184`）；
- `deep_ep.utils.testing.bench_kineto`（`:113`）+ `parse_num_bytes`（`:224`）测性能
  （第 55/91 篇详）。
- 参考值一律用 **torch 自带 NCCL 的集合算子**（`dist.all_gather_into_tensor` / `dist.reduce_scatter_*` /
  `dist.all_reduce`）构造——这是「DeepEP 自研 bucket kernel 算什么对」的唯一权威。

一个**实测边界**：`test_session.py:49/51` 用的 `dist.all_gather_single` / `dist.reduce_scatter_single`
在**本机 torch 2.12.0a0 里并不存在**（`dir(torch.distributed)` 只有 `all_gather_into_tensor` /
`reduce_scatter_tensor`，与第 112 篇 `test_pp.py` 的发现一致）。这两个 `*_single` 别名由 DeepEP 的
torch 补丁提供，故 `test_session.py` 在**未打补丁的 torch 上无法原样运行**——本篇复现时用
`*_tensor` 等价别名。

---

## 2. `test_all_gather.py`：纯 NVLink 的 copy-engine 语义与两类特例

`test`（`:18`）在 `torch.inference_mode()` 下做三件事：

### 2.1 批量 20 个 gathered 张量的逐位对拍（`:35-55`）

```python
allocator = deep_ep.BufferAllocator()
gathered_tensors = []
for _ in range(20):
    shard_numel = int(torch.randint(1, 1 << 20, ()).item()) * 8   # :37 随机 shard，8 的倍数
    gathered_tensors.append(allocator.allocate((num_ranks * shard_numel,), torch.float32))
```

每个 gathered 张量长度是 `num_ranks * shard_numel`，第 i 个的 `src` 是本 rank 的 shard（
`gathered[rank_idx*shard : (rank_idx+1)*shard]`，原地 `normal_().add_(rank_idx)`，`:44`），参考值用
`dist.all_gather_into_tensor(reference, src, group=group)`（`:49`）。判据是**两条硬断言**：

- `dst.data_ptr() == gathered.data_ptr()`（`:54`）——DeepEP 的 all-gather 是**原地**的：结果直接写回
  buffer 内该 gathered 张量的存储，不另分配；
- `torch.equal(dst, reference)`（`:55`）——**逐位**相等。fp32 数据、纯搬运（不求和），逐位是应有之义。

注意 `buckets = (gathered_tensors[i], gathered_tensors[(i+1)%20])`（`:42`）——每次 all-gather **两个**
张量（环形相邻），既测 batched 接口，又顺带测「多输入一次发射」。

### 2.2 「越界输入」特例，仅纯 NVLink（`:58-81`）

```python
if num_rdma_ranks == 1:        # :58 只有纯 NVLink 域才有这条路径
    ...
    src = torch.randn(shard_numel, ..., device='cuda').add_(rank_idx)   # :65 缓冲区外的张量
    ...
    handle = buffer.all_gather(srcs, dsts=buckets, group=group, num_sms=num_sms)
    del srcs                    # :77 立刻丢弃输入，检查生命周期守卫
    dsts = handle.wait()
```

**三个非平凡点**：

1. **`srcs` 可以是缓冲区外的任意连续 CUDA 张量**——但只在 `num_rdma_ranks == 1`（纯 NVLink /
   copy-engine 路径）时支持。跨节点（RDMA）路径要求输入已注册在对称缓冲（`bucket.py:361` 的 docstring
   明说「out-of-buffer inputs are only supported by the pure NVLink path」）。原因：copy-engine push 只
   需要本地线性地址，而 RDMA 需要已登记的对称窗口 + `get_sym_ptr` 翻译（第 54/56 篇）。
2. **一次调用里混合「越界 + 在原地」输入**：`j==0` 的 `src` 越界、`j==1` 的 `src` 是 buffer 内 shard
   （`:62-67`），验证同一 batched 调用能同时接受两者。
3. **生命周期守卫**：`:77` 在 `wait()` 前 `del srcs`——因为异步发射，源码张量要活到通信真正完成；
   `EventOverlap` 的 `tensors_to_record`（第 65/108 篇）持有引用，故提前删本地名字安全。

### 2.3 性能对拍（`:84-92`）

`bench_kineto` 分别按 kernel 名 `'ncclDevKernel_AllGather'` 与 `'all_gather'` 抓 NCCL 与自研 kernel，
用 `barrier_comm_profiling=True` + `barrier=buffer.barrier`（第 91 篇的精确测时法），打印
`... GB/s | ...x NCCL`。**`num_sms` 对 all-gather 恒为 0**（`bucket.py:290-291` 的 `get_theoretical_num_sms`
对 `all_gather` 直接 `return 0`）——纯 copy-engine 驱动，不占 SM。

---

## 3. `test_session.py`：会话内 bump 分配 + 就地集合调用

`test`（`:19`）先断言 `num_rdma_ranks == 1`（`:22`，「All-reduce is only implemented for the NVLink
domain」），再算缓冲区大小：

```python
alignment = deep_ep.get_num_rdma_alignment()                       # :23 = 64
numel = align(args.num_bytes // 4, num_ranks * 4)                  # :25
num_buffer_bytes = align(3 * align(numel * 4, alignment),
                         deep_ep.get_num_allocation_alignment())   # :28-29 = 2 MiB
buffer = deep_ep.BucketBuffer(group, num_buffer_bytes, explicitly_destroy=True)  # :31
```

三个集合算子各留一段 `align(numel*4, 64)`，总计再对齐到 2 MiB。这里 `align(x,y)=ceil(x/y)*y`
（`deep_ep/utils/math.py:26`），`get_num_rdma_alignment()=64`、
`get_num_allocation_alignment()=2*1024*1024`（`csrc/python_api.cpp:24/26`，
`deep_ep/include/deep_ep/common/compiled.cuh:68-69`，第 56 篇）。

### 3.1 参考值三件套与容差梯度（`:35-57`）

```python
dist.all_gather_into_tensor(all_gather_ref.view(-1), all_gather_src.view(-1), group=group)  # :49
dist.reduce_scatter_tensor(reduce_scatter_ref, reduce_scatter_src.view(-1), group=group)     # :51
all_reduce_ref = all_reduce_src.clone(); dist.all_reduce(all_reduce_ref, group=group)        # :55-56
```

判据梯度（这是本篇最值得复制的部分）：

| 算子 | 判据 | 为什么 |
|---|---|---|
| all_gather | `rtol=0, atol=0`（`:62/75`） | 纯搬运，无算术 → 逐位 |
| reduce_scatter | `rtol=1e-5, atol=1e-5`（`:65/80`） | 跨 rank **求和**，顺序不同 → 非逐位 |
| all_reduce | `rtol=1e-5, atol=1e-5`（`:68/85`） | 同 reduce_scatter |

这与第 62 篇「`test_reduce_scatter.py` 的 RDMA 路径用 `rtol=0` 逐位」形成对照：**逐位只在
「求和顺序与参考逐字一致」时才用**。单机 8 卡（NVLink 域）下自研 kernel 的归约顺序**未必**与 NCCL 的
`reduce_scatter_tensor` 一致，故测试保守放宽到 1e-5（**实测**这次 8×NVLink 恰好逐位——NCCL 在 NVLink
域用 ring 序，与反序环重建 bit 相同——但这是观测、不是契约）；而 RDMA-only 路径的 kernel 恰好按
「本地先行 + 反序环 `(rank-i)%R`」求和，与测试参考 `test_reduce_scatter.py:78-87` 逐字相同，才敢开
`rtol=0`。**`rtol=0` 不是「更敢断言」，而是「实现与参考同序」的指纹。**

### 3.2 会话内就地调用（`:59-85`）

```python
with buffer.session():                                  # :59 空会话，直接调 buffer 方法
    all_gather_dst = buffer.all_gather(all_gather_src, dsts=all_gather_dst, group=group).wait()
    reduce_scatter_dst = buffer.reduce_scatter(reduce_scatter_src, group=group).wait()
    all_reduce_dst = buffer.all_reduce(all_reduce_src, group=group).wait()

with buffer.session() as session:                       # :71 拿会话对象，在缓冲区内分配
    all_gather_inplace = session.allocate((num_ranks, *all_gather_src.shape), group=group)  # :72
    all_gather_inplace[rank_idx].copy_(all_gather_src)
    all_gather_result = buffer.all_gather(all_gather_inplace[rank_idx], group=group).wait()
    ...
```

`BucketSession`（`deep_ep/buffers/bucket.py:19`）的机制（第 83 篇的 `docs/93` 详）：

- `__enter__`（`:29-43`）把三个集合方法**用 `partial(self._collective_func, name)` 覆盖到 buffer
  实例属性上**——于是 `buffer.all_gather(...)` 在 `with` 块里走的是「接受未注册张量」的版本；
  并断言不嵌套（`assert not any(name in self.buffer.__dict__ ...)`，`:32`）；
- `__exit__`（`:41-43`）`delattr` 还原；
- `_allocate`（`:56-65`）是 **64B 对齐的 bump 分配器**：`offset = self.num_allocated_bytes`，
  越界即 `RuntimeError`（`:61-62`），分配出的是 storage 的 `narrow` 视图；
- `allocate`（`:90-101`）支持 `(shape)` 或 `(shape0, shape1, ...)` 两种形态（`is_batched`），
  group 缺省时落到 `groups[context_idx]`；
- `_check_group`（`:66-71`）强制**一个会话只用一个 group**。

第二种写法（`:71-85`）是**在 buffer 存储内分配输入、再调就地集合**：`all_gather_inplace[rank_idx]`
是本 rank 的 shard 视图，gather 结果写回整块 `all_gather_inplace`；`reduce_scatter`/`all_reduce`
则把整块输入就地改写。**两条路径的判据完全相同**（逐位 / 1e-5），坐实「会话只是把『未注册张量』
这层约束藏起来」（第 83 篇结论）。

---

## 4. `test_allocation.py`：`BufferAllocator` 物化计划与多 group 注册

`test`（`:11`）先建**两个**通信组：默认 `group_0`（`init_dist` 返回，`:12`）与**两两配对**的
`group_1`（`num_ranks//2` 个大小 2 的子组，`:13-20`，`num_ranks` 必须偶数）。

### 4.1 meta→cuda 物化（`:22-33`）

```python
plan = deep_ep.BufferAllocator()
bucket_0 = plan.allocate((3, 5), torch.float32)     # :23
bucket_1 = plan.allocate((7,), torch.bfloat16)      # :24
assert bucket_0.is_meta and bucket_1.is_meta        # :25
buffer = deep_ep.BucketBuffer(group=[group_0, group_1],
                              allocation_plan_or_num_bytes=plan, explicitly_destroy=True)  # :27-28
assert bucket_0.is_cuda and bucket_1.is_cuda        # :29  物化后原地变成 CUDA 张量
assert buffer.storage.dtype == torch.uint8          # :30  单一 uint8 存储
assert buffer.storage.numel() == plan.num_bytes     # :31
assert len(buffer.contexts) == 2                    # :32  两个 group → 两个 context
assert bucket_0.data_ptr() == buffer.storage.data_ptr()  # :33  首个分配落在存储起点
```

`BufferAllocator`（`deep_ep/buffers/allocator.py:10-11`）的 `allocate`（`:18-24`）返回**meta 占位张量**，
按 `alignment = _C.get_num_allocation_alignment()`（2 MiB）记录 `(tensor, offset)`、累加
`num_bytes`；`materialize`（`:26-36`）在拿到一块 `uint8` 存储后，对每个占位张量
`storage.narrow(0, offset, nbytes).view(dtype).view(shape)` 并 `torch.utils.swap_tensors`（`:35`）
**就地替换**——于是外部持有的名字 `bucket_0`/`bucket_1` 自动变成 CUDA 张量。三个断言
（`:31` 长度、`:33` 首指针）把「物化 = 就地把 meta 张量换成存储视图」的行为钉死。

一道**跨库脚注**：`BufferAllocator` 的对偶在 DeepGEMM 侧是 `deep_gemm.mega.SymmBuffer`
（第 42 篇）：同样是「先规划再落地单块 storage」，但 SymmBuffer 用 `torch.empty` + 手工
`view` 列表（`deep_gemm/mega/__init__.py:50-68`），没有 `swap_tensors` 这层「占位张量透明替换」。

### 4.2 多 group 与「重新构造」模式（`:35-44`）

```python
bucket_0.fill_(rank_idx); bucket_1.fill_(rank_idx + 1)      # :35-36
assert (bucket_0 == rank_idx).all(); assert (bucket_1 == rank_idx + 1).all()
buffer.destroy()                                            # :39

buffer = deep_ep.BucketBuffer(group_0, plan.num_bytes, explicitly_destroy=True)  # :41
assert buffer.storage.is_cuda and buffer.storage.dtype == torch.uint8
assert buffer.storage.numel() == plan.num_bytes
buffer.destroy()                                            # :44
```

`:35-38` 验证物化后的张量真的映射到本 rank 的私有存储（写主颜色、读回一致）。`:41-44` 是**第二种
构造方式**——直接给 `num_bytes`（不传 plan），得到一个只有单 group 的 buffer，`storage` 长度与
plan 一致。这解释了 `BucketBuffer.__init__` 的 `allocation_plan_or_num_bytes` 二态参数
（`deep_ep/buffers/bucket.py:184` 起）：传 `BufferAllocator` → 按计划物化 + 多 context；
传 `int` → 只分配裸存储 + 单 context。

`get_context_idx`（`bucket.py:295-297`）在这里被反复调用：
`assert len(self.groups) == 1 or group is not None, '`group` is required when the buffer has
multiple groups'`——**buffer 拥有多个 group 时，每次集合调用必须显式指定 group**，这正是
`test_allocation.py` 要用 `group=[group_0, group_1]` 与一个大小 2 的子组来测的点。

---

## 5. H100 真机实测

`harness/exp_deepep_bucket_tests.py`（单进程纯 host + 8×H100 NCCL 两段）复现三份测试的**全部
测试侧语义**：

- **Part A（8×H100 NCCL）**：`test_all_gather` 的参考构造——每 rank 造 shard、
  `all_gather_into_tensor` 得参考、验证「结果 = 各 rank shard 按 rank 序拼接」，并坐实
  `all_gather_single`/`reduce_scatter_single` 在 stock torch 2.12 缺失（用 `*_tensor` 等价）。
- **Part B（8×H100 NCCL）**：`test_session` 的三件参考 + 容差梯度——all_gather 逐位相等、
  all_reduce/reduce_scatter 在 1e-5 内；并实测「反序环重建 vs `reduce_scatter_tensor`」在该
  8×NVLink 机上**恰好逐位**（NCCL ring 序），说明测试用 1e-5 属保守而非必要。
- **Part C（单进程）**：`test_session`/`test_allocation` 的缓冲区算术（`align` 公式、2 MiB 对齐、
  bump 分配越界）、`BufferAllocator` 的纯 torch 物化复刻（用真实 2 MiB 对齐值）。
- **Part D（单进程）**：`get_context_idx` 的多 group 断言语义、`get_theoretical_num_sms` 对
  `all_gather` 恒返回 0。

真实设备路径（`deep_ep.BucketBuffer` 的 C++/NCCL GIN kernel）**仅分析**——本机 NCCL 版本闸门
（第 76/86 篇）拦在 `import deep_ep`。

**实测**（`harness/exp_deepep_bucket_tests.out.txt`，H100 8 卡，torch 2.12.0a0 / CUDA 13.2 / NCCL 2.29.7）：

- Part C：`numel=524288`（对齐 `num_ranks*4`）、session 缓冲 `6291456` B（3 段 + 2 MiB 对齐）；
  `BufferAllocator` 两次 2 MiB 对齐段 → `plan.num_bytes=4194304`，物化后
  `bucket_0.data_ptr()==storage.data_ptr()`、`bucket_1` 偏移恰 **2 MiB**；bump 分配 `(0, 64)`、越界抛
  `RuntimeError`。
- Part D：多 group + `group=None` 被断言拦下；`get_theoretical_num_sms('all_gather')==0`。
- Part A：`all_gather_into_tensor` 在 shard∈{8, 8192, 524288} 三档**逐位**等于「各 rank shard 按 rank
  序拼接」；顺带坐实 `dist.all_gather_single` 在 stock torch 2.12 **不存在**。
- Part B：all_gather 逐位；reduce_scatter 反序环重建 relerr **0.000e+00**（该机 8×NVLink 恰好逐位，
  但测试仍保守用 1e-5）；all_reduce `sum*scale=0.25` 匹配。
- **总判定 ALL PASS**。

（三份测试的行号已逐条核对 `~/github/DeepEP@93eb6eb`；`harness/exp_deepep_bucket_tests.py` 按
文件路径引用其参考语义。）

---

## 6. 边界与结论

- **本机不能跑端到端 bucket kernel**：`deep_ep._C` 依赖 NCCL ≥2.31（第 76/86 篇），且
  `test_session.py` 依赖 DeepEP 对 torch 的 `*_single` 补丁。故本篇的「真实设备」部分为源码精读。
- **能跑的部分**：三份测试的参考语义、逐位判据、缓冲区算术、分配物化全部是纯 host/torch，
  在 H100 8 卡上真机复现。
- **收口**：至此 DeepEP 的 `tests/{ep,bucket,buffer,pp,engram,comm,utils}` 七个子目录的验收测试
  （`test_ep.py` 第 75、`test_prefetch/reduce_grads` 第 90、`test_barrier` 第 91、`test_gate` 第 106、
  `test_pp/engram` 第 112、`test_mixed/perf` 第 84、`test_all_gather/session/allocation` 本篇）**全部
  被单独拆过**，DeepEP 侧「测试即正确性规范」这条线收口。
