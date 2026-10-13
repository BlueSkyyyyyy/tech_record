# DeepEP 的确定性 dispatch 排序：`EPHandle` 路由句柄与 `deterministic_sort`（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P52-1** 增量（第 62 篇）。
> 前面第 7 篇（`docs/07`）拆了 dispatch/combine 的**数据流**、第 11 篇拆了 DeepGEMM 内嵌版
> 的 ring/NVLink 重叠、第 48/51 篇拆了 combine 归约与 dispatch 拷贝 epilogue。但它们都建立在
> 一个隐含前提上：**「接收到的 token 排列是稳定的」**。实际上 dispatch 的槽位分配是**无锁原子
> 计数**，接收顺序随硬件时序而变；DeepEP 靠一个**纯 Python 后处理排序**
> `EPHandle.deterministic_sort` 把顺序「规范化」回源 token 全局索引序，从而让整个 MoE 前向
> **逐位可复现**。
>
> 这一层是 **架构无关** 的：`deterministic_sort`（`deep_ep/buffers/ep.py`）只做
> `torch.sort` / `index_select` / `scatter_add_`，**完全不碰 NCCL、对称显存或 kernel**。
> 因此本机 H100（sm90）可以 **逐行原样抽取真实源码并运行**，再用独立 host oracle 对拍——
> 这与第 24/26/31/34/56 篇「抽取架构无关设备代码在 sm90 跑真实数字」的方法一致。
>
> 源码坐标：`~/github/DeepEP`，公开版 commit `93eb6eb`（v1.2.1-41）。容器内安装的
> `ep.py` 与该源代码**逐字节相同**（抽取片段 sha256 一致，见 §5）。

## 1. 为什么需要确定性排序

### 1.1 dispatch 的槽位分配本来就是不确定的

dispatch 把 token 送到「目标 rank 的对称缓冲槽位」。槽位号来自一个**对称内存里的原子计数器**：

```
deep_ep/include/deep_ep/impls/ep/dispatch.cuh:341
    stored_dst_slot_idx = atomicAdd(workspace_layout.get_scaleup_atomic_sender_counter()
                                    + stored_dst_rank_idx, 1);
```

`atomicAdd` 的**返回顺序取决于 warp/CTA 的调度时序**：同一批 token、同样的 top-k，两次运行
可能落进不同的物理槽位（`dispatch.cuh:334` 的 `kReuseSlotIndices` 分支可复用旧索引，但那只在
「cached dispatch」时成立）。于是**接收端看到的 token 行序**在多次运行间可能不同。

对推理无所谓；但对训练/评测，行序变化会让：
- 逐位复现（bitwise reproducibility）失败，难以定位数值回归；
- 与 batch-invariance 相关的算子（如第 48 篇讨论的 `deterministic` 开关）失去意义。

### 1.2 两个开关

DeepEP 用两层开关处理：

- `EPBuffer(..., deterministic=True)`（`ep.py:296`）——**路由算法**本身走确定性分支；
- `torch.use_deterministic_algorithms` 与 `fill_uninitialized_memory` 不能同时开
  （`deep_ep/utils/envs.py:184-190` 的 `check_torch_deterministic`，因为 `torch.empty()`
  的初始化 kernel 可能与通信流重叠）。

当 `deterministic=True` 时，dispatch 结束后挂一个 epilogue（`ep.py:734-743`）：

```
ep.py:734
    if self.deterministic:
        deterministic_epilogue = functools.partial(
            handle.deterministic_sort, do_cpu_sync, is_cached_dispatch,
            recv_x, recv_sf, recv_topk_idx, recv_topk_weights, channel_linked_list)
        if deterministic_by_hook:
            event_overlap.register_hook_after_wait(deterministic_epilogue)
        else:
            deterministic_epilogue()
```

即：**排序发生在 dispatch 落地之后、把结果交给用户之前**（`async_with_compute_stream` 时挂到
event 的 `after_wait` 钩子上，避免打断通信/计算重叠）。

## 2. `EPHandle`：把前向的路由决策冻结下来

`EPHandle`（`ep.py:27`）是 dispatch 返回、combine 消费的**路由句柄**。它存的不只是
`topk_idx`，还有一整套「谁被送到了哪个专家/槽位」的前缀和与元数据，正是排序和反向所依赖的：
关键字段（`ep.py:28-62` 文档、`ep.py:82-106` 存储）：

| 字段 | 含义 |
|---|---|
| `topk_idx` | 原始 `[T, topk]` 专家索引（**只读**，见下） |
| `psum_num_recv_tokens_per_scaleup_rank` | 每 scaleup rank 去重后的接收数前缀和（末元素=总接收数） |
| `psum_num_recv_tokens_per_expert` | 每本地专家（对齐填充后）的接收数前缀和 |
| `num_unaligned_recv_tokens_per_expert` | 每本地专家**未对齐实际数**（仅 expand 模式填充） |
| `recv_src_metadata` | `[num_recv_tokens, 1+1+topk]`：`[src_token_global_idx, src_rank, slots...]` |
| `dst_buffer_slot_idx` / `channel_linked_list` | 槽位索引 / hybrid 链式列表 |

两个语义要点：
- **`topk_idx` 有版本守卫**（`ep.py:105`、`ep.py:108-111`）：句柄持有期间改动它会被
  `_version` 检查拦下，因为 combine 反向要用它重新解析路由。
- `recv_src_metadata` 的**第 0 列是「源 token 全局索引」**——这正是排序键；第 2 列起是槽位。
  其字节布局在设备侧由 `layout/ep/token.cuh:63-77` 的 `TokenLayout` 定义（第 61 篇逐字段讲过）。

## 3. `deterministic_sort` 逐段精读（`ep.py:113-205`）

方法是 **原地（in-place）** 的：把 `recv_x`、`recv_sf`、`recv_topk_weights`、`recv_topk_idx`
以及句柄的 `recv_src_metadata` 重排成规范序。

### 3.1 排序键与「排序前快照」

```
ep.py:136-140
    if not is_cached_dispatch:
        self.cached_recv_src_metadata_before_sort = self.recv_src_metadata.clone()
    assert self.cached_recv_src_metadata_before_sort is not None
    sort_keys = self.cached_recv_src_metadata_before_sort[:, 0]
```

关键设计：**排序键永远取自「排序前的快照」**。原因是 cached dispatch 时
`recv_src_metadata` **不会被重新生成**（拷贝 epilogue 直接按它摆位），所以必须缓存一份原始
元数据，才能在第二次排序时仍得到同一个规范序。快照变量在 `ep.py:101` 初始化为 `None`。

OOB（越界）尾块——当没有 CPU 同步时 `recv_src_metadata` 里可能有尚未填充的尾行：

```
ep.py:142-147
    num_recv_tokens = ...psum_num_recv_tokens_per_scaleup_rank[-1] if not do_cpu_sync
                      else self.recv_src_metadata.shape[0]
    if not do_cpu_sync:
        oob_tokens_mask = arange(...) >= num_recv_tokens
        sort_keys[oob_tokens_mask] = torch.iinfo(sort_keys.dtype).max
```

把这些尾行的键设为 dtype 最大值，保证它们**沉到末尾**，不参与有效 token 的规范化。

### 3.2 两个工具闭包

```
ep.py:148-157
    get_reverse_permutation(perm): result[perm] = arange(...)   # 逆置换
    permute(tensor, orig_indices): tmp = tensor[orig_indices]; tensor.copy_(tmp)  # 原地置换
```

`permute` 用「先算 `tensor[idx]`（产生新张量）再 `copy_` 回去」实现原地重排——避免在
autograd/视图场景下破坏张量身份。

### 3.3 非 expand 分支（`ep.py:159-172`）

非 expand 模式：接收行的顺序就是普通 token 顺序，直接按 `sort_keys` 排序：

```
ep.py:159-172
    permute(recv_x, orig_indices); permute(recv_sf, ...); permute(recv_topk_weights, ...); permute(recv_topk_idx, ...)
    if not is_cached_dispatch:
        permute(self.recv_src_metadata, orig_indices)
        if channel_linked_list is not None:   # hybrid：把链式列表也按逆置换重映射
            channel_linked_list[valid_mask] = to_indices[channel_linked_list[valid_mask]]
```

注意 **cached 时只排数据张量、不排 `recv_src_metadata`**：因为 cached 模式下元数据已经与
目标槽位对齐、且被复用，重排它反而会破坏后续 combine。

### 3.4 expand 分支（`ep.py:174-205`）：专家块内规范化

expand 模式的展开数组是**按专家分组**的（组间有对齐填充），所以不能全局按 src 排，而要
**「先按专家、再按块内 src 索引」双键排序**：

```
ep.py:184-197
    src_token_global_index_max_x2 = 1e10
    expert_token_idx_start = psum_num_recv_tokens_per_expert - num_unaligned_recv_tokens_per_expert
    token_idx2expert_idx = bucketize(arange(M), expert_token_idx_start[1:], right=True)
    sort_keys = token_idx2expert_idx * 1e10
    slots = cached_recv_src_metadata_before_sort[:, 2:]
    src_global_idx = cached_recv_src_metadata_before_sort[:, 0]
    valid_mask = slots >= 0
    sort_keys.scatter_add_(0, slots[valid_mask],
                           -1e10//2 + src_global_idx[valid_mask].to(int64))
```

三个非平凡点：
1. **`expert_token_idx_start` 是「未对齐起点」**（`psum - unaligned`），配合
   `bucketize(..., right=True)` 把每个展开行映射到专家号——这正是句柄前缀和字段的用途。
2. **键编码** `expert*1e10 + (-5e9 + src)`：`-5e9` 让有效行恒小于填充行（填充行键为
   `expert*1e10`），于是 **稳定排序后「块内有效槽在前、按 src 升序、填充槽殿后」**。
3. **`scatter_add_` 而非赋值**：一个槽在多元数据行里可能被引用（同槽多 topk），
   累加保证键的构造对多次引用稳健。

```
ep.py:199-205
    orig_indices = torch.sort(sort_keys, stable=True).indices.to(int32)
    permute(recv_x, ...); permute(recv_sf, ...); permute(recv_topk_weights, ...)   # 注意：不含 recv_topk_idx
    to_indices = get_reverse_permutation(orig_indices)
    self.recv_src_metadata[:, 2:][valid_mask] =
        to_indices[self.recv_src_metadata[:, 2:][valid_mask]]
```

expand 模式**只排 `recv_x/recv_sf/recv_topk_weights`**（专家块内的 token 数据），并把
`recv_src_metadata[:,2:]` 的**槽位指针**用逆置换**重映射到新位置**（因为下游 combine 反向靠它
定位）。**cached expand 时整个分支被跳过**（第 174 行的 `elif not is_cached_dispatch`）：拷贝
epilogue 已按元数据摆放，无需再动。

## 4. 与其它层的连接

- **谁写 `recv_src_metadata`**：dispatch 的拷贝 epilogue（`impls/ep/dispatch_copy_epilogue.cuh`，
  第 51 篇）把 `src_token_global_idx`/`src_rank`/槽位写进元数据；排序消费它的第 0 列。
- **谁读排好序的槽位**：combine 把 `handle.recv_src_metadata`、`psum_*`、
  `channel_linked_list` 交给设备侧归约（`ep.py:820-837`），第 48 篇的 `combine_reduce` 逐 token
  读它定位贡献。
- **反向/cached dispatch**：`dispatch(..., handle=...)` 复用句柄（`ep.py:652-668`），文档明确
  写着 `topk_weights` 「can be optionally provided (e.g. for backward pass with cached expand)」
  ——即**反向用同一套路由、配上游梯度权重**；确定性排序保证这套复用不会因排列漂移而出错。

## 5. H100 实测（`harness/exp_deepep_deterministic_sort.py` + `.sh` / `.out.txt`）

实验**不 import 包**（本机 `deep_ep/__init__.py` 的 `check_nccl_so()` 会因 NCCL 2.29.7 拒绝），
而是 `importlib.util.find_spec` 定位已安装 `ep.py`、用 `ast` **逐行原样抽取真实的
`deterministic_sort`** 运行：

```
deep_ep ep.py     : /usr/local/lib/python3.12/dist-packages/deep_ep/buffers/ep.py
deterministic_sort: 6171 chars, sha256=22f3026c1dfbc9f4
```

该 sha256 与 `~/github/DeepEP` 源码（commit `93eb6eb`）抽取片段**完全一致**，证明跑的是真源码。
独立 oracle 用「朴素双键排序 + 逆置换」的另一种写法复刻语义。**7 PASS / 0 FAIL**：

| 用例 | 验证什么 | 结果 |
|---|---|---|
| A 非 expand + 非 cached | `recv_x/idx/weights/metadata` 同步按 order 重排；快照=排序前；二次排序幂等 | 5 PASS |
| B 非 expand 确定性 | 6 种接收顺序 → 输出**逐位一致**，且 = 按 src 索引升序 | 2 PASS |
| C 非 expand OOB | 越界尾行被压到末尾，尾部恰为 OOB 集合 | 2 PASS |
| D expand + non-cached | 块内有效槽按 src 升序、填充殿后；`recv_x`/weights 同步置换；槽位指针=逆置换且自洽 | 5 PASS |
| D expand + cached | 展开数组与元数据**均不变**（拷贝 epilogue 已摆位） | 1 PASS |
| E channel_linked_list | 链式列表按逆置换更新、越界项跳过 | 1 PASS |
| F expand 确定性 | 6 种接收顺序 → 规范化展开顺序**逐位一致** | 1 PASS |

核心结论：`deterministic_sort` 把「接收顺序不确定」这一物理事实**在 host 侧完全抹平**——
不论到达顺序如何，用户看到的都是「按源 token 全局索引（expand 下再按专家分块）的规范序」，
且这个规范序**逐位可复现**。

## 6. 小结与边界

- dispatch 的槽位分配（`dispatch.cuh:341` 的 `atomicAdd`）在多次运行间可变；`deterministic=True`
  时由 `EPHandle.deterministic_sort` 把顺序规范化回源 token 索引序。
- 方法**纯 host/Python、架构无关**：非 expand 是「按 src 索引全局排序 + 快照保护」，expand 是
  「`expert*1e10 + (-5e9 + src)` 双键稳定排序」，并用逆置换更新槽位指针。
- `deterministic_sort` 的**触发点在拷贝 epilogue 之后**（与本专题第 51 篇的
  `dispatch_copy_epilogue` 直接衔接）；它与 `EPBuffer(deterministic=True)` 的**路由**确定性
  是两个层面，别混淆。
- 边界：本机无法运行真机多进程 dispatch/combine（NCCL 版本阻塞，见 ROADMAP「阻塞」），
  所以这里验证的是**排序语义本身**（对真实源码逐行抽取运行），而非它与真实接收缓冲的端到端联调。

**本篇实测脚本**：`harness/exp_deepep_deterministic_sort.py` / `.sh` / `.out.txt`（H100 真机）。
