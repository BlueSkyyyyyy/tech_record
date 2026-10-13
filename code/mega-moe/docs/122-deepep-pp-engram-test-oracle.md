# DeepEP 的两类非 all-to-all 通信原语（PP send/recv 与 Engram fetch）的验收测试即正确性规范

> 本专题（DeepSeek Mega MoE 前向 / 反向算子实现解析）的第 122 篇，对应路线图 **P112-1**。
> 前面 121 篇把 DeepEP 的**通信原语实现**几乎拆尽：EP all-to-all（第 7/58/61 篇）、bucket 集合通信
> （第 64/72/112 篇）、流水线并行 send/recv（第 68 篇）、Engram 远程抓取（第 69 篇）；
> 也把**每一族 Buffer 的验收测试**陆续做成了「测试即正确性规范」：EP（第 75/85 篇）、
> bucket session / mixed / perf（第 83/84/94 篇）、EPLB 冗余权重（第 100 篇）、barrier（第 101 篇）、
> PP send/recv 与 Engram fetch 这两族**却一直只被实现篇引用**——`tests/pp/test_pp.py`（152 行）
> 与 `tests/engram/test_engram.py`（155 行）从没被单独读过。本篇补上。
>
> **它们到底定义了什么「对」**：PP 测试定义的是「**在途张量背压下的收发调度**」——一个用整数时间戳
> 编排出的无死锁收发序列，不是算子的数值，而是**通信协议**；Engram 测试定义的是「**带索引远程读**」
> 的参考语义——把本 rank 的 shard 与所有 peer 的 shard 用 all-gather 拼成全局表，再按索引 gather
> 出「每个 token 该拿哪些 entry」，并用正序/逆序两种等待顺序验证层之间独立。两者共同回答了
> 「这两类非 all-to-all 原语在数值/协议层面算什么对」。
>
> **可跑性（关键）**：两层。① **端到端 buffer 路径本机跑不了**——两个测试都要构造
> `deep_ep.PPBuffer` / `deep_ep.EngramBuffer`，它们直接调 C++ 扩展 `_C.PPBuffer` / `_C.EngramBuffer`，
> 吃 NCCL 设备 API（`ncclDevComm`/`ncclGin`）；本机容器里 `deep_ep._C` 编译扩展根本不可导入
> （`deep_ep/utils/event.py:5` 的 `from deep_ep._C import EventHandle` 抛 `ModuleNotFoundError`），
> 即便导入成功，`deep_ep/__init__.py:55` 的 `check_nccl_so()` 还要卡 NCCL 版本、真正跑 kernel 更需
> ≥2.31 的 NCCL（与路线图「阻塞」同源）。故 **kernel 端到端仅源码分析**。
> ② **测试侧规范逻辑全在纯 Python/张量里，H100(sm90) 真机可跑**——PP 的调度生成只用到
> `random` + `torch.randn`，Engram 的参考 oracle 只用到 tensor gather。本篇 H100 实测
> **72 PASS / 0 FAIL**，见 `harness/exp_deepep_pp_engram_test.{py,sh,out.txt}`。
> 行号对照 DeepEP commit `93eb6eb`（v2.5.0）。目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 定位：这两份测试回答的问题

DeepEP 的四类通信设施，各有一份旗舰验收测试：

| 设施 | 实现篇 | 验收测试 | 测试即规范篇 | 测试定义的核心语义 |
|------|--------|----------|--------------|--------------------|
| EP all-to-all | 第 7/11/58/61 篇 | `tests/ep/test_ep.py` | 第 85 篇 | dispatch/combine 逐位 + expand 不变量 + 两套 psum |
| bucket 集合通信 | 第 64/72/112 篇 | `tests/bucket/test_*.py` | 第 83/84/94 篇 | 会话式临时分配 + 混合集合算子 |
| **PP send/recv** | 第 68 篇 | `tests/pp/test_pp.py` | **本篇** | **在途背压下的无死锁收发调度** |
| **Engram fetch** | 第 69 篇 | `tests/engram/test_engram.py` | **本篇** | **带索引远程读的参考 gather** |

与前三类的区别在**判据形态**：

- EP/bucket 测的是**数值**——`torch.equal` / `calc_diff` 比对张量内容；
- PP 测的是**协议**——`generate_stress_ops` 生成一个带时间戳的 send/recv 序列，测试「缓冲区容量 N
  之下收发不会互相饿死」，`torch.equal` 只用来确认「收到的正是对端发的」；
- Engram 测的是**索引语义**——参考侧把「远程表」用 all-gather 展平成全局表再 gather，测试关心
  「按索引抓到的是不是正确 owner 的那一行」。

---

## 2. 顶层结构与文件地图

两个文件结构高度对称（各自一个顶层 `test()` + `__main__` 里的 `argparse` + `mp.spawn`）：

```
tests/pp/test_pp.py
├── :3-6   os.environ.pop('NCCL_MIN/MAX_NCHANNELS'/'NCCL_NVLS_ENABLE')   # 必须在 import torch 之前
├── :19-50 generate_stress_ops(...)            # ★ 无死锁收发调度（纯 Python/张量）
├── :55-134 test(local_rank, num_local_ranks, args)   # 起组、建 PPBuffer、压测、profile
└── :137-152 __main__ / argparse / torch.multiprocessing.spawn

tests/engram/test_engram.py
├── :6-9   import deep_ep；init_dist/dist_print；per_token_cast_to_fp8；bench_kineto
├── :14-130 test(...)   # 建 EngramBuffer、写本地 shard、★ 参考 gather oracle、profile
│         └── :58-67 gen_indices()             # ★ 索引生成（表偏移 + 随机槽）
└── :133-155 __main__ / argparse / spawn
```

`test()` 的第一句都是 `init_dist(local_rank, num_local_ranks)`（PP `:56` / Engram `:15`），返回
`(rank, world_size, group)`；`__main__` 都用 `torch.multiprocessing.spawn` 起多进程（PP `:152` /
Engram `:155`）。这一层（起组、播种、打印）已在第 118 篇拆透，本篇不重复。

两者都是 `@torch.inference_mode()`（PP `:54` / Engram `:13`）——纯通信测试不需要构图。

---

## 3. `tests/pp/test_pp.py`：在途背压下的无死锁收发调度

### 3.1 `generate_stress_ops`（`:19-50`）：它到底生成什么

签名（`:19-20`）：`generate_stress_ops(rank_idx, num_ranks, num_sends, shape, num_max_inflight_tensors)`，
返回**本 rank 视角**的一个 op 列表。`assert num_max_inflight_tensors > 0`（`:21`）先钉死 N 为正。

**第一步：随机造「有向 pair」的收发时间戳**（`:22-31`）：

- 两个字典 `send_times` / `recv_times`（`:22-23`）以所有有序对 `(s, d)`（`s != d`）为键——但实际只会
  填充**相邻对**，因为 `:27` 的 `dst_rank_idx = (src_rank_idx + (±1)) % num_ranks` 把 dst 限死在
  环上邻居（±1）。这是 PP 的本质：**每个 rank 只跟前后两个邻居说话**。
- `:28` `st = random.randint(0, 10**8)` 是发送「计划时刻」；`:29` `rt = st + random.randint(1, 3*10**6)`
  是接收「计划时刻」，**恒晚于对应发送**（相差 1~3e6）。
- `:30-31` 分别把 st/rt 塞进对应 pair 的列表。注意一个 (s,d) pair 里 `send_times` 与 `recv_times`
  **各 append 一次、index 一一对应**：`send_times[(s,d)][i]` 与 `recv_times[(s,d)][i]` 是**同一次**
  随机生成的收发对。

**第二步：逐 pair 排序并做背压修正**（`:33-49`）：

```python
for (src_rank_idx, dst_rank_idx) in send_times:                  # :34  确定性顺序
    n = len(send_times[(src_rank_idx, dst_rank_idx)])            # :35
    sorted_send = sorted(send_times[(src_rank_idx, dst_rank_idx)])  # :36  升序副本
    sorted_recv = sorted(recv_times[(src_rank_idx, dst_rank_idx)])  # :37
    for i in range(n):                                            # :38
        # Keep each send after its previous `num_max_inflight_tensors` recv to avoid deadlock
        if i >= num_max_inflight_tensors:                        # :40
            sorted_send[i] = max(sorted_send[i], sorted_recv[i - num_max_inflight_tensors] + 1)  # :41-42
        sorted_recv[i] = max(sorted_recv[i], sorted_send[i] + 1)  # :43
        tensor = torch.randn(shape, dtype=torch.bfloat16, device='cuda')  # :44
        if src_rank_idx == rank_idx:   ops.append(('send', sorted_send[i], dst_rank_idx, i, tensor))  # :45-46
        if dst_rank_idx == rank_idx:   ops.append(('recv', sorted_recv[i], src_rank_idx, i, tensor))  # :47-48
ops.sort(key=lambda x: (x[1], x[3]))                              # :49
```

**核心不变量（`:39-43`）**：对每个有向 pair，**第 i 次 send 必须晚于第 (i−N) 次 recv**
（`:41-42`），即「同一时刻这个 pair 最多有 N 个张量在途」。这条修正和 `:43`（recv 恒晚于 send）
合起来，保证以时间戳全局排序执行时**每个 pair 的在途数上界恰为 N**：

> 在 `send_i` 之前，至多完成 `i−1` 次 send、至少完成 `i−N` 次 recv ⇒ 在途 ≤ `(i−1)−(i−N) = N−1`；
> 发出 `send_i` 后达到 N。这正是缓冲区 `num_max_inflight_tensors` 的容量。

**修正后仍单调**（H100 实测）：因为 `sorted_recv[i−N]+1` 抬高的只是少数时间戳，实测 18 组
（R×N×seed）里**每个 pair 的 send/recv 序列都保持非递增 → 配对仍是 FIFO**（见 §6）。

**`i` 字段（元组第 4 个）**：它是该 pair 内的序数，`:49` 用它作**时间戳并列时的 tie-break**，也让
send 与 recv 在多个 pair 交错时仍能按 `(pair, i)` 对齐。

### 3.2 为什么需要背压：利齿负样本

`generate_stress_ops` 若不做 `:40-42` 的修正，随机时间戳会让某次 send 落在它前 N 次 recv 之前，
于是缓冲区被击穿。H100 实测（`harness/exp_deepep_pp_engram_test.py` 的 `_gen_naive`）：对同样的
18 组配置，**去掉修正后有 31 处容量溢出**，而修正命中次数也恰是 **31**——即每一次抬高正好抵消一处
潜在溢出。这是本篇最主要的「为什么」：**这条看似随手的 max 就是无死锁的全部。**

### 3.3 执行环与判据（`:74-95`）

`test()` 对每个 seed（`:76`）：`torch.manual_seed(42+seed)` + `random.seed(42+seed)`（`:78-79`）
——**两条 RNG 都要同步**，因为调度用 `random`、张量内容用 `torch`；各 rank 是独立进程，靠相同的
种子得到**逐位相同的调度与张量**（这是 `torch.equal` 能成立的前提）。随后 `ops = generate_stress_ops(...)`
（`:80`）拿到本 rank 的 op 列表，按顺序执行（`:82-93`）：

- `send` → `buffer.send(tensor, peer)`（`:85`）；
- `recv` → 先 `torch.empty_like` 再 `buffer.recv(result, peer)`，然后
  `assert torch.equal(result, tensor)`（`:89`）——**收到的必须正是对端发的**；
- `:91-92` 用 `torch.cuda._sleep(int((timestamp - prev) / 10**8 * num_sleep_cycles))` 把时间戳差距
  翻译成 GPU 空转，**让计划时刻变成真实的时序压力**（而不是一次性把 op 全发完）。

默认 `--num-stress-iterations=4`、`--num-sends=128`（`:143-144`），故每 rank 每个 seed 会跑
~128 个随机 pair 的收发。

### 3.4 profiling 矩阵与带宽账目（`:97-130`）

`:99` `num_approx_rdma_cycles = int(num_max_tensor_bytes * 2 / get_rdma_gbs() * 1.5)`——用「来回字节数
÷ RDMA 带宽 ×1.5 余量」估算「一趟 RDMA 往返要多少 cycle」，供 `:115` 的 `torch.cuda._sleep` 模拟
「把 RDMA 延迟藏起来」。

矩阵是 **`hide_rdma_latency ∈ {True, False}` × `num_concurrent ∈ [1, min(3, N)]`**（`:105-106`）：

- `:111` `torch.zeros((131072, 32768), dtype=torch.int)` 故意分配一块大张量**冲掉 L2**，保证测的是真
  冷启动；
- `:114-115` `hide=True` 时 sleep 掉估算的 RDMA 往返 —— 模拟「接收端不等、先干别的」；
- `:126-129` 带宽账目：send 恒按**来回 2×** 算（`2*num_max_tensor_bytes/send_t`），recv 在
  `hide=True` 时也按 2×（因为隐藏了延迟、有效搬运翻倍）、否则按 1×。

`bench_kineto(..., kernel_names=('send_impl', 'recv_impl'), barrier_comm_profiling=True, ...)`
（`:119-122`）按 kernel 名把 `send_impl` / `recv_impl` 的耗时拆开——`bench_kineto` 基座见第 40 篇。

### 3.5 host API（`deep_ep/buffers/pp.py`）

测试调用的 `deep_ep.PPBuffer`（`pp.py:12`）是 `BufferBase` 的薄派生（生命周期契约见第 121 篇）：

- `__init__`（`:15-46`）：`comm.get_nccl_comm_handle(group)`（`:34`）→ 直接构造 `_C.PPBuffer(...)`
  （`:37`，**设备侧阻塞点**）→ 拿 `self.storage` → `torch.cuda.synchronize()` + `group.barrier()`
  + 再 synchronize（`:44-46`，保证 peers 可见）；
- `send`（`:48-54`）/ `recv`（`:56-68`）：只是 `self.runtime.send/recv(x, peer, num_sms)` 的转发，
  `num_sms=None` 时传 0（由 runtime 选全部 SM）；
- `destroy`（`:69-72`）：`super().destroy()` 后清 `storage` / `nccl_comm_handle`。

---

## 4. `tests/engram/test_engram.py`：带索引远程读的参考 gather

### 4.1 配置与容量推断（`:16-40`）

- `dtype`（`:16`）随 `--use-fp8` 在 `float8_e4m3fn` / `bfloat16` 间切；
- `num_entries_per_layer`（`:17`）由逗号分隔字符串解析，`num_layers`（`:18`）是其长度；
- `num_sf_packs = ceil_div(hidden, 128)`（`:19`）——FP8 时每 128 个 hidden 一个 scale pack；
- `get_storage_size_hint`（`:20-22`）与 `get_theoretical_config`（`:24-25`）都是 `EngramBuffer` 的
  classmethod，底层转调 `_C.EngramBuffer.calculate_storage_size`（`engram.py:113-129`）与
  `_C.EngramBuffer.get_theoretical_config`（`:131-144`）——**设备侧阻塞点**；
- `num_qps = args.num_qps if args.num_qps else num_allocated_qps`（`:26`）——0 表示用推荐值。

### 4.2 写本地 shard 与复制 scale factor（`:42-56`）

每个 rank 为每层造一个 `[num_entries, hidden]` 的本地 bf16（`:45`）。FP8 分支（`:46-51`）：

- `per_token_cast_to_fp8(local_bf16)`（`:47`）得到「fp8 storage + fp32 SF」——**参考量化器**见第 114 篇
  （`deep_ep/utils/math.py:31-41`，gran 硬编码 128、未打包 UE8M0）；
- `:49` 分配 `[num_ranks * num_entries, sf_cols]` 并 `dist.all_gather_single(sf, local_sf)`（`:50`）
  ——**把 SF 复制到每个 rank**，注释（`:48`）直说原因：「让任何 rank 都能在本地抓到任何 entry 的
  scale factor」。这是 Engram 的关键：**数据在远端、但 SF 被冗余到本地**，抓取时才不用再为 SF 单独
  发一次远程读。

`buffer.write(local_storages, sfs=sfs)`（`:56`）把本地 shard 写进 NCCL window。

### 4.3 `gen_indices`（`:58-67`）：索引的确定性结构

```python
def gen_indices():
    indices_per_layer = []
    for num_entries in num_entries_per_layer:                       # :60
        table_size = num_ranks * num_entries // num_entries_per_token  # :61  全局表行数 / npt（向下取整）
        table_offsets = torch.arange(npt, ...) * table_size          # :62-63  每个 token 槽的基址
        indices_per_layer.append(table_offsets + torch.randint(
            0, table_size, (T, npt), ...))                           # :64-66  基址 + 槽内随机
    return torch.stack(indices_per_layer)                           # :67
```

**结构要点**：`global 表`共 `num_ranks * num_entries` 行。把它按 `num_entries_per_token = npt` 个
不相交的区间切开，每个区间有 `table_size = (R*ne)//npt` 行；第 `slot` 个槽只从
`[slot*table_size, (slot+1)*table_size)` 取。于是：

- 同一 token 的 npt 个 entry **严格递增、互不相同**（H100 实测 sort+diff 全 > 0）；
- **每个 slot 的 range 不相交**，保证一个 token 不会重复拿同一个 entry；
- `//` 是**向下取整**，`R*ne` 往往不能被 npt 整除——实测 `R*ne = 2048/2068`、`npt=24` 时
  每层**剩余 `[8, 4]` 行永远不会被任何槽取到**（`2048 − 24·85 = 8`）。这是「表尾留空」的刻意设计，
  不是 bug。

### 4.4 参考 gather oracle 与两种等待顺序（`:69-96`）

判据只对 `--skip-check` 关闭时生效（`:70`）。参考路径：

1. 把各 rank 的本地 storage 用 `dist.all_gather_single` 拼成 `global_storage[layer]`
   （`:71-75`，形状 `[R*ne, hidden]`）——**模拟「把所有远端表拼成一张本地全局表」**；
2. 对**正序与逆序**两种层序各跑一遍（`:78`：`for layer_order in (range(L), reversed(range(L)))`）；
3. `ref_data_per_layer[l] = global_storages[l][indices[l].view(-1)].view(T, -1)`（`:80-83`）
   ——gather 后按 token 重排；FP8 时 `ref_sf_per_layer[l]` 同理（`:84-87`）；
4. `hooks = buffer.fetch(indices, num_qps=num_qps)`（`:89`）返回**每层一个 wait hook**；
5. 按 `layer_order` 逐个 `hooks[layer_idx]()` 并 `torch.equal`（`:93`/`:95`）。

**为什么正序/逆序都要测**：`fetch` 一次性发出**所有层**的 RDMA get、返回一组独立 hook，调用方可以
**任意顺序 wait**。正序/逆序都得到同一结果 = 「层之间没有隐藏的完成依赖」。

H100 实测的**独立复刻**（`harness/exp_deepep_pp_engram_test.py` Part B）：把参考路径换成「逐元素远程
读模型」——对每个索引 `idx`，`owner = idx // num_entries`、`row = idx % num_entries`，直接读
`local[owner][row]`——与 `global[idx]` 的 gather **逐位相等**，且正序/逆序结果逐位相等。这把测试
隐含的「index → (owner rank, row)」映射显式化。

### 4.5 profiling 账目（`:98-126`）

- `fetch_and_wait`（`:105-109`）：一次 `fetch` 后**逐层 wait**；
- `bench_kineto(..., kernel_names=('engram_fetch_impl', 'engram_fetch_wait_impl'), ...)`
  （`:111-116`）按名拆「发射」与「等待」两条 kernel；
- `total_t = issue_t + wait_t * num_layers`（`:118`）——注释（`:117`）点明 `wait_t` 是**每 launch 的
  平均**，而每次迭代要 wait `num_layers` 次，故总等待要乘层数；
- `mpps`（`:119`）用「层×token×entry / total_t」算百万次抓取/秒。

### 4.6 host API（`deep_ep/buffers/engram.py`）

`EngramBuffer`（`engram.py:31`）比 PPBuffer 复杂得多（hybrid 模式、共享句柄、GIN 后端）：

- 构造（`:41-104`）里 `:62-65` 按「本节点 16 卡」预留 RDMA storage（超显存时顺手设
  `NCCL_WIN_STRIDE`）、`:67` `check_nvlink_connections`（第 119 篇）、`:75` hybrid 模式经
  `_C.EngramBuffer.create_shared_handle` 建跨进程共享句柄、`:81-85` 用 `EnvOverride` 临时把
  `NCCL_GIN_TYPE=3`、`:87` 构造 `_C.EngramBuffer`；
- `get_storage_size_hint`（`:113-129`）/`get_theoretical_config`（`:131-144`）转 `_C`；
- `set_config`（`:146-151`）/`write`（`:153-155`）/`fetch`（`:157-174`）：`fetch` 里
  `num_qps = self.num_allocated_qps if num_qps == 0 else num_qps`（`:170`）并有
  `assert num_qps <= self.num_allocated_qps`（`:172`）。

---

## 5. 阻塞落点（kernel 端到端，仅分析）

| 落点 | 证据 | 说明 |
|------|------|------|
| `deep_ep._C` 不可导入 | `deep_ep/utils/event.py:5` `from deep_ep._C import EventHandle`；H100 实测 `ModuleNotFoundError` | 本机容器未安装可用的 DeepEP 扩展，引导阶段即失败 |
| NCCL 版本闸门 | `deep_ep/__init__.py:55 check_nccl_so()` | 二进制级比对 linked vs loaded NCCL（第 86 篇），本机不匹配 |
| PP 设备入口 | `pp.py:37` `_C.PPBuffer(...)` | 直接构造 C++ buffer |
| Engram 设备入口 | `engram.py:87` `_C.EngramBuffer(...)`、`:124`/`:141` 的 `_C` 静态推断 | 需要 GIN/RDMA 设备 API |

附带发现：两个测试都调用 `dist.all_gather_single`（PP 没用；Engram `:50`/`:74`），但
**该函数既不在本机 torch.distributed、也不在 DeepEP 的 Python 树**（`grep` 全仓为空）——它是
DeepEP 补丁版 `torch.distributed` 提供的扩展 op。本实验用 `torch.cat` / 显式索引复刻其语义
（「把各 rank 的 shard 拼进一块预分配张量」）。

---

## 6. H100(sm90) 实测小结

`harness/exp_deepep_pp_engram_test.{py,sh,out.txt}`，**72 PASS / 0 FAIL**
（`tests/pp/test_pp.py` sha256[:16] `a4a4368bcc9f52fb`、`tests/engram/test_engram.py`
`23d15e87d1ee9851`、`deep_ep/utils/math.py` `ed0dfb4bf9893769`）：

- **Part A（PP 调度，`:19-50` 原样 ast 抽取运行）**：
  - 6 组 `(R, num_sends, N)` × 3 seed = **18 组**，每个 op 都指向环上相邻 rank；
  - 全局时间线下**每个 pair 在途 ≤ N** 全部成立（含 N=1 的 R=8、N=3 的 R=3 奇数环）；
  - 背压抬高命中 **31 次**，修正后每 pair 的 send/recv 序列**仍单调**（FIFO 可配对）；
  - **利齿负样本**：去掉 `send_i ≥ recv_(i−N)+1` 修正后 **31 处容量溢出**被击穿；
  - 同步 `torch`+`random` 种子后同一 `(pair,i)` 的 send/recv 张量**逐位一致**；
  - 判据利齿：克隆 `torch.equal` 真、单元素 +1 后假。
- **Part B（Engram oracle）**：
  - `indices` 形状 `[2,128,24]`、范围合法、每槽 range 不相交、同 token 内严格递增（3 seed 全过）；
  - **floor 除法留空**：`R=4`、`ne ∈ {512,517}`、`npt=24` 时每层剩余 `[8, 4]` 行永不取到；
  - gather-oracle == 逐元素远程读模型（`owner`/`row` 映射）**逐位相等**、正逆序等待结果相同；
  - FP8 分支：`per_token_cast_to_fp8` 跑通，复制 SF 后 `owner/row` 显式远程读 == gather 结果逐位。
- **Part C（阻塞）**：`import deep_ep` 非零退出、失败落在 `deep_ep._C`；两族的 host 入口都经 `_C`；
  `dist.all_gather_single` 缺定义。
- 环境：NVIDIA H100 80GB HBM3、cc 9.0、torch 2.12.0a0、NCCL (2,29,7)、DeepEP `93eb6eb`。

---

## 7. 与专题的关系与关联篇目

- **第 68 篇**（PP send/recv 实现）回答了「`pp_send_impl`/`pp_recv_impl` 怎么用 TMA + inflight 做
  环形流水」；本篇补上它**对外行为算什么对**——那个 `inflight` 就是 `num_max_inflight_tensors`，
  而 `generate_stress_ops` 的背压修正正是对它的压力测试。
- **第 69 篇**（Engram fetch 实现）回答了「`engram_fetch_impl` 怎么做 indexed RDMA gather」；本篇
  补上它的**参考语义**——`global[idx]` 的 all-gather+gather 等价物。
- **第 118 篇**（DeepEP 进程引导）、**第 121 篇**（BufferBase 生命周期）、**第 119 篇**
  （NVLink 自检）是本篇测试运行时反复踩到的三层地基，不重复。
- **第 111 篇**（MoE 反向 H100 复刻）与本篇同属「把跑不了的 kernel 用测试侧/参考侧逻辑在 H100 上
  验收」的思路：kernel 端到端跑不了时，把**它承诺的不变量**抽出来单独验证。
