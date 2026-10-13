# DeepEP 的测试 / 进程引导层：`init_dist`、`init_seed`、`dist_print` 与 `EnvOverride`

> 本文对应路线图 **P108-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的**第 118 篇**。
>
> 前面 117 篇把 DeepGEMM / DeepEP 的 kernel、host 编排、测试基座、环境变量面几乎拆尽了，但还有一层
> **每个 DeepEP 测试跑之前都会执行、却从没有单独成篇**的东西：**进程 / 环境引导**。所有
> `tests/**/test_*.py` 的第一句都是 `rank, world, group = init_dist(local_rank, num_local_ranks)`，
> 之后每个 rank 靠 `init_seed` 播种、靠 `dist_print` 打印、构造 engram buffer 时靠 `EnvOverride`
> 临时改 `NCCL_GIN_*`。这些都是「跑起来之前」的 host 动作，**与具体 kernel 无关**，因此长期被
> 各篇当作 boilerplate 一句带过。
>
> 本篇把它单独讲透，并与两处邻居成对照：第 117 篇拆的是 DeepEP 的 **`EP_*` 环境变量面**
> （`get_env<T>` / persistent envs / `check_nccl_so`），第 40 篇拆的是 **DeepGEMM 的 `init_dist`**
> （`deep_gemm/utils/dist.py`）——两者同名不同物，本篇钉死 DeepEP 这一份。
>
> **可跑性**：整层是**纯 host / Python**，不碰 UMMA/TMA/NCCL 设备 API，也非 SM100 专属。
> 实验用 `sys.modules` 注入合成 `deep_ep._C`/`deep_ep.comm` 绕过 `import deep_ep` 的 `check_nccl_so`
> 闸门，然后**直接加载真实 `deep_ep/utils/envs.py` 源码**、真跑 `init_dist(0, 1)`（NCCL world_size=1）。
> **H100 真机 30 PASS / 0 FAIL**（`harness/exp_deepep_process_bootstrap.{sh,out.txt}`）。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - `deep_ep/utils/envs.py`：`EnvOverride:22-37`、`init_seed:40-51`、`get_local_seed:54-61`、
>   `get_global_seed:64-71`、`dist_print:74-86`、`init_dist:89-133`、`check_torch_deterministic:184-190`
> - 消费点：`deep_ep/buffers/engram.py:81-85`、`deep_ep/utils/refs.py:6/151`、
>   `deep_ep/buffers/ep.py:18-21/639/813`、`tests/**/test_*.py`
> - 实验：`harness/exp_deepep_process_bootstrap.py` + `.sh/.out.txt`

---

## 1. 为什么这层值得单独一篇

MoE 的 EP（专家并行）测试和普通单卡测试最大的不同，是它**天然是多进程 / 多 rank 的**：8 个进程各自
`import torch`、各自算随机数、各自构造 `EPBuffer`、各自跑 dispatch/combine，最后靠「同一份 Lua 心里
的期望」对拍。要让这 8 份独立进程「看到同一个世界」，测试入口必须做好三件事：

1. **握手成一个通信组**（否则 `dist.all_gather_object` 直接挂）；
2. **播种成可复现的伪随机序列**（否则「同 seed 逐位可复现」这种判据无从谈起）；
3. **日志与屏障同步**（否则 8 个 rank 的输出交错成一团乱码）。

`deep_ep/utils/envs.py` 的 `init_dist` / `init_seed` / `dist_print` 恰好就是这三件事，而 `EnvOverride`
是「临时、可回滚地改环境变量」的工具（engram buffer 构造时改 `NCCL_GIN_*`）。它们**全在 host、
全在 kernel 之前**，所以是理解 DeepEP 测试栈「怎么起来」的最后一块拼图。

一个关键对照：**同名函数在两个仓库里是两个东西**——DeepGEMM 的 `init_dist`
（`deep_gemm/utils/dist.py:10-35`，见第 40 篇）用 `init_method='env://'` 起组；DeepEP 的这一份用
`tcp://MASTER_ADDR:MASTER_PORT` 起组，还附带默认 dtype/device 的全局副作用。看代码时极易混淆。

---

## 2. 全景：`envs.py` 的功能分区

`envs.py`（283 行）其实是**四类东西混在一个文件里**：

| 分区 | 符号（行号） | 用途 | 消费点 |
|---|---|---|---|
| **进程引导** | `init_dist:89-133`、`init_seed:40-51`、`get_local/global_seed:54-71`、`dist_print:74-86` | 测试入口：起组、播种、打印 | 所有 `tests/**/test_*.py` |
| **环境工具** | `EnvOverride:22-37` | 临时改环境变量、退出即回滚 | `engram.py:81-85`（唯一） |
| **守卫 / 探测** | `check_torch_deterministic:184-190`、`check_nvlink_connections:146-181` | 前置断言 | `ep.py:639/813`；engram 构造 |
| **带宽 / 域探测** | `get_sm_*_gbs:193-204`、`get_nvlink_gbs:207-234`、`get_rdma_gbs:260-283`、`get_{physical,logical}_domain_size:136-143` | SM/QP 自动选型 | 第 73 篇已覆盖 |

本篇聚焦第一、二、三类里**尚未被单独讲过**的部分：`init_dist`/`init_seed`/`dist_print`/`EnvOverride`
（守卫 `check_torch_deterministic` 在第 83 篇已作为「dispatch 首行守卫」出过场，本篇只做实测复核）。
第四类带宽探测由第 73/86 篇覆盖，不重复。

---

## 3. `init_dist`（`:89-133`）：测试入口

### 3.1 三件副作用：SL、dtype、device

```python
def init_dist(local_rank: int, num_local_ranks: int, seed: int = 0):
    os.environ.setdefault('NCCL_IB_SL', '1')          # :105
    os.environ.setdefault('EP_OVERRIDE_RDMA_SL', '1') # :106
    ...
    torch.set_default_dtype(torch.bfloat16)           # :128
    torch.set_default_device('cuda')                  # :129
    torch.cuda.set_device(local_rank)                 # :130
```

**为什么是 `setdefault` 而不是直接赋值**（源码注释 `:103-104` 明说）：`init_dist` **只给测试用**，
默认把 InfiniBand 的服务等级（SL）压到 **1**，避免测试流量影响生产（生产用 **SL 3**）。用
`setdefault` 意味着「用户/集群预先设了 SL，就一定尊重用户的值」——实测确认（见 §8 Part B：
进程里先设 `NCCL_IB_SL=3`，再走一遍 setdefault，值仍是 `3`）。

**为什么顺手把全局 dtype 设成 bf16、device 设成 cuda**：DeepEP 的所有测试张量几乎都是 bf16、都在
GPU 上。在入口一次性设好，测试体里就少写一堆 `.to(torch.bfloat16).cuda()`。代价是**全局副作用**——
之后任何 `torch.empty(...)` 都默认在 cuda 上、`torch.tensor([1.])` 默认 bf16。这也是测试必须
`init_dist` 后才能跑的原因之一。

### 3.2 起组的两种寻址与 `device_id` 兼容

```python
    ip = os.getenv('MASTER_ADDR', '127.0.0.1')     # :108
    port = int(os.getenv('MASTER_PORT', '8361'))   # :109
    num_nodes = int(os.getenv('WORLD_SIZE', 1))    # :110
    node_rank = int(os.getenv('RANK', 0))          # :111

    sig = inspect.signature(dist.init_process_group)          # :117
    params = {'backend': 'nccl',
              'init_method': f'tcp://{ip}:{port}',
              'world_size': num_nodes * num_local_ranks,      # :121
              'rank': node_rank * num_local_ranks + local_rank}  # :122
    if 'device_id' in sig.parameters:                         # :124
        params['device_id'] = torch.device(f'cuda:{local_rank}')  # :126
    dist.init_process_group(**params)                         # :127
```

要点：

- **全局 rank 的排布**：`rank = node_rank * num_local_ranks + local_rank`，`world_size = num_nodes *
  num_local_ranks`。即「同一节点内先排满本地 rank，再排下一个节点」。
- **`tcp://` rendezvous**：用 `MASTER_ADDR:MASTER_PORT` 的 TCP store 握手（默认 `127.0.0.1:8361`）。
  与 DeepGEMM 那份 `init_method='env://'` 不同（第 40 篇），`env://` 依赖 `RANK/WORLD_SIZE/MASTER_*`
  四个环境变量且对参数拼写更敏感；这里把地址显式拼出来，更可控。
- **`device_id` 兼容分支 `:117-126`**：PyTorch 在某版本把 `init_process_group` 的 `device` 参数改名
  `device_id`（本机 torch 2.12 的签名里已经没有 `device`、只有 `device_id`）。DeepEP 用
  `inspect.signature` 运行时探测——**旧版没有该参数就不传、新版才传**。这是「跨 PyTorch 版本
  不写死 API」的一种惯用做法。实测本机签名含 `device_id`（§8 Part B）。

### 3.3 返回值：一个**新建的**子组

```python
    init_seed(seed)                                                   # :132
    return dist.get_rank(), dist.get_world_size(), \
           dist.new_group(list(range(num_local_ranks * num_nodes)))   # :133
```

第三项不是默认的 `WORLD` 组，而是**现 `dist.new_group` 出来的一个新组**，覆盖全部 rank。因此
`init_dist` 之后进程里同时存在两个通信组：默认的 `WORLD` 与这个新组。DeepEP 的 buffer 构造、
`all_gather_object`、`barrier` 都应该用返回的这个组——它才是「本次测试的通信域」。注意
`dist.new_group` 是**集合操作**，所有 rank 必须都调到，否则挂起。

---

## 4. `init_seed`（`:40-51`）：local 种子 = global + rank

```python
def init_seed(global_seed: int) -> None:      # :40
    global _local_seed, _global_seed
    _local_seed = global_seed + dist.get_rank()   # :48
    _global_seed = global_seed                    # :49
    torch.manual_seed(_local_seed)                # :50
    random.seed(_local_seed)                      # :51
```

**为什么不直接用同一个 seed 喂所有 rank**：如果 8 个 rank 用同一个种子，那么它们生成的「随机测试
数据」（topk 索引、token 内容、专家负载）会**完全一样**——这样的输入没有任何通信意义（每个 rank
收到的都相同），测试覆盖不到真实的异构流量。让 `local_seed = global_seed + rank`，就得到「**全局
可复现、rank 间互异**」的序列：换 `global_seed` 整体变、同 `global_seed` 下每 rank 又是不同的流。

两个消费面：

- `get_global_seed():64-71` 被 **`refs.generate_pre_combine_data`** 消费（`refs.py:148-152`）：
  它在合成「按 `(src_token_global_idx, top_k)` 定种子的 `sin` 数据」时，最后加一项
  `math.sin(float(get_global_seed()))` 做全局偏置。也就是说 **combine 的参考实现是可复现的**，
  而这条链的根就是 `init_dist` 传进来的 `seed`。
- `get_local_seed():54-61`——**全库没有任何消费者**。实验对两个仓库的 `.py` 做了普查，
  `get_local_seed` 出现次数为 0（仅定义处自己）。这是一个「写了但没用上」的死 API；真正被消费的
  只有 `get_global_seed`。

`init_dist` 在返回前已经调了 `init_seed(seed)`（`:132`），所以入口之后 `torch` 的 RNG 就是
`manual_seed(seed + rank)` 的状态。`tests/ep/test_ep.py:561-562` 在每个测试循环里再显式
`init_seed(seed)` 一次，保证每轮从同一状态开始。实测确认 `init_dist(0,1,seed=42)` 之后
`torch.randn(...)` 与手工 `torch.manual_seed(42)` 逐位相等（§8 Part B）。

---

## 5. `dist_print`（`:74-86`）：once-in-node + 屏障

```python
def dist_print(s: str = '', once_in_node: bool = False) -> None:   # :74
    global _local_rank
    assert _local_rank is not None                                 # :83
    if not once_in_node or _local_rank == 0:                       # :84
        print(s, flush=True)                                       # :85
    dist.barrier()                                                 # :86
```

- `_local_rank` 是模块级全局，**只由 `init_dist` 在 `:115` 写入**。所以 `init_dist` 之前调
  `dist_print` 会撞 `:83` 的裸断言（无消息）——这是个刻意的顺序约束。
- **每个 rank 都要调 `dist_print`**：即使 `once_in_node=True` 把它静音了，`:86` 的 `dist.barrier()`
  仍然执行。因此它既是打印工具、也是**同步点**：任一 rank 漏调就会挂住其余 rank。这与「只在
  rank0 打印、其余 rank 直接 return」的写法有本质区别。
- `flush=True`：避免 8 个进程的 stdout 缓冲交错。

实测（§8 Part D）：`local_rank==0` 时打印且屏障 1 次；`local_rank!=0` 时**不打印但仍屏障 1 次**；
`once_in_node=False` 时每个 rank 都打印。

---

## 6. `EnvOverride`（`:22-37`）：可回滚的环境变量覆盖

```python
class EnvOverride:
    def __init__(self, envs: List[Tuple[str, str, bool]]):        # :24
        self.envs = {name: value for name, value, enable in envs if enable}  # :25
    def __enter__(self):                                          # :27
        self.original_values = {name: os.environ.get(name) for name in self.envs}
        os.environ.update(self.envs)                              # :29
        return self
    def __exit__(self, *_):                                       # :32
        for name, value in self.original_values.items():
            if value is None:
                os.environ.pop(name)                              # :35
            else:
                os.environ[name] = value                          # :37
```

三处设计：

1. **三元的 `enable` 标志在 `__init__` 里就被过滤掉**（`:25` 的 `if enable`）：`enable=False` 的项
   **既不设置、也不记录、退出时更不触碰**。这让调用方可以「按条件决定要不要覆盖」而不用写
   `if/else`——`engram.py:81-85` 正是这么用的：
   ```python
   with EnvOverride([('NCCL_GIN_TYPE', '3', True),
                     ('NCCL_GIN_GDAKI_MAX_QP_RD_ATOMIC', '8', restrict_rd_atomic)]):
       ...
   ```
   `NCCL_GIN_TYPE=3`（强制 GDAKI 后端）恒开；`...MAX_QP_RD_ATOMIC=8` 只在 `restrict_rd_atomic`
   为真时才设。
2. **恢复逻辑区分「原本不存在」与「原本有值」**：`None` → `pop`，否则写回。所以退出后环境**精确
   回到进入前的样子**，不会留下一个空串。
3. **`__exit__(self, *_)` 吞掉异常签名**：无论正常退出还是 `with` 体内抛异常，都走同一段恢复
   逻辑。实测确认（§8 Part C：`with` 体内 `raise`，退出后变量仍被清理）。

这是 `envs.py` 里**唯一一个通用基础设施**，可惜整个仓库只有 engram 构造用了它——bucket / EP buffer
里那些「临时改 `NCCL_*` 再构造」的地方都可以用它统一。

---

## 7. 邻居速览（已被别的篇目覆盖，不重复）

- **`check_torch_deterministic:184-190`**：断言「确定性算法」与 `fill_uninitialized_memory`
  **不同时开**。原因见第 62/83 篇——两者同开时 `torch.empty()` 会启动一个初始化 kernel，可能与
  通信流重叠而报错。它在 `ep.py:639/813` 的 dispatch/combine 首行被调。本篇实测四种组合
  （§8 Part E：只有「都开」才抛）。
- **`semantic.py`**（第 83 篇）：`value_or:5-6`（「仅 `None` 才用默认」，`0`/`False` 是合法值）、
  `weak_lru:9-26`（对 `self` 弱引用，避免 `lru_cache` 把 buffer 永久钉住）。`ep.py:18/430/845`
  消费。
- **`find_pkgs.py`**（第 86 篇）：`find_pkg_root`/`find_nccl_root`/`get_nccl_lib_name`，被
  `__init__.py` 的 `check_nccl_so` 与 `setup.py` 用于定位 NCCL。
- **带宽 / 域探测**（第 73/86 篇）：`get_sm_*_gbs`/`get_nvlink_gbs`/`get_rdma_gbs`/
  `get_{physical,logical}_domain_size`，是 SM/QP 自动选型的输入。
- **`check_nvlink_connections:146-181`**：PCIe GPU 只能两两 NVLink，做 P2P 状态检查；第 86 篇
  一带而过，构造 engram buffer 时调用（`engram.py:67`）。

---

## 8. H100 实测（`harness/exp_deepep_process_bootstrap.{sh,out.txt}`）

实验**直接加载真实 `deep_ep/utils/envs.py` 源码**（`sys.modules` 注入合成 `deep_ep._C`/`deep_ep.comm`
绕过 `import deep_ep` 的 `check_nccl_so` 闸门），`envs.py` 与 `semantic.py` 的 sha256 见输出。
**30 PASS / 0 FAIL**：

- **Part A `init_seed`（5）**：`init_seed(100)` 时 rank 3 → `local_seed=103`、`global_seed=100`；
  torch 流逐位等于 `manual_seed(103)`、重复可复现；rank 7 → `local_seed=107`。
- **Part B `init_dist(0,1)`（10）**：真跑 NCCL world_size=1 起组——返回 `(0, 1, group.size()==1)`；
  `NCCL_IB_SL`/`EP_OVERRIDE_RDMA_SL` 被 `setdefault` 成 `1`；`torch.get_default_dtype()==bfloat16`；
  默认 device 变 cuda；`current_device()==0`；`get_local_seed()==42`；种子流逐位可复现。另确认本机
  `init_process_group` 签名**含 `device_id`**（`:124-126` 的兼容分支被走中）。
- **Part C `EnvOverride`（9）**：`enable=True` 生效 / `enable=False` 不生效；退出后未设的原键被
  `pop`、原有值被恢复；异常退出仍恢复；engram 那组「恒开 + 条件开」实测符合预期、退出后清理干净。
- **Part D `dist_print`（4）**：`local_rank==0` 时打印且屏障 1 次；`local_rank!=0` 时**不打印但仍
  屏障 1 次**；`once_in_node=False` 每个 rank 都打印；`_local_rank is None` 时撞 `:83` 断言。
- **Part E `check_torch_deterministic`（4）**：四种组合只有「确定性 ∧ fill」才抛，其余三种通过。
- **Part F `get_local_seed` 死 API**：全库消费者普查结果为**空**。

---

## 9. 小结

- DeepEP 所有测试的第一行 `init_dist` 做了三件 host 侧的事：**起组 + 全局副作用（SL/dtype/device）
  + 播种**；它返回的不是 `WORLD` 而是一个**新建的子组**。
- `init_dist` 用 `setdefault` 把 SL 压到 1（测试不干扰生产的 SL 3），并用 `inspect.signature` 兼容
  PyTorch 的 `device`→`device_id` 改名。
- `init_seed` 的 `local = global + rank` 是「全局可复现、rank 间互异」的根；`get_global_seed` 被
  combine 参考实现消费，而 `get_local_seed` 是**死 API**。
- `dist_print` 即使静音也**必屏障**（全 rank 都要调）；`EnvOverride` 用三元 `enable` 标志支持
  「条件覆盖」，退出精确回滚。二者是测试整齐可复现的基础设施。

这套东西是**架构无关的 host 层**：在 H100(sm90) 上 30 / 30 全过，与目标卡是否 SM100 无关。
