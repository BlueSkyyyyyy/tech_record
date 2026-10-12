# DeepGEMM 的测试基座：`calc_diff` 判据、`bench_kineto` 计时与 `uneven_all_gather`

> 前面 0~29 篇里每一个 kernel 的「对拍」「relerr」「TFLOPS GB/s」数字，都出自同一套**测试基座**
> `deep_gemm/testing/`（+ `deep_gemm/utils/dist.py`），却从没被系统拆过。
> 这一篇补上它——因为「DeepGEMM 怎么判定一个 kernel 是对的、怎么量它的速度」本身，就是理解
> 这个仓库其余所有结论的前提。
>
> 三个函数是全部测试的公用件：
> - **`calc_diff`**：DeepGEMM 全仓库的**正确性判据**。实测它恒等于**归一化平方误差**
>   `‖x−y‖²/(‖x‖²+‖y‖²)`，因此对称、∈[0,2]、=0 当且仅当逐元素相等；小误差下 `≈ rel_l2²/2`。
> - **`bench_kineto`**：用 PyTorch profiler 取**纯 kernel 时长**（另有简单的 event 版 `bench`），
>   并可一次分别计时多个 kernel（如 `transpose_fp32` + `sm90_fp8_gemm`）。
> - **`uneven_all_gather`**：MoE 测试用它把各 rank **不等长**的 topk 索引拼成全局表。
>
> 这三个都不是 SM100 专属、也不是 kernel 本身，**H100 真机可跑**。本文用真实数字讲清它们的语义与坑。

参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1），本机 `kernel_lab`（H100 sm90）。

---

## 1. 测试基座在仓库里的位置

`deep_gemm/testing/` 只有四个文件、共约 232 行，却在每个 `tests/test_*.py` 顶部被 `import *`：

```python
# deep_gemm/testing/__init__.py:1-4
from . import bench, numeric, utils
from .bench import *
from .numeric import *
from .utils import *
```

- `bench.py:7-33` `bench`（event 计时）、`:79-146` `bench_kineto`（profiler 计时），
  以及两个 stdout 抑制上下文 `empty_suppress:36-41` / `suppress_stdout_stderr:44-76`。
- `numeric.py:5-11` `calc_diff`、`:14-21` `count_bytes`、`:24-44` `assert_bitwise_equal`。
- `utils.py:6-8` `get_arch_major`、`:11-22` `test_filter`、`:25-37` `ignore_env`。
- `utils/dist.py:10-35` `init_dist`、`:38-66` `uneven_all_gather`、`:69-74` `dist_print`；
  由 `deep_gemm/utils/__init__.py:4` 重新导出 `init_dist, uneven_all_gather`。

典型调用形态（真实行号）：

```python
# tests/test_bf16.py:45 / :56
diff = calc_diff(d, ref_d)
t = bench_kineto(lambda: deep_gemm.bf16_gemm_nt(a, b, d, c=c), 'bf16_gemm', suppress_kineto_output=True)

# tests/test_mega_moe.py:11 / :76 / :354
from deep_gemm.utils.dist import dist_print, init_dist, uneven_all_gather
rank_idx, num_ranks, group = init_dist(local_rank, num_local_ranks)
gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
```

于是「DeepGEMM 判定对错用得是 `calc_diff < 阈值`」「吞吐是 `count_bytes(...) / t`」这两件事，
锚点全在这里。

---

## 2. `bench` 与 `bench_kineto`

### 2.1 `bench`：event 计时（整段，含发射）

```python
# bench.py:7-33
def bench(fn, num_warmups: int = 5, num_tests: int = 10, high_precision: bool = False):
    torch.cuda.synchronize()
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda')   # :11 256MB L2 flush
    cache.zero_()                                                          # :12
    for _ in range(num_warmups): fn()                                     # :15-16
    if high_precision:                                                     # :19
        x = torch.randn((8192, 8192), ...); y = torch.randn((8192, 8192), ...)
        x @ y                                                             # :22 一个大核
    start_event.record()                                                  # :27
    for i in range(num_tests): fn()                                       # :28-29
    end_event.record(); torch.cuda.synchronize()                          # :30-31
    return start_event.elapsed_time(end_event) / num_tests / 1e3          # :33
```

要点：

- 先 flush 一次 **256 MB** L2（`256e6//4` 个 int）。
- 计时区间是 `num_tests` 次 `fn()` 的**连续 event**，除以次数——**含每次的 CPU 发射开销**：
  只要单次 CPU 发射时间 > GPU kernel 时长，GPU 就会空转，event 时间被系统性抬高。
- `high_precision=True` 在计时前**先排队一个 `8192×8192` fp32 大 GEMM**（约十几 ms），
  其意图（源码注释 `:18`）是「用一个足够长的核把 CPU 的发射藏进 GPU 流水」——在 GPU 已被占满时，
  CPU 能趁机把后面 `num_tests` 次调用全排好队。本机实测这个改善**不稳定**（见 §7）。

### 2.2 `bench_kineto`：profiler 取纯 kernel 时长

```python
# bench.py:79-146（节选）
assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)    # :84
if int(os.environ.get('DG_USE_NVIDIA_TOOLS', 0)):
    return (1, ) * len(kernel_names) if is_tuple else 1                    # :89-90  跳过 profiling
...
schedule = torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)   # :101
profiler = torch.profiler.profile(activities=[...CUDA], schedule=schedule,
                                  acc_events=True)                         # :102-103
with profiler:
    for i in range(2):
        for _ in range(num_tests):
            if flush_l2:
                torch.empty(flush_l2_size, ...).zero_()                    # :108 每次清 L2
            if barrier is not None:
                torch.cuda._sleep(int(2e7))   # ~10ms 大核                  # :112
                barrier()                                                   # :113
            fn()
        torch.cuda.synchronize(); profiler.step()                           # :115-116
prof_lines = profiler.key_averages().table(sort_by='cuda_time_total', ...)  # :119
...
units = {'ms': 1e3, 'us': 1e6}                                              # :130
for name in kernel_names:                                                   # :132
    for line in prof_lines:
        if name in line:
            time_str = line.split()[-2]; num_str = line.split()[-1]         # :137-138
            for unit, scale in units.items():
                if unit in time_str:
                    total_time += float(time_str.replace(unit, '')) / scale * int(num_str)  # :141
```

精读要点（都是「为什么」）：

1. **`DG_USE_NVIDIA_TOOLS` 旁路**（`:88-90`）：设了该环境变量就直接返回 `1`（单名）或 `(1,)*n`（tuple），
   **不做 profiling**——因为 torch profiler 会和 nsys / ncu / compute-sanitizer 冲突。
   这是为什么本专题做 ncu 归因（第 15 篇）时从不混用 `bench_kineto`。
2. **schedule `wait=0,warmup=1,active=1,repeat=1`**（`:101`）：跑两批（`for i in range(2)`），
   **第一批是 warmup（不计），第二批 active 才计入**。所以实际被测的是第二批 `num_tests` 次。
3. **`flush_l2`（默认 True，`:107-108`）**：每次迭代前清 256 MB L2，保证测的是**冷 cache**；
   对读操作数多的大 GEMM 影响显著（§7 实测 4096³ 冷启 ~300 us vs 命中 ~225 us）。
4. **`barrier`（可选，`:109-113`）**：先跑一个 ~10 ms 的空转大核（`torch.cuda._sleep(2e7)`）
   把 GPU 占住，再调用户给的 `barrier()`——源码注释 `:110` 说这是为了「消除不均衡的 CPU 发射开销」，
   多卡场景下 `barrier` 传的是 dist barrier，让各 rank 对齐。
5. **按名字子串匹配、按单位换算**（`:132-143`）：返回的是**秒**（`ms` 除以 1e3、`us` 除以 1e6），
   所以调用方常用 `count_bytes(...) / 1e9 / t` 直接得 GB/s（如 `tests/test_bf16.py:62`）。
6. **`kernel_names` 可以是 tuple**（`:84`）：此时分别计时并返回元组，
   且默认 `with_multiple_kernels=False` 会断言每个名字在表里**至多出现一行**（`:121-123`），
   防止一个子串意外匹配到多个 kernel。这就是「一次 `fp8_gemm_nt` 分别抓到 `transpose_fp32` 和
   `sm90_fp8_gemm_1d2d`」的用法。

---

## 3. `calc_diff`：DeepGEMM 的归一化平方误差判据

```python
# numeric.py:5-11
def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()                       # :6  升 fp64
    denominator = (x * x + y * y).sum()                 # :7
    if denominator == 0:                                # :8  两者全零
        return 0.0
    sim = 2 * (x * y).sum() / denominator               # :10
    return 1 - sim                                      # :11
```

### 3.1 它到底是不是一个「误差」

把 `:10-11` 通分：

```
calc_diff = 1 − 2⟨x,y⟩ / (‖x‖²+‖y‖²) = (‖x‖² − 2⟨x,y⟩ + ‖y‖²) / (‖x‖²+‖y‖²) = ‖x−y‖² / (‖x‖²+‖y‖²)
```

也就是**归一化平方 L2 误差**（NSRE）。由此立刻得到一串性质，本机逐条核过：

| 性质 | 值（H100 实测） |
|---|---|
| 恒等于 `‖x−y‖²/(‖x‖²+‖y‖²)` | 5.00210553e-05 vs 5.00210553e-05（差 1.2e-16） |
| 对称 `calc_diff(x,y)=calc_diff(y,x)` | 差 0 |
| `=0` ⟺ 逐元素相等 | `calc_diff(x,x)=0` |
| 值域 `[0,2]` | `calc_diff(x,−x)=2.000000`、`calc_diff(x,0)=1.000000` |
| **两操作数同步缩放不变** | `calc_diff(3.7x,3.7y)=calc_diff(x,y)` |
| **单侧缩放不变**（仅 c=1） | `calc_diff(x,3.7y)=4.96e-01 ≠ 5.00e-05` |
| 小误差 `≈ rel_l2²/2` | e=1e-2：3.186e-05 vs 3.186e-05（比值 1.0000）；e=5e-2：0.9992 |
| DC 平移敏感 | `calc_diff(x,x+0.1)=4.98e-03` |

两个值得记的「坑」：

- **它不是缩放不变的**。同步缩放 `(cx, cy)` 不变；但**只缩放一侧**不成立，因为分子 `‖x−cy‖²` 与
  分母 `‖x‖²+c²‖y‖²` 只在 `c=1` 时相等。这和 `cosine_similarity`（对单侧正缩放不变）**不同**——
  `1−calc_diff = cosθ · 2‖x‖‖y‖/(‖x‖²+‖y‖²)`，那个额外的范数因子 `2ab/(a²+b²) ≤ 1` 恰好只在
  `‖x‖=‖y‖` 时为 1。所以 `calc_diff` 同时惩罚「方向不对」和「幅度不匹配」。
- **它不中心化**。`calc_diff(x, x+常数)` 会随常数增大——若两边的 DC 分量不同就会被计入。
  所以报告 `calc_diff` 时要和参考的构造方式一起看（DeepGEMM 的参考通常也由同一套量化器产出，
  DC 一致）。

实践上它有个很顺的手感：**小误差下 `calc_diff` 就是「相对 L2 误差的平方的一半」**——
`relerr 1e-2` 对应 `calc_diff ~5e-5`，`relerr 1e-1` 对应 `~5e-3`。仓库里 `calc_diff < 1e-2`
这类阈值（如 `tests/test_bf16.py` 用 `diff < 0.001`）等价于「相对 L2 误差 ≲ 4.5%」。

### 3.2 为什么用 fp64

`x, y = x.double()`（`:6`）：把 bf16/fp8 反量化后的小输入升到 fp64 再算内积，
**让判据本身不引入额外误差**，测到的就是 kernel 输出与参考的真差。这一点很关键——
否则 bf16 上的 `x*x` 会先自己溢出/舍入。

---

## 4. `count_bytes` 与 `assert_bitwise_equal`

### 4.1 `count_bytes`

```python
# numeric.py:14-21
def count_bytes(*tensors):
    total = 0
    for t in tensors:
        if isinstance(t, (tuple, list)):
            total += count_bytes(*t)          # 递归展开 (fp8, sf) 这样的操作数对
        elif t is not None:
            total += t.numel() * t.element_size()
    return total
```

因为 FP8 操作数是 `(data, scale)` **元组**（第 25/27 篇），吞吐统计要连 SF 一起算，
所以它递归处理 tuple/list、并跳过 `None`（可选参数如 `c`、`sf`）。

### 4.2 `assert_bitwise_equal`

```python
# numeric.py:24-44（节选）
def assert_bitwise_equal(x, y, label=''):
    assert x.shape == y.shape                       # :25  先断言形状
    assert x.dtype == y.dtype                       # :26  再断言 dtype
    x_bytes = x.contiguous().view(torch.uint8)      # :27  按字节比
    y_bytes = y.contiguous().view(torch.uint8)      # :28
    if torch.equal(x_bytes, y_bytes): return         # :29-30
    mismatch = x_bytes != y_bytes                    # :32
    mismatch_idx = mismatch.flatten().nonzero()[0].item()   # :33 首个不一致字节
    elem_idx = mismatch_idx // x.element_size()      # :34
    byte_in_elem = mismatch_idx % x.element_size()   # :35
    coord = tuple(torch.unravel_index(torch.tensor(elem_idx, ...), x.shape))  # :36-37
    raise AssertionError(f'bitwise mismatch ... first_byte={...} coord={...} '
                         f'byte_in_elem={...} x_byte={...} y_byte={...} ...')  # :38-44
```

它比 `torch.equal` 强的地方在**失败时的定位信息**：哪个字节、落在哪个元素、元素坐标、
元素内第几个字节、双方该字节的原值。第 12/14/25/26/35 篇里「逐位相等」的结论都是它下的断言。
实测改一个 bf16 元素（`(3,5)`）立即抛出：

```
bitwise mismatch (one_elem): num_bytes=256, num_mismatch=2, first_byte=106,
elem=53, coord=(3, 5), byte_in_elem=0, x_byte=132, y_byte=0, x_val=-1.03125, y_val=-0.03125
```

`num_mismatch=2` 是因为一个 bf16 元素占 2 字节；`byte_in_elem=0` 给出首个差异字节在元素内的偏移。

---

## 5. `testing/utils.py`：架构判定与两个装饰器

```python
# utils.py:6-8
def get_arch_major() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major

# utils.py:11-22
def test_filter(condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                func(*args, **kwargs)
            else:
                print(f'{func.__name__}:'); print(f' > Filtered by {condition}'); print()
        return wrapper
    return decorator

# utils.py:25-37
def ignore_env(name: str, condition: Callable):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if condition():
                saved = os.environ.pop(name, None)
                func(*args, **kwargs)
                if saved is not None:
                    os.environ[name] = saved
            else:
                func(*args, **kwargs)
        return wrapper
    return decorator
```

- `get_arch_major`：H100 返回 **9**；SM100 返回 10。整个仓库的「SM90/SM100 分支」都靠它
  （配合 `test_filter(lambda: get_arch_major() == 10)`）。
- `test_filter`：条件为假时**打印一行 `Filtered by ...` 并跳过函数体**——这是为什么跑测试时会看到
  「SM100 专属测试在 H100 上被跳过」的输出。
- `ignore_env`：条件为真时**临时 `pop` 掉环境变量**，函数返回后恢复；用于测「某个环境变量开关
  打开/关闭」的行为而不污染后续测试。

**实测发现的一个小陷阱**：两个装饰器的 wrapper 里都**没有 `return func(...)`**，是裸调用
（`:16`/`:19`、`:30`/`:33`）。所以它们**只适合有副作用的函数**——经 `ignore_env`/`test_filter`
装饰后，函数的**返回值恒为 `None`**。本机实测：一个 `return 12345` 的函数被 `ignore_env` 装饰后返回 `None`。
这不是 bug（仓库里它们只用于副作用 + 断言），但自己套用时要注意。

实测还确认：`ignore_env(cond=True)` 函数内读到 `None`（被移除）、`cond=False` 读到 `'sentinel'`（保留）、
返回后环境变量恢复为 `'sentinel'`。

---

## 6. `uneven_all_gather`：MoE 的「不等长拼接」原语

`tests/test_mega_moe.py:354` 用它把每个 rank 的 **`topk_idx`**（各 rank token 数不同）拼成全局表，
以统计专家负载：

```python
gathered_topk_idx = uneven_all_gather(topk_idx, group=group)
```

实现（`utils/dist.py:38-66`）：

```python
def uneven_all_gather(tensor, dim=0, group=None):
    world_size = dist.get_world_size(group)                                   # :39
    local_dim_size = torch.tensor([tensor.shape[dim]], ...)                   # :42
    all_dim_sizes = [torch.zeros_like(local_dim_size) for _ in range(world_size)]
    dist.all_gather(all_dim_sizes, local_dim_size, group=group)               # :44  先交换长度
    all_dim_sizes = [s.item() for s in all_dim_sizes]                         # :45
    max_dim_size = max(all_dim_sizes)                                         # :46
    if tensor.shape[dim] < max_dim_size:                                      # :49  不足者补零
        padding = torch.zeros(pad_shape, ...); tensor_padded = torch.cat([tensor, padding], dim=dim)
    else:
        tensor_padded = tensor.contiguous()
    gathered = [torch.zeros_like(tensor_padded) for _ in range(world_size)]
    dist.all_gather(gathered, tensor_padded, group=group)                     # :59  等长 all_gather
    trimmed = [torch.narrow(gathered[i], dim, 0, all_dim_sizes[i]) for i in range(world_size)]  # :62-65
    return torch.cat(trimmed, dim=dim)                                        # :66  去掉 padding 后拼
```

四步：**交换长度 → pad 到 `max` → 等长 `all_gather` → 逐段 `narrow` 再 `cat`**。
它是「把不等长张量凑成一次 NCCL all_gather」的通用桥；`init_dist:10-35` 则用 `tcp://ip:port`
初始化 NCCL，并兼容老/新 torch 的 `device_id` 签名（`:28-30`）。

**与 Mega MoE 的关系**：真正的 Mega MoE 通信走的是 DeepEP 的对称显存 dispatch/combine（第 03/08 篇），
不是这个 all-gather；`uneven_all_gather` 只在**测试的参考路径**里用——把参考实现需要的全局索引/
统计拼齐，好和 kernel 输出对比。它的语义（不等长 → pad → 拼）与 combine 里「按 rank 段拼接」
是同一类操作。

---

## 7. H100 实测

脚本 `harness/exp_testing_harness.py`（driver `harness/exp_testing_harness.sh`，输出
`harness/exp_testing_harness.out.txt`）在 `kernel_lab`（H100 sm90）上跑，六组 **ALL PASS**。

### 7.1 `calc_diff` 性质（§3 的表）

全部逐条核过：恒等式差 1.2e-16、对称差 0、`(x,x)=0`/`(x,−x)=2`/`(x,0)=1`、
同步缩放不变、单侧缩放 `4.96e-01`、小误差比值 1.0000、DC 平移 `4.98e-03`。

### 7.2 `bench` vs `bench_kineto`（真实 `fp8_gemm_nt`）

内含 `transpose_fp32` + `sm90_fp8_gemm_1d2d` 两个 kernel：

```
[小 256x256x512]
  bench(event, 含发射)          =    29.254 us
  bench(event, high_precision)  =     5.311 us        ← 大核藏发射：本次生效
  bench_kineto(纯 kernel,flush) =     2.717 us   ->    24.7 TFLOPS
  bench_kineto(纯 kernel,noflush)=     2.351 us
  bench_kineto(tuple) transpose =     1.684 us | fp8_gemm =     2.715 us
  bench / bench_kineto 比值      =  10.77x

[大 4096x4096x4096]
  bench(event, 含发射)          =   226.094 us
  bench(event, high_precision)  =   228.821 us        ← CPU 单次开销 < kernel，藏发射无意义
  bench_kineto(纯 kernel,flush) =   300.691 us   ->   457.1 TFLOPS
  bench_kineto(纯 kernel,noflush)=   224.848 us       ← 冷/热 L2 差 ~34%
  bench_kineto(tuple) transpose =     2.480 us | fp8_gemm =   301.141 us
  bench / bench_kineto 比值      =    0.75x
```

解读：

- **小 kernel**：event 计时含每次 CPU 发射（单次 ~5 us > kernel 2.7 us），故 `bench` 明显偏大；
  `high_precision` 用一个大核占住 GPU、让 CPU 趁机排队，**这次把 29.3 us 压到 5.3 us**（生效）。
  但本机 GPU 被多个自驱任务共享，数字有抖动（另一次运行里 `high_precision` 几乎无改善）。
- **大 kernel**：event 与 kineto 收敛；`bench` 甚至略快于 kineto，因为 kineto 默认 `flush_l2=True`
  每次冷启，`noflush` 的 224.8 us 才是热 cache 值。**冷/热 L2 对 4096³ 有 ~34% 差异**。
- **tuple 计时**：一次调用分别给出 `transpose`（1.7/2.5 us）与 `fp8_gemm`（2.7/301 us）——
  这就是全仓库把「SF 转置开销」从「GEMM 开销」里分出来的手段（第 25/32/37 篇反复引用）。
- **`DG_USE_NVIDIA_TOOLS=1`**：`bench_kineto` 直接返回 `1` / `(1,1)` 跳过 profiling，实测复现。

### 7.3 `uneven_all_gather`（8×H100）

8 个 rank 的 `n_tok = [17, 512, 4096, 3, 2048, 100, 777, 64]`（总 7617）：

```
8 rank 拼接正确（逐段 rank 编号 + 总长 7617，每 rank 结果一致）：True
dim=1 不等长拼接正确：True
H=7168、每 rank 8192 tok（117.4 MB）：11335.8 us，每 rank 发出 72.5 GB/s，
拼接结果吞吐 82.9 GB/s（未开 NVLS）
```

- **正确性**：`dim=0` 下每个 rank 收到按轮次顺序拼接的 7617 行、每段值等于该行 rank 编号；
  `dim=1` 同样正确。`all_correct` 用 `all_reduce(MIN)` 确认**所有 rank 结果一致**。
- **开销**：H=7168、每 rank 8192 tok（117.4 MB）时一次 ~11.3 ms，量级与第 01 篇的
  NPC all-to-all（~390 GB/s）相比偏低，原因是本机 NVSwitch 无 Fabric Manager，测试关掉了
  **NVLink SHARP 多播**（`NCCL_NVLS_ENABLE=0`，否则 `Failed to bind NVLink SHARP` 直接报错）。
  这说明 `uneven_all_gather` 的额外 pad（这里恰好等长不 pad）+ 两次 all_gather 的固定开销在小消息下占比可观。

---

## 8. 小结

- `deep_gemm/testing/` 是 DeepGEMM 全仓库的**正确性判据 + 计时基座**，
  每个 `tests/test_*.py` 都从它 import。三件套：
  - **`calc_diff`** = 归一化平方误差 `‖x−y‖²/(‖x‖²+‖y‖²)`：对称、∈[0,2]、=0 当且仅当相等、
    同步缩放不变、单侧缩放**不**不变、小误差 `≈ rel_l2²/2`。升 fp64 保证判据本身干净。
  - **`bench`**（event，含发射；`high_precision` 用大核藏发射）与 **`bench_kineto`**
    （profiler 纯 kernel 时长；`wait=0/warmup=1/active=1`；`flush_l2` 清 256MB；
    tuple 可分别计时多 kernel；`DG_USE_NVIDIA_TOOLS` 旁路）。
  - **`count_bytes`**（递归数 `(data, sf)` 对的字节）、**`assert_bitwise_equal`**（逐位 + 定位首个差异字节）。
- `testing/utils.py`：`get_arch_major`（H100=9 / SM100=10）、`test_filter`（假条件跳过并打印）、
  `ignore_env`（临时 pop 环境变量）；**两者 wrapper 不 return**（只用于副作用）。
- `utils/dist.py`：`init_dist`（tcp init）、`uneven_all_gather`（**交换长度→pad→all_gather→trim→cat**）、
  `dist_print`。MoE 测试用前者拼不等长 `topk_idx` 做参考统计。
- H100 六组 **ALL PASS**；`bench_kineto` 的纯 kernel 时长比 event 计时小得多（小 kernel ~10×），
  冷/热 L2 对大 GEMM 差 ~34%，`uneven_all_gather` 8 卡正确（受本机 NVLS 关闭影响，带宽 ~73 GB/s）。

一句话：**读懂这一层，才看得懂前面 29 篇里每一个 `calc_diff`/`TFLOPS`/`GB/s` 数字是从哪来的、量得准不准。**
