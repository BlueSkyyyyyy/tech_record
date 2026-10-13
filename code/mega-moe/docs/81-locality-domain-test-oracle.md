# SM 局部域的验收测试即正确性规范（`tests/test_locality_domain.py`，H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P71 增量（第 81 篇）。
> 第 34 篇（P24-1）拆了局部域的**实现**——pointer-chase 延迟探针、`try_probe` 两闸门、
> MLOPart / CUDA VMM 的本地化分配、以及调度器按域分区——但它所描述的「域到底**算什么对**」的
> 唯一规范 `tests/test_locality_domain.py`（167 行）从没被单独拆过。本篇把它当**可执行规格**逐段拆透，
> 并逐条核对第 34 篇里的结论（尤其「不可用时逐位等于 `get_even`」与「在 H100 上只有部分用例能跑」）。
>
> 与第 51/57/67/74/76/78/79/80 篇那批「验收测试即规范」一脉相承：本篇是同一系列里**唯一一份纯 host / API 层**
> 的规范——它一个 GPU kernel 都不启动，测的是「域表怎么划、本地化内存长什么样、MLOPart 生命周期怎么走」。
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.8.1）。

## 1. 它为什么是「规范」而不是「单元测试」

`tests/test_locality_domain.py` 没有用 `unittest`，就是 5 个顶层函数按顺序调用（`__main__`，`:160-167`）：

| 用例 | 行 | 测什么 | 需要 MLOPart？ |
|---|---|---|---|
| `test_sm_locality_domains` | `:32-49` | SM 域表 + 均衡表：两域都在、TPC 成对、恰好 50/50、只搬最少 TPC | 否 |
| `test_empty` | `:52-64` | 单域 `empty`：形状/连续性/逐位拷贝、域越界必须报错 | 是（分配时） |
| `test_localize` | `:73-105` | `localize` 切片形状/stride 粒度/`is_localized` 语义/往返一致 | 是（`:77` 显式断言） |
| `test_mlopart` | `:108-130` | MLOPart 进程生命周期：起/服务/退、内存比分配器活得久、重启 | 是 |
| `test_without_mlopart` | `:133-157` | 无 MLOPart 时：告警一次、拒绝 `localize`、SM 表仍可用 | 否（反向验证） |

两个 helper 也定义了契约：

- `assert_raises`（`:19-24`）：**只吞 `RuntimeError`**，其它异常原样抛出。于是测试里「越界必须报错」这句话
  被精确编码成「必须是一个 `RuntimeError`」——后面 §6 会看到这恰好能让 `test_empty` 的越界分支**在无 MLOPart 的
  H100 上也能验**。
- `random_bytes`（`:27-29`）：造随机的 fp8/int8/bf16 张量用于逐位比较。
- `as_plain`（`:67-70`）：`(num_domains, …, D/num_domains, …)` 沿 `dim` 折回普通张量——`localize` 的逆操作，
  用来做「往返逐位相等」的判据。

这份文件回答的是一个很具体的问题：**「DeepGEMM 的局部域子系统，其对外行为算什么对？」**
它不碰 SM100，也不碰任何 GEMM kernel；它测的是 `deep_gemm.locality_domain` 这个 API 面的语义。

## 2. 环境能力：为什么这份测试在本机只能跑两个用例

「能不能跑」完全取决于 `is_localization_available()`。它是一条**跨语言**的调用链：

```
deep_gemm.is_localization_available()        # deep_gemm/locality_domain.py:6 顶层 import
  → csrc/apis/locality_domain.hpp:25-27       is_localization_available() { return mlopart::is_available(); }
  → csrc/runtime/mlopart.hpp:24-26            get_module().attr("is_available")(uuid)
  → deep_gemm/utils/mlopart.py:63-64          is_available() { return _get(device_uuid).available }
  → mlopart.py:55-60                          _get() 惰性起一个 MLOPart 单例
  → mlopart.py:28-37                          MLOPart.__init__：socketpair 起子进程
  → mlopart.py:113-130                        main()：起 MPS 控制守护 + `start_server -mlopart`
  → mlopart.py:94-110                         serve()：断言本进程看见 2 个 device
```

关键那一行是 `serve()` 里的断言（`mlopart.py:99`）：

```python
assert num_devices.value == NUM_LOCALITY_DOMAINS, f'{num_devices.value} MLOPart devices'
```

MLOPart 的语义是「把一张物理卡切成 **2 个 device instance**，每个 instance 暴露一半 SM 和一半 HBM」
（第 34 篇 §1.2）。本机 MPS 只暴露 **1 个** device，断言失败 → 子进程退出 → 父进程 `recv` 拿不到 `READY`
→ `available=False`，并在顶层打印一次 `UserWarning: MLOPart is unavailable; localization has been disabled`
（`mlopart.py:37`）。于是 `is_localization_available()` 恒 False。

由此**本地化分配**（`locality_domain.empty` / `empty_per_domain` / `localize`）全线不可用：

```
locality_domain.empty(sizes, dtype, d)
  → csrc/apis/locality_domain.hpp:34-40   reserve + map，包成 torch 张量
  → csrc/runtime/locality_domain.hpp:72-81 map() { create(); cuMemMap; cuMemSetAccess }
  → csrc/runtime/locality_domain.hpp:34-38 create() { DG_HOST_ASSERT(0<=d<2); return mlopart::create_memory(...) }
  → csrc/runtime/mlopart.hpp:28-35         走 Python create_memory 拿 POSIX fd
  → mlopart.py:39-44                       assert self.available, 'MLOPart is unavailable...'
```

**H100 上任一合法域索引都会在这里 `AssertionError`。** 但注意 `create()` 先做 `DG_HOST_ASSERT`
（`locality_domain.hpp:36`）范围检查，**再**调 MLOPart——所以「域索引越界 → `RuntimeError`」这条契约
**与 MLOPart 无关，本机可单独验**（§4.1）。

能力矩阵（H100 真机，本实验 Part B）：

| API | 值 | 说明 |
|---|---|---|
| `get_num_locality_domains()` | `2` | 硬编码常量（`common/types.cuh:54`、`runtime/locality_domain.hpp:51-54`） |
| `get_granularity()` | `2097152` | 2 MiB，`cuMemGetAllocationGranularity(MINIMUM)`（`runtime/locality_domain.hpp:56-63`） |
| `is_localization_available()` | `False` | MPS 只给 1 个 device（`mlopart.py:99`） |
| device SM 数 | `132` | 域表长度 |

## 3. 域表规范：`test_sm_locality_domains`（`:32-49`）

这是本机**真机通过**的第一份规格。它一共 4 条断言：

```
test_locality_domain.py:39: assert sm_domain.dtype == balanced.dtype == torch.uint8 and 0 < sm_domain.sum().item() < num_sms
test_locality_domain.py:42: assert torch.equal(tpcs[:, 0], tpcs[:, 1])          # TPC 内两 SM 同域
test_locality_domain.py:45: assert balanced.sum().item() * 2 == num_sms          # 恰好 50/50
test_locality_domain.py:47: assert num_moved_tpcs == abs(sm_domain.sum().item() - num_sms // 2) // NUM_SMS_PER_TPC
```

即：**① 两个域都非空、dtype 是 uint8；② 一个 TPC 的两个 SM 必须同域；③ 均衡域表恰好一半一半；
④ `balance` 只搬动「最少数量」的 TPC。**

### 3.1 域表是怎么来的

`get_sm_locality_domains()`（`csrc/apis/locality_domain.hpp:120-124`）把结果**缓存在 `Runtime` 单例**里
（`runtime.hpp:36/38` 的 `sm_locality_domains` / `balanced_sm_locality_domains`），首次调用才计算，且有一条
**分水岭**：

```cpp
// csrc/apis/locality_domain.hpp:121-122
if (not runtime->sm_locality_domains.defined())
    runtime->sm_locality_domains = is_localization_available() ? probe_sm_locality_domains() : get_even_sm_locality_domains();
```

本机不可用 → 走 `get_even_sm_locality_domains()`（`:103-106`）：

```cpp
const auto tpc_idx = torch::arange(jit->device.get_num_sms() / kNumSMsPerTPC, ...);
return (tpc_idx % get_num_locality_domains()).to(torch::kUInt8).repeat_interleave(kNumSMsPerTPC);
```

即 TPC 偶数号 → 域 0、奇数号 → 域 1，每个 TPC 再展开成 2 个 SM（`kNumSMsPerTPC=2`）。
132 SM = 66 TPC → **66/66**。这就是本机 `get_sm_locality_domains()` 的输出，**逐位等于** `get_even`
（本实验 Part D 用纯 Python 复刻后逐位相等为 True）。

### 3.2 `balance` 与一个 libtorch 陷阱

`balance_sm_locality_domains`（`csrc/apis/locality_domain.hpp:126-136`）：

```cpp
const auto tpc_domain = tpcs.select(1, 0);                 // 每个 TPC 取第一个 SM 的域号
const int64_t num_tpcs = tpcs.size(0), num_domain_1 = tpc_domain.sum().item<int64_t>();
const uint8_t larger = num_domain_1 * 2 > num_tpcs;        // “多数”域（并列时取 0）
const auto surplus = (tpc_domain == larger).nonzero().flatten().slice(0, num_tpcs / 2);
balanced.view({-1, kNumSMsPerTPC}).index_fill_(0, surplus, larger ^ 1);
```

**第 132 行的 `slice(0, num_tpcs / 2)` 是真正的陷阱**：libtorch 的 `Tensor::slice(dim, start, end)`
里第二参是 **start（起点）而不是长度**。所以 `surplus` 取的是「多数域 TPC 列表」的**后半段**
`[num_tpcs/2:]`，而不是前 `num_tpcs/2` 个。这个语义正是「**已经 50/50 时不动**」的来源：

- 输入 66/66（均衡）：`num_domain_1 = 33` TPC，`larger = 0`，域 0 的 TPC 列表后半段为空 → `surplus` 为空 → **no-op**；
- 输入全 0（占满一域）：域 0 有 66 个 TPC，取后半段（33 个）翻成 1 → **66/66**；
- 输入 TPC 一致偏斜 40/26：域 1 的 40 个 TPC 取后半段（7 个）翻成 0 → **66/66**，搬 7 个 = `|80-66|/2`。

若把 `slice` 误当「取前一半」，均衡输入会被错误地翻成**全部一个域**——本实验 Part D 特意用这个
错误变体做「利齿」负样本：它让 `balanced=[0,132]`，**规格断言③（`sum*2==num_sms`）当场不成立**。
换句话说，测试的断言③+④**正好编码了下面这条实现事实**：`surplus` 必须取多数域的**后半段**，才能
既保证 50/50、又在已均衡时零搬动。

第 34 篇 §2.4 记录过「`balance` 的函数名有欺骗性——只在『一个域占满』时精确 50/50，一般化偏斜输入下
不是最小移动均衡器」。本篇的复刻与真机把这条收敛为：**对 TPC 一致的输入，它就是最小移动均衡器**
（`moved == |sum - num_sms/2|/2`，规格断言④）；第 34 篇观察到的「17/49」正是「40/26 偏斜搬 7 个」这类
非直观但正确的输出形态。

## 4. 本地化内存规范：`test_empty` / `test_localize`

### 4.1 `test_empty`（`:52-64`）——越界契约本机可验

规格本体（`:58-61`）：`locality_domain.empty(shape, dtype, d)` 必须是**连续**张量、`copy_` 后**逐位相等**；
再加一条 `:63` 的契约：

```
test_locality_domain.py:63: assert_raises(lambda: locality_domain.empty((1, ), torch.uint8, num_domains))
```

`num_domains = 2`，所以这是「域索引越界」。`empty` 里的顺序是 `reserve`（`apis:37`）→ `map`（`:38`）→
`create`（`runtime:74`）→ `DG_HOST_ASSERT(0<=d<2)`（`runtime:36`）。**这条 `DG_HOST_ASSERT` 先于
MLOPart 调用**，故在无 MLOPart 的 H100 上，`domain_idx ∈ {-1, 2}` 会抛出真正的 `RuntimeError`
（本实验实测正文：`Assertion error (locality_domain.hpp:36): 0 <= domain_idx and domain_idx < get_num_locality_domains()`），
`assert_raises` 通过。这坐实了 §1 那句话——**判据把「越界」和「不可用」分成了两种异常**：
越界是 C++ 契约（`RuntimeError`），不可用是 Python 层的 `AssertionError`。本机只能验前者（后者见 §5.2）。

### 4.2 `test_localize`（`:73-105`）——本机被 `:77` 挡在门外

这个用例的第一句就是硬门控：

```
test_locality_domain.py:77: assert deep_gemm.is_localization_available()
```

本机 False → 直接失败，故它整段在 H100 跑不了。但它定义的规格值得记下来，因为它是 `localize` 的**形状/布局契约**：

- **形状**（`:86`）：`localize((*B, N, K)) -> (num_domains, *B, N/num_domains, K)`（`dim` 可指定，`:79` 还测
  了 `dim=0`）。对应 `deep_gemm/locality_domain.py:16-24`：`unflatten(dim,(num_domains,-1)).movedim(dim,0)`。
- **每片连续**（`:87`）+ **`stride(0)` 对齐到粒度、且实际片字节在「一个粒度」内**（`:88-89`）：

  ```
  num_slice_bytes = localized.stride(0) * localized.element_size()
  assert num_slice_bytes % granularity == 0 and num_slice_bytes - granularity < localized[0].nbytes <= num_slice_bytes
  ```

  这正是 `empty_per_domain`（`apis/locality_domain.hpp:43-56`）的行为：`stride(0)` 被设成
  `align(slice_bytes, granularity)/elem_size`（`:46/51`），每片之间有空隙（对齐 padding），
  但浪费不超过一个粒度。
- **`is_localized` 语义**（`:90-94`）：本地化张量与 `torch.nn.Parameter(本地化, requires_grad=False)` 都 True；
  **切一个非域维的视图仍 True**（`:91` 的 `localized[:, :1]`）；但**普通张量、以及去掉域维的 `localized[0]`
  是 False**（`:92`）；而**非连续**张量（`:93` 的 `plain.mT`、`localized[..., :1]`）必须抛 `RuntimeError`。
  实现见 `csrc/utils/layout.hpp:142-152`：`is_localized` 先要求 `size(0)==num_domains` 且各片 stride 连续，
  再对每片 `get_locality_domain` 反查是否落在自己的域（`runtime/locality_domain.hpp:98-105`）；
  最后一行 `DG_HOST_ASSERT(localized or t.is_contiguous())`（`layout.hpp:150`）就是「非连续且非本地化 →
  报错」这句规格的落点。
- **往返 + 搬专家**（`:96-102`）：`as_plain(localized) == plain` 逐位；再把某个专家的本地化切片
  `localized[:,dst].copy_(localized[:,src])` 并同步改 `plain`，再比一次——证明「本地化张量也能当普通张量按行搬」。
- **收尾**（`:104`）：`deep_gemm.destroy_localizer()`。

### 4.3 `localize` 的拒绝语义（本机可验）

无 MLOPart 时，`localize` 与 `empty(合法域)` 一样在 `mlopart.py:40` 抛 `AssertionError`
（实测：`MLOPart is unavailable, so no localized memory can be allocated`）。`test_without_mlopart`
（§5.2）正是把这条写进了规格。

## 5. MLOPart 生命周期规范：`test_mlopart` / `test_without_mlopart`

### 5.1 `test_mlopart`（`:108-130`）

它测两件事：

1. **起与服务**（`:111-119`）：取 device UUID（`:111`），`mlopart.MLOPart(uuid)` 起进程（`:113`），
   对每个域 `create_memory(granularity, d)`（`:115`，即 §2 的 `serve()` 服务一次请求），记录耗时，
   `close()`（`:117`）后断言 `client.returncode == 0`（`:118`）。
2. **内存比分配器活得久**（`:121-129`）：`localize` 一个张量 → `destroy_localizer()`（只退分配器）
   → 张量数据仍然可读且 `as_plain == plain`；再次 `localize` 时**自动重启** MLOPart。
   这对应 `release_mlopart`（`apis/locality_domain.hpp:144-147`）的设计：**先把域表算出来
   （`get_balanced_sm_locality_domains()`）再 `mlopart::release()`**——因为域表一旦算出就是纯数据，
   不该随 MLOPart 进程消失。而底层被导出的内存句柄是 POSIX fd，进程退出后内存段仍由导出者持有
   （`mlopart.py:47` 注释「the exported memory outlives them」）。

本机需要真的能起 2-device MPS，故跑不了。

### 5.2 `test_without_mlopart`（`:133-157`）——本机真机通过

这个用例的巧妙之处：它**主动模拟「老 MPS 不认识 `-mlopart`」**——写一个假的 `nvidia-cuda-mps-control`
包一层，把命令行里的 ` -mlopart` 剥掉再转发给真 CLI（`tests/test_locality_domain.py:136` 的 `shutil.which` +
`:150-151` 的假脚本 `sed "s/ -mlopart//"`）。然后在子进程里跑一段脚本（`:137-148`），断言：

- `not deep_gemm.is_localization_available()`；
- **恰好一次** `UserWarning`，且文案含 `"MLOPart is unavailable"`（`:141`，对应 `mlopart.py:37`）；
- `get_sm_locality_domains().numel() == multi_processor_count`（`:142`）——**即使不可用，域表仍然可用**
  （走 §3.1 的 even 回退）；
- `localize(...)` 必须抛 `AssertionError` 且正文含 `"MLOPart is unavailable"`（`:143-148`）。

`result.returncode == 0` 且 stderr 含 `'1 MLOPart devices'`（`:154-155`）——后者正是 `serve()` 断言失败时
打印的那句，证明「不可用」走的是**同一条** `num_devices != 2` 路径，而不是别的分支。

这条规格回答了一个工程上很实际的问题：**「当宿主的 MPS 不支持 MLOPart 时，DeepGEMM 是否优雅退化？」**
答案是：告警一次、拒绝本地化、但域表照常给（even），不崩。

## 6. H100 实测（`harness/exp_locality_domain_test_oracle.{py,sh,out.txt}`）

实验在 `kernel_lab`（H100 80GB HBM3、sm90、132 SM）上跑，**16 PASS / 0 FAIL**。分六部分：

**Part A —— 顶层结构**：确认 5 用例 + 2 helper 可见、`__main__` 顺序、常量 `NUM_SMS_PER_TPC=2`。

**Part B —— 能力矩阵**：`domains=2`、`granularity=2 MiB`、`available=False`、`num_sms=132`；并给出
「为什么不可用」的完整调用链（§2）。

**Part C —— 原样跑两个可跑的用例**（导入真实测试模块、调用真实测试体）：

```
--- test_sm_locality_domains() ---
Testing SM locality domains:
 > [66, 66] SMs per domain, balanced [66, 66]
[PASS] C: 真实测试体 test_sm_locality_domains(): OK

--- test_without_mlopart() ---
Testing without MLOPart:
 > detected, `localize` refused, SM table available
[PASS] C: 真实测试体 test_without_mlopart(): OK
```

**Part D —— 独立复刻 host 数学 + 逐条核对规格断言**：

| 项 | 结果 |
|---|---|
| 独立 `get_even` 复刻 vs 真实 `sm_domains` | **逐位相等**（坐实第 34 篇「不可用时等于 `get_even`」） |
| 独立 `balance` 复刻 vs 真实 `balanced` | **逐位相等**（用正确的 `[num_tpcs/2:]`） |
| 规格断言① 0<sum<num_sms 且 dtype uint8 | PASS |
| 规格断言② TPC 内两 SM 同域 | PASS |
| 规格断言③ 恰好 50/50 | PASS |
| 规格断言④ 搬运数 == 公式 | PASS（0 == 0） |
| **利齿**：`surplus` 误取前半段 | 错误变体 `balanced=[0,132]`，**断言③ 不成立**（抓到实现错） |
| 合成输入 all-0 / all-1 / TPC 一致 40-26 / 已 50/50 | 分别 33/33/7/0 搬动，全符合公式 |

**Part E —— 契约负样本（与 MLOPart 无关，H100 可跑）**：

- E1 `assert_raises` 只吞 `RuntimeError`，`AssertionError` 穿透——验证 §1 的 helper 契约；
- E2 `is_localized(plain) == False`；`is_localized(plain.mT)` 抛 `RuntimeError`，正文
  `Assertion error (utils/layout.hpp:150): localized or t.is_contiguous()`——**§4.2 那条规格的真实落点**；
- E3 `empty` 域越界（-1/2）抛 `RuntimeError`（`locality_domain.hpp:36`）——**§4.1 那条契约的落点**；
- E4 无 MLOPart 时 `empty(合法域)` 与 `localize` 抛 `AssertionError`「MLOPart is unavailable」。

**Part F —— 跑不了的 3 个用例**：把 `test_empty` / `test_localize` 真的跑一遍，记录它们各自挂在哪一行
（`test_empty` → `:58` 的 `empty(...,0)` 在 `mlopart.py:40`；`test_localize` → `:77` 的可用性断言），
并定位 `test_mlopart` 的阻塞点（`:113` 起 MLOPart、`serve()` 断言 2 device）。

**小结**：5 个用例里 **2 个本机真机通过**（域表 / 无 MLOPart），其余 3 个的阻塞点全部定位到具体行；
而它们**与 MLOPart 无关的契约断言**（越界异常、`is_localized` 非连续拒绝、`localize` 拒绝）已在 Part E
在真机上单独验证。目标卡（有真正的 2 个物理域 / MLOPart）上，5 个用例应整份可跑。

## 7. 与第 34 篇的关系、小结

| | 第 34 篇（P24-1，`docs/34`） | 本篇（P71-1） |
|---|---|---|
| 对象 | 局部域的**实现**：探针 / `try_probe` / `balance` / MLOPart / VMM | `tests/test_locality_domain.py` 的**规范** |
| 回答 | 「域怎么测出来、内存怎么放上去」 | 「域表/本地化内存/MLOPart 算什么对、判据怎么编码」 |
| 手段 | 自写 pointer-chase 探针 + C++ host 复刻 + host-pinned 模拟远端域 | 原样跑真实测试体 + 独立复刻 host 数学 + 契约负样本 |
| 本机结论 | 无本地化时两域延迟相同 → 探针正确拒绝 → even 66/66 | 2 个用例真机通过；规格断言③+④ 编码了 `slice` 的「后半段」语义 |

**三个可复用结论**：

1. **域表的规范（`test_sm_locality_domains`）与实现（`get_even`/`balance`）互相锁死**：断言③（恰好 50/50）
   与断言④（最少搬动）合起来逼出的实现细节是——`surplus` 必须取多数域 TPC 的**后半段**。
   这正是 libtorch `slice(dim, start)` 「第二参是起点不是长度」这个陷阱被规格无意间钉住的地方；
   把 `slice` 读错成「前一半」，均衡输入会被错误翻成 `[0,132]`，被断言③当场抓住。
2. **一份测试可以用「异常类型」把两件事分开**：`assert_raises` 只认 `RuntimeError`，
   于是「域索引越界」（C++ `DG_HOST_ASSERT`）与「MLOPart 不可用」（Python `assert`）被区分成
   两种异常——这让**越界契约在无 MLOPart 的 H100 上也能验**，也让「不可用」的拒绝能在
   `test_without_mlopart` 里被独立断言。判据的设计本身就在表达「这两件事不是一回事」。
3. **优雅退化是规格的一部分**：`test_without_mlopart` 用「剥掉 `-mlopart` 的假 MPS」模拟老宿主，
   断言「告警一次 + 拒绝本地化 + 域表照给 even」。所以第 34 篇那句「不可用时 `get_sm_locality_domains()`
   逐位等于 `get_even`」不是巧合，而是被测试显式要求的**可执行契约**。

**边界**：本篇**没有任何 SM100 专属内容**——域表、本地化内存、MLOPart 生命周期都是 host/API 层，
与 GPU kernel 架构无关。H100 上「不可用」导致 3 个用例跑不了，是**环境**（MPS 只给 1 个 device）而非
架构/内核问题；目标卡上的差别只是「能真的分出两个域」，而这些用例的语义不变。

**本篇实测脚本**：`harness/exp_locality_domain_test_oracle.py` / `.sh` / `.out.txt`
（H100 真机，原样跑 2 个可跑用例 + 独立复刻 host 数学 + 契约负样本，16 PASS / 0 FAIL）。

参考链接（真实）：
- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
- CUDA VMM（`cuMemCreate` / 可共享句柄 / locality domain）：
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>
- NVIDIA MPS / MLOPart：<https://docs.nvidia.com/deploy/mps/index.html>
