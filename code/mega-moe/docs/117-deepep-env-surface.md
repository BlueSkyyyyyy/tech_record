# DeepEP 的环境变量 / 开关面：persistent envs、`get_env` 解析陷阱与 11 个运行时 `EP_*`

> 本文对应路线图 **P107-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的**第 117 篇**。
>
> 前面 116 篇把 DeepGEMM / DeepEP 的 kernel、host 编排、测试基座几乎拆尽了，但有一层**每次部署都要碰、
> 却从没有单独成篇**的东西：**环境变量**。DeepEP 的行为在四个时间点上被 `EP_*` 环境变量塑形——
> `import deep_ep` 时（引导闸门）、构造 `Context`/`Buffer` 时（域与 RDMA 配置）、**每次 API 调用**时、
> 以及 JIT **编译期**（nvcc flag）。它们散落在 `csrc/{utils,buffers,kernels,runtime}`、
> `deep_ep/{__init__,buffers,comm,utils}` 与 `setup.py` 里，读者只能在一篇篇 kernel 分析中零散地瞥见。
>
> 本篇把这条「部署开关面」单独讲透，并与第 38 篇（DeepGEMM 的全局运行时/编译开关面）成对照——
> 那是 **`DG_*`/`set_*`**，本篇是 **`EP_*`**。
>
> **可跑性**：整层是**纯 host / Python + 编译期**，不碰 UMMA/TMA/NCCL 设备 API，也非 SM100 专属。
> `get_env<T>` 用 g++ **直接 include 真实 `csrc/utils/system.hpp`** 编译运行；persistent envs / 引导闸门 /
> 全库清单普查用真实生成的 `deep_ep/envs.py` 与真实 `check_nccl_so` 函数体在 H100(sm90) 上跑。
> **H100 真机 38 PASS / 0 FAIL**（`harness/exp_deepep_envs.{sh,out.txt}` + `harness/deepep_env_probe.cpp`）。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - 解析器：`csrc/utils/system.hpp:12-29`
> - persistent envs：`setup.py:14-15`、`:63-74`；应用点 `deep_ep/__init__.py:8-16`
> - 引导闸门：`deep_ep/__init__.py:19-41`（`check_nccl_so`）
> - 运行时消费点见下文清单表
> - 实验：`harness/deepep_env_probe.cpp`（Part A）+ `harness/exp_deepep_envs.py`（Part B）+ `.sh/.out.txt`

---

## 1. 为什么 Mega MoE 的部署绕不开这层

Mega MoE 的「通信—计算重叠」把性能压榨到硬件极限：dispatch/combine 用几个 SM、走哪条 RDMA 通道、
NCCL 用哪份库、top-k 索引是 32 位还是 64 位——**每一个都直接决定 kernel 能不能起来、跑多快、结果对不对**。
这些都不是超参，而是**环境变量在四个时间点**定的：

| 时间点 | 谁在读 | 代表变量 | 起不来 / 跑错的后果 |
|---|---|---|---|
| `import deep_ep` | Python 引导 | `EP_SUPPRESS_NCCL_CHECK` | 二进制 NCCL 不符 → `AssertionError`，连 import 都过不去 |
| 构造 `Context` / `Buffer` | C++ host | `EP_DISABLE_GIN`、`EP_*_RDMA_SL`、`EP_NUM_MAX_LOCAL_RANKS` | GIN 不可用 → 断言；SL 选错 → 网络拥塞 |
| 每次 API 调用 | C++ / Python | `EP_AVOID_RECORD_STREAM`、`EP_BUFFER_DEBUG`、`EP_REUSE_NCCL_COMM` | 流同步错 → 数据竞态；或调试打印开关失灵 |
| JIT 编译期 | nvcc flag | `EP_GIN_GDAKI_DEBUG`、`EP_NUM_TOPK_IDX_BITS` | host 与 device 对 `topk_idx_t` 的理解错位 |

DeepGEMM 的第 38 篇已把 `DG_*`/`set_num_sms`/`set_tc_util` 那套讲清；DeepEP 这层是**它的对偶**，
却更「隐式」：DeepGEMM 的开关有 pybind 入口（`deep_gemm.set_*`），而 DeepEP 的 `EP_*` **全靠 `getenv`**，
没有回读、没有类型校验、还跨两种语言（C++ 与 Python）各解析一遍——这正是本篇要钉死的坑。

---

## 2. `get_env<T>`：21 行的解析器，两个真实陷阱

所有 C++ 侧 `EP_*` 都过这一个模板函数（`csrc/utils/system.hpp:12-29`）：

```cpp
template <typename dtype_t>
static dtype_t get_env(const std::string& name, const dtype_t& default_value = dtype_t()) {
    const auto c_str = std::getenv(name.c_str());
    if (c_str == nullptr) return default_value;
    if constexpr (std::is_same_v<dtype_t, std::string>) return std::string(c_str);
    else if constexpr (std::is_same_v<dtype_t, int>) { int value; std::sscanf(c_str, "%d", &value); return value; }
    else EP_HOST_ASSERT(false and "Unexpected type");
}
```

第 98 篇已经点过「未设返回默认值」「int 用 `sscanf` 取前缀」两条。本篇用 g++ **真跑真实头文件**
补上更硬的两条：

**陷阱 ①：畸形 / 空值下 `value` 未初始化（UB）。** `std::sscanf` 的返回值被**丢弃**。当值不是
数字开头（`abc`）或为空串（`EP_BUFFER_DEBUG=`）时，`sscanf` 返回 0 且**不写 `value`**，函数却照样
`return value` —— 于是返回栈上的垃圾值，**既不是默认值也不是 0**。H100 实测（`-O0`）：

```
[INFO] get_env<int>("abc")   = 21925 (默认值 -999 未生效，未设才算未设)
[INFO] get_env<int>("")      = 21925
[INFO] 哨兵复刻 sentinel=0x11111111 -> 0x11111111（未变）
[INFO] 哨兵复刻 sentinel=0x22222222 -> 0x22222222（未变）
```

> **为什么值得记**：一个「默认关」的开关写成 `EP_AVOID_RECORD_STREAM=yes`（本意打开），`getenv` 非空
> → `sscanf("yes")` 返回 0 → 返回垃圾；垃圾是否非零**纯看栈**，于是同一个配置在不同调用栈下
> **时开时关**、不可复现。而 `EP_AVOID_RECORD_STREAM` 恰好控制 `record_stream`（见 §4），
> 一旦被「随机」打开，CUDA Graph 场景可能静默改行为。**正确写法只有 `EP_X=0` / `EP_X=1`（或数字前缀）**。
> 真实调用点用 `get_env<int>("...", 0)`，其中 `0` 只在**完全未设**时生效（`nullptr` 分支），
> 这与「设为 `abc`」是两条完全不同的路径。实验里用哨兵复刻把「`sscanf` 不写 `value`」这一行为固定下来，
> 避开了真实读 UB 的随机性。

**陷阱 ②：只支持 `int` 与 `string`。** 传别的类型（如 `bool`）走 `EP_HOST_ASSERT(false)` 抛
`EPException`（实测消息 `system.hpp:27: false and "Unexpected type"`）——这防止了
`get_env<bool>("...")` 这种「想当然」。所以**所有 DeepEP 开关都是 `int`，没有真正的布尔**。

---

## 3. 两条通道：安装期固化的 persistent envs + 运行时 `EP_*`

DeepEP 的环境变量分两条互不重叠的通道，理解它们的关键是**生命周期**。

### 3.1 persistent envs（安装期固化进 `deep_ep/envs.py`）

`setup.py:14-15` 的白名单：

```
persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND', 'EP_JIT_CPP_STANDARD',
                        'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR', 'EP_DEFAULT_RDMA_SL', 'EP_OVERRIDE_RDMA_SL')
```

`pip install` 时 `generate_default_envs`（`setup.py:63-74`）把**当刻存在的**白名单变量写进生成的
`deep_ep/envs.py`；运行时 `deep_ep/__init__.py:8-16` 用 `try/except ImportError` 包住，
**只在「环境里没设」时补上、绝不覆盖**：

```python
from .envs import persistent_envs
for key, value in persistent_envs.items():
    if key not in os.environ:
        os.environ[key] = value
```

本机安装时白名单里恰好只有一条被固化（真实生成物）：

```
# Pre-installed environment variables
persistent_envs = dict()
persistent_envs['EP_NCCL_ROOT_DIR'] = '/tmp/nccl229'
```

即**这份 DeepEP 被编到 `/tmp/nccl229` 的 NCCL 上**；谁 `import deep_ep`，`EP_NCCL_ROOT_DIR`
就被钉到 `/tmp/nccl229`（除非显式覆盖）。H100 实测坐实「只补不覆盖」：子进程里显式
`EP_NCCL_ROOT_DIR=/explicit/wins` → 读回 `/explicit/wins`；完全清掉该变量 → 读回 `/tmp/nccl229`。
这正是 `check_nccl_so` 与 `find_nccl_root()` 能一致工作、而本机运行期仍失败（见 §5）的根因之一。

> `EP_JIT_CACHE_DIR` / `EP_JIT_PRINT_COMPILER_COMMAND` / `EP_JIT_CPP_STANDARD` 三条由 **DeepJIT 的
> `Env`**（`env_prefix == "EP"`）消费，`EP_NUM_TOPK_IDX_BITS` 横跨 host `_C` 与 device JIT 两面——
> 这三条与第 98 篇（`compiled.cuh` + `init_jit`）交叉。

### 3.2 运行时 / 编译期 `EP_*`：全库清单

用带引号正则（`["']EP_[A-Z0-9_]+["']`，**排除裸 `EP_*` 宏**如 `EP_HOST_ASSERT`/`EP_STATIC_ASSERT`）
扫描两个仓库的 `.py/.cpp/.cuh/.hpp/.cu`，得**恰好 16 个**带引号的 `EP_*` 名字 = **7 条 persistent + 9 条
仅运行时**，二者并集即 **11 条「运行时开关」**（其中 `EP_DEFAULT_RDMA_SL`/`EP_OVERRIDE_RDMA_SL`
既是 persistent 又是运行时读取）。真机普查结果：

| 变量 | 生命周期 | 消费点（commit `93eb6eb`） | 作用 / 默认 |
|---|---|---|---|
| `EP_SUPPRESS_NCCL_CHECK` | import | `deep_ep/__init__.py:24` | =1 时跳过 NCCL 二进制核对（引导逃生门） |
| `EP_REUSE_NCCL_COMM` | 每构造 | `deep_ep/comm/handle.py:64` | ≠0（**默认 1**）复用 PyTorch backend 的 `_comm_ptr` |
| `EP_DISABLE_GIN` | 构造 | `csrc/kernels/comm/context.cpp:124` | ≠0 时**不申请 GIN/GDAKI**（只 NVLink 模式） |
| `EP_DEFAULT_RDMA_SL` | 构造 | `context.cpp:149-150` | RDMA traffic class 基线 |
| `EP_OVERRIDE_RDMA_SL` | 构造 | `context.cpp:153-154`（`envs.py:106` setdefault=1） | 最高优先级覆盖 SL |
| `EP_BUFFER_DEBUG` | 构造 + 每调用 | C++ `context.cpp:41/48/103/112`、`buffers/ep.hpp:454/653`；Python `buffers/ep.py:309/535` | 打开调试打印 |
| `EP_AVOID_RECORD_STREAM` | 每调用 | `buffers/ep.hpp:163`、`buffers/bucket.hpp:51` | =1 时用事件持有张量、不调 `record_stream`（CUDA Graph 友好） |
| `EP_NUM_MAX_LOCAL_RANKS` | 构造 | `buffers/engram.py:61` | hybrid 下 RDMA storage 预留的本地卡数（**默认 16**） |
| `EP_USE_NVIDIA_TOOLS` | 每调用 | `utils/testing.py:145` | =1 时 `bench_kineto` 直接旁路（与 Nsight 冲突） |
| `EP_DISABLE_BARRIER_PROFILING` | 每调用 | `utils/testing.py:154` | ≠0 时关掉 barrier 的 kineto profiling |
| `EP_GIN_GDAKI_DEBUG` | 编译期 | `csrc/runtime/jit.hpp:43` | ≠0 时给 nvcc 加 `-DNCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG=1` |

（另有 5 条 persistent-only：`EP_JIT_CACHE_DIR`/`EP_JIT_PRINT_COMPILER_COMMAND`/`EP_JIT_CPP_STANDARD`/
`EP_NUM_TOPK_IDX_BITS`/`EP_NCCL_ROOT_DIR`，见 §3.1。）

---

## 4. 三条最值得单独讲的

### 4.1 `EP_SUPPRESS_NCCL_CHECK`：引导逃生门

`check_nccl_so`（`deep_ep/__init__.py:19-41`）是 DeepEP **唯一**在 `import` 阶段就执行的检查，
本文档系列「本机阻塞」正卡在这里。它有三把尺子：① `:24` 的抑制开关
（`int(os.environ.get('EP_SUPPRESS_NCCL_CHECK', 0))`，`=1` 直接 `return`）；② `:28-32` 扫
`/proc/self/maps` 确认没有**重复**加载的 `libnccl`；③ `:33-41` 用 `filecmp.cmp(..., shallow=False)`
**逐字节**比 PyTorch 链的 NCCL 与 DeepEP 链的 NCCL。H100 实测：

```
[PASS] EP_SUPPRESS_NCCL_CHECK=1 -> 直接返回 None（绕过二进制核对）
[INFO] 未抑制时 AssertionError: Invalid NCCL versions: /usr/lib/.../libnccl.so.2.29.7 (loaded)
       v.s. /usr/local/lib/python3.12/... (expected)
[PASS] 未抑制 -> AssertionError（本机 PyTorch NCCL != linked）
```

注意**「加载了才比」**：若进程里压根没加载 `libnccl`，`loaded_nccl_so` 会是 `None`，
`filecmp.cmp(None, ...)` 抛 `TypeError` 而非断言——这也解释了为什么上游是在
`import torch`（`:4`）之后才跑这函数。实验里先 `torch.cuda.nccl.version()` 强制加载才复现出断言。

### 4.2 `EP_AVOID_RECORD_STREAM`：CUDA Graph 友好的流同步

`record_stream` 是分配器的副作用、对 CUDA Graph 不感知，故 DeepEP 给了逃生门：打开后改用
`EventHandle::tensors_to_record` 持有张量引用（`buffers/ep.hpp:163-164`、`buffers/bucket.hpp:51-52`）。
源码注释（`ep.hpp:162`）点明「**this environment only applies to V2 APIs**」——即只对
`EPBuffer`/`BucketBuffer` 的 V2 流控制路径有效，老 V1 路径仍直接 `record_stream`。
第 109 篇（CUDA Graph）已用过这条，此处补上它**只影响 V2** 的边界。

### 4.3 `EP_NUM_MAX_LOCAL_RANKS`：默认 16 与一个显存副作用

`engram.py:61`：

```python
num_max_local_ranks = int(os.getenv('EP_NUM_MAX_LOCAL_RANKS', 16)) if allow_hybrid_mode else 1
num_registered_bytes = num_gpu_bytes + num_rdma_storage_bytes * num_max_local_ranks + (1 << 32)
```

即 **hybrid 模式默认按「本节点 16 张卡」预留 RDMA storage**（非 hybrid 恒按 1 份），
因为跨本地 rank 的对称窗口要一次性注册。若 `num_registered_bytes` 超显存，
`engram.py:64-65` 会**顺手设一个全局副作用** `os.environ['NCCL_WIN_STRIDE']`（对齐到 4 GiB）。
H100 实测（复刻公式）：

```
[PASS] 默认 16 / 非 hybrid 为 1
[PASS] 小缓冲不触发 NCCL_WIN_STRIDE registered=5.02 GiB
[PASS] 大缓冲触发 NCCL_WIN_STRIDE（对齐 4 GiB） registered=220.0 GiB -> 236223201280
[PASS] 非 hybrid 只按 1 份 RDMA storage 预留 hybrid=20 vs non=5 GiB
```

**为什么默认 16**：这是「本节点最多 16 卡」的保守上界（HGX/DGX 常见 8，但预留一倍安全）；
调大它换泛化、代价是显存；调小则可能在多于该数的节点上分配不足。

---

## 5. 跨语言陷阱：同一个 `EP_BUFFER_DEBUG`，两种解析

`EP_BUFFER_DEBUG` 同时被 C++ 与 Python 读，但两边的解析**语义不同**：

* C++：`get_env<int>("EP_BUFFER_DEBUG", 0)` → `"0"` 解析成 **0（关）**；
* Python：`os.environ.get('EP_BUFFER_DEBUG', 0)`（`buffers/ep.py:309/535`）→ `"0"` 是**非空字符串**，
  布尔为 **True（开）**。

H100 实测：

```
[PASS] C++: EP_BUFFER_DEBUG="0" -> get_env<int>=0（关）
[PASS] Python: EP_BUFFER_DEBUG="0" 为真（会打印）
```

即 `EP_BUFFER_DEBUG=0` 这个「看起来是关」的写法，**在 C++ 侧关、在 Python 侧开**——
调试输出会从一个进程里冒出来，而同一进程的 C++ 打印却沉默。这是两个语言各解析一遍环境变量的
典型代价。相比之下 `EP_SUPPRESS_NCCL_CHECK`/`EP_REUSE_NCCL_COMM`/`EP_USE_NVIDIA_TOOLS`/
`EP_DISABLE_BARRIER_PROFILING`/`EP_NUM_MAX_LOCAL_RANKS` 在 Python 侧都套了 `int(...)`，
语义与 C++ 一致；**唯独 `EP_BUFFER_DEBUG` 是裸真值**。

---

## 6. 与 DeepGEMM 开关面的对照

| 维度 | DeepGEMM（第 38 篇） | DeepEP（本篇） |
|---|---|---|
| 主入口 | `deep_gemm.set_*` / `get_*`（pybind，可回读） | 环境变量 `EP_*`（`getenv`，**不可回读**） |
| 解析器 | `deep_jit::Env`（`DG_` 优先 / `DJ_` 兜底） | `csrc/utils/system.hpp::get_env`（int/string） |
| 安装期固化 | `deep_gemm/envs.py` 的 `persistent_envs` | `deep_ep/envs.py` 的 `persistent_envs` |
| 引导期闸门 | 无（`import` 不联网、不比库） | **`check_nccl_so`**（可比库、可抑制） |
| 编译期开关 | `set_pdl` → 发射属性；`ignore_compile_dims` → 编译维 | `EP_GIN_GDAKI_DEBUG` → nvcc flag；`EP_NUM_TOPK_IDX_BITS` → host+device 宏 |
| 跨语言解析 | 只有 Python 侧 API，无 C++/Python 双读 | **C++ 与 Python 双读**（`EP_BUFFER_DEBUG` 陷阱） |

一句话：**DeepGEMM 的开关「有 API、有回读、单一解析」；DeepEP 的开关「是环境变量、无回读、双解析」**。
两者都靠 `persistent_envs` 解决「安装环境 ≠ 运行环境」，但 DeepEP 多一层安装期 NCCL 二进制绑定与
引导期核对——正是它让「拿到一个能跑的 NCCL 环境」成为 DeepEP 部署的硬门槛（见路线图「阻塞」）。

---

## 7. H100 实测小结

`harness/exp_deepep_envs.{sh,out.txt}` + `harness/deepep_env_probe.cpp`，**38 PASS / 0 FAIL**：

* **Part A（g++，真实 `system.hpp`）9 PASS**：未设→默认、`123abc`→123、string 直取、
  `bool`→`EPException`、畸形/空值→哨兵复刻坐实「`sscanf` 不写 value」、未设才走默认、
  `EP_BUFFER_DEBUG="0"` 的 C++ int 解析。
* **Part B（Python，真实 `deep_ep/envs.py` + `check_nccl_so`）**：persistent 白名单恰 7 条、
  真实生成物只固化 `EP_NCCL_ROOT_DIR=/tmp/nccl229`、「已设不覆盖 / 未设补上」两例、
  `EP_SUPPRESS_NCCL_CHECK=1` 短路返回、未抑制抛 `Invalid NCCL versions` 断言、全库带引号
  `EP_*` 恰 16 个（7 persistent + 9 仅运行时）、11 条运行时开关全部命中。

**结论**：DeepEP 的开关面是**纯 host / 编译期**、架构无关的一层，H100(sm90) 上可**逐条真跑**；
其中三个真坑（`get_env` 未初始化、`EP_BUFFER_DEBUG` 跨语言、`EP_NUM_MAX_LOCAL_RANKS` 的
`NCCL_WIN_STRIDE` 副作用）在源码里都无注释提示，只能靠实测/精读发现。

**参考链接**：DeepEP 仓库 <https://github.com/deepseek-ai/DeepEP>（本机核对 commit `93eb6eb`、
`deep_ep.__version__ == 2.5.0`）。
