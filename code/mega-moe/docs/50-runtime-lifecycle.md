# DeepGEMM 运行时对象的生命周期与资源所有权：lazy init / shutdown / cuBLASLt handle 与 workspace（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P40 增量（第 50 篇）。
> 第 48 篇（P38）拆过 DeepGEMM 顶层的**开关面**——`set_num_sms` / `set_tc_util` / `set_pdl` /
> `use_deterministic_algorithms` 等七个开关「改什么状态、被谁消费」。但那一篇只把 `Runtime`
> 当成「一个装着资源字段的 singleton」一笔带过，从没回答这层更基础的问题：**这个 singleton
> 什么时候被创建、什么时候被销毁、它持有的 cuBLASLt handle 和 workspace 归谁所有、进程退出时
> 的析构顺序为什么危险、`deep_gemm.shutdown()` 到底做了什么**。
>
> 本篇就把 `Runtime` 的**生命周期与资源所有权**单独拆透，并给出三条与内核无关、可在 H100 上
> 逐项观测的硬证据：① 用 LD_PRELOAD 探针统计 `cublasLtCreate/Destroy` 的真实调用次数；
> ② 用真实显存账证明「每个 CUDA stream 一份 16 MiB workspace」；③ 证明 `shutdown()` 的**终态语义**。
> 参考仓库 `~/github/DeepGEMM`，核对 commit `057ca59`（v2.8.1）。本文行号均在该 commit 上核对过。

---

## 1. 为什么值得单独讲：一个「有析构副作用的 singleton」

DeepGEMM 不是无状态函数库。除了第 48 篇讲的两份**可变状态** singleton，它还有**第三份常识被忽略的东西**：
一组**持有系统资源、且在析构时释放它们**的对象：

- `csrc/runtime/runtime.hpp:104`
  `inline auto runtime = deep_jit::LazyInit<Runtime>([](){ return std::make_shared<Runtime>(); });`
  ——`Runtime` 在构造时 `cublasLtCreate`（`:50-51`），在析构时 `cublasLtDestroy`（`:54-57`）。
- `csrc/runtime/jit.hpp:12` 的 `jit`（DeepJIT 运行时，管编译/缓存/发射，见第 33/47 篇）。
- `csrc/jit_kernels/heuristics/runtime.hpp:67` 的 `heuristics_runtime`（管选核/编译维）。

三者都是 `deep_jit::LazyInit<T>`。LazyInit 把「首次访问才构造」和「可以整体替换（含置空）」两件事
绑在一起，于是 `deep_gemm.shutdown()` 能用一行代码实现（`csrc/apis/config.hpp:12-14`）：

```cpp
m.def("shutdown", []() {
    runtime = deep_jit::LazyInit<Runtime>(nullptr);
});
```

注意这里**只重置了 `runtime`，没有动 `jit` / `heuristics_runtime`**——这决定了 `shutdown()` 的语义，
以及为什么它到底还是必要的。下面逐层拆开。

---

## 2. `LazyInit`：null factory 就是「毒药」

`third-party/deep_jit/include/deep_jit/utils/lazy.hpp:15-38` 是整个生命周期的地基：

```cpp
template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;
public:
    explicit LazyInit(std::nullptr_t) {}                                    // :21
    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)          // :23
        : factory(std::move(factory)) {}
    T* operator->() {                                                       // :26
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");  // :27
        if (ptr == nullptr)
            ptr = factory();                                                // :29
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");            // :30
        return ptr.get();
    }
    std::shared_ptr<T> get() { (void)operator->(); return ptr; }            // :34-37
};
```

三个性质，全部来自这十几行：

1. **首次访问才构造**：`ptr` 默认空，只有 `operator->` 被调用时才 `ptr = factory()`（`:28-29`）。
   所有 pybind 入口都写 `runtime->...`，所以「不碰任何 DeepGEMM API 就永远不创建 `Runtime`」。
2. **null factory = 终态**：`LazyInit(std::nullptr_t)`（`:21`）把 `factory` 置空，于是之后任何一次
   `runtime->` 都会在 `:27` 抛 `DJ_HOST_ASSERT`。这正是 `shutdown()` 的实现——**它不是「销毁后
   可以重建」，而是「把这个 singleton 永久标记为未初始化」**。
3. **可替换即析构**：`runtime = LazyInit<Runtime>(nullptr)` 是给全局 `runtime` **整体赋值**。左侧
   对象原来的 `std::shared_ptr<T> ptr` 成员被覆盖，引用计数归零 → 立刻执行 `~Runtime()`。

关于第 3 点有个细节：`~Runtime()`（`runtime.hpp:54`）被声明为 `noexcept(false)`，因为它内部
`cublasLtDestroy` 失败会 `throw`（`DG_CUBLASLT_CHECK`，`csrc/utils/exception.hpp:55-65`）。
换句话说，**`shutdown()` 是可能抛异常的**，不能假设它绝对安静。

---

## 3. `Runtime` 持有的两类资源：cuBLASLt handle 与 per-stream workspace

`Runtime`（`csrc/runtime/runtime.hpp:20-102`）真正「有所有权」的字段只有两组：

```cpp
static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;               // :26
std::unordered_map<c10::cuda::CUDAStream, torch::Tensor> cublaslt_workspaces;    // :28

cublasLtHandle_t cublaslt_handle;                                               // :31
bool use_pytorch_managed_cublaslt_handle;                                       // :32
bool use_temp_cublaslt_workspace;                                              // :33
```

### 3.1 handle：默认自建，两个原因

构造（`:40-52`）与析构（`:54-57`）：

```cpp
use_pytorch_managed_cublaslt_handle = deep_jit::get_env<int>("DG_USE_PYTORCH_CUBLASLT_HANDLE", 0) > 0;  // :44
use_temp_cublaslt_workspace = deep_jit::get_env<int>("DG_USE_TEMP_CUBLASLT_WORKSPACE", 0) > 0;          // :48
if (not use_pytorch_managed_cublaslt_handle)
    DG_CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));                        // :50-51
```

```cpp
~Runtime() noexcept(false) {
    if (not use_pytorch_managed_cublaslt_handle)
        DG_CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));                    // :55-56
}
```

**为什么默认自己建 handle？** 代码注释写得很直白（`:41-43`）：`at::cuda::getCurrentCUDABlasLtHandle`
在一些 PyTorch 版本上 **CPU 开销很大**，所以默认不用 PyTorch 的 handle，而是自己 `cublasLtCreate`
一份、由 `~Runtime()` 负责 `cublasLtDestroy`。`get_cublaslt_handle()`（`:59-65`）按开关在两者间切换：

```cpp
cublasLtHandle_t get_cublaslt_handle() const {
    if (use_pytorch_managed_cublaslt_handle)
        return at::cuda::getCurrentCUDABlasLtHandle();
    return cublaslt_handle;                                                     // :64
}
```

于是「谁创建、谁销毁」是一一对应的：自建 → 自毁；托管 → 谁也不毁（归 PyTorch）。

### 3.2 workspace：为什么是「每 stream 一份」，且为什么可以改成「每次一份」

`get_cublaslt_workspace`（`:67-76`）：

```cpp
torch::Tensor get_cublaslt_workspace(const c10::cuda::CUDAStream& stream) {
    const auto options = dtype(torch::kByte).device(stream.device());
    if (use_temp_cublaslt_workspace)
        return torch::empty({kCublasLtWorkspaceSize}, options);                 // :70 临时
    auto& workspace = cublaslt_workspaces[stream];                             // :72 按 stream 缓存
    if (not workspace.defined())
        workspace = torch::empty({kCublasLtWorkspaceSize}, options);            // :74
    return workspace;
}
```

**为什么 key 是 stream？** 因为 cuBLASLt 可能**异步**使用这块 workspace，所以不同 stream 不能共享
同一块（注释 `:27`）。**为什么要有 `use_temp_cublaslt_workspace` 分支？** 注释 `:45-48`：compute-sanitizer
测试会在 CUDA driver 已经卸载之后再析构 workspace tensor，从而触发 `cudaErrorCudartUnloading`；
改成「每次调用现开现弃」就能避开这个卸载顺序问题（上游 `tests/test_sanitizer.py:56` 正是设了
`DG_USE_TEMP_CUBLASLT_WORKSPACE=1`）。这是一个纯工程性的逃生阀。

调用点只有一处：`csrc/jit_kernels/impls/smxx_cublaslt.hpp:73-77`

```cpp
const auto handle = runtime->get_cublaslt_handle();                            // :74
const auto workspace = runtime->get_cublaslt_workspace(stream);                // :76
const auto workspace_bytes = workspace.nbytes();                               // :77
```

workspace 立刻被喂给 heuristic 的 `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`（`:85-86`）和
`cublasLtMatmul`（`:95-105`）。**16 MiB 这个数字不是随便定的**（注释 `:25`）：workspace 一旦大于
16 MiB，cuBLAS 对某些形状（如 `m=128, n=7168, k=16384`）反而会选出更差的 heuristic。

> 顺带说明 `Runtime` 里剩下两个字段 `sm_locality_domains` / `balanced_sm_locality_domains`
> （`:35-38`）是**懒探针**（读注释：`probed lazily`），所有权只在这两个 `torch::Tensor` 上，
> 属于第 34 篇（P24）的局部域探针，本篇不重复。

---

## 4. 退出顺序：为什么 `deep_gemm/__init__.py` 要 `atexit.register(shutdown)`

`deep_gemm/__init__.py` 在 import 末尾做了两件事（`:121-125`）：

```python
_C.init(os.path.dirname(os.path.abspath(__file__)))                             # :122
# Native cuBLASLt exit callbacks can free handle storage before C++ static destructors.
# Release handles and workspaces while Python, PyTorch and CUDA are still available.
atexit.register(shutdown)                                                       # :125
```

注释就是答案：**cuBLASLt 的原生退出回调可能先一步释放 handle 占用的存储，早于 C++ 静态析构函数**。
如果放任 `~Runtime()` 在扩展模块静态析构阶段才跑，它会在「handle 存储已经被 cuBLASLt 的退出回调
释放之后」再调用 `cublasLtDestroy`——典型 use-after-free。在 `atexit` 里提前 `shutdown()`，
就能在 **Python、PyTorch、CUDA 都还活着**的时候把 handle 和 workspace 释放干净。

这就是上游 `tests/test_lazy_init.py` 专门测的东西。它用一段 LD_PRELOAD 探针（`:22-52`）拦截
`cublasLtCreate/Destroy`：

```c
static uint32_t live_handles = 0, handle_token = 0;
static bool backend_dead = false;
static void verify_cleanup() {
    if (live_handles != 0) std::_Exit(92);        // 退出时还有活 handle → 失败
    std::fputs("DG_PROBE_CLEAN\n", stderr);
}
__attribute__((constructor)) static void initialize_probe() {
    std::atexit(verify_cleanup);                  // 最先注册 → 最后执行（LIFO）
}
static void stop_backend() { backend_dead = true; }
extern "C" uint32_t cublasLtCreate(void** handle) { std::atexit(stop_backend); ... ++live_handles; ... }
extern "C" uint32_t cublasLtDestroy(void* handle) { if (backend_dead) std::_Exit(91); ... --live_handles; ... }
```

探针在**最先**注册 `verify_cleanup`，因此在 atexit 的 LIFO 顺序里**最后**执行——它相当于一个
「进程退出时的账本结算」：只要退出时 `live_handles == 0` 就打印 `DG_PROBE_CLEAN`，否则 `_Exit(92)`。
上游测了 6 组「寿命」用例（`:85-92` / `:101-113`）：`unused` / `owned` / `managed` / `explicit` /
`managed_explicit` / `unused_explicit`，断言每组「create 次数 == destroy 次数 == 期望」且 `clean` 恰一次。

---

## 5. `shutdown()` 的确切语义（三条推论）

综合第 2~4 节，`deep_gemm.shutdown()` 的行为可以精确预测：

1. **只重置 `runtime`**（`config.hpp:13`）。`jit`、`heuristics_runtime` 是**另外两个独立 singleton**
   （`jit.hpp:12`、`heuristics/runtime.hpp:67`），不受影响。所以 `set_pdl` / `get_pdl`（走 `jit->`，
   `config.hpp:27-32`）在 `shutdown()` 之后**结构上仍然可用**，而一切走 `runtime->` 的入口会报错。
2. **终态、幂等**：`shutdown()` 把 `runtime` 换成 null-factory。再次 `shutdown()` 只是再赋一次同样的
   null-factory，无副作用（幂等）；而之后任何 `runtime->` 都抛 `lazy.hpp:27`。**`_C.init` 也救不回来**，
   因为它只调 `init_jit`（`jit.hpp:14-32`），根本不碰 `runtime`。
3. **`atexit` 注册的 `shutdown`**（`__init__.py:125`）保证正常退出时 handle 在 Python/PyTorch/CUDA
   仍在时被释放，规避第 4 节的 use-after-free；同时把 workspace tensor 一并析构（它们挂在该
   `Runtime` 的 map 里，`runtime.hpp:28`）。

---

## 6. H100 实测

实验：`harness/exp_runtime_lifecycle.py`（driver `harness/exp_runtime_lifecycle.sh`），
原始输出 `harness/exp_runtime_lifecycle.out.txt`。**24 PASS / 0 FAIL**。环境：H100 80GB HBM3、cc 9.0、
torch 2.12.0a0（CUDA 13.2）、DeepGEMM 2.8.1。全部与 UMMA/TMA/WGMMA 无关，是纯 host 资源与进程生命周期。

### 6.1 Part A —— handle 所有权（探针统计真实 create/destroy 次数）

6 组用例，每组都是独立子进程 + `CUDA_VISIBLE_DEVICES=''` + LD_PRELOAD 探针（默认自建 handle 时，
无 CUDA 下真实 `cublasLtCreate` 会失败，探针替换掉它使其可控）：

| 用例 | `DG_USE_PYTORCH_CUBLASLT_HANDLE` | 是否调用 `get_tc_util` | 是否显式 `shutdown` | 期望 create/destroy | 实测 |
|---|---|---|---|---|---|
| `unused` | 0（自建） | 否 | 否 | 0 / 0 | 0 / 0 / clean=1 |
| `owned` | 0（自建） | 是 | 否 | 1 / 1 | 1 / 1 / clean=1 |
| `managed` | 1（托管） | 是 | 否 | 0 / 0 | 0 / 0 / clean=1 |
| `explicit` | 0（自建） | 是 | 是 | 1 / 1 | 1 / 1 / clean=1 |
| `managed_explicit` | 1（托管） | 是 | 是 | 0 / 0 | 0 / 0 / clean=1 |
| `unused_explicit` | 0（自建） | 否 | 是 | 0 / 0 | 0 / 0 / clean=1 |

四件事被坐实：

- `unused` 完全不碰 API → **Runtime 从未构造 → 没有 create**（对应「首次访问才构造」）。
- `owned` → 恰好 **1 次 create、1 次 destroy**，说明 `~Runtime()` 确实在**进程退出前**跑完（clean=1）。
- `managed` → **0 次**：用了 PyTorch 的 handle，不归 DeepGEMM 管。
- `explicit` → 在 `owned` 基础上**显式提前 `shutdown()`**，结果同样 1/1——验证第 5 节推论 3：
  `atexit` 的 release 只是「更早释放」，次数不变。所有用例 `clean` 恒为 1，即退出时 `live_handles==0`。

### 6.2 Part B —— `shutdown()` 终态语义

子进程（托管 handle、无 GPU）输出：

```
B before: 100
B get_tc_util after shutdown: Assertion error (third-party/deep_jit/include/deep_jit/utils/lazy.hpp:27): factory != nullptr: lazy object must be initialized before use
B get_num_sms  after shutdown: Assertion error (third-party/deep_jit/include/deep_jit/utils/lazy.hpp:27): factory != nullptr: lazy object must be initialized before use
B after _C.init: Assertion error (third-party/deep_jit/include/deep_jit/utils/lazy.hpp:27): factory != nullptr: lazy object must be initialized before use
```

- `shutdown()` 前 `get_tc_util()==100`；`shutdown()` 后 `get_tc_util` / `get_num_sms` 均抛
  `lazy.hpp:27` 的 `factory != nullptr`。
- **连续两次 `shutdown()` 无异常**（幂等）。
- 之后调用 `_C.init(pkg_dir)` **仍然报同一错误**——确认 `init` 只重建 `jit`，不重建 `runtime`。

### 6.3 Part C —— 每 stream 一份 workspace 的真实显存账

`cublaslt_gemm_nt((32,64),(48,64))`，用 `torch.cuda.memory_allocated()` 的增量（单位 MiB）看
workspace 是否被持有：

```
held: stream0#1=16.0 MiB, stream0#2=16.0 MiB, stream1=32.0 MiB, out_ok=1
temp: stream0#1=0.0 MiB,  stream0#2=0.0 MiB,  stream1=0.0 MiB,  out_ok=1
```

- **held**（`DG_USE_TEMP_CUBLASLT_WORKSPACE=0`，默认）：首个 stream 用掉 **+16.0 MiB**；同一 stream
  第二次调用**仍是 16.0**（复用同一份）；换到第二个 stream 变成 **32.0**——正好是 **2 × 16 MiB**。
  这就是 `cublaslt_workspaces` 以 `CUDAStream` 为 key 的直接后果（`runtime.hpp:28,72-74`）。
- **temp**（`=1`）：三种情形都**持续占用 0 MiB**——因为每次调用 `torch::empty` 后即弃
  （`runtime.hpp:70`），不进入 map。输出结果仍正确（`out_ok=1`）。

### 6.4 Part D —— `DG_USE_PYTORCH_CUBLASLT_HANDLE` 是「整数」而不是「布尔」

`runtime.hpp:44` 用的是 `deep_jit::get_env<int>(...)> 0`。`get_env<int>` 对整型分支只会走
`std::from_chars`（`env.hpp:46-47,19-26`），因此**「true/false/yes/no」这类布尔写法在这里不被接受，
会直接 `DJ_PANIC`**。探针统计的 create 次数：

| 取值 | 解析结果 | create 次数 | 判定 |
|---|---|---|---|
| `"0"` | 0 → 自建 | 1 | PASS |
| `"1"` | 1 → 托管 | 0 | PASS |
| `"2"` | 2 → 托管 | 0 | PASS |
| `"-1"` | −1 → 自建 | 1 | PASS |
| `"false"` | 非整数 → `DJ_PANIC` | — | PASS（被拒） |
| `"true"` | 非整数 → `DJ_PANIC` | — | PASS（被拒） |
| `"abc"` | 非整数 → `DJ_PANIC` | — | PASS（被拒） |

这与第 48 篇讲的「`DG_*` 两级前缀解析」互补：前缀解析（`env.hpp:56-84`）只决定「查 `DG_X` 还是
`DJ_X`」，而**值的类型**由调用点模板参数决定；这里选了 `int`，所以 `true` 反而非法。
（对照 `DG_JIT_DEBUG` 之类走 `get_env<bool>` 的开关，`true/false/yes/no/非零整数`都收，见 `env.hpp:35-45`。）

### 6.5 Part E —— 交叉验证上游 `tests/test_lazy_init.py`

直接跑上游测试：`Ran 3 tests in 24.6s / OK`。三个用例正是「native shutdown」「真实 cuBLASLt shutdown」
「CUDA workspace 释放」（`test_lazy_init.py:73/116/141`），与我们的 A/B/C 三部分互为独立佐证。

---

## 7. 一张表：`Runtime` 的生命周期

| 阶段 | 触发 | 发生什么 | 依据 |
|---|---|---|---|
| 构造 | 首次 `runtime->...`（任意 `get_tc_util` 等） | 读 `DG_USE_*` 两个 env；自建则 `cublasLtCreate` | `runtime.hpp:40-52`、`lazy.hpp:28-29` |
| 使用 | 每个走 cuBLASLt 的入口 | 取 handle（自建/托管）+ 按 stream 取 workspace | `runtime.hpp:59-76`、`smxx_cublaslt.hpp:74-76` |
| 提前释放 | `deep_gemm.shutdown()` | `runtime = LazyInit(nullptr)` → `~Runtime()`（`cublasLtDestroy` + 析构 workspace map） | `config.hpp:13`、`runtime.hpp:54-57` |
| 正常退出 | Python `atexit` | 自动调用注册的 `shutdown`（在 Py/PyTorch/CUDA 尚存时释放） | `__init__.py:123-125` |
| 退出后 | 任何 `runtime->...` | 抛 `lazy.hpp:27`（除非 `_C.init` 之外另有重建，实际不可恢复） | `lazy.hpp:27`、`jit.hpp:14-32` |

## 8. 小结与边界

- **一句话**：`Runtime` 是「首次访问才构造、可被整体置空销毁」的 lazy singleton；`shutdown()`
  用一行 null-factory 赋值把它**永久标记为未初始化**，从而在 `atexit` 里提前释放 cuBLASLt handle
  与 per-stream workspace，规避原生 cuBLASLt 退出回调与 C++ 静态析构的 use-after-free 竞态。
- **三条可观测事实**（H100 真机）：① 探针证明 handle 恰好 create/destroy 各 1 次（自建）或 0 次（托管）；
  ② 显存账证明 workspace 严格「每 stream 一份、同 stream 复用」（16 / 16 / 32 MiB）；③ `shutdown()`
  幂等、终态、`init` 不可恢复。
- **与目标卡的边界**：本篇**没有任何 SM100 专属内容**，全部可在 H100（sm90）真跑；`Runtime` 的
  handle/workspace/lifecycle 与 GPU 架构无关。它上承第 48 篇（开关面）覆盖了「状态」，
  本篇补齐了同一份 singleton 的「生命周期」，二者合起来才把 DeepGEMM 顶层运行时讲完整。
