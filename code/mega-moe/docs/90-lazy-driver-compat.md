# 两库的「启动前地基」：驱动/运行时符号懒加载、版本闸门与张量视图 helper

> 前面 89 篇把 DeepGEMM / DeepEP 的算子、调度器、通信原语、测试栈几乎拆遍了，但两家仓库
> 各有一层**「什么都还没跑之前」的地基**从没被单独读过：DeepEP 的 `csrc/utils/lazy_driver.hpp`
> （怎么在不链接 `libcuda` 的前提下调用驱动 API）与 `csrc/utils/tensor.hpp`（host 侧的张量视图
> helper）；DeepGEMM 的 `csrc/utils/compatibility.hpp`（三个版本闸门）与它内嵌的 deep_jit 子模块
> `third-party/deep_jit/include/deep_jit/utils/lazy.hpp` / `backend/cuda/driver.hpp` /
> `utils/exception.hpp`（另一套懒加载 + libdw 回溯）。
>
> 这些文件都是**纯 host、架构无关**的，H100（sm90）上能**直接 include 真实头文件**真机跑。
> 本文把它们逐行拆开，并在 H100 上用真实设备驱动验证：两库共 **50 个驱动符号全部可解析**、
> **VMM + 可共享句柄（POSIX fd）跨句柄往返 2 MiB 逐字节一致**、版本闸门**伪造低版本会被
> `static_assert` 当场拦下**。主实验 `src/lazy_driver_compat.cu`（**30 项 ALL PASS**，另加 1 个
> 编译期负样本），原始输出见 `harness/exp_lazy_driver_compat.out.txt`。
>
> 参考坐标：DeepEP `93eb6eb`（v2.5.0）、DeepGEMM `057ca59`（Public release 26/09/30）。

## 1. 为什么需要「懒加载」这一层

DeepGEMM 与 DeepEP 都把 GPU kernel 交给 **DeepJIT 在运行时编译**（DeepGEMM `docs/33`、
DeepEP `docs/86`），安装时只装一个 host 扩展。这带来一个硬约束：

1. **安装/导入期不应依赖 CUDA 驱动**。`import deep_ep` / `import deep_gemm` 可能发生在没有可见
   GPU、甚至没有 `libcuda` 的环境（CI、文档构建机）。若在链接期 `-lcuda`，进程启动就要求驱动
   存在；而 `cuLaunchKernelEx` / `cuTensorMapEncodeTiled` / VMM 这些**只在真正发射 kernel 时才需要**。
2. **符号必须推迟到运行期解析**。这样「模块能 import」与「能不能用 GPU」被解耦：DeepGEMM 的
   顶层 `Runtime` 用 `LazyInit<T>` 把「首次解引用才构造」做成显式契约（`docs/50`），deep_jit 的
   `jit->compile/launch` 也同理。
3. **错误信息要好**。`dlsym` 失败时要把「哪个库、哪个符号、`dlerror()` 说了什么」一次性打出来，
   而不是未定义符号链接错误。

于是两家各写了一套 `dlopen` + `dlsym` 的薄封装。它们看着像「重复造轮子」，但**细节不同**，
第 3 节会逐点对照。

## 2. DeepEP 的 `csrc/utils/lazy_driver.hpp`（60 行）

### 2.1 handle：`get_driver_handle`（`:10-18`）

```cpp
static void* get_driver_handle() {
    static void* handle = nullptr;
    if (handle == nullptr) {
        handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
        EP_HOST_ASSERT(handle != nullptr and "Failed to load CUDA driver `libcuda.so.1`");
    }
    return handle;
}
```

三个细节：

- **`RTLD_LAZY | RTLD_LOCAL`**：`LAZY` 表示未用到的符号不立即解析（反正我们马上要 `dlsym`）；
  `LOCAL` 表示解析出的符号**不进全局符号表**——避免与其它同样加载 `libcuda` 的库（如 PyTorch）
  的符号发生可见性干扰。
- **函数内 `static`**：这是「只初始化一次」的最省事写法。C++11 起函数内 `static` 的初始化是
  线程安全的，但这里用的是**手写的 `if (handle == nullptr)`** 而非初始化器——所以严格说它**不是**
  线程安全的（两个线程同时首次进入可能各自 `dlopen`；不过 `dlopen` 对同一库返回同一 handle 并
  增加引用计数，实际不会泄漏）。
- **`EP_HOST_ASSERT`**（`deep_ep/common/exception.cuh`）：失败抛 `EPException`。

实测里有一个漂亮的旁证：DeepEP 的 `get_driver_handle()` 与 deep_jit 的 `get_cuda_handle()`
**返回完全相同的指针**（`0x7f…6ab0`）——因为 `dlopen` 对同一 soname 会复用已加载的共享对象并只
增加引用计数，与 `RTLD_LOCAL` 无关（Part B）。

### 2.2 为什么需要 `EP_CUDA_STRINGIFY`（`:20-21`）

```cpp
// Macro to expand CUDA API aliases before stringification
#define EP_CUDA_STRINGIFY(name) #name
```

`cuda.h` 里有一堆**版本别名宏**：`#define cuMemGetAddressRange cuMemGetAddressRange_v2`
（本机 CUDA 13.2 `cuda.h:96`）。DeepEP 的符号表里写的正是 `cuMemGetAddressRange_v2`（`:45`），
于是 `&name` 与 `#name` 都落在 `_v2` 上——**`dlsym` 必须查真实导出名 `cuMemGetAddressRange_v2`**。
若写成 `#define STRINGIFY(x) #x` 再传 `cuMemGetAddressRange`，会因宏展开顺序把名字定死为不带
`_v2` 的旧名而 `dlsym` 失败。实测 Part A 里 `cuMemGetAddressRange_v2` **可解析**（53 个符号
一个不缺），这条通路是活的。

### 2.3 核心宏 `DECL_LAZY_CUDA_DRIVER_FUNCTION`（`:23-34`）

```cpp
#define DECL_LAZY_CUDA_DRIVER_FUNCTION(name) \
template <typename... Args> \
static auto lazy_##name(Args&&... args) -> decltype(name(args...)) { \
    using FuncType = decltype(&name); \
    static FuncType func = nullptr; \
    if (func == nullptr) { \
        func = reinterpret_cast<FuncType>(dlsym(get_driver_handle(), EP_CUDA_STRINGIFY(name))); \
        EP_HOST_ASSERT(func != nullptr and "Failed to load CUDA driver API"); \
    } \
    return func(std::forward<decltype(args)>(args)...); \
}
```

逐点解释「为什么这么写」：

1. **返回类型用尾置 `decltype(name(args...))`**，而不是手写 `CUresult`。因为并非所有被包装的
   API 都返回 `CUresult`（例如 `cuMemGetAllocationGranularity` 也是 `CUresult`，但写法上不需要
   假设），尾置返回让编译器**按真实签名推导**，一处改动不会与头文件脱节。
2. **`FuncType = decltype(&name)`**：拿到真实函数指针类型，`dlsym` 返回的 `void*` 用
   `reinterpret_cast` 转回去。这是 C++ 里从 `dlsym` 恢复强类型函数指针的标准做法。
3. **`static FuncType func = nullptr;` + `if`**：函数内 static，首次调用解析、之后直取。
   同样**不是**初始化器线程安全形式（两个线程可能都 `dlsym` 一次，但结果相同、无害）。
4. **完美转发 `std::forward<decltype(args)>(args)...`**：参数按原值类别传下去（虽然驱动 API 都是
   出参指针、值类别不敏感，但约定统一）。
5. **模板 + `static` 函数**：`static` 给内部链接，每个 TU 各自持有一份 `func` 缓存——对 host 扩展
   足够（调用点集中在少数几个 TU）。

被声明的 **23 个符号**（`:36-58`）覆盖 DeepEP 运行期真正需要的四类驱动 API：
模块加载（`cuModuleLoad/Unload/GetFunction`）、发射（`cuLaunchKernelEx`、`cuFuncSetAttribute`）、
VMM 与可共享句柄（`cuMemCreate/Map/SetAccess/Release/...`、`cuMemExport/ImportFromShareableHandle`）、
内存池批次操作（`cuStreamBatchMemOp`），外加错误串（`cuGetErrorName/String`）与设备查询
（`cuCtxGetDevice`、`cuDeviceGet`、`cuDeviceGetAttribute`）。这正是「对称内存 / VMM 分配 /
裸 driver 发射」那条线的全集（对照 `docs/56`、`docs/64`）。

## 3. deep_jit 的 `utils/lazy.hpp`（71 行）——同一件事的另一种写法

### 3.1 `LazyInit<T>`（`:15-38`）

```cpp
template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;
public:
    explicit LazyInit(std::nullptr_t) {}                       // 终态对象：永不构造
    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}
    T* operator->() {
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr) ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        return ptr.get();
    }
    std::shared_ptr<T> get() { (void) operator->(); return ptr; }
};
```

这是 DeepGEMM 顶层 `Runtime` / `jit` 单例的地基（`docs/33` §5、`docs/50`）。两个语义被实测坐实：

- **null factory = 终态**：`LazyInit(nullptr)` 一旦被解引用立刻抛「lazy object must be initialized
  before use」（实测 Part E 抓到这条消息）——`shutdown` 后任何 `runtime->...` 都走这里。
- **工厂只跑一次**：实测 Part E 里带计数的 factory 两次 `live->value` 只触发 **1** 次构造
  （`工厂调用次数=1`）。

### 3.2 两个宏（`:42-54`、`:58-71`）

`DJ_DECL_LAZY_DL_HANDLE` 用一个**立即执行的 lambda**（IIFE）当 `static` 初始化器：

```cpp
inline void* handle_func_name() {
    static void* handle = [] {
        ::dlerror();                                     // 先清空，便于区分本次错误
        void* value = dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
        if (value == nullptr) {
            const char* error = ::dlerror();
            DJ_PANIC("failed to load {}: {}", lib, error == nullptr ? "unknown" : error);
        }
        return value;
    }();
    return handle;
}
```

`DJ_DECL_LAZY_DL_FUNCTION` 同理，但缓存的是**函数指针**：

```cpp
template <typename... Args>
static auto lazy_##name(Args&&... args) {
    static const auto func = [] {
        void* symbol = ::dlsym(handle_func_name(), DJ_STRINGIFY(name));
        if (symbol == nullptr) { const char* error = ::dlerror(); DJ_PANIC(...); }
        return reinterpret_cast<decltype(&name)>(symbol);
    }();
    return func(std::forward<Args>(args)...);
}
```

### 3.3 与 DeepEP 版的三处关键差异

| 维度 | DeepEP `DECL_LAZY_CUDA_DRIVER_FUNCTION` | deep_jit `DJ_DECL_LAZY_DL_FUNCTION` |
|---|---|---|
| 缓存形式 | `static FuncType func = nullptr;` + `if` 判空 | `static const auto func = IIFE();` |
| 线程安全 | 手写判空，非初始化器→理论竞态（无害） | **函数内 static 初始化器→C++11 起保证线程安全** |
| 返回类型 | 尾置 `-> decltype(name(args...))` | `auto` 让编译器推导 |
| 错误信息 | 仅 `EP_HOST_ASSERT` 布尔 | 带 `dlerror()` 文本 + `DJ_PANIC` 格式化 |
| handle 错误 | 仅断言 | `dlerror()` 清空 + 报错串 |

结论：**deep_jit 版更「现代」**——用初始化器拿到语言级线程安全、用 `dlerror` 给出可诊断的错误、
用 `const` 明确不可变。DeepEP 版更「朴素」，但功能等价；两版在本机驱动上都能解析全部符号
（Part A：deep_jit 27 个缺 0、DeepEP 23 个缺 0）。

### 3.4 `backend/cuda/driver.hpp`（63 行）

- **`:8` 的独立闸门**：`DJ_STATIC_ASSERT(CUDA_VERSION >= 12040, "DeepJIT requires CUDA 12.4 or newer")`
  ——**DeepJIT 只要求 12.4**，比 DeepGEMM 本体（12.9）宽。这解释了为什么 deep_jit 能作为独立
  子模块被更广地复用。
- **27 个符号**（`:14-42`）：在 DeepEP 那 23 个之上多了**动态库/kernel 运行时**一批
  （`cuTensorMapEncodeTiled`、`cuLibraryLoadFromFile/Unload/GetKernelCount/EnumerateKernels`、
  `cuKernelGetFunction`）与 VMM 的 `cuMemGetAccess/GetAllocationPropertiesFromHandle/MapArrayAsync`。
  前者正是 SM90/SM100 TMA 描述符编码与 `cuLibrary` 发射的入口（对照 `docs/56`）。
- **`check_cuda_driver`（`:44-57`）的自举性**：它用 `lazy_cuGetErrorName` / `lazy_cuGetErrorString`
  **自己**把 `CUresult` 翻成人话，避免依赖 `cuda_runtime` 的错误串（`cudaGetErrorString` 需要
  runtime，而我们要尽量少链接）。宏 `DJ_CUDA_DRIVER_CHECK(expr)` 把表达式字符串化进消息。

## 4. deep_jit 的 `utils/exception.hpp`：libdw 懒加载与回溯

`Exception` 的构造函数会拼一段 **C++ trace**，靠的是 `detail::DwflApi`（`exception.hpp:22-97`）：

- 结构体里放 **8 个 `dwfl_*` 函数指针**（`dwfl_begin` / `dwfl_linux_proc_report` / `dwfl_report_end`
  / `dwfl_addrmodule` / `dwfl_module_addrname` / `dwfl_module_getsrc` / `dwfl_lineinfo` /
  `dwfl_linux_proc_find_elf`），构造时 `dlopen("libdw.so.1", RTLD_LAZY | RTLD_LOCAL)` 再逐个
  `dlsym`——**libdw 同样是可选依赖**：装不上就是 `valid()==false`，回溯退化成 `dladdr` 的符号名。
- `find_debuginfo` 回调**直接返回 -1**（`:22-27`）：禁用外部 debuginfo 查找，只用进程内已有的
  行号信息（即编译时的 `-g`）。这解释了 `docs/08` 里为什么要 `apt-get install -y libdw-dev`：
  只有 `libdw.so.1`（运行时库）在，编译期头 `elfutils/libdwfl.h` 才过。
- `get_backtrace` 先 `backtrace()` 抓帧、再 `dladdr` 拿模块/符号、再交给 dwfl 补 `file:line`；
  遇到 Python 帧（`Py*`/`_Py*` 符号或 `python3`/`libpython` 模块）就**停止**（`:99-111`），
  不把 Python 解释器栈混进 C++ trace。

实测 Part F：8 个 `dwfl_*` 符号 **8/8 可解析**；构造异常打印出的 trace 带**真实行号**
（编译加 `-g` 后从 `??:0` 变成 `lazy_driver_compat.cu:284`），并正确在 `part_f()` / `main()` /
`libc` / `_start` 处展开。

## 5. DeepGEMM 的 `csrc/utils/compatibility.hpp`（12 行）

```cpp
DG_STATIC_ASSERT(TORCH_VERSION_MAJOR > 2 or (TORCH_VERSION_MAJOR == 2 and TORCH_VERSION_MINOR >= 3),
                 "DeepGEMM requires PyTorch 2.3 or newer");     // :9-10
DG_STATIC_ASSERT(CUDA_VERSION >= 12090, "DeepGEMM requires CUDA Driver API 12.9 or newer");   // :11
DG_STATIC_ASSERT(CUDART_VERSION >= 12090, "DeepGEMM requires CUDA Runtime API 12.9 or newer"); // :12
```

三个「为什么」：

- **为什么是编译期 `static_assert` 而不是运行期检查**：DeepGEMM 本体被 JIT 编译时会把
  `deep_gemm/include` 的源码喂给 nvcc；版本不满足时**在 JIT 编译阶段**就该失败，而不是等到
  kernel 发射。这也与 README「Requirements」一致（CUDA Toolkit 12.9+ / PyTorch 2.3+）。
- **为什么同时卡 Driver 与 Runtime 两个宏**：`CUDA_VERSION`（`cuda.h`）与 `CUDART_VERSION`
  （`cuda_runtime.h`）可以不同——混装 driver 头和 runtime 头是常见坑，两个都卡才安全。
- **12.9 的由来**：SM100（Blackwell）的 `tcgen05` / `cuTensorMapEncodeTiled` 一系列 API 在这个
  版本起稳定（对照 `docs/10` 主体 kernel 用的 UMMA/UTCTP）。注意它与 deep_jit 的 12.4 闸门
  （§3.4）**不是同一层**：deep_jit 可独立于 DeepGEMM 复用。

### 负样本（真实的「闸门会拦」证明）

只测「能编过」不够。实验构造了一个**伪造低版本**的 TU：

```cpp
#include <cuda.h>
#undef CUDA_VERSION
#define CUDA_VERSION 11000       // 伪造 11.0
#include "utils/compatibility.hpp"
```

编译**如期失败**，`nvcc` 报
`compatibility.hpp(11): error: static assertion failed with "DeepGEMM requires CUDA Driver API 12.9 or newer"`。
这条负样本把「闸门不是装饰」钉死了。本机真实值：`TORCH_VERSION=2.12.0`（major=2 minor=12）、
`CUDA_VERSION=13020`、`CUDART_VERSION=13020`，三个闸门全部通过（Part H）。

## 6. DeepEP 的 `csrc/utils/tensor.hpp`（27 行）与 locality JIT wrapper

`tensor.hpp` 提供两个张量视图 helper：

```cpp
using tensor_list_t = std::vector<std::optional<torch::Tensor>>;        // :12
template <int kNumDims>
static auto get_shape(const torch::Tensor& t) {                         // :14-20
    EP_HOST_ASSERT(t.dim() == kNumDims);
    return [&t] <size_t... Is> (std::index_sequence<Is...>) {
        return std::make_tuple(static_cast<int>(t.sizes()[Is])...);
    }(std::make_index_sequence<kNumDims>());
}
template <typename dtype_t = void>
static dtype_t* get_data_ptr(const std::optional<torch::Tensor>& t) {   // :22-25
    return t.has_value() ? t->data_ptr<dtype_t>() : nullptr;
}
```

- **`get_shape<kNumDims>` 用 `std::index_sequence` 把编译期维度展开成 `std::tuple`**：调用点写
  `auto [m, n] = get_shape<2>(t)`，维度数写死在模板里，运行期只做一次 `dim()` 断言——这样
  「按固定维数解包」在类型系统里就成立，不需要在 host 描述符构造处到处 `t.size(0)`/`t.size(1)`。
- **`get_data_ptr` 把 `std::optional<Tensor>` 统一成裸指针**：可选张量（如 DeepEP 里 SF 只在
  FP8 路径出现，否则传 `nullopt`）在 host→device 描述符转换处一律走它，`nullopt → nullptr`
  让 device 端「有 SF / 无 SF」变成同一种「指针是否为空」的判据（与 Mega MoE 的
  `with_sf` 分支同构）。

实测 Part G：`get_shape<2>(3×5)` 返回 `(3,5)`、`get_data_ptr(nullopt)==nullptr`、
`get_data_ptr(present)==data_ptr`、`get_shape<3>` 传 2D 张量抛 `EP_HOST_ASSERT`——四条全过。

另外补上 DeepGEMM 侧**最后一个未被任何 docs 引用的源文件**
`csrc/jit_kernels/impls/sm100_locality_domain.hpp`（41 行）：它是第 34 篇（P24-1）那颗
pointer-chase 延迟探针的 **host 启动包装层**。三个常量（`:13-15`）：
`kNumLocalityDomainProbeChunkBytes = 4096`、`kNumLocalityDomainProbeLineBytes = 128`、
`kNumLocalityDomainProbeHops = 4096/128 = 32`——即**每 4 KiB chunk 沿 128 B 行跳 32 跳**，
保证每一次计时跳都是一次 HBM fill（注释 `:12`）。wrapper 用 `std::format` 把
`(hops, buf.size(1), chunk_bytes)` 编进 JIT 源码（`:20-28`），`grid=num_sms`、`block=(32,1,1)`
（`:31-38`）——这正是 `docs/11`/`docs/34` 里「按域分任务计数」的探针发射端。实测 Part H 校验
`4096/128==32`。

## 7. 实测汇总（H100 / sm90，纯 host）

命令行：`bash harness/exp_lazy_driver_compat.sh`（`nvcc -arch=sm_90a` 直接 include 两库真实
头文件 + torch）。原始输出 `harness/exp_lazy_driver_compat.out.txt`。

| 组 | 验证内容 | 结果 |
|---|---|---|
| A | `dlopen("libcuda.so.1")`；两库符号清单可解析性 | **deep_jit 27 缺 0 / DeepEP 23 缺 0** |
| B | 两库各自 `dlopen` 的 handle 身份 | 同一指针 `0x7f…6ab0`；幂等 |
| C | 懒加载 wrapper vs 直接链接 | `lazy_cuDeviceGet`==`cuDeviceGet`（dev 0）；属性都=9（sm90） |
| D | VMM + POSIX fd 可共享句柄往返（全经懒加载） | gran=2 MiB；**导出→导入→跨句柄 2 MiB mismatch=0**；`cuMemGetAddressRange_v2` base/range 正确 |
| E | `deep_jit::LazyInit<T>` | 工厂只跑 1 次；null factory 解引用抛异常 |
| F | libdw 懒加载 + 异常回溯 | 8/8 `dwfl_*` 解析；trace 带**真实行号** |
| G | `tensor.hpp` helper | `get_shape` / `get_data_ptr` / 维度断言 全对 |
| H | 版本闸门 | 真实 `torch 2.12 / CUDA 13020` 通过；locality hops=32 |
| 负样本 | 伪造 `CUDA_VERSION=11000` | **`static_assert` 如期拦下**（`compatibility.hpp:11`） |

**合计 30 PASS / 0 FAIL + 1 编译期负样本。**

## 8. 与专题的关系与边界

- 这一层是**所有 Mega MoE / 反向 kernel 的「发射前地基」**：没有它，`import deep_gemm` /
  `import deep_ep` 都要在链接期绑定 `libcuda`，也就没有「JIT 在运行时按当前设备编译」的自由
  （`docs/33`、`docs/86`）。DeepEP 的对称内存分配（`docs/56`）走的正是本文 Part D 实测的那组
  VMM + 可共享句柄 API。
- 本文是**纯 host、架构无关**的：没有任何 UMMA/TMEM/TMA/NCCL GIN。因此 H100 上能跑**真实
  头文件 + 真实驱动**，不是仿真。
- **边界**：Mega MoE 主 kernel（SM100）与 DeepEP 的 NCCL GIN 设备 API 仍受本机环境限制
  （NCCL 2.29.7 < 2.32.3，见 ROADMAP「阻塞」），与本篇无关。DeepEP 侧 `lazy_driver.hpp` 列出的
  23 个符号在**本机 profile 570.211.01** 上全部可解析——说明这一层不是「阻塞」的成因；阻塞在
  更上层的 NCCL 设备通信。
- 链接：DeepGEMM（https://github.com/deepseek-ai/DeepGEMM）、DeepEP（https://github.com/deepseek-ai/DeepEP）、
  DeepEP V2.5 README「Requirements」、DeepGEMM README「Requirements」。
