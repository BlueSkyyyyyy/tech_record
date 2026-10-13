# 126 · 所有 kernel 共用的「错误 / 断言层」：`EPException` / `DGException` 与五个 check 宏

> 本专题到第 125 篇，把 DeepGEMM / DeepEP 的每一个 kernel、host 入口、调度器、布局、测试基座
> 都逐一拆过。但有一层**每一行 kernel 发射前/后都要踩、却从没被单独读过**的地板：**错误与断言层**。
> 你翻任意一篇的源码摘录——`k_grouped_fp8_gemm_nt_contiguous` 入口的 `DG_HOST_ASSERT`、
> dispatch 里的 `EP_HOST_ASSERT`、JIT 发射后的 `CUDA_RUNTIME_CHECK`、NCCL 调用后的 `NCCL_CHECK`——
> 它们都在调这一层，但这一层本身是什么、抛出的异常长什么样、怎么变成 Python 的报错，一直当空气。
>
> 本篇把这层地基单独拆透，并做一个**跨库对照**：DeepGEMM 与 DeepEP 各有一套几乎对称、又在
> **五个关键处不同**的实现。**这层是纯 host / C++ + 架构无关**：在 H100(sm90) 上**直接
> `#include` 两库真实头文件**、编译运行真实代码，再通过一个真实 pybind11 模块验证
> 「C++ 异常 → Python 报错」的翻译路径。
>
> 目标卡代号敏感：全文用「目标卡」指代。参考版本：`~/github/DeepGEMM` commit `057ca59`
> （v2.8.1）、`~/github/DeepEP` commit `93eb6eb`（v2.5.0）。

实验：`src/exception_layer.cu` + `src/exception_bridge.cpp` + `harness/exp_ep_dg_exception.{sh,py}`，
原始输出 `harness/exp_ep_dg_exception.out.txt`。**H100(sm90) 真机 36 PASS / 0 FAIL**。

---

## 1. 为什么单开一篇：把所有 kernel 绑在同一条「错误咽喉」上

DeepGEMM 全仓（`csrc/` + `deep_gemm/include/`）里 `DG_*` 断言/检查宏的用例有 **1082 处**，
DeepEP 有 **444 处**（普查见 §6）。它们不是可以忽略的调试代码——**它们定义了每个 kernel 的
前置契约**：形状对不对、对齐到没到、arch 匹不匹配、NCCL 版本够不够，全靠这一层在 kernel
真正发射**之前**拦下来。理解了这一层，前面那些「`gemm.hpp:369` 在 sm90 拒绝」
「`k_alignment` 契约」「`check_k_grouped_args`」才知道**报错是怎么冒出来的、冒出来长什么样**。

而且，**它决定了这套库怎么「对用户说话」**：两库都是 pybind11 扩展，所有 C++ 侧的错误最终都要
变成 Python 侧用户能看到的异常。这一层就是那个转换器（§5）。

三个动机：

1. **可证伪**：把「抛什么类型、消息格式是什么、什么条件下抛」变成可复现的实验，而不是读注释猜。
2. **对照**：两库是姊妹项目、写法高度对称，但**五个细节不同**——这些不同不是笔误，是各自
   host/device 编译模型差异的产物（§4）。
3. **闭环**：顺带抓到一个**真实陷阱**——DeepGEMM 的 `DG_UNIFIED_ASSERT` 在 nvcc 编译的 host
   代码里会展开成 `asm("trap;")`、汇编期直接报 `no such instruction: trap`（§4.3）。

---

## 2. 源码坐标

| 文件 | 行数 | sha256[:12] | 内容 |
|---|---:|---|---|
| `DeepEP/deep_ep/include/deep_ep/common/exception.cuh` | 94 | `489616d28c7e` | `EPException` + EP_* 宏 + 三个 `*_CHECK` |
| `DeepGEMM/deep_gemm/include/deep_gemm/common/exception.cuh` | 43 | `ab6b2bd75c06` | **device** 侧 `DG_*_ASSERT` / `DG_UNIFIED_ASSERT` |
| `DeepGEMM/csrc/utils/exception.hpp` | 67 | `162f410d0493` | **host** 侧 `DGException` + `DG_HOST_ASSERT` / `DG_CUDA_RUNTIME_CHECK` / `DG_CUBLASLT_CHECK` |
| `DeepGEMM/csrc/utils/compatibility.hpp` | 12 | `dedda83af6e4` | 三个 `DG_STATIC_ASSERT`（版本闸门） |
| `DeepGEMM/deep_gemm/include/deep_gemm/common/compile.cuh` | 18 | `a13373053608` | `DG_IN_CUDA_COMPILATION` 的定义处（`:5-7`） |

> **第一个结构性差异**：DeepEP 把 **host 与 device 的异常/断言全放进同一个头文件**
> （`common/exception.cuh`，94 行，一个 `EPException` 类 + 全部宏）；DeepGEMM **拆成两个文件**
> ——`include/deep_gemm/common/exception.cuh`（device，43 行）与 `csrc/utils/exception.hpp`
> （host，67 行，`DGException` 在这里）。这个拆分是 §4 一切差异的源头。

---

## 3. 两个异常类：对称的骨架，不同的一个词

两库的异常类**逐行对称**，都是「构造时把 `name/file/line/error` 拼成一个 `std::string`、
`what()` 返回它」，且都**只继承 `std::exception`**（没有额外虚函数）。

`EPException`（`common/exception.cuh:11-23`）：

```cpp
class EPException : public std::exception {
    std::string message = {};
public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        std::stringstream ss;
        ss << name << " exception (" << file << ":" << line << "): " << error;   // :18
        message = ss.str();
    }
    const char* what() const noexcept override { return message.c_str(); }        // :22
};
```

`DGException`（`csrc/utils/exception.hpp:13-24`）：

```cpp
class DGException final : public std::exception {
    std::string message = {};
public:
    explicit DGException(const char *name, const char* file, const int line, const std::string& error) {
        message = std::string(name) + " error (" + file + ":" + std::to_string(line) + "): " + error;  // :18
    }
    const char *what() const noexcept override { return message.c_str(); }        // :21-23
};
```

**唯一实质差别是 `exception` vs `error` 那个词**。其余是风格差：DeepEP 用 `stringstream`、
DeepGEMM 用 `std::string +` 拼接；DeepGEMM 的类多了 `final`。

真机核对（`exp_ep_dg_exception.out.txt:19,40`）：

```
what() = `Assertion exception (foo.cuh:42): cond failed`      # EP
what() = `Assertion error (bar.cuh:7): nope`                  # DG
```

### 3.1 一个额外的构造宏：`EPExceptionWithLineInfo`

DeepEP 还多一个便利宏（`common/exception.cuh:25`）：

```cpp
#define EPExceptionWithLineInfo(name, message) EPException(name, __FILE__, __LINE__, message)
```

它把 `__FILE__`/`__LINE__` 自动填进去——因为有些错误发生在**运行期**（不是静态契约），必须把
「出错的确切位置」带出去。全仓只有 **2 处**用它，都是「CPU 同步超时」这种运行期异常，例如
`csrc/buffers/ep.hpp:661`：

```cpp
const auto now = std::chrono::high_resolution_clock::now();
if (std::chrono::duration_cast<std::chrono::seconds>(now - start_cpu_time).count() > context->num_cpu_timeout_secs)
    throw EPExceptionWithLineInfo("Dispatch CPU wait", get_buffer_info());
```

真机核对：抛出的消息是
`Dispatch CPU wait exception (.../exception_layer.cu:74): buffer#0`——**name 是业务短语、
文件行号是 throw 点本身**（`exp_ep_dg_exception.out.txt:22`）。DeepGEMM 没有对应物。

---

## 4. 宏清单：五个真实差异

两库的宏几乎一一对应，但有 **5 处不同**。下表按「DeepEP 有 / DeepGEMM 有 / 差异」。

### 4.1 对照表

| 语义 | DeepEP | DeepGEMM | 差异 |
|---|---|---|---|
| host 断言 | `EP_HOST_ASSERT(cond)` `:27-34` | `DG_HOST_ASSERT(cond)` `exception.hpp:30-37` | 无（消息词不同） |
| host 不可达 | `EP_HOST_UNREACHABLE(msg)` `:36-38` | `DG_HOST_UNREACHABLE(msg)` `exception.hpp:39-41` | 无 |
| device 断言 | `EP_DEVICE_ASSERT(cond)` `:40-48` | `DG_DEVICE_ASSERT(cond)` `:15-23` | **DG 多一个 `DG_TRAP_ONLY_DEVICE_ASSERT` `:25-31`** |
| 编译期断言 | `EP_STATIC_ASSERT` `:7-9` | `DG_STATIC_ASSERT`（device `:33-35`、host `:26-28`） | **DG 在两个头重复定义、`#ifndef` 保护** |
| host/device 二选一 | `EP_UNIFIED_ASSERT` `:50-56` | `DG_UNIFIED_ASSERT` `:37-43` | **分派键不同：`__CUDA_ARCH__` vs `DG_IN_CUDA_COMPILATION`（§4.3）** |
| CUDA runtime 检查 | `CUDA_RUNTIME_CHECK` `:58-68` | `DG_CUDA_RUNTIME_CHECK` `exception.hpp:43-53` | 无 |
| CUDA driver 检查 | `CUDA_DRIVER_CHECK` `:70-82` | — | **DeepEP 独有** |
| NCCL 检查 | `NCCL_CHECK` `:84-94` | — | **DeepEP 独有** |
| cuBLASLt 检查 | — | `DG_CUBLASLT_CHECK` `exception.hpp:55-65` | **DeepGEMM 独有** |

### 4.2 为什么各自的 `*_CHECK` 不一样

- **DeepEP 有 `CUDA_DRIVER_CHECK` + `NCCL_CHECK`**：它的重活是**通信**——到处 `dlsym` CUDA 驱动
  API（VMM 分配对称内存，见第 66 篇）和 NCCL 设备 API（GIN，见第 63 篇）。它的
  `CUDA_DRIVER_CHECK`（`:70-82`）**不是**直接用 `cuGetErrorName`，而是走
  `lazy_cuGetErrorName` / `lazy_cuGetErrorString`（`csrc/utils/lazy_driver.hpp:36-37` 声明的
  懒加载包装）——所以这个宏**必须在 `namespace deep_ep` 内使用**，否则找不到 `lazy_*`。

  > 真机核对：`CUDA_DRIVER_CHECK(lazy_cuDeviceGet(&dev, 999))` 抛出
  > `CUDA driver exception (...:46): 101 (CUDA_ERROR_INVALID_DEVICE, invalid device ordinal)`
  > （`exp_ep_dg_exception.out.txt:33`）——消息里同时有**数值码 + 名称 + 描述**，证明
  > 懒加载的两个 `lazy_cuGetError*` 真的被调用到了。

  `NCCL_CHECK`（`:84-94`）的消息用 `ncclGetLastError(nullptr)`。真机核对：
  `NCCL_CHECK(ncclGetVersion(nullptr))` 抛 `NCCL exception (...): 4 (invalid argument ...)`
  （`out.txt:35-36`）。

- **DeepGEMM 有 `DG_CUBLASLT_CHECK`**：它有一条 cuBLASLt 回退路径（第 27 篇）——当不用自家
  JIT kernel 时走 cuBLASLt，于是需要检查 `cublasStatus_t`。真机核对
  `DG_CUBLASLT_CHECK(cublasLtCreate(nullptr))` 抛 `cuBLASLt error (...): 7 (an unsupported
  value or parameter ...)`（`out.txt:47`）。

- **`DG_TRAP_ONLY_DEVICE_ASSERT`（DeepGEMM 独有，`exception.cuh:25-31`）**：和 `DG_DEVICE_ASSERT`
  几乎一样，**唯一区别是不 `printf`、直接 `asm("trap;")`**。用途是「**在不能打印的场景**（例如
  printf 本身可能不安全/太吵）只要一个硬停」。全仓 23 处。DeepEP 没有对偶。

### 4.3 第三个差异：`*_UNIFIED_ASSERT` 的分派键（真实陷阱）

这是本篇最有价值的一处。两库都提供「**同一个宏在 host 段抛异常、在 device 段 trap**」的二选一：

DeepEP（`common/exception.cuh:50-56`）：

```cpp
#ifndef EP_UNIFIED_ASSERT
#ifdef __CUDA_ARCH__
#define EP_UNIFIED_ASSERT(cond) EP_DEVICE_ASSERT(cond)
#else
#define EP_UNIFIED_ASSERT(cond) EP_HOST_ASSERT(cond)
#endif
#endif
```

DeepGEMM（`common/exception.cuh:37-43`）：

```cpp
#ifndef DG_UNIFIED_ASSERT
#ifdef DG_IN_CUDA_COMPILATION
#define DG_UNIFIED_ASSERT(cond) DG_DEVICE_ASSERT(cond)
#else
#define DG_UNIFIED_ASSERT(cond) DG_HOST_ASSERT(cond)
#endif
#endif
```

**分派键完全不同**：

- DeepEP 用 `__CUDA_ARCH__`——这个宏**只在 nvcc 的 device 编译趟（pass）里定义**，host 趟不定义。
  这恰好是「这段代码是编给 GPU 还是 CPU」的**精确**判据。
- DeepGEMM 用 `DG_IN_CUDA_COMPILATION`，它定义在 `compile.cuh:5-7`：

  ```cpp
  #if defined(__NVCC__) or (defined(__clang__) and defined(__CUDA__)) or defined(__CUDACC_RTC__) or defined(__CLION_IDE__)
  #define DG_IN_CUDA_COMPILATION
  #endif
  ```

  `__NVCC__` 在 **nvcc 的整个编译过程（host 趟 + device 趟）都定义**。所以在一个 `.cu` 文件的
  **host 代码**里用 `DG_UNIFIED_ASSERT`，会**错误地**展开成 `DG_DEVICE_ASSERT` → `asm("trap;")`
  → 汇编期报错。

真机证据（`out.txt:109-112`）：把

```cpp
#include <cstdio>
#include <cuda_runtime.h>
#include <deep_gemm/common/exception.cuh>
int main() { DG_UNIFIED_ASSERT(1 == 2); return 0; }
```

交给 `nvcc -arch=sm_90 -c` 编译，得到：

```
/tmp/_p116_neg_unified.cu: Assembler messages:
/tmp/_p116_neg_unified.cu:4: Error: no such instruction: `trap'
```

对照：把 `EP_UNIFIED_ASSERT` 放同样的 host 段交给 nvcc，**正常编译通过**（`__CUDA_ARCH__` 未定义
→ 落到 `EP_HOST_ASSERT`）。真机 `out.txt:112`。

**为什么 DeepGEMM 线上没炸？** 因为它在 host 侧调用 `DG_UNIFIED_ASSERT` 的地方——例如
`layout/mega_moe.cuh:119-125` 的 `Workspace` 构造函数（`CUTLASS_HOST_DEVICE`，即
`__host__ __device__`）——分两种调用方式：

1. **device 侧**：由 nvcc 编 kernel 时展开，`DG_IN_CUDA_COMPILATION` 定义 → `DG_DEVICE_ASSERT`
   ✅ 正确；
2. **host 侧**：由**普通 C++ 编译器**（gcc/g++）编 `csrc/python_api.cpp` 时展开，此时
   `DG_IN_CUDA_COMPILATION` **不定义** → `DG_HOST_ASSERT` ✅ 正确。

两者都对，是因为 **DeepGEMM 的构建把 host TU 交给 host 编译器、不交给 nvcc**
（`setup.py:32` 的 `sources = ['csrc/python_api.cpp']`，`.cpp` 走 g++）。所以这个陷阱是
**「潜在」的**：只要有人在 `.cu` 的 host 段用 `DG_UNIFIED_ASSERT`（本机实验正是如此），就炸。
DeepEP 的 `__CUDA_ARCH__` 判据**对此免疫**——同样是 `.cu` host 趟，它照样落到 `EP_HOST_ASSERT`。

> 附一个同源的「脆弱点」：`layout/mega_moe.cuh` 只 include 了 **device** 的
> `deep_gemm/common/exception.cuh`，它出现的 `DG_UNIFIED_ASSERT` 在 gcc 下要落到
> `DG_HOST_ASSERT`，而这个宏**定义在 `csrc/utils/exception.hpp`**（device 头里没有）。
> 也就是说 `layout/mega_moe.cuh` 能不能用 gcc 编，取决于**之前有没有先把 host 异常头引进来**
> ——实测把 `csrc/apis/mega_moe.hpp` 单独喂给 g++ 会报
> `deep_gemm/common/exception.cuh:41: 'DG_HOST_ASSERT' was not declared`；而官方
> `python_api.cpp` 因为**先把 `apis/config.hpp`（→ `heuristics/runtime.hpp:6` → `utils/exception.hpp`）
> 引进来**，到 `apis/mega_moe.hpp` 时宏已就绪，故能编过。**顺序敏感**，这是拆分的代价。

### 4.4 `*_STATIC_ASSERT` 的重复定义

DeepGEMM 在 device（`exception.cuh:33-35`）和 host（`exception.hpp:26-28`）**各定义了一遍**
`DG_STATIC_ASSERT`，都用 `#ifndef` 保护——因为两个头可能被同一个 TU 同时包含。DeepEP 只在
device 头定义一次。真机：`EP_STATIC_ASSERT(sizeof(int) >= 4, ...)` 正常编译通过
（`out.txt:37-38`）。

---

## 5. 从 C++ 异常到 Python 报错：pybind11 的默认翻译器

两库都是 pybind11 扩展，但**都没有一行 `py::register_exception`**（全仓 grep 为空）。那
`EPException`/`DGException` 抛出去在 Python 里长什么样？

答案：pybind11 有一个**默认异常翻译器**——任何派生自 `std::exception` 的异常，若没有专门注册，
会被翻译成 Python 的 **`RuntimeError`**，`what()` 文本成为异常消息。

为了验证这不是空谈，本实验编译了一个**真实 pybind11 模块**（`src/exception_bridge.cpp`，
直接 `#include` 两库真实头文件），把 5 个真实宏包成函数抛进 Python：

| Python 调用 | 真机结果（`out.txt:71-75`） |
|---|---|
| `throw_ep_assert()` | `RuntimeError: Assertion exception (…exception_bridge.cpp:14): false && "ep assert from bridge"` |
| `throw_ep_line()` | `RuntimeError: Bridge exception (…:15): line-info payload` |
| `throw_ep_unreach()` | `RuntimeError: Assertion exception (…:16): ep unreachable from bridge` |
| `throw_dg_assert()` | `RuntimeError: Assertion error (…:17): false && "dg assert from bridge"` |
| `throw_dg_unreach()` | `RuntimeError: Assertion error (…:18): dg unreachable from bridge` |

五点都要注意：

1. **类型是 `RuntimeError`**，不是自定义类型——因为没注册。用户 `except RuntimeError` 或
   `except Exception` 都能接住，但**无法按异常类型区分「契约错误」和「运行期错误」**。
2. **消息里保留了文件:行号**——这正是这套设计的意义：Python 用户拿到报错能**直接跳回 C++ 源码行**。
   例如用户误传形状，看到 `... gemm.hpp:369 ...` 就知道去 `csrc/apis/gemm.hpp:369` 看那条契约。
3. **测试通过 pybind11 默认路径**，与官方 `python_api.cpp` 完全同构（`python_api.cpp` 也没注册异常）。
4. pybind11 对 `std::exception` 的默认翻译有文档依据（见文末链接）。
5. 因为两库的异常都**没有**做「异常类型 → 错误码」的映射，DeepGEMM/DeepEP 的错误在 Python 层
   **一律是 `RuntimeError`**。

---

## 6. 用法普查：这一层到底有多「地基」

以「宏 token 在 `csrc/` + `<lib>/include/` 下的出现次数」做统计（含 `#define` 与用例），
真机结果（`out.txt:80-104`）：

| DeepEP 宏 | 次数 | | DeepGEMM 宏 | 次数 |
|---|---:|---|---|---:|
| `EP_HOST_ASSERT` | 286 | | `DG_HOST_ASSERT` | 605 |
| `EP_STATIC_ASSERT` | 133 | | `DG_STATIC_ASSERT` | 370 |
| `CUDA_DRIVER_CHECK` | 32 | | `DG_HOST_UNREACHABLE` | 36 |
| `NCCL_CHECK` | 20 | | `DG_DEVICE_ASSERT` | 35 |
| `CUDA_RUNTIME_CHECK` | 16 | | `DG_CUBLASLT_CHECK` | 27 |
| `EP_DEVICE_ASSERT` | 13 | | `DG_TRAP_ONLY_DEVICE_ASSERT` | 23 |
| `EP_UNIFIED_ASSERT` | 10 | | `DG_UNIFIED_ASSERT` | 13 |
| `EPException` | 8 | | `DGException` | 6 |
| `EP_HOST_UNREACHABLE` | 2 | | `DG_CUDA_RUNTIME_CHECK` | 4 |
| `EPExceptionWithLineInfo` | 2 | | | |

**断言宏合计 DeepGEMM 1082 / DeepEP 444 处**（不算 `*_CHECK`）。这就是「地板」的体量：
每一处都是一条被执行的契约，且 DeepGEMM 的 host 断言数（605）远超 device（35）——说明**绝大多数
校验发生在 kernel 发射之前**。

`EP_UNIFIED_ASSERT` 的 10 处用例全在 **`__host__ __device__` 函数**里（`layout/ep/token.cuh:29/131/140`、
`layout/ep/workspace.cuh:43-46`），也正是这个宏存在的理由：同一段布局算术，host 构造时要校验、
device 用时也要校验。DeepGEMM 对应的 `DG_UNIFIED_ASSERT` 13 处集中在 `layout/mega_moe.cuh`。

---

## 7. 设备端 trap：断言失败怎么冒出来

`EP_DEVICE_ASSERT`（`common/exception.cuh:40-48`）与 `DG_DEVICE_ASSERT`
（`exception.cuh:15-23`）**逐字相同**：

```cpp
#define DG_DEVICE_ASSERT(cond) \
do { \
    if (not (cond)) { \
        printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
        asm("trap;"); \
    } \
} while (0)
```

即「先 `printf` 出文件行号与失败条件，再 `asm("trap;")` 硬停」。真机把每个宏各放进一个 kernel、
**每个 kernel 独立进程**运行（避免 trap 的 sticky 错误互相污染），`cudaDeviceSynchronize()` 的
返回是（`out.txt:56-65`）：

```
DEVICE EP_DEVICE_ASSERT   sync=cudaErrorLaunchFailure (unspecified launch failure)
DEVICE EP_UNIFIED_ASSERT  sync=cudaErrorLaunchFailure (unspecified launch failure)
DEVICE DG_DEVICE_ASSERT   sync=cudaErrorLaunchFailure (unspecified launch failure)
DEVICE DG_TRAP_ONLY       sync=cudaErrorLaunchFailure (unspecified launch failure)
DEVICE DG_UNIFIED_ASSERT  sync=cudaErrorLaunchFailure (unspecified launch failure)
```

五个宏**全部**把 device 打挂（同步返回非 success），其中 `EP_UNIFIED_ASSERT`/`DG_UNIFIED_ASSERT`
在 device 趟确实落到了各自的 device 断言（坐实 §4.3 的分派正确），`DG_TRAP_ONLY` 也 trap
（它只是省了 `printf`）。观察到的错误名是 `cudaErrorLaunchFailure`——在这台驱动/架构上，
`asm("trap;")` 的失败以后续同步的「unspecified launch failure」形式浮现（而不是 `device-side
assert triggered`）；**无论哪个名字，结论一致：device 契约被违反 → 整个 launch 作废**。

---

## 8. H100 实测汇总

驱动 `harness/exp_ep_dg_exception.sh`（nvcc 编 `src/exception_layer.cu` → 运行多模式）+
`harness/exp_ep_dg_exception.py`（跑二进制、编译 pybind 桥、宏普查、负向编译）。真机 **36 PASS / 0 FAIL**：

- **A host 模式 18/18**：异常消息格式、类层次、`EP_HOST_ASSERT`/`EP_HOST_UNREACHABLE`、
  `EP_UNIFIED_ASSERT` host 分派、`CUDA_RUNTIME_CHECK`、`CUDA_DRIVER_CHECK`、`NCCL_CHECK`、
  `EP_STATIC_ASSERT`、`DGException` 系、`DG_CUDA_RUNTIME_CHECK`/`DG_CUBLASLT_CHECK`。
- **A-dev 5/5**：五个 device 断言宏各自触发 trap（独立进程）。
- **B 5/5**：真实 pybind11 模块把两库异常翻成 `RuntimeError`、消息含文件行号。
- **C 4/4**：宏普查（断言宏 DeepEP 444 / DeepGEMM 1082）。
- **D 2/2**：`DG_UNIFIED_ASSERT` host 段负向编译失败（`no such instruction: trap`）、
  `EP_UNIFIED_ASSERT` host 段正常编译。

---

## 9. 边界与结论

- **能跑 / 能核**：整层是纯 host C++ + 架构无关。异常类、五个 check 宏、pybind 翻译、
  设备 trap（sm90 上 `asm("trap;")` 合法）、负向编译，全部在 H100(sm90) 真机核对。
- **仅分析**：这层里没有 SM100 专属代码；但它的**消费者**（mega moe 主 kernel、tcgen05、
  GIN 设备 API）仍是目标卡/需要 NCCL≥2.32 的部分（见 ROADMAP「阻塞」），那些报错路径本身
  在本机跑不了——本篇只保证**报错机制**本身对得上。
- **结论**：两库的「错误 / 断言层」是一套**高度对称、又在五处分叉**的实现：
  ① 单文件（DeepEP）vs host/device 拆两文件（DeepGEMM）；② DeepEP 独有
  `CUDA_DRIVER_CHECK`/`NCCL_CHECK`/`EPExceptionWithLineInfo`，DeepGEMM 独有
  `DG_CUBLASLT_CHECK`/`DG_TRAP_ONLY_DEVICE_ASSERT`；③ 二选一分派键
  `__CUDA_ARCH__`（健壮）vs `DG_IN_CUDA_COMPILATION`（nvcc 全程定义，**在 `.cu` host 段会炸**）；
  ④ 不注册异常类型、统一翻成 Python `RuntimeError`，靠**消息里的文件行号**实现「Python 报错
  直跳 C++ 源码」；⑤ 断言宏体量 DeepGEMM 1082 / DeepEP 444，且**绝大多数是 host 侧前置契约**。

一句话：**这套库的「可信度」一半由 kernel 决定、另一半由这层地板决定**——它在 kernel 发射前
用 1082 / 444 处断言把「不该跑的输入」拦在门外，并把失败翻译成带源码行号的 Python `RuntimeError`。

---

### 参考链接

- pybind11 异常翻译（`std::exception` → `RuntimeError`，以及如何注册自定义类型）：
  <https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html>
- DeepGEMM（`csrc/utils/exception.hpp` / `deep_gemm/include/deep_gemm/common/exception.cuh`）：
  <https://github.com/deepseek-ai/DeepGEMM>
- DeepEP（`deep_ep/include/deep_ep/common/exception.cuh`）：
  <https://github.com/deepseek-ai/DeepEP>
