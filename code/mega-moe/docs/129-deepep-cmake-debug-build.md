# 129 · DeepEP 的 CMake 调试构建链：`_C` + 两个 host runtime 静态库 + IDE 索引 TU

> 第 124 篇拆了「任何一行设备代码跑起来之前」的**构建 / 打包 / 安装期地基**——`find_pkgs.py` 与
> 两库的 `setup.py`，并一句话带过「`CMakeLists.txt` 首行自称 `only for debugging`、不参与 `pip install`」。
> 第 78 篇扫 DeepGEMM 的 profiling / 工具链时又顺带提过它的 `csrc/indexing/main.cu` 与 CMake。
> 但 **DeepEP 的 CMake 调试链本身从没被单独读过**：它到底编出什么、和 `setup.py` 差在哪、
> 那几行 `nvcc` 诊断 flag 落在谁身上、为什么仓库里到处是 `cmake-build-*` 这个名字。
>
> 本篇把这 46 行 `CMakeLists.txt`（+ 三行子 `CMakeLists.txt`）逐段讲透，并**在 H100 真机（纯构建 /
> 文件解析、架构无关）真跑一遍**：configure 通过、`_C` 扩展与两个静态库编出来、而 IDE 索引 TU
> 撞上一个**上游真实编译错误**。这条链和本机「运行期阻塞」（路线图「阻塞」）分工不同——
> 它恰好证明**编译期没问题，问题在运行期加载哪份 NCCL**。
>
> 目标源码：`~/github/DeepEP@93eb6eb`（`__version__ 2.5.0`，`CMakeLists.txt` sha256[:16]
> `fc96cd665d654507`）、`~/github/DeepGEMM@057ca59`（`CMakeLists.txt` sha256[:16] `8f05c0443af4daf5`）。

## 1. 为什么一个仓库要两套构建

DeepEP 有两套完全独立的构建入口，服务**两个不同目的**：

| | `setup.py`（真装包） | `CMakeLists.txt`（调试） |
|---|---|---|
| 谁在用 | `pip install` / `build.sh` / `install.sh` | IDE（CLion/VS Code 的 CMake 插件） |
| 产物 | `deep_ep._C` 扩展 wheel | `_C.so` + 两个静态库 + 索引 TU 目标 |
| 自述 | `DeepEP/setup.py:126-139` 的 `CUDAExtension` | `CMakeLists.txt:1`「only for debugging」 |
| 编什么 | 只编 **3 个 host `.cpp`**（`DeepEP/setup.py:81-83`） | 编 host `.cpp` **和** 一个 `.cu` 索引 TU |

两套的**源码集合其实一致**——三个 host 翻译单元：`csrc/python_api.cpp`、`csrc/kernels/comm/context.cpp`、
`csrc/kernels/driver/driver.cpp`。所有 device kernel（`deep_ep/include/deep_ep/impls/**/*.cuh`）
都**不在编译期实例化**，而是运行期 JIT（第 88/98 篇）；唯一的例外是那个「索引 TU」——

`csrc/indexing/main.cu`（34 行，sha256[:16] `a8ea13af1b1da3f5`）**没有任何逻辑**，它把全仓
所有 `impls/**/*.cuh` 一次性 `#include` 进来，末尾只有一个空的 `int main(){return 0;}`：

```cpp
// csrc/indexing/main.cu:1-30（节选）
#include <deep_ep/impls/comm/barrier.cuh>
#include <deep_ep/impls/bucket/reduce_scatter/{nvlink,rdma,hybrid}.cuh>
#include <deep_ep/impls/ep/{dispatch,combine,hybrid_dispatch,hybrid_combine,...}.cuh>
#include <deep_ep/impls/pp/pp_send_recv.cuh>
#include <deep_ep/impls/engram/engram_fetch.cuh>
int main() { return 0; }
```

它的唯一使命是给 IDE / clangd 一个**包含所有 kernel 的翻译单元**，好让「跳到定义」「找引用」
在 device 头文件里也能用。`setup.py` 里没有它，`CMakeLists.txt` 里专门给它建了一个
`deep_ep_indexing_cuda` 目标（`CMakeLists.txt:46`）。这就是「调试构建」的全部意义。

## 2. 目标图：一个 helper + 一条内部缓存变量

CMake 的目标图分三层，`csrc/CMakeLists.txt` 只有一行（sha256[:16] `87fb47d709e1f567`）：

```cmake
# csrc/CMakeLists.txt:1
add_subdirectory(kernels)
```

`csrc/kernels/CMakeLists.txt`（14 行，sha256[:16] `f75bdb762a1ab130`）先定义一个复用的工厂函数，
再把两个 host runtime 库拉进来，最后把它们的名字塞进一条 **`INTERNAL` 缓存变量**：

```cmake
# csrc/kernels/CMakeLists.txt:1-14
function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON     # ① 要被 _C.so 链接，必须 -fPIC
            CXX_STANDARD_REQUIRED ON
            CXX_STANDARD 20)
    target_link_libraries(${target_name} PUBLIC CUDA::cudart)
endfunction()

add_subdirectory(comm)                        # add_deep_ep_library(runtime_comm   context.cpp)
add_subdirectory(driver)                      # add_deep_ep_library(runtime_driver driver.cpp)

set(RUNTIME_CUDA_LIBRARIES runtime_comm runtime_driver CACHE INTERNAL "Runtime kernels")
```

三个非平凡点：

1. **`POSITION_INDEPENDENT_CODE ON`**（`:4`）：这两个静态库最终被链进共享库 `_C.so`，
   所以即使是 `.a` 也必须编译成 PIC，否则链接期报 `relocation ... can not be used when making a shared object`。
2. **`add_subdirectory` 的两个子目录各只有一行**：
   `csrc/kernels/comm/CMakeLists.txt:1` 是 `add_deep_ep_library(runtime_comm context.cpp)`，
   `csrc/kernels/driver/CMakeLists.txt:1` 是 `add_deep_ep_library(runtime_driver driver.cpp)`。
   `bucket/`、`ep/`、`pp/`、`engram/` 这些目录**没有** `CMakeLists.txt`——因为它们的实现在 device 头里、
   只经索引 TU 被「看见」。
3. **`RUNTIME_CUDA_LIBRARIES` 是 `CACHE INTERNAL`**（`:14`）：`INTERNAL` 属性意味着它不进 GUI、
   但**会跨目录可见**。顶层 `CMakeLists.txt:43` 正是靠它把两个静态库链进 `_C`：
   `target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)`。
   换句话说，「编哪些 runtime 库」由 `kernels/` 决定、「链哪些」由顶层决定，中间用缓存变量解耦。

## 3. 顶层 `CMakeLists.txt` 逐段

46 行、sha256[:16] `fc96cd665d654507`。按功能切四段。

### 3.1 诊断 flag（`:6-13`）

```cmake
set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -O3 -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC")
set(CUDA_SEPARABLE_COMPILATION ON)
list(APPEND CUDA_NVCC_FLAGS "-DENABLE_FAST_DEBUG")
list(APPEND CUDA_NVCC_FLAGS "-O3")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage")
list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-rdynamic")
list(APPEND CUDA_NVCC_FLAGS "-lineinfo")
```

- `--register-usage-level=10` 是 ptxas 的**寄存器用量激进档**（0–10，10 最激进）——调试构建想尽量
  把 kernel 压进寄存器、暴露 spilling；`--warn-on-local-memory-usage` 配套，任何 spill 都告警；
  `--verbose` 打印每个 kernel 的寄存器 / smem 用量。
- `CUDA_SEPARABLE_COMPILATION ON`（`:8`）开**可分离编译**——只有需要跨 TU 的 device 链接时才必要，
  这里主要是让索引 TU 能独立编。
- `-DENABLE_FAST_DEBUG`（`:9`）**在全仓没有任何 `#ifdef` 消费者**（真机普查：只有这一处定义）。
  它是个**残留开关**——本想给调试快路径留个宏，实际从没被用过。
- `-lineinfo`（`:13`）嵌行号信息，供 profiler/`compute-sanitizer` 定位（第 70 篇）。

### 3.2 环境探测与 NCCL 闸门（`:19-27`）

```cmake
find_package(CUDAToolkit REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Torch REQUIRED)
# NCCL
if (NOT NCCL_ROOT_DIR)
    message(FATAL_ERROR "NCCL_ROOT_DIR is not set.")
endif()
```

三个 `find_package` 分别要 CUDA、pybind11、Torch 的 CMake 配置。**Torch 不在默认搜索路径**，
真机必须显式 `-DCMAKE_PREFIX_PATH=$(python -c 'import torch;print(torch.utils.cmake_prefix_path)')`
（实测：不传就报 `Could not find a package configuration file provided by "Torch"`）。

`NCCL_ROOT_DIR` 的坑值得单记：`if (NOT NCCL_ROOT_DIR)`（`:25`）判的是 **CMake 变量**，
不是环境变量。真机第一次把 `NCCL_ROOT_DIR=...` 当**环境变量**传给 `cmake`，configure 照样在
`:26` 报 `FATAL_ERROR`——必须写成 `-DNCCL_ROOT_DIR=...`。这与 `setup.py` 走 `find_pkgs.find_nccl_root()`
（第 124 篇）是**两条独立的 NCCL 定位路径**。

### 3.3 include / link 目录（`:29-36`）

```cmake
set(CMAKE_CXX_STANDARD 20)  set(CMAKE_CUDA_STANDARD 20)
include_directories(deep_ep/include third-party/deep_jit/include)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include ${CUDA_TOOLKIT_ROOT_DIR}/include/cccl
                    ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS} ${NCCL_ROOT_DIR}/include .)
link_directories(${TORCH_INSTALL_PREFIX}/lib ${CUDA_TOOLKIT_ROOT_DIR}/lib ${NCCL_ROOT_DIR}/lib)
```

C++20 / CUDA20、`deep_ep/include` 与 `deep_jit/include` 两个 include 根，再加 CUDA cccl、
Torch、Python、NCCL 头与库。注意这里把 `${NCCL_ROOT_DIR}` 直接接进 include/link——所以
`-DNCCL_ROOT_DIR` 一旦指到哪份 NCCL，**编译期就用那份的头**（本机指到 pip 的 cu13 NCCL）。

### 3.4 两个产物目标（`:39-46`）

```cmake
add_subdirectory(csrc)                                   # → runtime_comm / runtime_driver
pybind11_add_module(_C csrc/python_api.cpp)              # → _C 扩展
target_link_libraries(_C PRIVATE nccl ${RUNTIME_CUDA_LIBRARIES} ${TORCH_LIBRARIES} torch_python)
cuda_add_library(deep_ep_indexing_cuda STATIC csrc/indexing/main.cu)  # → IDE 索引 TU
```

- `_C` 链了 `nccl` + 两个 runtime 静态库 + Torch（`:43`）——**它就是 `setup.py` 版 `deep_ep._C` 的孪生**，
  只是由 CMake 驱动。
- `deep_ep_indexing_cuda`（`:46`）是纯 `STATIC` 库、**谁都不链接它**——它存在的唯一理由是让 CMake
  的 IDE 插件把 `main.cu` 当成一个可索引的编译单元。

## 4. nvcc 诊断 flag 到底落在谁身上

这是本节最容易被误读的一点。上面那串 `--register-usage-level`、`--warn-on-local-memory-usage`
都是 **`CUDA_NVCC_FLAGS`**——只作用于 **`.cu`** 文件。而这个 CMake 工程里**唯一的 `.cu` 文件就是
`csrc/indexing/main.cu`**；三个 host 翻译单元是 `.cpp`，由 **CXX 编译器（g++）** 编译，
**根本不经过 nvcc**。

真机抓到的两条命令行（`--verbose` 构建）：

```
# .cu（索引 TU）—— nvcc，带满诊断 flag：
/usr/local/cuda/bin/nvcc ... -DENABLE_FAST_DEBUG -O3 \
  --ptxas-options=--verbose,--register-usage-level=10,--warn-on-local-memory-usage \
  -Xcompiler=-rdynamic -lineinfo ... -gencode arch=compute_90,code=sm_90 ...

# .cpp（driver.cpp）—— g++，一个诊断 flag 都没有：
/usr/bin/c++ ... -O3 -fPIC -std=gnu++20 ... -c .../driver.cpp
```

结论：`-DENABLE_FAST_DEBUG`（本来就没消费者）、寄存器档、spill 告警、`-lineinfo` **全部只影响
索引 TU**；两个 runtime 库只是普通 host C++。所以「调试构建」的调试价值，几乎全压在**让 IDE 索引
device 头**这件事上，而不是给 kernel 加诊断。

## 5. H100 真机实测

`harness/exp_deepep_cmake_build.{sh,py}`（纯构建 / 文件解析，不加载 GPU），完整输出见
`harness/exp_deepep_cmake_build.out.txt`。环境：cmake 3.31.6、CUDA 13.2、torch 2.12、
`find_nccl_root()` = `/usr/local/lib/python3.12/dist-packages/nvidia/nccl`。

| 步骤 | 结果 |
|---|---|
| `cmake -S . -B cmake-build-debug -DCMAKE_PREFIX_PATH=<torch> -DNCCL_ROOT_DIR=<pip nccl>` | **rc=0**（唯一警告：`library kineto not found`） |
| `--target runtime_driver` | **OK**，g++ 编译，`libruntime_driver.a` = **32794 B** |
| `--target runtime_comm`（含 `context.cpp`，要 NCCL 设备头） | **OK**，`libruntime_comm.a` = **627820 B** |
| `--target _C` | **OK**，`_C.cpython-312-x86_64-linux-gnu.so` = **1178816 B**，导出 `PyInit__C` |
| `--target deep_ep_indexing_cuda` | **rc=2（失败）** |

四目标在 `cmake --build --target help` 里都在（`_C` / `runtime_comm` / `runtime_driver` /
`deep_ep_indexing_cuda`）。三个「有用」的目标全过，**默认 `all` 会失败**——因为 `all` 也拉索引 TU。

### 5.1 索引 TU 撞上的上游真实错误

```
deep_ep/include/deep_ep/impls/bucket/reduce_scatter/rdma.cuh(619): error: no suitable
    user-defined conversion from "longlong4_t" (aka "longlong4_32a") to "const float4" exists
deep_ep/include/deep_ep/impls/bucket/reduce_scatter/rdma.cuh(623): error: ...
2 errors detected in the compilation of "csrc/indexing/main.cu".
```

读源码可见成因：`rdma.cuh:548` 把目的指针按 `longlong4_t` 推进
（`math::advance_ptr<longlong4_t>(buffer, ...)`），于是局部数组 `values[k]` 是 **`longlong4_t`**
（`:563`）；而在 reduce 的收尾处 `:619-623` 调 `ptx::st_with_gt_pred(dst_ptr + ..., values[k])` 时，
被选中的重载要的是一个 **`const float4&`**，`longlong4_t` 无法隐式转换（中间的累加走的是
`reinterpret_cast<float2*>(values)`（`:590`），说明 `values` 在语义上确实是 fp32 的 4-lane 打包，但
`st_with_gt_pred` 的指针类型推导没接上）。这是一个**只在索引 TU 这种「全量实例化」下才触发的
模板 / 重载缺口**。

关键：`setup.py` 不编这个 TU，所以 `pip install` 从头到尾不会碰到它；它只在「用 CMake 开 IDE 索引」
时暴露。这不是本机环境问题，而是**上游在该 commit 的一个真实编译缺口**。

## 6. `cmake-build-*` 这个名字从哪来

两种构建从不混淆，靠的就是一个命名约定加两处 ignore：

- `.gitignore` 第 12 行 `*/cmake-build-*/` —— IDE（CLion 默认）把构建目录建在
  `cmake-build-debug/` / `cmake-build-release/`，整目录被 Git 忽略。
- `DeepEP/setup.py:60`（在 `prepare_agent_files` 里，`:54-61`）：把 `csrc` 拷进 wheel、供装机侧按行号查错时，
  `shutil.ignore_patterns('cmake-build-*')` 把这些构建目录**从拷贝里剔除**。
  否则一个本地 IDE 的 `cmake-build-debug/` 会被打进 wheel。（第 124 篇 §7 提过 `prepare_agent_files`，
  这里补上它 ignore 的具体是什么。）

两处用同一套 `cmake-build-*` 前缀，是这条「调试构建不污染打包」约定的一体两面。

## 7. DeepGEMM 的对偶

DeepGEMM 也有一份同源的调试 CMake（34 行、sha256[:16] `8f05c0443af4daf5`），且**开头那句注释更直白**：

```cmake
# DeepGEMM/CMakeLists.txt:1
# NOTES: current just for CMake-based IDE (e.g. CLion) indexing, the real compilation is done via JIT
```

同：`-DENABLE_FAST_DEBUG` / `-O3` / `--register-usage-level=10,--warn-on-local-memory-usage` /
`CUDA_SEPARABLE_COMPILATION ON` / `CUDA_ARCH_LIST "9.0"` / `pybind11_add_module(_C csrc/python_api.cpp)` /
`cuda_add_library(deep_gemm_indexing_cuda STATIC csrc/indexing/main.cu)` —— **逐行同款**。

异（真机核对）：
- DeepGEMM 的 CMake **没有 NCCL 依赖**（不 `find_package`、不 `link nccl`），也没有
  `runtime_comm` / `runtime_driver` 静态库——它的 host 扩展只有 `python_api.cpp` 一个 TU，
  所有 device kernel 都是运行期 JIT（第 33/125 篇）。
- DeepGEMM 的 `CMakeLists.txt:25` 把 `third-party/cutlass/tools/util/include` 也加进 include——
  因为它要索引 cutlass 相关头。

所以两库的调试 CMake 是**同一套模板**：一个「全量 include 的索引 TU」+ 一个「`_C` 扩展」。
DeepEP 多出来的，正是它那两个**必须编译期编进 wheel 的 host runtime 库**（`context.cpp` 的 NCCL
设备 API 绑定 + `driver.cpp` 的 copy-engine，见第 64/86 篇）——这也解释了为什么只有 DeepEP 的
CMake 需要 `NCCL_ROOT_DIR`。

## 8. 与本机「阻塞」的关系

本机 DeepEP 运行期阻塞是「编译期链接 pip 的 NCCL 2.32.3、运行期 PyTorch 加载系统 2.29.7」
（路线图「阻塞」，第 86/107 篇）。

本篇的实测恰好把这条阻塞**从构建侧再证一遍**：
- `CMakeLists.txt:25` 的 `-DNCCL_ROOT_DIR` 指到 pip 的 `nvidia/nccl`，`context.cpp` **编译通过**
  （`libruntime_comm.a` 627820 B）——说明**编译期这份 NCCL 的头是全的**；
- 所以问题**不在构建**，而在运行期「链接的 soname 与已加载的 soname 是否为同一份逐字节文件」
  （`check_nccl_so` 的第三把尺子，第 117 篇）。

即：CMake 调试链能证明「用对头就能编过」，但**修不了运行期那个版本错配**——那需要一个链接
NCCL ≥ 2.32.3 的 PyTorch 环境。

## 9. 小结与边界

- **能跑 / 能核**：整条构建链是纯 host CMake / 文件解析、**架构无关**，H100 真机可完整复现
  （configure + 三目标编译 + 索引 TU 失败 + 产物尺寸 + nvcc/g++ 命令行 + 两处 ignore）。
- **不能核**：`_C.so` 的**运行期**加载仍卡在 NCCL 版本错配（路线图「阻塞」），本篇只到「编出来」为止；
  `rdma.cuh:619/623` 的编译缺口属上游，未改。
- **结论**：DeepEP 的 CMake 链是「一套模板两个产物」——`_C` 扩展（与 `setup.py` 孪生）+ 两个
  host runtime 静态库（DeepEP 独有、也是它需要 NCCL 的原因）+ 一个只为 IDE 索引存在的全量 TU。
  它的诊断 flag 只作用于那个索引 TU，host 库是普通 g++。到 129 篇为止，从 `find_pkgs` / `setup.py`
  到 CMake 调试链、再到运行期加载，**「构建 / 打包 / 安装期」这条线已闭环**。
