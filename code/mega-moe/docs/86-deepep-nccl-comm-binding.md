# DeepEP 的 NCCL 通信域绑定与 Python 引导层

> 本专题（P76-1）前面的 DeepEP 篇目——第 7 篇（dispatch/combine 数据流）、第 48/51 篇（两个
> epilogue）、第 53 篇（多 rank 屏障与 symmetric-memory 句柄）、第 58 篇（PP send/recv）、
> 第 61 篇（hybrid dispatch/combine）、第 63 篇（自动 SM/QP 选择）、第 65 篇（stream/event
> 重叠）、第 71/73/75 篇（EP 测试与 host 编排）——**全部从一个已经存在的通信组/对称窗口讲起**。
> 但「DeepEP 怎么把一个 PyTorch 的 NCCL 通信组变成一块可对称寻址的窗口」这层地基从没被单独
> 拆过。
>
> 这一层也正是本机**运行期阻塞的直接原因**：DeepEP 链接的是 pip 的 NCCL 2.32.3，而本机
> PyTorch 链的是系统 NCCL 2.29.7，`import deep_ep` 在引导阶段就被 `check_nccl_so()` 拦下
> （见 §9 实测）。所以本篇不仅是补一块拼图，也是把「阻塞」这四个字落到**真实行号**上。
>
> **本层是纯 host / Python**：包发现、NCCL 版本核对、comm 句柄生命周期、域大小算术，都不碰
> 任何 kernel / UMMA / WGMMA / NCCL 设备 API。故 H100（sm90）真机可逐条实测（§9）。
>
> 参考仓库：`~/github/DeepEP@93eb6eb`（v2.5.0）。行号均以该 commit 核对。

---

## §1 Python 引导顺序：`import deep_ep` 就是一个有副作用的初始化协议

`deep_ep/__init__.py` 的顶层代码**不是**简单的名字导出，而是一段按严格顺序执行的引导：

```
deep_ep/__init__.py
  :8-16    ① 灌入安装期固化的 persistent_envs
  :19-41   ② check_nccl_so()       —— 运行时/链接期 NCCL 二进制一致性核对
  :44-51   ③ init_jit()            —— 初始化 DeepJIT（把 NCCL 根目录传进去）
  :54-56   执行 ② ③
  :59-75   ④ 之后才 import comm / buffers / utils 等真正用到 `_C` 的模块
```

顺序不可颠倒，因为：

- ① 必须在 `_C` 加载前设置好环境（如 `EP_NCCL_ROOT_DIR`、`EP_NUM_TOPK_IDX_BITS`）——这些是
  **安装时**由 `setup.py` 烘进 `deep_ep/envs.py` 的（§4）。
- ② 必须在任何 NCCL 调用前，否则 PyTorch 可能已经载入了另一个 NCCL 库。
- ③ 必须在 ④ 之前，因为 `buffers/*` 的 C++ 扩展在构造时就假定 JIT 运行时已就绪。
- ④ 里每个模块 `import deep_ep._C as _C`，一旦 ② 抛异常，整个包就 import 失败。

`__init__.py:8-16` 的 `persistent_envs` 读取用 `try/except ImportError` 包住：源码树里没有
`envs.py`（只有 build 时才生成），于是从源码直接跑时这一块静默跳过。

---

## §2 包发现层 `find_pkgs.py`：NCCL 到底装在哪

`deep_ep/utils/find_pkgs.py` 只做一件事——**在众多可能被安装的 NVIDIA 包中，定位「哪个」
NCCL 才是这一个 DeepEP 应该链接/核对的那一个**。

### 2.1 `find_pkg_root(name, lib_name, optional)`（`:9-55`）

算法四步（`：22-50`）：

1. **环境变量短路**（`:22-25`）：先看 `EP_{NAME}_ROOT_DIR`（如 `EP_NCCL_ROOT_DIR`），再看
   `{NAME}_DIR`（如 `NCCL_DIR`）。任一存在**直接返回，不做任何搜索**。
2. **按 `sys.path` 顺序给每个 dist 的 site 目录打分**（`:27-28`）：`path_priority = {p: i}`，
   优先取**在 `sys.path` 里更靠前**的安装。这解决「同时装了 `nvidia-nccl-cu12` 与
   `nvidia-nccl-cu13`」时的歧义——和 Python `import` 的解析顺序保持一致。
3. **名字匹配**（`:30-33`）：只认 `nvidia-{name}` 或 `nvidia_{name}`（归一化 `-`/`_`）。因此
   只看 NVIDIA 官方 pip wheel，不看系统 `/usr/lib` 里的 `libnccl`。
4. **定位逻辑**（`:40-50`）：给了 `lib_name` 就从 `dist.files` 里找包含该名字的文件，取其所
   在目录；若目录名是 `lib` 则再上跳一级当作 root（pip wheel 惯例是 `<root>/lib/libnccl.so.2`）。
   没给 `lib_name` 就退化成 `<site>/nvidia/<name>`。

`optional=False`（默认）时找不到就 `assert`（`:52-54`），直接中止。

### 2.2 `find_nccl_root`（`:58-69`）与 `get_nccl_lib_name`（`:72-77`）

- `find_nccl_root` 只是 `find_pkg_root('nccl', lib_name='libnccl.so')` 的薄封装，并加了
  `@functools.lru_cache()`（`:58`）——**同进程内只搜一次**（实测 `hits=1`）。这也意味着改
  `EP_NCCL_ROOT_DIR` 后必须重启进程才生效。
- `get_nccl_lib_name`：在 `<root>/lib` 下优先返回无版本的 `libnccl.so`，否则返回排序后的
  `libnccl.so.*`（即 SONAME）。**为什么不用 `libnccl.so`**：pip 的 `nvidia-nccl` wheel 只
  发 `libnccl.so.2`，没有开发用软链。所以链接时必须用 `-l:libnccl.so.2`（§4）。

本机实测：`find_nccl_root()` = `/usr/local/lib/python3.12/dist-packages/nvidia/nccl`，
`get_nccl_lib_name()` = `libnccl.so.2`（目录里只有这一个文件）。

---

## §3 运行时核对 `check_nccl_so`：为什么要比二进制，而不是比版本号

`__init__.py:19-41` 的 `check_nccl_so` 有**三把尺子**：

1. **抑制开关**（`:24-25`）：`EP_SUPPRESS_NCCL_CHECK=1` 时直接返回。
2. **重复加载检测**（`:27-32`）：扫 `/proc/self/maps`，把所有映射了 `libnccl` 的路径收集起来；
   一旦发现**两个不同路径**（PyTorch 载入的 + 别人 preload 的）就断言失败
   （`Duplicate NCCL runtime found`）。同一路径出现多次是正常的（多段映射）。
3. **二进制级一致性**（`:33-41`）：取 DeepEP 期望的 `libnccl.so*`（`find_nccl_root`），和
   **实际被载入的那个**用 `filecmp.cmp(..., shallow=False)` 做**逐字节比较**。

**为什么是逐字节、不是版本号字符串**：NCCL 的设备通信 API（`ncclDevComm`、
`ncclGin`、`ncclTeamLsa`）的**结构体布局/符号**在不同 build 之间可能变。哪怕两边版本号都写
2.32.3，只要编译选项/补丁不同，混用就会在设备侧炸。逐字节比较是最保守也最可靠的做法。
`shallow=False` 不能省：`shallow=True` 只比 `stat`（大小/mtime），内容不同但大小相同的库会漏。

本机实测（§9 B2/B3）：loaded = `/usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7`，
linked = `.../nvidia/nccl/lib/libnccl.so.2`（2.32.3），`filecmp=False`，于是 import 直接
`AssertionError`——这就是「阻塞」在代码里的确切落点。

---

## §4 编译期链接 `setup.py`：把哪份 NCCL 烧进 `_C`

`setup.py` 负责把「本机的 NCCL」钉进 `_C`，与 §3 的运行时核对形成一对。

- **持久环境变量**（`:14-15`）：`persistent_env_names = ('EP_JIT_CACHE_DIR', 'EP_JIT_PRINT_COMPILER_COMMAND',
  'EP_JIT_CPP_STANDARD', 'EP_NUM_TOPK_IDX_BITS', 'EP_NCCL_ROOT_DIR', 'EP_DEFAULT_RDMA_SL',
  'EP_OVERRIDE_RDMA_SL')`。安装时若这些变量存在，会被写进 `deep_ep/envs.py`（`:63-74` 的
  `generate_default_envs`），运行时由 `__init__.py:8-16` 回灌。这样**集群上的缓存目录/SL 就
  固化进了 wheel**。
- **`find_pkgs` 提前加载**（`:17-20`）：setup 阶段直接按文件路径 exec `find_pkgs.py`，**避免
  触发 `deep_ep.__init__`**（否则会 import torch、还要 `check_nccl_so`）。
- **NCCL 头/库/链接**（`:90-93`）：
  ```
  include_dirs += ['{nccl_root_dir}/include']
  library_dirs += ['{nccl_root_dir}/lib']
  extra_link_args += ['-l:{get_nccl_lib_name(nccl_root_dir)}', '-Wl,-rpath,{nccl_root_dir}/lib']
  ```
  注意 `-l:`（带冒号）语法：它按**精确文件名**链接（`libnccl.so.2`），而不是 `-lnccl` 的
  常规搜索。再加 `-rpath` 保证运行时直接命中这一份，而非系统默认路径。
- **`_C` 的源码只有三个文件**（`:81-83`）：`csrc/python_api.cpp`、`csrc/kernels/comm/context.cpp`、
  `csrc/kernels/driver/driver.cpp`——**设备 kernel 全部走 DeepJIT 现编**（对应第 33 篇的
  DeepJIT），`_C` 本体只是 host 绑定 + 上下文管理。
- `EP_NUM_TOPK_IDX_BITS`（`:95-98`）把 top-k 索引位宽（32/64）编进 `_C`。

---

## §5 `_C` 的绑定面：`python_api.cpp`

`csrc/python_api.cpp:19-51` 的 `PYBIND11_MODULE(_C, m)` 顺序注册：

- **对齐常量**（`:24-26`）：`get_num_allocation_alignment`（2 MiB）、`get_num_tma_alignment`
  （32 B）、`get_num_rdma_alignment`（64 B）——对应 `compiled.cuh:67-69`。
- **`topk_idx_t`**（`:23`）：由 `EP_NUM_TOPK_IDX_BITS` 决定的标量类型，Python 侧
  `deep_ep.topk_idx_t` 直接引用（`__init__.py:72-76`）。
- **DeepJIT**（`:29`）、**Event**（`:32`）、**comm**（`:35`）、**base buffer**（`:38`）、
  **ep / engram / bucket / pp**（`:41-50`）。

本篇关心的是 `deep_ep::comm::register_apis`（`csrc/comm/handle.hpp:8-30`）：

- 暴露 C++ `Context` 类（`:9-14`）：`get_physical_domain_size` / `get_logical_domain_size`
  两个方法，以及 `num_workspace_bytes` / `num_gpu_buffer_bytes` / `num_rdma_storage_bytes`
  三个只读字段。
- 注册 `barrier`（`:16-22`）：一个把 `Context` + 三个布尔转给 §7 那个 C++ `barrier()` 的 lambda；
  里面断言 `context->barrier_signals != nullptr`（`:19`）。
- 注册裸 NCCL 原语（`:24-28`）：`get_local_nccl_unique_id` / `create_nccl_comm` /
  `destroy_nccl_comm` / `get_physical_domain_size` / `get_logical_domain_size`。
- 注册 `get_comm_stream`（`:29`），转 `stream.hpp:9-12` 的静态高优先级通信流（第 65 篇）。

---

## §6 `Context`：一条 PyTorch 通信组 → 一块对称窗口

`csrc/kernels/comm/context.cpp` 的 `Context` 构造（`:75-224`）是整层的核心。它把「一条 NCCL
host 通信组」翻译成设备 kernel 需要的所有坐标与指针。

### 6.1 输入断言（`:94-98`）

```
num_ranks ∈ (0, kNumMaxRanks=1024]
num_allocated_qps ∈ (0, kNumMaxQPs=1024]
num_workspace_bytes > 0 且 % kNumAllocationAlignmentBytes(2MiB) == 0
num_gpu_buffer_bytes ≥ 0 且 % 2MiB == 0
num_rdma_storage_bytes ≥ 0 且 % 2MiB == 0
```

即**所有对称缓冲尺寸都必须是 2 MiB 的整数倍**（对应第 56 篇的分配粒度）。

### 6.2 版本闸门（`:101-106, 156-162`）

先 `ncclGetVersion` 打日志（`:101-106`），再在 `:156-162`：

```
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize);   // 尺寸运行时查
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version
                   and "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. ...");
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t));            // 尺寸编译期固定
#endif
```

**为什么 2.31 是分水岭**：2.31 起 `ncclDevComm` 的运行时尺寸可查（`devCommRuntimeVersionSize`），
所以允许编译期/运行时版本不同；2.31 之前结构体布局由编译期固定，二者必须**完全同版本**。本机
链接的 pip 是 2.32.3（`>= 2.31`），所以走 `useRuntimeVersion` 分支、这条断言不会触发——但
§3 的 `check_nccl_so` 在**更早**就因二进制不同把它拦了（这正是「两个闸门、Python 那个先响」）。

### 6.3 GIN 需求与断言（`:115-155`）

`:116-117` 查 `ncclCommProperties`。若启用了 LSA multimem，断言 `props.multimemSupport`
（`:118-119`）。随后（除非 `EP_DISABLE_GIN=1`，`:124`）为设备通信申请 GIN（GDAKI）：

- `props.ginType` / `railedGinType` 非 NONE（`:125-128`）——`allow_hybrid_mode=0` 的多平面
  网络会在此报错。
- `props.ginSupport[NCCL_GIN_TYPE_GDAKI]`（`:129-131`）。
- `gin_min_stride = props.ginMinStride`（`:133`），并把 `ginType/ginContextCount/ginExclusiveContexts/
  ginQueueDepth` 填进 `reqs`（`:134-137`）。
- **`reqs.ginSignalCount = num_ranks + 2 * 2`**（`:140`）：跨 rank 屏障需要额外信号，`2*2`
  是给 `BarrierSignals` 双相位双槽预留的（第 53 篇）。
- 连接类型（`:141-146`）：hybrid 且用 RDMA storage 时用 `NCCL_GIN_CONNECTION_CUSTOM_STRIDE`
  + `ginCustomStride=gin_min_stride`，否则 hybrid 用 RAIL、非 hybrid 用 FULL。
- SL（traffic class）三级覆盖（`:148-154`）：`EP_DEFAULT_RDMA_SL` < `sl_idx` 参数 <
  `EP_OVERRIDE_RDMA_SL`。
- `ncclDevCommCreate`（`:164`）。

### 6.4 两域拆分（`:166-183`）

从 NCCL 自身拿 NVLink 域（`ncclTeamLsa`，`:167`）：

```
num_nvl_ranks = lsaTeam.nRanks;  nvl_rank_idx = lsaTeam.rank
num_rdma_ranks = num_ranks / num_nvl_ranks;  rdma_rank_idx = rank_idx / num_nvl_ranks
```

并断言域大小不超上限（`:170`）、`num_ranks % num_nvl_ranks == 0`、rank 的**混合进制**分解自洽
（`:171-172`：`rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx`）。

**scaleout / scaleup 的逻辑域**（`:174-183`）：

```
allow_hybrid_mode: scaleout = (num_rdma_ranks, rdma_rank_idx)  scaleup = (num_nvl_ranks, nvl_rank_idx)
否则:              scaleout = (1, 0)                            scaleup = (num_ranks, rank_idx)
is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks)
```

这和 `get_logical_domain_size`（`:59-63`）的公式一一对应——**非 hybrid 时把整个世界当作一个
scaleup 域**（第 63/71 篇的 hybrid vs direct 分派就建立在 `is_scaleup_nvlink` 上）。

### 6.5 对称内存 + window + 缓冲切分（`:185-223`）

- `symmetric::alloc(...)` 分配对称内存（`:186-190`，第 56 篇）；若给了 `main_context` 则
  **共享其对称内存**（`:191-193`，多个 buffer 复用同一窗口）。
- `ncclCommWindowRegister`（`:199`）：把整块对称内存注册成 NCCL 窗口。注释（`:196-197`）指出
  这是**集合调用**，内部已含 bootstrap barrier，故无需额外 barrier。
- `ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace)`（`:200`）拿到**本 rank 视角**
  的窗口基址，切成 `workspace` + `buffer = workspace + num_workspace_bytes`（`:201`）。
- 可选 multimem 指针（`:202-203`）；首建时 `cudaMemset(workspace, 0, ...)`（`:204-205`）。
- **RDMA storage windows**（`:207-223`）：当 `num_rdma_storage_bytes > 0` 时，为 `gin_min_stride`
  个 stride 各注册一个 `NCCL_WIN_GIN_ONLY` 窗口（`:212-217`），并把窗口数组拷到设备
  （`:219-222`）。本 rank 的 storage 起点按 `(rank_idx % gin_min_stride) * num_rdma_storage_bytes`
  定位（`:209`）。

### 6.6 `get_sym_ptr` 与 `finalize`（`:226-266`）

- `get_sym_ptr(ptr, dst_nvl_rank_idx)`（`:226-232`）：先算 `offset = ptr - workspace`，再用
  `ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx)` 换取**目标 rank 上同 offset 的地址**。
  这就是「对称寻址」的本质——**同一逻辑位置在不同 rank 上由 NCCL 翻译**。偏移量与目标 rank
  无关（§8 E5 实测）。
- `finalize`（`:244-266`）**严格逆序释放**：先设备端 window 数组（`:246-249`）→ 逐个
  `ncclCommWindowDeregister` RDMA storage（`:250-252`）→ 主窗口（`:256`）→ 释放对称内存
  （`:257`）→ 置空 `barrier_signals`（`:260`，之后 barrier 不可用）→ `ncclDevCommDestroy`
  （`:263`）→ 置 `finalized=true`（`:265`）。`get_physical/logical_domain_size()` 都断言
  `not finalized`（`:235`/`:240`）——用后即失效。

---

## §7 Python comm 层：句柄生命周期、缓存、barrier 包装

`deep_ep/comm/` 是 C++ `Context` 之上的薄 Python 层。

### 7.1 `NCCLCommHandle`（`comm/handle.py:12-38`）

一个**引用计数式**的句柄：`managed=True` 表示这条 NCCL host comm 是 DeepEP 自己建的，
`__del__` 时 `destroy_nccl_comm`（`:27-29`）；`managed=False` 表示它是**借**的 PyTorch
通信组，绝不能销毁。`destroy` 在构造时就绑定 `_C.destroy_nccl_comm`（`:25`）。

### 7.2 `get_nccl_comm_handle`（`handle.py:44-77`）

三条路径：

1. **缓存命中**（`:57-60`）：`_storage` 以 group 为键；`force_new_comm=False` 且存在则直接返回。
2. **复用 PyTorch 的 comm**（`:62-66`）：若 `group._get_backend(cuda)` 有 `_comm_ptr()`
   （新版 PyTorch 私有 API），且 `EP_REUSE_NCCL_COMM != 0`，就包一个 `managed=False` 句柄。
   **默认走这条**——零成本。
3. **自建 comm**（`:68-77`）：老版 PyTorch 没有 `_comm_ptr` 时，`all_gather_object` 收集
   `get_local_unique_id`，由 root 的 unique id `create_nccl_comm`，包 `managed=True`。
   `force_new_comm=True` 时键用 `time.time_ns()`（`:74`）——**故意不缓存**，每次都新建。

`destroy_all_managed_nccl_comm`（`:98-103`）只 `_storage.clear()`；被清掉的 managed 句柄在
引用归零时经 `__del__` 自动销毁。

### 7.3 域大小包装（`handle.py:80-95`）

`get_physical/logical_domain_size(group_or_buffer, ...)`：若对象带 `.context` 属性，直接调
`Context` 的同名方法（**不经过 `_C` 的设备侧查询**）；否则经 `_C.get_*_domain_size(comm, ...)`。
`get_logical_domain_size` 的 `allow_hybrid_mode` 默认 `None` → 当作 `True`（`:94`）。

### 7.4 `barrier`（`comm/barrier.py:4-11`）

Python `barrier(buffer, wait_comm_stream=True, with_cpu_sync=False, sequential=True, context_idx=None)`
只做一件转发：取 `buffer.context`（或 `buffer.contexts[context_idx]`），调
`_C.barrier(context, with_cpu_sync, sequential, wait_comm_stream)`。真正的实现在
`csrc/comm/barrier.hpp:9-42`：`with_cpu_sync` 走 `cudaDeviceSynchronize`（`:18-19`），
否则 `wait_comm_stream` 时先 `stream_wait(stream, get_comm_stream())`（`:20-22`），发射
`launch_barrier`（`:25-33`），再在收尾 `stream_wait(get_comm_stream(), stream)`（`:38-41`）
——**计算流只在 barrier 前后与通信流对齐**，中间不阻塞。

`launch_barrier`（`csrc/kernels/comm/barrier.hpp:17-54`）：线程数固定 512（`:26`），
SM 数 `(not sequential and num_scaleout_ranks > 1) ? 2 : 1`（`:30`），JIT 现编
`barrier_impl<...>`（`:33-41`），`cooperative=true` 发射（`:44-53`）——与本篇其余部分（host）
不同，这一步会真正编/发设备 kernel（设备侧细节见第 53 篇）。

---

## §8 域大小算术与对称偏移

`Context` 暴露给上层的两个域大小公式（§6.4）可以直接用纯算术复刻，并与独立 oracle 对拍
（§9 E）：

- **物理域**（`context.cpp:52-57`）：`(num_ranks / num_nvl_ranks, num_nvl_ranks)`，其中
  `num_nvl_ranks = ncclTeamLsa.nRanks`。
- **逻辑域**（`:59-63`）：hybrid → 物理域；非 hybrid → `(1, num_ranks)`。

**EP16、每节点 8 卡**：hybrid 时 `logical=(2,8)`，非 hybrid 时 `logical=(1,16)`——这正是
第 63/71 篇「direct vs hybrid 分派」的坐标来源。**EP8 单节点**：无论 hybrid 与否，
`logical` 都退化成 `(1,8)`（因为 `num_rdma_ranks=1`）。

---

## §9 H100 真机实测

实验：`harness/exp_deepep_nccl_binding.py` + `.sh` → `.out.txt`（H100 80GB HBM3、132 SM、驱动 570）。
做法：按文件路径直接加载真实 `find_pkgs.py`；用 `sys.modules` 注入合成 `deep_ep._C` stub 后
exec 真实 `comm/handle.py` / `comm/barrier.py`；用 `ast` 抽取真实 `check_nccl_so` 函数体运行；
用 `ctypes` `dlopen` 两个 NCCL 库读 `ncclGetVersion`。**PASS=35 / FAIL=0**。

| # | 断言 | 实测 |
|---|------|------|
| A1-A2 | `find_nccl_root` / `get_nccl_lib_name` | root=`.../nvidia/nccl`；lib=`libnccl.so.2`（目录仅此一个） |
| A3 | env 覆盖优先级 | `EP_NCCL_ROOT_DIR` 胜过 `NCCL_DIR`，无前者才回退 |
| A4-A5 | 未找到：optional 返回 None / 否则 AssertionError | `Cannot find package: ...` |
| A6 | 独立 oracle 复刻 `find_pkg_root` | 与真实结果**一致** |
| A7 | `lru_cache` | `CacheInfo(hits=1, misses=1)` |
| B2 | loaded ≠ linked（真机失配） | loaded=`/usr/lib/.../libnccl.so.2.29.7`，linked=`.../nvidia/nccl/lib/libnccl.so.2`，`filecmp=False` |
| B3 | `check_nccl_so` 抛 AssertionError | `Invalid NCCL versions: ... (loaded) v.s. ... (expected)` |
| B4 | 用 `shallow=False` | 源码核对 |
| C1 | 复用 PyTorch comm | `managed=False`，ptr 原样；二次调用同一对象 |
| C2 | `EP_REUSE_NCCL_COMM=0` | `managed=True`，`create_nccl_comm(uid, 8, 3)` 被调用一次 |
| C3 | `force_new_comm=True` | 绕缓存、新建 managed 句柄 |
| C4 | 析构语义 | managed 裸句柄析构 → `destroy_nccl_comm`；PyTorch 复用句柄清缓存后**不**销毁 |
| C5 | `destroy_all_managed_nccl_comm` | 清空 `_storage` |
| C6 | 域大小包装 | `.context` 分支 vs `_C` 分支都对 |
| D1-D2 | `barrier` 参数转发 | 4 参按序；`context_idx` 覆盖 `.context` |
| E1-E5 | 域算术 vs oracle | 8 配置逐一致；非 hybrid `(1,16)`/hybrid `(2,8)`；`ginSignalCount=num_ranks+4`；`get_sym_ptr` offset 与 dst rank 无关 |
| F1-F4 | NCCL 版本探测 | system `2.29.7`、pip `2.32.3`；`>=2.31` 走 `useRuntimeVersion` 分支但仍被 B3 先拦 |

关键真机数字：**loaded 系统 NCCL = 2.29.7，DeepEP 链接的 pip NCCL = 2.32.3**；
`import deep_ep` 在 `check_nccl_so`（`__init__.py:39`）抛 `AssertionError`，消息即
`Invalid NCCL versions: /usr/lib/x86_64-linux-gnu/libnccl.so.2.29.7 (loaded) v.s.
/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2 (expected)`。

---

## §10 与本机「阻塞」的关系

ROADMAP「阻塞」记的是「DeepEP 运行期实验受阻（环境，非内核）」。本篇把它的**首个确切落点**
钉到了行号：

1. **引导期**（本篇）：`import deep_ep` → `check_nccl_so`（`__init__.py:39`）因 PyTorch
   链系统 2.29.7、DeepEP 链 pip 2.32.3 而中止。绕过办法是 `EP_SUPPRESS_NCCL_CHECK=1`
   （§9 B1 实测直接返回）。
2. **构造期**（第 56 篇 + 本篇 §6.3）：即使绕过 ①，`Context` 构造要求 NCCL 提供
   `ncclDevCommCreate` / `ncclGin`（GDAKI）/ `ncclTeamLsa`，这些是 **≥2.31** 的设备 API；
   系统 2.29.7 没有。用系统头重建 `_C` 会编译失败；用 pip 2.32.3 构建但 PyTorch 载入系统
   2.29.7 则 ABI 不匹配 → `EPBuffer` 构造段错误。
3. **正确解法**：一个**链接 NCCL ≥ 2.32.3** 的 PyTorch 环境（让 `_C` 与 PyTorch 用同一份
   NCCL），使 §3 的 `filecmp` 通过、§6.2 的闸门走 `useRuntimeVersion`、§6.3 的 GIN 断言成立。

换言之，阻塞不是内核逻辑问题，而是**两个闸门（Python 二进制核对 + C++ 设备 API 版本）各自
要求 `_C` 与 PyTorch 使用同一份 ≥2.31 的 NCCL**。

---

## §11 与其他篇目的关系

- **第 56 篇**：`symmetric::alloc` / `ElasticSymmetricMemory`（§6.5 `:186-190` 的底层）。
- **第 53 篇**：`BarrierSignals` / `nvlink_barrier`（§6.3 `ginSignalCount`、§7.4 `launch_barrier`
  的设备侧）。
- **第 63/71 篇**：`direct` vs `hybrid` 分派建立在 §6.4 的 `is_scaleup_nvlink` 与 §8 的逻辑域上。
- **第 65 篇**：`get_comm_stream` / `EventOverlap`（§5 `:29`、§7.4 的 `wait_comm_stream`）。
- **第 33/37 篇（DeepGEMM DeepJIT）**：`init_jit`（`__init__.py:44-51`）把包目录与 NCCL 根目录
  传给 DeepJIT，与 DeepGEMM 的 `_C.init(dirname)` 同构。

## §12 源码索引（commit `93eb6eb`）

| 文件 | 关键行 |
|------|--------|
| `deep_ep/__init__.py` | persistent_envs `:8-16`、`check_nccl_so` `:19-41`、`init_jit` `:44-51`、执行 `:54-56` |
| `deep_ep/utils/find_pkgs.py` | `find_pkg_root` `:9-55`、`find_nccl_root` `:58-69`、`get_nccl_lib_name` `:72-77` |
| `deep_ep/comm/handle.py` | `NCCLCommHandle` `:12-38`、`get_nccl_comm_handle` `:44-77`、域大小 `:80-95`、`destroy_all_managed_nccl_comm` `:98-103` |
| `deep_ep/comm/barrier.py` | `barrier` `:4-11` |
| `csrc/python_api.cpp` | `PYBIND11_MODULE` `:19-51`、对齐 `:24-26`、register `:29-50` |
| `csrc/comm/handle.hpp` | `register_apis` `:8-30` |
| `csrc/comm/barrier.hpp` | `barrier` `:9-42`（`with_cpu_sync` `:18-19`、`stream_wait` `:20-23/38-41`） |
| `csrc/kernels/comm/api.hpp` | 声明 `:18-29`、`Context` 类 `:31-88` |
| `csrc/kernels/comm/context.cpp` | unique id `:23-29`、create `:31-44`、destroy `:46-50`、域大小 `:52-63`、`gin_min_stride` `:65-73`、ctor `:75-224`、`get_sym_ptr` `:226-232`、`finalize` `:244-266` |
| `csrc/kernels/comm/barrier.hpp` | `launch_barrier` `:17-54` |
| `setup.py` | persistent_envs `:14-20/63-74`、NCCL 链接 `:90-93`、`_C` 源码 `:81-83` |
| `deep_ep/include/deep_ep/common/compiled.cuh` | 对齐/上限常量 `:67-81` |
