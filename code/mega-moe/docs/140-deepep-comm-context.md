# 140. DeepEP 的 host 通信上下文 `comm::Context`：GIN 设备通信器、NCCL 运行时版本闸门与对称窗口生命周期

> 本专题到第 139 篇，已经把 DeepEP 的 dispatch/combine 数据流（第 7 篇）、两个 epilogue
> （第 48/51 篇）、多 rank 屏障与 symmetric-memory 句柄（第 53/63 篇）、对称内存分配（第 66 篇）、
> 自动 SM/QP 选择（第 73 篇）、NCCL 通信域绑定与 Python 引导层（第 86 篇）、四个 Buffer 家族
> （第 64/68/69/121 篇）都拆过。但它们**全部从一个已经存在的「上下文/窗口」讲起**——而把
> 「一个 PyTorch 的 NCCL 通信组」变成「一块可对称寻址、可发起 GIN RDMA、可两级域编排的窗口」
> 的那**一个 host 对象** `deep_ep::comm::Context`（`csrc/kernels/comm/context.cpp`，268 行），
> 从没被单独读过。
>
> 本篇补上它。三块内容此前都是分散的一句带过：① **GIN 设备通信器需求** `ncclDevCommRequirements_t`
> 的逐字段构造；② **NCCL 运行时版本闸门**（`#if NCCL_VERSION_CODE >= NCCL_VERSION(2,31,0)`）——
> 这正是本机「阻塞」的一处**机制级**根因；③ **对称窗口的字节布局与生命周期**（注册、`get_sym_ptr`
> 偏移翻译、`finalize()` 拆除顺序）。
>
> **端到端的 NCCL 设备 API（`ncclDevCommCreate` / `ncclCommWindowRegister`）本机 NCCL 2.29.7
> 不提供（见「阻塞」），仅源码分析**；但这一文件的**全部 host 算术**（域分解、signal 预算、
> 偏移布局、版本闸门判据、句柄缓存语义）架构无关，H100(sm90) 真机逐条实测。所有数字来自
> `harness/exp_deepep_context.{py,sh,out.txt}`，设备常量来自
> `src/deepep_context_arith.cu`（**直接 include 真实头文件** `deep_ep/common/compiled.cuh` +
> `math.cuh`）。
>
> 参考坐标：`~/github/DeepEP@93eb6eb`（v2.5.0）。行号均以该 commit 核对。

---

## 1. 为什么需要一个 `Context` 对象

一次 EP 通信（dispatch/combine）要同时用到四类资源，它们的生命周期**必须与 `EPBuffer` 对齐**：

1. **一个 NCCL host 通信器**（`ncclComm_t`）——用来创建**设备通信器**、注册窗口；
2. **一个 NCCL 设备通信器**（`ncclDevComm_t`）——device 侧 `ncclGin` / `ncclTeam` 的载体；
3. **一块注册进 NCCL 的对称窗口**（`ncclWindow_t`）——`ncclGetLsaDevicePointer` 能把
   「本 rank 的偏移」翻译成「目标 rank 的同偏移地址」；
4. **一批 RDMA storage 窗口**（每 `gin_min_stride` 个 rank 一份）——hybrid 模式下跨节点的
   RDMA 写目标。

`Context`（`csrc/kernels/comm/api.hpp:31-88`）就是这四类资源的**唯一所有者**，字段分三组：

| 组 | 字段（api.hpp） | 作用 |
|---|---|---|
| 域分解 | `rank_idx/num_ranks`、`num_rdma_ranks/num_nvl_ranks`、`rdma_rank_idx/nvl_rank_idx`、`num_scaleout_ranks/num_scaleup_ranks`、`scaleout_rank_idx/scaleup_rank_idx`、`is_scaleup_nvlink`、`gin_min_stride`（`:37-43`） | 决定 kernel 里「对谁做 NVLink、对谁做 RDMA」 |
| NCCL 句柄 | `comm`、`dev_comm`（`deep_jit::NoRefPtr`）、`window`（`:45-47`） | 三个原生句柄 |
| 缓冲指针 | `num_workspace_bytes/num_gpu_buffer_bytes/num_rdma_storage_bytes`、`workspace`、`barrier_signals`、`buffer`、`buffer_multimem`、`local_rdma_storage_ptr`、`rdma_storage_windows[_gpu]`（`:53-63`） | 窗口内的**字节视图**（第 4/5 节） |

一个关键设计：`symmetric_memory` 是 **`shared_ptr`**（`:33`）。多个 `Context` 可以共享**同一块**
对称内存（`main_context` 参数，构造 `context.cpp:185-193`），分别注册**自己的窗口**——这让
「一个物理缓冲、多套通信域」成为可能。

---

## 2. host 通信器的引导：`get_local_unique_id` / `create_nccl_comm`

（`context.cpp:23-50`）

```cpp
pybind11::bytearray get_local_unique_id();                       // :23-29  ncclGetUniqueId
int64_t create_nccl_comm(uid_bytes, num_ranks, rank_idx);        // :31-44  ncclCommInitRank
void    destroy_nccl_comm(int64_t nccl_comm);                    // :46-50  ncclCommAbort
```

`create_nccl_comm` 把 `ncclComm_t` 指针**当作 `int64_t` 直接返回给 Python**（`:43` 的
`reinterpret_cast<int64_t>`），Python 侧由 `deep_ep/comm/handle.py:12-29` 的 `NCCLCommHandle`
包住，`managed` 标志决定 `__del__` 时要不要 `destroy`。这是「DeePEP 自建 comm」这条路。

**更常见的是复用 PyTorch 的 comm**：`handle.py:44-77` 的 `get_nccl_comm_handle` 三分支——
① 命中全局缓存 `_storage`（`:59-60`）；② 新 PyTorch 有 `backend._comm_ptr` 且未设
`EP_REUSE_NCCL_COMM=0` 时**直接包住 PyTorch 的 comm**（`managed=False`，`:64-66`）；③ 否则
`all_gather_object` 广播 unique id 后 `create_nccl_comm`（`managed=True`，`:68-77`）。
第 86 篇讲过引导顺序；本篇的实测（§5）只闭环这三条分支的语义。

---

## 3. 物理域 / 逻辑域 / `gin_min_stride` 三个查询

（`context.cpp:52-73`）

```
get_physical_domain_size(comm) = (num_ranks / num_nvl_ranks, num_nvl_ranks)   // (RDMA, NVLink) :52-57
get_logical_domain_size(comm, hybrid) =
    hybrid ? (num_rdma_ranks, num_nvl_ranks)                                  // :61-62
           : (1,             num_rdma_ranks * num_nvl_ranks)                  // 单级：全在一个域
get_gin_min_stride(comm) = (props.ginMinStride, num_ranks / props.ginMinStride)  // :65-73
```

- **物理域**是运行时问 NCCL 的：`ncclTeamWorld(comm).nRanks` 与 `ncclTeamLsa(comm).nRanks`
  （`:54`，Lsa = Local SHARP/LSA 域 = 一个 NVLink 平面）。硬约束 `num_ranks % num_nvl_ranks == 0`
  （`:55`）。第 119 篇用它解释「NVLink 域越大、一次能直接互访的 GPU 越多」。
- **逻辑域**是**调度域**：hybrid 时等于物理域（scale-out = RDMA 维、scale-up = NVLink 维）；
  非 hybrid 时把整个世界当成**一个 NVLink 域**（`num_scaleup_ranks = num_ranks`，`scaleout=1`），
  于是所有通信都走 LSA。kernel 里 `num_scaleout_ranks/num_scaleup_ranks` 直接来自这里。
- **`gin_min_stride`** 是 GIN 的**最小跨节点步长**（multi-plane 网络里的 rail 数）：它同时是
  §6 里「RDMA storage 窗口个数」和 hybrid 分配的贴片步长。

H100 实测（`harness/exp_deepep_context.out.txt` §3）对 6 组 `(num_ranks, num_nvl_ranks)` ×
5 个 rank × hybrid/非-hybrid 共 **60 行逐字段 0 不符**，并独立验证不变量
`num_scaleout · num_scaleup == num_ranks`（hybrid）与 `(1, num_ranks)`（单级）。

---

## 4. GIN 设备通信器需求：`ncclDevCommRequirements_t` 逐字段

（`context.cpp:121-155`）

`Context` 构造时先问一次 `ncclCommProperties`（`:116-117`），然后**按需**（`EP_DISABLE_GIN==0`）
组装 GIN 需求。每个字段都有明确动机：

| 字段（context.cpp 行） | 值 | 为什么 |
|---|---|---|
| `lsaMultimem`（`:123`，在 `if` 外） | `enable_lsa_multimem` | 决定要不要 `ncclGetLsaMultimemDevicePointer`（`:203`）拿多播指针 |
| `ginType`（`:134`） | `NCCL_GIN_TYPE_GDAKI` | 只支持 GDAKI（Host-Initiated？不，是 GPU-Initiated 的 GDAKI 变体）；`:129-131` 断言 `props.ginSupport[GDAKI]` |
| `ginContextCount`（`:135`） | `num_allocated_qps` | 每 rank 分配的 QP 数（第 73 篇的 SM/QP 模型给出） |
| `ginExclusiveContexts`（`:136`） | `true` | 每 context 独占，不与其他通信共享 |
| `ginQueueDepth`（`:137`） | `qp_depth > 0 ? qp_depth : 1024` | 默认 `kDefaultQPDepth`（`compiled.cuh:81`） |
| `ginSignalCount`（`:140`） | `num_ranks + 2*2` | **定制 RDMA barrier 需要额外信号**：每 rank 一个 + 4 个（注释 `:139`） |
| `ginCustomStride` / `ginConnectionType`（`:141-146`） | hybrid 且有 storage → `CUSTOM_STRIDE`；hybrid → `RAIL`；否则 `FULL` | 三级连接拓扑：跨轨(custom stride) > 逐轨(rail) > 全连接(full) |
| `ginTrafficClass`（`:149-154`） | 三级优先级：`EP_DEFAULT_RDMA_SL` < `sl_idx` 参数 < `EP_OVERRIDE_RDMA_SL` | 后设置者胜（`:149/151/153` 顺序覆盖） |

两个「门」也在这里：hybrid 模式要求 `props.railedGinType != NONE`，否则报「网络配置问题，比如
多平面网络里 `allow_hybrid_mode=0`」（`:125-128`）；`props.ginSupport[GDAKI]` 必须为真（`:129-131`）。

**`ginSignalCount = num_ranks + 4`** 是第 63 篇「免复位屏障」的原点：`num_ranks` 个 rank 各占
一个信号槽，另外 4 个留给协议本身（第 63 篇 `barrier.cuh` 的双相位计数器）。

---

## 5. NCCL 运行时版本闸门：本机「阻塞」的机制级根因

（`context.cpp:156-164`）

```cpp
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 0)
    reqs.useRuntimeVersion = true;                                  // :157
    dev_comm.ptr = malloc(props.devCommRuntimeVersionSize);         // :158
#else
    EP_HOST_ASSERT(NCCL_VERSION_CODE == nccl_runtime_version and    // :160
        "Prior to NCCL 2.31, NCCL compile-time and runtime versions must be the same. "
        "Please re-compile DeepEP.");
    dev_comm.ptr = malloc(sizeof(ncclDevComm_t));                   // :161
#endif
    NCCL_CHECK(ncclDevCommCreate(comm, &reqs, dev_comm.ptr));       // :164
```

这段是**编译期宏**（`NCCL_VERSION_CODE`，来自 `nccl.h`）与**运行期查询**（`nccl_runtime_version`，
`ncclGetVersion`，`:101-102`）的交汇点：

- **编译期 ≥ 2.31**：设备通信器大小**不再是编译期常量**，改用运行期查询到的
  `props.devCommRuntimeVersionSize`（`:158`），并置 `reqs.useRuntimeVersion = true`——于是
  **编译期与运行期的 NCCL 版本可以不同**。本机 DeepEP 编译用的是 pip 的
  `nvidia-nccl-cu13 2.32.3` 头文件，走的就是这条分支。
- **编译期 < 2.31**：`ncclDevComm_t` 是固定布局，必须 `NCCL_VERSION_CODE == nccl_runtime_version`，
  否则当场 abort（`:160`）。

**为什么这就是「阻塞」**：本机 **DeepEP 编译期分支（pip 头 2.32.3）本身不要求运行时版本相等**
——所以 `context.cpp` 这一层在 2.29.7 上不会 abort。真正的闸门在**更外层**：
`deep_ep/__init__.py:19-41` 的 `check_nccl_so()` 在 import 阶段做的是**二进制逐字节比对**
（`filecmp.cmp(loaded, linked)`），而 PyTorch 加载的是系统的 `/lib/x86_64-linux-gnu/libnccl.so`
（2.29.7），DeepEP 期望的却是 pip 的 `.../nvidia/nccl/lib/libnccl.so.2`（2.32.3）——
两者不等 → `Invalid NCCL versions` abort。**这是两个独立的闸门**：

1. **Python 引导闸门**（`check_nccl_so`，二进制相等）——本机实际触发处；
2. **C++ 设备通信器闸门**（`context.cpp:156-162`，版本号相等，仅 <2.31 编译时生效）。

H100 实测（out.txt §4）用 ctypes 直接调两个真实 `libnccl` 的 `ncclGetVersion`：system
`2.29.7 (22907)`、pip `2.32.3 (23203)`；并复刻两分支判据——`pip 头` 走 runtime 分支且**不要求
运行时相等**（`ok=True`），而**假设**用 `2.30.0` 头对 `2.29.7` 运行时会落进 assert 分支被拦下
（`ok=False`）。两条 `check_nccl_so` 二进制比对（system vs pip 不等、system vs system 相等）
也逐条复刻通过。

---

## 6. 域分解、对称窗口与存储窗口的字节布局

（`context.cpp:166-224`）

设备通信器建好后才**真正知道** NVLink 域大小（`:167-169` 从 `ncclTeamLsa` 拿
`num_nvl_ranks/nvl_rank_idx`），于是建立三条恒等式（`:170-172`）：

$$
\text{num\_rdma\_ranks} = \frac{\text{num\_ranks}}{\text{num\_nvl\_ranks}},\qquad
\text{rank\_idx} = \text{rdma\_rank\_idx}\cdot\text{num\_nvl\_ranks} + \text{nvl\_rank\_idx}
$$

即「rank id 是 `(rail, lane)` 的**行主序**编码」。hybrid 与否只决定**逻辑域**（`:174-183`），
`is_scaleup_nvlink = (num_scaleup_ranks == num_nvl_ranks)`（`:183`）随后用于 kernel 里选
NVLink 还是 RDMA 通路。

**对称窗口**（`:195-205`）：

```
symmetric_memory = (main_context ? main_context->symmetric_memory        // :192 共享
                                 : symmetric::alloc(workspace+gpu_buffer, storage, ...));  // :186-190
ncclCommWindowRegister(comm, symmetric_memory->ptr, num_sym_bytes, &window, STRICT_ORDERING);  // :199
ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace);            // :200  本 rank 的窗口基址
buffer = advance_ptr(workspace, num_workspace_bytes);                    // :201  buffer = workspace + W
if (multimem) ncclGetLsaMultimemDevicePointer(window, num_workspace_bytes, &buffer_multimem);  // :203
if (!main_context) cudaMemset(workspace, 0, num_workspace_bytes);        // :205  仅 owner 清零
```

一句注释点明了关键：**`ncclCommWindowRegister` 是集合操作**，内部已经
`bootstrapBarrier` 了所有 rank，不需要外部再屏障（`:196-197`）。窗口内的布局是**链式加法**：
`workspace` 在偏移 0，`buffer` 紧跟其后（偏移 `num_workspace_bytes`），`buffer_multimem`
指向同一段的**多播视图**。注意 `cudaMemset` 只有 `main_context == nullptr`（owner）才做
（`:204-205`）——共享同一内存的从 `Context` 不重复清。

**RDMA storage 窗口**（`:207-223`）：当 `num_rdma_storage_bytes > 0` 时，主段之后还有一段
storage，**按 `gin_min_stride` 切片、每片一个窗口**：

```cpp
storage_base  = advance_ptr(sym->ptr, sym->num_sym_bytes);                       // :208 主段之后
local_storage = advance_ptr(storage_base, (rank_idx % gin_min_stride) * bytes);  // :209 本 rank 用哪片
for i in [0, gin_min_stride): ncclCommWindowRegister(comm, storage_base + i*bytes, ...,
                                                     &rdma_storage_windows[i], GIN_ONLY);  // :211-217
cudaMemcpy(rdma_storage_windows_gpu, rdma_storage_windows.data(), ...);         // :219-222 拷到 device
```

「每片一个 `NCCL_WIN_GIN_ONLY` 窗口」正是 hybrid 模式下「RDMA 可以命中任意 rail 的 storage」
的落点；`local_rdma_storage_ptr` 用 `rank % gin_min_stride` 选自己哪片（`:209`）。H100 实测
（out.txt §2）把这段 `advance_ptr` 串行加法在**真实头文件**下编译运行，与 Python 复刻逐字节
一致（workspace=0、buffer=`+num_workspace_bytes`、storage windows = `storage_base + i·bytes`、
local = `storage_base + (rank%stride)·bytes`）。

---

## 7. 偏移翻译 `get_sym_ptr` 与拆除 `finalize`

**`get_sym_ptr`**（`context.cpp:226-232`）是「对称寻址」的全部魔法：

```cpp
const auto offset = (uint8_t*) ptr - (uint8_t*) workspace;      // :229  本 rank 的窗口内偏移
ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx, &dst_ptr);  // :230  翻译成目标 rank 同偏移
```

「对称」的含义就是：**只要每个 rank 的缓冲区内部布局一致，同一个 `offset` 在每个 rank 都指向
同一逻辑位置**。于是「本 rank 的指针」减去 `workspace` 得到逻辑偏移，再让 NCCL 用目标 rank 的
窗口基址加回去。`barrier_signals` 就是调用方用 `set_barrier_signals`（`api.hpp:83-85`）挂进
`workspace` 里的某段。

**`finalize()`**（`:244-266`）的拆除顺序是**构造的严格逆序**：

1. `cudaFree(rdma_storage_windows_gpu)`（`:246-249`）；
2. 逐个 `ncclCommWindowDeregister` 掉 **storage 窗口**（`:250-252`）；
3. 注销**主窗口** `window`、`symmetric_memory.reset()`（`:255-257`）——先 deregister 再放内存，
   否则窗口指向已释放内存；
4. `barrier_signals = nullptr`（`:259-260`，注释「屏障不能再用了」）；
5. `ncclDevCommDestroy` + `free(dev_comm.ptr)`（`:262-264`）；
6. `finalized = true`（`:265`）。

`get_physical_domain_size()/get_logical_domain_size()` 的访问器都带 `EP_HOST_ASSERT(not finalized)`
（`:235/240`），所以 `finalize()` 之后再用上下文会立刻报错——这是**故意的**：
`finalized` 就是「这个对象失效了」的显式状态。

---

## 8. 与「阻塞」的对接，以及本篇没覆盖的部分

- **本机阻塞的准确落点**：不是 `context.cpp` 的版本闸门（pip 头 2.32.3 走 runtime 分支、
  不要求相等），而是 `deep_ep/__init__.py:19-41` 的 `check_nccl_so()` **二进制比对**：
  PyTorch 加载系统 2.29.7、DeepEP 期望 pip 2.32.3，两者文件不同 → `Invalid NCCL versions`。
  要跑通 DeepEP 端到端，需要一个 **PyTorch 也链接 pip 2.32.3** 的环境（见 ROADMAP「阻塞」）。
- **仅分析**：`ncclDevCommCreate` / `ncclCommWindowRegister` / `ncclGetLsaDevicePointer` /
  `ncclCommWindowDeregister` / `ncclDevCommDestroy` 这些 NCCL 设备 API 本机 2.29.7 不提供，
  无法真机验证；本篇只把它们在 `context.cpp` 里的**调用顺序与参数**逐行钉住。
  真机验证的是它们的**全部 host 算术前置**（域分解恒等式、signal 预算、偏移布局、版本判据）。

---

## 9. 实测汇总（H100 sm90，`harness/exp_deepep_context.{sh,out.txt}`）

| 组 | 内容 | 结果 |
|---|---|---|
| §0/§1 | 真实头文件常量（对齐/上限/`sizeof(topk_idx_t)`）python 解析 vs CUDA 直读 | 6/6 PASS |
| §2 | `advance_ptr` 布局（workspace/buffer/storage windows/local） | 8/8 PASS |
| §3 | 域分解 60 行逐字段 + 逻辑域乘积不变量 | 2/2 PASS |
| §4 | NCCL 版本闸门四组合 + `check_nccl_so` 两条二进制比对 | 6/6 PASS |
| §5 | `NCCLCommHandle` 缓存/托管三分支 | 3/3 PASS |

总计 **PASS=29 / FAIL=0**。设备侧 NCCL API 与端到端 dispatch/combine 因 NCCL 版本受限，仅分析。

---

## 10. 参考资料

- DeepEP 仓库：https://github.com/deepseek-ai/DeepEP （本地 `93eb6eb`，v2.5.0）
- DeepEP V2.5 PR：https://github.com/deepseek-ai/DeepEP/pull/763
- DeepGEMM（本专题主线）：https://github.com/deepseek-ai/DeepGEMM
- 上游相关篇目：本专题第 7 篇（dispatch/combine 数据流）、第 53/63 篇（屏障与句柄）、
  第 66 篇（对称内存分配）、第 73 篇（SM/QP 模型）、第 86 篇（NCCL 绑定与引导层）。
