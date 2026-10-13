# 66 · DeepEP 的对称内存分配层：GPU/CPU 弹性段、跨进程 shard 与 copy-engine 窗口的地基

> 本专题第 7 / 48 / 51 / 52 / 53 / 58 / 59 / 61 / 63 / 64 篇一直在讲 DeepEP 的 dispatch / combine /
> bucket kernel 与它们调用的 `comm::barrier`、`NCCLGin::get_sym_ptr`。这些全都默认「有一块**对称内存**」——
> 每个 rank 上布局完全一致、且能通过对称偏移互相寻址的一段显存。**这块内存是谁分配的、怎么在 GPU 与
> CPU 之间弹性拼接、怎么跨进程把对端的 RDMA storage shard 映射进本地 VA**，从没系统讲过。
> 本篇补上这层地基：`csrc/kernels/comm/symmetric.hpp`（332 行）+ `deep_ep/buffers/allocator.py`。
>
> **这一层是纯 host-side 的 CUDA VMM（`cuMemCreate/Map/SetAccess/Export/Import`）+ NCCL 主机 API
> （`ncclMemAlloc`）+ Linux `pidfd`，与 UMMA / tcgen05 / GIN 设备 API 无关，故 H100(sm90) 真机可跑。**
> 本实验**直接 `#include` 真实头文件**调用真实类，55 项设备侧断言 + 23 项 Python 断言全 PASS。
> 依赖 NCCL 设备 API 的 window/GIN 部分（`ncclCommWindowRegister`、`ncclGetLsaDevicePointer`、
> `ncclGetLsaMultimemDevicePointer`）**不在本层**，仅源码分析。

实验驱动：`harness/exp_deepep_symmetric_alloc.sh`（编译 `src/deepep_symmetric_alloc.cu`，
`sha256(symmetric.hpp)=fa24a2333e387330`，与 `~/github/DeepEP@93eb6eb` 一致）+ `harness/exp_deepep_symmetric_alloc.py`。
原始输出：`harness/exp_deepep_symmetric_alloc.out.txt`。

---

## 1. 这一层在栈里的位置

```
Python:  BufferAllocator(allocator.py)  -- 规划 2MiB 对齐的连续张量布局
             │
Buffer 基类: BufferBase(base.hpp)        -- main_context / print_memory_usage
             │
设备段分配:  symmetric::alloc(symmetric.hpp:302-330)   <-- 本篇
             │   ├─ GPUSymmetricMemory            (:120-135) ncclMemAlloc
             │   ├─ ElasticSymmetricMemory        (:139-190) CUDA VMM，GPU 段 + storage 段连续 VA
             │   └─ HybridElasticSymmetricMemory  (:193-300) 再 import 同节点对端 shard（pidfd）
             │
NCCL 窗口:   ncclCommWindowRegister / ncclGetLsaDevicePointer(context.cpp:199-200)  <-- 仅分析
             │
设备句柄:    NCCLGin::get_sym_ptr(handle.cuh，见第 63 篇)  -- 把本 rank offset 翻译成目标 rank 同 offset
```

一句话：`symmetric.hpp` 负责**分配 + 映射 + 导出/导入物理段**；`context.cpp` 把它注册成 NCCL window
并取出「LSA（同 NVLink 域）设备指针」；设备端 `get_sym_ptr` 才做对称偏移翻译。没有这一层，第 63 篇的
`red_add_rel` / `ld_acq_sys` 全都无从谈起。

---

## 2. 对齐常量：为什么是 2 MiB

`deep_ep/include/deep_ep/common/compiled.cuh:67-71`：

```cpp
constexpr int kNumTMAAlignmentBytes        = 32;
constexpr int kNumRDMAAlignmentBytes       = 64;
constexpr int kNumAllocationAlignmentBytes = 2 * 1024 * 1024;   // 2 MiB
static_assert(kNumRDMAAlignmentBytes % kNumTMAAlignmentBytes == 0);
static_assert(kNumAllocationAlignmentBytes % kNumRDMAAlignmentBytes == 0);
```

三层对齐各管一段：**32 B** 是 TMA（`cp.async.bulk.tensor` 的地址/长度粒度）；**64 B** 是 RDMA
（NCCL 网络的 wr 原子性）；**2 MiB** 是**整个对称段的物理分配粒度**——它必须整除 CUDA VMM 的
recommended granularity（`symmetric.hpp:72` 的断言），否则 `cuMemMap` 会对不齐。这三个常量经
`csrc/python_api.cpp:24` 的 `get_num_allocation_alignment` 暴露给 Python（`allocator.py:14` 消费）。

**H100 实测**：`granularity recommended=2097152 minimum=2097152`（GPU 与 HOST_NUMA 两种 prop 都是 2 MiB），
恰好等于 `kNumAllocationAlignmentBytes`——所以这条断言在本机不是「恰好通过」而是「设计上钉死」。

---

## 3. `DeviceContext`：分配属性怎么构造（`:21-75`）

`DeviceContext`（`:21-30`）一次性缓存三件事：`device_idx`（`cudaGetDevice`）、`device`（`cuDeviceGet`）、
`numa_idx`（`cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID)`）。后两个是为「CPU 弹性段」准备的——
展宽到 host NUMA 节点才能分配 CPU 侧的 pinned 内存。**H100 实测 `host_numa=0`**。

`build_alloc_prop(:41-75)` 是核心，做四件事：

1. **`type = PINNED`**，`location.type/id` 由调用方选 DEVICE 还是 HOST_NUMA（`:43-45`）；
2. **句柄类型**：基础是 `POSIX_FILE_DESCRIPTOR`；若 `CUDA ≥ 13` 且设备支持，再或上 `FABRIC`
   （`:47-56`）。FABRIC 是 MNNVL（跨节点 NVLink 域）用的句柄，普通单机不需要；
3. **GPUDirect RDMA**（仅 DEVICE 侧，`:58-67`）：查 `GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED`，
   置 `allocFlags.gpuDirectRDMACapable = 1`，否则**直接断言失败**——因为 EP 的对称段要被网卡直接读写；
4. **粒度检查**（`:69-73`）：`cuMemGetAllocationGranularity(RECOMMENDED)` 必须整除 2 MiB。

**H100 真机数据**：`fabric_supported=1, gpu_direct_rdma_vmm=1`。即本机**声称支持 FABRIC**——这一点在第 4 节
会引出一个真坑。

---

## 4. `cumem_create_with_fallback`：镜像 `ncclMemAlloc` 的回退（`:79-93`）

```cpp
if (prop->requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC) {
    CUresult err = lazy_cuMemCreate(handle, num_bytes, prop, 0);
    if (err == CUDA_ERROR_NOT_PERMITTED or err == CUDA_ERROR_NOT_SUPPORTED) {
        prop->requestedHandleTypes &= ~CU_MEM_HANDLE_TYPE_FABRIC;   // 清掉 FABRIC 重试
        CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0));
    } else {
        CUDA_DRIVER_CHECK(err);
    }
} else {
    CUDA_DRIVER_CHECK(lazy_cuMemCreate(handle, num_bytes, prop, 0));
}
```

注释「mirrors ncclMemAlloc logic」点明意图：**FABRIC 句柄是「可选项」**，设备属性说支持 ≠ 运行期
真的允许。于是先在 prop 里带上 FABRIC，`cuMemCreate` 若返回 `NOT_PERMITTED/NOT_SUPPORTED`，
就**就地清位重试**。

**H100 实测（本实验最值得记的一个真坑）**：第 1 节里 `fabric_supported=1`，可当本实验**强制**请求
FABRIC 句柄时，`cuMemCreate` 仍失败并走了回退分支——最终打印 `requestedHandleTypes = 0x1`
（=`POSIX_FILE_DESCRIPTOR`，FABRIC 位 `0x4` 被清掉）。也就是说：**属性报告支持、实际创建被拒**，
回退逻辑是真实生效的，不是摆设。这正是为什么这段代码不能简化成「查属性就决定」。

---

## 5. 三种分配器

`alloc(:302-330)` 按参数派发：

```cpp
if (num_rdma_storage_bytes > 0) {
    if (allow_hybrid_mode)  HybridElasticSymmetricMemory(...)   // :314
    else                    ElasticSymmetricMemory(...)          // :318
} else                      GPUSymmetricMemory(num_sym_bytes)    // :322
```

### 5.1 `GPUSymmetricMemory`（`:120-135`）——纯 GPU 默认路径

直接包 `ncclMemAlloc/ncclMemFree`。构造时断言 `num_sym_bytes` 是 2 MiB 的倍数（`:123`），
并**二次确认返回指针 2 MiB 对齐**（`:125`）。**H100 实测**：`ncclMemAlloc` 返回的指针
（`0x7f9198800000`）严格 2 MiB 对齐，kernel 铺/读一致；`非 2 MiB 倍数`与`0 字节`两种输入都被
`:123` 断言拒绝。这是 EP buffer 的常规路径（`ep.hpp:74` 传空的 `shared_comm`、`num_rdma_storage_bytes=0`）。

### 5.2 `ElasticSymmetricMemory`（`:139-190`）——GPU 段 + storage 段共享一段连续 VA

这是「弹性」的精髓。它把**两种物理 backing**（主对称段 + RDMA storage 段）拼进**同一段连续 VA**：

1. `cuMemAddressReserve(num_sym_bytes + num_rdma_storage_bytes, 2 MiB)`（`:160`）——只保留地址空间，
   2 MiB 对齐；
2. `cumem_create_with_fallback` + `cuMemMap` **主段**（`:163-164`）：`use_cpu=false` 时是 GPU 显存，
   `gpu_prop` 带 GPUDirect RDMA；
3. `cumem_create_with_fallback` + `cuMemMap` **storage 段**贴在主段之后（`:166-168`）：
   `use_cpu_rdma_storage ? cpu_alloc_prop(true) : gpu_alloc_prop(true)`（`:153`，注意 `skip_fabric=true`）；
4. `set_access`（`:170-176`）：整段（或分段）设为 READWRITE。

**为什么必须连续 VA**：dispatch/combine 用**一个基址 + 对称偏移**做寻址（`get_sym_ptr` 的
`offset = ptr - workspace`）。主段与 storage 段若在物理上分属 GPU / CPU，只要 VA 连续，设备端就
只看到「一段内存」，无需区分；`HybridElastic` 再进一步把对端的 shard 也贴进来。

**H100 实测**：两种 backing（GPU / CPU-NUMA）都成功；返回指针 2 MiB 对齐；主段铺 `0x5a`、storage 段
铺 `0xa5` 后**同时持有不同值**——证明两段物理 backing 独立、映射未重叠。

### 5.3 `HybridElasticSymmetricMemory`（`:193-300`）——把对端 shard 映射到本地

hybrid 模式下，每个 scaleup 域内 rank 的 storage shard 都要能在本 rank 直接寻址。构造（`:201-262`）：

- VA 预留 `num_sym_bytes + num_rdma_storage_bytes * num_scaleup_ranks`（`:215-219`）；
- 先 map 自己的 GPU 主段（`:224-225`）；
- 对域内每个 rank 的 `(pid, fd)`（`:229-252`）：
  - **本进程就用自己的 fd**（`:231`）；否则 `pidfd_open` + `pidfd_getfd` 把**对端进程的 fd 复制到本地**
    （`:232-238`），从而 `cuMemImportFromShareableHandle` 能拿到对端的物理段；
  - import 后 map 到 `num_sym_bytes + i * num_rdma_storage_bytes`（`:240-245`）——**按 rank 顺序依次贴**，
    所以「第 i 个 rank 的 storage」在对称偏移上位置固定；
  - **本进程 fd 存进 `local_export_fd`，在析构 `:270-271` 里 close**（保持物理内存活）；
- `set_access` 整段（`:254-261`）。

**H100 实测**：造 3 个本进程自己的 shard 当作「intra-node peers」传入（全走 local_fd 直用分支，不经过
pidfd），`alloc` 成功、指针 2 MiB 对齐、主段 + 3 个 shard 各自 kernel 铺/读一致、shard 写入不污染主段。
这条验证了**多 shard 连续映射 + 偏移布局**；跨进程的 pidfd 分支另见第 6 节。

---

## 6. `create_shared_handle` 与跨进程 `pidfd`

`HybridElasticSymmetricMemory::create_shared_handle(:281-299)` 是「发布一个可共享的物理段」：

1. `cumem_create_with_fallback`（`:289`）；
2. `cuMemExportToShareableHandle(&fd, handle, POSIX_FILE_DESCRIPTOR, 0)`（`:292-293`）；
3. **`cuMemRelease(handle)`**（`:296`）——关键注释「the POSIX FD keeps the physical memory alive」：
   分配句柄释放后，只要 fd 还开着，物理内存就不回收。所以返回值是 `(getpid(), fd)`。

**H100 实测**：
- **本进程导出→导入**（`:281-299` + 手动 import/map/set_access）：GPU 与 CPU 两种 storage 都成功，
  导入段 kernel 铺/读 `0x3c` 一致；`handle.pid == getpid()`。
- **真·跨进程**（`fork` + `exec` 子进程）——这正是 `pidfd_open`/`pidfd_getfd` 路径（`:232-238`）的真身：
  - 子进程 `create_shared_handle` → 映射自己的段 → 写 `MAGIC_CHILD` → 向 stdout 公布 `(pid, fd)`；
  - 父进程 `pidfd_open(child)` + `pidfd_getfd(pidfd, child_fd)` 拿到**对端进程 fd 的本地副本**，
    import + map + set_access；
  - 父读回 `MAGIC_CHILD`（跨进程同物理内存可见），写 `MAGIC_PARENT`，子进程序读回后 `exit(0)`。
  - **全部 PASS**。

> **实验方法的一个真坑**：最初用 `fork()` 直接复制父进程（已初始化 CUDA）去调 CUDA 会崩。正确做法是
> `fork` 后**立刻 `exec` 本二进制**（父进程用 `--child-publish` 模式），让子进程从干净状态 `cudaSetDevice`。
> 这和 cross-process 的真实 DeepEP 场景一致：各 rank 本来就是独立进程。

---

## 7. `alloc()` 的收尾与 NCCL 环境变量

`alloc(:302-330)` 末尾（`:325-328`）：

```cpp
if (num_rdma_storage_bytes > 0 and use_cpu_rdma_storage)
    setenv("NCCL_ELASTIC_BUFFER_REGISTER", "1", 0);
```

注释 `TODO: move all variables into NCCL runtime` 说明这是过渡手段：**只要分配器可能产出 CPU-backed
段，就开 NCCL 的 elastic buffer register**，让 NCCL 知道窗口里混了可弹性回收的内存。这是 host 段与
NCCL 注册之间的耦合点。

---

## 8. `context.cpp` 如何消费（仅供分析的部分）

`symmetric.hpp` 产出的 `SymmetricMemory` 在 `csrc/kernels/comm/context.cpp` 里被进一步加工：

- `symmetric::alloc(num_workspace_bytes + num_gpu_buffer_bytes, num_rdma_storage_bytes, ...)`
  （`:186-190`），hybrid 时传 `gin_min_stride` / `rank_idx / gin_min_stride` 作 scaleup 参数；
- `ncclCommWindowRegister(comm, raw_window_ptr, num_sym_bytes, &window, NCCL_WIN_STRICT_ORDERING)`
  （`:199`）——**集体调用**（注释 `:196-198` 说明它内部会 `bootstrapBarrier`，故无需额外 barrier）；
- `ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace)`（`:200`）取本 rank 在 NVLink 域内的
  设备指针；`enable_lsa_multimem` 时再取 `buffer_multimem`（`:203`，多播地址）；
- RDMA storage 被切成 `gin_min_stride` 个 window（`:207-221`），本 rank 的基址由
  `(rank_idx % gin_min_stride) * num_rdma_storage_bytes` 定位（`:209`）；
- `get_sym_ptr`（`:226-231`）就是第 63 篇 `NCCLGin::get_sym_ptr` 的 host 端底座：
  `offset = ptr - workspace` → `ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx, &dst_ptr)`；
- `finalize`（`:244-265`）：`cudaFree` 窗口副本 → `ncclCommWindowDeregister` → `symmetric_memory.reset()`
  （触发展宽内存的真正释放）。

**这些 `ncclCommWindowRegister` / `ncclGetLsa*` / GIN 是 NCCL 设备/注册 API，本机仅 2.29.7、
无 GIN 设备支持，故只做源码分析**，不在 H100 实验里跑。

---

## 9. Python 侧：`BufferAllocator`（`deep_ep/buffers/allocator.py`）

`symmetric.hpp` 分配的是「一整块连续 storage」；上层怎么把它切成若干带类型的张量？答案是
`BufferAllocator` 的**先规划、后落地**两步：

- `allocate(shape, dtype)`（`:19-26`）：`torch.empty(..., device='meta')` 造占位张量，记录
  `offset = align(num_bytes, alignment)`，再把 `num_bytes` 推到 `align(offset + nbytes, alignment)`。
  即每段起点都 2 MiB 对齐，段与段之间可能留 padding；
- `materialize(storage)`（`:28-36`）：要求 `storage` 是 CUDA/contiguous/uint8 且大小相等，然后对每个
  占位张量 `storage.narrow(offset, nbytes).view(dtype).view(shape)` 并用
  `torch.utils.swap_tensors` 换掉占位张量的底层——**所有张量共享同一块 storage**。

**H100 实测（加载真实源码运行）**：
- 对齐/规划与独立 oracle 逐项一致：4 段得到偏移 `[0, 2097152, 4194304, 6291456]`、总字节 8 MiB；
- materialize 后每个张量的 `data_ptr()` **精确等于** `storage.data_ptr()+offset`（真·共享内存），
  写 int32/uint8 张量后 raw storage 字节同步；
- 负例全部被断言拒绝：materialized 后再 allocate、storage 大小/dtype/设备不符、二次 materialize；
- 真实 MoE/EP 规模一次规划：4 段（含 `(1024,7168) bf16`）总分配 33554432 B ≥ 各段和 29361664 B，
  每段仍 2 MiB 对齐。

**设计意图**：为什么不在 2 MiB 对齐的 storage 里直接 `torch.empty`？因为 `torch.empty` 的分配器
**只保证 512 B 对齐**，不满足 RDMA/TMA 需求；`BufferAllocator` 用「一次 2 MiB 对齐的大 storage +
meta 占位 + swap」精确控制每段偏移，正是 `EPBuffer`/`EngramBuffer` 构造出 `context->buffer` /
`context->workspace` 这批视图的方式。

---

## 10. 实验结论（H100 真机，sm90a）

| 组 | 覆盖 | 结果 |
|---|---|---|
| Part 1 | `DeviceContext` / `build_alloc_prop`：属性 5 项 + FABRIC 位守恒 + 粒度断言 + NUMA | PASS |
| Part 2 | `cumem_create_with_fallback`：POSIX-only 分支 + 强制 FABRIC 回退（实测清位重试） | PASS |
| Part 3 | `ElasticSymmetricMemory`：GPU / CPU-NUMA 两种 backing，连续 VA，两段独立 | PASS |
| Part 4 | `create_shared_handle`：GPU / CPU 导出→导入，本进程再映射 | PASS |
| Part 5 | `HybridElasticSymmetricMemory`：3 shard 连续映射 + 偏移布局 | PASS |
| Part 6 | 跨进程 `pidfd_open`/`pidfd_getfd`：子写父读、父写子读 | PASS |
| Part 7 | `GPUSymmetricMemory`（`ncclMemAlloc`）+ 2 MiB 断言负例 | PASS |
| Python A–E | 常量/规划/落地/契约/真实规模（真实 `allocator.py` 源码） | PASS |

**设备侧 PASS=55 / Python PASS=23 / 0 FAIL**（见 `.out.txt`）。关键真机数字：
`host_numa=0`、`fabric_supported=1 但强制 FABRIC 创建被拒并回退到 0x1`、`granularity=2 MiB`、
`ncclMemAlloc` 指针 2 MiB 对齐、跨进程 magic 往返一致。

---

## 11. 边界与「仅分析」

- **仅分析**：`ncclCommWindowRegister` / `ncclGetLsaDevicePointer` / `ncclGetLsaMultimemDevicePointer`、
  RDMA storage 的 `NCCL_WIN_GIN_ONLY` 注册、以及真实多 rank 下的 scaleout/hybrid 编排——都是 NCCL
  注册/GIN 设备 API，本机 NCCL 2.29.7 不提供 GIN 设备支持（同第 53/63/64 篇的阻塞）。
- **本机数值局限**：跨进程 part 6 在同一台机器上跑「两个独立进程」，验证的是 **fd 传递 + 同物理内存
  可见性**这条机制（与真实多 rank 的 import 路径同构）；真实多节点/多卡 NVLink 域内的映射还需多 rank
  环境。
- **未触及**：`NCCL_ELASTIC_BUFFER_REGISTER` 对 NCCL 行为的实际影响需真实 NCCL 运行期才能量化。

**与专题主线的连接**：Mega MoE 的 dispatch/combine 之所以能「一个基址 + 对称偏移」直接读写远端
rank 的 token/SF/combine buffer，前提正是这里分配出的**布局一致、跨进程可寻址的对称段**。至此，
从 Python 调用 → host 对称内存分配 → NCCL 窗口 → 设备端 `get_sym_ptr` 的整条「对称内存」链路都已讲清。
