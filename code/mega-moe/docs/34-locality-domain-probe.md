# 34. SM 局部域（locality domain）探针与内存域分配

> 本文对应路线图 **P24-1**。第 11 篇（dispatch/combine）和第 03/09 篇（Mega MoE 主体 kernel）都反复出现
> `num_locality_domains = 2`：调度器把集群**按域分区**，少量任务时**不跨域偷取**，权重也被
> `localize` 到各自的域。但「**哪个 SM 属于哪个域、这件事是怎么测出来的**」一直没讲。
> 本篇补上这个前置步骤：一个自包含的 **pointer-chase 延迟探针** + **本地化内存分配**（MLOPart / CUDA VMM）。
>
> 它服务的对象很具体：DeepSeek 目标卡上 SM 到 HBM 的**距离并不均匀**——同一物理 die 内近、跨 die 远
> （第 11 篇实测远端 NVLink 带宽 ≈ 本地的 1/8）。所以 kernel 希望「就近算、就近取权重」。
> DeepGEMM 不依赖固定的 SM→域映射，而是**开机/首次调用时现场探针标定**，标不出稳定映射就退化为均匀映射。
>
> **架构无关，H100 真机可跑**：探针设备代码 `sm100_locality_domain.cuh` 只用 `%smid` +
> `ld.global.cg` + `clock64`（文件名叫 sm100 只是 DeepGEMM 2.x 只发 SM100 版），host 侧
> `csrc/apis/locality_domain.hpp` 的 probe/try_probe/balance 全是纯张量逻辑。
> 本机 **MLOPart 不可用**（需 MPS `-mlopart` + 双 device instance），所以「远端域」用
> host-pinned mapped memory 模拟，验证探针**能检出并稳定划分**。
>
> 行号核对：DeepGEMM `057ca59`（v2.8.1）。实验：`src/locality_domain_probe.cu`、
> `harness/exp_locality_domain_probe.{sh,out.txt}`。

与其它篇的关系：第 03/09/10 篇讲 Mega MoE 主体 kernel 的 warp/mbarrier/MMA；第 11 篇讲 dispatch/combine
和 NVLink 访存实测；第 18 篇讲选核启发式。这些都属于「**怎么算**」。本篇讲一条更底层的**数据放置前提**：
「**把哪些权重放在 SMs 附近**」，即内存与 SM 的域归属如何被决定。

---

## 1. 为什么 MoE 需要「局部域」

### 1.1 目标卡的 SM↔HBM 不均匀性

目标卡（及 H100/H200 这一代）的 HBM 由多个显存分区/栈组成，SM 到不同分区的延迟与带宽并不相同。
第 11 篇在 H100 上量到：**远端（跨 NV18 平面）读/写带宽 ≈ 本地的 1/8、裸延迟 ~2.83×**。
把权重放在「远」的分区，会让每个 CTA 的 TMA 载入变慢（Mega MoE 的 L1/L2 权重是逐 k-block 流水载入的，
见第 03/09 篇），直接拖慢重叠流水。

### 1.2 两个域：MPS 双 device instance / CUDA 13.4 的 locality domain

DeepGEMM 把「域」建模成固定的 **2 个**（`deep_gemm/include/deep_gemm/common/types.cuh:54`）：

```cpp
static constexpr uint32_t kNumDeviceLocalityDomains = 2;
```

- **CUDA 13.4 之前**：用 **MPS 的 MLOPart** 把一张物理卡切成 **2 个 device instance**，每个 instance
  暴露一半的 SM 和**一半的 HBM**（`nvidia-cuda-mps-control ... start_server -mlopart`）。
  代码里的注释明确写着这是「CUDA 13.4 之前的临时办法」
  （`csrc/runtime/locality_domain.hpp:13`、`csrc/runtime/mlopart.hpp:13`）。
- **CUDA 13.4 起**：改用正统的 `CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`
  / `CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT`（`csrc/runtime/locality_domain.hpp:35, 52` 的 `TODO`），
  以及 `cuMemCreate` 直接按域分配。两个 `TODO` 就是这两处升级点。

无论哪种后端，**SM→域** 的映射都要靠探针现测：因为 MPS instance、物理 die、BIOS/拓扑都可能变，
写死一张表不可靠。

### 1.3 全链路：从域标定到 kernel 内部

```
get_balanced_sm_locality_domains()  (host, 缓存)
        │  (uint8 per SM)
        ▼
csrc/apis/mega_moe.hpp:266 / :373          # 作为参数传给 mega_moe / bf16_mega_moe
        ▼
sm100_fp8_fp4_mega_moe.hpp:26, :221        # sm_locality_domains.data_ptr<uint8_t>()
        ▼
sm100_fp8_fp4_mega_moe.cuh:79, :945        # scheduler.mainloop(num_tokens, sm_locality_domains)
        ▼
scheduler/mega_moe.cuh:440                 # sm_locality_domain_idx = sm_locality_domains[sm_idx]
```

调度器里域的作用（`scheduler/mega_moe.cuh`）：

- `kNumLocalityDomains` 与 `kNumDeviceLocalityDomains` 一致性、以及「每个域拿整数个 cluster」的静态断言
  （`:226-228`）；
- **任务计数按域分开**：`get_next_local_task_idx` 在 `task_count_ptr + locality_domain_idx` 上做原子加
  （`:344-350`），任务索引映射到本域的 cluster 子集
  `n_cluster_idx = locality_domain_idx * kNumLocalClusters + ...`（`:354-360`）；
- **少量任务不偷域**：本域队列空了想偷对面域的活之前，要求
  `num_m_blocks * kNumClusters >= kNumMinStealWaves(=2) * (kNumSMs/2)` 才允许
  （`:366-369`，注释「few tasks, we disable work stealing to avoid cross-die latency」）；
- shared expert 的 L2 任务也要「所有域的 L1 任务都取够」才发（`:395-401`）。

---

## 2. 探针：用指针追逐量「SM 到各域内存」的延迟

### 2.1 设备代码（`impls/sm100_locality_domain.cuh:8-24`，逐行）

```cpp
template <uint32_t kNumHops, uint32_t kNumChunksPerSM, uint32_t kNumChunkBytes>
CUTLASS_GLOBAL __launch_bounds__(32, 1)
void locality_domain_probe_chase(const uint32_t* buf, uint16_t* out) {
    if (threadIdx.x != 0) return;                              // 单线程（warp 里只 lane0 干活）
    const auto sm_idx = ptx::get_sm_idx();                     // %smid
    for (uint32_t chunk_idx = 0; chunk_idx < kNumChunksPerSM; ++chunk_idx) {
        const auto chunk = buf + (uint64_t)(sm_idx * kNumChunksPerSM + chunk_idx) * (kNumChunkBytes / 4);
        uint32_t idx = ptx::ld_global_cg(chunk);               // 预热：先读 chunk[0]
        const auto start = clock64();
        #pragma unroll
        for (uint32_t i = 0; i < kNumHops; ++ i)
            idx = ptx::ld_global_cg(chunk + idx);              // 每跳读下一个 word
        const auto cycles = clock64() - start;
        out[sm_idx * kNumChunksPerSM + chunk_idx] = idx == 0xffffffffu ? 0u : (uint16_t)(cycles / kNumHops);
    }
}
```

四个设计点：

1. **一个 SM 一个 chunk**：`buf` 按 `(domain, sm, chunk, 4KB)` 排布，SM `s` 只追自己的 chunk，
   所以测到的是「**这个 SM** 访问**这块物理内存**」的延迟。
2. **每跳一行**：chunk 是 4KB = 32 行 × 128B；链的步长 `kChainLineStride = 7` 与 32 互质
   （`locality_domain.hpp:65` 注释「Coprime with the line count, so that the chain visits every line」），
   于是 32 跳**恰好把 32 行各访问一次**，不存在同一行重复命中缓存。（本实验 Part A 验证了这一点。）
3. **`ld.global.cg`**（`ptx/ld_st.cuh:249-253`）= `ld.global.cg.u32`，**绕开 L1、只走 L2/下游**：
   配合「4KB chunk + 每行只碰一次」，每次跳基本是一次下游（HBM/L2）访问，测到的是**真实访存延迟**，
   而不是 L1 命中。
4. **`clock64()`** 取 SM 周期数，除以跳数得 per-hop 周期，压进 `uint16`（`0xffffffff` 视为异常置 0）。

### 2.2 host 侧：建链、刷 L2、按域发射（`apis/locality_domain.hpp:69-90`）

```cpp
static void flush_l2() { torch::zeros({256 << 20}, ...); }                     // :69-71 写 256MB 把 L2 冲掉

static torch::Tensor get_probe_chain() {                                       // :73-79
    const auto chain = torch::zeros({kNumLinesPerChunk, kNumWordsPerLine}, ...);
    chain.select(1, 0).copy_((torch::arange(kNumLinesPerChunk, ...) + kChainLineStride)
                             % kNumLinesPerChunk * kNumWordsPerLine);          // 行 l 的 word0 -> 行 (l+7)%32
    return chain.view(-1);
}

static torch::Tensor probe(const torch::Tensor& buf) {                         // :82-90
    buf.view(torch::kInt).view({buf.size(0), -1, kNumChunkBytes / 4}).copy_(get_probe_chain());  // 链铺满每个 chunk
    const auto latency = torch::zeros({...}, buf.options().dtype(torch::kInt16));
    flush_l2();
    for (int d = 0; d < buf.size(0); ++d)
        sm100_locality_domain_probe_chase(buf[d], latency[d]);                 // 每个域一次 launch
    DG_HOST_ASSERT(latency.gt(0).all().item<bool>());                          // 有 0 就是异常
    return std::get<0>(latency.to(torch::kFloat).median(-1));                   // 对 8 个 chunk 取中位数 -> (域, SM)
}
```

注意 `latency` 的维度：`(num_domains, num_sms, kNumProbeChunksPerSM)`
（`empty_per_domain({num_sms, 8, 4096})`，`:111`）。`median(-1)` 把 8 个 chunk 聚成一个数，
得到 `(num_domains, num_sms)` 的延迟矩阵。

### 2.3 域判定的三条闸门（`apis/locality_domain.hpp:93-118`）

```cpp
static torch::Tensor try_probe_sm_locality_domains(const torch::Tensor& buf) {
    const auto [latency, order] = probe(buf).sort(0);            // 按域排：order[0]=较近的域
    const auto sm_domain = order[0].to(torch::kUInt8);
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});       // 2 SM 一个 TPC
    const auto clear = latency[1].ge(latency[0] * kMinFarNearRatio).all().item<bool>();  // 闸门①
    const auto consistent = torch::equal(tpcs, tpcs.select(1, 0).unsqueeze(1).expand_as(tpcs));  // 闸门②
    return clear and consistent ? sm_domain : torch::Tensor();   // 不过 -> undefined
}

static torch::Tensor probe_sm_locality_domains() {               // :109-118
    for (int attempt = 0; attempt < kNumMaxProbeAttempts; ++ attempt)
        if (const auto sm_domain = try_probe_sm_locality_domains(buf); sm_domain.defined())
            return sm_domain;
    // 警告 + 退化为均匀映射
    pybind11::module_::import("warnings").attr("warn")("... no stable mapping; using an even one");
    return get_even_sm_locality_domains();
}
```

- **闸门① `clear`**：对**每个 SM**，较远的域至少要比近的域慢 `kMinFarNearRatio = 1.25` 倍
  （`locality_domain.hpp:66`）。否则说明「两个域其实一样近」，不值得分域。
- **闸门② `consistent`**：同一个 **TPC（2 个 SM 共享一个纹理/光栅后端）** 的两个 SM 必须判到**同一个域**，
  否则拓扑不稳定。
- 两条都过才接受；最多试 `kNumMaxProbeAttempts = 5` 次（`:67`），还不行就 `get_even_sm_locality_domains()`
  均匀铺（`:103-106`，`(tpc_idx % 2)`）并 `warn`。

### 2.4 balance：把域调成 50/50（`apis/locality_domain.hpp:126-142`）

```cpp
static torch::Tensor balance_sm_locality_domains(const torch::Tensor& sm_domain) {
    const auto tpcs = sm_domain.view({-1, kNumSMsPerTPC});
    const auto tpc_domain = tpcs.select(1, 0);
    const int64_t num_tpcs = tpcs.size(0), num_domain_1 = tpc_domain.sum().item<int64_t>();
    const uint8_t larger = num_domain_1 * 2 > num_tpcs;                     // domain 1 是不是多数
    const auto surplus = (tpc_domain == larger).nonzero().flatten().slice(0, num_tpcs / 2);
    const auto balanced = sm_domain.clone();
    balanced.view({-1, kNumSMsPerTPC}).index_fill_(0, surplus, larger ^ 1);
    return balanced;
}
```

**它把「多数域」里前 `num_tpcs/2` 个 TPC 翻到少数域**。注意这**只在输入形如「一个域占满」时才给出精确 50/50**
（这正是探针在真实 MLOPart 下的输出形态：一个进程只看见自己 instance 的 SM，全部靠近同一物理域）。
输入本就是 50/50 时不动；输入是「50/66 对 16/66」这种一般化偏斜时，它会翻 33 个 TPC 得到 17/49，
**并不是一般化最小移动均衡器**。本实验把这条性质也测出来（Part D）——读代码时容易被函数名误导。

结果缓存在 `Runtime`（`csrc/runtime/runtime.hpp:36, 38`：`sm_locality_domains` / `balanced_sm_locality_domains`），
`get_*` 首次调用时才计算（`apis/locality_domain.hpp:120-124, 138-142`）。

---

## 3. 本地化内存：把权重真的放到某个域

### 3.1 MLOPart（CUDA 13.4 之前）

全部走 **CUDA VMM + IPC**：

- host（C++）`LocalityDomainAllocator`（`csrc/runtime/locality_domain.hpp:17-106`）：
  `reserve` = `cuMemAddressReserve`（`:65-70`）、`map` = `create`（拿句柄）+ `cuMemMap` + `cuMemSetAccess`
  （`:72-81`）、`release` = `cuMemUnmap` + `cuMemRelease` + `cuMemAddressFree`（`:83-96`）、
  `get_locality_domain` 反查某指针属于哪个域（`:98-105`）；
- `create` 不自己分配，而是走 **Python MLOPart**：`csrc/runtime/mlopart.hpp:24-35` 通过
  `deep_gemm.utils.mlopart.create_memory(uuid, bytes, domain_idx)` 拿到一个 **POSIX fd**，再用
  `cuMemImportFromShareableHandle` 导成 CUdeviceptr 句柄；
- Python 侧（`deep_gemm/utils/mlopart.py`）：`MLOPart` 用一个 `socketpair` 启动子进程（`:28-37`），
  子进程里起 **MPS 控制守护 + `start_server -mlopart`**（`:113-130`），`serve()` 里对每个请求
  `cuMemCreate(prop.location={DEVICE, domain_idx})` → `cuMemExportToShareableHandle` → `send_fds`
  把 fd 传回（`:94-110`）。`is_available` 会检查 MLOPart 是否真的提供了 `NUM_LOCALITY_DOMAINS = 2`
  个 device（`:99`，否则 `AssertionError: N MLOPart devices`）。
- 分配粒度 `get_granularity()` = `cuMemGetAllocationGranularity(MINIMUM)`（`:56-63`）——
  本机 H100 实测 **2 MiB**（见输出 Part E）。

### 3.2 对上层暴露的两个「按域分配」

- `empty(sizes, dtype, domain_idx)`（`:34-40`）：单域张量；
- `empty_per_domain(sizes, dtype)`（`:43-56`）：`(num_domains, *sizes)`，第 d 片在域 d，
  stride(0) 对齐到粒度，用 `from_blob` + 自定义 `release` 包成 torch 张量。

`deep_gemm/locality_domain.py:20-31` 的 `localize(t, dim)` 就是「把 `(*B, N, K)` 沿 dim 切成
`(num_domains, *B, N/num_domains, K)` 并逐片拷进对应域」；`csrc/utils/layout.hpp:142-153` 的
`is_localized` 用 `get_locality_domain` 反查每片指针确实落在自己的域。kernel 入参用
`check_weights_layout_2d/3d`（`csrc/utils/layout.hpp:155-171`）把本地化权重还原成逻辑形状。

### 3.3 TMA 描述符怎么看待本地化权重

`make_tma_weights_2d_desc`（`runtime_utils.hpp:297-309`）/ `_3d_desc`（`:311-323`）对
「`(N,K)` 或本地化 `(2, N/2, K)`」统一造出一个 **4D TMA 描述符 `(1, 2, N/2, K)`**：
把「域」当成 TMA 的一个外维，用 `slice_stride` 表达第 d 片的起点。于是 kernel 侧只用一个描述符、
用域坐标在 TMA 内选择第 d 片——**就近取权重**对 kernel 是透明的。

---

## 4. H100 真机实验

`src/locality_domain_probe.cu` **原样 include 真实设备代码** `sm100_locality_domain.cuh`，
host 侧用 C++ 逐行复刻 `try_probe / get_even / balance`（无 torch 依赖）。运行
`harness/exp_locality_domain_probe.sh`。H100 80GB HBM3、sm90、132 SM、50 MB L2、驱动 570.211.01。

### 4.1 Part A：链的置换性质

`stride=7, lines/chunk=32 -> visits all lines once = YES`。坐实 32 跳里每行恰好访问一次，
所以每个被计时的跳都是「首次触达」而非缓存重命中。

### 4.2 Part B：两个域都放 device HBM（模拟无本地化）

| 域 | min | median | max |
|---|---|---|---|
| domain0 | 591 | 625 | 665 |
| domain1 | 584 | 624 | 662 |

（单位：SM 周期/跳。）两个域延迟几乎相同，`try_probe: clear=F consistent=F valid=F, worst ratio=1.000`。
即**没有本地化差异时探针拒绝给出映射**（闸门①不过），于是走 `get_even_sm_locality_domains()`：
132 SM 均分成 **66/66**。这正好解释了本机 Part E 的真实行为（见 4.4）。

### 4.3 Part C：把 domain 1 放到 host-pinned memory（模拟「远处域」）

domain0 仍是 HBM，domain1 用 `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` 映射到设备地址空间：

| 域 | min | median | max |
|---|---|---|---|
| domain0 (HBM) | 588 | 627 | 686 |
| domain1 (HOST) | 2186 | 2333 | 2484 |

`try_probe: clear=T consistent=T valid=T, worst ratio=3.265`。探针**准确检出**「HBM 近、host 远」，
把 132 个 SM 全判到 domain0（也因此 `balance` 把它翻成 66/66）。这条用例证明探针的判定逻辑
（sort→order[0]→1.25× 闸门→TPC 一致）在真实设备上是**有效且稳定**的，只是本机没有两个真正的物理域。

### 4.4 Part D：host 逻辑边界用例（9 PASS / 0 FAIL）

- 清晰两域（TPC 成对一致）→ 接受，66/66 SMs；
- far/near 比 1.10 < 1.25 → 拒绝；
- 比**恰好** 1.25 → 接受（对照闸门是 `>=`）；
- TPC 内两 SM 不一致 → 拒绝；
- `balance(all-0)` / `balance(all-1)` → 66/66 SMs；
- `balance(50/66)` → 17/49（记录「只在占满时精确」的性质，非 bug 但易误读）；
- `get_even(8) == [0,0,1,1,0,0,1,1]`。

### 4.5 Part E：真实 DeepGEMM API（Python）

```
get_num_locality_domains  = 2
get_granularity           = 2097152          # 2 MiB
is_localization_available = False            # MLOPart 子进程断言 "1 MLOPart devices"
num_sms = 132  get_even = [0, 0, 1, 1, 0, 0, 1, 1]
sm_domains[:8] = [0, 0, 1, 1, 0, 0, 1, 1]  sum = 66  == get_even: True
balanced[:8] = [0, 0, 1, 1, 0, 0, 1, 1]  sum = 66
```

三件事被真机坐实：① 本机 MLOPart 不可用（MPS 只给 1 个 device，见 `mlopart.py:99` 的断言）；
② 此时 `get_sm_locality_domains()` **逐位等于** `get_even_sm_locality_domains()`（Part E 里 `== get_even: True`），
即 Part B/D 复刻的 fallback 就是真机行为；③ `get_granularity() = 2 MiB`。

### 4.6 域如何影响 Mega MoE 的调度（源码小结，SM100 专属仅分析）

变量 `sm_locality_domain_idx` 在 leader CTA 的调度 warp 里被读出（`kernel:945` → `scheduler:440`），
之后：任务计数按域分开（`scheduler:344-360`）、只有当「全局任务够多」（`>=2` 个波次 × 半机 SM）
才允许跨域偷取（`:366-369`）、shared expert 的 L2 任务要等所有域的 L1 任务到齐（`:395-401`）。
在 132 SM / 2 域的 H100 上，这等价于把「半机一台」当成调度单元——**先就地吃饱，再考虑跨域**。
Mega MoE 主体 kernel 是 SM100 专属，本机只做源码分析。

---

## 5. 结论与边界

1. **域不是一个配置项，而是一个现场测量结果**：`probe_sm_locality_domains()` 用 pointer-chase
   量「每个 SM 到每块内存」的延迟，`try_probe` 用两道闸门（far/near ≥ 1.25、TPC 内一致）判断映射是否
   稳定，最多试 5 次，失败则 `warn` + 均匀映射。探针设备代码架构无关，H100 上真机跑通。
2. **H100 实测**：无本地化时两域延迟相同（~624 周期/跳），探针**正确地拒绝**并回退到 66/66 均匀映射；
   把 domain1 换成 host-pinned 后探针**正确检出**（ratio ~3.3）并给出稳定、TPC 一致的划分。
3. **balance 的函数名有欺骗性**：它只在「一个域占满」时才给出精确 50/50，一般化偏斜输入下不是最小
   移动均衡器——本实验显式测出并记录。
4. **本地化内存走 CUDA VMM + MLOPart IPC**：`cuMemCreate`/`Export`/`Import` + `reserve`/`map`/`set_access`，
   粒度 2 MiB；Mega MoE 的权重经 `localize` 切到各域，TMA 描述符把域当外维，对 kernel 透明。
5. **边界**：真实「两个物理域」需要 MLOPart（MPS `-mlopart`）或 CUDA 13.4 的 locality domain，
   本机 H100 都没有；Part C 的 host-pinned 只是**方法论验证**，不是真实 locality domain 的带宽比值。
   主体 kernel 内域调度是 SM100 专属，仅源码分析。

参考链接（真实）：
- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
- CUDA VMM（`cuMemCreate` / 可共享句柄 / locality domain）：
  <https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__VA.html>
- NVIDIA MPS / MLOPart：
  <https://docs.nvidia.com/deploy/mps/index.html>
