# 所有 Mega MoE kernel 共用的任务调度器 `scheduler/mega_moe.cuh`

> 本专题（P110-1）此前单独拆过**三个**调度器：`scheduler/gemm.cuh`（第 108 篇，所有稠密 /
> grouped / MoE 反向 GEMM 共用的任务枚举器）、`scheduler/mega_gate.cuh` + `scheduler/mega_mhc.cuh`
> （第 92 篇，两个 mega 家族的分段屏障调度器）。唯独 **Mega MoE 主 kernel 自己的任务调度器**
> `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（**481 行**）从没被单独读过——它被前面
> 100+ 篇以「`scheduler:xxx`」的行号引用了 15+ 次，却只在第 02 篇 §4 有一节概述。
>
> 它是整条 **dispatch + linear1 + SwiGLU + linear2 + combine** 流水线**把工作切成 task、并发给
> 5 类 warp 分身的唯一入口**，也是 host 侧 ring buffer 容量公式 `get_num_max_live_pool_blocks`
> 的唯一定义处——换句话说，**「这块对称显存要开多大」和「每个 SM 下一块该算什么」都由它回答**。
>
> **为什么 H100(sm90) 可跑**（与第 108 篇同源理由）：本文件的分工是**纯整数 / 布局运算**——
> `get_num_l1_warmup_waves` / `get_num_max_live_pool_blocks` 是 `CUTLASS_HOST_DEVICE constexpr`；
> `TaskInfo` 是 `alignas(16)` 的 POD；`create_task` / `get_pool_block_offset` / `get_next_task_idx`
> 全是 warp 内建的整数映射，不碰 UMMA / TMA / tcgen05；`L2KBlockDependency` 只用
> `red.release.gpu.global.xor` + `ld.acquire.gpu.global`——sm90 起即支持。唯一 SM100 专属的是
> `publish_task` 的 `st.async.shared::cluster` + cluster transaction barrier、以及主 kernel 端的
> UMMA/TMA。故本篇在 H100 上**直接 include 真实头文件、编译运行真实设备代码**做架构无关部分的
> 对拍，`publish_task` / 主 kernel 的 cluster 协作**仅做源码分析**。
>
> 源码坐标（`~/github/DeepGEMM`，本地 commit `057ca59`，"Public release 26/09/30 (#462)"）：
> `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（481 行，sha256 前缀 `1bf76529`）。
> 实验：`src/mega_moe_scheduler.cu` + `harness/exp_mega_moe_scheduler.{sh,out.txt}`。

---

## 0. 它在三段式结构里的位置

第 89 篇把 Mega MoE 的 host 链拆成三段：`csrc/apis/mega_moe.hpp`（对齐 / 检查 / 公式）→
`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`（TMA 描述符 + JIT 模板实参 + launch）→
`impls/sm100_fp8_fp4_mega_moe.cuh`（设备 kernel）。调度器**不在这三段里**，它是 kernel 内部
的一条「中枢神经」：主 kernel 在 `sm100_fp8_fp4_mega_moe.cuh:292-304` 用模板参数实例化它，
把共享内存里的 task 槽、两根 mbarrier 交给它：

```cpp
// impls/sm100_fp8_fp4_mega_moe.cuh:292-304
auto scheduler = sched::MegaMoEScheduler<
    BLOCK_M, BLOCK_N, BLOCK_K,
    L1_SHAPE_N, L1_SHAPE_K,
    L2_SHAPE_N, L2_SHAPE_K,
    kNumExpertsPerRank,
    kNumSMs, kNumRanks,
    kNumRingBlocks, kNumLocalityDomains,
    kNumSharedExperts>(
        workspace,
        shared_storage.task_info_full_barriers,
        shared_storage.task_info_empty_barriers,
        shared_storage.task_infos);
```

**一根 warp 专门当调度 warpgoup**（`mainloop`，`:945` 触发），它把 task 写进共享内存的
`task_infos[2]` 双缓冲槽并用 mbarrier 通知；另外 4 类消费者 warp 在各自的 `while
(scheduler.get_next_task(task_info))` 循环里消费（`:696` A-loader、`:762` B-loader、`:855` MMA、
`:982` epilogue），**谁消费完谁 `release_task_info()`**（`:990`）。task 槽的容量是
`kNumScheduleStages = 2`（kernel `:188`），`task_info_full_barriers[i].init(1)`、
`task_info_empty_barriers[i].init(kNumScheduleConsumerThreads = 2 * kNumEpilogueThreads)`
（kernel `:276-278`）——即「生产者 1 个、消费者 2×epilogue 线程数」，与调度器里的
`task_info_full_barriers[sched_stage_idx].wait(...)` / `task_info_empty_barriers[...].arrive(0u)`
（`:248-259`）严格对偶。

一句话：**调度器负责「枚举任务 + 池块分配 + L1→L2 依赖 + 分域负载均衡」，主 kernel 负责
「按 task 搬数 / 算 MMA / 写回」。** 本文拆前者。

---

## 1. 两个正交的调度世界：routed 与 shared

`BlockPhase`（`:83-89`）枚举了 5 种任务相位：

```cpp
enum class BlockPhase : uint32_t {
    None = 0, Linear1 = 1, Linear2 = 2,
    SharedLinear1 = 3, SharedLinear2 = 4
};
```

- **routed（Linear1 / Linear2）**：走 dispatch → L1 → SwiGLU → L2 → combine 的正规军。
  L1 每块算 `[BLOCK_M, 2I]`（gate+up），L2 算 `[BLOCK_M, H]`。它们的输入要先等 dispatch 把
  token 从各 rank 拉齐（`fetch_expert_recv_count`，`:286-307`）。
- **shared（SharedLinear1 / SharedLinear2）**：共享专家分支（第 44 篇），**输入就是本 rank 的
  原始 token**，所以它在 dispatch 完成之前就能跑——调度器把它单独排成两条队列
  （`shared_l1_task_count` / `shared_l2_task_count`），`mainloop` 里在 `fetch_expert_recv_count`
  **之前**先把 SharedLinear1 发出去（`:445-452`）。

`TaskInfo<kHasShared>`（`:91-134`）两个变体的**布局必须一致**（`:136` 的
`DG_STATIC_ASSERT(sizeof(TaskInfo<true>) == sizeof(TaskInfo<false>))`），因为消费者 warp 是
`if constexpr` 共用同一段代码。8 个 `uint32_t` 字段里，`pool_block_idx` 同时是「全局 M 块号」
（用于 ring 寻址与 L2K 依赖），`valid_m` 是这一块真实有效的 token 行数（尾块不满 `BLOCK_M`），
`get_umma_aligned_valid_m()`（`:126-129`）把它向上对齐到 16——因为 UMMA 的 N 维最小单位是 16
（第 10 篇）。`is_shared()`（`:131-133`）用 `block_phase > BlockPhase::Linear2` 判定，仅当
`kHasShared=true` 时才可能为真。共享专家的两条队列几何由模板常量给出（`:182-185`）：

```cpp
static constexpr uint32_t SHARED_L1_SHAPE_N = L1_SHAPE_N * kNumSharedExperts;  // L1 沿 N 变宽
static constexpr uint32_t SHARED_L2_SHAPE_K = L2_SHAPE_K * kNumSharedExperts;  // L2 沿 K 变宽
```

这正是第 44 篇讲的「L1 沿 N、L2 沿 K 变宽」在类型系统里的落点。

---

## 2. 行容量公式：调度器与 host 之间唯一的几何契约

`get_num_max_live_pool_blocks`（`:48-80`）是**全文件最重要的一个数**——它给出「同时有多少个
M block 存活（在 ring 里占着槽）」，host 用它乘 `BLOCK_M` 定 ring 容量。

它的前提是 `get_num_l1_warmup_waves`（`:17-45`）：L1 必须先跑若干「预热波」把 L1 结果喂进 L2，
否则 L2 会等 L1 而 L1 又在等 L2 让出的槽——**死锁**。预热波数取两个下界的较大者（`:23-44`）：

- **第一个 L2 波会触及多个 M block**：`ceil_div(num_clusters, num_l2_n_clusters)` 个 M block 的
  **全部** L1 N 任务都要先发（`:24-26`）；
- **交错调度残留**：当每个 M block 的 L1 任务数 > L2 任务数时（`num_l1_n_clusters >
  num_l2_n_clusters`），交错会让每个 M block 多留 `(l1n - l2n)` 个 L1 任务没发；最后一个 M block
  是瓶颈，故预热阶段要把前面累积的 `(ntb-1)*(l1n-l2n)` 个补发（`:37-41`），再 +1 波盖住部分波的
  取整（`:41`）。

得到预热波数后，`get_num_max_live_pool_blocks`（`:48-80`）算「预热后活跃块 + L2/L1 交错导致的
前沿增长 + 一个全局波余量」，三部分求 min 上限到 `num_total_m_blocks`：

```cpp
// scheduler/mega_moe.cuh:61-79（节选）
const int num_l1_clusters    = num_total_m_blocks * num_l1_n_clusters;
const int num_l1_waves       = constexpr_ceil_div(num_l1_clusters, num_clusters);
const int num_l1_warmup_waves = min(get_num_l1_warmup_waves(...), num_l1_waves);
const int num_live_blocks_after_warmup = constexpr_ceil_div(warm_clusters, num_l1_n_clusters);
// mineral growth: 若 L2 沿 M 推进更快（l2n > l1n）
const int frontier_growth = l2n > l1n ? ceil_div(ntb * (l2n - l1n), l2n) : 0;
const int wave_margin = ceil_div(num_clusters, min(l1n, l2n));
return min(ntb, num_live_blocks_after_warmup + frontier_growth + wave_margin);
```

**host 侧怎么用它**（`csrc/apis/mega_moe.hpp:55-63`）：对**所有候选 `BLOCK_M`** 扫描，
`num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank`
（`+num_experts_per_rank` 是每专家尾块对齐的最坏浪费），取 `max(live_pool_blocks * block_m)`，
再对齐到 `kLCMCandidateBlockM = 1920`：

```cpp
// csrc/apis/mega_moe.hpp:55-63
int num_ring_tokens = 0;
for (const auto& block_m: layout::kCandidateBlockM) {
    const auto num_pool_blocks = ceil_div(num_max_routed_tokens, block_m) + num_experts_per_rank;
    const auto num_live_pool_blocks = sched::get_num_max_live_pool_blocks(
        num_pool_blocks, num_sms, hidden, intermediate_hidden);
    num_ring_tokens = std::max(num_ring_tokens, num_live_pool_blocks * block_m);
}
num_ring_tokens = math::align(num_ring_tokens, layout::kLCMCandidateBlockM);
```

**实验 Part A/A2（H100 真机）**：`get_num_max_live_pool_blocks` 与 `get_num_l1_warmup_waves` 的
设备/宿主 constexpr 调用结果，和独立复刻的公式在 10 组形状上**逐点相等**（`live` 与 `warm`
两列全对，含 `ntb` 从 7 到 1024、`nsm` 从 64 到 148）；并复刻了 host 的 ring 分配：在
DeepSeek 量级（`hidden=7168, inter=2048, ranks=8, max_tok=4096, topk=8, experts_per_rank=32,
num_sms=132`，`num_max_routed_tokens=262144`）下扫描候选 `BLOCK_M`，最大值出现在 `block_m=240`
（`ring=119280`），对齐 1920 得 **`num_ring_tokens = 120960`**。**34 PASS / 0 FAIL**。

---

## 3. L1→L2 的细粒度依赖：`L2KBlockDependency` 的位图与相位

L2 的每个 K block 需要 L1 的若干个 N block 先算完——这是第 02 篇 §4.3 说的「细粒度依赖与死锁
规避」。`L2KBlockDependency`（`:142-166`）用一位一 N block 的位图实现：

```cpp
// scheduler/mega_moe.cuh:142-166（节选）
template <uint32_t L1_SHAPE_N, uint32_t BLOCK_N, uint32_t BLOCK_K>
struct L2KBlockDependency {
    static constexpr uint32_t kNumL1BlockNs = L1_SHAPE_N / BLOCK_N;
    static constexpr uint32_t kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N / 2);
    static constexpr uint64_t kFullMask = kNumL1BlockNs == 64 ? ~0ull : (1ull << kNumL1BlockNs) - 1;
    const uint64_t* mask_ptr;
    uint64_t expected_mask, pending_mask = kFullMask;
    CUTLASS_DEVICE L2KBlockDependency(const uint64_t* mask_ptr, const uint32_t& generation_idx):
        mask_ptr(mask_ptr), expected_mask(generation_idx & 1 ? 0ull : kFullMask) {}
    CUTLASS_DEVICE static void arrive(const uint64_t* mask_ptr, const uint32_t& n_block_idx) {
        ptx::red_xor_rel(mask_ptr, 1ull << n_block_idx);          // 翻转该 bit，不累加
    }
    CUTLASS_DEVICE void wait(const uint32_t& k_block_idx) {
        const auto k_block_mask = ((1ull << kNumL1BlockNsPerL2KBlock) - 1) << (k_block_idx * kNumL1BlockNsPerL2KBlock);
        while (pending_mask & k_block_mask)
            pending_mask = ptx::ld_acq_gpu(mask_ptr) ^ expected_mask;
    }
};
```

三个非平凡点：

1. **为什么是 XOR 而不是原子加**：mask 按「ring 世代」复用。位图用 `^=` 翻转，于是 **世代 g 的
   全就绪 = 全 1、世代 g+1 的全就绪 = 全 0**；`expected_mask` 由 `generation_idx & 1` 选
   `{kFullMask, 0}`（`:153-154`），把「绝对位图」翻译成「本世代就绪位」。源码注释（`:140-141`）
   明说：**这套奇偶依赖 ring 容量**——所以第 2 节的行容量公式不只是「够不够」，而是「奇偶对不对」。
2. **一个 L2 K block 要几个 L1 N block**：`kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N/2)`
   （`:145`）。`BLOCK_N/2` 是因为 L1 的 `2I` 维被 gate/up 交错（第 02/03 篇的 gran-8 交错），
   所以一个 L2 K block 覆盖 `BLOCK_K` 个 hidden，对应 `2·BLOCK_K/BLOCK_N` 个交错 N 块。
3. **两处调用点**：消费者在 `impls/sm100_fp8_fp4_mega_moe.cuh:723-726` 构造依赖并
   `while (k_block_idx < num_k_blocks) l2_k_block_dependency.wait(k_block_idx)`；
   生产者在 L1 的 epilogue 落盘后 `:1246` 调 `L2KBlockDependency::arrive(..., n_block_idx)`。

`generation_idx = pool_block_idx / kNumRingBlocks`（消费者 `:723`）、`ring_block_idx =
pool_block_idx % kNumRingBlocks`（`:997-998`）——**池块号在 ring 上的「商」是世代、「余数」是槽**。

**实验 Part C（H100 真机，真实设备代码）**：`L1_SHAPE_N=4096, BLOCK_N=128, BLOCK_K=128` 下
`kNumL1BlockNs=32`、`kNumL1BlockNsPerL2KBlock=2`；`arrive` bit {0,1,2,3} 后 mask=`0xF` 且
`wait(K0)` / `wait(K1)` 均在有限自旋内返回；对 `generation=1`，空 mask（`0`）被 `expected_mask=0`
判为「全就绪」，`wait` 直接返回——**坐实奇偶相位语义**。

---

## 4. 池块 → (专家, 块内 M, 有效 M)：`create_task` 的 warp 协作

任务枚举最不平凡的一段是 `create_task`（`:309-342`）：给定一个**全局 M block 号**，warp 要合作
求出它属于哪个专家、在该专家内的第几个 M block、以及尾块的有效行数。这里没有 `grouped_layout`
之类的元数据表，而是**每 lane 缓存自己名下的专家 token 数**，用 warp 内建现算前缀和：

- 缓存来自 `fetch_expert_recv_count`（`:286-307`）：每个 lane 负责专家 `i*32 + lane_idx`
  （`kNumExpertsPerLane = ceil_div(kNumExpertsPerRank, 32)`，模板默认 `:175`），
  `while (value >> 32 != kNumSMs * kNumRanks)` 自旋等待所有 SM、所有 rank 都投完票
  （高 32 位是「已上报 SM×rank 数」，低 32 位是 token 计数）——**这就是 dispatch 完成信号**。
- `get_pool_block_offset`（`:272-280`）用一次 `__reduce_add_sync` 算出某专家之前所有专家的
  M block 累计数（前缀和 = 该专家的池块起点）。
- `create_task`（`:309-342`）：对每个 lane 名下的专家算 `ceil_div(tokens, BLOCK_M)`，
  用 `math::warp_inclusive_sum` 得 lane 前缀和，再看全局 `m_block_idx` 落在谁的区间里
  （`is_owner`），`__ballot_sync` 找出 owner lane，`ptx::exchange` 把
  `(local_expert_idx, owner_m_block_idx, valid_m)` 广播给全 warp。

```cpp
// scheduler/mega_moe.cuh:326-339（节选）
const bool is_owner = expert_idx < kNumExpertsPerRank and
    m_block_idx >= lane_pool_block_offset and m_block_idx < lane_pool_block_offset + num_m_blocks;
const uint32_t owner_mask = __ballot_sync(0xffffffff, is_owner);
if (owner_mask) {
    const uint32_t owner_lane_idx = static_cast<uint32_t>(__ffs(owner_mask) - 1);
    const uint32_t owner_m_block_idx = m_block_idx - lane_pool_block_offset;
    const uint32_t owner_valid_m = cute::min(num_tokens - owner_m_block_idx * BLOCK_M, BLOCK_M);
    result.local_expert_idx = ptx::exchange(expert_idx, owner_lane_idx);
    result.m_block_idx      = ptx::exchange(owner_m_block_idx, owner_lane_idx);
    result.valid_m          = ptx::exchange(owner_valid_m, owner_lane_idx);
}
```

**实验 Part D/E（H100 真机，真实设备代码）**：`create_task` / `get_pool_block_offset` /
`get_num_total_pool_blocks` / `get_num_tokens` 全在 GPU 上跑。用 `BLOCK_M=128`、
每专家 token 数 `{5, 0, 300, 200, 130, 1, 7, 250}`——池块总数 12（host 12）、
**全部 12 块的 `(expert, owner_m, valid_m)` 逐块与规范复刻 0 不符**（含 `valid_m` 尾块
`5/44/72/2/1/7/122` 等），`get_num_tokens` 抽查 0 不符。

---

## 5. 交错调度 + 分域负载均衡 + 工作窃取：`get_next_task`

routed 任务的枚举在 `get_next_task`（`:372-407`）里，它把第 2 节的「L1 预热 → L1/L2 交错」落成
一条状态机：

- 初始 `num_sched_l1_waves` 由 `fetch_expert_recv_count` 用**真实** `num_total_m_blocks` 重算
  （`:302-306`），每个调度 warp 认领一波 L1（`:375-377`），L1 耗尽后切到 L2；
- 切到 L2 时，若还有 L1 预热波没发就把它重置为 1（`:392-393`）——保持「L1 领先 L2」；
- **L2 发出后要等该池块所需的全部 L1 任务被所有调度队列认领**（`:399-402`）：

```cpp
// scheduler/mega_moe.cuh:399-402
const auto num_required_l1_tasks = (task_info.pool_block_idx + 1) * (kNumL1Clusters / kNumLocalityDomains);
for (uint32_t i = 0; i < kNumLocalityDomains; ++ i)
    while (ptx::ld_volatile(workspace.get_l1_task_count_ptr() + i) < num_required_l1_tasks) {}
```

任务号由 `get_next_local_task_idx`（`:344-349`）用一次 `atomic_add` 在**本域**计数上领取，
`get_next_task_idx`（`:351-361`）把它翻译成 `(m_block_idx, n_cluster_idx)`：

```cpp
// scheduler/mega_moe.cuh:354-360
constexpr uint32_t kNumLocalClusters = kNumClusters / kNumLocalityDomains;
const uint32_t task_idx = get_next_local_task_idx(task_count_ptr, locality_domain_idx);
if (task_idx >= num_m_blocks * kNumLocalClusters) return false;
m_block_idx  = task_idx / kNumLocalClusters;
n_cluster_idx = locality_domain_idx * kNumLocalClusters + task_idx % kNumLocalClusters;
```

**locality domain**（`kNumLocalityDomains ∈ {1, 2}`，`:226`）把 N cluster 均分给两个 die；本域队列
空了才允许**跨域窃取**（`:363-370`），但有个门槛——任务太少时拒绝偷（避免跨 die 延迟）：

```cpp
// scheduler/mega_moe.cuh:368-369
return kNumLocalityDomains > 1 and num_m_blocks * kNumClusters >= kNumMinStealWaves * (kNumSMs / 2) and
       get_next_task_idx<kNumClusters>(task_count_ptr, sm_locality_domain_idx ^ 1, num_m_blocks, m_block_idx, n_cluster_idx);
```

`kNumMinStealWaves = 2`（`:225`）：**本波总任务 ≥ 2 个全局波才偷**。

**实验 Part F（H100 真机）**：`kNumClusters=32, num_m=5, kNumDomains=2` 时，domain 0 领取的 80 个
task 恰好覆盖 `m∈[0,5) × n∈[0,16)`、domain 1 恰好覆盖 `n∈[16,32)`——**两域并集是
`[0, num_m*32)` 的双射，0 不符**；窃取阈值复刻为「`num_m*32 ≥ 64` 即 `num_m ≥ 2` 才偷域」。

---

## 6. 共享专家队列与整体 `mainloop`

`shared_mainloop`（`:421-436`）是一个**与 routed 无关的动态队列**（`:430-431` 注释「用动态调度
减少 L1/L2 不同 tile 形状的尾效应」），它复用同一个双缓冲 task 槽协议。`mainloop`（`:438-476`）
把两个世界编排起来：

1. `sm_locality_domain_idx` 由 SM 号查表得到（`:440`）；
2. **共享 L2 早发还是尾发**由 `is_shared_l2_early` 决定（`:443-444`）：
   `ceil_div(num_tokens, BLOCK_M) * kNumSharedL2Clusters <= kNumSMs/2`——即「共享 L2 一朵波塞得下
   dispatch 的空档」就在前面发，否则留到尾部；
3. **先发 SharedLinear1**（不等 dispatch）、再 `fetch_expert_recv_count`（等 dispatch）、再枚举
   routed 任务（`:445-465`）；
4. 若共享 L2 是尾发，在 routed 之后补上（`:467-471`）；
5. 最后发一个 `BlockPhase::None` 的**哨兵**（`:473-475`）让所有消费者退出循环。

`get_next_task(task_info)`（消费者调的那个，`:248-253`）只做「等 full barrier → 读槽 → 翻相位 →
返回」：

```cpp
// scheduler/mega_moe.cuh:248-259
CUTLASS_DEVICE bool get_next_task(task_info_t& task_info) {
    task_info_full_barriers[sched_stage_idx].wait(sched_phase);
    task_info = task_infos[sched_stage_idx];
    advance_sched_pipeline();
    return task_info.is_valid();
}
CUTLASS_DEVICE void release_task_info() const {
    ptx::fence_acq_rel_cta();                               // 先读完元数据
    task_info_empty_barriers[sched_stage_idx ^ 1].arrive(0u);
}
```

`advance_sched_pipeline`（`:242-246`）同时翻 `sched_stage_idx` 与 `sched_phase`；生产者
`publish_task`（`:409-419`）用 lane 0/1 把 32 字节 task 经 `ptx::st_async_cluster` 写进共享槽并
`arrive_and_expect_tx(sizeof(task_info_t), lane_idx)`——**这段是 SM100 cluster 专属，仅分析**。

---

## 7. Publish / consume 的顺序契约（第 89 篇的补课）

调度器把「认领任务（改全局计数）」和「写 task 槽」的顺序钉死成
`wait empty → get_next_task() → publish_task`（`:460-465` 的注释明说：`get_next_task()` 会推进
全局计数，绝不能早于消费者释放槽）。这与消费者侧「等 UMMA 到 → **先 `release_task_info()`**（`:990`）
→ 再算偏移」构成闭环：**task 槽的释放只由 epilogue 消费者做**（因为只有它知道 UMMA 的
`tmem_full` 何时到，`:986`），A/B-loader 消费的是同一个槽但不 release。这就是为什么
`task_info_empty_barriers` 的 init 计数是 `2 * kNumEpilogueThreads`（`:278`）。

---

## 8. 与另外三个调度器的对照

| 维度 | `scheduler/gemm.cuh`（第 108 篇） | `scheduler/mega_gate` / `mega_mhc`（第 92 篇） | **`scheduler/mega_moe.cuh`（本篇）** |
|---|---|---|---|
| 服务对象 | 稠密/grouped/MoE 反向 GEMM | Mega Gate / Mega mHC | **Mega MoE 主 kernel** |
| 任务粒度 | `(group, m_block, n_block)` | split-K + 分段屏障 | **池块（真实 token 行）+ 相位** |
| 分组来源 | `grouped_layout` / PSUM | 固定 grid | **warp 内建现算（`create_task`）** |
| 依赖 | k-grouped K 起点 / SF 前缀和 | `ScoreBarrier` / `SplitBarrier` | **`L2KBlockDependency` 位图 + ring 世代奇偶** |
| 负载均衡 | L2 swizzle | 无 | **locality-domain 分域 + 工作窃取** |
| 对 host 的契约 | 无 | 无 | **`get_num_max_live_pool_blocks`（ring 容量）** |
| H100(sm90) 可跑 | 全文件 | 除一行 `red.async` | **除 `publish_task` 的 cluster 写** |

三者共同的「家族特征」是**用编译期常量 + 整数运算把负载均衡做在设备端**，且都把一个
`constexpr` 公式暴露给 host 做资源预算——`gemm.cuh` 是 `get_num_1d_blocks_per_group`，
`mega_moe.cuh` 是 `get_num_max_live_pool_blocks`。

---

## 9. H100 真机实测小结

`harness/exp_mega_moe_scheduler.{sh,out.txt}` + `src/mega_moe_scheduler.cu`（同一 `.cu` 同时编
`sm_90a` 运行与 `sm_100a` 仅编译覆盖 SM100 分支），**34 PASS / 0 FAIL**：

| 组 | 核验内容 | 结果 |
|---|---|---|
| A | 行容量公式 vs 独立复刻（10 形状） | `live`/`warm` 两列全对 |
| A2 | 复刻 host ring 分配（DeepSeek 量级） | `block_m=240`、`ring=119280`、对齐 1920 → **120960** |
| B | `TaskInfo` ABI：`sizeof=32`、`alignof=16`、字段偏移、`is_valid`/`is_shared`/`get_umma_aligned_valid_m(100)=112` | 全对 |
| C | `L2KBlockDependency`：`arrive` XOR 翻转 + `wait` 相位（`kNumL1BlockNs=32`、perK=2） | 正例返回、世代奇偶语义正确 |
| D/E | `create_task` 的 warp 协作映射 + 池块前缀和（8 专家 12 池块） | **12/12 块 0 不符**、`get_num_tokens` 0 不符 |
| F | 分域枚举双射 + 工作窃取阈值 | 两域并集覆盖、0 不符；`num_m≥2` 才偷 |

**SM100 专属、仅分析**：`publish_task` 的 `ptx::st_async_cluster`（`ld_st.cuh:206-224`）与 cluster
transaction barrier 协作、主 kernel 端 UMMA/TMA、以及 `st_async`/`mapa.shared::cluster` 指令本体
（H100 无 cluster launch 语境）。

---

## 10. 结论

Mega MoE 主 kernel 之所以能把「通信 + 两段 GEMM + SwiGLU + combine」压进一次 launch，靠的不只是
UMMA/TMA，还有这个**481 行的设备端调度器**：

1. **它把「每 SM 下一块算什么」变成纯整数运算**——`create_task` 用 warp 内建在设备端现算
   `(专家, 块内 M, 有效 M)`，不依赖任何 host 预建的 `grouped_layout` 表；
2. **它把「L1 预热 + L1/L2 交错」编码成状态机**，用 `L2KBlockDependency` 的 XOR 位图 + ring 世代
   奇偶做零锁的细粒度依赖——**依赖的正确性与 ring 容量公式强耦合**；
3. **它把「跨 die 负载均衡」做成可选的工作窃取**，门槛 `kNumMinStealWaves` 让任务稀疏时不付跨
   die 延迟；
4. **它向 host 暴露唯一一个几何契约** `get_num_max_live_pool_blocks`——host 用它（对候选
   `BLOCK_M` 取 max 后对齐 1920）定 ring 容量，device 用真实 recv 计数定实际池块数（≤ 该上界）。

这条「host 定上界、device 定实际、二者共享一个 constexpr 公式」的模式，与第 108 篇的
`gemm.cuh` 一脉相承，是 DeepGEMM 把 kernel 复杂度与 host 资源预算解耦的统一手法。
