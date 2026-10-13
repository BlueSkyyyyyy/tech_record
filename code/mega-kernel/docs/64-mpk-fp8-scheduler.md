# MPK SM100 block-scaled FP8 linear 家族的内核内 tile 调度器走读（`linear_fp8_sm100_scheduler.cuh`，P150-1）

> 定位：本文补掉 `docs/32-mpk-sm100-linear-tasks.md` **§7「局限」明留的「未展开」**——
>
> > **未展开**：…… `_scheduler`/`_epilogue_utils`/`_cute_tie` 未逐行走；TP 变体的 `mla_mtp_tpN_main`
> > 主体只核了常量。
>
> 其中 `_cute_tie`（60 行宏工具）已由 `docs/60`（P146-1）§6 补掉；本文把剩下的
> **`linear_fp8_sm100_scheduler.cuh`（199 行）+ `linear_fp8_sm100_epilogue_utils.cuh`（30 行）**
> 逐行打开，并把「MPK 为什么和 DeepGEMM 不一样」量出来。`docs/32` §2.3 只映射了
> `Scheduler` 的模板参数清单（`_scheduler.cuh:38-52`）与文件行数，未走 `get_swizzled_block_idx` /
> `get_global_idx` / `get_next_block` 的函数体。
>
> 主题角度：megakernel 的核心是「把一个 task 的活分配进常驻 kernel」，而 `Scheduler` 正是
> **SM100 手写持久化 task 的「内核内 tile 调度」那一层**——和本专题 P15-1/P22-1/P36-1/P37-1
> （设备内调度）同轴，但发生在**单个 CTA 内部、对一组静态输出 tile** 上。
>
> 快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `/ssd/mirage_mpk_build`，
> 与 `docs/06`–`docs/63` 同快照）；DeepGEMM 对照 `~/github/DeepGEMM` commit `057ca59`。
> **纯源码走读 + 1 个 host 侧算术探针**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **MPK 把 DeepGEMM 的「多 SM 摊 tile」调度器改成了「单 CTA 顺序全扫」**：`get_next_block` 里
   tile 偏移被**硬编码成 `+0`**（DeepGEMM 是 `+ blockIdx.x`，`gemm.cuh:186`），生产发射
   **`kNumSMs=1`**（`task_register.cc:4839` 注释「persistent kernel: 1 CTA per task, processes all
   tiles」）。host 实测：`kNumSMs=1` 覆盖**全部** tile；而同一条注释里提议的
   `MPK_MAX_NUM_WORKERS`（**全仓唯一出现处就是这条注释，无任何定义**）若真被当 `kNumSMs` 传入，
   只会覆盖 **1 个** tile（1024 个 tile 里只访问 tile#0）。是「持久化内核里 `blockIdx.x` 是
   **worker 序号**而非 task 的 grid 序号」这个 launch 约定，把 DeepGEMM 的「多 SM 并行摊 tile」
   压成「每个 CTA 自己全扫」。
2. **MPK 的 `Scheduler` 是 DeepGEMM 的严格子集**（199 vs 314 行）：保留
   `get_num_1d_blocks_per_group` / `get_swizzled_block_idx` / `get_global_idx`，
   **删掉** `get_next_block` 的 3 个 grouped 分支（`MGroupedMasked`/`MGroupedContiguousWithPsumLayout`/
   `KGroupedContiguous`）、`is_tma_multicast_valid`、`get_aligned_effective_m_in_block`、
   `get_swizzled_block_idx` 里 SM90 的 `#if __CUDA_ARCH__ < 1000` multicast 修正，以及
   `kEnsureZeroPadding`/`kKAlignment`/`kSFKSpan` 三个模板参数。于是 MPK 里 `get_next_block`
   **只对 `GemmType::Normal`/`Batched` 返回 tile**，其余 grouped 类型一律 `return false`——
   而生产恰好只发 `GemmType::Normal`（`task_register.cc:4844`）；MoE 的 grouped GEMM 走**另一个**
   device 身体 `fp8_group_gemm_sm100.cuh`（`docs/33`，P119-1）。
3. **「顺序扫 × swizzle」是 tile 双射**（host 实测 M=32×N=32 / 13×5 / 1×9 / 17×3 全
   `BIJECTION OK`）：`get_swizzled_block_idx` 只改**访问顺序**（L2 局部性 / multicast 对齐），
   不改覆盖集合；所以「一个 CTA 顺序扫完全部 tile」在数值上等价于 DeepGEMM 的多 SM 摊分。
   连最小的 `_epilogue_utils.cuh`（30 行）也藏一枚「**有实现、零调用点**」的
   `EpilogueHeadSplits`（只有 `EpilogueIdentity` 被生产和单测用）——与 `docs/60` §6 的
   `_cute_tie` 同族。

---

## 1. 定位：一个 task 的「内核内调度」层

`linear_fp8_sm100_task_impl`（`linear_fp8_sm100.cuh:45`，`docs/32`）是一个 **warp 特化的
SM100 block-scaled FP8 GEMM CTA**：warp0 TMA 生产者 / warp1 UMMA 消费者（`is_leader_cta` 门控）/
warp2 UTCCP scale / warps≥8 epilogue。它要处理的所有输出 tile 由内嵌在 kernel 里的
`Scheduler` 对象给出：

```
include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_scheduler.cuh  (199 行)
  enum class IndexType { MN, K, SF_K };                                  // :8-12
  get_num_1d_blocks_per_group<...>()                                     // :14-34
  struct Scheduler { ... }                                               // :38-197
    get_next_k_group           :67-75
    Scheduler(...)             :77-103
    get_swizzled_block_idx     :105-129
    get_global_idx             :131-166
    get_next_block             :168-196
```

调用方 `linear_fp8_sm100.cuh`：
- 构造：`Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs>`
  （`:274-282`）；
- 四个角色 warp **各自持有一份** `Scheduler` 拷贝、各自 `while (scheduler.get_next_block(...))`
  迭代（warp0 TMA `:291`、warp1 UMMA `:439`、warp2 UTCCP `:549`、epilogue `:593`）——四者靠
  mbarrier 流水保持对齐；只要 `Scheduler` 的状态机是确定性的（只依赖 `current_iter`），四份拷贝
  自然同步。
- 每个 tile 的全局索引（m/n/k、SF k）由 `get_global_idx<...>(...)` 现算（`:300/317/321/328/378/392`）。

这正是 `docs/32` §1「一个 task = 一个 CTA、warp 特化」之外**没打开**的一层：**CTA 内部怎么决定
下一个算哪个 tile**。

---

## 2. `get_num_1d_blocks_per_group`：8 还是 16（`:14-34`）

用 `if constexpr` 之外的**编译期搜索**选一个「1D swizzle group」的宽度，候选只有 `{8, 16}`
（`:22`），代价函数 `usage`（`:23-27`）：

```
usage = kIsMulticastOnA ? candidate*BLOCK_N + ceil_div(kNumSMs,candidate)*BLOCK_M   // 组在 N 上
                        : candidate*BLOCK_M + ceil_div(kNumSMs,candidate)*BLOCK_N;  // 组在 M 上
```

选 `usage` 更小的那个（严格 `<`，`:28`；平手留 `8`，因 `{8,16}` 先访问 8）。
DeepGEMM 版逐字相同（`gemm.cuh:15-30`，只是把 `constexpr_ceil_div` 换成 `math::constexpr_ceil_div`）。

host 探针（`src/fp8_scheduler_order.cu` §A）在生产 `BLOCK_M=BLOCK_N=128` 下：

| kNumSMs | mcastA | chosen | usage(8) | usage(16) |
|---|---|---|---|---|
| 1  | false | **8**  | 1152 | 2176 |
| 132| false | **8**  | 3200 | 3200（平手→8） |
| 1  | true  | **8**  | 1152 | 2176 |
| 64 | false | **8**  | 2048 | 2560 |

生产（`kNumSMs=1`）恒选 `8`。注意 `kNumSMs` 在 MPK 恒为 1，故 `ceil_div(1,8)=1`，usage 退化成
`8·128+128=1152`——这个「宽度」在 MPK 语境里只影响 `get_swizzled_block_idx` 的 group 大小，
**不再承担「按 SM 数摊」的语义**。

---

## 3. `Scheduler` 构造：把 6 种 GEMM 形态折平（`:38-103`）

模板参数（`:38-51`）在 MPK 里只剩 7 个（DeepGEMM 有 10 个）：

```
MPK :  kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumMulticast, kIsMulticastOnA, kNumSMs
       + SF_K_ALIGNMENT = 512u, kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<...>()
DG  :  ……, kEnsureZeroPadding, kKAlignment, kSFKSpan, kNum1DBlocksPerGroup   // gemm.cuh:34-38
```

字段（`:53-65`）：`current_iter = -1`（tile 计数器，`get_next_block` 的 `++`）、`num_blocks` /
`num_m_blocks` / `num_n_blocks`、`num_blocks_in_group`、`is_peer_cta_alive`（**MPK 里恒 true、
从不被写**——它是 DeepGEMM SM90 multicast 的遗留字段）、`grouped_layout`、以及一组 grouped 用的
游标（`current_group_idx`/`current_m_cumsum`/`current_k_cumsum`/`current_sf_k_cumsum`/…）。

构造函数（`:77-103`）用 `if constexpr` 按 `kGemmType` 初始化：
- `Normal` / `Batched`：`num_blocks = num_m_blocks*num_n_blocks`（`:84-88`）；
- `MGroupedContiguous`：同上 + 存 `grouped_layout`（`:87-89`）；
- `MGroupedMasked` / `MGroupedContiguousWithPsumLayout`：存 `grouped_layout`，后者还读
  `current_psum_m = __ldg(grouped_layout)` 重算 `num_m_blocks`（`:90-96`）；
- `KGroupedContiguous`：`get_next_k_group` 跳过 `shape_k==0` 的组、预取 `next_group_idx`
  （`:97-102`）——`get_next_k_group`（`:67-75`）用 `__ldg` 跳过空组。

**关键**：构造函数支持 6 种类型，但 `get_next_block`（§6）只对 2 种返回 tile。所以对 grouped
类型，构造能算出一堆索引语义（供 `get_global_idx` 用），却**永远拿不到 tile**——这是「DeepGEMM 的
grouped 调度器被截断成 persistent 单 GEMM 调度器」的证据。

---

## 4. `get_swizzled_block_idx`：tile 顺序的 swizzle（`:105-129`）

给定线性 `block_idx`，输出 `(m_block_idx, n_block_idx)`。与 DeepGEMM（`gemm.cuh:108-143`）
逐行同构，**只删掉了 SM90 专用的 `#if __CUDA_ARCH__ < 1000` 段**（`gemm.cuh:122-131`，用来在
SM90 上动态关闭奇数 `num_blocks_in_group` 的 TMA multicast；SM100 用 `cta_group::2` 关不掉，
所以 MPK 里该修正不存在）。核心：

```
primary   = kIsMulticastOnA ? num_n_blocks : num_m_blocks;
secondary = kIsMulticastOnA ? num_m_blocks : num_n_blocks;
num_blocks_per_group = secondary * kNum1DBlocksPerGroup;
group_idx = block_idx / num_blocks_per_group;
first     = group_idx * kNum1DBlocksPerGroup;
in_group  = block_idx % num_blocks_per_group;
num_blocks_in_group = min(kNum1DBlocksPerGroup, primary - first);   // 尾巴截断
if (kIsMulticastOnA) { m = in_group / num_blocks_in_group; n = first + in_group % num_blocks_in_group; }
else                 { m = first + in_group % num_blocks_in_group; n = in_group / num_blocks_in_group; }
```

即「把 primary 维切成 `kNum1DBlocksPerGroup`（=8）宽的 group，group 内沿 primary 走、group 之间
沿 secondary 走」——目的与 DeepGEMM 相同：**提高 L2 命中**（相邻 CTA 访问相邻 tile）。

host 探针 §B 实测**双射性**（每种形状遍历全部 `block_idx ∈ [0, num_blocks)`）：

```
M=32 N=32 blocks (4096x4096)   tiles=1024 unique=1024 oob=0  BIJECTION OK
M=13 N=5 blocks (unaligned)    tiles=  65 unique=  65 oob=0  BIJECTION OK
M=1  N=9  (thin M)             tiles=   9 unique=   9 oob=0  BIJECTION OK
M=17 N=3  (mcastA on N)        tiles=  51 unique=  51 oob=0  BIJECTION OK
M=9  N=9  (small square)       tiles=  81 unique=  81 oob=0  BIJECTION OK
```

包括**非 8 整数倍**（13×5）与「primary 小于 group 宽」（M=1）的尾巴都无越界、无重复——
`num_blocks_in_group = min(...)` 的截断处理正确。这是「swizzle 只改顺序不改集合」的直接证据。

---

## 5. `get_global_idx`：索引语义的四种 offset（`:131-166`）

模板 `get_global_idx<kWithGroupOffset, kIndexType>`（默认 `IndexType::MN`）把「块内偏移」翻译成
「全局张量偏移」（`:137-165`）：
- `Normal`：`block_idx * block_size`（无 offset，`:137-138`）；
- `MGroupedContiguous`：`offset = max(0, __ldg(grouped_layout + m_block*BLOCK_M))`（按行查组起点，
  `:139-144`）；
- `MGroupedMasked`/`MGroupedContiguousWithPsumLayout`：`offset = current_group_idx`（`:145-149`）；
- `KGroupedContiguous`：按 `kIndexType` 选 `current_group_idx`（MN）/`current_k_cumsum`（K）/
  `current_sf_k_cumsum`（SF_K）（`:150-161`）；
- 其余（含 `Batched`）：`offset = (kIndexType==SF_K) ? current_group_idx : 0`（`:162-165`）。

与 DeepGEMM `gemm.cuh:147-181` 逐行同构（MPK 少了 psum 的 `get_aligned_effective_m_in_block`）。
调用点在生产 `GemmType::Normal` 下走 `:137-138`，`kWithGroupOffset` 为参数、`IndexType` 为
`MN`/`K`/`SF_K` 之一（`linear_fp8_sm100.cuh:317-335/378/392`）。

---

## 6. `get_next_block`：MPK 的「单 CTA 顺序扫」（`:168-196`）

MPK 的实现（`:168-196`）：

```
__device__ bool get_next_block(uint32_t& m_block_idx, uint32_t& n_block_idx) {
  // In persistent kernel: use block_offset=0 (each CTA handles its own
  // output slice, kNumSMs=1, processes all tiles sequentially)          // :170-171
  auto const next_block_idx = (++current_iter) * kNumSMs + 0;            // :172
  if constexpr (kGemmType == GemmType::Normal) {                         // :173
    if (next_block_idx >= num_blocks) return false;
    get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
    return true;
  } else if constexpr (kGemmType == GemmType::Batched) {                 // :179
    if (next_block_idx >= num_blocks * kNumGroups) return false;
    current_group_idx = next_block_idx / num_blocks;
    /* Batched 的 m/n 直接算（无 group swizzle） */                       // :183-192
    return true;
  } else {
    return false;                                                        // :193-195
  }
}
```

DeepGEMM 的对应行（`gemm.cuh:186`）是：

```
const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;
```

**两处差别**：
1. 偏移 `+0`（MPK）vs `+blockIdx.x`（DeepGEMM）——`blockIdx.x` 是「第几个 SM/CTA」，几个 CTA
   各取「同一 tile 序列里属于自己的那条 residue」；MPK 硬编码 0，配合 `kNumSMs=1` 就退化成
   `next_block_idx = current_iter`，即**顺序遍历 0,1,2,…**。
2. MPK 删了 DeepGEMM `get_next_block` 里 3 个 grouped 分支（`gemm.cuh:188-246`）、`is_peer_cta_alive`
   的 Normal 分支赋值（`gemm.cuh:265-267`）与 `is_tma_multicast_valid`（`gemm.cuh:275`）。

### 6.1 「顺序扫」覆盖全部 tile（host 实测）

`src/fp8_scheduler_order.cu` §C：MPK `kNumSMs=1` 对 M=32×N=32 / 13×5 / 1×9 / 7×3 **全部
`COVERS ALL ONCE`**（visits==unique==num_blocks）。

### 6.2 HACK 注释与发射值矛盾（host 实测放大）

生产发射处的注释（`task_register.cc:4818-4825`）写：

> HACK: The new kernel uses `blockIdx.x` for internal tile scheduling. In persistent kernel,
> `blockIdx.x` is the worker index (0..num_workers-1), not the task's grid block index.
> …… We use **`MPK_MAX_NUM_WORKERS`** (defined at compile time) as `kNumSMs`.

但**实际发射的是 `1`**（`task_register.cc:4839`：`code.e("    1,"); // kNumSMs (persistent
kernel: 1 CTA per task, processes all tiles)`），且 `MPK_MAX_NUM_WORKERS` 在整个快照里
**只出现在这条注释里、没有任何定义**（`harness/fp8_scheduler_order.sh` 末尾的 audit：
`MPK_MAX_NUM_WORKERS occurrences in whole MPK snapshot: 1`）。

host 探针 §D 把这个注释的**字面提议**当 `kNumSMs` 代入（`W=1056`）算出后果：

```
kNumSMs=1 (as shipped)   M=32 N=32 (num_blocks=1024)  visits=1024  COVERS ALL ONCE
kNumSMs=W=1056 (comment)                              visits=   1  GAP/DUP
DeepGEMM kNumSMs=132                                  visits=1024  COVERS ALL ONCE
```

即：注释描述的机制（用 worker 数当步长、偏移 0）会**只访问 tile#0**，与 MPK 已删掉 `+blockIdx.x`
的选择自相矛盾——「`kNumSMs` 保持 1 + 偏移 0」才是覆盖全部 tile 的正确组合，注释是**陈旧的**
（且引用了一个不存在的宏）。这是本专题反复见到的「**按注释推断代码行为会错**」的又一例
（对照 `docs/60` §?「XOR 注释」、`docs/62` §?「idescE 注释」）。

### 6.3 DeepGEMM 的多 SM sweep 是「划分」不是「重叠」（host 实测）

`src/fp8_scheduler_order.cu` §E：DeepGEMM 形式（`kNumSMs=132`、`blockIdx.x ∈ [0,132)`）的
**并集**对 1024 / 65 / 51 个 tile 全部 `COVERS ALL ONCE`——因为 `next = iter*S + bid` 按 `bid`
把 tile 序列**划分**成 S 条 residue，每条不重叠。所以「MPK 单 CTA 全扫」与「DeepGEMM 132 个 CTA
各扫 1/132」在**覆盖集合**上等价，区别只在并行度与中间量/barrier 的落点。

> 与 megakernel 主题的连接：DeepGEMM 的 `Scheduler` 假设「一个 grid、多个 CTA、每个 CTA 用
> `blockIdx.x` 认领 tile」——这是**grid 级**的静态划分。MPK 的持久化内核里 `blockIdx.x` 已被
> worker 角色占用（`docs/06` §?/`docs/07`），所以调度器只能退化成「一个 CTA 独立走完自己的
> tile 列表」。**同一个算法在两套 launch 约定下必须改写**——这正是 `docs/10` 记的
> 「融合边界不是越融越好，launch 约定会反向约束 device 身体」的另一面。

---

## 7. `_epilogue_utils.cuh`（30 行）：一个零调用点的 head-split 映射

`include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_epilogue_utils.cuh`：

- `EpilogueIdentity`（`:7-14`）：`apply_index_n(n) = n`（恒等）；
- `EpilogueHeadSplits<kLeft,kMid,kRight>`（`:16-30`）：`apply_index_n(n) = n + (n+ kRight)/(kLeft+kRight)*kMid`
  ——把 N 维（= MLA 输出维，由多个 head 段拼接）做一次**列重排**，`DG_STATIC_ASSERT` 三个段都是
  `STORE_BLOCK_N` 的整数倍（`:25-28`）。

接线：`epilogue_type_t::apply_index_n<STORE_BLOCK_N>` 在 `linear_fp8_sm100.cuh:623`（epilogue 写
N 索引）与 `:303`（residual 读 N 索引）被调用。全仓引用（`harness` audit 之外手工 grep）：

```
EpilogueIdentity  : production task_register.cc:4848 + 单测 runtime_kernel_wrapper_sm100.cu:604/633/834
EpilogueHeadSplits: 仅定义处（epilogue_utils.cuh:17），零实例化
```

即 **`EpilogueHeadSplits` 有实现、零调用点**——即便在这个 30 行的小文件里也有「为实现而实现、
并未接线」的一支（与 `docs/60` §6 的 `_cute_tie`、`docs/49` 的接线审计同族）。DeepGEMM 里
**没有** `EpilogueHeadSplits`（grep 零命中），故这是 MPK 为 MLA head-split 输出专门准备的、
尚未接上的钩子。

---

## 8. 与 DeepGEMM 调度器逐项对照

| 维度 | MPK `_scheduler.cuh`（199 行） | DeepGEMM `scheduler/gemm.cuh`（314 行） |
|---|---|---|
| `get_num_1d_blocks_per_group` | `:14-34`，候选 `{8,16}` | `:15-30`，逐字相同 |
| 模板参数 | 7 + `SF_K_ALIGNMENT`（`:38-45`） | 10：多 `kEnsureZeroPadding`/`kKAlignment`/`kSFKSpan`（`:34-38`） |
| `get_swizzled_block_idx` | `:105-129`，**无** SM90 段 | `:108-143`，含 `#if __CUDA_ARCH__<1000` 修正（`:123-131`） |
| `get_global_idx` | `:131-166` | `:147-181` |
| `get_aligned_effective_m_in_block` | 删除（0） | `:177-184`（psum 专用） |
| `get_next_block` Normal/Batched | `:173-195` | `:185-269`（另含 3 个 grouped 分支） |
| tile 偏移 | `+0`（硬编码，`:172`） | `+ blockIdx.x`（`:186`） |
| `is_peer_cta_alive` | 字段在、**从不被写** | Normal 分支写（`:265-267`） |
| `is_tma_multicast_valid` | 删除（0） | `:275` |
| 生产发射 | `kNumSMs=1`、`GemmType::Normal`（`task_register.cc:4839/4844`） | grid 多 CTA、`kNumSMs`=SM 数 |

函数计数（`harness/fp8_scheduler_order.sh` audit）亦证：
`is_tma_multicast_valid` MPK=0 / DG=1，`get_aligned_effective_m_in_block` MPK=0 / DG=1。

---

## 9. 实现口径与只记不判

- **`is_peer_cta_alive` 是死字段**：MPK 保留了声明与初值 `true`（`:58`），但 `get_next_block` 的
  Normal 分支不再写它（DeepGEMM 会写，`gemm.cuh:265-267`）——它只在 DeepGEMM SM90 multicast
  路径被读。MPK 生产 `kNumMulticast=1`，本就用不到。
- **构造支持 grouped、`get_next_block` 不支持**：两者不一致是**有意的**（生产只发 `Normal`），
  但如果有人在前端把 grouped GEMM 接到这个 task，会在**运行期静默产出 0 个 tile**（`get_next_block`
  直接 `return false`），而非编译期报错——`docs/60`/`docs/63` 记的「错配是编译期错」在这里是
  **反例**（只记不判，未在 SM100 上验证）。
- **`SF_K_ALIGNMENT=512u` 是模板默认**（`:45`），但它在 `Scheduler` 体内**从未被使用**（当前
  `get_global_idx` 的 SF_K 用 `current_sf_k_cumsum`）——又一枚「声明了但没用的参数」（对照
  `docs/55` 的 `WARPS_PER_CTA` 悬空模板参数）。
- **HACK 注释引用不存在的宏**（§6.2）：`MPK_MAX_NUM_WORKERS` 全仓 1 处（注释本身）。不构成对
  上游的 bug 判定（代码行为正确），但注释会误导读者以为走的是 worker 数。

---

## 10. 判据映射（C1–C5）

- 本文件是**单 task device 身体的内部调度层**——它不改变 task 的提交方式，故对 C1「单次 launch」
  无贡献；它是 `docs/07`（单 task device 实现）/`docs/32`（SM100 linear 身体）已经贡献的 **C3
  「范围 = 多算子/多 tile」在 tile 粒度上的补全**：`Scheduler` 让**一个 CTA**覆盖**它负责的全部
  输出 tile**（host 实测覆盖全集）。
- **不满足 C4/C5**：`get_next_block` 是**静态、确定性**的 tile 列表（`blockIdx.x`/`kNumSMs`
  在 launch 时定死），没有任何设备内动态派发、没有 worker/scheduler 角色划分——与
  `docs/03` §3 的「去中心化调度」是两回事。故判据**只贡献 C3**（同 `docs/32`、`docs/33` 这一层）。

---

## 11. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（本地 `/ssd/mirage_mpk_build`）。
- **主文件**：`include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_sm100_scheduler.cuh`
  （`IndexType :8-12`；`get_num_1d_blocks_per_group :14-34`；`struct Scheduler` 模板 `:38-51`、
  字段 `:53-65`、`get_next_k_group :67-75`、构造 `:77-103`、`get_swizzled_block_idx :105-129`、
  `get_global_idx :131-166`、`get_next_block :168-196`，tile 偏移硬编码 `:172`）。
- **epilogue utils**：`.../blackwell/linear_fp8_sm100_epilogue_utils.cuh`
  （`EpilogueIdentity :7-14`、`EpilogueHeadSplits :16-30`）。
- **调用方**：`.../blackwell/linear_fp8_sm100.cuh`（`task_impl :45`；`Scheduler` 构造 `:274-282`；
  warp0 TMA `:290-402`、warp1 UMMA `:403-534`、warp2 UTCCP `:535-575`、epilogue `:576-736`；
  `get_next_block` 调用点 `:291/439/549/593`；`get_global_idx` 调用点 `:300/317/321/328/378/392`；
  `apply_index_n` `:303/623`）。
- **生产发射**：`src/kernel/task_register.cc`（`register_linear_fp8_sm100_task :4772`、
  HACK 注释 `:4818-4825`、`kNumSMs=1` `:4837-4840`、`GemmType::Normal` `:4844`、
  `EpilogueIdentity` `:4848`、模板实参 `:4826-4868`）。
- **DeepGEMM 对照**（`~/github/DeepGEMM`，commit `057ca59`）：
  `deep_gemm/include/deep_gemm/scheduler/gemm.cuh`（`get_num_1d_blocks_per_group :15-30`、
  `struct Scheduler :39`、`get_swizzled_block_idx :108-143`（SM90 段 `:123-131`）、
  `get_global_idx :147-181`、`get_aligned_effective_m_in_block :177-184`、
  `get_next_block :185-269`（偏移 `:186`、`is_peer_cta_alive :265-267`）、
  `is_tma_multicast_valid :275`）。
- **本专题实验**：`src/fp8_scheduler_order.cu` + `harness/fp8_scheduler_order.sh` +
  `harness/fp8_scheduler_order.out.txt`（目标卡 H100/sm90a，host 侧算术，device 3，未 launch kernel）。
- **相关篇**：`docs/32`（本文补其 §2.3/§7 尾巴）、`docs/33`（MoE grouped GEMM 走另一支）、
  `docs/60`（`_cute_tie` 零调用点）、`docs/49`（接线审计）、`docs/07`（Hopper 单 task device）、
  `docs/03` §3（去中心化调度，与本层的静态 tile 列表对照）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （device task 内部实现，同 `docs/28`–`docs/63`）。

---

## 12. 局限（诚实标注）

- **纯源码走读 + host 算术探针，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，
  未取任何 device 计数；探针只重放**纯组合**的 tile 遍历算术（双射/覆盖），不代表真实 kernel 的
  时间行为。第 9 节的「grouped 会静默产 0 tile」是**静态推断**，未在 SM100 上触发验证。
- **未与 DeepGEMM SM100 路径对跑**：DeepGEMM 的 grouped/masked/psum 分支、`is_tma_multicast_valid`
  语义只在源码层对照，未在 SM100 上核数值。
- **未走的相邻层**：`linear_sm100_mpk.cuh`（非 FP8 的 SM100 linear）里的调度是否同构未核；
  `linear_fp8_sm100_tma_utils.cuh` / `linear_fp8_sm100_sm100_utils.cuh` 只借用了行号
  （`docs/32` 已给 `:16-116` 摘要），未逐行。
