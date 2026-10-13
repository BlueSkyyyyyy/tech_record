# DeepGEMM 的通用 GEMM 任务调度器：`scheduler/gemm.cuh`

> 前面 107 篇从 Mega MoE 主 kernel 一路拆到 DeepEP 的通信原语、测试栈、JIT 包装层、共享内存布局、
> 引用体检……但有一个文件被**反复引用却从没被单独读过**：`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`
> （314 行）。它是 `Scheduler`——**所有** DeepGEMM 稠密 / grouped / MoE 反向 GEMM 共用的「任务枚举器」：
> 把 `(BLOCK_M, BLOCK_N)` 的二维块网格 + 分组维度，翻译成每个 SM 该做哪些 `(group, m_block, n_block)`，
> 并回答「这个块在全局张量里的偏移是多少」。第 12/13 篇讲反向 wgrad/dgrad 时、第 21/18 篇讲 einsum 时、
> 第 39/43/103 篇讲 legacy / padding / swap-AB 时，都只贴了它的某几行行号，从没讲清**它整体怎么工作、为什么**。
>
> 本篇把它拆透。它唯一的架构分支是 `#if __CUDA_ARCH__ < 1000`（SM90 的「不齐整 TMA multicast」修补），
> 其余全是整数索引运算，**不碰 UMMA / TMA / tcgen05**——所以能在 H100(sm90) 上**直接 include 真实头文件、
> 编译运行真实设备代码**，逐字段对拍。SM100 分支只做编译期覆盖（同一 `.cu` 同时编 `sm_90a` 与 `sm_100a`）。
> 全部实测 **51 PASS / 0 FAIL**（`harness/exp_gemm_scheduler.{sh,out.txt}` + `src/gemm_scheduler.cu`，
> 源文件 sha256 `f48dfef6…`）。

---

## 1. 它解决什么问题：7 种 GEMM 形态、一个调度器

DeepGEMM 对外暴露非常多的 GEMM 入口（普通稠密、m-grouped、k-grouped、batched/einsum……），但**内核里
枚举任务块的那段逻辑是同一份代码**。`GemmType`（`common/types.cuh:13-21`）枚举了七种语义：

```
Normal=0, MGroupedContiguous=1, MGroupedMasked=2, KGroupedContiguous=3,
Batched=4, MGroupedContiguousWithPsumLayout=5, KGroupedContiguousWithPsumLayout=6
```

`Scheduler`（`scheduler/gemm.cuh:30-39`）把「形状 + 分组表」编译期特化成其中一种，对外只有四个方法：

- `get_next_block(m_block_idx, n_block_idx)`（`:185-272`）——每个 CTA 循环拉下一个任务块；
- `get_swizzled_block_idx(...)`（`:108-144`）——把线性块号映射成 `(m,n)`，带 L2 swizzle；
- `get_global_idx<withOffset, IndexType>(...)`（`:146-174`）——块号 → 全局张量偏移；
- `get_aligned_effective_m_in_block(...)`（`:176-183`）——swap-AB + psum 尾块的有效 M。

它的五个调用方（`grep` 全仓）：

| 内核 | 实例化位置 | 用到的 GemmType |
|---|---|---|
| `sm90_bf16_gemm.cuh` | `:135` | Normal / MGroupedContiguous / KGroupedContiguous |
| `sm90_fp8_gemm_1d1d.cuh` | `:161`（硬编 `kKAlignment=128, kSFKSpan=128`） | 同上 |
| `sm90_fp8_gemm_1d2d.cuh` | `:153` | 同上 |
| `sm100_bf16_gemm.cuh` | `:158` | 全部 7 种（含 psum 布局） |
| `sm100_fp8_fp4_gemm_1d1d.cuh` | `:199`（`kSFKSpan = kGranKA*4`） | 全部 7 种 |

也就是说：**MoE 反向的 wgrad（`k_grouped_*`）与 dgrad（`m_grouped_*`）走的就是这个调度器**
（反向 API 详见第 12/13 篇，swap-AB 详见第 103 篇）。理解它，等于把「反向 grouped GEMM 到底怎么把
一个 group 的 K 轴切块、怎么处理不等长 M」一次性讲清。

> Mega MoE 主 kernel 用的 `sched::MegaMoEScheduler`（`sm100_fp8_fp4_mega_moe.cuh:292`）是**另一个**
> 调度器（第 3/9 篇拆过），不在本篇范围。

---

## 2. `get_num_1d_blocks_per_group`：块分组大小的启发式（`:14-26`）

```cpp
for (const auto candidate: {8u, 16u}) {
    const auto usage = kIsMulticastOnA ?
        candidate * BLOCK_N + ceil_div(kNumSMs, candidate) * BLOCK_M :   // 组沿 N
        candidate * BLOCK_M + ceil_div(kNumSMs, candidate) * BLOCK_N;    // 组沿 M
    if (usage < min_usage) min_usage = usage, num_best_blocks = candidate;
}
```

它在 `{8,16}` 里选一个 `kNum1DBlocksPerGroup`（记为 `N1D`），作为 `get_swizzled_block_idx` 里
「一个 group 内放多少个 primary 方向的块」。`usage` 是一个**代理代价**：
`candidate * <primary block 宽> + ceil(SMs/candidate) * <secondary block 宽>`。第一项是一个组内
连续 primary 块的跨度（决定该组吃的 L2 行数），第二项是每 SM 分到的 secondary 迭代数。
选小的那个 = 「让一次 L2 复用的组跨度尽量小」。

注意这个函数是 `static constexpr`（`:15`），在**编译期**算完直接当 `Scheduler` 的最后一个模板参数
（`:38` 的 `kNum1DBlocksPerGroup = get_num_1d_blocks_per_group<...>()`）。实测它与暴力版
（`src/gemm_scheduler.cu` 的 `host_num_1d_blocks`）逐一相等：

```
A[BM128 BN128 132SM MA0] -> N1D=8   A[BM128 BN256 132SM MA0] -> N1D=16
A[BM128 BN128 132SM MA1] -> N1D=8   A[BM128 BN256 132SM MA1] -> N1D=8
A[BM64  BN128  78SM MA0] -> N1D=16  A[BM128 BN128   8SM MA0] -> N1D=8   (6/6 PASS)
```

`kIsMulticastOnA`（`MA`）是关键：A 操作数做 TMA multicast 时，连续 CTA 共享**同一块 A**，于是
分组的「主轴」是 N（组沿 N 排）；否则主轴是 M。

---

## 3. `get_swizzled_block_idx`：L2 swizzle + SM90 multicast 修补（`:108-144`）

任务块不是按行主序发，而是**按组 swizzle**，让同时活跃的 CTA 尽量落在同一片 L2 数据上：

```
primary   = MA ? num_n_blocks : num_m_blocks
secondary = MA ? num_m_blocks : num_n_blocks
num_blocks_per_group = secondary * N1D
group_idx = block_idx / num_blocks_per_group
first     = group_idx * N1D
in_group  = block_idx % num_blocks_per_group
num_blocks_in_group = min(N1D, primary - first)      // 末尾组可能不足 N1D
```

然后（`MA` 为真，组沿 N）`m = in_group / num_blocks_in_group; n = first + in_group % num_blocks_in_group`；
否则 `m = first + in_group % ...; n = in_group / ...`。

### 3.1 为什么有 `#if __CUDA_ARCH__ < 1000` 的一段（`:120-133`）

```cpp
#if __CUDA_ARCH__ < 1000
    if (kNumMulticast > 1 and num_blocks_in_group % 2 != 0) {      // :124
        if (in_group_idx < (num_blocks_in_group ^ 1) * secondary_num_blocks) {
            num_blocks_in_group = num_blocks_in_group ^ 1;         // :126  丢掉最后 1 个
        } else {
            in_group_idx = in_group_idx - (num_blocks_in_group ^ 1) * secondary_num_blocks;
            first_block_idx += num_blocks_in_group ^ 1;
            num_blocks_in_group = 1;                               // :130  剩下的自成一组
        }
    }
#endif
```

原因写在注释里（`:120-122`）：**SM90 可以动态关闭 TMA multicast，但 multicast 必须成对（`kNumMulticast=2`）
给出对齐的分组；SM100 用固定的 2-CTA cluster，无法动态关闭**。所以 SM90 上当一个组内的 primary 块数是
**奇数**时，要把最后 1 个块单独拆出来（`nbig^1` 是「去掉最低位」= 向上取到偶数），否则 2-CTA multicast
会读到越界的对端。`^ 1` 是这段代码的记号：`nbig ^ 1` 把 `nbig` 变成小于它的最大偶数。

### 3.2 H100 实测（穷举 block_idx）

我们在 sm90 上把 `get_swizzled_block_idx` 对 `[0, num_m*num_n)` **穷举**，与独立复刻逐位比
（`host_swizzle`），并检查映射是双射、不越界：

```
B[Normal onM NM2 r7c5]   7x5 MA=0 N1D=8 -> mismatch=0 oob=0 non-bijective=0 fixup_hits=35
B[Normal onN NM2 r5c7]   7x5 MA=1 N1D=8 -> mismatch=0 oob=0 non-bijective=0 fixup_hits=35
B[Normal onM NM2 r8c6]   8x6 MA=0 N1D=8 -> mismatch=0 oob=0 non-bijective=0 fixup_hits=0
B[MGrouped NM2 r13c3]  13x3 MA=0 N1D=8 -> mismatch=0 oob=0 non-bijective=0 fixup_hits=15
B[KGrouped  NM2 r5c13]  5x13 MA=0 N1D=8 -> mismatch=0 oob=0 non-bijective=0 fixup_hits=65
B[Normal onM NM2 r9c5 N1D16] -> mismatch=0 oob=0 non-bijective=0 fixup_hits=45
（10 组全 mismatch=0 / oob=0 / 非双射=0）
```

`fixup_hits` 是「修补真的改变了映射」的块数：`7x5`（primary=7 是奇数）→ **35** 个块被重排；
`8x6`（primary=8 是偶数）→ **0**。坐实这段 `#if` 不是摆设。注意 sm90 运行走的是**修补分支**，与 H100 上
真实反向 grouped GEMM 完全一致。**SM100 分支（无修补）只做了编译期覆盖**（同 `.cu` 编 `sm_100a` 成功），
本机无法运行验证。

---

## 4. `get_next_block`：七种形态的任务枚举（`:185-272`）

每个 CTA 的迭代是**网格步进**：`next_block_idx = (++current_iter) * kNumSMs + blockIdx.x`（`:186`）。
`kNumSMs` 进模板参数，于是「第 i 轮的第 b 个 CTA」拿到的线性块号是确定的——这保证 N 个 SM 恰好覆盖
`[0, N)` 一次。`current_iter` 从 `-1` 起（`:45`），`get_next_block` 返回 `false` 表示本轮任务已尽。

七种形态的差异全在「线性块号 `next_block_idx` 如何翻译成 `(group, m_block, n_block)`」：

- **`Normal` / `MGroupedContiguous`**（`:260-270` else 分支）：`num_blocks = num_m_blocks*num_n_blocks`
  （`:91-94`），`next_block_idx >= num_blocks` 即停。两者**打任务的方式相同**；区别只在
  `get_global_idx` / `is_computation_valid` 里怎么用 `grouped_layout`。`MGroupedContiguous` 的分组信息
  **不进 `get_next_block`**。
- **`MGroupedMasked`**（`:188-204`）：每个组的 `num_m_blocks = ceil(grouped_layout[g]/BLOCK_M)` 都不同，
  循环里维护 `current_m_cumsum`（已处理组的 m_block 累计），`next_block_idx == current_m_cumsum*num_n_blocks`
  处切组。任务数 = `Σ_g ceil(count_g/BM) * num_n_blocks`。
- **`MGroupedContiguousWithPsumLayout`**（`:205-225`）：`grouped_layout[g]` 是**累积 psum 结尾**（不是
  单组计数）。切组时 `last_psum_m = align(current_psum_m, BLOCK_M)`、`num_m_blocks = ceil((psum_g - last)/BM)`
  （`:216-219`），最后把 `m_block_idx += last_psum_m/BLOCK_M` 抬到本组全局 M 偏移（`:225`）。含义见 §7。
- **`is_k_grouped_contiguous`**（`:226-246`，含两个 K-grouped 变体）：M/N 固定，`num_blocks =
  num_m_blocks*num_n_blocks`；每切一组就更新物理 K 起点与 SF-K 前缀和（下一节）。
- **`Batched`**（`:247-259`）：`current_group_idx = next_block_idx / num_blocks`，任务数
  `num_blocks * kNumGroups`。

### 4.1 H100 实测：任务覆盖性是 `[0,N)` 的排列

我们让 `kNumSMs=8` 个 CTA 各跑真实 `get_next_block` 直到 `false`，收集所有记录的 `(nbi, group, m, n)`，
检查「每一条 `nbi ∈ [0,N)` 恰好出现一次」：

```
C[Normal 8SM]           tasks=24/24  miss=0 oob=0 dup=0
C[Batched 8SM G4]       tasks=96/96  miss=0 oob=0 dup=0
C[MGroupedContig 8SM]   tasks=24/24  miss=0 oob=0 dup=0
C[MGroupedMasked 8SM]   tasks=27/27  miss=0 oob=0 dup=0     # 组计数 {300,100,500,100}
C[MGroupedPsum 8SM]     tasks=24/24  miss=0 oob=0 dup=0     # psum 结尾 {256,300,512,1000}
C[KGroupedContig 8SM]   tasks=96/96  miss=0 oob=0 dup=0
C[KGroupedPsum 8SM KA=128] tasks=96/96  miss=0 oob=0 dup=0
C[KGroupedPsum 8SM KA=256] tasks=96/96  miss=0 oob=0 dup=0
```

`N` 全部由**规范**独立算出（masked：`Σceil(count_g/BM)*num_n`；psum：`Σceil((psum_g-align(psum_{g-1}))/BM)*num_n`），
不是照抄设备实现。**8 种形态全部恰好覆盖一次**（`miss=0/oob=0/dup=0`）。

### 4.2 K-grouped 的物理 K 起点与 SF-K 前缀和（`get_next_k_group:71-82`）

k-grouped 的两种变体决定「一个 group 的 K 从哪里开始、SF 行怎么编号」：

- 非 psum（`KGroupedContiguous`）：`current_k_start += current_shape_k; current_shape_k = grouped_layout[g]`
  （`:79-80`）——`grouped_layout[g]` 是**单组 K 长度**，起点是前缀和。
- psum（`KGroupedContiguousWithPsumLayout`）：`next_k_end = grouped_layout[g]`（累积结尾）、
  `current_k_start = align(current_k_end, kKAlignment)`、`current_shape_k = next_k_end - current_k_start`
  （`:74-77`）——**起点被对齐到 `kKAlignment`，长度 = 未对齐的结尾减对齐后的起点**。

每次从组 g 切到 g+1，还累加 SF 行号：`current_sf_k_cumsum += ceil_div(align(current_shape_k, kKAlignment), kSFKSpan)`
（`:241-242`），其中 `kSFKSpan` 是「一行 k-grouped SF 覆盖的 K 元素数」（SM100 FP8 设 `kGranKA*4`）。
模板里有一条硬约束：`DG_STATIC_ASSERT(not is_k_grouped_contiguous(kGemmType) or kKAlignment % 128 == 0)`
（`:42-43`）——**K 分组起点必须是 BLOCK_K(=128) 的整数倍**，否则物理切块会跨 block。

H100 实测把每个组的 `(k_start, sf_k_cumsum)` 与规范逐组比：

```
C[KGroupedContig 8SM]        k-group (kstart,sfcum) bad=0
C[KGroupedPsum 8SM KA=128]   k-group (kstart,sfcum) bad=0
C[KGroupedPsum 8SM KA=256]   k-group (kstart,sfcum) bad=0    # 不齐整端点 {300,900,1100,2000}
```

`KA=256` 那组故意用**不齐整的 psum 端点**：组 1 的物理 K 起点被对齐到 `align(300,256)=512`、长度
`900-512=388`（而非 600），组 2 起点 `align(900,256)=1024`、长度 `1100-1024=76`。设备与规范 `bad=0`，
把 `get_next_k_group` 的「对齐语义」钉死。

---

## 5. `get_global_idx`：块号 → 全局张量偏移（`:146-174`）

```cpp
template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
uint32_t get_global_idx(shape_dim, block_size, block_idx, m_block_idx=0);
```

- `IndexType::MN`（`:146`: `MN`）：A/B/C/D 的 M 或 N 方向；
- `IndexType::K`（`:10`）：K 方向（`block_size` 通常是 `BLOCK_K`）；
- `IndexType::SF_K`（`:11`）：scale-factor 的 K 方向。

各形态的偏移规则（`:149-173`）：

| GemmType | 偏移来源 |
|---|---|
| `Normal` | 恒 `block_idx*block_size`，忽略 group |
| `MGroupedContiguous` | `kWithGroupOffset` 时 `offset = max(0, grouped_layout[m_block_idx*BLOCK_M])`（组行起点）|
| `MGroupedMasked` / `MGroupedContiguousWithPsumLayout` | `offset = current_group_idx`（组号直接乘 `shape_dim`）|
| `is_k_grouped_contiguous` | 视 `kIndexType`：MN→`current_group_idx*shape_dim`；K→`current_k_start`；SF_K→`current_sf_k_cumsum` |
| `Batched` | 忽略 `kWithGroupOffset`；`SF_K` 用 `current_group_idx`，其余不用 |

一个易读错的点（我们的实验先踩了）：`MGroupedContiguous` 读的是**第 4 个参数** `m_block_idx` 指向的
`grouped_layout[m_block_idx*BLOCK_M]`，**不是第 3 个参数 `block_idx`**。所以对 A（只传 3 个参数、第 4 个
默认 0）来说，它读的是 `grouped_layout[0]`；真正需要组偏移的是 K-major 的 B（`sm90_bf16_gemm.cuh:177`
显式把 `m_block_idx` 当第 4 参传进去）。实测（`G[...]`）：

```
G[Normal]             MN(noff)=256/256 MN(off)=256/256  K=128/128 SFK(off)=1/1 SFK(noff)=1/1
G[MGroupedContig peer-eq]  MN(off)=256256/256256       # gl[3*128]=256 -> 256*1000+256
G[MGroupedContig peer-neq] MN(off)=1256/1256           # gl[3*128]=1   -> 1*1000+256
G[MGroupedMasked]     MN(off)=256/256                  # 用 current_group_idx(=0)
G[MGroupedPsum]       MN(off)=256/256                  # 同上
G[Batched]            SFK(off)=1/1 SFK(noff)=1/1
```

`MGroupedContiguous` 两行只差 `grouped_layout` 内容，偏移从 `256256` 变 `1256`，正是「第 4 参决定组行起点」
的直接证据。

---

## 6. psum 尾块：`get_aligned_effective_m_in_block`（`:176-183`）

```cpp
if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout and not kEnsureZeroPadding)
    return align(m_block_idx == last_psum_m/BLOCK_M + num_m_blocks - 1
                 ? current_psum_m - m_block_idx*BLOCK_M : BLOCK_M, 16);
return BLOCK_M;
```

仅对 **swap-AB + psum 布局 + 关零填充**（`kEnsureZeroPadding=false`）生效：Swap-AB 下 M 是可变短维
（第 103 篇），一个组的**最后一个块**可能不足 `BLOCK_M` 行，这时把有效 M 对齐到 **16**（`UMMA_STEP_N`，
SM100 UMMA 的 N 原子），避免 UMMA 读到越界行。其余情形恒 `BLOCK_M`。

实测：psum 组（`last_psum_m=256, current_psum_m=400, num_m_blocks=2`）尾块下标 = `256/128 + 2 - 1 = 3`，
`400 - 3*128 = 16` → 尾块有效 M = **16**，非尾块 = **128**；其它形态恒 128：

```
D[MGroupedPsum]  tail_eff_m=16/16 head_eff_m=128/128
D[Normal / MGroupedContig / MGroupedMasked / Batched]  tail_eff_m=128/128
```

---

## 7. `MGroupedContiguousWithPsumLayout`：psum 布局到底是什么

第 12 篇讲过「PSUM 是逻辑 K，contiguous 是物理 K」；在 m-grouped 侧，`grouped_layout[g]` 存的是
**每组 m 的累积结尾**（`psum_g`）。于是：

- 组 g 的 m_block 数 = `ceil((psum_g - align(psum_{g-1}, BLOCK_M)) / BLOCK_M)`（`:216-219`）；
- 组 g 在全局 M 里的偏移 = `align(psum_{g-1}, BLOCK_M)`（`:225` 的 `m_block_idx += last_psum_m/BLOCK_M`）。

`align(...)` 让**组起点按 BLOCK_M 对齐**，组内尾块再靠 `get_aligned_effective_m_in_block` 裁到 16。
这与第 103 篇 swap-AB 的「逐块动态 UMMA N」是同一套机制的两端。`is_computation_valid`（`:294-309`）
在 SM90 上用它判断某 warpgroup 的 M 是否落在 `current_psum_m` 之内：

```
valid(m0)=1/1  valid(m2)=1/1   # m=2 -> 256 < 400
```

---

## 8. SM90 专属的两个校验：multicast 与计算有效性

### 8.1 `is_tma_multicast_valid`（`:274-292`）

仅 SM90 有效（`:274` 注释）。组内只剩 1 个块时返回 `false`（`:276-277`，此时没有对端可 multicast）。
`Normal / Masked / k-grouped / Batched / psum` 直接返回 `true`；**`MGroupedContiguous` 走 K-major 时
额外要求「peer CTA 同属一组」**（`:287-289`）——因为 multicast 的两个 CTA 会一起加载同一块权重，跨组
就会拿到别的专家的权重。实测：

```
F[MGroupedContig NM2 peer-eq]  tma_mcast_valid=1/1   # gl[0]==gl[128]
F[MGroupedContig NM2 peer-neq] tma_mcast_valid=0/0   # gl[0]!=gl[128] -> 关掉 multicast
```

### 8.2 `is_computation_valid`（`:294-309`）

仅 SM90（`:294`）。`Normal`/`Batched` 恒真；`MGroupedContiguous` 看 `grouped_layout[...] >= 0`（负组号
= padding 行，不参与计算）；`MGroupedMasked`/psum 看组内 M 上界。这就是第 39/43 篇「padding 行要保证
D 零、且不参与计算」在调度器里的落点。

---

## 9. 与 Mega MoE 反向的关系（为什么这篇对专题重要）

MoE 反向的四个 grouped GEMM（第 4/12/13/14 篇）恰好落在两个 k-grouped + 两个 m-grouped 形态上：

- **wgrad**（`k_grouped_*`）：`D[e] = A_e^T B_e`，K 轴分组——走 §4.2 的 `current_k_start/sf_cumsum`。
  第 12 篇讲 SM90 逐组原地改写 tensormap 时用到的「组 K 起点」，正是这里 `get_next_k_group` 算出来的；
  第 26 篇的 SF 逐组打包、第 91 篇的 K-grouped 预打包契约，也对应 `current_sf_k_cumsum`。
- **dgrad**（`m_grouped_*`）：M 轴分组且**逐组不等长**——走 `MGroupedMasked`（masked 布局，
  `grouped_layout[g]`=单组行数）或 `MGroupedContiguousWithPsumLayout`（psum 布局，
  `grouped_layout[g]`=累积结尾）。第 13 篇、第 103 篇（swap-AB 强制 m-grouped）依赖的
  `get_aligned_effective_m_in_block` 就在 §6。

换句话说，**反向 grouped GEMM 的「任务怎么分、偏移怎么算、尾块怎么裁、multicast 怎么对齐」全部由这
314 行决定**；本篇把它的每个分支都跟规范对拍、0 不符。

---

## 10. 结论

- `scheduler/gemm.cuh` 是 DeepGEMM **所有** GEMM / grouped GEMM 的公共任务调度器：一个模板结构体
  覆盖 7 种 `GemmType`，对外四个设备方法。
- 三个「非平凡点」：
  1. `get_num_1d_blocks_per_group` 在**编译期**用「组跨度 + 每 SM 迭代数」的代理代价在 `{8,16}` 里选
     块分组大小，且随 `kIsMulticastOnA` 改变分组主轴（N vs M）；
  2. `get_swizzled_block_idx` 的 `#if __CUDA_ARCH__ < 1000` 是 **SM90 TMA multicast 必须成对**的修补
     （奇数分组拆出最后 1 块），SM100 的固定 2-CTA cluster 不需要也无法动态关闭；
  3. k-grouped 的 `get_next_k_group` 定义了「物理 K 起点对齐到 `kKAlignment`、长度 = 未对齐结尾 −
     对齐起点」以及 `SF-K` 前缀和，是 wgrad 的 K 分组与 SF 打包的地基。
- **H100(sm90) 直接 include 真实头文件、编译运行真实设备代码 51 PASS / 0 FAIL**：启发式 6/6、swizzle
  穷举 10 组（含 `fixup_hits=35/65/45` 的奇数分组）全部双射且与独立复刻逐位相等、8 种形态任务覆盖
  恰好一次、k-grouped 逐组 `(k_start, sf_cumsum)` 规范复核 0 不符、psum 尾块有效 M 16、multicast 对端
  一致性、`get_global_idx` 七形态偏移全对。SM100 分支仅编译期覆盖（`sm_100a` 编过，本机不可运行）。
