# Mega Gate / Mega mHC 的跨 CTA 分段屏障调度器与工作区布局

> 目标源码：`~/github/DeepGEMM@057ca59`（本机核对）。参考仓库：DeepGEMM
> <https://github.com/deepseek-ai/DeepGEMM>。
> 本篇把两个从没被单独读过的调度器文件拆透——`deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh`
> （41 行，`ScoreBarrier`）与 `scheduler/mega_mhc.cuh`（55 行，`SplitBarrier` / `Norm` / `Mix`）——
> 以及它们落脚的两个 `Workspace` 布局。它们只管**跨 CTA / 跨 split 的「到齐」同步**，用的是
> `%gridid` + 一个 64 位整数状态，**不碰 UMMA / TMA / WGMMA**；唯一 SM100 专属的是一行 `red.async`。
> 故在 H100(sm90) 上**直接 include 真实头文件、编译运行真实设备代码**
> （`src/mega_gate_mhc_split_barrier.cu` + `harness/exp_mega_gate_mhc_split_barrier.{sh,out.txt}`）。

---

## 0. 为什么补这一篇

前面有几篇擦边碰过这两个东西，但都只当「黑盒同步点」引用：

- **第 5 篇（P5-1，Mega Gate）**：在 `epilogue:39-78` 讲 TMEM 打分回读时提了一句「split-K 时先落 global
  score scratch，再用 `ScoreBarrier` 跨 CTA 汇聚（`scheduler:13-39`）」。
- **第 5 篇下篇（P5-2，Mega mHC）**：在讲跨 CTA 时提了一句「`SplitBarrier` 用『grid 序号+到达数』一次
  编码、免复位（`scheduler:13-49`，Norm/Mix 两个）」。
- **第 87 篇（P77-1）**：拆了 `comm/barrier.cuh` 的 `grid_sync` / `nvlink_barrier`——那对是**同步原语**
  （搭配 `SymBuffer` 对称指针跨 rank），跟本篇这两个**同一 kernel 内、按 token-block / m-block 分段**的
  屏障不是一回事。

这两个文件（共 96 行）是 Mega Gate 与 Mega mHC 在 **split-K 拆分后如何把同一个输出块的多个 CTA 的
部分结果汇总**的唯一落点。本篇把它们的**状态编码、免复位原理、`wait_init` 的无符号距离语义、
工作区几何、以及四个角色的调用链**讲清，并在真机上验证状态机。

---

## 1. 两个屏障：把一个 (grid, 到达数) 编码进一个 u64

两个类形状几乎一样，都是「静态模板 + 四个静态方法 + 一个全局 `uint64_t` 状态槽」：

```
scheduler/mega_gate.cuh  ScoreBarrier<kNumLogicalCtas>     （每个 token block 一个槽）
scheduler/mega_mhc.cuh   SplitBarrier<kBarrierIndex>       （Norm=0 / Mix=1，各按 m_block 分槽）
```

**状态编码**（`ScoreBarrier::encode_state:17-19`、`SplitBarrier::encode_state:20-22`）：

```cpp
// gate：每个逻辑 CTA 到一个就 +1
encode_state(g, n) = (g + 1) * kNumMaxLogicalCtas + n;              // kNumMaxLogicalCtas = 64
// mHC：每个 split 到一个就 +1
encode_state(g, n) = (g + 1) * kNumSplitBarriers * kNumMaxSplits + n; // 2 * 64 = 128
```

即 `状态 = (grid 标签) * (每组最大到达数) + (当前到达数)`。**代价**就是乘一个「够大的 stride」：
gate 用 `kNumMaxLogicalCtas`、mHC 用 `kNumSplitBarriers * kNumMaxSplits`，只要它 **严格大于** 本轮可能
到达的最大数，`n` 的进位就永远进不到标签字段（见 §2.4）。

**四个方法**（两类的行号一一对应）：

| 语义 | `ScoreBarrier` | `SplitBarrier` | 实现 |
|---|---|---|---|
| 重新 arm | `init:21-23` | `init:24-26` | `st_rel(ptr, encode_state(get_grid_idx()))` |
| 等 arm 好 | `wait_init:25-29` | `wait_init:28-34` | 自旋 `while (ld_acq_gpu(ptr) - base >= C)` |
| 到达一次 | `arrive:31-33` | `arrive:36-41` | `red_async_inc_rel(ptr)`（状态 +1） |
| 等全到齐 | `wait:35-38` | `wait:43-48` | 自旋 `while (ld_acq_gpu(ptr) != base + C)` |

底层三个 PTX 包装都在 `ptx/ld_st.cuh`：`st_rel:308-310`（`st.release.gpu.global.u64`）、
`red_async_inc_rel:340-342`（`red.async.release.gpu.global.add.u64`）、`ld_acq_gpu:360-364`
（`ld.acquire.gpu.global.u64`）；`%gridid` 取自 `ptx/utils.cuh:23-27` 的 `get_grid_idx`。

---

## 2. 四个「为什么」

### 2.1 为什么用 `%gridid` 当标签：免复位

`%gridid` 是**每次 launch 自增**的硬件网格标识（`mov.u64 %0, %%gridid`）。于是每发起一次 kernel，
`(g+1)*stride` 就是一个**全新的基数**：上一次 launch 留在状态槽里的终值（`g_prev*stride + C`）天然
落在本轮基数**之下**，不需要在每次 launch 前清零缓冲。这正是 host 侧 `torch.zeros` **只在每 stream
第一次分配时**做一次的原因（见 §5）。

对照第 87 篇的 `grid_sync`：那里用「每轮增量之和恰好 2³¹、只翻最高位、两轮回原值」实现免复位；
这里用「`%gridid` 每 launch +1、标签每轮全新」实现免复位。**两种思路都是「让状态在新一轮里天然不同，
从而无需显式清零」**。

### 2.2 `wait_init` 的无符号距离：陈旧拒绝 vs 本轮通过

`wait_init` 是整个机制里唯一不直观的一行（`ScoreBarrier:25-29`）：

```cpp
while (ptx::ld_acq_gpu(state_ptr) - current_grid_barrier_base >= kNumLogicalCtas);
// 其中 base = (g+1)*stride，循环条件是「距离 >= C」
```

它不等一个**具体值**，而是等**无符号距离 `< C`**。记当前读到 `x`、本轮基数 `base`：

- **陈旧态**（还没被本轮 `init` 覆盖）：`x = g*stride + C_prev`（上一轮终值），`x - base =
  C_prev - stride < 0`，按无符号回绕成极大值 → 条件恒真 → **继续等待**（正确拒绝）。
- **本轮已 arm**（`init` 刚写过）：`x = base`，距离 `0 < C` → **立即通过**。
- **本轮已有部分到达**：`x = base + k`（`0 ≤ k < C`），距离 `k < C` → **立即通过**（不误伤并发到达）。
- **本轮全到齐**：`x = base + C`，距离 `C ≥ C` → 继续等待（不过 `wait_init` 只在到达前调用一次，
  此时 `n` 必为 0，故不会踩到）。

也就是说：`wait_init` 的语义是「**等到这个槽被打上本轮标签为止**」，用同一套无符号算术同时拒绝了
「上一轮的残留」和「本轮还没 arm」两种情形。`ScoreBarrier` 与 `SplitBarrier` 的注释都写着同一句话
（`mega_gate.cuh:27` / `mega_mhc.cuh:32`）：*unsigned distance rejects stale grid tags but accepts
any current-grid arrival count*。

§6 的 Part B 用一个「延迟 init」实验把三种情形分别钉死：让 block0 跑真实 `wait_init`、block1 自旋
若干周期后（仅陈旧态时）调用 `init`，block0 返回后读「init 是否已发生」的旗标——陈旧态返回时旗标
必须为 1（说明它一直等到 init），本轮已 arm / 部分到达则返回时旗标为 0（说明它没有等 init）。

### 2.3 `arrive` 的 `red.async` 是 SM100 专属（一行）

`arrive` 调 `red_async_inc_rel` → `red.async.release.gpu.global.add.u64`。**这是唯一 SM100 专属的一行**：
ptxas 在 `-arch=sm_90a` 直接报

```
error: Feature 'st.async/red.async with .global state space' requires .target sm_100 or higher
error: Illegal modifier '.release' for instruction 'red.async'
```

（`red.async` 是异步 reduction，SM100 才有。）因此本机 H100 **跑不了真实 `arrive`**。§6 的 S 组用同一段
`ScoreBarrier<4>::arrive` 分别编 `sm_90a`（失败）与 `sm_100a`（成功）作为硬证据；而状态机的其余部分
（`encode_state` / `init` / `wait_init` / `wait` / `get_ptr`）全是 sm90 合法的，故真机实验里用
**同语义、非 async** 的 `red.release.gpu.global.add.u64` 作为 `arrive` 替身（源码注释已标出），
真实语义与真实设备代码一致。

### 2.4 状态位的「进位隔离」：stride 必须 > 最大到达数

`gate` 的 `ScoreBarrier` 在 `:15` 有一条静态断言 `kNumLogicalCtas < kNumMaxLogicalCtas`——这就是
「stride 必须大于最大到达数」的可执行形式。若 `kNumLogicalCtas == kNumMaxLogicalCtas`，最后一到达
会把 `n` 进位到标签字段，`wait` 的等值判断就会错位。mHC 侧的 `SplitBarrier` 没有这条断言，但它用
`kNumSplitBarriers * kNumMaxSplits`（=128）当 stride，而到达数最多 `kNumSplits ≤ kNumMaxSplits`
（`Workspace` 的 `DG_STATIC_ASSERT(kNumSplits <= kNumMaxSplits)`，`layout/mega_mhc.cuh:86`），
且 `kNumSplitBarriers = 2 ≥ 1`，故同样隔离。真机实测（§6）：终止态的 `term % stride` 恰好等于到达数。

---

## 3. 两个 barrier 的结构对照

| | Mega Gate `ScoreBarrier` | Mega mHC `SplitBarrier` |
|---|---|---|
| 槽的索引 | 每个 **token block** 一个 | 每个 **(BarrierIndex, m_block)** 一个 |
| 取址 | `Workspace::get_score_barrier_ptr` | `SplitBarrier::get_ptr`（模板 `kBarrierIndex`） |
| line 字节 | `kScoreBarrierLineBytes = 128`（`layout/mega_gate.cuh:22`） | `kSplitBarrierLineBytes = 128`（`layout/mega_mhc.cuh:39`） |
| 每组最大到达数 | `kNumMaxLogicalCtas = 64`（`:23`） | `kNumMaxSplits = 64`（`:37`）；stride 再乘 `kNumSplitBarriers = 2`（`:38`） |
| 两个实例 | 无（只有一个） | `Norm = SplitBarrier<0>` / `Mix = SplitBarrier<1>`（`:52-53`） |
| 状态数组规模 | `kNumMaxTokenBlocks = 65536` → **8 MiB** | `2 × kNumMaxMBlocks(16384) × 128` → **4 MiB** |

`SplitBarrier::get_ptr:15-18` 是关键一行：

```cpp
line_idx = kBarrierIndex * kNumMaxMBlocks + m_block_idx;
return advance_ptr<uint64_t>(gmem_split_barriers, line_idx * kSplitBarrierLineBytes);
```

即 `Norm` 与 `Mix` 各自占据 `[kBarrierIndex*kNumMaxMBlocks, (kBarrierIndex+1)*kNumMaxMBlocks)` 段，**互不
重叠**——这就是为什么同一个 `gmem_split_barriers` 缓冲能同时装两个屏障而不打架。真机 Part C1 验证了
`off0 = mb*128`、`off1 = (kNumMaxMBlocks + mb)*128`。

### 为什么 line 要 128 字节？

两个 `Workspace` 的 barrier 槽都按 **128 字节（L2 cache line）**对齐排布。屏障状态是一个 8 字节整数，
独占一整条 line 是为了**避免相邻 token block / m block 的屏障槽落在同一条 cache line 上**——否则两个
不同块的 `red` 会互相造成 L2 line 的写竞争（false sharing）。本实验 Part C1 实测 line = 128、槽
间距 = 128。

---

## 4. 工作区布局（两个 `Workspace`）

两个 barrier 都住在各自的全局工作区里，工作区的**其余部分**承载 split-K 的部分结果。host 侧先
按 `Workspace::get_num_*_bytes` 申请一段 scratch + 一个 barrier 张量，再把指针经 JIT 包装层送进 kernel。

### 4.1 Mega Gate `layout/mega_gate.cuh:78-108`

```cpp
// score scratch： [num_token_blocks * block_tokens, kNumSplits, kNumExperts]   （float）
get_num_scratch_bytes(tb, splits, bt, e) = tb * splits * bt * e * sizeof(float);
get_score_ptr(token_idx, split=0) = base + token_idx * (kNumSplits*kNumExperts) + split*kNumExperts;
get_score_barrier_ptr(token_block_idx) = barriers + token_block_idx * 128;   // kScoreBarrierLineBytes
```

### 4.2 Mega mHC `layout/mega_mhc.cuh:80-120`

```cpp
// 三块连续部分和：gemm_partials [mb, splits, BLOCK_M, N]，norm_sqr / x1_sqr [mb, splits, BLOCK_M]
get_num_m_blocks(T)               = ceil_div(T, BLOCK_M);              // BLOCK_M = 64
get_num_scratch_bytes(mb, splits) = mb*splits*(BLOCK_M*N + 2*BLOCK_M)*sizeof(float);
get_gemm_partial_ptr(task)        = base + task*(BLOCK_M*N);
get_hc_norm_sqr_sum_partial_ptr(task) = base + total + task*BLOCK_M;   // total = mb*splits
get_x1_sqr_sum_partial_ptr(task)      = base + 2*total + task*BLOCK_M;
```

`N = kNumHCOutputs = 4*(4+2) = 24`（`layout/mega_mhc.cuh:11-12`），所以每个 task 的 gemm partial 是
`64*24 = 1536` 个 float，外加两个 `64` float 的平方和。§6 Part D 逐项用真实静态函数验证了这些偏移。

### 4.3 host 侧

- **Mega Gate**：`csrc/apis/mega_gate.hpp:13-25` 的 `get_score_barriers` 按 **CUDAStream** 缓存一个
  `torch.zeros({kNumMaxTokenBlocks, 128}, kByte)`；`:20` 有一条 `DG_HOST_ASSERT(...CaptureStatus::None)`
  ——**CUDA Graph capture 之前必须先 warm-up**，否则「一次性清零」会被录进 graph 反复回放。
- **Mega mHC**：`csrc/apis/mega_mhc.hpp:26-38` 的 `get_split_barriers` 同理，形状
  `{kNumSplitBarriers, kNumMaxMBlocks, kSplitBarrierLineBytes}`，`:33` 同样的 capture 断言。
- 经 JIT 包装层送进 kernel：`csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp:23/84`、
  `sm100_mega_mhc.hpp:29/146`（`reinterpret_cast<uint64_t*>(split_barriers.data_ptr())`）。

---

## 5. 调用链：谁 init / 谁 wait_init / 谁 arrive / 谁 wait

### 5.1 Mega Gate（`sm100_bf16_mega_gate.cuh`）

任务分解 `kNumLogicalCtas = kNumMmaCtas * kNumExpertGroups * kNumSplitK`（`:55`），一个**worker group**
的 `kNumLogicalCtas` 个 CTA 一起流水遍历所有 token block（`:127-128`）。对每个 token block
（`:268-302`）：

```cpp
if (gate_warp_idx == ln - 1 and kNumLogicalCtas > 1 and elect_one) {   // :271-276
    if (logical_cta_rank == 0) ScoreBarrier::init(ptr);   // 本组秩 0 重新 arm
    else                       ScoreBarrier::wait_init(ptr); // 其余等 arm
}
... 各 CTA 把本片的 score 写进 scratch ...
if (... ) ScoreBarrier::arrive(ptr);   // :299  每片写完 +1
if (gate_warp_idx == 0 ... ) ScoreBarrier::wait(ptr);   // :302  等全组到齐再读全部 score
```

即「**ranks 0 init、其余 wait_init；每片 arrive；队首 wait**」四步，正是 §1 的协议。`logical_cta_rank
= slice_idx*kNumMmaCtas + mma_cta_rank`（`:75`）。

### 5.2 Mega mHC（`sm100_mega_mhc.cuh` + `epilogue/sm100_mega_mhc.cuh`）

mHC 用「**owner init + 全体任务 wait_init**」的分布式版本。某个 warp 先按
`m_block_idx = blockIdx.x; m_block_idx < num_m_blocks; += kNumSMs` 给**自己负责的那段** m block
`Mix::init`（`:173`）/ `Norm::init`（`:178`）；随后各 SM 按 `task_idx = blockIdx.x; task_idx < num_tasks;
+= kNumSMs` 领 `(m_block, split)` 任务，每个任务 `Norm::wait_init<kNumSplits>`（`:323`）/
`Mix::wait_init<kNumSplits>`（`:583`），算完后 `Norm::arrive<kNumSplits>`（`:379`）/
`Mix::arrive<kNumSplits>`（`:653`）。最终在 epilogue 里 `Mix::wait<kNumSplits>`（`epilogue:22`,
`run_mix_task`）/ `Norm::wait<kNumSplits>`（`epilogue:306`, `run_shifted_norm_task`）等全到齐，再读
`workspace` 里的部分结果做归约。

**注意 init 与 arrive 落在不同 SM**：init 由 owner SM 做，wait_init 由处理该 m_block 任务的任意 SM 做
——这正是 §2.2 那套无符号距离要处理的竞态。

### 5.3 deterministic 模式的差异

- **Mega Gate**：`heuristics/mega_gate.hpp:110-111` 的注释写着「deterministic mode is also batch
  invariant, so K is never split」，`:126` 于是 `num_split_k ∈ {1}` → `kNumLogicalCtas` 退化为
  `kNumMmaCtas * kNumExpertGroups`，且当它等于 1 时 `:271/:298/:301` 的 `kNumLogicalCtas > 1` 分支整体
  跳过——**屏障路径在单 CTA 下自动消失**（快路）。
- **Mega mHC**：`csrc/apis/mega_mhc.hpp:15-17` 的 deterministic 是把 `num_splits` **钉成默认值 16**
  （为解决 batch invariance），**不是关掉 split**，故 `SplitBarrier` 在确定性模式下依旧生效。

---

## 6. H100(sm90) 真实设备代码实测

`src/mega_gate_mhc_split_barrier.cu` 直接 `#include` 四个真实头文件，`harness/
exp_mega_gate_mhc_split_barrier.sh` 在 kernel_lab(H100/sm90) 上 `nvcc -arch=sm_90` 编译运行。真实头
文件 sha256：`scheduler/mega_gate.cuh = 4aef186a…`、`scheduler/mega_mhc.cuh = bf774cfa…`、
`layout/mega_gate.cuh = 543ca6ee…`、`layout/mega_mhc.cuh = ca34571a…`、`ptx/utils.cuh = 8a9d5e03…`、
`ptx/ld_st.cuh = 912e9f9a…`。

**Part A —— `ScoreBarrier` 全协议 + 免复位**：`C` 个逻辑 CTA 一个 token block，每个 block 内
「rank0 init / 其余 wait_init → 逐个 arrive → wait」，**连续两次 launch 中间不重置缓冲**：

| 用例 | mismatch | 终止态到达数字段 `term % 64` |
|---|---|---|
| C=4（首次清零） | 0 | 4 |
| C=8（不清零，2 次 launch） | 0 | 8 |
| C=1（不清零，2 次 launch） | 0 | 1 |

三次终止态都严格等于 `encode_state(grid_idx, C)`，且 `term % kNumMaxLogicalCtas == C`——**免复位与
进位隔离同时坐实**。

**Part B —— `wait_init` 的无符号距离**（block0 返回后 `init_seen_before_return`）：

| 用例（预置状态） | init_seen_before_return | 期望 | 判定 |
|---|---|---|---|
| 陈旧上一轮终止态 `g*64 + C` | 1（一直等到 init） | 1 | PASS |
| 本轮已 arm `(g+1)*64` | 0（立即通过） | 0 | PASS |
| 本轮已有 1 个到达 `(g+1)*64 + 1` | 0（立即通过） | 0 | PASS |

三点把 §2.2 的表格逐行验证：**陈旧被拒、arm 通过、部分到达不误伤**。

**Part C —— `SplitBarrier`**：`get_ptr` 偏移 `off0 = mb*128`、`off1 = (16384+mb)*128`（line=128）；
`Norm(idx0)` / `Mix(idx1)` × `splits∈{4,16}` × 8 SM × 5 m block，连续两次 launch 不清零，**四组
mismatch=0、终止态到达数 = splits**。

**Part D —— `Workspace` host 算术**：gate 的 `get_num_scratch_bytes`、`get_score_ptr` 偏移；mHC 的
`get_num_m_blocks`、`get_num_scratch_bytes`、三个 partial 指针偏移；以及常量表
（gate `kScoreBarrierLineBytes=128 / kNumMaxLogicalCtas=64 / kNumMaxTokenBlocks=65536`；mHC
`BLOCK_M=64 / kNumSplitBarriers=2 / kNumMaxSplits=64 / kNumMaxMBlocks=16384`）——**7/7 PASS**。

**Part S —— `arrive` 的架构闸门**：同一段 `ScoreBarrier<4>::arrive` 编 `sm_90a` **失败**
（`requires .target sm_100 or higher`）、编 `sm_100a` **成功**。

**总判定 31 PASS / 0 FAIL**（Part A/B/C/D 29 项 + S 组 2 项）。

```
== Part A: ScoreBarrier 全协议（免复位）==
A1 ScoreBarrier C=4 x2 launches (no reset)     PASS
A2 ScoreBarrier C=8 x2 launches (no reset)     PASS
A3 ScoreBarrier C=1 x2 launches (no reset)     PASS
== Part B: ScoreBarrier::wait_init 无符号距离 ==
B1 stale terminal rejected (wait until init)   PASS
B2 current-grid armed accepted immediately     PASS
B3 current-grid partial arrival accepted       PASS
== Part C: SplitBarrier get_ptr 偏移 + 全协议 ==
C1 SplitBarrier get_ptr line offsets           PASS
C2 Norm(idx0) splits=4 x2 (no reset)           PASS
C3 Mix (idx1) splits=4 x2 (no reset)           PASS
C4 Norm(idx0) splits=16 x2 (no reset)          PASS
C5 Mix (idx1) splits=16 x2 (no reset)          PASS
...
PASS=31 FAIL=0
总判定: ALL PASS
```

---

## 7. 边界与链接

- **H100 不可跑真实 `arrive`**：`red.async` 是 SM100 专属（§2.3）；真机用同语义同步 `red` 替身，
  其余全是真实头文件代码。**SM100 上真实 arrive 的正确性只能线上/源码层面判断**。
- **`%gridid` 递增性**：免复位依赖 `%gridid` 每次 launch 自增（本机两次背靠背 launch 实测终止态不同，
  即标签前进）。若硬件/驱动对同 context 连续 launch 复用 grid id 的极端情形，理论上前一轮终值可能被
  误当本轮标签——但 `wait_init` 的无符号距离与 `init` 的主动写入共同兜底。
- **相关篇章**：Mega Gate 实现第 5 篇、Mega mHC 实现第 5 篇下篇、Mega Gate/mHC 的测试 oracle 第 47/57 篇、
  Mega Gate/mHC 的 JIT 包装层第 86/89 篇、跨 SM/rank 同步原语（`grid_sync`/`nvlink_barrier`）第 87 篇。
- **参考链接**：DeepGEMM 仓库 <https://github.com/deepseek-ai/DeepGEMM>（本机核对 commit `057ca59`）；
  相关文件 `deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh` / `scheduler/mega_mhc.cuh` /
  `layout/mega_gate.cuh` / `layout/mega_mhc.cuh` / `ptx/{utils,ld_st}.cuh` /
  `impls/sm100_bf16_mega_gate.cuh` / `impls/sm100_mega_mhc.cuh` / `epilogue/sm100_mega_mhc.cuh` /
  `csrc/apis/{mega_gate,mega_mhc}.hpp`、`csrc/jit_kernels/impls/{sm100_bf16_mega_gate,sm100_mega_mhc}.hpp`、
  `csrc/jit_kernels/heuristics/mega_gate.hpp`。
