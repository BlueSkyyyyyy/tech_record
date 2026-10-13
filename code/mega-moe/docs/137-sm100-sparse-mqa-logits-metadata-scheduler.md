# SM100 sparse MQA logits 的 metadata kernel 逐段精读：两 token 归并去重、双 Q 槽打包与负载均衡调度

> 本文对应路线图 **P127-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 137 篇。
>
> 第 22 篇（P12-1）在拆稀疏 MQA logits 时，对 metadata kernel
> `scheduler/sm100_sparse_mqa_logits_metadata.cuh`（475 行）只给了 §3 约 30 行的**概览**（「有效块前缀 /
> 两 token 归并去重 / 物理块三态映射 / 末 CTA 建调度」）；第 136 篇（P126-1）把同一算子的**主 kernel**
> 做了完整逐段精读 + H100 真机验证。本篇把 metadata kernel 补到同样的深度：逐段读 475 行，并把其中
> **架构无关的全部逻辑**在 H100(sm90) 上跑**真实设备代码**。
>
> 与前面几篇不同，这里的结论更强：**整个 metadata kernel 都不是 SM100 专属**——它只用
> `ptx::` 的整数/原子原语、`cp.async`（sm80）、`math::cta_exclusive_sum`、`cute::` 下标工具，
> **既不碰 UMMA/tcgen05/UTCCP，也不碰 TMA**（这些都在主 kernel 里）。因此本篇直接把真实头文件
> include 进来、用真实 launch 参数**原样启动这个 kernel**，再把输出 metadata 与一份**独立重写的 host
> oracle**（有效块前缀 → 两 token 归并去重 → 槽打包 → 物理块映射 → split 头 → 调度覆盖）逐字段对拍。
>
> 源码坐标（`~/github/DeepGEMM`，本地 commit `057ca59`、Public release 26/09/30）：
> - metadata kernel：`deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh`（475 行）
> - 布局与工作区：`deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh`（156 行）
> - host 入口：`csrc/apis/attention.hpp:306-367`、`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:135-200`
> - 实测：`src/sparse_mqa_logits_metadata.cu` +
>   `harness/exp_sparse_mqa_logits_metadata.{sh,out.txt}`（H100，**ALL PASS**）
>
> 全文用「目标卡」指代目标 AI 卡。

## 1. 定位：主 kernel 前的「调度与索引表」

稀疏 MQA logits 是**两段式**：先跑一个轻量 metadata kernel 生成「调度表 + 索引表」，再跑 persistent
主 kernel（第 136 篇）。metadata kernel 干四件事：

1. **有效块前缀**：由每个 query 的 KV 长度推出它这一行 `sparse_kv_block_indices` 里前多少个块有效；
2. **两 token 归并去重**：`kBlockQ=2`（`layout:17`）——相邻两个 query token 一个块，把两条升序块号
   列表**归并成并集**；共享的块只加载/MMA 一次，两个 token 各取各的行（第 22 篇 §3.2）；
3. **压缩列映射**：给并集里每个块记录「它在 q0 / q1 各自列表里的槽号」，打包成 16+16 位；
4. **负载均衡调度**：把所有 KV split 切给 SM、按工作量排序 + 旋转。

它是主 kernel 的**硬契约**：主 kernel 直接按字节 reinterpret 这份 metadata（`impls:211-212`），并断言
`header->num_sms == kNumSMs`、`header->use_unaligned_ks == kUseUnalignedKs`（`impls:269-271`）——
**metadata 必须用与主 kernel 相同的 SM 数生成，否则重跑**。

## 2. 布局与工作区（`layout/sparse_mqa_logits.cuh`）

### 2.1 三个输出结构

```cpp
// layout:23-29（MetadataHeader，16B）——metadata 缓冲区开头
struct alignas(16) MetadataHeader {
    uint32_t num_kv_splits;   // 本 launch 产生的 KV split 总数
    uint32_t num_waves;       // 调度波数
    uint32_t use_unaligned_ks;
    uint32_t num_sms;         // 生成时的 SM 数（主 kernel 校验一致）
};

// layout:31-61（KVSplitHeader，16B）——每个 split 的头
static constexpr uint32_t kContiguousFlag  = 0x80000000u;  // 整个 split 物理连续（可 TMA）
static constexpr uint32_t kPartialTailFlag = 0x40000000u;  // 尾块越过 KV 末尾（仅连续版）
uint32_t q_token_base;
uint32_t packed_num_kv_blocks;  // 高 2 位标记 + 低位块数
uint32_t q0_slot_base, q1_slot_base;

// layout:63-72（KVBlockInfo，8B）——每个 split 内的块
uint32_t physical_kv_block_idx;
uint32_t packed_slot_offsets;   // q0 槽偏移 | (q1 槽偏移 << 16)
```

```cpp
// layout:74-84（ScheduleEntry，16B）——每 wave × 每 SM 一条
uint32_t kv_split_begin, kv_split_end, q_token_base, num_q_tokens;
```

`KVSplitHeader` 把「块数」和两个调度标记塞进一个 `uint32`（高 2 位），用
`get_num_kv_blocks = p & ~kFlagMask` 还原（`layout:50-60`）。`KVBlockInfo` 用
`kNumSparseSlotBits=16`（`layout:11`）分别存 q0/q1 的「split 内槽偏移」，`kInvalidSparseSlot=0xFFFF`
（`layout:12`）表示该 token 不含此块。为什么是**偏移**而不是绝对槽号：split 头只存一次
`q0_slot_base`/`q1_slot_base`，块内每个槽只需 16 位相对偏移（见 §5.5）。

### 2.2 工作区：三个计数器各占一条 L2 line

```cpp
// layout:141-146
struct alignas(128) WorkspaceState {
    uint32_t num_kv_splits;              // 全局 split 分配游标（atomicAdd）
    alignas(128) uint32_t next_q_offset; // 分页版动态取 Q 块的游标
    alignas(128) uint32_t num_finished_ctas;
};
```

三个 `alignas(128)` 让每个计数器独占一条 L2 cache line，**避免多 CTA 并发原子更新的伪共享**。
`WorkspaceState` 之后紧接着是 `QBlockInfo q_block_infos[num_q_tokens]`（`kernel:194`）：

```cpp
// layout:148-154（QBlockInfo，8B）
uint32_t kv_split_base;   // 该 Q 块第一个 split 的全局下标
uint32_t num_kv_splits;   // 该 Q 块产生多少 split
```

### 2.3 设备端共享内存（`scheduler:14-35`）

```cpp
template <uint32_t BLOCK_Q, uint32_t NUM_MAX_SPARSE_BLOCKS,
          uint32_t kNumKVBlocksPerSplit, uint32_t kNumKVSplitsPerEntry, uint32_t kNumThreads>
struct SharedStorage {
    static constexpr uint32_t kNumMaxMergedKVBlocks = BLOCK_Q * NUM_MAX_SPARSE_BLOCKS;
    static constexpr uint32_t kNumMaxKVSplits = ceil_div(kNumMaxMergedKVBlocks, kNumKVBlocksPerSplit);
    uint32_t logical_kv_block_indices[BLOCK_Q][NUM_MAX_SPARSE_BLOCKS];
    uint32_t packed_slots_by_merged_kv_block[kNumMaxMergedKVBlocks];
    uint32_t packed_slot_bases_by_kv_split[kNumMaxKVSplits];
    uint32_t warp_sums[kNumWarps];
    uint32_t num_waves;
    uint32_t kv_split_histograms[kNumWarps][kNumKVSplitsPerEntry + 1];
    struct alignas(16) { ... } q_block;   // 当前 Q 块的临时区
    uint32_t is_last_cta;
};
```

关键量：`kNumMaxMergedKVBlocks = 2·NB`、`kNumKVBlocksPerSplit = SPLIT_KV/SB`（`kernel:176`）。
`logical_kv_block_indices` 缓存**当前 Q 块**两个 token 的块号（§5.3）；`packed_slots_by_merged_kv_block`
缓存**当前 Q 块**归并后的打包槽（§5.5/§5.6）；`packed_slot_bases_by_kv_split` 分页版用；
`kv_split_histograms` 给负载均衡排序用（§6.3）。host 侧 smem 大小直接
`align(sizeof(SharedStorage),128)`，并在 JIT 源码里 `static_assert` 钉死（`hpp:175-177`）。

## 3. host 入口与 launch 参数

`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:135-200`：

- `kNumMetadataThreads = 256`、`kNumKVSplitsPerEntry = 8`（`:150-151`）；单 warp 数固定 8。
- `num_ctas = min(paged ? T : ceil(T/2), num_sms*4)`（`:156`）——一个 CTA 处理一个（或每轮取一个）Q 块，
  最多 `4×num_sms`。
- smem 尺寸手算（`:159-165`）：三块 `uint32` 数组 + `kNumWarps` warp_sums + `(1 + kNumWarps·(k+1))`
  直方图 + `q_block` 结构，最后 `align(...,128)`。
- 模板实参（`:178-186`）：`<kIsPaged, kUseUnalignedKs, BLOCK_Q=2, SPLIT_KV, SB, NB, PAGE_KV,
  kNumKVSplitsPerEntry, num_sms, kNumThreads>`；`num_sms` 来自 `runtime->get_num_sms()`。
- metadata 缓冲区尺寸公式（`:20-36`）：`16 + num_max_kv_splits·split_bytes + align(num_max_kv_splits,num_sms)·16`，
  其中连续版 `num_max_kv_splits = ceil(T/2)·ceil(2·NB/KPS)`、分页版 `= T·ceil(NB/KPS)`。

四个 host API（`attention.hpp:306/337/370/406`）一律断言 `arch_major==10`（`:313/344/380/414`），
所以**公开 API 在 H100 上会被拒**；但 kernel 本身架构无关，本实验直接绕过 API 调用它。

## 4. 模板契约（`kernel:180-188`）

```cpp
DG_STATIC_ASSERT(BLOCK_Q == 2 and kNumThreads == 256 and kNumSMs <= kNumThreads, ...);
DG_STATIC_ASSERT(SPLIT_KV % SPARSE_BLOCK_KV == 0 and SPLIT_KV % 128 == 0, ...);
DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS % 4 == 0, ...);          // 行 16B 对齐
DG_STATIC_ASSERT(NUM_MAX_SPARSE_BLOCKS <= 4096, ...);
DG_STATIC_ASSERT(kNumKVBlocksPerSplit <= kInvalidSparseSlot, ...);
DG_STATIC_ASSERT(kNumKVSplitsPerEntry > 0, ...);
DG_STATIC_ASSERT(not kIsPaged or PAGE_KV % SPARSE_BLOCK_KV == 0, ...);   // 块不跨页
DG_STATIC_ASSERT(not kIsPaged or not kUseUnalignedKs, ...);              // 分页不用 ks
```

`BLOCK_Q==2` 是「双 Q 槽打包」的硬前提（每个 16 位半区一个 token）；`NUM_MAX_SPARSE_BLOCKS%4==0`
让行按 16B 对齐（对齐版用 `cp.async` 一次搬 4 个 `uint32`，§5.3）。`kernel:191` 第一件事是
`cudaGridDependencySynchronize()`（PDL）——等上游（如主 kernel 上一拍或调用方）写完输入。

## 5. 主循环逐段

`kernel:202` 的 `q_token_idx = blockIdx.x * (kIsPaged ? 1 : BLOCK_Q)`，外层是一个 `while(true)`。
`:204-236` 的 `thread_idx==0` 段在 smem 里备好当前 Q 块的「待处理状态」，`:237` 之后全体读取。

### 5.1 Q 块构造与有效块前缀（`:204-236`）

```cpp
// :206-235（thread 0）
while (q_token_idx < num_q_tokens) {
    uint32_t num_q_block_tokens = min(BLOCK_Q, num_q_tokens - q_token_idx);
    if constexpr (kIsPaged) {
        const uint32_t request_idx = indices[q_token_idx];
        uint32_t request_q_token_base = q_token_idx;
        while (request_q_token_base > 0 and indices[request_q_token_base-1] == request_idx)
            --request_q_token_base;                              // 线性扫描回 request 头
        if ((q_token_idx - request_q_token_base) % BLOCK_Q != 0) {   // 只从偶数偏移起配对
            q_token_idx = gridDim.x + ptx::atomic_add(&workspace_state->next_q_offset, 1u);
            continue;
        }
        num_q_block_tokens = q_token_idx+1 < num_q_tokens and indices[q_token_idx+1] == request_idx ? BLOCK_Q : 1;
    }
    const auto get_num_kv_blocks = [&](const uint32_t q_idx) {
        const uint32_t kv_begin = kIsPaged ? 0 : min(cu_seq_len_k_start[q_idx], num_kv_tokens);
        const uint32_t kv_end = kIsPaged ? context_lens[q_idx]
                                         : max(kv_begin, min(cu_seq_len_k_end[q_idx], num_kv_tokens));
        if constexpr (not kIsPaged and not kUseUnalignedKs)
            DG_DEVICE_ASSERT(kv_end == kv_begin or kv_begin % SPARSE_BLOCK_KV == 0);  // 对齐前提
        return min(NUM_MAX_SPARSE_BLOCKS, ceil_div(kv_end - kv_begin, SPARSE_BLOCK_KV));
    };
    smem.q_block.q_token_base = q_token_idx;
    smem.q_block.num_q_tokens = num_q_block_tokens;
    smem.q_block.num_kv_blocks[0] = get_num_kv_blocks(q_token_idx);
    smem.q_block.num_kv_blocks[1] = num_q_block_tokens == BLOCK_Q ? get_num_kv_blocks(q_token_idx+1) : 0;
    break;
}
```

三个「为什么」：

1. **分页版的「同 request 内配对」与「偶数偏移」**：分页版 Q 块不能跨 request（block_table 按
   request 组织），且要求成对 token 的 block-table 行一致（host 注释 `attention.hpp:334-336`）。
   所以先线性扫回 request 头，只从 `(offset % BLOCK_Q == 0)` 的位置起配对；否则把该 Q 块**扔回全局游标**
   `next_q_offset`（`:216`），由其他 CTA 在别的轮次取走。这里用 `atomic_add` 让多个 CTA **无锁地动态
   认领** Q 块——这是它不用静态划分的原因：request 长度不均。
2. **有效块前缀**：连续版按 `[kv_begin, kv_end)` 的区间长度算块数，并 clamp 到 `NB`；非对齐版允许
   `kv_begin` 不对齐（`kv_end-kv_begin` 仍整块计），对齐版则断言非空段必须 `kv_begin%SB==0`——这正是
   host 注释 `attention.hpp:303-305` 的「有效块前缀必须落在对应 KV 范围内、严格升序」。
3. **两个 token 的块数分别算**（`num_kv_blocks[0/1]`），因为相邻 query 的 KV 长度可以不同（最后一个
   只有 `num_q_tokens=1`）。

### 5.2 稀疏索引加载（`:246-271`）

```cpp
// :248-266
for (uint32_t q_token_offset = 0; q_token_offset < BLOCK_Q; ++q_token_offset) {
    const auto src = sparse_kv_block_indices + (uint64_t)(q_token_base + q_token_offset) * NUM_MAX_SPARSE_BLOCKS;
    const auto dst = smem.logical_kv_block_indices[q_token_offset];
    const uint32_t num_kv_blocks = q_token_offset == 0 ? num_kv_blocks_in_q0 : num_kv_blocks_in_q1;
    if constexpr (kUseUnalignedKs) {
        if (num_kv_blocks != 0) {
            const uint32_t kv_offset = cu_seq_len_k_start[q_token_base + q_token_offset] % SPARSE_BLOCK_KV;
            for (uint32_t i = thread_idx*4; i < num_kv_blocks; i += kNumThreads*4) {
                const auto blocks = ptx::ld_evict_first(reinterpret_cast<const uint4*>(src + i));
                ptx::st_shared(dst + i,
                    blocks.x*SB + kv_offset, blocks.y*SB + kv_offset,
                    blocks.z*SB + kv_offset, blocks.w*SB + kv_offset);
            }
        }
    } else {
        for (uint32_t i = thread_idx*4; i < num_kv_blocks; i += kNumThreads*4)
            ptx::cp_async_cg<256>(reinterpret_cast<const uint4*>(src + i), reinterpret_cast<uint4*>(dst + i));
    }
}
if constexpr (not kUseUnalignedKs) { cutlass::arch::cp_async_fence(); cutlass::arch::cp_async_wait<0>(); }
```

- **一次搬 4 个 `uint32`（16B）**：步长 `kNumThreads*4`，靠 `NUM_MAX_SPARSE_BLOCKS%4==0` 保证不越界读。
- **对齐版走 `cp.async`**（`ptx::cp_async_cg<256>`）：异步拷贝到 smem，末尾 `cp_async_fence`+`cp_async_wait<0>`
  全部落盘。**非对齐版走同步的 `ld_evict_first` + `st_shared`**：因为加载时必须把每个块号就地变换成
  **绝对 token 起点** `blocks.x*SB + (ks%SB)`（`:257-259`）——这个变换做在寄存器里，`cp.async` 直接
  搬字节做不到。`ld_evict_first` 让这个只读一次的数据不占 L1（evict-first，第 38 篇）。
- 注意 `:264` 的 `cp_async_cg<256>` 是 **16B 传输**（`cp.async.cg` 固定 16 字节），模板参数 256 是
  **L2 prefetch hint**（`ld_st.cuh:167-176`），不是传输长度。

### 5.3 归并路径分块（`:273-291`）——本 kernel 最巧的一处

两条升序列表的归并本质是**顺序**的。要并行化，kernel 把输出区间 `[0, n0+n1)` 按线程切分：

```cpp
// :274-291
const uint32_t num_input_kv_blocks = num_kv_blocks_in_q0 + num_kv_blocks_in_q1;
const uint32_t merge_begin = thread_idx * num_input_kv_blocks / kNumThreads;   // 本线程输出起点
const uint32_t merge_end   = (thread_idx+1) * num_input_kv_blocks / kNumThreads;
uint32_t lo = max(merge_begin, num_kv_blocks_in_q1) - num_kv_blocks_in_q1;
uint32_t hi = min(merge_begin, num_kv_blocks_in_q0);
while (lo < hi) {                                       // 二分：求 merge_begin 处消耗了多少个 q0
    const uint32_t q0 = (lo+hi)/2, q1 = merge_begin - q0;
    if (q1 > 0 and q0 < num_kv_blocks_in_q0 and
        ld_shared(logical[1] + q1-1) >= ld_shared(logical[0] + q0))
        lo = q0 + 1;                                    // q0 还没排完，往后挪
    else
        hi = q0;
}
uint32_t q0_slot_idx = lo, q1_slot_idx = merge_begin - lo;   // 输出 merge_begin 的前置状态
```

**在找什么**：每个线程的起始「状态」是一对 `(q0_slot_idx, q1_slot_idx)`，满足
`q0_slot_idx + q1_slot_idx = merge_begin`——即恰好产出了 `merge_begin` 个输出。二分搜索 q0 的消耗量
`q0_slot_idx ∈ [max(0, merge_begin-n1), min(merge_begin, n0)]`：若「q1 的最后一个已消耗元素
`logical[1][q1-1]` ≥ q0 当前元素 `logical[0][q0]`」，说明 q0 这个元素还没被纳入（它排在后面），
于是 `lo=q0+1`。二分结束后每个线程就知道自己该从两条列表的哪个位置开始往下归并——**把顺序归并
变成了「先二分定位、再局部归并」**，每个线程只需 O(区间长) 的工作。

### 5.4 跨分区去重（`:293-298`）

```cpp
if (num_remaining_inputs > 0 and q0_slot_idx > 0 and q1_slot_idx < num_kv_blocks_in_q1 and
    ld_shared(logical[0] + q0_slot_idx-1) == ld_shared(logical[1] + q1_slot_idx)) {
    ++q1_slot_idx;
    --num_remaining_inputs;
}
```

因为归并**对相等元素两边都前进**（并集），一个重复项若正好落在两个线程的边界上，会被两个线程各
产出一次。上式检查「本线程第一个待产出项，是否等于上一个分区已产出的最后一个 q0 项」；若是，就
**跳过 q1 侧、不产出**（`++q1_slot_idx` 且剩余输出数减一）。与之配对的是 §5.5 末尾的「carry」逻辑。

### 5.5 归并 + 打包（`:300-326`）

```cpp
constexpr uint32_t kPresentBit = 1u << (kNumSparseSlotBits - 1);  // bit15 = 该 token 含此块
constexpr uint32_t kSlotIndexMask = kPresentBit - 1;              // bits0-14 = 槽号
const auto pack_slot = [](uint32_t slot, bool present) { return slot | (present ? kPresentBit : 0u); };
for (uint32_t off = 0; off < kNumKVBlocksPerThread; ++off) {
    if (num_remaining_inputs == 0) continue;
    const uint32_t q0_idx = q0_slot_idx < n0 ? ld_shared(logical[0] + q0_slot_idx) : ~0u;
    const uint32_t q1_idx = q1_slot_idx < n1 ? ld_shared(logical[1] + q1_slot_idx) : ~0u;
    const bool in_q0 = q0_idx <= q1_idx;
    const bool in_q1 = q1_idx <= q0_idx;
    const bool consume_q1 = in_q1 and num_remaining_inputs > in_q0;   // 最后一个重复项留给下一分区
    packed_slots_in_thread[off] = pack_slot(q0_slot_idx, in_q0) | (pack_slot(q1_slot_idx, in_q1) << kNumSparseSlotBits);
    ++num_merged_kv_blocks_in_thread;
    q0_slot_idx += in_q0;
    q1_slot_idx += consume_q1;
    num_remaining_inputs -= in_q0 + consume_q1;
}
DG_DEVICE_ASSERT(num_remaining_inputs == 0);
```

- **打包格式**：低 16 位是 `[present|slot15]` 的 q0，高 16 位是 q1。`pack_slot(·, present)` 里
  `slot` 就是**该元素在原列表里的下标**（q0 每产出一次 +1、q1 同理）。所以槽号天然是「token 自己
  列表里的位置」——正是压缩列映射需要的（第 22/136 篇：`(q_slot_base + slot)·SB + 块内偏移`）。
- **`in_q0`/`in_q1` 的相等判定**：`q0_idx <= q1_idx` 且 `q1_idx <= q0_idx` 同时为真当且仅当相等
  （重复块），此时两个 token 都含此块，两个 present 位都置。
- **carry 技巧**（`:317-318`）：若本元素是重复块但已是本线程的**最后一个输出**
  （`num_remaining_inputs == in_q0 == 1`），则只消耗 q0、把 q1 留到下一分区；下一分区入口的
  §5.4 去重检查会发现并跳过它。这样既不漏也不重。
- `num_remaining_inputs == 0` 时 `DG_DEVICE_ASSERT` 兜底——归并恰好产出并集大小。

### 5.6 压缩 + KV split 预留（`:328-341`）

```cpp
const uint32_t merged_kv_block_base = math::cta_exclusive_sum<kNumThreads>(
    num_merged_kv_blocks_in_thread, smem.warp_sums, num_merged_kv_blocks);   // CTA 内前缀和
const uint32_t num_kv_splits_in_q_block = ceil_div(num_merged_kv_blocks, kNumKVBlocksPerSplit);
if (thread_idx == 0) {
    smem.q_block.kv_split_base = num_kv_splits_in_q_block == 0 ? 0 :
        ptx::atomic_add(&workspace_state->num_kv_splits, num_kv_splits_in_q_block);  // 全局分配
    q_block_infos[q_token_base] = QBlockInfo(smem.q_block.kv_split_base, num_kv_splits_in_q_block);
    if constexpr (kIsPaged) { if (num_q_block_tokens == BLOCK_Q) q_block_infos[q_token_base + 1] = QBlockInfo(0,0); }
}
```

- `cta_exclusive_sum<256>`（`math.cuh:161-184`，第 44 篇）把每个线程局部归并的块数**压缩成紧凑的
  全局下标** `merged_kv_block_base`。
- 一个 Q 块的块数已知，它需要 `ceil(merged/KPS)` 个 split；用 `atomic_add` 从全局游标
  `num_kv_splits` **一次预留**这一段连续下标——这样各 CTA 无锁并发、每个 Q 块拿到一整段连续 split。
- **分页版把第二个 token 的 `q_block_info` 清零**（`:338-339`）：因为成对时 schedule 由第一个 token
  承载（§6.2），第二 token 不单独排。

`merged_kv_block_base + off` 处的 packed_slots 写进 smem（`:342-355`）：非分页写
`packed_slots_by_merged_kv_block[merged_idx]`；分页只在 **split 首块**（`merged_idx % KPS == 0`）写
`packed_slot_bases_by_kv_split[split]`、且**剥掉 present 位**（`& (mask | mask<<16)`）——分页版只需要
每 split 的槽基址。`:356` `__syncthreads()` 之后进入写出。

### 5.7 写出：`write_merged_kv_block`（`:359-415`）

```cpp
const auto write_merged_kv_block = [&](uint32_t merged_idx, uint32_t packed_slots) {
    const uint32_t kv_split_offset = merged_idx / kNumKVBlocksPerSplit;
    const uint32_t kv_split_idx = kv_split_base + kv_split_offset;
    const uint32_t kv_block_idx_in_split = merged_idx % kNumKVBlocksPerSplit;
    // 取本 split 首块的槽基址（分页取每 split 一份；非分页取每块一份）
    uint32_t packed_slot_bases = kIsPaged ? ld_shared(packed_slot_bases_by_kv_split + kv_split_offset)
                                          : ld_shared(packed_slots_by_merged_kv_block + merged_idx - kv_block_idx_in_split);
    const uint32_t q0_slot_idx = packed_slots & kSlotIndexMask;
    const uint32_t q1_slot_idx = (packed_slots >> 16) & kSlotIndexMask;
    const uint32_t q0_slot_base = packed_slot_bases & kSlotIndexMask;
    const uint32_t q1_slot_base = (packed_slot_bases >> 16) & kSlotIndexMask;
    const bool in_q0 = (packed_slots & kPresentBit) != 0;
    const bool in_q1 = ((packed_slots >> 16) & kPresentBit) != 0;
    const uint32_t logical_kv_block_idx = in_q0 ? ld_shared(logical[0] + q0_slot_idx)
                                                : ld_shared(logical[1] + q1_slot_idx);
    uint32_t physical_kv_block_idx;
    if constexpr (kIsPaged) {                          // 分页：查 block_table（块不跨页）
        const uint32_t logical_page_idx = logical_kv_block_idx / kNumKVBlocksPerPage;
        physical = block_table[q_token_base*block_table_stride + logical_page_idx]*kNumKVBlocksPerPage
                 + logical_kv_block_idx % kNumKVBlocksPerPage;
    } else {
        physical = logical_kv_block_idx * (kUseUnalignedKs ? 1 : SPARSE_BLOCK_KV);  // 对齐：乘 SB
    }
    const uint32_t q0_slot_offset = in_q0 ? q0_slot_idx - q0_slot_base : kInvalidSparseSlot;
    const uint32_t q1_slot_offset = in_q1 ? q1_slot_idx - q1_slot_base : kInvalidSparseSlot;
    if (kv_block_idx_in_split == 0) {                  // split 首块：写 header
        const uint32_t n = min(KPS, num_merged_kv_blocks - merged_idx);
        bool is_contiguous = false, has_partial_tail = false;
        if constexpr (not kIsPaged) { ... }            // 见下
        kv_splits[kv_split_idx].header = KVSplitHeader(q_token_base, n, is_contiguous,
            has_partial_tail, q0_slot_base, num_q_block_tokens == BLOCK_Q ? q1_slot_base : ~0u);
    }
    kv_splits[kv_split_idx].kv_block_infos[kv_block_idx_in_split] =
        KVBlockInfo(physical_kv_block_idx, q0_slot_offset, q1_slot_offset);
};
```

- **物理块三态**（第 22 篇 §3.3）：分页查 `block_table`；非对齐 `physical = logical`（此时 logical 已是
  绝对 token 起点）；对齐 `physical = logical*SB`。
- **偏移 vs 基址**：`q0_slot_offset = q0_slot_idx - q0_slot_base`，其中 `q0_slot_base` 取本 split 首块的
  槽号。主 kernel 反解 `slot = base + offset`，得到 token 自己列表里的位置（压缩列映射）。
- **`q1_slot_base = ~0u` 当第二 token 不存在**（`:412`）：主 kernel 只用 q1 的 base 解第二 token 的列，
  而第二 token 不存在时它的 present 位全为 0，不会用到。
- **split 首块判定 contiguous / partial-tail**（`:394-410`）：

```cpp
if constexpr (not kIsPaged) {
    const uint32_t last_packed = ld_shared(packed_slots_by_merged_kv_block + merged_idx + n - 1);
    const bool last_in_q0 = (last_packed & kPresentBit) != 0;
    const uint32_t last_q_slot = last_in_q0 ? (last_packed & kSlotIndexMask)
                                            : (last_packed >> 16) & kSlotIndexMask;
    const uint32_t last_logical = ld_shared(logical[last_in_q0 ? 0 : 1] + last_q_slot);
    const uint32_t last_kv_token_start = last_logical * (kUseUnalignedKs ? 1 : SB);
    if constexpr (not kUseUnalignedKs)
        is_contiguous = (n == KPS) and (last_logical == logical_kv_block_idx + n - 1);  // 满且连续
    has_partial_tail = not is_contiguous and last_kv_token_start + SB > num_kv_tokens;   // 尾块越界
}
```

**为什么只看最后一块**：归并后的块号**升序**（两条升序列表的并集），所以只有最后一块才可能
「越过 KV 张量末端」；`is_contiguous` 也只可能被最后一块破坏。`is_contiguous` 让主 kernel 走
「一条 TMA lane 替所有 copy warp」的快路，`has_partial_tail` 让它走 `cp.async` 尾块裁剪慢路
（第 136 篇 §4.1）。

写出分发（`:417-429`）：分页版**每线程归并的块自己写**（用寄存器里的 packed_slots），非分页版
**grid-stride 重扫一遍**（从 smem 读 packed_slots）——所以非分页才需要 `packed_slots_by_merged_kv_block`
整份缓存（§5.6）。

### 5.8 尾块 padding（`:431-436`）

```cpp
for (uint32_t s = num_merged_kv_blocks + thread_idx; s < num_kv_splits_in_q_block * KPS; s += kNumThreads)
    kv_splits[kv_split_base + s / KPS].kv_block_infos[s % KPS] =
        KVBlockInfo(0, kInvalidSparseSlot, kInvalidSparseSlot);
```

最后一个 split 可能不满 KPS 块；把它**填成无效槽**（`physical=0, 两槽=0xFFFF`），让主 kernel 的拷贝
循环**无分支**（按 KPS 读，无效槽自然被跳过）。

### 5.9 推进 Q 块（`:437-440`）

```cpp
q_token_idx = kIsPaged ? gridDim.x + ptx::atomic_add(&workspace_state->next_q_offset, 1u)
                       : q_token_idx + gridDim.x * BLOCK_Q;
```

分页版**动态认领**（`gridDim.x + old_offset`：先 +gridDim.x 再取回自己的号码，保证取到的都是还没被
别人认领的）；连续版**静态推进**（按块号 +`gridDim.x·BLOCK_Q`）。这就是 §5.1 里 `continue` 丢回的 Q
块会被后续轮次重新取走的原因。

## 6. 末 CTA 建调度（`:443-472`）

### 6.1 谁是末 CTA

```cpp
// :444-451
if (thread_idx == 0) {
    smem.is_last_cta = ptx::atomic_add_rel(&workspace_state->num_finished_ctas, 1u) + 1 == gridDim.x;
    if (smem.is_last_cta != 0) ptx::ld_acq(&workspace_state->num_finished_ctas);   // 取回全部生产者写入
}
__syncthreads();
if (smem.is_last_cta == 0) return;
```

`atomic_add_rel` + `ld_acq` 构成 release-acquire：最后一个 CTA（计数到 `gridDim.x`）**取回所有生产
者写入**的 `kv_splits`/`q_block_infos`，然后独自建调度；其余 CTA 直接返回。这是「先并行生产、再单
点汇总」的固定套路。

### 6.2 连续版：`build_contiguous_schedule`（`:37-73`）

```cpp
// :49-64
if (threadIdx.x < kNumSMs) {
    uint32_t s = ceil_div(total_kv_splits * threadIdx.x, kNumSMs);         // 均分 [begin,end)
    uint32_t s_end = ceil_div(total_kv_splits * (threadIdx.x+1), kNumSMs);
    while (s < s_end) {
        const uint32_t q_token_base = kv_splits[s].header.q_token_base;
        const auto qbi = q_block_infos[q_token_base];
        const uint32_t entry_end = min(s_end, qbi.kv_split_base + qbi.num_kv_splits);  // 不越 Q 块边界
        schedule_entries[num_sm_entries*kNumSMs + threadIdx.x] =
            ScheduleEntry(s, entry_end, q_token_base, min(BLOCK_Q, num_q_tokens - q_token_base));
        s = entry_end;
        ++num_sm_entries;
    }
    atomicMax(&smem.num_waves, num_sm_entries);
}
```

每个 SM（`threadIdx.x < kNumSMs`）先均分到一段 `[ceil(T·sm/SMs), ceil(T·(sm+1)/SMs))` 的 split，
再把这段**按 Q 块边界切开**（第 136 篇主 kernel 每个 entry 处理一个 Q 块）。`num_waves` 取各 SM
条目数的最大值（`atomicMax`，`:63`），不足的补 `(0,0,0,0)` 空条目（`:69-70`）。**为什么按 Q 块切**：
主 kernel 一个 entry = 一个 Q 块（读一份 Q、发一次 MMA），跨 Q 块的 entry 无法处理。切完条目数
可能不同于均分数，故用 `atomicMax` 取真实波数。

### 6.3 分页版：`build_paged_schedule` + `balance_wave_entries`（`:124-161`、`:75-122`）

分页版每个 token 的 split 数不同（KV 长度不同），先把每 token 的 split **切成有界条目**：

```cpp
// :132-153
for (uint32_t q_begin = 0; q_begin < num_q_tokens; q_begin += kNumThreads) {
    const uint32_t t = q_begin + threadIdx.x;
    const auto qbi = t < num_q_tokens ? q_block_infos[t] : QBlockInfo(0,0);
    const uint32_t n = qbi.num_kv_splits;
    const uint32_t n_entries = ceil_div(n, kNumKVSplitsPerEntry);          // 每 entry 至多 8 个 split
    const uint32_t entry_begin = num_entries + math::cta_exclusive_sum<kNumThreads>(n_entries, ...);
    const uint32_t per = n_entries == 0 ? 0 : n / n_entries;               // 尽量均分，余数分给前几个
    const uint32_t larger = n_entries == 0 ? 0 : n % n_entries;
    uint32_t s = qbi.kv_split_base;
    for (uint32_t e = 0; e < n_entries; ++e) {
        const uint32_t s_end = s + per + (e < larger);
        schedule_entries[entry_begin + e] = ScheduleEntry(s, s_end, t, ...);
        s = s_end;
    }
    num_entries += num_batch_entries;
    __syncthreads();
}
```

然后对每一 wave 做**直方图排序 + 旋转**（`:75-122`）：

```cpp
// :84-120（每个 warp 处理一个 wave）
for (uint32_t wave = warp_idx; wave < num_waves; wave += kNumWarps) {
    ...
    // 1) 按「entry 含几个 split」作 key，atomicAdd 直方图排名
    ranks[i] = atomicAdd(histogram + (end - begin), 1u);
    // 2) elect-one 把频率前缀和成起始排名
    if (elect_one_sync()) { for (x=0..K) { freq=hist[x]; hist[x]=rank_begin; rank_begin+=freq; } }
    // 3) 用排名决定目标 SM：重条目旋转开；两 wave 时第二 wave 反向，使活跃 SM 集合互补
    const uint32_t rank = histogram[end-begin] + ranks[i];
    const uint32_t dst = num_waves == 2 and wave == 1 ? kNumSMs - 1 - rank
                          : (rank + kNumSMs - wave * kNumSMs / num_waves) % kNumSMs;
    wave_entries[dst] = entries[i];
}
```

**为什么需要它**：indexer 选出的块数在 token 间**不均匀**（有的 query 10 个块、有的 40 个），若不
排序，某些 SM 可能一口气领到好几个重条目、其余 SM 空转。直方图按「split 数」排序后，重条目被
**均匀轮转**到不同 SM；两 wave 时第二 wave **反向**分配，让两个 wave 的繁忙 SM 错开（互补）。这
与第 23/24 篇 SM100 dense MQA 调度器的「按代价切分」是同一目标的不同手法：dense 用解析式代价模型
+ 二分，sparse 用**离散直方图 + 旋转**（因为稀疏条目数少、分布无解析式）。

### 6.4 收尾（`:463-472`）

末 CTA 写 `MetadataHeader`（`num_kv_splits`/`num_waves`/`use_unaligned_ks`/`num_sms`），并把
`workspace_state` 三个计数器**清零**——因为 workspace 被 host 按 stream 缓存复用（`attention.hpp:288-300`），
下次 launch 必须从 0 开始。注意清零发生在**所有读取之后**：`num_kv_splits` 已在 `:453` 读进局部变量
`total_kv_splits`。

## 7. H100 实测（真实设备代码）

因为整个 kernel 架构无关，`src/sparse_mqa_logits_metadata.cu` **直接 include 真实头文件**、用真实模板
参数 `<<<1, 256, smem>>>` 启动，再把输出与独立 host oracle 逐字段对拍。为消除 `atomic_add` 分配
顺序的不确定性，实验用 **grid=1**（单 CTA 顺序处理所有 Q 块），`kNumSMs=4`、`SPLIT_KV=128`（KPS=16）、
`SB=8`、`NB=32`。核对维度：归并顺序、每个 Q 槽的 `base+offset`（反解回原列表位置）、物理块三态、
split 头（块数 / `q0/q1_slot_base` / contiguous / partial-tail 标记）、以及调度表对 `[0,total)` 的覆盖。

```
---- A1 连续对齐 T=6 NB=32 SB=8 ----         header: splits=3 waves=1
   schedule: cover(miss=0,dup=0,bad=0) -> PASS
   结果: 核对 28 块 / 3 split，不一致=0 -> PASS
---- A2 连续对齐 尾块裁剪 T=4 Nkv=100 ----    核对 26 块 / 2 split，不一致=0 -> PASS
---- B1 连续非对齐 ks%8=5 T=4 ----            核对 20 块 / 2 split，不一致=0 -> PASS
---- C1 分页 单request T=8 ----               核对 36 块 / 4 split，不一致=0 -> PASS
---- A3 连续对齐 多wave T=16 ----             header: splits=16 waves=2
   wave 0: per-SM work=[4,4,4,4]  wave 1: per-SM work=[4,4,4,4]
   结果: 核对 192 块 / 16 split，不一致=0 -> PASS
---- C2 分页 负载不均 T=12 ----               header: splits=9 waves=2
   wave 0: per-SM work=[1,1,4,4]  wave 1: per-SM work=[4,1,0,0]
   结果: 核对 144 块 / 9 split，不一致=0 -> PASS
=== 汇总: ALL PASS（6 组样例，总不一致=0）===
```

要点：

- **归并去重正确**：A1 两 token 列表长度不同、刻意共享若干块；6 组共 **446 个 KV 块 / 36 个 split**，
  `base+offset` 反解、物理块、split 头、调度覆盖**全部 0 不符**。
- **三态物理块都验到**：对齐 `logical*SB`（A1/A2/A3）、非对齐 `logical=绝对 token 起点`（B1，
  `ks%8=5` 时值被平移到 `i*8+5`）、分页 `block_table` 页内映射（C1/C2，页大小 8 块）。
- **尾块裁剪**：A2（`Nkv=100`、每 token 20 块到 `i*5`）触发 `has_partial_tail`，split 头标记与
  oracle 一致。
- **负载均衡调度**：A3 连续版 16 split/4 SM = 2 wave，每 SM 恰 4 个 split（0 偏差）；C2 分页版
  split 数 2/1 交替、6 条目 → 2 wave，`balance_wave_entries` 把工作量轮转开（wave0 `[1,1,4,4]`、
  wave1 反向 `[4,1,0,0]`），**每个 split 恰好被覆盖一次**（`miss=dup=bad=0`）。
- 分页版 `q_block_info[qbase+1]` 被清零、只由第一个 token 承载调度——oracle 与 kernel 一致
  （C1/C2 的块核对全过）。

## 8. 结论与边界

- metadata kernel 是**完全架构无关**的：它只用 `ptx::` 整数/原子、`cp.async`、`cta_exclusive_sum`、
  `cute::` 下标。**唯一「SM100 关联」是 host API 的 `arch_major==10` 断言**（`attention.hpp:313/344/380/414`）
  与它所服务的主 kernel；kernel 本体在 H100 上**原样可跑、结果正确**。
- 本 kernel 的非平凡点集中在**归并去重**（§5.3 的二分归并路径 + §5.4/§5.5 的跨分区去重与 carry）与
  **负载均衡调度**（§6.3 的直方图排序 + 旋转 + 两 wave 反向）——它们都是「在无解析式分布的稀疏工作
  上做确定性并行划分」的手法，与第 22/23/24/136 篇的 sparse/dense 调度器互补。
- **仅分析**的部分属于主 kernel（UMMA/tcgen05/UTCCP/TMA）与 host 的 SM100 门控；本篇在 H100 上验证
  的是 metadata kernel 本体——这是本专题里**少数能整段跑真实 SM100 设备代码**的算子（另见第 23/24/30/102
  篇的调度器）。

## 9. 参考资料

- DeepGEMM 仓库：`https://github.com/deepseek-ai/DeepGEMM`（本地 `057ca59`，Public release 26/09/30）。
- 本专题：第 22 篇（稀疏 MQA logits 概览 + metadata）、第 136 篇（SM100 sparse 主 kernel 逐段精读）、
  第 19 篇（SM90 dense MQA）、第 23/24 篇（SM100 dense 调度器）、第 60 篇（MQA logits 验收测试）。
- 实测原始输出：`harness/exp_sparse_mqa_logits_metadata.out.txt`。
