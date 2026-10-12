# DeepGEMM SM100 dense MQA logits：分页版 metadata 负载均衡调度器（SM100 主 kernel 仅分析 + H100 真实调度器实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P14 增量（第 24 篇）。
> 第 19 篇拆了稀疏注意力 indexer 的 **dense MQA logits**（`fp8_fp4_mqa_logits` 连续 /
> `fp8_fp4_paged_mqa_logits` 分页）的 **SM90** 实现；第 22 篇拆了 **SM100 sparse** 版；
> 第 23 篇拆了 **SM100 dense 的连续版**调度器。本篇补上**分页版**：与连续版同一套主 kernel
> （`impls/sm100_mqa_logits.cuh`，block-scaled UMMA + TMEM，**SM100 专属，仅分析**），
> 但把 `sm100_mqa_logits_metadata` 换成分页请求模型——**varlen 请求分组、请求内 Q block 均分、
> 代价溢出回退、同一请求内多个 Q block 复用同一段 KV、分页坐标滑动缓存**。
> 这些调度逻辑**与 UMMA 无关**，在 H100 上编译运行**真实设备代码**逐条验证
> （`harness/exp_mqa_logits_paged_schedule.*`、`src/mqa_logits_paged_schedule.cu`）。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。主实现：
> `csrc/apis/attention.hpp:136-282`、`csrc/jit_kernels/impls/sm100_mqa_logits.hpp:13-61,117-381`、
> `deep_gemm/include/deep_gemm/scheduler/sm100_paged_mqa_logits.cuh`、
> `deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh`、
> `deep_gemm/include/deep_gemm/layout/mqa_logits.cuh`。

## 1. 四篇的位置关系

| 篇目 | 版本 | 调度 | 本机(H100/sm90) |
|---|---|---|---|
| 第 19 篇 | dense MQA logits | SM90 persistent round-robin（分页走 `sm90_paged_mqa_logits.cuh`） | SM90 主 kernel 可跑 |
| 第 22 篇 | sparse MQA logits | SM100 专用 metadata（合并去重 + 均衡） | 仅语义 |
| 第 23 篇 | dense MQA logits 的**连续版** | `sm100_mqa_logits_metadata` + `SM100MQALogitsScheduler`（KV span 并集） | **调度器真实设备代码可跑** |
| **本篇** | dense MQA logits 的**分页版** | `sm100_paged_mqa_logits_metadata` + `SM100PagedMQALogitsScheduler` | **调度器真实设备代码可跑** |

语义没变（第 19 篇 §2.1）：

```
logits[m, n] = Σ_h relu(<q[m,h,:], kv[n,:]>) · w[m,h] · kv_scale[n]
```

分页版与连续版的差别只在 **KV 的来源与调度模型**：连续版的 Q block 各自有一段
「有效的 KV 区间并集」（变长 k_start/k_end），分页版则是 **varlen 请求**——`indices` 把
Q token 划分成若干请求，同一请求共享一段 KV（`block_table` 的同一行），KV 以固定页
（`block_kv ∈ {32,64,128}`）存在 paged cache 里。

## 2. host API：分页 metadata 与主调用（`csrc/apis/attention.hpp`）

### 2.1 分页 metadata 入口（`:136-172`）

```cpp
static torch::Tensor get_paged_mqa_logits_metadata(context_lens, block_kv, num_sms, indices) {
    DG_HOST_ASSERT(context_lens.dim() == 2);            // :139 只支持 2D context lens
    ...
    auto schedule_metadata = torch::empty({num_sms + 1, 2}, ...);  // :149
    if (arch_major == 10) {
        DG_HOST_ASSERT(is_varlen and next_n == 1 and
                       (block_kv == 32 or block_kv == 64 or block_kv == 128));  // :154
        DG_HOST_ASSERT(reinterpret_cast<uintptr_t>(indices.data_ptr()) % 8 == 0); // :160
        sm100_paged_mqa_logits_metadata(...);           // :161-162
    } else if (arch_major == 9) {
        DG_HOST_ASSERT(not is_varlen and block_kv == 64);   // :164-165  SM90 不支持 varlen
        sm90_paged_mqa_logits_metadata(...);            // :166
    }
}
```

注意与连续版的三个区别：
1. **SM100 分页强制 varlen**（`:154`）：请求由 `indices` 给出，`next_n==1`（每行一个 token）。
2. `indices` 必须 **8 字节对齐**（`:160`），因为 metadata kernel 用 `uint2` 一次读两个 token（见 §3.1）。
3. SM100 的 `block_kv` 支持 32/64/128，SM90 只支持 64（`:202`）。metadata 张量形状
   `[num_sms+1, 2]`（`:149`，int32）——**每 SM 一个 (q_token_idx, kv_split_idx) 起点**，
   比连续版（每 SM 起点 + 计数 + 每 Q block 一个 span）紧凑得多。

### 2.2 主调用（`:174-282`），仅列与调度相关的部分

```cpp
const int split_kv = arch_major == 10 ? kMQALogitsSplitKV : 256;   // :258  SM100 = 384
// logits 行 stride 必须 1024B 对齐且能容纳整段 split
const int stride_logits_alignment = 1024 / elementSize(out_dtype); // :260
const int split_stride_alignment = split_kv / gcd(split_kv, stride_logits_alignment) * stride_logits_alignment; // :261
const auto aligned_max_context_len = align(max_context_len, split_stride_alignment); // :262
auto logits = torch::empty({batch_size * next_n, aligned_max_context_len}, ...); // :263
logits = logits.slice(-1, 0, max_context_len);                     // :264  只暴露有效区间
```

**fused KV cache 的布局**（`:200-223`）：`fused_kv_cache` 是 4D `[num_kv_blocks, block_kv,
num_heads_kv=1, head_dim+sf_bytes]`（`:205`），KV 值在前 `head_dim` 列、scale 紧随其后
（`:218-223` 用 `from_blob` 切出 `kv_cache` 与 `kv_cache_sf`）。`sf_bytes` 为 4（fp32）
或 4（packed UE8M0 int32，`:204`）。分页主 kernel 在 SM100 用 `tensor_map_kv`（3D，`:322-326`）
按页取 KV。`block_table` 是 `[batch_size, max_block_len]`、stride(1)==1（`:228`）。

## 3. metadata kernel 精读（`scheduler/sm100_paged_mqa_logits.cuh`）

整个文件 283 行，结构：
`RequestInfo`（`:12-47`）→ `sm100_paged_mqa_logits_metadata`（`:50-159`）→
`SM100PagedMQALogitsScheduler`（`:162-281`）。它 **include 了连续版的 `sm100_mqa_logits.cuh`**
（`:5`）以复用 `MQALogitsTask`、前缀和、二分。

### 3.1 请求分组：`RequestInfo` + 扫描 `indices`（`:12-47`、`:71-98`）

```cpp
struct RequestInfo {                 // :13-27
    uint32_t q_token_start, num_q_tokens, num_q_blocks, num_kv_splits, num_kv_pages;
    RequestInfo(q_token_start, num_q_tokens, context_len):
        num_q_blocks(ceil_div(num_q_tokens, BLOCK_Q)),
        num_kv_splits(ceil_div(context_len, SPLIT_KV)),
        num_kv_pages(ceil_div(context_len, PAGE_KV)) {}
    static RequestInfo from_q_token(q_token_idx, num_q_tokens_total, context_lens, indices) {
        const uint32_t request_id = indices[q_token_idx];      // :34
        uint32_t q_token_end_idx = q_token_idx + 1;
        while (q_token_end_idx < num_q_tokens_total and
               indices[q_token_end_idx] == request_id) ++ q_token_end_idx;  // :36-37
        return RequestInfo(q_token_idx, q_token_end_idx - q_token_idx,
                           context_lens[q_token_end_idx - 1]);  // :38  最后一个 token 的 context_len
    }
    void get_q_block_span(q_block_idx, token_offset, num_tokens) const {   // :42-46
        const uint32_t base = num_q_tokens / num_q_blocks,
                       remainder = num_q_tokens % num_q_blocks;
        token_offset = q_block_idx * base + min(q_block_idx, remainder);   // :44
        num_tokens = base + (q_block_idx < remainder ? 1 : 0);              // :45
    }
};
```

三个「为什么」：

- **请求 = `indices` 里连续相同的段**（`:36-37`）。`context_lens` 每个 token 一个，但请求的
  context 取 **最后一个 token** 的值（`:38`）。注释「the last token has the longest context」
  （`:29`）说明这是 **chunked prefill** 的约定：一个请求被拆成多行 Q，越靠后的行看到的 KV 越长，
  最后一行即完整上下文。
- **请求内 Q token 被「均分」到 `num_q_blocks` 个 Q block**（`:43-45`）：`base + remainder`
  而非固定 `BLOCK_Q`。这与连续版根本不同——连续版的 Q block 边界固定为 `BLOCK_Q`，
  有效区间靠 span 并集处理；分页版则让**每个 Q block 覆盖尽量相等的 token 数**，
  因为同一请求的 KV 对所有 Q block 是同一段，均分能让「KV 复用 × Q block」的代价最均匀。
- `num_kv_splits` 与 `num_kv_pages` 分别按 `SPLIT_KV=384` 与 `PAGE_KV=block_kv` 上取整
  （`:26-27`）。

**metadata kernel 的扫描**（`:71-98`）：用 `uint2` 一次读两个 token（`:73`、`:81`），
第一遍**数**请求起点（`:91-92`），`cta_exclusive_sum<kNumThreads>` 算出全局偏移（`:93`），
第二遍**写**起点（`:94`）。这样只扫两遍、无原子操作。

```cpp
const auto scan_request_starts = [&](auto&& on_start) {         // :77-90
    uint32_t prev_id = token_begin_idx > 0 and token_begin_idx < token_end_idx
                       ? indices[token_begin_idx - 1] : 0u;      // :78  跨线程边界的前一个 id
    for (; token_idx + 2 <= token_end_idx; token_idx += 2) {
        const uint2 ids = indices_vec2[token_idx / 2];            // :81
        if (token_idx == 0 or ids.x != prev_id) on_start(token_idx);     // :82-83
        if (ids.y != ids.x) on_start(token_idx + 1);             // :84-85  两 token 之间断点
        prev_id = ids.y;
    }
    if (token_idx < token_end_idx and (token_idx == 0 or indices[token_idx] != prev_id))
        on_start(token_idx);                                      // :88-89  尾部单 token
};
```

`:78` 的 `prev_id` 是用来处理「线程负责的区间起点恰好在请求中间」——必须先读前一个 token
的 id 才知道本区间第一个 token 是不是请求起点。

### 3.2 代价模型与【溢出回退】（`:109-142`）

```cpp
const auto get_split_cost = [&](const uint32_t& num_q_tokens) {          // :110-112
    return 2 * BLOCK_Q * ceil_div(num_q_tokens, BLOCK_Q) + num_q_tokens;
};
constexpr uint32_t kCostOverflowFlag = 0x80000000u;                       // :115
bool cost_overflow = false;
for (request_idx = thread_idx; ...) {                                     // :117-123
    const uint64_t cost = ceil_div(context_len, SPLIT_KV) * get_split_cost(num_q_tokens);
    request_work_prefix[request_idx] = (uint32_t)cost;                    // :121  截断
    cost_overflow |= cost > 0xffffffffu;                                  // :122
}
__syncthreads();
if (num_requests > 0) mqa_logits_metadata_prefix_scan<...>(...);          // :125-126
for (request_idx = thread_idx + 1; ...)                                   // :127-128
    cost_overflow |= request_work_prefix[request_idx] < request_work_prefix[request_idx - 1];
if (cost_overflow) atomicOr(num_requests_shared, kCostOverflowFlag);      // :129-130
__syncthreads();
const bool use_token_cost = (*num_requests_shared & kCostOverflowFlag) != 0;  // :132
if (use_token_cost) {                                                     // :133-141
    // 回退：每请求 work = num_kv_splits * num_q_tokens，重算前缀和
}
```

**为什么**：代价模型是 `num_kv_splits × split_cost`（`split_cost = 2·BLOCK_Q·num_q_blocks +
num_q_tokens`，`:111`）——KV/MMA 随 Q block 走，epilogue 随有效 token 走（`:109` 注释）。
两项之和可能超过 `uint32`。这里用**两个信号**判断溢出：单请求代价 > 2³²（`:122`），
以及**前缀和回绕**（后一项 < 前一项，`:128`）。命中后把代价换成更小的
`num_kv_splits × num_q_tokens`（少了 `2·BLOCK_Q·num_q_blocks` 项），并**把溢出标志压进
`num_requests` 字的最高位**（`:130`，`num_requests` 本身 < 2³¹ 所以安全），`use_token_cost`
再读出来（`:132`）。这是一个很省 smem 的技巧：**不额外开一个变量，借一个用不满的计数位**。

### 3.3 均衡边界：只在【请求的 split 粒度】切（`:144-158`）

```cpp
const uint32_t cost_per_sm = total_cost / kNumSMs;               // :145
const uint32_t cost_remainder = total_cost % kNumSMs;            // :146
for (sm_idx = thread_idx; sm_idx <= kNumSMs; sm_idx += kNumThreads) {
    const uint32_t target = sm_idx * cost_per_sm + min(sm_idx, cost_remainder);  // :148
    const uint32_t request_idx = mqa_logits_metadata_upper_bound(request_work_prefix, num_requests, target); // :149
    uint32_t q_token_idx = num_q_tokens_total, kv_split_idx = 0; // :150  尾哨兵 = 越过末尾
    if (request_idx < num_requests) {
        const uint32_t cost_before = request_idx == 0 ? 0 : request_work_prefix[request_idx - 1];
        uint32_t num_q_tokens, context_len;
        get_request_info(request_idx, q_token_idx, num_q_tokens, context_len);   // :154
        kv_split_idx = (target - cost_before) /
                       (use_token_cost ? num_q_tokens : get_split_cost(num_q_tokens)); // :155
    }
    reinterpret_cast<uint2*>(schedule_meta)[sm_idx] = make_uint2(q_token_idx, kv_split_idx); // :157
}
```

与连续版的关键差异（对照第 23 篇 §4）：
- 连续版有 **段启动代价 `kSegmentCost=3`**，边界 `locate` 要「减去启动代价再除以 split 代价」
  （`sm100_mqa_logits.cuh:152`）。分页版**没有启动项**，边界直接
  `(target - cost_before) / per_split`（`:155`）。
- 分页版只写 **`(q_token_idx, kv_split_idx)`** 一对（`:157`），不写 work 计数——因为请求内
  Q block 数可由 `(q_token_idx, context_len)` 反推。第 `kNumSMs` 项是**尾哨兵**
  `(num_q_tokens_total, 0)`（`:150`）。
- `BLOCK_Q` 在 metadata 里固定为 **4**（`sm100_mqa_logits.hpp:32`，注释「Tuned for H32」），
  与实际主 kernel 的 `BLOCK_Q = 128/num_heads` 可以不同（见 §6 实测）。

前缀和 `mqa_logits_metadata_prefix_scan`（`sm100_mqa_logits.cuh:57-79`）用的是
**偶/奇交错就地累加**（`:63-70`）再 `cta_exclusive_sum`，而二分
`mqa_logits_metadata_upper_bound`（`:82-93`）找第一个 `prefix > target`。

## 4. Scheduler 精读（`scheduler/sm100_paged_mqa_logits.cuh:162-281`）

### 4.1 结构与游标（`:163-212`）

```cpp
template <uint32_t kNumHeads, uint32_t BLOCK_Q, uint32_t SPLIT_KV, uint32_t PAGE_KV,
          uint32_t kSplitsPerChunk>
struct SM100PagedMQALogitsScheduler {
    static constexpr bool kIsPaged = true;                          // :164
    static constexpr uint32_t kPageKV = PAGE_KV;                    // :165
    static constexpr uint32_t kNumPagesPerSplit = SPLIT_KV / PAGE_KV;   // :166
    static constexpr uint32_t kNumCachedPages = 32;                 // :168  每 lane 缓存一页
    DG_STATIC_ASSERT(BLOCK_Q * kNumHeads <= 128, ...);             // :169
    DG_STATIC_ASSERT(SPLIT_KV % PAGE_KV == 0 and kNumPagesPerSplit <= kNumCachedPages, ...); // :170
    ...
    uint32_t end_q_token_idx, end_kv_split_idx;      // :179  下一个 SM 的起点
    Info current;                                    // :182  当前请求
    uint32_t current_kv_split_base, current_q_block_in_request;   // :183-184
    const uint32_t* task_block_table_row;            // :187  保留已发射任务的查表状态
    uint32_t task_num_kv_pages, cached_page_base, cached_page_coord;  // :188-190

    SM100PagedMQALogitsScheduler(sm_idx, num_q_tokens_total, context_lens, indices,
                                 block_table, block_table_stride, schedule_meta) {  // :192-212
        const auto start = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx];       // :202
        const auto end   = reinterpret_cast<const uint2*>(schedule_meta)[sm_idx + 1];   // :203
        end_q_token_idx = end.x; end_kv_split_idx = end.y;                              // :204-205
        current.q_token_start = start.x; current_kv_split_base = start.y;               // :207-208
        current_q_block_in_request = 0;                                                 // :209
        if (has_work()) current = Info::from_q_token(start.x, ...);                     // :210-211
    }
```

**每 SM 用 `schedule_meta[sm]` 作起点、`schedule_meta[sm+1]` 作终点**（`:202-205`），
不需要额外的 work 计数。

### 4.2 `next_q_block`：Q block / chunk / 请求三层推进（`:215-251`）

```cpp
bool has_work() const {                                          // :215-218
    return current.q_token_start < num_q_tokens_total and
           (current.q_token_start != end_q_token_idx or current_kv_split_base < end_kv_split_idx);
}

bool next_q_block(MQALogitsTask& task) {                          // :221-251
    if (not has_work()) return false;
    // 当前请求的 split 上界，在【下一个 SM 的起点】截断
    const uint32_t upper = (current.q_token_start == end_q_token_idx)
                         ? end_kv_split_idx : current.num_kv_splits;   // :226
    const uint32_t remaining = upper - current_kv_split_base;          // :227
    uint32_t q_block_token_offset;
    current.get_q_block_span(current_q_block_in_request, q_block_token_offset, task.num_q_tokens); // :229
    task.q_token_base = current.q_token_start + q_block_token_offset;  // :230
    task.kv_token_base = current_kv_split_base * SPLIT_KV;             // :231
    task.num_kv_splits = (current.num_q_blocks == 1) ? remaining
                       : min(remaining, kSplitsPerChunk);              // :232
    task.kv_shared_with_prev = current_q_block_in_request > 0;         // :233
    task_block_table_row = block_table + current.q_token_start * (uint64_t)block_table_stride; // :234
    task_num_kv_pages = current.num_kv_pages;                          // :235
    cached_page_base = numeric_limits<uint32_t>::max();                // :236  强制首次查表

    // 按 Q block → chunk → 请求 推进
    if (++ current_q_block_in_request == current.num_q_blocks) {       // :239
        current_q_block_in_request = 0;
        current_kv_split_base += task.num_kv_splits;                   // :241  一「波」走完所有 Q block
        if (current_kv_split_base >= upper and current.q_token_start != end_q_token_idx) {  // :242
            current.q_token_start += current.num_q_tokens;              // :244  下一个请求，split 归零
            current_kv_split_base = 0;                                 // :245
            if (has_work()) current = Info::from_q_token(...);         // :246-247
        }
    }
    return true;
}
```

**执行顺序是「Q block → chunk → 请求」**：对当前 `[current_kv_split_base, +kSplitsPerChunk)`
这段 KV，**每个 Q block 各发一个任务**（`:233` 的 `kv_shared_with_prev` 就是「同一段 KV 的上一个
Q block」）；所有 Q block 走完才把 `current_kv_split_base` 推进一个 chunk（`:241`）。
`kSplitsPerChunk=8`（`sm100_mqa_logits.hpp:17`）——**同一请求的 KV 被 Q block 复用 8 个 split 的量**。

**`num_q_blocks == 1` 时不分块**（`:232`）：只有一个 Q block 就不必复用，直接一次发完
`remaining` 个 split。这解释了实测里「tasks 数 < slots 数」。

**`has_work` 的双条件**（`:216-217`）保证：SM 在自己的起点与下一个 SM 起点之间推进，
且必须**严格早于**下一个 SM 的起点才算有工作——避免相邻 SM 重复发射边界任务。

**每 token KV 上界**（`:254-266`）：第二个 `next_q_block` 重载用 warp shuffle
（`ptx::exchange`，`:263`）把 `context_lens[min(q_token_base+lane, T-1)]` 广播给 BLOCK_Q 个 token，
`seq_k_start` 恒为 0（分页 varlen 每请求从 0 开始），尾行 clamp 到最后一个有效行（`:258`）。

### 4.3 分页坐标：每 lane 缓存一页的滑动窗口（`:269-280`）

```cpp
void get_kv_page_coords(const MQALogitsTask& task, const uint32_t& kv_split_idx,
                        int (&page_coords)[kNumPagesPerSplit]) {
    const uint32_t page_base = task.kv_token_base / PAGE_KV + kv_split_idx * kNumPagesPerSplit; // :271
    if (page_base < cached_page_base or
        page_base + kNumPagesPerSplit > cached_page_base + kNumCachedPages) {   // :272  窗口外则 refill
        const uint32_t page_offset = page_base + ptx::get_lane_idx();           // :273
        cached_page_base = page_base;
        cached_page_coord = page_offset < task_num_kv_pages
                          ? task_block_table_row[page_offset] : 0;              // :275  越界页映射到 0
    }
    #pragma unroll
    for (uint32_t page_idx = 0; page_idx < kNumPagesPerSplit; ++page_idx)
        page_coords[page_idx] = ptx::exchange(cached_page_coord,
                                              page_base - cached_page_base + page_idx); // :279
}
```

**机制**：`block_table` 的查表是「逻辑页 → 物理块」。这里不用每次都读 global，
而是让 **32 个 lane 各缓存一个连续页**（`kNumCachedPages=32`，`:168`），需要某个页时用
`ptx::exchange`（`:279`）从对应 lane **广播**过来。当目标页超出 `[cached_page_base,
cached_page_base+32)` 窗口时（`:272`）整体 refill 一次（每 lane 重读一页，`:273-275`）。
`kNumPagesPerSplit ≤ 32` 的静态断言（`:170`）保证一次 split 的页都能从窗口内取到。
越界页（`page_offset ≥ num_kv_pages`）映射到 0（`:275`），配合主 kernel 把无效 KV 置零。

## 5. H100 实测（`harness/exp_mqa_logits_paged_schedule.*`）

实验在 H100(sm90) 上**直接编译运行** `sm100_paged_mqa_logits_metadata` 与
`SM100PagedMQALogitsScheduler` 真实设备代码（模板实例化见 `sm100_mqa_logits.hpp:45-49` /
`:347-364` 的 host 调用），做四件事：

1. device metadata 的每 SM 边界 vs **host 独立复刻**（请求分组 / 溢出回退 / 前缀和 / 二分）逐字段对拍；
2. 真实 scheduler 发射任务，验证 **所有 `(请求, Q-block, split)` 恰好覆盖一次**（dup/miss/bad 全 0）；
3. 校验**块大小 ≤ 8、`kv_shared_with_prev`、每 token KV 上界**；
4. 用真实 `get_kv_page_coords` 验证分页查表（含滑动缓存中途 refill）与 host 逐字段一致。

关键结果（全部 **ALL PASS**）：

| 用例 | BLOCK_Q | H | 请求数 | slots(sched/meta) | 任务数 | 覆盖 | 分页校验 |
|---|---|---|---|---|---|---|---|
| uniform req=8 ctx=2048 | 4 | 32 | 64 | 768 / 768 | 348 | dup/miss/bad=0 | 4608/0 |
| skewed req/ctx | 4 | 32 | 9 | 152 / 152 | 152 | 0 | 888/0 |
| mixed heavy/light | 4 | 32 | 200 | 1250 / 1250 | 1122 | 0 | 3546/0 |
| PAGE_KV=32 | 4 | 32 | 40 | 320 / 320 | 280 | 0 | 3840/0 |
| PAGE_KV=128 | 4 | 32 | 40 | 640 / 640 | 312 | 0 | 1920/0 |
| **H=64 → BLOCK_Q=2** | 2 | 64 | 48 | **432 / 288** | 396 | 0 | 2592/0 |
| **H=128 → BLOCK_Q=1** | 1 | 128 | 48 | **864 / 288** | 792 | 0 | 3456/0 |
| skewed 8 SM | 4 | 32 | 6 | 238 / 238 | 66 | 0 | 744/0 |
| few reqs → empty SM | 4 | 32 | 3 | 66 / 66 | 20 | 0 | 396/0 |
| **cost overflow → token** | 4 | 32 | 1 | — | — | 边界 match | — |

**读出的几个点**：

- **边界逐字段 0 不符**：host 复刻的请求分组、代价模型、溢出回退、二分定位与真实 device
  metadata kernel 完全一致（含 132 SM 与 8 SM、含空 SM）。
- **覆盖恰好一次**：scheduled 任务在所有用例下 dup=miss=bad=0，且 `covered == slots`。
  任务数常 **小于** slots（如 768→348）——因为 `num_q_blocks==1` 且 KV 多的请求一次发完整个
  请求的 split，以及同一段 KV 被多个 Q block 复用后合并进一个任务计数。
- **streaming 覆盖的边界情形**：`skewed 8 SM` 的 per-SM work `[24..33]`（少 SM 时能看清
  均衡），`few reqs → empty SM` 验证了「请求数 < SM 数」时大量 SM 分到尾哨兵空区间。
- **`BLOCK_Q` 与 metadata `block_q=4` 不一致**：H=64（`BLOCK_Q=2`）时
  **sched slots 432 vs meta slots 288**，H=128（`BLOCK_Q=1`）时 **864 vs 288**——metadata 的
  代价模型按固定 `block_q=4` 估算（`sm100_mqa_logits.hpp:32` 注释「Tuned for H32」），
  头多时**明显低估** Q-block 工作量。这正是注释说「其他头数均衡略差」的原因。**但覆盖仍然精确**
  （dup/miss=0）——均衡只是估计，正确性由 `next_q_block` 的双条件保证。
- **分页查表**：含 PAGE_KV=32（每 split 12 页）与 chunk=8 的用例（单任务需 96 页 > 32 缓存），
  即**中途 refill** 被真实触发，查表结果与 host `block_table` 逐字段一致（mismatch=0）。
- **代价溢出回退**：单请求 4096 token、`context_len=1.56e8`，`split_cost` 代价 4.99e9 > 2³²
  触发回退，`use_token_cost=Y`、`total_cost=1.664e9`，边界仍与 host 完全一致。
  （该用例 split 数达 40 万，不做 scheduler 覆盖测试——已确认 metadata 的溢出路径正确。）

## 6. 结论与坑

1. **分页 vs 连续，调度是同一套模板**：前缀和 + 二分定位 + `MQALogitsTask` 复用；差别是
   分页以**请求**为分组单位（`indices` run-length）、以 `(q_token_idx, kv_split_idx)` 一对
   作每 SM 边界、没有段启动代价，且同一请求的 KV 被多个 Q block **复用**（`kv_shared_with_prev`
   + `kSplitsPerChunk` 分块）。
2. **溢出回退是工程细节里的亮点**：用一个计数位的最高位同时表达「溢出」与「回退后仍是有效计数」，
   省掉一个 smem 变量（`:115`、`:129-132`）。
3. **metadata 的 `block_q` 固定 4、与主 kernel `BLOCK_Q` 解耦**：这是「先跑一个便宜 metadata，
   主 kernel 按头数选 Q tile」的折中；代价模型只做均衡，不做正确性保证。
4. **分页坐标的滑动缓存**：`kNumCachedPages=32` 正好 ≥ 每 split 页数上限（PAGE_KV=32 时 12），
   一次 refill 覆盖一个 split 的窗口；`ptx::exchange` 把 lane 当寄存器堆使用。
5. **SM100 专属**：主 kernel（block-scaled UMMA + TMEM + UTCCP 搬 SF）在 `impls/sm100_mqa_logits.cuh`，
   本篇**仅分析**；metadata/scheduler 的调度逻辑与 UMMA 无关，故能在 H100 跑真实设备代码。
   host 侧 `get_paged_mqa_logits_metadata` 在 SM90 会走 `sm90_paged_mqa_logits_metadata`
   且**拒绝 varlen**（`attention.hpp:164`）。

## 参考

- DeepGEMM 仓库：`https://github.com/deepseek-ai/DeepGEMM`（本专题对照公开版 commit `057ca59`）
- DeepGEMM PR #95（SM100 支持）：`https://github.com/deepseek-ai/DeepGEMM/pull/95`
- DeepGEMM PR #316（EP8 基准 / Mega MoE 相关优化）：`https://github.com/deepseek-ai/DeepGEMM/pull/316`
- 本专题第 19 篇（SM90 dense/paged MQA logits）、第 22 篇（SM100 sparse）、第 23 篇（SM100 dense 连续版调度器）
- 实验：`code/mega-moe/src/mqa_logits_paged_schedule.cu`、
  `code/mega-moe/harness/exp_mqa_logits_paged_schedule.sh` / `.out.txt`
