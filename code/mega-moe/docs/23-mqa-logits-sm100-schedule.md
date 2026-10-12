# DeepGEMM SM100 dense MQA logits：主 kernel 与「连续版负载均衡调度器」（SM100 主 kernel 仅分析 + H100 真实调度器实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P13 增量（第 23 篇）。
> 第 19 篇拆了稀疏注意力 indexer 的 **dense MQA logits**（`fp8_fp4_mqa_logits` 连续 /
> `fp8_fp4_paged_mqa_logits` 分页）的 **SM90** 实现（H100 可跑）；第 22 篇拆了 **SM100 sparse** 版。
> 本篇补上中间缺席的一块：**SM100 dense 版**——它是 `sm90_fp8_mqa_logits` 的 MXFP4/MXFP8 升级，
> 主 kernel（`sm100_mqa_logits.cuh`）是 block-scaled UMMA + TMEM，**SM100 专属，仅分析**；
> 但它把 SM90 的「persistent round-robin」换成了**先跑一个 metadata kernel 做按代价负载均衡**，
> 而**这个调度器与 UMMA 无关**，因此可以在 H100 上编译运行**真实设备代码**并逐条验证
> （`harness/exp_mqa_logits_schedule.*`、`src/mqa_logits_schedule.cu`）。
>
> 为什么与 Mega MoE 专题相关：Mega MoE / sparse logits / 本篇 dense logits 都是
> 「**一个轻量 metadata/调度 kernel + 一个 persistent 主 kernel**」的两段式结构。
> Mega MoE 的调度器在 `scheduler/mega_moe.cuh`，sparse/dense logits 的调度器在
> `scheduler/sm100_*_mqa_logits.cuh`，**代价模型 + 前缀和 + 边界二分**是同一套模板。
> 精读它有助于理解 DeepGEMM 在「工作块极不均匀」（变长上下文）时如何做到 SM 间负载均衡。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。主实现：
> `csrc/apis/attention.hpp:46-134`、`csrc/jit_kernels/impls/sm100_mqa_logits.hpp`、
> `deep_gemm/include/deep_gemm/scheduler/sm100_mqa_logits.cuh`、
> `deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`、
> `deep_gemm/include/deep_gemm/layout/mqa_logits.cuh`。

## 1. 三篇的位置关系

| 篇目 | 版本 | 架构 | 调度 | 本机(H100/sm90) |
|---|---|---|---|---|
| 第 19 篇 | dense MQA logits | sm90 + sm100（主 kernel 分派） | persistent round-robin（SM90）；`schedule_meta`（SM100） | SM90 主 kernel 可跑 |
| 第 22 篇 | sparse MQA logits | 仅 sm100 | 专用 metadata kernel（合并去重 + 均衡） | 仅语义 |
| **本篇** | **dense MQA logits 的 SM100 版** | 仅 sm100（主 kernel） | **`sm100_mqa_logits_metadata` + `SM100MQALogitsScheduler`** | **调度器真实设备代码可跑** |

语义没变，仍是每 query token 对每个 KV token 的加权 ReLU MQA 分（第 19 篇 §2.1）：

```
logits[m, n] = Σ_h relu(<q[m,h,:], kv[n,:]>) · w[m,h] · kv_scale[n]
```

SM100 版把 QK 换成 **MXFP4/MXFP8**（Q 带 packed UE8M0 MX scale），输出 **bf16**（SM90 是 fp32），
并把「有效 KV 区间」的并集、以及跨 SM 的工作划分为 **split 粒度**。

## 2. host API（`csrc/apis/attention.hpp`）

### 2.1 metadata 入口（`:46-65`）

```cpp
static torch::Tensor get_mqa_logits_metadata(cu_seq_len_k_start, cu_seq_len_k_end,
                                             num_kv_tokens, num_heads) {
    DG_HOST_ASSERT(jit->device.get_arch_major() == 10);          // :49 仅 SM100
    ...
    const int block_q = get_mqa_logits_block_q(num_heads);       // :57
    const int num_sms = runtime->get_num_sms();                  // :58
    const int required_words = get_mqa_logits_metadata_num_words(num_q_tokens, block_q, num_sms);  // :59
    auto schedule_meta = torch::empty({required_words}, ...);    // :60
    sm100_mqa_logits_metadata(...);                              // :62-63
    return schedule_meta;
}
```

`block_q = 128 / num_heads`（`sm100_mqa_logits.hpp:21-24`，`num_heads <= 128 且 %4==0`）——
**Q tile 永远覆盖 128 个 (token, head) 行**：头多则每块 token 少，头少则每块 token 多。
`block_q=1`（H=128）/`2`（H=64）/`4`（H=32）/`8`（H=16）……**metadata 的粒度就是 block_q**。

### 2.2 主入口 `fp8_fp4_mqa_logits`（`:67-134`）

- `out_dtype = arch_major == 10 ? kBFloat16 : kFloat32`（`:80`）。
- `block_kv = arch_major == 10 ? kMQALogitsSplitKV : 256`（`:111`），即 **SM100 的 split 宽 384**、SM90 是 256。
- 输出按 `aligned_seq_len = align(seq_len, block_q)` 分配、行 stride 做 **1024 字节对齐**（`:112-115`），
  再切片成 `[seq_len, max_seqlen_k]`（`:117`）——TMA/max 对齐要求。
- 分派（`:120-129`）：SM100 收 `schedule_meta`（可选），SM90 **拒绝** `schedule_meta`（`:125`）、
  且只支持 H=32/64、E4M3。

`kMQALogitsSplitKV = 384`、`kMQALogitsNumTmemStages = 3`、`kMQALogitsNumSpecializedThreads = 128`
等常量在 `sm100_mqa_logits.hpp:11-18`；`kMQALogitsSplitCost=2 / kMQALogitsSegmentCost=3` 是**代价模型**
（见 §4.4）。

## 3. jit wrapper（`csrc/jit_kernels/impls/sm100_mqa_logits.hpp`）

### 3.1 metadata 缓冲区字数（`:64-69`）

```cpp
static int get_mqa_logits_metadata_num_words(num_q_tokens, block_q, num_sms) {
    const int num_q_blocks = ceil_div(num_q_tokens, block_q);
    // Int32 words: 3 per SM (start pair + split count), aligned to 2, then 2 per Q block (KV span)
    return align(3 * num_sms, 2) + 2 * num_q_blocks;
}
```

布局（与 `scheduler:32-34` 的 `get_mqa_logits_metadata_span_word_offset` 对应）：

```
[0 .. 2*num_sms)         : 每 SM 的起点 uint2 = (q_block_idx, split_offset)   // 8B 对齐
[2*num_sms .. 3*num_sms) : 每 SM 的 split 计数
align(3*num_sms, 2) 之后 : 每 Q block 的 MQALogitsKVSpan = (kv_token_base, num_kv_splits)  // 2 words
```

### 3.2 几何 `MQALogitsConfig`（`:117-202`）

- `umma_n = align(block_q * num_heads, 8)`（`:136`），因 `block_q*num_heads ≡ 128`，故 **UMMA_N = 128**。
- `num_q_stages = is_fp4 ? 2 : 3`、`num_kv_stages = is_fp4 ? 8 : 3`（`:137-138`）——FP4 半宽，流水更深。
- `smem_size()`（`:168-194`）**逐字节复刻** `layout::MQALogitsSharedStorage`（`layout/mqa_logits.cuh:16-65`），
  生成源码里再用 `static_assert(sizeof(...) == smem_size())`（`:254-256`）兜底。
- weights 行按 16B 补齐（`:156-158`，TMA 要求），单独造 tensormap（`:197-201`）。

## 4. 连续版调度器精读（`scheduler/sm100_mqa_logits.cuh`）

这是本篇的核心，也是 H100 实验直接跑的真实代码。

### 4.1 一个 Q block 的 KV 跨度 `get_mqa_logits_kv_span`（`:37-54`）

```cpp
uint32_t start = UINT_MAX, end = 0;
for (token in [0, BLOCK_Q)) {
    row = min(q_block*BLOCK_Q + token, num_q_tokens-1);      // 尾行 clamp 到最后有效行
    k_start = min(cu_seq_len_k_start[row], num_kv_tokens);
    k_end   = min(cu_seq_len_k_end[row],   num_kv_tokens);
    start = min(start, k_start); end = max(end, k_end);
}
kv_token_base = start / 4 * 4;                                // 向下 4 对齐
return { kv_token_base, ceil_div(end - kv_token_base, SPLIT_KV) };
```

- **`start / 4 * 4`**：KV scale 的 TMA 描述符要求 4 字节对齐（第 19 篇 §3.2 同一原因）。
- 一个 Q block 的**任务量 = 其 BLOCK_Q 行的有效区间并集**覆盖的 split 数。
- `MQALogitsKVSpan`（`:20-29`）就是 `(kv_token_base, num_kv_splits)`。

### 4.2 CTA 级前缀和 `mqa_logits_metadata_prefix_scan`（`:56-79`）

把 `values[0, num_items)` 做**就地 inclusive 前缀和**。两个细节：

- `num_items_per_thread = ceil_div(num_items, kNumThreads) | 1`（`:60`）强行取奇数，让每个线程拿到
  成对（even/odd）元素，循环里交错累加（`:65-70`），尾部单独处理（`:71-74`）——一种减少依赖链的写法。
- 线程内和用 `math::cta_exclusive_sum`（`:75`）跨 warp 归约，再把 `thread_offset` 加回本线程元素（`:76-77`）。

### 4.3 二分 `mqa_logits_metadata_upper_bound`（`:82-93`）

返回「第一个 `prefix[mid] > target`」的下标；用于把「目标工作量」映射回 Q block。

### 4.4 metadata kernel：代价模型与边界（`:96-162`）

单 CTA、`kNumThreads=256`。步骤：

1. **每个 Q block 的 span + 工作量**（`:122-128`）：
   ```cpp
   work_prefix[b] = span.num_kv_splits;
   cost_prefix[b] = kSplitCost*span.num_kv_splits + (span.num_kv_splits == 0 ? 0 : kSegmentCost);
   ```
2. **前缀和**（`:130-131`）得到 `total_work`、`total_cost`（`:132-133`）。
3. **按代价切分给 SM**（`:142-155`）：
   ```cpp
   cost_per_sm = total_cost / kNumSMs;
   cost_remainder = total_cost % kNumSMs;
   locate(i): target = i*cost_per_sm + min(i, cost_remainder);          // 前 cost_remainder 个 SM 多 1
              if target == total_cost: return (num_q_blocks, 0, total_work);   // 尾哨兵
              qb = upper_bound(cost_prefix, target);
              split_offset = min((max(target - cost_before, kSegmentCost) - kSegmentCost) / kSplitCost,
                                 work_prefix[qb] - work_before - 1);    // 落在 split 粒度
              return (qb, split_offset, work_before + split_offset);
   ```
   **关键点**：边界不落在 Q block 边界，而是落在 **split 边界**——`locate` 先定位到 Q block，
   再在块内减掉「启动代价 `kSegmentCost`」、按 `kSplitCost` 折算成 `split_offset`。
   于是同一 Q block 的 span 可以被相邻两个 SM **从中间切开**（实验 §6 的边界对拍验证了这点）。
4. **写回**（`:156-161`）：`uint2` 起点 + 计数写到 `schedule_meta`。

**代价模型为什么是 3:2**（`sm100_mqa_logits.hpp:11-12` 注释）：一个非空 Q block 有一次性启动开销
（载 Q/weights、建流水），被建模为 **1.5 个 split**（`kSegmentCost/kSplitCost = 3/2`）。
这解释了为什么空 span 的 Q block 代价为 0（不启动），而非空时 `cost = 2·splits + 3`。

### 4.5 `SM100MQALogitsScheduler`（`:164-271`）

两种模式，靠模板参数 `kUseSchedule` 切换：

- **scheduled**（`:216-239`）：从 `schedule_meta` 读本 SM 的 `(q_block, split_offset)` 起点与
  `remaining_splits`，顺序发射。一个 task 的 `num_kv_splits = min(span剩余, remaining)`（`:232`），
  发完递减 `remaining`；`remaining` 用尽即停。**恰好消费本 SM 被分配的 split 数**。
- **grid-stride**（`:240-269`）：`q_block_idx = sm_idx; += kNumSMs` 的轮转（`kUseSchedule=false`），
  即 SM90 的持久化方式，工作量取决于每个 Q block 的 span 大小——**这正是负载不均的来源**。

`MQALogitsTask`（`:12-18`）字段：`q_token_base / num_q_tokens / kv_token_base / num_kv_splits /
kv_shared_with_prev`。`kv_shared_with_prev` 只对 paged 有意义（连续版恒 false），
主 kernel 用它做 KV 复用（§5）。

### 4.6 分页版调度器（`scheduler/sm100_paged_mqa_logits.cuh`，对比）

分页版另有自己的 metadata kernel（`:50-159`），代价模型不同：
`get_split_cost(num_q_tokens) = 2*BLOCK_Q*ceil_div(num_q_tokens,BLOCK_Q) + num_q_tokens`（`:110-112`），
即「KV/MMA 代价 ∝ Q block 数，epilogue 代价 ∝ 有效 token 数」。它把 request 按 `indices` 分组
（`RequestInfo::from_q_token` `:30-39`），有 **cost 溢出回退**（`:114-132`，高位标志位 + 改用 token 代价），
每 SM 起点是 `(q_token_idx, kv_split_idx)`（`:147-158`）。`SM100PagedMQALogitsScheduler`（`:162-281`）
按「request → chunk(`kSplitsPerChunk`) → Q block」顺序发任务，并带一个**页坐标滑动缓存**
（`get_kv_page_coords` `:269-280`）。本篇实验不覆盖 paged（其 metadata 需要 `context_lens`/`indices`
语义），但结构与连续版同源。

## 5. SM100 主 kernel（`impls/sm100_mqa_logits.cuh`，仅分析）

### 5.1 五类 warp（与 Mega MoE 同构）

`kSpecWarpStart = kNumMathWarpGroups*4`（`:74`）；`SPLIT_KV=384` → `kNumMathWarpGroups=3`，故：

| warp | 行号 | 职责 |
|---|---|---|
| SF producer（`+0`） | `:278-280` | 发 SF-KV TMA + SF-Q；**也驱动 Q 流**（`produce_kv<true>`） |
| KV producer（`+1`） | `:281-283` | 发 KV TMA（`produce_kv<false>`） |
| SF 转置/UTCCP（`+2`） | `:284-364` | 把 `sf_q/sf_kv` 就地转成 4×32 布局并 UTCCP 进 TMEM |
| UMMA（`+3`） | `:365-429` | 单 warp 发 block-scaled UMMA |
| math warpgroups | `:430-533` | 读 TMEM、ReLU×权重归约、写 bf16 logits |

barrier 计数（`:110-128`）：`full_q.init(2)`（两个生产者）、`empty_q.init(kNumMathThreads+32)`、
`full_kv.init(2)`、`empty_kv.init(1)`、`full_tmem.init(1)`、`empty_tmem.init(128)`。
寄存器两档（`:140-146`）：specialized 56、math 由 CTA 入口池反推。

### 5.2 UMMA：A=KV、B=Q（`:371-421`）

- `UMMA_M = 128`（KV 行）、`UMMA_N = 128`（Q tile 的 token×head 列）、`UMMA_K = fp4?64:32`（`:90-92`）。
- `a_desc` 指向 `smem_kv`、`b_desc` 指向 `smem_q`（`:381-382`）——和 SM90 一样是 **KV 当 A、Q 当 B**，
  让「KV 行 × Q 列」的每头点积排成累加器列。
- SF 从 TMEM 取（`tmem_sfa`/`tmem_sfb`，`:400-419`），`make_runtime_instr_desc_with_sf_id`
  按 `gran_k` 拆多条指令（`:386-389`）。
- 每个 math warpgroup 吃一个 `UMMA_M=128` 的 KV tile（`tmem_stage_idx`），
  `kNumTmemStages == kNumMathWarpGroups`（`:99` 静态断言）。

### 5.3 math warpgroup：ReLU×权重归约（`:430-533`）

- 先按 `dispatch_num_block_tokens`（`:27-36`）把运行期「有效 token 数」编译期特化。
- weights 一次性读进寄存器（`:461-470`），`visit_tmem_head_chunks`（`:39-49`）按 16/8/4 宽读 TMEM。
- `reduce_pairs`（`:480-489`）用 `cvt_relu_bf16x2_f32` + `__hfma2` 做 **Σ_h relu(acc)·w 的 bf16x2 归约**，
  再 `__hadd2_rn` 折半，`st_global_low_bf16` 写出（`:473-479`）。
- 写地址含 `- seq_k_start[i]` 的**压缩偏移**（`:454-458`），只写落在 `[seq_k_start, seq_k_end)` 内的列（`:477`）。
- 读完某 KV split 的最后一块即 `release_tmem`（`:507-512`），让 UMMA 与归约重叠。

### 5.4 KV 复用 `reuse_kv`（`:150-153`）

`reuses_kv_stages(task) = task.kv_shared_with_prev and task.num_kv_splits == kNumKVStages`——
分页版里同一 chunk 的多个 Q block 共享 KV，若正好等于流水深度则**整段 KV 复用、SF 不重复转置**
（`:252-258`、`:335-342`）。连续版无此路径。

## 6. H100 实测：真实调度器设备代码逐条验证

实验 `src/mqa_logits_schedule.cu` + `harness/exp_mqa_logits_schedule.{sh,out.txt}` 在
H100(sm90, 132 SM) 上**直接编译运行** `scheduler/sm100_mqa_logits.cuh`（只需 cute/cutlass + PDL）。
每个用例做四件事：

1. 用真实 `sm100_mqa_logits_metadata` 生成 `schedule_meta`；
2. host 端**独立复刻**（span 公式 / 前缀和 / 代价模型 / `locate`）逐字段对拍；
3. 用真实 `SM100MQALogitsScheduler`（scheduled 与 grid-stride 两种模板实例）发射任务，
   验证**所有有效 split 恰好被覆盖一次**（无重复/无遗漏/无越界）；
4. 记录 per-SM 工作量，比较两种模式的均衡度。

| 用例（BQ/形状） | total_work | host 对拍 | scheduled 覆盖 | per-SM（sched） | per-SM（grid-stride） |
|---|---|---|---|---|---|
| BQ=2 uniform S=4096 | 22528 | spans/boundary 0 不符 | 22528/22528 dup0 miss0 | **[169..172]** | [165..176] |
| BQ=2 kstart%4≠0 | 22528 | 0 不符 | 22528/22528 | [169..172] | [165..176] |
| BQ=2 tail S=3970 | 22528 | 0 不符 | 22528/22528 | [169..172] | [165..176] |
| **BQ=2 skew+empty S=8192** | 24876 | 0 不符 | 24876/24876 | **[185..191]** | **[99..352]** |
| BQ=2 few-split skew S=700 | 2926 | 0 不符 | 2926/2926 | **[21..23]** | [17..32] |
| BQ=2 ragged NQ=4103 | 22572 | 0 不符 | 22572/22572 | [170..172] | [165..176] |
| BQ=4 uniform S=4096 | 11264 | 0 不符 | 11264/11264 | [84..87] | [77..88] |
| **BQ=4 skew+empty S=8192** | 19316 | 0 不符 | 19316/19316 | **[144..149]** | **[110..176]** |
| **8 SM skew S=4096** | 414 | 0 不符 | 414/414 | **[50..55]** | **[28..88]** |
| 8 SM tiny S=3000 | 160 | 0 不符 | 160/160 | [19..21] | [16..24] |

10 个用例 **全部 ALL PASS**。要点：

- **host 独立复刻与 device 完全一致**：`spans`（含 `start/4*4` 向下对齐、尾行 clamp）、
  每 SM 的 `(q_block, split_offset)` 起点与 split 计数**逐字段 0 不符**——坐实了 §4.1/§4.4 的算法描述。
- **覆盖性**：scheduled 与 grid-stride 两种模式下，真实 scheduler 发射的所有任务
  **恰好覆盖 `[0,total_work)` 每个 split 一次**（dup=0、miss=0、bad=0），
  说明「metadata 的边界 + scheduler 的任务切分」是一套自洽的划分。注意 scheduled 模式的任务数
  （如 2151）**大于 Q block 数**（2048）——因为**一个 Q block 会被相邻 SM 从中间切成两个 task**，
  这正是 §4.4 边界落在 split 粒度而非 Q block 边界的直接体现。
- **负载均衡收益**（均衡度 = per-SM 工作量 max−min）：
  - sk=8192 偏斜：scheduled **185..191（差 6）** vs grid-stride **99..352（差 253，3.5× 于均值）**；
  - 8 SM：scheduled **50..55（差 5）** vs grid-stride **28..88（差 60）**。
  即 grid-stride 的「整块 round-robin」在块大小不均时会严重倾斜，而按代价均衡后每个 SM 只差个位数 split。
- **缓冲区字数公式**：`T=4096` → `align(3·132,2)+2·2048 = 396+4096 = 4492` words（17968 B）；
  `T=4103`（非 block_q 整数倍）→ 4500 words。与 `:64-69` 公式一致。
- **边界用例**：`kstart%4≠0`（round-down）、尾部非 384 整数倍（末 split 裁剪）、
  `num_q_tokens` 非 block_q 整数倍（尾行 clamp）、空 span（`S=8192` 用例里 `i%7==0` 的行）、
  以及 `kNumSMs=8` 的小网格都通过。

## 7. 为什么这样设计：三种调度对比

| 方案 | 工作量分布 | 均衡度（skew S=8192） | 额外开销 |
|---|---|---|---|
| 串行（单 SM） | 全量 | — | 慢 |
| **grid-stride**（SM90 连续版） | `sm_idx + k·kNumSMs` 整块轮转 | [99..352] | 无（但倾斜） |
| **cost-balanced**（SM100 连续版） | 按 `2·splits+3` 代价二分到 split 边界 | **[185..191]** | 一次 metadata kernel launch |

变长上下文下，不同 query 的有效 KV 长度差异巨大（有的 token 看 8K、有的只看 64），
「按 Q block 整除分配」会让某些 SM 拿到一连串长块、另一些拿短块。SM100 的做法：
**先用一个极轻量 kernel（单 CTA、256 线程）把代价前缀和算出来，按 `total_cost/kNumSMs` 二分切分，
边界落在 split 粒度**——代价是 launch 开销，收益是几乎完美均衡。这与 sparse logits
（第 22 篇 §3.4 的 `balance_wave_entries` 直方图排序 + 旋转）是同一思想的两种实现。

## 8. 结论与边界

- SM100 dense MQA logits = **SM90 dense 的 MXFP4/MXFP8 + bf16 升级**，主 kernel 换 block-scaled UMMA，
  并新增**连续版 metadata 负载均衡调度器**。
- 该调度器（`sm100_mqa_logits_metadata` + `SM100MQALogitsScheduler`）**与 UMMA 无关**，
  在 H100 上编译运行真实设备代码，10 个用例、数万 split 的覆盖/均衡/对拍全部通过。
- **主 kernel（UMMA/TMEM/UTCCP 那部分）仍是 SM100 专属，仅分析**——与 `fp8_fp4_mega_moe`、
  Mega Gate/mHC、sparse logits 的边界一致。真正评测该 kernel 需要 SM100。

## 9. 参考资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM （公开版 commit `057ca59`）。
- 本专题第 19 篇（dense MQA logits，SM90 可跑）：`docs/19-attention-mqa-logits.md`。
- 本专题第 22 篇（稀疏 MQA logits，SM100 仅分析 + H100 语义实测）：`docs/22-sparse-mqa-logits.md`。
- 本专题第 06 篇（总览）：`docs/06-summary.md`。
