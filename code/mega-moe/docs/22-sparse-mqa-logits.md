# DeepGEMM 稀疏 MQA logits：稀疏注意力 indexer 的第二级打分（SM100 专属，仅分析 + H100 语义实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P12 增量（第 22 篇）。
> 第 19 篇拆了稀疏注意力 indexer 的 **dense MQA logits**（`fp8_fp4_mqa_logits` 连续 /
> `fp8_fp4_paged_mqa_logits` 分页，**SM90 可跑**）。本篇补上它的 **稀疏版**：
> `fp8_fp4_sparse_mqa_logits` / `fp8_fp4_paged_sparse_mqa_logits`——indexer 先用粗分选出
> 每 query 的 top-k 个 KV 块（`sparse_kv_block_indices`），算子在**被选中的稀疏块内**算细粒度
> logits，并把结果**压缩**存进 `[T, num_max_sparse_blocks * sparse_block_kv]`。
> 主 kernel 走 block-scaled UMMA（MXFP8/MXFP4），**SM100 专属，H100 跑不了**；
> 但它的**选择/合并/压缩列映射语义是架构无关的**，本篇在 H100 上用真实 dense SM90 kernel
> 取全量 logits 后逐条验证（`harness/exp_sparse_mqa_logits.*`）。
>
> 为什么与 Mega MoE 专题相关：Mega MoE 的关注点是「单 kernel 融合通信+计算」；而稀疏注意力
> indexer 展示的是 DeepGEMM 的另一类融合——**把「块选择 → 稀疏加载 → 打分 → 压缩输出」压进一个
> persistent kernel**，并用一个独立的 metadata kernel 做**负载均衡调度**。两种融合共同体现了
> DeepGEMM「先跑一个轻量调度/元数据 kernel，再跑主 kernel」的固定套路（与 `get_mqa_logits_metadata`、
> `get_paged_mqa_logits_metadata` 一脉相承）。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。
> 主实现（SM100）：`csrc/apis/attention.hpp:284-442`、`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp`、
> `deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh`、
> `deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh`、
> `deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh`。
> H100 语义实验：`harness/exp_sparse_mqa_logits.{py,sh,out.txt}`。

## 1. dense 与 sparse 的关系：一份 logits，两级粒度

第 19 篇的 dense MQA logits 给出每个 query token 对**所有** KV token 的相关分：

```
logits[m, n] = Σ_h relu(<q[m,h,:], kv[n,:]>) · w[m,h] · kv_scale[n]
```

对长上下文（`S=64K~128K`）逐个 KV token 都算一遍代价很高。DeepSeek 稀疏注意力的做法是先由
**indexer**（一个轻量网络）选出每 query 最相关的若干 **KV 块**（块大小 `sparse_block_kv ∈ {8,16}`），
再只在这些块内算细粒度 logits、做 top-k、进入真正的稀疏注意力。DeepGEMM 的 sparse kernel 就是
这一层的算子：它**不负责选块**（块选择由上游 indexer/调用方给出 `sparse_kv_block_indices`），
只负责「在被选块内打分并压缩输出」。

输入契约（`csrc/apis/attention.hpp:303-312` 注释）：

- `sparse_kv_block_indices`：`[num_q_tokens, num_max_sparse_blocks]` 的 int32，**每行按块号升序、唯一**；
  一行的**前缀**是「由该行 KV 长度推出的有效块」（见 §3.1）。
- 输出：`[num_q_tokens, num_max_sparse_blocks * sparse_block_kv]` 的 **BF16** 压缩 logits
  （`attention.hpp:396-399`），槽 `s`、块内偏移 `i` 落在列 `s*sparse_block_kv + i`。
  即**把 dense logits 在被选位置的取值按块压缩**：第 `s` 个被选块的 `SB` 个 token 分数占 `SB` 列。

与 dense 版的差异一览：

| 项 | dense（第 19 篇，SM90 可跑） | sparse（本篇，SM100 仅分析） |
|---|---|---|
| 计算范围 | 全量 `[T,S]`（按有效区间） | 仅被选稀疏块 |
| 输出 | fp32 全量（压缩到有效区间） | **bf16**，压缩到 `[T, NB*SB]` |
| dtype | E4M3 + per-token fp32 KV scale | MXFP4/MXFP8 + packed UE8M0 MX scale |
| 调度 | round-robin / `schedule_meta` | 专用 metadata kernel（合并去重 + 负载均衡） |
| 架构 | sm90 + sm100 | **仅 sm100**（`attention.hpp:313/380/414`） |

## 2. host API：四个新入口

`csrc/apis/attention.hpp` 注册（`:445-476`）了四个 sparse 专属 + 两个 metadata 入口：

| 入口 | 行号 | 作用 |
|---|---|---|
| `get_sparse_mqa_logits_metadata` | `:303-332` | 连续版：建 metadata |
| `get_paged_sparse_mqa_logits_metadata` | `:334-367` | 分页版：建 metadata |
| `fp8_fp4_sparse_mqa_logits` | `:369-403` | 连续版主 kernel |
| `fp8_fp4_paged_sparse_mqa_logits` | `:405-442` | 分页版主 kernel |

契约（全部在函数开头 `DG_HOST_ASSERT`，**均在 sm90 因 `arch_major == 10` 断言直接失败**）：

- 连续 `:380` / 分页 `:414`：`arch_major == 10 且 q_sf_optional.has_value()`——即 **必须带 MX scale**
  （Q 是 MXFP4/MXFP8）。
- `num_max_sparse_blocks % 4 == 0` 且 `<= 4096`（`:381`/`:415`）；`sparse_block_kv ∈ {8,16}`（`:382`/`:416`）。
- `num_heads % 4 == 0`、`num_heads <= kNumMaxHeads=32`、`head_dim == kHeadDim=128`（`:387-388`）。
- 连续版 KV 是 `[num_kv_tokens, head_dim]`（`:391`）；分页版 KV 是**融合缓存**
  `[num_kv_pages, page_kv, 1, head_dim_with_sf]`，其中 `head_dim_with_sf = head_dim/pack + sizeof(int)`
  （FP8 时 `+4`，FP4 时 `head_dim/2+4`），`page_kv % sparse_block_kv == 0`、页基址 16 字节对齐
  （`:426-433`）。
- paged 不支持 `use_unaligned_ks`（`:413` 注释 + jit hpp `:51`/`:152` 断言）。

## 3. metadata：选择、合并、压缩的内部布局

metadata 是**主 kernel 的调度与索引表**，由独立的小 kernel 生成。布局定义在
`layout/sparse_mqa_logits.cuh`：

```cpp
MetadataHeader { num_kv_splits; num_waves; use_unaligned_ks; num_sms; }   // :23-29，16B
KVSplitHeader  { q_token_base; packed_num_kv_blocks; q0_slot_base; q1_slot_base; }  // :31-61
KVBlockInfo    { physical_kv_block_idx; packed_slot_offsets; }             // :63-72
ScheduleEntry  { kv_split_begin; kv_split_end; q_token_base; num_q_tokens; }  // :74-84
```

- **`kBlockQ = 2`**（`:17`）：metadata 与主 kernel 都按「两个 query token 一个块」处理。
  这是 sparse 版的关键——**相邻两个 token 的稀疏块先合并**，共享一次 KV 加载与一次 MMA。
- `packed_num_kv_blocks` 的高位塞了两个标志（`:31-36`）：`kContiguousFlag=0x80000000`（整个
  split 的 KV 物理连续，可走 TMA）与 `kPartialTailFlag=0x40000000`（split 尾块越过 KV 张量末端，
  需要裁剪，仅连续版）。
- `KVBlockInfo.packed_slot_offsets` 用 **16+16 位**分别存 q0/q1 的「split 内槽偏移」（`:63-72`），
  `kInvalidSparseSlot = 0xFFFF` 表示该 Q 不含此块。
- `WorkspaceState`（`:141-146`）是 metadata kernel 的全局计数区，三个计数器各占一条 128B L2 线
  （`num_kv_splits` / `next_q_offset` / `num_finished_ctas`），避免伪共享。

metadata 的**尺寸公式**在 jit wrapper `csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:20-36`
（本篇实验 §7A 逐项复刻）：

```
kNumKVBlocksPerSplit = SPLIT_KV / SB
num_kv_split_bytes    = sizeof(KVSplitHeader) + kNumKVBlocksPerSplit * sizeof(KVBlockInfo)
num_max_kv_splits     = paged ? T * ceil_div(NB, kNumKVBlocksPerSplit)
                              : ceil_div(T,2) * ceil_div(2*NB, kNumKVBlocksPerSplit)
num_max_schedule      = align(num_max_kv_splits, num_sms)      // schedule 按 [wave][num_sms] 排布
bytes                 = 16 + num_max_kv_splits*num_kv_split_bytes + num_max_schedule*16
```

其中 `SPLIT_KV` 由 dtype 决定（`sm100_sparse_mqa_logits.hpp:15-18`）：**FP8 = 512、FP4 = 640**；
`num_math_warpgroups = SPLIT_KV / 128`（`:48`）即 **4 / 5 个 math warpgroup**。

### 3.1 有效块前缀（`num_kv_blocks`）

metadata kernel 对每个 query token 先算有效块数（`scheduler/...:222-229`）：

```cpp
const uint32_t kv_begin = min(cu_seq_len_k_start[q], num_kv_tokens);
const uint32_t kv_end   = max(kv_begin, min(cu_seq_len_k_end[q], num_kv_tokens));
return min(NUM_MAX_SPARSE_BLOCKS, ceil_div(kv_end - kv_begin, SPARSE_BLOCK_KV));
```

所以 `sparse_kv_block_indices[q]` 只有**前 `num_kv_blocks` 项**被读，其余是 padding。连续版还断言
「若该 token 的 KK 段非空，则 `kv_begin % SB == 0`」（`:227`）——即**对齐情形下块起点必须落在 SB 边界**。

### 3.2 两 token 合并去重（metadata kernel 的核心）

对每个 Q 块（`q0`、`q1` 两个 token），kernel（`scheduler/...:273-326`）把两条**升序**块号列表
**归并成并集**，并给每个并集元素记录「在 q0 里的槽号 / 在 q1 里的槽号 / 是否出现」，打包进
`packed_slots`（`:300-325`）。归并本身用了一个并行二分求每个线程的分区起点（`:279-287`），
并在分区边界丢掉跨界的重复项（`:294-298`）。合并的收益：**两个 token 共享的 KV 块只加载/MMA 一次**，
主 kernel 里两个 token 从同一份 KV 结果里各取各的行。

**压缩列映射**：每个 token 的输出列 = **它自己**列表里的槽号 `s` × SB + 块内偏移 `i`
（`impls/...:627-633`：`output_col_idx = (q_slot_base + q_slot_offset)*SB + token_in_kv_block`，
而 `q_slot_base + q_slot_offset` 恰好还原成该 token 的全局槽号）。所以「合并」只影响 KV 读取/计算，
**输出仍是每 token 各自的压缩布局**。

### 3.3 物理块号：连续 / 非对齐 / 分页三种（`scheduler/...:377-386`）

```cpp
if constexpr (kIsPaged)         physical = block_table[q_token_base][logical/kNumKVBlocksPerPage]*kNumKVBlocksPerPage
                                           + logical % kNumKVBlocksPerPage;      // 查 block table
else if constexpr (kUseUnalignedKs) physical = logical;      // logical 已含 SB*blk + ks%SB
else                            physical = logical * SPARSE_BLOCK_KV;  // 对齐：绝对 token 起点
```

- **对齐**：逻辑块 `b` 对应绝对 token `[b*SB, (b+1)*SB)`。
- **非对齐（`use_unaligned_ks`）**：逻辑块 `b` 对应 `[b*SB + ks%SB, (b+1)*SB + ks%SB)`——块边界被
  KV 起点偏移推开，会**跨过对齐语义下的块边界**（实验 §7C 打印三个块号验证）。
- **分页**：逻辑块先经 `block_table` 映射到物理页，`page_kv % SB == 0` 保证稀疏块不跨页。

### 3.4 负载均衡调度（`scheduler/...:37-161`、`:443-472`）

所有 CTA 产完各自 Q 块的 KV split 后，**最后一个 CTA**（`num_finished_ctas` 计数到 `gridDim.x`，
`:443-451`）统一建调度表：

- 连续版 `build_contiguous_schedule`（`:37-73`）：先把 `total_kv_splits` 按 SM 数**均分**（
  `ceil_div(total*sm, kNumSMs)`，`:50-53`），再把每个 SM 的区间**按 Q 块边界切开**（`:54-62`），
  得到 `[wave][num_sms]` 的 `ScheduleEntry`；`num_waves` 取各 SM 条目数的最大值。
- 分页版 `build_paged_schedule`（`:124-161`）：先按 `kNumKVSplitsPerEntry=8` 把每个 query 的
  split 切成有界条目，再调 `balance_wave_entries`。
- `balance_wave_entries`（`:75-122`）：对每一 wave，按「条目包含的 split 数」做**直方图排序**，
  再把重条目**旋转**分配到不同 SM（`:114-120`）；两 wave 时把第二 wave 的顺序**反向**，让活跃
  SM 集合互补（`:117`）。这正是「负载均衡」——indexer 选出的块数在 token 间不均匀，必须打散。

主 kernel 侧读调度表：`schedule_entries[wave_idx * kNumSMs + blockIdx.x]`（`impls/...:275-276`），
并断言 `header->num_sms == kNumSMs`、`header->use_unaligned_ks == kUseUnalignedKs`（`:269-271`）——
**metadata 与主 kernel 的 SM 数必须一致，否则重跑 metadata**。

## 4. 主 kernel 精读（`sm100_sparse_mqa_logits.cuh`）

### 4.1 模板与资源（`:146-199`）

`UMMA_M = 128`、`UMMA_N = BLOCK_Q*kNumHeads`、`UMMA_K = kIsFP4?64:32`、`SPLIT_KV = kNumMathWarpGroups*128`
（`:160-166`）。寄存器分四档再平衡（`:173-177`）：control 64 / KV-copy 64 / math
`min(120, ...)`，每档之和必须落在 CTA entry pool 内（`:197-199` 断言）。

### 4.2 五类 warp（与 Mega MoE 同构）

| 角色 | 行号 | 职责 |
|---|---|---|
| Q+metadata producer | `:263-322` | 发 Q/weights/SF-Q TMA；用 `cp.async` 搬每个 split 的 `KVSplitHeader`+`KVBlockInfo`；末尾发空 sentinel |
| SF transpose | `:323-369` | 把 `sf_q`/`sf_kv` 就地转置成 UTCCP 需要的 4×32 布局并搬进 TMEM |
| UMMA | `:370-415` | 单 warp 发 block-scaled UMMA（leader），commit multicast |
| KV copy | `:416-550` | 按 split 把稀疏 KV 块 + per-token scale 搬进 smem |
| math warpgroups | `:551-641` | 读 TMEM、ReLU×权重归约、写压缩 logits |

barrier 初始计数（`:222-240`）体现分工：`empty_q.init(kNumMathThreads+64)`、
`full_metadata.init(32)`（一个 KV-copy warp）、`full_kv.init(kNumKVCopyThreads+1)`
（+1 是 SF-transpose warp 的 UTCCP 完成）、`empty_kv.init(kNumMathThreads+1)`。

### 4.3 KV 访问的两条路径

`ContiguousSparseKVAccessor`（`:34-99`）与 `PagedSparseKVAccessor`（`:101-142`）把「逻辑块号 →
KV/SF 基址」抽象出来。连续版对**整段物理连续**的 split 走 **TMA 快路**（`:436-457`）：只由
copy warp 0 的一个 elected lane 用 `copy_contiguous_kv_split`（`:59-78`）发两组 TMA（先 SF 后 KV，
让 SF 的转置与 KV 搬运重叠），**其余所有 copy warp 一个 TMA 都不发**，只用 `arrive_count_pred`
释放 barrier。非连续/非对齐/越界尾块走**通用 `cp.async` 路**（`:506-549`），
并在 `kNeedsTailMask` 时对越界 token 用 `cp_async_*_zfill`（源长度传 0）零填充（`:461-505`）。
这种「contiguous 走 TMA、其余走 cp.async」的双路径，是 sparse 里 block 物理地址不保证连续的直接后果。

### 4.4 MMA 与压缩输出

UMMA 段（`:392-410`）对每个 math warpgroup 的 128-token tile 发
`kHeadDim/UMMA_K` 条 block-scaled fma，SF 从 TMEM 取（`kTmemStartColOfSFKV + ... + m_idx*4`
与 `kTmemStartColOfSFQ + ...`）。math warpgroup（`:600-625`）从 TMEM 读回每 token 的
`kNumHeads` 个点积，用 `ptx::cvt_relu_bf16x2_f32` + `__hfma2` 做 **`Σ_h relu(·)·w` 的
bf16x2 归约**（与 dense 版的「ReLU→加权求和」一致），最后 `__hadd_rn` 得到标量 bf16，
按 §3.2 的列映射写进 `logits`（`:627-633`）。`tmem_load_32dp32b` 按 `kNumHeadsPerLoad=gcd(H,16)`
整块读，避免读到最后一 token 之外的 TMEM 列（`:163-164`）。

## 5. 与 dense / Mega MoE 的关系

- **与 dense MQA logits**：dense 算全量、fp32；sparse 只算被选块、输出 bf16 压缩。两者的
  「ReLU→按头加权求和」完全一致，只是 sparse 多了一层「块选择/合并/压缩列映射」。
- **与 Mega MoE**：都是「**一个 metadata/调度 kernel + 一个 persistent 主 kernel**」的两段式，
  且主 kernel 都是 block-scaled UMMA + TMEM + UTCCP SF 通路。区别在 Mega MoE 的主 kernel 融合了
  **NVLink 通信**（dispatch/combine），而 sparse logits 融合的是 **indexer 的块选择与稀疏加载**。

## 6. 为什么 sparse 更快（量级）

设每 token 选 `NB` 个块、块大小 `SB`，则 sparse 的细粒度计算量约为 `T·NB·SB` 个位置，
而 dense 是 `Σ_t (ke_t - ks_t) ≈ T·S`。实验 §7B 里 `S=4096, NB=16, SB=8` 时压缩比约 **28×**；
上下文越长、`NB` 相对 `S/SB` 越小，收益越大。代价是：① metadata kernel 的一次额外 launch；
② 块号在 token 间不均匀，必须靠 §3.4 的负载均衡；③ KV 物理不连续时退到 `cp.async`（非 TMA）路径。

## 7. H100 实测（架构无关语义；主 kernel SM100 仅分析）

实验 `harness/exp_sparse_mqa_logits.{py,sh,out.txt}` 在 H100(sm90, 132 SM) 上跑。**主 kernel
跑不了**，故：
(A) 复刻 metadata 尺寸公式；(B) 用真实 dense SM90 kernel 取全量 logits，再验证稀疏选择+
压缩列映射+合并去重；(C) 验证 unaligned ks 偏移；(D) 验证四个 host 入口在 sm90 全部被拒。

| 检查 | 结果 |
|---|---|
| **A. metadata 尺寸公式** | 与 struct 尺寸（16/16/8/16/384/8B）一致；例 `T=4096,NB=64,SB=8,fp8,连续` → 2,230,288 B（544 B/token），`SB=16` 因 split 内块数减半、split 数翻倍 → 2,361,296 B；FP4 `SPLIT_KV=640` 更大 |
| **B. 压缩列映射** | `gather==L` **逐位 True**（压缩输出就是 dense logits 在被选位置的 gather）；`relerr(压缩 vs fp32 参考)` **3.5e-2~4.1e-2 = dense kernel 自身的量化噪声层**，说明选择/映射无额外误差 |
| **B. 合并去重** | 每 Q 块两 token 的块并集去重省下 128/1024、256/2048、1017/8192 个重复块（约 12.5%） |
| **B. 计算压缩比** | 有效 dense 位置 / sparse 位置 = **28.0× / 7.0× / 14.0×**（随 NB、SB 变化） |
| **C. unaligned ks** | `off=ks%SB=5`：逻辑块 3 覆盖绝对 `[29,37)`（对齐语义是 `[24,32)`），证明确实按 `b*SB+off` 平移、跨原块边界 |
| **D. host 契约** | `get_sparse_mqa_logits_metadata`(`:313`)、`fp8_fp4_sparse_mqa_logits`(`:380`)、`get_paged_sparse_mqa_logits_metadata`(`:344`) 在 sm90 全部因 `arch_major == 10` 断言被拒 |

结论：sparse 主 kernel 的**选择/合并/压缩语义可被架构无关地复现并验证**，且它相对 dense 的
收益（块压缩比、去重）可量化；但**真正的 kernel 增益（TMA 快路、负载均衡、block-scaled UMMA）
需要 SM100 才能评测**——这与内存里 `fp8_fp4_mega_moe`、Mega Gate/mHC 的边界一致。

## 8. 参考资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM （公开版 commit `057ca59`，v2.8.1）。
- DeepSeek-V3.2 / 稀疏注意力（DSA）indexer 的公开说明：见 DeepSeek-V3.2 技术报告与 DeepGEMM release notes。
- 本专题第 19 篇（dense MQA logits，SM90 可跑）：`docs/19-attention-mqa-logits.md`。
- 本专题第 06 篇（总览）：`docs/06-summary.md`。
