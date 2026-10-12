# 30 · SM90 paged MQA logits 的 metadata 负载均衡调度器

> 本专题第 19 篇拆了 SM90 dense MQA logits 的**连续版 + 分页版主 kernel**，第 22 篇拆了 SM100 sparse，
> 第 23/24 篇拆了 SM100 dense 的连续版 / 分页版**调度器**。唯独 **SM90 分页版自己的调度器**
> （`scheduler/sm90_paged_mqa_logits.cuh`：`sm90_paged_mqa_logits_metadata` + `SM90PagedMQALogitsScheduler`）
> 只在第 19 篇被一笔带过（「按 (q_idx, kv_split_idx) 分配任务」）。本篇补上这个缺口。
>
> 关键点：**这个调度器与 WGMMA/TMA/UMMA 完全无关**，只依赖 `math.cuh` / `ptx/utils.cuh` 与 PDL，
> 因此可以在 H100（sm90）上**直接编译运行真实设备代码**逐条验证（本文所有数字都是真机实测）。
> 参考仓库 `~/github/DeepGEMM`，核对 commit `057ca59`（Public release 26/09/30, PR #462）。

---

## 1. host 入口与实例化契约

分页入口 `fp8_fp4_paged_mqa_logits` 打印出的调度元数据由 `get_paged_mqa_logits_metadata` 生成
（`csrc/apis/attention.hpp:136-172`）：

- `:139-146` 契约：只支持 **2D context lens**（`context_lens.dim()==2`，`[batch, next_n]`，
  故 `is_context_lens_2d=true` 钉死 `:140`）、`num_sms>0`、int32、contiguous。
- `:149` 输出 `schedule_metadata` 形状 **`[num_sms + 1, 2]`**（每行一个 `(q_atom, kv_split)`，
  多一行是**终止哨兵**）。
- `:152-169` 架构分派：SM100 走 varlen（`:153-162`），SM90 走**非 varlen**（`:163-166`）：
  `DG_HOST_ASSERT(not is_varlen)`（`:164`）、`DG_HOST_ASSERT(block_kv == 64)`（`:165`），
  然后调 `sm90_paged_mqa_logits_metadata(...)`。

`sm90_paged_mqa_logits_metadata` 的 host 壳在 `csrc/jit_kernels/impls/sm90_fp8_mqa_logits.hpp:94-135`，
只在这里写死了几个常量：

- `:100` `constexpr int split_kv = 256;`——**调度 work 单位 = 256 个 KV token**。
- `:101` `constexpr int num_threads = 32;`——metadata kernel 只有**一个 warp**。
- `:102` `const int aligned_batch_size = align(batch_size, 32);`——编译期按 batch 对齐到 32。
- `:105` `num_smem_ints = is_varlen ? 3*aligned_batch_size+1 : aligned_batch_size`，
  `:106` smem = 该数 × 4B；`:107` 断言不超 smem 容量。
- `:110-120` JIT 实例化 `sched::sm90_paged_mqa_logits_metadata<aligned_batch_size, split_kv, num_sms, is_varlen>`。
- `:122-135` launch：`grid=(1,1,1)`、`block=(32,1,1)`、传 `(batch_size, next_n, is_context_lens_2d,
  context_lens, indices, schedule_metadata)`。

那**谁消费**这份元数据？SM90 分页主 kernel 在
`deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh:137-140` 构造调度器：

```
auto scheduler = sched::SM90PagedMQALogitsScheduler<
    kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumMathWarpGroups, 1>(
        blockIdx.x, batch_size, context_lens, schedule_meta, indices);
```

其中 `kNumMathWarpGroups = split_kv / mma_m`（host `sm90_fp8_mqa_logits.hpp:155-156`，`mma_m=64`、`split_kv=256`
→ **4**），也就是调度器的 `kNumBlocksPerSplit=4`；最后一个模板参数 `kNumNextNAtoms` 固定传 **1**。
`DG_STATIC_ASSERT(SPLIT_KV % BLOCK_KV == 0)`（`:140`）保证 `256 % 64 == 0`。
所以 **1 个 split = 4 个 KV block = 256 个 KV token**，调度器以 split 为粒度把 KV 段分给 SM，
主 kernel 内 4 个 math warpgroup 各领 1 个 block（`kv_idx + kv_group_idx`，kernel `:195`）。

> 与 SM100 dense 分页版（第 24 篇）的最大区别：SM100 的 work 单位是「varlen 请求内的 Q block × KV split」，
> 代价模型 `2·BLOCK_Q·blocks + tokens`；SM90 的 work 单位是纯 **KV split 段数**，代价模型就是
> 「段数」本身。两种设计各自的合理性见 §4。

---

## 2. metadata kernel 逐段精读（`scheduler/sm90_paged_mqa_logits.cuh:9-127`）

模板 `:9-12`：`<kAlignedBatchSize, SPLIT_KV, kNumSMs, kIsVarlen>`，只有一个 warp；
`:13` 断言 `kAlignedBatchSize % 32 == 0`。`:17` 先 `cudaGridDependencySynchronize()`（PDL），
让 metadata kernel 与前面生成 context lens / indices 的 kernel 重叠。

smem 布局 `:19-23`：非 varlen 只用 `prefix_sum[aligned_batch_size]`；
varlen 额外用 `varlen_atom_token_start` / `varlen_atom_context_len` / `varlen_num_atoms_shared`
三块（对应 host `:105` 的 `3*aligned+1`）。

### 2.1 varlen：先做「atom 分组」（`:26-45`）

varlen 下 `indices[t]` 给出 token 属于哪个请求。**相邻两个 token 若 `indices` 相同则合成一个 atom**
（`:34` `is_paired = t+1<batch_size and indices[t]==indices[t+1]`），atom 的 context 取**配对的后一个 token**
（`:35` `is_paired ? context_lens[t+1] : context_lens[t]`），然后 `t += is_paired?2:1`。
atom 起点记在 `varlen_atom_token_start`、context 记在 `varlen_atom_context_len`，总数写 `varlen_num_atoms_shared`。
`:41-45` 一个 `__syncwarp` 后 `num_items = atom_count`。

> 为什么要配对：DeepSeek 稀疏注意力里同一个请求的相邻 pair 共享一段 KV（block_table 同一行），
> 配对后 KV 只需扫描一次。这也和 SM100 分页 varlen 的「run-length 请求」是同一思想的两种写法
> （SM100 是 `indices[t]==indices[t+1]` 的 run-length，第 24 篇 §2）。
> 主 kernel scheduler 里 `refresh_num_kv_and_advance` 的 `current_advance = is_paired?2:1`（`:183`）
> 正对应这里的 2-token atom。

### 2.2 warp 内前缀和（`:47-69`）

对每个 atom 算 `x = ceil_div(context_len, SPLIT_KV)`（`:60`，`SPLIT_KV=256`），即该 atom 占多少 split。
`:62-65` 用 `__shfl_up_sync(0xffffffff, x, offset)`、`offset ∈ {1,2,4,8,16}` 做 warp 内 inclusive scan；
`:66` `x += sum` 把上一批 32 个的累计值接上；`:67` 存 `prefix_sum[i]`；`:68` `sum = shfl(x, 31)` 作为下一批的种子。
循环 `kAlignedBatchSize/32` 次（`:50`），超出 batch 的位置 context=0（`:55`/`:58` 的 `q_idx < num_items/batch_size` 守卫），
所以 `prefix_sum` 在尾部是**平坦的累计值**——这点在 §3 的二分和空 SM 处理里很关键。

非 varlen 的 `lens_idx` 是 `q_idx*next_n + next_n-1`（`:57`），即 **2D context lens 取该行最后一个 next_n 位置**。
`is_context_lens_2d` 为真时一定是这个取值方式（host 钉死 `is_context_lens_2d=true`）。

### 2.3 工作分配：反序均分 + 二分定位

**varlen（`:72-95`）**：

- `:74-75` `total = sum`（总 split 数）、`q = total/kNumSMs`、`r = total%kNumSMs`。
- `:76-78` **反序分配**：`pivot = kNumSMs - r`；前 `kNumSMs-r` 个 SM 分 `q` 段，后 `r` 个 SM 分 `q+1` 段。
  注释明说：当 `total < kNumSMs` 时，空 SM 会落到 `atom_idx==0`，从而让
  `refresh_num_kv_and_advance` 的取址**仍在界内**。
- `:79-95` 每个 SM 的起始段号 `seg_starts = sm_idx*q + (sm_idx>pivot ? sm_idx-pivot : 0)`（`:80`），
  在 `prefix_sum[0..num_items)` 上二分找**第一个 `prefix_sum[mid] > seg_starts`** 的 atom（`:81-87`，上界二分）。
- `:88-90` `atom_idx = lo`；`kv_split_idx = atom_idx==0 ? seg_starts : seg_starts - prefix_sum[atom_idx-1]`；
  `q_atom_idx = atom_idx<num_items ? varlen_atom_token_start[atom_idx] : batch_size`。
- `:93-94` 写 `schedule_metadata[sm_idx*2] = q_atom_idx`、`[sm_idx*2+1] = kv_split_idx`。
  varlen 的循环上界是 `sm_idx <= kNumSMs`（`:79`），**终止哨兵在这一趟里顺便写掉**。

**非 varlen（`:96-126`）**：

- `:97` `next_n_atom = (next_n>=2) ? 2 : 1`；`:98` `num_next_n_atoms = ceil_div(next_n, next_n_atom)`。
  由于 host 只允许 `next_n ∈ {1,2}`（host 端 `DG_HOST_ASSERT(next_n==1 or next_n==2)`，见
  `sm90_fp8_mqa_logits.hpp:192`），`num_next_n_atoms` 恒为 **1**——所以 `kNumNextNAtoms=1` 是自洽的。
- `:99` `total = sum * num_next_n_atoms`；`:100-101` 同样的 `q/r/pivot` 反序分配。
- `:102-121` 循环上界是 `sm_idx < kNumSMs`（`<`，不含终止行）。二分谓词变成
  `prefix_sum[mid] * num_next_n_atoms <= seg_starts`（`:107`）。
- `:111-116` 由 `q_idx` 反解 `offset_in_q`、`num_segs_q`、`atom_idx = offset_in_q/num_segs_q`（整数除）、
  `kv_split_idx = offset_in_q % num_segs_q`；`q_atom_idx = q_idx*num_next_n_atoms + atom_idx`。
  当 `num_next_n_atoms=1` 时 `atom_idx` 恒为 0（`offset_in_q < num_segs_q`），所以 **`q_atom_idx = q_idx`**：
  主 kernel 再经 `atom_to_token_idx` 乘 `next_n_atom` 展开成 token。
- `:122-125` `lane 0` 单独写终止哨兵 `schedule_metadata[kNumSMs*2] = batch_size*num_next_n_atoms`、`[...+1]=0`。

> 「`q_atom_idx = q_idx`」这一步就是第 19 篇没展开的关键：metadata 分的是 **q_idx（batch 行）**，
> 而不是 token。token 的展开延迟到 scheduler 的 `atom_to_token_idx`（§3.1），
> 从而让 `next_n` 只在一处出现。

---

## 3. scheduler 逐段精读（`:129-246`）

### 3.1 条件存储（EBO）与静态映射

`:130-136` 用模板特化 `SM90IndicesStorage<kIsVarlen>` 做空基类优化：只有 varlen 才真的存 `indices` 指针，
非 varlen 下这是**零开销**的空结构。

`:151-163` `atom_to_token_idx`：

- varlen 直接返回 `q_atom_idx`（因为 metadata 存的已是 token 起点）；
- 非 varlen：`kPadOddN = (not varlen) and (next_n%2==1) and (next_n>=3)`，`kNextNAtom = (varlen or next_n>=2) ? 2 : 1`。
  对 `next_n ∈ {1,2}` 且非 varlen，`kNumNextNAtoms=1`、`kPadOddN=false`，返回 `q_atom_idx * kNextNAtom`
  （`next_n=2` 时 token 起点 = `q_atom_idx*2`）。
  `kPadOddN` 分支是给 `next_n≥3` 的奇数准备的（当前 host 不支持），把「每 atom 2 token」的展开
  与 `next_n` 对齐，这里仅作留存说明。

`:165-171` `atom_to_block_table_row`：varlen 返回 `q_atom_idx`，非 varlen 返回 `q_atom_idx / kNumNextNAtoms`，
即 **block_table 按 (batch 行) 索引**（同一行的多个 next_n token 共享一行 block_table）。

### 3.2 状态刷新：`refresh_num_kv_and_advance`（`:179-192`）

核心是两条信息：「下一个 atom 要走几步」（`current_advance`）与「当前 atom 有多少 KV block」（`current_num_kv`）。

- varlen（`:180-185`）：用一次 `is_paired` 判断同时得到 `current_advance=2|1` 与 `current_num_kv=ceil_div(ctx, BLOCK_KV)`，
  context 取配对后一个（`:184`）。注释 `:177-178` 强调：**复用一次 `is_paired` 检查**，
  `indices` 每个 atom 边界只读一次。
- 非 varlen（`:186-191`）：`current_advance=1`；`q_idx = q_atom_idx/kNumNextNAtoms`；
  `lens_idx = is_context_lens_2d ? q_idx*kNextN + kNextN-1 : q_idx`（`:189`）；
  `current_num_kv = ceil_div(context_lens[lens_idx], BLOCK_KV)`。

注意 `current_num_kv` 是 **KV block 数**（`ceil(ctx/64)`），而 metadata 的 split 数是 `ceil(ctx/256)`；
两者差 4 倍，正是 `kNumBlocksPerSplit=4`。所以 scheduler 的 `current_kv_idx` 以 **block** 为单位，
从 `kv_split_idx * kNumBlocksPerSplit` 起步（`:205`）。

### 3.3 构造与推进（`:194-245`）

- 构造 `:194-210`：从 `schedule_meta` 里以 `uint2` 读出本 SM 的起止对
  （`:203-204` `reinterpret_cast<const uint2*>(schedule_meta)[sm_idx]` / `[sm_idx+1]`）；
  `:205-206` `current_kv_idx = current_pack.y * kNumBlocksPerSplit`、`end_kv_idx = end_pack.y * kNumBlocksPerSplit`；
  `:208-209` 无条件调用一次 `refresh_num_kv_and_advance(current_q_atom_idx)`——靠反序分配保证界内。
- `:215-221` `should_refresh_num_kv`：varlen 恒 true；非 varlen 只在 `q_atom_idx % kNumNextNAtoms == 0`
  （即 atom 组边界）刷新。对当前支持的 `kNumNextNAtoms=1`，两者都恒 true，所以 **SM90 上 varlen / 非 varlen
  的刷新语义实际等价**——差异只在「advance 可以是 2」以及 block_table 行的取法。
- `:223-241` `fetch_next_task`：先回填当前 `(q_atom_idx, kv_idx, num_kv)` 与 `last_advance`（`:224-227`）；
  `:229-230` 若已到 `(end_q, end_kv)` 返回 false；`:232-238` `current_kv_idx += kNumBlocksPerSplit`，
  超过 `current_num_kv` 就归零并把 `current_q_atom_idx += current_advance`（**用旧 advance**），
  满足条件且 `exist_q_atom_idx` 时刷新。
- `:243-245` `exist_q_atom_idx`：`q_atom_idx < end_q or (q_atom_idx == end_q and 0 < end_kv)`，
  用于防止越界刷新。

主 kernel 侧的消费（`impls/sm90_fp8_paged_mqa_logits.cuh:151-215`）：TMA warp 用一个 `fetch_next_task`
驱动的流水线，`prefetch_q = (q_idx != next_q_idx and exist_q_atom_idx(next_q_idx+1))`（`:179`）
只在 Q 真的换行时才预取；KV block 的物理块号通过 `block_table[q_idx*block_table_stride + (kv_idx + kv_group_idx + lane_idx*4)]`
（`:195-196`）读出并 `shfl` 广播（`:198`）。

---

## 4. 与 SM100 分页调度器的对照

| 维度 | SM90 分页（本篇） | SM100 分页（第 24 篇） |
|---|---|---|
| 支持模式 | **非 varlen**（batch + next_n∈{1,2} + 2D lens） | **varlen**（indices run-length，next_n==1） |
| block_kv | 只允许 64（`attention.hpp:165`） | 32/64/128（`attention.hpp:154`） |
| split | `split_kv=256` = 4×block_kv（固定） | `SPLIT_KV=384`（`kMQALogitsSplitKV`） |
| work 单位 | **KV split 段数**（`ceil(ctx/256)`） | **请求内 Q block × KV split** |
| 代价模型 | 段数本身（每段 KV work 相同） | `2·BLOCK_Q·blocks + tokens`（Q 随块、epilogue 随 token） |
| 分配 | 反序均分 `q/r/pivot` + 二分（`meta[sm], meta[sm+1]`） | 同款反序均分 + 二分（复用 `sm100_mqa_logits.cuh`） |
| 溢出处理 | 无（段数为 u32，最多 2³² 段） | 代价可溢出 → 借 `num_requests` 最高位回退 token 代价 |
| Q 粒度 | 稳定：metadata 直接按 q_idx 段分 | 请求内 Q **均分**（`base + remainder`），可跨 Q block 切 |
| KV 复用 | 隐式：同一 atom 的 split 落在相邻段 | 显式 `kv_shared_with_prev` + `kSplitsPerChunk` 分块 |
| 分页坐标 | **不在调度器**（主 kernel 逐 KV block 查 block_table） | 调度器内 `get_kv_page_coords`（每 lane 缓存一页滑动窗口） |

**为什么 SM90 敢用「纯段数」代价模型**：SM90 分页下同一请求（或同一 batch 行）的每个 KV split
工作量几乎相同（都是 256 个 KV token × 固定 head dim），且每个 SM 领的是一段**连续**的段号，
所以按段数均分就已经很均衡。实测（§5）per-SM split 数差距只有 ±1，`[3..4]` 或 `[7..8]`。
SM100 引入 varlen 后，一个请求内 Q token 数不等、Q block 与 split 的成本结构又不同，才需要两段式代价模型。

**为什么 SM90 把分页坐标挪出调度器**：SM90 主 kernel 每个 KV block 现查一次 `block_table`
（`impls/...:195-196`），没有滑动缓存；SM100 主 kernel 为了省 TMA 描述符构造做了「每 lane 缓存一页」
的滑动窗口，才把它放进调度器。这是**两代 kernel 取舍不同**，不是同一逻辑的两种封装。

---

## 5. H100 真机实测

在 H100（sm90，132 SM，CUDA 13.2 / 驱动 570）上**直接编译运行真实设备代码**
（`scheduler/sm90_paged_mqa_logits.cuh` 原样 include，只加 `cute/cutlass` + PDL）：

- 源码 `src/sm90_paged_mqa_logits_schedule.cu`、driver `harness/exp_sm90_paged_mqa_logits_schedule.sh`、
  原始输出 `harness/exp_sm90_paged_mqa_logits_schedule.out.txt`。
- 做法：(A) 真机跑 `sm90_paged_mqa_logits_metadata<...>` 生成 `schedule_meta`；
  (B) host 端**独立复刻**（varlen atom 分组 / warp 前缀和 / 反序分配 / 二分）逐字段对拍每 SM 边界；
  (C) 真机跑 `SM90PagedMQALogitsScheduler` 发射任务，和 host 复刻逐任务对拍，
  并验证「所有 `(atom, split)` 恰好覆盖一次」（dup/miss/bad=0）、`num_kv` / `advance` 正确。

| 用例 | 模式 | batch/atom | split 总数 | metadata 边界 | 任务覆盖 | per-SM split |
|---|---|---|---|---|---|---|
| uniform batch=64 ctx=2048 | 非 varlen, next_n=1 | 64 | 512 | 0 不符 | 512/512, dup=miss=bad=0 | [3..4] |
| skewed lens batch=60 | 非 varlen, next_n=1 | 60 | 450 | 0 不符 | 450/450 | [3..4] |
| few batch=3 | 非 varlen, next_n=1 | 3 | 18 | 0 不符 | 18/18 | [0..1] |
| batch=37 (unaligned) | 非 varlen, next_n=1 | 37 | 259 | 0 不符 | 259/259 | [1..2] |
| next_n=2 batch=40 | 非 varlen, next_n=2 | 40 | 400 | 0 不符 | 400/400 | [3..4] |
| next_n=2 last-col wins | 非 varlen, next_n=2 | 8 | 128 | 0 不符 | 128/128 | [0..1] |
| varlen uniform req=8 | varlen | 512→256 atom | 2048 | 0 不符 | 2048/2048 | [15..16] |
| varlen paired ctx cols | varlen | 84→43 atom | 443 | 0 不符 | 443/443 | [3..4] |
| varlen 3 req → 8 SM | varlen | 32→16 atom | 224 | 0 不符 | 224/224 | [28..28] |
| nonvarlen skewed 8 SM | 非 varlen | 6 | 60 | 0 不符 | 60/60 | [7..8] |

**十个用例 ALL PASS**，几点解读：

1. **metadata 边界与 host 复刻逐字段 0 不符**，包括 `batch=37`（`aligned_batch_size=64` 补零）与
   `varlen paired ctx cols`（配对后 context 取 `[t+1]`）——坐实 §2 的前缀和 padded 语义与二分公式。
2. **真实 scheduler 发射的任务恰好覆盖所有有效 split 一次**（`dup=miss=bad=0`、`tasks==segs`），
   且 `num_kv`、`advance`、token / block_table 行映射与 host 复刻逐任务一致。
3. **负载均衡质量**：per-SM split 数最多只差 1（`[3..4]`、`[7..8]`、`[15..16]`），
   印证 §4「纯段数代价模型在 SM90 分页下已足够」。
4. **配对（varlen pair）语义**：`varlen uniform req=8` 里 512 token 被合并成 **256 个 atom**
   （每请求 8 token → 4 atom），split 总数 2048；`varlen paired ctx cols`（84 token → 43 atom）
   验证了 metadata 取的是**配对后一个 token 的 context**。
5. **少 SM / 空 SM**：`varlen 3 req → 8 SM`（16 atom、224 split）8 个 SM 每个恰好 28 段；
   `few batch=3` 下大量 SM 分到空区间仍正确（`[0..1]`，无越界刷新）。

> 注：本实验只验证**调度逻辑**，不涉及 SM90 分页主 kernel 的 WGMMA/TMA 数值（那在第 19 篇已验证：
> 分页 3 组 `vs_sim` ~2e-3、230–370 TFLOPS）。

---

## 6. 结论与边界

- SM90 paged MQA logits 的调度器是一个**两段式**结构：metadata kernel（单 warp）做
  「atom 分组 → `ceil(ctx/256)` 前缀和 → 反序均分 → 二分定位」，主 kernel 的
  `SM90PagedMQALogitsScheduler` 做「读 `uint2` 起止 → 按 split 步进 → 刷新 `num_kv`/`advance`」。
- 它与 UMMA/tcgen05 无关，是**跨代可复用的 host/device 逻辑**，本篇在 H100 上把真实设备代码逐条跑通。
- **SM90 专属边界**：`block_kv==64`、非 varlen、`next_n∈{1,2}` 是当前 host 的硬约束（`attention.hpp:164-165`、
  `sm90_fp8_mqa_logits.hpp:192`）；varlen / `block_kv∈{32,128}` / `schedule_meta` 负载均衡在 **SM100 专属**
  （`attention.hpp:153-162`，仅分析）。`kPadOddN`（next_n≥3 奇数）与 `kNumNextNAtoms>1` 是源码为将来
  预留、当前 host 走不到的分支。
- 与第 19 篇（SM90 主 kernel）、第 23/24 篇（SM100 dense 连续/分页调度器）合并，MQA logits 的
  **三种架构 × 两种布局** 调度链路至此讲全。
