# MPK 采样链两段式 device 身体走读：`argmax` 与 temperature/top-k/top-p 采样的 partial+reduce（P142-1）

> 定位：本文补掉 `docs/37-mpk-sm100-misc-tasks.md` **§3 采样链**只给摘要、`docs/42-mpk-prob-bookkeeping.md`
> 只覆盖 `softmax_gather`/`prob_scatter`/`prob_extract` 的**中间缺口**——把「**一个 task = 一个 CTA** 的
> partial + reduce 两段式采样」逐行打开：`argmax_sm100.cuh`（177 行）与 `sampling_sm100.cuh`（419 行）
> 的 device 身体、其**块级归约/打包 id** 的公共骨架，以及 `tasks/common/sampling.cuh`（228 行）那份
> 「整体一遍 Gumbel」的旧版。对应 ROADMAP「下一步候选」第三条残留：
>
> > `docs/37` §3 采样链剩下的两段式 partial/reduce 身体（`argmax_sm100`/`sampling_sm100`）逐行。
>
> 快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/55` 同快照）。
> **纯源码走读（资料分析），无新实验**——本节 device 身体是 arch-agnostic 标量（见 §6），但按现有
> 约定仍只在目标卡 H100/sm90 上做静态走读、不构建运行。`softmax_gather`/`prob_scatter`/`prob_extract`
> 的逐行已在 `docs/42`，本文只作交叉引用、不重复。

---

## 0. 三条 headline

1. **（headline）采样链是「两段式」的第二次出现，且两段间只传「每 chunk 的极值 + 归一化配平量」**：
   `argmax` 与 `sampling` 都沿用同一骨架——partial 每 task 在**本 chunk** 上做块级归约、写 `topk_max`（argmax
   是 1）个候选 + 一个 chunk 极值；reduce 单 task 合并。区别在 `sampling` 的 partial **额外带下
   `chunk_max` 与 `sum_exp`**（`sampling_sm100.cuh:152-163` 注释），于是 reduce 能在**不重读 logits** 的
   前提下重建**精确**全词表 softmax 归一化（`:336-360`）——这是「薄 task 也要把中间量压到最小」的
   megakernel 适配（对上 `docs/04` §17「每个融合算子省一趟全张量 HBM 往返」）。
2. **`TASK_SAMPLING_SM100=265` 被两个不同 device 身体、三个 op 名共用**：`sampling_sm100`（common 库
   `sampling_from_logits_kernel`，`task_register.cc:2518-2550`）与 `sampling_reduce_sm100`
   （`sampling_reduce_sm100_kernel`，`:2606-2663`）**都 `register_task_variant(TASK_SAMPLING_SM100, …)`**，
   只有 partial 是独立号 `TASK_SAMPLING_PARTIAL_SM100=279`（`:1553-1603`，`runtime_header.h:167-168`
   注释「reduce stage is a variant of TASK_SAMPLING_SM100」）。**读 `TaskType` 表不能反推唯一 device 身体**
   （对上 `docs/24`/`docs/29` 的计数口径坑、`docs/37` §3.4）。
3. **（headline）`_sm100` 后缀不代表 SM100 专属，采样/argmax 在 cc90 就被选用**：前端
   `argmax_partial_layer` 对 `target_cc == 100 or 90` **都**登记 `"argmax_partial_sm100"`
   （`persistent_kernel.py:2627-2630`），只有 reduce 对 `cc==100` 才用 `sm100`、否则回 Ampere
   `argmax_reduce`（`:2650-2657`）；即 **Hopper 上走的是「sm100 partial + Ampere reduce」的混合对**，而
   二者的 `(chunk_index, relative_index)` 打包/解包语义**逐行一致**（§2.3）。这与 `docs/55` §0(A)
   「`glm_moe_router_sm100` 被 Hopper task_header 共享」、`docs/46` 的 `_sm100` 命名惯例是同一现象族。

---

## 1. 生产布线：一个两段式、四层坐标

`TaskType`（`include/mirage/persistent_kernel/runtime_header.h`，SM100 块 145–175）：

| op 名（`graph.cc`） | `TaskType` | `task_config`（`graph.cc`） | device 身体 |
|---|---|---|---|
| `argmax_partial_sm100` | `TASK_ARGMAX_PARTIAL_SM100 = 259` | `(1,2,259,variant)`（`:650-654`） | `argmax_sm100.cuh:83-122` |
| `argmax_reduce_sm100` | `TASK_ARGMAX_REDUCE_SM100 = 258` | `(2,1,258,variant)`（`:655-659`） | `argmax_sm100.cuh:124-175` |
| `sampling_partial_sm100` | `TASK_SAMPLING_PARTIAL_SM100 = 279` | `(1,2,279,variant)`（`:664-668`） | `sampling_sm100.cuh:169-265` |
| `sampling_reduce_sm100` | `TASK_SAMPLING_SM100 = 265` | `(2,1,265,variant)`（`:669-672`） | `sampling_sm100.cuh:276-417` |
| `sampling_sm100`（旧版） | `TASK_SAMPLING_SM100 = 265` | `(1,1,265,variant)`（`:660-663`） | `tasks/common/sampling.cuh:162-226` |

（`task_config` 三元/四元组是 `(num_inputs, num_outputs, TaskType, variant_id)`；argmax partial 是 1 入
2 出、reduce 2 入 1 出，与上表一致。）

前端 layer（`python/mirage/mpk/persistent_kernel.py`）：`argmax_partial_layer:2603`、`argmax_reduce_layer:2632`、
`sampling_sm100_layer:2659`、`sampling_partial_layer:2685`、`sampling_reduce_layer:2728`。

生产消费者（谁真的调这些 layer）：**Qwen3** 的 lm-head 尾部二选一
（`python/mirage/mpk/models/qwen3/builder.py:646-681`）——`do_sample=False` 走 `argmax_partial_layer` +
`argmax_reduce_layer`，`do_sample=True` 走 `sampling_partial_layer` + `sampling_reduce_layer`；
`tasks/common/sampling.cuh` 的旧版 `sampling_sm100_layer` 在本快照的 builder 里**无调用**（layer docstring
`:2667-2672` 自陈 *"Deprecated for production use"*，见 §4）。

**几何（megakernel 适配的关键）**：`qwen3/builder.py:649-681` 里 **partial 的 `grid_dim=(num_workers,1,1)`、
reduce 的 `grid_dim=(1,1,1)`**，`block_dim` 都是 `(128,1,1)`。因为常驻内核里 `blockIdx.x` 是 **worker
索引**（`docs/32` §5.3 / `docs/37` §0），`num_workers` 个 partial task 正好「**一个 worker 一个词表 chunk、
一个 task = 一个 CTA**」。Qwen3 词表补零到 153600、128 worker ⇒ **CHUNK = 1200 floats**
（`sampling_sm100.cuh:179` 注释原话 *"Qwen3 with 128 workers → CHUNK=1200 floats"*）。reduce 只有 1 个
task ⇒ 单 CTA 串行处理全部 batch 行。

`NUM_THREADS` 是编译期 worker 线程数 = **128**（`tasks/common/worker_config.h:18`），与 `block_dim=128`
和 `static_assert(TOPK_MAX <= NUM_THREADS)`（`sampling_sm100.cuh:176`、`:290`）闭合；`topk_max` 默认 32、
前端硬断言 `0 < topk_max <= 128`（`persistent_kernel.py:2710`）。

---

## 2. `argmax` 两段：块级归约 helper + 「打包 id」的 reduce

### 2.1 `block_reduce_max_idx_sm100`（`:36-76`）：结果只留在 thread 0

- `extern __shared__ char smem[]`（`:40`）——注意这是**全 TU 共享的动态 SMEM 基址**，同名符号必须声明一致
  （现「已知坑」；`docs/54` §已知坑）。它先把基址**向上对齐到 128 B**（`:41-42`），再在 `smem_idxs + 32`
  放一个 `T` 数组（`:43`，注释 *"max 32 warps"*）——即 slot 数按 **warp 上限**预留，不随 blockDim 变。
- 块内第一级：`warp_reduce_max_idx_sm100(val, idx)`（`:47` → `:22-34`）用 `__shfl_down_sync` 做
  `offset=16→1` 的蝶形；注意它把 `val` **先转 float 再 shfl**（`:26` `float tmp = __shfl_down_sync(…, (float)val, …)`）
  ——`T` 是 `bfloat16`，float 中转等价于本 case 的比较精度；比较是 `other_val > val`（严格大于），
  **相等时保留低 lane 的 `idx`** ⇒ **tie 取更小 index**。
- 同步用 `cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, /*bar-id*/ 6)`（`:45`）的
  `arrive_and_wait()`（`:57`）——**命名屏障 ID 6**（`sampling` 用 7，见 §3；命名屏障 ID 的「百家争鸣」
  口径张力在 `docs/37` §8、`docs/51`）。
- 第二级：只有 `my_warp_id == 0`（`:60`）参与，且**最终结果只写到 lane 0 的 `val/idx`**（`:71-74`）
  ——调用方要靠「只有 `tidx==0` 才算话」来使用（见下）。

### 2.2 `argmax_partial_sm100_kernel`（`:83-122`）：每 chunk 的局部极值

- `valid_len = min(CHUNK_SIZE, VOCAB_SIZE - chunk_start)`（`:96`）：**把 lm_head 的 padding 行排除出归约**
  （注释 `:94-96`），`chunk_start` 由运行时参数给出（`= task_offset * num_elements`，`task_register.cc:2479`）。
- 外层 `for batch_idx < num_active_tokens`（`:102`）：**每个 task 遍历本 task 的全部 batch 行**，而不是按
  `blockIdx` 取行——这正是 `docs/37` §0 的「一个 task = 一个 CTA、`blockIdx.x` 是 worker id」范式。
  `num_active_tokens` 由 `runtime_config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` 传入
  （`task_register.cc:2478`，qo_indptr 前缀和的末元素 = 总活跃 token 数）。
- 线程内先扫 `for i = tidx; i < valid_len; i += NUM_THREADS`（`:106-112`），比较 `val > local_max` ⇒ 同样
  小 index 优先；再次 `block_reduce_max_idx_sm100`（`:114`）。
- `tidx == 0` 写 `output_val[batch*NUM_PARTIAL_TASKS]`、`output_idx[batch*NUM_PARTIAL_TASKS]`（`:116-119`）
  ——**每个 batch 行在 partial 缓冲里只占 1 列**（argmax 的 `topk_max` 恒为 1），`local_idx` 是 **chunk 内相对
  下标**。

### 2.3 `argmax_reduce_sm100_kernel`（`:124-175`）：打包 `(chunk, relative)` 成 64-bit

- 合并时把 **`chunk_index` 放高 32 位、`relative_index` 放低 32 位**（`:153-154`
  `local_packed_idx = ((long long)i << 32) | partial_idxs[...]`），再交给同一个块级归约（`:158`）——
  这样「先比 value、value 相等再比 index」在**一次 64-bit 比较**里完成（value 仍是主键）。
- `tidx == 0` 解包：`winning_chunk_idx = packed >> 32`、`winning_relative_idx = packed & 0xFFFFFFFF`，还原
  `final_output[batch] = winning_chunk_idx * CHUNK_SIZE + winning_relative_idx`（`:160-166`）；全 `-inf` 时
  写 `-1` 哨兵（`:168-171`）。
- **与 Ampere `argmax_reduce_kernel` 逐行一致**：`tasks/ampere/argmax.cuh:116-163` 是同一段逻辑（只把
  `NUM_THREADS` 换成 `blockDim.x`、把命名屏障换成 `__syncthreads()`）。因此 §0 headline 3 里 cc90 的
  「`argmax_partial_sm100` + Ampere `argmax_reduce`」混合对是**数值自洽**的——两者约定 partial 的
  `output_idx` 存 **chunk 相对下标**、reduce 再乘 `CHUNK_SIZE` 补上 chunk 偏移。

---

## 3. `sampling_sm100.cuh` 两段：从「每 chunk 候选 + 归一化配平」到 Gumbel 抽签

文件头注释（`:16-32`）已经把设计讲清，下面逐段核对。

### 3.1 公共 helper（`:47-127`）

- `warp_reduce_max_idx_sampling`（`:47-58`）：与 argmax 的蝶形同构，但 idx 是 `int`（不是 `long long`），
  因为没有 chunk 打包需求。
- `block_reduce_max_idx_sampling`（`:62-90`）：**与 argmax 版的关键差别是「结果广播给每个线程」**
  （注释 `:60-61`：*"the result is broadcast to every thread, so control flow that depends on it stays
  uniform"*）。做法：warp 归约 → lane0 写 scratch → `bar.arrive_and_wait()`（命名屏障 **7**，`:64`）→
  **每个线程顺序扫 `num_warps` 个槽取最优**（`:80-85`）→ **第二次 `arrive_and_wait()`**（`:87`，注释
  *"Keep the scratch alive until every thread has read it"*）后才覆写 `val/idx`（`:88-89`）。两次屏障正是
  「广播式归约」的代价；它让 §3.3 的 top-k 弹出循环**边界对所有线程一致**。
- `block_reduce_sum_sampling`（`:92-114`）：同构的「shfl 求和 + 广播」，只用了 `scratch_val`。
- `sampling_gumbel_noise`（`:116-127`）：Philox 生成一个均匀数再 `-log(-log(u+ε)+ε)`。注释 `:116-118`
  点明**子序列 key 用 token id 而非候选槽**（`curand_init(seed, token_id, offset)`），使「一个 token 的噪声
  与词表怎么分块无关」——这是两段式能保持**确定性**的前提。
- `sampling_philox_offset_for_row`（`:131-150`）：把 logits 行映射到「拥有它的请求」的 decode step；
  生产路径按 `qo_indptr`/`request_ids`/`steps` 解析，`test_mode`（元数据未设）回退
  `fallback_offset + batch_idx`。运行期参数里 `fallback_offset = step[0] * batch_size`（`task_register.cc:2659`）。

### 3.2 `sampling_partial_sm100_kernel`（`:169-265`）：保留 `TOPK_MAX` 候选 + `(chunk_max, sum_exp)`

- 两条 `static_assert`：`TOPK_MAX <= NUM_THREADS`（`:176`，reduce 的 Gumbel 抽签「一线程一候选」假设）；
  整 chunk 进 SMEM 且加 scratch 后不超过 megakernel 动态 SMEM 预算（`:180-184`）。
- SMEM 布局：`scratch_val[32]` → `scratch_idx[32]` → `chunk[CHUNK_SIZE]`（`:194-199`，基址同样 128 B 对齐）。
- 每个 batch：把本 chunk 载入 SMEM 并**就地乘 `inv_temperature`**，越界位置写 `-INFINITY`（`:207-215`）。
- chunk 极值 + `block_reduce_max_idx_sampling`（`:218-227`）；`sum_exp = Σ exp(chunk[i] - chunk_max)`（`:229-235`
  仅当 `chunk_max_pos >= 0`）。`tidx==0` 把 **`chunk_max` 写到 slot `TOPK_MAX`**、**`sum_exp` 写到 slot
  `TOPK_MAX+1`**（`:237-241`），候选值/全局 id 从 slot 0 起（`:254-257`，id = `chunk_start + pos`）。
- 弹出循环 `for j < TOPK_MAX`（`:243-263`）：每轮块级求 argmax，`tidx==0` 写下一个候选并把该位置**掩成
  `-INFINITY`**（`:259`），再 `bar.arrive_and_wait()`（`:262`）保证所有线程看到掩码后进入下一轮 ⇒ 候选按
  **降序**输出。复杂度 `O(TOPK_MAX)` 轮块归约（朴素但确定，TOPK_MAX=32）。
- **注意与 argmax 的 id 口径差别**：sampling partial 的 `output_idx` 直接存 **全局词表 id**
  （`chunk_start + pos`），而 argmax partial 存 **chunk 相对下标**、由 reduce 补 chunk 偏移。两个 reduce
  因此解包方式不同（§3.3 直接取 `cand_idx`）。

### 3.3 `sampling_reduce_sm100_kernel`（`:276-417`）：重建 softmax → top-k/top-p → Gumbel

- SMEM 布局（`:309-317`）：`scratch_val[32]`/`scratch_idx[32]` + `cand_val[NUM_CANDIDATES]` +
  `cand_idx[NUM_CANDIDATES]` + `sel_val[TOPK_MAX]` + `sel_idx[TOPK_MAX]`，其中
  `NUM_CANDIDATES = NUM_PARTIAL_TASKS * TOPK_MAX`（`:289`）；又一条 SMEM 预算 `static_assert`（`:292-297`）。
- `k_limit = (top_k > 0 && top_k < TOPK_MAX) ? top_k : TOPK_MAX`（`:325`）——`top_k=0` 表示关闭 top-k 过滤。
- 每个 batch：把各 chunk 的候选聚进 `cand_val/cand_idx`（`:328-334`，`chunk = i / TOPK_MAX`、`slot = i % TOPK_MAX`）。
- **重建精确归一化**：先在 `NUM_PARTIAL_TASKS` 个 `chunk_max` 上求 `global_max`（`:336-348`）；再
  `total = Σ chunk_sum * exp(chunk_max - global_max)`（`:350-360`）——由各 chunk 的
  `(chunk_max, sum_exp)` **配平**出全词表的 `Σ exp(x - global_max)`，**不需要再读一遍 logits**。
- **top-k/top-p 弹出**（`:362-392`）：`mass_target = top_p < 1 ? top_p*total : INF`（`:365`）；循环
  `j < k_limit`，块级 argmax 选当前最大候选（`:368-377`），`tidx==0` 记入 `sel_val/sel_idx` 并把该候选掩
  `-INF`（`:381-385`）；`mass += exp(val - global_max)`、`num_selected = j+1`（`:386-387`，**所有线程都算，
  值一致**），`bar.arrive_and_wait()`（`:388`）后若 `mass >= mass_target` 则 break（`:389-391`）。
  注释 `:362-364` 强调循环边界**对所有线程一致**依赖的正是 §3.1 的广播式归约。
- **Gumbel 抽签**（`:394-409`）：`greedy`（temperature ≤ 0）时直接 `winner = 0`（`sel_val` 是降序，0 即 argmax）；
  否则 `tidx < num_selected` 的线程各自取 `sel_val[tidx] + Gumbel(philox, row_offset, sel_idx[tidx])`
  （`:400-406`，噪声 key 到 **token id**），再块级 argmax 得 `winner`（`:407`）。
- `tidx==0` 写 `final_output[batch]`（`:411-414`）；`bar.arrive_and_wait()`（`:415`）为下一次 batch 复用
  scratch。文件头注释 `:25-26` 解释 Gumbel-Max *"argmax of logit/T + Gumbel noise is an exact categorical
  draw, so no renormalization is needed"*——即 softmax 归一化只用在 top-p 质量阈上，抽签本身不需要。

---

## 4. 旧版 `sampling_from_logits_kernel`：整体一遍 Gumbel（`tasks/common/sampling.cuh`）

- 由 `register_sampling_sm100_task` 发射 `sampling_from_logits_kernel<256, 4, bfloat16, int>`
  （`task_register.cc:2543`），但**共用了 `TASK_SAMPLING_SM100`**（§0 headline 2）。
- 实现（`:162-226`）：用 CUB `BlockReduce<SamplingDataAndIndex<DType,IdType>, BLOCK_THREADS, BLOCK_REDUCE_WARP_REDUCTIONS>`
  当「argmax 归约」——把 `(data,index)` 定义成一个 `operator+` 取 max 的幺半群（`:71-95`），于是
  `Sum<VEC_SIZE>(cur_data)` 就是块级 argmax。它**一次遍历**词表，边读边加 Gumbel 噪声
  （`GenerateSamplingGumbelNoise`，`:99-149`；`VEC_SIZE=4`，`curand_uniform4`），最后 `tx==0` 写 index
  （`:219-221`）。
- 前端 `sampling_sm100_layer`（`persistent_kernel.py:2659-2683`）的 docstring 自陈它**已废弃用于生产**：
  *"no temperature/top-k/top-p, no vocab-pad bound, writes int into int64 output_tokens, and freezes the
  Philox offset at compile time. Prefer sampling_partial_layer + sampling_reduce_layer."*（`:2669-2672`）
  ——这解释了为何「同一 TaskType 号有两种身体」：新两段式复用了旧版的号，旧身体作为遗留变体仍在。

---

## 5. 与簿记算子的分工（交叉引用，不重复）

`softmax_gather`（`TaskType 282`）、`prob_scatter`（284）、`prob_extract`（286）是**投机解码窗口的概率簿记**
（`docs/42` 已逐行），与本文的两段式采样**不是同一条链**：它们算「**指定 token** 在 softmax 下的概率」
（`prob = exp(logit[target]-max-logsum)`，`softmax_gather_sm100.cuh:18-29`）而非抽取新 token。本文的采样链
才是 lm-head → 下一个 token 的默认尾段；两条链在 **DeepSeek-V3** 的投机解码里并存
（`deepseek_v3/builder.py:1876/2220` 用簿记、`qwen3/builder.py:649` 用采样）。

---

## 6. `_sm100` 命名与 arch 分派（为什么 arch-agnostic 却叫 sm100）

- `argmax_sm100.cuh` 与 `sampling_sm100.cuh` 都是**纯标量 + 块级 shfl/命名屏障**实现，无 TMA/UMMA/mma/TMEM，
  **没有 SM100 专属指令**：因此 `argmax_partial_layer` 让 `cc==90` 也走 `_sm100` 版
  （`persistent_kernel.py:2627`），`sampling_*_layer` 更是**完全不判 cc**（`:2683`/`:2722`/`:2763`）。
- 与 `tasks/ampere/argmax.cuh` 的 Ampere 版对比，唯一差异是同步原语（`NamedBarrier(…,6)` vs
  `__syncthreads()`）与遍历上界（`NUM_THREADS` vs `blockDim.x`）——**语义逐行等价**（§2.3），故 cc90 的
  「sm100 partial + Ampere reduce」混合对自洽。
- 结论与 `docs/55` §0(A) 的 `glm_moe_router_sm100`、`docs/46` `rotary_embedding_sm100` 同族：**`_sm100`
  是「按 SM100 生产路径整理 / SM100 块内编号」的命名，不等于「仅 SM100 可编译」**。读 MPK 的 device 目录时
  必须按**指令集内容**而非**文件名后缀**判定可否在目标卡执行（这也是本专题反复出现的「按名字/注释推断会错」）。

---

## 7. 判据映射与 megakernel 意义

| 判据 | 本段是否贡献 |
|---|---|
| C1 单次 launch | 间接（partial/reduce 都只是被 worker 调用的 `__device__` 身体；launch 由运行层统一发，`docs/06`） |
| C2 常驻 | 间接（同 C1） |
| C3 多算子范围 | **是**——「采样段用几个 task、谁是 partial 谁是 reduce」的清单 |
| C4 设备内协调 | 间接（partial→reduce 的数据依赖由 MPK event 承担；task 内用命名屏障 6/7） |
| C5 SM 级调度 | 否（task 内部无固定角色 worker/scheduler） |

**对 megakernel 专题的意义**：

- **薄 task 的「两段式」不是融合不足，而是资源约束下的正确切分**：词表（153600）远超单 CTA 一趟能归约的
  规模，且 partial 的块数被**绑定到 worker 数**，于是「一词表 chunk 一 worker」天然把重归约切平；reduce
  只面对 `NUM_PARTIAL_TASKS*TOPK_MAX`（128×32=4096）个候选，单 CTA 即可。这与 `docs/36`（MLA split-KV）、
  `docs/04` §17（split 而非傻融）同属「**按 SM 资源预算决定融合边界**」。
- **中间量最小化**：partial 只带 `TOPK_MAX+2` 个 float + `TOPK_MAX` 个 id（逐 batch），而非整张 `[batch,vocab]`
  概率矩阵——正是「每融合一个算子省一趟全张量 HBM 往返」（`docs/04` §17）在采样段的具体形态。
- **确定性适配**：Gumbel 噪声 key 到 **token id**（`:116-118`）而非候选槽，使「partial 怎么分块」不影响结果
  ——这是常驻内核里**同一图可复现**的必要条件（对上 `docs/04` §9 的正确性哨兵思路）。
- **广播式块归约**：采样用「结果广播 + 两次命名屏障」（§3.1）换取 top-k/top-p 循环的**统一控制流**，与
  argmax 的「只 thread 0 持有」相反——这是同一个块级归约在不同调用约束下的两种实现。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：所有行号为快照 `f9eb70c` 静态事实；本文 device 身体是 arch-agnostic
  标量，但按现有约定只做静态走读、未取任何 device 计数，未复算数值。
- **未与生产端到端对跑**：`qwen3/builder.py` 的 `do_sample` 分支未运行；「`grid_dim=(num_workers,1,1)` ⇒
  每 worker 一 chunk」是从前端传参与 `blockIdx.x == worker id` 约定做的**静态推断**，未构建跟踪。
- **两处 `TaskType` 复用只记不判**：`TASK_SAMPLING_SM100=265` 有旧版/新版两个身体（§4）；`258/259` 的
  partial/reduce 枚举值与名称顺序**相反**（`runtime_header.h:145-146`：`…REDUCE_SM100 = 258`、
  `…PARTIAL_SM100 = 259`）——是否影响任何按号索引的路径未追（`docs/24`/`docs/29` 的计数口径坑）。
- **命名屏障 ID 6/7 的口径张力**：与 `docs/37` §8、`docs/51` 记录的「用户命名屏障 ID 须 ≥8」注释并存，
  本文只记录、不判对错。
- **`MPK_MAX_NUM_BATCHED_REQUESTS`/`MPK_MAX_NUM_BATCHED_TOKENS`** 在快照里是 `persistent_kernel.cuh:58-59`
  的**注释掉的 `#define`**，实际取值由构建配置给；本文只按「qo_indptr 前缀和末元素 = 总 token 数」表述。
- **未展开**：CUB `BlockReduce` 的内部实现（`tasks/common/sampling.cuh:170`）只到「用 CUB 当 argmax 归约」
  这层；`__expf`/`curand_uniform` 的精度与鲁棒性未展开。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/55` 同快照）。
- **argmax**：`include/mirage/persistent_kernel/tasks/blackwell/argmax_sm100.cuh`（177 行；
  `warp_reduce_max_idx_sm100:22-34`；`block_reduce_max_idx_sm100:36-76`（命名屏障 6 `:45`、结果只留 thread 0
  `:71-74`）；partial `:83-122`（padding 排除 `:94-96`、`chunk_start` `:88`）；reduce `:124-175`
  （打包 `:153-154`、解包 `:160-166`、`-1` 哨兵 `:168-171`））。
- **argmax 登记/布线**：`src/kernel/task_register.cc:2440-2481`（partial，`qo_indptr` `:2478`、
  `task_offset*num_elements` `:2479`）、`:2483-2516`（reduce）；`src/kernel/graph.cc:650-659`；
  `python/mirage/mpk/persistent_kernel.py:2603-2657`（cc 分派 `:2627`/`:2650`）；
  `python/mirage/mpk/models/qwen3/builder.py:669-681`。
- **sampling（两段式）**：`.../tasks/blackwell/sampling_sm100.cuh`（419 行；文件头 `:16-32`；命名屏障 7
  `:45`；广播式块归约 `:60-90`（两次 barrier `:76`/`:87`）；`block_reduce_sum_sampling:92-114`；
  `sampling_gumbel_noise:119-127`；`sampling_philox_offset_for_row:131-150`；partial `:169-265`
  （static_assert `:176-184`、SMEM 布局 `:194-199`、写 max/sum `:237-241`、降序弹出 `:243-263`）；
  reduce `:276-417`（SMEM 布局 `:309-317`、重建归一化 `:336-360`、top-k/top-p `:362-392`、Gumbel `:394-409`））。
- **sampling 登记/布线**：`task_register.cc:2553-2604`（partial，`TASK_SAMPLING_PARTIAL_SM100`）、
  `:2606-2663`（reduce，`TASK_SAMPLING_SM100`，`fallback_offset` `:2659`）；`graph.cc:660-672`；
  `persistent_kernel.py:2685-2767`（layer，断言 `:2710`/`:2753`）；`qwen3/builder.py:648-667`。
- **旧版 common 采样**：`include/mirage/persistent_kernel/tasks/common/sampling.cuh`（228 行；
  `SamplingDataAndIndex:71-95`；`GenerateSamplingGumbelNoise:99-149`；`sampling_from_logits_kernel:162-226`）；
  `task_register.cc:2518-2550`；`persistent_kernel.py:2659-2683`（废弃自陈 `:2669-2672`）。
- **Ampere 对照**：`include/mirage/persistent_kernel/tasks/ampere/argmax.cuh`（165 行；partial `:77-112`、
  reduce `:116-163`，与 sm100 版逐行同构）。
- **`TaskType`/常量**：`include/mirage/persistent_kernel/runtime_header.h:145-146`（258/259）、
  `:152`（265）、`:167-168`（279 注释）、`:171/173/175`（282/284/286）；
  `tasks/common/worker_config.h:18`（`NUM_THREADS=128`）。
- **本专题交叉引用**：`docs/37` §3（采样链摘要）、`docs/42`（`softmax_gather`/`prob_scatter`/`prob_extract`
  逐行）、`docs/39` §（common 原语里的 Gumbel-Max）、`docs/32` §5.3 / `docs/37` §0（`blockIdx.x == worker id`）、
  `docs/46`/`docs/55`（`_sm100` 命名的 arch-agnostic 先例）、`docs/04` §17（损失三：融合省 HBM 往返）。
