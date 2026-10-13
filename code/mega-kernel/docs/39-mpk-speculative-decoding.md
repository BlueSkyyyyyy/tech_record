# MPK 投机解码 device 层走读：`tasks/speculative_decoding/*`（P125-1）

> 定位：本文补掉 `docs/37-mpk-sm100-misc-tasks.md` 与 `docs/38-mpk-common-primitives.md` 都点到、
> 但从未逐行打开的**一整层 device 目录**——`include/mirage/persistent_kernel/tasks/speculative_decoding/`
> （5 个头文件、1014 行）。`docs/14`/`docs/24`/`docs/25`/`docs/26` 只在**前端/枚举/家族**
> 层面列过 speculative 系列（`TASK_EAGLE3_*`、`TASK_MTP_*`、`TASK_TARGET_VERIFY_GREEDY`、
> `find_ngram_*`）的名字与 ID；`docs/34`/`docs/35`/`docs/36` 走读了 MLA-MTP 的 **attention** 身体，
> 但没有走这批**token 簿记 / 验证 / n-gram 查找**的 device 身体。本文把它们逐个打开，并把
> `TaskType`/`graph.cc`/`task_register.cc`/`persistent_kernel.py`/builder 五层的生产布线对上。
>
> 快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`（与 `docs/06`–`docs/38` 同快照）。
> **纯源码走读（资料分析），无新实验**——这批 task 里 MTP/Eagle3 是 SM100 路径，本机无 SM100；
> prompt-lookup 是 arch-generic 的，但 mirage 端到端执行另有已知构建/时序依赖（`ROADMAP.md` B1/B2），
> 本文只读代码、不构建、不运行。

---

## 0. 三条 headline

1. **这批 task 全部是「把答案查出来 / 把 token 搬对位置」的簿记型 device 身体，无一条吃 Tensor Core**：
   5 个文件里最大的 `eagle3_ops.cuh`（317 行）也只有 `memcpy`、沿 dim-1 的 `concat`、一张 d2t 查表、
   一个「写 token buffer + 发步进信号」的 commit；`target_verify_*.cuh` 全在**单线程（`t_id==0`）串行**
   做完接受判定。它们不搬权重、不算 GEMM，却是投机解码正确性的**枢纽**：`accepted_count`、
   `new_token_nums`、`tokens[]` 的每个位置都从这里写出去，供运行时 `prepare_next_batch` 推进 step。
   对应 megakernel 视角：**stage 融合把「大算子」压进 SM，却把「小簿记」留成独立 task**，正是
   `docs/04` §8/§10 的成本侧（task 数 / 事件扇入扇出）来源。
2. **同一「验证 draft token」语义在仓库里有三套并行实现，接线状态各不相同**：
   - `target_verify_greedy_kernel`（`target_verify.cuh:21-60`）——prompt-lookup 用的贪心最长匹配，
     被前端 handler（`persistent_kernel.py:2844/2860`）与 `graph.cc:550` 登记；
   - `target_verify_strict_kernel` / `target_verify_probabilistic_kernel` / `mtp_accept_commit_kernel`
     （`target_verify_mtp.cuh:44/112/255`）——DeepSeek-V3 MTP 用的三段式（strict / probabilistic /
     commit），被 `graph.cc:873-877` 登记、DeepSeek-V3 builder 真拼（`deepseek_v3/builder.py:1947/2007/2037`）；
   - `target_verify_synthetic_kernel`（`target_verify_mtp.cuh:191-242`）——**有定义、全仓零调用者**
     （`grep` 只命中定义本身），无登记、无 `TaskType`，是 `docs/16`/`docs/20` 那族
     「已实现未接线」在本目录的成员。
3. **Eagle3 的 Python 侧有一条「文档说要做、build_draft_loop 里没做」的落差**：
   `Eagle3Builder` 的类注释（`eagle3/builder.py:34-56`）声明了完整管线
   （`copy_layer` 抓 aux hidden → draft loop → `mtp_prepare_verify_layer → mtp_verify_strict_layer →
   mtp_accept_commit_layer`），但实际 `build_draft_loop`（`:271-449`）**只跑 draft loop 并 `return
   all_draft_ids`**：既没调 `copy_layer`（aux 由调用方作为入参传入），也没调它注释里的 verify 三段
   （`mtp_verify_strict_layer` 在 Python 侧只有 DeepSeek-V3 builder 调用）。而 Eagle3 专属的
   `eagle3_commit_layer`（`persistent_kernel.py:3154-3185`，对应 `TASK_EAGLE3_COMMIT`）在整个 Python
   侧**无任何调用者**——`TASK_EAGLE3_COMMIT` / `TASK_COPY` 只被登记，未在快照的 builder 里接线
   （静态事实，**只记录不判对错**，同 `docs/36` 对源码自陈的态度）。

---

## 1. 目录位置、包含关系与「谁把它编进生产」

`tasks/speculative_decoding/` 下 5 个文件：

| 文件 | 行数 | 提供 | 登记 `TaskType` | 生产 `task_header` include？ |
|---|---|---|---|---|
| `target_verify.cuh` | 62 | `target_verify_greedy_kernel` | `TASK_TARGET_VERIFY_GREEDY=114` | **否** |
| `prompt_lookup.cuh` | 154 | `find_ngram_{partial,global,global_sequential}_kernel` | `TASK_FIND_NGRAM_{PARTIAL=112,GLOBAL=113}` | **否** |
| `target_verify_mtp.cuh` | 290 | `target_verify_{strict,probabilistic,synthetic}_kernel`、`mtp_accept_commit_kernel` | `TASK_MTP_VERIFY_{STRICT=271,PROBABILISTIC=283}`、`TASK_MTP_ACCEPT_COMMIT=272` | 是（仅 blackwell，`:64`） |
| `mtp_token_ops.cuh` | 191 | `mtp_{token,float}_scatter_kernel`、`mtp_prepare_verify_input_kernel`、`mtp_build_embed_input_kernel` | `TASK_MTP_{TOKEN_SCATTER=273,FLOAT_SCATTER=285,PREPARE_VERIFY=274,BUILD_EMBED_INPUT=294}` | 是（仅 blackwell，`:63`） |
| `eagle3_ops.cuh` | 317 | `copy_layer_kernel`、`concat_kernel`、`eagle3_d2t_remap_kernel`、`eagle3_commit_kernel` | `TASK_{COPY=232,CONCAT=233}`、`TASK_EAGLE3_{D2T_REMAP=235,COMMIT=236}` | 是（仅 blackwell，`:62`） |

包含机制：生产 device 编译入口 `include/mirage/persistent_kernel/persistent_kernel.cuh:48-52` 按
`MIRAGE_GRACE_HOPPER/GRACE_BLACKWELL/else` 三选一 include 对应 `tasks/<arch>/task_header.cuh`；只有
**blackwell** 的 `task_header.cuh:62-64` include 了 `eagle3_ops.cuh` / `mtp_token_ops.cuh` /
`target_verify_mtp.cuh` 三件。**`target_verify.cuh` 与 `prompt_lookup.cuh` 不被任何 `task_header.cuh`
include**（全仓 `grep '"target_verify.cuh"\|"prompt_lookup.cuh"'` 零命中）；它们的 device 符号只在
**单测 wrapper** 侧出现过，且 `tests/runtime_python/runtime_kernel_wrapper.cu` 里对它们的 `using`
（`:13,27`）与 pybind `m.def`（`:1845`）**都被注释掉了**——单测 `setup.py:10` 只是把
`speculative_decoding` 加进 include 目录。这是一处**静态接线落差**：prompt-lookup 的
register/frontend 是活的（`TaskType 112/113/114`），但 device 头不在生产 include 集里（**只记录事实**）。

头文件风格也有两派：`target_verify.cuh:16` / `prompt_lookup.cuh:16` 用 `#include "../common.h"`，
而 `target_verify_mtp.cuh:16` / `mtp_token_ops.cuh:16` / `eagle3_ops.cuh:16` 用
`#include "tasks/common/common_header.cuh"`（即 `docs/38` 的伞状原语入口）。快照里
`tasks/common.h` 并不存在（`find` 只命中 `include/mirage/transpiler/common.h`），`../common.h`
的相对解析在 `tasks/speculative_decoding/` 下指向不存在的文件——同族写法也出现在
`hopper/norm_linear_hopper.cuh:18` 与多个 `ampere/*.cuh`（`common.h`），说明这是仓库里一条**跨目录
沿用的 include 约定**而非本目录独有；本文只记风格差异，不判可编译性。

另有一处**跨目录耦合**：`target_verify.cuh` 用到 `NUM_THREADS_PER_WARP`（`:57`），
`prompt_lookup.cuh` 用到 `NUM_THREADS`/`NUM_THREADS_PER_WARP`（`:40,53,63-65,123`），
都来自 `docs/38` §2.4 那条 `worker_config.h` 的 `NUM_THREADS=128` 约定——簿记 task 直接依赖
共享原语层的线程常数。

---

## 2. `target_verify.cuh`：prompt-lookup 的贪心最长匹配（62 行）

`target_verify_greedy_kernel<NUM_SPEC_TOKENS>`（`:21-60`）：

- **输入语义**（注释 `:19,39-40`）：`spec_token_id[0]` 是**原始 token**，`spec_token_id[1..K]`
  才是被猜的 token；`target_token_id[i]` 是 target 模型在第 `i` 个位置上的 argmax。
- **判定**（`:36-47`，仅 `t_id==0`）：从 `i=0` 起逐位比 `spec_token_id[i+1] != target_token_id[i]`，
  首个不等处 `accepted_count=i` 并 break；全等则 `accepted_count = NUM_SPEC_TOKENS`。结果写进
  `__shared__ local_accepted_num_smem`，`__syncthreads()` 广播（`:48-50`）。
- **回写**（`:53-59`）：`t_id < final_accepted_count+1` 的线程把 `target_token_id[t_id]` 写进
  `new_tokens[t_id]`（含「第一个不匹配的 target token」这个 bonus）；`t_id == NUM_THREADS_PER_WARP`
  （即 32）的线程写 `new_accepted_len[0] = final_accepted_count+1`。
- **接线**（`task_register.cc:1132-1159`）：`num_spec_tokens = input_ops[0].dim[1] - 1`，
  输出 `new_token_nums` 与 `runtime_config.tokens + step[0] + 1` **直接传运行期指针**
  （`:1156-1157`，不经 task graph 的 output tensor）——同一 task 同时改了 step 位置与
  `new_token_nums`，语义上把「接受长度」与「token 落位」绑在一次写里。

---

## 3. `prompt_lookup.cuh`：n-gram 查找的三档实现（154 行）

prompt-lookup 投机 = 「在当前已生成序列里找最后 `NGRAM_SIZE` 个 token 组成 n-gram 最早出现的位置，
把其后 `SPEC_LENGTH` 个 token 当作 draft」。三段：

### 3.1 `find_ngram_partial_kernel<NGRAM_SIZE, NUM_WORKERS>`（`:21-98`）

- 每 block 处理一段输入（`task_id = blockIdx.x`，`:27` 带 `// TODO: Is this good?` 注释），
  把 n-gram（输入末尾 `NGRAM_SIZE` 个 token，`:34-35,47-49`）搬进 SMEM，然后**网格跨步**扫
  `total_elements = input_token_num - NGRAM_SIZE` 个候选位置，`is_ngram` 者用
  `atomicMin(&block_min_idx, idx)` 取**最早**匹配（`:55-92`）。
- SMEM 里除 `input_tokens[NUM_THREADS]` 外还多载 `NGRAM_SIZE-1` 个「跨块边界」token
  （`:63-71`，由 warp 1 之后的线程搬运），避免把 n-gram 直接卡在 `NUM_THREADS` 边界上漏匹配。
- 未匹配时输出保持 `LLONG_MAX`（初始值，`:43-45`），作为「无 n-gram」的哨兵。
- block 结果由 `t_id==0` 写 `output[0] = block_min_idx`（`:95-97`）——所以输出张量是
  `(batch, num_parts=96)`（前端 `dims=(tokens.dim(0), 96)`，`persistent_kernel.py:2803`；
  `task_register.cc:1086` 从输出 tensor 的 `dim[1]` 取 `num_parts` 当 `NUM_WORKERS`）。

### 3.2 `find_ngram_global_kernel<NGRAM_SIZE, SPEC_LENGTH, NUM_PARTIAL_TASKS>`（`:101-140`）

- 单 block 归约所有 partial 结果（`:123-127` 的 grid-stride `atomicMin` 取全局最早位置）。
- 写 draft：`t_id==32` 写 `output_result[0] = tokens_ptr[step]`（**当前 token**，`:130-131`）；
  `t_id < SPEC_LENGTH` 的线程写 `output_result[t_id+1] = tokens_ptr[block_min_idx_shared +
  NGRAM_SIZE + t_id]`，带边界检查（`spec_token_idx <= step`），越界写 `-1`（`:132-138`）。
  ——**`t_id==32` 这个「第 33 号线程」而非 `t_id==0` 的选择是本文件的一个细节**：把「主 token」
  与「draft 槽 0」写成两条互不重叠的线程分支（槽 0 走 `t_id<SPEC_LENGTH` 分支）。
- 前端 `find_ngram_global_layer` 固定 `grid_dim=(1,1,1)`、`block_dim=(128,1,1)`
  （`persistent_kernel.py:2821-2827`），故实际只用 1 个 block 的 128 线程。

### 3.3 `find_ngram_global_kernel_sequential<NUM_PARTIAL_TASKS>`（`:142-153`）

纯串行版：从 `input_array[0..]` 取**第一个** `< LLONG_MAX` 的位置写 `output_result[0]`，
否则 `-1`。**全仓零调用者**（`grep` 只命中定义）——是三段里唯一「有定义、无登记、无前端」的
死代码，同 §0 headline 2 的 `target_verify_synthetic_kernel`。

### 3.4 前端两段 handler

`prompt_lookup_spec_handler`（`persistent_kernel.py:2795-2830`）= `find_ngram_partial_layer`
（输出 `partial_ngram_output`，`(mbt,96)` int64）→ `find_ngram_global_layer`
（输出 `spec_tokens`，`(mbt, spec_length+1)` int64）；`prompt_lookup_verify_handler`（`:2860-2878`）
→ `target_verify_greedy_layer`（`dims=(1,1)` 的 `verify_out`，`:2844-2858`）。两个 handler 注册在
`_spec_decode_handlers`/`_spec_verify_handlers`（`:528-533`），**方法名只有 `"promptlookup"`**；
`Eagle3Config`(method=`"eagle3"`)/`LookaheadConfig`(`"lookahead"`) 不在字典里，走 dispatcher
会 `ValueError`（`speculative.py:34-46` 定义，`persistent_kernel.py:2839` 派发）。

---

## 4. `target_verify_mtp.cuh`：MTP 三段验证 + 两枚「未接线」（290 行）

### 4.1 `target_verify_strict_kernel<NUM_DRAFT_TOKENS>`（`:44-91`）

与 §2 贪心版同构，但 token 数组语义不同：`draft_ids[i] != target_ids[i]` 逐位比（`:65-72`），
`accepted = i`；回写 `output_tokens[t_id] = target_ids[t_id]`（`t_id < final_accepted+1`，`:83-85`），
`accepted_count[0] = final_accepted+1`（`:87-90`）。注意 strict 版与 greedy 版的 **bonus 记号**都在
`accepted+1`，但 strict 版 `output` 与 `accepted_count` 是两个独立 output tensor（greedy 版把
accepted 写 `new_token_nums`、tokens 直接写序列缓冲）。登记 `TASK_MTP_VERIFY_STRICT=271`
（`graph.cc:873-876`，`make_tuple(2,2,...)` 表示 2 入 2 出）。

### 4.2 `target_verify_probabilistic_kernel<NUM_DRAFT_TOKENS>`（`:112-174`）

vLLM 式拒绝采样，且**只吃预算好的概率**（注释 `:99-101`：softmax 在单独 kernel 先算，
使本核 O(NUM_DRAFT) 而非 O(VOCAB)）：

- `p_draft == 0` → 贪心回退（比 token id，`:147-154`）；
- 否则 LCG（`rng_state = rng_state*6364136223846793005 + 1442695040888963407`，`:157`）
  取 `u = (rng_state>>33)/2^31`（`:158-159`），`p_target > u*p_draft` 才接受（`:161-166`）；
- 首个拒绝即停（`still_accepting`），末尾写 bonus `output_tokens[accepted] = target_ids[accepted]`
  与 `accepted_count[0]=accepted+1`（`:170-172`）。
- 三段里唯一再引入 `target_probs`/`draft_probs`/`seed` 三个输入张量的（登记
  `graph.cc:731-735`，`make_tuple(5,2,...)`；前端 `mtp_verify_probabilistic_layer`
  `persistent_kernel.py:3013-3048`）。

### 4.3 `target_verify_synthetic_kernel<NUM_DRAFT_TOKENS>`（`:191-242`）——**死代码**

按 `accept_prob = base_rate * decay^i` 的几何衰减 + LCG 接受（`:222-236`），注释自称「校准到目标
平均接受率」。**全仓零调用者、无登记、无 `TaskType`**（`grep target_verify_synthetic` 只命中定义）。
与 `find_ngram_global_kernel_sequential` 同族。

### 4.4 `mtp_accept_commit_kernel<NUM_DRAFT_TOKENS>`（`:255-288`）

`t_id==0` 写 `new_position[0] = current_position[0] + count` 与 `num_new_tokens[0] = count`
（`count = accepted_count[0]`，含 bonus，`:277-281`）；`t_id < count && t_id <= NUM_DRAFT_TOKENS`
的线程把 `output_tokens[t_id]` 拷进 `final_output[t_id]`（`:284-287`）。登记
`TASK_MTP_ACCEPT_COMMIT=272`（`graph.cc:877-880`，`make_tuple(3,3,...)`）。

---

## 5. `mtp_token_ops.cuh`：token/概率搬运与 MTP 嵌入输入（191 行）

四个极薄 task，全是「一维 batch 级 scatter / 拼接」：

- `mtp_token_scatter_kernel<BATCH,NUM_SLOTS,SLOT_IDX>`（`:37-49`）：`dst[b*NUM_SLOTS+SLOT_IDX] =
  src[b]`（`b=threadIdx.x`，`b<BATCH`）。注释 `:35-36` 明确 `SLOT_IDX`/`NUM_SLOTS` 是**编译期**
  参数（「from static unroll」）——配合 builder 里 `for step in range(num_draft_steps)` 的静态展开
  （`deepseek_v3/builder.py:1889-1894`，每步一个 slot）。
- `mtp_float_scatter_kernel<BATCH,NUM_SLOTS,SLOT_IDX>`（`:54-64`）：同上模板，载体 float32，
  存每步 `P_draft`（probabilistic 用）。
- `mtp_prepare_verify_input_kernel<NUM_DRAFT,MAX_SEQ_LEN>`（`:84-129`）：把主 token 写
  `tokens[req, step+1]`（`:109-112`），draft 写 `tokens[req, step+2+t_id]`（`:122-127`），
  并把 `num_new_tokens[req]` 夹到 `NUM_DRAFT+1` 与 `MAX_SEQ_LEN-cur_step-1` 的较小值（`:113-118`）
  ——**夹取逻辑**保证不越界。
- `mtp_build_embed_input_kernel<BATCH,MAX_SEQ_LEN>`（`:160-189`）：vLLM-aligned（注释 `:131-152`
  引 `vllm/v1/spec_decode/eagle.py L666-669`）——`i < BATCH-1` 读 `tokens[req, step+i+1]`
  （**shifted 的真值 prompt token**），`i == BATCH-1` 读 `output_tokens[i]`（当前 argmax）。
  prefill 时前段是 ground-truth、末位是生成 token；decode（mbt=1）只走末位分支。
  登记 `TASK_MTP_BUILD_EMBED_INPUT=294`（`graph.cc:890-897`，`make_tuple(1,1,...)`）；
  它的 `tokens`/`step` 是**直接读 `runtime_config.tokens`/`runtime_config.step`**
  （`task_register.cc:5089-5097`），不像其他 task 走 `input_ptrs`。

**关键语义修正点**（`:101-104` 注释）：`req` 取自 `task_desc->task_metadata.request_id`，
**不是** `blockIdx.x`——因为常驻 kernel 里 `blockIdx.x` 是 **worker 索引**（同 `docs/32` §5.3 的
`kNumSMs=1`、`docs/34` §3.5 的 TMEM-warp、`docs/37` §2 的薄 task「遍历全 row」）。

---

## 6. `eagle3_ops.cuh`：Eagle3 的四个片段（317 行）

### 6.1 `copy_layer_kernel<T,BATCH,HIDDEN>`（`:34-48`）

`(B,H)` 张量 `memcpy`（网格跨步 `for i=tid; i<total; i+=blockDim.x`）。注释 `:31-33` 说明用途：
Eagle3 要把 target 的**中间层 hidden** 抓进独立 aux buffer（因为 MPK 的 per-layer 中间张量
跨层复用）。`concat_kernel<T,BATCH,HIDDEN,N>`（`:60-80`）：把 N 个 `(B,H)` 沿 dim-1 拼成
`(B,N*H)`，`output[b*N*H + k*H + d] = inputs[k][b,d]`（`:70-79`）；注释 `:56-59` 列出 Eagle3 的两种
用法：`N=3`（h_low/h_mid/h_high 送 `eh_proj`（fc））与 `N=2`（`concat(embed,hidden)` 作 draft QKV 输入）。

### 6.2 `eagle3_d2t_remap_kernel<BATCH,DRAFT_VOCAB_REAL>`（`:102-122`）

`target[b] = hot_id + d2t[hot_id]`（sglang 约定，`d2t` 为 signed int64，`:84-89`）；越界
（`hot_id<0` 或 `>= DRAFT_VOCAB_REAL`）写 **0 哨兵**（`:118-120`）。注释 `:91-95` 给了越界来源：
`lm_head` 为满足 TMA 16B 对齐而**行 padding**（Eagle3 builder 把 32000 pad 到 32256，
`eagle3/builder.py:156-174`），argmax 可能落进 padding 行——写 0 后 verify 会拒绝、正常提交 bonus。

### 6.3 `eagle3_commit_kernel<K,BATCH,MAX_SEQ_LEN>`（`:157-315`）——本目录最长

注释 `:124-156` 列四职责：写 verified 前缀、写下一轮 K 个 draft、发 `new_token_nums` 步进信号、
把本轮 draft 快照进 `drafts_prev`（attach_input，跨迭代解「verify 需要上一轮 draft」而不违反
「每对 producer/consumer 唯一 edge」不变量）。实现：

- verified 前缀写 `tokens[req, step+1+t_id] = argmax[t_id]`（`t_id<ac`，且 `pos>=prompt_len`
  才写，`:197-202`）——`ac = accepted_count[0]`（来自 verify_strict，`[1,K+1]`）。
- **`src_slot` 选择是 K>1 的已知坑**（`:204-227`）：K=1 用 `src_slot = ac-1`（mbt 并行链）；
  K>1 时**强制 slot 0**，注释 `:211-220` 说明 K>1 走 `Q_LEN_OVERRIDE=1` 顺序循环、只有 slot 0
  有效，slots 1..mbt-1 会把 stale/garbage 带进下游 pipeline 并让 K>1 接受率崩。
- draft 写 `tokens[req, step+ac+1+t_id] = drafts[src_slot*K+t_id]`（`:231-236`）；`drafts_prev[t_id]`
  快照（`:250-252`）。
- `accept_hist` 调试历史（`:254-309`）：`atomicAdd(&hist[ac],1)` + 前 16 次迭代的 trace 记录
  （argmax[0..K]、old drafts_prev[0..K-1]）。**注释自陈一段真实 bug 调查**（`:258-287`）：
  K=2 时 step-0 接受率只有 18.6%（K=1 是 45.6%），逐项排除 Q_LEN_OVERRIDE、argmax grid_y、
  cross-step 污染后，device-side trace 定位到 **`paged_attention_sm100` 在 `MAX_TOKENS=3` 下
  slot 0 的 attention 输出与 `MAX_TOKENS=2` 不一致**（相同数学输入、相同 K/V），候选修复点列出
  `attention_sm100.cuh:519` 的 causal mask 公式与 `MMA_ITERS_M=2` 的 `-inf` NaN 传播。这是
  `docs/38` §6 那条 `m16n16k16` `"+f"` 修复的**同一 family**（MMA_ITERS_M>1 的 frag 污染）
  在投机解码上的下游表现。
- 最后 `new_token_nums[req] = ac`（`:312-314`）。

前端 `eagle3_commit_layer`（`persistent_kernel.py:3154-3185`）登记为 `TASK_EAGLE3_COMMIT`
（7 入 2 出，`graph.cc:913-923`），**但 Python 侧无调用者**（§0 headline 3）。

---

## 7. 两条流水：MTP（DeepSeek-V3）vs Eagle3（Qwen3）

两者都建在「draft 循环 + verify + commit」骨架上，但接线程度不同：

| 阶段 | DeepSeek-V3 MTP（`deepseek_v3/builder.py`） | Eagle3（`eagle3/builder.py`） |
|---|---|---|
| 嵌入输入 | `mtp_build_embed_input_layer`（`:1773`，vLLM-aligned） | 直接 `embed_layer`（`:311-316`） |
| draft 循环 | 静态展开 `for step in range(num_draft_steps)`（`:1789+`）：embed→enorm/hnorm→eh_proj→MLA decoder→norm→lm_head→argmax→（probabilistic 则 softmax_gather+float_scatter）→`mtp_token_scatter`（`:1885-1895`） | `for step in range(K)`（`:308-447`）：embed→（step0 fc）→rmsnorm×2→concat→qkv linear→`paged_attention_layer`（K>1 用 `q_len_override=1, tail_offset=K-step`，`:369-383`）→o_proj/residual→mlp→norm→lm_head→argmax→`eagle3_d2t_remap`→`mtp_token_scatter` |
| verify | `mtp_verify_strict_layer`（`:1947`）或 `mtp_verify_probabilistic_layer`（`:2007`，含 `prob_extract_layer`） | 由调用方按注释拼，`build_draft_loop` 内**未拼** |
| commit | `mtp_accept_commit_layer`（`:2037`） | `eagle3_commit_layer` 存在但**无调用者** |

DeepSeek-V3 侧还有一条「概率累积」旁路：main graph 里 `softmax_gather_layer` + `prob_scatter_layer`
（`:2220/2229`）把每迭代 `P_target(input_token)` 写进 `_target_prob_buffer[batch, step]`，
verify 时用 `prob_extract_layer`（`:1993`）按运行期 `offset=step` 切出 `[step+1..step+K]`
（前端 `prob_extract_layer` `persistent_kernel.py:2970-2994`）。这条链路用到了
`TASK_SOFTMAX_GATHER_SM100=282` / `TASK_PROB_SCATTER_SM100=284` / `TASK_PROB_EXTRACT_SM100=286`
——即 `docs/38` §2 采样家族之外的**又一组概率簿记任务**。

`verify_layer_dispatcher`/`draft_forward_layer_dispatcher`（`persistent_kernel.py:2832-2894`）
的 handler 字典只挂 `"promptlookup"`，Eagle3/MTP 走**直接调用 builder**的路径（不经 dispatcher）。

---

## 8. 判据与生产位置

- **C1（单次 launch）**：这些 task 都是常驻 megakernel 内的 device task，**并入**同一 kernel，
  不自己 launch。**贡献 C1**。
- **C2/C4/C5**：本目录是 task 身体，不构成新的协调/调度机制——调度仍由 `docs/06` 的
  worker/scheduler 事件环提供。**不贡献**（同 `docs/29`–`docs/38` 的 device task 走读）。
- **C3（多算子范围）**：投机解码的「draft→verify→commit→step 推进」是**跨多个 task 的段级管线**，
  它被压进单 kernel 的 task graph，是本专题 C3 边界在**推理服务投机路径**上的具体形态。
  **贡献 C3**。
- **与 megakernel 成本侧的关系**：这批 task 是 `docs/04` §8/§10 说的「task 数与事件扇入/扇出」的
  典型来源——验证/commit 是**最细粒度**的 stage（`accepted_count` 一被写出，commit 与
  `new_token_nums` 就能触发运行时步进），融合粒度越细、事件图越大。含义：投机解码的簿记 task
  **不适合再做更细的融合**（本就单线程、几十行），megakernel 的收益主要在**大算子**（MLA/MoE/GEMM）
  侧；簿记 task 保留为独立 device task 是合理折中。

---

## 9. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。MTP/Eagle3 是 SM100 路径，
  本机无 SM100；prompt-lookup 虽 arch-generic，但本文不构建（`ROADMAP.md` B1/B2 记有构建/时序依赖）。
- **include 落差只记不判**：§1/§0 指出 `target_verify.cuh`/`prompt_lookup.cuh` 不在生产
  `task_header` include 集、`../common.h` 在快照里不存在、单测 `using`/`m.def` 被注释——这些是
  **静态事实**，是否导致该路径不可用取决于运行层与构建脚本，本文不下结论。
- **未展开的更深层**：`softmax_gather_sm100.cuh`/`prob_scatter_sm100.cuh`/`prob_extract_sm100.cuh`
  三个概率簿记 device 身体的内部未逐行（只在 §7 给接线坐标）；`paged_attention_sm100` 的
  `MAX_TOKENS=3` slot-0 分歧（§6.3 的 bug 调查）未独立复现，只转述源码注释。
- **验收率数字是源码注释自报**：§6.3 的 45.6%/18.6% 与「B200 GPU 4」等来自 `eagle3_ops.cuh`
  注释，非本文实测。
- **`lookahead`/`eagle3` 的 handler 字典缺失**只记录派发会 `ValueError`，未追它们各自真正入口。

---

## 10. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- `include/mirage/persistent_kernel/tasks/speculative_decoding/`：`target_verify.cuh`（62 行）、
  `prompt_lookup.cuh`（154）、`target_verify_mtp.cuh`（290）、`mtp_token_ops.cuh`（191）、
  `eagle3_ops.cuh`（317）。
- include 集：`persistent_kernel.cuh:48-52`；`tasks/blackwell/task_header.cuh:62-64`。
- 生产布线：`runtime_header.h:102-104/132-135/158-161/171-175/187`；
  `src/kernel/graph.cc:541-552/726-745/873-923`；
  `src/kernel/task_register.cc:1065/1097/1132/2761/2791/5005/5021/5039/5058/5079/5100/5115/5133/5150`。
- 前端：`python/mirage/mpk/persistent_kernel.py:500-533/2769-2900/3000-3200`；
  `python/mirage/mpk/speculative.py:1-56`；`python/mirage/mpk/models/eagle3/builder.py:34-461`；
  `python/mirage/mpk/models/deepseek_v3/builder.py:1740-2075/2200-2240`。
- 单测（未接线见证）：`tests/runtime_python/runtime_kernel_wrapper.cu:13/27/1509-1590/1845`、
  `tests/runtime_python/test_prompt_lookup.py`、`tests/runtime_python/setup.py:10`。
- 前置走读：`docs/14`（前端 layer 统计）、`docs/24`（`TaskType` 目录：114/235/236/273）、
  `docs/26`（DeepSeek-V3/GLM/Inkling/GPT-OSS 与 DFlash/Eagle3 家族）、`docs/34` §3.5（常驻共驻
  device 适配：request_id 而非 blockIdx）、`docs/36`（MLA-MTP attention 家族）、
  `docs/37` §2（薄 task 的 `blockIdx.x==worker id`）、`docs/38` §2.4（`NUM_THREADS=128` 口径）
  与 §6（`m16n16k16` `"+f"` 修复）。
- 相关坑（`ROADMAP.md`「已知坑」）：「常驻共驻 device 适配」族、`docs/38` 的 128/256 线程口径。
