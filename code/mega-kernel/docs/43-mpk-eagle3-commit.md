# MPK Eagle3 commit device 身体走读：verify-aware 写回、跨迭代快照与 accept_hist 调试史（P129-1）

> 定位：补掉 `docs/39-mpk-speculative-decoding.md` §9 明留的尾巴——
> 「`eagle3_commit_kernel` 的 device 身体（`eagle3_ops.cuh:157-315`，`docs/39` §6.3 已摘要、
> 未逐行）与 `accept_hist` 调试历史」。也即 `ROADMAP`「下一步候选」第二条剩余。
>
> `docs/39` 只给了 `eagle3_ops.cuh` 四个 body 的**逐段摘要**（§6.1–§6.3），`docs/42-mpk-mtp-verify-chain.md`
> 只把 **MTP（DeepSeek-V3）** 的 scatter → verify → commit 链串起来，**都没有把 Eagle3 这一支的
> commit 身体逐行打开、也没有核对它的四职责与接口的精确对应**。本文做三件事：把四个 body 逐行读
> （§2–§4，重点是 `eagle3_commit_kernel`），核对接线（§5，含三处对 `docs/39` 的修正），并把设备侧
> 自陈的 bug 调查史（`accept_hist` + `MAX_TOKENS=3` slot-0 分歧）整理出来（§4.4）。
>
> 快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`（与 `docs/06`–`docs/42` 同快照）。
> **纯源码走读（资料分析），无新实验**——四个 body 只在 **blackwell** 的 `task_header.cuh` include
> 集里（`docs/39` §1），SM100 专属，本机（目标卡 H100/sm90）不可执行；本文不构建、不运行。

---

## 0. 三条 headline

1. **`eagle3_commit_kernel` 是全仓唯一一个「verify-aware 收尾 task」**：它在一次 device 调用里
   把四件事做完——写回 verified 前缀（`argmax`）、写下一轮 K 个 draft、发布 `new_token_nums` 步进
   信号、把本轮 draft 快照进跨迭代 `drafts_prev`（attach_input，不建图边）。**接口是「5 入 2 出」**
   （`graph.cc:920` 的 `std::make_tuple(5, 2, TASK_EAGLE3_COMMIT, …)`；`register` 侧
   `input_ptrs[0..4]` + `output_ptrs[0..1]`，`task_register.cc:5164-5175`），**不是 `docs/39` §6.3
   写的「7 入 2 出」**——那 7 个是前端 `eagle3_commit_layer` 传给 `customized()` 的 DTensor 数
   （`persistent_kernel.py:3178-3181`），其中 `num_new_tokens`/`drafts_prev` 两个被算作**输出**。
2. **「verify 需要上一轮 draft」这个跨迭代依赖被 `attach_input` 绕过**：`drafts_prev` 声明为
   attach_input（MPK 不把它的生产者当图边），所以 iter N 的 commit 在收尾写它、iter N+1 的
   `mtp_verify_strict` 在开头读它，靠运行时「iter N 全部 task 先于 iter N+1」的批间屏障成立，
   **不违反 MPK「每对 producer/consumer 只一条边」的不变量**（`eagle3_ops.cuh:138-142,238-252`，
   demo 侧注释 `demo/qwen3/demo_30B_A3B_eagle3.py:781-796`）。运行时消费端是
   `prepare_next_batch` 的 `#ifdef MPK_SPEC_DECODE` 分支（`persistent_kernel.cuh:295-316`）：
   `step += new_token_nums[req]`（<1 兜底为 1）。
3. **`accept_hist` 是一段 device 侧自陈的完整 bug 调查史**（`eagle3_ops.cuh:254-309`）：
   原子直方图 + 前 16 迭代的 trace，定位到「K=2 step-0 接受率 18.6% vs K=1 45.6%」的根因是
   **`paged_attention_sm100` 在 `MAX_TOKENS=3` 下 slot 0 输出与 `MAX_TOKENS=2` 不一致**（相同数学
   输入、相同 K/V），候选修复点指向 `attention_sm100.cuh` 的 `MMA_ITERS_M` / causal mask。这段调查
   与 `docs/38` §6 的 `m16n16k16` `"+f"` in-place 累加器修复是**同一 family**——后者的注释（当前快照
   `attention_sm100.cuh:505-523`）明确写「corrupted the committed EAGLE3 token stream at K>=2」。
   本文把两处证据并置（§4.4），并修正 `docs/39`「`eagle3_commit_layer` 在整个 Python 侧无任何
   调用者」——**demo 里有调用者**（`demo/qwen3/demo_30B_A3B_eagle3.py:880`）。

---

## 1. 文件定位与对 `docs/39` 的三处修正

`eagle3_ops.cuh`（317 行）提供四个 `__device__ __forceinline__` body，全部 arch-agnostic 但只被
**blackwell** 的 `task_header.cuh:62` include（`docs/39` §1 表）：

| body | 行 | 提供 | 生产 `TaskType` |
|---|---|---|---|
| `copy_layer_kernel<T,BATCH,HIDDEN>` | `:34-48` | `(B,H)` 连续张量 memcpy | `TASK_COPY=232` |
| `concat_kernel<T,BATCH,HIDDEN,N>` | `:60-80` | N 个 `(B,H)` 沿 dim-1 拼成 `(B,N*H)` | `TASK_CONCAT=233` |
| `eagle3_d2t_remap_kernel<BATCH,DRAFT_VOCAB_REAL>` | `:102-122` | hot 词表 id → target 词表 id | `TASK_EAGLE3_D2T_REMAP=235` |
| `eagle3_commit_kernel<K,BATCH,MAX_SEQ_LEN>` | `:157-315` | verify-aware 写回 + 快照 | `TASK_EAGLE3_COMMIT=236` |

`TaskType` 枚举值核对在 `include/mirage/persistent_kernel/runtime_header.h:132-135`；名字表在
`src/kernel/runtime.cc:1844-1847`。三处对 `docs/39` 的**修正/补记**：

1. **接口数**：`docs/39` §6.3 写「（7 入 2 出，`graph.cc:913-923`）」——实际 `graph.cc:920` 是
   `(5, 2)`；前端层的 7 个 DTensor 里 `num_new_tokens`/`drafts_prev` 是输出（§3.1）。
2. **调用者**：`docs/39` §0 headline 3 / §6.3 写 `eagle3_commit_layer`「在整个 Python 侧无任何
   调用者」——**`demo/qwen3/demo_30B_A3B_eagle3.py:880` 就是调用者**（属 `demo/` 演示套件，不在
   `python/mirage/mpk/models/` 的 builder 里）。准确说法应是「`models/eagle3/builder.py` 的
   `build_draft_loop` 不调它，但 demo 组装路径调它」。
3. **行号漂移**：`eagle3_ops.cuh:281` 注释把 causal-mask 公式指向 `attention_sm100.cuh:519`，当前
   快照 `:519` 落在 MMA 内循环注释处；真正与 `MAX_TOKENS`/`num_tokens` 相关的因果/有效长度计算在
   `attention_sm100.cuh:117,131-137,385`（§4.4）。只记行号漂移，不判语义。

---

## 2. 三个「簿记搬运」body（逐行）

### 2.1 `copy_layer_kernel`（`:34-48`）

纯网格跨步 memcpy：`total = BATCH*HIDDEN`，`for (i = tid; i < total; i += blockDim.x)`（`:45-47`）。
用途注释（`:31-33`）：Eagle3 要把 target 的**中间层 hidden** 抓进独立 aux buffer——因为 MPK 的
per-layer 中间张量**跨层复用**（同名中间张量会被后续层覆写），所以必须显式拷一份。`register_copy_task`
（`task_register.cc:5100-5113`）把 dtype 硬编码成 `bfloat16`（`code.e("...<bfloat16, $, $>("`)，
即该 task 只用于 bf16 hidden，不做泛型。

### 2.2 `concat_kernel`（`:60-80`）

按 `output[b, k*H .. (k+1)*H) = inputs[k][b, :]`（注释 `:54`）实现 dim-1 拼接：

```
for (i = tid; i < BATCH*HIDDEN; i += blockDim.x) {
    b = i / HIDDEN; d = i % HIDDEN; out_base = b*N*HIDDEN;
    #pragma unroll
    for (k = 0; k < N; k++) output[out_base + k*HIDDEN + d] = input_ptrs[k][i];
}
```

`input_ptrs` 是 `void const *const *`（设备侧指针数组，`:62`），`register_concat_task`
（`task_register.cc:5115-5131`）直接传 `task_desc->input_ptrs`（`:5128`）。**每一趟循环对 N 个输入
各写一次**（写放大 N 倍，但读是连续 `in[i]`）——Eagle3 两种用法（注释 `:56-59`）：`N=3` 把
h_low/h_mid/h_high 拼成 `3H` 送 `eh_proj`（fc）；`N=2` 把 `(embed, hidden)` 拼成 `2H` 作 draft QKV 输入。

### 2.3 `eagle3_d2t_remap_kernel`（`:102-122`）

**一线程一批元素**（`b = threadIdx.x`，`:113`）：把 draft 的小热词表 id 映射回 target 全词表 id：

```
target[b] = hot_id + d2t[hot_id];     // signed int64 加法（sglang 约定，:87）
```

越界保护（`:116-120`）：`hot_id < 0 || hot_id >= DRAFT_VOCAB_REAL` 时写 **0 哨兵**——来源是
`lm_head` 为满足 TMA 16B 对齐而做的**行 padding**（draft vocab 32000→32256），`argmax` 可能落进
padding 行；写 0 后 verify 会拒绝、正常提交 bonus。注意 `register_eagle3_d2t_remap_task`
（`task_register.cc:5133-5148`）只把 `BATCH_SIZE, DRAFT_VOCAB_REAL` 作模板参数，**`d2t` 表长度
由运行期张量维度决定**（前端 `draft_vocab_real=d2t_table.dim(0)`，`persistent_kernel.py:3191`）。

---

## 3. `eagle3_commit_kernel`（`:157-315`）逐行

### 3.1 签名与接口：5 入 2 出的精确对应

函数签名（`:157-168`）有 **8 个显式指针参数**（`tokens_buffer / argmax_out / draft_tokens_new /
accepted_count / step / prompt_length / new_token_nums / drafts_prev / accept_hist` 加
`request_id`）。但 `register` 侧（`task_register.cc:5161-5176`）的映射是：

| 形参 | register 表达式（行） | 来源类别 |
|---|---|---|
| `argmax_out` | `input_ptrs[0]`（`:5165`） | 图输入（来自 `argmax_reduce`） |
| `draft_tokens_new` | `input_ptrs[1]`（`:5166`） | 图输入（来自 `mtp_token_scatter`） |
| `accepted_count` | `input_ptrs[2]`（`:5168`） | 图输入（来自 `mtp_verify_strict`） |
| `tokens_buffer` | `input_ptrs[3]`（`:5164`） | 图输入（in-place 写穿） |
| `accept_hist` | `input_ptrs[4]`（`:5174`） | attach_input（debug，原子写穿） |
| `step` | `runtime_config.step`（`:5169`） | **全局**（不是 task 的图输入） |
| `prompt_length` | `runtime_config.prompt_length`（`:5170`） | **全局** |
| `new_token_nums` | `output_ptrs[0]`（`:5171`） | 图输出 |
| `drafts_prev` | `output_ptrs[1]`（`:5172`） | 图输出 |
| `request_id` | `task_desc->task_metadata.request_id`（`:5175`） | 元数据 |

这正是 `graph.cc:920` `(5, 2)` 的来处：**前 5 个 `input_ptrs` 是图输入、后 2 个 `output_ptrs`
是图输出**；`step`/`prompt_length` 走 `runtime_config` 全局（不占 `input_ptrs` 槽，故签名 8 指针
≠ 7 图张量）。前端 `eagle3_commit_layer`（`persistent_kernel.py:3154-3182`）先对 7 个 DTensor 逐个
`tb_graph.new_input(..., -1, -1, -1, True)`（`:3171-3177`，全按输入登记），再由
`task_config` 的 `(5,2)` 在编译期把它们**重解释**为「前 5 入 / 后 2 出」。

### 3.2 四职责

注释 `:124-156` 列四职责，实现按顺序：

1. **写 verified 前缀**（`:197-202`）：线程 `t_id < ac` 时 `tokens[req, step+1+t_id] = argmax[t_id]`，
   且 `pos >= prompt_len` 才写（不覆写 prompt）。`ac = accepted_count[0]`（`:193`，**批元素 0**，
   同 `docs/42` §C 契约），来自 `verify_strict` = `final_accepted + 1`，故 `ac ∈ [1, K+1]`。
2. **写下一轮 K 个 draft**（`:231-236`）：`tokens[req, step+ac+1+t_id] = drafts[src_slot*K+t_id]`，
   `t_id < K`，同样的 `pos >= prompt_len` 守卫。
3. **发布步进信号**（`:311-314`）：`t_id == 0` 时 `new_token_nums[req] = ac`。
4. **跨迭代 draft 快照**（`:244-252`）：先把**旧的** `drafts_prev[0..K-1]` 抄进寄存器
   `old_drafts_prev[8]`（供 trace），再 `drafts_prev[t_id] = drafts[src_slot*K+t_id]`（`t_id < K`）。

### 3.3 `src_slot`：K=1 vs K>1 的已知坑（`:204-227`）

```
int src_slot = (K > 1) ? 0 : (ac - 1);
if (src_slot < 0) src_slot = 0;
if (src_slot >= BATCH_SIZE) src_slot = BATCH_SIZE - 1;
```

- **K=1**：默认 attention 处理全部 `mbt` 个输入槽，每槽在不同「ac 假设」下预测一个 token；槽 `b`
  的预测对应位置 `step_N + b + 2`，故 commit 选 `src_slot = ac-1` 去对齐下一轮期望位置（注释 `:206-209`）。
- **K>1**：走 `Q_LEN_OVERRIDE=1` 的顺序循环，**draft kernel 只写 `attn_out` 的 slot 0**
  （下游 rmsnorm→MLP→argmax→d2t 链条每个都是 `grid_dim=(mbt,…)`，会从 slot 1..mbt-1 带进
  stale/garbage 写进 `all_draft_ids[1..mbt-1, :]`）。故**强制 `src_slot=0`**（注释 `:211-220`）：
  选那些行会为下一轮提交 garbage、并让 K>1 接受率崩塌。`ac>=2` 时链条位置会错位 `(ac-1)`，
  但「slot 0 的链至少是有效预测」。

### 3.4 `request_id` 与批元素契约

`req = request_id`（`:189`），`step`/`prompt_length` 均为 `[MAX_REQ]` 按 `req` 取（`:191-192`），
`tokens` 写 `req*MAX_SEQ_LEN+pos`——**这一支是按请求索引的**。但 `ac = accepted_count[0]`
（`:193`）**从 0 索引**，`drafts_prev[t_id]`（`:251`）**平铺无 `req` 偏移**——与
`verify_strict` 读 `draft_ids[i]` 从 0 起（`target_verify_mtp.cuh:51-72`）一致。即：**同一 commit task
内两套索引并存**（`tokens`/`step` 按 `request_id`，`accepted_count`/`drafts_prev` 固定批元素 0），
是 `docs/42` §C「verify/commit 家族无 `request_id`、从 0 索引」在 Eagle3 侧的同一现象。

---

## 4. `accept_hist`：device 侧自陈的调试史（`:254-309`）

### 4.1 直方图（`:288-290`）

`t_id==0 && accept_hist_ptr != nullptr` 时 `atomicAdd(&hist[ac], 1)`——bin `i` 计「本 iter `ac==i`」
的迭代数，`ac ∈ [1, K+1]`，bin 0 保留（sanity）。buffer 由 host 分配、`attach_input` 传入
（demo `:877-879`，`torch.zeros(256, int32)`）。

### 4.2 trace 布局（`:291-308`）

```
[0 .. K+1]        直方图 bin
[K+2]             trace 计数器（atomicAdd 写者序号）
[K+3 .. ]         每条记录 2K+2 个 int：(ac, argmax[0..K], old_drafts_prev[0..K-1])
复现上限 MAX_TRACE_ITERS = 16（内核里硬编码）
```

`trace_idx = atomicAdd(&hist[K+2], 1)`（`:298`），`trace_idx < 16` 才写（`:299`）；记录里
`argmax[i] & 0xFFFFFFFF` / `old_drafts_prev[i] & 0xFFFFFFFF` 只截低 32 位（`:303,306`，够 Qwen3 词表）。
`old_drafts_prev[8]`（`:244`）注释「K ≤ 8 well bounded for spec decode」——**trace 复现依赖 K≤8**。

### 4.3 自陈测量值（注释 `:258-262`，**源码自报、非本文实测**）

```
K=1: hist=[0,184,154]    → step-0 accept 45.6%
K=2: hist=[0,259,41,18]  → step-0 accept 18.6% (期望 ~46%)、step-1 cond 30.5%、per-draft 12.1%
（B200 GPU 4，Qwen3-30B-A3B，prompt-len 39，gen-len ~400）
```

### 4.4 根因调查（注释 `:264-287`，逐条排除）

- **非** `Q_LEN_OVERRIDE=1 + TAIL_OFFSET=K-step`：K=2 step 0 用默认 attention（mbt=3、无 override）
  仍只有 19.7%。
- **非** `argmax_partial` 的 `grid_dim=(workers,1,1)`：argmax kernel 内部按 `num_active_tokens=mbt`
  迭代，与 `grid_y` 无关，全部 mbt 槽都被算。
- **非** cross-step 状态污染：scatter 按列写（step k 写第 k 列），step 顺序被 MPK 串行化。
- **根因隔离**（device trace，2026-05-23）：bug 在 `paged_attention_sm100` 的 `MAX_TOKENS=3`，
  **不在** eagle3 commit/scatter。同一 prompt 上逐字节比 K=1（mbt=2）vs K=2（mbt=3）：iter 0
  `argmax[0]=12` 相同；iter 1 输入 `tokens[step]=12` 相同、past K/V 相同（causal mask 应让 slot 0
  与 slot 1/2 无关），但 K=1 得 `argmax[0]=17`、K=2 得 `argmax[0]=525`。
- **候选修复点**：`attention_sm100.cuh` 的 (a) `MMA_ITERS_M=2` 的 `-inf` `m_local` NaN 传播、
  (b) causal mask 公式、(c) `seq_len-num_tokens` 左移一位的 cache 读写重叠。

**并置证据**：当前快照 `attention_sm100.cuh:505-523` 的注释与上面同一 family，且**已给出修复**：
`if (m > 0) continue;` 的旧 workaround「corrupted the committed EAGLE3 token stream at K>=2（bonus
token at ac=K+1 读了未算的 slot）」，根因是 mma 累加器用 write-only `"=f"` 输出 + 独立 `"f"` 输入
别名同寄存器，编译器可跨 unrolled m 迭代复用 fragment 寄存器而交叉污染；**改成单一读写 `"+f"`
累加器操作数**（`attention_sm100.cuh:523` 的 `mma_m16n16k16_bf16bf16bf32(x_frag_f[m], q_frag,
kt_frag, x_frag_f[m])`）后每个 `x_frag_f[m]` 有独立 def-use 链、无 tile 被跳过。这正是 `docs/38` §6
登记的那条 load-bearing 修复在投机解码上的**下游表现**。另：`MAX_TOKENS` 默认 8
（`attention_sm100.cuh:52`），`MMA_ITERS_M = (MAX_TOKENS*NUM_QO_PER_KV+15)/16`（`:103`）。

---

## 5. 接线：前端层 / register / graph / demo（四层）

- **前端层**：`eagle3_commit_layer`（`persistent_kernel.py:3154-3182`）——参数签名带
  `target_argmax/draft_tokens_new/accepted_count/tokens_buffer/num_new_tokens/drafts_prev/accept_hist`
  七个 DTensor；`params=[num_draft_tokens, batch_size, max_seq_len]`（`:3169`）；`register_task` 名
  `"eagle3_commit"`（`:3182`）。
- **register 层**：`register_eagle3_commit_task`（`task_register.cc:5150-5177`）——模板参数
  `<K, batch_size, max_seq_len>`，按 §3.1 的映射发 `input_ptrs[0..4]`/`output_ptrs[0..1]`。
- **graph 层**：`graph.cc:913-920`——`task_config[op] = (5, 2, TASK_EAGLE3_COMMIT, variant_id)`。
- **demo 组装**：`demo/qwen3/demo_30B_A3B_eagle3.py:781-893`——把 `argmax_out`（而非
  `verified_output`）作 `target_argmax` 传入（单一 producer/consumer 边设计，`:862-865`），
  `draft_tokens_new=eagle3._attach_cache["eagle3_all_draft_ids"]`（`:882`），
  `grid_dim=(max_num_batched_requests,1,1)`/`block_dim=(128,1,1)`（`:888-889`）。demo 注释
  `:791-796` 进一步说明 iter 0 时 `drafts_prev` 零初始化 → verify 接受 0、只提交 1 个 bonus，故
  **prefill 必须在首次 `mpk()` 前把 prompt 预写进 `tokens`**。

> 静态观察（只记不判）：demo 注释 `:839` 提到 draft attention 是「新 flat-cache 的
> `eagle3_draft_chain_step_attn` kernel」，但该符号在快照里**不存在**（`grep -rn` 只命中这行注释）；
> demo 实际走 `eagle3.build_draft_loop`（`:854`）→ `paged_attention_layer`
> （`builder.py:356-383`，K>1 用 `q_len_override=1, tail_offset=K-step`）。注释与代码不同步。

---

## 6. 运行时消费端：`prepare_next_batch` 的 spec-decode 分支

`persistent_kernel.cuh:280-316` 的 `prepare_next_batch`：`#ifdef MPK_SPEC_DECODE`（`:295`）
（编译宏由前端 `persistent_kernel.py:348` 在 spec 模式下追加 `-DMPK_SPEC_DECODE`）：

```
if (step >= prompt_len) {              // decode 阶段
    step_advance = config.new_token_nums[request_id];   // eagle3_commit 写的
    if (step_advance < 1) step_advance = 1;             // 首 iter 兜底
} else { /* prefill：原语义，从 output_tokens 拷 num_tokens 个 */ }
config.step[request_id] = step + step_advance;
```

即 **`eagle3_commit` 是「一次消费」链的终点**：它写的 `new_token_nums` 被运行时读来推进 `step`。
`#ifndef MPK_SPEC_DECODE` 的 else 分支（`:317-327`）用 `num_tokens` 推进——对照可见 spec 路径唯一
改动的就是「步进量来源从 `qo_indptr` 差值换成 commit 发布的 `new_token_nums`」。

---

## 7. 与 MTP（DeepSeek-V3）commit 的对照

| 维度 | DeepSeek-V3 MTP（`docs/42`） | Eagle3（本文） |
|---|---|---|
| commit body | `mtp_accept_commit_kernel`（`target_verify_mtp.cuh:255-288`） | `eagle3_commit_kernel`（`eagle3_ops.cuh:157-315`） |
| 职责数 | 写 tokens + 发 `num_new_tokens`（2 项） | 写 verified 前缀 + 写 K draft + 步进信号 + 跨迭代快照（4 项） |
| 跨迭代 draft | 无（verify 直接在 `docs/42` 的 `prob_extract` 链上） | **新增 `drafts_prev` attach_input 快照** |
| `src_slot` 选择 | 无 | K=1 `ac-1` / K>1 强制 0（`:221`） |
| debug 插桩 | 无 | `accept_hist` 原子直方图 + trace（`:288-309`） |
| 接口 | 见 `docs/42` 表 | `(5,2)`（`graph.cc:920`） |

**含义**：Eagle3 commit 比 MTP commit 多做两件事，都是为了适配 Eagle3「draft 链 + 跨迭代 verify」
的形态——`drafts_prev` 快照解跨迭代依赖、`src_slot` 解 K=1/K>1 两套 attention 输出布局。
两者都属 `docs/39` §8 说的「簿记型 task（不吃 TC、几十~几百行、单线程收尾）」，是 megakernel
成本侧（`docs/04` §8/§10 的 task 数 / 事件扇入扇出）的典型来源。

---

## 8. 判据

- **C1（单次 launch）**：四个 body 都是常驻 megakernel 内的 device task，**并入**同一 kernel，
  不自己 launch。**贡献 C1**。
- **C2/C4/C5**：task 身体，不构成新的协调/调度机制——调度仍由 `docs/06` 的 worker/scheduler
  事件环提供。**不贡献**（同 `docs/29`–`docs/42` 的 device task 走读）。
- **C3（多算子范围）**：Eagle3「draft 循环 → verify_strict → commit → step 推进」是段级管线，
  被压进单 kernel 的 task graph，是 C3 边界在推理服务投机路径上的具体形态。**贡献 C3**。
- **成本侧锚点**：`eagle3_commit` 是「最细粒度收尾 stage」的又一例（`accepted_count` 一写出，
  commit 与 `new_token_nums` 即可触发运行时步进），融合粒度越细、事件图越大；簿记 task 保留为独立
  device task 是合理折中（同 `docs/39` §8 结论）。

---

## 9. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。`eagle3_ops.cuh` 只被
  blackwell `task_header.cuh` include（SM100 专属），本机（目标卡 H100/sm90）不可执行。
- **验收率/根因结论是源码注释自报**：§4.3 的 45.6%/18.6%、§4.4 的「B200 GPU 4」「2026-05-23」
  等来自 `eagle3_ops.cuh` 注释，**非本文实测**；`MAX_TOKENS=3` slot-0 分歧未独立复现。
- **对 `docs/39` 的修正只涉静态事实**：「7 入 2 出」→「5 入 2 出」、「无任何调用者」→「demo 有
  调用者」、causal-mask 行号漂移——只记事实，不判 `docs/39` 其它结论对错。
- **未展开的更深层**：`paged_attention_sm100`（`attention_sm100.cuh` 750 行）内部未逐行（只引其
  `MAX_TOKENS`/`MMA_ITERS_M`/`"+f"` 三处注释）；`concat_kernel` 的大张量写放大量化未做实测。
- **demo 组装只读到组装层**：`demo_30B_A3B_eagle3.py` 是 1018 行演示程序，本文只读其 Eagle3 接线段
  （`:781-893`）与 trace 打印段（`:993-1016`），未跑。

---

## 10. 参考坐标（可核对）

- device body：`include/mirage/persistent_kernel/tasks/speculative_decoding/eagle3_ops.cuh`
  （`copy_layer_kernel:34-48`、`concat_kernel:60-80`、`eagle3_d2t_remap_kernel:102-122`、
  `eagle3_commit_kernel:157-315`）。
- 上游对比：`target_verify_mtp.cuh:44-91`（`verify_strict`，`docs/39` §4.1 / `docs/42` §3）。
- 下游 attention：`include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh:45-52,103,505-523`。
- 接线：`src/kernel/task_register.cc:5100-5177`；`src/kernel/graph.cc:898-921`；
  `include/mirage/persistent_kernel/runtime_header.h:132-135`；`src/kernel/runtime.cc:1844-1847`。
- 前端/运行时：`python/mirage/mpk/persistent_kernel.py:3154-3182,348`；
  `include/mirage/persistent_kernel/persistent_kernel.cuh:280-316`。
- demo 组装：`demo/qwen3/demo_30B_A3B_eagle3.py:781-893,993-1016`。
- builder（draft 循环）：`python/mirage/mpk/models/eagle3/builder.py:220-449`。
