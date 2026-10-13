# MPK Eagle3 draft loop 走读：K=1 的 mbt 并行链、K>1 的 slot-0 退化与「proper mbt-parallel chain」缺口

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/45` 同快照）。
> 本专题的 Eagle3 相关篇目里，`docs/40` 走 device 目录、`docs/43` 走 `eagle3_ops.cuh` 四个 body、
> `docs/44` 走通用 `attention_sm100.cuh`，但**「谁把 K 个 draft step 拼成一张图」**——即
> `Eagle3Builder.build_draft_loop`（`models/eagle3/builder.py:271-449`，此前只在 `docs/43` §10 作为
> 坐标列出、从未逐行）——一直没打开。本文补上这段，并把它与「K>1 强制 `src_slot=0` 的缺口」串成一条线。
> **纯源码走读（资料分析）**：Eagle3 device 头只被 blackwell `task_header.cuh` include（SM100 专属），
> 本机（目标卡 H100/sm90）不可执行，**无新实验**。

---

## 0. 三条 headline

**(A) draft loop 是一张「`for step in range(K)` 展开的单图」**：每个 step 是一整层 draft transformer
（embed → [step0: aux concat + fc] → 双 RMSNorm → 2H concat → QKV linear → attention → o_proj+residual
→ post-LN → gateup → silu_mul → down+residual → final norm → lm_head → argmax partial/reduce →
d2t remap → scatter）。所有中间张量都是 `(mbt, …)` 且**跨 step 复用**，step 间的链靠
`draft_hidden`（`:238`）与 `target_token`（`:265`）接力，是**显式的串行链**而非数据并行。

**(B) K 只改一处：attention**。`K == 1` 用默认 `paged_attention_layer`（处理全部 `mbt` 个输入槽，
产生「mbt 条并行链」，每槽在不同 `ac` 假设下预测一个 token）；`K > 1` 改为每 step
`q_len_override=1, tail_offset=K-step`（`:381-382`），**每个 step 只处理 1 个 query token、只写 1 个
KV**。`mbt` 在这种配置下 = `num_draft_steps + 1`（demo `:267-269` 硬断言），所以 K>1 时 `mbt=K+1`
行里**只有 1 行有效**。

**(C) 缺口 = 「把 (B) 的 1 行有效推广回 mbt 行有效」**：下游 `rmsnorm/linear/argmax/d2t` 全是
`grid_dim=(mbt,…)` 或 `batch_size=mbt`，只有 1 行输入有效 → `all_draft_ids[1..mbt-1, :]` 全是
stale/garbage → `eagle3_commit_kernel` **被迫 `src_slot=0`**（`eagle3_ops.cuh:221`）。device 注释
（`:217-220`）自陈「Until a proper mbt-parallel K>1 chain is implemented」。本文把「proper chain
需要什么」拆成 4 条必要条件，并说清它与 `MAX_TOKENS=3` slot-0 bug family（`docs/38` §6 / `docs/44` §7 /
`docs/43` §4）的历史关系。

---

## 1. 定位与坐标

Eagle3 投机解码在 demo 里是**三段接线**（`demo/qwen3/demo_30B_A3B_eagle3.py:781-893`）：

| 段 | 注册层（demo） | 组件 | 产出 | 消费 |
|---|---|---|---|---|
| verify | `mpk.mtp_verify_strict_layer`（`:821`） | `target_verify_mtp.cuh`（`docs/43` §2 上游） | `accepted_count[mbt,1]` | commit / draft（签名预留） |
| draft | `eagle3.build_draft_loop`（`:833`） | `models/eagle3/builder.py:271-449` | `all_draft_ids[mbt,K]` | commit |
| commit | `mpk.eagle3_commit_layer`（`:864`） | `eagle3_ops.cuh:157-315`（`docs/43` §3） | `tokens/new_token_nums/drafts_prev` | 运行时 `prepare_next_batch`（`docs/43` §6） |

关键接线细节（源码事实）：

- verify 的输入是 `eagle3_drafts_prev`（`attach_input` 快照），**不是** `all_draft_ids`；commit 才
  以 `draft_tokens_new=eagle3._attach_cache["eagle3_all_draft_ids"]` 消费本 iter 的 draft
  （demo `:845/:866`）。跨迭代由 `drafts_prev` 快照承担（`docs/43` §3.2 职责 4）。
- `build_draft_loop` 的 `accepted_count` 形参**签名保留但未消费**（builder `:277-279` 注释：
  "kept in signature for demo compat; not used by the paged_attention path"）——即 K=1 的 draft
  attention 也**不读 `ac`**，只靠 commit 事后按 `ac-1` 选槽。这与 `eagle3_ops.cuh` 注释
  "Slot b's prediction is for position step_N + b + 2" 呼应（`docs/43` §3.3）。

---

## 2. `build_draft_loop` 逐行

### 2.1 维度：`mbt == K+1`

- `self.mbt = mpk.max_num_batched_tokens`（builder `:77`）。
- demo 在构造 spec config 前硬断言（`:267-269`）：
  `assert args.max_num_batched_tokens == args.num_draft_steps + 1`，即 **`mbt = K + 1`**。
  这就是注释里 "K=1 (mbt=2) vs K=2 (mbt=3)"（`eagle3_ops.cuh:273`）的来源。
- 其它维度全部按 `mbt` 分配（`_allocate_intermediates` `:220-269`），例如
  `attn_out (mbt, num_q*head_dim)`（`:245`）、`all_draft_ids (mbt, K)`（`:268-269`）。

### 2.2 每 step 的算子链（builder `:308-447`）

| # | layer | 行 | grid_dim | 备注 |
|---|---|---|---|---|
| 1 | `embed_layer` | `:311-316` | `(1,1,1)` | 输入 = `target_argmax_token`（step0）/ `target_token`（step>0，`:309`） |
| 2 | `concat_layer`(3H) + `linear_layer`(fc) | `:318-329` | `(1,1,1)` / `(tasks,1,1)` | 仅 step0；`step_hidden = hidden_in` |
| 3 | `rmsnorm_layer`×2 | `:334-343` | `(mbt,1,1)` | embed / step_hidden 各一 |
| 4 | `concat_layer`(2H) | `:344-348` | `(1,1,1)` | `[embed_normed, hidden_normed]` |
| 5 | `linear_layer`(qkv) | `:350-354` | `(tasks,1,1)` | → `attn_in (mbt, fused_qkv)` |
| 6 | **`paged_attention_layer`** | `:356-383` | `(max_req, num_kv_heads, 1)` | **K 分支唯一变量**，见 §3 |
| 7 | `linear_with_residual_layer`(o_proj) | `:385-389` | `(H/64,1,1)` | residual = `step_hidden` |
| 8 | `rmsnorm_layer`(post-LN) | `:391-395` | `(mbt,1,1)` | |
| 9 | `linear_layer`(gateup) + `silu_mul_layer` | `:397-406` | `(tasks,1,1)` | |
| 10 | `linear_with_residual_layer`(down) | `:407-411` | `(H/64,1,1)` | 产出 `draft_hidden`（喂下一 step 的 `step_hidden`） |
| 11 | `rmsnorm_layer`(final) + `linear_layer`(lm_head) | `:413-423` | `(mbt,1,1)` / `(tasks,1,1)` | → `logits_hot (mbt, padded_vocab)` |
| 12 | `argmax_partial_layer` + `argmax_reduce_layer` | `:425-434` | `(num_workers,1,1)` / `(1,1,1)` | → `hot_token (mbt,1)` |
| 13 | `eagle3_d2t_remap_layer` | `:436-441` | `(1,1,1)` | → `target_token (mbt,1)` |
| 14 | `mtp_token_scatter_layer` | `:443-447` | `(1,1,1)` | `all_draft_ids[b, step] = target_token[b]`，`batch_size=mbt` |

两条链式接力（源码事实）：
- `step_hidden`：`step0 = hidden_in`（fc 输出，`:330`），`step>0 = draft_hidden`（上一 step 的
  down+residual 输出，`:332`）。`draft_hidden` 正是第 10 步产的（`:409`）。
- `draft_in_token`：`step0 = target_argmax_token`（主模型 argmax，`:309`），`step>0 = target_token`
  （上一 step 的 d2t-remap 输出，`:309`）。**步骤之间没有并行空间**——`step k+1` 的输入就是
  `step k` 的输出。

这与 `docs/43` §3.1 的「全按输入登记、`(5,2)` 编译期重解释」是同一张图的另一半：图里每个 tensor
被 builder **反复复用**（同名 `_new` 缓存，`:111-117`），`draft_hidden`/`target_token` 因「跨 step
既要写又要读」而被显式命名（`:237-238`、`:264-265` 注释）。

---

## 3. K 分支：attention 的两套语义

`paged_attention_layer` 的前端签名把 Eagle3 的两个专用参数明记为「Eagle3 K>1 chain」
（`python/mirage/mpk/persistent_kernel.py:1303-1317`）：

```
params[6]:  q_len_override (Eagle3 K>1 chain)   # :1360
params[7]:  tail_offset    (Eagle3 K>1 chain)   # :1361
```

### 3.1 K=1：默认 attention → mbt 条并行链

builder `:356-368` 走默认分支（`q_len_override=0, tail_offset=0`）。device 侧
（`attention_sm100.cuh`）：

- `num_tokens = Q_LEN_OVERRIDE > 0 ? Q_LEN_OVERRIDE : (last_token_pos - first_token_pos)`（`:117-119`）
  → 处理全部 `mbt` 个输入 token。
- `eagle3_ops.cuh:206-209` 注释：K=1 产生 **mbt 条并行链**，槽 `b` 的预测对应位置
  `step_N + b + 2`，故 commit 选 `src_slot = ac - 1` 去对齐下一轮期望位置。

### 3.2 K>1：`q_len_override=1` → 每 step 只有 1 个 query token

builder `:369-383`：

```
q_len_override=1,
tail_offset=K - step,       # :381-382
```

device 侧连锁（`attention_sm100.cuh`，全部源码事实）：

- `num_tokens = 1`（`:117-119`）。
- `seq_len = (num_pages-1)*PAGE_SIZE + last_page_len − TAIL_OFFSET`（`:128-130`）——即用 `tail_offset`
  把「当前已分配 token」往回收 `K-step`，使本 step 的有效历史 = 到绝对位置 `step+k` 为止。
- 输出基址 `d_output = output_ptr + first_token_pos * O_STRIDE`（`:151-152`）——**写在哪一行由
  `first_token_pos = qo_indptr[request_id]` 决定**（`:111`）。
- 结尾 store loop 的作用域是 `num_tokens * NUM_QO_PER_KV * HEAD_DIM`（`:737-746`）——`num_tokens=1`
  时**只写 1 个 token 的行**。

builder `:284-289` 的 docstring 把设计意图写得很清楚：K>1 时「each step's attention sees all prior
steps' K/V writes via release/acquire fences」——**用自然计算链把 K 个 draft step 串行化**：step k
在绝对位置 `[step+k]` 写 1 个 KV，step k+1 通过 embed step k 的 d2t token 再往前走一格。

### 3.3 「only slot 0」的确切含义

`eagle3_ops.cuh:211-216` 注释：K>1 时 **draft kernel 只写 `attn_out` 的 slot 0**。结合 `:151-152`
与 demo 的单请求生成布局（`qo_indptr[request_id]=0`），本专题推断「有效行 = `first_token_pos` 行，
单请求生成时即第 0 行」——**此为静态推断，未构建跟踪**（见 §8）。

但**下游全是 mbt-wide** 是源码事实：`rmsnorm_layer` 用 `grid_dim=(mbt,1,1)`（builder `:337/:342/…`）、
`argmax_partial_sm100` 内部 `for batch_idx < num_active_tokens`（`argmax_sm100.cuh:102`）、
`argmax_reduce_sm100` 同（`:142`）、`eagle3_d2t_remap_kernel` 用 `b=threadIdx.x` 遍历 `BATCH_SIZE`
（`eagle3_ops.cuh:115-121`）、`mtp_token_scatter_kernel` 同样 `b < BATCH_SIZE`（`mtp_token_ops.cuh:45-48`）。
于是 `all_draft_ids[1..mbt-1, step]` 拿到的是 stale 行算出的 garbage。这正是 commit 里
`src_slot = (K > 1) ? 0 : (ac-1)`（`eagle3_ops.cuh:221`）的来处。

---

## 4. 「proper mbt-parallel K>1 chain」缺口的四项必要条件

把 device 注释（`:217-220`）说的目标——「让 K>1 也有 mbt 条**每槽各自有效**的链」——翻译成可核对的
条件（本专题综合推断，非源码结论）：

1. **attention 必须把 mbt 个输出行都算出来**：K=1 靠 `num_tokens=mbt` 一次算完；K>1 要保留
   `num_tokens=1` 的串行语义（“每 step 只预测 1 个新位置”），就**必须让 mbt 条链的预测值落在
   `attn_out` 的不同行**。`attention_sm100.cuh` 的 `num_tokens`/`d_start`/store-loop 三处都是
   单 token 假设（`:117-119`、`:151-152`、`:737-746`），需扩展成「mbt 个 query token、每个 token
   自己的 `tail_offset`」。device 注释里提到的 MMA `m`-loop（把 `MAX_TOKENS * NUM_QO_PER_KV`
   扩到覆盖 mbt）即指此。
2. **每槽的链要能独立推进**：现在 step 间的链是 `draft_hidden`（单一 `(mbt,H)` 张量，
   builder `:238`）与 `target_token`（单一 `(mbt,1)`，`:265`）。若 mbt 槽代表不同 `ac` 假设，
   它们的「下一位置」不同，`tail_offset` 必须是**per-slot** 的，不能像现在这样一个标量
   `K-step`（`:382`）。
3. **下游簿记必须 per-slot 索引**：`argmax_partial/reduce`（`argmax_sm100.cuh:102/142`）、
   `d2t_remap`（`eagle3_ops.cuh:115-121`）、`scatter`（`mtp_token_ops.cuh:45-48`）现在是
   「一次遍历 mbt」——它们本身没问题（都按 `batch` 那么遍历），问题在上游只有 1 行有效。
   一旦 attention 修好，这些 body **无需改**，链就自然贯通。故缺口的**唯一落点其实是 attention**。
4. **KV 写回不能互相踩**：K>1 的每 step 写 1 个 KV（`docs/43` §3.3 注释、builder `:284-289`）。
   若改成 mbt 槽并行，同一步内 1 请求的 mbt 个槽会**写同一个绝对位置区间**的不同假设，需要
   额外的页/位置分片——这是 K=1 默认分支（一次写 `[step, step+mbt)`）已经在做的，K>1 要复用同一
   机制。

**一句话**：`mbt-parallel chain` 不是「加一个并行维度」，而是「把 K>1 的**串行 step 循环**与 K=1 的
**mbt 并行槽**两种语义合一」——目前二者是 `if K==1 ... else ...`（builder `:356/:369`）的二选一。

---

## 5. 与 `MAX_TOKENS=3` slot-0 bug family 的关系

`eagle3_ops.cuh:264-287` 的根因调查（`docs/43` §4.4 已转述）给出的结论是：K>1 接受率崩塌
（K=2 step-0 18.6% vs K=1 45.6%）**不在 commit/scatter 逻辑，而在 `paged_attention_sm100` 的
`MAX_TOKENS=3`**——「同样数学输入给 slot 0，`MAX_TOKENS=2` 与 `=3` 的输出不一致」（`:277-283`）。
候选修复点两条落在本文件正上方：

- `(a) MMA_ITERS_M=2 garbage propagation`（对应 §4 条件 1 的 m-loop）——`attention_sm100.cuh:103`
  `MMA_ITERS_M = (MAX_TOKENS*NUM_QO_PER_KV + 15)/16`；`MAX_TOKENS=3` 时 `m`-loop 有 `m=1` 尾 tile。
- `(b) causal mask`（对应 `attention_sm100.cuh:556-559`）。

`docs/38` §6 把该 family 钉到 `ampere/mma.cuh` 的 `"+f"` in-place 累加器修复（`attention_sm100.cuh:505-516`
记录下游后果）；`docs/44` §7 把它钉回 `attention_sm100.cuh` 里 `if(m>0) continue` workaround 与
`:556-559` 的因果/窗口掩码。即：**「proper mbt-parallel K>1 chain」的历史阻塞，与 `MAX_TOKENS=3`
的 slot-0 分歧共用同一段 attention 代码**——修 m-loop（条件 1）既是补齐 K>1 链的前提，也是该 bug
family 的候选修复面。这把 `docs/43`（commit 侧）、`docs/38`（MMA）、`docs/44`（通用 attention）
三篇在同一处收口。

---

## 6. 一处静态落差：demo 注释描述的 attention 与 builder 实际调用的不是同一支

demo `:826-832` 的注释写 draft attention 是「new flat-cache `eagle3_draft_chain_step_attn` kernel
with `NUM_TOKENS=mbt`, `USE_AC_OFFSET=False`」，并说 K>1 prefill「gated by a static_assert in the
kernel until the MMA m-loop extension」。但 **builder 实际调用的是 `self.mpk.paged_attention_layer`**
（`:357` / `:370`），其 task 注册名是 `paged_attention`（前端 `persistent_kernel.py:1303`），快照里
**没有 `eagle3_draft_chain_step_attn` 这个名字**（`grep` 全仓零命中）。二者是**两套并存的 design**：

- demo 注释描述的 flat-cache 单 kernel（`NUM_TOKENS=mbt`、`static_assert` 门控 K>1 prefill）；
- builder 实际走的 `paged_attention_layer` + `q_len_override/tail_offset`（每 step 1 token）路径。

本专题只记该静态落差（注释与代码不符），**不判哪一支更新**——判「哪支在生产跑」须追 builder 的
实际调用（即 `paged_attention_layer`），这是本专题反复出现的「按注释/名字推断会错」族
（`docs/16`/`docs/20`/`docs/32` §3/`docs/45` §3.3）在 speculative-decoding 侧的又一实例。

---

## 7. 判据映射

| 判据 | 本段是否贡献 |
|---|---|
| C1 单次 launch | 间接（整个 Eagle3 draft+verify+commit 编进同一个常驻图；builder 只注册 task，launch 由 `PersistentKernel` 统一发） |
| C2 常驻 | 间接（同 C1） |
| C3 多算子范围 | **是**——本文正是「一张图里有了哪些 task、范围边界在哪」的清单（§2.2 表 + §3 分支） |
| C4 设备内协调 | 间接（跨 step/跨 iter 依赖由 MPK event 承担，`docs/06`/`docs/09`） |
| C5 SM 级调度 | 否（builder 层不涉及 worker/scheduler 划分） |

结论：**本文贡献 C3 的「范围与分支」**，与 `docs/43` §8（commit 只贡献 C1+C3）互补——两篇合起来
才把 Eagle3 这一支的图范围讲全。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。Eagle3 device 头只被
  blackwell `task_header.cuh` include，SM100 专属，本机（目标卡 H100/sm90）不可执行。
- **「有效行 = slot 0 / `first_token_pos` 行」是静态推断**：`eagle3_ops.cuh:211-216` 的「only slot 0」
  与 `attention_sm100.cuh:151-152` 的 `first_token_pos` 基址如何精确对应到 `all_draft_ids` 的行，
  未构建跟踪；单请求生成（`qo_indptr[0]=0`）下二者一致是合理推断，但**多请求/非 0 起始 pos 未核**。
- **§4 的四项必要条件是本专题综合推断**，非源码结论；device 注释只写了「Until a proper
  mbt-parallel K>1 chain is implemented」一句。
- **`MAX_TOKENS=3` slot-0 分歧未独立复现**：§5 只做源码事实串联，验收率/根因数字仍为注释自报
  （与 `docs/43` §9、`docs/44` §10 一致）。
- **demo 注释 vs builder 实际调用的落差只记不判**（§6）；未跑 demo。
- **draft loop 的计算图与 `_allocate_intermediates` 的复用**只读到静态注册层；未计 buffer 生命周期
  或 alias（如 `attn_out` 跨 step 覆写是否与目标图其它写者冲突）。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/45` 同快照）。
- builder：`python/mirage/mpk/models/eagle3/builder.py`
  （`__init__:58`；`self.mbt:77`；`_prepare_weights:119`；`_attach_weights:190`；
  `_allocate_intermediates:220`；`all_draft_ids:268-269`；`build_draft_loop:271-449`；
  `for step in range(K):308`；`draft_in_token:309`；fc/step0:318-330；attention K=1:356-368、
  K>1 `q_len_override=1/tail_offset=K-step`:369-383；o_proj+residual:385-389；down+residual:407-411；
  lm_head:418-423；argmax:425-434；d2t:436-441；scatter:443-447）。
- 前端：`python/mirage/mpk/persistent_kernel.py:1303-1412`（`params[6]/[7]` 明记 "Eagle3 K>1 chain"
  `:1360-1361`；`assert q_len_override <= max_tokens_per_request` `:1399-1401`）；
  `paged_attention_layer:1303`；`argmax_partial_layer:2603`；`argmax_reduce_layer:2632`；
  `mtp_token_scatter_layer:2895`；`mtp_verify_strict_layer:3057`；`eagle3_commit_layer:3154`；
  `eagle3_d2t_remap_layer:3184`。
- device：`tasks/blackwell/attention_sm100.cuh`（`Q_LEN_OVERRIDE/TAIL_OFFSET:43-44`；
  `num_tokens:117-119`；`seq_len:128-130`；`first_token_pos:111`；`d_output:151-152`；
  `store loop:737-746`；`"+f"` 注释 `:505-516`；因果/窗口掩码 `:556-559`；`MMA_ITERS_M:103`）。
- device 簿记：`tasks/speculative_decoding/eagle3_ops.cuh`（`src_slot:204-227`；`K=1 mbt parallel
  chains` 注释 `:206-209`；`proper mbt-parallel` 注释 `:217-220`；`MT=3` 根因调查 `:264-287`、
  `:273`；`d2t_remap b-loop:115-121`）；`tasks/speculative_decoding/mtp_token_ops.cuh:37-49`；
  `tasks/blackwell/argmax_sm100.cuh:84-177`（`batch_idx < num_active_tokens` `:102/142`）。
- demo：`demo/qwen3/demo_30B_A3B_eagle3.py`（verify `:821`；draft `:833`；commit `:864`；
  `mbt == K+1` 断言 `:267-269`；flat-cache 注释 `:826-832`；argmax grid `:760-766`）。
- 本专题交叉引用：`docs/40`（投机解码 device 目录）、`docs/42`（概率 MTP 验证链）、
  `docs/43` §3.3/§4.4/§10（commit + `src_slot` 坑 + builder 坐标）、`docs/44` §7/§10（通用 attention +
  `MAX_TOKENS=3` 分歧）、`docs/38` §6（`"+f"` 累加器）。
