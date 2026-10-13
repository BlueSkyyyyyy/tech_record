# MPK MTP 验证链走读：scatter → verify → commit 的消费链与批元素契约（P128-1）

> 定位：本文补掉 `docs/41-mpk-prob-bookkeeping.md` §6 明留的两条尾巴——
> （1）「`mtp_float_scatter` 的 device 身体（`mtp_token_ops.cuh`）未逐行」；
> （2）「`target_verify_*` 如何消费 `prob_extract` 的输出（拒绝采样比较 `P_target > u·P_draft`）
> 只按注释转述」。也即是 `ROADMAP`「下一步候选」第二条。
>
> `docs/39-mpk-speculative-decoding.md`（投机解码 device 层）与 `docs/41`（概率簿记）已分别给
> **每个 device 身体**的逐段摘要和**概率簿记三兄弟**的布线，但**没有把「draft 概率 → target 概率
> → 拒绝采样 → 提交」这条链的 producer/consumer 数据流串起来**，也没有核对 **verify/commit 家族
> 的索引契约**。本文做这两件事：先给一张端到端数据流图（§1），再把 `mtp_token_ops` 四个身体
> （§2）、`target_verify_*` 的三种消费语义（§3）、builder 侧的实际接线与落差（§4）逐处核对，并
> 指出这三段共有的一个**批元素契约**（§5，`docs/39`/`docs/41` 摘要里没有的一条）。
>
> 快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`（与 `docs/06`–`docs/41` 同快照）。
> **纯源码走读（资料分析），无新实验**——整条链只登记在 SM100 路径（`tasks/blackwell/task_header.cuh`
> include），且特性上下文（DeepSeek-V3 MTP 投机解码 verify 窗口）本机不具备，本文不构建、不运行。

---

## 0. 三条 headline

1. **概率 MTP 拒绝采样是一条「两处生产、一次消费」的跨迭代链**（`docs/41` 只给了每段的坐标，
   没给整条流水）：draft 侧 `softmax_gather → mtp_float_scatter`（编译期槽号 `step`）把 `P_draft`
   写进 `_draft_prob_buffer[:, step]`；main graph 侧 `softmax_gather → prob_scatter`（**运行期**
   `step` 计数器）把 `P_target(input_token)` 写进 `_target_prob_buffer[:, step]`；下一轮 verify 时
   `prob_extract`（device 侧**硬编码 `offset+1`**）切出 `target_probs[0..K-1]`，连同
   `_draft_prob_buffer` 一起送进 `target_verify_probabilistic`（`P_target > u·P_draft`）→
   `accepted_count`/`output_tokens` → `mtp_accept_commit`。**同一「按位置累积」语义在 draft 侧是
   编译期时间维、在 target 侧是运行期时间维**（`docs/41` §0 headline 3 逐字说了「两套载体」，
   本文把两条载体**在数据流上接成一条链**并核对接口张量）。
2. **`mtp_token_ops.cuh` 的四个身体分属两种索引契约**（`docs/39` §5 只逐段摘了算法，没并置）：
   `mtp_token_scatter`/`mtp_float_scatter` 用 `b = threadIdx.x` **批并行**（写满 `[BATCH, NUM_SLOTS]`
   全部行）；`mtp_prepare_verify_input`/`mtp_build_embed_input` 用 `task_metadata.request_id`
   **按请求**索引（写 `tokens[req*MAX_SEQ_LEN+…]`）。前者靠 `slot_idx` 编译期常量、后者靠 `step`
   运行期计数器——**同目录内两套 batch 索引哲学**并存。
3. **verify/commit 家族没有请求索引**（`docs/39` §4/§41 摘要未点出）：`target_verify_{greedy,strict,
   probabilistic}`、`mtp_accept_commit`、`eagle3_commit` 的签名**都不带 `request_id`**，而且
   `draft_ids[i]`/`target_ids[i]`/`output_tokens[i]`/`accepted_count[0]`/`current_position[0]`
   **一律从 0 索引**——即无论 `grid_dim=(mbt,1,1)` 起多少 block，每个 block 都算同一份「第 0 个
   批元素」。对照同一目录里 `mtp_prepare_verify_input` 显式取 `request_id`（注释 `mtp_token_ops.cuh:101-104`
   自陈「不是 `blockIdx.x`」），这条差异是**静态事实**。本文只记录契约、不下「是不是 bug」的结论。

---

## 1. 端到端数据流：draft 概率 / target 概率 / 拒绝采样 / 提交

把 `deepseek_v3/builder.py` 的三段（draft 循环 `:1782-1899`、main graph 的概率旁路 `:2204-2235`、
verify+commit `:1944-2040`）拼成一条链，标出**张量名 → 生产 task → 存储槽 → 消费 task**：

```
                      ┌─ draft 侧（每步静态展开 for step in range(num_draft_steps)）──────────┐
   mtp_step{step}_logits ──softmax_gather_layer(:1876)──> draft_prob_current [mbt,1]
                              └──mtp_float_scatter_layer(:1885, slot_idx=step)──> _draft_prob_buffer[:, step]
   draft_token_ids ──────────mtp_token_scatter_layer(:1891, slot_idx=step)───> all_draft_ids[:, step]

                      ┌─ main graph 侧（每次 verify forward）────────────────────────────────┐
   input_tokens ─┐
                 ├─softmax_gather_layer(main, :2220)──> _target_prob_current [mbt,1]
   lm_head_out ──┘         └──prob_scatter_layer(:2229, step_counter=step)──> _target_prob_buffer[:, step]

                      ┌─ verify 侧（下一轮）───────────────────────────────────────────────┐
   _target_prob_buffer ─prob_extract_layer(:1993, offset=step)──> target_probs [mbt,K]
   _draft_prob_buffer ─┐
   all_draft_ids       ├─mtp_verify_probabilistic_layer(:2007)──> accepted_count, verified_output_tokens
   target_token_ids    │      (P_target > u·P_draft，LCG 采样)
   rng_seed           ─┘
                        └─mtp_accept_commit_layer(:2037)──> new_position, final_output, num_new_tokens
```

逐环节锚点（本节核对）：

| 环节 | 生产 device 身体 | 登记 | 前端 layer | builder 调用 |
|---|---|---|---|---|
| draft 概率切片 | `mtp_float_scatter_kernel` `mtp_token_ops.cuh:54-64` | `TASK_MTP_FLOAT_SCATTER=285` | `persistent_kernel.py:2974-2990` | `:1885` |
| draft token 切片 | `mtp_token_scatter_kernel` `mtp_token_ops.cuh:37-49` | `TASK_MTP_TOKEN_SCATTER=273` | `:2895-2910` | `:1891` |
| target 概率累积 | `prob_scatter_task_impl` `prob_scatter_sm100.cuh:29-48` | `TASK_PROB_SCATTER_SM100=284` | `:2992-3008` | `:2229` |
| target 概率切片 | `prob_extract_task_impl` `prob_scatter_sm100.cuh:54-76` | `TASK_PROB_EXTRACT_SM100=286` | `:3010-3027` | `:1993` |
| 拒绝采样 | `target_verify_probabilistic_kernel` `target_verify_mtp.cuh:112-174` | `TASK_MTP_VERIFY_PROBABILISTIC=283` | `:3029-3055` | `:2007` |
| 提交 | `mtp_accept_commit_kernel` `target_verify_mtp.cuh:255-288` | `TASK_MTP_ACCEPT_COMMIT=272` | `:3077-3100` | `:2037` |

**链的时序锚点**（`docs/41` 只提了 `prob_extract` 的 `+1`）：`prob_scatter` 用运行期 `pos = step_ptr[b]`
写 `buffer[b, pos]`（`prob_scatter_sm100.cuh:42-46`），而 `prob_extract` 用 `offset + 1`
（`prob_scatter_sm100.cuh:68`，注释「verify starts at step+1」）读 `buffer[b, offset+1 + tid]`。
即**写入端用 `step`、读出端用 `step+1`**——整条 target 概率链的「窗口位移」由 device 侧一硬编码
`+1` 承担，而不是由调用方传 `step+1`。draft 侧则**没有这个位移**：`mtp_float_scatter` 写
`_draft_prob_buffer[:, step]`，verify 直接 `draft_probs[i]`（`target_verify_mtp.cuh:145`），
即 draft 位置 `i` 与 target 位置 `i` 一一对齐（都由 verify 从 0 起读）。

---

## 2. `mtp_token_ops.cuh` 四个身体的索引契约

文件 191 行、4 个 `__device__` 函数（`docs/39` §5 已逐段摘过算法，本节补**索引契约**与
`docs/41` §6 未逐行的 `mtp_float_scatter`）。

### 2.1 两个 scatter：`b = threadIdx.x` 批并行（`:37-64`）

```cuda
template <int BATCH_SIZE, int NUM_SLOTS, int SLOT_IDX>
__device__ __forceinline__ void mtp_float_scatter_kernel(src, dst) {   // :54-64
  int b = threadIdx.x;
  if (b < BATCH_SIZE) dst[b * NUM_SLOTS + SLOT_IDX] = src[b];
}
```

- **批并行**：`b` 直接取 `threadIdx.x`，`blockDim.x ≥ BATCH_SIZE` 时一次写满所有行；
  调用方 `block_dim=(128,1,1)`（`:1887`），要求 `mbt ≤ 128`。
- **`SLOT_IDX` 是编译期常量**（模板参数，`:37`/`:54`），由 Python 侧 `for step in range(...)`
  静态展开（`:1883`/`:1885`）提供——这正是 `docs/39` §5 记的「编译期槽号」。`num_slots` 也是
  编译期常量（`NUM_SLOTS`），故 `dst` 的 per-row stride 编译期可算。
- **读取端无界检查**：`src[b]` 只受 `b < BATCH_SIZE` 保护，不校验 `src` 实际长度；
  `dst[b*NUM_SLOTS+SLOT_IDX]` 同理。契约由调用方传对 `BATCH_SIZE`/`NUM_SLOTS` 保证。
- `mtp_token_scatter`（`:37-49`）与 `mtp_float_scatter`（`:54-64`）**逐字节同构**，只差载体
  （`long long` vs `float`）。登记语句也并列（`task_register.cc:2791-2806` float、
  `:5039-5056` token），`graph.cc` 都是 `make_tuple(1,1,...)`（`:736-739` float、`:881-884` token）。

### 2.2 两个 request 级身体：`request_id` 索引（`:84-129` / `:160-189`）

- `mtp_prepare_verify_input_kernel`（`:84-129`，`docs/39` §5 已摘）：`req = request_id`
  （`:104`），写 `tokens[req*MAX_SEQ_LEN + cur_step+1]`（`:111`）与 `tokens[req*MAX_SEQ_LEN +
  cur_step+2+t_id]`（`:122-127`）；`num_new_tokens[req]`（`:118`）。**带界 clamp**：
  `max_new = MAX_SEQ_LEN - cur_step - 1`（`:114`）、`num_new_tokens = min(NUM_DRAFT+1, max_new)`。
- `mtp_build_embed_input_kernel`（`:160-189`，`docs/39` §5 已摘）：`req = request_id`（`:175`），
  但**写出的是 `mtp_input[i]`（`:187`），`i` 是位置不是 `req*BATCH+i`**——即它写的是
  `[BATCH, 1]` 的**整条向量**（`mtp_step0_input_tokens`，`:1756-1759`），而读的是
  `tokens[req*MAX_SEQ_LEN + pos]`（`:183`）。`grid_dim=(max_num_batched_requests,1,1)`（`:1776`）
  下多个 block 会各自写同一 `mtp_input` 向量——**一条静态观察**（§5 同族），只记录不加判决。
- 两者的 `task_register` 都传入 `task_desc->task_metadata.request_id`（`task_register.cc:5074` /
  `:5094`），`graph.cc` 分别 `(4,1)`（`:885-889`）与 `(1,1)`（`:890-896`）。

**一句话**：同目录内 2 个 scatter 是「批维并行 + 编译期槽」，2 个 prepare/embed 是「请求维 + 运行期
`step`」——`docs/39` §5 逐段描述时未并置这两套 index 哲学。

---

## 3. `target_verify_*` 如何消费 `prob_extract` 的输出（`docs/41` §6 tail 2）

`target_verify_mtp.cuh` 三段验证各有一套「接受语义」，其中只有 **probabilistic** 真正消费
`prob_extract` 切出的 `target_probs`。

### 3.1 `target_verify_probabilistic_kernel`（`:112-174`）——唯一吃概率的一段

签名 7 参（5 入 2 出，`graph.cc:731-735` `make_tuple(5,2)`）：`draft_token_ids, target_token_ids,
target_probs, draft_probs, seed → accepted_count, output_tokens`。

- **逐位消费顺序**（`:143-168`）：对 `i in [0, NUM_DRAFT_TOKENS)` **串行**（单线程 `t_id==0`，
  `:138`）：
  - `p_target = target_probs[i]`（`:144`）——**就是 `prob_extract` 的 `output[b, tid]`**（
    `prob_scatter_sm100.cuh:74`），即 target 模型在 verify 位置 `step+1+i` 对输入 token 的概率。
  - `p_draft = draft_probs[i]`（`:145`）——`_draft_prob_buffer[:, i]`，由 `mtp_float_scatter`
    在 draft 步 `i` 写入。
  - `p_draft == 0` → **贪心回退**：比 `draft_ids[i] != target_ids[i]`（`:147-154`）。
  - 否则走 **LCG**：`rng_state = rng_state*6364136223846793005 + 1442695040888963407`（`:157`），
    `u = (rng_state>>33)/2^31`（`:158-159`），**`p_target > u*p_draft` 才接受**（`:161`）；
    首个拒绝即 `still_accepting=false`（`:165`）。
- **bonus token**：`output_tokens[accepted] = target_ids[accepted]`、`accepted_count[0] =
  accepted + 1`（`:170-172`）——被拒位置用 target 的 token 补上。
- **`target_probs`/`draft_probs` 都是 `[mbt, K]` 布局但内核从 `i=0` 平读**（`:144`/`:145`），
  即**消费的是批元素 0 的 K 个概率**（见 §5）。
- **`seed[0]` 是标量**（`:139`）：`seed` 张量形状 `[mbt,1]`（builder `:2002-2004`），但内核只读
  `seed[0]`——**每请求独立种子**这条设计在 device 侧未兑现（所有请求共用 `seed[0]`）。

### 3.2 `target_verify_strict_kernel`（`:44-91`）——不吃概率、逐位比 id

`draft_ids[i] != target_ids[i]` 首个不等即 `accepted=i`（`:65-72`），bonus 同构
（`output_tokens[t_id] = target_ids[t_id]`, `t_id<final_accepted+1`；`accepted_count[0]=
final_accepted+1`）。`docs/39` §4.1 已摘，本节不重复；**它在链上对应 `method=="strict"`**
（builder `:1946-1955`），是 DeepSeek-V3 默认路径。

### 3.3 `target_verify_greedy_kernel`（`target_verify.cuh:21-60`）——prompt-lookup 用的另一套

语义与 strict 同构但**记号偏移不同**：`spec_token_id[i+1] != target_token_id[i]`（`:41`，
`docs/39` §2 已摘）；登记 `graph.cc:550-554` `make_tuple(2,1)`。它走 `prompt_lookup_verify_handler`
（`persistent_kernel.py:2860-2878`）而非 MTP 链，本文只标注「与 MTP verify 同族但不同调用者」。

### 3.4 `mtp_accept_commit_kernel`（`:255-288`）——把 verify 结果并进序列

`count = accepted_count[0]`（`:278`，含 bonus）；`new_position[0] = current_position[0] + count`、
`num_new_tokens[0] = count`（`:279-280`）；`t_id < count && t_id <= NUM_DRAFT` 的线程把
`output_tokens[t_id]` 拷进 `final_output[t_id]`（`:284-287`）。登记 `graph.cc:877-880`
`make_tuple(3,3)`，前端 `:3077-3100`，builder `:2037-2040`。

---

## 4. builder 侧接线与落差（本节为 `docs/41` 之外的新证据）

`probabilistic` 分支（`deepseek_v3/builder.py:1956-2018`）的接线有**三处静态落差**，都可用
`_cached_new_tensor`（`:1172-1181`，**按 name 缓存**）核实：

1. **局部 `draft_probs` 是死张量**：`:1967-1969` 建了 `draft_probs = _cached_new_tensor(... name=
   "mtp_draft_probs")`；但真正传给 verify 的是 `_draft_prob_buffer`（`:2011`），**这个局部
   `draft_probs` 从未被引用**。它对应的「per-step draft logits → softmax_gather」抽取被
   `:1970-1974` 的 `for step_idx … pass`（TODO）跳过——即 `docs/41` §0 headline 3 说的「draft
   概率抽取未接」在数据流上表现为「建了目标张量、算不出来、改用别处的 buffer」。
2. **`rng_seed` 被建两次**：`:1977-1979` 与 `:2002-2004` 各调一次 `_cached_new_tensor(name=
   "mtp_rng_seed")`；因按 name 缓存（`:1176-1177`），第二次返回**同一个**张量，第一次纯冗余。
   它**没有任何初始化的任务**（builder 未写 seed 值），而 device 侧需要 `seed[0]` 才跑 LCG。
3. **`mtp_verify_probabilistic_layer` 在 guard 外调用**：`prob_extract` 的调用被
   `if step_tensor is not None and hasattr(self, '_target_prob_buffer')`（`:1992`）保护，但
   `mtp_verify_probabilistic_layer(...)`（`:2007`）**无条件调用**且传入 `target_probs`
   （`:2010`）。若 `step_tensor` 缺失或 `_target_prob_buffer` 未在 main graph 建（`:2209-2235`
   的 `if _use_prob_mtp` 分支），`target_probs` 就是**未写入的张量**——一条**静态契约落差**，
   是否在生产触发取决于 `meta_tensors["step"]` 与该分支的一致性，本文只记录。

对照 `strict` 分支（`:1946-1955`）只有 4 个输入、无概率张量，是一条**自洽**的链；`probabilistic`
分支则是「device 全实现（`docs/41` headline 1）+ builder 两处 TODO + 三处接线落差」的**半实现**。

---

## 5. 三段共有的批元素契约（headline 3 的证据）

把 §1 表里的 device 签名并置，可得到一条 `docs/39`/`docs/41` 摘要未明确记录的**承重契约**：

| device 身体 | 是否带 `request_id` | 索引基准 | batch 语义 |
|---|---|---|---|
| `mtp_token_scatter` / `mtp_float_scatter` | 否 | `b = threadIdx.x` | **批并行**（写满 `[BATCH, NUM_SLOTS]`） |
| `mtp_prepare_verify_input` | **是**（`:104`） | `req = request_id` | 每 block 一个请求 |
| `mtp_build_embed_input` | **是**（`:175`） | 读 `req`、写整向量 | 每 block 覆写整向量（§2.2） |
| `target_verify_greedy` | 否 | `i` 从 0 | **仅批元素 0** |
| `target_verify_strict` / `probabilistic` | 否 | `i` 从 0；`accepted_count[0]` | **仅批元素 0** |
| `mtp_accept_commit` | 否 | `current_position[0]`、`final_output[t_id]` | **仅批元素 0** |

即：**「验/提交」这一段在 device 侧只处理批元素 0**（`draft_ids[i]`/`target_ids[i]`/
`output_tokens[i]` 都从 0 平读），而同目录的「准备输入/构造嵌入」又是按 `request_id` 分请求的。
`target_verify_mtp.cuh:28-29` 的文件头注释写「All kernels process one batch element per thread
block」——**这条注释与「无 `request_id` 参数 + 从 0 索引」的实现相符**，但与调用方
`grid_dim=(mbt,1,1)`、`draft_token_ids=all_draft_ids`（`[mbt, K]`）的**多维入参不相符**。
`docs/39` 只记了「单线程 `t_id==0` 串行」，本文补上「**批元素 0**」这一维。

> 口径 caveat：常驻 kernel 里 `blockIdx.x == worker id`（`docs/32` §5.3、`docs/37` §2），
> 因此「一个 CTA 全包」是本仓的常态；verify 家族「从 0 索引」是否等价于「生产里 `mbt` 恒为 1
> 的 decode 场景」、prefill/verify 窗口是否真会 `mbt>1`，本文未追各 builder 的所有传参路径，
> **只作静态契约记录、不下结论**（同 `docs/39`/`docs/41` 对注释落差的态度）。

---

## 6. 与专题主线的关系（判据映射）

- **判据**：整条链上的每个 task 都只是**段内簿记**，各自贡献 C3（多 task 段级管线）与
  C1（并入单次 launch），**不涉及 C4/C5**——与 `docs/29`–`docs/41` 的 device task 走读同档。
  §1 的跨度（`softmax_gather → float_scatter → prob_extract → verify → commit`）是 `docs/08`
  （依赖分析器）里「event 扇入/扇出」的一类真源：概率 buffer 被**跨迭代**读写
  （main graph 写、下一轮 verify 读），正是 `docs/08` §2「最近写者 SSA + 阴影覆盖」要处理的
  view/barrier 场景（对照 `annotated_graph.cc` 的 view 引发 barrier 边）。
- **对「融合边界」的补充**（接 `docs/40` §0 / `docs/41` §5）：这条链把「softmax + gather」
  融进一个 task（`docs/41` §2），又用「scatter + extract」两个 task 在**跨迭代**的时间维上搬运
  概率——是「**融合服务于状态最小化**」而非吞吐的又一例：为避免 materialize 完整概率分布
  （`O(V)`），只在两个 `[mbt,1]` scratch 与两个 `[mbt,max_seq]` buffer 之间搬 `O(K)` 个数。
- **对「簿记型不轻」的补充**：`target_verify_probabilistic` 本身是 `O(K)` 的单线程串行
  （`k≤7`），但它的**上游** `softmax_gather`（`docs/41` §2）是 `O(B·V)` 的两趟全词表归约——
  链的**总成本由最重的簿记 task 决定**，而非 verify 那一段。

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。整条链登记在 SM100 路径
  （`tasks/blackwell/task_header.cuh:57/60/63/64` include `prob_*`/`mtp_token_ops`/`target_verify_mtp`/
  `softmax_gather`），特性是 DeepSeek-V3 MTP verify 窗口，本机（目标卡 H100/sm90）不可执行。
- **「注释/布线落差」只记不判**：§4 的三处（死 `draft_probs`、双 `rng_seed`、guard 外调用）、
  §5 的「批元素 0」都是**静态事实**；注释来自哪一版、是否有意在 decode-only 下成立，
  无从核对，**不下结论**（同 `docs/39` §8 对 include 落差的态度）。
- **未逐行**：`prob_scatter_sm100.cuh`/`softmax_gather_sm100.cuh` 的 device 身体已由 `docs/41`
  逐行，本文只引用；`seed` 的实际初始化位置（哪个 task/tensor 写 `mtp_rng_seed`）未追；
  `_draft_prob_buffer` 在 `strict` 路径下不存在（`:1880-1884` 只在 `probabilistic` 建），
  故 §1 的 draft 概率支路是 **probabilistic 专属**。
- **未与生产端到端对跑**：`docs/09`/`docs/10` 的 MPK 构建与端到端冒烟在 H100 上只跑通 dense
  RMSNorm / Qwen3 MLP，不含投机解码窗口；本文不构建。

---

## 8. 交叉引用

- 概率簿记三兄弟的 device 身体与五层布线：`docs/41-mpk-prob-bookkeeping.md`（`softmax_gather`/
  `prob_scatter`/`prob_extract`）。
- 投机解码 device 层总览与 `target_verify_*`/`mtp_token_ops`/`eagle3_ops` 的逐段摘要：
  `docs/39-mpk-speculative-decoding.md`（§4/§5/§6）。
- `blockIdx.x == worker id` ⇒ 一个 CTA 全包：`docs/32` §5.3、`docs/34` §3.5、`docs/37` §2。
- 依赖分析器里的 view/barrier 边与 event 扇入扇出：`docs/08-mpk-compiler.md` §2/§6。
- 中间量落点（损失三）与融合边界：`docs/04-experiments.md` §17、`docs/40` §0、`docs/41` §5。
- 动态 vs 静态时间维：`docs/04-experiments.md` §47/§48。
- Eagle3 的相近（但独立）commit：`docs/39` §6.3（`eagle3_commit_kernel`，注释自陈
  「Replaces mtp_prepare_verify + mtp_accept_commit」）。
