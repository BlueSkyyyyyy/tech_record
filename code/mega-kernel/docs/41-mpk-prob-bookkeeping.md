# MPK 概率簿记 task 走读：`softmax_gather` / `prob_scatter` / `prob_extract`（P127-1）

> 定位：本文补掉 `docs/39-mpk-speculative-decoding.md` §9 明留的「**未展开的更深层**」——
> 「`softmax_gather_sm100.cuh`/`prob_scatter_sm100.cuh`/`prob_extract_sm100.cuh` 三个概率簿记
> device 身体的内部未逐行（只在 `docs/37` §3.3 给接线坐标）」，也即是 `ROADMAP`「下一步候选」
> 第一条。`docs/37` §3.3 与 `docs/39` §7 已给这三者的**接线坐标**，本文把它们的 **device 身体
> 逐行打开**，并把「注释 / 代码 / 五层布线」三方并置，指出 `docs/37`/`docs/39` 摘要里没有的
> **落差与潜在约束**。
>
> 快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`（与 `docs/06`–`docs/40` 同快照）。
> **纯源码走读（资料分析），无新实验**——三个身体都登记在 SM100 路径（目标卡 H100/sm90 不可执行），
> 且它们的特性上下文（投机解码 verify 窗口）本机不具备，本文不构建、不运行。

---

## 0. 三条 headline

1. **它们是全仓最小的 device 身体，却是「活」的生产 task**：`softmax_gather_sm100.cuh` 115 行
   只导出 1 个函数、`prob_scatter_sm100.cuh` 78 行导出 2 个函数——三个身体合计 193 行、3 个
   `__device__` 函数，比 `docs/40` 退役的 `paged_attention.cuh`（496 行）还小一个量级。但它们
   **全在五层生产布线上**（`TaskType` 282/284/286 → `task_register.cc` → `graph.cc` →
   `persistent_kernel.py` 的 layer API → `deepseek_v3/builder.py` 的概率 MTP 旁路），与
   `docs/36`/`docs/40` 那种「有实现无调用者」的死代码**恰好相反**。它们是 `docs/37` §2 那句
   「薄 task 是 megakernel 的**调度与数值枢纽**」的最极端样本。
2. **每一个都有一处「注释 / 代码 / 布线」三方落差**（`docs/37`/`docs/39` 的行号级摘要看不到）：
   - `softmax_gather` 文件头注释（`:18-29`）写 `Grid: (BATCH_SIZE,1,1), Block: (256,1,1)`，
     且公式写 `prob = exp(logit - max - log_sum)`（`log_sum = log(sum(exp(...)))`）；而**代码**
     把 batch 循环塞进**单个 block** 内（`:46`）、算的是 `exp(logit_at_target - global_max) /
     global_sum`（**除法，非减 log-sum**，`:108`），**所有调用方**传的都是 `grid_dim=(1,1,1)`。
     即「注释假设每个 batch 一个 block」与「代码/布线是单 block 串行扫全 batch」**相反**。
   - `prob_scatter` 注释同样写 `Grid: (batch_size,1,1), Block: (1,1,1)`，而代码单线程串行扫
     batch（`:42`），调用方传 `grid=(1,1,1)`。
   - 这类「一个 CTA 全包」正是 `docs/32` §5.3 / `docs/34` §3.5 / `docs/37` §2 反复记的
     **常驻 kernel 里 `blockIdx.x == worker id`** 的直接后果：device 身体不能假设「grid 维 =
     数据维」，只能自己内部循环——但**注释是从「独立 kernel 版」复制来的**，于是过时。
3. **概率 MTP 这条路是「半实现」的**：三个 device 身体 + 五层布线都活着，但上游
   `deepseek_v3/builder.py` 里**仍留着两处 `TODO` `pass`**——`draft` 概率抽取是空的
   （`:1973-1974`），而注释里求的那个「**slice by runtime offset** kernel」其实**就是本文的
   `prob_extract_task_impl`**（`:1985` 的 TODO 与 `:1993` 的实际调用并存）。同时仓库里存在**两种
   把概率写进「每 step 槽」的机制**：draft 生成用**编译期槽号** `mtp_float_scatter`（`TaskType 285`，
   `for step in range(num_draft_steps)` 展开、`slot_idx` 是常量），main graph 用**运行期 step
   计数器** `prob_scatter`（`TaskType 284`）——同一目的两套载体。

---

## 1. 坐标表与「谁把它们编进生产」

| 文件 | 行数 | 导出函数 | 登记 `TaskType` | blackwell `task_header` include | 生产调用方 |
|---|---|---|---|---|---|
| `tasks/blackwell/softmax_gather_sm100.cuh` | 115 | `softmax_gather_task_impl<T,B,V>` | `TASK_SOFTMAX_GATHER_SM100=282` | `task_header.cuh:60` | `builder.py:1876/2220` |
| `tasks/blackwell/prob_scatter_sm100.cuh` | 78 | `prob_scatter_task_impl<B,P>` | `TASK_PROB_SCATTER_SM100=284` | `task_header.cuh:57` | `builder.py:2229` |
| （同上文件） | — | `prob_extract_task_impl<B,P,K>` | `TASK_PROB_EXTRACT_SM100=286` | （随 `prob_scatter` 一起） | `builder.py:1993` |

- **`prob_extract_sm100.cuh` 这个文件名不存在**：`docs/37`/`docs/39` 把它与 `prob_scatter` 并列
  叙述，实际上它只是 `prob_scatter_sm100.cuh` **同一个文件里的第二个函数**（`:54-76`）。这是一条
  文档层面的**文件粒度勘误**（`docs/37` §3.3 的「`prob_extract_task_impl`（`:54-76`）」行号是对的，
  但「未逐行的 `prob_extract_sm100.cuh`」这个文件名是把「函数名当文件名」了）。
- **五层布线与 `docs/39` §7 一致**，逐层坐标（本文核对）：
  - 枚举：`runtime_header.h:171/173/175`（282/284/286）+ `:174`（285 `mtp_float_scatter`）。
  - 注册：`task_register.cc:2731`（softmax_gather）/`:2808`（prob_extract）/`:2839`（prob_scatter）。
  - `graph.cc` 分派：`:726`（softmax_gather）/`:740`（prob_extract）/`:745`（prob_scatter），
    三者 `task_config` 都是 `std::make_tuple(2,1, type, variant_id)`（2 入 1 出）。
  - 前端 layer API：`persistent_kernel.py:2957`（`softmax_gather_layer`）/`:2992`
    （`prob_scatter_layer`）/`:3010`（`prob_extract_layer`）。
  - builder 调用：见上表。
- **三者都只进 blackwell `task_header`**：`softmax_gather`/`prob_scatter`（含 `prob_extract`）
  只在 `tasks/blackwell/task_header.cuh` 被 include，生产入口 `persistent_kernel.cuh:48-52`
  按 arch 三选一，故它们是 **SM100 专属**（同 `docs/37`/`docs/39` 的定位）。

---

## 2. `softmax_gather_task_impl` 逐行

模板 `template <typename T, int BATCH_SIZE, int VOCAB_SIZE>`（`:33`），注册处以
`T = cute::bfloat16_t`、`BATCH_SIZE=batch_size`、`VOCAB_SIZE=vocab_size` 实例化
（`task_register.cc:2752`，`vocab_size` 取自 `logits` 的 `dim[1]`，`batch_size` 取自 `dim[0]`）。

- **语义**（注释 `:18-29`）：给定 `logits[BATCH_SIZE, VOCAB_SIZE]`（bf16）与
  `token_ids[BATCH_SIZE]`（int64，`:39`），输出 `prob[b] = softmax(logits[b])[token_id[b]]`，
  **不 materialize 整条概率分布**。
- **外层循环**（`:46`）：`for (batch_idx = 0; batch_idx < BATCH_SIZE; ++batch_idx)`——**一个
  block 串行处理所有 batch**。这就是与文件头「Grid (BATCH_SIZE,1,1)」相反的那处（§0 headline 2）。
- **Phase 1 求 row max**（`:50-77`）：`local_max` 沿 `i += num_threads` 跨 vocab 累加
  （`:52-55`）；先 **warp 内** `__shfl_xor_sync(0xffffffff, …)` 蝶形归约（`:57-60`）；再写
  `__shared__ float smem_max[8]`（注释 *"max 8 warps"*，`:62`），`lane_id==0` 每 warp 写一槽
  （`:65-67`），`__syncthreads()`（`:68`）后由 `tid<8` 用 **8 lane 掩码** `0xff` 再蝶形归约
  （`:69-75`），结果 `smem_max[0]`，再 `__syncthreads()`（`:76`）广播给全 block（`:77`）。
- **Phase 2 求 sum(exp)**（`:79-101`）：同构，用 `__expf`（快速近似 exp，`:82`）与
  `smem_sum[8]`（`:88`）。
- **Phase 3 取目标 token 概率**（`:103-110`）：只有 `tid==0` 干活；`target_id` **带界检查**
  `(target_id >= 0 && target_id < VOCAB_SIZE)`，越界取 `-1e30f`（`:105-107`），最终
  `prob = __expf(logit_at_target - global_max) / global_sum`（`:108`）写入 `output_probs[batch_idx]`。
  注意这是**除法**，与注释 `:25` 的 `exp(logit - max - log_sum)`（减 log-sum）数学等价但写法不同
  ——`docs/37` §3.3 的摘要沿用了解释性写法，本文以代码为准。
- **每个 batch 末尾 `__syncthreads()`**（`:111`）：保证 `smem_max`/`smem_sum` 在下一轮 batch 复用前
  所有线程已同步（否则 `tid0` 的写会与下一轮 phase1 的读竞争）。
- **潜在约束**：`smem_max[8]`/`smem_sum[8]` 按 `warp_id = tid/32` 索引，**`blockDim.x ≤ 256`**
  才不会数组越界；调用方一律传 `block_dim=(256,1,1)`，正好打满。若前端误传 >256 线程会**静默越界**
  （无 `assert`）。这是 `docs/37`/`docs/39` 摘要里没有的一条**隐性 block-size 约束**。
- **工作复杂度**：单 CTA 对 `BATCH_SIZE × VOCAB_SIZE` 做两趟全扫（max 一趟、exp-sum 一趟），
  复杂度 `O(BATCH_SIZE·VOCAB_SIZE)`。对真实 serving（`mbt` 到 `max_num_batched_tokens`、
  `vocab` 到 `padded_vocab_size`）这是**一个 block 扛全 batch** 的重活——「簿记型**不等于**轻」
  （`docs/37` §2 的「遍历全 row」在性能上不是免费的，见 `docs/04` §38–§46 的 worker 取 task 标度）。

---

## 3. `prob_scatter_task_impl` / `prob_extract_task_impl` 逐行

两者同处 `prob_scatter_sm100.cuh`，都是「把每 token 概率在 per-position buffer 里搬来搬去」的
簿记算子（注释 `:18-25`/`:50-53` 自陈用于「跨 verify 迭代累积 `P_target(input_token)`」）。

### 3.1 `prob_scatter_task_impl<BATCH_SIZE, MAX_POSITIONS>`（`:29-48`）

- 输入 `prob[BATCH_SIZE]`（float32）、`buffer[BATCH_SIZE, MAX_POSITIONS]`（float32）、
  运行期 `step_ptr[BATCH_SIZE]`（int32）。
- `if (tid > 0) return;`（`:38-40`）——**只有 thread 0** 干活（但 `graph.cc` 仍按 2 入 1 出登记）。
- `for (b …)`（`:42`）单线程串行扫 batch；`pos = step_ptr[b]`（`:43`）；
  **`if (pos >= 0 && pos < MAX_POSITIONS)`** 才写 `buffer[b*MAX_POSITIONS + pos] = prob[b]`
  （`:44-46`）。
- **关键行为**：`pos` 越界时**静默不写**（既无错误、也无哨兵值）——`step` 计数器与 buffer 尺寸
  不一致时，这一趟就是**无操作**。这是「函数名骗人」之外的第二处需要运行层保证的契约
  （对照 §2 的 `softmax_gather` 越界取 `-1e30` 是**有定义**的）。
- 注释 `:24-25` 写 `Grid: (batch_size,1,1), Block: (1,1,1)`，**与代码的单 block 串行 batch 矛盾**
  （调用方 `builder.py:2229` 传 `grid_dim=(1,1,1), block_dim=(1,1,1)`，与代码一致）——§0 headline 2
  的第二例。

### 3.2 `prob_extract_task_impl<BATCH_SIZE, MAX_POSITIONS, NUM_EXTRACT>`（`:54-76`）

- 输入 `buffer[BATCH_SIZE, MAX_POSITIONS]`、运行期 `offset_ptr[BATCH_SIZE]`（int32）；
  输出 `output[BATCH_SIZE, NUM_EXTRACT]`。
- `if (tid >= NUM_EXTRACT) return;`（`:63-65`）——**`NUM_EXTRACT` 个线程**各取一段
  （调用方 `block_dim=(128,1,1)`、`num_extract=num_draft_steps`）。
- **`offset = offset_ptr[b] + 1`**（`:68`，注释 *"+1 because verify starts at step+1"*）——
  verify 窗口相对 `step` 的**硬编码 +1**，是这套「主图写 `[step]`、verify 读 `[step+1..]`」协议的
  锚点（`docs/39` §7 已提这一行，本文补「+1 写死在 device 侧而非调用方」）。
- `src_pos = offset + tid`；界内读 `buffer[b*MAX_POSITIONS + src_pos]`，**界外写 `0.0f`**
  （`:70-74`）——与 `prob_scatter` 的「静默跳过」不同，这里越界**显式补 0**。
- 数学上它就是「按运行期 offset 切片」这一原语：`builder.py:1985` 的
  *"TODO: need a slice by runtime offset kernel"* 与 `:1993` 的 `prob_extract_layer` 调用并存
  ——**TODO 想要的 kernel 已经存在**（只是 draft 侧没接，见 §4.3）。

---

## 4. 五层布线逐处核对（含落差）

### 4.1 `task_register.cc`

- `register_softmax_gather_sm100_task`（`:2731`）：`assert(params.size()==0)`、`num_inputs=2`、
  `num_outputs=1`、`assert(bgraph.operators.size()==3)`；从 `input_ops[0]` 取 `dim[0]/dim[1]`
  作 `batch/vocab`；发射 `kernel::softmax_gather_task_impl<cute::bfloat16_t, $, $>(input0,
  input1, output0)`（`:2752-2755`）——**logits dtype 硬编码 bf16**。
- `register_prob_extract_sm100_task`（`:2808`）：`params = [max_positions, num_extract]`；
  `batch` 取自 `input_ops[0].dim[0]`；发射 `prob_extract_task_impl<$, $, $>(input0, output0,
  static_cast<int const*>(input1))`（`:2829-2832`）——注意实参顺序是 **input0=buffer、
  output0=output、input1=offset**，与函数签名 `(buffer, output, offset)` 对齐。
- `register_prob_scatter_sm100_task`（`:2839`）：`params=[max_positions]`；发射
  `prob_scatter_task_impl<$, $>(input0, output0, input1)`（`:2859-2862`）——**prob 走 input0、
  buffer 走 output0、step 走 input1**（注释 `:2860-2862` 逐参标注）。
- 三者与 `graph.cc` 的 `(2,1)` 一致；`prob_extract`/`prob_scatter` 用「input 侧放运行期计数器、
  output 侧放 buffer」的约定**把状态张量编进 task 的 IO**，而不是新开张量类别。

### 4.2 前端 layer API（`persistent_kernel.py:2957-3027`）

- `softmax_gather_layer`（`:2957`）：断言 `logits.num_dims==2`；建 `TBGraph(grid_dim, block_dim,
  1, 64)`；三个 `new_input(..., io_category=-1, True)`，再 `customized([...])` +
  `register_task(..., "softmax_gather_sm100")`——**只是一个 task、无 K 分块**。
- `prob_scatter_layer`（`:2992`）/`prob_extract_layer`（`:3010`）：同构；前者 `params=[max_positions]`，
  后者 `params=[max_positions, num_extract]`。
- **`grid_dim`/`block_dim` 只是透传**（`TBGraph` 用），device 侧不用它选数据分片——所以
  「注释里的 Grid/Block」即使传错也只影响 kernel 的 launch 形状（一个 CTA 足够/多 CTA 冗余），
  **不影响正确性**（每个 CTA 都会串行算全 batch，多 CTA 会重复写同一输出）。这解释了为什么
  「注释 (BATCH_SIZE,1,1) vs 布线 (1,1,1)」没被历史暴露成 bug：**多 block 只是做重复工作**。

### 4.3 builder 侧（`deepseek_v3/builder.py`）——「半实现」证据

- **draft 生成**（`method == "probabilistic"` 分支，`:1876-1895`）：`softmax_gather_layer`
  （`:1876`，`grid=(1,1,1), block=(256,1,1)`）算 `draft_prob_current`，紧接着用
  **`mtp_float_scatter_layer`**（`:1885`，`slot_idx=step`，编译期常量）写进
  `_draft_prob_buffer[batch, step]`（`for step in range(num_draft_steps)` 展开）——**编译期槽号**。
- **main graph**（`build_from_dict`，`:2204-2234`）：先按 `rejection_sample_method=='probabilistic'`
  建 `_target_prob_buffer[mbt, max_seq_length]` 与 `_target_prob_current[mbt,1]`（`:2210-2218`），
  然后 `softmax_gather_layer`（`:2220`，`grid=(1,1,1), block=(256,1,1)`）+ **`prob_scatter_layer`**
  （`:2229`，`grid=(1,1,1), block=(1,1,1)`，`step_counter=self.mpk.meta_tensors["step"]`）——**运行期
  step 计数器**。
- **verify**（`:1956-1999`）：`:1970-1974` 的 `for step_idx … pass`（**draft 概率抽取 TODO 未实现**），
  `:1976` 注释 *"For now: use target_prob_buffer + dummy draft_probs"*，`:1985` 又一 TODO
  （*"need a slice by runtime offset kernel"*），`:1993` 实际调 `prob_extract_layer`。
- **两条「把概率放进每 step 槽」的路线**（同 §0 headline 3）：`mtp_float_scatter`（TaskType 285、
  compile-time `slot_idx`、`mtp_token_ops.cuh`）用于 **draft**；`prob_scatter`（TaskType 284、
  runtime `step`）用于 **target**。前者依赖 Python 侧把 `num_draft_steps` 静态展开，后者依赖
  运行期 `step` 元张量——**同一个「按位置累积」语义，两种时间维载体**（静态 vs 动态），正是
  `docs/04` §47/§48「动态必须落在调度」在生产代码里的现实注脚。

---

## 5. 与专题主线的关系

- **判据**：三者都只是**段内的簿记 task**，各自贡献 C3（多 task 段级管线）与 C1（并入单次
  launch），**不涉及 C4/C5**——与 `docs/29`–`docs/40` 的 device task 走读同档。它们是
  `docs/37` §2 所记「薄 task 决定 `TaskType` 数与事件扇入/扇出」的具体成员：`prob_scatter` 的
  输出 buffer 是**跨迭代**读写的（`softmax_gather → prob_scatter` → 下一迭代；verify 时
  `prob_extract` 读），即 `docs/08`（依赖分析器）里「event 扇入/扇出」会跨越 iteration 边界的
  一类真源。
- **对「融合边界」的补充**（接 `docs/40` §0）：`softmax_gather` 把 softmax 与 gather 融进一个
  task，但**只输出一个标量**（目标 token 概率）——它是「**算子融合服务于通信/状态最小化**」
  而非「融合服务于吞吐」的样本：融合的目的在于**不 materialize 整条概率分布**（注释 `:22`），
  把 `O(V)` 中间量压成 `O(1)` 输出。这与 `docs/04` §17（损失三 = 中间张量 HBM 往返）同向，
  只是这里省的是**概率分布这座中间山**。
- **对「簿记型不轻」的补充**：`docs/39` headline (A) 说这批投机 task「无一吃 TC、是簿记型」，
  本文给出反例的一半——`softmax_gather` 虽无 TC，却是**两趟全词表归约**的单 CTA 重活；
  「簿记型」描述的是**数值角色**（不产生 token、不做矩阵乘），不是**成本量级**。

---

## 6. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。三者都登记在 SM100 路径、
  特性是投机解码 verify 窗口，本机（目标卡 H100/sm90）不可执行，本文不构建。
- **「注释过时」只记不判**：§0 headline 2 指出三处注释与代码/布线不符（`Grid` 形状、`log_sum`
  写法、`prob_scatter` 的 `Grid`）。这是**静态事实**；注释来自哪一版、是否有意保留旧独立 kernel
  语义，无从核对，**不下结论**（同 `docs/39` 对 include 落差的态度）。
- **`blockDim>256` 越界、`pos` 越界静默**只作**静态约束记录**：是否会在生产触发取决于前端
  `block_dim`/`step` 的取值，本文未追各 builder 的所有传参路径。
- **draft 侧 TODO 是快照事实**：`:1973-1974`/`:1985` 的 TODO 与 `:1993` 的实际调用并存，只说明
  快照里 draft 概率抽取未接；该特性整体是否可用取决于运行层，本文不判。
- **未展开**：`mtp_float_scatter` 的 device 身体（`mtp_token_ops.cuh`）未逐行；`target_verify_*`
  如何消费 `prob_extract` 的输出（拒绝采样比较 `P_target > u·P_draft`）只按注释转述。

---

## 7. 交叉引用

- 接线坐标来源：`docs/37` §3.3（采样链）、`docs/39` §7（两条流水）。
- 同一批薄 task 的总览：`docs/37` §2/§3（`blockIdx.x == worker id` ⇒ 遍历全 row）。
- 概率簿记的兄弟 task：`docs/38` §2（采样家族 `sampling_sm100`/`argmax_sm100`）。
- 「簿记型 device 身体」的同族：`docs/39`（投机解码 device 层）、`docs/40`（退役的
  `paged_attention`，融合边界对照）。
- 中间量落点与损失三：`docs/04` §17（DRAM 流量）、§29–§30（L2 落点）、`docs/05` §1。
- 动态 vs 静态时间维：`docs/04` §47/§48（动态生产者）。
