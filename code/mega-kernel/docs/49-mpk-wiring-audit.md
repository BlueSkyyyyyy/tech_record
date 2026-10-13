# MPK 生产 task 接线可达性审计：登记 → 前端 layer → builder → 生产编译单元 四层交叉

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/48` 同快照）。
> 本专题在 `docs/16`/`docs/20`/`docs/32`/`docs/35`/`docs/39`/`docs/40`/`docs/45`/`docs/48` 里
> 反复撞见「**有实现 ≠ 已接线 ≠ 已编入**」，但每次都是**局部**发现。本文把它升格为**一次全仓审计**：
> 以 `register_task` 的 **95 个登记名 / 92 个被登记 `TaskType`** 为全集，逐层判定可达性，
> 把「孤儿 task」按失败的那一层分成三类，并给出**两枚此前未被任何一篇点名的新孤儿**
> （`attention` / `single_batch_extend_attention`，其 device 身体根本不在任何生产 `task_header` 的
> include 闭包内）。
> `docs/48` §7 明留的「`graph.cc`/`task_register` 里是否还残留其它「无生产者」的融合算子」即本文。
> **纯源码走读（资料分析）**：不构建、不运行；所有数字由 `harness/mpk_wiring_audit.py` 对快照静态算出
> （原始输出 `harness/mpk_wiring_audit.out.txt`）。**本轮无新实验。**

---

## 0. 三条 headline

**(A) 把「已实现未接线」升级为一次四层全仓审计，孤儿可分三类。** 定义四层可达性——
**L1 登记**（`graph.cc` 的 `register_task` 大表把名字映到 `TaskType`）、
**L2 前端**（`persistent_kernel.py` 的 `*_layer` 是否发射该名字）、
**L3 编译单元**（该 `TaskType` 发射的 device 符号，其定义头是否在三个生产 `task_header.cuh` 的
**传递 include 闭包**内）、
**L4 生产者**（`models/*/builder.py` / `demo/` 是否调用该 `*_layer`）。
四层的失败分别给出三类孤儿：**编不进去**（L3 失败，最硬）、**没人产**（L1 有登记但 L2/L4 无人发射/调用）、
**只 demo 用**（L4 只有 demo）。审计规模：L1 = 95 名 / 92 `TaskType`；L2 = 75 个 `*_layer`；
L3 闭包 ampere **29** / hopper **48** / blackwell **99** 个文件；L4 里 **17 个 layer 无生产 builder 调用**。

**(B)（headline）新发现两枚「编不进去」的孤儿 task——`attention` 与 `single_batch_extend_attention`。**
`register_attention_task`（`task_register.cc:534`）发射的设备符号是
`kernel::single_batch_decoding_kernel`（`:569`），它**只**定义在
`tasks/ampere/single_batch_decoding.cuh`；`register_single_batch_extend_attention_task`（`:671`）
发射 `kernel::single_batch_extend_kernel`（`:709`），**只**定义在 `tasks/ampere/single_batch_extend.cuh`。
而这两个头**都不在三个生产 `task_header.cuh` 的传递 include 闭包内**（§4）——即：它们分别对应的
`TASK_ATTENTION_1` / `TASK_SINGLE_BATCH_EXTEND_ATTENTION` **永远进不了常驻 megakernel 的编译单元**。
更进一步：`single_batch_extend.cuh` 自己的 `#include "common.h"` 指向**快照里 `tasks/` 下不存在的文件**，
即使把 include 加进去也编译不过。前端侧，`attention_layer` 只被 `demo/qwen3/demo_chat.py:150` 调用、
`single_batch_extend_attention_layer` 的唯一**生产**调用在 `models/qwen3/builder.py:386` 被**注释掉**
（同文件 `:399` 改走 `paged_attention_layer`），只剩 demo 调用。**这两条路是「只靠 demo 活、但 demo 也
编不过」的死路。**

**(C) 前端「生产者」侧也有一批登记名从未被任何 layer 活跃发射。** 6 个注册名
（`embedding_hopper` / `silu_mul_hopper` / `linear_hopper` / `linear_with_residual_hopper` /
`linear_cutlass_hopper` / `linear_cutlass_with_residual_hopper`）**没有任何 `*_layer` 会发射**——
`embed_layer`（`persistent_kernel.py:863`）与 `silu_mul_layer`（`:2527`）的三元表达式**两个分支写成同一个
名字**、`linear_layer`（`:2436-2447`）的 Hopper 分支只发 `linear_swapAB_hopper`、
`linear_cutlass_*` 的 register 调用在 `:2441`/`:2479` 被**注释**。
另有 **6 个 layer 零调用者**（`identity_layer` / `argmax_layer` / `sampling_sm100_layer` /
`mla_decode_layer` / `mla_reduce_layer` / `mla_prefill_tp8_layer`）。
所以 **「`TaskType` 登记表 = megakernel 能力上界」这句话要打折**：登记 ≠ 可发射 ≠ 可编入 ≠ 被调用。

---

## 1. 方法与坐标系：四层可达性

把一个「task 真能编进常驻 kernel 并跑」拆成四道门，逐门判定：

| 层 | 位置 | 判定 | 失败入库 |
|---|---|---|---|
| **L1 登记** | `src/kernel/graph.cc:439` 起 `register_task` 大表（`if (name == "…") … task_config[op] = (ni,no,TaskType,variant)`） | 名字 → `TaskType` | `docs/24` 已给「无登记」3 枚 |
| **L2 发射** | `python/mirage/mpk/persistent_kernel.py` 的 `*_layer` 里的 `register_task(tb_graph, "…")`（或 `task_name=` 参数、`multigpu.py` 的 `register_tasks`） | layer → register 名 | **本文新表**（§3） |
| **L3 编入** | `tasks/{ampere,hopper,blackwell}/task_header.cuh` 的**传递 include 闭包** | device 符号定义头 ∈ 闭包？ | **本文新表**（§4） |
| **L4 调用** | `python/mirage/mpk/models/**/builder.py` 与 `demo/**` 对 `*_layer` 的调用 | 有生产者？ | **本文新表**（§5） |

「L3 编入」的判定逻辑：`register_xxx_task` 函数体里用 `code.e("kernel::SYM<…>")` 发射设备端**符号名**
（`docs/07` §0 的机制），再用符号名反查它在 `tasks/**/*.cuh` 里的**定义文件**，最后看该文件是否落在
任一生产 `task_header.cuh` 的传递 include 闭包内。闭包用真实的 `#include "…"` 递归解析
（同目录 → `include/mirage/persistent_kernel/` → `include/` 三级回退），见 `harness/mpk_wiring_audit.py`。

坐标系锚点：`enum TaskType` 116 条（`docs/24` §1）、`register_task` 95 个名字 / 92 个被登记 `TaskType`
（`docs/24` §3，本文脚本复核一致）、三个生产 `task_header` 分别
`tasks/ampere/task_header.cuh`（29 文件）/`tasks/hopper/task_header.cuh`（48）/
`tasks/blackwell/task_header.cuh`（99）。

---

## 2. L1：登记全集（引 `docs/24`，不重复）

`docs/24` 已把 `graph.cc:439-930` 的整表打开并给出代际 ID 分区。本文只复核两个数：**95 个
`register_task` 名字**、**92 个被登记 `TaskType`**（脚本输出与 `docs/24` §3 的「95 处赋值、96 个名字、
92 个 `TaskType`」一致——差的一个名字是 `if/else if` 链首的 `embedding`，脚本已单独抓），并直接采用
`docs/24` §4 的「无登记」3 枚（`TASK_ATTENTION_2` / `TASK_PAGED_ATTENTION_2` / `TASK_ALLREDUCE`）。
下面三层才是本文的新内容。

---

## 3. L2：前端发射面——6 个登记名从未被任何 layer 活跃发射

`persistent_kernel.py` 有 **75 个 `*_layer` 方法**。逐个 layer 抽取它**活跃**（去掉注释后）发射的
register 名，与 L1 的 95 名求差，得到 **6 个「有登记、无发射」的名字**：

| 登记名 | TaskType | 为什么没有发射 |
|---|---|---|
| `embedding_hopper` | `TASK_EMBEDDING_HOPPER` | `embed_layer`（`:863`）写成 `"embedding" if self.target_cc == 90 else "embedding"`——**两支同名**，只会发 `embedding` |
| `silu_mul_hopper` | `TASK_SILU_MUL_HOPPER` | `silu_mul_layer`（`:2527`）写成 `"silu_mul" if self.target_cc == 90 else "silu_mul"`——**两支同名** |
| `linear_hopper` | `TASK_LINEAR_HOPPER` | `linear_layer`（`:2436-2447`）的 `target_cc in [90,100)` 分支只发 `linear_swapAB_hopper`（`:2440`/`:2443`） |
| `linear_with_residual_hopper` | `TASK_LINEAR_WITH_RESIDUAL_HOPPER` | 同上，`linear_with_residual_layer`（`:2475-2484`）只发 `linear_swapAB_with_residual_hopper` |
| `linear_cutlass_hopper` | `TASK_LINEAR_CUTLASS_HOPPER` | `linear_layer:2441` 的 register 调用被**注释** |
| `linear_cutlass_with_residual_hopper` | `TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER` | `linear_with_residual_layer:2479` 的 register 调用被**注释** |

**要点**：这 6 个名字的 device 身体**都编进了生产闭包**（如 `embedding_kernel_hopper` ←
`tasks/hopper/embedding_hopper.cuh`、`linear_kernel_hopper` ← `tasks/hopper/linear_hopper.cuh`），
所以它们的失败面**只在 L2**：代码在、编得进，但**前端永远不发射这个名字**。这与「L3 失败」是**正交**的
两类孤儿——`docs/24` §4 的「枚举有编号 ≠ 前端有生产者」在此从 3 枚扩到 6 枚 + 3 枚。

两类**看起来像孤儿但不是**的名字要排除：

- **动态拼名**：`quantize_fp8_layer:2182` 用 `task_name = "quantize_fp8_sm100" if scale_ue8m0 else
  "quantize_fp8_f32scale_sm100"`；`_mla_mtp_decode_tp_layer` 用 `task_name=` 关键字参数发
  `mla_mtp_decode_tp{2,4,8}_sm100`；`allreduce_layer:2502` 经
  `auto_select_allreduce_implementation` → `multigpu.py:176/184/205` 发
  `nvshmem_allgather_strided_put`/`reduction`/`nvshmem_tile_allreduce`。这些**都有活发射**，只是不走
  `register_task(tb_graph, "字面量")` 的形式（脚本已覆盖 `task_name=` 与三元分支；`multigpu.py` 另读）。
- **注释陷阱**：`docs/48` 的融合家族与此处的 `linear_cutlass_*` 都是「源码里**字面量在**、
  但整行被 `#` 注释」。所以判「有没有生产者」必须**先去注释再匹配**（脚本用 `strip_comments`）。

---

## 4. L3：生产编译单元闭包——11 个 device 头不在任何 `task_header` 内

三个生产 `task_header.cuh` 的传递 include 闭包规模：**ampere 29 / hopper 48 / blackwell 99** 个文件
（并集 112）。把 `tasks/` 下全部 `.cuh`/`.h` 与并集求差，得到 **11 个「不在任何生产闭包内」的 device 头**：

| device 头 | 对应的登记/发射 | 归属篇 |
|---|---|---|
| `tasks/ampere/norm_linear.cuh` | `norm_linear_task_impl` ← `rmsnorm_linear` | `docs/48` |
| `tasks/ampere/norm_linear_new.cuh` | 同名第二版，无独立发射 | `docs/48` |
| `tasks/ampere/silu_mul_linear.cuh` | `silu_mul_linear_task_impl` ← `silu_mul_linear_with_residual` | `docs/48` |
| `tasks/hopper/norm_linear_hopper.cuh` | （无发射，仅被单测 include 注释） | `docs/48` |
| `tasks/ampere/single_batch_decoding.cuh` | `single_batch_decoding_kernel` ← `attention` | **本文（新）** |
| `tasks/ampere/single_batch_extend.cuh` | `single_batch_extend_kernel` ← `single_batch_extend_attention` | **本文（新）** |
| `tasks/ampere/single_batch_gqa.cuh` | **无任何发射符**（`single_batch_gqa_kernel` 只被注释别名引用） | **本文（新）** |
| `tasks/speculative_decoding/prompt_lookup.cuh` | `find_ngram_{partial,global}_kernel` | `docs/39` |
| `tasks/speculative_decoding/target_verify.cuh` | `target_verify_greedy_kernel` | `docs/39` |
| `tasks/deprecated/paged_attention.cuh` | 无发射（退役） | `docs/40` |
| `tasks/hopper/matmul_demo_hopper.cuh` | 无发射（demo 用） | **本文（新，纯 demo）** |

把「设备符号 → 定义头 → 是否 ∈ 生产闭包」直接算一遍，得到 **7 枚「ORPHAN 符号」**（发射了、但定义头
不在闭包里）：

```
ORPHAN single_batch_decoding_kernel  -> tasks/ampere/single_batch_decoding.cuh      (attention)
ORPHAN single_batch_extend_kernel    -> tasks/ampere/single_batch_extend.cuh        (single_batch_extend_attention)
ORPHAN norm_linear_task_impl         -> tasks/ampere/norm_linear{,_new}.cuh         (rmsnorm_linear)
ORPHAN silu_mul_linear_task_impl     -> tasks/ampere/silu_mul_linear.cuh            (silu_mul_linear_with_residual)
ORPHAN find_ngram_global_kernel      -> tasks/speculative_decoding/prompt_lookup.cuh
ORPHAN find_ngram_partial_kernel     -> tasks/speculative_decoding/prompt_lookup.cuh
ORPHAN target_verify_greedy_kernel   -> tasks/speculative_decoding/target_verify.cuh
```

并额外核了这些头的**自身 include 能否解析**（这是比「没被 include」更硬的一层）：

| device 头 | 未解析的 `#include` |
|---|---|
| `norm_linear{,_new}.cuh` / `silu_mul_linear.cuh` / `single_batch_extend.cuh` / `single_batch_gqa.cuh` | `common.h`（`tasks/` 下不存在；只有 `tasks/common/common_header.cuh`；另有**不相关**的 `include/mirage/transpiler/common.h`） |
| `norm_linear_hopper.cuh` | `../common.h`、`../dmem_layout.cuh`、`../element_binary.cuh`、`../element_unary.cuh`、`../reduction.cuh`、`../smem_layout.cuh`、`../utils.cuh`（`tasks/` 根目录下不存在这些文件） |
| `prompt_lookup.cuh` / `target_verify.cuh` | `../common.h` |
| `single_batch_decoding.cuh` | 仅 `copy_sm80.cuh`/`dmem_layout.cuh`/`utils.cuh`（裸名，实际在 `tasks/common/`，取决于 build 的 `-I`；同目录的 `norm/smem_layout/mma/…` 可解析） |
| `deprecated/paged_attention.cuh` | 一长串裸名（退役，`docs/40`） |

**对照生产头的 include 约定**：活着的 Ampere 头用**全路径** `#include "tasks/common/common_header.cuh"`
（见 `tasks/ampere/argmax.cuh:16`），而这些孤儿头用**旧式裸名** `#include "common.h"`
（见 `tasks/ampere/norm_linear.cuh:17`）。**孤儿头与当前 include 约定不同代**——这是它们编不进去的
机械原因，也是「名字/登记会骗人」族（`docs/32` §3、`docs/45` §3.3）在构建层面的根。

---

## 5. L4：生产者面——17 个 layer 无生产 builder 调用

用去注释后的文本，把 `models/**/builder.py` 与 `demo/**` 对 `*_layer` 的调用抓出来，得到
**17 个「无生产 builder 调用」的 layer**（其中 9 个 demo 也不调）：

| layer | 发射的登记名 | demo 调用 | 备注 |
|---|---|---|---|
| `rmsnorm_linear_layer` | `rmsnorm_linear` | Y | 仅 `demo/qwen3/demo_chat.py` |
| `attention_layer` | `attention` | Y | 仅 `demo/qwen3/demo_chat.py:150` |
| `single_batch_extend_attention_layer` | `single_batch_extend_attention` | Y | 生产调用在 `qwen3/builder.py:386` 被注释 |
| `paged_attention_split_kv_layer` | `paged_attention_split_kv_{sm100,hopper}` | Y | 仅 demo |
| `paged_attention_split_kv_merge_layer` | `paged_attention_split_kv_merge_sm100` | Y | 仅 demo |
| `silu_mul_linear_with_residual_layer` | `silu_mul_linear_with_residual` | Y | 仅 demo（= `docs/48` 家族） |
| `copy_layer` | `copy` | Y | 仅 eagle3 demo |
| `eagle3_commit_layer` | `eagle3_commit` | Y | 仅 eagle3 demo（`docs/43`） |
| `mla_decode_layer` | `mla_decode_sm100` | n | 被 MTP 版顶替（`docs/35`） |
| `mla_reduce_layer` | `mla_reduce_sm100` | n | 同上 |
| `mla_prefill_tp8_layer` | `mla_prefill_tp8_sm100` | n | 零调用者 |
| `identity_layer` | `identity` | n | 零调用者（`identity_task_impl` 在闭包内） |
| `argmax_layer` | `argmax` | n | 零调用者（`argmax_partial/reduce` 才是活路） |
| `sampling_sm100_layer` | `sampling_sm100` | n | 零调用者（`sampling_partial/reduce` 是活路；三者共用 `TASK_SAMPLING_SM100`） |
| `find_ngram_partial_layer` | `find_ngram_partial` | n | 只被 `prompt_lookup_spec_handler`（`:2795` 起）内部调用 |
| `find_ngram_global_layer` | `find_ngram_global` | n | 同上（`:2821`） |
| `target_verify_greedy_layer` | `target_verify_greedy` | n | 只被 `prompt_lookup_verify_handler`（`:2875`）内部调用 |

**两类「不是孤儿但容易误判」**：`find_ngram_*` / `target_verify_greedy` 的 layer 虽无 builder/demo 直接调用，
但被 **prompt-lookup 投机解码 handler** 内部调用（`_spec_verify_handlers` / `_spec_decode_handlers` 字典，
`persistent_kernel.py:528-533`）。而这两个 handler 唯一的**生产**接入点（`qwen3/builder.py:577-684`）**也被注释**
（`# if spec_decode_config ... promptlookup`），只剩下 `demo/qwen3/demo_*.py` 走
`verify_layer_dispatcher`/`draft_forward_layer_dispatcher`（如 `demo_hopper.py:401/650`）。**即便是 demo 走通，
它们也会在 L3 撞墙**——`prompt_lookup.cuh`/`target_verify.cuh` 不在生产闭包（`docs/39`）。

---

## 6. 交叉：孤儿总表（按失败层）

把 L1–L4 的失败合并，得到本快照「登记了但端到端不可达」的 task 全景：

| 失败层 | 孤儿 | 数量 | 已有篇 |
|---|---|---|---|
| **L3 编不进去**（最硬，即使前端发射也进不了 kernel） | `attention`、`single_batch_extend_attention`、（`single_batch_gqa` 连发射都没有）、`rmsnorm_linear`、`silu_mul_linear_with_residual`、`find_ngram_partial`、`find_ngram_global`、`target_verify_greedy` | 8 | `docs/39`/`docs/48` + **本文新 2** |
| **L2 无发射**（代码在、编得进，但前端永不发名字） | `embedding_hopper`、`silu_mul_hopper`、`linear_hopper`、`linear_with_residual_hopper`、`linear_cutlass_hopper`、`linear_cutlass_with_residual_hopper` | 6 | **本文新** |
| **L4 无生产调用**（只 demo / 零调用） | `mla_decode`、`mla_reduce`、`mla_prefill_tp8`、`identity`、`argmax`、`sampling_sm100`（零调用）；`rmsnorm_linear`、`single_batch_extend_attention`、`paged_attention_split_kv*`、`copy`、`eagle3_commit`（demo-only） | 12 | `docs/35`/`docs/43` + **本文** |
| **L1 无登记**（`docs/24` §4） | `TASK_ATTENTION_2`、`TASK_PAGED_ATTENTION_2`、`TASK_ALLREDUCE` | 3 | `docs/24` |

> 注意层间**可叠加**：`rmsnorm_linear` / `single_batch_extend_attention` / `silu_mul_linear_with_residual`
> 同时命中 L3（编不进）与 L4（只 demo）；`find_ngram_*` / `target_verify_greedy` 同时命中 L3 与 L4（无生产）。
> 表里按「最硬失败层」归类。

**一句话结论**：`TaskType` 登记表（92 个）是**能力上界**，而在本快照真正「能被生产模型 builder 调用
→ 发射 → 编进生产 `task_header` 闭包」的**活跃子集明显更小**。这条比 `docs/24` 的「能力上界」结论更锋锐：
**上界之内还有一层「接线有效性」过滤**，且过滤掉的部分可精确定位到三个正交的层面。

---

## 7. 新发现详读：`attention` / `single_batch_extend_attention` 为何是死路

这两条是本文相对 `docs/07`–`docs/48` 的**新增孤儿**，值得单独核到行。

### 7.1 `attention`（`TASK_ATTENTION_1`）

- **L1 登记**：`graph.cc:492` `else if (name == "attention")` → `register_attention_task`；
  `task_config[op] = (7, 1, TASK_ATTENTION_1, variant)`。
- **L2 发射**：`register_attention_task`（`task_register.cc:534`）在 `:569` 发
  `kernel::single_batch_decoding_kernel<bfloat16, …>`，`:606` 返回 `register_task_variant(TASK_ATTENTION_1, …)`。
  前端 `attention_layer`（`persistent_kernel.py:904-969`）注册名 `"attention"`。
- **L3 编入**：符号 `single_batch_decoding_kernel` **只**定义在 `tasks/ampere/single_batch_decoding.cuh:38`；
  该头**不在**三个生产 `task_header` 闭包内（§4）。故 `TASK_ATTENTION_1` **进不了常驻 kernel**。
- **L4 调用**：`attention_layer` **只**被 `demo/qwen3/demo_chat.py:150` 调用（生产 builder 一律走
  `paged_attention_layer`，见 §5）。
- **旁证**：`tests/runtime_python/runtime_kernel_wrapper.cu:24` 的 `using kernel::single_batch_decoding_kernel;`
  被**注释**；该文件对 `single_batch_*` 的 wrapper 全部包在**从未 `#define` 的 `#ifdef DEPRECATED_TESTS`**
  里（`:30` 起）——与 `docs/48` 的融合家族是同一种「单测接线也已废弃」形态。
- **唯一的活引用**：`tests/runtime_python/test_attention/runtime_kernel_wrapper.cu:5` 直接
  `#include "single_batch_decoding.cuh"`，即**仍有一个 standalone 单测编译单元**会编它（能否解析其裸 include
  取决于该目录 `setup.py` 的 `-I`；本文只记静态事实、不判其对错）。**但生产常驻 kernel 从不编它。**

### 7.2 `single_batch_extend_attention`（`TASK_SINGLE_BATCH_EXTEND_ATTENTION`）

- **L1 登记**：`graph.cc:500` → `register_single_batch_extend_attention_task`，`TASK_SINGLE_BATCH_EXTEND_ATTENTION`。
- **L2 发射**：该 register 函数（`task_register.cc:671`）在 `:709` 发
  `kernel::single_batch_extend_kernel<bfloat16, …>`。前端 `single_batch_extend_attention_layer`
  （`persistent_kernel.py:1204-1302`）注册名 `"single_batch_extend_attention"`。
- **L3 编入**：`single_batch_extend_kernel` **只**定义在 `tasks/ampere/single_batch_extend.cuh:41`；
  不在生产闭包。且该头自己 `#include "common.h"`（`tasks/` 下不存在）→ **双重编不过**。
- **L4 调用**：`models/qwen3/builder.py:386` 的唯一生产调用**被注释**（同文件 `:399` 改走 `paged_attention_layer`）；
  活调用只剩 `demo/qwen3/demo_hopper.py:473`、`demo_30B_A3B*.py`、`demo/llama3/demo.py:547` 等。
- **含义**：`models/qwen3/builder.py:385` 的 `# TODO(Jianan Ji): spec_decode_config handling` 说明这是
  「旧实现被拆掉、spec_decode 分支尚未接回」的残留；而它残留的 device 身体又用旧 include 约定、进不了闭包。

---

## 8. 与既有篇的对照与勘误

| 结论 | 首见篇 | 本文的增量 |
|---|---|---|
| 融合家族（`norm_linear`/`silu_mul_linear`）「已实现未编入」 | `docs/48` | 从 2 个头扩为「**11 个 device 头不在闭包**」的全表，并把失败**分层**（这族是 L3+L4） |
| `prompt_lookup`/`target_verify` 不在生产 include 集 | `docs/39` | 归入 L3；并补 L4 证据（生产接入点 `qwen3/builder.py:577-684` 也注释） |
| `mla_decode`/`mla_reduce` 被 MTP 版顶替 | `docs/35` | 归入 L4「零调用 layer」 |
| `deprecated/paged_attention` 是孤儿 | `docs/40` | 归入 L3/L4 |
| `TaskType` 登记表 = 能力上界；3 枚无登记 | `docs/24` | **补出 6 枚「有登记无发射」+ 8 枚「编不进」+ 17 枚「无生产调用」**，并给四层审计方法 |
| `attention` / `single_batch_extend_attention` | — | **本文新发现**（L3 编不进，且 demo 也编不过） |
| `single_batch_gqa.cuh` / `matmul_demo_hopper.cuh` | — | **本文新发现**（无发射的死头） |

**没有勘误**：本文未推翻任何既有篇的结论；`docs/24` §4、`docs/48` §4 的事实经脚本复核一致。

---

## 9. 判据映射

| 判据 | 本段是否贡献 |
|---|---|
| C1 单次 launch | 间接（这些 task 若可编入，本是常驻图里的一个 task；本文只判「能否编入」） |
| C2 常驻 | 间接（同上） |
| C3 多算子范围 | **是**——本文正是「这张图**真正**能容纳哪些 task、边界画在哪」的**可达性边界**，比 `docs/24` 的「登记上界」更接近真实范围 |
| C4 设备内协调 | 否 |
| C5 SM 级调度 | 否（L3/L4 不涉及 worker/scheduler） |

**结论**：本文贡献 **C3 的「有效范围」**——把「登记 → 发射 → 编入 → 调用」四道门的通过集补齐，
是 `docs/24`（能力上界）、`docs/48`（融合家族局部审计）在**全仓、跨层**维度上的收口。

---

## 10. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有结论为快照 `f9eb70c` 静态事实；本文**不构建 mirage**来验证
  「把某个孤儿头加进 `task_header` 后能否编译/跑对」，只核出「当前状态的闭包/发射/调用关系」。
- **L3 闭包是「文本 `#include` 递归」而非编译器语义**：不处理 `#if`/`#ifdef` 条件 include（如
  `ampere/task_header.cuh:21` 的 `#if MIRAGE_USE_CUTLASS_KERNEL` 两分支**都**算了进去，故闭包偏保守/偏大）、
  不处理宏拼接的 include 名。结论「某头不在闭包内」因此是**充分条件**（不在文本闭包 ⇒ 一定不被直接/间接 include）。
- **L2 发射面靠正则**：动态拼名（三元、`task_name=`、`multigpu.py`）已覆盖，但若还有更隐晦的拼名（如
  `f"…{x}"`）会漏；本文对可疑项（`quantize_fp8_*`、`nvshmem_*`、`reduction`）**逐个人工核过**并入正文。
- **`single_batch_decoding.cuh` 的单测编译单元是否真能构建未核**：`test_attention/runtime_kernel_wrapper.cu`
  直接 include 它，但能否解析其裸 `tasks/common/*` include 取决于该目录 setup 的 include path，本文只记静态事实。
- **「孤儿」是「读码可见的不可达候选」，非 bug 判定**：某些 task 可能是「预留接口 / 待接线 / 特定模型专用」，
  本文不判上游对错（同 `docs/24` §4 口径）。
- **`deprecated/` 与 `demo/` 的定位**：`deprecated/paged_attention.cuh`、`matmul_demo_hopper.cuh` 分别是
  明确退役与 demo 专用，把它们计入「孤儿」是为了穷尽「不在生产闭包」的集合，不代表它们「应该有接线」。

---

## 11. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`。
- 审计脚本 / 原始输出：`harness/mpk_wiring_audit.py`、`harness/mpk_wiring_audit.out.txt`。
- L1 登记表：`src/kernel/graph.cc:439`（`register_task`）；`embedding:445`、`attention:492`、
  `single_batch_extend_attention:500`、`linear:505`、`silu_mul:514`。
- L2 前端：`python/mirage/mpk/persistent_kernel.py`（`embed_layer:842`、`rmsnorm_layer:865`、
  `attention_layer:904`、`single_batch_extend_attention_layer:1204`、`linear_layer:2390`、
  `linear_with_residual_layer:2449`、`allreduce_layer:2488`、`silu_mul_layer:2513`、`identity_layer:2529`、
  `mla_decode_layer:1627`、`mla_reduce_layer:1660`、`mla_prefill_tp8_layer:1720`、`argmax_layer:2591`、
  `sampling_sm100_layer:2659`、`prompt_lookup_spec_handler:2795`、`prompt_lookup_verify_handler:2860`、
  `quantize_fp8_layer:2162`）；`python/mirage/mpk/multigpu.py:176/184/205`。
- L3 编译单元：`tasks/ampere/task_header.cuh`、`tasks/hopper/task_header.cuh`、
  `tasks/blackwell/task_header.cuh`（三者 include 集）；`tasks/ampere/norm_linear.cuh:17`、
  `tasks/ampere/argmax.cuh:16`（活头用全路径的对照）。
- L3 发射符号定义头：`tasks/ampere/single_batch_decoding.cuh:38`、`tasks/ampere/single_batch_extend.cuh:41`、
  `tasks/ampere/single_batch_gqa.cuh:35`、`tasks/ampere/norm_linear{,_new}.cuh`、
  `tasks/ampere/silu_mul_linear.cuh`、`tasks/speculative_decoding/prompt_lookup.cuh`、
  `tasks/speculative_decoding/target_verify.cuh`。
- register 发射器：`task_register.cc`（`register_rmsnorm_linear_task:489`、`register_attention_task:534`、
  `register_single_batch_extend_attention_task:671`、`register_silu_mul_linear_with_residual_task:826`、
  `register_linear_hopper_task:1162`、`register_silu_mul_hopper_task:1925`、
  `register_embedding_hopper_task:1970`；发射点 `:569/:709/:1295/:1959/:1998`）。
- L4 生产者：`python/mirage/mpk/models/qwen3/builder.py:386`（注释）、`:399`（`paged_attention_layer`）、
  `:577-684`（promptlookup 注释）；`demo/qwen3/demo_chat.py:150`；`demo/qwen3/demo_hopper.py:401/473/650`；
  `demo/llama3/demo.py:547`；`tests/runtime_python/runtime_kernel_wrapper.cu:24-26/30`（注释/`DEPRECATED_TESTS`）；
  `tests/runtime_python/test_attention/runtime_kernel_wrapper.cu:5`。
- 本专题交叉引用：`docs/24`（登记全表 + 3 枚无登记）、`docs/35`（MLA 旧 decode/reduce 被顶替）、
  `docs/39`（`tasks/speculative_decoding/` 接线落差）、`docs/40`（退役版）、`docs/48`（融合家族接线审计）、
  `docs/07` §0（`code.e` 发射机制）、`docs/06` §2（`TaskType` 编码）。
