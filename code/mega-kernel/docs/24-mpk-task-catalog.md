# MPK 生产路径的 task 目录：`TaskType` 全表与代际 ID 分区

> 本文补齐 `docs/14`（P100-1）**§7 明留的尾巴**：
>
> > **未逐行走读全部 95 个任务分支**：本文只精读 `embedding`/`rmsnorm`/`rmsnorm_linear`/`linear`/
> > `allreduce` 等代表性 layer，其余（MoE 系列、MLA/paged-attention 系列、speculative 系列）只统计了
> > 机制，未逐一展开。
>
> `docs/14` §3.2 已经证明「`register_task` 是一张 95 分支的 name→TaskType 大表」并给出元组语义
> `(num_inputs, num_outputs, TaskType, variant_id)`（`include/mirage/kernel/graph.h:231-233`、
> `src/kernel/annotated_graph.cc:293-297`），但**没有把这张表打开**。本文把
> `include/mirage/persistent_kernel/runtime_header.h` 的 `enum TaskType` 与
> `src/kernel/graph.cc:439-930` 的登记大表**逐条列出**，给出 MPK 生产路径（`mpk` 分支）的
> **能力边界面**：它到底能编出哪些 task、按什么代际分区、哪些 `TaskType` 其实没有被任何前端登记。
> （`docs/14` §7 另列的「模型 builder 组装逻辑」「自动路径」「虚拟张量语义」三条尾巴不在本文范围——
> 前者见 `models/*/builder.py`，后两者见 `docs/15`–`docs/21`、`docs/08` §2。）
>
> 本文是**纯源码走读（资料分析）**，不构建、不运行 mirage，不引用任何性能数。快照
> `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`–`docs/22` 同一快照）。

## 0. 定位：这张表在编译栈里的位置

`docs/14` §2 的链路是「Python `*_layer()` → `customized()` 建 bGraph → `register_task(name)` 贴
`TaskType`」，`docs/08` 的依赖分析器消费 `task_configs` 建 tGraph，`docs/06`/`docs/07` 的运行层
按 `TaskType` 调 `_execute_task`。也就是说：

```
   Python 前端            register_task 大表           运行层
   ───────────    ─────────────────────────────    ──────────────────────
   名称字符串  ──►  name → (ni,no,TaskType,var)  ──►  device 实现函数
   "rmsnorm_hopper"   graph.cc:445-930              task_register.cc
```

**一个 `TaskType` 就是一个 device 实现函数**（`docs/07` §0 的 `task_register.cc:1181-1318`：
形状→编译期常量→`linear_kernel_hopper<...>`）。所以 `TaskType` 全表 = **MPK 能编进常驻 kernel 的
全部「算子实现」清单**，也就是这个 megakernel 编译器的**能力上界**。把这张表打开，正好回答
`docs/23` §5「何时该用 megakernel」里那句「融合的 task 类型越多，成本越高」的另一半——
**可融合的 task 类型究竟有多少种**。

## 1. `enum TaskType` 的代际 ID 分区

`runtime_header.h:87` 起是 `enum TaskType`。它**不是从 0 连续编号**，而是按「硬件代际 / 模型家族」
切成**预留 ID 块**（每条块用一对 `*_TASK_BEGIN`/`*_TASK_END` 占位符把范围圈起来）。这是一套
**显式的「代际命名空间」**设计：新代际只在自己的块里加编号，老编号永不复用（ABI/序列化稳定——
`docs/06` §2 的 `TaskId` 编码要能跨编译产物稳定）。

| ID 区间 | 块名（占位符） | 行号 | 用途 |
|---|---|---|---|
| 0 / 10 | 无（散落） | `:88-89` | 控制标记：`TASK_TERMINATE=0`、`TASK_BEGIN_TASK_GRAPH=10` |
| 101–121 | 无（散落） | `:91-111` | **通用/基线** compute task（架构无关的算子表示） |
| 150–198 | `TASK_HOPPER_TASK_BEGIN/END` | `:113-128` | **Hopper（sm90）** 专用实现（TMA/wgmma 版） |
| 200–204 | 无（散落） | `:197-201` | **scheduler 控制面** task（`TASK_SCHD_*`/`TASK_GET_*`） |
| 230–299 | `TASK_SM100_TASK_BEGIN/END` | `:130-196` | **SM100（Blackwell）** 专用实现（数量最多） |
| 300–349 | `TASK_MULTIGPU_TASK_BEGIN/END` | `:203-206` | **多卡通信** task（NVSHMEM） |
| 350–399 | `TASK_INKLING_TASK_BEGIN/END` | `:208-215` | **Inkling（Thinking Machines）** 模型专用 |
| 400–449 | `TASK_GLM_TASK_BEGIN/END` | `:217-220` | **GLM** 模型专用 |
| 450–499 | `TASK_GPT_OSS_TASK_BEGIN/END` | `:222-225` | **GPT-OSS** 模型专用 |

设计读取：**「同一算子的不同硬件代际实现」被显式拆成不同 `TaskType`**（如 `TASK_RMS_NORM=119`
vs `TASK_RMS_NORM_HOPPER=154`；`TASK_LINEAR=120` vs `TASK_LINEAR_HOPPER=152` vs
`TASK_LINEAR_SM100=253`）。这和 `docs/21` 讲的「Hopper 后端 / Blackwell 后端」是同一件事在
**枚举层**的投影：codegen 后端的选择最终落成「登记哪个 `TaskType`」，而选择发生在 **Python 前端的
`target_cc` 分支**里（`persistent_kernel.py:881`：`"rmsnorm_hopper" if self.target_cc >= 90 else "rmsnorm"`）。

## 2. `TaskType` 全表（按 ID 排序）

下面把 `enum TaskType` 的**全部 116 条命名条目**列出（含 12 个范围占位符），并在「登记名」列给出
`graph.cc` 大表里对应的前端名称与 `(num_inputs, num_outputs)`（标 **（无登记）** = 该 `TaskType` 未被
任何 `register_task` 名字登记，见 §4；`num_inputs` 带条件的是 `make_tuple` 里的三元表达式）。

### 2.1 控制标记与 scheduler 控制面

| ID | `TaskType` | 登记名 (ni,no) | 备注 |
|---|---|---|---|
| 0 | `TASK_TERMINATE` | — | 终止哨兵 |
| 10 | `TASK_BEGIN_TASK_GRAPH` | — | 图起始哨兵（`runtime.cc:253/1346` 注入） |
| 200 | `TASK_SCHD_TASKS` | — | scheduler 内部 |
| 201 | `TASK_SCHD_EVENTS` | — | scheduler 内部 |
| 202 | `TASK_GET_EVENT` | — | scheduler 内部 |
| 203 | `TASK_GET_NEXT_TASK` | — | scheduler 内部 |
| 204 | `TASK_SCHD_PREPARE_BATCH` | — | scheduler 内部（`docs/06` §4 的 `prepare_next_batch`） |

### 2.2 通用/基线 compute（101–121）

| ID | `TaskType` | 登记名 (ni,no) |
|---|---|---|
| 101 | `TASK_EMBEDDING` | `embedding` (2,1) |
| 102 | `TASK_RMS_NORM_LINEAR` | `rmsnorm_linear` (3,1) |
| 103 | `TASK_ATTENTION_1` | `attention` (7,1) |
| 104 | `TASK_ATTENTION_2` | **（无登记）** |
| 105 | `TASK_SILU_MUL_LINEAR_WITH_RESIDUAL` | `silu_mul_linear_with_residual` (3,1) |
| 106 | `TASK_ALLREDUCE` | **（无登记）**，注释 *"This legacy allreduce task will be removed soon"* |
| 107 | `TASK_REDUCE` | `reduction` (2,1) |
| 108 | `TASK_LINEAR_WITH_RESIDUAL` | `linear_with_residual` (3,1) |
| 109 | `TASK_ARGMAX` | `argmax` (1,1) |
| 110 | `TASK_ARGMAX_PARTIAL` | `argmax_partial` (1,2) |
| 111 | `TASK_ARGMAX_REDUCE` | `argmax_reduce` (2,1) |
| 112 | `TASK_FIND_NGRAM_PARTIAL` | `find_ngram_partial` (1,1) |
| 113 | `TASK_FIND_NGRAM_GLOBAL` | `find_ngram_global` (2,1) |
| 114 | `TASK_TARGET_VERIFY_GREEDY` | `target_verify_greedy` (2,1) |
| 115 | `TASK_SINGLE_BATCH_EXTEND_ATTENTION` | `single_batch_extend_attention` (7,1) |
| 116 | `TASK_PAGED_ATTENTION_1` | `paged_attention` (7,1) |
| 117 | `TASK_PAGED_ATTENTION_2` | **（无登记）** |
| 118 | `TASK_SILU_MUL` | `silu_mul` / `moe_silu_mul` (1,1) |
| 119 | `TASK_RMS_NORM` | `rmsnorm` (2,1) |
| 120 | `TASK_LINEAR` | `linear` (2,1) |
| 121 | `TASK_IDENTITY` | `identity` (1,1) |

### 2.3 Hopper sm90 专用（150–198）

| ID | `TaskType` | 登记名 (ni,no) |
|---|---|---|
| 150 | `TASK_HOPPER_TASK_BEGIN` | 占位符 |
| 151 | `TASK_LINEAR_WITH_RESIDUAL_HOPPER` | `linear_with_residual_hopper` (3,1) |
| 152 | `TASK_LINEAR_HOPPER` | `linear_hopper` (2,1) |
| 153 | `TASK_PAGED_ATTENTION_HOPPER` | `paged_attention_hopper` (7,1) |
| 154 | `TASK_RMS_NORM_HOPPER` | `rmsnorm_hopper` (2,1) |
| 155 | `TASK_LINEAR_SWAPAB_HOPPER` | `linear_swapAB_hopper` (2,1) |
| 156 | `TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER` | `linear_swapAB_with_residual_hopper` (3,1) |
| 157 | `TASK_LINEAR_CUTLASS_HOPPER` | `linear_cutlass_hopper` (2,1) |
| 158 | `TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER` | `linear_cutlass_with_residual_hopper` (3,1) |
| 159 | `TASK_SILU_MUL_HOPPER` | `silu_mul_hopper` (1,1) |
| 160 | `TASK_EMBEDDING_HOPPER` | `embedding_hopper` (2,1) |
| 161 | `TASK_MOE_W13_LINEAR_SM90` | `moe_w13_linear_sm90` (4,1) |
| 162 | `TASK_MOE_W2_LINEAR_SM90` | `moe_w2_linear_sm90` (4,1) |
| 163 | `TASK_SPLITK_LINEAR_SWAPAB_HOPPER` | `splitk_linear_swapAB_hopper` (2,1) |
| 164 | `TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER` | `paged_attention_split_kv_hopper` (7,2) |
| 198 | `TASK_HOPPER_TASK_END` | 占位符 |

**Hopper 块的特征**：`linear` 一族分出 **swapAB / cutlass / splitk** 三个正交变体，说明 Hopper 上
MPK 已经为「A/B 谁放 SMEM」「手写 vs CUTLASS」「split-K」准备了独立 `TaskType`；MoE 也在 sm90 上
有了 W13/W2 两段（`docs/02` 的 DeepGEMM MoE 分解的 MPK 呼应）。

### 2.4 SM100（Blackwell）专用（230–299）

| ID | `TaskType` | 登记名 (ni,no) |
|---|---|---|
| 230 | `TASK_SM100_TASK_BEGIN` | 占位符 |
| 231 | `TASK_SM100_TMA_START_TASK` | **（无登记）**，TMA 段起始标记 |
| 232 | `TASK_COPY` | `copy` (1,1) |
| 233 | `TASK_CONCAT` | `concat` (1,1) |
| 235 | `TASK_EAGLE3_D2T_REMAP` | `eagle3_d2t_remap` (2,1) |
| 236 | `TASK_EAGLE3_COMMIT` | `eagle3_commit` (5,2) |
| 248 | `TASK_MOE_W13_FP8_SM100` | `moe_w13_fp8_sm100` (6,1) |
| 249 | `TASK_MOE_W2_FP8_SM100` | `moe_w2_fp8_sm100` (6,1) |
| 251 | `TASK_SPLITK_LINEAR_SM100` | `splitk_linear_sm100` (2,1) |
| 252 | `TASK_LINEAR_WITH_RESIDUAL_SM100` | `linear_with_residual_sm100` / `linear_with_bias_sm100` (3,1) |
| 253 | `TASK_LINEAR_SM100` | `linear_sm100` (2,1) |
| 254 | `TASK_MOE_W13_LINEAR_SM100` | `moe_w13_linear_sm100` (4 或 5,1) |
| 255 | `TASK_MOE_W2_LINEAR_SM100` | `moe_w2_linear_sm100` (4 或 5,1) |
| 256 | `TASK_SM100_TMA_END_TASK` | 占位符（TMA 段结束） |
| 257 | `TASK_ATTN_SM100` | `paged_attention_sm100` (7 或 8,1) |
| 258 | `TASK_ARGMAX_REDUCE_SM100` | `argmax_reduce_sm100` (2,1) |
| 259 | `TASK_ARGMAX_PARTIAL_SM100` | `argmax_partial_sm100` (1,2) |
| 260 | `TASK_MOE_TOPK_SOFTMAX_SM100` | `moe_topk_softmax_sm100` (1,3) |
| 261 | `TASK_MOE_MUL_SUM_ADD_SM100` | `moe_mul_sum_add_sm100` (3,1) |
| 262 | `TASK_TENSOR_INIT` | `tensor_init` (2,1) |
| 263 | `TASK_PAGED_ATTENTION_SPLIT_KV_SM100` | `paged_attention_split_kv_sm100` (7,2) |
| 264 | `TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100` | `paged_attention_split_kv_merge_sm100` (2,1) |
| 265 | `TASK_SAMPLING_SM100` | `sampling_sm100` (1,1) / `sampling_reduce_sm100` (2,1) |
| 266 | `TASK_MLA_DECODE_SM100` | `mla_decode_sm100` (2,2) |
| 267 | `TASK_MLA_REDUCE_SM100` | `mla_reduce_sm100` (2,1) |
| 268 | `TASK_MLA_PREFILL_SM100` | `mla_prefill_sm100` (5,0) |
| 269 | `TASK_MLA_MTP_DECODE_SM100` | `mla_mtp_decode_sm100` (2,2) |
| 270 | `TASK_MLA_MTP_REDUCE_SM100` | `mla_mtp_reduce_sm100` (2,1) |
| 271 | `TASK_MTP_VERIFY_STRICT` | `mtp_verify_strict` (2,2) |
| 272 | `TASK_MTP_ACCEPT_COMMIT` | `mtp_accept_commit` (3,3) |
| 273 | `TASK_MTP_TOKEN_SCATTER` | `mtp_token_scatter` (1,1) |
| 274 | `TASK_MTP_PREPARE_VERIFY` | `mtp_prepare_verify` (4,1) |
| 275 | `TASK_QUANTIZE_FP8_SM100` | `quantize_fp8_sm100` / `quantize_fp8_f32scale_sm100` (1,2) |
| 276 | `TASK_LINEAR_FP8_SM100` | `linear_fp8_sm100` (4,1) |
| 277 | `TASK_LINEAR_FP8_WITH_RESIDUAL_SM100` | `linear_fp8_with_residual_sm100` (5,1) |
| 278 | `TASK_MLA_KV_GATHER_SM100` | `mla_kv_gather_sm100` (4,0) |
| 279 | `TASK_SAMPLING_PARTIAL_SM100` | `sampling_partial_sm100` (1,2) |
| 280 | `TASK_MOE_TOPK_SIGMOID_SM100` | `moe_topk_sigmoid_sm100` (2,3) |
| 281 | `TASK_ELEMENTWISE_ADD_SM100` | `elementwise_add_sm100` (2,1) |
| 282 | `TASK_SOFTMAX_GATHER_SM100` | `softmax_gather_sm100` (2,1) |
| 283 | `TASK_MTP_VERIFY_PROBABILISTIC` | `mtp_verify_probabilistic` (5,2) |
| 284 | `TASK_PROB_SCATTER_SM100` | `prob_scatter_sm100` (2,1) |
| 285 | `TASK_MTP_FLOAT_SCATTER` | `mtp_float_scatter` (1,1) |
| 286 | `TASK_PROB_EXTRACT_SM100` | `prob_extract_sm100` (2,1) |
| 287 | `TASK_MLA_MTP_DECODE_TP2_SM100` | `mla_mtp_decode_tp2_sm100` (2,2) |
| 288 | `TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100` | `mla_mtp_decode_tp2_reduce_sm100` (2,1) |
| 289 | `TASK_MLA_MTP_DECODE_TP4_SM100` | `mla_mtp_decode_tp4_sm100` (2,2) |
| 290 | `TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100` | `mla_mtp_decode_tp4_reduce_sm100` (2,1) |
| 291 | `TASK_MLA_MTP_DECODE_TP8_SM100` | `mla_mtp_decode_tp8_sm100` (2,2) |
| 292 | `TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100` | `mla_mtp_decode_tp8_reduce_sm100` (2,1) |
| 293 | `TASK_MLA_KV_GATHER_SPLIT_SM100` | `mla_kv_gather_split_sm100` (5,0) |
| 294 | `TASK_MTP_BUILD_EMBED_INPUT` | `mtp_build_embed_input` (1,1) |
| 295 | `TASK_MLA_PREFILL_TP8_SM100` | `mla_prefill_tp8_sm100` (4,1) |
| 296 | `TASK_DFLASH_ATTENTION_SM100` | `dflash_attention` (5,1) |
| 297 | `TASK_DFLASH_NORM_ROPE_SM100` | `dflash_norm_rope` (4,1) |
| 298 | `TASK_DFLASH_KV_STORE_SM100` | `dflash_kv_store` (2,1) |
| 299 | `TASK_SM100_TASK_END` | 占位符 |

**SM100 块（55 条）远大于 Hopper 块（14 条）**，且集中在三族：**MoE**（W13/W2 × bf16/fp8/sm90/sm100）、
**MLA/MTP**（decode/prefill/gather/TP2/4/8 + MTP verify/scatter/commit）、**spec-decoding & sampling**
（Eagle3、DFlash、topk、prob scatter/extract）。这正是 `docs/23` §6「生产案例」的镜像：MPK 的能力面被
**当前主流开源模型的算子集**拉满。

### 2.5 多卡 / 模型家族专用

| ID | `TaskType` | 登记名 (ni,no) |
|---|---|---|
| 300 | `TASK_MULTIGPU_TASK_BEGIN` | 占位符 |
| 301 | `TASK_NVSHMEM_ALLGATHER_STRIDED_PUT` | `nvshmem_allgather_strided_put` (1,1) |
| 302 | `TASK_NVSHMEM_TILE_ALLREDUCE` | `nvshmem_tile_allreduce` (1,1) |
| 349 | `TASK_MULTIGPU_TASK_END` | 占位符 |
| 350 | `TASK_INKLING_TASK_BEGIN` | 占位符 |
| 351 | `TASK_INKLING_SCONV_SM100` | `inkling_sconv` (3,1) |
| 352 | `TASK_INKLING_MOE_ROUTER_SM100` | `inkling_moe_router` (3,3) |
| 353 | `TASK_INKLING_ATTENTION_SM100` | `inkling_attention` (7,1) |
| 399 | `TASK_INKLING_TASK_END` | 占位符 |
| 400 | `TASK_GLM_TASK_BEGIN` | 占位符 |
| 401 | `TASK_GLM_MOE_ROUTER_SM100` | `glm_moe_router` (2,3) |
| 449 | `TASK_GLM_TASK_END` | 占位符 |
| 450 | `TASK_GPT_OSS_TASK_BEGIN` | 占位符 |
| 451 | `TASK_CLAMPED_SWIGLU` | `moe_clamped_swiglu` (1,1) |
| 499 | `TASK_GPT_OSS_TASK_END` | 占位符 |

两个多卡 task 与 `docs/17` §6 的两种 AllReduce 精确对应（策略 A `AllGather+Reduce` ↔
`NVSHMEM_ALLGATHER_STRIDED_PUT` + `TASK_REDUCE`；策略 B NVLS tile one-shot ↔
`NVSHMEM_TILE_ALLREDUCE`）。**Inkling/GLM/GPT-OSS 三块证明这个 `mpk` 分支是某个真实服务的生产框架**
（不是论文 demo），且把「模型专属算子」也纳入同一张 `TaskType` 表——**任何新模型都要在编译期占一段
ID 区间**。

## 3. 前端登记表：95 处赋值、96 个名字 → 92 个 `TaskType`

`Graph::register_task`（`src/kernel/graph.cc:439-930`）用 `name == "..."` 分支把前端名字映射到
`TaskType`（`docs/14` §3.2 已证它是「95 分支大表」）。精确计数（`grep` 复核）：

- **96 个不同 `name == "..."` 字符串**，但其中一处是 `name == "a" || name == "b"` 的**合并分支**
  （`:635` 的 `linear_with_residual_sm100 || linear_with_bias_sm100`），所以
- **95 处 `task_config[op] = ...` 赋值**（`grep -c 'task_config\[op\] *='` = 95，与 `docs/14` §3.2 一致），
- 落成 **92 个不同 `TaskType`**。

即 name 空间比 `TaskType` 空间**更细**，存在 4 组「多 name 共用一 Type」：

| `TaskType` | 共用它的前端名 | 形态 | 说明 |
|---|---|---|---|
| `TASK_SILU_MUL` | `silu_mul`、`moe_silu_mul` | 两处赋值 | 同一逐元素实现，两个语义词 |
| `TASK_SAMPLING_SM100` | `sampling_sm100`、`sampling_reduce_sm100` | 两处赋值 | 采样与它的 reduce 段共用一个入口 |
| `TASK_QUANTIZE_FP8_SM100` | `quantize_fp8_sm100`、`quantize_fp8_f32scale_sm100` | 两处赋值 | 缩放参数形式不同，实现同源 |
| `TASK_LINEAR_WITH_RESIDUAL_SM100` | `linear_with_residual_sm100`、`linear_with_bias_sm100` | 同一 `\|\|` 分支 | 源码注释自陈：*"A bias is a residual whose row stride is zero … two variants of one task type, not a second task type."*（`:637-638`） |

方向反过来——**「一名多 Type」在代码里不存在**：没有任何 `name ==` 分支按运行时条件返回不同
`TaskType`，**代际切换靠 Python 侧拼不同字符串**完成（§1 的 `target_cc` 三元表达式
`persistent_kernel.py:881`）。这条「name 细、Type 粗」的设计使**同一 `TaskType` 的不同语义变体
（bias=行 stride 为 0 的残差；采样 reduce；FP8 缩放形式）在编译期塌缩成同一段 device 代码**，
只靠 `variant_id` / `params` 区分——与 `docs/07` §0「一个 `TaskType` → 一个 `task_register` 生成函数」
的映射一致。

## 4. 「无登记」的 `TaskType`：遗留与标记

把 §2 全表与 §3 的 92 个被登记 Type 求差，**没被任何 `register_task` 名字产出**的条目只有 3 个
**可执行 task**（外加若干非 task 标记）：

| `TaskType` | 类别 | 判断 |
|---|---|---|
| `TASK_ATTENTION_2`(104) | 遗留 | `"attention"` 只登记 `TASK_ATTENTION_1`(103)，`_2` 无生产者 |
| `TASK_PAGED_ATTENTION_2`(117) | 遗留 | 通用 `"paged_attention"` 只登记 `TASK_PAGED_ATTENTION_1`(116)；SM100 上则登记 `TASK_ATTN_SM100`(257)，跳过了 `_2` |
| `TASK_ALLREDUCE`(106) | 遗留（自陈） | 源码注释 *"will be removed soon"*；多卡改用 `TASK_REDUCE` + NVSHMEM |

非 task 标记（不计入「无登记」）：`TASK_TERMINATE`(0)/`TASK_BEGIN_TASK_GRAPH`(10)（图哨兵）、
`TASK_SCHD_*`/`TASK_GET_*`(200-204)（scheduler 控制面）、`TASK_SM100_TMA_START_TASK`(231) 与
`TASK_SM100_TMA_END_TASK`(256)（TMA 段起止标记）。另有一个**不在枚举内**的哨兵
`TASK_INVALID_ID`（`unsigned long long const`，`:72`），用于 `TaskId` 编码。

> `TASK_ALLREDUCE` 的「遗留」有旁证：MPK 的 `allreduce_layer`（`persistent_kernel.py:2488`）并不登记
> `TASK_ALLREDUCE`，而是经 `auto_select_allreduce_implementation`（`:17`）选实现——多卡走
> `nvshmem_allgather_strided_put`(301) + `reduction`(107)，或 `nvshmem_tile_allreduce`(302)
> （`docs/17` §6）。`docs/14` §7 说「精读 `allreduce` layer」指的是这个 **layer**，不是 `TASK_ALLREDUCE`。

**`_1`/`_2` 成对却只用 `_1`**（attention / paged_attention）是一个可核对的上游事实：枚举为
「一段算子的多个 phase」预留了编号，但当前前端只产出 `_1`（SM100 的 paged attention 干脆换了
`TASK_ATTN_SM100` 这个新号）。这与 `docs/22` §6「已实现未启用 / 未接线」清单同型——**枚举有编号 ≠
前端有生产者**。（本文只证「无 `register_task` 名字产出」：`task_config` 是进入 tGraph 的唯一入口
（`annotated_graph.cc:287`），故记为「读码可见的死码候选」，非 bug 判定。）

## 5. ID 空洞：删除后不复用

枚举里有几段**故意留空、不回填**的编号：

- `234`、`237–247`、`250` 在 SM100 块里**缺号**（SM100 块 230–299 只有 55 个条目，槽位 70）；
- `165–197` 在 Hopper 块里**整段空**（只用到 164）；
- 模型家族块（300–349 / 350–399 / 400–449 / 450–499）各自只填 2–3 个，其余**全空**。

这不是疏漏，而是**代际命名空间 + 只增不改**的直接后果：`TaskType` 数值会进 `TaskDesc`/序列化
（`docs/06` §2），删一个 task 不能回收它的号（否则老编译产物/老 event 会串号），只能留洞。
**读 MPK 代码时，`TaskType` 的数值大小完全不代表算子顺序或优先级**——它只编码「哪个代际块」。

## 6. 与论文 / 其他专题的对照

- **论文 §4.1 operator decomposition**（快照行 462–468）讲的是**自动路径**把算子切成 task；本文的
  `TaskType` 属于**生产路径**——手工把「一个高层算子 = 一个 `TaskType`」写死（`docs/14` §34、
  `docs/22` §5「同一套惯用法实现两遍」）。论文没提这套枚举，它是线上实现自己长出来的。
- **`docs/07` §0**：一个 `TaskType` = 一个 device 实现（`task_register.cc` 的 `register_*_task` 系列），
  本文补上了「一共有哪些 Type、按什么块分区」的上层视图。
- **`docs/02` 生产案例**：Hopper 块的 `MOE_W13/W2_LINEAR_SM90` 与 DeepGEMM MoE 的 W13/W2 分解同构；
  SM100 块的 `MOE_*_FP8_SM100` 对应 DeepGEMM 的 FP8 mega MoE 方向。**MPK 用 `TaskType` 把
  DeepGEMM 手写的那些算子变成了可登记的编译期枚举**。
- **`docs/23` §5「少而重」**：本文给出「重」的另一面——**可融合的 task 类型有 90+ 种**，所以
  「多而杂」的风险是真实的（`docs/12` 的共驻税/足迹税正以此为前提）；生产里只会挑其中少数几条
  编进同一张图。

## 7. 局限与复现要点

- **纯源码走读，未构建/运行**：所有结论基于 `f9eb70c` 快照的枚举与登记表行号；不采信性能数。
- **计数口径**：`enum TaskType` 共 **116 条命名条目** = 12 个 `*_BEGIN/_END` 范围占位符 + 若干控制/段
  标记（`TASK_TERMINATE`/`TASK_BEGIN_TASK_GRAPH`/`TASK_SCHD_*`/`TASK_SM100_TMA_START_TASK`）+
  **92 个被前端登记的 compute/comm `TaskType`** + 3 个未被登记的遗留条目（§4）。前端侧是
  **95 处 `task_config[op] =` 赋值 / 96 个 `name ==` 字符串 → 92 个 `TaskType`**。`docs/14` §3.2
  「95 处 `task_config[op]=`」与此互相印证（`grep -c` 可复核 95）。
- **name→Type 由脚本解析 `graph.cc` 得到**：`graph.cc` 有 `name == "a" || name == "b"` 合并分支
  （如 `linear_with_bias_sm100`）与带条件 `num_inputs` 的 `make_tuple`（如 `moe_w13_linear_sm100`
  的 `(params.size()==1 && params[0]>0) ? 5 : 4`），逐分支正则需专门处理；表中已保留，但「95/96/92」
  这三个数是**解析+`grep` 复核**的结果，个别边界（合并分支算 1 还是 2、条件元组）可能有一种的读法差。
- **「死码候选」不是 bug 判定**：§4 只证「无 `register_task` 生产者」，不排除手写路径/宏间接使用。
- **模型家族块的语义靠命名推断**：`INKLING`/`GLM`/`GPT_OSS` 块的算子含义由 task 名推断，未逐行走读
  其 device 实现（那些在 `task_register.cc` / `include/mirage/persistent_kernel/tasks/`）。

## 8. 参考坐标

- 线上实现：`https://github.com/mirage-project/mirage`（分支 `mpk`，快照 commit `f9eb70c`，2026-10-05）。
  主文件：`include/mirage/persistent_kernel/runtime_header.h:87-256`（`enum TaskType`）、
  `src/kernel/graph.cc:439-930`（`register_task` 95 分支）、`src/kernel/annotated_graph.cc:287-311`
  （`task_config` 消费）、`include/mirage/kernel/graph.h:229-233`（`task_config` 类型）。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（§4.1 行 462-468）。
- 同专题相关篇：`docs/14`（前端登记表，本文补其 §7 尾巴）、`docs/07`（单 task device 实现）、
  `docs/06`（`TaskDesc`/`TaskId` 运行层）、`docs/17`（多卡 task）、`docs/22`（「已实现未启用」清单）、
  `docs/23`（「少而重」适用边界）。
