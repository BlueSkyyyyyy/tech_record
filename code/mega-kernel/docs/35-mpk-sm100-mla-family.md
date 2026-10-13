# MPK SM100 MLA attention 家族走读：KV gather / decode / reduce / prefill（P121-1）

> 定位：本文补掉 `docs/34-mpk-sm100-cute-tasks.md` **§6「局限」明留的尾巴**（原话）——
>
> > **未走读的相邻层**：`mla_prefill_sm100.cuh`/`mla_prefill_tp8_sm100.cuh`、`mla_decode_sm100.cuh`、
> > `mla_kv_cache_gather*`、`mla_sm100_2sm.cuh`、注意力之外的 SM100 task 家族仍未逐行（留待后续）。
>
> `docs/34` 走完了 MLA-MTP decode 家族（`mla_mtp_decode_sm100.cuh` + TP2/4/8），本文把**同一 attention 段
> 的其余 device 身体**打开：**KV-cache gather（两变体）/ decode / reduce / prefill（吸收版）/ prefill_tp8
> （未吸收版）**。`mla_sm100_2sm.cuh`（3165 行，**2SM 协作版**）与注意力之外的 SM100 task 家族**仍留待后续**
> （见 §9）。快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/34` 同快照）。
> **纯源码走读（资料分析），无新实验**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **KV-cache gather 与 prefill 是「真接线」的 device 身体**：`mla_kv_cache_gather_sm100.cuh`（160 行）/
   `mla_kv_cache_gather_split_sm100.cuh`（138 行）由 `mla_kv_gather_layer`/`mla_kv_gather_split_layer`
   →`graph.cc:861/866`→`register_mla_kv_gather{,_split}_sm100_task`（`task_register.cc:4878/4940`）发射；
   `mla_prefill_sm100.cuh`（715 行）由 `register_mla_prefill_sm100_task`（`task_register.cc:4275`，`:4351`
   调 `mla_prefill_sm100_task_impl`）发射。三者都是「paged KV → 连续 buffer（TMA 可读布局）→ attention」这条
   **前置数据搬运 + 分块预填充**链上的活代码。
2. **decode / reduce 这一对被 MTP 版取代、不进生产**：`mla_decode_sm100.cuh`（460 行）里的
   `mla_decode_sm100_task_impl` 全仓**只被一个 standalone 单测 wrapper 调用**
   （`tests/runtime_python/blackwell/sm100_mla/runtime_kernel_wrapper_mla_decode.cu:24`）；生产
   `register_mla_decode_sm100_task`（`task_register.cc:4102`）实际发射的是 **`mla_mtp_decode_sm100_task_impl<false>`**
   （`:4168`，注释 *"Use PR 651 MLA MTP decode kernel"*）。`mla_reduce_sm100.cuh`（77 行）里的
   `mla_reduce_sm100_task_impl` **全仓无任何调用者**，生产 reduce 走
   `mla_mtp_reduce_sm100_task_impl<256>`（`task_register.cc:4248`）。**即 `docs/34` 的 MTP 家族把旧
   decode/reduce 全面顶替**——这是本专题「已实现未接线」模式的第三种形态（同文件族内被后继实现覆盖）。
3. **prefill 有「吸收版」与「未吸收版」两条身体，对应两条并行策略**：`mla_prefill_sm100.cuh` 走
   **absorbed latent**（`D_CKV=512` + `D_KPE=64`，K/V 分两块 SMEM），用 `cp.async` + Ampere 风格
   `m16n8k16` MMA、256 线程、按 `(head, q_block, batch)` 三维铺 grid；`mla_prefill_tp8_sm100.cuh`（486 行）
   走 **unabsorbed**（`D_QK=192` 拼一起），用 **TMA 载 K/V** + 只让 **128 线程**参与、`bar.sync 3,128`
   命名屏障、按 `(head, q_block, batch)` 但 head 维是 **TP8 分片后的局部队头**。两者都在 DeepSeek-V3
   builder 里由 **`mbt >= 32` 的编译期开关**（`builder.py:245`）决定走哪条，并配合 prefill/decode 的
   **dual-dispatch**（prefill 跳过 `Q_LEN<16`、decode 跳过 `Q_LEN>8`）。

判据口径同 `docs/07`/`docs/29`–`docs/34`：**一个 device task 身体只贡献 C3**（单 task 可表达 / 数值正确），
C1/C2/C4/C5 由运行层拿（本文 §7 重申）。

---

## 1. 定位：MLA attention 家族在编译栈里的坐标

| 层 | 本文覆盖 | 上游/下游 |
|---|---|---|
| 前端（`docs/14`/`docs/25`/`docs/26`） | `PersistentKernel.mla_{kv_gather,kv_gather_split,decode,reduce,prefill,prefill_tp8}_layer` | DeepSeek-V3 builder 的 MLA 段 |
| task 目录（`docs/24`） | `TaskType` 266/267/268/278/293/295 家族 | 代际 ID 块 |
| 生产布线（`docs/31`） | `graph.cc` op→`register_*`→`register_task_variant` | `TaskType`/`task_config` |
| **device 身体（本文）** | **`mla_kv_cache_gather{,_split}_sm100.cuh` / `mla_decode_sm100.cuh` / `mla_reduce_sm100.cuh` / `mla_prefill_sm100.cuh` / `mla_prefill_tp8_sm100.cuh`** | `docs/07`（Hopper）/`docs/21`（Blackwell 自动）/`docs/32`/`docs/33`/`docs/34` |
| 运行层（`docs/06`/`docs/22` §8） | 常驻 worker 调 `_execute_task` 后进这些身体 | C1/C2/C4/C5 只在此拿 |

生产 `TaskType`（`runtime_header.h:153-189`，全部逐行核对）：

```text
TASK_MLA_DECODE_SM100        = 266   TASK_MLA_REDUCE_SM100       = 267
TASK_MLA_PREFILL_SM100       = 268   TASK_MLA_KV_GATHER_SM100    = 278
TASK_MLA_KV_GATHER_SPLIT_SM100 = 293 TASK_MLA_PREFILL_TP8_SM100  = 295
```

（MTP decode/reduce 与 TP 变体 `269/270/287–292` 见 `docs/34`；本文只覆盖上表六条。）
`task_header.cuh` 的 include 关系（`:40-53`）本身透露了接线：它**直接** include
`mla_kv_cache_gather{,_split}_sm100.cuh`（`:41-42`）、`mla_prefill_sm100.cuh`（`:51`）、
`mla_prefill_tp8_sm100.cuh`（`:52`）、`mla_reduce_sm100.cuh`（`:52`）、`mla_sm100_2sm.cuh`（`:53`），
但**不**直接 include `mla_decode_sm100.cuh`——后者只能经 `mla_dispatch_sm100.cuh`（`:40`）间接进入，
而 dispatch 自身在快照里**没有生产调用者**（§3）。

---

## 2. KV-cache gather：把 paged KV 拼成 TMA 可读的连续序列

两个文件做同一件事的两种输出布局：**非 split** 输出单块 `[max_seq_len, D_K=576]`（`c_latent|k_pe`
逐位置拼接），**split** 输出两块 `ckv_sep [max_seq_len, D_V=512]` + `kpe_sep [max_seq_len, D_K-D_V=64]`
（chunked-prefill 的 `mla_prefill_sm100` 要的布局，`mla_kv_cache_gather_split_sm100.cuh:8-11`）。

**共同骨架**（`mla_kv_cache_gather_sm100.cuh:53-70`，block=128 线程，grid=`(max_num_batched_requests,1,1)`）：

1. 从三张 1D 元数据表算本请求的窗口——`qo_indptr`（新 token 区间）、`paged_kv_indptr`（页区间）、
   `paged_kv_indices`（页号）、`paged_kv_last_page_len`（末页长度）；`seq_len = (num_pages-1)*PAGE_SIZE +
   last_page_len`（`:70`）。
2. **Step 1 追加新 token**（`:87-113`）：把本步新的 `c_latent[512]` 与 `k_pe[64]` 写进**页式缓存**的正确
   槽位（`page_idx*PAGE_STRIDE + pos_in_page`），用 `uint4`（8 个 bf16）向量化。
3. **Step 2 gather**（`:116-157`）：把整条序列从页式缓存拷进连续 buffer。**关键工程细节**（注释 `:119-123`
   自陈）：最初用「每个 `seq_pos` 一轮」的两级循环，只有 `D_K/8` 个 lane 活跃、且每轮一条依赖的
   load→store 链，**在 GB300 上 seq_len≈100 时慢 ~2.5×**；改成**扁平化 `(seq_pos × d8)` 访存 + 4 路
   ILP**（先发 4 个 `uint4` load 再统一 store，`:127-146`），每轮全部 128 线程都在工作、且打断了延迟链。

split 版（`mla_kv_cache_gather_split_sm100.cuh`）Step 1 完全同（缓存布局仍拼接），Step 2 改成逐
`seq_pos` 循环里分别写 `ckv_sep`/`kpe_sep`（`:113-135`）——即它**没有**做扁平化 ILP 优化（`seq_len` 循环
`:114`）。两文件的 `k_pe_new` 行 stride 默认 `K_PE_ROW_STRIDE = D_K - D_V = 64`，但 DeepSeek-V3 builder
把 `k_pe_out` 分配成 `[mbt, 128]`（padding 到 MMA_M=128），故注册时**显式传 `k_pe_row_stride=128`**
（`task_register.cc:4896/4961`），只前 64 列是真 rope 数据。

---

## 3. decode：旧 B200 standalone 身体（生产改走 MTP 版）

`mla_decode_sm100.cuh` 是 DeepSeek-V3 MLA decode 的原始 B200 实现（文件头 `:16-18`）：
**warp-specialized TMA/tcgen05 流水**，把 QK 与 PV 两条 GEMM 切成 `tile_s=128`、`BK=64`、`MMA_K=16` 的
分块（常量 `:30-42`）。结构：

- **两个 warp 分工**：`wid0`/`wid1`（`elect_sync()` 单线程代表）。`wid0` 管 TMA 载入（Q、K、V），
  `wid1` 管 `tcgen05` MMA 与 TMEM 分配——注意 `wid1` 用 `tcgen05.alloc.cta_group::1`（`:108-114`）+
  `dealloc`（`:453-457`），**这正是 `docs/34` §3.5 点名「常驻 kernel 内 TMEM alloc/dealloc 必须同一 warp」
  的隐患来源**：TP 版后来统一改成 warp0。
- **QK 阶段**（`:148-206`）：`ML A_K_ITERS=9` 次 K 分块，`MLA_NUM_QK_STAGES=4` 级 TMA 双缓冲，逐 stage
  `mbar_wait`+`tcgen05_mma`+`tcgen05_commit`；Q 只在头部一次 TMA 载入（`:119-135`）。
- **Softmax 阶段**（`:208-342`）：`tcgen05.ld` 从 TMEM 取 S，`row_max`/`corr`/`ts` 在线 softmax，
  再把 P 转 bf16 写回 SMEM（含手写 swizzle `:333`）。
- **PV 阶段**（`:344-395`）：`MLA_V_CHUNKS=8` 个 V 分块，`MLA_NUM_PV_STAGES=2`。
- **累加**（`:397-447`）：从 TMEM 读 PV 结果，`is_first` 时直接写、否则 online 修正
  `corr_inv*O_old + t*v_inv`；末尾写 LSE `La`（`:449-450`）。
- **线程守卫**（`:69-72`）：`if (tid >= MLA_TB) return;`（`MLA_TB=128`），为 MPK 的 256 线程 worker 留。

**接线真相**：文件提供 `mla_decode_sm100_task_impl(...)`（签名 `:56-66`，只有 `sk/si/bi`，**没有**
`q_len/head_group`）与一个特化 dispatch `mla_decode_dispatch<128,576,512>`（`mla_dispatch_sm100.cuh:52-72`）。
二者在快照里**都只有 standalone 单测 caller**。生产 `register_mla_decode_sm100_task`（`task_register.cc:4102`）
的注释 `:4167` 写明改用 *"PR 651 MLA MTP decode kernel"*，`:4168` 发射
`mla_mtp_decode_sm100_task_impl<false>`（带 `q_len/gi/si/bi` 全套参数）。**故 `mla_decode_sm100.cuh`
是被 `docs/34` 的 `mla_mtp_decode_sm100.cuh` 顶替的旧身体**，Python 层仍叫 `mla_decode_layer`/
`"mla_decode_sm100"`（`persistent_kernel.py:1627/1658`），但落地的 device 代码已换成 MTP 版。

---

## 4. reduce：per-thread-per-head 三趟，全仓无调用者

`mla_reduce_sm100.cuh`（77 行）把 split-KV decode 的**部分输出**合并：每个线程处理**一个 head**
（`if (tid >= NUM_HEADS) return; :40-42`），三趟扫 splits（`:46-74`）——

1. Pass 1 找 `lse_max`（`:47-51`）；
2. Pass 2 算 `sum_exp` 与 `inv_sum`（`:54-59`）；
3. Pass 3 对 `d_start..d_start+d_count` 逐维累加 `exp(lse-lse_max)*inv_sum * Oa[...,di,h]` 并写 bf16
   `O[batch,h,di]`（`:62-74`）。

3 趟里每趟都重新读 `La`（`:49/56/65`），是**朴素但直白**的实现。**关键事实**：`grep
mla_reduce_sm100_task_impl` 全仓**零调用者**；生产的 `register_mla_reduce_sm100_task`（`task_register.cc:4206`）
发射的是 `mla_mtp_reduce_sm100_task_impl<256>`（`:4248`，注释 *"PR 651 MLA MTP reduce kernel
(256 threads for MPK)"*）。即 reduce 也被 MTP 版顶替，本文件是**死代码**（同 §3）。

---

## 5. prefill（吸收版）：cp.async + m16n8k16，256 线程

`mla_prefill_sm100.cuh`（715 行）是 DeepSeek-V3 **MLA 吸收后**的 chunked prefill（文件头 `:16-19`：
*"Ampere-style m16n8k16 MMA with cp.async, online softmax, causal masking; 每 task 一个 (head, q_block)
对；256 线程 = 8 warp，~216KB SMEM"*）。要点：

- **形状**：`PF_D_CKV=512`、`PF_D_KPE=64`、`PF_BM=64`、`PF_BN=64`、`PF_MMA 16n8k16`（常量 `:35-61`）。
  SMEM 布局 = `Q_nope | Q_pe | stages*(CKV | KPE) | softmax 状态`（`:67-70`）。
- **单 stage 硬约束**（`:44-51`）：`PF_NUM_STAGES=1`。注释明确——2 stage 需 ~221KB，而 MPK 在 B200 的
  per-worker 动态 SMEM 预算 = 207KB−3KB 静态 = **204KB**，2 stage 溢出会触发
  *"Invalid `__shared__` write of size 16 bytes"*。降成 1 stage 后 148KB 可用，**KV 载入与计算改为串行**
  （正确性不变）。这是**常驻内核 SMEM 预算对 task 内部流水的直接让步**（同族：`docs/32` `kNumSMs=1`、
  `docs/33` TMEM 不 relinquish、`docs/34` TMEM warp 一致）。
- **双跳成分**（`:429-475`）：QK 拆成「`k_pe` 部分 4 个 MMA_K」+「`ckv` 部分」两段，与吸收后的
  `[Q_nope|Q_pe]·[CKV|KPE]^T` 求和一致。
- **1-stage 的顺序陷阱**（`:413-419` 注释）：1 stage 时 `next_stage == stage`，若在 tile 头预取下一块会
  覆盖当前块；故把预取**移到本迭代 PV 之后**、并加 `__syncthreads()`（`:625-635`）。注释自陈「只有在
  1-stage 时 tile N≥1 会被静默写坏」。
- **命名屏障保留区**（`:525-531/598-599`）：softmax 与写 P 用 `bar.sync {8+warp_m}, 64`，注释列出
  CUTLASS 保留的 barrier ID（Epilogue=1…Sm120Mainloop=7），**误用保留 ID 会留脏状态毒死后续共用同一
  MPK worker CTA 的 linear/MoE task**——与 `docs/34` 的 `bar.sync 12,128` 同一族工程约束。
- **dual-dispatch gate**（`:220-230`）：`Q_LEN < 16` 直接 `return`（小 chunk 交给 decode/MTP decode，
  后者快 ~10×）；`q_start >= Q_LEN` 也提前返回（避免 decode 步里 `q_block>0` 白烧 `O(KV_len)` 计算，
  注释引 `bugfix.md Bug 20`，`:239-250`）。
- **causal masking**（`:478-501`）：`kv_pos <= q_hist + q_pos` 且 `< S` 且 `q_pos < Q_LEN`，否则置
  `-INFINITY`；`q_hist = S - Q_LEN`（`:237`）把「chunk 内位置」映射到「全序列位置」。
- **在线 softmax + rowsum**：`ex2.approx`（`fast_exp2f :183-187`）、`mma_rowsum_bf16`（`:137-149`，
  用一条 MMA 的稀疏输出直接把 P 按行求和）——与 FlashAttention 惯用法同源。

---

## 6. prefill（未吸收 TP=8 版）：TMA K/V + 128 线程守卫

`mla_prefill_tp8_sm100.cuh`（486 行）是**未吸收**的 TP8 预填充（文件头 `:16-25` 列了相对 standalone
`__global__` 的四处改法：`__global__→__device__ __noinline__`、TMA 描述符改成 `CUtensorMap const*`
参数、`__syncthreads()` 改 `bar.sync 3,128`、入口加 thread guard）。要点：

- **形状**：`D_QK_NOPE=128`、`D_QK_ROPE=64`、`D_QK=192`、`D_V=128`、`BM=64`、`BN=128`、`NT=128`
  （常量 `:48-59`）。SMEM = `Q_nope | Q_pe | KN0 | KN1 | KP | V0 | V1 | mbar`（`:61-74`），
  每块 `[BN,64]` TMA block 16KB。
- **TMA 载 K/V**（`:290-302`）：每次 `mbar_tx(5*TMA_BLK)` 后发 5 条 `tma3d`（K 的 nope 两半 + rope +
  V 两半，`:293-297`）；单 mbarrier 累计一次等待（`:307`）。
- **128 线程命名屏障**（`:191-195`）：`task_sync()` = `bar.sync 3,128`，注释说明 barrier 0–2 被其它 v2
  task 占用——**又一处常驻共驻的命名屏障纪律**。
- **SMEM 1024 对齐**（`:223-230`）：MPK 动态 SMEM 起始只 128 对齐，而 128B swizzle TMA 要求目的地
  1024 对齐，故 `sb = (sb+1023)&~1023`（≤1KB 代价）。
- **QK/softmax/PV/归一化**（`:311-483`）：与吸收版同一套 `ldm4/ldm4t`+`hmma16`+`ex2`+rcp 例程，只是
  K 是 `[K_nope(128)|K_rope(64)]` 拼一起（`NMDK=8`、`NMRK=4`）。causal mask 在 `:358-373`。

---

## 7. 生产布线：谁走哪条路

- **Python 层**（`python/mirage/mpk/persistent_kernel.py`）：`mla_kv_gather_layer`（`:1569`，注册
  `"mla_kv_gather_sm100"`→`:1592`）、`mla_kv_gather_split_layer`（`:1594`→`:1625`）、`mla_decode_layer`
  （`:1627`→`"mla_decode_sm100"`:`1658`）、`mla_reduce_layer`（`:1660`→`:1688`）、`mla_prefill_layer`
  （`:1690`→`"mla_prefill_sm100"`:`1718`）、`mla_prefill_tp8_layer`（`:1720`→`:1747`）。
- **graph.cc dispatch**：`mla_decode_sm100`（`:767`）、`mla_reduce_sm100`（`:772`）、`mla_prefill_sm100`
  （`:777`）、`mla_prefill_tp8_sm100`（`:784`）、`mla_kv_gather_sm100`（`:861`）、
  `mla_kv_gather_split_sm100`（`:866`）。
- **编译期选路**（`models/deepseek_v3/builder.py:245`）：`self._use_prefill = mbt >= 32`——大 chunk 走
  `mla_prefill_sm100`、小 chunk（含 MTP speculative，`spec_length ≤ 7`）走 decode/MTP decode；阈值同时
  与两条 dual-dispatch gate（prefill `Q_LEN<16` return、decode `Q_LEN>8` return，见 §5/`docs/34`）配合，
  保证两者**定义域不重叠**。
- **task 注册参数**（`task_register.cc`）：decode `:4102`、reduce `:4206`、prefill `:4275`、prefill_tp8
  `:4369`、kv_gather `:4878`、kv_gather_split `:4940`。softmax scale 两处都显式计算
  `(1/sqrt(192))*mscale²`、`mscale = 0.1*log(40)+1`（`:4184-4188`/`:4293-4294`/`:4456-4460`）——注释特别
  强调 **`q_head_dim=192`（128 nope + 64 rope），不是吸收后的 `D_K=576`**。
- **内存布局契约**（`task_register.cc:4894-4896`/`4957-4961`）：`k_pe_out` 按 `[mbt,128]` 分配、
  `K_PE_ROW_STRIDE=128`，真实 rope 只前 64 列。

---

## 8. 与相邻文档的坐标关系与判据

- 与 `docs/07`（Hopper task 内部）/`docs/32`（SM100 手写线性）/`docs/33`（SM100 FP8 group-GEMM）/
  `docs/34`（CuTe 线性 + MLA-MTP）合起来，SM100 生产 task 的 device 身体覆盖面补齐到「**MLA attention 全段**」。
- 与 `docs/34` 的**分工**：`docs/34` 覆盖 MTP decode/reduce（+TP2/4/8）；本文覆盖 decode/reduce 的
  **旧 standalone 身体**（生产未接线）、KV gather 两变体、prefill 两身体。**headline 互补**：`docs/34` 说
  「MTP 家族是一整族 8 个 TaskType」；本文说「**旧 decode/reduce 因此成为同族内的死代码**」。
- 与 `docs/21`（Blackwell 自动 transpiler）/`docs/22` 的 2SM 协作：`mla_sm100_2sm.cuh`（3165 行）是
  **2SM 协作版 attention**，与 `docs/21` §2SM UMMA / P33 的 `cta_group::2` 原语同族，本文**未覆盖**（§9）。
- **判据**：本文六个 device 身体**均只贡献 C3**（单 task 可表达、数值由前端/运行层校验），C1/C2/C4/C5
  由运行层（`docs/06`/`docs/22` §8）拿——与 `docs/29`–`docs/34` 同口径。

---

## 9. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  「无调用者/生产改走 MTP」是**静态 `grep`** 结论，不构成运行时判定，也**不代表上游 bug**。
- **`bugfix.md` 不在快照里**：prefill 注释引的 Bug 20、`docs/34` 引的 Bug 14/19 在 `f9eb70c` 快照中
  **不存在**（只 `grep` 到引用），本文只转述源码注释自陈。
- **未走读的相邻层（留待后续）**：`mla_sm100_2sm.cuh`（3165 行，2SM 协作 attention——
  `task_header.cuh:53` 有 include、standalone 单测有 caller，但本文未逐行）、注意力之外的 SM100 task
  家族（`attention_sm100`/`dflash_*`/`inkling_*`/`glm_moe_router_sm100`/`prob_scatter_sm100`/
  `softmax_gather_sm100`/`topk_*`/`sampling_sm100`/`rotary_embedding_sm100` 等）、`mla_prefill_host.cu`
  这个 prefill 的原型来源（注释 `:201` 提到）。
- **未与生产端到端对跑**：SM100 路径本机不具备；`mla_prefill_tp8` 是对外部 standalone 的机械 port，本文
  只核了 port 后结构，未对照原始源码逐行。
- **split gather 未做扁平化 ILP**：与 `mla_kv_cache_gather_sm100` 的优化形态不同，原因未知（留待后续）。

---

## 10. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/34` 同快照）。
- **KV gather**：`include/mirage/persistent_kernel/tasks/blackwell/mla_kv_cache_gather_sm100.cuh`（160 行；
  `PAGE_STRIDE :58`、append `:87-113`、扁平化 ILP gather `:119-157`）；
  `mla_kv_cache_gather_split_sm100.cuh`（138 行；split 输出 `:113-135`）。
- **decode**：`mla_decode_sm100.cuh`（460 行；常量 `:30-44`；impl `:56`；alloc/dealloc `:108-114/453-457`；
  QK `:148-206`；softmax `:208-342`；PV `:344-395`；accumulate `:397-447`）；
  dispatch `mla_dispatch_sm100.cuh`（74 行；特化 `:52-72`）。
- **reduce**：`mla_reduce_sm100.cuh`（77 行；impl `:30`；三趟 `:46-74`）。
- **prefill**：`mla_prefill_sm100.cuh`（715 行；常量/单 stage `:35-51`；impl `:205`；dual-dispatch
  `:220-250`；QK `:421-476`；causal `:478-501`；softmax `:503-559`；PV `:601-623`；1-stage 预取 `:625-635`）。
- **prefill_tp8**：`mla_prefill_tp8_sm100.cuh`（486 行；port 注记 `:16-25`；常量 `:48-74`；`task_sync :193`；
  impl `:197`；TMA `:290-302`；QK `:311-357`；causal `:358-373`）。
- **生产布线**：`src/kernel/graph.cc:767/772/777/784/861/866`；
  `src/kernel/task_register.cc:4102`（decode→MTP `:4168`）、`:4206`（reduce→MTP `:4248`）、`:4275`
  （prefill `:4351`）、`:4369`（prefill_tp8 `:4386`）、`:4878`（gather `:4921`）、`:4940`（split `:4981`）；
  `include/mirage/persistent_kernel/runtime_header.h:153-189`（TaskType）；
  `python/mirage/mpk/persistent_kernel.py:1569/1594/1627/1660/1690/1720`；
  `python/mirage/mpk/models/deepseek_v3/builder.py:245`（`_use_prefill = mbt >= 32`）。
- **DeepSeek-V3 MLA 数学**：`sm_scale = (1/sqrt(192)) * mscale²`，`mscale = 0.1*log(40)+1 ≈ 1.36889`
  （`task_register.cc:4177-4188`）——`q_head_dim=192` 而非吸收后 `d_k=576`。
