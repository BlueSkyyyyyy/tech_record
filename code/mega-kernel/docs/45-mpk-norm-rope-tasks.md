# MPK 归一化 / 位置编码 task 家族走读（跨 Ampere/Hopper/Blackwell 三代）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/44` 同快照）。
> **纯源码走读（资料分析）**：SM100 在目标卡 H100/sm90 不可执行；Hopper 版本可执行但本文不新跑
> （实测锚点复用 `docs/09` 里在 H100 上跑过的 test_mode RMSNorm）。所有行号为 **一手核对** 过。
>
> 本文补齐 `docs/37` §4（P123-1 对 SM100 norm/rope 的**摘要**）与 `docs/07` §3.1（只走
> `rms_norm_hopper_impl`）留下的两条空白：**(a)** Ampere/Hopper 层的 norm/rope 实现（此前从未打开）；
> **(b)** 把 norm/rope 当成「一个跨三代演化的 task 家族」看它的**融合边界**与**同步机制**。
> 顺带**修正 `docs/37` §4.2 的一处静态事实错误**（见 §3.3）。

---

## 0. 定位：三层形态与「谁没有独立 task」

归一化（RMSNorm）是每个 Transformer 层出现两次、位置编码（RoPE）紧随 Q/K 的**薄算子**——在 MPK 的
tGraph 里，它们贡献的 task 数不比 GEMM 少，但每个 task 的算术极轻。整个仓库里 norm/rope 有
**三层形态**，且只有前两层里的一部分被登记为独立 `TaskType`：

| 形态 | 函数 | 输入/输出 | 是否独立 task |
|---|---|---|---|
| **(A) 独立 global-memory norm task** | `ampere::rms_norm_impl` / `hopper::rms_norm_hopper_impl` / `blackwell::dflash_norm_rope_sm100` | 全局张量 → 全局张量 | **是**（`TASK_RMS_NORM`=119 / `TASK_RMS_NORM_HOPPER`=154 / `TASK_DFLASH_NORM_ROPE_SM100`=297） |
| **(B) 调用方 SMEM 上的融合 norm（可融 RoPE）** | `ampere::rms_norm` / `hopper::rms_norm_hopper` / `blackwell::rms_norm_sm100` | SMEM 视图原地 | 否——被 attention task **内联调用** |
| **(C) 纯 RoPE** | `ampere::rotary_embedding` / `hopper::rotary_embedding_hopper` / `blackwell::rotary_embedding_sm100` | SMEM 视图原地 | **没有独立 task**——只在 attention 身体里内联调用 |

一个立刻可核对的事实：**全仓没有 `TASK_ROTARY` / `TASK_ROPE` 这样的枚举**（`runtime_header.h` 的
`enum TaskType` 里只有 `TASK_RMS_NORM`/`TASK_RMS_NORM_LINEAR`/`TASK_RMS_NORM_HOPPER` 与 DFlash 的
`TASK_DFLASH_NORM_ROPE_SM100`）。RoPE 要么被折进 attention 的 QK-norm 段，要么被折进
`dflash_norm_rope`——**它自己从不作为 DAG 上的一个节点**。这正是本专题反复讲的「**融合边界**」：
越靠近注意力的逐位置算子，越倾向于被折进 attention 身体（省一次 SMEM↔寄存器的落 HBM 往返，
对上 `docs/04` §17 的损失三），代价是每个 attention task 变重、且把「norm/rope 是否要单独调度」
这一自由度交还给融合方。

---

## 1. 形态 A：独立 global-memory norm task 的两条身体

两条身体由 Python 前端按计算能力二选一（`persistent_kernel.py:881`）：
`target_cc >= 90 ? "rmsnorm_hopper" : "rmsnorm"`，对应 `graph.cc:571-574` / `graph.cc:449-452`
两条 dispatch 分支。**目标卡 H100/sm90 走的是 Hopper 那条。**

### 1.1 `ampere/rmsnorm.cuh` — `rms_norm_impl`（`TASK_RMS_NORM`=119）

- 模板 `rms_norm_impl<T, BATCH_SIZE, HIDDEN_DIM>`（`:19-23`），**硬断言 `BATCH_SIZE == 1`**（`:24`）：
  每个 rmsnorm task 只处理一行（一个 token）。登记处同样断言 `batch_size == 1`
  （`task_register.cc:111-112`）。
- 用 `extern __shared__` 手切三段 SMEM：`shared_input_buffer` / `shared_weight_buffer` /
  `shared_output_buffer` + 尾部 `reduce_smem`（`:57-65`），偏移全部 `sizeof(T)*HIDDEN_DIM`
  对齐。
- 向量化粒度由 `BYTES_PER_THREAD = HIDDEN_DIM/NUM_THREADS*sizeof(T)` 推出 `BYTES_PER_CP ∈ {16,8,4}`
  （`:29-42` 的三分支 `if constexpr`）——即 **128/64/32-bit `cp.async`**。`CHUNK_SIZE`/`TILE_SIZE`/
  `NUM_TILES` 全为 `constexpr`（`:39-42`）。
- 双缓冲软件流水：先 warm up 第 0 块（`:67-74`），主循环里预取第 `for_idx+1` 块并 `cp_async_wait<1>`
  （`:80-90`），最后一块 `cp_async_wait<0>`（`:91-93`）。**这就是 `docs/07` §1 在 GEMM 里描述的
  同一套 `cp.async` 常见惯用法**，只是用在了逐元素算子。
- 三段式归约：warp 内 `shfl_xor`（`:102-105`）→ 每 warp 写 `reduce_smem[warp_idx]`（`:106-108`）
  → 跨 warp `shfl_xor`（`:110-114`）→ `threadIdx.x==0` 写回（`:115-118`）。注意**跨 warp 段用的是
  `NUM_WARPS`=4 的固定展开**，不是全局 `NUM_WARPS` 之外的动态量。
- 输出用 128-bit 直接写回（`:130-142` 的 `if constexpr` 三分支）。

### 1.2 `hopper/rmsnorm_hopper.cuh` — `rms_norm_hopper_impl`（`TASK_RMS_NORM_HOPPER`=154）

与 Ampere 版逐行对照后，Hopper 版有**三处实质差异**：

1. **放开单 batch**：`static_assert(BATCH_SIZE == 1)` 被**注释掉**（`:24`），主循环外包一层
   `for batch_idx < BATCH_SIZE`（`:75-159`），每 batch 重算 `curr_d_input/curr_d_output`（`:77-80`）。
   登记处也把 `assert(batch_size == 1)` 注释掉（`task_register.cc:1575`）。
2. **处理非整除行宽**：`EVEN_SPLIT = (HIDDEN_DIM % NUM_THREADS == 0)`（`:29`）。不整除时
   `BYTES_PER_THREAD = HIDDEN_DIM*sizeof(T)`（整行一个线程搬），`TILE_SIZE` 与 `NUM_TILES` 改成上取整
   （`:47-48`），主循环末尾用 `min(TILE_SIZE, HIDDEN_DIM - for_idx*TILE_SIZE)` 覆盖**短尾 tile**
   （`:110-115`），且拷贝前对 `off < HIDDEN_DIM` 逐线程守卫（`:83-88`/`:98-103`）。**Ampere 版靠
   `static_assert(HIDDEN_DIM % TILE_SIZE == 0)` 把这一情形排除在外**（`ampere/rmsnorm.cuh:41`）。
3. **线程数是模板默认参**：`<..., int NUM_THREADS = 256>`（`:19`），而不是用全局 `NUM_THREADS`。

- **实测锚点（H100）**：`docs/09` §7–§8 在目标卡上跑的 `test_rmsnorm_testmode.py`（
  `batch_size=16, hidden_dim=4096`，`block_dim=(256,1,1)`，`test_rmsnorm_testmode.py:30-49`）
  经 `rmsnorm_layer` → `"rmsnorm_hopper"` → **正是这条 `rms_norm_hopper_impl`**，端到端每次调用
  36.4–39.1 µs（对照 PyTorch 朴素 kernel-per-operator 54.0 µs）。**这是本专题里唯一有真机数字的
  norm task**——把「一条薄 task 端到端要几十 µs，且主导项是常驻 kernel 启动协议而非计算」
  （`docs/09` §8.3）落实在一个 concrete 身体上。

---

## 2. 形态 B：调用方 SMEM 上的融合 norm（+ 可选 RoPE），三代

三者签名几乎一致（`smem_input, weight_ptr, reduce_smem, eps, window_size, token_offset,
rotary_emd, cos_ptr, sin_ptr`），差别只在**同步原语**与**是否支持 partial RoPE**：

### 2.1 `ampere/norm.cuh` — `rms_norm`

- 用 **cooperative groups**：`this_thread_block()` + `tiled_partition<ROTARY_PARTICIPATING_THREADS>`
  （`:37-41`），同步用 `__syncthreads()`（`:73`/`:86`）+ `participating_group.sync()`（`:103`/`:119`）。
  注释 `:37` *"Ampere doesn't support cutlass barrier, use cooperative groups."*
- **不支持 partial RoPE**：rope 段固定按 `HEAD_DIM/2` 配对（`:110-117`），没有 `ROTARY_DIM` 模板参。

### 2.2 `hopper/norm_hopper.cuh` — `rms_norm_hopper`

- 同步改用 `wg_sync<NUM_THREADS>(BARRIER_ID)`（`:77`/`:90`/`:109`/`:126`），`BARRIER_ID` 默认 9
  （`:26`）。`wg_sync` 是 `tasks/hopper/utils.cuh:30-36` 的一条 `bar.sync %0, %1` 内联汇编（仅在
  `MIRAGE_GRACE_HOPPER`/`MIRAGE_GRACE_BLACKWELL` 下生效，否则 `brkpt`）。
- **支持 partial RoPE**：新增 `ROTARY_DIM` 模板参（默认 `HEAD_DIM`，`:30`），rope 配对只作用于
  `[0, ROTARY_DIM)`、其余直通（`:113-130`，注释 `:27-29` 点名 **GLM-4.6：128 维中旋转 64 维**）。
- `NUM_THREADS` 是**显式模板参**（无默认，`:25`），调用方负责传——`attention_sm100.cuh` 一律传
  `128`（`:440`/`:460`）。

### 2.3 `blackwell/norm_sm100.cuh` — `rms_norm_sm100`

- 同步改用 **cutlass `NamedBarrier`**，且分开两个 ID：`CONSUMER_WARPGROUP_SYNC_BARRIER_ID`=6 与
  `ROTARY_SYNC_BARRIER_ID`=7（`:25-26`），两个 barrier 对象 `wg_barrier`/`wg_barrier_rotary`
  （`:40-46`）。`wg_barrier_rotary` 的参与线程数是 `ROTARY_PARTICIPATING_THREADS`
  （`min(NUM_THREADS, HEAD_DIM)`，`:42-46`）。
- **不使用模板线程数，而用全局 `NUM_THREADS`**（`worker_config.h:18` 定义 = **128**），并整段包在
  `if (threadIdx.x < NUM_THREADS)`（`:47`）里——即常驻 worker 若是 256 线程，也只有前 128 参与。
- partial RoPE（`ROTARY_DIM`）：`:109-136`，与 Hopper 版语义一致，但 barrier 换成
  `wg_barrier_rotary.arrive_and_wait()` 两次（`:114`/`:131`），且注释 `:112-113` 明写
  *"Both barriers are reached uniformly by every participating thread"*。

> `docs/37` §4.1 已描述过 §2.3 的语义；本节补的是**它与 §2.1/§2.2 的逐条对照**：三代其实是
> 「同一份 `shfl_xor` 蝶形归约 + 可选 rope 配对」的三次移植，变的是**同步原语**（cooperative
> groups → `bar.sync` → `NamedBarrier`）与**partial RoPE 的支持与否**。

---

## 3. 形态 C：纯 RoPE 三代——以及一枚死代码

### 3.1 `ampere/rotary_embedding.cuh` — `rotary_embedding`

cooperative groups 版（`:33-41`），两步「先读配对 v1/v2、再写回」由
`participating_group.sync()` 在 `:63`/`:75` 夹住。**无 partial RoPE**（固定 `HEAD_DIM/2`）。

### 3.2 `hopper/rotary_embedding_hopper.cuh` — `rotary_embedding_hopper`

`wg_sync<ROTARY_PARTICIPATING_THREADS>(BARRIER_ID)`（默认 ID 9，`:27`）在 `:58`/`:75` 夹住
读-配对-写回。**有 partial RoPE**（`ROTARY_DIM`，`:28-31`，`:60-78`）。模板里带
`int NUM_THREADS = 128`**默认参**（`:26`）。

**关键调用事实**：`blackwell/attention_sm100.cuh`（SM100 通用 attention）在 `qk_norm=false, rope=true`
的分支里，调用的**正是这个 hopper 函数**——`:435`（q rope）与 `:455`（k rope），显式传 `NUM_THREADS=128`。
即 **SM100 的 attention 复用 Hopper 的 RoPE**。

### 3.3 `blackwell/rotary_embedding_sm100.cuh` — **死代码（`docs/37` §4.2 勘误）**

`rotary_embedding_sm100` 用 cutlass `NamedBarrier` ID 6（`:34-35`）夹住 `:60`/`:72`，
其余与 Hopper 版同构。但**全仓（含 `tests/`）没有任何一处调用它**——只有
`attention_sm100.cuh:23` 的 `#include`（以及本文件 `:27` 的定义）：

```
include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh:23:#include "rotary_embedding_sm100.cuh"
include/mirage/persistent_kernel/tasks/blackwell/rotary_embedding_sm100.cuh:27:... void rotary_embedding_sm100(...)
```

**勘误 `docs/37` §4.2**：原文称「`attention_sm100` 在 `qk_norm=false` 但 `rope=true` 时选它
（`rotary_embedding_sm100`）」——**这与源码不符**。`attention_sm100.cuh` 在该分支实际调用的是
`rotary_embedding_hopper`（`:435`/`:455`），`rotary_embedding_sm100` 从未被调用。这与本专题记过的
「有定义、无调用者」死代码族（`docs/38` §2.4/§6、`docs/16`、`docs/20`、`docs/32`、`docs/33`）同源，
是又一处「**按文件/名字推断『哪份代码在跑』会错**」的实例——判 RoPE 走哪条身体必须追到
`attention_sm100` 的**实际调用符号**，而不能只看它的 `#include`。

### 3.4 一个安全的共性：两步配对同步

三代 pure-rope 都是「**先读 `v1=col`、`v2=col±HALF`，再写回 `col`**」——由于写 `col` 会覆盖同 warp
另一线程读的 `col±HALF`，必须在两步之间插一次 block/warpgroup 同步（cooperative groups sync /
`wg_sync` / `NamedBarrier`）。这是 RoPE 作为「原地做大范围配对」算子的固有形状，也是它**难以像 norm
那样纯 warp 内归约**的原因；`rms_norm_sm100` 把 rope 折进去时，正是复用了同一把两段同步（§2.3）。

---

## 4. DFlash 的独立 norm+rope：一条「一线程一 (token,head)」的身体

`blackwell/dflash_norm_rope_sm100.cuh` 的 `dflash_norm_rope_sm100`（`:35-42`）与形态 B 相反：

- **输入输出都是全局张量**（`x/weight/cos/sin/out`，`:47-51`），不是 SMEM 视图；
- **一线程处理一个 (token, head)**（`:53-60` 的 `p / NUM_HEADS`、`p % NUM_HEADS`），
  norm 走寄存器 `float nv[HEAD_DIM]`（`:72`），无任何跨 lane 通信；
- 融合顺序：先 RMSNorm（`:62-69`）再 NeoX RoPE（`:78-86` 的 `i < HALF ? -nv[i+HALF] : nv[i-HALF]`）。
  `v` 不做（`:25` 注释）。
- 它是**独立 task**（`TASK_DFLASH_NORM_ROPE_SM100`=297），登记 `task_register.cc:185-219`
  （`eps` 硬编码 `1e-5`，`:218`），dispatch `graph.cc:462-466`，前端 `persistent_kernel.py:1007-1028`，
  由 DFlash builder 调用于 `dflash/builder.py:261/318/320`。

**为什么 DFlash 需要一条独立的 norm+rope，而通用 attention 却内联？** 因为 DFlash 是**非因果 block
attention**（`docs/37` §5.2），它的 q/k 预处理对 context 与 block 两个来源**分别做**、且要在进入
attention kernel 前就写回全局（供 KV store / 后续复用）；而通用 paged attention 的 q/k 本来就已经在
SMEM 里、顺手原地 norm+rope 最省。**同一种数学，两种融合边界，取决于下游消费的是 SMEM 视图还是全局张量**。

---

## 5. 同步机制与线程数口径：一个跨代「百家争鸣」表

| 函数 | 同步原语 | barrier ID | 线程数来源 | 支持 partial RoPE |
|---|---|---|---|---|
| `ampere::rms_norm_impl`（独立） | `cp.async` wait + `__syncthreads` | — | 模板默认 128? 用全局 `NUM_THREADS` | 否 |
| `hopper::rms_norm_hopper_impl`（独立） | `cp.async` wait + `__syncthreads` | — | **模板默认 256** | 否 |
| `ampere::rms_norm`（SMEM） | cooperative groups + `__syncthreads` | — | 全局 `NUM_THREADS`=128 | 否 |
| `hopper::rms_norm_hopper`（SMEM） | `wg_sync`（`bar.sync`） | 9 | **显式模板参**（调用方传 128） | 是 |
| `blackwell::rms_norm_sm100`（SMEM） | cutlass `NamedBarrier` | 6 / 7 | 全局 `NUM_THREADS`=128 | 是 |
| `ampere::rotary_embedding` | cooperative groups | — | 全局 `NUM_THREADS` | 否 |
| `hopper::rotary_embedding_hopper` | `wg_sync`（`bar.sync`） | 9 | **模板默认 128** | 是 |
| `blackwell::rotary_embedding_sm100` | cutlass `NamedBarrier` | 6 | 全局 `NUM_THREADS` | 是 |
| `blackwell::dflash_norm_rope_sm100` | 无（一线程一 (tok,head)） | — | 全局 `NUM_THREADS` | 否 |

三个口径坑，都是本专题「已知坑」的同一 family：

1. **`NUM_THREADS` 的四种来源并存**：全局 `worker_config.h:18 = 128`、hopper-rmsnorm 模板默认
   **256**、hopper-rope 模板默认 128、hopper SMEM-rmsnorm 显式模板参（调用方传值）。这与
   `docs/38` §2.4 记的「共享原语层按 128 写、上层常驻 worker 是 256」同源——**读这些身体前先确认
   它到底用哪个线程数**。
2. **barrier ID 的口径张力**：本家族用 **6 / 7 / 9**；而 `docs/34` §3.5 转述源码注释称用户命名屏障
   ID 须落 **≥8**（误用保留区 1–7 会留脏状态）。两者**并存**——`docs/37` §8 已记此张力，本文只补
   「norm/rope 家族也在 6/7/9 上」这一具体证据，**不判对错**（`bugfix.md` 不在快照里）。
3. **partial RoPE 的三代不对称**：Ampere 全家族（norm + rope）都不支持 partial，Hopper/Blackwell 才有
   `ROTARY_DIM`。这解释了为什么 **GLM-4.6 这类 partial-RoPE 模型只能走 `target_cc>=90` 的路径**。

---

## 6. 生产布线总表（五层，逐条可核对）

| 形态 | `TaskType`（`runtime_header.h`） | `register`（`task_register.cc`） | `graph.cc` dispatch | 前端（`persistent_kernel.py`） | builder |
|---|---|---|---|---|---|
| ampere 独立 rmsnorm | `TASK_RMS_NORM`=119（`:109`） | `register_rmsnorm_task`（`:91`，emit `:118`） | `"rmsnorm"` → `:449-452` | `rmsnorm_layer`（`:865-881`，cc<90 分支） | 各家族 builder 的 `rmsnorm_layer` 调用 |
| hopper 独立 rmsnorm | `TASK_RMS_NORM_HOPPER`=154（`:117`） | `register_rmsnorm_hopper_task`（`:1551`，emit `:1580`） | `"rmsnorm_hopper"` → `:571-574` | 同上（cc≥90 分支） | 同上 |
| paged-attention 融合 norm+rope | （不单独登记，属 `TASK_ATTN_SM100`） | `register_paged_attention_sm100_task`（`task_register.cc:2326`） | `"paged_attention"` → `:496` | `paged_attention_layer`（`:1303` 起，`rotary_dim` 参数 `:1318`） | 各注意力模型 builder |
| DFlash 独立 norm+rope | `TASK_DFLASH_NORM_ROPE_SM100`=297（`:193`） | `register_dflash_norm_rope_sm100_task`（`:185`，emit `:211`） | `"dflash_norm_rope"` → `:462-466` | `dflash_norm_rope_layer`（`:1007-1028`） | `dflash/builder.py:261/318/320` |

- **device 编译入口**：`tasks/<arch>/task_header.cuh` 决定哪些身体进常驻 kernel 编译单元——
  `ampere:13-14`（`rmsnorm.cuh`+`rotary_embedding.cuh`）、`hopper:21`（`rmsnorm_hopper.cuh`）、
  `blackwell:17-18`（hopper 的 `rmsnorm_hopper.cuh`+`rotary_embedding_hopper.cuh`）与 `:31`
  （`dflash_norm_rope_sm100.cuh`）。`rms_norm_sm100`/`rotary_embedding_sm100` **不直接进 task_header**，
  而是经 `attention_sm100.cuh`（`blackwell/task_header.cuh` 里的 `:36` 左右）间接编入——所以
  `rotary_embedding_sm100` 虽然进了编译单元，却依然零调用者（§3.3）。
- **可执行规范**：`test_mode` 的 rmsnorm 用例见 `docs/09`（`test_rmsnorm_testmode.py`）；
  本文的 Hopper 实测数字即来自它。

---

## 7. 判据映射

- 本批 device 身体**只贡献 C3**（单 kernel 可表达、数值由单测/HF 参考校验）：它们是**被 worker 调用的
  `__device__` 函数**，不含自己的 `<<<>>>`、不做跨算子协调、无固定角色 worker/scheduler。C1/C2/C4/C5
  全由运行层（`docs/06`）给——与 `docs/11`/`docs/29`–`docs/37` 同口径。
- 本家族的**融合范围价值**恰是本文的主线：**形态 B/C 是「省一趟 SMEM↔HBM 往返」的典型**（norm+rope
  从不落全局、直接在 attention 的 SMEM 视图上原地做，对上 `docs/04` §17 损失三）；**形态 A 则被拆成
  独立 task**，因为它服务的是「下游要全局张量」的场景（DFlash q/k 预处理、通用 rmsnorm 供 GEMM 消费）。
  **同一种数学、两种融合边界**，是 megakernel「融合到哪一层」这一设计轴在薄算子上的最小样本。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建/未运行本家族其它身体**：SM100 在目标卡 H100/sm90 不可执行；Hopper 版可执行，
  但本文除复用 `docs/09` 的 test_mode RMSNorm 数字外**未新跑任何实验**（无新 `src/`/`harness/`）。
- **「哪个 builder 调哪个 layer」是静态 `grep` 结论**，不构成运行时判定，也**不代表上游 bug**；
  `rotary_embedding_sm100` 的「零调用者」是静态事实（全仓 grep），其「死代码」定性是推断。
- **`docs/37` §4.2 的勘误只针对 rope 函数选择一处**；`docs/37` 其余结论不在本文范围。
- **`WORKER_NUM_THREADS`/`NUM_THREADS` 的运行时值未取**：常驻 worker 在 Blackwell 是 256
  （`worker_config.h:27-28` 在 `MIRAGE_GRACE_*` 下），但本家族的共享函数多用全局 `NUM_THREADS`=128；
  本文只记静态口径，`docs/38` §2.4 已标注「是否安全取决于调用点覆盖、只记不判」。
- **命名屏障 ID 6/7/9 与 `docs/34` §3.5「须 ≥8」并存不判**（同 `docs/37` §8）。
- **未展开的相邻层**：`norm_linear_hopper.cuh`（RMSNorm+Linear 融合）已在 `docs/07` §3.3 走过，
  本文不重复；`silu_mul_*`、`per_token_group_quantize_fp8`、MLA 家族内的 rope 复用不在本文范围。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/44` 同快照）。
- **形态 A**：`tasks/ampere/rmsnorm.cuh`（145 行；`rms_norm_impl` `:19-143`）；`tasks/hopper/
  rmsnorm_hopper.cuh`（162 行；`rms_norm_hopper_impl` `:19-160`）；`tasks/blackwell/
  dflash_norm_rope_sm100.cuh`（90 行；`dflash_norm_rope_sm100` `:35-88`）。
- **形态 B**：`tasks/ampere/norm.cuh`（127 行；`rms_norm` `:21-126`）；`tasks/hopper/norm_hopper.cuh`
  （136 行；`rms_norm_hopper` `:31-135`）；`tasks/blackwell/norm_sm100.cuh`（142 行；
  `rms_norm_sm100` `:31-141`）。
- **形态 C**：`tasks/ampere/rotary_embedding.cuh`（82 行；`rotary_embedding` `:27-80`）；
  `tasks/hopper/rotary_embedding_hopper.cuh`（84 行；`rotary_embedding_hopper` `:32-82`）；
  `tasks/blackwell/rotary_embedding_sm100.cuh`（80 行；`rotary_embedding_sm100` `:27-78`，死代码）。
- **同步原语**：`tasks/hopper/utils.cuh:30-36`（`wg_sync`）；`tasks/common/worker_config.h:18`
  （`NUM_THREADS=128`）。
- **接线**：`runtime_header.h:109/117/193`；`src/kernel/task_register.cc:91/1551/185`；
  `src/kernel/graph.cc:449-452/462-466/571-574`；`python/mirage/mpk/persistent_kernel.py:865-881/
  1007-1028`；`python/mirage/mpk/models/dflash/builder.py:261/318/320`；
  `tasks/{ampere,hopper,blackwell}/task_header.cuh`。
- **交叉引用**：`docs/07` §3.1（`rms_norm_hopper_impl` 摘要）、`docs/37` §4/§5（SM100 norm/rope/attention
  摘要，本文修正 §4.2）、`docs/38` §2.4/§6（`NUM_THREADS` 口径坑）、`docs/09` §7–§8（Hopper RMSNorm
  真机数字）、`docs/04` §17（损失三）。
