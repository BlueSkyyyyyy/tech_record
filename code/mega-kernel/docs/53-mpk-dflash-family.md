# MPK DFlash task 家族走读：非因果 block attention、per-head norm+RoPE 与独立 paged KV store（P139-1）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/52` 同快照）。
> **纯源码走读（资料分析），本轮无新实验**——SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，
> 与 `docs/35`–`docs/52` 同。
>
> 本文补掉 `docs/35` §9 明留的「**注意力之外的 SM100 task 家族**（`dflash_*`/`inkling_*`/…）未逐行」
> 这条尾巴里的一支：把 DFlash 家族三个 device 身体（`dflash_attention_sm100` / `dflash_norm_rope_sm100`
> / `dflash_kv_store_sm100`）从 `docs/37` §5.2 的摘要推进到**逐行**，并核清 `TaskType`/`graph.cc`/
> `register`/前端/builder 五层接线。DFlash 是 **Kimi-K2.6 的投机解码草稿**，与 Eagle3（`docs/40`/
> `docs/43`/`docs/44`/`docs/46`/`docs/47`/`docs/48`）并列的第二条投机解码支线。

## 0. 三条 headline

**(A) DFlash 是「单次非因果 block attention」，与通用 paged attention（`docs/44`）是两种数学**：
`B` 个 block query token **非因果**地看 `[context（`ctx_len`） ++ block（`B`）]` 全体 key（`dflash_attention_sm100.cuh:24-61`
头注释），绝对位置定义为「context key `j` → 位置 `j`，block key/query `i` → 位置 `ctx_len+i`」（`:41-42`），
sliding window 按**绝对距离 `|q_pos − key_pos|`** 过滤（不是因果）。这决定了它**不复用** `attention_sm100`
的 online-softmax/因果掩码，而是自写一套。

**(B)（headline）一个 body 里塞了两条实现，由编译期 shape 分派**：`USE_MMA = (sizeof(T)==2) &&
(B*NUM_QO_PER_KV == 64) && (HEAD_DIM%64==0) && (HEAD_DIM<=128)`（`:195-198`）——不满足就走标量参考
`dflash_attention_sm100_ref`（一 warp 一个 `(query,q_head)`，`:62-175`），满足就走 tensor-core 路径
（`:177-566`）。mma 路径的关键取舍是**softmax 与 O 累加完全 warp-local、无跨 warp 归约**：per kv head 的
64 个 `(query,q_head)` 行组成 MMA 的 M 维，4 个 warp 各持一个 16 行 m-tile（`:53-60`/`:359-362`）；
K/V 用 `cp.async` **双缓冲 64-key tile** 流式（`:256-274`/`:372-414`），sliding window 直接**整 tile 跳过**
（`:246-251`）。对 Kimi-K2.6 的生产 shape 是 `B=8, GQA 8:1` → `8*8=64` 行（`:54`，测试 `:5-14`）。

**(C) 家族剩下两个 task 是「把写回/预处理各拆成一段独立身体」**：`dflash_norm_rope_sm100`（90 行，
一线程一个 `(token,head)` 走寄存器做 per-head RMSNorm+NeoX RoPE，`v` 不做，`:20-34`）与
`dflash_kv_store_sm100`（66 行，按 slot 散射写 draft **自己的** paged cache，`:20-38`）。这与
`docs/40`/`docs/41` 的「**把写回从 attention 里拆出去**是 task 可复用/可调度的先决条件」同向：
DFlash 是唯一把「draft 自己攒分页上下文」显式拆成 `norm_rope → kv_store` 两段的投机解码支线。
另记一条**接线落差**：DFlash 三个 device 身体 + `DFlashBuilder` 全部实现、`TaskType`/`graph.cc`/
前端 layer 四层全通，但 `DFlashBuilder` 在快照里**零实例化**（唯一消费者是单测直接调 `dflash_attention_layer`）。

---

## 1. 家族定位与五层生产接线

三个 `TaskType`（`include/mirage/persistent_kernel/runtime_header.h:191-195`）：

| TaskType | op 名 | 入/出 | device 身体 |
|---|---|---|---|
| `TASK_DFLASH_ATTENTION_SM100 = 296` | `dflash_attention` | 5 / 1 | `dflash_attention_sm100.cuh`（568 行） |
| `TASK_DFLASH_NORM_ROPE_SM100 = 297` | `dflash_norm_rope` | 4 / 1 | `dflash_norm_rope_sm100.cuh`（90 行） |
| `TASK_DFLASH_KV_STORE_SM100 = 298` | `dflash_kv_store` | 2 / 1 | `dflash_kv_store_sm100.cuh`（66 行） |

五层接线（均在快照可核）：

- **L1 登记 / L3 编入**：三个头都被 `tasks/blackwell/task_header.cuh:29-31` include（**SM100 专属**，
  `docs/35` §9 的「编入」闭包内）；`graph.cc:457-470` 把 op 名映射到 `register_*`。
- **发射器**：`src/kernel/task_register.cc` 的 `register_dflash_attention_sm100_task:126-183`、
  `register_dflash_norm_rope_sm100_task:185-220`、`register_dflash_kv_store_sm100_task:222-256`。
- **前端 layer**：`python/mirage/mpk/persistent_kernel.py` 的 `dflash_attention_layer:970`、
  `dflash_norm_rope_layer:1007`、`dflash_kv_store_layer:1030`。
- **builder**：`python/mirage/mpk/models/dflash/builder.py`（364 行，`DFlashBuilder`）。
- **单测**：`tests/runtime_python/blackwell/sm100_dflash_attention/test_dflash_attention_testmode.py`
  （101 行，四个 case 覆盖 mma + grid split + sliding window + 标量 ref fallback）。

---

## 2. `dflash_attention_sm100`：一条 kernel 的两条实现（568 行）

### 2.1 语义契约（头注释 `:24-61`）

```
inputs : q[B, NQ*D] / ctx_k[ctx_len, NKV*D] / ctx_v[ctx_len, NKV*D]
         blk_k[B, NKV*D] / blk_v[B, NKV*D]        (已投影/归一化/RoPE)
output : out[B, NQ*D]
位置   : context key j -> j ; block key/query i -> ctx_len + i
sliding: 每 query 只看 |q_pos - key_pos| < sliding_window 的 key（非因果）
```
`v` 是**原始值**（不做 norm/RoPE，`:36`/`:38`，与 `dflash_norm_rope` 的「`v` 跳过」呼应）。头注释另说明
**head 数是「per-task」的**：当层被按 kv head grid-split（`grid.x = G`）时，运行期把每个输入指针偏到本
task 的 head 列切片，寄存器把 `NUM_Q_HEADS/G`、`NUM_KV_HEADS/G` 传进来（`:44-48`）——这就是 §1 前端
`dflash_attention_layer` 的 `G>1` 分支（`persistent_kernel.py:988-1003`）。

### 2.2 标量参考 `dflash_attention_sm100_ref`（`:62-175`）

- **执行体**：`if (threadIdx.x >= NUM_THREADS) return;`（`:88-90`）——**只让前 `NUM_THREADS`（=128，
  `tasks/common/worker_config.h:18`）个线程干活**，因为运行期 launch 的线程更多（builder `bd=(256,1,1)`，
  `builder.py:127`）。这是全仓反复出现的「runtime launches more」守卫（同 `docs/44`/`docs/45`）。
- **并行划分**：一 warp 一个 `(query, q_head)` 对（`pair = warp; pair < B*NUM_Q_HEADS; pair += WARPS`，
  `:104-106`），`WARPS = NUM_THREADS/32 = 4`；每 lane 持 `DPT = HEAD_DIM/32` 维（`:80`）。
- **QK 内积 + warp 归约**：每 lane 算 `HEAD_DIM/32` 维点积后 `__shfl_xor_sync` 蝶形归约（`:146-154`）。
- **在线 softmax**（`:119-166`）：`m_i/l_i/acc[DPT]`，`m_new=fmax`、`corr=exp(m_i−m_new)`、`p=exp(score−m_new)`；
  key 位置在 `[0,ctx_len)` 取 `ctx_*`、否则取 `blk_*`（`:135-145`）；sliding window 在标量路径按
  `|q_pos−j| >= window` 直接 `continue`（`:128-134`）。
- **epilogue** `1/l_i` 归一化写回（`:168-173`）。

### 2.3 tensor-core 路径（`:177-566`）

**shape 门 + 常量**（`:193-227`）：`NUM_QO_PER_KV = NUM_Q_HEADS/NUM_KV_HEADS`；`USE_MMA` 判据见 §0(B)；
`M_ROWS = B*NUM_QO_PER_KV` **必须是 64**（`static_assert(M_ROWS==64):226`）；`MMA_N_TILES = 64/16 = 4`
（每 tile 16 个 key 列）、`MMA_K_TILES = HEAD_DIM/16`（QK 的 k 维 / PV 的 n 维）。

**同步原语**（`:229-236`）：一条 `cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, 6)`
（**barrier ID 6**，与 `docs/37` §4.1/`docs/45` 的 norm/attention 家族同号，`docs/46` §6 记过这族
「barrier ID 百家争鸣」）；同样 `threadIdx.x >= NUM_THREADS` 的消费者守卫。

**窗口整 tile 跳过**（`:242-253`）：最小 query 位置是 `ctx_len`，故没有 query 能看到 `ctx_len−window+1`
以下的 key；`kv_start = ((ctx_len−window+1)/64)*64`，`total_kv = T_kv − kv_start`、`num_iters = ceil/64`。

**SMEM 布局**（`:255-295`）：一个 8 元素 `zero_buffer`（给越界 K/V 行喂 0）、`S_Q`（`M_ROWS*HEAD_DIM`）、
`S_K`/`S_K_buffer`/`S_V`/`S_V_buffer`（各 `64*HEAD_DIM`，**K 与 V 各自双缓冲**）、`S_O = S_Q`；
`static_assert(S_TOTAL <= MAX_DYNAMIC_SHARED_MEMORY_SIZE):274`。布局类型 `smem_row<T,3,3,3,...>`
（`QOSmem`/`KVSmem`，`:289-290`）——`B=M=S=3` 即 **128 B swizzle**，是 `ldsm` 读侧的 ABI
（`docs/51` §B 已逐行解释）。DTensor 侧 `dmem_row[_const]` 用**完整行宽** `Q_STRIDE/KV_STRIDE/O_STRIDE`
做 stride，指针是列切片（`:297-301`）。

**主循环 `for g in kv heads`**（`:305-564`）：每 kv head 一遍，开头 `wg_barrier.arrive_and_wait()` 保护
上一轮 `q_smem/o_smem` 在被重载前读完（`:306-307`）；指针按 `g` 偏移（`:309-315`）。K 与 V 的
`set_ptr` 复位到偶数槽（`:317-320`）。

- **载 Q 并转置布局**（`:325-337`）：gmem 行是 `[B × (NUM_QO_PER_KV*HEAD_DIM)]`，smem 行是
  `(query,q_head)` 对；`load_smem` 时做 `dst_row = src_row*NUM_QO_PER_KV + src_col/HEAD_DIM`、
  `dst_col = src_col%HEAD_DIM` 的重排（即把一行的 `NUM_QO_PER_KV` 个头拆成 `NUM_QO_PER_KV` 个 smem 行）。
- **prologue 载 KV tile 0**（`:339-357`）：`cp.async` 逐 16 B 块，越界（`j >= ctx_len`）取 `blk_*`；
  `cp_async_fence()` 发射。
- **per-thread flash 状态**（`:359-370`）：`m_local[2]`/`d[2]`（每 thread 持 2 个 row-half）、
  `o[MMA_K_TILES][8]`（本 warp 的 16×HEAD_DIM 累加器 fragment），Q fragment `q_frags[MMA_K_TILES][4]`。
- **iter 循环**（`:372-536`）：先 `cp.async` 预取下一 tile 到「本 iter 的 compute buffer」（`:376-400`，
  `cp_async_wait<1>` 保双缓冲），再**旋转 buffer 指针**（`:403-413`）、`wg_barrier`（`:414`）；Q 的
  fragment 只在 `iter==0` 时 `ldsm` 一次（Q 跨 iter 不变，`:416-424`）。
- **X = Q·Kᵀ**（`:426-446`）：4 个 n-tile × `MMA_K_TILES` 个 k，`mma_m16n16k16_bf16bf16bf32`（ampere
  手写 mma，`docs/39` 的 `"+f"` 累加器原语）；越界 key 列用 `zero_buffer` 顶替（`:440-441`）。
- **掩码 + running row max**（`:448-475`）：`is_valid = col < curr_iter_len`，sliding window 时再按
  `dpos = |ctx_len+qi − j_abs| < window` 收紧（`:459-465`），非法记 `-inf`；因**一行由 4 个 thread 各持
  2 个元素**，用 `shfl_xor(0x1/0x2)` 对 `m_local[0/1]` 做 warp 内归约（`:471-475`）。
- **rescale + exp + d 累加**（`:477-500`）：`rescale=exp((m_prev−m_local)*scale)`，`x_frag = exp(x*scale −
  m*scale)`（`-inf` 记 0），`d_partial` 同样 `shfl_xor` 归约。
- **rescale O + O += P·V**（`:502-530`）：先按 `rescale` 缩放 `o`；把 QK 的 f32 累加器 `convert_f32_to_bf16_uint32`
  转成 MMA 的 A 操作数（**QK 累加器碎片复用为 PV 的 A**，`:511-517`），64-key 维切成 4 个 k chunk
  （每个 score n-tile 一个）；`ldsm_t` 载 V（越界行用 `zero_buffer`，`:525-526`）再 `mma`。
- **epilogue**（`:538-563`）：每 warp **只写自己的 16 行**（**无跨 warp 归约**，`:538-539`），
  `o_smem.at(row,col) = o/d`；`wg_barrier` 后按行重排写回 gmem（smem `(qi,h)` 行 → gmem `qi` 行的
  `h*HEAD_DIM+` 列，`:555-563`）。若一行无有效 key，`o==0 && d==1` → 存 0（与 reference 一致，`:539-540`）。

---

## 3. 家族另两个 task

### 3.1 `dflash_norm_rope_sm100.cuh`（90 行）

与 `docs/45` §4.3 的摘要一致，逐行确认：**一线程一个 `(token, head)`**（`:53-54`，`p = tok*NUM_HEADS+head`），
全程寄存器、**无跨 lane**（`:32-34` 注释）。三步——RMSNorm over `HEAD_DIM`（fp32 平方和 + `rsqrtf(ss/D+eps)`，
`:62-69`）→ 归一化缓冲 `nv[HEAD_DIM]`（`:72-76`）→ **NeoX RoPE**（前/后半配对：`i<HALF` 用 `-nv[i+HALF]`、
否则用 `nv[i-HALF]`，`:78-86`）。`eps` 由发射器硬编码 `1e-5f`（`task_register.cc:218`）。输入输出都是
全局 `[N, NUM_HEADS, HEAD_DIM]`，`v` 不做（`:25`）。

### 3.2 `dflash_kv_store_sm100.cuh`（66 行）

**一个 task 处理全部 `num_tokens`，按 slot 散射**（`blockIdx` 无关，`:36-37`）：线程 `p` 拥有元素
`(tok, kv_head, dim)`（`rem = kv_head*D + dim`，`:55-57`）。关键简化解：draft 的页缓存是**顺序分配、
物理连续**的，故 slot `s` 的**扁平下标**就是 `s*KVD + rem`（`page=s/page_size`、`off=s%page_size`，
合并成 `(page*page_size+off)*KVD+rem`，`:58-61`）——普通 store 即覆盖写。注释点明用途：
**L4 materialization 把 committed context K/V（target 派生）写进 draft 自己的页缓存，覆盖上一 decode
step 在那些 slot 写的临时 block K/V**（`:23-26`）。

---

## 4. builder 与前端：DFlash 的一步解码（`builder.py`）

`DFlashBuilder`（`dflash/builder.py:70`）docstring（`:1-54`）描述每步解码的 L2→L6（全部在**一个常驻
kernel** 内）：L2 `ctx = hidden_norm(fc(target_hidden))`；L4 每层 `k=v` 投影 → `dflash_norm_rope` →
`dflash_kv_store` 写进该层页缓存；L5 draft block `[t0, MASK*(B-1)]` 跑一次非因果前向（每层读「页缓存
context ++ 本 block K/V」，层 0–L-2 滑窗 2048、最后一层 full）；L6 `final_hidden → target lm_head →
argmax`。

`materialize_context_kv`（`:222-268`）：L2 的 `linear`+`rmsnorm` 之后，每层 `linear(k)` → `dflash_norm_rope`
→ `dflash_kv_store`，`v` 则 `linear(v)` → `dflash_kv_store`（**`v` 不过 norm_rope**，`:265-268`），
三者的 `grid_dim` 差异很说明粒度——线性走 `_gpl(...)`（96/64 个 SM 块），而 `norm_rope`/`kv_store`
走 `grid_dim=(1,1,1)`（**单 CTA、内部遍历全部 token**，与 `docs/37` §7 的 `blockIdx.x==worker id`
适配范式一致）。

`build_draft_forward`（`:270-349`）：每层 `rmsnorm → linear(q) → norm_rope → linear(k) → norm_rope →
linear(v) → dflash_attention → linear(o) → elementwise_add → rmsnorm → linear(gateup) → silu_mul →
linear(down) → elementwise_add`。**attention 的 grid 是 `(num_kv_heads,1,1)`**（`:322-326`）——注释
明确「split attention across kv heads: `num_kv_heads` tasks/layer 而不是 1（每个 task = mma flash
路径上的 8 q heads / 1 kv head）」，即生产上 `G = num_kv_heads = 8`，四个单测 case 里的 `kimi`/`window`
也是 `G=8`。

**两条静态落差（只记不判）**：

1. `builder.py:51-53` 的 docstring 引参考单测 `tests/runtime_python/blackwell/sm100_dflash/`
   `{test_dflash_full_step_testmode.py, test_dflash_paged_xiter.py}`——快照里**该目录不存在**，只有
   `sm100_dflash_attention/test_dflash_attention_testmode.py`（覆盖的是 device 身体，不是端到端 step）。
2. `DFlashBuilder` 在 `__init__.py:1` 导出，但**全仓零实例化**（`grep DFlashBuilder` 只命中自身
   定义/导出/docstring）；即「device 家族 + builder 已实现，但跑起来的生产路径（目标 K2.6 主干进 MPK）
   尚未接上」——与 `docs/26` §5「draft builder 不注册、由目标 builder/demo 追加」的设想一致，但目前
   连 append 的调用者都还没有。这是 `docs/49`/`docs/50` 的「有实现 ≠ 已接线」在**模型 builder 层**
   的又一实例（三个 device task 本身是接通的，缺的是**上层生产者**）。

---

## 5. 与已有文档的关系（避免重复）

- `docs/37` §5.2 给了 DFlash 的**摘要**（非因果、两条实现、cp.async 双缓冲、K/V store），本文把它推到
  **逐行**并补了：SMEM 双缓冲的**两两 set_ptr 旋转**、QK 累加器**复用为 PV 的 A 操作数**、
  **一行由 4 thread 持有 + `shfl_xor` 归约**的 fragment 布局、`dflash_kv_store` 的「页连续 ⇒ 扁平
  slot」化简、以及 §4 的 builder 粒度与两处静态落差。
- `docs/45` §4.3 覆盖 `dflash_norm_rope`（本文不重复，仅交叉引用）。
- `docs/51` 覆盖 `smem_row<T,3,3,3,...>` 的 128 B swizzle 语义（本文只引用）。
- `docs/39` 覆盖 `tasks/speculative_decoding/*`（MTP/Eagle3）；DFlash 是**另一支**，device 目录也不同
  （`tasks/blackwell/dflash_*`），本文不重复那条链，只指出二者同为投机解码草稿。

---

## 6. 判据映射

| 判据 | DFlash 家族是否贡献 |
|---|---|
| C1 单次 launch | 间接（三个 task 只是同一常驻图里的节点，launch 由 `PersistentKernel` 统一发） |
| C2 常驻 | 间接（同 C1） |
| C3 多算子范围 | **是**——本文给出 DFlash 段「哪些 task、融合边界在哪、粒度多大」的清单（§1 表 + §4 builder 链） |
| C4 设备内协调 | 间接（跨 task/跨 step 依赖由 MPK event 承担，`docs/06`/`docs/09`） |
| C5 SM 级调度 | 否（device body 层不涉 worker/scheduler 划分） |

结论：**本文贡献 C3 的「DFlash 段的范围与融合边界」**，与 `docs/37`（摘要）、`docs/40`（退役
paged attention）、`docs/44`（通用 paged attention）互补。特别地，DFlash 把「draft 自己攒分页上下文」
拆成 `norm_rope` + `kv_store` 两个**独立薄 task**，是「**融合边界不是越融越好**」在投机解码侧的
又一实证（对照 `docs/41` 退役实现里把「写回」融进 attention 的做法被淘汰）。

---

## 7. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100 在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  本文所有行号为快照 `f9eb70c` 静态事实，不构成对上游的 bug 判定（唯一的例外是 §4 两条**静态落差**，
  也只记不判）。
- **数值等价未验证**：`dflash_attention_sm100_ref` 与 mma 路径是否逐位一致，只由单测
  `test_dflash_attention_testmode.py` 的 `atol=rtol=2e-2` 口径**声明**，本文未跑；两条实现的分派形状
  覆盖（如 `B=4,GQA 4:1` 走 ref、`B=8,GQA 8:1` 走 mma）只在单测配置里出现。
- **Kimi-K2.6 的模型侧未核**：`builder.py` 的 `hidden_size=7168`/`num_layers=6`/`num_q_heads=64`/
  `num_kv_heads=8` 等取自 `draft_config`，本文未加载真实 ckpt，未逐层核权重名与 HF 的对应。
- **`M_ROWS==64` 之外未探**：mma 路径对 `B*NUM_QO_PER_KV != 64` 的 shape 一律回退 ref，本文未穷举
  「哪些生产 shape 会回退」。
- **未与 DeepGEMM/其它 SM100 路径对跑**：与 `docs/35`–`docs/52` 同口径，SM100 路径本机不具备。

---

## 8. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/52` 同快照）。
- **device 身体**：`include/mirage/persistent_kernel/tasks/blackwell/dflash_attention_sm100.cuh`
  （568 行；头契约 `:24-61`；ref 模板 `:62`、`threadIdx` 守卫 `:88-90`、并行划分 `:104-106`、
  QK 蝶形 `:146-154`、在线 softmax `:119-166`、epilogue `:168-173`；mma `USE_MMA:195-198`、
  dispatch `:200-217`、常量/`static_assert:221-227`、NamedBarrier 6 `:229-236`、窗口跳过 `:242-253`、
  SMEM 布局 `:255-295`、kv-head 主循环 `:305-564`、Q 重排载入 `:325-337`、prologue `:339-357`、
  flash 状态 `:359-370`、iter/预取/旋转 `:372-414`、QK `:426-446`、掩码+max `:448-475`、
  exp/d `:477-500`、rescale+PV `:502-530`、epilogue `:538-563`）；
  `dflash_norm_rope_sm100.cuh`（90 行；契约 `:20-34`、norm `:62-69`、nv `:72-76`、RoPE `:78-86`）；
  `dflash_kv_store_sm100.cuh`（66 行；契约 `:20-38`、散射 `:55-63`）。
- **共享原语**：`tasks/common/worker_config.h:18`（`NUM_THREADS=128`）；`tasks/ampere/mma.cuh`
  （`mma_m16n16k16_bf16bf16bf32`）；`tasks/ampere/smem_layout.cuh` 的 `smem_row`（`docs/51`）。
- **登记 / 编入 / 布线**：`runtime_header.h:191-195`；`tasks/blackwell/task_header.cuh:29-31`；
  `src/kernel/graph.cc:457-470`；`src/kernel/task_register.cc:126-256`。
- **前端**：`python/mirage/mpk/persistent_kernel.py:970-1048`（三 layer，attention 的 grid-split
  断言 `:988-1003`）。
- **builder**：`python/mirage/mpk/models/dflash/builder.py`（docstring `:1-54`；`__init__:71-147`；
  YaRN cos/sin `:150-160`；`_prepare_weights:161-220`；`materialize_context_kv:222-268`；
  `build_draft_forward:270-349`；attention grid `:322-326`）。
- **单测**：`tests/runtime_python/blackwell/sm100_dflash_attention/test_dflash_attention_testmode.py`
  （四 case `:46-51`；`pk.dflash_attention_layer:67`；参考校验 `:82-92`）、
  `.../pytorch_reference.py:14`（`dflash_attention_ref`）。
- **本专题交叉引用**：`docs/37` §5.2（摘要）、`docs/40`/`docs/41`（退役 paged attention 与「拆写回」）、
  `docs/44`（通用 paged attention）、`docs/45` §4.3（`dflash_norm_rope`）、`docs/39`（投机解码 device 层）、
  `docs/26` §5（family builder 视角）、`docs/24`（`TaskType` 296/297/298）、`docs/51` §B（swizzle）。
