# MPK 通用 paged attention device 身体走读：`attention_sm100.cuh`（`multitoken_paged_attention_sm100_task_impl`）

> 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与本专题 `docs/06`–`docs/43` 同一快照）。
> 主文件 `include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh`（**恰好 750 行**，
> 唯一 device 函数 `multitoken_paged_attention_sm100_task_impl`）。**纯源码走读（资料分析），
> 本轮无新实验**；SM100 专属，在目标卡 H100/sm90 不可执行。
>
> 本篇补掉三条明留的尾巴：`docs/37` §5.1（只给了 750 行摘要）、`docs/39` §6.3（`paged_attention_sm100`
> `MAX_TOKENS=3` slot-0 分歧只按注释转述）、`docs/43` §9（「`paged_attention_sm100` 内部未逐行，
> 只引其 `MAX_TOKENS`/`MMA_ITERS_M`/`"+f"` 三处注释」）。

## 0. 三条 headline

- **(A) 它是 MPK 里唯一的「通用 paged GQA attention」device 身体，用 Ampere 手写 `m16n16k16` `mma.sync`
  （不是 wgmma/UMMA）**：一个 task = 一个 CTA × 一个 (request, KV head)，在**在线 softmax**
  （`m`/`d`/`o` 三分量全在寄存器）里流式扫完整条 KV cache，`m16n16k16` 的 QK^T 走 `1x4x1`
  warp 布局（warp 切 key 位）、PV 走 `1x1x4`（warp 切 p 位），**跨 4 warp 的 m/d/o 归约按 MMA m-tile
  分块**（省 smem，issue #702）；它同时把 **partial RoPE / sliding window / attention sink / QK-norm**
  全编进同一个 task（`:34-82` 的 14 个模板参数 + 15 个运行期入参）。判据**只贡献 C1+C3**。
- **(B) 它是 `00` §3「SM100 块里也有走 `mma.sync` 的通用 attention」的样板，且是一连串手写 MMA
  坑的活教材**：QK^T 的在线 softmax 里**因果/窗口掩码不是一行三元式**，而是 `:556-559` 的
  `is_valid = (row 有效) && (key_pos <= query_pos) && (窗口)`——`docs/39`/`docs/43` 转述的
  「causal mask 在 `:519`」实为**行号漂移**（`:519` 是 `k` 循环头），真正公式在 `:556-559`；
  而 `:505-516` 的注释把 `MAX_TOKENS=3` slot-0 分歧的根因钉在一处 **`"+f"` in-place 累加器** bug
  （`docs/38` §6 已从 `ampere/mma.cuh` 侧走读，本篇补上 attention 侧的现场）。
- **(C) 它的 SMEM 预算是一段「为把常驻 SMEM 压进 Blackwell 上限而反复重构」的重构史**，全写在
  源码注释里：**O 复用 Q 段**（`:187-196`，issue #740）、**跨 warp 归约的 `m`/`d`/`o` buffer 只存
  一个 MMA m-tile**（`:209-228`，issue #702）——两者都是「让 `max_num_batched_tokens` 能过
  20/25」的让步，是 C2（常驻）对 device task 的直接约束，与 `docs/33` §5「TMEM free 不释放锁」
  同族。

---

## 1. 文件定位、模板参数与接口

### 1.1 定位：被谁 include、登记成什么

- **device 身体入口**：`multitoken_paged_attention_sm100_task_impl`（`:62`），全文件唯一的
  `__device__` 函数。
- **生产编译单元**：`tasks/blackwell/task_header.cuh:28` 直接 `#include "attention_sm100.cuh"`；
  同一头里 `:29` `dflash_attention_sm100.cuh`、`:34` `inkling_attention_sm100.cuh`——**同一批
  非 MLA attention 家族**。另注 **`tasks/hopper/task_header.cuh:26` 也 include 了这个 blackwell
  文件**（Hopper 编译单元里也能实例化它，虽然生产走 `paged_attention_hopper`）。
- **注册**：`register_paged_attention_sm100_task`（`src/kernel/task_register.cc:2326-2438`）
  把模板实参 emit 成 `kernel::multitoken_paged_attention_sm100_task_impl<bfloat16, $, $, ...>`
  （`:2402-2418`），`TaskType = TASK_ATTN_SM100 = 257`（`docs/24` §2）。
- **前端布线**：`graph.cc:643-649`（`name == "paged_attention_sm100"` → `register_...` →
  `task_config[op] = (num_attn_inputs, 1, TASK_ATTN_SM100, variant_id)`）；Python 侧
  `PersistentKernel.paged_attention_layer`（`persistent_kernel.py:1303`）按 `target_cc` 三选一
  （`:1427-1432`），`params` 打包见 `:1354-1402`。

### 1.2 模板参数（`:34-61`）：一张被「功能开关」撑大的签名

| # | 参数 | 说明（源码注释） |
|---|---|---|
| T | 数据类型 | 生产实例化为 `bfloat16`（`task_register.cc:2402`） |
| NUM_QO_HEADS | Q head 数 | 每 task 只跑**一个 KV head**（见下） |
| NUM_KV_HEADS | KV head 数 | **恒传 1**（`task_register.cc:2405-2406` 传 `num_q_heads/num_kv_heads, 1`） |
| KV_CACHE_STRIDE / QKV_STRIDE / O_STRIDE | 三个张量行跨度 | |
| HEAD_DIM | head 维 | |
| MAX_SEQ_LEN | 最大序列 | |
| PAGE_SIZE | 页大小 | |
| Q_LEN_OVERRIDE | 查询行数覆盖 | Eagle3 K>1 链用（`:117-119`） |
| TAIL_OFFSET | 尾部偏移 | 同上（`:128-130`） |
| MAX_TOKENS | **每 CTA 缓冲的 query 行数** | 默认 8；`:45-52` 大段注释解释「≥mbt、又小到 per-row `S_Q`/`S_O` 放得下 SMEM」 |
| ROTARY_DIM | 部分 RoPE 维 | 默认 = `HEAD_DIM`；GLM-4.6 只转 128 维里的 64 维（`:53-56`） |
| WINDOW_SIZE | 滑窗宽度 | 0 = 全因果（`:57-59`） |
| PAGE_STRIDE_ROWS | 页间行距 | 0 = packed（`:60`） |

模板参数把「模型/硬件开关」（partial RoPE、滑窗、sink、页布局）**编译期固化**——这与
`docs/04`「多载体 kernel 的载体选择要放编译期 template」同一条经验：不同模型家族得到**不同的
`variant_id`**（`task_register.cc:2437` 的 `register_task_variant`），同一 `TaskType` 下多份特化。

### 1.3 运行期入参（`:62-82`）：packed QKV + 四条 indptr + 权表

- `qkv_ptr`：packed QKV，行 = token，列 = `head_dim*(num_qo+2*num_kv)`（`:93-95`）；
- `paged_k_cache_ptr`/`paged_v_cache_ptr`：NHD 页缓存 `[max_num_pages, page_size, num_kv_heads,
  head_dim]`（`:96-98`）；
- `qo_indptr_buffer` / `paged_kv_indptr_buffer` / `paged_kv_indices_buffer` /
  `paged_kv_last_page_len_buffer`：四条请求级元数据（`:111-130`）；
- `request_id`（`int16_t`）：**批元素号，不是 `blockIdx`**（`:71`，对上「已知坑」里
  「常驻 kernel 里 batch/request 号只能取 `task_metadata.request_id`」）；
- `qk_norm`/`rope`（bool），`q_norm_weight_ptr`/`k_norm_weight_ptr`/`cos_ptr`/`sin_ptr`，`q_eps`/`k_eps`；
- `sink_ptr`（`:80-82`）：**attention sink**，每 Q head 一个可学习 logit，只进 softmax 分母不进 value
  （`:724-729`）。

登记侧（`task_register.cc:2354-2355`）**输入数 `num_inputs = has_sink ? 8 : 7`**，输出 1——
这正是 `docs/24` 表里 `TASK_ATTN_SM100` 记的「(7 或 8, 1)」的真身（8 = 多一个 sink 输入）。

---

## 2. SMEM 预算：一段「把常驻 SMEM 压进上限」的重构史（`:168-231`）

一个 CTA 的 `extern __shared__ char smem[]`（`:233`）被手动切成若干区段，**顺序与大小都是
编译期常量**：

| 区段 | 偏移/大小 | 用途 |
|---|---|---|
| `zero_buf` | `S_ZERO_OFFSET=0`，`sizeof(T)*8` | **零块**，给边界 MMA 片段当占位 |
| `s_q` | `S_Q_OFFSET`，`sizeof(T)*MAX_TOKENS*NUM_QO_PER_KV*HEAD_DIM` | Q（也是 O 的别名段） |
| `s_k` / `s_v` | 各 `sizeof(T)*KV_TILE_SIZE*HEAD_DIM` | 当前 KV tile |
| `s_k_buffer` / `s_v_buffer` | 同 `s_k` | **双缓冲**的另一半 |
| `s_o` | **`S_O_OFFSET = S_Q_OFFSET`** | **O 复用 Q 段** |
| `s_q_norm_sum`/`s_k_norm_sum` | `sizeof(float)*4` | QK-norm 的 4-warp 归约 |
| `s_m_buffer`/`s_d_buffer` | `sizeof(float)*NUM_THREADS*2` | 跨 warp 的 m/d 归约（**只存一个 tile**） |
| `s_o_buffer` | `sizeof(float)*NUM_THREADS*64` | 跨 warp 的 o 归约（**只存一个 tile**） |

三处「用注释写死理由」的让步：

1. **Q 与 O 复用同一段**（`:187-196`）：注释论证 Q 在 KV 循环里最后一次读是 QK^T 的 MMA，
   之后有**三道 block 级屏障**把它和第一次写 `o_smem` 隔开，而两个视图（`QOSmem` vs `QOSmem`）
   覆盖同一 extent，故别名精确。**直接收益**：从预算里砍掉 `MAX_TOKENS*NUM_QO_PER_KV*HEAD_DIM`
   个元素，才让 `max_num_batched_tokens` 在 GQA 8:1 下**过 25**（issue #740）。
2. **`m`/`d` buffer 只存一个 MMA m-tile**（`:209-219` 注释）：跨 warp 归约「发布 tile m 的
   每线程值 → 消费 → 复用给 m+1」，若一次为全部 `MMA_ITERS_M` 个 tile 留空间，总 smem 会随
   `MAX_TOKENS` 涨，把 `max_num_batched_tokens` 卡在 20（issue #740）。
3. **`o` buffer 只存一个 tile 的累加器片段**（`:221-228` 注释）：`NUM_THREADS*64` float，
   否则 GQA ≥ 8:1（Qwen3-32B 64 Q / 8 KV）放不下 Blackwell smem 预算（issue #702）。

最后 `static_assert(S_TOTAL_OFFSET <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)`（`:230-231`）把预算
写死成**编译期断言**——`docs/04` 的「SMEM 是占用率第四个上限」在这个 task 里是硬约束。

另两处布局断言：`static_assert(HEAD_DIM % CP_CHUNK_SIZE == 0)`（`:264`，QKV 载入无尾巴须
16 B 整除）与 `static_assert(PAGE_SIZE % KV_TILE_SIZE == 0)`（`:268`，一个 KV 迭代只碰一页）。

---

## 3. 主流程：prologue + KV 循环

### 3.1 请求级元数据与提前返回（`:108-141`）

```
first_token_pos = qo_indptr[request_id]; last_token_pos = qo_indptr[request_id+1];
if (first_token_pos == last_token_pos) return;      // 空请求直接退出（:114-116）
num_tokens   = Q_LEN_OVERRIDE>0 ? Q_LEN_OVERRIDE : last-first;
num_pages    = paged_kv_indptr[rid+1] - paged_kv_indptr[rid];
seq_len      = (num_pages-1)*PAGE_SIZE + last_page_len[rid] - TAIL_OFFSET;
first_kv_iter= WINDOW_SIZE>0 ? max(seq_len-num_tokens-WINDOW_SIZE+1,0)/KV_TILE_SIZE : 0;
```

- **提前返回是均匀的**（`request_id` 对整个 CTA 一致），所以它发生在 `:142` 的
  `wg_barrier.arrive_and_wait()` **之前**不会引入发散。
- **滑窗跳过**（`:135-138`）：最早的那个 query 都看不到的 KV tile 直接不访问，`first_kv_iter`
  不是从 0 起——这也是 `:370` 缓冲翻转用 `(iter - first_kv_iter)` 而非 `iter` 的奇偶的原因
  （注释 `:368-369` 明说「parity counts from the first tile visited」）。
- `d_q`/`d_k`/`d_v` 的指针算术（`:144-147`）**每 task 只指一个 KV head**：
  `d_k = d_q + NUM_QO_PER_KV*HEAD_DIM; d_v = d_k + HEAD_DIM`。

### 3.2 屏障：`NamedBarrier(128, id=6/7)`（`:83-89`）

```
CONSUMER_WARPGROUP_SYNC_BARRIER_ID = 6;
ROTARY_SYNC_BARRIER_ID             = 7;
cutlass::arch::NamedBarrier wg_barrier(NUM_THREADS, 6);
if (threadIdx.x < NUM_THREADS) { ... }   // :90，NUM_THREADS=128
```

- `NUM_THREADS = 128`（`tasks/common/worker_config.h:18`，`docs/38` §2.4 已记它与 MPK 的 256 线程
  worker 的口径落差）。所以 `:90` 的 `threadIdx.x < 128` 让**后半 CTA（256 线程 worker 的后 128）
  不参与**；`wg_barrier` 是 `NamedBarrier(128, 6)`，只会合前 128——与 `docs/34` §3.5 / `docs/38`
  的「半 CTA 只同步活跃半 CTA 的命名屏障」一致。
- **但这里用 id 6/7**，而 `docs/34` §3.5 记 CUTLASS 用户区要求 **≥8**、`docs/37` §8 已标「ID 6/7
  与 `docs/34` 口径**并存不判**」。本篇只如实记录：本文件用 6/7，**不判对错**。

### 3.3 prologue：Q 载入 + 第一块 KV（`:270-307`）

- **Q 载入**（`:270-280`）：`cp.async`（`load_smem`，`copy_sm80.cuh:43-54`）把 packed Q 从
  `(token, qhead*HEAD_DIM)` 重排进 `q_smem(token*NUM_QO_PER_KV+qhead, :)`——即 smem 里 Q 是
  **按 (token, qhead) 展平成行**（`:274-278` 的 `dst_row = src_row*NUM_QO_PER_KV + src_col/HEAD_DIM`）。
- **第一块 KV**（`:282-305`）：先取 `page_idx_0`；对每个 16 B chunk，按
  `dst_row + cp_finished_seq_len < seq_len - num_tokens` 分流——**老 token 从页缓存读**
  （`page_idx_0*PAGE_STRIDE + page_offset`），**新 token 从 packed QKV 的 K/V 槽读**
  （`:299-303`）。然后 `cp_async_fence()`（`:306`）提交。
- 这一步的含义：**KV 写回页缓存与从页缓存读是同一 task 的两端**（写见 §3.5），与
  `docs/40` 的退役 `paged_attention.cuh` 把读+写融在一起的旧解法同构，只是这里写回被拆成独立代码段。

### 3.4 在线 softmax 状态初始化（`:309-328`）

```
m_local[MMA_ITERS_M][2]  = -inf;            // 每 m-tile 两个部分最大（行两半）
d[MMA_ITERS_M][2]        = 1.f;
o[MMA_ITERS_M][HEAD_DIM/16][8] 清零;
```

`MMA_ITERS_M = (MAX_TOKENS*NUM_QO_PER_KV + 15)/16`（`:103`）——**tile 数由 `MAX_TOKENS` 决定**。
注意 `inf = 5e4`（`worker_config.h:20`）而非真 `FLT_MAX`，`-inf` 即 `-5e4`。

### 3.5 KV 循环体（`:330-643`）

循环从 `first_kv_iter` 到 `num_iters`，每轮处理一个 `KV_TILE_SIZE=64` 的 KV tile：

1. **预取下一 tile**（`:331-366`）：`next_iter_len = min(seq_len - cp_finished_seq_len,
   KV_TILE_SIZE)`；非零时载入 `s_k`/`s_v` 并 `cp_async_fence(); cp_async_wait<1>();`（保留 1 组
   在飞，即经典双缓冲）。注释 `:336-338` 记一个**修过的坑**：预取上界必须用
   `next_iter_len`（正在载入的 tile）而非 `curr_iter_len`（正在消费的），否则首轮
   `next_iter_len < curr_iter_len` 会 OOB 进 QKV（issue #698/#692）。
2. **缓冲翻转**（`:368-380`）：按 `(iter-first_kv_iter)&1` 用 `set_ptr` 交换 `k_smem`/`k_buffer_smem`。
3. **`wg_barrier`**（`:381`）。
4. **QK-norm / RoPE**（`:388-471`）：
   - `qk_norm` 分支：**只在 `iter==first_kv_iter`** 对 Q 调一次 `rms_norm_sm100`（`:390-408`，
     含 rope、partial `ROTARY_DIM`、cos/sin 偏移 `seq_len-num_tokens`）；K 则**对每个有有效 token
     的 tile** 调 `rms_norm_sm100`（`:410-429`），窗口 = `kv_tokens_to_process`，
     `token_offset = curr_iter_len - kv_tokens_to_process`。
   - 否则若 `rope`：Q rope 只做首 tile（`:431-450`），K rope 逐 token（`:451-470`），走
     `rotary_embedding_hopper`（`tasks/hopper/rotary_embedding_hopper.cuh`，`docs/37` §4.3 已走读）。
5. **写回 KV cache**（`:475-491`）：新 token 的 K/V 从 `k_smem`/`v_smem` 搬进
   `paged_k_cache_dmem`/`paged_v_cache_dmem`（`dst_row = page_idx*PAGE_STRIDE + page_offset`）。
   `page_idx` 只取 `page_indices[first_kv_token_to_process/PAGE_SIZE]`——**假设一个 tile 不跨页**
   （`static_assert(PAGE_SIZE % KV_TILE_SIZE == 0)` 保证）。
6. **QK^T**（`:493-531`）：见 §4。
7. **`wg_barrier`**（`:532`）。
8. **在线 softmax + PV**（`:534-640`）：见 §5。

### 3.6 `kv_tokens_to_process` 的三角（`:383-387`）

```
kv_tokens_to_process = min(curr_iter_len,
                           max(iter*KV_TILE_SIZE + curr_iter_len - (seq_len-num_tokens), 0));
first_kv_token_to_process = iter*KV_TILE_SIZE + curr_iter_len - kv_tokens_to_process;
```

即：**当前 tile 里「属于老 token（必须走 norm/位置编码/写回）」的那一段**，其余是刚进来的新
token（norm 走 QKV 路、无位置编码需求以外的东西）。这是「同一 tile 混老新 token」的精确切分，
也是 `qk_norm`/`rope`/写回三处都用同一个 `kv_tokens_to_process` 的原因。

---

## 4. `X = QK^T`：`1x4x1` warp 布局 + 因果/窗口掩码（`:493-563`）

```
float x_frag_f[MMA_ITERS_M][8];  // 每 m-tile 一个 16x16 分数 tile
kt_col = (warp_idx<<4) + ((lane_idx>>4)<<3) + (lane_idx&0x7);
for m in MMA_ITERS_M:
  q_row = (m<<4) + (lane_idx & 0xF);
  for k in HEAD_DIM/16:
    q_col = (k<<4) + ((lane_idx>>4)<<3);
    kt_row = (k<<4) + (((lane_idx&0xF)>>3)<<3);
    src_ptr_Q  = q_row < num_tokens*NUM_QO_PER_KV ? q_smem(q_row,q_col) : zero_buffer(0,0);
    src_ptr_KT = kt_col < curr_iter_len ? k_smem(kt_col,kt_row) : zero_buffer(0,0);
    ldsm(src_ptr_Q, q_frag); ldsm(src_ptr_KT, kt_frag);
    mma_m16n16k16_bf16bf16bf32(x_frag_f[m], q_frag, kt_frag, x_frag_f[m]);
```

- **`1x4x1`**：4 个 warp 各负责 16 个 **key 位**（`kt_col` 由 `warp_idx` 定），m 维（query 行）
  由循环 `m` 展开——一个 warp 在 n（key）方向只碰自己那 16 列。
- **边界用 `zero_buffer` 兜**：Q 超出有效行、KT 超出 `curr_iter_len`，都读零块，不越界。
- `:505-516` 的注释是**本文件最重要的一段**（见 §6）。

随后是**掩码 + 部分最大**（`:534-563`）：

```
row = (m<<4) + (lane_idx>>2) + (((frag_idx&3)>>1)<<3);
col = (warp_idx<<4) + ((lane_idx&3)<<1) + ((frag_idx>>2)<<3) + (frag_idx&1);
token_idx = row / NUM_QO_PER_KV;  key_pos = col + iter*KV_TILE_SIZE;
query_pos = token_idx + seq_len - num_tokens;
is_valid  = (row < num_tokens*NUM_QO_PER_KV) && (key_pos <= query_pos)
            && (WINDOW_SIZE<=0 || key_pos > query_pos-WINDOW_SIZE);
x_frag_f[m][frag_idx] = is_valid ? x : -inf;
m_local[m][(frag_idx&3)>>1] = max(m_local[...], x);
```

**真正的因果/窗口公式在 `:556-558`**——`key_pos <= query_pos`（因果）与
`key_pos > query_pos-WINDOW_SIZE`（滑窗），无效位写 `-inf`。`docs/39`/`docs/43` 转述的
「causal mask 在 `:519`」是**行号漂移**（`:519` 是 `k` 循环头）；本篇以 `:556-559` 为准。
最后 `shfl.bfly` 在 4 线程行内做局部 max（`:565-572`）。

---

## 5. 在线 softmax 与 PV（`:574-640`）

- **rescale**（`:574-581`）：`rescale[m][b] = expf(m_prev*scale - m_local*scale)`；
- **d 部分和**（`:583-610`）：把 `x_frag_f` 从「分数」改成 `exp(x*scale - m*scale)`（无效位 →0），
  再 `shfl` 行内求和累进 `d`；
- **o rescale**（`:612-622`）：对已存累加器 `o[m][n][frag]` 乘 `rescale`；
- **PV**（`:624-639`）：
  ```
  uint32_t x_frag[MMA_ITERS_M][4], v_frag[4];
  for m:
    convert_f32_to_bf16_uint32(x_frag_f[m], x_frag[m]);   // 分数转 bf16 当 A 操作数
    v_row = (warp_idx<<4) + (lane_idx&0xF);
    for n in HEAD_DIM/16:
      v_col = (n<<4) + ((lane_idx>>4)<<3);
      src_ptr_V = v_row < curr_iter_len ? v_smem(v_row,v_col) : zero_buffer(0,0);
      ldsm_t(src_ptr_V, v_frag);                          // trans ldmatrix
      mma_m16n16k16_bf16bf16bf32(o[m][n], x_frag[m], v_frag, o[m][n]);
  ```
  **`1x1x4`**：warp 切 **p 位**（`v_row` 由 `warp_idx` 定），n 维由循环展开——与 QK^T 的
  `1x4x1` 正交（QK 的 warp 切 key，PV 的 warp 切 p）。V 用 `ldsm_t`（`ldmatrix.trans`，
  `copy_sm80.cuh:96-104`）是因为 PV 的 B 操作数是 `K×N` 需转置读。
- 循环尾 `curr_iter_len = next_iter_len`（`:642`）滚动窗口。

---

## 6. 跨 warp 归约：按 MMA m-tile 分块（`:645-735`）

KV 循环结束后，4 个 warp 各自持有部分 `m`/`d`/`o`，要合成每行的全局值。做法是
**逐 m-tile**：发布 → 屏障 → 归约 → 屏障 → 复用 buffer。

```
for m in MMA_ITERS_M:                        // :650
  m_local[m][b] *= (m_local[m][b]!=-inf ? scale : 1.f);
  s_m_buffer[tid*2 + b] = m_local[m][b];     // 发布
  s_d_buffer[tid*2 + b] = d[m][b];
  s_o_buffer[tid*64 + n*8 + frag] = o[m][n][frag];
  wg_barrier.arrive_and_wait();              // :664
  row_begin = m*16; row_end = min((m+1)*16, num_tokens*NUM_QO_PER_KV);
  for each elem in tile:
    t_idx = (row%8)*4 + (col%8)/2; mma_iter_n = col/16;
    frag_idx = ((col%16)/8)*4 + ((row%16)/8)*2 + (col%2);    // :696，fragment 布局见 :677-695
    for local_idx in 4:                       // 4 个 warp
      md_smem_offset = local_idx*32*2 + t_idx*2 + (frag_idx%4)/2;
      // 合并 m/d/o（在线 softmax 的交换律归约）
    if d_sink: d_global += expf(d_sink[row%NUM_QO_PER_KV] - m_global);   // :727-729
    o_smem.at(row,col) = bfloat16(o_global/d_global);        // :730
  wg_barrier.arrive_and_wait();              // :734
```

要点：
- **`s_m_buffer`/`s_d_buffer` 索引** `local_idx*32*2 + t_idx*2 + ((frag_idx%4)/2)`：`local_idx`
  枚举 4 个 warp（`*32` = 每 warp 32 lane），`t_idx`∈[0,32) 是「对应的行内线程」，末项取该
  线程两个部分值的哪一个。四路 `local_idx` 的在线 softmax 合并（`d = d_prev*exp(m_prev-m_global)
  + other_d*exp(other_m-m_global)`，`:713-714`）是**交换律归约**，与 flash-attention 的
  warp 内合并同构。
- **`s_o_buffer` 索引** `local_idx*32*64 + t_idx*64 + mma_iter_n*8 + frag_idx`：`*64` = 每线程
  64 float（`HEAD_DIM/16=8` × 8）。`o` 的归约同样乘 `exp(m_prev-m_global)` 缩放（`:721-722`）。
- **sink 只进分母**（`:724-729`）：`d_global += exp(d_sink[qhead] - m_global)`，`o`/分子不变——
  与 `docs/37` §5.1 一致，本篇逐行坐实。
- **buffer 只存一个 tile**：`:650` 的 for 每轮复用同三个 buffer，靠 `:664`/`:734` 两道
  `wg_barrier` 保序。注释 `:645-649` 明说「loop 边界不依赖 `threadIdx`，故两道屏障被所有线程
  均匀到达」。

最后**输出写回**（`:737-746`）：`o_smem` 按 `(src_row, src_col)` 反重排回 `o_dmem(dst_row,
dst_col)`（`dst_row = src_row/NUM_QO_PER_KV; dst_col = src_col + (src_row%NUM_QO_PER_KV)*HEAD_DIM`），
与 §3.3 的 Q 载入重排**互为逆**。

---

## 7. 因果掩码 / `MMA_ITERS_M` bug family：把 `docs/39`/`docs/43` 的尾巴钉到源码级

这是本篇补 `docs/43` §9 头号尾巴的核心。

### 7.1 `:505-516` 的修复注释（原文要点）

注释 `:506-516` 自陈：**早先的 workaround 是 `if (m > 0) continue;`**，它**丢掉了第一个之外的所有
tile**，牺牲了 query 槽 `≥ 16/NUM_QO_PER_KV`；**这败坏了已提交的 EAGLE3 token 流（K≥2）**——
因为 `ac=K+1` 的 bonus token 从一个**未被计算**的槽里读出来。**根因**：in-place MMA 累加器被声明
成**只写的 `"=f"` 输出 + 独立的 `"f"` 输入**（别名同一 C++ 对象），在 `#pragma unroll` 下编译器可
复用「已死亡」累加器的寄存器、**跨 fragment 污染**。**修复**：`mma_m16n16k16_bf16bf16bf32` 改用
**单一读写 `"+f"` 操作数**，每个 `x_frag_f[m]` 有独立 def-use 链，不再跳过任何 tile。

### 7.2 与 `docs/38` §6 / `docs/39` §6.3 / `docs/43` §4 三处并置

- `docs/38` §6 已从 `ampere/mma.cuh` 侧走读到同一修复（`include/.../tasks/ampere/mma.cuh:21-40`
  的 `"+f"(C[i])` 与 `(void)D`），并指出它是 `attention_sm100` `MMA_ITERS_M>1`、进而 EAGLE3
  token 流正确的 **load-bearing** 修复。**本篇补上 attention 侧的现场**：修复的消费者就是
  `:529` 的 `mma_m16n16k16_bf16bf16bf32(x_frag_f[m], q_frag, kt_frag, x_frag_f[m])`——
  C 与 D 传同一对象，正是 `"+f"` 的前提。
- `docs/39` §6.3 转述 `eagle3_commit_kernel` 注释里的 bug 调查（`paged_attention_sm100` 在
  `MAX_TOKENS=3` 下 slot 0 与 `MAX_TOKENS=2` 不一致），候选修复指向 `MMA_ITERS_M` / causal mask。
- `docs/43` §4.4 记「候选修复点 (a) `MMA_ITERS_M=2` 的 `-inf` `m_local` NaN 传播、(b) …」，
  并并置「当前快照 `:505-523` 的注释已给出修复：单读写 `"+f"` 累加器」。

### 7.3 本篇的三点事实校正

1. **因果掩码位置**：真正公式在 **`:556-559`**（`is_valid`），不是 `:519`。`:519` 是 PV/QK 的
   `k` 循环头。`eagle3_ops.cuh:281` 注释的 `:519` 属行号漂移（`docs/43` §1.3 亦已标漂移，但
   未给出正确位置；本篇给出）。
2. **`m_local` 的 `-inf` 传播**：`MMA_ITERS_M=2` 时若第二个 tile 全无效，`m_local[1][*]` 保持
   `-5e4`；`:651-652` 的 `m_local *= (m_local != -inf ? scale : 1.f)` 正是**为避免
   `-5e4*scale` 出 NaN** 而设的守卫；而 `:591-595` 的 `exp(x-m)` 对无效位直接写 0（`!= -inf`
   三元），不依赖 `exp(-inf-(-inf))`。所以「`-inf` NaN 传播」在当前快照**已被两处守卫覆盖**——
   与 `docs/43` §4.4 转述的「候选修复点 (a)」相比，源码现状是**已修**。
3. **`if (m>0) continue` 已不存在**：当前 `:504-531` 的 m 循环**对每个 m-tile 都算完整 attention**
   （注释 `:506` 明说），workaround 被移除。

> **口径**：以上为**静态源码事实**。本篇**不独立复现** `MAX_TOKENS=3` slot-0 分歧（SM100 不可在
> 目标卡执行），也不对上游 `f9eb70c` 作 bug 判定；`docs/39`/`docs/43` 关于「验收率/根因」的数字
> 仍为**源码注释自报**。

---

## 8. 生产布线：五层（逐条核对）

| 层 | 位置 | 事实 |
|---|---|---|
| `TaskType` | `docs/24`（`enum`）| `TASK_ATTN_SM100 = 257` |
| register | `src/kernel/task_register.cc:2326-2438` | `register_paged_attention_sm100_task`，emit `multitoken_paged_attention_sm100_task_impl<bfloat16, ...>`（`:2402-2418`），15 个模板实参、`request_id` 从 `task_metadata` 取（`:2427`） |
| op 映射 | `src/kernel/graph.cc:643-649` | `name=="paged_attention_sm100"` → `(num_attn_inputs, 1, TASK_ATTN_SM100, variant_id)` |
| 前端 | `python/mirage/mpk/persistent_kernel.py:1303-1432` | `paged_attention_layer`；按 `target_cc` 三选一（`:1427-1432`）；`params` 打包 `:1354-1402`；`grid_dim=(max_num_batched_requests, num_kv_heads)`（`:1404-1406`）；输入切片 `(-1,1,-1)`/`(-1,2,-1)`，sink `(-1,0,-1)`（`:1407-1420`）；`window_size`/`sink` 仅 SM100 支持（`:1424-1426`） |
| builder | `models/*/builder.py` | qwen3:399、glm4_moe:229、gpt_oss:302、eagle3:357/370 调用 `paged_attention_layer` |

**`params` 打包的两个细节**（`:1368-1402`）：默认尾部字段被裁到最近接受的尺寸（`6/8/10/11/12/13/14`，
`:1389-1394`），**仅在 `target_cc==100` 时才追加 `params[14]=max_tokens_per_request`**（`:1395-1402`）
——即 `MAX_TOKENS` 模板实参只有 SM100 路径显式给（其余回退 `max_tokens`，见
`task_register.cc:2384-2391`）。

---

## 9. 判据

- **C1（单次 launch）**：是常驻 megakernel 内的 device task，**并入**同一 kernel，不自 launch。
  **贡献 C1**。
- **C2/C4/C5**：task 身体不构成新的协调/调度机制——调度仍由 `docs/06` 的 worker/scheduler 事件环
  提供。**不贡献**（同 `docs/29`–`docs/43` 的 device task 走读）。但 §2 的 SMEM 让步是 **C2 的
  直接约束**（常驻预算逼出 Q/O 复用与 tile 分块归约）。
- **C3（多算子范围）**：把「QK-norm、RoPE（partial）、paged KV 读写、GQA attention、在线 softmax、
  sink、输出重排」压进**单个 task**，是 C3 边界在通用 attention 上的具体形态（`docs/37` §5.1 同结论，
  本篇给逐行坐标）。**贡献 C3**。

---

## 10. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。SM100 专属，目标卡
  H100/sm90 不可执行。
- **`MAX_TOKENS=3` slot-0 分歧未独立复现**：§7 只做「源码事实校正」，不判上游对错；验收率/根因
  数字仍为注释自报。
- **命名屏障 ID 6/7 的口径冲突只记不判**：与 `docs/34` §3.5「须 ≥8」、`docs/37` §8「并存不判」一致。
- **未展开的更深层**：`rms_norm_sm100`（`norm_sm100.cuh`）内部、`rotary_embedding_hopper`
  （`docs/37` §4.3 已走读，本篇只引调用点）、`smem_row`/`ldsm`/`dmem_row` 布局定义
  （`docs/38`/`docs/39` 已覆盖）未逐行；attn sink 的**数值影响**未实测。
- **`grid_dim` 第二轴（KV head）与 task 粒度的确切映射**：本篇据 `request_id` 入参与输入切片
  静态推断「一个 task = 一个 (request, KV head)」，未构建追踪。

---

## 11. 参考坐标（可核对）

- device body：`include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh`
  （`multitoken_paged_attention_sm100_task_impl:62-748`；模板参数 `:34-61`；SMEM 预算 `:168-231`；
  QK^T+掩码 `:493-563`；在线 softmax `:574-622`；PV `:624-639`；跨 warp 归约 `:645-735`；
  输出 `:737-746`；`"+f"` 修复注释 `:505-516`；因果/窗口掩码 `:556-559`）。
- 编译单元：`tasks/blackwell/task_header.cuh:28`、`tasks/hopper/task_header.cuh:26`。
- 注册/映射：`src/kernel/task_register.cc:2326-2438`；`src/kernel/graph.cc:643-649`。
- 前端：`python/mirage/mpk/persistent_kernel.py:1303-1432`。
- builder：`models/qwen3/builder.py:399`、`models/glm4_moe/builder.py:229`、
  `models/gpt_oss/builder.py:302`、`models/eagle3/builder.py:357/370`。
- 支撑原语：`tasks/ampere/mma.cuh:21-40`（`"+f"` 累加器，`docs/38` §6）；
  `tasks/common/copy_sm80.cuh:43-54`（`load_smem`）、`:87-104`（`ldsm`/`ldsm_t`）；
  `tasks/common/utils.cuh:50-58`（`convert_f32_to_bf16_uint32`）、`:61-67`（`shfl_xor_sync`）、
  `:118-121`（`clear_8_floats`）；`tasks/common/worker_config.h:18`（`NUM_THREADS=128`）。
- 本专题交叉引用：`docs/37` §5.1（摘要）、`docs/38` §2.4/§6、`docs/39` §6.3、`docs/40`（退役版）、
  `docs/43` §4/§9、`docs/24`（`TaskType` 257）。
- 上游对比：`tasks/deprecated/paged_attention.cuh`（`docs/40`）、`tasks/hopper/multitoken_paged_attention_hopper.cuh`。
