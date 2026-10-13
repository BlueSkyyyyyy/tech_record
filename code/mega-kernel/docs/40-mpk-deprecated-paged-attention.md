# MPK 退役实现走读：`tasks/deprecated/paged_attention.cuh`（P126-1）

> 定位：本文补掉 `docs/39-mpk-speculative-decoding.md` §9 与 `docs/37`/`docs/38` 都点到、却从未
> 逐行打开的那一处尾巴——「`tasks/deprecated/` 此前各篇均『不涉』，其唯一成员
> `paged_attention.cuh`（496 行）可直接走读」。前面 `docs/07`/`docs/33`–`docs/39` 走读了 MPK 各代
> 「一个 task = 一个 device 身体」，本文把**仓库里唯一归档在 `deprecated/` 下的 task** 打开，看清
> 「Megakernel 演进中被淘汰的 task 长什么样、为什么会被淘汰、它和今天的实现差在哪」。
>
> 快照 `mirage-project/mirage` 分支 `mpk`、commit `f9eb70c`（与 `docs/06`–`docs/39` 同快照）。
> **纯源码走读（资料分析），无新实验**——这个 task 既不进生产、也不进任何 `task_header` 的编译单元，
> 本文不构建、不运行。

---

## 0. 三条 headline

1. **它是仓库里唯一一个把「paged-attention 计算」与「新 token 的 K/V 写回页缓存」融进同一个 task
   的 device 身体**：任务末尾（`:473-484`）把当前 SMEM tile 里最后一行的 K/V 写回 `paged_*_cache`
   的「最后一个页的 `(seq_len-1)%PAGE_SIZE` 位置」。而今天的实现把这两件事**拆成两个 task**——
   `multitoken_paged_attention`（`tasks/ampere/*`，只读页缓存，注释 `:99-102` 明说「新 token 的
   metadata/页已预先分配、数据由后续另一步拷进页」）与独立的 KV 写入 task（MLA 走
   `mla_kv_cache_gather_sm100.cuh`、DFlash 走 `dflash_kv_store_sm100.cuh`，见 `docs/35`/`docs/36`/`docs/37`）。
   即：**Megakernel 的融合边界不是「越融越好」——把 KV append 从 attention 里拆出去，是这套实现
   后来做的选择**，本节把被拆掉的那一版并置出来。
2. **它是一套「自包含、单 batch × 单 KV head × 单 CTA」的 FA2 手写实现**：`m16n16k16` `mma.sync`
   （1x4x1 做 QK^T、1x1x4 做 PV）、在线 softmax（`m`/`d`/`o` 三段 rescale）、4 warp 在 key 维上
   分片 + warp0 跨 warp 合并、可选融合 QK-norm/RoPE、NHD 页缓存、双缓冲 64-key tile。信息密度与
   今天的 `attention_sm100.cuh`（`docs/37` §5.1）同级，但**没有** GQA（Q 与 KV head 比例）、
   滑窗、attention sink、partial RoPE、`PAGE_STRIDE_ROWS` 等后来加的一切——是一个**功能最早期的
   快照**。
3. **它的接线状态是「彻底的孤儿」**：全仓**没有**任何文件 `#include` 它（唯一引用在单测 wrapper
   `tests/runtime_python/runtime_kernel_wrapper.cu:363`，被 `#ifdef DEPRECATED_TESTS` 包住）；而
   `DEPRECATED_TESTS` **全仓无人定义**、该 wrapper 顶部的 `using kernel::paged_attention_task_impl;`
   被注释（`:18`）、pybind `m.def("paged_attention", …)` 也被注释（`:1886`）、且 wrapper 只 include
   了 `tasks/ampere|blackwell/task_header.cuh`（`:2-3`）而**从不 include 本文件**。于是它既不在生产、
   也不在任何现存测试的编译单元里——是 `docs/16`/`docs/20`/`docs/29` 那族「已实现未接线」里
   最彻底的一种：**连同接线尝试一起被归档**。

---

## 1. 目录位置、包含约定与「谁把它编进生产」

`tasks/deprecated/` 下只有这一个文件，496 行：

| 文件 | 行数 | 提供 | 登记 `TaskType` | 生产 `task_header` include？ | 任何 include？ |
|---|---|---|---|---|---|
| `deprecated/paged_attention.cuh` | 496 | `paged_attention_task_impl` | 无 | **否** | **否**（仅单测 wrapper 引用、且被宏包住） |

- **旧式裸 include 约定**（`:16-27`）：`#include "common.h"` / `"copy_sm80.cuh"` /
  `"dmem_layout.cuh"` / `"mma.cuh"` / `"norm.cuh"` / `"reduction.cuh"` / `"smem_layout.cuh"` /
  `"utils.cuh"` 等**无目录前缀**的头。今天的 `tasks/ampere/*` 与 `tasks/blackwell/*` 已改用
  `tasks/ampere/xxx.cuh` 形式的伞状 `task_header.cuh`（`docs/39` §1 记过这条演进，`docs/38`
  记过 `tasks/common/common_header.cuh` 才是今日的 arch-agnostic 地基）。裸 include 说明本文件
  写于「所有 task 头平铺在一个目录」的年代，今天这些名字（`common.h`/`copy_sm80.cuh`/…）**已不
  在 `persistent_kernel/` 的包含根下**（`common.h` 仅在 `include/mirage/transpiler/` 与
  `cpp_examples/` 存在）——即便有人想编译它，include 路径也先对不上。这是一条**静态事实**，
  **只记录不判对错**（同 `docs/39` 对 `target_verify.cuh` 的 `../common.h` 态度）。
- **它被放进 `deprecated/`，但没有任何 deprecation 机制引用它**：不是「用宏关掉的活代码」，
  而是「搬到一边、无人再碰」。对照 `docs/31`/`docs/32` 的「已实现未接线」（有 include、有注册、
  只是无调用者），这里连 include/注册都没有。

---

## 2. device 身体逐段走读

### 2.1 签名、布局假设与「一个 task 的粒度」

```
template <typename T, int NUM_Q_PER_KV, int HEAD_DIM, int PAGE_SIZE,
          int MAX_SEQ_LEN, int KV_STRIDE>            // :33-38
__device__ __forceinline__ void paged_attention_task_impl(
    void const *qkv_ptr, void *paged_k_cache_ptr, void *paged_v_cache_ptr,
    void *output_ptr, void const *paged_kv_indices_buffer_ptr, size_t seq_len,
    bool qk_norm, bool rope, void const *q_norm_weight_ptr,
    void const *k_norm_weight_ptr, void const *cos_ptr, void const *sin_ptr,
    float q_eps, float k_eps);                          // :43-56
```

- 注释 `:31-32`：*"for each task, we process one batch and one KV head at a time"*——**task 粒度 =
  一个 (batch, KV head)**，一个 CTA 一次只解码一个 KV head 的一组 query（`NUM_Q_PER_KV` 个 Q head
  共享一个 KV head，即最早的 GQA-共享形态，但**写死为「一个 KV head」**）。
- KV 缓存布局注释 `:39-41`：**NHD**，shape `[max_num_pages, page_size, num_kv_heads, head_dim]`。
  注意 `KV_STRIDE` 是模板参数（`dmem_row<T, KV_TILE_SIZE, HEAD_DIM, KV_STRIDE>`，`:85`），即行跨距
  与 head 数解耦——后来 `docs/35` §3 给 `mla_kv_cache_gather` 记录的「扁平化 + ILP」、`docs/37`
  §5 的 `PAGE_STRIDE_ROWS` 都是这条跨距自由度的发展。
- QKV 输入按**拼接**解析（`:71-80`）：`d_q = qkv_ptr`、`d_k = qkv_ptr + NUM_Q_PER_KV*HEAD_DIM`、
  `d_v = qkv_ptr + (NUM_Q_PER_KV+1)*HEAD_DIM`。注释 `:69-70` 明说 `qkv_ptr` **指向对应请求那一行**
  而不是整个张量起点——即调用方（runtime）要对每个 task 把指针偏移算好，**和今天
  `runtime.cc` 的 per-task 指针逻辑同构**（`docs/32` §5.3、`docs/34` §5、`docs/37` §1 都记过这一点）。

### 2.2 SMEM 分区与 `page_indices` 预取

- SMEM 用**手算偏移**（`:97-150`）：`zero_buffer`（`sizeof(T)*8`，给越界 lane 一个安全读址）、
  `s_q`（`NUM_Q_PER_KV*HEAD_DIM`）、`s_k`/`s_k_buffer`/`s_v`/`s_v_buffer`（各 `64*HEAD_DIM`，双缓冲）、
  `s_o`、`s_q_norm_sum`/`s_k_norm_sum`（各 4 float，给 4 warp）、`s_m_buffer`/`s_d_buffer`/`s_o_buffer`
  （各 `NUM_THREADS` float，跨 warp 合并用）。`NUM_THREADS` 取 `worker_config.h` 的 128（`docs/38`
  §2.4），单测 launcher 也按 `block=128`（见 §3）。
- 页表预取（`:162-172`）：`num_pages = ceil(seq_len/PAGE_SIZE)`；`page_indices` 是
  `__shared__ int[MAX_PAGES_PER_REQUEST]`。**静态事实**：这段代码只在 `num_pages % (16/sizeof(int))
  != 0` 时、把**尾部不对齐的几个页**拷进 `page_indices`（`:165-171`）；**对齐的头段没有任何显式
  拷贝**。本文只把它记为「预取逻辑不完整/被精简」的静态观察（这段在归档时被裁掉了一截的可能性
  存在），**不判对错**。注意越界 lane 用 `zero_buffer` 兜底（`:194`/`:232`/`:333-335`）的思路，
  与今天 `docs/37` §5.1 的 mask 是同一目的。
- `s_o_buffer` 的 size 常量 `S_O_BUFFER_SIZE` **被注释掉**（`:136`），而写入索引是
  `(threadIdx.x<<5)+…`（`:419-420`，按 32 float/线程，即需要 `NUM_THREADS*32` 个 float）。
  即「声明大小与使用大小不一致、大小常量被注释」。不过单测 launcher 申请了 112640 B 动态 SMEM
  （§3），`S_O_BUFFER_OFFSET`（≈68656 B）之后到 112640 B 之间的余量（≈43 KB）足以容纳
  `128*32*4 = 16384 B`，故**在 112640 的 launch 下不会逸出动态 SMEM**——同样**只记录不判对错**。

### 2.3 双缓冲 KV tile 载入 + 缓冲 rotate

- 首个 64-key tile 载入（`:187-207`）：每个 chunk 判断 `dst_row + cp_finished_seq_len < seq_len-1`；
  **真则从页缓存读**（`page_idx/page_offset` → `src_row = page_idx*PAGE_SIZE+page_offset`），
  **假则从 QKV 读那一行**（`:201-206`）——这正是「**被解码的新 token 还没进缓存，它在 QKV 里**」
  的处理，也是本文件「attention + append 融合」的前提。
- 主循环每轮载入下一 tile 并 `cp_async_fence()` + `cp_async_wait<1>()`（`:224-252`），然后
  **按 `iter` 奇偶交换双缓冲指针**（`:254-265`，`k_smem.set_ptr(...)`），`__syncthreads()` 后进入计算。
  这套「cp.async 双缓冲 + 指针 rotate」与 `docs/37` §5.2 DFlash「64-key tile 流式」、
  `docs/07`/`docs/13` 记的软件流水是同一家族，只是更朴素（无 mbarrier、无 warp specialization，
  纯 `cp.async` group 计数 + block 级同步）。

### 2.4 融合的 QK-norm / RoPE

`:268-314`：在第一个 tile 上对 Q 做 norm/rope、在最后一个 tile 上对 K 做——因为 K 的「当前 token」
行只出现在最后一个 tile（`:194`/`:232` 的条件保证只有 `seq_len-1` 那行来自 QKV）。
- `qk_norm` 分支调 `rms_norm<T, QOSmem|KVSmem, …>`（`:271-293`），带 `rope` flag 与
  `cos/sin + (seq_len-1)*HEAD_DIM` 偏移（**只旋转当前位置**）；
- 非 `qk_norm` 分支调 `rotary_embedding<T, …>`（`:299-311`）。

即 **norm 与 rope 被融进 attention task 本身**（不是独立 task）。今天 `docs/37` §4.1/§4.2 把
`norm_sm100`/`rotary_embedding_sm100` 做成了**可被 attention 复用的独立 device 身体**（`attention_sm100`
在 `qk_norm`/`rope` 时调它们，`:388-429`），粒度选择不同：这里是「内联进 task」，那里是「拆成
可复用 task」。注意本文件的 `rms_norm`/`rotary_embedding` 用的是 `ampere/` 下的旧签名
（`docs/38` §2.4 记过 `common/` 与 `ampere/` 的线程数口径落差）。

### 2.5 主体：QK^T → 在线 softmax → PV

- **QK^T**（`:318-340`）：`m16n16k16_bf16bf16bf32`（`tasks/ampere/mma.cuh:21`，`docs/38` §6 记过它的
  `"+f"` in-place 累加器修复——**本文件用的是 `x_frag_f` 同时作 C/D**，即已经是「单一读写」形态）。
  warp layout **1x4x1**：`kt_col = (warp_idx<<4)+…`，mma 只沿 K 维迭代（`:326-339`），越界 lane 读
  `zero_buffer`。`:320-321` 注释写明约束「`NUM_Q_PER_KV<=8`、`KV_TILE_SIZE==64`、`HEAD_DIM%16==0`」。
- **在线 softmax**（`:342-394`）：`m` 更新（本线程 4 个 fragment 的 max → 行内 4 thread 的
  `shfl_xor` 两级，`:346-360`）；`rescale = expf(m_prev*sm_scale - m*sm_scale)`（`:362`）；
  `d` 更新（对 fragment 做 `exp` 求和 + thread 内归约，`:364-385`）；`o` rescale（`:387-394`）。
  记一个**初始化 quirk**：`m=-inf, d=1.f`（`:212-213`）——`d` 初始化为 1 而非 0，看似可疑，但因
  首轮 `m_prev=-inf` 使 `rescale=exp(-inf-m)=0`、`d = 1*0 + d_partial = d_partial`，故对结果正确
  （`d` 的初值在首轮被清零），**只记事实**。
- **PV**（`:396-408`）：`ldsm_t`（转置 load）取 V，warp layout **1x1x4**，`o[n]` 累加（`:401-407`）。
- 与 `docs/37` §5.1 的 `attention_sm100` 对照：**同样的 `m16n16k16` mma.sync**（SM100 块里也有走
  mma.sync 的通用 attention），但 `attention_sm100` 用 **Q 与 O 复用同一 SMEM 段**（`:187-196`）、
  跨 warp 归约按 **MMA m-tile 分块**（`:645-735`），本文件则是**朴素的 `m`/`d`/`o` 全量 buffer +
  固定 4 warp 合并**。

### 2.6 跨 warp 合并（warp0）

`:413-454`：各线程把局部 `m`/`d`/`o` 写进 `s_m_buffer/s_d_buffer/s_o_buffer`（`s_o_buffer` 按
`threadIdx.x<<5` 展开成 32 float/线程），`__syncthreads()` 后**只有 `warp_idx==0` 遍历 4 个 warp 的
槽做在线合并**（`:429-453`，标准的两路 log-sum-exp 合并 `d_prev*exp(m_prev-m)+other_d*exp(other_m-m)`）。
`:456-471` 再归一化 `o/d` 写回 `o_smem`。**这正是 `docs/37` §5.1 用 issue #702 优化掉的形态**——那里
跨 warp buffer 只存一个 m-tile（避免 smem 随 `MAX_TOKENS` 增长），这里是全量
`sizeof(float)*NUM_THREADS*32`，且固定只支持 4 warp（`for t_idx<4`）。

### 2.7 KV cache append（本文件的「融合点」）

```
// update KV Cache
for (int col = threadIdx.x; col < HEAD_DIM; col += NUM_THREADS) {
  int src_row = (seq_len - 1) & (KV_TILE_SIZE - 1);   // = (seq_len-1) % 64
  int page_idx = page_indices[num_pages - 1];          // 最后一页
  int page_offset = (seq_len - 1) % PAGE_SIZE;
  int dst_row = page_idx * PAGE_SIZE + page_offset;
  paged_k_cache_dmem.at(dst_row, col) = k_smem.at(src_row, col);
  paged_v_cache_dmem.at(dst_row, col) = v_smem.at(src_row, col);
}                                                       // :473-484
```

把当前 token 的 K/V（在最后 tile 的第 `(seq_len-1)%64` 行）**写回页缓存的最后一个页**。
注释 `:477-478`：*"page allocation is done outside of the task, so we assume there are always
available slots in the last page"*——即**页分配在 task 外部**（与今天一致），但**数据写入在 task 内部**
（今天不一致）。这是本 task 与后世实现最本质的分野。

### 2.8 O 写回

`:486-493`：把 `o_smem` 按 `[NUM_Q_PER_KV, HEAD_DIM]` 写回 `o_dmem`。整个 task 无任何跨 task /
跨 CTA 通信（无 NVSHMEM、无原子、无 event），是**纯「一 task 一 CTA 自足」**的形态。

---

## 3. 与今天两代实现的对照

| 维度 | `deprecated/paged_attention.cuh`（本节） | `ampere/multitoken_paged_attention*`（生产） | `blackwell/attention_sm100.cuh`（生产） |
|---|---|---|---|
| task 粒度 | 1 batch × 1 KV head | 多 token × GQA（`NUM_QO_HEADS`/`NUM_KV_HEADS`） | 多 token × GQA（`TASK_ATTN_SM100`） |
| QKV / 页表输入 | 拼接 QKV + `paged_kv_indices_buffer` + `seq_len` | packed QKV + `qo_indptr`/`kv_indptr`/`indices`/`last_page_len` + `request_id`（`:36-40`） | 同左 + `sink_ptr` / partial RoPE / sliding |
| KV cache 写 | **融进 task**（`:473-484`） | **只读**：注释 `:99-102` 明说页已预分配、数据由后续另一步拷入 | 只读；写入走独立 task（`mla_kv_cache_gather`/`dflash_kv_store`） |
| QK-norm/RoPE | 内联进 task（`rms_norm`/`rotary_embedding`，`:268-314`） | 内联/可选 | 调独立 `norm_sm100`/`rotary_embedding_sm100`（`docs/37` §4） |
| 跨 warp 合并 | 全量 `m/d/o` buffer + 固定 4 warp（`:429-453`） | 按 m-tile 分块（issue #702） | 同左（`:645-735`） |
| Q/O SMEM | 分开 | 分开 | **复用**（issue #740） |
| 线上接线 | **无**（孤儿；`DEPRECATED_TESTS` 未定义、include/using/m.def 全缺） | 活（`register_task` / runner） | 活（`register_paged_attention_sm100_task`，`docs/37` §5.1） |
| 单测 launcher | `runtime_kernel_wrapper.cu:363/465`（`grid(1,1,1)`、`block(128,1,1)`、smem 112640，`:404-408`；调用模板 `<bf16,4,128,64,512,128>`） | `launch_multitoken_paged_attention`（活） | SM100 专用目录 |

**一个值得记的演进结论**：本文件把「attention + KV append + norm/rope」全塞进一个 task，粒度是
「一个 KV head」。后世把它**拆成多个 task**（attention 只读 + 独立 KV store + 独立 norm/rope），
换来的正是 megakernel 需要的两样东西：(a) **task 更小、类型更多**，调度器能把它们与别的 task
交错（`docs/04` §8/§10 的成本侧、`docs/11` §3 的 ready 结构）；(b) **attention task 可被 prefill/
decode/draft/verify 多处复用**（`docs/37`/`docs/39`），而不是每种场景各写一个「连写回都包进来」
的巨型 task。这与 `docs/05`/`docs/23` 的「少而重 vs 多而杂」是一体两面：**在同一个常驻 kernel 里，
把‘写回’从‘读’里拆出去，是让 task 可复用、可调度的先决条件**。

---

## 4. 与论文 §4.1/§4.2 对照 + 判据映射

- 论文 §4.2（快照 `mpk-v2.txt` 行 1126–1146）说 **task 的 CUDA 实现以 thread block 为单位生成**
  （Mirage superoptimizer 逐 task 搜出 thread-block 级实现）。本文件的形态正是「一个 task = 一个
  thread block 的完整实现」——`m16n16k16` mma + 在线 softmax + 双缓冲载入，**与论文描述一致**，
  且是**手写**（非超优化器产出），对上 `docs/07` §0/`docs/15` §7 的「superoptimizer vs 手写/CUTLASS
  并存」。它的存在说明：**「一个 task 一个 CTA」是 MPK 的 device 层契约，无论手写还是生成**。
- **判据映射**：本文件是**一个 device task body**，不单独贡献任何 C 判据中的 C1（单次 launch）/
  C2（常驻）/ C4（设备内协调）/ C5（去中心化调度）——这些由运行层拿（`docs/22` §8：**只有运行层
  拿 C1/C2/C4/C5**）。它只决定 C3 的**范围**（「一个 paged-attention 段被编成一个 task」），
  与 `docs/29`–`docs/39` 各 device task 走读同档。**其独特性在于：它是被淘汰的 C3 范围划分**。

---

## 5. 局限（诚实标注）

- **纯源码走读，未构建、未运行**：所有行号为快照 `f9eb70c` 静态事实。本文件不被任何编译单元
  include、且 include 路径（裸 `common.h` 等）在当前目录布局下已不可解析，本文不尝试修复/编译。
- **「退役原因」是推断**：本文给出的「把 KV append / norm / rope 拆成独立 task 换来可复用与可调度」
  是基于**并置两代代码**得出的工程解释，非仓库注释或提交信息所写；`tasks/deprecated/` 是一个目录
  名，**没有**任何 deprecation notice、CHANGELOG 或提交记录被本文核对。
- **两处静态观察只记不判**：`page_indices` 只写尾段（`:165-171`）、`S_O_BUFFER_SIZE` 被注释
  （`:136`）而 `s_o_buffer` 按 `threadIdx.x<<5` 使用（`:419`）——均为静态事实，是否构成 bug 取决于
  被裁掉的上游代码与 launch 尺寸，本文不下结论（同 `docs/36`/`docs/39` 对源码自陈的态度）。
- **`NUM_Q_PER_KV<=8` 等早期约束**、`d=1.f` 初始化 quirk 均照实转述，未做数值验证。
- 未展开：`dmem_layout.cuh`/`smem_layout.cuh`/`rms_norm`/`rotary_embedding`（ampere 版）的**内部**——
  这些是 `docs/38`（`tasks/common/*`）与 `docs/37` §4（SM100 norm/rope）的同族，本文只引用不重复。

---

## 6. 参考坐标（可核对）

- 快照：`mirage-project/mirage`，分支 `mpk`，commit `f9eb70c`。
- 本文件：`include/mirage/persistent_kernel/tasks/deprecated/paged_attention.cuh`（496 行，`tasks/deprecated/`
  唯一成员）。
- 唯一引用（且被宏包住、无 include）：`tests/runtime_python/runtime_kernel_wrapper.cu`——
  `using` 注释 `:18`、`#ifdef DEPRECATED_TESTS` `:342`、wrapper `:344-383`、launcher `:390-435`
  （`grid(1,1,1)`/`block(128,1,1)`/smem 112640，`:404-408`）、host fn `paged_attention:439-477`
  （实例化 `<bf16,4,128,64,512,128>` `:465`）、`m.def("paged_attention", …)` 注释 `:1886`；
  `DEPRECATED_TESTS` 全仓仅上述 `#ifdef`，**无任何 `#define`/`setup.py` 定义**。
- 对照实现：`tasks/ampere/multitoken_paged_attention.cuh:19-108`（GQA 派发）与
  `tasks/ampere/multitoken_paged_attention_32_64.cuh:99-102`（「页已预分配、数据后拷」的只读假设）；
  `tasks/blackwell/attention_sm100.cuh`（`docs/37` §5.1）。
- 独立 KV 写入 task：`tasks/blackwell/mla_kv_cache_gather_sm100.cuh`（`docs/35` §3）、
  `tasks/blackwell/dflash_kv_store_sm100.cuh`（`docs/37` §5.2）。
- 论文：`mpk-v2.txt` 行 1126–1146（§4.2 task 实现生成）。
- 相关分篇：`docs/37-mpk-sm100-misc-tasks.md`（§5 非 MLA attention）、
  `docs/38-mpk-common-primitives.md`（`ampere/mma.cuh` 的 `"+f"` 修复）、
  `docs/39-mpk-speculative-decoding.md`（§9 尾巴）、`docs/22-mpk-compiler-manual.md`（§8 判据总映射）。
