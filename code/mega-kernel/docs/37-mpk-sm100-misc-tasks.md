# MPK SM100 非 GEMM / 非 MLA task 家族走读：路由、top-k、采样、位置编码、归一化、量化与小算子（P123-1）

> 定位：本文补掉 `docs/35-mpk-sm100-mla-family.md` **§9「局限」明留的相邻尾巴**与
> `docs/36-mpk-sm100-mla-2sm.md` **§8「局限」明留的同一处**（原话）——
>
> > 注意力之外的 SM100 task 家族（`attention_sm100`/`dflash_*`/`inkling_*`/`glm_moe_router_sm100`/
> > `prob_scatter_sm100`/`softmax_gather_sm100`/`topk_*`/`sampling_sm100`/`rotary_embedding_sm100` 等）、
> > `mla_prefill_host.cu` 这个 prefill 的原型来源——本文未覆盖。
>
> `docs/33`–`docs/36` 把 SM100 的 **GEMM / FP8 / group-GEMM / MLA attention** 这条「重算子主线」逐段打开了；
> 但一个常驻 megakernel 里除了这些重 task，还塞满了**大量「轻但必需」的 task**：MoE 路由、top-k 选择、
> 采样、位置编码（RoPE）、RMSNorm、FP8 分组量化、残差/初始化、以及非 MLA 的 paged / DFlash / Inkling
> attention。本文把这批 `include/mirage/persistent_kernel/tasks/blackwell/` 下的**非 GEMM、非 MLA**
> device 身体逐个走通，并核对它们在 `TaskType` / `graph.cc` / `persistent_kernel.py` / builder 四层的
> 生产布线。快照 `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/36` 同快照）。
> **纯源码走读（资料分析），无新实验**——SM100（`sm_100f`/`sm_100a`）在目标卡 H100/sm90 不可执行。

---

## 0. 三条 headline

1. **这批 task 是 megakernel 的「调度与数值枢纽」，且几乎全是「极简 device 身体 + 静态角色」**：与
   `docs/33`/`docs/34` 的 FP8 GEMM（三/四角色 warp 特化 + UTCCP + TMEM）或 `docs/36` 的 2SM MLA
   （七 warp 五 pipeline + cluster）相反，本节这些 task 大多是**几十到几百行、单次 `__syncthreads`/命名
   屏障、无 TMA/TMEM** 的朴素实现：路由/采样是「一线程一行 + warp/subgroup 减少」、RoPE/RMSNorm 在
   **SMEM 上原地**做、小算子是逐元素网格。它们印证 megakernel 的「**少而重 + 多而薄**」混合：
   重 task 决定寄存器/代码足迹的**上限**（`docs/04` §8/§10），而这些薄 task 是**依赖图里的关节**——
   它们一多，`TaskType` 数与事件扇入/扇出就上去了（对上 `docs/08` 依赖分析、`docs/13` 端到端夹层）。
2. **「一个 task = 一个 CTA」在薄 task 上的两种常数/几何适配**：路由与采样把 `grid` 压成 `(1,1,1)` 或
   `num_partial_tasks`，`blockIdx.x` 在常驻内核里是 **worker 索引**（不是 tile 索引），故 device 身体一律
   **遍历「本 task 的全部 row」而不是按 `blockIdx` 取 row**；最直白的证据是
   `per_token_group_quantize_fp8.cuh:87-92` 的注释 *"Under the MPK persistent runtime, blockIdx.x is the
   PHYSICAL worker id, not the task's batch row … Fix: each task processes ALL rows"*——与 `docs/32`
   §5.3 的 `kNumSMs=1` HACK、`docs/34` §3.5 的 TMEM-warp 让步同族。
3. **同一「路由/采样」语义在仓库里有 3–4 份独立实现、由**模型家族**而非硬件代际决定**：MoE 路由有
   **group-aware 向量化**（`topk_sigmoid_sm100`，DeepSeek 系，要求专家数是 2 的幂）、**朴素标量 160 专家**
   （`glm_moe_router`，因 GLM 的 160 不是 2 的幂、注释自陈不能复用 sigmoid 版）、**logsigmoid-softmax
   256+2**（`inkling_moe_router`）三种；采样有 **common 库版** `sampling_from_logits_kernel`
   （`tasks/common/sampling.cuh:162`，整体一遍 Gumbel）与 **SM100 分块版**
   `sampling_partial_sm100_kernel`/`sampling_reduce_sm100_kernel` 两套，且 `register_sampling_sm100_task`
   与 `register_sampling_reduce_sm100_task` **共用同一个 `TASK_SAMPLING_SM100`**（`:2543`/`:2645`）。

---

## 1. 生产布线：一个 `TaskType`、一个 op 名、一个 Python layer、一个 builder 调用

先给四层坐标（这一批 task 的 `TaskType` 落在 `runtime_header.h` 的 **SM100 块（230–299）** 与
**家族块（350–353 / 401）**）：

| device 文件 | `TaskType`（`runtime_header.h`） | `graph.cc` op 名 → `task_config` | Python layer（`persistent_kernel.py`） |
|---|---|---|---|
| `attention_sm100.cuh` | `TASK_ATTN_SM100 = 257` | `"paged_attention"`（`:644-649`，`num_attn_inputs` 输入） | `paged_attention_layer` |
| `argmax_sm100.cuh` | `258/259` | `"argmax_partial_sm100"`/`"argmax_reduce_sm100"` | `argmax_partial_layer:2603`/`argmax_reduce_layer:2632` |
| `topk_softmax_sm100.cuh` | `260` | `"moe_topk_softmax_sm100"`（`:678-682`） | `moe_topk_softmax_routing_layer:1972` |
| `mul_sum_add_sm100.cuh` | `261` | `"moe_mul_sum_add_sm100"`（`:751-756`） | `moe_mul_sum_add_layer:2325` |
| `tensor_init.cuh` | `262` | `"tensor_init"`（`:675-677`） | `tensor_init_layer:1952` |
| `common/sampling.cuh` | `TASK_SAMPLING_SM100 = 265` | `"sampling_sm100"`（`:661-663`） | `sampling_sm100_layer:2659` |
| `per_token_group_quantize_fp8.cuh` | `TASK_QUANTIZE_FP8_SM100 = 275` | `"quantize_fp8_sm100"`（`:841-846`） | `quantize_fp8_layer:2162` |
| `sampling_sm100.cuh`（partial） | `TASK_SAMPLING_PARTIAL_SM100 = 279` | `"sampling_partial_sm100"`（`:665-669`） | `sampling_partial_layer:2685` |
| `topk_sigmoid_sm100.cuh` | `TASK_MOE_TOPK_SIGMOID_SM100 = 280` | `"moe_topk_sigmoid_sm100"`（`:683-687`） | `moe_topk_sigmoid_routing_layer:1995` |
| `elementwise_add_sm100.cuh` | `TASK_ELEMENTWISE_ADD_SM100 = 281` | `"elementwise_add_sm100"`（`:722-726`） | `elementwise_add_layer:2550` |
| `softmax_gather_sm100.cuh` | `TASK_SOFTMAX_GATHER_SM100 = 282` | `"softmax_gather_sm100"`（`:727-731`） | `softmax_gather_layer:2957` |
| `prob_scatter_sm100.cuh` | `284/286` | `"prob_scatter_sm100"`/`"prob_extract_sm100"`（`:741-750`） | `prob_scatter_layer:2992`/`prob_extract_layer:3010` |
| `dflash_attention/norm_rope/kv_store` | `296/297/298` | `"dflash_attention"` 等（`graph.cc:457-471`） | `dflash_*_layer:970/1007/1030` |
| `inkling_sconv/moe_router/attention` | `351/352/353` | `"inkling_sconv"`/`"inkling_moe_router"`/`"inkling_attention"`（`:472-491`） | `inkling_*_layer:1099/1125/1050` |
| `glm_moe_router_sm100.cuh` | `TASK_GLM_MOE_ROUTER_SM100 = 401` | `"glm_moe_router"`（`:477-481`） | `glm_moe_router_layer:1165` |

**前端生产者（谁真的调这些 layer）**：`glm4_moe/builder.py:305`（`glm_moe_router_layer`）、
`inkling/builder.py:363/410/414/419/445/526/603`（sconv/attention/kv_store/moe_router）、
`dflash/builder.py:261/320/324`（norm_rope/kv_store/attention）、`deepseek_v3/builder.py:905/1501/1876/
1993/2220/2229`（sigmoid 路由 + softmax_gather/prob_scatter/prob_extract）、`qwen3/builder.py:649/658`
（sampling partial+reduce）、`gpt_oss/builder.py:345`（topk_softmax）。
即这批 task 的**每一个都被至少一个模型家族 builder 真实调用**（与 `docs/35`/`docs/36` 的「有实现无
生产者」形成对照），是 C3（融合范围）里**真正在跑**的薄 task。

---

## 2. 路由三兄弟：同一「sigmoid + bias + top-k + 归一化」的三种硬件/规模适配

三种路由都出自 HF 参考数学、产物 ABI 一致（`weights[num_rows, TOPK+NSHARED]` /
`indices[NUM_TOTAL, num_rows]` / `active[NUM_TOTAL+1]` 压缩活跃专家），但实现策略完全不同：

**§2.1 `topk_sigmoid_sm100.cuh`（364 行，DeepSeek 系，group-aware 向量化）**：文件头 `:44-58` 写明
七步（sigmoid→加 bias→**每组 top-2 求和→选 top-K 组→掩码非选中组→组内 top-K 专家→取原始未加 bias
的 sigmoid 作为权重→归一化缩放**）。硬约束是**专家数须为 2 的幂**（`static_assert` `:122-123`）与
**专家数须能被 warp 整除**（`THREADS_PER_ROW = ELTS_PER_ROW/VPT`，`:130`）；线程布局 `VPT=8、
THREADS_PER_ROW=32`，即一个 warp 一行、每线程持 8 个专家（`:55-58`）。组分数用 **`__shfl_xor_sync`
在 `THREADS_PER_GROUP=4` 个线程内做 top-2 归约**（`merge_top2`，`:65-73`、`:208-230`），再把每组的
`group_score` 广播给整 warp 选 top-K 组（`:232-255`）。专家维 top-K 用 subgroup argmax + **index
tie-breaker（小 index 优先）**（`:243-253`）；选中的**未加 bias 原始分**通过 `__shfl_sync` 从拥有该
专家的线程取回（`:298-306`）。末尾 `:350-361` 把活跃专家 id 压缩进 `mpk_active_expert_ids`（计数在
`[NUM_EXPERTS]`、`atomicAdd`）。

**§2.2 `glm_moe_router_sm100.cuh`（155 行，GLM-4.x，160 专家标量）**：`:33-36` 注释直接说明**不能复用
DeepSeek 的 sigmoid 版**——*"its vectorized layout statically requires a power-of-2 expert count, and
GLM has 160 routed experts"*，于是退化成**一线程一行**的标量实现（`:85-134`）。它把共享专家折进专家张量
的第 `NUM_ROUTED` 槽（`:120-125`，权重恒 1.0、不缩放），故下游 `moe_w13 / silu_mul / moe_w2 /
mul_sum_add` 管线不变（`:27-31`）。Phase 0/2 与 sigmoid 版同构（清结构 `:73-83`、压缩活跃 id
`:137-152`，注释强调**先快照到寄存器再写**避免 `active[pos]` 覆盖未读标记）；`grid (1,1,1)`（`:44`）。

**§2.3 `inkling_moe_router_sm100.cuh`（168 行，Inkling，256 routed + 2 shared）**：`logits` 是
258 维（`:21`），选择用 `sigmoid(前256)+bias` 的 top-6（`:101-119`），但**权重**是
`softmax(logsigmoid(选中 logits))`（`:125-146`，含 6 routed + 2 shared "gamma"）——与 GLM/DeepSeek
的归一化口径都不同，用一个 `inkling_logsigmoid`（`-softplus(-x)`，`:45-50`）实现。同样是
`grid (1,1,1)` + 一线程一行 + 末尾压缩活跃 id。

**差异小结**：三者映射到**同一组 `mul_sum_add`/`silu_mul` 下游**，说明 megakernel 把「模型家族差异」
全部**折平在路由 task 内部**（对上 `docs/27` 家族 builder 层的「折平」论断）——`TaskType` 只多三个，
下游 MoE 管线零改动。

---

## 3. 采样链：从 argmax 到 top-k/top-p/Gumbel，以及 softmax 概率的 gather/scatter

**§3.1 `argmax_sm100.cuh`（177 行，partial+reduce 两段）**：`argmax_partial_sm100_kernel`（`:83-122`）
每个 partial task 在自己的 vocab chunk 上求局部 argmax，`block_reduce_max_idx_sm100`（`:36-76`）用
命名屏障 ID **6** + 每 warp 一个 `[val,idx]` 槽做块内归约；`argmax_reduce_sm100_kernel`（`:124-175`）
合并 partial，(chunk_index, relative_index) **打包进一个 64-bit 整数**方便比较（`:144-154`），
末线程还原成全局 vocab id（`:161-166`）。注释 `:94-96` 明确把 lm_head 的 padding 行排除出归约。

**§3.2 `sampling_sm100.cuh`（419 行，temperature/top-k/top-p/Gumbel）**：文件头 `:16-32` 讲清两段式
设计——**partial** 保留本 chunk 的 `TOPK_MAX` 个最大 scaled logit + chunk max + `exp-sum`（`:152-265`），
**reduce** 用这些（max,sum）对**重建全词表 softmax 归一化**（`:336-360`），再按降序弹候选直到
top-k 数或 top-p 质量达阈（`:362-392`），最后用 **Gumbel-Max** 抽签（`:394-409`，注释 `:19-26` 说明
*"argmax of logit/T + Gumbel noise is an exact categorical draw, so no renormalization is needed"*）。
命名屏障 ID **7**（`SAMPLING_BAR_ID`，`:45`）；Gumbel 噪声用 Philox，**把一个 token 的子序列 key 到
token id 而非候选槽**（`:116-127`），使噪声与 vocab 如何分块无关；`sampling_philox_offset_for_row`
（`:131-150`）把 logits 行映射到「拥有它的请求」的 decode step（test_mode 回退）。

**§3.3 概率的 gather/scatter**：`softmax_gather_sm100.cuh`（115 行）**不 materialize 全概率分布**，
只算 `prob = exp(logit[token] - max - log_sum)`（`:18-29`），用 warp shuffle + `__shared__ float
smem_max[8]`/`smem_sum[8]`（`:62/88`，**上限 8 warp = block 256**）做两级归约——它是「投机解码里取
target token 概率」这一步骤（对上 `deepseek_v3/builder.py:1876/2220`）。
`prob_scatter_sm100.cuh`（78 行）把每 token 概率按**运行期 step 计数器**散射进 `buffer[batch, step]`
（`:42-47`），`prob_extract_task_impl`（`:54-76`）从该 buffer 按 offset 提取一段（`:68` 注释
*"+1 because verify starts at step+1"*）——二者都是投机验证窗口的簿记算子。

**§3.4 一个易混点**：`register_sampling_sm100_task`（`:2518`）发射的是 **common 库**的
`kernel::sampling_from_logits_kernel<256,4,bfloat16,int>`（`tasks/common/sampling.cuh:162`，整体一遍
Gumbel），而 `register_sampling_partial_sm100_task`（`:2553`）发射 `sampling_partial_sm100_kernel`、
`register_sampling_reduce_sm100_task`（`:2606`）发射 `sampling_reduce_sm100_kernel`；**reduce 与
`sampling_sm100` 共用 `TASK_SAMPLING_SM100=265`**（`:2543`/`:2645` 都 `register_task_variant(265,…)`，
partial 才是 279）。即「同一个 `TaskType` 号可对应多个 op 名/device 符号」，读 `TaskType` 表时不能
反推唯一 device 身体（对上 `docs/24`/`docs/29` 的计数口径坑）。

---

## 4. 位置编码与归一化：在 SMEM 上原地做，且常与 RoPE 融合

**§4.1 `norm_sm100.cuh`（142 行）**：`rms_norm_sm100` 在**调用方传入的 SMEM 视图 `InputSmem`** 上原地
做 per-(token,head) RMSNorm（平方和 warp-reduce + 跨 warp `reduce_smem`，`:64-97`），并可选**融合
partial RoPE**（`:109-136`）：`ROTARY_DIM`（GLM-4.6 是 128 中的 64）控制只旋转 `[0, ROTARY_DIM)`、
其余直通（`:27-30`）。它接受 `consumer/rotary` 两个命名屏障 ID（默认 6/7，`:25-26`），供调用方
（如 `attention_sm100`）复用同一组屏障。注意注释 `:48` *"Avoid sync divergence dead lock"* 与
`:112-113` *"Both barriers are reached uniformly by every participating thread"*——这正是
`docs/34` §3.5 记的「半 CTA 提前 return 会让 `__syncthreads` 在缺席 warp 上静默跳过」的正面样板。

**§4.2 `rotary_embedding_sm100.cuh`（80 行）**：独立的 SMEM 上 RoPE（NeoX 前后半配对旋转，`:60-73`），
同样用命名屏障 ID 6 保护**先读配对再写回**的两步（`:60`/`:72`）。`attention_sm100` 在 `qk_norm=false`
但 `rope=true` 时选它（见 §5.1）。

**§4.3 `dflash_norm_rope_sm100.cuh`（90 行）**：与上面「SMEM 原地」不同，这是**一线程一个
(token,head)**（`:32-34` 注释）、走寄存器的 per-head RMSNorm + NeoX RoPE（`:62-86`），输入输出都是
全局 `[N, NUM_HEADS, HEAD_DIM]`，`v` 不做（`:25`）。用于 DFlash draft 的 q/k 预处理。

**§4.4 `per_token_group_quantize_fp8.cuh`（165 行）**：per-token 分组（`GROUP_SIZE=128`）absmax 量化到
FP8，支持 **UE8M0 scale**（`encode_ue8m0`，`:33-37`）打包成 uint32（列主序 `[packed_k, batch]`，
`:140-161`）或 fp32 scale 行主序（`:119-128`）。**头号 megakernel 适配**在 `:87-92`：常驻内核里
`blockIdx.x` 是物理 worker id，若按它取 row 会「corrupt rows and break determinism」，故每个 task
**遍历所有 row**（`:92` 内层循环，注释标 *"MPK row-identity fix"*）——写虽冗余但确定。

---

## 5. 非 MLA attention 家族：paged / DFlash / Inkling

**§5.1 `attention_sm100.cuh`（750 行，`multitoken_paged_attention_sm100_task_impl`）**：这是 MPK 的
**通用 paged GQA attention**（`TASK_ATTN_SM100`，由 `paged_attention` op 登记）。要点：packed QKV 输入、
NHD paged KV（`:93-98`）；**m16n16k16 mma（ampere `mma_m16n16k16_bf16bf16bf32`）**做 QK^T 与 PV
（`:529`/`:637`），**不是 wgmma/UMMA**——即 SM100 块里也有走 mma.sync 的通用 attention；per-token
在线 softmax（`m_local`/`d`/`o`，`:309-643`）；**Q 与 O 复用同一 SMEM 段**（`:187-196`，注释说明这是
让 `max_num_batched_tokens` 能过 25、GQA 8:1 的关键，issue #740）；跨 warp 归约按 **MMA m-tile 分块**
（`:645-735`，m/d/o buffer 只存一个 tile，避免 smem 随 MAX_TOKENS 增长，issue #702）。它同时支持
**partial RoPE**（`ROTARY_DIM`，`:53-56`）、**sliding window**（`WINDOW_SIZE`，`:57-59`）、**attention
sink**（`sink_ptr`，`:80-82`，只进分母不进 value，`:724-729`）与 **`qk_norm` 融合**
（`:388-429` 调 `rms_norm_sm100`）。`register_paged_attention_sm100_task`（`task_register.cc:2326`）
的注释 `:2327-2340` 列出全部 12 个参数（含 sink 的第 8 输入、GLM partial RoPE 的 `rotary_dim`）。

**§5.2 DFlash（`dflash_attention_sm100.cuh` 568 行 + `dflash_norm_rope_sm100.cuh` +
`dflash_kv_store_sm100.cuh`）**：非因果 block attention，`B` 个 block query token **非因果**地看
`[context ++ block]` 全体 KV（`:24-61`）。两条实现由 `if constexpr (USE_MMA)` 选（`:197-198`）：
**scalar 参考**（一 warp 一个 (query,q_head)，`:70-175`）与 **mma 路径**（要求 `B*NUM_QO_PER_KV == 64`
行、每 warp 一个 16 行 m-tile、**warp-local softmax 无跨 warp 归约**，`:185-566`）；K/V 用 cp.async
双缓冲 64-key tile 流式（`:372-536`），sliding window 直接跳过整 tile（`:243-253`）。`dflash_kv_store`
按 slot 散射写 draft 自己的 paged cache（`:39-64`），`dflash_norm_rope` 见 §4.3。三者的 `TaskType`
296/297/298 由 `dflash/builder.py` 调用。

**§5.3 Inkling（`inkling_attention_sm100.cuh` 245 行 + `inkling_sconv_sm100.cuh` + `inkling_moe_router`）**：
GQA decode attention（`P=ctx_len` 单新 token），参考数学 `:22-31` 带**每 (head,距离) 的相对位置 bias**
（上游用一个小 GEMM 预计算 `bias[NUM_Q_HEADS, EXTENT]`）与**可选 log-scaling**（`tau = 1 + alpha *
ln(max((P+1)/n_floor,1))`，`:102-106`，乘 `q` 与 bias 即整个 pre-softmax 分数）；4 warp 在 key 维上
stripe（`:151-152`），每 warp 每 q head 一个在线 softmax、V 累加器按 lane 分布（4 dims/lane），
末尾**跨 warp 用 SMEM 合并**（`:199-241`）。`inkling_sconv`（94 行）是深度可分离短卷积（k=4）+ 残差、
`conv_state` 原地滑动更新（`:59-91`），是 Inkling 的「无 RoPE、短卷积」结构的 device 落地（对上
`docs/27`）。

---

## 6. 轻量逐元素/簿记算子

- **`elementwise_add_sm100.cuh`（44 行）**：`out = a + b`（残差），`row*STRIDE+col` 寻址（`:35-41`）。
- **`mul_sum_add_sm100.cuh`（51 行）**：MoE 组合 `out = residual + Σ_topk weight[i] * input[i]`
  （`:34-48`），是路由权重（§2）的下游消费者。
- **`tensor_init.cuh`（30 行）**：把张量填 `init_val`（默认 0），`TASK_TENSOR_INIT`。
- **`prob_scatter`/`prob_extract`** 见 §3.3。

这些算子的价值在**依赖图几何**而非算力：它们是「路由→专家 GEMM→组合→残差」这条 MoE 链上的胶水，
一个 `mul_sum_add` 就把 top-k 个专家的输出按权重求和并加残差（省掉 k 次独立加法 + 中间张量），是
`docs/04` §17「损失三：每融合一个算子省一趟全张量 HBM 往返」在 MoE 尾部的具体形态。

---

## 7. 判据与生产位置

- **判据**：本批 device task **只贡献 C3**（单 kernel 可表达、数值由对应单测 / HF 参考校验）。它们
  是**被 worker 调用的 `__device__` 身体**，不含自己的 `<<<>>>`、不做跨算子协调、没有固定角色
  worker/scheduler——C1/C2/C4/C5 全由运行层（`docs/06`）给，**与 `docs/11`/`docs/29`–`docs/36` 同口径**。
  单个 task 的「融合范围价值」体现在**把若干算子折进一段身体**（如 `attention_sm100` 融 qk_norm+partial
  RoPE+sliding-window+sink；`norm_sm100` 融 RMSNorm+RoPE；`mul_sum_add` 融 top-k 加权和+残差）。
- **生产位置**：与 `docs/35`/`docs/36` 的「有实现无生产者」相反，本批 task **每个都有 builder 调用**
  （§1 表），是真正在跑的薄 task。`attention_sm100` 是**通用** attention（非 MLA 模型走它），
  `dflash_*`/`inkling_*`/`glm_moe_router` 是四个家族各自的专用身体。
- **对 megakernel 专题的意义**：(a) 把「**少而重**」（FP8 GEMM/MLA，`docs/33`–`docs/36`）与「**多而薄**」
  （本节）两侧都摆在同一常驻内核里，印证 `docs/04` §8/§10/§11 的成本侧结论——寄存器/代码足迹由**最重**
  task 决定，而薄 task 贡献的是**依赖图关节数与事件扇入/扇出**；(b) `per_token_group_quantize` 的
  row-identity 修复、路由/采样的 `grid (1,1,1)` + 遍历全 row，是「一个 task = 一个 CTA」在
  `blockIdx.x == worker id` 约定下的**通用适配范式**（与 `docs/32` §5.3 的 `kNumSMs=1`、`docs/34`
  §3.5 的 TMEM-warp 一致）。

---

## 8. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100 在目标卡 H100/sm90 不可执行，未取任何 device 计数；
  「哪个 builder 调用哪个 layer」是**静态 `grep`** 结论，不构成运行时判定，也**不代表上游 bug**。
- **未展开的相邻层**：`sampling_from_logits_kernel`（`tasks/common/sampling.cuh` 的 CUB BlockReduce
  内部）只到「用 CUB + Gumbel」这层；`attention_sm100` 的 `mma_m16n16k16_bf16bf16bf32` 汇编（`tasks/
  ampere/mma.cuh`）未逐条；`inkling_moe_router` 的 `route_scale * global_scale` 语义只按注释转述。
- **命名屏障 ID 的口径张力**：本节 `argmax_sm100`（ID 6）、`sampling_sm100`（ID 7）、`norm_sm100`
  默认（6/7）、`inkling_attention`（6）、`attention_sm100`（6/7）、`rotary_embedding_sm100`（6）
  都使用 **6/7**；而 `docs/34` §3.5 转述源码注释称用户命名屏障 ID 须落 **≥8**（误用保留区 1–7 会
  留脏状态）。两者**并存**——本文只记录，**不判对错**（`bugfix.md` 不在快照里，无法交叉核对）。
- **`mla_prefill_host.cu` 的原型来源仍未走**：`docs/35` §9 提到的这一条（prefill 注释 `:201`）不在
  本文范围，留待后续。
- **未与生产端到端对跑**：SM100 路径本机不具备；本批 task 的数值正确性依赖各自单测 / HF 参考
  （源码注释给出的参考实现），本文未复算。

---

## 9. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/36` 同快照）。
- **路由**：`tasks/blackwell/topk_sigmoid_sm100.cuh`（364 行；七步数学 `:44-58`；布局 `:128-163`；
  组 top-2 `merge_top2:65-73`/`:208-230`；选组 `:232-255`；专家 argmax `:265-337`；归一化 `:339-346`；
  压缩 `:350-361`）；`glm_moe_router_sm100.cuh`（155 行；参考数学 `:18-44`；清 `:73-83`；一行一线程
  `:85-134`；共享专家 `:120-125`；压缩 `:137-152`）；`inkling_moe_router_sm100.cuh`（168 行；
  `logsigmoid:45-50`；top-6 `:101-119`；softmax 权重 `:125-146`）。
- **采样/统计**：`argmax_sm100.cuh`（177 行；块归约 `:36-76`；partial `:83-122`；reduce `:124-175`）；
  `sampling_sm100.cuh`（419 行；Gumbel `:119-127`；partial `:152-265`；reduce `:275-417`）；
  `softmax_gather_sm100.cuh`（115 行）；`prob_scatter_sm100.cuh`（78 行）；
  `tasks/common/sampling.cuh:162`（`sampling_from_logits_kernel`）。
- **位置/归一化/量化**：`norm_sm100.cuh`（142 行；RMSNorm `:64-97`；融合 RoPE `:109-136`）；
  `rotary_embedding_sm100.cuh`（80 行）；`dflash_norm_rope_sm100.cuh`（90 行）；
  `per_token_group_quantize_fp8.cuh`（165 行；UE8M0 `:33-37`；row-identity 修复 `:87-92`；打包
  `:140-161`）。
- **attention**：`attention_sm100.cuh`（750 行；SMEM 复用（Q=O）`:187-196`；partial RoPE/sliding/sink
  `:53-82`；QK `:493-531`；PV `:624-639`；跨 warp 归约 `:645-735`）；`dflash_attention_sm100.cuh`
  （568 行；`USE_MMA` 门槛 `:195-198`；ref `:70-175`；mma `:185-566`）；`dflash_kv_store_sm100.cuh`
  （66 行）；`inkling_attention_sm100.cuh`（245 行；参考数学 `:22-31`；在线 softmax `:151-197`；
  跨 warp 合并 `:199-241`）；`inkling_sconv_sm100.cuh`（94 行）。
- **小算子**：`elementwise_add_sm100.cuh`（44 行）、`mul_sum_add_sm100.cuh`（51 行）、
  `tensor_init.cuh`（30 行）。
- **生产布线**：`include/mirage/persistent_kernel/runtime_header.h:144-152/162-175/191-195/210-219`
  （`TaskType`）；`src/kernel/graph.cc:457-491/644-756/841-846`（op→`task_config`）；
  `src/kernel/task_register.cc`（`register_paged_attention_sm100_task:2326`、
  `register_argmax_*_sm100:2440/2483`、`register_sampling_sm100_task:2518`、
  `register_sampling_partial_sm100_task:2553`、`register_sampling_reduce_sm100_task:2606`、
  `register_tensor_init_task:2666`、`register_elementwise_add_sm100_task:2699`、
  `register_softmax_gather_sm100_task:2731`、`register_prob_extract_sm100_task:2808`、
  `register_prob_scatter_sm100_task:2839`、`register_moe_topk_softmax_sm100_task:2866`、
  `register_moe_topk_sigmoid_sm100_task:2935`、`register_moe_mul_sum_add_sm100_task:3554`、
  `register_dflash_*_sm100_task:126/185/222`、`register_glm_moe_router_sm100_task:258`、
  `register_inkling_sconv/moe_router/attention_sm100_task:317/430/367`）；
  `python/mirage/mpk/persistent_kernel.py`（layer 坐标见 §1 表）。
- **家族 builder**：`python/mirage/mpk/models/{glm4_moe,inkling,dflash,deepseek_v3,qwen3,gpt_oss}/
  builder.py`（调用点见 §1）。
