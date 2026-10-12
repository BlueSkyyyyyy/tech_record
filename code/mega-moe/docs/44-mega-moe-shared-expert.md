# Mega MoE 的共享专家（shared expert）分支

> 本专题此前从「宿主编排」到「主 kernel 逐段」再到「反向 / 相关 mega-kernel / 测试栈」，
> 但一直漏了 Mega MoE 里一条**与路由专家并行、且贯穿 host→调度→kernel→combine 全链路**的分支：
> **共享专家（shared expert）**。DeepSeek 的 MoE 每层在 top-k 路由专家之外，还有一个**每个 token
> 恒定激活**的共享专家。Mega MoE 把它**融进同一个 mega-kernel**：共享专家不参与 dispatch（直接吃
> 本 rank 的原始 token），它有独立的 `BlockPhase::SharedLinear1/SharedLinear2` 两条调度队列，
> L1 结果写进 `shared_l2_acts`、L2 结果写进 combine buffer 的一个**额外槽**，最终由 combine 与路由
> 专家输出**在 fp32 里一起相加**（且共享项**不带 top-k 权重**）。
>
> **主 kernel 是 SM100 专属，本机 H100（sm90）跑不了端到端**；但本篇覆盖的四层里，host 编排层、
> 调度器公式与任务枚举器、UTCCP 的 SF 索引映射、以及共享专家的**数学**都与 UMMA 无关——
> 前三层直接对拍真实 binding / 真实设备代码，第四层用真实 SM90 BF16 GEMM kernel 跑出数字。
> 行号均已对照 DeepGEMM `057ca59`（Public release 26/09/30）。

## 0. 为什么共享专家值得单独讲

DeepSeek-V3/R1 的 MoE 层 `y = Σ_{i∈topk} w_i · Expert_i(x) + SharedExpert(x)`：路由专家稀疏激活，
共享专家**对所有 token 恒定全量激活**（DeepSeek-V3 用 1 个，中间维 = 路由专家中间维，
即本专题实验里的 `num_shared_experts=1`；代码把它参数化成「更宽的单个 MLP」`E_s·I`）。

这么设计对 kernel 有三个结构性后果，也是本篇主线：

1. **共享专家不依赖 dispatch**：它的输入就是原始 token `x`（`shared_l1_acts ≡ x`），所以在
   mega-kernel 里可以**在 dispatch 通信完成之前**就开始算，天然与通信重叠（`scheduler:445-452`）。
2. **共享专家的 L1 沿 N 轴变宽、L2 沿 K 轴变宽**：L1 的 gate+up 输出维是 `2·E_s·I`（N 轴），
   L2 的输入维是 `E_s·I`（K 轴）。这决定了两条共享队列的任务数与形状（`scheduler:182-185`）。
3. **共享专家的结果不是另一条 GEMM，而是 combine 的一个 bias 项**：它写进 combine buffer 的
   `num_topk` 号槽（第 `kNumTopk` 槽），被 combine 的 fp32 累加器**直接加一次**（无权重），
   见 `kernel:1287-1288`、`:1433`、`:1449-1450` 与测试参考 `tests/test_mega_moe.py:277-302,333`。

## 1. Host 契约：共享专家的形状、类型与检查

共享专家的 host 入口在 `csrc/apis/mega_moe.hpp`。`fp8_fp4_mega_moe` 的共享专家段是 `:206-228`：

- 共享权重是**两个 2D 张量**（不是 per-expert 的 3D）：`shared_l1_weights` 形状 `[S·2, H]`、
  `shared_l2_weights` 形状 `[H, S]`，其中 `S = shared_intermediate_hidden`（`:211-212`）。
- `num_shared_experts = shared_intermediate_hidden / intermediate_hidden`（`:216`），并断言
  `shared_intermediate_hidden % intermediate_hidden == 0`（`:218`）——即共享专家只能按
  **路由专家中间维的整数倍**加宽（`E_s ∈ {0,1,2,…}`，0 表示关闭）。
- 形状自洽：`shared_l1_n == S·2`（`:219`）、`shared_l1_k == hidden`（`:220`）、
  `shared_l2_n == hidden`（`:221`）。
- **共享权重恒为 FP8 E4M3**（`:222-223` 断言 `torch::kFloat8_e4m3fn`），即使路由权重是 FP4——
  这与 kernel 里 `shared_b_dtype_t = float_e4m3_t`（`kernel:154`）一致（见 §4）。
- 共享权重的 SF 也单独检查（`check_sf_layout`，`:224-227`），且 **`num_experts_per_rank` 传 `nullopt`**
  （因为共享权重没有专家维）。
- BF16 变体 `bf16_mega_moe` 的对应段在 `:322-340`（同样的 2D 形状与整除关系，无 SF）。

缓冲区大小与 view 这一层，`get_symm_buffer_size_for_mega_moe(..., num_shared_experts=0)`（`:35-40`）
先算 `shared_intermediate_hidden = intermediate_hidden · num_shared_experts`（`:53`），再把它交给
`layout::MegaMoEBuffer`（`:80-85`）。切片出的 12 个 view 里，共享相关的四个是（`:114-128`）：

- `shared_l1_acts` 就是 `x`（`= x`，`:114`）——共享专家直接吃原始 token。
- `shared_l1_acts_sf`：只读的 SF 页（K-major `x_sf` 经 UTCCP 重排后的共享 L1 版本），形状
  `[get_num_max_shared_sf_tokens(Tmax), hidden/128]`、列主序 stride `(1, num_max_shared_sf_tokens)`（`:115-119`）。
- `shared_l2_acts`：形状 `[Tmax, S]`（`:120-123`）。
- `shared_l2_acts_sf`：形状 `[get_num_max_shared_sf_tokens(Tmax), S/128]`（`:124-128`）。

### 1.1 `MegaMoEBuffer` 里的共享缓冲与「链式无空洞」分配

`deep_gemm/include/deep_gemm/layout/mega_moe.cuh` 的 `MegaMoEBuffer` 构造函数 `:367-450` 把共享
缓冲插在 input 与 routed ring 之间：

- `shared_intermediate_hidden = intermediate_hidden · num_shared_experts`（`:384`）；
- 共享 SF 容量 `num_max_shared_sf_tokens = get_num_max_shared_sf_tokens(Tmax)`（`:385`，仅 `with_sf`）；
- `shared_l1_token_buffer = input_token_buffer`（`:415`，**别名复用**，不额外分配）；
- `shared_l1_sf_buffer`（`:416-418`）、`shared_l2_token_buffer`（`:419-421`）、`shared_l2_sf_buffer`（`:422-424`），
  每块的 base 都是前一块的 `get_end_ptr()`，**链式无空洞**；当 `num_shared_experts==0` 时它们的
  token 数 / SF 行数取 0，占 0 字节（于是后面 ring 的起点退化为 `input_topk_weights` 之后，`:429-431`）。
- combine buffer 的最外层维是 `num_topk + (num_shared_experts > 0 ? 1 : 0)`（`:448`）——那个 `+1`
  就是共享专家的槽位。

`get_num_max_shared_sf_tokens`（`:36-40`）的值是 `ceil_div(Tmax, kMinCandidateBlockM=8) · 128`：
共享专家的 SF 池按**最小候选 BLOCK_M=8** 分块、每块补齐到 128 行，取所有候选 BLOCK_M 的最坏情形。

`Workspace`（`:93-126`）里与共享相关的信号是 `shared_l1_task_count[2]` / `shared_l2_task_count[2]`
（每域一个任务计数器，`:69-70`）与 `shared_l2_full_count[kNumMaxSharedL2Blocks]`（`:89`，容量
`1<<15 = 32768`，`:59`）；`num_shared_l2_pool_blocks = ceil_div(Tmax, kMinCandidateBlockM)`（`:117`，
断言 `<= 32768`，`:125`）——即共享 L2 的信号数组**每池块一个计数器**。

## 2. 调度：共享专家的两条相位队列

`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`。`BlockPhase` 枚举（`:83-89`）里共享占据两个值：

```
Linear1 = 1, Linear2 = 2, SharedLinear1 = 3, SharedLinear2 = 4
```

`TaskInfo::is_shared()`（`:131-133`）用 `block_phase > BlockPhase::Linear2` 判定，而且
`sizeof(TaskInfo<true>) == sizeof(TaskInfo<false>)`（`:136`）——**共享与非共享共用一个槽结构体**，
只多一个编译期标志。注意 `TaskInfo` 本身没有「是否共享」字段：相位就是标志。

### 2.1 形状关系：L1 沿 N、L2 沿 K

`MegaMoEScheduler` 的模板参数 `kNumSharedExperts`（`:174`）推导出（`:181-185`）：

```
kHasShared        = kNumSharedExperts > 0
SHARED_L1_SHAPE_N = L1_SHAPE_N · kNumSharedExperts      // = 2·I·E_s  （N 轴）
SHARED_L1_SHAPE_K = L1_SHAPE_K                          // = H
SHARED_L2_SHAPE_N = L2_SHAPE_N                          // = H
SHARED_L2_SHAPE_K = L2_SHAPE_K · kNumSharedExperts      // = I·E_s  （K 轴）
```

**为什么 L1 在 N、L2 在 K**：共享专家是一个有 `E_s·I` 个中间通道的前馈层。它的 L1（gate+up）
输出自然是 `2·E_s·I`（N 轴变宽）；而它的 L2 把 `E_s·I` 个中间激活压回 `H`，所以**收缩维 K**
是 `E_s·I`。H100 实测（Part B）逐条核过这三条关系，以及静态断言要求
`SHARED_* % (BLOCK_N·2)==0`、`% BLOCK_K==0`（`:192-195`）。

### 2.2 `shared_mainloop`：一条与路由无关的动态队列

`:421-436` 是共享专家专用的任务循环：

```
template <BlockPhase kBlockPhase, uint32_t kShapeN, uint32_t kShapeK>
void shared_mainloop(num_tokens, lane_idx, task_count_ptr) {
    constexpr uint32_t kNumNClusters = kShapeN / BLOCK_N / 2;
    const uint32_t num_m_blocks = ceil_div(num_tokens, BLOCK_M);
    while (true) {
        task_info_empty_barriers[sched_stage_idx].wait(...);      // 等消费者释放槽
        if (not get_next_task_idx<kNumNClusters>(task_count_ptr, num_m_blocks, m, n)) break;
        const uint32_t valid_m = min(num_tokens - m*BLOCK_M, BLOCK_M);
        publish_task(task_info_t(kBlockPhase, 0, m, n, /*pool=*/m, valid_m, kShapeN, kShapeK), lane_idx);
    }
}
```

要点：`local_expert_idx=0`、`pool_block_idx=m_block_idx`（`:434`）——共享专家没有专家维，
池块号就等于 M 块号；`valid_m` 按 `num_tokens` 裁剪尾块。任务通过 `get_next_task_idx` 从
**每域一个的原子计数器**上动态认领（`:344-370`），与路由任务用同一套机制（含局部域切分）。

### 2.3 「L2 提前 vs 推迟」与「共享 L1 与 dispatch 重叠」

`mainloop`（`:438-476`）把两件事串起来：

```
sm_locality_domain_idx = ...                                   // :440
kNumSharedL2Clusters   = SHARED_L2_SHAPE_N / BLOCK_N / 2       // :443
is_shared_l2_early     = ceil_div(num_tokens,BLOCK_M)*kNumSharedL2Clusters <= kNumSMs/2   // :444
if (kHasShared) {
    shared_mainloop<SharedLinear1, ...>();                     // :447-448  共享 L1 先发
    if (is_shared_l2_early)
        shared_mainloop<SharedLinear2, ...>();                 // :449-451  共享 L2 也提前
}
fetch_expert_recv_count();                                     // :455  这里才等 dispatch 结果
... routed L1/L2 交错调度 ...                                   // :457-465
if (kHasShared and not is_shared_l2_early)
    shared_mainloop<SharedLinear2, ...>();                     // :467-471  否则放尾
```

**共享 L1 无条件在 `fetch_expert_recv_count()` 之前发布**（`:446-448`），因为共享专家的激活
`shared_l1_acts ≡ x` 已在本地、不依赖 dispatch；这正好把共享 L1 的计算填进 dispatch 的网络等待窗口。
**共享 L2 的时机是一个启发式**（`:443-444`）：若 `ceil_div(num_tokens, BLOCK_M)·kNumSharedL2Clusters`
不超过一半 SM（即一轮就能塞进 dispatch 的空档），就提前（`is_shared_l2_early=true`），否则推迟到
路由任务之后（`:467-471`）。这个判据随 `BLOCK_M` 变（Part B 的表里能看到同一 `num_tokens` 下
`BLOCK_M` 越小越可能 `False`）。

H100 上我们**直接编译运行了真实设备代码** `get_next_task_idx<kNumClusters>`（Part A），验证
共享 L1（N 簇 4）与共享 L2（N 簇 8）的任务集合 `{(m,n)}` 在每域一个计数器下**恰好枚举一次**
（去重后大小 = 期望 = `num_m_blocks·N簇`），且每域只领走自己那半簇。同时把
`get_num_l1_warmup_waves`/`get_num_max_live_pool_blocks`（`:17-80`，host+device constexpr）与
`is_shared_l2_early`（`:443-444`）当**设备常量**复算，与 host 侧独立复刻逐项一致（Part B/C）。

## 3. Kernel 主循环里共享专家如何「借用」路由专家的机器

`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`。共享专家**复用整套 TMA/MMA/
epilogue/combine 骨架**，只在少数点按 `is_shared()` 分流。逐点如下（**SM100 专属，仅分析**）：

**模板与类型**：`kNumSharedExperts`（`:24`）、`kHasShared`（`:39`）、`SHARED_L2_SHAPE_K`（`:44`）、
`kNumSharedSFTokens`（`:53`）。`with_sf=true` 恒成立（`:131`）。数据类型注释（`:152`）：
「activations and shared weights are FP8 (e4m3); routed weights may be FP8 or FP4」——
`a_dtype_t = e4m3`（`:153`）、`shared_b_dtype_t = e4m3`（`:154`），且
`DG_STATIC_ASSERT(sizeof(weight_dtype_t) == sizeof(shared_b_dtype_t))`（`:839`）保证两者在 SMEM 里
寻址一致（FP4 的 `float_e2m1_unpacksmem_t` 也是 1 字节/元素，故大小相等）。

**TMA 选择**：A/SFA 的 tensor map 按相位四选一（`:697-704`），B/SFB 同理（`:763-770`）；
共享相位用 `tensor_map_shared_l1_acts/_sf/_weights/_sf` 等（`:70-78` 的 kernel 入参）。权重侧
`expert_idx = is_shared() ? 0 : local_expert_idx`（`:779`）——共享专家只有一个「专家组 0」；
共享权重的 SF 基址**不带 per-expert 偏移**（`sfb_k_idx = is_shared() ? k_block_idx·(BLOCK_K/128) : ...`，`:788`）。
共享权重走 `shared_b_dtype_t`/`tma::copy_nd<..., shared_b_dtype_t, 4>`（`:793`），路由权重走
`weight_dtype_t`（`:803`）。

**MMA**：leader CTA 的同一个 warp 发 UMMA，但**准备两个 instruction descriptor**：
`routed_instr_desc`（`weight_dtype_t`，`:827-831`）与 `shared_instr_desc`（`shared_b_dtype_t`，`:832-836`），
每任务按 `is_shared()` 选（`:859`）。

**依赖**：路由 L2 等 `l2_full_mask` 的位图（`:723-726`）；共享 L2 等的是
`shared_l2_full_count`（`:717-720`），期望值 `(SHARED_L2_SHAPE_K / BLOCK_N)·2`——即共享 L1 的
全部 N 块都写完。

**L1 epilogue**（`:1058-1253`）：

- 相位判定放宽为 `Linear1 or SharedLinear1`（`:1058`）；只有路由相位才等 L2 空（`:1059-1064`）。
- **共享专家的 SwiGLU 输出不带 top-k 权重**：`stored_cached_weight` 初值 1.0（`:1068`），只有在
  `not is_shared()` 时才会从 `l1_topk_weights_buffer` 重载（`:1095-1100`）。所以共享项权重恒为 1。
- L1 输出落点：`is_shared() ? tensor_map_shared_l1_output : tensor_map_l1_output`（`:1225`）——
  共享 L1 直接写 `shared_l2_acts`（即共享专家的 L2 输入缓冲），**不进 routed ring**。
- 完成通知：共享写 `red_add_rel(shared_l2_full_count[pool_block_idx], 1)`（`:1242-1244`）；
  路由则翻 `l2_full_mask` 位并增 `l1_empty_count`（`:1246-1250`）。
- 共享 SF 写 `shared_l2_sf_buffer`（而非 `l2_sf_buffer`），行 stride 用 `kNumSharedSFTokens`
  而非 `kNumSFRingTokens`（`:1199-1201`）。

**L2 epilogue → combine**（`:1281-1352`）：共享专家的 L2 输出经 NVLink 直写 combine buffer，
其**目标槽位固定为 `kNumTopk`**、目标 rank/ token 是本地：

```
cached_src_metadata[j] = is_shared()
    ? TokenSrcMetadata(sym_buffer.rank_idx, pool_m_idx + m_idx_in_block, kNumTopk)   // :1287-1288
    : *workspace.get_token_src_metadata_ptr(pool_m_idx + m_idx_in_block);            // :1289
```

**Combine**（`:1365-1519`）：`kNumTopk + (kNumSharedExperts>0 ? 1 : 0) <= 32`（`:1386`，保证一个 warp
能塞下所有槽）。每个 token 的槽索引里，共享槽就是 `kNumTopk`（`:1431-1433`），且 combine **先取
共享槽**（`slot_idx = (mask >> kNumTopk) ? kNumTopk : ffs(mask)-1`，`:1449-1450`），在同一个 fp32
累加器里与路由项相加（`:1472-1490`），最后 bf16 写回 `y`（`:1492-1519`）。**共享项既不乘权重、
也不乘任何路由概率，只加一次**。

## 4. 测试参考：共享专家在 baseline 里到底是什么

`tests/test_mega_moe.py` 给出了权威的参考语义（**架构无关**）：

- 共享权重构造（`:146-151`）：`shared_l1_weights = randn((S·2, H))`、`shared_l2_weights = randn((H, S))`，
  `S = shared_intermediate_hidden = intermediate_hidden·num_shared_experts`。
- FP8 模式下先量化、再做 gate/up 交织与 UTCCP 转置（`:167-176`），其中 L1 的共享权重会被
  `transform_weights_for_mega_moe` 交织（`deep_gemm/mega/__init__.py:112-164`）。
- `get_baseline_shared_bias`（`:277-302`）：`l1_out = x @ W1sᵀ`（T, `S·2`）→ SwiGLU → `l2_in`（T, S）
  → `y = l2_in @ W2sᵀ`（T, H）；BF16 走 `bf16_gemm_nt`（`:283-291`），FP8 走
  `fp8_gemm_nt(..., recipe=(1,1,32), disable_ue8m0_cast=True)`（`:292-301`）。
- 最后把这个 `y` 作为 **`combine` 的 `bias`**（`:333`）：`ep_buffer.combine(l2_y, handle, bias=shared_y)`。
  这正是 kernel 里「共享结果写进 combine 的额外槽、被累加一次」的 host 对照。
- 性能口径（`:368-395`）：共享专家的 FLOPs `= 2·T·(H·S + …)`，HBM 字节单独计（含共享 L1/L2 权重、
  共享 L1/L2 激活读写、`y` 写回）。

**UTCCP 的 SF 行索引映射**在 device 与 test 里各写了一遍，本篇逐点核过其等价性：

- device `transform_sf_token_idx`（`kernel:145-149`）：
  `t/BLOCK_M·align(BLOCK_M,128) + (idx & ~127) + (idx&31)·4 + ((idx>>5)&3)`（`idx = t % BLOCK_M`）。
- test `_to_shared_mega_moe_sf_layout`（`:36-50`）：对 `(block_idx, m_idx)` 写
  `block_idx·align(block_m,128) + (m_idx//128)·128 + (m_idx%32)·4 + (m_idx%128)//32`。
  两者内层完全一致（`(m_idx//128)·128 = m_idx & ~127`，`(m_idx%128)//32 = (m_idx>>5)&3`）。

H100 实测（Part C）：对 8 个候选 BLOCK_M、每组 5 个 M 块，device 映射与 host 参考**逐点 0 不符**，
且都映射成到 `[0, num_m_blocks·align(block_m,128))` 的**单射**（正是 UTCCP 4×32 转置的特征）。

## 5. H100 实测汇总

> 所有数字来自 `harness/exp_mega_moe_shared_expert.sh`（真实设备代码 + 真实 SM90 kernel），
> 完整输出见 `harness/exp_mega_moe_shared_expert.out.txt`。主 kernel 端到端仍需 SM100。

**CUDA 真实设备代码**（`src/mega_moe_shared_schedule.cu`，sm90 编译运行）：

- **[A] 共享任务枚举**：`SharedLinear1`（`SHARED_SHAPE_N=1024`，N 簇 4）发射 32 个任务、期望 32、
  去重 32、覆盖 OK；`SharedLinear2`（`SHARED_SHAPE_N=2048`，N 簇 8）发射/期望/去重 64、覆盖 OK；
  每域计数器下界（16,16）/（32,32）满足。**即共享 `shared_mainloop` 的枚举器在每域一个计数器下
  恰好覆盖所有 `(m_block, n_cluster)` 一次**。
- **[B] constexpr 公式**：`get_num_max_live_pool_blocks`（`:48-80`）与 `get_num_l1_warmup_waves`
  在 `num_total_m_blocks ∈ {8,32,128,512}` 下 code vs host 复刻**逐项一致**（live = 8/32/114/306，
  warmup 恒 2）。
- **[C] `is_shared_l2_early`**：`BLOCK_M=128, H=7168, num_sms=132`（L2 簇 8）下，
  `num_tokens=256/1024 → True`，`num_tokens=8192 → False`。

**Python**（host 对拍 + 数值）：

- **[A] host view**：5 组配置（shared ∈ {0,1,2}，fp8/bf16）逐字段对拍 **ALL OK**：
  `shared_l1_acts is x`；`shared_l1/l2_acts_sf` 行数 = `ceil_div(Tmax,8)·128`、列主序 stride
  `(1, num_max_shared_sf_tokens)`；`shared_l2_acts` 形状 `(Tmax, E_s·I)`；
  共享 SF 的 K 维 = `H/128`（L1）/`S/128`（L2）；`shared==0` 时三个共享 view 全部 undefined。
- **[B] 形状关系/决策**：`SHARED_L1_SHAPE_N=L1_SHAPE_N·E_s`、`SHARED_L2_SHAPE_K=L2_SHAPE_K·E_s`
  三组形状 OK；`is_shared_l2_early` 随 BLOCK_M 变化（`num_tokens=256`：BLOCK_M=64→False，
  128/240→True）；共享 L2 信号容量 `ceil_div(Tmax,8) ≤ 32768`（Tmax 到 262144 仍成立）。
- **[C] UTCCP 映射**：8 个 BLOCK_M **逐点 0 不符且单射**。
- **[D] 数值**（真实 `deep_gemm.bf16_gemm_nt`）：`T=1024,H=1024,I=512,E_s=2`，共享专家前向
  （`x@W1sᵀ → SwiGLU → @W2sᵀ`）vs **fp64 参考 relerr 3.447e-3**（bf16 底噪）；
  combine `y = Σ_k w_k·routed_k + shared`（共享项不加权、只加一次）vs fp64 **relerr 3.448e-3**；
  **反例**（把共享项重复 `topk=2` 次）relerr 8.747e-1——坐实「共享项只加一次」。

## 6. 小结：共享专家分支的「为什么」

一句话：**共享专家是 Mega MoE 里唯一一条「不走 dispatch、却走完整 L1/SwiGLU/L2 计算 + combine」
的通路**，它被设计成三处「借道」：

1. **借道 input buffer**：`shared_l1_acts ≡ x`，零拷贝复用激活（`layout:415`）。
2. **借道调度器**：加两个相位 `SharedLinear1/2` 和两个每域计数器，让共享 L1 填进 dispatch 空档、
   共享 L2 按 `is_shared_l2_early` 决定提前或补尾（`scheduler:443-471`）。
3. **借道 combine**：输出写进 combine buffer 的 `kNumTopk` 槽、被 fp32 累加器**无权重加一次**
   （`kernel:1287-1288,1449-1450`），从而「共享专家 + 路由专家」的求和**与 combine 融合、零额外 pass**。

它沿 N（L1）/ K（L2）变宽的中间维、以及恒 FP8 的共享权重（路由可 FP4），都是 DeepSeek 实际配置
在 kernel 里的映射。至此 Mega MoE 的**路由专家 + 共享专家 + combine** 三条数据通路全部讲清。

## 参考

- DeepGEMM（`057ca59`，Public release 26/09/30）：`csrc/apis/mega_moe.hpp`、
  `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、
  `deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、`deep_gemm/mega/__init__.py`、
  `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`、`tests/test_mega_moe.py`。
- 仓库：https://github.com/deepseek-ai/DeepGEMM
- 本专题：`docs/02`（整体数据流）、`docs/03`（block config）、`docs/09`（mbarrier）、
  `docs/10`（MMA/epilogue，UTCCP/SwiGLU）、`docs/11`（dispatch/combine）、
  `docs/20`（BF16 Mega MoE）、`docs/42`（host 编排与对称缓冲区布局）。
