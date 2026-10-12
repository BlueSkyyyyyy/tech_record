# `sm100_fp8_fp4_mega_moe.cuh` 逐段精读（三）：tcgen05 MMA 与 L1/L2 epilogue

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 7 篇，对应路线图 **P3-3**。
> P3 拆四段：模板/block config（`docs/03`）→ warp specialization & mbarrier（`docs/09`）→
> **tcgen05 / UTCCP / SwiGLU / epilogue（本文）** → dispatch & combine（P3-4，待做）。
>
> **重要**：本文讨论的 kernel 是 **SM100（Blackwell）专属**（tcgen05 MMA、2-CTA cluster、
> UTCCP、UE8M0 block scale、FP8×FP4、TMEM），本机 H100（sm90）**不跑端到端**，因此本节以
> **源码精读为主线**；epilogue 的**数值原语**（UE8M0 scale 推导、幂次量化、in-place SwiGLU）
> 在 H100 上用位运算逐位复现并给真实数字（§5）。
>
> 源码坐标（`~/github/DeepGEMM`，2026-09-30 公开版 commit `057ca59`）：
> - 主体 kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）
> - UMMA/descriptor 封装：`deep_gemm/include/deep_gemm/mma/sm100.cuh`
> - tcgen05 PTX：`deep_gemm/include/deep_gemm/ptx/tcgen05.cuh`
> - UE8M0 / 打包：`deep_gemm/include/deep_gemm/common/math.cuh`、`.../packing.cuh`
> - host 配置：`csrc/jit_kernels/heuristics/mega_moe.hpp`
>
> 以下行号均逐条核对过。

## 1. MMA 计算本体：`warp_idx == kNumDispatchWarps + 2`（`:819-938`）

上一节（`docs/09`）讲了 mbarrier 流水；这一节进入「谁真正发 MMA」。整个 kernel 只有
**一个 warp**（非 epilogue 4 个 warp 中的第 2 个，`:819`）在 **leader CTA**（`is_leader_cta`，`:824`）
里发出所有 UMMA 指令，另一个 CTA 什么都不发——这是 2-CTA `/2SM` MMA 的约定：一条
`tcgen05.mma.cta_group::2` 同时驱动两个 SM 的 tensor core。

### 1.1 形状约定：A/B 互换 + 2-CTA（`:158-168`）

```cpp
constexpr uint32_t LAYOUT_AD_M = 128;
constexpr uint32_t UMMA_M = LAYOUT_AD_M * 2;   // 256
constexpr uint32_t UMMA_N = BLOCK_M;           // Swap AB
constexpr uint32_t UMMA_BLOCK_K = 128;
constexpr uint32_t UMMA_K = 32;
constexpr uint32_t LOAD_BLOCK_M = BLOCK_M / 2; // Multicast on A
constexpr uint32_t LOAD_BLOCK_N = BLOCK_N;
```

这里有个反直觉点：**A/B 被刻意交换**。MoE 的线性层本质是 `Y[M,N] = X[M,K] @ W[N,K]^T`，
token 维是 M、输出维是 N。但 kernel 里 `UMMA_M=256`（= 2×128）、`UMMA_N=BLOCK_M`（token 维），
也就是说 **UMMA 的 M 轴对应「输出特征维」，N 轴对应「token 维」**——A 是权重、B 是激活。为什么：

- tcgen05 的 N 轴最大值是 256，且要求 16/8 对齐；token 分块 `BLOCK_M`（≤240）天然落在 N 轴，
  而特征维（128 的整数倍）放在 M 轴可被 2-CTA 拼成 256；
- `UMMA_N=BLOCK_M` 还让 `valid_m` 直接映射成 N 方向的有效长度（末块不满），
  于是 §1.2 能按实际 token 数动态缩 N；
- 激活是「一份被 2 个 CTA 共享」的（`LOAD_BLOCK_M = BLOCK_M/2`，A/B 交换后激活即 N 侧……，
  实际 2-CTA multicast 加载的是 128 宽的特征块），cluster 里两个 CTA 各算一半特征、共享 token 激活。

### 1.2 instruction descriptor：routed vs shared（`:825-850`）

```cpp
auto routed_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        weight_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
auto shared_instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        shared_b_dtype_t, a_dtype_t, float, cutlass::float_ue8m0_t, UMMA_M, UMMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();
```

两个 descriptor 唯一的区别是 **A 侧（权重）dtype**：routed 权重可以是 FP8 也可以是
**FP4**（`weight_dtype_t`，`:152-156` 断言只允许 `float_e4m3_t` 或 `float_e2m1_unpacksmem_t`），
shared（共享专家）权重恒为 FP8（`shared_b_dtype_t = float_e4m3_t`，`:154`）。两者 size 相同
（`:839` 静态断言），所以 **SMEM 地址算式完全通用**，只在 issue 时选对应 descriptor（`:859`）。
`block_scaled` + `float_ue8m0_t` 就是 MX 格式：每 32 个 K 元素一个 UE8M0 幂次 scale。

下面三件事在循环外只算一次：
- `sf_desc = make_sf_desc(nullptr)`（`:837`，`mma/sm100.cuh:42-49`）：UTCCP 用的 SMEM 描述符，
  固定 `SWIZZLE_NONE`，`{SBO,LBO}={8*16,0}`——「8×128bit 原子，K 方向只有一个原子」；
- `a_desc/b_desc = make_umma_desc<Major::K, ..., 128B swizzle>(smem_a[0], 0, 0)`（`:841-842`）：
  只取 **base `.lo`**（`:843-844`），后续用整数加法拼 stage / K 偏移，避免每次重建描述符。

### 1.3 每条 task 的 MMA 主循环（`:852-931`）

```cpp
while (scheduler.get_next_task(task_info)) {
    const auto num_k_blocks = task_info.shape_k / BLOCK_K;
    auto& instr_desc = task_info.is_shared() ? shared_instr_desc : routed_instr_desc;
    mma::sm100::update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m());
    const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
    const auto accum_phase    = (current_iter_idx ++ / kNumEpilogueStages) & 1;
    shared_storage.tmem_empty_barriers[accum_stage_idx].wait(accum_phase ^ 1);
    ...
```

逐点解释「为什么」：

1. **动态 N**（`:858-860`）：`update_instr_desc_with_umma_n` 把 `desc.n_dim_ = umma_n >> 3`
   （`mma/sm100.cuh:150-153`）。`get_umma_aligned_valid_m()` 先对 `valid_m` 做 16 对齐
   （`scheduler/mega_moe.cuh:127-129`）。末块 token 不满时按实际行数缩 N 轴，**不做无用的
   零行 MMA**——MoE 里每个专家的 token 数参差，这个优化很关键。
2. **TMEM 双缓冲**（`:863-866`）：累加器在 TMEM 里，只有 `kNumEpilogueStages=2`（`:177`）个槽。
   `current_iter_idx` 每处理完一个 task 自增，`accum_stage_idx` 轮转、`accum_phase` 每两轮翻一次。
   `tmem_empty.wait(accum_phase ^ 1)` 是「等 epilogue 把这个槽读空」——注意相位取反
   （初相位的推导见 `docs/09` §5.1）。
3. **K 循环**（`:883-930`）：

```cpp
for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; advance_pipeline(k_block_idx)) {
    shared_storage.full_barriers[stage_idx].wait(phase);      // 等 TMA 装好 A/B/SF
    const uint32_t a_desc_base_lo = a_desc_lo + stage_idx * sizeof(smem_a[0]) / 16;
    const uint32_t b_desc_base_lo = b_desc_lo + stage_idx * sizeof(smem_b[0]) / 16;
    if (cute::elect_one_sync()) {
        for (umma_k_block_idx = 0; umma_k_block_idx < BLOCK_K / UMMA_BLOCK_K; ++umma_k_block_idx) {
            // (a) UTCCP: SFA/SFB smem -> TMEM
            // (b) UMMA issue
        }
    }
    __syncwarp();
    empty_barrier_arrive(k_block_idx == num_k_blocks - 1);    // 释放 stage + 通知 epilogue
}
```

`stage_idx` 的推进、相位翻转由 `advance_pipeline`（`:308-314`）完成，`docs/09` §3 已讲。
描述符 base 用 `+ stage_idx * sizeof(...) / 16` 推进——**16 字节是 SMEM 描述符的地址单位**
（`make_smem_desc` 里 `start_address_ = uint_ptr >> 4`，`mma/sm100.cuh:30`）。

### 1.4 UTCCP：把 block scale 从 SMEM 搬进 TMEM（`:894-907`）

```cpp
using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_2cta;
for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++ i) {
    auto smem_ptr = shared_storage.smem_sfa[stage_idx]
                  + umma_k_block_idx * SF_BLOCK_M + i * kNumUTCCPAlignedElems;
    mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr);
    cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
}
// SFB 同理，写到 kTmemStartColOfSFB
```

- **为什么还要 UTCCP**：`tcgen05.mma...block_scale` 指令的 SF 操作数是 **TMEM 里的列地址**
  （`[tmem_sfa]`、`[tmem_sfb]`，见 §1.5），不能直接读 SMEM。UTCCP（`tcgen05.cp`）就是
  专门把 SMEM 的 scale 搬进 TMEM 的拷贝引擎。SMEM 里 SF 的布局在 kernel 外已按 UTCCP 的
  **4×32 转置 pattern** 重排（`transform_weights_for_mega_moe`，见 `docs/02` §3.4）。
- `kNumUTCCPAlignedElems = 128`（`:140`）：UTCCP 一次处理 128 元素，`SF_BLOCK_M` 必须是
  128 对齐（`:141` 静态断言）；SMEM 里每 128 元素一段，TMEM 里按 `+ i*4` 列步进
  （128b = 4 个 32-bit 列）。
- SFA/SFB 在 TMEM 的列区间由 `kNumAccumTmemCols`（累加器）之后紧接着放
  （`:229-234`）：`kTmemStartColOfSFA = UMMA_N*2`，`kTmemStartColOfSFB = kTmemStartColOfSFA + SF_BLOCK_M/32`。

### 1.5 UMMA 指令与 accumulate 标志（`:909-922`）

```cpp
for (uint32_t k = 0; k < UMMA_BLOCK_K / UMMA_K; ++ k) {   // 128/32 = 4 次
    const auto runtime_instr_desc =
        mma::sm100::make_runtime_instr_desc_with_sf_id(instr_desc, k, k);
    a_desc.lo = mma::sm100::advance_umma_desc_lo<Major::K, LOAD_BLOCK_M, 128, a_dtype_t>(
        a_desc_base_lo, umma_k_block_idx * UMMA_BLOCK_K * LOAD_BLOCK_M * sizeof(a_dtype_t), k * UMMA_K);
    b_desc.lo = ... (b 同理);
    ptx::SM100_MMA_MXF8F6F4_2x1SM_SS::fma(
        b_desc, a_desc, accum_stage_idx * UMMA_N,
        k_block_idx > 0 or umma_k_block_idx > 0 or k > 0, runtime_instr_desc,
        kTmemStartColOfSFB, kTmemStartColOfSFA);
}
```

- **内层 `k` 循环 4 次**：`UMMA_BLOCK_K=128`、`UMMA_K=32`，因为 **block_scale 的 SF 粒度是 32**
  （`gran_k=32`，`heuristics/mega_moe.hpp:217`）。`make_runtime_instr_desc_with_sf_id(instr_desc, k, k)`
  （`mma/sm100.cuh:144-148`）把 `a_sf_id_=b_sf_id_=k` 塞进 runtime descriptor 的**高 32 位**
  ——它告诉硬件「这次 MMA 用第 k 组 SF」。
- **accumulate 谓词**（`:920`）是全 kernel 的「第一次」判定：`k_block_idx>0 || umma_k_block_idx>0 || k>0`
  为真时累加、为假时**清零**累加器。即一条 task 的 **(k_block=0, umma_k=0, k=0) 这条指令用非累加模式**，
  之后的全部累加。这就是为什么 TMEM 槽不用显式清零。
- `fma` 的实现在 `ptx/tcgen05.cuh:67-86`：

```asm
tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p;
```

`cta_group::2` = 2SM；`kind::mxf8f6f4` = 支持 MX FP8/FP6/FP4；`block_scale` = 带 per-block scale；
`p` 就是 accumulate 谓词。**注意参数顺序是 `b_desc, a_desc`**（交换过），对应 §1.1 的 A/B 互换。

### 1.6 FP4：打包因子与描述符步进（`:152-156`、`packing.cuh`、`mma/sm100.cuh`）

routed 权重可以是 FP4（E2M1）。SMEM 里 FP4 **两个逻辑元素挤一个字节**：

```cpp
// common/packing.cuh:8-12
return cute::is_same_v<dtype_t, cutlass::float_e2m1_t> ? 2 : 1;
```

`advance_umma_desc_lo` / `make_umma_desc` 里的字节偏移都除以 `kPackFactor`
（`mma/sm100.cuh:92-96`、`103-104`、`118-119`），于是「同样的 SMEM 容量装下 2 倍 K」。
host 侧（`heuristics/mega_moe.hpp:80-82`）也按 1 字节/元素算 `block_k` 与 smem 预算。
这也解释了 §1.2 里 routed/shared 两个 descriptor 必须 `sizeof` 相同才能共用地址算式。

### 1.7 commit：一次 multicast 通知两个阶段（`:869-880`、`:927-929`）

```cpp
auto empty_barrier_arrive = [&](const bool& do_tmem_full_arrive) {
    auto umma_arrive = [](const uint64_t* barrier) {
        cutlass::arch::umma_arrive_multicast_2x1SM(barrier, /*kCTAMask=*/(1<<2)-1); };
    umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.empty_barriers[stage_idx]));
    if (do_tmem_full_arrive)
        umma_arrive(reinterpret_cast<uint64_t*>(&shared_storage.tmem_full_barriers[accum_stage_idx]));
    __syncwarp();
};
...
empty_barrier_arrive(k_block_idx == num_k_blocks - 1);
```

`tcgen05.commit` 是异步 MMA 与 mbarrier 的桥：MMA 全部完成后才 arrive。这里每个 K block 结束
都 arrive 一次 `empty_barriers[stage_idx]`（释放这个 TMA stage 给下一个 K block 复用），
但**只有当这是该 task 最后一个 K block 时**才额外 arrive `tmem_full_barriers[accum_stage_idx]`
（告诉 epilogue「TMEM 这个槽的累加结果好了」）。`multicast_2x1SM` 把 arrive 同时发给 cluster
里两个 CTA 的同一 barrier。task 最后还有一轮收尾 wait（`:933-937`），保证 barrier 析构安全。

## 2. Epilogue：`warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps`（`:946-1521`）

epilogue 是 **2 个 warpgroup（8 warp / 256 线程）**，每个 warpgroup 吃一半的 `BLOCK_M`
（`WG_BLOCK_M = BLOCK_M/kNumEpilogueWarpgroups`，`:969`），每个 warp 吃 `BLOCK_N/4`
（`:966`）。它同时负责 GEMM 后处理**和**最后的 combine 归约。

### 2.1 TMEM 双缓冲消费与提前释放（`:979-1034`）

```cpp
while (scheduler.get_next_task(task_info)) {
    const auto accum_stage_idx = current_iter_idx % kNumEpilogueStages;
    const auto accum_phase    = (current_iter_idx ++ / kNumEpilogueStages) & 1;
    shared_storage.tmem_full_barriers[accum_stage_idx].wait(accum_phase);
    ptx::tcgen05_after_thread_sync();
    scheduler.release_task_info();          // 立刻还调度槽
    ...
```

三个「为什么」：

- **TMEM 槽释放的时机**：累加器是稀缺资源（`kNumAccumTmemCols=UMMA_N*2`，`:229`）。epilogue
  并不等整块算完才释放，而是把 store block 逐个从 TMEM 读进寄存器，**读到最后一个有效块的瞬间**
  才 `release_tmem()`（`:1030-1033`、`:1006-1009`），并把这一步的读提前到 epilogue 前
  （`kNumEpiPrefetchStages`，`:179-180`，只有 `BLOCK_M==240` 时开 1 级）。这样 MMA 下一轮
  能更早拿到空槽——**epilogue 不在关键路径上**。
- **`scheduler.release_task_info()` 提前**（`:990`）：任务一确认算完（TMEM full 已到），
  立刻把调度槽还给生产者，不等数据搬出去。这就是 `docs/09` §4 说的「仅 epilogue release」。
- `SM100_TMEM_LOAD_16dp256b1x`（`:1017-1020`）每次把 8 个 fp32 从 TMEM 读进寄存器：
  一次 `tmem_addr`、一次 `tmem_addr | 0x00100000`（TMEM 地址 [31:16] 是列号，bit 20 = 列 +16）。

### 2.2 L1 epilogue：in-place SwiGLU + UE8M0 量化 + 转置上屏（`:1058-1253`）

进入条件 `block_phase == Linear1 or SharedLinear1`（`:1058`）。这是 Mega MoE 最精妙的一段：
linear1 的输出（gate/up）**不落地**，直接在寄存器里做 SwiGLU、量化成 FP8、写回全局当作
linear2 的激活。

**(a) 交错权重让 gate/up 成对**（`:1066-1067`）：kernel 外已把 L1 权重按 **8 粒度交错**
（`transform_weights_for_mega_moe`，`docs/02` §3.4）。于是 §2.1 的 8 个寄存器值天然是
`[g0,u0,g1,u1,g2,u2,g3,u3]`，`fp32_values[k*2+0]` 是 gate、`[k*2+1]` 是 up（`:1112-1113`）——
**无需任何 shuffle/寻址**就能配好对。

**(b) 乘 top-k 路由权重**（`:1093-1106`）：每 32 个 token 从 `l1_topk_weights_buffer` 读一次
权重缓存进寄存器（`stored_cached_weight`），再用 `ptx::exchange` 按 lane 广播成 fp32。
只有 routed block 才乘（shared 专家权重为 1）。

**(c) SwiGLU**（`:1108-1135`）：gate/up 先 `__float22bfloat162_rn` **round 到 bf16 再转回 fp32**，
然后 `silu(gate)*up*weight`。为什么要 round bf16——这是为了与 DeepSeek 推理参考实现的数值
对齐（BF16 激活路径），属于「算子忠实度」而非性能考虑。还带一个可选 clamp
（`kActivationClamp`，`:1116-1120`）。

**(d) amax + UE8M0 scale**（`:1137-1216`）：这是 SM100 MX 格式的核心。

```cpp
amax_values[i].x = math::warp_reduce<4, true>(thread_local_amax.x, math::ReduceMax<float>());
...
shared_storage.amax_reduction[epilogue_warp_idx][i*(ATOM_M/2) + lane_idx] = amax_values[i];
// 跨 warp-pair 再取 max（epilogue_warp_idx ^ 1）
const uint2 sf_exp = {math::get_ue8m0_sf_exp(amax_values[i].x),
                      math::get_ue8m0_sf_exp(amax_values[i].y)};
const float2 sf_inv = {math::get_ue8m0_sf_inv<float>(sf_exp.x), ...};
```

`get_ue8m0_sf_exp`（`common/math.cuh:114-135`）用**一次整数加法进位**替 `ceil`：
把 amax 的位模式加上 `(mantissaMask - quantMaxMantissa)` 再右移 mantissaBits，
恰好实现「向上取到不超过 qmax 的幂次」；`get_ue8m0_sf_inv`（`:137-147`）利用
「biased exponent 之和为 254 ⇒ 乘积精确为 1」。这两条在 §5.1/§5.2 逐位验证。
`warp_reduce<4>` 只跨 4 个 lane，因为 warp-pair（两个 warp）各算一半 SF 覆盖的行。

**(e) FP8 cast + STSM + TMA 转置上屏**（`:1178-1233`）：乘 `sf_inv` → `__nv_fp8x4_e4m3` →
`SM100_U8x4_STSM_T` 写 SMEM（64B swizzle，`:1186-1191`）。SF 本身按 **MN-major** 写到
`buffer.l2_sf_buffer`（`:1193-1217`），token 索引用 `transform_sf_token_idx`（与权重侧同一转置，
`:145-149`），代码里还把不变的项提出循环（`:1202-1213` 的注释）。最后
`SM90_TMA_STORE_2D`（`:1226-1232`）把 `smem_d.l1` **转置**写到 `tensor_map_l1_output`
（即 `l2_acts`，`impls/...hpp:72`）——L1 输出就是 L2 的激活（宽度 BLOCK_N/2，SwiGLU 后）。

**(f) 依赖通知**（`:1237-1252`）：`tma_store_wait` 后，leader warp 对
`l1_empty_count` 做 `red_add`、并用 `L2KBlockDependency::arrive` 在 `l2_full_mask` 上置位
（`scheduler/mega_moe.cuh:142-166`）。**每个 L1 N 块算完就点亮一个 bit，L2 的 K block 一凑齐
就能开跑**——这是 L1→L2 流水（生产/消费）的握手，避免等整个 L1 块全完。

### 2.3 L2 epilogue：BF16 直写远端 combine buffer（`:1254-1357`）

`else` 分支处理 `BlockPhase::Linear2`。输出是 BF16，**不做量化**，直接通过 NVLink 写到
目标 rank 的 combine buffer：

- 先读每个 token 的来源元数据 `TokenSrcMetadata{rank_idx, token_idx, topk_idx}`
  （`:1282-1290`，`layout/mega_moe.cuh:42-52`），提前发起以隐藏延迟；
- STSM 写 SMEM 后 `ld_shared` 取出，再 `*sym_buffer.map(dst_ptr, dst_rank_idx) = packed`
  （`:1339-1351`）——`map` 把本地指针映射成对称显存的**远端地址**，一条 16B store 过去。
  这是「计算完直接 scatter 到 combine 缓冲」，省掉「先写本地、再统一通信」的两段式。

### 2.4 Combine：top-k 归约写回（`:1360-1520`）

所有 L2 写完后跨 SM `grid_sync`（`:1412-1415`），并向对端 rank 推 `combine_ready_grid_idx`。
主循环按 token（可切 1/2 个 chunk，`:1376-1379`）走：

- 从 `combine_token_buffer` 的对应 rank/slot **TMA 1D load** 每个 top-k 专家对该 token 的贡献
  （`:1447-1466`），双缓冲（`combine_load_buffer`）重叠；
- 在 **fp32 寄存器**里累加所有 top-k（+ 可能第 `kNumTopk` 个 shared 专家）贡献（`:1472-1490`）；
- 再 cast 回 bf16、TMA 1D store 到输出 `y`（`:1492-1519`）。
  `wait_until(...)` 用 `ld_acq_sys` 轮询对端 readiness（`:1440`），确保对端 L2 写已可见。

## 3. 一条 task 的完整数据流（前向全链）

```
dispatch 拉取 token 激活 (FP8) + SF
  → [MMA warp] UTCCP(SF→TMEM) + UMMA(FP8xW, block_scale)  → TMEM 累加器
  → [epilogue] 读 TMEM → (L1) in-place SwiGLU + UE8M0 量化 → TMA 转置写 l2_acts
             ↘ (L2) BF16 → NVLink scatter 到远端 combine buffer
  → combine: TMA 拉 top-k 贡献 → fp32 累加 → bf16 → 写 y
```

L1/L2 是同一批 epilogue warp 按 `task_info.block_phase` 分支处理的两种「模式」，共享
TMEM→寄存器→SMEM→(TMA/NVLink) 的骨架；区别只在 L1 多一段 SwiGLU+量化、L2 多一段远端写。

## 4. 与朴素实现的差异（为什么快）

| 朴素实现 | Mega MoE | 省掉了什么 |
|---|---|---|
| linear1 输出写回 HBM，另起 kernel 做 SwiGLU + 量化 | L1 epilogue 寄存器内 in-place SwiGLU + 量化，TMA 直写 L2 激活 | 一次 HBM 往返 + 一个量化 kernel |
| 每个 MoE 层分别量化激活、单独存 SF | SF 在 epilogue 内算出并 MN-major 落盘，供下一层 UMMA 直接 UTCCP | 独立的 amax/quantize pass |
| 等整个 L1 块算完再启动 L2 | L2 K-block 用 bitmask 依赖，L1 的每个 N 块算完即点亮 | L1→L2 的整块串行等待 |
| combine 先写本地再 all-to-all | L2 BF16 直接 NVLink scatter 到远端 combine buffer | 本地中转 + 一次规约通信 |

## 5. H100 实测：epilogue 数值原语逐位核对

MMA/UTCCP/TMEM 是 SM100 专属，跑不了；但 epilogue 的**数值原语**是纯标量运算，可以在
H100 上用 Python 位运算逐位复现。实验 `harness/exp_ue8m0_sf.py`（输出
`harness/exp_ue8m0_sf.out.txt`，H100 80GB，驱动 570.211.01）：

```
== 1) `get_ue8m0_sf_exp` 位运算 vs 参考 (ceil(log2(amax/qmax))) ==
   [e4m3] 25009 个样本，不符 0 个
   [e2m1] 25009 个样本，不符 0 个
== 2) UE8M0 scale 与 inverse 乘积 ==
   sf_exp 范围 [105,136]，scale*inv 最大偏差 0.000e+00
== 3) block-scale FP8 量化往返（K=256，按 32 分组，torch FP8） ==
   kernel 幂次 scale : max abs err 9.886e-01, 相对 3.326e-02, 溢出 0
   理想 amax/qmax    : max abs err 8.434e-01, 相对 2.838e-02, 溢出 0
   幂次 scale / 理想 scale 最大比值 (应 <= 2): 1.9977
== 4) L1 epilogue in-place SwiGLU 数值 ==
   bf16-round gate/up vs 全 fp32：max abs 3.634e-01, 相对 4.524e-03
```

**结论**：

1. **进位代替 ceil 正确**（`:114-135`）：§5.1 用 25009 个样本（含 0、亚正规、幂次边界
   447/448/896）覆盖 E4M3 与 E2M1，位运算结果与「最小 e 使 `qmax·2^e ≥ amax`」逐点一致，
   **0 个不符**。说明 `(bits + mask - qmaxMantissa) >> mantissaBits` 确实等价于
   `ceil(log2(amax/qmax))`（下界 clamp 到 `-22`/`-126`）。
2. **scale/inverse 乘积恒为 1**（`:137-147`）：biased exponent 之和恒为 254，乘积偏差
   **精确 0**（不是近似）——量化本身不引入额外误差。
3. **幂次量化的代价有界**：kernel 的幂次 scale 相对误差 3.33e-2，理想 `amax/qmax` 是 2.84e-2；
   幂次 scale 与理想 scale 的比值 **≤ 1.9977 < 2**，且 0 溢出。这坐实了「**UE8M0 幂次 scale
   最多浪费 1 bit 动态范围，但换来一次整数加法算 SF、且可被 UTCCP 硬件搬运**」的取舍。
4. **in-place SwiGLU 的 bf16 round**（`:1108-1135`）：先 round gate/up 到 bf16 再算 silu，
   与全 fp32 参考的相对误差 **4.5e-3**——正是 bf16（~3 位十进制有效数字）的典型量级，
   解释了 kernel 为何要显式 round（与推理参考对齐，而非精度损失 bug）。

## 6. 小结

- **MMA 本体**（`:819-938`）只有 leader CTA 的一个 warp 发指令；A/B 互换 + 2-CTA 让
  `UMMA_M=256`、`UMMA_N=BLOCK_M`（token 维），并按 `valid_m` 动态缩 N；block_scale 的 SF
  经 UTCCP 从 SMEM 进 TMEM，内层按 `gran_k=32` 拆 4 条 UMMA；第一条指令非累加清零、其余累加。
- **Epilogue**（`:946-1521`）用 TMEM 双缓冲，读完最后有效块即释放；L1 分支做 in-place SwiGLU +
  UE8M0 量化 + 转置上屏（就是 L2 激活），L2 分支 BF16 直写远端 combine buffer；combine 段
  TMA 拉取 top-k 贡献、fp32 累加、bf16 写 `y`。
- 数值原语在 H100 上逐位复核通过（§5），UE8M0 幂次量化的取舍有真实数字支撑。

**下一步（P3-4 / P4）**：dispatch / combine 段的 NVLink 读写与 locality domain；反向的
wgrad/dgrad grouped GEMM（`k_grouped_*`）精读与 H100 数值对拍。
