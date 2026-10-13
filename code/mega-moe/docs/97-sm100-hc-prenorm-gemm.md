# SM100 HyperConnection prenorm GEMM：JIT 启动包装层与「TMA→cast→UMMA TS」三段流水（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P87 增量（第 97 篇）。
> 第 16 篇（P7-2）把 `tf32_hc_prenorm_gemm` 的**独立算子形态**讲了一遍——「mix 系数投影 +
> RMSNorm 平方和」融进一个 TF32 kernel、H100 上可跑、并量了数值/性能。但那一篇对 **SM100 分支**
> 只写了 4 条 bullet（`sm100_tf32_hc_prenorm_gemm.cuh:169-173/268-342`，标注「仅分析」）。
> 本篇把 SM100 那条路补齐：**host JIT 启动包装层**（纯 host，H100 可跑真机）+ **kernel 的逐行实现**
> （SM100 专属，仅分析）+ 把架构无关的部分（host 配置循环、split-K 分解、银行组 swizzle、
> cast-reduce 映射）在 H100 上跑**真实设备代码**核验。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。SM100 kernel 在 H100（sm90）不可运行，
> 明确标注「SM100 专属，仅分析」；本篇实测的是其中与 UMMA/TMEM 无关的那一层。

## 1. 缺口：四个 mega 家族包装层的最后一块

第 79 篇（P79-1）拆了 `sm100_bf16_mega_moe.hpp` / `sm100_mega_mhc.hpp` 两个 JIT 启动包装层，
第 86 篇（P86-1）补齐了 `sm100_fp8_fp4_mega_moe.hpp` / `sm100_bf16_mega_gate.hpp`，
但**同一目录下的 `sm100_tf32_hc_prenorm_gemm.hpp`（113 行）**一直不在那张表里——它是 mHC 家族里
唯一一个「非 mega-kernel、走独立 TF32 流水」的启动包装层。再加上它的 SM100 主 kernel 也从没被逐行读，
本篇把这块补上。

回顾数据流（第 16 篇 §1）：HyperConnection 里每个 token 的激活 `new_residual ∈ R^{m×k}` 立刻被两处消费：
投影出 **mix 系数** `D[m,n]=Σ_k A[m,k]·B[n,k]` 和提供 **RMSNorm 分母** `S[m]=Σ_k A[m,k]²`。
其中 `A`=bf16 激活、`B`=fp32 系数投影权重（保尾数精度）、`D/S`=fp32。独立版用于非融合路径。

## 2. host 启动包装层：`sm100_tf32_hc_prenorm_gemm.hpp`（纯 host，H100 真机）

整个 wrapper 只做四件事：**形状/常量推导 → TMA 描述符 → JIT 模板实参 → launch 参数**。

### 2.1 形状与契约（`:21-31`）

```cpp
constexpr int block_m = 64;               // :21
constexpr int block_k = 64;               // :22
constexpr int num_mma_threads = 128;      // :23
constexpr int num_cast_and_reduce_threads = 128;  // :24
const int block_n = align(n, 16);         // :26
DG_HOST_ASSERT(n <= block_n);             // :27
DG_HOST_ASSERT(n <= 128 and n % 8 == 0);  // :28  ← 与 SM90 的 n<=32 不同
DG_HOST_ASSERT(k % block_k == 0);         // :29
```

- **`block_n = align(n, 16)`**：N 轴很小（真实 `n≈hc×4`，测试用 24），对齐到 16 的倍数只为满足
  swizzle/TMA atom；`n<=128` 是 SM100 相比 SM90（`sm90_...hpp:30` 的 `n<=32`）**放宽的上限**
  （SM100 用 TMEM 累加，N 不再受寄存器累加器大小的限制）。
- **`block_m=64` / `block_k=64` 与 SM90 相同**（都是常量），`u` 两个 warpgroup 各 128 线程。
- **`k % block_k == 0`**：K 必须 64 对齐——因为 split-K 按 BLOCK_K=64 的整数块切（见 §4）。

### 2.2 TMA 描述符（`:32-50`）

A/B 都按 `cute::UMMA::Major::K`（K-major）造描述符，内层 swizzle 由 `get_swizzle_mode(block_k, elem_size)`
决定。A 是 bf16、B 是 **fp32**（`b` 在 host 契约里就是 `torch::kFloat`）。D 分两种：

```cpp
const auto tensor_map_d = num_splits == 1
    ? make_tma_cd_desc(d, m, n, block_m, block_n, d.stride(-2), 1, swizzle_cd_mode)          // :42-45
    : make_tma_3d_desc(d, n, m, num_splits, block_n, block_m, 1,
                       d.stride(-2), d.stride(-3), swizzle_cd_mode);                          // :46-50
```

注意 **3D 版把 `num_splits` 当第三维**（`d` 形状 `[num_splits, m, n]`），且 box 参数顺序变成
`(block_n, block_m, 1)`——这正是 split-K 把每个 split 的部分 D 写到独立切片所需的布局（§4）。

### 2.3 num_stages 预算循环与 SM90 的三处差异（`:52-67`）

```cpp
int num_stages = 12, smem_size = 0;
while (num_stages > 0) {
    const int smem_a_per_stage  = block_m * block_k * sizeof(nv_bfloat16);
    const int smem_b_per_stage  = block_n * block_k * sizeof(float);
    const int smem_cd           = block_m * swizzle_cd_mode;
    const int smem_barriers     = (num_stages * 4 + 1) * 8;   // :58
    const int smem_tmem_ptr     = 4;                          // :59
    smem_size = (smem_a_per_stage + smem_b_per_stage) * num_stages
              + smem_cd + smem_barriers + smem_tmem_ptr;
    if (smem_size <= SM100ArchSpec::smem_capacity) break;
    -- num_stages;
}
```

与 SM90 wrapper（`sm90_...hpp:55-79`）逐条对照，三处差异：

| | SM90 | SM100 |
|---|---|---|
| barrier 家族数 | `num_stages * 2 * 8`（full/empty 两组）| `(num_stages * 4 + 1) * 8`（**四组** + tmem_full）|
| TMEM 指针 | 无 | `+4` 字节（`tmem_ptr_in_smem`）|
| 收尾 | `smem_size = SM90ArchSpec::smem_capacity`（**吃满 227KB**）| 保持算出的 `smem_size` |

第一处来自 SM100 的三段流水需要 `full / full_cast / empty / empty_cast` 四级 mbarrier（§3），
再多一个 `tmem_full`（§3.4）；第三处是因为 SM90 用 register 累加器、必须动态 smem 吃满才能
`setmaxnreg`，而 SM100 用 TMEM 累加、smem 只需够 A/B stage 即可。

### 2.4 JIT 模板与 launch（`:79-110`）

JIT 微源码（`:79-100`）实例化 `sm100_tf32_hc_prenorm_gemm_impl<n, k, block_m, block_n, block_k,
num_splits, swizzle_cd_mode, num_stages, num_mma_threads, num_cast_and_reduce_threads>`——
**10 个模板参数全部是编译期常量**（没有 `compiled_dims` 那种运行时维度，因为 m 通过 `shape_m`
参数传、只用于边界）。launch（`:103-110`）：

```cpp
.grid_dim  = dim3(num_splits * ceil_div(m, block_m), 1, 1);   // :106
.block_dim = dim3(num_mma_threads + num_cast_and_reduce_threads, 1, 1);  // :107 = 256
```

grid 的第一维把 **M block 与 K split 交织**（`block_idx / kNumSplits = m_block_idx`，
`block_idx % kNumSplits = k_split_idx`，kernel `:129-131`）。

## 3. SM100 kernel 精读（SM100 专属，仅分析）

文件 `deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`（351 行）。
整段 kernel 体包在 `#if __CUDA_ARCH__ >= 1000` 里（`:48`），H100 上只会编译到 `#else` 的断言分支。

### 3.1 三段流水与五类 warp

一段 tile 的物理路径与前两篇（第 10 篇 Mega MoE MMA、第 35 篇 SM100 FP8/FP4）同源的
**「TMA → cast → MMA」三段解耦**：

```
A(bf16)/B(fp32) ──TMA──► smem ──cast warpgroup──► TMEM(A fp32) ──UMMA(TS)──► TMEM(D) ──epilogue──► smem ──TMA──► D
                              └── square-sum ──► sqr_sum[global]
```

warp 分工（`:140`/`:163`/`:268` 三分支）：

| warp | 角色 |
|---|---|
| warp 0（MMA 组内，`elect_one`）| TMA load A/B（`:142-160`）|
| warp 1（MMA 组内）| 发 UMMA（`:163-213`）|
| warp 0-3（整个 MMA 组）| epilogue：从 TMEM 读 D、STSM 进 smem、TMA store（`:232-262`）|
| warp 4-7（`kNumCastAndReduceThreads` 组）| cast-and-reduce：A bf16→fp32 写 TMEM + 平方和（`:268-342`）|

**为什么需要 cast warpgroup**：SM100 的 UMMA **没有寄存器 A 操作数**——TS 形态
（`SM100_MMA_TF32_TS`，`:169-170`）要求 A 在 **TMEM**、B 在 SMEM。而 A 在 HBM 里是 bf16
（省带宽，第 16 篇 §5.5 的 roofline 结论）。于是必须有人把 A 从 smem 的 bf16 转成 fp32 再写进 TMEM。
这一步顺便把平方和算了（§3.4）——这正是「搭便车」思想在 SM100 上的落点。

TMEM 列预算（`:75`）：

```cpp
constexpr uint32_t kNumTmemCols = utils::get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>();
```

`kNumCastStages=2`（`:52`，A 的双缓冲），所以 TMEM 里 **A 占 `BLOCK_K*2`=128 列、D 占 `BLOCK_N` 列**；
`get_num_aligned_tmem_cols`（`common/utils.cuh:42-50`）把它向上取整到 {32,64,128,256,512} 之一。
UMMA 的 A 基址就是 `BLOCK_K * cast_stage_idx + k * UMMA_K`（`:203` 的 `fma(..., BLOCK_K * cast_stage_idx + k * UMMA_K, ...)`）。

### 3.2 四组 mbarrier 的 init 计数（`:97-116`）

```cpp
full_barriers[i]->init(1);                        // TMA 完成 -> MMA 消费
full_cast_barriers[i]->init(kNumCastAndReduceThreads);  // =128，每个 cast 线程 arrive
empty_barriers[i]->init(1);                       // MMA 消费完 -> TMA 复用
empty_cast_barriers[i]->init(1);                  // UMMA commit 通知 -> cast 复用
tmem_full_barrier->init(1);                       // MMA 全部 commit -> epilogue
```

`full_cast_barriers` 的计数是 **128**（每个 cast 线程各 arrive 一次，`:331`），而其余三个是 **1**
（由 `elect_one` 或 UMMA commit 一次性 arrive）。这解释了 §2.3 的 `(num_stages*4+1)*8`：
每 stage 四条 mbarrier、再加一条全局 `tmem_full`。

### 3.3 UMMA 描述符与 K 推进（`:165-209`）

- `UMMA_M=BLOCK_M=64`、`UMMA_N=BLOCK_N`、**`UMMA_K = 32/sizeof(float) = 8`**（TF32 的 K 是 8）。
- `umma_t = cute::SM100_MMA_TF32_TS<tf32,tf32,float,BLOCK_M,BLOCK_N,K,K>`（`:169-170`），
  `make_instr_desc`（`:171-173`）编出指令描述符。
- B 描述符用 `make_umma_desc<Major::K, BLOCK_N, BLOCK_SWIZZLED_BK, kSwizzleBMode>`（`:176`），
  逐 stage 的基址 `b_desc.lo = b_desc_lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16`（`:177`），
  再用 `advance_umma_desc_lo`（`:202`）按 `atom_idx/in_atom_idx` 在 atom 内推进。
- 每条 `umma_t::fma(accum_offset, b_desc, b_n, a_cumulative_flag, instr_desc)`（`:203`）里
  `s > 0 or k > 0` 是**首条清零**：整个 tile 的第一条 UMMA 走非累加、其余累加。
- `cutlass::arch::umma_arrive`（`:207-208`）同时通知 `empty_cast_barriers` 和 `empty_barriers`——
  一次 commit 让 cast warpgroup 和 TMA warp 都能复用各自的 buffer。整段 K 循环结束后
  `umma_arrive(tmem_full_barrier)`（`:212`）唤醒 epilogue。

### 3.4 cast-and-reduce warpgroup（`:268-341`，三段流水的中间段）

```cpp
constexpr uint32_t BLOCK_M_PER_WARP = BLOCK_M / 4;    // :271 = 16
const uint32_t sub_warp_idx = warp_idx - kNumMMAThreads / 32;  // :272
float2 sum[2] = {float2{0,0}, float2{0,0}};            // :278
for (s = 0; s < num_total_stages; ++s) {
    full_barriers[stage_idx]->wait((s/kNumStages)&1);  // :283 等 TMA
    // LDSM 从 smem 取 A（bf16）到寄存器
    ptx::SM90_U32x4_LDSM_N::copy(...);                 // :298-300
    empty_cast_barriers[cast_stage_idx]->wait(...);    // :305 等 TMEM 空
    fp32x2_values[u][i] = __bfloat1622float2(...);      // :315 bf16 -> fp32
    sum[u] = __ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);  // :316 平方和
    cute::SM100_TMEM_STORE_16dp256b1x::copy(...);       // :321-324 写 TMEM
    full_cast_barriers[cast_stage_idx]->arrive();       // :331 通知 MMA
}
```

三个要点：

1. **平方和的支点就是这次 bf16→fp32 转换**——`__ffma2_rn(v, v, sum)`（`:316`）在把 A 写进 TMEM 的
   同一拍累加 `v²`，**不额外读 A、不额外访存**（第 16 篇 §5.2 的量：融合比「单独 matmul + 单独
   sqr_sum」快 ~47×，正因为省掉了第二遍 A 的读）。
2. **`SM90_U32x4_LDSM_N`**（`:298`）：从 smem 用 ldmatrix 取 bf16——注意这里复用的是 **SM90 的
   LDSM 指令到 sm90 的 smem 加载**（cast 段不碰 UMMA，所以这条在 H100 上语义相同）。
3. **`SM100_TMEM_STORE_16dp256b1x`**（`:321`）：把 fp32 的 A 写进 TMEM 的 `cast_stage_idx*BLOCK_K + i*8`
   列——`16dp256b` 就是「16 个 datapath × 256 bit」的 TMEM 写形态，与第 25/35 篇 SM100 主 kernel
   的 TMEM 读写几何同源。

### 3.5 归约与写回（`:334-341`）

```cpp
for (u = 0; u < 2; ++u) {
    const auto reduced = math::warp_reduce_sum<4>(sum[u].x + sum[u].y);   // :337
    const auto m_idx = m_block_idx*BLOCK_M + sub_warp_idx*BLOCK_M_PER_WARP + lane_idx/4 + u*8;  // :338
    if (lane_idx % 4 == 0 and m_idx < shape_m)
        sqr_sum[m_offset + m_idx] = reduced;                              // :340
}
```

- **`warp_reduce_sum<4>`**（`common/math.cuh:225-227` → `warp_reduce<4,false>`）：用
  `__shfl_xor_sync` 偏移 2、1 把**同组 4 个 lane**（`lane/4`）归约到一起，不清扫其他组。
- **`m_idx` 映射**：每个 warp（`sub_warp_idx`）管 16 行；组号 `lane/4` + `u*8` 选出 16 行里的第
  `g + u*8` 行。每组的 4 个 lane 各持部分 K 的平方和，归约后由 `lane%4==0` 写回。

## 4. split-K 分解与 D/S 的 3D scratch（`:126-134`）

```cpp
constexpr uint32_t kNumKBlocks = math::constexpr_ceil_div(SHAPE_K, BLOCK_K);   // :126
constexpr uint32_t kNumKBlocksPerSplit = kNumKBlocks / kNumSplits;             // :127
constexpr uint32_t kRemainKBlocks = kNumKBlocks % kNumSplits;                  // :128
const uint32_t block_idx   = blockIdx.x;                                        // :129
const uint32_t m_block_idx = block_idx / kNumSplits;                            // :130
const uint32_t k_split_idx = block_idx % kNumSplits;                            // :131
const uint32_t k_offset = (k_split_idx * kNumKBlocksPerSplit
                           + cute::min(k_split_idx, kRemainKBlocks)) * BLOCK_K; // :132
const uint32_t m_offset = shape_m * k_split_idx;                                // :133
const uint32_t num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks);  // :134
```

- **余数分给前 `kRemainKBlocks` 个 split**（`:132/:134` 的 `min(s, rem)` 与 `s < rem`），保证每个
  K block 恰被一个 split 处理。
- **`m_offset = shape_m * k_split_idx`**（`:133`）：`sqr_sum` 是 `[num_splits, m]`，切片 `s` 的线性
  起点就是 `s*m`。D 走 3D TMA（§2.2），由 `tensor_map_d` 的第三维定位 `k_split_idx`。
- **为什么每个 split 写独立切片**：不同 split 由不同 SM 并行算同一输出行的**不同 K 段**，无法在
  kernel 内原地累加；把 fp32 部分和落到独立切片，由 host 在外面求和——这就是第 16 篇 §5.3 量到的
  「小 M 并行度救星」（`m=137` 时 1.5→5.4 TFLOPS）的实现。

## 5. 银行组 swizzle：`get_swizzled_smem_offset`（`:18-34`）

```cpp
template <uint32_t kSwizzleMode, uint32_t kSwizzleBase = 16>
CUTLASS_DEVICE uint32_t get_swizzled_smem_offset(const uint32_t& offset, const uint32_t& lane_idx) {
    const auto bank_group_idx = offset + lane_idx * (kSwizzleMode / kSwizzleBase);   // :22
    constexpr uint32_t kNumBankGroups = 128 / kSwizzleBase;                          // :27 = 8
    constexpr bool kHasShortcut = (kSwizzleMode / kSwizzleBase) == kNumBankGroups;   // :28
    auto row = kHasShortcut ? (offset / kNumBankGroups + lane_idx) : (bank_group_idx / kNumBankGroups);
    auto col = kHasShortcut ? (offset) : (bank_group_idx % kNumBankGroups);
    col ^= row % (kSwizzleMode / kSwizzleBase);                                      // :31
    return row * 128 + col * kSwizzleBase;                                           // :33
}
```

在 `kSwizzleMode=128, kSwizzleBase=16`（本 kernel 的常见情形）下，`kHasShortcut=true`，
返回 `lane*128 + (offset ^ (lane%8))*16`——**这正是标准 128B swizzle `cute::Swizzle<3,4,3>`**
（XOR 键取 row 的 bits[0,3)、作用在 16B chunk 的 bits[4,7)）。这个 helper 在 STSM 写 D（`:238`）与
LDSM 读 A（`:297`）里共用。它的存在是因为这两个 warp 没有用 `cute::Swizzle` 对象，而是手算了地址。

## 6. H100 实测（`harness/exp_hc_prenorm_sm100.{sh,out.txt}`）

环境：H100 80GB HBM3（sm90），deep_gemm `057ca59`，CUDA 13.2。实验 `src/hc_prenorm_sm100.cu`
用 nvcc 直接 include 真实头文件（`sm90_/sm100_tf32_hc_prenorm_gemm.hpp` + `sm100_...cuh`），
把**架构无关**的部分当真实代码跑；SM100 的 UMMA/TMEM 指令不实例化（kernel 体在 `#if` 里）。
**21 PASS / 0 FAIL**，文件 sha256：`sm100_...hpp bb44f58e…`、`sm90_...hpp 9570a7ce…`、
`sm100_...cuh 37d28d51…`、`hyperconnection.hpp 0c73a4ad…`。

### 6.1 Part 1：host num_stages / smem 预算循环（SM90 vs SM100）

复刻 §2.3 的循环，对 6 个 n 值打印（`block_m=block_k=64`）：

| n | block_n | swz_cd | SM90 stages / smem | SM100 stages / smem |
|---|---|---|---|---|
| 8 | 16 | 64 | 12 / 151744 | 12 / 151948 |
| 16 | 16 | 64 | 12 / 151744 | 12 / 151948 |
| 24 | 32 | 128 | 12 / 204992 | 12 / 205196 |
| 32 | 32 | 128 | 12 / 204992 | 12 / 205196 |
| 64 | 64 | 128 | **rejected**（n>32）| 9 / 229676 |
| 128 | 128 | 128 | **rejected** | 5 / 213164 |

坐实三条：① SM100 的 n 上限放宽到 128（SM90 在 `n=64` 就被 `sm90_...hpp:30` 拒）；② SM100 的
smem 恰多 204 字节（`n=8`：151948−151744=204）——正是 barrier 计数的
`(num_stages*4+1)*8 − num_stages*2*8 = (2*num_stages+1)*8 = 25*8 = 200` 加 4B TMEM 指针，
与 `num_stages=12` 精确吻合（另两个 n 的差值也是 204）；③ SM90 末尾把 `smem_size` 覆盖成满
227KB（`sm90_...hpp:79`）、SM100 保留算出的值。

### 6.2 Part 2：split-K K-block 分解

对 `k∈{384,448,7168}` × `splits∈{3,4,6,16}` 逐一验证：
**每个 K block 恰被覆盖一次**（`coverage-exactly-once=YES`）、余数分给前若干 split。
数值上，`m=64,n=24,k=384,splits=4` 时「4 个 split 的部分 D 之和」与完整 fp32 GEMM 的
`max|Δ| = 1.144e-05`（fp32 累加顺序噪声），且部分 D 张量确为 3D `[num_splits, m, n]`
（`stride(0)==m*n`），坐实 §4 的 `m_offset`/3D TMA 语义。

### 6.3 Part 3：`get_swizzled_smem_offset` vs `cute::Swizzle<3,4,3>`

把 helper 编进一个 sm90 kernel，对 `lane∈[0,32)`、`c∈[0,8)` 逐一比较
`get_swizzled_smem_offset<128,16>(c,lane)` 与 `cute::Swizzle<3,4,3>(lane*128 + c*16)`：
**ALL MATCH**。再验证它在 32×128 字节的 atom 上是 **16B chunk 的双射**（每格恰好命中一次）。
`kSwizzleMode=64`（`block_n=16` 时的 `swz_cd`）走**非 shortcut 分支**：128 个不同 16B cell、
无碰撞。**结论：这篇 kernel 手算的 swizzle 地址与 cute 标准 128B swizzle 完全等价**——这是
「手写地址对不对」的一个硬核验。

### 6.4 Part 4：cast-and-reduce warpgroup 的映射

用真实设备代码跑 `math::warp_reduce_sum<4>` 与 `m_idx = m_block_idx*64 + sub*16 + lane/4 + u*8`
（`:337-338`）：给每个 lane 灌可预测的部分和，64 个输出行**全部与参考相等**（`ALL MATCH`），
坐实 §3.5 的「组内 4 lane 归约 + 行映射」。另 `fma2(v,v,0)` 的平方累加 fp32 vs fp64 rel
`5.901e-08`（第 44 篇同结论）。
（注：LDSM 的**跨 lane 数据分布**属于 SM100 专属路径的一部分，本篇只验证到归约/映射层，
数据分布按源码解读。）

### 6.5 Part 5：SM100 wrapper 的 TMA 描述符构造

用真 torch 张量（`m=4096,k=7168`）按 §2.2 的参数调 `make_tma_a_desc`/`make_tma_b_desc`/
`make_tma_cd_desc`/`make_tma_3d_desc`：`n=24`（block_n=32）与 `n=128`（block_n=128）两种都成功构造，
2D 与 3D（split=16）描述符指纹**不同**且非空。坐实 host 侧「A/B K-major、D N-major、split 走 3D」的
描述符组合在真机上合法。

## 7. 与第 16 篇 / 第 13 篇的关系、小结

- **SM90 vs SM100 一句话**：SM90 把 A 从 bf16 转 fp32 放**寄存器**喂 WGMMA；SM100 把 A 转 fp32 放
  **TMEM** 喂 UMMA(TS)，并多出一个 cast-and-reduce warpgroup 承担这次转换 + 平方和。两者的
  「平方和搭便车」思想完全一致（第 16 篇 §4 已指出），本篇把 SM100 那半的 host 包装层与 kernel
  逐行补全。
- **三个可复用结论**：① 包装层与 kernel 的「接口」就 10 个编译期模板常量 + `shape_m` 一个运行时参数，
  m 只用于边界（`:339`）；② split-K 的部分和落独立切片、host 外部求和，是这个「小 N、大 K」算子
  的唯一并行度来源；③ 手写的银行组 swizzle 与 cute 标准 128B swizzle 数值等价（实测 ALL MATCH），
  可放心把它当 `Swizzle<3,4,3>` 读。
- **边界**：主 kernel（TMA/UMMA TS/TMEM/LDSM 数据分布）是 **SM100 专属，仅分析**；本篇实测的是
  host 包装层 + 三个架构无关的设备片段（bank swizzle、cast-reduce 归约映射、平方累加）。

**本篇实测脚本**：`src/hc_prenorm_sm100.cu` / `harness/exp_hc_prenorm_sm100.sh` / `.out.txt`（H100 真机）。
