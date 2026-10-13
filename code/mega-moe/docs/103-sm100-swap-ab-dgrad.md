# 103. SM100 BF16 GEMM 的 `kSwapAB`（dgrad）分支：A/B 角色互换、转置写回与选核启发式

> 第 19 篇（P19-1）拆了 SM90 BF16 主 kernel、第 36 篇（P36-1）拆了 SM100 BF16 主 kernel
> `sm100_bf16_gemm.cuh` 的骨架与 `set_tc_util`，但第 36 篇只在一条 bullet 里写「`kSwapAB`：dgrad 的
> m-grouped 强制 `swap_ab=true`（`heuristics/sm100.hpp:37-44`）……」，没有讲清**它到底改了什么、
> 为什么要改、host 怎么选它、转置写回怎么落地**。而它恰恰是 SM100 上 MoE **dgrad**（以及所有
> m-grouped 稠密 GEMM）唯一能走的那条路。本文把这条分支单独拆透：host 启发式（`heuristics/sm100.hpp`）→
> JIT 启动包装层（`impls/sm100_bf16_gemm.hpp`）→ 设备端 A/B 角色互换（`impls/sm100_bf16_gemm.cuh`）→
> 转置写回 epilogue（`epilogue/sm100_store_cd_swap_ab.cuh`）。
>
> **重要约束**：`sm100_bf16_gemm` 主体是 **SM100 专属，仅分析**（`sm100_bf16_gemm.cuh:43` 整段
> `#if __CUDA_ARCH__ >= 1000`，UMMA/TMEM/`.16x256b` 全是 SM100 硬件）。但 **host 启发式是纯 host C++**，
> 可以在 H100 上直接 include 真实头文件、喂真实 `GemmDesc`、把「选出来的 layout/storage/pipeline/launch」
> 逐字段对回源码公式——本文的实测部分就是这么做的（`src/sm100_swap_ab_dgrad.cu`）。

---

## 1. 它解决什么问题：`UMMA_M` 被钉死在 128，而 m-grouped 的 M 是「短的、变的」

SM100 的 UMMA（tcgen05 MMA）有一个硬约束：**M 维（layout A/D 的行数）只有 64/128/256 三档**
（`sm100_bf16_gemm.cuh:252-254` 的静态断言）。BF16 主 kernel 里把它取成

```
constexpr uint32_t LAYOUT_AD_M = 128;                       // 66
constexpr uint32_t UMMA_M = LAYOUT_AD_M * kNumMulticast;    // 67
constexpr uint32_t UMMA_N = kSwapAB ? BLOCK_M : BLOCK_N;    // 68
```

对一个普通稠密 GEMM `D[m,n]=A[m,k]B[n,k]ᵀ`，M 通常很大（几千上万）、N 也大：`UMMA_M=128` 装 M、
`UMMA_N=BLOCK_N` 装 N，天经地义。

但 MoE **dgrad** 的 m-grouped 形状恰好相反：它是「**同一份权重 B，按专家组 A**」的输入梯度
（第 13 篇），

```
D[m,n] = Σ_k A[m,k]·B[n,k],   A = 该组激活（M 短且逐组不等），B = 权重（N=hidden，固定且大）
```

M 是**每组几十~几百、且逐组不等**的短维，N 才是那个大而固定的维度。若沿用「M 进 `UMMA_M=128`」，
组内 M 不足 128 时的零填充/利用率损失、以及 M 的对齐要求（contiguous layout 要对齐到 128/256）
都会很别扭。

`kSwapAB` 的解法很直接：**把 A、B 的操作数角色在 MMA 指令里换过来**。换过来之后：

- `UMMA_M = 128`（不变，仍是 layout A/D 的行数）现在装的是**原来的 N（hidden，固定大维）**；
- `UMMA_N = BLOCK_M` 现在装的是**原来的 M（逐组短的维）**，而 `UMMA_N` 是 16 对齐、**可以逐块更新**的
  （`sm100_bf16_gemm.cuh:286` 的 `update_instr_desc_with_umma_n`）。

代价是：写回 D 时必须**再转置回来**——这正是 `epilogue/sm100_store_cd_swap_ab.cuh` 存在的理由。

用一句话概括：**`kSwapAB` 用「MMA 内部换轴 + epilogue 转置写回」的代价，换取「让 `UMMA_N` 去承接那个
逐组可变、按 16 对齐的短 M 维，把 `UMMA_M=128` 留给固定的大 N 维」**。

---

## 2. 它在栈里的位置

```
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(...)            Python 入口
   └─ csrc/apis/gemm.hpp                                  host 入口（第 12/13 篇）
        └─ sm100_m_grouped_bf16_gemm_contiguous(...)        impls/sm100_bf16_gemm.hpp:126
             ├─ GemmDesc{ .gemm_type = GemmType::MGroupedContiguous, ... }
             ├─ get_best_config<SM100ArchSpec>(desc)        heuristics/common.hpp:16
             │     └─ SM100ArchSpec::get_layout_candidates  heuristics/sm100.hpp:30
             │            → 命中「m-grouped 分支」:36-46，强制 swap_ab=true
             └─ SM100BF16GemmRuntime::compile_and_launch    impls/sm100_bf16_gemm.hpp:22
                    → JIT 模板实参 swap_ab 送进 kernel（:66）
```

只有 SM100 会选 `swap_ab`：SM90 的启发式里有一条硬断言 `DG_HOST_ASSERT(layout.swap_ab == 0)`
（`csrc/jit_kernels/heuristics/sm90.hpp:125`），SM90 BF16 m-grouped kernel 也没有 swap 分支。
所以本文讲的整条路都是 SM100 专属的。

---

## 3. host 启发式：m-grouped 分支把 `swap_ab` 写死

`SM100ArchSpec::get_layout_candidates`（`heuristics/sm100.hpp:30`）开头先把 `block_k` 按 dtype 定死：

```
const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());   // :32
```

BF16 → `128*8/16 = 64`。随后立刻分叉出 m-grouped 的特例（`:34-46`）：

```cpp
// Always enable swap A/B (and multicasting if possible) for m-grouped GEMMs
if (desc.gemm_type == GemmType::MGroupedContiguous or
    desc.gemm_type == GemmType::MGroupedContiguousWithPsumLayout or
    desc.gemm_type == GemmType::MGroupedMasked) {
    const bool swap_ab = true;                                             // :38
    const auto block_n = 128;                                              // :39
    const auto block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout(); // :40
    const auto cluster_m = 1;                                              // :41
    const auto cluster_n = ceil_div(desc.n, block_n) % 2 == 0 and desc.num_sms % 2 == 0 ? 2 : 1; // :42
    const auto layout = Layout{swap_ab, block_m, block_n, block_k, cluster_m, cluster_n};
    std::vector<Layout> candidates = {layout};
    return candidates;                                                     // 唯一候选
}
```

四个被钉死的数字，每一个都有「为什么」：

| 常量 | 值 | 为什么 |
|---|---|---|
| `swap_ab` | `true` | 把可变短 M 维放到 `UMMA_N`（见 §1）；m-grouped 是这条分支唯一的用途 |
| `block_n` | `128` | 静态断言 `kSwapAB and BLOCK_N == LAYOUT_AD_M`（kernel `:76`）——swap 后 `UMMA_M` 装的是这个 N，必须 =128 |
| `block_m` | `mk_alignment_for_contiguous_layout()` | contiguous 布局的 M 对齐（默认 128，SM100 理论值 256，见 §6） |
| `cluster_m / cluster_n` | `1 / (0|2)` | swap 后 layout A/D 只能落在 cluster N（kernel 侧约束 `:81`），故 `cluster_m=1`；N 能整除整卡偶数时开 2-CTA 多播 |

注意这里返回的是**唯一候选**——一旦是 m-grouped，就不再枚举别的 layout。这是「m-grouped 的 swap_ab 不是
一个可选项，而是唯一的形态」的代码级证据。

对比一下非 m-grouped（`GemmType::Normal` / `Batched`）的分支（`:49` 起）：它在 `swap_ab ∈ {0,1}` 上完整枚举，
并施加三条与 swap 相关的合法性过滤：

```cpp
if (swap_ab == 1 and cluster_m > 1) continue;                 // :81-82  换轴后 A/D 只能在 cluster N
if (swap_ab == 0 and cluster_n > 1) continue;                 // :90-91  未换轴时 cluster N 不成立
...
if (swap_ab and block_n != layout_ad_m) continue;             // :121-123  swap 要求 block_n == 128
...
const auto umma_n = swap_ab ? block_m : block_n;              // :130
if (umma_n + tmem_sf_cols > 512) continue;                    // :131-132 TMEM 列预算
```

所以「普通稠密 GEMM 也可以 swap」——`get_layout_candidates` 会给 `swap_ab=1` 的候选，只是
`compare`（`:276`）按 wave 数/利用率挑最好的一个，未必选中。

---

## 4. 转成编译期常量：`UMMA_N` / `LOAD_BLOCK` / `STORE_BLOCK` / 线程数

给定一个 swap layout，`get_storage_config`（`heuristics/sm100.hpp:165`）算出 load/store 块：

```cpp
const auto load_block_m = layout.block_m / layout.cluster_n;                                  // :170
const auto load_block_n = layout.block_n / layout.cluster_m;                                  // :171
const auto store_block_m = layout.swap_ab ? umma_step_n : std::min(layout_ad_m, layout.block_m); // :172 umma_step_n=16
const auto store_block_n = layout.block_n;                                                    // :173
```

到 kernel 里（`impls/sm100_bf16_gemm.cuh`）就变成一组常量：

```
UMMA_M              = 128 * kNumMulticast              // :67  （swap 后 = 大 N）
UMMA_N              = kSwapAB ? BLOCK_M : BLOCK_N      // :68  （swap 后 = 逐组短 M）
LOAD_BLOCK_M        = BLOCK_M / (multicast 时)         // :70-71
STORE_BLOCK_M       = kSwapAB ? 16      : min(BLOCK_M, 128)          // :85
STORE_BLOCK_N       = kSwapAB ? BLOCK_N : kSwizzleCDMode/sizeof(cd)  // :86
kNumUMMAStoreThreads= kSwapAB ? kNumEpilogueThreads : STORE_BLOCK_M  // :87
```

两个 swap 专属的取值很关键：

- **`STORE_BLOCK_M = 16`**：转置写回时，一次搬「16 行 M」——`16 = UMMA_STEP_N`，正是 `UMMA_N` 的
  最小原子。epilogue 用 128 个线程（一个完整 warpgroup）去读 TMEM 的 128 行，故 `STORE_BLOCK_N` 必须 =128
  （kernel `:76` 与 epilogue `sm100_store_cd_swap_ab.cuh:31-33` 的注释「TMEM 有 128 行」）。
- **`kNumUMMAStoreThreads = kNumEpilogueThreads`（=128）**：非 swap 时只用 `STORE_BLOCK_M`（可能只有
  8/16/32…）个线程写，swap 时要一个整 warpgroup（≥32 的倍数，`:88`）才能覆盖 128 行。

然后 `get_pipeline_config`（`:203`）里，swap 与非 swap 对 CD 的 smem 预算也不同：

```cpp
const int smem_cd = (layout.swap_ab ? store_block_m * store_block_n * elemSize(cd_dtype)
                                    : store_block_m * swizzle_cd_mode) * num_tma_store_stages;  // :214-215
```

即 swap 的 C/D stage 大小是 `16 × 128 × sizeof(cd)`（**实打实的 tile**），非 swap 则是
`store_block_m × swizzle_cd_mode`（用 swizzle 后的字节数）。这也解释了为什么 swap 的 `store_block_n`
就是 `block_n`——转置后的 tile 行数是 block_n。

`get_launch_config`（`:254`）对所有情况都返回 `{num_sms, cluster_size, 256, 32, 128, 128, 128}`，即
每 block 256 线程 = 128 非 epilogue + 128 epilogue；swap 用掉全部 128 个 epilogue 线程（`kNumUMMAStoreThreads`
= 128）。

---

## 5. 设备端：A/B 怎么换、`UMMA_N` 怎么动态、D 怎么转置写回

### 5.1 TMA 载入：`load_block_m` 变动态

非 swap 时 TMA load warp 用固定 `LOAD_BLOCK_M`；swap 时它是**逐块算出来的**（`:177`）：

```cpp
const auto load_block_m = kSwapAB ? scheduler.get_aligned_effective_m_in_block(m_block_idx) / kNumMulticast
                                  : LOAD_BLOCK_M;
```

`get_aligned_effective_m_in_block`（`scheduler/gemm.cuh:177-184`）在 psum-layout、`ensure_zero_padding=false`
时返回「把 `current_psum_m` 对齐到 `UMMA_STEP_N=16`」的值，其余情况返回 `BLOCK_M`。也就是说
**可变 M 的对齐颗粒度是 16**，与 `UMMA_N` 的下限 16 一致。

### 5.2 指令描述符：major 顺序互换

```cpp
auto instr_desc = kSwapAB ? make_instr_desc<bf16, bf16, float, UMMA_M, UMMA_N, kMajorB, kMajorA>()  // :237-238
                          : make_instr_desc<bf16, bf16, float, UMMA_M, UMMA_N, kMajorA, kMajorB>(); // :239-240
```

swap 时把 A、B 的 major **对调**传进去（`kMajorB, kMajorA`），因为指令描述符描述的是「哪个操作数走哪个
矩阵槽」。随后每个 m-block 用 `update_instr_desc_with_umma_n(instr_desc, umma_n)`（`:286`）把 `UMMA_N`
改成实测的有效 M。

### 5.3 发 MMA：`fma(b_desc, a_desc, ...)`

```cpp
if (kSwapAB)
    mma_t::fma(b_desc, a_desc, accum_stage_idx * UMMA_N, umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc); // :310-312
else
    mma_t::fma(a_desc, b_desc, accum_stage_idx * UMMA_N, umma_k_idx > 0 or k_block_idx > 0, runtime_instr_desc); // :313-315
```

`fma` 的第一个描述符对应 UMMA 的 A 操作数、第二个对应 B。swap 时把 `b_desc` 放前面、`a_desc` 放后面，
至此 A/B 角色在 MMA 层彻底对调。

### 5.4 转置写回：`sm100_store_cd_swap_ab`

epilogue 的选择在 `:382-406`：

```cpp
if constexpr (kSwapAB) {
    const auto effective_m = scheduler.get_aligned_effective_m_in_block(m_block_idx);
    epilogue::sm100_store_cd_swap_ab<BLOCK_M, BLOCK_N, STORE_BLOCK_M, STORE_BLOCK_N, ...>(
        smem, tma_stage_idx, tmem_base_addr, base_m_idx, base_n_idx, current_group_idx, is_empty_group,
        effective_m, epilogue_warp_idx, lane_idx, epilogue_operator, false, ...);              // :385-397
} else {
    epilogue::sm100_store_cd<...>(...);                                                       // :400-412
}
```

`sm100_store_cd_swap_ab.cuh` 与非 swap 版的根本差别是**「TMEM 的列方向 = M、行方向 = N」**：

- **循环单元是 M**（`:60`）：`num_stores = effective_m / STORE_BLOCK_M`，即按 16 行 M 一块块搬。
- **一次从 TMEM 读 8 行**（`:69`）：`kNumTmemLoads = STORE_BLOCK_M / kNumSwizzleAtomRows`（16/8=2）。
- **转置 STSM**（`:156-165`）：FP8 输出用 `SM100_U8x8_STSM_T<uint32_t>`（`_T` = transposed），注释
  「the loaded fragment matches `stmatrix.m16n8.trans.b8`」。BF16 用 `SM90_U32x4_STSM_T<int>`
  （`:233-238`）也是转置的。
- **TMA store 的坐标是 `(n_idx, m_idx)`**（`:249-256`）：`n_idx = base_n_idx + i*STORE_BLOCK_N_ATOM`、
  `m_idx = base_m_idx + store_idx*STORE_BLOCK_M`，因为是 2D/3D `SM90_TMA_STORE`，写回全局 D 时坐标
  已经是「N 在前、M 在后」——转置在 smem→global 这一跳最终完成。

两条 swap 专属的额外约束（`:42-48`）：

```
DG_STATIC_ASSERT(not kWithOutputSF or (STORE_BLOCK_M == 16 and kNumUMMAStoreThreads == 128),
                 "Swap-AB FP8 output requires one warpgroup and a 16x128 store shape");
DG_STATIC_ASSERT(not kWithStochasticRounding or (is bf16 and not kWithAccumulation), ...);
```

即 **swap 下的 FP8 输出**（`QuantizeToFP8`，SM100 `Kernel1D1D` 的 FP8 输出对）必须正好是
`16×128` 的 store 形状 + 一个 warpgroup；**随机舍入**（`StochasticRoundToBF16`）在 swap 下最复杂——
因为 `.16x256b` 读回时，一条 lane 手里是「相邻两行 M 的同一 N 值」，per-quartet 的哈希要在
「间隔 4 条 lane」之间用 `__shfl_xor_sync` 逐级归约（`:206-224`），而这套哈希的常数与
`StochasticRoundToBF16`（第 31 篇）同源。

---

## 6. 一个会咬人的数字：`mk_alignment_for_contiguous_layout`

m-grouped 分支里 `block_m = heuristics_runtime->get_mk_alignment_for_contiguous_layout()`（`:40`）。
这个值不是常量，而是运行时配置（`heuristics/runtime.hpp`）：

- 默认 `kLegacyMKAlignmentForContiguousLayout = 128`（`:11`）；
- SM100 的「理论值」是 **256**（`get_theoretical_mk_alignment_for_contiguous_layout`，`runtime.hpp:59-67`：
  注释说 UMMA_N=256 允许固定 256 对齐，对 MoE cast / m-grouped / k-grouped 更友好，`expected_m` 被忽略）。

因此**同一份 m-grouped 形状，`block_m` 可能是 128 也可能是 256**，取决于 runtime 有没有把
`mk_alignment` 设成 256；`block_m` 又直接决定 `store_block_m`、`smem_cd`、`num_stages` 与 TMEM 列预算。
本文的 H100 实验对两种取值都扫了（见 §7）。

另外，`block_m` 还进 JIT 模板实参——`sm100_bf16_gemm.hpp:66` 附近的
`heuristics_runtime->get_mk_alignment_for_contiguous_layout()` 被直接格式化成模板常量
（对照第 23/33 篇的 DeepJIT 缓存键：对齐变了 → 模板常量变了 → 编译产物不同）。这是「配置改动会
静默改变 kernel 形态」的又一处落点。

---

## 7. H100 实测：host 启发式真机 + 转置等价

**主 kernel 是 SM100 专属、本机跑不了**；但「选 layout / 算 storage / 算 pipeline / 算 launch」全是
纯 host C++。实验 `src/sm100_swap_ab_dgrad.cu`（`harness/exp_sm100_swap_ab_dgrad.sh`，nvcc 直接
include 真实头文件 + torch）做了三件事：

### Part A —— 真机跑 `SM100ArchSpec::get_layout_candidates`

对 `m-grouped contiguous BF16`（`GemmType::MGroupedContiguous`）喂若干真实形状，断言：

- 候选**恰好 1 个**且 `swap_ab==1`、`block_n==128`、`cluster_m==1`；
- `block_m` 随 `heuristics_runtime->set_mk_alignment_for_contiguous_layout(128/256)` 逐位变化；
- `cluster_n` 的取值符合公式 `ceil_div(n,128)%2==0 and num_sms%2==0 ? 2 : 1`（含 `n=7168`→2、
  奇数倍 `n=128*57`→1 的对照）。
- 独立复刻一份 Python/表格式 oracle（在 C++ 里手写同一公式），与真实 header 的输出**逐字段相等**。

### Part B —— 真机跑 `get_storage_config` / `get_pipeline_config` / `get_launch_config`

对 `block_m ∈ {128,256}`、`cluster_n=2` 的组合，断言真实 header 返回的
`load_block_m = block_m/2`、`store_block_m = 16`、`store_block_n = 128`、
`swizzle_a = swizzle_b = swizzle_cd = 128`、`num_threads = 256`、
`num_non_epilogue_threads = 128`、`num_epilogue_threads = 128`、
`smem_cd = 16*128*sizeof(bf16)*2` 等。并验证 `get_layout_candidates` 在 `GemmType::Normal`
时**同时枚举 swap_ab∈{0,1}**（对照组）。

### Part C —— 数值佐证「swap = 转置」

swap 分支的语义等价于「把 `D=A·Bᵀ` 算成 `Dᵀ=B·Aᵀ`」。本机用 torch 在 H100 上对随机矩阵验证：

```
(A @ B.T).T  ==  B @ A.T          # 固定形状下逐位相等
```

以及 m-grouped 的逐组形式（每组 M 不等）：`(A_e @ B_e.T).T` 与 `B_e @ A_e.T` 的 `max|Δ|`
在 fp32 下 **9.5e-06**（形状不同 cuBLAS 会选不同 tiling，故取容差而非逐位）。这不能用 SM100 的
`UMMA_N` 动态行为证，但它坐实了「转置写回」这一步在数学上是无损的——swap 分支唯一要小心的只是
layout/精度，而不是数学。

### Part D —— SM90 对照：SM90 上根本没有 swap

断言 `SM90ArchSpec::get_layout_candidates` 对 m-grouped BF16 给的候选**全部** `swap_ab==0`
（48 个候选），并对照源码 `heuristics/sm90.hpp:125` 的 `DG_HOST_ASSERT(layout.swap_ab == 0)`。
这解释了为什么「MoE dgrad 在 H100 上走的是非 swap 的 m-grouped，在 SM100 上才被迫/主动换轴」——
第 13 篇与本文合起来才是完整的 dgrad 形态。

**实测总计 56 PASS / 0 FAIL**（`harness/exp_sm100_swap_ab_dgrad.out.txt`，H100 80GB HBM3、
CUDA 13.2、DeepGEMM `057ca59`）。关键真机数字：m-grouped `(4096,7168,2048,G=8)` 选出
`swap_ab=1, block_m=128, block_n=128, block_k=64, cluster=(1,2)`，`num_stages=9`、`smem_size=230204`；
把 `mk_alignment` 改成 256 后 `block_m=256`、`num_stages` 掉到 6、`smem_size=205628`；稠密 BF16
`(4096,7168,2048)` 枚举出 50 个候选、最优落回 `swap_ab=0`。

---

## 8. 结论与边界

- **一句话**：`kSwapAB` 是 SM100 m-grouped（含 MoE dgrad）的**唯一 layout**；它把「逐组可变、16 对齐的
  短 M 维」从 `UMMA_M` 挪到 `UMMA_N`，把 `UMMA_M=128` 让给固定的大 N 维，代价是 epilogue 必须
  **转置写回**（`sm100_store_cd_swap_ab`：16×128 store、转置 STSM、`(n_idx,m_idx)` 的 TMA 坐标）。
- **host 侧硬约束**：`block_n==128`（=layout_ad_m）、`cluster_m==1`、`block_m` 取
  `mk_alignment_for_contiguous_layout`（128 或 256）、`kNumUMMAStoreThreads=128`。
- **边界**：主 kernel 是 SM100 专属，UMMA/TMEM/`.16x256b`/转置 STSM 全部**仅分析**；本文的实测
  只用真实 header 的 host 逻辑 + torch 上的转置等价，**没有**、也无法在 H100 上编译/运行这颗 kernel。
- **易错点**：① 把 `block_m`（对齐）当成 128 写死——它在 SM100 可能是 256；② 把 `store_block_m=16`
  当成「块 M 被拆成 16」——它其实是 `UMMA_N` 的最小原子；③ 以为 swap 只是「把 A、B 参数位置对调」——
  它还改 instr_desc 的 major 顺序、把 `UMMA_N` 变成动态、并把整条 epilogue 换成转置版。

---

## 9. 引用

- DeepGEMM（commit `057ca59`，Public release 26/09/30）：
  `deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh:66-88/177/237-240/286/310-315/382-412`、
  `deep_gemm/include/deep_gemm/epilogue/sm100_store_cd_swap_ab.cuh:16-66/69-70/107-165/206-238/240-256`、
  `deep_gemm/include/deep_gemm/scheduler/gemm.cuh:177-184`；
  `csrc/jit_kernels/heuristics/sm100.hpp:30-46/81-91/121-132/165-173/203-215/254-258`、
  `csrc/jit_kernels/heuristics/sm90.hpp:125`、`csrc/jit_kernels/heuristics/runtime.hpp:11/59-67`、
  `csrc/jit_kernels/heuristics/config.hpp:16-135`、`csrc/jit_kernels/impls/sm100_bf16_gemm.hpp:22-66/126-...`。
- 线上资料：DeepGEMM GitHub PR [#316](https://github.com/deepseek-ai/DeepGEMM/pull/316)（Mega MoE 与优化）、
  [#95](https://github.com/deepseek-ai/DeepGEMM/pull/95)（MoE backward）。
- 本专题：第 12 篇（wgrad `k_grouped`）、第 13 篇（dgrad `m_grouped`）、第 14 篇（反向 kernel）、
  第 19 篇（SM90 BF16）、第 36 篇（SM100 BF16 主 kernel 与 `set_tc_util`）、
  第 31 篇（SM100 epilogue 算子）、第 92 篇（smem 布局结构层）。
- 实验：`src/sm100_swap_ab_dgrad.cu` + `harness/exp_sm100_swap_ab_dgrad.{sh,out.txt}`。
