# 32 · DeepGEMM 的 SM90 FP8 dense GEMM 主 kernel：1D1D vs 1D2D 精读 + H100 实测

> 第 29 篇补上了 SM90 的 **BF16** 主 kernel。这一篇回到 FP8——DeepGEMM 真正的立身之本——把 SM90 上
> **两个** FP8 稠密 GEMM 主 kernel 系统讲清：`sm90_fp8_gemm_1d1d.cuh` 与 `sm90_fp8_gemm_1d2d.cuh`。
> 前面第 10（SM100 mega MoE MMA）、12（wgrad K 分组）、13（dgrad）、21（FP8 einsum）、25（SF 布局）
> 篇都从不同侧面碰到过它们，但**没有一篇把「为什么有 1D1D/1D2D 两个 kernel、分派规则、两个 kernel
> 的骨架差异」讲完整**。
>
> 一句话结论：**两个 kernel 计算同一件事 `D = (A·sfa) @ (B·sfb)ᵀ`，区别只在 B（权重）的 block-scale
> 粒度**——`1D1D` 是「A、B 都按输出行做 per-token 1D scale」，`1D2D` 是「A per-token 1D + B 按
> 128×128 块做 2D scale」。粒度不同导致 SF 的 TMA/搬运/累加方式整套不同，于是拆成两个五脏俱全的
> SM90 kernel。
>
> **本文所有行号均以 commit `057ca59`（Public release 26/09/30）核对过。**
> 配套实验：`harness/exp_fp8_sm90_gemm.py` / `.sh` / `.out.txt`（H100 真机，编译运行 SM90 真实设备代码）。

---

## 1. 为什么是「两个」SM90 FP8 主 kernel

DeepGEMM 的 FP8 GEMM 是 **block-scaled**：每个 FP8 元素先除以一个 2 的幂（UE8M0）或 fp32 （非 UE8M0）
scale 再量化，乘积在 epilogue 里用 scale 复原。scale 的**粒度**决定 SF 张量形状、TMA 描述符、SMEM
布局与累加公式，于是分成两条实现：

| | `sm90_fp8_gemm_1d1d` | `sm90_fp8_gemm_1d2d` |
|---|---|---|
| SFA 粒度 | 1D，per-M-row（`gran_mn=1`，`gran_k=128`） | 同左（A 恒 per-token） |
| **SFB 粒度** | **1D，per-N-row**（`gran_mn=1`） | **2D，per-128×128 块**（`gran_mn=128`） |
| C/D dtype | **float，且必须 accumulate**（`hpp:84`） | **bf16，禁止 accumulate**（`hpp:81`） |
| DMM 累加方式 | 硬件 TMA **reduce-add** | 直接 TMA store |
| WGMMA | `MMA_64xNx32_F32E4M3E4M3_SS`（`K=32`） | 同左 |
| 支持的 `GemmType` | `Normal`、`KGroupedContiguous`（`cuh:52`） | `Normal`、`MGrouped{Contiguous,Psum,Masked}`、`Batched` |
| host 入口 | `sm90_fp8_gemm_1d1d` / `sm90_k_grouped_fp8_gemm_1d1d` | `sm90_fp8_gemm_1d2d` / `_m_grouped_*_1d2d` / `sm90_fp8_bmm` |
| `BLOCK_N` 启发式上限 | 160（fp32 输出） | 192（bf16 输出） |

两者共用 `mma/sm90.cuh:42-59` 的 `FP8MMASelector`（M=64、K=32）与 `scheduler/gemm.cuh` 的调度器。

### 1.1 分派规则（host）

`fp8_fp4_gemm_nt` 在 SM90 上按 **B 的 SF 粒度 `gran_n`** 二选一（`gemm.hpp:117-126`）：

```
if (arch_major == 9 and sfa.scalar_type() == torch::kFloat) {     // :117
    const int gran_n = recipe.has_value() ? std::get<1>(recipe.value())
                                          : std::get<0>(recipe_b.value());   // :120
    if (gran_n == 1) sm90_fp8_gemm_1d1d(...);                     // :122
    else             sm90_fp8_gemm_1d2d(...);                     // :125
}
```

- 用户不传 recipe 时走默认 recipe；`get_default_recipe`（`utils/layout.hpp:86-98`）在 **SM90 上唯一
  返回 `(1,128,128)`** → `gran_n=128` → **普通稠密 `fp8_gemm_nt` 默认走 1D2D**。
- 传 `recipe=(1,1,128)`（3-tuple）→ `gran_n=1` → 1D1D。
- 分组/batched 入口里 `m_grouped`、`bmm` 恒调 1D2D；`k_grouped`（wgrad）恒调 1D1D
  （`gemm.hpp:239/309/422`，见第 12 篇）。

这解释了 `tests/generators.py:132-137 get_kernel_types`：**SM90 默认返回 `Kernel1D2D`**，
而 `get_recipes` 只在 `is_wgrad` 时给 `(1,1,128)` → 1D1D。

### 1.2 为什么需要 SFB 的两种粒度

- **2D 块 scale（1D2D）**是量化感知训练/推理里最省 SF 显存的布局：把 128 个连续 N 行共享一个 K-block
  scale，SF 张量只有 `[n/128, k/128]`（本实验 `n=k=4096` 时 `32×32`，而 1D1D 是 `4096×32`）。
- **1D per-row scale（1D1D）**是 **wgrad** 的自然形态：MoE 反向里权重梯度 `dW` 的「输出行」是权重通道，
  每个通道一个 scale，配合 K 轴按专家分组（`KGroupedContiguous`）。它必须 **fp32 输出 + 累加**，因为
  K 被切成多个组、多个 launch，需要跨组/跨调用累加（第 12 篇）。

---

## 2. 1D2D kernel 精读（`sm90_fp8_gemm_1d2d.cuh`）

### 2.1 模板与静态契约（`:37-90`）

模板参数（`:37-46`）：`kMajorSFB`、`SHAPE_{M,N,K}`、`kNumGroups`、`BLOCK_M/N/K`、三个 swizzle、
`kNumStages`、TMA/math 线程数、`kNumTMAMulticast`/`kIsTMAMulticastOnA`、`kNumSMs`、`GemmType`、`cd_dtype_t`。

关键静态断言：

- `BLOCK_K == 128`，**「Only support per-128-channel FP8 scaling」**（`:56`）——这是全篇缩放语义的根。
- 「Too much B scales in a single block」（`:57-59`）：`ceil_div(BLOCK_N, BLOCK_K)==1 或 gcd(BLOCK_N,BLOCK_K)==BLOCK_N-BLOCK_K`。
  也就是说一个 `BLOCK_N` 最多跨**两个** 128 列 scale 组（见 §2.3 的 `scale_b_0/1`）。
- **`cd_dtype_t` 必须是 bf16**（`:62`）——1D2D 不累加，直接产 bf16。
- `BLOCK_M % WGMMA::M == 0 或 BLOCK_M < WGMMA::M`（`:67`）——允许 `BLOCK_M<64` 时一个 warpgroup 只发部分。

### 2.2 SMEM 布局与 D 的双缓冲（`:74-124`）

```
[ smem_d (BLOCK_M*BLOCK_N*2B, 1024B 对齐) ][ A × kNumStages ][ B × kNumStages ]
   [ SFA × kNumStages (float, 每 block_m) ][ SFB (单份, 非流水) ][ barriers ]
```

- `smem_d` 是 bf16 输出暂存（`:108`），store 时 `WGMMA→STSM→smem_d→TMA`。
- **SFA 随流水 multi-stage，SFB 只单份**（`:119`）：A 的 scale 会随 K block 前进被消费；B 的 2D 块
  scale 由 math warp **从 global 直读**（不走 TMA 流水，见 §2.3），故 SMEM 只留一份。
- barrier init（`:128-139`）：`full.init(1)`（TMA 生产者一个 elect-one 线程 arrival）、
  `empty.init(kNumTMAMulticast * kNumMathThreads/32)`——**空 barrier 的到达数是「多播 CTA 数 × math
  warp 数」**，因为每个 math warp 完成 MMA 后都要来 release 一次。

### 2.3 三类 warp 分工

**TMA warp（`:165-212`）**：由于 `BLOCK_M` 可能是 32，warp0/1 可能在发 WGMMA，所以这里特意用
**第三个** TMA warp 做 elect-one（`:171`，注释 `:170` 明说）。每个 `(m_block,n_block)` 上：

- `get_global_idx<kWithGroupOffsetA>`（`:193`）给 A 取址（masked 时带组偏移，`kWithGroupOffsetA` `:189`）；
- SFA 单独一条 TMA，按 `k_block` 取第 `get_global_idx<..., SF_K>`（`:196`）个 scale 块；
- B 用 `get_global_idx<true>`（`:201`）并带 `kNumTMAMulticast`；`full_barrier.arrive_and_expect_tx(...)`
  一次性等 A+B+SFA 三块字节（`:203`，**注意 SFB 不占 TMA**）。

**Math warps（`:213-442`）**：

- **B scales 由 math warp 从 global 直读**（`:237-248`），与 TMA store 重叠（注释 `:238`）。这里处理
  `num_former_iters`（`:230-235`）：当 `BLOCK_N` 不整除 `BLOCK_K` 时，`BLOCK_N` 会跨 128 scale 组
  边界，需要读「两行」B scale。
- 用一个局部 `make_smem_desc` 预计算 A/B 描述符低 32 位（`:221-224`），K 循环里只做加法偏移
  （`:311-312`）= 1D2D 的 descriptor 前进方式。
- 累加再乘 scale：`scale_a`（per-M-row，两个寄存器 `r_0/r_1` `:301-302`）× `scale_b`（`:331-334`），
  对每个 4 元素组用 **predicate** 选择 `scale_b_0`/`scale_b_1`（`:340-344`，注释 `:330` 强调
  「用 predicate 比两个循环快」）。
- `empty_barrier_arrive`（`:262-269`）：多播时 `is_peer_cta_alive ? lane_idx : block_rank_in_cluster()`。

**Epilogue（`:357-441`）**：STSM 把累加器（bf16x2）按 swizzle 写进 `smem_d`（`:378-420`），再
TMA store 2D（`:437`）或 batched 3D（`:434`）。`kSwizzleDMode` 决定写回地址算法：swizzle>0 时算
bank group/row/col 再异或（`:385-407`），否则裸 padding（`:410`）。

---

## 3. 1D1D kernel 精读（`sm90_fp8_gemm_1d1d.cuh`）

### 3.1 模板与静态契约（`:29-60`）

- `kNumTMAThreads == 128 and kNumMathThreads % 128 == 0`（`:50`）——1D1D 固定 128 个 TMA 线程。
- `BLOCK_K == 128`（`:51`）、`GemmType ∈ {Normal, KGroupedContiguous}`（`:52`）。
- **`cd_dtype_t` 必须是 float**（`:55`）——与 1D2D 的 bf16 恰好互补。

### 3.2 SMEM 与「K 分组改 tensormap」（`:67-124`）

1D1D 的 SMEM 比 1D2D 多一块 **`SMEM_TENSOR_MAP_SIZE`**（`:68`，仅 KGrouped 时非 0），用于在
**SHEM 里原地改写 A/B 的 TMA 描述符**。这正是第 12 篇讲的 wgrad 关键技巧：

- 基础描述符只能按**首个非空组**的 K 造（host `hpp:188-195` 用 `first_k`）；
- kernel 切到新组时（`:193-217`）用 `ptx::tensor_map_replace_global_addr_in_smem` 把 A/B 全局地址
  改成 `gmem + current_k_start*shape_m`（`:198-199`），并用
  `tensor_map_replace_global_inner_dim_stride_in_smem` 把 inner-dim stride 改成当前组的 `current_shape_k`
  （`:200-201`）；随后 `tma_desc_commit_group/wait_group`（`:204-205`）+ `tensor_map_release/acquire_gpu`
  （`:212-216`）防竞态。SF 的取址则走 **`current_sf_k_cumsum`**（`:228`），与物理 K 解耦（第 16 篇）。

### 3.3 流水 unroll 与寄存器重配（`:149-154`）

```
kNumPipelineUnrolls = (kGemmType == KGroupedContiguous ? 0 : kNumStages);   // :150
kNumTMARegisters  = (unroll==0 ? 40 : 24);   kNumMathRegisters = (unroll==0 ? 232 : 240);  // :153-154
```

KGrouped 时 `shape_k` 逐组动态，无法编译期展开 K 循环，所以 **`kNumPipelineUnrolls=0`**，并把更多
寄存器让给 TMA。普通 `Normal` 则按 `kNumStages` 展开 K 循环。

### 3.4 Math / epilogue

- TMA warp 是 **第一个** TMA warp（`:174` `warp_idx == kNumMathThreads/32`，与 1D2D 用第三个不同）。
- math warp 里 **每个 k 重算 `make_smem_desc`**（`:303-304`），而非 1D2D 的预计算+偏移。
- `BLOCK_M == WGMMA::M * (BLOCK_M<=64 ? 1 : 2)`（`:263`）。
- B scale 用 `float2` 一次读两个 n 的 scale（`:294`），A scale 两个寄存器（`:288-289`），
  promote 到 `final_accum`（`:318-325`）。
- epilogue：**直接 `st.shared` float2**（`:334-340`，无 swizzle，行 stride=BLOCK_N），
  `NamedBarrier::sync(128, math_wg_idx)` 后由一个 elect-one 线程发
  **`cute::SM90_TMA_REDUCE_ADD_2D`**（`:346-348`）——**这就是 1D1D 必须 fp32 + accumulate 的根**：
  它把结果 `+=` 进 D，D 的坐标是 `current_group_idx*shape_m + m_block*BLOCK_M + r_0`，K 分组时每组
  落在 D 的不同区段。

---

## 4. 对照表：1D1D vs 1D2D 的十处差异

| # | 维度 | 1D1D | 1D2D |
|---|---|---|---|
| 1 | SFB 粒度 | per-N-row（`[n,k/128]`） | per-128×128 块（`[n/128,k/128]`） |
| 2 | D dtype | float | bf16 |
| 3 | accumulate | 必须（`hpp:84`） | 禁止（`hpp:81`） |
| 4 | store | `SM90_TMA_REDUCE_ADD_2D`（`:346`） | `SM90_TMA_STORE_2D/3D`（`:437/434`） |
| 5 | SMEM D 暂存 | float2 裸写，无 swizzle | bf16 STSM + 128B swizzle |
| 6 | TMA warp | 第 1 个（`:174`） | 第 3 个（`:171`） |
| 7 | B scale 来源 | TMA 流水（每 stage） | math warp 从 global 直读（`:239-248`） |
| 8 | descriptor | 每 k 重算（`:303`） | 预计算+加法偏移（`:221-224`） |
| 9 | 支持 GemmType | Normal / KGroupedContiguous | Normal / MGrouped / Batched |
| 10 | `BLOCK_N` 上限 | 160（fp32 避 bank conflict，`sm90.hpp:42-54`） | 192 |

第 10 条来自选核启发式（第 18 篇）：`heuristics/sm90.hpp:42-48` 对 **1D1D + fp32 输出**把
`block_n` 起点设为 **24**（避开 STSM 写 fp32 时的 bank conflict），`end` 分别 160（1D1D）/192（1D2D）；
`block_m` 候选在 fp32 输出时**不含 256**（`:29-30`）。本实验抓到的实参正好吻合（§5）。

---

## 5. H100 实测（`harness/exp_fp8_sm90_gemm.py` / `.out.txt`）

### 5.1 分派与契约

用 `torch.profiler` 抓真实发射的 kernel：

- 默认稠密 `fp8_fp4_gemm_nt`（`m=n=k=4096`）→
  `sm90_fp8_gemm_1d2d_impl<(cute::UMMA::Major)0, 0u, 4096u, 4096u, 1u, 256u, 128u, 128u, 128u, 128u,
  128u, 3u, 128u, 256u, 2u, true, 132u, GemmType(0), bfloat16_t>`：`BLOCK_M=256, BLOCK_N=128,
  BLOCK_K=128, stages=3, multicast=2(on A)`。B 的 SF 形状 `32×32=[n/128,k/128]`。
- `recipe=(1,1,128)` + accumulate（D=float）→
  `sm90_fp8_gemm_1d1d_impl<0u, 4096u, 4096u, 1u, 128u, 152u, 128u, 128u, 128u, 4u, 128u, 256u, 2u,
  false, 132u, GemmType(0), float>`：`BLOCK_M=128, BLOCK_N=152, stages=4`。B 的 SF 形状
  `4096×32=[n,k/128]`。——注意 **`BLOCK_N=152` 正是 §4 第 10 条的 fp32 起点 24 派生**，`BLOCK_M` 没有 256。

契约反例被精确拦在 host 断言：

```
1d1d 无 c / D=bf16 被拒: Assertion error (sm90_fp8_gemm_1d1d.hpp:84): c.has_value()
1d2d + accumulate  被拒: Assertion error (sm90_fp8_gemm_1d2d.hpp:81): not c.has_value()
```

### 5.2 语义与交叉验证

把 A/B 用 DeepGEMM 自己的 `per_token_cast_to_fp8` / `per_block_cast_to_fp8` 量化，再把量化结果**反量化**
回 fp32 做参考 matmul（隔离出「量化误差」与「kernel 误差」两层）：

| m | n | k | 1d2d relerr | 1d1d relerr |
|---|---|---|---|---|
| 1024 | 1024 | 1024 | 6.90e-4 | 1.27e-4 |
| 2048 | 4096 | 2048 | 6.92e-4 | 1.27e-4 |
| 4096 | 7168 | 7168 | 6.92e-4 | 1.27e-4 |
| 128 | 4096 | 4096 | 6.85e-4 | 1.27e-4 |

- **1d2d ~6.9e-4** = bf16 输出舍入量级（epilogue 把 fp32 累加舍成 bf16）；
- **1d1d ~1.27e-4** 小 5×，因为它的 D 是 **fp32**，只剩下 FP8 量化本身的误差。

**交叉验证（坐实「同一数学」）**：把 1d2d 的 128 块 scale 按行广播成 per-column，再喂给 1d1d，两者
输出 **逐位相等（relerr=0.000e+00）**：

| m | n | k | 1d2d vs 1d1d(广播) relerr |
|---|---|---|---|
| 1024 | 1024 | 1024 | 0.000e+00 |
| 4096 | 4096 | 4096 | 0.000e+00 |
| 2048 | 4096 | 2048 | 0.000e+00 |

这正是 §1 结论的硬证据：**两个 kernel 只差 SFB 粒度**；粒度对齐后 bf16(1d2d) == bf16(fp32 的 1d1d)。

### 5.3 性能

| m | n | k | 1d2d us | 1d2d TFLOPS | 1d1d us | 1d1d TFLOPS |
|---|---|---|---|---|---|---|
| 4096 | 7168 | 7168 | 299.2 | **1407** | 439.7 | 957 |
| 4096 | 4096 | 4096 | 108.8 | 1264 | 175.1 | 785 |
| 8192 | 4096 | 4096 | 203.2 | 1353 | 315.7 | 871 |
| 2048 | 4096 | 2048 | 36.0 | 954 | 69.7 | 493 |
| 128 | 7168 | 2048 | 14.2 | 265 | 20.4 | 185 |

**1d2d 一致快 ~1.5×**，原因有三：① D 是 bf16、无 TMA reduce-add 回读；② `BLOCK_M=256`（1d1d 因
fp32 输出只能 128）；③ SFB 从 global 直读、SMEM 省了一份流水。这也解释了为什么**推理前向默认走 1d2d**：
它是纯 GEMM 的吞吐最优路径；而 wgrad 需要的「fp32 + 累加 + K 分组」只有 1d1d 能给。

---

## 6. 与专题其余部分的关联

- **MoE 反向**：dgrad（`m_grouped_*`）走 **1D2D**（M 轴分组、bf16 输出），wgrad（`k_grouped_*`）走
  **1D1D**（K 轴分组、fp32 累加）——第 12/13/14 篇的落点就在这两个 kernel 上。
- **FP8 einsum**：MLA 按头 batched 投影走 1D2D 的 `GemmType::Batched`（第 21 篇）。
- **SM100**：Mega MoE / 全部 SM100 GEMM 统一走 `sm100_fp8_fp4_gemm_1d1d`（第 3/10 篇），SM90 的
  **两个** kernel 在架构上市被一个「1D1D（支持 block-scale + FP4）」取代。本文讲的粒度/累加差异在
  SM100 上变成 UMMA + TMEM 的 SF 布局问题（第 21/25 篇）。
- **SF 布局**：两个 kernel 的 SFA 恒 per-token → 都要经 `get_mn_major_tma_aligned_tensor` 的
  `transpose_fp32`（本实验 profiler 里也抓到 `transpose_fp32<512,64,32,128,33>`）；这是第 25 篇的分派之一。

---

## 7. 引用链接（真实核对）

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM （公开版 commit `057ca59`）
- DeepGEMM PR #95：https://github.com/deepseek-ai/DeepGEMM/pull/95
- DeepGEMM PR #316：https://github.com/deepseek-ai/DeepGEMM/pull/316
- 源码（commit `057ca59`）：
  - `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`
  - `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`
  - `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp`、`sm90_fp8_gemm_1d2d.hpp`
  - `deep_gemm/include/deep_gemm/mma/sm90.cuh`、`scheduler/gemm.cuh`
  - `csrc/apis/gemm.hpp`、`csrc/apis/layout.hpp`、`csrc/utils/layout.hpp`
  - `csrc/jit_kernels/heuristics/sm90.hpp`

> 说明：本机为 H100（sm90），本篇两个 kernel 都是 **SM90 真实设备代码**，实验为真机数字。
> 文中未涉及 SM100 特有语义的推断均只针对 SM90。
