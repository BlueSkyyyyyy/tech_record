# MoE 反向的 wgrad API：`k_grouped_*_contiguous` 精读

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 12 篇（路线图 **P4-2**）。
> P4-1（MoE 反向数学与 grouped GEMM 映射，见上一篇）已把反向推导清楚，并指出 **wgrad 天然是 K 轴分组
> （`k_grouped`）**：各专家要收缩的 token 数不等长，落在收缩维 K 上，而 M/N 固定。本文逐行精读
> DeepGEMM 里实现这一族的两套 API——**SM100 的 `k_grouped_fp8_gemm_tn_contiguous`** 与
> **SM90 的 `k_grouped_fp8_gemm_nt_contiguous`**——讲清 `tn`/`nt` 的 major 差异、`ks_cpu`/`grouped_layout`
> 两种编码（contiguous / PSUM）、SF 的分组打包，以及调度器如何按组切 K 块。
>
> 约束提醒：`tn` 版是 **SM100 专属，仅分析**（H100 调用直接 `DG_HOST_UNREACHABLE`）；`nt` 版是
> DeepGEMM 给 Hopper 的实现，**H100 可跑**，本文用它做了真实 API 语义与数值核对
> （`harness/exp_kgrouped_api.py` + `.out.txt`）。

## 1. 从数学到 API：wgrad 的形状契约

P4-1 的 (3)(5) 给出 wgrad：

$$
dW^1_j=\sum_t x_t^{\mathsf T}\,dG_{tj}\quad(M=H,\;N=2I,\;K=\text{token}_j),\qquad
dW^2_j=\sum_t A_{tj}^{\mathsf T}\,dY_{tj}\quad(M=I,\;N=H,\;K=\text{token}_j).
$$

把所有专家按 K（token 数 $n_j$）拼起来：A、B 都是 `[sum_k, ·]` 的"行按专家分组"矩阵，输出 `D` 是
`[E, M, N]` 三块拼接。DeepGEMM README 第 80 行把这件事一句话概括：

> "We also provide a K-axis-grouped API for MoE weight backward (with M and N must remain fixed),
> please refer to `k_grouped_fp8_gemm_tn_contiguous` for more information."
> （`~/github/DeepGEMM/README.md:80`）

host 侧两个入口在 `csrc/apis/gemm.hpp`：
- `k_grouped_fp8_gemm_tn_contiguous`（`:320-371`）——**SM100 专属**；
- `k_grouped_fp8_gemm_nt_contiguous`（`:373-427`）——**SM90**；
- 另有 FP4 版 `k_grouped_fp4_gemm_nt_contiguous`（`:429` 起）。

两者对外签名几乎一样（pybind 注册见 `gemm.hpp:895-918`），都收 `(a, b, d, ks_cpu, grouped_layout,
c, recipe, compiled_dims, use_psum_layout, epilogue)`，`recipe` 默认 `(1, 1, 128)`（`gran_mn=1,
gran_k=128`）。真正的分叉在 **A/B 的物理 major 布局**。

## 2. `tn` vs `nt`：同一个数学，两种物理布局

先给结论（下表）：

| 入口 | 架构 | A / B 物理布局 | major | D | 说明 |
|---|---|---|---|---|---|
| `..._tn_contiguous` | SM100（tcgen05） | `[sum_k, M]` / `[sum_k, N]` | MN（M/N 连续） | `[E,M,N]` | wgrad 的"自然"布局；**仅分析** |
| `..._nt_contiguous` | SM90（WGMMA） | 每组 `[M, k_g]` / `[N, k_g]` 拼接 | K（K 连续） | `[E,M,N]` | H100 可跑；需按组改写 tensormap |

### 2.1 SM100 `tn`：A/B 转置、MN-major

`k_grouped_fp8_gemm_tn_contiguous` 的注释与形状检查（`gemm.hpp:340-349`）：

```cpp
// A/B use MN-major layouts `[sum_k, M]` and `[sum_k, N]`
const auto [num_groups, m, n] = get_shape<3>(d);
const auto [sum_k_ , m_] = get_shape<2>(a.first);
const auto [sum_k__, n_] = get_shape<2>(b.first);
const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups,
                                       use_psum_layout, k_alignment, static_cast<int>(a.first.size(0)));
DG_HOST_ASSERT(m == m_ and n == n_ and sum_k == sum_k_ and sum_k == sum_k__);
```

即 A、B 是二维 `[sum_k, M]` / `[sum_k, N]`，"MN-major" 指 M/N 维连续（`stride(-1)==1`，
判定见 `csrc/utils/layout.hpp:30-33`）。注意名字 `tn`：在 DeepGEMM 的约定里 `tn` 表示 D=AᵀB
（A 转置、B 不转置），恰好就是 wgrad 的 `AᵀB`。用 2.1 节实测（B 部分）验证：把每组激活按
`[sum_k, M]`（M 连续）沿 K 拼接，逐组算 `A_eᵀB_e` 与自然布局参考 **完全一致（0 误差）**。

分发的收尾（`gemm.hpp:360-370`）：

```cpp
if (arch_major == 10) {
    const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
    const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
    sm100_k_grouped_fp8_gemm_1d1d(a.first, sfa, b.first, sfb, c, d, m, n, grouped_layout, gran_k, k_alignment,
                                  cute::UMMA::Major::MN, cute::UMMA::Major::MN, compiled_dims, use_psum_layout,
                                  resolve_epilogue_class(epilogue_class, c, d));
} else {
    DG_HOST_UNREACHABLE("Unsupported architecture");   // sm90 走这一支
}
```

`sm100_k_grouped_fp8_gemm_1d1d`（`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325-403`）
强制 `Major::MN`（`:337`），GEMM type 按是否 PSUM 选
`KGroupedContiguous` / `KGroupedContiguousWithPsumLayout`（`:343`），并为 A/B 造 MN-major 的
TMA 描述符（`:360-369`，K 长度用 `sum_k = a.size(0)`）。它最终落到 tcgen05 的 1D1D kernel，
与 Mega MoE 前向里"把 A/B 互换 + 2-CTA"是同一套 UMMA 机制（见 `docs/10`）。

### 2.2 SM90 `nt`：K-major，按组原地改 tensormap

Hopper 上没有 tcgen05，DeepGEMM 给的是 `nt` 版。它的形状检查很不一样（`gemm.hpp:391-397`）：

```cpp
const auto sum_mk = a.first.numel();
const auto sum_nk = b.first.numel();
const int sum_k = check_k_grouped_args(ks_cpu, grouped_layout, num_groups, use_psum_layout, 128);
DG_HOST_ASSERT(sum_mk == static_cast<int64_t>(sum_k) * m);
DG_HOST_ASSERT(sum_nk == static_cast<int64_t>(sum_k) * n);
```

A/B 在这里是**一维扁平缓冲**（只查 `numel()`）：每组把该专家的 `[M, k_g]` 以 **K 连续**写成
`[M, k_g]` 的 row-major（`x[start:end].T.flatten()`，测试生成器
`tests/generators.py:535-539`），各组首尾相接。于是**沿 K 方向每跨一组，A 的基址就整体前移
`k_start · M` 个元素**。这正是 SM90 kernel 要"按组改写 tensormap"的原因。

`sm90_k_grouped_fp8_gemm_1d1d`（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:145-226`）：

```cpp
DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);      // :154 必须 accumulate
DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K); // :155
...
int first_k = 0, sum_k = 0, sum_sf_k = 0, max_k = 0;
for (int i = 0; i < num_groups; ++ i) {                                  // :159-166
    if (first_k == 0 and ks_cpu[i] != 0) first_k = ks_cpu[i];            // 首个非空组的 K
    sum_k += ks_cpu[i], sum_sf_k += ceil_div(ks_cpu[i], 128);
    max_k = std::max(max_k, ks_cpu[i]);
    DG_HOST_ASSERT(ks_cpu[i] % 128 == 0);                                // 每组必须 128 对齐
}
```

注意 `first_k`：SM90 的 base tensormap 只能按"某个具体 K"造描述符（`:188-195` 用 `first_k`），
而每组 K 不同、运行期才知道，所以只能拿**首个非空组的 K** 造一张基准 tensor map，之后在 kernel
里原地 patch。这也是 `k_grouped_fp8_gemm_nt_contiguous` 必须先分配一块
`num_sms * 4 * sizeof(CUtensorMap)` 的 `tensor_map_buffer` 的原因（`gemm.hpp:413-417`，
注释写明 "double buffering for both A and B operands (2 \* 2)"）。

kernel 里的原地改写（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh`）：

```cpp
if constexpr (kGemmType == GemmType::KGroupedContiguous) {
    *smem_tensor_map_a = tensor_map_a_base;   // :129-130 初值 = 首组
    *smem_tensor_map_b = tensor_map_b_base;
}
...
if (kGemmType == GemmType::KGroupedContiguous and last_group_idx != scheduler.current_group_idx) {
    const uint64_t current_k_offset = scheduler.current_k_start;
    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_a, gmem_a_ptr + current_k_offset * shape_m); // :198
    ptx::tensor_map_replace_global_addr_in_smem(smem_tensor_map_b, gmem_b_ptr + current_k_offset * shape_n); // :199
    ptx::tensor_map_replace_global_inner_dim_stride_in_smem(smem_tensor_map_a, scheduler.current_shape_k, scheduler.current_shape_k); // :200
    ...
    cute::tma_desc_commit_group();             // :204 保证在飞的 TMA 不再用旧 map
    cute::tma_desc_wait_group();               // :205
    *(gmem_tensor_map_a) = *(smem_tensor_map_a);   // :210 回写 gmem 供 peer CTA 用
    *(gmem_tensor_map_b) = *(smem_tensor_map_b);
    ptx::tensor_map_release_gpu();             // :212
    ptx::tensor_map_acquire_gpu(gmem_tensor_map_a); // :215
}
```

- `current_k_offset * shape_m` 正是 2.2 节说的"基址前移 `k_start·M`"；
- `inner_dim_stride` 改成 `current_shape_k`，让 TMA 在每个组内按**组自己的 K** 定长取块；
- `tma_desc_commit_group`/`wait_group`（`:204-205`）+ `release/acquire`（`:212-215`）是**竞态修复**：
  必须确保在飞的 TMA 不再引用旧 map，且线程间通过 gmem 的 acquire/release 看到新 map。这类
  切组 tensormap 竞态就是社区 PR
  [#343](https://github.com/deepseek-ai/DeepGEMM/pull/343) 处理的对象。

对比 2.1：SM100 `tn` 把 M/N 放在连续维、K 放外层，天然不需要按组 patch——因为 A/B 的
`[sum_k, M]` 布局里，组的边界就是"外层 K 的行区间"，TMA 描述符一次造好后，内核用
`kAWithGroupKOffset = is_k_grouped_contiguous(GemmType)` 在坐标里加组偏移即可
（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:243-245`）。**这就是 SM100 用
`tn` 的"为什么"**：wgrad 的激活/上游梯度本来就是 `[token, dim]`（dim 连续）的形状，`[sum_k, M]`
可以直接拼接、无需转置；SM90 的 `nt` 需要把每组转成 `[M, k_g]` 的 K 连续形式。

## 3. `ks_cpu` 与 `grouped_layout`：contiguous 与 PSUM 两种编码

`check_k_grouped_args`（`gemm.hpp:55-76`）是 host 侧唯一的参数校验：

```cpp
if (ks_cpu.has_value() and not ks_cpu.value().empty()) {
    DG_HOST_ASSERT(static_cast<int>(ks_cpu.value().size()) == num_groups);
    int sum_k = 0;
    for (const auto k: ks_cpu.value()) {
        DG_HOST_ASSERT(k % k_alignment == 0);   // 每组必须按 k_alignment 对齐（:69）
        sum_k += k;
    }
    return sum_k;
}
DG_HOST_ASSERT(use_psum_layout);                // 没有 ks_cpu 时必须 PSUM
return sum_k_if_ks_cpu_missing;
```

- `ks_cpu`：各组的 **K 大小**（可选 CPU 端向量，避免 device 同步）。传了它就以它为准，
  每组必须 `k_alignment` 的倍数；SM90 `nt` 强制要求传（`gemm.hpp:388`）。
- `grouped_layout`：device 端 `int32` 张量，长度 `num_groups`，含义由 `use_psum_layout` 决定
  （注释见 `gemm.hpp:345-346`）：
  - **contiguous**（`use_psum_layout=false`）：`grouped_layout[i]` = 第 i 组**对齐后的物理 K**；
  - **PSUM**（`use_psum_layout=true`）：`grouped_layout[i]` = 第 i 组**逻辑 K 的结束位置**，
    组起点 = `align(上一组结束, k_alignment)`。

调度器 `get_next_k_group()`（`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:71-82`）把这两种编码
统一成"物理起点 + 组内 K"：

```cpp
if constexpr (kGemmType == GemmType::KGroupedContiguousWithPsumLayout) {
    const auto next_k_end = grouped_layout[current_group_idx];
    current_k_start = math::align(current_k_end, kKAlignment);   // PSUM：起点对齐上一组结束
    current_shape_k = next_k_end - current_k_start;
    current_k_end = next_k_end;
} else {
    current_k_start += current_shape_k;                          // contiguous：直接累加
    current_shape_k = grouped_layout[current_group_idx];
}
```

PSUM 的意义：训练里每组的逻辑 token 数（比如 `[4096, 4103, 4090, ...]`）通常不会是 128 的倍数，
但物理存储要按 `k_alignment` 补齐。PSUM 只传"逻辑结束"，让 kernel 自己算对齐起点，
省去 host 侧再算一遍物理布局；对齐产生的 gap 行必须写零，对应的 SF padding 必须有效
（`gemm.hpp:333-334` 注释）。PSUM 只在 SM100 上支持（`enumerate_k_grouped_contiguous` 里
`psum_list = (False, True) if get_arch_major() == 10 else (False, )`，
`tests/generators.py:244`）。

## 4. SF 的分组打包

K-grouped 的 scale factor 也不能直接用普通 1D1D 的 MN-major 布局——因为每个组的 K 不同、
SF 行数也不同，必须**逐组独立打包**。入口是
`layout::transform_k_grouped_sf_into_required_layout`（`csrc/apis/layout.hpp:89-116`）：

```cpp
if (sf.scalar_type() == torch::kFloat and arch_major == 9)
    return get_mn_major_tma_aligned_tensor(sf);                       // :104-105 SM90：仅转 TMA 对齐/MN-major
if (sf.scalar_type() == torch::kFloat and arch_major == 10)
    return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(   // :108-109 SM100：打包 UE8M0 int32
        sf, grouped_layout, ks_cpu, gran_k, k_alignment, use_psum_layout);
if (sf.scalar_type() == torch::kInt and arch_major == 10 and gran_k == 32)
    return check_k_grouped_packed_ue8m0_tensor(sf, ...);             // :112-113 已是打包格式，只校验
```

SM100 的打包 kernel `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`
（`csrc/jit_kernels/impls/smxx_layout.hpp:218-283`）逐组：

```cpp
for (const auto k: ks_cpu.value()) {
    DG_HOST_ASSERT(k % k_alignment == 0 and k % gran_k == 0);
    const auto group_sf_k = k / gran_k;
    ref_sf_k += group_sf_k;
    packed_sf_k += ceil_div(group_sf_k, 4);      // :246 每 4 个 UE8M0(1B) 打成 1 个 int32
}
DG_HOST_ASSERT(ref_sf_k == sf_k);
```

即 `[sum_k]` 行的 FP32 SF 先按组切成 `[k/gran_k, mn]`，每组再沿 K 方向**每 4 个打包成 1 个
int32**（`packed_sf_k += ceil_div(group_sf_k, 4)`），得到 `[packed_sf_k, mn]` 的 `int32` 张量，
组内是紧凑的、组间不共用尾部的 3 个 padding 槽。已打包的输入走
`check_k_grouped_packed_ue8m0_tensor`（`:286-317`）只做校验，要求 `sf.stride(0) >= mn`、
`stride(0) % 4 == 0`（允许 K 方向有 stride padding）。测试里对 `gran_k=32` 的打包结果做了逐
元素核对（`tests/test_fp8_fp4.py:319-340`，`assert torch.equal(packed_a[1], ref_packed_a)`）。
SM100 kernel 造 SF 的 TMA 描述符时用 `sum_k` 的逻辑长度 `sfa.size(0) * gran_k * 4`（因为 int32
里塞了 4 个 gran_k）（`sm100_fp8_fp4_gemm_1d1d.hpp:377-380`）。

> 注意 `recipe` 的第三个元素 `gran_k`：SM90 只支持 `gran_k=128`（`k_alignment` 也必须 128），
> SM100 支持 `gran_k=32` 或 `128`（`layout.hpp:100-101`）。FP4 版要求 `k_alignment % 256 == 0`
> （`gemm.hpp:449-450`）。

## 5. 调度器：按组切 K 块

K-grouped 的并行度 = `M/BLOCK_M × N/BLOCK_N × num_groups`，组内再沿 K 迭代。调度器
（`scheduler/gemm.cuh`）与普通 GEMM 共用模板，只是多了几个 group 状态：

```cpp
uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;  // :66
```

组切换发生在 `get_next_block` 的 K-grouped 分支（`:226-246`）：

```cpp
} else if constexpr (is_k_grouped_contiguous(kGemmType)) {
    while (true) {
        if (current_group_idx == kNumGroups) return false;
        if (next_block_idx < (current_group_idx + 1) * num_blocks) break;
        current_group_idx ++;
        if (current_group_idx >= kNumGroups) return false;
        const auto aligned_shape_k = math::align(current_shape_k, kKAlignment);
        current_sf_k_cumsum += math::ceil_div(aligned_shape_k, kSFKSpan);  // SF 行跨组累加，自身独立计数
        get_next_k_group();
    }
    get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks, m_block_idx, n_block_idx);
}
```

组内 block 数 `num_blocks = num_m_blocks * num_n_blocks` 对所有组相同（M/N 固定），所以一个组
的 block 区间就是 `(group_idx+1)*num_blocks`；跨组时更新 `current_k_start`（物理 K 起点）与
`current_sf_k_cumsum`（SF 行累计，**每组的 SF 行数 = `ceil_div(align(k_g), 128)`，与 K 物理起点
解耦**——这正是 §4 逐组独立打包在 kernel 侧的对应）。取全局地址时按索引类型分别加偏移
（`get_global_idx`，`:157-168`）：`MN` 加 `group_idx * shape_dim`、`K` 加 `current_k_start`、
`SF_K` 加 `current_sf_k_cumsum`。SM90 kernel 里还顺手跳过空组
（`sm90_fp8_gemm_1d1d.cuh:179-180`：`if (kGemmType == KGroupedContiguous and current_shape_k == 0) continue;`）。

## 6. H100 实测（真实数字）

`harness/exp_kgrouped_api.py`（输出 `harness/exp_kgrouped_api.out.txt`）在 8×H100 的
`kernel_lab` 容器里跑 DeepGEMM 2.8.1（`arch_major=9`）：

**A. SM90 `nt` API 语义**（用 DeepGEMM 自带 `tests/generators.py` 造数据，FP8 per-128 量化，
输出 fp32 accumulate）：

| 用例 | 结果 |
|---|---|
| A1 contiguous，4 组 `ks=[512,384,640,256]`，`m=512,n=1024` | `D[e]` vs `A_eᵀB_e` relerr **1.98e-2**（FP8 量化噪声量级） |
| A2 含空组 `ks=[512,0,640,256]` | 空组 `D[1]` 全零 **True**；非空组 relerr **3.63e-2** |
| A3 `ks_cpu=[384,200]`（200 非 128 倍数） | 被拒：`gemm.hpp:69` `k % k_alignment == 0` |
| A4 `tn` 版在 sm90 | 被拒：`gemm.hpp:369` `Unsupported architecture` |
| A5 `groups=8, m=4096, n=7168, k/group=4096` | 4528 us，**425 TFLOPS** |

**B. 物理布局等价**（纯 torch float32，与架构无关，`logical_ks=[3,2,4]`、`padded=[4,4,4]`、
`M=5,N=6`）：

| 布局 | 与自然布局 `A_eᵀB_e` 的 relerr |
|---|---|
| SM100 `tn`：`[sum_k, M]` MN-major | **0.00e0**（逐位相等） |
| SM90 `nt`：每组 `[M, pad]` K-major 拼接 | **3.33e-8**（float32 舍入） |

> A1 的 1.98e-2、A2 的 3.63e-2 都是 **FP8 per-128 块量化**的正常噪声（与 `docs/08` §3 里
> dense/grouped FP8 的 ~3.7e-2 同级），不是 kernel 实现误差；A5 的 425 TFLOPS 是 H100 上
> 该形状下 FP8 wgrad 的真实吞吐（供后续 P4-4 做性能参考）。

## 7. 小结与下一步

- **两条路径一个数学**：`tn`（SM100/MN-major，`[sum_k, M]`）与 `nt`（SM90/K-major，每组
  `[M,k_g]`）都算 `D[e]=A_eᵀB_e`；差别只在物理布局与取数机制——SM100 靠坐标偏移，
  SM90 靠**按组原地改写 tensormap**（`sm90_fp8_gemm_1d1d.cuh:193-215`）。
- **参数契约**：`ks_cpu`（每组 K，必须 `k_alignment` 对齐）+ `grouped_layout`
  （contiguous=物理 K / PSUM=逻辑 K 结束，PSUM 仅 SM100）。SM90 `nt` 强制 accumulate。
- **为什么 wgrad 走 K 分组**：收缩维（token）各专家不等长；`tn`/`nt` 的选择是架构决定的
  物理布局取舍，不是数学要求。

**下一步**：
- **P4-3**：dgrad（`m_grouped_*`）与 contiguous/masked/PSUM 布局在训练中的用法。
- **P4-4**：在 H100 上把 SM90 `k_grouped_fp8_gemm_nt_contiguous` 接到真实 MoE 的 wgrad
  （`dW1=AᵀdG`、`dW2=xᵀdY` 的 token 分组），与 PyTorch autograd 做 kernel 级数值对拍，
  报误差与吞吐（本文已完成 API 语义与布局等价的前置核对）。
