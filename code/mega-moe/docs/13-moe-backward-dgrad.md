# MoE 反向的 dgrad API：`m_grouped_*` 与三种布局在训练中的用法

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 13 篇（路线图 **P4-3**）。
> P4-1 推出反向全景并给出「**wgrad 走 `k_grouped`、dgrad 走 `m_grouped`**」的角色表；P4-2 精读了
> wgrad 的 `k_grouped_*_contiguous`。本文反过来精读 **dgrad**：为什么对激活的梯度天然是 **M 轴分组**，
> 两个 host 入口（`m_grouped_fp8_fp4_gemm_nt_contiguous`、`..._nt_masked`）的形状契约，调度器
> 如何用 `MGroupedContiguous` / `MGroupedMasked` / `MGroupedContiguousWithPsumLayout` 三种 GEMM type
> 统一处理，以及 **contiguous / masked / PSUM** 三种布局在训练各阶段的分工。
>
> 约束提醒：SM100 版 dgrad 走 1D1D + tcgen05 + packed UE8M0（**仅分析**）；SM90 版走
> `sm90_m_grouped_fp8_gemm_contiguous_1d2d` / `..._masked_1d2d`，**H100 可跑**。本文用 H100
> 真机做了三种布局的 API 语义核对与同一性验证（`harness/exp_mgrouped_dgrad.py` + `.out.txt`）。

## 1. 从数学到 API：dgrad 的形状契约

P4-1 的 (3)(5) 把两处 dgrad 写出来：

$$
\underbrace{dA_{tj}=dY_{tj}\,(W^2_j)^{\mathsf T}}_{\text{linear2 dgrad}}
\quad(M=\text{token}_j,\;N=I,\;K=H\ \text{固定}),\qquad
\underbrace{dX_{tj}=dG_{tj}\,(W^1_j)^{\mathsf T}}_{\text{linear1 dgrad}}
\quad(M=\text{token}_j,\;N=H,\;K=2I\ \text{固定}).
$$

关键：**不等长的是每个专家收到的 token 数，落在 M 轴上；N/K 对所有专家相同**。这正是 DeepGEMM
README 的 grouped GEMM 定义（`README.md:78`）：

> "DeepGEMM groups only the M-axis, while N and K must remain fixed. This design is tailored for
> scenarios where experts in an MoE model share the same shape."

于是 dgrad 直接复用前向那套 `m_grouped_*` API（`docs/01` §1）。物理上把各专家的 token 行**沿 M 拼接**，
权重栈是 `[G, N, K]`，对应注释里的形状 `[M, K] @ [G, N, K].mT`（`gemm.hpp:194`、`:280`）。

把本文实验用的 linear2 dgrad 形状对上号：`A = dY`（`[M,H]`，K=H 连续）、`B = W^2`（`[G,I,H]`，
即 N=I、K=H）、`D = dA`（`[M,I]`）。H100 实测（`exp_mgrouped_dgrad.py` A1）：

```
A1 contiguous dgrad: G=6 m=1024 n(I)=256 k(H)=512 alignment=128
   grouped_layout: len=1024, gap rows(layout==-1)=303/1024 (30%)
   relerr(dA valid) vs dY_e@W2_e^T = 3.701e-02
```

relerr 3.7e-2 是 FP8 per-128 块量化的正常噪声（与 `docs/08` §3、`docs/12` 同级），不是 kernel 误差。

## 2. 两个 host 入口

### 2.1 contiguous：`m_grouped_fp8_fp4_gemm_nt_contiguous`

入口在 `csrc/apis/gemm.hpp:181-250`。形状与类型检查：

```cpp
// Shape must be `[M, K] @ [G, N, K].mT`            // :194
const auto major_a = get_major_type_ab(a.first);
const auto major_b = get_major_type_ab(b.first);
DG_HOST_ASSERT(major_a == cute::UMMA::Major::K);                        // :197 A 必须 K-major
if (fp8_fp4_requires_k_major(a.first, b.first))
    DG_HOST_ASSERT(major_b == cute::UMMA::Major::K);                    // :198-199
...
const auto [m , k ]        = check_ab_fp8_fp4(a.first, major_a, arch_major);      // :204
const auto [num_groups, n, k_] = check_grouped_ab_fp8_fp4(b.first, major_b, ...); // :205
const auto [m_, n_] = get_shape<2>(d);                                  // :206
DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);                        // :207
DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);                    // :209
```

注意两点契约：**A（激活/上游梯度）必须 K-major**（`:197`），因为它是逐 token 的 `[token, dim]`
（dim 连续）；**D 固定 BF16**（`:209`），即 dgrad 输出不量化。这与 wgrad 的 `tn` 版要求 MN-major
恰好相反——因为 wgrad 的 A/B 是「按 token 行拼接」，而 dgrad 的 A 是「按 token 行拼接的 `[M,K]`」、
B 是「权重栈 `[G,N,K]`」，两者都保持 K 连续。

然后是**布局判别**（`:213-220`），这也是 contiguous 与 PSUM 的分叉点：

```cpp
if (use_psum_layout) {
    const auto [num_groups_] = get_shape<1>(grouped_layout);   // PSUM：长度 = 组数
    DG_HOST_ASSERT(num_groups == num_groups_);
} else {
    const auto [m__] = get_shape<1>(grouped_layout);           // contiguous：长度 = m
    DG_HOST_ASSERT(m == m__);
    DG_HOST_ASSERT(not expected_m_for_psum_layout.has_value());
}
```

- **contiguous**：`grouped_layout` 是**长度 m 的逐行组号**（gap 行记 `-1`）；
- **PSUM**：`grouped_layout` 是**长度 G 的逐组前缀和结束行**，组起点 = `align(上一组结束, alignment)`。

最后按架构分发（`:236-249`）：SM90 走 `sm90_m_grouped_fp8_gemm_contiguous_1d2d`，SM100 走
`sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d`（`sm100_fp8_fp4_gemm_1d1d.hpp:169-252`）。

> PSUM 只作用于 **SFA**：`transform_sf_pair_into_required_layout` 的形参注释写明
> "PSUM layout only applies to SFA (B is weights with no gaps)"（`csrc/apis/layout.hpp:69`），
> `gemm.hpp:230` 也只把 `grouped_layout` 作为 `psum_sfa_layout` 传给 SFA 的打包。
> 道理很直白：**gap 行只出现在激活侧**（A/D），权重栈 B 是 `[G,N,K]`、每组紧凑无 gap。

### 2.2 masked：`m_grouped_fp8_fp4_gemm_nt_masked`

入口在 `gemm.hpp:269-318`。形状变成**稠密三维**：

```cpp
// Shape must be `[G, M, K] @ [G, N, K].mT`      // :280
DG_HOST_ASSERT(major_a == cute::UMMA::Major::K and major_b == cute::UMMA::Major::K); // :283
const auto [num_groups  , m , k ] = check_grouped_ab_fp8_fp4(a.first, ...);  // :288
const auto [num_groups__ , m_, n_] = get_shape<3>(d);                        // :290
const auto num_groups___ = static_cast<int>(masked_m.numel());               // :291
DG_HOST_ASSERT(num_groups == ... == num_groups___);                          // :292
```

- A/D 都是 `[G, M, K]` / `[G, M, N]`（`m = max_m`，所有组统一上界）；
- `masked_m`（长度 G 的 int）告诉 kernel 第 g 组只算前 `masked_m[g]` 行（`:296` 要求 `torch::kInt`）；
- 有效行数**在 device 端读**，host 不需要知道（这正是 CUDA graph 友好的原因）。

DeepGEMM 测试里 masked 与 PSUM 是同一份数据的两种表示：`generate_m_grouped_masked(..., use_psum_layout=True)`
用 `layout_masked_to_psum` 把 `[G,max_m,·]` 重排成带 gap 的 `[total_m,·]`（`tests/generators.py:422-443`）。

## 3. 调度器：三种 M-grouped GEMM type

`GemmType` 在 `sched::Scheduler`（`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`）里把三件事统一：

| GemmType | 触发入口 | `grouped_layout` 含义 | 边界 | D 物理布局 |
|---|---|---|---|---|
| `MGroupedContiguous` | `..._contiguous`（psum=False） | 长度 m 的逐行组号（gap=-1） | 逐行 | `[m, N]` 拼接 |
| `MGroupedMasked` | `..._masked` | 长度 G 的每组有效行数 | 逐组 | `[G, max_m, N]` 稠密 |
| `MGroupedContiguousWithPsumLayout` | `..._contiguous`（psum=True） | 长度 G 的逐组前缀和结束 | 逐组 | `[m, N]` 拼接 |

构造时（`:92-105`）：contiguous 的 `num_blocks` 是常量 `num_m_blocks*num_n_blocks`（每条内边界的
block 下界固定）；masked 与 PSUM 的 `num_m_blocks` 会**随组变化**，所以要额外维护
`current_m_cumsum` / `current_m_block_cumsum`。

### 3.1 逐组推进 `get_next_block`

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedMasked) {          // :188
    while (true) {
        if (current_group_idx == kNumGroups) return false;
        num_m_blocks = math::ceil_div(static_cast<uint32_t>(grouped_layout[current_group_idx]), BLOCK_M); // :195
        const auto current_m_block_cumsum = current_m_cumsum + num_m_blocks;
        if (next_block_idx < current_m_block_cumsum * num_n_blocks) break;
        current_group_idx ++, current_m_cumsum = current_m_block_cumsum; // :201
    }
    get_swizzled_block_idx(next_block_idx - current_m_cumsum * num_n_blocks, m_block_idx, n_block_idx); // :204
} else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { // :205
    while (true) {
        if (next_block_idx < (current_m_block_cumsum + num_m_blocks) * num_n_blocks) break;
        if (++ current_group_idx == kNumGroups) return false;
        last_psum_m = math::align(current_psum_m, BLOCK_M);            // :216 上一组按 BLOCK_M 对齐
        current_psum_m = grouped_layout[current_group_idx];           // :217 本组前缀和结束
        current_m_block_cumsum += num_m_blocks;
        num_m_blocks = math::ceil_div(current_psum_m - last_psum_m, BLOCK_M); // :219
    }
    get_swizzled_block_idx(...);
    m_block_idx += last_psum_m / BLOCK_M;                             // :225 平移到全局行
}
```

- masked：`grouped_layout[g]` 直接是有效行数，`num_m_blocks` 逐组重算；
- PSUM：前缀和结束 `current_psum_m`、上一组的**对齐**起点 `last_psum_m`，组内 block 数 =
  `ceil_div(current_psum_m - last_psum_m, BLOCK_M)`，最后把局部 `m_block_idx` 平移到 `last_psum_m/BLOCK_M`；
- contiguous 不需要循环：`num_blocks` 已知，仅在取 B/SF 时按行查组号。

### 3.2 取全局地址 `get_global_idx`

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedContiguous) {      // :151
    const auto offset = kWithGroupOffset ? cute::max(0, grouped_layout[m_block_idx * BLOCK_M]) : 0;
    return offset * shape_dim + block_idx * block_size;
} else if constexpr (kGemmType == GemmType::MGroupedMasked or
                     kGemmType == GemmType::MGroupedContiguousWithPsumLayout) {  // :154
    const auto offset = kWithGroupOffset ? current_group_idx : 0;
    return offset * shape_dim + block_idx * block_size;
}
```

这里是「**为什么必须按 M block 对齐**」（`docs/01` §1.2，SM90=128）的 kernel 侧解释：

- contiguous 的组号只取 **`grouped_layout[m_block_idx * BLOCK_M]`**——即用该 M block 的**首行**组号
  代表整个 block。若段起点没按 `BLOCK_M` 对齐，一个 block 会跨两组、首行组号就不再代表整块；
- masked/PSUM 直接按 `current_group_idx` 取，因为组边界已被调度器按 `BLOCK_M` 取整。

### 3.3 跳过无效行 `is_computation_valid`

```cpp
} else if constexpr (kGemmType == GemmType::MGroupedContiguous) {      // :299
    return grouped_layout[m_offset + m_block_idx * BLOCK_M] >= 0;      // gap 行 -> 整块跳过
} else if constexpr (kGemmType == GemmType::MGroupedMasked) {          // :301
    return m_offset + m_block_idx * BLOCK_M < grouped_layout[current_group_idx];
} else if constexpr (kGemmType == GemmType::MGroupedContiguousWithPsumLayout) { // :303
    return m_offset + m_block_idx * BLOCK_M < current_psum_m;
}
```

无效 block 走 `full_barrier->wait` + `empty_barrier_arrive` 的**空转分支**（SM90 kernel `:349-355`），
只推进流水线、不计算。这样三种布局都不需要 host 端预先裁掉 gap。

### 3.4 TMA multicast 的 peer 一致性

SM90 用 cluster 内 TMA multicast 广播权重 B：只有同一 cluster 的两个 CTA 属于**同一组**时才能共用。
`is_tma_multicast_valid`（`:275-292`）对 `MGroupedContiguous` 专门检查：

```cpp
const auto group_idx = grouped_layout[m_block_idx * BLOCK_M];           // :287
const auto peer_group_idx = grouped_layout[(m_block_idx ^ 1) * BLOCK_M];// :288
return group_idx == peer_group_idx;                                     // :289
```

masked/PSUM 因为段边界必然对齐到 `BLOCK_M`，天然满足（`:280` 直接 `return true`）。

## 4. SM90 kernel：`sm90_m_grouped_fp8_gemm_contiguous_1d2d` 与 `..._masked_1d2d`

### 4.1 host 侧造 tensormap 的差异

`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:139-214`（contiguous）与 `:216-281`（masked）几乎同构，
差别只在 **TMA 描述符的维数**：

| 张量 | contiguous | masked |
|---|---|---|
| A（激活） | 2D，`num_groups=1`（`:178-182`） | **3D**，`num_groups`（`:245-249`） |
| B（权重栈） | 3D，`num_groups`（`:183-187`） | 3D，`num_groups`（`:250-254`） |
| D（输出） | 2D，`num_groups=1`（`:188-192`） | **3D**，`num_groups`（`:255-259`） |
| SFA | 2D（`:193-194`） | **3D**（`:260-261`） |

**为什么**：contiguous 把各段 token 已经**首尾拼接**进一张 `[m,·]`，所以 A/D/SFA 是二维、基址偏移
靠 `grouped_layout` 算出来；masked 是 `[G,max_m,·]` 稠密块，A/D/SFA 需要第三维（组号）作为 TMA 的
batch 坐标。这解释了 kernel 里那一对标志：

```cpp
constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;  // :189
...
tma::copy<BLOCK_K, BLOCK_M, ...>(&tensor_map_a, &full_barrier, smem_a[stage_idx], k_idx,
         scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx), ...); // :192-194
```

A 只在 masked 下带组偏移；而 B **永远**带组偏移（`:201` 用 `get_global_idx<true>`）：
contiguous 用行首组号、masked 用 `current_group_idx`，都要选到对应专家的权重块。
D 的写回同理（`:426`、`:432-437`）。

### 4.2 SFB 的逐组基址

权重的 scale factor（SFB）也是 `[G, N, k/128]` 分组存储。math warp 在算每个 block 前，先用
调度器算出**本组 SFB 的基址**（`:240`）：

```cpp
auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(
    shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);                 // :240
```

`shape_n_sfb * shape_k_scales` 就是单组 SFB 的元素数；contiguous 下 offset 由
`grouped_layout[m_block_idx*BLOCK_M] * 该值` 给出，masked 下由 `current_group_idx * 该值` 给出。
之后 `local_sfb + i*stride_k_sfb` 按 major 取值（`:241-247`）。

### 4.3 空块与 store

`is_computation_valid` 为假的 block 只等 barrier、直接进 store 流程：contiguous 下 D 是拼接的
`[m,N]`，用 `get_global_idx<false>`；masked 下 D 是 `[G,max_m,N]`，用 `get_global_idx<true>` 写回对应组的
稠密块（`:426-437`）。masked 的无效行**不会被写**（`is_computation_valid` 已挡掉），因此它们保持原值。

## 5. 训练各阶段该用哪种布局

把这三种布局放回 MoE 训练/推理的真实流程（与 `docs/01` §1.2-§1.4 呼应）：

| 阶段 | 已知信息 | 推荐布局 | 理由 |
|---|---|---|---|
| 训练前向 / prefill | host 端已知每专家 token 数（本地路由或已 dispatch 落地） | **contiguous** | 段可精确对齐拼接，kernel 最简、无稠密冗余 |
| 训练 dgrad | 同上（前向的 token→专家映射已固定） | **contiguous** | dgrad 复用前向的 M 分组；`dY` 分段与 `dX` 分段同构 |
| 解码（CUDA graph） | 每专家 token 数由 **device** 上的 dispatch 决定 | **masked** | 固定 `[G,max_m,·]` 形状，计数作为张量传入，图捕获友好 |
| DeepEP 输出 / 动态段 | device 端只知道**每组前缀和结束** | **PSUM** | 把 device 已知的前缀和塞回 contiguous kernel，省掉稠密冗余 |

三条实践要点：

1. **dgrad 是「逐专家 dX 再 scatter-add」**。top-k 路由下一个 token 被 k 个专家处理，故
   `dX[token] = Σ_{j∈topk} dX_j[token]`。kernel 给的是**按专家分段的 dX**，反向框架需要在
   combine 的反向路径上把同一 token 的 k 份贡献累加回 token 原位置（前向 dispatch/combine 的镜像，
   见 `docs/01` §2 与 `docs/07`）。
2. **wgrad 留专家本地、无需 all-reduce**。dgrad 才需要跨专家（等同跨 rank）的 scatter-add；
   wgrad 只在该专家 owner 的 token 上收缩，天然完整（`docs/04` §6、`docs/12`）。
3. **共享专家（shared expert）走稠密路径**。它不参与 topk，输出直接加到 `y`，反向是普通 dense GEMM
   （`m_grouped` 的 G=1 退化），不经上述分组布局。
4. **PSUM 的 gap 行**：物理存储要按 alignment 补齐，gap 行可能未初始化。`ensure_zero_padding=True`
   时打包 kernel 保证 gap 为 0；为假时 kernel 用 `get_aligned_effective_m_in_block`
   （`scheduler/gemm.cuh:177-183`）把**最后一个部分块**的 UMMA N 缩到 16 对齐，避免读到脏数据。

## 6. H100 实测

`harness/exp_mgrouped_dgrad.py`（输出 `.out.txt`）在 H100（sm90）的 `kernel_lab` 里跑 DeepGEMM 2.8.1
（`arch_major=9`），用 linear2 dgrad 形状（A=dY、B=W2、D=dA）。

**A. 三种布局的 API 语义**：

| 用例 | 结果 |
|---|---|
| A1 contiguous，G=6、m=1024、`n(I)`=256、`k(H)`=512 | relerr(dA 有效行) **3.70e-2**；gap 行 303/1024（30%） |
| A2 masked，G=6、max_m=160、`masked_m=[84,100,92,102,102,103]` | relerr **3.67e-2**；有效 583/存储 960 行（61%） |
| A3 PSUM，前缀和 `[156,393,608,756,880,1057]` | relerr **3.67e-2** |
| A4 contiguous `grouped_layout` 长度 383 ≠ m=384 | 被拒：`gemm.hpp:218` `m == m__` |
| A5 DeepSeek 量级 dgrad（G=8、m=4352、n=2048、k=7168） | 349 us，**366 TFLOPS** |

**B. 三种布局的同一性**（同一份 dgrad 数据，分别走 masked / PSUM / contiguous，取有效行比较）：

```
B same dgrad data through 3 layouts: G=6 total valid=575 total_m(physical)=768 n=256 k=512
   masked   vs dY@W2^T ref : relerr=3.692e-02
   PSUM     vs dY@W2^T ref : relerr=3.692e-02
   contiguous vs ref       : relerr=3.692e-02
   masked vs PSUM          : relerr=0.000e+00
   PSUM   vs contiguous    : relerr=0.000e+00
```

**三种布局两两逐位相等**（masked↔PSUM↔contiguous 全 0 误差），确证它们只是「同一 M 分组计算的
三种边界表示」——数值上完全等价，差别只在**存储形状与组边界信息的编码方式**，以及由此决定的
空间开销（gap / 稠密冗余）与图捕获友好度。

> A1/A2/A3 的 ~3.7e-2 与 B 的残差同量级，均为 FP8 per-128 块量化噪声；A5 的 366 TFLOPS 是 H100
> 上该 dgrad 形状的真实吞吐（可作 P4-4 的性能参考）。SM100 的 1D1D 路径（tcgen05 + packed UE8M0 +
> 固定 256 对齐）为**仅分析**。

## 7. 小结与下一步

- **dgrad = `m_grouped`**：不等长在 **M（token）轴**、N/K 固定；与前向共用同一族 API，A 必须 K-major、
  D 固定 BF16。
- **三种布局一个计算**：contiguous（逐行组号）/ masked（逐组计数，device 已知）/ PSUM（逐组前缀和）。
  调度器用三个 `GemmType` 统一处理；`is_computation_valid` 负责跳过 gap / 无效行。
- **为什么段要对齐 BLOCK_M**：contiguous 用「M block 首行的组号」代表整块，不对齐就会跨组；
  SM90 的 TMA multicast peer 一致性也依赖它（`scheduler/gemm.cuh:287-289`）。
- **训练分工**：前向与 dgrad 用 contiguous（host 已知 token 数）；解码 CUDA graph 用 masked；
  DeepEP 动态输出用 PSUM 桥接到 contiguous kernel。

**下一步**：
- **P4-4**：在 H100 上把 SM90 `k_grouped_fp8_gemm_nt_contiguous`（wgrad）与 `m_grouped_*_contiguous`
  （dgrad）接到真实 MoE 的完整反向，与 PyTorch autograd 做 **kernel 级**数值对拍，报误差与吞吐。
- **P5-1 / P5-2**：Mega Gate / Mega mHC。
