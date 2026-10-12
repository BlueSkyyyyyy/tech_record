# DeepGEMM FP8 einsum 投影 `bhr,hdr->bhd`：block-scale 版按头 batched GEMM（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P11 增量（第 21 篇）。
> 第 18 篇拆了 einsum 家族的 BF16 MLA 投影 `bhr,hdr->bhd` / `bhd,hdr->bhr`
> （按 attention head 分 batch 的 batched GEMM）。本篇补上它的 **FP8 版本**：
> `fp8_einsum('bhr,hdr->bhd', ...)`——同一套骨架换成 **FP8 block-scale**（A per-token 1D scale、
> B per-128×128 block 2D scale），**有 SM90 实现，H100 可跑**；其余两条表达式的 FP8 变体是
> **SM100 专属（仅分析）**。
>
> 为什么与 Mega MoE 专题相关：Mega MoE 的 linear1/linear2 是**按专家分组**的 grouped GEMM；
> 而 MLA 的 q/kv 吸收投影是**权重共享、按头切块**的另一类 GEMM。DeepGEMM 把这两类都做成
> 「前向 skeleton + block-scale（UE8M0）量化」的统一体系：理解 FP8 einsum 的 SF 布局契约，
> 正是理解 Mega MoE 里 `transform_sf_into_required_layout` 与 UTCCP/UE8M0 通路的一条侧证。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。
> FP8 主实现（SM90）：`csrc/apis/einsum.hpp`、`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp`、
> `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`。
> H100 数值实验：`harness/exp_fp8_einsum.{py,sh,out.txt}`。

## 1. 数学：FP8 版仍是「每头一个 GEMM」

表达式与 BF16 版完全一致（第 18 篇 §1）：

```
bhr,hdr->bhd :  D[b,h,d] = Σ_r A[b,h,r]·B[h,d,r] = A[:,h,:] @ B[h].T      (m,n,k)=(b,d,r)
```

**对每个头 `h` 一个独立的 GEMM**，`h` 是 batch 维（不是 grouped GEMM 那种不等长分组）。
区别只在数值类型：A、B 都是 **FP8（E4M3）**，各带一组 **block-scale（SF）**；D 在 SM90 上只出 BF16。

MLAD 里对应的形状：`A:[b,h,r]`（token×头×latent）、`B:[h,d,r]`（头×head-dim×latent），
`b`（token/chunk）落在 batch 之外、`h` 是真正被切成 batch 的维。

## 2. host 分派：`fp8_einsum` 先 permute 成 `(batch,h)` 轴序

入口 `fp8_einsum()`（`csrc/apis/einsum.hpp:251-298`）与 BF16 的 `einsum()` 类似是字符串分派，
但有两条 FP8 专属约束：

- **只有 `bhr,hdr->bhd` 接受 FP8 `(d,sfd)` 输出对**（`:259-260` 注释）；其他表达式只收 BF16/FP32 D。
- SM90 只实现了这一条（其余两条带 `and arch_major == 10`，`:277`/`:287`）。

对 `bhr,hdr->bhd`（`:262-276`），代码把张量 **按 `(b,h)` 两维转置**，凑成
`fp8_bmm` 要求的 `(batch_size, m, n, k)` 轴序：

```cpp
// (batch_size, m, n, k): (h, b, d, r)
const auto perm_a = a.first.permute({1, 0, 2});        // [b,h,r] -> [h,b,r]
const auto perm_sfa = a.second.permute({1, 0, 2});     // [b,h,·] -> [h,b,·]
// d 同样 permute{1,0,2}；B 本来就是 [h,d,r]，无需动
fp8_bmm(perm_a, perm_sfa, b.first, b.second, perm_d, perm_c, recipe, "nk", epilogue_class);
```

即 **`h` 被当作 `batch_size`、`b` 被当作 `m`**。注释（`:265-267`）特别说明：当 D 是 FP8 输出对时，
SF 的列是沿 D 的最后两维 `(h,d)` 压平的（等价于对 `d.flatten(1,2)` 做 per-token cast），
因此这个 `(b,h)` permute **不改变 SF 布局**——这是「SF 按 D 的展平行量化」的直接后果。

### 2.1 `fp8_bmm` 的形状/类型契约（`:181-249`）

```cpp
// 形状必须是 `[B, M, K] @ [B, N, K].T`（:203）
const auto [batch_size  , m , k ] = get_shape<3>(a);
const auto [batch_size_ , n , k_] = get_shape<3>(b);
const auto [batch_size__, m_, n_] = get_shape<3>(d_tensor);
DG_HOST_ASSERT(batch_size == batch_size_ and batch_size == batch_size__);
DG_HOST_ASSERT(m == m_ and n == n_ and k == k_);          // :214
DG_HOST_ASSERT(a.scalar_type() == torch::kFloat8_e4m3fn); // :216
DG_HOST_ASSERT(b.scalar_type() == torch::kFloat8_e4m3fn); // :217
```

FP8 输出对（`(d, sfd)`）另有约束（`:218-227`）：

```cpp
DG_HOST_ASSERT(jit->device.get_arch_major() == 10);          // :224 仅 SM100
DG_HOST_ASSERT(d_tensor.stride(0) == n and n % 128 == 0);    // :225 批在列方向连续
check_sf_layout(sfd.value(), m, batch_size * n, 1, 32, ...); // :226 gran_k=32 packed UE8M0
```

然后经过 SF 变换（`:234-235`）后按 arch 分派（`:238-248`）：

```cpp
if (arch_major == 10) {
    sm100_fp8_bmm(...);                                       // :240  1d1d + 可自定义 epilogue
} else {
    DG_HOST_ASSERT(merged_epilogue_class == nullptr ...);     // :244  自定义 epilogue 仅 SM100
    const auto major_sfb = get_major_type_ab(sfb);            // :245
    DG_HOST_ASSERT(gran_k_a == 128 and gran_k_b == 128);      // :246  SM90 只支持 gran 128
    sm90_fp8_bmm(...);                                        // :247
}
```

## 3. SM90 实现：复用 `1d2d` kernel 的 `GemmType::Batched` 分支

`sm90_fp8_bmm`（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:283-352`）没有专用 kernel，
而是**复用前向 per-128 block-scale 的 `1d2d` 骨架**，只把 `GemmType` 设成 `Batched`：

```cpp
DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16);                       // :290 SM90 只出 bf16
DG_HOST_ASSERT(major_a == K and major_b == K);                            // :291 都 K-major
const auto desc = GemmDesc {
    .gemm_type = GemmType::Batched,                                       // :294
    .kernel_type = KernelType::Kernel1D2D,
    .m = m, .n = n, .k = k, .num_groups = batch_size, ... };              // :296 num_groups=h
const auto config = get_best_config<SM90ArchSpec>(desc);                  // :305
DG_HOST_ASSERT(config.storage_config.swizzle_a_mode == config.layout.block_k);  // :308 无 TMA split
const auto tensor_map_a = make_tma_3d_desc(a, k, m, batch_size, ...);     // :311-315 3D TMA
const auto tensor_map_b = make_tma_3d_desc(b, k, n, batch_size, ...);     // :318-322
const auto tensor_map_d = make_tma_3d_desc(d, n, m, batch_size, ...);     // :326-329
const auto tensor_map_sfa = make_tma_sf_desc(... sfa, m, k, ..., batch_size, ...);  // :331-332
```

关键点：**batch 是 3D TMA 的第三维**（`make_tma_3d_desc`，`csrc/jit_kernels/impls/runtime_utils.hpp:161`），
A/B/D/SFA 都多一个 batch stride；SFB 不在这里建托，而是 kernel 内按组算基址。

`1d2d` 的名字含义：**A 的 scale 是 1D（per-token，每 128 列一个），B 的 scale 是 2D（per-128×128 block）**。
kernel（`sm90_fp8_gemm_1d2d.cuh`）里：

- `DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling")`（`:56`）——
  这就是 host 侧 `gran_k==128` 断言的 kernel 端来源；
- `DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>)`（`:62`）——CD 只能是 BF16；
- `cudaGridDependencySynchronize()`（`:149`）——**PDL**，等前驱 kernel 完成再继续。

### 3.1 batched TMA 载入（`:186-202`）

```cpp
constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);              // :186
const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);// :187
tma::copy<..., kIsBatchedMM>(&tensor_map_a, ..., k_idx,
         scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx),
         num_tma_multicast_a, batch_idx);                                    // :192-194
tma::copy<BLOCK_M, BLOCK_K, 0>(&tensor_map_sfa, ..., m_block_idx*BLOCK_M, ...); // :195-196
tma::copy<..., kIsBatchedMM>(&tensor_map_b, &full_barrier, ..., batch_idx);  // :200-202
```

`batch_idx = current_group_idx` 直接作为 TMA 第 3 维坐标——Batched 的「组」被**当成第 0 维枚举**。

### 3.2 调度器：Batched 路径（`scheduler/gemm.cuh`）

`GemmType::Batched` 与 `Normal` 共用 `num_blocks = num_m_blocks * num_n_blocks`（`:90-91`），
`get_next_block` 在 `:247-259` 里按 `next_block_idx / num_blocks` 推 `current_group_idx`（batch），
`get_global_idx` 对 Batched **默认不加组偏移、只有 `SF_K` 索引型才加**（`:169-172`）：

```cpp
} else if constexpr (kGemmType == GemmType::Batched) {
    // Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
    const auto offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
    return offset * shape_dim + block_idx * block_size;
}
```

这解释了 kernel 里 SFB 的取址（`1d2d.cuh:240`）：

```cpp
auto previous_group_offset = scheduler.get_global_idx<true, IndexType::SF_K>(
                                 shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);   // :240
const uint32_t stride_n_sfb = kMajorSFB == MN ? 1 : shape_k_scales;                  // :241
const uint32_t stride_k_sfb = kMajorSFB == MN ? shape_n_sfb : 1;                     // :242
auto local_sfb = sfb + previous_group_offset + ((n_block_idx*BLOCK_N)/BLOCK_K)*stride_n_sfb; // :243
```

即 **SFB 逐 batch 基址 = `current_group_idx * (ceil(n/128)·ceil(k/128))`**——SFB 是
`[h, ceil(d/128), ceil(r/128)]` 连续排布（`h` 是第 3 维）。store 同理走 3D（`:433-438`）：

```cpp
if constexpr (kGemmType == GemmType::Batched) {
    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx, scheduler.current_group_idx); // :434-435
}
```

## 4. SF 布局的硬约束（实验直接印证）

| 张量 | 角色 | FP8 数据 | SF 形状 | 粒度 |
|---|---|---|---|---|
| A | `[b,h,r]` | `[b,h,r]` E4M3 | `[b,h,ceil(r/128)]` | per-token（1D，沿 r 每 128） |
| B | `[h,d,r]` | `[h,d,r]` E4M3 | `[h,ceil(d/128),ceil(r/128)]` | per-block（2D，128×128） |
| D | `[b,h,d]` | BF16 | —（SM90 无 FP8 输出） | — |

- SM90 只认 **gran 128**：`einsum.hpp:246` 的 `gran_k_a==128 and gran_k_b==128`；kernel 端
  `BLOCK_K==128`（`1d2d.cuh:56`）。传 `recipe=(1,64,64)` 会在更早的 `layout.hpp`（`:57` 未知变换 /
  `:119` SF 形状不符）被拒。
- A 必须 K-major（收缩维 `r` 连续）：`a.stride(-1)==1 or a.stride(-2)==1`（`einsum.hpp:206`），
  SM90 bmm 进一步要求 `major_a==major_b==K`（`sm90_fp8_gemm_1d2d.hpp:291`）。

## 5. SM100 分支（**仅分析**）

`sm100_fp8_bmm`（`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:405-481`）用 **`Kernel1D1D`**
（A、B 都是 1D scale）而非 `1D2D`，并支持 **FP8 输出对 + 自定义 epilogue**：

```cpp
.gemm_type = GemmType::Batched, .kernel_type = KernelType::Kernel1D1D, .num_groups = batch_size,  // :415-417
...
.tensor_map_sfa = make_tma_sf_desc(... gran_k_a, batch_size, ...);   // :454-455
.tensor_map_sfb = make_tma_sf_desc(... gran_k_b, batch_size, ...);   // :456-457
.epilogue_class = epilogue_class, .gran_k_a = gran_k_a, .gran_k_b = gran_k_b, ...  // :469-471
```

另外两条表达式的 FP8 版也只在 arch10 出现：

- `bhd,hdr->bhr`（`einsum.hpp:277-286`）：`(batch,m,n,k)=(h,b,r,d)`，B permute `{0,2,1}`，
  `compiled_dims="nk"`；
- `bhd,bhr->hdr`（`einsum.hpp:287-294`）：`(batch,m,n,k)=(h,d,r,b)`，A/B 都 permute `{1,2,0}`，
  `compiled_dims="mn"`，且 D 是 FP32（per-channel 量化）。

SM90 上这两条直接 `DG_HOST_UNREACHABLE`（`:296`）——**FP8 MLA 投影在 H100 上只有
`bhr,hdr->bhd` 一条可跑**。

## 6. H100 实测（`harness/exp_fp8_einsum.{py,sh,out.txt}`）

平台：H100 sm90、132 SM、CUDA 13.2 链、DeepGEMM commit `057ca59`。输入按 §4 布局量化
（`per_token_cast_to_fp8` / `per_block_cast_to_fp8`，均 `use_ue8m0=True`）。

### 6.1 语义与误差分层（`[A]`）

| b | h | r | d | vs 真值（bf16 出） | vs 隔离量化的 fp32 模拟 | 量化误差 |
|---|---|---|---|---|---|---|
| 32 | 128 | 512 | 128 | 3.95e-2 | 2.36e-3 | 3.82e-2 |
| 4096 | 128 | 512 | 128 | 3.55e-2 | 1.95e-3 | 3.57e-2 |
| 128 | 8 | 4096 | 1024 | 3.51e-2 | 2.95e-3 | 3.51e-2 |
| 4096 | 8 | 4096 | 1024 | 3.21e-2 | 2.55e-3 | 3.26e-2 |

**误差可以干净地分成两层**：把量化后的 A/B 反量化回 fp32 再 einsum，得到「隔离量化」参考，
`fp8_einsum` 输出与它的差只有 **~2–4e-3**（纯 bf16 输出舍入 + 累积顺序）；
而「隔离量化」参考与 fp64 真值的差（即 FP8 E4M3 量化本身）是 **~3–4e-2**。
这说明 kernel 的 block-scale 应用正确，误差主项就是 FP8 量化。

### 6.2 每头独立 GEMM 与头独立性（`[B]`）

- `bhr,hdr->bhd` 与「逐头 fp32 bmm（用同样的量化值）」relerr 仅 **2.497e-3**（bf16 输出舍入）。
- 把第 0 头的 B 全置零：输出头 0 **全 0**、其余头**逐位不变**——印证「头=独立 batch」且没有跨头串扰。

### 6.3 SF 布局（`[C]`）

`A` 的 SF 形状 `(4,8,8)` = `[b,h,r/128]`；`B` 的 SF 形状 `(8,4,8)` = `[h,d/128,r/128]`；
逐头 SFB 基址 stride = `4*8 = 32`，正是 §3.2 的 `shape_n_sfb*shape_k_scales`。

### 6.4 契约拒绝（`[D]`，真实 assert 站点）

| 非法输入 | 被拒位置 |
|---|---|
| SM90 请求 FP8 `(d,sfd)` 输出对 | `einsum.hpp:224`（`arch==10`） |
| A 非 K-major | `einsum.hpp:206` |
| `recipe=(1,64,64)` | `layout.hpp:119`（SF 形状不符；kernel 端另有 `einsum.hpp:246`/`1d2d.cuh:56`） |
| B 的 `h` 与 A 不符 | `einsum.hpp:214` |
| `bhd,hdr->bhr` on sm90 | `einsum.hpp:296`（Unsupported） |

### 6.5 确定性开关与 kernel 名（`[E]`）

`use_deterministic_algorithms(True/False)` 对 `fp8_einsum` **没有影响**——因为 cuBLASLt 回退只写在
BF16 `bhr_hdr_bhd`（`einsum.hpp:81-83`），FP8 路径恒走自家 kernel。两次都命中
`sm90_fp8_gemm_1d2d_impl<..., 132u, (GemmType)4, cutlass::bfloat16_t>`，
其中 **`GemmType)4` = `Batched`**（模板实参 `num_groups`/`kGemmType` 直接坐实）。

### 6.6 性能（`[F]`）

FLOPs = `2·b·h·r·d`，输入已量化；对照是 `torch.einsum`（BF16，非 FP8，故对照不同口径）。

| b | h | r | d | deep_gemm | TFLOPS | GB/s | torch(bf16) | 比 |
|---|---|---|---|---|---|---|---|---|
| 4 | 8 | 4096 | 1024 | 13.3 us | 20 | 2536 | 25.7 us | 1.93× |
| 32 | 8 | 4096 | 1024 | 12.4 us | 173 | 2838 | 26.8 us | 2.17× |
| 128 | 8 | 4096 | 1024 | 16.9 us | 509 | 2369 | 30.1 us | 1.78× |
| 4096 | 8 | 4096 | 1024 | 473 us | **581** | 505 | 749 us | 1.58× |
| 8192 | 8 | 4096 | 1024 | 1029 us | 534 | 432 | 1544 us | 1.50× |
| 4096 | 128 | 512 | 128 | 374 us | 184 | 1122 | 487 us | 1.30× |
| 8192 | 128 | 512 | 128 | 735 us | 187 | 1130 | 1011 us | 1.38× |

**规律**：`r=4096,d=1024`（大收缩维）在 `b≥128` 后进入 TFLOPS 平台期，峰值 **~581 TFLOPS**；
`r=512,d=128`（小 GEMM）只有 ~185 TFLOPS——因为 `(m,n,k)=(b,128,512)` 的 N/K 都很小，
每个头一个 GEMM，**粒度太细、SM 占用与 SF 搬运占比高**。小 `b`（4/32）时是「延迟 + 带宽受限」
（GB/s 已到 2500–2800，接近 HBM 上限），不是算力受限。

## 7. 与 BF16 einsum 的关系、与 grouped GEMM 的区别

- **同一数学、不同量化骨架**：BF16 版（第 18 篇）走 `sm90_bf16_gemm.cuh` 的 Batched；
  FP8 版走 `sm90_fp8_gemm_1d2d.cuh` 的 Batched。两者 host 侧都用 `(batch,m,n,k)` 轴序 +
  3D TMA，内核的组枚举逻辑一致（`scheduler/gemm.cuh:169-172/247-259`）。
- **与 grouped GEMM 的本质区别**：grouped（`m_grouped_*`/`k_grouped_*`）的「组」是**不等长的 token 段**，
  组内 GEMM 形状不同、靠 `grouped_layout` 前缀和调度；einsum 的「组」是**等长的头**，
  `num_groups=h` 只是 TMA 第 3 维，所有组形状相同。这也是为什么它复用 `Normal` 的块计数。
- **与 Mega MoE 的关系**：Mega MoE 的专家 GEMM 也是 FP8 block-scale，其 SF 同样要在
  `transform_sf_into_required_layout` 里转成内核要求的布局。本篇的 `1d2d`（A 1D / B 2D）
  与 host 侧 `gran_k==128` 契约，正是 Mega MoE/前向 per-128 block-scale 家族在
  einsum 投影上的一次独立复现。

## 8. 结论与边界

1. **可跑**：H100 上 `fp8_einsum('bhr,hdr->bhd')` 完整可用，语义正确（隔离量化后 ~2–4e-3，
   量化本身 ~3e-2），头独立、SF 布局契约与源码逐条对上。
2. **可跑证据**：大收缩维峰值 **581 TFLOPS**；小 `r/d` 受粒度限制仅 ~185 TFLOPS；
   小 `b` 受延迟/带宽限制。
3. **仅分析**：FP8 `(d,sfd)` 输出对、自定义 epilogue、`bhd,hdr->bhr` / `bhd,bhr->hdr`
   三条依赖 `Kernel1D1D` + SM100，`einsum.hpp:224/277/287` 明确 arch10 限定——**SM100 专属**。
4. **契约**：SM90 只支持 gran 128、K-major、BF16 D；本条路径**不受**确定性开关影响
   （无 cuBLASLt 回退），恒走 `sm90_fp8_gemm_1d2d_impl<... GemmType::Batched ...>`。
