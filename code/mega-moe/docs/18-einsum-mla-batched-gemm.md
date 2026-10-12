# DeepGEMM einsum `bhr,hdr->bhd` / `bhd,hdr->bhr`：按头分 batch 的投影 GEMM（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P8 增量（第 18 篇）。
> 第 17 篇拆了 einsum 家族的 `bmk,bnk->mn`（把 batch 并进收缩轴做 wgrad 归约）。
> 本篇补同一家族里另外两条**长度保持的投影式**：
> `bhr,hdr->bhd` 与 `bhd,hdr->bhr`——它们是**按 attention head 分组的 batched GEMM**，
> 不是 grouped GEMM，也不是归约 GEMM。
>
> 为什么与 Mega MoE 专题相关：MoE 里 linear1/linear2 的权重按**专家**分组（grouped），
> 但 MoE 之外还有一整类「权重被所有 token 共享、只是按结构维（头 / 隐空间维）切块」的
> 同权重 GEMM——DeepSeek 的 MLA（Multi-head Latent Attention）投影就是典型：
> 吸收矩阵 `W_UK:[h,d,r]`、`W_UV:[h,r,d]` 对每个头做一个小投影，
> `b`（token/chunk 数）在另一轴上。DeepGEMM 用一条 einsum 覆盖它。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`。
> 这两条表达式**有 SM90 实现，H100 可跑**；FP8 变体 `bhd,hdr->bhr` 是 SM100 专属（仅分析）。

## 1. 两条表达式的数学

两条表达式只是 **(m,n,k) 角色互换**：

```
bhr,hdr->bhd :  D[b,h,d] = Σ_r A[b,h,r]·B[h,d,r] = A[:,h,:] @ B[h].T      (m,n,k)=(b,d,r)
bhd,hdr->bhr :  D[b,h,r] = Σ_d A[b,h,d]·B[h,d,r] = A[:,h,:] @ B[h]        (m,n,k)=(b,r,d)
```

对**每个头 `h` 一个独立的 GEMM**，`h` 是 batch 维。这正是 MLA 里「同一个 token 批次
`b`、对每个头用各自的低秩矩阵做吸收投影」的形状：

| 表达式 | A | B（权重） | D | 收缩维 | 语义 |
|---|---|---|---|---|---|
| `bhr,hdr->bhd` | `[b,h,r]` | `[h,d,r]` | `[b,h,d]` | `r`（latent） | 低秩→头维（"absorb" / up-project） |
| `bhd,hdr->bhr` | `[b,h,d]` | `[h,d,r]` | `[b,h,r]` | `d`（head dim） | 头维→低秩（down-project） |

两者共享同一批权重 `B[h,d,r]`（测试里 `fy` 的切片），互为「转置方向的投影」。

## 2. host 分派与契约（`csrc/apis/einsum.hpp`）

入口 `einsum()`（`:150-179`）是一串硬编码的字符串分派：

```
:166  "bmk,bnk->mn"   -> bmk_bnk_mn
:169  "bhr,hdr->bhd"  -> bhr_hdr_bhd
:171  "bhd,hdr->bhr"  -> bhd_hdr_bhr
:173  "bhd,bhr->hdr"  -> bhd_bhr_hdr   （只走 cuBLASLt，见 §6）
```

### 2.1 `bhr_hdr_bhd`（`:60-92`）

契约（`:63-70`）：

```cpp
const auto [b , h  , r ] = get_shape<3>(A);   // A:[b,h,r]
const auto [h_, d  , r_] = get_shape<3>(B);   // B:[h,d,r]
const auto [b_, h__, d_] = get_shape<3>(D);   // D:[b,h,d]
DG_HOST_ASSERT(b==b_ and h==h_ and r==r_ and d==d_ and h==h__);
DG_HOST_ASSERT(A.scalar_type()==kBFloat16 and A.stride(2)==1);   // A 的收缩维 r 连续
DG_HOST_ASSERT(B.scalar_type()==kBFloat16 and B.stride(2)==1);   // B 的收缩维 r 连续
DG_HOST_ASSERT(D.stride(2)==1);                                  // D 的 n=d 连续
```

**三个 `stride(2)==1` 是关键**：它们把 A/B/D 都定为 K 轴（或 n 轴）连续，
于是 3D TMA 能用「最内维 = 收缩维」的连续 box 搬运，无需转置。

早退 `:73`（`h==0` 或零形状）后，分派（`:80-88`）：

```cpp
if (not deterministic and cublaslt_available and epilogue_alpha.has_value())
    cublaslt_bhr_hdr_bhd(...);              // :83  cuBLASLt (nvjet)
else if (arch_major == 9)
    DG_HOST_ASSERT(epilogue_class == nullptr and "Custom epilogues require SM100");
    sm90_bf16_bhr_hdr_bhd(A, B, C, D, b,h,r,d);   // :86
else if (arch_major == 10)
    sm100_bf16_bhr_hdr_bhd(A, B, C, D, b,h,r,d, "nk", resolved_epilogue_class);  // :88
```

### 2.2 `bhd_hdr_bhr`（`:94-126`）

同样的契约与分派，但形状角色不同（`:97-100`）：`A:[b,h,d]`、`B:[h,d,r]`、`D:[b,h,r]`。
`A.stride(2)==1`（d 连续）、`B.stride(2)==1`（**r 连续**）、`D.stride(2)==1`（r 连续）。

注意在 `bhd,hdr->bhr` 里，B 的「输出维 n=r」是连续的（`stride(2)==1`），
而收缩维 k=d 不是——所以它落到 **B 为 MN-major** 的实现（见 §3）。

## 3. 降维映射：这是一个 `GemmType::Batched`（SM90）

SM90 的两个 host 包装（`csrc/jit_kernels/impls/sm90_bf16_gemm.hpp`）把两条 einsum
翻译成**同一个 batched-GEMM desc**，只改 `(m,n,k)` 与 `major_b`：

`sm90_bf16_bhr_hdr_bhd`（`:332-384`）：

```cpp
:339  .gemm_type = GemmType::Batched,
:341  .m = b, .n = d, .k = r, .num_groups = h,          // 头数 h 就是 group 数
:344  .major_a = Major::K, .major_b = Major::K,         // A/B 都 K(=r) 连续
:345  .with_accumulation = tensor_c.has_value(),
```

`sm90_bf16_bhd_hdr_bhr`（`:386-438`）：

```cpp
:393  .gemm_type = GemmType::Batched,
:395  .m = b, .n = r, .k = d, .num_groups = h,
:398  .major_a = Major::K, .major_b = Major::MN,        // B 的 n(=r) 连续 => MN-major
:399  .with_accumulation = tensor_c.has_value(),
```

**为什么 A 总是 K-major？** 因为契约要求 `A.stride(2)==1`，而 A 的第 2 维在两条里都是
**收缩维**（`bhr` 的 r、`bhd` 的 d）。B 的第 2 维在 `hdr` 里恒为 r：对 `bhr,hdr->bhd`
r 是收缩维（K-major），对 `bhd,hdr->bhr` r 是输出维 n（MN-major）。同一个物理张量
`B[h,d,r]`，在两条表达式里**轴的角色不同，故 major 不同**——但都只靠 TMA 描述符
表达，kernel 不变。

**为什么不是 grouped？** grouped GEMM 每组的权重张量不同（按专家分段）；这里
`num_groups=h` 但**每组权重是同一张量 `B` 的一个切片**，只是按第三维索引，
没有「不等长分组 / 组号表」的概念——所以用 `Batched` 而非 `MGrouped*`。

## 4. 3D TMA：把「头」编码成第三维（`runtime_utils.hpp:161-202`）

Batched 的实现核心是 **3 维 TMA 描述符**。以 `bhr_hdr_bhd` 的 A 为例
（`sm90_bf16_gemm.hpp:353-356`）：

```cpp
make_tma_3d_desc(tensor_a,
                 /* gmem */ r, b, h,          // [内层=r, 中层=b, 外层=h]
                 /* smem */ block_k, load_block_m, 1,
                 tensor_a.stride(0),          // 中层 b 的 stride = h*r  (A:[b,h,r] stride(0))
                 tensor_a.stride(1),          // 外层 h 的 stride = r
                 swizzle_a_mode);
```

`make_tma_3d_desc`（`:161-202`）把它填进 `cuTensorMapEncodeTiled` 的 3 维形式：

```cpp
:187  const cuuint64_t gmem_dims[3]    = {gmem_dim_0, gmem_dim_1, gmem_dim_2};
:188  const cuuint32_t smem_dims[3]    = {smem_dim_0, smem_dim_1, smem_dim_2};
:189  const cuuint64_t gmem_strides[2] = {gmem_stride_0*elem, gmem_stride_1*elem};
```

即 **TMA 的第三维就是「头」**，用两个显式 stride（`stride(0)`, `stride(1)`）定位每个头
的数据块。**这就是为什么 B 允许带 padding**：测试里 `fy:[h,d,r+128]` 而 `y=fy[:,:,:r]`，
`y.stride(1)=r+128`——把它作为 `gmem_stride_1`（外层头 stride）传给 TMA，
TMA 会自动跳过每头之间 128 列的 padding，**无需把 B 拷成连续**（§7.4 实测逐位相等）。

### 4.1 kernel 里 batch 维如何参与

`sm90_bf16_gemm.cuh` 的 `GemmType::Batched` 分支：

```cpp
:186  constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
:187  const uint32_t batch_idx = (kIsBatchedMM ? scheduler.current_group_idx : 0);
:189  tma::copy<..., kIsBatchedMM>(&tensor_map_a, &full_barrier, smem_a[stage],
                                   k_a_idx, m_idx, num_tma_multicast_a, batch_idx);
```

`tma::copy` 的 `kIs3DTMA` 模板开关（`common/tma_copy.cuh:91-104`）决定用
`SM90_TMA_LOAD_3D`（`:23-27`）还是 2D，并把 `batch_idx` 作为**第三个坐标**传给 TMA：

```cpp
:99   if constexpr (kIs3DTMA)
:100      copy_nd<..., 3>(..., inner_idx, outer_idx, batch_idx);
:102      copy_nd<..., 2>(..., inner_idx, outer_idx);
```

所以「头」既在 TMA 描述符里（stride 布局），也在每个 TMA 指令的坐标里（`batch_idx`）。
**一个 TMA 描述符 + 一个坐标 = 选定某个头的某个 (m,k) 块**。

### 4.2 调度器：Batched 的编号方式（`scheduler/gemm.cuh`）

`Scheduler` 构造（`:90`）：

```cpp
if constexpr (kGemmType==Normal or kGemmType==Batched)
    num_blocks = num_m_blocks * num_n_blocks;
```

与 Normal 相同——**每组的块数是常量**（因为每组形状一样），总块数 = `num_blocks * h`。
取块（`:247-262`）：

```cpp
:248  if (next_block_idx >= num_blocks * kNumGroups) return false;
:251  current_group_idx = next_block_idx / num_blocks;      // 头号 = 块号 / 每组块数
:252  const auto block_idx = next_block_idx - current_group_idx * num_blocks;
:253  if constexpr (kIsMulticastOnA) { m_block_idx = block_idx / num_n_blocks; ... }
```

即**先枚举头（外层），再枚举每组内的 (m,n) 块（内层）**，各 SM 用
`(++current_iter)*kNumSMs + blockIdx.x` 做 round-robin（`:210` 前缀逻辑），共享一个
持久化循环。`get_global_idx` 的 Batched 分支（`:169-173`）几乎不做偏移——**m/n/k 坐标与
头无关，头只体现在 TMA 的 `batch_idx` 上**。

**Batched 禁用 multicast**（`heuristics/sm90.hpp:63-67`）：`disable_multicast = (... or
desc.gemm_type == GemmType::Batched)`，理由是「不同头的权重不同、无法跨 CTA 共享」，
于是 cluster 恒为 1（`cluster_m*n=1`，`:71-75`）。

### 4.3 写回：TMA store 3D / reduce-add 3D（`:374-379`）

```cpp
:374  if constexpr (kGemmType == GemmType::Batched) {
:375      using cute_tma_t = cute::conditional_t<kWithAccumulation,
:376          cute::SM90_TMA_REDUCE_ADD_3D, cute::SM90_TMA_STORE_3D>;
:377      cute_tma_t::copy(&tensor_map_cd, smem_ptr,
:378                       n_block_idx*BLOCK_N + in_block_n_offset, m_idx,
:379                       scheduler.current_group_idx);      // 第三个坐标 = 头
```

D 同样是 3 维 TMA（`sm90_bf16_gemm.hpp:364-367`：`[d,b,h]` 布局）。
**累加（`C` 提供）时换成 `SM90_TMA_REDUCE_ADD_3D`**：把 bf16 tile 直接 `+=` 到全局 D，
不经过「读 D + 加法 + 写 D」三步——这是 §7.3 里 `D=C+A@B` 的硬件实现。

### 4.4 实测确认的 kernel 实例

在 H100 上用 profiler 抓到的实例（`b=128,h=128,r=512,d=128`，`bhr,hdr->bhd`）：

```
sm90_bf16_gemm_impl<Major::K, Major::K, 0u, 128u, 512u, 128u,
                    128u, 64u, 128u, 128u,128u,128u, 6u,
                    128u, 256u, 1u, false, 132u,
                    (deep_gemm::GemmType)4, false, bfloat16_t>
```

模板参数按序：`SHAPE_M=0(动态) SHAPE_N=128(=d) SHAPE_K=512(=r) kNumGroups=128(=h)
BLOCK_M=128 BLOCK_N=64 BLOCK_K=128 stages=6 ... kGemmType=4(=Batched)`。
**`kNumGroups=h=128` 直接坐实「头=group」**；`SHAPE_M=0`（动态）因为 `m=b` 不编译期固定。

## 5. SM100 变体（仅分析）

`csrc/jit_kernels/impls/sm100_bf16_gemm.hpp` 里两条 host 包装与 SM90 结构一致
（`:343`/`:395` 同为 `GemmType::Batched`），3D TMA 坐标相同（`:356-364` / `:408-416`），
差异在计算本体：UMMA/tcgen05 + TMEM，而非 SM90 的 WGMMA/寄存器累加。
SM100 才能接受自定义 epilogue（`einsum.hpp:85`/`:119` 的 assert）与
**BF16 stochastic rounding**（`tests/test_einsum.py` 里 `arch_major==10` 才测）。

本机 H100（sm90）**跑不了** SM100 分支，故仅源码分析。

## 6. FP8 变体 `fp8_einsum`（`einsum.hpp:251-298`）

`fp8_einsum` 与 BF16 版是两条**平行**链路：它先把 `(data, sf)` 按目标 einsum 做
permute 以凑成 `fp8_bmm` 需要的 `(batch,m,n,k)` 顺序（`:262-294`）：

| 表达式 | 实现 | 说明 |
|---|---|---|
| `bhr,hdr->bhd` | `fp8_bmm`（`:276`），**SM90+SM100** | A/B 都 K-major；输出 SF 与 `d.flatten(1,2)` 对齐（`:218-226`） |
| `bhd,hdr->bhr` | `fp8_bmm`（`:286`），**仅 arch10** | B 需 `permute({0,2,1})` 变 MN-major |
| `bhd,bhr->hdr` | `fp8_bmm`（`:294`），**仅 arch10** | 跨 batch wgrad（`recipe=(1,1,128)`） |

`:262-276` 里显式做了 `permute({1,0,2})` 把 `(b,h,·)` 变成 `(h,b,·)` 以匹配 `fp8_bmm` 的
`(batch,m,n,k)=(h,b,d,r)`；注释（`:265-267`）解释 SF 列布局对 `(b,h)` 置换不变，
所以数据置换不影响 SF。SM90 走 `sm90_fp8_bmm`（`sm90_fp8_gemm_1d2d.hpp:283-349`，
`GemmType::Batched` + 3D TMA）；`bhd,hdr->bhr` 与 `bhd,bhr->hdr` 的 FP8 版在
sm90 被 `:277`/`:287` 的 `arch_major==10` 拒绝。**SM100 专属，仅分析。**

## 7. H100 实测（`harness/exp_einsum_mla.py` / `.out.txt`）

环境：H100 80GB HBM3 / sm90 / 132 SM，CUDA 13.2，PyTorch 2.12，DeepGEMM `057ca59`。

### 7.1 语义：bf16 输出与 torch.einsum **逐位相等**（§A）

16 组形状（`h,r,d ∈ {(128,512,128),(8,4096,1024)}`，`b ∈ {4,32,128,4096}`）：
两条表达式**全部 `torch.equal(z, torch.einsum(...)) == True`**（逐位），
相对 fp64 参考 relerr **2.0e-3 ~ 3.5e-3**（bf16 舍入量级）。
说明 DeepGEMM 的 fp32 累加 + bf16 舍入与 torch 的 bf16 einsum 走的是同一条数值路径。

### 7.2 映射验证（§B）

- `bhr,hdr->bhd` vs 「逐头 `bmm`（`x.permute(1,0,2) @ y.mT`）」relerr 2.3e-3 —— 语义等价。
- **独立性**：把头 0 的 B 置零 → 输出头 0 **全 0**，其余头**逐位不变**（`True`/`True`）——
  直接证明「每个头是互不干扰的独立 GEMM」。

### 7.3 累加 `D=C+A@B`（§C，TMA REDUCE_ADD_3D）

`b=32,h=8,r=4096,d=1024`：`bhr,hdr->bhd` reldiff(fp64)=5.3e-3、`bhd,hdr->bhr` 5.2e-3。
（比 §7.1 略大是因为 `c` 也是 bf16、且累加进 D 又舍入一次。）

### 7.4 B 带 padding（§D）

`b=128,h=8,r=4096,d=1024`，`fy:[h,d,r+128]` 取 `y=fy[:,:,:r]`（`stride(1)=4224`）：
- padded B vs 连续 B 输出**逐位相等** = True；
- vs `torch.einsum` **逐位相等** = True。

**这就是 3D TMA 外 stride 的价值**：MLA 里 `W` 常按 `[h, d, r_padded]` 对齐存放
（如 flash-attn 风格的 r+128 padding），这个 einsum 直接吃 stride 视图、不用重排。

### 7.5 确定性开关 = kernel 二选一（§E）

`b=128,h=128,r=512,d=128`，profiler 抓到的实际 kernel：

| `use_deterministic_algorithms` | kernel | 耗时 | reldiff(fp64) |
|---|---|---|---|
| **True** | `deep_gemm::sm90_bf16_gemm_impl<..., GemmType)4, ...>` | 10.0us | 2.165e-3 |
| **False** | `nvjet_sm90_tst_128x128_64x6_1x1_h_bz_TNT` | 9.5us | 2.165e-3 |

**确定性开关真的切换实现**：False 时走 cuBLASLt（nvjet），两者数值一致（reldiff 相同）。
这解释了 `einsum.hpp:81` 的分支——`epilogue_alpha.has_value()` 为真时优先 nvjet，
只有要求确定性（或自定义 epilogue / 非 SM100）才用自己的 kernel。**严谨起见**：
nvjet 更快约 5%（小形状），大形状差距见 §7.7。

### 7.6 契约（§F）

| 非法输入 | 结果 |
|---|---|
| `A.stride(2) != 1` | 拒（`einsum.hpp:68`） |
| A 为 fp32 | 拒（`einsum.hpp:156`，einsum 顶层要求 bf16） |
| B 的 `h` 不匹配 | 拒（`einsum.hpp:66`） |
| 未知表达式 | 拒（`einsum.hpp:177`） |

### 7.7 性能（§G）

`FLOPs = 2·b·h·r·d`，对比 `torch.einsum`（同一表达式）。`x = torch_us / deep_gemm_us`：

`bhr,hdr->bhd`：

| b | h | r | d | deep_gemm | TFLOPS | GB/s | torch | x |
|---|---|---|---|---|---|---|---|---|
| 4 | 128 | 512 | 128 | 5.5us | 12 | 3147 | 5.2us | 0.95 |
| 32 | 128 | 512 | 128 | 5.9us | 91 | 3716 | 5.7us | 0.96 |
| 128 | 128 | 512 | 128 | 10.1us | 212 | 3731 | 8.7us | 0.86 |
| 4096 | 128 | 512 | 128 | 489us | 141 | 1407 | 488us | 1.00 |
| 8192 | 128 | 512 | 128 | 1021us | 135 | 1331 | 1008us | 0.99 |
| 128 | 8 | 4096 | 1024 | 31.0us | 277 | 2505 | 30.2us | 0.97 |
| 4096 | 8 | 4096 | 1024 | 764us | 360 | 527 | 748us | 0.98 |
| 8192 | 8 | 4096 | 1024 | 1733us | 317 | 426 | 1544us | 0.89 |

`bhd,hdr->bhr`：

| b | h | r | d | deep_gemm | TFLOPS | GB/s | torch | x |
|---|---|---|---|---|---|---|---|---|
| 128 | 128 | 512 | 128 | 10.5us | 204 | 3591 | 11.5us | 1.09 |
| 4096 | 128 | 512 | 128 | 658us | 105 | 1046 | 555us | 0.84 |
| 8192 | 8 | 4096 | 1024 | 1725us | 319 | 428 | 1645us | 0.95 |

观察：

- **`deep_gemm` 与 torch 基本同速（0.84~1.09×）**。这类「小 `m`、多 batch」的 GEMM，
  cuBLASLt/torch 已经做得很好，DeepGEMM 的 packed 3D-TMA 主要是**省显存/免重排**，
  不是刷峰值。
- **小 `b` 延迟受限**：`b=4` 只有 12 TFLOPS（5.5us 里大部分是 launch/TMA 固定开销）；
  `b` 增大到 128 冲到 212~277 TFLOPS，之后**回落**——`b=8192,h=128` 掉到 135，
  因为 `k=r=512` 小、`m·n=128·128` 且 batch 巨大时改为**带宽受限**（GB/s 1331）。
- **`h=8,r=4096,d=1024`（大 k）最好**：`b=4096` 达 **360 TFLOPS**，因为 `k=d=1024`
  深、算术强度高（GB/s 仅 527），接近 compute-bound。
- **两条表达式吞吐相近**，但 `bhd,hdr->bhr` 在大 `b` 时略慢（105 vs 141）——它的 B 是
  MN-major，TMA 加载 B 的 box 形状不同（`[BLOCK_N, BLOCK_K]` vs `[BLOCK_K, BLOCK_N]`），
  大 `b`/带宽受限时更吃亏。

## 8. 与专题的联系与结论

- **einsum 家族第三种收缩形态**：`bmk,bnk->mn`（第 17 篇）是**归约式**（输出维在输入里
  被消掉），本篇两条是**投影式**（输出维在输入里保留、只是换轴）。DeepGEMM 用同一个
  `GemmType::Batched` 机制承载，区别只在 `(m,n,k)` 与 `major_b`。
- **「头=group」是纯布局概念**：`num_groups=h`、每组权重同源（`B` 的切片），
  靠 3D TMA 的第三维 + `batch_idx` 实现，没有组号表、没有不等长——这与 grouped GEMM
  的「按专家分段」是两种完全不同的并行轴。
- **对 MLA 训练/推理的意义**：吸收矩阵投影天然是按头 batched 的 GEMM，且权重常带
  padding（`r+128`）。这条 einsum 直接吃 stride 视图（§7.4），省掉一次
  `permute+contiguous`；带 `C` 时走 TMA reduce-add（§7.3），适配「累加进已有 buffer」的
  训练图。
- **性能定位**：它不是用来刷峰值的（与 torch 同速），而是**免重排、免中间量、
  布局原生**。真正需要 tiling 优化的 MoE wgrad/dgrad 仍是第 10~14 篇的 grouped GEMM。
- **边界**：SM100 分支、FP8 `bhd,hdr->bhr` / `bhd,bhr->hdr` 仅源码分析（需 SM100）；
  本机 H100 只覆盖 BF16 两条 + FP8 `bhr,hdr->bhd` 的 SM90 路径。

## 附录：实验复现

```
bash code/mega-moe/harness/exp_einsum_mla.sh > exp_einsum_mla.out.txt
```

环境：H100 80GB HBM3 / sm90 / 132 SM，CUDA 13.2，PyTorch 2.12，DeepGEMM 公开版
`057ca59`（editable 安装）。实验脚本同步放在 `harness/`，输出见 `*.out.txt`。
