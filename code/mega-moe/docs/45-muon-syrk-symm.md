# DeepGEMM 的 Muon / 对称矩阵 batched 算子：`batched_syrk` 与 `batched_symm`

> 前面第 27 篇讲了 DeepGEMM「什么时候把稠密 GEMM 交给 cuBLASLt」。本篇拆另外**两个从头到尾
> 就是 cuBLASLt-only 的算子**：`deep_gemm.batched_syrk`（`D = A·Aᵀ`）与 `deep_gemm.batched_symm`
> （`D = A·B`，`A` 对称）。它们看起来只是「批 GEMM 的语法糖」，实际上是为 **Muon 优化器**的
> Newton-Schulz 迭代量身做的窄接口，藏着三处非平凡设计：① 2D/3D 统一、② 老实地把
> **strided-batch** 暴露给 cuBLASLt（因此接受非连续 batch）、③ 强制 `FAST_TF32` 计算类型
> （所以 `float32` 输入的尾数会掉到 TF32）。全部结论在 **H100（sm90）真机实测**。
>
> 源码坐标：`~/github/DeepGEMM`，commit `057ca59`（Public release 26/09/30）。
> 涉及 `csrc/apis/gemm.hpp`、`csrc/apis/einsum.hpp`、`csrc/jit_kernels/impls/smxx_cublaslt.hpp`、
> `csrc/utils/layout.hpp`、`deep_gemm/__init__.py`、`tests/generators.py`、`tests/test_bf16.py`。

## 0. 结论先行

1. **它们是「无自研 kernel」的 API**：`batched_syrk`/`batched_symm` 在 `deep_gemm/__init__.py:33-36`
   被归到 `# cuBLASLt Kernels` 组，直接绑定 `_C`，**没有任何 `sm90_*/sm100_*` 分支**，也更谈不
   上回退——唯一实现就是 `cublaslt_batched_gemm`（`smxx_cublaslt.hpp:137-164`）。实测 profiler 抓到的
   全是 `nvjet_sm90_*`（cuBLASLt 的 Blackwell 前身内核）。
2. **它们接受 strided-batch**：`batched_syrk:802-803` 用的是 `get_major_type_ab<false>` /
   `check_major_type_cd<false>`（模板参数 `kRequireContiguousBatch=false`），**故意不加**「batch 维
   必须连续」的断言（`layout.hpp:19-28`），把批 stride 原样传给 cuBLASLt。实测带 padding 的
   非连续 batch（`stride(0)=16809999 ≠ rows·rows`）照样算对。
3. **`float32` 输入实际走 TF32**：`call_cublaslt_api:45-46` 固定用 `CUBLAS_COMPUTE_32F_FAST_TF32`
   （只有带 block-SF 的 NVFP4 才用 `CUBLAS_COMPUTE_32F`）。实测 `float32` syrk 对 fp32 参考
   relerr ~1.4–2.1e-4——正是 TF32 十位尾数的量级，与 dtype 表面上的「fp32 精度」不符。
4. 两者都**只接受行主序输出**（`check_major_type_cd` → `stride(-1)==1`），形状/dtype 契约严格
   （`gemm.hpp:803-806`），四类非法输入全部被 `RuntimeError` 拒绝。
5. 关系更近的 `einsum("bhd,bhr->hdr")`（`einsum.hpp:128-148`）也是**无条件 cuBLASLt**（源码
   `:146` 注释「TODO: investigate cuBLAS determinism」），但它是跨 batch 求和的权重梯度，
   走的是另一条 `cublaslt_bhd_bhr_hdr`（`smxx_cublaslt.hpp:223-234`）。

## 1. 为什么有这两个算子：Muon 的 Newton-Schulz

Muon 优化器对动量矩阵做正交化，用 Newton-Schulz 迭代逼近
`G (GᵀG)^{-1/2}`，核心原语是 **`A·Aᵀ`（syrk）**；当更新作用在对称矩阵上时又需要 **`A·B`（symm，
`A` 对称）**。这两步都在**每层参数矩阵**上做，所以天然是 `[num_batches, rows, cols]` 的批量形态。

DeepGEMM 直接把这个需求固化成两个 API。`tests/generators.py:176-191` 写明了来源：

```python
def enumerate_batched_syrk_symm() -> Generator:
    # Real Muon parameter shapes from 5120/7168 models. Batch merging is capped at 16 in hai-llm.
    b_m_k_list = [
        (16, 4608, 5120), (16, 1536, 5120), (16, 576, 5120),
        (8, 24576, 1536), (8, 32768, 512), (4, 5120, 16384),
        (8, 4096, 7168), (8, 7168, 2048),
    ]
    for dtype in (torch.bfloat16, torch.float):
        reset_seed()
        for b, m, k in b_m_k_list:
            yield b, m, k, dtype
```

注释里的两个信息很关键：**① 形状取自 5120/7168 的真实模型；② batch 合并上限 16**（`hai-llm`
把多个小矩阵在 batch 维拼起来以喂饱大 GEMM）。所以这里 `rows = min(m,k)`、`cols = max(m,k)`，
`batch ∈ {4,8,16}`，没有任何「大 M 大 N」的常规 GEMM 形状——这正是为什么不用自研 kernel：
它是一堆 **batch 多、单矩阵小、K 或 M 有一个很长** 的批操作，cuBLASLt 的启发式已经很够用。

## 2. host 入口逐段精读（`csrc/apis/gemm.hpp`）

### 2.1 `get_cublaslt_batched_view`：2D/3D 统一（`:791-794`）

```cpp
static auto get_cublaslt_batched_view(const torch::Tensor& tensor) {
    DG_HOST_ASSERT(tensor.dim() == 2 or tensor.dim() == 3);
    return tensor.dim() == 2 ? tensor.unsqueeze(0) : tensor;
}
```

两个算子都允许「无 batch 的 2D」或「带 batch 的 3D」；2D 被就地 `unsqueeze(0)` 成 `batch=1`。
实测：2D 与显式 `batch=1` 的 3D 结果 `calc_diff=0`（同一实现，`unsqueeze` 只是换 view）。

### 2.2 `batched_syrk`（`:796-810`）

```cpp
static void batched_syrk(const torch::Tensor& a, const torch::Tensor& d) {
    // D = A @ A.mT. A and D may be either unbatched 2D tensors or batched 3D tensors.
    const auto a_3d = get_cublaslt_batched_view(a);
    const auto d_3d = get_cublaslt_batched_view(d);
    const auto [num_batches, m, k] = get_shape<3>(a_3d);
    const auto [num_batches_, m_, n_] = get_shape<3>(d_3d);
    const auto major_a = get_major_type_ab<false>(a_3d);
    check_major_type_cd<false>(d_3d);
    DG_HOST_ASSERT(num_batches == num_batches_ and m == m_ and m == n_);
    DG_HOST_ASSERT(a_3d.scalar_type() == d_3d.scalar_type());

    if (num_batches == 0 or early_return(m, m, k, d_3d, std::nullopt))
        return;
    cublaslt_batched_gemm(a_3d, a_3d, d_3d, m, m, k, num_batches, major_a, major_a);
}
```

要点：**同一个 `a_3d` 同时当 lhs 和 rhs**（`A·Aᵀ`），`(m,n,k)` 传 `(m,m,k)`；major 用同一个。
注意 `early_return(m, m, k, ...)`（`:806`）——`d` 不是可选的 C，这里是借用「空批 / 零形状」的
统一早退逻辑。

### 2.3 `batched_symm`（`:812-832`）

```cpp
static void batched_symm(const torch::Tensor& a, const torch::Tensor& b,
                         const torch::Tensor& d) {
    // D = A @ B. A is symmetric by contract; cuBLASLt executes this as strided-batched GEMM.
    const auto a_3d = get_cublaslt_batched_view(a);
    const auto b_3d = get_cublaslt_batched_view(b);
    const auto d_3d = get_cublaslt_batched_view(d);
    const auto [num_batches, m, m_]   = get_shape<3>(a_3d);
    const auto [num_batches_b, m_b, k] = get_shape<3>(b_3d);
    const auto [num_batches_d, m_d, k_d] = get_shape<3>(d_3d);
    const auto major_a = get_major_type_ab<false>(a_3d);
    check_major_type_cd<false>(d_3d);
    DG_HOST_ASSERT(num_batches == num_batches_b and num_batches == num_batches_d);
    DG_HOST_ASSERT(m == m_ and m == m_b and m == m_d and k == k_d);
    DG_HOST_ASSERT(a_3d.scalar_type() == b_3d.scalar_type() and a_3d.scalar_type() == d_3d.scalar_type());

    if (num_batches == 0 or early_return(m, k, m, d_3d, std::nullopt))
        return;

    const auto b_transposed = b_3d.transpose(1, 2);
    const auto major_b = get_major_type_ab<false>(b_transposed);
    cublaslt_batched_gemm(a_3d, b_transposed, d_3d, m, k, m, num_batches, major_a, major_b);
}
```

和 syrk 的差别有两处值得记：

- **「A 对称」只是契约，不是运行时检查**（注释 `:814`）。`batched_symm` 并不验证 `a` 是否真的
  对称，它把 `D=A@B` 当作普通 strided-batched GEMM 算。实测里我们**主动构造对称 `A`**（`a+aᵀ`），
  否则与参考不一致。
- **B 先转置**：`b_transposed = b_3d.transpose(1,2)`（`:830`），再对转置后的 B 重新取 major。
  这与 `bhd,hdr->bhr` 里「内部把 B 转置」是同一套路，见 `smxx_cublaslt.hpp:215-219`。

### 2.4 为什么是 `<false>`：放宽 batch 连续性（`csrc/utils/layout.hpp:19-40`）

```cpp
template <bool kRequireContiguousBatch = true>
static void major_check(const torch::Tensor& t) {
    const auto dim = t.dim();
    DG_HOST_ASSERT(dim == 2 or dim == 3);
    if constexpr (kRequireContiguousBatch) {
        if (dim == 3)
            DG_HOST_ASSERT(t.stride(0) == t.size(-2) * t.size(-1));
    }
    DG_HOST_ASSERT(t.stride(-2) == 1 or t.stride(-1) == 1);
}
```

`kRequireContiguousBatch=true` 时（大多数 DeepGEMM API 的默认）会断言 `stride(0)==rows·cols`，
即 batch 维必须紧挨着铺开。而 syrk/symm **显式传 `<false>`**（`gemm.hpp:802-803`、`:821-822`），
只要求「最后两维之一是 stride 1」。这让它们能接受**带 padding 的批张量**——只要把批 stride
交给 cuBLASLt 即可。实测：

```
pad=(0, 0)   a.stride(0)=16777216 连续 batch    calc_diff=1.196e-08
pad=(3, 5)   a.stride(0)=16809999 非连续 batch   calc_diff=2.242e-08
```

即 Muon 场景下「每层矩阵长度不同、需要在一块大 buffer 里按批对齐 padding」的常见做法被直接支持。

## 3. cuBLASLt 翻译层（`csrc/jit_kernels/impls/smxx_cublaslt.hpp`）

### 3.1 列主序操作数交换（`:137-164`）

cuBLASLt 是列主序 API。DeepGEMM 不看 row-major 的 `A@Bᵀ`，而是利用
`(A Bᵀ)ᵀ = B Aᵀ`，把 lhs/rhs 的关键字**互换**，让 cuBLASLt 的列主序解释算出转置后的结果：

```cpp
static void cublaslt_batched_gemm(const torch::Tensor& lhs, const torch::Tensor& rhs,
                                  const torch::Tensor& out,
                                  const uint32_t& m, const uint32_t& n, const uint32_t& k,
                                  const uint32_t& num_batches,
                                  const cute::UMMA::Major& a_major, const cute::UMMA::Major& b_major) {
    const auto trans_a = b_major == cute::UMMA::Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;
    const auto trans_b = a_major == cute::UMMA::Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;
    const auto lhs_batch_offset = lhs.stride(0);
    const auto rhs_batch_offset = rhs.stride(0);
    const auto out_batch_offset = out.stride(0);
    ...
    const auto layout_a = b_major == cute::UMMA::Major::K ?
        get_cublaslt_layout(cuda_type_a, k, n, rhs.stride(-2), num_batches, rhs_batch_offset) :
        get_cublaslt_layout(cuda_type_a, n, k, rhs.stride(-1), num_batches, rhs_batch_offset);
    const auto layout_b = a_major == cute::UMMA::Major::K ?
        get_cublaslt_layout(cuda_type_b, k, m, lhs.stride(-2), num_batches, lhs_batch_offset) :
        get_cublaslt_layout(cuda_type_b, m, k, lhs.stride(-1), num_batches, lhs_batch_offset);
    const auto layout_d = get_cublaslt_layout(cuda_type_d, n, m, out.stride(-2),
                                              num_batches, out_batch_offset);
    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, false);
}
```

三处非平凡点：

- **`trans_a` 由 `b_major` 决定、`trans_b` 由 `a_major` 决定**（互相反着来），这就是「交换操作数」
  在 `cublasOperation_t` 上的体现。
- **batch 偏移直接取 `stride(0)`**（`:143-145`），不做任何「必须紧凑」的假设——和 §2.4 的 `<false>`
  一脉相承。`stride(0)` 对 2D `unsqueeze` 出来的 batch=1 是 `rows·cols`，对带 padding 的 3D 是
  实际的批间距。
- 注意 `call_cublaslt_api(..., false)` 尾参是 `accumulate=false`：syrk/symm **没有 C 输入**，
  `D = A·Aᵀ` 或 `D = A·B`，不累加。

### 3.2 strided-batch 布局属性（`:14-25`）

```cpp
static auto get_cublaslt_layout(const cudaDataType& type, const int& rows, const int& cols, const int64_t& ld,
                                const std::optional<int>& batch_count = std::nullopt,
                                const std::optional<int64_t>& batch_offset = std::nullopt) {
    cublasLtMatrixLayout_t layout;
    DG_CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&layout, type, rows, cols, ld));
    if (batch_count.has_value()) {
        DG_HOST_ASSERT(batch_offset.has_value());
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ...));
        DG_CUBLASLT_CHECK(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ...));
    }
    return layout;
}
```

`CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET` 正是「批 stride 可任意」的官方机制；
`batch_count`/`batch_offset` 是成对的 `optional`（`:17-18`），只有批算子才传。

### 3.3 计算类型：`FAST_TF32`（`:44-46`）

```cpp
// Block-scaled matmuls require the plain FP32 compute type
cublasComputeType_t compute_type = with_block_sf ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_TF32;
```

**所有非 block-scale 的 cuBLASLt 调用都用 `CUBLAS_COMPUTE_32F_FAST_TF32`**，包括 fp32 输入的
syrk/symm。这意味着 fp32 操作数会被降到 TF32（10 位尾数）参与乘法。实测 `float32` syrk 对 fp32
参考的 relerr 在 ~1.0e-4–2.1e-4，正对应 TF32 的舍入量级——如果你以为传 fp32 就是全精度，
这里会踩坑。（同一文件 `:61` 还设了 `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET = num_sms`，
与第 27 篇讲的一脉相承。）

## 4. 测试形状与断言（`tests/generators.py` / `tests/test_bf16.py`）

`test_cublaslt_batched_syrk`（`test_bf16.py:247-281`）与 `test_cublaslt_batched_symm`
（`:283-320`）是这两个算子的可执行契约。以 syrk 为例（`:247-280`）：

```python
for num_batches, m, k, out_dtype in enumerate_batched_syrk_symm():
    rows, cols = min(m, k), max(m, k)
    threshold = 6e-7
    for k_major in (True, False):
        for batch_shape, padding in (((), (3, 5)), ((num_batches,), (3, 5)), ((num_batches,), (0, 0))):
            shape = (rows, cols) if k_major else (cols, rows)
            a = torch.randn(*batch_shape, shape[0] + padding[0], shape[1] + padding[1],
                            device='cuda', dtype=out_dtype)[..., :shape[0], :shape[1]]
            if not k_major:
                a = a.mT
            ...
            d = torch.empty(*batch_shape, rows + padding[0], rows + padding[1], ...)[..., :rows, :rows]
            deep_gemm.batched_syrk(a, d)
            ref_d = (a.float() @ a.float().mT).to(out_dtype)
            diff = calc_diff(d, ref_d)
            assert diff < threshold
```

三个枚举维度覆盖了全部形态：**major（K 或 MN，靠 `.mT`）**、**batch_shape（2D、3D、或无 padding
的 3D）**、**padding（(3,5) 与 (0,0)）**。symm 版本（`:283-320`）把 major 扩成「A 与 B 各自独立」
的四组合 `(True,True),(True,False),(False,True),(False,False)`，并**显式给 `a` 上对称性**
（`a.copy_(a + a.mT)`）。`out_dtype` 遍历 `{bfloat16, float}`；参考一律先把操作数 `.float()`
再乘、最后再 `.to(out_dtype)`，于是断言只测「cuBLASLt 相对 fp32 理想结果」这一层误差
（threshold 6e-7 是归一化平方误差口径，等价于 ~1e-3 相对误差，容得下 bf16 输出舍入）。

## 5. H100 实测（`harness/exp_muon_syrk_symm.py`）

脚本对 `enumerate_batched_syrk_symm` 的 **8 个真实 Muon 形状 × {bf16, fp32}** 跑 syrk/symm，
`calc_diff` 归一化平方误差对拍、`bench_kineto` 计时。

### 5.1 `batched_syrk`（`D = A·Aᵀ`）

| batch | rows | cols | dtype | calc_diff | relerr | us | TFLOPS | GB/s |
|---|---|---|---|---|---|---|---|---|
| 16 | 4608 | 5120 | bf16 | 2.70e-8 | 2.33e-4 | 9400.0 | 370 | 153 |
| 16 | 1536 | 5120 | bf16 | 2.96e-8 | 2.43e-4 | 1063.0 | 364 | 308 |
| 16 | 576 | 5120 | bf16 | 3.95e-8 | 2.81e-4 | 254.0 | 214 | 414 |
| 8 | 1536 | 24576 | bf16 | 2.86e-7 | 7.56e-4 | 2451.0 | 379 | 262 |
| 8 | 512 | 32768 | bf16 | 3.74e-7 | 8.65e-4 | 259.8 | **529** | 1049 |
| 4 | 5120 | 16384 | bf16 | 1.48e-7 | 5.44e-4 | 9591.0 | 358 | 92 |
| 8 | 4096 | 7168 | bf16 | 3.30e-8 | 2.57e-4 | 5417.0 | 355 | 136 |
| 8 | 2048 | 7168 | bf16 | 3.89e-8 | 2.79e-4 | 1342.0 | 358 | 225 |
| 16 | 4608 | 5120 | fp32 | 2.08e-8 | 2.04e-4 | 19412.0 | 179 | 148 |
| 8 | 512 | 32768 | fp32 | 2.22e-8 | 2.11e-4 | 680.0 | 202 | 802 |

- **bf16 峰值 ~529 TFLOPS**（batch=8, 512×512×32768；长收缩维 + 小输出 → 计算偏重），
  典型形状稳定在 **~355–380 TFLOPS**。
- **fp32 只有 ~180 TFLOPS**，约为 bf16 的一半——因为 `FAST_TF32` 下 fp32 也是 TF32 乘法，
  而 fp32 操作数/输出带宽翻倍。
- relerr：bf16 ~1.4e-4–8.6e-4（输出 bf16 舍入）；fp32 ~1.0e-4–2.1e-4（**TF32 尾数**，非全精度）。

### 5.2 `batched_symm`（`D = A·B`，A 对称）

| batch | rows | cols | dtype | calc_diff | relerr | us | TFLOPS | GB/s |
|---|---|---|---|---|---|---|---|---|
| 16 | 4608 | 5120 | bf16 | 1.11e-8 | 1.49e-4 | 9831.0 | 354 | 223 |
| 8 | 4096 | 7168 | bf16 | 9.72e-9 | 1.39e-4 | 5415.0 | 355 | 223 |
| 8 | 512 | 32768 | bf16 | 8.96e-10 | 4.23e-5 | 487.2 | 282 | 1111 |
| 16 | 4608 | 5120 | fp32 | 4.31e-8 | 2.93e-4 | 22273.0 | 156 | 197 |
| 8 | 4096 | 7168 | fp32 | 4.30e-8 | 2.93e-4 | 12297.0 | 156 | 196 |

symm 的 bf16 吞吐与 syrk 相当（~350 TFLOPS），但 **fp32 的 relerr 整齐地落在 2.93e-4**
（所有形状几乎一致）：因为 `B` 未做对称化、其转置让 TF32 舍入系统性更均匀。同样地，fp32
吞吐 ~156 TFLOPS。

### 5.3 布局 / 形态 / 契约

```
syrk major=N stride=(4096,1)  calc_diff=1.457e-08
syrk major=T stride=(1,4096)  calc_diff=1.290e-08
2D vs 3D(batch=1)：d2 对称=True，d2/d3 calc_diff=0.000e+00（unsqueeze 同一实现）
pad=(0, 0)   a.stride(0)=16777216 连续 batch    calc_diff=1.196e-08
pad=(3, 5)   a.stride(0)=16809999 非连续 batch   calc_diff=2.242e-08
```

- K-major（`stride(-1)==1`）与 MN-major（`.mT`）都被接受且都正确；
- 2D 与 `batch=1` 的 3D **逐位相等**（`calc_diff=0`），坐实 §2.1；
- **非连续 batch 与连续 batch 都接受**，坐实 §2.4 的 `<false>` 放宽。

契约检查（四类非法输入全部被拒，均为 `RuntimeError`）：

```
D 非行主序(.mT)   → 拒绝   （check_major_type_cd：stride(-1)!=1）
D 形状不符        → 拒绝   （num_batches/m 不匹配，gemm.hpp:805）
a/d dtype 不符    → 拒绝   （gemm.hpp:806）
batch 数不符      → 拒绝   （gemm.hpp:805）
```

### 5.4 真实 kernel 名

```
batched_syrk: nvjet_sm90_tst_320x128_64x3_1x2_h_bz_coopB_TNT
batched_symm: nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_NNT
```

两者都是 cuBLASLt 的 `nvjet_sm90_*` 内核（`tst`/`coopB`/`NNT` 等编码 tile、stage 与
操作数转置），验证了「cuBLASLt-only」的判断。

### 5.5 `bhd,bhr->hdr`：跨 batch 权重梯度也走 cuBLASLt

`einsum("bhd,bhr->hdr")`（`einsum.hpp:128-148`）计算 `D[h,d,r] = Σ_b A[b,h,d]·B[b,h,r]`，
输出 **fp32**。它**无条件**调 `cublaslt_bhd_bhr_hdr`（`:147`，源码 `:146` 注释
「TODO: investigate cuBLAS determinism」），走 `smxx_cublaslt.hpp:223-234`：

```cpp
static void cublaslt_bhd_bhr_hdr(...) {
    const auto m = r, n = d, k = b;
    const auto trans_a = CUBLAS_OP_N;
    const auto trans_b = CUBLAS_OP_T;
    const auto layout_a = get_cublaslt_layout(CUDA_R_16BF, m, k, rhs.stride(0), h, rhs.stride(1));
    const auto layout_b = get_cublaslt_layout(CUDA_R_16BF, n, k, lhs.stride(0), h, lhs.stride(1));
    const auto layout_d = get_cublaslt_layout(CUDA_R_32F, m, n, out.stride(1), h, out.stride(0));
    call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate);
}
```

注意这里 **`h` 是 batch、`b` 是收缩维**：`k=b`（batch 当 K）、`trans_b=OP_T`。实测
`b=8,h=16,r=4096,d=2048`：`calc_diff=3.33e-16`、relerr 2.14e-8（**接近精确**，因 bf16 操作数
尾数 8 位 < TF32 10 位，且收缩维 K=b=8 极小），两次运行**逐位相等**（尽管有那条 TODO 注释，
此形状下确定性成立）；性能 266.6us / **8 TFLOPS** / **2020 GB/s**——K 太小，纯访存受限。

## 6. 与自研 kernel 的关系

把这三个算子和第 27 篇的「回退」放在一起看，DeepGEMM 的 cuBLASLt 用法分三类：

| 类别 | 代表 API | 有无自研 kernel | 何时用 cuBLASLt |
|---|---|---|---|
| 可回退 | `bf16_gemm_nt`、按头 batched einsum | 有（`sm90_*/sm100_*`） | 三与门满足时（第 27 篇） |
| **无条件** | `einsum("bhd,bhr->hdr")` | 无 | 永远（源码 TODO 未解） |
| **cuBLASLt-only** | `batched_syrk` / `batched_symm` | **无** | 唯一实现 |

`batched_syrk`/`batched_symm` 属于第三类：它们既不进 `get_best_config`（第 28 篇），也不参与
SM90/SM100 派发，是「用完就走」的窄接口。这也解释了为什么它们的 host 逻辑要比自研 kernel 的
入口简单得多——没有 block config、没有 JIT、没有 SF 布局，只剩形状/主序契约 + 一次翻译。

## 7. 小结

- `batched_syrk`/`batched_symm` 是 **Muon 优化器专用**、**cuBLASLt-only** 的两个 API，
  形状取自 5120/7168 真实模型、batch 上限 16（`generators.py:176-191`）。
- 三个设计要点：**2D↔3D 统一**（`get_cublaslt_batched_view`）、**放宽 batch 连续性**
  （`<false>` 模板参数 + `STRIDED_BATCH_OFFSET`）、**强制 `FAST_TF32`**（fp32 实际只有 TF32 精度）。
- 契约只有三条：最后两维之一 stride 1、输出行主序、形状/dtype/batch 一致——四类非法输入实测全被拒。
- H100 实测：bf16 峰值 **~529 TFLOPS**（典型 ~355–380），fp32 ~180；kernel 名为 `nvjet_sm90_*`；
  非连续 batch、2D、两种 major 全部算对。
- 相邻的 `einsum("bhd,bhr->hdr")`（跨 batch fp32 权重梯度）也是无条件 cuBLASLt，K=batch 极小
  时纯访存受限（实测 2020 GB/s / 8 TFLOPS）。
