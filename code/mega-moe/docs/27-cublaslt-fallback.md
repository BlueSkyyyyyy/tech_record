# DeepGEMM 的 cuBLASLt 回退路径与选核：当它不用自己的 kernel

> 前面 26 篇拆的都是 DeepGEMM **自己的** JIT kernel。但 DeepGEMM 里藏着一个容易被忽略的事实：
> 在某些情形下，**它会悄悄把活儿交给 cuBLASLt**。本篇把这条「回退路径」讲清：什么时候回退、
> 回退时怎么把 row-major 的 GEMM 翻译成 cuBLASLt 的列主序调用、以及为什么这是合理的工程取舍。
> 全部结论在 **H100（sm90）真机实测**，用 `torch.profiler` 抓实际发射的 kernel 名作为证据。
>
> 源码坐标：`~/github/DeepGEMM`，commit `057ca59`（Public release 26/09/30）。
> 涉及 `csrc/apis/gemm.hpp`、`csrc/apis/einsum.hpp`、`csrc/runtime/runtime.hpp`、
> `csrc/jit_kernels/impls/smxx_cublaslt.hpp`、`csrc/jit_kernels/impls/epilogue_class.hpp`。

## 0. 结论先行（一句话）

**稠密 BF16 GEMM 和按头 batched BF16 einsum（plain-store、可选 alpha）默认走 cuBLASLt**；
**FP8/FP4 GEMM 永远走 DeepGEMM 自己的 kernel**；打开 `use_deterministic_algorithms(True)` 或用
`set_num_sms(<device_sms>)` 限制 SM 数，就会把 BF16 也切回 DeepGEMM kernel。
H100 上两者性能几乎持平（0.97–1.02×），所以默认回退是「省事又不亏」的选择。

## 1. 触发条件：三个与门

回退不是全局开关，而是嵌在每个 API 的分派处。以稠密 BF16 为例（`gemm.hpp:518-523`）：

```cpp
// Dispatch into different implements; cuBLASLt only covers a plain store, optionally alpha-scaled
if (not heuristics_runtime->get_deterministic_algorithms() and runtime->is_cublaslt_available()
    and epilogue_alpha.has_value()) {
    cublaslt_gemm(a, b, d, m, n, k, major_a, major_b, c.has_value(), epilogue_alpha.value());
    return;
}
```

三个条件缺一不可：

1. **`not deterministic_algorithms`**（`config.hpp:33-35` → `heuristics/runtime.hpp:20-22`）。
   用户显式要求确定性算法时，不再用 cuBLASLt —— 因为 cuBLASLt 的算法选择/累加顺序不保证可复现。
   实测：默认 kernel 名是 `nvjet_sm90_...`；`use_deterministic_algorithms(True)` 后变成
   `deep_gemm::sm90_bf16_gemm_impl<...>`。
2. **`is_cublaslt_available()`**（`runtime.hpp:89-91`）：
   ```cpp
   bool is_cublaslt_available() { return get_num_sms() == jit->device.get_num_sms(); }
   ```
   即「没有用 `set_num_sms` 限制过 SM 数」。这是一个很巧的代理条件：`set_num_sms` 通常是
   为了在别的 kernel 上留出 SM（例如与通信重叠），此时把 BF16 GEMM 丢给无法被约束 SM 数的
   cuBLASLt 就破坏了一致性，于是干脆自己也别用。实测：`set_num_sms(124)` 后 bf16 GEMM 变回
   `deep_gemm::sm90_bf16_gemm_impl<...>`。
3. **`epilogue_alpha.has_value()`**：只有 **plain store（可选乘 alpha）** 这一种 epilogue 能被
   cuBLASLt 表达。见下节。

### 1.1 epilogue → alpha 的映射表

`resolve_epilogue_class`（`jit_kernels/impls/epilogue_class.hpp:111-127`）把「用户 epilogue 类 / 独立 alpha 参数 /
FP8 `(d,sfd)` 输出对」统一成一个 `EpilogueClass`，再由 `get_alpha()`（`jit_kernels/impls/epilogue_class.hpp:35`）
决定能否交给 cuBLASLt：

| EpilogueClass | `get_epilogue_operator_type` | `get_alpha()` | 可回退 cuBLASLt？ |
|---|---|---|---|
| `Identity`（默认，`jit_kernels/impls/epilogue_class.hpp:41-48`） | `Identity` | **`1.0f`** | 是 |
| `Alpha`（`jit_kernels/impls/epilogue_class.hpp:50-65`） | `ScaleByAlpha` | `alpha` | 是 |
| `FP8Quantization`（`jit_kernels/impls/epilogue_class.hpp:67-89`） | `QuantizeToFP8` | `nullopt` | 否 |
| `BF16StochasticRounding`（`jit_kernels/impls/epilogue_class.hpp:91-106`） | `StochasticRoundToBF16` | `nullopt` | 否 |

关键点：`Identity` 返回 `1.0f` 而不是 `nullopt`，所以**默认的「裸 GEMM」也会走 cuBLASLt**。
这不是 bug —— 它意味着 DeepGEMM 认为「单纯 A@B^T（+可选 alpha）这种最普通的形状，cuBLASLt
足够好」。

实测印证：
- `bf16_gemm_nt(a,b,d)` 默认 → `nvjet_sm90_tst_256x128_64x4_1x2_h_bz_coopA_TNT`（cuBLASLt）。
- `bf16_gemm_nt(a,b,d, alpha=0.5)` → 仍 `nvjet_*`，结果 = `0.5*(A@B^T)`（relerr 1.99e-4，即 bf16 舍入级）。
- `bf16_gemm_nt(a,b,d, epilogue=deep_gemm.epilogue.Alpha(2.0))` → 仍 `nvjet_*`。
- 在 SM90 上传 `epilogue=FP8Quantization(sfd)` 且 `d` 是 bf16 → 在 `gemm.hpp:119` 断言
  `epilogue_class == nullptr and "Custom epilogues require SM100"`（FP8 量化 epilogue 是 SM100 专属）。

## 2. 哪些 API 有回退、哪些没有

回退只出现在「**稠密/可映射为单个或 strided-batched GEMM**」的入口；**分组（grouped）算子没有**
（分组布局无法用一次 cuBLASLt 调用表达）。

| API | 源码位置 | 回退条件 | 实测默认选核 |
|---|---|---|---|
| `bf16_gemm_nt/nn/tn/tt` | `gemm.hpp:519-523` | 三与门 | **cuBLASLt** |
| `fp8_fp4_gemm_nt/nn/tn/tt` | `gemm.hpp:108-132` | **无** | DeepGEMM (`sm90_fp8_gemm_1d2d`) |
| `m_grouped_bf16_gemm_*` | `gemm.hpp:608-622` | 无 | DeepGEMM |
| `m_grouped_fp8_fp4_gemm_*` | `gemm.hpp:235-254` | 无 | DeepGEMM |
| `einsum("bhr,hdr->bhd")` | `einsum.hpp:80-83` | 三与门 | **cuBLASLt** |
| `einsum("bhd,hdr->bhr")` | `einsum.hpp:114-117` | 三与门 | **cuBLASLt** |
| `einsum("bhd,bhr->hdr")` | `einsum.hpp:128-148` | **无条件**（源码注释 `TODO: investigate cuBLAS determinism.`） | **cuBLASLt** |
| `fp8_einsum("bhr,hdr->bhd")` | `einsum.hpp`（FP8 分支） | 无 | DeepGEMM `sm90_fp8_bmm` |

> 为什么 FP8 不回退？因为 DeepGEMM 的 per-block scaling（UE8M0/UE4M3 的 block-scale GEMM）是
> 它自己的看家本领，cuBLASLt 只在 NVFP4 上有对应的 VEC16 UE4M3 支持（见 §4.1），无法覆盖
> DeepGEMM 的通用 FP8 recipe。所以 FP8 永远走自家 kernel。

实测（einsum，B=16,H=16,R=512,D=512）：
- `einsum("bhr,hdr->bhd")` 默认 → `nvjet_sm90_tst_64x16_64x16_2x1_v_bz_TNT`；
  deterministic → `deep_gemm::sm90_bf16_gemm_impl<...>`。
- `einsum("bhd,hdr->bhr")` 默认 → `nvjet_sm90_..._NNT`；deterministic → `sm90_bf16_gemm_impl`。
- `einsum("bhd,bhr->hdr")`（输出 h,d,r）→ **无论 deterministic 与否**都是
  `nvjet_sm90_tss_128x256_64x4_2x1_v_bz_coopA_NTN`：源码里 `bhd_bhr_hdr`（`einsum.hpp:128-148`）
  **没有任何 DeepGEMM 替代实现**，直接无条件调用 `cublaslt_bhd_bhr_hdr`
  （`smxx_cublaslt.hpp:222-235`），注释写着 `// TODO: investigate cuBLAS determinism.`。

## 3. 翻译层：row-major DeepGEMM ↔ column-major cuBLASLt

cuBLASLt 是**列主序**的。DeepGEMM 的 `gemm_nt` 语义是 `D[M,N] = A[M,K] @ B[N,K]^T`（row-major）。
核心技巧在 `cublaslt_gemm`（`smxx_cublaslt.hpp:115-135`）：

```cpp
const auto trans_a = b_major == Major::K ? CUBLAS_OP_T : CUBLAS_OP_N;   // 用 B 当 cuBLAS 的 A
const auto trans_b = a_major == Major::K ? CUBLAS_OP_N : CUBLAS_OP_T;   // 用 A 当 cuBLAS 的 B
const auto layout_a = ... get_cublaslt_layout(cuda_type_a, k, n, ...);  // rhs → [k,n]
const auto layout_b = ... get_cublaslt_layout(cuda_type_b, k, m, ...);  // lhs → [k,m]
const auto layout_d = ... get_cublaslt_layout(cuda_type_d, n, m, ...);  // out → [n,m]
call_cublaslt_api(trans_a, trans_b, layout_a, layout_b, layout_d, lhs, rhs, out, accumulate, alpha);
```

即：利用 `(A B^T)^T = B A^T`，把「row-major 的 `D=A@B^T`」重写成「column-major 的
`D^T=B@A^T`」。于是 cuBLASLt 的 A 操作数其实是 DeepGEMM 的 `rhs`（`B`），B 操作数是 `lhs`（`A`），
`layout_d` 的 rows/cols 是 `(n,m)`。`call_cublaslt_api` 里真正传参时也确实是
`b.data_ptr()` 当 A、`a.data_ptr()` 当 B（`smxx_cublaslt.hpp:98-99`）。

末尾几个 API 只是对 `cublaslt_gemm` 的转置包装（`gemm.hpp:742-755`）：
`cublaslt_gemm_nn = nt(a, b.transpose(0,1))`，`tn = nt(a.transpose(0,1), b.transpose(0,1))`，
`tt = nt(a.transpose(0,1), b)`。注意这些是**视图**（`transpose`），所以传入的 `stride` 在
`get_cublaslt_layout` 里被当作 leading dim 用（`smxx_cublaslt.hpp:128-132`），并不会真的搬数据。

### 3.1 描述符里的几个关键开关（`call_cublaslt_api`，`smxx_cublaslt.hpp:28-113`）

- **`CUBLAS_COMPUTE_32F_FAST_TF32`**（`:46`）：默认用 TF32 快速累加 —— 注意这与 DeepGEMM kernel
  的 bf16/FP8 累加路径不同，是「cuBLASLt 版」的数值特性。带 block-scale 时改为
  `CUBLAS_COMPUTE_32F`（`:46`）。
- **`CUBLASLT_MATMUL_DESC_FAST_ACCUM` 显式关闭**（`:69-71`）：对 FP8 输入，DeepGEMM 主动关掉
  cuBLASLt 的 FP8 fast-accum，说明它不想让 cuBLASLt 在 FP8 上用低精度累加（虽然 FP8 本来也不走回退，
  这里更多是给「显式 `cublaslt_*` API」用的防护）。
- **`CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`**（`:66-67`）：把目标 SM 数设为 `runtime->get_num_sms()`。
  这解释了 §1 的第 2 个条件：自动回退永远发生在 `num_sms == device_num_sms`，所以 target 就是满卡；
  但**显式调用** `cublaslt_gemm_nt` 时若限制过 num_sms，target 会跟着变小。
- **workspace 大写固定在 16 MiB**（`runtime.hpp:25-26`）：
  ```cpp
  // cuBLAS will select worse heuristics when workspace > 16 MiB with shape, e.g. m=128, n=7168, k=16384
  static constexpr size_t kCublasLtWorkspaceSize = 16 * 1024 * 1024;
  ```
  注释直接说明「workspace 超过 16 MiB 时 cuBLAS 会选到更差的启发式」，所以刻意封顶。
  workspace 按 **CUDA stream 分别缓存**（`runtime.hpp:27-28`），因为 cuBLASLt 会异步用它，
  不同 stream 不能共享。
- **handle 自建 vs 借 PyTorch 的**（`runtime.hpp:40-65`）：默认自己 `cublasLtCreate`，因为
  `at::cuda::getCurrentCUDABlasLtHandle` 在某些 PyTorch 版本 CPU 开销很大；可用
  `DG_USE_PYTORCH_CUBLASLT_HANDLE=1` 切回。`DG_USE_TEMP_CUBLASLT_WORKSPACE=1` 则每次调用临时建
  workspace（给 compute-sanitizer 用，避免 CUDA driver 卸载后析构报错，`runtime.hpp:45-48`）。
- **算法选择**（`:79-91`）：`cublasLtMatmulAlgoGetHeuristic` 只取 **1** 个结果，且
  `reduction_scheme_mask` 限定为 `NONE | COMPUTE_TYPE`，即不允许 3-pass/16-bit 等会改变数值的
  归约方案。然后 `cublasLtMatmul` 里 `D=C` 同指针（`:100-102`），由 `beta` 决定是否累加（`:94`）。

## 4. einsum 与 batched 的几何映射

`smxx_cublaslt.hpp` 后半段是一组把「带 batch 维的 einsum」压成 strided-batched GEMM 的 helper，
它们用 `cublasLtMatrixLayoutSetAttribute` 的 `BATCH_COUNT` / `STRIDED_BATCH_OFFSET`
（`:19-24`）把 batch 编码进 layout：

- **`cublaslt_bhr_hdr_bhd`**（`:187-202`）：`bhr,hdr->bhd` → `(m,n,k)=(d,b,r)`，
  trans=`T,N`，三个 layout 带头偏移（`h` 作 batch）。实测默认走它（kernel `..._TNT`）。
- **`cublaslt_bhd_hdr_bhr`**（`:205-220`）：`bhd,hdr->bhr` → `(m,n,k)=(r,b,d)`，trans=`N,N`。
- **`cublaslt_bhd_bhr_hdr`**（`:222-235`）：`bhd,bhr->hdr` → `(m,n,k)=(r,d,b)`，trans=`N,T`，
  输出 dtype 固定 `CUDA_R_32F`（FP32）。
- **`cublaslt_batched_gemm`**（`:137-163`）+ `batched_syrk`/`batched_symm`（`gemm.hpp:796-833`）：
  `batched_syrk` 是 `D_i = A_i @ A_i^T`，`batched_symm` 是 `D_i = A_i @ B_i`（A 对称约定），
  都走 strided-batched 回退。

实测正确性（vs 逐 batch 的 fp32 bmm，relerr 即 bf16 舍入级）：
`cublaslt_gemm_nt` 1.99e-4、`cublaslt_gemm_tt` 1.99e-4、`cublaslt_gemm_nt`+C（beta=1）6.58e-4、
`batched_syrk` 1.72e-3、`batched_symm` 1.66e-3。

### 4.1 NVFP4 的 block-scale（`cublaslt_nvfp4_gemm`）

这是回退里唯一带 block-scale 的路径（`smxx_cublaslt.hpp:165-185`）：
- 只支持 **NVFP4 recipe**：UE4M3 scale + 16 元素 block，用
  `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` + A/B scale pointer 描述（`:56-64`）。
- 形状用**逻辑 FP4 元素**数（每字节 2 个），所以 leading dim 乘 2（`:176-180`）。
- 输入必须是 NT（两操作数 K-major），映射到 cuBLASLt 的列主序 TN（`:170-173`）。
- SF 的 cuBLASLt tiled 布局是 `[ceil(mn/128), ceil(k/64), 32, 4, 4]` 字节（`gemm.hpp:774-780`
  的 `check_sf`：`numel == ceil_div(mn,128)*ceil_div(k,64)*512`）。

> NVFP4 的 block-scaled matmul 本质是 SM100 的 tensor core 特性，H100 上能否真正执行取决于
> cuBLASLt 是否给出了 SM90 kernel；本篇不深入，仅指出 host 契约。

## 5. H100 实测：cuBLASLt vs DeepGEMM，几乎一样快

脚本 `harness/exp_cublaslt_fallback.py`（`.sh` / `.out.txt`）。同一 shape 下比较
「默认（cuBLASLt）」与「`use_deterministic_algorithms(True)`（DeepGEMM SM90 kernel）」的耗时。
BF16 `A[M,K] @ B[N,K]^T`，iters=50，warmup=10：

| M | N | K | cuBLASLt TFLOPS | DeepGEMM TFLOPS | DeepGEMM/cuBLASLt |
|---:|---:|---:|---:|---:|---:|
| 1 | 7168 | 7168 | 2.6 | 2.7 | 1.00×（延迟受限） |
| 16 | 7168 | 7168 | 43.5 | 42.1 | 1.03× |
| 128 | 7168 | 7168 | 331.9 | 327.7 | 1.01× |
| 1024 | 7168 | 7168 | 356.1 | 367.3 | 0.97× |
| 4096 | 7168 | 7168 | 377.0 | 376.9 | 1.00× |
| 8192 | 7168 | 7168 | 369.2 | 366.8 | 1.01× |
| 4096 | 14336 | 7168 | 368.5 | 366.5 | 1.01× |
| 4096 | 7168 | 2048 | 393.6 | 387.2 | 1.02× |

**结论**：稠密 BF16 上两者在 H100 打平（都在 ~370–390 TFLOPS 的 HBM 带宽平台上）。
这正好解释了 DeepGEMM 为什么敢把「裸 BF16 GEMM」默认丢给 cuBLASLt —— 自己的 kernel 并没有
性能优势，而 cuBLASLt 省去 JIT 编译、随 cuBLAS 升级自动受益。DeepGEMM 的价值集中在
**block-scale FP8/FP4、grouped、mega-kernel** 这些 cuBLASLt 覆盖不到的地方。

同时注意：本机实测显示 `nvjet_sm90_*` 在 BF16 也不比 DeepGEMM 差，说明「回退 ≠ 更慢」。

## 6. 与前面各篇的关系 / 为什么值得单独讲

- 第 8 篇（`bmk,bnk->mn` einsum）、第 18 篇（BF16 MLA 投影）、第 25 篇（SF 布局）都提过
  「确定性开关会切 kernel」。本篇把这条线索**收口成一套完整的选核规则**：确定性、
  `set_num_sms`、epilogue 三者如何决定走自家还是 cuBLASLt。
- 对 MoE 反向的启示：训练里若要**可复现**（research/CI 对拍），应显式
  `deep_gemm.use_deterministic_algorithms(True)`；若在推理里做 SM 预留（如与通信重叠），
  `set_num_sms` 会顺带禁用 BF16 的 cuBLASLt 回退 —— 这是一个**跨 kernel 的隐式副作用**，容易踩。
- 对前向 Mega MoE 的启示：Mega MoE 是 FP8/FP4 grouped，完全不碰这条回退；但 MoE 里那些
  「dense BF16 投影」（如 mHC 的 `fn` 投影若用 BF16）在产品里可能实际跑在 cuBLASLt 上，
  做性能预算时要意识到这一点。

## 7. 踩坑与边界

1. **默认走 cuBLASLt 是「隐形」的**：用 profiler 才能看出。只看 `bf16_gemm_nt` 的返回值和
   correctness 看不出来，容易误以为在测 DeepGEMM kernel。
2. **deterministic 切换的是全局状态**，影响之后所有调用，C++ 侧无 per-call 开关
   （`config.hpp:33-35` 是全局 `heuristics_runtime`）。实验里务必 `reset`。
3. **`set_num_sms` 的副作用是双向的**：它既改 DeepGEMM 的调度，又通过
   `is_cublaslt_available()` 关掉 BF16 回退。
4. **`cublaslt_gemm_*` 是独立公开 API**（`gemm.hpp:981-990` bindings），可以直接调，不受
   三与门限制；此时 `SM_COUNT_TARGET` 跟随当前 `num_sms`。
5. **`einsum("bhd,bhr->hdr")` 没有 DeepGEMM 替代实现**：任何架构、无论确定性开关都走 cuBLASLt
   （`einsum.hpp:128-148` 直调 `cublaslt_bhd_bhr_hdr`，源码留了 `TODO` 查 determinism）。

## 8. 行号索引（核对过）

- `gemm.hpp:17-52` `early_return`（空问题/`k==0` 早退，`d` 类型、`sfd` 与 fp8 D 必须成对）。
- `gemm.hpp:108-132` FP8 dense 分派（**无** cuBLASLt）；`:119` SM90 拒绝自定义 epilogue。
- `gemm.hpp:515-523` BF16 dense 的三与门回退。
- `gemm.hpp:723-755` 显式 `cublaslt_gemm_nt/nn/tn/tt`；`:757-789` NVFP4；`:796-833` batched syrk/symm。
- `einsum.hpp:80-83 / 114-117 / 146-147` 三个按头分支的回退。
- `runtime.hpp:25-28` 16 MiB workspace + 按 stream 缓存；`:40-65` handle；`:66-76` workspace；
  `:89-91` `is_cublaslt_available`。
- `smxx_cublaslt.hpp:28-113` `call_cublaslt_api`（compute type、FAST_ACCUM、SM_COUNT_TARGET、
  heuristic、matmul）；`:115-135` 列主序交换；`:137-235` batched/einsum/nvfp4 helper。
- `csrc/jit_kernels/impls/epilogue_class.hpp:35 / 41-48 / 50-65 / 67-89 / 91-106 / 111-127` alpha 与 resolve。
- `config.hpp:33-35` `use_deterministic_algorithms`；`:15-26` `set/get_num_sms`。

## 附：实测原始输出

见 `harness/exp_cublaslt_fallback.out.txt`（H100 80GB HBM3，132 SM，CUDA 13.2 / PyTorch 2.12 /
DeepGEMM 2.8.1）。实验脚本 `harness/exp_cublaslt_fallback.py`，驱动 `harness/exp_cublaslt_fallback.sh`。
