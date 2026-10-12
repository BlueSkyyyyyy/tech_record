# DeepGEMM GEMM API 的输入主序与输出累积契约

> 本专题前面 46 篇（尤其第 10/12/13/22/27/28/32/35 篇）反复引用 `nt`、`K-major`、`m_grouped`、
> `C/D 累积`、`compiled_dims`，但一直把它们当「术语」用，从没系统讲过：**一次 GEMM 调用到底对
> 输入的主序（major）和输出的累积（C 是否参与、是否原地）有哪些硬约定，为什么这样定**。
> 本篇把 DeepGEMM 的 GEMM 入口层（`csrc/apis/gemm.hpp` + `csrc/utils/layout.hpp`）单独拆透。
>
> 一句话：**`nt` 是唯一实现，`nn/tn/tt` 只是把操作数转置后转调 `nt` 的薄壳；主序只由最后一维
> stride 决定；输出累积语义被 `early_return` 一个函数统一收口。** 全部结论在 H100 上实测核对。

参考源码（DeepGEMM commit `057ca59`，`~/github/DeepGEMM`）：

- `csrc/apis/gemm.hpp`：`early_return`（`:17-53`）、`check_k_grouped_args`（`:55-76`）、
  `fp8_fp4_gemm_nt/nn/tn/tt`（`:78-179`）、`m_grouped_*_nn_contiguous`（`:252-267`）、
  `bf16_gemm_nt/nn/tn/tt`（`:488-565`）、`m_grouped_bf16_gemm_nn_contiguous`（`:625-633`）、
  pybind 注册（`:835-994`）。
- `csrc/utils/layout.hpp`：`major_check`（`:18-27`）、`get_major_type_ab`（`:29-33`）、
  `check_major_type_cd`（`:35-40`）、`fp8_fp4_requires_k_major`（`:42-45`）、
  `get_logical_shape`（`:58-64`）、`check_ab_fp8_fp4`（`:66-73`）。
- 实测：`harness/exp_gemm_major_accum.{py,sh,out.txt}`（H100/sm90 真机）。

---

## 1. GEMM 入口层：一个基准 + 三个转置薄壳

DeepGEMM 的所有 GEMM 都遵循 **BLAS 的 `nt/nn/tn/tt` 命名**，其中字母表示**该操作数相对其
「规范存储」是否被转置**：

| 入口 | 调用形式 | 用户传的 A | 用户传的 B | 计算的数学 |
|---|---|---|---|---|
| `nt` | `D = A @ Bᵀ` | `[M, K]` | `[N, K]` | `A[M,K] @ B[N,K]ᵀ` |
| `nn` | `D = A @ B` | `[M, K]` | `[K, N]` | `A[M,K] @ B[K,N]` |
| `tn` | `D = Aᵀ @ B` | `[K, M]` | `[K, N]` | `A[K,M]ᵀ @ B[K,N]` |
| `tt` | `D = Aᵀ @ Bᵀ` | `[K, M]` | `[N, K]` | `A[K,M]ᵀ @ B[N,K]ᵀ` |

关键在于：**`nt` 是唯一的真实现，其余三个全部转置后转调 `nt`**。以 FP8/FP4 为例：

- `fp8_fp4_gemm_nn`（`gemm.hpp:135-148`）：`fp8_fp4_gemm_nt(a, {bᵀ, b_sfᵀ}, ...)`——**B 数据和 B 的
  scale factor 一起转置**（`:146`）。
- `fp8_fp4_gemm_tn`（`:150-164`）：A、B（含 SF）都转置（`:161-162`）。
- `fp8_fp4_gemm_tt`（`:166-179`）：只转置 A（`:177`）。

BF16 版本同理（`:537-565`）：`nn`→`b.transpose(0,1)`、`tn`→`a/b` 都转、`tt`→`a` 转。**没有第二份
kernel**，转置只改 shape/stride，不改数值——因此四个入口算出的结果应当**逐位相等**（见 §7 实测）。

> **为什么这么设计**：四个入口共享同一份经过极致调优的 kernel 与启发式；把「主序选择」压缩成
> 一次 `transpose(view)`，实现与测试面都最小。代价是转置产生的非连续/异常主序必须由
> 「主序契约」显式约束（§3）。

---

## 2. 主序（major）只由最后一维 stride 决定

深度学习里「K-major / MN-major」指张量在内存中**哪一维连续**。DeepGEMM 的判定极简，
在 `csrc/utils/layout.hpp`：

```cpp
// get_major_type_ab，layout.hpp:29-33
return t.stride(-1) == 1 ? cute::UMMA::Major::K : cute::UMMA::Major::MN;
```

- **`stride(-1) == 1` ⇒ K-major**：最后一维（对 `[M,K]` 是 K）连续。等价于「行主序的 `[M,K]`」。
- 否则 ⇒ **MN-major**：倒数第二维（MN）连续。

而 `major_check`（`:18-27`）先做两道前置校验：维度必须是 2 或 3（`:21`）、最后两维**至少一维
stride 为 1**（`:26`）——即必须是「两维里有一个连续」的合法矩阵视图，不能是任意 stride 的花式
张量。对 3D（batched）张量，默认还要求 **batch 维连续**（`:22-25`，`stride(0)==size(-2)*size(-1)`），
只有显式传 `get_major_type_ab<false>` 才放宽（Muon 的 `batched_syrk/symm` 用了这个，见第 45 篇）。

**输出 D 的主序是硬约束**——`check_major_type_cd`（`:35-40`）：

```cpp
// NOTES: the library only supports row-major output layouts   （layout.hpp:37）
DG_HOST_ASSERT(t.stride(-1) == 1);
```

也就是 **D 必须 N-major（行主序）**。原因：epilogue 的 TMA store（第 21 篇）与 STSM 都假定
输出按 N 方向连续铺开，只支持这一种。

> **和「输入可双主序」对比**：输入 A/B 允许 K-major 或 MN-major（kernel 用不同的 TMA 描述符与
> swizzle 适配），但输出只允许 N-major。这就是为什么 `gen_major_ab` 枚举里 A/B 各有两种、
> 而 D 永远只有一种。

---

## 3. 为什么 SM90 的 FP8/FP4 必须 K-major，SM100 却宽松

`fp8_fp4_requires_k_major`（`layout.hpp:42-45`）与 `fp8_fp4_gemm_nt`（`gemm.hpp:90-95`）：

```cpp
// layout.hpp:42-45
static bool fp8_fp4_requires_k_major(const torch::Tensor& a, const torch::Tensor& b) {
    return jit->device.get_arch_major() == 9 or
           (a.scalar_type() == kPackedFP4 and b.scalar_type() == kPackedFP4);
}
```

- **SM90（H100）恒为真**：FP8/FP4 的 A、B 都必须 K-major（`gemm.hpp:93-94`）。
  根因是 SM90 走 **WGMMA + 寄存器累加**（第 22 篇）：WGMMA 的 B 操作数只能从
  shared memory 以 K-major 布局喂入，硬件不提供 MN-major 的 B 路径。
- **SM100 只在「FP4 × FP4」时要求 K-major**：`tcgen05` 的 UMMA 支持 MN-major 操作数，
  唯独 **packed FP4 沿 K 两两打包**（一个字节装两个 K），缩放/解包路径要求 K 连续。
- **FP8/FP8 或 FP8/FP4 在 SM100 上可以 MN-major**。

推论：**在 H100 上，`fp8_fp4_gemm_nn` 对连续的 `[K,N]` B 必然被拒**——因为 `nn` 会把 B 转成
MN-major（`gemm.hpp:146`），立刻在 `:94` 断言失败。实测见 §7-B。

### 3.1 packed FP4 的「逻辑形状随主序翻倍」

FP4 每字节存两个元素，所以**逻辑 K 与物理存储不同**。`check_ab_fp8_fp4`（`layout.hpp:66-73`）：

```cpp
// layout.hpp:66-73
auto [mn, k] = get_shape<2>(ab);
if (ab.scalar_type() != torch::kFloat8_e4m3fn) {
    DG_HOST_ASSERT(ab.scalar_type() == kPackedFP4 and arch_major == 10);
    major == cute::UMMA::Major::K ? (k *= 2) : (mn *= 2);   // 打包维在主序方向上
}
```

- K-major 的 packed FP4 存储为 `[MN, K/2]`，逻辑 K = 物理末维 × 2（`:70`）。
- MN-major 的存储为 `[K, MN/2]`，逻辑 MN = 物理末维 × 2（`:70`）。

这条与第 37 篇的 `transpose_packed_fp4`（FP4 重打包转置）遥相呼应：**FP4 的「转置」不是换
stride，而是真的重排字节**，所以 `nn/tn/tt` 的 FP4 变体在 SM100 上需要重打包（见第 31 篇
`cast_fp8_fp4_with_major`）。

---

## 4. grouped 的 nn 变体与 K-grouped 的 tn/nt

### 4.1 `m_grouped_*_nn_contiguous`（dgrad 侧）

`m_grouped_fp8_fp4_gemm_nn_contiguous`（`:252-267`）与 `m_grouped_bf16_gemm_nn_contiguous`
（`:625-633`）都是**沿 `(1,2)` 转置 B**（`bᵀ[G,N,K]`），即用户传 `[G,K,N]`：

```cpp
// gemm.hpp:264
m_grouped_fp8_fp4_gemm_nt_contiguous(a, {b.first.transpose(1, 2), b.second.transpose(1, 2)}, ...);
```

注意它**没有 `expected_m_for_psum_layout` 参数**（`:266` 传 `std::nullopt`）——nn 变体不支持 PSUM。
对 BF16，`m_grouped` 的 A 恒 K-major（`:577`），B 可 K 或 MN，所以 H100 上 nn 变体可用（实测 §7-C）。

### 4.2 K-grouped 的 `tn_contiguous` vs `nt_contiguous`

wgrad（K 轴分组）在 SM90 与 SM100 走**不同的 major 约定**：

- `k_grouped_fp8_gemm_nt_contiguous`（`:373-427`）：**仅 SM90**（`:421`），A/B 为 K-major（`:423`
  传 `Major::K, Major::K`），因为 SM90 的 wgrad 就是「每组 `[M,k_g]` K-major 拼接」。
- `k_grouped_fp8_gemm_tn_contiguous`（`:320-371`）：**仅 SM100**（`:362`），A/B 为 MN-major
  `[sum_k, M]`/`[sum_k, N]`（`:365-366`）。这正是第 10/12 篇讲的两条路径。
- BF16 的 K-grouped 只有 `k_grouped_bf16_gemm_tn_contiguous`（`:676-722`）：SM90 走
  `Major::MN, Major::MN`（`:714`），且 **C 必填、必须累积**（`:710` `DG_HOST_ASSERT(c.has_value())`）。

> 命名有点反直觉：这里的 `tn/nt` 描述的是**「K 分组张量在内存里的主序」**，而非重走 §1 的
> 转置薄壳。「tn」= K 在 MN 维外侧（MN-major，K 为慢维），「nt」= K 为快维（K-major）。

---

## 5. 输出累积契约：`early_return` 一个函数收口

每一个 GEMM 入口在通过形状/dtype 校验后，都会先调一次 `early_return`
（`gemm.hpp:17-53`），由它统一处理「空问题、退化的 K、C/D 关系」。它返回 `true` 表示
「已处理完、直接返回，不要 launch kernel」。

### 5.1 空问题与 dtype 校验（`:20-37`）

```cpp
if (m == 0 or n == 0) return true;                       // :21-22  空转
DG_HOST_ASSERT(not sfd.has_value() or not c.has_value()); // :26  量化 D 永不累积
const bool is_cd_same = c.has_value() and c->data_ptr() == d.data_ptr(); // :27
if (is_cd_same) DG_HOST_ASSERT(c->sizes() == d.sizes() and c->strides() == d.strides()); // :28-29
DG_HOST_ASSERT(d 是 bf16/float/fp8_e4m3);                 // :30-31
DG_HOST_ASSERT((d 是 fp8) == sfd.has_value());            // :33  FP8 D 必带 SFD
if (c) { check_major_type_cd(c); assert(d.dtype == c.dtype); } // :34-37
```

三个「为什么」：

1. **量化 D 永不累积**（`:26`）：量化输出带 SFD（per-token/block scale），若再叠加一个 C，
   `D = C + A@B` 的 scale 无法用单一 SFD 表达。所以短暂的「FP8 输出 + C」组合被直接禁止。
2. **FP8 D 必带 SFD**（`:33`）：FP8 是「有损存储」，必须同时给出量化 scale（SFD），否则反量化
   无从谈起；反之给 SFD 就必须是 FP8 输出。
3. **C 与 D 必须同 dtype/同主序**（`:36`/`:35`）：kernel 用 `TMA reduce-add`（第 19/22 篇）
   把 C 直接加进 D，类型与步长必须逐位一致。

### 5.2 k == 0 的退化（`:40-47`）

```cpp
if (k == 0) {
    if (not is_cd_same) c ? d.copy_(c) : d.zero_();   // :41-42
    if (sfd) sfd->zero_();                            // :44-45
    return true;
}
```

`K==0` 时 `A@Bᵀ` 恒为 0，于是 `D = C`（`C` 存在）或 `D = 0`；**FP8 输出的 SFD 也要清零**
（`:44-45`），因为「零张量」的 scale 约定为 0（`d = 0 · 2^sf`，sf 取 0 安全）。这条在 MoE
里真实出现：某专家本 batch 没有 token（空组）。

### 5.3 C ≠ D 时先拷贝，C == D 时原地（`:27-29`、`:49-51`）

```cpp
// With accumulation, do copy before GEMM (assuming the GEMM kernel does not support different C/D)
if (c.has_value() and not is_cd_same) d.copy_(c.value());   // :50-51
```

两个含义：

- **C 与 D 不同张量**：先把 C 拷进 D，kernel 再做 `D += A@Bᵀ`。这样 kernel 只需支持「D 既是
  输入累加器又是输出」（原地），不必支持「C、D 两个不同地址」。
- **C 与 D 同地址（原地）**：`is_cd_same` 为真，跳过拷贝，kernel 直接原地累加。这是省一次全量
  读写的关键路径，也是 `assert(c->strides()==d->strides())`（`:29`）存在的原因——原地要求形状
  步长完全一致。

> **少数例外**：k-grouped 的 SM90 入口强制 `c.has_value()`（`gemm.hpp:403`/`710`），即 wgrad
> **必须**是累积模式（`D = C + Σ A_eᵀB_e`）；SM100 则可以直出（第 35 篇）。这是「累积契约」在
> 不同 arch 上的分支。

---

## 6. `compiled_dims` 默认值随入口变化

pybind 注册里一个容易忽略的细节（`gemm.hpp:838-955`）：

- `nt`/`nn` 的 `compiled_dims` 默认 **`"nk"`**（`:842`/`:850`/`:874`）：N、K 编进模板常量，
  M 是运行时维度（因为 m 在 MoE/m-grouped 里动态变化）。
- `tn`/`tt` 的默认是 **`"mn"`**（`:858`/`:866`）：A 被转置后 **m 与 k 角色互换**，于是把 M、N 编进
  模板，K 变运行时。
- K-grouped 全用 `"mn"`（`:900`/`:908`/`:916`）。

这直接反映在实测抓到的 kernel 模板实参里（§7-E）：`nt` 是 `(…, m=0, n=384, k=256, …)`，
`tn` 是 `(…, m=512, n=384, k=0, …)`——**同一形状、不同入口，编译期常量化的是不同的维**。

---

## 7. H100 真机实测

`harness/exp_gemm_major_accum.py`（H100 80GB / sm90 / 132 SM），要点：

### A. bf16 四个入口逐位相等，且 `nt` 双主序通吃

```
entry | eff.a eff.b |    passed a    passed b |  calc_diff     relerr
  nt |     K     K |    (256, 1)    (256, 1) |  3.892e-11  8.823e-06
  nn |     K    MN |    (256, 1)    (384, 1) |  3.892e-11  8.823e-06
  tn |    MN    MN |    (512, 1)    (384, 1) |  3.892e-11  8.823e-06
  tt |    MN     K |    (512, 1)    (256, 1) |  3.892e-11  8.823e-06
  nt vs nn/tn/tt 逐位相等 = True / True / True
  nt(MN-A,K-B)/(K-A,MN-B)/(MN-A,MN-B) 与 nt 逐位相等 = True
```

`eff.a/eff.b` 是转置后**真正送进 `nt` 的主序**（`nn` 转 B→MN、`tn` 两个都 MN、`tt` 只 A→MN）。
四个入口输出**逐位相同**，且 `nt` 自己用 `.T.contiguous().T` 造出的 MN-major A/B 也给出逐位相同
结果——**坐实「转置只改 stride、不改数学；输入双主序均可」**。

### B. major 检测与 SM90 K-major 硬约束

```
A [m,k] 连续                     stride=(256, 1) → K-major
A [m,k] MN-major(.T.contig().T)  stride=(1, 512) → MN-major
fp8 nt (A/B 都 K-major) → 通过  relerr=7.015e-04（SF 全 1，隔离纯 FP8 舍入）
fp8 nn (B→MN-major)     → 被拒 RuntimeError: gemm.hpp:94: major_b == cute::UMMA::Major::K
```

`nn` 在 H100 上必然被拒，与 §3 的源码推断一致。

### C. m_grouped bf16 nt vs nn

```
alignment=128  G=2 m=256 n=256 k=512
nt: major_b=K   calc_diff=2.160e-09  relerr=6.572e-05
nn: major_b=MN  calc_diff=2.160e-09  relerr=6.572e-05
nt vs nn 逐位相等 = True
```

BF16 的 m-grouped 接受 B 的两种主序。

### D. `early_return` 契约逐条

```
D1 无 C            : relerr=1.650e-03（纯乘积，bf16 输出 vs fp32 参考）
D2 分离 C          : relerr=2.339e-03（C+AB）；vs 纯 AB relerr=8.791e-02（C 确实参与了）
D3 原地 C==D       : relerr=2.339e-03；与 D2(分离C) 逐位相等 = True
D4 k==0, c=None    : D 全零 = True
D5 k==0, c=C       : D==C 逐位 = True
D6 m==0 / D7 n==0  : launch 空转、无异常（D 形状 [0,256] / [256,0]）
D8 C dtype 不符     : 被拒 RuntimeError（gemm.hpp:36）
D9 C 非行主序       : 被拒 RuntimeError（check_major_type_cd，gemm.hpp:35）
D10 fp8 入口给 FP8 D: 被拒 RuntimeError（fp8_fp4_gemm_nt 只收 bf16/fp32 D，gemm.hpp:106）
```

**D3 与 D2 逐位相等**是「C==D 原地」与「C≠D 先拷」语义等价的直接证据；`D4/D5` 坐实 k==0 退化。

### E. 真实发射的 kernel 名（profiler）

```
deterministic=False（默认，bf16 稠密→cuBLASLt）:
  nvjet_sm90_tst_64x40_64x16_2x4_h_bz_TNT
deterministic=True（强制 DeepGEMM）:
  nt: sm90_bf16_gemm_impl<(Major)0,(Major)0, 0u,384u,256u, ..., 128u,128u,...>
  nn: sm90_bf16_gemm_impl<(Major)0,(Major)1, 0u,384u,256u, ..., 128u, 64u,...>
  tn: sm90_bf16_gemm_impl<(Major)1,(Major)1, 512u,384u,  0u, ...,  64u,...>
  tt: sm90_bf16_gemm_impl<(Major)1,(Major)0, 512u,384u,  0u, ..., 128u,...>
```

三个观察：

1. **主序进了模板实参**：`(Major)0` = K、`(Major)1` = MN。`nt/nn/tn/tt` 的实参恰好是
   `(K,K)/(K,MN)/(MN,MN)/(MN,K)`——与 §1 的转置映射完全对应，**源码推断被 kernel 名坐实**。
2. **`compiled_dims` 进了模板实参**：`nt/nn` 是 `(0, 384, 256)`（m 运行时），`tn/tt` 是
   `(512, 384, 0)`（k 运行时）——与 §6 一致。
3. 默认走 **cuBLASLt `nvjet_sm90_tst_..._TNT`**（第 27 篇的三与门），`use_deterministic_algorithms`
   才强制 DeepGEMM；且 **block_n 会因主序不同而不同**（nt=128，nn/tn/tt=64），因为启发式里
   MN-major 操作数影响 swizzle/TMA 描述符的合法性（第 28 篇）。

---

## 8. 小结与边界

- **入口层极薄**：`nt` 一个实现，`nn/tn/tt` 全是 `transpose` 后转调；转置只改 shape/stride。
- **主序判定一句话**：`stride(-1)==1 ? K : MN`；输入双主序、输出强制 N-major。
- **arch 差异一句话**：SM90 FP8/FP4 必须 A/B 都 K-major（WGMMA）；SM100 仅 FP4×FP4 必须。
- **累积契约一句话**：`early_return` 统一了「m/n==0 空转、k==0 退化、C==D 原地、C≠D 先拷、
  量化 D 不累积」；K-grouped 的 SM90 wgrad 是唯一强制累积的例外。
- **边界（仅分析）**：SM100 的 MN-major FP8、FP4 重打包转置（`transpose_packed_fp4`）与
  `sm100_fp8_fp4_gemm_1d1d` 的 major 分支在 H100 上跑不了，只作源码/资料分析（第 31/35/37 篇）。
  本篇所有数字均来自 H100 真机（`harness/exp_gemm_major_accum.out.txt`）。

> **对 MoE 的意义**：前向/反向的 grouped GEMM 全是 `nt`/`tn` 家族的特例；MoE 的「空组」走
> k==0 退化、wgrad 走强制累积、dgrad 走 `m_grouped` + N-major 输出——本篇的契约正是第 9~14 篇
> 反向推导能成立的地基。
