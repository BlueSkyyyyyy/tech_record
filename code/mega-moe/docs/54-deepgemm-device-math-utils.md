# DeepGEMM 的设备端数学与工具层：UE8M0 取整、bf16 打包、warp/CTA 归约

> 本文对应路线图 **P44-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 54 篇。
> 前面 0~43 篇把一个个 kernel（GEMM / attention / grouped 反向 / mega moe / mega gate / mHC）
> 都拆过，但每个 kernel 都在调**同一层没有系统讲过**的设备端数学与工具函数——
> `deep_gemm/include/deep_gemm/common/{math,utils,types,packing,compile,exception,cute_tie}.cuh`。
> 本文补上这一层：讲清「UE8M0 scale 的整数进位取整、bf16/FP8 打包、warp/CTA 归约、TMEM 列对齐」
> 这些小函数**为什么长这样**，并在 H100(sm90) 上**直接编译运行真实设备代码**逐条验证数值。
>
> **可跑性**：`common/math.cuh` 除 `fma2` 的 SM100 快路（`__ffma2_rn`）外全部架构无关，
> sm90 可编译可运行；本机实测 **ALL PASS**。目标卡（SM100）专属的只是那几个 fused 指令的快路，
> 会明确标注。

---

## 1. 为什么单开一篇：每个 kernel 都踩的「公共地板」

前面各篇经常出现这样的句子：「epilogue 用 `get_ue8m0_sf_exp` 把 amax 量化成 scale」
「scheduler 用 `cta_exclusive_sum` 数前缀」「sinkhorn 用 `warp_reduce_sum` 逐行/逐列归约」
——这些函数一直当黑盒用。拆开看，它们其实是 DeepGEMM 的**公共地板**，分四类：

| 类别 | 代表函数 | 依赖它的上游 |
| --- | --- | --- |
| **取整 / 算术** | `ceil_div`/`align`/`advance_ptr` | 几乎全部 host/device 代码、`layout/mega_moe.cuh` |
| **快速数学** | `fma2`/`fast_rcp` | Mega MoE / mHC epilogue 的 SwiGLU（第 10/20 篇） |
| **打包 / 量化** | `get_ue8m0_sf_exp`/`scale_bf16x2_into_fp8x4`/`cast_into_bf16_and_pack` | 所有 SM100 epilogue（第 21/31 篇）、Mega MoE L1/L2（第 10/34 篇）、mHC（第 5 篇） |
| **归约 / 扫描** | `warp_reduce_*`/`warp_inclusive_sum`/`cta_exclusive_sum` | mHC sinkhorn（第 5 篇）、Mega MoE（第 3/34 篇）、MQA logits 调度器（第 13/14/20 篇） |

值得注意的是，**取整/量化的「参考定义」在 Python 侧**（`deep_gemm/utils/math.py`，第 27 篇），
device 侧这套函数是它的**逐位等价快路**——本篇的核心实验之一就是把两者逐点对拍。

---

## 2. 文件地图与调用图

```
deep_gemm/common/
├── compile.cuh      # DG_IN_CUDA_COMPILATION / CUTLASS_*_NOINLINE 宏
├── exception.cuh    # DG_STATIC_ASSERT / DG_DEVICE_ASSERT / DG_UNIFIED_ASSERT
├── cute_tie.cuh     # CUTE_TIE(...) 把 cute tuple 解构进若干变量
├── math.cuh         # ★ 算术 / 快速数学 / 打包 / UE8M0 / 归约（236 行，本文主角）
├── utils.cuh        # PatternVisitor / Vectorized / get_num_aligned_tmem_cols
├── types.cuh        # MmaKind / GemmType / KernelType / EpilogueOperatorArgs / 域数
└── packing.cuh      # get_smem_pack_factor（FP4 每字节 2 个逻辑元素）
```

调用图（节选真实调用点）：

- `math::get_ue8m0_sf_exp / get_ue8m0_sf_inv`：
  `sm100_fp8_fp4_mega_moe.cuh:1173-1176`（L1 SwiGLU 输出的 amax→SF）、
  `epilogue/sm100_store_cd.cuh:154-155`、`epilogue/sm100_mega_mhc.cuh:128/251/366-367`。
- `math::get_packed_bf16_amax`：`epilogue/sm100_store_cd.cuh:154`、`epilogue/sm100_mega_mhc.cuh`。
- `math::scale_bf16x2_into_fp8x4`：`epilogue/sm100_store_cd.cuh`、`..._swap_ab.cuh`、`sm100_mega_mhc.cuh`。
- `math::cast_into_bf16_and_pack`：`sm100_{fp8_fp4,bf16}_mega_moe.cuh`、`epilogue/sm100_store_cd.cuh`。
- `math::fma2 / fast_rcp`：`sm100_fp8_fp4_mega_moe.cuh:1127-1131`（SwiGLU）。
- `math::warp_reduce<4,true>`：`sm100_fp8_fp4_mega_moe.cuh:1146-1149`（amax 归约）。
- `math::cta_exclusive_sum`：`scheduler/sm100_mqa_logits.cuh:75`、
  `scheduler/sm100_paged_mqa_logits.cuh:93`、`scheduler/sm100_sparse_mqa_logits_metadata.cuh:138/330`。
- `math::warp_inclusive_sum`：`scheduler/mega_moe.cuh:324`（数有效 M block）。
- `math::advance_ptr`：`layout/mega_moe.cuh:148/245/322/330/340`、kernel 里所有「按字节偏移取址」。
- `utils::get_num_aligned_tmem_cols`：`sm100_fp8_fp4_mega_moe.cuh:232`、`sm100_bf16_gemm.cuh:95`。
- `get_smem_pack_factor`：`sm100_fp8_fp4_gemm_1d1d.cuh:75-76`、`mma/sm100.cuh:92/103`。
- `utils::Vectorized<>`：`impls/smxx_layout.cuh:14/85`（SF 转置的向量化宽度）。

---

## 3. `common/math.cuh` 精读（236 行）

### 3.1 指针与算术（`:12-54`）

- `advance_ptr<dtype_t>(ptr, num_bytes)`（`:13-16`）：**按字节**前进，返回 `dtype_t*`。Mega MoE 的
  缓冲区是一块裸字节、各视图按字节偏移切出来（第 42 篇），所以整个 layout/kernel 都靠它取址。
- `ceil_div`（`:19-22`）/ `constexpr_ceil_div`（`:24-27`）：`(a+b-1)/b`。
- `align<T, kDoCeilAlignment>`（`:29-32`）：`ceil_div(a,b)*b`；`kDoCeilAlignment=false` 时**向下**对齐
  （`(a/b)*b`）——TMA/描述符布线常用。
- `constexpr_align/gcd/min`（`:34-47`）：编译期版本，用于模板参数与 static_assert。
- `swap`（`:49-54`）：`CUTLASS_DEVICE` 版（不含 `<algorithm>`，避免 host/device 混编问题）。

### 3.2 快速数学（`:56-72`）

```cpp
// math.cuh:57-66
CUTLASS_DEVICE float2 fma2(const float2& a, const float2& b, const float2& c) {
#if defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)
    return __ffma2_rn(a, b, c);          // 目标卡：一条 packed FMA2
#else
    return make_float2(__fmaf_rn(a.x,b.x,c.x), __fmaf_rn(a.y,b.y,c.y));  // sm90 回退
#endif
}
```

- `fma2` 只在**目标卡**上有一条 packed 指令；sm90 上退化为两次 `__fmaf_rn`。**两者数学等价**
  （都是单次舍入的 fused FMA），本机实测 131072 个分量**逐位相等**。
- `fast_rcp`（`:68-72`）：`rcp.approx.ftz.f32`。**有误差**——本机实测相对误差 ≤1.19e-7（≈2⁻²³），
  所以只在 `kFastMath` 才走它（`sm100_fp8_fp4_mega_moe.cuh:1127-1131` 的 SwiGLU 分母）。

### 3.3 bf16 / FP8 打包（`:74-109`）

- `cast_into_bf16x2`（`:75-78`）/ `cast_into_bf16_and_pack`（`:80-84`）：把两个 fp32 **RNE** 成 bf16x2
  并按位打包成一个 `int`（低 16 位在前）。`__float22bfloat162_rn` 就是 RNE。
- `get_packed_bf16_amax<kNumPacked>`（`:88-103`）：对**打包的 bf16 对**做 HMNMX2 树归约——
  先 `__habs2` 取绝对值对，再 `__hmax2` 两两合并成 `kNumPacked/2` 棵树，然后逐层 `stride/=2` 归约，
  最后 `__hmax(x,y)`。**要求 kNumPacked 是 2 的幂**（`:90-91` static_assert）。
- `scale_bf16x2_into_fp8x4`（`:106-109`）：

```cpp
// math.cuh:106-109
return __nv_fp8x4_e4m3(__hmul2(lower, sf_inv_lower),
                       __hmul2(upper, sf_inv_upper)).__x;   // 4 个 E4M3 字节打包成 uint32
```

  关键在**幂次 scale 的乘法是精确的**（`sf_inv` 是 2 的幂，见 §3.4），所以「乘 scale → 转 E4M3」
  与「先转后缩放」在数值上不会引入额外误差；本机实测 32768 组输出 vs fp32 参考转 E4M3 **逐位相等**。

### 3.4 UE8M0 SF 指数：整数加法的进位完成「向上取到 2 的幂」（`:111-147`）

这是本篇**最值得讲**的一处。Python 参考（`utils/math.py:13-16`）的 `ceil_to_ue8m0` 定义是：
「把 scale 取到**≥|x| 的最小 2 的幂**」（尾数非 0 就进位到下一个 2 的幂）。device 侧不做除法、
不做浮点取整，而是**直接操作 amax 的位**：

```cpp
// math.cuh:114-135（E4M3 情形）
constexpr uint32_t kMantissaBits     = kIsFP32 ? 23 : 7;
constexpr uint32_t kMantissaMask     = (1u << kMantissaBits) - 1;
constexpr uint32_t kQuantMaxMantissa = (kIsFP8 ? 0x60u : 0x40u) << (kMantissaBits - 7); // 1.75 / 1.5
constexpr uint32_t kQuantMaxExponent = kIsFP8 ? 8 : 2;                                   // 448 / 6
constexpr uint32_t kMinSFExponent    = kIsFP8 ? 105 : 1;                                 // 地板
uint32_t amax_bits = ...;
const auto rounded_exp = (amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits; // ← 进位
return cute::max(rounded_exp, kMinSFExponent + kQuantMaxExponent) - kQuantMaxExponent;
```

**为什么这样能「向上取整」**：设 `amax = m·2^e`（1≤m<2），则
`amax_bits = (e+127)<<23 + mant`。加 `kMantissaMask - kQuantMaxMantissa` 后：
- 若 `mant ≤ 0x600000`（即 m≤1.75），`mant + (2^23-1-0x600000) < 2^23`，**不产生进位**；
- 若 `mant > 0x600000`（m>1.75），**进位**使高位 +1。

于是 `rounded_exp = e+127 + (m>1.75)`；再减去 `kQuantMaxExponent`，得到的就是
`ceil_to_ue8m0(amax/448)` 的**带偏指数**（SF 值 = `2^(结果-127)`）。E2M1 同理，用 `1.5·2²=6`。
这正是源码注释说的「*the carry of the integer addition performs the exponent ceiling*」。

`get_ue8m0_sf_inv<dtype_t>`（`:137-147`）给出倒数：

```cpp
return __uint_as_float((254u - sf_exp) << 23);   // float 版；bf16 版 << 7
```

因为 SF 的指数域是 `sf_exp`（值 `2^(sf_exp-127)`），其倒数指数域就是 `254-sf_exp`
（**互为倒数的带偏指数和为 254**）——所以 `SF * SF_inv == 1.0` **精确成立**（本机实测两 format 均 ALL 1.0）。
这也是 §3.3 里 `sf_inv` 乘法精确的原因。

> **与 Python 参考的关系**：`get_ue8m0_sf_exp<quant_t>(amax)` 等于
> `(ceil_to_ue8m0(clamp(amax, min)/quant_max).view(int) >> 23)`（带地板钳位）。
> 本机用**真实 `deep_gemm.utils.math.ceil_to_ue8m0`** 对 65536 个样本（含 2 的幂、1.75/1.5 边界、
> 0、均匀随机）逐点对拍，**E4M3/E2M1 × FP32/BF16 四个组合 mismatch 全为 0**。

### 3.5 归约与扫描（`:149-232`）

- `warp_inclusive_sum`（`:150-159`）：`__shfl_up_sync` 的 Hillis-Steele 前缀和（lane≥offset 才累加）。
- `cta_exclusive_sum<kNumThreads>`（`:161-184`）：**块内互斥前缀和**。三步——
  ① 每 warp 内 `warp_inclusive_sum`，lane31 把 warp 总和写 `warp_sums`；② `__syncthreads`；
  ③ 对 `warp_sums` 再做一次 `warp_inclusive_sum`，用 `ptx::exchange` 取「本 warp 之前所有 warp 的总和」。
  返回 `lane 的 inclusive - 自身值 + 之前 warp 总和`（即 exclusive），并顺带输出 `total`。
  它是 MQA logits 各调度器把「每个 token/request 的量」变成「每 SM 的起始偏移」的关键工具
  （第 13/14/20 篇），本机实测 256 线程 vs 参考前缀和 **0 不符**。
- `ReduceSum/Max/Min/And/Or` 仿函数（`:186-198`）：注意 `ReduceMax/Min<float>` **特化**成
  `fmaxf`/`fminf`——注释 `:187-189` 指出泛型 float 三元表达式会编成 `FSETP+FSEL`（NaN 语义不同），
  指令数与依赖链翻倍，而 `fmaxf` 是单条 `FMNMX`。
- `warp_reduce<kNumLanesPerGroup, kIntergroupReduce>`（`:200-221`）**两种模式**（本机实测确认语义）：

  | 模式 | 蝶形偏移 | 语义 |
  | --- | --- | --- |
  | `kIntergroupReduce=false`（**intra**） | `2^j`（当 `N ≥ 2^{j+1}`） | 在**连续 N lane 组**内归约 |
  | `kIntergroupReduce=true`（**inter**） | `N·2^j`（≤16） | 在**lane 同余 mod N** 的 lane 间归约（跨组） |

  实测（base 取 `(7i+3)%23-11`）：intra N=4/8/16/32 分别把 warp 切成 8/4/2/1 组、组内求和/取最大；
  inter N=4/8/16 分别得到 4/8/16 组、每组 8/4/2 个**同余** lane——全部与独立 host 模型 **0 不符**。
  `warp_reduce_max<kNumRoutes>`（mHC sinkhorn，`epilogue/sm100_mega_mhc.cuh:66/68/81`）与
  `warp_reduce<4,true>`（Mega MoE amax，`sm100_fp8_fp4_mega_moe.cuh:1146`）就分别用这两种模式。

---

## 4. `common/utils.cuh`（52 行）

- `PatternVisitor`（`:11-22`）：把一个「索引→值」的 lambda 包成可 `[i]` 取值的对象，
  用于给 TMA/描述符构造传「第 i 个元素是什么」（各 impl 大量使用）。
- `Vectorized<kNumBytes>::zeros()`（`:24-40`）：按字节宽度返回全零的向量类型
  （`kNumBytes%16==0`→`uint4`，`%8`→`uint2`，`%4`→`uint`）；`vec_t` 即该类型。
  用于 SF 转置 kernel 的向量化读写（`smxx_layout.cuh:14/85`）。
- `get_num_aligned_tmem_cols<kNumCols>`（`:42-50`）：把 TMEM 列数**向上取整到 {32,64,128,256,512}**
  （static_assert ≤512）。目标卡的 TMEM 是 512 列 / SM（第 25 篇），分配粒度就是这几个档——
  `sm100_fp8_fp4_mega_moe.cuh:232` 用它把「累加器 + SFA + SFB」三块列数合成一个 `kNumTmemCols`，
  这正是第 25 篇「512 列预算与 overlap」的落点。本机实测 `[1,32,33,64,65,128,129,256,257,512]`
  → `[32,32,64,64,128,128,256,256,512,512]` 全对。

## 5. `common/types.cuh`（56 行）

- `MmaKind`（`:7-11`）：`BF16 / MXFP8FP4 / MXF4`——第 20/10 篇的 mma 分流依据。
- `GemmType`（`:13-21`）：`Normal / MGroupedContiguous / MGroupedMasked / KGroupedContiguous /
  Batched / MGroupedContiguousWithPsumLayout / KGroupedContiguousWithPsumLayout`——第 12/13 篇的布局族。
  两个 `constexpr` 判定 `is_m_grouped_contiguous`（`:23-29`）/`is_k_grouped_contiguous`（`:31-37`）
  把「带 PSUM 的变体」也归入对应族（本机实测两类均返回 `true`）。
- `KernelType`（`:39-43`）：`Kernel1D1D / Kernel1D2D / KernelNoSF`——第 22/29/32 篇的选核。
- `EpilogueOperatorArgs`（`:45-52`）：所有 epilogue 算子共享的**无状态**参数基类（第 21/31 篇：
  `sm100_fp8_fp4_gemm_1d1d.cuh:57` 强制 `sizeof` 相等）。
- `kNumDeviceLocalityDomains = 2`（`:54`）：SM 局部域的数量（第 24 篇探针与调度的依据）。

## 6. `common/packing.cuh`（14 行）

`get_smem_pack_factor<dtype_t>`（`:8-12`）：**FP4（`float_e2m1_t`）在 SMEM 里两个逻辑元素挤一个字节**，
返回 2；其它返回 1。第 25/35 篇里 SMEM 描述符/偏移「减半」的根源，也是
`sm100_fp8_fp4_gemm_1d1d.cuh:76` 那条 `A/B SMEM pack factor must match` 断言的由来。

## 7. 三个基础设施头

- `compile.cuh`：`DG_IN_CUDA_COMPILATION`（`:5-7`，nvcc/clang/crt 下定义）——**这就是为什么
  `math.cuh:56` 的 `#ifdef DG_IN_CUDA_COMPILATION` 能把纯 device 函数隔在 host 编译之外**。
  还有 `CUTLASS_{HOST_DEVICE,DEVICE}_NOINLINE`（`:9-17`）。
- `exception.cuh`：`DG_DEVICE_ASSERT`（`:15-23`，printf+trap）、`DG_TRAP_ONLY_DEVICE_ASSERT`（`:25-31`）、
  `DG_STATIC_ASSERT`（`:33-35`）、`DG_UNIFIED_ASSERT`（`:37-43`，按是否在 CUDA 编译期选 device/host）。
  前面各篇引用的行号断言全部来自这里。
- `cute_tie.cuh`：`cute::ignore_t`（`:7-14`）+ `CUTE_TIE`（`:42-50`），把 cute 的 tuple 解构进一批变量
  （`sm90_fp8_mqa_logits.cuh` / `sm90_fp8_paged_mqa_logits.cuh` 用它拆 TMA 坐标）。

---

## 8. H100 实测（`src/device_math_utils.cu` + `harness/exp_device_math_utils.sh`）

在 kernel_lab（H100 80GB HBM3, sm90）上以 `nvcc -arch=sm_90 -O3` **直接 include 真实头文件**
编译运行（四个头 sha256 见 `.out.txt`），Python 侧用真实 `deep_gemm.utils.math.ceil_to_ue8m0`
与 torch 做独立参考。**总判定 ALL PASS**：

| 组 | 内容 | 结果 |
| --- | --- | --- |
| A1/A2 | `get_ue8m0_sf_exp(E4M3/E2M1, fp32)` vs `ceil_to_ue8m0` | mismatch **0/65536** |
| A3/A4 | 同上但 bf16 输入 | mismatch **0/65536** |
| A5 | `SF * SF_inv == 1.0` 精确性 | E4M3/E2M1 均 **True**（exp 范围 E4M3 105–138、E2M1 1–144） |
| B | `get_packed_bf16_amax<8>` vs `abs().amax()` | mismatch **0/32768** |
| C | `cast_into_bf16_and_pack` vs `torch.bfloat16`(RNE) | mismatch **0/65536** |
| D | `scale_bf16x2_into_fp8x4` vs fp32×2^e→E4M3 | mismatch **0/32768** |
| E1 | `fma2`（sm90 回退）vs fp64 单次舍入 | mismatch **0/131072**，max diff 0 |
| E2 | `fast_rcp` 相对误差 | max **1.192e-7**、mean 1.16e-8 |
| F | `cta_exclusive_sum`/`warp_inclusive_sum` vs 参考前缀和 | mismatch **0/256**，total 一致 |
| G | `warp_reduce<intra/inter>` 7 组 vs 独立 host 分组模型 | mismatch **0/32**（组大小/偏移逐条对） |
| H1 | `get_num_aligned_tmem_cols` 10 档 | 全对 |
| H2 | `Vectorized<16/8/4>` 宽度 + `get_smem_pack_factor(E2M1/E4M3)` | `[16,8,4,2,1]` 全对 |
| H3 | `is_m_grouped_contiguous`/`is_k_grouped_contiguous` | `[1,1]` |
| H4 | `ceil_div`/`align`（含向下对齐）/`constexpr_gcd`/`kNumDeviceLocalityDomains` | `[8,1024,896,8,1024,12,17,2]` 全对 |

**两个可复用的结论**：

1. **device 侧 UE8M0 取整与 Python 参考逐位等价**（含 2 的幂与 1.75/1.5 边界），所以各 kernel 测试
   用「参考量化器定义的精确操作数」作 oracle 是自洽的（第 27/43 篇的判据因此成立）。
2. **`warp_reduce` 的 intra/inter 两种模式语义已实测钉死**：intra = 连续 N lane 组内归约，
   inter = lane 同余 mod N 之间归约——读 mHC/Mega MoE 的 sinkhorn/amax 归约时不会再误判分组。

## 9. 边界与「仅分析」

- **SM100 专属**：仅 `fma2` 的 `__ffma2_rn` 快路（`math.cuh:58-59`）；sm90 走回退，二者本机实测逐位相等。
  另 `get_num_aligned_tmem_cols` 的取值本身架构无关，但 TMEM 列预算是目标卡概念（`common/utils.cuh:43`）。
- **本机未能覆盖**：`Vectorized<kNumBytes>` 的 `ulonglong4` 分支（源码 TODO，`utils.cuh:27`）
  当前代码里还没启用，故未涉及。
- 其余函数（算术/打包/归约/扫描）都是 `__device__` 内联、架构无关，sm90 可编译可运行。

## 10. 参考

- DeepGEMM 仓库：`https://github.com/deepseek-ai/DeepGEMM`（本地 commit `057ca59`，Public release 26/09/30）。
- 源码：`deep_gemm/include/deep_gemm/common/{math,utils,types,packing,compile,exception,cute_tie}.cuh`；
  Python 参考量化器 `deep_gemm/utils/math.py:13-23`（第 27 篇）。
- 相关篇目：第 5 篇（mHC sinkhorn 归约）、第 10/20 篇（Mega MoE SwiGLU/打包）、第 13/14/20 篇
  （MQA logits 调度器的 `cta_exclusive_sum`）、第 21/31 篇（epilogue 算子）、第 24 篇（局部域数）、
  第 25 篇（TMEM 预算）、第 27/43 篇（参考量化器与逐位判据）。
