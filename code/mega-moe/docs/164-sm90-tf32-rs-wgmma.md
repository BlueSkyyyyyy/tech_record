# SM90 TF32 WGMMA 的 RS（寄存器 A）路径：补齐 P39 的未做项，并定位 tf32 舍入的真正来源

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P154-1 增量（第 164 篇）**。
> 第 49 篇（P39-1）把 SM90 WGMMA 的**共享内存描述符、选择器、异步组**拆透，并在 H100 上跑了
> BF16/FP8 的 **SS 路径**（A、B 都在 smem）真实数值；但它在 §10 明确列了两条未做，其中一条是
> **`TF32MMASelector` 的 RS 路径数值**（A 在寄存器、B 在 smem）。本篇补上这一格。
>
> 补这一格的顺手收获，是一个会咬人的反直觉事实：**`wgmma` 的 tf32 指令对裸 fp32 操作数是「截断」
> 的，`tf32_hc_prenorm_gemm` 里观测到的「就近舍入（RNE）」其实来自 TMA 描述符的
> `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32`，而不是 MMA 指令本身。** 这修正了第 16 篇对 tf32 语义的归因。
>
> **SM90 设备代码，H100（sm90a）真机可跑**。源码坐标：`~/github/DeepGEMM` commit `057ca59`。
> 实验：`src/tf32_rs_wgmma.cu` + `harness/exp_tf32_rs_wgmma.{sh,out.txt}`。

---

## 1. RS 路径为什么存在、谁在用

WGMMA 的 A 操作数有两种来源：**SS**（A、B 都在共享内存，用 64 位描述符寻址，第 49 篇已拆）与
**RS**（A 在**寄存器**、B 在 smem）。DeepGEMM 里 FP8/BF16 的选择器只走 SS，**只有 TF32 走 RS**——
`mma/sm90.cuh:168-190` 的 `TF32MMASelector` 在 `kUseRS=false` 时直接
`DG_STATIC_ASSERT(kUseRS, "SS mode is not supported for TF32MMASelector for now")`（`:187`）。

TF32 RS 在 DeepGEMM 里**只有一个消费者**：`impls/sm90_tf32_hc_prenorm_gemm.cuh:157`
`using WGMMA = typename mma::sm90::TF32MMASelector<WGMMA_N, true>::type;`（`true` 即启用 RS）——
它是第 16/79/97 篇讲过的 **HyperConnection prenorm GEMM**：残差多流混合的系数投影，需要高于 bf16
的尾数精度，于是走 TF32。为什么必须 RS？因为这条 GEMM 有一段**融合**——A（残差，bf16）在搬进寄存器的
同时顺路把平方和算出来（`:196-202`），这是「投影 + RMSNorm 平方和」一拍的支点（第 16 篇 §5）。
A 既然已经进了寄存器，就用 RS 免去再写回 smem。

## 2. `TF32MMARS` / `TF32MMASelector`（`mma/sm90.cuh:151-190`）

`TF32MMARS<N_, MMA>`（`:151-166`）：

- `wgmma(float* a, uint64_t const& desc_b, float* d, bool scale_d)`（`:158`）——注意签名：**A 是
  `float*`（寄存器数组）、B 是 64 位描述符**。`call_fma_impl`（`:153-156`）把 `a[0..3]` 四个
  `uint32_t`（= 4 个 tf32 值）连同 `desc_b` 和一个 `float* d` 累加器数组一起喂给 cute 的
  `MMA::fma`。
- 静态属性：`M=64`、`K=8`、`kNumAccum = M*N/128 = N/2`（`:162-165`）。**K=8** 是 tf32 WGMMA 的
  收缩步长（BF16 是 16、FP8 是 32，见第 49 篇），因为 tf32 元素占 4 字节。

`TF32MMASelector<N, kUseRS>`（`:169-190`）把 N 映射到具体的
`cute::SM90::GMMA::MMA_64xNx8_F32TF32TF32_RS_TN`（N ∈ {8,16,32,64,128,256}，`:173-178`）。

## 3. A 的寄存器布局：`ALayout_64x8`

RS 的 A 分片是 128 个线程 × 4 个 tf32 寄存器，映射到 64×8 的 (m,k)。这个映射在 cute 里叫
`ALayout_64x8`（`cutlass/include/cute/atom/mma_traits_sm90_gmma.hpp:446-448`）：

```
ALayout_64x8 = Layout<Shape <Shape <  _4,_8, _4>, Shape <    _2,  _2>>,
                      Stride<Stride< _64,_1,_16>, Stride<    _8,_256>>>
```

按 cute 的坐标约定（线程坐标 `(t0,t1,t2)` 展平为 `lane = t0 + 4·t1 + 32·t2`，寄存器坐标
`(v0,v1)` 展平为 `reg = v0 + 2·v1`），它等价于：

- `k = (lane % 4) + 4·(reg / 2)`
- `m = ((lane / 4) % 8) + 16·(lane / 32) + 8·(reg % 2)`

而 `sm90_tf32_hc_prenorm_gemm.cuh:180-200` 的手写取址恰好也是这个：`row = warp*16 + lane/4`、
`elem_offset = lane%4`，寄存器 `a[0..3]`（经 `__bfloat1622float2` 转 fp32）分别放
`(row, eo)`、`(row+8, eo)`、`(row, eo+4)`、`(row+8, eo+4)`。本篇实验把设备端
`alayout_64x8` dump 出来，与**两条独立路径**逐项对比：cute 公式 512 项 **0 不符**、hc_prenorm
手写取址 512 项 **0 不符**、对 64×8 双射 **512/512**（§7 A/B）。

## 4. 真实 WGMMA RS 数值：整数精确、分数**截断**

`src/tf32_rs_wgmma.cu` 的 `tf32_rs_gemm` 在 H100 上跑真实 `wgmma...f32.tf32.tf32`（RS）：A 从
全局按 §3 的映射装进寄存器、B 用 128B swizzle 铺进 smem（BLOCK_K=32 float 恰好 = 128B，满足
K-major 描述符 `kSwizzleMode == BLOCK_K*sizeof(dtype)` 的约束，`mma/sm90.cuh:251`）。四组输入：

| case | A | B | D vs fp64（max abs） | 结论 |
|---|---|---|---|---|
| `n64_int` / `n128_int` | 小整数 | 小整数 | **0.000e+00** | tf32 精确表示整数，逐位精确 |
| `n64_a_frac` | 分数 | 整数 | full 2.8e-2 / **trunc 9.5e-7** / rne 4.4e-2 | 隔离 **A（寄存器）** 的 tf32 舍入 |
| `n64_b_frac` | 整数 | 分数 | full 3.8e-2 / **trunc 1.9e-6** / rne 6.2e-2 | 隔离 **B（smem）** 的 tf32 舍入 |
| `n64_both` / `n128_both` | 分数 | 分数 | full 9.1e-2 / **trunc 3.0e-5** / rne 9.8e-2 | 两者都截断 |

即：无论 A 在寄存器、还是 B 走 smem 描述符，**WGMMA 对裸 fp32 都是「截断到 tf32」（丢掉低 13 位
尾数）**，与 PTX 对 `.tf32` 的定义（只取高 19 位）一致。为排除累加干扰，再做「单元素探针」（A 是
delta、其余 0，B[n,0] 取 8 个 trunc 与 rne 不同的边界值）：**8/8 都是截断**（§7 C）。

## 5. 反直觉发现：RNE 来自 **TMA 描述符**，不是 WGMMA

第 16 篇（P7-2）真跑 `tf32_hc_prenorm_gemm` 时，发现 D 与「RNE-to-tf32 参考」更近
（`trunc=4.982e-04 vs rne=1.665e-05`，比值 0.03），据此写「**硬件 tf32 对 fp32 操作数是 RNE**」。
但 §4 的裸 WGMMA 明明是截断。矛盾在哪？

在 **TMA 描述符的数据类型**。`csrc/jit_kernels/impls/runtime_utils.hpp:85-90`：

```cpp
static CUtensorMapDataType aten_dtype_to_tensor_map_dtype(dtype, allow_tf32, fp4_unpacked_smem) {
    if (allow_tf32 and dtype == torch::kFloat)
        return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    ...
}
```

`sm90_tf32_hc_prenorm_gemm.hpp:34-43` 给 A、B 描述符都传了 `allow_tf32=true`。B 是 fp32，于是
它的 tensor map dtype 变成 **`CU_TENSOR_MAP_DATA_TYPE_TFLOAT32`**；A 是 bf16，这条分支不触发
（`allow_tf32 and dtype==kFloat` 为假）。也就是说：**B 的 fp32→tf32 舍入发生在 TMA 搬入 smem 时**
（`cp.async.bulk.tensor` 用 TFLOAT32 描述符），等 WGMMA 读它时已是 tf32，WGMMA 那步的截断不再起作用。

本篇直接验证：把同一批 fp32 边界值用 no-swizzle 的 2D TMA 分别按 `FLOAT32` / `TFLOAT32` 载入 smem
（`src/tf32_rs_wgmma.cu` 的 `tma_probe` + 内联 `cp.async.bulk.tensor.2d` + mbarrier）：

| tensor map dtype | smem == 原 fp32 | == 截断 | == RNE |
|---|---|---|---|
| `CU_TENSOR_MAP_DATA_TYPE_FLOAT32` | **8/8** | 0/8 | 0/8 |
| `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` | 0/8 | 2/8 | **8/8** |

**TMA 用 TFLOAT32 描述符载入 fp32 时，硬件做了 fp32→tf32 的就近舍入（RNE）。** 再用 DeepGEMM
端到端 delta 探针（`tf32_hc_prenorm_gemm`，A 是 bf16 delta、B[n,0] 是边界值）验证：D == RNE
**8/8**——三者闭环。

因此准确的机制是：

```
B(fp32) --TMA(TFLOAT32 描述符, RNE 舍入)--> smem(tf32) --wgmma(截断, 已在 tf32 精度)--> D
A(bf16) --TMA(BFLOAT16 描述符, 无损)--> smem --__bfloat1622float2--> 寄存器 --wgmma(截断, bf16 无损)--> D
```

A 侧因为源是 bf16（8 位尾数 < tf32 的 10 位），无论截断还是 RNE 都无损；**只有 B 侧（fp32）会真正
经过一次 tf32 舍入，而这次舍入的「就近」属性是 TMA 给的。** 第 16 篇的观测没错，归因需要精确到
「TMA 描述符 dtype」这一层。这也解释了为什么 CUTLASS 建议 tf32 GEMM 走 tf32 tensor map：硬件替你把
fp32 就地舍到 tf32，kernel 侧不必再写一遍 `cvt`。

## 6. 与周边篇目对接

| 篇 | 关系 |
|---|---|
| 第 49 篇（P39-1，`docs/49`） | 拆了 SM90 WGMMA 描述符/选择器/异步组 + BF16/FP8 SS 数值；§10 点名「TF32 RS 未做」，本篇补上 |
| 第 56 篇（P46-1，`docs/56`） | 拆了 `runtime_utils.hpp` 的 `make_tma_*` 描述符构造层；本篇补上其中 `allow_tf32 → CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` 这条**会改变数值**的分支的实测后果 |
| 第 16 篇（P7-2，`docs/16`） | 拆了 `tf32_hc_prenorm_gemm`（有 SM90 实现）；本篇修正它对 tf32 就近舍入的归因（真凶是 TMA dtype） |
| 第 79/97 篇（`docs/79`/`docs/97`） | HC prenorm GEMM 的验收测试与 JIT 包装层；CMake/契约侧一致 |

## 7. H100 真机实测汇总

实验 `harness/exp_tf32_rs_wgmma.{sh,out.txt}`（`kernel_lab`，H100/sm90a），真实头文件 sha256 见
`.out.txt`：

**A) 选择器静态属性**——`TF32MMASelector<8/64/128/256>: M=64, K=8, kNumAccum=4/32/64/128`，全对。

**B) A 寄存器布局**——512 项 dump：与 cute `ALayout_64x8` 公式 **mismatch=0**、与
`sm90_tf32_hc_prenorm_gemm.cuh:180-200` 手写取址 **mismatch=0**、64×8 双射 **512/512**。

**C) 真实 WGMMA RS 数值**——整数输入逐位精确（`n64_int`/`n128_int` maxdiff 全 0）；分数输入下
`trunc` 参考与 D 的差 ~1e-6（仅 fp32 累加噪声）、`rne` 参考差 ~1e-1，**单元素探针 8/8 == 截断**。

**D) TMA tensor map 探针**——`FLOAT32`：smem 与原字节 **8/8 逐位相同**；`TFLOAT32`：smem ==
RNE-to-tf32 **8/8**。

**E) DeepGEMM 端到端 delta 探针**——`tf32_hc_prenorm_gemm` 的 B 端到端 == RNE-to-tf32 **8/8**。

**编译期负样本**——`TF32MMASelector<64,false>`（SS 形态）被 `DG_STATIC_ASSERT` 拒绝：
`"SS mode is not supported for TF32MMASelector for now"`。总判定 **ALL PASS**。

## 8. 结论与边界

- **补上的一格**：SM90 TF32 WGMMA 的 RS 路径（A 在寄存器、B 在 smem）在 H100 上真机可跑、
  与 fp64 参考逐位/近逐位一致；A 的寄存器布局 `ALayout_64x8` 与 hc_prenorm kernel 的手写取址、
  cute 公式三方互证。
- **反直觉事实**：`wgmma...f32.tf32.tf32` 对裸 fp32 操作数是**截断**；`tf32_hc_prenorm_gemm` 里
  观测到的 **RNE 来自 TMA 描述符 `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32`**（`runtime_utils.hpp:88-89`，
  由 `allow_tf32=true` 触发）。要用 tf32 GEMM 且想拿到「就近舍入」，就得让 TMA 用 tf32 dtype；
  若直接把 fp32 塞进寄存器/smem 喂 WGMMA，得到的是截断。
- **边界**：本实验用的是真实的 `TF32MMARS`/`TF32MMASelector` 与真实 PTX `wgmma`，架构无关地
  复现了 tf32 数值路径；`sm100_tf32_hc_prenorm_gemm`（UMMA TS 形态）仍 SM100 专属、仅分析。
  TMA 的 tf32 舍入行为是**驱动/硬件**行为，本机为 H100 + 驱动 570 上测得。

（第 49 篇拆了 SM90 WGMMA 底座、第 56 篇拆了 TMA 描述符构造、第 16 篇拆了 HC prenorm GEMM；
本篇把「TF32 RS 路径 + tf32 舍入的真正来源」这一直缺席的一格补上。）
