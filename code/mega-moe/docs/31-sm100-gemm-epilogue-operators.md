# SM100 GEMM 的 epilogue：store 路径与四个 epilogue 算子

> 本文对应路线图 **P21-1**。第 27 篇讲了「DeepGEMM 什么时候**不用**自家 kernel（cuBLASLt 回退）」，
> 其中提到「plain-store epilogue 才回退」；但**用自家 kernel 时尾部到底干了什么**一直只被顺带提到
> （第 03/07 篇讲 Mega MoE 自己的 epilogue、第 25 篇讲 SF 布局）。本篇补上 DeepGEMM **所有 SM100
> GEMM 共用的那套 epilogue 机制**：一个可插拔的 `EpilogueClass` → device 侧
> `epilogue::operators` 的**单个算子对象**，随 kernel 一起 launch，并在 `sm100_store_cd` /
> `sm100_store_cd_swap_ab` 里对从 TMEM 读回寄存器的一批累加器值 `apply_values()`。
>
> 四种算子：`Identity`、`ScaleByAlpha`、`StochasticRoundToBF16`（随机舍入 + 四元组哈希 PRNG）、
> `QuantizeToFP8`（BF16 舍入 → per-32 列 amax → UE8M0 SF → E4M3，SF 按 MN-major 打包）。
>
> 本机 H100（sm90）跑不了 SM100 GEMM（`gemm.hpp:119` 对自定义 epilogue 直接断言
> "Custom epilogues require SM100"），且 `StochasticRoundToBF16` 用的 `cvt.rs.bf16x2.f32` 是
> **SM100 专属指令**（sm_90 ptxas 直接报 "Feature '.rs' not supported"）。但算子的核心算法
> （哈希 PRNG、量化/SF 数学、SF 寻址）是架构无关的整数/浮点运算。本文把 `operators.cuh:12-92`
> **逐行原样抽取**到 H100 上编译运行（两个 SM100 硬件原语用明确标注的等价模拟补上），用真实数字
> 验证：ScaleByAlpha 缩放、四元组哈希的均匀性/去相关、随机舍入的**无偏性 vs RNE**、
> `QuantizeToFP8` 的 D/SFD 与 torch 参考**逐位一致**（见 `harness/exp_epilogue_operators.{sh,out.txt}`）。

参考实现：`~/github/DeepGEMM`，公开版 commit `057ca59`（版本 2.8.1）。本机 H100/sm90、CUDA 13.2。

## 1. 为什么 SM100 要「可插拔 epilogue」

### 1.1 SM90 与 SM100 的差异

SM90（Hopper）的 GEMM kernel 尾部是**写死**的：epilogue 只接受一个标量 `alpha`（BLAS 线性组合
的系数）与 `kWithAccumulation` 布尔，输出 dtype 决定走哪条 store 分支
（第 29 篇：SM90 BF16 kernel 的 epilogue `:284-388`，bf16 走 STSM+128B swizzle、fp32 走裸
`st.shared`）。要加一种「量化输出」「随机舍入」就得改 kernel。

SM100（Blackwell）则把这套尾部**参数化成对象**：

- host 侧 `EpilogueClass`（`csrc/jit_kernels/impls/epilogue_class.hpp:19-39`）描述一个 epilogue：
  - `get_epilogue_operator_type()` 返回 device 侧类名字符串（如
    `"epilogue::operators::QuantizeToFP8"`，`:73-75`），JIT 时拼进生成代码；
  - `make_epilogue_operator_args(m, n)` 产出该算子的运行期状态 `EpilogueOperatorArgs`
    （`:25`、`:60-62`、`:77-81`）；
  - `compiler_options()` 允许算子覆盖编译选项；
  - `get_alpha()` 告诉 host「这个 epilogue 是否恰好等价于一个 BLAS 标量 alpha」（`:35`）——
    正是第 27 篇 cuBLASLt 回退三与门里的那一项；
  - `output_sfd()` 返回该 epilogue 要写出的 SFD（`:38`）。
- device 侧 `EpilogueOperatorArgs`（`deep_gemm/include/deep_gemm/common/types.cuh:47-52`）是**唯一**
  的状态载体：

  ```cpp
  struct EpilogueOperatorArgs {
      uint32_t* sfd = nullptr;      // QuantizeToFP8 的 SFD 指针
      uint32_t sfd_stride = 0;
      uint32_t shape_m = 0, shape_n = 0;
      float alpha = 1.0f;
  };
  ```

  每个算子（`operators.cuh`）都继承它，`sm100_fp8_fp4_gemm_1d1d.cuh:57` 用
  `DG_STATIC_ASSERT(sizeof(epilogue_operator_t) == sizeof(EpilogueOperatorArgs), "Epilogue operators
  must not add state ...")` **强制算子不能加状态**：算子只是「行为」，状态全在基类里。

- kernel 把它当参数收（`sm100_fp8_fp4_gemm_1d1d.cuh:41`、
  `sm100_bf16_gemm.cuh:39` 的 `const __grid_constant__ epilogue_operator_t epilogue_operator`），
  在 store 段对每个寄存器批 `apply_values(values)`。

**为什么这样做**：让「量化 / 随机舍入 / alpha」这些尾部变换成为**正交的编译期选择**，一套 UMMA
主循环可以配任意 epilogue，且 host 无需为每种组合写一个 kernel 入口。代价是 epilogue 必须
**无状态**、且 JIT 要对每个 `epilogue_operator_t` 重新实例化模板。

### 1.2 host 如何解析出一个 epilogue

用户可以从三条互斥路径指定 epilogue（`epilogue_class.hpp:108-127` 的 `resolve_epilogue_class`）：

```cpp
DG_HOST_ASSERT((epilogue_class != nullptr) + alpha.has_value() + sfd.has_value() <= 1 ...); // :116-117
if (alpha.has_value())  resolved = make_shared<AlphaEpilogue>(alpha);        // :119-120
if (sfd.has_value())    resolved = make_shared<FP8QuantizationEpilogue>(sfd);// :121-122
if (resolved == nullptr) resolved = make_shared<IdentityEpilogue>();         // :123-124
resolved->check(c, d);                                                        // :125
```

即：

| 用户入口 | 解析成 | device 算子 | 输出约束 |
|---|---|---|---|
| 不给（默认 / `c`、`alpha` 都没有） | `IdentityEpilogue`（:41-48） | `Identity` | bf16 或 fp32（`:30`） |
| `alpha=…` 或 `epilogue.Alpha(a)` | `AlphaEpilogue`（:50-65） | `ScaleByAlpha` | 同 Identity |
| FP8 `(d, sfd)` 输出对 或 `epilogue.FP8Quantization(sfd)` | `FP8QuantizationEpilogue`（:67-89） | `QuantizeToFP8` | **直接 E4M3 输出**、无 `c`（`:84-85`） |
| `epilogue.BF16StochasticRounding()` | `BF16StochasticRoundingEpilogue`（:91-106） | `StochasticRoundToBF16` | **直接 BF16 输出**、无 `c`（`:102-104`） |

Python 侧在 `csrc/apis/epilogue_class.hpp:13-25` 注册为 `deep_gemm.epilogue.{Identity, Alpha,
FP8Quantization, BF16StochasticRounding}`——测试里大量使用（如 `tests/test_bf16.py:50` 对
`bf16_gemm_nt` 挂 `BF16StochasticRounding()`；`tests/test_einsum.py:62/93` 对 einsum 挂）。

**一个关键事实：这套 epilogue 机制是 SM100 专属。** `gemm.hpp` 里每个 API 在 SM90 分支都断言
`epilogue_class == nullptr and "Custom epilogues require SM100"`（`:119/238/308/385/528/611/664/712`）。
einsum 同理（`einsum.hpp:85`、`:246`）。因此第 27 篇看到的「SM90 BF16 默认走 cuBLASLt」与
「SM100 才有自定义 epilogue」是同一枚硬币的两面。

### 1.3 `StochasticRoundToBF16` 需要 arch-specific target

`BF16StochasticRoundingEpilogue::compiler_options()`（`epilogue_class.hpp:97-100`）返回
`{.arch = jit->device.get_arch(false)}`——**`false` 表示非 family 架构（`sm_100a` 而非 `sm_100`）**。
注释写明原因（`:97`）：`cvt.rs` 需要架构专属 target。这正是本文开头实测到的事实：
`cvt.rs.bf16x2.f32` 在 `sm_90` 上 ptxas 直接拒绝，本机不可运行该硬件指令。

## 2. 四个算子（`deep_gemm/include/deep_gemm/epilogue/operators.cuh`）

算子在 `namespace deep_gemm::epilogue::operators` 下，每个都继承 `EpilogueOperatorArgs`
（`Identity: EpilogueOperatorArgs`，`:12`）。**算子之间不组合**（`:9` 注释 "the operators do not
compose with each other"）。

### 2.1 `Identity`（`:12-15`）

```cpp
struct Identity: EpilogueOperatorArgs {
    template <uint32_t kNumValues>
    CUTLASS_DEVICE void apply_values(uint32_t (&)[kNumValues]) const {}
};
```

空操作。默认 epilogue；也是**所有算子的基类**（其余三个都继承 `Identity` 而非直接继承
`EpilogueOperatorArgs`）。

### 2.2 `ScaleByAlpha`（`:20-30`）

只缩放**乘积项**（注释 `:17-19`：支持任意 beta 需要一条单独的 C 载入路径，不属于纯值算子）：

```cpp
template <uint32_t kNumValues>
CUTLASS_DEVICE void apply_values(uint32_t (&values)[kNumValues]) const {
    DG_STATIC_ASSERT(kNumValues % 2 == 0, "Alpha scaling requires float2 alignment");
    const auto values_f32x2 = reinterpret_cast<float2*>(values);
    const auto alpha_f32x2 = make_float2(alpha, alpha);
    for (uint32_t value_idx = 0; value_idx < kNumValues / 2; ++ value_idx)
        values_f32x2[value_idx] = __fmul2_rn(values_f32x2[value_idx], alpha_f32x2);
}
```

把 `uint32_t` 批视为 `float2` 批，用 **`__fmul2_rn`（packed fp32x2 乘，sm_100 指令
`mul.f32x2`）**逐对乘 `alpha`。`__fmul2_rn` 在 CUDA 13.2 只声明在 `crt/sm_100_rt.h`，且声明被
`#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000` 包住——host pass 可见、sm90 device pass 不可见，
这正是本实验需要在 sm90 device pass 补一个「逐分量 `__fmul_rn`」等价实现的原因（头文件文档也写明
"per component same as `__fmul_rn`"）。

### 2.3 `StochasticRoundToBF16`（`:32-65`）——四元组哈希随机舍入

这是四个算子里最有意思的。它把 fp32 累加器**随机舍入**到 bf16，随机位由一个**确定性哈希**给出
（不依赖 RNG 状态，只依赖数据本身与列号）：

```cpp
// :34-39  每个 quartet 成员用不同乘法常数做部分哈希
static uint32_t partial_hash(const uint32_t& value, const uint32_t& quartet_offset) {
    return value * (quartet_offset == 0 ? 0x5671d42bu :
                    quartet_offset == 1 ? 0x9995e499u :
                    quartet_offset == 2 ? 0xace1b8a5u : 0xe153538du);
}
// :41-48  雪崩混合，再按 quartet_offset 选 16 位
static uint32_t select_random_bits(uint32_t h, const uint32_t& quartet_offset) {
    h ^= h >> 23; h *= 0x7feb352du; h ^= h >> 16;
    h *= quartet_offset < 2 ? 0x846ca68bu : 0xd35a2d97u;
    h ^= h >> 11;
    return (h >> (quartet_offset % 2 * 16)) & 0xffffu;
}
// :50-64  四元组（4 个 N 相邻值）共享一个哈希和：局部项求和 + quartet_n_idx
static void cast_quartet(a, b, c, d, quartet_n_idx, packed_ab&, packed_cd&) {
    auto h = partial_hash(a,0) + partial_hash(b,1) + partial_hash(c,2) + partial_hash(d,3) + quartet_n_idx;
    packed_ab = ptx::cvt_rs_bf16x2_f32(f(a), f(b), select_random_bits(h,0) | select_random_bits(h,1)<<16);
    packed_cd = ptx::cvt_rs_bf16x2_f32(f(c), f(d), select_random_bits(h,2) | select_random_bits(h,3)<<16);
}
```

**为什么是「四元组」**：`cvt.rs.bf16x2.f32` 一次转换两个 fp32，需要每个元素一个 16 位随机数。
四个 N 相邻值的哈希**共享同一个求和 `h`**、再各自取 `select_random_bits(h, offset)` 的不同 16 位，
好处是：① 一条 32 位哈希即可给 4 个元素产生随机（省寄存器/运算）；② `h` 混入了 `quartet_n_idx`
（列号），让**不同列**的随机不同；③ `h` 混入了四个值的部分哈希，让**不同数据**的随机不同。
这是「用数据本身的哈希当随机源」——不需要外部 RNG、可复现、且能保证同一位置的舍入决策稳定。

**为什么需要随机舍入**：把 fp32 累加器降到 bf16 时，RNE（就近取偶）对「恰好在中点/总偏向一侧」
的分布会引入**系统性偏差**（本文实测在精确中点处偏差 $-3.9\times10^{-3}$）。随机舍入以
「向上概率 = 小数部分 / ulp」的方式舍入，**期望无偏**（实测偏差 $\sim10^{-5}$，纯统计噪声），
在低精度训练里能避免误差累积。`cvt.rs` 就是硬件版的无偏随机舍入。

### 2.4 `QuantizeToFP8`（`:69-92`）——带 SF 的量化输出

```cpp
struct QuantizeToFP8: Identity {
    static constexpr uint32_t kSFGranN = 32;                        // :70  每 32 列一个 SF
    template <typename sf_t>
    void store_sf(row_idx, group_n_idx, batch_idx, sf) const {      // :78-91
        if (row_idx >= shape_m or group_n_idx >= shape_n) return;   // :82
        const auto sf_idx = (batch_idx * shape_n + group_n_idx) / kSFGranN;  // :84
        const auto sf_word_ptr = sfd + (sf_idx / 4) * sfd_stride + row_idx;  // :85
        if constexpr (is uint32_t) *sf_word_ptr = sf;               // :87  整字写
        else reinterpret_cast<uint8_t*>(sf_word_ptr)[sf_idx % 4] = sf; // :89  单字节写
    }
};
```

`QuantizeToFP8` 本身只是个**寻址器**：真正的量化数学在 `math.cuh` 与 store 路径里（见 §3）。
`store_sf` 的关键是 **MN-major 打包**：`sf_idx = (batch*n + col)/32` 是「SF 组列号」，
每 4 个 SF 组合成一个 int32 字，**字内 4 字节 = 列方向相邻的 4 个 bin**（`:89` 的 `sf_idx%4`），
而字之间沿 `row_idx` 连续（`:85` 的 `+ row_idx`，stride=1）、不同字列间隔 `sfd_stride`。
这与第 25 篇讲的 SFA 的「MN-major + TMA 对齐 + 4 字节 LSB-first 打包」完全一致——FP8 输出的 SFD
复用同一套布局。

> 备注：`store_sf` 的 `batch_idx * shape_n + group_n_idx`（`:84`）说明了注释 `:73-74` 的口径——
> **batched GEMM 把所有 batch 的 SF 列沿 `(batch, n)` 拍平**，相当于把 D 视作
> `(m, batch*n)` 的二维张量做 per-token 量化（einsum 的 FP8 输出对正是这个契约，见 §4.3）。

## 3. store 路径：`apply_values` 在哪里被调用

算子的 `apply_values` 在两条 store 实现里被调用：

- `sm100_store_cd.cuh`（`:124`）：普通布局（store shape 由 `STORE_BLOCK_M/N` 决定）；
- `sm100_store_cd_swap_ab.cuh`（`:100/112`）：A/B 互换布局（`STORE_BLOCK_N == 128` 一个 warpgroup
  读满 128 行 TMEM，`:35-37`）。

两者都由 `sm100_fp8_fp4_gemm_1d1d.cuh:530/546` 与 `sm100_bf16_gemm.cuh:385/400` 按是否 swap-AB 分派。
共同的骨架是（以 `sm100_store_cd.cuh` 为例）：

```cpp
// :34-50  编译期判别 + 静态断言
constexpr bool kWithOutputSF = is_same_v<epilogue_operator_t, QuantizeToFP8>;            // :35
constexpr bool kWithStochasticRounding = is_same_v<epilogue_operator_t, StochasticRoundToBF16>; // :36-37
constexpr uint32_t kSFGranN = QuantizeToFP8::kSFGranN;                                   // :38
DG_STATIC_ASSERT(not kWithOutputSF or is E4M3 D, ...);                                   // :46
DG_STATIC_ASSERT(not kWithOutputSF or STORE_BLOCK_N % kSFGranN == 0, ...);               // :47
DG_STATIC_ASSERT(not kWithStochasticRounding or (is bf16 and not accum), ...);           // :48-50
```

### 3.1 四条 store 分支（`:144-193`）

对每个从 TMEM 读回的寄存器批（`tmem_load_t` 依 `kNumElemsPerLoad` 选 `.32x32b4x/8x/32x`，`:116-119`），
先 `epilogue_operator.apply_values(values)`（`:124`），空组写 0（`:126-127`），再按 epilogue 分四路：

1. **`kWithOutputSF`（`:144-174`）**：先 `cast_into_bf16x2` 把 fp32 对**舍入成 bf16 对**（`:147-150`），
   注释 `:145-146` 点明动机——**先 bf16 再量化，使输出与「BF16 D 走独立 per-token cast kernel」
   逐位一致**（2 的幂缩放对 bf16 精确）。然后：
   - `get_packed_bf16_amax` 树形归约出组内 bf16 amax（`math.cuh:89-103`，用 `__habs2`/`__hmax2`）；
   - `get_ue8m0_sf_exp<E4M3>(amax)` 出 UE8M0 指数（`math.cuh:115-135`，见下）；
   - `get_ue8m0_sf_inv<bf16>(sf_exp)` 出 2 的幂倒数（`math.cuh:140-147`，`(254-exp)<<7` → 其值为
     $2^{127-\text{exp}}$）；
   - `scale_bf16x2_into_fp8x4` 缩放并打包 4 个 E4M3（`math.cuh:106-109`，
     `__nv_fp8x4_e4m3(__hmul2(lower, sf_inv), __hmul2(upper, sf_inv))`）；
   - SF 字节按 `store_whole_sf_word`（`:102`，`kNumLoads==4 && shape_n%(4*32)==0` 时）攒成整字
     `sf_word |= sf_exp << (load_idx*8)`（`:172`）最后 `store_sf`（`:197`），否则逐 bin 写（`:174`）。
2. **`float` 输出（`:175-176`）**：裸 `st_shared` 4 个 fp32。
3. **`kWithStochasticRounding`（`:177-184`）**：注意注释 `:178`——`sm100_store_cd` 里**每批 8 个值
   N 连续 = 恰好两个四元组**，于是 `quartet_n_idx = (n_idx + load_idx*8)/4`（`:180`），两次
   `cast_quartet`（`:182-183`）后 STSM。
4. **普通 bf16（`:185-193`）**：`cast_into_bf16_and_pack`（RNE）后 STSM。

`math::get_ue8m0_sf_exp`（`math.cuh:115-135`）值得单独说：它用「整数加法的进位」实现**指数上取整**——
`(amax_bits + kMantissaMask - kQuantMaxMantissa) >> kMantissaBits`（`:133`），进位当且仅当 amax 的
尾数超过最大有限值的尾数（E4M3 为 `0x60`=`1.75`，`:123`）。随后 `max(rounded_exp, kMinSFExponent +
kQuantMaxExponent) - kQuantMaxExponent`（`:134`）做**指数地板**，对应「per-token cast kernel 的 amax
clamp」（`:125-127`：E4M3 用 `max(amax, 1e-4)`=2⁻²²，E2M1 用 `max(amax, 6·2⁻¹²⁶)`）。

### 3.2 swap-AB 变体的随机舍入：跨 lane 归约哈希

`sm100_store_cd_swap_ab.cuh` 的 `kWithStochasticRounding`（`:192-221`）比普通版复杂得多，因为
swap-AB 的 `.16x256b` 载入让每个 lane 拿到的是**M 相邻的两行 × 一个 N 值**（`:192-199` 注释），
四元组不再在单个 lane 内。于是：

```cpp
const auto quartet_offset = lane_idx / 4 % 4;                       // :201
for (value_idx...) hashes[value_idx] = partial_hash(values[value_idx], quartet_offset);  // :203-205
for (xor_mask = 4; xor_mask <= 8; xor_mask <<= 1)                    // :207-211
    for (value_idx...) hashes[value_idx] += __shfl_xor_sync(0xffffffffu, hashes[value_idx], xor_mask);
for (value_idx += 2) wrapped = cvt_rs(...,                                             // :213-220
    select_random_bits(hashes[value_idx]     + quartet_n_idx, quartet_offset) |
    select_random_bits(hashes[value_idx + 1] + quartet_n_idx, quartet_offset) << 16);
```

即「**每个 lane 各算自己那一项的 `partial_hash`，再用 `shfl_xor`（mask 4、8）把四元组的
4 项在 lane 间加总**，最后各自取自己 offset 的 16 位」。注释 `:198-199` 说明这是为了「保持 8 条
哈希链重叠」。这体现了 `partial_hash`/`select_random_bits` 被拆成**可分布**的两半的用意。

### 3.3 TMA store 与 SF 的写回时机

store 到 smem 后，统一 `tma_store_fence` + `NamedBarrier`（`:200-202`），由 warp0 的 elect-one 发
`SM90_TMA_STORE_2D/3D`（batched / k-grouped contiguous 走 3D，accumulate 走
`SM90_TMA_REDUCE_ADD_*`，`:204-212`）。另一处细节：读完所有被重叠的 TMEM 列后，在**任何 cast/store
之前**就 `tmem_overlap_barrier->arrive`（`:129-136`），让 MMA warp 尽早复用 TMEM——这与第 03 篇
Mega MoE epilogue 的「提前 release TMEM」是同一思路。

## 4. 接线：谁在用这套 epilogue

### 4.1 SM100 BF16 / FP8-FP4 稠密 GEMM

`bf16_gemm_nt`（`gemm.hpp:488-535`）与 `fp8_gemm_nt` 族在 arch10 分支把
`resolved_epilogue_class` 传给 `sm100_bf16_gemm` / `sm100_fp8_fp4_gemm_*`；SM90 分支保留旧签名
（只有标量 `alpha`）。`sm100_fp8_fp4_gemm_1d1d.cuh:62-66` 还额外约束：
**FP8 输出（`kWithOutputSF`）只支持 `GemmType::Batched`、不累积、E4M3 D**。

### 4.2 einsum（按头 batched）

`einsum('bhr,hdr->bhd')` / `('bhd,hdr->bhr')` 在 arch10 走 `sm100_bf16_*` 并接
`resolved_epilogue_class`（`einsum.hpp:76/88`、`:110`）；arch9 则断言无自定义 epilogue。

### 4.3 FP8 输出对 = `FP8Quantization` 的另一种写法

`fp8_bmm`（`einsum.hpp:183-249`）说明 FP8 `(d, sfd)` 输出对与用户传
`epilogue.FP8Quantization(sfd)` 是**同一件事**（`:190` 注释）：

```cpp
if (d_fp8 != nullptr) { merged = make_shared<FP8QuantizationEpilogue>(d_fp8->second); }  // :197-199
...
if (sfd.has_value()) {
    DG_HOST_ASSERT(arch_major == 10);                       // :213  SM100 专属
    DG_HOST_ASSERT(d_tensor.stride(0) == n and n % 128 == 0); // :214-215  batch 沿列连续、不跨组
    check_sf_layout(sfd, m, batch_size * n, 1, 32, ...);    // :216  与 per-token cast 同布局
}
...
sm100_fp8_bmm(..., resolve_epilogue_class(merged, c, d_tensor)); // :240-242
else { DG_HOST_ASSERT(merged == nullptr and "Custom epilogues require SM100"); ... } // :245-247
```

这正好补齐了第 21 篇（FP8 einsum）里「SM100 独占 FP8 输出对」那句话的落点：它落到
`QuantizeToFP8` 这套 store 路径上。

## 5. H100 实测（真实设备代码）

实验见 `harness/exp_epilogue_operators.sh` + `src/epilogue_operators.cu`，输出
`harness/exp_epilogue_operators.out.txt`。做法：harness 把 `operators.cuh:12-92` 逐行原样抽取
（抽取片段 sha256 `d00b6b6f…`）编译，两个 SM100 硬件原语用**明确标注的等价模拟**补上
（`__fmul2_rn` = 逐分量 `__fmul_rn`；`cvt.rs` = 无偏随机舍入软件等价）。此外先探测并打印
`cvt.rs` 在 sm_90 的编译失败，坐实其 SM100 专属。

实测结果（H100 80GB HBM3，sm90，CUDA 13.2）：

- **硬件原语探测**：`cvt.rs.bf16x2.f32` 在 sm_90 上 `ptxas: Feature '.rs' not supported on
  .target 'sm_90'`——`StochasticRoundToBF16` 的硬件路径确实是 SM100 专属。

- **A) ScaleByAlpha**：`alpha=0.5`，1024 个随机值，与逐分量 `__fmul_rn` **逐位不一致 0 个**、
  最大相对误差 0。坐实「`__fmul2_rn` = 逐分量 RNE 乘」且只缩放乘积项。

- **B) 四元组哈希统计**（$n=2^{20}$）：每 bit 的 1 比例偏离 0.5 最大 **0.00118**；
  低 16 位 64 桶**卡方 86.1**（df=63，期望 ≈63）；归一化均值 **0.49975**；
  offset 两两相等计数 `14/15/20/11/12/14`（≈0）——哈希**近似均匀**且**四个 offset 相互独立**。
  （卡方 86 对 63 略高是「用数据哈希当随机源」的预期偏差，不影响无偏性。）

- **C) 随机舍入无偏性 vs RNE**（每类 20 万次，随机位均匀）：

  | 输入（1.0 附近） | SR 均值偏差 | SR 向上比例 | RNE 均值偏差 |
  |---|---|---|---|
  | 精确中点（低 16 位 0x8000） | $+1.25\times10^{-6}$ | 0.5002 | $-3.91\times10^{-3}$ |
  | 1/8 处（0x2000） | $+1.10\times10^{-5}$ | 0.1264 | $-9.77\times10^{-4}$ |
  | 3/8 处（0x6000） | $+1.62\times10^{-5}$ | 0.3771 | $-2.93\times10^{-3}$ |

  SR 偏差是统计噪声级（$\sim10^{-5}$，且向上比例精确等于「小数部分/ulp」0.5/0.125/0.375），
  **RNE 则在每个点固定偏向一侧**（偏差 $\sim10^{-3}$）。这就是低精度训练要随机舍入的原因。

- **D) QuantizeToFP8 的 D/SFD**（$M=64,N=128$，含一行 ×100 极端行、一组全零）：与
  torch 参考（bf16 round → per-32 amax → UE8M0 exp → 2 的幂倒数 → E4M3）对拍：
  **D 逐字节不一致 0/8192**、**SFD 逐字节不一致 0/256**；极端行 SF exp=128、全零组 SF exp=105
  （=指数地板 $105$）。**ALL PASS**——真实 device 管线（`get_packed_bf16_amax` /
  `get_ue8m0_sf_exp` / `get_ue8m0_sf_inv` / `scale_bf16x2_into_fp8x4` / `store_sf`）数学正确、
  MN-major 寻址正确。

## 6. 与其它篇的关系与结论

- **第 03/07 篇**（Mega MoE 自带的 epilogue）：Mega MoE 的 L1 in-place SwiGLU + UE8M0 量化 +
  L2 直写是**它自己**写的融合 epilogue，不走本篇这套通用机制（因为它要 SwiGLU/combine 语义）。
  但两者共享同一批 `math.cuh` 原语（`get_ue8m0_sf_exp/inv`、bf16 round）。
- **第 25 篇**（SF 布局）：本篇 `QuantizeToFP8`/`store_sf` 生成的正是那套 MN-major + 4 字节
  打包的 SFD，可直接作为下一次 GEMM 的 SFA。
- **第 27 篇**（cuBLASLt 回退）：`get_alpha()` 的取值（Identity→1.0、Alpha→alpha、
  FP8Quant/Stochastic→nullopt）决定了第 27 篇所述回退三与门；本篇给出了「不回退时尾部到底干什么」。
- **第 29 篇**（SM90 BF16 kernel）：SM90 的 epilogue 是写死的两条 store 分支；SM100 把它参数化成
  四个算子。两篇对照即「写死的 epilogue」vs「可插拔的 epilogue」。

**结论**：SM100 GEMM 的 epilogue 是一套「host `EpilogueClass` 选择 + device 无状态算子 + 共享
store 路径」的机制，四种算子（Identity / ScaleByAlpha / StochasticRoundToBF16 / QuantizeToFP8）
覆盖了「裸输出 / alpha 缩放 / 低精度训练随机舍入 / 带 SF 的 FP8 量化输出」四类尾部需求。
其中随机舍入用「数据四元组哈希 + 硬件 `cvt.rs`」实现**无偏**舍入，FP8 量化用「先 bf16 再 per-32
UE8M0」实现与独立 cast kernel 逐位一致的输出。两者在 H100 上都能用架构无关的部分验证到
逐位/统计正确。
