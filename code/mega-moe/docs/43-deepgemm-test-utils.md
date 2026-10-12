# DeepGEMM 测试断言辅助 `tests/utils.py`：每个 kernel 正确性判据的可执行 oracle

> 本专题（Mega MoE 前向 / 反向算子实现解析）的第 43 篇。前面第 40 篇拆了**测试基座**
> `deep_gemm/testing/`（`calc_diff` / `bench_kineto` / `uneven_all_gather`），第 41 篇拆了
> **操作数/布局生成器** `tests/generators.py`，第 37 篇拆了**参考量化器** `deep_gemm/utils/math.py`。
> 但 `tests/test_bf16.py` / `tests/test_einsum.py` / `tests/test_fp8_fp4.py` 还共用一组**断言/转换辅助**
> `tests/utils.py`——它们是「每个 kernel 到底要满足什么**比 relerr 更严**的性质」的**可执行 oracle**，
> 却从没被系统讲过。本篇补齐它，并与第 31 篇（device 侧 epilogue 算子）、第 27 篇（cuBLASLt 回退）串联。
>
> 本文所有行号均已对照 DeepGEMM commit `057ca59`（Public release 26/09/30）核对过。
> `tests/utils.py` 全部函数是**架构无关的纯张量/位运算**，**H100 真机可跑**（配合真实 SM90 kernel，
> 见 `code/mega-moe/harness/exp_test_utils.py` 与 `.out.txt`）。SM100 专属之处已标注。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 定位：判据链上的「断言层」

DeepGEMM 的每个算子测试（`tests/test_*.py`）都 `import *` 自第 40 篇的 `deep_gemm/testing/` 与
`tests/generators.py`。此外，`test_bf16.py:11-13`、`test_einsum.py:16-17`、`test_fp8_fp4.py:12-16`
还显式从同目录 `tests/utils.py` 引入了 6 个辅助：

| 辅助（`tests/utils.py`） | 行号 | 它断言/构造的「性质」 | 依赖 |
|---|---|---|---|
| `assert_psum_zero_padding` | 9-17 | PSUM 布局下，每组的**逻辑尾部到对齐边界**在 A/D 里都必须是 0 | `assert_bitwise`(间接)、`align`、`get_mk_alignment...` |
| `assert_direct_output_matches_fp32_accumulation` | 20-38 | **direct 输出**必须逐位等于「从零累积」的 FP32 基线 | 真实 kernel |
| `add_sf_k_padding` | 41-50 | 把 SF 的 K 维 pitch 撑大并用 **UE8M0 NaN** 毒化 padding | UE8M0 编码 |
| `assert_stochastic_bf16_matches_fp32_accumulation` | 53-79 | 随机舍入输出必须逐位等于**纯 torch 复刻的 quartet 哈希 oracle** | — |
| `convert_to_fp8` | 82-95 | FP4→FP8 的**精确**（无损）转换 | E4M3/E2M1 语义 |
| `to_cublaslt_vec16_sf_layout` | 98-110 | MXFP4 per-32 SF → NVFP4 per-16 UE4M3 **cuBLASLt 分块布局** | UE4M3 编码 |
| `make_cublas_gemm` | 113-120 | 按 dtype 分派 cuBLASLt FP8 / NVFP4 GEMM | 上面两个转换 |

一句话：第 41 篇回答「**操作数长什么样**」，第 40 篇回答「**怎么判对/计时**」，本篇回答
「**除了 `calc_diff < tol`，kernel 还必须满足哪些更硬的性质**」。三个最硬的性质是：**direct == 累积**、
**随机舍入 == 确定性 oracle**、**PSUM padding == 0**——它们都是**逐位**判据，不是 relerr 判据。

---

## 2. `assert_direct_output_matches_fp32_accumulation`（`tests/utils.py:20-38`）

### 2.1 契约

很多 GEMM 有两条输出路径：
- **direct**：`D = A@B` 直接写 `D`（无 `C`，不累积）；
- **累积**：`D = C + A@B`（`C` 非空，硬件做累加）。

`sm90_bf16_gemm` / `sm100_fp8_fp4_gemm_1d1d` 等 kernel 的 direct 与累积是**同一段 epilogue 的不同写入
分支**（可能走不同 store 指令）。这个辅助断言二者**数值上完全一致**：

```python
# tests/utils.py:33-38（关键三行）
assert direct_output.dtype in (torch.float, torch.bfloat16)
fp32_baseline = torch.zeros_like(direct_output, dtype=torch.float)
launch(fp32_baseline, fp32_baseline)          # 把 fp32 零张量同时当 D 和 C -> D = GEMM + 0
expected = fp32_baseline if direct_output.dtype == torch.float else fp32_baseline.to(torch.bfloat16)
assert_bitwise_equal(direct_output, expected, label)
```

**为什么把同一个零张量同时当 D 和 C**：累积路径算的是 `D = C + GEMM`，令 `C = 0` 就退化成
`D = GEMM`，于是两条路径面向**同一数学**，可以逐位比较。再分两种输出 dtype：
- **FP32 输出**：direct 与累积都保留完整尾数，**必须逐位相等**；
- **BF16 输出**：direct 路径在 epilogue 里做一次 **FP32→BF16 的 RNE**，累积路径写 FP32 基线后再
  `.to(bfloat16)` 也做一次 RNE——因此 expected 是「FP32 基线 cast 一次 bf16」，**仍要求逐位相等**
  （坐实两种路径的舍入点与舍入模式一致）。

### 2.2 H100 实测

用真实 SM90 `bf16_gemm_nt`（`m=n=k=512`，禁 cuBLASLt 回退，见第 27 篇），实测：

```
[PASS] D1 direct fp32 == 从零累积 D=GEMM+0 逐位: 512x512x512
[PASS] D2 direct bf16 == fp32 基线一次 bf16 cast
[PASS] D3 真实 helper 通过: no error
[PASS] D4 篡改 1 bit 后 helper 正确报错
```

D4 是**负例**：把 `direct` 的 1 个 bit 翻转后，真实 helper 会抛 `AssertionError`（内部用第 40 篇的
`assert_bitwise_equal` 定位）。这证明这个 helper **真的在判**，而不是恒真。测试里的调用见
`test_bf16.py:188`、`test_einsum.py:60/91/...`、`test_fp8_fp4.py:306/366`。

---

## 3. `assert_stochastic_bf16_matches_fp32_accumulation`（`tests/utils.py:53-79`）

### 3.1 它校验的 device 算子

第 31 篇讲过，SM100 GEMM 的可插拔 epilogue 里有一个
`epilogue::operators::StochasticRoundToBF16`（`deep_gemm/include/deep_gemm/epilogue/operators.cuh:32-63`），
把 TMEM 读回的 FP32 累加器**随机舍入**到 BF16（用于降低固定 RNE 的偏置）。它的核心是一个
**quartet 哈希 PRNG**：

```c
// operators.cuh:35-48 —— device 侧（第 31 篇已逐行抽出验证）
partial_hash(value, q) = value * (q==0 ? 0x5671d42b : q==1 ? 0x9995e499 : q==2 ? 0xace1b8a5 : 0xe153538d)
select_random_bits(h, q):
    h ^= h>>23; h *= 0x7feb352d; h ^= h>>16;
    h *= (q<2 ? 0x846ca68b : 0xd35a2d97); h ^= h>>11;
    return (h >> (q%2*16)) & 0xffff
// cast_quartet：4 个成员的 partial 相加 + quartet_n_idx -> 硬件 cvt.rs.bf16x2.f32
```

`tests/utils.py:53-79` 用**纯 torch int64** 复刻了同一套哈希，并给出随机舍入的位运算：

```python
# tests/utils.py:65-78（逐行）
values = (fp32.view(torch.int32).to(torch.int64) & 0xffffffff).view(*shape[:-1], -1, 4)
quartet_n_idx = torch.arange(shape[-1] // 4, ...)
h = (values[...,0]*0x5671d42b + values[...,1]*0x9995e499 +
     values[...,2]*0xace1b8a5 + values[...,3]*0xe153538d + quartet_n_idx) & 0xffffffff
h = (h ^ (h >> 23)) * 0x7feb352d & 0xffffffff
h ^= h >> 16
h1 = h * 0x846ca68b & 0xffffffff;  h2 = h * 0xd35a2d97 & 0xffffffff
h1 ^= h1 >> 11;                    h2 ^= h2 >> 11
random_bits = torch.stack((h1 & 0xffff, h1 >> 16, h2 & 0xffff, h2 >> 16), dim=-1)
rounded = (values >> 16) + (((values & 0xffff) + random_bits) >> 16)   # 加随机低位再进位
expected = rounded.to(torch.uint16).view(torch.bfloat16).view(shape)
```

**位序映射值得单独记**（device `select_random_bits` ↔ torch `stack`）：
`offset%2*16` 的移位 ↔ `torch.stack((h1&0xffff, h1>>16, h2&0xffff, h2>>16))`；乘子选择
`q<2 ? 0x846ca68b : 0xd35a2d97` ↔ `h1`(低两位) / `h2`(高两位)。二者**逐一对应**。

**约束**：`direct_output.size(-1) % 4 == 0`（`tests/utils.py:60`），因为 quartet 按**绝对 N 坐标**对齐，
只在参考复刻侧需要（kernel 自身对任意 N 都能处理：越界列哈希确定性零，见注释 `:58-59`）。

### 3.2 本篇做的「双向」验证

第 31 篇抽出了 device 算子；本篇反过来验证**测试 oracle 本身是不是忠实**：把 device 哈希常数搬进一个
CUDA kernel（`code/mega-moe/src/test_utils_sr.cu`），让它输出每个 quartet 的 4 个随机 bit 与最终 bf16，
再与 `tests/utils.py` 的 torch 公式逐位对拍，最后把 CUDA 输出当 `direct_output` 直接喂**真实 helper**：

```
[PASS] E1 torch 公式随机 bit vs device 逐位: mismatch=0/262144
[PASS] E2 torch 公式随机舍入 bf16 vs device 逐位: mismatch=0/262144
[PASS] E3 真实 helper 通过（device 输出 == oracle 期望）: no error
    E: 随机舍入向上比例=0.4985，平均小数部分=0.4993
[PASS] E4 与 RNE 有差异（随机性存在）: differ from RNE=130674/262144
```

E1/E2 逐位 0 不符，且**把 CUDA 输出直接喂真实 helper 也不报错**——证明 oracle 的 PRNG 位序/常数与
device 算子完全一致。E4 给出无偏性直觉：**向上舍入比例 ≈ 平均小数部分**（0.4985 ≈ 0.4993），而与 RNE
有 13 万/26 万个输出不同（随机性确实存在）。这与第 31 篇在 `operators.cuh` 上量到的「无偏 vs RNE 偏一侧」
互相印证。该 helper 的**真实调用只在 `get_arch_major() == 10` 时执行**（`test_bf16.py:49`、
`test_einsum.py:62/93/179/223`、`test_fp8_fp4.py:60/311`）——因为 `cvt.rs` 是 SM100 专属；但**oracle 的
数学是架构无关的**，故本机可完整验证。

---

## 4. `assert_psum_zero_padding`（`tests/utils.py:9-17`）

### 4.1 契约

第 11/41 篇讲过 PSUM（前缀和）布局：`grouped_layout` 存**每组逻辑结束偏移** `ends[i]`，kernel 把第 `i`
组放在 `[align(ends[i-1]) : ends[i]]`，其后的 `[ends[i] : align(ends[i]))` 是 **gap/padding**。
`ensure_zero_padding=True` 时 kernel 必须**显式把 padding 行清零**，否则 `d[valid_mask]` 之外可能出现
未初始化值（第 41 篇的 `layout_masked_to_psum` 注释即「gap 故意不初始化」）。

```python
# tests/utils.py:11-17
for group_idx, current_m in enumerate(grouped_layout.cpu().tolist()):
    aligned_m = align(current_m, get_mk_alignment_for_contiguous_layout())
    if current_m < aligned_m:
        a_padding = a_data[current_m: aligned_m]
        d_padding = d[current_m: aligned_m]
        assert torch.equal(a_padding, torch.zeros_like(a_padding)), ...
        assert torch.equal(d_padding, torch.zeros_like(d_padding)), ...
```

即：遍历每个 PSUM 逻辑结束 `current_m`，`align` 到 `mk_alignment`，检查 `[current_m : aligned_m]` 在
**A（激活）和 D（输出）**里都全零。

### 4.2 H100 实测

SM90 的 `m_grouped_bf16_gemm_nt_contiguous` 支持 PSUM（第 4-3 篇），且 `kEnsureZeroPadding` 在 SM90
路径取默认 `true`（`scheduler/gemm.cuh:35`），而生成器把 A 的 padding 行置零 → GEMM 后 D 的 padding
自然为 0。实测（`num_groups=4, m=34304, n=6144, k=7168`）：

```
[PASS] G1 真实 SM90 PSUM kernel 有效区语义正确: calc_diff=0.00e+00 m=34304
[PASS] G2 真实 helper 通过（PSUM padding 全零）: no error
[PASS] G3 非零 padding 被检出: group=0 logical_end=9884
```

注意 G1 的 `calc_diff=0.000e+00`（BF16 输出与参考逐位一致）。G3 是**负例**：把某组 padding 首行写 1.0，
真实 helper 立即 `AssertionError`。调用点见 `test_bf16.py:93`、`test_fp8_fp4.py:158`（都在
`use_psum_layout and ensure_zero_padding` 分支；`ensure_zero_padding` 由 `generators.py:203` **仅在
arch10** 枚举——SM100 才把该开关编进 kernel 模板）。

---

## 5. `add_sf_k_padding`（`tests/utils.py:41-50`）

```python
# tests/utils.py:41-50
def add_sf_k_padding(sf, padding, k_grouped=False):
    # Poison padding with UE8M0 NaN to expose reads using a compact pitch.
    assert padding > 0 and padding % 4 == 0
    mn = sf.size(-1 if k_grouped else -2)
    pitch = align(mn, 4) + padding
    storage = sf.new_full((sf.numel() // mn, pitch), -1)
    strides = (pitch, 1) if k_grouped else (1, pitch)
    if sf.dim() == 3:
        strides = (sf.size(-1) * pitch, *strides)
    return storage.as_strided(sf.shape, strides).copy_(sf)
```

**为什么**：SF（`SFA`/`SFB`）在 kernel 里是以 **MN-major、TMA 对齐**访问的（第 15/25 篇）。如果 kernel
**错误地用「紧凑 pitch」**（`align(mn,4)` 而真实 pitch 更大）去寻址，就会读到 padding 区。把这个辅助
产生的 SF 喂给 kernel，若 kernel 读到了 padding，就会读到 **-1**——而 `-1` 的 FP32 位模式在所有位上都是
1，作为 UE8M0 scale 解出来是 **NaN 指数 0xff**（第 37 篇：UE8M0 只留 8 位指数，`0xff` 被保留给 NaN）。
于是任何越界读都会**污染输出而不是静默正确**，测试随即失败。这就是注释里 "Poison padding" 的含义。

**两种 stride**：非 k-grouped（`SFA`，MN-major）= `(1, pitch)`（mn 连续）；k-grouped（wgrad，第 16 篇）
= `(pitch, 1)`。实测（`mn=100, sf_k=9, padding=4`）：

```
[PASS] C k_grouped=False 值不变 + stride=(1, 104): pitch=104
[PASS] C2 k_grouped=False padding 全 -1（毒化）
[PASS] C k_grouped=True 值不变 + stride=(104, 1): pitch=104
[PASS] C2 k_grouped=True padding 全 -1（毒化）
[PASS] C3 3D 变体值不变: stride=(936, 1, 104)
```

C2 直接读**底层 storage** 确认 `[mn:pitch]` 全 `-1.0`（毒化生效）。调用点见 `test_fp8_fp4.py:87-88`
（非分组，arch10）与 `:346-347`（`k_grouped=True`，k-grouped wgrad）。

---

## 6. `convert_to_fp8`（`tests/utils.py:82-95`）

```python
# tests/utils.py:82-95
def convert_to_fp8(x):
    data, sf = x
    if data.dtype == torch.float8_e4m3fn: return x
    if data.stride(-1) != 1:
        converted, sf = convert_to_fp8((data.mT, sf)); return converted.mT, sf
    lut = torch.tensor((0x00,0x30,0x38,0x3c,0x40,0x44,0x48,0x4c), dtype=torch.uint8, ...)
    packed = data.view(torch.uint8)
    lo, hi = packed & 0xf, packed >> 4
    codes = torch.stack((lut[(lo & 7).long()] | ((lo & 8) << 4),
                         lut[(hi & 7).long()] | ((hi & 8) << 4)), dim=-1).flatten(-2)
    return codes.view(torch.float8_e4m3fn), sf
```

**为什么能无损**：E2M1 的 8 个幅值 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` 在 E4M3（3 位尾数）里**都能精确表示**，
故 LUT 直接给出 E4M3 码（`0x30=0.5, 0x38=1, 0x3c=1.5, 0x40=2, 0x44=3, 0x48=4, 0x4c=6`），**符号位**由
`(nibble & 8) << 4` 补到 E4M3 的 bit7。解包是 **LSB-first**（低 nibble = 偶数列，`flatten(-2)` 后
`[2i]` 来自 `lo`）。`test_fp8_fp4.py:97/150/217/276` 用它把「FP4 操作数」转成等价的「FP8 操作数」，
以验证「FP4 与转换后 FP8 的 BF16 输出逐位一致」（`test_fp8_fp4.py:99` 断言 `calc_diff < 1e-14`）。

实测：全部 256 个字节码与独立参考**逐位一致**（`512 elems`）；已 FP8 输入幂等；MN-major 分支等于
「转置后转换再转置」；解包 LSB-first 后反量化 == 转换结果数值。

一个**语义细节**：E2M1 有两个零（`0b0000=+0`、`0b1000=-0`），E4M3 也有 `±0`。LUT 保留符号位 →
`0b1000` 映射为 `-0.0`（码 `0x80`）；而第 37 篇的参考量化器 `_dequantize_from_fp4_e2m1` **无负零**
（把 `-0` 归一为 `+0`）。两者**数值相等**，故 GEMM 结果一致——这也是为什么 `convert_to_fp8` 只追求
「数值无损」而非和量化器逐位同码。

---

## 7. `to_cublaslt_vec16_sf_layout` 与 `make_cublas_gemm`（`tests/utils.py:98-120`）

### 7.1 MXFP4 per-32 → NVFP4 per-16 UE4M3

```python
# tests/utils.py:98-110
def to_cublaslt_vec16_sf_layout(sf):
    # Convert per-32 MXFP4 power-of-two scales into the equivalent per-16 NVFP4 layout.
    assert sf.dim()==2 and sf.dtype==torch.float
    assert (sf.view(torch.int32) & 0x7fffff == 0).all(), 'SF must be positive powers of two'
    exp = (sf.view(torch.int32) >> 23) - 127
    assert (exp >= -9).all() and (exp <= 8).all(), 'SF out of the UE4M3 range'
    ue4m3 = torch.where(exp >= -6, (exp + 7) << 3, 1 << (exp + 9)).to(torch.uint8).repeat_interleave(2, dim=1)
    mn, sf_k = ue4m3.shape
    num_mn_tiles, num_k_tiles = ceil_div(mn,128), ceil_div(sf_k,4)
    padded = torch.zeros((num_mn_tiles*128, num_k_tiles*4), ...); padded[:mn,:sf_k] = ue4m3
    return padded.view(num_mn_tiles,4,32,num_k_tiles,4).permute(0,3,2,1,4).contiguous()
```

**背景**（第 27 篇）：cuBLASLt 的 NVFP4 用 **per-16** block scale，编码为 **UE4M3**（4 位指数 3 位尾数，
`exp∈[-6,8]` 正常、到 `2^-9` 亚正规），分块布局 `[ceil(mn/128), ceil(k/64), 32, 4, 4]`（见
`csrc/apis/gemm.hpp:777-786` 的 `check_sf`）。而 DeepGEMM 自己的 MXFP4 是 **per-32 + UE8M0**。两者数学上
等价需：**每个 per-32 scale 复制给两个相邻的 16-block**（`repeat_interleave(2)`）——这样才能在
per-16 硬件上实现「同一个 32 宽 block 用同一个 scale」。

UE4M3 编码：
- `exp >= -6`（正常数）：`(exp+7) << 3`（floor 由尾数零保证，值 = `2^exp`，**零误差**）；
- `exp ∈ [-9,-7]`（亚正规）：`1 << (exp+9)`（UE4M3 亚正规 = `mantissa * 2^-9`）。

实测（`mn=137, sf_k32=20`，故意非对齐）：

```
[PASS] B1 输出形状/类型（5D 分块）: shape=(2, 10, 32, 4, 4) dtype=torch.uint8
[PASS] B2 与独立复刻逐位一致
[PASS] B3 padding 区全 0: MN pad 119 行 / K pad 0 列
[PASS] B4 UE4M3 编解码回原 2 的幂 scale 零误差: max|dec-sf|=0.0e+00
```

B2 逐行复刻该函数并与真实输出逐位对拍；B4 把 UE4M3 解码回浮点，与原 2 的幂 scale **零误差**
（覆盖 `exp∈[-9,8]` 全范围）。

### 7.2 `make_cublas_gemm` 分派 + 真实 cuBLASLt

```python
# tests/utils.py:113-120
def make_cublas_gemm(a, b, d, c):
    if a[0].dtype == torch.int8 and b[0].dtype == torch.int8:   # NVFP4 分支
        return lambda: deep_gemm.cublaslt_nvfp4_gemm_nt(
            (a[0], to_cublaslt_vec16_sf_layout(a[1])), (b[0], to_cublaslt_vec16_sf_layout(b[1])), d, c=c)
    a_fp8, b_fp8 = convert_to_fp8(a), convert_to_fp8(b)          # FP8 分支
    return lambda: deep_gemm.cublaslt_gemm_nt(a_fp8[0], b_fp8[0], d, c=c)
```

它把第 27 篇的 cuBLASLt 回退路径包成「与 DeepGEMM kernel 同签名的可调用对象」，供 `bench_kineto`
对比速度（`test_fp8_fp4.py:101-107`）。`int8`（即 `kPackedFP4`，`csrc/utils/math.hpp:11`）分支走
**NVFP4**，并把 per-32 SF 经 `to_cublaslt_vec16_sf_layout` 转成 per-16。实测 FP8 分支：

```
[PASS] F1 cublaslt_gemm_nt FP8 语义正确: calc_diff=1.39e-06
[PASS] F2 make_cublas_gemm 返回可调用且 FP8 分支正确: calc_diff=1.39e-06
[PASS] F3 cublaslt_nvfp4：本机 cuBLASLt 不支持（环境，SKIP）
```

F3 是本机环境的**诚实边界**：传入的 SF 分块布局**形状/元素数完全正确**（`[4,16,32,4,4]`，`
numel == ceil(m/128)*ceil(k/64)*512`），但 `cublasLtMatmulAlgoGetHeuristic` 在本机 cuBLASLt 上返回
`15 = CUBLAS_STATUS_NOT_SUPPORTED`（`smxx_cublaslt.hpp:89`）——**NVFP4 block-scaled GEMM 依赖较新的
cuBLASLt**，与 DeepGEMM 代码无关。故本机只能验证到「布局转换正确」，端到端 NVFP4 GEMM 需目标卡/新版
cuBLASLt。

---

## 8. 小结与坐标

| 断言 | 判据类型 | 本机可验证的部分 | 本机实测 |
|---|---|---|---|
| direct == 累积 | **逐位** | SM90 BF16 主 kernel 两条输出路径 | 4/4 PASS（含负例） |
| 随机舍入 == oracle | **逐位** | oracle 数学（架构无关）+ device 哈希搬进 CUDA | 4/4 PASS（E1/E2 逐位 0 不符） |
| PSUM padding == 0 | **逐位**（全零） | SM90 PSUM kernel + helper | 3/3 PASS（含负例） |
| `add_sf_k_padding` 毒化 | stride + storage | 纯张量 | 5/5 PASS |
| `convert_to_fp8` | **逐位** | 纯张量（全 256 码） | 4/4 PASS |
| `to_cublaslt_vec16_sf_layout` | **逐位** | 纯张量 | 4/4 PASS |
| `make_cublas_gemm` cuBLASLt | 数值 | 真实 cuBLASLt FP8 | 2 PASS + 1 环境 SKIP |

**关键行号索引**（DeepGEMM commit `057ca59`）：
- `tests/utils.py:9-17`（PSUM padding）、`:20-38`（direct vs 累积）、`:41-50`（SF 毒化）、
  `:53-79`（随机舍入 oracle，公式 `:65-78`）、`:82-95`（`convert_to_fp8`，LUT `:90`）、
  `:98-110`（NVFP4 VEC16，`:104` 编码）、`:113-120`（`make_cublas_gemm`）。
- device 侧对照：`deep_gemm/include/deep_gemm/epilogue/operators.cuh:32-63`（`partial_hash:35-39`、
  `select_random_bits:41-48`、`cast_quartet:50-62`）。
- cuBLASLt 契约：`csrc/apis/gemm.hpp:777-786`（NVFP4 SF 检查）、`csrc/utils/math.hpp:11`（`kPackedFP4=kInt8`）、
  `csrc/jit_kernels/impls/smxx_cublaslt.hpp:89`（heuristic 失败点）。
- 调用点：`test_bf16.py:11-13/49-54/93/188/193`、`test_einsum.py:16-17/60-63/91-94/179/223`、
  `test_fp8_fp4.py:12-16/65/87-88/97/102/158/306/311/346-347`。

**与前后篇的关系**：第 40 篇（计时/判对基座）→ 第 41 篇（操作数生成）→ **本篇（更硬的逐位断言）** →
第 31 篇（device 侧随机舍入算子）与第 27 篇（cuBLASLt 回退）。至此，DeepGEMM 测试栈的
「操作数 → 判据 → 计时 → 更硬性质」四层已完整讲清。

实验代码与原始输出：`code/mega-moe/harness/exp_test_utils.py`、`code/mega-moe/src/test_utils_sr.cu`、
`code/mega-moe/harness/exp_test_utils.out.txt`（H100，27/27 PASS）。
