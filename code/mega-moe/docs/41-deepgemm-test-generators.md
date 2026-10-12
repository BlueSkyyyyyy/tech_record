# DeepGEMM 测试的操作数/布局生成器：`tests/generators.py`

> 上一篇（第 40 篇）拆了测试基座 `deep_gemm/testing/`：**怎么判对**（`calc_diff`）、**怎么量快**
> （`bench_kineto`）、**怎么拼不等长索引**（`uneven_all_gather`）。但每个 `tests/test_*.py`
> 在调用这两个函数之前，还有一步一直被跳过——**它自己构造出来的「操作数」到底长什么样**。
> 这一步就是 `tests/generators.py`（608 行，被 `test_bf16.py` / `test_fp8_fp4.py` /
> `test_einsum.py` / `test_attention.py` / `test_mega_moe.py` / `test_layout.py` … 全部 `import`）。
>
> 它不是「随便造点随机数」：这个文件**把 DeepGEMM 所有 kernel 期望的操作数约定（operand major、
> 三种 grouped 布局、PSUM 间隔、K 分组 SF 打包）用 Python 写成了可执行的参考**。读懂它，
> 前面各篇里反复出现的 `major_a/major_b`、`grouped_layout`、`use_psum_layout`、`ks_cpu` 才真正落地。
>
> 它**不是 kernel、不含 SM100 专属 PTX**，因此 **H100（sm90）真机可跑**。本文用真实数字逐条核对
> 它编码的约定，并用真实 SM90 kernel 做端到端对拍。

参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1），本机 `kernel_lab`（H100 sm90）。
实验：`harness/exp_test_generators.py`（driver `harness/exp_test_generators.sh`），
输出 `harness/exp_test_generators.out.txt`，**41 PASS / 0 FAIL**。

---

## 1. 它在仓库里的位置：三层测试栈

```
tests/test_fp8_fp4.py     ← 逐 kernel 的测试主体：调用 kernel、算 calc_diff、bench
   ├── tests/generators.py ← 构造「操作数 + 参考输出 + 布局」（本文）
   │      └── deep_gemm/utils/math.py ← 参考量化器（第 37 篇）
   └── deep_gemm/testing/  ← 判对/计时/分布式（第 40 篇）
```

`generators.py` 的两个职责，正是测试能成立的两个前提：

1. **枚举**（`enumerate_*`）：给定 `dtype` 与 arch，列出「要测哪些 (shape, layout, accumulate,
   out_dtype) 组合」。arch 条件（sm90/sm100）在这里把大量 SM100 专属候选**提前剪掉**。
2. **生成**（`generate_*`）：把一组枚举参数变成真实的 CUDA 张量（含参考量化、布局重排、padding、
   参考输出 `ref_d`），供测试直接喂给 kernel。

顶部导入即可看出这两件事（`generators.py:8-15`）：`get_arch_major`（arch 分支）、
`per_token_cast_to_fp8 / per_channel_cast_to_fp8 / per_block_cast_to_fp8 / per_token_cast_to_fp4 /
transpose_packed_fp4`（参考量化，第 37 篇）、`get/set_mk_alignment_for_contiguous_layout`。

---

## 2. 三个基础类型

### 2.1 `KernelType`（`:18-30`）

```python
class KernelType(enum.Enum):
    Kernel1D1D = 0   # A/B 都 per-token 1D scale  -> 第 32 篇
    Kernel1D2D = 1   # SFA 1D + SFB per-128 块      -> 第 32 篇
    KernelNoSF = 2   # 无 SF（BF16）                -> 第 19/29 篇
```

它只描述 **B 侧（或两侧）的 SF 粒度**，是「用哪个主 kernel」的测试侧标签。`get_kernel_types`
（`:101-105`）按 arch/dtype 决定候选：

```python
def get_kernel_types(dtype):
    if dtype == torch.bfloat16: return (KernelNoSF, )
    return (Kernel1D2D,) if get_arch_major() == 9 else (Kernel1D1D,)
```

sm90 FP8 只有 `1D2D`（`gran_n=128`），sm100 FP8/FP4 只有 `1D1D`——这与第 32 篇的结论
（`gemm.hpp:117-126` 按 `gran_n==1?1d1d:1d2d` 分派）完全一致，只是从**测试枚举**这一侧看。

### 2.2 `MajorTypeAB`（`:33-41`）

`KMajor` / `MNMajor`，逐 operand（A、B）可选。`get_major_ab(allow_a_mn_major, allow_b_mn_major)`
（`:108-115`）把两个 flag 展开成 `(major_a, major_b)` 组合；不允许 MN-major 的一侧被跳过。

### 2.3 `QuantConfig`（`:44-86`）——操作数的量化「配方」

`QuantConfig` 是 `(gran_k_a, gran_k_b, is_fp4_a, is_fp4_b)` 四元组：

```python
_legacy_quant_config = (128, 128, False, False)           # :45  传统 per-128 FP8xFP8
def get_recipes(self, is_wgrad=False):                    # :60-67
    recipe, recipe_a, recipe_b = None, None, None
    if self.is_legacy():
        recipe = (1, 1, 128) if is_wgrad else None
    else:
        recipe_a = (1, self.gran_k_a)
        recipe_b = (1, self.gran_k_b) if self.is_fp4_b or is_wgrad else (self.gran_k_b, self.gran_k_b)
    return recipe, recipe_a, recipe_b
```

三个「为什么」：

- **legacy 前向 recipe 返回 `None`**：说明 SM90 默认走全 `(1,128,128)`（第 25 篇
  `utils/layout.hpp` 的默认），测试没必要显式覆盖。
- **legacy wgrad 返回 `(1,1,128)`**：K 分组 GEMM 必须 A/B 都 per-token（`gran_n=1`），
  即强制走 `1D1D`（第 12/32 篇）。
- **非 legacy 的 B recipe 平时是 `(gran_k, gran_k)`（块状 2D），但 `is_fp4_b or is_wgrad`
  时退化成 `(1, gran_k)`（per-channel 1D）**——因为 FP4 与 K 分组都不支持 B 侧 2D 块 scale。

`max_diff`（`:69-74`）是**每类量化的容差**：有 FP4 → 0.02（双边）/0.01（单边），否则 0.001。
`get_list_from_dtype`（`:76-86`）给 bf16 返回 `[None]`，给 FP4 返回单一 `(32,32,True,True)`，
给 FP8 返回 `[(128,128,F,F)]`，**只有 SM100 才追加 `(128,32,F,T)` / `(32,32,T,T)`**。

实测（`exp_test_generators.out.txt` §B）：
```
legacy 前向 recipe = (None, None, None)
legacy wgrad recipe = ((1, 1, 128), None, None)
fp4/fp4 recipes wgrad = (None, (1, 32), (1, 32))     # B 退化
legacy/fp8fp4/fp4fp4 max_diff = 0.001 / 0.01 / 0.02
```

---

## 3. mk-alignment：枚举器的**全局副作用**（一个真坑）

第 13/25/26 篇反复出现的 `get_mk_alignment_for_contiguous_layout()` 在 C++ 侧是一个**全局变量**
（`heuristics/runtime.hpp:49-64`），测试通过 `deep_gemm.set_mk_alignment_for_contiguous_layout`
（绑定于 `csrc/apis/layout.hpp:133-137`）改它。默认值 = `get_theoretical_...`（`:57-64`）：
sm90 恒 `128`，sm100 恒 `256`（注释：UMMA_N=256 起，统一 256 更利于 MoE cast / M-grouped /
K-grouped）。

**坑**：`enumerate_k_grouped_contiguous`（`:258`）、`enumerate_sf_layout`（`:284`）、
`enumerate_k_grouped_sf_layout`（`:297`）、`enumerate_k_grouped_psum_sf_layout`（经前者）
在**每次迭代里都调用 `set_mk_alignment_for_contiguous_layout(...)`**。所以「先跑枚举器、再读
alignment」会读到枚举器最后留下的值。实测：

```
初始 mk-alignment（任何枚举器调用之前） = 128
[side-effect] normal 后=128, k_contiguous 后=128,
              sf_layout 后=128, k_grouped_sf_layout 后=384
```

这解释了 test 里为什么在真正调用 `generate_*` 前**显式**设一次 alignment（例如
`tests/test_fp8_fp4.py:127-128` `alignment = get_theoretical...(); set_mk_alignment...`）：

```
进程初始 alignment == theoretical == 128 （sm90）
set(256) / get() roundtrip 通过
```

本文后续每个实验都先 `set_mk_alignment_for_contiguous_layout(128)` 归位。

---

## 4. operand major 的物理编码（`:318-347`）

`cast_fp8_fp4_with_major`（`:318-326`）是理解 `major_*` 的关键：

```python
def cast_fp8_fp4_with_major(x, major, gran_k, is_fp4, use_ue8m0, use_block_cast_for_fp8=False):
    if is_fp4:
        x_fp4 = per_token_cast_to_fp4(x, use_ue8m0=use_ue8m0, gran_k=gran_k)
        return x_fp4 if major.is_k_major() else (transpose_packed_fp4(x_fp4[0]).T, x_fp4[1])
    else:
        x_fp8 = per_block_cast_to_fp8(...) if use_block_cast_for_fp8 else per_token_cast_to_fp8(...)
        return x_fp8 if major.is_k_major() else (x_fp8[0].T.contiguous().T, x_fp8[1])
```

- **FP8 MN-major = `.T.contiguous().T`**：先做成 `(k,m)` 连续、再取转置 view，得到形状
  仍是 `(m,k)` 但 **stride `(1,m)`**（K 是内维）。实测：K-major `stride (128,1) contiguous=True`；
  MN-major `stride (1,64) contiguous=False`，且两者转置后逐位一致。
- **FP4 MN-major = `transpose_packed_fp4(x).T`**：`transpose_packed_fp4`（`utils/math.py:128-141`）
  把 `(M, K/2)` 解包→转置→重打包成 `(K, M/2)`，再 `.T` 得 `(M/2, K)`。实测：

```
K-major fp4 packed (M,K/2) = (64,64)
MN-major fp4 packed        = (32,128) = (M/2, K)
transpose_packed_fp4: (M,K/2) -> (K,M/2)   [PASS]
MN-major fp4 == transpose_packed_fp4(K-major).T   [PASS]
由 transpose_packed_fp4(f_mn.T) 无损还原 K-major   [PASS]
```

> **为什么 FP8 的 MN-major 用「转置 view」而 FP4 要重打包**：FP8 每元素 1 字节，MN-major 只是
> 换 stride；FP4 每 2 个 K 元素挤进 1 字节，一旦 MN 成为内维，**K 的配对就变成 MN 的配对**，
> nibble 必须重新打包，无法只靠 stride 表达。这正是第 35 篇 FP4「packed、K-major、gran32」
> 约束的测试侧来源。

注意 **SF 不随 major 改变**（实测 `sf_km == sf_mn`）：布局只影响 data，SF 由专用转置/打包走
（第 15/25/26 篇）。

---

## 5. 三种 grouped 布局的生成（M 轴分组 = dgrad）

### 5.1 `generate_m_grouped_contiguous`（`:378-419`）

```python
actual_ms  = [int(expected_m * U(0.7,1.3)) for _ in range(num_groups)]   # :383
aligned_ms = [align(a, get_mk_alignment_for_contiguous_layout()) ...]     # :384
m = sum(aligned_ms)                                                       # :385
...
for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
    if use_psum_layout:
        grouped_layout[i] = actual_end                                    # :400 长度 G 的结束偏移
    else:
        grouped_layout[start: actual_end] = i                             # :402 逐行组号
        grouped_layout[actual_end: aligned_end] = -1                      # :403 padding = -1
    valid_mask[start:actual_end] = True
    a[actual_end: aligned_end] = 0                                        # :406 padding 置零
    ref_d[start: aligned_end] = a[start: aligned_end] @ b[i].t()          # :407
    start = aligned_end
```

要点：
- **contiguous = 长度 m 的逐行组号**，padding 行 `-1`（同时也是 `valid_mask` 的补集）；
- **PSUM = 长度 G 的「每组逻辑结束偏移」**，组 i 的有效行在 `[align(prev_end), end_i)`；
- **A 的 padding 行显式置零**（`:406`，注释：让量化后的 SFA padding 规则、非未初始化）。

实测（`out.txt` §F，G=4）：
```
contiguous: m=768, 逐行组号唯一值 [-1,0,1,2,3], padding 237 行, valid=531
  padding 行 == ~valid_mask                [PASS]
  有效行组号单调不减                        [PASS]
  contiguous A padding 行量化后全零         [PASS]
PSUM: ends=[154,403,633,749], alignment=128, m=768, valid=531
  单调不减 / align(ends[-1])==m / valid_mask==[align(prev),end)   [PASS]
  PSUM grouped_layout == build_psum_layout_from_ks(actual_ms)     [PASS]
```

> `ends[-1]=749` 但 `m=768`：因为 `m` 是**对齐后**总和，而组间还要插 `align` 间隙，
> `align(749,128)=768`。这与第 13 篇讲的「段必须对齐 BLOCK_M」是同一件事的两种视角。

### 5.2 `generate_m_grouped_masked`（`:446-485`）与 PSUM 转换

masked 布局是 `[G, max_m, ·]` 稠密三维 + 长度 G 的 `masked_m`（每真实行数）。两个 PSUM 转换器：

- `layout_masked_to_psum`（`:422-431`）：把 3D masked 张量按 `[align(prev), psum_m[i])` 塞进
  2D PSUM 张量，**gap 故意不初始化**（注释：用于验证 pack kernel 跳过 gap）；
- `masked_valid_mask_to_psum`（`:434-443`）：只构造 valid_mask，**gap 必须为 False**（否则
  `d[valid_mask]` 会读到被跳过的行，破坏确定性检查）。

实测（`out.txt` §G）：
```
masked: a.shape=(3,4096,128), masked_m=[241,230,190], valid=661
psum_m=[241,486,702], psum a.shape=(768,128)
valid 总数相等 / 逐组 masked 有效行 == psum 对应区间   [PASS]
```

### 5.3 真实 SM90 kernel 对拍（证明生成器「kernel-ready」）

把生成的操作数直接喂真实 kernel，与 `ref_d` 用 `calc_diff` 对：

| 入口 | 布局 | calc_diff |
|---|---|---|
| `fp8_fp4_gemm_nt` | 稠密 K/K | 6.72e-4 |
| `m_grouped_fp8_fp4_gemm_nt_contiguous` | m 分组 contiguous | 6.83e-4 |
| `m_grouped_fp8_fp4_gemm_nt_masked` | m 分组 masked | 6.85e-4 |
| `k_grouped_fp8_gemm_nt_contiguous` | k 分组 contiguous | 2.22e-4 |

全部 < `max_diff=1e-3`。**这说明生成器的约定与真机 kernel 完全匹配**。

> `m_grouped_fp8_fp4_gemm_nt_masked` 的真实签名（`tests/test_fp8_fp4.py:207`）是
> `(a, b, d, masked_m, expected_m, **common)`；PSUM 则改走 `..._nt_contiguous(..., use_psum_layout=True,
> expected_m_for_psum_layout=expected_m, ...)`（`:204-205`）。生成器对两者返回的 `grouped_layout`
> 分别是 `masked_m` / `psum_m`（`:469/:485`）。

---

## 6. K 轴分组（wgrad）：`generate_k_grouped_contiguous`（`:551-608`）

### 6.1 布局：两种 `grouped_layout` 的含义

```python
host_ks_cpu = [align(k, k_alignment) for k in logical_ks_cpu]        # :558
if use_psum_layout:
    logical_group_ends = build_psum_layout_from_ks(logical_ks_cpu, k_alignment)  # :561
    grouped_layout_cpu = logical_group_ends
else:
    # 每组逻辑数据放在物理块开头，尾部是 padding
    logical_group_ends.append(physical_start + logical_k)            # :569
    grouped_layout_cpu = host_ks_cpu                                 # :571  长度 G 的每组物理 K
```

- **contiguous**：`grouped_layout = host_ks_cpu`（长度 G，每组**物理** K，已对齐）；
- **PSUM**：`grouped_layout = logical_group_ends`（长度 G，每组**逻辑** K 结束偏移），由
  `build_psum_layout_from_ks`（`:542-548`）给出：

```python
def build_psum_layout_from_ks(real_ks, k_alignment):
    psum, prev_end = [], 0
    for k in real_ks:
        end = align(prev_end, k_alignment) + k
        psum.append(end); prev_end = end
    return psum
```

实测（`out.txt` §H，real_ks=[100,200,260,300]）：
```
host_ks(对齐后) = [128,256,384,384]  total_k=1152
contiguous grouped_layout = [128,256,384,384]          # 物理 K
PSUM grouped_layout       = [100,328,644,1068]         # 逻辑结束偏移
build_psum_layout_from_ks == align(prev)+k 递推          [PASS]
```

> **两种布局的分工**（第 13/26 篇）：contiguous 的 K 长度是**物理对齐 K**，最省事；
> PSUM 的 K 长度是**逻辑 K**，靠「每段起点向上对齐」留 gap——DeepEP 动态输出（真实 token 数
> 未知）只能给逻辑长度，故用 PSUM 桥接。

### 6.2 按组零填充与 SF

- A/B 是 `(total_k, m)`/`(total_k, n)`，**只有每组开头的 logical_k 行被写入随机数，其他全 0**
  （`:576-587`）。实测 MN-major A 的非零行数 `860` 恰等于 `Σ logical_k`，且位置与外推一致。
- **K-major 的 data 是 1D**：`k_grouped_cast_fp8_fp4_with_major`（`:535-539`）把每组
  `(k_g, mn)` 转成 `(mn, k_g)` 后 `flatten()`，得到 `[total_k*mn]` 的一维 K-major 拼接
  （每段在组内连续）。这是第 32 篇 SM90 `1D1D` 原地改 tensormap 的输入形态。
- **K 分组用 `per_channel_cast_to_fp8`**（`:528`），不是 per-token：`per_channel_cast_to_fp8`
  （`utils/math.py:52-59`）沿第 0 维以 `gran_k` 分组求 amax，SF 形状 `(K/gran_k, MN)`——即
  **scale 随收缩维 K 变化、在 MN 上共享**。这正是 wgrad「沿收缩维 K 分组」所需的粒度。
- **每组 SF 行独立对齐补零**（`:520-531`）：per-group 先零填充到 `host_k`，再量化，SF 行数
  `= Σ ceil_div(host_k, gran_k)`。实测 K-major SF 形状 `(mn, 9)`，`Σ ceil_div(host_k,128)=9`。

### 6.3 测试变体（`:265-273`）

```python
def enumerate_k_grouped_contiguous_test_variants(real_ks_cpu):
    yield list(real_ks_cpu)                                                  # 原样
    empty_ks_cpu = [0 if random() < 0.25 else k for k in real_ks_cpu]
    empty_ks_cpu[0], empty_ks_cpu[-1] = 0, real_ks_cpu[-1]                   # 首 0、末非 0
    yield empty_ks_cpu
    yield [k if i == len//2 else 0 for ...]                                  # 单组
```

实测三种变体：`[100,200,260,300]` / `[0,200,260,300]` / `[0,0,260,0]`——专门压 **空组**
（第 12/16 篇的「空组 D 保留 / 写零」）与「首空」边界。

### 6.4 真实 SM90 wgrad 对拍

```
k_grouped_fp8_gemm_nt_contiguous vs ref_d calc_diff = 2.22e-4   [PASS]
```

（`recipe=(1,1,128)`、accumulate=True、fp32 输出，与 `tests/test_fp8_fp4.py:258-262` 的调用一致。）

---

## 7. 枚举器全貌（sm90 实测数量）

`out.txt` §A 的普查结果（dtype 取 FP8 e4m3）：

| 枚举器 | 行号 | sm90 用例数 | 说明 |
|---|---|---|---|
| `enumerate_normal` | `:122-173` | 42 | 前向 3 个 m × 7 个 NK × ... + 反向 |
| `enumerate_m_grouped_contiguous` | `:194-210` | 16 | G∈{4,8}、`get_major_ab` 组合 |
| `enumerate_m_grouped_masked` | `:213-226` | 32 | 4 组 `(G,m)` |
| `enumerate_k_grouped_contiguous` | `:229-262` | 7 | 7 个 `(G,m,n,expected_k)` |
| `enumerate_transpose` | `:311-315` | 60 | SF 转置 kernel |
| `enumerate_sf_layout` | `:276-290` | 108 | 单组 SF 布局 |
| `enumerate_k_grouped_sf_layout` | `:293-300` | 48 | K 分组 SF |
| `enumerate_k_grouped_psum_sf_layout` | `:303-308` | 48 | K 分组 SF + PSUM |

可见 arch 条件如何收窄：
- `enumerate_normal` 的 sm90 FP8 反向：A **必须** K-major、wgrad 强制 `Kernel1D1D`（`:161-163`）；
- `enumerate_m_grouped_contiguous` 的 B 侧在 sm90 FP8 下**只能是 K-major**（`:207`），
  实测 `(major_a, major_b)` 组合仅 `[('KMajor','KMajor')]`；
- `enumerate_k_grouped_contiguous` 的 sm90 FP8：SF 只有 `(128,128)`、`psum_list=(False,)`、
  `cd_options=[(True, fp32)]`（`:235/:244/:246`）——只有 accumulate + fp32（第 12/32 篇）。
- 首个 K-grouped 形状 `(8, 768, 2048, 128)` 有注释「many small groups, for stressing the SM90
  in-place tensor map update」（`:250`）——正是第 12 篇原地改 tensormap 的压力用例。

---

## 8. 结论

`tests/generators.py` 把 DeepGEMM 的**操作数契约**从 kernel 源码里「翻译」成 Python 参考：

1. **major 编码**：FP8 MN-major = `.T.contiguous().T`（换 stride），FP4 MN-major =
   `transpose_packed_fp4(x).T`（重打包，因 K 配对变成 MN 配对）；SF 与 major 无关。
2. **三种 grouped 布局**：contiguous（逐行组号 / 每组物理 K）、masked（3D + `masked_m`）、
   PSUM（长度 G 的逻辑结束偏移，由 `build_psum_layout_from_ks` / `align(prev)+k` 递推）。
3. **对齐是全局副作用**：枚举器会改写 `mk_alignment`，test 必须显式设；sm90=128、sm100=256。
4. **K 分组用 per-channel SF**（`(K/gran_k, MN)`），每组 SF 行独立对齐补零；
   FP4/K 分组的 B 侧 recipe 退化为 `(1,gran_k)`。
5. `max_diff` 把「可接受的量化噪声」按精度分档（FP8 1e-3 / FP8xFP4 1e-2 / FP4 2e-2）。

**H100 真机**：41 PASS / 0 FAIL；四个真实 SM90 kernel（稠密 / m-contig / m-masked / k-contig）
用生成器造的操作数与 `ref_d` 对拍 `calc_diff` 6.7e-4 / 6.8e-4 / 6.8e-4 / 2.2e-4，全部达标。
SM100 专属分支（`get_list_from_dtype` 的额外 FP8 配置、PSUM SF 打包、`theoretical=256`）
在本机未执行，已标注。

---

## 附：参考坐标

- 源码：`tests/generators.py`（commit `057ca59`），`deep_gemm/utils/math.py`，
  `csrc/jit_kernels/heuristics/runtime.hpp:49-64`，`csrc/apis/layout.hpp:133-141`。
- 调用方：`tests/test_fp8_fp4.py:18-23,30-31,44,117,130,182,198,201-207,258-262,373`；
  `tests/test_bf16.py`、`tests/test_einsum.py`、`tests/test_attention.py`、`tests/test_mega_moe.py`。
- 相关篇：第 12 篇（wgrad K 分组）、第 13 篇（dgrad M 分组三布局）、第 25/26 篇（SF 布局与
  K 分组打包）、第 27 篇（参考量化器）、第 32 篇（SM90 FP8 1D1D/1D2D）、第 40 篇（测试基座）。
- 本专题实验：`harness/exp_test_generators.py` / `.sh` / `.out.txt`。
