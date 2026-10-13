# SF layout 家族的验收测试即正确性规范（`tests/test_layout.py`）

> 本专题（Mega MoE 前向 / 反向算子实现解析）的第 78 篇。block-scale GEMM 有一条**贯穿所有
> FP8/FP4 kernel 的前置步骤**——把 fp32 block-scale（SF）转成 kernel 期望的 **MN-major + TMA 对齐
> （+ UE8M0 打包）**布局。本专题已拆过它的**实现**：第 25 篇（`docs/25-sf-layout-transform.md`）
> 拆了单组 SF 的转置 / 打包四分派与 device `smxx_layout.cuh`，第 26 篇
> （`docs/26-k-grouped-sf-packing.md`）拆了 K 轴分组的「每组 SF 行独立对齐补零」。
>
> 但**「变换后的布局到底长什么样才算对」的唯一规范**——`tests/test_layout.py`——一直只被顺带引用。
> 本篇把它当作**「SF 布局变换算什么对」的可执行规格**来拆：不是再讲一遍 kernel 实现，而是讲清楚
> **每个出口要被哪些逐位判据卡**，以及测试自带的那份**纯 torch 参考实现**凭什么当权威。
>
> `test_sf_layout_kernels` 的三个 kernel（`transpose_fp32` / `transpose_and_pack_fp32_into_ue8m0` /
> `pack_fp32_into_ue8m0`）**都不是 SM100 专属，H100(sm90) 真机可跑**；只有两个 k-grouped 函数
> 被 `get_arch_major()==10` 门控跳过。本文所有行号均对照 DeepGEMM commit `057ca59`
> （Public release 26/09/30）核对过。实测见
> `code/mega-moe/harness/exp_sf_layout_test_oracle.{sh,py,out.txt}`。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 定位：为什么布局变换需要一份逐位 oracle

一个 block-scale GEMM 的入口是「量化后的操作数 + 它的 scale」。但 kernel 从 TMA 里读到的 scale
**不是**用户手里那份 fp32 SF，而是一份被 DeepGEMM 重新排过版的紧凑表示：**MN-major、按 TMA 粒度
对齐、并把 4 个 UE8M0 指数打包进一个 `int32`**。这一步在 `csrc/apis/layout.hpp:11-58` 的
`transform_sf_into_required_layout` 里完成，由 JIT（`csrc/jit_kernels/impls/smxx_layout.hpp`）
在首次调用时现编一个小 kernel。

关键在于：**这一步是纯数据搬运 + 位域重排，一旦错一格不会崩，只会让数值悄悄偏小或偏大**。
所以测试不能用 `calc_diff`（容差）判，必须用**逐位相等**（`torch.equal`），并且必须有一份
**不依赖任何 kernel 的参考实现**作为 oracle。`tests/test_layout.py` 的全部内容就是这两件事：

```text
tests/test_layout.py
├── get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl  :20-42   ← 纯 torch oracle
├── test_sf_layout_kernels                                    :45-80   ← 单组 SF（H100 可跑）
├── test_k_grouped_sf_layout_kernels                          :83-115  ← K 轴分组（SM100 专属）
├── test_k_grouped_psum_sf_layout_kernels                     :118-154 ← K 轴分组 + PSUM（SM100 专属）
└── __main__                                                   :157-163 ← 顺序调用三个
```

它对应三条 Python 入口（`deep_gemm/utils/layout.py:1-9` 只是把 `_C` 的 binding 平铺出来）：

| 测试函数 | Python 入口 | C++ 实现 | 本专题 |
|---------|------------|---------|--------|
| `test_sf_layout_kernels`（`use_ue8m0=False`） | `get_mn_major_tma_aligned_tensor` | `smxx_layout.hpp:60-106` | 第 25 篇 |
| `test_sf_layout_kernels`（`use_ue8m0=True`） | `get_mn_major_tma_aligned_packed_ue8m0_tensor` | `smxx_layout.hpp:133-216` | 第 25 篇 |
| `test_k_grouped_sf_layout_kernels` | `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor` | `smxx_layout.hpp:218-283` | 第 26 篇 |
| `test_k_grouped_psum_sf_layout_kernels` | 同上（`use_psum_layout=True`） | 同上 | 第 26 篇 |

---

## 2. 纯 torch oracle：`get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`

这是测试的**权威**。它不用任何 DeepGEMM 代码，只用 `view` / `>>` / `view(dtype)` / `mT`
几件标准操作，把「fp32 SF → 打包后的 MN-major 布局」定义清楚（`test_layout.py:20-42`）：

```python
def get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(x):
    assert x.dtype == torch.float and x.dim() in (2, 3)      # :21
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8) # :24  ← UE8M0 = fp32 的最高字节
    mn, k = x.shape[-2], x.shape[-1]                         # :27
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True                 # :29-30
    b = x.shape[0]                                           # :31
    aligned_mn = get_tma_aligned_size(mn, 4)                 # :32
    aligned_k  = align(k, 4)                                 # :33
    padded = torch.zeros((b, aligned_mn, aligned_k), dtype=torch.uint8)  # :34
    padded[:, :mn, :k] = ue8m0_tensor                        # :35
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)  # :36
    transposed = torch.zeros((b, aligned_k // 4, aligned_mn), dtype=torch.int).mT  # :39
    transposed[:, :, :] = padded                             # :40
    aligned_x = transposed[:, :mn, :]                        # :41
    return aligned_x.squeeze(0) if remove_dim else aligned_x # :42
```

逐段解释「为什么」：

1. **`x.view(torch.int) >> 23`（`:24`）**：UE8M0 只存**无符号 8 位指数**。SF 在进入这一步之前已被
   `ceil_to_ue8m0` 向上取到 2 的幂（第 37 篇），尾数恒为 0，所以 fp32 的**最高字节**就是
   `(sign=0, exp 8 位)`，右移 23 位正好取出。若 SF 不是 2 的幂，这里会静默丢掉尾数——这也是
   第 25 篇 device 侧那条「只允许 2 的幂」debug 陷阱的同一件事。
2. **`aligned_mn = get_tma_aligned_size(mn, 4)`（`:32`）、`aligned_k = align(k, 4)`（`:33`）**：
   两个不同的对齐口径。`aligned_k` 是**打包**需要（4 个 UE8M0 凑 1 个 `int32`）；`aligned_mn` 是
   **TMA** 需要（`get_tma_aligned_size`，`utils/math.hpp:23-27`，把长度补到使字节数 16 的倍数）。
3. **`padded.view(-1).view(dtype=torch.int)`（`:36`）**：把 `[b, aligned_mn, aligned_k]` 的 uint8
   按小端 4 字节合成 `int32`，得 `[b, aligned_mn, aligned_k/4]`。注意这里是 **LSB-first**：
   连续 4 个 SF 的第 0 个落在 int32 的最低字节。
4. **`empty(...).mT` 再赋值（`:39-40`）**：把 `[b, aligned_mn, aligned_k/4]` **转置**成
   `[b, aligned_k/4, aligned_mn]`——即最终 tensor 的第 0 维是「packed K」、第 1 维是「MN」，且
   MN 维**连续**（`stride=1`），K 维 stride = `aligned_mn`。这就是 kernel 里 TMA/SF 读取
   期望的 **MN-major + TMA 对齐**布局。

C++ 侧其实**自带了一份同构的 torch 回退** `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch`
（`smxx_layout.hpp:108-131`），逐行与上面等价（同样 `view(kInt32).bitwise_right_shift(23)`、
`aligned_mn = get_tma_aligned_size(mn,4)`、`aligned_k = align(k,4)`）。当输入是 3D 批、或
`(mn*sf_k)%4 != 0` 时，host 会退回这份 torch 实现（`smxx_layout.hpp:152-153` / `:187-188`），
所以「测试 oracle」和「生产回退」是同一条数学。

> 注意测试 oracle 与 C++ 回退的**一个措辞差异**：Python 用 `x.view(torch.int)` 后 `>> 23`；
> C++ 用 `view(torch::kInt32).bitwise_right_shift(23)`。对非负 SF 完全等价。

---

## 3. `test_sf_layout_kernels`：单组 SF 的两条判据（H100 可跑）

函数（`test_layout.py:45-80`）对 `enumerate_sf_layout()` 的每个用例造一份 `fp32_sf`，然后按
`use_ue8m0` 分两路验收：

```python
for mn, k, with_transpose, use_ue8m0, num_groups, gran_k in enumerate_sf_layout():   # :47
    x = torch.randn((num_groups * mn, k), dtype=torch.bfloat16)                       # :48
    x, fp32_sf = per_token_cast_to_fp8(x, use_ue8m0=use_ue8m0, gran_k=gran_k)          # :49
    fp32_sf = fp32_sf if num_groups == 1 else fp32_sf.view(num_groups, mn, -1)          # :50
    fp32_sf = fp32_sf if with_transpose else                                       # :51
        fp32_sf.transpose(-1, -2).contiguous().transpose(-1, -2)
    ...
```

- **操作数来源**：真实 `per_token_cast_to_fp8`（第 37 篇拆过的参考量化器），保证 SF 是「合法
  量化器产物」而不是随手造的 fp32——它已是 2 的幂（UE8M0 前提），也覆盖了 SF 的零填充。
- **`with_transpose`（`:51`）**：控制输入是「K-major 连续」还是「列主序非连续」。这一位直接决定
  下游 kernel 走哪条路（见 §6）。`transpose(-1,-2).contiguous().transpose(-1,-2)` 造出的是
  **列主序、非连续** 张量（存储仍是 `[sf_k, mn]` 连续）。

### 3.1 `use_ue8m0=True` 路（`:54-60`）：三条最严判据

```python
impl, name = get_mn_major_tma_aligned_packed_ue8m0_tensor, 'pack_fp32_into_ue8m0'   # :55
packed_sf     = impl(fp32_sf)                                                       # :56
ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(fp32_sf)    # :57
assert torch.equal(packed_sf, ref_packed_sf)                                        # :58
assert packed_sf.shape == ref_packed_sf.shape                                       # :59
assert all(packed_sf.stride(i) == ref_packed_sf.stride(i) ...)                      # :60
```

三条缺一不可：**值逐位（`:58`）+ shape（`:59`）+ stride（`:60`）**。为什么连 stride 都要卡？
因为 MN-major 布局的「形状」有两种等价写法，但 kernel 的 TMA 描述符是按**具体 stride** 编的
（第 56 篇）：`torch.equal` 只比值，一个「值对但 MN 维不连续」的 tensor 喂给 GEMM 会读错地址。
本专题实测 54 个 UE8M0 用例**逐位 0 不符、shape 0 不符、stride 0 不符**（§6 Part B）。

### 3.2 `use_ue8m0=False` 路（`:61-70`）：布局断言 + 保值

```python
impl, name = get_mn_major_tma_aligned_tensor, 'transpose'                           # :62
transposed_sf = impl(fp32_sf)                                                       # :63
tma_aligned_mn, sf_k = get_tma_aligned_size(mn, fp32_sf.element_size()), ceil_div(k, gran_k)  # :64
if num_groups > 1:
    assert transposed_sf.size(0) == num_groups                                      # :66
    assert transposed_sf.stride(0) == tma_aligned_mn * sf_k                         # :67
assert transposed_sf.shape[-2:] == (mn, sf_k)                                       # :68
assert transposed_sf.stride()[-2:] == (1, tma_aligned_mn)                           # :69
assert torch.equal(fp32_sf, transposed_sf)                                          # :70
```

这一路不做 UE8M0 打包，只做转置，所以判据是「**形状 + stride 精确**」+「**值不变**」：
`torch.equal(fp32_sf, transposed_sf)（:70）` 断言结果与输入**逐位相同**——因为转置只是**换
stride 的视图**，不搬数据（这正是 C++ `smxx_layout.hpp:60-106` 的语义：可能只是一个
`empty_strided` 的别名）。多组的额外断言（`:66-67`）钉住 batch 维的 stride 公式
`tma_aligned_mn * sf_k`。

### 3.3 perf 段的容错（`:72-79`）

```python
try:
    t = bench_kineto(lambda: impl(fp32_sf), name)                                   # :74
except AssertionError:
    t = 0    # Some cases may fallback to PyTorch impl                              # :76-77
```

测试**显式允许**某些用例不发射 CUDA kernel（走 host 早退 / torch copy 回退），此时
`bench_kineto` 找不到同名 kernel 会抛 `AssertionError`，测试把它吞成 `t=0`。这正是 §6 Part F
要区分的东西：**同一份 `impl` 调用，有的用例发射真 kernel，有的根本不发射**。

---

## 4. `test_k_grouped_sf_layout_kernels`：K 轴分组的逐位 oracle（SM100 专属）

这个函数（`:83-115`）开头就是门控：

```python
if get_arch_major() != 10:
    print(' > Skipped (packed k-grouped SF only supported on SM100)')
    return                                                                          # :85-87
```

H100(sm90) 上它**直接返回**（本专题 Part E 实测打印 `Skipped`）。它的判据构造方式很巧：
**每个分组单独套用同一份单组 oracle**（`:102-104`）：

```python
for group_idx in range(num_groups):
    ref_packed_sf = get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
                        split_fp32_sf[group_idx].T).T                               # :103
    assert torch.equal(split_packed_sf[group_idx], ref_packed_sf)                   # :104
```

即：先把 `fp32_sf` 沿 K 维按各组 `sf_ks` 切开，每组**转置后**喂给单组 oracle，输出**再转置回来**。
这印证了第 26 篇的核心结论——「**K-grouped 就是每个组各自做一份单组变换，然后沿 K 拼起来**」，
组与组之间**独立对齐补零**（`:96-97` 注释）。尾块零填充也被逐位检查：

```python
if sf_ks[group_idx] % 4:
    packed_bytes = split_packed_sf[group_idx][-1].view(torch.uint8).view(mn, 4)     # :106
    assert torch.count_nonzero(packed_bytes[:, sf_ks[group_idx] % 4:]) == 0          # :107
```

每一组的**最后一个 packed 行**里，超出本组 K 跨度的那几列（`sf_ks%4` 之后）必须全零——
这就是「每组独立补零」的直接证据。

---

## 5. `test_k_grouped_psum_sf_layout_kernels`：PSUM 的 K 语义（SM100 专属）

同样有 arch 门控（`:120-122`）。它的关键差异在 K 的两个口径（`:124-130`）：

```python
for mn, real_ks_cpu, aligned_ks_cpu, psum_layout, num_groups, gran_k, k_alignment \
        in enumerate_k_grouped_psum_sf_layout():                                    # :124
    sf_ks        = [align(k, k_alignment) // gran_k for k in real_ks_cpu]            # :127
    packed_sf_ks = [ceil_div(sf_k, 4) for sf_k in sf_ks]                            # :128
    ...
    packed_sf = get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(
        fp32_sf, grouped_layout, aligned_ks_cpu, gran_k, k_alignment,
        use_psum_layout=True)                                                        # :138
```

- `real_ks_cpu` 是**逻辑 K**（未对齐，`enumerate` 里故意减去 `gran_k//2`，见
  `generators.py:305`）；
- `aligned_ks_cpu = align(real_k, k_alignment)` 是**物理 K**（每组必须对齐到 `k_alignment`）；
- PSUM 布局里存的是**逻辑 K**，kernel 自己按 `k_alignment` 补到物理 K（第 26 篇
  `smxx_layout.cuh:246` 的 `align(prev, k_alignment)`）。

判据仍是对每组套单组 oracle（`:131-134`），并额外验证**未同步 / 空 `ks_cpu`** 时分配的上界安全
（`:142-147`）——这正是第 26 篇讲的那条 `packed_sf_k = (sf_k + num_groups*3)/4` 上界。

---

## 6. H100 实测

脚本 `harness/exp_sf_layout_test_oracle.py` 直接 `import test_layout` 复用它的 oracle 与测试体，
在 `kernel_lab`（H100 80GB HBM3, sm90）上跑出 **8 PASS / 0 FAIL**：

### Part A：顶层结构（真机 introspection）

- `test_layout` 有 3 个 `test_*` 函数；`__main__`（`:157-163`）顺序调用三者，先
  `test_sf_layout_kernels()` 后两个 k-grouped；
- 两个 k-grouped 函数**裸露在 `__main__` 里**（无装饰器），靠函数内部 `get_arch_major()!=10`
  自己跳过——H100 上它们打印 `Skipped` 后即返回。

### Part B：oracle vs 真实 kernel（全枚举逐位对拍）

`enumerate_sf_layout()` 在本机产出 **108 个用例**（其中 UE8M0 54 个），对每个 UE8M0 用例做
`torch.equal` + shape + stride 三重对拍：**逐位 0 不符、shape 0 不符、stride 0 不符**。这就是
「测试自带 oracle 与生产 kernel 完全一致」的硬证据。

### Part C：两条 kernel 路径互证 + 负样本控制

对同一个 `mn=4097, k=7168` 的 SF：

- 「已 MN-major」输入 → 走 `pack_fp32_into_ue8m0` 快路（`smxx_layout.hpp:185-214`）；
- 「未 MN-major」输入 → 走 `transpose_and_pack_fp32_into_ue8m0`（`smxx_layout.hpp:163-173`）；
- 两者输出**与同一 oracle 逐位相等**（`True`/`True`）。

负样本：把结果里任一 packed `int32` 字加 1，`torch.equal` 立即变 `False`——坐实这把尺子抓得住
1 个字的扰动。

### Part D：真实测试体

直接调用 `test_layout.test_sf_layout_kernels()`（测试体一行未改），断言全过、无异常。

### Part E：sm90 门控 + host 契约

- 两个 k-grouped 函数在 H100 上打印 `Skipped`；
- **即便绕过测试直接调 API**，`get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor` 也在
  `smxx_layout.hpp:224` 的 `DG_HOST_ASSERT(jit->device.get_arch_major() == 10 ...)` 处被拒：

  ```text
  RuntimeError: Assertion error (.../smxx_layout.hpp:224):
      jit->device.get_arch_major() == 10 and (gran_k == 32 or gran_k == 128)
  ```

  这就是「架构闸门：H100 上这条路径根本进不去」的最后一道保证。

### Part F：profiler 区分「发射 kernel」vs「host 早退 / torch 回退」

用 `torch.profiler` 抓真实 kernel 名，逐用例给出**同一 `impl` 调用实际走了哪条路**：

| 用例特征 | 实际发射的 kernel |
|---------|------------------|
| `use_ue8m0=1, mn=4097, transpose=1` | `transpose_and_pack_fp32_into_ue8m0<512u,48u,56u,128u,1u,false>` |
| `use_ue8m0=1, mn=4096, transpose=0`（已 MN-major） | `pack_fp32_into_ue8m0<1u,512u,128u,16u,true,false>` |
| `use_ue8m0=1, mn=8192, k=128, transpose=1` | `transpose_and_pack_fp32_into_ue8m0<512u,48u,1u,128u,1u,false>` |
| `use_ue8m0=0, mn=4097, transpose=1`（K-major 连续） | `transpose_fp32<512u,64u,56u,128u,57u>` |
| `use_ue8m0=0, mn=4097, transpose=0`（列主序非连续） | `{}`（`out.copy_` torch 回退，`smxx_layout.hpp:71-74`） |
| `use_ue8m0=0, mn=4096, transpose=0`（已列主序） | `{}`（`smxx_layout.hpp:64-65` host 早退） |

**这是只看测试源码绝对看不出的一条事实**：`transpose_fp32` 只在「输入连续但非列主序」时才
发射；输入若已是列主序（非连续）则退回 `out.copy_`（第 25 篇提到过，但这里用 profiler 坐实）。
`use_ue8m0=1` 的情况相反：因为它必然要重新打包，所以总能发射 kernel——要么 `transpose_and_pack`
（未 MN-major），要么 `pack_fp32_into_ue8m0`（已 MN-major）。

### Part G：枚举计数 + 真实吞吐

| 枚举器 | arch9（H100） | 假 arch10 |
|--------|:--:|:--:|
| `enumerate_sf_layout` | **108** | 218 |
| `enumerate_k_grouped_sf_layout` | 48 | 48 |
| `enumerate_k_grouped_psum_sf_layout` | 48 | 48 |

`enumerate_sf_layout` 的差异来自 `generators.py:276-277`（arch9 只有 `gran_k=128`，arch10 多
`gran_k=32`）与 `:287-290`（arch10 额外两组 `sf_k=908/1211`）。两个 k-grouped 枚举器不含
arch 分支（都由函数内部门控拦掉）。

真实 `bench_kineto` 吞吐（透明标注 0us 的「无 kernel」用例）：`mn=4096,k=7168` 约 3us、
`mn=8192,k=7168,ng=4` 约 8us；`transpose=0,use_ue8m0=0` 两个用例 0us（Part F 已解释）。

---

## 7. 小结与边界

- `tests/test_layout.py` 是 SF 布局变换的**唯一可执行规格**。它的权威是**两份**纯 torch 参考实现
  的组合：单组用 `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl`（`:20-42`），
  K 分组用「每组切出来套单组 oracle 再拼回」（`:102-104`）。
- 判据分两档：**UE8M0 打包路**卡「值逐位 + shape + stride」，**转置路**卡「shape + stride + 值不变」。
  三者的**根因**都是「布局是 kernel 描述符的物理契约，错一个 stride 不会崩只会算错」。
- H100(sm90) 真机跑通 `test_sf_layout_kernels`（108 用例 / 54 UE8M0 逐位 0 不符），
  两个 k-grouped 函数与对应 host API 均被 arch 闸门拒（`smxx_layout.hpp:224`）。
- 唯一「仅分析」的部分是 `test_k_grouped_*` 的 kernel 本体与其 PSUM 分配上界——SM100 专属，
  但其数学已在第 26 篇用**逐行抽取的真实设备代码**在 H100 上验证过，本篇的 host 契约与枚举计数
  也已在 H100 上真跑。

---

### 参考

- DeepGEMM（`057ca59`）：`tests/test_layout.py`、`tests/generators.py:276-308`、
  `deep_gemm/utils/layout.py`、`csrc/apis/layout.hpp`、`csrc/jit_kernels/impls/smxx_layout.hpp`。
- 本专题：`docs/25-sf-layout-transform.md`（单组 SF 实现）、`docs/26-k-grouped-sf-packing.md`
  （K 分组实现）、`docs/37-deepgemm-reference-quantizer.md`（`ceil_to_ue8m0` 与
  `per_token_cast_to_fp8`）、`docs/56-tma-descriptor-layer.md`（TMA 对齐口径）、
  `docs/41-deepgemm-test-generators.md`（`enumerate_sf_layout` 的全局 `mk_alignment` 副作用）。
- 实测：`code/mega-moe/harness/exp_sf_layout_test_oracle.{sh,py,out.txt}`。
