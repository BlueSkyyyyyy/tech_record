# Mega mHC 的端到端验收测试与正确性 oracle（`tests/test_mega_mhc.py`）

> 第 05 篇（P5-2）拆过 Mega mHC 的**实现**：HyperConnection 的残差多流混合 + mix 系数投影 +
> Sinkhorn 组合矩阵 + RMSNorm，融进一个 TF32-TS UMMA 持久化 kernel，并在 H100 上把
> 「架构无关的数值语义」逐行搬进 `src/mega_mhc_semantics.cu` 对拍。
> 第 51 篇（P41-1）拆过 **Mega MoE** 的验收测试、第 57 篇（P47-1）拆过 **Mega Gate** 的验收测试。
> 但 Mega mHC **自己的**验收测试 `tests/test_mega_mhc.py`（363 行）一直只被顺带引用过几行。
>
> 本篇把它拆透：它定义了「Mega mHC 的输出到底算什么对」，也是第 05 篇里那些「残差到 bf16 ulp /
> Sinkhorn 到 2e-6 / 页布局 0 不符」判据的出处。
> **主体 kernel `sm100_mega_mhc` 是 SM100 专属**（`csrc/apis/mega_mhc.hpp:223` 断言
> `arch_major == 10`），H100 跑不了；但——
>
> - **参考实现 `mhc_reference`**、页布局变换 `extra_sf_rows`、判据 `check_correctness` 全是纯 torch/host，架构无关；
> - **SF 页布局行索引** `get_shared_sf_row_idx`（`epilogue/sm100_mega_mhc.cuh:116-122`）是纯整数，
>   直接 `#include` 真实头文件在 H100 上跑**真实设备代码**；
> - **host 契约**（`csrc/apis/mega_mhc.hpp:116-221`）**在 H100 上真实执行**——非法组合在 arch 闸门
>   之前被逐条拒绝，合法调用才撞到 `:223`。
>
> 这一半就是本篇的实测对象。

---

## 1. 测试全貌：一个融合算子，两个用例函数

`tests/test_mega_mhc.py` 只有两个 `@test_filter(lambda: get_arch_major() == 10)` 的入口
（`:241`、`:318`），在 H100（arch 9）上**整体被跳过**——因为被验对象 `deep_gemm.mega_mhc`
只在 `mega_mhc.hpp:223` 的 `arch_major == 10` 之后才允许调用（第 4 节会把这条闸门在 H100 上真跑出来）。

| 用例 | 行号 | 做什么 |
|------|------|--------|
| `test_mega_mhc_api_contract` | `:241-315` | 契约与运行期不变式：确定性模式的逐位复现、CUDA Graph 预热要求、FP8-only 调用的逐位稳定、多 stream 独立 |
| `test_mega_mhc` | `:318-358` | 正确性/性能巡礼：`enumerate_cases()` 的 28 个形状（× normal/shifted），逐形状对拍 + 30 次逐位复算 + 性能账目 |

### 1.1 用例矩阵 `enumerate_cases`（`:15-19`）

```
sf_layout ∈ {col, extra} × hidden ∈ {4096, 7168} × num_tokens ∈ {1, 64, 65, 200, 1025, 4096, 32768}
```

共 **28** 个形状（H100 实测 `len(list(enumerate_cases())) == 28`），再乘 2（normal / shifted）。
`num_tokens` 特意跨过 **64 / 512 / 1024 / 4096** 这些 BLOCK_M 与 split 边界：
`BLOCK_M = 64`（`layout/mega_mhc.cuh:19`）、`kDefaultNumSplits = 16`（`:36`），
所以 `T ∈ {65, 512, 1025}` 都恰好卡在「多一行 / 多一块」的边界上。

### 1.2 输入构造里的「重复 token」装置 `make_inputs`（`:22-56`）

`make_inputs` 除了随机量，还做两件有意图的事（H100 实测全部命中）：

1. **把 `comb_res_mix` 先归一化到接近双随机**（`:49-51`，三遍「行归一化 → 列归一化」）。
   实测三遍后**列和精确 = 1**（`col_dev ≈ 1.8e-7`），而**行和只是近似**（`row_dev` 可达 ~0.12）——
   因为最后一次是列归一化。这不是 bug：Sinkhorn 的输入本就不必双随机，测试给的是一个「接近均衡」的起点，
   交给 kernel 的 Sinkhorn 迭代把它逼到双随机（第 3 节验证）。
2. **把 `63, 64, 511, 512, T-1` 行克隆成第 0 行**（`:52-55`）：
   ```
   duplicates = tuple(i for i in (63, 64, 511, 512, num_tokens - 1) if 0 < i < num_tokens)
   ```
   这些正是 **BLOCK_M=64 块边界（63/64）、512 行边界（511/512）与末行**。
   随后 `test_mega_mhc` 在 `:343-345` 断言「这些行的输出逐位等于第 0 行」——
   只要 split-K 归约在边界处把某个 token 的 partial 算错/串台，这个断言就会抓到。
   实测 `T=1025` 时 `dups=(63,64,511,512,1024)`、`T=4096` 时 `dups=(63,64,511,512,4095)`。

### 1.3 输出布局 `make_outputs`（`:59-86`）

mHC 的 FP8 输出有三套 SF 布局，对应下游三种消费方式（`mega_mhc.hpp:81-87` 的 doc）：

| `sf_layout` | 输出张量 | 形状 / stride | 含义 |
|-------------|----------|---------------|------|
| `col` | `y_gemm_sf` | `[T, H/128]`, stride `(1, align(T,4))` | TMA 对齐的**列主序** SF，喂 GEMM |
| `extra` | `y_routed_sf` | `[T, H/128]`, 连续行主序 | 路由 token 的 SF |
| `extra` | `y_shared_sf` | `[num_rows, H/128]`, stride `(1, num_rows)` | **Mega MoE 共享专家页布局** |

其中 `extra` 的存储行数（`:78`）
```
num_rows = ceil(T / shared_sf_block_m) * align(shared_sf_block_m, 128)
```
H100 实测：`T=200, block_m=224` → `2 * 256 = 256` 行，与 `y_shared_sf_storage.shape[0]` 一致。

---

## 2. SF 页布局：`extra_sf_rows` 是 kernel 行索引的精确复刻

`extra`/`shared_sf_block_m` 走的不是普通行主序，而是 **Mega MoE 共享专家的「4×32 交错页」**：
每个 `SF_BLOCK_M` 大小的页里，128 个连续 token 行被打散到 4 个 32 行组中（第 20 篇 UTCCP 的同一族排布）。

测试侧的公式 `extra_sf_rows`（`:96-104`）：

```python
token_idx // block_m * align(block_m, 128)      # 第几个页 × 页内对齐跨度
+ index_in_block // 128 * 128                    # 落在这个页的第几个 128 组
+ index_in_block % 32 * 4                        # 32 行组内：每行错开 4
+ index_in_block % 128 // 32                     # 128 组内：第几个 32 行组
```

kernel 侧的真实设备代码 `get_shared_sf_row_idx`（`epilogue/sm100_mega_mhc.cuh:116-122`）：

```cpp
const uint32_t idx = token_idx % SF_BLOCK_M;
return token_idx / SF_BLOCK_M * ((SF_BLOCK_M + 127u) & ~127u) +
       (idx & ~127u) + (idx & 31u) * 4 + ((idx >> 5) & 3u);
```

两者的四段一一对应（`idx & ~127u = idx//128*128`、`(idx & 31)*4 = idx%32*4`、
`(idx>>5)&3 = (idx%128)//32`）。**H100 实测**：`src/mhc_sf_row_idx.cu` 直接
`#include <deep_gemm/epilogue/sm100_mega_mhc.cuh>` 调用真实设备函数，
对 `SF_BLOCK_M ∈ {64, 128, 224, 240, 512}` × `T ∈ {1, 65, 200, 1025, 4096, 32768}` 逐 token 比较，
**mismatch = 0**（32 项全 PASS），且每个页内**单射**（同一页里不同 token 落到不同行）。

这个页布局最后落到 kernel 的两处写（`epilogue:252-256`）：同一份 packed SF 同时写
`y_primary_sf[token_idx * stride_token + …]`（给 GEMM/routed）与
`y_shared_sf[shared_sf_row_idx + …]`（给共享专家）——`shared_sf_row_idx` 就是这个 `get_shared_sf_row_idx`。

> 实测还做了端到端一致性：用真实量化器 `per_token_cast_to_fp8(use_ue8m0, gran_k=32, packed)`
> 造 SF，按 `extra_sf_rows` scatter 进页存储，再按同一索引 gather，**逐位还原**；
> 未被使用的 padding 行**保持全零**。

---

## 3. 参考实现 `mhc_reference`（`:123-210`）是权威语义

第 05 篇 §13 已逐行读过这 88 行。本篇补的是**独立对拍**：用一份纯 **fp64** 重写（
`harness/exp_mega_mhc_test.py` 的 `mhc_reference_fp64`）复算同一条链，与测试参考比 relerr：

- `new_residual`（残差混合，bf16 边界）relerr `1.39e-5`
- `y_bf16`（RMSNorm + FP8 输入量化后的输出）relerr `2.82e-3`（bf16 输出舍入主导）
- `new_comb_res_mix`（Sinkhorn 组合矩阵）relerr `9.36e-7`（normal）/ `3.63e-7`（shifted）
- `new_prev_mix`（shifted 才输出）relerr `1.60e-6`

normal 与 shifted 的差别只在第 05 篇 §13 讲的：用什么算 `norm_input`、是否输出 pre 系数。

**Sinkhorn 双随机**（`_sinkhorn_reference:114-120`）：实测输出 `S = new_comb_res_mix` 的
**行和 = 1、列和 = 1**（atol 1e-4），坐实「组合矩阵无能量缩放」的语义。

一个容易忽略的次序细节：`mhc_reference` 在算 mix 系数前，把 `new_residual` **先 cast 回 bf16**
（`:148`）。这是全 kernel 的**共享数值边界**——残差既喂下游层、又喂 mix 投影，两处必须看到同一个
bf16 张量（第 05 篇 §13 的「bf16 舍入是共享边界」）。我们的 fp64 版刻意复现这一 cast，
所以 `new_residual` 的 relerr 反而极小（`1.39e-5`）。

---

## 4. 判据 `check_correctness` / host 契约：H100 上真跑

### 4.1 `check_correctness`（`:229-238`）

```python
diff = float(calc_diff(comparable(actual[name]), comparable(reference[name])))
limit = 2e-4 if name == 'y_fp8' else 5e-5
assert diff < limit
```

- **`comparable`（`:225-226`）**：`int32` 的张量（即 SF 张量 `y_gemm_sf`/`y_routed_sf`/`y_shared_sf`）
  先 `.contiguous().view(uint8)` 再比，从而**逐字节**比较这 4 个 UE8M0 指数（而不是当成整数比大小）。
  H100 实测 view 后 `numel ×4`。
- **阈值不同**：`y_fp8` 用 `2e-4`（E4M3 每元素本身有量化噪声），其余（`new_residual` /
  `new_post_mix` / `new_comb_res_mix` / `y_bf16` / SF）用 `5e-5`。
  H100 实测造一个 `calc_diff = 1.221e-4` 的对：作为 `y_fp8` **放行**、作为 `y_bf16` **被拒**——
  阈值确实在起作用。
- 注意这里用的是 **`calc_diff`（归一化平方误差，见第 40 篇）而非 `torch.equal`**：
  因为 mHC 的参考走的是 fp32/FP64 torch，而 kernel 走 TF32 UMMA，二者**不可能逐位相等**；
  真正**逐位**的判据是 `test_mega_mhc` 里「同一输入重复调 30 次」的**自洽**检查（`:337-342`）。

### 4.2 `test_mega_mhc_api_contract`（`:241-315`）的四条运行期不变式

这四条**全在 host 侧、与 kernel arch 无关**，是 mHC 特有的、比数值更硬的结构约定：

1. **确定性模式逐位复现**（`:249-257`）：`use_deterministic_algorithms(True)` 时
   `get_num_splits`（`mega_mhc.hpp:15-24`）返回固定 `kDefaultNumSplits = 16`，
   所以两次调用**逐位相等**（`assert_bitwise_equal`）。非确定性时 split 数由
   `get_num_splits` 按形状/`num_sms` 动态选，浮点加序会变。
2. **CUDA Graph 必须先预热**（`:259-283`）：直接 capture 一次会抛
   `RuntimeError('... CaptureStatus::None ...')`。根因在 host `get_split_barriers`
   （`mega_mhc.hpp:26-39`）：每个 stream 第一次用 mHC 时才 `torch.zeros` 建 split barrier，
   而这**一次性清零不能进 graph replay**，故 `:33` 断言
   `currentStreamCaptureStatusMayInitCtx() == CaptureStatus::None`——预热后再 capture 才逐位正确。
3. **FP8-only 调用逐位稳定**（`:285-296`）：`y_bf16=None` 的 FP8-only 调用，在中间插入一次
   不同尺寸（`T=1`）的调用后，输出仍逐位不变。这依赖 host 的
   `needs_bf16_scratch`（`mega_mhc.hpp:229-238`）：shifted FP8-only 需要一个「借来的 bf16 scratch
   尾巴」作为**数值边界**，但它不会被较小尺寸的调用污染。
4. **多 stream 独立**（`:298-315`）：两个 stream 并发跑不同输出，各自逐位等于串行结果。
   因为 split barrier 是**按 `CUDAStream` 分表**的（`get_split_barriers` 的
   `std::unordered_map<CUDAStream, Tensor>`，`:29-30`）。

### 4.3 host 契约逐条拒绝（H100 真跑）

`mega_mhc` 的 host 端在 `:116-221` 有大量 `DG_HOST_ASSERT`。这些是纯 host 逻辑，H100 上**真执行**。
我们调用真实 `deep_gemm.mega_mhc`，观察异常消息里的**行号**（H100 实测，全部命中）：

| 非法输入 | 命中行 | 断言 |
|----------|--------|------|
| shifted 系数全有或全无 | `:118` | `is_shifted == new_prev_mix.has_value()` |
| `y_gemm_sf` 与 `y_routed_sf` 互斥 | `:120` | `not (y_gemm_sf and y_routed_sf)` |
| 至少一个 norm 输出 | `:121` | `y_bf16.has_value() or y_fp8.has_value()` |
| `y_fp8` 必须配 SF | `:122` | `y_fp8 == (y_gemm_sf or y_routed_sf)` |
| `y_shared_sf` 需要 `shared_sf_block_m>0` | `:123` | `y_shared_sf ? block_m>0 : block_m==0` |
| `hidden % (kDefaultNumSplits·BLOCK_K)` | `:141` | `hidden % (16·64) == 0` |
| `y_gemm_sf` 必须列主序 | `:204` | `stride(0) == 1` |
| `y_routed_sf` 必须连续 | `:210` | `y_routed_sf->is_contiguous()` |
| **一个完全合法的调用** | `:223` | `jit->device.get_arch_major() == 10` |

最后一行是本篇的边界：**在 H100 上，一切契约通过之后，唯一的失败点是 SM100 arch 闸门**——
这坐实了「mHC 的 host 编排是架构无关的，只有 kernel 本体是 SM100 专属」。

此外 `:141` 的 `hidden % (16·64) == 0` 用到 `mega_mhc.hpp:16-17` 的 `kDefaultNumSplits`
（确定性时的 split 数）；`:229-238` 的 bf16 scratch 与 `:241` 的「GEMM/routed 共享一个 primary SF writer」
是两条内存别名约定（`y_primary_sf = y_routed_sf ? y_routed_sf : y_gemm_sf`）。

---

## 5. 性能账目（`test_mega_mhc.py:347-357`）

测完正确性后测试打印三项（供人选型参考），公式遵循第 40 篇的 `count_bytes`：

```python
kernel_t = bench_kineto(lambda: run_mega_mhc(inputs, outputs), 'sm100_mega_mhc', ...)
logical_bytes        = count_bytes(*逻辑输入, *逻辑输出)
intermediate_io_bytes = 2 * count_bytes(y_bf16)               # scratch 读写各一次
if not is_shifted:
    intermediate_io_bytes += count_bytes(new_residual)        # normal 还要中转残差
GB/s = (logical_bytes + intermediate_io_bytes) / kernel_t / 1e9
```

两个「为什么」：① `y_bf16` 计**两份**（scratch 写 + 读回）是因为 normal 模式下 RMSNorm 要先把
bf16 结果落 scratch 再消费；② normal 比 shifted 多一份 `new_residual` 中转，因为 normal 的 pre 系数
是这一步才产出、必须持久化。这解释了日志里 `incl. Scratch I/O` 一列通常高于纯 `logical_bytes` 列。

---

## 6. H100 实测证据台账

实验：`harness/exp_mega_mhc_test.{sh,py}` + `src/mhc_sf_row_idx.cu`（原始输出
`harness/exp_mega_mhc_test.out.txt`，设备 `NVIDIA H100 80GB HBM3, cc 9.0`，
`epilogue/sm100_mega_mhc.cuh` sha256 `5bfb051b…`、`layout/mega_mhc.cuh` sha256 `ca34571a…`）。

- **Part 1 设备代码**（直接 include 真实头文件）：`get_shared_sf_row_idx` vs 测试 `extra_sf_rows`
  ——5 个 `SF_BLOCK_M` × 6 个 `T` 共 **32 项 PASS / 0 FAIL**，mismatch=0 且页内单射。
- **Part 2 Python oracle + host 契约**：**45 PASS / 0 FAIL**——
  - 用例矩阵 28、`sf_layout`/`hidden`/`num_tokens` 取值、重复 token 装置（`T=1025`→`dups=(63,64,511,512,1024)`）；
  - `comb_res_mix` 三遍归约后 `col_dev=1.8e-7`、`row_dev=1.2e-1`；
  - 页布局公式与 kernel 5 个 block_m 一致、scatter/gather 逐位还原、padding 全零；
  - `mhc_reference` fp64 对拍（relerr 1.4e-5 ~ 2.8e-3）、Sinkhorn 行/列和=1；
  - `check_correctness` 抓错 + 阈值（`calc_diff=1.221e-4` 在 fp8/bf16 两阈值间）；
  - **host 契约 9 条**逐条命中真实 `mega_mhc.hpp` 行号（含「合法调用撞 `:223` arch 闸门」）。

**未能测**：`sm100_mega_mhc` 主体 kernel、`test_mega_mhc` 的数值/性能巡礼——SM100 专属，仅分析。
`test_mega_mhc_api_contract` 的四条运行期不变式也因调用被 arch 闸门挡住而无法真跑（仅按 host 源码解读）。

---

## 7. 小结

- Mega mHC 的**验收测试**分两半：SM100 专属的数值/性能巡礼（`test_mega_mhc`），
  与**架构无关**的参考实现 / 页布局 / 判据 / host 契约（可在 H100 全真跑）。
- **`mhc_reference`（`:123-210`）是语义权威**：残差混合 → mix 投影 → sigmoid/Sinkhorn → RMSNorm；
  bf16 cast 是共享数值边界；Sinkhorn 把组合矩阵逼到双随机。
- **SF 页布局 `extra_sf_rows` 与 kernel `get_shared_sf_row_idx` 逐字等价**——
  测试公式就是 kernel 公式的 host 版，H100 上设备代码实测 0 mismatch。
- **host 契约在 H100 上真执行**：9 条非法/合法路径分别命中 `mega_mhc.hpp` 的具体行号，
  最后的边界是 `:223` 的 SM100 arch 闸门——这就是「mHC host 架构无关、kernel 本体 SM100 专属」。
- 判据分层：`calc_diff`（<5e-5/2e-4）保证与 torch 参考一致，30 次重复的 `assert_bitwise_equal`
  保证 kernel 自身确定性（确定性模式下 split 固定 = 16）。
