# DeepGEMM 块级 scale 布局转换：`transform_sf_into_required_layout` 与 UE8M0 打包

> 本文对应路线图 **P15-1**。前面第 10、12、13、21 篇都反复出现「SF（scale factor）要先变成
> MN-major + TMA 对齐」这句话，但一直没把这件**贯穿所有 FP8/FP4 kernel 的前置步骤**单独讲清楚。
> 本文补上：DeepGEMM 里所有带 block-scale 的 GEMM，输入都是「每 `gran_mn × gran_k` 块一个 FP32
> scale」，可裸 scale 是 `[mn, ceil(k/gran_k)]` 行主序；而 kernel 的 TMA 描述符要求 scale
> **MN-major（MN 维 stride=1）且每行 16B 对齐**，SM100 更进一步要求把 scale 量化成 **UE8M0**
> （只留 fp32 的 8 位指数）并按 4 个打包进一个 int32。这些都在 host 侧
> `transform_sf_into_required_layout` 里完成，GPU 侧由 `impls/smxx_layout.cuh` 的三个 kernel
> 承载。**其中转置与打包 kernel 都不是 SM100 专属**，本机 H100(sm90) 可编译可运行，
> 本文用真实设备代码逐条验证语义（`harness/exp_sf_layout.{py,sh,out.txt}`，ALL PASS）。

参考实现：`~/github/DeepGEMM`，公开版 commit `057ca59`（版本 2.8.1）。

## 1. 为什么需要这一步

DeepGEMM 的 GEMM 主体用 TMA（Tensor Memory Accelerator）把操作数从 global 搬进 shared memory，
TMA 描述符对「一行」的**起点对齐**有硬要求（16 字节），且为了 coalesce，希望**变化最快的维是 MN**。
于是 block-scale 的数据布局不能随手写成 `[mn, k_scale]` 行主序，必须：

1. **转置成 MN-major**：让同一 MN 位置在不同 k_scale 上连续，行距（pitch）为 `tma_aligned_mn`；
2. **按 16B 对齐行距**：`tma_aligned_mn = align(mn, 16/elem_size)`；
3. **SM100 额外 UE8M0 + 打包**：UMMA 的 block-scaled MMA 只吃 8 位指数 scale，
   且一次消费 4 个相邻 k_scale（打包进一个 int32）。

这些变换**不是可选的装饰**：`fp8_gemm_nt` 等公开入口内部第一件事就是调用
`layout::transform_sf_pair_into_required_layout`（`csrc/apis/gemm.hpp:113-114`、`:231-232`），
用户传裸 scale 即可。理解它才能理解第 10/12/13/21 篇里那些 SF 的 stride 和逐 batch 基址。

## 2. host 分派总览（`csrc/apis/layout.hpp:11-58`）

`transform_sf_into_required_layout(sf, mn, k, recipe, num_groups, is_sfa, disable_ue8m0_cast, psum_layout)`
先解析 recipe，再做形状检查，最后按 (dtype, gran, arch) 分派。四条可达分支：

| 行号 | 条件 | 动作 |
|---|---|---|
| `:38-39` | FP32、`gran_mn==1 & gran_k==128`、`arch==9 或 disable_ue8m0_cast` | `get_mn_major_tma_aligned_tensor`（转置） |
| `:42-43` | FP32、`gran_mn==128 & gran_k==128`、`arch==9 或 disable_ue8m0_cast` | 只 `check_sf_layout`（128×128 已是列主序友好，**不转置**） |
| `:46-51` | FP32、`gran_k∈{32,128}`、`arch==10` | 广播到 `gran_mn` 行后 `get_mn_major_tma_aligned_packed_ue8m0_tensor`（转置+打包） |
| `:54-55` | INT、`gran_mn==1`、`gran_k∈{32,128}`、`arch==10` | 已打包，只 `check_sf_layout` |

其余落到 `:57` `DG_HOST_UNREACHABLE("Unknown SF transformation")`。

**recipe 的两种形态**（`layout.hpp:23-32`）是这一节最容易记反的点：

- **2-tuple `(gran_mn, gran_k)`**：断言 **`not is_sfa`**（`:28`）——单边显式变换，`is_sfa` 必须留空；
- **3-tuple `(gran_mn_a, gran_mn_b, gran_k)`**：断言 **`is_sfa`**（`:24`）——成对变换，
  由 `is_sfa=True/False` 决定取 A 还是 B 的 `gran_mn`。

`transform_sf_pair_into_required_layout`（`:60-87`）就是按这条约定，对 SFA 传 `is_sfa=true`、
SFB 传 `false`（`:80-83`）。`get_default_recipe`（`csrc/utils/layout.hpp:86-98`）给出默认：
SM90 唯一默认 `(1,128,128)`；SM100 看 SFB dtype——FP32 走 legacy `(1,128,128)`、INT 走 1D1D `(1,1,128)`。

## 3. SM90 可达路径一：`get_mn_major_tma_aligned_tensor`（转置）

host 编排 `csrc/jit_kernels/impls/smxx_layout.hpp:60-106`：

- `:64` **幂等早退**：若已是列主序 TMA 对齐（`stride(1)==1 and stride(2)==tma_aligned_mn`）直接返回原对象；
- `:67-69` 否则开一个 `empty_strided({B, mn, sf_k}, {tma_aligned_mn*sf_k, 1, tma_aligned_mn})`，
  即逻辑形状不变、物理改为 MN-major + pitch `tma_aligned_mn`；
- `:76-80` 转置 kernel 配置 `block_mn=64, block_sf_k=128, 512 线程`。

device kernel `transpose_fp32`（`smxx_layout.cuh:10-72`）：`extern __shared__` 分块中转，
`:36` 先 `cudaGridDependencySynchronize()`（PDL，等上游量化 kernel 写完 scale），再 load→smem→store。
store（`:70`）就是 `out[global_sf_k * tma_aligned_mn + global_mn] = smem[mn * kSmemSFK + sf_k]`——
一次显式转置。`kSmemSFK` 有 `+1` 或 padding 以免 smem bank 冲突（`:16`、`:79-80`）。

## 4. SM90 可达路径二：UE8M0 打包

UE8M0 = 8 位指数、0 位尾数的 8 位浮点码。打包 kernel 只搬 fp32 的**指数字节**，
并 4 个 LSB-first 塞进一个 int32。host 编排 `smxx_layout.hpp:133-216`：

- `:136-139` 输出 `empty_strided({B, mn, ceil(sf_k/4)}, {…, 1, tma_aligned_mn})`，int32；
- `:151-184` **行主序输入**（需要转置）走 `transpose_and_pack_fp32_into_ue8m0`；
- `:185-213` **已 MN-major 输入**（`stride(1)==1 and stride(2)==mn`，`:189`）走 `pack_fp32_into_ue8m0`（只打包不转置）；
- 两条路在 `mn % 4 != 0` 或 batch>1 时回退到纯 torch 实现 `get_mn_major_tma_aligned_packed_ue8m0_tensor_torch`（`:108-131`）。

device `transpose_and_pack_fp32_into_ue8m0`（`smxx_layout.cuh:76-191`）：
`:99` PDL 同步；`:107-114` 若 PSUM 布局，先把每组的有效 MN 区间算进 smem；
`:161-190` 打包——把 4 个 fp32 的指数拼成一个 int32：

```
packed = (values[0] >> 23) | (values[1] >> 15) | (values[2] >> 7) | (values[3] << 1)
```

byte0=values[0]>>23、byte1=values[1]>>23、byte2=values[2]>>23、byte3=values[3]>>23，
即 4 个 ue8m0 字节 **LSB-first**。`:176` 越界补 0、`:188-189` 只写 `in_bounds_mn` 的行。
注意 `:177-178`（和 `pack_fp32_into_ue8m0` 的 `:279-282`）有一条 debug 陷阱：
`(values[j] & 0x807fffff) == 0`——**只允许符号 0、尾数 0（纯 2 的幂）**，
因为打包会静默丢掉尾数与符号位。所以上游 `per_token_cast_to_fp8` 一定先 `ceil_to_ue8m0`
把 scale 上取整到 2 的幂（`deep_gemm/utils/math.py:13-16`）。

`pack_fp32_into_ue8m0`（`:193-293`）用于「已 MN-major」输入：一个 warp 负责一行 packed K，
`:227-267` 先沿 `grouped_layout` 累加找到本 warp 所属组及其安全 SF 行区间
（k-grouped / PSUM 的核心，`:260-261` 记 padding），`:269-292` 再逐 lane 打包 4 行。

## 5. `check_sf_layout` 契约（`csrc/utils/layout.hpp:101-140`）

无论哪条分支，进来先过 `check_sf_layout`（`layout.hpp:35`）：

- `:115` `sf.dim() == num_groups.has_value() + 2`（有分组则 3D）；
- `:118` `sf.size(-2) == ceil_div(mn, gran_mn)`；
- `:119` `sf.size(-1) == ceil_div(k, gran_k * (dtype==Float ? 1 : 4))`（INT 已打包，K 维除以 4）；
- `:122-130` TMA stride 检查：`stride(-2)==1`（MN 连续）、`stride(-1) >= tma_aligned_mn`、`stride(-1) % 4 == 0`；
- `:133-138` SM90 SFB 专用：必须连续，或转置后连续（`stride(-1)==1 and stride(-2)==size(-1)`，或反之）。

这就是第 12 篇 `k_grouped`、第 21 篇 `bmm` 里那些「A per-token 1D SF / B per-128×128 2D SF」
布局判断的同一套检查。

## 6. PSUM 布局：跳过未初始化的 gap 行

contiguous 分组布局里，每组 token 段要对齐到 `m_alignment`，段尾到下一段首之间是 **gap 行**
（没有真实 token，scale 未初始化）。PSUM 布局把每组的**逻辑结束**作为前缀和 `grouped_layout[g]`，
kernel 端用「gap 不读不写」绕过：

- `transpose_and_pack_fp32_into_ue8m0` `:107-126`：把每组有效 MN 区间 `[align(prev), end)` 预计算进 smem，
  `is_valid_mn` 只对落在某区间内的行放行；
- `:142-147` host 侧要求 PSUM 一定是 2D 连续、`num_sf_batches==1`；
- `:148-149/158` PSUM 下 `m_alignment` 与组数从 host 传入，smem 里多留 `num_psum_groups*2*4` 字节；
- `:188` 只在 `in_bounds_mn`（物理在范围内）时写；gap 行要么保留原值要么被写安全有限码
  （**不能**写 UE8M0 的 `0xff`，那是 NaN，`:187` 注释）。

## 7. H100 实测（`harness/exp_sf_layout.{py,sh,out.txt}`）

全部在 kernel_lab(H100) 真机跑，六组 **ALL PASS**：

- **A `get_tma_aligned_size`**：`align(x, 16/elem_size)` 对 3 种 element_size × 10 个 x 逐点一致；
  `(4095,4)→4096`（fp32 scale 行距）、`(4095,1)→4096`。
- **B 转置**：4 组 `(mn,sf_k)`（含 `mn=137` 非 4 倍数、`sf_k=9` 奇数）值与输入 `val_diff=0`、
  输出 stride 精确为 `(1, tma_aligned_mn)`、且**幂等**（二次调用返回同一 `data_ptr`）。
- **C UE8M0 打包**：CUDA kernel 与复刻的 torch 参考（`smxx_layout.hpp:108-131`）**逐位一致**；
  例 `out[0,0]=0x848a847c` → 4 个指数字节 `[124,132,138,132]`，正是 `sf[0,0:4]` 的指数（LSB-first）；
  `ceil_to_ue8m0` 结果尾数恒为 0 且 `raw ≤ pow2 < 2·raw`；`mn=137` 走 torch 回退也逐位一致；
  **已 MN-major 输入**走 `pack_fp32_into_ue8m0` 与 `transpose_and_pack` 结果逐位一致（两个 kernel 互证）。
- **D PSUM 布局**：`groups_end=[300,700,1024]`、`m_alignment=128` → 152/1024 行是 gap；
  有效行与无条件打包逐位一致、gap 行与原值不同（确实被跳过）、且无 `0xFF`(NaN) 码。
- **E recipe 分派**：`(1,128)`/`(128,128)` 在 SM90 均 OK（前者转置 stride `(1,512)`，
  后者返回原张量 `(2,1)` 不转置）；`(1,32)` 无论 `disable_ue8m0_cast` 与否都
  **`layout.hpp:57 Unknown SF transformation`**（gran 32 是 SM100 专属）；
  2-tuple 带 `is_sfa` / 3-tuple 不带 `is_sfa` 分别命中 `:28`/`:24` 断言——契约记反会被当场拦下。
- **F 端到端**：真跑 `fp8_gemm_nt`（SM90 1D2D），`m=1024 n=512 k=1024` 对 fp32 参考 relerr
  `6.80e-4`（就是 FP8 per-128 量化噪声量级）；把 SF 先显式 `transform_sf_into_required_layout`
  再喂给同一 kernel，输出与喂裸 SF **逐位相等**（因为 host API 内部本就调同一变换、且变换幂等）。

## 8. 与 Mega MoE 的联系

Mega MoE（第 10 篇）里权重 SF 走的是 `transform_weights_for_mega_moe`（`deep_gemm/mega/__init__.py`），
那是**另一条更特殊的路径**：除了这里讲的 MN-major + UE8M0，还要对 L1 权重做 UTCCP 4x32 转置、
按 8 粒度交错 gate/up。但两者共享同一个前提——**block-scale 必须能喂进 TMA + UMMA 的既定布局**。
把本文的通用路径讲清楚后，第 10/12/13/21 篇里那些 SF stride、逐 batch 基址、`gran_k=128`
的约束就不再是散落的细节，而是同一个「scale 布局契约」在不同算子上的投影。

## 9. 线上资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- DeepGEMM README（Grouped GEMM / block-scale 布局说明）：https://github.com/deepseek-ai/DeepGEMM#readme
- 上游 PR #95（Mega MoE）：https://github.com/deepseek-ai/DeepGEMM/pull/95
- 上游 PR #316（Mega MoE EP8 基准）：https://github.com/deepseek-ai/DeepGEMM/pull/316
