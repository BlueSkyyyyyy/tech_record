# 187. FP8 K-grouped wgrad 的 SF 链路端到端：布局 → SF → kernel（H100 兑现）

> 第 12 篇（`docs/12`）拆了 SM90 FP8 `k_grouped_nt` 的 kernel/API，第 16 篇（`docs/26`）拆了
> K-grouped SF 的**打包**（SM100-only），第 186 篇（`docs/186`）用 BF16 MN-major 同构体证明了目标卡
> FP8 `k_grouped_tn` 的**累积结构**，并把「tn 的 SFA/SFB 也要走同一套 `transform_k_grouped_sf_into_required_layout`」
> 留给第 16 篇——**这条链一直没被真正串起来**：SF 从哪来、被变换成什么形状、kernel 侧怎么按组取址、
> 目标卡（SM100 packed）与 H100（SM90 fp32）两条通路差在哪、SF 与操作数主序是什么关系。
>
> 本篇把「布局 → SF → kernel」端到端接上，并在 H100 上真跑一个**完整链路**（量化 → SF → 变换 → kernel），
> 回答一个具体问题——**目标卡 tn 路径的 SF 需求，和本机能跑的 SM90 nt 路径，几何上是不是同一件事？**

答案：几何同构。K-grouped SF 的**分组几何**（每组行数、组间偏移、一行覆盖多少 K）是跨架构不变量，
只有「存储形态」不同（SM90 存 fp32、SM100 打包成 UE8M0 的 int32）；而 SF 的**输入朝向**则跟随操作数
主序（K-major ⇒ `[mn, sf_k]`，MN-major ⇒ `[sf_k, mn]`），两条变换都把它归一成 **MN 连续**输出。

参考版本：`~/github/DeepGEMM` commit `057ca59`（v2.8.1）；本机 H100(sm90)、DeepGEMM 2.8.1、CUDA 13.2。
实测脚本：`harness/exp_kgrouped_sf_chain.{py,sh}`，原始输出 `harness/exp_kgrouped_sf_chain.out.txt`。

---

## 0. 结论先行

1. **三条 K-grouped SF 通路**（`transform_k_grouped_sf_into_required_layout`，`csrc/apis/layout.hpp:89-116`）：

   | 输入 SF dtype | 架构 | 变换 | 输出 |
   |---|---|---|---|
   | `float`（per-token 1D fp32） | **SM90** | `get_mn_major_tma_aligned_tensor`（`:104-105`） | 同逻辑形状，**MN 连续**、K stride=`align(mn,4)` |
   | `float` | **SM100** | `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`（`:108-109`） | `[packed_sf_k, mn]`，packed UE8M0 int32 |
   | `int`（已打包） | SM100（gran_k=32） | `check_k_grouped_packed_ue8m0_tensor`（`:112-113`） | 仅校验 |

2. **SF 输入朝向跟随操作数主序**（`tests/generators.py:488-540` 的 `k_grouped_cast_fp8_fp4_with_major`）：
   K-major（`nt`，SM90）侧 SF 是 `[mn, Σ sf_k]`、MN-major（`tn`，SM100）侧是 `[Σ k/gran_k, mn]`。
   两种输入被各自变换**归一成 MN 连续**，kernel 统一按 `Major::MN` 的 SF TMA 描述符读取（`sm100_fp8_fp4_gemm_1d1d.hpp:377-380`）。

3. **分组几何是跨架构不变量**（本篇核心）：

   - 每组 SF 行数由 scheduler 累加给出：`current_sf_k_cumsum += ceil_div(align(shape_k, KALIGN), kSFKSpan)`
     （`scheduler/gemm.cuh:242`）。
   - SM90 fp32 路径 `kSFKSpan=128`（`sm90_fp8_gemm_1d1d.cuh:161`，一行 fp32 覆盖 128 个 K）；
     SM100 packed 路径 `kSFKSpan=kGranKA*4`（`sm100_fp8_fp4_gemm_1d1d.cuh:199`，一行 int32 = 4 个 scale × `gran_k`）。
   - 所以 **gran_k=32 时，SM100 packed 的每组行数 == SM90 fp32 的每组行数**（都是 `host_k/128`）——
     目标卡 tn 与 H100 nt 的 SF 几何是**同一张表**。

4. **H100 真机**：完整链路（真实参考量化器 `per_channel_cast_to_fp8` → 真实变换 → 真实 SM90 FP8 kernel）
   走通；显式预变换 SF 与 API 内部变换**逐位相等**；kernel vs「反量化操作数上的精确 GEMM」`~1.1e-4`
   （隔离 kernel 误差），vs bf16 原值 `3.62e-2` = FP8 量化误差。SM100 打包 API 在 H100 被 arch 门控拒
   （`smxx_layout.hpp:224`）。

---

## 1. 为什么需要单独一篇：SF 这条链

K-grouped wgrad（`D[e] = A_e^T B_e`）有两个操作数 A/B，各带一份 block-scale SF。前面各篇分别讲清了
**kernel 的累加结构**（第 12/186 篇）、**SF 怎么打包**（第 16 篇），但没有把「SF 从操作数到 kernel」
这条链讲成一条线：

- SF 的**来源**是什么形状？（参考量化器 `per_channel_cast_to_fp8`，`deep_gemm/utils/math.py:52-59`）
- API 收到 SF 后做了什么变换？两条架构路径（SM90/SM100）分别落到什么形状？
- kernel 侧怎么按组找到某组某段的 SF？跨组偏移公式是什么？
- 目标卡（SM100 packed）与本机（SM90 fp32）的 SF 需求，是不是**同一套几何**？

第 186 篇结尾写了「tn 的 SFA/SFB 也要走同一套变换（`gemm.hpp:363-364`），本篇不重复」——
本篇就是来补这条链的，并给它一个可跑的真机证据。

---

## 2. host：SF 从入口到 kernel（`csrc/apis/gemm.hpp`、`csrc/apis/layout.hpp`）

### 2.1 两个 K-grouped FP8 入口对 SF 的调用

- **tn（MN-major，SM100-only）** `k_grouped_fp8_gemm_tn_contiguous`（`gemm.hpp:320-371`）：
  在 arch 分派内对 A/B 的 SF 各调一次变换（`:363-364`）：
  ```
  const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, k_alignment, use_psum_layout);
  const auto sfb = layout::transform_k_grouped_sf_into_required_layout(b.second, ...);
  ```
  非 arch10 直接 `DG_HOST_UNREACHABLE`（`:369`）——本机 H100 跑不了这条的确切行号。

- **nt（K-major，SM90）** `k_grouped_fp8_gemm_nt_contiguous`（`gemm.hpp:373-427`）：
  也调同一变换，但 `k_alignment` 硬钉 128、`use_psum_layout=false`（`:410-411`）：
  ```
  const auto sfa = layout::transform_k_grouped_sf_into_required_layout(a.second, ks_cpu, grouped_layout, recipe, 128, false);
  ```
  ⇒ **两条入口共用同一个 SF 变换**，这正是第 186 篇「tn 与 nt 结构同构」在 SF 侧的落点。

### 2.2 变换本体：`transform_k_grouped_sf_into_required_layout`（`layout.hpp:89-116`）

- 契约（`:95-101`）：`sf.dim()==2`、`recipe` 必须是 `(1,1,gran_k)`；arch9 只允许 `gran_k==128 且 k_alignment==128`，
  arch10 允许 `gran_k∈{32,128}`。
- **SM90 fp32 分支**（`:104-105`）→ `get_mn_major_tma_aligned_tensor(sf)`（`smxx_layout.hpp:60-106`）。
- **SM100 fp32 分支**（`:108-109`）→ `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`（`smxx_layout.hpp:218-283`）。
- **SM100 已打包分支**（`:112-113`）→ `check_k_grouped_packed_ue8m0_tensor`（`smxx_layout.hpp:286-317`）。

注意：**SM90 分支并不做任何「按组」处理**——它把整份 SF 当成一张 `[mn, sf_k]` 做一次 MN-major 转置。
那么「每组独立对齐」是怎么实现的？答案是**量化阶段就把每组补齐了**：`k_grouped_cast_fp8_fp4_with_major`
（`generators.py:488-540`）对每组单独补零到 `host_k`（`align(logical_k,128)`）再量化，于是每组已经贡献
`host_k/128` 行 SF；kernel scheduler 再按同样公式挪组偏移。SF 的「分组」由**数据本身**表达，而不是变换函数。

---

## 3. SF 几何：一行覆盖多少 K、组偏移怎么走

### 3.1 scheduler：SF 组偏移（`scheduler/gemm.cuh`）

k-grouped 的取址（`get_global_idx`，`:157-168`）对 `IndexType::SF_K` 返回
`offset = current_sf_k_cumsum`，即 `current_sf_k_cumsum + block_idx*block_size`。每跨一组，累加一次
（`:240-242`）：

```
const auto aligned_shape_k = math::align(current_shape_k, kKAlignment);
current_sf_k_cumsum += math::ceil_div(aligned_shape_k, kSFKSpan);
```

`kSFKSpan`（默认 128，`:37`）注释即「K covered by one k-grouped SF row」。
- SM90 fp32：一行 FP32 覆盖 `gran_k=128` 个 K ⇒ `kSFKSpan=128`（`sm90_fp8_gemm_1d1d.cuh:161` 传 `128u,128u`）。
- SM100 packed：一行 int32 打包 4 个 UE8M0 scale ⇒ 覆盖 `gran_k*4` 个 K，故 kernel 明确传 `kGranKA*4`
  （`sm100_fp8_fp4_gemm_1d1d.cuh:199`），且 `shape_sfa_k = ceil_div(shape_k, kGranKA*4)`（`:155`）。

### 3.2 packed 行数公式（`smxx_layout.hpp:218-283`）

SM100 packed 的每一组行数（`:244-246`）：`group_sf_k = k/gran_k`、`packed_sf_k += ceil_div(group_sf_k, 4)`，
并把 `ref_sf_k == sf_k` 作为输入契约（`:248`）。对 gran_k=32：`ceil_div(host_k/32, 4) = host_k/128`
——**与 SM90 fp32 的每组 `host_k/128` 完全相同**。gran_k=128 时每组按 `128*4=512` K 向上取整，尾块会多出
额外 padding 行（本例 `host_ks=[512,384,768,256]` 的 packed 行数为 `[1,1,2,1]`，合计 5 行覆盖 2560 K > 1920）。

### 3.3 一条更容易被忽略的事实：不会有「纯 padding 的 SF 行」

因为 `host_k = align(logical_k, 128)` ⇒ `host_k - logical_k < 128`，每组的 padding 行**总与真实行同处
最后一个 K-block 内**。所以该 block 的 SF 由真实行的 `amax` 决定，padding 行反量化恒为 0；不存在
「一整个 K-block 全是 padding」的 SF 行。（H100 实测 `a_sf` 全为正、`min=3.261e-03`。）这也解释了为什么
SM90 变换可以「不分组」——padding 从不出现在 SF 的组边界之外。

---

## 4. H100 实测：布局 → SF → kernel 端到端

`harness/exp_kgrouped_sf_chain.py`（H100 sm90，`G=4, m=512, n=1024`，`logical_ks=[500,300,700,250]`
故意取**未对齐** K 以逼出 padding，`host_ks=[512,384,768,256]`、`sum_k=1920`）。

### A. 量化 → SF → transform → kernel（真实 FP8 nt）

- 物理布局 A/B = `[sum_k, M]` 的 K-major（K 连续），展开 `numel=983040`。
- 输入 SF `a_sf=(512, 15)`（`[mn, Σ sf_k]`，`Σ sf_k=15`），SM90 nt 侧约定。
- 显式变换后 `a_sf_t=(512, 15)`、`stride=(1, 512)`——**MN 连续、MN TMA 对齐 stride=512=`align(m,4)`**。
- **再次变换早退**（`smxx_layout.hpp:63-65` 的列主序早退）：stride 相同、值逐位相同。
- **隐式（API 内部变换）vs 显式（预变换 SF）逐位相等**（bit-diff=0.0000）⇒ 内部变换就是这条链路、且确定。
- 逐组误差（kernel vs 反量化操作数上的精确 GEMM）：

  | group | 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | logical_k | 500 | 300 | 700 | 250 |
  | kernel vs 反量化参考 | 1.158e-4 | 1.058e-4 | 1.164e-4 | 1.100e-4 |
  | kernel vs bf16 原值 | 3.636e-2 | 3.598e-2 | 3.628e-2 | 3.629e-2 |

  即：kernel 在反量化操作数上是精确的（~1e-4 来自 fp32 求和），对 bf16 原值的差**全部来自 FP8 量化**
  （`量化本身 vs bf16 原值` 与 `kernel vs bf16 原值` 逐组几乎相同）。

### B. SF 几何

```
group 0: host_k= 512  sf_rows=4  sf_offset=0
group 1: host_k= 384  sf_rows=3  sf_offset=4
group 2: host_k= 768  sf_rows=6  sf_offset=7
group 3: host_k= 256  sf_rows=2  sf_offset=13
累计 SF 行数=15 == a_sf.shape[1]=15  => True
```

逐组行数/偏移与 scheduler 公式（`scheduler/gemm.cuh:242`，`kSFKSpan=128`）**逐项一致**。

### C. 目标卡（SM100）tn 的 SF 需求

- packed 行数（`smxx_layout.hpp:244-246`）：gran_k=32 → `[4,3,6,2]`、总计 **15**（**== SM90 fp32 行数**）；
  gran_k=128 → `[1,1,2,1]`、总计 5（每组按 512K 向上取整）。一行分别覆盖 **128 / 512** 个 K。
- 一行覆盖的 K 数 = `kSFKSpan = gran_k*4`（kernel `sm100_fp8_fp4_gemm_1d1d.cuh:199/155`）。
- **H100 上调用 SM100-only 打包 API 被拒**（`get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`）：
  ```
  RuntimeError: Assertion error (csrc/apis/../jit_kernels/impls/smxx_layout.hpp:224):
                jit->device.get_arch_major() == 10 and (gran_k == 32 or gran_k == 128) and k_ali...
  ```
  ⇒ 打包的**执行**是 SM100 专属，但其**几何**是本机可复算、且与 SM90 路径同构（§3）。

### D. FP8 nt（K-major + SF）vs BF16 tn（MN-major，无 SF）同题对拍

| 路径 | vs fp64(bf16 原值) |
|---|---|
| FP8 nt（K-major + SF） | 3.624e-2 |
| BF16 tn（MN-major，无 SF） | 4.049e-07 |
| FP8 nt vs BF16 tn | 3.624e-2（bit-diff=1.0000） |

两条物理布局、同一数学 `D[e]=A_e^T B_e`、同一输出 `[G,m,n]`，差异**只在精度层**（FP8 量化 vs BF16），
不在布局或累加结构——这正是第 186 篇结论的 SF 侧补充。

### E. 吞吐（仅量级参考；8 卡共享节点）

| 路径 | t | TFLOPS |
|---|---|---|
| FP8 nt（SF 已 TMA 对齐 → 内部早退） | 11546 us | 167 |
| FP8 nt（SF 原始 → 每次内部 transpose） | 11571 us | 166 |
| BF16 tn（无 SF） | 17331 us | 111 |

`G=8, m=4096, n=7168, k/group=4096`。SF 变换开销可忽略（167 vs 166）；两路径都远低于第 12 篇空闲 GPU 上
FP8 nt 的 425 TFLOPS，本机为共享节点、**只作量级参考**。

---

## 5. 与前面各篇的关系与边界

- **第 12 篇**（FP8 nt wgrad）：本篇是它的「SF 侧」——把 `a.second/sf` 从哪来、怎么变换、怎么按组取址讲完。
- **第 16 篇**（K-grouped SF 打包）：本篇不重复打包 device 代码，只把**打包前后的几何**与 SM90 路径对齐，
  并补上「SF 输入朝向随主序、变换后归一成 MN 连续」。
- **第 186 篇**（tn 累积结构）：本篇补上它明确留给第 16 篇的 SF 链路，并用真机确认「两条入口共用同一变换」。
- **第 91 篇**（SF 官方契约对照）：本篇的 `check_k_grouped_packed_ue8m0_tensor` 契约与它一致。

**边界**：① SM100 packed 变换的**执行**（`pack_fp32_into_ue8m0` device 代码）SM100 专属、本机仅调用被拒，
其 device 级正确性见第 16/78 篇；② 本机 FP8 tn 主 kernel 跑不了，本篇用 SM90 nt + BF16 tn 两条真机路径
交叉覆盖；③ 吞吐数字为共享节点量级。

---

## 6. 参考

- DeepGEMM 仓库（v2.8.1，commit `057ca59`）：<https://github.com/deepseek-ai/DeepGEMM>
- K-grouped host 入口：`csrc/apis/gemm.hpp:320-371`（tn）、`:373-427`（nt）、`:676-722`（bf16 tn）
- K-grouped SF 变换：`csrc/apis/layout.hpp:89-116`；device/host 实现 `csrc/jit_kernels/impls/smxx_layout.hpp:218-317`
- SF 几何：`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:37/157-168/240-242`、
  `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:161`、
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:155/199`
- K-grouped jit：`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325-403`、
  `csrc/jit_kernels/impls/sm90_bf16_gemm.hpp:264-330`
- 参考量化器：`deep_gemm/utils/math.py:52-59`；测试生成器：`tests/generators.py:488-540`、`:551-608`
- 官方测试：`tests/test_bf16.py:158-214`、`tests/test_fp8_fp4.py`、`tests/test_layout.py`
