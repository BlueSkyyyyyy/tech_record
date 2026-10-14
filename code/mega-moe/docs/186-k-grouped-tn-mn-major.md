# 186. MN-major K-grouped wgrad：SM100 FP8 `k_grouped_tn` 生产路径精读（+ H100 可跑对照）

> 前面的反向系列（第 12 篇 wgrad / 第 13 篇 dgrad / 第 14 篇对拍 / 第 15 篇 ncu 归因 /
> 第 174 篇累积结合序 / 第 175 篇跨 rank）用的都是 **SM90 能跑的那条 FP8 `k_grouped_nt`**
> （K-major，`A/B: [M, sum_k]`）。但 **Mega MoE 的 wgrad 在目标卡（SM100）上走的是另一条**——
> `k_grouped_fp8_gemm_tn_contiguous`，物理布局 **MN-major**（`A/B: [sum_k, M]`），本机 H100 **跑不了**
> （host 直接 `DG_HOST_UNREACHABLE`）。第 12 篇只给了它一行「仅分析」，本篇把它拆到底：**从 host 契约
> 到 scheduler 的 K 分组，再到 kernel 里每块独立的 fp32 累加链**，回答一个具体问题——
>
> **第 174/175 篇那些「wgrad 累积序决定逐位结果」的结论，能不能迁移到目标卡这条 tn 路径上？**

答案：能。原因是 tn 与 nt 在**结构上共用同一套 K-grouped 语义**（每组一个独立累加器、组内 K 单调
串行），差异只在「物理布局是 MN 还是 K 连续」与「是否支持 PSUM」。为把这条结论**坐实成真机证据**，
本机用 **BF16 家族的 SM90 MN-major 同构实现**（`k_grouped_bf16_gemm_tn_contiguous`）在 H100 上真跑，
逐位验证「组独立 / 组排列无关 / 逐次确定」。

参考版本：`~/github/DeepGEMM` commit `057ca59`（v2.8.1）；本机 H100(sm90)、DeepGEMM 2.8.1、CUDA 13.2。
实测脚本：`harness/exp_kgrouped_tn_mn_major.{py,sh}`，原始输出
`harness/exp_kgrouped_tn_mn_major.out.txt`。

---

## 0. 结论先行

1. **布局 / 架构矩阵**：wgrad 的 K-grouped 有四种落地形态，两条物理布局 × 两个架构：

   | API | 物理布局 | K 连续轴 | 可跑架构 | 主机入口 |
   |---|---|---|---|---|
   | `k_grouped_fp8_gemm_nt_contiguous` | K-major `[M, sum_k]` | K | **SM90**（本机可跑） | `gemm.hpp:373-427` |
   | `k_grouped_fp8_gemm_tn_contiguous` | MN-major `[sum_k, M]` | M/N | **仅 SM100** | `gemm.hpp:320-371` |
   | `k_grouped_bf16_gemm_tn_contiguous` | MN-major `[sum_k, M]` | M/N | **SM90 + SM100** | `gemm.hpp:676-722` |
   | `k_grouped_fp4_gemm_nt_contiguous` | K-major | K | 仅 SM100 | `gemm.hpp:429` |

   目标卡的生产 wgrad 走第二行（FP8 MN-major）；第三行的 **BF16 版本是本机唯一能真跑的 MN-major 同构体**。

2. **为什么 tn 与 nt 是「同一个算子换布局」**：`gemm.hpp:320-371` 的 tn 与 `:373-427` 的 nt 都满足
   `D[e] = A_e^T B_e`（输出恒为 `[num_groups, m, n]`），只是 A/B 的内存主轴不同。tn 额外支持
   **PSUM 布局与自定义 epilogue、直接输出**；nt 在 SM90 上强制 `c` 必填（累积）且禁 PSUM。

3. **K 累积结构（本篇核心）**：SM100 tn kernel 对**每个输出 tile（每组、每 (m,n) 块）**用一个独立
   累加器；组内 K 按 `k_block_idx`（BLOCK_K=128）**外循环**、`umma_k_idx`（UMMA_K=32）**内循环**
   单调推进，**只有第一条 UMMA 清零、其余全 accumulate**（`sm100_fp8_fp4_gemm_1d1d.cuh:437,441`）。
   ⇒ 每组的 K 折叠加法链是「固定的、单调的、组间不相交的」。

4. **H100 真机证据**（BF16 MN-major 同构 kernel）：全组一次 launch v.s. **逐组单独 launch 逐位相等**；
   组顺序置换后输出按同置换**逐位相等**；同配置 20 次 **0 次 bit 变动**。这三条正是迁移结论的前提。

5. **迁移**：因此第 174 篇（wgrad「切 K / 结合序」改 bit、幅度 ≲7e-8）与第 175 篇（跨 rank 归约）
   的结论**对目标卡的 FP8 tn 路径同样成立**——它们只依赖「每组一条组内单调 fp32 累加链」，而这在
   tn 的 scheduler + kernel 里被明确实现。唯一不可从源码推出的是 **UMMA 与 WGMMA 的硬件 K-内部
   归约序**是否一致（两者都在 K=32 一级做张量核归约），这一格只能等能跑 SM100 的环境。

---

## 1. 为什么需要单独一篇：wgrad 的「另一条腿」

第 12 篇讲了 FP8 wgrad 的 host 入口 `k_grouped_fp8_gemm_nt_contiguous`（`gemm.hpp:373-427`）与
SM90 kernel `sm90_fp8_gemm_1d1d.cuh`，并在结尾标注：

> `k_grouped_fp8_gemm_tn_contiguous`（MN-major，`[sum_k, M]`）为 **SM100 专属，仅分析**。

那行「仅分析」留下了三个没回答的问题：

- **它长什么样**？host 契约、压给 kernel 的 GemmType、TMA 描述符和 nt 差在哪。
- **它的 K 累积和 nt 一样吗**？第 174 篇关于「一次大 GEMM v.s. 切 microbatch 求和改 bit」的结论，
  是在 SM90 nt 上量的；如果 tn 的累积结构不同，结论就不能默认迁移。
- **能不能在 H100 上拿到一点真机证据**？毕竟目标卡跑不了。

本篇把这三点补齐，并且——这是关键——发现 **`k_grouped_bf16_gemm_tn_contiguous` 在 SM90 上就有实现**
（`sm90_bf16_gemm.hpp:264-330`），它的物理布局与 tn **逐位同构**（`[sum_k, M]` MN-major、输出
`[G, m, n]`），于是本机 H100 能真跑一个「MN-major K-grouped」的实物，作为 tn 的可核对对照。

---

## 2. host 契约：tn v.s. nt（`csrc/apis/gemm.hpp`）

### 2.1 tn：MN-major、`recipe=(1,1,gran_k)`、仅 arch10

`k_grouped_fp8_gemm_tn_contiguous`（`gemm.hpp:320-371`）逐条：

- **必须 1D1D**：`recipe[0]==recipe[1]==1`（`:331`）——1D1D 指 A/B 各用 1D tensor map。
- **gran_k ∈ {32,128}**、`k_alignment % 128 == 0`（`:335-338`）；`k_alignment` 由
  `get_mk_alignment_for_contiguous_layout()` 给出（`:336`）。
- **MN-major 物理布局**（`:340-343`）：`a.first: [sum_k, M]`、`b.first: [sum_k, N]`，
  输出 `d: [num_groups, m, n]`（`:341` 的 `get_shape<3>(d)`）。
- **分组语义**（`:345-348`）：注释写得很清楚——**无 PSUM 时 `grouped_layout[i]` 是第 i 组 pad 后的 K**；
  **有 PSUM 时它是「逻辑 K 结束」，组起点是 `align(previous_end, k_alignment)`**。两条路都走同一个
  `check_k_grouped_args`（`:347`）。
- **派发**（`:361-370`）：`arch_major==10` 才继续；先做 SFA/SFB 的 K-grouped 布局转换
  （`:363-364`，即第 16 篇讲的 `transform_k_grouped_sf_into_required_layout`），再调
  `sm100_k_grouped_fp8_gemm_1d1d(..., Major::MN, Major::MN, ..., use_psum_layout, ...)`（`:365-367`）。
  非 arch10 直接 `DG_HOST_UNREACHABLE`（`:369`）——**这就是本机 H100 跑不了 tn 的确切行号**。

### 2.2 nt：K-major、强制累积、SM90

`k_grouped_fp8_gemm_nt_contiguous`（`gemm.hpp:373-427`）：

- `recipe == (1,1,128)` 硬钉（`:384`），且 `epilogue_class == nullptr`（`:385`，自定义 epilogue 需 SM100）。
- **无 PSUM**（`:388`）：`not use_psum_layout and ks_cpu.has_value() and not ks_cpu.value().empty()`。
- 物理布局 `a: [sum_k, M]` 的行数 × M：`sum_mk == sum_k * m`（`:396`）、`sum_nk == sum_k * n`（`:397`）。
  注意这里 `a.first` 是 `[M, sum_k]`（K-major），`numel()==sum_k*m` 对两种布局一样——**形状检查看不出主轴，
  主轴藏在 `is_contiguous()` + TMA 描述符里**。
- `c` 必填且连续（`:403`）——即 **nt 在 SM90 上只能累积**（`D += A^T B`）。
- 派发 arch9 → `sm90_k_grouped_fp8_gemm_1d1d(..., Major::K, Major::K, ...)`（`:421-423`）。

### 2.3 BF16 tn：本机可跑的 MN-major 同构体

`k_grouped_bf16_gemm_tn_contiguous`（`gemm.hpp:676-722`）与 FP8 tn 的物理布局完全一致
（`a: [sum_k, m]`、`b: [sum_k, n]`、`d: [G, m, n]`，`:686-688`；`k_alignment % 128 == 0`，`:690-691`），
但派发覆盖两个架构：

- **arch9**（`:708-714`）：`sm90_bf16_k_grouped_gemm(a, b, c, d, m, n, ks_cpu, grouped_layout,
  Major::MN, Major::MN, ...)`；约束 **`c` 必填**（`:710`）、**禁 PSUM**（`:711`）、禁自定义 epilogue（`:712`）。
- **arch10**（`:715-718`）：`sm100_bf16_k_grouped_gemm(..., use_psum_layout, ...)`，支持 PSUM 与
  epilogue（与 FP8 tn 对齐）。

`sm90_bf16_k_grouped_gemm`（`sm90_bf16_gemm.hpp:264-330`）强制 `Major::MN, Major::MN`（`:272`）、
每组 `k % 128 == 0`（`:274-278`）、`GemmType::KGroupedContiguous`（`:284`），TMA 也用 MN-major
（`:299-313`）。**这就是本机 H100 能真跑的那条 MN-major K-grouped 实物。**

> **为什么这一条能当 tn 的对照**：两者在 host 层「按 K 分组、物理 `[sum_k, MN]`、输出 `[G,m,n]`、
> K 组内连续」的**全部结构约定相同**，只有「SM90/SM100」与「BF16/FP8（有无 SF）」两处不同。
> 结构性结论（组独立、组内 K 单调）因此可迁移；纯数值（SF 量化、MMA 指令）不可迁移。

---

## 3. jit 装配：`sm100_k_grouped_fp8_gemm_1d1d`（`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325-403`）

- **GemmType 二选一**（`:342-344`）：`use_psum_layout ? KGroupedContiguousWithPsumLayout :
  KGroupedContiguous`，`kernel_type = Kernel1D1D`。这就是 scheduler 分支的入口。
- **形状**（`:339-340`）：`sum_k = a.size(0)`（MN-major 的第 0 维就是 K）；`expected_k = ceil(sum_k/G)`
  只用于选核（`:353` 注释「SM100 未用 expected_k」）。
- **TMA 描述符**（`:360-380`）：A/B 都是 `make_tma_a_desc(Major::MN, a, m, sum_k, ...)`（`:360-369`），
  即「先 MN、后 K」的 2D tensor map；CD 是 **3D**（`:370-375`，`make_tma_3d_desc(d, n, m, num_groups, ...)`，
  第三维 = 组号）。SFA/SFB 也走 MN（`:377-380`），K 跨度 `sfa.size(0)*gran_k*4`。
- **launch**（`:383-402`）：`grid_dim = (num_sms,1,1)`、`block_dim`、`cluster_dim`；
  带上 `k_alignment`、`grouped_layout.data_ptr()`、五个 tensor map。

对比 nt 的 SM90 jit（`sm90_fp8_gemm_1d1d.hpp`）：nt 需要按组**原地改写 tensormap**（第 12 篇），
而 tn 因为 A/B 是「先 MN 后 K」的连续 2D + 组只改 K 起点，用 K 偏移即可（见 §4），**不需要 per-group
tensormap 双缓冲**——这也是 host `gemm.hpp:416` 的 `tensor_map_buffer` 只出现在 nt 路径的原因。

---

## 4. scheduler：K 分组怎么切、边界在哪里（`deep_gemm/include/deep_gemm/scheduler/gemm.cuh`）

### 4.1 状态与取组（`:64-82`）

```
uint32_t current_shape_k, current_k_start = 0, current_sf_k_cumsum = 0;  // :66
uint32_t current_k_end = 0;                                              // :68 (PSUM)
```
`get_next_k_group()`（`:71-82`）：

- **PSUM**（`:72-77`）：`grouped_layout[i]` 是逻辑 K 结束 `next_k_end`；组起点
  `current_k_start = align(next_k_end, kKAlignment)`；`current_shape_k = next_k_end - current_k_start`。
- **contiguous**（`:78-81`）：`current_k_start += current_shape_k`；`current_shape_k = grouped_layout[i]`
  （即第 i 组 pad 后的物理 K）。

**为什么 PSUM 要 `align`**：PSUM 允许逻辑组结束落在任意地址，但 A/B 的组起点必须整 BLOCK_K 对齐
（`kKAlignment % BLOCK_K == 0`，静态断言 `:42-43`），所以把物理起点向上对齐、把「逻辑结束」当组边界。

### 4.2 K 取址（`:157-168`）

`get_global_idx<kWithGroupOffset, IndexType::K>` 对 k-grouped 返回
`offset + block_idx * block_size`，其中 `offset = current_k_start`（`:162-163`），
`IndexType::SF_K` 时 `offset = current_sf_k_cumsum`（`:164-165`），`IndexType::MN` 时
`offset = current_group_idx * shape_dim`（`:160-161`）。
⇒ **组内 K 块 = `current_k_start + k_block_idx*BLOCK_K`，物理连续、单调递增。**

### 4.3 块调度：每组独立、逐组推进（`:226-246`）

k-grouped 的 `get_next_block`：

```
num_blocks = num_m_blocks * num_n_blocks;          // :102 每组一套 (m,n) 网格
while (current_group_idx != kNumGroups) {          // :227-230
    if (next_block_idx < (current_group_idx + 1) * num_blocks) break;      // :233 本组内
    current_sf_k_cumsum += ceil_div(align(current_shape_k,kKAlignment), kSFKSpan);  // :242
    get_next_k_group();                                                     // :243
}
get_swizzled_block_idx(next_block_idx - current_group_idx * num_blocks, ...);  // :246
```

含义：**每个组都拥有同一套完整的 (m,n) 块**（因为输出是 `[G,m,n]`，每组各写自己的 D），
块先扫完当前组的所有 (m,n)，再进下一组。⇒ **每个输出 tile 属于唯一一组、只累积那一组的 K**。

> **这就是「组是独立累加器」在源码层的确切落点**：`num_blocks` 每组重复、`current_k_start` 逐组推进、
> 而 kernel 侧每条 (m,n) tile 的 K 循环只覆盖自己组的那段 `current_shape_k`（§5）。组间既不共享
> 累加器、也不共享 K 区间——**改变组的排列或把组拆成独立 launch，不改变任何一条加法链**。

---

## 5. kernel：每块一条单调 fp32 累加链（`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh`）

### 5.1 骨架与约束

- 模板（`:24-37`）：`kMajorA/kMajorB`、`kGranKA/kGranKB`、`kKAlignment`、`SHAPE_*`、`BLOCK_*`、
  `kNumGroups`、`GemmType`、`kWithAccumulation` 等。整段 `#if __CUDA_ARCH__>=1000`（`:47`）——SM100 专属。
- **BLOCK_K**（`:81`）：FP8 路径 `BLOCK_K == 128`（FP4 用 MXF4 时 256）；**UMMA_K**（`:72`）
  FP8=32、FP4=64。故一个 K 块 = 4 条 FP8 UMMA。
- **K-grouped 相关断言**（`:100-104`）：A/B gran_k 必须相等 ∈{32,128}；`kKAlignment % BLOCK_K == 0`。
- scheduler 构造（`:199-200`）传入 `kGemmType, ..., kNumGroups, ..., kKAlignment, kGranKA*4`。

### 5.2 TMA：MN-major 与 K 偏移

A/B 的 TMA（`:280-291`）按 major 分支：MN-major 时 `tma::copy<LOAD_BLOCK_M, BLOCK_K, ...>(..., m_idx,
k_a_idx, ...)`（`:283-285`）；K 偏移 `k_a_idx = get_global_idx<kAWithGroupKOffset, K>(shape_k, BLOCK_K,
k_block_idx, ...)`（`:246-247`），而 `kAWithGroupKOffset = is_k_grouped_contiguous || Major::MN`（`:243`）
——**对 tn，A 和 B 都带组 K 偏移**（因为物理主轴是 MN，K 要靠 `current_k_start` 定位）。

### 5.3 K 循环与累积（`:359-451`）

```
const auto num_total_k_blocks = max(1u, ceil_div(scheduler.current_shape_k, BLOCK_K));   // :359
for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {  // :361
    ...
    for (uint32_t umma_k_idx = 0; umma_k_idx < BLOCK_K / UMMA_K; ++umma_k_idx) {   // :416
        ...
        mma_t::fma(a_desc, b_desc, accum_stage_idx * (...),
                   umma_k_idx > 0 or k_block_idx > 0,          // :441  ← accumulate 标志
                   runtime_instr_desc, tmem_col_sfa, tmem_col_sfb);
    }
}
```

关键在第四个参数（`:437`/`:441`）：**`accumulate = (umma_k_idx > 0) or (k_block_idx > 0)`**。
第一条 UMMA（`k_block_idx==0 && umma_k_idx==0`）**不清零就覆盖**（accumulate=false），其余全部
累加。⇒ 对该输出 tile，结果 = 沿 K **单调**折叠 `(k_block 外、umma_k 内)`，**没有任何乱序或树形归约**。

空组：`is_empty_group = is_k_grouped_contiguous && current_shape_k == 0`（`:526`），epilogue 据此跳过
store（`:530-560`）——符合第 12 篇量到的「空组不产出」。

### 5.4 输出取址

epilogue 里 `kCDWithGroupOffset = not m_grouped && not k_grouped`（`:523`）——**对 k-grouped 为 false**，
即输出的组号**不**编进扁平 M 坐标，而是走 `tensor_map_cd` 的**第三维**（3D TMA，§3）。`current_group_idx`
作为 store 的组索引（`:536`/`:552`）。⇒ 组间输出物理隔离，第 & 组写第 g 张 `[m,n]`。

---

## 6. 推演：为什么 174/175 的结论迁移到 tn

把 §4/§5 合起来，tn 路径对**每个输出元素**定义的是一条确定的 fp32 加法链：

1. 链的**输入**是该组（且仅该组）K 区间 `[current_k_start, current_k_start+current_shape_k)` 内的
   A/B 块，按 K 单调；
2. 链的**粒度**是 K 块 128（4×UMMA K=32）；
3. 链**组间不相交**（每组独立累加器、独立 D 张量）。

第 174 篇量到：把一条 K 加法链**切开再相加**（microbatch 累积）或**改结合序**（forward/reverse/
pairwise-tree）会改 bit，幅度 ≲7e-8。这些操作**只作用于链的内部**——而 tn 拥有和 nt 完全相同的
链结构，所以结论对 tn 同样成立。第 175 篇（跨 rank 归约）发生在链之外（DP 层），更不受影响。

**唯一不可从源码判定的一格**：单条 UMMA（K=32）内部张量核的归约序，是否与 SM90 WGMMA 一致。
两者都在 K=32 一级归约，但硬件实现不同；这一点在 `sm100_fp8_fp4_gemm_1d1d.cuh` 里是 `mma_t::fma`
调用的黑箱（`:440`），只能等能跑 SM100 的环境实测。**除此之外，tn 与 nt 的累积结构同构。**

---

## 7. H100 实测：MN-major K-grouped 的实物核对

本机（H100 sm90）用 **`k_grouped_bf16_gemm_tn_contiguous`** 真跑 MN-major K-grouped，验证 §6 的
结构性前提。数据按 DeepGEMM 的 MN-major 约定造（`tests/generators.py:551-591`：`a: [sum_k, m]`、
组数据落在 padded 槽起始、槽尾补零）。原始输出见 `.out.txt`。

**A1 逐组正确性**（`G=4, m=512, n=1024, logical_ks=[512,384,640,256]`，BF16 输入 / fp32 累加 vs fp64）：

| group | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| relerr | 4.32e-7 | 3.16e-7 | 5.53e-7 | 2.05e-7 |

BF16(BF16 乘 16-bit 尾数、fp32 可无损容纳)→误差只来自 fp32 求和舍入，~1e-7 量级，坐实 `D[e]=A_e^T B_e`。

**A2 组独立累加器**（全组一次 launch v.s. 每组单独一次 launch）：**4/4 逐位相等**。
⇒ 组之间确实不共享任何累加状态（与 §4.3/§5.3 的源码结构一致）。

**A3 组排列无关**（置换 `perm=[2,0,3,1]` 后重排物理组）：输出按同置换 **逐位相等**。
⇒ 每组的 K 加法链与其在物理张量里的位置无关（§4.2：链只由 `current_k_start` 与该组数据决定）。

**A4 契约**：SM90 BF16 tn 用 PSUM → `RuntimeError: Assertion error (csrc/apis/gemm.hpp:711):
not use_psum_layout and ...`。⇒ 真机确认 **PSUM 是 SM100-only**（与 `gemm.hpp:710-712` 一致）。

**A5 确定性**：同配置 20 次运行，**0 次 bit 变动**（固定 kernel、无原子写、每 tile 单写者）。

**A6 吞吐**（对照：第 12 篇 FP8 `nt` 在 `m=4096,n=7168`、8 组同形状 **425 TFLOPS**）：

| G | m | n | k/group | 时间 | TFLOPS |
|---|---|---|---|---|---|
| 8 | 4096 | 7168 | 512 | 5955 us | 40 |
| 8 | 4096 | 7168 | 1024 | 6230 us | 77 |
| 8 | 7168 | 7168 | 512 | 11663 us | 36 |

**注意**：这条 BF16 MN-major k-grouped 的吞吐**远低于** FP8 `nt`（40–77 v.s. 425 TFLOPS），说明
**SM90 的 BF16 MN-major 不是性能路径**（MN-major 无 multicast、BF16 双字节搬运、组多导致尾波）。
所以本实验只把它当**正确性 / 结构对照**，**不能**把它当目标卡 FP8 tn 的性能代理。本机 8 卡节点共享，
数字受争用，仅作量级参考。

**B 累积序解离**（`G=4`，前向同数据）：kernel v.s. fp64 `relerr=3.98e-7`、torch fp32 `bmm` v.s. fp64
`1.05e-7`，而 kernel 与 torch fp32 之间 **bit 不同比例 92.2%**、`max|差|=1.14e-4`。
⇒ 两个都「正确」（~1e-7）却**几乎处处 bit 不同**——正是第 174 篇「可复现性 ≠ 数值精度」在 MN-major
布局上的复现。

---

## 8. 与前面各篇的关系与边界

- **第 12 篇**（FP8 nt wgrad 精读）：本篇是它的「另一半」——tn 布局与 SM100 落地。第 12 篇给的那一行
  「tn 仅分析」在这里被展开。
- **第 16 篇**（K-grouped SF 打包）：tn 的 SFA/SFB 也要走同一套 `transform_k_grouped_sf_into_required_layout`
  （`gemm.hpp:363-364`），本篇不重复。
- **第 13 篇**（dgrad m-grouped）：dgrad 不等长轴在 M、每行独立（第 180 篇：dgrad 对 token 序免疫）；
  wgrad 不等长轴在 **K**、组内单调（本篇 + 第 174 篇）。两者正好是「分组轴落在哪」的对偶。
- **第 174 / 175 篇**：本篇把它们的累积序结论**从 SM90 nt 迁移到目标卡 FP8 tn**，并给出迁移前提的
  H100 实物验证。
- **第 19 篇**（sm90 BF16 稠密 kernel）：本篇用的是它的 K-grouped 入口 `sm90_bf16_k_grouped_gemm`
  （`sm90_bf16_gemm.hpp:264-330`），但聚焦在 MN-major 布局与组语义。

**边界**：① SM100 FP8 tn 主 kernel 与 UMMA 内部归约序**仅分析**；② BF16 MN-major kernel 是
**正确性对照**、非性能代理；③ nm 命令不可跨架构，SM100 端到端待能跑 SM100 的环境。

---

## 9. 参考

- DeepGEMM README / release（`#462` v2.8.1）：<https://github.com/deepseek-ai/DeepGEMM>
- K-grouped wgrad host 契约：`csrc/apis/gemm.hpp:320-371`（tn）、`:373-427`（nt）、`:676-722`（bf16 tn）
- SM100 FP8 tn jit：`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325-403`
- SM100 FP8 tn kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:24-104/199-291/359-560`
- SM90 BF16 MN-major k-grouped：`csrc/jit_kernels/impls/sm90_bf16_gemm.hpp:264-330`
- K-grouped scheduler：`deep_gemm/include/deep_gemm/scheduler/gemm.cuh:42-82/101-105/157-174/226-246`
- 官方测试：`tests/test_bf16.py:158-214`、`tests/generators.py:551-591`
