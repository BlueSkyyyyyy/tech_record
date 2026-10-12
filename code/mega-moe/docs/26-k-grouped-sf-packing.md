# K-grouped SF 打包：MoE wgrad 的按组 block-scale 布局

> 本文对应路线图 **P16-1**。第 25 篇讲了**通用**（单组）的 SF 布局转换与 UE8M0 打包，
> 第 10 / 12 篇讲了 MoE 反向 wgrad 的 **K 轴分组** GEMM（`k_grouped_fp8_gemm_*`）。
> 两者的交叉点一直没展开：**当 GEMM 沿收缩维 K 分组、每组长度还各不相同时，
> block-scale 也要「按组拼接、按组独立打包」，每组的打包行必须重新对齐、尾部补零**。
> 这就是 host `transform_k_grouped_sf_into_required_layout`（`csrc/apis/layout.hpp:89-116`）与
> device `pack_fp32_into_ue8m0<kNumGroups,...>`（`deep_gemm/include/deep_gemm/impls/smxx_layout.cuh:193-293`）
> 负责的事。
>
> host 入口在 SM90 被 `layout.hpp:100-101` 直接拒绝（SM100 专属），**但 device kernel 与
> UMMA/TMEM 无关**，只做「找 owner 组 → 逐 4 行打包指数」。因此本文在 H100(sm90) 上**直接实例化并
> launch 仓库里的真实设备代码**（逐行原样抽取 `smxx_layout.cuh:193-293`，见
> `harness/exp_kgrouped_sf_pack.{sh,out.txt}`），用真实数字逐行对拍 host 参考，**全部 PASS**。

参考实现：`~/github/DeepGEMM`，公开版 commit `057ca59`（版本 2.8.1）。

## 1. 为什么 wgrad 的 scale 必须「按组重排」

MoE 训练里，专家的 linear1/linear2 权重梯度天然是「**沿收缩维 K 分组**」的 GEMM
（wgrad 的不等长轴是 K=token，见第 09/12 篇）：第 $e$ 个专家收缩 $k_e$ 个 token，输出
`[M, N]` 固定。DeepGEMM 用 `k_grouped_fp8_gemm_nt_contiguous`（SM90）或
`k_grouped_fp8_gemm_tn_contiguous`（SM100 tn）表达它，每组各有自己的 `[M, k_e]` / `[N, k_e]` 操作数。

block-scale 的问题随之而来。FP8 操作数的 scale 是「每 `gran_k` 个收缩元素一个」：

- SM90 wgrad 的 SF 是 **FP32、MN-major、不打包**（`layout.hpp:104-105` 直接走
  `get_mn_major_tma_aligned_tensor`），因此第 12 篇看到的只是「每组一个 MN-major 的 fp32 scale 段」；
- SM100 的 UMMA block-scaled MMA 只吃 **UE8M0**（8 位指数）scale，且**一次消费 4 个相邻
  `gran_k` 段、打包进一个 int32**（第 25 篇）。打包单元是「4 个 SF 行」，而每组的 SF 行数是
  `k_e / gran_k`，**一般不是 4 的倍数**。于是不能简单地把每组的打包结果首尾相接——
  否则下一组的第一个 int32 会「跨组」混入上一组的尾部行。正确做法是：
  **每组从新的 packed 行开始，组内 SF 行数不足 4 的倍数时在尾部补零（padding）**。

这一「按组独立对齐 + 补零」就是本文要讲的唯一核心，也是 host 计算 `packed_sf_k` 与 device
kernel 里 `num_padding_sf_rows` 的来源。

## 2. host：入口契约与两种部署

### 2.1 `transform_k_grouped_sf_into_required_layout`（`csrc/apis/layout.hpp:89-116`）

```cpp
DG_HOST_ASSERT(sf.dim() == 2);                                  // :95
DG_HOST_ASSERT(std::get<0>(recipe)==1 and std::get<1>(recipe)==1); // :96  recipe=(1,1,gran_k)
const int gran_k = std::get<2>(recipe);                         // :98
DG_HOST_ASSERT((arch_major==9  and gran_k==128 and k_alignment==128) or
               (arch_major==10 and (gran_k==32 or gran_k==128) and k_alignment%128==0)); // :100-101
if (sf is Float and arch==9)  return get_mn_major_tma_aligned_tensor(sf);          // :104-105
if (sf is Float and arch==10) return get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(...); // :108-109
if (sf is Int   and arch==10 and gran_k==32) return check_k_grouped_packed_ue8m0_tensor(...);     // :112-113
DG_HOST_UNREACHABLE(...);                                        // :115
```

三个要点：

1. **recipe 必须是 `(1,1,gran_k)`**（`:96`）——K-grouped 的 gran_mn 只能是 1（每个 SF 行对应一个
   MN 位置），gran_k 是 32（SM100 FP4/细粒度）或 128（FP8）。
2. **SM90 与 SM100 是两条完全不同的路**：SM90 只做 MN-major 转置（`:104-105`），不做 UE8M0 打包；
   SM100 才走打包（`:108-109`）。这解释了第 12 篇里 SM90 nt 与 SM100 tn 的 SF 形态差异。
3. `k_alignment` 契约：SM90 必须恰好 128，SM100 必须是 128 的倍数（`:100-101`）。

### 2.2 `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`（`smxx_layout.hpp:218-283`）

这是真正的 host 编排，**SM100 专属**（第一行就是 `arch==10` 断言，`:224`）：

```cpp
DG_HOST_ASSERT(jit->device.get_arch_major()==10 and gran_k in {32,128} and k_alignment%128==0); // :224
const auto [sf_k, mn] = get_shape<2>(sf);                       // :225  输入 [sf_k, mn]
DG_HOST_ASSERT(sf.is_contiguous());                             // :228
DG_HOST_ASSERT(num_groups<=128 and mn%4==0);                    // :229
```

`num_groups` 取自 `grouped_layout.numel()`（`:226`）。随后按「是否有同步好的 `ks_cpu`」分两路：

**contiguous（有 `ks_cpu`，`:240-248`）**——每组 K 显式给出，`packed_sf_k` 是**精确值**：

```cpp
for (k : ks_cpu) {
    DG_HOST_ASSERT(k%k_alignment==0 and k%gran_k==0);           // :243
    group_sf_k = k / gran_k;                                    // :244
    ref_sf_k  += group_sf_k;                                    // :245
    packed_sf_k += ceil_div(group_sf_k, 4);                     // :246  ← 每组独立上取整
}
DG_HOST_ASSERT(ref_sf_k == sf_k);                               // :248  sf 行数 = Σ k_e/gran_k
```

`:246` 的 `ceil_div(group_sf_k, 4)` 正是「每组独立补零到 4 的倍数」，且**没有跨组复用尾部**。

**PSUM（无 `ks_cpu`，`:249-253`）**——组边界由 device 从 `grouped_layout`（逻辑 K 结束的前缀和）
读出，host 只能给一个**上界**：

```cpp
// Exact group sizes are read from the psum layout by the pack kernel.
// This upper bound allows three tail-padding slots per group.
packed_sf_k = (sf_k + num_groups * 3) / 4;                      // :252
```

为什么是 `+ num_groups*3`？每组最多浪费 3 个 SF 行的尾部（`group_sf_k % 4` 最大为 3），
换算成 packed 行相当于每组最多多出 `3/4` 个 int32，向上取整后整式即 `(sf_k+3G)/4`。
输出分配为**连续** `[packed_sf_k, mn]`（`:257`，行距 = `mn`），与第 25 篇非分组路径用
`empty_strided(..., {..,1,tma_aligned_mn})`（`:136`）不同——因为 UMMA 消费的是按组连续铺开的
`[sum(packed_K_g), MN]`：

```cpp
// K-grouped packed SF is [sum(packed_K_g), MN], with contiguous MN and an aligned K pitch.
```
（`:285`，`check_k_grouped_packed_ue8m0_tensor` 的注释）

最后把 `num_groups / gran_k / k_alignment / use_psum_layout` 等传进 JIT 运行时
（`:264-281`），launch 配置固定 `block_mn=128 / block_packed_sf_k=16 / num_threads=512`（`:259-261`）。

### 2.3 已打包（INT）路径：`check_k_grouped_packed_ue8m0_tensor`（`smxx_layout.hpp:286-317`）

若 SF 已是 packed int32，只做检查（**gran_k 必须为 32**，`:293`）：

- 形状/步长：`mn%4==0`（`:298`）、`stride(0)>=mn and stride(1)==1`（`:299`）、**`stride(0)%4==0`**（`:300`）
  ——K pitch 仍要 4（int32）对齐；
- contiguous：`packed_sf_k >= Σ ceil_div(k, gran_k*4)`（`:304-312`）；
- PSUM：必须 `use_psum_layout`（`:313-314`）。

## 3. device：`pack_fp32_into_ue8m0<kNumGroups,...>`（`smxx_layout.cuh:193-293`）

模板参数 `kNumGroups, kNumThreads, BLOCK_MN, BLOCK_PACKED_SF_K, kTransposed=true, kUsePsumLayout=false`
（`:193-194`）。**每个 warp 负责一个 packed 行**（`:214-219`）；每个 lane 处理 4 个 MN 位置
（uint4 对齐，`mn%4==0`，`:199` 注释 + `:207`）。

### 3.1 找 owner 组与累积 padding（`:224-267`）

每个 warp 先算出自己这一行 `packed_sf_k_idx = blockIdx.y*BLOCK_PACKED_SF_K + warp_idx`（`:217`），
然后**从第 0 组开始线性扫描**，用 lane 广播把各组 layout 值读进来（`shfl`，`:243`），逐组累加：

```cpp
uint32_t num_prefix_sf_rows = 0, num_prefix_packed_rows = 0, prev_group_end = 0;
bool owner_group_found = false;
for (group_idx = 0; group_idx < kNumGroups; ++group_idx) {
    layout_value = __shfl_sync(..., layout_cache[group_idx%4], group_idx/4);      // :243 = grouped_layout[g]
    if (kUsePsumLayout) group_k = layout_value - align(prev_group_end, k_alignment); // :246
    else                group_k = layout_value;                                     // :248
    aligned_group_k = kUsePsumLayout ? align(group_k, k_alignment) : group_k;        // :250
    num_group_sf_rows = aligned_group_k / gran_k;                                    // :251
    group_sf_row_start = num_prefix_sf_rows;                                         // :252
    group_sf_row_end   = group_sf_row_start + num_group_sf_rows;                     // :253
    num_prefix_sf_rows     += num_group_sf_rows;                                     // :254
    num_prefix_packed_rows += ceil_div(num_group_sf_rows, 4u);                       // :255
    if (packed_sf_k_idx < num_prefix_packed_rows) { owner_group_found = true; break; }// :256-259
    if (remainder = num_group_sf_rows % 4; remainder > 0)                            // :260
        num_padding_sf_rows += 4 - remainder;                                        // :261
    if (kUsePsumLayout) prev_group_end = layout_value;                               // :262-263
}
if (not owner_group_found) return;                                                   // :265-266
```

三处细节值得注意：

1. **`break` 在累加 padding 之前**（`:256-261`）。所以 owner 组自身贡献的尾部 padding 不计入
   自己的 `num_padding_sf_rows`——该偏移只累积「**更早**各组留下的空洞」。这决定了下节的取址公式。
2. **PSUM 的组起点是「上一组结束向上对齐到 `k_alignment`」**（`:246` 的 `align(prev_group_end, k_alignment)`），
   再 `align(..., k_alignment)` 得实际 SF 行数（`:250`）。即 PSUM 里组间的 K 空隙既包含
   `k_alignment` 对齐 gap，也包含 `gran_k` 上取整。
3. `align` 是向上取整（`math.cuh` 的 `align<T, true>`）。contiguous 路径不做对齐（`:248`），
   因为 host 已断言 `k % k_alignment == 0`（`:243`）。

### 3.2 逐行打包（`:269-292`）

拿到 owner 组后，每个 lane 对 4 个相邻 SF 行取数，**输入行号要减去累积 padding**：

```cpp
for (mn_idx = lane; mn_idx < in_block_mn_uint4; mn_idx += 32) {
    uint4 values[4];
    for (j = 0; j < 4; ++j) {
        values[j] = make_uint4(0,0,0,0);
        sf_row_idx = packed_sf_k_idx*4 + j - num_padding_sf_rows;          // :275  ← 减累积 padding
        if (group_sf_row_start <= sf_row_idx and sf_row_idx < group_sf_row_end)
            values[j] = ((const uint4*)(sf + sf_row_idx*mn))[mn_idx];      // :277
        DG_TRAP_ONLY_DEVICE_ASSERT((values[j].x & 0x807fffffu) == 0);      // :279-282 只允许 2 的幂
    }
    packed.x = (values[0].x>>23)|(values[1].x>>15)|(values[2].x>>7)|(values[3].x<<1); // :287
    ...
    ((uint4*)(out + packed_sf_k_idx*mn))[mn_idx] = packed;                 // :291  K pitch = mn
}
```

- `:275`：`- num_padding_sf_rows` 把「本地 packed 行的第 j 个槽」映射回**全局输入 SF 行**；
  落在 `[start,end)` 之外（组尾补零）取 0。
- `:287`：4 个 UE8M0 指数按 **LSB-first** 打进一个 int32（byte0 的指数在最低 8 位……），与第 25 篇同。
- `:291`：输出行距用 `mn`（不是 `tma_aligned_mn`），与 host 的连续 `[packed_sf_k, mn]` 分配一致。
- `:279-282`：`DG_TRAP_ONLY_DEVICE_ASSERT` 要求 SF 的符号位 + 尾数位全 0（即精确 2 的幂），
  否则打包会静默丢信息——这是第 25 篇也提到的 debug 陷阱。

## 4. H100 实测：直接运行真实设备代码

**背景**：host 入口在 sm90 被 `smxx_layout.hpp:224` 的 `arch==10` 断言拒绝（`:F` 节实测），
但 device kernel 与架构无关。我们把 `smxx_layout.cuh:193-293` **逐行原样抽取**后编译运行
（`harness/exp_kgrouped_sf_pack.sh`，`nvcc -arch=sm_90`，输出里打印了抽取片段的 sha256），
并用**独立 host 参考**（逐行复刻 `:224-292` 的组归属 + padding + 位运算）逐行对拍。

- **A) num_groups=1 基线**：退化情形应与「无分组」语义一致 —— PASS（2 例）。
- **B) contiguous、gran_k=32**：3 组无 padding（`{256,384,128}` → 8/12/4 行）、
  3 组有 padding（`{256,320,192}` → 8/10/6 行，10 与 6 各补 2 个 SF 行）、
  **中间空组**（`{256,0,192}`）、8 组（含多种非 4 倍数）——**mism=0，逐行全对**。
- **C) contiguous、gran_k=128**：`k_alignment=128/256`、组内行数 1/2/3（大量 padding）——**全对**。
- **D) PSUM**：`grouped_layout` 存逻辑 K 结束前缀和，组间 gap 由 `align` 补齐。两组：
  `gran=32,kalign=128,ends={400,800,1024}` → 实际 16/12/4 行、`packed_sf_k` 上界 10（实际用 8），
  尾部 2 个 int32 保持**未初始化（poison 0xAA）**——坐实 host 上界公式「宽松、允许多余行」；
  `gran=128,kalign=256,ends={256,640,1024}` → 实际 2/4/2 行、上界 4（实际用 3）。**mism=0**。
- **E) 输出布局**：k-grouped 的 K pitch = `mn`（连续），非分组为 `get_tma_aligned_size(mn,4)`（16B 对齐）。
- **F) host wrapper 在 sm90 的拒绝**：真实 python API `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor(...)`
  抛 `Assertion error (smxx_layout.hpp:224): ... get_arch_major() == 10 ...`（RuntimeError），
  说明「kernel 可跑、host 包装 SM100 专属」的分层。

关键原始输出（完整见 `harness/exp_kgrouped_sf_pack.out.txt`）：

```
  [B2 with padding                   ] mn=512  gran_k=32  kalign=128  ng=3 contig sf_k=24   packed=7  (actual 7) mism=0 poison_rows=0 PASS
  [B3 empty group                    ] mn=512  gran_k=32  kalign=128  ng=3 contig sf_k=14   packed=4  (actual 4) mism=0 poison_rows=0 PASS
  [D1 gran=32 kalign=128             ] mn=512  gran_k=32  kalign=128  ng=3 PSUM   sf_k=32   packed=10 (actual 8) mism=0 poison_rows=1024 PASS
  [D2 gran=128 kalign=256            ] mn=512  gran_k=128 kalign=256  ng=3 PSUM   sf_k=8    packed=4  (actual 3) mism=0 poison_rows=512 PASS
== 汇总: ALL PASS ==
```

## 5. 小结与坐标表

| 事项 | 位置 |
|---|---|
| K-grouped SF host 入口（`(1,1,gran_k)`、sm90 转置 / sm100 打包） | `csrc/apis/layout.hpp:89-116` |
| 精确 `packed_sf_k = Σ ceil_div(k/gran_k,4)`（contiguous） | `smxx_layout.hpp:240-248` |
| PSUM 上界 `(sf_k+3G)/4` | `smxx_layout.hpp:249-253` |
| 连续输出 `[packed_sf_k, mn]`（K pitch = mn） | `smxx_layout.hpp:257` |
| 已打包 INT 检查（`gran_k==32`、`stride(0)%4==0`） | `smxx_layout.hpp:286-317` |
| 找 owner 组 / 累积 padding / `break` 时序 | `smxx_layout.cuh:224-267` |
| 减累积 padding 取址 / LSB-first 打包 / trap 只允许 2 的幂 | `smxx_layout.cuh:269-292` |
| H100 实验（真实设备代码 + host 参考逐行对拍） | `harness/exp_kgrouped_sf_pack.{sh,out.txt}` |

**结论**：K-grouped SF 打包 = 第 25 篇的 UE8M0 打包 **+ 按组独立对齐补零**。它与 UMMA/TMEM 无关，
H100 上可直接运行真实 kernel；差异只在 host 包装（SM100 专属）与「输出连续 `[packed,mn]`」
（非分组是 TMA 对齐 strided）。理解了这条链，第 12 篇里 SM100 wgrad 的 `grouped_layout`、
`ks_cpu`、PSUM 三件套就有了统一的 scale 侧对应。
