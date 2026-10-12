# 28 · DeepGEMM 的「选核」：`get_best_config` 启发式精读 + H100 对拍

> 前面 27 篇拆的都是 DeepGEMM 的 **kernel 本身**（MMA 怎么发、SF 怎么打包、调度怎么跑）。
> 第 27 篇（cuBLASLt 回退）讲了「DeepGEMM 什么时候**不用**自家 kernel」。这一篇补上最后一块拼图：
> 当它决定用自家 kernel 时，**block_m / block_n / block_k / cluster / stage / 线程数** 是怎么选的？
> 这就是 `csrc/jit_kernels/heuristics/` 里的 `get_best_config`。它是**架构无关**的纯 host 启发式，
> 在 H100（sm90）上**可以真机验证**——本文用 Python 逐行重写 SM90 那一套，再和真实 C++ 用
> `DG_PRINT_CONFIGS=1` 打印出来的选核结果做 20 组形状的逐字段对拍。
>
> **本文所有行号均以 commit `057ca59`（Public release 26/09/30）核对过。**

---

## 1. 选核在整条链路里的位置

一次 `fp8_gemm_nt` 调用大致经过：host API（`csrc/apis/gemm.hpp`）→ kernel wrapper
（`csrc/jit_kernels/impls/sm90_*_gemm.hpp`）里**填一个 `GemmDesc`**，再调
`get_best_config<SM90ArchSpec>(desc)` 得到 `GemmConfig`，最后按 `GemmConfig` 编译并 launch。

```
GemmDesc(形状/类型/gemm_type/kernel_type/major/num_sms/expected_*)
        │  sm90_fp8_gemm_1d2d.hpp:84-94 / :158-171 / :227-238
        │  sm90_fp8_gemm_1d1d.hpp:87-97  / :169-180
        │  sm90_bf16_gemm.hpp:84-102 / :152-165 / :215-226 / :283-294 / :338-348
        ▼
get_best_config<SM90ArchSpec>   (common.hpp:18-56)
        │  get_layout_candidates(desc)        枚举合法 Layout   (sm90.hpp:16-118)
        │  get_layout_info(desc, layout)      给每个 Layout 打分 (sm90.hpp:202-239)
        │  compare(a, b)                     取分最小者         (sm90.hpp:241-244)
        ▼
Layout(block_m,n,k,cluster_m,n)
        │  get_storage_config   → swizzle / load-store block (sm90.hpp:120-146)
        │  get_pipeline_config  → smem_size / num_stages     (sm90.hpp:148-188)
        │  get_launch_config    → threads / math+tma 划分     (sm90.hpp:190-200)
        ▼
GemmConfig { Layout, StorageConfig, PipelineConfig, LaunchConfig }  (config.hpp:158-172)
```

`get_best_config` 本体只有十几行（`common.hpp:18-56`）：

```cpp
desc.check_validity();                                   // config.hpp:49-61
const auto layout_candidates = ArchSpec::get_layout_candidates(desc);
auto layout = layout_candidates[0];
auto layout_info = ArchSpec::get_layout_info(desc, layout);
for (int i = 1; i < layout_candidates.size(); ++i) {     // 线性扫描取最小
    const auto candidate_info = ArchSpec::get_layout_info(desc, layout_candidates[i]);
    if (ArchSpec::compare(candidate_info, layout_info))
        layout = layout_candidates[i], layout_info = candidate_info;
}
```

注意 `compare` 用的是**严格小于**，所以并列时保留**先枚举到的**（这一点在复刻时必须一致）。
`DG_PRINT_CONFIGS` 环境变量打开时，它把 `desc` 与选出的 `GemmConfig` + `LayoutInfo` 打印一次
（`common.hpp:43-54`，用 `static unordered_set` 按 desc 去重），这正是本文对拍的抓手。

### 1.1 五个数据结构（config.hpp）

| 结构 | 行号 | 内容 |
|---|---|---|
| `GemmDesc` | `config.hpp:12-86` | 输入：`gemm_type`/`kernel_type`/`m,n,k,num_groups`/三个 dtype/`major_a,b`/`with_accumulation`/`num_sms`/`tc_util`/`compiled_dims`/`expected_*` |
| `Layout` | `:89-104` | 决策：`swap_ab` + `block_m,n,k` + `cluster_m,n` |
| `StorageConfig` | `:106-121` | 派生：load/store block + 三个 swizzle mode |
| `PipelineConfig` | `:123-135` | 派生：`smem_size`/`num_stages`/`num_tma_store_stages` |
| `LaunchConfig` | `:137-156` | 派生：`num_threads` 及其 tma/math 划分 |
| `LayoutInfo` | `:175-188` | 打分：`num_waves`/`last_wave_util`/`num_cycles` |

`expected_*` 的语义很关键（`config.hpp:30-33`）：**0 表示回退到真实 m/n/k/num_groups**。
这允许调用方传入「启发式用的期望形状」，与实际执行形状解耦——例如 m-grouped PSUM 下
`expected_m` 取 `expected_m_for_psum_layout` 而非物理 m（`sm90_fp8_gemm_1d2d.hpp:169-171`、
`sm90_bf16_gemm.hpp:163-165`），这样选核只按「有效 token 数」估，不按 padding 后的物理行数估。

`check_validity`（`config.hpp:49-61`）里和选核最相关的一条是 **`num_sms % 2 == 0`**——因为
cluster=2 的合法性与 wave 计算都假设 SM 数是偶数。

---

## 2. SM90：候选枚举（`get_layout_candidates`，sm90.hpp:16-118）

### 2.1 block_m 候选（:17-36）

```cpp
if (Normal or Batched or KGroupedContiguous) {
    block_m_candidates = {64, 128};
    if (desc.m <= 16) block_m_candidates.push_back(16);
    if (desc.m <= 32) block_m_candidates.push_back(32);
    if (desc.cd_dtype != torch::kFloat)              // BF16 输出才允许 256
        block_m_candidates.push_back(256);
} else if (MGroupedContiguous or ...WithPsumLayout) {
    block_m_candidates = {heuristics_runtime->get_mk_alignment_for_contiguous_layout()};
} else if (MGroupedMasked) {
    block_m_candidates = {64, 128};
}
```

三件事：
1. **小 m 才有小 block_m**（`m<=16` 加 16，`m<=32` 加 32）：block_m 太大时最后一个 M tile
   会大量越界/浪费；小 m 用小块避免 TMA L2 OOB。
2. **`cd_dtype != Float` 才有 256**：FP32 输出时 block_m=256 寄存器/ smem 撑不住；
   BF16/FP8 输出才放行（注释 `// BF16 output GEMM supports 256`）。
3. **m-grouped contiguous 只有一个候选**：block_m 被**钉死**为
   `mk_alignment_for_contiguous_layout`（SM90 固定 128，见 `runtime.hpp:12,18`）。
   这是 P4-3 讲过的「段必须对齐 BLOCK_M」的直接后果——选核都不能自由挑 block_m。

`mk_alignment` 默认 128（`runtime.hpp:12,18`），且 `get_theoretical_...` 在非 arch10
上恒返回 128（`runtime.hpp:57-59`）；只有 SM100 才升到 256（`:61-63`）。

### 2.2 block_n 候选（:38-57）

```cpp
int step = lcm(16, get_block_n_multiple_of());     // 默认 multiple=1 → step=16
int start = step;
if (Kernel1D1D and cd_dtype == Float) {            // 1D1D + FP32 输出：避免 bank conflict
    start = 24; block_n_candidates.push_back(16);
}
int end = 256;
if (Kernel1D2D) end = 192;
if (Kernel1D1D) end = 160;
for (int i = start; i <= end; i += step) block_n_candidates.push_back(i);
```

- block_n 以 `lcm(16, block_n_multiple_of)` 为步长枚举（用户可用
  `set_block_size_multiple_of` 调，见 `apis/config.hpp:39-47`）。
- **上限随 kernel 变**：1D2D ≤192、1D1D ≤160、NoSF（BF16）≤256。
- 1D1D + FP32 输出时步长从 **24** 起（还额外加一个 16）——因为 1D1D 是单 warpgroup 存储，
  FP32 的 store atom 在 16 的倍数上会和 smem 的 bank 冲突；`:43-48` 的注释
  `// Avoid bank conflicts for 1D1D kernel FP32 output` 说的就是它，并且那里还断言
  A/B 都是 K-major。

### 2.3 block_k、multicast、cluster（:59-79）

```cpp
const int block_k = 128 * 8 / get_num_element_bits(desc.get_mma_kind());
```

`block_k` **唯一确定**：让一行的字节数 = `128 * 8 / bits`（FP8=128、BF16=64）。
含义是「一个 K stage 的字节数固定」（128×8 bit 的 TMA swizzle 原子）。

```cpp
const bool disable_multicast =
    (KGroupedContiguous and num_groups > 4) or     // K 分组多 → 关 multicast
    (Batched);                                      // batched 不支持
for (cluster_m = 1; cluster_m <= (disable?1:2); ++cluster_m)
  for (cluster_n = 1; cluster_n <= (disable?1:2); ++cluster_n) {
      if (cluster_m * cluster_n > 2) continue;      // SM90 只支持 cluster 2
      if (num_sms % (cluster_m * cluster_n) != 0) continue;
      ...
  }
```

cluster 只枚举到 **2**（`cluster_m*cluster_n<=2`），即 `(1,1)/(2,1)/(1,2)` 三选一；
K 分组数 >4 或 batched 时只留 `(1,1)`。

### 2.4 合法性过滤（:81-111）

对每个 `(block_m, block_n)`：

| 过滤 | 行号 | 为什么 |
|---|---|---|
| 1D2D unroll：`block_n>block_k` 时 `block_n%(block_n-block_k)` 或 `block_k%(...)` 必须有一整除 | `:84-85` | 1D2D 沿 K 展开的循环要求 |
| masked/psum：`ceil_div(n,block_n) % (cluster_m*cluster_n) == 0` | `:89-91` | masked 多播要求 N 方向块数能被 cluster 整除 |
| `block_m>128 and block_n>128` 跳过 | `:94-95` | 寄存器不够 |
| swizzle ≥64B（`swizzle_a/b_mode % 64 == 0`） | `:102-103` | 32B swizzle 性能差 |
| `num_stages>=3`，且小矩阵（`bm*bn < 128*192`）要 `>=4` | `:105-108` | 盖住 TMA 延迟 |

`num_stages` 由 `get_pipeline_config` 算出（见 §3.2），所以这里是「先按 stage 反筛候选」。
全部通过才 `candidates.push_back(layout)`（`:110`）。最后断言候选非空（`:116-117`）。

---

## 3. SM90：从 Layout 派生其余配置

### 3.1 `get_storage_config`（:120-146）

- `load_block_m/n` = `layout.block_m/n`（SM90 不做 swap AB，`:125` 断言 `swap_ab==0`）。
- `store_block_m`：1D1D 固定 `wgmma_m=64`（`:129`），其余等于 `block_m`。
- swizzle 由**内维**决定：K-major 时用 `block_k`，MN-major 时用 load block；
  再经 `get_swizzle_mode`（`utils.hpp:22-30`）从 `{128,64,32,16}` 里挑第一个整除的。
- `swizzle_cd`：**FP32 输出不给 swizzle，置 0**（`:138-139`）。

### 3.2 `get_pipeline_config`（:148-188）

```cpp
smem_cd        = align(block_m*block_n*elem_cd, 1024);        // C/D TMA 缓冲
smem_barriers  = 16 * 8 * 2;
smem_a_per_stage = load_block_m*block_k*elem_a;
smem_b_per_stage = load_block_n*block_k*elem_b;
smem_sfa_per_stage = (kernel==NoSF) ? 0 : align(block_m*4,128);   // 1D2D/1D1D
smem_sfb_per_stage = (kernel==1D1D) ? align(block_n*4,128) : 0;
smem_extra_sfb = 1D2D ? align(ceil_div(k,block_k)*4*use_uniform_sfb, 8) : 0;
smem_tensormap = KGroupedContiguous ? 4*sizeof(CUtensorMap) : 0;  // =512B
num_stages = min((232448 - smem_extra) / smem_per_stage, 16);
```

要点：**smem 容量 232448B（:14）是硬约束**；`num_stages` 上限 16；K-grouped 额外留
4 个 tensormap（因为 SM90 按组原地改 tensormap，正是 P4-2 讲的
`sm90_fp8_gemm_1d1d.cuh:193-215`）。所以：
- 小形状（block_m/n 不大）几乎总能拿满 16 级流水；
- 大形状（如 block_m=256,n=128）每 stage 占用大，只能拿 3~4 级。

### 3.3 `get_launch_config`（:190-200）

```cpp
num_tma_threads = 128;
num_math_threads = layout.block_m <= 64 ? 128 : 256;
```

即 **block_m≤64 → 一个 math warpgroup（128 线程），否则两个（256）**。
总线程 = 128(TMA) + 128/256。

---

## 4. SM90：代价模型（`get_layout_info`，sm90.hpp:202-239）

打分**只建模 L1/L2 周期**（注释 `:227-228`：HBM 带宽与总计算量在各配置间近似不变）：

```cpp
num_blocks = ceil_div(expected_m, bm) * ceil_div(expected_n, bn) * expected_num_groups;
num_waves  = ceil_div(num_blocks, num_sms);
last_wave_util = (num_blocks % num_sms == 0) ? num_sms : num_blocks % num_sms;

l2_bw = min(64*num_sms, 8e6/1.3e3);       // B/cycle
l1_bw = 128*num_sms;                      // B/cycle

// 每块的数据搬运量
n_l2_ab   = expected_k * (bm/cluster_n + bn/cluster_m) * elem_ab;   // multicast 省一半
n_l1_ab   = expected_k * (bm + bn) * elem_ab;
n_l1_tc   = expected_k * (max(64,bm) + bn) * elem_ab + bm*bn*elem_cd;
n_l1_l2_cd= bm*bn*elem_cd * (with_accumulation ? 2 : 1);           // 累积要回读 D

l2_cycles = (n_l2_ab + n_l1_l2_cd) * num_blocks / l2_bw;
l1_cycles = (n_l1_ab + n_l1_tc + n_l1_l2_cd) * num_blocks / l1_bw;
wave_eff  = num_blocks / (num_waves * num_sms);
num_cycles = max(l1_cycles, l2_cycles) / wave_eff;

if (cluster_size > 1 and num_waves <= 1) num_cycles = INT64_MAX;   // 单波禁多播
```

四个「为什么」：

1. **`bm/cluster_n + bn/cluster_m`**：开了 cluster(=2) 后 A 或 B 被 multicast 到邻 SM，
   每块的 L2 读减少；cluster_n 分 bm、cluster_m 分 bn（与 SM90 的 layout A/B 约定一致）。
2. **`num_l1_l2_cd` 的 `with_accumulation?2:1`**：累积模式（`D=C+A·B`）要把 D 读回来再加，
   C/D 的流量翻倍——这条在 P7-1 的访存归因里也出现过（wgrad 强制累积）。
3. **`wave_eff` 除以「波效率」**：不是所有配置都能整除地用满所有 SM；最后不满的那一波按
   `num_blocks/(num_waves*num_sms)` 打折惩罚。所以**恰好整除**的配置会被偏爱。
4. **单波禁多播**（`:234-236`）：若只跑一波还开 cluster，多播省不了几次 L2 读却增加同步，
   直接置 `INT64_MAX`（永远输）。

`compare`（:241-244）就一行：`a.num_cycles < b.num_cycles`。

---

## 5. SM100 对照（**SM100 专属，仅分析**）

SM100ArchSpec（`sm100.hpp`）是**另一套完全不同的启发式**，差别值得单独说：

- **`swap_ab` 进入决策**（`:50`）：SM100 是 tcgen05，A 可以在不同侧，于是枚举 `swap_ab∈{0,1}`，
  并据此限制 cluster（`:79-91`：swap 后 A/D 只能放 cluster N）、block_n（`:121-124`：swap 时
  block_n 必须是 layout A/D 的 UMMA M=128）。
- **m-grouped 特殊快路**（`:34-46`）：直接 `swap_ab=true, block_n=128, block_m=mk_alignment,
  cluster_m=1, cluster_n = (ceil_div(n,128)%2==0 and num_sms%2==0 ? 2 : 1)`，**只给一个候选**。
- **TMEM 容量约束**（`:126-132`）：`umma_n + tmem_sf_cols > 512` 就跳过——这是 SM100 独有的
  tensor memory 预算（SF 用 UTCCP 占用 TMEM 列）。
- **swizzle 强制 128B**（`:137-141`）：A/B 只要有一个 K-major，就要求两个 swizzle 都恰好 128。
- **`get_num_tma_store_stages`**（`:193-207`）：k-grouped 且满足条件时用**单级** TMA store，
  以「节奏化」store 流量、提升 DRAM 吞吐（`num_tma_store_stages=1`）。
- **代价模型直接放弃**：`get_layout_info` 的 `num_cycles` **恒为 0**（`:271-272`，
  `// TODO: calculate expected cycles`）——SM100 目前只按 `compare` 的多级 tie-break 选：

```cpp
// sm100.hpp:275-301
1) 只要某一方 num_waves==1 → 单波更优
2) cluster 更大更优（更爱多播）
3) num_waves 更小更优
4) last_wave_util 更大更优
5) block_m + block_n 更小更优
6) block_m * block_n 更小更优
```

**结论**：SM90 是「**L1/L2 周期最小化**」的定量模型；SM100 退化成「**单波 → 多播 → 少波 →
尾波利用率 → 小块**」的词典序规则。两者的 `get_layout_candidates` 也几乎不共享代码。

---

## 6. H100 真机对拍（`harness/exp_best_config.py`）

方法：用 `DG_PRINT_CONFIGS=1` 让真实 C++ 对每类算子各打印一次它选出的配置，同时用 Python
**逐行重写** §2-§4 的算法，读回打印出的 `GemmDesc` 再算一遍，**逐字段**比对
（block_m/n/k、cluster、三个 swizzle、store_block_m、smem_size、num_stages、num_threads、
num_waves、last_wave_util、num_cycles 共 15 项）。

覆盖的算子/形状（20 组，覆盖 Normal/1D2D、Normal/NoSF、MGroupedContiguous、
MGroupedContiguousWithPsum、MGroupedMasked、KGroupedContiguous/1D1D、Batched）：

| # | 用例 | C++/Python 选出的 Layout | waves | cycles | stages | threads |
|---|---|---|---|---|---|---|
| 1 | dense fp8 256×512×1024 | bm64,bn16,bk128,cm1,cn1 | 1 | 1800 | 16 | 256 |
| 2 | dense fp8 16×128×256 | **bm16**,bn16 | 1 | 231 | 16 | 256 |
| 3 | dense fp8 32×256×512 | **bm32**,bn16 | 1 | 544 | 16 | 256 |
| 4 | dense fp8 8192×7168×7168 | **bm256,bn128,cn2** | 14 | 616447 | 3 | 384 |
| 5 | dense fp8 128×192×128 | bm64,bn16 | 1 | 258 | 16 | 256 |
| 6 | dense bf16 256×512×1024 | bm64,bn16,bk**64** | 1 | 3557 | 16 | 256 |
| 7 | dense bf16 16×128×256 | bm16,bn16,bk64 | 1 | 445 | 16 | 256 |
| 8 | dense bf16 4096×7168×7168 | bm256,bn128,cn2 | 7 | 609279 | 3 | 384 |
| 9 | dense bf16 64×160×384 | bm64,bn16,bk64 | 1 | 1359 | 16 | 256 |
| 10 | dense bf16 512×512×1024 | bm64,bn**32**,bk64 | 1 | 4305 | 16 | 256 |
| 11 | m-contig fp8 G=6 128×256×512 | bm**128**,bn32,cn1 | 1 | 1931 | 10 | 384 |
| 12 | m-psum fp8 G=6 128×256×512 | bm128,bn32,cn1 | 1 | 1932 | 10 | 384 |
| 13 | m-contig fp8 G=8 256×2048×7168 | bm128,bn160,**cm2** | 2 | 65791 | 5 | 384 |
| 14 | m-masked fp8 G=6 160×256×512 | bm64,bn32 | 1 | 1141 | 16 | 256 |
| 15 | m-masked fp8 G=8 256×1024×512 | bm128,bn128 | 1 | 3514 | 5 | 384 |
| 16 | k-grouped fp8 nt G=4 512×1024 | bm128,**bn136**,bk128 | 1 | 6611 | 4 | 384 |
| 17 | k-grouped fp8 nt G=8 4096×7168 | bm128,**bn152**,cn1 | 94 | 2626647 | 4 | 384 |
| 18 | k-grouped fp8 nt G=2 128×256 | bm64,bn16 | 1 | 393 | 16 | 256 |
| 19 | einsum batched bf16 b=16,h=16,r=d=512 | bm16,bn64,bk64 | 1 | 1800 | 16 | 256 |
| 20 | dense fp8 256×512×2048（`block_mult=(32,32)`） | bm64,bn**32** | 1 | 4304 | 16 | 256 |

**结果：20/20 组、15 个字段全部一致（0 不符）。** 这坐实了 §2-§4 的复刻与真实 C++ 完全等价。
几个可以直接从表里读出的「为什么」：

- **#2/#3**：m=16/32 时选了 bm=16/32 —— 正是 `sm90.hpp:25-26` 的小 m 分支。
- **#4/#8**：8192×7168 / 4096×7168 大形状选 **bm=256,cn=2**：256 是因为 `cd_dtype=BF16`
  （`:29-30`）；cn=2 是因为大 N 能被 2 整除且多播省 L2 读，且 waves=14/7 不是单波。
- **#16/#17**：k-grouped 的 **bn=136 / bn=152** 精确对应 1D1D+FP32 的 `start=24, step=16`
  （24,40,…,152）——而且 FP32 输出的 smem 不给 swizzle，这也是为什么它会落在 152 这种
  「非整数」但满足内维字节数的点。两组都 `cm=cn=1`（G=8 因 `num_groups>4` 禁多播；
  G=4 虽允许但代价模型仍选了不 cluster）。
- **#6 vs #1**：同样 256×512×1024，BF16 的 `block_k=64`（bits=16），FP8 的 `block_k=128`
  （bits=8）——`sm90.hpp:60` 的 `128*8/bits`。
- **#11 vs #14**：m-grouped contiguous 的 block_m **被钉死在 128**（`:33`），而 masked 用的是
  `{64,128}`（`:35`）——因此 #14 能选 64、#11 只能 128。
- **#20**：设 `block_size_multiple_of=(32,32)` 后 block_n 步长变 32，于是从默认的 bn=16
  变成 bn=32（`sm90.hpp:40`）。
- **#19**：batched 禁多播（`:67`），所以恒 `cm=cn=1`。

### 6.1 候选排序：把代价模型「打开」

对两个代表性 desc，Python 复刻给出全部候选按 `num_cycles` 排序（`exp_best_config.out.txt` 末尾）：

```
dense fp8 8192x7168x7168: 候选数=90
  选出   bm=256,bn=128,cn=2   waves=14 cycles=616447
  次优   bm=256,bn=112,cn=2   waves=16 cycles=673791  (差 9.3%)
  Top: bm=256,bn=128,cn=2 / bm=256,bn=112,cn=2 / bm=128,bn=192,cm=2 / ...

k-grouped fp8 nt G=8 4096x7168: 候选数=20
  选出   bm=128,bn=152,cn=1   waves=94  cycles=2626647
  次优   bm=128,bn=136,cn=1   waves=103 cycles=2697120  (差 2.7%)
```

可见选择不是「随便挑」，而是在 90 个合法配置里按 L1/L2 周期目标做了**近似最优**的搜索；
次优与最优往往只差几个百分点，说明代价模型的分辨率足够细。也解释了为什么 P17-1 里
DeepGEMM 与 cuBLASLt 的性能几乎打平——选核这一层已经调得比较到位。

---

## 7. 几条容易踩的契约

1. **`DG_PRINT_CONFIGS` 按 desc 去重**（`common.hpp:44-54`）：同一 desc 只打印一次。
   对拍时要保证每个被测 desc 唯一（或接受只打印一次）。
2. **默认 BF16 dense 根本不进启发式**：它走 cuBLASLt（P17-1），必须
   `use_deterministic_algorithms(True)` 或限制 `num_sms` 才进 `get_best_config`。
   本实验的 BF16 用例都开了 deterministic。
3. **`set_num_sms` 会同时改两件事**：既改 `desc.num_sms`（影响 `num_sms%(cluster)` 与 waves），
   又经 `is_cublaslt_available()` 决定是否回退（P17-1）。
4. **`block_size_multiple_of` / `mk_alignment` 不在 `GemmDesc` 里**，它们是
   `HeuristicsRuntime` 的全局状态（`runtime.hpp:14-18`），绑定见
   `apis/config.hpp:39-47` 与 `apis/layout.hpp:133-141`。
5. **SM90 只到 cluster 2**：`cluster_m*cluster_n<=2`（`sm90.hpp:74`）；SM100 同样。
6. **只有 SM90 用定量代价模型**：SM100 的 `num_cycles` 恒 0（`sm100.hpp:271-272`），
   改用词典序 tie-break。

---

## 8. 小结

- **选核 = 枚举合法 Layout + 打分 + 取最小**（`common.hpp:18-56`）。候选由
  `get_layout_candidates` 决定，打分由 `get_layout_info` 决定，二者都是架构相关、纯 host 的。
- **SM90 是「L1/L2 周期最小化」**：显式建模每个 block 的 A/B/C/D 搬运字节、multicast 的
  收益、累积模式 D 回读的翻倍、以及尾波效率惩罚；单波禁多播。
- **SM100 退化成词典序规则**（单波 → 多播 → 少波 → 尾波 → 小块），并额外受 TMEM 容量、
  swap AB、128B swizzle 约束；代价模型标了 TODO。
- **H100 上 20/20 组、15 字段全部对拍一致**，证明复刻等价；候选排序显示它确实在一个
  90 元素的搜索空间里做了近似最优选择。
- **与 P17-1 合起来**才是完整的「选核」：P17-1 回答「用不用自家 kernel（cuBLASLt 回退）」，
  本篇回答「用自家 kernel 时怎么配」。

配套实验：`harness/exp_best_config.py` / `.sh` / `.out.txt`（H100 真机，20 组对拍 + 候选排序）。

### 关键源文件

- `csrc/jit_kernels/heuristics/common.hpp`（`get_best_config`）
- `csrc/jit_kernels/heuristics/sm90.hpp`（SM90 全套）
- `csrc/jit_kernels/heuristics/sm100.hpp`（SM100 全套，仅分析）
- `csrc/jit_kernels/heuristics/config.hpp` / `runtime.hpp` / `utils.hpp`
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp` / `sm90_fp8_gemm_1d1d.hpp` / `sm90_bf16_gemm.hpp`
  （构造 `GemmDesc` 的各 host 入口）
- 绑定：`csrc/apis/config.hpp:15-47`、`csrc/apis/layout.hpp:133-141`
