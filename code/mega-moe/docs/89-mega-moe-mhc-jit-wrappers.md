# Mega MoE / Mega mHC 的 JIT 启动包装层：从 host 张量到 kernel 实参的最后一公里

> 前面第 5（Mega Gate/mHC）、第 10（BF16 Mega MoE）、第 42（Mega MoE host 缓冲区）篇分别拆过
> **kernel 本体**和 **Python/host 缓冲区编排**，但真正把二者接起来的两个文件一直没人单独读过：
> `csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp`（157 行）与
> `csrc/jit_kernels/impls/sm100_mega_mhc.hpp`（149 行）。
> 它们是 **Mega MoE / Mega mHC 的 JIT 启动包装层**——夹在 `csrc/apis/*.hpp`（host 入口、参数校验）
> 与 `deep_gemm/include/.../impls/*.cuh`（SM100 主体 kernel）之间的「最后一公里」。
>
> 主 kernel 都是 **SM100（Blackwell）专属**，本机 H100（sm90）跑不了端到端；但这两个 wrapper 是
> **纯 host C++**，做的事情与 Tensor Core、TMEM、UMMA 全无关系——只做
> 「host 张量 → 形状/常量推导 → TMA 描述符 → JIT 模板实参 → launch 参数」。
> 本文把其中架构无关的部分抽出来，在 H100 上**直接 include 真实头文件**真机验证
> （`src/mega_moe_mhc_jit_wrapper.cu`，**50 项 ALL PASS**，见 `harness/exp_mega_moe_mhc_jit_wrapper.out.txt`）。
>
> 参考坐标：DeepGEMM `057ca59`（Public release 26/09/30）。

## 1. 两个 wrapper 在「三层结构」里的位置

DeepGEMM 的每个算子都是同一个三段式，Mega 家族也不例外：

```
csrc/apis/<name>.hpp              ← host 入口：pybind 注册、张量形状/dtype/设备的 DG_HOST_ASSERT
        │  （校验完直接调用）
csrc/jit_kernels/impls/*.hpp      ← 启动包装层（本文主角）：
        │                            · 调启发式拿 block/tiling/线程配置
        │                            · 把张量翻译成 TMA 描述符（CUtensorMap）
        │                            · 把常量拼成 JIT 源码的模板实参
        │                            · 组装 launch 配置（grid/block/cluster/smem）
        │  jit->compile(...) / jit->launch(...)
deep_gemm/include/.../impls/*.cuh ← SM100 主体 kernel（__global__，只认裸指针 + CUtensorMap + 常量）
```

为什么值得单独拆这一层？因为它承担了三件**最容易错、又最不显眼**的活：

1. **描述符选择**：同一个张量，BF16 和 FP8 要走不同的 `make_tma_*`，swizzle 模式还随元素字节数变化。
   选错了不会立刻崩，只会在 TMA 搬数据时地址错位（第 3、4 节会看到一处真实的「差一点就错」）。
2. **常量进模板**：`activation_clamp` 直接以**十六进制浮点字面量**编进 JIT 源码（`to_string(float)`），
   `is_shifted` / `store_bf16` / `y_fp8` 则被压成模板 `bool`——kernel 里因此全是 `if constexpr`，零运行时开销。
3. **ABI**：mHC 通过 `MixArgs` / `NormArgs` 两个 POD 结构体传参（`__grid_constant__`），
   它们的字段布局就是 host↔device 的契约。

对照已经拆过的 FP8 版：`sm100_fp8_fp4_mega_moe.hpp`（`docs/44` §2 拆过它的共享专家分支）
是 BF16 版的「满配参照」——BF16 版正是从它**删掉整条 block-scale（SF）通路**得到的。本文两处对照着讲。

## 2. BF16 Mega MoE 启动包装层（`sm100_bf16_mega_moe.hpp`）

### 2.1 形状与常量的二次推导（`:33-44`）

host 入口 `csrc/apis/mega_moe.hpp:285-395`（`bf16_mega_moe`）已经把张量校验完了，
wrapper 只做少量派生：

- `num_ranks = sym_buffer_ptrs.size()`（对称缓冲参与者的 rank 数，`:33`）；
- `num_experts = num_experts_per_rank * num_ranks`（`:34`）；
- `num_ring_tokens = l1_acts.size(0)`（ring buffer 里能放多少 token，`:35`）；
- `shared_intermediate_hidden = intermediate_hidden * num_shared_experts`（`:36`）。

然后调启发式拿配置（`:39-42`）：

```cpp
const auto config = get_mega_moe_config(
    num_ranks, num_experts, num_experts_per_rank,
    num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
    num_ring_tokens, 0, MmaKind::BF16);          // ← 注意第 10 个参数（num_sf_ring_tokens）传 0
```

这里第一个「BF16 专属」的点：`get_mega_moe_config`（`heuristics/mega_moe.hpp:197-263`）内部按
`MmaKind` 分流，BF16 的 `num_sf_ring_tokens` 恒被忽略（`:244` 只对 `is_mma_with_sf` 保留），
wrapper 干脆传 `0`。

紧跟一行断言（`:43-44`）：

```cpp
// The BF16 kernel has a single store block size for both layers
DG_HOST_ASSERT(config.store_block_m_l1 == config.store_block_m_l2);
```

**为什么**：`get_block_config_for_mega_moe`（`heuristics/mega_moe.hpp:88-127`）会算出两个存回块大小
`store_block_m_l1` / `store_block_m_l2`（`:113-115`）——FP8 路径允许 L1、L2 用不同的小块
（prefill 时 L1 用 24、L2 用 40，节省 smem 换流水深度）。但 BF16 kernel 的模板**只有一个
`STORE_BLOCK_M`**（`sm100_bf16_mega_moe.cuh:27`），wrapper 必须断言二者相等才安全。
本机实测（num_ranks=2/E=8/H=512/I=256）：BF16 与 FP8 都给 `store_l1=store_l2=32`，
但这是该配置恰好相等，断言在其它配置下才真正起作用。

### 2.2 描述符：11 个 → 但 BF16 只有 10 条「真」路径

wrapper 一共构造 **10 个 TMA 描述符**（`:46-86`），全部声明为 `const auto`：

| # | 描述符 | 调用 | 来源行 |
|---|---|---|---|
| 1 | `tensor_map_l1_acts` | `make_tma_2d_desc(l1_acts, hidden, num_ring_tokens, block_k, load_block_m, stride(-2), swizzle_acts_mode)` | `:47-51` |
| 2 | `tensor_map_l1_weights` | `make_tma_weights_3d_desc(l1_weights, block_k, load_block_n, swizzle_weights_mode)` | `:52` |
| 3 | `tensor_map_l1_output` | `make_tma_2d_desc(l2_acts, intermediate_hidden, num_ring_tokens, block_n/2, store_block_m_l1, stride(-2), swizzle_acts_mode)` | `:53-57` |
| 4 | `tensor_map_l2_acts` | 同 #1，内层改 `intermediate_hidden` | `:58-62` |
| 5 | `tensor_map_l2_weights` | 同 #2 | `:63` |
| 6-10 | `tensor_map_shared_*` | 见下（共享专家回退） | `:65-86` |

**与 FP8 版的第一处结构性差异**：FP8 wrapper（`sm100_fp8_fp4_mega_moe.hpp:50-139`）有 **10 条
routed 描述符**（5 个数据 + 5 个 SF），加 10 条共享专家描述符，共 20 个；BF16 版只需 10 个，
因为**整条 SF 通路（`l1_acts_sf`/`l1_weights_sf`/`l2_acts_sf`/`l2_weights_sf` + 4 个共享 SF）
被删掉了**。SF 那 4 个用的是 `make_tma_sf_desc`（`runtime_utils.hpp:384-408`），BF16 版里一次都不出现。

### 2.3 共享专家回退：指针别名而非重新编码（`:65-86`）

共享专家（shared expert）是 DeepSeek 每层恒激活的那一路（`docs/44`）。当 `num_shared_experts == 0`
时，wrapper 不构造新描述符，而是**直接复用 routed 的**：

```cpp
const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
    shared_l1_acts, hidden, num_max_tokens_per_rank, config.block_k, config.load_block_m,
    shared_l1_acts.stride(-2), config.swizzle_acts_mode) : tensor_map_l1_acts;   // ← 别名
```

`:65-70`、`:71-72`、`:73-78`、`:79-84`、`:85-86` 五处都是这个三元式模式。
**为什么**：kernel 里共享专家路径由模板布尔 `kHasShared = (kNumSharedExperts > 0)` 控制
（`sm100_bf16_mega_moe.cuh:36`），不开时整个共享段 `if constexpr` 消除，描述符永远不会被读——
但仍必须传一个**合法**的 `CUtensorMap` 实参（`__grid_constant__` 参数不能省），复用 routed 的最省事。
本机实测：`num_shared==0` 时 `shared_l1_acts` 描述符与 `l1_acts` 描述符 **128 字节逐字节相等**。

注意与 FP8 版的一个细节差异：FP8 的共享 L1 权重用 `make_tma_weights_2d_desc`（`:107`，2D），
而 routed L1 权重用 3D；BF16 的共享 L1 权重也用 `make_tma_weights_2d_desc`（`:71-72`）。

### 2.4 swizzle 随元素字节数变化：一处真实的「差一点就错」（`:53-57` vs FP8 `:72-76`）

这是本文最有意思的一处。看 L1 输出（SwiGLU 之后写回 `l2_acts`）的 swizzle 参数：

```cpp
// BF16 wrapper :53-57
make_tma_2d_desc(l2_acts, intermediate_hidden, config.num_ring_tokens,
                 config.block_n / 2, config.store_block_m_l1,
                 l2_acts.stride(-2),
                 config.swizzle_acts_mode);          // = 128
// FP8 wrapper :72-76
make_tma_2d_desc(l2_acts, intermediate_hidden, config.num_ring_tokens,
                 config.block_n / 2, config.store_block_m_l1,
                 l2_acts.stride(-2),
                 config.swizzle_acts_mode / 2);      // = 64
```

FP8 的注释（`sm100_fp8_fp4_mega_moe.hpp:69-71`）解释了「为什么减半」：SwiGLU 后 N 宽度减半
（`block_n/2`），所以 swizzle 也减半。但 BF16 版**不减半**——看似矛盾，实则都对：

`make_tma_2d_desc`（`runtime_utils.hpp:130-131`）有一条隐藏规整——**只要 swizzle 非 0，
smem 内层 box 就被强制改成 `swizzle_mode / elem_size`**。于是：

| | 元素字节 | block_n/2（元素） | 传入 swizzle | 实际 smem 内层（元素） | 实际 swizzle 字节 |
|---|---|---|---|---|---|
| BF16 | 2 | 64 | `128` | `128/2 = 64` ✓ | 128B（=64×2B） |
| FP8 | 1 | 64 | `64` | `64/1 = 64` ✓ | 64B（=64×1B） |

两者最终的 **smem 内层都是 64 个元素**，但 swizzle 的**字节数随元素大小减半**（128B vs 64B）。
反过来看更清楚：如果 BF16 照抄 FP8 的 `/2`，传入 swizzle `64`，则 smem 内层会被强制成
`64/2 = 32` 个元素 ≠ `block_n/2 = 64`，**TMA 每行只搬一半的宽度**——不崩，但结果全错。
所以这个「不对称」不是笔误，是**元素字节数补偿**。本机实测：BF16 的 `smem_inner=64`，
FP8 的 `smem_inner=64`，二者一致；并显式断言 BF16 的 `smem_inner != 32`。

### 2.5 把常量拼进 JIT 源码（`:96-130`）

wrapper 用 `std::format` 拼出一段微源码，里面只有一个 `__instantiate_kernel`，
`smm100_bf16_mega_moe_impl<...>` 的**显式模板实参**（`sm100_bf16_mega_moe.cuh:21-51` 共 20 个，
其余靠默认参数补齐）：

```cpp
const auto kernel = jit->compile("sm100_bf16_mega_moe", std::format(R"(
#include <deep_gemm/impls/sm100_bf16_mega_moe.cuh>
...
    auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_moe_impl<
        {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
        {}, {}, {}, {}, {}, {}, {}, {}, {}
    >);
...)", num_max_tokens_per_rank, hidden, intermediate_hidden, num_experts, num_shared_experts,
      num_topk, config.block_m, config.block_n, config.block_k, config.store_block_m_l1,
      config.num_ring_tokens, config.num_stages, config.num_bytes_per_pull,
      config.num_dispatch_threads, config.num_non_epilogue_threads, config.num_epilogue_threads,
      num_sms, num_ranks, to_string(activation_clamp), fast_math ? "true" : "false"));
```

模板参数与 kernel 声明的对应（`sm100_bf16_mega_moe.cuh:21-51`）：
`kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumSharedExperts, kNumTopk,
BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M, kNumRingTokens, kNumStages, kNumBytesPerPull,
kNumDispatchThreads, kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp, kFastMath`。

两个值得单独讲的点：

- **`to_string(activation_clamp)`**（`runtime_utils.hpp:75-83`）：把 `float` 编成**十六进制浮点字面量**，
  幂值用 `0x1p+0f` 形式（有限值）或 `cute::numeric_limits<float>::infinity()`（±inf），
  且对 NaN 直接 `DG_HOST_UNREACHABLE`（`:82`）。本机实测 7 个有限值（含 `-0.0`、次正规 `1e-5`、
  `FLT_MAX`、最小正规数）字面量——喂回 `strtof` 后**逐位回到原值**；±inf 编成 `±infinity()` 表达式，
  NaN 被拒绝。
- **`fast_math ? "true" : "false"`**：`fast_math` 直接决定 kernel 里用不用 `__expf` / 近似倒数
  （`sm100_bf16_mega_moe.cuh:961-969`）——所以它必须进模板（不同分支是两个编译产物），
  而 `activation_clamp` 也进模板（值不同即不同编译产物）。

对比 FP8 版：它的模板列表多 4 个显式实参（`store_block_m_l2, sf_block_m, sf_block_n,
num_sf_ring_tokens`，共 24 个），且末尾多一个 `to_string(l1_weights.scalar_type())`
（因为 FP8 权重可能是 e4m3 也可能是 packed FP4，dtype 必须进模板）。BF16 版固定 bf16，
省掉这 5 个。

### 2.6 launch 配置（`:132-154`）

```cpp
jit->launch(kernel, {
    .num_smem_bytes = config.smem_size,
    .grid_dim  = dim3(num_sms, 1, 1),                    // 只用本 GPU 的全部 SM
    .block_dim = dim3(config.num_dispatch_threads +      // 128
                      config.num_non_epilogue_threads +  // 128
                      config.num_epilogue_threads, 1, 1),// 256 → 共 512
    .cluster_dim = dim3(2, 1, 1),                        // 固定 2-CTA cluster
}, y.data_ptr(), cumulative_local_expert_recv_stats_ptr, num_tokens,
   layout::SymBuffer<>(sym_buffer_ptrs, rank_idx), tensor_map_l1_acts, ..., sm_locality_domains.data_ptr<uint8_t>());
```

- `grid = get_num_sms_for_mega_moe()`（`heuristics/mega_moe.hpp:67-71`）——**必须用满整卡 SM**，
  因为调度器要按 locality domain 分区（`docs/11`/`docs/34`）。
- `cluster_dim = (2,1,1)`：2-CTA 是写死的（`get_block_config_for_mega_moe` 返回值第一项恒为 2，
  `heuristics/mega_moe.hpp:126`），因为 UMMA 用 `2x1SM` 指令（`docs/10`）。
- 参数顺序与 kernel 形参（`sm100_bf16_mega_moe.cuh:53-67`）严格一致：
  `y, stats, num_tokens, SymBuffer, 10 个描述符, sm_locality_domains`。
  `SymBuffer`（`layout/sym_buffer.cuh`）由**裸指针数组 + 本 rank 索引**构造，是跨 rank 对称寻址的地基
  （`docs/87`）。

## 3. Mega mHC 启动包装层（`sm100_mega_mhc.hpp`）

mHC（Multi-head HyperConnection，`docs/05` 第二部分）与 Mega MoE 的 wrapper 风格不同：
它用**三个 lambda** 把「同一类张量 → 描述符」的规律收拢，用**两个 POD 结构体**传参。

### 3.1 三个 TMA lambda（`:37-51`）

```cpp
const auto make_hidden_tma_desc = [&](const torch::Tensor& t) {          // 2D：token × hidden
    return make_tma_2d_desc(t, hidden, num_tokens, mhc_layout::BLOCK_K, mhc_layout::BLOCK_M,
                            static_cast<int>(t.stride(0)), mhc_layout::kSwizzleMode); };
const auto make_residual_tma_desc = [&](const torch::Tensor& t) {        // 3D：route 当第三维
    return make_tma_3d_desc(t, hidden, num_tokens, mhc_layout::kNumRoutes,
                            mhc_layout::BLOCK_K, mhc_layout::BLOCK_M, mhc_layout::kNumRoutes,
                            static_cast<int>(t.stride(0)), static_cast<int>(t.stride(1)),
                            mhc_layout::kSwizzleMode); };
const auto make_coeff_tma_desc = [&](const torch::Tensor& t, const int channels, const int swizzle_mode) {
    return make_tma_2d_desc(t, channels, num_tokens, channels, mhc_layout::BLOCK_M,
                            static_cast<int>(t.stride(0)), swizzle_mode); };
```

常量来自 `layout/mega_mhc.cuh:18-31`：`BLOCK_M=64`、`BLOCK_N=kNumHCOutputs=24`、`BLOCK_K=64`、
`kSwizzleMode=128`；模型几何 `kNumRoutes=4`、`kNumHCOutputs=kNumRoutes*(kNumRoutes+2)=24`（`:11-12`）。

三个 lambda 分别对应三类张量：

- **hidden 类**（`x` / `new_residual` 的输出 / 非 shifted 时的 `y_bf16`）：标准 2D `[T, H]`。
- **residual 类**（`residual` / `new_residual`）：`[T, kNumRoutes, H]`——**route 是第三维**，
  用两个**显式 stride** 表达（`stride(0)` 给 token 维、`stride(1)` 给 route 维），
  避免物化转置视图。为什么这么设计？看 kernel 侧共享内存声明
  （`layout/mega_mhc.cuh:126-127`）：`residual[kNumIOStages][kNumRoutes][BLOCK_M*BLOCK_K/2]`
  ——**route-major、每个 64×64 连续**，正好是「3D TMA 的 dim2 是 route、每行 128B」的产物。
- **coeff 类**（`post_mix` / `comb_res_mix` / `shifted_prev_mix`）：per-token 的小通道张量，
  内层宽度 = channels。

本机实测：对 `[128,4,512]` bf16 的 residual，一次 3D TMA 载入 box=`[64(k) × 64(m) × 4(route)]`、
swizzle 128，落进 smem 后与「route-major 线性序 + 128B swizzle」的 host 预测 **0/32768 不符**，
证明「route 当第三维」的取址语义正确。

### 3.2 coeff 的 swizzle 选择（`:74-78`）

```cpp
const auto tensor_map_post_mix = make_coeff_tma_desc(post_mix, mhc_layout::kNumRoutes, 0);
const auto tensor_map_comb_res_mix = make_coeff_tma_desc(
    comb_res_mix, mhc_layout::kNumRoutes * mhc_layout::kNumRoutes, mhc_layout::kSwizzleMode / 2);
const auto tensor_map_shifted_prev_mix = make_coeff_tma_desc(
    is_shifted ? shifted_prev_mix.value() : post_mix, mhc_layout::kNumRoutes, 0);
```

- `post_mix`：内层 4 通道 × 4B = 16B，太小，**swizzle = 0**（不 swizzle）。
- `comb_res_mix`：内层 16 通道 × 4B = 64B，够一个 64B swizzle，故 `kSwizzleMode/2 = 64`。
- `shifted_prev_mix`：不 shifted 时**别名到 `post_mix`**（`is_shifted ? ... : post_mix`），
  与 BF16 的共享专家回退同款省事写法。

这与第 2.4 节的 BF16/FP8 swizzle 差异是**同一类设计语言**：swizzle 字节数总是「内层实际字节数」
能容下的最大档（0/32/64/128，`runtime_utils.hpp:109-115`），不够就退到 0。本机实测：
`post_mix` swizzle=0、`comb_res_mix` swizzle=64，且两者描述符**不相等**（值 0 与 64 都生效）。

### 3.3 `fn` 的 TF32 描述符（`:68-73`）

`fn` 是 HyperConnection 的「mix 系数投影」矩阵（`docs/05`/`docs/16`），形状
`[kNumHCOutputs=24, kNumRoutes*hidden]`、dtype **fp32**：

```cpp
const auto tensor_map_fn = make_tma_3d_desc(
    fn, hidden, mhc_layout::kNumHCOutputs, mhc_layout::kNumRoutes,
    mhc_layout::BLOCK_K, mhc_layout::kNumHCOutputs, mhc_layout::kNumRoutes,
    static_cast<int>(fn.stride(0)), hidden,
    mhc_layout::kSwizzleMode, 0, true);        // ← allow_tf32 = true
```

**为什么 `allow_tf32=true`**：mHC 的 mix 投影 UMMA 走 **TF32 TS**（`docs/05`），
`fn` 在 smem 里以 TF32 形式喂给 MMA。`allow_tf32` 让
`aten_dtype_to_tensor_map_dtype`（`runtime_utils.hpp:85-89`）把 fp32 张量的描述符 dtype 编成
`CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` 而不是 `FLOAT32`——这决定了 TMA 搬进 smem 的字节解释方式。
本机实测 `aten_dtype_to_tensor_map_dtype(kFloat, true, …) == TFLOAT32`，且 `fn` 的三维描述符
（gmem=(512,24,4)、strides=(2048,512)、smem=(64,24,4)、swizzle 128、TF32）与独立直连编码逐字节相等。

### 3.4 `MixArgs` / `NormArgs`：`__grid_constant__` POD 契约（`:82-107`）

mHC kernel 有两个大参数包（`sm100_mega_mhc.cuh:48-49`），定义在 `layout/mega_mhc.cuh:45-71`：

| 结构体 | 字段数 | 语义 | sizeof/align |
|---|---|---|---|
| `MixArgs` | 5 指针 + 4 float + 1 uint32 | 缩放/基址、新 mix 输出指针、4 个 eps 标量、Sinkhorn 迭代数 | 64 / 8 |
| `NormArgs` | 1 uint32 + 3 指针 + 2 float + 2 指针 + 2 int64 + 1 指针 + 1 int64 | token 数、权重/残差指针、eps/scale、bf16/fp8 输出、primary/shared SF 指针与 stride | 88 / 8 |

wrapper 逐个字段地填（`:82-107`），其中 `y_fp8` / `y_primary_sf` / `y_shared_sf` 用
`has_value() ? ptr : nullptr` 三态（`:101-106`），与 kernel 里对应 `if constexpr` 分支的模板布尔呼应。
本机实测 22 个字段的 `offsetof` + 两个 `sizeof` 与手算**逐项相等**，确认 host 侧填充与 device 侧读取
对齐（`NormArgs` 因 `uint32_t` 后的 4 字节 padding 而 sizeof=88，正是 ABI 的关键点）。

### 3.5 模板布尔与 launch（`:109-146`）

```cpp
const auto kernel = jit->compile("sm100_mega_mhc", std::format(R"(
#include <deep_gemm/impls/sm100_mega_mhc.cuh>
    auto ptr = reinterpret_cast<void*>(&sm100_mega_mhc_impl<
        {}, {}, {}, {}, {}, {}, {}   >);
)", hidden, num_splits, num_sms,
    mix_args.new_prev_mix != nullptr,   // kIsShifted
    store_bf16,                          // kStoreBF16
    norm_args.y_fp8 != nullptr,          // kStoreFP8
    shared_sf_block_m));                 // SF_BLOCK_M
```

7 个模板实参对应 kernel 声明 `sm100_mega_mhc.cuh:37-38`：
`<kHidden, kNumSplits, kNumSMs, kIsShifted, kStoreBF16, kStoreFP8, SF_BLOCK_M>`。
三个 `bool` **全部由「指针是否为空」推出**——这正是 §3.4 里那些 `has_value()?...:nullptr`
的用途：**同一个三元表达式既决定参数值，也决定模板分支**，二者天然一致。

- `kIsShifted` 来自 `new_prev_mix != nullptr`（`:85`）——shifted 与非 shifted 是**两个编译产物**，
  因为非 shifted 路径把 Mix+Pre+RMSNorm 融成 per-token worker，shifted 是两 token 一 task
  （`docs/05`：`epilogue:139-281` vs `:291-390`）。
- `kStoreFP8` 来自 `y_fp8 != nullptr`；`SF_BLOCK_M` 是 shared SF 的块大小（0 表示不写 shared SF）。

launch（`:129-146`）：`grid = (num_sms,1,1)`、`block = (kNumThreads,1,1)`（`kNumThreads = 6*4*32 = 768`，
`layout/mega_mhc.cuh:31`，6 个 warpgroup 的静态角色）、**无 cluster**（单 CTA 用 `Allocator1Sm`，
`sm100_mega_mhc.cuh:54`），与 Mega MoE 的 2-CTA 相反。还有一个可观测的调试口（`:56-62`）：
`DG_PRINT_CONFIGS` 打开时打印 `Mega normal/shifted mHC: T=.. H=.. splits=.. IO/Fn/A stages=.. threads=.. smem=..`。

### 3.6 上游 `csrc/apis/mega_mhc.hpp` 给 wrapper 喂了什么

`csrc/apis/mega_mhc.hpp:89-252` 是 mHC 的 host 入口，wrapper 的 `num_splits` / `scratch` /
`split_barriers` / `y_primary_sf` 都由它算出来：

- `num_splits = get_num_splits(num_m_blocks, hidden, num_sms)`（`:226`）——split-K 的份数；
- `scratch` 用 `Workspace<>::get_num_scratch_bytes(...)` 分配（`:227-232`），
  尺寸公式见 `layout/mega_mhc.cuh:97-101`（每 task 的 gemm partial + 两个 sqr-sum partial）；
- `needs_bf16_scratch`（`:229-238`）——shifted FP8 的 FP8 是从**舍入后的 bf16** 推出来的，
  故 FP8-only 调用会**借用 scratch 尾部**放那块 bf16 数值边界；
- `y_primary_sf` = `y_routed_sf` 或 `y_gemm_sf`（`:241`）——GEMM SF 与 routed SF 只差 stride，
  共用同一个 primary writer。

这套「先算字节 → `torch::empty` → `narrow` 出子视图 → 传裸指针」的编排，是第 42 篇讲过的
host 缓冲区风格在 mHC 上的复用。

## 4. 两个 wrapper 的横向对照

| | `sm100_bf16_mega_moe.hpp` | `sm100_fp8_fp4_mega_moe.hpp` | `sm100_mega_mhc.hpp` |
|---|---|---|---|
| 主体 kernel | `sm100_bf16_mega_moe_impl`（20 模板参） | `sm100_fp8_fp4_mega_moe_impl`（24） | `sm100_mega_mhc_impl`（7） |
| 描述符数 | 10（无 SF） | 20（含 10 个 SF） | 8 + 2 POD 参数包 |
| 特征描述符 | 2D/3D weights，无 SF | `make_tma_sf_desc`（MN-major，gran_k=32） | 3D route 维 + TF32 `fn` |
| 常量进模板 | `activation_clamp`（hexfloat）、`fast_math` | 多 `to_string(dtype)` | 三个 bool + `SF_BLOCK_M` |
| cluster | `(2,1,1)` | `(2,1,1)` | 无（`Allocator1Sm`） |
| 回退别名 | 共享专家 → routed 描述符 | 共享专家 → routed 描述符 | 非 shifted → `post_mix` / `x` |

三者共用一个「模板」：**host 把启发式配置、TMA 描述符、常量、launch 几何四样东西拼起来，
剩下的交给 DeepJIT**（`docs/33`/`docs/52`）。

## 5. H100 实测（`harness/exp_mega_moe_mhc_jit_wrapper.{sh,out.txt}`）

`src/mega_moe_mhc_jit_wrapper.cu` 用 nvcc 直接 include 真实头文件
（`runtime_utils.hpp` / `heuristics/mega_moe.hpp` / `layout/mega_mhc.cuh` / 间接 `layout/mega_moe.cuh`）
+ torch，在 H100（sm90）上真机跑。一个编译期小坑：真实头里的 `DG_UNIFIED_ASSERT` 在
「nvcc 单 TU 的 host pass」会落成 `asm("trap;")`，host 汇编器不认识 `trap`——本实验在 include 前
把它钉到 host 版 `assert` 即可（`-w` 下无副作用）。**总判定 ALL PASS（50/50）**：

- **Part 1（10 项）** `to_string(float)`：7 个有限值字面量喂回 `strtof` **逐位回到原值**
  （含 `-0.0` → `-0x0p+0f`、`1e-5` → `0x1.4f8b58p-17f`、`FLT_MAX` → `0x1.fffffep+127f`）；
  ±inf 编成 `±cute::numeric_limits<float>::infinity()`；NaN 被 `DG_HOST_UNREACHABLE` 拒绝。
- **Part 2（24 项）** `MixArgs` 10 个 + `NormArgs` 12 个字段 `offsetof` + 两个 `sizeof`，
  与手算**逐项相等**（`MixArgs`=64/8、`NormArgs`=88/8）。
- **Part 3（6 项）** mHC 三个 lambda：`post_mix`(swizzle 0) / `comb_res_mix`(swizzle 64) /
  `fn`(3D、TF32、swizzle 128) / `x`(2D) 四个描述符与独立直连编码**逐字节相等**，
  `post_mix`≠`comb_res_mix`；**真实 3D TMA 载入** route 当第三维 + 128B swizzle，**0/32768 不符**。
- **Part 4（10 项）** BF16 Mega MoE wrapper：真机跑 `get_mega_moe_config`——
  `block_k` BF16=64 / FP8=128、BF16 `num_sf_ring_tokens=0`、BF16 `store_l1==store_l2`；
  `l1_output` BF16 swizzle=128 / FP8 swizzle=64 但 smem_inner 同为 64（且 BF16≠32）；
  共享专家回退**逐字节别名相等**；`l1_weights` 4D 描述符（num_domains=2）与直连编码相等。

真机配置数字（num_ranks=2, E=8, H=512, I=256）：BF16
`block_m=128 block_n=128 block_k=64 store_l1=l2=32 num_stages=8 smem=218532 threads=128/128/256`；
FP8 同配置 `block_k=128 num_sf_ring_tokens=1920 num_stages=8 smem=225700`——BF16 因无 SF
省下约 7 KB smem。（`num_bytes_per_pull` 的上限也按 MmaKind 二分：BF16 4096B / FP8 8192B，
见 `heuristics/mega_moe.hpp:224-229`，故大 hidden 下 BF16 的 pull 次数更少。）

## 6. 小结

- Mega MoE / mHC 的 JIT 启动包装层是**纯 host C++**，与 SM100 指令无关；本机 H100 上
  直接 include 真实头文件即可验证其全部 host 逻辑（**50 项 ALL PASS**）。
- 三个「非平凡点」：① **描述符 swizzle 随元素字节数与通道宽度变化**
  （BF16 l1_output 取 128 / FP8 取 64，靠 `make_tma_2d_desc` 的 `smem_inner = swizzle/elem` 规整补偿）；
  ② **常量/开关进模板**（`activation_clamp` 用 hexfloat 字面量、三个 bool 由指针空值推出）；
  ③ **回退用指针别名**（共享专家 → routed；非 shifted → `post_mix`）而非重新编码。
- 主 kernel 仍 **SM100 专属、仅分析**：端到端 `bf16_mega_moe` / `mega_mhc` 在 H100 上编译/运行都会失败
  （`csrc/apis/mega_mhc.hpp:223` 断言 `arch_major==10`；`mega_moe.hpp:367` 只在 arch10 派发）。

## 参考链接（真实）

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
- Mega MoE PR：<https://github.com/deepseek-ai/DeepGEMM/pull/316>
- DeepEP 仓库：<https://github.com/deepseek-ai/DeepEP>
