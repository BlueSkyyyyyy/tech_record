# FP8 Mega MoE / Mega Gate 的 JIT 启动包装层：两份「满配」包装的最后一公里

> 第 89 篇（P79）拆了 **BF16 Mega MoE** 与 **Mega mHC** 两个 JIT 启动包装层，并顺手把 FP8 Mega MoE
> 包装层当「对照」数了数描述符。但真正**满配**的 `sm100_fp8_fp4_mega_moe.hpp`（224 行）和
> **Mega Gate** 的 `sm100_bf16_mega_gate.hpp`（92 行）一直没被单独读过——本篇补上，凑齐
> `sm100_{fp8_fp4_mega_moe, bf16_mega_moe, bf16_mega_gate, mega_mhc}.hpp` 四个包装层。
>
> 四个主体 kernel 全是 **SM100（Blackwell）专属**，本机 H100（sm90）跑不了端到端；但包装层是
> **纯 host C++**：只做「host 张量 → 启发式配置 → TMA 描述符（`CUtensorMap`）→ JIT 模板实参 → launch」。
> 本篇把这些架构无关的部分抽出来，在 H100 上**直接 include 真实头文件**真机验证
> （`src/mega_moe_gate_jit_wrapper.cu`，**143 项 ALL PASS**，见
> `harness/exp_mega_moe_gate_jit_wrapper.out.txt`）。
>
> 参考坐标：DeepGEMM `057ca59`（Public release 26/09/30）。宿主环境：CUDA 13.2 / H100 80GB HBM3。

## 1. 两份包装在「三段式」里的位置（复用第 89 篇的分层）

```
csrc/apis/<name>.hpp              ← host 入口：pybind 注册 + 形状/dtype/设备断言
        │
csrc/jit_kernels/impls/*.hpp      ← 启动包装层（本文主角）：
        │                            · get_*_config() 拿 block / stage / 线程 / smem
        │                            · make_tma_*() 把张量翻译成 CUtensorMap
        │                            · std::format() 把常量拼成 JIT 微源码的模板实参
        │                            · 组装 launch（grid/block/cluster/smem）
        │  jit->compile(...) / jit->launch(...)
deep_gemm/include/.../impls/*.cuh ← SM100 主体 kernel（__global__，只认裸指针 + CUtensorMap + 常量）
```

与第 89 篇的两个相比，本篇的两个包装层承担同一类工作，但各有「满配」特征：

| wrapper | 文件 | 行数 | 特征 |
|---|---|---|---|
| FP8/FP4 Mega MoE | `sm100_fp8_fp4_mega_moe.hpp` | 224 | 18 个描述符（含 9 个 block-scale SF）、25 个模板实参 |
| BF16 Mega Gate | `sm100_bf16_mega_gate.hpp` | 92 | 2 个描述符、15 个布尔/常量模板实参、自管 scratch + score barrier |

## 2. FP8/FP4 Mega MoE 启动包装层（`sm100_fp8_fp4_mega_moe.hpp`）

### 2.1 入参与二次推导（`:16-48`）

入口是一个 22 参数的 `static void`（`:16-36`）：`y`、12 个激活/权重/SF 张量、`sm_locality_domains`、
可选的 `cumulative_local_expert_recv_stats`、对称缓冲指针数组与几何参数、`activation_clamp`、`fast_math`。
host 入口 `csrc/apis/mega_moe.hpp:167`（`fp8_fp4_mega_moe`）已经把 recipe（`(1,1,32)`）、
权重布局（`check_weights_layout_3d`）、SF 布局（`check_sf_layout(...,kGranMN=1,kGranK=32)`）
全校验完了，wrapper 只做四行派生（`:37-41`）：

```cpp
const auto num_ranks = (int)sym_buffer_ptrs.size();                    // 参与对称缓冲的 rank 数
const auto num_experts = num_experts_per_rank * num_ranks;             // 全局专家数
const auto num_ring_tokens = (int)l1_acts.size(0);                     // ring 容量（token）
const auto num_sf_ring_tokens = (int)l1_acts_sf.size(0);               // SF ring 容量（token）
const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;  // E_s 折叠
```

与 BF16 版（`sm100_bf16_mega_moe.hpp:33-36`）**唯一区别**是多取了 `num_sf_ring_tokens`
（SF ring 行数），并把 `shared_intermediate_hidden` 用在共享专家的 L1 N / L2 K 上。

启发式调用（`:44-48`）固定传 `MmaKind::MXFP8FP4`：

```cpp
const auto config = get_mega_moe_config(
    num_ranks, num_experts, num_experts_per_rank,
    num_max_tokens_per_rank, num_tokens, num_topk, hidden, intermediate_hidden,
    num_ring_tokens, num_sf_ring_tokens,          // ← BF16 版这里写 0
    MmaKind::MXFP8FP4);
```

`get_mega_moe_config`（`heuristics/mega_moe.hpp:197-263`）内部按 `MmaKind` 分流：
`block_k = 128 / elem_bytes`（BF16=64、FP8=128，`:117`）、`sf_block_m/n` 只在带 SF 时由
`SM100ArchSpec::get_sf_uttcp_aligned_block_sizes` 给出（`:212-213`）、`num_sf_ring_tokens`
只在带 SF 时保留（`:244`）、pull 上限二分（`:224`，FP8=8192B / BF16=4096B）。

### 2.2 18 个 TMA 描述符（`:50-139`）

先算 SF 的 smem 外层维（`:51-52`）：

```cpp
constexpr int kGranK = 32;
const int sf_smem_outer_dim = config.block_k / (kGranK * 4);   // FP8 block_k=128 → 1
```

**为什么除以 4**：UE8M0 的 block scale 按 `gran_k=32` 分组、每 4 个 int8 打包成一个 int32 存
（第 25/26 篇），所以 `block_k=128` 的 K 轴只有 `128/32/4 = 1` 个打包 SF 槽——这就是 SF TMA 描述符
的 smem 外层维。全部 18 个描述符列在下表（行号已核对）：

| # | 描述符 | 构造 | 行 | shape_mn / shape_k | groups |
|---|---|---|---|---|---|
| 1 | `l1_acts` | `make_tma_2d_desc` | `:53-57` | [ring, hidden] | – |
| 2 | `l1_acts_sf` | `make_tma_sf_desc(MN,…)` | `:58-62` | `num_sf_ring_tokens, hidden` | 1 |
| 3 | `l1_weights` | `make_tma_weights_3d_desc` | `:63` | [2I, H] × E_rank | – |
| 4 | `l1_weights_sf` | `make_tma_sf_desc(MN,…)` | `:64-68` | `2I, hidden` | E_rank |
| 5 | `l1_output` (= l2_acts) | `make_tma_2d_desc` | `:72-76` | [ring, I]，swizzle/2 | – |
| 6 | `l2_acts` | `make_tma_2d_desc` | `:77-81` | [ring, I] | – |
| 7 | `l2_acts_sf` | `make_tma_sf_desc(MN,…)` | `:82-86` | `num_sf_ring_tokens, I` | 1 |
| 8 | `l2_weights` | `make_tma_weights_3d_desc` | `:87` | [H, I] × E_rank | – |
| 9 | `l2_weights_sf` | `make_tma_sf_desc(MN,…)` | `:88-92` | `hidden, I` | E_rank |
| 10-18 | `shared_*` | 三元式（见 §2.3） | `:94-139` | 同上但共享形状 | 1 |

**这与第 89 篇的计数不一致**：第 89 篇 §2.2 与 §4 把 FP8 写成「10 条 routed + 10 条 shared = 20」，
但按源码声明逐个数只有 **9 条 routed**（上表 #1–#9：4 个数据 + 4 个 SF + 1 个 L1 输出）
**+ 9 条 shared = 18**（本机用 `grep -c "const auto tensor_map"` 与 launch 实参计数双重核对）。
本文以源码为准。

### 2.3 共享专家回退：指针别名（`:94-139`）

与 BF16 版同款写法——`num_shared_experts == 0` 时共享描述符**直接复用 routed 的**（合法且最省事，
`__grid_constant__` 实参不能省，但 kernel 里 `if constexpr(kHasShared)` 永不读它）：

```cpp
const auto tensor_map_shared_l1_acts = num_shared_experts > 0 ? make_tma_2d_desc(
    shared_l1_acts, hidden, num_max_tokens_per_rank, config.block_k, config.load_block_m,
    shared_l1_acts.stride(-2), config.swizzle_acts_mode) : tensor_map_l1_acts;
```

当 `num_shared_experts = S > 0` 时，共享形状把 `E_s = S` 折进不同轴——这正是第 44 篇（P34）
`SHARED_L1_SHAPE_N = L1_SHAPE_N · E_s`（N 轴）/ `SHARED_L2_SHAPE_K = L2_SHAPE_K · E_s`（K 轴）的
host 侧落地：

- `shared_l1_acts`：[max_tok, hidden]（同 routed acts，恒激活的“激活”就是输入 `x`）；
- `shared_l1_weights`：用 `make_tma_weights_2d_desc`（`:107`，2D，**非路由的 3D**），N = `2·I·S`；
- `shared_l1_output`/`shared_l2_acts`：[max_tok, `I·S`]，K/N 折进 `I·S`；
- `shared_l2_weights`：[hidden, `I·S`]；
- 4 个共享 SF 全部 `num_groups=1`（`:100-105/108-113/126-131/134-139`），因为共享专家只有 1 个
  “逻辑专家”，SF 没有 per-expert 基址。

注意 FP8 的共享 SF 用 `shared_l1_acts_sf.size(0)`（layout 给的 `get_num_max_shared_sf_tokens`）
作为 `shape_mn`，而 routed acts SF 用 `config.num_sf_ring_tokens`——两者**不是**同一个上界。

### 2.4 SF 描述符为何是 MN-major + 无 swizzle（`:58-62` 等）

四个 `make_tma_sf_desc` 的第一实参恒为 `cute::UMMA::Major::MN`，末两参恒为
`sf_smem_outer_dim, l1_acts_sf.stride(-1)`。看 `make_tma_sf_desc`（`runtime_utils.hpp:384-408`）：

- `DG_HOST_ASSERT(major == MN)`（`:393`）+ `DG_HOST_ASSERT(swizzle_mode == 0)`（`:396`）——
  SF 不 swizzle（注释 `:395` “maybe swizzle SF as well”是 TODO）；
- `shape_mn = get_tma_aligned_size(shape_mn, element_size)`（`:398`，int32 → `align(mn,4)`）；
- `gmem_outer_stride = sf_k_stride == 0 ? shape_mn : sf_k_stride`，且断言 `>= shape_mn`（`:400-401`）——
  故传入的 `stride(-1)` 必须是 MN-major 的**非连续外层 stride**。

host 侧的对齐来源是 `csrc/apis/mega_moe.hpp:104-115`：所有 SF 张量都用
`torch::from_blob(..., {M, K/128}, {1, M})` 造出「M 连续、最后一维 stride = M」的列主序布局，
所以 `stride(-1) = M = num_sf_ring_tokens`（或 `get_num_max_shared_sf_tokens`），恰好 ≥ `align(M,4)`。
这是第 25 篇「MN-major + TMA 对齐」在 Mega MoE 的实例。

### 2.5 模板实参与 launch（`:149-221`）

`jit->compile` 拼出一段只含 `__instantiate_kernel` 的微源码（`:149-173`），显式给出
`sm100_fp8_fp4_mega_moe_impl` 的 **25 个模板实参**（`:174-189`）；kernel 声明见
`sm100_fp8_fp4_mega_moe.cuh:21-55`：

```
kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumSharedExperts, kNumTopk,
BLOCK_M, BLOCK_N, BLOCK_K, STORE_BLOCK_M_L1, STORE_BLOCK_M_L2, SF_BLOCK_M, SF_BLOCK_N,
kNumRingTokens, kNumSFRingTokens, kNumStages, kNumBytesPerPull,
kNumDispatchThreads, kNumNonEpilogueThreads, kNumEpilogueThreads, kNumSMs, kNumRanks,
kActivationClamp, kFastMath, weight_dtype_t
```

比 BF16 版（20 个）多了 5 个：`STORE_BLOCK_M_L2`（BF16 只有一个 store block）、`SF_BLOCK_M/N`、
`kNumSFRingTokens`，以及末位的 **`to_string(l1_weights.scalar_type())`**（FP8 权重可能是 e4m3，
也可能是 packed FP4，dtype 必须进模板；BF16 固定 bf16 省掉）。

launch（`:192-221`）与 BF16 版一致：`grid = get_num_sms_for_mega_moe()`（`:146`，必须用满整卡
SM 以分 locality domain）、`block = 128+128+256 = 512`、`cluster = (2,1,1)`。launch 实参顺序与
kernel 形参（`sm100_fp8_fp4_mega_moe.cuh:57-81`）**严格同序**：`y, stats, num_tokens, SymBuffer,
18 个描述符, sm_locality_domains`。

## 3. Mega Gate 启动包装层（`sm100_bf16_mega_gate.hpp`）

Mega Gate（第 13 篇）把「BF16 打分 GEMM + softplus/sqrt + bias 排名 + top-k + 逻辑→物理映射」
压进单 kernel。它的包装层与 MoE 不同：**描述符只有 2 个，参数几乎全是布尔开关，且自管两块 scratch**。

### 3.1 入参与配置（`:15-33`）

入口 14 参数（`:15-25`）：`x`、`weight`、7 个可选张量（`bias/image_bias/image_token_mask/mask/
fix_routing_mask/force_random/out`）、`routing_args`、`score_barriers` 与几何参数。

```cpp
const auto num_aligned_experts = align(num_routed_experts, kExpertAlignment);   // :26，向上取 128
const auto config = get_mega_gate_config(num_tokens, hidden, num_routed_experts, runtime->get_num_sms()); // :28
```

**两块 scratch**（`:30-34`）：

```cpp
const auto num_token_blocks = ceil_div(num_tokens, config.block_tokens);
const auto num_scratch_bytes = mega_gate_layout::Workspace<>::get_num_scratch_bytes(
    num_token_blocks, config.num_split_k, config.block_tokens, num_aligned_experts);
const auto scratch = torch::empty({(int64_t)num_scratch_bytes}, x.options().dtype(torch::kByte));
```

score scratch 是 split-K 的落盘区：每个逻辑 CTA 把自己的 K 分片写进
`[num_token_blocks * block_tokens, num_split_k, num_aligned_experts]`（`layout/mega_gate.cuh:78-108`，
`get_num_scratch_bytes` 的公式 = `blocks·splits·block_tokens·E_aligned·4`，本机逐项核对通过）。
`score_barriers` 则由 host 入口 `csrc/apis/mega_gate.hpp:13-26` 按 stream 缓存一份并 `torch::zeros`
（注释 `:19`：warm-up 必须在 CUDA Graph capture 前做，否则一次性清零会被录进图里重放）。

### 3.2 两个描述符（`:36-41`）

```cpp
const auto tensor_map_x = make_tma_a_desc(Major::K, x, num_tokens, hidden,
    config.load_block_m, mega_gate_layout::BLOCK_K, x.stride(0), 1, mega_gate_layout::kSwizzleMode);
const auto tensor_map_weight = make_tma_b_desc(Major::K, weight, num_routed_experts, hidden,
    config.load_block_n, mega_gate_layout::BLOCK_K, weight.stride(0), 1, mega_gate_layout::kSwizzleMode);
```

打分 GEMM 是 `scores = x @ weightᵀ`，A=`x`（[T,H]）、B=`weight`（[E,H]），**都 K-major**
（`Major::K`，因为 H 是收缩维且在最后一维连续）。`BLOCK_K = 64`、`kSwizzleMode = 128`
（`layout/mega_gate.cuh:10-11`）。注意这里用 `make_tma_a_desc`/`make_tma_b_desc`
（`runtime_utils.hpp:325-365`）做 K-major 的 inner/outer 二分（`get_inner_outer_dims`），
与 MoE 里 acts 直接用 `make_tma_2d_desc` 不同。

### 3.3 15 个模板实参（`:44-65`）

```cpp
auto ptr = reinterpret_cast<void*>(&sm100_bf16_mega_gate_impl<
    {}, {},              // SHAPE_K, kNumRoutedExperts(=num_aligned_experts)
    {},                  // kNumGateThreads
    {}, {}, {},          // kNumMmaCtas, kNumSplitK, kNumExpertGroups
    {},                  // kNumTopk
    {}, {}, {}, {}, {}, {}, {}, {}   // 8 个 bool：padded/bias/image_mask/mask/fix/random/physical/unmapped
>);
```

8 个 bool 的实参全由「是否有值」推出（`:63-65`）：

```cpp
num_routed_experts != num_aligned_experts,   // kHasPaddedExperts（专家数非 128 倍数才需 padding）
bias.has_value(), image_token_mask.has_value(), mask.has_value(), fix_routing_mask.has_value(),
force_random.has_value(), routing_args.to_physical_map != nullptr, routing_args.unmapped_topk_idx != nullptr
```

kernel 声明 `sm100_bf16_mega_gate.cuh:25-31` 与之逐一对齐。**`num_aligned_experts` 进模板而非
`num_routed_experts`**：kernel 的 `UMMA_M = kNumExpertsPerGroup = kNumRoutedExperts/kNumExpertGroups`
要用对齐后的值，真实专家数只是 padding 的“有效范围”。

### 3.4 `RoutingArgs`：`__grid_constant__` POD 契约（`layout/mega_gate.cuh:31-43`）

路由参数打包成一个 POD：5 指针（`to_physical_map`/`logical_count`/`topk_idx`/`unmapped_topk_idx`/
`topk_weights`）+ 1 int64 stride + 4 uint32（`num_routed_experts`/`num_shared_experts`/
`num_duplicate_experts`/`rank_idx`）+ 1 float（`routed_scaling_factor`）。host 入口
`csrc/apis/mega_gate.hpp:160-172` 逐字段填充（可选项用 `? ptr : nullptr`）。本机核对：
`sizeof == 72`、指针块 `@0..40`、整数/浮点块 `@48..64`——`unmapped_topk_idx_stride` 后 4 字节
padding 使 `num_routed_experts` 落在 offset 48，是 host↔device ABI 的易错点。

### 3.5 配置搜索（`heuristics/mega_gate.hpp:106-187`）

Gate 的任务分解是三维枚举（`:114-126`）：

```
num_logical_ctas = num_expert_groups × num_mma_ctas × num_split_k
```

- `num_expert_groups`：`kNumExpertsPerGroup ∈ {128, 256}`（一个 UMMA M，`:115-120`）；
- `num_mma_ctas ∈ {1,2}`：256-expert group 必须 2-CTA（`:123-124`）；
- `num_split_k`：只到 `kNumMaxSplitK=8`，**deterministic 模式恒为 1**（`:110-111`，注释
  “deterministic mode is also batch invariant, so K is never split”——与第 13/47 篇的产能确定性一致）。

候选打分（`compare_mega_gate:89-100`）按词典序：先 **更长 TMEM 转置**（`num_token_cols` 越大越好）、
再多 SM、再 CTA pair、再少 split、再少 wave。`num_token_cols`（`:148-149`）受
`kNumMaxGateWarpgroups*kNumTransposeColsPerWarpgroup = 7*8 = 56` 封顶。
`num_gate_threads`（`:159-166`）在单 wave 时按「TMEM 转置单元数」与「top-k 轮数」两路取均衡，
多 wave 时直接拉满 7 个 warpgroup（896 线程）。

### 3.6 launch（`:69-89`）

```cpp
jit->launch(kernel, {
    .num_smem_bytes = config.smem_size,
    .grid_dim  = dim3(config.num_launch_sms, 1, 1),                         // 可能少于整卡 SM
    .block_dim = dim3(kNumNonEpilogueThreads + config.num_gate_threads, 1, 1), // 128 + gate
    .cluster_dim = dim3(config.num_mma_ctas, 1, 1),
}, tensor_map_x, tensor_map_weight, bias_ptr, image_bias_ptr, image_token_mask_ptr, mask_ptr,
   fix_routing_mask_ptr, force_random_ptr, routing_args, scratch.data_ptr(), score_barriers.data_ptr(),
   num_tokens, block_tokens, num_stages, num_worker_groups, num_token_blocks);
```

与 MoE 的两个不同：① **grid 不必用满整卡**（`num_launch_sms = min(worker_groups,token_blocks)*logical_ctas`，
`:146`，小 batch 会少开 SM）；② **cluster 随 `num_mma_ctas` 变**（1 或 2，`Allocator1Sm` vs `Allocator2Sm`，
`sm100_bf16_mega_gate.cuh:42`）。`num_stages`（`:170-171`）按 smem 反算，`kNumFixedSmemBytes =
kStageDataOffset`、每 stage 字节 = `(load_block_m+load_block_n)*BLOCK_K*2`。

## 4. H100 真机实测（`harness/exp_mega_moe_gate_jit_wrapper.{sh,out.txt}`）

`src/mega_moe_gate_jit_wrapper.cu` 用 nvcc 直接 include 真实头文件（`sm100_fp8_fp4_mega_moe.hpp` /
`sm100_bf16_mega_moe.hpp` / `sm100_bf16_mega_gate.hpp` + 两个 heuristics）与 torch，在 H100(sm90) 上跑
**架构无关的 host 逻辑**。同第 89 篇的一个编译期坑：真实头里的 `DG_UNIFIED_ASSERT` 在 nvcc 单 TU 的
host pass 会落成 `asm("trap;")`，本实验在 include 前把它钉到 host `assert`。**总判定 PASS=143 / FAIL=0**。

### 4.1 Part 1：FP8 vs BF16 配置不变量（4 形状 × 2 MmaKind × 10 断言）

真机 `get_mega_moe_config` 全过，关键数字（DeepSeek-V3 规模，ranks=8/E=256/topk=8/H=7168/I=2048）：

```
FP8 : block_m=240 block_n=128 block_k=128 store_l1=24 store_l2=8 sf_block=256x128
      num_sf_ring_tokens=61440 num_stages=5 smem=203380 num_bytes_per_pull=7168
BF16: block_m=240 block_n=128 block_k=64  store_l1=40 store_l2=40 sf_block=0x0
      num_sf_ring_tokens=0      num_stages=6 smem=226692 num_bytes_per_pull=3584
```

坐实的 10 条不变量：`block_k=128/elem_bytes`、`block_n=128`、`load_block_m=block_m/2`（A 轴多播）、
线程三分 `128/128/256`、swizzle 恒 128、`num_ring_tokens=ring`、BF16 无 SF（`sf_block=0`、`sf_ring=0`）、
FP8 SF 块 UTCCP 对齐（`align(block_m,128)×128`）、store block 阶梯
（`≤16→8, ≤64→16, ≤192→32, else FP8=24/BF16=40`；`block_m=240` 且 FP8 时 `store_l2=8`）、
`num_bytes_per_pull` 二分、`smem ≤ 232448 且 stages ≥ 2`。
另注意 `single-rank-small`（ranks=1/E=32/T=256）配置出 `block_m=64`，验证了 block_m 随
`num_expected_tokens` 动态收缩（`heuristics/mega_moe.hpp:100-110`）。

### 4.2 Part 2：FP8 描述符组合 + 共享回退（10 项）

- `sf_smem_outer_dim == block_k/(gran_k*4) == 1`（FP8）；
- `num_shared==0` 时 4 个共享描述符与 routed **128 字节逐字节相等**（指针别名等价）；
- 用**独立编写的 `cuTensorMapEncodeTiled` 编码器**对 3 个描述符互证，与 wrapper 输出**逐字节相等**：
  `l1_acts`（`hidden × ring`，128B swizzle）、`shared_l2_acts`（`I·S × T`）、
  `l1_acts_sf`（MN-major、int32、无 swizzle，box = `sf_block_m × 1`）；
- 从源码数出 JIT 微源码的模板占位符：FP8 = **25**、BF16 = **20**、Gate = **15**，与 kernel 声明一致。

### 4.3 Part 3：Mega Gate（常量 / ABI / 配置，含 deterministic 对照）

常量与几何：`BLOCK_K=64`、`kSwizzleMode=128`、`kNumNonEpilogueThreads=128`、`kNumMaxBlockTokens=256`、
`kNumMinBlockTokens=16`、`kExpertAlignment=128`、`kNumMaxLogicalCtas=64`、`kNumMaxStages=32`；
`sizeof(SharedStorage)=4656`、`kStageDataOffset=5120`（1024 对齐）。
`RoutingArgs` `sizeof=72`、offset 全对。scratch 公式逐项对上。
`caches_logical_count` 真值表（无 map 恒 false、bias+image 时按 4KiB 预算）。

四个形状 × deterministic off/on，config 全过，且**deterministic 把 split-K 打回 1**：

```
T=4096 H=7168 E=256  det=0: groups=1 ctas=2 splitK=1 block_tokens=64  token_cols=56 threads=512 smem=230400
T=64   H=7168 E=256  det=0: groups=2 ctas=1 splitK=8 block_tokens=16  token_cols=16 threads=256 smem=226304
T=64   H=7168 E=256  det=1: groups=1 ctas=2 splitK=1 block_tokens=16  token_cols=16 threads=256 smem=231424
T=1024 H=7168 E=384  det=0: groups=3 ctas=2 splitK=4 block_tokens=208 token_cols=56 threads=896 smem=220160
T=1024 H=7168 E=384  det=1: groups=3 ctas=2 splitK=1 block_tokens=48  token_cols=24 threads=384 smem=230400
```

小 batch（T=64）在 deterministic 关闭时用满 `splitK=8` 提速，打开时退回 `splitK=1`——与第 13/47 篇
「确定性模式从不 split K」一致；这也解释了 `check_topk_weights` 的逐位复现前提（第 47 篇 §4.3）。

## 5. 小结

- 四个 Mega 家族 JIT 包装层全部拆完（P79 的 BF16 MoE + mHC，本篇的 FP8 MoE + Gate）。它们都是
  **纯 host C++**，与 SM100 指令无关，H100 上直接 include 真实头文件即可验证全部 host 逻辑
  （本篇 **143 项 ALL PASS**）。
- FP8 Mega MoE 包装层的三个非平凡点：① **SF 描述符全 MN-major + 无 swizzle**，
  `sf_smem_outer_dim = block_k/(gran_k*4)`（打包 UE8M0 的 K 槽数）；② **共享专家的 E_s 折叠**——
  L1 N 轴 / L2 K 轴乘 `S`，且共享权重用 2D、路由权重用 3D 描述符；③ **18 个描述符**（更正第 89 篇
  的 20 计数），25 个模板实参（比 BF16 多 SF 冗余与 weight dtype）。
- Mega Gate 包装层的三个非平凡点：① **8 个模板 bool 全部由「指针/选项是否有值」推出**，
  与 kernel 的 `if constexpr` 分支天然一致；② **`RoutingArgs` POD 的 ABI**（`sizeof=72`，32/64 位
  字段交界处的 padding）；③ **任务分解三维枚举 + deterministic 禁 split-K**，grid 不必用满整卡、
  cluster 随 `num_mma_ctas` 变。
- 主 kernel 仍 **SM100 专属、仅分析**：`fp8_fp4_mega_moe` / `bf16_mega_gate` 在本机编译/运行都会失败
  （`mega_moe.hpp:367` 只在 arch10 派发；`mega_gate.hpp:159` 断言 `arch_major==10`）。

## 参考链接（真实）

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>
- Mega MoE PR：<https://github.com/deepseek-ai/DeepGEMM/pull/316>
- MoE backward PR：<https://github.com/deepseek-ai/DeepGEMM/pull/95>
- DeepEP 仓库：<https://github.com/deepseek-ai/DeepEP>
