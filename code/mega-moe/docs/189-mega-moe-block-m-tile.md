# Mega MoE 的 BLOCK_M 到底买到什么：一个 M-tile 的真实代价（H100 代理 GEMM 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P179-1 增量（第 189 篇）**。
>
> Mega MoE 主 kernel 沿每个专家的 token 数 M 切 **`BLOCK_M`**，这个值由 host 启发式
> `get_block_config_for_mega_moe`（`csrc/jit_kernels/heuristics/mega_moe.hpp:88-127`）从候选集
> `{8,16,32,64,96,128,192,240}` 里挑，规则的核心是一句注释（`:97`）：
> 「**每个 M block ≈ 128 行的额外代价**（它要重载权重 tile），所以每个专家用**尽量少的块**、
> 用能覆盖这些行的**最小 tile**」。第 178 篇（`docs/188`）只量了这个启发式的**输出**（topk 变时
> `block_m` 怎么跳阶梯），**从没量过 tile 本身的时延**。本篇补上这一格。
>
> **约束**：Mega MoE 主 kernel（UMMA/UTCCP/NVLink 直写、240 行 tile）是 **SM100 专属、仅分析**。
> 但「一个 M block 真实要花多久」可以用 H100 上**真实 SM90 FP8 grouped GEMM** 当代理量出来：
> `m_grouped_fp8_fp4_gemm_nt_masked` 与 L1/L2 同形（A=`[tokens,H]`、B=权重），且它的 `expected_m`
> 形参能**强制 `block_m`**（SM90 masked 候选只有 `{64,128}`，`heuristics/sm90.hpp:34-35`）。
> 全部数字来自 `harness/exp_block_m_tile.{py,sh,out.txt}`。
>
> 目标卡代号敏感：全文用「目标卡」指代。

参考版本：`~/github/DeepGEMM` commit `057ca59`（v2.8.1）；本机 `kernel_lab`（H100 sm90）。

---

## 1. 源码事实：BLOCK_M 从哪来、kernel 用它数出多少块

### 1.1 host 启发式：候选集 + 「每块 ≈128 行」的代价模型

`csrc/apis/mega_moe.hpp:21-31` 把 `get_block_m_for_mega_moe` 暴露给 Python（`deep_gemm.get_block_m_for_mega_moe`），
它就是 `heuristics/mega_moe.hpp:88-127`：

```cpp
// heuristics/mega_moe.hpp:94-114
const float num_expected_tokens = static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts;
const float num_covered_tokens = num_expected_tokens + std::sqrt(num_expected_tokens);
// Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest blocks
// per expert and the smallest tile covering them: one block up to 192 rows, then blocks of up to 240 rows.
// A single 240-row block per expert only pays off with many local experts (about 14 or more blocks per rank).
int block_m = num_expected_tokens <= 10 ? 16 : 32;
if (num_expected_tokens > 24) {
    int num_blocks = static_cast<int>(std::ceil(num_covered_tokens / 240));
    if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
        num_blocks = 2;
    for (const int& candidate: {64, 128, 192, 240}) {
        block_m = candidate;
        if (num_blocks * candidate >= num_covered_tokens)
            break;
    }
}
```

候选集的权威来源是 `deep_gemm/include/deep_gemm/layout/mega_moe.cuh:13-16`：

```cpp
static constexpr int kNumCandidateBlockMs = 8;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192, 240};
```

（`kMaxCandidateBlockM=240` 在 `:15`，`kLCMCandidateBlockM=1920` 在 `:17`，后者是 token 池的对齐粒度，
见 `get_num_max_pool_tokens :19-28`。）

规则可以读成一句**代价模型**：对每个专家，块数 $b_e=\lceil m_e / B\rceil$，总代价 $\approx
\sum_e \big(\text{padding 到 }B + 128\big)\cdot b_e$——即「块越多，权重重载越多」。

### 1.2 kernel 侧：块数 = $\sum_e \lceil m_e/\text{BLOCK\_M}\rceil$

主 kernel 侧这不是估计而是**精确计数**。`scheduler/mega_moe.cuh` 里调度器按每专家 token 数把 token
池切成 M block：

```cpp
// scheduler/mega_moe.cuh:270-282 —— 每个专家的池块偏移
uint32_t num_blocks = 0;
for (uint32_t i = 0; i < kNumExpertsPerLane; ++ i) {
    if (i * 32 + ptx::get_lane_idx() < expert_idx)
        num_blocks += math::ceil_div(stored_num_tokens_per_expert[i], BLOCK_M);   // :277
}
return __reduce_add_sync(0xffffffff, num_blocks);
```

`get_num_total_pool_blocks()`（`:283-285`）就是 $b=\sum_e \lceil m_e/\text{BLOCK\_M}\rceil$，
随后 `:301-302`：

```cpp
num_total_m_blocks = get_num_total_pool_blocks();
const uint32_t num_total_l1_tasks = num_total_m_blocks * kNumL1Clusters;
```

每个 M block 是一个 `BlockPhase::Linear1`（`:380-386`，`create_task(..., L1_SHAPE_N, L1_SHAPE_K)`），
它要 TMA 载入该专家对应的**权重 tile**。所以「块数」直接等于「权重重载次数」——`:97` 注释说的就是这个。
`owner_valid_m`（`:332-334`）只用 `valid_m = min(num_tokens - m_block_idx*BLOCK_M, BLOCK_M)` 裁尾，
即**块内 Padding 不额外补算**，但权重 tile 是按整块载的。

---

## 2. Part A · 启发式复刻（真实绑定 vs Python 重写）

`harness/exp_block_m_tile.out.txt` Part A 用 Python 逐行重写 `:88-127`，对 `R=8,E∈{64,256},T∈{8192,1024},topk∈{2,4,8}`
九组配置调真实 `deep_gemm.get_block_m_for_mega_moe` 对拍：

| exp_tok/expert | covered | py_bm | C++ bm |
|---|---|---|---|
| 2048.0 | 2093.3 | 240 | 240 |
| 512.0 | 534.6 | 192 | 192 |
| 64.0 | 72.0 | 128 | 128 |
| 128.0 | 139.3 | 192 | 192 |
| 256.0 | 272.0 | 192 | 192 |

`mismatch=0`。规则本身没问题，问题只是：**它假设的「每块 ≈128 行」在真机上是不是真的**。

---

## 3. Part B · tile 代价曲线：块内行几乎免费，跨块才跳一次

用真实 SM90 FP8 `m_grouped_fp8_fp4_gemm_nt_masked`，**一个专家**（G=1），L1 形状
（$K=H=7168$，$N=2I=4096$，权重 $29.4$ MB/专家），用 `expected_m` 强制 `block_m∈{64,128}`，
扫 M=1…256。表（单位 μs，`#blocks=ceil(M/bm)`）：

**L1，强制 `block_m=128`：**

| M | 1 | 16 | 32 | 64 | 80 | 96 | 112 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| #blocks | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 |
| t (μs) | 93.9 | 94.5 | 94.0 | 94.0 | 93.1 | 93.1 | 93.1 | 92.9 | 102.6 | 102.5 | 102.7 |

**L1，强制 `block_m=64`：**

| M | 1 | 16 | 32 | 48 | 64 | 80 | 96 | 112 | 128 | 160 | 192 | 256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| #blocks | 1 | 1 | 1 | 1 | 1 | 2 | 2 | 2 | 2 | 3 | 3 | 4 |
| t (μs) | 93.7 | 93.8 | 93.4 | 91.1 | 91.1 | 99.3 | 99.4 | 99.4 | 99.4 | 107.2 | 107.2 | 195.7 |

**读法**（这是本篇最硬的观察）：

1. **一个 block 内，行几乎是免费的**：`block_m=128` 时 M 从 1 涨到 128（128 倍行数），时延从
   93.9 μs 到 92.9 μs——**完全不变**。一个块的代价 = 把该专家的权重（29.4 MB）流一遍（$\sim$93 μs
   $\Rightarrow$ $\sim$316 GB/s，单专家并行度低所以远低于峰值），**块内行被这次权重流掩盖**。
2. **每个额外块加一次「重载税」**：`block_m=128` 从 1 块到 2 块（M 128→160）跳 **+9.6 μs**；
   `block_m=64` 第 2 块 **+8.2 μs**、第 3 块 **+7.8 μs**。这个增量 ≈ 29.4 MB 的**二次读取**（此时权重
   已在 50 MB L2 里）——$\sim$29.4 MB / 9.6 μs $\approx$ **3.1 TB/s**，正是 L2 带宽量级。
3. **第一次触碰权重的代价（93 μs）远大于后续重载（$\sim$9 μs）**：说明启发式说的「块=权重重载」在
   **L2 层**成立，但**首块**才是大头。这正是 Part D 里「多专家」灾难的来源。

L2 形状（$K=I=2048$，$N=H=7168$，权重 $14.7$ MB）同构、更轻：`block_m=128` 首块 $\sim$87.8 μs 且
M=1…128 全平，额外块 **+5.4 μs**；`block_m=64` 额外块仅 +2.9/+3.1 μs（权重更小，重载更便宜）。

**与 `:97` 的「≈128 行」对照**：真机实测的**代价形态**与注释一致——「块 = 一次权重重载」；
但具体常数不能照搬：H100 代理里额外一块 ≈ +9.6 μs，折算成行（按大 M 行率 $\sim$0.39 μs/行）≈ 25 行，
而首块 ≈ 240 行。`:97` 的「128」是**目标卡 SM100 mega kernel 的标定值**，用 UMMA/TMEM 的 weights
pipeline 与 NVLink 布局，H100 代理给不出同一个常数（见 §6 局限）。本篇能给的是**模型的形状**：
「块内近免费 + 每块一次重载税」。

---

## 4. Part C · block_m 的收益：同活儿块越少越快

固定同一份工作，只换 `block_m`（`expected_m` 强制），L1 形状：

| M | bm=64（2 块起） | bm=128 | 加速 |
|---|---|---|---|
| 96  | 99.2 μs | 93.1 μs | 1.07× |
| 128 | 99.3 μs | 93.1 μs | 1.07× |
| 160 | 107.3 μs | 102.5 μs | 1.05× |
| 192 | 107.2 μs | 102.5 μs | 1.05× |
| **256** | **192.8 μs** | **103.8 μs** | **1.86×** |

M=256 时 `bm=64` 要 4 块、`bm=128` 只要 2 块，于是 **1.86×**。这坐实了启发式「用能覆盖这些行的
**最大** tile（在候选里）来减少块数」的动机——只有当 M 恰好跨过块边界时收益才显现；M 落在一块内时，
`block_m` 大小几乎不影响时间（1.05–1.07× 只是配置差异）。

---

## 5. Part D · 多专家税：为什么启发式怕「many local experts」

`:99` 那句「a single 240-row block per expert only pays off with many local experts」讲的是**每个专家
一份独立权重**：专家越多，第一次触碰权重（首块 93 μs 那种）就越多。把**固定总行数=512** 摊给
G 个专家（每专家 = 一份独立权重，M=512/G），L1 形状：

| G | M/专家 | 总行 | t (μs) | TFLOPS | 相对 G=1 |
|---|---|---|---|---|---|
| 1 | 512 | 512 | 201.9 | 149 | 1.00× |
| 2 | 256 | 512 | 193.9 | 155 | 0.96× |
| 4 | 128 | 512 | 206.7 | 145 | 1.02× |
| 8 | 64 | 512 | 324.6 | 93 | 1.61× |
| 16 | 32 | 512 | 560.5 | 54 | 2.78× |
| **32** | **16** | 512 | **1118.4** | **27** | **5.54×** |

G 从 1 涨到 32（同样 512 行 → 32 份 16 行），**慢 5.54×**、TFLOPS 从 149 掉到 27。原因就是 Part B 的
第 1 条：每份权重无论用几行都要流一遍，16 行的专家和 512 行的专家付**一样**的权重流代价。
这正是生产里「每 rank 专家数」很重要的量化背景（DeepSeek-V3 每 rank $\sim$32 专家，`num_experts/num_ranks`
进 `:106` 的 `<14` 判断，以及第 178 篇量到的 `block_m` 台阶）。

---

## 6. 结论、边界与真机证据

**结论**：

1. `BLOCK_M` 本质上不是「GEMM tile 大小」变量，而是**权重重载次数**变量。kernel 侧块数
   `num_total_m_blocks = Σ_e ceil(m_e/BLOCK_M)`（`scheduler/mega_moe.cuh:277/301`）就是重载次数。
2. 真机（H100 代理）给出 **`:97` 代价模型的形状**：**块内行几乎免费**（M 1→128 时延不变），
   **每个额外块一次重载税**（L1 $\sim$+9.6 μs、L2 $\sim$+3–5 μs，L2 带宽量级），且**首块**
   （从 HBM 冷读整份权重）远贵于后续块。
3. 因此启发式的两条规则都被真机支持：① 「用能覆盖的**最小** tile、块**尽量少**」（Part C：M=256
   时 bm128 vs bm64 = 1.86×）；② 「**多专家**时别用超大 tile」（Part D：G 32 vs 1 = 5.54×，
   因为每专家首块代价无法摊薄）。
4. `:97` 里「≈128 行」是目标卡 SM100 mega kernel 的标定常数，H100 代理**量不出同一个数**，
   只能确认模型形态。

**边界 / 局限**：

- SM90 `m_grouped_*_masked` 的 `block_m` 候选只有 `{64,128}`（`heuristics/sm90.hpp:34-35`），
  无法覆盖 mega MoE 的 `{8,16,32,96,192,240}`；本篇是**两点**代理，不是全曲线。
- 代理用 SM90 WGMMA + 密接 masked 布局，未复刻 SM100 的 UMMA/TMEM、环形 pool、NVLink 直写、
  2-CTA cluster 与 locality domain——**只做「一个 M block 的代价形态」的对照**，不冒充主 kernel 性能。
- 8 卡节点共享，报数已多次重复取 min，仍可能与独占基线有差异。
- 正确性抽查：代理 kernel vs 反量化 fp32 参考 **relerr 1.473e-3**（仅 bf16 输出舍入），确保量的
  是真 GEMM 而非空跑。

**真机证据台账**（`harness/exp_block_m_tile.{py,sh,out.txt}`，H100 80GB HBM3 / sm90 / DeepGEMM 2.8.1）：

| 项 | 数字 |
|---|---|
| 启发式复刻 vs 真实绑定 | 9 组配置 `mismatch=0` |
| L1 `bm=128` 块内 M=1→128 | 93.9 → 92.9 μs（**不变**） |
| L1 `bm=128` 每额外块 | **+9.6 μs**（$\approx$ L2 重载 29.4 MB / 3.1 TB/s） |
| L1 `bm=64` 每额外块 | +8.2 / +7.8 μs |
| L2 `bm=128` 每额外块 | **+5.4 μs**（权重 $14.7$ MB） |
| block_m 收益（M=256） | bm64 192.8 μs vs bm128 103.8 μs = **1.86×** |
| 多专家税（512 行，G=1→32） | 201.9 → 1118.4 μs = **5.54×**（149→27 TFLOPS） |
| 代理 kernel vs 反量化参考 | relerr **1.473e-3** |

主 kernel SM100 专属、仅分析；以上代理数字均来自 H100 真机 SM90 FP8 grouped GEMM。
