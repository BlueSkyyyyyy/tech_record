# MoE 的 top-k 到底买到了什么：路由数 k 对前向/反向成本与 grouped GEMM 效率的完整扫描

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P178-1 增量（第 188 篇）**。
>
> 前面各篇反复用到 top-k：第 01 篇给出「K 倍放大」通信量模型（dispatch $\sim T\cdot d\cdot H$、
> combine $\sim T\cdot k\cdot H\cdot 2$），第 02/42 篇讲 `num_topk` 进对称缓冲，第 11 篇讲 dispatch
> 扫描 token-topk。但**从没把 top-k 当成一个自变量扫过**：直觉上 topk 越贵越线性（字节、FLOPs 都 ∝ k），
> 可生产里 DeepSeek-V3 偏偏用 **topk=8**。本篇把它拆开，发现 **topk 不只线性放大成本，还悄悄改了
> kernel 自己挑的 tile 与 grouped GEMM 的效率**——于是「每多一个 top-k」的边际代价不是简单线性。
>
> **约束**：Mega MoE 主 kernel（tcgen05/UMMA/UTCCP/NVLink 直写）是 **SM100 专属、仅分析**。
> 但 top-k 进入的**host 启发式**（`get_block_m_for_mega_moe` / `get_symm_buffer_size`）是真实
> pybind 绑定、**H100 可直接调用**；grouped GEMM 与通信量模型也都能在本机（H100 sm90）真跑/真算。
> 全部数字来自 `harness/exp_topk_cost.{py,sh,out.txt}`。
>
> 目标卡代号敏感：全文用「目标卡」指代。

参考版本：`~/github/DeepGEMM` commit `057ca59`（v2.8.1）；本机 `kernel_lab`（H100 sm90）。

---

## 1. top-k 进入算子的四个地方

先精确地说清「topk 一变，什么跟着变」。源码里 top-k 不只是循环上界：

### 1.1 专家容量与对称缓冲（∝ topk，被每 rank 专家数封顶）

`csrc/apis/mega_moe.hpp:48-50`：

```cpp
const auto num_experts_per_rank = num_experts / num_ranks;
const auto num_active_topk = std::min(num_topk, num_experts_per_rank);
const auto num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk;
```

ring 容量按 `num_max_routed_tokens` 迭代所有候选 BLOCK_M 取最大（`:55-63`）。layout 侧的池容量公式
`layout/mega_moe.cuh:21-28`：

```cpp
get_num_max_pool_tokens = align(num_ranks*num_max_tokens_per_rank*min(num_topk, experts_per_rank)
                                + experts_per_rank*(kMaxCandidateBlockM-1), kLCMCandidateBlockM);
```

即容量 ∝ `min(topk, num_experts_per_rank)`——**topk 超过每 rank 专家数后不再增长**。

### 1.2 每 token 的 topk 索引/权重缓冲（∝ topk）

`layout/mega_moe.cuh:396-397`：

```cpp
const auto input_topk_idx_layout     = layout::Data(num_topk * sizeof(int64_t), false);
const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float),  false);
```

按 `num_max_tokens_per_rank` 行分配（`:407-412`）。combine 的远端缓冲按 **top-k 号各一个 slot**
（`:447-449`，`num_topk + (共享专家?1:0)`）。

### 1.3 BLOCK_M 启发式（用「每专家期望 token 数」∝ topk 当输入）

`csrc/jit_kernels/heuristics/mega_moe.hpp:88-110` 是 host 端选 tile 的规则，第一行就把 topk 折进去：

```cpp
const float num_expected_tokens = (float)num_tokens * num_ranks * num_topk / num_experts;   // :94
...
int block_m = num_expected_tokens <= 10 ? 16 : 32;                                           // :100
if (num_expected_tokens > 24) {                                                              // :101
    int num_blocks = ceil(num_covered_tokens / 240);                                         // :102
    if (num_blocks == 1 and num_covered_tokens > 192 and num_experts / num_ranks < 14)
        num_blocks = 2;                                                                      // :103-104
    for (candidate : {64, 128, 192, 240}) { block_m = candidate;
        if (num_blocks * candidate >= num_covered_tokens) break; }                           // :105-109
}
```

`get_block_config_for_mega_moe` 返回 `{cluster=2, block_m, store_block_m_l1, store_block_m_l2,
block_k, num_epilogue_threads=256}`（`:126`），而它被 `get_block_m_for_mega_moe`（`mega_moe.hpp:21-30`）
经 pybind 暴露（`:394`）——**所以「topk → tile」这一步在 H100 上可以直接调用核对**。

### 1.4 kernel 内：dispatch 扫描与 combine 归约（都循环 kNumTopk）

`impls/sm100_fp8_fp4_mega_moe.cuh`（**SM100 专属、仅分析**）：

- dispatch warp 扫 `kNumTopk` 个 `(token, topk)`：`:347-366`（`kNumActivateLanes = kNumTokensPerWarp*kNumTopk`）、
  计数 `:369-371`、编码 `token_topk_idx` 写远端 `:384-390`；
- 拉取侧解码 `src_token_idx = src_token_topk_idx / kNumTopk`、`src_topk_idx = % kNumTopk`（`:531-534`），
  权重按 `+ src_token_topk_idx` 取（`:579-581`）；
- combine：**每个 top-k 号一个远端 slot**（`:1346-1351`），归约时「共享槽优先、其余按 top-k 号」
  （`:1449-1450`），在寄存器里 `float2` 累加全部 top-k 贡献（`:1471-1490`），再 cast bf16 + TMA 上屏
  （`:1493-1519`）。

于是 topk 通过四条路径影响成本：**容量/缓冲、tile 选择、字节、FLOPs**。前两条只在源码里能看，
后两条能真量。

---

## 2. Part A · host 启发式随 topk（H100 真机绑定，架构无关）

直接调 `deep_gemm.get_block_m_for_mega_moe(...)` 与
`deep_gemm._C.get_symm_buffer_size_for_mega_moe(...)`，并用 Python **逐行重写** `heuristics:88-127`
对拍。三种规模（`num_ranks=8`，`mma_type=fp8xfp4`，`H=7168, I=2048`）：

### 2.1 BLOCK_M 是 topk 的阶梯函数

`E=256, T=1024`（每专家期望 token $=T R k/E=32k$）：

| topk | 期望 token/专家 | `block_m` |
|---:|---:|---:|
| 1 | 32 | **64** |
| 2 | 64 | **128** |
| 3 | 96 | 128 |
| 4 | 128 | **192** |
| 6 | 192 | **240** |
| 8 | 256 | **192** |
| 12 | 384 | **240** |
| 16 | 512 | **192** |

`block_m` 随 topk 一路跨过 24/192 阈值（64→128→192→240），**且非单调**：topk=8、16 时
`num_covered_tokens > 240`，启发式改成「每专家 2/3 个块」，于是落回 192。`E=64, T=8192`
一开始就 `>192`，全为 240；`E=256, T=8192` 只在 topk 1→2 处 192→240。**Python 重写与 C++ 绑定
逐行 mismatch = 0**。

> 这解释了为什么 topk 不只是「循环多跑几次」：它把每专家的 M 维从「装不满一个块」推到「要几个块」，
> 而 Mega MoE 的注释（`:97-99`）说「每个 M 块约多花 128 行（重载权重）」，所以 tile 选择直接是性能。

### 2.2 对称缓冲字节数：先近线性、后饱和

`E=64, T=8192, H=7168, I=2048`：

| topk | num_bytes(MB) | Δ(MB) |
|---:|---:|---:|
| 1 | 653.51 | — |
| 2 | 1170.08 | +516.57 |
| 3 | 1686.64 | +516.57 |
| 4 | 2176.66 | +490.02 |
| 6 | 3183.25 | +1006.59 |
| 8 | 4216.41 | +1033.16 |
| 12 | 4686.57 | +470.16 |
| 16 | 5156.72 | +470.16 |

前期每 topk 约 +500 MB（≈线性），但 **topk 8→12→16 每 +1 topk 只 +118 MB**：ring 容量由
`get_num_max_live_pool_blocks(num_pool_blocks, num_sms, ...)`（`mega_moe.hpp:59-61`）封顶，
`num_pool_blocks` 涨到 SM 放不下的程度后 ring 不再变大。**对称缓冲不是无条件随 topk 线性膨胀**。

---

## 3. Part B · 真实 SM90 FP8 grouped GEMM 随 topk（H100 真机）

固定 `T=4096, H=7168, I=2048`，让每专家分到 `T·topk/E` 个 token，扫 topk。dgrad 用
`m_grouped_fp8_gemm_nt_contiguous`（M=token 分组，`gemm.hpp:181`），wgrad 用
`k_grouped_fp8_gemm_nt_contiguous`（K=token 分组，`gemm.hpp:373`）；dgrad 的 `m(aligned)` 与 wgrad 的
`sumK` 都取**对齐后**的行数/K（即 kernel 真正计算的量），`pad` = 对齐后 / 真实。

### 3.1 多专家（E=64）：topk 对 GEMM 时间几乎免费

| topk | tok/专家 | dgrad m(aligned) | pad | t(us) | TFLOPS | wgrad sumK | pad | t(us) | TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 8192 | 2.00 | 1353 | 178 | 8192 | 2.00 | 13303 | 18 |
| 2 | 128 | 8192 | 1.00 | 1353 | 178 | 8192 | 1.00 | 13308 | 18 |
| 3 | 192 | 16384 | 1.33 | 1570 | 306 | 16384 | 1.33 | 13565 | 35 |
| 4 | 256 | 16384 | 1.00 | 1576 | 305 | 16384 | 1.00 | 13564 | 35 |
| 6 | 384 | 24576 | 1.00 | 2235 | 323 | 24576 | 1.00 | 13989 | 52 |
| 8 | 512 | 32768 | 1.00 | 2671 | 360 | 32768 | 1.00 | 14441 | 67 |

归一化（基准 topk=1）：

| topk | dgrad t/t0 | dgrad FLOPs/t0 | wgrad t/t0 | wgrad FLOPs/t0 |
|---:|---:|---:|---:|---:|
| 4 | 1.17 | 2.00 | 1.02 | 2.00 |
| 8 | 1.97 | 4.00 | **1.09** | 4.00 |

**wgrad 的 K 翻了 4×、时间只涨 9%**：把 topk 从 1 加到 8，权重梯度几乎不多花时间。dgrad 时间也
只涨 1.97×（FLOPs 4×）。原因是两重的：① `pad` 从 2.00 收缩到 1.00（topk=1 时每专家 64 个 token
填不满 128 对齐，白算一半）；② wgrad 的 TFLOPS 从 18 被 topk 拉到 67（K 从 128 涨到 512，
K-block 流水更满）——正是第 135/145 篇「wgrad 不是 K-bound、是每专家固定开销/输出主导」的体现：
K 变大只是让那份固定开销摊薄。**E=64 时 topk 在 GEMM 侧近乎免费。**

### 3.2 少专家（E=8）：K/专家大，dgrad 反而近线性

| topk | tok/专家 | dgrad m | t(us) | TFLOPS | wgrad sumK | t(us) | TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 512 | 4096 | 414 | 291 | 4096 | 1772 | 68 |
| 2 | 1024 | 8192 | 667 | 361 | 8192 | 1802 | 133 |
| 4 | 2048 | 16384 | 1304 | 369 | 16384 | 2009 | 239 |
| 8 | 4096 | 32768 | 2449 | 393 | 32768 | 3389 | 284 |

归一化：dgrad t/t0 = **5.92×**（FLOPs 8×、已近计算受限 ~390 TFLOPS）、wgrad t/t0 = 1.91×
（FLOPs 8×、TFLOPS 68→284）。**E=8 时每专家 K 本就够大，topk 的边际是接近线性的 dgrad 成本**。

对比即结论：**topk 的 GEMM 边际代价强烈依赖 `E`（每专家 K）**——E 大时固定开销主导、topk 免费；
E 小时转向计算受限、topk 接近线性。生产（E=256、每 rank 32 专家）恰是「多专家」端。

正确性抽查（topk=4，单次 fresh 运行 vs 反量化参考）：`relerr = 3.16e-4`（隔离 kernel 误差，量级与
第 12/14 篇一致）。

---

## 4. Part C · 通信量：唯一严格线性 ∝ topk 的项

dispatch 每 rank 送出 `T·topk·H·1B`（fp8）、combine `T·topk·H·2B`（bf16），都 ∝ topk。
`T=4096, H=7168`：

| topk | dispatch MB | combine MB | total MB | ~us @393 GB/s |
|---:|---:|---:|---:|---:|
| 1 | 29.4 | 58.7 | 88.1 | 224 |
| 4 | 117.4 | 234.9 | 352.3 | 897 |
| 8 | 234.9 | 469.8 | 704.6 | 1793 |

（393 GB/s 是第 01 篇 8×H100 NCCL a2a 的渐近有效带宽，达 NVLink 上限 ~82%。）**通信是 topk 成本里
唯一严格线性、且不可被「效率提升」对冲的一项**——这正是 Mega MoE 要把 dispatch/combine 与计算重叠的
量化依据（第 02/144/165 篇）。

---

## 5. 综合：topk 的边际成本怎么分解

把三层放一起，回答「每多一个 top-k 要付什么」：

| 通道 | 随 topk | 本机实测 | 能否被效率对冲 |
|---|---|---|---|
| dispatch/combine 字节 | **严格线性** | k:1→8 total 88→705 MB | 否（只能靠重叠） |
| 对称缓冲 | 近线性→饱和 | 500 MB/topk → 118 MB/topk | 部分（ring 封顶） |
| BLOCK_M / tile | 阶梯 | 64→128→192→240→192 | 是（更大 K 更满） |
| dgrad | FLOPs 线性、效率随 K 升 | E=8: t 5.92× / E=64: t 1.97× | 部分 |
| wgrad | FLOPs 线性、效率随 K 升 | E=8: 1.91× / E=64: **1.09×** | 是（近免费） |

**一句话**：topk 的代价**主要是通信（严格线性）和 dgrad（近线性）**，而 **wgrad 与 tile 选择反而
随 topk 变好**；多专家（生产形态）把 wgrad 的边际压到近乎免费。这给「为什么 DeepSeek 敢用 topk=8」
一个逐项的账：它把成本堆在**本来就最该被重叠的通信**上，而不是最贵的 GEMM 上。

---

## 6. 与前面各篇的关系与边界

- **第 01 篇**（EP 通信）：本篇把它的「K 倍放大」模型用 topk 扫开，并指出 GEMM 侧非线性。
- **第 02 / 42 篇**（对称缓冲）：本篇给出 `num_bytes` 随 topk 的「先线性后饱和」实测。
- **第 135 / 145 篇**（wgrad 访存受限）：本篇的 E=64 wgrad t/t0=1.09× 是它的直接推论。
- **第 152 篇**（grouped 负载不均）：本篇的 `pad` 列与它「偏斜只付对齐税」同源。
- **第 166 篇**（偏斜路由的通信代价）：本篇的 Part C 是「均匀路由下通信 ∝ topk」的干净上界。

**边界**：① Mega MoE 主 kernel 的 topk 相关段（dispatch 扫描、combine 归约、UMMA）SM100 专属，
本篇只做源码坐标；② Part A 的 `block_m` 是**启发式**，实际 SM100 kernel 的 tile 还受 TMEM/SMEM 约束；
③ Part B 用 SM90 FP8 `nt` 入口、非目标卡 `tn`（但第 186 篇证明两者结构同构）；④ 吞吐为共享节点量级。

---

## 7. 参考

- DeepGEMM（v2.8.1，commit `057ca59`）：<https://github.com/deepseek-ai/DeepGEMM>
- host 容量/缓冲/启发式：`csrc/apis/mega_moe.hpp:21-30/35-63/394`、
  `csrc/jit_kernels/heuristics/mega_moe.hpp:88-127`
- layout：`deep_gemm/include/deep_gemm/layout/mega_moe.cuh:21-28/396-397/407-412/436-449`
- kernel（SM100 专属、仅分析）：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:347-390/531-608/1346-1519`
- grouped GEMM host 契约：`csrc/apis/gemm.hpp:181`（m_grouped）、`:373-427`（k_grouped）
- 实验：`harness/exp_topk_cost.{py,sh,out.txt}`
