# 生产级 MegaKernel 案例：源码坐标清单

> 本文是「MegaKernel 资料与实现分析」的第 3 篇。当前处于 **P0-2 侦察**阶段：先把三类生产/工程案例的
> **真实源码坐标**（仓库 commit、文件、核对过的行号）钉下来，作为后续 P3 精读的索引。
> 本文件随后会扩充为完整案例分析；骨架与坐标先行。
>
> 与本仓库 `../mega-moe/` 专题的分工：mega-moe 聚焦 DeepGEMM MoE 的**前向/反向实现细节**；
> 本文只从 **megakernel 范式**视角记录「融了什么、怎么调度、坐标在哪」。

## 0. 三类案例总览

| 案例 | 形态 | 代表坐标 |
|---|---|---|
| **DeepGEMM Mega MoE** | 段级 mega-kernel：EP dispatch + L1/L2 GEMM + SwiGLU + EP combine 单 kernel，重叠 NVLink 与 Tensor Core | `~/github/DeepGEMM` @ `057ca59` |
| **DeepGEMM Mega Gate / Mega mHC** | 段级 mega-kernel：MoE 路由打分 / HyperConnection（mHC）整段单 kernel | 同上 |
| **hpc-ops Megakernel** | 工程路线图条目（尚未实现）：把连续算子融成单 kernel 降 launch / 中间流量 | `~/github/hpc-ops` @ `ab1a402`，README 行 230 |
| **Kimi K3 fused mega-kernels** | 训练数据里的「内核优化任务套件」：从单算子到 fused mega-kernel | `~/github/ai-infra-book/references/text/kimi-k3.txt` §4.2.4 |

对照参考：**MPK**（Mirage Persistent Kernel，arXiv 2512.22219）是「整图自动 mega-kernel 化」的编译器+运行时，
区别于上述「手工段级 mega-kernel」。线上实现 `https://github.com/mirage-project/mirage`。

---

## 1. DeepGEMM Mega 家族（`~/github/DeepGEMM` @ `057ca59`）

> 核对基准：本地 clone 的 `git log --oneline -1` = `057ca59 Public release 26/09/30`。
> README 中三者的定位见行 3（总述）与行 14（2026.09.10 发布：Sparse Indexer、Mega Gate、Mega mHC、DeepJIT）。

### 1.1 Mega MoE

- **README 说明**：`README.md` 行 **112–144**（`#### Mega MoE`）——「fuses and overlaps EP dispatch,
  linear 1 and linear 2 (FP8xFP4 or FP8xFP8), SwiGLU, and EP combine into a single mega-kernel,
  overlapping NVLink communication and tensor core computation」；需多进程 + symmetric memory（行 114）。
- **Python API**：`deep_gemm.get_symm_buffer_for_mega_moe` / `transform_weights_for_mega_moe` /
  `fp8_fp4_mega_moe`（README 行 119–141；完整示例 `tests/test_mega_moe.py`）。
- **host 侧 API**：`csrc/apis/mega_moe.hpp`（400 行）
  - `fp8_fp4_mega_moe(...)` 入口 行 155；`bf16_mega_moe(...)` 行 285；ring capacity 计算 行 45–64。
  - 调试开关 `DG_COMM_KERNEL_DEBUG`（README 行 187）。
- **heuristics**：`csrc/jit_kernels/heuristics/mega_moe.hpp`（265 行）
  - `struct MegaMoEConfig` 行 23（block tiling / SF block / ring tokens / stages / thread 布局）；
  - `get_num_sms_for_mega_moe()` 行 67（**用满整卡 SM**，行 66 注释）；
  - `get_block_config_for_mega_moe(...)` 行 88（按每 expert 期望 token 数选 tile）。
- **JIT 入口**：`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp` 的 `sm100_fp8_fp4_mega_moe` 行 16；
  `csrc/jit_kernels/impls/sm100_bf16_mega_moe.hpp` 的 `sm100_bf16_mega_moe` 行 16。
- **device 实现**：
  - `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）：`SharedStorage` 行 196；
    dispatch send buffer / barriers 行 198–214；warp 角色划分（dispatch / MMA / epilogue）行 341 起；
    GEMM epilogue warps 行 955 起；task 的 phase 分派（Linear1/Linear2/SharedLinear1/SharedLinear2）行 697–770。
  - `deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_moe.cuh`（1316 行）。
- **调度器**：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（481 行）
  - `enum class BlockPhase` 行 83；`struct TaskInfo` 行 92；`struct L2KBlockDependency` 行 143；
    `struct MegaMoEScheduler` 行 180；L1→L2 死锁避免的 warmup wave 推导 行 15–37；
    **ring buffer** 容量上界 `get_num_max_live_pool_blocks` 行 47–75；
    `l2_full_mask` 位掩码驱动 L2 依赖 行 140–160。
- **布局 / symmetric memory**：`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`（459 行）
  - `struct TokenSrcMetadata`（combine 回写）行 43；`struct MegaMoESignals`（ring + NVLink 同步）行 54；
    `struct Workspace` 行 93；`struct Data` 行 256；`struct Buffer` 行 287；`struct MegaMoEBuffer` 行 345。

### 1.2 Mega Gate

- **发布坐标**：README 行 14（Mega Gate，2026.09.10）。
- **host 侧 API**：`csrc/apis/mega_gate.hpp`（196 行）；namespace `deep_gemm::mega_gate` 行 11；
  注意 CUDA graph 预热注释 行 19（「Warm up each stream before capture so one-time zeroing is not replayed」）。
- **heuristics**：`csrc/jit_kernels/heuristics/mega_gate.hpp`（189 行）。
- **JIT 入口**：`csrc/jit_kernels/impls/sm100_bf16_mega_gate.hpp` 的 `sm100_bf16_mega_gate` 行 15。
- **device 实现**：`deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh`（447 行）
  - persistent schedule 行 127；warp 角色：TMA load / MMA issue 行 133–212；**Gate warps** 行 246 起；
    每个 logical CTA 对 token 打分、再全局排名（top-k）行 322–420。
- **调度器**：`deep_gemm/include/deep_gemm/scheduler/mega_gate.cuh`（41 行）；`struct ScoreBarrier` 行 14。
- **布局**：`deep_gemm/include/deep_gemm/layout/mega_gate.cuh`（110 行）
  - `struct RoutingArgs` 行 31；`struct SharedStorage` 行 56；`struct Workspace`（global score scratch）行 80。

### 1.3 Mega mHC（HyperConnection）

- **发布坐标**：README 行 14（Mega mHC，2026.09.10）；HyperConnection 在总述 行 3。
- **host 侧 API**：`csrc/apis/mega_mhc.hpp`（287 行）；namespace `deep_gemm::mega_mhc` 行 11；
  核心 `mega_mhc(...)` 行 89；CUDA graph 预热注释 行 32。
- **JIT 入口**：`csrc/jit_kernels/impls/sm100_mega_mhc.hpp` 的 `sm100_mega_mhc` 行 16。
- **device 实现**：`deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh`（704 行）
  - 顶部注释：persistent mHC + static grid-stride roles 行 35–36；
    **静态角色映射** 行 185–190：「WG0 TMA/MMA, WG1-2 Post, WG3 workspace epilogue, WG4 Mix, WG5 Norm」。
- **调度器**：`deep_gemm/include/deep_gemm/scheduler/mega_mhc.cuh`（55 行）；`struct SplitBarrier` 行 14；
  Norm 依赖 Shifted 物化 X1 的注释 行 51。
- **布局**：`deep_gemm/include/deep_gemm/layout/mega_mhc.cuh`（170 行）
  - `struct MixArgs` 行 45；`struct NormArgs` 行 58；`struct Workspace`（split-K partial scratch）行 81；
    `struct SharedStorage` 行 122（Full/empty pairs 连接静态角色，行 143）。

### 1.4 一处观察（供 P3 展开）

DeepGEMM 的三个 mega kernel 都采用**「持久化 CTA + 静态角色 + 设备内同步（mbarrier / 信号）/ ring buffer」**
的结构：CTA 一次 launch、角色固定（dispatch/MMA/epilogue 或 WG0–WG5），靠设备内信号而非 host launch 推进。
这正是 megakernel 范式在**段级**（MoE 段、Gate 段、HC 段）的工程落地，与 MPK 的**整图 + 编译器生成 +
去中心化动态调度**形成「手工静态 vs 自动动态」的对照。

---

## 2. hpc-ops Megakernel roadmap（`~/github/hpc-ops` @ `ab1a402`）

- 核对基准：`git log --oneline -1` = `ab1a402 update hpc-ops build to C++20 (#76)`。
- README 行 **230**（Roadmap 段）：
  「**Megakernel**: Fuse multiple consecutive operators into a single kernel to reduce inter-kernel
  launch overhead and intermediate memory traffic.」——仍是**路线图条目（未实现）**，
  定位为「降 launch 开销 + 降中间张量流量」，与 P1 动机一致。
- 已实现的相邻手段（可作为段融合的工程证据）：`Fused MoE`（README 行 103–105、表 行 162）、
  `Fused AllReduce + Residual + RMSNorm`（行 120–132、表 行 163）、`Fused sampler`（行 137、表 行 164）。

## 3. Kimi K3 fused mega-kernels（`~/github/ai-infra-book/references/text/kimi-k3.txt`）

- 章节 **§4.2.4 Kernel Optimization Tasks**，快照行 **1069–1081**。
- 关键点（原文核对）：
  - 行 1070：「a large-scale suite of kernel tasks ranging from single-operator kernels to **fused
    mega-kernels**, sourced from high-quality GitHub repositories such as Flash Linear Attention」；
  - 行 1071–1073：覆盖 CUDA / Triton / CuTe DSL / Gluon / ThunderKittens / TileLang 与 BF16/FP8/FP4；
  - 行 1074–1078：奖励 = 正确性（对齐 PyTorch 参考，超误差阈值得 0）+ 性能（对齐专家实现得 0.5，
    接近硬件 roofline 逼近 1）；
  - 行 1078–1081：**防作弊**明确惩罚「CUDA graph replay、input caching、precision reduction」——
    说明在评测语境里 CUDA Graph 被视作「绕过真实优化」的手段，这与本专题 P4「CUDA Graph 治 launch
    开销但不改变串行性」的定位相互印证。
- 用途：为 P3-3 提供「fused mega-kernels 已成为模型能力训练任务」的产业侧证据。

---

## 4. 下一步（承接 P3）

- **P3-1** 精读 Mega MoE：dispatch / ring buffer / L2 依赖（`scheduler/mega_moe.cuh` 行 140–170、
  行 376–430）与 symmetric memory 布局（`layout/mega_moe.cuh`），与 `../mega-moe/` 交叉引用避免重复。
- **P3-2** Mega Gate / Mega mHC 各写清「融了哪些算子、为什么、性能动机」。
- **P3-3** 汇总 Kimi K3 §4.2.4 与 hpc-ops roadmap 的工程视角。

> 说明：本文件坐标均在本机 clone 上逐行核对；如远端更新，行号可能漂移，以对应 commit 为准。
