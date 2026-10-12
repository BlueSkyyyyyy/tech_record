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

### 1.5 P3-1 精读：Mega MoE 的 megakernel 范式

> 本节只回答 megakernel 范式问题：**「一个 kernel 如何覆盖整段 MoE 前向」「谁在设备内调度」「如何不靠
> host launch 推进」**。算子级数据流、SM100 tcgen05 / FP8xFP4 / UE8M0 等硬件细节见
> `../mega-moe/docs/02-mega-moe-architecture.md`，本文不重复。行号均在 `~/github/DeepGEMM` @ `057ca59`
> 逐行核对。

#### (1) 单 kernel 边界：五段融进一次 launch

MoE 前向朴素实现是五段串行：EP dispatch → linear1 → SwiGLU → linear2 → EP combine。Mega MoE 把五段
**全部塞进一次 kernel launch**（README 行 112–114：「fuses and overlaps EP dispatch, linear 1 and linear 2,
SwiGLU, and EP combine into a single mega-kernel」）。判据 C1（单次 launch）在段级成立：host 侧只负责
准备对称显存 buffer 并调用 `fp8_fp4_mega_moe`（`csrc/apis/mega_moe.hpp:155`），kernel 内部不再有跨算子
launch。与 MPK 的差别在**范围与生成方式**：MPK 由编译器把整张图切 task 并生成（`docs/01-mpk-paper.md` §3），
Mega MoE 是手工写死的**段级** mega-kernel（判据 C3 = 段级多算子，非整图）。

#### (2) 静态角色划分：warp 即「算子专属执行单元」

kernel 用 `warp_idx` 一次性把执行单元划成固定角色（`sm100_fp8_fp4_mega_moe.cuh:342` 起）：
- **dispatch warps**：`warp_idx < kNumDispatchWarps`（默认 `kNumDispatchThreads=128` → 4 warp，
  heuristics 行 220），持久拉取远端 token / 统计 expert 计数 / 写 combine 元数据（行 342–689）。
- **non-epilogue warps**（`kNumNonEpilogueThreads=128` → 4 warp，heuristics 行 221）：
  - `warp_idx == kNumDispatchWarps`：GEMM **TMA load A**（activations+SFA，行 690）；
  - `+1`：GEMM **TMA load B**（weights+SFB，行 756）；
  - `+2`：**MMA issue**（仅 leader CTA 发 UMMA，行 819）；
  - `+3`：**in-kernel scheduler mainloop**（仅 leader CTA，行 940）。
- **epilogue / combine warps**：`warp_idx >= kNumDispatchWarps + kNumMMANonEpilogueWarps`（行 946 起），
  按 `epilogue_wg_idx` 再分 WG。

每类角色用 `warpgroup_reg_dealloc/alloc` 分配不同寄存器预算，并静态断言总数 ≤ 64K（行 326–335）——这就是
「SM 上角色固定、寄存器按最重角色封顶」的手工版，对应 MPK §7「寄存器固定为各 task 最大值」的边界。
**关键点**：`warp 7` 是设备内调度器，对应 MPK 的 scheduler SM（`docs/01-mpk-paper.md` §4）——即
megakernel 必须有**一个不吃 Tensor Core 的调度角色**在设备内推进工作，而不是 host。

#### (3) 设备内协调：静态任务表 + ring buffer + 位掩码依赖

调度器 `sched::MegaMoEScheduler`（`scheduler/mega_moe.cuh:180`）以 **`TaskInfo`**（行 92）为单位下发任务，
`BlockPhase` 枚举（行 83–89）区分 `Linear1/Linear2/SharedLinear1/SharedLinear2`。与 MPK 的 tGraph 相似，
依赖被显式建模，但这里是**手工的两层**：

- **L1→L2 细粒度依赖**：每个完成的 L1 N block 用 `red_xor` 翻转 `l2_full_mask` 的一个 bit，L2 的每个 K block
  只需等自己那一段 bit 就绪（`L2KBlockDependency`，行 140–166）。这是「输出 tile 重叠即依赖」的手工特例。
- **ring buffer 容量上界**：`get_num_max_live_pool_blocks`（行 48–80）给出「同时活着」的 pool block 上界，
  host 侧 `mega_moe.hpp:41–64` 镜像同一公式来分配 ring 显存；ring 容量保证 `l2_full_mask` 的奇偶 generation
  语义成立（行 141 注释）。
- **L1→L2 死锁规避**：交错调度会让 L1 任务落后，若无 warmup wave 则 L2 可能先占满 SM 等不到 L1。
  `get_num_l1_warmup_waves`（行 15–45）用闭式公式推导出必须**先集中发 L1** 的最小 wave 数。
- **负载均衡**：locality domain + work stealing（`kNumMinStealWaves` 行 223–228；`get_next_task_idx` 行 363–370）。
- **combine 就绪**：用每次 launch 唯一的 `grid idx` 标记 peer 是否写完（`layout/mega_moe.cuh:70–72`），
  避免跨 launch 的信号复位。

数据缓冲区同样用对称显存切片：`layout::Buffer::get_rank_buffer`（`layout/mega_moe.cuh`）按 rank 切分，
dispatch/combine 读远端、写本地都走同一套布局。

#### (4) 通信-计算重叠怎么落进单 kernel

重叠不是靠多流，而是靠**(2) 的静态角色并发**：dispatch warps 持续做 NVLink pull（行 433 起，`nvlink_barrier` /
`grid_sync` 只在阶段边界同步），与此同时 MMA/epilogue warps 持续吃本地 L1 ring 做 GEMM/SwiGLU——两组 warp
在同一批 SM 上同时跑，NVLink 延迟被 Tensor Core 计算掩盖。这与 MPK §5.1「事件驱动、异构互连无需拓扑画像」
的目标一致，但 MPK 由 tGraph event 自动触发，Mega MoE 由固定的阶段边界（dispatch 统计 → grid sync →
pull → ring 满 count）手工编排。

#### (5) 与 MPK 的逐维对照（段级手工 vs 整图自动）

| 维度 | DeepGEMM Mega MoE | MPK |
|---|---|---|
| 融合范围 | **段级**（MoE 前向五段，C3 部分成立） | **整图**（每 batch size 一张 tGraph） |
| 单次 launch | 是（C1） | 是（C1） |
| 角色划分 | **手工静态**（warp 编号写死，行 342–960） | 编译器生成 task，worker/scheduler 固定两类 |
| 依赖表示 | 手工：位掩码 `l2_full_mask` + ring generation | tGraph task/event（自动 dependency analysis + fusion） |
| 调度 | 设备内 `MegaMoEScheduler`（warp 7）静态两层交错 + work stealing | 去中心化事件驱动，JIT/AOT 混合下发 |
| host 职责 | 仅 launch + 预热/准备 buffer | 仅 launch；图与 codegen 全在编译期 |
| 硬件相关 | 整个 kernel（SM100 专属，含 UMMA/TMA） | 仅 per-task codegen |

结论：Mega MoE 是 megakernel 范式在**段级**的成功工程落地，验证了「单 launch + 设备内角色 + ring buffer
调度 + 通信计算重叠」在真实生产模型上的可行性与收益；它没有回答的是 MPK 的**自动整图**问题——依赖仍是
手工编码、融合范围仍是段。两者恰好构成「手工静态段级 → 自动动态整图」的完整谱系两端（见 `docs/00` §4 T3/T4）。

#### (6) 落地约束

需**多进程 + 对称显存**（README 行 114）、SM100 专属、`intermediate_hidden*2` 与 `hidden` 须是
`2×128` 的整数倍（`scheduler/mega_moe.cuh:56–57` 的静态断言）。这些约束正是「段级手工 mega-kernel」
相对 MPK「编译器自动生成」在通用性上的代价。

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

- **P3-1** ✅ 已精读 Mega MoE 的 megakernel 范式，见 §1.5（单 kernel 边界 / 静态 warp 角色 / ring buffer +
  位掩码依赖 / 通信-计算重叠 / 与 MPK 逐维对照）。算子级细节仍以 `../mega-moe/` 为准。
- **P3-2** Mega Gate / Mega mHC 各写清「融了哪些算子、为什么、性能动机」。
- **P3-3** 汇总 Kimi K3 §4.2.4 与 hpc-ops roadmap 的工程视角。

> 说明：本文件坐标均在本机 clone 上逐行核对；如远端更新，行号可能漂移，以对应 commit 为准。
