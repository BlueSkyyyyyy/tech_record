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

#### P3-2(1) Mega Gate 精读：融了哪些算子、为什么

Mega Gate 是 MoE **路由段**的段级 mega-kernel（判据 C1/C2/C4/C5 成立，C3 为段级）。host 侧
`bf16_mega_gate`（`csrc/apis/mega_gate.hpp:62`）一次调用即完成「打分 GEMM + 整个 top-k 路由」，其函数注释
（行 28–60）列出的融合范围是：

朴素路由链至少是 5–6 个算子（每层 MoE 各一遍）：
1. **gate GEMM**：`scores[T,E] = x[T,H] @ weight[E,H]ᵀ`（BF16，`sm100_bf16_mega_gate.cuh:168` 的 UMMA 主循环）；
2. **scoring 激活**：`sqrt(softplus(·))`（`epilogue/sm100_mega_gate.cuh:19–37` 的 `sqrt_softplus`）；
3. **bias / image_bias 加**（`sm100_bf16_mega_gate.cuh:379–399`，`smem.metadata_cache` 行 89–91）；
4. **mask / force_random / fix_routing** 分支（行 326–357）；
5. **top-k 选择**（`select_warp_topk`，`epilogue/sm100_mega_gate.cuh:107–152`；warp 内 4 元素排序网络 +
   3-bit 游标仲裁）；
6. **权重归一化 + 缩放**（`store_topk_token`，行 187–216：`score / topk_sum * routed_scaling_factor`）；
7. **logical→physical expert 映射**（重复 expert 打散到各 rank，行 198–208）+ 输出 `topk_idx/topk_weights`。

**为什么要融（性能动机）**：这一段的瓶颈是**中间张量规模 + launch 次数**，而非 FLOPs。
- 打分矩阵 `scores[T, num_aligned_experts]` 若物化，是 `T × E_aligned × 4B`。以测试用到的
  `T=32768, E=512`（`tests/test_mega_gate.py` 的 `enumerate_cases`）估算约 **64 MB 写 + 64 MB 读**
  （估算值，非实测）；N 个 MoE 层叠加后是可观的纯 HBM 往返——正是 `docs/00` §1.4 的损失三。
- 路由段是**低算术强度**的一小段（`E` 维 GEMM + 大量 reduce/排序/分支），kernel-per-operator 会把
  「1 次 GEMM launch + 1 次 top-k launch + 若干归一化/映射算子」逐个 launch，落入损失二的固定开销；
  而它每层都执行，launch 次数被层数放大。
Mega Gate 把 7 个算子压成**一次 launch**：GEMM 结果只经 TMEM→global score scratch 一次落盘，随后 gate
warps 就地把分数读回做 top-k，不再有「独立打分 kernel → 独立 top-k kernel」之间的 kernel barrier 与中间物化。

#### P3-2(2) 单 kernel 边界与静态角色

- **单次 launch**：host 只准备 scratch/score-barrier 并 launch；`sm100_bf16_mega_gate.hpp:69–75` 的
  `jit->launch` 用 `grid_dim = num_launch_sms`、`block_dim = kNumNonEpilogueThreads + num_gate_threads`、
  `cluster_dim = num_mma_ctas` 一次发出。device 入口 `__launch_bounds__` 在 `sm100_bf16_mega_gate.cuh:32`。
- **静态角色**（`sm100_bf16_mega_gate.cuh`，warp 编号一次写死）：
  - `warp 0`：**TMA load**（x 与 weight，行 134–167，persistent 按 `worker_group_idx` grid-stride 扫 token block）；
  - `warp 1`（仅 leader CTA）：**MMA issue**（行 168–244，`SM100_MMA_F16BF16_SS` / `2x1SM`）；
  - `warp_idx >= kNumNonEpilogueWarps`：**Gate warps**（行 245–432），负责 TMEM transpose
    （`sm100_store_tmem_scores_to_global`，epilogue 行 41）与逐 token top-k。
  - 常量见 `layout/mega_gate.cuh:12–31`（`kNumNonEpilogueThreads=128`、`kNumEpilogueStages=2`）。
- **persistent schedule**：行 127 注释「the `kNumLogicalCtas` CTAs of a worker group stride over the token
  blocks together」，行 128 取 `worker_group_idx`——一个 token block 由 `num_logical_ctas =
  num_expert_groups × num_mma_ctas × num_split_k` 个 CTA 协作（heuristics 行 133），CTA 常驻扫多个 token block
  （C2）。这就是 MPK「worker/scheduler 固定、按 task 推进」的手工段级版，只是这里没有独立 scheduler warp，
  **调度被编排成 role 内固定的 grid-stride 循环**。

#### P3-2(3) 设备内协调：score barrier + 原子的 grid 标签

- **ScoreBarrier**（`scheduler/mega_gate.cuh:14–39`）：一个 token block 的**所有 logical CTA** 各 arrive 一次，
  wait 端观察到全部到达后才让 gate warps 排名。`arrive` 用 `red_async_inc_rel`（行 32，GPU 级异步原子自增），
  `init/wait_init` 用 `grid_idx` 编码状态位（行 17–29），从而**跨 launch 复位不靠 host**——与 Mega MoE 的
  `l2_full_mask` generation、mHC 的 SplitBarrier 是同一套「grid 标签 + 异步原子」手法。
- **split-K 归约**：打分 GEMM 可切 K（`num_split_k` 到 8，heuristics 行 54），各 split 写自己的
  `[token, split, expert]` 切片（`layout/mega_gate.cuh:79–102` 的 `Workspace`），gate warps 在
  `load_scores`（device 行 304–321）里按 token 求和后再激活——把「sqrt-softplus 非线性 + 跨 split 求和」
  的先后顺序固化进 kernel（行 284–286 注释）。
- **元数据缓存**：bias/image_bias/logical_count 先一次性搬进 4 KiB 的 `metadata_cache`
  （`layout/mega_gate.cuh:46–51`、行 64），避免 ranking 循环里反复读 global。

#### P3-2(4) 计算-通信/计算-访存重叠

Mega Gate 没有跨卡通信，但重叠的是**Tensor Core 与 gate 逻辑**：MMA warp 推进第 i+1 个 token block 的
UMMA（行 182–237）时，gate warps 正在排名第 i 个 block 的分数（行 262–428），两组 warp 在同一批 SM 上
并发；TMA 又在为更后面的 stage 预取（`num_stages` 由 heuristics 行 170–174 按 SMEM 容量算）。
这与 MPK §5.1「事件驱动、计算与下一任务重叠」的目标一致，但由**固定 stage 环形缓冲 + TMEM full/empty
barrier**（`tmem_full_barriers`/`tmem_empty_barriers`，行 279、297）手工编排。代价是 `kNumGateWarps`
有限（heuristics 行 49–53：最多 7 个 warpgroup），当 `num_waves > 1` 时直接用满（行 164），说明
「gate 逻辑与 MMA 争 SM」正是这里的资源约束。

#### P3-2(5) 与 MPK 的逐维对照

| 维度 | DeepGEMM Mega Gate | MPK |
|---|---|---|
| 融合范围 | **段级**（MoE 路由段 7 算子） | 整图 |
| 单次 launch | 是（C1） | 是（C1） |
| 角色划分 | 手工静态：warp0 TMA / warp1 MMA / gate warps；调度编进 grid-stride 循环（无独立 scheduler warp） | 编译器生成 task；worker/scheduler 两类固定角色 |
| 依赖表示 | 手工：`ScoreBarrier` + split-K scratch 归约顺序 | tGraph task/event（自动） |
| 调度 | 设备内 static grid-stride + score barrier 原子 | 去中心化事件驱动、JIT/AOT 混合 |
| 重叠 | Tensor Core（下一 block）与 gate 排名（当前 block） | 事件驱动的跨 task 流水 |
| 硬件相关 | 整 kernel（SM100 UMMA/TMA/TMEM） | 仅 per-task codegen |

#### P3-2(6) 落地约束

SM100 专属（`mega_gate.hpp:159` 断言 `arch_major == 10`）；`hidden % 256 == 0`（行 80 的
`kHiddenAlignment`）；`num_routed_experts ≤ 512 且 %4==0`（行 84–85）；`num_physical_topk ≤ 32`（行 109）；
`num_tokens ≤ 2^20`（行 156，`layout` 行 19）。此外所有 logical CTA 必须 co-resident 且不超 64 个
（heuristics 行 134），否则无法用 score barrier 协调。这些静态约束正是「手工段级 mega-kernel」相对
MPK「编译期按形状自动建图」在通用性上的代价。

> 说明：Mega Gate 为 SM100 专属，本机 H100/sm90 无法端到端运行，本节为**仅资料分析**
> （行号均在 `~/github/DeepGEMM` @ `057ca59` 逐行核对）。

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

#### P3-2(1) Mega mHC 精读：融了哪些算子、为什么

Mega mHC 把 HyperConnection 的**整个层内步骤**融进一个持久化 kernel（判据 C1/C2/C4/C5 成立，C3 段级）。
host 侧 `mega_mhc`（`csrc/apis/mega_mhc.hpp:89`）的函数注释（行 41–44）一句话概括：
「Execute shifted or normal mHC, generate the next mixes, and apply RMSNorm with optional FP8 E4M3 casting」。
展开后融合了：

1. **HC pre-mix GEMM**：`fn[num_hc_outputs, hc_mult*H] @ [T, hc_mult*H]ᵀ → [T, num_hc_outputs]`，TF32
   （`sm100_mega_mhc.cuh:233–285` 的 `SM100_MMA_TF32_TS`；`kNumRoutes=4`, `kNumHCOutputs=24`，
   `layout/mega_mhc.cuh:11–12`）；
2. **Post 混合**：`post_mix` 与 `comb_res_mix`（跨 route 组合）对 `x`/`residual` 的加权（
   `sm100_mega_mhc.cuh:484–508`，`__fmul2_rn`/`__ffma2_rn`）；
3. **Sinkhorn 归一化**（comb logits 的行/列归一化，`epilogue/sm100_mega_mhc.cuh:63–88`，
   `num_sinkhorn_iters` 次交替归一）；
4. **残差更新**：写出 `new_residual`（行 344–347 的 `tma_store_3d`）；
5. **RMSNorm**：对新的 HC 输出做均方归一 + 权重（`run_normal_worker` / `run_shifted_norm_worker`，
   epilogue 行 141、394）；
6. **可选 FP8 E4M3 量化 + 打包 scale factor**（`store_fp8_and_pack_sf`，epilogue 行 124），
   以及 Mega MoE 共享专家 layout 的 SF（api 行 84–87）；
7. **生成下一层的 mix 系数** `new_prev_mix/new_post_mix/new_comb_res_mix`（`run_mix_task`，epilogue 行 17）。

**为什么不融会亏（性能动机）**：HyperConnection 在残差流上额外复制 `hc_mult=4` 条并行通道，使
`x`/`residual`/`new_residual` 等张量都是 `[T, 4, H]`，体量约为普通残差的 4 倍。这一段是
**访存主导 + 小 GEMM**：pre-mix GEMM 的 N 维只有 24（`kNumHCOutputs`），算术强度极低，kernel-per-operator
会把它和 post/comb mix、Sinkhorn、RMSNorm、FP8 量化拆成 5–7 个 kernel，每个都读写一份十几 MB 的
`[T,4,H]` 中间张量，并叠加多次 launch——损失一/二/三同时命中。融成单 kernel 后：GEMM 与 Post/Mix/Norm
按 split 流水重叠，FP8 量化也不再需要独立 kernel（下游 Mega MoE 直接吃 `y_fp8` + SF）。

#### P3-2(2) 单 kernel 边界与静态角色

- **单次 launch**：`sm100_mega_mhc.hpp:129–146` 一次 launch，`grid_dim = num_sms`、`block_dim = kNumThreads`
  （`= 6 × 4 × 32 = 768`，`layout/mega_mhc.cuh:31`）。device 入口 `__launch_bounds__(kNumThreads, 1)` 行 39。
- **静态角色映射**（`sm100_mega_mhc.cuh:185`）：**WG0**（4 warp）做 TMA/MMA，**WG1–2** Post，
  **WG3** workspace epilogue，**WG4** Mix，**WG5** Norm。WG0 内部再细分（行 212–384）：
  - `warp 0`：residual + X 的 TMA load（行 212–223）；
  - `warp 1`：Fn 的 TMA load（行 224–232）；
  - `warp 2`：HC MMA（行 233–285）；
  - `warp 3`：输出 store + 系数预取 + Norm 到达（行 286–385）。
- **寄存器重配置**：每种静态角色用 `warpgroup_reg_dealloc/alloc` 分配不同预算（行 88–99）：
  control 32 / Post 136（shifted）或 112 / lightweight 40 / shifted-Norm 96 / normal 80，
  并 `DG_STATIC_ASSERT(kMaxRegisterBudget * 128 <= 64512)`（行 98）——这正是 MPK §7「寄存器固定为各
  task 最大值」的手工版；注释（行 105–107）特别说明运行时索引在每个 role 分支内重算，以免值跨重配置存活而 spill。
- **split-K**：一个 task = 一个 `(m_block, K-split)` 对（行 28–31 注释），split 数由
  `get_num_splits`（api 行 15–24）按「最少 split 且达到最短最长 task」选；deterministic 模式固定 16。

#### P3-2(3) 设备内协调：双 SplitBarrier + 静态 Full/empty 环

- **SplitBarrier**（`scheduler/mega_mhc.cuh:13–49`）：`Norm = SplitBarrier<0>`、`Mix = SplitBarrier<1>`
  （行 52–53，两条独立 barrier 线）。语义是「某 m-block 的所有 split 各 arrive 一次，wait 端看到全部
  到达才放行」。`arrive` 同样用 `red_async_inc_rel`（行 40），`init/wait_init` 用 `grid_idx` 编码
  （行 20–34）——与 Mega Gate `ScoreBarrier` 完全同构，只是 barrier 数量与 domain 不同。
- **依赖顺序被编码进 barrier**：顶部注释（行 35–36）「Norm receives one arrival only after Shifted X1 stores
  and statistics are committed」；Post 统计先经 `full_stats_barrier`（行 365、542）汇合，再由 WG3 归约后
  触发 `Mix::arrive`（行 646–653）——所以 Mix/Norm 的启动条件是**手工声明的**，而非自动依赖分析。
- **SMEM 的 Full/empty 环形屏障**（`layout/mega_mhc.cuh:143–167`）：`full_io/empty_io`、`full_fn/empty_fn`、
  `full_a/empty_a`、`full_accum/empty_accum`、`full_coeff/empty_coeff` 成对把静态角色连成流水线，
  **无需 CTA 级 block 边界同步**（注释行 143）；跨 CTA 只用两条 gmem SplitBarrier。这是「生产者-消费者
  warp 组」在同一 CTA 内的手工 tGraph。

#### P3-2(4) 计算-访存/计算-计算的流水与重叠

- **WG0 内部**：TMA（warp0/1）→ MMA（warp2）→ 输出 store（warp3）三级流水，靠 `kNumIOStages=4`、
  `kNumFnStages=2`、`kNumAStages=4`、`kNumAccumStages=2`（`layout/mega_mhc.cuh:25–29`）交错多个 task。
- **跨 WG**：WG1–2 的 Post 计算与 WG0 的 MMA 通过 `full_a/empty_a` barrier 重叠（行 475–530），
  WG3 的 accumulator drain 又与下一个 task 的 MMA 重叠（`accum_pipeline`，行 580–629）。
- **系数预取**：`should_prefetch_next_coeff`（行 334–339）在最后一个 K block 提前取下一 task 的
  post/comb 系数（行 354–355），并注释「avoid competing with the task's main TMA wave」——把访存尽量
  塞进 store-drain 窗口。
- **Normal 模式的角色复用**：Normal 不需要 Shifted 的 Mix/Norm 队列，于是**回收已退休的 WG**
  （行 668–682）立即投入 fused Normal worker，并用 `next_norm_task_ticket`（`layout` 行 167）做
  互斥 task 队列——同一批 CTA 在一个 kernel 内**换角色**，进一步压 launch 与空转。

#### P3-2(5) 与 MPK 的逐维对照

| 维度 | DeepGEMM Mega mHC | MPK |
|---|---|---|
| 融合范围 | **段级**（HC 层：pre-mix GEMM + mix + Sinkhorn + RMSNorm + FP8 量化） | 整图 |
| 单次 launch | 是（C1） | 是（C1） |
| 角色划分 | 手工静态 WG0–WG5，且 Normal 模式**回收退休 WG 复用** | 编译器生成 task；worker/scheduler |
| 依赖表示 | 手工：两条 `SplitBarrier`（Norm/Mix）+ 一套 SMEM Full/empty 环 | tGraph task/event（自动） |
| 调度 | 设备内 static grid-stride + 异步原子 barrier + 寄存器重配置 | 去中心化事件驱动、JIT/AOT 混合 |
| 重叠 | TMA/MMA/Post/epilogue 多级流水 + 系数预取 + store-drain 窗口 | 事件驱动跨 task 流水 |
| 硬件相关 | 整 kernel（SM100 TF32 UMMA/TMA/TMEM/FP8） | 仅 per-task codegen |

#### P3-2(6) 落地约束

SM100 专属（`mega_mhc.hpp:223` 断言 `arch_major == 10`）；`hc_mult` 必须等于 `kNumRoutes == 4`
（行 137–138）；`hidden % (16 × 64) == 0`（行 141，`kDefaultNumSplits × BLOCK_K`）；
`num_tokens ≤ 2^20`（行 140）；`y_bf16` 与 `y_fp8` 至少给一个（行 121），FP8 还必须配 SF（行 122）。
`SharedStorage` 必须塞进 SMEM（`sm100_mega_mhc.hpp:53–54` 断言）。这些约束与 Mega Gate/MoE 同源：
**手工段级融合换来的通用性损失**（维度固定、layout 写死），正是 MPK 想用编译器自动化掉的。

> 说明：Mega mHC 为 SM100 专属，本机 H100/sm90 无法端到端运行，本节为**仅资料分析**
> （行号均在 `~/github/DeepGEMM` @ `057ca59` 逐行核对）。

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

> 核对基准：`git log --oneline -1` = `ab1a402 update hpc-ops build to C++20 (#76)`。行号均在本机 clone
> 逐行核对。

### 2.1 roadmap 定位（未实现）

README 行 **230**（Roadmap 段）：
「**Megakernel**: Fuse multiple consecutive operators into a single kernel to reduce inter-kernel
launch overhead and intermediate memory traffic.」——仍是**路线图条目（未实现）**，
定位恰好命中 `docs/00` §1 的**损失二（launch 开销）+ 损失三（中间张量 HBM 往返）**，只差损失一的一体化调度。
它被排在 Extended Quantization、Next-Gen Hardware、Low-Precision Communication 同级，说明业界仍把
megakernel 当作**待攻克的下一阶段**，而非已落地手段——这与本专题「段级已落地、整图仍研究中」的判断一致。

### 2.2 已实现的相邻段融合（工程证据）

hpc-ops 虽然没实现「megakernel」，但已经把相邻算子做了**段级融合**，且用的是 PDL/双 kernel 而非持久化单 kernel，
恰好是谱系 T2/T3 的工程样本：

- **Fused MoE**（README 行 103–115；表 行 162，`hpc/fuse_moe.py`）：把 **routing + Gate-Up GEMM +
  activation quantization + Down GEMM + top-k weighted reduction** 融成「一条 pipelined execution path」。
  两点值得记：① 用 routing index **直接读原始 token**，省掉独立 gather 段（行 107–109）——与 DeepGEMM
  Mega MoE 的「fused gather-GEMM」动机同源（`docs/00` §1.4 / MPK §6.4）；② 低延迟区**移除 Warp
  Specialization**、改用 cp.async，把延迟隐藏从「CTA 内软件流水」下移到「**跨 CTA 硬件调度**」（行 112–114），
  再用 **PDL 把各阶段串起来**减少 launch bubble（行 115）。这说明产业界选择的是「多 kernel + PDL」而非
  「持久化单 kernel」——因为段内还有 host 友好的起步点。
- **Fused AllReduce + Residual + RMSNorm**（行 120–132；表 行 163，`hpc/allreduce.py`）：把
  AllReduce + 残差加 + RMSNorm 融成 NVLink-native kernel。**高吞吐模式**用 CUDA multicast 服务大 token
  prefill；**低延迟模式**用 Lamport P2P **双 kernel** 设计 + PDL overlap 服务小 token decode（行 129–130）。
  这是 T3（通信-计算重叠）——注意它**仍是多 kernel**，重叠靠 PDL/host，不是设备内 scheduler。
- **Fused sampler**（行 137–151；表 行 164，`hpc/sampler.py`）：把 repetition penalty / temperature /
  top-k / top-p / softmax / random sampling / penalty mask update 折成 **2 个 CUDA kernel**
  （外加 temperature-only 快路径），用 shared-memory 计数、local heap/merge 选择等减少 global 往返与固定开销。

### 2.3 工程侧结论

hpc-ops 的路线图与已实现项共同画出一条清晰的演进台阶：**单算子 → 段融合（Fused MoE / AllReduce+RMSNorm /
sampler，PDL + 多 kernel）→ [roadmap] megakernel（连续算子并成单 kernel）**。它落在谱系 T2/T3，
而 §1 的 DeepGEMM mega 家族已经站上 T4（段级），MPK 则指向 T4（整图自动）。三者恰好覆盖
`docs/00` §4 的完整谱系，且 hpc-ops 明确把「megakernel」写成下一步，是「段融合已成熟、整段单 kernel 仍在推进」
这一判断的产业侧旁证。

## 3. Kimi K3 fused mega-kernels（`~/github/ai-infra-book/references/text/kimi-k3.txt`）

> 章节 **§4.2.4 Kernel Optimization Tasks**，快照行 **1069–1081**（本机逐行核对）。

### 3.1 原文要点

- **任务谱宽**（行 1069–1071）：Kimi K3 为强化 GPU kernel 优化能力，构建了「a large-scale suite of kernel
  tasks ranging from **single-operator kernels to fused mega-kernels**」，数据来源为高质量 GitHub 仓库
  （如 **Flash Linear Attention**）。即：**fused mega-kernel 已经作为一种「能力标尺」写进大模型的训练任务集**。
- **编程模型/精度覆盖**（行 1071–1073）：CUDA、Triton、CuTe DSL、Gluon、ThunderKittens、TileLang，
  精度 BF16 / FP8 / FP4——与本专题关注的手写 CUDA（DeepGEMM）与编译器路径（Triton/TileLang）同域。
- **奖励函数**（行 1073–1076）：每个 kernel 给一份 PyTorch 参考实现；数值误差超阈值 **得 0**（正确性硬门限）；
  性能对齐专家实现得 **0.5**，逼近硬件 roofline 才向 **1** 逼近。即评估**同时压正确性与性能上限**，
  不是「跑了就行」。
- **防作弊**（行 1076–1078）：明确惩罚 reward-hacking 策略——「**CUDA graph replay、input caching、
  precision reduction**」。
- **模型能力外溢**（行 1062–1066，视觉推理段佐证同一套 agent 环境思路）：K3 在带 Python 沙箱的 agent 环境里
  迭代写码/执行/验证，说明这里的 kernel 任务也是「在同一类闭环里自我提升」的训练信号。

### 3.2 对本专题的两个关键印证

1. **CUDA Graph 被评测系统视作「绕过真实优化」**（行 1077）。这与本专题 P4 的定位完全一致：CUDA Graph 只
   消除 launch 开销、**不改变 kernel 间的串行依赖**，因此若用它刷分就等价于「用 host 侧 replay 掩盖没有真正
   融合/重叠」，必须判为 hacking。反过来也说明：**megakernel 的价值被定义为「真融合 + 真重叠」，而非
   launch 次数**——与 `docs/00` §4 判据 C4/C5（调度进设备、SM 级）同调。
2. **精度缩减被判作弊**（行 1077）从侧面确认：megakernel 的性能收益应来自**调度/访存/重叠**，而不是降低数值
   精度；这也呼应 MPK §7「手写 kernel 可包成 task」与 DeepGEMM 各 mega kernel 都坚持 BF16/FP8 精度语义。

3. **产业需求侧信号**：把「single-operator → fused mega-kernels」做成规模化训练套件，意味着**「会写
   mega-kernel」已成为被显式训练与考核的模型能力**，与 hpc-ops roadmap、DeepGEMM 三件套构成需求、路线、
   落地三层证据链。

### 3.3 与 DeepGEMM / MPK 的对照

| 维度 | Kimi K3（训练任务套件） | DeepGEMM mega 家族 | MPK |
|---|---|---|---|
| 角色 | **需求/能力侧**：模型要会写 mega-kernel | **落地侧**：手工段级 mega-kernel | **方法侧**：编译器自动整图 |
| 融合范围 | 任务从单算子覆盖到 fused mega-kernel | 段级（MoE / Gate / mHC） | 整图 |
| 编程模型 | CUDA/Triton/CuTe/Gluon/TK/TileLang | 手写 CUDA（SM100 专属） | 编译器 + per-task codegen |
| 评价 | 正确性硬门限 + roofline 打分 + 防作弊 | 生产性能（重叠 NVLink/Tensor Core 等） | 端到端 1.0–1.7× |
| 对 CUDA Graph 态度 | **惩罚 replay**（视作绕过） | 用 CUDA graph 仅做过预热/捕获 | 超越 CUDA Graph（SM 级细粒度） |

### 3.4 小结（P3-3）

Kimi K3 §4.2.4 给出的是**需求侧**证据：融合 mega-kernel 已从「论文里的整图执行模型」下沉为「被规模
训练的模型技能」，且评测系统用「防 CUDA graph replay / 防精度缩减」把「真融合」与「假绕过」区分开。
结合 §2 的 hpc-ops 路线图（把 megakernel 列为下一步）与 §1 的 DeepGEMM 三件套（段级 T4 已落地），
可以给出一条工程结论：**megakernel 的产业进展呈现「需求（K3 训练）→ 路线（hpc-ops roadmap）→ 落地
（DeepGEMM 段级）→ 自动化（MPK 整图）」四段式**，本专题后续 P4（相关技术对照）与 P6（汇总）将据此展开。

---

## 4. 下一步（承接 P3 / 转入 P4）

- **P3-1** ✅ Mega MoE megakernel 范式，见 §1.5（单 kernel 边界 / 静态 warp 角色 / ring buffer + 位掩码依赖 /
  通信-计算重叠 / 与 MPK 逐维对照）。算子级细节仍以 `../mega-moe/` 为准。
- **P3-2** ✅ Mega Gate（§1.2：路由段 7 算子融合 / ScoreBarrier + split-K 归约 / Tensor Core 与 gate 排名重叠）
  与 Mega mHC（§1.3：HC 层 pre-mix GEMM + mix + Sinkhorn + RMSNorm + FP8 量化融合 / 双 SplitBarrier +
  静态 Full-empty 环 / Normal 模式回收退休 WG）。
- **P3-3** ✅ Kimi K3 §4.2.4（需求/能力侧 + 防作弊对 CUDA Graph 的判定）与 hpc-ops roadmap（路线图 + 已实现的
  PDL 段融合）工程视角，见 §2/§3。**P3 全部完成（3/3）**。
- **下一步 P4**：相关技术对照 `docs/03-related-tech.md`——CUDA Graphs（P4-1）、PDL（P4-2，H100 实测）、
  persistent kernel vs megakernel + in-kernel 调度（P4-3）、与通信-计算 overlap（DeepEP/FlashMLA）对比（P4-4）。

> 说明：本文件坐标均在本机 clone 上逐行核对；如远端更新，行号可能漂移，以对应 commit 为准。
