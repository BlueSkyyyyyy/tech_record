# MegaKernel 一页纸总结：原理、设计 checklist 与适用边界

> 本文是「MegaKernel 资料与实现分析」的收官篇（P6-1）。它不再展开新资料，而是把前面五篇
> （动机与谱系 `docs/00`、MPK 精读 `docs/01`、生产案例 `docs/02`、相关技术对照 `docs/03`、
> 实验 `docs/04`）压缩成三件可直接用的东西：**原理一页纸、设计 checklist、何时该/不该用**。
> 引用 MPK 论文快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt` 的真实行号；
> 本机实测均标注目标卡（H100 / sm90）。与 `../mega-moe/` 专题的 MoE 前反向实现细节不重复。

## 30 秒版本

- **动机（三笔账）**：kernel barrier 挡住跨算子流水、每 token 数百次 launch 的固定开销、中间张量
  HBM 往返。三者同源——以「整个算子」为推进单位、由 host 反复 launch。
- **定义（五条判据）**：单次 launch、常驻、跨算子融合、设备内协调、SM 级调度（C1–C5），
  分水岭是 **C4/C5**。
- **收益锚点**：Qwen3-8B/A100 上 vLLM/SGLang **14.5 ms/token** → MPK **12.5 ms**，硬件下界约
  **10 ms**；README 对外宣称 **1.2×–6.7×**。
- **代价**：常驻占住 SM、图必须静态、寄存器按最大 task 预留、设备内调度一旦挂起就没有报错。
- **什么时候值得**：小模型 + 新卡 + 大量小算子串成的链；大 batch、GEMM-bound 的负载收益有限。

## 1. 原理一页纸

### 1.1 要解决什么：kernel-per-operator 的三类系统性损失

主流执行方式是「一个算子一个 kernel」，按依赖顺序 launch 到同一 stream。它带来三类损失
（`docs/00` §1，MPK §1 行 74–89）：

| 损失 | 症状 | 量化锚点（论文报告值） |
|---|---|---|
| **损失一 kernel barrier** | 同 stream 相邻 kernel 间隐式全屏障，禁止跨算子软件流水与细粒度重叠 | 跨任务流水关闭→开启降 **1.2–1.3×**；细粒度计算-通信重叠关闭→开启降 **1.1×**（MPK §6.6 行 1586–1615） |
| **损失二 launch 开销** | 每 token 数百到上千次 launch 的固定成本 | Qwen3-8B 每 token **293** 次 launch；eager **3.8 µs→1.1 ms**、Graph **0.8 µs→0.2 ms**；MPK in-kernel scheduler 仅 **0.28%**（§6.6 行 1618–1628） |
| **损失三 中间张量 HBM 往返** | 上游算子写回 device memory、下游再读回 | MoE 的 gather 预处理在 SGLang 上占 MoE 总时间 **高达 11%**（§6.4 行 1551–1553） |

三类损失同源：**以「整个算子」为推进单位、由 host 反复 launch**。越小的模型、越新的 GPU，固定开销
占比越大（MPK §6.3 行 1501–1513）；端到端锚点 Qwen3-8B/A100：vLLM/SGLang **14.5 ms/token**、
MPK **12.5 ms**，硬件下界约 **10 ms**（§6.3 行 1513）。

### 1.2 定义与判据

**MegaKernel**：把（一整段甚至整个模型的）多个算子**融合进一个持久化 kernel**——kernel 一次
launch、在 GPU 上长期驻留，内部用**去中心化的 in-kernel 调度**把工作分给不同 SM/warp，
并在 kernel 内实现**跨算子软件流水**与**通信-计算重叠**（`docs/00` §3.1）。

五条判据（`docs/00` §3.2）——**同时满足 C1–C5 才是真 MegaKernel**，分水岭是 C4/C5：

| 判据 | 含义 |
|---|---|
| **C1 单次 launch** | 整段/整图只调用一次 kernel |
| **C2 常驻（persistent）** | kernel 全程驻留，SM 不因 kernel 边界清空 |
| **C3 多算子融合范围** | 覆盖跨算子依赖链，而非单个/几个相邻算子 |
| **C4 设备内协调** | 用 device memory 上的信号/ring/原子跨 SM 推进，而非 host 逐段 launch |
| **C5 SM 级调度** | 调度粒度细到 SM/线程块级 task，能表达跨算子流水与通信-计算重叠 |

一句话：**persistent 是机制（C1+C2），mega 是范围（+C3）与调度（+C4/C5）**（`docs/03` §3.1）。

### 1.3 机制：MPK 的两层结构（自动路径的代表）

MPK 是首个把多 GPU 推理**自动**编译成单 mega-kernel 的编译器 + 运行时（`docs/01`）。

**编译层**（MPK §3–§4，快照行 345–1146）：把算子图编译成 **SM 级 tGraph**——节点是
**task**（一个 SM 上的工作单元）与 **event**（任意 task 子集的屏障），task/event 交替、事件驱动
（§3 行 360–370）。关键洞察：**同一计算图有多种 tGraph**，质量取决于 event 插得是否 precise
（图 4(c) 反例，行 387–401）——收益不来自「合进一个 kernel」，而来自**依赖切得够细**。
五步流水线：operator decomposition（每算子 32–47 task，行 462–468）→ dependency analysis
（输出 tile 重叠即插 event，行 469–519）→ event fusion（两种集合融合，事件减少 **37–118×**，
行 520–779）→ normalization（空 task 把扇入/扇出压到 1，<1%，行 780–951）→ linearization
（BFS 让同一 event 触发的 task 索引连续，footprint 压 **4.4–15×**，Algorithm 1 行 952–1125）。
§4.2 以 **thread block** 为单位复用 Mirage superoptimizer 生成 CUDA（行 1126–1146）。

**运行层**（MPK §5，快照行 1147–1368）：一个持久化 kernel 内，SM 固定切成少数 **scheduler**
（每 SM 4 warp、共 16，占 4 SM）与多数 **worker**（每 SM 一个，维护 JIT/AOT 双 task queue），
角色在 launch 时定死。**事件驱动**：task 完成→通知 triggering event→event 激活→入 scheduler
队列→派发下游 task（§5.1）。计算与通信统一成同一 tGraph 的 task，**无需拓扑画像**，异构互连
自适应（§5.1 行 1218–1225）。**JIT/AOT 混合**：数据相关耗时的 operator（如 attention）标 JIT
以均衡负载，global barrier 之后标 AOT 以降下发时延（§5.2 行 1296–1302）。三项优化：
paged shared memory（32 KB/页，解锁跨 task 流水，§5.3）、cross-task software pipelining
（拆成 pre-loading + compute 重叠，**1.2–1.3×**）、task descriptor 预取（352 B/份，§5.3）。
`torch.compile(backend=MPK)` 即可接入（§6.1 行 1393–1395）。

### 1.4 收益从哪来：损失 ↔ 手段的因果链

| 收益 | 机制 | 实测/报告锚点 |
|---|---|---|
| 消除 launch 开销（损失二） | 单次 launch + 设备内调度；host 不再逐 kernel 提交 | 本机 P5-1：persistent 总时间恒定 ~2 µs，N=4096 达 **1988×**；MPK 293 launch→0.28% |
| 消除 kernel barrier / 跨算子流水（损失一） | 无 kernel 边界 + 事件驱动 task 交错 | MPK pipeline **1.2–1.3×**；本机 P5-2 证明「单 kernel 内协调」在目标卡便宜（~1.65 µs/次） |
| 消除中间张量 HBM 往返（损失三） | 融合后中间结果留片上/复用 | 本机 P5-2：fused vs barrier **~2.0×**（16→8 B/elem） |
| 通信-计算重叠（损失一的多设备特例） | 通信与计算同为 task、同一 scheduler | MPK overlap **1.1×**；DeepEP/FlashMLA 停在 kernel 级 |

关键校准（本专题实测，`docs/04`）：**三类收益各自只在对应区间显形**——launch 收益只在
launch-bound 区间（P5-1）、损失三在带宽主导区间（P5-2）、PDL/重叠量 = `min(tail, prologue)`
且**不随链长累积**（P5-3）。不要把它们混记在一笔账上。

### 1.5 谱系与两种落地路径

四档谱系（`docs/00` §4）：T1 单算子内优化 → T2 局部/段融合 → T3 通信-计算重叠 →
T4 整图/整段 mega-kernel。T1–T3 调度权都在 host；T4 才把推进搬进 device。T4 内分手
**手工**（FlashDMoE、Spector、DeepGEMM 三件套、TileRT）与**自动**（MPK）。

- **自动整图（方法侧，MPK）**：编译器建图 + 去中心化事件驱动运行时。
- **手工段级（落地侧，DeepGEMM Mega MoE/Gate/mHC @ `057ca59`）**：三者是**同一套**
  「持久化 CTA + 手工静态角色 + 设备内异步原子 barrier + SMEM 环形流水」的段级 mega-kernel，
  差别只在 barrier 数量与角色数（`docs/02` §1）。满足 C1/C2/C4/C5，C3 只到一段。
- **需求侧信号（Kimi K3 §4.2.4 行 1069–1081）**：把「single-operator → fused mega-kernels」
  做成规模化训练任务套件，并用「防 CUDA graph replay / 防精度缩减」把真融合与假绕过区分开。
- **工程路线（hpc-ops README 行 230）**：把 megakernel 列为 roadmap 下一步（未实现），
  已实现的是 PDL + 多 kernel 的段融合（T2/T3）。

工程结论：产业进展呈 **「需求（K3）→ 路线（hpc-ops）→ 落地（DeepGEMM 段级 T4）→
自动化（MPK 整图 T4）」四段式**（`docs/02` §3.4）。

### 1.6 与既有手段的边界（一句话版）

| 手段 | 治哪类损失 | 粒度 | 差距 |
|---|---|---|---|
| 算子融合 | 损失三（+局部损失二） | 几个相邻算子 | 只融局部，无法覆盖整图（`docs/00` §2.1） |
| CUDA Graphs | 损失二（host 提交） | 整图（静态） | 不破依赖串行；残留 kernel-switch；形状变要重捕（`docs/03` §1） |
| PDL | 损失一特例（相邻 kernel 重叠） | kernel 对（需重写） | 重叠 = `min(tail, prologue)`，不随链累积（`docs/03` §2、`docs/04` §3） |
| 多流 | 无依赖算子并行 | 算子/stream | 有依赖即串行（`docs/00` §2.4） |
| 通信-计算 overlap | 隐藏跨设备通信延迟 | 单 kernel/单集合算子 | 调度权仍 host、中间量落 HBM（`docs/03` §4） |

**共同分水岭**：只有把调度权从 host 移到 device（C4）、粒度从算子细到 SM（C5），才跨过
「降级手段」与「真 mega-kernel」的分界。

## 2. 设计 checklist

把上面前提落成一张可勾选的表。每一行都对应本专题查证过的机制或坑。

### 2.1 判定阶段（先决定做不做）

- [ ] **量化了三类损失占比**：launch 是否 launch-bound？（MPK 293 launch/token 才值得）
      中间张量是否是带宽瓶颈？（P5-2：带宽主导区间损失三才 ~2×）跨算子是否有可重叠的
      compute/comm？（P4-4：无通信则无 overlap 收益）
- [ ] **确认依赖图足够深/窄**：真实 LLM forward tGraph「deep 而非 wide」，normalization
      开销 <1%（MPK §6.7 行 1656–1674）；若图大量 fork/join，辅助 task/event 开销上升。
- [ ] **确认算子数足够多**：算子太少时单次 launch 的开销省不了多少（P5-1 结论）。

### 2.2 融合边界阶段

- [ ] **划清单 kernel 覆盖范围**：段级（DeepGEMM，C3 到一段）还是整图（MPK）？范围越大，
      寄存器上限与 shape 约束越硬。
- [ ] **切分粒度选 SM/task**：分解策略以「最小化 device→shared memory 搬运」为目标、
      task 数与 SM 数成正比（MPK §4.1 行 467–468）。
- [ ] **检查每个算子的 tile 可独立性**：为跨算子流水留出「下一 task 可预取」的窗口
      （MPK §5.3 pre-loading/compute 拆分）。

### 2.3 依赖与调度阶段

- [ ] **依赖必须 precise**：用「输出 tile 是否重叠」判定最细依赖，而非算子级粗依赖
      （MPK §3 图 4(c) 反例、§4.1 dependency analysis）。**这是收益的真正来源。**
- [ ] **消除冗余同步**：对同 successor-set / predecessor-set 的 event 做融合
      （MPK Def 4.1/4.2；实测事件减少 37–118×）。
- [ ] **表示要做成定长无间接索引**：normalization 把每 task 的扇入/扇出压到 1（一个
      dependent-event id + 一个 triggering-event id）；linearization 用区间编码 event 扇出。
- [ ] **选择协调机制**：中心化 cooperative grid sync（简单、在目标卡 ~1.65 µs/次，但硬同步）
      vs 去中心化异步原子 + ring buffer（可重叠、生产级）。DeepGEMM 三者用**位掩码 + ring
      generation** / **ScoreBarrier** / **双 SplitBarrier**，都是后者（`docs/02` §1）。
- [ ] **处理负载不均**：运行时才知的工作量（MoE 每专家 token 数、attention 变长）用 JIT
      或 hybrid balancer（MPK §5.2、§6.4）。
- [ ] **防死锁**：交错调度下上游可能饿死下游——DeepGEMM 用 warmup wave + ring 容量上界
      闭式推导（`scheduler/mega_moe.cuh` 行 15–80）；通用做法是保证「先集中发上游」的最小 wave。
- [ ] **跨 launch 复位不靠 host**：用每 launch 唯一的 grid idx 编码 generation
      （DeepGEMM `l2_full_mask` 奇偶、Mega Gate/mHC 的 `grid_idx` 状态位）。

### 2.4 资源与正确性阶段

- [ ] **寄存器是硬约束**：per-thread register 固定为所有 task 类型的**最大值**（MPK §7 行
      1741–1749）；融合越多算子越易成瓶颈。DeepGEMM 用 `warpgroup_reg_dealloc/alloc` 手工分配 +
      静态断言 ≤ 64K。
- [ ] **shared memory 用分页抽象化解**：按 task 粒度 acquire/release、单调使用模式
      （MPK §5.3 paged shared memory，32 KB/页）。
- [ ] **中间结果优先留片上**：这是损失三收益的来源（P5-2 实测 ~2×）；跨 kernel 传递必然落 HBM。
- [ ] **数值正确性门限**：融合不改变数值（P5-2 `max|Δz|=0`）；勿以降精度换性能——Kimi K3
      明确把 precision reduction 判为作弊（`docs/02` §3）。
- [ ] **端到端回归**：per-task 参考实现 + 整图逐段校验；PDL 场景验证内存可见性
      （P5-3 `fail=0`、P4-2 checksum）。

### 2.5 落地/工程阶段

- [ ] **选择路径**：手工段级（可控、SM100 专属、不可泛化）vs 编译器自动（MPK，跨代只改
      per-task codegen）。MPK 把「硬件相关」收缩到唯一组件 per-task codegen（§7 行 1749–1764）。
- [ ] **接入手写 kernel**：每个 task 是统一调用约定的 device function，手写 kernel 可包成
      task 进调度（MPK §7）——不必全自动。
- [ ] **动态形状策略**：MPK 按 batch size 的 2 的幂生成多张静态 tGraph，运行时选图
      （§6.1 行 1427–1436）——这是**有限离散的动态**，不是任意变长。
- [ ] **调试与预热点**：CUDA Graph capture 前要预热流（DeepGEMM 注释 `mega_gate.hpp:19`）；
      device graph launch 仍停在 kernel 级，不能替代 in-kernel 调度。

## 3. 何时该 / 不该用 MegaKernel

### 3.1 该用（收益可期）

1. **launch-bound 的小算子图**：每 token 数百次 launch、单算子计算不足以淹没 ~2 µs launch
   间隔（decode、小 batch、小模型）。这是 MPK 端到端收益最大的区间（§6.3：越小模型/越新 GPU 越明显）。
2. **中间张量 HBM 往返主导**：一连串逐元素/norm/Gate 类低算术强度算子反复读写大张量
   （MoE gather 占 11%、Mega Gate 打分矩阵、mHC 的 `[T,4,H]`）——损失三可省近 2×（P5-2）。
3. **有跨算子/通信-计算重叠机会**：MatMul→AllReduce、dispatch→GEMM→combine、attention 变长，
   细粒度 event 能把 compute 与 comm 真正并起来（MPK overlap 1.1×）。
4. **图相对静态、算子多、可离线编译**：整图 mega-kernel 需要编译期建图；离线/batched serving
   最合适（MPK 评测设定）。
5. **团队有能力或工具**：要么有深 GPU 经验（手写段级，如 DeepGEMM），要么有编译器（MPK）。
   否则手工 mega-kernel 的工程成本与不可泛化会吞掉收益。

### 3.2 不该用（会亏或做不动）

1. **compute-bound 的大算子**：单个大 GEMM/attention 时长几十~几百 µs，launch/中间量占比可忽略
   （P5-1：计算主导时三策略趋同，1.06×）。此时用 T1 单算子优化更划算。
2. **形状/控制流高度动态**：CUDA Graph 都要重捕，整图 mega-kernel 的「每 batch size 一张静态图」
   更受限；任意变长序列/结构不适用。
3. **寄存器压力爆炸**：融合算子越多，per-thread register 被顶到最大值，occupancy 崩掉——
   融合范围有硬上限（MPK §7）。
4. **算子异构且严重不均衡**：纯静态划分会拖尾（MoE 倾斜路由）；没有 JIT/hybrid balancer 时收益不稳。
5. **需要跨架构可移植**：手写段级 mega-kernel 常整 kernel 绑死某代硬件（DeepGEMM 三件套均
   SM100 专属，本机 H100/sm90 不可跑）；只有编译器路径能把硬件相关收缩到 codegen。
6. **正确性/可调试性优先、算子数少**：单 kernel 内协调难调试（死锁、内存可见性），算子少时
   省下的 launch 不值得这份复杂度。

### 3.3 决策速查

| 你的情形 | 建议 |
|---|---|
| 大 GEMM / attention 单算子，compute-bound | 先做 T1 单算子优化，不必 mega |
| 多个小算子、launch 占比高、图静态 | CUDA Graph 先兜底；仍不够再上段级/整图 mega |
| 依赖链有 compute/comm 可重叠、但只相邻两个 kernel | PDL 定向补丁（注意不累积，P5-3） |
| 一整段低算术强度算子反复读大张量 | 段级融合（如 DeepGEMM），直接收获损失三 |
| 整图、多 GPU、要自动且可跨代 | MPK 式编译器 + in-kernel runtime（方法侧） |
| 团队强、只针对单一硬件、要极致 | 手写段级 mega-kernel（落地侧） |

## 4. 本专题实测校准（目标卡 H100 / sm90）

| 实验 | 结论 | 坐标 |
|---|---|---|
| P0-3 环境探测 | persistent/cooperative、CUDA Graph、PDL 三项可编译可运行 | `docs/04` §0、`harness/probe_env.out.txt` |
| P5-1 launch 开销 | eager per-单元恒定 ~1.8 µs；persistent 总时间恒定 ~2 µs（N=4096 达 1988×）；graph 只消 host 提交 | `docs/04` §1、`harness/launch_overhead.out.txt` |
| P5-2 两算子融合 | 带宽主导区间 fused vs barrier **~2.0×**（16→8 B/elem）；launch（损失二）在本区间近乎为零 | `docs/04` §2、`harness/two_op_fusion.out.txt` |
| P5-3 PDL 链 | 每条边重叠 **`min(prologue,tail)`**，收益**不随 K 累积**（A 组趋 2.0×、B 组趋 1.33×） | `docs/04` §3、`harness/pdl_pipeline.out.txt` |
| P4-1 CUDA Graph | 1024 节点 2.98×；per-kernel 降到 ~0.78 µs（不归零） | `docs/03` §1、`harness/cuda_graph.out.txt` |
| P1-2 多流/Graph | 无依赖 4 流 3.65×；有依赖 4 流 0.96×、依赖链 graph 1.05×（不破串行） | `docs/00` §2.4、`harness/mitigations.out.txt` |

一句话校准：**实测把论文的宏观锚点拆到了微观区间**——launch、中间量、重叠三类收益各有其显形
区间，融合/持久化/PDL 只在对应区间兑现，不能一概而论。

## 5. 关键坐标与延伸阅读

- 动机与谱系：`docs/00-megakernel-overview.md`（含 MPK 论文章节地图 §6）。
- MPK 精读：`docs/01-mpk-paper.md`（tGraph / 编译器 / in-kernel runtime / 评估）。
- 生产案例：`docs/02-production-cases.md`（DeepGEMM Mega MoE/Gate/mHC @ `057ca59`、
  Kimi K3 §4.2.4、hpc-ops README 行 230）。
- 相关技术对照：`docs/03-related-tech.md`（CUDA Graphs / PDL / persistent / 通信-计算 overlap，
  含 DeepEP `93eb6eb2` 与 FlashMLA `ba89a346` 的真实行号）。
- 实验：`docs/04-experiments.md`。
- MPK 论文：arXiv:2512.22219v2；线上实现 `https://github.com/mirage-project/mirage`。
- MoE 前/反向实现细节（本专题不重复）：`../mega-moe/docs/02-mega-moe-architecture.md`、
  `../mega-moe/docs/07-deepep-dispatch-combine.md`。

> P6-1 为**纯资料汇总**，无新增 H100 实验；所有实测数字均引自本专题 P0-3/P1-2/P4-1/P5-1/P5-2/P5-3，
> 原始输出在 `harness/*.out.txt`。**P6 完成，本专题路线图任务清单全部完成。**
