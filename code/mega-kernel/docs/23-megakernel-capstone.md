# MegaKernel 专题总纲 v2：结论地图、工程坑与实测校准（P0–P108）

> 本文是「MegaKernel 资料与实现分析」的**第二代总纲**（任务 P109-1，纯资料汇总，无新增实验）。
> 第一代收官篇 `docs/05-summary.md`（P6-1）只覆盖到 P0–P6；此后专题又推进了 100+ 个增量——
> 源码走读 `docs/06`–`docs/22`、实验 `docs/04` §6–§85、三本实验手册 `docs/11`–`docs/13`、
> 手工先例 `docs/10`。本文把这些结论压成**一张五层总图 + 五张表**，不改写 `docs/05` 的正文，
> 而是补上它之后新增的证据与边界。所有数字都带坐标（分篇/论文行号/源码行号），**不引用未复现的
> `benchmark/` 数字**。
>
> 主参考仍是 MPK 论文（arXiv 2512.22219，快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`）
> 与线上实现 `https://github.com/mirage-project/mirage`（分支 `mpk`，快照 `f9eb70c`）。

## 1. 一页纸 v2：三类损失 → 机制 → 判据

### 1.1 三类损失（动机，`docs/00` §1，MPK §1 行 74–89）

| 损失 | 症状 | 论文锚点 | 本专题实测（目标卡 H100 / sm90） |
|---|---|---|---|
| 一 kernel barrier | 同 stream 相邻 kernel 隐式全屏障，禁跨算子软件流水 | 流水 1.2–1.3×、overlap 1.1×（§6.6 行 1586–1615） | 单 kernel 内跨 task 流水 **1.18×**、跨层预取峰值 **1.19–1.20×**（`docs/04` §7/§16） |
| 二 launch 开销 | 每 token 数百–上千次 launch | Qwen3-8B 293 launch/token、scheduler 0.28%（§6.6 行 1618–1628） | eager per-单元 ~1.8 µs（近恒定）、persistent 总时间恒定 ~2 µs、N=4096 **1988×**（`docs/04` §1） |
| 三 中间张量 HBM 往返 | 上游写回 device、下游读回 | MoE gather 占 11%（§6.4 行 1551–1553） | fused 恒 **8 B/元素**、eager = Q×、字节比**精确 = Q**（ncu，`docs/04` §17） |

**一句话更新（相对 `docs/05` §1.1）**：三类损失不仅各有「显形区间」，其**直接字节证据**现在
有了——损失三的加速**完全等于省下的 DRAM 字节比**（`docs/04` §17），损失二只在 launch-bound 区间
有意义（`docs/04` §1/§4），损失一在单 kernel 内可回收且量级恰落论文 1.2–1.3×（`docs/04` §7）。

### 1.2 判据 C1–C5（`docs/00` §3.2）与「只在运行层拿到」

五条判据不变：**C1 单次 launch、C2 常驻、C3 多算子范围、C4 设备内协调、C5 SM 级调度**。

**关键修正（源码走读的总结论，`docs/22` §8）**：把 MPK 编译栈逐层走读后可见——前端、依赖分析、
搜索器、codegen（CUDA/Triton/Blackwell）、单 task device、多卡**全部只在 C3 的「范围/粒度」上加码**；
**C1/C2/C4/C5 只在运行层（`docs/06`）拿到**。换 codegen 目标语言（CUDA↔Triton）或换硬件后端
（Hopper→Blackwell）都拿不到 C1/C2/C4/C5（`docs/20` §7/`docs/21` §12）。

## 2. 结论地图：五层

把 P0–P108 的所有增量归到五层。「已验证结论」列的每一条都能在对应分篇找到实验或源码坐标。

| 层 | 回答什么 | 分篇 | 已验证结论（要点） |
|---|---|---|---|
| **范式/动机** | 是什么、凭什么、与降级手段差在哪 | `docs/00`/`03`/`05`/`10` | 真 mega-kernel 的分水岭是 **C4/C5**（把调度权从 host 移到 device）；T4 内分手（FlashMoE/Llama-1B/DeepGEMM）/自动（MPK） |
| **编译层** | 图从哪来、怎么变成常驻 kernel | `docs/01`/`08`/`14`–`22` | 两条链：生产（手工前端→`annotated_graph`→runtime→**常驻 megakernel**）vs 自动（`superoptimize`→transpiler→**kernel-per-operator**）在 `KNGraph` 后分叉 |
| **运行层** | 设备内怎么调度、协调 | `docs/06`/`07`/`11` | 128 worker + 16 scheduler SM（1:32）；event 驱动就地入队；per-worker 环；release/acquire 内存序；描述符预计算只推 id |
| **数据面** | worker 能到多快、吃多少资源 | `docs/04` §59–§85、`docs/12`/`13` | 数据面可到 ~3000 GB/s（≈96% HBM 峰值）；寄存器固定为各 task 最大值是唯一不可分页资源；代码足迹随 task 类型数累积 |
| **部署** | 多租户、共存、跨卡 | `docs/09`/`17`、`docs/04` §27–§28 | 整机 megakernel 会饿死共存负载；Green Contexts 确定隔离（0.76×），MPS 最坏 9.9×（非确定）；多卡下 C1 从优化变硬约束 |

### 2.1 编译层：生产路径 vs 自动路径（`docs/22` §0–§5）

- **生产路径**（拿 C1–C5）：手工前端（每个 `*_layer()` 建 bGraph + `register_task`，`docs/14`）→
  依赖分析器十步 (a)–(j)（`docs/08`）→ runtime 展开 `EventDesc`/`FullTaskDesc`（`docs/08` §7）→
  常驻 worker/scheduler（`docs/06`）。论文 §4.1 五步 ↔ 实现十步，normalization 在实现里是
  **「拒绝 case 2/3 + LCM 降级」**而非 Figure 6 重写（`docs/08` §4）。
- **自动路径**（只到 C3）：`KNGraph.superoptimize()` → `search.cc` 三层 DFS + 三个正交判断器
  （`docs/15`/`16`）→ transmiler（`docs/18`）→ runtime 模板库（`docs/19`）；另有平行的 Triton 后端
  （`docs/20`）与 Blackwell 2SM 后端（`docs/21`）。**`mpk` 生产前端不调 superoptimize**（`docs/15` §7）。
- **「同一套硬件惯用法实现了两遍」**（`docs/22` §5）：warp-specialized TMA+wgmma 既有生产 task
  （`docs/07`）也有 runtime 模板库（`docs/19`）；Triton codegen 有新/旧两套。**难的不是编出快 kernel，
  是「编成一张图、常驻着去中心化调度」。**

### 2.2 运行层：部件与「量化理由」（`docs/11`/`docs/13`）

`docs/11` 把调度器拆成六个可量化部件，几乎逐条对上 MPK：

| 部件 | 设计结论 | 对上 MPK / DeepGEMM |
|---|---|---|
| 角色划分 | 重 task 下 `PS*:G≈1:25~1:32`（§58） | MPK 4 scheduler : 128 worker = **1:32**（`docs/01` §5 行 1157–1164） |
| per-worker 环 | 「一消费者一环」前缀精确才可 `D>1`（§57/§58） | MPK `per_worker_queue_len`（快照 `persistent_kernel.cuh:1790-1791`） |
| event 驱动 | 就地入队 `G*≡1`（§34/§45） | MPK §5.1 完成者直接通知 event |
| 内存序 | release 发布 + acquire 读（§36/§37） | MPK `mpk_atoms.cuh`；DeepGEMM `red.async.inc.release` |
| 预取 | 把固定串行缝藏进执行（§43） | MPK §5.3 task descriptor 预取 |
| 描述符来源 | **编译期预计算 + 只推 id**（§47/§48） | MPK `runtime.cc` 预计算 `FullTaskDesc` |

## 3. 实测校准表 v2

### 3.1 损失侧（收益从哪来）

| 实验 | 结论 | 坐标 |
|---|---|---|
| launch 开销 | eager ~1.8 µs/单元恒定；persistent 总时间恒定 ~2 µs；N=4096 **1988×**；graph 只消 host 提交 | `docs/04` §1 |
| N 算子链融合 | e/fused 随 N 近似线性（≤8 时 0.99→7.13×），N≥16 转计算主导饱和 9.4–10.3× | `docs/04` §4 |
| 两算子融合 | 带宽主导区间 fused vs barrier **~2.0×**（16→8 B/elem） | `docs/04` §2 |
| **损失三字节** | fused DRAM 恒 **8 B/元素**；eager/graph = `Q×517 MB`、`vs_fused` **精确 = Q**；barrier 字节与 eager 几乎相同 | `docs/04` §17（ncu） |
| 单 kernel 内流水 | R=64 时 `K=1→2` 拿 **1.18×**、再加深无增益 | `docs/04` §7 |
| 跨层流水+环形预取 | 峰值 **1.19–1.20×**（vs barrier）、收益≈访存占比、纯访存净亏 | `docs/04` §16 |

### 3.2 成本侧（megakernel 要付什么）

| 实验 | 结论 | 坐标 |
|---|---|---|
| 寄存器足迹 | 轻 task 因最重 task 被钉在低预算，访存带宽掉 **1.33×–2.29×** | `docs/04` §8 |
| 代码足迹 | 执行 task 类型数 P=256 达 **2.18×**；纯 **L1i miss**（`no_instruction` 0.63%→59.19%） | `docs/04` §10 |
| 双层足迹叠加 | 两笔代价在绝对时间上**相加而非相乘**（预测 1.5731 vs 实测 1.5747 ms） | `docs/04` §11 |
| 寄存器足迹可绕开性 | `setmaxnreg` + warp 特化收回代价 **1.3×–1.8×**；warp-specialized 只是前提，真正避免 spill 的是 setmaxnreg | `docs/04` §12 |
| 网格大小 | speedup 在**占用率上限**（1056 block）处饱和 212.9×，而非 #SM；`grid.sync()` 有 ~1.1 µs 地板 | `docs/04` §5 |

### 3.3 调度/正确性侧

| 实验 | 结论 | 坐标 |
|---|---|---|
| 去中心化最小骨架 | C5 机制可用且**逐位不改数值**；均衡负载 decen≈barrier；**小负载慢到 0.556×**；异构动态负载 decen 1.167× vs barrier 1.007× | `docs/04` §9 |
| DAG 形状敏感性 | 严格分层均衡图集中式 barrier 恒最快；MPK 批量取 task 甜点 **K≈W/G** | `docs/04` §15 |
| 有界池死锁 | 死锁边界 = **阻塞数 `B(π)`**（贪心峰值池占用）、`CAP*=G*=B(π)+1`；层号乱序时「层号升序」不再是拓扑序 | `docs/04` §31–§33 |
| 有界环背压 | `CAP_safe + G ≈ W_max + 1`；容量是吞吐硬下界 | `docs/04` §35 |
| 发布内存序 | store→store 保序，store→load 可绕过；release/acquire 代价 **+0.36 µs/槽** | `docs/04` §36 |
| MPSC | 必须**连续放行**（连续 CAS）；占位即就绪在决定性正对照下必错 | `docs/04` §37 |
| 取 task 载体 | 真实 352 B 下 `cp.async`/TMA bulk 成本与大小无关；大 K + `D=2` 最优 | `docs/04` §39–§41 |

### 3.4 数据面侧（worker 能到多快）

| 实验 | 结论 | 坐标 |
|---|---|---|
| 数据面峰值 | 抬每 block 在飞字节（`T×U`）可到 **~2983 GB/s（≈89–96% HBM 峰值）** | `docs/04` §45、`docs/13` §4 |
| 等积塌缩 | 饱和区吞吐只由 **`O·T`** 决定，与 `(T,O)` 拆分/32 vs 40 regs 无关 | `docs/13` §4（`docs/04` §70） |
| 寄存器预算 | `__maxnreg__(R)` 是上限非设定值；最优 = 「不 spill 的最小 R」；`R≤32` 恒 ≥ 线程上限 | `docs/12` §6、`docs/04` §69 |
| 块粒度 T | 次级量（~4%），落点是发射侧 **LSU 背压**，不是 L2 停顿；跨进程/跨卡稳健 | `docs/12` §7、`docs/04` §82–§85 |
| 共驻税 | 大体加和 ±13%；唯一系统裂缝 = 发射受限（`scalar`）恒低 ~13%；`l2res` 超加和 1.111；税住在 SM **间**、分区收不回 | `docs/12` §2–§4 |

### 3.5 部署侧

| 实验 | 结论 | 坐标 |
|---|---|---|
| Green Contexts | SM 硬分区确定隔离：victim **0.76× 基线**、persist 无损、双向；干扰开关是「剩余占用率槽位」 | `docs/04` §27 |
| MPS | 跨进程真并发但**非确定**：cap 略低于 100 时 slowdown `≈1/(1−P)`，P=90→**9.9×**；cap 应设 P≤25 | `docs/04` §28 |
| 多卡 | `nvshmemx_collective_launch` 不支持 split worker/scheduler → **C1 从优化变硬约束** | `docs/17`、`docs/22` §4 |
| MPK 端到端 | 单算子 36–39 µs、4 算子 46 µs（vs torch 朴素 3.5–4.8×）；延迟由「双常驻 kernel 启动/终止协议」主导 | `docs/09` §8/§12 |

### 3.6 降级手段对照（kernel-per-operator 这条线的天花板）

| 手段 | 治哪类损失 | 天花板（实测） | 坐标 |
|---|---|---|---|
| CUDA Graph | 损失二（host 提交） | 1024 节点 2.98×；per-kernel 摊到 ~0.78 µs**不归零**；条件节点每行程 4.6 µs、比 host launch 还贵 ~2× | `docs/03` §1、`docs/04` §25 |
| PDL | 损失一特例 | 每条边重叠 **`min(prologue,tail)`**、**不随链长累积**；满占用下重叠饱和 ~0.74×ideal | `docs/04` §3/§6 |
| PDL × Graph | 二者能否相乘 | capture **保留 PDL 语义**，但叠加是**次乘非相乘**（0.92× 乘积） | `docs/04` §13 |
| Programmatic 边 | 提前「看到进度」 | 只把下游**调度点**前移，`gridDependencySynchronize` 仍等**完成**；无通用「中间进度」信号 | `docs/04` §21/§24 |
| Device graph launch | 设备侧分发 | ~4 µs 地板（vs in-kernel ~40 ns） | `docs/04` §26 |
| 多流 | 无依赖算子并行 | 有依赖即串行 | `docs/00` §2.4 |

**共同分水岭不变**：只有 C4/C5 才跨过「降级手段」与「真 mega-kernel」的分界（`docs/05` §1.6）。

## 4. 工程坑清单（把「已知坑」与实验结果方法论收口）

以下每条都是本专题踩过的真实坑，写 mega-kernel 或做其性能实验时会再遇到：

1. **寄存器不可分页**：per-thread register 固定为所有 task 类型的最大值（MPK §7 行 1741–1749）。
   `__maxnreg__(R)` 只是**上限**（`R≥` 自然需求即失效），与 `__launch_bounds__` 互斥但同 `numRegs`
   等价；`R=32` 是多数数据面 task「不 spill 的最小预算」下界（`docs/04` §69/§82）。
2. **指令足迹随 task 类型累积**：执行代码工作集超 L1i 后取指 miss；**不能用「各 task 足迹倍率相乘」
   估成本**（绝对时间上相加，`docs/04` §10/§11）。
3. **占用率是四重封顶**：`min(线程上限 2048/T, 寄存器 65536/(regs·T), 块上限 32, SMEM_perSM/block_smem)`；
   小 `T` 时 SMEM 会先于线程封顶（`docs/04` §82）。
4. **有界 ready 池会死锁**：集中准入须 `CAP>B(π)` 且 `G>B(π)`（联合约束），warmup 须覆盖**完整依赖
   深度**；正解是**事件驱动就地入队**（`G*≡1`）或真拓扑线性化（`docs/04` §31–§33/§45）。
5. **队列内存序不能只 relaxed**：多数场景「没出事」是时序侥幸，拉窗口即错；release/acquire 代价
   ~0.36 µs/槽、大 payload 摊薄（`docs/04` §36）。多生产者必须**连续 CAS 放行**（`docs/04` §37）。
6. **`D` 不是自由性能旋钮**：共享前缀环下 `D>1` 死锁或静默数值错，唯一正确形态 strided + 每槽回收
   或「一消费者一环」（`docs/04` §55/`docs/11` §7）。
7. **`grid.sync()` 有 ~1.1 µs 地板、`cluster.sync()` 更便宜但有硬上限 16 block / 逐维整除**
   （`docs/04` §5/§18/§20）。
8. **L2 persisting 是零和交易**：只削 DRAM 读 ~30%，carveout 应≈中间量大小、不该拉满；整图融合里
   「靠 L2 留中间量」最不可依赖（`docs/04` §29/§30）。
9. **ncu 口径坑**：默认 `--cache-control=all` 会 flush L2；`--replay-mode application` 会放大程序
   自报计时 ~2×，只采信硬件计数器与 `gpu__time_duration`；`smsp__average_warp_latency` 方向可能相反
   （`docs/04` §29/§83/§84）。
10. **`clock64` 是 per-SM 计数器**，跨 SM 相减会串入时钟偏移；跨 SM 绝对时刻用 `%globaltimer`。
11. **「吞吐超 HBM 上限」是丢 work 的第一信号**：`atomicAdd` 写在 kernel 主体会被每线程执行，须
    `tid==0` 守卫 + 共享内存广播（`docs/04` §77）。
12. **「有实现」≠「已启用」**：MPK 快照里至少 12 处「算法写完但未接线」（`docs/22` §6，如
    `range_propagation` 两个 `check_range` 首行 `return true`、`setmaxnreg` 被注释）。
13. **复现 ~1% 量级结论须跨进程/跨卡**，比绝对效应时固定卡；卡间 offset 与效应同量级（`docs/04` §85）。
14. **MPK 端到端不收敛的根因是 launch 顺序**（scheduler 必须先于 worker launch），不是内存序/dtype/
    occupancy；z3-solver 与 tilelang 版本互斥（ROADMAP「阻塞」B1/B2）。

## 5. 何时该 / 不该用（v2）

`docs/05` §3 的判定仍成立，v2 补上**实测 break-even**：

- **该用**：launch-bound 的小算子图（每 token 数百次 launch）；中间张量 HBM 往返主导的低算术强度算子链
  （损失三字节比 = Q，融合直接省整趟往返）；有跨算子/通信-计算可重叠且图相对静态；团队有深 GPU 经验或
  编译器。**整图多卡**（要自动且跨代）→ MPK 式编译器；**单硬件极致** → 手写段级（DeepGEMM）。
- **不该用**：compute-bound 大算子（计算主导时三策略趋同 ~1.06×，`docs/04` §1）；形状/控制流高度动态
  （MPK 每 batch size 一张静态图，`docs/01` §5.8）；寄存器压力爆炸；算子异构严重不均衡且无 JIT/hybrid
  balancer；**多租户环境**（整机 megakernel 会饿死共存负载，须 Green Contexts 隔离）。
- **成本侧的新约束**：融合的 task 类型越多，「多而杂」的共驻税 + 双层足迹税越大——**「少而重」而非
  「多而杂」是同时规避两层足迹与共驻税的同一个设计选择**（DeepGEMM mega 家族只融合少数极重 task
  类型即此，`docs/12` §9/`docs/02` §1）。

## 6. 与 v1 的差异、坐标与延伸

相对 `docs/05-summary.md`（P6-1），本文新增/修正：

- 判据层面：**C1/C2/C4/C5 只在运行层拿到**（`docs/22` §8 的总映射）。
- 损失侧：损失三有了 **ncu 字节证据**（= Q）；损失一在单 kernel 内量到 1.18×（`docs/04` §7/§17）。
- 成本侧：**寄存器足迹 + 代码足迹 + 共驻税**三笔（`docs/04` §8/§10/§12、`docs/12`）。
- 调度/正确性侧：有界池的死锁普适量 `B(π)`、内存序代价、环形缓冲容量下界（`docs/04` §31–§58）。
- 部署侧：Green Contexts / MPS / 多卡硬约束（`docs/04` §27/§28、`docs/17`）。
- 降级手段：CUDA Graph / PDL / Programmatic / device graph 的**天花板**（`docs/04` §13/§21–§26）。

坐标：

- 范式与动机：`docs/00`（含 MPK 章节地图 §6）、`docs/03`（CUDA Graphs/PDL/persistent/overlap）、
  `docs/10`（FlashMoE / Llama-1B 一手先例）。
- 论文精读：`docs/01`（tGraph / 编译器 / in-kernel runtime / 评估）。
- 生产案例：`docs/02`（DeepGEMM Mega MoE/Gate/mHC @ `057ca59`、Kimi K3 §4.2.4、hpc-ops）。
- 实验：`docs/04`（§0–§85）+ 三本手册 `docs/11`（调度器）/`docs/12`（共驻税）/`docs/13`（端到端数据面）。
- 源码走读：`docs/06`–`docs/08`、`docs/14`–`docs/22`（含总图手册 `docs/22`）。
- 构建与端到端：`docs/09`。
- 一页纸 v1：`docs/05-summary.md`。

> P109-1 为**纯资料汇总**，无新增 H100 实验；所有数字引自上述分篇与其 `harness/*.out.txt` 原始输出。
> 与 `docs/22`（源码侧手册）、`docs/11`–`docs/13`（实验侧三本手册）并列，本文是**整专题的第二代总纲**。
