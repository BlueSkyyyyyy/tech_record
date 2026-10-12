# MegaKernel 资料与实现分析 · 路线图（活文档）

> 目标：系统梳理 **MegaKernel**（persistent / mega-kernel，把整张算子图融合进单个持久化 kernel）
> 的设计动机、表示、编译/运行时（以 **MPK** 论文为主）与生产级案例（**DeepGEMM mega 家族**、Kimi K3），
> 并给出可操作的原理总结。本专题为长期自驱任务，由 `scripts/autopilot.sh` 驱动（与仓库内其它自驱任务并行）。
>
> 本专题以**资料分析**（论文/源码精读）为主；H100 上能做的小实验（launch 开销、persistent kernel、
> CUDA Graph、PDL）尽量做并给实测。

## 交付物（Definition of Deliverable）

1. **分析文档**（`docs/`，内联到博客）
   - MegaKernel 的动机：kernel-per-operator 的三类损失（kernel barrier / launch 开销 / 中间张量读写）。
   - MPK 论文精读：SM 级图表示（§3）、编译器（§4）、in-kernel 去中心化运行时、评估（§5）。
   - 生产级案例：DeepGEMM Mega MoE / Mega Gate / Mega mHC；Kimi K3 的 fused mega-kernels。
   - 相关技术对照：CUDA Graphs、PDL（Programmatic Dependent Launch）、persistent kernel、
     communication-computation overlap、in-kernel 调度。
2. **可运行小实验**（`harness/`、`src/`）：launch 开销、persistent kernel vs 多 kernel、CUDA Graph、PDL 的实测。
3. **实测证据**：能跑的给真实数字与原始输出；纯资料分析明确标注。

## 环境 / 参考坐标

- 容器 `kernel_lab`（`scripts/lab.sh`），GPU H100 sm90。
- MPK 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`（2717 行）。
- DeepGEMM mega 家族：`~/github/DeepGEMM`；hpc-ops：`~/github/hpc-ops`。
- **目标卡代号敏感**：一律写「目标卡」。**不要编造 URL**。

## 任务清单

状态：`[ ]` 未开始 · `[~]` 进行中 · `[x]` 完成。

### P0 侦察与框架

- [x] **P0-1** 通读 MPK 论文快照，产出章节地图与关键图/表清单（进 `docs/00` §6「MPK 论文导航」）。
- [x] **P0-2** 收集案例：`~/github/DeepGEMM`（mega_moe/mega_gate/mega_mhc）、`~/github/hpc-ops`、Kimi K3 报告相关段落。
      产出 `docs/02-production-cases.md`（案例源码坐标清单，含 commit 与核对过的行号）。
- [x] **P0-3** 环境探测：`kernel_lab` 能否跑 persistent kernel / CUDA Graph / PDL 小实验（H100 支持 PDL）。
      结果见 `docs/04-experiments.md` §0，源码 `src/probe_env.cu`，原始输出 `harness/probe_env.out.txt`：
      三项均**可编译可运行**（cooperative grid sync 0.330ms/200 轮；CUDA Graph 1.56×；PDL ok）。

### P1 动机与背景（`docs/00-megakernel-overview.md`）

- [x] **P1-1** kernel-per-operator 执行模型与三类损失：kernel barrier（跨算子无法软件流水）、
      launch 开销、中间张量 HBM 往返。见 `docs/00` §1（量化锚点：293 launch/token、3.8µs eager→1.1ms、
      0.8µs graph→0.2ms、scheduler 0.28%、流水 1.2–1.3×、重叠 1.1×、gather 占 MoE 11%）+ P0-3 实测佐证。
- [x] **P1-2** 已有缓解手段与不足：算子融合、CUDA Graphs、PDL、多流；为什么还不够（跨算子细粒度重叠、SM 级调度）。
      见 `docs/00` §2（逐条贴 MPK 快照行号：融合 §2.2 行 313–328；CUDA Graphs §1 行 87–89 + §6.3 行 1504–1509；
      PDL §1 行 78–84；多流/细粒度重叠 §1 行 85–86；通信-计算 overlap §8 行 1878–1917；相关抽象 §8 行 1864–1877）。
      P1-2 实测：`src/mitigations.cu` + `harness/mitigations.out.txt`（目标卡 H100/sm90，64×17µs kernel）：
      无依赖 4 流 **3.65×**、有依赖链 4 流 **0.96×**（≈串行）、依赖链 CUDA Graph **1.05×**，
      实证「多流只并行无依赖算子、Graph 不破依赖串行」。
- [x] **P1-3** MegaKernel 的定义与谱系：`docs/00` §3 给定义（MPK §1 行 90–101 的三好处）+ **五条判据
      C1–C5**（单次 launch / 常驻 / 多算子范围 / 设备内协调 / SM 级调度）+ §3.3 术语澄清（persistent /
      cooperative / fusion / CUDA Graph / PDL / TileRT）；§4 给四档谱系 T1 单算子 → T2 局部/段融合 →
      T3 通信-计算重叠 → T4 整图 mega-kernel（各档贴 MPK 快照真实行号），T4 内分手 `FlashDMoE`/`Spector`
      与自动 `MPK`，并说明 DeepGEMM 三者为「段级 mega-kernel」。

### P2 MPK 论文精读（`docs/01-mpk-paper.md`）

- [x] **P2-1** SM 级图表示（§3）：以 SM 为粒度的数据依赖；与 CUDA Graphs 的对比。
      见 `docs/01-mpk-paper.md` §2（逐条贴 §3 快照行号 345–448）：task/event 交替语义（行 360–370）、
      MatMul→AllReduce 细粒度重叠（行 371–386）、同图多解与图 4(c) 反例（行 387–401）、
      与 CUDA Graphs 五维对比表（行 402–414）、Figure 5 预告（行 415–448）。纯资料分析。
- [x] **P2-2** MPK 编译器（§4）：tGraph 生成、每个 task 的 CUDA 实现生成。
      见 `docs/01-mpk-paper.md` §3（逐条贴 §4 行号 449–1146）：§4.1 五步流水线——operator decomposition
      （行 462–468，§6.7 每算子 32–47 task、Qwen3-8B 293→13,867 task）、dependency analysis（行 469–519，
      「输出 tile 重叠」即插 event）、event fusion（Def 4.1/4.2 行 520–779，§6.7 37–118×）、
      normalization（行 780–951，Figure 6 空 task 把 event 扇入/扇出压到 1，<1%）、linearization
      （Algorithm 1 行 952–1098，BFS 保证同 event 触发 task 连续，§6.7 4.4–15×）；§4.2（行 1126–1146）
      以 **thread block** 为单位复用 Mirage superoptimizer 生成 CUDA。纯资料分析。
- [x] **P2-3** in-kernel 并行运行时（§5）：去中心化调度、跨算子软件流水、计算-通信重叠。
      见 `docs/01-mpk-paper.md` §4（逐条贴 §5 行号 1147–1368 与 §6.1/§6.6）：§5 总览 worker/scheduler
      划分（行 1148–1174，固定角色、SM 数匹配）；§5.1 事件驱动（行 1175–1231，start event、event 激活、
      异构互连无需拓扑画像）；§5.2 混合下发 JIT/AOT（行 1232–1308，两步 vs 一步同步、编译期 operator 标签、
      global barrier 分界、worker 双队列优先 JIT、AOT round-robin 预入队）；§5.3 三优化
      （paged shared memory 行 1312–1325、cross-task software pipelining 行 1326–1363、
      task description 预取 行 1364–1368）；实现/配置（44K C++/42K CUDA/10K Python、semaphore、
      环形缓冲 atomicAdd、Table 1 行 1437–1460）；§6.6 三消融（pipeline 1.2–1.3×、overlap 1.1×、
      launch reduction 293→0.28%）。纯资料分析。
- [x] **P2-4** 评估（§6）完成：见 `docs/01-mpk-paper.md` §5（逐条贴 §6 行 1369–1737 + §7 行 1739–1764）。
      §6.3 单 batch 相对 SGLang/vLLM **1.0–1.7×**（Qwen3-8B/A100 14.5→12.5ms，逼近 ~10ms 硬件下界；
      vs 原生 PyTorch >10×）；§6.4 MoE 混合负载均衡 + fused gather-GEMM（gather 占 SGLang 11%）；
      §6.5 TP 下 8×H100 相对 PyTorch 最高 **10×**、相对 SGLang/vLLM **1.1–1.4×**；
      §6.6/§6.7 消融表格化归档；§7 Discussion 边界（寄存器固定为各 task 最大值、per-task codegen
      是唯一硬件相关组件、手写 kernel 可包成 task）；§5.8 给复现要点与 **8 条局限**
      （仅 LLM serving / offline batched / batch≤16 / 多 GPU 只测 dense TP / 每 batch size 一张静态图 /
      去中心化调度未探全局协调 / 寄存器上限 / page size 固定）。纯资料分析。

### P3 生产案例（`docs/02-production-cases.md`）

- [x] **P3-1** DeepGEMM **Mega MoE**：dispatch+GEMM+SwiGLU+combine 单 kernel、symmetric memory、
      ring buffer 调度（与 `mega-moe` 专题交叉引用，避免重复）。见 `docs/02` §1.5（megakernel 范式视角：
      单 kernel 边界、静态 warp 角色含 warp7 in-kernel scheduler、`l2_full_mask` 位掩码依赖 +
      `get_num_max_live_pool_blocks` ring 上界 + warmup wave 防死锁、对称显存 rank 切片、
      与 MPK 逐维对照表；算子级细节交叉引用 `../mega-moe/docs/02-mega-moe-architecture.md`）。
- [x] **P3-2** DeepGEMM **Mega Gate** 与 **Mega mHC**（HyperConnection）：各自融合了什么、为什么。
      见 `docs/02-production-cases.md` §1.2/§1.3 的「P3-2 精读」（逐条贴 `057ca59` 核对行号）：
      Mega Gate 拍平「打分 GEMM + sqrt-softplus + bias + mask/force_random/fix_routing + top-k 选择 +
      归一化缩放 + logical→physical 映射」7 段为一次 launch，静态角色 warp0 TMA/warp1 MMA/gate warps，
      `ScoreBarrier`（异步原子 + grid 标签）+ split-K scratch 归约协调，Tensor Core 与 gate 排名重叠；
      Mega mHC 拍平「HC pre-mix TF32 GEMM + post/comb mix + Sinkhorn + 残差更新 + RMSNorm + FP8 量化/SF
      + 下一层 mix 生成」，静态 WG0–WG5（Normal 模式回收退休 WG），双 `SplitBarrier`（Norm/Mix）+
      静态 Full/empty SMEM 环 + 寄存器重配置。均 SM100 专属，**仅资料分析**。
- [x] **P3-3** Kimi K3 的 fused mega-kernels（RL 内核任务套件）、hpc-ops 的 Megakernel roadmap。
      见 `docs/02-production-cases.md` §2（hpc-ops：roadmap 行 230 未实现，命中损失二/三；已实现的
      Fused MoE / AllReduce+RMSNorm / sampler 用 PDL+多 kernel，属 T2/T3）与 §3（Kimi K3 §4.2.4
      快照行 1069–1081：single-operator→fused mega-kernels、多 DSL/精度、正确性门限+roofline 打分、
      防 CUDA graph replay/精度缩减作弊）。给出「需求(K3)→路线(hpc-ops)→落地(DeepGEMM 段级)→
      自动化(MPK 整图)」四段式工程结论。纯资料分析。

### P4 相关技术对照（`docs/03-related-tech.md`）

- [x] **P4-1** CUDA Graphs：消除 launch 开销但仍有 kernel barrier；与 megakernel 的关系。
      见 `docs/03-related-tech.md` §1（机理/API 三步/capture 模式/显式构图；MPK §1 行 87–89、§3 行 402–414；
      本机实测 `src/cuda_graph.cu` + `harness/cuda_graph.out.txt`：eager ~1.7–2.3 µs/kernel 近恒定、
      graph per-kernel 摊到 ~0.78 µs（不归零）、1024 节点 2.98×、单次 `cudaGraphLaunch` 固定 ~3.1 µs、
      capture+instantiate 一次 ~104 µs；device graph launch 仍停在 kernel 级；判据仅部分满足 C1）。
- [x] **P4-2** PDL（Programmatic Dependent Launch）：跨 kernel 重叠；H100 实测。
      见 `docs/03-related-tech.md` §2（三原语机理 `cudaLaunchAttributeProgrammaticStreamSerialization`
      / `cudaTriggerProgrammaticLaunchCompletion` / `cudaGridDependencySynchronize`，MPK §1 行 78–84 +
      引用 [26] 快照行 2242–2244；本机实测 `src/pdl.cu` + `harness/pdl.out.txt`：**重叠量 = min(tail, prologue)**，
      prologue=0→1.01×、prologue=tail→1.51×、prologue>tail 被 tail 封顶 1.33×；判据全不满足 C1–C5）。
- [x] **P4-3** persistent kernel vs megakernel；in-kernel 调度（decentralized scheduling）。
      见 `docs/03-related-tech.md` §3：3.1 术语边界（persistent=C1+C2，megakernel=C1–C5，persistent 是机制/mega 是范围）；
      3.2 persistent 的机制收益（消除 kernel barrier、跨算子软件流水、时分复用资源折中，逐条贴 MPK §2.1 行 223–239 /
      §2.2 行 313–331 / §7 行 1741–1749）；3.3 去中心化调度机理（worker/scheduler 角色、固定于 launch、event 驱动、
      不依赖拓扑画像、本地状态 + 原子 ring buffer、混合 JIT/AOT，贴 §5 行 1147–1308 + §6.1 行 1410–1413 + §5.3 行 1364–1368）；
      3.4 本机可验证边界（复用 P0-3 cooperative grid sync 0.330ms/200 次、132 SM，并如实标注它与 MPK 去中心化 event queue
      是两种协调方式，调度机制部分纯资料分析）；3.5 判据映射；3.6 小结。纯资料分析 + 复用 P0-3 实测，量化对照留待 P5。
- [x] **P4-4** 与「通信-计算 overlap」（DeepEP / FlashMLA 双 kernel）的对比。见 `docs/03-related-tech.md` §4：
      4.1 两种真实机制——DeepEP 通信流 + CUDA event（`event.py:39-71`、`README.md:261-289`）
      与 FlashMLA PDL 双 kernel（`splitkv_mla.cuh:1226,1338-1348`、`combine.cu:59,207-219`、deep-dive `:60`）；
      4.2 重叠对象与锚点（复用 P4-2 `min(tail,prologue)`；MPK §6.3 行 1600–1616 overlap 1.1×）；
      4.3 未消除项（粒度=kernel/集合算子、host 调度权、中间张量落 HBM、launch 不变）；4.4 判据映射
      （FlashMLA 仅部分 C4）；4.5 小结。**P4 收官**。

### P5 实验（`docs/04-experiments.md`）

- [x] **P5-1** launch 开销实测：N 个小 kernel 串行 vs CUDA Graph vs 单 persistent kernel。
      见 `docs/04-experiments.md` §1，源码 `src/launch_overhead.cu`，原始输出
      `harness/launch_overhead.out.txt`（目标卡 H100/sm90）：Section A（K=0）eager per-单元
      恒定 ~1.75–2.2µs、graph ~2.1×、persistent 总时间恒定 ~2µs（N=4096 达 1988×）；
      Section B（N=256 扫 K）用 `persist1`(grid=1，设备侧工作与 eager 等价) 证明二者随 K
      收敛（48.9×→1.12×，差额即 launch 开销），`persistG`(grid=132) 靠并行度保持 133.8×；
      graph 也趋 1.06×。结论：graph 消 host 提交、persistent 连设备 grid 启动也消，只在
      launch-bound 区间有意义。
- [x] **P5-2** 一个简单「两算子融合成单 persistent kernel」的最小示例 + 实测收益。
      见 `docs/04-experiments.md` §2，源码 `src/two_op_fusion.cu`，原始输出
      `harness/two_op_fusion.out.txt`（目标卡 H100/sm90）：对一对逐元素依赖算子
      `y=f(x)`→`z=g(y)`，比较 eager2（2 launch，中间量落 HBM）/ graph2（1 次提交，
      仍走 HBM）/ barrier（单 cooperative persistent kernel + grid.sync()，仍走 HBM）/
      fused（单 persistent kernel，中间量只在寄存器）。结论：M=16–256Mi 元素、带宽主导区间，
      **fused 比 barrier 快 ~2.0×**（16 B/elem→8 B/elem，表观带宽四者一致 ~2.8 TB/s），
      即**损失三**（中间张量 HBM 往返）是主导收益；而 eager2→barrier/graph2 ≈1.0×，
      **损失二在本区间近乎为零**（印证 P5-1「launch 只在 launch-bound 有意义」）；
      cooperative grid sync（660 block）相对 eager2 无可测开销。正确性 max|Δz|=0。
- [x] **P5-3** PDL 小实验：把 K 个 kernel 用 PDL 串成**链**，验证收益是否随 K 累积。
      见 `docs/04-experiments.md` §3，源码 `src/pdl_pipeline.cu`，原始输出
      `harness/pdl_pipeline.out.txt`（目标卡 H100/sm90，iters=30，K=1/2/4/8/16）：
      no-PDL 严格串行（每 kernel ≈ span=pl+crit+tl）；PDL 每条链边重叠 **恰好 `min(prologue,tail)`**，
      稳态每 kernel = `span − min(prologue,tail)`。A 组 pl=tail=200000：speedup 1.00→1.87×（趋 2.0×）；
      B 组 tail=3×pl：1.00→1.31×（趋 1.33×，被较短的 prologue 封顶）。**结论：PDL 链式收益不随 K
      叠加**，上限由 `min(prologue,tail)` 占比决定——是「1 级 kernel 级流水」，与 MPK §5.3（行 1326–1363）
      单 kernel 内 task 级可累积软件流水本质不同；正确性 fail=0。

### P6 汇总

- [x] **P6-1** `docs/05-summary.md`：原理一页纸（§1）+ 设计 checklist（§2）+ 何时该/不该用 megakernel（§3）
      + 本专题实测校准表（§4）+ 关键坐标（§5）。汇总 `docs/00`–`docs/04` 全部结论，不展开新资料、
      不产生新实验；配套博客 post `mega-kernel-06-summary` 并更新 `mega-kernel-00-index` 目录。
      **P6 完成、本专题任务清单全部完成。**

### P7 线上实现源码走读（`docs/06-mpk-source.md`）

- [x] **P7-1** MPK 线上实现（`mirage-project/mirage`，分支 `mpk`，快照 commit `f9eb70c`）运行层源码走读：
      编译层 `src/kernel/runtime.cc`（生成 `_init_persistent_kernel` + `_execute_task`，`runtime.cc:1885-1903/1939`）、
      数据结构 `runtime_header.h`（TaskId/EventId 编码、`TaskDesc`、`EventDesc`、`RuntimeConfig`）、
      设备侧 worker 循环 `execute_worker`（`:957-1252`，事件计数精确触发 + 原子 ring 入队）、
      设备侧 scheduler 循环 `execute_scheduler`（`:1254-1523`，事件三档派发 + `prepare_next_batch` 连续服务）、
      宿主 init/launch（`:1610-2005`）、内存序原语 `mpk_atoms.cuh`、SM→worker/scheduler 划分
      （`utils.py:25-29`，H100 128 worker/16 scheduler 对上论文 §5.1）。与论文逐项对照表 + C1–C5 全满足；
      **纯源码走读（资料分析），本轮未构建/运行 mirage**。

### P8 博客发布

- [x] **P8-1** 补全博客 post 与目录：为 `docs/02`/`03`/`04`/`06` 补建 `content/posts/mega-kernel-*`
      的 `doc_include` 内联页，并更新 `mega-kernel-00-index` 目录（原第 3/4/5 项「待写」）。
      新增 post：`mega-kernel-03-cases`（weight 4，docs/02）、`mega-kernel-04-related`（weight 5，docs/03）、
      `mega-kernel-05-experiments`（weight 6，docs/04）、`mega-kernel-07-mpk-source`（weight 8，docs/06）。
      纯发布接线，不新增资料/实验。

### P9 实验扩展（`docs/04-experiments.md`）

- [x] **P9-1** 把 P5-2 的「两算子融合」推广到 **N 个依赖算子链**，看两类损失随 N / M 的标度
      （对应「下一步」可选方向②，补 P5-1 §1.6 / P5-2 §2.6 的「更多算子 / 更小 M」）。
      见 `docs/04-experiments.md` §4，源码 `src/n_op_fusion.cu`，原始输出
      `harness/n_op_fusion.out.txt`（目标卡 H100/sm90，iters=100）：
      Section A（N=8 扫 M 16Ki→16Mi）e/persist 1.43×→0.99×、e/fused ~7×；
      Section B（M=16Mi 扫 N 1→32）eager 表观带宽恒 ~2.75 TB/s、fused 流量恒 8 B/elem，
      **e/fused 随 N 近似线性（0.99→1.98→3.91→7.13×，N≤8）**，N≥16 因融合转入计算主导
      饱和在 9.4→10.3×；Section C（M=16Ki 扫 N）launch 主导，e/persist 升到 **1.54×**、
      e/graph 1.57×（graph≈persist）。结论：**损失三随 N 标度（融合收益上界≈N×）、
      损失二只在轻负载+多算子时显形**。正确性 max|Δz|=0。

### P10 实验扩展 2（`docs/04-experiments.md`）

- [x] **P10-1** 扫 persistent kernel 的网格大小 + 量化设备内协调成本随网格的标度
      （对应「下一步」可选方向②：P5-1 §1.6「扫 persistent 网格大小」+ P5-3 §3.6「多 block
      资源约束」）。见 `docs/04-experiments.md` §5，源码 `src/grid_sweep.cu`，原始输出
      `harness/grid_sweep.out.txt`（目标卡 H100/sm90，threads=256，iters=200）：
      Section A（固定 N=8192 单元扫 G 1→1056）speedup 123.5×(@#SM=132)→**212.9×(@占用上限
      1056)**，故并行度**在占用率上限而非 #SM 处饱和**（256 线程/SM 只有 12.5% 占用）；
      `ns/unit/blk` 在 G≤#SM 稳 ~2.2–2.4µs、G>#SM 抬升（1056→12.2µs）=墙钟/效率取舍；
      Section B（N=64 扫 G 跨 N）speedup **恰在 G=N=64 封顶 33.8×**、G>N 持平（过配纯浪费）；
      Section C（cooperative `grid.sync()` 扫 G）有 **~1.1µs 固定地板**（G=1 也 1133ns，来自
      全局内存栅栏），G>#SM 后次线性增长（528→1.73µs、1056→2.40µs）。结论：persistent 网格
      应按**占用率**选（`min(占用上限,N)`）而非按 SM 数选，对上 MPK §5.1 worker/scheduler
      划分；~1.1µs 硬屏障地板解释了生产 mega-kernel 为何用异步原子 ring barrier/去中心化
      event queue。正确性：三种网格 sink[0]=12288 一致。

### P11 单 task device 实现源码走读（`docs/07-mpk-tasks.md`）

- [x] **P11-1** MPK 单 task device 实现走读（Hopper）：补齐 `docs/06` 停在「worker 调用
      `_execute_task`」的下一层——**一个 task 内部是一段怎样的 device 代码**。见
      `docs/07-mpk-tasks.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`
      同快照）。§0 编译层映射（`task_register.cc:1181-1318`：形状→编译期常量→TMA 类型别名→
      `linear_kernel_hopper<...>` 调用，一个 task = 一个 CTA）；§1 `linear_hopper.cuh` 精读
      （warp specialization 生产者/消费者、`Kstages` K/N 双重流水、TMA+mbarrier transaction-count、
      swizzled `smem_tma`/`mma_descriptor`、wgmma、寄存器内残差融合）；§2 支撑原语
      （`barrier.cuh` mbarrier / `tma.cuh` / `wgmma.cuh` / `utils.cuh` setmaxnreg（被注释））；
      §3 异构 task 形态（`rmsnorm_hopper` cp.async 归约 / `silu_mul_hopper` 纯逐元素 /
      `allreduce.cuh` NVSHMEM 通信 task / `norm_linear_hopper` 段内融合）；§4 task 内同步
      （mbarrier）vs task 间同步（原子队列）两套机制；§5 与论文 §4.2/§5.3 对照 + 实现延伸
      （wait_group 0 流水浅、单 warp 发 TMA、纯 superoptimization→手写/CUTLASS 混合）；§6 判据映射
       （task 层只贡献 C3）。**纯源码走读（资料分析），本轮无新实验**。

### P12 实验扩展 3（`docs/04-experiments.md`）

- [x] **P12-1** 多 block PDL 的 occupancy 资源约束（补齐 P5-3 §3.6「多 block PDL 资源约束」
      与 P10-1 §5.5「PDL 后继 grid 的 block 何时拿到 SM 资源」两条尾巴）。
      见 `docs/04-experiments.md` §6，源码 `src/pdl_multiblock.cu`，原始输出
      `harness/pdl_multiblock.out.txt`（目标卡 H100/sm90，iters=30，blockDim=256/128 各一遍）：
      **上游 occupancy 是 PDL 重叠的第一约束**——上游 `Gu` 只占机器一小部分（≈4 block/SM）
      时有效重叠≈理想（~1.00×ideal）；上游逼近/占满占用上限时滑到 **~0.74×ideal**（两种
      blockDim 收敛到同一天花板）。Section B 证**下游 grid 大小不是约束**（总占用≤cap 时重叠
      恒定），约束在上游侧。Section C 证满占用下重叠**饱和**于 ~0.74×ideal、`prologue=0` 时
      重叠≈0（无「可提前做的活」即无收益）。含义：PDL 是「向上游借 occupancy 余量」，
      而 megakernel 恰恰把余量占满（§5、MPK §5.1），故它改在单 kernel 内做 task 级重叠。
      正确性 fail=0。**本轮新增实验。**

### P13 实验扩展 4（`docs/04-experiments.md`）

- [x] **P13-1** 单 kernel 内跨 task 软件流水：把**损失一**（kernel barrier 禁止跨算子软件流水）
      实测出来（补齐 P5-2 §2.6 / §5.6「未做跨 tile software pipelining」的尾巴）。
      见 `docs/04-experiments.md` §7，源码 `src/in_kernel_pipeline.cu`，原始输出
      `harness/in_kernel_pipeline.out.txt`（目标卡 H100/sm90，iters=30，G=132，T=256，E=8，
      276 MB 输入 ≫ L2）：同一 kernel 内「`cp.async` 载入 chunk → R 轮 FMA 消费」依赖链，
      唯一变量是流水深度 `K`（`K=1` 阶段串行 = kernel barrier / 不做流水的 persistent kernel）。
      Section A（R=64 扫 K）：**K=1→2 拿到 1.18×、再加深无增益**（us/chunk 2.885→2.436→平）；
      Section B（K=4 扫 R）：R=0 纯访存 **2.17×**（藏相邻拷贝延迟）、随 R 增大 speedup→理论上界
      `(Tmem+Tcomp)/max(Tmem,Tcomp)`，R≥128 时 speedup/上界=1.00（访存完全藏进计算）；
      Section C（R=64,K=4 扫 chunks）：16→256 收敛 1.06×→1.18×（序言填流水一次性成本）。
      结论：**损失一在单 kernel 内可回收，量级 1.18×，正落在 MPK §6.6（行 1586–1596）报告的
      1.2–1.3×**；增益几乎全在深度 1→2，解释了 MPK/DeepGEMM 为何用 `Kstages=3/6`（`docs/07` §1.2）；
      比 P5-2 的 `grid.sync()` 形态更接近 megakernel（异步可重叠协调而非全局会合），但仍**不满足 C5**
      （无去中心化 worker/scheduler 调度）。**本轮新增实验**；对应当前「下一步」可选方向②/③之外的
      机制补测（损失一此前只有资料分析）。

### P14 实验扩展 5：megakernel 的成本侧（`docs/04-experiments.md`）

- [x] **P14-1** megakernel 的资源足迹代价：**寄存器被固定为各 task 类型所需的最大值**
      （MPK §7 行 1740–1748 明确点名的唯一例外，此前仅资料分析）。
      见 `docs/04-experiments.md` §8，源码 `src/megakernel_footprint.cu`，原始输出
      `harness/megakernel_footprint.out.txt`（目标卡 H100/sm90，iters=30，G=528=4 block/SM，T=256，
      轻输入 537 MB ≫ L2）。四个 Section：A 寄存器/占用率表（`fused_light<H>` 的 numRegs **恰等于**
      `heavy_kernel<H>` = `max(light,heavy)`，无 spill）；B 轻路径代价扫描（同一轻工作，kernel 里带
      重 task 就让带宽掉：H≤48→1.00×、H=64→**1.33×**（24 warp/SM）、H≥128→**2.29×**（8 warp/SM），
      对应占用率 32→24→8）；C 时分复用场景（单 kernel 轻后重 vs 两次 launch：H=64→**1.27×**、
      H=128→**1.79×**，账 = 轻阶段被降速部分 + 重阶段）；D SMEM 足迹 occupancy 敏感性（32 KB/block→
      6 block/SM、160 KB→1）。核心结论：**异构 task 融合时，轻 task 因最重 task 的寄存器需求损失
      1.33×–2.29× 访存带宽（占用率被拉低）**；MPK 用时分复用 + paged SMEM（§5.3 行 1313–1325）规避
      了 SMEM 与计算资源，**唯寄存器不可分页**（§7 行 1748）——本节从正反两面量化印证该论断。
       判据不变（C1–C3，无 C5）；是 megakernel 的**成本侧**度量，与 P5/P9/P13 收益侧互补。**本轮新增实验。**

### P15 实验扩展 6：最小去中心化调度 megakernel（`docs/04-experiments.md`）

- [x] **P15-1** 把 MPK §5 的**去中心化调度（C5）**用原生 CUDA 真正跑起来（此前只有资料分析与
      源码走读，本专题所有 persistent 骨架都停在 `grid.sync()` 集中式、明确「不满足 C5」）。
      见 `docs/04-experiments.md` §9，源码 `src/decentralized_sched.cu`，原始输出
      `harness/decentralized_sched.out.txt`（目标卡 H100/sm90，iters=100，G=132=1 block/SM，T=256）：
      最简 DAG = C 条链×L 层串行依赖，三策略（eager L 次 launch / barrier 单 cooperative kernel +
      逐层 `grid.sync()` / decen 单 kernel + 设备侧原子 ready 队列 + 每 task 事件计数器，无全局屏障）。
      - **C5 机制可用且不动数值**：6 个 case 三策略**逐位一致**（互比误差 0.00e+00）。
      - **均衡负载下 decen ≈ barrier ≈ eager**（0.99–1.02×）；**小负载（每层 task 数<block 数）
        decen 慢到 0.556×**——空队列 `__nanosleep` 自旋开销。
      - **收益在「task 多且异构」时的动态负载均衡**：C=528≫G=132、段长伪随机 1–8×，**barrier 仅
        1.007×，decen 1.167×**（对上 MPK §6.4 行 1551–1553 混合负载均衡动机）。
      - barrier 相对 eager 只快 1.007–1.033×，印证 P10-1 的 `grid.sync()` 在 132 block 上 ~1.1µs
        **本就很便宜**——去中心化要赢在**均衡/结构**而非「省屏障」。
      判据：decen 是 C1–C5 全满足的 T4 最小骨架。**本轮新增实验。**


### P16 实验扩展 7：megakernel 的成本侧 2（`docs/04-experiments.md`）

- [x] **P16-1** megakernel 的**代码足迹代价**：整图融合 = 许多「per-task device 函数」挤进一个
      kernel 镜像（MPK §4.2 行 1138 / §7 行 1750、1760），被执行代码工作集超过 L1 指令缓存后
      取指 miss。见 `docs/04-experiments.md` §10，源码 `src/code_footprint.cu`，原始输出
      `harness/code_footprint.out.txt`（目标卡 H100/sm90，device 3，G=1056=8 block/SM，T=256，
      steps/block=512，共 1.38 亿元素，iters=30；单 body=65 FFMA=1040 B SASS）。三个 Section：
      A 代码足迹扫描（固定总工作量扫被执行 task 类型数 P=1…256）：P≤8 无代价、P=32 起 1.30×、
      **P=256 达 2.18×（32.1→14.7 TFLOP/s）**，拐点对应执行工作集 ~8KB→33KB→266KB；
      B **代码工作集隔离**（同表同调用数同工作量，spread=每流扫 P 个 body vs sticky=每流固定 1 个）：
      **sticky 在所有 P 下恒 ~32.7 TFLOP/s（=基准）、spread 到 P=256 慢 2.21×**——代价来自每条
      执行流的**时间代码工作集**而非分发表大小；C ncu 归因：spread P=1→256 的
      `no_instruction`（取指停顿）**0.63%→59.19%**，而 `branch_resolving` 反降、`imc_miss`≈0、
      指令数仅 +8%、周期 ×2.01，sticky P=256 回到 1.61%——**纯 L1i miss**。与 P14-1 的寄存器足迹
       合为 megakernel 的**双层不可分页足迹**（寄存器按各 task 最大值 / 指令按各 task body 累积）；
       对生产案例的解释：DeepGEMM mega 家族只融合**少数极重** task 类型正是对的方向。**本轮新增实验。**

### P17 实验扩展 8：双层足迹的叠加（`docs/04-experiments.md`）

- [x] **P17-1** 把 P14-1 的**寄存器足迹**与 P16-1 的**代码足迹**编进**同一个 kernel** 测合成
      （补齐 §8.6 / §10.6 两条明确写着的「未在同一 kernel 内叠加测」的尾巴）。
      见 `docs/04-experiments.md` §11，源码 `src/two_layer_footprint.cu`，原始输出
      `harness/two_layer_footprint.out.txt`（目标卡 H100/sm90，device 3，iters=30，G=528=4 block/SM，
      T=256，steps/block=512，访存数组 537 MB ≫ L2）。一个 kernel `mega<P,H>` 同时含三条路径
      （occupancy 敏感的访存路径 / I-cache 敏感的 P 个互异 `body` 代码路径 / 抬寄存器用的重路径），
      `do_mem`/`do_comp` 运行时隔离，P、H 编译期可控。核心结论：(1) **两笔足迹如实复现**——
      B1 访存 `H=128` 2.305×（对上 §8 的 2.29×）、B2 代码 `P=256,H=0` 2.106×（对上 §10 的 2.18×）；
      Section A 证**同 H 下 P=1 与 P=256 的 numRegs 完全相同**（26/72/168）→ 代码足迹不动寄存器；
      (2) **纯计算路径上两笔代价在绝对时间上「相加」而非相乘**——占用率差 0.8709 ms + 代码差
      0.3688 ms，独立相加预测 1.5731 ms vs 实测 both 1.5747 ms（差 0.1%），相乘预测 7.61× vs
      实测 4.723×；**代码足迹绝对增量几乎不随占用率变**（+0.3688 vs +0.3704 ms），「低占用率放大
      取指停顿」的假设不成立；(3) **混合负载下总代价略低于边际相加**（预测 2.3563 ms vs 实测
      2.0149 ms，低 ~14%，跨 phase 重叠折扣）；(4) **绝对组合惩罚仍大**——纯计算 both 4.723×、
      混合 both 3.795×。要点：**不能用「各 task 足迹倍率相乘」估 megakernel 成本**；「少而重」而非
      「多而杂」是同时规避两层足迹的同一个设计选择（对上 DeepGEMM `docs/02`）。判据不变（C1–C3，
      无 C5）。**本轮新增实验**；与 P14-1/P16-1 合成成本侧完整链条。

### P18 实验扩展 9：寄存器足迹的可绕开性（`docs/04-experiments.md`）

- [x] **P18-1** megakernel 寄存器足迹的**可绕开性**：用 `setmaxnreg` warp 级寄存器再平衡收回
      P14-1 的代价（补齐 §8.6 / §11.6 两条「未测 setmaxnreg 动态重配置」尾巴，并兑现 §11.5 的假设；
      对应「下一步」可选方向④延伸）。见 `docs/04-experiments.md` §12，源码 `src/setmaxnreg_rebalance.cu`，
      原始输出 `harness/setmaxnreg_rebalance.out.txt`（目标卡 H100/sm90，**`-gencode arch=compute_90a,
      code=sm_90a`**，device 3，iters=30，G=528，T=256，n=537 MB ≫ L2）。三组对照（`separate` /
      `fused_naive` / `ws(U0)` 不调 setmaxnreg / `ws(U1)` 调）：
      **A** `ws<*,lb2,U1>` 的 `numRegs` 被 `__launch_bounds__(256,2)` 压到 **128**（2 block/SM，是
      `fused_naive` 的 2 倍），但 `heavy<192>` 重路径 **localB=0（无 spill）**——ptxas 因 setmaxnreg
      而不压重路径寄存器；`ws<128,U0>` 同样 128 regs 但 **spill 32 B**；
      **B** 复现 §8-B（naive 轻访存带宽掉 1.34×–2.29×）；
      **C**（收益侧）`ws(U1)` 相对 `fused_naive` **0.54×–0.77×**（快 1.3×–1.8×）、逼近甚至优于
      `separate`（**0.92×–1.24×**，H≤192 因轻重 warpgroup 同 kernel 并发而反超）；而 `ws(U0)` 在
      H≥192 因 spill 爆炸（比 U1 慢 **7.7×**）。核心结论：**warp-specialized 只是前提，真正避免
      spill、收回代价的是 setmaxnreg**；它要求「显式 warp 角色 + 入口寄存器预算」，故 MPK 自动
      路径（`docs/07` §2 注释掉 setmaxnreg）与「一个 task=一个 CTA」的粒度用不上，而 DeepGEMM
      手工角色正好用熟。判据不变（C1–C3，无 C5）。**本轮新增实验。**

### P19 实验扩展 10：PDL × CUDA Graph 叠加（`docs/04-experiments.md` §13）

- [x] **P19-1** 补齐 `docs/04` §6.6「未测 PDL 与 CUDA Graph 叠加」与 `docs/03` §2.3「可叠加」论断：
      **把 PDL 链 capture 成 CUDA Graph，测二者能否『相乘』**。见 `docs/04-experiments.md` §13，
      源码 `src/pdl_graph.cu`，原始输出 `harness/pdl_graph.out.txt`（目标卡 H100/sm90，固定 device 3，
      iters=30）。K 个 `chain_kernel` 串成 PDL 链，四策略公平对照 `eager`/`ePDL`/`graph`/`gPDL`
      （正确性哨兵 `expect=12345` 烘进 kernel，排序被破坏则 `fail` 累加）。三 Section：A 大 span
      pl=tail=200000、B 大 span tail=3×pl、C 小 span pl=tail=4000（launch 可感）。结论：
      **Q1 CUDA Graph capture 完整保留 PDL 语义**——`gPDL/g` 逐行等于 `e/ePDL`（A 到 1.87×、B 1.31×）、
      四策略 `fail` 全 0；**Q2 叠加是次乘不是相乘**——C 段 K=64：`graph` 1.18×、`ePDL` 1.82×、
      相乘预测 2.15×，实测 `gPDL` 仅 **1.97×（~0.92× 乘积）**，因二者挤占同一段「kernel 间串行缝」
      （Graph 单独省 ~1.05µs/kernel，PDL 之下边际降到 ~0.28µs/kernel）；**Q3 上限独立**——
      A/B 大 span 下 `graph≈eager`（launch 可忽略）、PDL 上限仍是 `min(tail,prologue)/span`，
      加不加 Graph 都不变。判据不变（`graph`/`gPDL` 全不满足 C1–C5），是 kernel-per-operator 两块
      降级拼一起也补不齐的       降级项。**本轮新增实验**。

### P20 实验扩展 11：跨算子细粒度重叠（`docs/04-experiments.md` §14）

- [x] **P20-1** 把 MPK §2.1（快照行 241–293，Figure 3 的 `MatMul→AllGather`）与 §3（行 371–386）
      点名的**跨算子细粒度 kernel overlap** 实测出来（P13-1 只覆盖「单 task 内」；P5-2/P15-1 的
      DAG 是**整层**依赖）。见 `docs/04-experiments.md` §14，源码 `src/cross_op_overlap.cu`，原始输出
      `harness/cross_op_overlap.out.txt`（目标卡 H100/sm90，device 3，iters=30，G=132，T=256，
      NT=8192 tile×TS=4096，producer 访存受限/计算受限可调、consumer 寄存器内 R 轮 FMA）。
      三策略公平对照：`serial`（2 eager kernel）/`barrier`（单 coop kernel + `grid.sync()`）/
      `pipe`（单 kernel，`Gp` producer + `Gc` consumer block **并发** + 逐 tile ready 标志 `gen`）。
      核心结论：(1) **逐 tile 细粒度依赖数值安全**——`barrier`/`pipe` 相对 `serial` 输出**逐位一致**
      （max|Δ|=0），`s/b≈1.00×`（损失二在本区间不显形）；(2) **但静态切 block 的细粒度重叠在所有
      情况下都慢于算子级屏障**（`pipe/b` 恒 <1，0.29–0.87×），原因两层且被量出：**(a) 两阶段都要
      ~整台机器**——Section D 显示访存受限的 producer 直到 GB≈132 才喂满 HBM（256 线程/block 占用率低）、
      计算受限的 consumer 到 1056 block 仍在加速（2.41×、未饱和），故「完美重叠下界」
      `ideal=max(Tp(Gp),Tc(Gc))` 至多与 `Tp+Tc` 打平（R≥128 时反而更大）；**(b) 朴素逐 tile 同步开销大**
      ——`pipe` 比 `ideal` 还慢 1.07–2.22×（每 tile 一次原子+两次 `threadfence`+`Gc` block 自旋）；
      (3) producer 变计算受限（Section B）把 `pipe/b` 从 0.83× 压到 0.41×，反向印证 MPK「不同硬件资源」
      前提；(4) Section C 有内点最优 `Gp=32`，仍不翻盘。判据：`barrier` C1/C2/C4、`pipe` 加 C3 但静态角色
      无 C5。**含义**：跨算子重叠的载体必须是**共享的 SM 内共驻**（P13-1/P18-1 warp 特化）而非**分割的
      block**，同步必须**批量/环形化**（MPK ring buffer、DeepGEMM ring barrier）而非每 tile 原子。
      **本轮新增实验**，补齐损失一的**跨算子**形态。

### P21 MPK 依赖分析器源码走读（`docs/08-mpk-compiler.md`）

- [x] **P21-1** 补齐 `docs/06`（运行层，从 tGraph 之后讲起）与 `docs/07`（单 task device）
      **上游**的编译层：论文 §4.1 的「算子图 → tGraph」五步（分解/依赖分析/event fusion/
      normalization/linearization）在线上实现里到底长什么样。见 `docs/08-mpk-compiler.md`
      （快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`/`docs/07` 同快照；
      主文件 `src/kernel/annotated_graph.cc` 1014 行 + 头 `include/mirage/kernel/annotated_graph.h` 198 行）。
      §0 定位（`runtime.cc:243` 调用、`:642` 按 `ordered_layers` emit）与十步流水线 (a)–(j)；§1 IR
      （`TaskView` 的 `event_dim`（张量维）vs `last3`（网格轴）、`derive_last3`、边身份 =
      `(prod_layer,out_slot,cons_layer,in_slot)` 而非 guid）；§2 Step (a) **最近写者 SSA** + bbox
      `intersect/subtract` **阴影覆盖**（`:119-425`）+ view 引发的 barrier 边（`:391-400`）；§3 Step (b)
      Kahn 环检测 / (c) **残余边剥离=传递规约**（`:429-551`，一次可达性再原子剥离）；§4 Step (d) distinct
      计数 / (e) **case 2/3 拒绝**（`:553-662`）——论文 normalization 是 Figure 6 空-task 重写，实现是
      **拒绝非法角色组合 + LCM 降级**；§5 Step (f) 两趟 Kahn（depth,index 定序，`:664-745`）=论文
      Algorithm 1 linearization；§6 Step (g) 逐边 **GCD**（`:770`）+ (h) **fork LCM**（`:786-885`）+
      (i) **join LCM**（`:887-958`）=论文 successor/predecessor-set fusion（Def 4.1/4.2），论文用集合论、
      实现用分块算术，目标一致；§7 `runtime.cc` 把 IR 展开成 `EventDesc`/`FullTaskDesc`
      （`dfs_create_events_add_tasks:76-219`、fork 交错 emit `:748-795`、join `:810-872`）；§8 与论文
      §4.1 逐条对照表；§9 局限（纯走读未构建、`parallel_path_design.md` 在快照里不存在、normalization
      口径差异）。**纯源码走读（资料分析），本轮无新实验。** 配套博客 post `mega-kernel-09-mpk-compiler`
      （weight 10）并更新 `mega-kernel-00-index` 目录。

### P22 实验扩展 12：去中心化调度的 DAG 形状敏感性（`docs/04-experiments.md` §15）

- [x] **P22-1** 把 P15-1 的「C 条链」推广到**分层扇入/扇出图**（L 层 × W 宽 × FIN 前驱），
      同一图/工作量上对照四种设备内协调机制：`barrier`（逐层 grid.sync）/`wave`（层软屏障 +
      原子工作计数器，可批量取 task）/`decen`（逐 task 原子 ready 队列 + 事件计数器，P15-1 机制
      + 扇入扇出）/`static`（静态 grid-stride + 事件计数器轮询，无中央队列）。见
      `docs/04-experiments.md` §15，源码 `src/fanin_sched.cu`，原始输出
      `harness/fanin_sched.out.txt`（目标卡 H100/sm90，device 3，iters=20，G=132，T=256）。
      五 Section：A 扫 W、B 扫 FIN、C 扫 wave 批量 K、D 扫 task 长度 SEG、E 扫层数 L。核心结论：
      (1) 四机制在 12 case 上 **fail=0**、vs 串行参考 max rel|Δ| 6e-8–1.2e-7（C5 机制对任意
      扇入/扇出正确）；(2) **严格分层均衡图上集中式 `barrier` 恒最快**（wave/static 0.82–0.99×、
      decen 0.27–0.47×），且 `grid.sync` 地板相对每层工作量占比小而「多而薄」的层数增加（E，
      L 32→128）也不翻盘——去中心化要赢在均衡/结构（对上 P15-1、P10-1）；(3) **`decen` 的每 task
      固定开销**：固定 W=512 把 SEG 4096→32768，`decen/b` 从 **0.278×→0.956×**（长 task 摊薄），
      解释 P15-1「小负载慢」；(4) MPK §5.2 **批量取 task 甜点 K≈W/G**（K=4 1.028×、K=16 骤降
      0.530×、K=32 0.270×）；(5) `static` 无 work-stealing，随扇入 FIN 变差（0.852→0.821×）；
      (6) 实现坑：自拼「数据写 + 独立 done flag 发布」在大扇入下间歇读错，改用逐 task 事件计数器
      （MPK §5.1 event 同构）才稳。判据：四者均无固定角色 worker/scheduler，仍差 C5 一步。
      **本轮新增实验。**

## 每轮增量流程（autopilot）

1. 读本文件「任务清单」选**一个未完成子项**（优先 P0→P1→…）。
2. 写/改 `docs/<NN>-*.md`（会被内联到博客，**里面不要写 {{< >}} 短代码**）与必要的 `src/`、`harness/`。
3. 论文/源码精读贴**真实章节号/行号**；线上资料给真实链接（不要编造 URL）。
4. 能跑的用 `scripts/lab.sh exec "..."` 跑真实数字留 `*.out.txt`；跑不了的标注「仅资料分析」。
5. `/tmp/hugo_bin/hugo --gc --minify` 确认无 ERROR；`git add` 只加本专题路径；commit 前缀 `megakernel:`；
   `git pull --rebase` 后 push（网络失败只本地 commit 并在本文件记「待推送」）。
6. 更新本文件状态与「当前进度/下一步」。
7. **不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。**

## 已知坑

- 目标卡代号敏感：写「目标卡」。
- 不要编造 URL；线上资料只给已知过得住的（MPK 论文/GitHub、DeepGEMM、hpc-ops）。
- 被内联的 `docs/*.md` 里**不要写 `{{< >}}` 短代码**；站内链接只在 `content/posts` 里用 `relref`。
- 新建博客 post 的 front matter `date` **不要写成未来时间**：Hugo 默认不构建 future 内容，
  会导致 `relref` 报 `REF_NOT_FOUND`（症状：`hugo list all` 能看到该页，但引它会失败）。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。

## 当前进度 / 下一步

- 当前进度：脚手架就绪；**P0 全部完成**；**P1 全部完成（3/3）**；**P2 全部完成（4/4）**；
  **P3 全部完成（3/3）**；**P4 全部完成（4/4）**；**P5 全部完成（3/3）**；**P6 全部完成（1/1）**；
  **P7 源码走读（1/1）完成**（本轮新增，对应「下一步」可选方向③）。
  **P8 博客发布（1/1）完成**（本轮新增，对应「下一步」可选方向①）。
  **P9-1 实验扩展（1/1）完成**（对应「下一步」可选方向②）。
  **P10-1 实验扩展 2（1/1）完成**（续做「下一步」可选方向②中剩余的两点）。
  **P11-1 单 task device 实现走读（1/1）完成**（对应「下一步」可选方向③「走读
  `tasks/hopper/` 具体 task 的 device 实现」）。
  **P12-1 多 block PDL occupancy 资源约束（1/1）完成**（本轮新增，补齐 P5-3 §3.6 / P10-1 §5.5
  两条尾巴，完成「下一步」可选方向②中最后一点）。
  **P13-1 单 kernel 内跨 task 软件流水（1/1）完成**（本轮新增，把此前只有资料分析的**损失一**
  实测出来，补齐 P5-2 §2.6 / §5.6 的未覆盖项）。
  **P14-1 megakernel 资源足迹代价（1/1）完成**（本轮新增，把 MPK §7「寄存器固定为各 task 最大值」
  这条只有资料分析的**成本侧**局限量化出来，补齐 `docs/01` §5.8 / MPK §7 行 1740–1748）。
  **P15-1 最小去中心化调度 megakernel（1/1）完成**（本轮新增，把 MPK §5 的 **C5 机制**用原生
  CUDA 跑起来，补齐此前所有 persistent 骨架「不满足 C5」的空白）。
  **P16-1 megakernel 代码足迹代价（1/1）完成**（本轮新增，把 MPK「每 task 一个 device function」
  这条只有资料分析的**指令足迹/成本侧**局限量化出来，与 P14-1 的寄存器足迹合成「双层足迹」）。
  **P17-1 双层足迹叠加（1/1）完成**（本轮新增，把 P14-1 的寄存器足迹与 P16-1 的代码足迹编进**
  同一个 kernel** 测合成律，补齐 §8.6 / §10.6 两条「未叠加」尾巴）。
  **P18-1 寄存器足迹的可绕开性（1/1）完成**（用 `setmaxnreg` warp 级再平衡把 P14-1 的
  代价基本收回，补齐 §8.6 / §11.6「未测 setmaxnreg」两条尾巴并兑现 §11.5 的假设）。
  **P19-1 PDL × CUDA Graph 叠加（1/1）完成**（本轮新增，补齐 `docs/04` §6.6「未测 PDL 与 CUDA
  Graph 叠加」与 `docs/03` §2.3「可叠加」论断：capture 保留 PDL 语义、叠加次乘非相乘、上限独立）。
  **P20-1 跨算子细粒度重叠（1/1）完成**（本轮新增，把 MPK §2.1 Figure 3 的 `MatMul→AllGather` 型
  跨算子细粒度依赖实测出来：数值安全但静态切 block 重叠恒慢于屏障，重叠需 SM 共驻而非分块）。
  **P21-1 MPK 依赖分析器源码走读（1/1）完成**（本轮新增，补齐 `docs/06`/`docs/07` 上游的编译层：
  论文 §4.1 的「算子图 → tGraph」五步在 `src/kernel/annotated_graph.cc` 的真实实现——最近写者 SSA +
   bbox 阴影覆盖、残余边剥离、case 2/3 拒绝、fork/join LCM event fusion、按 depth 的 Kahn linearization）。
  **P22-1 去中心化调度的 DAG 形状敏感性（1/1）完成**（本轮新增，把 P15-1 的链推广到分层扇入/
  扇出图、对照 barrier/wave/decen/static 四机制、实现 MPK §5.2 批量取 task 的甜点标度）。
  **本专题任务清单（P0–P22）全部完成**，无未完成子项。
  - **P0-1** MPK 论文章节地图见 `docs/00` §6；
  - **P0-2** 案例源码坐标清单见 `docs/02-production-cases.md`（DeepGEMM mega 家族 `057ca59`、
    hpc-ops `ab1a402` README 行 230、Kimi K3 §4.2.4 快照行 1069–1081）；
  - **P0-3** 环境探测完成，见 `docs/04-experiments.md` §0：H100/sm90 + CUDA 13.2 上
    persistent kernel（cooperative grid sync）、CUDA Graph、PDL **三项均实测可编译可运行**，
    后续 P4/P5 无降级项。
  - **P1-1** kernel-per-operator 三类损失完成：`docs/00` §1 按「损失一 kernel barrier / 损失二 launch 开销 /
    损失三中间张量 HBM 往返」展开，逐条贴 MPK 快照真实行号（§1 行 74–89、§2.1 行 223–293、§2.2 行 311–331、
    §6.4 行 1551–1553、§6.6 行 1586–1628、§6.3 行 1501–1513），并复用 P0-3 实测佐证。
  - **P1-2** 已有缓解手段与边界完成：`docs/00` §2 分 2.1 算子融合 / 2.2 CUDA Graphs / 2.3 PDL / 2.4 多流 /
    2.5 通信-计算 overlap / 2.6 相关抽象 / 2.7 小结表，逐条贴 MPK 快照行号，并给出 P1-2 实测
    （`src/mitigations.cu`、`harness/mitigations.out.txt`）：无依赖 4 流 3.65×、有依赖 4 流 0.96×、
    依赖链 graph 1.05×。结论：这些手段停在算子粒度或静态图，未触及 SM 级跨算子调度。
  - **P1-3** 定义与谱系完成：`docs/00` §3（定义 + 五条判据 C1–C5 + 术语澄清）、§4（四档谱系 T1–T4，
    手工 FlashDMoE/Spector vs 自动 MPK；DeepGEMM 三者为段级 mega-kernel），判据与 §2 缓解手段对齐。
  - **P2-1** SM 级图表示（tGraph）完成：新建 `docs/01-mpk-paper.md`，§2 精读论文 §3
    （快照行 345–448）：task/event 交替语义与激活规则（行 360–370）、MatMul→AllReduce 细粒度重叠
    （行 371–386）、同图多解与图 4(c) 反例（行 387–401）、与 CUDA Graphs 五维对比（行 402–414）、
    Figure 5 normalization/linearization 预告（行 415–448）；纯资料分析，复用 P1-2 实测作佐证。
  - **P2-2** MPK 编译器（§4）完成：`docs/01-mpk-paper.md` §3 精读论文 §4（快照行 449–1146）与 §6.7
    （行 1629–1737）：tGraph 生成五步（decomposition→dependency analysis→event fusion（Def 4.1/4.2）
    →normalization（Figure 6 空 task 降扇入/扇出到 1）→linearization（Algorithm 1 BFS 连续区间编码）），
    以及 §4.2 以 thread block 为单位的 Mirage superoptimization；量化：32–47 task/op、fusion 37–118×、
    linearization 4.4–15×（B200，Table 2）。纯资料分析。
  - **P2-3** in-kernel runtime（§5）完成：`docs/01-mpk-paper.md` §4 精读论文 §5（行 1147–1368）与
    §6.1/§6.6：worker/scheduler 角色划分（4 SM/16 warp scheduler + 其余 worker，固定于 launch）、
    事件驱动执行、JIT/AOT 混合下发（两步 vs 一步同步、global barrier 分界）、三项运行时优化
    （paged shared memory / cross-task software pipelining / task descriptor 预取）、
    §6.6 消融（pipeline 1.2–1.3×、compute–comm overlap 1.1×、launch 293→in-kernel scheduler 0.28%）。
     纯资料分析，复用 P0-3/P1-2 实测佐证。
  - **P2-4** 评估（§6 + §7）完成：`docs/01-mpk-paper.md` §5。§6 四问组织（行 1371–1390）；
    §6.1/§6.2 实现与设定（`torch.compile(backend=MPK)`、offline batched、batch 1–16、
    prompt 64/生成 1024、page 32KB、5 模型×3 代 GPU）；§6.3 单 batch 1.0–1.7×
    （Qwen3-8B/A100 14.5→12.5ms；vs 原生 PyTorch >10×）；§6.4 MoE 混合负载均衡 + fused
    gather-GEMM（11%）；§6.5 TP/8×H100 相对 PyTorch 最高 10×、相对 SGLang/vLLM 1.1–1.4×；
    §6.6/§6.7 消融归档；§7 Discussion（寄存器固定为各 task 最大值、per-task codegen 唯一硬件相关、
    手写 kernel 可包成 task）；§5.8 复现要点 + 8 条局限（仅 LLM serving / offline / batch≤16 /
    多 GPU 只测 dense TP / 每 batch size 一张静态图 / 去中心化调度 / 寄存器上限 / page size 固定）。
- **P3-2** Mega Gate / Mega mHC 完成：`docs/02-production-cases.md` §1.2/§1.3 各扩为完整精读（融了哪些
  算子、为什么、单 kernel 边界 / 静态角色 / 设备内协调 / 重叠 / 与 MPK 逐维对照 / 落地约束），沿用 §1.5
  框架。核心结论：**三者是同一套「持久化 CTA + 手工静态角色 + 设备内异步原子 barrier + SMEM 环形流水」
  的段级 mega-kernel，差别只在 barrier 数量与角色数**（MoE：位掩码 + ring generation；Gate：ScoreBarrier
  + split-K 归约；mHC：双 SplitBarrier + 注册重配置）。纯资料分析（SM100 专属，本机 H100/sm90 不运行）。
- **P3-3** Kimi K3 + hpc-ops 工程视角完成：`docs/02` §2 扩为「roadmap 定位 / 已实现的 PDL 段融合 /
  工程侧结论」，§3 扩为「原文要点（快照行 1069–1081）/ 对本专题的两个关键印证（CUDA Graph 被评测
  系统视作绕过、精度缩减判作弊）/ 与 DeepGEMM、MPK 对照表 / 四段式工程结论」。核心判断：
  **megakernel 产业进展呈「需求(K3 训练套件)→路线(hpc-ops roadmap)→落地(DeepGEMM 段级 T4)→
  自动化(MPK 整图 T4)」四段式**。纯资料分析。
- **下一步：无**（P0–P21 全部完成）。后续可选的增量方向（不属现有清单）：① ~~为 `docs/02`/`03`/`04`/`06`/`07`
  补建对应博客 post~~（P8-1 完成 `02`/`03`/`04`/`06`；P11-1 补建 `mega-kernel-08-mpk-tasks`）；
  ② ~~补做 P5-2 §2.6 / P5-1 §1.6 提到的「更小 M / 更多算子 / 多 block」实验~~（**P9-1 完成 N 算子链 + 小 M**；
  **P10-1 完成 P5-1 §1.6「扫 persistent 网格大小」**；**P12-1 完成 P5-3 §3.6 / P10-1 §5.5「多 block
  PDL occupancy 资源约束」**；**P13-1 完成 P5-2 §2.6 / §5.6「跨 tile software pipelining」的损失一实测**；
  **P14-1 完成 megakernel 成本侧「寄存器固定为各 task 最大值」实测**；**P15-1 完成 C5 去中心化调度最小实现**）；
  ③ ~~走读 `tasks/hopper/*.cuh` 单 task 内部实现~~（P11-1 完成）；剩「在容器内 `pip install -e .`
  构建 mirage sm90 路径并跑 `demo/qwen3/demo.py --use-mirage` 取本机 MPK 端到端数字」（构建成本高，未做）；
  ④ ~~megakernel 成本侧「代码足迹 / 指令缓存」实测~~（**P16-1 完成**，与 P14-1 寄存器足迹合成双层足迹）；
  ⑤ ~~megakernel 成本侧「寄存器足迹的可绕开性」：`setmaxnreg` 实测~~（**P18-1 完成**，收回 P14-1 的代价）；
  ⑥ ~~`docs/04` §6.6「未测 PDL 与 CUDA Graph 叠加」~~（**P19-1 完成**，测出 capture 保留 PDL 语义、
  叠加次乘非相乘、上限独立）；
  ⑦ ~~P15-1 §9.6「未测扇入/扇出 DAG、批量/预取取 task」~~（**P22-1 完成**：分层扇入/扇出图 ×
  barrier/wave/decen/static 四机制、MPK §5.2 批量取 task 甜点 K≈W/G、静态分配随扇入变差）；
  剩「在容器内 `pip install -e .` 构建 mirage sm90 路径并跑端到端数字」（构建成本高，未做）；
  剩「跨层流水/环形缓冲预取」（P22-1 §15.6 标注未做，是 MPK §5.3 的下一层机制）。
- **本轮（P22-1）**：`docs/04-experiments.md` 新增 §15「去中心化调度的 DAG 形状敏感性：扇入/扇出 ×
  静态分配 vs 动态队列」——新建 `src/fanin_sched.cu` + `harness/fanin_sched.out.txt`（目标卡 H100/sm90，
  device 3，iters=20，G=132=#SM，T=256）。把 P15-1 的链推广成**分层扇入/扇出图**（`t=l*W+j`，前驱 =
  层 l-1 的 `(j*FIN+k) mod W`，FIN 路汇聚；数值 = 本地段和 + Σ前驱），四机制公平对照：
  **barrier**（逐层 grid.sync）/ **wave**（层软屏障 + 原子工作计数器，支持批量领 K 个）/
  **decen**（逐 task 原子 ready 队列 + 事件计数器，P15-1 机制 + 扇入扇出）/ **static**（静态 grid-stride
  + 事件计数器轮询，无中央队列）。五 Section：A 扫 W（256→2048）、B 扫 FIN（1/4/8）、C 扫 wave 批量 K、
  D 扫 task 长度 SEG（4096→32768）、E 扫层数 L（32/64/128）。核心结论：(1) **12 case 四机制全 fail=0**、
  vs 串行参考 max rel|Δ| 6e-8–1.2e-7（C5 机制对任意扇入/扇出正确）；(2) **严格分层均衡图上 `barrier`
  恒最快**（wave/static 0.82–0.99×、decen 0.27–0.47×），Section E 层数 32→128 不翻盘——每层工作量
  ≥ 几 µs 时 `grid.sync` 地板（~1.1µs，P10-1）占比小，去中心化要赢在均衡/结构（对上 P15-1 §9.4）；
  (3) **`decen` 是每 task 固定开销**：W=512 固定、SEG 4096→32768，`decen/b` **0.278×→0.956×**（长
  task 摊薄固定开销），给 P15-1 §9.4「小负载慢到 0.556×」一般化解释；(4) MPK §5.2 **批量取 task 甜点
  K≈W/G**（K=4 →1.028×；K=16 骤降 **0.530×**、K=32 **0.270×**，因超每 worker 每层均摊 task 数导致
  层末尾空转）；(5) `static` 无 work-stealing，随扇入 FIN 变差（0.852→**0.821×**）；(6) 实现坑：
  自拼「写 val → threadfence → 独立 done flag」发布在大扇入下间歇读错（rel 1e-2–1e0），改用逐 task
  事件计数器（`atomicSub` 归零，= MPK §5.1 event、`docs/06` §3）才稳。判据：四机制均满足 C1–C4、
  无固定角色 worker/scheduler，仍差 C5 一步。**本轮新增实验**，补齐 P15-1 §9.6 两条尾巴。
- **本轮（P21-1）**：新建 `docs/08-mpk-compiler.md`（MPK 依赖分析器 / tGraph 构建源码走读，**纯源码走读、
  无新实验**）+ 博客 post `mega-kernel-09-mpk-compiler`（weight 10）并更新 `mega-kernel-00-index` 目录第 9 项。
  快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06`/`docs/07` 同一快照，行号已核对）。
  补齐 `docs/06`（从 tGraph 之后讲起）上游的编译层：论文 §4.1 五步在 `src/kernel/annotated_graph.cc`
  （1014 行）+ `include/mirage/kernel/annotated_graph.h`（198 行）的真实实现。§0 定位（`runtime.cc:243`
  调用、`:642` 按 `ordered_layers` emit）与十步流水线 (a)–(j)；§1 IR（`TaskView` 的 `event_dim`（张量维）
  vs `last3`（网格轴）、`derive_last3:42-58`、边身份 = `(prod,out_slot,cons,in_slot)` 而非 guid）；
  §2 Step (a) **最近写者 SSA**（`last_writers:154`）+ bbox `intersect/subtract` **阴影覆盖**（`:331-369`）
  + view 引发 barrier 边（`:391-400`）；§3 Step (b) Kahn 环检测 / (c) **残余边剥离=传递规约**
  （`:488-550`，一次可达性再原子剥离）；§4 Step (d) distinct 计数 / (e) **case 2/3 拒绝**（`:583-662`）
  ——**论文 normalization 是 Figure 6 空-task 重写，实现是「拒绝非法角色组合 + LCM 降级」**；§5 Step (f)
  两趟 Kahn（depth,index 定序 `:664-745`）= 论文 Algorithm 1 linearization；§6 Step (g) 逐边 **GCD**
  （`:770`）+ (h) **fork LCM**（`:786-885`）+ (i) **join LCM**（`:887-958`）= 论文 successor/predecessor-set
  fusion（Def 4.1/4.2），论文用集合论、实现用分块算术，目标一致；§7 `runtime.cc` 把 IR 展开成
  `EventDesc`/`FullTaskDesc`（`dfs_create_events_add_tasks:76-219`、fork 交错 emit `:748-795`、join
  `:810-872`）；§8 与论文 §4.1 逐条对照表；§9 局限（纯走读未构建、`parallel_path_design.md` 在快照里
  并不存在、normalization 口径差异是硬边界）。**本轮纯资料分析，无新实验。**
- **本轮（P20-1）**：`docs/04-experiments.md` 新增 §14「跨算子细粒度重叠：为什么『切 block 做重叠』
  赢不过算子级屏障」——新建 `src/cross_op_overlap.cu` + `harness/cross_op_overlap.out.txt`（目标卡
  H100/sm90，device 3，iters=30，G=132=SM，T=256，NT=8192 tile×TS=4096 float=134 MB/数组）。
  构造 MPK §2.1 Figure 3 型 workload：producer（`buf=src+1`，`float4`，`PR` 轮 FMA 调访存/计算受限）
  -> consumer（每线程 16 元素读进寄存器一次后 `R` 轮 FMA 归约）；tile 1:1 依赖。三策略：`serial`
  （2 eager kernel）/`barrier`（单 coop kernel + `grid.sync()`）/`pipe`（单 kernel，`Gp` producer +
  `Gc` consumer block 并发 + 逐 tile ready 标志（`atomicExch`+`gen`+`threadfence`））。四个 Section：
  **A**（PR=0 扫 R）、**B**（R=80 扫 PR）、**C**（R=80 扫 Gp）、**D**（单阶段随 block 数缩放）。
  核心结论：(1) **逐 tile 细粒度依赖数值安全**——`barrier`/`pipe` 相对 `serial` **逐位一致**
  （max|Δ|=0.000e+00），`s/b≈1.00×`；(2) **静态切 block 的细粒度重叠恒慢于算子级屏障**
  （`pipe/b` 0.29–0.87×），两层原因：**(a)** Section D 表明访存受限 producer 到 GB≈132 才喂满 HBM
  （0.358→0.103 ms，256 线程/block 占用率低）、计算受限 consumer 到 1056 block 仍在加速
  （0.287→0.119 ms，2.41×），故 `ideal=max(Tp(Gp),Tc(Gc))` 至多与 `Tp+Tc` 打平（R≥128 反而更大）；
  **(b)** `pipe` 比 `ideal` 还慢 1.07–2.22×（每 tile 原子+两次 `threadfence`+`Gc` block 自旋）；
  (3) Section B：producer 变计算受限把 `pipe/b` 从 0.83× 压到 **0.41×**，反向印证 MPK「不同硬件资源」
  前提；(4) Section C 内点最优 `Gp=32`（`pipe/b=0.83×`）仍不翻盘。判据：`barrier` C1/C2/C4、
  `pipe` 加 C3 但静态角色无 C5。**含义**：跨算子重叠的载体是**共享 SM 内共驻**（P13-1/P18-1 warp 特化）
  而非**分割 block**，同步须**批量/环形化**（MPK ring buffer、DeepGEMM ring barrier）。
  **本轮新增实验**，补齐损失一的**跨算子**形态。
- **本轮（P19-1）**：`docs/04-experiments.md` 新增 §13「PDL 与 CUDA Graph 的叠加：两种降级手段能不能
  『相乘』？」——新建 `src/pdl_graph.cu` + `harness/pdl_graph.out.txt`（目标卡 H100/sm90，固定 device 3，
  iters=30）。K 个 `chain_kernel`（`[prologue]→gridDependencySync→[crit 校验/写 seq]→trigger→[tail]`）
  串成 PDL 链，四策略公平对照：`eager`（无 PDL）/`ePDL`（有 PDL）/`graph`（把 eager capture 成图）/
  `gPDL`（把 ePDL capture 成图）；正确性哨兵 `expect=12345` 烘进 kernel、`seq` capture 前清 0，
  PDL 排序被破坏则下游读到 0 → `fail` 累加。三 Section：A 大 span pl=tail=200000、B 大 span tail=3×pl、
  C 小 span pl=tail=4000/crit=1000（span≈4.6µs，与 launch 同量级）。核心结论：
  **Q1 CUDA Graph capture 完整保留 PDL 语义**——`gPDL/g` 在 A/B/C 逐行等于 `e/ePDL`（A 到 **1.87×**、
  B **1.31×**），四策略 `fail` 全 0（否则 `gPDL` 会退化成 `graph`）；**Q2 叠加是次乘非相乘**——C 段 K=64：
  `graph` **1.18×**、`ePDL` **1.82×**、相乘预测 2.15×，实测 `gPDL` 仅 **1.97×（~0.92× 乘积）**，因二者
  挤占同一段「kernel 间串行缝」（Graph 单独省 ~1.05µs/kernel，PDL 之下边际降到 ~0.28µs/kernel）；
  **Q3 上限独立**——A/B 大 span（0.2–0.8ms）下 `graph≈eager`（差<0.5%，损失二只在 launch-bound 显形）、
  PDL 上限仍是 `min(tail,prologue)/span`，加不加 Graph 都改不了。判据：`graph`/`gPDL` 全不满足 C1–C5，
  是 kernel-per-operator 两块降级拼一起也补不齐的降级项。**本轮新增实验**，补齐 §6.6 的尾巴。
- **本轮（P18-1）**：`docs/04-experiments.md` 新增 §12「megakernel 寄存器足迹的『可绕开性』：
  `setmaxnreg` warp 级再平衡」——新建 `src/setmaxnreg_rebalance.cu` +
  `harness/setmaxnreg_rebalance.out.txt`（目标卡 H100/sm90，`-gencode arch=compute_90a,code=sm_90a`，
  device 3，iters=30，G=528，T=256，n=537 MB ≫ L2）。三组对照（`separate` = kernel-per-operator /
  `fused_naive` = P14-1 朴素融合 / `ws(U0)` = warp-specialized **不调** setmaxnreg / `ws(U1)` = 调）：
  **A** `__launch_bounds__(256,2)` 把 `ws<*,U1>` 入口寄存器压到 **128**（2 block/SM，`fused_naive` 的
  2 倍），但 `heavy<192>` **localB=0 无 spill**（ptxas 因 setmaxnreg 不压重路径）；`ws<128,U0>` 同 128
  regs 却 **spill 32 B**；**B** 复现 §8-B（naive 轻访存带宽 1.34×–2.29×）；**C**（收益侧）`ws(U1)`
  相对 `fused_naive` **0.54×–0.77×**（快 1.3×–1.8×）、相对 `separate` **0.92×–1.24×**（H≤192 因轻重
  warpgroup 同 kernel 并发而反超），而 `ws(U0)` 在 H≥192 因 spill 爆炸（比 U1 慢 **7.7×**）。核心：
  **warp-specialized 是前提，setmaxnreg 才是收回寄存器足迹代价的关键**；它要求「显式 warp 角色 +
  入口寄存器预算」，故 MPK 自动路径（`docs/07` §2 注释掉 setmaxnreg、一 task=一 CTA）用不上，
  DeepGEMM 手工角色正好用熟。判据不变（C1–C3，无 C5）。**本轮新增实验**，补齐 §8.6 / §11.6 两条尾巴、
  兑现 §11.5 的假设。另记编译坑：`setmaxnreg` 只在 `sm90a` 可用，且无入口寄存器数时 ptxas 打印
  `(C7508) 'setmaxnreg' ignored` 并静默忽略。
- **本轮（P17-1）**：`docs/04-experiments.md` 新增 §11「megakernel『双层足迹』的叠加：寄存器足迹 ×
  代码足迹」——新建 `src/two_layer_footprint.cu` + `harness/two_layer_footprint.out.txt`（目标卡
  H100/sm90，**固定在空闲的 device 3**，iters=30，G=528=4 block/SM，T=256，steps/block=512，访存数组
  537 MB ≫ L2）。一个 kernel `mega<P,H>` 同时含三条路径：occupancy 敏感的**访存路径**（537 MB
  grid-stride 求和）、I-cache 敏感的**代码路径**（P 个互异 `body`，block 粒度经函数指针表间接调用）、
  **重寄存器路径**（`heavy_path<H>`，只调用一次/block 只为抬寄存器，`H=0` 时 `if constexpr` 丢弃）。
  `do_mem`/`do_comp` 运行时隔离，P、H 编译期可控，总工作量与 P/H 无关。四个 Section：**A** 寄存器/
  占用率表（**同 H 下 P=1 与 P=256 的 numRegs 完全相同** 26/72/168，8/3/1 block/SM = 64/24/8 warp/SM
  → 代码足迹不动寄存器，结构上独立）；**B1** 访存隔离 `H=128` **2.305×**（对上 §8 的 2.29×）；
  **B2** 代码隔离 `P=256,H=0` **2.106×**（对上 §10 的 2.18×）；**B3** 叠加。核心结论：
  (1) 两笔足迹如实复现；(2) **纯计算路径上二者在绝对时间上「相加」而非相乘**——占用率差 0.8709 ms
  + 代码差 0.3688 ms，独立相加预测 **1.5731 ms** vs 实测 both **1.5747 ms**（差 0.1%），相乘预测
  7.61× vs 实测 4.723×；**代码足迹绝对增量几乎不随占用率变**（H=0 +0.3688、H=128 +0.3704 ms），
  「低占用率放大取指停顿」不成立；(3) **混合负载总代价略低于边际相加**（预测 2.3563 ms vs 实测
  2.0149 ms，低 ~14%，跨 phase 重叠折扣）；(4) **绝对组合惩罚仍大**——纯计算 both 4.723×、混合 both
  3.795×。要点：**不能用「各 task 足迹倍率相乘」估 megakernel 成本**；「少而重」而非「多而杂」是
  同时规避两层足迹的同一个设计选择（对上 DeepGEMM `docs/02`）。判据不变（C1–C3，无 C5）。
  **本轮新增实验**；与 P14-1/P16-1 合成成本侧完整链条，补齐 §8.6 / §10.6 两条「未叠加」尾巴。
- **本轮（P16-1）**：`docs/04-experiments.md` 新增 §10「megakernel 的代码足迹代价：融合的 task
  类型数 -> 执行代码工作集 -> I-cache」——新建 `src/code_footprint.cu` +
  `harness/code_footprint.out.txt`（目标卡 H100/sm90，**固定在空闲的 device 3**，因 device 0 当时被
  其它并行任务占满；G=1056=8 block/SM，T=256，steps/block=512，1.38 亿元素，iters=30；单个
  task body = 65 条 FFMA = **1040 B SASS**，cuobjdump 核对）。构造「一个 kernel 内含 P 个互异
  device function body、以 block 粒度经函数指针表间接调用」的 workload，**总工作量与 P 无关**，
  唯一变量 P = 融合并实际执行的 task 类型数。三个 Section：**A**（固定工作量扫 P=1…256）P≤8 无代价、
  P=16 起 1.10×、P=32 1.30×、**P=256 2.18×**（32.1→14.7 TFLOP/s），拐点对应执行代码工作集
  ~8KB→33KB→266KB；**B**（同 P、同表、同调用数、同工作量；`spread` 每流扫 P 个 body vs `sticky`
  每流固定 1 个）**sticky 在所有 P 下恒 ~32.7 TFLOP/s（= P=1 基准）、spread 到 P=256 慢 2.21×**
  ——代价来自**每条执行流的时间代码工作集**而非分发表大小；**C**（ncu 归因）spread P=1→256 的
  `smsp__warp_issue_stalled_no_instruction_per_warp_active` **0.63%→59.19%**，而 `branch_resolving`
  反降（4.9%→2.2%）、`imc_miss`≈0、执行指令数仅 +8%（376.7→407.0M）、周期 ×2.01，sticky P=256
  回到 1.61% / 基准周期——**纯 L1 指令缓存 miss / 取指带宽**。核心结论：**代码足迹是 megakernel 一项
  真实、可测、且与寄存器足迹（P14-1）独立的不可分页代价**；MPK §6.7 只把 **元数据** footprint
  降了 4.4–5.9×（行 1680、1701），per-task codegen 的**指令** footprint 未量化，本节补上。判据不变
  （C1–C3，无 C5）；对生产案例的解释：DeepGEMM mega 家族只融合**少数极重** task 类型正是对的方向。
  **本轮新增实验**；与 P14-1 合成 megakernel 的「双层足迹」成本画像。
- **本轮（P15-1）**：`docs/04-experiments.md` 新增 §9「最小『去中心化 in-kernel 调度』megakernel：把
  C5 机制跑起来」——新建 `src/decentralized_sched.cu` + `harness/decentralized_sched.out.txt`（目标卡
  H100/sm90，iters=100，G=132=1 block/SM，T=256）。最简 DAG = C 条链×L 层串行依赖，三策略公平对照：
  **eager**（L 次 launch）/ **barrier**（单 cooperative kernel + 逐层 `grid.sync()`）/ **decen**
  （单 kernel + **设备侧原子 ready 队列 + 每 task 事件计数器、无全局屏障**，= MPK §5 去中心化调度微缩版）。
  核心结论：(1) **C5 机制可用且不动数值**——6 个 case 三策略**逐位一致**（互比误差 0.00e+00），
  vs 串行参考 max rel ≤ 4.1e-7（浮点累加序）；(2) **均衡负载下 decen ≈ barrier ≈ eager**（0.99–1.02×），
  而**每层 task 数<block 数时 decen 慢到 0.556×**（空队列 `__nanosleep` 自旋）；(3) **收益在「task 多且
  异构」时的动态负载均衡**——C=528≫G=132、段长伪随机 1–8×，**barrier 仅 1.007×、decen 1.167×**（对上
  MPK §6.4 行 1551–1553 混合负载均衡）；(4) barrier 相对 eager 只快 1.007–1.033×，印证 P10-1 的
  `grid.sync()` 在 132 block 上 ~1.1µs **本就很便宜**——**去中心化要赢在均衡/结构而非「省屏障」**。
  判据：decen 是 C1–C5 全满足的 T4 最小骨架（此前所有 persistent 骨架都停在 `grid.sync()`、明确不满足 C5）。
  **本轮新增实验**；补齐 C5 的实测空白，与 P13-1（损失一）、P14-1（成本侧）合起来覆盖 megakernel 的
  「收益 / 成本 / 调度机制」三面。
- **本轮（P14-1）**：`docs/04-experiments.md` 新增 §8「megakernel 的资源足迹代价：寄存器被固定为各
  task 最大值」——新建 `src/megakernel_footprint.cu` + `harness/megakernel_footprint.out.txt`（目标卡
  H100/sm90，iters=30，G=528=4 block/SM，T=256，轻输入 537 MB ≫ L2）。构造「轻访存 task（占用率敏感）
  + 重计算 task（H 个常驻寄存器）」并把两类编进同一 kernel，隔离出 MPK §7（行 1740–1748）点名的**唯一
  例外——per-thread 寄存器固定为各 task 最大值**。四个 Section：A 寄存器/占用率表（**`fused_light<H>`
  的 numRegs 恰等于 `heavy_kernel<H>` = `max(light, heavy)`**、无 spill）；B 轻路径代价扫描（同一 537 MB
  轻访存，kernel 里带重 task 就让带宽 2589→1944→1132 GB/s：H≤48→**1.00×**、H=64→**1.33×**（24 warp/SM）、
  H≥128→**2.29×**（8 warp/SM），对应占用率 32→24→8）；C 时分复用场景（单 kernel「轻后重」vs 两次 launch：
  H=64→**1.27×**、H=128→**1.79×**，账 = `light_降速(0.474)+heavy(0.117)`≈0.583 与实测吻合）；D SMEM 足迹
  occupancy 敏感性（32 KB/block→6 block/SM、100 KB→2、160 KB→1）。核心结论：**异构 task 融合时轻 task 因
  最重 task 的寄存器需求损失 1.33×–2.29× 访存带宽（占用率被拉低、无 spill）**；MPK 用时分复用 + paged
  SMEM（§5.3 行 1313–1325）规避了 SMEM 与计算资源，**唯寄存器不可分页**——本节从正反两面量化印证 §7 论断。
  判据不变（C1–C3，无 C5）；是 megakernel 的**成本侧**度量，与 P5/P9/P13 收益侧互补。**本轮新增实验。**
- **本轮（P13-1）**：`docs/04-experiments.md` 新增 §7「单 kernel 内跨 task 软件流水：把损失一测出来」——
  新建 `src/in_kernel_pipeline.cu` + `harness/in_kernel_pipeline.out.txt`（目标卡 H100/sm90，iters=30，
  G=132，T=256，E=8，输入 276 MB ≫ L2）。同一 kernel 内「`cp.async` 载入 chunk → R 轮 FMA 消费」依赖链，
  唯一变量是流水深度 `K`（`K=1` = 阶段串行 = kernel barrier / 不做流水的 persistent kernel）。三组扫描：
  Section A（R=64 扫 K）：**K=1→2 拿到 1.18×、再加深无增益**（us/chunk 2.885→2.436→平）；
  Section B（K=4 扫 R）：R=0 纯访存 **2.17×**（藏相邻拷贝延迟）、R 越大 speedup→理论上界
  `(Tmem+Tcomp)/max(Tmem,Tcomp)`（R≥128 时 speedup/上界=1.00，访存完全藏进计算）；
  Section C（R=64,K=4 扫 chunks）：16→256 收敛 1.06×→1.18×（序言填流水一次性成本）。
  核心结论：**损失一在单 kernel 内可回收，量级 1.18×，正落在 MPK §6.6（行 1586–1596）报告的 1.2–1.3×**；
  增益几乎全在深度 1→2，解释了 MPK/DeepGEMM 为何用 `Kstages=3/6`（`docs/07` §1.2 的 K/N 双重流水）；
  比 P5-2 的 `grid.sync()` 形态更接近 megakernel（异步可重叠协调、无全局会合），但仍**不满足 C5**
  （无去中心化 worker/scheduler 调度）。**本轮新增实验**；对应「补测损失一」这一机制空白（此前仅资料分析）。
- **本轮（P12-1）**：`docs/04-experiments.md` 新增 §6「多 block PDL 的 occupancy 资源约束」——
  新建 `src/pdl_multiblock.cu` + `harness/pdl_multiblock.out.txt`（目标卡 H100/sm90，iters=30，
  blockDim=256/128 各一遍）。把 P4-2/P5-3 的 `grid=1` 一对 kernel 扩到多 block，三组扫描：
  Section A（`Gd=1` 扫上游 `Gu`）、B（`Gu=cap/2` 扫下游 `Gd`）、C（`Gu=cap` 扫 `prologue`）。
  核心结论：**上游 occupancy 是 PDL 重叠的第一约束**——上游仅占机器一小部分（≈4 block/SM）时
  有效重叠≈理想（~1.00×ideal）；上游逼近/占满占用上限时滑到 **~0.74×ideal**（两种 blockDim 同一
  天花板）。**下游 grid 大小不是约束**（Section B：总占用≤cap 时重叠恒定）。满占用下重叠**饱和**
  于 ~0.74×ideal、`prologue=0` 时≈0。含义：**PDL 是「向上游借 occupancy 余量」**，而 megakernel
  恰恰把余量占满（本文件 §5、MPK §5.1 行 1148–1174），故它改在单 kernel 内用去中心化 event queue
  做 task 级重叠。判据映射不变（PDL 仍全不满足 C1–C5）。正确性 fail=0。**本轮新增实验**；
  完成「下一步」可选方向②中最后一点（多 block PDL 资源约束）。
- **本轮（P11-1）**：新建 `docs/07-mpk-tasks.md`（MPK 单 task device 实现走读，**纯源码走读，无新实验**）
  + 博客 post `mega-kernel-08-mpk-tasks`（weight 9）并更新 `mega-kernel-00-index` 目录第 8 项。
  快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（与 `docs/06` 同一快照，行号已核对）。
  补齐 `docs/06` 停在「worker 调用 `_execute_task`」的下一层：§0 编译层映射
  （`task_register.cc:1181-1318`，形状→`Kstages`/`output_atom_size`/TMA 类型别名→
  `linear_kernel_hopper<...>` 调用；**一个 task = 一个 CTA**，Hopper 256 线程恰好铺满 worker block）；
  §1 `linear_hopper.cuh` 精读（warp specialization：生产者单 warp lane0 发 TMA + mbarrier
  transaction-count 预取 A/B/残差 `:219-260`，消费者 wgmma + swizzled descriptor `:274-308`，
  **寄存器内融合残差** `:331-339`，TMA store 回写 `:342-357`；`Kstages=3/6` 做 K/N 双重流水）；
  §2 支撑原语（`barrier.cuh` mbarrier、`tma.cuh`、`wgmma.cuh`、`utils.cuh` setmaxnreg **被注释**）；
  §3 异构 task 形态（`rmsnorm_hopper` cp.async 两级归约、`silu_mul_hopper` 纯逐元素、
  `allreduce.cuh` NVSHMEM 通信 task、`norm_linear_hopper` 段内融合）；§4 **task 内同步（mbarrier）
  与 task 间同步（原子队列 + 事件计数器）是两套机制**，论文 §5.3 cross-task pipelining 由运行层负责；
  §5 与论文 §4.2（行 1126–1146）/§5.3（行 1312–1368）对照 + 实现延伸（`mma_async_wait` wait_group 0
  流水浅、单 warp 发 TMA、论文「纯 superoptimization」线上演化为手写模板 + CUTLASS 封装混合）；
  §6 判据映射（**task 层只贡献 C3**，C1/C2 由运行层、C4/C5 由 worker/scheduler 给）。
  与生产案例对照：`linear_hopper` 是 DeepGEMM mega 家族「单 task 尺度」的对应物（`docs/02` §1）。**P11 完成。**
- **本轮（P10-1）**：`docs/04-experiments.md` 新增 §5「persistent kernel 的网格大小与设备内协调成本」——
  新建 `src/grid_sweep.cu` + `harness/grid_sweep.out.txt`（目标卡 H100/sm90，threads=256，iters=200），
  补齐 P5-1 §1.6（网格未扫）与 P0-3/P5-2（只测过一个网格点的屏障成本）。三个子实验：
  Section A（固定 N=8192 单元扫 grid 1→1056）证 **并行度在占用率上限饱和而非 #SM**——speedup
  123.5×(@#SM=132)→212.9×(@1056)，因 256 线程/block 单 block/SM 只占 12.5% warp 槽位；
  `ns/unit/blk` 在 G≤#SM 稳、G>#SM 抬升（墙钟/效率取舍）。Section B（N=64 扫 G 跨 N）证
  **speedup 恰在 G=N 封顶、过配纯浪费**。Section C（cooperative `grid.sync()` 扫 G）证
  **硬屏障有 ~1.1µs 固定地板**（G=1 也 1133ns，全局内存栅栏）、G>#SM 后次线性增长到 2.4µs。
  工程结论：persistent 网格按占用率选 `min(占用上限,N)`（对上 MPK §5.1 worker/scheduler 划分），
  ~1.1µs 硬屏障地板解释生产 mega-kernel 为何走异步原子 ring barrier / 去中心化 event queue
  （MPK §5/§6.1，DeepGEMM 见 `docs/02` §1.5）。正确性 sink[0]=12288 一致。**本轮新增实验**；
  对应「下一步」可选方向②中剩余两点之一（扫 persistent 网格）。
- **本轮（P9-1）**：`docs/04-experiments.md` 新增 §4「N 算子链的融合：损失随 N 的标度」——
  新建 `src/n_op_fusion.cu` + `harness/n_op_fusion.out.txt`（目标卡 H100/sm90，iters=100）。
  把 P5-2 的两算子融合推广到 N 个依赖算子（eagerN / graphN / persistN / fusedN），
  Section A（N=8 扫 M）、B（M=16Mi 扫 N）、C（M=16Ki 扫 N）。核心结论：
  **损失三随 N 标度**——eagerN 表观带宽恒 ~2.75 TB/s、fusedN 流量恒 8 B/elem，故
  e/fused 在 N≤8 近似 N×（1.98/3.91/7.13×），N≥16 因融合自身转入计算主导而饱和（9.4/10.3×）；
  **损失二只在 launch-bound 区间显形**——大 M 下 e/persist≈1.00×，小 M+大 N 下升到 1.54×；
  本机 graph≈persist（grid 启动代价 ≈ grid barrier 代价）。正确性 max|Δz|=0。
  判据映射同 P5-2。**本轮新增实验**；对应的可选方向②（更多算子/更小 M）已完成。
- **本轮（P8-1）**：博客发布接线——为 `docs/02`/`03`/`04`/`06` 补建 4 个 `doc_include` 内联 post
  （`mega-kernel-03-cases`/`-04-related`/`-05-experiments`/`-07-mpk-source`，weight 4/5/6/8），
  并更新 `mega-kernel-00-index` 目录（原 3/4/5 项「待写」→ 真实 relref 链接，新增第 7 项源码走读）。
  **纯发布接线，无新资料、无新实验**。`hugo` 构建：8 个 mega-kernel 页面全部生成、本专题**无 ERROR**
  （唯一站点 ERROR 是另一并行专题 flash-kda 的在途文件 `flash-kda-04-numerics`，不属本专题路径）。
  踩坑：post `date` 初填当晚 21:48–21:51（写入后数分钟的未来时间）触发 **REF_NOT_FOUND**，
  回填到过去后消失（印证「date 不要写成未来时间」坑）。P8-1 完成。
- **本轮（P7-1）**：新建 `docs/06-mpk-source.md`（MPK 线上实现运行层源码走读，**纯资料分析**）。
  快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（2026-10-05）。走通「Python 构图 →
  `runtime.cc` 生成 `_init_persistent_kernel`/`_execute_task` → `init_persistent_kernel` 分配队列/事件 →
  `launch_persistent_kernel` 单次常驻」全链路：§1 编译层与 Python 前端（`persistent_kernel.py:413/501/3207`、
  `_execute_task` 生成 `runtime.cc:1885-1903`）；§2 数据结构（`runtime_header.h` 的 TaskId/EventId 编码、
  `TaskDesc` 128b、`EventDesc.num_triggers`、`RuntimeConfig`）；§3 worker 循环 `execute_worker`
  （`:957-1252`，`ld.acquire` 轮询游标 + `cp.async` 批量取描述符 + 事件计数阈值
  `num_triggers*iteration` 精确等依赖 + `atom.add.release` 入 sched 队列）；§4 scheduler 循环
  `execute_scheduler`（`:1254-1523`，4 warp/SM、事件三档 LAUNCH/MASSIVE/DEPENDENT 派发、
  `END_OF_TASK_GRAPH`→`prepare_next_batch` 连续服务）；§5 init/prepare/launch（`:1610-2005`，
  prepare 预置启动事件、单 kernel 或 worker/scheduler 双 stream）；§5.3 内存序原语 `mpk_atoms.cuh`；
  §6 与论文逐项对照表；§7 C1–C5 全满足。关键印证：`utils.py:25-29` 的 `scheduler=4*(sm-worker)` 与
  H100（132 SM）→128 worker/16 scheduler 精确对上论文 §5.1。**P7 完成；本轮无新实验**（mirage 完整
  编译器/运行时构建成本过高，未在容器内编译运行）。
- **本轮（P6-1）**：新建 `docs/05-summary.md`（收官汇总，**纯资料汇总、无新实验**）——§1 原理一页纸
  （三类损失 ↔ C1–C5 判据 ↔ MPK 编译层/运行层两层结构 ↔ 收益因果链 ↔ 谱系与两种落地路径 ↔ 与既有手段
  边界）；§2 设计 checklist（判定/融合边界/依赖调度/资源正确性/落地工程五组，每行对应已查证机制或坑，
  如 precise 依赖、event fusion、normalization 定长编码、warmup wave 防死锁、grid idx generation、
  寄存器上限、paged SMEM）；§3 何时该/不该用（各 6 条 + 决策速查表）；§4 本专题实测校准表
  （P0-3/P1-2/P4-1/P5-1/P5-2/P5-3，原始输出 `harness/*.out.txt`）；§5 关键坐标。新建博客 post
  `content/posts/mega-kernel-06-summary/index.md`（weight 7，`doc_include` 内联）并更新
  `mega-kernel-00-index` 目录第 6 项。**P6 完成、P0–P6 全部收官。**
- **本轮（P5-3）**：`docs/04-experiments.md` 新增 §3「PDL 链式流水（P5-3）」——新建 `src/pdl_pipeline.cu`
  + `harness/pdl_pipeline.out.txt`（目标卡 H100/sm90，iters=30，K=1/2/4/8/16）。把 P4-2 的「一对 kernel」
  推广到「K 个 kernel 用 PDL 串成链」，每 kernel = `[prologue]→gridDependencySync→[临界段 crit]→trigger→[tail]`，
  用 `seq[K+1]`+代次 gen 做正确性门限（fail=0）。结果：no-PDL 每 kernel ≈ `span=pl+crit+tl` 严格串行；
  PDL 每条链边重叠 **恰好 `min(prologue,tail)`**，稳态每 kernel = `span−min(prologue,tail)`。
  A 组 pl=tail=200000：speedup 1.00→1.87×（K=16，趋 2.0×）；B 组 tail=3×pl：1.00→1.31×（趋 1.33×，
  被较短的 prologue 封顶，与「尾巴越长越能藏」的直觉相反）。**核心结论：PDL 链式收益不随 K 累积**，
  上限由 `min(prologue,tail)` 在单 kernel 时长中的占比决定——它是「只有 1 级的 kernel 级流水」，
  每级仍需 host 逐个 launch、中间量仍可落 HBM。与 MPK §5.3（行 1326–1363）单 kernel 内 task 级、
  可跨算子可累积的软件流水形成对照，是 `docs/03` §2.4「粒度停在 kernel 对」的量化证据。
  判据映射与 P4-2 一致：全不满足 C1–C5。局限见 §3.6（grid=1、合成段、均匀链）。P5-3 完成、P5 收官；
  本轮新增实验。
- **本轮（P5-2）**：`docs/04-experiments.md` 新增 §2「两算子融合成单 persistent kernel（P5-2）」——
  新建 `src/two_op_fusion.cu` + `harness/two_op_fusion.out.txt`（目标卡 H100/sm90）。对一对逐元素
  依赖算子 `y=f(x)` → `z=g(y)`，比较四种策略：eager2（2 launch，中间量落 HBM）/ graph2（1 次提交，
  仍走 HBM）/ barrier（单 cooperative persistent kernel + `grid.sync()`，仍走 HBM）/ fused（单
  persistent kernel，中间量只在寄存器）。差分对齐 `docs/00` §1 三类损失：eager2→barrier 度量
  **损失二**、barrier→fused 度量**损失三**。结果（M=16/64/256Mi 元素，带宽主导区间）：
  **barrier/fused ≈ 2.0×**（16→8 B/elem，四者表观带宽一致 ~2.8 TB/s），故**损失三是主导收益**；
  eager2/barrier ≈1.0×、eager2/graph2 ≈1.0×，**损失二在本区间近乎为零**，直接印证 P5-1「launch 只在
  launch-bound 有意义」；cooperative grid sync（660 block×256 线程）相对 eager2 无可测开销（呼应
  P0-3 的 ~1.65 µs/次）。正确性 max|Δz|=0。判据映射：fused 满足 C1–C3（持久化版算子融合，T2 上限）；
  barrier 补齐 C4（单 kernel 内协调）→ 段级 megakernel 最小骨架。局限见 §2.6（未做跨 tile software
  pipelining / 去中心化调度，barrier 是硬同步而非生产级异步原子 ring barrier）。P5-2 完成，
  下一步 P5-3；本轮新增实验。
- **本轮（P5-1）**：`docs/04-experiments.md` 新增 §1「Launch 开销实测（P5-1）」——对 N 个独立
  等量单元比较 eager / CUDA Graph / 单 persistent kernel（新 `src/launch_overhead.cu`、
  `harness/launch_overhead.out.txt`，目标卡 H100/sm90）。Section A（K=0）：eager per-单元恒定
  ~1.75–2.2µs 随 N 线性、graph ~2.1×（不归零）、persistent 总时间恒定 ~2µs（N=4096 达 1988×）。
  Section B（N=256 扫 K）：新增 `persist1`(grid=1) 与 eager 设备侧工作等价，二者随 K 由 48.9×
  收敛到 1.12×，**证明差额就是 launch 开销**；`persistG`(grid=132) 靠并行度保持 133.8×；
  graph 也趋 1.06×。结论：graph 消 host 侧 N 次提交、persistent 连设备侧 N 次 grid 启动也消，
  只在 launch-bound 区间有意义；与 MPK §6.6 launch 293→0.28% 的宏观锚点一致。P5-1 完成，
  下一步 P5-2；本轮新增实验。
- **本轮（P4-4）**：`docs/03-related-tech.md` §4 补入「通信-计算 overlap」精读——4.1 两种真实机制：
  DeepEP 在独立通信流上用 CUDA event 与 MoE 计算重叠（`async_with_compute_stream` / `previous_event` /
  `defer_epilogue` + `register_hook_after_wait`，`deep_ep/utils/event.py:39-71`、`deep_ep/buffers/ep.py`、
  `README.md:261-289`），FlashMLA 用 PDL 重叠 `splitkv_mla`+`combine`（上游 trigger
  `splitkv_mla.cuh:1226`、下游 grid-dependency-sync `combine.cu:59`、双侧 launch attribute
  `splitkv_mla.cuh:1338-1348`/`combine.cu:207-219`、deep-dive `:60`，并注意到源码保留「不用 PDL」注释）；
  4.2 重叠对象/锚点（复用 P4-2 `min(tail,prologue)`，MPK §6.3 行 1600–1616 overlap 1.1×）；4.3 未消除项
  （粒度=kernel/集合算子、host 调度权、中间张量落 HBM、launch 不变）；4.4 判据映射（FlashMLA 仅部分 C4）；
  4.5 小结。**P4-4 完成、P4 收官**；纯资料分析 + 复用既有实测，本轮无新实验。
- **本轮（P4-3）**：`docs/03-related-tech.md` §3 补入 persistent kernel vs megakernel 与 in-kernel 调度精读——
  3.1 澄清「persistent=C1+C2 是机制、megakernel=C1–C5 是范围」；3.2 persistent 的三项机制收益（无 kernel barrier /
  跨算子软件流水 §2.1 行 235–239、§2.2 行 313–331 / 时分复用资源折中 §7 行 1741–1749）；3.3 去中心化调度机理
  （worker 占物理 SM + scheduler 占 warp、4 scheduler warp/SM 共 16、固定于 launch；event 驱动、统一表示计算与通信、
  不依赖拓扑画像；本地状态 + device 原子 ring buffer 避免全局协调，§6.1 行 1410–1413；混合 JIT/AOT 两步 vs 一步；
  descriptor 352B 预取）；3.4 本机可验证边界（复用 P0-3：cooperative grid sync 0.330ms/200 次、132 SM，
  并**如实标注**它是中心化屏障、与 MPK 去中心化 event queue 不同，调度器机制为纯资料分析）；3.5 判据映射
  （persistent 仅 C1/C2，in-kernel 调度才补齐 C3/C4/C5）；3.6 小结。P4-3 完成，P4-4 待续；本轮未产生新实验。
- **本轮（P4-2）**：`docs/03-related-tech.md` §2 补入 PDL 精读——机理（三原语：下游
  `cudaLaunchAttributeProgrammaticStreamSerialization`、上游 `cudaTriggerProgrammaticLaunchCompletion`、
  下游 `cudaGridDependencySynchronize`；触发在全体 block 触发或上游退出时生效）、真实资料
  （CUDA C Programming Guide PDL 一节 = MPK 引用 [26] 快照行 2242–2244）、本机实测（新 `src/pdl.cu`、
  `harness/pdl.out.txt`：公平对照下 **重叠量 = min(tail, prologue)**，prologue=0→1.01×、
  prologue=100k→1.26×、prologue=200k→1.51×、prologue=400k→1.33× 被 tail 封顶；checksum 501 轮全部
  正确读到上游数据）、未消除项（粒度=kernel 对、需逐 kernel 改造、launch 次数/中间张量不变）、
  判据映射（全不满足 C1–C5，与多流同档降级项）。P4-2 标记完成；纯 §2 完成，P4-3/4 待续。
- **本轮（P4-1）**：新建 `docs/03-related-tech.md` §1「CUDA Graphs」——机理（capture→instantiate→replay，
  三种 capture 模式、显式构图 API、节点/子图/device graph launch）、真实资料（NVIDIA 2019 博客 9.6→3.8→3.4 µs）、
  本机实测（新 `src/cuda_graph.cu`、`harness/cuda_graph.out.txt`：eager per-kernel 近恒定 ~1.7–2.3 µs，
  graph 摊到 ~0.78 µs 不归零，N=1024 2.98×，单次图提交固定 ~3.1 µs，capture+instantiate ~104 µs）、
  未消除项（kernel barrier/依赖串行沿用 P1-2 D 对照 1.05×、图静态、device graph launch 仍 kernel 级）、
  与 MPK tGraph 对照（§3 行 402–414）及 C1–C5 判据映射（仅部分满足 C1）。P4-1 标记完成；
  纯 §1 完成，P4-2/3/4 待续。
- **本轮（P3-3）**：`docs/02-production-cases.md` §2/§3 扩为 P3-3 精读；P3-3 标记完成、P3 收官；
  纯资料分析，未产生新实验。

## 阻塞

- （空）
