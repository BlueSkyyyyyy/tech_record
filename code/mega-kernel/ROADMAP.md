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

### P23 实验扩展 13：跨层软件流水 + 环形缓冲预取（`docs/04-experiments.md` §16）

- [x] **P23-1** 把 MPK §5.3（快照行 1312–1368）的 **paged shared memory + cross-task software
      pipelining** 实测出来（补齐 P22-1 §15.6 明确留的尾巴「真正的跨层流水需要环形缓冲/预取」）。
      见 `docs/04-experiments.md` §16，源码 `src/cross_layer_pipeline.cu`，原始输出
      `harness/cross_layer_pipeline.out.txt`（E=16 主运行）与 `harness/cross_layer_pipeline_e32.out.txt`
      （E=32 重载敏感性）（目标卡 H100/sm90，device 3，iters=30，L=64，G=132=SM，T=256，
      tile=16KB/32KB，权重 528MB/1056MB ≫ L2）。依赖模型：层 `l`、worker `j` 的 task = 私有权重
      tile（cp.async 载入、**与 DAG 无关可预取**）+ 跨层激活值（读 worker `(j+1)%G` 在层 `l-1`
      写的 `act[l][(j+1)%G]`，**必须等**）+ `E` 链 ×`R` 轮 FMA 归约。四策略公平对照：`eager`
      （L 次 launch，kernel-per-operator）/`barrier`（单 kernel + 每层 grid.sync）/`decen`（单 kernel
      + 逐槽事件计数器，无预取）/`ring`（decen + K 页环形 SMEM 跨 task 预取）。三 Section：A 扫 R、
      B 扫环形深度 K（R=16）、C 扫层数 L。核心结论：(1) 四策略对 `act[L][*]` **逐位一致**
      （互比 `max rel|Δ|=0.00e+00`）；(2) **单内核相对 kernel-per-operator 的主收益仍是消 launch**
      ——`eager/barrier` 随 L 8→256 从 1.14× 升到 **1.25×**（对上 P5-1/P9-1 损失二）；(3) **细粒度
      事件调度 `decen` 稳定优于每层 `grid.sync()` 的 `barrier` 约 3–14%**（`d/b` 0.856–0.972×，
      访存越重优势越大；解释 P22-1「barrier 恒快」为何只在每层 `W≥256` 的图成立——本节 `W=G`）；
      (4) **跨层预取 `ring` 相对 `barrier` 峰值快 1.19–1.20×**（E=16 R=8 `r/b`=0.844×、E=32 R=8
      `r/b`=0.834×），收益 ≈ 访存时间占比、随计算强度 R 衰减（R=256 仅 1.03×），落在 MPK §6.6
      （行 1593）的 **1.2–1.3×** 区间；(5) **纯访存（R=0）时流水净亏**（`r/b` 1.036–1.376×，
      无计算可藏）；(6) **环形深度 K 的收益取决于 task 访存占比**——轻载 E=16 时 K=1→8 的 `r/d`
      恒 0.933–0.941×（前驱事件等待已够藏 16KB），重载 E=32 时 K=1→4 单调改善
      （0.970→**0.919×**）、K≈4 饱和。判据：`ring` 满足 C1–C4，按静态序列取 task、无固定角色
      worker/scheduler，仍差 C5 一步（C5 由 P15-1 §9 覆盖）。**本轮新增实验**，补齐 P22-1 §15.6、
      并与 P13-1（单 task 内流水）/P20-1（跨算子静态分块重叠）合成「软件流水三层」画像。

### P24 实验扩展 14：损失三的直接字节证据（`docs/04-experiments.md` §17）

- [x] **P24-1** 把「损失三=中间张量 HBM 往返」从 P5-2 的**算术推断**升级为 **ncu 直接测量**
      （补齐 P5-2 §2.6 / P9-1「只测时间不测字节」的缺口）。见 `docs/04-experiments.md` §17，
      源码 `src/dram_traffic.cu`，原始输出 `harness/dram_traffic.out.txt`（计时）与
      `harness/dram_traffic_ncu.out.txt`（ncu 字节 + 原始 CSV）（目标卡 H100/sm90，device 3，iters=30，
      N=64Mi elem=256 MB/数组 ≫ 50 MB L2，Q=1/2/4/8 逐元素算子链）。四策略
      `eagerQ`/`graphQ`/`barrier`/`fused` 公平对照，`ncu -k regex:dt_` 取
      `dram__bytes_read.sum`+`dram__bytes_write.sum`。核心结论：(1) **fused 的 DRAM 流量是常量**
      `≈ 2×N×4B = 8 B/元素`（实测 518.5 MB，读 268.4+写 250.0），与 Q 无关；(2) **eager/graph 流量
      `≈ Q × 517 MB`**，`vs_fused` = 1.99/3.98/7.96 = **精确等于 Q**——每融合一个算子消掉一趟完整
      全张量 HBM 往返；(3) **barrier 与 eager 字节几乎相同**（仅写多 2–4%），证 `grid.sync()` 只省
      launch 不省流量（损失二与损失三正交）；(4) **时间与字节严格成正比**——两档表观带宽同 ~2.75 TB/s，
      `f/b` 时间比（0.50/0.25/0.12）与字节比（1/1.99/3.98/7.96 的倒数）吻合，fused 相对 eager 的加速
      **完全 = 省下的 DRAM 字节比**（对上 P5-2 推断的 ~2.8 TB/s，把推断升级为实测）。判据不变
      （`fused` C1–C3、`barrier` +C4，无 C5）。**本轮新增实验**；与 P9-1（时间标度）互为表里。

### P25 线上实现构建与端到端冒烟（`docs/09-mpk-build.md`）

- [x] **P25-1** 在 `kernel_lab`（目标卡 H100/sm90，CUDA 13.2）容器里**从零构建 mirage**
      （快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`），并用 `PersistentKernel` API
      生成/编译/launch 一个**最小 megakernel**（单层 RMSNorm，随机张量，无 HF 权重），补齐
      `docs/06` §0「构建成本过高、未做」的尾巴。见 `docs/09-mpk-build.md`、脚本
      `harness/mpk_smoke.py`、原始输出 `harness/mpk_build_run.out.txt`（构建）与
      `harness/mpk_smoke.out.txt`（运行）：
      - **主机侧全链路成功**：`pip install -e .`（`mirage_runtime` + cython 扩展）成功、
        `import mirage`→0.2.4；生成 18 task / 3 event 的 megakernel；nvcc JIT 编译 sm_90a
        成功（~27 s）；`worker kernel & scheduler kernel` 双 stream launch 成功。
      - **设备侧未收敛**：`torch.cuda.synchronize()` 永不返回、GPU 100% 自旋；4 worker 也复现
        （排除 SM 占用不足）。**本轮无法给出端到端延迟/正确性数字**，记为**部分完成 + 一条阻塞**。
      - 关键技术坑：运行期依赖 `tg4perfetto@git` 直连 github 超时（改用 PyPI 同名包 +
        `--no-deps --no-build-isolation` 绕过）；mirage 钉 `z3-solver==4.16.0.0` 与容器内
        `tilelang`（要求 `<4.15.5`）互斥——**跑完已把 z3 恢复 4.15.4.0**（故当前 `import mirage`
        失败，下次需重钉 4.16，用完再切回）。
      - 假设（**待验证**）：H1 iteration/事件计数错位、H2 sm90 `MPK_ENABLE_TMA`/`GRACE_HOPPER`
        task 单体内同步卡住、H3 快照 offline/test 路径不完整。下轮见 ROADMAP「阻塞」与 §「下一步」。
      - **续做（P25-1 续）见 `docs/09` §7**：官方 `test_rmsnorm_testmode.py`（128/16）plain **3/3
        挂起**；加 worker 轮询自旋计数插桩后 **3/3 收敛且正确**；消融把根因定位到 **worker 取 task
        轮询循环（`persistent_kernel.cuh:1013`）的时序竞态**（H2 与 H1-阈值形态均排除）。原始输出
        `harness/mpk_converge_{plain,instrumented,attribution}.out.txt`、`harness/mpk_dbg_trace.out.txt`，
        patch `harness/mpk_watchdog.patch`（全量）/`harness/mpk_workaround_only.patch`（最小修复）。
      - **续做 2（P25-1 续 2，端到端延迟）见 `docs/09` §8**：在收敛构建（patch
        `harness/mpk_latency_run.patch`：worker 轮询自旋计数 + `-DMPK_DBG_WATCHDOG`）上测单层
        RMSNorm megakernel 的**每次调用端到端延迟 36.4–39.1 µs**（扫 128/16、64/16、16/16，
        逐 iter CUDA event 计时），对照 PyTorch 朴素 kernel-per-operator **54.0 µs**（约 1.4–1.5×）；
        128/16 连跑 **3/3 收敛**（37.6–38.1 µs，抖动 <1.5%），plain 构建复现挂起（`EXIT=124`）。
        延迟**几乎与 worker 数无关**（39.05→36.40 µs），由「双常驻 kernel 启动/终止协议」主导而非计算。
        脚本 `harness/mpk_latency.py`，原始输出 `harness/mpk_latency.out.txt` /
        `harness/mpk_latency_repeat.out.txt`。**端到端延迟这一条已补齐**。
      - **续做 3（P25-1 续 5，根因钉死）见 `docs/09` §11**：用 **cuda-gdb attach**（非侵入式，避免
        §9.3/§9.4 的「插桩会扰动竞态」）拿到**设备侧证据**——挂起时设备上**只有 `worker_kernel` 常驻**
        （128 block 全在 `persistent_kernel.cuh:1016` 的取 task `ld_acquire` 上自旋），`scheduler_kernel`
        **不在设备上**（`cuda kernel 1` → `Invalid coordinates`）；**4:4 下 worker 只占 4 个 SM、128 个
        SM 空闲，scheduler 仍不在** → 与 occupancy 无关，死锁实为「**scheduler 从未被派发**」。再用一个
        **只交换 launch 顺序**的消融 `harness/mpk_sched_first.patch`（`-DMPK_DBG_SCHED_FIRST`，不动任何
        kernel 代码）钉死根因：把 scheduler 放到 worker **之前** launch，**plain 构建 4/4 收敛且正确**
        （prewarm=none ×3 + cast_1elem ×1，`max|out-ref|≤0.03125`），默认顺序同会话复现 HANG。clean
        （scheduler-first）构建端到端延迟 **39.1 / 43.5 µs**（与 §8 同量级，证明 §8 不是内插桩产物）。
        脚本 `harness/mpk_cudagdb_probe.py`/`mpk_cudagdb.sh`/`mpk_cudagdb.cmds`/`mpk_sched_first.sh`，
        原始输出 `harness/mpk_cudagdb.out.txt`/`_w4s4`、`harness/mpk_sched_first.out.txt`、
        `harness/mpk_latency_schedfirst.out.txt`。**P25-1 记 `[x]`**：构建+JIT+launch+收敛+正确性+延迟
        全打通，不收敛根因由 §11.4 的顺序消融钉死（launch 顺序，非内存序/dtype/host 排序/频率/occupancy）。

### P26 实验扩展 15：多算子 megakernel 的端到端延迟（`docs/09-mpk-build.md` §12）

- [x] **P26-1** 把 `docs/09` 的端到端延迟从「单层 RMSNorm」推进到「**一条 4 算子 DAG 编进同一个
      常驻 kernel**」，并扫 batch，兑现阶段「下一步」可选方向③（clean scheduler-first 构建上重跑
      多算子/更大 batch 延迟）。见 `docs/09-mpk-build.md` §12，源码/脚本
      `harness/mpk_latency_multi.py`、运行器 `harness/mpk_latency_multi.sh`，原始输出
      `harness/mpk_latency_multi.out.txt`（batch 扫描）与 `harness/mpk_latency_multi_config.out.txt`
      （worker 数扫描）（目标卡 H100/sm90，device 7，iters=200，warmup=20，`f9eb70c`，clean 绕过
      `harness/mpk_sched_first.patch`，**不动 kernel 代码**）。
      被测图 = Qwen3 dense MLP block：`rmsnorm → linear(gate+up) → silu_mul → linear_with_residual`
      （hidden=4096、inter=2048、bf16），API 同快照 `test_qwen3_mlp_testmode.py`。核心结论：
      - **多算子延迟与单算子同量级**：4 算子块 128/16、batch=16 = **46.4 µs**，比 §8 单算子 39.05 µs
        只慢 ~7.4 µs（1.19×）——3 个算子（~+160 task）全在**同一个常驻 kernel** 内（C1 仍单次提交、
        设备内 event 依赖、无 host launch/kernel barrier）。
      - **对 kernel-per-operator 的加速随算子数放大**：torch 朴素同块 175–205 µs（vs 单算子 54 µs），
        故比值从单算子 **1.38×**（§8）升到 4 算子 **3.5–4.8×**；「加 3 算子的边际代价」MPK ~7.4 µs
        vs torch ~148 µs（约 5%），兑现 §8.3「算子数/中间量越大 MPK 优势越大」，对上 `docs/04`
        §9/§17 损失二/三。
      - **延迟对 batch 与 worker 数均近乎不敏感**：batch 1→16 在 41.5–49.6 µs 间无单调趋势、worker
        128→16 只差 ~10%——主导项仍是「双常驻 kernel 启动 + scheduler 冷启动 + done-event 终止协议」
        （§8.3），非本用例计算量。
      - **快照硬约束（batch≤16）**：batch=32 的 MLP 图**编译期**就失败——`linear_swapAB_hopper.cuh:71`
        的 `static_assert(BATCH_SIZE <= 16, "…in swapAB")`（注释 `:69-70`），脚本 try/except 记
        `BUILD_FAIL` 后继续（整轮 `exit=0`）。
      - 所有成功 case `max|out−ref|=1.56e-02`（bf16 容差内）；`Total tasks` 163/166/170/178 随 batch
        递增、events 恒 37。判据：C1 在多算子图成立（判据十二/十三/十四见 §12.5）。**本轮新增实验。**

### P27 手工 mega-kernel 先例精读（`docs/10-megakernel-precedents.md`）

- [x] **P27-1** 把 MPK 反复转引、但本专题此前只作转引的两个**手工 mega-kernel 先例**回到一手资料：
      （a）**FlashMoE**（`arXiv:2506.04667`，NeurIPS '25；代码 `github.com/osayamenja/FlashMoE`）——
      首个把**分布式 MoE 一整个算子段**融进单常驻 kernel（§1 **Table 1**：1 launch vs 最多 550），
      §3 actor 模型（Processor/Subscriber/Scheduler，OS 块 4 warp）+ §3.1 tile 级统一 task（每 task 一个
      `__device__` 函数）+ §3.2 对称张量布局的非阻塞 NVSHMEM（**Theorem 3.1** write-write conflict-free、
      temporal buffering `2r=4×`、§3.2.1 in-place padding）；§4 结果 SM 利用率 **93.17%**（vs DeepEP 14%）、
      8 卡延迟最高 **6.4×**、吞吐 **17.7 MTokens/s（5.7×）**、overlap **4×**、128 experts 领先 **6.6×**
      （全程 FP32 对基线 FP16）；§5 自陈局限（工程复杂度、期待 compiler/DSL）。
      （b）**Spector 等 Llama-1B megakernel**（Hazy Research 博客 2025-05-27 *Look Ma, No Bubbles!*；
      代码 `github.com/HazyResearch/Megakernels`）——把**整个 Llama-1B 前向 ~100 算子**融进单 kernel，
      三条 stall 对上三类损失（512 block/148 SM → 80 SM 空转；H100 stream launch **2.1 µs** / graph
      **1.3 µs**；PDL `cudaGridDependencySynchronize` 太粗）；三套机制（设备内解释器 + 每 SM 指令序列 /
      **SMEM 分页 13×16 KiB** / global 计数器细粒度依赖，MLP 分 4 chunk），H100 **<1 ms**、B200 **<680 µs**、
      比 vLLM ~2.5×、78% 带宽，B200 600 µs 分解（激活 250 / 计算 200 / 等权重 30 / 同步 40 / 杂项 80）。
      与 DeepGEMM / MPK 逐维对照表（四者均满足 C1–C5；分野=融合范围段级/整图 × 调度手工/自动；段级 T4
      内部另有静态↔动态光谱）。见 `docs/10-megakernel-precedents.md`（新文档，§6 关键坐标含真实链接）。
      配套博客 post `mega-kernel-11-precedents`（weight 12）并更新 `mega-kernel-00-index` 目录第 11 项。
      **纯资料分析（两个先例分别需 8×H100+NVSHMEM 多进程 / 完整 Llama 权重与编译链，本机不具备），
      本轮无新实验**；相关机制实测交叉引用 `docs/04` §1/§9/§14/§16/§17。

### P28 设备侧数值证据：-G 调试构建 + cuda-gdb 读出事件计数（`docs/09-mpk-build.md` §13）

- [x] **P28-1** 兑现「下一步」可选方向①：把 §11.7「无设备侧类型信息、读不出『卡在哪个 event 计数』」
      的尾巴补上——**用 `-G`（`--device-debug`）重建 JIT 内核**，让 cuda-gdb 能解析 `RuntimeConfig`
      字段类型，再 attach 挂起进程把**数值**读出来。见 `docs/09-mpk-build.md` §13；脚本
      `harness/mpk_debug_counters.sh`、命令 `harness/mpk_debug_counters.cmds`、补丁
      `harness/mpk_debug_build.patch`（只给 `get_compile_command()` 拼 `MPK_EXTRA_DEFINES`，不碰 kernel
      代码）、探针 `harness/mpk_cudagdb_probe.py`（新增 `MPK_KEEP_TEMPDIR=1`，阻止 JIT 临时目录在
      编译后被回收，cuda-gdb 才能加载模块符号），原始输出 `harness/mpk_debug_counters.out.txt`
      （目标卡 H100/sm90，device 7，快照 `f9eb70c`）。核心结论：
      - **[更强证据] `-G` 构建下宿主阻塞在第二次 `cudaLaunchKernel`**（栈 `cuLaunchKernel→cudaLaunchKernel
        →launcher .so #14`，即 scheduler 那次 launch 不返回），设备上**只有 `worker_kernel`**、
        `cuda kernel 1`→`Invalid coordinates`（与 §11.2/§11.3 一致且更强）。
      - **[数值] worker 零 task**：`worker_queue_last_ready_task_id[0]=[1]=0`；worker0 局部
        `worker_id=0, queue_idx=0, queue_pos=queue_len=0`（空队列取 task 分支）。
      - **[数值] 卡在第一个 event 派发前**：`config.num_events=3`；`sched_queue_last_ready_event_id=
        next_free_event_id=1`（队列有 1 个未消费 event）；`all_event_counters[0/1/2]=0/0/0`，
        `all_event_num_triggers[0/1/2]=1/1/16`——连阈值 16 的 `END_OF_TASK_GRAPH` 都停在 0。
      - **判断**：不是「某 event 阈值差几个凑不满」，而是 **scheduler 队列首个 event 从未被消费 →
        无 task 派发 → 全部事件计数恒 0**，把 §11.4 的 launch-顺序根因推进到「第一个 event 派发前夜」。
      - **口径 caveat**：`-G` 关闭优化会改变 codegen/宿主行为（默认构建宿主两次 launch 均返回、
        scheduler 仅 pending），故数值刻画 **`-G` 挂起态**，其设备侧结构与默认挂起态相同（§11 交叉验证）；
        `next/last_task_pos` 两个 `__shared__` 游标仍 `<optimized out>`。**不构成对上游的 bug 判定**。
        **本轮新增实验**。

### P29 实验扩展 16：Thread Block Clusters + DSM（`docs/04-experiments.md` §18）

- [x] **P29-1** 补齐 `docs/04` §6.6 明确留的「sm90 的 cluster 未覆盖」尾巴，并把「跨 SM 中间量」
      这一档从 P20-1（需 SM 共驻）与 P24-1（落/不落 HBM）**推进到硬件原语**：Hopper **Thread Block
      Cluster + 分布式共享内存（DSM）**——cluster 内 block 用 `map_shared_rank` 直读邻居 SMEM，
      中间量不落 L2/HBM；生产案例 DeepGEMM Mega Gate 已用 `cluster_dim = num_mma_ctas`（`docs/02`
      §1.2）。见 `docs/04-experiments.md` §18，源码 `src/cluster_dsm.cu`，脚本
      `harness/cluster_dsm.sh`，原始输出 `harness/cluster_dsm.out.txt`（目标卡 H100/sm90，
      SMs=132，**固定空闲 device 1**，nvcc 13.2，**每配置独立进程**——一次 cluster launch failure
      会污染 CUDA context/sticky）。主线三个 Section：
      - **A（DSM 远端读 vs 本地 SMEM）**：`C=1` 退化为本地基线，16 KB/block、grid=128、R=20000。
        **远端读与本地 SMEM 同量级**：`C=2` 0.99×、`C≥4` 只 ~1.17×（56.3→66.3 µs）；对照
        plain 全局流式读（256 MB）仅 **690.8 GB/s**。
      - **S（ring 段位移，双缓冲 ping-pong、1 `cluster.sync`/轮，段放 global scratch vs 自己 SMEM）**：
        `maxerr=0` 与解析参考逐位一致、跨进程可复现。**DSM 恒快于 HBM/L2 往返且随段增大放大**：
        seg 1/2/4/8 KB → `dsm/hbm` = **1.06 / 1.11 / 1.17 / 1.27×**（8 KB 时 dsm 1330 vs hbm 1049
        GB/s；scratch 仅 2 MB 驻 L2，故这是 **L2 往返**对照；DSM 已超 L2）。
      - **C（协调/容量）**：`cluster.sync()` = **504→767 ns**（C=2→16），比 grid-wide `grid.sync()`
        （1113 ns）便宜 **1.5–2.2×**（只同步 cluster 内几个 SM）；**cluster 硬上限 16 block**
        （C=32 `cluster misconfiguration`），且 **grid 必须是 C 的整数倍**（131 非 8 倍数即失败）。
      核心结论：**DSM 是「SMEM 级」的跨 SM 通道**（局部、片上、局部屏障）；这给 P20-1 的「重叠需
      SM 共驻」以硬件形态，给 P24-1 的损失三补「L2↔DSM」一级；但 **cluster 装不下整机（≤16 block，
      限同一 GPC）→ 132 SM 整机 megakernel 只能切成 ~8 个 cluster，跨 cluster 仍走 L2/HBM**，与 MPK
      去中心化 event queue / DeepGEMM ring barrier 的「按需局部同步」同目标。是 **C4 的更细实现，
      非 C5**（无固定 worker/scheduler 动态派发）。踩坑：**变界 stride 循环读远端 SMEM 会稳定触发
      `unspecified launch failure`（-O3 代码生成问题）**，改用固定索引单元素读规避。判据不变
      （C1–C3；cluster.sync/DSM 细化了 C4）。**本轮新增实验。**

### P30 实验扩展 17：cluster 内 TMA multicast（`docs/04-experiments.md` §19）

- [x] **P30-1** 补齐 `docs/04` §18.6 明确留的「cluster 内的 `mbarrier`/TMA 组合未覆盖」尾巴，
  把 P29-1 的「已到片上后跨 SM 交换（DSM）」补上另一半「全局→C×SMEM 的**载入**」：
  Hopper **TMA multicast**（`cp.async.bulk.tensor ... .multicast::cluster`）一次读、投递到 cluster
  内 C 个 CTA 的 SMEM。见 `docs/04-experiments.md` §19，源码 `src/tma_multicast.cu`，脚本
  `harness/tma_multicast.sh`，原始输出 `harness/tma_multicast.out.txt`（目标卡 H100/sm90a、
  **固定空闲 device 3**、nvcc 13.2、每配置独立进程避免 sticky、`cuTensorMapEncodeTiled` 建 2D tmap、
  `cudaLaunchKernelEx` + `cudaLaunchAttributeClusterDimension`）。kernel 把 SMEM 切 `L` 槽（≈192 KB）、
  每轮对 `L` 槽各发一次 TMA、单 mbarrier 累计 `L·chunk` 一次等待（深度 L 全重叠）；两模式
  `mode0`（每 CTA 各发）/`mode1`（仅 rank0 发 multicast，mask=全）。核心结论：
  - **语义实测确认**：mode1 只 rank0 发、其余 CTA 什么都不发，却都在**自己的** mbarrier（同一
    SMEM 偏移）等到、全部 case **`maxerr=0`**——证明 multicast 把 transaction bytes 记到 mask 内
    每个目的 CTA 的 mbarrier（对上 DeepGEMM `common/tma_copy.cuh:77–86` leader-only + mask 用法）；
  - **delivery 吞吐两模式打平**：单 cluster 扫 C=2/4/8/16 deliver 215→1563 GB/s（单 CTA ~97–105 GB/s、
    随 C 线性），多 cluster grid=128 饱和在 ~6.2–6.8 TB/s（SMEM 写入/TMA issue 成瓶颈），
    multicast 仅一致地**快 0.1–1.1%**（chunk 4–32 KB 稳定 ~0.8–1.0%）；
  - **收益在源侧**：ncu（C=8/grid=128/8 KB）显示 L2 read sectors 少 **2.10×**、`lts__t_bytes` 少
    **1.84×**，而 **DRAM 字节完全相同**（71.28 MB）——因 working set `R·L·chunk≈37.5 MB` 装进 50 MB L2
    且各 CTA 同步读同一序列（L2 去重）；**指令级源读少 C×是上界模型，真值以 ncu 的 2× 为准**；
  - **含义**：multicast 省的是 **L2 请求/带宽**（~2×），在 megakernel 里把这块省下的 L2 带宽让给
    其它并发 task 的载入——正是 DeepGEMM Mega MoE「**Always multicast on A**」（`heuristics/mega_moe.hpp:141–143`、
    `impls/sm100_bf16_mega_moe.cuh:130`、`cluster_dim=(2,1,1)` `impls/sm100_bf16_mega_moe.hpp:138`，
    一手核对 `057ca59`）选「被多 CTA 共享、最值省源带宽」那一路的理由。
  判据：cluster kernel 满足 C1–C3、multicast/`cluster.sync` 细化 C4，仍非 C5。踩坑：`.multicast::cluster`
  在 `-arch=sm_90` 下 ptxas advisory（须 `sm_90a`）；DSM 手工扇出的 mode2 因远端 SMEM stride 循环
  代码生成问题（§18.6 同类）成 strawman 已移除。**本轮新增实验**；与 P29-1（§18）合成「跨 SM 搬运两跳」。

### P31 实验扩展 18：2D/3D Thread Block Cluster（`docs/04-experiments.md` §20）

- [x] **P31-1** 补齐 `docs/04` §18.6 / §19.6 两条都留的同一个尾巴「**未测 2D/3D cluster**」，
  把 P29-1（§18 DSM）与 P30-1（§19 TMA multicast）从 1D cluster 推广到**多维**，并补上 §19 未测的
  **multicast 子集掩码**。生产案例的 cluster 本就是多维的：DeepGEMM Mega MoE `cluster_dim=(2,1,1)`
  （`impls/sm100_bf16_mega_moe.hpp:138`）、Mega Gate `cluster_dim=num_mma_ctas`（`docs/02` §1.2）。
  见 `docs/04-experiments.md` §20，源码 `src/cluster_2d.cu`，脚本 `harness/cluster_2d.sh`，原始输出
  `harness/cluster_2d.out.txt`（目标卡 H100/sm90a，**固定空闲 device 3**，nvcc 13.2，每配置独立进程）。
  五 Section：(1) **rank 布局**——实测确认 `block_rank() == x + y*cx + z*cx*cy`（行主序线性下标；
  `(2,2,1)`/`(4,2,1)`/`(4,4,1)`/`(2,2,2)` 全 mismatch=0）；(2) **2D DSM 环形段位移**与 1D 同档
  （C=2/4/8/16 → 587.9/552.4/533.0/527.8 GB/s，对上 §18 1D seg2KB 的 560.7 GB/s；`maxerr=0`）；
  (3) **TMA multicast 整群（mode1）vs 逐行子集（mode2）**——两模式 `maxerr=0`，证明 16-bit mask
  按线性 rank **精确生效**（逐行只发给该行 `cx` 个 rank、各 CTA 只收本行 leader 的数据）；
  (4) ncu 源侧：mode1 比 mode0 L2 read sectors 少 **4.38×**、DRAM 不变；mode2 的 L2 sectors 恰为
  mode1 的 `cy=4` 倍、DRAM 为 mode0 的 4 倍（**因逐行读不同源区、无 L2 去重**——子集广播的价值在
  **投递范围正确**而非省流量）；(5) `cluster.sync` 与 cluster 形状无关、只与 block 数有关
  （448→505 ns @ C 2→16）；**容量边界不变**：3D cluster 可用（`(2,2,2)=8` ok），硬上限仍 16 block，
  且 grid **每个维度**都须是 cluster 对应维的整数倍（1D 的「grid 是 C 的倍数」在多维下变逐维约束）。
  踩坑：2D launch 逐维对齐、DSM 环形的解析参考在 2D 下 cluster block 全局线性 id **不连续**、
  首版沿用 1D 公式给 C=16 的假 `maxerr=40000`。判据不变（C1–C3，`cluster.sync`/multicast 细化 C4，
  仍非 C5）。**本轮新增实验**；与 P29-1（§18）/P30-1（§19）合成「跨 SM 搬运」的多维形态。

### P32 实验扩展 19：事件可见性的前移原语（`docs/04-experiments.md` §21）

- [x] **P32-1** 补齐 `docs/04` §20.6 明留的「`cudaLaunchAttributeProgrammaticEvent` 未覆盖」尾巴，
  把 kernel 级「进度/事件」原语能前移到哪一点量出来（直接回答 megakernel「中间进度能否被别的执行体
  更早看到」）。见 `docs/04-experiments.md` §21，源码 `src/programmatic_event.cu`、脚本
  `harness/programmatic_event.sh`、原始输出 `harness/programmatic_event.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、`iters=30`、`pro=tail=100000` cyc、`step=20000`）。三 Section：
  - **A（单 block，事件触发点 vs A 时间轴）**：用一对计时事件夹住 cross-stream 消费 kernel 的起点。
    `post`/`early`/`none`/`bstart` **全 ≈ A 整段（108 vs 116 µs）= 完成点放行**，只有 `lce`
    （`LaunchCompletionEvent`）落到 **7.77 µs**（全 block 启动即放行）。即 **ProgrammaticEvent 的
    提前 trigger（含 `triggerAtBlockStart`）不放行 stream/host 依赖**，与 `cuda.h:2227` 的
    「依赖方（含 host）不保证在 release 时刻观察到」一致；`driver_types.h:4082` 又补 LCE「名义上类似
    `triggerAtBlockStart` 的 programmatic event，只是对 `cudaGridDependencySynchronize()` 不可见」。
  - **B（多 block 错峰扫 G）**：`early`/`bstart`/`post` **完全重合**、随 `(G-1)*step` 线性（G=128 →
    1391 µs），`lce` 恒定 ~7.5 µs——cross-stream 下唯一真正「全 block 开始即放行」的是 LCE。
  - **C（同 stream PDL，trigger 到底放行了什么）**：无 trigger 208.5 µs / 有 trigger 157.7 µs ≈
    `pro+max(tail,prol)+crit`——**trigger 只允许下游 prologue 与上游 tail 重叠**，B 的
    `cudaGridDependencySynchronize()` 仍等 A **完成**；**挂不挂 ProgrammaticEvent 属性结果相同**
    （157.72 vs 157.74），设备端口由 trigger 本身给，事件属性只是把端口记录到 cudaEvent 供图
    `PROGRAMMATIC` 边（`cuda.h:2047/2060`）用。
  含义：kernel 级**没有**「中间进度早于完成、可被别的执行体看到」的通用信号——「下游尽早启动」要靠
  PDL port、同步点仍是完成；「下游知道上游已开始」靠 LCE、但它不能当数据依赖；「第 i 个 tile 放行第 i
  个 task」只能把执行体放进**同一常驻 kernel** 用设备内 event counter/原子 ring barrier（MPK
  `docs/06` §3、本专题 P15-1 §9；DeepGEMM 的 warmup wave/`ScoreBarrier` 即此，`docs/02` §1.5/§1.2）。
  判据：六档全为 kernel-per-operator 降级项，**均不满足 C1–C5**。同步在 `docs/03` §2.6 交叉引用。
  **本轮新增实验。**

### P33 SM100 2SM 协作原语：`cta_group::2` TMA load + tcgen05 2-SM MMA 的 sm90 投影（`docs/04-experiments.md` §22）

- [x] **P33-1** 补齐 §20.6 / 「下一步」都点名的两条 **SM100 专属原语**：`SM100_TMA_2SM_LOAD`
      （`cp.async.bulk.tensor ... .cta_group::2`，一个 CTA 发、tile 按 2 个 SM 分区投递）与
      **tcgen05 2-SM MMA**（`tcgen05.mma.cta_group::2`，一对 SM 协作算一次、TMEM 各持 M 的一半）。
      这两条只在 SM100 可执行，故分两步：**(a) 资料分析** CUTLASS（DeepGEMM 内
      `third-party/cutlass`，commit `057ca59`）一手语义；**(b) sm90 投影实测**。见
      `docs/04-experiments.md` §22，源码 `src/two_sm_load.cu`，脚本 `harness/two_sm_load.sh`，
      原始输出 `harness/two_sm_load.out.txt`（目标卡 H100/sm90a、**固定空闲 device 2**、nvcc 13.2、
      cluster 恒 C=2、每配置独立进程）。核心结论：
      - **资料分析**：`copy_sm100_tma.hpp` 的 `SM100_TMA_2SM_LOAD_{1D..5D}`（`:52/78/104/130/156`）、
        dispatcher `:182`、`_MULTICAST_*`（`:229` 起）、`_IM2COL_*`（`:402` 起）；PTX 带 `.cta_group::2`
        （`:92`）；障碍句柄 `& Sm100MmaPeerBitMask(0xFEFFFFFF)`（`:45/63`）注释 *"Executed by both CTAs.
        Set peer bit to 0 so that the transaction bytes will update CTA0's barrier."*；集体层
        `sm100_mma_warpspecialized.hpp:234` *"Only one thread issues the TMA and updates the barriers in a
        2SM MMA"*；MMA 侧 `SM100_MMA_{TF32,F16BF16}_2x1SM_SS`（`mma_sm100_umma.hpp:382/422`，
        `static_assert(M==128||256,"...for 2 CTA cluster MMA.")` `:384`）由 `elect_one_sync()`
        单线程发 `tcgen05.mma.cta_group::2`（`:406/446`），`umma_arrive_2x1SM`（`barrier.h:779-789`）
        单点到达一对 SM 的屏障。DeepGEMM Mega MoE/Gate `cluster_dim` 的最小单位就是 2SM
        （`sm100_bf16_mega_moe.hpp:138`/`sm100_bf16_mega_gate.hpp:74`，交叉引用 `docs/02` §1.2/§1.5）。
      - **sm90 投影（一个 2-CTA cluster 把 tile A+B 装进两个 SM）** 三模式：mode0 `indep`（两 CTA
        各发各的）、mode1 `mcast`（rank0 TMA multicast 复制投递）、mode2 `split`（rank0 两个
        `cp.async.bulk.shared::cluster` 把 A 发自己、B 发对端，`mapa` 求对端 SMEM/mbarrier；=
        `cta_group::2` 投影）。全部 case `maxerr=0.00000`。
      - **结果**：多 pair（grid=128）chunk 4→32KB，**mcast 越大越占优**（`m1/m0` 0.964→**0.825**，
        32KB 快 1.21×），**split 越大越吃亏**（`m2/m0` 1.032→**1.130**）；三模式随 R **严格线性**。
      - **ncu 源侧字节（grid=128/chunk=8KB/R=200）**：mode0/mode2 DRAM 读 **209.76/209.73 MB**、
        L2 read sectors **6,553,600**（完全相同），mode1 **104.89 MB / 3,276,800**（**恰减半**）。
        ⇒ **multicast 减半的是源流量；分区投递不省源流量**（A、B 各读一次与发出者数量无关），
        2SM 的价值在硬件层（一次发、一个屏障、一对 SM 分持 TMEM），**不在**减流量；sm90 缺席
        `.cta_group::2`，软件投影（单发连发含远端）反而慢 ~13%。
      - 判据：满足 C1–C3、`cta_group::2`/multicast 细化 C4，**仍非 C5**（与 §18/§19 同档）。
      - 局限：sm90 投影非 SM100 本体（两套独立 mbarrier，非 CTA0 单点）、未测 tcgen05 2SM MMA、
        mode1 与 mode0/2 语义不同（复制 vs 分区）、固定 C=2、测的是载入原语非端到端。
      **本轮新增实验**；与 §18/§19/§20/§21 合成「跨 SM 协作原语」的 sm100 展望。

### P34 生产案例 sm90 等价复刻：Mega Gate 的 `cluster_dim` 路径（`docs/04-experiments.md` §23）

- [x] **P34-1** 兑现「下一步」剩余可选方向「把 DeepGEMM Mega Gate 的 `cluster_dim` 路径做一次 sm90
      等价复刻（Mega Gate 本身 SM100 专属）」。见 `docs/04-experiments.md` §23，源码
      `src/mega_gate_cluster.cu`，脚本 `harness/mega_gate_cluster.sh`，原始输出
      `harness/mega_gate_cluster.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2）。
      复刻 Mega Gate 的**协调路径**（一手核对 `057ca59`）：
      - **CTA pair 切专家维**：`num_mma_ctas=2` → `cluster_dim=2`，一对 CTA 把打分 GEMM 的专家维对半切
        （`sm100_bf16_mega_gate.cuh:55-58,136-142`；`impls/sm100_bf16_mega_gate.hpp:74`）；
      - **split-K + 全局 score scratch** `[token][split][expert]`（`layout/mega_gate.cuh:78-107`）；
      - **ScoreBarrier**（64-bit 状态 `(grid+1)*kNumMaxLogicalCtas + arrived`，无符号距离去陈旧；
        `scheduler/mega_gate.cuh:13-39`）会合后，跨 split 归约（`sm100_bf16_mega_gate.cuh:304-321`）
        + `sqrt_softplus`/top-k（`epilogue/sm100_mega_gate.cuh:19-23,107-152`）。
      对照变体 `fused`（分数留 SMEM、不写 scratch、不设 barrier）。四 Section：
      - **A 正确性**：7 组 `SPLIT×MMA`(+fused) 与参考**逐 token 一致**（`idx_mismatch=0/16384`，
        `wmax≤2.09e-07`）——CTA-pair + split-K + ScoreBarrier **不动数值**。
      - **B 占满机器**：固定 token（NT=16）只增逻辑 CTA，`grid 16→128` 延迟 **3.51×**
        （`SPLIT=2` 1.63× vs `MMA=2` 1.57×）——multi-CTA 逻辑任务的价值是**填 SM**，对上
        Mega Gate heuristic `num_worker_groups=num_sms/num_logical_ctas`（`heuristics/mega_gate.hpp:137`）。
      - **C 已饱和**（NT=128，grid≈#SM）：再 split 只增加波次与协调开销（SPLIT2 1.53×、gate 比 fused
        慢 **~11%** / +24 µs）——饱和区间 multi-CTA 只剩开销。
      - **D ncu**：fused vs gate SPLIT=1 的 **L2 写差 +0.39M sectors（≈+12.6 MB）**、SPLIT4×MMA2 的
        DRAM 写 1.94→**19.84 MB**——split-K 部分分**必经 L2/DRAM 一次往返回合**（`docs/04` §17 损失三的
        gate 形态，只是 Mega Gate 只走**一次** round-trip）。判据：gate 满足 C1–C4、仍非 C5
        （静态 grid 角色）。**本轮新增实验。** 讲清 sm90 投影边界：无 tcgen05/TMEM/`cta_group::2`，
      软件 cluster + 独立 barrier 代替硬件 2SM。

### P35 显式构图 + PROGRAMMATIC 边（`docs/04-experiments.md` §24）

- [x] **P35-1** 补齐 `docs/04` §13.6「未测显式构图 API（`cudaGraphAddKernelNode` +
      `cudaGraphKernelNodeSetAttribute` 的 PDL 属性）」与 §21.6「未测 CUDA Graph 的 `PROGRAMMATIC`
      边」两条尾巴：**用显式构图把一个 programmatic 边做进图里，能否让下游 kernel 在上游完成之前
      被调度**？见 `docs/04-experiments.md` §24，源码 `src/graph_programmatic.cu`，脚本
      `harness/graph_programmatic.sh`，原始输出 `harness/graph_programmatic.out.txt`（主配置）与
      `harness/graph_programmatic_sens.out.txt`（`pro>tail` 敏感性）（目标卡 H100/sm90、**固定空闲
      device 3**、nvcc 13.2、iters=30）。一张图 A→B，`%globaltimer` 打点，四档边
      `block`/`prog_nt`/`prog`/`lce`（`cudaGraphEdgeData.from_port` = Default/Programmatic/
      Programmatic/LaunchCompletion）。核心结论：(A) **关键量是「下游调度点」`sched`**——
      `block`/`prog_nt` 的 `sched≈A_total`（完成即放行），`prog`（A 在 prologue 后 trigger）的
      `sched=51.0µs≈pro`、`lce` 的 `sched=0.32µs≈0`——**图的 programmatic 边把下游调度点前移到
      上游完成之前**（下游 prologue 与上游 tail 重叠），这正是 stream 上 `cudaStreamWaitEvent`
      做不到的；(B) **但 `rel`（下游 `cudaGridDependencySynchronize` 返回）恒 ≈ A_total**——端口只
      决定下游**何时被调度**，sync 仍等**上游完成**，与 §21 完全一致；(C) Section B 证 programmatic
      端口在**所有 block 都 trigger** 时激活（`prog_sched ≈ pro+(G-1)*stagger`）；(D) Section C 证
      **capture 图与显式构图在 PDL 语义上逐项相同**（`sched` 51.01 vs 51.04）；(E) 节点级 PDL 序列化
      属性被驱动**拒绝**（`driver_types.h:3988/4067` 标 "Valid for launches"，不在 graph node 列表），
      端口只由边提供。判据不变（四档均不满足 C1–C5），是 CUDA Graph 这条降级路线的**下界证据**。
      **本轮新增实验。**

### P36 CUDA Graph 条件/循环节点（`docs/04-experiments.md` §25）

- [x] **P36-1** 把 CUDA Graph 的**控制流**维度补齐：CUDA 12.3+ 的 **conditional graph nodes**
      （IF/WHILE/SWITCH，设备侧求值条件）能否让**一张图**表达「数据相关 / 可变行程」控制流？——
      对照 MPK「每 batch size 一张静态图」局限（`docs/01` §5.8）与运行层 in-kernel `iteration`
      循环（`docs/06` §2/§4）。见 `docs/04-experiments.md` §25，源码 `src/graph_conditional.cu`、
      脚本 `harness/graph_conditional.sh`、原始输出 `harness/graph_conditional.out.txt`（目标卡
      H100/sm90、**固定空闲 device 3**、nvcc 13.2）。四策略公平对照 **eager**（host N 次 launch）/
      **unroll**（N 节点静态图，每 N 重新 instantiate）/ **cwhile**（1 个 WHILE 条件节点，同一
      graphExec 服务任意 N）/ **kinternal**（单 kernel 内 for N 次 = 持久化内核微缩版）。核心结论：
      (A) 四条曲线对 N **全部线性**，每行程成本 eager 2.23 / unroll 1.27 / **cwhile 4.60** /
      **kinternal 0.229 µs**——`cwhile` 每行程一次**设备侧 kernel 重分发**，**比 host launch 还贵
      ~2×**，**比 in-kernel 循环贵 ~20×**；(B) 纯控制（work=0）同序：cwhile 4.30 vs eager 2.19 vs
      kinternal **恒定 ~2.14 µs**（N 无关）；(C) instantiate：unroll 随 N 线性（~1 µs/节点，N=64→66 µs）、
      cwhile 常数 ~15.5 µs；IF 节点同一 graphExec 按设备数据走不同分支；(D) 四策略 acc **逐位一致**
      （max|Δ|=0）。**含义**：条件节点解决「表达能力」（一图覆盖变长/数据相关），**不解「性能」**
      （每行程仍付一次 kernel 分发）；MPK 用 in-kernel `iteration` + `prepare_next_batch`（`kinternal`
      档）而非图循环，量级证据支持这一选择。判据：四策略均不满足 C1–C5（kinternal 仅 C1/C2）。
      与 §24（依赖维度→调度级）正交，合起来给出 CUDA Graph 降级路线的天花板。**本轮新增实验。**


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

- 当前进度：**P36-1 完成（1/1）**。补齐 CUDA Graph 的**控制流**维度——CUDA 12.3+ 的
  **conditional graph nodes**（IF/WHILE/SWITCH，设备侧求值条件）能否让**一张图**表达「数据相关 /
  可变行程」控制流，对照 MPK「每 batch size 一张静态图」局限（`docs/01` §5.8）与运行层 in-kernel
  `iteration` 循环（`docs/06` §2/§4）。`docs/04-experiments.md` 新增 **§25**，源码
  `src/graph_conditional.cu`、脚本 `harness/graph_conditional.sh`、原始输出
  `harness/graph_conditional.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、iters=50、
  Nmax=64、G=32、T=256、work=64）。四策略公平对照 **eager**（host N 次 launch）/ **unroll**（N 节点
  静态图，每 N 重新 instantiate）/ **cwhile**（1 个 WHILE 条件节点，同一 graphExec 服务任意 N）/
  **kinternal**（单 kernel 内 for N 次 = 持久化内核微缩版）。核心结论：(A) 四条曲线对 N **全部线性**，
  每行程成本 eager 2.23 / unroll 1.27 / **cwhile 4.60** / **kinternal 0.229 µs**——`cwhile` 每行程一次
  **设备侧 kernel 重分发**，**比 host launch 还贵 ~2×**、**比 in-kernel 循环贵 ~20×**（N=64 时
  cwhile/eager 2.11×、kinternal/eager 0.117×）；(B) 纯控制（work=0）同序：cwhile 4.30 vs eager 2.19 vs
  kinternal **恒定 ~2.14 µs**（N 无关，一次 launch）；(C) instantiate（host 时间）：unroll 随 N 近线性
  （~1 µs/节点，N=64→66.5 µs）、cwhile 常数 ~15.5 µs；**IF 节点**同一 graphExec 按设备侧数据
  （`x[0]` vs 阈值）走「执行/跳过」两分支；(D) 四策略 `acc[]` **逐位一致**（max|Δ|=0.000e+00）。
  **含义**：条件节点解决「**表达能力**」（一个 graphExec 覆盖变长行程 / 数据相关分支，结构上解除
  「每 batch size 一张图」），但**不解「性能」**——每行程仍付一次 kernel 分发；MPK 用 in-kernel
  `iteration` + `prepare_next_batch`（`kinternal` 档，~0.23 µs/行程）而非图循环，量级证据支持这一选择。
  判据：四策略均不满足 C1–C5（`kinternal` 仅 C1/C2）。与 §24（依赖维度→调度级）正交，合起来给出
  CUDA Graph 这条降级路线的天花板：**能表达动态依赖的调度时机与控制流，但粒度恒停在 kernel 节点**。
  **本轮新增实验。** 此前 **P35-1 完成（1/1）**。补齐 `docs/04` §13.6「未测显式构图 API」与 §21.6「未测 CUDA Graph
  的 `PROGRAMMATIC` 边」两条尾巴——把「显式构图 + programmatic 边能否让下游在上游完成之前被调度」
  实测出来。`docs/04-experiments.md` 新增 **§24**，源码 `src/graph_programmatic.cu`、脚本
  `harness/graph_programmatic.sh`、原始输出 `harness/graph_programmatic.out.txt` +
  `harness/graph_programmatic_sens.out.txt`（目标卡 H100/sm90、固定空闲 device 3、nvcc 13.2、
  iters=30）。一张图 A→B、`%globaltimer` 打点、四档边 `block`/`prog_nt`/`prog`/`lce`
  （`cudaGraphEdgeData.from_port` = Default/Programmatic/Programmatic/LaunchCompletion、
  `type` = Default/Programmatic）。核心结论：(A) **关键量是「下游调度点」`sched`**——`block`/`prog_nt`
  的 `sched≈A_total`（完成即放行），`prog`（A 在 prologue 后 trigger）的 **`sched=51.0µs≈pro`**、
  `lce` 的 **`sched=0.32µs≈0`**——**图的 programmatic 边把下游调度点前移到上游完成之前**（下游
  prologue 与上游 tail 重叠），这正是 stream 上 `cudaStreamWaitEvent` 做不到的（§21）；(B) **但 `rel`
  （下游 `cudaGridDependencySynchronize` 返回）恒 ≈ A_total**——端口只决定下游**何时被调度**，sync
  仍等**上游完成**，与 §21 一致；(C) Section B 证 programmatic 端口在**所有 block 都 trigger** 时激活
  （`prog_sched ≈ pro+(G-1)*stagger`，G=1→32）；(D) Section C 证 **capture 图与显式构图 PDL 语义逐项
  相同**（`sched` 51.01 vs 51.04、`rel` 均 101.60）；(E) 节点级 PDL 序列化属性被驱动**拒绝**
  （`driver_types.h:3988/4067` 标 "Valid for launches"，不在 graph node 属性列表），端口只由边提供。
  敏感性检查（`pro=200000`）`prog_sched=101.5µs≈pro` 确认 `sched` 精确跟随 trigger 点。判据不变
  （四档均不满足 C1–C5），是 CUDA Graph 这条降级路线的**下界证据**。**本轮新增实验。**
  此前 **P34-1 完成（1/1）**。兑现「下一步」剩余可选方向「把 DeepGEMM Mega Gate 的
  `cluster_dim` 路径做一次 sm90 等价复刻」——`docs/04-experiments.md` 新增 **§23**，源码
  `src/mega_gate_cluster.cu`、脚本 `harness/mega_gate_cluster.sh`、原始输出
  `harness/mega_gate_cluster.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2）。
  把 Mega Gate 的**协调路径**（非 SM100 指令；一手核对 `057ca59`）用 sm90 复刻：**CTA pair 切专家维**
  （`cluster_dim=num_mma_ctas`，`sm100_bf16_mega_gate.hpp:74` / `sm100_bf16_mega_gate.cuh:55-58,136-142`）、
  **split-K 写全局 score scratch** `[token][split][expert]`（`layout/mega_gate.cuh:78-107`）、
  **ScoreBarrier**（`(grid+1)*64+arrived` 无符号距离去陈旧；`scheduler/mega_gate.cuh:13-39`）会合后
  跨 split 归约 + `sqrt_softplus`/top-k（`epilogue/sm100_mega_gate.cuh:19-23,107-152`）。对照变体
  `fused`（分数留 SMEM、无 scratch/barrier）。核心结论：(A) **7 组 `SPLIT×MMA`(+fused) 与参考逐 token
  一致**（`idx_mismatch=0/16384`，`wmax≤2.1e-07`）；(B) **占满机器**——固定 token（NT=16）只增逻辑
  CTA，`grid 16→128` 延迟 **3.51×**（SPLIT=2 1.63× 略优于 MMA=2 1.57×），对上 Mega Gate heuristic
  `num_worker_groups=num_sms/num_logical_ctas`（`heuristics/mega_gate.hpp:137`）；(C) **已饱和**
  （NT=128，grid≈#SM）再 split 只增波次与协调开销（gate 比 fused 慢 **~11%** / +24 µs）；
  (D) ncu：fused vs gate SPLIT=1 的 **L2 写差 +0.39M sectors（≈+12.6 MB）**、SPLIT4×MMA2 的 DRAM 写
  1.94→**19.84 MB**——split-K 部分分必经 L2/DRAM 一次往返回合（`docs/04` §17 损失三的 gate 形态，
  但 Mega Gate 只走**一次** round-trip）。判据：gate 满足 C1–C4、仍非 C5。**本轮新增实验。**
  此前 **P33-1 完成（1/1）**。补齐 §20.6 / 「下一步」都点名的两条 **SM100 专属原语**
  （`SM100_TMA_2SM_LOAD` / tcgen05 2-SM MMA）：先**资料分析** CUTLASS（DeepGEMM 内
  `third-party/cutlass`，`057ca59`）——`copy_sm100_tma.hpp` 的 `SM100_TMA_2SM_LOAD_{1D..5D}`（`:52/78/…`）、
  `.cta_group::2` PTX（`:92`）、障碍 `& Sm100MmaPeerBitMask(0xFEFFFFFF)`（`:45/63`，注释「两 CTA 都执行、
  清 peer 位→事务字节更新 CTA0 屏障」）、`sm100_mma_warpspecialized.hpp:234`「2SM MMA 里只有一个线程
  发 TMA/更新屏障」、`SM100_MMA_{TF32,F16BF16}_2x1SM_SS`（`mma_sm100_umma.hpp:382/422`，`elect_one_sync()`
  单线程发 `tcgen05.mma.cta_group::2`）、`umma_arrive_2x1SM`（`barrier.h:779-789`）；再在 H100 上做
  **sm90 投影实测**——`docs/04-experiments.md` 新增 **§22**，源码 `src/two_sm_load.cu`、脚本
  `harness/two_sm_load.sh`、原始输出 `harness/two_sm_load.out.txt`（目标卡 H100/sm90a、固定空闲
  device 2、nvcc 13.2、cluster 恒 C=2、每配置独立进程）。核心结论：一个 2-CTA cluster 装 tile A+B
  三模式（mode0 indep / mode1 mcast / mode2 split=单 CTA 连发 A→己、B→对端 via `mapa`），全 `maxerr=0`；
  **mcast 随 chunk 增大越占优**（grid=128、chunk 4→32KB，`m1/m0` 0.964→0.825，32KB 快 **1.21×**）、
  **split 越大越吃亏**（`m2/m0` 1.032→1.130）；ncu 源侧 **mode0/mode2 DRAM 209.76/209.73 MB、L2 read
  sectors 6,553,600 完全相同，mode1 恰减半（104.89 MB / 3,276,800）**——**multicast 减半的是源流量，
  分区投递不省源流量**，2SM 的价值在硬件层（一次发、一个屏障、一对 SM 分持 TMEM）而非减字节，sm90
  缺席 `.cta_group::2`、软件投影反慢 ~13%。判据满足 C1–C3、细化 C4，仍非 C5。**本轮新增实验。**
  此前 **P32-1 完成（1/1）**。补齐 `docs/04` §20.6 明留的「`cudaLaunchAttributeProgrammaticEvent`
  未覆盖」尾巴，把 kernel 级「进度/事件」原语能前移到哪一点量出来——`docs/04-experiments.md` 新增
  **§21**，源码 `src/programmatic_event.cu`、脚本 `harness/programmatic_event.sh`、原始输出
  `harness/programmatic_event.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  `iters=30`、`pro=tail=100000` cyc、`step=20000`）。核心结论（三 Section）：
  - **A** 单 block：`post`/`early`/`none`/`bstart` 全 ≈ A 整段（108 vs 116 µs）= **完成点放行**，
    只有 `lce`（`LaunchCompletionEvent`）落到 **7.77 µs**（全 block 启动即放行）→ **ProgrammaticEvent
    的提前 trigger（含 `triggerAtBlockStart`）不放行 stream/host 依赖**（对上 `cuda.h:2227`；
    `driver_types.h:4082` 说 LCE 只是对 `cudaGridDependencySynchronize()` 不可见）。
  - **B** 多 block 错峰：`early`/`bstart`/`post` 完全重合、随 `(G-1)*step` 线性（G=128→1391 µs），
    `lce` 恒定 ~7.5 µs。
  - **C** 同 stream PDL：无 trigger 208.5 / 有 trigger 157.7 µs ≈ `pro+max(tail,prol)+crit`——**trigger
    只允许下游 prologue 与上游 tail 重叠**，`gridDependencySynchronize()` 仍等上游完成；挂不挂
    ProgrammaticEvent 属性结果相同（157.72 vs 157.74），端口由 trigger 本身给、事件属性只为图
    PROGRAMMATIC 边记录（`cuda.h:2047/2060`）。
  含义：kernel 级**没有**「中间进度早于完成、可被别的执行体看到」的通用信号，这正是 MPK/DeepGEMM 把
  协调下沉到设备内 event counter/原子 ring barrier 的原因。同步 `docs/03` §2.6 交叉引用。**本轮新增实验。**
  此前 **P31-1 完成（1/1）**。补齐 `docs/04` §18.6 / §19.6 两条都留的「**未测 2D/3D cluster**」
  尾巴，把 P29-1（§18 DSM）/ P30-1（§19 TMA multicast）从 1D 推广到**多维**，并补上 §19 未测的
  **multicast 子集掩码**——`docs/04-experiments.md` 新增 **§20**，源码 `src/cluster_2d.cu`、脚本
  `harness/cluster_2d.sh`、原始输出 `harness/cluster_2d.out.txt`（目标卡 H100/sm90a、**固定空闲
  device 3**、nvcc 13.2、每配置独立进程）。核心结论：**rank 布局实测确认** `block_rank()==
  x+y*cx+z*cx*cy`（行主序线性下标，4 组 cluster 全 mismatch=0）；**2D DSM 环形与 1D 同档**
  （C=2/4/8/16→588/552/533/528 GB/s，对上 §18 的 560.7，maxerr=0）；**multicast 子集掩码精确生效**
  （整群 mode1 与逐行 mode2 全 maxerr=0，逐行只发给该行 cx 个 rank）；**源侧** mode1 比 mode0 L2 read
  sectors 少 **4.38×**、DRAM 不变，而 mode2 的 L2 恰为 mode1 的 cy=4 倍、DRAM 为 mode0 的 4 倍
  （逐行读不同源、无去重——子集广播价值在**投递范围正确**）；`cluster.sync` 只与 block 数有关
  （448→505 ns）；**容量边界不变**（3D 可用、仍 16 block 上限、grid 逐维对齐）。踩坑：2D launch
  逐维对齐、2D 下 cluster block 的全局线性 id 不连续导致首版假 `maxerr`。**本轮新增实验。**
  此前 **P30-1 完成（1/1）**。补齐 `docs/04` §18.6 明确留的「cluster 内 `mbarrier`/TMA 组合
  未覆盖」尾巴：**TMA multicast**（一次读、投递到 cluster 内 C 个 CTA 的 SMEM）——`docs/04-experiments.md`
  新增 **§19**，源码 `src/tma_multicast.cu`、脚本 `harness/tma_multicast.sh`、原始输出
  `harness/tma_multicast.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、每配置独立进程）。
  核心结论：**语义实测确认**（仅 rank0 发 multicast、其余 CTA 在各自 mbarrier 等到、全部 `maxerr=0`，
  对上 DeepGEMM `tma_copy.cuh:77–86`）；**delivery 吞吐两模式打平**（单 cluster 215→1563 GB/s 随 C 线性、
  多 cluster 饱和 ~6.2–6.8 TB/s，multicast 仅快 0.1–1.1%）；**收益在源侧**（ncu：L2 read sectors 少
  **2.10×**、L2 流量少 **1.84×**，DRAM 字节相同——working set 装进 L2 且同步读同序列，L2 去重；
  指令级源读少 C× 是上界，真值以 ncu 的 2× 为准）。含义：multicast 省 **L2 请求/带宽**，把省下的
  让给其它并发 task——正是 DeepGEMM Mega MoE「Always multicast on A」（一手核对 `057ca59`）的理由。
  与 P29-1（§18 DSM）合成「跨 SM 搬运两跳」。**本轮新增实验。**
  此前 **P29-1 完成（1/1）**。补齐 `docs/04` §6.6 明确留的「sm90 的 cluster 未覆盖」尾巴：
  **Thread Block Clusters + 分布式共享内存（DSM）** 作为跨 SM「不落 HBM」的中间量交换原语。
  `docs/04-experiments.md` 新增 **§18**，源码 `src/cluster_dsm.cu`、脚本 `harness/cluster_dsm.sh`、
  原始输出 `harness/cluster_dsm.out.txt`（目标卡 H100/sm90、SMs=132、**固定空闲 device 1**、nvcc 13.2、
  **每配置独立进程**避免 sticky context）。核心结论：**DSM 远端读与本地 SMEM 同量级**（C=2 0.99×、
  C≥4 仅 1.17×；对照 plain 全局流式读 690.8 GB/s）；**ring 段位移交换 dsm/hbm = 1.06→1.27×**
  （seg 1→8 KB，maxerr=0 逐位一致且可复现；scratch 仅 2 MB 驻 L2，故是 L2 往返对照，DSM 8 KB 时
  1330 GB/s 已超 L2 的 1049 GB/s）；`cluster.sync()` = 504→767 ns 比 `grid.sync()`（1113 ns）便宜
  **1.5–2.2×**；**cluster 硬上限 16 block、grid 必须是 C 的整数倍** → 132 SM 整机 megakernel 只能
  切 ~8 个 cluster、跨 cluster 仍走 L2/HBM（= C4 的更细实现，非 C5）。踩坑：**变界 stride 循环读
  远端 SMEM 稳定触发 `unspecified launch failure`（-O3 代码生成），改固定索引单元素读规避**。
  相关机制与 P20-1（§14 SM 共驻）、P24-1（§17 损失三字节）交叉引用。**本轮新增实验。**
  此前 **P28-1 完成（1/1）**：兑现「下一步」可选方向①：**用 `-G`（`--device-debug`）重建
  JIT 内核 + 保留 JIT 临时目录 + cuda-gdb attach 挂起进程**，把 §11.7 留下的「读不出卡在哪个 event 计数」
  补上——`docs/09-mpk-build.md` 新增 **§13**，脚本 `harness/mpk_debug_counters.sh`、命令
  `harness/mpk_debug_counters.cmds`、补丁 `harness/mpk_debug_build.patch`、探针
  `harness/mpk_cudagdb_probe.py`（`MPK_KEEP_TEMPDIR`），原始输出 `harness/mpk_debug_counters.out.txt`
  （目标卡 H100/sm90、device 7、快照 `f9eb70c`，用时先把 z3 重钉 `4.16.0.0`、跑完按 B2 切回 `4.15.4.0`）：
  ① `-G` 下宿主阻塞在**第二次** `cudaLaunchKernel`、设备上只有 `worker_kernel`（比 §11 更强）；
  ② 数值：`worker_queue_last_ready_task_id[0/1]=0/0`（worker 零 task）、`sched_queue_last_ready=
  next_free=1`（1 个未消费 event）、`all_event_counters[0/1/2]=0/0/0` 对阈值 `1/1/16`；③ 结论：不是阈值
  凑不满，而是**首个 event 从未被派发**。此前 **P27-1 完成（1/1）**：新建 `docs/10-megakernel-precedents.md`（**纯资料分析**）+ 博客
  post `mega-kernel-11-precedents`（weight 12）并更新 `mega-kernel-00-index` 目录第 11 项：把 MPK §2.2
  （快照行 325–344）与 §8（行 1851–1862）反复转引、但本专题此前只作转引的两个**手工 mega-kernel 先例**
  回到一手资料——**FlashMoE**（`arXiv:2506.04667`，NeurIPS '25；§1 Table 1：1 launch vs ≤550、§3 actor
  + tile task + Theorem 3.1 非阻塞 NVSHMEM、§4 SM 利用率 93.17%/8 卡 6.4×）与 **Llama-1B megakernel**
  （Hazy Research 2025-05-27；~100 算子单 kernel、H100 <1 ms/B200 <680 µs、SMEM 分页 + global 计数器）。
  与 DeepGEMM/MPK 逐维对照（四者均满足 C1–C5，分野=融合范围 × 手工/自动）。**本轮无新实验**，相关机制
  实测交叉引用 `docs/04` §1/§9/§14/§16/§17。此前 **P26-1 完成（1/1），P0–P26 全部完成**。P26-1 在 §11 的
  clean scheduler-first 构建上把 `docs/09` 的端到端延迟从单算子推进到 **4 算子 DAG 单 megakernel**
  （`docs/09` §12，`harness/mpk_latency_multi.py/.sh` + `harness/mpk_latency_multi{,_config}.out.txt`）：
  多算子延迟 41.5–49.6 µs、比单算子只慢 ~1.2×，相对 kernel-per-operator 从单算子 1.38× 升到
  **3.5–4.8×**，并记录快照 `linear_swapAB_hopper.cuh:71` 的 **batch≤16 编译期硬约束**。
  此前 **P25-1（续 5）根因钉死、P25-1 记 `[x]`**：用 **cuda-gdb
  attach** 拿到设备侧证据（挂起时只 `worker_kernel` 常驻、spinning 在取 task 的 `ld_acquire`；
  `scheduler_kernel` 从未被派发，4:4 下 128 SM 空闲亦如此），并用**只交换 launch 顺序**的消融
  （scheduler 先于 worker）让 plain 构建 **4/4 收敛且正确**——详见 `docs/09` §11 与下方本轮条目。
  此前 **P25-1 续 4 完成**（根因再定位 2：证伪 §9 的「默认流历史」「host launch 排序」两条候选、
  排除 co-residency 与频率）。此前 **P25-1 续 3 完成**（把挂起降到 launch/握手时序敏感竞态、
  证伪 worker 轮询循环内存序）。此前 **P24-1 完成（1/1）**（用 ncu 直接测量损失三的 HBM 字节，
  把 P5-2 的算术推断升级为实测）。此前：脚手架就绪；**P0 全部完成**；**P1 全部完成（3/3）**；**P2 全部完成（4/4）**；
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
  **P22-1 去中心化调度的 DAG 形状敏感性（1/1）完成**（把 P15-1 的链推广到分层扇入/
  扇出图、对照 barrier/wave/decen/static 四机制、实现 MPK §5.2 批量取 task 的甜点标度）。
  **P23-1 跨层软件流水 + 环形缓冲预取（1/1）完成**（本轮新增，把 MPK §5.3 的 paged SMEM +
  cross-task software pipelining 实测出来，补齐 P22-1 §15.6 明留的尾巴）。
  **P0–P24 全部完成**。**P25-1（线上实现构建与端到端冒烟）完成 `[x]`**：目标卡 H100/sm90 上
  mirage（`f9eb70c`）**构建 + JIT 编译 + launch + 收敛 + 正确性 + 端到端延迟**全部打通，
  「不收敛」根因由 `docs/09` §11 的 **cuda-gdb 设备侧证据 + launch 顺序消融**钉死（scheduler 先于
  worker launch 即 plain 构建 4/4 收敛；见「阻塞」B1 已解决）。**本专题 P0–P25 全部完成**。
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
- **下一步：全专题继续（P0–P36 全部完成）**。P36-1 已把 CUDA Graph 的**控制流**维度补成实测
  （`docs/04` §25：conditional graph nodes 一图覆盖可变行程/数据相关分支，但每行程一次设备侧
  kernel 重分发，比 in-kernel 循环贵 ~20×——「表达能力」而非「性能」）。P35-1 已把 §13.6/§21.6 两条「Graph 显式构图 /
  programmatic 边」的尾巴补成实测（`docs/04` §24：programmatic 边把下游**调度点**前移到上游完成之前、
  但 `gridDependencySynchronize` 仍等完成、capture ≡ 显式构图、节点级 PDL 属性被拒）。P33-1 已把
  §20.6 / 「下一步」都点名的两条
  **SM100 专属原语**处理掉（`docs/04` §22：`SM100_TMA_2SM_LOAD` / tcgen05 2-SM MMA 的资料分析 +
  sm90 投影实测——multicast 复制投递源流量恰减半、分区投递不省源流量、2SM 是硬件层把一对 SM 绑成
  一个计算单元）。P31-1 已把 §18.6/§19.6 的「未测 2D/3D cluster」补成
  实测（`docs/04` §20：rank 布局 `x+y*cx+z*cx*cy`、2D DSM 与 1D 同档、multicast 逐行子集掩码精确、
  容量逐维约束）。P30-1 已把 §18.6 的「cluster 内 `mbarrier`/TMA 组合
  未覆盖」补成实测（`docs/04` §19：TMA multicast 一次读、C 个 SMEM 收——语义确认、delivery 打平、
  源侧 L2 少 2×、DRAM 不变）。P29-1 已把 §6.6 的「sm90 cluster 未覆盖」补成实测
  （`docs/04` §18：Cluster + DSM 跨 SM 交换原语、`cluster.sync` vs `grid.sync`、cluster 容量边界）。
  P28-1 已把 §11.7 的「读不出卡在哪个 event」用 `-G` +
  cuda-gdb 补成数值证据（`docs/09` §13）。P27-1 已把 MPK 转引的两个手工 mega-kernel 先例（FlashMoE、
  Llama-1B megakernel）回到一手资料（`docs/10`，纯资料分析）。P26-1 已在 clean scheduler-first 构建上把端到端延迟
  从单算子推进到 4 算子 DAG（`docs/09` §12），**兑现下面的可选方向③**。P25-1 已构建 + launch +
  **收敛 + 正确性 + 端到端延迟**，并且「不收敛」根因已由 `docs/09` §11 钉死：**cuda-gdb attach**
  证据（挂起时只有 `worker_kernel` 常驻、spinning 在取 task 的 `ld_acquire`；`scheduler_kernel`
  从未被派发，4:4 下 128 SM 空闲亦如此）+ **launch 顺序消融**（scheduler 先于 worker launch →
  plain 构建 4/4 收敛且正确，`mpk_sched_first.patch`）。§9/§10 追的 dtype 转换只是「扰动 scheduler
  派发时机」的同一旋钮。   **无遗留必做项**；如再增量可选：
  ① ~~用 `-G`（设备调试信息）重编译以读取 `config.all_event_counters`，把「卡在哪个 event」的数值也钉死~~
     （**P28-1 完成**：`-G` + 保留 JIT 临时目录 + cuda-gdb → worker 零 task、`all_event_counters` 全 0、
     `sched_queue` 首个 event 未消费；见 `docs/09` §13）；
  ② 把 scheduler-first 的顺序离线报告给 MPK 上游核对（不构成 bug 判定）；
  ③ ~~在 clean scheduler-first 构建上重跑多算子/更大 batch 的端到端延迟~~（**P26-1 完成**：4 算子
  43–50 µs / 相对 kernel-per-operator 3.5–4.8×，并记录 batch≤16 编译期约束；更大 batch>16 受
  `linear_swapAB_hopper.cuh:71` 阻，未做）。
  注意 z3 版本互斥（「阻塞」B2）。现象刻画见 `docs/09` §7–§11，延迟见 §8/§11.6/§12。
- 其它可选的增量方向（不属现有清单）：① ~~为 `docs/02`/`03`/`04`/`06`/`07`
  补建对应博客 post~~（P8-1 完成 `02`/`03`/`04`/`06`；P11-1 补建 `mega-kernel-08-mpk-tasks`）；
  ② ~~补做 P5-2 §2.6 / P5-1 §1.6 提到的「更小 M / 更多算子 / 多 block」实验~~（**P9-1 完成 N 算子链 + 小 M**；
  **P10-1 完成 P5-1 §1.6「扫 persistent 网格大小」**；**P12-1 完成 P5-3 §3.6 / P10-1 §5.5「多 block
  PDL occupancy 资源约束」**；**P13-1 完成 P5-2 §2.6 / §5.6「跨 tile software pipelining」的损失一实测**；
  **P14-1 完成 megakernel 成本侧「寄存器固定为各 task 最大值」实测**；**P15-1 完成 C5 去中心化调度最小实现**）；
  ③ ~~走读 `tasks/hopper/*.cuh` 单 task 内部实现~~（P11-1 完成）；~~「在容器内 `pip install -e .`
  构建 mirage sm90 路径并跑端到端数字」~~（**P25-1 已构建 + JIT 编译 + launch 成功**，端到端执行
  未收敛，见「阻塞」B1）；
  ④ ~~megakernel 成本侧「代码足迹 / 指令缓存」实测~~（**P16-1 完成**，与 P14-1 寄存器足迹合成双层足迹）；
  ⑤ ~~megakernel 成本侧「寄存器足迹的可绕开性」：`setmaxnreg` 实测~~（**P18-1 完成**，收回 P14-1 的代价）；
  ⑥ ~~`docs/04` §6.6「未测 PDL 与 CUDA Graph 叠加」~~（**P19-1 完成**，测出 capture 保留 PDL 语义、
  叠加次乘非相乘、上限独立）；
  ⑦ ~~P15-1 §9.6「未测扇入/扇出 DAG、批量/预取取 task」~~（**P22-1 完成**：分层扇入/扇出图 ×
   barrier/wave/decen/static 四机制、MPK §5.2 批量取 task 甜点 K≈W/G、静态分配随扇入变差）；
  剩「MPK 端到端执行收敛」的问题（构建已在 **P25-1** 打通，见「阻塞」B1）。
   ⑧ ~~P22-1 §15.6「跨层流水/环形缓冲预取」~~（**P23-1 完成**：MPK §5.3 的 paged SMEM +
   cross-task software pipelining 实测——跨层预取峰值 1.19–1.20×（vs barrier）、深度按访存占比选
   K≈4、纯访存时净亏；eager/barrier 1.25× 验证消 launch 主收益）；
   ⑨ ~~`docs/04` §6.6「sm90 cluster 未覆盖」~~（**P29-1 完成**：Thread Block Cluster + DSM 跨 SM
   交换原语实测——DSM 远端读 ≈ 本地 SMEM、dsm/hbm 1.06–1.27×、`cluster.sync` 比 `grid.sync` 便宜
   1.5–2.2×、cluster 上限 16 block/grid 须为 C 倍数；见 `docs/04` §18）。
   剩余可选：~~cluster 内 TMA multicast + `mbarrier` 组合~~（**P30-1 完成**：`docs/04` §19）、
   ~~2D/3D cluster + multicast 子集掩码~~（**P31-1 完成**：`docs/04` §20）、
   ~~SM100 `SM100_TMA_2SM_LOAD` / tcgen05 2-SM MMA~~（**P33-1 完成**：`docs/04` §22 资料分析 +
    sm90 投影实测）、~~`cudaLaunchAttributeProgrammaticEvent`~~（**P32-1 完成**：`docs/04` §21）、
     ~~把 DeepGEMM Mega Gate 的 `cluster_dim` 路径做一次 sm90 等价复刻~~（**P34-1 完成**：`docs/04` §23
     ——CTA-pair 切专家维 + split-K score scratch + ScoreBarrier 的 sm90 等价复刻；multi-CTA 逻辑任务
     的价值在「占满机器」，饱和后只剩协调开销）、~~显式构图 + CUDA Graph 的 `PROGRAMMATIC` 边~~
     （**P35-1 完成**：`docs/04` §24——programmatic 边把下游**调度点**前移到上游完成之前、但
     `cudaGridDependencySynchronize` 仍等完成、capture ≡ 显式构图、节点级 PDL 属性被驱动拒绝）。
    **至此「下一步」剩余可选方向全部完成，全专题无遗留必做项。**
- **本轮（P25-1 续）**：续做 P25-1 的「MPK 端到端跑通」，把上轮「执行不收敛」从玄学死锁推进到
  **可复现、可归因的时序竞态**。`docs/09-mpk-build.md` 新增 **§7**（收敛性诊断），新原始输出
  `harness/mpk_converge_plain.out.txt` / `harness/mpk_converge_instrumented.out.txt` /
  `harness/mpk_converge_attribution.out.txt` / `harness/mpk_dbg_trace.out.txt`，patch
  `harness/mpk_watchdog.patch`（全量插桩）与 `harness/mpk_workaround_only.patch`（最小修复）。
  快照仍为 `f9eb70c`，目标卡 H100/sm90，**`CUDA_VISIBLE_DEVICES=7` 隔离空闲卡**，单次 `timeout 200s`。
  核心结论：
  - **[复现] 用快照自己的用例排除自建脚本因素**：直接跑官方
    `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`（官方 `get_configurations_from_gpu(0)`
    = **128 worker/16 scheduler**），plain 构建 **3/3 挂起**（`EXIT=124`、GPU 100% 自旋）——
    与 `mpk_smoke.py` 无关，是「快照 + 本机」的行为。
  - **[修好] 插桩后收敛且正确**：在 `persistent_kernel.cuh` 的 worker 取 task 等待循环（`:1013`）与
    scheduler 事件等待循环（`:1335`）加自旋计数守门狗（`printf`+`__trap()`）＋ scheduler 0 逐事件打印，
    经 `persistent_kernel.py:365-371` 加 `-DMPK_DBG_WATCHDOG`；full instrumented **3/3 收敛**
    （`EXIT=0`），`max|out-ref|`=0.015625/0.015625/0.03125（bf16 容差内）；**守门狗从未触发**。
  - **[trace] 完整事件序列实锤**：`harness/mpk_dbg_trace.out.txt` 显示
    seed `event2(910,nt16,iter0)` → `event1(903,nt1)` → 16 个 rmsnorm 触发 `event2(910,iter1)` →
    `prepare_next_batch->FALSE` → `event0(911)` 终止，与 `docs/09` §4 的收敛路径逐步吻合。
  - **[判别] 排除 H2、排除 H1 的"阈值凑不满"**：16 个 task 全部触发 event2（否则到不了 nt16、
    不会二次派发 event2、也就不会有 `prepare_next_batch->FALSE`），故非 task 单体内 TMA/mbarrier 卡住。
  - **[归因] 消融定位到 worker 轮询循环**：A plain 3/3 挂、B prints-only 2/2 挂、
    C worker-spin-only(no prints) 2/2 收敛、D worker-spin+prints 2/2 收敛、E full 3/3 收敛。
    **唯一决定性变量 = `execute_worker` 取 task 等待循环里"每轮多一次自旋自增+分支"的时序扰动**；
    scheduler 侧逐事件 printf（B）不足以修复。最小修复 patch 只加该自旋计数即让官方用例稳定收敛，
    **反证竞态在 worker 侧"等 task"的抢跑/丢失唤醒**上。
  - **局限**：未把根因收敛到某条 load/store 内存序；端到端延迟仍未测（只测正确性+收敛/挂起）。
    下一步候选：`compute-sanitizer --tool racecheck` 抓竞争、把自旋换成 `__nanosleep` 退避验证纯时序、
    向 MPK 上游核对 `f9eb70c` worker 抢跑路径。**不构成对上游的 bug 判定**。P25-1 仍记 `[~]`。
- **本轮（P33-1）**：`docs/04-experiments.md` 新增 **§22**「2SM 协作：SM100 `cta_group::2` TMA load
  与 tcgen05 2-SM MMA 的 sm90 投影」——新源码 `src/two_sm_load.cu`、脚本 `harness/two_sm_load.sh`、
  原始输出 `harness/two_sm_load.out.txt`（目标卡 H100/sm90a、**固定空闲 device 2**、nvcc 13.2、
  cluster 恒 C=2、128 线程/CTA、每配置独立进程）。分两步：
  **(a) 资料分析**（CUTLASS，DeepGEMM 内 `third-party/cutlass`，commit `057ca59`）：
  `include/cute/arch/copy_sm100_tma.hpp` 的 `SM100_TMA_2SM_LOAD_{1D..5D}`（`:52/78/104/130/156`）、
  dispatcher `SM100_TMA_2SM_LOAD`（`:182`）、`_MULTICAST_*`（`:229` 起）、`_IM2COL_*`（`:402` 起）；
  PTX 带 `.cta_group::2`（2D 在 `:92`）；障碍句柄 `& Sm100MmaPeerBitMask`（`=0xFEFFFFFF`，`:45/63`）
  注释 *"Executed by both CTAs. Set peer bit to 0 so that the transaction bytes will update CTA0's
  barrier."*；集体层 `include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp:234` *"Only one thread
  issues the TMA and updates the barriers in a 2SM MMA"*；MMA 侧 `SM100_MMA_TF32_2x1SM_SS`（`:382`）/
  `SM100_MMA_F16BF16_2x1SM_SS`（`:422`）`static_assert(M==128||256,"...for 2 CTA cluster MMA.")`（`:384`）、
  `tcgen05.mma.cta_group::2.kind::{tf32,f16}`（`mma_sm100_umma.hpp:406/446`）由 `elect_one_sync()` 单线程发、
  `umma_arrive_2x1SM`（`include/cutlass/arch/barrier.h:779-789`）单点到达一对 SM 的屏障；DeepGEMM
  Mega MoE/Gate `cluster_dim` 最小单位即 2SM（`sm100_bf16_mega_moe.hpp:138`/`sm100_bf16_mega_gate.hpp:74`）。
  **(b) sm90 投影实测**：一个 2-CTA cluster 把逻辑 tile（chunk A+B）装进两个 SM 的 SMEM，三模式——
  mode0 `indep`（两 CTA 各发各的 bulk）、mode1 `mcast`（rank0 TMA multicast 复制投递）、mode2 `split`
  （rank0 两次 `cp.async.bulk.shared::cluster`：A→己、B→对端（`mapa` 求对端 SMEM 与 mbarrier）=
  `cta_group::2` 投影）。全部 case `maxerr=0.00000`。核心结论：(1) **多 pair（grid=128）模式差随 chunk
  放大**——chunk 4→32KB，`m1/m0` **0.964→0.825**（mcast 32KB 快 **1.21×**）、`m2/m0` **1.032→1.130**
  （split 越大越慢）；三模式随 R **严格线性**（每轮固定成本恒定，无隐藏批处理）。(2) **ncu 源侧字节
  是决定性证据**（grid=128/chunk=8KB/R=200）：mode0/mode2 DRAM 读 **209.76/209.73 MB**、L2 read
  sectors **6,553,600**（**完全相同**），mode1 **104.89 MB / 3,276,800**（**恰减半**）——**multicast
  减半的是源流量**（同一 tile 只从源读一次），**分区投递不省源流量**（A、B 各读一次与发出者数量无关）。
  (3) **2SM 的价值在硬件层**（一次 TMA 发、一个屏障、一对 SM 分持 TMEM），**不在**减字节；sm90 缺席
  `.cta_group::2`，软件投影（单 CTA 连发含远端 SMEM 写）在 32KB 反慢 ~13%——sm90 上「两 CTA 各发各的」
  优于「单发分区」。判据：满足 C1–C3、`cta_group::2`/multicast 细化 C4（设备内协调），**仍非 C5**。
  局限：sm90 投影非 SM100 本体（两套独立 mbarrier，非 CTA0 单点）、未测 tcgen05 2SM **算**侧、mode1
  与 mode0/2 语义不同（复制 vs 分区）、固定 C=2、测的是载入原语非端到端。**本轮新增实验**；与
  §18/§19/§20/§21 合成「跨 SM 协作原语」的 sm100 展望。
- **本轮（P28-1）**：`docs/09-mpk-build.md` 新增 **§13**「设备侧数值证据：-G 调试构建 + cuda-gdb 读出事件
  计数」。新增补丁 `harness/mpk_debug_build.patch`（只给 `get_compile_command()` 拼 `MPK_EXTRA_DEFINES`，
  **不碰 kernel 代码**）、脚本 `harness/mpk_debug_counters.sh`、命令 `harness/mpk_debug_counters.cmds`，
  探针 `harness/mpk_cudagdb_probe.py` 增 `MPK_KEEP_TEMPDIR=1`（阻止 mirage 的 `TemporaryDirectory` 在
  `compile()` 后回收 `test.cpython-…so`，否则 cuda-gdb 报 `Failed to map shared library`、拿不到类型），
  原始输出 `harness/mpk_debug_counters.out.txt`（目标卡 H100/sm90、device 7、快照 `f9eb70c`、
  `cuda-gdb` 13.2；用前重钉 z3 `4.16.0.0`、用后按 B2 切回 `4.15.4.0`）。核心结论：
  (1) **`-G` 构建宿主阻塞在第二次 `cudaLaunchKernel`**（栈 `cuLaunchKernel→cudaLaunchKernel→launcher .so`），
  设备上**只有 `worker_kernel`**、`cuda kernel 1`→`Invalid coordinates`——与 §11 设备侧证据一致且更强；
  (2) **数值**：`worker_queue_last_ready_task_id[0]=[1]=0`、worker0 `queue_idx=queue_pos=queue_len=0`
  （worker 一个 task 都没收到）；`sched_queue_last_ready_event_id=next_free_event_id=1`（1 个 event
  未消费）；`all_event_counters[0/1/2]=0/0/0` 对阈值 `all_event_num_triggers[0/1/2]=1/1/16`；
  (3) **判断**：不是「某 event 差数凑不满」，而是 **scheduler 队列首个（种子）event 从未被消费 →
  worker 零 task → 所有事件计数恒 0**，把 §11.4 的 launch-顺序根因从「设备上查不到 scheduler」推进到
  「第一个 event 派发前夜」；(4) **口径 caveat**：`-G` 关优化会改变 codegen 与宿主 launch 行为，
  数值刻画 **`-G` 挂起态**，设备侧结构与默认挂起态相同（§11 已交叉验证），不构成对上游 bug 判定；
  `next/last_task_pos` 两个 `__shared__` 游标仍 `<optimized out>`。**本轮新增实验**；兑现「下一步」①。
- **本轮（P27-1）**：新建 `docs/10-megakernel-precedents.md`（手工 mega-kernel **先例精读**，
  **纯资料分析、无新实验**）+ 博客 post `mega-kernel-11-precedents`（weight 12）并更新
  `mega-kernel-00-index` 目录第 11 项。补齐本专题此前只作**转引**的一手材料：MPK §2.2（快照行 325–344）
  与 §8（行 1851–1862）点名的两个手工 mega-kernel——**(a) FlashMoE**（`arXiv:2506.04667`，NeurIPS '25，
  代码 `github.com/osayamenja/FlashMoE`）：§1 Table 1「单 MoE 层 1 launch vs COMET 33 / Megatron-CUTLASS
  85 / Megatron-TE 261 / +DeepEP 432 / DeepSpeedMoE 550」，§3 actor 模型（Processor/Subscriber/Scheduler，
  OS 块 4 warp）+ §3.1 统一 task（每 task 一个 `__device__` 函数）+ §3.2 对称张量布局的非阻塞 NVSHMEM
  （**Theorem 3.1** write-write conflict-free、temporal buffering `2r=4×`、§3.2.1 in-place padding），
  §4 结果 **SM 利用率 93.17%**（vs DeepEP 14%）、8 卡延迟最高 **6.4×**、吞吐 **17.7 MTokens/s（5.7×）**、
  overlap **4×**、128 experts **6.6×**（全程 FP32 对基线 FP16），§5 局限（工程复杂度、期待 compiler/DSL）；
  **(b) Llama-1B megakernel**（Hazy Research 博客 2025-05-27 *Look Ma, No Bubbles!*，代码
  `github.com/HazyResearch/Megakernels`）：整图 ~100 算子单 kernel，三条 stall（512 block/148 SM→80 SM
  空转；H100 stream launch **2.1 µs**/graph **1.3 µs**；PDL `cudaGridDependencySynchronize` 太粗）、三套
  机制（设备内解释器 + 每 SM 指令序列 / **SMEM 分页 13×16 KiB** / global 计数器细粒度依赖、MLP 分 4 chunk），
  结果 H100 **<1 ms**、B200 **<680 µs**、比 vLLM ~2.5×、78% 带宽，B200 600 µs 分解（激活 250/计算 200/
  等权重 30/同步 40/杂项 80）。§3 与 DeepGEMM/MPK **逐维对照表**：四者均满足 C1–C5，分野=融合范围
  （段级/整图）× 调度（手工/自动），段级 T4 内部另有静态（DeepGEMM/Llama 指令）↔动态（FlashMoE actor）
  光谱；两篇先例的自陈局限正是 MPK 编译器路线的立足点。§6 给出可核对的真实链接与 MPK 快照行号。
  相关机制实测交叉引用 `docs/04` §1/§9/§14/§16/§17。**本轮无新实验。**
- **本轮（P26-1）**：`docs/09-mpk-build.md` 新增 **§12**「多算子 megakernel 的端到端延迟」——新建
  `harness/mpk_latency_multi.py`（4 算子 DAG 单 megakernel + batch 扫描 + torch kernel-per-operator
  参照）、运行器 `harness/mpk_latency_multi.sh`（clean 绕过 `harness/mpk_sched_first.patch`，**不动
  kernel 代码**），原始输出 `harness/mpk_latency_multi.out.txt`（batch 扫描）与
  `harness/mpk_latency_multi_config.out.txt`（worker 数扫描）。被测图 = Qwen3 dense MLP block
  `rmsnorm → linear(gate+up) → silu_mul → linear_with_residual`（hidden=4096、inter=2048、bf16），
  目标卡 H100/sm90、device 7、iters=200、快照 `f9eb70c`。核心结论：
  (1) **多算子延迟与单算子同量级**——4 算子块 128/16、batch=16 = **46.4 µs**，比 §8 单算子 39.05 µs
  只慢 ~7.4 µs（1.19×），因 3 个新增算子（~+160 task）全在**同一个常驻 kernel** 内（C1 仍单次提交、
  设备内 event 依赖、无 host launch / kernel barrier）；(2) **对 kernel-per-operator 的加速随算子数
  放大**——torch 朴素同块 175–205 µs，比值从单算子 **1.38×**（§8）升到 4 算子 **3.5–4.8×**，
  「加 3 算子的边际代价」MPK ~7.4 µs vs torch ~148 µs（约 5%），兑现 §8.3「算子数/中间量越大 MPK
  优势越大」，对上 `docs/04` §9/§17 损失二/三；(3) **延迟对 batch(1→16) 与 worker(128→16) 均近乎
  不敏感**（41.5–49.6 µs / ~10%），主导项仍是「双常驻 kernel 启动 + scheduler 冷启动 + done-event
  终止协议」；(4) **快照硬约束**——batch=32 的 MLP 图**编译期**失败于 `linear_swapAB_hopper.cuh:71`
  的 `static_assert(BATCH_SIZE <= 16, "…in swapAB")`（注释 `:69-70`），脚本记 `BUILD_FAIL` 后继续。
  所有成功 case `max|out−ref|=1.56e-02`；`Total tasks` 163/166/170/178、events 恒 37。判据 C1 在多算子
  图成立（§12.5 判据十二/十三/十四）。**本轮新增实验**；兑现「下一步」可选方向③。z3 已按 B2 用后
  恢复 `4.15.4.0`。
- **本轮（P25-1 续 5，根因钉死）**：续做 P25-1 的**根因定位**（§10.7 候选①「设备侧事件级证据」），
  用 **cuda-gdb attach 已挂起的进程**（非侵入式，避免 §9.3/§9.4 的插桩扰动），把「不收敛」钉死到
  **launch 顺序**。`docs/09-mpk-build.md` 新增 **§11**；新脚本 `harness/mpk_cudagdb_probe.py`、
  `harness/mpk_cudagdb.sh`、`harness/mpk_cudagdb.cmds`、`harness/mpk_sched_first.sh`；patch
  `harness/mpk_sched_first.patch`；新原始输出 `harness/mpk_cudagdb.out.txt`（128/16）、
  `harness/mpk_cudagdb_w4s4.out.txt`（4/4）、`harness/mpk_sched_first.out.txt`（顺序消融矩阵）、
  `harness/mpk_latency_schedfirst.out.txt`（clean 构建延迟）。快照 `f9eb70c`、目标卡 H100/sm90、
  `CUDA_VISIBLE_DEVICES=7`。核心结论：
  - **[设备侧证据] 挂起时只有 worker 常驻**：`info cuda kernels` 只列 `worker_kernel<<<(128,1,1),(256,1,1)>>>`
    （128 block 全 `running`，占 128 SM），PC 落在 `persistent_kernel.cuh:1016` 的
    `ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id[…])`（内联自 `mpk_atoms.cuh:52`）——
    **worker 卡在取 task 的等待循环**。`cuda kernel 1` → `Invalid coordinates`：**`scheduler_kernel` 不在设备上**。
  - **[排除 occupancy] 4:4 下 scheduler 仍不在**：worker 只用 4 个 block（至多 4 个 SM，132 SM 里 128 个空闲），
    `scheduler_kernel` 依旧不出现 → 死锁实为「**scheduler 从未被派发**」，修正 §10.4「两常驻 kernel 同时自旋」。
  - **[根因] 交换 launch 顺序即收敛**：`launch_persistent_kernel()` 默认先 worker 后 scheduler
    （`:1991-2006`，代码注释已提示 scheduler 先启动也可）；用 `#ifdef MPK_DBG_SCHED_FIRST` 把 scheduler
    放到 worker 之前 launch，**plain 构建 4/4 收敛且正确**（none ×3 + cast_1elem ×1，`max|out-ref|≤0.03125`），
    同会话默认顺序复现 HANG。⇒ **根因 = worker 的整机持久化 grid 先启动后饿死了 scheduler 的派发**。
  - **[统一前几轮]** §9 的 dtype 转换 / §9.3 的插桩敏感 / §10 的全 stream not-ready，都是**同一个
    「扰动 scheduler 派发时机」**的旋钮的不同表现；循环内存序（§9.2）、host 排序（§10.3）、频率（§10.5）
    均无关。
  - **[延迟复核] clean 构建同量级**：scheduler-first（无任何 kernel 代码改动）128/16、200 iters
    两次 mean 39.13 / 43.53 µs（median 39.12 / 40.54），与 §8 的 36.4–39.1 µs 一致 → §8 不是内插桩产物。
  - **局限/定性**：无设备侧类型信息（只有 `-lineinfo`）未能读出「卡在哪个 event 计数」；仅本机单 operator；
    **不构成对上游 bug 判定**（也可能是驱动对「整机持久化 kernel + 后继 kernel」的派发行为）。
    **P25-1 记 `[x]`**。
- **本轮（P25-1 续 4）**：续做 P25-1 的**根因再定位 2**，把 §9 的两条候选（「默认流历史」/「host
  launch 排序」）**都证伪**，并把挂起**定位到设备侧**。`docs/09-mpk-build.md` 新增 **§10**；新原始输出
  `harness/mpk_race_stream.out.txt`/`_stream2`/`_stream3`（流来源/规模/完成 + side-stream 非 cast 对照）、
  `harness/mpk_launch_sync.out.txt`（host 排序宏）、`harness/mpk_wait_trace.out.txt`/`_grid`（两 stream
  轮询）、`harness/mpk_clock_lock.out.txt`（锁频）、`harness/mpk_nsys.out.txt`（nsys 尝试）；
  脚本 `harness/mpk_race_stream.sh`、`harness/mpk_launch_sync.sh`、`harness/mpk_wait_trace.sh`、
  `harness/mpk_nsys.sh`、`harness/mpk_nsys_parse.py`；patch `harness/mpk_launch_sync.patch`
  （`MPK_DBG_LAUNCH_SYNC`/`MPK_DBG_DEVICE_SYNC`/`MPK_DBG_WAIT_TRACE`，经 `MPK_EXTRA_DEFINES` 注入）。
  快照 `f9eb70c`、目标卡 H100/sm90、`CUDA_VISIBLE_DEVICES=7`。核心结论：
  - **[判据五] 与默认流/完成/规模都无关**：cast 放到**独立非默认流**（`cast_s2`）、**完全不 sync**
    （`cast_s2_nosync`，可能仍在飞）、**只 1 个元素**（`cast_1elem`）**全 PASS**；`none_s2`（只建流不
    发 kernel）与 `none` **HANG** → §9.5「默认流历史」证伪，触发器是「设备上跑过一个 dtype 转换 kernel」。
  - **[判据六] 不是「任意 side-stream kernel」**：把 `mul/copy/clamp/sleep/matmul` 全改到独立流再同步
    （`mpk_race_stream3`）**全 HANG**，只有 `cast_s2` PASS → 签名**精确是 dtype 转换**这一类，与其他
    逐元素 kernel / `cudaMemcpyAsync` / cuBLAS / 纯时间都无关。
  - **[判据七] host 侧强制排序无效**：`prepare_kernel` 后插 `cudaStreamSynchronize(default)` 或
    `cudaDeviceSynchronize()`（`MPK_DBG_LAUNCH_SYNC`/`MPK_DBG_DEVICE_SYNC`）**仍 HANG**（cast 仍 PASS）
    → §9.5「launch/握手时序」假设也证伪，死锁在常驻 kernel 启动之后。
  - **[判据八] 两个常驻 kernel 都在设备侧自旋**：host 侧 `cudaStreamQuery` 轮询（`MPK_DBG_WAIT_TRACE`）
    显示 HANG 时 worker/scheduler **两条 stream 均 `no error` 启动、且 `device not ready` 持续 2s**
    （都还在跑）；PASS 时二者 **it=1 即 DONE**。`4:4/16:16/64:16` 三档**全部两 stream 自旋** →
    **排除 co-residency/SM 占用**，是 worker↔scheduler **双向死锁**。
  - **[排除频率]** `nvidia-smi -lgc 1980` 锁频后 none 仍 HANG、cast 仍 PASS → 非频率/boost 时序竞态。
  - **[nsys 失败]** 挂起用例经 `os._exit` 退出致 CUPTI 未 flush（只 3 条 API），改 `cudaProfilerStop`/
    `--capture-range` 后 nsys 在挂起 kernel 上不收敛/超时；host API 时间线证据缺失，但已从行为上排除 host 排序。
  - **机制候选（假设，非对上游 bug 判定）**：H-a 设备级「首次 dtype 转换」懒初始化路径（最可能）/
    H-b 与 `__nanosleep` 自旋的 L2 状态交互（弱，§9.2 已证循环内改动不修复）/ H-c 快照 `MODE_OFFLINE`
    + `test_mode` 终止协议本就依赖执行时序。**P25-1 仍记 `[~]`**，下一步：worker/scheduler 事件等待点
    printf 守门狗、compute-sanitizer synccheck/racecheck、上游核对 `MODE_OFFLINE` 终止协议。
- **本轮（P25-1 续 3）**：续做 P25-1 的**根因再定位**，把 §7.5 的「竞态在 worker 轮询循环」**降级**
  为「launch/握手层面的时序敏感竞态」。`docs/09-mpk-build.md` 新增 **§9**（判据矩阵 + 与 §7 的关系 +
  机制候选 + 局限），新原始输出 `harness/mpk_race_attr.out.txt`（前置工作矩阵 + grid）、
  `harness/mpk_race_sweep.out.txt`（**修正后**的循环内变体消融，替换上轮 device 0 的作废结果）、
  `harness/mpk_wlwatch.out.txt`（printf-only 守门狗）；脚本 `harness/mpk_race_attr.sh`、
  `harness/mpk_wlwatch.py`（`mpk_race_probe.py` 扩了 `cast/copyto32/to_half/to64/to_same/alloc/addrow`
  等前置工作）。快照 `f9eb70c`、目标卡 H100/sm90、`CUDA_VISIBLE_DEVICES=7` 隔离空闲卡（**GPU 0 当时
  被占满，上轮 device 0 的 sweep 已作废**）。核心结论：
  - **[判据一] 只有「dtype 转换 kernel」能避开挂起**：`cast`(bf16→fp32)/`copyto32`/`to_half`(→fp16)/
    `to64`(→fp64) 全 **PASS**；而 `sync`/`sleep`(300ms)/`copy`/`mul`/`clamp`/`addrow`/`matmul`/`alloc`
    全 **HANG**；`to_same`(同 dtype、**不发射 kernel**) 亦 **HANG**。含 cast 的规约
    `sumall/sumrows/mean/softmax/rmsnorm` 全 PASS。→ **不是纯时间、不是分配、不是访存/计算、不是规约
    本身**，决定变量是「前置流上真的跑过一个 dtype 转换 kernel」（与输出尺寸无关）。
  - **[判据二] grid 无关**：`4:4 / 16:16 / 64:16 / 128:16` **四档全 HANG**（worker kernel ~1 block/SM
    的 205824 B SMEM，但小 grid 也挂 → 排除 occupancy）。
  - **[判据三] worker 轮询循环的**内存序类**改动都不修复**：修正 `mpk_race_sweep.py`（上轮 `counter`
    变体把 `_dbg_spins` 误声明在循环体内、被编译器折叠，且跑在 device 0 污染），在 device 7 重跑
    `plain/counter/sleep0/sleep1000/compiler_barrier/threadfence/relaxed` 各 2 次，**全部 HANG ×2**。
    → 排除「编译器把 `next_task_pos` load 提升出循环」（`mpk_atoms.cuh:50-54` 的 `ld_acquire_gpu_u64`
    确实缺 `"memory"` clobber，但加 `asm volatile("" ::: "memory")` 不修复）、「缺 fence」
    （`__threadfence` 不修复）、「acquire 语义」（`ld.relaxed` 不修复）、「自旋节奏」（改 `__nanosleep` 值
    不修复）。
  - **[判据四] worker 轮询循环不是 worker[0] 长时间等待处**：新 `mpk_wlwatch.py` 在循环内插
    **只 `printf` 不 `trap`** 的守门狗（每 2×10⁷ 轮打印 `next/last/ready0`）→ **PASS 且无任何 STUCK
    打印**（trap 变体无法区分，因 scheduler kernel 仍自旋、`synchronize()` 仍不返回）。
  - **[与 §7 的关系]** §7.5 的修复（`_dbg_spins`+`printf`+阈值 2×10⁸）收敛，而同语义的 `counter`
    变体（`_dbg_spins`+`__trap`+阈值 2×10⁷）HANG——**同一处逻辑的两种写法给出相反结果**，说明它是
    **对 worker kernel codegen/调度敏感的时序扰动式偶然修复**，不能升级为「竞态在 worker 等 task」。
  - **[机制候选，假设]** `worker_kernel`/`scheduler_kernel` 在两条每次调用新建的 `cudaStreamNonBlocking`
    流上 launch（`persistent_kernel.cuh:1932-1935`），与 torch 默认流仅靠 `prepare_kernel` + 
    `prepare_done_event` + `cudaStreamWaitEvent` 排序（`:1963-1969`、`:1981-1986`）；最可能是
    **launch/握手时序**（一个紧邻的 dtype 转换 kernel 改变了驱动把这几条 launch 下发/共驻的次序）。
    **不构成对上游的 bug 判定**。P25-1 仍记 `[~]`。
  - **下一步候选**：nsys 抓 host 侧 launch/流时序；把前置 cast 放到独立非默认流验证「设备全局态 vs
    默认流历史」；launcher 在 `prepare_kernel` 后显式同步或让常驻 kernel 继承当前流；向 MPK 上游核对。
- **本轮（P25-1 续 2）**：续做 P25-1 的「端到端延迟」。在**收敛构建**（把 §7.5 的最小修复固化成
  `harness/mpk_latency_run.patch`：`execute_worker` 取 task 轮询循环加 `MPK_DBG_WATCHDOG` 门控的
  自旋计数、编译命令加 `-DMPK_DBG_WATCHDOG`）上，用新脚本 `harness/mpk_latency.py` 测
  「一次 megakernel 调用」从 host 提交到双常驻 kernel 终止的完整往返：`batch=16, hidden=4096` bf16、
  `CUDA_VISIBLE_DEVICES=7`、快照 `f9eb70c`，每次 `pk()` 前后记一对 CUDA event，20 warmup + 200 timed
  iters。核心结论：
  - **端到端延迟 36.4–39.1 µs/次调用**，扫 128/16（= 论文 §5.1）、64/16、16/16 三档：
    mean 39.05 / 37.59 / 36.40 µs、median 38.99 / 37.47 / 36.38、sd 2.14 / 1.27 / 1.34；
    正确性三档 `max|out-ref| = 0.03125`（bf16 容差内）。
  - **对照 PyTorch 朴素 kernel-per-operator RMSNorm 54.0 µs**（同 batch/hidden，同一计时循环）：
    MPK 约 **1.4–1.5×**；本用例只 1 个算子、张量极小，故差距是 launch（C1 单次提交）+ 消除中间量
    的固定收益量级，对上 `docs/04` §1/§17 的损失二/三。
  - **延迟几乎与 worker 数无关**（132→32 block，只降 ~7%）：主导项是「双常驻 kernel 的启动 +
    设备内 scheduler 冷启动 + done-event 终止握手」，不是计算/并行度。
  - **重复稳定性**：128/16 连跑 3 次 **3/3 收敛 PASSED**，200-iter 均值 37.64 / 37.78 / 38.08 µs
    （抖动 <1.5%）。**plain 对照**（仅删 `-DMPK_DBG_WATCHDOG`）复现挂起 `EXIT=124`。
  - 原始输出 `harness/mpk_latency.out.txt`（config 扫描）/ `harness/mpk_latency_repeat.out.txt`
    （重复 + plain 对照）；`docs/09-mpk-build.md` 新增 **§8**（含 §8.3 读法/§8.4 局限）。
  - **局限**：单层微内核、单次调用（非论文 §6 整模型端到端）、延迟来自「打过时序修复的构建」、
    batch/hidden 未扫。P25-1 **端到端延迟已补齐**，仍因根因内存序未钉死而记 `[~]`。
- **本轮（P25-1）**：新建 `docs/09-mpk-build.md`（MPK 线上实现的**构建与端到端冒烟**，与 `docs/06`/`07`/`08`
  同一快照 `f9eb70c`）+ 博客 post `mega-kernel-10-mpk-build`（weight 11）并更新 `mega-kernel-00-index`
  目录第 10 项；脚本 `harness/mpk_smoke.py`，原始输出 `harness/mpk_build_run.out.txt`（构建）与
  `harness/mpk_smoke.out.txt`（运行）。把 `docs/06` §0「构建成本过高、未做」的尾巴推进到实机：
  - **主机侧全链路成功**：`git submodule update`（cutlass/json/z3，github 需重试）→ `pip install -e .`
    （`mirage_runtime` 原生库 + cython 扩展 + Rust `abstract_subexpr`/`formal_verifier`）→
    `import mirage` 0.2.4；用 `PersistentKernel`（`persistent_kernel.py:669` 默认参数、`test_mode=True`）
    搭单层 RMSNorm 最小 megakernel，生成 **18 task / 3 event**（`runtime_header.h:229-235`：
    `EVENT_TERMINATION`=911×1 / `EVENT_LAUNCH_DEPENDENT_TASKS`=903×1 / `EVENT_END_OF_TASK_GRAPH`=910×16），
    nvcc JIT 编译 **sm_90a 成功（~27 s/次）**，`worker kernel & scheduler kernel` 双 stream launch 成功
    （`persistent_kernel.cuh:1977`，`split_worker_scheduler` 恒 true `:1795`）。
  - **设备侧未收敛**：`pk()` 后 `torch.cuda.synchronize()` 永不返回、GPU 100% 自旋；**4 worker/4 scheduler
    也复现**（排除 SM 占用不足）。**本轮无法给出端到端延迟/正确性数字**，P25-1 记为 `[~]`，并新增
    「阻塞」B1/B2。
  - 关键坑：运行期依赖 `tg4perfetto@git` 直连 github 超时 → 改用 PyPI 同名包 + `--no-deps
    --no-build-isolation`；mirage 钉 `z3-solver==4.16.0.0` 与容器内 `tilelang`（`<4.15.5`）互斥 →
    跑完已恢复 `4.15.4.0`（B2）。假设（**待验证**）H1 iteration/事件计数错位、H2 sm90
    `MPK_ENABLE_TMA`/`GRACE_HOPPER` task 单体内同步、H3 快照 offline/test 路径不完整，下轮见 §「下一步」。
- **本轮（P24-1）**：`docs/04-experiments.md` 新增 §17「损失三的直接字节证据：用 ncu 量化『融合消掉
  多少 HBM 流量』」——新建 `src/dram_traffic.cu` + `harness/dram_traffic.out.txt`（计时）与
  `harness/dram_traffic_ncu.out.txt`（ncu 字节 + 原始 CSV）（目标卡 H100/sm90，固定空闲 device 3，
  iters=30，N=64Mi elem=256 MB/数组 ≫ 50 MB L2，Q=1/2/4/8 逐元素算子链 `y_q=f(y_{q-1})`）。
  四策略公平对照 **eagerQ**（Q 次 launch，中间量落 HBM）/ **graphQ**（capture 成图）/ **barrier**
  （单 kernel + 逐算子 grid.sync，中间量仍落 HBM）/ **fused**（单 kernel，中间量只在寄存器）。
  用 `ncu -k regex:dt_ --metrics dram__bytes_read.sum,dram__bytes_write.sum` 直接数 DRAM 物理字节。
  核心结论：(1) **fused 流量常量** `= read 268.4 + write 250.0 ≈ 518.5 MB = 8 B/元素`，与 Q 无关；
  (2) **eager/graph 流量 ≈ Q × 517 MB**，`total/fused` = **1.99/3.98/7.96 = 精确等于 Q**（Q=2/4/8）——
  每融合一个算子消掉一趟完整全张量 HBM 往返（读 256+写 256 MB）；(3) **barrier ≈ eager 字节**
  （仅写多 2–4%），证 `grid.sync()` 只省 launch、不省流量（损失二/三正交）；(4) **时间 ∝ 字节**——
  两档表观带宽同 ~2.75 TB/s，fused 相对 eager 的加速（时间 4.0×/8.0×）**完全等于其省下的字节比**
  （4.11×/8.25×），把 P5-2 的「表观带宽×假设字节」**算术推断升级为 ncu 字节实测**。计时表另证
  `e/g≈1.0`、`b/e≈1.0`（256 MB 已带宽主导，损失二≈0，对上 P5-1/P5-2）。判据：`fused` C1–C3、
  `barrier` +C4，均无 C5。正确性四档 `max|z_eager−z_fused|=0`。**本轮新增实验**；与 P9-1（时间标度）
  互为表里，补齐 P5-2 §2.6 未直接量 HBM 字节的缺口。
- **本轮（P23-1）**：`docs/04-experiments.md` 新增 §16「跨层软件流水 + 环形缓冲预取：kernel-per-operator
  与持久化内核的真正分野」——新建 `src/cross_layer_pipeline.cu` + `harness/cross_layer_pipeline.out.txt`
  （E=16 主运行）与 `harness/cross_layer_pipeline_e32.out.txt`（E=32 重载敏感性）（目标卡 H100/sm90，
  固定空闲 device 3，iters=30，L=64，G=132=SM，T=256，tile 16KB/32KB，权重 528MB/1056MB ≫ L2）。
  把 MPK §5.3（快照行 1312–1368）的 paged shared memory + cross-task software pipelining、即
  P22-1 §15.6 明留的「跨层流水」尾巴实测出来：层 l、worker j 的 task = 私有权重 tile（cp.async 载入、
  与 DAG 无关可预取）+ 跨层激活值（读 worker (j+1)%G 在层 l-1 写的槽，必须等）+ E 链 ×R 轮 FMA 归约。
  四策略公平对照 **eager**（L 次 launch，kernel-per-operator）/ **barrier**（单 kernel + 每层 grid.sync）/
  **decen**（单 kernel + 逐槽事件计数器，无预取）/ **ring**（decen + K 页环形 SMEM 跨 task 预取）。
  三 Section：A 扫 R（0→256）、B 扫环形深度 K（R=16）、C 扫层数 L（8→256）。核心结论：
  (1) **四策略对 act[L][*] 逐位一致**（互比 max rel|Δ|=0.00e+00）；(2) **单内核相对 kernel-per-operator
  的主收益仍是消 launch**——eager/barrier 随 L 8→256 由 1.14× 升到 1.25×（对上 P5-1/P9-1 损失二）；
  (3) **细粒度事件调度 decen 稳定优于每层 grid.sync 的 barrier 约 3–14%**（d/b 0.856–0.972×，访存越重
  优势越大），解释 P22-1「barrier 恒快」为何只在每层 W≥256 的图成立（本节 W=G=132）；
  (4) **跨层预取 ring 相对 barrier 峰值快 1.19–1.20×**（E=16 R=8 r/b=0.844×、E=32 R=8 r/b=0.834×），
  收益 ≈ 访存时间占比、随计算强度 R 衰减（R=256 仅 1.03×），落在 MPK §6.6（行 1593）的 1.2–1.3× 区间；
  (5) **纯访存（R=0）时流水净亏**（r/b 1.036–1.376×，无计算可藏）；(6) **环形深度 K 的收益取决于 task
  访存占比**——轻载 E=16 时 K=1→8 的 r/d 恒 0.933–0.941×（前驱事件等待已够藏 16KB），重载 E=32 时
  K=1→4 单调改善（0.970→0.919×）、K≈4 饱和。判据：ring 满足 C1–C4，按静态序列取 task、无固定角色
  worker/scheduler，仍差 C5 一步（C5 由 P15-1 §9 覆盖）。**本轮新增实验**，补齐 P22-1 §15.6，并与
  P13-1（单 task 内流水）/P20-1（跨算子静态分块重叠）合成「软件流水三层」画像。
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

- **B1（P25-1）MPK 端到端执行不收敛**：在目标卡 H100/sm90 上 mirage（`f9eb70c`）**能构建、
  能 JIT 编译、能 launch**，但常驻 kernel 的 worker/scheduler 事件循环**不终止**（`pk()` 后
  `torch.cuda.synchronize()` 永不返回、GPU 100% 自旋）；4 worker/4 scheduler 亦复现，排除
  SM 占用不足。详见 `docs/09-mpk-build.md` §3–§4 与 `harness/mpk_smoke.out.txt`。
  **本轮（P25-1 续）已诊断到根因层面（`docs/09` §7）**：官方 test_mode 用例 plain **3/3 挂起**，
  在 `execute_worker` 取 task 轮询循环（`persistent_kernel.cuh:1013`）加一处**无害的自旋计数**
  即 **3/3 收敛且正确**（`max|out-ref|≤0.03125`）；消融显示决定性变量就是这个 wheel loop 的时序扰动，
  scheduler 侧逐事件 printf 不足以修复。**判定：这是 worker↔scheduler 的时序敏感同步竞态（丢失唤醒/
  抢跑），非 task 单体内 TMA/mbarrier 卡住（H2 排除），也非事件计数"凑不满"（H1 该形态排除）**。
   **本轮（P25-1 续 2）已补齐端到端延迟**（`docs/09` §8）：收敛构建上单层 RMSNorm megakernel
   36.4–39.1 µs/次调用（对照 PyTorch 朴素 54.0 µs），128/16 连跑 3/3 收敛、plain 复现挂起
    （`EXIT=124`）。**本轮（P25-1 续 3）把根因从「worker 轮询循环内存序」再定位到
   「launch/握手层面的时序敏感竞态」**（`docs/09` §9，device 7、替换上轮 device 0 的作废 sweep）：
   循环内**内存序类**改动（`asm memory` 屏障 / `__threadfence` / `ld.relaxed` / 改 `__nanosleep` 值）
   **全部不修复**；只有**前置的 dtype 转换 kernel**（`x.to(fp32/fp16/fp64)`、`copy_` 跨 dtype）能避开，
   而同 dtype 的 `sync`/`sleep`/`copy`/`mul`/`matmul`/`alloc`/`to_same` 都不能；grid `4:4`–`128:16`
   四档全挂。故 §7.5 的「自旋计数修复」是**对 worker kernel codegen 敏感的时序扰动式偶然修复**，
   不能作为「竞态在 worker 等 task」的证据。仍未做：nsys 级别的 host 侧 launch/流时序证据、
   非 Hopper-TMA task 隔离 H2。patch：`harness/mpk_watchdog.patch`（全量插桩）／
   `harness/mpk_workaround_only.patch`（最小修复）／`harness/mpk_latency_run.patch`（延迟实验用的
   收敛修复 + 编译宏）；新证据 `harness/mpk_race_attr.out.txt`、`harness/mpk_race_sweep.out.txt`、
    `harness/mpk_wlwatch.out.txt`。
   **本轮（P25-1 续 4）把根因再定位到「设备侧双向死锁」**（`docs/09` §10）：§9 的「默认流历史」「host
   launch 排序」两条候选**均被证伪**——前置 cast 放到独立非默认流/完全不 sync/只 1 元素仍能避开
   （§10.1），只有 dtype 转换这一类 kernel 有效而同 dtype `mul/copy/clamp`、cuBLAS、`_sleep` 独立流均无效
   （§10.2），`prepare_kernel` 后插 `cudaStreamSynchronize`/`cudaDeviceSynchronize` 仍挂（§10.3）；
   host 侧 `cudaStreamQuery` 轮询显示 HANG 时 worker/scheduler **两条 stream 都持续 `device not ready`**
   （PASS 时 it=1 即 DONE），`4:4/16:16/64:16` 三档皆然 → 排除 co-residency，是 worker↔scheduler
   **双向自旋死锁**（§10.4）；锁频 `-lgc 1980` 不改变结果（§10.5）；nsys 因挂起用例 CUPTI 未 flush +
   挂起 kernel 上不收敛而失败（§10.7）。机制候选见 §10.6（H-a 设备级首次 dtype 转换懒初始化 / H-c
   快照 `MODE_OFFLINE`+`test_mode` 终止协议时序依赖），**仍不构成对上游 `f9eb70c` 的 bug 判定**。
   新证据 `harness/mpk_race_stream{,2,3}.out.txt`、`harness/mpk_launch_sync.out.txt`、
   `harness/mpk_wait_trace{,_grid}.out.txt`、`harness/mpk_clock_lock.out.txt`、`harness/mpk_nsys.out.txt`；
   新 patch `harness/mpk_launch_sync.patch`。
    **本轮（P25-1 续 5）已解决（`docs/09` §11）**：用 **cuda-gdb attach**（非侵入式）拿到设备侧证据——
    挂起时设备上**只有 `worker_kernel` 常驻**（PC 在 `persistent_kernel.cuh:1016` 取 task 的 `ld_acquire`），
    `scheduler_kernel` **从未被派发**（`cuda kernel 1` → `Invalid coordinates`；4:4 下 128 SM 空闲亦然），
    修正 §10.4「两 kernel 同时自旋」；再用**只交换 launch 顺序**的消融 `harness/mpk_sched_first.patch`
    钉死根因：**scheduler 先于 worker launch → plain 构建 4/4 收敛且正确**（默认顺序同会话复现 HANG）。
    clean（scheduler-first）构建延迟 39.1/43.5 µs（同 §8 量级），故 §8 的延迟亦非内插桩产物。
    **B1 解除**（根因 = launch 顺序，绕过 = scheduler-first；不构成上游 bug 判定）。
    **不阻塞本专题其余任务**（纯资料分析均已完成；P0–P25 全部完成）。
- **B2（环境）z3-solver 版本互斥**：mirage 需 `z3-solver==4.16.0.0`，容器内 `tilelang` 需
  `<4.15.5`。本轮跑完 mirage 后已恢复 `4.15.4.0`，因此当前 `import mirage` 会失败；下次使用
  mirage 前需重钉 `4.16.0.0`（用完切回），避免影响其它并行专题。
