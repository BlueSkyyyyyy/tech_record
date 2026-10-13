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

### P37 Device Graph Launch 作为「in-kernel 分发」（`docs/04-experiments.md` §26）

- [x] **P37-1** 补齐 `docs/04` §1.6（P4-1）留下的判词「**device graph launch 仍停在 kernel 级**」：
      把「设备自己发图」这条**最直接的设备侧派发**路线量出来，并与 P36-1（§25）的 `kinternal`
      （in-kernel 循环）/`cwhile`（条件节点重分发）对照。见 `docs/04-experiments.md` §26，源码
      `src/device_graph.cu`、脚本 `harness/device_graph.sh`、原始输出 `harness/device_graph.out.txt`
      （主配置 N=20000×reps=5）与 `harness/device_graph_sens.out.txt`（敏感性 N=2000×reps=20）
      （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`-arch=sm_90a -rdc=true -lcudadevrt`）。
      §26.2 头文件核对：`cudaGraphInstantiateFlagDeviceLaunch`=4（`driver_types.h:3910`）、
      `cudaGraphUpload`（`cuda_runtime_api.h:12940`）、device 侧 `cudaGraphLaunch`
      （`cuda_device_runtime_api.h:290`）+ 三个特殊流句柄 `cudaStreamGraph{TailLaunch,FireAndForget,
      FireAndForgetAsSibling}`（`:175-177`）+ 自引用 `cudaGetCurrentGraphExec()`（`:311`，读
      `%current_graph_exec`）。**结论**：
      (A) **两条实测约束**——把调用 `cudaGraphLaunch` 的 kernel 直接 `<<<>>>` 启动 → **`operation not
      supported`**（发起者必须是 graph 节点）；host-graph 节点内对**同一 exec 连发两次** → 第 1 次
      ok、第 2 次 **`invalid argument`**（**单实例在飞**，故循环只能走**自重启图**）。
      (B) **设备侧发图不比 host 便宜**：每单元 `eager`（host 发 kernel）2078.5 / `hgraph`（host 发
      1-node 图）2370.7（1.14×）/ **`srelaunch`（设备自重启图）4285.4（2.06×）** / `kinternal`（常驻
      内核内循环）**39.3 ns（0.019×）**；敏感性与逐位 `sink` 全对。
      (C) **单次 device 图启动往返 4128 ns ≈ `srelaunch` 每行程 4.29 µs**——成本就是「一次设备图启动」
      本身，与自重启体无关。
      **含义**：device graph launch / 条件节点重分发（§25 `cwhile` 4.6 µs）/ 自重启图都落在**同一个
      ~4 µs 的「设备侧 kernel/图重分发」地板**，而 in-kernel 边界只要 ~40 ns；**设备能发图 ≠ 设备发图
      便宜**，要把单价从 ~4 µs 压到 ~40 ns 只能**留在同一个常驻 kernel 内**（MPK in-kernel event queue /
      DeepGEMM ring barrier）。判据：四策略均不满足 C1–C5（`srelaunch` 由设备发起但粒度仍是整个图）。
      **本轮新增实验**；与 §24/§25 合成「CUDA Graph 降级路线天花板」的第三块（设备侧派发）。


### P38 实验扩展 20：多租户共存与 SM 分区（`docs/04-experiments.md` §27）

- [x] **P38-1** 把「megakernel = 单次 launch 占满整机」这一被前面所有小节默认接受的前提**翻到成本面**：
  整机持久化内核对**共存负载**（另一模型/租户/profiler/通信）的干扰，以及 Hopper **Green Contexts**
  SM 分区能否硬隔离。补齐 MPK §5「worker/scheduler 分配固定在 launch 时、匹配物理 SM 数」（快照行 1168）
  的**部署含义**，并把 P10-1（占用率上限）/P14-1（占用率代价）从「单 kernel 内」推进到「跨 kernel」维度。
  见 `docs/04-experiments.md` §27，源码 `src/green_ctx_sched.cu`、脚本 `harness/green_ctx_sched.sh`、
  原始输出 `harness/green_ctx_sched.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  buf=256 MB、rounds=64，device 1 复现同量级）。被测 = 访存为主的整机 `persist_kernel`（网格按占用率
  上限铺满）与普通 `victim_kernel`（多 block 求和，末 block 写 mapped-pinned `done`）。四 Section：
  - **A 干扰**：persist 铺满 132 SM 但每 SM 仅 1/8 槽位（Gp=132）→ victim **1.0× 基线**（可共驻）；
    Gp=1056（8 blk/SM 占满整机）→ victim 被饿死到 **134.5× 基线**（5.613 ms ≈ persist 自身 5.59 ms）
    ——**干扰的开关是「剩余占用率槽位」而非「占了几张 SM」**（对上 P10-1/P14-1）。
  - **B Green Contexts 分区**：`cudaDevSmResourceSplit` 把 132 SM 切成 A=64/B=68（B 组用 backfill 拿余数）
    + `cudaGreenCtxCreate`。B1 persist 占满 A、victim 在 B → **0.76× 基线**（≈ 无干扰），且 persist 自身
    **7.56 ms 与 D 段 smA=64 单独跑 7.55 ms 一致** → **隔离双向**；B2 对照（victim 放 A 的同分区另一流）
    → 又饿死到 **184.3× 基线**——**隔离来自分区本身、不是「换个 context」**。
  - **C 流优先级**：persist 低优 / victim 高优 → victim 仍 **133.65× 基线**——**优先级不抢占已运行的
    block**，不能替代分区。
  - **D 分区代价**：persist 吞吐随分到的 SM 近似线性——32→1335、64→2277、96→2821 GB/s（96 SM 近整机
    HBM 上限后饱和）。**给共存租户留 SM = 按比例削减 megakernel 可达带宽**。
  判据：三者均为单 kernel、`persist` 满足 C1/C2；Green Contexts 是 **C4 的更粗一层**（launch 前定死
  「谁用哪些 SM」，与 P29-1 cluster 的片内分区同目标），**均不涉 C3/C5**。踩坑：H100 的 132 SM 与
  `smCoscheduledAlignment=8` 不整除，第二组必须 `cudaDevSmResourceGroupBackfill`；`smA=128` 余 4 SM
  小于最小分组而 split 失败。**本轮新增实验。**

### P39 实验扩展 21：MPS 作为第二条共存隔离路线（`docs/04-experiments.md` §28）

- [x] **P39-1** 补齐 `docs/04` §27.6 明确留的「**未测 MPS**（Multi-Process Service）作为另一条
      隔离路线」尾巴，并把 §27 的**进程内**（两条流 / green context）共存推进到**跨进程**维度：
      没有 MPS 时不同进程的 kernel **不能真正并发**（驱动时分/抢占）；MPS 才允许跨进程真并发，
      并可用 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P` 给每个 client 设「线程容量 / 可见 SM 上限」。
      见 `docs/04-experiments.md` §28，源码 `src/mps_isolation.cu`、脚本 `harness/mps_isolation.sh`、
      原始输出 `harness/mps_isolation.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
      buf=256 MB、victim=20 趟求和、persist=40000 rounds）。两个**独立进程**（`persist` 整机访存
      持久化 kernel / `victim` 普通求和 kernel、CUDA event 计时含被饿死等待）对照：
      - **无 MPS**：victim **1.792 ms ≈ 基线 1.781 ms（1.006×）**，persist 只慢 1.4%——跨进程 kernel
        **不共驻**、由驱动时分/抢占，victim 每轮拿整机时间片，代价转嫁到 persist。**与 §27 进程内
        两条流的 134× 饿死完全相反**（同卡「同进程并发」比「跨进程无 MPS」对邻居更糟）。
      - **MPS 默认（各 100%，真并发）**：victim **2.780 ms = 1.55× 基线**，persist 带宽几乎不变
        （3034 GB/s）。
      - **`ACTIVE_THREAD_PERCENTAGE=P` 是硬线程容量上限**：它把该进程可见 `multiProcessorCount`
        按 P 缩放（100→132、90→118、75→98、50→66、25→32、10→13 SM），persist 网格随之下调。
      - **反直觉：干扰峰值在「cap 略低于 100」**——victim slowdown 对 **P≥50 精确服从 `≈1/(1−P)`**：
        P=90→**9.90×**（1/0.10=10）、P=75→3.98×、P=60→2.80×、P=50→2.01×；**给 persistent 留 10%
        容量就把 victim 饿死 ~10×，比不设 cap（1.55×）还糟 ~6 倍**；cap 越低越恢复（P=10→1.47×）。
      - persist 自身带宽随 cap **次线性**（3035→509 GB/s，P=100→10）——「分区账」是带宽降幅 < 线程降幅。
      - **三路线对比**：Green Contexts（§27，同进程 SM 张数硬分区）victim **0.76× 基线**、persist
        无损、双向且**确定**；MPS（跨进程线程容量软限）**最坏 9.9×** 且非确定；无 MPS（跨进程时分）
        victim ≈ 基线但**非并发**。**要确定性公平隔离用 Green Contexts/独占；MPS 只保证「能并发」，
        不保证公平**——给整机 megakernel 做 MPS 隔离时 cap 应设得**明显低**（P≤25）。
      判据：`persist` 单 kernel 满足 C1/C2；MPS 是**进程/驱动层**约束、比 Green Contexts 更外一层
      （C4 的更粗粒度容器），均不涉 C3/C5。踩坑：**MPS client 若设 `CUDA_VISIBLE_DEVICES` 会连不上
      server**（`no CUDA-capable device`），须 server 设 CVD、client 不设；server 用独立 pipe 目录、
      实验后 `echo quit` 清理，仅影响 device 3。**本轮新增实验。**

### P40 实验扩展 22：L2 常驻作为损失三的第三档落点（`docs/04-experiments.md` §29）

- [x] **P40-1** 兑现 `docs/04` §17.7（P24-1）明确留的假设「若张量能整块驻留 L2（≤50 MB），eager 的
      中间量往返可能命中 L2 而不落 DRAM——损失三在工作集超过 L2 时才充分显现」，并把「中间量的内存落点」
      从已有的**寄存器**（P5-2 融合）/ **SMEM+DSM**（P29-1）/ **HBM**（P24-1）补上缺失的 **L2** 一档。
      见 `docs/04-experiments.md` §29，源码 `src/l2_residency.cu`、脚本 `harness/l2_residency.sh`、
      原始输出 `harness/l2_residency.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
      iters=30，X=Z=256 MB 流式、polluter=512 MB、中间量 `y` 尺寸 S=4..64 MB，H100 L2=50 MB /
      maxPersistingL2=31 MB / maxWindow=128 MB）。序列 = `K1(读 x→写 y) → polluter(512MB 冲刷) → K2(读 y→写 z)`，
      四模式公平对照 `base`（不插 polluter，y 靠最近性）/ `pollute`（插 polluter，y 落 HBM）/
      `persist`（`cudaAccessPolicyWindow` persisting，carveout=min(S,31MB)）/ `pfull`（carveout=max 31MB）。
      计时 K2-only + ncu `dram__bytes_read.sum`/`lts__t_sector_hit_rate.pct`。核心结论：
      - **Q1 中间量在不在 L2 由「写读之间流过什么」决定，不是尺寸单独决定**：`base` 在 S≤16 MB 时 K2
        几乎全命中 L2（DRAM 读 6.7 KB–1.7 MB、命中 90–96%），一插 512 MB polluter 立刻涨到**恰好 = S**、
        命中 52%（y 被逐出）。
      - **Q2 `persisting` 能保护但只部分且零和**：`persist` 把 K2 的 DRAM 读相对 `pollute` 降
        **~23%（S=4）–30%（S=8/16）**（命中抬到 62%）；但 S≥32 MB 一旦超过 31 MB carveout 就**反噬**
        （计时 1.47×→**2.20×**），而 `pfull`（carveout 拉满）即便 S 小也稳定慢 **~1.5×**——
        **persisting 是拿普通 L2 换常驻 L2 的零和交易，应按中间量大小配、不该拉满**。
      - `base` 的「免费 L2」有硬边界：S=32 MB 起退化（24.8/32 MB 从 DRAM）、S≥48 MB 等价 `pollute`。
      判据：四模式都是 kernel-per-operator（K1/K2 两次 launch），谈的是中间量落点、不改 C 判据；
      `base`/`persist` 的 `z` 逐位一致（max|Δ|=0）。**含义**：megakernel / MPK 让整图很多张量同时流经 L2
      （= 「polluter 恒存在」），故「指望中间量留在 L2」是整图融合里**最不可依赖**的一档（31 MB 硬上限 +
      与 normal 流量零和）——从内存层级解释了 MPK §5.3 为何用 **paged shared memory** 而非 L2、最稳的仍是
      融合进寄存器（P5-2）与锁进 SMEM（P29-1）。踩坑：ncu 默认 `--cache-control=all` 会在 replay 前 flush L2，
      首版三模式字节完全相同；须 `--replay-mode application --cache-control none`。**本轮新增实验。**

### P41 实验扩展 23：L2 persisting 的精细化调参 + 多窗口/多流共存（`docs/04-experiments.md` §30）

- [x] **P41-1** 补齐 `docs/04` §29.6（P40-1）明确留的「只测单窗口、单 stream、单一 persisting 段，
  未测多窗口、多 stream 各自窗口、`hitRatio<1` 的概率化分区」尾巴，把「L2 常驻」从「能不能用」推进到
  「**怎么配才不亏**」，直接对应 megakernel「整图很多中间量同时活着、争同一个 carveout」的处境。
  见 `docs/04-experiments.md` §30，源码 `src/l2_persist_tuning.cu`、脚本 `harness/l2_persist_tuning.sh`、
  原始输出 `harness/l2_persist_tuning.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  `iters=30`，X=Z=256 MB 流式、polluter=512 MB、每段 8–16 MB，L2=50 MB / maxPersist=31 MB）。
  三个新问题：Q1 固定 carveout 扫 `hitRatio` 的保护/时间标度；Q2 `pfull`（carveout 拉满）的 ~1.5× 代价
  能否用小 carveout + `hitRatio<1` 收回；Q3 两个并发中间量各挂窗口共享一个 carveout 能否叠加。核心结论：
  - **零和代价的开关是 carveout 大小而非 hitRatio**：`f=0.00`（挂窗但全 Streaming = 只划 carveout 不保护）
    随 carveout 单调恶化 **4 MB→0.996、8 MB→0.981、16 MB→1.204、31 MB→1.921**；`pfull` 的 ~1.5× 惩罚
    **无法用 hitRatio 收回**（31 MB 行 f=0→1.921 仅降到 f=1→1.504）。
  - **正确姿势：carveout ≈ 中间量实际大小、配得上时 f 取高**——最优 `carveout=16, f=1.0 → 0.860×`；
    单窗口 ncu 亦证 `carveout=8`（=S）比 16/31 保护更好（S=8：6.05 vs 6.42/6.96 MB vs 无窗 8.40）。
  - **`hitRatio<1` 的唯一价值场景**：carveout 被压到 < 需求时救火——两个 8 MB 窗口、**carveout=8 < 总需求
    16、f=1.0 → 1.026×（thrash 崩）**，把 **f 降到 0.5 → 0.879×（反超）**。
  - **多窗口保护可叠加但要求 carveout ≥ 同活中间量总量**：`carve=16 both f=1.0 → 0.814×`（两个 K2 都快），
    而 `carve=31 both f=1.0 → 1.343×`（过度预留盖过保护）。
  - **保护上限仍 ~30%**（单窗口 f=1 削 DRAM 读 ~30%，与 §29 一致）；persisting 是提示非承诺。
  - **口径 caveat**：多 kernel + 多流下 ncu replay 的逐 kernel DRAM 读与自然计时背离（ncu 显示只有 stA
    侧受保护、但自然计时两个 K2 都加速、`oneB` 单 stream 又证明 stB 完全能保护），故**多窗口 split 归因
    以自然计时为准、ncu 逐 kernel 拆分只作参考**。判据不变（kernel-per-operator，谈落点）。**本轮新增实验。**

### P42 实验扩展 24：有界 ready 池 + warmup wave 的防死锁与容量下界（`docs/04-experiments.md` §31）

- [x] **P42-1** 把 checklist（`docs/05` §2.3「防死锁：交错调度下上游可能饿死下游——DeepGEMM 用 warmup
  wave + ring 容量上界闭式推导」）里**只作资料转引**的一条做成实测：megakernel 的 in-kernel scheduler
  用**有界** ready 池时，「**准入顺序**」与「**池容量**」如何共同决定**死锁**与**吞吐**。
  见 `docs/04-experiments.md` §31，源码 `src/ring_deadlock.cu`，脚本 `harness/ring_deadlock.sh`，
  原始输出 `harness/ring_deadlock.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  `work_iters=20000`）。一手核对 `~/github/DeepGEMM`（`057ca59`）`scheduler/mega_moe.cuh`：
  `get_num_l1_warmup_waves()`（`:15-45`，注释 *"ensure no L1 -> L2 deadlock"*）、
  `get_num_max_live_pool_blocks()`（`:47-80`，"Host-side ring capacity helper"，闭式上界）、
  `l2_full_mask` 奇偶 generation 复用依赖 ring 容量（`:140-141,153-164`）、运行期 `get_next_task()`
  在返回 L2 task 前**等所有需要的 L1 task 已被取走**（`:398-403`）、`fetch_expert_recv_count()`
  取 `num_sched_l1_waves=min(min_l1_warmup_waves,num_total_l1_waves)`（`:302-306`）。模型：单常驻 kernel
  （block 0 = scheduler，其余 G = worker），`M` 个 m-block × `Kp` producer（L1）+ `Kc` consumer（L2，
  等本块全部 producer），有界池 `CAP` + 顺序 `naive`（每块 consumer 先）/`warmup(W)`（先发前 W 块 producer）。
  设备侧 `clock64` 无进度守门狗使死锁 case 也正常返回。核心结论：
  - **§A 死锁边界 = 首个 m-block 的 consumer 数 `Kc`**：`M=32,Kp=8,Kc=4` 时 naive 在 **`CAP≤4`** 死锁、
    `CAP≥5` 即 ok（池被「等前置的 consumer」堵死；挤进一个 producer 即脱困）；`warmup(W=M=32)` 任意
    `CAP=1..12` 全 ok。
  - **§B warmup 必须够宽**：`CAP=4=Kc` 下 `done=12·W`，`W<M=32` 都死锁、`W=M` 才全解（每救一个 block
    完整跑完它的 12 个 task）。
  - **§E 边界随 `Kc` 平移**：`M=16,Kp=8`，naive 死锁阈值 `Kc=2→CAP≤2`、`Kc=4→≤4`、`Kc=8→≤8`；
    同 `Kc` 下 `warmup(W=M=16)` 在 `CAP=2/4/8` 全脱困。
  - **§C 容量是吞吐硬下界**：即便不死锁，`CAP<G` 时 worker 饿着——`G=64` 扫 `CAP`，`CAP=2` 慢
    **27.6×**、`CAP=8` 6.9×、`CAP≥G=64` 饱和（0.334→0.326 ms）——正对应 `get_num_max_live_pool_blocks`
    的 "max live"。
  - **§D worker 标度**：`G=1→18.15 ms`、`G=128→0.207 ms`（相对 `G=16` 5.58×），`G≤32` 近线性、
    `G>32` 转次线性（任务数/协调成瓶颈）。
  判据：`warmup`/`naive` 均 **C1–C4**、worker 无固定角色，**仍差 C5**（与 P15-1/P22-1 同档）；是
  megakernel 的**正确性/容量成本侧**，与 P14-1/P16-1/P17-1 资源足迹互补。正确性 `rel=-1.84e-10`。
  **本轮新增实验。**

### P43 实验扩展 25：有界 ready 池 × 多级依赖 DAG（`docs/04-experiments.md` §32）

- [x] **P43-1** 补掉 P42-1 §31.6 明留的尾巴「未测**多级扇入扇出 + 有界池**的组合」：把 §31 的
  **单层 fan-in** 死锁模型推广到**分层依赖 DAG**（`L` 层 × 每层 `W` 个 task × 每 task 依赖上一层
  `FIN` 个前驱），在有界 ready 池上对照四种**准入策略**（`wide` 层降序 / `level` 严格拓扑层序 /
  `warmup(D)` 多级 warmup / `gated` 依赖门控）。见 `docs/04-experiments.md` **§32**，源码
  `src/layered_pool.cu`、脚本 `harness/layered_pool.sh`、原始输出 `harness/layered_pool.out.txt`
  （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`work_iters=20000`、设备侧 `clock64`
  无进度守门狗使死锁 case 也正常返回）。核心结论：
  - **死锁边界 = 「阻塞数」`B=(L-1)*W`**（§A2：资源分量各自富余时 `CAP*` 与 `G*` **都精确 = `B+1`**）；
    把 §31 单层的 `Kc` 推广成「完整依赖深度上累积的、先被准入却在等前驱的 task 数」（§E 证与扇入
    宽度 `FIN` **无关**，只由层序几何 `L,W` 定）。
  - **从池容量单约束升级为 `(CAP,G)` 联合约束**（§A3）：`L=8`（`B=112`）下必须 **`CAP>B` 且 `G>B`**
    同时成立才达；`G=#SM` 固定时**单靠加大 ring 容量救不了**，必须先把 `B` 压到 `<G`——这正是
    DeepGEMM 抢发 L1 warmup（`scheduler/mega_moe.cuh:15-45`）的第一性理由。
  - **warmup 深度须覆盖「完整依赖深度」**（§B）：`D` 每 warm 一层把剩余阻塞降 `W`，须 `D≈L-1` 才在
    `CAP=W` 全解（对照 §31 的「宽度须 `W=M`」）。
  - **两种零成本正解**（§C）：`level`（拓扑层序）与 `gated`（依赖门控）**任意 CAP/G 无死锁**；
    `level` 吞吐无损（最快），`gated` 的朴素扫描实现在大 CAP 下慢 **~2.2×**（`level/gated`→0.457）
    ——门控语义对应 MPK event 驱动入队（`docs/06` §3），真实实现（event→task 直接映射）不付扫描账。
  - **首次预测被实测推翻**：初版只算池容量（`B+1`），`L=6/8` 在 `G=64` 下实测**任何 CAP 都死锁**，
    才发现第二约束是 **worker 被阻塞 task 占满**——改测 §A2/§A3 后才定量。
  判据：四策略均单常驻 kernel、scheduler/worker 角色分离（C1–C4），worker 无固定角色、**仍差 C5**
  （与 P42-1/P15-1/P22-1 同档）。正确性 `rel=0.00e+00`。**本轮新增实验。**

### P44 实验扩展 26：不规则依赖 DAG × 有界 ready 池（`docs/04-experiments.md` §33）

- [x] **P44-1** 补掉 P43-1 §32.7 明留的尾巴「依赖模型是规则的分层 × 每层等宽 × 上一层扇入——**未测
  不规则/跨层依赖、多级扇入扇出混合**」：把 §32 的规则分层 DAG 推广到**不规则 DAG**（每层宽度不等 /
  跨层 skip 前向边 / 多级扇入 / **层号乱序**），并给出死锁边界的**普适定义**。见
  `docs/04-experiments.md` **§33**，源码 `src/irregular_pool.cu`、脚本 `harness/irregular_pool.sh`、
  原始输出 `harness/irregular_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  `work_iters=20000`、`G=64`）。核心结论：
  - **普适量 `B(π)`（贪心峰值池占用）**：沿准入 order 逐个准入、立刻排空任何「前驱全 done」的就绪
    task，池内「已准入未完成」的峰值即 `B(π)`；实测 **`CAP*=G*=B(π)+1`**。§31 的 `Kc`、§32 的
    `(L-1)*W` 都是它的特例。**关键修正**：`B` 不能简化成「第一个就绪 task 的位置」——源在前、后面再
    堆满阻塞的形态会漏算（§33.5 `scramble=3`）。
  - **§A 不规则宽度**：闭式 `(L-1)*W0+1` 恒为 49，对 grow/shrink/zigzag 给出错值；真边界随宽度变
    （`B`=54/42/32 → `CAP*`=55/43/33），`CAP*=G*` 逐行相等（再次印证 `(CAP,G)` 联合约束）。
  - **§B 跨层前向 skip / 多级扇入**：前驱只来自更浅层 → 「层号升序」仍是拓扑序，`level`/`topo` 在
    `CAP=1` 即安全；`wide` 的 `B(π)=N-W0` 与 `span` 无关。
  - **§C 层号乱序（headline）**：把名义层号旋转后「层号升序」**不再是拓扑序**——`B_level` 从 0 骤升到
    16/32/48、小池死锁；真拓扑序 `topo`（Kahn）恒 `B=0`。**「按名义层/op 名遍历」≠「拓扑线性化」**，
    对照 MPK 依赖分析器用 Kahn 定序（`docs/08` §5）。
  - **§D/§E**：`gated`（依赖门控，任意图无死锁）扫描代价为 `topo` 的 **1.65–2.51×**（朴素扫描的工程
    代价，MPK 用 event→task 直取不付）；warmup 的正确判据是「**剩余阻塞 `B'<G`**」而非「warm `D` 层」
    （乱序下任何 `D` 都救不了）。
  判据：四策略均单常驻 kernel、scheduler/worker 角色分离（C1–C4），**仍差 C5**（与 P42-1/P43-1 同档）；
  正确性 `rel=0.0e+00`。**本轮新增实验**；把 §31/§32 的防死锁理论收口到一般 DAG。

### P45 实验扩展 27：事件驱动就地入队（`docs/04-experiments.md` §34）

- [x] **P45-1** 补掉 P44-1 §33.8 明留的三条尾巴（`N≤70`、只验证 `G*` 与 `CAP*` 同行相等即「未独立扫
  `G`」、`gated` 是**朴素扫描**实现）：把 §33 的「依赖门控 `gated`」从**调度器扫描**改成 **MPK §5.1
  的真机制**——每个 task 一个**事件计数器**（未完成前驱数），由完成该前驱的 **worker 直接 push**
  「计数归零」的后继进 ready 队列，scheduler 不参与准入。见 `docs/04-experiments.md` **§34**，源码
  `src/event_driven_pool.cu`、脚本 `harness/event_driven_pool.sh`、原始输出
  `harness/event_driven_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
  `work_iters=20000`）。四机制公平对照 `raw`（按序不查就绪 = §33 `wide`/`level`）/
  `scan`（= §33 `gated`，就绪才准入）/ `topo`（预计算 Kahn 序）/ `event`（MPK §5.1 事件驱动）。核心结论：
  - **§A 三机制数值逐位一致**（8/8 case 含不规则/乱序 `rel=0.0e+00`）；`event` 的「完成即 push」不动数值。
  - **§B 死锁边界**：`raw` 复现 `B(π)+1`（`reg,scr=1` `B=16` → `CAP=1` 死锁、`CAP=17` 活）；
    `scan`/`topo` 恒 `B≡0`；**`event` 更是 `CAP=1` 且 `G=1` 即活**（ready 队列只装就绪 task）。
  - **§C 更大 `N`（补 §33.8 的 `N≤70`）**：`N` 64→4096，`scan/event` 1.3–3.0×（扫描随 `N` 变慢），
    `topo≈event`（`topo/event` 0.95–0.98×）——两者都不扫描。
  - **§D 独立扫 `G`（补 §33.8 的尾巴）**：不规则 `shrink,scr=1` 下 **`raw CAP* = raw G* = B+1`**
    （6/30/126，精扫 `N=122,B=29`：`G=28/29` 死锁、`G=30` 活 → 须 `G>B`）；**`event G*≡1`**（单 worker
    也活）——有界池死锁在事件驱动下**根本不存在**。
  - **§E 扇入 `FIN`**：`FIN 2→16` 抬高 `scan/event` 到 **2.4–3.5×**；`event` push 代价近持平。
  判据：四机制单常驻 kernel、scheduler/worker 角色分离（C1–C4），`event` 最接近 C5 但仍把 scheduler
  角色留在 block 0，**与 §31–§33 同档、仍差 C5 一步**。**本轮新增实验**；把 §31–§33 的防死锁理论
  从「调度器扫描/预排序」推进到「依赖事件天然触发」的 MPK 机制，并合上独立扫 `G` 与更大 `N`。

### P46 实验扩展 28：有界 ready 环形缓冲（`docs/04-experiments.md` §35）

- [x] **P46-1** 补掉 P45-1（§34）§34.9 明留的尾巴「`event` 的 ready 队列**无界**（按 `N+4` 开数组）、
  真实实现要**环形缓冲 + 容量上界**」：把 §34 的事件驱动调度装进一个**有界** ready 环形缓冲 +
  **阻塞式背压**，测「有界」会付出什么代价。真实锚点：MPK 固定 `per_worker_queue_len=8192` /
  `per_sched_queue_len=4096`（快照 `f9eb70c`，`persistent_kernel.cuh:1790-1791`）+ 不溢出 `assert`
  （`:1027`），注释 `:1782-1789` 明说容量是被**速率**逼上去的；DeepGEMM Mega MoE
  `get_num_max_live_pool_blocks()`（`scheduler/mega_moe.cuh:47-80`）闭式给容量上界 +
  `get_num_l1_warmup_waves()`（`:15-45`）防死锁。见 `docs/04-experiments.md` **§35**，源码
  `src/bounded_event_ring.cu`、脚本 `harness/bounded_event_ring.sh`、原始输出
  `harness/bounded_event_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
  `work_iters=20000`）。`unb`（=§34 无界）vs `bnd`（有界 `CAP` + 阻塞背压）。核心结论：
  - **§A/§B 两类容量下界别混用**：`unb` 恒可完成；`bnd` 需一个小容量，`G ≤ W_max` 时
    **`CAP_safe + G ≈ W_max + 1`**（`W_max`=最大层宽）——G 个 worker 在飞、环只缓冲宽层里
    `W_max−G` 个未被取走的就绪 task；`G ≥ W_max` 时触底 ~2。**与 §31–§34 的 `B(π)+1` 不同**：
    后者是**准入顺序/依赖深度**决定的图静态量，这里是**并发就绪数 ÷ worker 数**的排队量。
  - **§C 容量是安全阀不是吞吐旋钮**：`CAP<CAP_safe` 必死锁（守门狗），`CAP≥CAP_safe` 后延迟完全
    持平（0.375 ms，差 <1%）——区别于 §31 §C「容量是吞吐硬下界」（那里 worker 会占着未就绪 task 自旋）。
  - **§D 正确性**：`unb` 与 `bnd(CAP=W_max+1)` 8/8 case **逐位相同 `sum`**（`rel` 1.5e-9–2.4e-9）——
    有界/无界只改「何时入队」不改「算什么」。
  - **§E 规模标度**：`CAP_safe ≈ max(2, W_max − G + O(1))`，随 `W_max` 增、随 `G` 降；MPK/DeepGEMM
    取远超峰值就绪数的固定常量保守兜底，正兜住本节量出的「实际需求很小、但边界不确定」。
  - **headline caveat**：靠近安全边界时**单次可完成、多次可能死锁**（`CAP_hard < CAP_safe`），
    边界是「环满 + 无 worker 在 pop 相」的时序竞态。判据：`unb`/`bnd` 均单常驻 kernel、设备侧
    推进（C1–C4），仍差 C5 一步（与 §31–§34 同档）。**本轮新增实验。**

### P47 实验扩展 29：in-kernel 队列的发布内存序（`docs/04-experiments.md` §36）

- [x] **P47-1** 补掉 P46-1（§35）§35.7 明留的尾巴「入队是『先推 `tail`、后写数据』（**无 release 发布**），
  理论上消费者可能在数据可见前读到槽；真实实现（MPK）用 release/acquire 把它做实——可作后续实验」：
  把「发布内存序」从资料分析做成实测（此前本专题所有合成队列都默认 relaxe+侥幸）。真实锚点：MPK
  队列协议「`st.relaxed.gpu` 写数据 + `atom.add/.cas.release.gpu` 推游标 / `ld.acquire.gpu` 读游标 +
  `ld.relaxed.gpu` 读数据」（快照 `f9eb70c`，`mpk_atoms.cuh:17-96`；worker 入队 `:1215-1228`、
  scheduler 出队 `:1340`/`:1357-1360`、派发 `:1495-1502`；`docs/06` §5.3）；DeepGEMM ring barrier
  的 `red.async.inc.release` + 无符号距离 wait（`docs/04` §23 引 `scheduler/mega_gate.cuh:13-39`）。
  见 `docs/04-experiments.md` **§36**，源码 `src/pub_order_ring.cu`、脚本 `harness/pub_order_ring.sh`、
  原始输出 `harness/pub_order_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2）。
  三 Section：A message-passing litmus（store→store）relaxed vs release/acquire + store-buffering
  litmus（正对照）；B 单常驻 in-kernel 生产者→消费者环（payload `mix(s,i)` 逐字校验，五发布序
  `wro`/`rel`/`pub`/`half`/`pubslp`）；C 每槽 release/acquire 的代价。核心结论：
  - **§A sm90 序画像 = store→store 保序、store→load 可绕过**：MP litmus relaxed **0/70M**（store→store
    未重排，`wro` 的「写→推」正属此类）；正对照 SB litmus relaxed **100%（5M/5M×2）**（store→load
    重排完全可观测，**证明试纸有效**）。MP relaxed 1497.5 vs rel+acq 2226 ns/trial（一对 fence ~+728 ns）。
  - **§B 「发布先于写」在本机自然时序下 0 错、但拉开窗口 100% 错**：`pub`（= §35/§34 真实程序序）
    与 `half` 在 `PL=256`/`8192`、百万级槽上**全 0 错**（producer 的 payload store 早在消费者
    轮询→CAS→读 的往返内落盘、总是抢先）；而人为拉长窗口的 `pubslp` **20000/20000、2000/2000 全错**
    ——证明**检测器有效**，§34/§35「没出事」是**时序侥幸、非形式保证**（§35.7 判断正确）。正确程序序
    `wro`（无 fence）与 MPK 协议 `rel` **全 0 错**。
  - **§C release/acquire ≈ +0.36 µs/槽（几乎恒定）**：`wro`→`rel` 差 `PL=1,G=2 +372.1`（1.33×）、
    `PL=1,G=34 +355.1`（1.92×）、`PL=256,G=34 +362.1`（**1.06×**）——fence 是固定成本，大 payload
    摊薄后近免费。解释 MPK 为何「relaxed 写数据 + **单条** `atom.cas.release.gpu` 推游标」而非全程 fence
    （每队列操作只付一次 release+一次 acquire，靠批量取 task / 环形缓冲摊薄）。
  判据：litmus/环与「几个常驻 kernel」正交，是队列**内存正确性**地基；环满足 C1–C4、仍差 C5 一步
  （与 §31–§35 同档）。踩坑（已修，记入 §36.6）：① 背压误用 `head`（抢槽即自增）导致生产者覆盖消费者
  正在读的槽 → 假阳性 19995/20000；② SB litmus 在写读之间放了带原子的屏障、顺带 flush store 抹掉窗口
  → 测出 0。**本轮新增实验。**

### P48 实验扩展 30：MPSC 队列的发布协议（`docs/04-experiments.md` §37）

- [x] **P48-1** 补掉 P47-1（§36）§36.6 明留的尾巴「环仍是合成模型……未测多生产者（MPSC）」：
  把 §36 的 SPSC 队列推广到**多生产者单消费者（MPSC）**，并把「发布内存序」升级为「发布**协议**」——
  多生产者下「占位号乱序」逼出一个 SPSC 没有的新要求：**连续放行**。真实锚点（快照 `f9eb70c`）：
  MPK 的 `sched_queue` 是 **MPSC**（128 worker / 16 scheduler，`get_rand_sched_id()`
  `persistent_kernel.cuh:857-869` ⇒ 8 生产者/队列），push 三步 = 占位(`atom.add.release`) →
  relaxed 写 payload → **连续 CAS 放行**(`atom.cas.release(last_ready,pos,pos+1)`，`:1206-1228`)；
  而 `worker_queue` 是 SPSC（仅 scheduler 一个生产者）用单条 `atom.add.release`（`:1495-1502`）。
  见 `docs/04-experiments.md` **§37**，源码 `src/mpsc_ring.cu`、脚本 `harness/mpsc_ring.sh`、原始输出
  `harness/mpsc_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2，grid=P+1、每 block 1 线程，
  payload 自描述、逐字校验）。核心结论：
  - **§A 两种完成门控协议多生产者下全对**：`mpk`（连续 CAS 放行）P=1/2/4/8/32、`pertile`（逐槽
    release 标志）、紧容量 CAP=32、`spsc`（P=1）**全 `bad=0`**。
  - **§B 结构不严谨的替代在自然时序下「侥幸」0 错**：`reserve_count`（非连续完成计数 `atomicAdd(done,1)`）
    全 0；`pub_then_write`（占位即就绪）各 1/400000（连跑 0/1/1）——与 §36.3「时序侥幸非形式保证」同源。
  - **§B2 决定性正对照 `stall0`（低号槽 `r<4` 延迟 ~110µs）分出胜负**：`reserve_count`/`pub_then_write`
    **`bad=8` 稳定复现（3/3）**，而 `mpk`/`pertile` **恒 0（3/3）**——完成门控把消费者挡在低号槽外。
    即 **MPK 第③步「连续 CAS 放行」是「占位乱序 ⇒ 必须连续」的必要工程**。
  - **§C 放行代价 ≈ +0.38 µs/项**：`PL=1` 时 `mpk` P=1 相对 `spsc` 基线 +381 ns（与 §36.4 的 +355–372 ns
    一致）；P=8 并行藏掉、`pertile` 比中央 CAS 略便宜（1120 vs 1175）；PL=256 打平（§36.4 同结论）。
  判据：单常驻 kernel、设备侧推进（C1–C4）、仍差 C5 一步（与 §31–§36 同档）。**本轮新增实验**；
  P47-1 新留尾巴（§37.7）：消费者单线程致每项 ~2.3µs 往返主导、`stall0` 为人为扰动、未做真实 task
  描述符、litmus 仍未扫 SM 距离/`ncu` 内存序事件、未与 DeepGEMM SM100 路径对跑。

### P49 实验扩展 31：MPSC 批量出队 + 多线程消费者 + 真实描述符（`docs/04-experiments.md` §38）

- [x] **P49-1** 补掉 P48-1（§37.7）明留的两条尾巴——「消费者是**单线程**、每项 ~2.3µs 同步往返主导成本」
  （生产实现会**批量取 task**，`docs/04` §15 的 K≈W/G）与「payload 是纯 `uint32`、**未做真实 task 描述符**」：
  把 §37 的 MPSC 队列升级为「**批量出队 + 一个 warp 的并行校验 + 128 B 真实 `TaskDesc`**」。
  真实锚点（快照 `f9eb70c`）：worker 拉取 task 不是逐条——`thread0` 先读 task 区间、再用 **128-bit
  `cp.async` 把 `TaskDesc` 批量搬进 SMEM**（`persistent_kernel.cuh:1063-1078`）；`TaskDesc`
  （`runtime_header.h:296-336`，`alignas(16)`，`input_ptrs[8]`/`output_ptrs[3]` `:327-328`）~128 B；
  `per_worker_queue_len=8192`（`:1790-1791`）；`prepare_next_batch`（`:1378-1423`）。见
  `docs/04-experiments.md` **§38**，源码 `src/mpsc_batch.cu`、脚本 `harness/mpsc_batch.sh`、原始输出
  `harness/mpsc_batch.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2，grid=P+1；
  生产者协议与 §37 `mpk` 逐字一致）。消费者三档（唯一变量）：`serial`（1 线程逐条）/ `batchK`
  （1 线程一次等 K 条、一次放行 K 槽）/ `warpK`（1 warp，每 lane 并行校验 1 条）；描述符
  `w0=task id/w1=op/w2=layer/w3=iter/w[i>=4]=mix`，`DW∈{8,16,32}`=32/64/128 B，另测「只查 task id」路由粒度。
  核心结论：
  - **§A 批量/多线程出队不动数值**：`DW=32`、`P=8` 下 `serial`/`batchK=32`/`warpK=32` **全 `bad=0 / 800000`**
    （连续 CAS 游标保证「读到 `last_ready=c+K` ⇒ 前 `c+K` 槽已写」，批量读安全）；连跑 3 次恒 0。
  - **§B 单线程批量只收回「放行往返」，约 1.2–1.5× 封顶**：`DW=8`、`P=8`，K=1→32 为
    2144.63→1462.61 ns/item（**1.466×**，K≥16 饱和）；`DW` 越大收益越小（1.47→1.31→1.16×），
    说明瓶颈从「放行往返」转到「**串行读 payload**」。
  - **§C/§D 多线程（warp）并行读描述符，128 B 下快 21×**：`serial` vs `warp`，`DW=8` 2144.62→272.79
    （**7.86×**）、`DW=16` 3677.69→307.39（**11.96×**）、`DW=32` 6353.50→302.08（**21.03×**）；
    `serial` 随描述符线性变贵（~170 ns/字）而 `warp` 几乎不变。§D 拆因（`DW=32`）：`serial` 路由 1218 vs
    全校验 6353（**5.2×**）、`batch` 路由 341.64 vs 5477（**16.0×**）、而 `warp` 路由 301 vs 303（**几乎
    相同**）——**能否并行读 payload 是「路由型」与「全描述符型」消费的分水岭**。
  - **§E 多线程把瓶颈从消费者翻到生产者**：`DW=32` 下 `batch` 与 `P` 无关（恒 ~5.47µs，消费者自卡）；
    `warp` 随 P 快速下降（P=1→1786、4→448、8→**303** ns/item），到 P=8 已接近生产者侧成本。
  判据：单常驻 kernel、设备侧推进（C1–C4）、`warp` 并行取 task 但无固定角色动态派发，**仍差 C5 一步**
  （与 §31–§37 同档）。**本轮新增实验**；收口 §37.7 两条尾巴并指出「批量」同时是**正确性前提**与**性能出路**。
  P49-1 新留尾巴（§38.7）：描述符仍是 `mix` 合成布局、未实测 `cp.async` 批量加载本身收益、未扫更大 `P`/
  多消费者 block、未与 DeepGEMM SM100 路径对跑；§37.7 的「litmus 扫 SM 距离/`ncu`」仍余。

### P50 实验扩展 32：worker 取 task 的载体——`cp.async` 批量加载 TaskDesc（`docs/04-experiments.md` §39）

- [x] **P50-1** 补掉 P49-1（§38.7）明留的尾巴「`cp.async` 只是等效语义（§38 用普通 `ld.relaxed` 读，
  **未实测 `cp.async` 批量加载本身的收益**——那是另一条线）」「`warp` 下 `cp.async`/`mbarrier` 的批量
  加载形态」：把 worker「取 task」的**载体选择**单独量出来。真实锚点（快照 `f9eb70c`）：worker 用所有
  线程协作、每次 16 B 的 `cp.async` 把 `TaskDesc` 批量搬进 SMEM（`persistent_kernel.cuh:1063-1078`；
  单缓冲 `cp_async_wait<0>` `:1076-1077`）；`load_smem` = `cp.async.ca.shared.global.L2::128B`
  （16 B、预取 128 B、绕过寄存器；`tasks/common/copy_sm80.cuh:44-57`）；`TaskDesc`
  （`runtime_header.h:296-336`）真实 **352 B**（论文 §6.1 行 1365）；论文 §5.3「Pre-fetching task
  descriptions」（行 1364-1368）。见 `docs/04-experiments.md` **§39**，源码 `src/cpasync_desc_fetch.cu`、
  脚本 `harness/cpasync_desc_fetch.sh`、原始输出 `harness/cpasync_desc_fetch.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、单 block 持久化消费者、256 MB ≫ L2）。四取数路径公平对照
  `scalar`（4 B 寄存器）/`vec4`（128-bit 寄存器）/`cpasync`（SMEM 单缓冲）/`cpasync_pf`（双缓冲预取）。核心结论：
  - **§A/§B 载体不动数值**（四路径 checksum 全 `bad=0`）。**`cp.async` 的杀手锏是「成本与描述符大小
    无关」**：小描述符（≤128 B）时单缓冲 `cpasync` 反比 `vec4` 慢 1.15–1.44×；**到真实 352 B（`DW=88`）
    翻盘——`cpasync` 58.0 vs `vec4` 132.9（2.29×）、vs `scalar` 167.1（2.88×）**，因寄存器读被寄存器压力
    拖垮（`vec4` ns/字 1.21→1.51、`scalar` 1.87→1.90），而 `cpasync` 从 128 B→352 B 几乎不动（55.7→58.0）。
  - **§C 批量大小 `K`**：`cp.async` 的价值集中在**任务稀疏（`K` 小）**——`K=1` 时 `cpasync_pf` 474.5 vs
    `vec4` 1030.3（2.17×）、`K=4` 127.9 vs 264.6（2.07×）；`K≥32`（task 足够多）则 `vec4` 反超
    （38.6 vs 55.7），只有加预取才追到 ~44.9。
  - **§D 预取（双缓冲）** 相对单缓冲的 `cpasync` 稳定收回 **7%–40%**（`K` 越小越多：`K=4` 1.40×、
    `K=64` 1.07×）——**`cp.async` 单用只是「换载体」，加预取才是「流水」**，对应论文 §5.3 的 pre-fetch。
  - **§E 线程数是弱标度**（单 block 受延迟约束；扩展维度是多 worker，未测）。
  含义：**选 `cp.async` 为载体在真实 352 B 描述符下是对的（2.3×），而论文说的 pre-fetch 相对快照的单缓冲
  `wait<0>` 还能再收 1.07–1.40×**；§38 用普通 load 的结论只在它的 128 B/大 `K` 口径下成立。判据不变
  （单常驻 kernel、设备侧推进 C1–C4，**无 C5**，与 §36–§38 同档）。**本轮新增实验**；P50-1 新留尾巴
   （§39.7）：单 block 非多 worker 带宽口径、`mbarrier` 形态未测、未与 DeepGEMM SM100 路径对跑。

### P51 实验扩展 33：多 worker 取 task 带宽标度 + TMA/mbarrier 形态（`docs/04-experiments.md` §40）

- [x] **P51-1** 补掉 P50-1（§39.7）明留的两条尾巴——「**单 block** ⇒ 测的是『取一批的延迟』而非
  『访存带宽』，要多 worker 的带宽标度需另做多 block 实验」与「**`mbarrier` 形态未测**（`cp.async` 用
  `commit_group`/`wait_group` 软件计数，未测硬件 mbarrier/TMA 风格）」：把 §39 的「单 block 消费者」
  扩成 **G 个 worker block**，并新增两条 **TMA bulk + 硬件 mbarrier** 取数路径。见
  `docs/04-experiments.md` **§40**，源码 `src/multi_worker_desc_fetch.cu`、脚本
  `harness/multi_worker_desc_fetch.sh`、原始输出 `harness/multi_worker_desc_fetch.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`-arch=sm_90a`，256 MB ≫ L2）。真实锚点：
  worker 16 B `cp.async` 单缓冲（`persistent_kernel.cuh:1063-1078`）；硬件 mbarrier + TMA bulk 形态在
  task 单体内部（`tasks/hopper/tma_2d.cuh:119`、`barrier.cuh:42/59/75`），本节搬到「取 task」层；
  `TaskDesc` 352 B、论文 §5.3 pre-fetching（行 1364-1368）；worker 数按 SM（§5.1 行 1168）。六路径
  `scalar`/`vec4`/`cpasync`/`cpasync_pf`/`bulk`(TMA+mbarrier 单)/`bulk_pf`(双)，全 `bad=0`。核心结论：
  - **§B 多 worker 标度：aggregate 带宽随 G 近似严格线性、到 G=2×SM 仍未饱和**——`vec4` G=1 3.3→
    G=264 **803 GB/s**（每 worker 恒 ~3.0 GB/s）、`cpasync` ~2.4 GB/s/worker；**取 task 是「per-worker
    延迟绑定」，带宽只能靠堆 worker 换**（对上 MPK §5.1 worker 数按 SM）。128 B 下 `vec4` per-worker 最高。
  - **§C TMA bulk 单缓冲 ≡ 16 B `cp.async`**（同 K 逐行相等，K=32 0.41 vs 0.40）——省指令不改吞吐；
    **收益全在双缓冲预取**：`bulk_pf` 每个 K 最快（K=1 3.21 vs `cpasync` 4.90 = 1.53×）。
  - **§D 真实 352 B 下 `bulk_pf` 独一档**：0.28 ns/item = **1263 GB/s**，比 `vec4`（1.05）快 **3.75×**、
    比 `cpasync_pf`（0.42）快 **1.52×**（硬件 mbarrier 单线程 issue 让预取近乎免费）；`cpasync`/`bulk`
    几乎不随 `DW` 变（复现 §39.2，但在多 worker + 绝对 GB/s 口径：`cpasync` 903.8 vs `vec4` 335.5）。
  - **§E 多 worker 下线程数不再是变量**（并行度由 worker 数提供）。
  - **踩坑（顺带修正 §39 潜在缺陷）**：`cpasync` 双缓冲**最后一批**用 `wait_group 1` 会立即返回、可能
    读到未完成的 SMEM（首版 `K=4`/`DW=8` 各一例 `bad=1`，复现）；改成最后一批 `wait_group 0` 后全
    `bad=0`。**§39 的 `cpasync_pf` 有同构写法**，当时 `bad=0` 是时序侥幸（与 §36.3 同源），结论不受
    影响但记为待修隐患。
  判据 C1–C4、无 C5（与 §36–§39 同档）。**本轮新增实验**；P51-1 新留尾巴（§40.8）：1D bulk 非
  TensorMap、无动态生产者、未测 K×SMEM 联合约束、未与 DeepGEMM SM100 路径对跑。

### P52 实验扩展 34：取 task 的 SMEM 预算 K × 环形深度 D（`docs/04-experiments.md` §41）

- [x] **P52-1** 补掉 P51-1（§40.8）明留的第一条尾巴「**未测 `K` 与 SMEM 容量的联合约束**」：
      把 §40 的「双缓冲」推广成 **D 深环形缓冲**（每槽 K 条 `TaskDesc`），于是描述符暂存的 SMEM
      = `D·K·352 B`，`K` 与 `D` 是一对联合参数；量「同等 SMEM 预算下大批量浅环 vs 小批量深环」。
      见 `docs/04-experiments.md` **§41**，源码 `src/fetch_smem_budget.cu`、脚本
      `harness/fetch_smem_budget.sh`、原始输出 `harness/fetch_smem_budget.out.txt`（目标卡 H100/sm90a、
      **固定空闲 device 3**、nvcc 13.2、DW=88=352 B 对齐 MPK §6.1、载体=TMA bulk+硬件 mbarrier）。
      核心结论：
      - **§A 任意 (K,D) 多 worker 下 `bad=0`**（16 组）；**§B 环深 D：1→2 拿到全部预取收益、`D≥2`
        完全平坦**（K=8：402.8→579.9 GB/s 后恒 ~580）；§D（K=1 扫 D=64…640）同证吞吐与 D 无关。
      - **§C（headline）K×SMEM 联合约束**：同等 SMEM 下吞吐**单调取决于 K**（每批一次 mbarrier 等待 +
        两次 `__syncthreads()` 的固定开销摊到 `K` 条）——`K=1,D=186`（用满 64 KB）73.8 GB/s vs
        `K=64,D=2`（只用 44 KB）**2096 GB/s（28×）**；`K=128,D=2` 达 **2667 GB/s**（S=128KB）、
        `K=128,D=4` **2693**（S=192KB，逼近 HBM ~80%）。**最优 = 大 K + D=2**，深环是浪费；也把
        §40 §D 的 K 上限从 32 推到 128（K=32/D=2 的 1239 与 §40 的 1263 一致）。
      - **§D**：环可长到 225 KB（占满一整个 SM 的 SMEM）仍正确（opt-in=232448 B），但深环无用。
      - **§E 与 task 执行争 SMEM**：每 block SMEM 从 0.7→88 KB 把占用率从 **16→2 block/SM**，
        aggregate 带宽却守住 **2.3–2.7 TB/s**（大 K 的在飞字节补偿 worker 数下降）；最优在
        **K=8（5.5 KB、16 block/SM）2672 GB/s**——「中等 K + 高占用率」略胜「大 K + 低占用率」。
      - 含义：取 task 的 SMEM 是**软约束**（可换 worker 数、总带宽守恒）；正确姿势是**把预算堆到
        批量 K（K≈32–128）、预取深度只要 2**，对应 MPK §5.3 paged shared memory（行 1313-1325）
        的「描述符暂存页」应按一整批开。判据 C1–C4、无 C5（与 §36–§40 同档）。踩坑：`NDESC` 未按
        `K` 截断致 `bad=1`/假 41 TB/s；mbarrier 数组放静态 `__shared__` 与动态 SMEM 叠加超 opt-in
        报 `invalid argument`（改放动态 SMEM 尾部）。**本轮新增实验**；P52-1 新留尾巴（§41.8）：
        ~~未跑 16 B `cp.async` 深环载体~~、~~未测 `cp.async` 释放寄存器对下游 task 的收益~~（**均已由
        P53-1/§42 补掉**：16 B cp.async 深环与 bulk 在同一 K×D 平面对跑、载体高下随 K 翻转；cp.async
        并不省寄存器，价值在「成本与描述符大小无关」+「零寄存器代价预取」）、
        无动态生产者、未与 DeepGEMM SM100 路径对跑。

### P53 实验扩展 35：取 task 深环的第三种载体（`docs/04-experiments.md` §42）

- [x] **P53-1** 补掉 P52-1（§41.8）明留的两条尾巴——「载体固定为 TMA bulk + mbarrier，**未同时跑
      16 B `cp.async` 环**（其预取机制 `wait_group` 是软件计数，深环写法见 §40.8 的 drain 坑）」与
      「**未测 `cp.async` 释放的寄存器对下游 task 执行的实际收益**」：把快照的**原生**取 task 载体
      （16 B `cp.async` 进 SMEM，`persistent_kernel.cuh:1063-1078`）放回 §41 的 `K×D` 平面对跑，
      并单独量载体对下游 task 寄存器足迹的影响。见 `docs/04-experiments.md` **§42**，源码
      `src/cpasync_ring_budget.cu`、脚本 `harness/cpasync_ring_budget.sh`、原始输出
      `harness/cpasync_ring_budget.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
      DW=88=352 B 对齐 MPK §6.1）。两条环载体 `bulk`（TMA bulk+硬件 mbarrier，运行时 D）与 `ca`
      （每 lane 16 B `cp.async.cg` + 软件 `wait_group(D-1)`，**模板 D=1…32**，尾批 `wait_group 0` 兜底）。
      核心结论：
      - **§A 任意 (K,D) 多 worker `bad=0`**（16 组）；cp.async 预取增益也**全在 `D=1→2`**（1.45–1.62×）、
        `D≥2` 平坦，与 §41 bulk 同构。
      - **§B（headline）载体高下随 `K` 翻转**：`K≤32` cp.async 反超 **1.06–1.23×**、`K=64` 打平、
        `K=128` bulk 反超 **1.07–1.13×**——根因是 **cp.async 发指令条数随 `K` 线性增长**（`K=128`
        整批 2816 条、每 lane 22 条）而 bulk 一条指令搬整批。
      - **§C** `K=8`：两载体都在 `D≥2` 平坦（bulk 恒 ~560、ca 恒 ~640–657），**稳态 cp.async 高 ~1.17×**；
        「取 task 环只需 2 槽」对两载体都成立。
      - **§D「cp.async 释放寄存器」被否证**：`H=0` 时 `fetch_task_ca` **28** regs vs `fetch_task_reg` **19**
        ——SMEM 寻址/逐 16 B 地址增量抵消了省；下游 task 一重（`H≥32`）两者 `numRegs` **完全相同**（40/168/254）、
        无 spill；取数侧 ca 恒慢 3–7%（每批两次 `__syncthreads`）。故 cp.async 的价值是「成本与描述符
        大小无关」（§39）与「**零寄存器代价预取**」（`D` 个批放 SMEM），**不是**取数代码省寄存器。
      判据 C1–C4、无 C5（与 §36–§41 同档）。**本轮新增实验**；P53-1 新留尾巴（§42.7）：1D 非 TensorMap、
      无动态生产者、未叠动态入队、未与 DeepGEMM SM100 路径对跑。

### P54 实验扩展 36：取 task 与 task 执行的流水（`docs/04-experiments.md` §43）

- [x] **P54-1** 补掉 §39–§42（P50-1..P53-1）的整合缺口——那四节把「worker 取 task」的载体
      （寄存器 / 16 B `cp.async` / TMA bulk）与缓冲预算（批量 `K` × 环深 `D`）量透了，但**全在
      「只取数、不执行 task」的口径**下（消费者只 `fold` 一下）。本节第一次把「取 task」与**真正
      task 执行**放进同一循环，回答：描述符取数在 worker 循环里是不是关键路径？单缓冲（快照形态，
      `cp_async_wait<0>`）留下的串行缝能否被预取藏掉、随 task 变长怎么变？见
      `docs/04-experiments.md` **§43**，源码 `src/fetch_exec_overlap.cu`、脚本
      `harness/fetch_exec_overlap.sh`、原始输出 `harness/fetch_exec_overlap.out.txt`（目标卡
      H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`G=132`、`NDESC=393216`×352 B ≈ 132 MB ≫ L2）。
      取数载体 = TMA bulk + 硬件 mbarrier；环形缓冲 `D` 槽；task 执行 = 数据相关折叠 + `R` 轮串行
      LCG 计算压舱（`R` = 唯一的执行时间旋钮）；另有 `do_fetch=0` 档隔离纯执行。核心结论：
      - **§A 正确性**：`R=0` 小 `NDESC` 的 `D∈{1,2,3,4}` 全 `bad=0`；`R=1024` 大 `NDESC` 的
        `D∈{1,2,3,4}` 的 `sink` **逐位相同**——取+执行环不动数值。
      - **§B（headline）预取收益随 task 变长单调衰减、绝对收益恒定**——`D=1`（= 快照单缓冲）vs
        `D=3` 的 speedup 从 `R=0` 的 **1.354×** 单调降到 `R=262144` 的 **1.003×**，而绝对差
        `D=1−D=3` 恒 ~**0.017–0.023 ms/run**（与 `R` 无关）——单缓冲多付的是一条**固定长度的串行缝**
        （≈每 worker-batch 0.9µs），**不随计算量放大**（与「预取藏进计算、越算越划算」的直觉相反）。
      - **§C 环深**：`R=1024` 下 `D=1→2` 收回大部分（1.19×）、`D=2→3` 再 1.04×、`D≥3` 平坦
        （与 §41 §B/§42 §C「环只需 2 槽」同构）。
      - **§D 决定性分解**：`fetch_only` 恒 0.061 ms（吞吐地板）、`exec_only` 随 `R` 线性增长；
        **`D=3 = max(fetch_only, exec_only)`**（`p/max` 1.001–1.047，`R` 越大越精确）——**预取把
        描述符取数完全藏在 task 执行后面**（`exec≥fetch` 时取数对总时间零贡献）；`D=1`（快照单缓冲）
        始终落在 `exec_only + ~0.02 ms`，`D=1/(f+e)` 0.74→0.99（多 worker HBM 重叠让单缓冲也没「完全
        串行」，但达不到 `D=3` 的零缝）。
      - **§E 批量**：`R=1024` 下 speedup 随 `K`（32→128）从 1.154 升到 1.228（batch 越富、同步边界越少）。
      含义：**取 task 在重 task 下完全可藏，只在轻 task 下显形**；MPK 自动融合把算子切成**多而薄**的
      task（§01 §3：293→13867 task），故 pre-fetching（论文 §5.3 行 1364-1368）对 MPK 比「少而重」的
      DeepGEMM 段级 mega-kernel 更重要——从执行侧解释了 §5.3 的优化取舍。与 P13-1（数据面载入流水，受
      `T_mem/T_comp` 限制）/P23-1（权重面预取，争 HBM）合成「软件流水三面：数据/权重/控制面」。判据
      C1–C4、无 C5（与 §36–§42 同档）。**本轮新增实验**；P54-1 新留尾巴（§43.7）：执行是合成纯 ALU
      压舱（不争 HBM）、`D=1` 非「完全串行」纯量、未叠动态生产者、未与 DeepGEMM SM100 对跑。

### P55 实验扩展 37：取 task × 吃 HBM 的 task 执行（`docs/04-experiments.md` §44）

- [x] **P55-1** 补掉 §43.7（P54-1）明留的第一条尾巴「`exec_task` 是**纯 ALU 计算压舱、不吃 HBM**
  （真实 task 会同时吃 HBM/TC），届时『取数与执行是否争带宽』需另测」：把 §43 的「取 task 与执行
  同环」接上一块 **≫L2 的数据面流式读**（每 batch 连续 `K·EDW` 个字，coalesced `uint4`），让
  「取 task 的 TMA bulk（控制面，源=HBM）」与「执行 task 的数据面读（HBM）」**共用同一条 HBM 带宽**。
  见 `docs/04-experiments.md` **§44**，源码 `src/fetch_exec_hbm.cu`、脚本 `harness/fetch_exec_hbm.sh`、
  原始输出 `harness/fetch_exec_hbm.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `G=132`、`NDESC=393216`×352 B ≈ 132 MB ≫ L2、每配置 2 次 warmup + min-of-5 reps）。四 Section：
  核心结论：
  - **§A 正确性**：`EDW=8` 下 `D=1..4` 的 `sink` **逐位一致**；加数据面/预取/环深不动数值。
  - **§B（headline）加数据面后 `D=3` 仍趋 `max(fetch,exec)`、不会与执行相加**——`p/max`（= `D3/exec`）
    在 `EDW=16` 峰值 **1.247**、随数据面增大回落到 `EDW=1024` 的 **1.010**，全程 `p/(f+e) ≤ 0.957`；
    最坏点 `EDW=16` 若串行相加 `D=3/exec` 应 ≈ **1.94**、实测仅 **1.25**（取数独立代价只暴露 ~27%）。
    `EDW=0` 复现 §43（`p/max=1.003`）。原因：1 block/SM 的数据面 `uint4` 流式只到 **~1.2 TB/s**，
    低于描述符流单独能拉的 **~1.8 TB/s**（132 MB/0.074 ms）→ HBM 有余量，控制面条带被预取压在上面。
  - **§C 扫计算强度 `R`（`EDW=256`）**：`p/max` 平稳 ~1.03–1.04，随执行变长（`exec_only` 0.38→1.12 ms）
    **收敛回 1.009**——**执行越重、取数越藏**（与 §43 同向）。
  - **§D 取 task 净加价**：`D3−exec` 稳定 **0.01–0.02 ms**（fetch_only 0.0741 ms 的 **2%–26%**），
    峰值 0.0195 ms 出现在**数据面刚好装进 L2（`EDW=16`）**的窄区（数据面 L2 命中、无 HBM 往返可藏）。
  含义：**取 task 与执行不构成零和争抢**——只要执行够长且 HBM 未被打满，预取就把控制面藏住；这与
  DeepGEMM「重且吃 HBM」的段级 task（`p/max→1.01`）和 MPK「多而薄」的 task 都自洽。判据 C1–C4、
  无 C5（与 §36–§43 同档）。**本轮新增实验**；P55-1 新留尾巴见 §44.7：数据面未打满 HBM（饱和侧的
  争抢未测）、执行是合成 `uint4` 读（非 GEMM/RMSNorm）、`do_fetch=0` 是另一条代码路径（含固定同步缝）、
  未叠动态生产者、未与 DeepGEMM SM100 路径对跑。

### P56 实验扩展 38：取 task × 吃 HBM 的 task 执行（饱和侧）（`docs/04-experiments.md` §45）

- [x] **P56-1** 补掉 P55-1（§44）§44.7 明留的第一条尾巴「**数据面未打满 HBM（饱和侧的争抢未测）**」：
  把 §44 的「取 task 与吃 HBM 的执行同环」推到**数据面把 HBM 打满**，回答「饱和后控制面字节是否转为
  可见争抢」。见 `docs/04-experiments.md` **§45**，源码 `src/fetch_exec_saturate.cu`、脚本
  `harness/fetch_exec_saturate.sh`、原始输出 `harness/fetch_exec_saturate.out.txt`（目标卡 H100/sm90a、
  **固定空闲 device 3**、nvcc 13.2、`G=132=#SM`、`K=128`、`DW=88`=352 B、`D=1`(单缓冲)/`D=2`(预取)、
  `NDESC=393216`（描述符 132 MB）、数据面 `NDESC·EDW` 字（EDW=4096 时 6.4 GB）、min-of-9 reps）。
  唯一新机制变量 = **block 线程数 `T`**（§44 固定 `T=256`）。四 Section：核心结论：
  - **§A 正确性**：`D=1..4` 的 `sink` 逐位一致（加线程/数据面/预取不动数值）。
  - **§B 数据面与控制在「每 block 线程数」上最优相反（headline 之一）**：数据面读带宽随 `T` 上升并
    **在 `T=1024` 饱和到 ~2983 GB/s（EDW=4096，≈ H100 HBM3 峰值 ~89%）**，而 `T=256` 无论数据多大都
    卡在 **~1.2 TB/s**（= §44 的口径，故 §44「未饱和」的根源是**线程数**、不是 block 数：换 `G=264`
    反因 SMEM 环 88 KB/block + 寄存器限 1 block/SM 而略慢）；反之**控制面 `fetch_only` 取数带宽随 `T`
    下降**（`T=256→1024` 由 **1791→891 GB/s**，多出的线程只增 `mbar_wait` 自旋与 `__syncthreads` 宽度）。
  - **§C 饱和态扫 EDW（headline 之二）**：`D=2` **全程趋 `max(fetch,exec)`、从不接近相加**——
    `p/max` 在 `EDW=64` 峰值 **1.132**、回落 `EDW=4096` 的 **1.015**，全程 `p/(f+e) ≤ 0.947`；最坏点
    `EDW=64`（`fetch/exec≈0.89`）若相加应 ≈ **1.89**、实测仅 **1.13**（取数独立代价只暴露 ~15%）。
    `EDW=4096` 数据面已饱和，`D=2` 只比纯执行慢 **1.5%**、总带宽 **3007 vs 3052 GB/s 同档**——控制面
    钻的是数据面没吃满的 ~10% HBM 余量。**§44.7 尾部问题的答案：饱和不暴露可见争抢**。
  - **§D 单缓冲代价随数据面消失**：快照 `D=1` 相对 `D=2` 的惩罚从 `EDW=0`（纯控制面）的 **1.115×**
    单调降到 `EDW=4096` 的 **1.004×**；绝对缝 `D1−exec` 稳定 ~0.02–0.04 ms（固定长度串行缝）。
  含义：(1) 「控制面描述符取数会不会与执行争带宽」**恒为『不争』**（只要描述符字节 ≪ task 主体数据，
  即 MPK「多而薄」task / DeepGEMM 段级重 task 的处境）；(2) 新机制——控制面要少线程、数据面要多线程，
  真实 megakernel 单 worker block 只有一个尺寸（快照 `T=256`：控制最优、数据最差），故要同块兼顾只能
  **warp specialization**（P13-1/P18-1 的路子），从执行侧解释生产 mega-kernel 普遍用 warp 特化。
  判据 C1–C4、无 C5（与 §36–§44 同档）。**本轮新增实验**；P56-1 新留尾巴（§45.7）：占用率被 SMEM 环
  +寄存器锁在 1 block/SM（饱和旋钮是块内线程数而非 block 数）、数据面是合成 `uint4` 求和、未叠动态
  生产者、未与 DeepGEMM SM100 路径对跑。

### P57 实验扩展 39：多 block/SM 的取 task × 吃 HBM 执行（`docs/04-experiments.md` §46）

- [x] **P57-1** 补掉 P56-1（§45）§45.7 明留的第一条尾巴「占用率被 SMEM 环锁在 1 block/SM，
  若要测**多 block/SM 争 HBM**需另配小 SMEM 环，但小 `K` 会让控制面 TMA 低效，**两者不可兼得**」：
  把 §45 的「数据面要多线程 vs 控制面要少线程」这对**相反最优**放到**多 block/SM** 维度上量——
  数据面饱和的旋钮是「每 SM 总线程数」还是「块内 `blockDim`」？小环的代价来自 `K` 还是占用率？
  见 `docs/04-experiments.md` **§46**，源码 `src/fetch_exec_multiblock.cu`、脚本
  `harness/fetch_exec_multiblock.sh`、原始输出 `harness/fetch_exec_multiblock.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`、数据面至 6.4 GB、
  min-of-9）。机制与 §44/§45 同骨架，唯一新变量 = **SMEM 环大小 ⇒ 占用率**（`K=16,D=2` ⇒ 11 280 B
  ⇒ 8 blk/SM @T256；`K=128,D=2` ⇒ 90 128 B ⇒ 2 blk/SM）。四 Section 核心结论：
  - **§A** 多 block + 小 `K` 下 `D=1..4` 的 `sink` 逐位一致（不动数值）。
  - **§B（headline 之一）数据面饱和的旋钮是「每 SM 总线程数」而非「`blockDim`」**：所有
    `thr/SM=1024` 的拆法（`B,T=1,1024`/`2,512`/`4,256`/`8,128`）都落在 **2928–3009 GB/s**、
    `thr/SM=512` 落 2157–2209、`256` 只 1242；且 **`B=4,T=256`（3009）≥ §45 的 `B=1,T=1024`
    （2928）**——多块不劣于单块大 T。**即 §45 的「单块 `blockDim` 矛盾」可由多 block/SM 化解**：
    保留控制面友好的 `T=256`、用 `B=4` 块凑够 1024 线程/SM。
  - **§C（headline 之二）控制面取数高度依赖 `K`、对占用率较不敏感**：1 block/SM、`T=256` 下
    `K=8/16/32/128` = **437/691/937/1782 GB/s**（`K` 小则每批一次 mbarrier 等待 + 两次
    `__syncthreads` 摊薄）；多 block（同 `blockDim`、更高 `thr/SM`）对控制面是**中性偏轻**的
    伤害（`K=16/32` 慢 1.26×/1.70×、`K=8` 反快 1.23×），**轻于 §45 的「大 `blockDim`」（掉 2.0×）**。
    ⇒ §45.7 的「小环 ↔ 控制面低效」取舍是真的，但**代价来自 `K` 而非占用率**。
  - **§D（headline 之三）小环 + 多块下 `D=2` 仍藏住控制面**：`G=1056,B=8,T=256,K=16`，`p/max`
    峰值 **1.126**（EDW=64）→ **1.020**（EDW=4096），全程 `p/(f+e) ≤ 0.914`；`D=1` 代价 `s1/exec`
    1.021–1.139。虽控制面单条慢（524 vs §45 的 1791 GB/s），其贡献仍被数据面吸收。
  - 含义：**数据面认「总线程/SM」、控制面认「`K`」两个正交量** ⇒ 设计上应「用多块凑 1024 线程/SM、
    把 `K` 尽量做大」，二者在 SMEM 预算内可同时满足；把 §45 的 **warp specialization** 与本节的多
    block/SM 并列为「同 block 内 / 跨 block」两条兼顾路径。判据 C1–C4、无 C5（与 §36–§45 同档）。
     **本轮新增实验**；P57-1 新留尾巴（§46.7）：合成 `uint4` 数据面、未观测到超饱和的争抢、未叠动态
     生产者、未与 DeepGEMM SM100 路径对跑。

### P58 实验扩展 40：取 task × 动态生产者（`docs/04-experiments.md` §47）

- [x] **P58-1** 补齐 §39.7/§40.8/§42.7/§43.7/§44.7/§45.7/§46.7 **每一节都明留的同一条尾巴**
      「**未叠动态生产者**」：§39–§46 把「worker 取 task（控制面）」的载体/缓冲预算/占用率/HBM
      饱和都量透了，但取数数组全是 **host 预写的静态块**；真实 megakernel 由设备侧 scheduler
      **现场**把 task 推进 ready 队列。本节把「描述符来源」接进同一流水，并拆成两个正交成本：
      **调度（预计算描述符 + 设备只推就绪）** vs **生成（设备现场造 352 B payload）**。
      见 `docs/04-experiments.md` **§47**，源码 `src/dyn_producer_fetch.cu`、脚本
      `harness/dyn_producer_fetch.sh`、原始输出 `harness/dyn_producer_fetch.out.txt`（目标卡
      H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`（静态数组
      138.4 MB ≫ L2）、`K=32,D=2,T=256`、worker `G=131`（producer 另占 1 块）、min-of-9）。
      三模式：`static`（m0，host 预写数组）/`sched`（m2，预写数组 + producer 只推 `head` + worker
      `ld.acquire` 等）/`gen`（m1，producer 现场生成 payload 写有界环 + 背压 + `fence.proxy.async`
      + release 发布）。核心结论：
      - **§A 三模式逐位一致**（D=1..4，`got=404219327039195`）——现场生成 + release 发布 +
        TMA 跨 proxy 读 + 有界环背压这套协议**不动数值**。
      - **§B（headline）动态分两层、差 ~27×**：`EDW=0` 下 `static` 0.1605 ms / 862.5 GB/s（0.41 ns/条）、
        **`sched` 1.3614 ms（8.48×，3.46 ns/条 ≈ 285M task/s）**、**`gen` 36.489 ms（227.4×，92.8 ns/条）**。
        现场生成的瓶颈是**单 SM 标量写吞吐**（`NDESC/256=1536` 轮 × ~0.9 µs），与 `G` 无关。
      - **§C** `gen` 的额外计算被存储隐藏（`PW≤1024` 恒 227×），`sched` 的额外计算直接显形
        （`PW=4096` → 42.4×）——生成瓶颈在**存**、调度瓶颈在**决策**。
      - **§D 有界环容量**：死锁边界 ~`2·(G·K)`（`CAP=4192` 死锁、`8384` 活），安全容量之上吞吐不变
        （安全阀非吞吐旋钮，与 §35 一致）。
      - **§E 接数据面**：`sched` 固定成本 ~1.2 ms，`EDW=4096` 时被藏到 **1.014×**（询动态调度近乎
        免费，与 §44–§46「控制面被数据面藏住」一致）；`gen` 恒 ~36.5 ms **不随数据面缩小**。
      含义：**megakernel 的「动态」必须落在调度（推 task id），不能落在生成 payload**——正是 MPK
      （`runtime.cc` 预计算 `FullTaskDesc`、scheduler 只映射入队，`docs/06` §2/§3/§7）与 DeepGEMM
      （编译期静态角色）的做法；论文 §5.2 的 AOT/JIT 分野在此有量化理由。判据：`sched`/`gen` 单常驻
      kernel + producer/worker 角色（C1–C4），producer 固定角色 + 设备侧动态发布就绪，**比 §36–§46 更
      靠近 C5**，但 worker 仍静态分配 batch ⇒ **仍差 C5 一步**。**本轮新增实验**；P58-1 新留尾巴
      （§47.8）：`gen` 单块 producer/标量写（可分散/向量化但方向不变）、`sched` 是单块循环上界
      （MPK 16 scheduler SM 更低）、合成描述符/数据面、未与 DeepGEMM SM100 路径对跑。

### P59 实验扩展 41：动态生产者的多块并行（`docs/04-experiments.md` §48）

- [x] **P59-1** 补掉 P58-1（§47）§47.8 明留的前三条尾巴——「`gen` 的 producer 是**单块**……真实系统
      若真要生成会分散到多块/SM（代价按生产者块数下降）」（尾巴①）／「`gen` 用标量 strided 写」
      （尾巴②）／「`sched` 的 1.36 ms 是**单块 producer** 的循环吞吐，MPK 实际用 **16 scheduler SM**」
      （尾巴③）：把 §47 的单块 producer 扩成 **P 块（独占 P 个 SM）**，量 gen/sched 代价随 P 的标度、
      饱和点，以及「动态必须落在调度、不能落在生成」的结论在 P 维度上是否保持。
      见 `docs/04-experiments.md` **§48**，源码 `src/dyn_producer_scale.cu`、脚本
      `harness/dyn_producer_scale.sh`、原始输出 `harness/dyn_producer_scale.out.txt`（目标卡 H100/sm90a、
      **固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`（静态数组 138.4 MB ≫ L2）、
      `K=32`、`D=2`、`T=256`、`grid=P+G=132=NSM` 全共驻、min-of-9）。机制 = §47 骨架 + **环按 producer
      分区**（每生产者 `CAPp=CAP/P` 槽、自己的 `head[p]/tail[p]`、**P 条独立 SPSC 子环**，worker 按
      `owner=bb/bp` 定位）；`P∈{1..128}`。五 Section 核心结论：
      - **§A 正确性**：`P∈{1,2,4}`×`D∈{1,2}`、分区子环 + 背压 + 每生产者游标，static/sched/gen
        **全部逐位一致**（`got=404219327039195`）。并记一个边界：**P=1 的「跨步消费预取领先量」**
        （worker 第 2 次投机预取等 `G` 个 batch 之外 → 浅环死锁），**多 producer 的分段子环反而放宽
        该约束**（owner 定位把 `need` 变局部）。
      - **§B（headline）标度**：static=0.169 ms/819.5 GB/s。**`gen` 的 1/P 只到 P≈8**——`gen·P` 在
        P=1..8 恒 **36.5→40.0**、P=16 起偏离（42/61.7/80.7），**最优 P=64=1.261 ms（28.9× 单块）但仍
        7.46× static**，P=128 退点（`G=4` 消费者瓶颈）。**`sched` 撞「静态读地板」**：1.32 ms（P=1）→
        **0.262 ms（P=32，1.55× static）**后不再降；**P=16=0.275 ms（1.63× static）正落 MPK 的 16
        scheduler SM 量级**（把 §47「单块 sched 上界」钉成实测）。⇒ 「动态必须落在调度、不能落在生成」
        在 P 维度**依然成立**。
      - **§C**：`gen` 的算被存藏住（P=16 时 `PW≤4096` 全程 gen/static 只 15.5→17.1）、`sched` 的算
        线性显形（绝对值按 ~1/P 缩、形式不变）——§47.4 两条在 P 维度保持/强化。
      - **§D**：小窗口 `CAPp=K` 必死锁；满速需 `CAPp ≥ bp·K`（每 producer 整个区间，不 wrap）——
        P=4 下 `bp·K/2` 慢 2.34×、`bp·K/8` 慢 3.44×；**容量需求随 P 按区间摊薄**（与 §47.5 窗口同源）。
      - **§E 接数据面**：单块 `gen` 恒 ~36.4 ms 不随数据面缩小（`gP1/max` 215×→7.1×）；**P=16 大幅
        收窄**（`gP16/max` 15.5×→**1.52×** @EDW=4096）——「生成盖不住执行」在 P 大 + 执行长下**近乎
        翻案成 1.5× 尾巴**，但仍未完全藏住；真实实现仍选编译期预计算（调度已免费，不必担生成的风险）。
      含义：MPK 的 **16 scheduler SM 有量化依据**（再多也被 worker 读描述符地板卡住）；本节每条子环是
      **单 writer（SPSC）**，与 MPK `sched_queue` 的 **MPSC**（多 scheduler 连续 CAS 放行）是两种拓扑
      （§37）。判据 C1–C4、比 §36–§47 更接近 C5（P 个固定角色 producer），但 worker 仍静态分配 batch
      ⇒ **仍差 C5 一步**。**本轮新增实验**；P59-1 新留尾巴（§48.8）：`gen` 仍标量 strided 写（尾部②
      未单独向量化，但饱和已转写带宽）、P=128 是消费者瓶颈、`sched` 地板随描述符数组大小变、run-to-run
      抖动 ~34%（P=16 gen）、合成描述符/数据面、未与 DeepGEMM SM100 路径对跑。

### P60 实验扩展 42：动态生产者的写入向量化（`docs/04-experiments.md` §49）

- [x] **P60-1** 补掉 P59-1（§48）§48.8 明留的**尾巴②**——「`gen` 仍是**标量 strided 写**（每线程顺序写
      88 字、跨线程步长 352 B、非 coalesced）；向量化/协作写未单独做，向量化至多改变饱和点、不改
      『生成 ≫ 预计算读』的量级」：把「写入形态」分解成两个正交旋钮——**(a) 存储宽度**（4 B 标量 →
      16 B `uint4`）与 **(b) 访存 coalescing**（逐线程 strided vs 跨线程连续），各自量贡献与饱和点。
      见 `docs/04-experiments.md` **§49**，源码 `src/dyn_producer_vector.cu`、脚本
      `harness/dyn_producer_vector.sh`、原始输出 `harness/dyn_producer_vector.out.txt`（目标卡 H100/sm90a、
      **固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、
      `T=256`、`grid=P+G=132=NSM` 全共驻、min-of-7）。唯一新变量 `gvar`：0 标量 / 1 逐线程 `uint4` /
      2 协作 coalesced（payload 内容同 `gen_word`）；另加 `mode 5 pure`（只 launch P 个 producer、
      无 worker/背压）量纯写吞吐。五 Section 核心结论：
      - **§A**：三种写入形态 × P∈{1,2}，`static`/`sched`/`gen` **全部逐位一致**（`got=404219327039195`）
        ——向量化/协作写语义透明。
      - **§B（headline）**：static=0.1711 ms/809.1 GB/s。逐线程 `uint4`（v4）在 P=1 把 `gen`
        **36.58→10.49 ms（3.5×）**、相对 static **213.8×→61.3×**；协作 coalesced 约 **1.9×**（113.9×）。
        **饱和点右移**：标量 `gen·P` 在 P≥8 起偏离（P≈8 后不再按 1/P 改善）、v4 到 P=64 仍近似线性；
        但最优 v4（P=64）仍是 **3.44× static** ⇒ §48「动态必须落在调度、不能落在生成」**依旧成立**。
      - **§C（机制）**：`mode 5` 纯写——单 SM **3.8（标量）→13.2（v4）GB/s**，瓶颈是 **store 发射/指令
        吞吐而非 DRAM 写带宽**；整机写天花板标量 ~**207 GB/s**（P≈32 饱和）、v4 **827**、coop **875**
        （P=128），向量化抬高 ~**4×**。**coalescing 是更弱的第二旋钮**：v4（不 coalesce）在低 P 更快
        （索引开销），只在 P=128 写带宽成墙时 coop 才反超（875 vs 827）——真正值钱的是 **16 B store**。
      - **§D 接数据面**：P=16 / EDW=4096，标量 `s/max=1.52`（复现 §48 §E），协作向量化压到 **`co/max=1.32`**
        ——收窄但未翻案；EDW≤256 仍 8.5–15.4×。
      - **§E**：高 P 下 `gen` 慢于纯写（P=64 116 vs 175 GB/s）是 **producer 写与 worker 读争 HBM**。
      含义：给 MPK「**编译期预计算 `FullTaskDesc` + scheduler 只做事件→id 映射**」（`runtime.cc` /
      `persistent_kernel.cuh:1254-1523`）与 DeepGEMM「静态角色 + 编译期常量」以量化理由——**造一趟描述符
      比读回来贵一个量级**，且要吃生产者 SM。判据 C1–C4、仍差 C5 一步。**本轮新增实验**；P60-1 新留尾巴
       （§49.7）：**TMA bulk store（SMEM→global）作更宽写载体未测**（最值得做的下一条）、高 P 读写争抢
       未单独量化、合成描述符、未与 DeepGEMM SM100 对跑。

### P61 实验扩展 43：TMA bulk store 作写载体（`docs/04-experiments.md` §50）

- [x] **P61-1** 补掉 P60-1（§49）§49.7 明留的**「最值得做的下一条尾巴」**——「`uint4`（16 B）是寄存器
  store 的上限；**TMA bulk store**（`cp.async.bulk.global.shared::cta`）理论可把「写」接上近 HBM 整机写
  带宽（§49 coop 到 P=128 已 875 GB/s），但须先在 SMEM 造好（SMEM 搬运/预算是否划算未测）」。
  见 `docs/04-experiments.md` **§50**，源码 `src/dyn_producer_tma.cu`、脚本
  `harness/dyn_producer_tma.sh`、原始输出 `harness/dyn_producer_tma.out.txt`（目标卡 H100/sm90a、**固定空闲
  device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、`T=256`、
  `grid=P+G=132=NSM` 全共驻、min-of-7）。新增 `gvar 3`（batch 粒度协作生成到 SMEM 暂存槽 →
  `cp.async.bulk` 整批 11 KB 落 global）、`mode 6 sgen`（只生成到 SMEM 隔离暂存代价）、`mode 7 pure-big`
  （每 `NB` 个 batch 合成一次大块 store）。六 Section 核心结论：
  - **§A**：四载体（标量/v4/coop/TMA）× P∈{1,2}、static/sched/gen **逐位一致**（`got=404219327039195`）
    ——跨 async proxy 的 store→load 握手（`wait_group 0` + `fence.proxy.async.global` + release/acquire）语义透明。
  - **§B（headline）证伪 §49 假设**：static=0.1698 ms/815 GB/s；TMA store 的 `gen` **几乎等于标量**
    （`tma/s` 0.72–0.96，P=1 35.14 vs 36.61 ms），比 v4 慢 2–4×、比 coop 慢 1.37–1.82×——
    **TMA store 接不上读带宽**。
  - **§C 纯写**：P=128 标量 206.5 / v4 812.4 / coop 881.7 / **tma(serialized，正确握手) 472.6** /
    **tma(rolling) 722.5** GB/s。**任何 P 下 TMA 都不超过 16 B 寄存器 store**（roll/v4 0.44–0.89）。
  - **§C2 拆代价**（P=128）：只生成到 SMEM（`sgen`）**766 GB/s** → rolling store **722**（几乎白送）→
    serialized store **471**（正确握手的每批 `wait_group 0` 砍 39%）。两跳（reg→SMEM→global）上限 766。
  - **§C3**：serialized 路径的 `wait_group.read` 深度 `tw=1/2/3` 在 P=128 = 475.9/471.5/473.0 GB/s
    **完全相同**（每批已 `wait_group 0`，深度旋钮失效）。
  - **§C4 粒度**：大块 store 值钱——P=128 从 NB=1（11 KB）722.5 抬到 NB=8（90 KB）**827.8 GB/s**，
    但仍 **0.94× coop（881.5）**、与 v4 打平，**从未超过**；`NB=16` 超 H100 opt-in 上限做不了。
  含义：**「现场生成描述符」这条路的写端天花板，即使接上最宽写载体也是 ~828（rolling）/472（正确握手）**，
  远低于 worker **读**端的 809–3000 GB/s——给 MPK「编译期预计算 `FullTaskDesc` + scheduler 只映射」
  与 DeepGEMM「静态角色/编译期常量」以**更强**量化理由。判据 C1–C4、仍差 C5 一步。**本轮新增实验**；
  P61-1 新留尾巴（§50.7）：~~**「正确握手 + 流水」的组合未测**（FIFO commit + 发布 head 前只
  `wait_group (win-1)`，理论把 472 拉回 722–828 且保持正确——最值得做的下一条）~~（**已由 P62-1/§51 补掉，
  结论：预测被证伪**）、高 P 读写争抢未量化、store 粒度受 SMEM 限（跨 block 协作 stage 未测）、合成描述符、
  未与 DeepGEMM SM100 对跑。

### P62 实验扩展 44：TMA store 的「正确握手 + 流水」（`docs/04-experiments.md` §51）

- [x] **P62-1** 补掉 P61-1（§50）§50.7 明留的**「最值得做的下一条尾巴」**——「serialized 的 472 来自『每批
  `wait_group 0`』这一过保守握手，一个**流水但正确**的形态（FIFO commit + 发布第 j 批 head 前只
  `wait_group (win-1)`、head 落后 win 批）理论可拉回 rolling 的 ~722–828 且保持正确」。
  见 `docs/04-experiments.md` **§51**，源码 `src/dyn_producer_tma_pipe.cu`、脚本
  `harness/dyn_producer_tma_pipe.sh`、原始输出 `harness/dyn_producer_tma_pipe.out.txt` + ncu 归因
  `harness/dyn_producer_tma_pipe_ncu.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、`T=256`、`grid=132` 全共驻、min-of-7）。
  唯一新变量 = 发布握手（serialized win=1 vs 窗口 win≥2，载体固定 TMA store）；另用一族**哨兵口径**
  （`win<0`，**故意不正确**）拆代价。核心结论（**§50.7 预测证伪 + §50 归因修正**）：
  - **§A**：static/win=1/win=2/3/4 **逐位一致**（`got=404219327039195`）——发布滞后 `win-1` 批后跨 proxy
    可见性仍正确。
  - **§B（有 worker）窗口只压 2.2–3.7%**（P=1 209.99×→203.04×、P=16 14.67×→14.13×、P=64 5.75×→5.63×），
    离 v4/coop 仍差 1.6–2.4×。**§C 纯写 P=128：ser 453.7 → pipe 462.8（+2%）→ rolling 733.2 / v4 827.8**，
    **远追不回**。**§E 深窗**（`sb`=win=16，SMEM 180 KB）也只 **468.3（+1.6%）**——**饱和极快，窗口不是旋钮**。
  - **§F（headline 归因，P=128 纯写）**：ser 453.6 / pipe2 461.9 / **nowait 486.3**（拆完成等待，仅 +7%）/
    nofence 484.9 / **nobp 664.3**（拆背压+head，**+46%**）/ **skipbp 664.5**（**只**拆背压）/ **skiphd 664.8**
    （**只**拆 head）/ rolling 733.1 / sgen 719.3 / v4 825.6。⇒ **§50 归因的「每批 `wait_group 0`」不是主因**
    （拆掉只 +7%）；**真正贵的是「每批发布一次跨 SM 进度」**（`st.release head` + 背压门 `ld.acquire tail`），
    且**两者同时存在才昂贵**（单拆任一个都回 664）——是**协议簿记开销、不是 DMA 语义开销**。
  - **ncu**：DRAM 写字节相同（115.8 MB），正确握手 **barrier 停顿 31.19% vs 11.85%**、并多出
    `long_scoreboard`/`membar` 停顿。
  - 含义：**写端正确握手的瓶颈不在 store 引擎、不在完成等待，而在逐批报进度**——正是 MPK「编译期预计算
    `FullTaskDesc` + scheduler 只做事件→id 映射、批量/环形摊薄一次 release」绕开的那本账；**「预计算 + 读」
    是描述符通路唯一可行形态**的结论更强。判据 C1–C4、仍差 C5。**本轮新增实验**；P62-1 新留尾巴（§51.8）：
    「只拆一个就够快」的**组合交互未钉到 SASS/依赖链一级**（开放问题）、合成描述符、纯写口径无 worker、
    未与 DeepGEMM SM100 对跑。

### P63 实验扩展 45：TMA store 握手的三维正交归因（`docs/04-experiments.md` §52）

- [x] **P63-1** 补掉 P62-1（§51）§51.8 明留的**开放问题**——「§51.6 量出『只拆背压（`skipbp`）或只拆
  发布（`skiphd`）都回 664、两者同时才 454』，但未到 **SASS/依赖链一级给出因果**（可能是 loop-carried
  依赖阻止软件流水）」。**唯一新变量 = 归因编码**：把「发布握手」从 §51 的**单整数互斥编码**换成
  **六个正交布尔位**（`WA` 完成等待 / `BP` 背压 / `HD` 发布 / `REL` head release↔relaxed /
  `ACQ` tail acquire↔relaxed / `FENCE` proxy fence），并补**逐相位 `clock64` 计时**与 **SASS 反汇编**。
  见 `docs/04-experiments.md` **§52**，源码 `src/dyn_producer_tma_dep.cu`、脚本
  `harness/dyn_producer_tma_dep.sh`、原始输出 `harness/dyn_producer_tma_dep.out.txt`（目标卡 H100/sm90a、
  **固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、
  `T=256`、`grid=132` 全共驻、min-of-7 + `cuobjdump -sass`）。核心结论：
  - **§A 正确性**：正交重编码的正确档 `ser`(mode1) / 摊薄背压(mode11) / 摊薄发布(mode13) /
    两者摊薄(mode12) 与 static **逐位一致**（`got=404219327039195`）——摊薄是**合法**进度协议。
  - **§B（headline 一）§51.6 的「只拆一个就够快」是编码假象**：§51 的 `nobp=(win<=-3)` **吞掉**
    `win=-4(skipbp)/-5(skiphd)`，那两行**实际都等于 `nobp`**。正交编码给出真实 L 型加性分解（P=128 纯写）：
    store-only 地板 **630**；**只加 release 发布 → 521**；**只加背压会合 → 527**；**两者同加 → 455**；
    **再叠完成等待 → 437**。**`ld.acquire`→relaxed 完全无效**（435≈ser）、`fence` 免费（517≈skipbp）、
    **`st.release`→relaxed 收回 +13.7%（497）**——**贵的是 release store，不是 acquire/fence**。
  - **§B2**：摊薄「发布」比摊薄「背压」更值钱；两者一起摊薄 `S=32` 达 **569**（**+30%**）但**仍差地板
    630 的 10%**（摊薄减频、不能归零）；`S` 单调但饱和（`S=8→32` 仅 +3.5%）。
  - **§C × P**：结论在 `P∈{16..128}` 逐档不变、**近似线性**（单 producer SM 瓶颈）。
  - **§D（headline 二）逐相位 + SASS 钉死因果**：`gen`=3426 cycles/batch 恒定（固定成本）；`bp` 段
    背压时 ~840–907（`skipbp` 时 43，**主因是 `__syncthreads`/`BAR.SYNC` 而非 acquire**）；`pub` 段
    `ser`=1359 / `nowait`=927 / `relhead`=432 / `skiphd`=65。反汇编 `dptd_kernel` 见发布路径原样
    `MEMBAR.ALL.CTA → MEMBAR.ALL.GPU → ST.E.64.STRONG.GPU`——**`st.release.gpu` 落成两条 GPU 级 MEMBAR
    + 强序 store**，每批把该 SM 访存流水 order 一次。⇒ 写端瓶颈 = **每批 release 的 MEMBAR + 会合 barrier**，
    正是 MPK「预计算 `FullTaskDesc` + scheduler 只映射 + 批量/环形摊薄一次 release」绕开的那本账
    （`persistent_kernel.cuh:1254-1523`；`docs/06` §5.3）。判据 C1–C4、仍差 C5。**本轮新增实验**；
    P63-1 新留尾巴（§52.8）：SASS 计数是整 kernel 而非限定基本块、合成描述符、纯写口径无 worker、
    未与 DeepGEMM SM100 对跑。

### P64 实验扩展 46：写端握手归因 × 写载体（`docs/04-experiments.md` §53）

- [x] **P64-1** 补掉 P63-1（§52）§52.8 明留的尾巴——「**换栈（如 v4/coop 协作写，见 §49/§50）会改变它，
  故本节比值不能外推到寄存器写载体**」。**唯一新变量 = 写载体**（同一 §52 骨架 + 同一六位正交握手
  编码、同 `DW=88`/`K=32`/`D=2`/`SB=4`/`T=256`/`grid=P`、纯写 `mode 9`、min-of-7）：
  carrier 0 = **TMA bulk store**（SMEM 暂存 → `cp.async.bulk`，= §52）；carrier 1 = **寄存器 copy-out**
  （SMEM 暂存 → 全 block `uint4` load/store）；carrier 2 = **寄存器直写**（不落 SMEM，每线程生成
  `uint4` 直写环 = §49/§50 的 v4 形态）。见 `docs/04-experiments.md` **§53**，源码
  `src/dyn_producer_carrier_dep.cu`、脚本 `harness/dyn_producer_carrier_dep.sh`、原始输出
  `harness/dyn_producer_carrier_dep.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2）。
  核心结论：
  - **§A 正确性**：三载体正确档（`ser` / `stridedbp`）与 static **逐位一致**（`got=404219327039195`）
    ——换写载体不改变握手可见性契约。
  - **§B/§B2（headline）§52 的 L 型加性、release 主导在**三载体上逐档同形****（排序三列一致：
    `reltail≈ser < relhead≈relboth < skipbp≈skiphd < nobp`；`reltail` 全 ≈1.00 = acquire 免费；
    `relhead` 全 ≈1.10–1.13 = release 收 ~10–13%）。**但绝对数值处处不同、快载体相对罚更狠**：
    地板 `nobp` = **regD 721.6 > tma 609.9 > regS 588.7**，`ser` = **regD 455.7 > tma 427.5 >
    regS 390.0**，`ser→nobp` 落差 **29.9%(tma) / 33.6%(regS) / 36.8%(regD)**（§52.8 的「比值不能外推」
    被证实——但方向明确、排序不变）。**寄存器 store 只在直写不经 SMEM 时优于 TMA**；经 SMEM 暂存
    （regS）反不如 TMA（588.7 < 609.9）。
  - **§C × P**：三载体逐 P 同序（regD>tma>regS）、近似线性。
  - **§D 摊薄**：对三载体都有效且都饱和（S≥16 +1–2%）；regD 在所有 S 恒最快（S=32 达 639.8，比 tma
    高 16%）。
  - **§E（headline 之二）逐相位 `clock64`**：载体差异落在 `gen`/`store` 此消彼长——tma/regS 的 `gen`
    ≈3.5k cycles（协作写 SMEM）、regD 跳过 SMEM（`gen`=254）把生成折进 `store`（2933），净地板
    regD 3666 < tma 4442 < regS 4602；协议代价落 `pub`（release 收回 tma≈854 / regS≈535 / regD≈683
    cycles）。**握手税绝对值三载体相近（~2.0–2.4k cycles/batch），占比随地板抬升而变大** ⇒ 快载体
    不能靠「更快」摊掉协议，反而暴露更高占比——给 MPK「预计算 `FullTaskDesc` + 批量/环形摊薄一次
    release」以**更强**理由。判据 C1–C4、仍差 C5。**本轮新增实验**；P64-1 新留尾巴见 §53.9
    （纯写无 worker、regD 的 store 段含生成计时、`DW/K/D` 未扫、合成描述符、未与 DeepGEMM SM100 对跑）。

### P65 实验扩展 47：写端握手归因的端到端形态（`docs/04-experiments.md` §54）

- [x] **P65-1** 补掉 P64-1（§53）§53.9 明留的尾巴「**纯写口径**（`mode 9`、`grid=P`、无 worker）：
  只量 producer 写端；**带 worker 的端到端归因未在本节跑**（§52.8 另一条尾巴）」，把 §52/§53 的
  写端握手归因放回**有消费者**的流水，回答「有界环绑定时 §53 的握手税还看得见吗、谁在临界路径」。
  见 `docs/04-experiments.md` **§54**，源码 `src/dyn_producer_carrier_e2e.cu`、脚本
  `harness/dyn_producer_carrier_e2e.sh`、原始输出 `harness/dyn_producer_carrier_e2e.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`P=32 producer + G=100 worker`（grid=132
  全共驻）、`DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=1`、`T=256`、min-of-3）。
  新增 **mode 21 = 动态生产者(gen) + G worker + 可选数据面**（worker 用 TMA bulk load 取描述符并执行：
  控制面折叠 + `uint4` 数据面流式读 + R 轮压舱）；复用 §52 六位正交编码与三写载体。核心结论：
  - **§A 正确性**：mode21 `ser` 与无生产者 `static` **逐位一致**（`got=404219327039195`）。
  - **§B（headline 一）有界环绑定（`CAPp`=16 批）时，背压×容量的乒乓往返主导（~20×）**：`BP=1`
    的档全部塌到 ~5 GB/s（vs 纯写同协议 110 GB/s），而 `relhead`/`reltail` 与 `ser` 几乎重合
    ——**内存序税被完全盖过**；`BP=0` 的 `skipbp`/`nobp_wa` 不被环窗卡住（105/101 GB/s ≈ 纯写）。
    塌陷开关精确是**背压门**，不是 release/acquire、也不是载体。
  - **§B2（headline 二）环容量不绑定（`CAP=NDESC`）时 §53 归因复现但衰减**：`reltail≈ser < relhead <
    nobp_wa < skipbp < strall8`（`acquire` 免费、`release` 主因），但幅度从纯写的 **1.44×** 降到端到端
    **1.17×（EDW=0）/1.12×（EDW=256）**——worker 侧与写侧部分重叠把固定缝摊掉一部分。
  - **§C 临界路径**：`BP=0` 的端到端与纯写同量级（producer 写侧就是临界路径）；`BP=1` 端到端恒
    ~4–5.5 且**与 P 几乎无关**（瓶颈从写吞吐换成环窗往返）→ 对 MPK 固定大队列 / DeepGEMM 容量上界 +
    warmup wave / 「批量取 task」以**端到端**量化理由：**把窗开够大/批量取，比重写单条 release 重要一个量级**。
  - **§D 载体**：`BP=1` 时载体差异被环窗抹平；`BP=0` 时复现 §53 的 `regD>tma>regS` 排序。
  - 判据：`static`/`ser`/`nobp_wa` 均 C1–C4、**仍差 C5**（与 §47–§53 同档）。踩坑：**后台覆盖 `/tmp`
    二进制重跑致 `nvcc` ETXTBSY 静默失败**（读到旧二进制，`strings` 分辨不出）——改用新路径编译+前台运行；
    worker 跨步分配 + `D>1` 预取会跨 producer 子环边界**死锁**，端到端同步对照取 `D=1`。**本轮新增实验。**

### P66 实验扩展 48：worker 取 task 的预取深度 D × 分配形态（`docs/04-experiments.md` §55）

- [x] **P66-1** 补掉 P65-1（§54）§54.8 明留的两条尾巴——「worker 取 task 环深固定 `D=1`：**预取深度
  是 §41/§42 的独立变量**」与坑②「**跨步分配 `bb=w+t*Gw` 的 `D>1` 预取跨 producer 子环边界 →
  有界环死锁**」：把 §54 端到端流水里被钉死的「worker 取 task」维度放开，量 (1) 预取深度 `D` 能否把
  §43「每批串行缝」藏住、抬高端到端吞吐，(2) **分配形态**（strided `bb=w+t*Gw` vs contig
  `bb=w*chunk+t`）在动态生产者 + 有界环下谁是安全形态。见 `docs/04-experiments.md` **§55**，源码
  `src/worker_prefetch_e2e.cu`、脚本 `harness/worker_prefetch_e2e.sh`、原始输出
  `harness/worker_prefetch_e2e.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `P=32 producers + G=100 workers`（grid=132 全共驻）、`DW=88`=352 B、`NDESC=393216`（138.4 MB
  ≫ L2）、`K=32`、`T=256`、min-of-3；新增 `wassign` 维度与 worker 侧守门狗）。核心结论：
  - **§A 正确性**：宽松环下 mode21 `ser` × {strided,contig} × `D∈{1,2,4}` 与 static **逐位一致**
    （`got=404219327039195`）——`D`/分配在环不紧张时不改数值。
  - **§B（headline 一）`D` 的纯性能收益很小且立刻饱和**：宽松环端到端 `D=1→2` 仅 +2–3%
    （EDW=0 85.6→88.1、EDW=256 68.8→70.3）、`D≥2` 饱和；**端到端由 producer 写侧主导**（§54）。
  - **§C（headline 二）接入动态生产者 + 有界环后 `D>1` 不可得**：紧环 `CAPp=16` 批下 **strided
    `D>1` 死锁**（`ABORT`，`got=0`；钉死 §54 坑②），**contig `D=1..8` 全部数值 WRONG（静默错）**
    却"更快"（8.9 vs 正确 `ser` 4.8 GB/s）——错误形比死锁形更隐蔽；唯一正确形态是 **strided + D=1**
    （4.8 GB/s，复现 §54 §C 的 `e2e_ser`）。
  - **§E 机制隔离**：contig 错在 **per-producer 的 `tail` 是「前缀计数器」**——当 worker 边界与
    producer 边界不对齐（`chunk≠bp`）时一个 producer 被相邻两个 worker 并发消费、高 batch 先被消费
    → `tail` 越过未消费/在飞 slot → producer 覆写。实测 split(`chunk<bp`) WRONG/split(`chunk>bp`)
    WRONG/aligned(`chunk==bp`，本机 `G=P=66`) OK；strided 三种都对（近似前缀序）。
  - **§D worker 侧地板（无生产者 mode0）**：`D` 的纯上限 = **`D=1→2` +19%（EDW=0 557.9→663.2）/
    +7%（EDW=256 215.8→230.9）**，`D=2` 精确饱和——§43 的每批取 task 缝在 worker 侧 `D=2` 可消。
  - **含义**：**`D` 不是自由的性能旋钮，而是被环协议锁死的自由度**（性能上限小 + 端到端拿不到 +
    有生产者时语义不可得）。要让 `D>1` 可用必须改环协议——把「前缀 `tail`」换成**每槽独立回收/就绪**
    （MPK per-worker ring / DeepGEMM ring generation 逐槽发布）或保证「一个 producer ↔ 一个
    consumer」拓扑；给 MPK「固定大队列 + 批量取 + 每 worker 独立环」以量化理由。判据 C1–C4、仍差 C5。
    **本轮新增实验。** P66-1 新留尾巴（见 §55.8）：合成描述符、死锁边界未闭式扫描、contig 错是
    「共享前缀 tail 协议」性质（换 per-slot 回收可正确）、未与 DeepGEMM SM100 对跑。

### P67 实验扩展 49：共享前缀 vs 每槽回收的环协议（`docs/04-experiments.md` §56）

- [x] **P67-1** 补掉 P66-1（§55）§55.8 明留的两条尾巴——「**contig 的静默错是『共享前缀 `tail` 协议』
  的性质，不是『连续分配』固有的**——换 per-slot 回收协议后连续分配可正确」与「**死锁/错误的具体
  边界未做闭式扫描**」。把 §55 的**共享前缀**（prefix head/tail）有界环换成**每槽独立回收/就绪**
  （perslot `ready`/`reclaim`，对齐 DeepGEMM Mega Gate ring generation、`scheduler/mega_gate.cuh:13-39`），
  在同一骨架 / 同一工作 / 同一分配形态上对跑。见 `docs/04-experiments.md` **§56**，源码
  `src/perslot_ring.cu`、脚本 `harness/perslot_ring.sh`、原始输出 `harness/perslot_ring.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`P=32 producer + G=100 worker`（grid=132
  全共驻）、`DW=88`=352 B、`K=32`、`T=256`、min-of-3）。核心结论：
  - **§A 宽松环**：两协议、两分配、任意 `D` 的 `sink` 全部 = 参考（协议不动数值）；`static` 298 GB/s
    vs 带 producer ~82 GB/s = 写侧地板。
  - **§B 紧环 `CAPp=16`（§55.8 预测证实）**：`prefix+strided+D>1` 死锁、`prefix+contig` 静默错；
    **`perslot+contig` 在 `D=1..8` 全部 OK**——换 per-slot 回收后连续分配**不再错**。但 contig 正确档慢
    （~3–3.5 vs strided 17 GB/s）：解锁的是**正确性**不是性能。
  - **§C 边界对齐**：`prefix` 下只有「无 producer 被拆」（`chunk==bp`，`66/66`）才 OK、两种 split 都
    WRONG；`perslot` 下两种 split 都 OK——错的载体是**前缀计数器**，不是「对齐」。
  - **§D 解锁阈值（新闭式）**：`perslot+contig` 需 `CAPp ≥ D`；`perslot+strided` 需 `CAPp ≥ bp`
    （=128、**与 `D` 无关**，比 prefix 的 `CAPp≈32` 更严）；`prefix+strided+D=2` 在 `CAPp=32` 即 OK。
    ⇒ per-slot 把 strided 的门**收紧**到「必须恰好回收该槽」。
   - **§E 协议开销**：宽松环 per-slot 比 prefix 只慢 ~1.3–1.4%（写侧地板掩盖）——代价在**解锁阈值**、
     不在协议本身。判据 C1–C4、仍差 C5（与 §47–§55 同档）。**本轮新增实验。**

### P68 实验扩展 50：每 worker 独立环（一对一对齐）解锁 prefix+D>1（`docs/04-experiments.md` §57）

- [x] **P68-1** 把 §55.7 / §56.7 **两处都明留、却从未实现**的那条「出路」做出来——「**保证『一个
  producer ↔ 一个 consumer』的拓扑，或每 worker 独立环**」（= MPK 快照 `per_worker_queue_len=8192`
  的真实形态）。关键观察：**当 `P == G` 时全局 contig 分配（`bb=w·chunk+t`）的 `chunk==bp`、`own==w`，
  即每个环恰好一个 producer、一个 consumer**，前缀计数器精确 → `D>1` 应可用；本节把该配置跑出来。
  见 `docs/04-experiments.md` **§57**，源码 `src/perworker_ring.cu`、脚本 `harness/perworker_ring.sh`、
  原始输出 `harness/perworker_ring.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `DW=88`=352 B、`K=32`、`T=256`、min-of-3/5；`pair` 用 `P=G=66`、`shared` 用 §55/§56 的 `P=32 G=100`，
  两者 `grid=132`；prefix 协议固定，唯一变量 = worker→batch 映射）。核心结论：
  - **§A 宽松环（`CAPp=bp`）全 OK**；**clean control（同 `P=G=66`）只换映射**：`shared-strided` 90 vs
    `pair` 130 GB/s = **1.45×**（`pair` 去掉了「每个 `tail[p]` 被多 worker 并发 `atom.add.release`」的
    跨 worker 前缀计数器争用）。**即每 worker 独立环本身就不只为 `D`，还是吞吐的 1.45×。**
  - **§B（headline）紧环 `CAPp=4`**：**`pair` 在 `D=1..4` 全部 OK（129/131/85）**，而 `shared-strided`
    `D≥2` 全 `ABORT`、`shared-contig` `D=1/2` 静默 `WRONG`——**对齐成「一消费者一环」后 prefix+`D>1`
    从死锁/静默错变为正确**；§55.7/§56.7 出路预测**证实**。
  - **§C 解锁阈值（新闭式）**：`pair` 正确性阈值精确 **`CAPp* = D`**（`CAPp<D` 死锁）；但 `CAPp==D`
    只 ~85 GB/s、`CAPp≈2D` 才回 ~130（等号处背压往返暴露）——**`CAPp* = D` 是下限，实用取 `CAPp≈2D`**，
    呼应 §41/§42/§54「环深只需 2 槽」。
  - **§D/§E `D` 的纯性能增量很小**：`pair` `D=1→2` **+6%**、`D≥2` 饱和——解锁的是**可用性**（死锁/静默错
    →正确）而非「`D` 越大越快」，与 §43/§54 一致。三拓扑同 grid 同工作：`pair` ~1.6×，但此格混入
    `P=66 vs 32` 的 producer 并行度（干净对照见 §A 的 1.45×）。
  - **代价/局限**：本节靠 `P==G` 获得「一消费者一环」，故 `grid=2G`（producer 侧随 G 膨胀）；真实
    MPK/DeepGEMM 是「少 producer（scheduler）× 多 per-worker 环」——一生产者顺序填多环、每环仍单消费
    单生产，producer 侧不膨胀（**本节未实现，记为局限**）。判据 C1–C4、仍差 C5（与 §47–§56 同档）。
    **本轮新增实验**；P68-1 新留尾巴（§57.8）：未实现「少 producer 轮转填多 per-worker 环」、
    `CAPp*=D` 只单配置验证、合成描述符、未与 DeepGEMM SM100 对跑。

### P69 实验扩展 51：少 producer（scheduler）× 多 per-worker 环（`docs/04-experiments.md` §58）

- [x] **P69-1** 补掉 §57.8 明留的**头号尾巴**——真实 MPK 形态是「**少 producer（scheduler）× 多
  per-worker 环**」：一个 producer 顺序/轮转填多条环、每条环仍单生产者单消费者（前缀精确），
  **producer 侧不随 G 膨胀**（§57 的 `pair` 靠 `P==G`、`grid=2G`，是本机一直缺的配置）。
  见 `docs/04-experiments.md` **§58**，源码 `src/sched_perworker_ring.cu`、脚本
  `harness/sched_perworker_ring.sh`、原始输出 `harness/sched_perworker_ring.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`K=32`、`T=256`、min-of-3/5）。
  机制：**把「环所有权」与 block 解耦**——G 个 worker 各持环 `w`；**PS 个 producer** 各持有连续一段
  worker 环、**段内轮转**填；每条环仍单生产者单消费者、前缀精确；**`grid=PS+G`**（`PS==G` 退化为
  §57 `pair`）。另加 **计算重量旋钮 `CW`**（worker 每批纯 ALU 延迟、不改 `sink`），模拟「计算型 task」。
  核心结论：
  - **§A/§B/§E 正确性与解锁是「拓扑」性质、与 producer 数无关**：宽松环 `PS∈{1..32}×D∈{1..8}` **32/32 OK**；
    紧环 `CAPp=4` 下 `D≤4` 全 `OK`、`D=8>CAPp` 全 `ABORT`（**每个 PS 都一样**）；阈值精确 = per-ring 的
    `CAPp≥D`（§E，`PS=4` 与 `PS=16` 同形）。**§57 的解锁不是 `P==G` 特有的**。
  - **§C 少 producer 是共享瓶颈**：`CW=0` 吞吐 **≈4.2×PS GB/s 精确线性**（单 producer SM 标量描述符写
    上限 ~4.2 GB/s，对上 §49 的 ~3.8），故轻负载下**必须 ≈G 个 producer**（= §57 `pair` 被迫 `P==G` 的物理原因）。
  - **§D（headline）task 越重、需要的 producer 越少**：中位数饱和判据 `PS*` 随 `CW` 单调下降——
    `CW=0→PS*≈G`、`CW=2^14→16`、`CW=2^16→8`、`CW=2^18→≈4`；重 task 下 **`PS*:G≈1:25`**，
    **正对上 MPK 的 4 scheduler SM : 128 worker SM（1:32）**（§5 行 1157–1164、§6.2 行 1460）。
  - **§F 与 §57 `pair` 对照 + codegen 警告**：本 harness 的 `PS=66`（逻辑=§57 `pair`）测得 **153–164 GB/s**，
    §57 通用实现（同一逻辑映射）测得 **~133 GB/s**；`ncu` 显示差异在 DRAM 流量/L2 命中与屏障停顿
    （逻辑工作、`sink` 完全相同，源于通用 worker 热循环的运行时分支/除法 codegen）。故 **绝对 `descGB/s`
    对编译产物敏感、跨源码只可定性对照**，本节所有比较都在同一源码族内。
  判据 C1–C4、仍差 C5（与 §47–§57 同档）。**本轮新增实验**；P69-1 新留尾巴（§58.10）：`CW` 是纯 ALU
  延迟（真实 task 是访存+计算混合，曲线形状可信、拐点只对本微基准）、`grid≤132` 才全共驻故 `PS` 上限
  压到 32、合成描述符、未与 DeepGEMM SM100 对跑。

### P70 汇总 2：in-kernel scheduler 设计手册（`docs/11-in-kernel-scheduler-design.md`）

- [x] **P70-1** 把 P42–P69（`docs/04-experiments.md` §31–§58）这一长串**调度器微基准**收口成一张
      **可操作设计手册**（补齐 `docs/05-summary.md` 只覆盖到 P6、后续 28 节调度器实验散落在 400 KB
      的 `docs/04` 里、缺一份「调度器内部机制」层综合的缺口）。见
      `docs/11-in-kernel-scheduler-design.md`（纯资料汇总，无新实验）。结构：§0 瓶颈从「算」转向「管」；
      §1 参考架构六部件（拓扑/ready 结构/入队协议/内存序/取 task 载体/描述符来源）；§2 集中式 vs
      去中心化（§9/§15）；§3 ready 结构与死锁的两个来源——图静态量 `B(π)`/`CAP*=G*=B(π)+1`（§31–§34）
      与排队量 `CAP_safe+G≈W_max+1`（§35），及事件驱动 `G*≡1`（§34）；§4 队列协议（§36 release 税、
      §37 连续放行、§38 批量/warp 读）；§5 取 task 载体×预算×流水（§39–§46：大 K+D=2、预取缝固定长度、
      数据面认总线程/SM、控制面认 K）；§6 描述符来源（§47–§54：预计算 vs 现场生成差 27×、release
      MEMBAR 税、端到端环窗主导）；§7 环协议与预取深度 `D`（§55–§58：`D` 被协议锁死、一消费者一环、
      `PS*:G≈1:25~1:32` 对上 MPK 4:128）；§8 成本模型速查表（12 条实测数字）；§9 设计决策表
      （14 条决定 → 推荐/代价/证据）；§10 对 MPK/DeepGEMM 逐条印证；§11 局限（codegen 敏感、合成
      描述符、判据普遍停 C4、未对 SM100）。配套博客 post `mega-kernel-12-sched-design`（weight 13）
      并更新 `mega-kernel-00-index` 目录第 12 项。**本专题任务清单此前全完成，本轮为其新增的汇总增量。**

### P71 实验扩展 24：真实 HBM 数据面下的 producer:worker 配比（`docs/04-experiments.md` §59）

- [x] **P71-1** 补掉 §58.10 明留的尾巴——「**`CW` 是纯 ALU 延迟，只拉长 task、不产生真实 HBM
      流量**；真实 task 的『重』是访存+计算混合，`PS*`-vs-`CW` 曲线**形状可信、精确拐点只对本微基准
      成立**」。给 §58 的 worker 加一条**真实 HBM 数据面**（每 batch 用 `__ldcs` 绕 L2 流式读 `MD` 个
      `uint4`），把「喂饱 G 个 worker 所需 producer 数 `PS*`」随 task 重量的曲线从**纯 ALU** 推进到
      **访存主导**。见 `docs/04-experiments.md` §59，源码 `src/sched_perworker_ring_mem.cu`、脚本
      `harness/sched_perworker_ring_mem.sh`、原始输出 `harness/sched_perworker_ring_mem.out.txt`
      （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`K=32`、`T=256`、`G=100`、
      宽松环、`D=2`、min-of-3）。五 Section：
      - **§A 数据面不改数值**：`MD` 计入 `sink` 后仍**逐位对 host 参考**（0/16/128 KB 全 `OK`）。
      - **§B headline**：`PS*` 随**访存重量 `MD`** 单调下降——`MD=0/8192→PS*≥32≈G`、
        `MD=32768（512 KB）→PS*=16（1:6.25）`、**`MD=131072（2 MB）→PS*≈4（1:25）`**；
        **与 §58 纯计算档同族、对上 MPK 的 1:32**——「重 task 少 producer」对**访存型 task 同样成立**。
      - **§C 对照复现 §58**：`CW=2^16→PS*=16`、`CW=2^18→PS*≈4`（本节 `NB=1000` vs §58 `NB=8192`，
        总工作量不同故差一档、形状一致）。
      - **§D 两笔重量同向叠加**：`CW=65536` 单独 `PS*=16`，叠加 `MD=32768` 后 `PS*=8`。
      - **§E 控制面不争 HBM**：achieved 数据面带宽随 `PS` 只升不降（650→952 GB/s），producer 写描述符
        （~5 GB/s）可忽略——**控制面瓶颈在每批 `st.release` 的 MEMBAR/会合，不在 HBM 带宽**。
      判据：内核 **C1–C4**、仍差 C5（与 §47–§58 同档）。踩坑：`prepare_data` 早于 `ensure()`、须惰性
      分配 `g_dref`/`g_data`。局限：数据面只到 ~950 GB/s、**未把 HBM 推到饱和**，真饱和争用未测。
      **本轮新增实验**；补齐 §58.10 唯一未覆盖的「重量维度」，与 §47–§58 合成调度器**取 task/环协议/
      配比**链条的闭合。

### P72 实验扩展 25：把数据面推到 HBM 饱和（`docs/04-experiments.md` §60）

- [x] **P72-1** 补掉 §59.6/§59.9 明留的**唯一尾巴**——「本微基准的数据面只到 ~950 GB/s、**未把 HBM
      推到饱和**，故『描述符写与数据面读不争 HBM』只在远低于峰值区间成立；真饱和争用未测」。给 §59
      的数据面加**内存级并行度 `U`（ILP 展开因子）**：§59 是单累加器依赖式读（每线程 1 个在飞 load），
      本节改成 `U` 个独立累加器 + `U` 个在飞 load，把数据面推到饱和，再在**饱和点重问 §59.6 的开放
      问题**（加 producer 是否拖慢数据面）。见 `docs/04-experiments.md` §60，源码
      `src/sched_dataplane_sat.cu`、脚本 `harness/sched_dataplane_sat.sh`、原始输出
      `harness/sched_dataplane_sat.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
      `DW=88`=352 B、`K=32`、`G=100`、宽松环、`D=2`、min-of-3）。六 Section：
      - **§A ILP 不改数值**：`sink` 逐位对 host 参考（`U=1/4/8/16` 全 `OK`；整数异或和可重排）。
      - **§B headline**（`T=256`）：`U=1→2` 把数据面从 **926→1490 GB/s**（在飞 load 翻倍），
        `U=16` 达 **2218**——**§59 的 ~950 GB/s 是「单累加器依赖链」的产物，不是数据面固有**。
      - **§E `T×U` 矩阵**：`T=1024,U=4` 与 `T=512,U=16` 都到 **~2840 GB/s**（H100 峰值 3.35 TB/s 的
        **85%**、§45 饱和值 ~2983 的 95%）；饱和由「每 block 在飞字节 `T×U`」定；`T=1024,U=16`
        `too many resources requested`（→ 记 `UNSUP` 跳过）。
      - **§C HBM 争用（闭合 §59.6）**：饱和配置（`T=1024,U=8`）下 `mode0`（**无 producer** 天花板）
        **2792 GB/s** vs `mode1`（`PS≥8`）**2763–2837**——**饱和点上加 producer 仍不拖慢数据面**，
        §59.6「不争用」从非饱和区间**扩展到饱和区间**。`PS` 唯一作用是喂满：`PS=1` 896、`PS=2`
        1742、`PS=4` 2687、**`PS≥8` 饱和**（`PS*≈8`，比 §59 的 4 高，因数据面快了 ~3×）。
      - **§D 对照**（同 `T=1024`、`U=1`）：也到 ~2.4 TB/s（≈2.5×§59），说明 §59 低吞吐是 `T`、`U`
        两旋钮都小；`U` 的边际在**低 `T` 档**最显著。
      - **§60.7 对应**：MPK worker 用 TMA bulk（高 MLP），本就不在 §59 低 MLP 区间；4:128 落在
        「task 又重又快」区间（`PS*` 更小）。
      判据：内核 **C1–C4**、仍差 C5（与 §47–§59 同档）。踩坑：`T=1024,U=16` 的**发射失败**
      （`too many resources`）被延迟到 `cudaEventSynchronize`，首版误报 `WRONG`（`ms≈3 µs`）——
      加 `cudaGetLastError()` 确认并改记 `UNSUP`。局限：只到峰值 ~85%（`T=1024` 占用 50%、
      再堆 `T×U` 受寄存器/SM 限），真 100% 峰值争用仍留一丝未测。**本轮新增实验**；补齐 §59.6
      「不争用」结论在饱和区间的适用性，闭合 §47–§59 的**取 task / 环协议 / 配比 / 数据面 MLP** 链条。

### P73 实验扩展 52：数据面到真 HBM 峰值 + 控制面争用（`docs/04-experiments.md` §61）

- [x] **P73-1** 补掉 P72-1（§60）§60.9 明留的**唯一尾巴**——「数据面只到 HBM 峰值的 ~85%
      （`T=1024` 单 block/SM = 50% 占用、再堆 `T×U` 受寄存器/SM 限），**真 100% 峰值下的争用
      仍留一丝未测**」：把「数据面带宽天花板」从**调度器骨架**（§60 的 per-worker 环 + TMA 取
      task + 折叠）里**抽出来单独量**，回答 (Q1) 数据面能否到真峰值、靠**占用率**还是**栽体**，
      (Q2) 真峰值下描述符写是否开始争 HBM。见 `docs/04-experiments.md` **§61**，源码
      `src/dataplane_peak.cu`、脚本 `harness/dataplane_peak.sh`、原始输出
      `harness/dataplane_peak.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
      每配置总读 ≈ 4 GiB、HBM3 峰值取 3.35 TB/s）。相对 §60 的三个新变量：(a) 数据面与调度器
      **解耦**成纯带宽微基准；(b) 数据面**栽体**新增 **TMA bulk load→SMEM→折叠**（MPK worker
      真实形态）与寄存器 `__ldcs` 对跑；(c) 争用用**同核并发 writer 块**在峰值配置下扫写速率
      找转折点。核心结论：
      - **§A 两栽体折叠逐位对 host 参考**（reg/tma 均 `OK`）。
      - **§B/§C（headline 一）数据面到 ~3219 GB/s = HBM3 峰值的 96%**（reg、`T=1024`、
        2 block/SM、`U=4`），比 §60 的 2840 快 1.13×；**缺的旋钮是占用率**（§60 卡在 1 block/SM
        = 50%），**ILP 与占用率是替代品**（256 线程/SM 需 `U=4` 才到 2718；满占用率下 `U` 从
        1→4 只 3198→3219）。`__launch_bounds__(1024,2)` 把 `U=4` 的 40 regs 压到 32、1→2 block/SM，
        `T=1024` 单栽体即达 3208。
      - **§D（headline 二）TMA bulk 栽体到 ~3075 GB/s（reg 峰值的 96%、峰值的 92%）**，但
        **用更少寄存器**（`DC=2048` 用 32 regs→2 block/SM，reg `U=4` 要 40）；分块 `<1024` 塌
        （`DC=512` 只 2118）。**TMA 的价值在省寄存器而非更快**，对上 MPK worker 用 TMA 搬 tile。
      - **§E 占用率/寄存器**：reg `U≥4` 因 >32 regs 掉到 1 block/SM——**正是 §60 卡 85% 的机制**。
      - **§F（headline 三）真峰值下「描述符写不争 HBM」结论变弱**：读带宽随写速率**超线性**
        下降——`writeGB/s=21.7`（占读 0.7%）读降 **7.9%**、`78.6`（2.6%）降 **16.7%**（HBM
        **bus turnaround**，非字节比账）。**与 §59.6/§60.5 不矛盾而是限定区间**——「不争用」是
        **读余量**的函数而非字节比的函数（§60 在 85% 有 15% 余量时吸收，§61 在 96% 几乎无余量
        时 7% 即现）；生产描述符流量 ~15 GB/s（~0.5%）落在读降 3–7% 档。F3 对照证「SM 槽位挤占」
        是另一笔叠加代价。
      判据：微基准 **C1**、多算子数据面形态；未含设备内调度故**不完整满足 C2–C5**（立场是给
       §47–§60 的 C1–C4 骨架配一张数据面能力表）。**本轮新增实验**；P73-1 新留尾巴（§61.10）：
       纯读微基准（无环协议/取 task/task 计算）、writer 是合成写、合成数据分区、未与 DeepGEMM
       SM100 对跑。

### P74 实验扩展 53：端到端骨架推上占用率 + 夹层成本分解（`docs/04-experiments.md` §62）

- [x] **P74-1** 补掉 P73-1（§61）§61.10 明留的两条尾巴——(a)「纯读微基准是**能力上界**、非端到端
      吞吐，与 §60 端到端的差即『调度/取 task/折叠的**夹层成本**』」(b)「争用测试的 writer 是**合成写**、
      非真实描述符/环协议」。把 §61 的**占用率旋钮**（多 block/SM + `__launch_bounds__(TB,MINB)`）
      搬回 §60 的**完整骨架**，在**同一二进制**内用三档 `mode`（2=data-only / 0=+fold / 1=full）
      把夹层逐层剥开。见 `docs/04-experiments.md` **§62**，源码 `src/e2e_saturation.cu`、脚本
      `harness/e2e_saturation.sh`、原始输出 `harness/e2e_saturation.out.txt`（目标卡 H100/sm90a、
      **固定空闲 device 3**、nvcc 13.2、`DW=88`=352B、`K=32`、`D=2`、`NB=2048`、`MD=65536`=1MiB/batch，
      全图数据面总读 ≈ 2.0 GiB，HBM3 峰值取 3.35 TB/s，min-of-3）。核心结论：
      - **§A 三档 mode 逐位对 host 参考**（`OK`）；**§F** `MINB` 把 `numRegs` 压到 **32**、`T=256`
        达 **8 blk/SM**（满占用率）、`U=4` 无 spill（`U=8` spill 72B）。
      - **§B 同二进制数据面天花板 ~3156 GB/s = 峰值 94%**；抬占用率两条等价路径：`T=256` 加大 `G`
        （100→1040：974→3138），或 §60 的 `T=1024,U=8`（`G=100` 即 3063）——**ILP 与占用率是替代品**。
      - **§C 端到端 full 骨架 G 越大反而越慢**：`T=256` 路线峰值仅 2058（G=528）、G=1040 降到 1768；
        `T=1024` 路线 G 从 100 加到 1040 从 2831 跌到 1910。**§60 的「少而大」worker 块最优**
        （`G=100,chunk=21,T=1024,U=8` = **2831 GB/s = 85%**，精确复现 §60 的 2840）。
      - **§D/§D2（headline）夹层分解**：`G=1040`：data-only 3139 → +fold 3105 → full 1761
        （**fold −1.1% 免费、环+producer −43.9%**）；§60 配置：3078 → 3118 → 2913
        （**fold +1.3% 免费、环+producer −6.6%**）。**§61.10 把 gap 猜成「取 task/折叠」是错的**，
        真正的成本是**控制面（环协议+producer）**且随 `G` 恶化。
      - **§E/§G 定位**：G=1040 扫 PS 1→32（343→**2500**，仍升）；固定 G/PS 扫 `chunk` 2→8 只 +3.6%
        （**每 task 开销不是瓶颈**）——成本来自**「少 producer 轮转服务多 per-worker 环」的扇出**
        （对上 §58 `PS*:G≈1:25~1:32`）。**故 megakernel 的正确设计压力是调度器服务速率，不是纯并行度。**
      判据：骨架 **C1–C4**、仍**差 C5**（静态序列取 task、无固定角色 worker/scheduler）。**本轮新增实验**；
      P74-1 新留尾巴（§62.10）：只到数据面 94%、合成描述符/环、`G`↔`chunk` 耦合、co-residency 未严格、
      未与 DeepGEMM SM100 对跑。

### P75 实验扩展 54：producer 服务模型（每环服务成本 vs 环数）（`docs/04-experiments.md` §63）

- [x] **P75-1** 补掉 P74-1（§62）§62.10 明留的尾巴「`G` 与 `chunk` 天然耦合（`chunk=⌈NB/G⌉`）……
  **更细的 producer 服务模型（每环服务成本 vs 环数）留待后续**」：把 §62 **完整端到端骨架**里
  的 **producer 写端单独抽出来**（无 worker / 无背压 / 无数据面），用**唯一控制变量**——每个
  producer 连续服务的**环数 `wpp`**（固定每 producer 总批数 `Btot=chunk*wpp`）——直接量「每环
  服务成本」，再在同一二进制里加回 **plain-load consumer** 做对照，把 §62.7 口头归因的
  「producer 轮转服务多 per-worker 环的扇出」钉到 producer 侧（或证伪）。见
  `docs/04-experiments.md` **§63**，源码 `src/producer_service_model.cu`、脚本
  `harness/producer_service_model.sh`、原始输出 `harness/producer_service_model.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、批=11264 B（`DW=88`=352 B/行）、`T=256`、
  min-of-20）。六个 Section：
  - **§A 正确性**：producer 写值（block 归约）**逐位**命中 host 参考（`sink=ref=1485236534821600`）；
    §F 带 consumer 档 `sink=2×ref`（写值+读回值）逐位命中。
  - **§B（headline）固定每 producer 工作量，吞吐不随环数下降**：`PS=32`、`Btot=64`，`wpp=1→64`
    吞吐 **158.2→166.0**（略升后平台）、`cyc/batch` **4336→4126**（±2.5%）——**不存在可测的『每环
    服务成本』**（`wpp=1` 略慢是同一条 `head` cache line 连续 64 次 release 的轻微串行化）。
  - **§C** `wpp=1` 扫 `Btot`：`cyc/batch` **恒 ~4335–4357**、吞吐只随 `Btot` 小幅爬升=一次性启动摊薄
    ——producer 成本结构是**纯每批**。
  - **§D** `PS=1→32` 线性（5.2→165.1 GB/s），每 producer 独立 ~5.2 GB/s、无跨 producer 争用。
  - **§F（对照）** `PS=8`、`NB=2048`、扫 `R=128→1024`：**producer-only 恒 42.6–42.7**、
    **+plain-load consumer 只降 ~12% 且同样平**（37.3–37.4）——consumer 只加一笔**固定**开销、
    **不随环数放大**。
  - **结论/修正**：producer 服务成本 = **每批一次 `fence+st.release`+两次 `__syncthreads`**
    （`cyc/batch≈4135–4357`≈**~2.1 µs**、单 SM 描述符天花板 **~5.2 GB/s**，对上 §52 的 release 税）；
    **§62.7 的「producer 环扇出」归因被隔离证伪**——§62 端到端「`G` 越大越慢」只能住在 **worker 侧**
    （TMA 取 task / 折叠 / 数据面）或其**占用率/共驻**（§62.3 的上限）里。对 MPK/DeepGEMM 的意义：
    event→task 直取 / AOT 预入队 / ring generation 位复用绕的是**每批固定 release 税**，不是「多环
    服务成本」；`PS*:G≈1:25~1:32` 的压力来自**单 producer ~5 GB/s 上限**而非每环一罚。判据：写端
    单 measure、不含 TMA/折叠/数据面，**不完整满足 C1–C5**。局限：consumer 是 plain-load、合成
    描述符、寄存器写载体、无背压、未与 DeepGEMM SM100 对跑。**本轮新增实验。**

### P76 实验扩展 55：worker 侧复杂度阶梯 × 环数 G 归因（`docs/04-experiments.md` §64）

- [x] **P76-1** 补掉 P75-1（§63）§63.9 明留的尾巴——「consumer 是 **plain-load**，故『加 consumer
  仍平』只能证『环数本身无成本』，**不能**直接证明 §62 的 `G` 依赖来自 TMA 路径——需把
  TMA+折叠+数据面按 §62 加回」。把 §62 骨架里的 **worker 侧**放上一条**复杂度阶梯**（`wmode`
  0=plain 寄存器读环 / 1=+TMA 取 task 进 SMEM / 2=+HBM 数据面 / 3=仅数据面），在**同一二进制、
  同一占用率**（`T=256,MINB=8,U=4`=8 blk/SM）下扫 worker 环数 `G`，把 §62 的「`G` 越大越慢」
  钉到**哪一层 worker 机制**。见 `docs/04-experiments.md` **§64**，源码
  `src/worker_side_attribution.cu`、脚本 `harness/worker_side_attribution.sh`、原始输出
  `harness/worker_side_attribution.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `DW=88`=352 B/task、`K=32`、`D=2`、`PS=8`、`NB=2048`、`MD=65536`=1 MiB/batch、min-of-3）。
  核心结论：
  - **§A** 四档阶梯 `sink` 逐位对 host 参考（全 `OK`）。
  - **§B headline（固定 NB，chunk 随 G 缩小）复现 §62**：`tma+d` 1740@G=1040（§62 报 1768，<2%）；
    相对 `data-only`（3076）跌 **−43.4%**；`plain` ≡ `tma`（无数据面时都被 `PS=8` producer ~31 GB/s
    钉住）——**§63 的「consumer 平」是被 producer 盖住、不等于免费**。
  - **§C headline（固定 chunk=4，`NB=4·G`）拆出闭式**：把开销折成**每 block**——**环消费者 prologue
    ≈ +0.35 µs/block（plain 就付）**、**TMA 取 task 额外 ≈ +0.3 µs/block（tma 合计 ~0.69）**，
    **跨 `G=100..1040` 恒定**（plain +0.48/0.44/0.34/0.30/0.35；tma +0.70/0.71/0.66/0.63/0.69）——
    **§62 的 `G` 依赖 = worker 侧每 block 一笔固定 prologue**，`chunk` 越小越显形，TMA 把它翻倍；
    `chunk=4` 下 `G=1040` full 从 §B 的 1740 抬到 **2083**。
  - **§D** 固定 `G=264` 扫 `MD`：`MD=0` 时 plain ≡ tma，数据面一上来 **plain 比 tma 快 4–6%**
    （+0.29 µs/block，与 §C 同）——**TMA 的代价只在快路径上可见**。
  - **§E** 四档同占用率（`numRegs=32`、8 blk/SM），比较不受占用率混淆。
  判据 C1–C4、仍差 C5（与 §47–§63 同档）。**本轮新增实验**；P76-1 新留尾巴（§64.9）：合成描述符/
  数据面、载体未含 `cp.async`、co-residency 未严格、未与 DeepGEMM SM100 对跑。

### P77 实验扩展 56：worker 侧取 task 载体第三种——`cp.async`（`docs/04-experiments.md` §65）

- [x] **P77-1** 补掉 P76-1（§64）§64.9 明留的尾巴「载体是寄存器读（plain）/ TMA，**未含 `cp.async`**
      （§42 的第三种载体）；`TMA vs plain` 的 +0.3 µs/block 只对本形态成立」：把 MPK worker 的
      **原生**取 task 载体（16 B `cp.async` 单缓冲进 SMEM，快照 `persistent_kernel.cuh:1063-1078` 的
      `cp_async_wait<0>`；`load_smem`=`cp.async.ca.shared.global.L2::128B`，`tasks/common/copy_sm80.cuh:44-57`）
      加进 §64 的 worker 侧复杂度阶梯，在**同一骨架、同一占用率**（`T=256,MINB=8,U=4`=8 blk/SM）下与
      plain / TMA 对跑，回答「**cp.async 的每 block prologue 是否低于 TMA**」。见
      `docs/04-experiments.md` **§65**，源码 `src/worker_carrier_cpasync.cu`、脚本
      `harness/worker_carrier_cpasync.sh`、原始输出 `harness/worker_carrier_cpasync.out.txt`（目标卡
      H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B/task、`K=32`、`D=2`、`PS=8`、
      `NB=2048`、`MD=65536`=1 MiB/batch、min-of-3）。新增 `wmode 4=cpasync` / `wmode 5=cpasync+d`；
      六档阶梯。核心结论：
      - **§A/§B/§D/§F 六档 `sink` 逐位对 host 参考**（全 `OK`）——16 B `cp.async` 单缓冲取 task +
        SMEM 折叠 + HBM 数据面 + 有界环都不动数值。
      - **§B（headline 一）cp.async ≡ plain、TMA 随 `G` 掉队**：固定 `NB=2048` 扫 `G`，`ca/plain`
        **0.99–1.01×**；`tma/plain` 从 `G=132` 的 0.969 掉到 `G=1040` 的 **0.813×**。`MD=0` 时三载体
        逐点相等（都 producer 钉住）。
      - **§C（headline 二）TMA 的 +0.3 µs/block prologue 是 TMA 专属**：固定 `chunk=4`、折成每
        block `(full−data)/G`——`tma−plain` 恒 **+0.25…+0.32 µs/block**（复现 §64），而
        `cp.async−plain` 恒 **−0.04…+0.005 µs/block（≈0）**。即 **cp.async 的每 block prologue 与
        plain 寄存器读同档、不付 TMA 那笔**——TMA 的 setup（`mbarrier.init ×D`+`arrive.expect_tx`+
        首次 `cp.async.bulk`+`mbar_wait`）**不是「用 SMEM 取 task」的通性**。
      - **§D** 固定 `G=264` 扫 `MD`：`MD≥32768` 起 cp.async ≈ plain（`MD=131072`：2795.7 vs 2791.6）、
        TMA 掉 ~5–9%（2654.0）——载体差异只在有数据面的快路径上可见。
      - **§F** 固定 `G=132` 扫 `chunk`：`chunk=1` 三载体相等；`chunk≥2` 起 cp.async ≥ plain > TMA，
        且都随 `chunk` 增大向 `data-only` 收敛。
      含义：**解释 MPK 为何用 16 B `cp.async` 而非 TMA bulk 取 task**——常驻 worker 每 SM 付一次
      fetch setup，cp.async 的 setup 比 TMA 轻 ~0.3 µs/block；TMA 的价值在载入 task 单体 tile
      （§39/§42/§61/§62），不在取 task 描述符。与 §42「K≤32 cp.async 反超」在同一 `K=32` 口径下
      吻合。判据 C1–C4、仍差 C5（与 §47–§64 同档）。**本轮新增实验**；P77-1 新留尾巴（§65.9）：
      合成描述符/数据面、cp.async 为单缓冲（深环预取未在本阶梯内测）、co-residency 未严格、
      未与 DeepGEMM SM100 对跑。

### P78 实验扩展 57：worker 侧取 task 载体第四种——`cp.async` **深环**（`docs/04-experiments.md` §66）

- [x] **P78-1** 补掉 P77-1（§65）**§65.9 明留的尾巴**「**cp.async 是单缓冲**（快照形态），
      **未做 `D` 深环预取**（§42 的 cp.async 深环已量『预取增益全在 D=1→2』）」：把 cp.async 从
      **单缓冲**升级成 **`DCA` 槽软件流水深环**（`cp.async.wait_group (DCA-1)`、尾批 `wait_group 0`
      兜底、逐槽复用 `__syncthreads()`），接进 §64/§65 的端到端 worker 阶梯。见
      `docs/04-experiments.md` **§66**，源码 `src/worker_carrier_cpasync_ring.cu`、脚本
      `harness/worker_carrier_cpasync_ring.sh`、原始输出 `harness/worker_carrier_cpasync_ring.out.txt`
      （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`K=32`、TMA `D=2`、
      `PS=8`、占用率固定 `T=256,MINB=8,U=4`、`NB=2048`、`MD=65536`、min-of-3；新增 `WMODE 6=ca-ring`
      / `7=ca-ring+d`，`WMODE`/`DCA` 均编译期 template）。核心结论：
      - **§A 正确性**：单缓冲 + 深环 `DCA=2/4/8` 共 9 档 `sink` 逐位对 host 参考（全 `OK`）。
      - **§B/§C/§G headline**：**深环没有任何一个点快过单缓冲**——`ca-single ≡ plain`（±1%）；
        `DCA=2` 掉到 ≈ TMA 档（`caD2/plain` 1.08→1.35× 随 `G` 增大）、`DCA=4` 更差（1.23→1.61×）。
        §G（`G=132,chunk=8`）扫 `DCA=1..8` **单调变慢**：1206→1136→1041→893 taskGB/s。
      - **§E 两笔净成本**：(a) 逐批**槽复用 `__syncthreads()`**；(b) **SMEM 足迹随 `DCA` 线性增长**
        拉低占用率（`DCA=2`→8、`DCA=4`→5、`DCA=8`→2 blk/SM）。
      - **含义**：§42 的「预取增益全在 D=1→2」**不迁移**到端到端阶梯——352 B/task 取数已被 HBM
        数据面/producer 就绪等待藏住（§65），深环**无缝可回收**，只剩足迹与同步；再次确认 MPK 用
        **16 B `cp.async` 单缓冲**（`persistent_kernel.cuh:1063-1078`）是经过取舍的选择。判据 C1–C4、
        仍差 C5。**本轮新增实验**；P78-1 新留尾巴（§66.9）：与 §65 存在**跨二进制 codegen 偏移**
        （本版 plain `localB=8` 有 8 B 溢出、`G≤132` 慢 ~1.6×，故只采信同族内相对排序）、小 `G`
        受同机争用抖动、合成描述符/数据面、未与 DeepGEMM SM100 对跑。

### P79 实验扩展 28：§66 的「跨二进制 codegen 偏移」归因（`docs/04-experiments.md` §67）

- [x] **P79-1** 补掉 P78-1（§66）**§66.9 明留的第一条尾巴**「**跨二进制 codegen 偏移**：本版内核
      plain 路径 `numRegs=32/localB=8` 有 8 B 局部栈溢出，`G≤132` 比 §65 慢 ~1.6×，故只采信同族相对
      排序」——把它从**诚实标注**推进到**归因**：8 B 溢出从哪来？它是不是 §66 plain 慢的原因？抬掉它
      能否收回？见 `docs/04-experiments.md` **§67**，源码沿用 `src/worker_carrier_cpasync_ring.cu`
      （只新增 `MINB=6/4` 实例化与 `regs` 子命令）、脚本 `harness/carrier_reg_budget.sh`、原始输出
      `harness/carrier_reg_budget.out.txt`（目标卡 H100/sm90a、**自动选最闲 device**、nvcc 13.2、
      `DW=88`=352 B/task、`K=32`、`PS=8`、`T=256`、`U=4`、`NB=2048`、`MD=65536`=1 MiB/batch、
      min-of-3；把 `__launch_bounds__(256,MINB)` 的 `MINB∈{8,6,4}` 当**寄存器预算旋钮**，`MINB=8→32` /
      `6→42` / `4→64` regs/线程）。核心结论：
      - **§A 溢出归因**：`MINB=8` 下 plain / TMA 各 `localB=8`（溢出），`cp.async` 单缓冲与深环
        `localB=0`；放宽 `MINB` 后 plain 到 **`MINB=4`（52 regs）**、TMA 到 **`MINB=6`（40 regs）** 才
        干净 ⇒ **8 B 溢出是 32-reg 上限打的，不是「运行期分支选择载体」**（源码本就是 `if constexpr` +
        编译期 template）。**载体寄存器耐压序**：`cp.async` 单缓冲/深环（≤32）> TMA（~40）> plain
        直接读（~52）——**直接寄存器读最耗、SMEM 载体最省**，给 MPK 用 16 B `cp.async` 单缓冲取 task
        （`persistent_kernel.cuh:1063-1078`）再添一个**寄存器侧**理由。
      - **§B 修正因果（headline）**：溢出**不是** §66 plain 慢的原因——`MINB: 8→6→4` 把 plain 在
        `G=100/132` 提速 **1.64×/2.43×、1.59×/2.14×**（收回 §66.9 的 ~1.6×），但**一直 `localB=0`
        的 `ca1` 与 plain 逐点重合（<1%）且同步变化** ⇒ 主导量是**寄存器预算 → 数据面 ILP**，不是那
        8 B；与 §60/§61/§72「数据面认每 SM 总线程/ILP，ILP 与占用率是替代品」一致（**降占用率 8→4
        blk/SM 反把 1 MiB/batch 数据面从 931 抬到 2263 GB/s**）。
      - **TMA 反向 + 排序稳定**：`tma` 在 `G=100/132` 随 `MINB` 下降**变慢**（1149→927→926），在
        `G=528/1040` **变快**（1791→2391→2420）——§66.9「TMA/深环在大 `chunk`/大 `MD` 反超 plain」是
        **配置/占用率耦合**的产物。而**任一固定 `MINB` 内** `ca1 ≈ plain`、`caR` 一致更差——**§66「单缓冲
        `cp.async ≡ plain`、深环无收益」在三个 `MINB` 下都成立**（尾巴补掉）。
       - 判据 **C1–C4**、仍差 C5（与 §66 同档）。**已据此修正 ROADMAP「已知坑」**（那条把溢出归到
         「运行期 `if(wmode==…)` 分支」的说明是**错的**）。**本轮新增实验。**

### P80 实验扩展 59：寄存器预算 × 占用率的单变量解耦（`docs/04-experiments.md` §68）

- [x] **P80-1** 补掉 P79-1（§67）**§67.4 明留的尾巴**「`MINB` 同时改**寄存器预算**与**占用率**两个
      物理量，**未把二者解耦**；『ILP 主导』是排除法 + 推断，**非单变量隔离**」——用**两个独立旋钮**
      把二者拆开：**占用率**由 `grid=numSM·O` 钉死（每 SM 恰好 O 块，`resident=min(O,reg-limit)`），
      **寄存器预算**由 `__launch_bounds__(256,MINB)` 钉死（`MINB∈{8,6,4,2}`）。见
      `docs/04-experiments.md` **§68**，源码 `src/reg_occ_decouple.cu`（纯读 `uint4` 数据面微基准，
      载体 = 寄存器 `__ldcs` / `cp.async` 单缓冲，编译期 template）、脚本
      `harness/reg_occ_decouple.sh`、原始输出 `harness/reg_occ_decouple.out.txt`（目标卡 H100/sm90a、
      **自动选最闲 device**、nvcc 13.2、`T=256`、`numSM=132`、`smem/SM=233472 B`、总读 2 GiB、
      `reps=3`、min-of-3）。核心结论：
      - **§A 正确性**：`reg U=4` 与 `ca DC=2048` 折叠 `sink` 逐位等于 host 参考（均 `OK`）。
      - **§B（headline Q1，单变量隔离）同占用率下多给寄存器让数据面快 1.70×**：占用率钉死后，寄存器
        预算仍是一阶变量。`O=1`（`resident=1` 三档完全相同）下 `MINB=8`（32 regs）**1668.7 GB/s**
        vs `MINB≥6`（40–44 regs）**2847.2** = **1.70×**；`O=2` 1.21×、`O=4` 1.03×。即**数据面认的是
        每线程在飞 load 数（寄存器预算撑起的 ILP），不是占用率本身**——把 §67 的推断升级成隔离实验。
        （`§F` 实测 `numRegs`：MINB=8→**32**、6→40、4→44、2→40，均 `localB=0`；真正分界是 32 vs ≥40。）
      - **§C（Q2）寄存器预算与占用率是替代品**：32-reg 预算靠占用率 **O=1→4 补回 1.83×**（1670→3060）；
        ≥40-reg 预算 **O=2 即饱和**（2844→3163→3166）。这给出 §60/§61/§72「ILP 与占用率是替代品」的
        **机理**（都在供「每 SM 在飞 load 总量」）。
      - **§D 交叉验证**：用 **SMEM padding** 钉占用率（grid 固定 `numSM·8`）与 grid 钉法**同量级
        （差 3–6%）**；但 SMEM-pad 不是干净旋钮——`pad=24 KB` 占用率仍 8 却掉 4.5%（H100 统一 L1/SMEM，
        多要 SMEM 挤掉 L1），故 headline 采信 grid 钉法。
      - **§E 载体交叉验证**：`cp.async` 单缓冲在固定 O 下扫 `MINB` **逐点相同（<0.2%）**——其自然寄存器
        用量（31–32）低于 32-reg 上限，对 `MINB` **完全免疫**（§67 的静态 `numRegs` 表 → 运行时无代价）。
        同一 `MINB=8` 下寄存器载体掉 1.03–1.70×、`cp.async` 一点不掉 ⇒ **SMEM 载体在低寄存器预算下严格
        占优**，给 MPK 用 16 B `cp.async` 单缓冲取 task 再添一个运行时理由。
      - 判据 **C1–C5 全不满足**（纯数据面微基准，无环/无调度器，既非单 kernel 常驻也非多算子/去中心化）；
      仅借它把 §67 的成本侧论断隔离到单变量。局限已诚实标注（合成 `uint4`、block 粒度占用率未扫 `T`、
      `O=1` 绝对值受同机 HBM 争用、未用 `__maxnreg__`、未对 SM100）。**本轮新增实验。**

### P81 实验扩展 60：用 `__maxnreg__` 做精确寄存器上限（`docs/04-experiments.md` §69）

- [x] **P81-1** 补掉 P80-1（§68）**§68.7 明留的尾巴**「『占用率』固定为 **block 粒度**（`T=256`，
  **未**单独扫 `T`）」与「**未用 `__maxnreg__`** 单独压寄存器；`MINB` 只是 ptxas 的 ceiling/hint」：
  用 CUDA 12.4+ 的 **`__maxnreg__(R)`**（PTX `.maxnreg`）把每线程寄存器上限设成**精确**值 `R`，
  与占用率（`grid=numSM·O` 与每块线程数 `T` **各自独立**）解耦。见 `docs/04-experiments.md` **§69**，
  源码 `src/maxnreg_decouple.cu`、脚本 `harness/maxnreg_decouple.sh`、原始输出
  `harness/maxnreg_decouple.out.txt`（目标卡 H100/sm90a、**自动选最闲 device**、nvcc 13.2、`numSM=132`、
  载体 = 寄存器 `__ldcs` uint4 + `U=4` 路 ILP、总读 2 GiB、`reps=3`、min-of-3）。三问三答：
  - **§A 正确性**：`mr R=32/64` 折叠 `sink` 逐位等于 host 参考（`562601267102948`，均 `OK`）。
  - **§B（Q1 headline）精确 `R`，`O=1`、`T=256`**：`__maxnreg__` 是**精确**上限——`R≤44` 时
    `numRegs≡R`（32/36/40/44），`R≥48` 时 ptxas 自然需求就是 44（封顶失效）。§68 的 **1.70×**
    （32 vs ≥40）**如实复现**，但**分界不是二元跳变而是连续斜坡**：32→36→40 = 1668.6→**2137.1**→2841.8
    GB/s（36-reg 已收回 ~41% 的差），40 起饱和——**修正 §68.3「跨过 40 即饱和」的二元措辞**。
  - **§C（Q2）精确 `__maxnreg__` vs `__launch_bounds__` 在匹配 `numRegs`（32/40/44）下对跑**：
    **逐位等价（<0.3%）**——决定吞吐的是寄存器**数**本身、不是压出它的属性；证 §68 用 `MINB` 做旋钮
    无属性特有偏差。
  - **§D（Q3 headline）固定 `O=1`、固定 `R` 只扫 `T`**：`R=32` 从 `T=128→1024` 单调升
    **899→3136 GB/s（3.49×）**，`R=32,T=1024`（3136）≈ `R=64,T=1024`（3191，仅差 1.7%）——
    **每线程寄存器预算与每 SM warp 数是替代品**（都供「每 SM 在飞 load 总量 `≈O·T·U`（×每线程 MLP）」），
    把 §68 §C 的 `(寄存器预算, 占用率 O)` 替代性**推广到 `T` 维度**；饱和点在 HBM 面 ~3200 GB/s。
  - **§E** 精确 `R` 下扫 `O` 复现 §68 §C（32-reg 靠 O=1→4 补 1.83×、≥40-reg O=2 饱和）。
  判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；与 P80-1（§68）合成「寄存器预算 ×
  占用率 × 每块线程数」的单变量解耦链条，并给 ROADMAP「已知坑」补一条 `__maxnreg__` 的语义注记。

### P82 实验扩展 61：`T`×`O` 二维面（`docs/04-experiments.md` §70）

- [x] **P82-1** 补掉 P81-1（§69）**§69.7 明文尾巴**「未测 `T` 与 `O` **同时大于 1** 的二维面
      （`O·T` 联合钉『每 SM 总线程』）」：§68 只动 `O`、§69 §D 只动 `T`，`O·T` 乘积里另一维恒为 1。
      本轮在**固定每线程寄存器预算** `R`（`__maxnreg__(R)`）下同时扫 `T` 与 `O`，走一张 `T`×`O` 面，
      检验数据面是否只由**每 SM 总线程 `O·T`**（×每线程 MLP `U`）这个**合量**决定。见
      `docs/04-experiments.md` **§70**，源码 `src/t_occ_surface.cu`、脚本 `harness/t_occ_surface.sh`、
      原始输出 `harness/t_occ_surface.out.txt`（目标卡 H100/sm90a、**自动选最闲 device**（落 device 2）、
      nvcc 13.2、`numSM=132`、载体 = 寄存器 `__ldcs` uint4 + `U=4`、`grid=numSM·O`、总读 2 GiB、
      `reps=3`、min-of-3）。核心结论：
      - **§A 正确性**：5 组代表 (R,T,O) 折叠 `sink` 逐位对 host 参考（`562601267102948`，全 `OK`）。
      - **§E（headline）等积**：`O·T≥1024` 起**塌缩成单变量**——`O·T=1024` 的 12 个配置
        （`R=32/40/44` × `T=128/256/512/1024`）全落 **3045–3198 GB/s（极差 1.05×）**、`O·T=2048` 的 4 个落
        **2995–3181（1.06×）**——**饱和区吞吐只由 `O·T` 决定，与 (T,O) 拆分、与 32 vs 40/44 regs 无关**。
      - **未饱和区寄存器预算仍是先决条件**：`R=32` 相对 `R=40/44` 在 `O·T=128/256/512` 落后
        **2.16×/1.76×/1.24×**——寄存器预算决定「达到饱和所需的最小 `O·T`」（40/44 在 `O·T≈512` 近饱和、
        32 要到 ~1024–2048，约差一倍，与 32 regs 只撑约一半在飞深度一致）。
      - **同 `O·T` 组内 `T` 越大略快**（块少、块级序言省）：`O·T=2048` 的 `R=32` 行 `T=128/O=16`（2995）
        → `T=1024/O=2`（3181，1.06×），对上 §64/§76 的每 block ~0.3–0.35 µs prologue。
      - **§F 两个天花板**：线程面 2048（仅 `R=32` 可达）、寄存器面受**分配粒度**影响——`R=32→2048 线程`、
        `R=40→1536`、`R=44→1280`（`R=64` 被自然需求封顶成 44 regs，§69 复现）。
      判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；把 §68/§69 的替代性从单旋钮推广到
      二维面，闭合 P80/P81/P82 的寄存器×占用率×线程数链条。P82-1 新留尾巴（§70.7）：`U=4` 固定且与
      寄存器预算耦合、无法测 `O·T>2048`（H100 硬限）、只测连续 `__ldcs` 流、未与 DeepGEMM SM100 对跑。

### P83 实验扩展 62：访存模式 × `O·T` 合量（`docs/04-experiments.md` §71）

- [x] **P83-1** 补掉 P82-1（§70）**§70.7 明文尾巴**「只测**连续/规则**的 `__ldcs` uint4 流，未扫
      strided / 非对齐 / **L2 命中**等访存模式」：把「数据面只认『每 SM 总线程 `O·T`』」这条合量拿到
      **四种访存模式**上复检，并把 §70 的二维面推广到访问模式维度。见 `docs/04-experiments.md` **§71**，
      源码 `src/access_pattern_surface.cu`、脚本 `harness/access_pattern_surface.sh`、原始输出
      `harness/access_pattern_surface.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 2）**、
      nvcc 13.2、`numSM=132`、逻辑总读 1 GiB（uint4 计）、`__maxnreg__(32)`、`U=4`、`reps=3`、min-of-3；
      四模式 `m0 coalesced`（HBM 流式）/`m1 sparse8x`（每 8 读 1，物理 4×）/`m2 scalar4B`（4 B 无向量）/
      `m3 l2res16M`（16 MiB 常驻 L2）；同一逻辑字节数）。核心结论：
      - **§A 四模式折叠 `sink` 逐位对 host 参考**（`OK`）；`ptxas -v`：`m0/m1` 32 regs、`m2/m3` 28 regs、
        均无 spill。
      - **§B（Q1/Q2）合量普适、天花板差一个量级**：coalesced/scalar/l2res 吞吐**都在 `O·T≈1024`
        附近进入平台**（模式无关的通用旋钮），但峰值 `coalesced` **3056** / `scalar4B` **2597** /
        `l2res16M` **23448** / `sparse8x` 逻辑仅 **713**（等效物理 2848，撞 HBM wall）GB/s；
        **低 `O·T` 起始效率随模式差很多**（coalesced 在 `O·T=256` 已 90%，scalar/l2 只 39–45%，抬到
        1024 补 **2.22×/2.43×**）。**`sparse8x` 是唯一例外——从 256 起即平，`O·T` 不是旋钮**（已撞
        物理流量墙）。
      - **§C（headline，合量的裂缝）同一 `O·T=1024` 下 `(T,O)` 拆分**：`coalesced` 极差 **1.04×**
        （复现 §70），但 `l2res16M` **1.47×**（`T=256/O=4` 22385 vs `T=1024/O=1` **15217**）——**L2/
        延迟受限的模式要「多而小」的 block**（隐藏 L2 延迟），与 coalesced 的「少而大」相反；`scalar4B`
        也偏大 `T`（1.15×）。**「只认 `O·T`」只对 HBM 带宽受限严格成立**。
      - **§D `ncu` 归因**（T=256,O=4）：`sparse8x` DRAM 读 **4.29 GB**（物理 4×，128 B 行只用 16 B）；
        `scalar4B` 的 DRAM 字节/sectors 与 coalesced **完全相同**（低速纯**指令发射**）；
        `l2res16M` 几无 DRAM（2.04 MB）。三种「非 HBM」约束（物理流量 4×/发射/L2 延迟）被分开。
      - **含义**：真实 megakernel 的 task 访存模式混杂，「按 `O·T` 铺满机器」是**通用**旋钮但**非万能**
        ——稀疏 task 先撞物理流量墙（应靠 coalescing/TMA multicast/DSM 降流量，P30-1/P31-1）、L2/延迟
        受限 task 要「多而小的 block」（影响 worker 划分）、发射受限 task 靠更多 warp——是 megakernel
        **访存模式异质性**的成本侧约束（与 §8/§10/§14 的资源足迹互补）。
       判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；P83-1 新留尾巴（§71.7）：合成访问
       （stride 固定 8 / 16 MiB / 无向量宽度）、未测 `O·T>2048`、未扫非 8 倍数 strided/非对齐、
       未与 DeepGEMM SM100 对跑。

### P84 实验扩展 63：异构访存模式共驻（`docs/04-experiments.md` §72）

- [x] **P84-1** 补掉 P83-1（§71）**§71.7 明文尾巴**「四模式此前都是**单独整机跑、未测同一 grid 内
      多模式共驻**」（以及「非真实 tile」的一半）：把四种访存模式塞进**同一个常驻 grid**（每 block 按
      `assign[]` 选模式、host 用误差扩散法**交错**分配以保跨 SM 共驻），复检 §71.7 headline 四
      「一个常驻 kernel 内 task 类型越杂、越难一套调度参数通吃」。见 `docs/04-experiments.md` **§72**，
      源码 `src/hetero_access_mix.cu`、脚本 `harness/hetero_access_mix.sh`、原始输出
      `harness/hetero_access_mix.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 2）**、
      nvcc 13.2、`numSM=132`、逻辑总读 1 GiB（uint4 计）、`__maxnreg__(32)`、`U=4`、`reps=3`、min-of-3；
      `pat_mix` 32 regs、0 spill）。核心结论：
      - **§A 混合（4/4/4/4 交错）折叠 `sink` 逐位对 host 参考**（`OK`）；**§B 单模式基线复现 §71**
        （coal 3041 / sparse 710 / scalar 2598 / l2res 25248 GB/s @ `T=256,O=4`）。
      - **§C（headline 一）共驻大体加和**：九种混合的 measured/(时间切片调和平均期望) ∈
        **[0.867, 1.111]**——同 grid 混跑与「各模式轮流独占整机」相差 ±13% 以内。**唯一系统性裂缝 =
        `scalar4B`（发射受限）**：凡含 scalar 且占可观份额的混合恒低 ~13%（coal+scalar 0.867、
        scalar+l2res 0.867），**与 `(T,O)` 正交**（§D 扫遍不可补）；不含 scalar 的都 ≥1（coal+sparse
        1.015、coal+l2res **1.111 超加和**、sparse+l2res 1.030）。**`l2res` 是不争 HBM 的「白吃者」**
        （藏进 HBM 停顿）。
      - **§D（headline 二）异质性税分两类**：`equal 25×4` 1.030、`coal+sparse` 1.023（**无税**，
        被瓶颈模式主导、最佳单一 `(T=1024,O=1)` 即 ≥oracle），但 `coal+scalar` **0.871**（**18% 税**，
        跨所有 `(T,O)` 补不回）。⇒ §71.7「难一套参数通吃」的正确形态：**访存模式之间近似可通吃；
        发射受限 task × 访存受限 task 的同核共驻不可通吃**。
      - **§E（headline 三）稀疏低掺杂「毒化」**：`coal+sparse` 扫 0→100%，`sparse=10–30%` 时
        `meas/exp` **0.855–0.915**（掺一点反而比加和更亏，物理仅 ~2.55–2.67 TB/s 未到墙），`≥50%`
        回到加和（物理稳在 **2.77–2.92 TB/s ≈ HBM 墙**）。**少量稀疏 task 混入大量流式 task 的代价最
        隐蔽**（须先 coalescing/降物理流量或隔离）。
      - **§D ncu**：混合的物理 DRAM 字节 = `Σ_p f_p·phys_p` **逐字节精确**（equal4 1.61 GB=1.50×、
        coal+sparse 2.68 GB=2.50×）——把 `sparse`=物理流量、`scalar`=发射/调度、`l2res`=延迟三约束
        分开钉死。
      判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；P84-1 新留尾巴（§72.8）：合成访问、
      共驻映射是「按 block 交错」而非真实「按 task 类型分区/SM 角色绑定」、`O·T≤2048`、未扫非 8 倍数
      strided/非对齐、未与 DeepGEMM SM100 对跑。

### P85 实验扩展 64：按 SM 角色绑定 vs 按 block 交错（`docs/04-experiments.md` §73）

- [x] **P85-1** 补掉 P84-1（§72）**§72.8 明文尾巴**「共驻映射是『按 block 交错』，非真实 megakernel 的
      『按 task 类型分区/SM 角色绑定』」：把 §72 的**同一组模式**从「按 block 交错」（共享 SM）换成
      「**按 SM 角色绑定**」（每个 SM 只跑一种模式，`mode = mode_of_sm[smid]`），**保持逻辑工作份额
      `f_p` 不变**，判定共驻税/超加和是**SM 内（发射槽共享）还是 SM 间（HBM/L2/全机）**现象，并对照
      `bind`（SM 数 ∝ 份额）与 `bind_bal`（SM 数 ∝ 份额/每SM速率）。见 `docs/04-experiments.md` **§73**，
      源码 `src/hetero_sm_bind.cu`、脚本 `harness/hetero_sm_bind.sh`、原始输出
      `harness/hetero_sm_bind.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、
      nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`__maxnreg__(32)`、`U=4`、min-of-3；`pat_hetero`
      32 regs、0 spill；`bind` 用 O=1/T=1024、每 SM 一 block 独占区、无原子，与 §72 同 32 warp/SM）。
      核心结论：
      - **§A** `interleave`/`bind` 折叠 `sink` 均逐位对 host 参考（`OK`），不同 SM 数 132/132 均匀。
      - **§B** 单模式基线复现 §71/§72（coal 3113.5 / sparse 707.8 / scalar 2902.6 / l2res 16241 GB/s）。
      - **§C（headline 一）`bind ≈ interleave`**：五个混合的 `meas/E` **逐行几乎重合（差 ≤1.1%）**
        ——equal4 1.039/1.033、coal+scalar **0.902/0.898**、coal+sparse 1.030/1.021、
        coal+l2res **1.112/1.113**、scalar+l2res 0.789/0.792。⇒ **共驻税与 `l2res` 超加和是全局
        （SM 间/HBM/L2）现象、不是 SM 内发射槽共享**；**证伪 §72.8 缓解方向 (a)「把发射受限 task 聚到
        一起/SM 隔离」能收回该税**。§F ncu：`bind` 与 `interleave` 的 `dram__bytes_read`(1.07 GB)、
        L1 sector(33567072)、活跃 warp(49.99%)、`inst/cycle`(0.04) **逐项相同**。
      - **§C（headline 二）`bind_bal` 反而更差**：SM 数 ∝ `1/r1` 均分时 `equal4` **0.753**、
        `coal+l2res` 0.854、`coal+sparse` 0.933（仅 `scalar+l2res` 0.964 例外）——因 `sparse`/`l2res`
        **低 SM 数即饱和 HBM/L2**、每 SM 速率对 SM 数**非线性**，「按常数 `1/r1` 分配 SM」前提不成立。
      - **§E** `coal+scalar` 扫 scalar 的 SM 份额：内点最优 sc≈96 仅 **0.935**（仍欠 ~6.5%），
        份额过小/过大都塌——**即便最优分区也收不回税**，再次指向全局机制。
      判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；P85-1 新留尾巴（§73.8）：
      `bind` 用 O=1（未测每 SM 多块角色绑定）、`bind_bal` 用常数每 SM 速率模型（未按「SM 数→速率」
      曲线最优配平）、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100。

### P86 实验扩展 65：SM 分配器「常数速率 vs 实测曲线配平」（`docs/04-experiments.md` §74）

- [x] **P86-1** 补掉 P85-1（§73）**§73.8 明文尾巴**「`bind_bal` 用**常数**每 SM 速率模型（`r1_p`），
  未用「SM 数→速率」曲线，故其差**部分**是模型粗糙而非分区本身；未做按曲线最优配平」：把 §73 的
  **常数斜率** SM 分配（`bind` ∝ 份额 / `bind_bal` ∝ `f_p/r1_p`）对照**实测「SM 数→分区速率」曲线 +
  贪心 makespan 最小化**（`bind_opt`）。见 `docs/04-experiments.md` **§74**，源码
  `src/hetero_sm_curve.cu`、脚本 `harness/hetero_sm_curve.sh`、原始输出
  `harness/hetero_sm_curve.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、
  nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`T=1024,O=1`、`__maxnreg__(32)`、`U=4`；`pat_hetero`
  32 regs/0 spill/8192 B smem）。四模式/三既有映射**完全复用 §71–§73**，**唯一新变量 = SM 分配器**。
  核心结论：
  - **§A/§B** 正确性逐位对 host 参考；单模式基线复现 §73（coal 3104.9/sp 705.4/sc 2883.9/l2 16241.3）。
  - **§B2（新数据）实测速率曲线 `rate_p(n)`**：饱和膝各不相同——**`sparse8x` 在 `n≈32` 即压平**
    （32→132 的 rate 701.6→708.9、每 SM 21.9→5.37），`coalesced`/`scalar4B` 到 `n=96` 才压平，
    `l2res16M` 全程近线性（每 SM 115–192）——§73.5「每 SM 速率非线性」的曲线版。
  - **§C（headline）曲线配平「一半成立、一半是新谬误」**：`bind_opt` 在**近线性混合**上收回并反超——
    `coal+scalar` `bind_bal` 0.865→**0.947**（**反超 `interleave` 0.909**）、`scalar+l2res` 0.969→**0.987**；
    但在**含 HBM 饱和模式（`sparse`）**的混合上**比常数斜率还差**——`coal+sparse` 0.936→**0.664**
    （`bind_opt` 分 `[coal=9,sparse=123]`，把 123 SM 全喂给已饱和的 sparse、饿死 coal），`equal4` 0.757。
    **根因：单跑速率曲线把「加 SM 到已饱和模式」的边际价值估高了——是「常数速率谬误」的另一种坐标。**
  - **§D** 分配明细：`bind_opt` 与 `bind_bal` 在含 `sparse` 行同向且更狠；在 `coal+scalar` 更均衡
    （`[38,94]` vs `[64,68]`）。
  - **§E** `coal+scalar` 扫 scalar SM 份额复现 §73.6 内点最优（sc≈96 → 0.937），`bind_opt` 选点
    `[38,94]` 落在内点、实测 0.947。**§F ncu** `bind` 与 `bind_opt` 的 DRAM 字节/sector/活跃 warp/
    发射率**逐项相同**（同 §73.7，差纯是并发时序）。
  - **含义**：§73.5「必须按曲线配平」细化为**必须按「并发（in-situ）曲线」配平**——分配改变争用、
    争用改变速率，是一个**不动点**；这给「分区角色绑定比交错更难自动调参」再添硬理由，也解释生产实现
    （MPK event queue / DeepGEMM 静态角色 + ring barrier）**为何走「共享 SM + 设备内调度」而非静态 SM 分区**。
    判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；P86-1 新留尾巴（§74.9）：
    **单跑曲线未做 in-situ/迭代不动点**（本节核心发现即它不够，最直接下一步）、曲线只采 14 点、
    `bind` 用 O=1、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100。

### P87 实验扩展 66：SM 分配器「单跑曲线 vs in-situ 迭代不动点」（`docs/04-experiments.md` §75）

- [x] **P87-1** 补掉 P86-1（§74）**§74.9 明文尾巴**（原话）「`bind_opt` 的速率曲线是**单跑**测得
      （本节的核心发现即它不够），**未做 in-situ/迭代不动点**——这是最直接的下一步」：把 SM 分配器的
      速率来源从**单跑（solo）**换成**当前并发下实测（in-situ）**——(a) `bind_insitu`（map=4）**迭代
      不动点**：从 §74 的 `bind_opt`（单跑曲线）出发、用**实测整机 makespan** 做**局部爬山**（挪
      `step` 个 SM、改善 >0.5% 才接受、`step=16→…→1`），只用 in-situ 测量、不需任何速率模型；
      (b) **条件 in-situ 速率曲线**：两模式扫 `n_p`、用设备侧 `clock64` 量**并发下**各分区实测速率。
      见 `docs/04-experiments.md` **§75**，源码 `src/hetero_sm_insitu.cu`、脚本
      `harness/hetero_sm_insitu.sh`、原始输出 `harness/hetero_sm_insitu.out.txt`（目标卡 H100/sm90a、
      **自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`T=1024,O=1`、
      `__maxnreg__(32)`、`U=4`；`pat_hetero` 32 regs/0 spill/8192 B smem）。四种模式/既有四映射**完全
      复用 §71–§74**，**唯一新变量 = 速率来源**。核心结论：
      - **§A** 正确性 `sink` 逐位对 host 参考（`1791753031`，`OK`）；**§B** 单模式基线/SM 曲线复现 §73/§74。
      - **§C（headline 一）in-situ 完全收回 §74 的崩盘**：`coal+sparse` 从 `bind_opt` 的 **0.664 → 1.024**
        （追平 `interleave` 1.032）；`coal+l2res` 0.886→**1.168**、`scalar+l2res` 0.983→**1.030**（均反超
        `interleave`）。**§74 的崩盘确系「单跑曲线把『给已饱和模式加 SM』的边际价值估高」。**
      - **§C（headline 二）两类东西 in-situ 也修不了**：① `coal+scalar` 的 ~5% **全局发射税反不回**
        （`bind_insitu` **0.952** ≈ `bind_opt` 0.953）；② **`equal4` 的角色绑定结构上限**——四模式全混时
        `bind_insitu` **0.789** 仍输 `interleave` 1.043（任何静态 SM 分区都不如「共享 SM + 交错」）。
      - **§D（headline 三）条件 in-situ 曲线 = 病因**：单跑 `solo_coal(n)` 一路涨，但 in-situ `rate_coal`
        直到 `n≈64` 仅 627–1082（并发争用压低）；in-situ `rate_sparse` 随 coal 扩张**单调下降**
        （704→626→592→…→558），overall 峰值在 **`n_coal≈32`**——而单跑贪心选了 `n_coal=9`。**in-situ
        一眼看清「给已饱和 sparse 加 SM 边际价值≈0/为负」。**
      - **§E** `bind_insitu` 把 `coal+sparse` 从 `[9,123]` 纠回 `[33,99]`；**§F** 爬山轨迹 15 次评测即把
        崩盘案 766.5→1172.1（>50%），`coal+scalar` 原地不动（起点已最优）；**§G ncu** 两分配器硬件计数
        **逐项相同**（DRAM 1.07 GB / sectors / 活跃 warp / 发射），差纯是并发时序。
      判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；P87-1 新留尾巴（§75.9）：`bind_insitu`
      是**局部爬山**（`equal4` 两次运行收敛到不同分配，`l2res16M` 对运行序敏感）、条件曲线只 2 条两模式对、
      `n` 步长 16、`bind` 用 O=1、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100。

### P88 实验扩展 67：in-situ SM 分配器的**全局性/可复现性**（`docs/04-experiments.md` §76）

- [x] **P88-1** 补掉 P87-1（§75）**§75.9 头号尾巴**（原话）「`bind_insitu` 是**局部爬山**（可能停在局部
  最优，`equal4` 尤甚——本轮两次运行收敛到不同分配 `[25,32,30,45]`/`[8,84,20,20]`，因 `l2res16M` 的 L2
  驻留对运行序敏感）」：把 §75 的**单起点爬山**升级为**多起点爬山**（`bind_global`，map=5）并做
  **oracle 穷举**，四问——Q1 可复现性 / Q2 多起点 / Q3 oracle / Q4 评价噪声地板。见
  `docs/04-experiments.md` **§76**，源码 `src/hetero_sm_global.cu`、脚本 `harness/hetero_sm_global.sh`、
  原始输出 `harness/hetero_sm_global.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、
  nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`T=1024,O=1`、`__maxnreg__(32)`、`U=4`；`pat_hetero` 32 regs/
  0 spill/8192 B smem，与 §73–§75 同）。四模式/五既有映射**完全复用 §71–§75**，**唯一新变量 = 起点集合
  + oracle 穷举**。核心结论：
  - **§C（Q1，headline 一）单起点 `bind_insitu` 不可复现**：同一 `bind_opt` 起点重复 `N=8` 次，
    `equal4` 收敛到**双模态**——`[8,84,20,20]`（~1465，`meas/E≈0.78`）或 `[25,3x,3x,4x]`（~1899–2035，
    `meas/E≈1.03–1.09`），**spread(max/min)=1.398**；`coal+sparse` 则稳定（1.007，最优面平坦）。
    §75 报的 `equal4=0.789` 正是「卡在局部最优」的低频支。**是否跳出局部最优取决于前几次评测的时序
    噪声**（0.5% 接受阈值附近一次抖动翻转首个 move 方向）。
  - **§D（Q2，headline 二）多起点 `bind_global` 收回局部最优**：`equal4` 从 `bind_insitu` 0.789 →
    **1.039**（追平 `interleave` 1.042）；`coal+scalar` 0.945→0.979（仍差 `E` ~2%）。
  - **§E（Q3，headline 三）多起点追平 oracle 到 ~1%**：四两模式混合穷举 `n_p=1..131`，`bG/org` =
    **1.001/0.996/1.000/0.991**；`org_n` 与 §75 条件曲线峰位一致（sparse 32 / l2res 93 / scalar+l2res 110）。
    **`coal+scalar` 的 `bind_global`（2937）≈ oracle（2935）但都 `<E`（3004）~2.3%**——全局发射税仍在。
  - **§F（Q4）噪声地板 ≈1.0%**（同一分配重复 `R=8`：`spread=1.010`），故 §C 的 1.398 是**优化器路径
    效应、非测量噪声**；但多起点 best-of-M 本身也被 ~1% 噪声影响（§D 1942 vs §F2 2032）——给的是
    「一个好解」不是「确定最优」。**§F2 轨迹**：`bind_opt` 起点反而收敛到较差支（1883），偏置/随机起点
    收敛到 2032。**§F ncu**：两分配器硬件计数逐项相同（差纯并发时序）。
  - **含义**：SM 静态分区分配器不仅要 in-situ（§75）、还要**多起点 + 合法分配守门 + 承认 ~1% 噪声地板**，
    且即便如此在高度异构图只**追平**共享 SM + 交错、在发射×访存混合上留全局税——再次给生产实现
    （MPK 去中心化 event queue / DeepGEMM 静态角色 + ring barrier）**「共享 SM + 设备内调度」**而非
    「离线静态 SM 分区」以量化理由。判据 **C1–C5 全不满足**（纯数据面微基准）。**本轮新增实验**；
    P88-1 新留尾巴（§76.10）：多起点是**有限起点集 + 局部爬山**（非全局搜索）、oracle 只两模式、
    `bind` O=1、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100。**踩坑已记入「已知坑」**
    （评测角色绑定分配前必须验 `sum==numSM` 且活跃模式 ≥1，否则 `compute_layout` 丢 work → 吞吐虚高）。

### P89 实验扩展 68：SM 角色绑定 × 每 SM 多块（O>1）（`docs/04-experiments.md` §77）

- [x] **P89-1** 补掉 §73–§76 **反复出现的明文尾巴**（§73.8/§74.9/§75.9/§76.10，原话）「**`bind` 用 `O=1`
  （每 SM 一 block），未测每 SM 多块的角色绑定**」：把 §73 的「按 SM 角色绑定」**正确推广到任意每 SM
  block 数 `O`**（每 SM 的 `O` 个 block 用**设备侧 per-SM 原子游标**领 `local∈[0,O)`、读该 SM 区的第
  `local` 个 `BP` 子段），**唯一新变量 = `O`**（保持 `O·T=1024`：`O=1/T1024`、`O=2/T512`、`O=4/T256`）。
  见 `docs/04-experiments.md` **§77**，源码 `src/hetero_sm_bind_occ.cu`、脚本
  `harness/hetero_sm_bind_occ.sh`、原始输出 `harness/hetero_sm_bind_occ.out.txt`（目标卡 H100/sm90a、
  自动选最闲 device（落 device 6）、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB（uint4 计）、`U=4`、
  `__maxnreg__(32)`；`pat_hetero` 32 regs/0 spill/8200 B smem，与 §73–§76 同）。四模式/五混合/三映射
  **完全复用 §71–§76**。核心结论：
  - **§A 正确性**：`bind` 的 `O=1/2/4` 折叠 `sink` **逐位等于 host 参考**（2890531677 / 537941000 /
    106554789，全 `OK`）；`O=4` placement 占用 **132/132 SM、每 SM 恰 4 块**。per-SM 原子子段游标在
    `O>1` 下数值安全。
  - **§B 单模式基线（`O·T=1024`）**：`coalesced` 3026/2987/3022、`sparse` 704/698/700、`scalar`
    2881/2874/2590 三档近平（复现 §70/§71「吞吐由合量 `O·T` 定」）；**唯一例外 `l2res16M`
    `O=1→4` 从 15399→20472（1.33×）**——它偏爱「多而小」的 block（对上 §71.7）。
  - **§C（headline）`bind ≈ interleave` 在三档 `O` 下逐行不变（差 ≤1%）**：`coal+scalar` 的 ~9–13% 税
    在 `O>1` 下**不消失**（`O=4` 反略放大 0.868/0.862）⇒ **§73「共驻税是全局（SM 间/HBM/L2）、非 SM 内
    发射槽」对每 SM block 数稳健**；§72.8 缓解 (a)「加 warp 摊」在「**固定 `O·T`、只改块粒度**」意义上
    **不兑现**（真正的「加 warp」= 提高 `O·T`，是另一条轴）。`bind_bal` 仍全面劣于 `bind`/`interleave`
    且与 `O` 无关（`equal4` 0.76、`coal+l2res` 0.81–0.91），与 §73/§74/§85 一脉相承。
  - **§D/§E**：`bind_bal` 分配示例 `equal4 [20,87,21,4]`；`coal+scalar` 扫 scalar SM 份额 `O=1` vs
    `O=4` 曲线**几乎重合**（内点最优都在 `sc≈84–96`）——分区配平取舍与每 SM block 数无关。
  - **§F ncu**：`bind` 在 `O=1` vs `O=4` 硬件计数**逐项相同**（DRAM 1.07 GB、扇区 33567072、活跃 warp
    49.99%/49.20%、`inst/cycle` 0.04）⇒ §C 的「税不随 `O` 变」是**真·物理不变**。
  - **含义（收口 §71–§77 的 SM 分配链条）**：静态 SM 分区既要 in-situ 速率、多起点 + 合法守门 + 认噪声
    （§75/§76），**又要知道「每 SM 几个 block 不改全局税、只改个别 task 速率」**——没有任何 SM 分区参数
    组合能消掉发射受限 × 访存受限的税，再证生产实现为何走「共享 SM + 设备内调度」。判据 **C1–C5 全不满足**
    （纯数据面微基准）。**本轮新增实验**；P89-1 新留尾巴（§77.8）：`O` 只 `1/2/4` 且固定 `O·T=1024`
    （**未单独扫「每 SM warp 数」**，故「加 warp 摊」的完整验证仍缺口）、合成访问、单 `U=4`/
    `__maxnreg__(32)`、只测 H100、未测 `O>4`。**踩坑已记入「已知坑」**（per-SM 原子游标必须 `tid==0`
    + `__syncthreads()` 广播，否则每线程各领号 → 绝大多数线程 `BP=0` 空转 → 吞吐虚高 + 结果错）。

### P90 实验扩展 69：SM 角色绑定 × 每 SM warp 数（`docs/04-experiments.md` §78）

- [x] **P90-1** 补掉 P89-1（§77）**§77.8 明留的尾巴**（原话）「`O` 只取 `1/2/4` 且**固定 `O·T=1024`**，
  故**未单独扫「每 SM warp 数（`O·T`）」**——『加 warp 摊』的完整验证仍缺口」：把**每 SM 总线程 / warp 数
  `Wsm=O·T`** 变成**唯一新变量**，在同一 SM 角色绑定、同一逻辑工作、同一模式份额、同一 SM 分配器下扫它。
  见 `docs/04-experiments.md` **§78**，源码 `src/hetero_sm_warps.cu`、脚本 `harness/hetero_sm_warps.sh`、
  原始输出 `harness/hetero_sm_warps.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、
  nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、载体 = 寄存器 `__ldcs` uint4 + `U=4`、`__maxnreg__(32)`；
  `pat_hetero` 32 regs/0 spill/8200 B smem）。四模式/逻辑字节/XOR 参考/误差扩散 `assign[]`/SM 分配/
  per-SM 原子子段游标**完全复用 §71–§77**；**唯一新变量 = `Wsm`**（固定 `T=256` 扫 `O=1/2/4/8`，另补
  `Wsm=1024` 的 `T` 拆分 `O2/T512`、`O1/T1024`）。核心结论：
  - **§A 正确性**：`bind` 在 `O=1/2/4/8` 逐位对 host 参考、`O=8` 占 132/132 SM（每 SM 恰 8 块）。
  - **§B 单模式**：`sparse8x` 是唯一「`Wsm` 不是旋钮」的模式（700–715 GB/s，物理流量墙）；`scalar4B`
    最吃 warp（1166→2882）；`l2res16M` 偏爱「多而小的块」（`O4/T256` 20104 > `O1/T1024` 14595）。
  - **§C/§C2（headline）税随 `Wsm` 收回**：`coal+scalar` 的 `bind` `meas/E` **0.772（256）→ 0.825（512）
    → 0.913（1024）→ 0.995（2048）**——**§72.8 缓解 (a)「加 warp 摊」兑现，当且仅当「加 warp」= 提高
    `Wsm`**（改块粒度/ SM 分区都不行）。固定 `Wsm=1024` 内 `(T,O)` 拆分不变（0.898/0.912/0.913，复现
    §77）。**绝对量口径**：`Wsm>1024` 后 `coal+scalar` 实测已近平台（`E−实测` 缺口从 ~240 缩到 ~13，
    故加 warp 主要让混合**可预测**、并消掉分区调参痛点）。`bind ≈ interleave` 每列差 ≤1%。
  - **§E 高 `Wsm` 下分区近乎「可通吃」**：`coal+scalar` 扫 scalar SM 份额，低 `Wsm=256` 是「两端好、
    中间深谷」（`sc=12` 掉到 **0.147**）；高 `Wsm=2048` 基本压平 ~1.0（`sc≥36` 全 0.92–1.06）——
    §72.8「难一套参数通吃」再收窄一层：**`Wsm=1024` 口径下成立、抬到 2048 即恢复**。
  - **§F ncu**：`O1/T256` vs `O8/T256` 物理字节几乎不变（DRAM 1.07→1.13 GB、扇区 +0.07%），
    活跃 warp **12.50% → 98.97%**——`Wsm` 的机制就是补满在飞 warp，与 §70/§80–§82 同源。
  - **对 §73「税是全局」的精确限定**：税**不住**在块边界（§77）/ SM 分区（§73），但**是每 SM warp 预算
    的函数**，`Wsm→` 满占用率（2048）即基本消失；数据面饱和与共驻税是同一个「在飞 warp 预算」旋钮。
    判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§77 同档）。**本轮新增实验**；P90-1 新留尾巴
    （§78.8）：`Wsm=2048` 只经 `O=8/T=256`（未补 `O=2/T=1024`，「warp 数 ≠ block 数」未完全解耦）、
    合成访问、单 `U=4`/`__maxnreg__(32)`、只测 H100、未与 DeepGEMM SM100 对跑。

### P91 实验扩展 70：同一 `Wsm` 内的 `(T,O)` 拆分（`docs/04-experiments.md` §79）

- [x] **P91-1** 补掉 P90-1（§78）**§78.8 明留的头号尾巴**（原话）「**`Wsm=2048` 只经 `O=8/T=256`**，
  未补 `O=2/T=1024`，故『**warp 数 ≠ block 数**』在 2048 处**未完全解耦**（`Wsm=1024` 处有 `T` 拆分交叉，
  税对拆分不敏感）」：把 `Wsm ∈ {512,1024,2048}` **每个都铺全可行 `(T,O)` 拆分**（`O=Wsm/T`），
  **唯一新变量 = 同一 `Wsm` 下的拆分**；`Wsm=2048` 补到 **`T128/O16`**（16 块/SM、恰占满寄存器文件）。
  见 `docs/04-experiments.md` **§79**，源码 `src/hetero_sm_warps_split.cu`、脚本
  `harness/hetero_sm_warps_split.sh`、原始输出 `harness/hetero_sm_warps_split.out.txt`（目标卡 H100/sm90a、
  **自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`U=4`、`__maxnreg__(32)`；
  `pat_hetero` 32 regs/0 spill/8200 B smem）。四模式 / 混合 / 映射**完全复用 §71–§78**。核心结论：
  - **§A 正确性**：`bind` 在 `O=1/2/4/8/16` 逐位对 host 参考（`O=16/T128` 也 132/132 SM、每 SM 恰 16 块、
    无 spill）——per-SM 原子子段游标在高 `O` 数值安全；
  - **§B solo**：**`coalesced`/`sparse8x` 在每 `Wsm` 内不随拆分变**（合量 `Wsm` 支配，复现 §70/§77）；
    但 **`scalar4B` 同 `Wsm=2048` 内极差 1.35×（偏大 `T`）**、**`l2res16M` 极差 1.61×（偏小 `T`）**
    ——「合量支配」只对 HBM 带宽受限模式严格成立（与 §71.7/§83 一脉）；
  - **§C（headline）`coal+scalar` 的税在每 `Wsm` 内对拆分稳定（spread ≤1.8%）**：`Wsm=512`
    `[0.817 0.825 0.830]`、`Wsm=1024` `[0.901 0.913 0.912]`、**`Wsm=2048` 四拆分 `[0.978 0.969 0.987
    0.987]`**——**§78.8 的疑虑证否**：§78 的「税随 `Wsm` 收回」是「`Wsm` 一提上来就在所有拆分上同时
    收回」，非单点巧合（`bind ≈ interleave` 每列差 ≤1% 也保持）；
  - **§F ncu**：`Wsm=2048` 三拆分（`T1024/O2`、`T512/O4`、`T256/O8`）扇区数 **逐字节相同
    （33579744）**、活跃 warp 三档都 **~99%**、DRAM 仅 `O=8` +5%（块级 prologue）——税的回收藏在
    「把每 SM 在飞 warp 补满」这一个机制量里，与块粒度无关（对上 §78.7）。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§78 同档）。**本轮新增实验**；P91-1 新留尾巴
  （§79.6）：`meas/E` 的 spread 混入 `E` 的拆分依赖（`scalar+l2res`/`equal4` 需与绝对量并读）、
  合成访问、单 `U=4`/`__maxnreg__(32)`、`Wsm` 只到 2048、只测 H100、未与 DeepGEMM SM100 对跑；
  至此 §71–§79 的「SM 分区 / 块粒度 / 每 SM warp 数 / `(T,O)` 拆分」四旋钮链条闭合。

### P92 实验扩展 71：共驻税的「口径」与「每线程 ILP」（`docs/04-experiments.md` §80）

- [x] **P92-1** 补掉 P91-1（§79）**§79.6 明留的两条尾巴**——(tail 1)「`meas/E` 的 spread 混入
  `E` 的拆分依赖（`scalar+l2res`/`equal4`），故比值需与绝对量并读」；(tail 2)「`U=4`/
  `__maxnreg__(32)` 单一（每线程 ILP 未当独立变量）」。把「共驻税」从**比值 `meas/E`**换成
  **绝对量**（mixed 绝对 GB/s + 固定参考归一 `taxRef=E_ref/mixed`），并把**每线程 ILP `U`**
  变成独立旋钮（固定 `Wsm=1024/T256/O4` 扫 `U∈{1,2,4,8}`），再测 `U×Wsm` 替代性。
  见 `docs/04-experiments.md` **§80**，源码 `src/hetero_sm_tax_ilp.cu`（kernel 模板化在 `U`
  上）、脚本 `harness/hetero_sm_tax_ilp.sh`、原始输出 `harness/hetero_sm_tax_ilp.out.txt`
  （目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读
  1 GiB、寄存器 `__ldcs` uint4 + `__maxnreg__(32)`；`pat_hetero_u<1/2/4/8>` = 30/28/32/32 regs、
  **四档全 0 spill**）。四访存模式/逻辑字节/XOR 参考/误差扩散 `assign[]`/SM 分配/per-SM 原子子段
  游标**完全复用 §71–§79**。核心结论：
  - **§A 正确性**：`U=1/2/4/8` 的 `bind` 与 `interleave` **逐位对 host 参考**（`U` 不动数值）；
  - **§B 单模式随 `U`**（`Wsm=1024`）：`coalesced` `U=2` 即饱和（1.07×）、`sparse8x` 撞物理墙
    （1.00×），而 **`scalar4B` `U=8` 2.22×、`l2res16M` 2.23×**——发射/L2-延迟受限模式的单模式
    吞吐把 `U` 当一阶量（对上 §68/§69/§70）；
  - **§C（headline 一，口径）**：`Wsm=2048` 内 4 拆分的 spread——`scalar+l2res` **绝对 mixed spread
    1.084 而 `meas/E` spread 1.205**（多出的 ~1.11× 是 `E_self` 漂移：5417→4146，因 l2res solo 随
    `T` 缩升、scalar solo 随 `T` 缩降），**§79.3 判断证实**；反过来 **`coal+scalar` 绝对 spread 1.188
    （`T=1024` 3037 vs `T=128` 2552、大 `T` 快 1.19×）而 `meas/E` spread 仅 1.017**——比值**掩盖**
    了绝对性能的拆分依赖。⇒ **「税（比值）与拆分无关」≠「混合绝对吞吐与拆分无关」**；
  - **§D（headline 二，ILP）**：**固定 `Wsm=1024` 只动 `U=1→8`，共驻税同样大幅收回**——
    `coal+scalar` `meas/E` **0.708→0.982**、绝对 **1238.8→2880.0（2.32×）**；`scalar+l2res`
    **0.587→0.992**、**1325.0→4985.1（3.76×）**；`coal+sparse`（HBM 墙）对 `U` 无感。
    ⇒ 支配量是**每 SM 在飞 load 量 `Wsm·U`**，`Wsm` 只是其一因子（§78/§79 的「税随 `Wsm` 收回」
    是「单动 `Wsm`」特例）；
  - **§E `U`×`Wsm` 替代性**：等 `Wsm·U` 的 (Wsm,U) 组合塌到 ±2–10%（如 `4096`：2048/U2=2443、
    1024/U4=2543、512/U8=2690），**高 `U`、低 `O` 略优**（块级 prologue 少，对上 §64/§76）——
    把 §60/§68/§70 的「ILP↔占用率↔线程数」替代性**推广到「每线程 ILP ↔ 每 SM warp 数」**；
  - **§F ncu**：`U=1/4/8` 的 DRAM 字节（1.07 GB）、扇区（33567072）、活跃 warp（~49%）**逐项相同**，
    唯 `inst/cycle` 0.06→0.04——**`U` 只改在飞深度（ILP），不改占用率/流量**。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§79 同档）。**本轮新增实验**；P92-1 新留尾巴
  （§80.7）：`U≤8` 且 `__maxnreg__(32)` 固定（放宽寄存器预算后的 `U∈{16,…}` 未测）、`U×Wsm`
  替代性只用 `coal+scalar` 一个混合、`Wsm=2048` 内拆分只到 `T128/O16`、合成访问、只测 H100、
  未与 DeepGEMM SM100 对跑。

### P93 实验扩展 72：寄存器预算 × 每线程 ILP 的二维解耦（`docs/04-experiments.md` §81）

- [x] **P93-1** 补掉 P92-1（§80）**§80.7 明留的两条尾巴**——(tail 1)「`U ≤ 8` 且
  `__maxnreg__(32)` 固定（放宽寄存器预算后的 `U∈{16,…}` 未测——§68/§69/§81 已知寄存器预算与
  ILP 耦合）」；(tail 2)「`U×Wsm` 替代性只用 `coal+scalar` 一个混合、`T` 固定 256」。
  把**寄存器预算 `R`（`__maxnreg__(R)`，模板参数）**与**每线程 ILP `U`** 都当独立旋钮做 `U×R`
  二维；每个 `(U,R)` 先用 **occupancy API** 实测每 SM 可达块数 `occ`，再令 `O=occ`、
  `grid=numSM·occ`（**保证 bind 全图常驻、不丢 work**，避开 P89-1/§77 的丢 work 陷阱）。
  见 `docs/04-experiments.md` **§81**，源码 `src/hetero_sm_tax_reg_ilp.cu`、脚本
  `harness/hetero_sm_tax_reg_ilp.sh`、原始输出 `harness/hetero_sm_tax_reg_ilp.out.txt`
  （目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读
  256 MiB（uint4 计）、`U∈{1,2,4,8,16,32}`×`R∈{32,48,64,96,128}`、`T=256`）。四访存模式/
  逻辑字节/XOR 参考/误差扩散 `assign[]`/SM 分配/per-SM 原子子段游标**完全复用 §71–§80**。核心结论：
  - **§A 正确性**：`U=16/32` 与 `R` 高低**都不动数值**，bind occupancy-aware 全图逐位对参考；
  - **§B 资源表**（`ptxas -v` + occupancy API）：`numRegs` 自然需求随 `U` 增长（`U=1/2/4/8/16`
    = 30/28/40/54/84，`U=32`≥128）；`R` 是**上限不是设定值**（`R≥自然需求`时用自然需求，
    复现 §69）；只有 `<32,32>`（U=32,R=32）真 spill（12 B stores/48 B loads），`U≤16` 在 `R=32`
    都 **0 spill 且 `occ=8`（Wsm=2048，满占用率）**；**加 `R` 的唯一效果是砍 `occ`**（同 `U` 从
    `R=32→128`，`occ` 8→2–6，无多出来的在飞深度）；
  - **§C（headline 一，ILP 天花板）**：固定 `R=128`（全档无 spill）扫 `U`——`coalesced` 对 `U`
    不敏感（±2%，HBM 墙），`scalar4B` 1.33×、`mixed`（coal+scalar）**1.46×**、`meas/E` **0.810→1.014**，
    **天花板在 `U≈8`**（`meas/E` 0.973），`U=16` 回落 0.942（占用率损失盖过 ILP 增益）；但**等
    `Wsm=512` 下 `U=32`(mixed 2776) > `U=16`(2609)** ⇒ 支配量是「每 SM 在飞 load 总量」而非 warp 数；
  - **§D（headline 二，解耦）**：固定 `U` 扫 `R`——**抬 `R` 不变好**（`U=8` 的 `R=32` 给最高
    `mixed=2769`/`meas/E=1.000`，抬到 `R≥48` 掉到 ~2686/~0.96），证明**寄存器预算对数据面 task 只经
    占用率起作用**；**最优是「不 spill 的最小 `R`」**（本例 `U≤16` 用 `R=32`：满占用 + 0 spill）；
    「给寄存器」只在**服务于更高 `U`** 时才值得；
  - **§E（Q3，统一性）**：第二混合 `scalar+l2res` **不服从**「每 SM 在飞 load」——`l2res` solo
    13.2–14.8 TB/s（L2 驻留），`meas/E` 在 **0.949–1.117** 波动、随 `U` 无单调收回、峰值**>1 超加和**
    （对上 §72「L2 驻留 × HBM 流式超加和」）⇒ 该机制量只对**两端皆 HBM/发射受限**的混合成立；
  - **§F ncu**：`(U,R)` 三档 DRAM 读字节几乎相同（277–282 MB）、**活跃 warp 按 `occ` 比例降**
    （97.8→49.3→24.8% = occ 8→4→2）——只改占用率、不改流量。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§80 同档）。**本轮新增实验**；对 megakernel 的
  worker 块设计给出「**不要为保险多要寄存器**：把每 task 类型的 `R` 设成其 `U` 所需自然需求」的结论
  （与 §8「寄存器被固定为各 task 最大值」互为两面）。P93-1 新留尾巴（§81.7）：`U≤32`/`R≤128`/`T=256`
  固定、`U=32,R=32` 才 spill（门槛偏窄）、§C 内 ILP 与占用率仍部分纠缠、§E 只到 `R∈{32,64,128}`、
  合成访问、只测 H100、未与 DeepGEMM SM100 对跑。

### P94 实验扩展 73：每块线程数 T 作为第三旋钮 + R<32 角点（`docs/04-experiments.md` §82）

- [x] **P94-1** 补掉 P93-1（§81）**§81.7 明留的两条尾巴**——(tail 1)「`T=256` 固定、**`R<32` 未测**」；
  (tail 2)「§C 固定 `R=128` 时 `occ` 随 `U` 变（**ILP 与占用率在 §C 内仍部分纠缠**）」。把**每块线程数
  `T`** 变成第三旋钮，令 `R=32` 让 `U≤16` 都 0 spill、`occ` 不随 `U` 变，从而把「`Wsm` 钉在 2048」
  的**纯 ILP 曲线**与 §81 `§C`（`R=128`、`occ` 随 `U` 降）对照；并补 `R∈{24,28}` 的下界角点。
  见 `docs/04-experiments.md` **§82**，源码 `src/hetero_sm_tax_T.cu`、脚本
  `harness/hetero_sm_tax_T.sh`、原始输出 `harness/hetero_sm_tax_T.out.txt`（目标卡 H100/sm90a、
  **自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读 256 MiB、四访存模式/
  XOR 参考/误差扩散 `assign[]`/SM 分配/per-SM 游标**完全复用 §71–§81**、`T` 为运行期 `blockDim`）。
  核心结论：
  - **§A 正确性**：`R=24`（最低预算）、`U=16`、`T∈{128,256}`、interleave+bind 全图**逐位对参考**；
  - **§B 资源表（四重封顶）**：H100 占用率 = `min(线程 2048/T, 寄存器 65536/(regs·T), 块 32, SMEM)`；
    **`R≤32` 的 `occ` 一列全等于线程上限 `2048/T`**（T=128→16、256→8、512→4、1024→2），把 `R` 从
    32 压到 24 **一格不涨**；`R>32` 才让寄存器成约束；**每块 8 KiB 静态归约数组**在小 `T` 时成
    **第四个上限**（`T=64` 被压到 `occ=24`、`Wsm=1536`）；
  - **§C（headline 一，纯 ILP）**：固定 `Wsm=2048` 扫 `U`——`meas/E` **0.81→≈1.00，天花板 `U≈8`**、
    mixed 1.45×；**`U=16` 是平顶（0.99/0.98）而非 §81 `§C` 的回落（0.942）** ⇒ **§81 `U=16` 的回落
    主要是占用率耦合**（`Wsm` 1024→512），税随 `U` 收回本身是**纯 ILP 效应**；
  - **§D（headline 二）**：固定 `Wsm=2048` 扫 `T∈{128,256,512}`，`coal+scalar` 的 mixed/meas/E
    **±2% 不变** ⇒ **§79「税只认 `Wsm`」对发射受限×访存受限混合也成立**（`T` 唯一作用是在寄存器
    预算内凑够 `Wsm`）；
  - **§E `R<32`**：`R=24/28` 的 `occ` 与 `R=32` 相同（线程上限封顶）却**唯一引入 spill**（localB 16/8），
    mixed 单调更差 ⇒ **`R=32` 是「不 spill 的最小预算」的下界**；`__maxnreg__` 下限 24（`R<24` 被
    ptxas 调到 24）；`R<32` 的 `meas/E>1` 再次印证「比值放大基线漂移、须与绝对量并读」（§80/§92）；
  - **§F（Q3）第二混合 `scalar+l2res`**：固定 `Wsm=2048` 下 mixed **随 `T` 单调升**（+4.7%，meas/E
    超加和 >1）⇒ **L2 延迟受限混合不服从「税只认 `Wsm`」，少而胖的块更好**（机制假设：L2 延迟
    停顿与块调度状态，未做 ncu 归因）。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§81 同档）。**本轮新增实验**；对 megakernel 的
  worker 块设计给出「寄存器预算只需到 max(自然需求,32) 下界、`T` 对 HBM/发射混合可自由凑 `Wsm`、
   但对 L2 延迟受限 task 优先少而胖、并注意每块 SMEM 是第四占用率上限」的结论。P94-1 新留尾巴（§82.7）：
   `U≤16`/`T` 5 档/每块 8 KiB 静态归约数组使小 `T` 混入 SMEM、§F 未 ncu 归因、只测 H100、未与 SM100 对跑。

### P95 实验扩展 74：§82.6「少而胖更好」的 ncu 归因（`docs/04-experiments.md` §83）

- [x] **P95-1** 补掉 P94-1（§82）**§82.7 明留的尾巴**（原话）「§82.6 的『少而胖更好』只给**现象 +
  机制假设**，未做 ncu 归因到具体的 L2 停顿/调度槽指标」。**复用 §82 同一内核不改一行**
  （`src/hetero_sm_tax_T.cu` 的 `run` 模式），用 ncu warp stall-reason 直方图 + L2/L1 sector 命中 +
  issue/warp-active，在固定 `Wsm=2048`（`U=8,R=32,bind`）下扫 `T∈{128,256,512}`，对混合
  `scalar+l2res`（`seq=0,0,1,1`）及两单模式 + 控制组 `coal+scalar`（`seq=1,0,1,0`）归因。
  见 `docs/04-experiments.md` **§83**，脚本 `harness/hetero_sm_tax_T_attr.sh`，原始输出
  `harness/hetero_sm_tax_T_attr.out.txt`（目标卡 H100/sm90a、device 6、nvcc 13.2、
  `ncu --replay-mode application --cache-control none`）。核心结论：
  - **§A 先复现现象**：无 ncu 计时 `scalar+l2res` 混合 4330→4449→4518 GB/s（T=128/256/512，+4.3%），
    与 §82.6 的 +4.7% 同形（绝对值低 ~8%，device/时钟差）；但幅度只有 ~4%、**在噪声地板附近**。
  - **§B/§C（headline，机制假设被否）**：ncu **聚合**计数**不支持** §82.6 的猜测——
    **(1)** 「L2 延迟停顿」`long_scoreboard` 与「调度槽争用」`not_selected` **都不随 `T` 变**：
    `l2res` solo 的 `long_sb` 50.16%→50.26%、`not_selected` 9.08%→9.04%、`issue_active` 54.26%→54.96%
    （L2 驻留侧**逐项 T-不变**）；混合 `not_selected` 恒 1.8%。**(2)** 唯一随 `T` 单向变的是
    `warps_active`（混合 93.25→97.51）与 `lg_throttle`（**LSU 背压**：混合 11.57→8.95、
    `scalar` solo 17.82→10.97、`coal+scalar` 11.44→5.29），是**块 prologue/占用率**与 **LSU 背压**
    的次生效应、非 L2 停顿。**(3)** 按模式分解，**T 依赖住在 `scalar4B`（发射/LSU 受限）这一半**
    （solo 吞吐 2372→2453→2425、`issue_active` 9.34→9.85、`lg_throttle` 大降），**L2 驻留的 `l2res`
    这一半 T-不变** ⇒ **§82.6 把原因放错了半边**（不是「L2 延迟受限侧」而是「发射/LSU 受限侧」）。
  - **§D 控制组 `coal+scalar`** 同样 `not_selected` 恒 <0.6%、`lg_throttle` 随 T 降（11.4→5.3），
    与 §82.4「税只认 `Wsm`」一致。
  - **含义**：块粒度对 L2/发射混合的收益应记在**发射受限 task 的 LSU 背压/驻留 warp** 上，而非
    「减少 L2 停顿争用」；一阶旋钮仍是 `Wsm`（§78–§82），块粒度只是 ~3–5% 的次级量。
    判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§82 同档）。**踩坑已记入「已知坑」**
    （ncu application-replay 会放大程序**自报** `ms`/`GB/s` 约 2×，只能用硬件计数器与
    `gpu__time_duration`）。**本轮新增实验（复用内核、无新 src）**。

### P96 实验扩展 75：块粒度 T 次级效应的统计稳健性 + 内存延迟分解（`docs/04-experiments.md` §84）

- [x] **P96-1** 补掉 P95-1（§83）**§83.5 明留的两条尾巴**（原话）「混合 T 趋势只有 ~4%、单模式
  ~3–5%，**只取 3 次 run 均值**……据此只可说『L2 停顿不进 T』这条否定性结论，不可把 +4.3% 当成
  确定的一阶现象」与「**未做 `--section MemoryWorkloadAnalysis` 的延迟分解**」。**复用 §82/§83 同一
  内核**（`src/hetero_sm_tax_T.cu`，**新增 `robust` 子命令，不改 `run`/`bench` 语义**）：(1) 稳健计时
  ——7 配置各 **S=30 次单采样 + round-robin 交错**，给 min/median/mean/std 与 `T512/T128`、
  `T256/T128` 的 median 比 **+ bootstrap 95% CI（B=2000）**；(2) ncu 每 warp **绝对**停顿周期——
  `long_scoreboard`（等 L2/global）与 `lg_throttle`（LSU 背压）随 `T` 的分解 +
  `--section MemoryWorkloadAnalysis_Tables`。见 `docs/04-experiments.md` **§84**，脚本
  `harness/hetero_sm_tax_T_robust.sh`、原始输出 `harness/hetero_sm_tax_T_robust.out.txt`
  （目标卡 H100/sm90a、device 4、nvcc 13.2）。核心结论：
  - **§A 统计判定：+4.3% 是真效应**——混合 `scalar+l2res` 的 `T512/T128 = 0.9586`（吞吐 **+4.3%**），
    CI95 `[0.9528,0.9627]` **不含 1.000**；`T256/T128 = 0.9691` 同样显著。§83.5 的不敢定论来自
    min-of-3 + 比值口径 + `T=128` 方差偏大；**但幅度确小、仍是次级量**（一阶仍 `Wsm`）。
  - **§A 两端相反**：`scalar4B` 单模式 `T512/T128 = 0.9777`（**+2.3%**，显著），`l2res16M`
    `= 1.0269`（**−2.6%**，显著）——混合净收益全来自发射受限侧，`l2res` 侧**轻度变差**（§83 的
    「T-不变」是 min-of-3 比值口径的过粗结论）。
  - **§B 延迟分解（headline）：`T` 不减轻 L2 延迟**——每 warp `long_scoreboard` 周期**随 T 升**
    （混合 +6%、`scalar4B` +13%、`l2res16M` +16%）；唯一随 T 单调降的是 **`lg_throttle` LSU 背压**
    （混合 **−24%**、`scalar4B` **−43%**）⇒ 「少而胖」的机制落点是**发射受限侧的 LSU 背压**，非
    「减少 L2 停顿争用」（§82.6 猜测被否、§83 定性被绝对量确认）。
  - **§B4** 混合同 `Wsm` 下 DRAM ~74%、L2 命中仅 11%、全局读 30.05/32 B/扇区 ⇒ 是 **DRAM + LSU**
    题、非 L2 带宽题。**口径 caveat**：ncu `smsp__average_warp_latency` 绝对值在 replay 下方向与自然
    计时相反，吞吐只采信 `robust` 计时、ncu 只作停顿项定性分解。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§83 同档）。**本轮新增实验（复用内核 + 新子命令 +
  新脚本）**。

### P97 实验扩展 76：块粒度 `T` 趋势的跨进程 / 跨 device 稳健性（`docs/04-experiments.md` §85）

- [x] **P97-1** 补掉 P96-1（§84）**§84.4 明留的尾巴**（原话）「`T=128` 的别处方差……与 bootstrap
  都基于同一 `S=30` 单次实验；**CI 是『同一进程内 30 次采样』的，不含跨进程 / 跨 device 的系统差**」。
  **复用 §82/§83/§84 同一内核**（`src/hetero_sm_tax_T.cu`，**新增 `xproc <S> <seed>` 子命令，不改
  `run`/`bench`/`robust` 语义**）：把「换进程、换卡」当新的独立重复——在 **4 块空闲 H100 上各起 6 个
  独立进程（共 24 进程）**，每进程 `S=30` 单采样且**每 round 内 Fisher-Yates 打乱访问顺序**、打印
  原始样本，外部按**进程**做 bootstrap 汇总。见 `docs/04-experiments.md` **§85**，脚本
  `harness/hetero_sm_tax_T_xproc.sh` + 汇总器 `harness/hetero_sm_tax_T_xproc.py`、原始输出
  `harness/hetero_sm_tax_T_xproc.out.txt`（目标卡 H100/sm90a、device 3/4/6/7 各 6 进程、nvcc 13.2、
  固定 `U=8,R=32,map=bind`、`Wsm=2048`，与 §82–§84 同配置）。核心结论：
  **(1) 跨进程 / 跨 device 下 +4.3% 稳健（headline）**——混合 `scalar+l2res` 的 24 进程
  `T512/T128` median 比 = **0.9598**（吞吐 **+4.2%**），跨进程 bootstrap CI95 `[0.9574,0.9629]`
  **不含 1.0**，与 §84 同进程结果 `0.9586/[0.9528,0.9627]` 基本重合，且 **24/24 进程逐个显著**；
  **(2) 方差分解**——混合三档 betweenStd（0.33–0.63%）**<** withinStd（1.00–1.32%）的一半 ⇒ 换进程/
  换卡的系统差小于同进程噪声、§84 的同进程 CI 未系统性低估；**(3) 唯一例外 `l2res16M` 单模式**——
  `T512` betweenStd **2.21% > 1.57%** withinStd，L2 驻留负载有真实卡间差（device 7 绝对值偏高 ~4%）；
  **(4) 设备间小 offset**——四卡比值全 < 1（0.9538–0.9634，device 3 比 device 6/7 低 ~0.9%，卡间
  std ~0.4%），远小于效应但**与 ~1% 绝对效应的百分位差同量级**（比绝对效应须固定卡）。判据
  **C1–C5 全不满足**（纯数据面微基准，与 §71–§84 同档）。**本轮新增实验（复用内核 + 新子命令 + 新脚本 + 汇总器）**。

### P98 汇总 3：异构 task 共驻与 SM 资源成本手册（`docs/12-megakernel-coresidency.md`）

- [x] **P98-1** 把 P83–P97 这一长串**数据面 / 异构共驻微基准**（实验在 `docs/04-experiments.md`
      §71–§85）收口成一张**可操作的成本手册**——回答「把许多**不同类型的 task** 塞进同一个常驻 grid
      后，SM 资源怎么分、异构共驻付出什么税、哪些旋钮自由、哪些被物理约束锁死」。见
      `docs/12-megakernel-coresidency.md`（本文档为**纯资料汇总，无新增 H100 实验**；所有数字引自
      `docs/04` §71–§85 及 `harness/*.out.txt`）。十节：§0 第二层成本；§1 通用旋钮 `O·T` 与天花板
      （§70/§71：合量普适但非万能、`sparse` 撞物理墙、`l2res` 要小块）；§2 异构共驻税（§72：大体加和
      ±13%、唯一系统裂缝 `scalar` 发射受限 ~13% 且与 `(T,O)` 正交、`l2res` 白吃者超加和 1.111、
      sparse 低掺杂 10–30% 毒化 0.855）；§3 税住在 SM **间**非 SM 内（§73：`bind≈interleave` 逐行、
      `bind_bal` 反更差、分区不能绕开）；§4 SM 分配器（§74–§76：常数速率谬误两坐标、单跑曲线高估饱和
      模式、in-situ 收回崩盘、多起点追平 oracle ~1%、噪声地板 ~1%）；§5 一阶旋钮 = 每 SM 在飞 warp
      `Wsm`（§77–§79：税随 `Wsm` 收回 0.772→0.995、`O` 与 `(T,O)` 拆分都不动税）；§6 支配量 `Wsm·U`
      与寄存器只经占用率（§68/§69/§80–§82：四重占用率封顶、`R` 取不 spill 最小）；§7 块粒度 `T` 次级
      （§83–§85：落点在发射侧 LSU 背压非 L2 停顿、跨进程/跨卡稳健、卡间差）；§8 决策表；§9 回到 MPK/
      DeepGEMM（共享 SM + 设备内调度是物理结论、「少而重」而非「多而杂」）；§10 局限。配套博客 post
      `mega-kernel-13-coresidency`（weight 14）并更新 `mega-kernel-00-index` 目录。**P98-1 尾部**：
      `docs/04` §59–§70（数据面到 HBM 峰值 / 端到端骨架 / worker 侧复杂度 / 寄存器×占用率解耦）
      尚未并入任何手册——留作下一轮可选增量。

### P99 汇总 4：端到端骨架与 worker 数据面能力手册（`docs/13-endtoend-dataplane.md`）

- [x] **P99-1** 补掉 P98-1（`docs/12` §10 末句）明留的尾巴：「`docs/04` §59–§70（数据面到 HBM 峰值 /
  端到端骨架 / worker 侧复杂度 / 寄存器×占用率解耦）尚未并入任何手册——留作下一轮可选增量」。
  把 **P71-1–P82-1**（`docs/04-experiments.md` §59–§70）这一串**端到端调度骨架 + worker 数据面
  微基准**收口成一张**能力/成本手册**——回答两件事：(Q1) 一条 `producer → worker` 常驻流水端到端
  天花板在哪、夹层（取 task/折叠/环协议）吃掉多少；(Q2) worker 块的资源旋钮（寄存器预算/占用率/
  每块线程数/每线程 ILP）谁支配数据面吞吐。见 `docs/13-endtoend-dataplane.md`（本文档为**纯资料汇总，
  无新增 H100 实验**；所有数字引自 `docs/04` §59–§70 及 `harness/*.out.txt`）。七节：§0 第三本手册
  在链条里的位置；§1 数据面能到多快、要多少 producer（§59 `PS*` 随访存重量反向 `MD=2MB→PS*≈4` 对上
  MPK 1:32；§60 ILP/`T×U` 把数据面从 926 抬到 2840、饱和点不争用；§61 到 3219=峰值 96% / TMA 3075
  省寄存器 / 真峰值下描述符写读降 3–7%）；§2 端到端夹层（§62 同一二进制三档：fold ±1% 免费、环+producer
  −43.9%@G1040、最优是「少而大」85%；§63 producer 服务成本纯每批 ~2.1 µs、`§62.7` 扇出归因被隔离
  证伪）；§3 worker 侧复杂度与载体（§64 每 block prologue plain 0.35/TMA 0.69 µs；§65 cp.async≡plain、
  TMA +0.3 µs/block 是其专属；§66 深环无收益、单缓冲最优）；§4 寄存器×占用率×T×ILP（§67 8B 溢出归因
  `MINB`、耐压序 cp.async>TMA>plain；§68 同占用率下 32 vs ≥40 regs 1.70×、替代品；§69 `__maxnreg__`
  连续斜坡 32→36→40、与 `T` 替代；§70 饱和区只认合量 `O·T`、未饱和区寄存器先决）；§5 决策表；
  §6 回到 MPK/DeepGEMM（16 B cp.async 三重复合最优、1:32 量化依据、预计算描述符只推 id）；
  §7 局限。配套博客 post `mega-kernel-14-endtoend-dataplane`（weight 15）并更新 `mega-kernel-00-index`
  目录第 14 项。**P99-1 尾部**：至此 `docs/11`（§31–§58）/`docs/12`（§71–§85）/`docs/13`（§59–§70）
  三本手册合拢「调度机制 / 共驻税 / 端到端+数据面」闭环；`docs/04` 再无未并入手册的微基准节。

### P100 前端/算子层源码走读（`docs/14-mpk-frontend.md`）

- [x] **P100-1** 补齐 `docs/06`（运行层）/`docs/07`（单 task device）/`docs/08`（依赖分析器）三篇
      **共同的上游空白**：这三篇都从「已存在的 bGraph / `task_configs`」讲起，而「一张算子图是谁、
      怎么建出来的」一直没走；`docs/08` §对照表明写「算子分解不在 `annotated_graph.cc`；由
      `task_configs` + bgraph `grid_dim` 提供，分解策略本身在 `task_register.cc`/前端」。见
      `docs/14-mpk-frontend.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与
      `docs/06`/`07`/`08` 同快照）。七节：§0 编译栈四层（前端/依赖分析/运行层/device）与本文件定位；
      §1 前端三层对象 `KNGraph`（`kernel.py:208`，高层算子 API `:240-282` vs 手工 task API `:283/:802`）、
      `TBGraph`（`threadblock.py:4`，`new_input` `:8-15`）、`PersistentKernel`（`persistent_kernel.py:413`，
      `:501` 建 `KNGraph(disable_fingerprint=True)`）；§2 一个 layer 的三步骨架（以 `rmsnorm_layer`
      `:865-881`/`embed_layer` `:842-863`/`linear_layer` `:2390-2456`/`allreduce_layer` `:2488-2512`
      为例——建 `CyTBGraph(grid_dim,block_dim,1,64)`、`new_input(tensor,input_map,forloop_dim,store_in_dmem)`
      声明 grid→tensor 映射、`customized`+`register_task` 固化；`input_map`/`output_map` 就是
      `docs/08` 里 `EdgeInfo`/`TaskView::axis_map` 的来源；`grid_for_rmsnorm_linear_layer`
      `models/utils.py:3-14`）；§3 C++ 侧 `KNOperator`（`operator.h:29-53`，不含 bGraph）vs
      `KNCustomizedOp`（`customized.h:40` 带 bGraph）、`Graph::customized`→`create_customized_op`
      （`customized.cc:35-105`，按位置校验 `inputs` 与 `TB_INPUT_OP` 对齐）、`Graph::register_task`
      （`graph.cc:439`，取 `operators.back()`、**95 分支** name→`(num_inputs,num_outputs,TaskType,
      variant_id)`、写 `task_config` 成员 `graph.h:230-233`）、`core.pyx:1118-1129` 里 `bgraph` 形参
      被绑定层丢弃；§4 消费侧（`runtime.cc:222-243` `build_annotated_graph`；`annotated_graph.cc:278-283`
      **只认 `KN_CUSTOMIZED_OP`** → 高层算子不进 tGraph；`runtime.cc:1395-1420` 用 `bgraph.grid_dim`
      展开 task；`view/narrow/split` 是虚拟张量不进 `operators` → `is_barrier_edge`）；§5 与论文
      §4.1（行 462–468）/§4.2（行 1126–1146）逐条对照——**论文描述自动分解、`mpk` 生产前端是手工
      任务图**（对上 `docs/07` §0 的「superoptimizer vs 手写/CUTLASS 并存」）；§6 判据映射——前端只
      决定 C3 的**范围**（每 `customized()` 一个算子），C1/C2/C4/C5 由运行层保证；§7 局限（纯走读未构建、
      未逐行走 95 个 task、模型 builder/多卡切分未深挖、自动 superoptimizer 路径未覆盖）。**纯源码走读
      （资料分析），本轮无新实验。** 配套博客 post `mega-kernel-15-mpk-frontend`（weight 16）并更新
       `mega-kernel-00-index` 目录第 15 项。

### P101 自动超优化路径源码走读（`docs/15-mpk-superoptimizer.md`）

- [x] **P101-1** 补掉 `docs/14`（P100-1）**§7 明留的尾巴**——「**未覆盖自动（superoptimizer）路径**：
  本文只说明 `mpk` 前端不走它；Mirage 主分支的超优化器如何从高层算子搜出 threadblock 实现
  （`src/search/`、`src/transpiler/`）未在本文范围」。把论文 §4.1「operator decomposition」与 §4.2
  「task implementation generation（thread-block 级 superoptimization）」在源码里**逐段走通**。
  见 `docs/15-mpk-superoptimizer.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与
  `docs/06`/`07`/`08`/`14` 同快照；纯源码走读、无新实验）。九节：
  §0 自动路径 vs `docs/14` 生产前端的对照表（`KNGraph.superoptimize()` `kernel.py:543` vs 手工
  `*_layer()`；两条路共用下游依赖分析 + 运行层，都汇到同一 `KNCustomizedOp`）；§1 入口五段链路
  （缓存 `graph_dataset.find` `kernel.py:559-571` → `search()` `core.pyx:1314` → `cython_search`
  `search_c.cc:21` → `KernelGraphGenerator` `search.cc:20`/`generate_kernel_graphs` `search.cc:394`
  → `from_json`/`KNGraph` → compile+profile 选优 `kernel.py:596-692`；`*_to_explore` 参数透传
  `search_c.h:20-31`/`search_c.cc:47-79`）；§2 搜索空间（`GeneratorConfig` 默认值 `config.cc:6-66`：
  `max_num_threadblock_graph_op=9`/`kernel_graph_op=5`/`threadblock_graphs=1`、`knop/tbop` 候选、
  `frange={4,16,64}`；分区候选 `DimStrategy`：`get_grid_dim_cand` `dim_strategy.cc:30-148` 卡
  `[32, MAX_NUM_THREADBLOCKS_PER_KERNEL=4096]` `config.h:31`、`get_input_map_cand`/`get_output_map_cand`/
  `get_forloop_*`；「最小化 device→SMEM 加载」由 ⑤ 实测代替代价模型）；§3 搜索算法
  `generate_next_operator`（`search.cc:129-392`）三层 case——K1/K2/K3（K3 建 bGraph 并降入
  `LV_THREADBLOCK`，即分区发生点）+ B1/B2（tile 内算子、封装 `KNCustomizedOp`），三横切剪枝
  `check_abstract_expr`/`check_range`/`check_order`（`search.cc:82-102,504-519`）、OpenMP 并行阈值
  `multithread_threshold_depth=5`（`search.cc:27,140-152`）；§4 验证（`ProbabilisticVerifier` 指纹
  `probabilistic_verifier.cc:8-60` vs `FormalVerifier`、`OutputMatch` `output_match.h`、
  `verify`/`save_results` `search.cc:521-567`；**只查功能等价不查性能**）；§5 生成
  （`kernel.py:606-692` 扫 `pipeline_stages∈{2,3,4}×num_warp_groups∈{2,3,4}` + profile 选优，
  `max_smem_size` 检查 `kernel.py:440-443`；`generate_cuda_program` `core.pyx:1398` →
  `transpile()` `transpile.cc:792` → `Transpiler::generate_code` `transpiler.h:127-134` 的 resolve/
  schedule/codegen 串；`transpile_kn_custom_op` `transpiler_tb.cc:253`、`get_threadblock_schedule`
  `sched_tb_graph.cc:144`/`TBSched` `sched_tb_graph.h:44-52`）；§6 与论文 §4.1/§4.2 逐条对照表
  （快照论文行 462-468/1132-1142/1835-1836）；§7 回填 `docs/07` §0「superoptimizer vs 手写并存」与
  `docs/08`（只认 `KN_CUSTOMIZED_OP` → 两条路下游等价）；§8 判据映射（只贡献 C3 范围，同 `docs/14`）；
  §9 局限（纯走读；`abstract_expr`/`range_propagation`/`symbolic_graph` 未逐行；transpiler 只给管线；
  未复现 benchmark 数；多 GPU/NVSHMEM 未展开）。**纯源码走读（资料分析），本轮无新实验。**
  配套博客 post `mega-kernel-16-mpk-superoptimizer`（weight 17）并更新 `mega-kernel-00-index` 第 16 项。

### P102 搜索器辅助模块源码走读（`docs/16-mpk-search-passes.md`）

- [x] **P102-1** 补掉 `docs/15`（P101-1）**§9 明留的尾巴**——「只精读主搜索循环；`abstract_expr/`（代数
  表达式等价类）、`range_propagation/`（range 推导）、`symbolic_graph/`（符号化搜索）三块只给了接口，
  未逐行走读其内部算法」。把主搜索循环之外的**三个判断器**逐行走通。见
  `docs/16-mpk-search-passes.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与
  `docs/06`/`07`/`08`/`14`/`15` 同快照；纯源码走读、无新实验）。七节：
  §0 定位（`abstract_expr` 管「算的是什么」/`range_propagation` 管「读哪一段」/`symbolic_graph` 管
  「形状是变量否」三块正交，附主循环调用点表）；§1 `abstract_expr`——C++ 15 个节点类型
  （`abstract_expr.h:42-231`，`RMS`/`Red` 把归约长度折进节点身份，`abstract_expr.cc:328-330` degree 1
  = 不归约）双表示 `to_string`/`to_egg`，Rust egg e-graph（`abstract_subexpr/src/lib.rs`：代数律
  `:38-53`、`sum` 换序 `:55-62`、**动态质因数分解规则**给定 `(sum N a)⇄(sum j (sum k a))` `:72-105`、
  饱和 runner `with_iter/time/node_limit` `:133-144`、`get_egraph`/`egg_equiv` FFI `:121-183`），
  `preprocess` 对每个输出 `initialize_final_expr`（`search.cc:487-495`）、候选 `check_abstract_expr`
  查子表达式（`search.cc:504-519`，**`constraints` 参数有默认值但函数体未用**）；§2 `range_propagation`
  ——`Range`(`range.h`)/`RangeSet`+`PropagationPath` 防环(`range_set.h:41-51`)/`TBRange`(tensor|block|
  forloop `tbrange.h:30-34`)；forward/backward/multiplicative_interact 逐算子规则（`irange.cc:291-464`，
  `EXP_AS_IDENTITY` `:296-305`、matmul 转置扩维 `:443-452`）、跨 KN/TB 递归（`:472-521`）、d2s/s2d 网格
  映射（`:55-268`，明注只处理单 STensor 情形 `:88-90`）、驱动器 `range_propagate`（`:1050-1062`）、
  入口 `preprocess` 建 `init/target_ranges`（`search.cc:482-484`）、`check_range`（`search.cc:338`）
  ——**但两个 `check_range` 重载都在首行 `return true;`（`irange.cc:1068-1069,1085-1086`）⇒ 算法已写完
  但当前不贡献剪枝**；§3 `symbolic_graph`——`TensorDimExpr` 树(Var/Const/Add/Mul/Div/Pow/Ite)带
  `get_value`+`to_z3`（含 **log-scaled** 把乘变加 `tensor_dim_expr.cc:355-382`）、`SymbolicTensorDim`
  结构等值、`SymbolicMap` 映射矩阵（默认给每格一个布尔变量 `symbolic_map.cc:6-16`）、`SymbolicKN/TBGraph`
  与按语义生成的维度约束（matmul 批量维=1+收缩维相等 `symbolic_graph.cc:209-224`、add/mul `EQUAL_OR_ONE`
  `:152-161`、映射矩阵行/列 0-1 `:308-330`）；**关键落差**：约束可满足性**两套引擎**——TB 用
  `TensorDimConstraints`+**z3** `s.check()`（`tensor_dim_constraints.cc:38-47`），KN 用
  `check_satisfiability` 的 **rule-based 启发式**（注释 `// rule-based checking for now`、
  `tensor_dim_constraint.cc:88-148`，几乎不拒绝）；§4 符号路径三处「已实现未接线」——
  `instantiate_symbolic_graph` 首行 `++counter; return false;`（`search.cc:836-842`）、
  `generate_kernel_graphs_symbolic` **无任何调用者**（`search_c.cc:106` 只调 `generate_kernel_graphs`）；
  §4 调用点汇总表（哪一步真剪枝：代数子表达式/算子序=是，维度约束联合/区间范围=否）；§5 与论文
   §4.1/§4.2 对照 + 判据映射（三块都只贡献 C3）；§6 局限（纯走读、未量化剪枝收益、egg 规则集/z3 与
   启发式等价性未证）。**纯源码走读（资料分析），本轮无新实验。**
   配套博客 post `mega-kernel-17-mpk-search-passes`（weight 18）并更新 `mega-kernel-00-index` 第 17 项。

### P103 MPK 多卡/NVSHMEM 路径源码走读（`docs/17-mpk-multigpu.md`）

- [x] **P103-1** 补齐 `docs/15`（P101-1）**§9 明留的尾巴** 与 `docs/16`（P102-1）**§6 明确转引的
   同一条**——「**多 GPU/NVSHMEM 路径未展开**：`resolve_distributed_config`（`transpiler.h:53-58`）的
   `gpu_dim`/NVSHMEM 分支与 `multigpu.py` 的张量并行切分未走读」。前面六篇源码走读全是单卡视角，
   而论文标题/摘要反复强调 multi-GPU。见 `docs/17-mpk-multigpu.md`（快照 `mirage-project/mirage`
   分支 `mpk` commit `f9eb70c`，与 `docs/06`/`07`/`08`/`14`/`15`/`16` 同快照；纯源码走读、无新实验）。
   §0 编译栈各层的「GPU 维」坐标表（结论：MPK **没有**为多卡另造 runtime，只加两样东西）；
   §1 启动栈（MPI 定 world/rank `mpk.py:298-312`、`use_nvshmem = world_size>1`、
   `nvshmemx_init_attr(MPI_COMM_WORLD)` + `nvshmem_barrier_all`、launch/finalize 各一道跨卡 barrier、
   **每 rank 各生成 tGraph、各编一个 launcher `.so`**）；§2（headline）**多卡被迫单 kernel**——
   `nvshmemx_collective_launch` 顺序启动、不支持 split worker/scheduler
   （`persistent_kernel.cuh:1988-1990,2025`），故多卡下「C1 单次 launch」从优化变**唯一路径**；
   §3 event id 多卡编码 `nvshmem_tag:16|owner:16|idx:32`（`runtime_header.h:75-80`）与语义差
   （普通 event 由本 GPU worker 触发、**NVSHMEM event 由别的 GPU 触发**）；
   §4 编译期：`get_num_subtasks` 只对 `ALLGATHER_STRIDED_PUT` 返回 `num_gpus-1`
   （`runtime.cc:66-74`）、subtask 的 `trigger_event` 改写成**目标 GPU 的 event id**
   （`:129-150`/`:604-639`）、`layer_is_multigpu`（`:657-659`）、而 `TILE_ALLREDUCE` **不打标签**
   （`:442-446`）；§5 设备侧：worker 对 NVSHMEM event 用 `nvshmem_signal_wait_until`
   （`persistent_kernel.cuh:1095-1102`）、触发分支为空操作（信号在 task 内部、
   `:1234-1248`）；§6 **两种 AllReduce**：策略 A `AllGather+Reduce`
   （`multigpu.py:156-184`；`putmem_nbi` + `nvshmemx_signal_op` 到目标 GPU 计数器
   `tasks/ampere/allreduce.cuh:31-58` + 本地 `reduction_kernel:67-92`，=论文 §6.5 行 1563-1567）
   与策略 B `NVSHMEM Tile`（`multigpu.py:187-208`；Hopper `nvshmemx::tile_sum_reduce_block` NVLS
   `ONE_SHOT_PULL_NBI` `tasks/hopper/allreduce.cuh:29-63`、Blackwell **手写** `multimem.ld_reduce`
   + 内联散布式 barrier 以保共驻 `tasks/blackwell/allreduce.cuh:377-514`；是**本地 task**、
   跨卡同步在 task 内）；自动选择（SM≥90+VMM+multicast+peer → Tile，否则 A，`multigpu.py:224-247`）；
   附对照表；§7 张量与 TP residual「只 rank0 加、否则 allreduce 重复计 world_size 次」
   （`persistent_kernel.py:2347-2361,2470-2474`；交叉引用 `../mega-moe/`）；
   §8 host launcher per-rank（`persistent_kernel.py:38-141`）、host 无 per-token 通信调度；
   §9 与论文对照表（**最大落差：论文只描述策略 A，SM≥90 主路径已是 NVLS tile=策略 B，论文未提**）；
   §10 判据 C1–C5 映射（多卡把 C3 扩到「跨卡算子+通信」、C4 从设备内扩到设备间、C1 从优化变硬约束）；
   §11 局限（纯走读未跑多卡、未走 `nvshmemx_collective_launch` 内部、策略 A 布局假设、
   team 分配代价未测；`docs/15` 另两条尾巴仍未覆盖；跨卡重叠只有单机替身 `docs/04` §14）。
   **纯源码走读（资料分析），本轮无新实验。** 配套博客 post `mega-kernel-18-mpk-multigpu`（weight 19）
   并更新 `mega-kernel-00-index` 第 18 项。

### P104 MPK 编译器 codegen 源码走读（`docs/18-mpk-transpiler.md`）

- [x] **P104-1** 补掉 `docs/15`（P101-1）**§9 末句明留的尾巴**——「`transpiler` 只给管线结构
   （`src/transpiler/*` 约 8 千行），**未逐条读 SASS 级生成逻辑**」（P103-1 §11 记为「`docs/15`
   另两条尾巴仍未覆盖」之一）。把前半段 `docs/15` §5 停在「`transpile()` → resolve 阶段」的
   **下半段**走完：一次 `KNGraph.compile()` 怎样把 thread-block 图变成能被 `nvcc` 编译、能 launch 的
   CUDA。见 `docs/18-mpk-transpiler.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
   与 `docs/06`/`07`/`08`/`14`/`15`/`16`/`17` 同一快照；纯源码走读、无新实验）。§0 入口链路
   （`kernel.py:396/431` → `core.pyx:1420` → `transpile.cc:461/793` → `transpiler.h:128`；
   生成的 `.cu` 与 `runtime.h:14/15/31/47` 的两个生成函数 + `HARD_CODE` 拼成 `.so`）；§1 ctor 两段
   **图归一化**（online-softmax 重写 `transpile.cc:101/276-301`；把复合 TB 算子拆原语
   `transpile.cc:482-767`：`FORLOOP_ACCUM_RED_LD_SUM`→accum+reduction `:669`、`RED_LD_RMS` `:682`、
   `RMS_NORM` `:734`；收尾断言 `:769-789`）——这是论文 §4.1「decomposition」在编译器侧的落点；
   §2 五步 resolve（`transpiler.h:52` / `resolve_dtensor_meta.cc:27` / `resolve_tb_fusion.cc:11`
   （只融「单消费者 elementwise unary 或 accum_no_red」到前驱）/ `resolve_tensor_layout.cc:121` /
   `plan_dtensor_memory.cc:22`（**只分配不复用、128B 对齐**））；§3 `transpile_ugraph`
   （`transpiler_kn.cc:332`：header/`_execute_mugraph` 逐 op 展开、非 custom 算子→`kn::gemm`/
   `ElementUnaryKernel`/`ElementBinaryKernel`/`ReductionKernel`、`get_dtensor_ptr:92` 三类指针、
   `KN_CUSTOMIZED_OP:614` 按 cc 分派三后端、Hopper `<<<>>>`/Blackwell `launch_kernel_on_cluster`、
   `TranspileResult:857`）；§4 单个 custom op 的 **schedule + swizzle + memory plan 三件套**
   （`sched_tb_graph.cc:144`：`TBSchedOpMeta` 决策含 chunked/async `:19-70`、**192 寄存器预算决定
   accum 在寄存器** `:239-260`、三段 `pre/loop/post`、`ops2sched` 插 `SYNCTHREADS` + `refine_opmeta_on_chain`；
   swizzle `plan_tb_swizzle.cc:9` XOR/shift；memory `plan_stensor_memory.cc:256` 含双缓冲偏移）；
   §5 codegen 形态（`transpiler_tb.cc:253`：签名 `:280`、SMEM 视图 `:322`、`Input*Copy` 原子 `:342`、
   软件流水 prologue `:836` + 主循环 `:1421` + `cp_async_fence/wait<1>` + `SWAP` 双缓冲、融合 epilogue
   `:861`；Hopper 升格 **warp specialization**——producer warpgroup 发 TMA `transpiler_tb_hopper.cc:900`、
   consumer 跑 wgmma `:1426`、`hopper_async_pipeline.consumer_wait/release` `:283/1472`；`tma_reg`/`mma_reg`
   的 `wg_decrease/increase_regs` **被注释** `:902-906`，与 `docs/07` §2 同一事实）；§6 与论文 §4.2
   四项 intra-SM 优化逐条对照（software pipelining / register reuse / layout-swizzle 均有落点，§4.1
   自动分区落在搜索器）；§7 判据 **只贡献 C3**（跨 customized op 仍 kernel-per-operator，C1/C4/C5 缺位）；
   §8 局限（纯走读未跑、未逐句展开 CuTe 模板装配、Hopper/Blackwell 只给骨架、runtime 模板当黑盒、
   未复现 benchmark、`docs/transpiler/transpiler.md` 快照里不存在）。**纯源码走读（资料分析），
   本轮无新实验。** 配套博客 post `mega-kernel-19-mpk-transpiler`（weight 20）并更新
   `mega-kernel-00-index` 第 19 项。

### P105 MPK runtime 模板库源码走读（`docs/19-mpk-runtime-templates.md`）

- [x] **P105-1** 补齐 `docs/18`（P104-1）**§8 明留的尾巴**——「`include/mirage/transpiler/runtime/**`
  （`kn::gemm` / `ElementUnaryKernel` / `ReductionKernel` / `InputAtom` / `Hopper_Matmul`）是被生成
  代码调用的库，只当作黑盒引用」。见 `docs/19-mpk-runtime-templates.md`（快照
  `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/18` 同一快照；纯源码走读、
  无新实验）。把「被生成 CUDA 调用的运行期肉身」逐层走通——**23 个头、3017 行**。§0 两层 namespace
  坐标（`kn::` = 一整个张量算子 = 一次 launch 的兜底；`tb::` = 在一个 `custom_kernel_N` 内按 thread
  调用的算子实现）；§1 入口 `runtime.h`（前向声明生成函数 `:14/15`、`extern "C"` 首调用 `_init`
  `:31-43`、Python wrapper `:47-60`；runtime `config.h` 的 `USE_NVSHMEM/NUM_GPUS` 开关 `:4-15`——
  且注意 `threadblock/utils.h:3` 指的是另一个 `include/mirage/config.h`，`NUM_THREADS_PER_GROUP=128`
  `:36-38`）；§2 `kn::` 四个整张量算子（element_unary `BLOCK_SIZE=512`、element_binary 的 stride-0
  广播 `:52-89`、reduction 的 `REDUCTION_FACTOR` unroll、**matmul 直接包 `cublasGemmStridedBatchedEx`
  `kernel/matmul.h:117-140`**）；§3 `tb::` 层——G↔S 四类拷贝原子（NonChunked/Chunked sync、
  `cp.async.cg` 16B `input.h:150-152`、Hopper TMA `:166-254`、Blackwell TMA multicast `:353-384`）、
  epilogue 寄存器链（`epilogues.h` 的 `Epilogue*→EpilogueStore`、`++epilogue_scalars` 指针前移）、
  forloop_accum/reduction 含 **online-softmax 的 max/diff**（`reduction.h:82-150`）、三档 matmul
  （通用 CuTe S2R/R2S + ldmatrix 选择 `matmul.h`、Hopper `GMMA::ss_op_selector` + cooperative 2-CTA
  `hopper_matmul.h:90-107`、Blackwell UMMA+TMEM 累加器 `blackwell_matmul.h:125-192`）、流水
  （`PipelineTmaAsync` / `PipelineTmaUmmaAsync` + cluster `pipeline_init_arrive/wait`）、profiler tag/
  globaltimer、`wg_decrease/increase_regs` 受 `MIRAGE_GRACE_HOPPER` 门控（`utils.h:33-51`）；§4 生成
  代码 ↔ 模板四条接口约定；§5 **headline——同一套硬件惯用法（warp specialization+TMA+wgmma/UMMA+
  mbarrier+寄存器 epilogue）在仓库里实现了两遍**（`docs/07` 手写 MPK task vs 本 runtime 库），为
  `docs/15` §7 的「工程落差」补运行期证据；§6 论文 §4.2 四项 intra-SM 优化逐条对照；§7 判据
  **只贡献 C3**；§8 局限（纯走读未跑、未逐句展开 CuTe 模板装配、`triton_kernels.py`/参数生成逻辑未走、
  Hopper/Blackwell 只给骨架）。**纯源码走读（资料分析），本轮无新实验。** 配套博客 post
   `mega-kernel-20-mpk-runtime`（weight 21）并更新 `mega-kernel-00-index` 第 20 项。

### P106 MPK 另一条 codegen 后端：Triton 路径源码走读（`docs/20-mpk-triton-backend.md`）

- [x] **P106-1** 补齐 `docs/19`（P105-1）**§8 明留的尾巴**——「`triton_kernels.py`（Python 侧）未走」。
  顺这条尾巴往上，发现它是一整条**平行于 NVCC/CUDA transpiler 的第二条 codegen 后端**。见
  `docs/20-mpk-triton-backend.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与
  `docs/06`–`docs/19` 同快照；纯源码走读、无新实验）：
  §0 四坐标链路（`kernel.py:543/553/706-745` → `triton_profiler.py` → `core.pyx:1461-1469` →
  `src/triton_transpiler/*` → 运行期 `triton_kernels.py`），并指出 `search_c.cc:16` 的 `backend`
  参数**从未被引用**、只影响搜完后的编译支路；
  §1 前端选择（`__call__`/`triton_call :300-315`、`KERNELS_PATH` 注入 `:711`、`importlib` 加载
  `execute_mugraph` `:733-740`；`demo/triton_rms_norm.py` 为最小示例）+ `triton_profiler.py`
  **每候选图起独立子进程 profile**（`:160-185` `subprocess` + 解析 stdout 的 `Average time`）；
  §2 `TritonTranspiler` ctor（`transpile.cc:41-298`）**克隆新图**、`dtensor_mapping` 重建 op、
  customized op 把复合 TB 算子**手工拆开**（`FORLOOP_ACCUM_RED_LD_RMS` `:228-241`、`TB_RMS_NORM`
  `:251-261` 且 `:256` 的 reduction 被注释）、末尾后置校验（`:280-297`）；
  §3 `transpile_ugraph`（`:299-565`）四段生成 Python 模块（header/inputs 硬编码 fp16/`execute_mugraph`
  /逐 op 调用 `ops.matmul`·`elementwise_*_kernel`·`reduce_sum_kernel`·`custom_kernel_N`）与
  `output_shapes`；§4 `transpile_kn_custom_op`（`transpile_tb.cc:209-643`）——**无 warp specialization/
  TMA/wgmma**，纯 `tl.arange`+`program_id` 直译、形状**对齐到 2 的幂** + pad 掩码、累加器恒 fp32、
  两遍（accum 前/后）遍历、`tl.dot` 恒 `.to(fp32)` 且 matmul 的 mask **算了没用**；
  §5 运行期 `triton_kernels.py`（66 行，3 个 `@triton.jit`）——**OP 魔数 = C++ 枚举数值**（`type.h`
  KN `:82-101`）形成隐式 ABI，`KN_POW=1203`/`KN_SILU=1104` 无分支只 device_print、`128` 硬编码；
  §6 **headline——「Triton codegen」也实现了两遍**：旧 path `src/kernel/triton_code_gen.cc`
  （`Graph::generate_triton_program`，序列化 `NewKernelParams`+`make_block_ptr`+`out_dtype=fp16`）已成
  **事实死代码**（仅剩 `core.pyx:901-906` 定义、无 Python 调用者），新 path 直读图走 arange——附七维
  对照表；另指出 `TritonTranspilerConfig.target_cc` **已接线未使用**（与 `docs/16` 同型）；
  §7 与论文 §4.2 对照 + **C1–C5 只贡献 C3**（再次印证「换 codegen 目标语言拿不到 C1/C2/C4/C5，
  必须改执行模型」）+ 生产 `mpk` 前端不调 superoptimize（接 `docs/15` §7）；
  §8 局限。**纯源码走读（资料分析），本轮无新实验。** 配套博客 post
  `mega-kernel-21-mpk-triton-backend`（weight 22）并更新 `mega-kernel-00-index` 第 21 项。

### P107 MPK 自动路径（Blackwell 版）源码走读（`docs/21-mpk-blackwell-transpiler.md`）

- [x] **P107-1** 补齐 `docs/18`（P104-1）**§8 明留的尾巴**——「Hopper/Blackwell 只给骨架；
  Blackwell 的 `2sm_mma` / `launch_kernel_on_cluster` 只标出调用点与行号，未逐行走读其内部」
  （Hopper 的一半已由 `docs/07` 覆盖）。见 `docs/21-mpk-blackwell-transpiler.md`
  （快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/20` 同快照；
  CuTe/CUTLASS 引用快照自带的 `deps/cutlass`、本机 `f3fde583`；**纯源码走读、无新实验**——
  Blackwell/SM100 在目标卡 H100/sm90 不可执行，与 `docs/17` 同口径）：
  headline——**Blackwell 后端 = Hopper 后端 + 「把计算单元从 1 CTA 扩到 2 CTA」的一整条链**。
  §1 入口三硬约束（`transpiler_tb_blackwell.cc:368-377`：`target_cc==B200` / warpgroup 数上限 /
  `num_threads==(consumer+producer)*128`）；§2 集群与 2SM（`cluster_shape :410-416`、
  `cluster_layout_vmnk`+`elect_one_cta :519-526`、`sm100_make_2sm_trivial_tiled_mma :504-512`，
  CUTLASS 侧 `sm100_common.inl:373-429` + `static_assert(M==128||256) :377`）；§3 TMEM + MMA mbarrier
  （`Allocator2Sm` / `arrive_cnt=forloop_range*cluster/2` / `consumer_sync_mask`，
  `generate_Tmem_mbarrier_init_code :319-345`，SMEM 三类区域 `:428-441`）；§4 输入
  （`BlackwellAsyncPipeline` = `PipelineTmaUmmaAsync` `blackwell_pipeline.h:15-16`；
  `InputTMAAsyncCopy_Blackwell` 的 **multicast mask：A 沿 N / B 沿 M**，`input.h:352-360`）；
  §5 布局（`partition_shape_A/B`+`sm100_smem_selector`+`UMMA::tile_to_mma_shape`，`blackwell_matmul.h:56-88`）；
  §6 `plan_tb_swizzle_blackwell.cc`（16B chunk 下界 + 128B bank 对齐）；§7 主循环（producer warpgroup 发
  TMA `:1036-1078`、consumer `:1465-1508`、`elect_one_cta && elect_one_warp` 门 `:1143-1146`、
  `umma_arrive_multicast_2x1SM`→`tcgen05.commit.cta_group::2...multicast::cluster` `:1202-1204`/CUTLASS
  `barrier.h:813-828`）；§8 累加器（TMEM `get_mma_tC`/`write_tC_to_sC`、in-reg 写回前
  `wait_barrier` `:1536`、融合 epilogue `:1083-1121`）；§9 host 侧（host 再定义 `TiledMMA`
  `transpiler_kn.cc:747-778` + `cutlass::launch_kernel_on_cluster :824-831`，对照 Hopper 朴素 `<<<>>>`
  `:819-823`）；§10 Hopper↔Blackwell 逐维对照表；§11 论文 §4.2 对照（语义一致、实现升级；2SM 路径
  **论文未提、快照新增**，同 `docs/17`）；§12 C1–C5（**只贡献 C3**）；§13 局限（未跑 SM100 本体、
  未逐行展开 CuTe 模板装配、`cluster_dim` 来源未深挖、循环内单 warp 门未验证）。**纯源码走读
  （资料分析），本轮无新实验。** 配套博客 post `mega-kernel-22-mpk-blackwell-transpiler`（weight 23）
  并更新 `mega-kernel-00-index` 第 22 项。

### P108 汇总 5：MPK 源码走读手册（`docs/22-mpk-compiler-manual.md`）

- [x] **P108-1** 把 P7–P11、P21、P100–P107（`docs/06`–`docs/08`、`docs/14`–`docs/21` 共**十一篇**
      MPK 线上实现源码走读）收口成**一页编译栈总图 + 决策表 + 落差清单**（补齐 `docs/05` 只覆盖到 P6、
      `docs/11`–`docs/13` 只覆盖实验侧、源码走读系列**无汇总手册**的缺口）。见
      `docs/22-mpk-compiler-manual.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
      CuTe/CUTLASS 用快照自带 `deps/cutlass` 本机 `f3fde583`；**纯资料汇总，无新实验**）。结构：
      §0 headline——**源码里有两条链、共用 `KNGraph` 后在 `KNCustomizedOp` 处分叉**（生产路径
      `annotated_graph`→runtime→常驻 megakernel 拿 C1–C5；自动路径 `superoptimize`→transpiler→
      kernel-per-operator 只到 C3）；§1 编译栈七层坐标表（前端/搜索/图 IR/codegen/运行层/device/多卡 →
      哪条链用哪层 → 分篇）；§2 生产路径逐层（`docs/14`/`docs/08`/`docs/06`/`docs/07`：论文 §4.1 五步 →
      `annotated_graph.cc` 十步 (a)–(j)，normalization = 拒绝 case2/3 + LCM 降级；128 worker/16
      scheduler；单 task = 一个 CTA）；§3 自动路径逐层（`docs/15`/`docs/16`/`docs/18`/`docs/19`/
      `docs/20`/`docs/21`：一台 superoptimizer 的 CUDA/NVCC、runtime 模板库、Triton、Blackwell 四个
      后端）；§4 多卡（`docs/17`：C1 从优化变硬约束）；§5「**同一套惯用法实现了两遍**」三行清单；
      §6「**已实现但未启用/未接线**」十一行清单（跨 `docs/08`/`docs/16`/`docs/18`/`docs/19`/`docs/20`/
      `docs/21` 汇合，含 `check_range return true`、`instantiate_symbolic_graph return false`、
      `search_c.cc:16 backend` 未引用、旧 Triton codegen 死代码、两份缺失的自引用文档、两处被注释的
      `setmaxnreg`、生产前端不调 `superoptimize`）；§7 论文描述 vs 线上实现落差清单；§8 判据 C1–C5
      逐层总映射（**只有运行层拿 C1/C2/C4/C5**）；§9 源码坐标总表（文件/行数/角色/分篇）；§10 局限；
      §11 关键坐标。**纯资料汇总（资料分析），本轮无新实验**；配套博客 post
      `mega-kernel-23-compiler-manual`（weight 24）并更新 `mega-kernel-00-index` 第 23 项。
      **本专题任务清单此前全完成，本轮为源码走读系列新增的汇总增量；与 `docs/11`/`docs/12`/`docs/13`
      （实验侧三本手册）并列。**

### P109 汇总 6：全专题总纲 v2（`docs/23-megakernel-capstone.md`）

- [x] **P109-1** 补齐「第一代收官篇 `docs/05`（P6-1）只覆盖到 P0–P6」的缺口——此后专题推进了 100+
      增量（源码走读 `docs/06`–`docs/22`、实验 `docs/04` §6–§85、三本实验手册 `docs/11`–`docs/13`、
      手工先例 `docs/10`），却**没有一份把这些结论重新收口的总纲**（`docs/22` 只收源码侧、
      `docs/11`/`docs/12`/`docs/13` 只收实验侧三块）。见 `docs/23-megakernel-capstone.md`
      （快照 `f9eb70c`；**纯资料汇总，无新实验**，不改写 `docs/05` 正文、不引用未复现的 `benchmark/`
      数字）。结构：§1 一页纸 v2（三类损失 → 机制 → 判据 + 关键修正「**C1/C2/C4/C5 只在运行层拿到**」，
      `docs/22` §8）；§2 结论地图五层（范式/动机、编译层、运行层、数据面、部署）+ 编译层两链分叉 +
      运行层六部件对上 MPK；§3 **实测校准表 v2**（6 张：损失侧 / 成本侧 / 调度正确性侧 / 数据面侧 /
      部署侧 / 降级手段天花板）；§4 **工程坑清单 14 条**（把「已知坑」与实验方法论口）；§5 何时该/不该用
      v2（补实测 break-even + 「少而重」结论）；§6 与 v1 差异 + 坐标。**纯资料汇总（资料分析），本轮无新
      实验**；配套博客 post `mega-kernel-24-capstone`（weight 25）并更新 `mega-kernel-00-index` 第 24 项。
      **与 `docs/05`（v1 一页纸）、`docs/22`（源码侧手册）、`docs/11`–`docs/13`（实验侧三手册）并列，
      本文是整专题第二代总纲。**

### P110 生产路径 task 目录源码走读（`docs/24-mpk-task-catalog.md`）

- [x] **P110-1** 补齐 `docs/14`（P100-1）**§7 明留的尾巴**——「未逐行走读全部 95 个任务分支」：
      把 `src/kernel/graph.cc` 的 `register_task` 95 分支大表**打开**，给出 MPK 生产路径的
      **`TaskType` 能力边界面**。见 `docs/24-mpk-task-catalog.md`（快照 `mirage-project/mirage`
      分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/22` 同快照；**纯源码走读（资料分析），本轮无
      新实验**）。headline——**`enum TaskType`（`runtime_header.h:87-225`）按「硬件代际 / 模型家族」
      切成预留 ID 块**：0/10 控制标记、101-121 通用 compute、150-198 Hopper、200-204 scheduler、
      230-299 SM100（最多，55 条）、300-349 多卡 NVSHMEM、350-399 Inkling、400-449 GLM、450-499
      GPT-OSS；块间留洞（234/237-247/250 等）是「只增不改」的后果——`TaskType` 进 `TaskDesc`/
      序列化（`docs/06` §2），删号不能回收。§2 列全 116 条命名条目（12 个 `*_BEGIN/_END` 占位符）
      + 每条的 `register_task` 名字与 `(num_inputs, num_outputs)`。§3 精确计数：**96 个
      `name == "..."` 字符串 / 95 处 `task_config[op]=` 赋值 → 92 个不同 `TaskType`**（`docs/14`
      §3.2 的「95 分支」被坐实），4 组「多 name 共用一 Type」（`silu_mul`/`moe_silu_mul`、
      `sampling_sm100`/`_reduce_`、`quantize_fp8_sm100`/`_f32scale_`、`linear_with_residual_sm100`
      `||` `linear_with_bias_sm100`——源码注释自陈 *"two variants of one task type"*）。§4 求差得
      **只有 3 个可执行 `TaskType` 无前端生产者**（`TASK_ATTENTION_2`/`TASK_PAGED_ATTENTION_2` 遗留、
      `TASK_ALLREDUCE` 注释 *"will be removed soon"*，后者旁证：`allreduce_layer` 实走
      `nvshmem_*`+`reduction`）；`TASK_ATTN_SM100` **确由 `paged_attention_sm100` 登记**（抽 7/8 输入
      由 `params[11]>0` 定）。§6 与论文 §4.1（自动路径）、`docs/02`（DeepGEMM MoE W13/W2）、
       `docs/17`（两种 AllReduce）、`docs/23` §5「少而重」对照。判据不变（生产路径的 task 定义）。
       配套博客 post `mega-kernel-25-mpk-task-catalog`（weight 26）并更新 `mega-kernel-00-index`
       第 25 项。

### P111 前端模型组装与请求/元数据管理层源码走读（`docs/25-mpk-model-builder.md`）

- [x] **P111-1** 补齐 `docs/14`（P100-1）**§7 明留的尾巴**——「未走读模型 builder 的完整组装逻辑
      （`models/*/builder.py`）、分布式/张量并行切分（`multigpu.py`）、请求/元数据管理（`mpk.py` 的
      `MPKMetadata`）」（`docs/24` §头部也把它列为不在其范围的三条尾巴之一）。见
      `docs/25-mpk-model-builder.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
      与 `docs/06`–`docs/24` 同快照；**纯源码走读（资料分析），本轮无新实验**）。补的是
      `docs/14`（层 API）与 `docs/24`（task 大表）之间**更上一层**的问题：**谁把 95 个手工
      `*_layer()` 拼成一个模型**。headline——**megakernel 的融合边界是这一层用 Python 手写死的**：
      模型架构知识被写成 `builder.py` 类，而非编译器从高层算子自动搜出（自动路径 `docs/15` 生产前端不调）。
      结构：§0 定位（模型 builder + `MPKMetadata` 决定「哪些算子进同一张图/哪些中间量进对称显存」，
      只划 C3 范围、不拿 C1/C2/C4/C5，`docs/22` §8）；§1 注册与发现（`model_registry.py:5-29` 装饰器
      多别名 → `models/__init__.py:1-17` `rglob("builder.py")` 自动 import=注册 → `MPK.build():471-480`
      取类，两入口 `build_from_model`/`build_from_config`；dflash/eagle3 不注册=草稿 builder）；
      §2 `MPKMetadata`（`mpk.py:15-83` 六组字段）+ `MPK` 生命周期（`init_mpi:298-312`、
      `compensate_meta_tensors:314-352`、`runner→generate_task_graph:482-489`(带 `num_gpus/my_gpu_id`)
      →`compile:491-494`→`__call__:499-508`）+ **mode→`-DMODE_*` 宏**（`persistent_kernel.py:320-334`，
      六种模式；容量常量 `-DMPK_MAX_*:336-345`）；§3 以 Qwen3 为规范样本逐段走
      `build_from_dict:560-690`（vocab pad 153600、attach 输入/RoPE、`new_intermediate_tensors`、
      embed、`build_layers`、final norm+lm_head、sampling/argmax 两段式）与 `build_layers:288-558`
      （手写层序 + 手写 grid/block；**`use_splitk = target_cc==100` 的硬件代际分支**；残差靠张量身份
      `self.x=` 串联）；§3.3 关键机制 **`new_intermediate_tensors` 两条路**（`online_notoken`=固定
      tensor+`attach_input` 裸指针 vs 其它=`new_tensor`+`io_category`）——**`cuda_tensor` vs
      `nvshmem_tensor` 由 builder 手选，决定哪些中间量进对称显存/跨卡**（对上 `docs/17` §3）；
      §4 `models/utils.py`（`shuffle_tensors`/`inplace_shuffle_tensors` 权重交织、
      `grid_for_rmsnorm_linear_layer:3-15`/`grid_for_splitk_linear_layer:17-34`（含 Qwen3-32B o_proj
      的错位 TMA 描述符 bug 记录））证明前端仍是「人肉把维度换算成 task 网格」；§5 多卡
      （builder 内 `//world_size` 手写切分 + `multigpu.py:17-298` 策略模式：`CollectiveCapabilities`
      查 peer/VMM/multicast/posix 四项能力、`auto_select_allreduce_implementation:224-247`
      `cc>=90+VMM+multicast+peer→NvshmemTile` 否则 `AllgatherReduce`、`allocate_nvshmem_teams:74-96`；
      **allgather/broadcast/reduce_scatter 三种 `auto_select_*` 仍 `NotImplementedError`**）；
      §6 其它模型家族（GLM-4.6/Inkling/GPT-OSS/DeepSeek-V3/DFlash/Eagle3 特色：MoE shared 折叠成
      expert、MLA 权重吸收、FP8 UE8M0、投机草稿）——**正是 `docs/24` 家族 ID 块（350-399 Inkling /
      400-449 GLM / 450-499 GPT-OSS）的前端生产者**；§7 同层附属件（`kv_planner.py` 异构 KV 流共享页池
      → `PersistentKernel.kv_groups`；`base_dynamic_shard_loader.py` 声明式 TP/EP 分片加载；
      `online_pinned_runtime.py` 双锁页环 host 泵=常驻 kernel 连续服务的落地）；§8 论文 §4.1 对照 +
      **判据映射（本层不拿任何 C1–C5，只划 C3 范围 / 为 C4/C5 备输入）**；§9 局限；§10 坐标。
      配套博客 post `mega-kernel-26-mpk-model-builder`（weight 27）并更新 `mega-kernel-00-index` 第 26 项。

### P112 模型家族 builder 走读（`docs/26-mpk-family-builders.md`）

- [x] **P112-1** 补齐 `docs/25`（P111-1）**§9 明留的尾巴**——「未逐行展开每个家族的 `_build_*`：
      GLM/Inkling/DeepSeek-V3 的 MoE/MLA 管线只给算子名与文件头架构说明，未逐行走读其权重折叠/吸收
      细节」。见 `docs/26-mpk-family-builders.md`（快照 `mirage-project/mirage` 分支 `mpk` commit
      `f9eb70c`，与 `docs/06`–`docs/25` 同快照；**纯源码走读（资料分析），本轮无新实验**）。
      headline——**模型家族的结构差异全部在 builder 层被手工「折平」成统一的标准 task 管线**（运行层
      只认 `linear`/`rmsnorm`/`paged_attention`/`moe_w13`/`silu_mul`/`moe_w2`/`moe_mul_sum_add`/
      `allreduce`）。结构：§0 定位（家族 builder 规模表：Qwen3 696 / DeepSeek-V3 **2255** / GLM 454 /
      GPT-OSS 374 / Inkling 705；dflash/eagle3 不注册=草稿子图）；§1 **DeepSeek-V3** 主走读——
      MLA 权重吸收（`kv_b_proj`→`q_b_proj`，docstring `:9-10`；V 反吸收融进 o_proj `:776-792`）、
      FP8 UE8M0 重定标（`_requantize_fp8_for_ue8m0:475-542`：dequant float32 块 scale→2 的幂→
      重新量化→打包 uint32 列主序）、`k_pe` pad 64→128 凑 `MMA_M`（`:636-653`）、
      `_build_mla_attention_layer:563-792` 七步 + prefill/decode 双 dispatch + TP2/4/8 `mla_mtp_decode`
      分支（`:743-774`）、`_build_moe_mlp:832-1160`（路由张量专家维在前 `:842-860`、sigmoid+bias
      top-k `:898-911`、FP8 group GEMM + `moe_fp8_m_split` `:44-60`/`:961-984`、`scale_inv`
      `repeat_interleave(128)` 展开逐行 `:920-931`、共享专家折叠 + TP row-parallel residual
      `:1064-1159`）、MTP（`:1649-...`/`:1183-1261`，草稿步编译期静态展开、权重 modulo 复用）；
      §2 **GLM-4.6**（fused-QKV+bias 作 residual `:193-218`、部分 RoPE `rotary_dim=64`+qk-norm
      `:220-237`、`glm_moe_router_layer` `:305-311`、**共享专家折叠成 expert 160 slot 8 权重 1.0**
      `:312-357`）；§3 **Inkling**(无 RoPE 相对位置+深度短卷积 `:594-614`、`inkling_moe_router_layer`
      `:526-531`、**共享专家折叠成 256/257**、外层残差后置)；§4 **GPT-OSS**（`plan_kv_cache` 滑窗+全
      attention 双 KV 流共享页池 `:17-47`、`gate_up` 交织布局重排 `:158-171`、sinks/window/group_id
      `:302-312`、o_proj 残差/bias 拆位 `:314-330`、`moe_clamped_swiglu_layer` `:354-357`）；
      §5 **投机 draft builder** DFlash/Eagle3（不注册、追加到目标图，DFlash 单次非因果前向 + 跨步
      paged KV、Eagle3 K 步自回归）；§6 **家族差异→统一 task 管线的折叠表**（左=家族原样 / 右=折平后
      标准算子 / 家族专用 `TaskType`），并给出 `docs/24` 家族 ID 块（350–399 Inkling / 400–449 GLM /
      450–499 GPT-OSS，`runtime_header.h:207-225`）的**具体前端生产者函数**；§7 论文 §4.1/§4.2/§5.8
      对照；§8 判据映射（**家族 builder 层只贡献 C3 范围**，家族专用 `TaskType` 的调度机制不分家族）；
      §9 局限；§10 坐标。配套博客 post `mega-kernel-27-mpk-family-builders`（weight 28）并更新
      `mega-kernel-00-index` 第 27 项。

### P113 权重预处理层源码走读（`docs/27-mpk-weight-prep.md`）

- [x] **P113-1** 补齐 `docs/26`（P112-1）**§8 明留的尾巴**——「`demo.py` 的权重预处理（Phase-2 吸收、
      gate_up 交织拆分）未走读：DeepSeek-V3 多处注释指向 `demo.py` 的 load-time 处理」。见
      `docs/27-mpk-weight-prep.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与
      `docs/06`–`docs/26` 同快照；**纯源码走读（资料分析），本轮无新实验**）。headline——**这一层在
      Python 里用普通 PyTorch 张量变换把「HF checkpoint 物理布局」改写成「builder 逻辑契约布局」**，
      且存在**两条独立实现**：离线 `convert.py main()`（写 per-rank safetensors、反量化 BF16）vs
      demo 内联 `Phase 1/2`（仅 `--layers` 选择性加载时跑、保留 FP8）。结构：§0 定位与两路径表；§1 输入
      契约（`is_fp8:298`、`find_scale_for_weight:272-291`、`dequantize_fp8:75-123` 三档 scale）；
      §2 MLA 吸收 `absorb_kv_into_q`（`convert.py:150-233`：per-head `k_nopeᵀ@q_nope`、concat rope、
      `[H*576,q_lora]`）+ V 反吸收融进 `o_proj`（`einsum('dhn,hnk->dhk')` `demo.py:519`，`H*128→H*512`）
      + 拆 `q_b_nope`/`q_b_pe` 双形态（chunked-prefill vs decode）；**并指出 Phase 2 的吸收块
      `:553-632` 因 Phase 1 已 `del kv_key`（`:525`）而不会执行**（同型「已实现未跑」）；§3 FP8 重定标
      `_quantize_f32_to_checkpoint_fp8:452-466`（128×128 块、`scale_inv=amax/448`）与 `dequantize_fp8`
      互逆；§4 gate/up **按 chunk 交织**而非 `cat`（`shuffle_tensors` `utils.py:50-128`，`:118-126` 逐
      chunk 交错），对照 Qwen3 demo 的 `mpk.shuffle_tensors`（`:563`/`:709`，`persistent_kernel.py:811`）；
      §5 MoE 专家折叠 `w13=cat(gate,up)`/`w2=down`（`:663-691`）、共享专家留给 builder 折叠；§6 对齐
      （16B，`:693-708`）+ TP 切分（内联 `_TP_SHARD_RULES:719-760`、w13 拆半再切 `:774-787`；对照离线
      `_SHARD_RULES:439-499`）——**如实记录两条 shard 规则不一致**（embedding/lm_head/q_a_proj 离线切、
      内联复制；专家离线复制、内联切），不判对；§7 输出键 ↔ builder 消费契约表（对照 GLM demo 无内联
      预处理=空层）；§8 判据映射（**本层不贡献任何 C 判据，只是 C3 的输入准备**）；§9 局限；§10 坐标。
      配套博客 post `mega-kernel-28-mpk-weight-prep`（weight 29）并更新 `mega-kernel-00-index` 第 28 项。

### P114 MPK 运行层「可执行规范」源码走读（`docs/28-mpk-testmode-harness.md`）

- [x] **P114-1** 补齐 `docs/25`（P111-1）**§9 明留的尾巴**——「`MirageModelConfig` 的调用方
      （如 `demo/`/`tests/`）未走读：本文只看库侧契约」。把「谁来验库里契约」这一层打开：见
      `docs/28-mpk-testmode-harness.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
      与 `docs/06`–`docs/27` 同快照；**纯源码走读（资料分析），本轮无新实验**）。headline——**`test_mode`
      把一张常驻 megakernel 图「跑一次就退」**：`-DMPK_TEST_MODE` 同时改**编译期**（缺 meta tensor 允许传空
      指针 `persistent_kernel.py:3413`）与**设备端终止协议**（`persistent_kernel.cuh:328` 把生产里
      「跑到 `max_seq_length`/EOS 才 done」的请求完成条件替换成 `if(true)`），于是每个测试都是一次
      「单次 launch → 常驻 kernel 自终止 → 比 PyTorch 参考」的最小端到端。结构：§0 定位（测试套件 =
      编译栈的**可执行规范**；一张图被驱动的 9 步共同骨架）；§1 `PersistentKernel` 公开 API 面
      （`get_default_init_parameters:669`、`attach_input:746`、82 个 `*_layer()`（`:842-3128`）、
      `compile:3207`、`__call__:3574`、`wait:3610`、`finalize:3614`、`target_cc:536`）与 worker/scheduler
      划分（`utils.py:25-53`：目标卡 132 SM → **128 worker/16 scheduler**，对上 `docs/06`/`docs/07`）；
      §2 `test_mode` 机理（`-DMPK_TEST_MODE:319`、`_apply_test_mode_meta_defaults:622`、设备端 `if(true)`
      终止协议）并解释它与 `docs/09` §7–§11 用**同一批用例**诊断不收敛的关系；§3 **依赖分析（`docs/08`）
      的可执行规范**——`fork_point`/`join_point`/`diamond_fork_join`（case 4）/`residual_stripping`
      （假并行路径剥离）/`case2_case3_negative`（子进程查 stderr，接受两者耦合）五个用例把
      `annotated_graph.cc` 的 fork/join 分类、case 2/3 拒绝、残余边剥离从源码规则变成可复现断言；
      §4 `grid_dim_sweep`——event 数随 grid 标度（每 batch item 一个 fork event）；§5 `test_mode/`
      其余 18 个用例的前端/device 契约目录（含 SM100 linear bias、GPT-OSS clamped SwiGLU、FP8 MoE
      流水、滑窗/attention sink、sampling 边界、argmax vocab pad、`kv_planner`）；§6 `demo/` 入口对照
      （`demo_rms_norm.py` 走 **superoptimize 自动路径**、测试套件**无一处** `superoptimize`——从调用面
      再证 `docs/15` §7「生产前端不调 superoptimizer」）；§7 C1–C5 映射（`test_mode/` 是本专题唯一能在
      **运行层**同时踩到 C1–C5 的源码面）；§8 局限；§9 坐标。配套博客 post
      `mega-kernel-29-mpk-testmode`（weight 30）并更新 `mega-kernel-00-index` 第 29 项。

### P115 MPK「单 kernel 单测」层源码走读（`docs/29-mpk-unit-tests.md`）

- [x] **P115-1** 补齐 `docs/28`（P114-1）**§8 明留的尾巴**——「**未覆盖 `tests/runtime_python/` 的非
      `test_mode` 部分**（`test_rmsnorm.py` 等走 `runtime_kernel` pybind 直接调 device kernel，不经
      `PersistentKernel`——那是『单 kernel 单测』，与 megakernel 端到端是两层，其 wrapper
      `runtime_kernel_wrapper.cu` 未走读）」。见 `docs/29-mpk-unit-tests.md`（快照
      `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/28` 同快照；
      **纯源码走读（资料分析），本轮无新实验**）。headline——**这一层把「一个 device task」当成
      「一整颗 kernel」由 host 手写 `<<<>>>` 直接 launch**，不经 `PersistentKernel`/event/tGraph/
      worker-scheduler，只和一个逐张量 PyTorch/numpy 参考比数值；论文从未提及（实现质量层）。
      §1 五套彼此独立的 pybind extension（主 `runtime_kernel` 1895 行三档 gencode + `-DMIRAGE_UNIT_TEST`；
      hopper 1405 行；hopper-moe 492 行；blackwell 家族 12 个 `setup*.py`；cute/hopper `gemm.cu` 556 行），
      并指出 `MIRAGE_UNIT_TEST` 全仓仅两处引用（`setup.py:44` 定义 / `tasks/ampere/moe_linear.cuh:256`
      消费：用 `blockIdx.y` 手动推进 expert 指针，注释 *"don't need in real MPK"*）；
      §2 **headline：pybind 接线大面积失配**——主 wrapper 只注册 3 个（`moe_linear`/`multitoken_paged_
      attention`/`sampling_from_logits`，`:1848/1887/1890`）而 19 个主目录测试调用 14 个符号（16 个
      导入/调用即失败），hopper wrapper 只注册 `linear_swapAB` 一个（`:1400`），而 hopper-moe 与
      blackwell 家族（当前代际）逐条对齐——「通用历史层失配、当前代际对齐」，是同专题「同一套惯用法
      实现了两遍」母题的测试层翻版；§3 单测形态解剖（`moe_linear`：`__global__` 只转发一次、把
      `blockIdx.x` 当 worker id、`launch_*` 手算 grid/block/96KiB SMEM、host 按形状硬编码派发模板、
      末尾 `cudaDeviceSynchronize`）；§4 与 megakernel 端到端（`docs/28`）逐维对照（task=整核 vs
      task=CTA 内一段；手写尺寸 vs 编译器算）；§5 CI 现状（`gpu-tests.yml` 已失效、现役 `mpk-qwen-ci`
      不跑本层——与 §2 失配互为因果）；§6 判据映射（**本层不贡献任何 C 判据**，只是 C3 范围内「单
      device task 数值正确」的回归网）；§7 局限（纯静态对照、未构建/未跑、不判对错）；§8 坐标。
      配套博客 post `mega-kernel-30-mpk-unit-tests`（weight 31）并更新 `mega-kernel-00-index` 第 30 项。

### P116 单测 wrapper 家族走读（`docs/30-mpk-unit-wrappers.md`）

- [x] **P116-1** 补齐 `docs/29`（P115-1）**§7 明留的尾巴**——「§3 只精读 `moe_linear`，**未逐行走读
      12 个 blackwell wrapper、`cute/hopper/gemm.cu` 与 `attention_tail_wrapper.cu` 的内部**」。
      见 `docs/30-mpk-unit-wrappers.md`（快照 `f9eb70c`，与 `docs/06`–`docs/29` 同快照；**纯源码走读
      （资料分析），本轮无新实验**）。headline——**修正 `docs/29` §2「当前代际对齐」**：blackwell
      家族有三类更隐蔽的失配——(a) **同名扩展撞名**：`sm100_linear/setup.py:22/25` 与
      `sm100_moe/setup.py:25/28` 都 `name='runtime_kernel_blackwell'`，但符号互异
      （linear 侧 `linear_sm100_mpk`/`linear_splitk_sm100` `:660-665` vs moe 侧 `topk_softmax_sm100`/
      `moe_w13_linear_sm100`/`moe_w2_linear_sm100`/`mul_sum_add_sm100`/`silu_mul` `:697-711`），
      两组测试都裸 `import runtime_kernel_blackwell` → 同一环境只能存在一个；
      (b) **源文件缺失**：`sm100_mla_prefill_tp8/setup.py:17` 指向不存在的
      `runtime_kernel_wrapper_mla_prefill_tp8.cu`（构建必失败）；(c) **无 setup/pybind 的 standalone
      程序**：`sm100_mla_mtp_decode/test_tp{2,4,8}_wrapper.cu`（含 `int main:145`、thin `__global__
      shim:24-67`）。§1 全景：**14 目录 / 18 `.cu`（含 3 standalone）/ 13 `setup*.py` / 6876 行**、
      统一带 `-DMIRAGE_GRACE_BLACKWELL`（多数带 `-DMPK_ENABLE_TMA`）；§2 逐目录三方对照表
      （撞名/缺源/standalone 三类 + 9 个逐条对齐扩展），统计 `blackwell/` 26 个 `test_*.py`（8 走
      `PersistentKernel`、18 裸 import 扩展）；§3「一个 task = 一颗 kernel」在 SM100 的两形态——
      形态 A `grid_dim(1,1,1)`+`block 256`+`cluster_dim(1,1,1)`+`smemBytes 224*1024`+
      `cutlass::launch_kernel_on_cluster`（`sm100_linear` `:54-56/147/263-266/282`、`sm100_moe`
      `:295-297/326-449`）、形态 B `hopper/attention_tail_wrapper.cu` 的 `extern "C" launch`
      （`:74-91`，SMEM poison `:20-24`、`<<<1,256,205824>>>` `:60-73`，nvcc→`.so`
      由 `test_attention_tail_hopper.py:54-56` 驱动）；§4 MoE wrapper 内部
      （`moe_linear_sm100_task_impl` `moe_linear_sm100.cuh:50-55`、`SM100_MMA_F16BF16_SS` TiledMMA
      `:66-73`、`mma_coord_vmnk = make_coord(0,_,_,_)` `:63`，即 tcgen05/UMMA/TMEM）；§5
      `cute/hopper/gemm.cu`（CUTLASS standalone、只注册 `linear_mpk` `:554-556`、`linear` 被注释、
      `grid(1)/block(256)/smem 100000` `:401-404`、dispatch 仅 `OUTPUT_SIZE=64`/`BATCH∈{8,16}`/
      `REDUCTION=4096` `:485-530`）——「同一套惯用法实现了两遍」的第三个肉身；§6 `hopper-moe` 复核
      （`:486-492` 与 `docs/29` §2 一致）；§7 判据**不贡献任何 C 判据**（同 `docs/29` §6）。
      配套博客 post `mega-kernel-31-mpk-unit-wrappers`（weight 32）并更新 `mega-kernel-00-index` 第 31 项。

### P117 FP8 单测 wrapper 与 TP standalone 走读（`docs/31-mpk-fp8-wrappers.md`）

- [x] **P117-1** 补掉 `docs/30`（P116-1）**§9「局限」明留的两条尾巴**——「**未逐行走读
      `sm100_linear_fp8`（1107 行）/`sm100_fp8_moe`（810 行）等大 wrapper 的 fp8 scale 重排与 UMMA 内部**，
      只核了注册符号与构建宏」与「`test_tp*_wrapper.cu` 的内部（`shim` 之后的 TP 切分）**未展开**」。
      见 `docs/31-mpk-fp8-wrappers.md`（快照 `f9eb70c`，与 `docs/06`–`docs/30` 同快照；**纯源码走读
      （资料分析），本轮无新实验**——SM100 在目标卡 H100/sm90 不可执行）。headline：
      **(1) fp8 家族 host 侧有一层 `docs/30` 漏掉的共享层** `tests/runtime_python/blackwell/common/`——
      **UE8M0 packed scale 的布局契约**：同时接受 **row-major**（stride `[packed_k,1]`）与 **DeepGEMM
      column-major**（stride `[1,ceil16(outer)]`）两种布局，`detect_scale_layout`
      （`sm100_fp8_runtime_registry.h:95-110`）在 C++ 与 Python（`sm100_fp8_scale_layout.py:52-62`）**逐行
      同构**；`materialize_internal_scale_tensor`（wrapper `:174-200`）把非 DeepGEMM 布局**物化**成列主序；
      **(2) 同一「128K 块 + UE8M0」语义在仓库有两种载体**——`sm100_linear_fp8` 把 UE8M0 **打包在 host**
      （uint32），`sm100_fp8_moe` 传 **fp32 scale** 由 device **现场转 UE8M0**（`fp8_group_gemm_sm100.cuh:76-87`
      + `fp8_tcgen05_after_thread_sync:153`，走 UTCCP→TMEM + `SM100_MMA_MXF8F6F4_SS`，7-warp 特化
      `:89-108`）；**(3) fp8 MoE 单测 wrapper 的 2D grid 手推 `blockIdx.y` 指针**，注释自陈在
      **「emulating the MPK runtime's per-CTA pointer offset logic from `runtime.cc`」**（`:13-14/112-113/
      128-143`）——与 `docs/29` 的 `MIRAGE_UNIT_TEST` 同源；**(4) headline：`sm100_linear_fp8` 引用了
      `f9eb70c` 快照里不存在的头文件** `blackwell/linear_fp8_sm100_splitk.cuh`（`#include` 在文件顶部
      `:19`，全仓无此文件；split-K 符号 `kernel::linear_fp8_sm100_splitk_wrapper` `:812` 亦无定义）⇒
      该扩展在快照里**无法编译**（连非 split-K 路径也编译不过）——是 `docs/30` 报的 `sm100_mla_prefill_tp8`
      「缺源文件」的**同族、更靠前**一处；**(5) TP standalone**（`sm100_mla_mtp_decode/test_tp{2,4,8}_wrapper.cu`）
      是**对外部 standalone 内核（`v037_tp*`）的机械 port 验证器**（`__global__→__device__ __noinline__`、
      **删 PDL** `cudaTriggerProgrammaticLaunchCompletion`、加 thread guard，头部注释 `:1-13`），TP =
      **head 维切分**（`NUM_HEADS = 128/64/32/16`），且 tp2/tp4/tp8 **各自重定义一份 `ptx` 命名空间**
      （`elect_sync`/`mbar_*`/`tcgen05_mma`/`tcgen05_commit`）——「同一套惯用法实现 N 遍」的又一肉身。
      判据**不贡献任何 C 判据**（同 `docs/29` §6/`docs/30` §7）；唯一与 C1–C5 有关的事实是 TP port
      **删掉了 PDL**（MPK 运行时不支持，对上 `docs/03` §2.5）。配套博客 post
      `mega-kernel-32-mpk-fp8-wrappers`（weight 33）并更新 `mega-kernel-00-index` 第 32 项；
      并把「快照里 `sm100_linear_fp8` 缺头文件」补进「已知坑」。

### P118 SM100 手工线性 task device 内部走读（`docs/32-mpk-sm100-linear-tasks.md`）

- [x] **P118-1** 补掉 `docs/31`（P117-1）**§10「局限」明留的两条尾巴**——「`linear_fp8_sm100` 的家族里
      `_scheduler`/`_epilogue_utils`/`_cute_tie` 未逐行走」与「`docs/24` 的 `TaskType` 目录里这些 SM100
      task 的**登记/接线未复核**」。见 `docs/32-mpk-sm100-linear-tasks.md`（快照 `f9eb70c`，与
      `docs/06`–`docs/31` 同快照；**纯源码走读（资料分析），本轮无新实验**——SM100 `sm_100f` 在目标卡
      H100/sm90 不可执行）。headline：
      **(A) device 内部**：`linear_fp8_sm100` 家族实为 **10 个文件**（`docs/31` §4 写「11 个头」是笔误），
      且是**两套互不相干的 device 身体**——身体 A `linear_fp8_sm100.cuh`（828 行）是**手写 SMEM 描述符 +
      裸 PTX `tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale`** 的四角色 warp 特化
      （warp0 TMA 生产者 `:290-402` / warp1 UMMA leader `:403-534`（`mma_t :486-487`、UTCCP scale
      copy `:463-483`）/ warp2 UTCCP scale 转置 `:535-575` / epilogue warps `:576-734` 走 TMEM→寄存器
      （`SM100_TMEM_LOAD_32dp32b{4,8}x`）→残差加→BF16 pack→`SM90_TMA_STORE`），`Allocator1Sm`/`2Sm`
      按 `kNumMulticast` 选（`:61-63`）但**只实例化了单 CTA 的 `SM100_MMA_MXF8F6F4_SS`**；身体 B
      `linear_fp8_1d2d_sm100.cuh`（727 行）是 CuTe 高层实现（`make_tiled_mma`/`local_tile`/
      `tile_to_mma_shape :117-188`）；**公开 pybind `linear_fp8_1d2d_sm100` 实际走身体 A**，身体 B
      **只有 `task_header.cuh:37` 的 `#include`、全仓无调用者**（同 `docs/16`/`20`/`22` 的「已实现未接线」）。
      **(B) 生产布线（补「未与生产路径对跑」）**：op `linear_fp8_sm100`/`linear_fp8_with_residual_sm100`
      （`graph.cc:852-858`）→ `TASK_LINEAR_FP8_SM100=276`/`…_WITH_RESIDUAL=277`（`runtime_header.h:163-164`）
      → `register_linear_fp8_sm100_task`（`task_register.cc:4772`）生成 `linear_fp8_sm100_task_impl<…>`
      （`:4826-4868`），实参为 `BLOCK 32/16/128`、**`kNumStages=25`（压 207 KB SMEM）**、
      **`kNumMulticast=1`（不用 2SM/cluster）**、**`GemmType::Normal` + `EpilogueIdentity`**；
      **顶部有明确标注的 HACK**（`:4817-4825`）：持久化内核里 `blockIdx.x` 是 **worker 索引**，
      故把 `kNumSMs` 设为 `MPK_MAX_NUM_WORKERS` 使**单 CTA 一趟处理完所有 tile**——与
      `_scheduler.cuh:170-172` 的注释同源，把「one task = one CTA」从 `docs/07` 的 Hopper 形态推进到
      SM100 block-scaled 形态；BF16 同族 `linear_sm100_mpk.cuh`（生产登记 `TASK_LINEAR_SM100=253`，
      `task_register.cc:2151/2304`）无 block scale，与 FP8 版/MoE 版三足对照。
      判据**不贡献任何 C 判据**（同 `docs/29` §6/`docs/30` §7/`docs/31` §9；C1/C2 由运行层拿）。
      配套博客 post `mega-kernel-33-mpk-sm100-linear`（weight 34）并更新 `mega-kernel-00-index` 第 33 项；
      并      把「公开 `linear_fp8_1d2d_sm100` 实际走 1d1d 身体 / 身体 B 未接线」与「SM100 手写 GEMM 的
      `kNumSMs=1` 适配」补进「已知坑」。

### P119 SM100 block-scaled FP8 MoE group-GEMM device 内部走读（`docs/33-mpk-fp8-group-gemm.md`）

- [x] **P119-1** 补掉 `docs/32`（P118-1）**§7「局限」明留的「未展开的更深层」**——「`fp8_group_gemm_sm100.cuh`
      的 **UMMA 主循环 / UTCCP 序列 / epilogue scatter** 仍未逐行」。见
      `docs/33-mpk-fp8-group-gemm.md`（快照 `f9eb70c`，与 `docs/06`–`docs/32` 同快照；**纯源码走读
      （资料分析），本轮无新实验**——SM100 `sm_100f` 在目标卡 H100/sm90 不可执行）。headline：
      **(A) 数据通路**：`fp8_group_gemm_sm100.cuh`（1567 行）是「1 个 CTA 跑 **3 条 warp 特化流水**」——
      8 warp 里 `warp5 DMA`（生产者）/`warp6 SCALE`（尺度生产者）/`warp4 MMA`（消费者+生产者）/
      `warps0-3 EPILOGUE`（消费者）由 **6 个 `mbarrier`**（`a_full` TransactionBarrier / `b_full` 32-lane /
      `ab_empty` / `acc_full` / `acc_empty` 4-arrival / `sf_ready`）+ 2 个 `NamedBarrier`
      （`TmemAllocBarrier` 160 / `EpilogueBarrier` 128）串联；swapAB 约定把 **UMMA-M=输出维、
      UMMA-N=token 维**（`:63-72`）。**warp5**：A 走 TMA（坐标 `(K_off, expert*ORIG_OUT+M_off)`，
      `set_barrier_transaction_bytes`）+ B 走**手写 SWIZZLE_128B `cp.async`**（`off ^ (((off>>7)&7)<<4)`）
      带 **per-token routing 检查**（未路由写零）；乐观 peek + lookahead（`try_wait_barrier`）贯穿三 warp。
      **warp4**：等 `a_full/b_full/sf_ready` → `SM100_UTCCP_4x32dp128bit_1cta` 把 scale 从 SMEM 搬进
      TMEM → 每 K-tile 发 **4 条 `SM100_MMA_MXF8F6F4_SS`**（`bK/UMMA_K=128/32`，`first_tile&&k_sub==0`
      覆盖否则累加）；描述符 lo-word **按 stage 预存到 lane 再用 `__shfl_sync` 广播**；`umma_arrive`
      保证 DMA 不覆写。**warp6**：fp32 scale → **UE8M0 `(__float_as_uint(sf)>>23)&0xFF`** → pack 4 复本
      → **`fp8_utccp_warp_transpose`（4×32→32×4，注释自称 identical to DeepGEMM）** → `sf_ready`；
      末尾有 **`FIX 2026-04-22` 竞态修复**（scale warp 先等 `ab_empty` 再覆写 sfa/sfb，TP=4/w2/mbt=64 下
      UTCCP 与覆盖赛跑）。**(B) epilogue**（`:1440-1552`）：`TMEM_LOAD_32dp32b1x` 载 FP32 累加器 →
      `NumericConverter` 转 BF16 → 按 routing **scatter `mOutput(n,topk-1,m)`**（仅被路由 token 写）；
      `acc_full/acc_empty` 双 barrier；**TMEM `free` 不 `release_allocation_lock`**（`:1559-1564`，注释
      引 `moe_linear_sm100.cuh:820` 的 "don't do relinquish for megakernel"）——**C2 常驻对 device task
      的直接让步**（与 `docs/32` `kNumSMs=1` 同族）。**(C) 生产布线（补 `docs/31` §10「TaskType 登记
      未全表复核」中本文件这一支）**：`register_moe_fp8_sm100_task`（`task_register.cc:3204-3452`）发射
      `TASK_MOE_W13_FP8_SM100=248`/`TASK_MOE_W2_FP8_SM100=249`（`runtime_header.h:136-137`），
      `task_config=(6,1,...)`（`graph.cc:713-720`），实参 **`MMA_M=128`/`MMA_N=16`/`bk=128`/
      `num_ab_stages=8`/`num_acc_stages=2`**，`EXPERT_STRIDE=grid_dim.x`，`expert_offset=bid.x`
      （`runtime.cc:352-359`）⇒ **x 维切专家、y 维切 N**。**(D) 一条「定义但未接线」证据**（延续本专题
      方法论）：文件内       `fp8_make_sf_smem_desc`/`fp8_advance_umma_desc_lo`/`fp8_tcgen05_after_thread_sync`
      **有定义无调用者**，实际用的是 `sm100_utils.cuh` 的共享版本。与 DeepGEMM 静态对应
      （`mma/sm100.cuh:42-48` `make_sf_desc`、`sm100_fp8_fp4_gemm_1d1d.cuh:461-480` 转置、
      `sm100_fp8_fp4_mega_moe.cuh:837/841-842/895` mega MoE）。判据**不贡献任何 C 判据**
      （同 `docs/29`–`docs/32`；C1/C2 由运行层拿）。配套博客 post `mega-kernel-34-mpk-fp8-group-gemm`
      （weight 35）并更新 `mega-kernel-00-index` 第 34 项；并把「scale warp 需等 `ab_empty` 防 UTCCP
      覆盖赛跑」与「`fp8_group_gemm_sm100.cuh` 三个本地 helper 未接线」补进「已知坑」。

### P120 SM100 第二套 device 身体：CuTe 线性 + MLA-MTP decode 走读（`docs/34-mpk-sm100-cute-tasks.md`）

- [x] **P120-1** 把 `docs/32`（P118-1）**§7「局限」明留的两条尾巴**（`docs/33` §11 已明确「本文未处理」）
  一起走完——「**身体 B**（`linear_fp8_1d2d_sm100.cuh`）的 CuTe 主循环只核了结构与惯用法、未逐行走完
  727 行」与「**`mla_mtp_tpN_main` 主体仍未走**」。见 `docs/34-mpk-sm100-cute-tasks.md`（快照
  `mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`，与 `docs/06`–`docs/33` 同快照；**纯源码走读
  （资料分析），本轮无新实验**——SM100 `sm_100f` 在目标卡 H100/sm90 不可执行）。headline：
  **(A) 身体 B 是「CuTe 高层」第二实现、生产未接线**：`linear_fp8_1d2d_sm100.cuh`（727 行）用
  `cute::make_tiled_mma`（`:143`）/`SM100_MMA_MXF8F6F4_SS<…,float_ue8m0_t,MMA_M,MMA_N,K,K>`（`:134-141`）/
  `local_tile`（`:158/161`）/`partition_shape_A/B`（`:164-174`）/`UMMA::tile_to_mma_shape`+`Layout_K_SW128_Atom`
  （`:182-188`）写四段流水（warp5 loader `:350-446`、warp4 UTCCP+4×UMMA `:450-609`、warps0-3 epilogue
  `:613-718`，`PipedSharedStorageWithSF` 见 `storage.cuh:69-100`）；`grep` 全仓**只有定义无调用者**、
  `task_header.cuh:37` 只 `#include`，公开 pybind `linear_fp8_1d2d_sm100`
  （`runtime_kernel_wrapper_sm100.cu:1101`）实走身体 A 的 `launch_linear_fp8_1d1d_sm100`（`:962`）。
  **(B) MLA-MTP decode 家族**（`mla_mtp_decode_sm100.cuh` 748 行 + `tp{2,4,8}`）：把一份完整 MLA attention
  （QK→softmax→PV→online rescale→归一化，`:77-671`）+ reduce（`:676-746`）压进一个 device task，
  `NUM_HEADS=128/64/32/16`；揭示**常驻 kernel 内跨 task 复用的三类 device 适配**——① **TMEM alloc/dealloc
  必须用 warp0**（与共驻 FP8 CUTLASS task 一致，否则后续 FP8 task 挂在其 `allocate()`，TP 版注释
  `:190-195`）；② **命名屏障 `bar.sync 12,128`**（只同步活跃 128 线程，避 `bugfix.md` Bug 14；ID 须落
  CUTLASS 用户区避 Bug 19）替代 `__syncthreads`；③ **256→128 线程守卫 + dual-dispatch gate（`Q_LEN>8`
  return）**。(C) 生产布线：8 个 `TaskType`（269/270/287–292），`graph.cc:791-840` dispatch、
  `task_register.cc:4405-4471`，前端 `persistent_kernel.py:1778/1812`、DeepSeek-V3
  `builder.py:744/748`；TP 变体是**对 ferret standalone 的机械 port**（文件头 `tp2:3-13`，删 PDL）。
  判据**不贡献任何 C 判据**（同 `docs/29`–`docs/33`；C1/C2/C4/C5 由运行层拿）。配套博客 post
  `mega-kernel-35-mpk-sm100-cute`（weight 36）并更新 `mega-kernel-00-index` 第 35 项；并把一条坑补进
  「已知坑」（常驻 kernel 内 TMEM alloc/dealloc 必须同一 warp）。**本专题此前任务清单全完成，本轮为其
  新增的增量。**

### P121 SM100 MLA attention 家族走读（`docs/35-mpk-sm100-mla-family.md`）

- [x] **P121-1** 补掉 `docs/34`（P120-1）**§6「局限」明留的尾巴**（原话）「未走读的相邻层：
  `mla_prefill_sm100.cuh`/`mla_prefill_tp8_sm100.cuh`、`mla_decode_sm100.cuh`、`mla_kv_cache_gather*`、
  `mla_sm100_2sm.cuh`、注意力之外的 SM100 task 家族仍未逐行」中**除 `mla_sm100_2sm.cuh` 与注意力之外
  家族以外**的 MLA attention 段——把 KV gather（两变体）/ decode / reduce / prefill（吸收版）/
  prefill_tp8（未吸收版）逐行走完。见 `docs/35-mpk-sm100-mla-family.md`（快照 `mirage-project/mirage`
  分支 `mpk`，commit `f9eb70c`，与 `docs/06`–`docs/34` 同快照；**纯源码走读（资料分析），本轮无新实验**
  ——SM100 `sm_100f` 在目标卡 H100/sm90 不可执行）。三条 headline：
  - **(A) KV gather 与 prefill 是真接线的 device 身体**：`mla_kv_cache_gather_sm100.cuh`（160 行，
    grid=`(max_num_batched_requests,1,1)`、block=128）做「append 新 token 进页式缓存 + gather 成连续
    `[max_seq_len,576]`」，其中 gather 用**扁平化 `(seq_pos×d8)` + 4 路 ILP**（注释自陈原两级循环在
    GB300 上慢 ~2.5×，`:119-157`）；`mla_kv_cache_gather_split_sm100.cuh`（138 行）输出分块
    `ckv_sep[·,512]`+`kpe_sep[·,64]`（chunked prefill 布局，`:113-135`）；`mla_prefill_sm100.cuh`
    （715 行，吸收版：`D_CKV=512`+`D_KPE=64`、cp.async + Ampere `m16n8k16`、256 线程、**单 stage 受
    常驻 SMEM 预算约束** `:44-51`、dual-dispatch gate `:220-250`、命名屏障保留区纪律 `:525-531`）。
  - **(B) 旧 decode/reduce 被 PR 651 MTP 版顶替、同族内成死代码**：`mla_decode_sm100.cuh`（460 行）
    的 `mla_decode_sm100_task_impl` 全仓**只被一个 standalone 单测 wrapper 调用**
    （`tests/.../runtime_kernel_wrapper_mla_decode.cu:24`），生产 `register_mla_decode_sm100_task`
    （`task_register.cc:4102`）发射的是 **`mla_mtp_decode_sm100_task_impl<false>`**（`:4168`，注释
    *"Use PR 651 MLA MTP decode kernel"*）；`mla_reduce_sm100.cuh`（77 行，per-thread-per-head 三趟）
    的 `mla_reduce_sm100_task_impl` **全仓零调用者**，生产 reduce 走 `mla_mtp_reduce_sm100_task_impl<256>`
    （`:4248`）。是本专题「已实现未接线」的第三种形态（同文件族内被后继实现覆盖）。
  - **(C) prefill 两条身体对应两种并行策略**：吸收版（`mla_prefill_sm100`）与未吸收 TP8 版
    （`mla_prefill_tp8_sm100.cuh`，486 行：`D_QK=192`、TMA 载 K/V `:290-302`、**128 线程守卫 +
    `bar.sync 3,128`** `:191-213`、SMEM 1024 对齐 `:223-230`）由 DeepSeek-V3 builder 的编译期开关
    `self._use_prefill = mbt >= 32`（`builder.py:245`）选路，并配合 prefill/decode 两条 dual-dispatch
    gate（定义域不重叠）。
  生产布线逐条核对：`graph.cc:767/772/777/784/861/866`、`task_register.cc:4102/4206/4275/4369/4878/4940`、
  `runtime_header.h:153-189`（TaskType 266/267/268/278/293/295）、`persistent_kernel.py:1569/1594/1627/
  1660/1690/1720`。判据**不贡献任何 C 判据**（同 `docs/29`–`docs/34`；C1/C2/C4/C5 由运行层拿）。
  配套博客 post `mega-kernel-36-mpk-sm100-mla`（weight 37）并更新 `mega-kernel-00-index` 第 36 项。
  P121-1 新留尾巴（§9）：**`mla_sm100_2sm.cuh`（3165 行，2SM 协作 attention）未逐行**（最直接的下一步）、
  注意力之外的 SM100 task 家族未覆盖、split gather 未做扁平化 ILP、未与生产端到端对跑。**本轮新增增量。**

### P122 SM100 2SM MLA attention（整颗 CUTLASS kernel）走读（`docs/36-mpk-sm100-mla-2sm.md`）

- [x] **P122-1** 补掉 `docs/35`（P121-1）**§9「局限」明留的头号尾巴**——`mla_sm100_2sm.cuh`
  （3165 行，2SM 协作 attention，`task_header.cuh:53` 有 include、standalone 单测有 caller）未逐行。
  见 `docs/36-mpk-sm100-mla-2sm.md`（快照 `f9eb70c`，与 `docs/06`–`docs/35` 同快照；**纯源码走读
  （资料分析），本轮无新实验**——SM100 `sm_100f`/`sm_100a` 在目标卡 H100/sm90 不可执行）。三条 headline：
  - **(A) 它是「整颗 kernel」而非「一个 task body」**：文件头 `:1-20` 自陈 *"Self-contained MLA Device
    Kernel for Blackwell (SM100) … flashinfer dependencies inlined"*；落地形态是 CUTLASS 惯用法——
    `device::MLA<Kernel>::run()` 用 `ClusterLauncher::launch`（`:3102-3109`）自持发 `(2,1,1)` cluster grid，
    带完整 `can_implement`/`initialize`/`update`/`get_workspace_size`（`:2968-3158`），**不经过 MPK 运行层
    `_execute_task`**。全仓 caller 只有 standalone 单测 wrapper（`runtime_kernel_wrapper_sm100_mla.cu:23`，
    pybind `mla_attention`）+ `task_header.cuh:53` 的聚合 include；**无 `TaskType`/`graph.cc` op/
    `register_*_task`/Python layer**——比 `docs/35` §3/§4 的旧 decode/reduce 更彻底（那两者进了注册表被顶替，
    它从未进注册表），同族于 `docs/30` 的 `cute/hopper/gemm.cu`（CUTLASS standalone）。
  - **(B) 2SM 是全文件第一性设定**：`kIs2Sm=true`（`:610`）→ `ClusterShape=Shape<_2,_1,_1>`（`:617-618`）、
    `TmemAllocator=Allocator2Sm`（`:631-632`）、`kWarpsInN=2`（`:635`）、`KernelTmaWarpSpecialized2SmSm100`
    （`:689-692`）、`AtomThrShapeMNK`（`:740-743`）。一对 CTA 一体：只有 **leader CTA**（`cta_coord_v==0`，
    `:1102-1104`）发 UMMA 与 TMEM 分配（`:1365`）、TMA `is_leader` 也要求它（`:1134-1135`）；TMEM 按
    `kWarpsInN` 对半切（`kSizeS=64`/`kSizeO=256`/`kSizeAccO=128`，`:816-829`）；softmax 的 `row_max`/`row_sum`
    靠 **SMEM 交换 + 命名屏障（128 线程）+ `peer_index=(threadIdx.x+64)%128`** 做 2×2 数据并行归约
    （`:2447-2457`/`:2847-2856`）；LSE 只由前半 CTA 写（`threadIdx.x<64`，`:2657-2659`/`:2713-2717`，注释
    自陈 *"the index is wrong"*）。这正是 `docs/04` §22（P33 `cta_group::2`）/`docs/21`（Blackwell 2SM
    transpiler）那套 SM100 2SM 原语在 MLA 上的**手工消费者**。
  - **(C) 七 warp 五 pipeline 把整颗 FlashInfer 风格查表注意力压进一条流水**：warp 角色 `kCompute(0-3)`/
    `kMma(4)`/`kLoad(5,6)`/`kLoadPageTable(7)`（`kWarpAssignment=0x4221'3333`，`:640-675`）、五 pipeline
    `PipelineLoadQK/S/P/O/PT`（`:749-761`，`PipelineLoadPV` 与 LoadQK 共用一段 SMEM staging，KC/VC 走 union，
    `:841-844`）；分页 KV 用 `CustomStride`/`Gather`（`:100-265`）在 CuTe 层表达、`load_cpasync` 的
    `copy_split`（`:1606-1622`）先算完所有 page-table 再发拷贝；`load_tma` 走 TMA 描述符。主循环
    `mma`:2185-2388 / `softmax`:2390-2524 / `rescale`:2526-2575 / `epilogue`:2577-2720 / `compute`:2722-2882。
    与 `docs/35` §2 的手写 KV gather（flatten+ILP）是同一问题的两种表达。
  判据**只贡献 C3**（独立 kernel、自持 launch/cluster_sync/TMEM 生命周期 → C1/C2/C4/C5 均不满足，同
  `docs/29`–`docs/35`）；是「一个 task = 一整颗 CUTLASS cluster kernel」的形态上限参照，反证 MPK 生产
  为何必须按常驻内核约束重写 task（C2 与 CUTLASS device 层「发一次跑完退」根本冲突）。配套博客 post
  `mega-kernel-37-mpk-sm100-mla-2sm`（weight 38）并更新 `mega-kernel-00-index` 第 37 项。P122-1 新留尾巴
  （§8）：CUTLASS `collective_builder` 侧 QK/PV atom 未展开、`B2B` 宏路径未细究、`docs/35` §9 其余
  尾巴（注意力之外的 SM100 task 家族、`mla_prefill_host.cu` 原型）仍未覆盖。**本轮新增增量。**

### P123 SM100 非 GEMM / 非 MLA task 家族走读（`docs/37-mpk-sm100-misc-tasks.md`）

- [x] **P123-1** 补掉 `docs/35`（P121-1）**§9「局限」明留的相邻尾巴**与 `docs/36`（P122-1）**§8「局限」
  明留的同一处**——「注意力之外的 SM100 task 家族（`attention_sm100`/`dflash_*`/`inkling_*`/
  `glm_moe_router_sm100`/`prob_scatter_sm100`/`softmax_gather_sm100`/`topk_*`/`sampling_sm100`/
  `rotary_embedding_sm100` 等）」。`docs/33`–`docs/36` 把 SM100 的**重算子主线**（FP8 GEMM / group-GEMM /
  MLA attention）逐段打开；本文把**同一 `tasks/blackwell/` 目录下「轻但必需」的薄 task**逐个走通，并核对
  它们在 `TaskType`/`graph.cc`/`persistent_kernel.py`/builder 四层的生产布线。见
  `docs/37-mpk-sm100-misc-tasks.md`（快照 `f9eb70c`，与 `docs/06`–`docs/36` 同快照；**纯源码走读
  （资料分析），本轮无新实验**——SM100 `sm_100f`/`sm_100a` 在目标卡 H100/sm90 不可执行）。三条 headline：
  - **(A) 薄 task 是 megakernel 的「调度与数值枢纽」，几乎全是「极简 device 身体 + 静态角色」**：与
    `docs/33`/`docs/34` 的 FP8（三/四角色 warp 特化 + UTCCP + TMEM）或 `docs/36` 的 2SM MLA（七 warp 五
    pipeline + cluster）相反，本批是**几十到几百行、朴素 `__syncthreads`/命名屏障、无 TMA/TMEM** 的实现；
    重 task 决定寄存器/代码足迹**上限**（`docs/04` §8/§10），这些薄 task 是**依赖图关节**（贡献 `TaskType`
    数与事件扇入/扇出，对上 `docs/08`/`docs/13`）。
  - **(B) 「一个 task = 一个 CTA」在薄 task 上的常数/几何适配**：`blockIdx.x` 在常驻内核里是 **worker
    索引**，故 device 身体一律**遍历「本 task 全部 row」而非按 `blockIdx` 取 row**——最直白证据是
    `per_token_group_quantize_fp8.cuh:87-92` 注释（*"blockIdx.x is the PHYSICAL worker id … Fix: each
    task processes ALL rows"*），同族于 `docs/32` §5.3 的 `kNumSMs=1` HACK、`docs/34` §3.5 的 TMEM-warp。
  - **(C) 同一「路由/采样」语义在仓库里有 3–4 份独立实现、由模型家族而非硬件代际决定**：路由有
    group-aware 向量化 `topk_sigmoid_sm100`（要求专家数 2 的幂）、160 专家标量 `glm_moe_router`
    （注释自陈不能复用 sigmoid 版）、logsigmoid-softmax `inkling_moe_router`（256 routed+2 shared）三种，
    却映射到**同一组 `mul_sum_add`/`silu_mul` 下游**；采样有 common 库 `sampling_from_logits_kernel`
    与 SM100 分块 `sampling_partial`/`sampling_reduce` 两套，且 `register_sampling_sm100_task` 与
    `register_sampling_reduce_sm100_task` **共用同一 `TASK_SAMPLING_SM100`**（`task_register.cc:2543/2645`）。
  §1 给四层布线表（`TaskType` 块 230–299 + 家族块 351–353/401 → `graph.cc` op → `persistent_kernel.py`
  layer → builder 调用）+ 关键事实：**本批每个 task 都有 builder 调用**（glm4_moe/inkling/dflash/
  deepseek_v3/qwen3/gpt_oss），与 `docs/35`/`docs/36` 的「有实现无生产者」相反；§2 路由三兄弟精读；
  §3 采样链（argmax partial+reduce 的 64-bit 打包、sampling 两段式与 Gumbel-Max、softmax_gather 不
  materialize 全分布、prob_scatter/extract 簿记）；§4 SMEM 原地 RoPE/RMSNorm（含 `norm_sm100` 融合 partial
  RoPE 与「半 CTA 提前 return 静默跳过同步」的正面样板 `:48/112-113`）、`dflash_norm_rope`（寄存器版）、
  `per_token_group_quantize`（UE8M0 packed）；§5 非 MLA attention（通用 `attention_sm100` 走 **m16n16k16
  mma**、Q=O SMEM 复用（issue #740）、m-tile 分块跨 warp 归约（#702）、partial RoPE/sliding/sink；
  DFlash 非因果 ctx+block 的 ref+mma 双实现；Inkling 相对位置 bias + log-scaling）；§6 轻量逐元素/簿记；
  §7 判据（**只贡献 C3**）+ 生产位置。判据同 `docs/29`–`docs/36`。配套博客 post
  `mega-kernel-38-mpk-sm100-misc`（weight 39）并更新 `mega-kernel-00-index` 第 38 项。P123-1 新留尾巴
  （§8）：`sampling_from_logits_kernel` 的 CUB 内部未展开、`mma_m16n16k16` 汇编未逐条、命名屏障 ID 6/7
  与 `docs/34` §3.5「须 ≥8」的口径张力**并存不判**、`mla_prefill_host.cu` 原型来源仍未走。**本轮新增增量。**

### P124 MPK 共享原语层走读（`docs/38-mpk-common-primitives.md`）

- [x] **P124-1** 补掉 `docs/37`（P123-1）**§8「局限」明留的两条尾巴**（原话）——「`sampling_from_logits_kernel`
  （`tasks/common/sampling.cuh` 的 CUB BlockReduce 内部）未展开、`mma_m16n16k16`（`tasks/ampere/mma.cuh`）
  汇编未逐条」与「`mla_prefill_host.cu` 原型来源仍未走」。`docs/06`–`docs/37` 走读了各代际「一个 task =
  一个 device 身体」，但**它们全站在同一层从被反复 `#include` 却未逐行打开的「共享原语库」**上。见
  `docs/38-mpk-common-primitives.md`（快照 `f9eb70c`，**纯源码走读、无新实验**——这些原语是
  arch-agnostic/Hopper 的，且本文只读不构建）。三条 headline：**(A)** `tasks/common/` 是全部 device task 的
  arch-agnostic 地基（7 头文件 919 行；伞状 `common_header.cuh:17-21` 被 **41 个** task 文件引用、`utils.cuh`
  被 **33 个**直引；`ampere/mma.cuh` 被两个**非 MLA 的 SM100 attention** 引用）——**「Blackwell」目录的通用
  attention 仍在用 Ampere 手写 `mma.sync` + `cp.async` + CUB**；**(B)** `common/sampling.cuh` 是 FlashInfer
  移植的「一趟式 Gumbel-Max」（`SamplingDataAndIndex::operator+` 定义成取较大者 `:76-95`，借 CUB
  `BlockReduce::Sum<VEC_SIZE>` 当 argmax 归约 `:212-215`；curand Philox 噪声 `:104-121`；只写 argmax index
  `:219-221`），生产登记 `TASK_SAMPLING_SM100`（`task_register.cc:2543/2550`；`runtime_header.h:152`），
  与 SM100 分块版共用同一 `TaskType`（`:2663`，`runtime_header.h:167-168`）；**(C)** `ampere/mma.cuh` 44 行
  里藏着 load-bearing 的 codegen 坑——累加器从只写 `"=f"` + 独立 `"f"` 输入改成**单一读写 `"+f"(C[i])`**
  （`:22-42`）修掉「K≥2/`MMA_ITERS_M`≥2 寄存器复用跨 fragment 污染」，`attention_sm100.cuh:506-516` 记其
  下游后果（曾 `if(m>0) continue` 跳 tile → 损坏 EAGLE3 投机 token 流）。§2.4 记一处**口径落差**：
  `utils.cuh:97-116` 的 `clear_smem_buffer` 以 `NUM_THREADS=128`（`worker_config.h:18`）跨步，而上层
  Blackwell worker 是 256 线程（**只记录不判对错**）；§4 指出 `common/copy_sm80.cuh`（104 行）**正是本专题
  `docs/04` §39–§56 实测的「取 task / 数据面载体」device 本体**（`cp.async.ca...L2::128B` + `ldmatrix.x4`）；
  §7 把 `mla_prefill_host.cu` 尾巴**收口为「不在快照、外部来源」**（同族：`mla_prefill_tp8_sm100.cuh:17`
  的 `src/kernel/mla_prefill_tp8.cu`、`mla_mtp_decode_tp8_sm100.cuh:3` 的 `ferret/*` 均不在仓库）。
  判据**只贡献 C3**（与 `docs/29`–`docs/37` 同档）。配套博客 post `mega-kernel-39-mpk-common-primitives`
  （weight 40）并更新 `mega-kernel-00-index` 第 39 项。P124-1 新留尾巴见 §9（CUB 内部未展开、`bfloat16.h`
  278 行只读接口、`clear_smem_buffer` 128/256 线程口径并存不判、原型源文件缺失不可核、未与生产端到端对跑）。
  **本轮新增增量。**

### P125 MPK 投机解码 device 层走读（`docs/39-mpk-speculative-decoding.md`）

- [x] **P125-1** 补掉 `docs/14`/`docs/24`/`docs/26` 只在**前端/枚举/家族**层列过名字、`docs/34`–`docs/37`
  只走 **MLA-MTP attention** 身体的那一整层 device 目录——`include/mirage/persistent_kernel/tasks/
  speculative_decoding/`（5 头文件 1014 行：`target_verify.cuh`/`prompt_lookup.cuh`/`target_verify_mtp.cuh`/
  `mtp_token_ops.cuh`/`eagle3_ops.cuh`）。见 `docs/39-mpk-speculative-decoding.md`（快照 `f9eb70c`，
  **纯源码走读、无新实验**；MTP/Eagle3 是 SM100 路径、prompt-lookup 虽 arch-generic 但按 B1/B2 不构建）。
  三条 headline：**(A)** 这批 task 全是「把答案查出来 / 把 token 搬对位置」的**簿记型 device 身体、无一吃
  TC**（最长 `eagle3_ops.cuh` 317 行也只有 memcpy/dim-1 concat/d2t 查表/commit；`target_verify_*` 全在
  `t_id==0` 单线程串行判定），却是投机正确性枢纽（`accepted_count`/`new_token_nums`/`tokens[]` 落位→
  运行时 `prepare_next_batch` 步进），是 `docs/04` §8/§10「task 数/事件扇入扇出」的典型来源；**(B)** 同一
  「验证 draft」语义仓库里**三套并行实现、接线各异**——prompt-lookup 贪心 `target_verify_greedy_kernel`
  （`target_verify.cuh:21`，`TaskType 114`）、DeepSeek-V3 MTP 的 strict/probabilistic/commit
  （`target_verify_mtp.cuh:44/112/255`，`TaskType 271/283/272`）、以及 **`target_verify_synthetic_kernel`
  （`:191-242`）与 `find_ngram_global_kernel_sequential`（`prompt_lookup.cuh:142`）两枚「有定义、全仓零
  调用者、无登记无 `TaskType`」死代码**（同 `docs/16`/`docs/20` 的「已实现未接线」族）；**(C)** Eagle3
  Python 侧落差——`Eagle3Builder` 类注释（`eagle3/builder.py:34-56`）声明完整 verify 管线，但
  `build_draft_loop`（`:271-449`）**只跑 draft loop 并 `return all_draft_ids`**，`copy_layer` 与
  `eagle3_commit_layer` 在快照 Python 侧**无调用者**。§1 记一处**静态接线落差只记不判**：生产 include
  入口 `persistent_kernel.cuh:48-52` 按 arch 三选一，**只有 blackwell** `task_header.cuh:62-64` 引了 MTP/
  Eagle3 三件；`target_verify.cuh`/`prompt_lookup.cuh` **不被任何生产 `task_header` include**，只在单测
  wrapper（`runtime_kernel_wrapper.cu:13/27/1845` 的 `using`/`m.def` 均**被注释**）出现，而 register/
  frontend（`TaskType 112/113/114`）是活的；两文件用 `../common.h`（快照不存在，同族见
  `hopper/norm_linear_hopper.cuh:18` 与多个 `ampere/*.cuh`）。§5 坐实 `request_id` 而非 `blockIdx.x`
  取批号（常驻 kernel `blockIdx.x` 是 worker 索引，同 `docs/32` §5.3/`docs/34` §3.5/`docs/37` §2）；
  §6.3 转述 `eagle3_commit_kernel` 注释里一段真 bug 调查（K=2 接受率 18.6% vs K=1 45.6%，device-side
  trace 定位到 `paged_attention_sm100` `MAX_TOKENS=3` 的 slot-0 输出分歧，候选点 `attention_sm100.cuh:519`
  causal mask 与 `MMA_ITERS_M=2` 的 `-inf` NaN 传播——同 `docs/38` §6 `"+f"` 修复的 MMA_ITERS_M>1 family）。
  判据**只贡献 C1（并入单 launch）+ C3（段级多 task 管线）**，与 `docs/29`–`docs/38` device task 走读同档。
  配套博客 post `mega-kernel-40-mpk-speculative-decoding`（weight 41）并更新 `mega-kernel-00-index`
  第 40 项。P125-1 新留尾巴见 §9（include 落差只记不判、`softmax_gather`/`prob_scatter`/`prob_extract`
  三个概率簿记身体未逐行、`paged_attention_sm100` `MAX_TOKENS=3` slot-0 分歧未独立复现只转述注释、
  接受率数字为源码注释自报、`lookahead`/`eagle3` handler 字典缺失只记 `ValueError`）。
  **本轮新增增量。**

### P126 MPK 退役实现走读（`docs/40-mpk-deprecated-paged-attention.md`）

- [x] **P126-1** 补掉 `docs/39`（P125-1）**§9「局限」第二句**与 `docs/37`/`docs/38` 都点到、却
  从未逐行打开的那一处尾巴——「`tasks/deprecated/` 此前各篇均『不涉』，其唯一成员
  `paged_attention.cuh`（496 行）可直接走读」（`ROADMAP`「下一步候选」第二条）。见
  `docs/40-mpk-deprecated-paged-attention.md`（快照 `mirage-project/mirage` 分支 `mpk` commit
  `f9eb70c`，与 `docs/06`–`docs/39` 同快照；**纯源码走读（资料分析），本轮无新实验**）。
  三条 headline：
  - **(A) 它是仓库里唯一把「paged-attention 计算」与「新 token K/V 写回页缓存」融进同一个 task 的
    device 身体**（`:473-484` 把最后 tile 第 `(seq_len-1)%64` 行写回最后一页）；而今天的实现把这件事
    拆成两个 task（`multitoken_paged_attention*` 只读、注释 `:99-102` 明说「页已预分配、数据后拷」，
    KV 写入走 `mla_kv_cache_gather_sm100`/`dflash_kv_store_sm100`）——**融合边界不是越融越好，
    「把写回从读里拆出去」是让 task 可复用/可调度的先决条件**，对上 `docs/05`/`docs/23`「少而重 vs
    多而杂」。
  - **(B) 自包含单 batch × 单 KV head × 单 CTA 的 FA2 手写实现**：`m16n16k16` mma（1x4x1 做 QK^T、
    1x1x4 做 PV）、在线 softmax（`m/d/o` 三段 rescale，`d=1.f` 初始化 quirk 经首轮 `rescale=0`
    清零）、4 warp 分片 + warp0 全量 `m/d/o` buffer 合并（对照 `docs/37` §5.1 issue #702/#740 的
    m-tile 分块与 Q/O SMEM 复用）、融合 QK-norm/RoPE、NHD 页缓存 + 双缓冲 64-key tile——功能是后世
    全部特性的最早期快照。
  - **(C) 接线状态是「彻底的孤儿」**：全仓无任何文件 `#include` 它（唯一引用在单测 wrapper
    `runtime_kernel_wrapper.cu:363`，被 `#ifdef DEPRECATED_TESTS` 包住，而 `DEPRECATED_TESTS`
    全仓无人 `#define`）；`using kernel::paged_attention_task_impl;`（`:18`）与
    `m.def("paged_attention", …)`（`:1886`）**全被注释**；且 wrapper 只 include ampere/blackwell
    `task_header`（`:2-3`）从不 include 本文件——比 `docs/16`/`docs/20`/`docs/29` 的「已实现未接线」
    更彻底：**连同接线尝试一起被归档**。
  另记两条**静态观察只记不判**：`page_indices` 只写尾段（`:165-171`）、`S_O_BUFFER_SIZE` 被注释
  （`:136`）而 `s_o_buffer` 按 `threadIdx.x<<5` 使用（`:419`）；旧式裸 include 约定（`common.h` 等）
  在当前目录布局下已不可解析。判据**只贡献 C3 的范围**（一个 paged-attention 段 = 一个 task），
  且是**被淘汰的 C3 范围划分**（与 `docs/29`–`docs/39` 同档）。配套博客 post
  `mega-kernel-41-mpk-deprecated-paged-attention`（weight 42）并更新 `mega-kernel-00-index` 第 41 项。
  P126-1 新留尾巴（§5）：退役原因是从两代代码并置推断（无 deprecation notice/提交记录可核）、
  `dmem_layout`/`smem_layout`/ampere `rms_norm`/`rotary_embedding` 内部未展开、未构建。**本轮新增增量。**

### P127 概率簿记 task 走读（`docs/41-mpk-prob-bookkeeping.md`）

- [x] **P127-1** 补掉 `docs/39`（P125-1）**§9「未展开的更深层」**第一句与 `ROADMAP`「下一步候选」
  第一条——「`softmax_gather_sm100.cuh`/`prob_scatter_sm100.cuh`/`prob_extract_sm100.cuh` 三个概率
  簿记 device 身体的内部未逐行（只在 `docs/37` §3.3 给接线坐标）」。见
  `docs/41-mpk-prob-bookkeeping.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
  与 `docs/06`–`docs/40` 同快照；**纯源码走读（资料分析），本轮无新实验**；三者均登记 SM100 路径，
  目标卡 H100/sm90 不可执行、特性上下文本机不具备）。三条 headline：
  - **(A) 全仓最小的 device 身体，却是「活」的生产 task**：`softmax_gather_sm100.cuh`（115 行、
    1 函数）+ `prob_scatter_sm100.cuh`（78 行、2 函数）合计 193 行 3 个 `__device__`，比退役的
    `paged_attention.cuh`（496 行）还小一个量级，但**全在五层生产布线**（`TaskType`
    282/284/286 → `task_register.cc:2731/2808/2839` → `graph.cc:726/740/745`（均 `(2,1)`）→
    `persistent_kernel.py:2957/2992/3010` → `deepseek_v3/builder.py:1876/1993/2220/2229`）——
    与 `docs/36`/`docs/40` 的「有实现无调用者」死代码**恰好相反**。勘误：`prob_extract_sm100.cuh`
    这个文件名不存在，`prob_extract_task_impl` 只是 `prob_scatter_sm100.cuh:54-76` 的第二个函数。
  - **(B) 三处「注释 / 代码 / 布线」三方落差**（行号级摘要看不到）：`softmax_gather` 文件头
    注释（`:18-29`）写 `Grid:(BATCH_SIZE,1,1)` 且公式写 `exp(logit-max-log_sum)`，而**代码**把
    batch 循环塞进**单 block**（`:46`）、算的是 `exp(logit_at_target-global_max)/global_sum`
    （**除法非减 log-sum**，`:108`），**所有调用方**传 `grid=(1,1,1)`；`prob_scatter` 注释同写
    `Grid:(batch_size,1,1)` 而代码单线程串行 batch（`:42`）、调用方传 `grid=(1,1,1)`——根源是
    `docs/32` §5.3/`docs/34` §3.5/`docs/37` §2「常驻 kernel 里 `blockIdx.x==worker id` ⇒ 身体自己
    内部循环」，注释是从「独立 kernel 版」复制的过时物。另记一条**隐性约束**：`smem_max[8]`/
    `smem_sum[8]` 按 `warp_id` 索引 ⇒ `blockDim.x ≤ 256`，前端误传 >256 会**静默越界**（无 assert）。
  - **(C) 概率 MTP 旁路是「半实现」**：三个 device 身体 + 五层布线都活着，但 `builder.py`
    仍留两处 `TODO`/`pass`——draft 概率抽取是空的（`:1973-1974`），而注释求的「slice by runtime
    offset kernel」（`:1985`）**就是** `prob_extract_task_impl`（`:1993` 实际调用并存）。同时
    存在**两种「写进每 step 槽」的载体**：draft 生成用**编译期槽号** `mtp_float_scatter`
    （`TaskType 285`、`for step …` 展开、`slot_idx` 常量），main graph 用**运行期 step 计数器**
    `prob_scatter`（`TaskType 284`）——同一目的两套时间维载体，是 `docs/04` §47/§48「动态必须落在
    调度」在生产代码里的现实注脚。另补两条语义：`prob_scatter` 的 `pos` 越界**静默不写**、
    `softmax_gather` 的 `target_id` 越界**有定义取 `-1e30`**（对照 `prob_extract` 越界**补 0**）。
  判据**只贡献 C1（并入单 launch）+ C3（段级多 task 管线）**，与 `docs/29`–`docs/40` device task
  走读同档；并给 `docs/40` 的融合边界以对照样本（`softmax_gather` 的融合目的是**不 materialize
  整条概率分布**、`O(V)`→`O(1)`，同 `docs/04` §17 损失三方向）；对 `docs/39` headline (A) 的
  「簿记型」补一层：那是**数值角色**（不吃 TC、不产 token），**不等于成本轻**（`softmax_gather`
  是单 CTA 两趟全词表归约的重活）。配套博客 post `mega-kernel-42-mpk-prob-bookkeeping`（weight 43）
  并更新 `mega-kernel-00-index` 第 42 项。P127-1 新留尾巴（§6）：注释过时只记不判、`blockDim>256`
  越界与 `pos` 越界静默只作静态约束记录、`mtp_float_scatter`（`mtp_token_ops.cuh`）身体未逐行、
   `target_verify_*` 如何消费 `prob_extract` 输出只按注释转述、未构建。**本轮新增增量。**

### P128 MTP 验证链消费链走读（`docs/42-mpk-mtp-verify-chain.md`）

- [x] **P128-1** 补掉 `docs/41`（P127-1）**§6 明留的两条尾巴**——「`mtp_float_scatter` 的 device
  身体（`mtp_token_ops.cuh`）未逐行」与「`target_verify_*` 如何消费 `prob_extract` 的输出（拒绝采样
  比较 `P_target > u·P_draft`）只按注释转述」（`docs/39` §4/§5 只给 `target_verify_*` 与
  `mtp_token_ops` 的逐段摘要、未串起消费链）；
  也即 `ROADMAP`「下一步候选」第二条。见 `docs/42-mpk-mtp-verify-chain.md`（快照
  `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/41` 同快照；**纯源码走读
  （资料分析），本轮无新实验**；整条链只登记 SM100 路径、特性上下文本机不具备）。三条 headline：
  - **(A) 概率 MTP 拒绝采样是一条「两处生产、一次消费」的跨迭代链**（`docs/41` 只给每段坐标、
    `docs/39` 只给各身体摘要，均未串起数据流）：draft 侧 `softmax_gather(q:1876) →
    mtp_float_scatter(q:1885, slot_idx=step)` 写 `_draft_prob_buffer[:, step]`；main graph 侧
    `softmax_gather(q:2220) → prob_scatter(q:2229, step_counter=step)` 写 `_target_prob_buffer[:, step]`；
    verify 侧 `prob_extract(q:1993, offset=step) → target_probs` + `_draft_prob_buffer` 一起送
    `target_verify_probabilistic(q:2007) → accepted_count/output_tokens → mtp_accept_commit(q:2037)`。
    时序锚点：**写入端用 `step`、读出端用 device 侧硬编码 `offset+1`**（`prob_scatter_sm100.cuh:42-46`
    vs `:68`）；draft 侧无位移（verify 从 `i=0` 平读 `:145`）。
  - **(B) `mtp_token_ops.cuh` 四身体分属两套 batch 索引契约**（`docs/39` §5 未并置）：
    `mtp_{token,float}_scatter`（`:37-64`）用 `b=threadIdx.x` **批并行**（`blockDim≥BATCH`，写满
    `[BATCH,NUM_SLOTS]`）；`mtp_prepare_verify_input`（`:104`）/`mtp_build_embed_input`（`:175`）用
    `task_metadata.request_id` **按请求**索引。前者靠编译期 `SLOT_IDX`、后者靠运行期 `step`。
    `mtp_float_scatter` 与 `mtp_token_scatter` 逐字节同构（仅载体 `float` vs `long long`），登记
    `task_register.cc:2791-2806`/`:5039-5056`、`graph.cc` 均 `(1,1)`（`:736-739`/`:881-884`）。
  - **(C) verify/commit 家族「无 `request_id`、从 0 索引」的批元素契约**（`docs/39`/`docs/41`
    摘要未点出）：`target_verify_{greedy,strict,probabilistic}`、`mtp_accept_commit`、`eagle3_commit`
    签名都不带 `request_id`，且 `draft_ids[i]`/`target_ids[i]`/`output_tokens[i]`/`accepted_count[0]`/
    `current_position[0]` 一律从 0 索引 ⇒ **device 侧只处理批元素 0**（`target_verify_mtp.cuh:28-29`
    注释自陈「one batch element per thread block」），与调用方 `grid_dim=(mbt,1,1)` 的多维入参不符；
    `seed[0]` 为标量 ⇒ 「每请求独立种子」未兑现。**只作静态契约记录、不判 bug**。
  另补 builder 侧三处**新落差**（`docs/41` 未记，可由 `_cached_new_tensor` 按 name 缓存 `:1172-1181`
  核实）：局部 `draft_probs`（`:1967-1969`）**是死张量**（真正传的是 `_draft_prob_buffer` `:2011`）；
  `rng_seed` **被建两次**（`:1977`/`:2002`，同 name 缓存 ⇒ 第二次冗余、且无初始化）；
  `mtp_verify_probabilistic_layer`（`:2007`）**在 `prob_extract` 的 guard（`:1992`）之外无条件调用**
  且传 `target_probs`（`:2010`），`step_tensor` 缺失时该张量未写入。判据**只贡献 C1+C3**，与
  `docs/29`–`docs/41` device task 走读同档；跨度对应 `docs/08` §2 的跨迭代 view/barrier 边。
  配套博客 post `mega-kernel-43-mpk-mtp-verify-chain`（weight 44）并更新 `mega-kernel-00-index`
  第 43 项。P128-1 新留尾巴（§7）：`seed` 实际初始化位置未追、`comment/落差`只记不判、
   `_draft_prob_buffer` 是 probabilistic 专属、`mtp_build_embed_input` 多 block 覆写整向量只记不判、
   未构建、未与生产端到端对跑。**本轮新增增量。**

### P129 Eagle3 commit device 身体走读（`docs/43-mpk-eagle3-commit.md`）

- [x] **P129-1** 补掉 `docs/39`（P125-1）**§9「未展开的更深层」**明留的尾巴——「`eagle3_commit_kernel`
  的 device 身体（`eagle3_ops.cuh:157-315`，§6.3 已摘要、未逐行）与 `accept_hist` 调试历史」，
  也即 `ROADMAP`「下一步候选」第二条剩余。见 `docs/43-mpk-eagle3-commit.md`（快照
  `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/42` 同快照；**纯源码走读
  （资料分析），本轮无新实验**；四个 body 只被 blackwell `task_header.cuh` include，SM100 专属，
  本机不可执行）。三条 headline：
  - **(A)** `eagle3_commit_kernel` 是全仓唯一的「verify-aware 收尾 task」，一次 device 调用做完四件事
    （写 verified 前缀 `:197-202` + 写 K 个新 draft `:231-236` + `new_token_nums` 步进信号 `:311-314`
    + `drafts_prev` 跨迭代快照 `:244-252`）；`src_slot` 是 K=1 `ac-1` / K>1 **强制 0** 的已知坑
    （`:204-227`，K>1 走 `Q_LEN_OVERRIDE=1` 只写 slot 0）。**修正 `docs/39` 两处**：接口是
    **`(5,2)`**（`graph.cc:920`）而非「7 入 2 出」、`eagle3_commit_layer` **在 demo 有调用者**
    （`demo/qwen3/demo_30B_A3B_eagle3.py:880`）而非「无任何调用者」。
  - **(B)** 「verify 需要上一轮 draft」的跨迭代依赖被 **attach_input 快照**绕过（不建图边、靠批间
    屏障），运行时消费端是 `prepare_next_batch` 的 `#ifdef MPK_SPEC_DECODE` 分支
    （`persistent_kernel.cuh:295-316`：`step += new_token_nums[req]`，<1 兜底 1）；同一 commit 内
    `tokens`/`step` 按 `request_id`、而 `accepted_count[0]`/`drafts_prev[t_id]` 固定批元素 0
    （与 `docs/42` §C 同现象）。
  - **(C)** `accept_hist`（`:254-309`）是一段 device 侧自陈的 bug 调查史——原子直方图（`:288-290`）
    + 前 16 迭代 trace（布局 `[0..K+1]` bin / `[K+2]` 计数器 / 每记录 `2K+2` int，`:291-308`）；
    根因定位到 `paged_attention_sm100` 的 `MAX_TOKENS=3` slot-0 输出与 `MAX_TOKENS=2` 不一致
    （`:264-287`），候选修复点指向 `attention_sm100.cuh` 的 `MMA_ITERS_M`/causal mask；与 `docs/38`
    §6 的 `m16n16k16` `"+f"` in-place 累加器修复是**同一 family**（当前快照
    `attention_sm100.cuh:505-523` 注释明确「corrupted the committed EAGLE3 token stream at K>=2」）。
  §7 与 MTP commit（`docs/42`）逐维对照（职责数 2 vs 4、新增 `drafts_prev`、`accept_hist`）；
  判据**只贡献 C1+C3**，与 `docs/29`–`docs/42` device task 走读同档。配套博客 post
  `mega-kernel-44-mpk-eagle3-commit`（weight 45）并更新 `mega-kernel-00-index` 第 44 项。
  P129-1 新留尾巴（§9）：`paged_attention_sm100` 内部未逐行、验收率/根因系源码注释自报、
  causal-mask 行号漂移只记不判、demo 只读接线段、未构建。**本轮新增增量。**

### P130 通用 paged attention device 身体走读（`docs/44-mpk-paged-attention-sm100.md`）

- [x] **P130-1** 补掉三条明留的尾巴——`docs/37` §5.1（只给 750 行摘要）、`docs/39` §6.3
  （`paged_attention_sm100` `MAX_TOKENS=3` slot-0 分歧只按注释转述）、`docs/43` §9 头号
  （「`paged_attention_sm100` 内部未逐行，只引其 `MAX_TOKENS`/`MMA_ITERS_M`/`"+f"` 三处注释」）。
  把 MPK **唯一的通用 paged GQA attention** device 身体逐行打开，见
  `docs/44-mpk-paged-attention-sm100.md`（快照 `mirage-project/mirage` 分支 `mpk` commit
  `f9eb70c`，与 `docs/06`–`docs/43` 同快照；`attention_sm100.cuh` 恰 750 行、唯一 device 函数
  `multitoken_paged_attention_sm100_task_impl:62-748`；**纯源码走读（资料分析），本轮无新实验**；
  SM100 专属、本机不可执行）。三条 headline：
  - **(A)** 它是 MPK 里**唯一**用 Ampere 手写 `m16n16k16` `mma.sync`（非 wgmma/UMMA）的通用
    paged GQA attention：一个 task = 一个 (request, KV head)（`request_id` 入参 + 输入按 KV head
    切片），在 per-token 在线 softmax（`m`/`d`/`o` 全在寄存器）里流式扫完整条 KV cache；
    QK^T `1x4x1`（warp 切 key，`:503-531`）、PV `1x1x4`（warp 切 p，`:624-639`）；把
    **QK-norm / partial RoPE / sliding window / attention sink / 输出重排**全编进单 task
    （14 模板参 + 15 运行期入参，`:34-82`）。
  - **(B)** §2 把「为压进 Blackwell SMEM 上限」的**重构史**坐实：**O 复用 Q 段**（`:187-196`，
    issue #740，砍掉 `MAX_TOKENS*NUM_QO_PER_KV*HEAD_DIM` 元素才让 mbt 过 25）、**跨 warp 归约的
    `m`/`d`/`o` buffer 只存一个 MMA m-tile**（`:209-228`，issue #702，让 GQA ≥8:1 放得下）；
    `static_assert(S_TOTAL_OFFSET<=MAX_DYNAMIC_SHARED_MEMORY_SIZE)`（`:230-231`）。是 C2（常驻）
    对 device task 的直接约束（同 `docs/33` §5「TMEM free 不释放锁」）。
  - **(C)** 把 `MAX_TOKENS=3` slot-0 bug family 钉回源码级（§7）：**真正因果/窗口掩码在
    `:556-559` 的 `is_valid`**（`docs/39`/`docs/43` 转述的「causal mask 在 `:519`」是行号漂移，
    `:519` 是 k 循环头）；`:505-516` 注释自陈旧 workaround `if(m>0) continue` 丢弃 query 槽
    `≥16/NUM_QO_PER_KV`、败坏 EAGLE3 K≥2 token 流，根因 = `"=f"`+`"f"` 别名累加器在 `unroll`
    下跨 fragment 污染，修复 = `"+f"`（`ampere/mma.cuh`，`docs/38` §6）——本篇补上 attention
    侧消费者 `:529`；并校正「`-inf` NaN 传播」在当前快照**已被两处守卫覆盖**（`:591-595`、
    `:651-652`）。
  §8 给五层布线（`TaskType 257`/`task_register.cc:2326-2438`/`graph.cc:643-649`/
  `persistent_kernel.py:1303-1432`/builder 四家族）；命名的 `NamedBarrier(128, id=6/7)`
  （`:83-89`）与 `docs/34` §3.5「须 ≥8」的口径冲突**只记不判**（同 `docs/37` §8）。判据
  **只贡献 C1+C3**，与 `docs/29`–`docs/43` device task 走读同档。配套博客 post
  `mega-kernel-45-mpk-paged-attention-sm100`（weight 46）并更新 `mega-kernel-00-index` 第 45 项。
  P130-1 新留尾巴（§10）：`MAX_TOKENS=3` slot-0 分歧未独立复现、命名屏障 ID 6/7 只记不判、
  `rms_norm_sm100`/`rotary_embedding_hopper`/`smem_row` 内部未逐行、sink 数值影响未实测、
  grid 第二轴与 task 粒度映射系静态推断、未构建。**本轮新增增量。**

### P131 归一化 / 位置编码 task 家族跨代走读（`docs/45-mpk-norm-rope-tasks.md`）

- [x] **P131-1** 补掉 `docs/37` §4（P123-1 对 SM100 norm/rope 只给**摘要**）、`docs/07` §3.1
  （只走 `rms_norm_hopper_impl`）与 `docs/44` §10（P130-1 明留「`rms_norm_sm100`/
  `rotary_embedding_hopper` 内部未逐行」）三处空白：把「每个 Transformer 层出现两次的薄算子」
  norm/rope 当成**一个跨 Ampere/Hopper/Blackwell 三代演化的 task 家族**逐行打开。见
  `docs/45-mpk-norm-rope-tasks.md`（快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，
  与 `docs/06`–`docs/44` 同快照；**纯源码走读（资料分析），本轮无新实验**；SM100 本机不可执行）。
  三条 headline：
  - **(A) 三层形态与「谁没有独立 task」**：形态 A「独立 global-memory norm task」两条身体——
    `ampere/rmsnorm.cuh` 的 `rms_norm_impl`（`TASK_RMS_NORM`=119，硬断言 `BATCH_SIZE==1`
    `:24`、128/64/32-bit `cp.async` 双缓冲 `:29-100`、三段式蝶形归约 `:102-118`）与
    `hopper/rmsnorm_hopper.cuh` 的 `rms_norm_hopper_impl`（`TASK_RMS_NORM_HOPPER`=154，
    **注释掉单 batch 断言** `:24` + 支持多 batch `:75` + `EVEN_SPLIT`/短尾 tile `:29-48,110-115`）；
    Python 前端按 `target_cc>=90` 二选一（`persistent_kernel.py:881`），**目标卡 H100/sm90 走 Hopper
    那条**——即 `docs/09` §7–§8 真机实测的 test_mode RMSNorm（36.4–39.1 µs）本体。形态 B/C 是
    「调用方 SMEM 视图上的融合 norm / 纯 RoPE」，**从不落 HBM、被 attention 内联调用**。
  - **(B) 融合边界 + 死代码发现**：全仓**没有 `TASK_ROTARY`/`TASK_ROPE`**——RoPE 要么折进 attention
    的 QK-norm 段，要么折进 DFlash 独立 task；**同一种数学、两种融合边界**取决于下游消费 SMEM 还是
    全局张量（`dflash_norm_rope_sm100` 一线程一 `(token,head)` 是唯一独立的 norm+rope）。**发现并修正
    `docs/37` §4.2 的静态事实错误**：`rotary_embedding_sm100.cuh` 仅被 `attention_sm100.cuh:23`
    `#include`、**全仓（含 `tests/`）零调用者**，`attention_sm100` 在 `qk_norm=false,rope=true` 分支
    实际调用的是 **`rotary_embedding_hopper`**（`:435/:455`）——又一处「按文件/名字推断哪份在跑会错」
    （同 `docs/16`/`docs/20`/`docs/32`/`docs/33`/`docs/38` 死代码族）。
  - **(C) 同步机制与线程数口径的「百家争鸣」**：三代融合 norm/rope 分别用 cooperative groups
    （Ampere，`:37-41`）、`wg_sync`=`bar.sync`（Hopper，`utils.cuh:30-36`）、cutlass `NamedBarrier`
    （Blackwell，ID 6/7）三套同步；`NUM_THREADS` 有**四种来源并存**（全局 `worker_config.h:18`=128、
    hopper-rmsnorm 模板默认 256、hopper-rope 模板默认 128、hopper SMEM-rmsnorm 显式模板参），
    barrier ID 用 6/7/9（对上 `docs/34` §3.5「须 ≥8」张力，只记不判）；**partial RoPE 三代不对称**
    （Ampere 全家族不支持、Hopper/Blackwell 才有 `ROTARY_DIM`，解释 GLM-4.6 只能走 cc≥90）。
  §6 给五层布线总表（`TaskType 119/154/297` / `task_register.cc:91/1551/185` /
  `graph.cc:449-452/462-466/571-574` / 前端 / builder `dflash/builder.py:261/318/320`）。
  判据**只贡献 C3**，与 `docs/11`/`docs/29`–`docs/44` 同口径。配套博客 post
  `mega-kernel-46-mpk-norm-rope`（weight 47）并更新 `mega-kernel-00-index` 第 46 项。
  P131-1 新留尾巴（§8）：本家族其它身体未构建/未运行（复用 `docs/09` 数字）、死代码定性系静态推断、
  `smem_row`/`norm_linear_hopper` 未重复走读、barrier ID 张力不判、`WORKER_NUM_THREADS` 运行期值未取。
  **本轮新增增量。**

### P132 Eagle3 draft loop 走读：K=1 mbt 并行链 / K>1 slot-0 退化 / proper-chain 缺口（`docs/46-mpk-eagle3-draft-loop.md`）

- [x] **P132-1** 补掉 `docs/40`/`docs/43`/`docs/44` 一直没打开的「谁把 K 个 draft step 拼成一张图」
  ——即 `ROADMAP`「下一步候选」第一条 `eagle3_commit` 的 K>1「proper mbt-parallel chain」缺口
  （`eagle3_ops.cuh:217` 自陈未实现）的**静态追述**。见 `docs/46-mpk-eagle3-draft-loop.md`
  （快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/45` 同快照；
  **纯源码走读（资料分析），本轮无新实验**；Eagle3 device 头只被 blackwell `task_header.cuh` include，
  SM100 专属、本机不可执行）。三条 headline：
  - **(A) draft loop 是一张「`for step in range(K)` 展开的单图」**：`Eagle3Builder.build_draft_loop`
    （`models/eagle3/builder.py:271-449`）每 step 一整层 draft transformer（15 段算子链，`§2.2` 表），
    中间张量全 `(mbt,…)` 且跨 step 复用（`_allocate_intermediates:220-269`），靠 `draft_hidden`(`:238`)
    与 `target_token`(`:265`) 两条接力串行——**没有数据并行空间**。
  - **(B) K 只改 attention 一处**：K=1 走默认 `paged_attention_layer`（`num_tokens=mbt`，产生
    「mbt 条并行链」，每槽一个 `ac` 假设）；K>1 改 `q_len_override=1, tail_offset=K-step`
    （builder `:381-382`，前端明记 "Eagle3 K>1 chain" `persistent_kernel.py:1360-1361`），device 侧
    `num_tokens=1`（`attention_sm100.cuh:117-119`）、`seq_len` 回收 `TAIL_OFFSET`（`:128-130`）、
    store loop 只写 1 行（`:737-746`）。因 demo 硬断言 `mbt==K+1`（`:267-269`），K>1 时 mbt 行里
    **只有 1 行有效**；下游 `rmsnorm/linear/argmax(partial/reduce)/d2t/scatter` 全 mbt-wide
    （`argmax_sm100.cuh:102/142`、`eagle3_ops.cuh:115-121`、`mtp_token_ops.cuh:45-48`）→
    `all_draft_ids[1..mbt-1,:]` 全是 garbage → commit **被迫 `src_slot=0`**（`eagle3_ops.cuh:221`
    `(K>1)?0:(ac-1)`）。
  - **(C) 缺口 = 把 1 行有效推广回 mbt 行有效**：把 device 注释目标拆成 4 条必要条件（attention 要写
    mbt 个输出行 / 每槽 `tail_offset` 需 per-slot / 下游簿记本已 per-batch 无需改 / KV 写回不能互踩），
    并说清它与 `MAX_TOKENS=3` slot-0 bug family（`docs/38` §6 `"+f"`、`docs/44` §7、`docs/43` §4）
    **共用同一段 attention 代码**（`MMA_ITERS_M:103` 的 m-loop 既是补齐前提也是候选修复面）。另记一处
    静态落差：demo 注释描述的 flat-cache `eagle3_draft_chain_step_attn` 单 kernel（`:826-832`）与
    builder 实走的 `paged_attention_layer`（`:357/:370`）是**两套并存 design**（快照无前者符号，
    只记不判）。
  判据：**本文贡献 C3 的「范围与分支」**，与 `docs/43` §8（commit 只贡献 C1+C3）互补。配套博客 post
  `mega-kernel-47-mpk-eagle3-draft-loop`（weight 48）并更新 `mega-kernel-00-index` 第 47 项。
  P132-1 新留尾巴（文档 §8）：未构建；「有效行 = slot 0 / `first_token_pos` 行」是静态推断（多请求未核）；
  4 条必要条件系综合推断；`MAX_TOKENS=3` 分歧未独立复现；demo 注释落差只记不判。
  **本轮新增增量。**

### P133 Eagle3 权重/中间张量装配层走读：shuffle 布局 / attach 缓存 / pad→argmax→d2t 契约链（`docs/47-mpk-eagle3-weight-prep.md`）

- [x] **P133-1** 补掉 `docs/46`（P132-1）**§9「参考坐标」里列出、正文却一直没打开**的
  `Eagle3Builder` 前三步——`_prepare_weights:119`、`_attach_weights:190`、
  `_allocate_intermediates:220`，即「下一步候选」第二条 `Eagle3Builder` 尚未走的
  `_prepare_weights`/`_attach_weights`。见 `docs/47-mpk-eagle3-weight-prep.md`
  （快照 `mirage-project/mirage` 分支 `mpk` commit `f9eb70c`，与 `docs/06`–`docs/46` 同快照；
  **纯源码走读（资料分析），本轮无新实验**；Eagle3 device 头只被 blackwell `task_header.cuh` include，
  SM100 专属、本机不可执行）。三条 headline：
  - **(A) `_prepare_weights` 是布局契约层、不是数值层**：把 `q/k/v` 用
    `shuffle_tensors([q,k,v], num_kv_heads, 0)`（`models/utils.py:50-128`，纯 `torch.empty`+切片、
    无算数）按 GQA KV 头交错成 fused 布局 `[Q_g0,K_0,V_0,Q_g1,…]`（输入维 `2H`，对应
    `concat([embed_normed,hidden_normed])` + `fused_outdim_qkv`）；`gate/up` 用
    `shuffle_tensors([gate,up], rmsnorm_num_tasks//2, 0)` 交错，组数 = `grid_for_rmsnorm_linear_layer(2*I)//2`
    与下游 `silu_mul` grid `gateup_num_tasks//2` 一一对应；`d2t` 转 int64 + contiguous，约定
    `target_id = hot_id + d2t[hot_id]`（`eagle3_ops.cuh:85-88`）。
  - **(B) 三处硬编码 padding/dummy 是上游 task 接口的最小满足**：`lm_head` 32000 行补零到 **32256
    = 96×336**，同时服务「`grid_for_rmsnorm_linear_layer` 返回 96 整除」与「336 行 bf16 = 672 B =
    42×16 B 的 OUTPUT TMA 对齐」；`paged_attention_layer` 硬断言 `q_norm/k_norm` 非空
    （`persistent_kernel.py:1345-1352`），故塞 `(head_dim,)` 全零 dummy + `enable_qk_norm=False`；
    paged draft KV cache 是 `(max_num_pages,page_size,num_kv_heads,head_dim)` 的**自持独立** buffer。
  - **(C) `_attach`/`_new` 用一个 `_attach_cache` 兼「去重 + 张量注册表」**，`_attach` 把 torch tensor
    存 `_kept_tensors` 防 GC（常驻 kernel 存裸 GPU 指针，`attach_input:772-774` 注释）；demo 从
    `eagle3._attach_cache["eagle3_all_draft_ids"]` 取 scatter 的 dst（demo `:882`）。
    **最有价值的隐藏契约链（§5）**：`_prepare_weights` 制造全零 padding（`:155-174`）→ builder 调
    `argmax_partial_layer` **不传 `vocab_size`**（`:425-429`）故补零行（logit=0）在真实 logits 全负时
    会获胜 → 下游 `eagle3_d2t_remap_layer(draft_vocab_real=32000)` 在 device 侧
    `eagle3_d2t_remap_kernel`（`eagle3_ops.cuh:116-120`）对 `hot_id≥32000` 写 **sentinel 0** 兜底。
    三处任一改动都要同时核对另外两处。
  判据：**本文贡献 C3 的「权重/中间量清单与布局契约」**，与 `docs/46`（C3 范围/分支）、`docs/44`
  （commit/d2t device 身体）互补。配套博客 post `mega-kernel-48-mpk-eagle3-weight-prep`（weight 49）
  并更新 `mega-kernel-00-index` 第 48 项。P133-1 新留尾巴（文档 §7）：未构建；`shuffle_tensors` 布局
  未与 device 端 QKV 解包逐元素对照；`draft_vocab_size`/`intermediate_size` 取自注释/断言未加载真实
  ckpt；`_allocate_intermediates` buffer 生命周期/alias 未做数据流分析；§5 契约链是静态推断
  （「argmax 选中补零行」需真实 logits 全负这一前提，未构造实例证明）；未与其它 family builder 对照。
  **本轮新增增量。**

### P134 段内融合 task 家族走读 + 接线审计（`docs/48-mpk-fusion-tasks.md`）

- [x] **P134-1** 补掉「下一步候选」（§「当前进度/下一步」）第三条残留——「`docs/07` §3.3 的
  `norm_linear_hopper` 之外的融合层」：把整个**段内融合 task 家族**（Ampere `norm_linear` /
  `norm_linear_new` / `silu_mul_linear` + Hopper `norm_linear_hopper`）逐行打开，并做一次
  **接线审计**。见 `docs/48-mpk-fusion-tasks.md`（快照 `mirage-project/mirage` 分支 `mpk` commit
  `f9eb70c`，与 `docs/06`–`docs/47` 同快照；**纯源码走读（资料分析），本轮无新实验**）。三条 headline：
  - **(A) 融合边界存在、且跨代各写一遍**：`norm_linear`（RMSNorm+Linear）与 `silu_mul_linear`
    （SiLU×mul+Linear[+residual]）在仓库有**四份独立 device 身体**——Ampere `norm_linear.cuh`
    （461 行：`mul_broadcast_row` 现算归一化 + `SQUARE/MULSCALAR` → `reduction_sum_col ADDSCALAR/SQRT`
    + `div_col`，`K_PIPE_MAX=2`、`NUM_ITERS_M=1` 注释自陈 batch>16 会错）、同名第二版
    `norm_linear_new.cuh`（465 行：多 `num_active_tokens`、`smem_row<3,3,3>` swizzle、新 SMEM 公式、
    有效性检查改用 active-token）、`silu_mul_linear.cuh`（371 行：gate/up 拼接输入
    `d_mul=input+REDUCTION_SIZE`、SMEM 现算 SiLU、可选 residual）、Hopper
    `norm_linear_hopper.cuh`（367 行：四组 K-stage `Barrier` + TMA 三段流 + wgmma 的 warp 特化、
    `TILE_SIZE=64`→batch 须 64 倍数、`wg_decrease/increase_regs` 被注释）。
  - **(B)（headline）整个家族在快照里「已实现、未编入」**：`TaskType`（`TASK_RMS_NORM_LINEAR=102`、
    `TASK_SILU_MUL_LINEAR_WITH_RESIDUAL=105`）、`graph.cc:453/522`、`task_register.cc:489/826` 发射器、
    前端 `rmsnorm_linear_layer:883`/`silu_mul_linear_with_residual_layer:2570` **四层全通**，但——
    **没有任何生产 `task_header.cuh`（ampere/hopper/blackwell）include 这四个头**；唯一引用的单测
    `runtime_kernel_wrapper.cu` 把调用整段包在**从未定义的 `#ifdef DEPRECATED_TESTS`** 里（`using`
    别名 `:17/:23` 也被注释）、Hopper wrapper `:21/:28` 全被注释；且四头自身的 `#include "common.h"`
    （Ampere）/`"../common.h"`（Hopper）指向快照里**不存在的文件**——**即使加进 `task_header` 也编译不过**。
    是「已实现未接线」在**融合算子**层的整族实例（对照 `docs/37`/`docs/38` 薄 task「每个都有真实 builder
    调用」正好相反）。
  - **(C) 生产实际走「拆开的薄 task」**：Qwen3 MLP 段是 `rmsnorm_layer:505`→`linear_layer:512`→
    `silu_mul_layer:520`→`linear_with_residual_layer:540` 四次注册，融合版在 `models/` **零调用者**；
    历史指纹 `models/utils.py:3` 的 `grid_for_rmsnorm_linear_layer` 名字残留自融合时代、却被拆开后的
    norm/linear 用来算 grid。与 `docs/23` §5 / `docs/40`「拆开是 task 可复用/可调度的先决条件」一致——
    这次被拆掉的是**归一化**。
  另修正 `docs/07` §3.3 的默认读法（`norm_linear_hopper` 是**未编入**的融合实现，机理描述仍对）；判据
  只贡献 **C3 的范围**（同 `docs/22` §8）。配套博客 post `mega-kernel-49-mpk-fusion-tasks`（weight 50）
  并更新 `mega-kernel-00-index` 第 49 项；并把「融合 task 家族未编入 + include 陈旧不可编译」补进「已知坑」。
  P134-1 新留尾巴（文档 §7）：未构建（未验证修 `common.h` 后能否编译）；`norm_linear` vs
  `norm_linear_new` 取舍系静态推断不判；`my_gpu_id==0` 作 residual 的 TP 语义未追多卡赋值链；未扫其它
  「无生产者」融合算子；`docs/44` §10 仍余的 `smem_row`/`ldsm`/`dmem_row` 布局内部未逐行。**本轮新增增量。**

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
- `cp.async.wait_group N` 的 `N` 必须是**立即数**：写 cp.async 深环（软件流水）时环深 `D` 得模板化
  （见 P53-1/§42.7），且尾批要 `wait_group 0` 兜底否则漏等（§40.8 的 drain 坑）。
- 16 B `cp.async` 的 SMEM 目的地址须 **16 B 对齐**：`extern __shared__` 显式 `__align__(16)`
  （见 P77-1/§65.9）。
- **多载体 kernel 的载体选择要放编译期 template**：把载体选择留在运行期 `if (wmode==…)`，未走的
  分支会把寄存器压力顶上去 → **局部溢出（`localB>0`）→ 整核降速**（P78-1/§66.9：plain 路径
  `numRegs=32/localB=8`、`G≤132` 比 §65 同配置慢 ~1.6×）。`if constexpr` + 模板参数可消除；但
  比较深环等同一内核内变体时应只采信**同族相对排序**。
  **（P79-1/§67 修正）**：上面对溢出来源的猜测（「运行期 `if (wmode==…)` 分支」）是**错的**——源码
  本就是 `if constexpr` + 编译期 template，`ptxas -v` 只对各实例独立报溢出。真实来源是
  **`__launch_bounds__(256,8)` 的 32-reg 上限**（`MINB=8`）；放宽 `MINB` 即消除（plain 需 `MINB=4`
  52 regs、TMA 需 `MINB=6` 40 regs、`cp.async` 单缓冲/深环一直 0 溢出）。且该 8 B 溢出**不是** plain
  慢的原因（`localB=0` 的 `cp.async` 单缓冲与 plain 逐点重合、同步变快）——主导量是**寄存器预算 →
  数据面 ILP**。载体寄存器耐压序 `cp.async` > TMA > plain 直接读。比较同一内核内载体变体时，除固定
  `MINB` 采信同族相对排序外，还须知**排序可能随占用率/配置翻转**（TMA 在大 `G` 反超）。
- **`__maxnreg__(R)`（CUDA 12.4+ / PTX `.maxnreg`）只是「上限」，不是「设定值」**：ptxas 在 `R ≥` 该
  kernel **自然需求**时用自然需求（P81-1/§69：本专题数据面 body 自然需求 44 regs，故 `R∈{48,64,96,128}`
  实测 `numRegs` 都是 44、封顶失效；只有 `R≤44`（32/36/40/44）才精确等于 `R`）。要**精确**寄存器数须
  把 `R` 压到自然需求**以下**；`__maxnreg__` 与 `__launch_bounds__` **互斥**（不能同时用），但相同
  `numRegs` 下二者逐位等价（§69 §C）。
- **寄存器分配粒度会把寄存器预算转成每 SM 线程上限**：per-SM 可达线程 ≈ `65536/ceil(regs,8)`——
  P82-1/§70：`R=32→2048 线程`、`R=40→1536`、`R=44→1280`（**不是**理论 `65536/R` 的 1638/1489）。
  故「多给寄存器」的直接代价是**砍掉可达的每 SM 线程数**；低寄存器预算要靠更多线程补回（`O·T` 约需翻倍）。
  **（P94-1/§82 补：占用率是四重封顶，SMEM 是独立于寄存器的第四个上限）** H100 每 SM 占用率
  = `min(线程上限 2048/T, 寄存器上限 65536/(regs·T), 块上限 32, SMEM 上限 SMEM_perSM/block_smem)`。
  **只要 `regs ≤ 32`，寄存器上限恒 ≥ 线程上限**，故把 `__maxnreg__` 压到 32 以下**买不到任何占用率**
  （只会 spill）——`R=32` 是多数数据面 task「不 spill 的最小预算」的下界（§82.5）；`__maxnreg__` 下限
  24（`R<24` 被 ptxas 调到 24 并告警）。**每块静态 SMEM 会在小 `T` 时先于线程封顶**（§82.2：一块带
  8 KiB 归约数组的 kernel 在 `T=64` 被压到 `occ=24` 而非线程上限 32）——写「固定 `Wsm` 扫 `T`」的
  对照实验时，必须先让所有 `T` 的 SMEM 不成为约束，否则把「SMEM 封顶的占用率损失」误读成「块粒度效应」。
- **访存模式异质性共驻的「不可通吃」裂缝**（P84-1/§72）：把多种访存模式放进同一常驻 grid，混合吞吐
  与「各模式轮流独占整机」的时间切片期望相差 ±13%（**大体加和**），但有两处真裂缝——(a) **发射受限
  模式（`scalar4B` 型，低算术强度逐元素）与访存受限模式共驻恒低 ~13%，且与 `(T,O)` 正交**（异质性税
  0.871，换任何静态配置都补不回；掩码=抢发射/调度槽）；(b) **稀疏/gather 在低占比（10–30%）时最
  「毒化」**同 grid（`meas/exp` 掉到 0.855，物理未到墙），须先 coalescing/降物理流量或隔离到 SM 子集。
  反之 **L2 驻留 task 与 HBM 流式 task 共驻是超加和净收益**（1.111×，L2 停顿藏进 HBM 停顿）。故
  「一个常驻 kernel 里 task 类型越杂越难一套参数通吃」应细化为**「访存模式之间近似可通吃、发射受限
  × 访存受限不可通吃」**。
- **共驻税不能用「SM 角色绑定/空间隔离」绕开**（P85-1/§73）：把 §72 的按 block 交错换成**每 SM 只跑
  一种模式**（保持逻辑工作份额不变），五个混合的 `meas/E` **逐行不变**（`coal+scalar` 0.90、
  `coal+l2res` 1.11、`scalar+l2res` 0.79），ncu 的 DRAM 字节/sector/活跃 warp/`inst/cycle` **逐项相同**
  ⇒ `scalar` 的 ~10% 发射受限税与 `l2res` 超加和住在**全局（SM 间/HBM/L2/全机发射）**、不在 SM 内
  发射槽。**缓解只能改 task 的访存形态（向量化/降发射）或占比，不能靠分区/SM 归属**。另：「按 `1/r1`
  分 SM」的 balanced 分区**反而更差**（`sparse`/`l2res` 低 SM 数即饱和 HBM/L2、每 SM 速率非线性）。
- **分区角色绑定的速率曲线必须「并发（in-situ）」测，不能「单跑（solo）」测**（P86-1/§74）：把 SM 分配器
  从常数斜率换成**实测「SM 数→分区速率」曲线 + 贪心 makespan 配平**后，在**近线性混合**上确能收回
  `bind_bal` 的分区自伤（`coal+scalar` 0.865→**0.947**、反超 `interleave` 0.909），但在**含 HBM 饱和模式
  （`sparse`）**的混合上**比常数斜率还差**（`coal+sparse` 0.936→**0.664**）——**单跑曲线把「加 SM 到已饱和
  模式」的边际价值估高了，是「常数速率谬误」的另一种坐标**。分配改变争用、争用改变速率，是一个**不动点**，
  故   真实 megakernel 里一条 task 类型的「速率 vs 给的 SM 数」必须在别的 task 同时在跑时量。
- **kernel 主体里的 `atomicAdd(&ctr[smid],1)` 会被「每线程」执行，不是「每 block」一次**（P89-1/§77）：
  想给同一 SM 上的多个 block 发**唯一的局部序号**时，若把 `atomicAdd` 直接写在 kernel 主体（无 `tid==0`
  守卫），则一个 `T=1024` 的 block 会把计数加 1024 次（本节 `sm_ctr` 求和 = `132×1024=135168` 而非
  `132`），于是「`local<O`」只有极少数线程满足、其余线程 `BP=0` **空转**——症状是**吞吐虚高（coal 报
  10.7 TB/s，物理 HBM 上限 ~3.3 TB/s）+ 数值错**。必须 `if(tid==0) s_local=atomicAdd(...); __syncthreads();`
  用共享内存广播。**「吞吐超过 HBM 上限」是「丢 work/空转」的第一信号**，遇到先查有没有分支让大部分
  线程不干活。
- **`clock64`/`%clock` 是 per-SM 计数器、跨 SM 相位/频率可能不同**：用 `clock64` 做 per-block 计时时，
  必须用**每条 block 自己的 `(end-start)`**，**不能**跨 block 取 `max(end)-min(start)`——后者会把不同 SM
  的时钟偏移串进去（P87-1/§75：首版跨 SM 相减算出 0.1 GB/s 的假速率，改用 `(end-start)` 后正常）。
  要跨 SM 比较绝对时刻应改用 `%globaltimer`（全局同步）。
- **评测角色绑定分配前必须验 `sum==numSM` 且活跃模式 ≥1**（P88-1/§76）：本专题的 `setup_bind` +
  `compute_layout` 对「活跃模式的 SM 数 = 0」或「各模式求和 `!= numSM`」的非法分配，会把该模式的
  `BP` 置 0（**丢 work**），而 `run()` 的吞吐分子仍用常数 `TOT_U4` → **吞吐虚高**。P88-1 开发时
  `memcpy(nsm, cand, sizeof(nsm))` 在 `int* nsm` 形参上只拷 8 字节（2 个 int），产生过 `sum=473`
  的非法分配、报到 **26 TB/s（4.25× oracle）**的假「超加和」；改用 `sizeof(int)*NPAT` 并加 `alloc_valid`
  守门后恢复。**这类「以实测吞吐为目标函数的优化器」必须先验分配合法性**，否则把「丢 work」误读成
   「分配更优」。
- **`meas/E` 的 spread 可能是「`E` 基线对配置敏感」而非「税对配置敏感」**（P91-1/§79）：本专题的
  `meas/E` 用**同一配置下的 solo 基线**算 `E`。当某模式（如 `l2res16M`/`scalar4B`）的 solo 吞吐本身
  随 `(T,O)` 拆分变化时（§79.2：同 `Wsm=2048` 内 `scalar4B` 极差 1.35×、`l2res16M` 极差 1.61×），
  混合的 `meas/E` spread 会**混入基线漂移**——P91-1 里 `scalar+l2res` 在 `Wsm=2048` 的 spread 达 1.201，
  但其**实测绝对 GB/s 几乎恒定**（`5103/5093/5103/5094`），变的是 `E`。故**判定「税只由某量定」时，
  不能只看 `meas/E` 的 spread，必须同时看绝对吞吐**（P91-1 的 `coal+scalar` 两模式都近似 HBM/发射
  受限、`E` 对拆分不敏感，其 spread ≤1.8% 才是干净的）。**「合量 `O·T` 支配吞吐」只对 HBM 带宽受限
  模式严格成立**；发射/L2-延迟受限模式的单模式吞吐仍认块粒度 `T`。
  **（P92-1/§80 补：比值与绝对量是两种问题、会双向误导）** 同一 `Wsm=2048` 内 4 拆分的对照：
  (a) `scalar+l2res` **绝对 mixed spread 1.084 而 `meas/E` spread 1.205**——比值**放大**了基线漂移
  （`E_self` 5417→4146，因 `l2res16M` solo 随 `T` 缩升、`scalar4B` solo 随 `T` 缩降）；(b) 反过来
  `coal+scalar` **绝对 mixed spread 1.188（`T=1024` 3037 vs `T=128` 2552、大 `T` 快 1.19×）而
  `meas/E` spread 仅 1.017**——比值**掩盖**了绝对性能对块粒度 `T` 的真实依赖。故**「税（比值）与
  拆分无关」≠「混合绝对吞吐与拆分无关」**：任何「税只由某量定」的结论都要（1）声明是比值还是绝对量、
  （2）二者并读。绝对量也不会被 `E` 漂移污染（`taxRef=E_ref/mixed` 用固定参考列归一，spread 与绝对量
  同形）。
- **`T=128/O=16`+`__maxnreg__(32)` 恰好占满 H100 寄存器文件**（P91-1/§79）：`128 线程 × 32 regs × 16
  块 = 65536 regs/SM`，ptxas 报 0 spill、launch 成功（132/132 SM、每 SM 恰 16 块、`O=16` 的 bind 逐位
  对参考）。要把「warp 数 ≠ block 数」推到极端（16 块/SM）时这是可行边界；再高（`O>16`）受 `T≥…`、
  32 regs/SM 上限限制不可行。
- **给数据面 task 多要寄存器有害无益：寄存器预算 `R` 只经占用率起作用**（P93-1/§81）：固定每线程
  ILP `U` 时，把 `__maxnreg__(R)` 从「不 spill 的最小值」往上抬，`occ`（每 SM 可达块数）只降不升、
  `numRegs` 顶多回到自然需求、**在飞深度不增**——`U=8` 的 `R=32`（满占用 `occ=8`、0 spill）给出最高
  `mixed`/`meas/E=1.000`，抬到 `R≥48` 掉到 ~0.96。故应把每个 task 类型的 `R` 设成其 `U` 所需的自然
  需求（`U=1/2/4/8/16` 分别 30/28/40/54/84 reg），只在**为了更高 `U`** 时才加 `R`。与 §8「megakernel
  寄存器被固定为各 task 最大值」互为两面。
- **「每 SM 在飞 load 量 `Wsm·U`」不是普适的共驻税机制量**（P93-1/§81）：它只对**两端皆 HBM/发射
  受限**的混合（如 `coal+scalar`）成立；混合里只要有一端是 **L2 驻留**（如 `scalar+l2res`），`meas/E`
  就在 0.949–1.117 间波动、随 `U` 无单调收回、峰值可 **>1（超加和，对上 §72）**。用该机制量做 worker
  块设计前须先确认混合两端的资源类别。
- **`__maxnreg__(R)` 可用非类型模板参数传入**（P93-1/§81）：`template<int U,int R> __global__ void
  __maxnreg__(R) k(...)` 在 nvcc 13.2/sm90a 编译通过，可把「寄存器预算」与「每线程 ILP」放进同一
  `U×R` 内核面；`R ≥ 自然需求`时是上限、`R <` 时精确等于 `R`（同 §69）。
- **`ncu --replay-mode application` 会放大被测程序「自报」的计时约 2×**（P95-1/§83）：被 profile 的
  app 在每个 replay pass 里用 CUDA event 自测的 `ms`/`GB/s` 不可信——本例 `scalar+l2res` T=128
  自报 0.1145 ms，而无 ncu 只有 0.0613 ms（约 1.9×）；同一内核 ncu 的 `gpu__time_duration.sum`
  = 60.96 µs 才是真值。**用 ncu 时只采信硬件计数器与 `gpu__time_duration`，不要采信程序自报计时**；
  要绝对时间就另跑一遍无 ncu 的计时。
- **ncu 的 `smsp__average_warp_latency*`（每 warp 绝对延迟）在 `--replay-mode application` 下方向可能
  与自然计时相反**（P96-1/§84）：本专题数据面 body 在 `T=128→512` 自然计时**更快**（混合 +4.3%），
  但 ncu 的 `smsp__average_warp_latency.ratio` 反而**升**（71508→76587 cycle）——块尾/replay 扰动。
  **吞吐结论只采信无 ncu 的交错计时**；ncu 的 warp-latency 只用于**同口径相对趋势**（如
  `long_scoreboard` vs `lg_throttle` 谁在动），不可拿绝对值判快慢。
- **跨进程 / 跨 device 的复现要显式做，别默认「同进程 CI」代表一切**（P97-1/§85）：本专题微基准的
  同一进程内 bootstrap CI **不含**进程启动态 / 时钟 / 卡个体差。P97-1 把 24 进程 × 4 卡当独立重复后，
  **混合 `scalar+l2res` 的 T 趋势在跨进程/跨卡下依旧稳健**（betweenStd < withinStd），但**L2 驻留负载
  （`l2res16M`）有真实卡间差**（betweenStd 反超 withinStd，某卡绝对值偏高 ~4%），且混合比值有小的
  per-device offset（卡间 std ~0.4%，与 ~1% 的绝对效应百分位差**同量级**）。故：(a) 判定「某效应稳健」
  要么用跨进程/跨卡重复，要么明确声明只是同进程内；(b) 比**绝对效应**（尤其 `meas/E` 的百分位差）时
  必须**固定卡**或用同卡基线，不能拿不同卡上的数相减。
- **跨进程复现时把编译期/运行期状态也当变量**（P97-1/§85）：`xproc` 每 runner 用 seed 驱动
  Fisher-Yates 打乱每 round 的配置访问顺序——固定 round-robin 顺序会把「固定顺序 × 时钟漂移」这个
  系统性混淆带进「进程间差」里，误当成真实的进程间方差。
- **读 `mpk` 快照源码时，「有实现」≠「已启用」**（P102-1/`docs/16`）：`src/search/` 里至少三处
  「算法写完但未接线」——(a) `range_propagation` 的两个 `check_range` 重载**首行就 `return true;`**
  （`irange.cc:1068-1069,1085-1086`），后面 ~1100 行区间推理当前**根本不执行**、不贡献剪枝；
  (b) 符号路径 `instantiate_symbolic_graph` **首行 `++num_symbolic_graphs; return false;`**
  （`search.cc:836-842`），后面的候选赋值枚举/`to_kernel_graph`/`verify` 走不到；
  (c) `generate_kernel_graphs_symbolic`（`search.cc:446`）**无任何调用者**（`search_c.cc:106` 只调
   `generate_kernel_graphs`）。引用这些函数的行为前先确认它真被调用。
  **（P121-1/`docs/35` 补：第三种形态——「同文件族内被后继实现顶替」）** 有些实现**有调用者、也在生产
  注册路径上**，但注册函数发射的是**另一个文件**的实现：`register_mla_decode_sm100_task`
  （`task_register.cc:4102`）注释写 *"Use PR 651 MLA MTP decode kernel"*、`:4168` 发射的是
  `mla_mtp_decode_sm100_task_impl<false>`（不是同名的 `mla_decode_sm100_task_impl`）；
  `register_mla_reduce_sm100_task`（`:4206`）发射 `mla_mtp_reduce_sm100_task_impl<256>`（`:4248`）。
  于是 `mla_decode_sm100.cuh` 的 impl **只剩 standalone 单测 wrapper 调用**
  （`tests/.../runtime_kernel_wrapper_mla_decode.cu:24`）、`mla_reduce_sm100.cuh` 的 impl **零调用者**。
  判「一个 device 身体是否真在生产跑」不能只看「有定义/被 include/有 Python layer 同名 API」——须追到
  注册函数**实际 `code.e(...)` 发射的那个符号名**。
- **符号搜索的约束可满足性有两套引擎、精度不对称**（P102-1/`docs/16` §3.3）：`SymbolicTBGraph::conds`
  是 `TensorDimConstraints`（可回退集合 + **z3** `s.check()`，`tensor_dim_constraints.cc:38-47`），
  精确；`SymbolicKNGraph::conds` 只是 `unordered_set` + `check_satisfiability`
  （`tensor_dim_constraint.cc:88-148`，注释 `// rule-based checking for now`，只对常量相等/
  `const/var` 除法做启发式、**其余一律返回 `true`**），几乎不拒绝。别把「KN 级约束通过」当成合法性证明。
  （另：`TensorDimConstraints::satisfiable()` 每加一条约束就 `std::cerr` 打印整组约束
  `tensor_dim_constraints.cc:43`，是调试残留、会污染 stderr。）
- **`check_abstract_expr` 的维度约束参数是「摆设」**（P102-1/`docs/16` §1.3）：签名带
  `TensorDimConstraints const &constraints = {}`（`search.h:90-91`），但函数体只用
  `subexpr_to_final_expr` 查 egg 子表达式、**完全没用 `constraints`**（`search.cc:504-519`）。
  代数的（e-graph）与几何的（range）两路剪枝当前只有前者生效。
- **`TaskType` 的三个计数口径别混用**（P110-1/`docs/24` §7）：`enum TaskType` 是 **116 条命名条目**
  （含 12 个 `*_BEGIN/_END` 范围占位符 + 控制/scheduler/TMA 标记），`register_task` 是 **95 处
  `task_config[op]=` 赋值 / 96 个 `name == "..."` 字符串**（一对 `||` 合并分支），二者求交得
  **92 个被登记的 `TaskType`**。`docs/14` §3.2 说的「95 分支」指赋值数，不是枚举数；`TASK_ATTN_SM100`
  **由 `paged_attention_sm100` 登记**（不是死码），真正的无生产者可执行 `TaskType` 只有 3 个
  （`TASK_ATTENTION_2`/`TASK_PAGED_ATTENTION_2`/`TASK_ALLREDUCE`）。
- **SM100 单测的两个不同职责扩展撞名 `runtime_kernel_blackwell`**（P116-1/`docs/30` §2.1）：
  `tests/runtime_python/blackwell/sm100_linear/setup.py:22/25` 与 `blackwell/sm100_moe/setup.py:25/28`
  **都**以 `name='runtime_kernel_blackwell'` 构建，但注册符号互异（linear 侧 `linear_sm100_mpk`/
  `linear_splitk_sm100` `:660-665`，moe 侧 `topk_softmax_sm100`/`moe_w13_linear_sm100`/
  `moe_w2_linear_sm100`/`mul_sum_add_sm100`/`silu_mul` `:697-711`），而两组测试都**裸**
  `import runtime_kernel_blackwell`——同一 `sys.path` 里两个同名 `.so` 只能存在一个，失败症状是
  **`AttributeError`（符号来自另一个扩展）**、随安装顺序而变，比「注册表裁剪」更难定位。复制/构建
  SM100 单测前先确认目标目录的扩展名是否与邻居冲突。同层另两条：`sm100_mla_prefill_tp8/setup.py:17`
  指向**不存在的**源文件（构建必失败）、`sm100_mla_mtp_decode/` 的 `test_tp*_wrapper.cu` 是**无
  setup/pybind、含 `int main`** 的 standalone 程序（不在 pytest 收集内）。是否「真失败」取决于各目录的
  build/`sys.path` 隔离，本文只记录静态事实、**不判对错**。
- **快照 `f9eb70c` 里 `sm100_linear_fp8` 引用不存在的头文件**（P117-1/`docs/31` §6）：
  `tests/runtime_python/blackwell/sm100_linear_fp8/runtime_kernel_wrapper_sm100.cu:19`
  `#include "blackwell/linear_fp8_sm100_splitk.cuh"`，但全仓**无此文件**；且 split-K 符号
  `kernel::linear_fp8_sm100_splitk_wrapper`（`:812`）在快照里**无定义**（仅此一处提及）——因 include
  在文件顶部**无条件**，连 `split_k==1` 的非 split-K 路径也编译不过。与 `docs/30` 报的
  `sm100_mla_prefill_tp8`（setup `sources=` 指向缺失 `.cu`）是**同族、更靠前**一处：**读单测 wrapper
  时不能只看 `setup.py` 的 `sources=`/`name=`/注册表，必须读它的 `#include` 头**。静态事实、**不判对错**。
- **`sm100_fp8_moe` 的 scale 是 fp32、`sm100_linear_fp8` 的 scale 是 UE8M0 packed uint32**（P117-1/`docs/31`
  §2/§5）：同为「128K 块 + UE8M0」语义，`linear` 在 host 打包成 uint32 并支持 row-major/DeepGEMM
  列主序两种布局（`common/sm100_fp8_scale_layout.*`），`moe` 传 fp32 由 device 现场转 UE8M0
  （`fp8_group_gemm_sm100.cuh:76-87`）。比对两者 scale 契约前先确认载体，别把两种约定的张量互喂。
- **「1d2d」公开符号 ≠ 「1d2d」device 身体**（P118-1/`docs/32` §3）：`sm100_linear_fp8` 的 pybind 名是
  `linear_fp8_1d2d_sm100`（`runtime_kernel_wrapper_sm100.cu:1101`），但其 dispatch 实际调
  `launch_linear_fp8_1d1d_sm100`（`:962`，身体 `linear_fp8_sm100.cuh`）；同目录的
  `linear_fp8_1d2d_sm100.cuh`（727 行，CuTe 风格 `linear_fp8_1d2d_sm100_task_impl`）在快照里
  **只有 `task_header.cuh:37` 的 `#include`、全仓无调用者**。按名字/目录推断「哪份 device 代码在跑」
  会错，须回到 wrapper 的 dispatch + `grep` 调用者。
- **SM100 手写 GEMM 进持久化内核时用 `kNumSMs=1` 收成单 CTA**（P118-1/`docs/32` §5.3）：`linear_fp8_sm100`
  本是「`blockIdx.x` 当 tile scheduler」的多 CTA GEMM，但 MPK 的持久化 worker 里 `blockIdx.x` 是
  **worker 索引**，`register_linear_fp8_sm100_task` 只能把 `kNumSMs` 设成 `MPK_MAX_NUM_WORKERS`
   使首块恒有效、单 CTA 一趟遍历所有 tile（`task_register.cc:4817-4840` 的 HACK 注释 + `kNumStages=25`
   压 207 KB SMEM）。读这类 task 的 grid/block 语义前先确认「它是单 CTA 循环还是多 CTA grid」。
- **SM100 block-scaled task 里「尺度流水」与「数据流水」共享 stage 编号，且 scale 覆写必须被 `ab_empty`
  门控**（P119-1/`docs/33` §5）：`fp8_group_gemm_sm100.cuh` 里 DMA/MMA/SCALE 三个 warp **必须同序**遍历
  同一 `(expert,m_tile,n_tile,k_tile)`，`smem_buf=(num_prev_k_blk+k_tile)%NUM_AB_STAGE` 才一致；scale warp
  写 `sfa/sfb_smem` 前**必须**先 `wait_barrier(ab_empty)`——`ab_empty` 由 MMA 的 `umma_arrive`
  （`tcgen05.commit`）触发，会等**所有** tcgen05（含 UTCCP）完成；否则在 `k_tile_count==NUM_AB_STAGE`
  （如 TP=4/w2/mbt=64）时 scale warp 会绕回覆盖 UTCCP 还在读的 buffer（源码 `:1357-1368` 的
  `FIX 2026-04-22` 即此）。改 pipeline 深度或 TP 前先复核这条门。
- **`fp8_group_gemm_sm100.cuh` 里有三个「有定义、无调用者」的本地 helper**（P119-1/`docs/33` §9.4）：
  `fp8_make_sf_smem_desc`（`:283-296`）、`fp8_advance_umma_desc_lo`（`:267-271`）、
  `fp8_tcgen05_after_thread_sync`（`:153-155`）——MMA warp 实际调的是 `sm100_utils.cuh` 的
  `make_sf_desc`/`advance_umma_desc_lo`，scale warp 用的是 `fence_view_async_shared`。按函数名/文件内定义
  推断「哪份实现真在跑」会错（同 `docs/16`/`docs/20`/`docs/32` 的「已实现未接线」族）；须 `grep` 调用者。
- **常驻 kernel 内共驻的 SM100 task：TMEM alloc/dealloc 必须用同一个 warp**（P120-1/`docs/34` §3.5）：
  CuTe 的 TMEM allocator 要求 `tcgen05.alloc/dealloc` 由**同一 warp** 在所有共驻于持久化 kernel 的 task
  里发出。MPK 的 FP8 CUTLASS 家族（`linear_fp8`/`moe_linear`/`fp8_group_gemm`）都用 **warp0**；若别的
  task（如 MLA-MTP decode）用别的 warp 分配 TMEM，同 worker 上**后续 FP8 task 会挂在其
  `tmem_allocator.allocate()`** 里（`mla_mtp_decode_tp2_sm100.cuh:190-195` 注释）。写/移植常驻 device task
  时先确认它与邻居用同一 warp 做 TMEM 生命周期。另：MPK worker 是 256 线程、许多手写 task 只用前 128，
  **提前 `return` 的半 CTA 会在 SM100 ITS 下漂移领先、使后续 `__syncthreads()`（bar0）在缺席 warp 上
  触发 → 静默跳过下一个 task 身体 → 挂起**（`bugfix.md` Bug 14）；正确做法是只同步活跃半 CTA 的**命名
  屏障 `bar.sync 12,128`**（ID 须落 CUTLASS 用户区 `≥8`，误用保留区 ID 1–7 会留脏状态毒死后续 task，
  Bug 19）。`bugfix.md` 本身**不在快照里**，以上仅为源码注释自陈因果。
- **共享原语层 `tasks/common/` 是按 Ampere 128 线程写的、上层常驻 worker 是 256 线程**（P124-1/`docs/38` §2.4）：
  `worker_config.h:18` 硬编码 `NUM_THREADS=128`，`utils.cuh:97-116` 的 `clear_smem_buffer` 以
  `i += NUM_THREADS`（128）跨步清零，而同文件 `vec_zero_t::fill_zero`（`:124-145`）用 `blockDim.x`——在
  MPK 的 256 线程 Blackwell worker（`docs/34` §3.5）里前者只让前 128 线程参与 bulk 循环。**是否安全取决于
  调用点是否另有覆盖，本文只记口径、不判对错**。写/移植常驻 device task 时，凡用到 `NUM_THREADS`/
  `NUM_WARPS`/`clear_smem_buffer` 的公共助手，先确认它与所在 worker 的真实线程数一致。
- **`ampere/mma.cuh` 的 in-place 累加器必须用单一读写 `"+f"`**（P124-1/`docs/38` §6）：`mma_m16n16k16_bf16bf16bf32`
  若把累加器写成只写 `"=f"(D[i])` + 独立 `"f"(C[i])` 输入，`#pragma unroll` 下编译器会复用「已死亡」累加器的
  寄存器、**跨 fragment 污染**（注释 `:22-30` 所称「K≥2/`MMA_ITERS_M`≥2 bug」）；`attention_sm100.cuh:506-516`
  记其下游后果：曾 `if(m>0) continue` 跳过后继 m-tile → 损坏 EAGLE3 投机 token 流。写多 tile 展开的
  `mma.sync` 手写 kernel 时，累加器务必 `"+f"`（源/目的同寄存器）。另：`common/copy_sm80.cuh:51/68` 的
  `cp.async.ca.shared.global.L2::128B` 正是本专题 `docs/04` §39–§56 实测的取 task 原语本体。
- **投机解码 device 头「被登记」≠「被编进生产」（P125-1/`docs/39` §1）**：生产 device 编译入口
  `persistent_kernel.cuh:48-52` 按 arch 三选一 include `tasks/<arch>/task_header.cuh`，**只有 blackwell**
  的 `task_header.cuh:62-64` 引了 MTP/Eagle3 三件；`tasks/speculative_decoding/target_verify.cuh`
  与 `prompt_lookup.cuh`（`TaskType 112/113/114` 的 device 身体）**不被任何生产 `task_header` include**，
  只在单测 wrapper（`tests/runtime_python/runtime_kernel_wrapper.cu:13/27/1845`）出现且其 `using`/`m.def`
  **都被注释**——而 `graph.cc:541-552` 的 register 与前端 handler 是活的。**「`grep` 到注册/前端/TaskType」
  不代表该 device 头真进了常驻 kernel 的编译单元**；判 prompt-lookup 路径可用性须同时核 `task_header.cuh`
  的 include 集与单测接线（本专题只记静态事实、不判对错）。同目录另两枚「有定义、全仓零调用者」死代码：
  `target_verify_synthetic_kernel`（`target_verify_mtp.cuh:191-242`）、`find_ngram_global_kernel_sequential`
  （`prompt_lookup.cuh:142-153`）。另：`target_verify.cuh`/`prompt_lookup.cuh` 用 `#include "../common.h"`
  而 `target_verify_mtp.cuh`/`mtp_token_ops.cuh`/`eagle3_ops.cuh` 用 `"tasks/common/common_header.cuh"`
  （快照无 `tasks/common.h`；同族写法见 `hopper/norm_linear_hopper.cuh:18`、多个 `ampere/*.cuh`）——
  读这批头时先确认它的 include 约定与所在的 `task_header` 是否真在编译路径上。
- **常驻 kernel 里 batch/request 号只能取 `task_metadata.request_id`，不能取 `blockIdx.x`**
  （P125-1/`docs/39` §5）：`mtp_prepare_verify_input_kernel`（`mtp_token_ops.cuh:101-104`）、
  `mtp_build_embed_input_kernel`、`eagle3_commit_kernel` 都显式用 `request_id`；因为常驻 kernel 里
  `blockIdx.x` 是 **worker 索引**（同 `docs/32` §5.3 的 `kNumSMs=1`、`docs/34` §3.5 TMEM-warp、
  `docs/37` §2 薄 task「遍历全 row」）。写/移植设备端投机/采样 task 时，凡按 `batch`/`req` 索引
  全局缓冲，一律走 `task_metadata.request_id`。
- **「被 `#include`」≠「被调用」：RoPE/norm 也有一枚同族死代码**（P131-1/`docs/45` §3.3）：
  `tasks/blackwell/rotary_embedding_sm100.cuh` 只被 `attention_sm100.cuh:23` `#include`、
  **全仓（含 `tests/`）零函数调用**；`attention_sm100` 在 `qk_norm=false, rope=true` 分支实际调的是
  **`rotary_embedding_hopper`**（`:435`/`:455`）。**勘误 `docs/37` §4.2**（原称 attention_sm100
  「选它」）。这是「按文件/名字/`#include` 推断哪份 device 代码在跑会错」的又一实例（同 `docs/32` §3、
  `docs/33` §9.4、`docs/38`、`docs/16`/`docs/20`）——判 rope 走哪条必须追 `attention_sm100` 的
  **实际调用符号**。同族口径坑还有：本家族 `NUM_THREADS` 有四种来源（全局 128 / hopper-rmsnorm 模板
  默认 256 / hopper-rope 默认 128 / hopper SMEM-rmsnorm 显式模板参）并存，barrier ID 用 6/7/9。
- **段内融合 task 家族「四层布线全通、却一个编译单元都没进」**（P134-1/`docs/48` §4）：
  `TASK_RMS_NORM_LINEAR=102`/`TASK_SILU_MUL_LINEAR_WITH_RESIDUAL=105` 的 `graph.cc` 分支、
  `task_register` 发射器（发射 arch-agnostic 名 `kernel::norm_linear_task_impl`/
  `silu_mul_linear_task_impl`）与前端 `rmsnorm_linear_layer`/`silu_mul_linear_with_residual_layer`
  都是活的，但**没有任何生产 `task_header.cuh` include 这四个头**（`ampere/norm_linear.cuh`/
  `norm_linear_new.cuh`/`silu_mul_linear.cuh`/`hopper/norm_linear_hopper.cuh`）；唯一引用它们的
  `tests/runtime_python/runtime_kernel_wrapper.cu` 把调用整段包在**全仓从未 `#define` 的
  `#ifdef DEPRECATED_TESTS`** 里、`using` 别名（`:17/:23`）也被注释，Hopper wrapper `:21/:28` 亦被注释；
  且四头自身 `#include "common.h"`/`"../common.h"` **指向快照里不存在的文件**（不存在 `tasks/common.h`，
  只有 `tasks/common/common_header.cuh`）。故「`grep` 到 `TaskType`/注册/前端」**不等于该 device 身体被
  编进任何 kernel**——判一个融合 task 是否真能被点亮，必须核对 `task_header.cuh` 的 include 集**且**该头
  自己的 include 能解析（同 `docs/32` §3、`docs/45` §3.3、`docs/24` §4 的「名字/注册会骗人」族）。
  另：`norm_linear` 与 `norm_linear_new` **同名互斥**（都定义 `norm_linear_task_impl`，靠 include 选择、
  同时 include 会重定义），当前两版都没进任何编译单元。

## 当前进度 / 下一步

- 当前进度：**P134-1 完成（1/1）**。补掉「下一步候选」第三条残留——`docs/07` §3.3 的
  `norm_linear_hopper` 之外的**段内融合 task 家族**，并做一次**接线审计**。见
  `docs/48-mpk-fusion-tasks.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：
  **(A)** 融合边界存在且跨代各写一遍——Ampere `norm_linear.cuh`（461 行）、同名第二版
  `norm_linear_new.cuh`（465 行）、`silu_mul_linear.cuh`（371 行）、Hopper `norm_linear_hopper.cuh`
  （367 行 TMA+wgmma warp 特化）；**(B)（headline）整个家族「已实现、未编入」**——`TaskType 102/105`、
  `graph.cc`、`task_register` 发射器、前端 layer 四层全通，却**无任何 `task_header` include 这四个头**、
  唯一引用的单测整段包在从未定义的 `#ifdef DEPRECATED_TESTS` 里、四头自身的 `#include "common.h"`
  指向快照里不存在的文件；**(C)** 生产实际走「拆开的薄 task」（Qwen3 `rmsnorm_layer`→`linear_layer`→
  `silu_mul_layer`→`linear_with_residual_layer`）。并修正 `docs/07` §3.3 的默认读法。配套博客 post
  `mega-kernel-49-mpk-fusion-tasks`（weight 50）并更新 `mega-kernel-00-index` 第 49 项。
- 此前：**P133-1 完成（1/1）**。补掉 `docs/46` §9「参考坐标」里列出、正文却没打开的
  `Eagle3Builder` 前三步（`_prepare_weights:119`/`_attach_weights:190`/`_allocate_intermediates:220`），
  即「下一步候选」第二条 `Eagle3Builder` 尚未走的 `_prepare_weights`/`_attach_weights`。见
  `docs/47-mpk-eagle3-weight-prep.md`（快照 `f9eb70c`，**纯源码走读、无新实验**；SM100 专属、本机
  不可执行）。三条 headline：**(A)** `_prepare_weights` 是**布局契约层**——`shuffle_tensors`
  按 GQA KV 头 / 按 rmsnorm-linear 任务块把 QKV、gate/up 交错成 fused 布局（纯 `torch.empty`+切片、
  无算数），`d2t` 的 `target_id = hot_id + d2t[hot_id]`；**(B)** 三处硬编码 padding/dummy 是上游
  task 接口的最小满足——`lm_head` 32000→32256（96×336，服务 96 grid 整除 + 336 行 16 B TMA 输出对齐）、
  `paged_attention_layer` 硬断言需要的 dummy qk-norm、自持 paged draft KV cache；**(C)** `_attach`/`_new`
  用同一 `_attach_cache` 兼去重层与张量注册表（裸 GPU 指针须 `_kept_tensors` 防 GC）。并接出**最有价值的
  隐藏契约链**（§5）：补零 → `argmax` 不传 `vocab_size` 允许 padding 获胜 → `d2t_remap` 对
  `hot_id≥32000` 写 sentinel 0 兜底。判据贡献 C3 的「权重/中间量清单与布局契约」，与 `docs/46`/`docs/44`
  互补。配套博客 post `mega-kernel-48-mpk-eagle3-weight-prep`（weight 49）并更新 `mega-kernel-00-index`
  第 48 项。P133-1 新留尾巴见文档 §7（未构建、shuffle 布局未与 device 端 QKV 解包逐元素对照、
  数值取自注释/断言未加载真实 ckpt、buffer 生命周期未做数据流分析、§5 契约链系静态推断）。
- 此前：**P132-1 完成（1/1）**。补掉 `docs/40`/`docs/43`/`docs/44` 一直没打开的「谁把 K 个 draft
  step 拼成一张图」，也即「下一步候选」第一条——`eagle3_commit` 的 K>1「proper mbt-parallel chain」
  缺口（`eagle3_ops.cuh:217` 自陈未实现）的静态追述。见 `docs/46-mpk-eagle3-draft-loop.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**；SM100 专属、本机不可执行）。三条 headline：
  **(A)** `Eagle3Builder.build_draft_loop`（`builder.py:271-449`）是「`for step in range(K)` 展开的
  单图」，每 step 一整层 draft transformer（15 段算子链）、所有中间张量 `(mbt,…)` 跨 step 复用、
  靠 `draft_hidden`/`target_token` 两条接力串行（无数据并行空间）；
  **(B)** K 只改 attention 一处——K=1 默认处理全部 `mbt` 输入槽（「mbt 条并行链」），K>1 改
  `q_len_override=1`+`tail_offset=K-step` 每 step 只写 1 行；因 `mbt==K+1`（demo 断言），K>1 只有
  1 行有效、下游 mbt-wide 簿记把 garbage 写进 `all_draft_ids[1..]`，commit 被迫 `src_slot=0`；
  **(C)** 把缺口拆成 4 条必要条件，并说清它与 `MAX_TOKENS=3` slot-0 bug family 共用同一段 attention
  代码；另记 demo 注释的 flat-cache 单 kernel 与 builder 实走 `paged_attention_layer` 的静态落差。
  判据贡献 C3、与 `docs/43` 互补。配套博客 post `mega-kernel-47-mpk-eagle3-draft-loop`（weight 48）
  并更新 `mega-kernel-00-index` 第 47 项。P132-1 新留尾巴见文档 §8（未构建、「有效行=slot 0」系推断、
  4 条必要条件系推断、slot-0 分歧未复现、注释落差只记不判）。
- 下一步候选：可继续走其一——`docs/35` §9 遗留的其它未走读相邻层（CUTLASS `collective_builder` 侧
  QK/PV atom——`mla_sm100_2sm.cuh` 的 CUTLASS 内部，`docs/36` §8 明留）；或 `docs/44` §10 仍余的
  `smem_row`/`ldsm`/`dmem_row` 布局内部（`docs/48` §7 亦记为尾巴）；或 `docs/48` §7 新留的其它
  「无生产者」融合算子扫描 + `my_gpu_id==0` 作 residual 的 TP 语义追链；或 `docs/46` §6 的
  FLAT-CACHE draft attention 设计的完整追述（`docs/47` 已补 `_prepare_weights`/`_attach_weights`）。
- 此前：**P131-1 完成（1/1）**。补掉 `docs/37` §4（P123-1 对 SM100 norm/rope 只给摘要）、
  `docs/07` §3.1（只走 `rms_norm_hopper_impl`）与 `docs/44` §10（P130-1 明留「`rms_norm_sm100`/
  `rotary_embedding_hopper` 内部未逐行」）三处空白：把 norm/rope 当成**跨 Ampere/Hopper/Blackwell
  三代演化的 task 家族**逐行打开，见 `docs/45-mpk-norm-rope-tasks.md`（快照 `f9eb70c`，
  **纯源码走读、无新实验**；SM100 专属、本机不可执行）。三条 headline：
  **(A)** 三层形态——两条独立 global-memory norm 身体（Ampere `rms_norm_impl` 断言 `BATCH_SIZE==1`、
  Hopper `rms_norm_hopper_impl` 放开多 batch + 短尾 tile；前端按 `target_cc>=90` 二选一，
  **H100/sm90 走 Hopper**，即 `docs/09` 真机测过的 test_mode RMSNorm 本体）vs 被 attention 内联的
  SMEM 融合 norm/rope；
  **(B) 融合边界 + 死代码**：全仓无 `TASK_ROTARY`/`TASK_ROPE`（RoPE 要么折进 attention、要么折进
  DFlash 独立 task，**同数学两边界**）；**修正 `docs/37` §4.2 静态事实错误**——`rotary_embedding_sm100`
  全仓零调用者，`attention_sm100` 实调 `rotary_embedding_hopper`（`:435/:455`）；
  **(C) 同步/线程数口径百家争鸣**：cooperative groups / `wg_sync` / cutlass `NamedBarrier` 三套同步、
  `NUM_THREADS` 四种来源、barrier ID 6/7/9、partial RoPE 三代不对称（Ampere 不支持 → GLM-4.6 只能
  cc≥90）。判据只贡献 C3。配套博客 post `mega-kernel-46-mpk-norm-rope`（weight 47）并更新
  `mega-kernel-00-index` 第 46 项。P131-1 新留尾巴见 §8（未构建、死代码定性系推断等）。
- 下一步候选：可继续走其一——`eagle3_commit` 的 K>1「proper mbt-parallel chain」缺口（源码注释
  `eagle3_ops.cuh:217` 自陈未实现）的静态追述；或 `docs/35` §9 遗留的其它未走读相邻层
  （`mla_prefill_host.cu` 原型、CUTLASS `collective_builder` 侧 QK/PV atom）；或 `docs/44` §10 仍余的
  `smem_row` 内部与 `docs/07` §3.3 的 `norm_linear_hopper` 之外的融合层。
- 此前：**P130-1 完成（1/1）**。补掉三条明留的尾巴——`docs/37` §5.1（只给 750 行摘要）、
  `docs/39` §6.3、`docs/43` §9 头号（「`paged_attention_sm100` 内部未逐行」）。把 MPK 唯一的
  通用 paged GQA attention device 身体逐行打开，见 `docs/44-mpk-paged-attention-sm100.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**；SM100 专属、本机不可执行）。三条 headline：
  **(A)** 唯一用 Ampere 手写 `m16n16k16` `mma.sync`（非 wgmma/UMMA）的通用 paged GQA：
  一个 task = 一个 (request, KV head)、在线 softmax 全在寄存器、QK^T `1x4x1` / PV `1x1x4`、
  QK-norm / partial RoPE / sliding window / attention sink 全编进单 task；
  **(B)** §2 坐实「为压进 Blackwell SMEM 上限」的重构史——O 复用 Q 段（`:187-196`，issue #740）、
  跨 warp 归约的 m/d/o buffer 只存一个 tile（`:209-228`，issue #702），是 C2 对 device task 的约束；
  **(C)** §7 把 `MAX_TOKENS=3` slot-0 bug family 钉回源码级——真正因果/窗口掩码在 `:556-559`
  （转述的 `:519` 是行号漂移）、`:505-516` 的 `if(m>0) continue` workaround → `"=f"`/`"f"` 别名
  累加器跨 fragment 污染 → `"+f"` 修复（补上 `ampere/mma.cuh` 的 attention 侧消费者 `:529`），
  并校正「`-inf` NaN 传播」已被 `:591-595`/`:651-652` 两处守卫覆盖。判据只贡献 C1+C3。配套博客
  post `mega-kernel-45-mpk-paged-attention-sm100`（weight 46）并更新 `mega-kernel-00-index` 第 45 项。
  P130-1 新留尾巴见 §10（slot-0 分歧未独立复现、命名屏障 ID 6/7 只记不判、未构建等）。
- 此前：**P129-1 完成（1/1）**。补掉 `docs/39`（P125-1）**§9「未展开的更深层」**明留的尾巴
  「`eagle3_commit_kernel` 的 device 身体（`eagle3_ops.cuh:157-315`）与 `accept_hist` 调试历史」。
  见 `docs/43-mpk-eagle3-commit.md`（快照 `f9eb70c`，**纯源码走读、无新实验**；SM100 专属）。
  三条 headline：**(A)** `eagle3_commit_kernel` 是全仓唯一「verify-aware 收尾 task」，一次调用做完
  四件事（写 verified 前缀 + 写 K draft + `new_token_nums` 步进 + `drafts_prev` 跨迭代快照），
  `src_slot` 是 K=1 `ac-1`/K>1 强制 0 的已知坑；**修正 `docs/39` 两处**——接口是 **`(5,2)`**
  （`graph.cc:920`）而非「7 入 2 出」、`eagle3_commit_layer` **在 demo 有调用者**
  （`demo/qwen3/demo_30B_A3B_eagle3.py:880`）而非「无任何调用者」；**(B)** 跨迭代 draft 依赖由
  attach_input 快照绕过，运行时消费端是 `prepare_next_batch` 的 `#ifdef MPK_SPEC_DECODE`
  （`persistent_kernel.cuh:295-316`）；**(C)** `accept_hist`（`:254-309`）是 device 侧自陈 bug 调查史，
  根因指向 `paged_attention_sm100` `MAX_TOKENS=3` slot-0 分歧，与 `docs/38` §6 的 `"+f"` MMA 修复
  同一 family。判据只贡献 C1+C3。配套博客 post `mega-kernel-44-mpk-eagle3-commit`（weight 45）
  并更新 `mega-kernel-00-index` 第 44 项。P129-1 新留尾巴见 §9。
- 此前：**P128-1 完成（1/1）**。补掉 `docs/41`（P127-1）**§6 明留的两条尾巴**——「`mtp_float_scatter`
  的 device 身体（`mtp_token_ops.cuh`）未逐行」与「`target_verify_*` 如何消费 `prob_extract` 的输出
  （拒绝采样比较 `P_target > u·P_draft`）只按注释转述」，也即 `ROADMAP`「下一步候选」第二条。见
  `docs/42-mpk-mtp-verify-chain.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：
  **(A)** 概率 MTP 拒绝采样是一条「两处生产、一次消费」的跨迭代链——draft 侧
  `softmax_gather(:1876)→mtp_float_scatter(:1885,slot_idx=step)`、main graph 侧
  `softmax_gather(:2220)→prob_scatter(:2229,step)`、verify 侧 `prob_extract(:1993,offset=step)→
  target_verify_probabilistic(:2007)→mtp_accept_commit(:2037)`，时序锚点是「写 `step`、读 device 侧
  硬编码 `offset+1`」（`prob_scatter_sm100.cuh:42-46` vs `:68`）；**(B)** `mtp_token_ops.cuh` 四身体
  分属两套 batch 索引契约——两个 scatter 批并行（`b=threadIdx.x`）、prepare/embed 按 `request_id`；
  **(C)** verify/commit 家族（`target_verify_{greedy,strict,probabilistic}`/`mtp_accept_commit`/
  `eagle3_commit`）**无 `request_id`、从 0 索引 ⇒ device 侧只处理批元素 0**（注释 `:28-29` 自陈），
  `seed[0]` 为标量。另补 builder 三处新落差（死 `draft_probs` `:1967-1969`、双 `rng_seed`
  `:1977`/`:2002`、guard 外调用 `:2007`）。判据只贡献 C1+C3。配套博客 post
  `mega-kernel-43-mpk-mtp-verify-chain`（weight 44）并更新 `mega-kernel-00-index` 第 43 项。
  P128-1 新留尾巴见 §7（`seed` 初始化未追、落差只记不判、`_draft_prob_buffer` probabilistic 专属、
  `mtp_build_embed_input` 多 block 覆写只记不判、未构建）。
- 此前：**P127-1 完成（1/1）**。补掉 `docs/39`（P125-1）**§9「未展开的更深层」**第一句与
  `ROADMAP`「下一步候选」第一条——把 `softmax_gather_sm100.cuh`/`prob_scatter_sm100.cuh`（含
  `prob_extract_task_impl`）三个概率簿记 device 身体逐行打开，见 `docs/41-mpk-prob-bookkeeping.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**；三者均为 SM100 路径）。三条 headline：**(A)** 全仓
  最小的 device 身体（193 行 3 函数）却是**活的生产 task**——五层布线全通（`TaskType` 282/284/286、
  `task_register.cc:2731/2808/2839`、`graph.cc:726/740/745`、`persistent_kernel.py:2957/2992/3010`、
  `builder.py:1876/1993/2220/2229`），与 `docs/36`/`docs/40` 死代码相反；**勘误**：`prob_extract_sm100.cuh`
  文件名不存在，`prob_extract_task_impl` 是 `prob_scatter_sm100.cuh:54-76` 的第二个函数；**(B)** 三处
  「注释/代码/布线」落差——`softmax_gather` 注释写 `Grid:(BATCH_SIZE,1,1)`+`exp(logit-max-log_sum)`
  而代码单 block 串行 batch（`:46`）+ 除法（`:108`）、调用方一律 `grid=(1,1,1)`；`prob_scatter` 同；
  根源是「常驻 kernel `blockIdx.x==worker id` ⇒ 身体内部循环」（`docs/32`/`docs/34`/`docs/37`）。
  另记 `smem_max[8]`/`smem_sum[8]` ⇒ `blockDim.x≤256` 的**隐性约束**（无 assert）；**(C)** 概率 MTP
  旁路**半实现**——device+布线全活但 `builder.py` 仍留 `TODO`（draft 概率抽取空 `:1973-1974`，注释求的
  「slice by runtime offset kernel」其实就是 `:1993` 的 `prob_extract`）；且**两种每 step 槽载体并存**
  （draft 用编译期槽号 `mtp_float_scatter` TaskType 285 / main graph 用运行期 `prob_scatter` TaskType 284）。
  判据只贡献 C1+C3。配套博客 post `mega-kernel-42-mpk-prob-bookkeeping`（weight 43）并更新
  `mega-kernel-00-index` 第 42 项。P127-1 新留尾巴见 §6（注释过时只记不判、`mtp_float_scatter` 身体
  未逐行、`target_verify_*` 消费 `prob_extract` 只按注释转述、未构建）。
- 此前：**P126-1 完成（1/1）**。补掉 `docs/39`（P125-1）§9 尾巴与 `ROADMAP`「下一步候选」第二条——把仓库里**唯一归档在 `tasks/deprecated/` 下的 task** `paged_attention.cuh`（496 行）逐段打开，见 `docs/40-mpk-deprecated-paged-attention.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：**(A)** 它是仓库里唯一把「paged-attention 计算 + 新 token K/V 写回页缓存」融进同一 task 的 device 身体（`:473-484`），而今天拆成「只读 attention + 独立 KV store + 独立 norm/rope」——**融合边界不是越融越好，「把写回从读里拆出去」是 task 可复用/可调度的先决条件**；**(B)** 自包含单 batch×单 KV head×单 CTA 的 FA2 手写实现（`m16n16k16` mma 的 1x4x1/1x1x4、在线 softmax、4 warp 分片 + warp0 全量合并、融合 QK-norm/RoPE、NHD 页缓存 + 双缓冲 64-key tile），是后世全部特性的最早期快照；**(C)** 接线是「彻底的孤儿」——无任何 include、`DEPRECATED_TESTS` 全仓无人 `#define`、`using`/`m.def` 全被注释、wrapper 只 include ampere/blackwell `task_header`。另记两条静态观察只记不判（`page_indices` 只写尾段、`S_O_BUFFER_SIZE` 被注释）与旧式裸 include 约定。判据**只贡献 C3 的范围**（且是被淘汰的划分）。配套博客 post `mega-kernel-41-mpk-deprecated-paged-attention`（weight 42）并更新 `mega-kernel-00-index` 第 41 项。P126-1 新留尾巴见 §5（退役原因系推断、布局/旧 norm/rope 内部未展开、未构建）。
- 此前：**P125-1 完成（1/1）**。补掉 `docs/14`/`docs/24`/`docs/26` 只列名字、`docs/34`–`docs/37` 只走 MLA-MTP attention 的那一整层 device 目录 `tasks/speculative_decoding/`（5 头文件 1014 行），见 `docs/39-mpk-speculative-decoding.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：**(A)** 全部是「查答案 / 搬 token」的簿记型 device 身体、无一吃 TC，却是投机正确性枢纽（`accepted_count`/`new_token_nums`/`tokens[]` 落位→运行时步进），是 `docs/04` §8/§10「task 数/事件扇入扇出」来源；**(B)** 同一「验证 draft」语义三套并行实现接线各异——prompt-lookup 贪心（`TaskType 114`）、DeepSeek-V3 MTP strict/probabilistic/commit（`271/283/272`）、两枚「有定义零调用者」死代码（`target_verify_synthetic_kernel`、`find_ngram_global_kernel_sequential`）；**(C)** Eagle3 Python 落差——类注释声明完整 verify 管线但 `build_draft_loop` 只跑 draft loop，`copy_layer`/`eagle3_commit_layer` 快照无调用者。§1 记静态 include 落差（`target_verify.cuh`/`prompt_lookup.cuh` 不在生产 `task_header` include 集、单测 `using`/`m.def` 被注释，只记不判）；§6.3 转述 `eagle3_commit_kernel` 注释里 `paged_attention_sm100` `MAX_TOKENS=3` slot-0 分歧的 bug 调查（同 `docs/38` §6 `"+f"` 的 MMA_ITERS_M>1 family）。判据**只贡献 C1+C3**。配套博客 post `mega-kernel-40-mpk-speculative-decoding`（weight 41）并更新 `mega-kernel-00-index` 第 40 项。P125-1 新留尾巴见 §9（include 落差只记不判、`softmax_gather`/`prob_scatter`/`prob_extract` 三身体未逐行、`MAX_TOKENS=3` slot-0 分歧未独立复现、接受率数字为注释自报、`lookahead`/`eagle3` handler 字典缺失只记 `ValueError`）。
- 此前：**P124-1 完成（1/1）**。补掉 `docs/37`（P123-1）**§8「局限」明留的两条尾巴**——「`sampling_from_logits_kernel`（`tasks/common/sampling.cuh` 的 CUB BlockReduce 内部）未展开、`mma_m16n16k16`（`tasks/ampere/mma.cuh`）汇编未逐条」与「`mla_prefill_host.cu` 原型来源仍未走」。见 `docs/38-mpk-common-primitives.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：**(A)** `tasks/common/` 是全部 device task 的 arch-agnostic 地基（伞状 `common_header.cuh` 被 41 个 task 引用、`utils.cuh` 被 33 个直引；`ampere/mma.cuh` 被两个非 MLA SM100 attention 引用）——「Blackwell」目录的通用 attention 仍在用 Ampere 手写 `mma.sync` + `cp.async` + CUB；**(B)** `common/sampling.cuh` 是 FlashInfer 移植的一趟式 Gumbel-Max（借 CUB `BlockReduce::Sum` 当 argmax 归约），生产登记 `TASK_SAMPLING_SM100`、与 SM100 分块版共用同一 `TaskType`；**(C)** `ampere/mma.cuh` 的 `"+f"` in-place 累加器修复是 load-bearing 的（决定 `attention_sm100` `MMA_ITERS_M>1`、进而 EAGLE3 token 流正确）。另记 `clear_smem_buffer` 的 128/256 线程口径落差、`copy_sm80.cuh` 即 `docs/04` §39–§56 实测原语本体、`mla_prefill_host.cu` 收口为「不在快照、外部来源」。判据**只贡献 C3**。配套博客 post `mega-kernel-39-mpk-common-primitives`（weight 40）并更新 `mega-kernel-00-index` 第 39 项。P124-1 新留尾巴见 §9（CUB 内部未展开、`bfloat16.h` 只读接口、`clear_smem_buffer` 128/256 口径并存不判、原型源文件缺失不可核、未与生产端到端对跑）。
- 此前：**P123-1 完成（1/1）**。补掉 `docs/35`（P121-1）**§9**与 `docs/36`（P122-1）**§8**
  明留的同一处相邻尾巴——「注意力之外的 SM100 task 家族」，见 `docs/37-mpk-sm100-misc-tasks.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**；SM100 在目标卡 H100/sm90 不可执行）。把
  `tasks/blackwell/` 下「非 GEMM、非 MLA」的薄 task 逐个打开，并核对 `TaskType`/`graph.cc`/
  `persistent_kernel.py`/builder 四层布线。三条 headline：**(A)** 与 `docs/33`–`docs/36` 的重 task
  相反，本批是「极简 device 身体 + 静态角色」（几十到几百行、朴素命名屏障、无 TMA/TMEM），是 megakernel 的
  **调度与数值枢纽**（决定 `TaskType` 数与事件扇入/扇出，对上 `docs/04` §8/§10 成本侧）；**(B)** 因
  `blockIdx.x == worker id`，薄 task 一律**遍历全 row**（`per_token_group_quantize_fp8.cuh:87-92` 的
  row-identity 修复，同族于 `docs/32` §5.3 `kNumSMs=1`、`docs/34` §3.5 TMEM-warp）；**(C)** 同一「路由/
  采样」语义有 3–4 份**由模型家族决定**的实现（`topk_sigmoid`/`glm_moe_router`/`inkling_moe_router` 路由
  三兄弟映射同一 `mul_sum_add` 下游；采样 common 库版 vs SM100 分块版；`TASK_SAMPLING_SM100` 被
  `sampling_sm100` 与 `sampling_reduce_sm100` 共用）。**关键事实：本批每个 task 都有真实 builder 调用**
  （glm4_moe/inkling/dflash/deepseek_v3/qwen3/gpt_oss），与 `docs/35`/`docs/36` 的「有实现无生产者」相反。
  判据**只贡献 C3**。配套博客 post `mega-kernel-38-mpk-sm100-misc`（weight 39）并更新
  `mega-kernel-00-index` 第 38 项。P123-1 新留尾巴见 §8（CUB 内部未展开、`mma_m16n16k16` 未逐条、命名
  屏障 ID 6/7 与 `docs/34` §3.5「须 ≥8」口径**并存不判**、`mla_prefill_host.cu` 原型仍未走）。
- 此前：**P122-1 完成（1/1）**。补掉 `docs/35`（P121-1）**§9「局限」明留的头号尾巴**——
  `mla_sm100_2sm.cuh`（3165 行，2SM 协作 attention）未逐行，见 `docs/36-mpk-sm100-mla-2sm.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：**(A)** 它是 MLA 家族里唯一的「整颗
  CUTLASS kernel」形态——`device::MLA::run()` 用 `ClusterLauncher::launch` 自持发 `(2,1,1)` cluster
  grid（`:3102-3109`），全仓 caller 只有 standalone 单测 wrapper + `task_header.cuh:53` 聚合 include，
  **无 `TaskType`/注册/前端**（比 `docs/35` §3/§4 的旧 decode/reduce 更彻底）；**(B)** `kIs2Sm=true`
  （`:610`）派生 `Allocator2Sm`/`2SmSm100`/`kWarpsInN=2`、只有 leader CTA 发 UMMA+分配 TMEM、TMEM 对半切、
  softmax 2×2 数据并行 peer 归约（SMEM 交换 + 命名屏障 + `peer_index=(threadIdx.x+64)%128`）、LSE 只由
  前半 CTA 写（源码自陈索引未重排），是 P33 `cta_group::2` 的真实 SM100 消费者；**(C)** 七 warp 五
  pipeline（`PipelineLoadQK/S/P/O/PT`）把整颗查表注意力（QK→softmax→PV→rescale→epilogue）压进一条流水，
  分页 KV 用 `CustomStride`/`Gather` 在 CuTe 层表达。判据**只贡献 C3**。配套博客 post
  `mega-kernel-37-mpk-sm100-mla-2sm`（weight 38）并更新 `mega-kernel-00-index` 第 37 项。P122-1 新留尾巴见
  §8（CUTLASS atom 未展开、`B2B` 未细究、`docs/35` §9 其余尾巴未覆盖）。
- 此前：**P121-1 完成（1/1）**。补掉 `docs/34`（P120-1）**§6「局限」明留的 MLA attention 段尾巴**
  （KV gather 两变体 / decode / reduce / prefill 吸收版 / prefill_tp8 未吸收版），见
  `docs/35-mpk-sm100-mla-family.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。三条 headline：
  **(A)** KV gather（扁平化 ILP gather `:119-157`）与 prefill（吸收版 715 行 / 未吸收 TP8 版 486 行、
  单 stage 受常驻 SMEM 预算约束、dual-dispatch gate）是**真接线**的 device 身体；**(B)** 旧
  `mla_decode_sm100`/`mla_reduce_sm100` 被 PR 651 MTP 版顶替——`mla_decode_sm100_task_impl` 只有
  standalone 单测 caller、`mla_reduce_sm100_task_impl` **零调用者**，生产走
  `mla_mtp_decode_sm100_task_impl<false>`/`mla_mtp_reduce_sm100_task_impl<256>`（`task_register.cc:4168/4248`），
  是「已实现未接线」的第三种形态（同族内被后继覆盖）；**(C)** prefill 两条身体由 DeepSeek-V3 builder
  `_use_prefill = mbt >= 32`（`builder.py:245`）选路。判据**不贡献任何 C 判据**。配套博客 post
  `mega-kernel-36-mpk-sm100-mla`（weight 37）并更新 `mega-kernel-00-index` 第 36 项。P121-1 新留尾巴见
  §9（头号：`mla_sm100_2sm.cuh` 2SM 协作版未逐行）。
- 此前：**P120-1 完成（1/1）**。把 `docs/32`（P118-1）**§7 明留、`docs/33` §11 明确「未处理」的两条
  尾巴**一起走完——**身体 B**（`linear_fp8_1d2d_sm100.cuh` 的 CuTe 主循环）与 **`mla_mtp_tpN_main` 主体**。
  见 `docs/34-mpk-sm100-cute-tasks.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。headline：**(A)** 身体 B
  是「CuTe 高层」第二实现（727 行四段流水、`PipedSharedStorageWithSF`），**生产未接线**（无调用者、pybind
  `linear_fp8_1d2d_sm100` 实走身体 A）；**(B)** MLA-MTP decode 家族（单卡 748 行 + TP2/4/8，`NUM_HEADS`
  128/64/32/16）把完整 MLA attention+reduce 压进一个 device task，揭示**常驻共驻的三类 device 适配**
  （TMEM alloc 用 warp0 / 命名屏障 `bar.sync 12,128` / 256→128 线程守卫 + dual-dispatch）；**(C)** 生产 8 个
  `TaskType`（269/270/287–292）、前端 `persistent_kernel.py:1778/1812`、DeepSeek-V3 builder；TP 是对 ferret
  standalone 的机械 port。判据**不贡献任何 C 判据**。配套博客 post `mega-kernel-35-mpk-sm100-cute`
  （weight 36）并更新 `mega-kernel-00-index` 第 35 项；新增一条「已知坑」（常驻 kernel 内 TMEM alloc/dealloc
  必须同一 warp）。
- 此前：**P119-1 完成（1/1）**。补齐 `docs/32`（P118-1）**§7「局限」明留的「未展开的更深层」**
  （`fp8_group_gemm_sm100.cuh` 的 UMMA 主循环 / UTCCP 序列 / epilogue scatter 未逐行）——把 SM100
  block-scaled FP8 MoE group-GEMM 的 device 身体逐段打开，见 `docs/33-mpk-fp8-group-gemm.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**）。headline：**(A)** 1 个 CTA 跑 **3 条 warp 特化流水**
  （DMA/SCALE/MMA/epilogue + 6 `mbarrier` + 2 `NamedBarrier`）：DMA 的 TMA+手写 swizzle `cp.async`
  （routing 检查）、MMA 的 `UTCCP`+**4×`SM100_MMA_MXF8F6F4_SS`**（描述符 `__shfl` 广播）、SCALE 的
  **fp32→UE8M0→warp 转置**（含 `FIX 2026-04-22` 竞态修复）、epilogue 的 **TMEM→BF16→routing scatter**；
  **(B)** TMEM `free` 不 `release_allocation_lock`——C2 常驻对 device task 的直接让步；**(C)** 生产布线
  `register_moe_fp8_sm100_task`→`TASK_MOE_{W13,W2}_FP8_SM100=248/249`（`(6,1)` / `MMA_N=16` /
  `num_ab_stages=8` / `expert_offset=bid.x`）；**(D)** 三个本地 helper「有定义无调用者」。判据**不贡献
  任何 C 判据**（同 `docs/29`–`docs/32`）。配套博客 post `mega-kernel-34-mpk-fp8-group-gemm`（weight 35）
  并更新 `mega-kernel-00-index` 第 34 项；并把两条坑补进「已知坑」。
- 此前：**P118-1 完成（1/1）**。补齐 `docs/31`（P117-1）**§10「局限」明留的两条尾巴**
  （`linear_fp8_sm100` 家族内部 / SM100 task 的生产登记未复核）——把 SM100 block-scaled FP8 线性
  task 的 device 身体与生产布线打开，见 `docs/32-mpk-sm100-linear-tasks.md`（快照 `f9eb70c`，
  **纯源码走读、无新实验**）。headline：**(A) `linear_fp8_sm100` 家族实为 10 文件、两套 device 身体**
  （A：手写描述符+裸 PTX `tcgen05…block_scale` 的四角色 warp 特化；B：CuTe 高层实现）——公开 pybind
  `linear_fp8_1d2d_sm100` 实走身体 A，**身体 B 只有 `#include`、无调用者**；**(B) 生产布线**：op→
  `TASK_LINEAR_FP8_SM100=276`→`register_linear_fp8_sm100_task`，`kNumStages=25`/`kNumMulticast=1`/
  `GemmType::Normal`/`EpilogueIdentity`，并有**明确标注的 `kNumSMs=1` HACK**（持久化内核里
  `blockIdx.x` 是 worker 索引）⇒「one task = one CTA」在 SM100 的具体适配；BF16 同族
  `linear_sm100_mpk`（`TASK_LINEAR_SM100=253`）。判据**不贡献任何 C 判据**（同 `docs/29`–`docs/31`）。
  配套博客 post `mega-kernel-33-mpk-sm100-linear`（weight 34）并更新 `mega-kernel-00-index` 第 33 项；
  并把两条坑补进「已知坑」。
- 此前：**P117-1 完成（1/1）**。补齐 `docs/30`（P116-1）**§9「局限」明留的两条尾巴**
  （fp8 大 wrapper 内部 / `test_tp*_wrapper.cu` 内部）——把 `sm100_linear_fp8` / `sm100_fp8_moe` /
  `sm100_mla_mtp_decode` 三个「重目录」打开，见 `docs/31-mpk-fp8-wrappers.md`（快照 `f9eb70c`，
  **纯源码走读、无新实验**）。headline：**(a) 补上 `docs/30` 漏掉的共享层** `common/`
  （`sm100_fp8_scale_layout.{h,py}` + `sm100_fp8_runtime_registry.h`）——**UE8M0 packed scale 的
  row-major / DeepGEMM 列主序布局契约**，C++ 与 Python 逐行同构；host 侧把非 DeepGEMM 布局**物化**成
  列主序；**(b) 同一「128K+UE8M0」语义两种载体**（linear host 打包 uint32 vs MoE device 现场转，
  走 UTCCP→TMEM + `MXF8F6F4` 7-warp 特化）；**(c) fp8 MoE 单测 2D grid 手推 `blockIdx.y` 指针**，
  注释自陈 emulating `runtime.cc`；**(d) headline：`sm100_linear_fp8` 引用快照里不存在的头文件
  `linear_fp8_sm100_splitk.cuh`（+ 未定义符号），该扩展不可编译**；**(e) TP standalone** 是对外部
  `v037_tp*` 内核的**机械 port 验证器**（删 PDL、加 thread guard），TP = head 维切分
  （`NUM_HEADS=128/64/32/16`），三个变体各自重定义 `ptx`。判据**不贡献任何 C 判据**（同 `docs/29`/`docs/30`）。
  配套博客 post `mega-kernel-32-mpk-fp8-wrappers`（weight 33）并更新 `mega-kernel-00-index` 第 32 项；
  并把「`sm100_linear_fp8` 缺头文件」补进「已知坑」。
- 此前：**P116-1 完成（1/1）**。补齐 `docs/29`（P115-1）**§7 明留的尾巴**「§3 只精读 `moe_linear`，
  未逐行走读 12 个 blackwell wrapper、`cute/hopper/gemm.cu` 与 `attention_tail_wrapper.cu` 的内部」——
  见 `docs/30-mpk-unit-wrappers.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。headline——**修正
  `docs/29` §2「当前代际对齐」**：blackwell 家族有三类更隐蔽的失配（**同名扩展撞名**
  `runtime_kernel_blackwell`：`sm100_linear` vs `sm100_moe`，符号互异 / **源文件缺失**
  `sm100_mla_prefill_tp8` / **无 setup/pybind 的 standalone 程序** `sm100_mla_mtp_decode/`）；
  并补齐「一个 task = 一颗 kernel」在 SM100 的单 CTA+cluster 形态（`grid(1,1,1)`/224 KB SMEM/
  `SM100_MMA_*`+TMEM）与两条**非 pybind**单测路线（`extern "C" launch` 的 attention_tail、
  CUTLASS standalone 的 `cute/hopper/gemm.cu`）。判据**不贡献任何 C 判据**（同 `docs/29` §6）。
  配套博客 post `mega-kernel-31-mpk-unit-wrappers`（weight 32）并更新 `mega-kernel-00-index` 第 31 项；
  并把「SM100 单测同名扩展撞名」补进「已知坑」。
- 此前：**P115-1 完成（1/1）**。补齐 `docs/28`（P114-1）**§8 明留的尾巴**「未覆盖
  `tests/runtime_python/` 的非 `test_mode` 部分（走 `runtime_kernel` pybind 直接调 device kernel、
  不经 `PersistentKernel` ——『单 kernel 单测』层，其 wrapper 未走读）」——把「一个 device task 当成
  一整颗 kernel 由 host 手写 `<<<>>>` 直接 launch」的这一层打开：见 `docs/29-mpk-unit-tests.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**）。headline——**五套独立 pybind extension；主/hopper
  wrapper 的注册大面积失配（主只注册 3 个而测试调用 14 个、hopper 只注册 1 个）、hopper-moe 与
  blackwell 家族逐条对齐**；`MIRAGE_UNIT_TEST` 宏全仓仅 `setup.py:44` 定义、`moe_linear.cuh:256`
  消费（用 `blockIdx.y` 手推 expert 指针、真实 MPK 不需要）；本层**不贡献任何 C 判据**，只是 C3 范围内
  「单 device task 数值正确」的回归网，且现役 CI（`mpk-qwen-ci`）已不跑它。配套博客 post
  `mega-kernel-30-mpk-unit-tests`（weight 31）并更新 `mega-kernel-00-index` 第 30 项。
- 此前：**P114-1 完成（1/1）**。补齐 `docs/25`（P111-1）**§9 明留的尾巴**「`MirageModelConfig` 的调用方
  （如 `demo/`/`tests/`）未走读」——把「谁来验库里契约」这一层打开：见 `docs/28-mpk-testmode-harness.md`
  （快照 `f9eb70c`，**纯源码走读、无新实验**）。headline——**`test_mode` 把一张常驻 megakernel 图「跑一次
  就退」**：`-DMPK_TEST_MODE` 同时改**编译期**（缺 meta tensor 允许传空指针 `persistent_kernel.py:3413`）与
  **设备端终止协议**（`persistent_kernel.cuh:328` 把生产「跑到 `max_seq_length`/EOS 才 done」的请求完成条件
  替换成 `if(true)`），于是每个测试都是一次「单次 launch → 常驻 kernel 自终止 → 比 PyTorch 参考」的最小端到端。
  走读 `PersistentKernel` 公开 API（82 个 `*_layer()`、`compile:3207`、`__call__:3574`、`finalize:3614`；
  worker/scheduler 划分 `utils.py:25-53`，目标卡 132 SM→**128/16** 对上 `docs/06`/`docs/07`）；把 `docs/08`
  依赖分析器的 fork/join/case 2·3/residual-stripping 逐条配成**可执行规范**（5 个用例），`grid_dim_sweep`
  给 event 数随 grid 的标度见证，§5 列出其余 18 个 test_mode 用例的契约目录；§6 从调用面再证 `docs/15` §7
  「生产前端不调 superoptimizer」；§7 把 `docs/22` §8 判据总映射补成「运行层可运行断言表」。配套博客 post
  `mega-kernel-29-mpk-testmode`（weight 30）并更新 `mega-kernel-00-index` 第 29 项。
- 此前：**P113-1 完成（1/1）**。补齐 `docs/26`（P112-1）**§8 明留的尾巴**「`demo.py` 的权重预处理
  （Phase-2 吸收、gate_up 交织拆分）未走读」——把 megakernel 编译栈**最上游、最不像 kernel** 的一层打开：
  见 `docs/27-mpk-weight-prep.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。headline——**权重预处理层
  用普通 PyTorch 张量变换把「HF checkpoint 物理布局」改写成「builder 逻辑契约布局」**，且有**两条独立
  实现**（离线 `convert.py main()` 写 per-rank safetensors、反量化 BF16；demo 内联 Phase 1/2 保留 FP8，
  仅 `--layers` 时跑）。走读 MLA 权重吸收（`absorb_kv_into_q`）+ V 反吸收融进 `o_proj` + 拆 `q_b_nope`/
  `q_b_pe` 双形态、FP8 重定标（128×128 块、`scale_inv=amax/448`）、gate/up **按 chunk 交织**（非 `cat`）、
  MoE 专家折叠 `w13`/`w2`、16B 对齐与 TP 切分；并指出 **Phase 2 的吸收块因 Phase 1 已删 `kv_key` 而不会
  执行**（同型「已实现未跑」）、**离线/内联两条 shard 规则不一致**（只记录不判对）。判据映射：本层**不贡献
  任何 C 判据**，只是 C3（融合范围）的输入准备。配套博客 post `mega-kernel-28-mpk-weight-prep`（weight 29）
  并更新 `mega-kernel-00-index` 第 28 项。
- 此前：**P112-1 完成（1/1）**。补齐 `docs/25`（P111-1）**§9 明留的尾巴**「未逐行展开每个家族的
  `_build_*`（GLM/Inkling/DeepSeek-V3 的 MoE/MLA 管线只给算子名与文件头架构说明）」——把「家族结构差异
  如何在 builder 层被折平」打开：见 `docs/26-mpk-family-builders.md`（快照 `f9eb70c`，**纯源码走读、
  无新实验**）。headline——**模型家族的结构差异全部在 builder 层被手工「折平」成统一的标准 task 管线**；
  以 DeepSeek-V3（2255 行）为主走读 MLA 权重吸收、FP8 UE8M0 重定标、MoE group GEMM 与 MTP，对照
  GLM-4.6（fused-QKV+bias、共享专家折叠成 expert 160）/ Inkling（无 RoPE 相对位置+短卷积、共享专家
  256/257）/ GPT-OSS（双 KV 流共享页池、gate_up 交织重排、clamped SwiGLU）与 DFlash/Eagle3 草稿
  builder；§6 折叠表给出 `docs/24` 家族 ID 块的**具体前端生产者函数**。判据映射：**家族 builder 层只贡献
  C3 范围**，家族专用 `TaskType` 的调度机制不分家族。配套博客 post `mega-kernel-27-mpk-family-builders`
  （weight 28）并更新 `mega-kernel-00-index` 第 27 项。
- 此前：**P111-1 完成（1/1）**。补齐 `docs/14` §7 尾巴「未走读模型 builder 的完整组装逻辑
  （`models/*/builder.py`）、分布式/张量并行切分（`multigpu.py`）、请求/元数据管理（`mpk.py` 的
  `MPKMetadata`）」——把「谁把 95 个手工 `*_layer()` 拼成一个模型」这一层打开：见
  `docs/25-mpk-model-builder.md`（快照 `f9eb70c`，**纯源码走读、无新实验**）。headline——
  **megakernel 的融合边界由这一层用 Python 手写死**：`models/<家族>/builder.py` 决定层序/手工
  grid/权重交织吸收/**哪些中间量进对称显存（`io_category` cuda vs nvshmem）**，`MPKMetadata`+`MPK`
  决定 serving 模式（六种 `-DMODE_*`）/容量常量（`-DMPK_MAX_*`）/多卡 world-rank/meta tensors 指针。
  §1 注册与发现（`model_registry.py:5-29` + `models/__init__.py:1-17` 自动 import + `MPK.build:471-480`）；
  §2 `MPKMetadata`/`MPK` 生命周期（`compensate_meta_tensors:314-352`、`generate_task_graph:482-489` 带
  `num_gpus/my_gpu_id`、`mode→-DMODE_*:320-334`）；§3 Qwen3 规范样本（`build_from_dict:560-690`、
  `build_layers:288-558`、`new_intermediate_tensors:120-287` 的 `cuda`/`nvshmem` 两路、`use_splitk=
  target_cc==100` 硬件代际分支）；§4 `models/utils.py` 权重交织与 grid helper（含 Qwen3-32B o_proj
  错位 TMA bug 记录）；§5 多卡（builder 内 `//world_size` 切分 + `multigpu.py` 能力查询与
  `auto_select_allreduce_implementation:224-247`；其余三种集合通信仍 `NotImplementedError`）；
  §6 其它家族（GLM/Inkling/GPT-OSS/DeepSeek-V3/DFlash/Eagle3）——**`docs/24` 家族 ID 块的前端生产者**；
  §7 `kv_planner`/动态分片加载器/`online_pinned_runtime`；§8 判据映射（**本层只划 C3 范围、不拿
  C1/C2/C4/C5**，`docs/22` §8）。配套博客 post `mega-kernel-26-mpk-model-builder`（weight 27）并更新
  `mega-kernel-00-index` 第 26 项。
- 此前：**P110-1 完成（1/1）**。补齐 `docs/14` §7 尾巴「未逐行走读全部 95 个任务分支」——把
  MPK 生产路径的 **`TaskType` 目录**打开：见 `docs/24-mpk-task-catalog.md`（快照 `f9eb70c`，
  **纯源码走读、无新实验**）。headline——**`enum TaskType` 按硬件代际/模型家族切成预留 ID 块**
  （0/10 控制、101-121 通用、150-198 Hopper、200-204 scheduler、230-299 SM100、300-349 多卡、
  350-399 Inkling、400-449 GLM、450-499 GPT-OSS），块间留洞=只增不改；§2 列全 116 条命名条目
  （含 12 个范围占位符），§3 坐实 `docs/14` 的「95 分支」= **96 name / 95 赋值 → 92 个 `TaskType`**
  （4 组多 name 共用一 Type），§4 求差得**仅 3 个可执行 `TaskType` 无前端生产者**
  （`TASK_ATTENTION_2`/`TASK_PAGED_ATTENTION_2`/`TASK_ALLREDUCE`）；并坐实 `TASK_ATTN_SM100` 由
  `paged_attention_sm100` 登记（非死码）。配套博客 post
  `mega-kernel-25-mpk-task-catalog`（weight 26）并更新 `mega-kernel-00-index` 第 25 项。
- 此前：**P109-1 完成（1/1）**。把 P0–P108 的 100+ 增量收口成**第二代总纲**
  `docs/23-megakernel-capstone.md`（补齐 `docs/05` 只到 P6、源码侧/实验侧各有手册但**无整专题总纲**的
  缺口）。headline——**五层结论地图 + 六张实测校准表 + 14 条工程坑 + 适用边界 v2**；关键修正
  「**C1/C2/C4/C5 只在运行层拿到**」（`docs/22` §8）；损失三的 ncu 字节证据（= Q）、损失一单 kernel 内
  1.18×、成本侧三笔（寄存器/代码/共驻税）、调度侧 `B(π)`/内存序/环容量、部署侧 Green Contexts vs MPS、
  降级手段天花板，全部带坐标。配套博客 post `mega-kernel-24-capstone`（weight 25）并更新
  `mega-kernel-00-index` 第 24 项。**纯资料汇总（资料分析），本轮无新实验**；与 `docs/05`（v1）、
  `docs/22`（源码侧）、`docs/11`–`docs/13`（实验侧三手册）并列。
- 此前：**P108-1 完成（1/1）**。把 P7–P11、P21、P100–P107（`docs/06`–`docs/08`、`docs/14`–`docs/21`
  共**十一篇** MPK 线上实现源码走读）收口成**一页编译栈总图 + 决策表 + 落差清单**（补齐 `docs/05` 只到
  P6、`docs/11`–`docs/13` 只覆盖实验侧、**源码走读系列无汇总手册**的缺口）。见
  `docs/22-mpk-compiler-manual.md`（快照 `f9eb70c`，CuTe/CUTLASS 本机 `f3fde583`；**纯资料汇总、无
  新实验**）。headline——**源码里有两条链、共用 `KNGraph` 后在 `KNCustomizedOp` 处分叉**：生产路径
  （手工前端 → `annotated_graph` → runtime → **常驻 megakernel，拿 C1–C5**）vs 自动路径
  （`superoptimize` → transpiler → **kernel-per-operator，只到 C3**）。§0 双链分叉；§1 七层坐标表；
  §2 生产路径逐层（`docs/14`/`08`/`06`/`07`：论文 §4.1 五步 ↔ `annotated_graph.cc` 十步 (a)–(j)，
  normalization = 拒绝 case2/3 + LCM 降级；128 worker/16 scheduler；单 task = 一个 CTA）；§3 自动路径
  （`docs/15`/`16`/`18`/`19`/`20`/`21`：一台 superoptimizer 的 CUDA/runtime 模板库/Triton/Blackwell
  四后端）；§4 多卡（`docs/17`：C1 变硬约束）；§5「**同一套惯用法实现两遍**」清单；§6「**已实现未
  启用/未接线**」十一条清单（跨篇汇合）；§7 论文 vs 实现落差清单；§8 C1–C5 逐层总映射（**只有运行层
  拿 C1/C2/C4/C5**）；§9 源码坐标总表；§10/§11 局限与坐标。配套博客 post
  `mega-kernel-23-compiler-manual`（weight 24）并更新 `mega-kernel-00-index` 第 23 项。**与
  `docs/11`/`docs/12`/`docs/13`（实验侧三本手册）并列，源码走读系列首次成册。**
- 此前：**P107-1 完成（1/1）**。补齐 `docs/18`（P104-1）**§8 明留的尾巴**——「Hopper/Blackwell
  只给骨架；Blackwell 的 `2sm_mma` / `launch_kernel_on_cluster` 只标出调用点与行号，未逐行走读其内部」
  （Hopper 的一半已由 `docs/07` 覆盖）。见 `docs/21-mpk-blackwell-transpiler.md`（快照 `f9eb70c`，
  CuTe/CUTLASS 用快照自带 `deps/cutlass`、本机 `f3fde583`；**纯源码走读、无新实验**——Blackwell/SM100
  在目标卡 H100/sm90 不可执行，与 `docs/17` 同口径）。headline——**Blackwell 后端 = Hopper 后端 +
  「把计算单元从 1 CTA 扩到 2 CTA」的一整条链**：2SM TiledMMA（`sm100_make_2sm_trivial_tiled_mma`，
  M∈{128,256}）→ cluster 布局 + leader CTA（`elect_one_cta`）→ multicast TMA（A 沿 N / B 沿 M）→
  **TMEM** 累加（`Allocator2Sm` + `TMemLoadOp`）→ `tcgen05.commit.cta_group::2...multicast::cluster`
  会合（`umma_arrive_multicast_2x1SM`）→ `cutlass::launch_kernel_on_cluster`；对照 Hopper 单 CTA
  wgmma + `PipelineTmaAsync` + 朴素 `<<<>>>`。把这套与 `docs/04` §18–§23 的 sm90 实测连接：cluster /
  `cta_group::2` 从「手写实验/资料里的 PTX」变成「codegen 默认生成的目标」。判据 **C1–C5 只贡献 C3**；
  2SM 路径**论文未提、快照新增**（同 `docs/17`）。配套博客 post `mega-kernel-22-mpk-blackwell-transpiler`
  （weight 23）、更新 `mega-kernel-00-index` 第 22 项。
- 此前：**P106-1 完成（1/1）**。补齐 `docs/19`（P105-1）**§8 明留的尾巴**——「`triton_kernels.py`
  （Python 侧）未走」；顺这条尾巴往上是**一整个平行于 NVCC/CUDA transpiler 的第二条 codegen 后端**。
  见 `docs/20-mpk-triton-backend.md`（快照 `f9eb70c`，纯源码走读无新实验）：把
  `superoptimize(backend="triton")` → `triton_profiler.py`（**每候选图独立子进程** profile）→
  `core.pyx:1461-1469` → `TritonTranspiler`（`src/triton_transpiler/transpile.cc` 579 行 +
  `transpile_tb.cc` 645 行）→ 运行期 `triton_kernels.py`（66 行）逐层走通。headline：**「Triton
  codegen」也实现了两遍**——旧 path `src/kernel/triton_code_gen.cc`（`NewKernelParams` 序列化 +
  `make_block_ptr` + `out_dtype=fp16`，已成**事实死代码**、无 Python 调用者）vs 新 path（直读图、
  `tl.arange`+`program_id` 直译、`tl.dot` 恒 `.to(fp32)`）；另记 `search_c.cc:16` 的 `backend` 参数
  **未被引用**、`TritonTranspilerConfig.target_cc` **已接线未使用**（与 `docs/16` 同型的「已实现未
  启用」）。判据 **C1–C5 只贡献 C3**（换 codegen 目标语言拿不到 C1/C2/C4/C5）；`mpk` 生产前端不调
  superoptimize（接 `docs/15` §7）。配套博客 post `mega-kernel-21-mpk-triton-backend`（weight 22）、
  更新 `mega-kernel-00-index` 第 21 项。
- 此前：**P105-1 完成（1/1）**。补齐 `docs/18`（P104-1）**§8 明留的尾巴**——「`include/mirage/
  transpiler/runtime/**` 是被生成代码调用的库，只当黑盒引用」。见 `docs/19-mpk-runtime-templates.md`
  （快照 `f9eb70c`，纯源码走读无新实验）：把**被生成 CUDA 调用的 23 个 device/thread-block 模板头
  （3017 行）**逐层走通——§0 两层 namespace（`kn::` 整算子兜底 / `tb::` 在 `custom_kernel_N` 内按
  thread 调用）；§1 入口 `runtime.h` + 两个同名 `config.h` 坐标坑；§2 `kn::` 四算子（含 **matmul 直接
  调 cuBLAS**）；§3 `tb::` 拷贝原子（cp.async/TMA/multicast）、epilogue 寄存器链、online-softmax 归约、
  三档 matmul（CuTe / Hopper GMMA cooperative / Blackwell UMMA+TMEM）、mbarrier 流水、profiler、
  `wg_*_regs`；§4 生成代码↔模板四条接口；§5 **headline：同一套硬件惯用法在仓库里实现了两遍**
  （`docs/07` 手写 MPK task vs 本 runtime 库），补 `docs/15` §7 工程落差的运行期证据；§6 论文 §4.2
  对照；§7 只贡献 C3；§8 局限。配套博客 post `mega-kernel-20-mpk-runtime`（weight 21）、更新
  `mega-kernel-00-index` 第 20 项。
- 此前：**P104-1 完成（1/1）**。补掉 `docs/15`（P101-1）**§9 末句明留的尾巴**——「`transpiler`
  只给管线结构（`src/transpiler/*` 约 8 千行），**未逐条读 SASS 级生成逻辑**」（P103-1 §11 记为
  「`docs/15` 另两条尾巴仍未覆盖」之一）。见 `docs/18-mpk-transpiler.md`（快照 `f9eb70c`，
  纯源码走读无新实验）：把一次 `KNGraph.compile()` 从 `kernel.py:396/431` → `core.pyx:1420` →
  `transpile.cc:461/793` → `transpiler.h:128` 的下半段走完——§1 ctor 两段**图归一化**（online-softmax
  重写 + 复合 TB 算子拆原语，收尾断言只留原语）；§2 五步 resolve（元数据/融合/布局/显存，显存**只分配
  不复用**）；§3 `transpile_ugraph` 逐 op 展开成 `_init`+`_execute_mugraph`（非 custom 算子→runtime 模板，
  custom 算子按 cc 分派三后端、Hopper `<<<>>>`/Blackwell cluster launch）；§4 单个 custom op 的
  **schedule/swizzle/memory plan** 三件套（含 **192 寄存器预算决定 accum 进寄存器**、三段调度插
  `SYNCTHREADS`、XOR/shift swizzle）；§5 codegen 形态（双缓冲软件流水；Hopper **warp specialization**
  producer-TMA/consumer-wgmma，`wg_decrease/increase_regs` 被注释）；§6 与论文 §4.2 四项 intra-SM 优化
  逐条对照；§7 判据**只贡献 C3**（跨 customized op 仍 kernel-per-operator）。配套博客 post
  `mega-kernel-19-mpk-transpiler`（weight 20）、更新 `mega-kernel-00-index` 第 19 项。
- 此前：**P103-1 完成（1/1）**。补齐 `docs/15`（P101-1）**§9 明留的尾巴** 与 `docs/16`（P102-1）
  **§6 转引的同一条**——「**多 GPU/NVSHMEM 路径未展开**」。见 `docs/17-mpk-multigpu.md`（快照
  `f9eb70c`，纯源码走读无新实验）：把前面六篇单卡走读之外的**多卡路径**逐段走通——§0 编译栈各层
  「GPU 维」坐标表（MPK 未为多卡另造 runtime）；§1 MPI 定 world/rank + `nvshmemx_init_attr` +
  launch/finalize 跨卡 barrier + **每 rank 各生成 tGraph/各编 launcher**；§2（headline）**多卡被迫
  单 kernel**（`nvshmemx_collective_launch` 不支持 split worker/scheduler，`persistent_kernel.cuh:1988-1990`）
  ——多卡下 C1 从优化变硬约束；§3 event id `nvshmem_tag|owner|idx` 编码与「NVSHMEM event 由别的 GPU
  触发」语义；§4 编译期只给 `ALLGATHER_STRIDED_PUT` 打标签、`num_gpus-1` subtask、`trigger_event`
  改写成目标 GPU（`runtime.cc:66-74,129-150,604-639,657-659`），`TILE_ALLREDUCE` 是本地 task；§5
  worker 用 `nvshmem_signal_wait_until` 等、触发分支空操作（信号在 task 内）；§6 两种 AllReduce
  （策略 A `AllGather+Reduce`=论文 §6.5；策略 B `NVLS tile` one-shot pull，Blackwell 手写
  `multimem.ld_reduce` 保共驻，**论文未提**）；§7 TP residual 只 rank0 加；§8 per-rank launcher；
  §9 论文对照表；§10 C1–C5（C3 扩到跨卡算子+通信、C4 扩到设备间）。配套博客 post
  `mega-kernel-18-mpk-multigpu`（weight 19）、更新 `mega-kernel-00-index` 第 18 项。
- 此前：**P102-1 完成（1/1）**。补掉 `docs/15`（P101-1）**§9 明留的尾巴**——「只精读主搜索
  循环；`abstract_expr/`、`range_propagation/`、`symbolic_graph/` 三块未逐行走读其内部算法」。
  见 `docs/16-mpk-search-passes.md`（快照 `f9eb70c`，纯源码走读无新实验）：把主搜索循环之外的**三个
  判断器**逐行走通——§1 `abstract_expr`（C++ 15 节点 + Rust **egg** e-graph：代数律/`sum` 换序/动态
  质因数分解规则 `lib.rs:72-105`、饱和 runner、`get_egraph`/`egg_equiv` FFI；对最终输出建 e-graph、
  候选查子表达式 `search.cc:504-519`，**`constraints` 参数体未用**）；§2 `range_propagation`（区间代数
  `Range`/`RangeSet`+`PropagationPath` 防环/`TBRange`(tensor|block|forloop)、forward/backward/
  multiplicative_interact 逐算子规则、d2s/s2d 网格映射、`range_propagate` 驱动器——**但两个
  `check_range` 首行 `return true`（`irange.cc:1068/1085`）⇒ 已实现未接线、当前不剪枝**）；§3
  `symbolic_graph`（`TensorDimExpr` 树 + `to_z3` 含 log-scaled、`SymbolicMap` 映射矩阵、按语义生成的
  维度约束；**约束可满足性两套引擎：TB 用 z3（`tensor_dim_constraints.cc:38-47`）、KN 用 rule-based
  启发式（`tensor_dim_constraint.cc:88-148`，几乎不拒绝）**）；§4 指出符号路径三处「已实现未接线」
  （`instantiate_symbolic_graph` 首行 `return false` `search.cc:836-842`、`generate_kernel_graphs_symbolic`
  **无调用者**）并汇总「哪一步真剪枝」；§5 论文对照 + 判据（三块都只贡献 C3）。配套博客 post
  `mega-kernel-17-mpk-search-passes`（weight 18）、更新 `mega-kernel-00-index` 第 17 项。
- 此前：**P101-1 完成（1/1）**。补掉 `docs/14`（P100-1）**§7 明留的尾巴**——「未覆盖自动
  （superoptimizer）路径」。见 `docs/15-mpk-superoptimizer.md`（快照 `f9eb70c`，纯源码走读无新实验）：
  把论文 §4.1「operator decomposition」与 §4.2「thread-block 级 superoptimization」在源码里逐段走通——
  `KNGraph.superoptimize()`（`kernel.py:543`）→ `cython_search`（`search_c.cc:21`）→
  `KernelGraphGenerator::generate_next_operator`（`search.cc:129-392`，K1/K2/K3 + B1/B2 三层 DFS、
  三横切剪枝）→ `ProbabilisticVerifier` 指纹验证 → `transpile()`（`transpile.cc:792`，resolve/
  schedule/codegen 串）→ `kernel.py:606-692` 扫 `pipeline_stages × num_warp_groups` + profile 选优。
  §6 与论文 §4.1/§4.2 逐条对照；§7 回填 `docs/07` §0「superoptimizer vs 手写并存」并指出 `mpk` 生产
  前端**不调** `superoptimize`（`models/`/`mpk/` 无命中，只有 `tests/`/`demo/`/`benchmark/` 调），
  与论文「自动分解」形成明确工程落差；§8 判据映射（只定 C3 的范围）。配套博客 post
  `mega-kernel-16-mpk-superoptimizer`（weight 17）、更新 `mega-kernel-00-index` 第 16 项。
- 此前：**P100-1 完成（1/1）**。补齐 `docs/06`/`docs/07`/`docs/08` 三篇**共同的上游空白**——
  「一张算子图是谁、怎么建出来的」。见 `docs/14-mpk-frontend.md`（快照 `f9eb70c`，纯源码走读无新实验）：
  MPK 线上前端不是「高层算子自动分解」的编译器，而是一套**手工作者 task API**——每个 `*_layer()` 建一张
  bGraph（`customized()`）+ 一条 `task_config`（`register_task()`），95 个手工分支登记 95 个 TaskType；
  `annotated_graph.cc:278-283` 只认 `KN_CUSTOMIZED_OP`，前端 `new_input` 的 `input_map`/`output_map` 正是
  `docs/08` 依赖分析的输入。§5 与论文 §4.1/§4.2 对照，§6 判据映射（前端只定 C3 的范围）。配套博客 post
  `mega-kernel-15-mpk-frontend`（weight 16）、更新 `mega-kernel-00-index` 第 15 项。
- 此前：**P99-1 完成（1/1）**。补掉 P98-1（`docs/12` §10 末句）明留的尾巴——把
  P71-1–P82-1（`docs/04-experiments.md` §59–§70）这一串**端到端调度骨架 + worker 数据面微基准**
  收口成第三本手册 `docs/13-endtoend-dataplane.md`（**纯资料汇总、无新增 H100 实验**；数字全部引自
  `docs/04` §59–§70 及 `harness/*.out.txt`）。七节：§0 手册在链条里的位置 → §1 数据面能到多快 /
  要多少 producer（`PS*` 随访存重量反向 `MD=2MB→PS*≈4` 对上 MPK 1:32；ILP 把数据面 926→2840；
  §61 到 3219=峰值 96%、TMA 3075 省寄存器、真峰值下描述符写读降 3–7%）→ §2 端到端夹层（fold ±1%
  免费、环+producer −43.9%@G1040、最优「少而大」85%；§63 producer 成本纯每批 ~2.1 µs、扇出归因
  证伪）→ §3 worker 侧载体（每 block prologue plain 0.35/TMA 0.69 µs；cp.async≡plain、TMA +0.3
  µs/block 是其专属；深环无收益）→ §4 寄存器×占用率×T×ILP（同占用率 32 vs ≥40 regs 1.70×、
  `__maxnreg__` 连续斜坡、饱和区只认合量 `O·T`）→ §5 决策表 → §6 回到 MPK/DeepGEMM → §7 局限。
  配套博客 post `mega-kernel-14-endtoend-dataplane`（weight 15）并更新 `mega-kernel-00-index` 目录
  第 14 项。**至此 `docs/11`/`docs/12`/`docs/13` 三本手册合拢「调度机制 / 共驻税 / 端到端+数据面」
  闭环，`docs/04` 再无未并入手册的微基准节。**
- 此前：**P98-1 完成（1/1）**。把 P83–P97（`docs/04-experiments.md` §71–§85）这一长串
  **数据面 / 异构共驻微基准**收口成一张**成本手册** `docs/12-megakernel-coresidency.md`——回答
  「多 task 类型共驻一个常驻 grid 时 SM 资源怎么分、共驻税多大、哪些旋钮自由」。**纯资料汇总、无新增
  H100 实验**；数字全部引自 `docs/04` §71–§85 及 `harness/*.out.txt`。十节（§0 第二层成本 → §1 通用
  旋钮 `O·T` 非万能 → §2 共驻大体加和 ±13%、唯一系统裂缝 `scalar` 发射受限 ~13%、`l2res` 超加和
  1.111、sparse 低掺杂毒化 0.855 → §3 税住在 SM **间**非 SM 内、分区不能绕开 → §4 SM 分配器：常数
  速率谬误两坐标、in-situ 收回崩盘、多起点追平 oracle ~1%、噪声地板 ~1% → §5 一阶旋钮 = 每 SM 在飞
  warp `Wsm`（税 0.772→0.995 随 `Wsm` 收回）→ §6 支配量 `Wsm·U`、寄存器只经占用率、四重占用率封顶
  → §7 块粒度 `T` 次级、落点在发射侧 LSU 背压非 L2 → §8 决策表 → §9 回到 MPK/DeepGEMM（共享 SM +
  设备内调度是物理结论、「少而重」而非「多而杂」）→ §10 局限）。配套博客 post
  `mega-kernel-13-coresidency`（weight 14）并更新 `mega-kernel-00-index` 目录。**P98-1 尾部**：
  `docs/04` §59–§70（数据面到 HBM 峰值 / 端到端骨架 / worker 侧复杂度 / 寄存器×占用率解耦）尚未并入
  任何手册——留作下一轮可选增量。
- 此前：**P97-1 完成（1/1）**。补掉 P96-1（§84）**§84.4 明留的尾巴**——「bootstrap CI 基于
  同一进程内 S=30 单次实验，**不含跨进程 / 跨 device 的系统差**」。复用 §82/§83/§84 同一内核
  （`src/hetero_sm_tax_T.cu`，新增 `xproc <S> <seed>` 子命令，不动 `run`/`bench`/`robust`）：把
  「换进程、换卡」当**新的独立重复**——4 块空闲 H100 各起 6 个独立进程（**共 24 进程**）、每进程
  `S=30` 且**每 round 内 Fisher-Yates 打乱访问顺序**、打印原始样本，外部按**进程**做 bootstrap。见
  `docs/04-experiments.md` **§85**，脚本 `harness/hetero_sm_tax_T_xproc.sh` + 汇总器
  `harness/hetero_sm_tax_T_xproc.py`、原始输出 `harness/hetero_sm_tax_T_xproc.out.txt`（目标卡
  H100/sm90a、device 3/4/6/7 各 6 进程、nvcc 13.2、`U=8,R=32,map=bind`、`Wsm=2048`）。核心结论：
  **(1) 跨进程 / 跨 device 下 +4.3% 稳健**（混合 `T512/T128` 24 进程 median 比 = **0.9598**，吞吐
  **+4.2%**，跨进程 CI95 `[0.9574,0.9629]` 不含 1.0，与 §84 同进程 `0.9586/[0.9528,0.9627]` 重合，
  **24/24 进程逐个显著**）；**(2) 方差分解**——混合 betweenStd（0.33–0.63%）< withinStd（1.0–1.3%）的
  一半 ⇒ 换进程/换卡系统差小于同进程噪声；**(3) 例外 `l2res16M` 单模式** betweenStd 2.21% > withinStd
  1.57%，L2 驻留负载有真实卡间差（device 7 绝对值偏高 ~4%）；**(4) 设备间小 offset**（四卡比值全 <1，
  device 3 低 ~0.9%，卡间 std ~0.4%，与 ~1% 绝对效应百分位差同量级 ⇒ 比绝对效应须固定卡）。判据
  **C1–C5 全不满足**。**本轮新增实验（复用内核 + 新子命令 + 新脚本 + 汇总器）**。
- 此前：**P96-1 完成（1/1）**。补掉 P95-1（§83）**§83.5 明留的两条尾巴**——「混合 T 趋势只有
  ~4%、只取 3 次 run，在噪声地板附近」与「未做 MemoryWorkloadAnalysis 延迟分解」。复用 §82/§83
  同一内核（`src/hetero_sm_tax_T.cu`，新增 `robust` 子命令），7 配置各 **S=30 次单采样 + round-robin
  交错** + bootstrap 95% CI 判定趋势真伪，并用 ncu 每 warp **绝对**停顿周期做延迟分解。见
  `docs/04-experiments.md` **§84**，脚本 `harness/hetero_sm_tax_T_robust.sh`、原始输出
  `harness/hetero_sm_tax_T_robust.out.txt`（目标卡 H100/sm90a、device 4、nvcc 13.2）。核心结论：
  **(1) +4.3% 的 T 趋势统计显著**（混合 `T512/T128=0.9586`，CI95 `[0.9528,0.9627]` 不含 1.0；
  `T256/T128=0.9691`），§83.5 的不敢定论源于 min-of-3 + 比值口径 + `T=128` 方差偏大；**(2) 净收益
  全来自发射受限侧**（`scalar4B +2.3%`、`l2res16M −2.6%`，均显著）；**(3) 延迟分解：`T` 不减轻 L2
  延迟**（每 warp `long_scoreboard` 周期随 T 升 6–16%），随 T 单调降的是 **LSU 背压 `lg_throttle`
  （−24%/−43%）**——「少而胖」机制落点是发射受限侧的 LSU 背压、非「减少 L2 停顿争用」；**(4)**
  混合同 `Wsm` 下 DRAM ~74%、L2 命中仅 11% ⇒ 是 DRAM + LSU 题。判据 C1–C5 全不满足。
  **本轮新增实验（复用内核 + 新子命令 + 新脚本）**。
- 此前：**P95-1 完成（1/1）**。补掉 P94-1（§82）**§82.7 明留的尾巴**「§82.6『少而胖更好』
  未做 ncu 归因」——复用 §82 同一内核，用 ncu warp stall-reason 直方图 + L2/L1 命中 + issue/warp-active
  在固定 `Wsm=2048` 下扫 `T`，对混合 `scalar+l2res` + 两单模式 + `coal+scalar` 控制组归因。见
  `docs/04-experiments.md` **§83**，脚本 `harness/hetero_sm_tax_T_attr.sh`、原始输出
  `harness/hetero_sm_tax_T_attr.out.txt`（目标卡 H100/sm90a、device 6、nvcc 13.2、
  `ncu --replay-mode application`）。核心结论：**(1)** 先复现混合 T 趋势（4330→4449→4518 GB/s，+4.3%，
  与 §82.6 +4.7% 同形，但效应在噪声地板附近）；**(2) 聚合计数否掉 §82.6 的机制假设**——「L2 延迟停顿」
  `long_scoreboard` 与「调度槽争用」`not_selected` **都不随 `T` 变**（`l2res` solo 50.16%→50.26% /
  9.08%→9.04%），唯一随 T 单向变的是 `warps_active`（升）与 `lg_throttle`（LSU 背压，降）；
  **(3) 按模式分解**：T 依赖住在 `scalar4B`（发射/LSU 受限）这一半（solo +3.4%、`lg_throttle` 17.8→11.0），
  L2 驻留的 `l2res` 这一半 T-不变 ⇒ **§82.6 把原因放错了半边**（不是 L2 延迟侧，而是发射/LSU 侧）。
  **踩坑记入「已知坑」**（ncu application-replay 放大程序自报计时 ~2×）。**本轮新增实验（复用内核、无新 src）**。
- 此前：**P94-1 完成（1/1）**。补掉 P93-1（§81）**§81.7 明留的两条尾巴**——(tail 1)「`T=256`
  固定、`R<32` 未测」；(tail 2)「§C 固定 `R=128` 时 `occ` 随 `U` 变（ILP 与占用率仍部分纠缠）」。
  把**每块线程数 `T`** 变成第三旋钮，令 `R=32` 让 `U≤16` 都 0 spill、`occ` 不随 `U` 变，做出
  「`Wsm` 钉 2048」的**纯 ILP 曲线**，并补 `R∈{24,28}` 下界角点。见 `docs/04-experiments.md` **§82**，
  源码 `src/hetero_sm_tax_T.cu`、脚本 `harness/hetero_sm_tax_T.sh`、原始输出
  `harness/hetero_sm_tax_T.out.txt`（目标卡 H100/sm90a、自动选最闲 device（落 device 6）、nvcc 13.2、
  `numSM=132`、逻辑总读 256 MiB、`U∈{1,2,4,8,16}`/`R∈{24,28,32,48,64,96,128}`/`T∈{64,128,256,512,1024}`）。
  核心结论：**(1) §B 四重封顶**——占用率 = `min(线程 2048/T, 寄存器 65536/(regs·T), 块 32, SMEM)`，
  **`R≤32` 的 `occ` 全等于线程上限 `2048/T`**，压 `R` 到 24 一格不涨；每块 8 KiB 静态归约数组在
  `T=64` 时成第四上限（occ=24）；**(2) §C 纯 ILP**——固定 `Wsm=2048` 扫 `U`，`meas/E` 0.81→≈1.00、
  天花板 `U≈8`、`U=16` 平顶而非 §81 的回落 ⇒ **§81 `U=16` 回落主要是占用率耦合**；**(3) §D**——固定
  `Wsm=2048` 扫 `T∈{128,256,512}`（coal+scalar）±2% 不变 ⇒ **§79「税只认 `Wsm`」对发射受限×访存受限
  混合也成立**；**(4) §E `R<32`**——`R=24/28` 与 `R=32` 同 occ 却唯一引入 spill，mixed 单调更差 ⇒
  `R=32` 是「不 spill 的最小预算」下界，`__maxnreg__` 下限 24；**(5) §F**——`scalar+l2res` 固定 `Wsm`
  下 mixed 随 `T` 单调升（+4.7%、超加和）⇒ L2 延迟受限混合**不**认 `Wsm`、少而胖更好。判据
  **C1–C5 全不满足**（与 §71–§81 同档）。**本轮新增实验**。
- 此前：**P93-1 完成（1/1）**。补掉 P92-1（§80）**§80.7 明留的两条尾巴**——(tail 1)
  「`U ≤ 8` 且 `__maxnreg__(32)` 固定（放宽寄存器预算后的 `U∈{16,…}` 未测）」；(tail 2)
  「`U×Wsm` 替代性只用 `coal+scalar` 一个混合、`T` 固定 256」。把**寄存器预算 `R`（`__maxnreg__(R)`，
  模板参数）**与**每线程 ILP `U`** 都当独立旋钮做 `U×R` 二维；每个 `(U,R)` 用 **occupancy API**
  实测 `occ` 后令 `O=occ`、`grid=numSM·occ`（保证 bind 全图常驻、不丢 work）。见
  `docs/04-experiments.md` **§81**，源码 `src/hetero_sm_tax_reg_ilp.cu`、脚本
  `harness/hetero_sm_tax_reg_ilp.sh`、原始输出 `harness/hetero_sm_tax_reg_ilp.out.txt`（目标卡
  H100/sm90a、自动选最闲 device（落 device 6）、nvcc 13.2、`numSM=132`、逻辑总读 256 MiB、
  `U∈{1,2,4,8,16,32}`×`R∈{32,48,64,96,128}`、`T=256`）。核心结论：**(1) §A** `U=16/32` 与 `R`
  高低都不动数值；**(2) §B 资源表**：自然需求随 `U` 增（30/28/40/54/84/≥128），`R` 是上限非设定值
  （复现 §69），只有 `<32,32>` 真 spill，`U≤16` 在 `R=32` 都 0 spill 且满占用 `occ=8`；加 `R` 唯一
  效果是砍 `occ`；**(3) §C（ILP 天花板）**：固定 `R=128` 扫 `U`，`mixed` 1.46×、`meas/E` 0.810→1.014，
  **天花板在 `U≈8`**（0.973），`U=16` 回落 0.942，但等 `Wsm=512` 下 `U=32`> `U=16` ⇒ 支配量是
  「每 SM 在飞 load 总量」；**(4) §D（解耦）**：固定 `U` 抬 `R` **不变好**（`U=8` 的 `R=32` 最优，
  `meas/E=1.000`）⇒ 寄存器预算**只经占用率起作用**，最优是「不 spill 的最小 `R`」，「给寄存器」只在
  服务更高 `U` 时值得；**(5) §E（统一性）**：第二混合 `scalar+l2res` **不服从**该机制（`meas/E`
  0.949–1.117 波动、可超加和，对上 §72）⇒ 机制量只对**两端皆 HBM/发射受限**的混合成立；
  **(6) §F ncu**：`(U,R)` 三档 DRAM 字节几乎相同、活跃 warp 按 `occ` 比例降（97.8→49.3→24.8%）。
  判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§80 同档）。**本轮新增实验**。
- 此前：**P92-1 完成（1/1）**。补掉 P91-1（§79）**§79.6 明留的两条尾巴**——(tail 1)
  「`meas/E` 的 spread 混入 `E` 的拆分依赖，故比值需与绝对量并读」；(tail 2)「`U=4`/
  `__maxnreg__(32)` 单一（每线程 ILP 未当独立变量）」。把「共驻税」从**比值 `meas/E`**换成
  **绝对量**（mixed 绝对 GB/s + 固定参考归一 `taxRef=E_ref/mixed`），并把**每线程 ILP `U`**
  变成独立旋钮（kernel 模板化在 `U` 上，固定 `Wsm=1024/T256/O4` 扫 `U∈{1,2,4,8}`），再测
  `U×Wsm` 替代性。见 `docs/04-experiments.md` **§80**，源码 `src/hetero_sm_tax_ilp.cu`、脚本
  `harness/hetero_sm_tax_ilp.sh`、原始输出 `harness/hetero_sm_tax_ilp.out.txt`（目标卡 H100/sm90a、
  自动选最闲 device（落 device 6）、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、寄存器 `__ldcs` uint4 +
  `__maxnreg__(32)`；`pat_hetero_u<1/2/4/8>` = 30/28/32/32 regs、四档全 0 spill）。四访存模式/
  逻辑字节/XOR 参考/误差扩散 `assign[]`/SM 分配/per-SM 原子子段游标完全复用 §71–§79。核心结论：
  **(1) §A `U=1/2/4/8` 逐位对 host 参考**（`U` 不动数值）；**(2) §B 单模式随 `U`**：`coalesced` `U=2`
  即饱和（1.07×）、`sparse8x` 撞物理墙（1.00×），而 `scalar4B` `U=8` **2.22×**、`l2res16M` **2.23×**；
  **(3) §C（口径，headline 一）**：`Wsm=2048` 内 4 拆分的 spread——`scalar+l2res` **绝对 mixed spread
  1.084 而 `meas/E` spread 1.205**（多出的 ~1.11× 是 `E_self` 漂移 5417→4146，**§79.3 判断证实**）；
  反之 `coal+scalar` **绝对 spread 1.188（大 `T` 快 1.19×）而 `meas/E` spread 仅 1.017**（比值**掩盖**
  绝对依赖）⇒ **「税（比值）与拆分无关」≠「混合绝对吞吐与拆分无关」**；**(4) §D（ILP，headline 二）
  固定 `Wsm=1024` 只动 `U=1→8`，共驻税同样大幅收回**——`coal+scalar` `meas/E` **0.708→0.982**、绝对
  **1238.8→2880.0（2.32×）**，`scalar+l2res` **0.587→0.992**、**1325.0→4985.1（3.76×）**；`coal+sparse`
  对 `U` 无感 ⇒ 支配量是**每 SM 在飞 load 量 `Wsm·U`**（`Wsm` 只是其一因子）；**(5) §E `U`×`Wsm`
  替代性**：等 `Wsm·U` 组合塌到 ±2–10%（`4096`：2048/U2=2443、1024/U4=2543、512/U8=2690）、高 `U`
  低 `O` 略优 ⇒ 把 §60/§68/§70 的「ILP↔占用率↔线程数」替代性**推广到「每线程 ILP ↔ 每 SM warp 数」**；
  **(6) §F ncu**：`U=1/4/8` 的 DRAM 字节/扇区/活跃 warp（~49%）逐项相同、唯 `inst/cycle` 0.06→0.04
  ⇒ `U` 只改在飞深度（ILP）。判据 **C1–C5 全不满足**（纯数据面微基准，与 §71–§79 同档）。**本轮新增
  实验**；P92-1 新留尾巴见 §80.7（`U≤8` 且 `__maxnreg__(32)` 固定、放宽寄存器预算后的 `U∈{16,…}`
  未测；`U×Wsm` 替代性只用 `coal+scalar` 一个混合；`Wsm=2048` 内拆分只到 `T128/O16`；合成访问；
  只测 H100；未与 DeepGEMM SM100 对跑）。**「SM 分区/块粒度/每 SM warp 数/(T,O) 拆分/每线程 ILP」
  五旋钮链条闭合。**
- 此前：**P91-1 完成（1/1）**。补掉 P90-1（§78）**§78.8 明留的头号尾巴**（原话）「`Wsm=2048`
  只经 `O=8/T=256`，未补 `O=2/T=1024`，故『warp 数 ≠ block 数』在 2048 处**未完全解耦**」：把
  `Wsm ∈ {512,1024,2048}` **每个都铺全可行 `(T,O)` 拆分**（`O=Wsm/T`，`Wsm=2048` 补到 `T128/O16`），
  **唯一新变量 = 同一 `Wsm` 下的拆分**。见 `docs/04-experiments.md` **§79**，源码
  `src/hetero_sm_warps_split.cu`、脚本 `harness/hetero_sm_warps_split.sh`、原始输出
  `harness/hetero_sm_warps_split.out.txt`（目标卡 H100/sm90a、自动选最闲 device（落 device 6）、
  nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`U=4`、`__maxnreg__(32)`；`pat_hetero` 32 regs/0 spill/
  8200 B smem）。核心结论：
  - **(1) §A `bind` 在 `O=1/2/4/8/16` 逐位对 host 参考**（`O=16/T128` 也 132/132 SM、每 SM 恰 16 块、
    无 spill）——per-SM 原子子段游标在高 `O` 数值安全；
  - **(2) §B**：`coalesced`/`sparse8x` 在每 `Wsm` 内不随拆分变（合量 `Wsm` 支配）；但 `scalar4B` 同
    `Wsm=2048` 内极差 **1.35×**（偏大 `T`）、`l2res16M` 极差 **1.61×**（偏小 `T`）——「合量支配」只对
    HBM 带宽受限模式严格成立（与 §71.7/§83 一脉）；
  - **(3) §C（headline）`coal+scalar` 的税在每 `Wsm` 内对拆分稳定（spread ≤1.8%）**：`Wsm=512`
    `[0.817 0.825 0.830]`、`Wsm=1024` `[0.901 0.913 0.912]`、**`Wsm=2048` 四拆分
    `[0.978 0.969 0.987 0.987]`**——**§78.8 疑虑证否**：§78 的「税随 `Wsm` 收回」是「`Wsm` 一提上来就在
    所有拆分上同时收回」、非单点巧合（`bind ≈ interleave` 每列差 ≤1% 保持）；
  - **(4) §F ncu**：`Wsm=2048` 三拆分（`T1024/O2`、`T512/O4`、`T256/O8`）扇区数**逐字节相同
    （33579744）**、活跃 warp 三档 **~99%**、DRAM 仅 `O=8` +5%（块级 prologue）——税的回收藏在「把每 SM
    在飞 warp 补满」这一个机制量里、与块粒度无关（对上 §78.7）。
  判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P90 全完成，本轮为其新增的实验增量**；
  P91-1 新留尾巴见 §79.6（`meas/E` 的 spread 混入 `E` 的拆分依赖、合成访问、单 `U=4`/`__maxnreg__(32)`、
  `Wsm` 只到 2048、只测 H100、未与 DeepGEMM SM100 对跑）。**至此 §71–§79 的「SM 分区 / 块粒度 / 每 SM
  warp 数 / `(T,O)` 拆分」四旋钮链条闭合。**
- 此前：**P90-1 完成（1/1）**。补掉 P89-1（§77）**§77.8 明留的尾巴**（原话）「`O` 只取
  `1/2/4` 且固定 `O·T=1024` → **未单独扫『每 SM warp 数（`O·T`）』**，故『加 warp 摊』的完整验证仍
  缺口」：把**每 SM 总线程 / warp 数 `Wsm=O·T`** 变成**唯一新变量**，在同一 SM 角色绑定、同一逻辑工作、
  同一模式份额、同一 SM 分配器下扫它。见 `docs/04-experiments.md` **§78**，源码
  `src/hetero_sm_warps.cu`、脚本 `harness/hetero_sm_warps.sh`、原始输出
  `harness/hetero_sm_warps.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、nvcc 13.2、
  `numSM=132`、逻辑总读 1 GiB、载体 = 寄存器 `__ldcs` uint4 + `U=4`、`__maxnreg__(32)`；`pat_hetero`
  32 regs/0 spill/8200 B smem）。四模式/逻辑字节/XOR 参考/误差扩散 `assign[]`/SM 分配/per-SM 原子子段
  游标完全复用 §71–§77；**唯一新变量 = `Wsm`**（固定 `T=256` 扫 `O=1/2/4/8`，另补 `Wsm=1024` 的 `T`
  拆分 `O2/T512`、`O1/T1024`）。核心结论：
  - **(1) §A `bind` 在 `O=1/2/4/8` 逐位对 host 参考**（`O=8` 占 132/132 SM、每 SM 恰 8 块）；
  - **(2) §B 单模式**：`sparse8x` 是唯一「`Wsm` 不是旋钮」的模式（700–715 GB/s，物理流量墙）；`scalar4B`
    最吃 warp（1166→2882）；`l2res16M` 偏爱「多而小的块」（`O4/T256` 20104 > `O1/T1024` 14595）；
  - **(3) §C/§C2（headline）税随 `Wsm` 收回**：`coal+scalar` 的 `bind` `meas/E` **0.772（256）→ 0.825
    （512）→ 0.913（1024）→ 0.995（2048）**——**§72.8 缓解 (a)「加 warp 摊」兑现，当且仅当「加 warp」=
    提高 `Wsm`**（块粒度/SM 分区都不行）；固定 `Wsm=1024` 内 `(T,O)` 拆分不变（0.898/0.912/0.913，复现
    §77）；`bind ≈ interleave` 每列差 ≤1%。**绝对量口径**：`Wsm>1024` 后实测已近平台（`E−实测` 缺口
    ~240→~13），故加 warp 主要是让混合**可预测**、并消掉分区调参痛点；
  - **(4) §E 高 `Wsm` 下分区近乎「可通吃」**：`coal+scalar` 扫 scalar SM 份额，低 `Wsm=256` 是「两端好、
    中间深谷」（`sc=12` 掉到 **0.147**）；高 `Wsm=2048` 基本压平 ~1.0（`sc≥36` 全 0.92–1.06）；
  - **(5) §F ncu**：`O1/T256` vs `O8/T256` 物理字节几乎不变（DRAM 1.07→1.13 GB、扇区 +0.07%），活跃
    warp **12.50% → 98.97%**——`Wsm` 的机制就是补满在飞 warp，与 §70/§80–§82 同源。
  **对 §73「税是全局」的精确限定**：税不住在块边界（§77）/ SM 分区（§73），但**是每 SM warp 预算的
  函数**，`Wsm` 到满占用率（2048）即基本消失；数据面饱和与共驻税是同一个「在飞 warp 预算」旋钮。
  判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P89 全完成，本轮为其新增的实验增量**；
  P90-1 新留尾巴见 §78.8（`Wsm=2048` 只经 `O=8/T=256`、未补 `O=2/T=1024` → 「warp 数 ≠ block 数」在
  2048 处未完全解耦；合成访问、单 `U=4`/`__maxnreg__(32)`、只测 H100、未与 DeepGEMM SM100 对跑）。
- 此前：**P89-1 完成（1/1）**。补掉 §73–§76 **反复出现的明文尾巴**（§73.8/§74.9/§75.9/§76.10）
  「`bind` 用 `O=1`（每 SM 一 block），未测每 SM 多块的角色绑定」：把 §73 的「按 SM 角色绑定」**正确
  推广到任意 `O`**（per-SM 原子子段游标），唯一新变量 = 每 SM block 数 `O`（保持 `O·T=1024`）。见
  `docs/04-experiments.md` **§77**，源码 `src/hetero_sm_bind_occ.cu`、脚本
  `harness/hetero_sm_bind_occ.sh`、原始输出 `harness/hetero_sm_bind_occ.out.txt`（目标卡 H100/sm90a、
  自动选最闲 device（落 device 6）、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`U=4`、`__maxnreg__(32)`；
  `pat_hetero` 32 regs/0 spill/8200 B smem）。核心结论：**(1) §A `bind` 的 `O=1/2/4` 逐位对 host 参考**
  （`O=4` placement 132/132 SM、每 SM 恰 4 块）；**(2) §B 三档 `O` 下 `coalesced`/`sparse`/`scalar` 近平**
  （复现「吞吐由合量 `O·T` 定」），**唯一例外 `l2res16M` `O=1→4` 快 1.33×**（偏爱「多而小」block）；
  **(3) §C（headline）`bind ≈ interleave` 在三档 `O` 下逐行不变（差 ≤1%）**、`coal+scalar` 的 ~9–13% 税
  在 `O>1` 下**不消失** ⇒ **§73「共驻税是全局、非 SM 内发射槽」对每 SM block 数稳健**，§72.8 缓解 (a)
  「加 warp 摊」在「固定 `O·T`、只改块粒度」意义上**不兑现**；`bind_bal` 仍全面劣于 `bind`/`interleave`
  且与 `O` 无关；**(4) §E** `coal+scalar` 扫 SM 份额 `O=1` vs `O=4` 曲线几乎重合（内点最优 `sc≈84–96`）；
  **(5) §F ncu** `bind` 在 `O=1` vs `O=4` 硬件计数逐项相同（DRAM 1.07 GB、扇区 33567072、活跃 warp
  49.99%/49.20%）⇒ 税不随 `O` 变是**真·物理不变**。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题
  此前 P0–P88 全完成，本轮为其新增的实验增量**；P89-1 新留尾巴见 §77.8（`O` 只 1/2/4 且固定 `O·T=1024`
  → **未单独扫每 SM warp 数**、「加 warp 摊」完整验证仍缺口；合成访问、单 `U=4`/`__maxnreg__(32)`、
  只测 H100、未测 `O>4`）。**踩坑已记入「已知坑」**（per-SM 原子游标必须 `tid==0` + `__syncthreads()`
  广播，否则每线程各领号 → 大多数线程 `BP=0` 空转 → 吞吐虚高（超 HBM 上限）+ 数值错）。
- 此前：**P88-1 完成（1/1）**。补掉 P87-1（§75）**§75.9 头号尾巴**「`bind_insitu` 是**局部爬山**
  （`equal4` 两次运行收敛到不同分配、因 `l2res16M` 的 L2 驻留对运行序敏感）」：把 §75 的**单起点爬山**
  升级为**多起点爬山**（`bind_global`，map=5）+ **oracle 穷举**，四问——Q1 可复现性 / Q2 多起点 /
  Q3 oracle / Q4 噪声地板。见 `docs/04-experiments.md` **§76**，源码 `src/hetero_sm_global.cu`、脚本
  `harness/hetero_sm_global.sh`、原始输出 `harness/hetero_sm_global.out.txt`（目标卡 H100/sm90a、
  **自动选最闲 device（落 device 6）**、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`T=1024,O=1`、
  `__maxnreg__(32)`、`U=4`；`pat_hetero` 32 regs/0 spill/8192 B smem，与 §73–§75 同）。四模式/五既有
  映射**完全复用 §71–§75**，**唯一新变量 = 起点集合 + oracle 穷举**。核心结论：
  **(1) §C（Q1，headline 一）单起点 `bind_insitu` 不可复现**：同一 `bind_opt` 起点重复 `N=8` 次，
  `equal4` 收敛到**双模态**——`[8,84,20,20]`（~1465，`meas/E≈0.78`）或 `[25,3x,3x,4x]`（~1899–2035，
  `meas/E≈1.03–1.09`），**spread=1.398**；`coal+sparse` 稳定（1.007）。§75 的 `equal4=0.789` 正是低频支；
  **(2) §D（Q2，headline 二）多起点收回局部最优**：`equal4` 0.789→**1.039**（追平 `interleave` 1.042）、
  `coal+scalar` 0.945→0.979（仍差 `E` ~2%）；**(3) §E（Q3，headline 三）多起点追平 oracle 到 ~1%**：
  四两模式混合穷举 `n_p=1..131`，`bG/org`=**1.001/0.996/1.000/0.991**，`org_n` 与 §75 曲线峰位一致；
  **`coal+scalar` 的 `bind_global`（2937）≈ oracle（2935）但都 `<E`（3004）~2.3%**（全局发射税仍在）；
  **(4) §F（Q4）噪声地板 ≈1.0%**（同分配重复 `R=8`：`spread=1.010`），故 §C 的 1.398 是**优化器路径
  效应、非测量噪声**，但多起点 best-of-M 也被 ~1% 噪声影响（§D 1942 vs §F2 2032）——给「一个好解」非
  「确定最优」；**§F2 轨迹**显示 `bind_opt` 起点反收敛到较差支（1883）、偏置/随机起点到 2032；**§F ncu**
  两分配器硬件计数逐项相同。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P87 全完成，
  本轮为其新增的实验增量**；P88-1 新留尾巴见 §76.10（多起点是有限起点集 + 局部爬山、oracle 只两模式、
  `bind` O=1、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100）。**踩坑已记入「已知坑」**（评测角色
  绑定分配前须验 `sum==numSM` 且活跃模式 ≥1，否则丢 work → 吞吐虚高）。
- 此前：**P87-1 完成（1/1）**。补掉 P86-1（§74）**§74.9 明文尾巴**「`bind_opt` 的速率曲线是
  **单跑**测得，**未做 in-situ/迭代不动点**」：把 SM 分配器的速率来源从**单跑**换成**当前并发下实测**——
  (a) `bind_insitu`（map=4）从 §74 `bind_opt`（单跑曲线）出发、用**实测整机 makespan** 做**局部爬山**
  （挪 `step` 个 SM、改善 >0.5% 才接受、`step=16→…→1`）；(b) **条件 in-situ 速率曲线**（两模式扫 `n_p`、
  设备侧 `clock64` 量并发下各分区实测速率）。见 `docs/04-experiments.md` **§75**，源码
  `src/hetero_sm_insitu.cu`、脚本 `harness/hetero_sm_insitu.sh`、原始输出
  `harness/hetero_sm_insitu.out.txt`（目标卡 H100/sm90a、自动选最闲 device、nvcc 13.2、`numSM=132`、
  逻辑总读 1 GiB、`T=1024,O=1`、`__maxnreg__(32)`、`U=4`；`pat_hetero` 32 regs/0 spill/8192 B smem）。
  四模式/四映射**完全复用 §71–§74**，**唯一新变量 = 速率来源**。核心结论：
  **(1) §C headline 一——in-situ 完全收回 §74 崩盘**：`coal+sparse` 由 `bind_opt` **0.664→1.024**（追平
  `interleave` 1.032）、`coal+l2res` 0.886→**1.168**、`scalar+l2res` 0.983→**1.030**（均反超 interleave），
  **§74 崩盘确系「单跑曲线把『给已饱和模式加 SM』的边际价值估高」**；**(2) §C headline 二——两类东西
  in-situ 也修不了**：`coal+scalar` 的 ~5% 全局发射税反不回（0.952≈bind_opt 0.953）、`equal4` 的角色绑定
  结构上限（0.789 仍输 interleave 1.043，任何静态分区不如共享 SM+交错）；**(3) §D headline 三——条件
  in-situ 曲线即病因**：单跑 `solo_coal` 一路涨、in-situ `rate_coal` 到 `n≈64` 仅 627–1082，in-situ
  `rate_sparse` 随 coal 扩张单调降（704→…→558），overall 峰值 `n_coal≈32`，而单跑贪心选 `n_coal=9`；
  **(4) §E/§F** `bind_insitu` 把 `[9,123]` 纠回 `[33,99]`、15 次评测使崩盘案 766.5→1172.1；**§G ncu**
  两分配器硬件计数逐项相同（差纯并发时序）。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前
  P0–P86 全完成，本轮为其新增的实验增量**；P87-1 新留尾巴见 §75.9（`bind_insitu` 局部爬山、`equal4`
  对运行序敏感、条件曲线只 2 条两模式对、`bind` O=1、合成访问、未对 SM100）。
- 此前：**P86-1 完成（1/1）**。补掉 P85-1（§73）**§73.8 明文尾巴**「`bind_bal` 用**常数**每 SM
  速率模型（`r1_p`），未用「SM 数→速率」曲线，故其差**部分**是模型粗糙而非分区本身；未做按曲线最优
  配平」：把 §73 的**常数斜率** SM 分配（`bind` ∝ 份额 / `bind_bal` ∝ `f_p/r1_p`）对照**实测「SM 数→
  分区速率」曲线 + 贪心 makespan 最小化**（`bind_opt`）。见 `docs/04-experiments.md` **§74**，源码
  `src/hetero_sm_curve.cu`、脚本 `harness/hetero_sm_curve.sh`、原始输出
  `harness/hetero_sm_curve.out.txt`（目标卡 H100/sm90a、**自动选最闲 device（落 device 6）**、nvcc 13.2、
  `numSM=132`、逻辑总读 1 GiB、`T=1024,O=1`、`__maxnreg__(32)`、`U=4`；`pat_hetero` 32 regs/0 spill/
  8192 B smem）。四模式/三既有映射**完全复用 §71–§73**，**唯一新变量 = SM 分配器**。核心结论：
  **(1) §A/§B** 正确性逐位对 host 参考、单模式基线复现 §73；**(2) §B2（新数据）实测速率曲线
  `rate_p(n)`**——饱和膝各不相同：**`sparse8x` 在 `n≈32` 即压平**（rate 701.6→708.9、每 SM 21.9→5.37），
  `coalesced`/`scalar4B` 到 `n=96` 才压平，`l2res16M` 全程近线性；**(3) §C（headline）曲线配平
  「一半成立、一半是新谬误」**：`bind_opt` 在**近线性混合**上收回并反超——`coal+scalar` `bind_bal`
  0.865→**0.947**（**反超 `interleave` 0.909**）、`scalar+l2res` 0.969→**0.987**；但在**含 HBM 饱和
  模式（`sparse`）**的混合上**比常数斜率还差**——`coal+sparse` 0.936→**0.664**（`bind_opt` 分
  `[coal=9,sparse=123]`，把 SM 全喂给已饱和的 sparse、饿死 coal），`equal4` 0.757。**根因：单跑速率
  曲线把「加 SM 到已饱和模式」的边际价值估高了——「常数速率谬误」的另一种坐标**；**(4) §E** `bind_opt`
  选点 `[38,94]` 落在 §73.6 内点最优（sc≈96）、**§F ncu** `bind` 与 `bind_opt` 硬件计数逐项相同。
  判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P85 全完成，本轮为其新增的实验增量**；
  P86-1 新留尾巴见 §74.9（**单跑曲线未做 in-situ/迭代不动点**——本节核心发现即它不够、曲线只采 14 点、
  `bind` 用 O=1、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100）。
- 此前：**P85-1 完成（1/1）**。补掉 P84-1（§72）**§72.8 明文尾巴**「共驻映射是『按 block 交错』，
  非真实 megakernel 的『按 task 类型分区/SM 角色绑定』」：把 §72 的同一组模式从「按 block 交错」
  （共享 SM）换成「**按 SM 角色绑定**」（每 SM 只跑一种模式，`mode = mode_of_sm[smid]`）、**保持逻辑
  工作份额不变**，判定共驻税/超加和是 **SM 内还是 SM 间**现象。见 `docs/04-experiments.md` **§73**，
  源码 `src/hetero_sm_bind.cu`、脚本 `harness/hetero_sm_bind.sh`、原始输出
  `harness/hetero_sm_bind.out.txt`（目标卡 H100/sm90a、自动选最闲 device（落 device 6）、nvcc 13.2、
  `numSM=132`、逻辑总读 1 GiB、`__maxnreg__(32)`、`U=4`、min-of-3；`pat_hetero` 32 regs、0 spill；
  `bind` 用 O=1/T=1024 每 SM 一 block 独占区、无原子，与 §72 同 32 warp/SM）。核心结论：
  **(1) §A 两种映射折叠 `sink` 均逐位对 host 参考、132/132 SM 均匀；§B 单模式基线复现 §71/§72；
  (2) §C headline 一 `bind ≈ interleave`**——五混合 `meas/E` 逐行几乎重合（差 ≤1.1%）：
  equal4 1.039/1.033、coal+scalar **0.902/0.898**、coal+sparse 1.030/1.021、coal+l2res **1.112/1.113**、
  scalar+l2res 0.789/0.792 ⇒ **共驻税与 `l2res` 超加和是全局（SM 间/HBM/L2）现象、不是 SM 内发射槽
  共享**，**证伪 §72.8 缓解方向 (a)**；§F ncu 显示 `bind` 与 `interleave` 的 DRAM 字节/sector/活跃 warp/
  `inst/cycle` **逐项相同**；**(3) §C headline 二 `bind_bal`（SM 数 ∝ 1/r1）反而更差**（equal4 0.753、
  coal+l2res 0.854）——`sparse`/`l2res` 低 SM 数即饱和 HBM/L2，每 SM 速率非线性，「按常数 1/r1 分 SM」
  前提不成立；**(4) §E** coal+scalar 扫 scalar 的 SM 份额，内点最优 sc≈96 仍仅 **0.935**（最优分区也
  收不回税）。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P84 全完成，本轮为其新增的
  实验增量**；P85-1 新留尾巴见 §73.8（`bind` O=1 未测每 SM 多块、`bind_bal` 常数速率模型未按曲线
  最优配平、合成访问、单 `U=4`/`__maxnreg__(32)`、未对 SM100）。
- 此前：**P84-1 完成（1/1）**。补掉 P83-1（§71）**§71.7 明文尾巴**「四模式此前都是**单独整机
  跑、未测同一 grid 内多模式共驻**」：把四种访存模式塞进**同一个常驻 grid**（每 block 按 `assign[]`
  选模式、host 误差扩散法**交错**分配保跨 SM 共驻），复检 §71.7 headline 四「一个常驻 kernel 内
  task 类型越杂、越难一套调度参数通吃」。见 `docs/04-experiments.md` **§72**，源码
  `src/hetero_access_mix.cu`、脚本 `harness/hetero_access_mix.sh`、原始输出
  `harness/hetero_access_mix.out.txt`（目标卡 H100/sm90a、自动选最闲 device（落 device 2）、
  nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、`__maxnreg__(32)`、`U=4`、min-of-3；`pat_mix` 32 regs、0 spill）。
  核心结论：**(1) §A 混合折叠 `sink` 逐位对 host 参考**，§B 单模式基线复现 §71；**(2) §C（headline
  一）共驻大体加和**——九种混合 measured/时间切片期望 ∈ **[0.867, 1.111]**（±13%），**唯一系统性
  裂缝 = `scalar4B`（发射受限）**：含 scalar 的混合恒低 ~13% 且与 `(T,O)` 正交；`coal+l2res` 反而
  **超加和 1.111×**（l2res 是不争 HBM 的白吃者）；**(3) §D（headline 二）异质性税分两类**——
  `equal4`/`coal+sparse` ~1.03（无税，被瓶颈模式主导）、`coal+scalar` **0.871**（18% 不可补的争用
  税）⇒「难一套参数通吃」的正确形态是**层内（访存模式间）可通吃、发射受限 × 访存受限不可通吃**；
  **(4) §E（headline 三）稀疏低掺杂「毒化」**——`sparse=10–30%` 时 `meas/exp` 0.855–0.915（物理未到
  墙），`≥50%` 回加和（物理稳在 2.77–2.92 TB/s≈HBM 墙）；**(5) ncu** 物理 DRAM 字节 = `Σ f_p·phys_p`
  **逐字节精确**（1.50×/2.50×）。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P83
  全完成，本轮为其新增的实验增量**；P84-1 新留尾巴见 §72.8（共驻映射按 block 交错、非真实按 task
  类型分区/SM 角色绑定；`O·T≤2048`；未扫非 8 倍数 strided/非对齐；未对 SM100）。
- 此前：**P83-1 完成（1/1）**。补掉 P82-1（§70）**§70.7 明文尾巴**「只测**连续/规则**的
  `__ldcs` uint4 流，未扫 strided / 非对齐 / **L2 命中**等访存模式」：把「数据面只认『每 SM 总线程
  `O·T`』」这条合量拿到**四种访存模式**上复检，并把 §70 的二维面推广到**访问模式**维度。见
  `docs/04-experiments.md` **§71**，源码 `src/access_pattern_surface.cu`、脚本
  `harness/access_pattern_surface.sh`、原始输出 `harness/access_pattern_surface.out.txt`（目标卡
  H100/sm90a、自动选最闲 device（落 device 2）、nvcc 13.2、`numSM=132`、逻辑总读 1 GiB、
  `__maxnreg__(32)`、`U=4`、min-of-3）。核心结论：**(1) §A 四模式折叠 `sink` 逐位对 host 参考**、
  无 spill；**(2) §B 合量普适但天花板差一个量级**——coalesced/scalar/l2res 都在 `O·T≈1024` 入平台
  （模式无关的通用旋钮），峰值 3056 / 2597 / 23448 GB/s，`sparse8x` 逻辑仅 713（等效物理 2848，
  撞 HBM wall）；**(3) §C（headline）合量的裂缝**——同一 `O·T=1024` 下 `coalesced` 极差 1.04×（复现
  §70），但 `l2res16M` **1.47×**（L2/延迟受限要「多而小」的 block，`T=256/O=4` 22385 vs
  `T=1024/O=1` 15217），`scalar4B` 偏大 `T` 1.15×；**(4) §D ncu** 把三种非 HBM 约束分开——`sparse`
  物理流量 4×（DRAM 4.29 GB）、`scalar` DRAM 与 coalesced 相同（纯指令发射）、`l2res` 几无 DRAM
  （2.04 MB、L2 延迟）。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P82 全完成，
  本轮为其新增的实验增量**；P83-1 新留尾巴见 §71.7（合成访问、未测 `O·T>2048`、未扫非 8 倍数
  strided/非对齐、未对 SM100）。**P80–P83 的「寄存器预算 × 占用率 × 线程数 × 访存模式」成本侧链条闭合。**
- 此前：**P82-1 完成（1/1）**。补掉 P81-1（§69）**§69.7 明文尾巴**「未测 `T` 与 `O` **同时大于 1**
  的二维面（`O·T` 联合钉『每 SM 总线程』）」：在**固定每线程寄存器预算** `R`（`__maxnreg__(R)`）下
  同时扫每块线程数 `T` 与每 SM 块数 `O`，检验数据面是否只由**每 SM 总线程 `O·T`** 这个**合量**决定。
  见 `docs/04-experiments.md` **§70**，源码 `src/t_occ_surface.cu`、脚本 `harness/t_occ_surface.sh`、
  原始输出 `harness/t_occ_surface.out.txt`（目标卡 H100/sm90a、自动选最闲 device（落 device 2）、
  nvcc 13.2、`numSM=132`、载体 = 寄存器 `__ldcs` uint4 + `U=4`、`grid=numSM·O`、总读 2 GiB、min-of-3）。
  核心结论：**(1) §E headline 合量假设在饱和区成立**——`O·T=1024` 的 12 个配置（`R=32/40/44` ×
  `T=128..1024`）全落 **3045–3198 GB/s（极差 1.05×）**、`O·T=2048` 的 4 个落 **2995–3181（1.06×）**，
  吞吐只由 `O·T` 定、与拆分和 32 vs 40/44 regs 无关；**(2) 未饱和区寄存器预算仍是先决条件**——`R=32`
  相对 `R=40/44` 在 `O·T=128/256/512` 落后 **2.16×/1.76×/1.24×**，寄存器预算决定「达到饱和所需的最小
  `O·T`」（40/44 在 `O·T≈512` 近饱和、32 要到 ~1024–2048，约差一倍）；**(3) 同 `O·T` 组内 `T` 越大略快**
  （块级 prologue 省，`O·T=2048` 的 `R=32` 行 T128/O16 2995 → T1024/O2 3181）；**(4) §F 两个天花板**
  ——线程面 2048（仅 `R=32` 可达）、寄存器面受分配粒度影响（`R=32→2048`、`R=40→1536`、`R=44→1280`；
  `R=64` 被自然需求封顶成 44 regs）。判据 **C1–C5 全不满足**（纯数据面微基准）。**本专题此前 P0–P81
  全完成，本轮为其新增的实验增量**；P82-1 新留尾巴见 §70.7（`U` 固定且与寄存器耦合、无法测 `O·T>2048`、
  只测连续流、未对 SM100）。**至此 P80/P81/P82 的「寄存器预算 × 占用率 × 线程数」二维面链条闭合。**
- 此前：**P81-1 完成（1/1）**。补掉 P80-1（§68）**§68.7 明留的两条尾巴**「占用率固定为 **block
  粒度**（`T=256`，**未**单独扫 `T`）」「**未用 `__maxnreg__`** 单独压寄存器；`MINB` 只是 ptxas 的
  ceiling/hint」：用 CUDA 12.4+ 的 **`__maxnreg__(R)`** 把每线程寄存器上限设成**精确**值，与占用率
  （`grid=numSM·O` 与每块线程数 `T` 各自独立）解耦。见 `docs/04-experiments.md` **§69**，源码
  `src/maxnreg_decouple.cu`、脚本 `harness/maxnreg_decouple.sh`、原始输出
  `harness/maxnreg_decouple.out.txt`（目标卡 H100/sm90a、**自动选最闲 device**、nvcc 13.2、`numSM=132`、
  载体 = 寄存器 `__ldcs` uint4 + `U=4`、总读 2 GiB、`reps=3`、min-of-3）。核心结论：
  **(1) §B `__maxnreg__` 是精确上限**——`R≤44` 时 `numRegs≡R`（32/36/40/44）、`R≥48` 时被自然需求
  44 封顶；§68 的 **1.70×**（32 vs ≥40）如实复现，但分界**不是二元跳变而是连续斜坡**（32→36→40 =
  1668.6→**2137.1**→2841.8 GB/s，36-reg 已收回 ~41% 的差），**修正 §68.3 的二元措辞**；
  **(2) §C** 精确 `__maxnreg__` 与 `__launch_bounds__` 在匹配 `numRegs`（32/40/44）下**逐位等价
  （<0.3%）**——决定吞吐的是寄存器**数**、不是属性；**(3) §D（headline）每线程寄存器预算与每 SM warp
  数是替代品**——`O=1` 固定 `R=32` 时 `T=128→1024` 令吞吐 **899→3136 GB/s（3.49×）**，`R=32,T=1024`
  ≈ `R=64,T=1024`（3136 vs 3191，仅差 1.7%），把 §68 的 `(寄存器预算, 占用率 O)` 替代性**推广到 `T`
  维度**（都供「每 SM 在飞 load 总量 `≈O·T·U`（×每线程 MLP）」），饱和点在 HBM 面 ~3200 GB/s；
  **(4) §E** 精确 `R` 下扫 `O` 复现 §68 §C。判据 **C1–C5 全不满足**（纯数据面微基准）。**已给 ROADMAP
  「已知坑」补 `__maxnreg__` 语义注记**（只是上限、≥自然需求即失效；与 `__launch_bounds__` 互斥）。
  **本专题此前 P0–P80 全完成，本轮为其新增的实验增量。**
- 此前：**P80-1 完成（1/1）**。补掉 P79-1（§67）**§67.4 明留的尾巴**「`MINB` 同时改寄存器预算与
  占用率、**未解耦**；ILP 主导是排除法推断、非单变量隔离」：用**两个独立旋钮**把二者拆开——占用率由
  **`grid=numSM·O`** 钉死（`resident=min(O,reg-limit)`），寄存器预算由 **`__launch_bounds__(256,MINB)`**
  钉死（`MINB∈{8,6,4,2}`）。见 `docs/04-experiments.md` **§68**，源码 `src/reg_occ_decouple.cu`、
  脚本 `harness/reg_occ_decouple.sh`、原始输出 `harness/reg_occ_decouple.out.txt`（目标卡 H100/sm90a、
  自动选最闲 device、nvcc 13.2、`T=256`、`numSM=132`、总读 2 GiB、min-of-3）。核心结论：
  **(1) §B 单变量隔离（headline）**——占用率钉死到 `O=1`（每 SM 恰 1 块）后，`MINB=8`（32 regs）
  **1668.7 GB/s** vs `MINB≥6`（40–44 regs）**2847.2** = **1.70×**（O=2 1.21×、O=4 1.03×）⇒
  数据面认的是**每线程在飞 load 数（寄存器预算 ILP）**、不是占用率本身，把 §67 的推断升级为隔离实验；
  **(2) §C**——32-reg 预算靠占用率 O=1→4 补回 1.83×、≥40-reg 预算 O=2 即饱和 ⇒ 寄存器预算与占用率
  是**替代品**（给出 §60/§61/§72 的机理）；**(3) §E**——`cp.async` 单缓冲自然只用 31–32 regs、对
  `MINB` **完全免疫**（同 `MINB=8` 下寄存器载体掉、它一点不掉），给 MPK 用 16 B `cp.async` 取 task 添
  运行时理由；**(4) §D**——SMEM-pad 与 grid 两种占用率口径同量级（差 3–6%），但 SMEM-pad 挤 L1 不干净，
  headline 采信 grid 钉法。判据 C1–C5 全不满足（纯数据面微基准）。**本专题此前 P0–P79 全完成，本轮为
  其新增的实验增量。**
- 此前：**P79-1 完成（1/1）**。补掉 P78-1（§66）**§66.9 明留的第一条尾巴**「跨二进制 codegen
  偏移」（plain `numRegs=32/localB=8`、`G≤132` 慢 ~1.6×）：把 `__launch_bounds__(256,MINB)` 的
  `MINB∈{8,6,4}` 当**寄存器预算旋钮**（32/42/64 regs/线程），对 plain / TMA / `cp.async` 单缓冲 /
  深环逐一对跑。见 `docs/04-experiments.md` **§67**，源码沿用 `src/worker_carrier_cpasync_ring.cu`
  （只加 `MINB=6/4` 实例化与 `regs` 子命令）、脚本 `harness/carrier_reg_budget.sh`、原始输出
  `harness/carrier_reg_budget.out.txt`（目标卡 H100/sm90a、自动选最闲 device、nvcc 13.2、`T=256`、
  `U=4`、`NB=2048`、`MD=65536`、min-of-3）。核心结论：**(1) 8 B 溢出 = 32-reg 上限（不是运行期
  分支）**——`MINB=8` 下 plain/TMA 溢出、`cp.async` 不溢出；放宽后 plain 需 `MINB=4`（52 regs）、
  TMA 需 `MINB=6`（40）；**(2) 载体寄存器耐压序 `cp.async` 单缓冲/深环 > TMA > plain 直接读**
  （直接读最耗、SMEM 载体最省）；**(3) 溢出不是慢因**——`localB=0` 的 `ca1` 与 plain 逐点重合、
  同步随 `MINB` 变快（`G=100/132` 提速 1.64×/2.43×、1.59×/2.14×），主导量是**寄存器预算 → 数据面
  ILP**（降占用率 8→4 blk/SM 反把数据面从 931 抬到 2263 GB/s）；**(4) 排序随配置翻转**——TMA 在
  大 `G` 反超，但任一固定 `MINB` 内 §66 结论（`ca1 ≡ plain`、深环更差）都成立。**已修正 ROADMAP
  「已知坑」**（原把溢出归到「运行期 `if(wmode==…)` 分支」的说明是错的）。判据 C1–C4、仍差 C5。
  **本专题此前 P0–P78 全完成，本轮为其新增的实验增量。**
- 此前：**P78-1 完成（1/1）**。补掉 P77-1（§65）**§65.9 明留的尾巴**「**cp.async 是单缓冲**
  （快照形态），**未做 `D` 深环预取**（§42 已量『预取增益全在 D=1→2』）」：把 cp.async 从单缓冲升级成
  **`DCA` 槽软件流水深环**（`cp.async.wait_group (DCA-1)`、尾批 `wait_group 0`、逐槽复用
  `__syncthreads()`），接进 §64/§65 的端到端 worker 阶梯。见 `docs/04-experiments.md` **§66**，源码
  `src/worker_carrier_cpasync_ring.cu`、脚本 `harness/worker_carrier_cpasync_ring.sh`、原始输出
  `harness/worker_carrier_cpasync_ring.out.txt`（目标卡 H100/sm90a、固定空闲 device 3、nvcc 13.2、
  `DW=88`=352 B、`K=32`、TMA `D=2`、`PS=8`、占用率固定 `T=256,MINB=8,U=4`、`NB=2048`、`MD=65536`、
  min-of-3；新增 `WMODE 6=ca-ring`/`7=ca-ring+d`，`WMODE`/`DCA` 均编译期 template）。核心结论：
  (1) **§A 正确性**：单缓冲 + 深环 `DCA=2/4/8` 共 9 档 `sink` 逐位对 host 参考（全 `OK`）；
  (2) **§B/§C/§G（headline）深环没有任何一个点快过单缓冲**——`ca-single ≡ plain`（±1%）；
  `DCA=2` 掉到 ≈ TMA 档（`caD2/plain` 随 `G` 从 1.08 涨到 1.35×）、`DCA=4` 更差（1.23→1.61×）；
  §G（`G=132,chunk=8`）扫 `DCA=1..8` **单调变慢** 1206→1136→1041→893 taskGB/s；
  (3) **§E 两笔净成本**：逐批**槽复用 `__syncthreads()`** + **SMEM 足迹随 `DCA` 线性增长**拉低占用率
  （`DCA=2`→8、`DCA=4`→5、`DCA=8`→2 blk/SM）；
  (4) **含义**：§42 的「预取增益全在 D=1→2」**不迁移**到端到端阶梯——352 B/task 取数已被 HBM 数据面/
  producer 就绪等待藏住，深环无缝可回收，只剩足迹与同步；再次确认 MPK 用 16 B `cp.async` 单缓冲
  （`persistent_kernel.cuh:1063-1078`）是经过取舍的选择。判据 C1–C4、仍差 C5。
  **本专题此前 P0–P77 全完成，本轮为其新增的实验增量。**
- 此前：**P77-1 完成（1/1）**。补掉 §64.9 明留的尾巴「载体是寄存器读（plain）/ TMA，
  **未含 `cp.async`**（§42 的第三种载体）」：把 MPK worker 的**原生**取 task 载体（16 B `cp.async`
  单缓冲进 SMEM，快照 `persistent_kernel.cuh:1063-1078` 的 `cp_async_wait<0>`）加进 §64 的 worker 侧
  复杂度阶梯，在同一骨架、同一占用率（`T=256,MINB=8,U=4`=8 blk/SM）下与 plain / TMA 对跑。见
  `docs/04-experiments.md` **§65**，源码 `src/worker_carrier_cpasync.cu`、脚本
  `harness/worker_carrier_cpasync.sh`、原始输出 `harness/worker_carrier_cpasync.out.txt`
  （目标卡 H100/sm90a、固定空闲 device 3、nvcc 13.2、`DW=88`=352 B、`K=32`、`D=2`、`PS=8`、
  `NB=2048`、`MD=65536`=1 MiB/batch、min-of-3；新增 `wmode 4=cpasync` / `5=cpasync+d`）。核心结论：
  (1) **六档 `sink` 逐位对 host 参考**（全 `OK`）；
  (2) **§B `cp.async ≡ plain`、TMA 随 `G` 掉队**：固定 `NB=2048` 扫 `G`，`ca/plain` **0.99–1.01×**；
  `tma/plain` 从 `G=132` 的 0.969 掉到 `G=1040` 的 **0.813×**；
  (3) **§C（headline）TMA 的每 block prologue 是 TMA 专属**：固定 `chunk=4` 折成 `(full−data)/G`，
  `tma−plain` 恒 **+0.25…+0.32 µs/block**（复现 §64），而 `cp.async−plain` 恒 **−0.04…+0.005 µs/block
  （≈0）**——**cp.async 的 prologue 与 plain 寄存器读同档、不付 TMA 那笔**；
  (4) **§D/§F** 载体差异只在有数据面的快路径上可见，cp.async 逐点 ≈ plain、TMA 掉 5–12%。
  含义：解释 MPK 为何用 16 B `cp.async` 而非 TMA bulk 取 task（setup 轻 ~0.3 µs/block；TMA 的价值在
  载入 task 单体 tile）。判据 C1–C4、仍差 C5。**本专题此前 P0–P76 全完成，本轮为其新增的实验增量。**
- 此前：**P76-1 完成（1/1）**。补掉 §63.9 明留的尾巴「consumer 是 plain-load，不能直接证明
  §62 的 `G` 依赖来自 TMA 路径——需把 TMA+折叠+数据面按 §62 加回」：把 §62 骨架里的 **worker 侧**
  放上一条**复杂度阶梯**（`wmode` 0=plain 寄存器读环 / 1=+TMA 进 SMEM / 2=+HBM 数据面 / 3=仅数据面），
  **同一二进制、同一占用率**（`T=256,MINB=8,U=4`=8 blk/SM）下扫 worker 环数 `G`。见
  `docs/04-experiments.md` **§64**，源码 `src/worker_side_attribution.cu`、脚本
  `harness/worker_side_attribution.sh`、原始输出 `harness/worker_side_attribution.out.txt`
  （目标卡 H100/sm90a、固定空闲 device 3、nvcc 13.2、`DW=88`=352 B、`K=32`、`D=2`、`PS=8`、
  `NB=2048`、`MD=65536`=1 MiB/batch、min-of-3）。核心结论：
  (1) **§B 复现 §62**——固定 `NB`、`chunk` 随 `G` 缩小，`tma+d` 1740@G=1040（§62 报 1768），
  相对 `data-only`（3076）跌 **−43.4%**；`plain`≡`tma`（都被 `PS=8` producer ~31 GB/s 钉住），
  **§63 的「consumer 平」是被 producer 盖住、不等于免费**；
  (2) **§C 拆出闭式（`chunk=4`、`NB=4·G`）**——开销折成**每 block 恒定**：**环消费者 prologue
  ≈ +0.35 µs/block（plain 就付）**、**TMA 额外 ≈ +0.3（tma 合计 ~0.69）µs/block**，跨
  `G=100..1040` 稳定；**§62 的 `G` 依赖 = worker 每 block 一笔固定 prologue**，TMA 翻倍；`chunk=4`
  把 `G=1040` full 从 1740 抬到 **2083**；
  (3) **§D** 固定 `G=264` 扫 `MD`：`MD=0` 时 plain≡tma，有数据面后 **plain 比 tma 快 4–6%**
  （+0.29 µs/block）——**TMA 代价只在快路径可见**。判据骨架 **C1–C4**、仍差 C5。**本专题此前
  P0–P75 全完成，本轮为其新增的实验增量。**
- 此前：**P75-1 完成（1/1）**。补掉 §62.10 明留的尾巴「更细的 producer 服务模型（每环服务
  成本 vs 环数）留待后续」：把 §62 完整骨架里的 **producer 写端单独抽出来**（无 worker/背压/数据面），
  唯一控制变量 = 每个 producer 服务的**环数 `wpp`**（固定每 producer 总批数 `Btot=chunk*wpp`），
  再加回 **plain-load consumer** 对照。见 `docs/04-experiments.md` **§63**，源码
  `src/producer_service_model.cu`、脚本 `harness/producer_service_model.sh`、原始输出
  `harness/producer_service_model.out.txt`（目标卡 H100/sm90a、固定空闲 device 3、nvcc 13.2、
  批=11264 B、`T=256`、min-of-20）。核心结论：
  (1) **无『每环服务成本』**——`PS=32,Btot=64`，`wpp=1→64` 吞吐 158.2→166.0 略升后平台、
  `cyc/batch` 4336→4126（±2.5%）；
  (2) **producer 成本 = 纯每批**（`wpp=1` 扫 `Btot` 的 `cyc/batch` 恒 ~4335–4357、吞吐只随 `Btot`
  爬升=启动摊薄），每批 `fence+st.release`+两次 `__syncthreads` ≈ **~2.1 µs**、单 SM 天花板
  **~5.2 GB/s**（对上 §52 release 税）；
  (3) **§D** `PS=1→32` 线性（5.2→165.1 GB/s）、无跨 producer 争用；
  (4) **§F 对照** `PS=8,NB=2048` 扫 `R=128→1024`：producer-only 恒 42.6–42.7、**+plain-load
  consumer 只降 ~12% 且同样平**（37.3–37.4）——consumer 只加一笔**固定**开销、不随环数放大；
  (5) **修正 §62.7**：producer 环扇出归因**被隔离证伪**，§62 端到端「`G` 越大越慢」只能住在
  **worker 侧**（TMA/折叠/数据面）或其**占用率/共驻**里。判据：写端单 measure，不完整满足 C1–C5。
  **本专题此前 P0–P74 全完成，本轮为其新增的实验增量。**
- 此前：**P74-1 完成（1/1）**。补掉 §61.10 明留的两条尾巴——(a)「纯读微基准是能力上界、非
  端到端吞吐，与 §60 端到端的差即『调度/取 task/折叠的夹层成本』」(b)「争用 writer 是合成写、非真实
  描述符/环协议」。把 §61 的**占用率旋钮**（多 block/SM + `__launch_bounds__(TB,MINB)`）搬回 §60 的
  **完整骨架**，在**同一二进制**内用三档 `mode`（2=data-only / 0=+fold / 1=full）逐层剥开夹层。见
  `docs/04-experiments.md` **§62**，源码 `src/e2e_saturation.cu`、脚本 `harness/e2e_saturation.sh`、
  原始输出 `harness/e2e_saturation.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `DW=88`=352B、`K=32`、`D=2`、`NB=2048`、`MD=65536`=1MiB/batch，全图数据面总读 ≈ 2.0 GiB，峰值
  取 3.35 TB/s）。核心结论：
  (1) **同二进制数据面天花板 ~3156 GB/s = 峰值 94%**；`MINB` 把 regs 压到 32、`T=256` 达 8 blk/SM
  （满占用率）、`U=4` 无 spill；抬占用率两条等价路径 `T=256` 加大 `G`（100→1040：974→3138）或 §60
  的 `T=1024,U=8`（`G=100` 即 3063）——**ILP 与占用率是替代品**；
  (2) **端到端 full 骨架 `G` 越大反而越慢**——`T=256` 路线峰值仅 2058（G=528）、G=1040 降到 1768；
  `T=1024` 路线 G 100→1040 从 2831 跌到 1910；**§60 的「少而大」worker 块最优**（`G=100,chunk=21,
  T=1024,U=8` = **2831 GB/s = 85%**，精确复现 §60 的 2840）；
  (3) **夹层分解（headline）**：`G=1040`：3139→3105→1761（**fold 免费、环+producer −43.9%**）；
  §60 配置：3078→3118→2913（**fold 免费、环+producer −6.6%**）——**§61.10 猜的「取 task/折叠」不是
  瓶颈，真正成本是控制面（环协议+producer）且随 `G` 恶化**；
  (4) **定位到扇出**：G=1040 扫 PS 1→32（343→2500 仍升），固定 G/PS 扫 chunk 2→8 只 +3.6%
  （每 task 开销非瓶颈）——成本来自**「少 producer 轮转服务多 per-worker 环」的扇出**（对上 §58
  `PS*:G≈1:25~1:32`）。**故 megakernel 的设计压力是调度器服务速率，不是纯并行度。**
  判据骨架 C1–C4、仍差 C5。**本专题此前 P0–P73 全完成，本轮为其新增的实验增量。**
- 此前：**P73-1 完成（1/1）**。补掉 §60.9 明留的**唯一尾巴**——**数据面只到 HBM 峰值
  ~85%（`T=1024` 单 block/SM = 50% 占用）、真 100% 峰值下争用仍留一丝未测**。把「数据面带宽
  天花板」从 §60 的**调度器骨架**里抽出来单独量：与调度器解耦成纯带宽微基准、数据面载体新增
  **TMA bulk→SMEM** 与寄存器 `__ldcs` 对跑、争用用**同核并发 writer 块**在峰值配置下扫写速率。
  见 `docs/04-experiments.md` **§61**，源码 `src/dataplane_peak.cu`、脚本
  `harness/dataplane_peak.sh`、原始输出 `harness/dataplane_peak.out.txt`（目标卡 H100/sm90a、
  固定空闲 device 3、每配置总读 ≈ 4 GiB、HBM3 峰值取 3.35 TB/s）。核心结论：
  (1) **数据面到 ~3219 GB/s = HBM3 峰值的 96%**（reg、`T=1024`、2 block/SM、`U=4`），比 §60 的
  2840 快 1.13×——**缺的旋钮是占用率**（§60 卡 1 block/SM），**ILP 与占用率是替代品**（256 线程/SM
  需 `U=4` 到 2718；满占用率下 `U=1→4` 只 3198→3219）；`__launch_bounds__(1024,2)` 把 `U=4` 的
  40 regs 压到 32、1→2 block/SM，`T=1024` 单载体即 3208；
  (2) **TMA bulk 载体到 ~3075（reg 峰值 96%、峰值 92%）、但用更少寄存器**（`DC=2048` 32 regs，
  reg `U=4` 要 40），分块 `<1024` 塌（`DC=512` 只 2118）——TMA 价值在省寄存器；
  (3) **真峰值下「描述符写不争 HBM」变弱**：读带宽随写速率**超线性**降（`write=21.7 GB/s`
  = 读的 0.7% 就读降 7.9%、`78.6` 降 16.7%，HBM bus turnaround），**§59.6/§60.5 的「不争用」是
  读余量的函数而非字节比的函数**（§60 在 85% 有 15% 余量能吸收，§61 在 96% 无余量则 7% 即现）；
  生产描述符流量 ~15 GB/s（~0.5%）落在读降 3–7% 档；F3 另证「SM 槽位挤占」是叠加的一笔。
  判据 **C1**、多算子数据面形态，未含调度故不完整满足 C2–C5（立场是给 §47–§60 骨架配能力表）。
  **本专题此前 P0–P72 全完成，本轮为其新增的实验增量。**
- 此前：**P72-1 完成（1/1）**。补掉 §59.6/§59.9 明留的**唯一尾巴**——**§59 的数据面只到
  ~950 GB/s、未把 HBM 推到饱和，故「描述符写不争 HBM」只在非饱和区间成立**。给 §59 的数据面加
  **ILP 展开因子 `U`**（每线程 `U` 个独立累加器 + `U` 个在飞 `__ldcs`），把数据面推到饱和，再在
  **饱和点重问 §59.6**。见 `docs/04-experiments.md` **§60**，源码 `src/sched_dataplane_sat.cu`、脚本
  `harness/sched_dataplane_sat.sh`、原始输出 `harness/sched_dataplane_sat.out.txt`（目标卡 H100/sm90a、
  固定空闲 device 3、`DW=88`、`K=32`、`G=100`、宽松环、`D=2`）。六 Section 核心结论：
  (1) **ILP 不改数值**（整数异或和可重排，`sink` 逐位对 host 参考，`U=1/4/8/16` 全 `OK`）；
  (2) **§59 的 ~950 GB/s 是「单累加器依赖链」的产物、不是数据面固有**——`T=256` 下 `U=1→2`
  把数据面从 **926→1490 GB/s**（在飞 load 翻倍）、`U=16` 达 **2218**；
  (3) **`T×U` 矩阵**：`T=1024,U=4` / `T=512,U=16` 都到 **~2840 GB/s**（H100 峰值 3.35 TB/s 的 **85%**、
  §45 饱和值 ~2983 的 95%），饱和由「每 block 在飞字节 `T×U`」定；`T=1024,U=16` `too many resources`
  （记 `UNSUP` 跳过）；
  (4) **闭合 §59.6（headline）**：饱和配置下 `mode0`（**无 producer** 天花板）**2792 GB/s** vs
  `mode1`（`PS≥8`）**2763–2837**——**饱和点上加 producer 仍不拖慢数据面**，「不争用」从非饱和区间
  **扩展到饱和区间**；`PS` 唯一作用是喂满（`PS=1` 896 → `PS≥8` 饱和，`PS*≈8`，比 §59 的 4 高，因
  数据面快了 ~3×）；
  (5) **同 `T=1024`、`U=1` 的对照**也到 ~2.4 TB/s（≈2.5×§59），说明 §59 低吞吐是 `T`、`U` 两旋钮
  都小，`U` 的边际在**低 `T` 档**最显著；
  (6) **对 MPK 的意义**：worker 用 TMA bulk（高 MLP）本就不在 §59 低 MLP 区间，4:128 落在
  「task 又重又快」区间（`PS*` 更小）。判据 **C1–C4**、仍差 C5。局限：只到峰值 ~85%（`T=1024`
  占用 50%、再堆 `T×U` 受寄存器/SM 限），真 100% 峰值争用仍留一丝未测。**本专题此前 P0–P71
  全完成，本轮为其新增的实验增量。**
- 此前：**P71-1 完成（1/1）**。补掉 §58.10 明留的尾巴——**§58 的 task 重量旋钮 `CW` 是纯 ALU
  延迟、不产生真实 HBM 流量**。给 §58 的 worker 加一条**真实 HBM 数据面**（每 batch `__ldcs` 流式读
  `MD` 个 `uint4`、绕 L2），把「喂饱 G 个 worker 所需 producer 数 `PS*`」-vs-task 重量曲线从**纯 ALU**
  推进到**访存主导**。见 `docs/04-experiments.md` **§59**，源码 `src/sched_perworker_ring_mem.cu`、脚本
  `harness/sched_perworker_ring_mem.sh`、原始输出 `harness/sched_perworker_ring_mem.out.txt`（目标卡
  H100/sm90a、固定空闲 device 3、`DW=88`、`K=32`、`T=256`、`G=100`）。五 Section 核心结论：
  (1) **数据面不改数值**（`MD` 计入 `sink` 后逐位对 host 参考，全 `OK`）；(2) **`PS*` 随访存重量 `MD`
  单调下降**——`MD=0/8192→PS*≥32≈G`、`MD=512 KB→PS*=16`、**`MD=2 MB→PS*≈4`（`PS*:G≈1:25`）**，与
  §58 纯计算档同族、**对上 MPK 的 4:128=1:32**（「重 task 少 producer」对访存型 task 同样成立）；
  (3) 复现 §58（`CW=2^16→PS*=16`、`CW=2^18→PS*≈4`，总工作量差一档、形状一致）；(4) **两笔重量同向叠加**
  （`CW=65536` 单独 `PS*=16`、叠加 `MD=32768` 后 `PS*=8`）；(5) **控制面不争 HBM**——achieved 数据面
  带宽随 `PS` 只升不降（650→952 GB/s），producer 写描述符 ~5 GB/s 可忽略（控制面瓶颈在 MEMBAR/会合）。
  判据 C1–C4、仍差 C5。局限：数据面只到 ~950 GB/s、未把 HBM 推到饱和，真饱和争用未测。
  **本专题任务清单此前 P0–P70 全完成，本轮为其新增的实验增量。**
- 此前：**P70-1 完成（1/1）**。把 P42–P69（`docs/04-experiments.md` §31–§58）这一长串**调度器
  微基准**收口成一张**可操作设计手册** `docs/11-in-kernel-scheduler-design.md`（纯资料汇总，无新实验），
  配套博客 post `mega-kernel-12-sched-design`（weight 13）并更新目录第 12 项。核心是在 `docs/04` 的
  400 KB 原始实验之外，给出**调度器内部机制层**的综合：§1 六部件参考架构；§3 ready 结构死锁的**两个
  不同来源**（图静态量 `B(π)`/`CAP*=G*=B(π)+1` vs 排队量 `CAP_safe+G≈W_max+1`）与事件驱动 `G*≡1`；
  §4 队列协议三条硬约束（release 发布 / 多生产者连续放行 / 批量 warp 读）；§5 取 task 载体×预算
  （大 K+D=2、预取缝固定长度、数据面认总线程/SM、控制面认 K）；§6 描述符来源（预计算 vs 现场生成
  27×、release MEMBAR 税、端到端环窗主导）；§7 环协议（`D` 被协议锁死、一消费者一环、
  `PS*:G≈1:25~1:32` 对上 MPK 4:128）；§8 成本模型速查表（12 条实测）；§9 设计决策表（14 条）；
  §10 对 MPK/DeepGEMM 逐条印证；§11 局限。
- 此前：**P69-1 完成（1/1）**。补掉 §57.8 明留的**头号尾巴**——真实 MPK 形态是「**少
  producer（scheduler）× 多 per-worker 环**」：一个 producer 顺序/轮转填多条环、每条环仍单生产者单
  消费者（前缀精确），**producer 侧不随 G 膨胀**（§57 的 `pair` 靠 `P==G`、`grid=2G`）。见
  `docs/04-experiments.md` **§58**，源码 `src/sched_perworker_ring.cu`、脚本
  `harness/sched_perworker_ring.sh`、原始输出 `harness/sched_perworker_ring.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`K=32`、`T=256`、min-of-3/5）。
  机制 = **把「环所有权」与 block 解耦**（G 个 per-worker 环 × PS 个 producer 各持连续一段、段内轮转
  填；`grid=PS+G`，`PS==G` 退化为 §57 `pair`），另加**计算重量旋钮 `CW`** 模拟计算型 task。核心结论：
  - **正确性/解锁与 producer 数无关**（§A/§B/§E）：宽松环 `PS×D` 32/32 OK；紧环 `CAPp=4` 下 `D≤4` 全
    OK、`D=8` 全 ABORT（每个 PS 一致）；阈值精确 = per-ring 的 `CAPp≥D`、`PS=4/16` 同形——**§57 的解锁
    是拓扑性质、不是 `P==G` 特有**。
  - **§C 少 producer 是共享瓶颈**：`CW=0` 吞吐 **≈4.2×PS GB/s 精确线性**（单 producer SM 标量描述符
    上限 ~4.2 GB/s，对上 §49），故轻负载必须 ≈G 个 producer（= §57 `pair` 被迫 `P==G` 的物理原因）。
  - **§D headline：task 越重、需要的 producer 越少**——`PS*` 随 `CW` 单调下降：`CW=0→PS*≈G`、
    `2^14→16`、`2^16→8`、`2^18→≈4`；重 task 下 **`PS*:G≈1:25`**，正对上 **MPK 的 4 scheduler SM : 128
    worker SM（1:32）**（§5 行 1157–1164、§6.2 行 1460）。
  - **§F codegen 警告**：本 harness `PS=66`（逻辑=§57 `pair`）测得 153–164 GB/s vs §57 通用实现 ~133 GB/s
    （`ncu`：DRAM/L2 与屏障停顿差异，逻辑工作与 `sink` 相同）——**绝对 `descGB/s` 对编译产物敏感、跨源码
    只可定性对照**，本节比较均在同一源码族内。判据 C1–C4、仍差 C5。**本轮新增实验**；P69-1 新留尾巴
    （§58.10）：`CW` 纯 ALU、`grid≤132` 才全共驻故 `PS≤32`、合成描述符、未与 DeepGEMM SM100 对跑。
- 此前：**P68-1 完成（1/1）**。把 §55.7 / §56.7 **两处都明留、却从未实现**的那条「出路」
  做出来——「**保证『一个 producer ↔ 一个 consumer』的拓扑，或每 worker 独立环**」（= MPK 快照
  `per_worker_queue_len=8192` 的真实形态）。关键观察：**`P == G` 时全局 contig 分配的 `chunk==bp`、
  `own==w`，即每个环恰好一个 producer、一个 consumer**，前缀计数器精确 → `D>1` 应可用；本节把该
  一直缺的配置跑出来。`docs/04-experiments.md` 新增 **§57**，源码 `src/perworker_ring.cu`、脚本
  `harness/perworker_ring.sh`、原始输出 `harness/perworker_ring.out.txt`（目标卡 H100/sm90a、**固定
  空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`K=32`、`T=256`、min-of-3/5；`pair` 用 `P=G=66`、
  `shared` 用 §55/§56 的 `P=32 G=100`，两者 `grid=132`；prefix 协议固定，唯一变量 = worker→batch 映射）。
  核心结论：
  - **§A 宽松环全 OK**；**clean control（同 `P=G=66`）只换映射**：`shared-strided` 90 vs `pair` 130 GB/s
    = **1.45×**（`pair` 去掉了「每个 `tail[p]` 被多 worker 并发 `atom.add.release`」的跨 worker 前缀
    计数器争用）——**每 worker 独立环本身就不只为 `D`，还是吞吐的 1.45×**。
  - **§B（headline）紧环 `CAPp=4`**：**`pair` 在 `D=1..4` 全部 OK（129/131/85）**，而 `shared-strided`
    `D≥2` 全 `ABORT`、`shared-contig` `D=1/2` 静默 `WRONG`——**对齐成「一消费者一环」后 prefix+`D>1`
    从死锁/静默错变为正确**；§55.7/§56.7 出路预测**证实**。
  - **§C 解锁阈值（新闭式）**：`pair` 正确性阈值精确 **`CAPp* = D`**；但 `CAPp==D` 只 ~85 GB/s、
    `CAPp≈2D` 才回 ~130（等号处背压往返暴露）→ **`CAPp* = D` 是下限、实用取 `CAPp≈2D`**（呼应
    §41/§42/§54「环深只需 2 槽」）。
  - **§D/§E `D` 的纯性能增量很小**：`pair` `D=1→2` **+6%**、`D≥2` 饱和——解锁的是**可用性**而非
    「`D` 越大越快」（与 §43/§54 一致）。判据 C1–C4、仍差 C5（与 §47–§56 同档）。**本轮新增实验**；
    P68-1 新留尾巴（§57.8）：未实现「少 producer 轮转填多 per-worker 环」（真实 MPK 形态）、
    `CAPp*=D` 只单配置验证、合成描述符、未与 DeepGEMM SM100 对跑。
- 此前：**P67-1 完成（1/1）**。补掉 P66-1（§55）§55.8 明留的两条尾巴——「**contig 的静默错是
  『共享前缀 `tail` 协议』的性质，不是『连续分配』固有的**——换 per-slot 回收协议后连续分配可正确」
  与「**死锁/错误的具体边界未做闭式扫描**」。把 §55 的**共享前缀**有界环换成**每槽独立回收/就绪**
  （perslot `ready`/`reclaim`；对齐 DeepGEMM Mega Gate ring generation、`scheduler/mega_gate.cuh:13-39`），
  同一骨架 / 同一工作 / 同一分配形态对跑。`docs/04-experiments.md` 新增 **§56**，源码
  `src/perslot_ring.cu`、脚本 `harness/perslot_ring.sh`、原始输出 `harness/perslot_ring.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`P=32 producer + G=100 worker`（grid=132
  全共驻）、`DW=88`=352 B、`K=32`、`T=256`、min-of-3）。核心结论：
  - **§55.8 预测证实**：紧环 `CAPp=16` 下 `perslot + contig` 在 `D=1..8` **全部 OK**（`prefix+contig`
    则全 `WRONG`）——共享前缀 `tail` 换成每槽回收后连续分配**不再静默错**；但 contig 正确档慢
    （~3–3.5 vs strided 17 GB/s）：解锁的是**正确性**不是性能。
  - **§C 边界对齐**：`prefix` 下只有「无 producer 被拆」（`chunk==bp`，`66/66`）才 OK、两种 split 都
    WRONG；`perslot` 下两种 split 都 OK——**错的载体是前缀计数器，不是「对齐」**。
  - **§D 解锁阈值（新闭式，补 §55.8）**：`perslot + contig` 需 `CAPp ≥ D`；`perslot + strided` 需
    `CAPp ≥ bp`（=128、**与 `D` 无关**）；`prefix + strided + D=2` 在 `CAPp≈32` 即 OK。⇒ per-slot 把
    strided 的门**收紧**到「必须恰好回收该槽」——**per-slot 不是万能钥匙**。
  - **§E 协议开销**：宽松环 per-slot 比 prefix 只慢 ~1.3–1.4%（写侧地板掩盖）——代价在**解锁阈值**、
    不在协议本身。判据 C1–C4、仍差 C5（与 §47–§55 同档）。**本轮新增实验。**
- 此前：**P66-1 完成（1/1）**。补掉 P65-1（§54）§54.8 明留的两条尾巴——「worker 取 task 环深固定
  `D=1`（**预取深度是 §41/§42 的独立变量**）」与坑②「**跨步分配 `bb=w+t*Gw` 的 `D>1` 预取跨 producer
  子环边界 → 有界环死锁**」：把 §54 端到端流水里被钉死的「worker 取 task」放开（`D` × 分配形态）。
  `docs/04-experiments.md` 新增 **§55**，源码 `src/worker_prefetch_e2e.cu`、脚本
  `harness/worker_prefetch_e2e.sh`、原始输出 `harness/worker_prefetch_e2e.out.txt`（目标卡 H100/sm90a、
  **固定空闲 device 3**、nvcc 13.2、`P=32 producers + G=100 workers`（grid=132 全共驻）、`DW=88`=352 B、
  `NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`T=256`、min-of-3）。核心结论：**`D` 不是自由的性能旋钮，
  而是被环协议锁死的自由度**——(1) 纯性能上限小且立刻饱和（§D 无生产者 worker 地板 `D=1→2` +19%
  （EDW=0 557.9→663.2）/ +7%（EDW=256 215.8→230.9），`D=2` 精确饱和）；(2) 端到端拿不到（§B 宽松环
  `D=1→2` 仅 +2–3%，被 producer 写侧主导）；(3) **接入动态生产者 + 有界环后语义不可得**（§C 紧环
  `CAPp=16`：strided `D>1` **死锁**（`ABORT`/`got=0`，钉死 §54 坑②）、contig `D=1..8` **静默数值错**
  却"更快"——错误形比死锁形更隐蔽；唯一正确形态是 **strided + D=1**=4.8 GB/s 复现 §54 §C）。§E 机制
  隔离：contig 错在 **per-producer `tail` 是「前缀计数器」**——worker/producer 边界不对齐（`chunk≠bp`）
  时 producer 被两 worker 并发消费、`tail` 越过未消费/在飞 slot 被覆写；对齐（`chunk==bp`，本机 G=P=66）
  则 OK。要让 `D>1` 可用须改协议（每槽独立回收/就绪，MPK per-worker ring / DeepGEMM ring generation）。
  判据 C1–C4、仍差 C5。**本轮新增实验。**
- 此前：**P65-1 完成（1/1）**。补掉 P64-1（§53）§53.9 明留的尾巴「**纯写口径**（`mode 9`、
  `grid=P`、无 worker）：只量 producer 写端；**带 worker 的端到端归因未在本节跑**（§52.8 另一条尾巴）」，
  把 §52/§53 的写端握手归因放回**有消费者**的流水。`docs/04-experiments.md` 新增 **§54**，源码
  `src/dyn_producer_carrier_e2e.cu`、脚本 `harness/dyn_producer_carrier_e2e.sh`、原始输出
  `harness/dyn_producer_carrier_e2e.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `P=32 producer + G=100 worker`（grid=132 全共驻）、`DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、
  `K=32`、`D=1`、`T=256`、min-of-3）。新增 **mode 21 = 动态生产者 + G worker + 可选数据面**。核心结论：
  - **§A** mode21 `ser` 与无生产者 `static` **逐位一致**（`got=404219327039195`）。
  - **§B（headline 一）有界环绑定（`CAPp`=16 批）时，背压×容量的乒乓往返主导（~20×）**：`BP=1` 的档
    全部塌到 ~5 GB/s（vs 纯写同协议 110），`relhead`/`reltail` 与 `ser` 重合——**内存序税被完全盖过**；
    `BP=0` 的 `skipbp`/`nobp_wa` 不被环窗卡住（105/101 GB/s ≈ 纯写）。开关精确是**背压门**。
  - **§B2（headline 二）环容量不绑定（`CAP=NDESC`）时 §53 归因复现但衰减**：`reltail≈ser < relhead <
    nobp_wa < skipbp < strall8`，幅度从纯写 1.44× 降到端到端 **1.17×（EDW=0）/1.12×（EDW=256）**。
  - **§C 临界路径**：`BP=0` 端到端与纯写同量级（producer 临界）；`BP=1` 端到端恒 ~4–5.5 且**与 P 无关**
    → 对 MPK 固定大队列 / DeepGEMM 容量上界 + warmup wave /「批量取 task」以**端到端**理由：**把窗开够大/
    批量取，比重写单条 release 重要一个量级**。
  - **§D 载体**：`BP=1` 时载体差异被环窗抹平；`BP=0` 时复现 `regD>tma>regS`。判据 C1–C4、**仍差 C5**。
  踩坑：后台覆盖 `/tmp` 二进制重跑致 `nvcc` **ETXTBSY 静默失败**（读到旧二进制）——改新路径编译+前台运行；
  worker 跨步分配 + `D>1` 预取会跨 producer 子环边界**死锁**，端到端同步对照取 `D=1`。**本轮新增实验。**
- 此前：**P64-1 完成（1/1）**。补掉 P63-1（§52）§52.8 明留的尾巴——「**换栈（如 v4/coop 协作写，
  见 §49/§50）会改变它，故本节比值不能外推到寄存器写载体**」。**唯一新变量 = 写载体**：
  carrier 0 = **TMA bulk store**（= §52）；carrier 1 = **寄存器 copy-out**；carrier 2 = **寄存器直写**。
  `docs/04` **§53**，源码 `src/dyn_producer_carrier_dep.cu`。核心结论：三载体正确档与 static 逐位一致；
  `§52 的 L 型加性、release 主导在三载体上逐档同形`（`reltail≈ser`、`relhead` 收 ~10–13%），但绝对数值
  不同、快载体相对罚更狠（`ser→nobp` 落差 29.9%/33.6%/36.8%）；寄存器 store 只在**直写不经 SMEM** 时
  优于 TMA，经 SMEM 暂存（regS）反不如 TMA。判据 C1–C4、仍差 C5。
- 此前：**P63-1 完成（1/1）**。补掉 P62-1（§51）§51.8 明留的**开放问题**——「§51.6 的『只拆背压
  或只拆发布都回 664、两者同时才 454』**未钉到 SASS/依赖链一级**」。**唯一新变量 = 归因编码**：把
  「发布握手」从 §51 的**单整数互斥编码**换成**六个正交布尔位**（`WA`/`BP`/`HD`/`REL`/`ACQ`/`FENCE`），
  并补**逐相位 `clock64`** 与 **SASS 反汇编**。`docs/04-experiments.md` 新增 **§52**，源码
  `src/dyn_producer_tma_dep.cu`、脚本 `harness/dyn_producer_tma_dep.sh`、原始输出
  `harness/dyn_producer_tma_dep.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `DW=88`=352 B、`NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、`T=256`、`grid=132` 全共驻、
  min-of-7 + `cuobjdump -sass`）。核心结论：
  - **§A** 正交重编码的正确档（`ser`/摊薄背压/摊薄发布/两者摊薄）与 static **逐位一致**
    （`got=404219327039195`）——摊薄是**合法**进度协议。
  - **§B（headline 一）§51.6 的「只拆一个就够快」是编码假象**：§51 的 `nobp=(win<=-3)` **吞掉**
    `win=-4/-5`，那两行**实际都等于 `nobp`**。真实 L 型加性分解（P=128 纯写）：地板 **630** →
    **只加 release 发布 521** / **只加背压会合 527** → **两者同加 455** → **再叠完成等待 437**；
    **`ld.acquire`→relaxed 完全无效（435≈ser）**、`fence` 免费、**`st.release`→relaxed +13.7%（497）**
    ——**贵的是 release store，不是 acquire/fence**。
  - **§B2** 摊薄「发布」比「背压」值钱；两者一起摊薄 `S=32` 达 **569（+30%）**但**仍差地板 10%**。
  - **§C × P**：结论 `P∈{16..128}` 逐档不变、近似线性（单 producer SM 瓶颈）。
  - **§D（headline 二）逐相位 + SASS**：`gen`=3426 cycles/batch 恒定；`bp` 段背压时 ~840–907
    （**主因是 `__syncthreads`，非 acquire**）；`pub` 段 `ser`1359/`nowait`927/`relhead`432/`skiphd`65。
    反汇编发布路径 = `MEMBAR.ALL.CTA → MEMBAR.ALL.GPU → ST.E.64.STRONG.GPU`——**`st.release.gpu` 每批把
    该 SM 访存流水 order 一次**。⇒ 写端瓶颈 = **每批 release 的 MEMBAR + 会合 barrier**，正是 MPK
    「预计算 `FullTaskDesc` + scheduler 只映射 + 批量/环形摊薄一次 release」绕开的那本账
    （`persistent_kernel.cuh:1254-1523`）。判据 C1–C4、仍差 C5。**本轮新增实验**；P63-1 新留尾巴见 §52.8
    （SASS 计数是整 kernel、合成描述符、纯写口径无 worker、未与 DeepGEMM SM100 对跑）。
- 此前：**P62-1 完成（1/1）**。补掉 P61-1（§50）§50.7 明留的**「最值得做的下一条尾巴」**——把动态
  生产者 TMA bulk store 的**发布握手**从 serialized（每批 `wait_group 0` 全 drain）换成**窗口流水**
  （FIFO commit + 发布第 `bj-win+1` 批 head 前只 `wait_group (win-1)`、head 落后 win 批），看能否把 §50
  的 472 GB/s 拉回 rolling 的 ~722–828。`docs/04-experiments.md` 新增 **§51**，源码
  `src/dyn_producer_tma_pipe.cu`、脚本 `harness/dyn_producer_tma_pipe.sh`、原始输出
  `harness/dyn_producer_tma_pipe.out.txt` + ncu 归因 `harness/dyn_producer_tma_pipe_ncu.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`、`K=32`、`D=2`、`T=256`、
  `grid=132` 全共驻、min-of-7）。唯一新变量 = 发布握手；另用一族**哨兵口径**（`win<0`、**故意不正确**）拆代价。
  核心结论（**§50.7 预测证伪 + §50 归因修正**）：
  - **§A** static/win=1/2/3/4 **逐位一致**（`got=404219327039195`）——窗口化发布语义透明。
  - **§B（有 worker）窗口只压 2.2–3.7%**（P=16 14.67×→14.13×）；**§C 纯写 P=128 ser 453.7→pipe 462.8（+2%）**，
    rolling 733.2 / v4 827.8，**远追不回**；**§E 深窗**（`sb`=win=16、SMEM 180 KB）也只 **468.3（+1.6%）**。
  - **§F（headline 归因）**：ser 453.6 / nowait 486.3（拆完成等待仅 **+7%**）/ **nobp 664.3（拆背压+head，+46%）**/
    skipbp 664.5（**只**拆背压）/ skiphd 664.8（**只**拆 head）/ rolling 733.1 / sgen 719.3 / v4 825.6。
    ⇒ **§50 的「每批 `wait_group 0`」不是主因；真正贵的是「每批发布一次跨 SM 进度」**（`st.release head` +
    背压门 `ld.acquire tail`），且**两者同时存在才昂贵**（单拆任一个都回 664）。
  - **ncu**：DRAM 写字节相同（115.8 MB），正确握手 **barrier 停顿 31.19% vs 11.85%**、多出 `long_scoreboard`/`membar`。
  含义：**写端正确握手瓶颈不在 store 引擎、不在完成等待，而在逐批报进度**——正是 MPK「编译期预计算
  `FullTaskDesc` + scheduler 只做事件→id 映射、批量/环形摊薄一次 release」绕开的那本账；**「预计算 + 读」是
  描述符通路唯一可行形态**的结论更强。判据 C1–C4、仍差 C5。**本轮新增实验**；P62-1 新留尾巴见 §51.8
  （「只拆一个就够快」的组合交互**未钉到 SASS/依赖链一级**——开放问题；合成描述符；纯写口径无 worker；
  未与 DeepGEMM SM100 对跑）。
- 此前：**P61-1 完成（1/1）**。补掉 P60-1（§49）§49.7 明留的**「最值得做的下一条尾巴」**——把动态
  生产者的写载体从寄存器 store（标量/v4/coop）扩到最宽的 **TMA bulk store**（`cp.async.bulk.global.shared::cta`，
  SMEM→global），看它能否把「写」接上近 HBM 带宽。`docs/04-experiments.md` 新增 **§50**，源码
  `src/dyn_producer_tma.cu`、脚本 `harness/dyn_producer_tma.sh`、原始输出 `harness/dyn_producer_tma.out.txt`
  （目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、`NDESC=393216`、`K=32`、`D=2`、
  `T=256`、`grid=132` 全共驻、min-of-7）。核心结论（**证伪 §49 假设**）：
  - **§A** 四载体 × static/sched/gen **逐位一致**（`got=404219327039195`）——跨 async proxy store→load
    握手（`wait_group 0`+`fence.proxy.async.global`+release/acquire）语义透明。
  - **§B（headline）** static=0.1698 ms/815 GB/s；TMA store 的 `gen` **几乎等于标量**（`tma/s` 0.72–0.96，
    P=1 35.14 vs 36.61 ms），比 v4 慢 2–4×、比 coop 1.37–1.82×。**TMA store 接不上读带宽。**
  - **§C/§C2** 纯写 P=128：v4 812 / coop 882 / **tma(serialized 正确握手) 472.6** / **tma(rolling) 722.5**
    GB/s；`sgen`（只生成到 SMEM）766 → rolling 722（store 几乎白送）→ serialized 471（正确握手每批
    `wait_group 0` 砍 39%）。**两跳上限 766、任何 P 下 TMA 都 ≤ 16 B 寄存器 store（roll/v4 0.44–0.89）。**
  - **§C3** serialized 的 `wait_group.read` 深度 tw=1/2/3 在 P=128 恒 ~474（深度旋钮失效）。
  - **§C4** 大块 store 值钱：NB=1(11KB) 722.5 → NB=8(90KB) **827.8 GB/s**，但仍 **0.94× coop**、与 v4 打平，
    **从未超过**；NB=16 超 H100 opt-in 上限。
  含义：**写端天花板（最宽载体）~828（rolling）/472（正确握手）≪ 读端 809–3000 GB/s**——给 MPK
  「预计算 `FullTaskDesc` + scheduler 只映射」与 DeepGEMM「静态角色/编译期常量」以更强量化理由。
  判据 C1–C4、仍差 C5。**本轮新增实验**；P61-1 新留尾巴见 §50.7（**「正确握手+流水」未测**——FIFO commit +
  发布 head 前只 `wait_group (win-1)`，理论把 472 拉回 722–828 且保持正确，**最值得做的下一条**；高 P 读写
  争抢未量化、store 粒度受 SMEM 限、合成描述符、未与 DeepGEMM SM100 对跑）
  （构建期踩坑：TMA 写路径初版对 `mode 2` 误生效向未分配 `ring` 发 bulk store → `illegal memory access`/
  后续挂起；修正为仅 `is_gen||smem_only` 走 TMA 写路径后恢复）。
- 此前：**P60-1 完成（1/1）**。补掉 P59-1（§48）§48.8 明留的**尾巴②**——「`gen` 仍是**标量
  strided 写**；向量化/协作写未单独做」：把 producer 写入形态分解成 **存储宽度**（4 B→16 B `uint4`）
  与 **coalescing**（逐线程 strided vs 协作连续）两个正交旋钮，实测各自贡献与饱和点。`docs/04-experiments.md`
  新增 **§49**，源码 `src/dyn_producer_vector.cu`、脚本 `harness/dyn_producer_vector.sh`、原始输出
  `harness/dyn_producer_vector.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、
  `NDESC=393216`（138.4 MB ≫ L2）、`K=32`、`D=2`、`T=256`、`grid=P+G=132=NSM` 全共驻、min-of-7）。
  `gvar`：0 标量 / 1 逐线程 `uint4` / 2 协作 coalesced（payload 同 `gen_word`）；`mode 5 pure` 量纯写。
  核心结论：
  - **§A** 三形态 × P∈{1,2}，static/sched/gen **逐位一致**（`got=404219327039195`）。
  - **§B（headline）** static=0.1711 ms/809.1 GB/s；逐线程 `uint4` 在 P=1 把 `gen` **36.58→10.49 ms
    （3.5×）**、相对 static **213.8×→61.3×**；协作 coalesced ~**1.9×**；**饱和点右移**（标量 P≈8 后
    偏离、v4 到 P=64 仍近线性），但最优 v4（P=64）仍 **3.44× static** ⇒ §48「动态必须落在调度、不能
    落在生成」依旧成立。
  - **§C（机制）** 纯写：单 SM **3.8→13.2 GB/s**（标量→v4），瓶颈是 **store 发射而非 DRAM 写带宽**；
    整机写天花板标量 ~**207**（P≈32 饱和）→ v4 **827** / coop **875**（P=128），抬高 ~4×。**coalescing
    是更弱第二旋钮**（v4 低 P 更快、P=128 才被 coop 反超）——真正值钱的是 **16 B store**。
  - **§D** P=16/EDW=4096 标量 `s/max=1.52`（复现 §48 §E）→ 协作向量化 **1.32**，收窄未翻案。
  - **§E** 高 P 下 `gen` 慢于纯写（写与 worker 读争 HBM）。
  含义：给 MPK「编译期预计算 `FullTaskDesc` + scheduler 只映射」与 DeepGEMM「静态角色/编译期常量」
  以量化理由。判据 C1–C4、仍差 C5 一步。**本轮新增实验**；P60-1 新留尾巴见 §49.7（**TMA bulk store
  作更宽写载体未测**——最值得做的下一条、高 P 读写争抢未单独量化、合成描述符、未与 DeepGEMM SM100 对跑）。
- 此前：**P59-1 完成（1/1）**。补掉 P58-1（§47）§47.8 明留的**前三条尾巴**——「`gen` 的 producer
  是**单块**（可分散到多块/SM）」（①）「`gen` 标量 strided 写」（②）「`sched` 的 1.36 ms 是**单块
  producer** 上界，MPK 实际用 **16 scheduler SM**」（③）：把 §47 的单块 producer 扩成 **P 块（独占 P 个
  SM）**，量 gen/sched 代价随 P 的标度与饱和点。`docs/04-experiments.md` 新增 **§48**，源码
  `src/dyn_producer_scale.cu`、脚本 `harness/dyn_producer_scale.sh`、原始输出
  `harness/dyn_producer_scale.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`=352 B、
  `NDESC=393216`（静态数组 138.4 MB ≫ L2）、`K=32`、`D=2`、`T=256`、`grid=P+G=132=NSM` 全共驻、min-of-9）。
  机制 = §47 骨架 + **环按 producer 分区**（每生产者 `CAPp=CAP/P` 槽、自己的 `head[p]/tail[p]`、
  **P 条独立 SPSC 子环**、worker 按 `owner=bb/bp` 定位）；`P∈{1..128}`。核心结论：
  - **§A** `P∈{1,2,4}`×`D∈{1,2}` static/sched/gen **逐位一致**（`got=404219327039195`）；并记 P=1 的
    「跨步消费预取领先量」浅环死锁边界，**分段子环反而放宽该约束**。
  - **§B（headline）** static=0.169 ms/819.5 GB/s；**`gen` 的 1/P 只到 P≈8**（`gen·P` P=1..8 恒 36.5→40.0、
    P=16 起 42/61.7/80.7），**最优 P=64=1.261 ms（28.9× 单块）仍 7.46× static**，P=128 退点（`G=4`）；
    **`sched` 撞「静态读地板」**：1.32→**0.262 ms（P=32，1.55× static）**后不再降，**P=16=0.275 ms
    （1.63× static）正落 MPK 的 16 scheduler SM 量级**。⇒「动态必须落在调度、不能落在生成」在 P 维度成立。
  - **§C** `gen` 算被存藏住（P=16 时 `PW≤4096` 全程只 15.5→17.1）、`sched` 算线性显形（绝对值按 ~1/P 缩）。
  - **§D** 小窗口 `CAPp=K` 必死锁；满速需 `CAPp ≥ bp·K`（每 producer 整个区间）；容量需求随 P 按区间摊薄。
  - **§E 接数据面**：单块 `gen` 恒 ~36.4 ms（7.1× static @EDW4096）；**P=16 收窄到 1.52×**（近乎翻案成尾巴）。
  判据 C1–C4、比 §36–§47 更接近 C5（P 个固定角色 producer）但 worker 仍静态分配 batch ⇒ **仍差 C5 一步**。
  **本轮新增实验**；P59-1 新留尾巴见 §48.8（标量写、P=128 消费者瓶颈、sched 地板随描述符数组大小变、
  run-to-run 抖动 ~34%、合成产物、未与 DeepGEMM SM100 对跑）。
- 此前：**P58-1 完成（1/1）**。补掉 §39.7/§40.8/§42.7/§43.7/§44.7/§45.7/§46.7 **每一节都明留的
  同一条尾巴**「**未叠动态生产者**」：§39–§46 把「worker 取 task」的载体/缓冲/占用率/HBM 饱和都量透，
  但取数数组全是 **host 预写的静态块**；真实 megakernel 由设备侧 scheduler **现场**推进 task。
  `docs/04-experiments.md` 新增 **§47**，源码 `src/dyn_producer_fetch.cu`、脚本
  `harness/dyn_producer_fetch.sh`、原始输出 `harness/dyn_producer_fetch.out.txt`（目标卡 H100/sm90a、
  **固定空闲 device 3**、nvcc 13.2、`DW=88`、`NDESC=393216`、`K=32,D=2,T=256`、worker `G=131`、min-of-9）。
  三模式 = `static`（预写数组）/`sched`（预写数组 + producer 只推 `head` + worker acquire 等）/
  `gen`（producer 现场生成 352 B payload 写有界环 + 背压 + `fence.proxy.async` + release 发布）。
  核心结论：
  - **§A** 三模式任意 `D=1..4` **逐位一致**（`got=404219327039195`）——现场生成 + release 发布 +
    TMA 跨 proxy 读 + 有界环背压**不动数值**。
  - **§B（headline）动态分两层、差 ~27×**：`EDW=0` 下 `static` **0.1605 ms/862.5 GB/s（0.41 ns/条）**、
    `sched` **1.3614 ms（8.48×，3.46 ns/条）**、`gen` **36.489 ms（227.4×，92.8 ns/条）**。
    现场生成瓶颈 = **单 SM 标量写吞吐**（1536 轮 × ~0.9 µs），与 `G` 无关。
  - **§C** `gen` 的额外计算被存储隐藏（`PW≤1024` 恒 227×），`sched` 的额外计算直接显形（`PW=4096`→42.4×）。
  - **§D** 死锁边界 ~`2·(G·K)`（`CAP=4192` 死锁、`8384` 活），安全容量之上吞吐不变（安全阀非旋钮）。
  - **§E** `sched` 固定成本 ~1.2 ms，`EDW=4096` 时被藏到 **1.014×**（**动态调度近乎免费**，与 §44–§46
    「控制面被数据面藏住」一致）；`gen` 恒 ~36.5 ms **不随数据面缩小**。
  含义：**megakernel 的「动态」必须在调度（推 task id）、不能在生成 payload**——正是 MPK（`runtime.cc`
  预计算 `FullTaskDesc`、scheduler 只映射入队）与 DeepGEMM（编译期静态角色）的做法，给论文 §5.2 的
  AOT/JIT 分野以量化理由。判据 C1–C4，producer 固定角色 + 设备侧动态发布**比 §36–§46 更靠近 C5**，
  但 worker 仍静态分配 batch ⇒ **仍差 C5 一步**。**本轮新增实验**；P58-1 新留尾巴见 §47.8。
- 此前：**P57-1 完成（1/1）**。补掉 P56-1（§45）§45.7 明留的第一条尾巴——§45 量出「数据面
  要多线程（`T=256→1024` 抬到 2983 GB/s）vs 控制面要少线程（掉到 891 GB/s）」这对相反最优，但占用率
  被 SMEM 环锁在 1 block/SM，自陈「测多 block/SM 需小环，但小 `K` 让控制面低效，两者不可兼得」。
  `docs/04-experiments.md` 新增 **§46**，源码 `src/fetch_exec_multiblock.cu`、脚本
  `harness/fetch_exec_multiblock.sh`、原始输出 `harness/fetch_exec_multiblock.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`DW=88`、`NDESC=393216`、数据面至 6.4 GB、min-of-9）。
  唯一新变量 = **SMEM 环 ⇒ 占用率**（`K=16,D=2` ⇒ 11 280 B ⇒ 8 blk/SM @T256；`K=128` ⇒ 90 128 B ⇒ 2）。
  四 Section 核心结论：
  - **§A** 多 block + 小 `K` 下 `D=1..4` 的 `sink` 逐位一致。
  - **§B（headline 之一）数据面饱和旋钮是「每 SM 总线程数」而非「`blockDim`」**：所有 `thr/SM=1024`
    拆法（`1,1024`/`2,512`/`4,256`/`8,128`）落 **2928–3009 GB/s**、`512` 落 2157–2209、`256` 只 1242；
    **`B=4,T=256`（3009）≥ §45 的 `B=1,T=1024`（2928）**——多块不劣于单块大 T ⇒ **§45 的单块
    `blockDim` 矛盾可由多 block/SM 化解**（保留 `T=256`、用 `B=4` 凑 1024 线程/SM）。
  - **§C（headline 之二）控制面高度依赖 `K`、对占用率较不敏感**：1 blk/SM、`T=256` 下
    `K=8/16/32/128` = **437/691/937/1782 GB/s**（小 `K` 每批 mbarrier 等待 + 两次 `__syncthreads`
    摊薄）；多 block 的控制面惩罚（1.26×/1.70×）**轻于 §45 大 `blockDim`（2.0×）**。⇒ §45.7 的
    「小环 ↔ 控制面低效」取舍是真的，但**代价来自 `K` 而非占用率**。
  - **§D（headline 之三）小环 + 多块下 `D=2` 仍藏住控制面**：`G=1056,B=8,T=256,K=16`，`p/max`
    峰值 **1.126**（EDW=64）→ **1.020**（EDW=4096），全程 `p/(f+e) ≤ 0.914`。
  含义：**数据面认「总线程/SM」、控制面认「`K`」两个正交量** ⇒ 「用多块凑 1024 线程/SM、把 `K` 做大」
  二者在 SMEM 预算内可同时满足；把 §45 的 **warp specialization**（同 block）与本节的多 block/SM
  （跨 block）并列为两条兼顾路径。判据 C1–C4、无 C5。**本轮新增实验**；P57-1 新留尾巴见 §46.7
  （合成 `uint4` 数据面、未观测超饱和争抢、未叠动态生产者）。
- 此前：**P56-1 完成（1/1）**。补掉 P55-1（§44）§44.7 明留的第一条尾巴——§44 得到
  `D=2≈max(fetch,exec)`，但自陈口径边界「**数据面未打满 HBM**」（1 block/SM + `uint4` 只到
  ~1.2–1.4 TB/s），「若执行把 HBM 打满、控制面是否转为可见争抢未测」。`docs/04-experiments.md`
  新增 **§45**，源码 `src/fetch_exec_saturate.cu`、脚本 `harness/fetch_exec_saturate.sh`、原始输出
  `harness/fetch_exec_saturate.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `G=132=#SM`、`K=128`、`DW=88`、`D=1/D=2`、`NDESC=393216`、数据面至 6.4 GB、min-of-9）。唯一新
  机制变量 = **block 线程数 `T`**。四 Section 核心结论：
  - **§A** `D=1..4` 的 `sink` 逐位一致（不动数值）。
  - **§B（headline 之一）数据面与控制在「每 block 线程数」上最优相反**：数据面读带宽随 `T` 上升并
    **在 `T=1024` 饱和到 ~2983 GB/s（EDW=4096、≈ HBM3 峰值 ~89%）**，`T=256` 恒卡 **~1.2 TB/s**
    （=§44 口径，其「未饱和」根源是线程数而非 block 数）；**控制面 `fetch_only` 取数带宽随 `T` 下降**
    （`T=256→1024`：**1791→891 GB/s**，多线程只增 `mbar_wait` 自旋/`__syncthreads` 宽度）。
  - **§C（headline 之二）饱和不暴露可见争抢**：`D=2` 全程趋 `max(fetch,exec)`，`p/max` 峰值
    **1.132**（EDW=64）→ **1.015**（EDW=4096），全程 `p/(f+e) ≤ 0.947`；最坏点 EDW=64 若相加应
    ≈**1.89**、实测 **1.13**；EDW=4096 数据面已饱和，`D=2` 只比纯执行慢 **1.5%**、总带宽
    **3007 vs 3052 GB/s 同档**（控制面钻 ~10% HBM 余量）。
  - **§D** 快照单缓冲 `D=1` 相对 `D=2` 的惩罚 **1.115×→1.004×**（EDW=0→4096），绝对缝恒 ~0.02–0.04 ms。
  含义：(1) 控制面描述符取数**恒不与执行争带宽**（只要描述符 ≪ task 主体数据）；(2) 控制面要少线程、
  数据面要多线程，单 worker block 只有一个尺寸（快照 `T=256` 控制最优/数据最差），要兼顾只能
  **warp specialization**（P13-1/P18-1），从执行侧解释生产 mega-kernel 普遍 warp 特化。判据 C1–C4、
  无 C5。**本轮新增实验**；P56-1 新留尾巴见 §45.7（1 block/SM 限制、合成数据面、未叠动态生产者）。
- 此前：**P55-1 完成（1/1）**。补掉 §43.7（P54-1）明留的第一条尾巴——§43 的 `exec_task` 是
  **纯 ALU 压舱、不吃 HBM**，故得 `D=3=max(fetch,exec)`；但它自己点名「真实 task 会同时吃 HBM，
  『取数与执行是否争带宽』需另测」。本节把 §43 的「取 task 与执行同环」接上一块 **≫L2 的数据面流式读**
  （每 batch 连续 `K·EDW` 个字、coalesced `uint4`），让控制面 TMA 与数据面读**共用同一条 HBM**。
  `docs/04-experiments.md` 新增 **§44**，源码 `src/fetch_exec_hbm.cu`、脚本 `harness/fetch_exec_hbm.sh`、
  原始输出 `harness/fetch_exec_hbm.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、
  `G=132`、`NDESC=393216`×352 B ≈ 132 MB ≫ L2、每配置 2 次 warmup + min-of-5 reps 压噪声）。核心结论：
  - **§A** `EDW=8` 下 `D=1..4` 的 `sink` **逐位一致**（加数据面/预取/环深不动数值）。
  - **§B（headline）加数据面后 `D=3` 仍趋 `max(fetch,exec)`、不与执行相加**——`p/max`（= `D3/exec`）
    `EDW=0` 复现 §43 的 **1.003**、`EDW=16` 峰值 **1.247**、随数据面增大回落 `EDW=1024` 的 **1.010**；
    全程 `p/(f+e) ≤ 0.957`。最坏点 `EDW=16` 若串行相加应 ≈ **1.94**、实测 **1.25**（取数独立代价只
    暴露 ~27%）。因 1 block/SM 的数据面 `uint4` 流式只到 **~1.2 TB/s** < 描述符流单独的 **~1.8 TB/s**
    → HBM 有余量，控制面条带被预取压上。
  - **§C** 扫 `R`（`EDW=256`）：`p/max` ~1.03–1.04，随执行变长（`exec_only` 0.38→1.12 ms）**收敛回 1.009**
    ——**执行越重、取数越藏**（与 §43 同向）。
  - **§D** 取 task 净加价 `D3−exec` 稳定 **0.01–0.02 ms**（fetch_only 0.0741 ms 的 2%–26%），峰值
    0.0195 ms 出现在**数据面刚好装进 L2（`EDW=16`）**的窄区（无 HBM 往返可藏）。
  含义：**取 task 与执行不构成零和争抢**——只要执行够长且 HBM 未打满，预取就把控制面藏住；对 DeepGEMM
  「重且吃 HBM」的段级 task（`p/max→1.01`）与 MPK「多而薄」的 task 都自洽。判据 C1–C4、无 C5。
  **本轮新增实验**；P55-1 新留尾巴见 §44.7（数据面未打满 HBM、合成执行、`do_fetch=0` 是另一代码路径）。
- 此前：**P54-1 完成（1/1）**。补掉 §39–§42（P50-1..P53-1）的整合缺口——那四节把「worker 取
  task」的载体与缓冲预算量透了，但**全在「只取数、不执行 task」的口径**下。本节第一次把「取 task」与
  **真正 task 执行**放进同一循环，量「描述符取数能否被 task 执行藏住」。`docs/04-experiments.md`
  新增 **§43**，源码 `src/fetch_exec_overlap.cu`、脚本 `harness/fetch_exec_overlap.sh`、原始输出
  `harness/fetch_exec_overlap.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、`G=132`、
  `NDESC=393216`×352 B ≈ 132 MB ≫ L2）。取数载体 = TMA bulk + 硬件 mbarrier（`D` 槽环）；task 执行 =
  数据相关折叠 + `R` 轮串行 LCG 压舱（`R` = 唯一执行时间旋钮）；`do_fetch=0` 档隔离纯执行。核心结论：
  **§A** `D=1..4` checksum 与 host 逐位一致（`R=0`）/ 逐位相同（`R=1024`）；**§B（headline）** 预取
  `D=1→D=3` 的 speedup 随 `R` **单调衰减**（`R=0` 1.354× → `R=262144` 1.003×），而**绝对收益恒定**
  ~0.017–0.023 ms/run（单缓冲只多一条固定长度的串行缝，不随计算放大）；**§C** `D=2` 基本拿满（同 §41/§42）；
  **§D 决定性分解**：`fetch_only` 恒 0.061 ms、`exec_only` 随 `R` 线性，**`D=3 = max(fetch_only,
  exec_only)`（p/max 1.001–1.047）→ 预取把描述符取数完全藏在执行后**，`D=1` 落在 `exec+~0.02ms`；
  **§E** speedup 随 `K`（32→128）1.154→1.228。含义：**取 task 在重 task 下完全可藏、只在轻 task 下显形**
  ——MPK「多而薄」的 task（§01 §3）更需要 pre-fetching（§5.3 行 1364-1368），DeepGEMM「少而重」不敏感。
  判据 C1–C4、无 C5。**本轮新增实验**；P54-1 新留尾巴见 §43.7。
- 此前：**P53-1 完成（1/1）**。补掉 P52-1（§41.8）明留的两条尾巴——「载体固定为 TMA bulk +
  mbarrier，未同时跑 16 B `cp.async` 环」与「未测 `cp.async` 释放的寄存器对下游 task 执行的实际收益」：
  把快照**原生**取 task 载体（16 B `cp.async` 进 SMEM）放回 §41 的 `K×D` 平面对跑。`docs/04-experiments.md`
  新增 **§42**，源码 `src/cpasync_ring_budget.cu`、脚本 `harness/cpasync_ring_budget.sh`、原始输出
  `harness/cpasync_ring_budget.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、DW=88=352 B）。
  核心结论：cp.async 与 bulk 高下随 `K` 翻转、「cp.async 释放寄存器」被否证（价值是「成本与描述符大小
  无关」+「零寄存器代价预取」）。判据 C1–C4、无 C5。
- 此前：**P52-1 完成（1/1）**。补掉 P51-1（§40.8）明留的第一条尾巴「**未测 `K` 与 SMEM 容量的
  联合约束**」：把 §40 的「双缓冲」推广成 **D 深环形缓冲**（每槽 K 条 `TaskDesc`），描述符暂存 SMEM
  = `D·K·352 B`，量「同等 SMEM 预算下大批量浅环 vs 小批量深环」。`docs/04-experiments.md` 新增 **§41**，
  源码 `src/fetch_smem_budget.cu`、脚本 `harness/fetch_smem_budget.sh`、原始输出
  `harness/fetch_smem_budget.out.txt`（目标卡 H100/sm90a、**固定空闲 device 3**、nvcc 13.2、DW=88=352 B）。
  载体 = TMA bulk + 硬件 mbarrier。核心结论：
  - **§A 任意 (K,D) 多 worker `bad=0`**；**§B 环深 D：1→2 拿到全部预取收益、D≥2 完全平坦**
    （K=8：402.8→579.9 GB/s 后恒 ~580）；§D（K=1 扫 D=64…640）同证吞吐与 D 无关。
  - **§C（headline）K×SMEM 联合约束**：同等 SMEM 下吞吐**单调取决于 K**（每批同步开销摊到 K 条）——
    `K=1,D=186`（用满 64 KB）73.8 GB/s vs `K=64,D=2`（只用 44 KB）**2096 GB/s（28×）**；`K=128,D=2`
    达 **2667**、`K=128,D=4` **2693 GB/s**（逼近 HBM ~80%）。**最优 = 大 K + D=2**，深环是浪费。
  - **§D** 环可长到 225 KB（占满一整个 SM）仍正确（opt-in=232448 B），但深环无用。
  - **§E 与 task 执行争 SMEM**：每 block SMEM 0.7→88 KB 把占用率从 **16→2 block/SM**，aggregate
    带宽守住 **2.3–2.7 TB/s**（大 K 在飞字节补偿 worker 数下降）；最优 K=8（5.5 KB、16 block/SM）
    2672 GB/s。⇒ 取 task 的 SMEM 是**软约束**，正确姿势是**堆到批量 K（≈32–128）、预取深度只 2**，
    对应 MPK §5.3 paged SMEM（行 1313-1325）。判据 C1–C4、无 C5。**本轮新增实验**；P52-1 新留尾巴
    （§41.8）：~~未跑 16 B `cp.async` 深环载体~~、~~未测 `cp.async` 释放寄存器对下游 task 的收益~~
    （**均已由 P53-1/§42 补掉**）、无动态生产者、未与 DeepGEMM SM100 路径对跑。
- 此前：**P51-1 完成（1/1）**。补掉 P50-1（§39.7）明留的两条尾巴——「**单 block** 测的是『取一批
  的延迟』而非『访存带宽』，要多 worker 的带宽标度需另做多 block 实验」与「**`mbarrier` 形态未测**」：
  把 §39 的「单 block 消费者」扩成 **G 个 worker block**，并新增两条 **TMA bulk + 硬件 mbarrier** 取数
  路径。`docs/04-experiments.md` 新增 **§40**，源码 `src/multi_worker_desc_fetch.cu`、脚本
  `harness/multi_worker_desc_fetch.sh`、原始输出 `harness/multi_worker_desc_fetch.out.txt`（目标卡
  H100/sm90a、**固定空闲 device 3**、nvcc 13.2、256 MB ≫ L2）。六路径 `scalar`/`vec4`/`cpasync`/
  `cpasync_pf`/`bulk`(TMA+mbarrier 单)/`bulk_pf`(双)，全 `bad=0`。核心结论：
  - **§B 多 worker 标度**：aggregate 带宽随 G 近似严格线性、到 G=2×SM 仍未饱和——`vec4` G=1 3.3→
    G=264 **803 GB/s**（每 worker 恒 ~3.0 GB/s）；**取 task 是 per-worker 延迟绑定，带宽只能堆 worker**
    （对上 MPK §5.1 worker 数按 SM）。128 B 下 `vec4` per-worker 最高。
  - **§C** TMA bulk 单缓冲 **≡** 16 B `cp.async`（K=32 0.41 vs 0.40）；收益全在双缓冲预取
    （`bulk_pf` K=1 3.21 vs `cpasync` 4.90 = 1.53×）。
  - **§D 真实 352 B 下 `bulk_pf` 独一档**：0.28 ns/item = **1263 GB/s**，比 `vec4` 快 **3.75×**、
    比 `cpasync_pf` 快 **1.52×**；`cpasync`/`bulk` 几乎不随 `DW` 变（`cpasync` 903.8 vs `vec4` 335.5）。
  - **§E** 多 worker 下线程数不再是变量。**踩坑**：`cpasync` 双缓冲最后一批 `wait_group 1` 会漏等
    （首版 `bad=1`），改 `wait_group 0` 修复；**§39 `cpasync_pf` 有同构写法**，当时 `bad=0` 是侥幸
    （与 §36.3 同源），结论不受影响但记为待修隐患。
  判据 C1–C4、无 C5。**本轮新增实验**；P51-1 新留尾巴（§40.8）：1D bulk 非 TensorMap、无动态生产者、
  未测 K×SMEM 联合约束、未与 DeepGEMM SM100 路径对跑。**P0–P51 全部完成。**（P51-1 尾巴的
  「K×SMEM 联合约束」已由 **P52-1** 补掉，见上「当前进度」。）
- 此前：**P50-1 完成（1/1）**。补掉 P49-1（§38.7）明留的尾巴「`cp.async` 只是等效语义（§38 用
  普通 `ld.relaxed` 读，**未实测 `cp.async` 批量加载本身的收益**——那是另一条线）」「`warp` 下
  `cp.async`/`mbarrier` 的批量加载形态」：把 worker「取 task」的**载体选择**单独量出来。
  `docs/04-experiments.md` 新增 **§39**，源码 `src/cpasync_desc_fetch.cu`、脚本
  `harness/cpasync_desc_fetch.sh`、原始输出 `harness/cpasync_desc_fetch.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、单 block 持久化消费者、256 MB ≫ L2）。四取数路径 `scalar`（4 B
  寄存器）/`vec4`（128-bit 寄存器）/`cpasync`（SMEM 单缓冲）/`cpasync_pf`（双缓冲预取）。真实锚点：
  worker 协作 16 B `cp.async` 批量搬 `TaskDesc` 进 SMEM（`persistent_kernel.cuh:1063-1078`；单缓冲
  `cp_async_wait<0>` `:1076-1077`）；`load_smem` = `cp.async.ca.shared.global.L2::128B`
  （`tasks/common/copy_sm80.cuh:44-57`）；`TaskDesc` 真实 **352 B**（论文 §6.1 行 1365）；论文 §5.3
  「Pre-fetching task descriptions」（行 1364-1368）。核心结论：
  - **§A/§B 载体不动数值**（四路径全 `bad=0`）。**`cp.async` 的杀手锏是「成本与描述符大小无关」**：
    小描述符（≤128 B）单缓冲 `cpasync` 反比 `vec4` 慢 1.15–1.44×；**真实 352 B（`DW=88`）翻盘——
    `cpasync` 58.0 vs `vec4` 132.9（2.29×）、vs `scalar` 167.1（2.88×）**（寄存器读被寄存器压力拖垮，
    `cpasync` 从 128 B→352 B 几乎不动 55.7→58.0）。
  - **§C 批量大小 `K`**：`cp.async` 价值集中在**任务稀疏（`K` 小）**——`K=1` 474.5（pf）vs 1030.3（vec4，
    2.17×）、`K=4` 127.9 vs 264.6（2.07×）；`K≥32` 则 `vec4` 反超（38.6 vs 55.7）。
  - **§D 预取（双缓冲）** 相对单缓冲稳定收回 **7%–40%**（`K` 越小越多）——**`cp.async` 单用只是「换载体」，
    加预取才是「流水」**，对应论文 §5.3 pre-fetch。**§E 线程数是弱标度**（单 block 受延迟约束）。
  判据 C1–C4、无 C5（与 §36–§38 同档）。**本轮新增实验**；P50-1 新留尾巴（§39.7）：单 block 非多 worker
  带宽口径、`mbarrier` 形态未测、未与 DeepGEMM SM100 路径对跑。
- 此前：**P49-1 完成（1/1）**。补掉 P48-1（§37.7）明留的两条尾巴——「消费者是**单线程**、
  每项 ~2.3µs 同步往返主导成本」（生产实现会批量取 task）与「payload 是纯 `uint32`、**未做真实 task
  描述符**」：把 §37 的 MPSC 队列升级为「**批量出队 + 一个 warp 的并行校验 + 128 B 真实 `TaskDesc`**」。
  `docs/04-experiments.md` 新增 **§38**，源码 `src/mpsc_batch.cu`、脚本 `harness/mpsc_batch.sh`、原始输出
  `harness/mpsc_batch.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、grid=P+1；
  生产者协议与 §37 `mpk` 逐字一致）。真实锚点：worker 用线程 0 读区间 + **128-bit `cp.async` 批量搬
  `TaskDesc` 进 SMEM**（`persistent_kernel.cuh:1063-1078`）；`TaskDesc`（`runtime_header.h:296-336`，
  `alignas(16)`，`input_ptrs[8]`/`output_ptrs[3]` `:327-328`）~128 B；`per_worker_queue_len=8192`
  （`:1790-1791`）；`prepare_next_batch`（`:1378-1423`）。消费者三档 `serial`/`batchK`/`warpK`，
  描述符 `DW∈{8,16,32}`=32/64/128 B，另测「只查 task id」路由粒度。核心结论：
  - **§A 批量/多线程不动数值**：`DW=32`、`P=8` 下 `serial`/`batchK=32`/`warpK=32` **全 `bad=0 / 800000`**；
    连跑 3 次恒 0（连续 CAS 游标保证「读到 `last_ready=c+K` ⇒ 前 `c+K` 槽已写」）。
  - **§B 单线程批量只收回「放行往返」，1.2–1.5× 封顶**：`DW=8`、`P=8`，K=1→32 为 2144.63→1462.61 ns/item
    （**1.466×**，K≥16 饱和）；`DW` 越大收益越小（1.47→1.31→1.16×）——瓶颈从「往返」转到「串行读 payload」。
  - **§C/§D 多线程（warp）并行读描述符，128 B 下快 21×**：`serial`→`warp`，`DW=8` 7.86×、`DW=16` 11.96×、
    `DW=32` **21.03×**（6353.50→302.08 ns/item）；`serial` 随描述符线性变贵（~170 ns/字）而 `warp` 几乎
    不变。§D 拆因：`serial` 路由 1218 vs 全校验 6353（5.2×）、`batch` 341.64 vs 5477（16.0×）、`warp`
    301 vs 303（几乎相同）——**能否并行读 payload 是「路由型」与「全描述符型」消费的分水岭**。
  - **§E 多线程把瓶颈从消费者翻到生产者**：`batch` 与 `P` 无关（恒 ~5.47µs）；`warp` 随 P 下降
    （1→1786、4→448、8→303 ns/item）到 P=8 接近生产者侧成本。
  判据 C1–C4、仍差 C5 一步（与 §31–§37 同档）。**本轮新增实验**；收口 §37.7 两条尾巴。
- 此前：**P48-1 完成（1/1）**。补掉 P47-1（§36）§36.6 明留的尾巴「未测多生产者（MPSC）」：
  把 §36 的 SPSC 队列推广到 **MPSC**，并把「发布内存序」升级为「发布**协议**」。`docs/04-experiments.md`
  新增 **§37**，源码 `src/mpsc_ring.cu`、脚本 `harness/mpsc_ring.sh`、原始输出 `harness/mpsc_ring.out.txt`
  （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2）。真实锚点：MPK `sched_queue` 是 MPSC
  （128 worker/16 scheduler，`get_rand_sched_id` `persistent_kernel.cuh:857-869` ⇒ 8 生产者/队列；
  push = 占位`atom.add.release` → relaxed 写 → **连续 CAS 放行** `:1206-1228`；`worker_queue` SPSC 用单条
  `atom.add.release` `:1495-1502`）。核心结论：`mpk`/`pertile` 多生产者全对；结构不严谨替代自然时序
  侥幸 0 错但 `stall0` 下 `bad=8`（3/3）、正确协议恒 0；放行 ≈ +0.38 µs/项。判据 C1–C4、仍差 C5 一步。
  **（P48-1 全专题任务清单已完成。）**
- 此前：**P47-1 完成（1/1）**。补掉 P46-1（§35）§35.7 明留的尾巴「入队是『先推 `tail`、后写数据』
  （**无 release 发布**），理论上消费者可能在数据可见前读到槽；真实实现（MPK）用 release/acquire 把它
  做实——可作后续实验」：把「发布内存序」做成实测。`docs/04-experiments.md` 新增 **§36**，源码
  `src/pub_order_ring.cu`、脚本 `harness/pub_order_ring.sh`、原始输出 `harness/pub_order_ring.out.txt`
  （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2）。核心结论：
  - **§A sm90 序画像 = store→store 保序、store→load 可绕过**：MP litmus relaxed **0/70M**；
    正对照 SB litmus relaxed **100%（5M/5M×2）**（证明试纸有效）。MP relaxed 1497.5 vs rel+acq
    2226 ns/trial（一对 fence ~+728 ns）。
  - **§B「发布先于写」自然时序 0 错、拉开窗口 100% 错**：`pub`（= §35/§34 真实程序序）/`half` 百万槽
    全 0 错（producer 抢先）；`pubslp`（人为拉长窗口 ~2.2µs）**20000/20000、2000/2000 全错**（检测器有效）
    ——故 §34/§35「没出事」是**时序侥幸非形式保证**；正确序 `wro`（无 fence）与 MPK 协议 `rel` 全 0 错。
  - **§C release/acquire ≈ +0.36 µs/槽（恒定）**：`PL=1,G=2 +372.1`（1.33×）、`PL=1,G=34 +355.1`（1.92×）、
    `PL=256,G=34 +362.1`（**1.06×**）；解释 MPK 用「relaxed 写 + 单条 `atom.cas.release.gpu` 推游标」
    而非全程 fence。判据 C1–C4、仍差 C5 一步（与 §31–§35 同档）。**本轮新增实验。**
- 此前：**P46-1 完成（1/1）**。补掉 P45-1（§34）§34.9 明留的尾巴「`event` 的 ready 队列**无界**
  （按 `N+4` 开数组）、真实实现要**环形缓冲 + 容量上界**」：把 §34 的事件驱动调度装进一个**有界**
  ready 环形缓冲 + **阻塞式背压**，测「有界」付出什么代价。`docs/04-experiments.md` 新增 **§35**，
  源码 `src/bounded_event_ring.cu`、脚本 `harness/bounded_event_ring.sh`、原始输出
  `harness/bounded_event_ring.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
  `work_iters=20000`）。`unb`（=§34 无界）vs `bnd`（有界 `CAP` + 阻塞背压）。核心结论：
  - **§A/§B 两类容量下界别混用**：`unb` 恒可完成；`bnd` 需一个小容量，`G ≤ W_max` 时
    **`CAP_safe + G ≈ W_max + 1`**（`W_max`=最大层宽）——G 个 worker 在飞、环只缓冲宽层里
    `W_max−G` 个未被取走的就绪 task；`G ≥ W_max` 时触底 ~2。**与 §31–§34 的 `B(π)+1` 不同**：
    后者是**准入顺序/依赖深度**决定的图静态量，这里是**并发就绪数 ÷ worker 数**的排队量。
  - **§C 容量是安全阀不是吞吐旋钮**：`CAP<CAP_safe` 必死锁（守门狗），`CAP≥CAP_safe` 后延迟完全
    持平（0.375 ms，差 <1%）——区别于 §31 §C「容量是吞吐硬下界」。
  - **§D 正确性**：`unb` 与 `bnd(CAP=W_max+1)` 8/8 case **逐位相同 `sum`**（`rel` 1.5e-9–2.4e-9）。
  - **§E 规模标度**：`CAP_safe ≈ max(2, W_max − G + O(1))`；MPK 固定 8192/4096 + `assert`
    （`persistent_kernel.cuh:1790-1791/:1027`）、DeepGEMM `get_num_max_live_pool_blocks`
    （`scheduler/mega_moe.cuh:47-80`）都是「取远超峰值就绪数的固定常量保守兜底」。
  - **headline caveat**：靠近安全边界时**单次可完成、多次可能死锁**（`CAP_hard < CAP_safe`），
    边界是「环满 + 无 worker 在 pop 相」的时序竞态。判据：`unb`/`bnd` 均单常驻 kernel、设备侧
    推进（C1–C4），仍差 C5 一步（与 §31–§34 同档）。**本轮新增实验。**
- 此前：**P45-1 完成（1/1）**。补掉 P44-1 §33.8 明留的三条尾巴（`N≤70`、只验证 `G*` 与 `CAP*`
  同行相等即「未独立扫 `G`」、`gated` 是**朴素扫描**实现）：把 §33 的「依赖门控 `gated`」从**调度器
  扫描**改成 **MPK §5.1 的真机制**——每个 task 一个**事件计数器**（未完成前驱数），由完成该前驱的
  **worker 直接 push**「计数归零」的后继进 ready 队列。`docs/04-experiments.md` 新增 **§34**，源码
  `src/event_driven_pool.cu`、脚本 `harness/event_driven_pool.sh`、原始输出
  `harness/event_driven_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`、
  `work_iters=20000`）。四机制对照 `raw`/`scan`/`topo`/`event`。核心结论：
  - **§A 三机制数值逐位一致**（8/8 case 含不规则/乱序 `rel=0.0e+00`）；`event` 的「完成即 push」不动数值。
  - **§B 死锁边界**：`raw` 复现 `B(π)+1`（`reg,scr=1` `B=16` → `CAP=1` 死锁、`CAP=17` 活）；
    `scan`/`topo` 恒 `B≡0`；**`event` 更是 `CAP=1` 且 `G=1` 即活**（ready 队列只装就绪 task）。
  - **§C 更大 `N`（补 `N≤70`）**：`N` 64→4096，`scan/event` 1.3–3.0×（扫描随 `N` 变慢），
    `topo≈event`（`topo/event` 0.95–0.98×）——两者都不扫描。
  - **§D 独立扫 `G`（补 §33.8）**：不规则 `shrink,scr=1` 下 **`raw CAP* = raw G* = B+1`**（6/30/126；
    精扫 `N=122,B=29`：`G=28/29` 死锁、`G=30` 活 → 须 `G>B`）；**`event G*≡1`**——有界池死锁在事件
    驱动下根本不存在。
  - **§E 扇入 `FIN`**：`FIN 2→16` 抬高 `scan/event` 到 **2.4–3.5×**；`event` push 代价近持平。
  判据：四机制单常驻 kernel、scheduler/worker 角色分离（C1–C4），`event` 最接近 C5 但仍把 scheduler
  角色留在 block 0，**与 §31–§33 同档、仍差 C5 一步**。**本轮新增实验**；把 §31–§33 的防死锁理论
  从「调度器扫描/预排序」推进到「依赖事件天然触发」的 MPK 机制。
- 此前：**P44-1 完成（1/1）**。补掉 P43-1 §32.7 明留的「依赖模型是规则分层、**未测不规则/跨层
  依赖、多级扇入扇出混合**」尾巴：把 §32 的规则分层 DAG 推广到**不规则 DAG**（每层宽度不等 / 跨层
  skip 前向边 / 多级扇入 / **层号乱序**），并给出死锁边界的普适定义。`docs/04-experiments.md` 新增
  **§33**，源码 `src/irregular_pool.cu`、脚本 `harness/irregular_pool.sh`、原始输出
  `harness/irregular_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  `work_iters=20000`、`G=64`）。核心结论：
  - **普适量 `B(π)`**（贪心模拟的峰值池占用）——沿准入 order 逐个准入、立刻排空任何「前驱全 done」的
    就绪 task；实测 **`CAP*=G*=B(π)+1`**。§31 的 `Kc`、§32 的 `(L-1)*W` 都是其特例；且 `B` **不能**
    简化成「第一个就绪 task 的位置」（源在前、阻塞在后的形态会漏算）。
  - **§A 不规则宽度**：闭式 `(L-1)*W0+1` 恒 49、给出错值；真边界 `B`=54/42/32 → `CAP*`=55/43/33，
    `CAP*=G*` 逐行相等。
  - **§B 前向 skip/多级扇入**：前驱只在更浅层 → 层号升序仍是拓扑序，`level`/`topo` `CAP=1` 安全；
    `B(π)=N-W0` 与 `span` 无关。
  - **§C 层号乱序（headline）**：层号升序**不再是拓扑序** → `B_level` 骤升、小池死锁；真拓扑序
    `topo`（Kahn）恒 `B=0`。「按名义层/op 名遍历」≠「拓扑线性化」（对上 MPK `docs/08` §5）。
  - **§D/§E**：`gated` 扫描代价为 `topo` 的 **1.65–2.51×**（朴素扫描的工程代价）；warmup 判据是
    「**剩余阻塞 `B'<G`**」而非「warm `D` 层」。
  判据：四策略均 C1–C4、worker 无固定角色，**仍差 C5**（与 P42-1/P43-1 同档）；正确性 `rel=0.0e+00`。
  **本轮新增实验**；把 §31/§32 的防死锁理论收口到一般 DAG。
- 此前：**P43-1 完成（1/1）**。补掉 P42-1 §31.6 明留的「未测**多级扇入扇出 + 有界池**组合」尾巴：
  把 §31 的**单层 fan-in** 死锁模型推广到**分层依赖 DAG**（`L` 层 × 每层 `W` 个 task × 上一层 `FIN`
  扇入），在有界 ready 池上对照四策略（`wide` 层降序 / `level` 拓扑层序 / `warmup(D)` 多级 warmup /
  `gated` 依赖门控）。`docs/04-experiments.md` 新增 **§32**，源码 `src/layered_pool.cu`、脚本
  `harness/layered_pool.sh`、原始输出 `harness/layered_pool.out.txt`（目标卡 H100/sm90、**固定空闲
  device 3**、nvcc 13.2、`work_iters=20000`）。核心结论：
  - **死锁边界 = 阻塞数 `B=(L-1)*W`**：资源分量各自富余时 `CAP*` 与 `G*` **都精确 = `B+1`**（§A2）；
    与扇入宽度 `FIN` **无关**（§E），把 §31 单层的 `Kc` 推广成「完整依赖深度上累积的阻塞 task 数」。
  - **从「池容量单约束」升级为 `(CAP,G)` 联合约束**（§A3）：`L=8`（`B=112`）须 **`CAP>B` 且 `G>B`**
    同时成立；`G=#SM` 固定时单靠加大 ring 救不了，必须先 warmup/拓扑序把 `B` 压到 `<G`。
  - **warmup 深度须覆盖完整依赖深度**（§B，`D≈L-1`，对照 §31 的「宽度须 `W=M`」）。
  - **两种零成本正解**（§C）：`level`/`gated` 任意 CAP/G 无死锁；`level` 吞吐无损最快，`gated` 的
    朴素扫描实现大 CAP 慢 **~2.2×**（真实 event→task 直接映射不付扫描账，对上 MPK `docs/06` §3）。
  - **预测被实测推翻**：初版只算池容量（`B+1`），`L=6/8` 在 `G=64` 下实测任何 CAP 都死锁，才发现
    第二约束是 **worker 被阻塞 task 占满**（§A3 联合平面定量）。
  判据：四策略均 C1–C4、worker 无固定角色，**仍差 C5**（与 P42-1/P15-1/P22-1 同档）；正确性
  `rel=0.00e+00`。**本轮新增实验。**
- 此前：**P42-1 完成（1/1）**。把 `docs/05` §2.3「防死锁」checklist 里**只作资料转引**的一条
  做成实测：megakernel 的 in-kernel scheduler 用**有界 ready 池**时，「**准入顺序**」与「**池容量**」
  如何共同决定**死锁**与**吞吐**。`docs/04-experiments.md` 新增 **§31**，源码 `src/ring_deadlock.cu`、
  脚本 `harness/ring_deadlock.sh`、原始输出 `harness/ring_deadlock.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、`work_iters=20000`）。一手核对 DeepGEMM(`057ca59`)
  `scheduler/mega_moe.cuh`：`get_num_l1_warmup_waves()`（`:15-45`）、`get_num_max_live_pool_blocks()`
  （`:47-80`）、`l2_full_mask` 奇偶 generation 依赖 ring 容量（`:140-141,153-164`）、运行期
  `get_next_task()` 返回 L2 前等 L1 取走（`:398-403`）。模型：单常驻 kernel（block 0 = scheduler、
  其余 = worker），`M` m-block × `Kp` producer + `Kc` consumer（fan-in），有界池 `CAP` + 顺序
  `naive`（consumer 先）/`warmup(W)`（先发前 W 块 producer）。核心结论：
  - **死锁边界 = 首个 m-block 的 consumer 数 `Kc`**：`M=32,Kp=8,Kc=4` naive 在 `CAP≤4` 死锁、
    `CAP≥5` ok；`warmup(W=M)` 任意 `CAP` 全 ok（§A）。
  - **warmup 必须够宽**：`CAP=4` 下 `done=12·W`，`W<M=32` 全死锁、`W=M` 才全解（§B）。
  - **边界随 `Kc` 平移**：naive 死锁阈值 `Kc=2→≤2`、`Kc=4→≤4`、`Kc=8→≤8`（§E）。
  - **容量是吞吐硬下界**：即便不死锁，`CAP<G` 时 worker 饿着——`G=64` 时 `CAP=2` 慢 **27.6×**、
    `CAP≥G` 饱和（§C），正对应 `get_num_max_live_pool_blocks` 的 "max live"。
  - **worker 标度**：`G=1→18.15 ms`、`G=128→0.207 ms`（5.58× vs G=16），`G>32` 转次线性（§D）。
  判据：`warmup`/`naive` 均 C1–C4、worker 无固定角色，**仍差 C5**（与 P15-1/P22-1 同档）；是
  megakernel 的**正确性/容量成本侧**，与 P14-1/P16-1/P17-1 资源足迹互补。正确性 `rel=-1.84e-10`。
  设备侧 `clock64` 无进度守门狗使死锁 case 也正常返回（`ms≈505` 的行即守门狗耗时）。**本轮新增实验。**
- 此前：**P41-1 完成（1/1）**。补齐 `docs/04` §29.6（P40-1）明确留的「只测单窗口、单 stream、
  单一 persisting 段，**未测多窗口、多 stream 各自窗口、`hitRatio<1` 的概率化分区**」尾巴，把「L2 常驻」
  从「能不能用」推进到「**怎么配才不亏**」，直接对应 megakernel「整图很多中间量同时活着、争同一个
  carveout」的处境。`docs/04-experiments.md` 新增 **§30**，源码 `src/l2_persist_tuning.cu`、脚本
  `harness/l2_persist_tuning.sh`、原始输出 `harness/l2_persist_tuning.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、iters=30，X=Z=256 MB 流式、polluter=512 MB、每段 8–16 MB，
  L2=50 MB / maxPersist=31 MB）。核心结论：
  - **零和代价的开关是 carveout 大小而非 hitRatio**：`f=0.00`（挂窗但全 Streaming = 只划 carveout 不保护）
    随 carveout 单调恶化 **4 MB→0.996、8 MB→0.981、16 MB→1.204、31 MB→1.921**；`pfull` 的 ~1.5× 惩罚
    **无法用 hitRatio 收回**（31 MB 行 f=0→1.921 仅降到 f=1→1.504）。
  - **正确姿势：carveout ≈ 中间量实际大小、配得上时 f 取高**——最优 `carveout=16, f=1.0 → 0.860×`；
    单窗口 ncu 亦证 carveout=8（=S）比 16/31 保护更好（S=8：6.05 vs 6.42/6.96 MB vs 无窗 8.40）。
  - **`hitRatio<1` 的唯一价值场景**：carveout 被压到 < 需求时救火——两个 8 MB 窗口、**carveout=8 < 总需求
    16、f=1.0 → 1.026×（thrash 崩）**，把 **f 降到 0.5 → 0.879×（反超）**。
  - **多窗口保护可叠加但要求 carveout ≥ 同活中间量总量**：`carve=16 both f=1.0 → 0.814×`；而
    `carve=31 both f=1.0 → 1.343×`（过度预留盖过保护）。保护上限仍 ~30%。
  - **口径 caveat**：多 kernel + 多流下 ncu replay 的逐 kernel DRAM 读与自然计时背离（ncu 显示只有 stA
    侧受保护、自然计时两个 K2 都加速、单 stream 又证明 stB 完全能保护），**多窗口 split 归因以自然计时为准**。
  判据不变（kernel-per-operator，谈落点）。**本轮新增实验。**
- 此前：**P40-1 完成（1/1）**。兑现 `docs/04` §17.7（P24-1）明确留的假设「若张量能整块驻留
  L2（≤50 MB），eager 的中间量往返可能命中 L2 而不落 DRAM——损失三在工作集超过 L2 时才充分显现」，
  把「中间量的内存落点」从已有的**寄存器**（P5-2 融合）/ **SMEM+DSM**（P29-1）/ **HBM**（P24-1）
  补上缺失的 **L2** 一档。`docs/04-experiments.md` 新增 **§29**，源码 `src/l2_residency.cu`、脚本
  `harness/l2_residency.sh`、原始输出 `harness/l2_residency.out.txt`（目标卡 H100/sm90、**固定空闲
  device 3**、nvcc 13.2、iters=30，X=Z=256 MB 流式、polluter=512 MB、中间量 `y` 尺寸 S=4..64 MB，
  H100 L2=50 MB / maxPersistingL2=31 MB）。序列 `K1(读x→写y)→polluter(512MB 冲刷)→K2(读y→写z)`，
  四模式 `base`/`pollute`/`persist`(carveout=min(S,31MB))/`pfull`(carveout=max)。核心结论：
  - **中间量在不在 L2 由「写读之间流过什么」决定，不是尺寸单独决定**：`base` 在 S≤16 MB 时 K2
    几乎全命中 L2（DRAM 读 6.7 KB–1.7 MB、命中 90–96%），插 512 MB polluter 立刻涨到**恰好 = S**、命中 52%。
  - **`persisting` 只部分保护且零和**：`persist` 相对 `pollute` 把 K2 的 DRAM 读降 **~23%（S=4）–30%
    （S=8/16）**；S≥32 MB 超过 31 MB carveout 即**反噬**（计时到 **2.20×**），`pfull` 拉满 carveout
    即便 S 小也稳定慢 **~1.5×**——persisting 是「普通 L2 ↔ 常驻 L2」的零和交易，应按中间量大小配。
  - `base` 的「免费 L2」边界：S=32 MB 起退化、S≥48 MB 等价 `pollute`。判据不变（kernel-per-operator，
    谈落点）；`base`/`persist` 的 `z` 逐位一致。踩坑：ncu 默认 `--cache-control=all` 会 replay 前 flush
    L2，须 `--replay-mode application --cache-control none`。**本轮新增实验。**
- 此前：**P39-1 完成（1/1）**。补齐 `docs/04` §27.6 明确留的「**未测 MPS**（Multi-Process
  Service）作为另一条隔离路线」尾巴，把 §27 的**进程内**共存（两条流 / green context）推进到**跨进程**
  维度：无 MPS 时不同进程的 kernel 不能真并发（驱动时分/抢占），MPS 才允许跨进程真并发并可用
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=P` 设「线程容量 / 可见 SM 上限」。`docs/04-experiments.md` 新增
  **§28**，源码 `src/mps_isolation.cu`、脚本 `harness/mps_isolation.sh`、原始输出
  `harness/mps_isolation.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、buf=256 MB、
  victim=20 趟求和、persist=40000 rounds）。两个**独立进程**对照。核心结论：
  - **无 MPS**：victim **1.792 ms ≈ 基线 1.781 ms（1.006×）**、persist 只慢 1.4%——跨进程 kernel
    **不共驻**、驱动时分/抢占，victim 拿整机时间片；**与 §27 进程内两条流的 134× 饿死完全相反**。
  - **MPS 默认（各 100% 真并发）**：victim **1.55× 基线**，persist 带宽几乎不变（3034 GB/s）。
  - **`ACTIVE_THREAD_PERCENTAGE=P` 是硬线程容量上限**：按 P 缩放该进程可见 SM 数（100→132、90→118、
    75→98、50→66、25→32、10→13）。
  - **反直觉：干扰峰值在「cap 略低」**——victim slowdown 对 **P≥50 精确服从 `≈1/(1−P)`**：P=90→**9.90×**、
    75→3.98×、60→2.80×、50→2.01×；**留 10% 容量就把 victim 饿死 ~10×、比不设 cap（1.55×）还糟 ~6 倍**；
    cap 越低越恢复（P=10→1.47×）。persist 带宽随 cap **次线性**（3035→509 GB/s）。
  - **三路线**：Green Contexts（§27）victim **0.76× 基线**、确定、双向；MPS 最坏 **9.9×**、非确定；
    无 MPS victim≈基线但**非并发**。**要确定性隔离用 Green Contexts/独占；MPS 只保证能并发、不保证
    公平**——给整机 megakernel 做 MPS 隔离时 cap 应设得**明显低**（P≤25）。
  判据：`persist` C1/C2；MPS 是进程/驱动层约束（C4 的更粗容器），均不涉 C3/C5。踩坑：MPS client 设
  `CUDA_VISIBLE_DEVICES` 会连不上 server。**本轮新增实验，全专题任务清单仍全部完成。**

- 此前：**P38-1 完成（1/1）**。把「megakernel = 单次 launch **占满整机**」这一被前面所有小节默认
  接受的前提**翻到成本面**：整机持久化内核对**共存负载**（另一模型/租户）的干扰，以及 Hopper
  **Green Contexts** SM 分区能否硬隔离——补齐 MPK §5「worker/scheduler 分配固定在 launch 时、匹配物理
  SM 数」（快照行 1168）的**部署含义**，把 P10-1（占用率上限）/P14-1（占用率代价）从「单 kernel 内」
  推进到「跨 kernel」维度。`docs/04-experiments.md` 新增 **§27**，源码 `src/green_ctx_sched.cu`、脚本
  `harness/green_ctx_sched.sh`、原始输出 `harness/green_ctx_sched.out.txt`（目标卡 H100/sm90、
  **固定空闲 device 3**、nvcc 13.2、buf=256 MB、rounds=64，device 1 复现同量级）。四 Section 核心结论：
  - **A 干扰**：persist 铺满 132 SM 但每 SM 仅 1/8 槽位（Gp=132）→ victim **1.0× 基线**（可共驻）；
    Gp=1056（占满整机）→ victim 被饿死 **134.5× 基线**（等满 persist 的 5.59 ms）——**干扰的开关是
    「剩余占用率槽位」而非「占了几张 SM」**（对上 P10-1/P14-1）。
  - **B Green Contexts 分区**：132 SM 切 A=64/B=68（`cudaDevSmResourceSplit` + backfill）→ B1 隔离下
    victim **0.76× 基线**、persist 自身 7.56 ms ≈ 单独跑 7.55 ms（**双向隔离**）；B2 对照（victim 放
    A 的同分区另一流）又饿死 **184.3×**——**隔离来自分区本身**。
  - **C 流优先级**：persist 低优 / victim 高优 → 仍 **133.65×**——**优先级不抢占已运行 block**。
  - **D 分区代价**：persist 吞吐随 SM 近似线性（32→1335、64→2277、96→2821 GB/s）——**给共存租户留 SM
    = 按比例削减 megakernel 可达带宽**。
  判据：均单 kernel、`persist` C1/C2；Green Contexts 是 **C4 的更粗一层**（launch 前定死 SM 归属，
  与 P29-1 cluster 片内分区同目标），均不涉 C3/C5。踩坑：H100 的 132 SM 与 `smCoscheduledAlignment=8`
  不整除，第二组须 `cudaDevSmResourceGroupBackfill`。**本轮新增实验，全专题任务清单仍全部完成。**

- 此前：**P37-1 完成（1/1）**。补齐 `docs/04` §1.6（P4-1）留下的判词「**device graph launch 仍停在
  kernel 级**」——把「设备自己发图」这条**最直接的设备侧派发**路线量出来，并与 P36-1（§25）的
  `kinternal`/`cwhile` 对照。`docs/04-experiments.md` 新增 **§26**，源码 `src/device_graph.cu`、脚本
  `harness/device_graph.sh`、原始输出 `harness/device_graph.out.txt`（主配置 N=20000×reps=5）与
  `harness/device_graph_sens.out.txt`（敏感性 N=2000×reps=20）（目标卡 H100/sm90、**固定空闲 device 3**、
  nvcc 13.2、`-arch=sm_90a -rdc=true -lcudadevrt`）。四策略公平对照 **eager**（host 发 kernel）/
  **hgraph**（host 发 1-node 图）/ **srelaunch**（device-launchable 图自重启 = 设备派发）/
  **kinternal**（常驻内核内循环）。核心结论：
  (A) **两条实测约束**——调用 `cudaGraphLaunch` 的 kernel 直接 `<<<>>>` 启动 → **`operation not
  supported`**（发起者必须是 graph 节点）；host-graph 节点内对**同一 exec 连发两次** → 第 2 次
  **`invalid argument`**（单实例在飞，故循环只能走**自重启图**）。
  (B) **设备侧发图不比 host 便宜**：每单元 `eager` 2078.5 / `hgraph` 2370.7（1.14×）/ **`srelaunch`
  4285.4（2.06×）** / `kinternal` **39.3 ns（0.019×）**；敏感性逐项复现（2105.9/2394.8/4294.1/40.0），
  `sink` 全对（=100000）。
  (C) **单次 device 图启动往返 4128 ns ≈ `srelaunch` 每行程 4.29 µs**——成本就是设备图启动本身。
  **含义**：device graph launch / `cwhile`（§25，4.6 µs）/ 自重启图都落在**同一个 ~4 µs 的「设备侧
  kernel/图重分发」地板**，而 in-kernel 边界只要 ~40 ns；**设备能发图 ≠ 设备发图便宜**，要把单价压到
  ~40 ns 只能**留在同一个常驻 kernel 内**（MPK in-kernel event queue / DeepGEMM ring barrier）。
  判据：四策略均不满足 C1–C5（`srelaunch` 由设备发起但粒度仍是整个图）。**本轮新增实验**；与
  §24/§25 合成「CUDA Graph 降级路线天花板」的第三块（设备侧派发）。
  此前 **P36-1 完成（1/1）**。补齐 CUDA Graph 的**控制流**维度——CUDA 12.3+ 的
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
- **下一步：全专题继续（P0–P37 全部完成）**。P37-1 已把 P4-1 §1.6 的「device graph launch 仍停在
  kernel 级」补成实测（`docs/04` §26：设备侧自重启图 4.29 µs/单元 ≈ 2× host kernel launch、单次
  device 图启动往返 4.13 µs，与 §25 条件节点 4.6 µs 同落 ~4 µs 地板；in-kernel 循环仅 ~40 ns；两条
  约束=发起者须为 graph 节点 / 同一 exec 单实例在飞）。P36-1 已把 CUDA Graph 的**控制流**维度补成实测
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
     **至此「下一步」剩余可选方向全部完成**；随后 **P36-1/P37-1/P38-1/P39-1** 又各自补掉了
     `docs/04` 明留的尾巴（条件节点 `cwhile`、device graph launch、Green Contexts 分区、MPS 跨进程
      隔离），**全专题无遗留必做项**。此后 **P40-1/P41-1/P42-1/P43-1** 又依次补掉：L2 常驻（中间量第三档
      落点）、L2 persisting 调参/多窗口、**有界 ready 池 + warmup wave 的防死锁/容量下界**、
      **有界池 × 多级依赖 DAG**（`docs/04` §29/§30/§31/§32）。
      **P42-1 尾巴中的「多级扇入扇出 + 有界池」已由 P43-1 补掉**（`docs/04` §32：边界 = 阻塞数
      `B=(L-1)*W`、`(CAP,G)` 联合约束、warmup 深度 ≈`L-1`、拓扑序/门控零成本防死锁）。
      **P43-1 尾巴中的「不规则/跨层依赖、多级扇入扇出混合」已由 P44-1 补掉**（`docs/04` §33：
      普适量 `B(π)`=贪心峰值池占用、`CAP*=G*=B(π)+1`、闭式 `(L-1)*W` 只是等宽特例、
      **层号乱序时「层号升序」不再是拓扑序**→须真拓扑线性化）。
      **P44-1 新留尾巴**（见 `docs/04` §33.8）：`N≤70`、`scramble` 只是层号整体旋转（非任意图重编号）、
      未测更大 `N`/与真实 DeepGEMM SM100 路径对跑、未独立扫 `G` 在不规则图上的边界；`gated` 仍是
      朴素扫描实现（其 ~1.7–2.5× 慢非语义固有）。其中「更大 `N`」「独立扫 `G`」「`gated` 是扫描」
      三条已由 **P45-1（`docs/04` §34）** 处理：把 `gated` 换成 MPK §5.1 事件驱动就地入队，`N` 扩到
      4096、独立扫 `G` 确认 `raw CAP*=G*=B+1` 而 `event G*≡1`；仍余「与真实 DeepGEMM SM100 路径对跑」
      （SM100 专属，本机不具备）。**P45-1 新留尾巴**（`docs/04` §34.9）：`event` 的 ready 队列用的是
      **无界**数组（`N+4`），真实实现要**环形缓冲 + 容量上界**——已由 **P46-1（`docs/04` §35）** 处理：
      给事件驱动加上**有界环形缓冲 + 阻塞背压**，量出 `CAP_safe + G ≈ W_max + 1`（排队量，非 §31–§34 的
      静态 `B(π)+1`）、安全容量以上不加吞吐、边界不确定；仍余「与真实 DeepGEMM SM100 路径对跑」。**P46-1
      新留尾巴**（`docs/04` §35.7）：入队内存序是「先发布 tail 后写数据」（未加 release 发布）；
      真实实现用 release 语义做实（MPK `docs/06` §3）——已由 **P47-1（`docs/04` §36）** 处理：实测 sm90
      **store→store 保序**（MP litmus 0/70M）、**store→load 可绕过**（正对照 SB 100%）；「发布先于写」
      自然时序 0 错但拉开窗口 `pubslp` 100% 错（检测器有效、§35「没出事」是时序侥幸）；正确序 `wro`
      无 fence 全对、MPK 协议 `rel` 全对，release/acquire 代价 ≈ **+0.36 µs/槽**（大 payload 摊薄 ~1.06×）。
       **P47-1 新留尾巴**（`docs/04` §36.6）：litmus 只测 2 block×1 线程（未扫 SM 距离/`ncu` 内存序事件、
       未给重排概率分布），环未测多生产者（MPSC）/真实 task 描述符/与 DeepGEMM SM100 路径对跑。
       其中「多生产者（MPSC）」已由 **P48-1（`docs/04` §37）** 处理：把 SPSC 推广到 MPSC，实测
       **MPK 连续 CAS 放行 / 逐槽 release 在多生产者下全对**，而「非连续完成计数 / 占位即就绪」在决定性
       正对照 `stall0` 下必错（bad=8），「自然时序 0 错」只是侥幸；放行代价 ≈ +0.38 µs/项（与 §36.4 一致），
       并指出 **MPK 的两类队列对应两种协议（MPSC→连续 CAS、SPSC→单条 release）由生产者个数决定**。
        **P48-1 新留尾巴**（`docs/04` §37.7）：消费者单线程致每项 ~2.3µs 往返主导（§C 只看差值）、
        `stall0` 为人为扰动、未做真实 task 描述符、litmus 仍未扫 SM 距离/`ncu` 内存序事件、未与
        DeepGEMM SM100 路径对跑。**P49-1（§38）/P50-1（§39）/P51-1（§40）** 续接取 task 通路
        （批量出队+多线程消费者+真实描述符 / `cp.async` 载体 / 多 worker 带宽标度 + TMA/mbarrier），
        最新 **P52-1（§41）** 补掉 P51-1 §40.8 的「K × SMEM 联合约束」：**环深 D≥2 无益、批量 K 才是
        主旋钮**（大 K + D=2 最优，K=128/D=2 达 2667 GB/s），取 task 的 SMEM 是**软约束**（换 worker
        数、总带宽守恒）。仍余「未跑 16 B `cp.async` 深环 / 无动态生产者 / `cp.async` 释放寄存器对
        下游 task 的收益 / 与 DeepGEMM SM100 对跑」。其中两条已由 **P53-1（§42）** 补掉（16 B
        `cp.async` 深环与 bulk 对跑、`cp.async` 不省寄存器）；**P54-1（§43）** 把「取 task」与「执行
        task」放进同一环（预取完全藏住取数）；**P55-1（§44）** 给执行接上吃 HBM 的数据面（未饱和区间
        仍不争）；**P56-1（§45）** 把数据面推到 **HBM 饱和**（`T=1024` 达 ~2983 GB/s），确认控制面
        **仍被藏住**（`p/max` 1.015–1.132），并量出「控制面要少线程、数据面要多线程」这对相反最优；
        **P57-1（§46）** 用多 block/SM 化解该矛盾（数据面认「总线程/SM」、控制面认 `K`）；
        **P58-1（§47）** 把「未叠动态生产者」补上（动态须落在**调度**、不能落在**生成**）；
        **P59-1–P62-1（§48–§51）** 把动态生产者标度到 P 块 / 向量化 / TMA store / 窗口握手；
        **P63-1（§52）** 用正交编码 + 逐相位 + SASS 修正 §51 的「只拆一个就够快」（编码假象），
        钉死写端瓶颈 = **每批 `st.release` 的 GPU 级 MEMBAR + 会合 barrier**；
        **P64-1（§53）** 换写载体（TMA / 寄存器 copy-out / 寄存器直写）验「比值不可外推」——
        分解同形、**握手税近似绝对（~2.0–2.4k cycles/batch）故占比随地板抬升而变大**（29.9%→36.8%），
        寄存器 store 只在直写不经 SMEM 时优于 TMA。
        **P65-1（§54）** 把写端握手归因放回**有消费者**的端到端流水（有界环绑定时背压×容量乒乓
        主导 ~20×、盖过内存序税；不绑定时 §53 归因复现但衰减到 ~1.1–1.2×）。
        **P66-1（§55）** 放开 §54 里被钉死为 `D=1` 的「worker 取 task」维度——**`D` 不是自由性能
        旋钮而是被环协议锁死的自由度**（纯性能上限 `D=2` 已饱和 / 端到端被写侧盖住 / 动态生产者 +
        有界环下 strided `D>1` 死锁、contig 静默错；唯一正确形态 strided+D=1）。
- **本轮（P50-1）**：`docs/04-experiments.md` 新增 **§39**「worker 侧取 task 的载体：`cp.async` 批量
  加载 TaskDesc 值多少」——新源码 `src/cpasync_desc_fetch.cu`、脚本 `harness/cpasync_desc_fetch.sh`、
  原始输出 `harness/cpasync_desc_fetch.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、
  单 block、256 MB ≫ L2）。补 §38.7「未实测 `cp.async` 批量加载本身收益 / `warp` 下 `cp.async` 形态」：
  四取数路径 `scalar`/`vec4`/`cpasync`（SMEM 单缓冲）/`cpasync_pf`（双缓冲预取），描述符 `DW∈{8,16,32,88}`、
  `DW=88`=**352 B 对齐论文 §6.1**。核心结论：**(1) 四路径全 `bad=0`**；**(2) `cp.async` 的成本与描述符
  大小无关——真实 352 B 下 `cpasync` 58.0 vs `vec4` 132.9（2.29×）、vs `scalar` 167.1（2.88×）**，小描述符
  （≤128 B）时反而略慢于 `vec4`；**(3) `cp.async` 价值集中在任务稀疏（`K` 小）**：`K=1` pf 474.5 vs vec4
  1030.3（2.17×），`K≥32` 则 `vec4` 反超；**(4) 双缓冲预取相对单缓冲稳定收回 7%–40%**（= 论文 §5.3
  pre-fetch；快照 worker 却是单缓冲 `wait<0>`）。判据 C1–C4、无 C5。**本轮新增实验**；收口 §38.7 载体线。
- **本轮（P49-1）**：`docs/04-experiments.md` 新增 **§38**「MPSC 批量出队 + 多线程消费者 + 真实描述符」——
  新源码 `src/mpsc_batch.cu`、脚本 `harness/mpsc_batch.sh`、原始输出 `harness/mpsc_batch.out.txt`（目标卡
  H100/sm90、**固定空闲 device 3**、nvcc 13.2、grid=P+1）。补 §37.7「单线程消费者 ~2.3µs/项往返」
  与「未做真实 task 描述符」两条尾巴：消费者三档 `serial`/`batchK`/`warpK`、描述符 `DW∈{8,16,32}`（真实
  `TaskDesc` 128 B = `runtime_header.h:296-336`）、生产者协议 = §37 `mpk`（`persistent_kernel.cuh:1206-1228`）。
  核心结论：**(1) 批量/多线程 8/8 case `bad=0`**（连续 CAS 保证批量读安全）；**(2) 单线程批量只收回「放行
  往返」1.2–1.5× 封顶**（DW 越大越少）；**(3) warp 并行读描述符在 128 B 下快 21×**（`serial` 随描述符
  ~170 ns/字线性变贵、`warp` 恒 ~300 ns/item，§D 拆因证明分水岭是「能否并行读 payload」）；**(4) warp
  把瓶颈从消费者翻到生产者**（P=8 时 303 ns/item 接近生产者侧成本）。判据 C1–C4、仍差 C5 一步。
  **本轮新增实验**；收口 §37.7 两条尾巴。
- **本轮（P44-1）**：`docs/04-experiments.md` 新增 **§33**「有界 ready 池 × 不规则依赖 DAG」——
  新源码 `src/irregular_pool.cu`、脚本 `harness/irregular_pool.sh`、原始输出
  `harness/irregular_pool.out.txt`（目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、`G=64`）。
  把 §32 的规则分层 DAG 推广到不规则 DAG（不等宽 / 跨层 skip / 多级扇入 / 层号乱序），给出边界
  普适量 `B(π)`（贪心峰值池占用）：**`CAP*=G*=B(π)+1`**，§31/§32 的 `Kc`/`(L-1)*W` 是其特例；
  闭式在不规则宽度下失效（§A：49 → 55/43/33）；前向 skip 不改 `B`、层号升序仍拓扑（§B）；**层号乱序
  让层号升序不再是拓扑序**（§C：`level` 死锁、`topo` 恒 `B=0`）；`gated` 扫描代价 1.65–2.51×（§D）；
  warmup 判据是「剩余阻塞 `B'<G`」（§E）。判据 C1–C4、仍差 C5。**本轮新增实验。**
- **本轮（P40-1）**：`docs/04-experiments.md` 新增 **§29**「L2 常驻：损失三的第三档中间量落点」——
  兑现 §17.7（P24-1）的假设，把「中间量的内存落点」补上 **L2** 这一档（寄存器/SMEM+DSM/HBM 之外）。
  源码 `src/l2_residency.cu`、脚本 `harness/l2_residency.sh`、原始输出 `harness/l2_residency.out.txt`
  （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2、iters=30）。序列 `K1→polluter(512MB)→K2`、
  四模式 `base`/`pollute`/`persist`/`pfull`、ncu 取 K2 的 DRAM 读字节 + L2 命中率。核心结论：
  (1) 中间量的 L2 常驻由「写读之间流过什么」决定而非尺寸——`base` S≤16 MB 命中 90–96%、插 polluter
  立刻涨到 = S（命中 52%）；(2) `persisting` 只降 DRAM 读 **~23–30%**、且超 31 MB carveout 即**反噬**
  （**2.20×**），拉满 carveout（`pfull`）即便 S 小也慢 **~1.5×**——零和交易；(3) `base` 免费 L2 边界
  在 S≈32 MB。含义：整图融合里「靠 L2 留中间量」最不可依赖，解释了 MPK §5.3 用 paged SMEM。
  踩坑：ncu 默认 `--cache-control=all` 会 flush L2，须 `--replay-mode application --cache-control none`。
  **本轮新增实验。**
- **本轮（P39-1）**：`docs/04-experiments.md` 新增 **§28**「多租户共存隔离的第二条路线：MPS」——
  新源码 `src/mps_isolation.cu`、脚本 `harness/mps_isolation.sh`、原始输出 `harness/mps_isolation.out.txt`
  （目标卡 H100/sm90、**固定空闲 device 3**、nvcc 13.2）。补 §27.6「未测 MPS」尾巴，把共存从「进程内
  两条流」推进到**跨进程**：`persist`（整机访存持久化）与 `victim`（求和）两个独立进程，对照
  无 MPS / MPS 默认 100% / MPS `ACTIVE_THREAD_PERCENTAGE=P`。核心结论：(1) **无 MPS 跨进程 kernel
  不共驻**——victim 1.006× 基线、persist 只慢 1.4%（驱动时分/抢占，与 §27 进程内 134× 饿死相反）；
  (2) **MPS 真并发**默认 1.55×；(3) **`ACTIVE_THREAD_PERCENTAGE` 缩放可见 SM 数**（100→132…10→13）；
  (4) **干扰峰值在 cap 略低**：P≥50 时 slowdown `≈1/(1−P)`（P=90→9.90×、75→3.98×、50→2.01×），
  **留 10% 容量饿死 victim ~10×、比不设 cap 还糟 ~6×**；persist 带宽随 cap 次线性（3035→509 GB/s）；
  (5) 三路线对比——Green Contexts 0.76×/确定，MPS 最坏 9.9×/非确定，无 MPS ≈基线但非并发；
  **整机 megakernel 做 MPS 隔离时 cap 应明显低（P≤25）**。判据：`persist` C1/C2，MPS 是进程/驱动层
  约束（C4 更粗容器），不涉 C3/C5。踩坑：MPS client 设 `CUDA_VISIBLE_DEVICES` 会连不上 server。
  **本轮新增实验。**
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
