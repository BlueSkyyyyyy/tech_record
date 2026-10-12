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
  **本专题任务清单（P0–P7）全部完成**，无未完成子项。
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
- **下一步：无**（P0–P7 全部完成）。后续可选的增量方向（不属现有清单）：① 为 `docs/02`/`03`/`04`/`06`
  补建对应博客 post（目前仅 `docs/00`/`01`/`05` 有 post）；② 补做 P5-2 §2.6 / P5-1 §1.6 提到的
  「更小 M / 更多算子 / 多 block」实验；③ 在容器内 `pip install -e .` 构建 mirage sm90 路径并跑
  `demo/qwen3/demo.py --use-mirage` 取本机 MPK 端到端数字，或继续走读 `tasks/hopper/*.cuh` 单 task
  内部实现（P7-1 已走通运行层骨架，见 `docs/06`）。
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
