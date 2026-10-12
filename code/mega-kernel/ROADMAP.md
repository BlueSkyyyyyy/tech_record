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
- [ ] **P2-4** 评估（§6）：相对 PyTorch/vLLM/SGLang 的 1.0–1.7× 延迟、TP 下最高 10× 吞吐；
      复现要点与局限（模型/GPU 代际/batch）。（注：评估实际在 §6，非早期误记的 §5。）

### P3 生产案例（`docs/02-production-cases.md`）

- [ ] **P3-1** DeepGEMM **Mega MoE**：dispatch+GEMM+SwiGLU+combine 单 kernel、symmetric memory、
      ring buffer 调度（与 `mega-moe` 专题交叉引用，避免重复）。
- [ ] **P3-2** DeepGEMM **Mega Gate** 与 **Mega mHC**（HyperConnection）：各自融合了什么、为什么。
- [ ] **P3-3** Kimi K3 的 fused mega-kernels（RL 内核任务套件）、hpc-ops 的 Megakernel roadmap。

### P4 相关技术对照（`docs/03-related-tech.md`）

- [ ] **P4-1** CUDA Graphs：消除 launch 开销但仍有 kernel barrier；与 megakernel 的关系。
- [ ] **P4-2** PDL（Programmatic Dependent Launch）：跨 kernel 重叠；H100 实测。
- [ ] **P4-3** persistent kernel vs megakernel；in-kernel 调度（decentralized scheduling）。
- [ ] **P4-4** 与「通信-计算 overlap」（DeepEP / FlashMLA 双 kernel）的对比。

### P5 实验（`docs/04-experiments.md`）

- [ ] **P5-1** launch 开销实测：N 个小 kernel 串行 vs CUDA Graph vs 单 persistent kernel。
- [ ] **P5-2** 一个简单「两算子融合成单 persistent kernel」的最小示例 + 实测收益。
- [ ] **P5-3** PDL 小实验（可选）。

### P6 汇总

- [ ] **P6-1** `docs/05-summary.md`：原理一页纸 + 设计 checklist + 何时该/不该用 megakernel。

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

- 当前进度：脚手架就绪；**P0 全部完成**；**P1 全部完成（3/3）**；**P2 进行中（3/4）**。
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
- **下一步：P2-4** 评估（`docs/01-mpk-paper.md` §5）：§6 总览（四大问题）、§6.3 端到端
  （相对 vLLM/SGLang 的 1.0–1.7× 延迟、TP 下最高 10× 吞吐）、§6.4 动态负载、§6.5 多 GPU 扩展、
  §6.6 运行时消融（已在 P2-3 引用）、§6.7 编译期消融（已在 P2-2 引用）；给出复现要点与局限
  （模型/GPU 代际/batch）。纯资料分析。

## 阻塞

- （空）
