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
- [ ] **P0-3** 环境探测：`kernel_lab` 能否跑 persistent kernel / CUDA Graph / PDL 小实验（H100 支持 PDL）。

### P1 动机与背景（`docs/00-megakernel-overview.md`）

- [ ] **P1-1** kernel-per-operator 执行模型与三类损失：kernel barrier（跨算子无法软件流水）、
      launch 开销、中间张量 HBM 往返。
- [ ] **P1-2** 已有缓解手段与不足：算子融合、CUDA Graphs、PDL、多流；为什么还不够（跨算子细粒度重叠、SM 级调度）。
- [ ] **P1-3** MegaKernel 的定义与谱系（从单算子融合 → 段融合 → 整图单 kernel），术语澄清。

### P2 MPK 论文精读（`docs/01-mpk-paper.md`）

- [ ] **P2-1** SM 级图表示（§3）：以 SM 为粒度的数据依赖；与 CUDA Graphs 的对比。
- [ ] **P2-2** MPK 编译器（§4）：tGraph 生成、每个 task 的 CUDA 实现生成。
- [ ] **P2-3** in-kernel 并行运行时（§5）：去中心化调度、跨算子软件流水、计算-通信重叠。
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
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。

## 当前进度 / 下一步

- 当前进度：脚手架就绪；**P0-1 完成**（MPK 论文章节地图见 `docs/00` §6）；**P0-2 完成**——案例源码坐标清单
  写入 `docs/02-production-cases.md`：DeepGEMM mega 家族（`057ca59`）的 api/heuristics/impl/scheduler/layout
  文件与行号、hpc-ops Megakernel roadmap（`ab1a402` README 行 230）、Kimi K3 §4.2.4（快照行 1069–1081）。
- **下一步：P0-3** 环境探测：在 `kernel_lab`（H100）确认能否编译/运行 persistent kernel、CUDA Graph、PDL
  小实验（`code/mega-kernel/scripts/lab.sh exec "..."`），为 P4/P5 实验铺垫；跑不了的标注「仅资料分析」。

## 阻塞

- （空）
