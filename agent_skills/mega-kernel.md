# mega-kernel — MegaKernel 资料与实现分析的自驱流程

## 触发场景

用户说「继续 MegaKernel」「mega-kernel」「megakernel」。本仓库长期自驱任务之一。

## 前置条件

- 读 `agent_guide.md`（目录约定/坑，尤其：**不要写目标卡代号**、`doc_include` 内联渲染机制）。
- 读控制面板 `code/mega-kernel/ROADMAP.md`（每次先读，从「下一步」做起，做完更新）。
- 环境：复用 `kernel_lab` 容器（`code/mega-kernel/scripts/lab.sh` 指向 kernel-opt 的脚本）。

## 关键事实

- **主参考 MPK 论文**：*MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs*（Mirage Persistent Kernel，
  arXiv 2512.22219）；本机快照 `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`；实现 `mirage-project/mirage`（线上）。
- **生产案例**：`~/github/DeepGEMM` 的 mega 家族（Mega MoE / Mega Gate / Mega mHC）；
  `~/github/hpc-ops`（Megakernel roadmap）；`~/github/ai-infra-book/references/text/kimi-k3.txt`（fused mega-kernels）。
- 本专题**以资料分析为主**；H100 上能做的小实验（launch 开销 / persistent kernel / CUDA Graph / PDL）要跑真实数字。
- 与 `code/mega-moe/` 专题交叉引用但**不重复**：mega-moe 聚焦 DeepGEMM MoE 前反向；本专题聚焦 megakernel 范式与案例。
- 博客专题由 `code/mega-kernel/docs/*.md` **实时内联**渲染（`{{< doc_include "..." >}}`）。

## 步骤（每轮一个增量）

1. 读 ROADMAP「下一步」，选一个未完成子项（P0→P1→…）。
2. 写/改 `docs/<NN>-*.md` 与必要的 `src/`、`harness/`。
3. 论文贴真实章节号、源码贴真实行号；线上资料给真实链接（不编造 URL）。
4. 能跑的用 `scripts/lab.sh exec "..."` 跑真实数字；跑不了的标注「仅资料分析」。
5. 更新 `ROADMAP.md`；commit（`megakernel:` 前缀）→ push。
6. 不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。

## 验收标准

- [ ] 原理/机制讲清楚，论文章节号与源码行号真实核对，线上链接真实
- [ ] 能跑的部分有实测数字与原始输出；跑不了的部分明确标注
- [ ] 博客专题页面（内联 docs）构建无 ERROR
- [ ] 无目标卡代号等敏感信息

## 已知坑

- 目标卡代号敏感：一律写「目标卡」。
- 不要编造 URL。
- 被内联的 docs 里**不要写 `{{< >}}` 短代码**；站内链接只在 content/posts 用 relref。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
