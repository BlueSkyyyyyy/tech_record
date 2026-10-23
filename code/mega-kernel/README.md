# mega-kernel — MegaKernel（巨型/持久化融合内核）资料与实现分析

系统梳理 **MegaKernel**（也叫 persistent / mega-kernel）：把整张算子图**融合进单个持久化 kernel**、
在 SM 级别做跨算子软件流水与通信-计算重叠的设计与研究。

材料来源（本机已有）：
- **MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs**（Mirage Persistent Kernel，arXiv 2512.22219）——
  论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`。
- **DeepGEMM** 的 mega 家族：Mega MoE / Mega Gate / Mega mHC（生产级案例）。
- **Kimi K3** 技术报告（fused mega-kernels 任务套件）、`~/github/hpc-ops`（roadmap 中的 Megakernel）。
- 通用背景：kernel-per-operator 执行模型、CUDA Graphs、PDL、persistent kernel。

- 博客专题：`content/posts/mega-kernel-*`，正文由 `code/mega-kernel/docs/*.md` 经 `doc_include` **实时内联**。
- 说明：本专题以**资料分析**为主（论文/源码精读），H100 上能做的小实验尽量做并标注。

## 目录

```
mega-kernel/
  ROADMAP.md                 # 任务/进度/下一步（autopilot 每轮先读）
  docs/                      # 分析文档（内联到博客）
  docs/appendix/             # 长日志附录（不内联：实验原始记录，供对账）
  harness/                   # 可用的小实验 / 复现
  src/                       # 参考实现（如有）
  scripts/                   # lab / autopilot
```

## 参考坐标

| 资源 | 位置 | 用途 |
|---|---|---|
| MPK 论文快照 | `~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt` | 主参考：SM 级图表示、in-kernel runtime |
| MPK 项目 | `https://github.com/mirage-project/mirage`（线上） | 实现代码 |
| DeepGEMM mega 家族 | `~/github/DeepGEMM`（Mega MoE / Gate / mHC） | 生产级案例 |
| hpc-ops | `~/github/hpc-ops/README.md`（Megakernel roadmap） | 工程视角 |
| Kimi K3 报告 | `~/github/ai-infra-book/references/text/kimi-k3.txt` | fused mega-kernels 任务套件 |
