# flash-kda — Kimi Delta Attention（KDA）前向 / 反向算子实现解析

以 **FlashKDA**（MoonshotAI，`~/github/FlashKDA`）为参考，逐层拆解 **Kimi Delta Attention（KDA）**
的前向与反向算子实现：数学形式（gated delta rule 的分块形式）→ CUDA kernel 结构（K1 prepare / K2 recurrence）
→ 数值对拍 → 性能与 `ncu` 剖析 → 反向的推导与实现。

- 目标：把 KDA 前反向讲透，并给出**可运行**的参考实现与实测数字。
- 参考实现：FlashKDA（forward，SM90+，`CHUNK=16`，`D=128`）；反向以 `flash-linear-attention` 的
  `chunk_kda` autograd / 自行推导为参照。
- 环境：复用 `kernel_lab` 容器（`scripts/lab.sh` 指向 kernel-opt 的脚本；含 CUDA/nvcc/ncu/PyTorch）。
- 博客专题：`content/posts/flash-kda-*`，正文由 `code/flash-kda/docs/*.md` 经 `doc_include` **实时内联**。

## 目录

```
flash-kda/
  ROADMAP.md                 # 任务/进度/下一步（活文档，autopilot 每轮先读）
  docs/                      # 分析文档（内联到博客）
  docs/appendix/             # 长日志附录（不内联：逐轮实验原始记录，供对账）
  harness/                   # 探测参考实现、dump/对拍/基准
  src/                       # 我们自己的参考实现（PyTorch 数学 + 可选 CUDA）
  scripts/                   # lab / autopilot
```

## 参考坐标

| 资源 | 位置 | 用途 |
|---|---|---|
| FlashKDA forward（SM90+） | `~/github/FlashKDA/csrc/` | 主要参照：K1 `fwd_kernel1.cuh`、K2 `fwd_kernel2.cuh` |
| FlashKDA 设计笔记 | `~/github/FlashKDA/docs/20260420-flashkda-v1-deep-dive.md` | chunk=16、两段式、精度取舍 |
| FLA KDA（含反向/autograd） | `pip install flash-linear-attention`（容器内） | 反向数学与对拍参照 |
| §KDA 论文 | Kimi Linear / KDA 技术报告 | 数学定义 |

## 快速开始

```bash
cd ~/proj/tech_record/code/flash-kda
scripts/lab.sh exec "python -c 'import torch; print(torch.cuda.get_device_name(0))'"
```
