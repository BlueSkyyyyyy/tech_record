# mega-moe — DeepSeek Mega MoE 前向/反向算子解析的自驱流程

## 触发场景

用户说「继续 Mega MoE」「mega-moe」「DeepGEMM MoE」。本仓库长期自驱任务之一。

## 前置条件

- 读 `agent_guide.md`（目录约定/坑，尤其：**不要写目标卡代号**、`doc_include` 内联渲染机制）。
- 读控制面板 `code/mega-moe/ROADMAP.md`（每次先读，从「下一步」做起，做完更新）。
- 环境：复用 `kernel_lab` 容器（`code/mega-moe/scripts/lab.sh` 指向 kernel-opt 的脚本）。

## 关键事实

- 参考：`~/github/DeepGEMM`（Mega MoE、grouped GEMM、反向 wgrad）、`~/github/DeepEP`（dispatch/combine、对称显存）。
- **Mega MoE 是 SM100（Blackwell）专属**，本机 H100（sm90）**跑不了端到端**；这部分只做源码精读 + 线上资料，
  明确标注「仅分析」。H100 可行实验：grouped GEMM / backward、DeepEP 通信原语。
- 关键文件：`csrc/apis/mega_moe.hpp`、`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`、
  `scheduler/mega_moe.cuh`、`layout/mega_moe.cuh`、`tests/test_mega_moe.py`；反向 `k_grouped_fp8_gemm_tn_contiguous`。
- 博客专题由 `code/mega-moe/docs/*.md` **实时内联**渲染（`{{< doc_include "..." >}}`）。

## 步骤（每轮一个增量）

1. 读 ROADMAP「下一步」，选一个未完成子项。
2. 写/改 `docs/<NN>-*.md` 与必要的 `src/`、`harness/`。
3. 能跑的用 `scripts/lab.sh exec "..."` 跑真实数字；跑不了的标明「仅分析」。
4. 更新 `ROADMAP.md`；commit（`megamoe:` 前缀）→ push。
5. 不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。

## 验收标准

- [ ] 数据流/机制讲清楚，源码行号真实核对，线上资料给真实链接
- [ ] 能跑的部分有实测数字与原始输出；跑不了的部分明确标注
- [ ] 博客专题页面（内联 docs）构建无 ERROR
- [ ] 无目标卡代号等敏感信息

## 已知坑

- 目标卡代号敏感：一律写「目标卡」。
- H100 上**不要**跑 `fp8_fp4_mega_moe`（SM100 专属）。
- 被内联的 docs 里**不要写 `{{< >}}` 短代码**；站内链接只在 content/posts 用 relref。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
