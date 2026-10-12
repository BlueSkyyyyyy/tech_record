# flash-kda — KDA 前向 / 反向算子实现解析的自驱流程

## 触发场景

用户说「继续 KDA」「flash-kda」「Kimi Delta Attention」。本仓库长期自驱任务之一。

## 前置条件

- 读 `agent_guide.md`（目录约定/坑，尤其：**不要写目标卡代号**、`doc_include` 内联渲染机制）。
- 读控制面板 `code/flash-kda/ROADMAP.md`（每次先读，从「下一步」做起，做完更新）。
- 环境：复用 `kernel_lab` 容器（`code/flash-kda/scripts/lab.sh` 指向 kernel-opt 的脚本）。

## 关键事实

- 参考实现 **FlashKDA**：`~/github/FlashKDA`（forward，SM90+，`CHUNK=16`，$D=128$）。
  - API：`csrc/flash_kda.cpp`；launch：`csrc/fwd.h`；K1 `csrc/smxx/fwd_kernel1.cuh`；K2 `fwd_kernel2.cuh`。
  - **现成 PyTorch 参照**：`tests/torch_ref.py`（`torch_ref()` 逐式对应分块公式，含 `inv_fwd_subst_16`）。
  - 设计笔记：`docs/20260420-flashkda-v1-deep-dive.md`。
- 反向参照：容器内 `pip install flash-linear-attention` 后 `from fla.ops.kda import chunk_kda`，用
  `torch.autograd` 拿反向；或基于 `src/kda_bwd_ref.py` 自行推导。
- ours 实现：`src/`（纯 PyTorch 参考，自带自测）；`harness/`（探测/dump/对拍/基准）。
- 博客专题由 `code/flash-kda/docs/*.md` **实时内联**渲染（`{{< doc_include "..." >}}`），改 docs 即等于改博客。

## 步骤（每轮一个增量）

1. 读 ROADMAP「下一步」，选一个未完成子项（P0→P1→…）。
2. 写/改 `docs/<NN>-*.md` 与必要的 `src/`、`harness/`。
3. 用 `scripts/lab.sh exec "..."` 在容器里跑，留 `*.out.txt`；数值对拍给真实 max abs/rel。
4. 更新 `ROADMAP.md`；commit（`kda:` 前缀）→ push。
5. 不要创建 `AUTOPILOT_STOP`（只有用户 `scripts/autopilot.sh stop` 才停）。

## 验收标准

- [ ] 文档公式与 FlashKDA 源码/`torch_ref.py` 逐项对得上，行号真实核对
- [ ] 有实测数字（对拍误差 / 性能 / ncu）；原始输出留档
- [ ] 博客专题页面（内联 docs）构建无 ERROR
- [ ] 无目标卡代号等敏感信息

## 已知坑

- 目标卡代号敏感：一律写「目标卡」。
- 被内联的 docs 里**不要写 `{{< >}}` 短代码**；站内链接只在 content/posts 用 relref。
- 容器 workdir 是 `code/kernel-opt`，`docker exec` 跑脚本要用**绝对路径**。
- FlashKDA 当前只支持 $K=V=128$、SM90+；bf16 输入。
