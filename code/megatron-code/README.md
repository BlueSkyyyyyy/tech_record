# Megatron-LM 源码精读专题

对应博客系列「Megatron-LM 源码精读」（`content/posts/megatron-code-01` ~ `megatron-code-16`）。

- 参考仓库：`/home/xieminglin/proj/Megatron-LM`
- 分析基准 commit：`f713506cea2e7705dd2ebb00c5c58a046ff974fe`（2026-08-31，`26.04-alpha.rc1-814-gf713506ce`）
- 文章中源码引用格式：`文件路径:行号`，必要时附 GitHub 永久链接：
  `https://github.com/NVIDIA/Megatron-LM/blob/f713506cea2e7705dd2ebb00c5c58a046ff974fe/<path>#L<n>`

## 目录

- `analysis-notes/` —— 写作素材：对源码仓库并行深读产出的原始分析报告（按文件组织，未经重写，行号引用以这些笔记为准在成文前抽查）：
  - `01-02-03-structure-parallel-state.md` —— 整体结构 + parallel_state.py 拓扑（对应第 1/2/3 篇）
  - `02-parallelism-principles.md` —— TP/SP/PP/CP/DP/FSDP 原理与代码映射（第 2 篇）
  - `04-05-recompute-rng.md` —— 重计算 + 随机种子（第 4/5 篇）
  - `06-12-te-fused.md` —— Transformer Engine 集成 + fused 算子（第 6/12 篇）
  - `07-08-11-offload-zero-optimizer.md` —— CPU offload + ZeRO/FSDP + 优化器（第 7/8/11 篇）
  - `09-10-dataset-checkpoint.md` —— 数据集 + checkpoint（第 9/10 篇）
  - `13-rl.md` —— 强化学习（第 13 篇）
  - `14-moe.md` —— MoE（第 14 篇）
  - `15-multimodal.md` —— 多模态（第 15 篇）
  - `16-cp-details.md` —— Context Parallel 细节（第 16 篇）

## 注意

- analysis-notes 是中间素材，可能包含「待核实」标记；成文时必须按 `agent_skills/code-dive.md` 的验收标准抽查行号。
- 若 Megatron-LM 本地仓库更新，以上行号会漂移，需重新核对。
