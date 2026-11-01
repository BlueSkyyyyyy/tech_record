---
title: "Megatron 源码精读（二十六）：2026-09 → 2026-10 上游更新速览"
date: 2026-11-01
draft: false
tags: ["megatron-lm", "系列", "训练框架", "版本更新"]
categories: ["训练框架"]
weight: 26
series: ["megatron-code"]
---

前 25 篇的基准是 commit `f713506ce`（2026-08-31，`26.04-alpha.rc1-814-gf713506ce`）。到 2026-10-29 的 `c4220af3a`，上游 `dev` 又走了 **1208 个 commit**，`megatron/core/` 下 **509 个文件**变更。

**一句话结论**：主干（并行拓扑、1F1B、DistOpt、DCP）没变，但出现了三条新主线——**GTP**（在 TP 之上再切一刀权重）、**MFSDP v2**（重写 FSDP 的 shard 布局与优化器）、**DSA 家族 + mHC**（DeepSeek-V4 系结构）；MoE 的工程化（paged stash、megakernel、MoonEP）也有实质推进。

本篇只讲「会影响前 25 篇结论」的变化，每条给出文件与 PR 号，方便回查。

---

## 1. 一张表：变了什么、影响哪篇

| 主题 | 变化 | 影响篇目 |
|---|---|---|
| GTP | 新增「TP 之上再切权重」的并行轴，按权重 all-gather | 02、21、24、25 |
| MFSDP v2 | 重写 FSDP：`TensorAtomic`/`BlockAtomic`/`RowAtomic` placement | 08、11 |
| Muon | sharded momentum、LayerShardedMuon、per-head QKV | 11 |
| CP | 新增 `context_parallel/` 包；dynamic CP 接入标准 schedule | 03、16、18 |
| DSA 家族 | min-memory kernel、KPool indexer、NoPE MLA、DSA over GQA | 06、14、16 |
| mHC | Manifold-Constrained Hyper-Connections + 融合 kernel | 12、17 |
| MoE 工程 | paged stash / sync-free、megakernel(MOK)、MoonEP 复制 | 14、20 |
| 重计算 | `CheckpointWithoutOutput` 重构、FP8 状态修复 | 04、06 |
| 确定性 | Triton autotune 固定表、batch-invariant | 05、25 |
| 数据/存档/推理 | varlen 新数据集、ckpt merge、分离式推理 | 09、10、22 |

---

## 2. GTP：把 ZeRO-3 搬到模型并行轴上

`megatron/core/tensor_parallel/generalized_tensor_parallelism.py`（#4967）引入了 **Generalized Tensor Parallelism**：

$$
\text{GTP} = \text{TP} \times \text{GTP\_remat}
$$

- `TP` 这一刀照旧：每个 rank 永久持有 $1/\mathrm{TP}$ 的权重，GEMM 的输出仍是 TP 分片；
- `GTP_remat` 这一刀是 ZeRO-3 式的：每个 rank 只常驻 $1/\mathrm{GTP\_remat}$ 的**那一份 TP 分片**，GEMM 前 all-gather 出完整 TP 分片，用完就释放，反向对 wgrad 做 reduce-scatter。

和 FSDP 的区别有两点，值得记：

1. **切在模型并行轴上，不是数据并行轴**。所以它和 TP 组合，而不是抢同一维；
2. **一次只物化一个权重**，不是一整个 bucket。这样才能做到「按权重预取」和低精度 gather（FP8/MXFP8/NVFP4）。

开关是 `--tensor-parallel-num-weight-shards`（= TP × `gtp_weight_remat_size`），专家侧另有 `--expert-tensor-parallel-num-weight-shards`（EGTP），需要 TE ≥ 2.19；`gtp_weight_remat_size=1` 时路径与普通 TP 完全一致。

**对第 24 篇账本的修正**：每 rank 常驻权重从 $2P/(\mathrm{TP}\cdot\mathrm{PP})$ 变成

$$
\frac{2P}{\mathrm{TP}\cdot \mathrm{GTP\_remat}\cdot \mathrm{PP}}
$$

梯度、master、优化器 state 同步缩小。代价是前向/反向各多一次 all-gather + reduce-scatter，靠预取藏进计算（第 25 篇的通信账要加一项，但基本落在「可隐藏」里）。

## 3. MFSDP v2 与 Muon

`megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/` 是一套**并行开发**的新 FSDP（设计文档 `mfsdp_design.md`），核心是把「怎么切」变成三种 placement：

| placement | 切法 | 用途 |
|---|---|---|
| `RowAtomic` | 按整行切，各 rank 等长 | 普通浮点参数的默认 |
| `BlockAtomic(k)` | 按 k 行一块切，块不跨 rank | 块量化（MXFP8 用 k=32） |
| `TensorAtomic` | 整个参数归一个 rank，rank 间可不等长 | **Muon**：权重/梯度/动量同 owner，step 无通信 |

这解释了为什么 Muon 需要单独的 FSDP：Muon 的 Newton–Schulz 正交化要在「完整矩阵」上做，如果按行切碎就没法算。

Muon 侧同期更新：MFSDP v2 的 sharded momentum（#8313）、`LayerShardedMuon` 的流式 Newton–Schulz（#8379）、per-head QKV 正交化（#7085）。

> 注意：MFSDP v2 是**独立代码路径**，`use_megatron_fsdp` 的旧实现仍在；两者不要混着配。

## 4. Context Parallel：从「布局」升级为「子系统」

- 新增 `megatron/core/context_parallel/`：`groups.py`（CP 组构造）、`layout.py`（zigzag/contiguous）、`sequence_roll.py`（序列滚动）。旧的 `context_parallel_layout/` 还在，但已不是唯一入口（#6776/#6764）。
- **dynamic CP** 接入标准 train/eval schedule（#7047，"DCP V2"）：由 sequence packing scheduler **按 microbatch 选择 CP 组**，所以 `context_parallel_size=1` 时也可能有 CP group，配置校验要按「静态 CP>1 或 dynamic CP 开启」判断。
- 结构相关的硬约束：GDN2 只支持 headwise CP 且拒绝 contiguous 与 dynamic CP；DSA 只支持 `cp_comm_type="all_gather"`。

第 16 篇讲的 zigzag 负载均衡与 all-to-all 布局转换仍然成立，只是入口和调度层变了。

## 5. Attention：DSA 家族与 mHC

**DSA（DeepSeek Sparse Attention）**在这一季从「一个 kernel」长成了一个家族：

- `dsa_min_memory.py` / `dsa_min_memory_triton.py`：省显存的稀疏 attention 实现（#7152）；
- `kpool_indexer.py`：KPool 版 indexer；同时支持 NoPE MLA 与 indexer 的 FP8 精度控制（#7522）；
- `dsa_gqa.py`：DSA over GQA 进入 hybrid 模型（#7156）；
- `cp_balanced_indexer.py`：CP 下的 indexer 负载均衡。

**mHC（Manifold-Constrained Hyper-Connections）**是另一条线：用 Sinkhorn 投影把多流残差约束成双随机矩阵，`fused_mhc_kernels.py` 提供 `sinkhorn` / `h_aggregate` / `h_post_bda` / `proj_rms_compute_h` 四个融合算子，并支持 VPP（#7336）。它改变了 Transformer 层的连接方式，第 17 篇讲的 spec 插槽正是为这种结构准备的。

## 6. MoE：从「能跑」到「跑得稳、跑得省」

三件事：

1. **paged stash 打通 + sync-free dispatch**（#6828/#8507）：`paged_stash.py` 基线就有，这一季把它接到 device-initiated GroupedLinear 与 HybridEP 的无同步路径上。`--moe-paged-stash` 把反向要用的专家激活放进分页 buffer（可 spill 到 pinned host）省显存，`--moe-expert-rank-capacity-factor` 预分配 dispatch/expert buffer，消掉每步的动态查询与重分配。
2. **megakernel（MOK）**（#6572）：`moe/megakernel/` 提供 backend-neutral 的 MoE megakernel 接入层，把路由、dispatch、专家计算、combine 尽量融合，减少 kernel 启动与中间张量。
3. **MoonEP 式专家复制**（#7450）：`virtual_expert_load_balancer.py` 按 router 输出在线规划冗余专家，把热点专家复制到多个 rank，缓解负载倾斜。

第 14 篇的四段式（route → dispatch → expert → combine）没变，变的是每一段内部的实现与调度。

## 7. 训练工程：重计算、确定性、存档

- **重计算**：`CheckpointWithoutOutput` 内部重构（#8334/#8390），把「存/取/释放」的生命周期理顺；修复 `tensor_parallel.checkpoint` 重算时 FP8 autocast 状态丢失（#7538）、GDN/KDA 重算丢 FP8/FP4 context（#8203）。第 4 篇的配置语义不变。
- **确定性**：Triton autotune 的配置选择可以固定（#6076），新增 `megatron/core/tuning/` 与 `--triton-autotune-*`，内置 sm100/sm103 的表；配合 `--deterministic-mode` 可避免「不同 rank 选了不同 tile 导致归约顺序不同」。
- **存档**：新增 checkpoint merge 工具文档；`dist_ckpt` 支持 fp8/fp4 param-gather 的仅推理流式反量化（#7405）。
- **推理**：新增 `megatron/core/inference/disaggregation/`，支持 KV handoff 的分离式推理（prefill/decode 分池）。
- **MTP**：`multi_token_prediction.md` 补齐了 MTP 结构与 `e2e_tv` 损失，配套 `fused_mtp_prefix` / `fused_mtp_tv` 融合算子。

## 8. 怎么跟进上游

```bash
# 1. 看这段时间改了什么
git log --oneline <old>..origin/dev -- megatron/core | head -50

# 2. 只看某个子系统
git diff --stat <old> origin/dev -- megatron/core/transformer/moe

# 3. 读官方文档（新增的都在 docs/user-guide/features/）
ls docs/user-guide/features
```

三条经验：

- **先看 docs 再看代码**。这一季新增的 `context_parallel.md`、`megatron_fsdp.md`、`paged_stash.md`、`generalized_tensor_parallel.md` 质量都不错，能省很多时间。
- **行号一定会漂**。本系列统一用「文件 + 符号名」定位，行号只作辅助；跨版本对照时以符号名为准。
- **实验特性要看清 gate**。GTP 需要 TE ≥ 2.19，MFSDP v2 是独立路径，paged stash 与 CUDA graph 有额外约束——先读 doc 的 Prerequisites。

## 9. 小结

- **主干稳定**：并行拓扑、1F1B、DistOpt、DCP 的骨架没变，前 25 篇的结论仍然成立。
- **新增一条并行轴**：GTP 在 TP 之上再做 ZeRO-3 式权重重物化，改的是显存账本，不是并行语义。
- **FSDP 被重写**：MFSDP v2 用 placement 描述切法，为 Muon 这种「要完整矩阵」的优化器铺路。
- **结构在演进**：DSA 家族与 mHC 说明 Megatron 正在把 DeepSeek-V4 系的结构纳入主干。
- **MoE 的重点转向工程**：paged stash、megakernel、专家复制，目标都是「少同步、省显存、抗倾斜」。

---

**行号基线**：本篇基于 `f713506ce` → `c4220af3a` 的 diff 与上游文档；具体行号请以 `c4220af3a` 的源码为准。
