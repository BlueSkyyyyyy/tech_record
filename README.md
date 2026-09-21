# tech_record

AI 训练系统工程的学习笔记博客 + 配套代码仓库。

- 在线站点：<https://blueskyyyyyy.github.io/tech_record/>
- GitHub：<https://github.com/BlueSkyyyyyy/tech_record>

## 内容方向

| Series | 主题 |
|---|---|
| Megatron-LM 实战 | 分布式训练框架配置、源码分析、踩坑记录 |
| Kernel / 算子 | Triton、Liger-Kernel 等算子源码阅读与性能分析 |
| 优化器 | Muon / MuonClip 等新兴优化器原理与实现 |
| 量化与推理 | 量化方案细节、推理部署 |

## 仓库结构

```
content/posts/<article-name>/index.md   # 博客文章（Hugo + PaperMod）
code/<article-name>/                    # 文章配套可运行代码，目录名与文章一一对应
.github/workflows/deploy.yml            # GitHub Pages 自动部署
```

## 文章索引

### Flash Attention 精读系列（7 篇）

1. [原理与数学推导：online softmax、分块、IO 复杂度](content/posts/flash-attention-01-theory/index.md)
2. [Triton 教程版前向逐行精读](content/posts/flash-attention-02-triton-fwd/index.md)
3. [FlashAttention-2/3 CUDA 前向实现](content/posts/flash-attention-03-cuda-fwd/index.md)
4. [反向梯度推导与 recompute 实现](content/posts/flash-attention-04-bwd/index.md)
5. [Gluon / TileLang / Liger 多实现对比](content/posts/flash-attention-05-dsl-zoo/index.md)
6. [官方仓库源码全景：FA2 / FA3 / FA4 三套实现怎么组织](content/posts/flash-attention-06-official-code/index.md)
7. [反向传播深潜：从四套实现到 MLA 反向](content/posts/flash-attention-07-bwd-deep/index.md)

配套代码：[code/flash-attention/](code/flash-attention/)（`ref_impl.py` 前向/反向参考实现 + `bwd_variants_ref.py` 反向变体参考，均含自测）

### Megatron-LM 源码精读系列（25 篇）

1. [整体代码结构与启动链路](content/posts/megatron-code-01-structure/index.md)
2. [模型并行的原理（TP/SP/PP/CP/DP/FSDP）](content/posts/megatron-code-02-parallel-principles/index.md)
3. [并行拓扑：parallel_state.py 精读](content/posts/megatron-code-03-parallel-topology/index.md)
4. [重计算原理与代码](content/posts/megatron-code-04-recompute/index.md)
5. [随机种子的设置](content/posts/megatron-code-05-rng-seeds/index.md)
6. [与 Transformer Engine 的关系](content/posts/megatron-code-06-transformer-engine/index.md)
7. [CPU offload 实现](content/posts/megatron-code-07-cpu-offload/index.md)
8. [ZeRO-1 / FSDP 实现](content/posts/megatron-code-08-zero-fsdp/index.md)
9. [数据集处理](content/posts/megatron-code-09-dataset/index.md)
10. [checkpoint 处理](content/posts/megatron-code-10-checkpoint/index.md)
11. [优化器](content/posts/megatron-code-11-optimizer/index.md)
12. [fused 算子](content/posts/megatron-code-12-fused-kernels/index.md)
13. [强化学习](content/posts/megatron-code-13-rl/index.md)
14. [MoE 实现与优化](content/posts/megatron-code-14-moe/index.md)
15. [多模态（LLaVA）实现](content/posts/megatron-code-15-multimodal/index.md)
16. [Context Parallel 细节](content/posts/megatron-code-16-context-parallel/index.md)
17. [MCore 架构与 layer spec 机制](content/posts/megatron-code-17-mcore-arch/index.md)
18. [Pipeline 调度细节](content/posts/megatron-code-18-pipeline-schedule/index.md)
19. [并行组装地图](content/posts/megatron-code-19-parallel-assembly/index.md)
20. [通信与计算 overlap](content/posts/megatron-code-20-overlap/index.md)
21. [TP 通信原语与并行线性层](content/posts/megatron-code-21-tp-communication/index.md)
22. [分布式 checkpoint 底层（dist_checkpointing）](content/posts/megatron-code-22-dist-checkpointing/index.md)
23. [Mamba/SSM 与 Hybrid 模型](content/posts/megatron-code-23-mamba-ssm-hybrid/index.md)
24. [显存估算：精度、并行与激活的完整账本](content/posts/megatron-code-24-memory-estimation/index.md)
25. [一步训练耗时估算：计算、通信与气泡的完整账单](content/posts/megatron-code-25-step-time-estimation/index.md)

配套代码：[code/megatron-code/](code/megatron-code/README.md)（写作素材与源码分析笔记）

### Kernel / 算子性能（1 篇）

- [H100 上 Transformer Engine 算子的性能与 Roofline 对比](content/posts/te-perf-roofline/index.md)

配套代码：[code/te-perf/](code/te-perf/)（TE 原生 kernel 微基准，含 launch overhead 测量）

### CUDA 算子调优系列（连载中）

从零手写并优化 CUDA 算子，每篇含可运行代码、H100 实测数据与 `ncu` 剖析。路线图见 [code/kernel-opt/ROADMAP.md](code/kernel-opt/ROADMAP.md)。

1. [开篇：GPU 怎么跑一个算子，怎么判断它快不快](content/posts/cuda-kernel-opt-01-overview/index.md)
2. [第一个 kernel：从 vector add 看懂线程层次](content/posts/cuda-kernel-opt-02-first-kernel/index.md)
3. [正确测量：计时陷阱、有效带宽与 ncu 入门](content/posts/cuda-kernel-opt-03-measurement/index.md)
4. [访存合并与向量化：同一个 copy，差 7 倍](content/posts/cuda-kernel-opt-04-coalescing/index.md)
5. [共享内存与 bank conflict：矩阵转置](content/posts/cuda-kernel-opt-05-transpose/index.md)
6. [归约与 warp shuffle：求和为什么能慢 1000 倍](content/posts/cuda-kernel-opt-06-reduction/index.md)
7. [Softmax 优化：从 3 个 kernel 到 1 个](content/posts/cuda-kernel-opt-07-softmax/index.md)
8. [GEMM 入门：从朴素三重循环到 shared-memory tiling](content/posts/cuda-kernel-opt-08-gemm/index.md)
9. [GEMM 进阶：寄存器分块、向量化与 double buffering](content/posts/cuda-kernel-opt-09-gemm-advanced/index.md)
10. [ncu 深潜：occupancy、warp stall、roofline 与 source/SASS](content/posts/cuda-kernel-opt-10-ncu-deep/index.md)
11. [launch 配置与 occupancy：ILP、`__launch_bounds__` 与 thread tile](content/posts/cuda-kernel-opt-11-launch-occupancy/index.md)
12. [异步拷贝与流水线：cp.async 多级流水线把访存藏起来](content/posts/cuda-kernel-opt-12-async-pipeline/index.md)
13. [Tensor Core 入门：从 WMMA 到裸 mma.m16n8k16 + ldmatrix](content/posts/cuda-kernel-opt-13-tensor-core/index.md)
14. [融合与 epilogue：给 GEMM 加上 bias 和激活](content/posts/cuda-kernel-opt-14-fusion-epilogue/index.md)
15. [MLA 注意力（一）：从数学到第一个能跑的 kernel](content/posts/cuda-kernel-opt-15-mla-attn/index.md)
16. [MLA 注意力（二）：单 kernel 融合，把 S/P 从显存里干掉](content/posts/cuda-kernel-opt-16-mla-fused/index.md)
17. [MuonClip 的 Newton–Schulz 正交化：15 个 GEMM 的链式算子](content/posts/cuda-kernel-opt-17-muonclip-ns/index.md)
18. [DSA 稀疏注意力（一）：lightning indexer 与 exact top-k](content/posts/cuda-kernel-opt-18-dsa-indexer-topk/index.md)
19. [DSA 稀疏注意力（二）：稀疏 MLA 消费端（gather top-k + cp.async 流水）](content/posts/cuda-kernel-opt-19-dsa-sparse-attn/index.md)
20. [wgmma + SW128 swizzle：把 MLA 的 smem 布局换对](content/posts/cuda-kernel-opt-20-mla-wgmma-sw128/index.md)
21. [MLA 极限冲刺（二）：修好 V 转置访存 + cp.async 预取 K](content/posts/cuda-kernel-opt-21-mla-wgmma-pipe/index.md)
22. [FP8 GEMM（一）：e4m3 的 mma.sync 与 wgmma，per-tensor / per-block 缩放](content/posts/cuda-kernel-opt-22-fp8-gemm/index.md)

配套代码：[code/kernel-opt/](code/kernel-opt/README.md)（公共工具 + 容器/编译/剖析脚本 + 每篇实验 + [优化技巧台账/性能榜](code/kernel-opt/TECHNIQUES.md)）

### 其他

- [开篇：为什么写这个博客](content/posts/hello-world/index.md)

## 本地开发

```bash
# 安装 Hugo extended（>= 0.140）
hugo server -D          # 本地预览 http://localhost:1313/tech_record/
hugo --gc --minify      # 本地构建到 public/
```

写作流程：新建 `content/posts/<slug>/index.md`（frontmatter 含 title/date/tags），如需配套代码则在 `code/<slug>/` 下创建同名目录，并在上方索引表中登记。

AI agent 在本仓库工作：入口 [agent_guide.md](agent_guide.md)，任务技能见 [agent_skills/](agent_skills/)。
