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

### Flash Attention 精读系列（5 篇）

1. [原理与数学推导：online softmax、分块、IO 复杂度](content/posts/flash-attention-01-theory/index.md)
2. [Triton 教程版前向逐行精读](content/posts/flash-attention-02-triton-fwd/index.md)
3. [FlashAttention-2/3 CUDA 前向实现](content/posts/flash-attention-03-cuda-fwd/index.md)
4. [反向梯度推导与 recompute 实现](content/posts/flash-attention-04-bwd/index.md)
5. [Gluon / TileLang / Liger 多实现对比](content/posts/flash-attention-05-dsl-zoo/index.md)

配套代码：[code/flash-attention/](code/flash-attention/ref_impl.py)（纯 PyTorch 参考实现，含自测）

### Megatron-LM 源码精读系列（20 篇）

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

配套代码：[code/megatron-code/](code/megatron-code/README.md)（写作素材与源码分析笔记）

### Kernel / 算子性能（1 篇）

- [H100 上 Transformer Engine 算子的性能与 Roofline 对比](content/posts/te-perf-roofline/index.md)

配套代码：[code/te-perf/](code/te-perf/)（TE 原生 kernel 微基准，含 launch overhead 测量）

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
