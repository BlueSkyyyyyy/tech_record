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
23. [FP8 GEMM（二）：TMA + mbarrier + warp specialization（768 → 1217 TFLOPS）](content/posts/cuda-kernel-opt-23-fp8-gemm-tma/index.md)
24. [FP8 GEMM（三）：per-block 缩放的两堵墙——寄存器与 ptxas 序列化](content/posts/cuda-kernel-opt-24-fp8-gemm-pb/index.md)
25. [MoE grouped GEMM：一个 kernel 吃掉 384 个 expert](content/posts/cuda-kernel-opt-25-moe-grouped-gemm/index.md)
26. [MoE grouped GEMM 的 L2 墙：TMA cluster multicast 实测只赚 4%](content/posts/cuda-kernel-opt-26-moe-cluster-multicast/index.md)
27. [MoE 的 router、top-k 与 token 置换：搬运才是大头，置换已贴 HBM 天花板](content/posts/cuda-kernel-opt-27-moe-router/index.md)
28. [MoE 门控 GEMM 的 wgmma + TMA 冲刺：从 cuBLAS 的 28% 到 80%](content/posts/cuda-kernel-opt-28-moe-gate-wgmma/index.md)
29. [MoE 门控 GEMM 的 TMA cluster multicast：广播 A 还是广播 B？](content/posts/cuda-kernel-opt-29-moe-gate-multicast/index.md)
30. [MoE 专家 FFN 的融合：gather 输入 + SwiGLU epilogue + unpermute 融合](content/posts/cuda-kernel-opt-30-fused-moe/index.md)
31. [MoE grouped GEMM 叠上 DeepSeek 的 per-block FP8 缩放：寄存器墙在分组场景下更便宜](content/posts/cuda-kernel-opt-31-moe-grouped-fp8pb/index.md)
32. [RMSNorm / QK-Norm 与残差、FP8 量化的融合：一个纯访存算子怎么贴到 87% HBM](content/posts/cuda-kernel-opt-32-fused-norm/index.md)
34. [MoE 专家 FFN 的 FP8 化：权重流量减半换来 1.9~2.2×，以及 unpermute 融合在 FP8 下为什么失效](content/posts/cuda-kernel-opt-34-fused-moe-fp8/index.md)
35. [DSA Compressor：把 KV 用「门控池化」压成更少的 key，以及一个被 L2 卡住的合并投影](content/posts/cuda-kernel-opt-35-dsa-compressor/index.md)
36. [DSA Compressor 投影的「L2 墙」拆解：三种压 L2 的手段全负，以及用 FP8 把投影推到 1.4×](content/posts/cuda-kernel-opt-36-dsa-compressor-fp8/index.md)
37. [FP8 per-block GEMM 的第三条路：把 ue8m0 的 2 的幂 scale 折进操作数（940→1208 TFLOPS）](content/posts/cuda-kernel-opt-37-fp8-pb-prescale/index.md)
38. [Paged KV-cache / flash-decoding 推理注意力：从 46% 打到 93% HBM](content/posts/cuda-kernel-opt-38-paged-kv/index.md)
39. [MLA 吸收式 decode + FP8 KV-cache：一个张量核也吃不饱的「平衡」算子](content/posts/cuda-kernel-opt-39-mla-decode/index.md)
40. [W4A16 dequant-GEMM：权重 4-bit 化在 decode 赚的那 1.4×，以及反量化税](content/posts/cuda-kernel-opt-40-w4a16-gemm/index.md)
41. [W4A16 的 warp specialization：把 occupancy 从 6.5% 提到 22%，为什么只快 7%？](content/posts/cuda-kernel-opt-41-w4a16-warp-specialization/index.md)
42. [MLA decode 的 wgmma 改造：消掉 QK dup，却撞上 V 转置税，再用混合 mma 绕开（1.53×）](content/posts/cuda-kernel-opt-42-mla-decode-wgmma/index.md)
43. [W4A16 decode 的跨 item 持久化流水：把 wave 2.58 压到 1，再跟每 stage 的除法/原子税算账](content/posts/cuda-kernel-opt-43-w4a16-persist/index.md)
44. [W4A16 的 M=1 GEMV：M=1 时张量核有 63/64 的行在空转，不如回去做带宽最优的 GEMV](content/posts/cuda-kernel-opt-44-w4a16-decode-gemv/index.md)
45. [W4A8 + dp4a：把 M=1 decode 的反量化 ALU 税从 60% 打到 30%，贴上 HBM 带宽（1.53×）](content/posts/cuda-kernel-opt-45-w4a8-dp4a-gemv/index.md)
46. [小 M decode 的 GEMV ↔ 张量核分派：交叉点不是 M≈78，而是 M≈2.5 / M≈17](content/posts/cuda-kernel-opt-46-w4a16-smallm-dispatch/index.md)
47. [给 W4A16 小 M 的 mma_b16 叠 warp specialization：一个负结果，和真正的墙（反量化指令 + KSPLIT 并行度）](content/posts/cuda-kernel-opt-47-w4a16-mma-ws/index.md)
48. [W4A8 的整数张量核（IMMA）：把反量化的 ALU 砍掉一半，小 M 权重带宽翻倍（1.92×）](content/posts/cuda-kernel-opt-48-w4a8-imma/index.md)
50. [W4A8 IMMA 的收尾：K-blocked 权重布局把 `long_scoreboard` 打进 5.7×，以及持久化在这里为什么无效](content/posts/cuda-kernel-opt-50-w4a8-imma-pipe/index.md)
51. [W4A8 IMMA 的 warp-private pipeline：把 barrier 从 1.61 清零之后，墙去哪了](content/posts/cuda-kernel-opt-51-w4a8-imma-warp/index.md)
52. [W4A8 IMMA 的 TMA warp-private：把 2 KB 权重交给 TMA 引擎（和一个必须自己造的无冲突布局）](content/posts/cuda-kernel-opt-52-w4a8-imma-tma/index.md)
53. [融合 RoPE：交错配对、对半配对，以及一个被访问模式封顶的纯访存算子](content/posts/cuda-kernel-opt-53-fused-rope/index.md)
54. [融合 cross-entropy：把 2.12 GB 的 logits 从显存里省掉，和那个卡在 L2 上的 LM head](content/posts/cuda-kernel-opt-54-fused-cross-entropy/index.md)
55. [FP8 融合 cross-entropy：当 LM head 权重减半，那个卡在 L2 上的 GEMM 会变快吗（1.95×）](content/posts/cuda-kernel-opt-55-fp8-fused-ce/index.md)
56. [DeepSeek-V4 的路由专家其实是 FP4：真实 checkpoint 的格式、无损折进 FP8、与 decode GEMV 的实测（2.16×）](content/posts/cuda-kernel-opt-56-v4-fp4-moe/index.md)
57. [FP4 专家权重接回 grouped MoE：现场折 vs 预折叠，与一份被「变换税」吃掉的字节红利（负结果）](content/posts/cuda-kernel-opt-57-v4-fp4-moe-fold/index.md)
58. [FP4 decode GEMV 的第二轮：从 34.8% 到 38.7%，以及为什么 split-K / 无 smem / 位运算解码都救不了它](content/posts/cuda-kernel-opt-58-v4-fp4-decode/index.md)
59. [FP4 decode 的 grouped GEMV：把 top-6 的 6B 个 (token, expert) 对按专家折叠，权重只读一遍](content/posts/cuda-kernel-opt-59-v4-fp4-grouped/index.md)
60. [FP4 grouped decode 的 LUT 真的是墙吗：244.6M 次 bank conflict 被消掉，只快了 2.9%](content/posts/cuda-kernel-opt-60-v4-fp4-lut/index.md)
61. [FP4 decode 的 MoE FFN 端到端：w1/w3→SwiGLU→w2，以及 cp.async 到底救不救得了延迟墙（67.6% 权重 roofline）](content/posts/cuda-kernel-opt-61-v4-fp4-ffn/index.md)
62. [FP4 解码的整 chunk 位拼装：把每 nibble 一次 LDS 换成两条 prmt，K1 从 73.6% 到 83.4% HBM（1.13×）](content/posts/cuda-kernel-opt-62-v4-fp4-ffn2/index.md)
63. [K2 的权重在显存里其实是连续的：用 TMA 1D bulk 搬整 warp row-tile，把 FP4 decode FFN 推到 90% 权重 roofline（1.10×）](content/posts/cuda-kernel-opt-63-v4-fp4-ffn3/index.md)
64. [大 batch 的 FP4 decode FFN：B=256 掉到 63% 不是 occupancy，是一份指令预算（解码占 43%）](content/posts/cuda-kernel-opt-64-v4-fp4-ffn4/index.md)

配套代码：[code/kernel-opt/](code/kernel-opt/README.md)（公共工具 + 容器/编译/剖析脚本 + 每篇实验 + [优化技巧台账/性能榜](code/kernel-opt/TECHNIQUES.md)）

### FlashAttention 反向工程（专题）

借鉴 flash-attention / TransformerEngine，把 **FlashAttention 反向**整理成**单文件 / 两文件**形式
（fp16 / bf16 / fp8），编译、ncu 剖析、与 TE 数值和性能对标，并面向目标 AI 卡做移植。

1. [实现结构与优化手段梳理（FA/TE 对照）](content/posts/fa-bwd-01-catalog/index.md)
2. [fp16 反向实现与优化](content/posts/fa-bwd-02-fp16/index.md)
3. [bf16 反向实现与优化](content/posts/fa-bwd-03-bf16/index.md)
4. [FP8 反向设计](content/posts/fa-bwd-04-fp8-design/index.md)
5. [FP8 反向实现与优化](content/posts/fa-bwd-05-fp8-impl/index.md)
6. [数值与性能汇总（GQA/MQA/MLA）](content/posts/fa-bwd-06-summary/index.md)
7. [目标卡移植注意事项](content/posts/fa-bwd-07-porting/index.md)
8. [目标形状重点分析 & FA 为何比 TE 慢（ncu 归因）](content/posts/fa-bwd-08-fa-vs-te/index.md)
9. [用 SM90 的 FA3 重测：更正「FA 比 TE 慢」](content/posts/fa-bwd-09-fa3-sm90/index.md)

配套代码：[code/flash-attention/fa-bwd/](code/flash-attention/fa-bwd/README.md)
（各篇正文由 `docs/` 下的 Markdown 通过 `fa_include` shortcode 实时内联渲染）。

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
