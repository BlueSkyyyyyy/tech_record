# 配套代码

本目录存放博客文章的可运行代码。目录名与 `content/posts/` 下的文章目录名一一对应。

| 文章 | 代码 | 说明 |
|---|---|---|
| [开篇](../content/posts/hello-world/) | — | 无代码 |
| [Flash Attention 精读系列](../content/posts/flash-attention-01-theory/) | [flash-attention/ref_impl.py](flash-attention/ref_impl.py) | 纯 PyTorch 参考实现：online softmax 递推、分块前向（含 LSE）、recompute 反向，含自测（`python ref_impl.py`） |
| [Flash Attention（七）反向深潜](../content/posts/flash-attention-07-bwd-deep/) | [flash-attention/bwd_variants_ref.py](flash-attention/bwd_variants_ref.py) | 反向变体参考：Δ 恒等式、dropout、softcap、GQA 跨 head 归约，全部与 autograd 对拍（`python bwd_variants_ref.py`） |
| [通算并行研读（专题）](../content/posts/comm-overlap-00-index/) | [comm-overlap/](comm-overlap/README.md) | 7 个纯 CPU 模型：性能账本、流水线调度、DP bucket、MoE 三段式、SM 分区、decode 阈值、trace 审计，每个都带 `--test` |
| [Megatron-LM 源码精读系列](../content/posts/megatron-code-01-structure/) | [megatron-code/](megatron-code/README.md) | 源码分析笔记（analysis-notes/），基于 commit f713506ce |
| [H100 上 TE 算子的性能与 Roofline 对比](../content/posts/te-perf-roofline/) | [te-perf/](te-perf/) | `bench_te.py`：TE 原生 kernel（rmsnorm / bwd / bwd_add / fused_attn）微基准 + launch overhead |
| [CUDA 算子调优系列](../content/posts/cuda-kernel-opt-01-overview/) | [kernel-opt/](kernel-opt/README.md) | CUDA 手写算子系列：公共头 `cuda_utils.cuh` + 容器/编译/`ncu` 脚本 + 每篇实验（vector add、计时陷阱…） |
| [LLM 模型结构变迁（专题）](../content/posts/llm-arch-00-index/) | [llm-arch-evolution/](llm-arch-evolution/README.md) | `budget.py`：参数量 / FLOPs / KV 三维预算计算器，纯标准库，`--test` 对拍 9 个已发布模型（误差 ≤1.3%） |
| [VLA 精读（专题）](../content/posts/vla-00-index/) | [vla/](vla/README.md) | 7 个纯 CPU 演示：256 bin 量化与 DCT、block-causal mask、flow vs L2、q01/q99 归一化、chunk 延迟、RTC 权重、Wilson 区间，全部带 `--test` |
| [扩散模型精读（专题）](../content/posts/diffusion-00-index/) | [diffusion/](diffusion/README.md) | 8 个纯 CPU 演示：噪声日程 SNR、最小 DDPM、DDIM/Euler/Heun、CFG 锐化、latent/DiT/视频预算、FM vs 扩散、蒸馏、FID 偏差，全部带 `--test` |
| [LeetCode 题解精讲](../content/posts/leetcode-00-index/) | [leetcode/](leetcode/README.md) | 分类刷题题解 + Python/C++ 自测代码，统一入口 `scripts/run_all.py`，长期自驱 |
| [KDA 前反向算子实现解析](../content/posts/flash-kda-00-index/) | [flash-kda/](flash-kda/README.md) | 以 FlashKDA 为参考拆解 Kimi Delta Attention 前向/反向算子，长期自驱 |
| [DeepSeek Mega MoE 前反向算子实现解析](../content/posts/mega-moe-00-index/) | [mega-moe/](mega-moe/README.md) | 以 DeepGEMM/DeepEP 为参考分析 Mega MoE 前向与 MoE 反向，长期自驱 |
| [MegaKernel 资料与实现分析](../content/posts/mega-kernel-00-index/) | [mega-kernel/](mega-kernel/README.md) | MPK 论文 + DeepGEMM mega 家族等 MegaKernel 范式分析，长期自驱 |
| [Mirage / MPK 项目分析](../content/posts/mirage-00-index/) | [mirage/](mirage/README.md) | 上手路径：两条产品线、compile() 端到端、运行时、H100 构建与实跑记录 |
| [vLLM 研读（专题）](../content/posts/vllm-00-index/) | [vllm/](vllm/README.md) | 4 个纯 CPU 脚本：KV 预算、块分配、前缀缓存、调度装箱，每个都能 `--test` |
| [SGLang 研读（专题）](../content/posts/sglang-00-index/) | [sglang/](sglang/README.md) | 纯 CPU 脚本：树 vs 块对拍、显存预算、调度装箱、overlap 时间线、跳词收益，每个都能 `--test` |
