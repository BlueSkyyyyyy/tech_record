# tech_record

AI 训练系统工程的学习笔记博客 + 配套代码仓库。

- 在线站点：<https://blueskyyyyyy.github.io/tech_record/>
- GitHub：<https://github.com/BlueSkyyyyyy/tech_record>

## 内容方向

| Series | 主题 |
|---|---|
| AI 日报 | 每天 21:00 抓 Hugging Face Daily Papers，按主题分组的中文速览（保留数字、点明取舍） |
| Megatron-LM 实战 | 分布式训练框架配置、源码分析、踩坑记录 |
| 通算并行 | 通信与计算 overlap：流水线、DP/TP/EP、kernel 与推理侧 |
| Kernel / 算子 | Triton、Liger-Kernel 等算子源码阅读与性能分析 |
| 优化器 | Muon / MuonClip 等新兴优化器原理与实现 |
| 量化与推理 | 量化方案细节、推理部署 |
| 扩散模型 | DDPM、SDE、DiT、flow matching、少步与视频生成 |
| vLLM 研读 | 推理引擎架构：PagedAttention、前缀缓存、调度、执行、分布式与调优 |
| AI Agent 系统工程 | loop、工具/MCP、context、Codex/Gemini harness、评测、CUA、多 agent、agentic RL、安全 |

## 仓库结构

```
content/posts/<article-name>/index.md   # 博客文章（Hugo + PaperMod）
code/<article-name>/                    # 文章配套可运行代码，目录名与文章一一对应
.github/workflows/deploy.yml            # GitHub Pages 自动部署
```

## 文章索引

### AI 日报（专题，长期更新，最新在上）

每天 21:00 抓 [HF Daily Papers](https://huggingface.co/papers) 当天全部论文，按主题分组写速览：Top 10 表 + 每篇「问题 / 方法 / 数字 / 为什么值得看」+ 三个观察。
导读与索引：[AI 日报（导读）](content/posts/ai-daily-00-index/index.md)。

1. [2026-11-13 · 34 篇：System One 的可靠性、双流世界模型与 32× 压缩](content/posts/ai-daily-2026-11-13/index.md)
2. [2026-11-12 · 21 篇：教科书监督、GTA V 智能体与推理成本](content/posts/ai-daily-2026-11-12/index.md)
3. [2026-11-11 · 39 篇：机器人上下文学习、权重级修复与「证据链」](content/posts/ai-daily-2026-11-11/index.md)
4. [2026-11-10 · 39 篇：触觉基础模型、世界模型基础设施与「观测对齐」](content/posts/ai-daily-2026-11-10/index.md)
5. [2026-11-09 · 34 篇：agent harness 工程化、世界模型与「可审计的评测」](content/posts/ai-daily-2026-11-09/index.md)
6. [2026-11-06 · 32 篇：决策模型、潜空间通信与「评测的可靠性」](content/posts/ai-daily-2026-11-06/index.md)
7. [2026-11-05 · 39 篇：回合级信用、机器人数据与推理的「物理账」](content/posts/ai-daily-2026-11-05/index.md)

配套代码：[code/ai-daily/](code/ai-daily/README.md)（`fetch_papers.py`：HF API + 镜像 + 页面内嵌 JSON 三级兜底，`--test` 离线自测）

### LLM 模型结构变迁（专题，9 篇）

从 GQA 到 MLA 到 CSA2、从线性混合到 896 专家、从残差加宽到条件记忆，沿「序列 / 宽度 / 深度 / 位置」四个方向
梳理 2023-2026 的结构演化。每篇配论文数字、config 行号与源码行号；专题首页含三根轴、七次搬家与阅读路线：
[LLM 模型结构变迁（导读）](content/posts/llm-arch-00-index/index.md)。

1. [导读：2026 年的结构地图——三个方向与七次搬家](content/posts/llm-arch-00-index/index.md)
2. [先把账本立起来：参数、算力、KV 三张预算表](content/posts/llm-arch-01-budget/index.md)
3. [注意力四代：从 GQA 到 MLA 到稀疏压缩](content/posts/llm-arch-02-attention/index.md)
4. [线性注意力与 3:1 混合：把 KV 变成状态](content/posts/llm-arch-03-linear/index.md)
5. [MoE 十年：从 8 专家到 896 专家](content/posts/llm-arch-04-moe/index.md)
6. [残差流的加宽：HC、mHC 与 AttnRes](content/posts/llm-arch-05-residual/index.md)
7. [位置的退场：RoPE、部分 RoPE 与 NoPE](content/posts/llm-arch-06-position/index.md)
8. [条件记忆：Engram、MTP 与 DSpark](content/posts/llm-arch-07-memory/index.md)
9. [五个收敛、三个分叉与三个赌注](content/posts/llm-arch-08-trends/index.md)
10. [残差连接的第二次生命：从 ResNet 到 AttnRes 与 Kimi-K3](content/posts/llm-arch-09-attnres/index.md)

配套代码：[code/llm-arch-evolution/](code/llm-arch-evolution/README.md)（纯标准库预算计算器，`python budget.py --test` 对拍 9 个模型）

### VLA 精读（专题，13 篇）

从动作表示、动作头、数据与本体、推理延迟四条线拆解 Vision-Language-Action：RT-2/Octo/OpenVLA 的 token 账、
π0/π0.5 的 block-causal mask 与 state 字符串、GR00T 的双系统与本体条件化、OXE 的归一化陷阱、
RTC 的 chunk 拼接，以及 2026 年的 RL 后训练与 latent 世界模型。每篇配论文页码、源码行号和可运行 CPU 演示；
专题首页含三条硬约束、四个设计轴与三条阅读路线：
[VLA 精读（导读）](content/posts/vla-00-index/index.md)。

1. [导读：让模型输出动作，究竟难在哪](content/posts/vla-00-index/index.md)
2. [动作表示：256 个 bin 的天花板](content/posts/vla-01-action-space/index.md)
3. [架构：从 RT-2 到 π0 的 token 账](content/posts/vla-02-architecture/index.md)
4. [动作头：diffusion 与 flow matching](content/posts/vla-03-action-head/index.md)
5. [π0：两个 Gemma、一条 mask，和 73 毫秒的账](content/posts/vla-04-pi0/index.md)
6. [π0.5：把 state 写成字符串，和「知识绝缘」的真相](content/posts/vla-05-pi05/index.md)
7. [GR00T N1/N1.7：双系统与「每个本体一套 MLP」](content/posts/vla-06-groot/index.md)
8. [数据：OXE 的 1M 轨迹为什么不够](content/posts/vla-07-data/index.md)
9. [推理系统：chunk、延迟预算与 RTC](content/posts/vla-08-serving/index.md)
10. [LeRobot 代码走读：π0/π0.5 的 PyTorch 版](content/posts/vla-09-lerobot-code/index.md)
11. [OpenVLA 代码走读：binning 与 OFT 的并行解码](content/posts/vla-10-openvla-code/index.md)
12. [评测：LIBERO 的 AUC 不是成功率](content/posts/vla-11-evaluation/index.md)
13. [2026 地图：RL、世界模型与饱和的榜单](content/posts/vla-12-2026-map/index.md)

配套代码：[code/vla/](code/vla/README.md)（7 个纯 CPU 脚本：分箱、mask、flow、归一化、chunk 延迟、RTC、置信区间，每个都有 `--test`）

### 扩散模型精读（专题，13 篇）

从「一条从噪声到数据的概率路径」出发，串起 DDPM 的变分下界、score SDE 的连续时间视角、DDIM/DPM-Solver 的少步采样、
classifier guidance 与 CFG、LDM 的 latent 压缩、DiT/MMDiT 的架构、flow matching 的直线路径、LCM/DMD 的蒸馏、
视频扩散的时空 token，以及训练/推理/评测的工程账。每篇配论文数字、公式和可运行 CPU 演示；
专题首页含五次搬家、四个设计轴与三条阅读路线：
[扩散模型精读（导读）](content/posts/diffusion-00-index/index.md)。

1. [导读：从去噪到传输，一条主线](content/posts/diffusion-00-index/index.md)
2. [DDPM：一千步去噪的账](content/posts/diffusion-01-ddpm/index.md)
3. [分数与 SDE：DDPM、SDE、ODE 是同一件事](content/posts/diffusion-02-score-sde/index.md)
4. [快速采样：DDIM、DPM-Solver 与 10 步的极限](content/posts/diffusion-03-fast-sampling/index.md)
5. [引导：从 classifier guidance 到 CFG](content/posts/diffusion-04-guidance/index.md)
6. [Latent Diffusion：把像素换成 latent](content/posts/diffusion-05-latent-diffusion/index.md)
7. [DiT 与 MMDiT：Transformer 接管去噪网络](content/posts/diffusion-06-dit/index.md)
8. [Flow Matching 与 Rectified Flow：把路径拉直](content/posts/diffusion-07-flow-matching/index.md)
9. [少步生成：一致性模型、LCM 与 DMD](content/posts/diffusion-08-few-step/index.md)
10. [视频扩散：时空 token 与 3D VAE](content/posts/diffusion-09-video/index.md)
11. [训练工程：EMA、CFG dropout 与显存账本](content/posts/diffusion-10-training/index.md)
12. [推理系统：步数、CFG 双 batch 与缓存](content/posts/diffusion-11-inference/index.md)
13. [评测与 2026 地图：FID 的坑与四个分叉](content/posts/diffusion-12-eval-2026/index.md)

配套代码：[code/diffusion/](code/diffusion/README.md)（9 个纯 CPU 脚本：噪声日程、DDPM、采样器、CFG、latent/DiT/视频预算、FM 对比、蒸馏、FID 偏差，每个都有 `--test`）

### vLLM 研读（专题，13 篇）

以 main @ c7a7607（v0.33.0 之后）的源码为基线，按「请求进来 → KV 分块 → 前缀复用 → 调度 → 执行 → 采样 → 投机 → 分布式 → KV 传输 → 调优」的顺序拆开一个推理引擎。
每篇从一个「账本」切入：显存、token 预算、通信量、延迟，配真实 `file:line` 与可运行的纯 CPU 模拟：
[vLLM 研读（专题）：推理引擎的四个账本](content/posts/vllm-00-index/index.md)。

1. [导读：推理引擎的四个账本](content/posts/vllm-00-index/index.md)
2. [请求的一生：从 HTTP JSON 到 token](content/posts/vllm-01-request-lifecycle/index.md)
3. [PagedAttention：块、块表与 KV 显存账本](content/posts/vllm-02-paged-attention/index.md)
4. [前缀缓存：一条哈希链省下多少 prefill](content/posts/vllm-03-prefix-caching/index.md)
5. [调度器：token budget、chunked prefill 与抢占](content/posts/vllm-04-scheduler/index.md)
6. [执行侧：InputBatch、attention metadata 与 CUDA Graph](content/posts/vllm-05-execution/index.md)
7. [Attention 后端：FlashAttention、FlashInfer 与 MLA 怎么选](content/posts/vllm-06-attention-backends/index.md)
8. [torch.compile：切分、融合与编译预算](content/posts/vllm-07-torch-compile/index.md)
9. [采样：logits 处理、logprobs 与结构化输出](content/posts/vllm-08-sampling/index.md)
10. [投机解码：n-gram、EAGLE 与拒绝采样](content/posts/vllm-09-spec-decode/index.md)
11. [分布式：TP/PP/DP/EP 与 Executor 的进程账](content/posts/vllm-10-distributed/index.md)
12. [KV 传输：connector、P/D 分离与 offload](content/posts/vllm-11-kv-transfer/index.md)
13. [量化、显存与调优：把模型塞进去并跑满](content/posts/vllm-12-quantization-tuning/index.md)

配套代码：[code/vllm/](code/vllm/README.md)（4 个纯 CPU 脚本：KV 预算、块分配、前缀缓存、调度模拟，每个都有 `--test`）

### SGLang 研读（专题，13 篇）

以 main @ 1ef3335（2026-11-03）的源码为基线，从「前缀树」这条主线拆开 SGLang：RadixAttention 的基数树、
KV 池与 slot 分配、PrefillAdder 的三重预算、两级 overlap（批间流水 + 批内 TBO）、extend/decode 两套
attention metadata、结构化输出、EAGLE 草稿树、DSL 与 prefix-aware router、PD 分离与 HiCache、EPLB，
最后一篇把它和 vLLM 放在同一负载下逐维度对照（含两者的参数映射表与选型决策树）：
[SGLang 研读（专题）：把前缀树当调度器用](content/posts/sglang-00-index/index.md)。

1. [导读：把前缀树当调度器用](content/posts/sglang-00-index/index.md)
2. [进程与请求的一生](content/posts/sglang-01-architecture/index.md)
3. [RadixAttention：用基数树做前缀缓存](content/posts/sglang-02-radix-attention/index.md)
4. [KV 池、slot 分配器与显存账本](content/posts/sglang-03-memory-pool/index.md)
5. [调度器：PrefillAdder、chunked prefill 与 retract decode](content/posts/sglang-04-scheduler/index.md)
6. [两级 overlap：批间流水、批内 TBO 与 CUDA Graph](content/posts/sglang-05-overlap/index.md)
7. [Attention 后端：extend 与 decode 两条路径](content/posts/sglang-06-attention-backends/index.md)
8. [结构化输出：压缩 FSM、跳词解码与 grammar 缓存](content/posts/sglang-07-constrained/index.md)
9. [投机解码：EAGLE 草稿树、draft-extend 与验证](content/posts/sglang-08-speculative/index.md)
10. [前端 DSL、Router 与 cache-aware 负载均衡](content/posts/sglang-09-frontend/index.md)
11. [PD 分离与 HiCache 三级缓存](content/posts/sglang-10-pd-disagg/index.md)
12. [TP/EP/DP-attention 与 EPLB](content/posts/sglang-11-distributed/index.md)
13. [总对比：同一道题，两种解法（vs vLLM）](content/posts/sglang-12-vs-vllm/index.md)

配套代码：[code/sglang/](code/sglang/README.md)（纯 CPU 脚本：树 vs 块对拍、显存预算、调度装箱、
overlap 时间线、跳词收益，每个都有 `--test`）

### AI Agent 系统工程（专题，13 篇）

从「模型 + harness + 环境」三个部件出发，拆开 agent 的 loop、工具/MCP、context 流水线、
Codex/Gemini/mini-swe-agent/OpenHands 四套 harness、评测指标、computer-use、多 agent 账本、agentic RL 与安全边界。
每篇配论文数字、源码行号和可运行 CPU 演示；专题首页含四本账、阅读地图与版本基线：
[AI Agent 系统工程（导读）](content/posts/agent-00-index/index.md)。

1. [导读：模型、Harness 与环境的三体问题](content/posts/agent-00-index/index.md)
2. [Agent Loop 解剖：一条消息如何变成一次工具调用](content/posts/agent-01-loop/index.md)
3. [工具层：schema、MCP 与 Code Mode](content/posts/agent-02-tools/index.md)
4. [Context Engineering：预算、压缩与信息衰减](content/posts/agent-03-context/index.md)
5. [Harness 解剖（一）：Codex CLI 的三层循环、沙箱与 apply_patch](content/posts/agent-04-codex/index.md)
6. [Harness 解剖（二）：Gemini CLI 的近 70 万行与 mini-swe-agent 的 190 行](content/posts/agent-05-gemini-mini/index.md)
7. [Harness 解剖（三）：OpenHands 的事件溯源、500 步熔断与可恢复会话](content/posts/agent-06-openhands/index.md)
8. [评测：SWE-bench、Terminal-Bench 与 Harness 混淆](content/posts/agent-06-eval/index.md)
9. [Computer-use 与浏览器 Agent：动作空间、grounding 与恢复](content/posts/agent-07-cua/index.md)
10. [多智能体：context 隔离、handoff 与 token 倍率](content/posts/agent-08-multiagent/index.md)
11. [Agentic RL：环境、reward 与 GSPO](content/posts/agent-09-rl/index.md)
12. [记忆与安全：从 MemGPT 到 CaMeL](content/posts/agent-10-memory-security/index.md)
13. [2026 论文地图与十个开放问题](content/posts/agent-11-trends/index.md)

配套代码：[code/agent/](code/agent/README.md)（9 个纯 CPU 脚本：循环模拟、MCP 握手、token 截断、
apply_patch、pass@k、坐标 grounding、多 agent 成本、GRPO/GSPO、taint 检查，每个都有 `--test`）

### 通算并行研读（专题，12 篇 + 补遗 4 篇）

从流水线气泡、DP bucket、TP 融合讲到 MoE all-to-all、persistent kernel 与推理侧 overlap。
每篇配论文数字、源码行号和可运行的 CPU 模型；专题首页含四种动作（切/错/融/让）、三条约束与阅读路线：
[通算并行研读（专题）](content/posts/comm-overlap-00-index/index.md)。

1. [先算清三笔账：关键路径、暴露通信与 overlap tax](content/posts/comm-overlap-01-performance-model/index.md)
2. [流水线气泡消灭史：GPipe → 1F1B → Zero Bubble](content/posts/comm-overlap-02-pipeline/index.md)
3. [DualPipe：双向流水如何把 bubble 压到 (PP/2-1)](content/posts/comm-overlap-03-dualpipe/index.md)
4. [数据并行：bucket、hook 与 reduce-scatter 的时差](content/posts/comm-overlap-04-data-parallel/index.md)
5. [张量并行：从 all-reduce 到 AG+GEMM 融合](content/posts/comm-overlap-05-tensor-parallel/index.md)
6. [MoE 专家并行：dispatch、expert、combine 的三段式](content/posts/comm-overlap-06-moe/index.md)
7. [Kernel 与 SM：persistent kernel、warp specialization 与分区](content/posts/comm-overlap-07-kernel-sm/index.md)
8. [编译与自动调度：从 Syncopate 到 Lagom](content/posts/comm-overlap-08-compiler/index.md)
9. [推理侧：decode 的通算并行与它的阈值](content/posts/comm-overlap-09-inference/index.md)
10. [实测手册：用 nsys/ncu 把 overlap 账算出来](content/posts/comm-overlap-10-measurement/index.md)
11. [总结：2026 年的地图与五个开放问题](content/posts/comm-overlap-11-summary/index.md)
12. [补遗·总览：2026 最新论文雷达与阅读路线](content/posts/comm-overlap-12-latest/index.md)
13. [补遗一：UniEP 深读](content/posts/comm-overlap-13-uniep/index.md)
14. [补遗二：MegaFlux 深读](content/posts/comm-overlap-14-megaflux/index.md)
15. [补遗三：Entwine、mKernel 与 2026 生态](content/posts/comm-overlap-15-tile-ecosystem/index.md)

配套代码：[code/comm-overlap/](code/comm-overlap/README.md)（7 个纯 CPU 模型，每个都能 `--test` 自测）

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
7. [对标分析：FA2 / FA3 / TE 与目标形状](content/posts/fa-bwd-07-analysis/index.md)
8. [性能调优历程与方法论](content/posts/fa-bwd-08-optimization-journey/index.md)
9. [目标卡移植注意事项](content/posts/fa-bwd-09-porting/index.md)

配套代码：[code/flash-attention/fa-bwd/](code/flash-attention/fa-bwd/README.md)
（各篇正文由 `docs/` 下的 Markdown 通过 `fa_include` shortcode 实时内联渲染）。

### KDA 前反向算子实现解析（专题，8 篇）

以 FlashKDA 为参考，把 **Kimi Delta Attention** 的前向/反向讲透：数学与分块公式 → K1/K2 kernel 精读
→ 数值对拍 → 反向推导与 kernel → 性能与优化。专题首页（含阅读路线与全目录）：
[KDA 前反向算子实现解析](content/posts/flash-kda-00-index/index.md)。

1. [数学与分块公式](content/posts/flash-kda-01-theory/index.md)
2. [前向 K1：prepare](content/posts/flash-kda-02-fwd-k1/index.md)
3. [前向 K2：recurrence](content/posts/flash-kda-03-fwd-k2/index.md)
4. [数值对拍](content/posts/flash-kda-04-numerics/index.md)
5. [反向：推导](content/posts/flash-kda-05-bwd/index.md)
6. [反向：kernel](content/posts/flash-kda-06-bwd-kernel/index.md)
7. [性能与优化](content/posts/flash-kda-07-perf/index.md)
8. [一页纸汇总](content/posts/flash-kda-08-summary/index.md)

配套代码：[code/flash-kda/](code/flash-kda/README.md)

### MegaKernel 资料与实现分析（专题，75 篇）

把「整张算子图融进一个常驻 kernel」讲清楚：动机与判据 → MPK 论文与编译栈 → 运行时与调度 →
生产 task 目录 → 存储布局与硬件原语 → 实测手册。专题首页（含 6 条阅读路线与按主题的篇目地图）：
[MegaKernel 资料与实现分析](content/posts/mega-kernel-00-index/index.md)。

入门三篇：[MegaKernel 是什么](content/posts/mega-kernel-01-overview/index.md)、
[MPK 论文：tGraph](content/posts/mega-kernel-02-paper/index.md)、
[一页纸总结](content/posts/mega-kernel-06-summary/index.md)。

配套代码：[code/mega-kernel/](code/mega-kernel/README.md)

### Mirage / MPK 项目分析（专题，4 篇）

一个开源项目的上手路径：两条产品线与仓库地图 → 从 HF 模型到 megakernel 的编译链路 →
生成的 kernel 怎么跑 → 本机构建与实跑记录。专题首页：
[Mirage / MPK 项目分析](content/posts/mirage-00-index/index.md)。

1. [项目全景](content/posts/mirage-01-overview/index.md)
2. [从 HF 模型到 megakernel](content/posts/mirage-02-endtoend/index.md)
3. [生成的 megakernel 怎么跑](content/posts/mirage-03-runtime/index.md)
4. [构建与实跑记录](content/posts/mirage-04-build/index.md)

配套材料：[code/mirage/](code/mirage/README.md)

### paper阅读（专题，连载中）

逐篇精读值得读的系统/算法论文：讲清真实问题、设计取舍背后的物理约束、以及能迁移的经验。

1. [DSec 精读：DeepSeek 的 Agentic RL 沙箱平台，为什么不是「一个容器运行时」](content/posts/paper-reading-01-dsec/index.md)
2. [DeepSeek-V4.1-Flash 精读：把 KV Cache 压到 890 bytes/token，架构、精度、部署三线并进](content/posts/paper-reading-02-dsv41-flash/index.md)

### LeetCode 题解精讲（专题，连载中）

把 LeetCode 高频经典题按数据结构/算法模式分类，由易到难、相似题成组，
每道题配 Python 与 C++ 可运行代码，并总结可迁移的通用规律。路线图见 [code/leetcode/ROADMAP.md](code/leetcode/ROADMAP.md)。

1. [专题导读：分类目录与学习路线](content/posts/leetcode-00-index/index.md)（全 35 篇目录）
2. [数组与双指针](content/posts/leetcode-01-array/index.md)
3. [哈希表](content/posts/leetcode-02-hash/index.md)
4. [滑动窗口](content/posts/leetcode-03-sliding-window/index.md)
5. [前缀和与差分](content/posts/leetcode-04-prefix-sum/index.md)
6. [二分查找](content/posts/leetcode-05-binary-search/index.md)

动态规划专题（多篇）：[动态规划（一）线性/序列/背包/网格/区间/状态机](content/posts/leetcode-13-dynamic-programming/index.md)、
[数位 DP](content/posts/leetcode-31-digit-dp/index.md)、[状态压缩](content/posts/leetcode-32-bitmask/index.md)、
[树形 DP](content/posts/leetcode-34-tree-dp/index.md)、[动态规划（二）DP 优化](content/posts/leetcode-35-dp-optimization/index.md)。

配套代码：[code/leetcode/](code/leetcode/README.md)
（各篇正文由 `docs/` 下的 Markdown 通过 `lc_include` shortcode 实时内联渲染）。

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
