# 通算并行研读：资料与研究笔记

> 系列 slug：`comm-overlap`。本文件是写作素材，不是文章正文。
> 所有数字均来自本地论文文本（`~/github/overlap-research/html/`）或论文/仓库公开页面；
> 引用时保留论文自己的口径与硬件条件。

## 0. 一句话框架

通算并行不是"把通信提前发、把 wait 推后"这么一句。它是在**依赖约束**下，
对**通信与计算这两类工作**做四种操作之一：

| 动作 | 含义 | 典型手段 | 代表 |
|---|---|---|---|
| **切** (split) | 把张量/算子/批次切成更小的独立单元，制造可并行的窗口 | bucket、microbatch、chunk、tile、nano-batch | Megatron bucket、Flux tile、Comet shared tensor |
| **错** (stagger) | 重排执行顺序，让 A 的通信落在 B 的计算窗口里 | 1F1B、Zero Bubble、DualPipe、wavefront | GPipe/1F1B/ZB/DualPipe |
| **融** (fuse) | 把通信指令塞进计算 kernel / 让通信走专用硬件 | AG+GEMM、GEMM+RS、TMA、NVSHMEM、in-network | Flux、T-CCL、INC-MoE |
| **让** (yield) | 显式分配 SM/warp/带宽/流给通信或计算 | SM 分区、warp specialization、NCCL 参数 | Comet、Weave、Lagom |

三条硬约束：

1. **依赖**：数据没准备好就不能算；细粒度重叠的前提是"部分准备好"。
2. **资源**：通信不是免费的，它吃 SM、HBM/L2 带宽、功耗；overlap 有 tax。
3. **语义**：梯度必须对、optimizer 必须同步、数值必须可复现；有些 overlap 会改变语义（CO2 的 stale update）。

## 1. 性能账本

对一层/一步：

$$
T = T_{\text{compute}} + T_{\text{exposed}} + T_{\text{bubble}}
$$

- 通信总量 $T_{\text{comm}}$，被藏住的比例（overlap efficiency）：
  $$
  \eta = 1 - \frac{T_{\text{exposed}}}{T_{\text{comm}}}
  $$
- 流水线气泡率：$\text{bubble rate} = T_{\text{bubble}} / T$。
- 一个常被忽略的第三项：**overlap tax**。通信与计算争 SM/带宽，会让计算本身变慢。
  Characterizing（2507.03114）测得平均 **18.9%**、最高 **40%** 的计算减速；
  串行反而只比 overlap 慢 **10.2%**（平均）。

### 1.1 流水线

- 1F1B（ZB 论文 Table 2 口径）：$\text{bubble}=(p-1)(T_F+T_B+T_W)$，
  $T_{\text{ideal}}=m(T_F+T_B+T_W)$。
  $p=8,m=24,F=B=W=1$：$21/93=22.58\%$（论文实测 24.31%，因 $T_F/T_B/T_W$ 不等）。
- ZB-H2：$\text{bubble}=(p-1)(T_F+T_B-2T_W)$；$W$ 足够大时为 0。
- DualPipe：$\text{bubble}=(PP/2-1)(F\&B+B-3W)$；参数量 2×，激活 $PP+1$。
- 关键点：**"零气泡"是跨 step 的稳态性质**。单步仍有 $(p-1)T_F$ 的 fill；
  只有把 optimizer step 的同步也藏掉（ZB 的 optimizer post-validation / DualPipe 的跨 chunk 排布），
  才能让 W 的尾巴接上下一轮 F。

### 1.2 DP bucket

$N$ 个 bucket，单桶计算 $c$、通信 $t$（含 launch），单通道：

$$
T_{\text{exposed}} = t + (N-1)\max(0, t-c)
$$

- $t<c$：暴露就是一个尾桶 $t$，桶越细越好（但 launch 会反噬）。
- $t>c$：每桶落后 $t-c$，暴露随 $N$ 线性增长。
- 最优桶数由"尾延迟 vs launch/带宽"共同决定；`bucket_sim.py --sweep` 可复现凹陷。

### 1.3 TP

一段 all-gather + GEMM，切成 $n$ 块：

$$
T = \max(nc, ng) + \min(c,g)
$$

- $g$ 是每块通信，$c$ 是每块计算。$n=1$ 时 $T=c+g$，没有重叠。
- 重叠上限由 $\max(c,g)$ 决定；通信比计算慢时，再怎么切也藏不完。

### 1.4 MoE

dispatch → expert GEMM → combine。三段流水，通信通道与计算通道各一条：

- 单 chunk：$T=d+c+b$。
- 多 chunk：$T \approx \max(\sum c, \sum d+\sum b) + \text{startup}$。
- 真实系统还有 routing skew、shared expert、EP/TP 交叉，见 Comet/DeepEP/Weave。

## 2. 论文谱系

### 2.1 流水线调度

| 论文 | ID | 机制 | 关键数字 |
|---|---|---|---|
| GPipe | 1811.06965 | 全 F 再全 B，microbatch 填 pipeline | 经典 bubble $(p-1)/(m+p-1)$ |
| 1F1B | 2104.04473 (Megatron-LM) | warmup 后 F/B 交替 | 内存从 $m$ 降到 $p$ 量级 |
| Zero Bubble | 2401.10241 | 拆 B 为 B/W，W 填气泡；自动调度；optimizer 同步 bypass | 比 1F1B 最高 **+23%**，放宽内存 **+31%**；p=8,m=24 气泡 24.31%→0.29% |
| DualPipe | 2412.19437 / deepseek-ai/DualPipe | 双向流水，每卡两个 chunk，F/B 跨方向重叠 | 气泡 $(PP/2-1)(F\&B+B-3W)$；V3 训练 2.788M H800 GPU hours |
| DualPipeV | Sea AI Lab "cut-in-half" | V 形，PP stage 减半 | 2× 参数，$PP/2$ 设备 |
| STP | 2510.27257 | TP×PP 编织，把 TP all-reduce 塞进 PP 的计算 | +12.2% LLM / +16.7% MLLM；TP bubble 27.5% @TP8/seq6144 |
| FlowMoE | 2510.00207 | 整个 transformer block 统一流水，AR 低优先级填隙 | 时间 −13%~−57%，SM 48–89% |

### 2.2 DP / ZeRO / FSDP

| 论文/系统 | 机制 | 关键数字 |
|---|---|---|
| Megatron DDP | bucket 化 + backward hook + async reduce-scatter | `overlap_grad_reduce` 默认 False |
| ZeRO / FSDP | param all-gather 预取、grad reduce-scatter | 通信量与参数量成正比 |
| MegaScale | DP prefetch、PP send/recv 解耦、TP/SP 融进 Linear | 12,288 GPU MFU **55.2%**，1.34× |
| Domino | TP 的 X 行切 + B 列切，AllReduce 与相邻算子流水 | 最高 **1.3×**，达无通信上限 >90% 单节点 |
| CO2 | local-SGD，τ 步不同步，异步 all-reduce | 宣称 100% 扩展性；τ 大泛化差 |

### 2.3 TP / kernel 融合

| 论文/系统 | 机制 | 关键数字 |
|---|---|---|
| Flux | tile 级 AG+GEMM / GEMM+RS 融合进一个 kernel | overlap efficiency 最高 **96%**；训练 1.24×，prefill 1.66×，decode 1.30×（跨集群最大） |
| T-CCL | TMA 卸载 collective，单线程发起 | vs NCCL 最高 **2.4×**（不限）/ **3.42×**（限 8/9 CTA）；AG 128KB 占 1.32 SM vs NCCL 3.30 |
| TileLink | tile-centric primitive，编译器生成 | 1.17×–20.76×；AG+GEMM 1.27×；MoE 层 1.14× vs vLLM |
| CommFuse | RS/AG 拆 ring P2P，rank-adaptive 排序消 tail | 通信开销 43.8→0.1 ms；时延 −36.9% |
| FlashOverlap | tile 完成即 signal，重排后调 NCCL | 1.02–1.65×；192KB tile 仅 13% 带宽 |

### 2.4 MoE / EP

| 论文/系统 | 机制 | 关键数字 |
|---|---|---|
| DeepEP | dispatch/combine kernel，低延迟模式，NVLink/RDMA | 开源；SM/QP 可配；`EventOverlap` |
| Comet | shared tensor 依赖分析 + CTA specialization | 通信占 **47%**；层 1.96×，e2e 1.71×；隐藏 86.5% |
| Lancet | 整图 overlap，前向切 non-MoE，后向 dW 塞 A2A | 非重叠通信 −77%；1.3× |
| Weave | megakernel 内动态 SM 调度 + chunk pipeline | 层 2.89×，e2e 1.33×；overlap 47.1% vs Comet 14.3%/DeepEP 9.2% |
| DisagMoE | attention/FFN 拆到不同 GPU 组，AF-Pipe | 1.81× vs Megatron-1F1B；1.34× vs SOTA |
| INC-MoE | 交换机内 AG/RS，双向链路并发 | TTFT −33.1%，吞吐 +49.5%（prefill） |
| Analytical RM | wave-quantized 解析模型选 CTA 数 | 层 1.771×；regret 3.22% |
| Tile-level signaling | 只打回程 A2A，14/108 SM 常驻通信 | 最高 2.64×；cCTA=2 反而更慢 |

### 2.5 推理 / serving

| 论文/系统 | 机制 | 关键数字 |
|---|---|---|
| NanoFlow | nano-batch + 复制算子 + MILP 资源分配 | 吞吐 **1.91×**；达最优 68.5% |
| Flash Communication | INT6 量化 TP all-reduce（不是 overlap） | 通信 >3×；TTFT 2.06×（L40）/1.19×（A100） |
| TokenWeave | token 切两半 + fused AR-RMSNorm | 1.28× 延迟，1.19× 吞吐；fused kernel 2–8 SM；<1K 负收益 |
| T-CCL + vLLM | 低 SM collective 后端 | 对话 1.10×，decode 1.15× |

### 2.6 编译器 / 自动调优

| 论文/系统 | 机制 | 关键数字 |
|---|---|---|
| Centauri | 通信分区 + 调度（ASPLOS'24） | 需查原文；本系列只作背景 |
| TE-CCL / UserBuffers | NVIDIA TE 的通信-计算融合路径 | 代码在 TransformerEngine；本系列引 `ub_cfgs` |
| Triton-distributed | Triton 编译器生成 overlap kernel | 2504.19442；TileLink 的后端 |
| Syncopate | communication chunk 抽象 + Triton 源到源 | 平均 1.3×，最高 4.7×；chunk 最优 2–3 |
| Lagom | 联合调 NC/NT/C，cost model + priority search | 1.07–1.33× vs NCCL；AutoCCL 0.87× 反例 |
| CommFuse | 分解 + 融合消 tail | 见 2.3 |
| HOCCL | collective 从 GPU core 卸载 | 2609.34334（背景） |

## 3. 代码索引

| 仓库 | 本地路径 | 关键文件 | 用途 |
|---|---|---|---|
| Megatron-LM | `~/proj/Megatron-LM` @ `f713506ce` | `megatron/core/distributed/{distributed_data_parallel.py,param_and_grad_buffer.py,distributed_data_parallel_config.py}`；`pipeline_parallel/schedules.py` | DP bucket/hook；PP schedule |
| DualPipe | `~/github/overlap-research/DualPipe` | `dualpipe/dualpipe.py`（440 行）、`dualpipe/dualpipev.py`、`dualpipe/utils.py`（WeightGradStore） | 双向流水调度 |
| Flux / Comet | `~/github/overlap-research/flux` | `src/ag_gemm/sm90_all_gather_gemm_tma_warpspecialized_cooperative.hpp`；`src/moe_gather_rs/gemm_grouped_v3_gather_rs.cc`（`gather_rs_ctas=28`）；`docs/design.md` | AG+GEMM、GEMM+RS、CTA 分区 |
| DeepEP | `~/github/DeepEP` @ `b5a33eef` | `deep_ep/buffers/ep.py`（dispatch 562、combine 770）；`deep_ep/utils/event.py` | MoE A2A overlap API |
| TransformerEngine | `~/github/TransformerEngine` @ `172bd937` | `transformer_engine/pytorch/module/base.py:607` `fill_userbuffers_buffer_for_all_gather`；`module/linear.py:414` | TE 的 userbuffers 路径 |
| vLLM | `~/github/vllm` | 推理侧 TP/EP 入口 | 背景 |
| TileLang / Triton | `~/github/tilelang`、`~/github/triton` | 编译器背景 | 背景 |

## 4. 可直接引用的数字（带口径）

- Comet：MoE 层通信占 **47%**；隐藏 **86.5%** 通信；单层 **1.96×**，e2e **1.71×**（8×H800）。
- Weave：overlap ratio **47.1%** vs Comet 14.3% / DeepEP 9.2% / PK 5.8% / TD 5.1%；
  GEMM 中段仍有 mid bubble，91% SM 活跃；层 **2.89×**，e2e **1.33×**（4×H100, EP=4）。
- Characterizing：overlap 让计算平均慢 **18.9%**、最高 **40%**；串行只慢 **10.2%**；
  峰值功耗 +25%；H100 平均 38% TDP、峰值 140% TDP（4×A100/H100/MI250/MI210）。
- Flux：overlap efficiency 最高 **96%**（A100 NVLink），H800 NVLink **37–93%**；
  H800 上 decode **没有加速**；小 m 时 **0.95×** 慢于基线。
- TokenWeave：TP 通信 **20%**（最高 23%）；fused kernel **2–8 SM**（此前 16–20+）；
  **1.28×** 延迟 / **1.19×** 吞吐；Mixtral-8x22B 在 1K/2K 净开销。
- T-CCL：AG 128KB 活跃 SM **1.32** vs NCCL **3.30**；同算法换后端 1.04×→1.14×（TP=4）。
- Syncopate：平均 **1.3×**、最高 **4.7×**；tile 顺序单独造成 **>2×** 差异；chunk 最优 **2–3**。
- Lagom：计算瓶颈下多给通信资源可把性能拖到 **0.87×**；最优 (NC=2, C=684KB) **1.35×**。
- Zero Bubble：1F1B **24.31%** → ZB-H2 **0.29%**（p=8,m=24）；吞吐 +23%/+31%。
- DualPipe：V3 每个 rank 20 个 SM 做通信；NVLink 160 GB/s vs IB 50 GB/s；
  up to 4 nodes / 3.2 experts per node / 13 max；1:1 compute:comm 比。

## 5. 文章分工与交叉引用

| # | slug | 主题 | 主要素材 |
|---|---|---|---|
| 00 | `comm-overlap-00-index` | 导读：四种动作、三个约束、一张地图 | 本笔记 §0–2 |
| 01 | `comm-overlap-01-performance-model` | 性能账本：关键路径、气泡、overlap tax | `overlap_model.py`、`schedule_sim.py` |
| 02 | `comm-overlap-02-pipeline` | 流水线气泡消灭史：GPipe→1F1B→ZB | 2401.10241、Megatron schedules |
| 03 | `comm-overlap-03-dualpipe` | DualPipe：双向流水与全 overlap | DualPipe 代码、2412.19437 |
| 04 | `comm-overlap-04-data-parallel` | DP bucket/hook/reduce-scatter | Megatron 代码、`bucket_sim.py` |
| 05 | `comm-overlap-05-tensor-parallel` | TP：从 all-reduce 到 fused GEMM | Flux、T-CCL、TileLink、CommFuse |
| 06 | `comm-overlap-06-moe` | MoE EP：A2A 的三段式 | DeepEP、Comet、Lancet、Weave、DisagMoE |
| 07 | `comm-overlap-07-kernel-sm` | Kernel/SM 层：persistent、warp spec、SM 分区 | Weave、T-CCL、TokenWeave、Characterizing |
| 08 | `comm-overlap-08-compiler` | 编译与自动调度 | Syncopate、Lagom、TileLink、Triton-distributed |
| 09 | `comm-overlap-09-inference` | 推理侧：decode 的通算并行 | NanoFlow、FlashComm、TokenWeave、T-CCL |
| 10 | `comm-overlap-10-measurement` | 实测手册：nsys/ncu、指标、反模式 | Characterizing、`overlap_audit.py` |
| 11 | `comm-overlap-11-summary` | 总结：2026 地图与开放问题 | 全部 |
| 12 | `comm-overlap-12-latest` | 2026-11 补遗：UniEP、MegaFlux、Entwine、mKernel 等 | arXiv 2604.19241 等；见文末补遗表 |

## 6. 写作口径

- 论文数字必须带硬件/模型/配置；跨集群取最大值时注明。
- 代码行号以本地 commit 为准，引用符号名优先。
- 模拟器输出标注"模型，不是实测"。
- 负结果照写：Flux H800 decode 无加速、TokenWeave <1K 变慢、Lagom AutoCCL 0.87×、
  Tile-level cCTA=2 更慢、Characterizing overlap 让计算慢 18.9%。

---

## 2026-11 补遗：UniEP 与最新论文（arXiv 扫描）

> 触发：正文漏掉 2026-04 的 UniEP。2026-11-03 用四组 arXiv 查询按提交时间倒序扫到 117 条 2025-10 后候选，人工筛 15 篇。

### 重点

| 系统 | arXiv | 硬件/口径 | 关键数字 | 代码 |
|---|---|---|---|---|
| UniEP | 2604.19241 | 2×Hopper 8 卡集群（200/400 GB/s/dir） | 1.03–1.38× over SOTA；kernel 1.30–1.57× vs Comet；128 GPU 127→138 B tokens/day（1.09×）；bitwise | 论文称在 `ByteDance-Seed/Triton-distributed`，main 未见独立目录 |
| MegaFlux | 2610.00671 | 8×B200，147 配置/方向 | fwd 1.45×、bwd 1.28×，峰值 2.14/2.64；vLLM 1.13–1.26× | 未见公开 |
| Entwine | 2609.11562 | TP LLM workload | 1.232× geo（最高 1.433×）vs cuBLAS+NCCL；+3.1–9.8% vs SOTA | 论文称发表后开源 |
| mKernel | 2609.13585 | 2×16 H200 | GEMM+AR 1.72×；Ring Attention 1.88× | `uccl-project/mKernel` |
| Mixture-of-Kittens | 2609.36070 | NVL72 / 512 GPU | 最高 2.37×；生产 1.41× | `cursor/mixture-of-kittens` |
| NCCL EP | 2603.13606 | H100 多节点 | `ncclEpDispatch/Combine`；LL 1–128 token，HT 4096+ | 并入 NCCL 生态 |
| ThunderEP | 2609.40093 | RTX 4090/5090 PCIe | dispatch 2.00×、combine 1.53×、e2e 1.66× | 未见公开 |
| FarSkip-Collective | 2511.11505 | DeepSeek-V3 / Llama4 Scout | prefill overlap 97.3%；TTFT 32.6%；训练 A2A 88.9% | 未确认 |
| HyperParallel-MoE | 2605.23764 | Ascend A3 | Dispatch-to-Combine 1.58× | gitcode mindspore |
| Piper | 2605.05049 | HPC | MFU 2–3.5× vs X-MoE；A2A 带宽 1.2–9× | 未见公开 |
| HybridEP | 2510.19470 | 跨 DC | 受限带宽最高 5.6×；1k DC 1.45× | 未见公开 |
| FlowMoE | 2510.00207 | MoE 训练 | 时间 −13–57%、能耗 −10–39% | 未见公开 |
| Resource-aware overlap | 2606.09200 | A40/A100/H100/MI250X | 最多 −25.5% 时间 | 未见公开 |
| ParallelKittens | 2511.13940 | Hopper/Blackwell | DP/TP 2.33×、SP 4.08×、EP 1.22× | 未确认 |
| Iris | 2511.12500 | Triton | GEMM+AllScatter 1.79× | 未确认 |

### 对正文的修正

1. 语义约束增加“逐位可复现”（deterministic/bitwise），不只是梯度正确。
2. 单 CUDA stream + persistent kernel + 动态 SM 角色也能做 token/tile 级 overlap。
3. 自动调参从加分项变成必选项；但搜索成本本身是工程负担。
4. 硬件分叉：scale-up（NVL72）、scale-out（IB）、PCIe 消费卡、Ascend 各有最优解。
