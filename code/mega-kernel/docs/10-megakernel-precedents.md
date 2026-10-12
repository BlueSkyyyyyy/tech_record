# 手工 MegaKernel 先例精读：FlashMoE 与 Llama-1B megakernel

> 本文是「MegaKernel 资料与实现分析」的第 11 篇，对应任务 **P27-1**。
> 定位：本专题此前把 MPK（自动整图）、DeepGEMM mega 家族（手工段级）读透，但 MPK 反复点名的两个
> **手工 mega-kernel 先例**——**FlashMoE**（单 kernel 分布式 MoE）与 **Spector 等人的 Llama-1B
> 低延迟 megakernel**——一直只作为 MPK 的转引出现（`docs/00` §4.2、`docs/03` §3、`docs/05` §5），
> 从未回到一手资料。本文补上这块：直接精读两篇一手材料，钉下章节号与真实链接，并与 MPK / DeepGEMM /
> 本专题实验逐维对照。
>
> **性质：纯资料分析。** 两个先例分别要求 8×H100 + NVSHMEM 多进程（FlashMoE）和完整的 Llama 权重/
> 编译工具链（Llama-1B megakernel），本机单卡容器内不具备端到端复现条件，故本轮不新增实机实验；
> 与之相关的**机制**已有本专题实测（launch 开销 P5-1、跨算子重叠 P20-1、去中心化调度 P15-1 等），
> 下文逐条交叉引用。

## 0. 为什么是这两个先例

MPK 论文在**两处**把「手工 mega-kernel」作为自己的对照物（均为本机快照
`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`，逐行核对）：

- **§2.2（快照行 325–344）**：「existing compilers can only fuse small groups of local operators …
  Existing mega-kernels are instead handcrafted … For example, **FlashDMoE** fuses mixture-of-experts
  computation and inter-GPU communication into a single kernel [4], while **Spector et al.** manually
  designed and implemented a low-latency mega-kernel for LLAMA-1B [35,11].」（行 333–343）
- **§8 Related Work（快照行 1851–1861）**：同两点，并补一句判断——手工做法
  「require extensive engineering effort and deep GPU expertise, and they **do not generalize across
  models or GPUs**」（行 1861–1862），因此 MPK 走编译器路线。

从 MPK 的参考文献可核到**一手来源**（真实链接，非编造）：

| 先例 | 一手材料 | 链接 | MPK 引用编号 |
|---|---|---|---|
| **FlashMoE** | Aimuyo, Oh, Singh, *FlashMoE: Fast Distributed MoE in a Single Kernel*, **NeurIPS '25**（arXiv v3, 2025-11-08） | 论文 `https://arxiv.org/abs/2506.04667`；代码 `https://github.com/osayamenja/FlashMoE` | [4]（快照行 1958–1965） |
| **Llama-1B megakernel** | Spector, Juravsky, Sul, Dugan, Lim, Fu, Arora, Ré，*Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B*，Hazy Research 博客（2025-05-27） | 博客 `https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles`；代码 `https://github.com/HazyResearch/Megakernels` | [11]（快照行 2041–2049）+ LLaMA 论文 [35] |

> 本节把 MPK 快照行号、论文/博客真实章节号与两篇一手材料的内容对齐；两篇均已在本轮实际抓取核对。
> 注意 MPK 快照把它写作 *FlashDMoE*，一手论文标题是 *FlashMoE*（作者在 v2/v3 改名），指同一工作。

**这两个先例在谱系中的位置**（对照 `docs/00` §4 的 T1–T4 与 §3.2 的判据 C1–C5）：

- **FlashMoE** = **段级 T4**（把 MoE 一整个分布式算子段融进单 kernel，与本专题 `docs/02` §1 的
  DeepGEMM Mega MoE 同一档），但调度更「动态」——用**设备内 actor 调度器**按 tile ready 派活。
- **Llama-1B megakernel** = **整图 T4**（整个 Llama-1B 前向 100 个算子在单 kernel 内），是 MPK 想
  自动化的那一档的**手工对照组**。

---

## 1. FlashMoE：把分布式 MoE 融进一个常驻 kernel

> 来源：arXiv:2506.04667v3（NeurIPS '25）。下文 §号均为论文真实章节号，已抓取核对。
> 评测平台：单节点 **8×H100 80G + NVLink**，MoE 最多 128 experts、序列 16K tokens（§4）。

### 1.1 动机：同步集合通信 + 频繁 launch（论文 §1、§2）

FlashMoE 把 MoE 的两大瓶颈归结为「**CPU 管理的调度 + host 发起的通信 + 频繁 kernel launch**」
（Abstract）。论文 §1 给的三条 launch 危害与本专题 `docs/00` §1 的三类损失几乎逐字对应：

1. **跨 GPU 内核启动不确定** → 放大 **straggler**（对应损失一 kernel barrier 在分布式下的表现）；
2. **多余的同步点** → GPU 等 peer 或 CPU（对应损失一）；
3. **kernel 边界反复 global memory 往返**（对应**损失三**）。

论文还点名 **CUDA Graph 只能缓解第一条、且与 MoE 的动态路由不兼容**（§1）——这与本专题
`docs/03` §1 对 CUDA Graph「静态图、动负载需重捕」的定位，以及 `docs/02` §3.2（Kimi K3 把
graph replay 判为作弊）的判断一致。

**launch 次数的硬证据**（论文 §1 **Table 1**，单 MoE 层 Gate→Dispatch→Expert→Combine，2×A100、
每卡 32 experts，用 Nsight Systems 计数）：

| 实现 | 单层 GPU op 数 |
|---|---|
| **FlashMoE（本文）** | **1** |
| COMET | 33 |
| Megatron-LM CUTLASS | 85 |
| Megatron-LM TE | 261 |
| Megatron-LM + DeepEP | 432 |
| DeepSpeedMoE | 550 |

即 FlashMoE 把「最多 550 个短命 kernel」压成 **1 个常驻 kernel**（C1+C2），并把 GPU SM utilization
从 DeepEP 的 **14%**（论文 §2 用 Nsight Systems CUDA API trace 量得）提到 **93.17%**（§4.2）。

### 1.2 设计：actor 模型 + tile 级 task + 对称张量布局（论文 §3）

**（a）actor 模型与静态角色（§3 概览 + Figure 5/6）。** FlashMoE 把每个 GPU 的线程块划分为三类
**actor**（论文 §3、附录 §F）：

- **Processor**（$N-1$ 个块）：干活——两个 GEMM 与逐元素操作、tile 通信；
- **Subscriber**（OS 块里 3 个 warp）：解码来自其他 GPU 的 *tile packet* 成 task descriptor；
- **Scheduler**（OS 块里 1 个 warp）：把计算 task 派给空闲 Processor 块。论文强调它
  **多线程化（提高派发吞吐）+ work-conserving（保证 SM 不闲）**。

最后一整个块作为 **OS 块（4 warp = 3 subscriber + 1 scheduler）**，只占极少资源，其余全部给计算。
Algorithm 1（§3）给出近乎源码级的骨架：`FusedGate` → `Dispatch` → 非 OS 块跑 `processor::start()`、
OS 块 `warpID==0` 跑 `scheduler::start()`、否则 `subscriber::start()`。

> **与本专题对照**：这正是 `docs/00` §3.2 的 **C5**——调度粒度细到 **block/warp 级 task**，
> 且调度器在**设备内**。与本专题 T4 最小骨架 `docs/04` §9（P15-1 去中心化调度：单 kernel +
> 设备侧原子 ready 队列 + 每 task 事件计数器）是同一机制，只是 FlashMoE 额外把**跨 GPU 通信**
> 也纳入 task 图。与本专题 P22-1 §15 的 `decen`（逐 task 原子 ready 队列）也同构。

**（b）统一 task 抽象（§3.1）。** task descriptor 是三元组 $t=(\mathcal{M}, \star, \phi)$——
元数据、二元张量算子（矩阵乘 $\cdot$ 或 Hadamard $\odot$）、逐元素激活。其语义（式 4）为
$\mathcal{F}_t(A,B,C,D): C \leftarrow \phi(A \star_t B + D)$。关键工程点：**每个 task 实现为一个
`__device__` 函数**，Processor 在运行时按描述符调用；FFN 是 $t_1,t_2$ 两次矩阵乘 + 激活链，
expert-combine 是一次 Hadamard + 累加。这与 MPK `docs/07`「一个 task = 一个 CTA / 一个 device
函数」的粒度**高度一致**——都是把「调用哪段设备代码」做成**可被调度器动态选择的数据**。

**（c）对称张量布局 + 非阻塞通信（§3.2）。** 跨 GPU 用 **device-initiated、one-sided 的 NVSHMEM
PGAS** 通信（而非 AlltoAll 集合通信）。为在无锁下保证正确，定义对称布局
$L \in \mathbb{R}^{P\times R\times B\times E\times C\times H}$（$P$=EP world size、$R$=通信轮、
$B$=staging buffer、$C$=capacity、$H$=hidden），核心手段是 **temporal buffering**——把 token 矩阵
**过分配至少 $2r$ 倍**（MoE 里 $2r=4$），从而所有访问 **write-write conflict-free**
（**Theorem 3.1**，证明在附录 §C），省掉一次性传输中的同步。§3.2.1 再给 **in-place padding**：
把 padding 做在本地对称缓冲内、并按 tile 高度 $b_M=128$ 对齐 capacity，**不把空 token 发到网络**
（payload-efficient），代价是布局约 $4\times$ token buffer（推理下 ≤2% 显存，附录 §D）。

> **与本专题对照**：对称显存 + 非阻塞访问，正是 DeepGEMM Mega MoE 的 symmetric memory（`docs/02`
> §1.1/§1.5）与 MPK 的 ring buffer / NVSHMEM allreduce task（`docs/07` §3）的同一族技术。
> FlashMoE 的「过分配 $2r$ 倍换无锁」是这条技术线上一份**可核对的正确性论证**（有定理 + 证明）。

### 1.3 评估（论文 §4）

- **延迟（§4.1）**：4 GPU 上对 Megatron-TE 最高 **4.6×**（16K tokens）；8 GPU 上最高 **6.4×**。
- **GPU 利用率（§4.2）**：**93.17%** 平均 SM 利用率，是 FasterMoE（9.67%）的 **9×+**、DeepEP+Megatron
  （13.55%）的 **6.8×**、Megatron-TE（59.11%）的 4×。
- **吞吐（§4.3）**：8 GPU 达 **17.7 MTokens/s**，比基线高 **5.7×**（且 FlashMoE 全程 **FP32**，
  基线是 FP16——即它在**精度吃亏**下仍赢）。
- **overlap efficiency（§4.4）**：weak scaling 定义 $O_e=T(2)/T(N_G)$，8 GPU 上比基线高 **4×**。
- **expert 可扩展性（§4.5）**：experts 从 8→128，FlashMoE 延迟**平稳**，基线超线性增长，128 experts
  时领先最高 **6.6×**。

> **与本专题实测对照**：FlashMoE §4.2 的「SM 一直有活干」= 本专题 P10-1 §5（persistent 网格按
> **占用率**而非 SM 数选、过配浪费）、P14-1 §8（占用率被最重 task 拉低导致带宽掉 1.33–2.29×）的
> **正面目标**；它用 actor 调度 + tile 级 work-conserving 把占用率顶到 93%。§4.4 的 overlap 又对上
> P20-1 §14（跨算子重叠的载体必须是**共享 SM 内共驻/细粒度 task**，而非分割 block）。

### 1.4 局限（论文 §5）

论文自陈三条：**工程复杂度**（「fully fused, persistent kernels demand deep GPU + distributed-systems
expertise」，并明确期待 **compiler/DSL 抽象**来降低门槛——这正是 MPK 的切入点）；
**FP16 路径未调优**（附录 §H）；**只做推理**（训练需融合反向与梯度通信）。第一条与 MPK §8 行 1862
「手工不泛化」的判断互相印证。

---

## 2. Llama-1B megakernel：把整个前向融进一个 megakernel

> 来源：Hazy Research 博客 *Look Ma, No Bubbles!*（2025-05-27）+ 代码仓 `HazyResearch/Megakernels`。
> 本文无正式章节号，下文以博客小标题定位；所有数字均来自博客原文或仓库 README。

### 2.1 动机：memory pipeline bubbles（博客「Separate Kernels Kill the Vibe」）

- 场景：**batch=1、Llama-3.2-1B（1.24B 参数）**、生成单序列——**强 memory-bound**，性能取决于
  「能否一直在从 global memory 加载权重」。
- 现象：vLLM / SGLang 在这条 workload 上 **H100 带宽利用率最多 ~50%**。根因是**把一次前向拆成约
  **100 个** 独立 kernel**，每个 kernel 都有 setup/teardown，其间权重加载停摆（bubble）。
- 三条 stall（与 `docs/00` §1 三类损失一一对应）：
  1. **kernel 串行边界**——一个 kernel 的 block 要等上一 kernel 全部 block 结束（损失一）；
     例：down projection 跑 512 个 block，而 B200 只有 148 个 SM，末尾 **80 个 SM 空转**。
  2. **launch/teardown 开销**——博客实测：随机 kernel 在 H100 上 **stream launch ≈ 2.1 µs**，
     **CUDA graph 仅降到 ≈ 1.3 µs**（损失二）。这与本专题 P5-1/P4-1 实测「eager 每 kernel
     ~1.7–2.3 µs、graph 摊薄但**不归零**」**同量级**。
  3. **权重/激活加载延迟**——想「边算边载下一个权重」。博客点名 **PDL 的
     `cudaGridDependencySynchronize` 太粗**：「必须等全部 Q/K/V 完成才能开始 attention，而不能
     一有 head ready 就开始」——这正是本专题 P4-2/P5-3 实测的 **PDL 停在 kernel 对、重叠
     上限 = `min(prologue,tail)`** 的定性版本。
- 量化：memory 理论上限 3.35 TB/s ÷ 2.48 GB ≈ **1350 次前向/s**；但 7 launch/layer × 16 layer、
  每次乐观 5 µs stall，就掉到 ~**770 次/s**。

### 2.2 设计：三个问题、三套机制（博客「How to Megakernel」）

**问题 1/3：如何融合约 100 个算子？——设备内解释器（on-GPU interpreter）。**
每个 SM 收到一串 **instruction**，每个 instruction 用**同一个 CUDA 模板**实现（load/store/compute
样板函数）；**调度在 Python 侧预先算好**，且**一张调度可复用数百次前向**。Llama-1B 用到 7 类
instruction：fused RMSNorm&QKV&RoPE、attention、attention reduction（ThunderGQA）、O-proj+residual、
fused RMSNorm&up-gate&SiLU、down-proj+residual、RMSNorm&LM-head。
> 与本专题对照：这是「**per-SM 静态指令序列**」——比 FlashMoE 的**动态** actor 调度更静态，
> 类似本专题 P23-1 §16 的 `ring`（按静态序列取 task + 环形预取）；对应 MPK 则类比 tGraph 的
> **linearization**（`docs/08` §5，把 task 排成连续区间后下发）。

**问题 2/3：如何跨 instruction 共享 SMEM、消除 memory bubble？——SMEM paging。**
把 H100 前 **213 kB shared memory 分成 13 个 16 KiB 页**；instruction 显式**申请/释放**页，
解释器把释放的页立刻转给下一个 instruction，使其**尽早开始发权重加载**，即使前一个 instruction
还在写回结果。这就是 MPK §5.3「**paged shared memory**」（`docs/01` §4、快照行 1312–1325）的
**手工版原形**，也是本专题 P23-1 §16 环形缓冲预取要复刻的机制。

**问题 3/3：单 kernel 内如何同步？——global memory 里的计数器系统。**
kernel 前把全局计数器数组清 0；instruction 完成时**自增**对应计数器，开始前**等待**其依赖计数器
到达目标值。优化例子：Llama 的 MLP 在朴素 PDL 下要等**整个 hidden state** 才能开始 down projection，
megakernel 改成**把中间状态分 4 个 chunk、每 chunk 一个计数器**，down projection 只需等自己那块
——即**逐 chunk 的细粒度依赖**。
> 与本专题对照：这套「计数器 = 依赖阈值」就是 FlashMoE 的 packet signal、DeepGEMM 的 ring
> generation/`l2_full_mask`（`docs/02` §1.5）、MPK 的 `EventDesc.num_triggers`（`docs/06` §2）
> 的同一抽象；本专题 P15-1 §9、P22-1 §15、P23-1 §16 的 `decen`/`ring` 用的也是**逐 task 事件
> 计数器**（并记录了一个真实的坑：自拼「写数据 + 独立 done flag」在大扇入下会读错，必须用计数器）。

### 2.3 结果（博客「Putting It All Together」）

- H100 上**首次**把 16-bit 1B+ 参数模型前向做到 **<1 ms**；B200 进一步 **<680 µs**。
- 相对基线（vLLM/SGLang 均用 CUDA graph + torch compile）：H100 上比 vLLM **~2.5×**、比 SGLang
  **>1.5×**；B200 上比 vLLM **>3.5×**。达 H100 **78% 内存带宽**。
- B200 单次前向 600 µs 的分解：**250 µs** 花在激活的 store→一致性等待→load（每 instruction 两次
  load 延迟 + 两次 store 延迟）；**200 µs** RMSNorm + matvec（其中 95% 是 matvec；Blackwell 上
  Tensor Core 略有益、Hopper 上**纯 CUDA core 更好**）；**30 µs** 等权重（流水生效）；**40 µs**
  跨 warp 低层同步（**异步 barrier 即使已在 pass 态也要 ~60 ns/次**）；**80 µs** setup 等杂项。
> 这个分解对本专题有直接意义：**即使整图融进单 kernel，剩余代价仍集中在「激活的 store/load/
> wait」和「低层同步」**——正好对上 P5-2 §2（损失三是主收益）、P24-1 §17（ncu 直接量出中间张量
> HBM 字节随算子数标度）与 P10-1 §5（硬 barrier ~1.1 µs 地板 → 生产 mega-kernel 改用异步原子
> 环/去中心化 event queue）。博客里「异步 barrier 60 ns/次」与 P10-1 的 `grid.sync()` ~1.1 µs 地板
> 是同一量级问题的两个尺度。

### 2.4 代码与可复现性

代码仓 `HazyResearch/Megakernels`（MIT）README 给出：`git submodule update --init --recursive`
（含 **ThunderKittens** 子模块）→ `uv pip install -e .` → `export GPU=H100|B200` →
`cd demos/low-latency-llama && make` → `python megakernels/scripts/generate.py mode=mk …`。
即它**真开源可复现**，但需要完整模型权重与 H100/B200 编译环境；本机单卡容器不具备条件，
故本文**不代跑**（标注「纯资料分析」）。其指令模板落在 `util/mk_init/sources/src/` 下、以
`PROJECT_NAME_LOWER` 命名的那份 `.cu` 骨架文件里，正是博客里「共用 CUDA 模板」的落点。

---

## 3. 三个「手工 mega-kernel」与 MPK / DeepGEMM 的范式对照

把两个先例与已读透的 DeepGEMM、MPK 放在一起（判据取 `docs/00` §3.2 的 C1–C5）：

| 维度 | **FlashMoE** | **Llama-1B megakernel** | **DeepGEMM mega 家族** | **MPK** |
|---|---|---|---|---|
| 融合范围 | 段级：分布式 MoE 一整个算子（含通信） | **整图**：Llama-1B 前向 ~100 算子 | 段级：MoE / Gate / mHC | **整图**（每 batch size 一张图） |
| C1 单次 launch | ✅（Table 1：1 vs ≤550） | ✅ | ✅ | ✅ |
| C2 常驻 | ✅ persistent | ✅ | ✅ | ✅ |
| C3 多算子 | 段级（MoE 段） | 整图 | 段级 | 整图 |
| C4 设备内协调 | ✅ NVSHMEM 信号 + SMEM/global 信号 | ✅ global 计数器 | ✅ 位掩码 + ring generation | ✅ tGraph event + ring |
| C5 SM/块级调度 | ✅ **动态** actor 调度（work-conserving） | ✅ **静态** per-SM 指令序列 | ✅ 手工静态 warp 角色 | ✅ 去中心化动态 event queue |
| 调度来源 | 手工（actor + tile task） | 手工（Python 预排指令） | 手工（角色写死） | **编译器自动** |
| 硬件/模型泛化 | 需 NVSHMEM、按平台调 tile | **一个模型写死** | SM100 专属 | per-task codegen（唯一硬件相关） |
| 一手材料 | arXiv 2506.04667（NeurIPS '25） | Hazy 博客 2025-05-27 | 本文档 `docs/02` §1 | 论文 arXiv 2512.22219 + `mirage-project/mirage` |

**读法**：

1. **四者都满足 C1–C5**，差别只在 **融合范围**（段级 vs 整图）与 **调度是手工还是自动**。
   这再次确认 `docs/00` §4.2 的分水岭不在「是否单 kernel」，而在 **C4/C5 的调度权归属与粒度**。
2. **FlashMoE 与 DeepGEMM Mega MoE 是同一档（段级 T4）的两种风味**：DeepGEMM 用**静态 warp 角色 +
   位掩码依赖**，FlashMoE 用**动态 actor 调度 + tile task + 非阻塞 NVSHMEM**。后者更接近 MPK 的
   「去中心化动态调度」，前者更接近 Llama megakernel 的「静态指令序列」。于是段级 T4 内部也有一条
   **静态 ↔ 动态**光谱。
3. **Llama-1B megakernel 是「整图 T4 手工对照组」**：它和 MPK 目标相同（整图单 kernel），但把一个
   模型、两种 GPU 写死；MPK §8 行 1861–1862 说它「不泛化」正指此。它同时**独立复现了 MPK 的三大
   机制**（Paged SMEM、跨算子软件流水、细粒度 event 依赖），是对 MPK 设计选择最有力的旁证。
4. **两篇一手材料的自陈局限，恰好是 MPK 的卖点**：FlashMoE §5「期待 compiler/DSL 抽象」、
   Llama megakernel 写死模型 —— 与 MPK Abstract/§1「首个自动编译整图 mega-kernel」的定位严丝合缝。

---

## 4. 对本专题既有结论的印证

- **损失一/二/三的普适性**：两个先例各自用**独立的方法与硬件**重现了同一组瓶颈。Llama megakernel
  博客用 H100 实测的 **stream 2.1 µs / graph 1.3 µs** 与本专题 P5-1/P4-1 在**目标卡 H100** 上量到的
  「eager ~1.7–2.3 µs、graph 摊薄不归零」同量级；博客对 **PDL 太粗**的批评 = 本专题 P5-3 §3
  「PDL 收益不随 K 累积、上限 `min(prologue,tail)`」的定性版。
- **机制可复刻性**：FlashMoE 的 actor 调度 ≈ P15-1 §9 的 C5 最小骨架；两篇的计数器依赖 ≈
  P15-1/P22-1/P23-1 的逐 task 事件计数器；Llama 的 paging ≈ P23-1 §16 的环形缓冲预取 +
  MPK §5.3 的 paged SMEM。说明本专题那些「合成的」实验**确实抓到了生产/一线实现的核心机制**。
- **成本侧仍成立**：Llama megakernel 的 600 µs 分解里，激活 store/load/wait（250 µs）+ 低层同步
  （40 µs）占近半——即使整图融合，**中间张量往返与同步开销仍是主要残项**，与 P24-1 §17（DRAM 字节
  随 Q 标度）、P14-1/P16-1（寄存器/代码**双层不可分页足迹**）的成本画像一致。
- **「段级先落地、整图靠自动」得到一手佐证**：DeepGEMM（生产段级）+ FlashMoE（学术段级，1 launch、
  93% SM）说明段级 T4 已成熟；Llama megakernel（整图但写死）与 MPK（整图自动）说明整图 T4 仍靠
  「要么写死、要么编译器」。这补强了 `docs/02` §3.4 的「需求(K3)→路线(hpc-ops)→落地(DeepGEMM 段级)
  →自动化(MPK 整图)」四段式。

## 5. 小结（P27-1）

本轮把 MPK 反复转引的**两个手工 mega-kernel 先例**回到一手资料并核对：

- **FlashMoE**（arXiv 2506.04667，NeurIPS '25）：首个把**分布式 MoE 一整个算子段**融进单常驻 kernel
  （§1 Table 1：1 launch vs 最多 550），用 **actor 模型**（Processor/Subscriber/Scheduler，§3）+
  **tile 级统一 task**（§3.1，每 task 一个 device 函数）+ **对称张量布局的非阻塞 NVSHMEM 通信**
  （§3.2，Theorem 3.1 write-write conflict-free，in-place padding），把 SM 利用率从 DeepEP 的 14%
  提到 **93.17%**（§4.2）、8 卡延迟最高降 **6.4×**（§4.1，且 FP32 对 FP16）。
- **Llama-1B megakernel**（Hazy Research 2025-05-27，代码 `HazyResearch/Megakernels`）：把**整个
  Llama-1B 前向（~100 算子）**融进单 kernel，H100 **<1 ms**、B200 **<680 µs**，比 vLLM ~2.5×，
  靠三套机制——**设备内解释器 + 每 SM 指令序列**、**SMEM 分页（13×16 KiB）**、**global 计数器
  细粒度依赖**——独立复现了 MPK 的核心设计。
- **范式结论**：四者（+DeepGEMM、MPK）都满足 C1–C5，分野在**融合范围（段级/整图）**与
  **调度手工/自动**；段级 T4 内部还有**静态（DeepGEMM/Llama 指令）↔ 动态（FlashMoE actor）**
  光谱。两篇先例的自陈局限（泛化、工程复杂度）正是 MPK 编译器路线的立足点。

**上述均为一手资料分析，本轮无新实机实验；相关机制的实测证据见 `docs/04` §1/§9/§14/§16/§17。**

## 6. 关键坐标（可核对）

- MPK 快照（转引处）：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`
  §2.2 行 325–344、§8 行 1851–1862；参考文献 [4] 行 1958–1965、[11] 行 2041–2049、[33] TileRT 行 2321–2325。
- FlashMoE：论文 `https://arxiv.org/abs/2506.04667`（v3 HTML `https://arxiv.org/html/2506.04667v3`），
  代码 `https://github.com/osayamenja/FlashMoE`。
- Llama-1B megakernel：博客 `https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles`，
  代码 `https://github.com/HazyResearch/Megakernels`（含 `ThunderKittens` 子模块、
  `demos/low-latency-llama`）。
- 相关对照：TileRT `https://github.com/tile-ai/TileRT`（MPK [33]）；ThunderKittens [30]、
  PipeThreader [8]（MPK §8 提到的抽象层互补工作）。
