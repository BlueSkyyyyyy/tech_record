# MoE 与专家并行（EP）基础

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 1 篇。先把 MoE 的计算图与专家并行
> （Expert Parallel）通信语义讲清楚，后面精读 DeepGEMM / DeepEP 的融合实现时才有坐标系。

## 1. MoE 的计算图

一个标准 MoE 层（如 DeepSeek 系列）对每个 token $x\in\mathbb{R}^{H}$ 做：

$$
\text{router:}\quad s = \mathrm{softmax}(x W_g)\in\mathbb{R}^{E},\qquad
\text{topk:}\quad \mathcal{T}(x)=\operatorname{topk}_k(s),
$$

$$
y = \sum_{i\in\mathcal{T}(x)} \tilde s_i \cdot \mathrm{FFN}_i(x),\qquad
\mathrm{FFN}_i(x)=W^{down}_i\,\big(\mathrm{SwiGLU}(W^{up}_i x)\big),
$$

其中 $E$ 是专家数，每个 token 选 $k$ 个专家（DeepSeek-V3 为 $k=8$，另有共享专家）。
$H$ 是 hidden，专家中间维 $I$（含 SwiGLU 门与值两条线性层，故 up 矩阵宽 $2I$）。

**两段线性**（不含共享专家）：
- **linear1**：`[tokens, H] @ [H, 2I]` → `[tokens, 2I]` → SwiGLU → `[tokens, I]`；
- **linear2**：`[tokens, I] @ [I, H]` → `[tokens, H]`。

因为不同专家处理的 token 数不等，工程上把「每个专家一段 token」拼成一个大矩阵做 **grouped GEMM**：
**只分组 M 轴，N/K 固定**（各专家形状相同）。DeepGEMM 提供两种布局：

- **contiguous**：各专家段连续拼接，段首对齐到块大小 `get_mk_alignment_for_contiguous_layout()`；
  用于训练前向与推理 prefill（token 数已知）。
- **masked**：给一个 mask 张量，kernel 只算有效部分；用于解码阶段 CUDA graph 下 CPU 不知道各专家 token 数
  的情形（可用 DeepEP 低延迟 kernel 的输出）。

## 2. 专家并行（EP）与 all-to-all

当专家分布在多个 rank（GPU）上时，token 需要按路由结果**跨 rank 搬运**：

- **dispatch（all-to-all）**：每个 rank 把自己本地 token 中需要送到专家 $i$（在 rank $r_i$）的部分，
  发到对应 rank；接收别的 rank 发来的 token，得到「本 rank 专家要处理的 token」。
- **combine（all-to-all 反向）**：专家算完后，把结果按原路送回 token 所属 rank，再按 topk 权重
  $\tilde s_i$ 加权求和得到最终 $y$。

核心矛盾：**通信（NVLink all-to-all）与计算（Tensor Core）互相等待**。
朴素实现是「dispatch → GEMM → combine」串行，NVLink 与 SM 轮流空闲。

## 3. 为什么要「Mega MoE」

DeepSeek 的 **Mega MoE** 把这五段——
**EP dispatch → linear1 → SwiGLU → linear2 → EP combine**——融合进**一个 mega-kernel**，
让 **NVLink 通信与 Tensor Core 计算在时间上重叠**：

- 用 **symmetric memory**（对称显存，各 rank 地址对齐）做通信缓冲区；
- dispatch 来的 token 进 **ring buffer / pool blocks**，由 `scheduler/mega_moe.cuh` 动态分配；
- **dispatch warp** 持续拉取/推送 NVLink 数据，**MMA/epilogue warp** 持续吃 buffer 里的 tile 做 GEMM，
  两者用 mbarrier 同步；通信延迟被计算掩盖。

这也是本专题的主线：先讲清这五段各自是什么（本文 + 后续），再逐段精读 `sm100_fp8_fp4_mega_moe.cuh`。

> 注：Mega MoE 主实现是 **SM100（Blackwell）专属**（用 tcgen05 MMA、UE8M0 scale、FP8xFP4）。
> 本机 H100（sm90）跑不了端到端，相关部分以源码分析为主。

## 4. MoE 的反向

训练时需要三个方向的梯度：

- **dgrad（对输入 $x$ 的梯度）**：把 `dFFN` 通过 linear2、SwiGLU、linear1 传回去，本质是
  **对「按专家分组」的输入做 GEMM**，即 `m_grouped_*`（分组 M 轴）。
- **wgrad（对权重的梯度）**：$\mathrm{d}W_i = \text{acts}_i^\top \mathrm{d}\text{out}_i$，
  各专家形状相同（M/N 固定）、只有 K（token 数）随专家变化 → DeepGEMM 的
  **`k_grouped_fp8_gemm_tn_contiguous`**（K 轴分组）。README 明确说这是「MoE weight backward」。
- **router 反向**：softmax/topk 的梯度（topk 是硬选择，只有被选中的专家有梯度）。

反向还要再走一遍 EP 通信（dgrad 前需要把上游梯度 dispatch 到专家所在 rank；wgrad 后需要 combine）。

## 5. 线上资料与版本坐标（路线图 P0-2）

本专题参考的公开实现与发布说明。**所有链接均来自 GitHub 仓库的真实页面/PR**（核对于 2026-10-12）。

### 5.1 仓库

- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM （本地 `~/github/DeepGEMM`，本文核对 commit `057ca59`）
- DeepEP（EP 通信 / dispatch / combine）：https://github.com/deepseek-ai/DeepEP （本地 `~/github/DeepEP`）
- DeepJIT（DeepGEMM / DeepEP 运行期 JIT 运行时）：https://github.com/deepseek-ai/DeepJIT

### 5.2 版本时间线（DeepGEMM README「News」+ PR）

| 日期 | 内容 | 链接 |
|---|---|---|
| 2025-05-14 | 首个 **weight gradient** kernel（dense + MoE backward） | [PR #95](https://github.com/deepseek-ai/DeepGEMM/pull/95) |
| 2025-07-20 | 全量重构，SM90/SM100 双架构 + 低 CPU 开销 JIT | [PR #112](https://github.com/deepseek-ai/DeepGEMM/pull/112) |
| 2026-04-17 | **引入 Mega MoE**、FP4 Indexer、**FP8xFP4 GEMM**、PDL | [PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304) |
| 2026-04-24 | Mega MoE 优化 + **基准数据** | [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316) |
| 2026-09-10 | **Mega Gate / Mega mHC**、Sparse Indexer、MoE 反向提速 | [PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432) |
| 2026-09-30 | **locality-aware Mega MoE**、epilogue 类、README 公开版 | [PR #462](https://github.com/deepseek-ai/DeepGEMM/pull/462) |

### 5.3 关键结论（来自 PR 正文，直接支撑后续精读）

- **Mega MoE 只支持 FP8 × FP4**，且要求 PyTorch ≥ 2.9（PR #304）。这解释了 kernel 名
  `sm100_fp8_fp4_mega_moe` 与线性层激活量化（L1 epilogue 里 amax→UE8M0 那段）的必要性。
- **融合范围**：dispatch + linear1 + SwiGLU + linear2 + combine 五段进单 kernel，重叠 NVLink 与
  张量核（PR #304）。这正是 `docs/02` 的架构主线。
- **Mega Gate**：把 routing GEMM + 多个 MoE routing 操作融合，提速 **30–75%**；
  **Mega mHC**：融合多个 Hyper-Connection 操作 + RMSNorm，提速 **45–85%**（PR #432）。
  这两个是本专题 `docs/05` 的对象。
- **MoE 反向**：PR #432 称反向 GEMM 提速约 **20%**；而最早的 wgrad kernel（PR #95）即比内部
  CUTLASS 实现快约 20%。对应 API 是 `wgrad_gemm_fp8_fp8_fp32_nt`（dense）与
  `k_grouped_wgrad_gemm_fp8_fp8_fp32_nt`（MoE，K 轴分组），即路线图 P4-2 的
  `k_grouped_fp8_gemm_tn_contiguous` 前身。
- **正确性/竞态**：注意 [PR #343](https://github.com/deepseek-ai/DeepGEMM/pull/343)（SM90 contiguous
  k-grouped GEMM 在切组时**就地改 tensormap** 与在途 TMA load 竞态，会算错）、
  [PR #441](https://github.com/deepseek-ai/DeepGEMM/pull/441)（Mega MoE task-info slot 释放顺序）。
  这两条是精读反向/调度器时不能忽略的「为什么这么写」。
- **Locality-aware**（PR #462）：可选的**权重本地化 + domain-aware 调度**，通过 MLOPart 做本地化分配、
  发现 SM locality，官方称重载下省约 100W、GPU 频率提升 5–10%。对应 `docs/02` §4.4 的
  `locality domain` 与 kernel 里的 `sm_locality_domain_idx`。

### 5.4 DeepEP 侧

DeepEP README（核对于本地 `~/github/DeepEP`，V2.5）说明：提供高吞吐 / 低延迟的 **EP all-to-all**
（dispatch / combine，含 FP8 dispatch），编译期走 DeepJIT，**不再依赖 NVSHMEM**（V2 起改用轻量
NCCL Gin backend）。与 Mega MoE 相关的两条：

- EPv2 把高吞吐与低延迟 API 统一到单个 `EPBuffer`，并给出**适配 grouped expert GEMM 的 expanded layout**；
- `EPBuffer.lb_prefetch_weights` / `lb_reduce_grads` 支持**冗余专家**：前向经 NVLink 预取专家权重与量化
  scale，反向把冗余专家的 FP32 梯度累加回原专家（对应 MoonEP / UltraEP 思路）。

> 说明：DeepEP 的 V2/V2.5 目标平台是 Hopper 及以上，但本专题把 dispatch/combine 与 symmetric
> memory 的接口/数据流作为 P0-3 单独分析，与 SM100 专属的 Mega MoE 主体分开。

## 6. 下一步

- `docs/02-mega-moe-architecture.md`：Mega MoE 整体架构与通信-计算重叠机制（已含线上基准坐标）。
- `docs/03-mega-moe-kernel.md`：`sm100_fp8_fp4_mega_moe.cuh` 逐段精读。
- `docs/04-moe-backward.md`：wgrad/dgrad 的 grouped GEMM 实现与数值实验。
- `docs/05-mega-gate-mhc.md`：Mega Gate / Mega mHC。
