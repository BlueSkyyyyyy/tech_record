# MoE 与专家并行（EP）基础

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 1 篇。我们先把 MoE 的计算图和专家并行
> （Expert Parallel）的通信语义讲清楚，后面精读 DeepGEMM / DeepEP 的融合实现时才有坐标系。

## 0. 先记住一个画面

**MoE 层像一家综合医院的分诊台。**

每个病人（token）进门先到分诊台（router）打分，然后被分到 8 个科室（top-k 的 8 个专家）。
每个科室自己有一套流程（FFN：先升维、过激活、再降维），最后把 8 个科室的诊断意见按分诊台给的
权重加权合并，才是这个病人的最终结果 $y$。

这个画面能解释 MoE 里几乎所有的工程问题：

| 医院里的现象 | MoE 里对应的东西 | 本文 |
|---|---|---|
| 各科室病人数不一样，热门科室排长队 | 专家负载不均，需要 grouped GEMM 按专家分组算 | §1 |
| 病人要去不同院区（GPU）的科室 | 专家并行 EP，token 要跨卡搬 | §2 |
| 每个病人挂 8 个科室，病历要复印 8 份 | 通信量随 top-k 线性放大 | §2.1 |
| 送病人和等诊断互相干等 | 通信与计算串行，NVLink 与 SM 轮流空闲 | §2.2 |

> **类比的边界**：医院里病人只有一个，而 MoE 里一个 token 会被**复制 k 份**分别送进 k 个专家，
> 再把 k 份输出加权合并。这个「复制」是后面通信量放大的根源，医院类比里没有。

读完这一篇，你应该能回答三个问题：

1. 一个 MoE 层的计算图长什么样？为什么工程上要落成 grouped GEMM？
2. contiguous / masked / PSUM 三种布局分别解决什么问题？代价是什么？
3. 为什么说 EP 的通信量是「k 倍放大」的？为什么 Mega MoE 非要把通信和计算塞进一个 kernel？

## 1. MoE 的计算图与 grouped GEMM 布局

### 1.1 一个 token 的路径，以及为什么变成 grouped GEMM

一个标准 MoE 层（如 DeepSeek 系列）对每个 token $x\in\mathbb{R}^{H}$ 做两件事：

$$
\text{router:}\quad s = \mathrm{softmax}(x W_g)\in\mathbb{R}^{E},\qquad
\text{topk:}\quad \mathcal{T}(x)=\operatorname{topk}_k(s),
$$

$$
y = \sum_{i\in\mathcal{T}(x)} \tilde s_i \cdot \mathrm{FFN}_i(x),\qquad
\mathrm{FFN}_i(x)=W^{down}_i\,\big(\mathrm{SwiGLU}(W^{up}_i x)\big),
$$

其中 $E$ 是专家数，每个 token 选 $k$ 个专家（DeepSeek-V3 为 $k=8$，此外还有共享专家）。
$H$ 是 hidden 维，专家中间维是 $I$（SwiGLU 有门和值两条线性层，所以 up 矩阵宽 $2I$）。

拆开看，就是**两段线性**（不含共享专家）：

- **linear1**：`[tokens, H] @ [H, 2I]` → `[tokens, 2I]` → SwiGLU → `[tokens, I]`；
- **linear2**：`[tokens, I] @ [I, H]` → `[tokens, H]`。

**为什么工程上一定要做成 grouped GEMM？** 问题出在「各科室病人数不一样」。

不同专家分到的 token 数不等，如果每个专家单独发一次 GEMM，就是 8 次形状不一的小 GEMM，
每次都填不满 GPU，launch 开销还翻倍。

工程上的做法是：把「每个专家分到的那一段 token」**首尾相接**拼成一个大矩阵，一次算完。
这就是 **grouped GEMM**——**只分组 M 轴，N/K 固定**（因为各专家形状相同，
DeepSeek-V3 的所有专家都是 `[7168, 2*2048]`）。

> 换成医院的说法：不是给每个科室单独派一辆车，而是把所有病人按科室排成一队，
> 车队一次出发，只是每到一站换一批人。

DeepGEMM 的 grouped API 命名规则很规整：

- `m_grouped_*`：分**组 M 轴**，用于前向与 dgrad；
- `k_grouped_*`：分**组 K 轴**，用于 wgrad。

分 G 组时输入是 `[G, N, K]` 的权重栈（`csrc/apis/gemm.hpp:181` 顶部注释 `[M, K] @ [G, N, K].mT`）。
注意新版 API 把数据与量化 scale factor 打包成 tuple 传入，例如 `a=(a_data, a_sf)`、`b=(b_data, b_sf)`。

接下来的三种布局，本质上是在回答同一个问题：**「分段信息」从哪来、以什么形式交给 kernel？**

### 1.2 布局一：contiguous（连续拼接，host 已知边界）

**语义**：把 G 个专家的 token 段**首尾相接**成一个 `[sum_m, K]` 的 A。
`grouped_layout` 是长度 `sum_m` 的 int 张量，`grouped_layout[row] = 专家号`（无效填充行记 `-1`）。

**硬约束**：**每段的起始行必须对齐到 `get_mk_alignment_for_contiguous_layout()`**
（README §Grouped GEMMs (contiguous layout)，`README.md:76-78`）。

这个对齐值在 SM90 是历史值 **128**，在 SM100 上理论值提高到 **256**——因为 SM100 新增了
`UMMA_N=256`，固定 256 对齐同时对 MoE cast / M-grouped / K-grouped 都更友好
（`csrc/jit_kernels/heuristics/runtime.hpp:57-64`）。这就是「段与段之间会有 gap 行」的来源。

**为什么非要对齐**：kernel 按 `BLOCK_M` 切 tile，A 的 scale factor（SFA）在转换/打包时也是按块布局的。
如果段起点任意，一个 `BLOCK_M` tile 就可能跨两个专家——一个 tile 只能用一份权重，跨了就没办法了。
对齐到块大小，就保证**每个 M tile 只属于一个专家**，于是整个 tile 能用同一份 `[N, K]` 权重，
kernel 里不需要 per-row 的专家分支（`csrc/apis/gemm.hpp:230-246`；SM90 走
`sm90_m_grouped_fp8_gemm_contiguous_1d2d`，SM100 走 `sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d`）。

**什么时候用**：训练前向与推理 prefill。此时 token 数已知，host 端能精确算出各段并对齐。

### 1.3 布局二：masked（掩码，device 已知有效行数）

**语义**：A/D 是稠密的 `[G, max_m, K]` / `[G, max_m, N]`，另外给一个 `masked_m`
（长度 G 的 int，表示每组有效 M），kernel 对每组只算前 `masked_m[i]` 行
（`csrc/apis/gemm.hpp:269-318`）。

这里有个容易踩的坑：`masked_m` 与 contiguous 用的是**同一个 `grouped_layout` 变量名**，
但含义完全不同——masked 下它是「每组有效行数」，contiguous 下它是「每行的组号」。
两者的 API 入口也不同（`m_grouped_fp8_gemm_nt_masked` vs `..._contiguous`）。

**为什么要 masked**：解码阶段开 **CUDA graph** 时，CPU 在 launch 之前根本不知道每个专家会收到多少
token（token 数要等 GPU 上的 DeepEP dispatch 算出来）。既然没法提前做对齐拼接，那就固定形状：
`[G, max_m, K]`（对图捕获友好），把「有效行数」作为 GPU 张量传进去，kernel 在 device 端读。
典型输入就是 DeepEP 低延迟 dispatch 的输出（README §Grouped GEMMs (masked layout)，`README.md:82-86`）。

### 1.4 布局三：PSUM（前缀和，动态段大小也能连续存）

contiguous 要求每段对齐、`grouped_layout` 是每行的组号。但很多场景（比如 DeepEP 的输出）
只知道**每组的结束位置**，而且组大小参差不齐。

PSUM 布局换个编码：`grouped_layout[i]` 直接就是**第 i 组有效 token 的前缀和结束行号**。
组的起点 = `align(上一组的结束, alignment)`，组间的 gap 行由 kernel 跳过
（`docs/scaling-factor-format.md:203-215`）。

它的价值在于「把 masked 的动态信息重新塞回 contiguous 的单张量表示」，于是推理侧原本只能走 masked
的场景，也能走**更高效的 contiguous kernel**——`tests/test_fp8_fp4.py:203-205` 里
`use_psum_layout=True` 时，正是用 `m_grouped_fp8_fp4_gemm_nt_contiguous(..., use_psum_layout=True,
expected_m_for_psum_layout=...)` 替代 masked 入口。

映射到 Mega MoE：dispatch 产出的正是这种「每专家结束位置 + 对齐拼接」的形态
（`docs/02` §3 的 pooled buffer）。

> **三种布局其实是三层抽象**：**contiguous**（host 已知，行级组号）→ **masked**（device 已知，
> 组级计数）→ **PSUM**（device 已知，组级前缀和 + 连续存储）。信息越往后越少假设、越多动态。

### 1.5 H100 实测：三种布局跑出来差多少

`harness/exp_grouped_layout.py`（输出 `harness/exp_grouped_layout.out.txt`）在 H100 上用
DeepGEMM 自带 generators 造数据、与 PyTorch 参考对拍。取 G=6、n=256、k=512，FP8（SM90 1D2D）：

| 布局 | 有效 M（各组） | 物理存储 | relerr |
|---|---|---|---|
| contiguous | 91/117/123/89/81/73 | `[768, 512]`，其中 **194/768 是 gap 行**（对齐到 128） | 3.69e-2 |
| masked | 93/86/119/84/106/76 | `[6, 160, 512]` 稠密，`max_m=160` | 3.69e-2 |
| PSUM | 前缀和 `[81,240,334,462,582,730]` | `[768, 512]`，组起点 0/128/256/384/512/640 | 3.69e-2 |

三个数字读出来的结论：

1. **三种布局残差完全一致**（都是 FP8 per-128 块量化的约 3.7e-2）。说明它们只是「同一个计算的
   不同边界表示」，数值上等价——选哪个是工程问题，不是数学问题。
2. **contiguous 的代价是 gap 行**：194/768 ≈ 25% 的存储是空的。
3. **masked 的代价是按 `max_m` 放大**：160×6=960 行，实际只用了 559 行。

一句话：这两种布局都是**用空间/对齐，换 kernel 的简单与图捕获友好**。

## 2. 专家并行（EP）与 all-to-all

当专家分布在多个 rank（GPU）上时，token 必须按路由结果**跨 rank 搬运**：

- **dispatch（all-to-all）**：每个 rank 把自己本地 token 中、需要送到专家 $i$（位于 rank $r_i$）
  的那部分，发到对应 rank；同时接收别的 rank 发来的 token，得到「本 rank 专家要处理的 token」。
- **combine（all-to-all 反向）**：专家算完后，把结果按原路送回 token 所属的 rank，
  再按 topk 权重 $\tilde s_i$ 加权求和，得到最终 $y$。

核心矛盾只有一句：**通信（NVLink all-to-all）与计算（Tensor Core）互相等待。**
朴素实现是「dispatch → GEMM → combine」串行，于是 NVLink 和 SM 轮流空闲。

### 2.1 算一笔账：为什么说通信是「k 倍放大」

设每个 rank 本地 $T$ 个 token、hidden $H$、每个 token 选 $k$ 个专家、激活按 FP8（1 B）、
专家输出按 BF16（2 B）。关键在**一个 token 会被复制多份**。

**dispatch 方向**：token 的 $k$ 个专家可能落在 $d$ 个不同 rank 上（$d\le k$）。
DeepEP 只在**目标 rank 粒度去重**——若一个 token 的多个 topk 落在同一 rank，线上只传一份
（`dispatch.cuh:103`-`:105` 的 `ptx::deduplicate`），落地后再由 copy epilogue 展开成「每专家一个槽」。
于是每个 rank 的发送量约为

$$B_{\text{disp}} \;\approx\; T\cdot d\cdot H\cdot 1\text{ B}.$$

**combine 方向**：专家输出是「每 (token, topk) 一行」，$k$ 份都要按原路送回源 rank 再做加权归约，
这里几乎没得去重，故

$$B_{\text{comb}} \;\approx\; T\cdot k\cdot H\cdot 2\text{ B}.$$

注意这里 **combine 的系数是 $k$ 而不是 $d$**：dispatch 传的是「token 本身」，一个 token 去同一个
rank 的多个专家还能合并成一份；combine 传的却是「$k$ 个专家各自的输出」，天然就是 $\sim k$ 倍放大
（复合 topk 权重 $\tilde s_i$ 也在此时乘上，见 `docs/07` §5 的 bf16 加权归约）。

**这条模型为什么重要**：MoE 的激活/输出通信量随 $k$ **线性增长**，而计算量只随「实际被选中的
专家数」线性增长（每个 token 只进 $k$ 个 FFN）。$k=8$ 时，通信的「数据放大」非常可观，
一旦串行，NVLink 立刻成为瓶颈。

拿 DeepSeek-V3 的量级算一遍（$H=7168$、$I=2048$、$k=8$、EP=8，$d\approx k$，每 rank $T=4096$）：

| 项 | 字节数（每 rank） | 说明 |
|---|---|---|
| dispatch | $\approx 4096\times8\times7168\times1\text{B}\approx 235$ MB | FP8 激活，$d\approx k=8$ |
| combine | $\approx 4096\times8\times7168\times2\text{B}\approx 470$ MB | BF16 输出 |
| **通信合计** | $\approx 705$ MB | 单向出流量 |

同一层的 Tensor Core 计算量（linear1 $[T,H]@[H,2I]$ + linear2 $[T,I]@[I,H]$）约
$T\cdot H\cdot 2\cdot(2I+I)\approx 4096\times7168\times2\times6144\approx3.6\times10^{11}$ FLOP。
在 H100 FP8 约 $0.99$ PFLOP/s 下，单层约 **0.36 ms**；而通信 $705$ MB 按 §2.3 实测的
$\approx390$ GB/s 有效带宽，约 **1.8 ms**。

**结论：通信比计算还贵约 5 倍。**

### 2.2 所以必须让通信与计算重叠

- **串行**（dispatch → GEMM → combine）：每层约 $0.36+1.8\approx2.2$ ms。NVLink 传完等 SM，
  SM 算完等 NVLink。
- **重叠**（本专题主线的 Mega MoE，或 DeepEP 的 `async_with_compute_stream` + PDL，见 `docs/07` §6）：
  通信与 GEMM 并行，理想下界降到 $\max(\text{comm},\text{compute})\approx1.8$ ms 加一点固定时延，
  也就是省掉「较小那一侧」的全部时间。上例中省下约 $0.36$ ms/层（$\sim16\%$）；
  当计算量更大（token 更多），或者 FP4 让计算更便宜、通信更贵时，收益更大。

这正是 DeepSeek 要把这五段塞进**一个 mega-kernel**、并用 ring buffer 解耦通信与计算的理由。

> 上面是**通信量的粗估模型**，不是实测内核数字：真实值取决于专家放置（$d$）、是否 FP8 combine、
> 冗余专家、是否 scale-out（RDMA）等。下一节给出 H100 上 all-to-all 原语的真实带宽/时延坐标。

### 2.3 H100 实测：all-to-all 原语的时延与有效带宽

Mega MoE 主体是 SM100 专属（本机跑不了），但 EP all-to-all 这个**通信原语**在 H100 上是可测的。

`harness/exp_ep_comm.py`（输出 `harness/exp_ep_comm.out.txt`）用本机 8×H100 + NCCL
`all_to_all_single`（等价于 dispatch/combine 的「每 rank 把所有本地 token 重新分配给所有 rank」），
在 DeepSeek-V3 的 $H=7168$、EP=8 下测得：

| 数据类型 | 本地 token | 每 rank 发送量 | 时延 | 有效带宽 |
|---|---|---|---|---|
| FP8 | 4 096 | 29.4 MB | 121.9 us | 240.9 GB/s |
| FP8 | 16 384 | 117.4 MB | 354.1 us | 331.6 GB/s |
| FP8 | 65 536 | 469.8 MB | 1 291.7 us | 363.7 GB/s |
| FP8 | 262 144 | 1 879.0 MB | 4 840.4 us | 388.2 GB/s |
| BF16 | 4 096 | 58.7 MB | 193.8 us | 303.0 GB/s |
| BF16 | 262 144 | 3 758.1 MB | 9 582.7 us | 392.2 GB/s |

两点观察，都和「通信-计算重叠」直接相关。

**第一，渐近带宽约 390–395 GB/s。** 对「时延-消息量」做最小二乘拟合：FP8 约 393 GB/s、
BF16 约 395 GB/s。同机 `nvidia-smi nvlink -s` 报每卡 18 条 link、每条 26.562 GB/s，
单向聚合硬件上限约 **478 GB/s**——实测达到约 **82%**。缺口来自 all-to-all 的多对多协议开销、
每跳同步与通信域建立。

**第二，小消息有明显固定时延**：外推到 0 消息约 50–90 us（视拟合区间而定）。
这就解释了为什么**解码小 batch 时会「带宽接近 0、但耗时几十 us」**（`docs/02` §8 的现象）：
一次 all-to-all 的握手/同步成本与数据量几乎无关。此时唯一的出路是**与它重叠**，
而不是想办法把消息做大。

> 说明：DeepEP 的真实 dispatch/combine 会把上面的量再乘上 §2.1 的放大系数 $d$（dispatch）
> 与 $k$（combine），并叠加「全局计数握手」（`docs/07` §4.1）等延迟。本表量的是**原语裸开销**，
> 属于 H100 可跑的近似坐标，不是 SM100 Mega MoE 的实测。

## 3. 为什么要「Mega MoE」

DeepSeek 的 **Mega MoE** 把这五段——

**EP dispatch → linear1 → SwiGLU → linear2 → EP combine**——

融合进**一个 mega-kernel**，让 **NVLink 通信与 Tensor Core 计算在时间上重叠**。三个关键设计：

- 用 **symmetric memory**（对称显存，各 rank 地址对齐）做通信缓冲区；
- dispatch 来的 token 进 **ring buffer / pool blocks**，由 `scheduler/mega_moe.cuh` 动态分配；
- **dispatch warp** 持续拉取/推送 NVLink 数据，**MMA/epilogue warp** 持续吃 buffer 里的 tile 做 GEMM，
  两者用 mbarrier 同步——通信延迟被计算掩盖。

这也是本专题的主线：先讲清这五段各自是什么（本文 + 后续），再逐段精读 `sm100_fp8_fp4_mega_moe.cuh`。

> 注：Mega MoE 主实现是 **SM100（Blackwell）专属**（用 tcgen05 MMA、UE8M0 scale、FP8xFP4）。
> 本机 H100（sm90）跑不了端到端，相关部分以源码分析为主。

## 4. MoE 的反向：三个方向，两套分组

训练时需要三个方向的梯度。

- **dgrad（对输入 $x$ 的梯度）**：把 `dFFN` 通过 linear2、SwiGLU、linear1 传回去。
  本质是**对「按专家分组」的输入做 GEMM**，即 `m_grouped_*`（分组 M 轴）。
- **wgrad（对权重的梯度）**：$\mathrm{d}W_i = \text{acts}_i^\top \mathrm{d}\text{out}_i$。
  各专家形状相同（M/N 固定）、只有 K（token 数）随专家变化 → 用 DeepGEMM 的
  **`k_grouped_fp8_gemm_tn_contiguous`**（K 轴分组）。README 明确说这就是「MoE weight backward」。
- **router 反向**：softmax/topk 的梯度。topk 是硬选择，所以只有被选中的专家有梯度。

> 可以这样记：**dgrad 分组 M（每个专家一段 token），wgrad 分组 K（每个专家一段 token 数），
> 因为它们把 token 数放在了不同的矩阵维度上。**

反向还要再走一遍 EP 通信：dgrad 前要把上游梯度 dispatch 到专家所在的 rank，wgrad 后需要 combine。

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
  `sm100_fp8_fp4_mega_moe`，以及线性层激活量化（L1 epilogue 里 amax→UE8M0 那段）的必要性。
- **融合范围**：dispatch + linear1 + SwiGLU + linear2 + combine 五段进单 kernel，重叠 NVLink 与
  张量核（PR #304）。这正是 `docs/02` 的架构主线。
- **Mega Gate**：把 routing GEMM + 多个 MoE routing 操作融合，提速 **30–75%**；
  **Mega mHC**：融合多个 Hyper-Connection 操作 + RMSNorm，提速 **45–85%**（PR #432）。
  这两个是本专题 `docs/05` 的对象。
- **MoE 反向**：PR #432 称反向 GEMM 提速约 **20%**；而最早的 wgrad kernel（PR #95）就已比内部
  CUTLASS 实现快约 20%。对应 API 是 `wgrad_gemm_fp8_fp8_fp32_nt`（dense）与
  `k_grouped_wgrad_gemm_fp8_fp8_fp32_nt`（MoE，K 轴分组），即路线图 P4-2 的
  `k_grouped_fp8_gemm_tn_contiguous` 前身。
- **正确性/竞态**：有两个 PR 值得记住——[PR #343](https://github.com/deepseek-ai/DeepGEMM/pull/343)
  （SM90 contiguous k-grouped GEMM 在切组时**就地改 tensormap**，会与在途 TMA load 竞态导致算错）、
  [PR #441](https://github.com/deepseek-ai/DeepGEMM/pull/441)（Mega MoE task-info slot 释放顺序）。
  这两条是精读反向/调度器时不能忽略的「为什么这么写」。
- **Locality-aware**（PR #462）：可选的**权重本地化 + domain-aware 调度**，通过 MLOPart 做本地化分配、
  发现 SM locality，官方称重载下省约 100W、GPU 频率提升 5–10%。对应 `docs/02` §4.4 的
  `locality domain` 与 kernel 里的 `sm_locality_domain_idx`。

### 5.4 DeepEP 侧

DeepEP README（核对于本地 `~/github/DeepEP`，V2.5）说明：它提供高吞吐 / 低延迟的 **EP all-to-all**
（dispatch / combine，含 FP8 dispatch），编译期走 DeepJIT，**不再依赖 NVSHMEM**
（V2 起改用轻量 NCCL Gin backend）。与 Mega MoE 相关的有两条：

- EPv2 把高吞吐与低延迟 API 统一到单个 `EPBuffer`，并给出**适配 grouped expert GEMM 的 expanded layout**；
- `EPBuffer.lb_prefetch_weights` / `lb_reduce_grads` 支持**冗余专家**：前向经 NVLink 预取专家权重与
  量化 scale，反向把冗余专家的 FP32 梯度累加回原专家（对应 MoonEP / UltraEP 思路）。

> 说明：DeepEP 的 V2/V2.5 目标平台是 Hopper 及以上，但本专题把 dispatch/combine 与 symmetric
> memory 的接口/数据流作为 P0-3 单独分析，与 SM100 专属的 Mega MoE 主体分开。
> 这部分源码精读见 `docs/07-deepep-dispatch-combine.md`：对称显存（NCCL window）、
> dispatch 的 notify/搬运 warp、dispatch copy epilogue 的 expanded layout、combine 的 bf16 加权归约。

## 6. 下一步

- `docs/02-mega-moe-architecture.md`：Mega MoE 整体架构与通信-计算重叠机制（已含线上基准坐标）。
- `docs/03-mega-moe-kernel.md`：`sm100_fp8_fp4_mega_moe.cuh` 逐段精读。
- `docs/04-moe-backward.md`：wgrad/dgrad 的 grouped GEMM 实现与数值实验。
- `docs/05-mega-gate-mhc.md`：Mega Gate / Mega mHC。
