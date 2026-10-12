# `sm100_fp8_fp4_mega_moe.cuh` 逐段精读（一）：模板参数与 block config

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 5 篇，对应路线图 **P3-1**。
> P3 计划把主体 kernel 拆成四段讲：**模板/block config（本文）→ warp specialization & mbarrier →
> tcgen05 / UTCCP / SwiGLU → dispatch & combine**。
>
> **重要**：本文讨论的 kernel 是 **SM100（Blackwell）专属**（tcgen05 MMA、2-CTA cluster、
> UE8M0 block scale、FP8xFP4），本机 H100（sm90）**不跑端到端**。但 **block config 是纯 host 启发式、
> 不依赖硬件**，本文用 H100 上可用的 DeepGEMM binding 做了真机对拍（见 §5）。
>
> 源码坐标（`~/github/DeepGEMM`，2026-09-30 公开版 commit `057ca59`）：
> - API 与尺寸推导：`csrc/apis/mega_moe.hpp`
> - block/pipeline 启发式：`csrc/jit_kernels/heuristics/mega_moe.hpp`
> - 编译与 launch：`csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`
> - 主体 kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）
> - 候选 block M：`deep_gemm/include/deep_gemm/layout/mega_moe.cuh:13-17`
> - 调度器 task 结构：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`
>
> 以下行号均逐条核对过。

## 1. 一次 launch 的物理坐标：grid / cluster / block

launch 在 `csrc/jit_kernels/impls/sm100_fp8_fp4_mega_moe.hpp`：

- kernel 用 JIT 现编，实例化的模板实参来自 `config`（`:149-189`），
- `grid_dim = dim3(num_sms, 1, 1)`（`:195`）——**每个 SM 一个 CTA**，`num_sms` 取整卡 SM 数
  （`get_num_sms_for_mega_moe`，`heuristics/mega_moe.hpp:67-71`，理由是 locality domain 要均匀铺满整卡）；
- `cluster_dim = dim3(2, 1, 1)`（`:197`）——**2-CTA cluster**，这是 SM100 2-SM UMMA 与 TMEM 分配的要求；
- `block_dim = dim3(num_dispatch_threads + num_non_epilogue_threads + num_epilogue_threads, 1, 1)`
  （`:196`），即 `128 + 128 + 256 = 512` 线程 = **16 个 warp / 一个 SM**；
- `num_smem_bytes = config.smem_size`（`:194`）。

也就是说：**grid 大小由硬件 SM 数固定，block 内线程数由角色固定，真正随问题规模变化的是
`BLOCK_M/N/K`、`num_stages`、pull 粒度**——这些全部由 host 启发式算出来。下面逐个讲。

## 2. `get_block_config_for_mega_moe`：BLOCK_M / N / K 与 store block

函数在 `heuristics/mega_moe.hpp:88-127`。输入是 `(num_ranks, num_experts, num_max_tokens_per_rank,
num_topk, num_tokens, mma_kind)`，返回 `(cluster_size, block_m, store_block_m_l1, store_block_m_l2,
block_k, num_epilogue_threads)`。

### 2.1 BLOCK_N、BLOCK_K 是常量

- `block_n = 128`（`get_mega_moe_config`，`:209`）。kernel 里有硬断言 `BLOCK_N == LAYOUT_AD_M`（`:168`）、
  `BLOCK_N == 128`（epilogue 段 `:974`），这是 128B swizzle 与 UMMA 的 N 粒度决定的。
- `block_k = 128 / get_num_mma_elem_bytes(mma_kind)`（`:117`）。FP8/FP4 元素 1 字节 → **BLOCK_K=128**；
  BF16 → **BLOCK_K=64**。对 FP8/FP4 而言，因为采用 MXFP8 的 **K 方向 32 元素一个 scale（`kGranK=32`）**，
  BLOCK_K=128 恰好是 4 组 scale，且 `BLOCK_K / UMMA_BLOCK_K == 1`（`UMMA_BLOCK_K=128`）。

### 2.2 BLOCK_M：route 分布 + 每 block 的固定开销

MoE 的 M 维是**每个专家收到的 token 数**，动态且不齐。启发式先估期望：

```
num_expected_tokens = num_tokens * num_ranks * num_topk / num_experts      # :94
num_covered_tokens  = num_expected_tokens + sqrt(num_expected_tokens)       # :95
```

`+sqrt` 是**给路由抖动留一个标准差的余量**（作者注释 "plus a one-sigma routing margin"），避免
热门专家因对齐溢出而多起一轮。

选块的核心权衡在 `:97-110` 的注释里写得很清楚：

> *Every M block costs roughly 128 extra rows (its tasks reload the weight tiles), so use the fewest
> blocks per expert and the smallest tile covering them.*

翻译：**M block 本身不贵，贵在它会让该专家的权重 tile 被重新 TMA 加载**——每个额外 M block 近似
多付 128 行的权重重载成本（`+128 rows`）。所以策略是「**先用尽量少的 block 覆盖，再选能装下的最小
M tile**」，让每个 block 的 M 尽量大（摊薄权重重载），但又不要大到对齐浪费。

具体：

- `num_expected_tokens <= 10` → `block_m = 16`；`10 < … <= 24` → `block_m = 32`（`:100`）。
  小 batch 时 M 很小，直接取小 tile。
- `> 24` 时按 **240 行/block** 估需要的 block 数 `num_blocks = ceil(covered/240)`（`:102`）；
  再在候选 `{64,128,192,240}` 里从**小到大**挑第一个满足 `num_blocks * candidate >= covered` 的
  （`:105-109`），保证「最少 block、最小 tile」。
- 特例（`:103-104`）：`num_blocks == 1` 且 `covered > 192` 且 **每卡专家数 < 14** 时，拆成 2 个 block。
  注释（`:99`）说明：单块 240 只有在**本卡专家够多（约 ≥14 block/rank）**时才划算；专家少时 240 的
  对齐浪费和尾块更伤，所以宁可拆两个。

候选集来自 `layout/mega_moe.cuh:13-17`：`{8,16,32,64,96,128,192,240}`，且 `kLCMCandidateBlockM=1920`
（8 个候选的最小公倍数）用于 ring buffer 容量对齐。host 侧有断言保证选出的 `block_m` 在候选集内
（`heuristics/mega_moe.hpp:120-123`）。

### 2.3 store block：省 smem 换流水深度

`store_block_m_l1/l2` 是 epilogue 每次 TMA-store 的 M 粒度（`:112-115`）：

```
block_m <= 16 -> 8 ; <= 64 -> 16 ; <= 192 -> 32 ; else: fp8xfp4 -> 24, bf16 -> 40
store_block_m_l2 = (block_m == 240 and fp8xfp4) ? 8 : store_block_m_l1
```

BLOCK_M=240 时 L1 store 取 24、L2 取 8（非对称），让「L1 FP8 输出暂存」和「L2 BF16 暂存」
两份 smem 取 max 后更省（见 §3 的 `smem_cd`），从而让出更多 smem 给 **TMA 流水线阶段**。

### 2.4 cluster 与 epilogue 线程是常量

返回 `{2, block_m, …, block_k, 2*128}`（`:126`）：cluster 固定 2；`num_epilogue_threads = 256`。
`num_dispatch_threads = num_non_epilogue_threads = 128` 在 `get_mega_moe_config:220-221` 固定。
SF block 由 UTCCP 对齐规则给出（`:212-213` → `heuristics/sm100.hpp:18-28`）：
`sf_block_m = align(block_m,128)`、`sf_block_n = align(block_n,128)`；kernel 侧断言 `SF_BLOCK_M ==
align(BLOCK_M,128)`、`SF_BLOCK_N == BLOCK_N`（`:141-142`）——**SF 的 K 方向 32 个元素一个字节，
但 M 方向必须 128 对齐才能 UTCCP 4x32 转置进 TMEM**（P2-1 已解析该转置，`:145-149`）。

## 3. `get_pipeline_config_for_mega_moe`：smem 预算 → `num_stages`

函数在 `heuristics/mega_moe.hpp:129-195`。SM100 每 SM smem 上限 `SM100ArchSpec::smem_capacity =
232448` 字节（`heuristics/sm100.hpp:16`，≈227 KiB）。它把 smem 分成**固定区**与**每 stage 区**，
用剩余空间最大化 stage 数：

- 固定区（`:145-172`）：dispatch 专家计数 + 发送缓冲、`smem_cd`（L1/L2 输出暂存取 max）、
  amax warp-pair 归约缓冲、各类 barrier、schedule task payload、tmem 指针。
- 每 stage 区（`:174-184`）：`smem_a`（`load_block_m × block_k`）、`smem_b`（`block_n × block_k`）、
  `smem_sfa/sfb`、full/empty barrier。
- `num_stages = (smem_capacity - smem_fixed) / smem_size_per_stage`（`:191`），断言 `>= 2`，
  返回值里 `smem_size = smem_fixed + num_stages * smem_size_per_stage`（`:194`）。

注意 `load_block_m = block_m / 2`（`:143`）：**A 在 cluster 内做 2-CTA multicast**，每个 CTA 只载一半 M
（kernel `:165` 注释 "Multicast on A"）。所以固定 BLOCK_N、调 BLOCK_M 时每 stage 的 A 部分同比缩小，
而 B（权重）不变——这也是「M block 越少、B 重载越少」的成本模型的物理来源。

pull 粒度 `num_bytes_per_pull`（`get_mega_moe_config:223-229`）从 `hidden * elem_bytes` 起反复减半，
直到 `<= 8192`（FP8/FP4）或 `<= 4096`（BF16），控制 dispatch 一次 1D-TMA 拉多少字节。

## 4. 线程划分：16 warp 的五类角色

模板参数（kernel `:21-55`）里把线程数拆成三项，并推出 warp 数：

```
kNumDispatchWarps        = kNumDispatchThreads / 32        = 4
kNumMMANonEpilogueWarps  = kNumNonEpilogueThreads / 32     = 4
kNumEpilogueWarps        = kNumEpilogueThreads / 32        = 8
kNumEpilogueWarpgroups   = 8 / 4                           = 2
kNumThreads              = 128 + 128 + 256                 = 512
```

kernel 里用 `warp_idx` 做角色分派（`:341-948`），五类角色如下（本 warp 序号按 `kNumDispatchWarps=4`
展开）：

| warp | 角色 | 关键代码 | 做什么 |
|------|------|----------|--------|
| 0–3 | **dispatch**（4 warp, 128 线程，48 或 96 寄存器） | `:342-688` | 统计每专家 token、all-to-all 汇总、从远端 NVLink 拉 token/SF 写本地 ring buffer |
| 4 | **TMA load A**（1 warp） | `:690-755` | 按 task 载激活 `smem_a` + SFA，`tma::copy` 到 full barrier |
| 5 | **TMA load B**（1 warp） | `:756-818` | 载权重 `smem_b` + SFB（含 FP4 unpack），2-CTA multicast |
| 6 | **MMA issue**（1 warp，仅 leader CTA） | `:819-938` | UTCCP 把 SFA/SFB 搬进 TMEM，发 `SM100_MMA_MXF8F6F4_2x1SM_SS` |
| 7 | **scheduler mainloop**（1 warp，仅 leader CTA） | `:939-945` | 跑 `scheduler.mainloop`，生产 task_info |
| 8–15 | **epilogue + combine**（2 warpgroup, 256 线程，208 或 168 寄存器） | `:946-1528` | TMEM→寄存器→smem→TMA store（L1/L2），SwiGLU、amax、combine 写回 |

> 注意 warp 4–7 合起来就是 `kNumMMANonEpilogueWarps=4` 的「非 epilogue」组：一个载 A、一个载 B、
> 一个发 MMA、一个跑调度——这是典型的 **warp specialization**：TMA/MMA 各自单 warp、靠 mbarrier 解耦。

epilogue 的 M 维切分（`:963-974`）：**2 个 warpgroup 各分 BLOCK_M/2，每个 wg 内 4 个 warp 分 BLOCK_N/4**，
再按 `STORE_BLOCK_M_L1/L2` 切成 store block，最后按 `ATOM_M=8` 切原子。这就是 §2.3 那些 store block
常量真正被消费的地方。

### 4.1 寄存器再分配：为什么按每卡专家数分档

模板参数里有 `kUseMoreEpilogueRegisters = kNumExpertsPerRank <= 64`（`:328`），两档寄存器预算：

| 角色 | 每卡专家 ≤64 | 每卡专家 >64 |
|------|-------------|-------------|
| dispatch | 48 | 96 |
| non-epilogue | 40 | 72 |
| epilogue | 208 | 168 |

`:332-335` 有硬约束：三者加权和 `<= 64512`。验算：
- ≤64：`48*128 + 40*128 + 208*256 = 6144 + 5120 + 53248 = 64512`
- >64：`96*128 + 72*128 + 168*256 = 12288 + 9216 + 43008 = 64512`

两边都**正好顶满 64512 寄存器**（SM100 每 SM 64K×32bit 的一部分）。注释（`:327`）说明原因：**每卡专家
越多，scheduler/task 相关状态越多，dispatch 组要多占寄存器**，于是从 epilogue 借；反之 epilogue
（SwiGLU + combine 的寄存器缓冲）多占。这是一个用寄存器预算做 warp 间资源再平衡的例子。

## 5. H100 实测：block config 启发式真机对拍

端到端 kernel 跑不了，但 `get_block_config_for_mega_moe` 是**纯 host 逻辑**，且 DeepGEMM 把它通过
pybind 暴露为 `deep_gemm.get_block_m_for_mega_moe`（`deep_gemm/__init__.py:85`、
`csrc/apis/mega_moe.hpp:21-30,393-394`），因此在 H100 上可以**真机对拍**。

实验 `harness/exp_block_config.py` 做三件事：
1. 按源码逐行用 Python 复刻 `get_block_config_for_mega_moe`（§2）与 `get_pipeline_config_for_mega_moe`（§3）；
2. 调用真实 binding，逐点比较 `block_m`；
3. 打印每种形状下的配置表（block M / store / K / pull / stages / smem）。

在 `kernel_lab`（H100 sm90）运行，原始输出见 `harness/exp_block_config.out.txt`：

```
ranks  exp    tok topk        mma  blkM stL1 stL2  blkK   pull stages  smemKiB real_match
    8  256   4096    8    fp8xfp4   240   24    8   128   7168      5    198.6         OK
    8  256    512    8    fp8xfp4   192   32   32   128   7168      6    223.4         OK
    8  256    128    8    fp8xfp4    64   16   16   128   7168      9    226.9         OK
    8  256     32    8    fp8xfp4    16    8    8   128   7168     10    213.7         OK
    8  256      8    8    fp8xfp4    16    8    8   128   7168     10    213.7         OK
    8   64   4096    8    fp8xfp4   240   24    8   128   7168      5    198.6         OK
    2   16   4096    4    fp8xfp8   240   40   40   128   7168      5    213.1         OK
    8  256   4096    8  bf16xbf16   240   40   40    64   3584      6    221.4         OK
   32  256   4096    8    fp8xfp4   240   24    8   128   7168      5    198.6         OK

复刻 vs DeepGEMM binding：全部一致
```

几个可读的结论：
- **大 batch（4096 token/rank、256 专家）**：`num_expected = 4096*8*8/256 = 1024`，`covered ≈ 1056`，
  `num_blocks = ceil(1056/240) = 5`，选到 `block_m=240`；此时 FP8xFP4 走非对称 store（L1=24/L2=8），
  smem 198.6 KiB，**5 级流水**。
- **小 batch（8 token/rank）**：`num_expected = 8*8*8/256 = 2 <= 10` → `block_m=16`，**10 级流水**
  （tile 小、每 stage 便宜，能把 smem 铺满流水）。
- **BF16**：`block_k=64`、pull=3584（因 BF16 limit=4096），store 取 40，6 级流水。
- 复刻与实际 binding 在 9 组形状上**逐点一致**，证明对启发式的理解无误。

> 说明：`num_stages`/`smem` 是本文对 `get_pipeline_config_for_mega_moe` 的复刻值；`block_m` 则有
> DeepGEMM 真实 binding 背书。二者共用 §3 的公式，因此复刻出的 stages 可作为配置量级参考。

## 6. 小结与接口

- **grid/cluster/线程是常量**：每 SM 一个 CTA、cluster=2、512 线程 16 warp；真正可变的是
  `BLOCK_M`（8 档候选）、`BLOCK_K`（FP8=128 / BF16=64）、`num_stages`、pull 粒度。
- **BLOCK_M 启发式的成本模型**：M block 数的代价主要是**权重重载（约 128 行/block）**，所以「最少
  block + 最小可覆盖 tile」；`+sqrt` 给路由抖动留余量。
- **smem 预算最大化流水**：固定区（输出暂存、barrier、dispatch）先扣掉，剩余全给 A/B/SF 的 stage。
- **线程分工**：4 dispatch / 4 非 epilogue（载 A、载 B、发 MMA、调度）/ 8 epilogue+combine；
  寄存器按每卡专家数在两档间再平衡，两档都恰好顶满 64512。

下一篇（P3-2）将从这个骨架继续：**mbarrier / 相位如何在 dispatch-warps 与 loader/MMA/epilogue
之间建立生产者-消费者关系**，以及 TMA 载入激活/权重/scale 的具体坐标。

配套实验：`harness/exp_block_config.py`、输出 `harness/exp_block_config.out.txt`。
