# Mega MoE 前向 / 反向实现解析：总览（一页纸）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的收尾（路线图 **P6-1**），
> 把 P0~P5 的结论串成一篇总览：**数据流图、关键机制、与朴素实现的开销对比、反向要点一页纸**。
> 细节分别见 `docs/01`~`docs/14`。
>
> 全文以 **DeepGEMM**（`deepseek-ai/DeepGEMM`，公开版 commit `057ca59`）与
> **DeepEP**（`deepseek-ai/DeepEP`，v2.5.0 / commit `93eb6eb`）为参考坐标。
> **约束**：Mega MoE 主体是 **SM100（Blackwell）专属**，本机 H100（sm90）跑不了端到端；
> 凡 SM100 专属部分均标注「仅分析」，能上 H100 的数值/访存/通信实验给真实数字。

## 0. 三句话版本（先看这个）

1. **MoE 前向像一条「送到别人院区做检查」的流水线**：分诊（router）→ 把病人送到各专家所在 GPU
   （dispatch）→ 各专家算 FFN → 结果送回并加权（combine）。搬人的时间比干活的时间还长——因为
   每个病人都要挂 8 个科室，通信量是 token 量的 k 倍。
2. **Mega MoE 的招数是「一个后厨、五个工位」**：五段塞进一个常驻 kernel，中间只留一个固定格数的
   传菜窗口（ring buffer）。备菜员（dispatch warp）不等全部备齐才交班，灶台（MMA）见菜就炒，
   装盘（epilogue）炒好一段就往下传。于是搬人的时间被炒菜的时间盖住。
3. **反向没有对应的大融合 kernel**：它天然能拆成四个 grouped GEMM（wgrad 走 `k_grouped`、
   dgrad 走 `m_grouped`），wgrad 还留在专家本地 rank、不用 all-reduce。真正的瓶颈是 wgrad
   （K 轴是 token，瘦长、算术强度低）。

下面的章节是这三句话的证据与细节。

## 1. 一句话总览

一个 MoE 层的前向是 **EP dispatch → linear1 → SwiGLU → linear2 → EP combine** 五段串行；朴素实现里
NVLink 通信与 Tensor Core 计算互相等待。**Mega MoE 把五段压进一个 persistent kernel**：用
**对称显存**当通信缓冲、用 **ring buffer + mbarrier + 全局计数** 把 **dispatch warp**（持续拉远端 token）
与 **MMA/epilogue warp**（持续吃本地 tile 做 GEMM）解耦，让通信时延被计算掩盖。反向则**没有**对应的大融合
kernel——它天然是可分解的四个 grouped GEMM（wgrad 走 `k_grouped`、dgrad 走 `m_grouped`），wgrad 留在
专家本地 rank、无需 all-reduce。

## 2. 前向数据流（单 rank、一次 launch 内）

```
 远端 rank 输入 token ──► [对称显存 input_token/sf/topk]（launch 前 host 写入，仅本 rank 读）
                              │
        ┌─────────────────────▼───────────────────────────────────────┐
        │ dispatch warps (0..3)                                        │
        │  ① 本地原子计数每专家 token → 全局 atomic + grid_sync + NVLink │
        │     barrier 汇总 expert_recv_count                            │
        │  ② 无锁 round-robin 认领 token → TMA 分 chunk 远端 pull token  │
        │     + topk weight + SF(UTCCP 转置) → 写本地 l1 ring buffer     │
        │  ③ red_add l1_full_count，块满发信号                           │
        └───────────────┬──────────────────────────────────────────────┘
                        │ l1_full_count == BLOCK_M*(gen+1)
        ┌───────────────▼───────────────┐   TMA + mbarrier 解耦
        │ TMA load A (warp4) 激活+SFA    │──► smem_a[stage] (FP8/FP4)
        │ TMA load B (warp5) 权重+SFB    │──► smem_b[stage]，2-CTA multicast
        └───────────────┬───────────────┘
                        ▼
        ┌───────────────────────────────┐   UTCCP 把 SF 搬进 TMEM
        │ MMA issue (warp6, leader CTA)  │──► SM100_MMA_MXF8F6F4_2x1SM_SS::fma
        │  A/B 互换, UMMA_M=256,UMMA_N=BLOCK_M → TMEM 累加器(双缓冲)
        └───────────────┬───────────────┘
                        ▼ tmem_full_barrier
        ┌────────────────────────────────────────────────────────────┐
        │ epilogue/combine warps (8..15, 2 warpgroup)                 │
        │  L1: TMEM → bf16 → in-place SwiGLU(gran-8 交错) → amax 归约   │
        │      → UE8M0 量化 → FP8 TMA store → arrive l2_full_mask 位   │
        │  L2: TMEM → bf16 → combine_token_buffer                      │
        │      → grid_sync → 通知 peer combine_ready_grid_idx          │
        │  combine: 读本 rank topk → TMA 拉各 expert 所在 rank 的 bf16 │
        │      结果 → fp32 加权累加 → 写 y                             │
        └────────────────────────────────────────────────────────────┘
```

关键点：**dispatch 与 GEMM 之间没有阶段栅栏**，只有 ring slot 的满/空计数；dispatch 填一批、MMA 就开算，
后面 token 边拉边算。L1→L2 也**不是整层 barrier**，而是按 K block 的位图依赖（`l2_full_mask`）逐段交接。

## 3. 关键机制清单

| 机制 | 解决什么 | 关键坐标（真实核对） |
|---|---|---|
| 对称显存 + 跨 rank 信号 | 各 rank 按同偏移映射同一块 buffer，承载通信与同步 | `layout/mega_moe.cuh:54`（`MegaMoESignals`）、`:345`（`MegaMoEBuffer`） |
| dispatch / combine 融合 | 通信走 TMA 远端 pull / NVLink 直写，不进中间张量 | kernel `:342-689`（dispatch）、`:1346-1351`（combine 直写） |
| ring buffer + pool block | 通信与计算按 slot 流水，无阶段栅栏 | `scheduler/mega_moe.cuh:48`（ring 上界）、`get_symm_buffer_size_for_mega_moe`（`mega_moe.hpp:35`） |
| warp specialization | TMA/MMA/调度/epilogue 各司其职，mbarrier 解耦 | kernel 五类 warp：`:342/:690/:756/:819/:939/:946` |
| 细粒度 L1→L2 依赖 | linear2 按 K block 在 linear1 完成一段后立即开始 | `scheduler:145-166`（`L2KBlockDependency`） |
| 死锁规避 | 先 L1 warmup 若干波，避免 L2 占满 ring 等 L1 | `scheduler:372-407`、`get_num_l1_warmup_waves`（`:14-45`） |
| tcgen05 UMMA + A/B 互换 | `UMMA_M=256`（2-CTA）、`UMMA_N=BLOCK_M`，按 `valid_m` 动态缩 N | kernel `:819-938`、`mma/sm100.cuh:150-153` |
| UTCCP + UE8M0 block scale | SF 从 smem 4x32 转置进 TMEM；FP8/FP4 幂次 scale | kernel `:894-907`、`math.cuh:114-147`、`packing.cuh:8-12` |
| in-place SwiGLU | gate/up 按 8 粒度交错，epilogue 直接配对 `silu(gate)*up` | `mega/__init__.py:112`、kernel `:1108-1135` |
| locality domain | 双 die 卡按域分任务、少量任务不偷域，减少跨 die 延迟 | `types.cuh:54`、`scheduler:344-370`、`csrc/apis/locality_domain.hpp:74-142` |
| 预取/冗余专家（DeepEP） | `lb_prefetch_weights` / `lb_reduce_grads`，冗余专家梯度累加 | `docs/07` §5，DeepEP `ep.py` |

## 4. 与朴素实现的开销对比

设每 rank $T$ 个 token、hidden $H$、每 token 选 $k$ 个专家、激活 FP8、输出 BF16，EP=8（详见 `docs/01` §2）：

| 方案 | 每层耗时（量级） | 瓶颈 | 说明 |
|---|---|---|---|
| 串行 dispatch→GEMM→combine | $\approx 0.36 + 1.8 \approx 2.2$ ms | NVLink | SM 与链路轮流空闲 |
| chunked 分步流水 | $\max(\text{comm},\text{compute}) + $ 固定开销 | 两者择大 | 需手工切 chunk、多次 launch |
| **Mega MoE（单 kernel 重叠）** | 理想 $\approx \max(\text{comm},\text{compute})$ | 大 batch→算力，小 batch→延迟 | 通信与计算在时间上重叠，省掉较小一侧 |

- 通信量本身是 **K 倍放大**的：dispatch $\approx T\cdot d\cdot H\cdot 1\text{B}$，combine $\approx T\cdot k\cdot H\cdot 2\text{B}$
  （$d\le k$ 是目标 rank 去重后的份数）。以 DeepSeek-V3 量级（$H{=}7168$、$k{=}8$、EP=8、$T{=}4096$）计约 **705 MB**，
  在 H100 实测 ~390 GB/s 下约 **1.8 ms**，**比同层 FP8 计算（~0.36 ms）还贵约 5 倍**——这就是「必须重叠」的量化理由。
- 官方 PR #316 的 SM100 实测印证了这一点（仅分析）：batch=1 时 compute 才 5–7 TFLOPS、互联带宽近 0
  （纯 dispatch/combine 往返延迟）；大 batch 时 compute 达 2038–2438 TFLOPS、互联 400–530 GB/s（确实在重叠）；
  相对 legacy 稳定 **1.5–2x**，越小 batch 收益越大。

## 5. 主体 kernel 精读要点（四篇索引）

| 篇 | 主题 | 最关键的「为什么」 |
|---|---|---|
| `docs/03`（P3-1） | 模板/block config | grid=`num_sms`、cluster=2、512 线程；`BLOCK_N=128`、`BLOCK_K=128/64`；`BLOCK_M` 启发式成本模型 = **每多一个 M block ≈ 多 128 行权重重载**，故「最少 block + 最小可覆盖 tile」 |
| `docs/09`（P3-2） | warp specialization / mbarrier | 两层解耦（TMA↔MMA 的 full/empty、MMA↔epilogue 的 tmem_full/empty）；`advance_pipeline` 相位翻转、生产者 phase=1 约定；仅 epilogue release task slot |
| `docs/10`（P3-3） | tcgen05 MMA / L1·L2 epilogue | 只有 leader CTA 单 warp 发 UMMA；`A/B 互换 + 2-CTA` 使 `UMMA_N=BLOCK_M`；UTCCP 搬 SF；in-place SwiGLU（bf16 round）+ UE8M0 量化 + TMA 转置上屏；combine fp32 归约 |
| `docs/11`（P3-4） | dispatch / combine / NVLink | 分布式前缀和计数 + 无锁 round-robin 认领；分 chunk 远端 pull；combine L2 直写远端 + grid-index readiness 水位线；locality domain 分域调度 |

配套博客文章：`mega-moe-05-kernel`、`mega-moe-06-mbarrier`、`mega-moe-07-mma-epilogue`、`mega-moe-08-dispatch-combine`。

## 6. MoE 反向要点（一页纸）

把 token 级 MoE 层反向拆开，真正是「大 GEMM」的只有四处（`docs/04`）：

| 梯度 | 数学式 | 不等长轴 | kernel 族 | (M,N,K) |
|---|---|---|---|---|
| `dW2` linear2 wgrad | $dW^2_j = A_j^{\mathsf T} dY_j$ | K=token | `k_grouped_*` | (I, H, $n_j$) |
| `dA` linear2 dgrad | $dA_{tj} = dY_{tj}(W^2_j)^{\mathsf T}$ | M=token | `m_grouped_*` | ($n_j$, I, H) |
| `dW1` linear1 wgrad | $dW^1_j = x_j^{\mathsf T} dG_j$ | K=token | `k_grouped_*` | (H, 2I, $n_j$) |
| `dX` linear1 dgrad | $dX_{tj} = dG_{tj}(W^1_j)^{\mathsf T}$ | M=token | `m_grouped_*` | ($n_j$, H, 2I) |

- **判据**：wgrad 的不等长轴是收缩维 K（token）→ `k_grouped`；dgrad 的不等长轴是输出 M（token）→ `m_grouped`，
  N/K 固定（对应 README 第 80 行「K-axis-grouped API for MoE weight backward」）。
- **其余算子**保持 torch：combine 反向（$dY=w\odot dz$、$q=\langle Y,dz\rangle$）、SwiGLU 反向、router/topk 反向。
  **router 反向最大的坑**：归一化 topk 权重的分母 $P_t$ 必须是**原始 softmax 概率在选中集上的和**（不是归一化权重之和，后者恒 1）。
- **布局**：三种等价布局 contiguous（host 已知、行级组号）/ masked（device 已知、组级计数）/ PSUM（device 已知、组级前缀和），
  训练前向与 dgrad 用 contiguous、解码 CUDA graph 用 masked、DeepEP 动态输出用 PSUM 桥接。段必须对齐 `BLOCK_M`（SM90=128/SM100=256）。
- **为什么反向没有 mega-kernel**：反向通信规模小、计算可分解为独立 grouped GEMM、wgrad 留守专家本地 rank 无需 all-reduce。
- **wgrad 是反向瓶颈**：K=token 偏小 → 算术强度低（实测 ~100 TFLOPS）；dgrad 的 K 是隐藏维 → ~450+ TFLOPS。

配套博客文章：`mega-moe-09-backward`、`mega-moe-10-wgrad`、`mega-moe-11-dgrad`、`mega-moe-12-backward-kernels`。

## 7. 外围 mega-kernel：Mega Gate 与 Mega mHC

| 算子 | 融合内容 | 三个非平凡点 | 文档 |
|---|---|---|---|
| **Mega Gate** | BF16 打分 GEMM + softplus/sqrt + bias 排名 + top-k + logical→physical 映射 | ① UMMA 累加器写 TMEM、gate warp 用 `tmem_load_32dp32b` 直读；② **排名用 score+bias、权重用 score**；③ warp 内排序网络 + 3-bit 游标归并的**确定性** top-k | `docs/05` 第一部分 |
| **Mega mHC** | HyperConnection 残差多流混合 + mix 投影 + sigmoid pre/post + Sinkhorn 双随机 comb + RMSNorm(+FP8) | ① UMMA 走 **TF32 TS**（系数投影需高于 bf16 尾数，激活已在 `new_residual` 强制 bf16）；② Sinkhorn 用 warp 内 `shfl_xor` 逐行/列归约；③ 跨 CTA `SplitBarrier` 用「grid 序号+到达数」一次编码、免复位 | `docs/05` 第二部分 |

两者都是 **SM100 专属，仅分析**（`mega_gate.hpp:158-159`、`mega_mhc.hpp:223`），但其架构无关语义在 H100 上逐行移植对拍。

## 8. H100 实测证据台账（能跑的都给真实数字）

| 实验 | 脚本 | 关键数字 |
|---|---|---|
| H100 kernel 可用性 | `harness/probe_h100_env.py` | dense/grouped FP8、BF16、SM90 `k_grouped_*_nt_contiguous` 可跑；`tn` 与 `fp8_fp4_mega_moe` 被 `gemm.hpp:369` 拒（SM100 专属） |
| grouped 三布局 | `harness/exp_grouped_layout.py` | contiguous/masked/PSUM relerr 一致 **3.69e-2**（FP8 噪声），数值等价 |
| EP all-to-all 原语 | `harness/exp_ep_comm.py` | 8×H100 NCCL：渐近 **FP8 ~393 / BF16 ~395 GB/s**（达 478 GB/s 硬件上限 ~82%）、固定时延 **50–90 us** |
| block config 启发式 | `harness/exp_block_config.py` | 复刻 vs `deep_gemm.get_block_m_for_mega_moe` 9 组形状 **全部一致**；FP8xFP4 大 batch 5 级流水/198.6 KiB |
| mbarrier 流水 | `src/mbarrier_pipeline.cu` | 相位约定坐实；96 MiB `cp.async.bulk` NSTAGES 1→4 串行 5.58→3.44 ms（1.62×），去校验 2.29→0.79 ms（2.88×） |
| UE8M0 / SwiGLU | `harness/exp_ue8m0_sf.py` | 位运算 SF 在 25009 样本 **0 不符**；bf16-round SwiGLU 相对误差 **4.5e-3** |
| NVLink 远端访存 | `src/nvlink_access.cu` | 远端延迟 ~991 ns ≈ 本地 ~350 ns 的 **2.83×**；远端带宽 ~368 GB/s ≈ 本地 ~2.85 TB/s 的 **1/8** |
| MoE 反向数学 | `harness/exp_moe_backward_math.py` | float64：四项梯度 vs autograd relerr ≤ **2.9e-16**，中心差分 ≤5.7e-8 |
| wgrad API | `harness/exp_kgrouped_api.py` | nt 4 组 relerr 1.98e-2；`m=4096,n=7168`（8 组）**425 TFLOPS**；tn 在 sm90 被拒 |
| dgrad API | `harness/exp_mgrouped_dgrad.py` | 三布局两两**逐位相等**；`G=8,m=4352,n=2048,k=7168` **366 TFLOPS** |
| 反向接 kernel | `harness/exp_moe_backward_kernels.py` | `vs deq` ~**2.6e-4**（隔离 kernel+布局）、`vs autograd` 2.5e-2~5.1e-2 且 **cos > 0.999**；wgrad ~103/106、dgrad ~449/543 TFLOPS |
| Mega Gate 路由 | `src/mega_gate_routing.cu` | ~7 万条 top-k **0 不符**、权重 relerr ≤1.5e-7、重复专家 **max/min=1.0000** |
| Mega mHC 语义 | `src/mega_mhc_semantics.cu` | 残差/RMSNorm 到 bf16 ulp、mix 系数 ~2e-7、Sinkhorn 双随机到 2e-6、E4M3 **0.5 ulp** |

## 9. 相关线上资料

- DeepGEMM：https://github.com/deepseek-ai/DeepGEMM
- DeepEP：https://github.com/deepseek-ai/DeepEP
- DeepJIT：https://github.com/deepseek-ai/DeepJIT
- [PR #95](https://github.com/deepseek-ai/DeepGEMM/pull/95) 首个 weight gradient kernel（dense + MoE backward）
- [PR #304](https://github.com/deepseek-ai/DeepGEMM/pull/304) 引入 Mega MoE、FP4 Indexer、FP8xFP4 GEMM、PDL
- [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316) Mega MoE 优化 + 基准数据（§4 解读）
- [PR #343](https://github.com/deepseek-ai/DeepGEMM/pull/343) SM90 contiguous k-grouped 切组改 tensormap 的竞态修复
- [PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432) Mega Gate / Mega mHC、Sparse Indexer、MoE 反向提速
- [PR #462](https://github.com/deepseek-ai/DeepGEMM/pull/462) locality-aware Mega MoE、epilogue 类、README 公开版

## 10. 结论与边界

- **前向**：Mega MoE = 对称显存做通信缓冲 + ring buffer 解耦通信与计算 + 细粒度 mask 依赖 + warp specialization；
  单 kernel 省掉中间张量与多次 launch，通信延迟被计算掩盖，官方 SM100 实测相对 legacy 稳定 1.5–2x。（**SM100 专属，仅分析**）
- **反向**：无 mega-kernel，四个 grouped GEMM + gather/scatter-add；wgrad 是瓶颈（K=token 瘦长、算术强度低），
  dgrad 接近峰值。H100 实测 `vs deq ~2.6e-4` 证明 kernel+布局正确。
- **外围**：Mega Gate / Mega mHC 把路由与 HyperConnection 各自压成单 kernel（**SM100 专属，仅分析**），
  语义已在 H100 逐行对拍。
- **边界**：本机 H100 不能跑 `fp8_fp4_mega_moe` / `mega_gate` / `mega_mhc` 端到端（tcgen05/2-CTA/UTCCP/UE8M0 专属）；
  能跑的是 grouped GEMM、反向、EP/访存的**原语级**数值实验——本专题已把这条边界上的证据补齐（§8）。
