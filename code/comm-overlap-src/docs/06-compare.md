# 对照：UniEP 与 Flux 各自把「等」切到多细

前五篇分别拆了两份代码。这一篇只做一件事：把它们放到同一张表上，然后给一个能解释「为什么是这个粒度」的框架。
如果只想记一句话：**overlap 的收益不来自"通信和计算同时跑"，而来自"把等待切小，并让每一小块等待都发生在别的事情正在跑的时候"。**

## 1. 一张表

| 维度 | UniEP（EP，MoE） | Flux（TP，AG-GEMM / GEMM-RS） |
|---|---|---|
| 通信执行者 | 同 kernel 内的 dispatch warp（TMA G2S + S2G）/ epilogue warp（SIMT 128b store） | 另一条 stream 上的 `cudaMemcpyAsync` / NVSHMEM put（拷贝引擎或别的 SM） |
| 计算执行者 | 同 kernel 内的 MMA warp（tcgen05） | 同 kernel 内的 MMA warp group |
| 等待原语 | mbarrier（SMEM 内）+ `ld.global.acquire.gpu` 自旋在 `expert_signals[i]` | `ld.global.acquire.sys` 自旋在 `barrier_buffer[chunk]`（`SystemBarrier::wait_eq`） |
| 等待粒度 | **一个 expert** 的 token 全部落地 | **一个 M-chunk**（`M / (world_size × SPLIT)` 行）落地 |
| 谁决定"该等哪块" | tile scheduler：M-contiguous 顺序 ⇒ 当前 group（expert）就是下一个要等的 | tile scheduler 的 M 重映射 + `M_idx × TILE_SIZE_M / m_per_data_chunk` ⇒ 当前 tile 依赖的 chunk 区间 |
| 反压 / 复用 | SMEM 里的 dispatch ring buffer（`dispatch_num_stages` 行）+ mbarrier 空/满 | 单一 `input_buffer`；flag 只增不复位，每次 forward 结束由 `reset_signals()` 清零 |
| 确定性 | 显式设计（M 连续 + 固定顺序 top-k 归约） | AG-GEMM 逐元素仍确定（K 完整、单写者）；但 RS 的 `red.global` 原子加让求和顺序不定 |
| 代价 | kernel 内 8 warp 角色耦合、SMEM 要在 dispatch buffer 与 GEMM stage 之间分账、per-expert 计数器要复位 | 多一条 stream、flag 数量 `world_size × SPLIT`、拷贝与 GEMM 争抢 HBM/L2 带宽、flag 复用需要跨调用清零 |

## 2. 一个能解释粒度的模型

设第 $c$ 块数据在 $t^{\text{arrive}}_c$ 到达，而计算侧最早在 $t^{\text{need}}_c$ 需要它。暴露出来的等待是

$$
T_{\text{exposed}} = \sum_c \max\bigl(0,\; t^{\text{arrive}}_c - t^{\text{need}}_c\bigr)
$$

overlap 的所有技巧，本质只有两条路：

1. **把 $t^{\text{need}}_c$ 往后推**：调度重排。Flux 的 M tile 重映射（先算本地那份数据）、UniEP 的 M-contiguous 顺序（同一个 expert 的 tile 连着算，只需等一次）。
2. **把 $t^{\text{arrive}}_c$ 往前拉**：预取 + 多 stage。UniEP 的 dispatch ring（`dispatch_num_stages`）、Flux 的 `SPLIT`（默认 1；调大就把每个 chunk 再切成 $SPLIT$ 份，先到的几份先解除等待）。

而「粒度」决定这个求和里有多少个 $c$。$c$ 越细，$\max(0, \cdot)$ 越容易等于 0；但每细一档都要多付一份元数据：
UniEP 是每个 expert 一个 int32 计数器 + 一个 signal；Flux 是每个 chunk 一个 flag + 一次 `CUStreamWriteValue`。

> 关键不是"越细越好"，而是 **细到「等待」小于「填充等待所需的工作量」就够**。
> UniEP 选 expert 而不是 token：因为 GEMM 本来就要按 expert 分组算，expert 是天然的调度单元。
> Flux 选 chunk 而不是 tile：chunk 是拷贝引擎一次 DMA 的粒度，比 tile 粗一档，flag 数量才可控。

## 3. 两份代码在同一个问题上的镜像选择

有意思的是，两份实现对「数据进场」和「结果出场」的方向选择是一致的：

| | 进场（dispatch / all-gather） | 出场（combine / reduce-scatter） |
|---|---|---|
| UniEP | **pull**：TMA 从对端 symmetric buffer 读自己需要的行 | **push**：epilogue 直接把行写到目标 rank 的 staging |
| Flux | **pull**：`cudaMemcpyAsync` 从对端 buffer 读整段 | **push**：同机 `red.global.add.f32` 累加到 owner，owner 再跨机 `nvshmemx_putmem_nbi_block` |

原因不难理解：

- **进场用 pull**：接收方知道自己要什么（expert 的 token 行 / 整个 chunk），可以按需取，还能顺便做去重与对齐；
- **出场用 push**：结果一行只属于一个目标 rank，直接写过去最省事，不需要目标侧再起一次收集；
  而且 push 天然和 epilogue 的产出节奏重合——算完一个 tile 就推一个 tile，通信被摊平。

代价也一致：push 之后目标侧必须知道"这块齐了没有"，于是两边都出现了计数器/flag
（UniEP 的 `expert_signal_counters`、Flux 的 `barrier_ptr + tile_idx` 计数）。

## 4. 什么时候选哪种

| 场景特征 | 倾向 | 理由 |
|---|---|---|
| EP / MoE，通信量不规则、按 expert 分组 | 分区型（UniEP） | 分组本身就是计算单元，signal 可以挂在 expert 上；SM 内角色分工能吃掉小消息的启动开销 |
| TP，通信规则、数据连续、单次量大 | 融合型（Flux） | 一次 DMA 搬一大段最划算；chunk flag 实现简单、对 GEMM 主体零侵入 |
| 想要 bitwise 可复现 | 必须显式设计调度 | 只有固定 tile 顺序 + 固定归约顺序才能做到（见 [03](03-uniep-determinism.md)） |
| 只有单卡/无 NVLink | 都别上 | 两份代码的 overlap 都建立在"通信有独立引擎/独立 SM + 远端可直写"之上 |

## 5. 两边共有的五个坑

1. **flag 的生命周期**：Flux 在 `forward` 结尾调 `reset_signals()`（先等 `cp_event`，再跨 rank 同步，最后 `barrier_buffer.zero_()`）；
   UniEP 在每次 launch 前 `_reset_dispatch_signals_barrier()` 并做一次 group barrier，凑齐所有 rank 才继续。
   忘了复位 → 第二次调用直接读到上一次的 1，数据错但不报错。
2. **内存序**：`arrive` 必须是 release（`fence.acq_rel.sys` + `red.relaxed`），`wait` 必须是 acquire。
   只写 `volatile` 读会拿到旧数据却看不到新写的 payload。
3. **scope 选错**：跨卡必须 `sys`（Flux `system_barrier.hpp`）或 `gpu` + 对称内存 + 网络序（UniEP intranode 场景用 `gpu` scope 就够）。
4. **persistent kernel 的 CTA 数**：grid 超过物理 SM 数会让"等对端"的 CTA 占着 SM 不撒手，另一端排不上队 → 死锁或长尾。
   两边都用 `max_active_clusters` / `min_blocks_per_mp=1` 把 grid 卡在硬件容量内。
5. **对齐**：UniEP 需要 `expert_alignment ≥ cluster_tile_m`（否则 M-contiguous 调度和 signal 粒度对不上），
   host 侧直接 `max(user_value, GEMM_CLUSTER_TILE_M)` 兜底；
   Flux 的 `m_per_data_chunk = M / (world_size × SPLIT)` 是整数除法，**没有显式整除检查**（`gather()` 里只检查了 `chunk_size` 能否被 `n_dim × elem_size` 整除）——
   除不尽时 chunk 边界会落在 tile 中间，只能靠"多等一个 chunk"兜住，读代码时要自己算一遍。

## 6. 想继续往下读

- 论文层面的结论与数字：`content/posts/comm-overlap-13-uniep/`、`comm-overlap-14-megaflux/`、`comm-overlap-12-latest/`。
- 只想要"账"不想要代码：`code/comm-overlap/` 里的 7 个纯 Python 模型。
- 想看 MoE 的 mega-kernel 另一种写法：`code/mega-moe/`（DeepGEMM + DeepEP 那一路）。
