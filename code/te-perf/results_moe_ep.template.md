# MoE EP / Group-GEMM 性能测试结果汇总（uniEP / DeepEP / TE group-gemm）

> `kernel耗时` 为 **CUPTI 实测的纯 device kernel 执行时间**（µs）：由 `torch.profiler` 采集一次调用所 launch 的全部 kernel，**每个 kernel 先对 50 次调用取中位**、再按"每次调用的 launch 次数"加权求和，**不含 launch / 主机派发开销**；`wall中位` 为每次调用两侧 CUDA event 计时的中位值（含 launch 与跨卡等待）。设备均为 **NVIDIA H100**（80GB HBM3，单机 8 卡 NVLink NV18 全互联，EP=8）。原始数据：`moe_ep_uniep.csv`、`moe_ep_deepep.csv`、`moe_ep_tegmm.csv`、`moe_ep_uniep_sm_sweep.csv`、`moe_ep_uniep_topk8_M1024_H1024.csv`；一键复现脚本 `run_all.sh`，表格由 `build_results_doc.py` 自动生成。
>
> 三条阅读前提（会显著影响绝对值，不影响同机对比的有效性）：
> 1. **本机是"残血"通路，但不是选错容器导致的**（已逐一验证，证据见 §5.1）：宿主驱动 **570.211.01**；`kimi26_train` 容器**没有**挂载 `/dev/nvidia-nvswitch*`、`/dev/nvidia-nvlink`、`/dev/infiniband`，所以 NVLS 与 IB 不可用；但用 `--privileged` + 全套 device 的容器复测，**NVLS 依旧 bind 失败**（3 种 CUDA/NCCL 组合、且强制加载宿主原生 `libcuda.so.570.211.01` 也一样），属**主机级 Fabric Manager/NVSwitch 问题**；flash_comm 的 fabric 句柄需要 IMEX（本机 `nvidia-imex` inactive、无 `/dev/nvidia-caps-imex-channels`），同样与容器无关，故只能走 `FLASH_COMM_BUFFER_BACKEND=cuda_ipc`。**但这些"残血"项对本文的单机 EP=8 结果没有影响**：被测的 10 个 op 里没有任何 NCCL 集合通信（flash_comm 的 barrier 走自己的 NVLink 对称内存 kernel `barrier_all_on_stream`，DeepEP 的 `csrc` 里 NCCL 集合调用为 0，NCCL 只在 init 阶段交换 handle 时用到），NVLS 只加速 NCCL 的 multicast 大 buffer 集合通信，fabric 句柄只是 buffer 的导出/映射方式（单机它和 `cuda_ipc` 都落到同一条 NVLink P2P 写路径）。逐条证据见 §5.1。
> 2. **共享机器有 ~3ms 级主机停顿**（50 次调用里约 5–10% 会撞上，p90 常是中位的 3–4 倍）。因此本文 wall 一律报 **中位**，CUPTI 也用"按 kernel 取中位"而非平均值，否则单次停顿会把均值抬高 2 倍（实测 TE FC1@H3072：均值 1689µs vs 中位 898µs）。另外同一档在不同批次之间还有 **±10–20% 的绝对漂移**（机器负载），所以跨框架比较只用同一批次内的数据，不要拿本文的绝对值和别的机器比。
> 3. **三家 dispatch 的行数语义不同**，GEMM/端到端对比必须先对齐行数，见 §2。

| 记号 | 含义 |
|---|---|
| `T1024` | **每卡** 1024 token（全局 8192 token） |
| `T128` | **每卡** 128 token（全局 1024 token） |
| `H` / `I` | hidden size / MoE intermediate size，本文取 `H = I` |
| `E32` / `EP8` | 32 个专家 / 8 卡 EP，每卡 4 个本地专家 |
| `topk16` | 每个 token 选 16 个**互不相同**的专家（uniform routing） |

## 0. 结论速览

- **group GEMM：uniEP 的 CuTeDSL grouped GEMM 全面快于 TE 的 `GroupedLinear`**，按"每行耗时"归一（两家行数不同，见 §1）后：`T1024` 档 **1.15–1.41×**，`T128` 档 **1.39–2.02×**。折算单卡有效算力 uniEP **418–799 TFLOPS**（H100 bf16 dense 峰值 989.4 的 42%–81%）、TE **207–647 TFLOPS**（最差是 `T128/H1024` 的 FC2，只有峰值的 21%）。
- **通信：DeepEP 的 combine 明显更快，dispatch 同档略优**。同为 per-(token,expert) 展开语义（`T1024`）：combine **68.7µs vs 120.2µs**（H1024）、**174.3µs vs 401.3µs**（H3072），快 **1.75–2.3×**；dispatch **120.0µs vs 139.9µs**、**339.1µs vs 437.1µs**，快 **1.2–1.3×**。`T128` 档两边都被 ~60–90µs 的 launch/host 开销吃掉，wall 上打平（79.4 vs 85.2、91.2 vs 84.7）。
- **uniEP 的 MegaKernel 只在"大 batch + 大 H"赚钱**：`T1024/H3072` 下 MK1（dispatch+FC1 融合）**1138.9µs vs 非融合 1317.9µs（快 14%）**，MK2（FC2+combine 融合）**766.6µs vs 830.6µs（快 8%）**，整层融合 e2e **2186µs vs 2851µs（快 23%）**。
- **小 shape 上融合是负收益**：`T1024/H1024` MK1 **344.9 vs 243.9µs（慢 41%）**，整层融合与非融合打平（597.5 vs 597.6µs），但 wall 反而慢 32%（744.9 vs 565.8µs）；`T128` 档 MK1 慢 **1.5–2.4×**，整层 `T128/H1024` 慢 30%。原因见 §4.3：MK1 有 ~100µs 级固定代价、MK2 约 10µs，且融合后尾部无法提前退出（持久化 132 block、专家信号初始化、尾部 tile 不能提前退出），而小 shape 下 dispatch/GEMM 本身只有 25–130µs。
- **最佳组合是"混搭"而非单栈**：DeepEP 通信 + uniEP GEMM，`T1024/H3072` 一层 MoE ≈ **2158µs**（分量相加），比 uniEP 全融合（2186µs）快 1.3%，比 DeepEP 通信 + TE GEMM（2330µs）快 7%；`T1024/H1024` 混搭 **448µs** 对 uniEP 非融合 597.6µs，快 **25%**。
- **SwiGLU 是隐藏大头**：uniEP 没提供融合 SwiGLU，`T1024/H3072` 的 `silu(gate)*up` 要 **358.3µs（占整层 16%）**、`T1024/H1024` 要 107.1µs（18%）。纯 torch 实现，换 fused SwiGLU 可直接砍掉一大半。
- **topk=16 不是 MegaKernel 变慢的原因**：topk 降到 8（`T1024/H1024`）融合依旧慢（e2e 367.9 vs 335.4µs），所以慢在小 shape 的固定开销。
- **"每卡 128 token"这一档（全局 1024 token）三家都不划算**：整层只有 200–460µs，其中 dispatch+combine 占 86–125µs；这一档要么增大 EP，要么别分布式。

## 1. 测试配置

| 配置 | 全局 token | 每卡 token | dispatch 后行数（uniEP / DeepEP-expanded） | H | I | topk | E | EP |
|---|---|---|---|---|---|---|---|---|
| T1024 H1024 | 8192 | 1024 | 17408 / 16896 | 1024 | 1024 | 16 | 32 | 8 |
| T1024 H3072 | 8192 | 1024 | 17408 / 16896 | 3072 | 3072 | 16 | 32 | 8 |
| T128 H1024 | 1024 | 128 | 2816 / 2432 | 1024 | 1024 | 16 | 32 | 8 |
| T128 H3072 | 1024 | 128 | 2816 / 2432 | 3072 | 3072 | 16 | 32 | 8 |

- 行数由 routing 决定：`T1024` 时每卡发出 1024×16 = 16384 个 (token, expert) 对，8 卡共 131072 个，均匀落到 32 个专家上 → 每卡 4 个专家共 ~16531 行；`T128` 时为 ~2116 行。表中行数是 **rank 0** 的值，其它 rank 因本地专家分布不同会差 ±5%（实测 uniEP 为 16384/16896/17152/17408）。
- uniEP 把每个专家的行数对齐到 `cluster_tile_m = 256`，故 16531 → 17408、2116 → 2816；DeepEP expanded 对齐到 128，故 16531 → 16896、2116 → 2432。**两者差 ~3%（T1024）/ ~15%（T128）**，这是下面 GEMM 对比要注意的分母差异。

## 2. 行数语义（对比前必读）

| 路径 | 一次 dispatch 发出的行 | `T1024` 行数 | `T128` 行数 | 与 GEMM 的接口 |
|---|---|---|---|---|
| uniEP `dispatch_cutedsl` | 每个 (token, expert) 一行，按专家分组、按 256 对齐 | 17408 | 2816 | `recv_expert_counts`（per-expert，已对齐） |
| DeepEP `dispatch(expert_alignment=1)` | 每个 (token, **源 rank**) 一行（同源去重） | 7807 | 979 | 需再 `do_expand` 才能喂 grouped GEMM |
| DeepEP `dispatch(do_expand=True, expert_alignment=128)` | 每个 (token, expert) 一行，按 128 对齐 | 16896 | 2432 | `handle.psum_num_recv_tokens_per_expert` |

**所以**：比较 dispatch 带宽时用 `deepep_dispatch`（去重语义，行数少一半，天然占优，不代表更强）；比较"喂给 expert GEMM 的通信成本"时用 `deepep_dispatch_expanded`（与 uniEP 同语义）。本文 §3.2 两个都给，§3.3 端到端用 expanded。

## 3. 结果

### 3.0 怎么读"两行"的单元格：上=wall 中位，下=CUPTI kernel

§3.2 / §3.3 的通信与端到端单元格写成 `上/下` 两行，因为**同一个 op 的两个口径回答的是不同问题**：

| | 怎么测（`moe_bench_common.measure`） | 含什么 | 不含什么 |
|---|---|---|---|
| **上：`wall 中位`** | 50 次调用背靠背入队，**每次**调用前后各记一个 CUDA event，最后一次 `synchronize()` 后统计 50 个样本、**取中位**（同批次还给了 `p90`/`max`） | kernel 执行 + **kernel 之间的 launch 间隔** + 主机派发时间 + **跨卡等待/同步**（dispatch 的轮询、combine 的 barrier） | profiler 开销（这一轮不开 profiler） |
| **下：`CUPTI kernel`** | 另起一轮 `torch.profiler`（CUDA activity）采同样 50 次；按 `(kernel 名, grid, block)` 分组，**每个 kernel 取 50 次的中位**，乘"每次调用的 launch 次数"，再对所有 kernel 求和 | **纯 device 执行时间** | launch 延迟、kernel 之间的空隙、主机时间、kernel 起来之前的跨卡等待 |

所以 **`wall − kernel` ≈ 这个 op 的"非计算开销"**（launch + 派发 + 同步 + 跨卡等待）。实测典型值：

| op | 配置 | wall 中位 | CUPTI kernel | 差值 | 怎么理解 |
|---|---|---:|---:|---:|---|
| `uniep_dispatch` | T1024/H1024 | 148.7 | 139.9 | +8.8 | 大 shape，kernel 主导，launch 只占 6% |
| `uniep_dispatch` | T1024/H3072 | 452.5 | 437.1 | +15.4 | 同上，3% |
| `deepep_combine_expanded` | T1024/H3072 | 179.6 | 174.3 | +5.2 | 同上，3% |
| `uniep_swiglu` | T1024/H3072 | 412.4 | 358.3 | +54.1 | SwiGLU 是 **3 个 kernel**（silu / mul / `.contiguous()` 拷贝），每次切换都有 launch 间隔 |
| `deepep_dispatch_expanded` | T128/H1024 | 79.4 | 25.0 | **+54.4** | 小 shape：kernel 只 25µs，其余 54µs 是 launch + host 准备 |
| `te_grouped_gemm_fc1` | T128/H1024 | 170.8 | 27.3 | **+143.5** | TE 走 4 次 grouped GEMM launch + Python 侧参数打包，host 成了瓶颈 |
| `te_grouped_gemm_fc1` | T1024/H3072 | 835.7 | 965.2 | **−129.5** | 见下面"为什么 kernel 会大于 wall" |
| `uniep_e2e_sequential` | T1024/H3072 | 2464.3 | 2850.9 | **−386.6** | 同上（5 个 kernel 的链） |

**为什么 `kernel` 有时反而大于 `wall`**（−5%～−15%），两个原因，都不是测错：

1. **"逐 kernel 中位求和" ≠ "整段中位"**：每个 kernel 的中位在 50 次里独立取；若 A 在某些迭代慢、B 在另一些迭代慢（主机抖动），`sum(median_k)` 会大于 `median(sum_k)`；
2. **跨迭代重叠**：50 次调用背靠背入队（中间不 `synchronize`），第 i 次 combine 的尾巴常与第 i+1 次 dispatch 的头重叠，于是 `wall`（单次 event 间隔）比"把每个 kernel 时长相加"更短。

**实践上怎么选**：

- 比"同等的活谁算得快"（算法/框架效率、TFLOPS）→ 用 **kernel**：剔除 launch 与主机差异，本文 §3.1 / §4.1 就是这个口径；
- 判断"这层在训练关键路径上要等多久"、判断融合是否划算 → 用 **wall**：它含 launch 与跨卡同步，§4.3 的 MegaKernel 盈亏主要看它；
- 两者差距 <15% 说明这层没有 host/launch 瓶颈；差距很大（如 TE 在 T128 差 143µs）说明**该优化 launch 次数或主机路径，而不是 kernel 本身**。

{{TABLE_GEMM}}

{{TABLE_COMM}}

{{TABLE_E2E}}

{{TABLE_RAW}}

## 4. 分析

### 4.1 group GEMM：uniEP（CuTeDSL）vs TE（`GroupedLinear`）

按"每行耗时的 flop 归一"比较（行数不同，必须归一）：

| 配置 | FC1 有效算力 uniEP | FC1 有效算力 TE | FC2 有效算力 uniEP | FC2 有效算力 TE |
|---|---|---|---|---|
{{TABLE_TFLOPS}}

- uniEP 的 grouped GEMM 是手写 CuTeDSL（tile 128×256、warp-specialized、per-expert 变长 M），**最小档（2816 行）也有 418–452 TFLOPS，最大档 799 TFLOPS（峰值的 81%）**；TE 走的 cuBLASLt grouped 路径在小 m 段掉得厉害（`T128/H1024` FC2 仅 207 TFLOPS、`T128/H3072` FC1 仅 393 TFLOPS，同样 2816/2116 行时 uniEP 是 418/794）。
- 大 m 段（`T1024/H3072`，每专家 ~4133 行）差距收窄但仍存在：uniEP 746/765 TFLOPS vs TE 647/603 TFLOPS（领先 15–27%）；`T128` 档拉开到 1.4–2.0×。
- **只有 TE 能直接做反向**（`GroupedLinear` 自带 autograd），本文给了 fwd+bwd 数据：`T1024/H3072` FC1 880.8 → **2072.2µs（2.4×）**、FC2 429.3 → **1712.4µs（4.0×）**；小 shape 上 kernel 只有 58–72µs，wall 却 544–688µs（autograd + 每个 gemm 的 wgrad launch），训练时这部分能被 pipeline 掩盖但不能忽略。uniEP / DeepEP 只提供前向。
- TE 还有 **launch 开销**问题：`T128/H1024` FC1 的 kernel 只要 27.3µs，wall 中位却 170.8µs（4 次 grouped GEMM launch + 参数打包 + 主机抖动）。在真实训练 loop 里这部分能被上一层 kernel 掩盖，但 microbenchmark 里非常显眼；uniEP/DeepEP 的融合路径只有 1–2 次 launch。
- 注意 TE 的 kernel 时间比 wall 中位还高一点（965.2 vs 835.7 @H3072 FC1）：CUPTI 的"每 kernel 取中位再求和"在多个 kernel 之间不相关时会略高于"整段的 wall 中位"，属于口径差异，量级（±15%）可忽略。

### 4.2 通信：DeepEP 快的部分与不快的地方

- **combine 是 DeepEP 的强项**（`T1024/H3072`：174.3 vs 401.3µs，2.3×）。uniEP 的 `combine` 走 "push + NVL barrier + topk reduce" 三步（`push_mode="tile"`），barrier 是全卡同步点，要等 17408 行的 push 全部落地；DeepEP 的 combine 用 expand-mode 直接对本地 buffer 做 reduce，省掉了显式 barrier。
- **dispatch 两家同档、DeepEP 略优**（120.0 vs 139.9µs @H1024；339.1 vs 437.1µs @H3072）。uniEP 还需要额外一次 `layout`（66µs：histogram + cumsum + allgather + layout kernel），DeepEP 的 layout 是纯 device 的。
- **`T128` 档要看 wall 而不是 kernel**：DeepEP 的 dispatch kernel 只有 25.0µs，但 wall 中位 79.4µs（新增一次 launch + host 侧准备）；uniEP 是 50.8 / 85.2µs。两者在一层 ~200µs 的规模上已经没区别，谁快取决于当次主机抖动。
- **`do_cpu_sync`（训练默认）对主机停顿极其敏感**：安静时两种约定没差别（`T1024/H3072` expanded：cpu_sync 339.1/320.8µs vs async 314.7/323.0µs；`T1024/H1024`：120.0/121.1 vs 113.0/119.9），但在主机停顿窗口里 cpu_sync 的单次 dispatch 会被放大到 **3.3–6.8ms**（两次独立观测：一次是 H=3072、一次是本机抖动期），而 async 约定只涨到 ~320–400µs。原因是 cpu_sync 要在 host 侧停等 recv count，任何调度延迟都直接叠加到关键路径上。表里两种约定都列了（`deepep_dispatch*` = cpu_sync，`deepep_dispatch*_async` = DeepEP 自己 test harness 的约定）。**建议：训练里用 async + 一次 device→host 的 counts 回读，或直接用上一轮 counts（一步延迟），别在关键路径上反复 cpu_sync。**

### 4.3 MegaKernel 何时赚：固定开销 vs 可重叠部分

MegaKernel 的收益上限 = `min(comm, gemm)`，代价 = 持久化 kernel 的固定开销 + 尾部无法流水：

| 配置 | dispatch | FC1 | 非融合合计 | MK1 | MK1 盈亏 | FC2 | combine | 非融合合计 | MK2 | MK2 盈亏 |
|---|---|---|---|---|---|---|---|---|---|---|
{{TABLE_MK}}

- `T1024/H3072`：comm(437) 与 gemm(881) 都很大，重叠空间大，MK1 赚 14%、MK2 赚 8%；整层 2 次 launch 的 e2e 赚 23%。
- `T1024/H1024`：gemm 只有 104µs，MK1 的固定开销 ~100µs（344.9 vs 243.9），融合反而慢 41%；MK2 只多 12.5µs 但丢掉了重叠收益，慢 7%；整层打平（wall 慢 32%）。
- `T128`：两个阶段都只有 25–134µs，固定开销完全吃掉收益（MK1 慢 51–140%，MK2 慢 6–10%）。
- 但 `T128/H3072` 的**整层** e2e 融合反而快 5%（437.2 vs 458.9µs）：非融合要走 5 次 launch + 一次 layout，每次 ~20–40µs 的 launch/host 开销累加起来比 MK1 的固定开销还贵。所以"融合是否划算"要按**整层 launch 数**算，不能只看单个 MK。
- SM 预算默认值已接近最优（`gemm_num_sm=None, comm_num_sm=None` = 整机 132 SM 动态分工）：`T1024/H3072` MK1 默认 1198µs、(132,132) **1129µs**、(96,None) 1372µs、(64,32) 1789µs、(32,96) 3224µs；`T1024/H1024` 默认 **311µs**、(132,132) 346µs、(96,None) 389µs。可见 MK1 的慢不是 SM 划分没调好（显式 132/132 只快 6%），而是固定开销与尾部依赖。

### 4.4 SwiGLU 是隐藏成本

| 配置 | SwiGLU (uniEP 路径) | 占整层非融合 e2e |
|---|---|---|
{{TABLE_SWIGLU}}

uniEP 只提供 dispatch/GEMM/combine 的融合，SwiGLU 得自己写。`T1024/H3072` 时 358.3µs 相当于 FC2 的 83%、整层的 16%；`T1024/H1024` 时 107.1µs 甚至超过了 FC1（104.0µs）。原因是 torch 的 `F.silu(g) * u` 会走 3 个 kernel（silu、mul、以及 `.contiguous()` 的一次拷贝），17408×3072 的中转张量来回搬。换成单 kernel 的 fused SwiGLU（一次读 2I 写 I）能压到 ~100µs 量级，整层再快 10%+。

### 4.5 topk 与 SM 预算不是瓶颈

| 探针 | 数据 | 结论 |
|---|---|---|
| topk=16 → 8（`T1024/H1024`） | dispatch 139.9 → 74.3µs（行数 17408 → 8960）、FC1 104.0 → 57.5µs、MK1 344.9 → 256.9µs、e2e 融合 597.5 → 367.9µs / 非融合 597.6 → 335.4µs | 融合依旧慢 10%，**MegaKernel 的劣势来自固定开销，不是 topk=16 展开的行数** |
| `gemm_num_sm` 扫描（MK1，`T1024/H3072`） | 默认 1198µs、(132,132) 1129µs、(96,None) 1372µs、(64,32) 1789µs、(32,96) 3224µs | 默认"132 SM 动态分工"已接近最优，显式指定只快 6% |
| `gemm_num_sm` 扫描（MK1，`T1024/H1024`） | 默认 311µs、(132,132) 346µs、(96,None) 389µs | 同上；小 shape 下 MK1 的 ~100µs 固定开销与 SM 划分无关 |

> `group_gemm_combine`（MK2）没有 `comm_num_sm` 参数（只有 `gemm_num_sm`），且默认值已是最优档，故只报默认值。

### 4.6 选型建议（本机、bf16、单机 8 卡）

| 场景 | 建议 |
|---|---|
| 大 batch（≥1024 token/卡）+ 大 H（≥3072） | **uniEP 全融合**（2186µs）≈ **DeepEP 通信 + uniEP GEMM 自拼**（2158µs）；前者少 4 次 launch、无 host 依赖，推荐前者 |
| 大 batch + 小 H（1024） | **DeepEP 通信 + uniEP GEMM**（448µs），比 uniEP 非融合（597.6µs）快 25%；MegaKernel 别开（打平但 wall 差 32%） |
| 小 batch（128 token/卡） | 通信用 DeepEP expanded（dispatch 46.7–25.0µs / combine 40.0–29.3µs），GEMM 用 uniEP（比 TE 快 1.4–2.0×）；`T128/H3072` 可以开融合（省 launch），`T128/H1024` 别开 |
| 需要反向 | 只有 TE 的 `GroupedLinear` 现成可用（本文给了 fwd+bwd 数据）；uniEP/DeepEP 只有前向 |

## 5. 复现与环境

```bash
# 全部数据（~45 分钟，8×H100，串行跑，避免抢卡干扰）
bash run_all.sh
# 单独跑某一档（每档必须独立进程，见下）
env NCCL_NVLS_ENABLE=0 FLASH_COMM_BUFFER_BACKEND=cuda_ipc NCCL_LSA_TEAM_SIZE=8 \
  torchrun --nproc_per_node=8 bench_uniep_ep.py --tokens 1024 --hidden 3072
env LD_PRELOAD=/opt/nccl232/nvidia/nccl/lib/libnccl.so.2 EP_DISABLE_GIN=1 NCCL_NVLS_ENABLE=0 \
  torchrun --nproc_per_node=8 bench_deepep_ep.py --tokens 1024 --hidden 3072
env NCCL_NVLS_ENABLE=0 torchrun --nproc_per_node=8 bench_te_group_gemm.py --tokens 1024 --hidden 3072
```

| 文件 | 作用 |
|---|---|
| `moe_bench_common.py` | routing 生成（三家共用同一份 routing）、per-iteration CUDA event 统计、CUPTI 中位统计、CSV 落盘 |
| `bench_uniep_ep.py` | uniEP 非融合 5 个 op + MK1/MK2 + 融合/非融合 e2e |
| `bench_deepep_ep.py` | DeepEP plain/expanded/async 的 dispatch、combine、comm-only e2e |
| `bench_te_group_gemm.py` | TE grouped GEMM FC1/FC2、fwd+bwd、FC1+SwiGLU+FC2 |
| `check_uniep_moe.py` | 与纯 torch MoE 参考实现对比（正确性门禁） |
| `sweep_sm_uniep.py` | MegaKernel 的 `gemm_num_sm`/`comm_num_sm` 扫描（§4.5） |
| `build_results_doc.py` | 由 CSV + `results_moe_ep.template.md` 生成本文档（表格部分全自动） |
| `diag_deepep2.py` | DeepEP 的 `do_cpu_sync` / async 约定对比探针（§4.2 的证据） |

### 5.1 "残血"项逐条定位（不是选错容器的锅）

宿主：8×H100 80GB SXM（NV18 全互联、144 条 NVLink 全 UP）、driver **570.211.01**、`nvidia-fabricmanager-570` active（`Fabric State: Completed`）、`nvidia-imex` **inactive**、`/dev/nvidia-caps/` 只有 cap0-2（**无 imex channel**）。

| 能力 | `kimi26_train` | `--privileged` + 全套 device 的新容器 | 结论 |
|---|---|---|---|
| `NCCL_NVLS_ENABLE=1`（NVLS/SHARP multicast） | ✗ 无 `/dev/nvidia-nvswitch*` | ✗ **仍失败**：`Failed to bind NVLink SHARP (NVLS) Multicast memory … CUDA error 1 'invalid argument'`；换 NCCL 2.29.7 / 2.32.6 / CUDA 12.9 运行时、以及强制 `LD_PRELOAD=libcuda.so.570.211.01`（driver API 12080，无 compat 层）都一样 | **主机级**（FM/NVSwitch），换容器无解 |
| flash_comm fabric 句柄（`cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)`） | ✗ `operation not permitted` @ `shareable_block.cpp:90` | ✗ 同样失败 | **主机级**（缺 IMEX） |
| DeepEP GIN / IBGDA | ✗ 无 `/dev/infiniband`（`assert props.ginType != NONE`） | ✓ 可初始化（`num_allocated_qps=129`） | 换容器可开，但**单机 NVLink 路径无收益**：dispatch 134.8µs / combine 67.7µs，与 `EP_DISABLE_GIN=1` 的 113.7 / 67.5µs 同档（差异在抖动范围内） |
| `/dev/infiniband`（跨机 IB） | ✗ | ✓ | 只影响多机，本文全部单机 |
| 8 卡 P2P / NVLink | ✓ | ✓ | — |

另外注意：镜像 `dsv4-inf` 是 **CUDA 13.2**，而宿主驱动是 570（CUDA 12.8 时代），所以容器内实际加载的是 **forward-compat 的 `libcuda.so.595.58.03`**（`/usr/local/cuda/compat/lib.real/`）。这条 compat 层理论上会限制 multicast/fabric 类功能，但上表已用"强制宿主原生 libcuda"排除它——**真正的原因是 FM/NVSwitch 侧，不是 compat、也不是容器**。

**这些项对单机 EP=8 的测试结果影响 ≈ 0**（代码级核对，不是估计）：

| 项 | 单机 EP=8 的 op 里实际走什么 | 结论 |
|---|---|---|
| NCCL 集合通信（NVLS 的作用域） | **每个 op 内部一次都没有**：`ep_group_barrier()` 单机走 `_ep.barrier_all_on_stream()`（flash_comm 自己的 NVLink 对称内存 barrier）；dispatch/combine 是设备端 peer 写 + `ncclGetPeerPointer`；`DeepEP/csrc` 里 `ncclAllGather/ncclAllReduce/...` 出现 **0 次**。NCCL 只在 **init** 用到（`all_gather_object` 换 handle、`all_reduce(init_status)`、`dist.barrier`），不进 kernel 时间 | NVLS 缺失不影响 |
| IB / GIN | 单机全部走 NVLink P2P；实测开 GIN 后 dispatch 134.8µs / combine 67.7µs，与关掉时 113.7 / 67.5µs 同档 | 不影响（已验证） |
| fabric vs `cuda_ipc` | 只是 buffer 的导出/映射方式（`cuMemCreate(FABRIC)` vs `cudaIpcOpenMemHandle`），单机两者都映射到同一条 NVLink P2P 写路径；fabric 的真实用途是跨机/IMEX | 不影响 |
| `/dev/infiniband` | 只影响跨机 | 不影响 |

所以本文数字对"单机 8 卡 EP=8"这个场景是可用的；文档里保留"残血"说明只是为了解释为什么换成跨机或多机时这些通路会成为瓶颈。

如果要拿"满血"数字，需要在**宿主机**侧修两件事，然后重跑本文脚本即可（脚本无需改动）：

1. NVLS：检查/重启 `nvidia-fabricmanager`，确认 NVSwitch multicast 可用（`nvidia-smi -q | grep -A3 Fabric` 之外，直接跑 `NCCL_NVLS_ENABLE=1` 的 8 卡 allreduce 验证）；
2. fabric/IMEX：启动 `nvidia-imex` 并把 `/dev/nvidia-caps-imex-channels` 透传进容器，flash_comm 才能用 fabric 句柄替代 `cuda_ipc`。

**形状必须一档一进程**：早期把 4 档塞进同一个 torchrun 进程时，第 3 档（H=3072）的 DeepEP 数字整体劣化 10 倍（allocator 状态 + JIT/cuBLASLt 缓存），现在 `run_all.sh` 每档 `torchrun` 一次。

## 6. 附录 A：与上游 harness 的交叉验证

为了确认"自己写的 driver 没引入偏差"，用 FlashComm 自带的 `tests/ep_overlap/test_ep_overlap_group_gemm_combine.py`（`--num_tokens 1024 --hidden_size 1024 --intermediate_size 1024 --num_experts 32 --topk 16 --drop_ratio 0.1 --num_worst_tokens 0`）跑同一形状，对比 RANK 0 输出：

| 项 | 上游 harness（median） | 本文（kernel，17408 行） | 折算到同口径 |
|---|---|---|---|
| `group_gemm_only`（=FC2，14819 行、含 10% drop） | 54.0µs | 55.2µs | 行数多 17% 却只慢 2%，一致 |
| `combine_only`（push+barrier+reduce） | 122.5µs | 120.2µs | 一致 |
| `fused_kernel`（MK2 kernel，不含 barrier+reduce） | 136.9µs | — | 本文 MK2 含 barrier+reduce，不可直接比 |
| `fused_e2e`（MK2 含 barrier+reduce） | 169.2µs | 187.9µs | 行数多 17% 后一致（187.9/1.17 ≈ 161µs） |
| `sequential`（FC2+combine） | 176.9µs | 175.4µs（55.2+120.2） | 一致 |
| 融合收益 | 176.9 → 169.2（省 4%） | 175.4 → 187.9（慢 7%） | 都在 ±10% 内：**H=1024 时 MK2 基本不赚** |

结论：FC2 / combine / MK2 三条都对齐（差异可由 17% 行数解释），说明 driver 没有系统性偏差，也说明"MK2 在 H=1024 不赚钱"是库本身的特性、不是本文 driver 的问题。

上游 harness 还有两点值得记录：
- `test_ep_overlap_group_gemm_combine.py` 在 topk=16 下可以正常跑（§6 数据即来自它），说明 `group_gemm_combine` 支持 topk=16；
- 但 `test_ep_overlap_dispatch_group_gemm.py` 的 `cuda` baseline 走 `flash_comm.ep`（CUDA C++ kernel），其 `SUPPORTED_TOPK` 宏只列了 **6 和 8**，topk=16 会直接 `RuntimeError: Unsupported value for SUPPORTED_TOPK: 16`。本文 topk=16 的所有数据都走 CuTeDSL 路径（`flash_comm.ep_overlap`），CUDA 路径未参与。

## 7. 附录 B：正确性

`check_uniep_moe.py` 用"纯 torch 参考实现"（all-gather routing → 逐专家 gather+matmul → all-reduce 求和）对齐 uniEP 的融合/非融合输出：

| 配置 | e2e_sequential 最大绝对误差 | e2e_fused 最大绝对误差 | 相对误差 |
|---|---|---|---|
| T1024 H1024 | 1.43e-2 | 1.43e-2 | 4.96e-3 |
| T1024 H3072 | 7.97e-2 | 7.97e-2 | 4.73e-3 |

- 相对误差 ~5e-3 与 bf16 累加 16 项的预期一致（`mean|ref|` 分别 0.46 / 2.64）。
- **融合与非融合的结果完全逐位相同**（两者最大绝对误差一模一样，说明 MK1/MK2 与"dispatch+GEMM"、"GEMM+combine"是同一套数学），因此融合路径的加速不是"少算了"。

## 8. 附录 C：踩坑记录（本机特有）

| 现象 | 原因 | 处理 |
|---|---|---|
| `Failed to bind NVLink SHARP (NVLS) Multicast memory` | **主机级**：`--privileged`+全套 device 容器里同样失败（详见 §5.1），与容器无关 | `NCCL_NVLS_ENABLE=0`（唯一可用选项） |
| flash_comm 初始化 `CUDA Driver Error: operation not permitted at csrc/buffer/shareable_block.cpp:90` | `cuMemCreate(CU_MEM_HANDLE_TYPE_FABRIC)` 需要 IMEX：本机 `nvidia-imex` inactive 且无 imex channel（§5.1） | `FLASH_COMM_BUFFER_BACKEND=cuda_ipc` |
| `combine_cutedsl` 断言 `tile_n must be a positive multiple of num_threads_per_warp * atom_v` | 默认 `tile_n=128` 不满足 bf16 的 32×8 | 传 `tile_m=128, tile_n=256` |
| `deep_ep` 导入报 `Invalid NCCL versions: 2.29.7 (loaded) v.s. 2.32.3` | 系统 NCCL 太旧 | `LD_PRELOAD=/opt/nccl232/nvidia/nccl/lib/libnccl.so.2` |
| DeepEP `assert props.ginType != NONE`（`NCCL GIN is unavailable`） | 容器未挂 `/dev/infiniband`（换 `--privileged` 容器可初始化，但单机数字无变化，§5.1） | `EP_DISABLE_GIN=1`（只跑 intranode） |
| 第一个被测 op 的 kernel 时间比 wall 还大（839µs vs 115µs） | CUPTI attach 偏差 | `measure()` 先跑一个"丢弃"的 profiler session |
| TE `GroupedLinear` 反向报 "backward through the graph a second time" | 输入被复用 | 每次 bwd 前 `.detach().clone().requires_grad_()` |
| DeepEP combine 比正常慢 20×（4.1ms vs 0.18ms） | 把 dispatch 的输出 buffer 直接喂给 combine（与内部 recv buffer 别名） | combine 输入先 `.detach().clone()` |
| `flash_comm.ep_overlap` 的 `group_gemm` 报 `B must be k-major with strides (K, 1, N*K)` | 对 `permute(1,2,0)` 后又做了 `.contiguous()` | 只 permute，不要 contiguous |
| 同一进程内连跑 4 档 shape，第 3 档的 DeepEP 数字整体劣化 ~10 倍 | allocator 状态 + DeepJIT/cuBLASLt 缓存 | 每档单独 `torchrun`（`run_all.sh` 已如此） |
| 偶发单次调用 3–7ms（p90 是中位的 3–4 倍），且**整档**数据全被污染 | 共享机器上的主机调度停顿（其它容器），与 GPU 无关 | wall 取中位；`probe_quiet()` 在每档开测前用 8 卡 all_reduce 探活并等待；`measure_robust()` 发现 wall≫kernel 时重测 |
| DeepEP expand 布局的 `m_splits` 重建不出来（`psum` 差值 16532 vs `exp_x` 行数 16640，rank 0 对、其它 rank 错） | 有效行位于每个对齐块的**尾部**（`EPHandle` 的 `expert_token_idx_start = psum - unaligned` 语义），不是简单的 cumsum 对齐 | 放弃"实测串行 pipeline"，改报**分量相加**（§3.3 已标注为估算） |
