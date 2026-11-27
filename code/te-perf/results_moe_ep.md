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

- **反向：uniEP 没有 autograd，必须自己用 `group_gemm` 拼**（包里 `grep autograd/backward` 为空）。拼出来的 dgrad/wgrad GEMM 与前向基本 1:1（0.94–1.28×），但**周边开销很大**：EP 权重梯度 2 次 `all_reduce`（H3072 合计 1186µs）、无融合 SwiGLU 反向（1260µs）、dgrad/wgrad 需要的转置准备（1683µs）。一层 fwd+bwd：**uniEP 拼装 8232µs vs TE autograd 4578µs**（T1024/H3072，均不含转置准备；uniEP 含跨卡 reduce、TE 不含）；但 **`T128/H1024` 上 wall 反超 2.9×**（447 vs 1304µs，TE 的 autograd 被 Python 主机开销拖死；kernel 仍是 TE 小：165 vs 445），`T128/H3072` 两边 wall 接近（2177 vs 1339，TE 略优）。详见 §3.4 / §3.5 / §4.5。
- **group GEMM：uniEP 的 CuTeDSL grouped GEMM 全面快于 TE 的 `GroupedLinear`**，按"每行耗时"归一（两家行数不同，见 §1）后：`T1024` 档 **1.19–1.41×**，`T128` 档 **1.35–2.06×**。折算单卡有效算力 uniEP **418–799 TFLOPS**（H100 bf16 dense 峰值 989.4 的 42%–81%）、TE **211–629 TFLOPS**（最差是 `T128/H1024` 的 FC2，只有峰值的 21%）。
- **通信：DeepEP 的 combine 明显更快，dispatch 同档略优**。同为 per-(token,expert) 展开语义（`T1024`）：combine **68.7µs vs 120.2µs**（H1024）、**174.3µs vs 401.3µs**（H3072），快 **1.75–2.3×**；dispatch **120.0µs vs 139.9µs**、**339.1µs vs 437.1µs**，快 **1.2–1.3×**。`T128` 档两边都被 ~60–90µs 的 launch/host 开销吃掉，wall 上打平（79.4 vs 85.2、91.2 vs 84.7）。
- **uniEP 的 MegaKernel 只在"大 batch + 大 H"赚钱**：`T1024/H3072` 下 MK1（dispatch+FC1 融合）**1138.9µs vs 非融合 1317.9µs（快 14%）**，MK2（FC2+combine 融合）**766.6µs vs 830.6µs（快 8%）**，整层融合 e2e **2186µs vs 2851µs（快 23%）**。
- **小 shape 上融合是负收益**：`T1024/H1024` MK1 **344.9 vs 243.9µs（慢 41%）**，整层融合与非融合打平（597.5 vs 597.6µs），但 wall 反而慢 32%（744.9 vs 565.8µs）；`T128` 档 MK1 慢 **1.5–2.4×**，整层 `T128/H1024` 慢 30%。原因见 §4.3：MK1 有 ~100µs 级固定代价、MK2 约 10µs，且融合后尾部无法提前退出（持久化 132 block、专家信号初始化、尾部 tile 不能提前退出），而小 shape 下 dispatch/GEMM 本身只有 25–130µs。
- **最佳组合是"混搭"而非单栈**：DeepEP 通信 + uniEP GEMM，`T1024/H3072` 一层 MoE ≈ **2158µs**（分量相加），比 uniEP 全融合（2186µs）快 1.3%，比 DeepEP 通信 + TE GEMM（2356µs）快 8%；`T1024/H1024` 混搭 **448µs** 对 uniEP 非融合 597.6µs，快 **25%**。
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
- 反向用**同一份 routing / 同一套行切分**（`bench_uniep_bwd.py`）：`recv_expert_counts` 是**未对齐**的每专家计数，
  实际的 dispatch 行数是把它按 256 对齐后的和（对齐填充行经实测**全为零**，所以 wgrad 的 gather-K 可以直接用对齐块）。
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
| `te_grouped_gemm_fc1` | T1024/H3072 | 839.5 | 992.1 | **−152.6** | 见下面"为什么 kernel 会大于 wall" |
| `uniep_e2e_sequential` | T1024/H3072 | 2464.3 | 2850.9 | **−386.6** | 同上（5 个 kernel 的链） |

**为什么 `kernel` 有时反而大于 `wall`**（−5%～−15%），两个原因，都不是测错：

1. **"逐 kernel 中位求和" ≠ "整段中位"**：每个 kernel 的中位在 50 次里独立取；若 A 在某些迭代慢、B 在另一些迭代慢（主机抖动），`sum(median_k)` 会大于 `median(sum_k)`；
2. **跨迭代重叠**：50 次调用背靠背入队（中间不 `synchronize`），第 i 次 combine 的尾巴常与第 i+1 次 dispatch 的头重叠，于是 `wall`（单次 event 间隔）比"把每个 kernel 时长相加"更短。

**实践上怎么选**：

- 比"同等的活谁算得快"（算法/框架效率、TFLOPS）→ 用 **kernel**：剔除 launch 与主机差异，本文 §3.1 / §4.1 就是这个口径；
- 判断"这层在训练关键路径上要等多久"、判断融合是否划算 → 用 **wall**：它含 launch 与跨卡同步，§4.3 的 MegaKernel 盈亏主要看它；
- 两者差距 <15% 说明这层没有 host/launch 瓶颈；差距很大（如 TE 在 T128 差 143µs）说明**该优化 launch 次数或主机路径，而不是 kernel 本身**。

### 3.1 group GEMM（FC1 = X@W1ᵀ: N=2I, K=H；FC2 = A@W2ᵀ: N=H, K=I）

| 配置 | 行数 uniEP / TE | FC1 uniEP (µs) | FC1 TE (µs) | FC2 uniEP (µs) | FC2 TE (µs) |
|---|---|---:|---:|---:|---:|
| T1024/rank H1024 | 17408 / 16531 | 104.0 | 139.5 | 55.2 | 67.3 |
| T1024/rank H3072 | 17408 / 16531 | 880.8 | 992.1 | 429.3 | 516.2 |
| T128/rank H1024 | 2816 / 2116 | 26.1 | 26.4 | 14.1 | 21.1 |
| T128/rank H3072 | 2816 / 2116 | 133.8 | 201.8 | 66.5 | 103.0 |

### 3.2 dispatch / combine（通信 op：上=wall 中位 µs，下=CUPTI kernel µs）

| 配置 | dispatch uniEP | dispatch DeepEP(expanded) | dispatch DeepEP(expanded, async) | combine uniEP | combine DeepEP(expanded) |
|---|---:|---:|---:|---:|---:|
| T1024/rank H1024 | 148.7<br>139.9 | 121.1<br>120.0 | 119.9<br>113.0 | 128.1<br>120.2 | 73.3<br>68.7 |
| T1024/rank H3072 | 452.5<br>437.1 | 320.8<br>339.1 | 323.0<br>314.7 | 409.8<br>401.3 | 179.6<br>174.3 |
| T128/rank H1024 | 85.2<br>50.8 | 79.4<br>25.0 | 93.8<br>61.1 | 61.7<br>36.4 | 48.7<br>29.3 |
| T128/rank H3072 | 84.7<br>56.6 | 91.2<br>46.7 | 95.1<br>68.2 | 63.8<br>57.7 | 47.7<br>40.0 |

### 3.3 端到端一层 MoE（单位 µs；上=wall 中位，下=CUPTI kernel）

> 「估算」两列是**逐项实测值相加**（DeepEP 通信 + 对应 GEMM + SwiGLU），不是一次调用实测：本机的 DeepEP expand 布局无法可靠重建 `m_splits`（见 §8），故不报串行 pipeline 的整段实测值。

| 配置 | uniEP 融合 (MK1+SwiGLU+MK2) | uniEP 非融合 (5 次 launch) | 估算: DeepEP 通信 + TE GEMM | 估算: DeepEP 通信 + uniEP GEMM | DeepEP 纯通信 (expanded) |
|---|---:|---:|---:|---:|---:|
| T1024/rank H1024 | 744.9<br>597.5 | 565.8<br>597.6 | 496 | 448 | 213.5<br>211.9 |
| T1024/rank H3072 | 2396.7<br>2186.0 | 2464.3<br>2850.9 | 2356 | 2158 | 572.2<br>585.7 |
| T128/rank H1024 | 417.7<br>260.6 | 227.8<br>199.7 | 154 | 147 | 124.6<br>53.3 |
| T128/rank H3072 | 589.8<br>437.2 | 401.1<br>458.9 | 467 | 363 | 142.8<br>98.0 |

### 3.4 uniEP 反向：没有 autograd，用 group_gemm 拼出来（µs；上=wall 中位，下=CUPTI kernel）

> uniEP 只提供前向 kernel（`dispatch` / `group_gemm` / `combine` / 两个 MegaKernel），**包里没有任何 `autograd.Function`**。下表是把反向拆成同样 primitive 后逐项实测的结果，数值正确性已对齐 torch autograd（§7.1：dW1 rel 3.3e-3、dW2 逐位相同、dA 3.8e-3）。

> 计时口径：dgrad/wgrad 的输入梯度用**随机张量**（与链路里真实的 `dy1` 同形状同分布，只影响计时、不影响正确性；链路正确性由 §7.1 的 full-chain 检查覆盖）；wgrad 的 `K` 按 `max(rows_e)` 补齐到对齐边界，补齐区间在 A 侧为零（实测），因此结果与不补齐等价。

| 环节 | T1024 H1024 | T1024 H3072 | T128 H1024 | T128 H3072 |
|---|---:|---:|---:|---:|
| FC1 前向（参照） | 114.8<br>104.0 | 854.0<br>880.8 | 41.2<br>26.1 | 140.1<br>133.8 |
| FC1 dgrad（dY@W1） | 184.7<br>101.5 | 857.0<br>829.4 | 35.8<br>23.8 | 140.7<br>132.9 |
| FC1 wgrad GEMM（dYᵀ@A） | 99.4<br>94.5 | 883.3<br>908.9 | 35.5<br>22.0 | 177.6<br>170.8 |
| FC1 dW all_reduce | 126.6<br>123.3 | 798.1<br>784.8 | 126.1<br>122.1 | 779.4<br>774.2 |
| FC1 fwd+bwd 合计 | 459.3<br>447.9 | 3483.6<br>3468.6 | 215.8<br>202.5 | 1237.5<br>1211.6 |
| FC2 前向（参照） | 66.0<br>55.2 | 439.5<br>429.3 | 38.6<br>14.1 | 71.4<br>66.5 |
| FC2 dgrad（dO@W2） | 66.7<br>55.7 | 445.4<br>420.8 | 33.2<br>14.0 | 74.0<br>68.7 |
| FC2 wgrad GEMM（dOᵀ@A2） | 54.7<br>50.7 | 515.2<br>460.6 | 31.4<br>12.5 | 93.2<br>86.9 |
| FC2 dW all_reduce | 80.9<br>75.9 | 406.1<br>400.9 | 81.0<br>75.8 | 403.4<br>401.6 |
| FC2 fwd+bwd 合计 | 265.9<br>256.1 | 1808.6<br>1768.0 | 134.3<br>126.0 | 644.5<br>627.9 |
| SwiGLU 反向（torch） | 440.4<br>412.1 | 1299.6<br>1260.1 | 107.0<br>61.8 | 228.5<br>187.1 |
| 转置准备（每步 2 次） | 754.2<br>560.2 | 2188.4<br>1683.0 | 262.7<br>106.2 | 374.0<br>270.8 |
| 一层 fwd+bwd（不含转置） | 1277.3<br>1356.9 | 6869.4<br>8232.2 | 447.1<br>445.0 | 2177.3<br>2609.5 |
| 一层 fwd+bwd（含转置） | 1880.1<br>1807.0 | 8519.8<br>9421.9 | 608.4<br>566.5 | 2473.0<br>2836.2 |

### 3.5 反向对比：uniEP 手工拼装 vs TE autograd（µs；上=wall 中位，下=CUPTI kernel）

> 口径差异（重要）：uniEP 两列**包含**EP 权重梯度的 2 次 `all_reduce`、**不含**转置准备（另见 §3.5 最后两行 / §4.6）；TE 是纯本地 `GroupedLinear` autograd，**不含**任何跨卡通信（TE 侧没有 EP 通信层，与 §3.3 前向口径一致）。

| 配置 | FC1 fwd+bwd uniEP | FC1 fwd+bwd TE | FC2 fwd+bwd uniEP | FC2 fwd+bwd TE | 一层 fwd+bwd uniEP | 一层 fwd+bwd TE |
|---|---:|---:|---:|---:|---:|---:|
| T1024/rank H1024 | 459.3<br>447.9 | 642.7<br>346.3 | 265.9<br>256.1 | 806.8<br>271.4 | 1277.3<br>1356.9 | 1340.4<br>914.8 |
| T1024/rank H3072 | 3483.6<br>3468.6 | 1804.5<br>2086.7 | 1808.6<br>1768.0 | 1380.9<br>1711.1 | 6869.4<br>8232.2 | 3969.6<br>4577.9 |
| T128/rank H1024 | 215.8<br>202.5 | 641.6<br>58.1 | 134.3<br>126.0 | 793.1<br>72.2 | 447.1<br>445.0 | 1303.9<br>165.0 |
| T128/rank H3072 | 1237.5<br>1211.6 | 657.1<br>382.3 | 644.5<br>627.9 | 804.1<br>316.2 | 2177.3<br>2609.5 | 1339.4<br>810.5 |

### 3.6 全量明细（原始口径：`kernel耗时` = CUPTI 纯 device kernel；`wall中位` / `p90` 见 §0 提示 2）

| op | dtype | shape | device | kernel耗时(µs) | wall中位(µs) | wall p90(µs) | 备注 |
|---|---|---|---|---:|---:|---:|---|
| **uniEP** | | | | | | | |
| uniep_layout | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 62.6 | 57.1 | 61.6 | layout only (histogram + cumsum + allgather + layout kernel) |
| uniep_layout | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 63.2 | 61.4 | 68.4 | layout only (histogram + cumsum + allgather + layout kernel) |
| uniep_layout | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 62.9 | 59.5 | 65.2 | layout only (histogram + cumsum + allgather + layout kernel) |
| uniep_layout | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 63.9 | 59.8 | 65.3 | layout only (histogram + cumsum + allgather + layout kernel) |
| uniep_dispatch | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 139.9 | 148.7 | 168.5 | recv_rows=17408 (unaligned sum=16531, align=256) local_m=[4129, 4112, 4135, 4155] |
| uniep_dispatch | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 437.1 | 452.5 | 505.3 | recv_rows=17408 (unaligned sum=16531, align=256) local_m=[4129, 4112, 4135, 4155] |
| uniep_dispatch | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 50.8 | 85.2 | 93.7 | recv_rows=2816 (unaligned sum=2116, align=256) local_m=[511, 525, 552, 528] |
| uniep_dispatch | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 56.6 | 84.7 | 91.6 | recv_rows=2816 (unaligned sum=2116, align=256) local_m=[511, 525, 552, 528] |
| uniep_group_gemm_fc1 | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 104.0 | 114.8 | 115.7 | N=2I=2048, K=1024, rows=17408 |
| uniep_group_gemm_fc1 | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 880.8 | 854.0 | 972.3 | N=2I=6144, K=3072, rows=17408 |
| uniep_group_gemm_fc1 | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 26.1 | 41.2 | 47.2 | N=2I=2048, K=1024, rows=2816 |
| uniep_group_gemm_fc1 | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 133.8 | 140.1 | 199.2 | N=2I=6144, K=3072, rows=2816 |
| uniep_swiglu | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 107.1 | 113.2 | 113.7 | rows=17408, I=1024 |
| uniep_swiglu | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 358.3 | 412.4 | 413.2 | rows=17408, I=3072 |
| uniep_swiglu | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 16.3 | 41.2 | 42.1 | rows=2816, I=1024 |
| uniep_swiglu | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 54.6 | 65.4 | 66.2 | rows=2816, I=3072 |
| uniep_group_gemm_fc2 | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 55.2 | 66.0 | 103.1 | N=H=1024, K=I=1024, rows=17408 |
| uniep_group_gemm_fc2 | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 429.3 | 439.5 | 443.9 | N=H=3072, K=I=3072, rows=17408 |
| uniep_group_gemm_fc2 | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 14.1 | 38.6 | 41.9 | N=H=1024, K=I=1024, rows=2816 |
| uniep_group_gemm_fc2 | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 66.5 | 71.4 | 72.6 | N=H=3072, K=I=3072, rows=2816 |
| uniep_combine | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 120.2 | 128.1 | 130.9 | push_mode=tile (push + NVL barrier + topk reduce) |
| uniep_combine | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 401.3 | 409.8 | 419.6 | push_mode=tile (push + NVL barrier + topk reduce) |
| uniep_combine | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 36.4 | 61.7 | 70.3 | push_mode=tile (push + NVL barrier + topk reduce) |
| uniep_combine | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 57.7 | 63.8 | 68.0 | push_mode=tile (push + NVL barrier + topk reduce) |
| uniep_dispatch_group_gemm | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 344.9 | 440.4 | 515.4 | MegaKernel 1: dispatch + FC1 overlap |
| uniep_dispatch_group_gemm | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1138.9 | 1260.5 | 1327.5 | MegaKernel 1: dispatch + FC1 overlap |
| uniep_dispatch_group_gemm | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 184.6 | 304.9 | 355.2 | MegaKernel 1: dispatch + FC1 overlap |
| uniep_dispatch_group_gemm | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 288.1 | 402.3 | 422.0 | MegaKernel 1: dispatch + FC1 overlap |
| uniep_group_gemm_combine | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 187.9 | 194.1 | 195.9 | MegaKernel 2: FC2 + combine + reduce |
| uniep_group_gemm_combine | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 766.6 | 773.6 | 779.9 | MegaKernel 2: FC2 + combine + reduce |
| uniep_group_gemm_combine | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 55.4 | 84.2 | 90.3 | MegaKernel 2: FC2 + combine + reduce |
| uniep_group_gemm_combine | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 132.3 | 139.6 | 141.9 | MegaKernel 2: FC2 + combine + reduce |
| uniep_e2e_fused | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 597.5 | 744.9 | 767.9 | MK1 + swiglu + MK2 (2 launches) |
| uniep_e2e_fused | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 2186.0 | 2396.7 | 2433.8 | MK1 + swiglu + MK2 (2 launches) |
| uniep_e2e_fused | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 260.6 | 417.7 | 431.7 | MK1 + swiglu + MK2 (2 launches) |
| uniep_e2e_fused | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 437.2 | 589.8 | 604.7 | MK1 + swiglu + MK2 (2 launches) |
| uniep_e2e_sequential | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 597.6 | 565.8 | 571.6 | dispatch+FC1+swiglu+FC2+combine (5 launches) |
| uniep_e2e_sequential | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 2850.9 | 2464.3 | 2488.4 | dispatch+FC1+swiglu+FC2+combine (5 launches) |
| uniep_e2e_sequential | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 199.7 | 227.8 | 234.4 | dispatch+FC1+swiglu+FC2+combine (5 launches) |
| uniep_e2e_sequential | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 458.9 | 401.1 | 404.4 | dispatch+FC1+swiglu+FC2+combine (5 launches) |
| **uniEP 反向** | | | | | | | |
| uniep_fc1_dgrad | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 101.5 | 184.7 | 185.6 | dY1@W1: N=H=1024, K=2I=2048, rows=17408 |
| uniep_fc1_dgrad | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 829.4 | 857.0 | 1458.4 | dY1@W1: N=H=3072, K=2I=6144, rows=17408 |
| uniep_fc1_dgrad | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 23.8 | 35.8 | 39.5 | dY1@W1: N=H=1024, K=2I=2048, rows=2816 |
| uniep_fc1_dgrad | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 132.9 | 140.7 | 141.6 | dY1@W1: N=H=3072, K=2I=6144, rows=2816 |
| uniep_fc1_wgrad_gemm | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 94.5 | 99.4 | 100.1 | 仅 dY1^T@A: M=2I=2048, N=H=1024, K=4352, E=4 |
| uniep_fc1_wgrad_gemm | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 908.9 | 883.3 | 1539.3 | 仅 dY1^T@A: M=2I=6144, N=H=3072, K=4352, E=4 |
| uniep_fc1_wgrad_gemm | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 22.0 | 35.5 | 40.9 | 仅 dY1^T@A: M=2I=2048, N=H=1024, K=768, E=4 |
| uniep_fc1_wgrad_gemm | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 170.8 | 177.6 | 320.5 | 仅 dY1^T@A: M=2I=6144, N=H=3072, K=768, E=4 |
| uniep_fc1_allreduce | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 123.3 | 126.6 | 128.2 | dW1 EP all_reduce 17MB |
| uniep_fc1_allreduce | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 784.8 | 798.1 | 807.9 | dW1 EP all_reduce 151MB |
| uniep_fc1_allreduce | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 122.1 | 126.1 | 128.0 | dW1 EP all_reduce 17MB |
| uniep_fc1_allreduce | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 774.2 | 779.4 | 786.8 | dW1 EP all_reduce 151MB |
| uniep_fc1_wgrad | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 226.2 | 233.1 | 235.8 | dY1^T@A (K padded 4352) + all_reduce 17MB |
| uniep_fc1_wgrad | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1648.6 | 1658.7 | 1667.1 | dY1^T@A (K padded 4352) + all_reduce 151MB |
| uniep_fc1_wgrad | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 149.4 | 149.5 | 151.2 | dY1^T@A (K padded 768) + all_reduce 17MB |
| uniep_fc1_wgrad | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 950.8 | 959.7 | 976.2 | dY1^T@A (K padded 768) + all_reduce 151MB |
| uniep_fc2_dgrad | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 55.7 | 66.7 | 104.1 | dOUT@W2: N=I=1024, K=H=1024, rows=17408 |
| uniep_fc2_dgrad | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 420.8 | 445.4 | 762.6 | dOUT@W2: N=I=3072, K=H=3072, rows=17408 |
| uniep_fc2_dgrad | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 14.0 | 33.2 | 38.2 | dOUT@W2: N=I=1024, K=H=1024, rows=2816 |
| uniep_fc2_dgrad | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 68.7 | 74.0 | 74.8 | dOUT@W2: N=I=3072, K=H=3072, rows=2816 |
| uniep_fc2_wgrad_gemm | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 50.7 | 54.7 | 55.9 | 仅 dOUT^T@A2: M=H=1024, N=I=1024, K=4352, E=4 |
| uniep_fc2_wgrad_gemm | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 460.6 | 515.2 | 850.1 | 仅 dOUT^T@A2: M=H=3072, N=I=3072, K=4352, E=4 |
| uniep_fc2_wgrad_gemm | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 12.5 | 31.4 | 34.0 | 仅 dOUT^T@A2: M=H=1024, N=I=1024, K=768, E=4 |
| uniep_fc2_wgrad_gemm | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 86.9 | 93.2 | 159.0 | 仅 dOUT^T@A2: M=H=3072, N=I=3072, K=768, E=4 |
| uniep_fc2_allreduce | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 75.9 | 80.9 | 82.0 | dW2 EP all_reduce 8MB |
| uniep_fc2_allreduce | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 400.9 | 406.1 | 415.6 | dW2 EP all_reduce 75MB |
| uniep_fc2_allreduce | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 75.8 | 81.0 | 82.4 | dW2 EP all_reduce 8MB |
| uniep_fc2_allreduce | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 401.6 | 403.4 | 409.6 | dW2 EP all_reduce 75MB |
| uniep_fc2_wgrad | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 133.2 | 140.0 | 141.9 | dOUT^T@A2 (K padded 4352) + all_reduce 8MB |
| uniep_fc2_wgrad | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 856.9 | 863.7 | 874.7 | dOUT^T@A2 (K padded 4352) + all_reduce 75MB |
| uniep_fc2_wgrad | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 90.3 | 96.6 | 98.1 | dOUT^T@A2 (K padded 768) + all_reduce 8MB |
| uniep_fc2_wgrad | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 492.9 | 501.1 | 505.3 | dOUT^T@A2 (K padded 768) + all_reduce 75MB |
| uniep_swiglu_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 412.1 | 440.4 | 441.7 | silu(g)*u 反向: 3 个 torch kernel (dg, du, cat) |
| uniep_swiglu_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1260.1 | 1299.6 | 1300.4 | silu(g)*u 反向: 3 个 torch kernel (dg, du, cat) |
| uniep_swiglu_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 61.8 | 107.0 | 115.1 | silu(g)*u 反向: 3 个 torch kernel (dg, du, cat) |
| uniep_swiglu_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 187.1 | 228.5 | 229.8 | silu(g)*u 反向: 3 个 torch kernel (dg, du, cat) |
| uniep_transpose_prep | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 560.2 | 754.2 | 755.7 | 4 个转置拷贝 (dY1^T, A^T ×2, dOUT^T) -> K padded 4352 |
| uniep_transpose_prep | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1683.0 | 2188.4 | 2189.9 | 4 个转置拷贝 (dY1^T, A^T ×2, dOUT^T) -> K padded 4352 |
| uniep_transpose_prep | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 106.2 | 262.7 | 268.5 | 4 个转置拷贝 (dY1^T, A^T ×2, dOUT^T) -> K padded 768 |
| uniep_transpose_prep | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 270.8 | 374.0 | 375.0 | 4 个转置拷贝 (dY1^T, A^T ×2, dOUT^T) -> K padded 768 |
| uniep_wgrad_allreduce | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 230.7 | 210.4 | 212.5 | 2 次 EP all_reduce: 25MB |
| uniep_wgrad_allreduce | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1527.1 | 1188.1 | 1200.4 | 2 次 EP all_reduce: 226MB |
| uniep_wgrad_allreduce | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 228.7 | 211.6 | 214.5 | 2 次 EP all_reduce: 25MB |
| uniep_wgrad_allreduce | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1525.1 | 1187.6 | 1198.8 | 2 次 EP all_reduce: 226MB |
| uniep_fc1_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 447.9 | 459.3 | 461.7 | FC1 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc1_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 3468.6 | 3483.6 | 3506.7 | FC1 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc1_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 202.5 | 215.8 | 218.9 | FC1 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc1_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1211.6 | 1237.5 | 1244.2 | FC1 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc2_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 256.1 | 265.9 | 268.8 | FC2 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc2_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1768.0 | 1808.6 | 1819.5 | FC2 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc2_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 126.0 | 134.3 | 170.1 | FC2 fwd + dgrad + wgrad(+all_reduce) |
| uniep_fc2_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 627.9 | 644.5 | 649.4 | FC2 fwd + dgrad + wgrad(+all_reduce) |
| uniep_layer_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 1356.9 | 1277.3 | 1280.3 | FC1 fwd+SwiGLU+FC2 fwd + FC2 bwd+SwiGLU bwd+FC1 bwd (2×EP all_reduce) |
| uniep_layer_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 8232.2 | 6869.4 | 6889.4 | FC1 fwd+SwiGLU+FC2 fwd + FC2 bwd+SwiGLU bwd+FC1 bwd (2×EP all_reduce) |
| uniep_layer_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 445.0 | 447.1 | 452.4 | FC1 fwd+SwiGLU+FC2 fwd + FC2 bwd+SwiGLU bwd+FC1 bwd (2×EP all_reduce) |
| uniep_layer_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 2609.5 | 2177.3 | 2190.2 | FC1 fwd+SwiGLU+FC2 fwd + FC2 bwd+SwiGLU bwd+FC1 bwd (2×EP all_reduce) |
| uniep_layer_fwd_bwd_full | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 1807.0 | 1880.1 | 1883.6 | 同上 + 每步 dy1^T / a2^T 转置准备（真实训练必须付这部分） |
| uniep_layer_fwd_bwd_full | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 9421.9 | 8519.8 | 8550.3 | 同上 + 每步 dy1^T / a2^T 转置准备（真实训练必须付这部分） |
| uniep_layer_fwd_bwd_full | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 566.5 | 608.4 | 629.1 | 同上 + 每步 dy1^T / a2^T 转置准备（真实训练必须付这部分） |
| uniep_layer_fwd_bwd_full | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 2836.2 | 2473.0 | 2487.4 | 同上 + 每步 dy1^T / a2^T 转置准备（真实训练必须付这部分） |
| **DeepEP** | | | | | | | |
| deepep_dispatch | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 113.7 | 109.5 | 111.4 | recv_rows=7807 |
| deepep_dispatch | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 315.2 | 297.1 | 301.2 | recv_rows=7807 |
| deepep_dispatch | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 24.5 | 65.5 | 71.0 | recv_rows=979 |
| deepep_dispatch | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 46.5 | 78.5 | 84.9 | recv_rows=979 |
| deepep_dispatch_expanded | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 120.0 | 121.1 | 129.1 | expanded_rows=16896 align=128 local_m=[4129, 4112, 4135, 4155] |
| deepep_dispatch_expanded | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 339.1 | 320.8 | 325.0 | expanded_rows=16896 align=128 local_m=[4129, 4112, 4135, 4155] |
| deepep_dispatch_expanded | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 25.0 | 79.4 | 85.0 | expanded_rows=2432 align=128 local_m=[511, 525, 552, 528] |
| deepep_dispatch_expanded | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 46.7 | 91.2 | 96.9 | expanded_rows=2432 align=128 local_m=[511, 525, 552, 528] |
| deepep_dispatch_async | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 105.1 | 111.6 | 113.1 | recv_rows=7807 async+defer, no host sync |
| deepep_dispatch_async | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 291.7 | 300.0 | 303.6 | recv_rows=7807 async+defer, no host sync |
| deepep_dispatch_async | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 61.8 | 96.9 | 98.8 | recv_rows=979 async+defer, no host sync |
| deepep_dispatch_async | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 74.0 | 99.7 | 105.2 | recv_rows=979 async+defer, no host sync |
| deepep_dispatch_expanded_async | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 113.0 | 119.9 | 121.9 | expanded_rows=16896 async+defer, no host sync |
| deepep_dispatch_expanded_async | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 314.7 | 323.0 | 328.5 | expanded_rows=16896 async+defer, no host sync |
| deepep_dispatch_expanded_async | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 61.1 | 93.8 | 96.7 | expanded_rows=2432 async+defer, no host sync |
| deepep_dispatch_expanded_async | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 68.2 | 95.1 | 101.7 | expanded_rows=2432 async+defer, no host sync |
| deepep_combine | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 67.5 | 71.9 | 72.7 | in_rows=7807, topk_weights reduction |
| deepep_combine | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 180.4 | 185.6 | 187.7 | in_rows=7807, topk_weights reduction |
| deepep_combine | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 31.1 | 50.8 | 55.0 | in_rows=979, topk_weights reduction |
| deepep_combine | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 36.5 | 48.7 | 52.1 | in_rows=979, topk_weights reduction |
| deepep_combine_expanded | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 68.7 | 73.3 | 74.1 | in_rows=16896, expand-mode reduce (training path) |
| deepep_combine_expanded | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 174.3 | 179.6 | 181.2 | in_rows=16896, expand-mode reduce (training path) |
| deepep_combine_expanded | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 29.3 | 48.7 | 51.1 | in_rows=2432, expand-mode reduce (training path) |
| deepep_combine_expanded | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 40.0 | 47.7 | 52.0 | in_rows=2432, expand-mode reduce (training path) |
| deepep_e2e | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 191.2 | 191.9 | 193.6 | dispatch + combine (comm only, no expert GEMM) |
| deepep_e2e | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 540.6 | 526.1 | 530.8 | dispatch + combine (comm only, no expert GEMM) |
| deepep_e2e | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 54.6 | 133.7 | 143.9 | dispatch + combine (comm only, no expert GEMM) |
| deepep_e2e | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 92.6 | 151.8 | 156.0 | dispatch + combine (comm only, no expert GEMM) |
| deepep_e2e_expanded | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 211.9 | 213.5 | 215.9 | expanded dispatch + combine (matches uniEP e2e comm semantics) |
| deepep_e2e_expanded | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 585.7 | 572.2 | 577.0 | expanded dispatch + combine (matches uniEP e2e comm semantics) |
| deepep_e2e_expanded | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 53.3 | 124.6 | 135.1 | expanded dispatch + combine (matches uniEP e2e comm semantics) |
| deepep_e2e_expanded | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 98.0 | 142.8 | 154.2 | expanded dispatch + combine (matches uniEP e2e comm semantics) |
| **TE group-gemm** | | | | | | | |
| te_grouped_gemm_fc1 | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 139.5 | 181.6 | 204.7 | N=2I=2048, K=1024, rows=16531 |
| te_grouped_gemm_fc1 | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 992.1 | 839.5 | 862.6 | N=2I=6144, K=3072, rows=16531 |
| te_grouped_gemm_fc1 | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 26.4 | 172.5 | 185.9 | N=2I=2048, K=1024, rows=2116 |
| te_grouped_gemm_fc1 | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 201.8 | 186.5 | 198.9 | N=2I=6144, K=3072, rows=2116 |
| te_grouped_gemm_fc2 | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 67.3 | 177.4 | 201.7 | N=H=1024, K=I=1024, rows=16531 |
| te_grouped_gemm_fc2 | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 516.2 | 425.8 | 432.7 | N=H=3072, K=I=3072, rows=16531 |
| te_grouped_gemm_fc2 | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 21.1 | 171.9 | 183.5 | N=H=1024, K=I=1024, rows=2116 |
| te_grouped_gemm_fc2 | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 103.0 | 172.9 | 182.4 | N=H=3072, K=I=3072, rows=2116 |
| te_grouped_gemm_layer | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 336.7 | 390.9 | 425.6 | FC1 + swiglu + FC2 (forward) |
| te_grouped_gemm_layer | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1815.3 | 1569.0 | 1690.8 | FC1 + swiglu + FC2 (forward) |
| te_grouped_gemm_layer | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 65.9 | 385.0 | 420.3 | FC1 + swiglu + FC2 (forward) |
| te_grouped_gemm_layer | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 353.8 | 390.8 | 420.0 | FC1 + swiglu + FC2 (forward) |
| te_grouped_gemm_fc1_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 346.3 | 642.7 | 687.3 | fwd+bwd (grad wrt input+weight), N=2I=2048 |
| te_grouped_gemm_fc1_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 2086.7 | 1804.5 | 1888.4 | fwd+bwd (grad wrt input+weight), N=2I=6144 |
| te_grouped_gemm_fc1_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 58.1 | 641.6 | 667.2 | fwd+bwd (grad wrt input+weight), N=2I=2048 |
| te_grouped_gemm_fc1_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 382.3 | 657.1 | 682.6 | fwd+bwd (grad wrt input+weight), N=2I=6144 |
| te_grouped_gemm_fc2_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 271.4 | 806.8 | 839.2 | fwd+bwd, N=H=1024, K=I=1024 |
| te_grouped_gemm_fc2_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 1711.1 | 1380.9 | 1491.4 | fwd+bwd, N=H=3072, K=I=3072 |
| te_grouped_gemm_fc2_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 72.2 | 793.1 | 833.3 | fwd+bwd, N=H=1024, K=I=1024 |
| te_grouped_gemm_fc2_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 316.2 | 804.1 | 861.6 | fwd+bwd, N=H=3072, K=I=3072 |
| te_grouped_gemm_layer_fwd_bwd | torch.bfloat16 | (T1024/rank, H1024, I1024, topk16, E32, EP8) | H100 | 914.8 | 1340.4 | 1371.1 | FC1 + swiglu + FC2 fwd+bwd (autograd; weight grads 本地, 不含跨卡 reduce) |
| te_grouped_gemm_layer_fwd_bwd | torch.bfloat16 | (T1024/rank, H3072, I3072, topk16, E32, EP8) | H100 | 4577.9 | 3969.6 | 4081.3 | FC1 + swiglu + FC2 fwd+bwd (autograd; weight grads 本地, 不含跨卡 reduce) |
| te_grouped_gemm_layer_fwd_bwd | torch.bfloat16 | (T128/rank, H1024, I1024, topk16, E32, EP8) | H100 | 165.0 | 1303.9 | 1353.2 | FC1 + swiglu + FC2 fwd+bwd (autograd; weight grads 本地, 不含跨卡 reduce) |
| te_grouped_gemm_layer_fwd_bwd | torch.bfloat16 | (T128/rank, H3072, I3072, topk16, E32, EP8) | H100 | 810.5 | 1339.4 | 1389.2 | FC1 + swiglu + FC2 fwd+bwd (autograd; weight grads 本地, 不含跨卡 reduce) |

## 4. 分析

### 4.1 group GEMM：uniEP（CuTeDSL）vs TE（`GroupedLinear`）

按"每行耗时的 flop 归一"比较（行数不同，必须归一）：

| 配置 | FC1 有效算力 uniEP | FC1 有效算力 TE | FC2 有效算力 uniEP | FC2 有效算力 TE |
|---|---|---|---|---|
| T1024/rank H1024 | 702 | 497 | 662 | 515 |
| T1024/rank H3072 | 746 | 629 | 765 | 604 |
| T128/rank H1024 | 452 | 336 | 418 | 211 |
| T128/rank H3072 | 794 | 396 | 799 | 388 |

- uniEP 的 grouped GEMM 是手写 CuTeDSL（tile 128×256、warp-specialized、per-expert 变长 M），**最小档（2816 行）也有 418–452 TFLOPS，最大档 799 TFLOPS（峰值的 81%）**；TE 走的 cuBLASLt grouped 路径在小 m 段掉得厉害（`T128/H1024` FC2 仅 211 TFLOPS、`T128/H3072` FC1 仅 396 TFLOPS，同样 2816/2116 行时 uniEP 是 418/794）。
- 大 m 段（`T1024/H3072`，每专家 ~4133 行）差距收窄但仍存在：uniEP 746/765 TFLOPS vs TE 629/604 TFLOPS（领先 19–27%）；`T128` 档拉开到 1.35–2.06×。
- **反向：只有 TE 现成**（`GroupedLinear` 自带 autograd）。`T1024/H3072`：FC1 992.1 → **2086.7µs（2.1×）**、FC2 516.2 → **1711.1µs（3.3×）**；小 shape 上 kernel 只有 58–72µs，wall 却 **642–793µs**（autograd + 每次 gemm 的 wgrad launch），训练时能被上层 kernel 部分掩盖但不能忽略。**uniEP / DeepEP 完全没有反向**——uniEP 需要按 §4.5 用 `group_gemm` 自拼（本文已实测，见 §3.4/§3.5）。
- TE 还有 **launch 开销**问题：`T128/H1024` FC1 的 kernel 只要 27.3µs，wall 中位却 170.8µs（4 次 grouped GEMM launch + 参数打包 + 主机抖动）。在真实训练 loop 里这部分能被上一层 kernel 掩盖，但 microbenchmark 里非常显眼；uniEP/DeepEP 的融合路径只有 1–2 次 launch。
- 注意 TE 的 kernel 时间比 wall 中位还高一点（992.1 vs 839.5 @H3072 FC1）：CUPTI 的"每 kernel 取中位再求和"在多个 kernel 之间不相关时会略高于"整段的 wall 中位"，属于口径差异，量级（±15%）可忽略。

### 4.2 通信：DeepEP 快的部分与不快的地方

- **combine 是 DeepEP 的强项**（`T1024/H3072`：174.3 vs 401.3µs，2.3×）。uniEP 的 `combine` 走 "push + NVL barrier + topk reduce" 三步（`push_mode="tile"`），barrier 是全卡同步点，要等 17408 行的 push 全部落地；DeepEP 的 combine 用 expand-mode 直接对本地 buffer 做 reduce，省掉了显式 barrier。
- **dispatch 两家同档、DeepEP 略优**（120.0 vs 139.9µs @H1024；339.1 vs 437.1µs @H3072）。uniEP 还需要额外一次 `layout`（66µs：histogram + cumsum + allgather + layout kernel），DeepEP 的 layout 是纯 device 的。
- **`T128` 档要看 wall 而不是 kernel**：DeepEP 的 dispatch kernel 只有 25.0µs，但 wall 中位 79.4µs（新增一次 launch + host 侧准备）；uniEP 是 50.8 / 85.2µs。两者在一层 ~200µs 的规模上已经没区别，谁快取决于当次主机抖动。
- **`do_cpu_sync`（训练默认）对主机停顿极其敏感**：安静时两种约定没差别（`T1024/H3072` expanded：cpu_sync 339.1/320.8µs vs async 314.7/323.0µs；`T1024/H1024`：120.0/121.1 vs 113.0/119.9），但在主机停顿窗口里 cpu_sync 的单次 dispatch 会被放大到 **3.3–6.8ms**（两次独立观测：一次是 H=3072、一次是本机抖动期），而 async 约定只涨到 ~320–400µs。原因是 cpu_sync 要在 host 侧停等 recv count，任何调度延迟都直接叠加到关键路径上。表里两种约定都列了（`deepep_dispatch*` = cpu_sync，`deepep_dispatch*_async` = DeepEP 自己 test harness 的约定）。**建议：训练里用 async + 一次 device→host 的 counts 回读，或直接用上一轮 counts（一步延迟），别在关键路径上反复 cpu_sync。**

### 4.3 MegaKernel 何时赚：固定开销 vs 可重叠部分

MegaKernel 的收益上限 = `min(comm, gemm)`，代价 = 持久化 kernel 的固定开销 + 尾部无法流水：

| 配置 | dispatch | FC1 | 非融合合计 | MK1 | MK1 盈亏 | FC2 | combine | 非融合合计 | MK2 | MK2 盈亏 |
|---|---|---|---|---|---|---|---|---|---|---|
| T1024/rank H1024 | 139.9 | 104.0 | 244 | 344.9 | 慢41% | 55.2 | 120.2 | 175 | 187.9 | 慢7% |
| T1024/rank H3072 | 437.1 | 880.8 | 1318 | 1138.9 | 快14% | 429.3 | 401.3 | 831 | 766.6 | 快8% |
| T128/rank H1024 | 50.8 | 26.1 | 77 | 184.6 | 慢140% | 14.1 | 36.4 | 50 | 55.4 | 慢10% |
| T128/rank H3072 | 56.6 | 133.8 | 190 | 288.1 | 慢51% | 66.5 | 57.7 | 124 | 132.3 | 慢6% |

- `T1024/H3072`：comm(437) 与 gemm(881) 都很大，重叠空间大，MK1 赚 14%、MK2 赚 8%；整层 2 次 launch 的 e2e 赚 23%。
- `T1024/H1024`：gemm 只有 104µs，MK1 的固定开销 ~100µs（344.9 vs 243.9），融合反而慢 41%；MK2 只多 12.5µs 但丢掉了重叠收益，慢 7%；整层打平（wall 慢 32%）。
- `T128`：两个阶段都只有 25–134µs，固定开销完全吃掉收益（MK1 慢 51–140%，MK2 慢 6–10%）。
- 但 `T128/H3072` 的**整层** e2e 融合反而快 5%（437.2 vs 458.9µs）：非融合要走 5 次 launch + 一次 layout，每次 ~20–40µs 的 launch/host 开销累加起来比 MK1 的固定开销还贵。所以"融合是否划算"要按**整层 launch 数**算，不能只看单个 MK。
- SM 预算默认值已接近最优（`gemm_num_sm=None, comm_num_sm=None` = 整机 132 SM 动态分工）：`T1024/H3072` MK1 默认 1198µs、(132,132) **1129µs**、(96,None) 1372µs、(64,32) 1789µs、(32,96) 3224µs；`T1024/H1024` 默认 **311µs**、(132,132) 346µs、(96,None) 389µs。可见 MK1 的慢不是 SM 划分没调好（显式 132/132 只快 6%），而是固定开销与尾部依赖。

### 4.4 SwiGLU 是隐藏成本

| 配置 | SwiGLU (uniEP 路径) | 占整层非融合 e2e |
|---|---|---|
| T1024/rank H1024 | 107.1 µs | 18% |
| T1024/rank H3072 | 358.3 µs | 13% |
| T128/rank H1024 | 16.3 µs | 8% |
| T128/rank H3072 | 54.6 µs | 12% |

uniEP 只提供 dispatch/GEMM/combine 的融合，SwiGLU 得自己写。`T1024/H3072` 时 358.3µs 相当于 FC2 的 83%、整层的 16%；`T1024/H1024` 时 107.1µs 甚至超过了 FC1（104.0µs）。原因是 torch 的 `F.silu(g) * u` 会走 3 个 kernel（silu、mul、以及 `.contiguous()` 的一次拷贝），17408×3072 的中转张量来回搬。换成单 kernel 的 fused SwiGLU（一次读 2I 写 I）能压到 ~100µs 量级，整层再快 10%+。

### 4.5 反向：uniEP 没有 autograd，拼出来比 TE 贵 1.5–3.2×

uniEP 是纯 kernel 库（`group_gemm` / `dispatch` / `combine` / 两个 MegaKernel），**package 里没有任何 `autograd.Function`**（`grep -rn "autograd\|def backward"` 为空）。所以"uniEP 的反向"在今天的代码里并不存在，只能像 §3.4 那样用同一个 `group_gemm` 拼：

| 环节 | 用 uniEP 怎么算 | 成本特征（T1024/H3072 实测） |
|---|---|---|
| FC1 dgrad | `group_gemm(dY1, W1ᵀ)`，N=H、K=2I | **829 µs ≈ 前向 881 的 0.94×**，和 fwd 一个量级 |
| FC1 wgrad | `group_gemm(dY1ᵀ, A)`，M=2I、N=H、K=max(rows_e) | 909 µs ≈ 1.03× 前向；K 需要 padding（4352 vs 实际 4129） |
| FC1 dW reduce | `all_reduce`（EP 组内求和） | **785 µs**，单独一项就接近一整个 wgrad GEMM |
| FC2 dgrad / wgrad | 同上两式 | 421 / 461 µs ≈ 1.0× / 1.07× 前向 |
| FC2 dW reduce | `all_reduce` | **401 µs** |
| SwiGLU bwd | 只有 torch 实现（3–8 个 elementwise kernel） | **1260 µs**，比 FC2 整个 fwd+bwd 还贵 |
| 转置准备 | `dY1ᵀ` / `Aᵀ` / `A2ᵀ` 两次拷贝 | **1683 µs**，比最大的 wgrad GEMM 还贵 |

三条结论：

1. **GEMM 本身不是问题**：dgrad / wgrad 与同形状前向基本 1:1（0.94–1.28×），说明 CuTeDSL grouped GEMM 在反向复用得很干净——这一点比 TE 的 `GroupedLinear` 还好（TE 的 FC2 fwd+bwd 是 fwd 的 4.0×）。
2. **加起来的问题在"周边"**：一层 fwd+bwd 里 2 次 `all_reduce`（1186 µs）+ SwiGLU 反向（1260 µs）+ 转置（1683 µs）合计 **4129 µs，占 9422 µs 的 44%**。这三项都不是 GEMM，而是 uniEP 目前**没有提供**的东西（缺融合反向 kernel、缺权重梯度 `reduce-scatter`、缺 strided dgrad/wgrad）。
3. **真要用 uniEP 训练，必须先补这三块**：否则 §3.5 里 uniEP 的一层 fwd+bwd 会比 TE autograd 慢 **1.5×（T1024/H1024：1357 vs 915）/ 1.8×（T1024/H3072：8232 vs 4578）/ 3.2×（T128/H1024：2609 vs 810）**。反过来，**`T128/H1024` 上 wall 是 uniEP 赢**（447 vs 1304 µs）：TE 的 autograd 在小 shape 上被 Python 主机开销吃掉，而 uniEP 拼装是 5–6 个裸 kernel——小 batch 场景这条很关键（`T128/H3072` 例外：2177 vs 1339，TE 仍略优）。

> 注意 `all_reduce` 这两个数在这台机器上偏悲观：NVLS 不可用（§5.1），151 MB 的 `all_reduce` 只能走 ring/tree，实测 ~193 GB/s/rank。满血 NVLS 机器上这两项应该能降 1/3–1/2——这也是唯一一处"残血环境确实影响本文数字"的地方，但它只影响**反向**，不影响 §3.1–§3.3 的前向结论。

### 4.6 topk 与 SM 预算不是瓶颈

| 探针 | 数据 | 结论 |
|---|---|---|
| topk=16 → 8（`T1024/H1024`） | dispatch 139.9 → 74.3µs（行数 17408 → 8960）、FC1 104.0 → 57.5µs、MK1 344.9 → 256.9µs、e2e 融合 597.5 → 367.9µs / 非融合 597.6 → 335.4µs | 融合依旧慢 10%，**MegaKernel 的劣势来自固定开销，不是 topk=16 展开的行数** |
| `gemm_num_sm` 扫描（MK1，`T1024/H3072`） | 默认 1198µs、(132,132) 1129µs、(96,None) 1372µs、(64,32) 1789µs、(32,96) 3224µs | 默认"132 SM 动态分工"已接近最优，显式指定只快 6% |
| `gemm_num_sm` 扫描（MK1，`T1024/H1024`） | 默认 311µs、(132,132) 346µs、(96,None) 389µs | 同上；小 shape 下 MK1 的 ~100µs 固定开销与 SM 划分无关 |

> `group_gemm_combine`（MK2）没有 `comm_num_sm` 参数（只有 `gemm_num_sm`），且默认值已是最优档，故只报默认值。

### 4.7 选型建议（本机、bf16、单机 8 卡）

| 场景 | 建议 |
|---|---|
| 大 batch（≥1024 token/卡）+ 大 H（≥3072） | **uniEP 全融合**（2186µs）≈ **DeepEP 通信 + uniEP GEMM 自拼**（2158µs）；前者少 4 次 launch、无 host 依赖，推荐前者 |
| 大 batch + 小 H（1024） | **DeepEP 通信 + uniEP GEMM**（448µs），比 uniEP 非融合（597.6µs）快 25%；MegaKernel 别开（打平但 wall 差 32%） |
| 小 batch（128 token/卡） | 通信用 DeepEP expanded（dispatch 46.7–25.0µs / combine 40.0–29.3µs），GEMM 用 uniEP（比 TE 快 1.4–2.0×）；`T128/H3072` 可以开融合（省 launch），`T128/H1024` 别开 |
| 需要反向 | TE 的 `GroupedLinear` 现成（autograd，本文给了 fwd+bwd）；**uniEP 没有 autograd**，需按 §4.5 用 `group_gemm` 自拼（dgrad/wgrad GEMM 与前向 1:1，但权重梯度 `all_reduce`、SwiGLU 反向、转置三块要自己补，合计占反向 44%）。T1024 上 TE 更快（4578 vs 8232µs @H3072），`T128/H1024` 上 uniEP 拼装的 **wall** 反超 2.9×（447 vs 1304µs） |

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
# 反向（uniEP 用 group_gemm 拼 + TE autograd）
env NCCL_NVLS_ENABLE=0 FLASH_COMM_BUFFER_BACKEND=cuda_ipc \
  torchrun --nproc_per_node=8 bench_uniep_bwd.py --tokens 1024 --hidden 3072
```

| 文件 | 作用 |
|---|---|
| `moe_bench_common.py` | routing 生成（三家共用同一份 routing）、per-iteration CUDA event 统计、CUPTI 中位统计、CSV 落盘 |
| `bench_uniep_ep.py` | uniEP 非融合 5 个 op + MK1/MK2 + 融合/非融合 e2e |
| `bench_deepep_ep.py` | DeepEP plain/expanded/async 的 dispatch、combine、comm-only e2e |
| `bench_te_group_gemm.py` | TE grouped GEMM FC1/FC2、fwd+bwd、FC1+SwiGLU+FC2 前向与 fwd+bwd |
| `bench_uniep_bwd.py` | uniEP 反向：dgrad/wgrad/SwiGLU bwd 用 `group_gemm` 拼装 + 对齐 torch autograd 的检查（§3.4/§7.1） |
| `check_uniep_moe.py` | 与纯 torch MoE 参考实现对比（正确性门禁） |
| `sweep_sm_uniep.py` | MegaKernel 的 `gemm_num_sm`/`comm_num_sm` 扫描（§4.6） |
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

### 7.1 反向正确性（`bench_uniep_bwd.py --check`）

反向不是照着 `group_gemm` 猜的：`check_chain()` 在**同一份 per-expert 行切分**上跑一遍 torch autograd
（`OUT = (silu(A@W1ᵀ)*…)@W2ᵀ`，`loss = <OUT, dOUT>`），再和"用 uniEP kernel 拼出来的 dA / dW1 / dW2"逐元素比：

| 配置 | dW1 相对误差 | dW2 相对误差 | dA 相对误差 |
|---|---|---|---|
| T1024 H1024 | 3.45e-3 | 0（逐位相同） | 5.78e-3 |
| T1024 H3072 | 3.32e-3 | 0（逐位相同） | 3.87e-3 |
| T128 H1024 | 3.30e-3 | 0（逐位相同） | 4.02e-3 |
| T128 H3072 | 3.27e-3 | 0（逐位相同） | 3.82e-3 |

- 误差量级与 bf16 累加一致；`dW2` 逐位相同说明两边都是"K 方向顺序 fp32 累加、最后一次舍入"，没有 split-K 差异。
- 检查里 dgrad/wgrad 全部走 uniEP kernel（不是 numpy/torch 复算），所以通过即代表**这条拼装路径本身**是对的。
- 另外确认了一件影响实现的事：uniEP dispatch 的**对齐填充行是零**（8 个 rank 的 padding 块 `abs().sum()==0`），
  因此 wgrad 可以直接把整块对齐区间（含 padding）当 K 用，不需要额外的 mask。

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
| uniEP "没有反向"：`group_gemm` 不是 `nn.Module`、包里没有任何 `autograd.Function` | 它是纯 kernel 库（和 DeepEP 一样只做前向） | 按 §4.5 用 `group_gemm` 拼 dgrad/wgrad，并用 torch autograd 对齐（§7.1） |
| `group_gemm` 传 `problem_sizes_m` 用 Python `list` 报 `'list' object has no attribute 'shape'` | 它直接读 `.shape` | 传 `torch.as_tensor(counts, dtype=int32, device="cuda")` |
| `all_reduce` 实测只有 ~193 GB/s/rank（151MB 用 785µs） | 本机 NVLS 不可用（§5.1），只能走 ring/tree | 反向的权重梯度 reduce 是**唯一**被"残血环境"影响的数字，满血机器应快 1/3–1/2 |
| DeepEP expand 布局的 `m_splits` 重建不出来（`psum` 差值 16532 vs `exp_x` 行数 16640，rank 0 对、其它 rank 错） | 有效行位于每个对齐块的**尾部**（`EPHandle` 的 `expert_token_idx_start = psum - unaligned` 语义），不是简单的 cumsum 对齐 | 放弃"实测串行 pipeline"，改报**分量相加**（§3.3 已标注为估算） |
