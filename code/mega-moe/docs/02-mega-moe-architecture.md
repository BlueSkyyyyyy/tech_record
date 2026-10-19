# Mega MoE 整体架构与通信-计算重叠

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 2 篇，对应路线图 **P0-1**
> （§8 为 P0-2 的线上基准补充）。目标是把 DeepGEMM 里 Mega MoE 的**数据流**和
> **通信-计算重叠机制**先画清楚，作为后面逐段精读的坐标系。
>
> **重要**：Mega MoE 主实现 `sm100_fp8_fp4_mega_moe.cuh` 是 **SM100（Blackwell）专属**
> （tcgen05 MMA、2-CTA cluster、UE8M0 block scale、FP8xFP4）。本文凡涉及该 kernel 的部分
> 均标注「SM100 专属，仅分析」，本机 H100（sm90）不运行端到端。
>
> 源码坐标（`~/github/DeepGEMM`）：
> - API：`csrc/apis/mega_moe.hpp`
> - 主体 kernel：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`（1528 行）
> - 调度：`deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh`（481 行）
> - 布局/缓冲区：`deep_gemm/include/deep_gemm/layout/mega_moe.cuh`（459 行）
> - block config 启发式：`csrc/jit_kernels/heuristics/mega_moe.hpp`（265 行）
> - 测试/基准：`tests/test_mega_moe.py`（467 行）
>
> 以上行号均按 2026-09-30 公开版（commit `057ca59`）核对。

## 0. 先记住一个画面：一个后厨，五个工位

把 Mega MoE 想成一个后厨，五道工序原本是五个独立的房间：

**备菜（EP dispatch）→ 热炒（linear1）→ 调味（SwiGLU）→ 回锅（linear2）→ 上菜（EP combine）**

**原来的做法**：每道工序一个房间。备菜房把菜做完、封箱、送到热炒房；热炒房开火前必须等备菜房
**全部**收工；上菜也要等回锅全部结束。结果是灶台和备菜员轮流闲着。

**Mega MoE 的做法**：五道工序搬进同一个后厨，中间只有一个**固定格数的传菜窗口**（ring buffer）。

- 备菜员（dispatch warp）不停往窗口里放菜；
- 灶台（MMA warp）看到窗口里有菜就开炒，不等备菜员把今天的菜全备完；
- 装盘（epilogue warp）把炒好的菜摆盘，一部分直接回锅（linear2），一部分端走（combine）；
- 窗口的**满/空指示灯**（mbarrier 的 full/empty barrier）负责互相通知——
  窗口满了备菜员等，窗口空了灶台等。

这样一来，NVLink 的等待时间被 Tensor Core 的计算填满了。这就是「通信-计算重叠」的全部直觉，
剩下的都是工程细节。

> **类比的边界**：现实后厨里「等菜」只是站着；GPU 上「等」意味着 warp 停摆、SM 空转。
> 另外真实 kernel 的窗口格数、预热波数、死锁上界都要在编译期白盒推导出来（§3.3、§4.3），
> 推错就会**死锁**——这不是类比能覆盖的代价。

读完这一篇，你应该能回答三个问题：

1. 五段计算分别落在哪些 warp 上？它们之间靠什么同步？
2. 为什么 dispatch 和 GEMM 之间**没有**阶段栅栏？`l2_full_mask` 位掩码解决了什么问题？
3. 为什么这套结构比「先通信后计算」快，代价又是什么？

## 1. 一段话总览

一个 MoE 层的前向，是五段串行计算：**EP dispatch → linear1 → SwiGLU → linear2 → EP combine**。

朴素实现里，NVLink 通信和 Tensor Core 计算互相等待，SM 与链路轮流空闲。Mega MoE 把这五段
塞进**一个 persistent kernel**：

- 用**对称显存**（symmetric memory）当通信缓冲区；
- 把 **dispatch warp**（持续拉远端 token）与 **MMA/epilogue warp**（持续吃本地 tile 做 GEMM）
  用 **ring buffer + mbarrier + 全局计数**解耦；
- 于是通信延迟被计算掩盖。

**入口在哪**：API 是 `fp8_fp4_mega_moe`（`csrc/apis/mega_moe.hpp:155`）。它在 `arch_major == 10`
时派发到 `sm100_fp8_fp4_mega_moe`（`:256`），其它架构直接
`DG_HOST_UNREACHABLE("Unsupported architecture")`（`:276`）——这就是「SM100 专属」最直接的证据。

同一文件里还有 `bf16_mega_moe`（`:285`，派发到 `sm100_bf16_mega_moe`，`:367`），
以及两个与 kernel 无关的尺寸计算：**block config** 与 **symmetric buffer 尺寸**
（`get_block_m_for_mega_moe:21`、`get_symm_buffer_size_for_mega_moe:35`）。

## 2. 数据流脑图

下面的图是一次 kernel launch 内、单个 rank 上的数据流（R = 远端 rank，L = 本地）。
第一次看不用逐字读，先看**从上到下的五个阶段**，每段的细节在后面几节：

```
                        ┌─────────────────────────── symmetric memory (per-rank 对齐) ───────────────────────────┐
                        │                                                                                          │
  rank R 的输入 token    │  input_token_buffer / input_sf_buffer / input_topk_idx / input_topk_weights             │
  ──────────────────────►│  (x, x_sf, topk_idx, topk_weights，launch 前由 host 写入；仅本 rank 自己读)              │
                        │                                                                                          │
                        │        ┌──────────────────────────────────────────────────────────────┐                  │
                        │        │  dispatch warps (warp 0..3)  —— 持续把远端 token 拉进 ring     │                  │
                        │        │  1) 统计每个 expert 收到多少 token (atomicAdd_block)          │                  │
                        │        │  2) 全局 grid_sync + NVLink barrier 汇总 expert_recv_count     │                  │
                        │        │  3) TMA 1D load 远端 input_token_buffer → smem_send_buffer     │                  │
                        │        │     → TMA 1D store 本地 l1_token_buffer[pool_token % ring]     │                  │
                        │        │  4) 写 SF / topk_weight / TokenSrcMetadata                     │                  │
                        │        │  5) red_add l1_full_count[ring_block] += 1（最后一个 token 补齐）│                  │
                        │        └───────────────┬──────────────────────────────────────────────┘                  │
                        │                        │ l1_full_count == BLOCK_M*(gen+1)                                │
                        │                        ▼                                                                 │
                        │   ┌──────────────────────────┐    full/empty barriers (mbarrier)                        │
                        │   │ TMA load warp A (warp 4) │◄──── 等待 ring 满 + 空 barrier ──► smem_a[stage] (FP8)      │
                        │   │ TMA load warp B (warp 5) │◄──── TMA 载权重+SF ─────────────► smem_b[stage] (FP8/FP4)  │
                        │   └───────────┬──────────────┘                                                            │
                        │               ▼                                                                           │
                        │   ┌──────────────────────────┐   tcgen05 2-CTA UMMA (swap A/B)                            │
                        │   │ MMA issue warp (warp 6)  │──► TMEM accumulator (kNumEpilogueStages=2 级流水)             │
                        │   └───────────┬──────────────┘   完成后 tmem_full_barrier                                  │
                        │               ▼                                                                           │
                        │   ┌────────────────────────────────────────────────────────────┐                          │
                        │   │ epilogue/combine warps (warp 8..15，2 warpgroups)           │                          │
                        │   │  L1: 读 TMEM → SwiGLU → 求 amax → UE8M0 量化 → FP8 存         │                          │
                        │   │      l1_output(TMA) 与 l2_token_buffer / l2_sf_buffer       │                          │
                        │   │      → arrive(l2_full_mask)[n_block] → red_add(l1_empty)    │                          │
                        │   │  L2: 读 TMEM → bf16 → 存 combine_token_buffer               │                          │
                        │   │      → red_add(l2_empty_count)                              │                          │
                        │   └───────────┬────────────────────────────────────────────────┘                          │
                        │               │ 所有 L2 写完 → grid_sync → 通知 peer: combine_ready_grid_idx = grid_idx   │
                        │               ▼                                                                           │
                        │   ┌────────────────────────────────────────────────────────────┐                          │
                        │   │ combine (epilogue warps)                                    │                          │
                        │   │  读本 rank topk_idx → TMA load 各选中 expert 所在 rank 的      │                          │
                        │   │  combine_token_buffer → bf16 加权累加 → 写 y                 │                          │
                        │   └────────────────────────────────────────────────────────────┘                          │
                        └──────────────────────────────────────────────────────────────────────────────────────────┘
```

**这张图最关键的一句话**：**dispatch 与 GEMM 之间没有「阶段栅栏」**，只有 ring slot 的满/空计数。
dispatch 先把一批 token 填进 ring，MMA 就能开算，后面的 token 边拉边算。这就是重叠的来源。

读图时注意三处「谁等谁」：

1. `l1_full_count == BLOCK_M*(gen+1)`：warp4 等的是**某个 ring block 填满**，不是「全部 token 到齐」。
2. `full/empty barriers`：warp4/5（搬数据）与 warp6（发 MMA）之间的一对红绿灯。
3. `l2_full_mask`：linear2 等的是**某个 K block 就绪**，不是「整个 linear1 做完」。这条是 §4.3 的主角。

## 3. 缓冲区布局（`layout/mega_moe.cuh`）

### 3.1 `MegaMoEBuffer`：一块大对称显存切成若干段

`MegaMoEBuffer`（`layout/mega_moe.cuh:345`）把一段连续显存按固定顺序切开。
可以把它看成「后厨里那排贴着标签的保鲜柜」：

| buffer | 用途 | 行号 |
|---|---|---|
| `input_token_buffer` / `input_sf_buffer` / `input_topk_idx_buffer` / `input_topk_weights_buffer` | 本 rank 的输入，dispatch 读取 | `:401`-`:412` |
| `shared_l1_token_buffer` (=input_token) / `shared_l1_sf_buffer` | 共享专家 L1 输入（复用 input token） | `:415`-`:418` |
| `shared_l2_token_buffer` / `shared_l2_sf_buffer` | 共享专家 L2 输入/输出 | `:419`-`:424` |
| `l1_token_buffer` / `l1_sf_buffer` / `l1_topk_weights_buffer` | 路由专家 ring：dispatch 写入、linear1 读取 | `:427`-`:438` |
| `l2_token_buffer` / `l2_sf_buffer` | ring：linear1 的 SwiGLU+量化输出 → linear2 输入 | `:440`-`:445` |
| `combine_token_buffer` | 各 topk 槽的 L2 bf16 结果，供 combine 读回 | `:447`-`:449` |

整块大小由 `get_num_bytes()`（`:452`）给出，即从 `workspace.signals` 到 `combine_token_buffer` 末尾。

### 3.2 `Workspace` 与 `MegaMoESignals`：跨 rank 的「对讲机」

`MegaMoESignals`（`:54`）定义在对称 buffer 的起始处。所有 rank 按同样偏移映射，
所以它就是**跨 NVLink 同步**的载体——每个 rank 写自己那份，但读得到别人的。

字段分五类，每一类对应一种「等人」：

- **grid / NVLink 同步**：`grid_sync_count[4]`、`nvl_barrier_counter`、`nvl_barrier_signals[2]`
  （`:62`-`:64`）。用于整颗 grid 的对齐与跨卡屏障。
- **任务计数**（按 locality domain 分桶）：`l1_task_count` / `l2_task_count` /
  `shared_l1_task_count` / `shared_l2_task_count`（`:67`-`:70`）。scheduler 用 `atomic_add`
  在这些计数器上领任务（见 §4）。
- **combine 就绪**：`combine_ready_grid_idx[peer]`（`:74`）。语义是「peer 的 grid index」——
  只有当 peer 已完成对本 rank 的 L2 写入时才置位。因为 grid index 每次 launch 唯一，所以**不需要复位**。
- **专家 token 计数**：`expert_send_count` / `expert_recv_count` / `expert_recv_count_sum`
  （`:78`-`:80`）。dispatch 用它做 all-to-all 的收发握手。
- **ring 满/空信号**：`l1_full_count` / `l1_empty_count` / `l2_full_mask` / `l2_empty_count`（`:82`-`:86`）。

最后一行里的 `l2_full_mask` 值得单独点出来：它是**每 L1 的 N block 一个 bit** 的位掩码（`:85` 注释）。
这是 linear2 能「按 K block 细粒度」启动的关键——不是等整层，而是等它需要的那几个 bit。

`Workspace::get_num_bytes()`（`:128`）在其后追加两块数据：**源 token-topk 索引**
（`[local expert][source rank][token]`，`:136`）和 **combine 回写元数据** `TokenSrcMetadata`（`:139`）。
访问器分别是 `get_src_token_topk_idx_ptr`（`:241`）与 `get_token_src_metadata_ptr`（`:249`）。

### 3.3 ring buffer 的容量：不能拍脑袋

ring 大小不是估的，是**推出来的**。

host 侧 `get_symm_buffer_size_for_mega_moe`（`csrc/apis/mega_moe.hpp:35`）遍历所有候选 `BLOCK_M`，
取「最坏情况存活 pool block 数」（`sched::get_num_max_live_pool_blocks`，`scheduler/mega_moe.cuh:48`）
乘 `BLOCK_M` 的最大值，作为 `num_ring_tokens`（`:57`-`:63`）。

为什么要用「最坏情况」？因为 scheduler 里那套死锁上界推导（见 §4.3）就是为「ring 不能太小」
服务的：ring 一小，L1/L2 就会互相等死；ring 一大，又白吃显存。

### 3.4 专家权重与 scale 布局（`transform_weights_for_mega_moe`，P2-1）

路由专家的权重**不在**对称显存里，而是作为独立张量在 launch 前由 host 重排好，再经 TMA 描述符
送进 kernel。host 入口是 `deep_gemm.mega.transform_weights_for_mega_moe`（`mega/__init__.py:146`），
它按 `l1_weights` / `l2_weights` 是 **FP8/FP4 元组**还是 **BF16 张量**分成两条路径（`:153`）：

| 权重 | BF16 路径（`:160`-`:163`） | FP8/FP4 路径（`:153`-`:159`） |
|---|---|---|
| L1 | `_interleave_weights(l1)` | weights `_interleave_weights` + SF `_transpose_sf_for_utccp(_interleave_weights(sf))` |
| L2 | 不变 | weights 不变 + SF `_transpose_sf_for_utccp` |

原始形状（`tests/test_mega_moe.py:130`-`:133`）：`l1_weights [E, 2I, H]`（前半 gate、后半 up），
`l2_weights [E, H, I]`；FP8/FP4 时 SF 为 `[E, N, K/32]`（`:105`-`:118`，经
`transform_sf_into_required_layout` 变成 packed-int + MN-major）。共享专家则是 2D 版本（`:147`-`:150`）。

#### 3.4.1 为什么 gate/up 要按 8 粒度交错

`_interleave_weights(t, gran=8)`（`mega/__init__.py:112`）把 N 轴从前半/后半的 `[gate | up]`
改成 `[gate0..7, up0..7, gate8..15, up8..15, ...]`。

原因在 kernel 的 L1 epilogue：它用 `SM100_TMEM_LOAD_16dp256b1x`（`sm100_fp8_fp4_mega_moe.cuh:1017`-`:1020`）
读 TMEM，注释（`:1066`-`:1067`）明确写着「SwiGLU in-place using granularity 8 interleaved weights」。

交错之后，同一个 8 行 atom 的相邻两个 256b 组，正好就是「gate 8 个 / up 8 个」，
epilogue 可以直接 `bf16_gate` / `bf16_up` 配着做 `silu(gate)*up`（`:1112`-`:1134`），
**不用跨半个 tile 去取 up**。`gran=8` 与 `ATOM_M=8`（`:181`）、以及下面 4x32 的 UTCCP 三者对齐。

**实测**（`harness/exp_weight_transform.py` + `.out.txt`，H100 纯 PyTorch）：输入行序
`G0..G15,U0..U15` 经交错后变为 `G0..7,U0..7,G8..15,U8..15`，与期望逐行一致；
把交错 GEMM 输出按配对还原出的 `silu(gate)*up` 与原始布局结果**最大绝对误差 0**。

#### 3.4.2 SF 的 UTCCP 4x32 转置

FP8/FP4 的 scale 还要再转一次：`_transpose_sf_for_utccp(sf)`（`mega/__init__.py:129`）
对每个 128 行组做 `(4,32) -> (32,4)` 的转置。kernel 侧对应的映射是 `transform_sf_token_idx`
（`sm100_fp8_fp4_mega_moe.cuh:145`-`:149`）：

```
new_idx = (idx & ~127) + (idx & 31) * 4 + ((idx >> 5) & 3)
```

即原行 `r`（组内 `r = b*32 + c`，`b∈[0,4)`、`c∈[0,32)`）落到新位置 `c*4 + b`。

**为什么必须转**：MMA 前要把 SF 从 smem 搬进 TMEM，走的是 `SM100_UTCCP_4x32dp128bit_2cta`
（`:895`、`:900`、`:906`）。这条指令按「4 行 × 32 列」的 pattern 分发 128 个 datapath，
所以 smem 里的 mn 顺序必须**预先按这个 pattern 排好**——顺序不对，指令就把 scale 铺到错误的位置上。

**实测**：对 256 行做 `_transpose_sf_for_utccp` 后，用 kernel 公式逐行反查，**不符行数 = 0**；
与 `deep_gemm.mega._transpose_sf_for_utccp` 的结果最大差 0（`.out.txt`）。

#### 3.4.3 SF 张量在 kernel 里怎么索引

`transform_sf_into_required_layout`（`csrc/apis/layout.hpp:11`）在 SM100、`gran_k=32` 时走
`get_mn_major_tma_aligned_packed_ue8m0_tensor`（`csrc/jit_kernels/impls/smxx_layout.hpp:133`）：
把 4 个 UE8M0 字节打成 1 个 int32，并排成 **MN-major + TMA 对齐**
（元素 `(g,m,c)` 的 stride 为 `{packed_k*aligned_mn, 1, aligned_mn}`，`smxx_layout.hpp:137`-`:138`）。

于是 kernel 里：

- `shape_sfb_k = ceil_div(shape_k, kGranK*4) = K/128`（`:774`）；
- SFB 的 K 偏移是 `local_expert_idx * shape_sfb_k + k_block*(BLOCK_K/128)`（`:788`）——
  乘 expert stride、按每 128 个 K 一个 int32 前进；
- TMA 以 `<BLOCK_N, 1>`（`:805`-`:806`）从 MN-major 的连续 mn 方向取 128 行。

权重本体的 TMA 描述符由 `make_tma_weights_3d_desc`（`csrc/jit_kernels/impls/runtime_utils.hpp:311`）
把 `[E, N, K]` 变成 4D `(K, N/2, 2, E)`（N 被 locality domain 一分为二），SF 描述符见
`make_tma_sf_desc`（`:384`）。TMA load 里的 `half_n = shape_n/2`、`half_idx` 与 `n_block_idx`
（`:775`、`:778`）就是配合「A/B swap + 2-CTA 各取一半 N」的寻址。

> 以上变换**全部在 host 完成、不占 kernel 时间**；kernel 只消费排好的权重与 SF。
> 这些路径都需要 SM100 的 UTCCP/tcgen05，**本机 H100 跑不了端到端**，
> 本节只做源码精读 + 纯 PyTorch 重排语义核对。

## 4. 调度器（`scheduler/mega_moe.cuh`）

### 4.1 任务与阶段

`TaskInfo`（`:91`）描述一个 tile 任务：`block_phase`、`local_expert_idx`、`m_block_idx`、
`n_cluster_idx`、`pool_block_idx`、`valid_m`、`shape_n`、`shape_k`。

`BlockPhase`（`:83`）有五种：`Linear1`、`Linear2`、`SharedLinear1`、`SharedLinear2`、`None`（哨兵）。

一个「M block」= 某个 expert 的连续 `BLOCK_M` 个 token（可能不满）。
`valid_m` 记录实际有效行数，`get_umma_aligned_valid_m()`（`:127`）把它对齐到 16，供 UMMA 动态裁剪 N。

> 一句话：**一个 task = 一个 (expert, M block, N cluster) 三元组**。task 就是「一张工单」。

### 4.2 两种领工单的方式

- **路由专家**：全局原子计数器 `l1_task_count` / `l2_task_count`，
  由 `get_next_task_idx`（`:352`）把线性 task index 拆成 `(m_block_idx, n_cluster_idx)`。
- **共享专家**：走 `shared_l1/l2_task_count`，由 `shared_mainloop`（`:421`）驱动。
  它**不依赖 dispatch**，所以既可以在 dispatch 结果到达前先算（`:447`-`:451`），
  也可以放到末尾（`:467`-`:471`），由 `is_shared_l2_early`（`:444`）决定。

### 4.3 L1→L2 的细粒度依赖与死锁规避

这是 scheduler 最精妙、也最容易读懵的部分。可以拆成三问。

**第一问：L2 什么时候能开算？**

L1 的每个 N block 完成后，翻转 `l2_full_mask` 中自己那一位（`L2KBlockDependency::arrive`，`:156`）；
linear2 的每个 K block 只等覆盖它的那几位（`wait`，`:161`-`:165`）。

为什么能这么切？因为 L1 的 N 轴 = 2×intermediate，L2 的 K 轴 = intermediate，两者有固定对应关系
（`kNumL1BlockNsPerL2KBlock = BLOCK_K / (BLOCK_N/2)`，`:145`）。所以 L2 可以「L1 每完成一段就吃一段」，
不必等整个 L1 做完。

**第二问：那为什么会死锁？**

因为 L1 任务比 L2 任务多（`num_l1_n_clusters > num_l2_n_clusters`）。如果调度器先猛发 L2 任务，
这些 L2 会占满 ring 等 L1，而 L1 又还没被发出来——两边互等，卡死。

**第三问：怎么破？**

主循环 `get_next_task`（`:372`-`:407`）先做**若干波 L1 warmup**（`num_sched_l1_waves`，
在 `fetch_expert_recv_count` 里按 `get_num_l1_warmup_waves` 计算，`:14`-`:45`），
保证「任何 M block 的 L2 任务被发之前，它的 L1 任务已经全部被发出」；
L2 分支里还显式等待该 pool block 对应的 L1 task 计数到齐（`:399`-`:402`）。

**ring 容量上界**：`get_num_max_live_pool_blocks`（`:48`-`:80`）给出「同时存活 pool block 数」
的保守闭式上界（warmup 后存活 + frontier 增长 + 一波余量），host 用它定 ring 大小。

> 这三问连起来就是一句话：**细粒度依赖让 L2 能早开始，但早开始会抢光 ring；
> 所以先用 warmup 把 L1 铺开，再用闭式上界把 ring 定大。**

### 4.4 `locality domain`：先吃自己锅里的

`kNumDeviceLocalityDomains` 个 domain（比如双 die GPU 上按 die 划分）。
`sm_locality_domain_idx`（`:228`）让每个 SM 优先领本 domain 的任务，队列空了再去 work-steal
（`:363`-`:370`）；任务少时（`kNumMinStealWaves=2`）干脆关掉 steal，避免跨 die 延迟（`:225`）。
域数不可整除时退化为 1（kernel 里 `:85`-`:87`）。

道理很朴素：跨 die 取任务，取回来的时间可能比算它还长。

## 5. warp 分工（主体 kernel）

`sm100_fp8_fp4_mega_moe_impl`（`impls/...cuh:56`）的线程划分由启发式给出
（`heuristics/mega_moe.hpp:220`-`:221`、`:126`）：

- `kNumDispatchThreads = 128`（4 warps）：dispatch；
- `kNumNonEpilogueThreads = 128`（4 warps）：TMA load A、TMA load B、MMA issue、scheduler mainloop；
- `kNumEpilogueThreads = 256`（8 warps = 2 warpgroups）：GEMM epilogue + combine。

对应 kernel 里的 `if (warp_idx < ...)` 分支：

| warp | 角色 | 行号 |
|---|---|---|
| 0..3 | dispatch：拉 token、写 ring、发 full 信号 | `:342` |
| 4 | TMA load **激活** + SFA | `:690` |
| 5 | TMA load **权重** + SFB | `:756` |
| 6 | MMA issue（仅 leader CTA） | `:819` |
| 7 | scheduler mainloop（仅 leader CTA） | `:939` |
| 8..15 | epilogue / combine | `:946` |

寄存器也按角色重新分配（`warpgroup_reg_dealloc/alloc`，`:344/:692/:948`），
把寄存器预算倾向 epilogue（`kNumEpilogueRegisters`，`:331`）。

> 一句话读法：**warp 0–3 管「进货」，4–5 管「上料」，6 管「炒」，7 管「派单」，8–15 管「装盘上桌」。**

### 5.1 dispatch 段（`:342`-`:689`）

1. **本地计数**：读 `input_topk_idx`，`atomicAdd_block` 到 smem 的 `expert_token_count`
   （`:369`-`:371`）。
2. **全局汇总**：把每个 expert 的本 rank 计数 `atomic_add` 进对称的 `expert_send_count`
   （`:374`-`:381`），再 `grid_sync` + NVLink barrier，然后 SM0 把结果写进各 rank 的
   `expert_recv_count` / `_sum`（`:398`-`:418`）。
   到这里，每个 rank 都知道「该 expert 总共收到多少 token」，以及「从哪个 rank 来多少」。
3. **拉取**：`for token_idx ...`（`:449`）按 expert 顺序遍历本地要处理的 token，用**迭代 min-peeling**
   在各 rank 间 round-robin（`:480`-`:528`）算出源 rank / 源 token；用 TMA 1D load 把远端
   `input_token_buffer` 分块搬进 smem，再 TMA store 到本地 `l1_token_buffer[pool_token % ring]`
   （`:552`-`:575`）。SF 走普通 LDG/STS（`:584`-`:600`），topk 权重与 `TokenSrcMetadata` 也一并写好
   （`:603`-`:608`）。每填一个 token 就 `red_add l1_full_count`，块内最后一个补齐到 `BLOCK_M`
   （`:612`-`:616`）。
4. **清理**：与 epilogue 的 combine 重叠（`:621`-`:623` 注释），复位计数与 ring 信号，供下一层使用。

一个值得注意的工程选择：dispatch 用 **TMA 1D** 做「远端 → smem → 本地」，**不是**直接 NVLink load。
这样把远端延迟和本地写解耦，而 smem 当暂存天然能流水。

### 5.2 TMA load 段（`:690`-`:818`）

- **warp4 载激活**：先等该 ring block 的 `l1_full_count` 达到 `BLOCK_M*(gen+1)`（`:713`-`:716`），
  再 TMA load `smem_a` 与 SFA，arrive `full_barriers`（`:742`-`:752`）。
- **warp5 载权重**：按 expert 取权重 tile（共享专家时用 `copy_nd` 带 expert 维，`:793`），
  SFB 的 K 偏移含 `local_expert_idx * shape_sfb_k`（`:788`），TMA load 后 arrive（`:791`-`:814`）。
- **linear2 额外等** `L2KBlockDependency`（`:723`-`:726`），即 §4.3 的细粒度依赖。

### 5.3 MMA 段（`:819`-`:938`）

只由 leader CTA 跑。**始终交换 A/B**（注释 `:159`），`UMMA_M = 128*2`（2-CTA），
`UMMA_N = BLOCK_M`（`:160`-`:163`）。

每个 k block 的流程是：等 `full_barriers` → UTCCP 把 SFA/SFB 铺进 TMEM（`:894`-`:907`）
→ `SM100_MMA_MXF8F6F4_2x1SM_SS::fma` 发射（`:918`），累加进 `accum_stage_idx` 对应的 TMEM 列
（双缓冲 `kNumEpilogueStages=2`，`:229`）。完事 arrive `empty_barriers` 与 `tmem_full_barriers`
（`:874`-`:878`）。

### 5.4 epilogue / combine 段（`:946`-`:1520`）

- **L1 阶段**（`:1058`-`:1253`）：从 TMEM 读回 → **SwiGLU**（`silu(gate)*up`，`:1108`-`:1135`）
  → 线程 / warp / warp-pair 三级 amax 规约（`:1137`-`:1170`）→ 转 UE8M0 缩放因子并量化成 FP8
  （`:1172`-`:1191`）→ TMA store 到 `l1_output`（`:1222`-`:1233`），同时把 SF 以 MN-major 写进
  `l2_sf_buffer`（`:1193`-`:1216`）。最后 arrive `l2_full_mask` 的对应位，并 `red_add l1_empty_count`
  （`:1237`-`:1251`）。
- **L2 阶段**（`:1254` 起）：读 TMEM → bf16 → 存 `combine_token_buffer`，`red_add l2_empty_count`。
- **combine**（`:1365`-`:1520`）：先 `grid_sync` 确保本地 L2 写完，SM0 把本 rank 的 grid index
  写到每个 peer 的 `combine_ready_grid_idx`（`:1417`-`:1419`）；每个 epilogue warp 认领 token chunk
  （`:1426`），读 topk 槽（`:1430`-`:1434`），等选中 expert 所在 rank 就绪（`:1436`-`:1440`），
  TMA load 各槽的 bf16 结果到 smem，**在寄存器里按 topk 权重累加**（`:1472`-`:1490`），
  再写回 `y`（`:1511`-`:1518`）。共享专家作为额外一个槽并入（`:1433`、`:1450`）。

## 6. block config 与线程数（`heuristics/mega_moe.hpp`）

- `get_block_config_for_mega_moe`（`:88`）根据「每专家期望 token 数 + 一 sigma」选 `block_m`
  （`:94`-`:110`）。经验规律：**每个 M block 大约要多花 128 行权重重载的代价，所以尽量少 block、
  用小 tile**：≤10 用 16，≤24 用 32，否则在 `{64,128,192,240}` 里选能覆盖 `num_covered_tokens`
  的最小者（`:105`-`:109`）。
- `store_block_m_l1/l2`（`:112`-`:115`）在 FP8xFP4 prefill 下取更小的非对称值，以省 smem、加深流水。
- `block_n` 固定 128（`:209`），`block_k = 128 / elem_bytes`（`:117`，FP8→128、FP16/BF16→64）。
- 线程布局：dispatch 128、non-epilogue 128、epilogue 256（`:220`-`:221` + `:126` 返回值 `2*128`）。
- 动态段数：`get_pipeline_config_for_mega_moe`（`:129`）按 smem 容量算出最大 `num_stages`（`:191`）。
  dispatch send buffer、C/D staging、SF tile、barrier、task info 都计入固定开销（`:145`-`:188`）。

## 7. 为什么比「先通信后计算 / chunked」快

五条原因，每条都对应前面某一节的机制：

1. **单 kernel、没有 launch/同步间隙**：五段在一个 persistent kernel 内完成，省掉中间张量的物化
   和多次 kernel 启动，也避免 CUDA graph 之外的空转。
2. **通信按 ring slot 流水**：dispatch 不必等全部 token 到齐，MMA 只等当前 M block 的 ring 满
   （`:712`-`:716`），后面的 token 继续拉。NVLink 传输与 UMMA 在时间上重叠。
3. **linear1→linear2 细粒度依赖**：`l2_full_mask` 让 linear2 按 K block、在 linear1 完成一段后
   立即开始（§4.3），而不是等整层 barrier。
4. **共享专家与路由专家交错**：共享专家不吃 dispatch，于是被塞进 dispatch 的等待空隙里
   （`:443`-`:451`）——相当于「等货的时候先干点别的」。
5. **combine 与 workspace 清理重叠**：`:621`-`:623`、`:1417`-`:1422` 的同步注释显示，
   dispatch 的清理和 epilogue 的 combine 是互相掩护的。

**代价是复杂度**：ring 容量、warmup 波数、死锁上界都要在编译期/白盒推导出来
（`get_num_max_live_pool_blocks`），稍有差池就是死锁或者白占显存。

## 8. 线上基准坐标（PR #316，SM100 专属，仅分析）

DeepGEMM [PR #316](https://github.com/deepseek-ai/DeepGEMM/pull/316) 公布了 Mega MoE 在
**EP8**（8 路专家并行）下的实测，8 个 rank 取平均。两组配置（按官方描述）：

- **V4-Flash**：256 专家、top-k=6、hidden 4096、intermediate 2048；
- **V4-Pro**：384 专家、top-k=6、hidden 7168、intermediate 3072。

| 配置 | Batch/rank | 时间 (us) | Compute (TFLOPS) | 显存 (GB/s) | 互联 (GB/s) | vs legacy |
|---|---|---|---|---|---|---|
| Flash | 1 | 56.5 | 5 | 1311 | 1 | 1.96x |
| Flash | 512 | 146.5 | 1056 | 3192 | 266 | 1.73x |
| Flash | 8192 | 1283.1 | 1928 | 998 | 499 | 1.56x |
| Flash | 32768 | 4855.5 | 2038 | 794 | 529 | 1.62x |
| Pro | 1 | 108.1 | 7 | 1758 | 1 | 1.61x |
| Pro | 512 | 369.6 | 1098 | 4619 | 182 | 1.54x |
| Pro | 8192 | 2818.5 | 2304 | 1094 | 393 | 1.50x |
| Pro | 32768 | 10655.2 | 2438 | 692 | 417 | 1.54x |

四点解读，正好呼应前面的机制：

1. **低延迟（batch=1）时互联带宽近 0**：只有 1 个 token/rank，几十 us 里大部分是固定开销与
   跨 rank 握手，compute 才 5–7 TFLOPS——说明这套 fused kernel 的瓶颈不在算力，
   而在 dispatch/combine 的往返延迟（§5.1 的全局计数 + NVLink barrier 是主要成本）。
2. **大 batch 时 compute 高达 2038 / 2438 TFLOPS**：接近 FP8xFP4 峰值，此时互联也吃到
   400–530 GB/s，说明 NVLink 与张量核确实在**重叠**（§7 的 ring slot 流水）。
3. **显存带宽在中等 batch 反而最高**（Flash 512 达 3192 GB/s）：token 数适中时 ring 复用率高，
   dispatch 搬运与 GEMM 读写并存；batch 再大就被 compute 主导。
4. **相对 legacy 稳定 1.5–2x**：batch 越小收益越大（1.61–1.96x），因为 legacy 实现是
   「通信→计算→通信」串行，低延迟下通信占比最高。

> 这些数字来自官方 PR，**未在本机复现**（Mega MoE 是 SM100 专属）。本机 H100 只做可跑的
> grouped GEMM / backward 数值实验，见 `docs/04`。

## 9. 小结与下一步

- Mega MoE = **对称显存做通信缓冲 + ring buffer 解耦通信与计算 + 细粒度 mask 依赖 + warp specialization**。
- 一句话串起来：**dispatch 按 ring slot 供货，MMA 按 M block 开炒；L1 epilogue 炒好一段就翻转
  `l2_full_mask` 的一个 bit，L2 按 K block 接着吃；最后 combine 等 peer 就绪上菜。
  四段之间只有计数器和位掩码，没有阶段栅栏。**
- 本文建立了数据流与角色划分；细节（tcgen05/UTCCP、TMA 描述符、SwiGLU 量化、combine 累加精度）
  留给 `docs/03-mega-moe-kernel.md` 逐段精读。
- 反向（wgrad/dgrad）见 `docs/04-moe-backward.md`；Mega Gate / mHC 见 `docs/05-mega-gate-mhc.md`。
