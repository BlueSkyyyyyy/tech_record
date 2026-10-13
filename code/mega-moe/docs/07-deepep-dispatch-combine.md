# DeepEP dispatch / combine 与 symmetric memory 精读

> 本文对应路线图 **P0-3**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 7 篇。
> 目标：把 EP 通信这一侧的**对称显存、dispatch/combine kernel 的接口与数据流**讲清楚，
> 给 `docs/01` 的 EP 语义、`docs/02` 的「通信-计算重叠」补上真实实现坐标。
>
> **与 Mega MoE 主体的关系**：DeepGEMM 的 `sm100_fp8_fp4_mega_moe` 是 **SM100 专属**（本机 H100 不跑）；
> 而 DeepEP 的 dispatch/combine 目标平台是 **Hopper 及以上**，是**独立可分析、原则上 H100 可用**的通信库。
> 本文属于**源码精读**，本机未安装 `deep_ep`（见 §8），因此不含实测数字，数字部分留给后续 P0-4。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
> - Python API：`deep_ep/buffers/ep.py`（`EPBuffer.dispatch:562`、`EPBuffer.combine:770`）
> - host 侧 launch：`csrc/kernels/ep/dispatch.hpp`（228 行）、`csrc/kernels/ep/combine.hpp`（200 行）
> - 设备侧 kernel：`deep_ep/include/deep_ep/impls/ep/dispatch.cuh`（407 行）、
>   `combine.cuh`（246 行）、`dispatch_copy_epilogue.cuh`（328 行）、`combine_utils.cuh`（172 行）
> - 缓冲区布局：`deep_ep/include/deep_ep/layout/ep/token.cuh`（146 行）、`workspace.cuh`（114 行）
> - 对称显存/窗口：`csrc/kernels/comm/symmetric.hpp`、`csrc/kernels/comm/context.cpp`

## 1. 一段话总览

DeepEP 的 MoE 通信是**两段 all-to-all**：

- **dispatch**：每个 rank 读本地 token 的 `topk_idx`，把「要送到别的 rank 专家」的 token 经
  NVLink/RDMA 搬到目标 rank 的**对称缓冲**里；目标 rank 再按「本地专家」重新排布，产出一个
  **expanded layout**（按专家分组、每 token 一个 slot），正好喂给 grouped GEMM。
- **combine**：专家算完后，把结果按原路送回 token 的源 rank，并按 topk 权重求和。

实现的关键是三层抽象：**对称显存（symmetric memory + NCCL window）→ 固定布局的 token buffer →
warp 分工的流水 kernel**。整个 dispatch/combine 各自都是一个 cooperative kernel（clustered grid +
全局 barrier），而不是 host 端逐 token 处理。

## 2. symmetric memory：为什么通信 buffer 要「对称」

### 2.1 对称分配

`EPBuffer` 构造时通过 `comm::Context` 分配一段**所有 rank 地址一致**的显存
（`csrc/buffers/ep.hpp:71` 起）。物理分配走 CUDA VMM（`cuMemCreate`，`symmetric.hpp:78-79` 的
`cumem_create_with_fallback`，优先尝试 FABRIC handle）；分配出的段在
`csrc/kernels/comm/context.cpp:186`：

```cpp
symmetric_memory = symmetric::alloc(
    num_workspace_bytes + num_gpu_buffer_bytes, num_rdma_storage_bytes, ...);
```

分配完立刻注册成 **NCCL window**（`context.cpp:199`）：

```cpp
NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, symmetric_memory->num_sym_bytes,
                                  &window, NCCL_WIN_STRICT_ORDERING));
NCCL_CHECK(ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &workspace));
buffer = math::advance_ptr(workspace, num_workspace_bytes);
```

- `workspace` 是本 rank 的窗口基址，`buffer` 是窗口内 workspace 之后的通信 buffer 区（`:200`-`:201`）。
- 若支持 LSA multimem，还额外取一个 multimem 指针 `buffer_multimem`（`:202`-`:203`，用于 NVLink 组播）。
- **「对称」的含义**：每个 rank 都拿到同样大小、同样内部偏移的段。于是 kernel 里只要有一个
  本地指针 + 目标 rank 号，就能算出远端同一逻辑位置的地址（`get_sym_ptr`，`context.cpp:226`-`:232`：

```cpp
const auto offset = static_cast<uint8_t*>(ptr) - static_cast<uint8_t*>(workspace);
NCCL_CHECK(ncclGetLsaDevicePointer(window, offset, dst_nvl_rank_idx, &dst_ptr));
```

  ）。这就是 dispatch/combine 里 `gin.get_sym_ptr<team_t>(..., dst_rank)` 的底层依据。

### 2.2 scale-out / scale-up 两级域

`context.cpp:174`-`:183` 把世界拆成两级：**scale-out（RDMA，跨节点）与 scale-up（NVLink，节点内）**。
`allow_hybrid_mode` 时 `num_scaleout_ranks = num_rdma_ranks`、`num_scaleup_ranks = num_nvl_ranks`。
设备侧据此决定 `kIsScaleupNVLink` 与用 `ncclTeamTagLsa`（NVLink）还是 `ncclTeamTagWorld`（混合）
（`dispatch.cuh:30`、`combine.cuh:28`）。非混合模式直接退化为「一个 NVLink 域 = 整个 world」。

## 3. 对称缓冲的布局（`layout/ep/token.cuh`）

对称段被切成 **[Workspace | buffer]**；`buffer` 内再按 `TokenLayout`/`BufferLayout` 组织：

- **`TokenLayout`（`token.cuh:10`-`:30`）**：单个 token 在缓冲里的字节安排，顺序为
  `hidden`（对齐到 TMA 粒度）→ `SF`（量化 scale factor，FP8 dispatch 用）→ `metadata`
  （topk idx + topk weight + 源 rank/token 索引 + linked list）。各段的访问器：
  `get_hidden_ptr:51`、`get_sf_ptr:55`、`get_topk_idx_ptr:63`、`get_topk_weights_ptr:67`、
  `get_src_token_global_idx_ptr:71`、`get_linked_list_idx_ptr:75`、`get_mbarrier_ptr:79`。
- **`BufferLayout`（`token.cuh:84-85`）**：`[num_ranks][num_max_tokens_per_rank]` 的二维数组，
  `get_rank_buffer(r):121` 定位到某 rank 那一段，`get_token_buffer(t):138` 定位到某 token。
  这正是「对称」落地的形态：rank 维 × token 维，各 rank 结构完全相同。

`with_metadata` 区分两种用法：dispatch/combine 主 kernel 带完整 metadata（`with_metadata=true`），
combine 的接收侧也不需要 topk 元数据时会置 false。

跨 rank 的控制信号单独放在**窗口最前面的 `EPSignals`**（`workspace.cuh:9`-`:23`），
经 `EPWorkspaceLayout`（`:25`）访问，包含：

- `barrier_signals`：kernel 级 grid barrier（`gpu_barrier`）；
- `notify_reduction` / `scaleup_rank_expert_count[2]` / `scaleout_rank_expert_count[2]`：
  dispatch 的**每 rank 每专家收 token 计数**汇总（双缓冲，send/recv 各一组，`:61`-`:64`）；
- `scaleup_atomic_sender_count`（`:17`）：给 token 分配接收端 slot 的原子计数器；
- `channel_scaleup_tail` / `scaleout_channel_signaled_tail`：hybrid 模式流水尾指针。

## 4. dispatch kernel 数据流（`impls/ep/dispatch.cuh`）

`dispatch_impl`（`:32`）把整个 grid 的 warp 分成两类（`kNumNotifyWarps` + `kNumDispatchWarps`，
`dispatch.hpp:20` 取 `kNumNotifyWarps = 4`），用一个 cooperative grid 完成 all-to-all。

### 4.1 notify warps（`dispatch.cuh:78`-`:257`）：先算清「谁收多少」

1. **本地直方图**（`:85`-`:107`）：每个 warp 步进遍历本地 token，lane `l < kNumTopk` 读
   `topk_idx`，把目标专家 `atomicAdd_block` 进 smem 的 `expert_count`；对目标 **rank** 做
   `ptx::deduplicate` 去重后计数（一个 token 的多个 topk 若落同一 rank 只算一次，`:103`-`:105`）。
2. **全局归约**（`:110`-`:146`）：把 `(1<<32)|count` 用 `ptx::red_add` 累加进对称的
   `notify_reduction`，高 32 位当「到达的 SM 数」；SM0 轮询直到 `== kNumSMs`，得到全局计数并清零。
3. **跨 rank 交换**（`:149`-`:200`）：SM0 把本 rank 的 rank/expert 计数 `gin.put_value` 写到每个
   peer 的 `scaleup_*_count`（NVLink 逐元素；RDMA 用 `gin.put` 批量，`:170`-`:177`），再轮询等待
   本 rank 收到所有人的计数。
4. **对齐与前缀和**（`:204`-`:256`）：对每个**本地专家**求和得到待收 token 数，按
   `kExpertAlignment` 向上取整（`:214`，对齐 grouped GEMM 的 M 块）；`warp 0` 算 rank 的
   **inclusive** 前缀和（`:252`），`warp 1` 算专家的 **exclusive** 前缀和（`:255`，供 expanded
   layout 直接用基址累加）。若 `do_cpu_sync`，同时写 mapped host workspace（`:223`-`:229`）。

这四步解释了为什么 dispatch 是**延迟敏感**的：它包含两次全局同步（同 rank grid barrier +
跨 rank put/wait 往返）。`docs/02` §8 里 batch=1 时互联带宽近 0、耗时却几十 us，主要就是这里。

### 4.2 dispatch warps（`dispatch.cuh:258`-`:390`）：搬 token

每个 dispatch warp 拥有一个 smem TMA 暂存块（`tma_buffer`，`:262`-`:264`），按步进遍历 token：

- **TMA 载入**：`tma_load_1d` 把本地 `x` 的 hidden 读进 smem（`:287`-`:290`）；SF 用 `cp_async_ca`
  按 stride 载入并 `cp_async_mbarrier_arrive` 接入同一个 mbarrier（`:294`-`:311`）。
- **读 topk + 源元数据**：写 `topk_idx`/`topk_weights` 到 smem（`:316`-`:323`），并记下
  `rank_idx * kNumMaxTokensPerRank + token_idx` 作为源位置（`:327`-`:328`）。
- **分配接收 slot**（`:332`-`:348`）：对目标 rank 去重后 `atomicAdd` 到
  `scaleup_atomic_sender_count[dst_rank]` 拿到接收端 slot（`kReuseSlotIndices=false` 时）；
  缓存模式下直接从 `dst_buffer_slot_idx` 读回（`:334`-`:338`）。
- **发出**（`:350`-`:389`）：等 TMA 到齐后，`tma_store_1d` 写远端 `recv_buffer` 的对应 slot
  （`:369`-`:373`，`gin.get_sym_ptr` 得到远端地址）；非 NVLink 可达的 rank 先写本地 send buffer
  再 `gin.put` RDMA（`:378`-`:388`）。

### 4.3 copy epilogue → expanded layout（`impls/ep/dispatch_copy_epilogue.cuh`）

主 dispatch kernel 只保证「token 落到目标 rank 的对称 buffer」；真正排成 **grouped-GEMM 友好**
布局的是**启用了 PDL 的第二个 kernel**（`launch_dispatch_copy_epilogue`，`dispatch.hpp:168`），
主 kernel 末尾 `cudaTriggerProgrammaticLaunchCompletion()`（`dispatch.cuh:399`）触发它重叠启动。

- `cudaGridDependencySynchronize()`（`dispatch_copy_epilogue.cuh:61`）等主 kernel 数据可见；
- 遍历本 rank 收到的 token（`:71`），TMA 读回 smem（`:90`-`:94`）；
- **expand 模式**：用一个全局原子 `atomicAdd(psum_num_recv_tokens_per_expert + expert, 1)`
  给每个 token 分配它在「按专家拼接的张量」中的行号 `dst_tensor_idx`（`:121`-`:123`），
  再 TMA store 到 `recv_x[dst_tensor_idx]`（`:139`-`:143`）。这就是 README 所述
  **「one slot per expert per token」的 expanded layout**，`recv_topk_weights` 也变成一维
  （每行一个权重，`:180`-`:186`）。
- 可选 **zero padding**（`:232`-`:325`）把专家之间的对齐空隙清零，保证 grouped GEMM 读到的是 0 而非脏数据。

> 这条数据流正是 DeepGEMM Mega MoE 前端 dispatch 的语义来源：Mega MoE 在 kernel 内部用 ring buffer
> 代替了这里的「对称 buffer → copy epilogue」两段式，但**「按本地专家分组、每 token 一个槽」**的
> 布局目标完全一致。对照 `docs/02` §5.1 的 dispatch 段。

## 5. combine kernel 数据流（`impls/ep/combine.cuh`）

`combine_impl`（`:29`）是 dispatch 的逆过程，但多了**加权归约**：

1. **就绪栅栏**（`:79`-`:81`）：先全 grid barrier，确保对端 buffer 可用。
2. **遍历待归约 token**（`:84`-`:87`），从 `src_metadata` 解出源 token、源 rank、源 topk
   slot（`:88`-`:93`）——这些元数据正是 dispatch copy epilogue 写入的。
3. **定位写回地址**（`:97`-`:107`）：NVLink 可达则直接算远端地址（`nvlink_bypass`）；
   否则先写本地 send buffer。
4. **归约**（`:126`-`:214`）分三种：
   - **不归约**（`:128`-`:144`）：直接把该 slot 的 bf16 结果 TMA 写回；
   - **本地归约**（`:145`-`:177`，`kAllowMultipleReduction`）：一份 token 的多个 topk 落在同一 rank 时，
     在**寄存器里做 bf16 加**（`combine_reduce`，`combine_utils.cuh:55`），
     `combine_utils.cuh:67`-`:70` 的 `enable_hadd_bypass` 用 bf16 直接相加而非转 float，省去 cast；
     另有 `use_rank_layout`（`combine_utils.cuh:8-9`）在 `num_ranks <= num_topk` 时按 rank 而非 topk 排布；
   - **expanded send**（`:178`-`:214`）：不复用 slot，每 topk 单独送。
5. **写回**：NVLink 用 `tma_store_1d` 到远端；RDMA 先落 send buffer 再 `gin.put`（`:229`-`:237`）。
6. **结束 barrier**（`:241`-`:243`）。

> 训练里 **combine 的前向就是 dispatch 的反向**：README 明说 `dispatch_backward` 实际上调 combine、
> `combine_backward` 实际上调 dispatch（`ep.py` 的示例 helper）。这对应 `docs/04` 反向中
> 「dgrad 前需要把上游梯度 dispatch、wgrad 后需要 combine」的通信结构。

## 6. 接口：Python → host → kernel

- **Python**（`deep_ep/buffers/ep.py`）：`EPBuffer.dispatch:562` / `EPBuffer.combine:770`，
  参数含 `do_expand`、`do_zero_padding`、`expert_alignment`、`num_sms`、`previous_event`、
  `async_with_compute_stream`、`defer_epilogue`。`expert_alignment` 建议取 DeepGEMM 的
  `get_mk_alignment_for_contiguous_layout()`（README「Example use in model training and inference」）。
- **host launch**（`csrc/kernels/ep/dispatch.hpp:31`）：先按 smem 容量算 dispatch warp 数
  （`:75`-`:79`，单节点）或 hybrid warp 数（`:80`-`:85`），再用 DeepJIT 以模板参数**即时编译**
  （`:115`-`:121`），最后 cooperative launch（`:130`，`cluster_dim = 2 - (num_sms % 2)`）。
  注意所有形状（hidden、topk、专家数、对齐……）都是**编译期模板参数**，这是 DeepEP 能在设备侧
  把布局算成常量、避免运行期分支的原因。
- **重叠**：`async_with_compute_stream` + `defer_epilogue` 让通信 kernel 与计算流并行，
  真正需要结果时再 `.wait()` 跑 copy epilogue（README「Deferred epilogues」）；
  PDL（`cudaTriggerProgrammaticLaunchCompletion` + `cudaGridDependencySynchronize`）进一步把
  「主 dispatch」与「copy epilogue」重叠。这与 Mega MoE 在单 kernel 内做重叠是**同一种
  「通信掩计算」思想的不同粒度实现**（对照 `docs/02` §7）。

## 7. 与 Mega MoE / 目标卡的定位

- **DeepEP 侧**：目标平台「Hopper or newer」，V2.5 起纯 NCCL Gin backend（不再依赖 NVSHMEM），
  理论上 H100 可用；但**本机未安装** `deep_ep`（§8），故本文只做源码精读。
- **Mega MoE 主体**：DeepGEMM `sm100_fp8_fp4_mega_moe` 是 **SM100 专属，仅分析**；它把这套
  「对称 buffer + dispatch/combine」逻辑内联进单个 persistent kernel，并用 ring buffer 取代
  中间的 copy epilogue 物化。DeepEP 的 dispatch/combine 可视为其**非融合参照实现**。
- **冗余专家通信**：DeepEP 还提供 `lb_prefetch_weights` / `lb_reduce_grads`（`ep.py:854`/`:886`），
  前向经 NVLink 预取专家权重与量化 scale、反向把冗余专家梯度累加回原专家（README「Expert load
  balancing」），对应 MoonEP / UltraEP 思路——与 `docs/02` §4.4 的 locality-aware 调度互为补充。

## 8. 本机可行性（H100）

探测（`code/mega-moe/scripts/lab.sh exec`）：

```
torch 2.12.0a0, CUDA 13.2, nvcc V13.2.78   # 满足 DeepEP 的 CUDA 13.1+ / PyTorch 2.10+
python -c 'import deep_ep' -> ModuleNotFoundError: No module named 'deep_ep'
```

即**编译环境版本达标，但 `deep_ep` 未安装**。要跑真实 dispatch/combine：
需要 CUDA Toolkit 13.1+、NCCL ≥ 2.32.3、多卡 NVLink 域，并按 README 用 `bash install.sh`（走 DeepJIT）。
这些属于路线图 **P0-4（环境探测）**，本文不展开。因此本篇**不含实测数字**，属纯源码分析。

## 9. 小结与下一步

- DeepEP = **对称显存（NCCL window + 地址一致）→ 固定 token 布局 → cooperative warp 分工 kernel**；
  目标是产出 grouped GEMM 直接可用的 **expanded layout**。
- dispatch 有 notify（全局计数 + 跨 rank 握手）与搬运（TMA/NVLink/RDMA）两类 warp；
  combine 是逆过程并做 bf16 加权归约；两者互为反向。
- 与 Mega MoE 的关系：同一语义的**非融合参照实现**，Mega MoE 把它内联进单 kernel 并用 ring buffer 解耦。
- **下一步**：P0-4 环境探测（确认 H100 上能编译/跑哪些 kernel，是否可以装 DeepEP 跑通单/多卡
  dispatch/combine）；随后 P1/P3 继续 Mega MoE/反向。
