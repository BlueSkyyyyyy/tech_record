# DeepEP 的 EPLB 冗余专家：权重预取与梯度归约（Mega MoE 反向侧的通信原语）

> 本文对应路线图 **P43-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 53 篇。
> 目标：把 DeepEP 里与 **动态专家复制（EPLB / redundant experts）** 配套的两个通信原语
> —— `lb_prefetch_weights`（前向：把本 rank 的专家权重推到各 peer 的冗余槽）与
> `lb_reduce_grads`（反向：把各 peer 冗余副本的梯度累加回 owner 专家）——从 host 编排到
> 负载均衡切块讲透。
>
> **与专题的关系**：`docs/07`（P0-3）讲了 DeepEP 的 dispatch/combine；`docs/04/12/13/14`（P4）
> 讲了 MoE 反向的四个 grouped GEMM。这两者之间有一条**反向通信**的空档：在 EP 里为了负载均衡
> 会把热门专家复制到多个 rank（每个副本都会分到 token、都要算梯度），训练反向必须把这些**副本的
> FP32 梯度搬回来求和**，同时前向要先把权重**广播到副本**。这正是本文的两个 primitive。
>
> **可跑性**：它们**不是 SM100 专属**——kernel 只是 TMA + NVLink 对称指针，Hopper 即可。
> 但它们用 **NCCL 设备 API（`ncclDevComm` / `ncclGin`）**，本机系统 NCCL 2.29.7 的头不提供该
> API（与 `docs/07` §8 同源阻塞），因此**两个 kernel 仅做源码分析**；而它们的**负载均衡切块逻辑、
> host 公式、归约语义架构无关**，在 H100(sm90) 上以真实设备代码 + 纯张量实测。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - Python API：`deep_ep/buffers/ep.py`（`lb_get_theoretical_num_sms:846`、`lb_prefetch_weights:854`、`lb_reduce_grads:886`）
> - host 编排：`csrc/buffers/ep.hpp`（`lb_prefetch_weights:984`、`lb_reduce_grads:1044`）；
>   launch 壳 `csrc/kernels/ep/prefetch_weights.hpp`、`reduce_grads.hpp`
> - 负载均衡布局：`deep_ep/include/deep_ep/layout/ep/eplb.cuh`（`EPLBTask:15`、`build_tasks:29`、
>   `EPWeightIterator:91`）、`layout/ep/workspace.cuh`（`lb_chunk_idx_counter:21`）
> - 设备 kernel：`deep_ep/include/deep_ep/impls/ep/prefetch_weights.cuh`、`reduce_grads.cuh`
> - 测试：`tests/ep/test_prefetch_weights.py`、`tests/ep/test_reduce_grads.py`
> - 实验：`src/eplb_redundancy.cu` + `harness/exp_eplb_redundancy.{sh,out.txt}`（H100）

## 1. 问题：为什么要「复制专家」以及由此产生的通信

标准 EP 里每个 rank 持有 `num_local_experts = E / num_ranks` 个专家，token 按 top-k 路由到
专家所在 rank。现实中专家热度极不均匀：**热门专家所在 rank 既是计算瓶颈也是负载瓶颈**。
`README` §Expert load balancing（`:330-377`）描述的**动态专家复制**把热门专家的**副本**放到
同域其他 rank 的空闲槽上，让副本也参与计算，从而把负载摊平。这带来三件事：

1. **前向**：副本所在 rank 必须拿到主人的**权重（以及量化 scale）**——`lb_prefetch_weights`；
2. **计算**：由**上层训练框架**负责把 token 重路由到任意副本、跑专家 GEMM（DeepEP 不管）；
3. **反向**：每个副本都会算出自己那份梯度，必须**加回主人**——`lb_reduce_grads`。

`README:332` 指明动机来自 [MoonEP](https://github.com/MoonshotAI/MoonEP)（在线规划 + 动态冗余专家 +
权重预取 + 梯度归约）与 [UltraEP](https://github.com/Dots-Infra/UltraEP)（[论文](https://arxiv.org/abs/2606.04101)，
按 post-gating 负载做复制与 token 重路由、把复制限制在 NVLink 域内）。所以这两个 primitive 只在
**用户自己已实现复制计划与 token 重路由**的前提下才用得上。

### 1.1 冗余映射 `redundancy_mapping` 的数据模型

所有 rank 共享一张 `[num_nvlink_ranks, num_redundant_experts]` 的 int32 表（`README:341`、
`ep.py:872-874`）：

- 行 `r` = NVLink（LSA）域内第 `r` 个 rank，列 `c` = 该 rank 的第 `c` 个冗余槽；
- 元素 `mapping[r][c]` = 放在该槽里的**全局专家 id** `owner_rank * num_local_experts + local_expert_idx`，
  或 `-1` 表示空槽；
- 约定**一个 rank 的冗余槽里不会放它自己的专家**（避免自我交换），空槽跳过。

关键点：**每个 rank 都需要对整张表做同一份判断**——「哪些槽里装着**我的**专家」。第 2 节的
`build_tasks` 就是干这个的。

## 2. host 编排层：从 API 到 launch

### 2.1 Python 侧

`EPBuffer.lb_prefetch_weights:854-883` / `lb_reduce_grads:886-905` 做三件事：把单张量包成 list、
按需取 `num_sms`、返回 `EventOverlap`（异步、在 comm stream 上，`wait()` 后才可用，`README:337`）。
其中 `num_sms` 的取值很有意思（`ep.py:43/65` 与 `846-852`）：

```python
num_sms = self.lb_get_theoretical_num_sms() if num_sms == 0 else align(num_sms, 2)
```

`lb_get_theoretical_num_sms:846-852` 是一个**解析式带宽模型**：它读 `get_nvlink_gbs()`
（`envs.py:208-235`，解析 `nvidia-smi nvlink -s` 的每条链路求和后乘 0.9 效率因子），再假设
每 SM 的 HBM 读/写带宽是常数 `180/45 GB/s`（`envs.py:194-204`，注释即 `TODO: architecture-specific`），
取

$$
\text{num\_sms} = \text{align}\Big(\max\Big(\Big\lceil\frac{\text{nvlink}}{180}\Big\rceil,
\Big\lceil\frac{\text{nvlink}}{45}\Big\rceil\Big), 2\Big)
$$

即**既不能读得比 HBM 快、也不能写得比 HBM 快**，取瓶颈侧，并对齐到偶数（`cluster_dim` 需要偶数，
见 §2.3）。注意注释 `ep.py:848-850` 明说这个估计**与冗余映射无关**：ring 流量/rank 偏斜可能先撞
链路瓶颈，所以它只是「SM 容量预算」的上界。

### 2.2 C++ 侧：缓冲区、契约

`csrc/buffers/ep.hpp` 的两个入口把 Python 参数翻成 device 描述符：

- `lb_prefetch_weights:984-1041`：支持**一对多**（`expert_weights` 与 `redundant_expert_weights`
  是等长非空序列，`README:334`）——所以**权重和量化 scale 可以一起搬、无需 dtype 转换**；
  `num_weights <= kNumMaxWeightEntries = 16`（`eplb.cuh:69`）。逐对检查维度/连续性/每专家字节数
  （`:1017-1026`），填成 `EPWeightList`。
- `lb_reduce_grads:1044-1075`：**只有一份** `[num_local_experts, hidden]` 的 FP32 专家梯度与
  `[num_redundant_experts, hidden]` 的 FP32 冗余梯度（`:1060-1061`），语义是**原地累加**
  （`README:335`：「initialize `expert_grads` with the local contribution, or zeros, before reduction」）。

两者共同的硬约束：每专家字节数 `num_bytes_per_expert % 32 == 0`（TMA 对齐，
`reduce_grads.hpp:42`、`compiled.cuh:67 kNumTMAAlignmentBytes = 32`）。冗余张量必须落在
**LB region**（`ep.py:252/273-274`、由 `BufferAllocator` 计划分配，`allocator.py:10-36`），
与 EP dispatch/combine 缓冲区分开。

### 2.3 launch：每 warp 一个 chunk 缓冲

`csrc/kernels/ep/{prefetch_weights,reduce_grads}.hpp` 是同一套骨架：

- **每个 warp 私有一块 fixed-size chunk 缓冲 + 一个 mbarrier**；
  prefetch 用 `num_chunk_bytes = 12 KiB`（`prefetch_weights.hpp:30`，因为一个 chunk 要**同时**做
  load 和 store），reduce 用 `24 KiB`（`reduce_grads.hpp:31`，只 load 进 smem、store 是
  reduce-add 出去）。
- `num_warps = (num_smem_bytes - sizeof(EPLBSharedMemoryLayout)) / (num_chunk_bytes + sizeof(mbarrier))`
  （`prefetch_weights.hpp:33-36`），`num_warps <= 32`；所以 **chunk 大小直接决定并发 warp 数**
  ——reduce 用双倍 chunk 是因为负载均衡下每 rank 要拉的字节数不同，**大 chunk 减少 TMA 次数**。
- `grid_dim = num_sms`、`cooperative = true`（集群内全 grid barrier）、
  `cluster_dim = dim3(2 - (num_sms % 2), 1, 1)`（偶数 SM → cs=2，奇数 → cs=1；
  `prefetch_weights.hpp:56`、`reduce_grads.hpp:61`）。这解释了 §2.1 为何要对齐到偶数。

## 3. 负载均衡切块：`layout/ep/eplb.cuh`

这是两个 primitive 真正的「脑子」，也是本机唯一能跑真实设备代码的部分。

### 3.1 从映射到任务表：`build_tasks`

`EPLBSharedMemoryLayout:21-67` 是一块 32 KiB 的 smem 布局，`build_tasks:29-64` 把
「全表扫描 + 只留我的」压成一张紧凑任务表：

```cpp
for (int base = 0; base < kNumRanks * kNumRedundantExperts; base += kNumThreads) {
    const int i = base + thread_idx;
    const int expert_idx = i < ... ? redundancy_mapping[i] : -1;
    const int is_local = expert_idx >= 0 and expert_idx / kNumLocalExperts == rank_idx;
    BlockScan(smem_workspace).ExclusiveSum(is_local, task_idx, num_iter_found_tasks);
    if (is_local) {
        task_idx += num_found_tasks;
        tasks[task_idx] = { expert_idx % kNumLocalExperts, i / kNumRedundantExperts, i % kNumRedundantExperts };
    }
    num_found_tasks += num_iter_found_tasks;
    __syncthreads();
}
```

四个「为什么」：

1. **`i = r * kNumRedundantExperts + c`**（行主序展开），于是 `i / D = r`（**目标 rank**）、
   `i % D = c`（**目标冗余槽**）。`expert_idx % L` 是**我的本地专家号**。
   所以一个 `EPLBTask {local_expert_idx, dst_nvl_rank_idx, redundant_expert_idx}` 精确描述
   「把我的第 `local_expert_idx` 个专家，推到 rank `dst` 的第 `red` 个冗余槽」。
2. **`expert_idx / L == rank_idx` 是「谁该推」**：只有专家的 **owner rank** 才会为它建任务；
   每个非空映射项因此**恰好被一个 rank 认领一次**（§5 Part A/D 实测无重复）。
3. **`cub::BlockScan` 的 ExclusiveSum 做块内压缩**，`num_found_tasks` 是跨 tile 的运行偏移——
   映射表可能比线程数长（`kNumRanks * kNumRedundantExperts > kNumThreads`），需要分块续接。
4. **顺序 = 映射表顺序**（`tasks[]` 按 `i` 递增排），故任务是**确定性**的。

容量：`kNumMaxTasks = (32768 - 4) / 12 = 2730`（`eplb.cuh:23`；`EPLBTask` 12 字节），
`EP_STATIC_ASSERT(sizeof(EPLBSharedMemoryLayout) <= 32768)`（`:67`；恰为 `4 + 2730×12 = 32764`）。
超过会触发 `EP_DEVICE_ASSERT(task_idx < kNumMaxTasks)`（`:49`）。

### 3.2 chunk 工作窃取：`EPWeightIterator`

`EPWeight/EPWeightList:71-87` 是描述符（`redundant_expert_weights`、`expert_weights`、
`num_bytes_per_expert`）。`EPWeightIterator:91-140` 是**全 rank 共享一个原子计数器的动态工作窃取**：

```cpp
int chunk_idx = atomicAdd(chunk_idx_counter, 1) - num_skipped_chunks;   // 领一块
while (weight_idx < kNumWeights) {                                     // 单调前进到所属权重
    const int num_chunks = ceil_div(weights[weight_idx].num_bytes_per_expert, kNumBytesPerChunk) * num_tasks;
    if (chunk_idx < num_chunks) break;
    chunk_idx -= num_chunks; num_skipped_chunks += num_chunks; weight_idx ++;
}
if (weight_idx == kNumWeights) return false;                           // 领完
const auto& [red_w, exp_w, B] = weights[weight_idx];
const auto& [local_e, dst_rank, red_idx] = smem_layout.tasks[chunk_idx % num_tasks];
const int64_t off = (chunk_idx / num_tasks) * kNumBytesPerChunk;
num_chunk_bytes = min(kNumBytesPerChunk, B - off);                     // 尾块裁剪
src_ptr = exp_w  + local_e * B + off;                                  // 本地侧
dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(red_w + red_idx * B + off, dst_rank);  // 远端侧
```

要点：

- **声明顺序 = `weight → chunk → task`**（注释 `:90`）：所有权重排成一个线性的 chunk 序列，
  每个权重的序列是 `num_tasks × ceil_div(B, CH)` 长，其中 `chunk_idx % num_tasks` 是**任务**
  （即一条专家→冗余槽的搬运动作）、`chunk_idx / num_tasks` 是**该专家内部的 chunk 序号**。
  于是同一专家的 chunk 在序列里**连续**，`off` 单调，cache 友好。
- **`atomicAdd` 返回旧值**：多 warp / 多 SM 在同 rank 内自发抢块，无需 host 静态划分——
  这就是「负载均衡」二字：谁的 warp 空谁多领。
- **`num_skipped_chunks`** 让 `chunk_idx` 在跨权重时重新基准化；权重推进是**单调**的，不会回头。
- **`src_ptr` 恒为本地、`dst_ptr` 恒为远端**（`gin.get_sym_ptr<Lsa>` 做 LSA 对称指针翻译，
  见 `docs/11` 的 `SymBuffer::map` 语义）。两个 primitive 共用这个 iterator，只把
  「读哪侧、写哪侧」反过来（§4）。
- **尾块裁剪**：`ceil_div` 计入不整除的最后一块，`min(CH, B - off)` 给出真实字节数；
  下界是每专家字节数必须是 32B 的倍数，而 chunk 是 12/24 KiB，尾块自然仍是 32B 对齐。
- 计数器 `lb_chunk_idx_counter` 在**对称 workspace** 里（`workspace.cuh:21`，经
  `EPWorkspaceLayout::get_lb_chunk_idx_counter_ptr():109-111`），每 rank 一份；kernel 开头由
  `sm_idx==0 && warp==0` 清零（`prefetch_weights.cuh:59-61`）。

## 4. 两个设备 kernel：一对镜像

`reduce_grads.cuh:24-93` 与 `prefetch_weights.cuh:25-93` 是**逐行对称**的骨架
（同参数、同 smem 划分、同 barrier 结构），差别只在 elect-one 循环里的两行：

| | `lb_prefetch_weights`（前向） | `lb_reduce_grads`（反向） |
|---|---|---|
| `EPWeightIterator` 模板 | `<kNumWeights, CH>`（多权重：权重 + scale）`prefetch:72` | `<1, CH>`（单权重：专家梯度）`reduce:71` |
| 第 1 步 | `tma_load_1d(buf, iterator.src_ptr)`：**从本地读**专家权重 `prefetch:78` | `tma_load_1d(buf, iterator.dst_ptr)`：**从 peer 读**冗余梯度 `reduce:77` |
| 第 2 步 | `tma_store_1d(iterator.dst_ptr, buf)`：**写 peer** 冗余槽（覆盖）`prefetch:82` | `tma_store_reduce_add_f32(iterator.src_ptr, buf)`：**累加回本地**专家梯度 `reduce:82` |
| 净效果 | 本地专家权重 **分发**到各 peer 冗余槽 | 各 peer 冗余梯度 **归约**回本地专家梯度 |

两端用 `ptx::mbarrier_arrive_and_set_tx` + `tma_load_1d` + `mbarrier_wait_and_flip_phase`
做「load→等→store」单缓冲流水（`prefetch:76-79`、`reduce:76-79`），再
`tma_store_commit` + `tma_store_wait_read<0>` 保证 smem chunk 用毕可复用
（`prefetch:83-84`、`reduce:83-84`）。前后各有一次 `comm::gpu_barrier`（entry/completion，
`prefetch:65-67/90-92`、`reduce:65-67/90-92`）——因为这两步是**域内集合操作**：即便某 rank
没有分到任何冗余槽，也必须参与 barrier（`README:337`）。

**为什么 reduce 必须用 `cp.reduce.async.bulk`（TMA reduce-add）而不是 load+add+store**：
多个 peer 可能同时把梯度加到同一个 owner 专家上，reduce-add 是**元素级原子加**（`cp.reduce.async.bulk`
with add），**与顺序无关**；测试正是靠这一点用整数 pattern + `torch.equal` 做**逐位验收**
（`test_reduce_grads.py:31-33`：「order-independent … lets the check use `torch.equal`」）。

## 5. H100 实测（真实设备代码）

`harness/exp_eplb_redundancy.sh` 编译运行 `src/eplb_redundancy.cu`（**逐行原样抽取**
`build_tasks` 与 `EPWeightIterator::next`；后者唯一的改动是把 `gin.get_sym_ptr<Lsa>` 换成
语义等价的对称指针桩 `offsets[dst]+ptr`，与 DeepGEMM `layout/sym_buffer.cuh:33-40` 一致），
外加纯 Python 复刻 host 公式。原始输出见 `harness/exp_eplb_redundancy.out.txt`。

- **Part A — `build_tasks` 压缩**（`EPLBSharedMemoryLayout::build_tasks` 真实设备代码）：
  16 组配置 **0 不符**。覆盖 `ranks×local×redundant ∈ {(8,4,16),(8,4,40),(4,8,8),(16,2,32)}`、
  多 rank、线程数 128/256/512、**映射长度 > 线程数**（8×40=320 > 128，走分块续接）、
  含 `-1` 空槽、全空（tasks=0）、全本地（rank 2 得 32 个任务）、末 rank 15。
  验证了任务表的**内容、顺序、数量**都与 host 参考一致。
- **Part B — `EPWeightIterator` 切块**（真实设备代码 + 对称指针桩）：配置
  `num_weights=3, num_tasks=7, bytes={4596,2048,3079}, CH=1024`，**4 个 warp 并发认领**。
  参考 chunk 总数 **77**，设备认领 **77**，`src/dst` 偏移三元组与主机参考**逐条一致**（YES）；
  非整除尾块 `bytes[0]%CH=500` 被正确裁成 500 字节。**全局 counter 收尾于 81**——因为
  `atomicAdd` 永远加一，4 个 warp 在「已领完」后又各空领了一次（77+4），正好坐实 iterator
  的**单调**收尾语义（`weight_idx == kNumWeights` 立即返回 false）。
- **Part C — host 公式**（真实 `nvidia-smi nvlink -s`）：H100 上 `get_nvlink_gbs()`（0.9×）
  = **430.3 GB/s**；`ceil(430.3/180)=3`（读侧）、`ceil(430.3/45)=10`（**写侧瓶颈**）、
  `align(10,2)=` **10 SMs** = 目标卡理论 LB SM 数。附带验证 `num_sms` 参数对齐：
  `1→2, 3→4, 7→8, 12→12, 13→14, 0→10（理论值）`。
- **Part D — 归约语义**（纯 tensor）：用 `build_tasks` 复刻 `lb_reduce_grads`
  「副本梯度累加回 owner」，与参考（owner 贡献 + 所有指向它的副本贡献）**relerr 0.000e+00**；
  **反例**（写成覆盖而非累加）relerr **7.822e-01**——坐实「必须累加」。另验证 `build_tasks`
  领走的冗余槽（10 个）恰好等于非空映射项（10 个）且**无重复**。

对应上游测试（`tests/ep/*.py`）用了更硬的判据：`make_pattern`（`test_reduce_grads.py:23-40`）
造**整数斜坡**（周期 4095、按 key 平移），使 FP32 求和**精确无舍入**，于是可用 `torch.equal`
逐位比较（`:192/202`）；hidden 覆盖完整块、尾块与缓冲区复用（`:20` 的
`(1<<20, 1<<18, 16392, 6144, 3080, 3072, 3064, 5120, 1536, 384, 24)`）；性能项用
`ring`/`spread` 两种映射把「per-peer 链路瓶颈」与「聚合 NVLink 瓶颈」分离（`:64-75`）。

## 6. 小结与边界

- 这两个 primitive 是 **EPLB / 动态冗余专家** 的通信底座：**前向推权重、反向收梯度**，
  一对镜像，且只负责通信，token 重路由与专家 GEMM 由框架自管（`README:337`）。
- 架构无关的三层：**host 公式**（带宽模型 → SM 数）、**负载均衡切块**（`build_tasks` 压缩 +
  `EPWeightIterator` 原子窃取）、**归约语义**（reduce-add 顺序无关）。这三层在 H100 上实测。
- **两个 kernel 用 NCCL 设备 API，本机 NCCL 2.29.7 不提供 → 仅源码分析**（与 `docs/07` §8、
  ROADMAP「阻塞」同源）。修好 NCCL ≥ 2.32.3 的 PyTorch 环境后可做端到端 8×H100 实测。
- 链接：[DeepEP](https://github.com/deepseek-ai/DeepEP)、[MoonEP](https://github.com/MoonshotAI/MoonEP)、
  [UltraEP](https://github.com/Dots-Infra/UltraEP)、[UltraEP 论文](https://arxiv.org/abs/2606.04101)。
