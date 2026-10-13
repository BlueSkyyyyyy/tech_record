# DeepEP 的流水线并行 send/recv 原语 `pp_send_recv`

前面第 7/58/61 篇拆了 DeepEP 的 **EP all-to-all**（dispatch / combine），第 64 篇拆了与它并列的
**bucket 集合通信**（all_gather / reduce_scatter / all_reduce）。本篇补上 DeepEP 通信设施里
**第三个、也是最少被提及**的原语：**流水线并行（pipeline parallelism）的相邻 rank send/recv**，
`deep_ep.PPBuffer` / `deep_ep::pp::pp_send_impl` / `pp_recv_impl`
（`csrc/kernels/pp/pp_send_recv.hpp`、`deep_ep/include/deep_ep/impls/pp/pp_send_recv.cuh`）。

它服务的是训练里「把一层的激活/梯度在流水线相邻 stage 之间前后传递」这一步：每个 rank 只跟
**环上的前后两个邻居**说话，因此不需要 all-to-all 的计数/前缀和，也不需要 bucket 的算法选型——
它要解决的是另一个问题：**用 TMA 把任意长度张量高效搬到对端，并用「在途张量数（inflight）」把
收发做成有背压的环形流水**。

> **可跑性**：PP 的**数据面**（`pp_tma_copy` 的 TMA 1D 两级流水）与**寻址面**
> （`get_pp_buffer_offset` + 4 段对称缓冲的 block / arrival / release 索引）都是架构无关的；
> 只有数据面的**第二跳** `comm::NCCLGin::put`（push 到对端 recv 槽）与 `NCCLGin::signal`
> （`StrongVASignalAdd` 通知对端）需要 `nccl_device.h` 的 GIN 设备 API（本机 NCCL 2.29.7 不提供）
> ——这部分**仅源码分析**。本篇在 **H100（sm90）真机**跑真实设备代码验证数据面与寻址面。

---

## 1. 它解决什么问题：带背压的环形流水

流水并行的第 `r` 个 stage 要把张量 `x` 发给第 `r+1` 个 stage，同时从第 `r-1` 个 stage 收张量。
最朴素的实现是「配一对 `ncclSend/Recv`」，但那样：
1. 发出去的张量不能立刻复用它的显存——必须等接收方确认收到；
2. 相邻两次发送如果接收方没消费，会覆盖还没被读走的数据；
3. 想要「发送与计算重叠」，就必须让「发送」是一个**可以异步推进、带在途配额**的操作。

DeepEP 的做法是把每对相邻 rank 之间的传输组织成**定长槽（slot）的环形缓冲**：
每个方向最多允许 `num_max_inflight_tensors` 个张量同时在途；发送方第 `i` 次发送只有在
「接收方已消费到第 `i - F` 个」之后才能开始——这就是**背压**，也是「短张量不发、长张量流水」的由来。

配置面（`deep_ep/buffers/pp.py:12-67` 的 `PPBuffer`、`csrc/buffers/pp.hpp`）：

- `num_max_tensor_bytes`：单个 send/recv 张量的字节上限，构造时按 `kNumRDMAAlignmentBytes = 64`
  对齐（`pp.hpp:45`）；
- `num_max_inflight_tensors = F`：在途张量数上限；
- 存储 `num_storage_bytes = align(4 * num_max_tensor_bytes * F, kNumAllocationAlignmentBytes)`
  （`pp.hpp:47-49`）——**4** 是因为每个 rank 同时有两个角色（收 + 发）× 两个邻居，
  即 4 段缓冲（下面第 3 节）；
- `send`/`recv` 都断言目标必须是环上的**相邻 rank**（`pp.hpp:86-88`、`:103-105`），
  且张量 32B 对齐、字节数是 32 的倍数、不超过上限（`:83-85`、`:100-102`）。

host 侧发射（`csrc/kernels/pp/pp_send_recv.hpp`）：

- `get_num_qps`（`:20-23`）= `min(2, context.num_allocated_qps)`——最多分到 **2 个 QP**
  （RDMA 带宽一般 2 条 QP 够）；
- kernel 用 DeepJIT 现编（`:38-44`），模板实参是
  `<num_sms, num_ranks, num_smem_bytes, num_qps, num_gpu_timeout_cycles>`；
- 发射用 **cooperative launch**（`.cooperative = true`，`:53`），因为 kernel 里要
  `cooperative_groups::this_grid().sync()`；
- `kNumThreads = 32`（`:32`）——**整个 kernel 只有一个 warp**，所有重活交给 TMA 与 GIN。

---

## 2. 数据面：`pp_tma_copy` —— 一段可复用的多 SM TMA 流水

`pp_tma_copy`（`pp_send_recv.cuh:27-102`）是 PP 的数据搬运引擎，把 `num_bytes` 从 `src_ptr`
搬到 `dst_ptr`：

1. **共享内存布局**（`:37-41`）：前 `kNumStages * kNumTMABytesPerStage` 字节是 TMA 暂存，
   之后放 `kNumStages` 个 mbarrier。`kNumTMABytesPerStage` 由总 smem 减去 mbarrier 后
   **均分并对齐到 32B**（`:30-33`）。默认 `kNumStages = 2`，静态断言要求 ≥ 2。
2. **跨 SM 分区**（`:50-55`）：把总字节切成 `num_tma_blocks = num_bytes / 32` 个 32B 块，
   每个 SM 分连续的一段 `[start_block_idx, end_block_idx)`——**每个 SM 负责一段连续字节区间**，
   互不重叠。`num_tma_blocks_per_sm = ceil_div(total, kNumSMs)`（`:52`）。
3. **两级流水**（`:65-98`）：先为前 `kNumStages` 个迭代发 `tma_load_1d` 并
   `mbarrier_arrive_and_set_tx`（`ptx::tma_load_1d` + `mbarrier::complete_tx::bytes`，
   `common/ptx.cuh:122-134`、`:65`）；稳态里对每个迭代「等本 stage 的 load（
   `mbarrier_wait_and_flip_phase`）→ 发 `tma_store_1d` → `tma_store_commit`」，随后
   `tma_store_wait()` 确认这个 stage 的 store 落盘后，再往同一 stage 发下一个 load（`:87-97`）。
   收尾 `tma_store_wait()` 排空所有在途 store（`:101`）。
4. **单线程发射**：调用点在 `pp_send_impl` 里被 `if (ptx::elect_one_sync())` 包住
   （`cuh:133`、`:142`）——TMA 与 mbarrier 都是单线程发起的。

为什么用 TMA 而不是 `__ldg`/`st.global`：TMA（`cp.async.bulk`）**不需要寄存器中转**，
不占用通用寄存器和 LSU 带宽，把整段搬运变成「一条指令 + 一个 mbarrier」，让这唯一一个 warp
几乎空出来等 GIN；同时 `tma_store_wait` 天然给出「stage 已可复用」的同步点。
（这套 TMA 原语本身在第 59 篇已单独拆过。）

---

## 3. 寻址面：`get_pp_buffer_offset` 与 4 段对称缓冲

发送方与接收方地址都必须落在**对称窗口**（第 66 篇）的同一逻辑 offset 上。PP 把每个 rank 的
storage 切成 **4 段**，每段 `F` 个槽、每槽 `num_max_tensor_bytes` 字节：

```
block:   0                1                2                3
role:  收-从上一个(prev)  收-从下一个(next)  发-给上一个(prev)  发-给下一个(next)
```

方向的编码由 `get_pp_buffer_offset<kNumRanks>(src, dst)`（`cuh:20-25`）给出，返回
`(local_idx, peer_idx)`：

```
next = (src + 1) % kNumRanks
if dst == next:  return (0, 1)   # dst 是「下一个」：本方 recv half=0，对端看它的 half=1
else:            return (1, 0)   # dst 是「上一个」：本方 recv half=1，对端看它的 half=0
```

注释（`cuh:17-19`）说得很清楚：`local_idx` 是**本方**为这个 peer 准备的缓冲 half，
`peer_idx` 是**对端用来寻址同一 half 的对称索引**——因为对称窗口里「同 offset 即同逻辑位置」，
两边必须用对方的编号。

由此推出的所有索引（设备代码与本篇 oracle 逐字段对拍）：

| 量 | 表达式 | 出处 |
|---|---|---|
| recv block | `src_idx_in_local` (=`local_idx`) | `cuh:199-200` |
| send staging block | `dst_idx_in_local + 2` (=`peer_idx + 2`) | `cuh:127-128` |
| put 的对称目标 block | `local_idx_in_dst` (=`local_idx`) | `cuh:129-130` |
| send 的槽号 | `send_count[peer_idx] % F` | `cuh:125-126` |
| recv 的槽号 | `recv_count[local_idx] % F` | `cuh:197-198` |
| arrival 索引 | `[local_idx][qp]`（收发两侧一致） | `cuh:156` / `:207` |
| release 索引 | `peer_idx`（收发两侧一致） | `cuh:136` / `:229` |

关键不变式：**发送方 staging 的对端对称目标 block == 接收方的 recv block**
（都是 `local_idx`），且 arrival/release 索引收发两侧一致——发送方把 arrival 加到对端的
`arrival[local_idx]`，接收方就在自己的 `arrival[local_idx]` 上等；接收方写 `release[peer_idx]`，
发送方就在自己的 `release[peer_idx]` 上等。

四个方向恰好占满 `{0,1,2,3}`：recv-from-prev=0、recv-from-next=1、send-to-prev=2、
send-to-next=3（第 5 节实验逐 rank 验证）。

`PPSignals`（`layout/pp/workspace.cuh`）就是这些计数器的容器：

```c
struct alignas(kNumRDMAAlignmentBytes) PPSignals {   // 64B 对齐
    BarrierSignals barrier;      // 复用 comm barrier 的双相位屏障
    int64_t send_count[2];       // 按方向（peer_idx）
    int64_t recv_count[2];       // 按方向（local_idx）
    int64_t arrival[2][2];       // [方向][QP]
    int64_t release[2];
};
```

---

## 4. 协议：两跳 send / recv（第二跳为 GIN，仅分析）

### `pp_send_impl`（`cuh:106-171`）

1. 算方向与索引（`:120`），取 `send_count` 与槽号（`:125-126`）；
2. **背压等待**（`:133-141`）：在 `release[peer_idx] >= send_count - F + 1` 之前自旋
   （`ptx::ld_acquire_sys` + `comm::timeout_while`）。也就是「第 `i` 次发送要等对端消费完第
   `i - F` 个」；超时打 `DeepEP PP send timeout, recv buffer is full`。
3. **第一跳**：`pp_tma_copy(x → send_buffer)`（`:142`）把用户张量搬进本地 staging 槽；
4. `this_grid().sync()`（`:144`）：**等所有 SM 都把自己的那段搬完**，才能发第二跳——
   因为 `put` 由个别 SM 按 QP 发出；
5. **第二跳**（`:146-170`）：按 QP 切分字节（`num_bytes_per_qp`，`:149`），每个分到的 SM
   用 `NCCLGin::put` 把 staging 的一段 push 到对端 `recv_buffer + offset`，并附带
   `ncclGin_StrongVASignalAdd` 把对端的 `arrival[local_idx][qp]` **原子 +1**（`:154-166`）；
   即使某 QP 没有数据也要 `gin.signal`（接收方每个 QP 都会等，`:159-166`）；
6. `sm_idx == 0` 才 `send_count[peer] += 1`（`:168-169`）。

### `pp_recv_impl`（`cuh:175-234`）

1. 算方向与索引（`:190`），取 `recv_count` 与槽号（`:197-198`）；
2. **等到达**（`:203-216`）：`kNumQPs` 个线程各等一个 QP 的 `arrival[local_idx][qp] >= recv_count + 1`；
3. **第一跳（反向）**：`pp_tma_copy(recv_buffer → x)`（`:218-220`）把对端写进来的槽 TMA 拷到
   用户张量；
4. `this_grid().sync()`（`:221`）；
5. **释放**（`:224-233`）：`sm_idx == 0` 用 GIN 向对端 `release[peer_idx]` `StrongVASignalAdd +1`，
   并把 `recv_count[local_idx] += 1`。

### 为什么是两跳（本地 staging 再 RDMA）

- 用户张量 `x` 未必在对称窗口里、也未必允许被远端直接 put；先把 `x` TMA 到**对称的 staging 槽**
  后，第二跳就只依赖「对称 offset」，与用户内存布局解耦。
- staging 槽复用靠 `tma_store_wait` + `grid sync` 保证：本 SM 的 store 落盘后全体对齐，才发 RDMA。

### 为什么用 `StrongVASignalAdd` 而不是普通写

arrival/release 是**单调计数器**，多个 QP、多个 SM 会并发地加；用 `VASignalAdd` 的原子加
既完成「通知」又完成「计数」，无需读改写；配合对称窗口的 `get_sym_offset` 只传一个 offset
（`handle.cuh:58-61`）与一个增量。

### 为什么必须 cooperative launch

两处 `this_grid().sync()` 是**跨 SM 的全局屏障**（合并 32 个 SM 的分段搬运 / 汇总 release），
只有 co-resident 的 cooperative grid 才能安全地全网格同步（第 52 篇的 PDL/grid-sync 同源）。

---

## 5. H100(sm90) 真机实验

实验见 `src/deepep_pp_send_recv.cu` + `harness/exp_deepep_pp.{sh,py}`（输出 `exp_deepep_pp.out.txt`）。
把 `get_pp_buffer_offset`（`cuh:20-25`）与 `pp_tma_copy`（`cuh:27-102`）**逐行原样抽取**
（片段 sha256：`c3a3b0d1817b3bcb` / `ef702e38dd6caa81`，与 `~/github/DeepEP@93eb6eb` 一致），
直接 include 真实 DeepEP 头文件跑真实设备代码。

**（1）设备端索引映射（Part A）**：对 `N = 2..8` 的**全部 168 个有序对 `(src,dst)`**，
用编译进 kernel 的 `get_pp_buffer_offset` 算出 send/recv 两侧的 12 个导出字段，
与 Python 独立复刻**逐字段 0 不符**；并验证不变式：每个 rank 的 4 段缓冲 `{0,1,2,3}`
被 recv×2 / send×2 **恰好各占一次**，`put` 的对称目标 == 接收方 recv block、
arrival/release 两侧索引一致。
（`N == 2` 时唯一邻居同时是 prev 与 next，两方向退化成同一 `(0,1)`，4 段假设不成立——
这也是 PP 实际要求 `num_ranks > 1` 且真为环形流水的边界。）

**（2）数据面 TMA 搬运（Part B）**：`pp_tma_copy` 在 5 种 `num_sms ∈ {1,2,4,8,32}` ×
8 种字节数 `{32, 512, 4096, 12288, 65536, 100000, 1048576, 159968}` 共 **40 组全部 byte-exact**
（`mismatch=0`）。其中字节数故意跨越 `kNumTMABlocksPerStage`（96KB smem、两级流水下约 1535 块/stage）
的整数倍，验证跨 stage 边界、跨 SM 分区边界的正确性。
真机踩到一个只有实跑才暴露的点：**动态共享内存 96KB 必须先 `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize)`**，
否则 launch 静默失败——初始版本 40 组 `mismatch` 全是「dst 保持初值」，补上 opt-in 后全过。

**（3）`PPSignals` 布局（Part C）**：`sizeof(PPSignals) = 192`、`alignof = 64`，
偏移 `barrier=0 / send_count=64 / recv_count=80 / arrival=96 / release=128`；
`BarrierSignals = 64B`（counter 8 + 2×int 4，补齐到 64）。全部落在 16 MiB 信号区
（`kNumMaxSignalBytes`）内且按 RDMA 64B 对齐。

**（4）调度不变式（Python）**：**逐行抽取** `tests/pp/test_pp.py:19-50` 的
`generate_stress_ops`（真实源码，sha256 `56d5e37722ddc497`）运行 6 组
`(R, S, F) = (2,16,4)/(4,64,4)/(8,128,8)/(4,128,1)/(4,128,2)/(8,200,16)` × 4 个种子，
全部满足：**每条链路上第 `i` 个 send 严格晚于第 `i - F` 个 recv**（槽已释放的充分条件）、
`sends == recvs` 计数、`ops` 按 `(timestamp, index)` 全序。这坐实了 host 测试生成器的调度
正是 kernel 背压条件 `release >= send_count - F + 1` 的一条**安全持序**。

**（5）缓冲几何（`pp.hpp:45-49`）**：`T = 4096×7168×2`（bf16 满张量）、`F = 4` 时
storage = **896 MiB**；`F = 16` 时 **3584 MiB**——解释 `num_max_inflight_tensors` 是显存
与延迟隐藏的权衡旋钮。`num_max_tensor_bytes` 按 64B 对齐保证每个槽基址 64B 对齐
（TMA 32B / RDMA 64B 的共同要求）。

设备汇总 **PASS=4 / FAIL=0**，Python oracle **PASS=6 / FAIL=0**。

---

## 6. 小结与边界

- PP send/recv 是 DeepEP 里与 EP all-to-all、bucket 三足鼎立的第三类通信原语：**环形、相邻、
  带背压、可重叠**。它的关键不是算法（没有计数/前缀和），而是**几何与协议的正确性**：
  对称窗口里 4 段缓冲的对称寻址 + monotonic arrival/release 计数器 + 两跳 TMA/GIN。
- **可跑**：`get_pp_buffer_offset` 的整数映射、`pp_tma_copy` 的多 SM TMA 两级流水、
  `PPSignals` 布局、host 调度不变式——本篇已用 H100 真机坐实。
- **仅分析**：`NCCLGin::put` / `signal` 的 GIN 第二跳（`handle.cuh:171-217`）、
  `PPBuffer::destroy` 里销毁前的 `comm::barrier` 收尾（`pp.hpp:113-123`）——需要 NCCL ≥ 2.32 的
  设备 GIN API，本机不提供（见 ROADMAP「阻塞」）。真实多 rank 端到端 send/recv 与性能数字
  （`tests/pp/test_pp.py` 的 `bench_kineto` 部分）待环境就绪后再补。
