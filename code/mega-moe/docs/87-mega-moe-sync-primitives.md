# DeepGEMM 的跨 SM / 跨 rank 同步原语：`comm/barrier.cuh`

> 第 11 篇在拆 dispatch / combine 时，用一张「原语表」列了 `comm::grid_sync` 与
> `comm::nvlink_barrier` 两行（`comm/barrier.cuh:33-51` / `:53-92`）；第 38 篇拆 PTX 底座时，
> 又把它们当作 `ptx::` 内存序原语的**调用点**引用（内存序表里的 `barrier.cuh:42/45/75/82`），
> 并明确写了「`grid_sync` 的整体语义因共享 GPU 会死锁，仅源码解读」。但这两行一直没被
> **单独**拆开讲过：它们到底怎么做到「一类是全 SM 的 barrier、另一类要跨 8 张卡」，
> 各自的计数/相位是怎么设计成**免复位、可无限复用**的。
>
> 本篇补上这层。它是 Mega MoE 里所有「通信-计算重叠」正确性的地基——
> `sm100_fp8_fp4_mega_moe.cuh` 里 4 处 `grid_sync`、2 处 `nvlink_barrier`、1 处 combine
> 就绪轮询都踩在它上面。**代码全是 `__device__` 内联、架构无关**（不碰 UMMA / TMA / 2-SM），
> 因此可以在 H100(sm90) 上**直接 include 真实头文件、编译运行真实设备代码**逐条验证，不用
> 再「仅源码解读」。

---

## 1. 一段话总览

`comm/barrier.cuh`（94 行）只提供四个东西：

| 名字 | 行号 | 作用 | 谁在用 |
|---|---|---|---|
| `wait_until` | `:14-24` | 带 60 秒超时的自旋等待 | 所有轮询（grid/NVLink/combine 就绪） |
| `cluster_sync_with_relaxed_arrive` | `:26-31` | cluster 内 barrier，用 `arrive.relaxed` 换速度 | 2-CTA tensor memory 分配前 |
| `grid_sync` | `:33-51` | **单 rank 内全 SM** barrier | dispatch 写源索引后、combine 写 L2 后 |
| `nvlink_barrier` | `:53-92` | **跨 rank** barrier（= grid_sync + NVLink 信号 + grid_sync） | dispatch 写计数后、清 workspace 后 |

外加一个被它们依赖的设备端结构：`layout/sym_buffer.cuh` 的 `SymBuffer`（把本地地址翻译成
对端 rank 的对应地址）。

三个设计要点贯穿全文，也解释了「为什么」：

1. **用原子累加编码「到达」而不是有序遍历**：`grid_sync` 用 `atom.release.gpu.add`，
   `nvlink_barrier` 的信号用 `red.release.sys.add`——都不需要按任何顺序、都能无锁并发。
2. **用「和恰好是 2 的幂」来免复位**：`grid_sync` 每轮让计数器加 `2^31`（mod 2^32），
   只翻转最高位、两轮回到原值，所以不需要在每轮前清零；`nvlink_barrier` 用「+1 加满、-1 减回」
   让信号在两个 slot 间轮转、也自动归零。
3. **跨设备必须 `sys` 作用域**：NVLink 信号和 combine 就绪用 `ld.acquire.sys` / `red.release.sys`，
   因为它们要跨 rank 排序（对照第 38 篇内存序阶梯）。

---

## 2. 它在 Mega MoE 里的位置

`sm100_fp8_fp4_mega_moe.cuh` 里的调用点（行号已核对）：

| 调用 | 行号 | 作用 |
|---|---|---|
| `cluster_sync_with_relaxed_arrive()` | `:238` | 2-CTA tensor memory 分配前的 cluster 同步 |
| `grid_sync<kNumSMs, kDispatchGridSyncIndex>` | `:393-396` | dispatch 各 SM 写完源索引 → 全 SM 才继续 |
| `nvlink_barrier<..., kBeforeDispatchPullBarrierTag>` | `:422-428` | 所有 rank 的源索引都写好，才开始远端 pull |
| `nvlink_barrier<..., kAfterWorkspaceCleanBarrierTag>` | `:683-689` | 所有 rank 清完 workspace |
| `grid_sync<kNumSMs, kEpilogueGridSyncIndex>` | `:1412-1415` | 所有本地 L2 写完成 |
| `wait_until(... ld_acq_sys(peer_ready_ptr) == grid_idx ...)` | `:1440` | combine 逐 token 等对端 L2 就绪 |

两个 grid-sync 用**不同计数器**（`kDispatchGridSyncIndex=0` / `kEpilogueGridSyncIndex=1`，
`:338-339`），因为同一个 kernel 里 dispatch 段与 epilogue 段可能并发推进，不能共用一个计数。
两个 NVLink barrier 用**不同 `kTag`**（`:323-324` 的 `kBeforeDispatchPullBarrierTag=1` /
`kAfterWorkspaceCleanBarrierTag=2`），用于超时打印时区分是哪一次 barrier 卡住。
BF16 变体是同一套骨架（`sm100_bf16_mega_moe.cuh:193/348/377/619/1200`）。

这些信号都住在 `layout::MegaMoESignals`（`layout/mega_moe.cuh:54-90`）里：

```
:55   static constexpr uint32_t kNumMaxGridSyncCounters = 4;
:62   uint32_t grid_sync_count[kNumMaxGridSyncCounters];   // 4 个独立的 grid barrier 计数
:63   uint32_t nvl_barrier_counter;                        // NVLink barrier 的状态字（低 2 位）
:64   int      nvl_barrier_signals[2];                     // 两个信号 slot（有符号，因为会 -1）
:91   DG_STATIC_ASSERT(offsetof(MegaMoESignals, combine_ready_grid_idx) == 128, ...);
```

`Workspace` 给出取址封装：`get_grid_sync_count_ptr<kIndex>()`（`:151-156`）、
`get_nvl_barrier_counter_ptr()`（`:158-161`）、`get_nvl_barrier_signal_ptr(phase)`（`:163-167`，
注释「the signal is signed, as we may minus」）。

---

## 3. `SymBuffer`：设备端对称指针翻译

跨 rank 寻址的最小抽象（`layout/sym_buffer.cuh`，44 行）。每个 rank 上有一片**同样布局**的
对称缓冲；`SymBuffer` 只存三样东西（`:11-13`）：

```
int64_t  base;                    // 本 rank 的基址
int64_t  offsets[kNumMaxRanks];   // offsets[i] = rank_i 基址 - 本 rank 基址
uint32_t rank_idx;                // 本 rank 号
```

构造函数（`:19-25`）把 `c[rank_idx]` 当 base、其余存成**相对偏移**。翻译就一行（`:33-40`）：

```
mapped_ptr = offsets[dst_rank_idx] + (int64_t)ptr;
```

即「把本地指针加上对端基址与本地的差」，得到**对端 rank 上的同一个逻辑位置**。
两个细节：

- `kNumRanks == 1` 有快路直接返回 `ptr`（`:35-36`），单卡编译期就不产生任何地址运算。
- `kNumMaxRanks = 72`（`:7`）：DeepSeek 用的最大 EP 规模（8×9 或 72 卡），所以偏移表固定 72 项，
  `SymBuffer` 可当 `__grid_constant__` 直接传进 kernel（`sm100_fp8_fp4_mega_moe.cuh:60`）。

**为什么用「offset 数组」而不是「指针数组」**：`__grid_constant__` 传进 kernel 的东西必须在
每个 rank 上是**同一份常量**，但各 rank 的绝对地址不同。存相对偏移，则 8 个 rank 可以把同一
张 offset 表编进 kernel 参数（只要各 rank 的基址差是常量）；device 端再加回本地 base 即可。

---

## 4. `wait_until`：带超时的自旋

```
:12   constexpr int64_t kNumTimeoutCycles = 60ll * 2000000000ll;   // 60s @ 2GHz
:16   wait_until(pred, print_timeout):
:17       start = clock64();
:18       while (not pred())
:19           if (clock64() - start >= kNumTimeoutCycles) {
:20               print_timeout();
:21               DG_DEVICE_ASSERT(false and "Timeout");
:22           }
```

为什么要有它：**所有无锁自旋都必须有退出路径**，否则一个 rank 出错会让整机 kernel 永久挂住
（NCCL 能做超时 abort，但 DeepGEMM 是裸 `red`/`ld` 轮询）。它把 60 秒的代价换成一个
**能打印出「卡在哪一次 barrier、rank 号、计数器当前值、期望值」的确定性失败**——
`print_timeout` 回调正因此遍布 kernel（`barrier.cuh:46-47/83-84`、
`sm100_fp8_fp4_mega_moe.cuh:1441`）。

注意它的超时是**每调用点独立**的（`start_clock` 是 local），不是全局截止时间。

---

## 5. `grid_sync`：单 rank 全 SM barrier，以及「加 2³¹ 免复位」

Mega MoE 是**常驻单 kernel**：一个 grid 的所有 SM 必须协同（dispatch 段 / 各 epilogue 段）。
`grid_sync` 就是它们之间的 barrier。核心实现（`comm/barrier.cuh:33-51`）：

```
:38   static constexpr uint32_t kFinishSumTag = 0x80000000u;      // 2^31
:39   sync_scope();                                                // ① 本地同步（__syncthreads / 命名 barrier）
:40   if (thread_idx == 0) {
:41       count_ptr = workspace.get_grid_sync_count_ptr<kIndex>();
:42       old = ptx::atomic_add_rel(count_ptr,
:43                     sm_idx == 0 ? (kFinishSumTag - (kNumSMs - 1)) : 1);
:45       wait_until([&]{ return ((new = ld_acq(count_ptr)) ^ old) & kFinishSumTag; }, ...);
:50   sync_scope();                                                // ②
```

### 为什么是「和恰好 2³¹」

设 barrier 开始时计数器为 `S`。本轮所有 SM 的增量之和 = `(kNumSMs-1)×1 + (2³¹-(kNumSMs-1))`
= **恰好 2³¹**。于是：

- 结束值 = `S + 2³¹ (mod 2³²)`，只把**最高位翻转**，低 31 位原样不变；
- 下一轮再加 2³¹ → 又翻回来。**两轮回到原值**，所以计数器**永远不用清零、可无限复用**
  （这正是注释不在但设计在的理由；`grid_sync` 没有 reset 代码）。
- 每个 SM 的 `old` 是它自己那次 `atomic_add` 的**返回值**（此前所有增量的和）。由于本轮增量
  全是正数且总和正好 2³¹，任何 `old` 与本轮结束值只可能在**最高位**上不同：
  - 起始 `S` 高位为 0 时，所有中间值都 `< 2³¹`（高位 0），结束值 `< 2³²` 高位 1 → `old ^ new` 高位 = 1；
  - 起始 `S` 高位为 1 时，中间值都 `≥ 2³¹`（高位 1），结束值回绕到高位 0 → `old ^ new` 高位 = 1。

所以 `(new ^ old) & 0x80000000` 是「全体到齐」的**充要**判据。`sm_idx == 0` 那个「大增量」
保证「恰好一个 SM 一次性补足到 2³¹」，避免需要按 31 位掩码判断。

### 它保证什么、不保证什么

`:42` 是 `atom.release.gpu`、`:45` 是 `ld.acquire.gpu`——**gpu 作用域**的 release/acquire。
所以 barrier 前 thread 0 的普通写，对 barrier 后所有 SM 的 thread 0 可见（后文实测 mismatch=0）。
但它**只是 barrier**：barrier 之后各 SM 立刻独立前进，**不会**替你把「读」保护到下一次写。
本实验第一版就踩了这个坑——每轮复用同一格 `flags[sm]`，快的 SM 在慢的 SM 还没读完本轮时
就写进了下一轮的值，产生一堆 mismatch；改成**每轮独立槽**（`flags[r*G+sm]`）后归零。
这正说明 `grid_sync` 是「同步点」而非「读保护」，真实 kernel 里两次 grid_sync 之间
**不再有别的 SM 写**（`sm100_fp8_fp4_mega_moe.cuh:426` 的注释
"After the grid sync above, there is no more writes by other SMs (except 0)"）正是这个约定。

`kNumSMs` 是模板参数——它必须等于**实际常驻的 block 数**（grid 大小），因为判据依赖总和恰好
是一份 2³¹。这就是本机共享 GPU 时「不能随便跑大 grid」的原因（第 38 篇的顾虑）；
本篇用小 grid（≤64）就能安全验证。

---

## 6. `nvlink_barrier`：跨 rank barrier 的状态机

跨 rank 的信号没法像单 rank 那样用一个原子和解决——因为「读别人的内存」要走 NVLink，而
「别人的信号写过来」也需要一个能区分**这一次**和**上一次**的机制。实现（`comm/barrier.cuh:53-92`）：

```
:60   DG_STATIC_ASSERT(kNumRanks <= kNumThreads, ...);
:63   if (sync_prologue) grid_sync(...);                        // ① 本 rank 全 SM 对齐
:67   if (sm_idx == 0) {                                        // ② 只 SM 0 参与跨 rank
:68       counter_ptr = workspace.get_nvl_barrier_counter_ptr();
:69       status     = (*counter_ptr) & 3;
:70       signal_phase = status & 1;   signal_sign = status >> 1;
:71       signal_ptr   = workspace.get_nvl_barrier_signal_ptr(signal_phase);
:74       if (thread_idx < kNumRanks)
:75           ptx::red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx),
:75                                signal_sign ? -1 : 1);      // 给每个 peer 的信号槽 ±1
:79       if (thread_idx == 0) {
:80           ptx::red_add(counter_ptr, 1);
:81           target = signal_sign ? 0 : (int)kNumRanks;
:82           wait_until([&]{ return ptx::ld_acq_sys(signal_ptr) == target; }, ...);
:90   if (sync_epilogue) grid_sync(...);                       // ③ 本 rank 全 SM 对齐
```

三件套：**本 rank grid_sync → 跨 rank 信号 → 本 rank grid_sync**。

### 谁给谁发信号

线程 `i`（`i < kNumRanks`）执行 `red_add_rel_sys(map(signal_ptr, i), ±1)`：把**本 rank**的
`nvl_barrier_signals[phase]` 地址映射到 **rank i** 的同一个位置，加 ±1。也就是说：

- 本 rank 向**每一个 peer** 的 `signal[phase]` 槽贡献 ±1；
- 本 rank 自己的 `signal[phase]` 槽，会收到**每一个 peer** 贡献的 ±1（共 `kNumRanks` 个）。

于是 thread 0 只需等自己那个槽累加到 `±kNumRanks`。用**有符号**是为了能减（`:64`），
让信号加满后能减回 0、**免复位**。

### phase / sign 状态机（为什么是 `&3`）

`counter` 每调用一次 barrier 加 1（`:80`）。`status = counter & 3` 的 4 个状态编码了
`(phase, sign)`：

| `counter&3` | `phase = &1` | `sign = >>1` | 动作 |
|---|---|---|---|
| 0 | 0 | 0 | `signal[0]` +1，target = `+kNumRanks` |
| 1 | 1 | 0 | `signal[1]` +1，target = `+kNumRanks` |
| 2 | 0 | 1 | `signal[0]` −1，target = `0` |
| 3 | 1 | 1 | `signal[1]` −1，target = `0` |

于是：

- **phase 交替**（0,1,0,1,…）：相邻两次 barrier 用**不同** slot，避免「上一次的信号还没被读完、
  这一次就覆盖」的竞争；
- **sign 每两次翻一次**（+,+,−,−）：每个 slot 被 +1 加满后，两次 barrier 之后再用 −1 清回 0——
  **同样免复位、可无限复用**；
- 4 次一个周期，`counter` 回到原值。

`sign` 的语义就是「**上一次是加、这一次是减**」；`target` 随之在 `kNumRanks` 与 `0` 间切换。
`kTag` 只用于超时打印区分（`barrier.cuh:83-84` 打印 phase/sign/tag）。

---

## 7. `cluster_sync_with_relaxed_arrive`

```
:26   cluster_sync_with_relaxed_arrive():
:29       cute::cluster_arrive_relaxed();
:30       cute::cluster_wait();
```

标准 `cluster_sync` 是 `arrive` + `wait`，其中 `arrive` 隐含**释放语义**（保证 cluster 内
其它 CTA 看得到本 CTA 之前的写）。`arrive.relaxed` 去掉这个保证换来更快。因此它只用在
**不需要跨 CTA 数据可见性**的场景——Mega MoE 里是 **2-CTA tensor memory 分配前**的裸同步
（`sm100_fp8_fp4_mega_moe.cuh:237` 注释 "A cluster sync is essential for 2CTA tensor memory
allocation"，`:238` 调用）；同样的用法在 1D2D / BF16 等 kernel 里是
`kNumTMAMulticast > 1 ? cluster_sync_with_relaxed_arrive() : __syncthreads()`
（`sm90_fp8_gemm_1d2d.cuh:142` 等）——多播开启时 cluster 内 CTA 只需「都到齐」即可。

---

## 8. H100 实测

脚本 `harness/exp_comm_barrier.sh` + 源码 `src/comm_barrier.cu`（输出
`harness/exp_comm_barrier.out.txt`）。**直接 include 真实头文件**
（sha256：`barrier.cuh ef290692…`、`sym_buffer.cuh b157512f…`、`mega_moe.cuh 53008d96…`），
`nvcc -arch=sm_90` 编译，H100(sm90)、sm_count=132、8 卡。

**Part A —— `grid_sync`（单卡）**：G ∈ {1,2,8,16,64,7}，各跑 **2000** 轮。
每轮每 block 的 thread 0 往**独立槽**写 `flags[r*G+sm]=r+1`，grid_sync 后读回所有槽核对：

```
GRIDSYNC G=1  rounds=2000 mismatch=0 counter_seq=2147483648,0,2147483648,0,2147483648,0
GRIDSYNC G=2  rounds=2000 mismatch=0 counter_seq=2147483648,0,2147483648,0,2147483648,0
GRIDSYNC G=8  rounds=2000 mismatch=0 ...     GRIDSYNC G=16 rounds=2000 mismatch=0 ...
GRIDSYNC G=64 rounds=2000 mismatch=0 ...     GRIDSYNC G=7  rounds=2000 mismatch=0 ...
```

三点坐实：① 有序性——barrier 前所有 block 的写，barrier 后被所有 block 读到（mismatch=0，
含奇数 G=7 走 `2³¹-(N-1)` 分支）；② **免复位**——2000 轮后计数器仍在 `0` / `2147483648`
（=2³¹）之间翻转，从不累积、从不需要清零，与 `(r·2³¹) mod 2³²` 的预期**逐点相等**；
③ 可无限复用（2000 轮 ≈ 每轮两次 2³¹）。

**Part B —— `nvlink_barrier` 状态机（单卡虚拟多 rank）**：把 `SymBuffer` 的
offsets 全设 0，令 N 个「虚拟 rank」的信号落到同一地址；N∈{2,8}，各跑 1000 轮，
记录每轮的 `(counter&3, signal[0], signal[1])`：

```
NVLVIrt N=2 G=8 rounds=1000 trace_bad=0
  (status,sig0,sig1): (1,2,0) (2,2,2) (3,0,2) (0,0,0) (1,2,0) (2,2,2) (3,0,2) (0,0,0)
NVLVIrt N=8 G=8 rounds=1000 trace_bad=0
  (status,sig0,sig1): (1,8,0) (2,8,8) (3,0,8) (0,0,0) (1,8,0) (2,8,8) (3,0,8) (0,0,0)
```

`trace_bad=0`：4 轮一个完整周期，`signal[0]` 在 `+N→0`、`signal[1]` 在 `+N→0` 交替，
**每 4 次 barrier 两个信号槽都精确归零**——坐实 §6 的 phase/sign 状态机（`N=2` 显示 `+2/+8`，
`N=8` 显示 `+8/+8`，target 恰为 `kNumRanks`）。这是「单卡就能跑出跨 rank 信号算术」的巧法。

**Part C —— `nvlink_barrier` 真实 8 卡（UVA peer 对称缓冲）**：每卡 `cudaMalloc` 一块
等长缓冲（`MegaMoESignals` + 每轮每 rank 一个 8 字节数据槽），两两开 `cudaDeviceEnablePeerAccess`，
8 卡各起一个 grid（G=16）并发跑 **30** 轮 barrier；每轮每 rank 用 `st_rel_sys` 写自己那格、
barrier 后用 `ld_acq_sys` 读**全部 8 个 peer**的值：

```
NVLRANK ranks=8 G=16 rounds=30 mismatch=0
```

即：8 卡并发 barrier 全部按时返回（无 60s 超时），且 barrier 后**能逐字节读到每个对端本轮写的值**
（`sys` 作用域释放/获取跨 NVLink 生效）。这把第 11 篇「SM 0 每 rank 一个线程参与 NVLink 信号」
的机制在真机上跑通了。

**Part D —— `cluster_sync_with_relaxed_arrive`（2-CTA cluster）**：

```
CLUSTER dim=2 seen_peer=1/1
```

两个 CTA 在 `arrive.relaxed + wait` 后都能看到对方的写（本机实测 1/1）。**但这只是记录，不是保证**：
relaxed arrive 不承诺跨 CTA 的写可见性，真机很可能因为 L2 强一致而「碰巧」可见——
正如第 38 篇对单卡 release/acquire 的弱序对照所注。它的正确用途是本篇 §7 说的
「只需到齐、不需可见性」的 tensor memory 分配场景。

**总判定 ALL PASS。**

---

## 9. 结论与边界

- `comm/barrier.cuh` 是 Mega MoE「单 mega-kernel + 通信-计算重叠」的**同步地基**，
  四个原语各司其职：本地全 SM（`grid_sync`）、跨 rank（`nvlink_barrier`）、超时自旋
  （`wait_until`）、cluster 裸同步（`cluster_sync_with_relaxed_arrive`）。
- 两个「计数/相位」设计都追求**免复位、可无限复用**：`grid_sync` 靠「和恰好 2³¹ 只翻最高位」，
  `nvlink_barrier` 靠「phase 交替 + sign 每两次翻转、±1 自清零」。
- `SymBuffer` 用**相对偏移**而非绝对指针，使同一张表能编进所有 rank 的 `__grid_constant__` 参数。
- **真机验证**（H100/sm90，架构无关代码直接编译运行）：`grid_sync` 6 组 ×2000 轮 mismatch=0 且
  计数器按预期翻转；`nvlink_barrier` 虚拟 N∈{2,8} 状态机 trace_bad=0；真实 8 卡 ×30 轮
  mismatch=0；cluster relaxed 1/1。
- **边界**：`kNumSMs` 必须等于实际常驻 block 数（本实验用小 grid 规避共享 GPU 的风险）；
  `redux.sync.max.f32` 等 SM100 专属指令不在本篇范围；真正的超时/trap 路径未触发
  （需刻意构造死锁，代价是 60 秒 + trap），故只做源码解读。

## 10. 参考

- DeepGEMM 仓库：<https://github.com/deepseek-ai/DeepGEMM>（本机核对 commit `057ca59`，Public release 26/09/30）。
- 参考文件：`deep_gemm/include/deep_gemm/comm/barrier.cuh`、`deep_gemm/include/deep_gemm/layout/sym_buffer.cuh`、
  `deep_gemm/include/deep_gemm/layout/mega_moe.cuh`、`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
  / `sm100_bf16_mega_moe.cuh`。
- 相关篇目：第 11 篇（dispatch/combine 原语表）、第 38 篇（PTX 内存序底座）、第 09 篇（mbarrier 与流水）。
