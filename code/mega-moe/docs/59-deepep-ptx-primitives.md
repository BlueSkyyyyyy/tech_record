# DeepEP 的设备端公共层：mbarrier、TMA、multimem、谓词化访存与内存序

> 本文对应路线图 **P49-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 59 篇。
> 目标：把 DeepEP **所有 kernel 共同踩的那层地板**讲透——`deep_ep/include/deep_ep/common/ptx.cuh`
> （612 行）以及同一目录的 `math.cuh` / `utils.cuh` 与 `layout/common/barrier.cuh`。
> 前面第 7 篇精读了 DeepEP 的 dispatch/combine 数据流，第 53 篇拆了 EPLB 冗余，第 58 篇拆了 combine
> 归约 epilogue，但**它们每一行 `mbarrier`/`cp.async`/`TMA`/`red.release.sys`/`__match_any_sync` 都来自
> 这一层，却从没系统讲过**。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
> - 设备端 PTX 原语：`deep_ep/include/deep_ep/common/ptx.cuh`（612 行，sha256 `e175b3a0…`）
> - 整数/指针/打包：`deep_ep/include/deep_ep/common/math.cuh`（68 行，sha256 `b49dbcb9…`）
> - 通用网格步进搬运：`deep_ep/include/deep_ep/common/utils.cuh`（77 行，sha256 `8cf9acfc…`）
> - 屏障信号布局：`deep_ep/include/deep_ep/layout/common/barrier.cuh`（25 行，sha256 `f761c574…`）
> - 编译期常量/别名：`deep_ep/include/deep_ep/common/compiled.cuh`
>
> **可跑性**：这一层**绝大多数是架构无关的内联 asm**，在 sm90a 上都是合法 ISA；带
> `__CUDA_ARCH__ >= 1000` 守卫的原语（`st.bulk`、256 位 `ld/st`、打包 `fadd2/fmul2`、融合
> `add.rn.f32.bf16`）在 sm90 **会走文件里给出的回退分支**，本文直接验证回退分支；`multimem.*`
> 需要 NVLink 多播映射、`cvt.rs.bf16x2` 是 SM100 指令，二者只做 host 语义 oracle。
> 因此本专题**直接 include 真实头文件、编译运行真实设备代码**，在 H100 上
> `harness/exp_deepep_ptx_primitives.{sh,out.txt}`（**41 PASS / 0 FAIL**）+ `src/deepep_ptx_primitives.cu`。

## 1. 一段话总览

`common/ptx.cuh` 是 DeepEP 的「设备端标准库」：它把 dispatch/combine/bucket/engram 里反复出现的
低层动作收敛成**具名内联 asm 封装**，大致分八组：

| 组 | 代表原语 | 行号 | 谁在用 |
|----|----------|------|--------|
| 线程/选举 | `get_warp_idx` / `get_lane_idx` / `elect_one_sync` | `:23-46` | 所有 TMA 发射点 |
| mbarrier | `mbarrier_init_with_fence` / `_arrive` / `_arrive_and_set_tx` / `_wait_and_flip_phase` / `_invalidate` | `:49-83` | dispatch/combine/reduce_grads |
| TMA & `cp.async` | `tma_load_1d` / `tma_store_1d` / `tma_store_reduce_add_f32` / `cp_async_ca` / `cp_async_mbarrier_arrive` / `st_bulk` | `:85-209` | 全部通信 kernel |
| NVLink 多播 | `multimem_ld_reduce_add_f32` / `multimem_st_f32` / `multimem_cp_async_bulk` | `:162-195` | bucket all_reduce / reduce_scatter |
| 谓词化访存 | `ldg_with_{ge,gt}_pred` / `ld_with_*_pred` / `st_with_*_pred` | `:223-362` | combine 去重、dispatch 裁剪 |
| 内存序 | `ld_volatile` / `red_add*` / `ld_acquire_sys` / `st_relaxed_*` / `st_release_sys` / `fence_*` | `:364-478` | 跨 rank readiness |
| 寄存器再平衡 | `warpgroup_reg_realloc`（`setmaxnreg`） | `:454-469` | hybrid_combine / bucket |
| warp 内建 & 浮点 | `exchange` / `gather` / `match` / `deduplicate` / `reduce_or/add` / `warp_*_sum` / `fadd4` / `fmul4` / `accumulate` / `cvt_rs_bf16x2` | `:481-608` | dispatch/combine/bucket 处处 |

一句话：**DeepGEMM 的 `docs/38` 讲过它自己的 PTX 底座；DeepEP 这层看着像，但为「跨 rank 通信」专门
加了 multimem、`sys` 作用域内存序、谓词化 `ld.global.nc` 与 grid-stride 搬运**——这正是本文的重点。

## 2. mbarrier：五原语，与「谁在等谁」

`ptx.cuh:12-13` 把 `cuda::barrier<thread_scope_block>` 重命名为 `mbarrier`，相位是 `uint32_t`。
五个原语（`:49-83`）：

```
mbarrier_init_with_fence(ptr, count)   // mbarrier.init + fence.mbarrier_init.release.cluster
mbarrier_invalidate(ptr)               // mbarrier.inval（配对，销毁前）
mbarrier_arrive(ptr)                   // mbarrier.arrive
mbarrier_arrive_and_set_tx(ptr, bytes) // mbarrier.arrive.expect_tx —— 声明「还要等 bytes 字节」
mbarrier_wait_and_flip_phase(ptr, ph)  // mbarrier.try_wait.parity 自旋 + ph ^= 1
```

三个「为什么」：

1. **`init` 必须带 `fence.mbarrier_init.release.cluster`**（`:52`）。mbarrier 初始化写的是共享内存，
   而 TMA / `cp.async.bulk` 走的是**异步代理（async proxy）**；没有这条 fence，异步引擎可能看到
   未完成的 init。它把 init 对 cluster 内所有线程/代理 release。
2. **`wait` 是 `try_wait.parity` 自旋**（`:70-83`），不是 `mbarrier.test_wait` 一次了事——等待期间允许
   其他 warp 继续跑；`0x989680`（一千万）是 suspend hint。**相位每 wait 成功一次就 `^=1`**，
   这样同一个 mbarrier 可以复用任意多轮（`.parity` 用 0/1 区分奇偶轮）。
3. **`.expect_tx` 是「事务计数」**（`:65-68`）：TMA 的 `cp.async.bulk ... mbarrier::complete_tx::bytes`
   在拷贝完成时把**字节数**加到 mbarrier 的 tx-count；只有「arrival 到齐 **且** tx-count 归零」相位才翻。
   dispatch 里 `mbarrier_arrive_and_set_tx(mbarrier_ptr, kNumHiddenBytes)`（`impls/ep/dispatch.cuh:353`）
   正是「等这一 token 的 `kNumHiddenBytes` 字节 TMA 落地」。

这里有一个**特别容易看错的地方**：dispatch 的 mbarrier 初始化计数是 **1**（`dispatch.cuh:273`），
可 `cp.async.mbarrier.arrive` 却被 32 个 lane 都调用（`:309`）。原因是 `cp.async.mbarrier.arrive`
（**没有 `.noinc`**）在**发射时把 pending count +1、拷贝完成时 arrive −1**，净零，但让 mbarrier
「卡住」直到所有 `cp.async` 写完。于是：

- `:309` 32 个 lane 各发一次 cp.async 的 arrive ⇒ pending 从 1 涨到 33，拷贝完成后再降回 1；
- `:353` 由 elected one 补一次 `arrive.expect_tx(hidden_bytes)` ⇒ pending 归 0，同时挂上 tx 预算；
- `:354` 等相位翻转 ⇒ 既保证 `cp.async`（SF）写完，又保证 TMA（激活）到齐。

本专题用一个**同构**的小用例验证了这条链：`mbarrier_init(count=1)` → 64 个 lane 各发
`cp_async_ca` + `cp_async_mbarrier_arrive` → 全块同步后 elected one 补一次 `mbarrier_arrive` → 等相位。
结果 `cp.async.ca` 载入的数据完整正确（Part 6，PASS），坐实「issue 时 +1、完成时 −1」的计数模型。

## 3. TMA 与 `cp.async`：三条搬运路径

`ptx.cuh` 暴露了三条不同的搬运路径，选哪条取决于「数据要从哪到哪、是否要在落地时做归约」：

1. **`tma_load_1d`（`:122-134`）** —— `cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes`。
   注意目标地址空间是 `shared::cluster`（不是 `shared::cta`），所以天然支持 cluster 内寻址；默认
   `L2CacheHint::kEvictFirst`（`:118`，`0x12f…`），因为**载入激活通常读一次就不再用**。
2. **`tma_store_1d`（`:136-146`）** —— `cp.async.bulk.global.shared::cta.bulk_group`，走 `bulk_group`
   而非 mbarrier，用 `tma_store_commit()`（`:158-160`）+ `tma_store_wait<N>()`（`:107-110`）配平；
   默认 `kEvictNormal`（`:119`，`0x100…`），因为**写出去的数据马上会被对端读**。
   `tma_store_wait_read`（`:112-115`）是 `.read` 变体，只保证「共享内存可安全改写」，不等到对端可见。
3. **`tma_store_reduce_add_f32`（`:148-156`）** —— `cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32`，
   **元素级原子加**（`dst += src`，不是覆盖）。这是 EP 梯度归约的关键：`reduce_grads.cuh:82`、
   `bucket/all_reduce/rdma.cuh:183`、`bucket/all_reduce/hybrid.cuh:253` 都用它把分片累加回目标。
   本专题 Part 4 用「两次归约写 = init + a + b」直接坐实了它是累加（`dst[3]=110.0`），
   对照普通 `tma_store_1d` 则覆盖（`dst[3]=3.0`）。

`cp.async` 那一对（`:197-209`）：`cp_async_ca` 是 `cp.async.ca.shared::cta.global.L2::128B`（走 L1，
用于 SF 这种小、会复用的数据），`cp_async_mbarrier_arrive` 见 §2。`st_bulk`（`:85-101`）在
`arch>=1000` 用 `st.bulk.weak.shared::cta`（一条指令清零整块 smem，见 `dispatch_copy_epilogue.cuh:239`
清 hidden 缓冲），在 sm90 回退为「逐 lane 写 0」循环——本文 Part 5 验证回退把 smem 清零正确。

## 4. NVLink 多播：`multimem.*`（`all_reduce`/`reduce_scatter` 的核心）

`:162-195` 三个 multimem 原语是 DeepGEMM 底座里**没有**的——它们服务于 NVLink 域内的**多播寻址**：

```
multimem.ld_reduce.relaxed.gpu.global.add.v4.f32  {d}, [addr];   // 读「多播地址」，返回所有 rank 之和
multimem.st.relaxed.gpu.global.v4.f32             [addr], {v};   // 把同一个 v 写到所有 rank
multimem.cp.async.bulk.global.shared::cta.bulk_group [dst], [src];  // smem→多播地址（一份写全体）
```

`bucket/all_reduce/nvlink.cuh:44-50` 的妙处：每个 rank 只写自己那 1/N 的 shard（`multimem_st` 广播），
再用 `multimem_ld_reduce_add` 一次读回**全 rank 的和**——**每个 rank 的 NVLink 流量从 O(N) 降到 O(1)**，
且归约由交换硬件完成、顺序无关。`with_gt_pred` 包装（`:167/181`）用 `setp.gt.s32` 给这条指令加谓词，
让同一段循环既能处理满块也能跳过尾巴。

**本机限制**：`multimem` 需要多播对象（`cuMulticastCreate` + 绑定内存），本机可建但代价高，本文只做
host 语义 oracle（Part 13：8 rank 同地址求和 = 顺序无关正确）。**标注：需 NVLink 多播映射，仅分析。**

## 5. 谓词化访存 + `iterate_ld_st`：把「越界裁剪」编进指令

EP 通信里到处是「按变量长度载入/存储」，DeepEP 的做法不是 `if (idx < n) load;`，而是**把比较编进
访存指令的谓词**（`:223-362`）：

```
ldg_with_ge_pred (int4)   ld.L1::no_allocate.L2::cache_hint.global.nc.v4.s32  @p
ldg_with_gt_pred (int4)   同上，setp.gt
ld_with_ge_pred  (int4)   同上去掉 .nc（走一致性路径）
ld_with_gt_pred  (float4) 浮点版
st_with_ge_pred  (dtype)  4 字节谓词化存
st_with_gt_pred  (float4) float4 谓词化存
```

三个要点：
- **`.nc`**（non-coherent / 纹理路径）只在只读数据上用（`ldg_*`），绕过 L1 一致性开销；
- **`L1::no_allocate` + `L2::cache_hint`（evict_first）**：这些数据是流式的，不进 L1、L2 里也尽快淘汰，
  避免把有用的激活挤出缓存；
- **谓词为假时返回 0 / 不写**。`combine_reduce` 就是靠它跳过 `-1` 占位槽（第 58 篇）。

`:280-318` 的 256 位 `longlong4_t` 版只在 `arch>=1000` 提供（sm90 无 256 位全局 ld）；`ldg(int4)`
（`:320-322`）就是 `__ldg`。

`common/utils.cuh:14-75` 的 `iterate_ld_st` 是这组谓词原语的**宿主**：一个模板化的网格步进循环，
`kNumElemsPerThread` 个元素一批，主循环对整批用 `ld_func(ptr)`（默认谓词恒真），**尾部**改用
`ld_func(ptr, remaining, offset)` 让谓词按剩余长度裁剪。`bucket/all_reduce/nvlink.cuh:44` 用它把
「每 rank 的 shard」扫完——`<kNumSMs, kNumThreads, 8>`（每次 8 个 `float4`）。本文 Part 8 用
`iterate_ld_st<1,128,4>` 处理 1000 个元素，验证主循环 + 尾部谓词把**全部**元素写对。

## 6. 内存序：跨 rank 就绪为什么必须 `sys` 作用域

`:364-478` 是一整套内存序原语，按「作用域 × 序」排成阶梯：

| 原语 | PTX | 语义 |
|------|-----|------|
| `ld_volatile` | `ld.volatile.global.{u32,u64}` | 轮询标志（编译器不得优化） |
| `red_add` | `red.gpu.global.add.u64` | GPU 作用域原子加，**不返回旧值** |
| `red_add_rel_gpu` / `_sys` | `red.release.{gpu,sys}.global.add` | release 语义原子加 |
| `ld_acquire_sys` | `ld.acquire.sys.L1::no_allocate.global` | 跨卡 acquire（不分配 L1） |
| `st_relaxed_gpu` / `_sys` | `st.relaxed.{gpu,sys}.global` | 弱序写 |
| `st_release_sys` | `st.release.sys.global` | 跨卡 release 发布 |
| `fence_acquire_gpu` / `fence_acq_rel_sys` | `fence.acquire.gpu` / `fence.acq_rel.sys` | 显式 fence |

**为什么是 `sys`**：对称缓冲里的「计数/就绪标志」在**另一个 GPU** 的显存里（经 NVLink/RDMA 可达）。
`gpu` 作用域只保证本 GPU 内的可见性，**跨卡必须 `.sys`**。dispatch 的 `red_add_rel_sys`（`:384-390`）
把「到达计数」原子加并 release，combine 用 `ld_acquire_sys`（`:400-413`）旋等对端水位——这对
release-acquire 正是 `docs/11` 讲的 readiness 机制。

本文 Part 9 用两个用例验证：① 三路 `red_add`（gpu / rel_gpu / rel_sys）在 256 线程下总和 = 768；
② **每轮独立 cell** 的生产者-消费者（生产者 `__threadfence` + `st_release_sys` 发布、消费者
`ld_acquire_sys` 旋等）2000 轮**坏数据计数 = 0**——release-acquire 下「看到 flag 就一定能看到 data」。

## 7. 寄存器再平衡、warp 内建与浮点打包

- **`warpgroup_reg_realloc<kNumRegs, kNumThreads>`（`:454-469`）** 走 `setmaxnreg.{inc,dec}.sync.aligned`
  （**sm90a 起**）。初始预算 `kNumInitialRegisters = (65536 / kNumThreads / 8) * 8`；≥255 时直接早退
  （128/256 线程的 block 就是这样），否则按目标 inc/dec。它让「生产者 warp 少寄存器、消费者 warp 多
  寄存器」在一段 kernel 内动态重配——`hybrid_combine.cuh:112/358`、`bucket/*/hybrid.cuh` 用它把寄存器
  倾斜给多播/归约 warp。本文 Part 12 验证公式（512 线程 → 128）并实跑 `setmaxnreg.dec` 到 40。
- **warp 内建（`:481-560`）**：`exchange<T>`（`__shfl_sync`，大类型按 int 拆）、`gather`=`__ballot_sync`、
  `reduce_or`/`reduce_add`/`all`/`any`、`match`=`__match_any_sync`、`get_master_lane_idx`=`bfind.u32`
  （组内**最高** lane）、`deduplicate(v,lane)`=是否本 lane 是 `match(v)` 组的 master、`ffs<T>`=`__ffs(-1)`
  （0-based，8 字节走 `__ffsll`）、`warp_inclusive_sum`/`warp_exclusive_sum`（`shfl_up` 的 Hillis-Steele）。
  `deduplicate` 是 combine 去重的基石（`combine_reduce_epilogue.cuh:87`）；`dispatch.cuh:340` 也用它
  给「同一 rank 的多个 top-k」只分配一次槽。本文 Part 10 逐项对拍（ballot `0x49249249`、master=3、
  dedup master 数=8、inc/exc=496/465…）。
- **浮点打包（`:562-608`）**：`fadd4`/`fmul4`（含 `fmul4(a, scalar)`）在 SM100 用 `__fadd2_rn`/`__fmul2_rn`
  的 packed float2 快路，sm90 **回退为逐元素标量**；`accumulate(float2&, nv_bfloat162)` 在 SM100 用
  融合 `add.rn.f32.bf16`（bf16→fp32 加，一步），sm90 回退 `__bfloat1622float2` 再加；`cvt_rs_bf16x2`
  是 **`cvt.rs.bf16x2.f32`（随机舍入到 bf16）——SM100 专属**（`bucket/reduce_scatter/rdma.cuh:415` 用它
  在 RDMA reduce-scatter 前做无偏的 fp32→bf16）。本文 Part 11 验证 sm90 回退与标量**逐位一致**；
  实测 `cvt.rs` 在 `sm_90a` 直接 ptxas 报 `Feature '.rs' not supported`，`sm_100a` 通过。
  Part 13 的 host oracle 展示了它的语义：随机位阈值 `0x0000→0.9997、0x4000→0.7493、0x8000→0.4992、
  0xc000→0.2501、0xffff→0`，即**阈值越大越少进位、且无偏**。

## 8. `math.cuh` 与 `barrier.cuh`：两个「看着简单但关键」的约定

`math.cuh` 有一个**贯穿 dispatch 的编码技巧**值得单列：`encode_decode_positive`（`:30-33`）
`= -value - 1`，配 `is_decoded_positive_ready`（`:25-28`）`= value >= 0`。它是一对对合：`f(f(x))=x`。
dispatch 把「计数」编码成**负数**写进对称缓冲（`dispatch.cuh:128/226`），消费者读回后
`encode_decode_positive` 还原再判 `>=0`（`:189-190`）。这样做的原因：**对称缓冲初值是 0**，
「未就绪」需要一个与「任何合法就绪值」都不同的哨兵——用 `0` 当「未写」，用负数编码「已就绪的计数」，
判据就退化成一次符号判断。本文 Part 0 验证对合性、`raw=0 ⇒ 未就绪`、`raw<0 ⇒ 就绪`。

其余：`ceil_div`/`align`（含 `kDoCeilAlignment=false` 向下取整，`:15-18`）、`advance_ptr`/`ptr_diff`、
`pack2`/`unpack2`（`:44-66`，**低位在前**，Part 0b 在设备端验证 `0x1111,0x2222 → 0x22221111`）。

`layout/common/barrier.cuh:11-22` 的 `BarrierSignals` 是**NVL 屏障的共享布局**：一个
`nvl_barrier_counter`（64 位计数器）+ 两个 `nvl_barrier_signals[2]`（滑动相位），并对齐到
`kNumRDMAAlignmentBytes = 64`。两个 accessor 只是取址封装，但「计数 + 双相位」正是 DeepEP 屏障
「加减式、免复位」的实现方式（对照 `docs/11` 的 `grid_sync`/`nvlink_barrier`）。本文 Part 0 验证
`sizeof == 64` 与取址。

## 9. 与 DeepGEMM PTX 层（`docs/38`）的对照

| 维度 | DeepGEMM `ptx/*`（docs/38） | DeepEP `common/ptx.cuh`（本文） |
|------|------------------------------|----------------------------------|
| 定位 | 单机 GEMM kernel 的访存/MMA 底座 | **跨 rank 通信** kernel 的底座 |
| 多播 | 无 | `multimem.{ld_reduce,st,cp.async.bulk}` |
| 内存序 | 主要 `gpu` 作用域 + tensormap release/acquire | 明确分 `gpu`/`sys` 两档，跨卡一律 `sys` |
| 谓词访存 | `ld_evict_first`、`cp_async_cg_zfill` | 一整套 `*_with_{ge,gt}_pred` + `ld.global.nc` |
| mbarrier | 五原语 + 相位翻转 | 五原语 + 相位翻转（语义一致）+ `cp.async.mbarrier.arrive` 计数模型 |
| warp 内建 | `redux.sync.max.f32`（SM100）等 | `match`/`deduplicate`/`fns`/`reduce_or` 更偏「通信去重」 |
| 浮点打包 | `accumulate`/`cvt_relu_bf16x2` | `fadd4`/`fmul4`/`accumulate`/`cvt_rs_bf16x2` |

两者都遵循同一套「SM100 专属原语在 sm90 编译期回退」的写法，但 DeepEP 因为要跨卡，多了一整套
`sys` 作用域内存序与 multimem。

## 10. H100 实测（`harness/exp_deepep_ptx_primitives.{sh,out.txt}`，41 PASS / 0 FAIL）

直接 `nvcc -gencode arch=compute_90a,code=sm_90a` include 真实头文件，编译运行 **真实设备代码**：

| Part | 内容 | 硬结果 |
|------|------|--------|
| 0 | `math.cuh` 语义 + `BarrierSignals` | 对合成立；`sizeof==64`；`pack2` 低位在前 |
| 1 | `elect_one_sync` / warp/lane idx | 每 warp 恰选 1；warp_idx=[0 1 2 3]；lane0=0 |
| 2 | mbarrier 相位翻转 | 两次 wait 后 phase = 1, 0 |
| 3 | TMA 1D 载入 + `expect_tx` | 256 float 完整正确 |
| 4 | TMA store / `cp.reduce.add.f32` | 归约写 = init+a+b（110.0）；普通 store 覆盖（3.0） |
| 5 | `st.bulk`（sm90 回退） | 清零后非零元素 = 0 |
| 6 | `cp.async.ca` + `cp.async.mbarrier.arrive` | 载入正确（坐实 issue+1/完成−1） |
| 7 | 命名 barrier（`bar.sync`/`barrier.sync`） | 两处 counter 都 = 128 |
| 8 | 谓词化 ld/st + `iterate_ld_st` | 越界 lane 不写；谓词假返回 0；1000 元素全对 |
| 9 | 内存序 | `red_add×3` = 768；2000 轮生产者-消费者坏数据 = 0 |
| 10 | warp 内建 | ballot `0x49249249`；master=3；dedup=8；inc/exc=496/465 |
| 11 | `fadd4`/`fmul4`/`accumulate` 回退 | 与标量逐位一致 |
| 12 | `warpgroup_reg_realloc` | 公式 512→128；`dec` 到 40 正常完成 |
| 13 | multimem / `cvt.rs` oracle | 求和顺序无关；进位比例 0.9997→0.7493→0.4992→0.2501→0 |

**踩坑记录**：
- `-arch=sm_90a` 在本机 nvcc 13.2 下 ptxas 目标仍是 `sm_90`，`setmaxnreg` 直接报
  `not supported on .target 'sm_90'`；**改用 `-gencode arch=compute_90a,code=sm_90a` 才正常**。
- `cp.async.mbarrier.arrive` 的计数**不是「每 lane 一次 arrival」**：它 issue 时 +1、完成时 −1，
  净零但会「卡住」到拷贝完成。第一版把 mbarrier init count 设成线程数 → `try_wait` 永远等不到，
  kernel 挂死；改成 `init=1` + elected 补一次 `arrive` 后与 DeepEP 一致。
- `cvt.rs.bf16x2.f32` 在 `sm_90a` 被 ptxas 拒绝（`Feature '.rs' not supported`），`sm_100a` 通过
  → 实证它是 **SM100 专属**（与 `docs/21` 的 `cvt.rs` 结论一致）。

## 11. 结论与边界

- `common/ptx.cuh` 是 DeepEP 的**设备端公共地板**：mbarrier、TMA/cp.async、multimem、谓词化访存、
  内存序、寄存器再平衡、warp 内建、浮点打包八组。前面第 7/11/53/58 篇里所有「裸操作」都源于此。
- **架构无关部分 H100 真跑 41/41 通过**，且回退分支与标量参考逐位一致。
- **严格 SM100 / 需多播映射、本文仅分析**：`st.bulk`（真指令）、256 位全局 ld/st、打包 `fadd2/fmul2`、
  融合 `add.rn.f32.bf16`、`cvt.rs.bf16x2`、`multimem.*`。本机对前三者验证回退、对后两者做语义 oracle。
- 本文补齐了「EP 通信算子 → 公共设备原语」的最后一层地板。至此，本专题从 Python host 编排
  （第 42 篇）到主 kernel（第 10/11 篇）再到设备原语层，DeepEP 侧已有完整纵切。
