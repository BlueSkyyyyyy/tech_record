# DeepGEMM 的 PTX 原语底座：内存序 / 原子 / 共享内存搬运 / warp 通信

> 本专题（0~27 篇）把 DeepGEMM 从 GEMM 主 kernel、attention、grouped 反向一路拆到 Mega MoE 的
> 通信-计算重叠，但这些 kernel 全都踩在**同一层没有系统讲过的裸 PTX 原语**上：
> `deep_gemm/include/deep_gemm/ptx/{ld_st,utils,tma}.cuh`、`common/{tma_copy,ring_pipeline}.cuh`。
> 本文补上这一层，讲清「为什么 Mega MoE 的 dispatch/combine 正确性、跨 SM / 跨 rank 的 readiness、
> grid_sync、tensormap 原地改」都依赖它；并在 H100 上把架构无关的部分**直接编译运行真实设备代码**验证语义。
>
> 这些原语都是 `__device__` 内联 PTX，**不是 SM100 专属**，sm90 可编译可运行。
> 目标卡（SM100）专属的部分（`cvt.rs` 随机舍入、`redux.sync.f32`）只做源码标注。

---

## 1. 为什么单开一篇：这层是「正确性」而非「性能」的来源

DeepGEMM 的上层 kernel 有两类逻辑：

- **性能逻辑**：TMA 流水、UMMA/WGMMA、block-scale——前面各篇已拆透；
- **正确性逻辑**：谁先谁后、写到远端后对方什么时候能看见、计数器怎么不重不漏——**全部编码在这一层 PTX 里**。

最典型的三个例子（本文都会回到它们）：

1. **Mega MoE combine 的 readiness 水位线**：epilogue 用 `ptx::st_rel_sys`（`st.release.sys`）把本 rank 的
   `grid_idx` 写到**远端 rank 的对称显存**（`sm100_fp8_fp4_mega_moe.cuh:1419`），combine 段用
   `ptx::ld_acq_sys`（`ld.acquire.sys`）自旋等对端**同一个** `grid_idx`（`:1440`）。这里 scope 必须是
   `sys`（跨 GPU），release/acquire 必须成对，否则会读到「数据没到、标志先到」。
2. **跨 SM 的 grid barrier**：`comm/barrier.cuh:33-51` 的 `grid_sync` 用
   `ptx::atomic_add_rel`（`atom.release.gpu`，`:42`）加一个「完成和」+ 低位负载，再用
   `ptx::ld_acq`（`ld.acquire.gpu`，`:45`）自旋等 tag 位翻转——**用一次原子加就完成了一次全 grid 计数**。
3. **L1→L2 的按块依赖位图**：L1 每完成一个 N block 就 `ptx::red_xor_rel`（`red.release.gpu.or`/`xor.b64`）
   翻转自己的位（`scheduler/mega_moe.cuh:157`），L2 用 `ptx::ld_acq_gpu` 读整张 64 位掩码
   并 XOR 掉上一代的期望值（`:164`）。`red`（不返回旧值）比 `atom` 便宜，且 release 保证位写入对读者可见。

把这些原语替换成「普通 `st` / `ld`」在这些场景下**会偶发挂死或读脏**——这正是本层的存在意义。

---

## 2. 原语分层与调用图

```
        ┌──────────────────────── DeepGEMM kernels ────────────────────────┐
        │  sm90/sm100 GEMM · attention · mega moe · mega gate/mhc · layout  │
        └───────┬───────────────┬───────────────┬──────────────┬───────────┘
                │               │               │              │
   common/ring_pipeline  ptx/tma.cuh    ptx/ld_st.cuh   ptx/utils.cuh
   （相位翻转）          （tensormap/   （内存序/原子/  （smid/shfl/
                          mbarrier/TMA）  搬运/cache）     redux/bf16）
                │               │               │              │
                └─────── common/tma_copy.cuh（2~5D copy_nd）─────┘
                                  │
                        common/{math,packing,types,exception}.cuh
```

- `ptx/ld_st.cuh`（387 行）：所有内存序与原子语义的唯一出口，也是共享内存搬运（`ldmatrix`/`stmatrix`/
  `cp.async`/`st.async.cluster`）的封装。
- `ptx/tma.cuh`（155 行）：tensormap 的 fence/原地改写、mbarrier 家族、`cp.async.bulk.tensor`。
- `ptx/utils.cuh`（89 行）：`%smid`/`%laneid`/`%gridid`、named barrier、`shfl`、`redux`、bf16 融合加法。
- `common/tma_copy.cuh`（106 行）：把「2~5 维 tensor map、2-CTA（`2SM`）/cluster multicast」统一成一个 `copy_nd`。
- `common/ring_pipeline.cuh`（40 行）：`(stage_idx, phase)` 环形推进——mega kernel 流水的骨架。

---

## 3. 内存序阶梯：relaxed / acquire / release × cta / gpu / sys

`ptx/ld_st.cuh` 的原子与访存 API 按「强度 × 作用域」张成一张表。**注意它没有提供 acquire load 的
32 位 sys 化之外的花样**——DeepGEMM 只挑它需要的组合，且每个组合都有真实调用点：

| API（`ptx::`） | PTX | 作用域 | 返回旧值 | 真实调用点 |
|---|---|---|---|---|
| `ld_acq` | `ld.acquire.gpu.global.b32` | gpu | — | `comm/barrier.cuh:45` grid_sync 自旋 |
| `ld_acq_gpu` | `ld.acquire.gpu.global.b64` | gpu | — | `scheduler/mega_moe.cuh:164` L2 位图；`mega_gate.cuh:28/37`、`mega_mhc.cuh:33/47` |
| `ld_acq_sys` | `ld.acquire.sys.global.{b32,b64,s32}` | sys | — | `comm/barrier.cuh:82` NVLink barrier；`sm100_fp8_fp4_mega_moe.cuh:1440` combine 等对端 |
| `ld_volatile` | `ld.volatile.global.{b32,b64}` | (无) | — | `scheduler/mega_moe.cuh:294/402` 轮询专家计数 / L1 任务计数 |
| `st_rel` | `st.release.gpu.global.u64` | gpu | — | 跨 SM 发布标志 |
| `st_rel_sys` | `st.release.sys.global.u64` | sys | — | `sm100_fp8_fp4_mega_moe.cuh:1419` / `bf16_mega_moe.cuh:1207` 发布 `grid_idx` 到远端 |
| `st_relaxed_sys` | `st.L1::no_allocate.relaxed.sys.global.u64` | sys | — | 弱序远端发布（配 `ld_acq_sys`） |
| `atomic_add_rel` | `atom.release.gpu.global.add.u32` | gpu | 旧值 | `comm/barrier.cuh:42`；`sm100_sparse_mqa_logits_metadata.cuh:445` 数末 CTA |
| `red_add_rel` | `red.release.gpu.global.add.u32` | gpu | —（不返回）| `sm100_fp8_fp4_mega_moe.cuh:613/1243` dispatch 计数通知 |
| `red_or_rel_sys` | `red.release.sys.global.or.b64` | sys | — | 远端位图置位 |
| `red_or_rel_gpu` | `red.release.gpu.global.or.b64` | gpu | — | 本卡位图置位 |
| `red_xor_rel` | `red.release.gpu.global.xor.b64` | gpu | — | `scheduler/mega_moe.cuh:157` L1 翻转位 |
| `red_async_inc_rel` | `red.async.release.gpu.global.add.u64` | gpu | —（fire-and-forget）| `mega_gate.cuh:32`、`mega_mhc.cuh:40` SplitBarrier |
| `red_add_rel_sys` | `red.release.sys.global.add.s32` | sys | — | `comm/barrier.cuh:75` NVLink barrier 信号 |
| `atomic_add[_sys]` | `atom[.sys].global.add.{u32,u64}` | gpu/sys | 旧值 | 见 B) 实验 |

### 三个「为什么」

- **为什么 combine 用 sys 而不是 gpu？** `gpu` 作用域只保证本 GPU（含其所有 SM + L2）内的可见性；
  对端 rank 的读写经过 NVLink，必须 `sys` 才能跨设备排序。`comm/barrier.cuh:75` 的
  `red_add_rel_sys(sym_buffer.map(signal_ptr, thread_idx), ±1)` 向每个远端 rank 的对称显存投递 ±1，
  本 rank 用 `ld_acq_sys` 等到累加到 `0` 或 `kNumRanks`（`:82`），加减交替、**相位免复位**。
- **为什么用 `red` 而非 `atom`？** `red`（reduce）不把旧值写回寄存器，省一次依赖；只有需要「我是第几个到达」
  时才用 `atom`（如 `atomic_add_rel` 返回旧值判断最后一个 CTA）。
- **为什么 `ld_volatile` 会出现在这里（无内存序）？** 轮询一个**单调递增的计数**（专家收数、L1 任务计数）时，
  调用方只关心「不小于某个阈值」，不需要跨变量的顺序，`volatile` 保证不被优化掉即可——比 acquire 便宜。

### H100 实测：生产者-消费者契约

实验 `src/ptx_primitives.cu` 内核 `k_mo`：64 个 slot，偶数 block（生产者）用普通 `st` 写满 128 个 payload
后发布标志，奇数 block（消费者）自旋到标志再读 payload 并校验。两种模式：

- mode=1：生产者 `ptx::st_rel`（`st.release.gpu`），消费者 `ptx::ld_acq_gpu`（`ld.acquire.gpu`）；
- mode=2（对照，弱序）：生产者裸存 `flag32[slot]=1`，消费者 `ptx::ld_volatile` 自旋。

200 轮 × 64 slot × 128 word：

```
MO mode=1 ... fail_words=0      ← release/acquire 契约：0 失败
MO mode=2 ... fail_words=0      ← 弱序在单卡 H100 上 200 轮也未观测到错读
```

**诚实说明**：单卡 H100 的 L2 一致性很强，200 轮内弱版本也没「撞」出错误；但这**不是** ISA 保证——
跨节点（`sys`）或更大规模竞争下弱序会失败。这里量到的是「契约正确」，而弱序的 0 失败只能说明
单卡 L2 够强，**不能**当成可以省掉 acquire/release 的依据。

---

## 4. `ptx/ld_st.cuh` 其余部分

### 4.1 矩阵搬运（LDSM / STSM）

- `SM90_U32x2_LDSM_N` / `SM90_U32x4_LDSM_N`（`:33-49`）：`ldmatrix.sync.aligned.x{2,4}.m8n8.shared.b16`，
  从共享内存把 8×8 b16 块按 lane 分发到寄存器，供 WGMMA 使用。
- `SM90_U32x2_STSM_N` / `_T` / x4 `_T`（`:51-85`）：`stmatrix...trans`，寄存器→共享（`.trans` 转置版）。
- `SM100_U8x4_STSM_T` / `SM100_U8x8_STSM_T`（`:87-107`）：`stmatrix.sync.aligned.m16n8.x{1,2}.trans.b8`，
  是 SM100 上 FP8 输出的专用转置存储（第 31 篇的 store 路径用到）。

### 4.2 共享内存 / 异步搬运

- `ld_shared` / `st_shared`（`:115-167`）覆盖 u32/f32/f32x2/f32x4/u32x4/b128；注意 `st_shared(const void*, x,y,z,w)`
  展开成 `st.shared.v4.u32`，`b128` 版用 `"q"` 约束。
- `cp_async_cg<kNumL2PrefetchBytes>`（`:169-179`）：`cp.async.cg.shared::cta.global.L2::{64,256}B`，
  16B 粒度、带 L2 预取提示。
- `cp_async_cg_zfill<...>`（`:182-192`）：末尾多一个 `num_src_bytes` 操作数，**拷前 N 字节、其余 16-N 字节清零**
  （`cp.async.cg ... , 16, %2`）。
- `cp_async_ca_zfill`（`:195-198`）：4B 版 `cp.async.ca`，用于 4B/0B 的零填充。
  这两者正是 sparse MQA logits 越界尾块的写法（`sm100_sparse_mqa_logits.cuh:477/493`）。
- `mapa_shared`（`:200-204`）：`mapa.shared::cluster.u32`，把本 CTA 的共享地址映射到 cluster 内某个 CTA。
- `st_async_cluster`（`:206-224`）：`st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`，
  **写远端 CTA 共享内存并自动给远端 mbarrier 记 bytes**。Mega MoE 调度器用它把 `TaskInfo` 广播给伙伴 CTA
  （`scheduler/mega_moe.cuh:412`，先 `arrive_and_expect_tx(sizeof(task_info_t))` 再 `st_async_cluster`）。
- `st_shared_bulk`（`:226-231`）：`st.bulk.weak.shared::cta`，一次把整段共享内存清零
  （`sm100_fp8_fp4_mega_moe.cuh:245`、`sm100_bf16_mega_moe.cuh:200`，dispatch 段清 workspace）。

### 4.3 全局访存 / cache hint / 预取

- `kEvictFirstCacheHint = 0x12f0000000000000`（`:234`）+ `ld_evict_first`（`:236-242`）：
  `ld.weak.global.L1::no_allocate.L2::cache_hint.v4.b32`，**不污染 L1、且标记为 L2 即将淘汰**。
  HC epilogue（`sm100_mega_mhc.cuh:198/309`）与 sparse metadata（`sm100_sparse_mqa_logits_metadata.cuh:256`）
  在流式读取一次性数据时用它。
- `st_global_low_bf16`（`:245-247`）：`st.global.b16` **只写低 16 位**（32 位源寄存器被截断到 b16）。
- `ld_global_cg`（`:249-253`）：`ld.global.cg.u32`，绕过 L1（只走 L2）。
- `ld_gez_pred`（`:366-379`）：`setp.ge.s32` + 谓词化 `ld.global.L2::256B.v4.s64`，谓词为负时返回 0——
  无分支的安全越界读。
- `prefetch_l1`（`:382-384`）：`prefetch.global.L1`。

### 4.4 fence / 其它原子

- `fence_acq_rel_cta`（`:110-112`）：`fence.acq_rel.cta` 的 CTA 级围栏。
- `atomic_add[_sys]`（`:284-300`）：32/64 位原子加，返回旧值。
- `st_rel`（`:308-310`）：64 位 release 存。

---

## 5. `ptx/utils.cuh`：warp 与身份原语

| API | 指令 / 语义 | 真实调用点 |
|---|---|---|
| `get_sm_idx` / `get_lane_idx` / `get_grid_idx`（`:11-27`） | `%smid` / `%laneid` / `%gridid` | 调度与域分区、`grid_idx` 水位线 |
| `sync_aligned`（`:29-31`）/ `sync_unaligned`（`:33-35`） | `bar.sync` vs `barrier.sync` | named barrier（warp 数不整除时用后者） |
| `exchange`（`:37-47`） | 按 lane 的 `__shfl_sync`（支持任意 POD） | paged MQA logits 的 block-table 广播（第 24 篇） |
| `reduce_max_sync`（`:49-53`） | `redux.sync.max.f32` | Mega Gate 确定性 top-k 破平（`epilogue`；**SM100 专属指令**） |
| `cvt_relu_bf16x2_f32`（`:55-63`） | SM100 `cvt.rn.relu.bf16x2.f32`；否则 `__floats2bfloat162_rn(fmaxf)` | ReLU+bf16 打包 |
| `accumulate`（`:65-74`） | SM100 `add.rn.f32.bf16`（bf16 直接加进 f32）；否则转 f32 再加 | combine 段 fp32 累加 |
| `accumulate_square`（`:76-86`） | SM100 `fma.rn.f32.bf16`（就地平方累加）；否则转 f32 再 `x*x` | HC prenorm 的平方和搭便车（第 16 篇） |

**为什么要有 SM100 融合指令分支**：`add.rn.f32.bf16` / `fma.rn.f32.bf16` 让「bf16 操作数 → fp32 累加」
在**一条指令**里完成，省掉 `cvt` 且不经过中间 f32 寄存器；SM90 只能走回退路径（`__bfloat1622float2` + 加）。
回退路径的数值语义 = bf16 精确转 f32 后运算，因此**与融合指令逐位一致**（bf16→f32 是精确的）。

### H100 实测

- `cvt_relu_bf16x2`（走 `#else` 回退）：65536 对输入，与**独立实现的 RNE bf16 舍入**（位运算 `+0x7fff+lsb`）
  逐位对比 **mismatch=0/65536**。样例：`in=(-1.5,-0.996)→relu_bf16=(0,0)`；`in=(0.516,1.02)→(0.5156,1.0234)`。
- `accumulate` / `accumulate_square` 回退：`acc=(0.5156,1.0234)`、`sq=(0.2659,1.0474)`，与「bf16 精确转 f32
  后相加 / 平方」一致。

---

## 6. `ptx/tma.cuh`：tensormap 原地改、mbarrier、TMA

### 6.1 tensormap 的 fence 与原地改写

- `tensor_map_release_gpu`（`:9-11`）/ `tensor_map_acquire_gpu`（`:13-16`）：
  `fence.proxy.tensormap::generic.{release,acquire}.gpu`——tensormap 是「代理内存」，改写前后必须 fence。
- `tensor_map_replace_global_addr_in_smem`（`:18-22`）：
  `tensormap.replace.tile.global_address.shared::cta.b1024.b64`——**在共享内存里原地改 tensormap 的基址**。
- `tensor_map_replace_global_inner_dim_stride_in_smem`（`:24-32`）：
  同时改 `global_dim` 与 `global_stride`（CUDA ≥ 12.3）。

这正是第 12 篇 wgrad `k_grouped_fp8_gemm_nt_contiguous` 的做法：base tensormap 只能按首个非空组的 K 造，
kernel 里切组时**原地 patch** 基址 `+k_start*M` 与 inner stride，再 `commit/wait` + release/acquire 防竞态
（`sm90_fp8_gemm_1d1d.cuh:193-215`，对应 PR #343）。

### 6.2 mbarrier 家族

| API | 语义 | 用途 |
|---|---|---|
| `mbarrier_arrive`（`:35-39`） | `mbarrier.arrive.shared::cta.b64` | 普通到达 |
| `mbarrier_arrive_count`（`:41-45`） | 带计数到达 | 一次到达记多个 |
| `mbarrier_arrive_count_pred`（`:47-56`） | 谓词化到达 | 只 elect-one 线程到达 |
| `mbarrier_arrive_and_set_tx`（`:58-62`） | `arrive.expect_tx` | TMA 生产者声明将到的字节数 |
| `mbarrier_wait_and_flip_phase`（`:64-78`） | `try_wait.parity` 自旋 + `phase ^= 1` | 消费者等相位翻转（内联 PTX 忙等，带超时立即数 `0x989680`） |

`mbarrier_wait_and_flip_phase` 的相位翻转约定与第 09 篇（mbarrier 精读）呼应：**生产者 phase 从 1 开始**
（buffer 初空），消费者每次 `try_wait.parity` 成功后 `phase ^= 1`。

### 6.3 TMA 指令

- `tma_load_1d`（`:80-94`）：`cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`，
  默认 hint 是 `EVICT_FIRST`（读进来的一维数据马上要被消费，尽快淘汰）。
- `tma_store_1d/2d/3d`（`:96-132`）：`cp.async.bulk[.tensor.Nd].global.shared::cta.bulk_group`，
  默认 hint 是 `EVICT_NORMAL`（写出去的数据马上要用回）。
- `tma_store_wait<kNumRemainingWaits>`（`:134-138`）：`cp.async.bulk.wait_group`。
- `tma_gather4`（`:140-153`）：`cp.async.bulk.tensor.2d...tile::gather4`——一次 TMA 抓 4 个任意行
  （paged MQA logits 的物理块定位相关）。

> `tma_load_1d` 的 `dst` 是 `shared::cluster`（可落到 cluster 内远端 CTA 的共享内存），
> 这解释了 Mega MoE dispatch 段为何能「分 chunk TMA 远端 pull」（第 08 篇）。

---

## 7. `common/tma_copy.cuh`：2~5 维 copy_nd 与 2-CTA / multicast

- `get_inner_block_atom_size<...>`（`:12-15`）：内层 atom 的存储尺寸——`swizzle==0` 时是
  `BLOCK_INNER / kPackFactor`（FP4 打包因子 2），否则是 `swizzle / sizeof(dtype)`。
- `LoadOps<2..5>`（`:17-37`）：每一维对应 `SM90_TMA_LOAD_Nd` / `SM100_TMA_2SM_LOAD_Nd` / `SM90_TMA_LOAD_MULTICAST_Nd`
  三个实现。
- `copy_nd`（`:39-89`）：按 `BLOCK_INNER / BLOCK_INNER_ATOM_LOGICAL` 拆成若干 atom 循环，
  按 `num_tma_multicast` 三分支：
  - `==1`：普通 `Load::copy`；
  - `>1` 且 arch≥1000：`Load2SM`（**2-CTA 只给 leader CTA 发信号**）；
  - `>1` 且 arch≥900：只有 `block_rank_in_cluster()==0` 发 `LoadMulticast`（`(1<<num)-1` 掩码）。
- `copy`（`:91-104`）：2D/3D 的便捷壳（`kIs3DTMA` 决定带不带 `batch_idx`）。

**为什么 2-CTA 和 multicast 分开**：SM100 的 `UMMA_2x1SM`（2-CTA MMA）天然让两个 CTA 共享一次 TMA；
SM90 的 WGMMA 没有 2-CTA 变体，只能由 leader CTA 广播（multicast）。同一份 `copy_nd` 用 arch 分支
把两种语义封在一个接口后。

---

## 8. `common/ring_pipeline.cuh`：流水相位翻转

`RingPipeline<kNumStages>`（`:14-38`）只有 `stage_idx` 与 `phase` 两个字段，`advance(step)` 返回推进前的
`(stage_idx, phase)`：

- **幂次容量快路**（`:26-28`）：`kNumStages` 是 2 的幂时，`%` 与 `/` 降级为位运算
  （`stage_idx = next % kNumStages; phase ^= next / kNumStages`）；
- **通用路径**（`:29-35`）：`stage_idx += step`，超过容量就回绕并 `phase ^= 1`。

注释点明「callers advance by at most one full ring, so the phase toggles at most once」（`:20`）——
这正是所有 mega kernel「双缓冲 / N 级流水」相位管理的基石。

H100 实测 `k_ring<{1,2,4,8}>` 跑 40 步，打印 `(stage,phase)` 轨迹，与 Python 独立复刻
（同一套 modulo/异或规则）**逐项 0 不符**：

```
RING ns=1 ... 0,0 0,1 0,0 0,1 ...
RING ns=2 ... 0,0 1,0 0,1 1,1 ...
RING ns=4 ... 0,0 1,0 2,0 3,0 0,1 1,1 2,1 3,1 ...
RING ns=8 ... 0,0 ... 7,0 0,1 ... 7,1 ...
```

---

## 9. H100 实测汇总（`harness/exp_ptx_primitives.{sh,out.txt}`、`src/ptx_primitives.cu`）

真机：H100 80GB HBM3 / sm90 / driver 570.211.01；`ptx::` 原语全部来自真实头文件
（sha256：`ld_st.cuh 912e9f9a…`、`utils.cuh 8a9d5e03…`、`ring_pipeline.cuh d231e0ce…`）。总判定 **ALL PASS**。

| 组 | 测什么 | 结果 |
|---|---|---|
| A | `st.release.gpu` + `ld.acquire.gpu` 生产者-消费者 | 200 轮×64 slot×128 word，**fail=0**；弱序对照也 0（单卡 L2 强，非 ISA 保证） |
| B | `atomic_add`(返回旧值) vs `red_add`/`red_add_rel` | 三条路径求和都 = 524800 = 1024·1025/2；`atomic` 返回的 1024 个旧值**互不相同** 1024/1024 |
| C | `RingPipeline::advance` 相位轨迹 | ns∈{1,2,4,8}、40 步，与 Python 独立复刻**逐项 0 不符** |
| D | `cp_async_cg<256>` + `cp_async_cg_zfill<256>`(8B) | 128 块 **mismatch=0**（后 8 字节正确清零） |
| D | `cp_async_ca_zfill`(0/4B) | 64 块 **mismatch=0**（0B→全零、4B→低 4 字节对） |
| E | `ld_evict_first` 正确性 / 流式带宽 | 256 MiB 读：普通 `ld` 0.238 ms / **1129 GB/s**，`ld_evict_first` 0.210 ms / **1279 GB/s**（不污染 L1，约 +13%） |
| F | `cvt_relu_bf16x2`（回退）/ `accumulate(_square)` | pack 与独立 RNE 参考 **mismatch=0/65536**；acc/sq 与「bf16→f32 精确」一致 |

**未能测的**：`redux.sync.max.f32`（SM100 专属指令，sm90 ptxas 不支持）与 `grid_sync` 的整体语义
（需要全 grid block 常驻；本机 GPU 被多个自驱任务共享，132 个 block 无法同时驻留，会死锁——
已从实验中移除，仅在 §3 做源码级解读）。`cvt.rs.bf16x2` 同理（第 31 篇已证 sm90 报 `Feature '.rs' not supported`）。

---

## 10. 结论与边界

1. **这层不是可有可无的封装**：Mega MoE 的 dispatch/combine 正确性完全由 `sys` 作用域的
   release/acquire + `red`/`atom` 的可见性语义决定，换成普通访存会偶发失败。
2. **作用域阶梯要记牢**：`cta` < `gpu` < `sys`；本卡内同步用 `gpu`（便宜），跨 rank 才升到 `sys`。
3. **`red` 和 `atom` 的取舍**：需要「第几个到达」→ `atom`（返回旧值）；纯通知 → `red`（省依赖）；
   fire-and-forget 计数 → `red.async`。
4. **tensormap 是代理内存**：原地改前 `release`、读前 `acquire`——第 12 篇 wgrad 的按组改描述符依赖它。
5. **架构无关性**：绝大多数原语（内存序、原子、`cp.async`、`st.async.cluster`、`RingPipeline`）sm90 可跑，
   本文已在 H100 逐条验证；仅 `cvt.rs`、`redux.sync.f32`、2-CTA `Load2SM` 是 SM100 专属。

> 配套实验：`code/mega-moe/src/ptx_primitives.cu`、`code/mega-moe/harness/exp_ptx_primitives.{sh,out.txt}`。
