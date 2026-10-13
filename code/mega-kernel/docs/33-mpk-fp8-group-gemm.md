# MPK SM100 block-scaled FP8 MoE group-GEMM device 内部走读（`fp8_group_gemm_sm100.cuh`，P119-1）

> 定位：本文补掉 `docs/32-mpk-sm100-linear-tasks.md` **§7「局限」明留的「未展开的更深层」**——
>
> > **未展开**：`fp8_group_gemm_sm100.cuh` 的 UMMA 主循环 / UTCCP 序列 / epilogue scatter 仍未逐行
> > （本文把重心放在 `linear_fp8_sm100` 家族与生产布线）；身体 B 的 CuTe 主循环只核了结构与惯用法，
> > 未逐行走完 727 行；`mla_mtp_tpN_main` 主体仍未走。
>
> 与 `docs/31`（P117-1）§5 只走读了「warp 角色 + scale 布局契约」的粗粒度高程图不同，本文把
> `fp8_group_gemm_sm100.cuh`（**1567 行**）的**三条 device 数据通路逐段打开**：warp5 DMA 的
> TMA + 手写 swizzle `cp.async` + routing 检查、warp4 MMA 的 **UTCCP scale 搬运 + 4×block-scaled
> UMMA + 描述符预计算/`__shfl` 广播**、warp6 SCALE 的 fp32→UE8M0→pack→**warp 转置**，以及
> warps0-3 epilogue 的 **TMEM→寄存器→BF16→routing scatter**。快照 `mirage-project/mirage` 分支
> `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/32` 同快照）；**纯源码走读、无新实验**——SM100
> （`sm_100f`）在目标卡 H100/sm90 不可执行。

---

## 0. 两条 headline

1. **「一个 task = 一个 CTA」在 SM100 block-scaled FP8 group-GEMM 上是「1 个 CTA 跑 3 条 warp 特化
   流水」**：256 线程 = 8 warp，`warp5 DMA`（生产者）/`warp6 SCALE`（尺度生产者）/`warp4 MMA`
   （消费者+生产者）/`warps0-3 EPILOGUE`（消费者）由 **6 个 `mbarrier` 串联成三段生产者-消费者
   环**（A/B 数据环 `a_full/b_full/ab_empty`、尺度环 `sf_ready`、累加器环 `acc_full/acc_empty`），
   外加 2 个 `NamedBarrier` 做 CTA 内跨 warp 会合。这是 `docs/07`（Hopper `linear_hopper.cuh`）
   「warp 特化 + TMA/mbarrier」惯用法在 **SM100 block-scaled** 上的直系后代，多出来的是**尺度因子
   的第三条流水**（`docs/07`/`docs/08` 没有）。
2. **生产布线新增一条**：`register_moe_fp8_sm100_task`（`task_register.cc:3204`）把这份 device 身体
   以 **`MMA_N=16`、`num_ab_stages=8`** 的实参发射成 `TASK_MOE_W13_FP8_SM100=248` /
   `TASK_MOE_W2_FP8_SM100=249` 两个 `TaskType`（`task_config=(6,1,...)`，6 输入 1 输出），
   `expert_offset = bid.x`（`runtime.cc:356-359`），`EXPERT_STRIDE = grid_dim.x`
   （`task_register.cc:3298`）——**2D grid 的 x 维切专家、y 维切 N**。这补齐了 `docs/31` §10
   「`docs/24` 的 SM100 `TaskType` 登记未全表复核」这条尾巴里**属于本文件的那一支**。

---

## 1. 定位：数学操作、swapAB 约定与 6 输入 1 输出

文件头注释（`:49-130`）把语义写死。对每个分配给本 task 的专家 `e`：

- `output[token, topk_slot, :] = input_fp8[token, :] @ weight_fp8[e, :, :].T`（`:54-55`）；

其中 `input_fp8:[batch,K]`（FP8 E4M3）、`weight_fp8:[num_experts,N,K]`（行主序）、
`output:[batch,num_topk,N]`（**BF16**）。

**swapAB 约定**（`:63-72`）是关键：UMMA 硬件算 `C[M,N] += A[M,K]*B[N,K]`（两者 K 主序），本 kernel
把 **UMMA 的 A ← weight `[N_weight,K]`**（M 维 = 权重行 = 输出维）、**UMMA 的 B ← input `[B_tokens,K]`**
（N 维 = token = 批维）、**C ← `[N_weight,B_tokens]`**。即 **MMA_M = 输出维、MMA_N = token 维**——
这与「常规 GEMM 把 token 当 M」相反，是 MoE 里「权重行才是大 N、token 是小 M」的自然选择。

**尺度因子格式**（`:74-87`）：FP8 E4M3 + **per-block float32 尺度**，在 kernel 内转 UE8M0：
- SFA（权重尺度）`weight_scale [num_experts, N, K/128]`，**per-row per-128-K-block**；
- SFB（激活尺度）`input_scale [batch, K/128]`（`W13_LINEAR=true`）或 `[batch, num_topk, K/128]`
  （`W13_LINEAR=false`），per-token per-128-K-block。

> 这正是 `docs/31` §5.1 强调的**「同一份 128K 块 + UE8M0 语义、两种载体」**里 **MoE 的一侧**：
> 这里传进来的是 **fp32**，由 warp6 现场转 UE8M0；linear 单测则在 host 就打包成 uint32
> （`docs/31` §2）。本文件头 `:76-77` 的原话就是「**converted to UE8M0 in-kernel**」。

**8 warp 角色**（`:89-108`，与 `docs/31` §5.2 一致，此处给数据流坐标）：

| warp | 线程 | 角色 | 关键 action |
| --- | --- | --- | --- |
| 0-3 | 0-127 | EPILOGUE | TMEM→寄存器→BF16→routing scatter（`:1440-1552`）|
| 4 | 128-159 | MMA | 等 A/B/scale → `UTCCP` → 4×FP8 UMMA（`:1017-1323`）|
| 5 | 160-191 | DMA | TMA 载 weight + `cp.async` 载 input（带 routing 检查）（`:806-988`）|
| 6 | 192-223 | SCALE | fp32→UE8M0→pack→warp 转置（`:1339-1419`）|
| 7 | 224-255 | IDLE | 落在 `if-else` 链外（`:1553`）|

**2D 分工**（`:110-119`）：`grid_dim.x` 切专家（`EXPERT_STRIDE=X`，CTA `i` 处理专家
`i, i+X, i+2X…`）、`grid_dim.y` 切输出维（`OUTPUT_SIZE = ORIG_OUTPUT_SIZE / Y`）。

---

## 2. 共享内存布局、barrier 与 TMEM 列布局

### 2.1 `MoEFP8SharedStorage`（`:169-256`）

模板 9 个参数中前 3 个是 **placeholder**（`:169-174`，注释说明只用于模板推导、结构体不用它们）。
真正决定布局的是 `Num_Experts / Num_AB_Stage / Num_ACC_Stage / MMA_M / MMA_N / bK`：

- **A（weight）**：`alignas(128) uint8_t A[Num_AB_Stage][MMA_M*bK]`（`:186`），TMA 带 SWIZZLE_128B；
- **B（input）**：`alignas(128) uint8_t B[Num_AB_Stage][MMA_N*bK]`（`:192`），手写 swizzle；
- **SFA/SFB**：`uint32_t sfa_smem[Num_AB_Stage][128]` / `sfb_smem[…]`（`:205-207`），**SWIZZLE_NONE**，
  每项 128 个 uint32（每 uint32 = 4 个 UE8M0 字节，对应 128-K 块内 4 个 32 元素 K-subtile；因为
  粒度是 per-128-K，**4 个字节是同一个 UE8M0 值的 4 份复本**，见 `:194-198`）；
- `expert_mask[Num_Experts]`（`:250`）：**注释明说当前 impl 未用**（专家直接读 `mMask` 全局张量）；
- `tmem_base_ptr`（`:255`）：TMEM 基列地址，由 warp0（epilogue）分配后经 named barrier 广播。

> 关于用「裸 `uint8_t` 数组而不用 CuTe layout 张量」的理由（`:163-168`）：CuTe 的 `smem_ptr_flag_bits`
> 会断言元素大小 == swizzle atom 大小，`uint8_t` + SWIZZLE_128B 会触发断言；于是 **A 靠 TMA 自动
> swizzle、B 在 DMA warp 手算 swizzle**。

### 2.2 六个 `mbarrier`（`:209-244` + 初始化 `:632-651`）

| barrier | 类型 | init 计数 | 谁 signal | 谁 wait |
| --- | --- | --- | --- | --- |
| `a_full` | **ClusterTransactionBarrier** | 1 | TMA 字节数完成 | MMA（warp4）|
| `b_full` | ClusterBarrier | 32 | 32 lane `cpasync_barrier_arrive_noinc` | MMA |
| `ab_empty` | ClusterBarrier | 1 | MMA `umma_arrive` | DMA + **SCALE** |
| `acc_full` | ClusterBarrier | 1 | MMA `umma_arrive` | EPILOGUE |
| `acc_empty` | ClusterBarrier | 4 | EPILOGUE 4 warp 各 1 elected | MMA |
| `sf_ready` | ClusterBarrier | 1 | SCALE warp `elect_one_sync` | MMA（发 UTCCP 前）|

初始化只在 **warp0** 做（`:632`），用 `initialize_barrier_array_aligned`；随后 `fence_barrier_init()`
（`:761`）+ `__syncthreads()`（`:769`）保证对全体可见。

### 2.3 两个 `NamedBarrier`（`:654-666`）

- `tmem_allocation_result_barrier`：计数 `32+128=160`（分配 warp + 4 epilogue warp），
  ReservedNamedBarriers::TmemAllocBarrier；火后 `tmem_base_ptr` 有效（`:658-662`）；
- `epilogue_wg_barrier`：计数 128（4 epilogue warp），确保读 TMEM 完成再 signal `acc_empty`（`:663-666`）。

### 2.4 TMEM 列布局（`:460-498`）

SM100 的 TMEM 是 128 行 × N 列的片上存，UMMA 直接读写。本 kernel 分配三段：

```
[ACC_0 | ACC_1 | ... | SFA_0..3 | SFB_0..3]
|--- num_tmem_acc_cols ---||-- 4 --||-- 4 --|
```

`num_tmem_acc_cols = MMA_N * NUM_ACC_STAGE`（`:481`；**生产 `MMA_N=16` → 32**，注释里的
`128*2=256` 是「MMA_N=128 时」的示例）。加 4+4 个尺度列后，**TMEM 列数必须向上取整到 2 的幂**
（64/128/256/512，`:491-498`）——生产 `32+8=40 → 64` 列。

### 2.5 CuTe 的「BF16 代理」（`:500-523`）——一个可迁移的工程技巧

CuTe 在这个 CUTLASS 版本里**FP8 MMA traits 不全**，于是代码用一个 **BF16 `TiledMMA` 当代理**
（`SM100_MMA_F16BF16_SS<bf16,bf16,float,MMA_M,MMA_N,K,K>`，`:512-519`）来推导**累加器 fragment
形状与 TMEM 布局**。理由是「FP8 与 BF16 UMMA 在相同 `(MMA_M,MMA_N)` 下产生**完全相同**的 FP32
TMEM 累加器布局」（`:503-511`）。真正的 FP8 UMMA 则**手写 PTX** 发（见 §4）。这也是 `:1428-1431`
说 epilogue「与 BF16 MoE task 架构上完全相同」的原因。

---

## 3. warp5 DMA：TMA weight + 手写 swizzle `cp.async` input

DMA warp 的三重循环顺序是 **专家 → m_tile → n_tile → k_tile**（`:817-987`），
`total_k_tile_count` 跨所有 work unit 累计，用于算**旋转 smem buffer 与 barrier phase**
（`:809-813`）。核心是**乐观 peek + lookahead**（`:837-842`、`:972-979`）：先
`try_wait_barrier(ab_empty)` 非阻塞探一次，若已空则省掉阻塞等待；每个 K-tile 结尾又**前瞻下一个**
buffer 的 `ab_empty`（`:975-979`），把 barrier 检查与当前 TMA/`cp.async` 执行重叠。

### 3.1 weight（操作数 A）：TMA 批量拷贝

只一条 lane（`elect_one_sync`）发指令（`:876-892`）：

- TMA 坐标 `{k_tile*TILE_SIZE, m_tile*OUTPUT_ATOM_SIZE + expert_idx*ORIG_OUTPUT_SIZE}`（`:877-879`）
  ——即 `(K_offset, M_offset_in_global_weight_tensor)`；
- `weight_smem.set_ptr(...)` 指向该 stage buffer，`set_barrier_transaction_bytes(a_full, MMA_M*bK)`
  （=`128*128=16384` 字节，`:685`），再 `tma_cp_async(a_full, …)`。

### 3.2 input（操作数 B）：手写 `cp.async` + routing 检查 + 显式 swizzle

**不用 TMA**，因为每个 token 是否要载入取决于 MoE routing（`:894-914`）。32 lane 协作，每 lane
`512` 字节 = `32` 条 16 B `cp.async`（`:917-923`）。对每条 16 B：

- 算出 `(row,col)`，`token_idx_n = n_tile*MMA_N + row`；
- 读 routing `topk_idx_n = tRoutingIndex(token_idx_n)`（带 `BATCH_SIZE` 越界守卫，`:931-933`）；
- **显式 SWIZZLE_128B**：`swizzled_off = linear_off ^ (((linear_off>>7)&7)<<4)`（`:935-937`）；
- 若 `token_idx_n < BATCH_SIZE && topk_idx_n > 0` 则 `cp.async.ca … 16` 从
  `mInput(token_idx_n, [topk,] k_tile*bK+col)` 载入（`W13_LINEAR` 决定索引形态，`:941-956`）；
  **否则写 16 字节零**（`:957-960`）——保证 UMMA 吃到中性数据。
- 结尾 `cp.async.commit_group`，再 32 lane `cpasync_barrier_arrive_noinc(b_full)`
  （`:962-970`）。

> **工程点 1**：A 用 TMA（硬件自动 swizzle）、B 手写 swizzle——同一 tile 两种载入路径，差异唯一
> 来源就是 per-token routing。**工程点 2**：TMA 的 `set_barrier_transaction_bytes`（字节计数）
> 与 `cp.async` 的 `b_full`（32 lane 到达计数）**用两种不同的 barrier 语义**，见 §2.2。

---

## 4. warp4 MMA：UTCCP + 4×block-scaled UMMA

MMA warp 先等 `tmem_allocation_result_barrier` 拿到 `tmem_base_ptr`（`:1018-1021`），算出
`sfa_tmem = base + num_tmem_acc_cols`、`sfb_tmem = base + num_tmem_acc_cols + 4`（`:1026-1028`）。

### 4.1 指令描述符与操作数描述符（`:1034-1099`）

- `instr_desc = make_instr_desc_block_scaled<e4m3,e4m3,float,ue8m0,MMA_M,MMA_N,K,K>()`
  （`:1042-1050`）——编码 dtype/形状/major；
- `sf_desc = kernel::sm100::make_sf_desc(nullptr)`（`:1057`）；
- `a_desc = make_umma_desc<Major::K, MMA_M, bK, 128>(&A[0][0],0,0)`、
  `b_desc = …<Major::K, MMA_N, bK, 128>(&B[0][0],0,0)`（`:1068-1078`）。

**描述符预计算 + `__shfl` 广播**（`:1080-1099`）：`NUM_AB_STAGE` 个 stage 的 descriptor **lo-word**
（含 smem 基址，16 B 粒度）分别预存到 **lane 0..(STAGE-1)** 的寄存器；运行时用
`__shfl_sync(0xffffffff, a_desc_lo, smem_rd_buf)` 把当前 stage 的 lo-word 广播给全体（`:1214-1217`）——
省掉每 stage 重算描述符。

### 4.2 内层：每 K-tile 三件事（`:1151-1311`）

对每个 K-tile（同样用 peek/lookahead 等 `a_full/b_full/sf_ready`，`:1128-1175`）：

1. **UTCCP**（`:1177-1199`）：`elect_one_sync` 下，先把 `sf_desc` 地址 replace 成
   `sfa_smem[smem_rd_buf]`，`UTCCP_t::copy(sf_desc, sfa_tmem)`；SFB 同理。
   `UTCCP_t = SM100_UTCCP_4x32dp128bit_1cta`（`:1032`）——**4 列 × 32 深 × 128 bit、单 CTA**，
   即一次把 128 个 uint32 搬进 TMEM 尺度列。随后 `__syncwarp()`（`:1199`）。
2. **4× UMMA**（`:1201-1281`）：`bK/UMMA_K = 128/32 = 4` 个 K-subtile，每个：
   - `sfa_id = sfb_id = k_sub`（`:1232-1233`）——选择 packed uint32 的第几个字节
     （因为 4 字节是复本，per-128-K 粒度下选哪个都一样，`:1226-1231`）；
   - `make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id)`（`:1237-1239`）；
   - `advance_umma_desc_lo<Major::K,…>(base_lo, 0, k_sub*UMMA_K)` 把 A/B 描述符推进到该 sub-tile
     （`:1245-1256`）；
   - **`SM100_MMA_MXF8F6F4_SS::fma(a_desc,b_desc, tmem_c, scale_c, runtime_instr_desc, sfa_tmem, sfb_tmem)`**
     （`:1271-1278`）。`scale_c = (first_tile && k_sub==0)?0u:1u`（`:1275`）——**首个 sub-tile 覆盖**、
     其后全部累加。
   - **警告注释**（`:1012-1015`、`:1220-1222`）：`fma()` 内部自带 `elect_one_sync`，**不要在外面再套
     一层**，否则死锁。
3. **`umma_arrive(ab_empty)`**（`:1288-1289`）：`umma_arrive` 会**等所有在飞 UMMA 完成**再到达，
   故 DMA 不会覆盖 UMMA 还在读的 smem。

一个 work unit 全部 K-tile 累完后，`umma_arrive(acc_full)` 通知 epilogue（`:1317-1318`）。

### 4.3 底层 PTX（`sm100_utils.cuh:127-150`）

`SM100_MMA_MXF8F6F4_SS::fma` 是一段内联 PTX：
`tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [tmem_c], desc_a, desc_b, desc, [tmem_sfa], [tmem_sfb], p;`
（`sm100_utils.cuh:138-139`）。pred `p = (scale_c != 0)` 就是 accumulate 标志。
`make_runtime_instr_desc_with_sf_id` 把 `a_sf_id_/b_sf_id_` 塞进 `InstrDescriptorBlockScaled` 再左移
32 位（`sm100_utils.cuh:116-123`）。`make_umma_desc` 对 K-major 算 `SBO = 8*BLOCK_K*sizeof(dtype)`、
`LBO=0`、`SWIZZLE_128B`（`sm100_utils.cuh:94-114`）。

> **可迁移点**：`advance_umma_desc_lo` 只支持 K-major（`static_assert`，
> `sm100_utils.cuh:81-82`）；`make_umma_desc` 断言 `kSwizzleMode == BLOCK_K*sizeof(dtype)`
> （`:98-100`）。这两条把「描述符推进」简化成线性加，但也就**限定在 K-major + swizzle 全宽**。

---

## 5. warp6 SCALE：fp32→UE8M0→pack→warp 转置

这是本文件相对 `docs/07`/`docs/08` **新增的第三条流水**（把尺度工作从 MMA warp 关键路径剥离，
`:1336-1337` 明说「overlaps it with DMA loads and prior UMMA execution」）。三重循环与 DMA 同序，
`smem_buf = (num_prev_k_blk + k_tile) % NUM_AB_STAGE`（`:1355`）。

**先等 `ab_empty` 再写**（`:1357-1368`，带 `FIX 2026-04-22` 注释）：这是**一个真实的竞态修复**——
早期版本 scale warp 不等 MMA 读完就覆写 `sfa/sfb_smem`，在 **TP=4、w2、mbt=64
（`k_tile_count == NUM_AB_STAGE == 8`）**时 scale warp 会绕回并覆盖 UTCCP 还在读的 buffer；
`ab_empty` 由 MMA 的 `umma_arrive`（`tcgen05.commit`）触发，会等所有 tcgen05（含 UTCCP）完成，
故正好把 scale warp 与 UTCCP 读隔开。

- **SFA**（`:1370-1380`）：`i = lane_idx; i < 128; i += 32`，`row = m_tile*MMA_M + i`、
  `global_row = expert_idx*ORIG_OUTPUT_SIZE + row`，读 `mWeightScale(global_row, k_tile)`（越界填
  1.0f），**UE8M0 = `(__float_as_uint(sf)>>23)&0xFF`**（只取 fp32 指数），再 pack 成
  `u | u<<8 | u<<16 | u<<24`；
- **SFB**（`:1382-1403`）：默认 `0x7F`（UE8M0(1.0) 填充），仅 `i<MMA_N` 且 token 被路由时读
  `mInputScale(...)` 转 UE8M0；同样 pack；
- `__syncwarp()` 后对两个 buffer 各做一次 **`fp8_utccp_warp_transpose`**（`:1407-1408`），再
  `fence_view_async_shared()`，最后 `elect_one_sync` → `arrive_barrier(sf_ready)`（`:1411-1414`）。

**warp 转置算法**（`:298-336`）：128 个 uint32 原地转置，`lane` 读
`smem[(i ^ (lane>>3))*32 + lane]`（i=0..3）、写 `smem[lane*4 + (i ^ (lane>>3))]`——一个
**4×32 → 32×4** 的 warp 内转置，**注释原话「identical to DeepGEMM's
`utccp_required_smem_warp_transpose`」**（`:314`）。因为 UTCCP 期望**列主序**，而 scale 载入代码
按行主序写（`:302-319`）。

> **可迁移点**：换算「4 字节复本 + 1 次 warp 转置」是 per-128-K 量化的**性能优化**（避免 per-32-K
> 的细粒度尺度路径，`:1226-1231`），代价是**牺牲了未来换 per-32-K 粒度的灵活性**（源码已注明）。

---

## 6. warps0-3 epilogue：TMEM→寄存器→BF16→routing scatter

### 6.1 TMEM 分配与加载（`:1441-1480`）

- **warp0** 调 `tmem_allocator.allocate(kNumTmemColsTotal, &tmem_base_ptr)`（`:1446`）；
- 全体经 `tmem_allocation_result_barrier.arrive_and_wait()`；`tCtAcc.data() = tmem_base_ptr`；
- 用 BF16 代理的 fragment 结构建 **TMEM→寄存器** 的 `TiledCopy`
  `make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc(_,_,_,0))`（`:1465-1468`，**32 dp × 32 b**
  的硬件 TMEM 载入）。寄存器侧先用**假的 BF16 张量**推导形状、再造**真 FP32** 张量接收
  （`:1473-1480`）。

### 6.2 每 work unit（`:1482-1551`）

1. 等 `acc_full`（`:1499-1500`）；
2. `cute::copy(tiled_copy_t2r, tTR_tAcc(…, acc_buf_idx), tTR_rAcc)`——把 FP32 累加器批量载进寄存器
   （`:1503-1505`）；
3. **`epilogue_wg_barrier.arrive_and_wait()`** 后 `elect_one_sync → arrive_barrier(acc_empty)`
   （`:1511-1515`）——4 warp 各 1 到达，与 `acc_empty` 的 init=4 匹配；
4. `converter(FP32→BF16)` 转寄存器（`:1517-1521`，`NumericConverter<bf16,float>`）；
5. **routing scatter 写**（`:1523-1544`）：每线程负责 `m_idx = m_tile*MMA_M + threadIdx.x`
   （一行），对 `i in [0,MMA_N)`：`n_idx = n_tile*MMA_N + i`，仅
   `n_idx<BATCH_SIZE && topk_idx>0 && m_idx<OUTPUT_SIZE` 时写
   `mOutput(n_idx, topk_idx-1, m_idx) = tCrC[i]`——**未路由的 token 输出位置不动**；
6. 再 `epilogue_wg_barrier.arrive_and_wait()` 后 `num_tiles_executed++`（`:1547-1548`）。

### 6.3 收尾（`:1555-1564`）

`__syncthreads()`；**warp0** `tmem_allocator.free(tmem_base_ptr, kNumTmemColsTotal)`——**注释明说
不调 `release_allocation_lock()`**（`:1561-1563`），因为 MPK 常驻 kernel 要把 TMEM allocator 锁在
warp0 跨 task 复用（引 `moe_linear_sm100.cuh:820` 的 "don't do relinquish for megakernel"）。

> 这是 **megakernel（C2 常驻）对 device task 的直接约束**：一个 task 的 epilogue 不能把 TMEM
> 分配器「交还」，否则常驻 worker 的下一个 task 会重新竞争锁。与 `docs/32` §5.3 的 `kNumSMs=1`
> 是同一类「为常驻适配」的 device 侧让步。

---

## 7. 三段流水与 barrier 时序（小结图）

```
warp6 SCALE ──sf_ready──┐
                         ▼
warp5 DMA ──a_full/b_full──▶ warp4 MMA ──acc_full──▶ warps0-3 EPILOGUE
   ▲                          │                         │
   └─────────ab_empty─────────┴─────────acc_empty────────┘
              （MMA 读完 A/B）        （EPILOGUE 读完累加器）
```

- A/B 数据环与尺度环**共享 stage 编号** `smem_rd/wr_buf = (num_prev_k_blk + k_tile) % NUM_AB_STAGE`
  （`:833`/`:1124`/`:1355`）——三个 warp 必须**同序遍历同一 `(expert,m_tile,n_tile,k_tile)`**
  （`:1105-1107` 明说「Both warps must iterate in the same order」）；
- 累加器环独立按 `num_tiles_executed % NUM_ACC_STAGE`（`:1117`/`:1491`）双缓冲；
- 所有等待都用 `try_wait_barrier` 先探再阻塞（乐观 peek + lookahead），这是本文件与 `docs/07`
  朴素 `wait_barrier` 的差别之一。

---

## 8. 与 DeepGEMM 的对应（源码坐标）

文件头与函数注释多处声明「派生自 DeepGEMM SM100 path」：

- `sm100_utils.cuh:26-28`：「adapted from the DeepGEMM SM100 Blackwell path」；
- `fp8_make_sf_smem_desc`（本文件 `:273-296`）注释「This matches DeepGEMM's `make_sf_desc`」——
  对应 **`~/github/DeepGEMM`（`057ca59`）`deep_gemm/include/deep_gemm/mma/sm100.cuh:42-48`**
  的 `make_sf_desc`（SWIZZLE_NONE、`SBO=8*16`、LBO=0）；
- `fp8_utccp_warp_transpose`（本文件 `:298-336`）「identical to DeepGEMM's
  `utccp_required_smem_warp_transpose`」——对应
  **`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:461-480`**（同样的 XOR 读/写模式）；
- **Mega MoE 的对应物**在 `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh`
  （`make_sf_desc :837`、`make_umma_desc :841-842`、`SM100_UTCCP_4x32dp128bit_2cta :895`）——
  那是 **DeepGEMM 的段级 mega-kernel**（见 `docs/02` 与本专题 `../mega-moe/`），
  与本文的 **MPK device task** 是「同惯用法、不同集成层」：DeepGEMM 把整段 MoE 编进一个手工
  mega-kernel，MPK 把它做成一个可被常驻 worker 调度的 task。

> **交叉引用而不重复**：本文件的算子级正确性/数值验证见 `../mega-moe/`（DeepGEMM MoE 前反向）；
> 本文只关心「这段 device 身体如何作为一个 task 嵌进 MPK 常驻 kernel」。

---

## 9. 生产布线：`register_moe_fp8_sm100_task`

### 9.1 登记与发射（`task_register.cc:3204-3452`）

- **输入契约**（`:3208-3215`）：6 输入 `input_fp8 / input_scale / weight_fp8 / weight_scale /
  routing_indices / expert_mask` + 1 输出；断言形状（`:3235-3265`）；
- **张量视图**逐条 `code.e(...)` 生成（`:3300-3419`）：TMA weight（`:3304-3322`，
  `GMEM_ROW=(E-1)*orig+out`）、`mInput`（`W13_LINEAR` 决定 2D/3D，`:3324-3348`）、`mInputScale`、
  `mWeightScale`（`:3373-3385`）、`mRoutingIndices`（`:3387-3395`）、`mMask`（`:3397-3403`）、
  `mOutput`（`:3405-3419`）；
- **模板实参**（`:3279-3291` + `:3422-3446`）：`MMA_M=128`、**`MMA_N=16`**、`bK=128`、
  **`num_ab_stages=8`**（注释 `:3283-3288`：TP=4 时 4 stage 会 hang，8 stage 与 BF16
  `moe_linear_sm100` 对齐、总 smem ≈148 KB 落在 205 KB 预算内）、`num_acc_stages=2`、
  `num_c_stages=4`；`EXPERT_STRIDE = grid_dim.x`（`:3298`）；
- **发射的 kernel 调用**（`:3422-3446`）把上述张量 + `task_desc->task_metadata.expert_offset`
  传给 `fp8_moe_group_gemm_sm100_task_impl`。

### 9.2 `TaskType` 与 grid/元数据映射

- `TASK_MOE_W13_FP8_SM100 = 248`、`TASK_MOE_W2_FP8_SM100 = 249`
  （`include/mirage/persistent_kernel/runtime_header.h:136-137`）；
- `graph.cc:713-720`：`name=="moe_w13_fp8_sm100"` → `register_moe_fp8_sm100_task(..., true)` 且
  `task_config=(6,1,TASK_MOE_W13_FP8_SM100,…)`；`moe_w2_fp8_sm100` → `false` /
  `TASK_MOE_W2_FP8_SM100`，同样 `(6,1)`；
- `runtime.cc:352-359`：对这两个 `TaskType`（连同 SM90/SM100 BF16 MoE）设
  **`task.task_metadata.expert_offset = bid.x`**；`expert_stride = grid_dim.x`
  ⇒ **x 维切专家、y 维切 N**；
- 名字表 `runtime.cc:1859-1860`。

> 与 `docs/32` §5.3 的 `linear_fp8_sm100` 对比：那个线性 task 用 `kNumSMs=1` HACK 把多 CTA grid
> **收成单 CTA**；本 group-GEMM task 则**保留 2D grid**（x=专家、y=N 切分），因为它的并行维度
> 天然就是 2D。两者都是「one task = one CTA」在 SM100 的不同适配。

### 9.3 单测 wrapper 的差异（`tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper.cu`）

单测与生产**同 `MMA_N=16`**（wrapper `:44` 注释「matches production task_register」），但
`DSV3_NUM_AB_STAGE=4`（wrapper `:50`）vs 生产 8；单测用 **2D grid 手推 `blockIdx.y` 指针**
（`docs/31` §5.3），生产靠 runtime 的 `input_map/output_map` 偏移（`runtime.cc:1026-1042`）。

### 9.4 一条「定义但未用」的证据（延续本专题的方法论）

本文件内定义的三个 helper **有定义、无调用者**（全文件 `grep` 仅命中定义行）：

- `fp8_make_sf_smem_desc`（`:283-296`）——MMA warp 实际用的是 `kernel::sm100::make_sf_desc`
  （`:1057`）；
- `fp8_advance_umma_desc_lo`（`:267-271`）——实际用 `kernel::sm100::advance_umma_desc_lo`
  （`:1246/1252`）；
- `fp8_tcgen05_after_thread_sync`（`:153-155`）——scale warp 用的是
  `fence_view_async_shared`（`:1409`），未用 tcgen05 fence。

即：**task 文件自带一份「本地实现」作为参照，真正跑的是 `sm100_utils.cuh` 的共享版本**
（与 `docs/16`/`docs/20` 记的「已实现未启用」是同型现象，但**更轻**：这里只是未接线，不是死码分支）。

---

## 10. 判据映射（C1–C5）

与 `docs/07`/`docs/29`–`docs/32` 同口径：**一个 device task 身体只贡献 C3（融合范围）里
「单 device task 可被表达 / 数值正确」的一环，本身不拿 C1/C2/C4/C5**。

- **贡献 C3**：把「6 输入 1 输出的 grouped FP8 MoE GEMM」编成一个可被 `register_task` 生成、
  可被常驻 worker 调度的 device 函数；
- **不贡献 C1/C2**（单次 launch / 常驻）：由运行层（`docs/06`/`docs/22` §8）拿到；
- **不贡献 C4/C5**（设备内协调 / SM 级调度）：task 内的 `mbarrier`/`NamedBarrier` 是 **CTA 内
  跨 warp 同步**，不是跨 task/跨 SM 的去中心化调度；
- **唯一与常驻有关的 device 侧事实**：`tmem_allocator` 不 `release_allocation_lock()`
  （§6.3）——为 C2 让步的适配，与 `docs/32` §5.3 的 `kNumSMs=1` 同族。

---

## 11. 局限（诚实标注）

- **纯源码走读，未构建 / 未运行**：SM100（`sm_100f`）在目标卡 H100/sm90 不可执行，未取任何 device
  计数；「有定义无调用者」是**静态 `grep`** 结论，不构成运行时判定。
- **未展开的相邻层**：`docs/32` §7 的另两条尾巴**本文未处理**——身体 B
  （`linear_fp8_1d2d_sm100.cuh`）的 727 行 CuTe 主循环、`mla_mtp_tpN_main` 主体；本文件只覆盖
  `fp8_group_gemm_sm100.cuh` 这一条。
- **生产对跑未做**：DeepGEMM SM100 mega MoE 路径（SM100 专属）本机不具备，未对照；本文只给
  `sm100_utils.cuh`/`fp8_group_gemm_sm100.cuh` 与 DeepGEMM `mma/sm100.cuh`、
  `sm100_fp8_fp4_gemm_1d1d.cuh` 的**静态对应**。
- **CUDA / CUTLASS 版本敏感性**：`tcgen05.mma.…kind::mxf8f6f4.block_scale`、`SM100_UTCCP_4x32dp128bit_1cta`、
  `SM100_TMEM_LOAD_32dp32b1x` 均为 SM100 专属；本机 CUDA 13.2/sm90 不编译此文件，未验证其可编译性。

---

## 12. 参考坐标（可核对）

- 快照：`mirage-project/mirage` 分支 `mpk`，commit `f9eb70c`（与 `docs/06`–`docs/32` 同快照）。
- **主文件**：`include/mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_sm100.cuh`
  （文件头语义 `:49-130`；三个本地 helper `fp8_ld_shared :134-140` / `fp8_st_shared :144-148` /
  `fp8_tcgen05_after_thread_sync :153-155`；`MoEFP8SharedStorage :169-256`；
  **未接线 helper** `fp8_advance_umma_desc_lo :267-271` / `fp8_make_sf_smem_desc :283-296` /
  `fp8_utccp_warp_transpose :322-336`；`fp8_moe_group_gemm_sm100_task_impl :430-1565`；
  常量 `bK :452`/`UMMA_K :458`；TMEM 布局 `:481-498`；BF16 代理 `:512-519`；
  barrier 初始化 `:632-651`；NamedBarrier `:661-666`；**warp5 DMA `:806-988`**；
  **warp4 MMA `:1017-1323`**（UTCCP `:1188-1198`、UMMA `:1223-1280`）；
  **warp6 SCALE `:1339-1419`**（FIX 注释 `:1357-1368`、UE8M0 `:1378`/`:1399`、转置 `:1407-1408`）；
  **epilogue `:1440-1552`**；TMEM free 不 relinquish `:1559-1564`）。
- **支撑原语**：`include/mirage/persistent_kernel/tasks/blackwell/sm100_utils.cuh`
  （`make_smem_desc :30-49`、`make_sf_desc :51-63`、`replace_smem_desc_addr :65-73`、
  `advance_umma_desc_lo :75-86`、`make_umma_desc :88-114`、
  `make_runtime_instr_desc_with_sf_id :116-123`、`SM100_MMA_MXF8F6F4_SS::fma :127-150`）。
- **生产**：`src/kernel/task_register.cc`（`register_moe_fp8_sm100_task :3204-3452`、
  模板实参 `:3279-3291`、发射 `:3422-3446`）；`src/kernel/graph.cc:713-720`；
  `src/kernel/runtime.cc:352-359`（`expert_offset=bid.x`）、`:1859-1860`；
  `include/mirage/persistent_kernel/runtime_header.h:136-137`（`TaskType` 248/249）、`:278`
  （`expert_offset` 字段）；`include/mirage/kernel/task_register.h:131`。
- **单测**：`tests/runtime_python/blackwell/sm100_fp8_moe/runtime_kernel_wrapper.cu`
  （DSV3 常量 `:43-53`、三处 task 调用 `:191/:266/:727`）；bf16 对照
  `runtime_kernel_wrapper_bf16.cu`。
- **DeepGEMM 对应**（`~/github/DeepGEMM`，commit `057ca59`）：
  `deep_gemm/include/deep_gemm/mma/sm100.cuh:42-48`（`make_sf_desc`/`replace_smem_desc_addr`）；
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh:461-480`
  （`utccp_required_smem_warp_transpose`）；mega MoE 对应物
  `deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:837/841-842/895`。
- 论文快照：`~/github/ai-infra-book/experiments/ch05/05-08/sources/mpk-v2.txt`——**未提及本层**
  （实现质量 / device task 内部，同 `docs/28`–`docs/32`）；「手工 kernel 可包成 task」见 `docs/05`。
- 同专题相关篇：`docs/32`（本文补其 §7 尾巴）、`docs/31`（本文补其 §10 尾巴）、
  `docs/07`（Hopper 单 task device 流水）、`docs/08`（依赖分析器）、`docs/21`
  （Blackwell transpiler 2SM codegen）、`docs/24`（`TaskType` 目录）、`docs/22` §8（C 判据总映射）。
- 交叉专题：DeepGEMM MoE 前向/反向正确性验证见 `../mega-moe/`（本专题不重复）。
