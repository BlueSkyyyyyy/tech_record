# MPK SM100 block-scaled FP8 的 SF 段并置走读：dense linear vs grouped MoE

> 快照：`mirage-project/mirage` 分支 `mpk` commit `f9eb70c`（本地 `/ssd/mirage_mpk_build`）。
> 对照 CUTLASS：`deps/cutlass` commit `f3fde583`。对照 DeepGEMM：`~/github/DeepGEMM` commit `057ca59`。
> 目标卡 H100/sm90 **不可执行 SM100**：本文为**纯源码走读**，另配一个 **host 侧算术探针**
> （`src/sf_segments_probe.cu`，`harness/sf_segments_probe.{sh,out.txt}`，device 3，未 launch kernel）。
>
> 本文补的是「下一步候选」第一条：`docs/33`（MoE group-GEMM）/`docs/34`（dense block-scaled 线性）/
> `docs/60`（`tmem_sf_frg`）/`docs/68`（支撑层描述符）各覆盖了 SF（scale factor）段的一部分，
> 但**从未把两条 device 身体的 SF 段并置逐行对走**。被对走的两个文件：
> `linear_fp8_sm100.cuh`（828 行，dense，单张量线性）与 `fp8_group_gemm_sm100.cuh`（1567 行，
> grouped MoE）。

## 0 定位：SF 段在 block-scaled FP8 里是哪一段

SM100（Blackwell）的 `tcgen05.mma...kind::mxf8f6f4.block_scale` 在算 FP8（E4M3）矩阵乘时，除了 A/B
两个 FP8 操作数，还要一份**每 32-K 元素一个的 UE8M0 尺度**（8-bit 纯指数）。硬件从 **TMEM** 里读这份
尺度（不是 SMEM）。所以每个 block-scaled task 都有一串固定动作，本文统称「**SF 段**」：

1. 把尺度（host 侧是 float32 的 per-128-K 量化 scale）**搬进 SMEM**；
2. 转换/打包成 **UE8M0**，并按「一行一个 uint32、每 uint32 装 4 个 K-子块」的组织方式落 SMEM；
3. 做一次 **warp 内 4×32 → 32×4 转置**（因为 `UTCCP` 的源必须是列块）；
4. 用 **UTCCP**（`SM100_UTCCP_4x32dp128bit_1cta`）把 SMEM 尺度搬到 **TMEM 的 scale 列**；
5. 发 UMMA 时用 **`sf_id`**（指令描述符里的两位）告诉硬件「这次子块用哪个 32-K 的尺度」。

两条身体走这五步，但**每一步的实现机制都不同**——因为它们被嵌进了两种不同的 warp 角色与流水里。

约定：下文的 A 操作数在 dense 里是输入激活（生产接 `input_fp8`，M=batch），B 是权重（`weight_fp8`，
N=output）；`SFA` = A 的尺度（M 维）、`SFB` = B 的尺度（N 维）。**注意**两个文件顶部的注释按
DeepGEMM 的老命名把「SFA=weight / SFB=input」写反了（见 §3.6），但代码与生产接线一致（`task_register`
把 `input_scale` 接到 SFA、`weight_scale` 接到 SFB）。

## 1 dense `linear_fp8_sm100.cuh` 的 SF 段

这是「一个 task = 一个 CTA、单 CTA 顺序扫所有 tile」的 dense 线性身体（`docs/32` §5.3 记录了它的
`kNumSMs=1` HACK）。warp 角色：**warp0 = TMA 载入（A/B/SF/residual/CD 全走 TMA）**、
**warp1 = UMMA**（`is_leader_cta`）、**warp2 = SF 专职转置**、`warp≥kNumNonEpilogueThreads/32` = epilogue。

### 1.1 编译期常量与容量（`:78-171`）

- `kNumUTCCPAlignedElems = 128`（`:78`）：一次 UTCCP 搬 128 个 uint32（= 4 列 × 32 行）。
- **粒度是模板参数**：`kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4`（`:83`），SFB 同理（`:84`），
  `static_assert(kGranKA==32 || 128)`（`:85-88`）。生产实例化 **`kGranKA = kGranKB = 128`**
  （`task_register.cc:4828`），故 `kNumSFAStagesPerLoad = 4`。
- **host 已打包**：`shape_sfa_k = ceil_div(shape_k, kGranKA * 4)`（`:93`）——对 gran 128 即
  `shape_k / 512`。也就是 SFA 张量是 `[M, K/512]` 的 **uint32**，**一个 uint32 里 4 个 UE8M0 字节 =
  4 个相邻 128-K 块的 scale**。这与 `docs/31` §2/§5「`sm100_linear_fp8` 的 scale 是 UE8M0 packed
  uint32」一致：**host 侧就打包好、device 不转换**。
- SMEM 每 stage 尺度：`SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t)`（`:144`），其中
  `SF_BLOCK_M = align(BLOCK_M, 128)`（`:140`），SFB 同理（`:145`）。
- TMEM 列：`kNumSFATmemCols = SF_BLOCK_M/32`（`:159`）、`kNumSFBTmemCols = SF_BLOCK_N/32`（`:160`）；
  累加器 `kNumAccumTmemCols = kNumEpilogueStages * kNumMWaves * BLOCK_N`（`:165`）；
  总列数 `get_num_aligned_tmem_cols<accum+SFA+SFB>()`（`:167-169`）；三段起点
  `kTmemStartColOfSFA = kNumAccumTmemCols`、`kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols`
  （`:170-171`）。生产 `BLOCK_M=32/BLOCK_N=16`（`:4833`）→ `SF_BLOCK_M=SF_BLOCK_N=128` →
  SFA=SFB=**4 列**（与 MoE 硬编码的 4 恰好相同，见 §2.1）。

### 1.2 SMEM 布局与 barrier（`:206-229`）

尺度 SMEM 排在 A/B 之后：`sf_start_ptr = smem_buffer + SMEM_CD_SIZE + SMEM_RESIDUAL_SIZE +
kNumStages*(SMEM_A+SMEM_B)`（`:206-208`），SFA 与 SFB 各 `kNumStages` 份（`:209-217`）。
关键是一个**三态 barrier 结构**（`:223-229`）：

- `full_barriers[stage]`：TMA 完成；
- `empty_barriers[stage]`：UMMA/UTCCP 消费完（`umma_arrive`）；
- **`with_sf_full_barriers[stage]`**（`:227-229`）：**SF 转置完成**——warp2 到达、warp1 等待。
  这是 dense 独有的「TMA 完成 ≠ SF 可用」两跳握手。

### 1.3 TMA 载入 SF（warp0，`:372-398`）

在 warp0 的 K-block 循环里，A/B 用同一个 `full_barriers[stage_idx]` 做 `arrive_and_expect_tx`
（`:340-364`），SF 则**按 `kNumSFAStagesPerLoad` 门控**、用 TMA 从 `tensor_map_sfa/sfb` 载入：

- SFA：`if (k_block_idx % kNumSFAStagesPerLoad == 0)` → `tma_copy<BLOCK_M,1,0>(...)`（`:372-384`），
  `num_arrival_bytes += kTmaBoxM_SFA * sizeof(uint32_t)`（`:384`，一个 BLOCK_M 行 × 1 个 uint32 列
  = 512 B @ BLOCK_M=128）；
- SFB：同样门控（`:386-397`），额外把 `m_block_idx` 传进 `get_global_idx<true,IndexType::SF_K>`
  （`:392-396`）。

**SF 的字节被并进 A/B 那个同一个 `full_barriers[stage_idx]` 事务计数**（`:400` 统一
`arrive_and_expect_tx(num_arrival_bytes)`）——即 dense 里 SF 与 A/B **共享 stage barrier**，没有独立
的 sf_ready。

### 1.4 独立的转置 warp（warp2，`:535-575`）

warp2 与 warp0/warp1 平行跑另一个 `get_next_block` 循环：

- 等 `full_barriers[stage_idx]`（`:554`，TMA 完成）；
- 对 SFA/SFB 按 `kNumS*StagesPerLoad` 门控做 `utccp_required_smem_warp_transpose`
  （`:556-571`）；转置实现是
  `values[i] = ld_shared(smem_ptr + (i ^ (lane_idx>>3))*32 + lane_idx)` +
  `st_shared(smem_ptr + lane_idx*4 + (i ^ (lane_idx>>3)), values[i])`（`:536-547`）；
- `fence_view_async_shared()`（`:562`/`:570`）后 **`with_sf_full_barriers[stage_idx]->arrive(0u)`
  （`:573`）** 通知 warp1。

即 dense 的转置**专门占一个 warp、且有自己的 barrier**，与 TMA 载入解耦。

### 1.5 UTCCP + UMMA（warp1，`:408-527`）

warp1 建三样东西：UMMA 指令描述符 `make_instr_desc_block_scaled<a,b,f32,ue8m0,UMMA_M,UMMA_N,K,K>`
（`:408-416`）、**`make_sf_desc(nullptr)`**（`:417`）、A/B 的 UMMA SMEM 描述符（`:420-429`）。
K-block 循环里：

- 等 **`with_sf_full_barriers[stage_idx]`**（`:460-461`）——转置已好；
- 发 UTCCP（`:463-484`）：`cute::SM100_UTCCP_4x32dp128bit_1cta`，按 `SF_BLOCK_M/128` 循环，
  `replace_smem_desc_addr(sf_desc, smem_sfa[stage_idx] + i*128)` 后
  `cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i*4)`；SFB 同理（`:474-482`）；
- 发 UMMA（`:492-525`）：对 `k=0..BLOCK_K/UMMA_K-1`（=4）算 `sfa_id/sfb_id`，
  **`sfa_id = (kGranKA==32 ? k : sfa_stage_in_group_idx)`**（`:495-498`），再
  `make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id)`（`:499-500`），
  `mma_t::fma(a_desc, b_desc, col, accumulate, runtime_instr_desc,
  kTmemStartColOfSFA + w*(128/32), kTmemStartColOfSFB)`（`:516-522`，SFA 还按 m-wave `+w*4`）。

要点：**gran=128 时 `sfa_id = k_block_idx % 4`**，正好**选到那个 uint32 里第几个 UE8M0 字节**——
即 dense 的 4 个字节是不同的 4 个 128-K scale，`sf_id` 是**真的按 K 子块选**（见 §3.2）。

## 2 grouped MoE `fp8_group_gemm_sm100.cuh` 的 SF 段

这是「一个 CTA 跑 3 条 warp 特化流水」的 grouped MoE group-GEMM（`docs/33` 记录了数据通路与竞态修复）。
warp 角色：**warp5 = DMA**、**warp6 = SCALE**、**warp4 = MMA（UTCCP + UMMA）**、**warps0-3 = EPILOGUE**。

### 2.1 常量与尺度格式（`:74-85` / `:194-207` / `:443-498`）

- 文件头把尺度契约写死（`:74-85`）：weight scale `[num_experts, N, K/128]`、input scale
  `[batch, K/128]`（W13）或 `[batch, num_topk, K/128]`（W2），**fp32**。
- SMEM 尺度缓冲：`sfa_smem[Num_AB_Stage][128]` / `sfb_smem[Num_AB_Stage][128]`（`:205-207`，
  固定 128 个 uint32/槽、`alignas(128)`）。注释**明说**（`:197-198`）：
  *"granularity is per-128-K (one scale per row per K-tile), all 4 bytes in each uint32 are identical
  replicas of the same UE8M0 value"*。
- `bK = 128`、`UMMA_K = 32`（`:452-458`）。
- TMEM 列**硬编码**：`num_tmem_acc_cols = MMA_N * NUM_ACC_STAGE`（`:481`）、
  `kNumSFATmemCols = kNumSFBTmemCols = 4`（`:482-485`）、
  `kTmemStartColOfSFA/SFB`（`:486-489`）、总列数**显式 round 到 2 的幂**
  `kNumTmemColsTotal`（`:495-498`）。

### 2.2 尺度 warp（warp6，`:1339-1419`）——载入 + 转换 + 打包 + 转置一条龙

warp6 与 DMA/MMA 平行跑「expert × m_tile × n_tile × k_tile」同序循环：

- **先等 `ab_empty_mbar_ptr[smem_buf]`**（`:1357-1368`，注释 `FIX 2026-04-22`）：必须等 MMA 侧
  `tcgen05.commit`（含 UTCCP）完成才可覆写 `sfa/sfb_smem`——这正是「写端握手」的 5 步里的保护，
  与 dense 靠 `empty_barriers` 的结构不同；
- **SFA**（`:1370-1380`）：`mWeightScale(global_row, k_tile)` 读 fp32，`row<OUTPUT_SIZE` 否则 1.0；
  `ue8m0 = (__float_as_uint(sf_val)>>23)&0xFF`；**打包成 4 个相同字节**
  `ue8m0 | (ue8m0<<8) | (ue8m0<<16) | (ue8m0<<24)`（`:1379`）；
- **SFB**（`:1382-1403`）：默认 `0x7F`（= UE8M0(1.0)），**route-aware**——`W13_LINEAR` 时
  `mInputScale(token_idx_n, k_tile)`，否则 `mInputScale(token_idx_n, topk_idx-1, k_tile)`（`:1394-1398`），
  且对 `token_idx_n < BATCH_SIZE && topk_idx > 0` 设防（`:1391-1392`），同样打包 4 份（`:1402`）；
- **转置**（`:1406-1408`）：`fp8_utccp_warp_transpose(sfa_buf)` / `(sfb_buf)`，实现与 dense 的
  `(i^(lane>>3))` XOR 版**逐字相同**（`:322-336`）；
- `fence_view_async_shared()`（`:1409`）+ **`sf_ready_mbar_ptr` arrive**（`:1411-1414`）。

即 MoE 的**一个 warp 干完了 dense 的 TMA（warp0）+ 转置（warp2）两件活**，用独立 `sf_ready` 通知。

### 2.3 MMA warp（warp4，`:1017-1323`）——UTCCP + UMMA

- 建 `make_instr_desc_block_scaled<e4m3,e4m3,f32,ue8m0,MMA_M,MMA_N,K,K>`（`:1042-1050`）；
- **`sf_desc = kernel::sm100::make_sf_desc(nullptr)`**（`:1057`，走共享原语 `sm100_utils.cuh`）；
- K-tile 循环里先 peek/await `sf_ready`（`:1135-1137`/`:1171-1175`），再 `elect_one_sync` 发 UTCCP
  （`UTCCP_t = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta`，`:1032`）：
  `replace_smem_desc_addr(sf_desc, sfa_smem[smem_rd_buf])` → `UTCCP_t::copy(uint64(sf_desc), sfa_tmem)`
  （`:1188-1198`，SFB 同理）；
- 发 UMMA（`:1223-1279`）：`sfa_id = sfb_id = k_sub`（`:1232-1233`），注释**明说**「4 字节相同，
  `sf_id` 选择不影响结果、只有在未来更细粒度时才有意义」（`:1226-1231`）；
  `SM100_MMA_MXF8F6F4_SS::fma(..., sfa_tmem, sfb_tmem)`（`:1271-1278`）。

### 2.4 TMEM 分配（epilogue warp0，`:1445-1451`）

`tmem_allocator.allocate(kNumTmemColsTotal, &shared_storage.tmem_base_ptr)`（`:1446`）由 **warp0**
发（与 `docs/34` §3.5「共驻 SM100 task 的 TMEM 生命周期须同 warp」一致），`tmem_base` 经
`tmem_allocation_result_barrier` 广播给 warp4（`:1020-1021`）。

## 3 并置对照

### 3.1 逐维对照表

| 维度 | dense `linear_fp8_sm100.cuh` | grouped MoE `fp8_group_gemm_sm100.cuh` |
|---|---|---|
| 尺度输入 dtype | host 已打包的 **uint32（4×UE8M0）**，`[M,K/512]`（`:93`） | device 读 **fp32**，`[·,K/128]`（`:74-85`） |
| UE8M0 转换点 | **host**（device 不转，`docs/31` §2） | **device** warp6：`(>>23)&0xFF`（`:1378`/`:1399`） |
| 打包 | host 把 4 个**不同** K-子块 scale 装进 1 uint32 | device 把**同一个** per-128 scale 复制 4 份（`:1379`/`:1402`） |
| SMEM 载入载体 | **TMA**（`tensor_map_sfa/sfb`，`:372-398`） | warp6 **标量 `mWeightScale`/`mInputScale`**（`:1370-1403`） |
| 载入与 A/B 的 barrier | **共享** `full_barriers[stage]`，字节并进同一 `arrive_and_expect_tx`（`:400`） | **独立** `sf_ready_mbar_ptr[stage]`（`:244`/`:1413`） |
| 转置角色 | **专职 warp2**（`:535-575`） | **同一 warp6**（`:1406-1408`） |
| 转置完成信号 | `with_sf_full_barriers[stage]`（`:229`/`:573`），TMA 与转置**两跳** | `sf_ready_mbar_ptr`（`:1413`） |
| UTCCP 发起者 | warp1（UMMA warp，`:463-484`） | warp4（MMA warp，`:1188-1198`） |
| SF descriptor helper | 本地 `make_sf_desc`（`:417`） | `kernel::sm100::make_sf_desc`（`:1057`）；另有一枚**未接线**的本地 `fp8_make_sf_smem_desc`（`:283-296`，`docs/33` §9.4） |
| TMEM SF 列数 | **动态** `SF_BLOCK_M/32`、`SF_BLOCK_N/32`（`:159-160`），可 >4 | **硬编码 4**（`:482-485`） |
| TMEM 总列 round | `get_num_aligned_tmem_cols<>`（`:167-169`） | 显式 2 幂 round（`:495-498`） |
| `sf_id` | `(kGranKA==32?k:k_block_idx%4)`（`:495-498`） | `k_sub`（`:1232-1233`），4 字节相同 ⇒ 空操作 |
| 粒度 | **参数化 32 或 128**（`:83-88`） | **硬编码 per-128**（`bK=128`：`:452`） |
| SFB 语义 | N 维权重/激活尺度（生产接 `weight_scale`） | **route-aware** per-token scale（`:1391-1398`） |
| SF 覆写保护 | `empty_barriers`（TMA 重填）+ 两跳结构 | 显式 `ab_empty` 等待（`FIX 2026-04-22`，`:1357-1368`） |
| warp 数 / 角色 | warp0 TMA / warp1 UMMA / warp2 转置 / epilogue | warp5 DMA / warp6 SCALE / warp4 MMA / warps0-3 EPILOGUE |

### 3.2 headline 一：同一个 `sf_id` 机制服务两种粒度，但语义在两条身体里相反

SM100 block-scaled UMMA 的 `sf_id` 两位（`docs/60` §4/§5、`docs/68` §B）用来在一个「4 字节打包」
里选一个 32-K 子块对应的尺度字节。两条身体都用它，但：

- **dense（gran=128）**：`shape_sfa_k = K/512` ⇒ 一个 uint32 装**4 个相邻 128-K 块的 UE8M0**，
  `sfa_id = k_block_idx % 4` 把 4 个 `k` 映射到 4 个字节 ⇒ **4 个字节必须互不相同**，
  `sf_id` 是**真正的地址选择**；
- **MoE（per-128 硬编码）**：一个 uint32 的 4 字节是**同一个 per-128 scale 的复制**（`:197-198`
  源码注释自陈）⇒ `sfa_id = k_sub` 选哪一字节都一样 ⇒ `sf_id` 是**空操作**（注释 `:1226-1231` 明说）。

即：**「host 打包 4 个不同字节 + device 按 `sf_id` 选」与「device 复制 4 份相同字节 + `sf_id` 无意义」
是同一硬件接口的两种极端用法**。dense 借此在 gran 32 时能把 4 个 per-32 scale 塞进同一套 UTCCP 路径
（`kNumSFAStagesPerLoad` 随之变 1）；MoE 借 `bK=128` 与 per-128 量化把这件事绕开。

### 3.3 headline 二：SF 的「谁搬 / 谁转 / 谁等」被按 warp 预算切成了两种拓扑

两条身体都遵守「UTCCP 的源必须是列块、必须先转置」这条硬件契约（`docs/60` §B），但围绕它排布了
**两种不同的流水拓扑**：

- **dense = 三级流水、职责最细**：TMA（warp0）异步搬 SF → 专职转置（warp2，独立
  `with_sf_full`）→ UMMA（warp1）发 UTCCP。好处是 SF 的载入（TMA）与转置可各自与别的 stage 重叠；
  代价是**多一个 warp + 一条 barrier**（`with_sf_full_barriers`），且 SF 字节要并进 A/B 的
  TMA 事务计数（`:400`）。
- **MoE = 单 warp 全包**：warp6 一次循环里把「标量读全局 → UE8M0 → 打包 → 转置 → `fence` → 通知」
  做完，MMA warp（warp4）只等 `sf_ready` 后发 UTCCP。省掉一个 warp 与 TMA 描述符，代价是
  **SF 的全局读是标量**（不能像 TMA 那样成块）、且要**显式等 `ab_empty`** 才敢覆写
  （`FIX 2026-04-22`）。

这与本专题反复出现的母题一致：**「同一套硬件惯用法在仓库里实现了不止一遍」**（`docs/20` §5、
`docs/57`、`docs/61`）。SF 段是这个母题的又一实例——甚至**转置函数本身**也有三份等价写法
（dense/MoE 带 XOR、`1d2d`/DeepGEMM 无 XOR，`docs/59` §C）。

### 3.4 headline 三：host 侧的「打包粒度」决定了 device 侧的「转换点」

`docs/31` §2/§5 已记录「`sm100_fp8_moe` 传 fp32 尺度、`sm100_linear_fp8` 传 UE8M0 packed uint32」。
本文补上**为什么这必然导致两条身体无法共享 SF 段代码**：dense 的 SF 张量已经按 512-K 打包，device
只需 TMA 搬运；MoE 的 SF 张量是 per-128 fp32，device 必须现场转 UE8M0。**转换点不同 ⇒ 载入载体不同
（TMA vs 标量）⇒ barrier 结构不同 ⇒ warp 角色不同**。这是一条从 host 打包约定一路传导到 device warp
划分的因果链。对 megakernel 的含义：**「让 host 多打包一点」能显著减轻常驻 kernel 的 device 侧工作量**
（MoE 的 warp6 是纯为这个补的）——与 `docs/04` §47/§48「动态必须落在调度、不能落在生成」同向。

### 3.5 与 `docs/60`/`docs/68`/`docs/59` 的接合

- `docs/60` 在 host 上求出了 CUTLASS `tmem_sf_frg` 的 `(dp, col)` 映射，并**只记不判**地指出
  「CUTLASS M 轴组间列 stride=4 vs MPK UTCCP 4 列块组间 stride=1」。本文的两条身体都属于 MPK
  手工路径：**都是「4 列一组、组间 stride 1」**（dense `kTmemStartColOfSFA + i*4`、MoE `sfa_tmem`），
  故那个 open question 是「MPK 手工 vs CUTLASS 高层」的分野，**不是 dense vs MoE 的分野**。
- `docs/68` §B 已逐位钉死 `make_sf_desc = SWIZZLE_NONE + stride 8*16` 与
  `make_runtime_instr_desc_with_sf_id = <<32`。本文把它接到**两个真实调用点**：dense `:417/:499`、
  MoE `:1057/:1237`。
- `docs/59` §C 已实测三份转置逐位等价。本文确认 dense `:536-547` 与 MoE `:322-336` 是**带 XOR** 的
  那两份，且**都消费各自 SMEM 里的 128-uint32 槽**（dense 由 TMA 写、MoE 由 warp6 写）。

### 3.6 一处命名落差（只记不判）

两个文件顶部注释的 SF 命名**互相矛盾**：dense `linear_fp8_sm100.cuh:52-53` 与 MoE
`fp8_group_gemm_sm100.cuh:204/206` 都写「SFA = weight scale / SFB = input(activation) scale」；
但生产接线（`task_register.cc:4852-4859`）把 `input_scale`（第 2 个输入）接到 **SFA**、
`weight_scale`（第 4 个输入）接到 **SFB**，而 A 操作数是 `input_fp8`（M=batch）——即 **SFA 是 A 的
尺度（input/M 维）、SFB 是 B 的尺度（weight/N 维）**。代码本身自洽（`get_global_idx<...,SF_K>`
按 M/N 索引），只是顶部注释沿用了 DeepGEMM 的旧命名。**只作静态落差记录**。

## 4 host 侧算术探针

`src/sf_segments_probe.cu`（host 侧、无 SM100 指令）逐字重放 SF 段的**纯算术**并打印：

1. **UE8M0 转换**：`(__float_as_uint(x)>>23)&0xFF` 对一组值逐值验证——`1.0→0x7F`、`2.0→0x80`、
   `0.5→0x7E`、`4.0→0x81`、`448→0x87`，并确认 `0x7F = UE8M0(1.0)`（正是 MoE SFB 的 padding 默认值）；
2. **dense 打包**：给 4 个**不同**的 per-128-K scale `{0.5,2,0.25,4}`，拼成 uint32 得
   `0x817D807E`，再用 `(packed >> (8*sf_id)) & 0xFF` 逐 `k_block` 解出 `0x7E/0x80/0x7D/0x81`
   全部命中 ⇒ **证明 dense 的 `sf_id` 是真地址选择**（4 字节互不相同）；
3. **MoE 打包**：给 1 个 per-128 scale `2.0`（UE8M0 `0x80`），复制 4 份得 `0x80808080`
   ⇒ 4 字节相同 ⇒ `sf_id` 空操作（4/4 相同 PASS）；
4. **行数账**：K=1024/4096/7168 时 dense `shape_sfa_k = K/512` = 2/8/14（uint32/行）vs
   MoE per-row scale 数 `K/128` = 8/32/56（fp32/行）——**恒 4× 关系**；
5. **转置双射**：`(i^(lane>>3))*32+lane ↔ lane*4+(i^(lane>>3))` 对 4×32 网格读/写集合都 128/128
   且**相等**（双射）；并给出「写入槽 `g` ← 哪些 `(r,c)`」的几何（如 `g=0←(0,0)`、`g=64←(0,16)`、
   `g=127←(3,31)`）；无 XOR 版（DeepGEMM `1d1d`）的写集合与 XOR 版**相同**⇒ **XOR 只是功能中性的
   迭代序置换**（与 `docs/59` §C 一致）。

原始输出见 `harness/sf_segments_probe.out.txt`。探针只覆盖 SMEM 侧的打包/转置算术；
**UTCCP/TMEM/UMMA 是 SM100 专属指令，本机 H100/sm90 不可执行**，其语义只从
`docs/59`/`docs/60`/`docs/68` 的源码与 CUTLASS 契约引用。

## 5 判据与结论

- **判据**：SF 段是 device body 内部的生产者/消费者数据流，**只贡献 C3**（段级多 task 融合的范围），
  与 `docs/33`/`docs/34`/`docs/59`/`docs/60`/`docs/68` 同口径。C1/C2/C4/C5 由运行层拿。
- **一句话结论**：dense 与 MoE 的 SF 段是**同一硬件契约（UTCCP + `sf_id` + TMEM scale 列）的两种极端
  工程实现**——dense 把「转换/打包」推给 host、用 TMA + 专职转置 warp 换吞吐；MoE 把它留在 device、
  用一个 SCALE warp 全包换 warp 预算。**host 侧的打包粒度是这两条路的分水岭**。

## 6 局限

- 未构建、未在 SM100 上运行两条身体（目标卡 H100/sm90 不可执行）；UE8M0 打包/转置的数值等价性
  由 host 探针与 `docs/59` 支撑，**UTCCP→TMEM 的精确 dp↔深度映射仍未验证**（`docs/59` §8 同一
  open question）。
- `SF_BLOCK_M/N` 只在生产参数（`BLOCK_M=32/BLOCK_N=16 → 128/128`）下核；更大 `BLOCK_M`（>128）
  时 dense 的 `kNumSFATmemCols` 会 >4、UTCCP 会循环多次，该路径未展开。
- `kNumEpilogueStages`/TMEM 列 round 的具体数值只按公式推导、未实例化打印（与 `docs/68` 的 host
  探针不同，本文未把 `get_num_aligned_tmem_cols` 重放）。
- MoE warp6 的 `sf_ready` 与 `ab_empty` 两门时序只按源码注释（`FIX 2026-04-22`）转述，未在
  SM100 上复现竞态。
- DeepGEMM 对照只到 `1d1d`/`mega_moe` 的 `make_sf_desc`/转置/`SF_BLOCK_M` 结构，未逐行对走。
