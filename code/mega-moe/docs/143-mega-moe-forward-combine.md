# Mega MoE 前向 single-step 预算（下）：L2 写回与 combine 归约，补全第 142 篇缺的「非 GEMM 段」

> 本文对应路线图 **P133-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 143 篇。
>
> 第 142 篇（P132-1）用 ncu 把**前向**的两段 routed GEMM 归因清楚了，并给出一个
> **前向 single-step 预算**：`L1 m_grouped 204.1us + L1 epilogue 24.2us + L2 m_grouped 111.1us = 339.4us`。
> 但那个预算**只算到 L2 的 GEMM 输出**——融合 kernel 里 L2 输出还要再走两段才变成 `y`：
> **L2 bf16 epilogue 写回**（把 L2 结果经 NVLink 直写回源 token 所在 rank 的 combine 缓冲）与
> **combine 归约**（把 top-k 份贡献拉回、bf16→fp32 累加、写 `y`）。本篇把这两段补上：逐段精读
> 源码、在 H100 上复刻其**数据布局与数值语义**、量出**字节与用时**，把前向 step 预算补成完整的一张。
>
> 核心结论一句话：**前向的「通信 + 归约」腿和第 142 篇算的「GEMM + L1 epilogue」腿是同一量级**
> （本机量得串行 ~292us vs 计算 ~339us），所以融合 kernel 必须把它们**重叠**起来——这正是
> Mega MoE 这台机器存在的理由；第 142 篇的 339.4us 只是这台机器的**计算侧**，不是整 step。
>
> 可跑性：融合 kernel 的 L2 写回（TMEM/SMEM/NVLink 直写）与 combine（TMA + `ptx::accumulate`）
> 是 **SM100 专属**（`csrc/apis/mega_moe.hpp:256/367` 只在 `arch_major==10` 派发）→ **仅分析**；
> 但两段的**数据布局**（邮箱 `[topk][token][hidden]`、`TokenSrcMetadata`）与**数值原语**
> （`cast_into_bf16_and_pack` 的 RNE、`ptx::accumulate` 的 bf16→fp32 累加）**架构无关**，
> H100(sm90) 可**直接 include 真实头文件跑真实设备代码**。远端写回腿的带宽用 2 卡真机量。
>
> 原始输出：`harness/exp_moe_forward_combine.out.txt`（`harness/exp_moe_forward_combine.{sh,py}`、
> `src/moe_forward_combine.cu`；H100 80GB HBM3, sm90, 驱动 570）。真实头文件 sha256：
> `common/math.cuh 576e9db1…`、`ptx/utils.cuh 8a9d5e03…`。

## 1. 为什么补这两段：第 142 篇的预算缺了什么

第 142 篇的前向 step 预算只覆盖到 `linear2` 的 GEMM 输出为止。但在融合 kernel
`sm100_fp8_fp4_mega_moe.cuh` 里，L2 的结果并不是终点，它还要：

1. 被 **L2 epilogue** 从 TMEM 读回、`round` 成 bf16，经 NVLink **直写回源 token 所在 rank** 的
   `combine_token_buffer`（`impls:1254-1357`）——这是「combine 的**写端**」；
2. 被 **combine 段**把 top-k 份贡献从**本地**邮箱拉回、在寄存器里 **bf16→fp32 累加**一次、
   回 bf16 写 `y`（`impls:1365-1520`）——这是「combine 的**读端/归约端**」。

第 142 篇明确把「L2 的 NVLink 直写 combine」列为 **SM100 专属、仅分析**、没有进预算。本篇补上。

顺带澄清一个容易被各篇含糊的术语：Mega MoE 的「combine」与 DeepEP 的 combine（第 58/85 篇）
**语义相同、方向相反**——DeepEP combine 是**收方主动 pull**（源 rank 拉回专家结果再本地求和）；
Mega MoE 的 combine 是**专家 rank 主动 push**（`sym_buffer.map` 直写源 rank、源 rank 只做本地求和）。
两条腿的**字节量都是 `T·topk·H·2B`**（第 1 篇的通信量模型里的 combine 项），只是一个记在「写」
一个记在「读」。本篇量的是 push-then-reduce。

## 2. L2 epilogue 写回（`impls:1254-1357`）

这是 L2 分支的 epilogue（与 L1 分支 `:1066-1234` 并列）。逐段：

- **TMEM 读回**（`:1011-1022`，被 L1/L2 共用）：`load_epi_block` 用
  `cute::SM100_TMEM_LOAD_16dp256b1x` 把一个 store block 的累加器从 TMEM 读进寄存器
  `raw_values[kNumBuffers][kNumAtomsPerStore][ATOM_M]`，是**寄存器 ring**（`kNumPrefetchStages` 级预取）。
  这是 SM100 的 TMEM 语义，sm90 无——**仅分析**。
- **RNE 打包 + STSM**（`:1293-1317`）：每个 atom 把 8 个 fp32 值两两 `round` 成 bf16
  ——`math::cast_into_bf16_and_pack`（`common/math.cuh:80-84`，内部 `__float22bfloat162_rn`）——
  再用 `ptx::SM90_U32x4_STSM_T<uint32_t>::copy`（4 个打包 int → 一次 `stmatrix...x4.trans`）
  写进 smem，地址带 **128B swizzle**（`:1305-1309` 的 `(col ^ row)` 与 `kNumBankGroupBytes=16`）。
- **远端直写**（`:1322-1352`）：等 smem ready（`:1320` `sync_aligned`）后，每 warp 写 2 行，
  每 lane 从 smem 读 16B（`ld_shared<float4>`），写到
  `buffer.combine_token_buffer.get_rank_buffer(dst_topk_idx).get_data_buffer(dst_token_idx)`
  （`:1346-1347`）——注意 `get_rank_buffer` 的参数在**这里其实是 top-k 槽号**（见 §4），真正的
  **目的 rank** 来自 src metadata：`*sym_buffer.map(dst_ptr, dst_rank_idx) = packed`（`:1351`），
  `sym_buffer.map` 把本地地址换成**远端 rank 的同 offset 地址**（`layout/sym_buffer.cuh:33-40`）。
- **src metadata**（`:1282-1290`）：写回前先读回本行 token 的 `cached_src_metadata`
  （`layout::TokenSrcMetadata`，共享专家用 `(rank_idx, pool_m_idx, kNumTopk)` 就地构造，
  路由专家查 `workspace.get_token_src_metadata_ptr`）——这就是「结果该发给谁」的地址簿。

一句话：**L2 epilogue = TMEM→寄存器→RNE bf16→smem(swizzle)→NVLink 远端 store**，把 `T·topk·H·2B`
的 bf16 结果推到源 rank。

## 3. combine 归约（`impls:1365-1520`）

### 3.1 chunk 划分（`:1367-1386`）

复用 smem 头部（`kNumReusableSmemBytes`），每个 epilogue warp 私有「2 个 load stage + 1 个 store」
（`kNumChunkSlots=3`，`:1391-1400`）。`kNumChunks ∈ {1,2}` 由「smem 容得下且 hidden 不太大」
决定（`:1376-1377`），`kNumChunkBytes = kNumHiddenBytes/kNumChunks`。一串 `DG_STATIC_ASSERT`
把「TMA 16B 对齐 / 32 个 lane 各一个 uint4」钉死（`:1383-1385`）。

### 3.2 readiness 水位线（`:1405-1441`）

combine 归约必须等**所有可能给本 rank 写结果的 peer** 都写完。kernel 用 **grid index 当水位线**：

1. 写端完成后，本地**先 `comm::grid_sync`**（`:1412-1415`，「所有本地 L2 写都过了这个点」）；
2. SM0 的 `epilogue_thread_idx` 前 `kNumRanks` 条线程把「我从各 peer 收到的 grid index」
   用 `ptx::st_rel_sys` 写到 peer 的 `combine_ready_grid_idx[own_rank]`（`:1417-1419`）；
3. 归约前每个 token 等自己选中的专家所属 rank 的 `combine_ready_grid_idx[r] == own_grid_idx`
   （`:1436-1441` 的 `wait_until`，用 `ptx::ld_acq_sys`）。

「比较相等」而不是「大于等于」是因为 `grid_idx = ptx::get_grid_idx() + 1`（`:1425`）每轮 launch
自增：一次 launch 的写入只认**自己这一代**的标签（与第 102 篇 Mega Gate/mHC 的 `%gridid`
免复位、第 87 篇 `grid_sync` 的 2³¹ 免复位是同一族技巧）。

### 3.3 逐 chunk 归约（`:1426-1520`）

- 每个 `(token, chunk)` 由一个 warp 认领，grid-stride（`:1426`）；
- 读本 token 的 top-k 槽号（`:1431-1434`，共享专家槽 `=kNumTopk` 排在最前）；
- `move_mask_and_load`（`:1447-1466`）：按槽位从**本地** `combine_token_buffer` 用 `tma_load_1d` 拉
  一个 chunk，**两级流水**（`:1469-1475` 先发下一槽再累加当前）；
- 累加用 **fp32**：`ptx::accumulate(reduced[...], bf16_values[...])`（`:1485`）——sm100 上是
  融合 `add.rn.f32.bf16`，sm90 回退是 `__bfloat1622float2` + 加（`ptx/utils.cuh:65-74`）；
- Cast 回 bf16（`:1493-1507` 的 `__float22bfloat162_rn`）→ `tma_store_1d` 写 `y`（`:1511-1518`）。

**这就是为何 combine 是数值上的「最后一次」**：top-k 的贡献先各自 `round` 成 bf16（L2 epilogue），
再在 fp32 里求和、最后再 `round` 回 bf16（`y` 是 bf16）。

## 4. 邮箱布局：`combine_token_buffer[topk][token][hidden]`

`layout/mega_moe.cuh:447-449`：

```
combine_token_buffer = Buffer(bf16_token_layout,
                              num_topk + (num_shared_experts > 0 ? 1u : 0u),   // 第 0 维 = 槽号
                              num_max_tokens_per_rank);                        // 第 1 维 = token
```

- 第 0 维是 **top-k 槽号（+ 共享专家一个槽）**，**不是 rank**；`get_rank_buffer(i)`（`:326-333`）
  只是按「第 0 维索引」跨一个 per-slot 块（`get_num_bytes_per_rank = num_max_tokens·bytes`）。
- 远端 rank 通过 `sym_buffer.map` 写进来（第 10 维不携带 rank，rank 在地址里）。
- `TokenSrcMetadata{rank_idx, token_idx, topk_idx}`（`:43-52`）是**写端的地址簿**：专家 e 拿到
  某个 pool row 后，靠它知道「这个结果属于哪个 token 的第几个槽、该发去哪个 rank」。
- 第 44/41 篇的**共享专家**多用**一个槽**（`kNumTopk` 号），combine 时它**先被读**（`:1450`）、
  且**权重恒 1.0**（第 44 篇）。

## 5. H100 实测（`src/moe_forward_combine.cu` + `exp_moe_forward_combine.{sh,py}`）

融合 kernel 本体 **SM100 专属、仅分析**；本实验把两段的**数据布局 + 数值原语**搬到裸 CUDA，
**直接 include** 真实的 `common/math.cuh`/`ptx/utils.cuh` 跑真实设备代码。

### 5.1 语义核对（逐位）

- **A) 端到端 scatter→reduce**（`G=8, R=16, H=256, T=24, topk=2`）：模拟真实 token→(专家,槽)
  分配，写端用 `math::cast_into_bf16_and_pack`、读端用 `ptx::accumulate`：
  - 邮箱 `mailbox[slot][token]` 内容 vs RNE bf16 参考 **mismatch 0/12288**；
  - `y[t]=Σ_slot bf16`（bf16 栅格上逐位）**mismatch 0/6144**、`max|Δ|=0`。
- **Python 独立 oracle**（纯 torch，`exp_moe_forward_combine.py`）：同一语义逐位 True；
  「漏一个 slot」反例 **diff 8171/8192**（判据有齿）。

### 5.2 DeepSeek 量级字节与用时（`T=2048, topk=2, H=7168`）

| 段 | 线上字节 | H100(sm90) 实测 | 说明 |
|---|---|---|---|
| combine 归约（读 topk 邮箱 + 写 y） | 读 58.7 MB + 写 29.4 MB = **88.1 MB** | **106.5 us · 827 GB/s** | 移植 kernel；真机走 TMA + 融合 `add.rn.f32.bf16` |
| L2 打包（fp32→RNE bf16，HBM→HBM 参照） | 读 67.1 MB + 写 33.6 MB = 100.7 MB | 73.4 us · 1371 GB/s | 只量化「打包 + 搬运」上限；真机读是 TMEM（片上） |
| 远端邮箱搬运（rank0→rank1，`T·topk·H·2B`） | **58.7 MB** | **186.1 us · 315.6 GB/s** | 2 卡 NCCL（共享节点、非独占 → 受争用下界） |

对照带宽常数：本地 HBM **3.35 TB/s**（第 15/132 篇口径）、单卡对远端 streaming **~369 GB/s**
（第 11 篇 `exp_nvlink_access`，裸 peer store）、2 卡 NCCL send **315.6 GB/s**（本篇，受争用）。

读法：

- combine 归约测到 **827 GB/s**，远低于本地 HBM——因为 sm90 上 `ptx::accumulate` 是
  `bf16→f32` 转换 + 加两步、且 warp-per-token 占用率不高；真机用融合指令 + TMA 深流水会更快。
  **它是一个架构无关的「下限/参照」，不是 SM100 的真值。**
- **关键量级**：光是 **L2 写回的 58.7 MB 推到远端** 就要 ~186 us（316 GB/s），
  **和 L2 GEMM 的 111.1 us 同量级**；combine 本地归约再 ~107 us。**两段加起来 ~292 us，与第 142 篇的
  计算腿 339.4 us 同量级。**

## 6. 补全前向 single-step 预算

把第 142 篇（计算侧）与本节（通信+归约侧）拼起来（per-rank、单 rank 形状代理，
`T=2048, topk=2, H=7168, I=2048`）：

```
计算腿（第 142 篇）
  linear1 m_grouped         204.1 us   (1178 TFLOPS)
  L1 epilogue (SwiGLU+量化)  24.2 us   (3129.7 GB/s)
  linear2 m_grouped         111.1 us   (1082 TFLOPS)
  ------------------------------------------------
  小计                      339.4 us   (计算 roofline 下界 182.3 us → 达 54%)

通信 + 归约腿（本篇，串行口径）
  L2 bf16 写回（NVLink 远端） ~186 us   (58.7 MB @ 316 GB/s, 受争用下界)
  combine 归约（本地读+写 y）  106.5 us  (88.1 MB @ 827 GB/s, 架构无关参照)
  ------------------------------------------------
  小计                      ~292 us

两腿之和 ≈ 631 us；计算腿占 54%，通信+归约腿占 46%
```

结论：

1. **第 142 篇的 339.4 us 只是「计算侧」**，把所有非 GEMM 段补上后前向 step 的串行口径接近翻倍。
2. **两腿同量级（46% vs 54%）**——这正是 Mega MoE 把 dispatch/epilogue warp 与 MMA warp
   **解耦重叠**的原因（第 2/3/9 篇）：L2 写回与 combine 在真机里由 epilogue warp 与 MMA 并发推进，
   不落在这条串行时间线上；dispatch 的远端 pull（第 1/11 篇）同理。
3. 与第 11 篇的远端延迟数（远端裸延迟 2.83×本地）一致：**通信腿的瓶颈是远端带宽/延迟**，
   所以 kernel 要把它切 chunk、和计算重叠（第 2 篇的 ring/mbarrier）。

## 7. 与既有篇章的关系与边界

| 篇目 | 做了什么 | 本篇补的 |
|---|---|---|
| 第 142 篇（P132-1） | 前向两段 GEMM 的 **ncu 归因 + 计算侧预算** | 计算侧之外的 **L2 写回 + combine 归约** 段与预算 |
| 第 11 篇（P3-4） | dispatch/combine 段的**机制**（写回、readiness、归约） | 这两段的**字节/用时预算**与语义逐位核对 |
| 第 1 篇（P1-2） | 前向**通信量模型**（combine `T·k·H·2B`） | 把这个模型落到**具体 kernel 段**并量出用时 |
| 第 58/85 篇（DeepEP） | DeepEP combine（**pull** 方向）的归约 | Mega MoE combine（**push** 方向）+ 邮箱布局 |
| 第 44 篇（P34-1） | 共享专家的 L2/combine 槽 | 共享槽在 combine 里**先读、权重恒 1.0** 的位置 |

**能跑/能核**（`harness/exp_moe_forward_combine.{sh,py}` + `src/moe_forward_combine.cu`）：
邮箱布局 `[slot][token][hidden]` + `TokenSrcMetadata` 的 scatter→reduce 语义（逐位）、
combine 归约的移植定时、L2 打包带宽参考、2 卡远端邮箱搬运带宽。均 H100(sm90) 真机。

**仅分析**：融合 kernel 的 TMEM 读回（`SM100_TMEM_LOAD_16dp256b1x`）、SMEM swizzle STSM 的
`stmatrix...x4.trans` 就地路径、NVLink 远端 store 的地址翻译（`sym_buffer.map`）、
readiness 的 `st_rel_sys`/`ld_acq_sys` 与 `grid_sync`——**SM100 专属**。本机的数（尤其带宽）
受**共享 8 卡节点**影响，是**受争用的下界**，报数时已注明。
