# SM100 sparse MQA logits 主 kernel 逐段精读：gather 式 KV 访问、双 Q 槽打包与压缩列映射

> 本文对应路线图 **P126-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 136 篇。
>
> 第 22 篇（P12-1）在拆稀疏 MQA logits 时，对主 kernel `impls/sm100_sparse_mqa_logits.cuh` 只给了
> §4 一节约 40 行的**概览**（「五类 warp、block-scaled UMMA、contiguous TMA 快路、`cp.async` 尾裁剪、
> math warpgroup 的 ReLU×w 归约、压缩列映射」）；同一主 kernel 的 **dense 版**后来在第 113 篇
> （P103-1）做了一次完整的逐段精读 + H100 真机验证。本篇把稀疏版补齐到同样的深度：逐段读 688 行主
> kernel 与 202 行 JIT 包装层，并把其中**架构无关**的六段在 H100(sm90) 上跑真实设备代码。
>
> **可跑性**：主 kernel 的 UMMA（`sm100_sparse_mqa_logits.cuh:373` 的 `SM100_MMA_MXF8F6F4_SS` /
> `MXF4_SS`）、tcgen05、UTCCP、TMA 是 **SM100 专属，仅分析**；但下面六段是纯整数 / 纯 bf16 运算，
> 在 H100(sm90a) 上**直接抽取真实函数原样编译运行**（并 include 真实 `layout/sparse_mqa_logits.cuh`
> 以核 `sizeof`/`offsetof`）：
>
> | 段 | 真实行号 | 实测 |
> |---|---|---|
> | A `get_swizzled_kv_chunk_idx` | `:28-32` | smem 16B-chunk XOR swizzle：对合置换，`N∈{64,128}` |
> | B `KVSplitHeader`/`KVBlockInfo` | `layout:31-72` | 标记位 + 双 Q 槽 16+16 位打包/解码 32 例 0 不符 |
> | C KV 访问几何 | `:96-98`、`:121-131` | 连续尾块裁剪 + 分页页内映射（40 块 0 不符） |
> | D 压缩列映射 | `:588-590`、`:627-633` | token 分解双射 + slot 解码 + `(base+slot)*SB+t` 双射 |
> | E 资源账 | `:180-196`、`layout:103-138` | 线程数/TMEM 列/`SharedStorage` 4 组 `offsetof` 全中 |
> | F SF 4×32 转置 | `:325-333` | `src[i*32+lane] → dst[lane*4+i]` 128 元素 0 不符 |
>
> 源码坐标（`~/github/DeepGEMM`，本地 commit `057ca59`、Public release 26/09/30）：
> - 主 kernel：`deep_gemm/include/deep_gemm/impls/sm100_sparse_mqa_logits.cuh`（688 行）
> - 共享内存布局：`deep_gemm/include/deep_gemm/layout/sparse_mqa_logits.cuh`（156 行）
> - metadata 调度器：`deep_gemm/include/deep_gemm/scheduler/sm100_sparse_mqa_logits_metadata.cuh`
> - host 入口：`csrc/apis/attention.hpp:303-442`、`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp:11-202`
> - 实测：`src/sparse_mqa_logits_kernel.cu` +
>   `harness/exp_sparse_mqa_logits_kernel.{sh,out.txt}`（H100，**ALL PASS**）
>
> 全文用「目标卡」指代目标 AI 卡。

## 1. 主 kernel 全景

### 1.1 两个入口、同一核心

稀疏 MQA logits 有两个 `CUTLASS_GLOBAL` 入口，都只是薄壳，核心是
`sm100_sparse_mqa_logits_core_impl`（`:146-645`）：

- **连续版** `sm100_sparse_mqa_logits`（`:647-665`）：构造 `ContiguousSparseKVAccessor`，传入
  `tensor_map_kv`/`tensor_map_sf_kv`（KV 是连续张量，可走 TMA）。
- **分页版** `sm100_paged_sparse_mqa_logits`（`:667-686`）：构造 `PagedSparseKVAccessor`
  （`:101-142`，KV 是 fused page cache），`kUseUnalignedKs` 强制 `false`（`:684`；host 侧
  `hpp:152` 也断言分页与 unaligned 互斥）。

**唯一的分叉点是 `KVAccessor` 与其 `kSupportsContiguousTMA` / `kNeedsTailMask` 两个静态布尔**
（`:36-37`、`:103-104`）——这是本 kernel 用模板把两条 KV 访问路径折叠进一份核心逻辑的关键。

### 1.2 网格与几何

`grid = num_sms`、`block = get_num_threads(num_math_warpgroups)`（`hpp:114-116/124-126`）。几何常量
集中在 `:157-166`：

- `UMMA_M = 128`（KV 行）；`UMMA_N = BLOCK_Q * kNumHeads`（`:161`）；`UMMA_K = fp4?64:32`（`:165`）。
  **注意与 dense 版不同**：dense 版把 `UMMA_N` 钉死在 128，这里 `BLOCK_Q=2` 且
  `kNumHeads ≤ kNumMaxHeads=32`（`layout:13`），所以 `UMMA_N ≤ 64`——KV 行（M）远多于 (token,head)
  列（N）。
- `SPLIT_KV = kNumMathWarpGroups * UMMA_M`（`:166`）：一个 KV split 恰好由若干个 128 行的 UMMA tile
  组成，**每个 math warpgroup 吃一个 tile**。
- `kNumQStages=2`、`kNumTmemStages=5`（`hpp:44-45`）；`num_kv_stages ∈ {3(fp8), 5(fp4)}`（`hpp:49`）
  ——FP4 的 KV 每元素半字节，同样的 smem 预算能放更多级。
- `split_kv = fp4?640:512`（`hpp:15-18`）→ `num_math_warpgroups = split_kv/128 ∈ {5(fp4), 4(fp8)}`。

### 1.3 六类 warp

`kNumThreads = get_num_threads(nwg) = (nwg + 1 + nwg/4) * 128`（`layout:19-21`），所以线程数是
**math + 1（Q/metadata 生产者）+ nwg/4（KV copy）+ 2（SF 转置、UMMA）+ 1（填充）** 的 128 对齐版。
warp 角色（`:202-206`、`:261-262`）：

| warp | 行号 | 职责 |
|---|---|---|
| `0 .. nwg*4-1` | `:551-641` | **math warpgroup**：读 TMEM、ReLU×w、压缩写 logits |
| `kNumMathWarpGroups*4`（`kQAndMetadataWarpIdx`） | `:263-322` | **Q/metadata 生产者**：发 Q/SF-Q/weights TMA + `cp.async` 搬 metadata，驱动 Q/KV 流水 |
| `+1`（`kSFTransposeWarpIdx`） | `:323-369` | **SF 转置 warp**：4×32→32×4 转置后 UTCCP 进 TMEM |
| `+2`（`kUMMAWarpIdx`） | `:370-415` | **UMMA warp**：单 warp 发 block-scaled 矩阵乘 |
| `kNumThreads/32 - nwg ..` | `:416-550` | **KV copy warps**：把稀疏 KV 块 gather 进 smem |

以 fp8（nwg=4、768 线程、24 warp）为例：math 0–15、Q/metadata 16、SF 17、UMMA 18、KV copy 20–23，
warp 19 落在 `:642` 的 `else` 分支（`warpgroup_reg_dealloc`）；以 fp4（nwg=5、896 线程、28 warp）为例：
math 0–19、Q 20、SF 21、UMMA 22、KV copy 23–27。

**与 dense 版的关键差异**：稀疏版的 KV 是**不可预测的稀疏块 gather**（物理块号来自 metadata），
所以主体走 `cp.async` 逐 16B chunk 搬，而不是 Cutlass 的 TMA + swizzle。这直接引出 §2 的自定义
smem swizzle 和 §4 的尾块裁剪。

### 1.4 TMEM 列预算

`:180-196` 把 TMEM 切成三块（与 dense 版同构）：

```
kNumAccumTmemCols   = UMMA_N * kNumTmemStages       // 5 段累加器
kNumSFQColsPerStage = kNumSFQ / 32                  // kNumSFQ = align(BLOCK_Q*kNumMaxHeads,128) = 128 → 4
kNumSFKVColsPerStage= SPLIT_KV / 32
kTmemStartColOfSFQ  = kNumAccumTmemCols
kTmemStartColOfSFKV = kTmemStartColOfSFQ + kNumQStages * kNumSFQCols
kNumTmemCols = get_num_aligned_tmem_cols<kTmemStartColOfSFKV + kNumKVStages * kNumSFKVColsPerStage>()
```

`DG_STATIC_ASSERT(kNumTmemCols <= 512 and kNumThreads <= 1024)`（`:196`）。第 9 节实测四组配置的
raw 列数（heads=32）为 fp8 **376** / fp4 **428**，都对齐到 512。

### 1.5 barrier 清单与寄存器两档

`:214-242` 初始化（仅在 `kQAndMetadataWarpIdx` 的 elect-one lane）：

- Q 流水：`full_q.init(1)`、`full_sf_q.init(1)`、`empty_q.init(kNumMathThreads + 64)`
  （生产者 warp + UMMA warp 各 32 + 全部 math 线程）。
- KV 流水：`full_metadata.init(32)`、`full_sf_copy.init(kNumKVCopyThreads)`、
  `full_kv.init(kNumKVCopyThreads + 1)`（**SF 转置 warp 贡献最后 1 个到达**，`:232-233`）、
  `empty_kv.init(kNumMathThreads + 1)`。
- TMEM：`full_tmem.init(1)`、`empty_tmem.init(128)`。

寄存器两档（`:173-177`）：control/SF/UMMA 固定 64，math 用 `cute::min(120, 池反推)`，
`DG_STATIC_ASSERT`（`:197-199`）保证「math 总和 + control + KV copy × 组数」不超 CTA 入口池。
`:244` 由 SF 转置 warp 分配 TMEM，`:259` 先 `cudaGridDependencySynchronize()`（PDL，等 metadata kernel）。

## 2. A 段：gather 式 KV 的自定义 smem swizzle

```cpp
// impls/sm100_sparse_mqa_logits.cuh:28-32
template <uint32_t kSwizzleMode>
CUTLASS_DEVICE uint32_t get_swizzled_kv_chunk_idx(const uint32_t logical_chunk_idx) {
    constexpr uint32_t kMask = kSwizzleMode / 16 - 1;
    return (logical_chunk_idx & ~kMask) | ((logical_chunk_idx & kMask) ^ ((logical_chunk_idx >> 3u) & kMask));
}
```

**为什么需要它**：稀疏 KV 在通用路径下由 `cp_async_cg<256>`（16B chunk）搬进 smem（`:540-543`），
而不能像 dense 版那样交给 TMA 自动 swizzle。于是 smem 的 KV 排布必须手工打散以避免 bank conflict：
把 `logical_chunk_idx` 的低 `b = log2(kMask+1)` 位与「右移 3 位后的同宽位域」做 XOR
（`kMask = kSwizzleMode/16 - 1`，即 swizzle 模式是「以 16B 为单位」的模式；模板实参是
`kHeadDim/kPackFactor`，fp8=128、fp4=64，见 `:496` 与 `:57`）。

**性质**（H100 实测坐实）：它是**块内对合置换**——`f(f(x)) = x`，且 `f` 是 `[0,N)` 上的双射；
与「独立位域 XOR 模型」逐点相等。对 fp8（`kSwizzleMode=128`、`kMask=7`）XOR 的是 bits[0,3) 与
bits[3,6)；对 fp4（`kSwizzleMode=64`、`kMask=3`）是 bits[0,2) 与 bits[3,5)。这是精简版的
`cute::Swizzle`（少一个使用的地址位），足够打散 16B chunk 的 bank。

## 3. B 段：metadata 的两种打包结构

metadata kernel（`scheduler/sm100_sparse_mqa_logits_metadata.cuh`，第 22 篇）产出的头/块信息被主
kernel 直接按字节 reinterpret（`:211-212` 的 `kv_splits`），所以布局是硬契约：

```cpp
// layout/sparse_mqa_logits.cuh:31-61（KVSplitHeader，节选）
static constexpr uint32_t kContiguousFlag  = 0x80000000u;
static constexpr uint32_t kPartialTailFlag = 0x40000000u;
static constexpr uint32_t kFlagMask = kContiguousFlag | kPartialTailFlag;
uint32_t packed_num_kv_blocks;  // 高 2 位标记 + 低位块数
CUTLASS_HOST_DEVICE static constexpr uint32_t get_num_kv_blocks(uint32_t p) { return p & ~kFlagMask; }
CUTLASS_HOST_DEVICE static constexpr bool is_contiguous(uint32_t p)     { return (p & kContiguousFlag) != 0; }
CUTLASS_HOST_DEVICE static constexpr bool has_partial_tail(uint32_t p)  { return (p & kPartialTailFlag) != 0; }
```

```cpp
// layout/sparse_mqa_logits.cuh:63-72（KVBlockInfo，节选）
KVBlockInfo(uint32_t physical_kv_block_idx, uint32_t q0_slot_offset, uint32_t q1_slot_offset):
    physical_kv_block_idx(physical_kv_block_idx),
    packed_slot_offsets(q0_slot_offset | (q1_slot_offset << kNumSparseSlotBits)) {}
```

**两个非平凡点**：

1. **split 头把「块数」和两个调度标记塞进一个 `uint32`**：`is_contiguous` 让连续版主机在
   `:437-457` 走「一条 TMA lane 替所有 copy warp」的快路；`has_partial_tail` 让连续版在 `:463-505`
   走尾块裁剪慢路。两个标记都藏在最高 2 位，`get_num_kv_blocks` 用 `& ~kFlagMask` 还原块数。
2. **每个 KV 块对两个 Q token 各有一个「split 内槽号」**，各占 16 位（`kNumSparseSlotBits=16`，
   `kInvalidSparseSlot=0xFFFF` 表示该 Q token 此块无效）。第 22 篇的 §3.2 讲了两 token 归并去重——
   这里就是它落到一个 `uint32` 的物理形式。

实测：`KVSplitHeader` 四种块数 × 四种标记组合、`KVBlockInfo` 四种 q0/q1（含 4095、0xFFFF）共
32 例，打包/解码 **0 不符**。

## 4. C 段：两条 KV 访问路径的几何

### 4.1 连续版：尾块裁剪

```cpp
// impls/sm100_sparse_mqa_logits.cuh:96-98
CUTLASS_DEVICE uint32_t get_num_valid_tokens(const KVBlockRef kv_block_ref) const {
    return cute::min(SPARSE_BLOCK_KV, num_kv_tokens - cute::min(kv_block_ref, num_kv_tokens));
}
```

连续版的 `KVBlockRef` 就是 `physical_kv_block_idx`，而它在此处被当作 **KV token 起点**
（`resolve_kv_block` `:80-82` 直接透传；`get_kv_block` `:84-88` 再乘 `kNumQKBytesPerToken`，
`:86-87`）。`get_num_valid_tokens` 给出「从该起点到 KV 末尾、最多一个 sparse block」的有效 token 数。
`:461-505` 的 `kNeedsTailMask` 分支据此把无效 token 用 `cp_async_cg_zfill`（源 size=0）零填充，
`:493-496`——**这是稀疏块与「不满的尾块」共存的唯一处理点**。实测（SB=16, num_kv=100）：
起点 0/5/84 满 16；起点 88 只剩 12、90 只剩 10；起点 ≥100 为 0（不越界读）。

### 4.2 分页版：页内映射

```cpp
// impls/sm100_sparse_mqa_logits.cuh:121-131（节选）
constexpr uint32_t kNumKVBlocksPerPage = PAGE_KV / SPARSE_BLOCK_KV;
const uint32_t physical_page_idx = physical_kv_block_idx / kNumKVBlocksPerPage;
const uint32_t kv_block_idx_in_page = physical_kv_block_idx % kNumKVBlocksPerPage;
const auto page = fused_kv_cache + (uint64_t)physical_page_idx * kv_page_stride_bytes;
const uint32_t token_idx_in_page = kv_block_idx_in_page * SPARSE_BLOCK_KV;
return KVBlockRef(
    page + token_idx_in_page * kNumQKBytesPerToken,                                  // KV 区
    reinterpret_cast<const uint32_t*>(page + PAGE_KV * kNumQKBytesPerToken) + token_idx_in_page);  // SF 区
```

**为什么能这么做**：`DG_STATIC_ASSERT(PAGE_KV % SPARSE_BLOCK_KV == 0)`（`:679`）保证
**sparse 块不跨页**，于是「物理块 → (页号, 页内 token 起点)」是一次除法/取模。fused page cache 里
KV 区在前、SF 区紧随其后（SF 基址 = 页基址 + `PAGE_KV * bytes/tok`），SF 每 token 一个 `uint32`
（`token_idx_in_page` 直接是 SF 下标）。实测（PAGE=64, SB=16, bytes/tok=128）40 块 0 不符。

## 5. D 段：math warp 的 ReLU×w 归约与压缩列映射

math warpgroup（`:551-641`）一个 split 里每个线程负责一个 KV token（`:587-590`）：

```cpp
const uint32_t token_in_kv_split = warpgroup_idx * UMMA_M + math_thread_idx;   // :588
const uint32_t kv_block_idx_in_split = token_in_kv_split / SPARSE_BLOCK_KV;    // :589
const uint32_t token_in_kv_block = token_in_kv_split - kv_block_idx_in_split * SPARSE_BLOCK_KV;  // :590
```

**读 TMEM 按头分块**：`kNumHeadsPerLoad = math::constexpr_gcd(kNumHeads, 16u)`（`:164`）——整数个
4/8/16 头地读，避免读到最后一个 token 之外的 TMEM 列（`:163` 注释）。每块做
`cvt.rn.relu.bf16x2.f32` + `__hfma2` 累到两个 bf16x2 累加器（`:617-622`），最后
`__hadd2_rn(sum_0,sum_1)` 再 `__hadd_rn` 成一个 bf16（`:624-625`）——与 dense 版同源，只是权重下标
按 `(head_base+head_offset)/2` 取（`:620-621`）。

**压缩列映射**（`:627-633`）：

```cpp
const uint32_t q_slot_offset = (packed_slot_offsets >> (q_token_offset * kNumSparseSlotBits)) & kInvalidSparseSlot;
if (q_slot_offset != kInvalidSparseSlot) {
    const uint32_t q_slot_base = q_token_offset == 0 ? q0_slot_base : q1_slot_base;
    const uint32_t output_col_idx = (q_slot_base + q_slot_offset) * SPARSE_BLOCK_KV + token_in_kv_block;
    output_rows[q_token_offset][output_col_idx] = reduced;
}
```

**三个要点**（H100 实测坐实）：

1. **token 分解是 `[0, SPLIT_KV)` 的双射**：`warpgroup*128 + tid`（128 个线程 × nwg 组）恰好覆盖
   512 个 KV 行，`block*SB + t` 还原原 token。
2. **每个 Q token 只有效块写、无效块跳过**（`kInvalidSparseSlot`）。给定有效块在 split 内的槽号是
   相异的 `0..n-1`（第 22 篇 metadata kernel 的承诺），`(q_slot_base + slot)*SB + t` 会把有效块映射到
   **互不重叠、紧密的压缩列区间** `[q_slot_base*SB, (q_slot_base+n)*SB)`，每列恰写一次——这就是
   「压缩输出」`[T, num_max_sparse_blocks*SB]`（`attention.hpp:369-372`）的物理保证。
3. **两个 Q token 各有独立的 `q_slot_base`**（`q0_slot_base`/`q1_slot_base`，`:595-596`），因为它们
   有效块数可能不同，压缩区间起点也不同。

## 6. E 段：资源账

- **线程数** `get_num_threads(nwg)`：fp8（nwg=4）→ **768**，fp4（nwg=5）→ **896**（`layout:19-21`）。
- **TMEM 列对齐**：真实 `get_num_aligned_tmem_cols<N>()`（`common/utils.cuh:42-50`）把列数向上取到
  `{32,64,128,256,512}` 之一。实测边界 `32/33/64/128/256/376/428/512` 全部与模型一致。各配置
  （heads=32）raw：fp8 **376**、fp4 **428**，都对齐到 **512**（恒不越界，`:196`）。
- **每 copy warp 的 KV 块数/ SF 块数**：`kNumKVBlocksPerWarp = UMMA_M/SB`、`kNumSFChunksPerKVBlock
  = SB*4/16`；两条静态断言（`:424-425`）要求 `kvb % 2 == 0` 且 `kvb * sfch == 32`——实测 SB=8 得
  16×2、SB=16 得 8×4，都成立。
- **`SharedStorage` 布局**（`layout:103-138`）：成员顺序 + `alignas` 就是共享内存布局本身
  （第 82 篇）。实测 4 组配置的 8 个数据成员 `offsetof` 与独立字节模型**全部命中**：

  | 配置 | `sizeof` (B) | vs SM100 `smem_capacity=232448` |
  |---|---|---|
  | fp8 / SB=16 | 222208 | OK |
  | fp8 / SB=8 | 223232 | OK |
  | fp4 / SB=16 | 229376 | OK |
  | fp4 / SB=8 | 230912 | OK |

  host wrapper 直接 `sizeof(SharedStorage<...>)` 作 `num_smem_bytes`，并在 JIT 源码里
  `static_assert(sizeof(...) == num_smem_bytes)` 把它钉死（`hpp:73-80/101-103`）。

## 7. F 段：SF 的 4×32 → 32×4 转置

SF 转置 warp（`:323-369`）在把 `sf_q`/`sf_kv` 送 UTCCP 前，先做 warp 内 4×32→32×4 转置
（`:325-333`）：

```cpp
for (uint32_t sf_base = 0; sf_base < num_sf; sf_base += kNumUTCCPAlignedElems) {
    uint32_t values[4];
    for (uint32_t i = 0; i < 4; ++ i) values[i] = ptx::ld_shared(smem_ptr + sf_base + i * 32 + lane_idx);
    __syncwarp();
    ptx::st_shared(smem_ptr + sf_base + lane_idx * 4, values[0], values[1], values[2], values[3]);
}
```

这与 dense 版的 `utccp_required_smem_warp_transpose`（第 113 篇 §6）逐字同源：`dst[lane*4+i] ==
src[i*32+lane]`，实测 128 元素 0 不符。区别只是稀疏版一次转 `kNumSFQ`（Q，128 个）与
`SPLIT_KV`（KV，512/640 个），再 `SM100_UTCCP_4x32dp128bit_1cta::copy` 进 TMEM
（`:346-364`，注意是 **1cta** 版，不同于 Mega MoE 的 2cta 版）。

## 8. JIT 包装层要点

`csrc/jit_kernels/impls/sm100_sparse_mqa_logits.hpp`：

- `get_sparse_mqa_split_kv`（`:15-18`）：fp4=640 / fp8=512。
- `get_num_metadata_bytes`（`:20-36`）：分页与连续两种 split 上界公式（第 22 篇 §7 已复刻）。
- `launch_sm100_sparse_mqa_logits`（`:38-133`）：`swizzle_mode = fp4?64:128`（`:57`）、
  `num_kv_stages = fp4?5:3`（`:49`）、`num_smem_bytes = sizeof(SharedStorage<...>)`（`:73-79`）；
  TMA 描述符 Q/SF-Q/weights（`:58-64`）与连续版的 KV/SF-KV（`:69-71`）；JIT 源码里带
  `static_assert(sizeof(SharedStorage<...>) == num_smem_bytes)`（`:101-103`）。
- metadata kernel（`:135-200`）：`kNumMetadataThreads=256`、`kNumKVSplitsPerEntry=8`、
  `num_ctas = min(num_q_tokens 或 ceil/2, num_sms*4)`、smem 预算 `align(...,128)`。
- 四个 host 入口（`attention.hpp:357/409` 断言 `arch_major==10`）在 sm90 全部被拒（第 22 篇 §7）。

## 9. H100 实测

`src/sparse_mqa_logits_kernel.cu`（直接 include 真实 `layout/sparse_mqa_logits.cuh`）+ 
`harness/exp_sparse_mqa_logits_kernel.{sh,out.txt}`，在 `NVIDIA H100 80GB HBM3`（sm90）上
**ALL PASS**：

```
## Part A. get_swizzled_kv_chunk_idx         N=64/128：bijection/involution/=model 全 yes
## Part B. KVSplitHeader / KVBlockInfo      打包/解码 32 例，不一致 0
## Part C. 连续尾块裁剪 + 分页页内映射        尾块 12/10/0 正确；40 块映射不一致 0
## Part D. 压缩列映射                        token 分解 512/512 唯一；区间 [80,400) 每列恰一次
## Part E. 线程数 768/896；TMEM 对齐 8 档全对；SharedStorage 4 组 offsetof 8/8
## Part F. SF 4×32→32×4                     不一致 0 / 128
=== 汇总: ALL PASS ===
```

关键数字：

- **smem swizzle** 是块内对合置换，与独立 XOR 位域模型逐点相等（fp8 mask=7 / fp4 mask=3）。
- **尾块裁剪**：起点 88→12、90→10、≥100→0，坐实 `min(SB, num_kv - min(ref, num_kv))`。
- **分页映射**：40 个连续块（PAGE=64、SB=16）的 KV/SF 字节偏移 0 不符，页内 4 块连续、
  跨页跳一个 `page_stride`，SF 区在页内 KV 区之后。
- **TMEM 预算**：fp8 raw 376 / fp4 raw 428 → 都对齐 512（恒 `<= 512`）。
- **`SharedStorage`**：fp8/sb16 = 222208 B、fp4/sb8 = 230912 B，均 `< smem_capacity (232448)`。

## 10. 结论与边界

- 稀疏 MQA logits 主 kernel 用**两条 `KVAccessor` 模板路径**把「连续张量（可 TMA）」与
  「fused page cache（不可 TMA）」折叠进同一份核心逻辑；分叉只在
  `kSupportsContiguousTMA`/`kNeedsTailMask` 两个编译期布尔。
- 与 dense 版（第 113 篇）最大的结构性差异是**KV 走 `cp.async` gather 而非 TMA**：于是有了
  §2 的自定义 smem swizzle、§4 的尾块零填充，以及 `num_kv_stages` 随 FP8/FP4 变化的流水深度。
- **SM100 专属、仅分析**：UMMA（`SM100_MMA_MXF8F6F4_SS`/`MXF4_SS`）、tcgen05、UTCCP（1cta）、
  PDL `cudaGridDependencySynchronize`。本篇在 H100 上验证的是其中**架构无关的六段**（swizzle /
  打包 / 访问几何 / 压缩列映射 / 资源账 / SF 转置），不涉及任何 SM100 指令。
- 与 SM100 dense MQA（第 113 篇）的对照：dense `UMMA_N=128`（钉死），sparse `UMMA_N=2*kNumHeads≤64`；
  dense `kNumTmemStages==kNumMathWarpGroups`，sparse `kNumTmemStages=5` 固定且 `>=` math 组数；
  dense 的 KV 走 TMA，sparse 走 `cp.async` gather。

## 11. 参考资料

- DeepGEMM 仓库：`https://github.com/deepseek-ai/DeepGEMM`（本地 `057ca59`，Public release 26/09/30）。
- 本专题：第 19 篇（SM90 dense MQA）、第 22 篇（SM100 sparse MQA 概览 + metadata）、
  第 23/24 篇（SM100 dense 调度器）、第 60 篇（MQA logits 验收测试）、第 113 篇（SM100 dense 主 kernel）。
- 实测原始输出：`harness/exp_sparse_mqa_logits_kernel.out.txt`。
