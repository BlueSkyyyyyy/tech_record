# SM100 dense MQA logits 主 kernel 逐段精读：TMEM 头块遍历、bf16x2 归约与分页生产者均分

> 本文对应路线图 **P103-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 113 篇。
>
> 第 19/23 篇已经拆过 SM100 dense MQA logits 的**调度器**（`sm100_mqa_logits_metadata` +
> `SM100MQALogitsScheduler`，H100 真机跑真实设备代码）；第 22 篇拆过稀疏版的 metadata 与主
> kernel 概览。但第 23 篇 §5 对主 kernel（`impls/sm100_mqa_logits.cuh`）只给了**一页摘要**：
> 「五类 warp、UMMA、UTCCP、`visit_tmem_head_chunks` 按 16/8/4 读 TMEM、ReLU×权重 bf16x2 归约、
> 压缩偏移写回」——列了结论，没有逐段读，也没有对其中**架构无关**的部分做真机验证。本篇把它
> 补全：逐段精读主 kernel，并把其中不依赖任何 SM100 指令的五段逻辑在 H100 上跑真实设备代码。
>
> **可跑性**：主 kernel 的 UMMA（`ptx::SM100_MMA_MXF8F6F4_SS`）、tcgen05、UTCCP、2-SM TMA 是
> **SM100 专属，仅分析**；但下面五段是纯整数 / 纯 bf16 运算，在 H100(sm90a) 上**直接抽取真实
> 函数原样编译运行**（只把 SM100 专属的 `ptx::tmem_load_32dp32b` 换成记录桩）：
>
> | 段 | 真实行号 | 实测 |
> |---|---|---|
> | A `dispatch_num_block_tokens` | `:27-36` | 运行期 token 数 → 编译期 `Int<min(max(N,1),BLOCK_Q)>` |
> | B `visit_tmem_head_chunks` | `:39-49` | 16/8/4 头块分解，起点连续、宽度和恰为 H |
> | C `reduce_pairs` + `store_token` | `:473-489` | bf16x2 累加的 ReLU×w 归约 + 压缩写回（逐位） |
> | D `issue_paged_split` 均分算术 | `:184-221` | 两生产者对 KV 页/SF 份额的划分 |
> | E `utccp_..._transpose` | `:288-296` | `src[i*32+lane] → dst[lane*4+i]` 的 4×32→32×4 转置 |
>
> 源码坐标（`~/github/DeepGEMM`，本地 commit `057ca59`、Public release 26/09/30）：
> - 主 kernel：`deep_gemm/include/deep_gemm/impls/sm100_mqa_logits.cuh`（609 行）
> - 共享内存布局：`deep_gemm/include/deep_gemm/layout/mqa_logits.cuh`（67 行）
> - 设备数学/PTX：`ptx/{utils,ld_st,tcgen05}.cuh`、`mma/sm100.cuh`
> - host 入口：`csrc/apis/attention.hpp:18-282`、`csrc/jit_kernels/impls/sm100_mqa_logits.hpp:11-155`
> - 实测：`src/mqa_logits_dense_epilogue.cu` +
>   `harness/exp_mqa_logits_dense_epilogue.{sh,out.txt}`（H100，**ALL PASS**）
>
> 全文用「目标卡」指代目标 AI 卡。

## 1. 主 kernel 全景（补齐第 23 篇只给的摘要）

### 1.1 网格与几何

`sm100_mqa_logits_core_impl`（`:52-534`）是一次 launch 的物理坐标。`CSRC` 侧 host 把模板参数
算好（`sm100_mqa_logits.hpp:113-155` 的 `MQALogitsConfig`）：

- **Q tile 恒为 128 个 (token, head) 行**：`block_q = get_mqa_logits_block_q(num_heads) = 128/num_heads`
  （`sm100_mqa_logits.hpp:21-25`），`umma_n = align(block_q*num_heads, 8) = 128`（`:143`）。
  即 H=64 时 `BLOCK_Q=2`、H=128 时 `BLOCK_Q=1`（`attention.hpp:110`）。
- `SPLIT_KV = kMQALogitsSplitKV = 384`（`sm100_mqa_logits.hpp:15`），`num_math_threads =
  (384/128)*128 = 384`（`:131`）→ `kNumMathWarpGroups = 3`（kernel `:58`）。
- `kNumTmemStages = 3`（`sm100_mqa_logits.hpp:16`），且 `DG_STATIC_ASSERT(kNumTmemStages ==
  kNumMathWarpGroups)`（kernel `:99`）——**每个 math warpgroup 恰好对应一个 `UMMA_M=128` 的 KV
  tile / 一段 TMEM 累加器**。
- 每个 KV split 一条 UMMA：`UMMA_M=128`（KV 行）× `UMMA_N=128`（(token,head) 列），
  `UMMA_K = fp4?64:32`（kernel `:90-92`），A=KV、B=Q（`:381-382`）——与 SM90 版一致，
  「KV 行 × Q 列」使得每头点积排成累加器列。

### 1.2 五类 warp

`kSpecWarpStart = kNumMathWarpGroups*4 = 12`（`:74`），所以：

| warp | 行号 | 职责 |
|---|---|---|
| 0..11 | `:430-533` | **math warpgroup**（3 组 × 4 warp）：读 TMEM、ReLU×w、写 logits |
| 12（`+0`） | `:278-280` | **SF / Q 生产者**：发 SF-KV + SF-Q，并驱动整个 Q 流水（`produce_kv<true>`） |
| 13（`+1`） | `:281-283` | **KV 生产者**：发 KV TMA（`produce_kv<false>`） |
| 14（`+2`） | `:284-364` | **UTCCP**：把 `sf_q/sf_kv` 就地转 4×32 布局并搬进 TMEM |
| 15（`+3`） | `:365-429` | **UMMA**：单 warp 发 block-scaled 矩阵乘 |

生产者侧 barrier（`:110-129`）：`full_q.init(2)`（两个生产者都 arrive）、`empty_q.init(kNumMathThreads+
32)`、`full_kv.init(2)`、`empty_kv.init(1)`、`full_tmem.init(1)`、`empty_tmem.init(128)`。
寄存器两档（`:140-146`）：specialized 固定 56，math 从 CTA 入口寄存器池反推，
`DG_STATIC_ASSERT` 保证不超额。`:148` 先 `cudaGridDependencySynchronize()`（PDL，等待上游
metadata kernel）。

### 1.3 TMEM 列预算

`:101-108` 把 TMEM 切成三块：

```
kNumAccumTmemCols   = UMMA_N * kNumTmemStages          // 3 段累加器
kNumSFQColsPerStage = kNumSFQ / 32                     // kNumSFQ  = align(UMMA_N,128) = 128 → 4
kNumSFKVColsPerStage= kNumSFKV / 32                    // kNumSFKV = align(SPLIT_KV,128) = 384 → 12
kNumTmemCols = get_num_aligned_tmem_cols<accum + kNumQStages*SFQ + kNumKVStages*SFKV>
```

`kTmemStartColOfSFQ = kNumAccumTmemCols`、`kTmemStartColOfSFKV = ... + kNumQStages*SFQ`，
`DG_STATIC_ASSERT(kNumTmemCols <= 512)`（`:108`）。`kNumSFQ/kNumSFKV` 与 `kNumUTCCPAlignedElems=128`
都来自 `layout/mqa_logits.cuh:20,26-27`：**SF 进 TMEM 前必须先按 128 元素对齐并用 UTCCP 转成
4×32 布局**（§8 的 E 段）。

## 2. A 段：运行期 token 数 → 编译期 `cute::Int`

```cpp
// impls/sm100_mqa_logits.cuh:27-36
template <uint32_t kBlockQ, uint32_t kCandidate = kBlockQ, typename Fn>
CUTLASS_DEVICE void dispatch_num_block_tokens(const uint32_t& num_block_tokens, Fn&& fn) {
    if constexpr (kCandidate <= 1) { fn(cute::Int<1>{}); }
    else if (num_block_tokens >= kCandidate) { fn(cute::Int<kCandidate>{}); }
    else { dispatch_num_block_tokens<kBlockQ, kCandidate - 1>(num_block_tokens, static_cast<Fn&&>(fn)); }
}
```

**为什么需要它**：分页模式下最后一个 Q block 的有效 token 数 `task.num_q_tokens` 是运行期值
（`:522`），但 token 循环 `for (i = 0; i < kNumValidTokens; ++i)` 必须**编译期展开**——因为
每个 token 的 `weights[i]` 是一整行 `nv_bfloat162[BLOCK_Q][kNumHeads/2]` 寄存器数组（`:444`），
下标必须是常量。`dispatch_num_block_tokens` 把运行期整数「折射」成一组模板实例
`Int<BLOCK_Q>, Int<BLOCK_Q-1>, …, Int<1>`，每个实例里 `kNumValidTokens` 是 `constexpr`。

**两个要点**（H100 实测坐实）：

1. **夹取区间是 `[1, BLOCK_Q]`**。`N >= kCandidate` 才取大者、否则向下递归到 1，所以
   `kNumValidTokens = max(1, min(N, BLOCK_Q))`：`N=0`（空 Q block）也退化为「处理 1 个 token」的
   实例，而不是 0（`:29-30` 的 `kCandidate <= 1` 分支正是终点）。这与第 13 篇调度器「按代价二分、
   边界落在 split 粒度」相呼应：**空/不满的 Q block 仍按 1 个 token 的尾巴处理**，不影响正确性。
2. 连续模式下 `process_q_block(cute::Int<BLOCK_Q>{})`（`:524`）直接跳过折射——连续模式没有
   局部 Q block 不满的概念（每块都补满到 `BLOCK_Q`）。

实测（`BLOCK_Q ∈ {1,2,4}` × `N = 0..BLOCK_Q+2`）全部满足 `Int<max(1,min(N,BLOCK_Q))>`（见 §9）。

## 3. B 段：TMEM 按头切 16/8/4

```cpp
// impls/sm100_mqa_logits.cuh:39-49
template <uint32_t kNumHeads, uint32_t kHeadBase = 0, typename Fn>
CUTLASS_DEVICE void visit_tmem_head_chunks(const uint32_t& tmem_col, float* accum, Fn&& fn) {
    constexpr uint32_t kNumRemaining = kNumHeads - kHeadBase;
    if constexpr (kNumRemaining > 0) {
        constexpr uint32_t kNumChunkHeads = kNumRemaining >= 16 ? 16 : (kNumRemaining >= 8 ? 8 : 4);
        ptx::tmem_load_32dp32b<kNumChunkHeads>(tmem_col + kHeadBase, reinterpret_cast<uint32_t*>(accum));
        cutlass::arch::fence_view_async_tmem_load();
        fn(cute::Int<kHeadBase>{}, cute::Int<kNumChunkHeads>{});
        visit_tmem_head_chunks<kNumHeads, kHeadBase + kNumChunkHeads>(tmem_col, accum, static_cast<Fn&&>(fn));
    }
}
```

**为什么是 16/8/4**：`ptx::tmem_load_32dp32b<W>`（`ptx/tcgen05.cuh:191-207`）把 `W` 个 32 位
值从「32 个 lane × 每 lane 一段」的 TMEM 布局读进寄存器，`W` 只支持 `4/8/16/32/64/128`
（`DG_STATIC_ASSERT`）。`32dp` = 32 data path，即一次覆盖 32 行 KV；每行有几个 head 由 `W` 定。
主 kernel 里 token i 的 heads 落在 TMEM 列 `[i*kNumHeads, (i+1)*kNumHeads)`（`:515`），
所以分解的粒度直接决定 TMEM 读指令的宽度。

**为什么贪心 16→8→4 就够**：头数被 `DG_STATIC_ASSERT(kNumHeads % 4 == 0)`（`:441`）钉死，
剩余量永远是 4 的倍数，`min(rem,16)` 向下取到最近的合法宽度（16/8/4）恰好整除。H100 实测
`H ∈ {4,8,12,16,20,24,32,64,128}`：分解起点连续、宽度和恰为 H（如 H=12→8+4、H=20→16+4、
H=128→16×8）。

**为什么 `accum` 只有 16 宽**：`:445` 声明 `float accum[kNumHeads < 16 ? kNumHeads : 16]`——
单次最多读 16 个 head，读一块、归约一块（§4），所以一块缓冲就够。

**为什么 process 用 `Int<kHeadBase>`/`Int<kNumChunkHeads>`**：回调需要知道当前块在权重数组里的
偏移（`weights[i] + kHeadBase/2`）与块宽，作为编译期常量传下去，`#pragma unroll` 才能完全展开。

## 4. C 段：ReLU×w 的 bf16x2 归约 + 压缩写回

这是整个 kernel 的 epilogue 数学本体（`:452-519` 的 `process_q_block`）。

### 4.1 权重一次性读寄存器

```cpp
// :461-470
for (i = 0; i < kNumValidTokens; ++i) {
    const auto smem_weights_row = smem.smem_weights[q_stage_idx] + i * kNumWeightElementsPerRow;
    const auto packed_row = reinterpret_cast<const uint32_t*>(smem_weights_row);
    for (j = 0; j < kNumHeads/2; ++j) { weights[i][j] = *reinterpret_cast<const nv_bfloat162*>(packed_row + j); }
}
```

权重是 `[num_q_tokens, num_heads]` BF16（`sm100_mqa_logits.hpp:185-188`），每行按 16 字节 TMA
对齐补零。**一个 token 的整行权重驻留寄存器**——因为同一 token 的所有 KV 行共享同一组权重，
而 KV 行数（`SPLIT_KV=384`）远多于头数，读一次远优于每 KV 行重读。

### 4.2 bf16x2 两路累加

```cpp
// :473-479  store_token
const auto sum = __hadd2_rn(sum_0, sum_1);
const auto result = __hadd2_rn(__low2bfloat162(sum), __high2bfloat162(sum));
if (seq_k_start[i] <= kv_offset and kv_offset < seq_k_end[i])
    ptx::st_global_low_bf16(output_bases[i] + kv_offset, result);

// :480-489  reduce_pairs
for (head_offset = 0; head_offset < num_heads; head_offset += 4) {
    const auto accum_pair_0 = make_float2(chunk[head_offset],     chunk[head_offset + 1]);
    const auto accum_pair_1 = make_float2(chunk[head_offset + 2], chunk[head_offset + 3]);
    sum_0 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_0), chunk_weights[head_offset/2],     sum_0);
    sum_1 = __hfma2(ptx::cvt_relu_bf16x2_f32(accum_pair_1), chunk_weights[head_offset/2 + 1], sum_1);
}
```

**语义**：`out = Σ_h relu(acc_h) · w_h`。但注意关键设计——**整个头归约在 bf16 里做**：
`cvt_relu_bf16x2_f32` 先把 `relu(acc)` RNE 到 bf16x2（`ptx/utils.cuh:55-63`），
`__hfma2` 是 bf16x2 的融合乘加，`sum_0/sum_1` 是两条 bf16x2 累加器（偶/奇对），
最后 `__hadd2_rn` 折半再 broadcast 到两个 bf16、`st_global_low_bf16` 只存低 16 位
（`ptx/ld_st.cuh:245-247`）。

**为什么可以在 bf16 里归约**：这是一条**索引器（indexer）打分**，不是最终注意力输出。
它只用来 top-k 选块/选 token，量化到 bf16 带来的抖动不会改变选块结果的量级（第 22 篇测过
dense kernel 自身的量化噪声层 ~3.5e-2）。用 bf16 累加省的是寄存器与 ALU 吞吐——这正是
`s_0/s_1` 两条累加器存在的意义：把 `h` 维的两路独立点积并行起来，最后才相加。

### 4.3 压缩偏移与区间裁剪

`output_bases[i] = logits + (task.q_token_base + i)*logits_stride - seq_k_start[i]`（`:455-458`）。
写地址是 `output_bases[i] + kv_offset`，所以**输出列 = `kv_offset - seq_k_start[i]`**：每个 Q
token 的有效 KV 区间 `[seq_k_start[i], seq_k_end[i])` 被**压缩**成从 0 开始的连续列。
`:477` 的 `if` 保证只写落在区间内的 KV 列（区间外的列可能是别的 token 的有效列，也可能在
缓冲之外）。`logits_stride` 是行长，`output` 是 bf16 指针。

H100 实测（`H=64`、64 个 KV 行、窗口 `[8,40)`、stride 64）：
`output[0..32)` 与 device 结果逐位相等、`output[32..64)` 保留 poison `0xA5A5`（窗口外一格不碰）。

## 5. D 段：分页模式下两个生产者如何均分 KV 页

连续模式下 `issue_contiguous_split`（`:163-181`）把 `SPLIT_KV` 按 128/256 token 一条 TMA 切成
2~3 条（`kNumKVTokensPerTMA = SPLIT_KV % 256 == 0 ? 256 : 128`，`:164`）。分页模式复杂得多：

```cpp
// :184-221  issue_paged_split<kNumPagesPerSplit>
constexpr uint32_t kPageKV = SPLIT_KV / kNumPagesPerSplit;
constexpr uint32_t kNumGatherPages = kNumPagesPerSplit / 4 * 4;
constexpr uint32_t kNumSFTMAs = kNumGatherPages / 4 + kNumPagesPerSplit - kNumGatherPages;
constexpr uint32_t kNumKVPagesFromSFProducer = kNumPagesPerSplit > 2*kNumSFTMAs
                                             ? (kNumPagesPerSplit - kNumSFTMAs)/2 : 0;
constexpr uint32_t kv_page_begin = kIsSFProducer ? 0 : kNumKVPagesFromSFProducer;
constexpr uint32_t kv_page_end   = kIsSFProducer ? kNumKVPagesFromSFProducer : kNumPagesPerSplit;
```

其中 `kNumPagesPerSplit = SPLIT_KV / PAGE_KV`（`scheduler/sm100_paged_mqa_logits.cuh:166`），
`PAGE_KV ∈ {32,64,128}`，`SPLIT_KV=384`。

**SF 生产者**（`:194-211`）：先对 `kNumGatherPages` 个页**每 4 页合并成一条 `tma_gather4`**
（`:196-204`），剩 `kNumPagesPerSplit - kNumGatherPages` 页逐个 `tma::copy`（`:205-211`）。
所以 SF 生产者一共发 `kNumSFTMAs` 条指令。**KV 生产者**（`:214-220`）：对 `[kv_page_begin,
kv_page_end)` 逐页发 KV TMA。

**「为什么」**：分页下 KV 存在不连续的物理页里，搬一个 split 要发 `kNumPagesPerSplit` 条 KV TMA
和若干条 SF TMA，两条流必须**不重不漏**地覆盖。设计上：

- **KV 页由两个生产者分段**：SF 生产者做 `[0, kv_from_sf)`、KV 生产者做 `[kv_from_sf, pages)`，
  两段拼起来恰好是 `[0, pages)`。
- **是否让 SF 生产者分 KV 活，取决于它的 SF 活有多重**。`kNumSFTMAs` 被当作「SF 生产者的工作量
  等价页数」（一条 gather4 ≈ 1 页）；只有当 `pages > 2*kNumSFTMAs` 时，才把 `(pages-SFTMAs)/2`
  页 KV 也交给它，使两生产者**总工作量≈对半**。

H100 实测三档（`:184-191`）：

| PAGE_KV | pages | gather | SF-TMAs | KV(SF/KV) | 覆盖 |
|---|---|---|---|---|---|
| 32 | 12 | 12 | 3 | 4/8 | OK |
| 64 | 6 | 4 | 3 | 0/6 | OK |
| 128 | 3 | 0 | 3 | 0/3 | OK |

`kv_from_sf + kv_from_kv == pages` 恒成立。PAGE_KV=32 时 SF 生产者「4 页 KV + 3 条 SF TMA = 7
单位」对 KV 生产者「8 单位」；PAGE_KV=64/128 时两条流的 SF 活相对更重，就不再给它 KV 活，避免
一边倒。一条 gather4 能省 3 条指令，所以「每 4 页合并」优先。

## 6. E 段：SF 的 warp 内 4×32 → 32×4 转置

```cpp
// :288-296  utccp_required_smem_warp_transpose
uint32_t values[4];
for (i = 0; i < 4; ++i) values[i] = ptx::ld_shared(smem_ptr + i*32 + lane_idx);
__syncwarp();
ptx::st_shared(smem_ptr + lane_idx*4, values[0], values[1], values[2], values[3]);
```

**为什么必须转**：SF 从 global 经 TMA 进来时是「MN-major、4 个连续 scale 一行」的布局
（这正是第 25 篇讲的 TMA 对齐打包），而 UTCCP `4x32dp128bit`（`cute::SM100_UTCCP_4x32dp128bit_1cta`，
`:322-323`）要的是「32 个 lane × 4 拍」的布局。转置就是**原地**把 `src[i*32+lane]` 搬到
`dst[lane*4+i]`——4 个 32 位读 + 1 个 `st.shared.v4` 写。H100 实测 128 个元素恒等变换
`dst[lane*4+i] == src[i*32+lane]`，0 不符。

UTCCP warp（`:284-364`）对每个 stage 的 SF-Q（`:309-325`）与 SF-KV（`:344-362`）各转一遍，然后
`mma::sm100::replace_smem_desc_addr(sf_desc, smem_ptr)` + `SM100_UTCCP_...::copy(..., kTmemStartCol...)`
把 SF 搬进 TMEM 的 SF 列。`reuse_kv` 时（`:335-343`）跳过——SF 已在 TMEM 里、SMEM 也已转好，
再转一次就错了。

## 7. 主 kernel 与调度器的分工

主 kernel 依赖 `sm100_mqa_logits.hpp` 的 `MQALogitsConfig` 与 `csrc/apis/attention.hpp` 的 host：
`fp8_fp4_mqa_logits`（`attention.hpp:67-134`）在 `arch_major == 10` 时调 `sm100_mqa_logits(...,
schedule_meta)`（`:120-123`）；分页版 `fp8_fp4_paged_mqa_logits`（`:174-282`）结构相同。调度器
（第 13/23 篇）产出 `schedule_meta`，主 kernel 里 `make_scheduler(sm_idx)`（`:558-561`/`:594-597`）
构造 `SM100MQALogitsScheduler` / `SM100PagedMQALogitsScheduler`，每个 warp 各持一个实例
（`next_q_block(task)` 的参数依赖 warp 是 math 还是 specialized）。

- math warpgroup 用 `next_q_block(task, seq_k_start, seq_k_end)`（`:448`）额外取回每 token 的
  有效 KV 区间（用于 §4.3 的压缩写回）。
- producers/UTCCP/UMMA 用 `next_q_block(task)`（`:224`/`:302`/`:395`）。
- **`kNumTmemStages == kNumMathWarpGroups`**（`:99`）使「一个 KV split 的 3 段 UMMA」正好被
  3 个 math warpgroup 各吃一段，`tmem_pipeline.advance(kNumMathWarpGroups)`（`:495`）按 3 推进。

## 8. 与 SM90 dense MQA logits 的对照

| 维度 | SM90（第 19 篇，`sm90_fp8_mqa_logits.cuh`） | SM100（本篇，`sm100_mqa_logits.cuh`） |
|---|---|---|
| 点积 | WGMMA，Q 在 B 侧 / KV 在 A 侧 | UMMA（`MXF8F6F4`/`MXF4`），A=KV/B=Q，`UMMA_M=128/N=128/K=32\|64` |
| 累加器 | 寄存器（`wgmma` 直接落寄存器） | **TMEM**，3 段（每 math warpgroup 一段） |
| scale | per-tensor/行 scale | **block-scale**：`UTCCP_4x32` 把 SF 搬进 TMEM（第 10 篇） |
| 归约 | fp32 累加 + `shfl_xor` | **bf16x2 两路累加** + `cvt.relu`（§4） |
| 输出 | fp32（`attention.hpp:80` out=Float） | **bf16**（`attention.hpp:80` arch10→BFloat16） |
| 调度 | persistent round-robin | `sm100_mqa_logits_metadata` + 代价均衡（第 13/23 篇） |
| head 遍历 | 每 warp 处理固定 head 广播 | **16/8/4 TMEM 头块** + 权重驻寄存器（§3/§4） |

SM90 版能真机跑（第 19 篇 497 TFLOPS）；SM100 版只能分析。但两版的**数学语义一致**
（ReLU×w 头归约 + 压缩区间），只是 SM100 换成 UMMA/TMEM/bf16。

## 9. H100 实测

实验 `src/mqa_logits_dense_epilogue.cu`（`harness/exp_mqa_logits_dense_epilogue.sh`，
真机 `NVIDIA H100 80GB HBM3`、132 SM、sm90）。五段全部 **ALL PASS**：

- **Part A**：`BLOCK_Q ∈ {1,2,4}` × `N ∈ [0, BLOCK_Q+2]`，19 个点全部 `Int<max(1,min(N,BLOCK_Q))>`。
- **Part B**：`H ∈ {4,8,12,16,20,24,32,64,128}` 的头块分解与 host 独立复刻**逐项 0 不符**。
- **Part C**：device「一次性归约」与「按头块分段归约」**逐位相等 0/64**；device 与 host 逐位
  重放（host 用 double 精确重放 `__hfma2`）**0/64**；bf16 累加 vs fp32 参考：`max|Δ|=1.99e-1`、
  `mean|Δ|=5.5e-2`，`max |Δ|/Σ|term| = 2.55e-3`（≈2^-9 的逐项舍入积累；`max |Δ|/|ref|` 可达
  O(1) 纯粹是相消所致，`RMS(ref)=11.3`）；压缩写回窗口与值 **0 错**、窗口外保留 poison。
- **Part D**：三档分页几何的 KV 页划分覆盖无重叠（4/8、0/6、0/3），`kv_from_sf+kv_from_kv==pages`。
- **Part E**：4×32→32×4 转置 128 元素 **0 不符**。

（C 段的 host 重放把 `__hfma2`（device-only）用「double 精确算 `a*b+c` 再 RNE 到 bf16」复刻；
因 bf16 乘积只有 16 位有效位，double→float 精确、float→bf16 是单次舍入，与硬件一致。）

## 10. 结论与边界

- SM100 dense MQA logits 主 kernel = **SM90 dense 的 UMMA/TMEM/block-scale/bf16 升级**：
  A=KV/B=Q、3 段 TMEM 累加（每 math warpgroup 一段）、SF 经 UTCCP 进 TMEM。
- 其中**五段架构无关逻辑**首次被逐段读 + H100 真机验证：运行期 token 数折射（A）、TMEM 头块
  16/8/4 分解（B）、bf16x2 两路 ReLU×w 归约 + 压缩写回（C）、分页两生产者 KV 页均分（D）、
  SF 的 warp 内 4×32 转置（E）。
- **UMMA / tcgen05 / UTCCP / 2-SM TMA 仍是 SM100 专属，仅分析**——与 `fp8_fp4_mega_moe`、
  Mega Gate/mHC、sparse logits 的边界一致。真正评测该 kernel 需目标卡。
- 值得记住的语义：MQA logits 的**头归约刻意在 bf16 里做**（两路累加），因为它是索引器打分、
  不是最终输出；`outs = Σ_h relu(acc_h)·w_h` 的量化抖动不改变选块量级（第 22 篇）。

## 11. 参考资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM （公开版 commit `057ca59`）。
- 本专题第 19 篇（SM90 dense MQA logits，可跑）：`docs/19-attention-mqa-logits.md`。
- 本专题第 22 篇（SM100 sparse MQA logits）：`docs/22-sparse-mqa-logits.md`。
- 本专题第 23 篇（SM100 dense MQA logits 调度器，可跑）：`docs/23-mqa-logits-sm100-schedule.md`。
- 本专题第 10 篇（UTCCP 把 SF 搬进 TMEM）：`docs/10-mega-moe-mma-epilogue.md`。
- 本专题第 06 篇（总览）：`docs/06-summary.md`。
