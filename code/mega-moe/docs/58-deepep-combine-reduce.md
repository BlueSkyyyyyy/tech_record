# DeepEP combine 的归约 epilogue：top-k 去重、rank/topk 双布局与 bf16 直加旁路

> 本文对应路线图 **P48-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 58 篇。
> 目标：把 `docs/07` 只一笔带过的 **combine 归约 epilogue**（`combine_reduce` 家族）讲透——
> 它是 EP 通信在「专家算完之后、把结果加回源 token」这一步的**逐行实现**，也是 `docs/01` 的
> 「K 倍放大」在归约侧的落点。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`，`deep_ep.__version__ == 2.5.0`）：
> - 归约函数与选择逻辑：`deep_ep/include/deep_ep/impls/ep/combine_utils.cuh`（172 行）
> - 归约 epilogue kernel：`deep_ep/include/deep_ep/impls/ep/combine_reduce_epilogue.cuh`（145 行）
> - 主 combine kernel：`deep_ep/include/deep_ep/impls/ep/combine.cuh`（246 行，本地归约在 `:145-177`）
> - token/buffer 布局：`deep_ep/include/deep_ep/layout/ep/token.cuh`（146 行）
> - warp 内建与搬运原语：`deep_ep/include/deep_ep/common/ptx.cuh`
> - host launch 壳：`csrc/kernels/ep/combine.hpp:148-198`；调用点 `csrc/buffers/ep.hpp:925-950`
>
> **可跑性**：这两段设备代码**不依赖 NCCL GIN、UMMA、WGMMA**，只用 warp 内建 + 带谓词的
> `ld.global.nc` + TMA store（sm90 起）+ PDL。故在 **H100（sm90）上直接 include 真实头文件、
> 编译运行真实设备代码**，用真实对称 `recv_buffer` 布局喂进去，再与独立 host oracle **逐位对拍**
> （`harness/exp_deepep_combine_reduce.{sh,py,out.txt}`）。sm100 的融合 `add.rn.f32.bf16` 快路
> （`ptx.cuh:593-602`）在 sm90 自动回退为 bf16→fp32 add。

## 1. 一段话总览

combine 是 dispatch 的逆过程：专家算完后，每个专家输出要**按原路送回 token 的源 rank**，
并按该 token 的 top-k 路由权重求和。DeepEP 把它拆成**两阶段**：

1. **主 combine kernel**（`combine.cuh`，热路径，跑在 comm stream 上）：把每个 rank 的激活搬运回来，
   并做**本地归约**——同一个 token 的多个被选专家若落在**同一 rank**，在寄存器里先加成一个值。
2. **归约 epilogue**（`combine_reduce_epilogue.cuh`，可**延迟**到 compute stream 上再跑）：
   跨 rank 求和 + 加 bias（共享专家/残差）+ 写回 `combined_x` 与 top-k 权重。

两阶段之间隔着一次「按 rank 去重」：不管一个 token 有几个 top-k 落在同一 rank，那个 rank 只贡献一次。

## 2. 为什么 combine 需要一个独立的归约 epilogue

`combine` 的返回值里有一个 `EventOverlap`（`docs/07` §6）：**epilogue 可以延迟**。在
`csrc/buffers/ep.hpp:925-950`，host 把「分配输出 + 启动归约 epilogue」包成一个 **lambda**，等框架真正
需要 `combined_x` 时才在**当前 stream**（通常是 compute stream）上执行：

- 好处：主 combine kernel 只负责**通信搬运**，归约计算与前一层的后处理（比如写 bias）落在同一 stream，
  天然重叠；README `:279` 明说「`bias` 可以在 combine epilogue 里加共享专家输出或残差」。
- 代价：epilogue 与主 kernel 之间要靠 **PDL** 建立顺序。`combine_reduce_epilogue.cuh:57-59`：

  ```
  // Will block until the main combine kernel has finished and all data are visible
  // NOTES: PDL is used, please do not use `__ldg`
  cudaGridDependencySynchronize();
  ```

  这解释了为什么 epilogue 里读 `combined_topk_idx` 用普通 `[]`（`:67`）而不是 `__ldg`：**PDL 的
  依赖同步与 `__ldg`（非一致只读路径）配合会读到没有同步保证的数据**。host 侧
  `csrc/kernels/ep/combine.hpp:182-189` 的 launch 里 `enable_pdl = true`（`:188`）与之对应，
  且 `num_warps` 按「smem 能放下几个 token」推导（`:163-164`，上限 32）。

## 3. 两张布局：rank-layout vs topk-layout

归约 epilogue 的第一个抉择是**通信缓冲的外层维是什么**，由 `combine_utils.cuh:8-18` 的两个 constexpr
决定：

```
use_rank_layout<allow_multiple_reduction, num_ranks, num_topk>()      // :8-13
  = allow_multiple_reduction && num_ranks <= num_topk
get_num_tokens_in_layout<...>()                                        // :15-18
  = use_rank_layout ? num_ranks : num_topk
```

- **rank-layout**（`num_ranks <= num_topk`）：缓冲外层维 = **rank 数**，一个 rank 一格。
- **topk-layout**（否则）：缓冲外层维 = **top-k 数**，每个 top-k 槽一格。

在 `combine_reduce_epilogue.cuh:21-23`，`kNumRanks = (scaleout==1 ? scaleup : scaleout)` 同时被用来
推导 `kUseRankLayout` / `kNumTokensInLayout`，并直接进 kernel 模板：

```
const auto comm_buffer = layout::BufferLayout<false>(
    comm_token_layout, kNumTokensInLayout, kNumMaxTokensPerRank, recv_buffer);   // :44-45
```

**为什么需要两种布局**：当 `num_ranks <= num_topk` 时，任何 token 的 top-k 至多覆盖每个 rank 一次，
`rank` 天然唯一，直接用 rank 当索引最省；当 `num_ranks > num_topk` 时 rank 数多于 top-k 槽数，
用 rank 当外层维会浪费缓冲，于是改用 top-k 槽当索引。同一份 host 代码在
`combine.cuh:26-27`（主 kernel）与 `combine_reduce_epilogue.cuh:22-23`（epilogue）里**用同一个
constexpr**，保证两阶段的缓冲布局一致。

## 4. top-k 去重：`deduplicate` + `compute_topk_slots`

跨 rank 求和时，**一个 token 的多个 top-k 落在同一 rank 只能算一次**。做法是先在 warp 内按 key
去重，把「每个 key 的最高 lane」标出来做掩码，再从最低位往上把有效槽挤到前面：

```
auto reduce_valid_mask = should_deduplicate ?
    ptx::gather(ptx::deduplicate(deduplicate_key, lane_idx) and stored_dst_rank_idx >= 0) :
    ptx::gather(stored_dst_rank_idx >= 0);              // :86-88
int topk_slot_idx[kNumTokensInLayout];
compute_topk_slots(
    topk_slot_idx, reduce_valid_mask,
    [=](const int& idx) { return kUseRankLayout ? ptx::exchange(stored_dst_rank_idx, idx) : idx; });  // :89-95
```

- `ptx::deduplicate`（`ptx.cuh:544-546`）= `get_master_lane_idx(match(value)) == lane_idx`，
  `match` 是 `__match_any_sync`（`:519-521`），`get_master_lane_idx` 是 `bfind.u32`（`:537-542`，
  即 `31-__clz`，取**最高** lane）。所以每个同 key 组里**只有最高 lane 存活**。
- `compute_topk_slots`（`combine_utils.cuh:41-53`）用 `__ffs(mask)-1` 从最低位往上取，每取一个
  `mask &= mask-1`；即使槽用完了也**无条件** `exchange`（`:48-49`：`to avoid BRA.DIV`），
  最后按 `lowest_idx >= 0` 决定是否置 `-1`（`:51`）。`kNumValidTopk = kNumTokensInLayout`，
  多出来的槽是 `-1`。
- `-1` 槽**不会越界读**：`combine_reduce` 的源指针函数算出 `get_rank_buffer(-1)` 的越界地址，
  但真正的加载是 `ptx::ldg_with_ge_pred(ptr, slot_idx, 0)`（`ptx.cuh:223-236`），
  谓词 `setp.ge.s32 p, slot, 0` 让 `@p ld...` 对 `-1` 直接跳过、返回 0。这是「先算地址、后判谓词」
  的经典写法，也是为什么 `combine.cuh:160-163` 能放心地把 `slot_idx` 直接乘进偏移。

## 5. `combine_reduce`：bf16 直加旁路 vs fp32 累加

`combine_reduce`（`combine_utils.cuh:55-172`）是归约本体，向量类型 `vec_t` 由
`CombineVecTraits`（`:28-39`）决定：sm100 曾想用 32B 的 `longlong4_t`，但实测有性能退化，
`kUseLonglong4 = false`（`:34`），最终 sm90/sm100 **都用 16B 的 `int4`**。

它分两条路径，判据在 `:67-70`：

```
const bool enable_hadd_bypass =
    (bias_0 == nullptr and bias_1 == nullptr) and
    (kNumValidTopk <= 2 or topk_slot_idx[2] < 0);
```

- **bypass 路径**（`:73-109`）：有效槽 ≤2 且无 bias 时，直接读 slot0、slot1 两块 bf16 向量，
  用 `bf162_view_0 += bf162_view_1`（`:104-105`）**在 bf16 里相加**再写回。省掉「bf16→fp32→bf16」两次转换。
- **通用路径**（`:110-171`）：先把 bias 用 `ptx::accumulate`（`ptx.cuh:593-602`）累进 `float2`
  累加器，再逐槽 bf16→fp32 累加（`:126-140`），最后**一次性** `__float22bfloat162_rn` 转回 bf16（`:147-153`）。

**两个路径的数值关系（本机实测坐实）**：对 **≤2 个 bf16 操作数**，`bf16_add(a,b)` 等于
`round_bf16(float(a)+float(b))`（两个 bf16 之和在 fp32 里精确），所以 bypass 与通用**逐位一致**
（20 万随机值对 **0 处不同**）；而对 **≥3 个槽**，链式 bf16 加会**累积两次舍入**，
20 万随机三元组里 **22.10%** 与「fp32 累加后一次舍入」不同——这正是 `enable_hadd_bypass` 把上界卡在
2 的原因。

`ptx::accumulate` 本身也带架构分支：sm100 用一条融合 `add.rn.f32.bf16`（`ptx.cuh:593-597`），
sm90 回退成 `__bfloat1622float2` 再两次 fp32 加（`:599-601`）——**语义相同、指令数不同**。

## 6. 两个调用点：本地归约与跨 rank 归约

`combine_reduce` 被**同一段模板**复用在两处，靠 `get_src_buffer_ptr_func` 区分：

| 调用点 | 源指针函数 | 槽含义 | 作用 |
|---|---|---|---|
| 主 combine 本地归约 `combine.cuh:158` | `advance_ptr<vec_t>(x, slot*kHiddenBytes)`（本地激活 `x`） | topk 槽 | 同 rank 的多个专家输出先加成一个 |
| 归约 epilogue `combine_reduce_epilogue.cuh:101` | `comm_buffer.get_rank_buffer(slot).get_token_buffer(t)` | rank 或 topk 槽 | 跨 rank 求和到最终 `combined_x` |

主 kernel 里 `combine_reduce` 的 `kNumExpectedTopk = constexpr_ceil_div(kNumTopk, kNumRanks)`（`combine.cuh:158`
的第三个模板实参）——**每个 token 在一个 rank 上最多 ceil(topk/ranks) 个专家**；epilogue 里则是
`kNumTokensInLayout`（`:101`）。两者都满足「只读有效槽、`-1` 谓词跳过」的约定。

## 7. 另外两件事：bias 与 top-k 权重的回写

- **bias**（`combine_reduce_epilogue.cuh:54-55, 111-114`）：`bias_0`/`bias_1` 各自按
  `[num_combined_tokens, hidden]` 的 token buffer 取址，作为 `combine_reduce` 的额外加项。这正是
  README `:279` 说的「在 combine epilogue 里加共享专家输出或残差」。**bias 非空会强制走通用路径**
  （`:68-70` 的左半），因此有了 bias 就不再享受 bf16 直加旁路——数值上更稳。
- **top-k 权重回写**（`:127-141`）：`combined_topk_weights[t][lane]` 从
  `comm_buffer.get_rank_buffer(kUseRankLayout ? rank : master_lane_idx).get_token_buffer(t).get_topk_weights_ptr()[lane]`
  取值，无效槽写 0。注意 `master_lane_idx = get_master_lane_idx(match(stored_dst_rank_idx))`（`:129`）：
  topk-layout 下用「同 rank 组里最高 lane」当槽索引，与 `compute_topk_slots` 的取法一致。

## 8. H100 实测

`harness/exp_deepep_combine_reduce.{sh,py,out.txt}`（真机 `NVIDIA H100 80GB HBM3, sm_90, CUDA 13020`）
**16 PASS / 0 FAIL**：

- **Part 0（host constexpr）**：`use_rank_layout` / `get_num_tokens_in_layout` 13 组表全对；用
  `static_assert` 编译期坐实；`get_max_unroll_factor<64,4>=2 / <256,4>=4 / <512,4>=4 / <32,4>=1`；
  sm90 `CombineVecTraits<int4>` = 16B。
- **Part 1（直接调用真实 `combine_reduce`）**：bypass（2 槽）vs bf16 折叠 oracle **0/256**；
  通用（4 槽）vs fp32 累加一次舍入 oracle **0/256**；有 bias 强制通用路径 **0/256**；
  200k 随机 bf16 值对 bypass==通用 **0 处不同**；200k 随机三元组链式 bf16 vs 一次舍入
  **44195 处不同（22.10%）**。
- **Part 2（完整 `combine_reduce_epilogue_impl`，真实对称 recv_buffer）**：
  - 配置 A **rank-layout**（1 scaleout × 4 scaleup，experts=32，topk=4，ranks=4）：覆盖
    「同 rank 重复（ranks `0,0,2,3`→只算 `{0,2,3}`）」「全同 rank」「互异」「单 rank」「两两重复」
    五类 token，输出 vs oracle **0/1280**、top-k 权重 **0/20**；加 bias 版 **0/1280**。
  - 配置 B **topk-layout**（1 scaleout × 8 scaleup，experts=64，topk=4，ranks=8>topk）：
    输出 vs oracle **0/1280**、权重 **0/20**。
- **Python 独立 oracle**（`harness/exp_deepep_combine_reduce.py`）：用纯 Python 重写 constexpr 公式 +
  `__ffs` 去重选择逻辑，对同两套配置逐 token 给出 `slots` / `distinct_ranks`，**ALL PASS**，
  与 CUDA 侧互为独立实现。例如 topk-layout 下 ranks `0,0,1,2` 的 master lane 是 `{0:1,1:2,2:3}`，
  slots = `[1,2,3,-1]`。

## 9. 与 Mega MoE combine 的对照

DeepGEMM 的 `sm100_fp8_fp4_mega_moe.cuh` 有**自己的** combine 段（`docs/11` §combine：L2 直写远端
combine buffer + grid-index 水位线 + fp32 chunk 归约）。两者的共同点是「fp32 累加 + 一次舍入」；
差异在于：

- **数学相同**：都是「按 top-k 权重把多个专家输出加回源 token」，且都按 rank 去重（一份贡献只加一次）。
- **实现不同**：DeepGEMM 把 combine 融进 mega-kernel，用 NVLink 对称缓冲的 `sym_buffer.map` 直写 +
  `st_rel_sys`/`ld_acq_sys` 做 readiness；DeepEP 把它拆成「主 kernel 搬运/本地归约」+「可延迟的归约
  epilogue」。DeepEP 这里多出的两个细节——**bias 槽**（承载共享专家/残差，对应 `docs/44` 的共享专家）
  与 **rank/topk 双布局**——正是「通用通信库」为了覆盖不同 EP 拓扑必须付的复杂度。
- **`int4` 向量 + `-1` 谓词跳过**：与 DeepGEMM `docs/11` 的 combine 逐块 TMA 拉取+fp32 累加是同一类
  「可展开的连续维搬运」思路，只是 DeepEP 用「按 token 一个 warp」而非「按 hidden chunk」。

## 10. 链接与边界

- DeepEP 仓库：https://github.com/deepseek-ai/DeepEP ；README `:16`（deferred EP epilogues / cached
  expanded layouts）、`:279`（combine `bias` 可加共享专家输出或残差）、`:285`（defer epilogue 与
  `async_with_compute_stream`）。
- DeepEP V2.5 release（本地 commit `93eb6eb` 所属）：https://github.com/deepseek-ai/DeepEP/releases
- 边界：本篇只覆盖 combine 的**归约侧**；dispatch 的 notify/copy 与 symmetric memory 见 `docs/07`，
  EPLB 冗余专家的权重预取/梯度归约见 `docs/53`。主 combine kernel 的 NVLink/RDMA 双路径与
  `ncclGin`（`combine.cuh:100-106`）因本机 NCCL 2.29.7 不提供 GIN 设备 API 而**仅分析**（见「阻塞」）。
