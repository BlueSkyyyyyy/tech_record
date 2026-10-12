# Mega Gate 与 Mega mHC：MoE 相关的两个融合 mega-kernel

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的 Mega-kernel 篇（路线图 **P5-1 / P5-2**）。
> 前四篇（`docs/02`~`docs/11`）把主体 Mega MoE kernel 逐段读完了；本篇换两个外围算子：
>
> - **第一部分（P5-1）Mega Gate**——把「router 打分 GEMM（`x @ W_gate^T`）」与「MoE top-k 路由
>   （softplus→sqrt、加 bias 排名、选 top-k、归一化权重、物理专家映射）」融合进**一个 kernel**。
> - **第二部分（P5-2）Mega mHC**——把「HyperConnection（HC）的残差多流混合 + mix 系数投影 +
>   Sinkhorn 组合矩阵 + RMSNorm（+ 可选 FP8 量化）」融合进**一个 kernel**。
>
> 参考实现 `~/github/DeepGEMM`（公开版 commit `057ca59`）：
> Mega Gate：`csrc/apis/mega_gate.hpp`、`impls/sm100_bf16_mega_gate.cuh`、`layout/mega_gate.cuh`、
> `scheduler/mega_gate.cuh`、`epilogue/sm100_mega_gate.cuh`、`heuristics/mega_gate.hpp`、`tests/test_mega_gate.py`；
> Mega mHC：`csrc/apis/mega_mhc.hpp`、`impls/sm100_mega_mhc.cuh`、`layout/mega_mhc.cuh`、
> `scheduler/mega_mhc.cuh`、`epilogue/sm100_mega_mhc.cuh`、`tests/test_mega_mhc.py`。
>
> **约束提醒**：两个算子都是 **SM100 专属，仅分析**——`mega_gate.hpp:158-159` 与
> `mega_mhc.hpp:223` 都硬断言 `arch_major == 10`，H100(sm90) 端到端跑不了。但其中**架构无关**的
> 数值语义可在 H100 上逐行移植并对拍，本文给出真实数字：
> Mega Gate 见 `src/mega_gate_routing.cu` + `harness/exp_mega_gate_routing.{sh,out.txt}`；
> Mega mHC 见 `src/mega_mhc_semantics.cu` + `harness/exp_mega_mhc_semantics.{sh,out.txt}`。

## 1. 它融合了什么：从「GEMM + softmax + topk」到单 kernel

MoE 的 router 本来是这样一段独立算子链（以目标卡 DeepSeek-V3 系为例）：

```
logits = x @ W_gate^T                # [T,H] x [E,H] -> [T,E] 打分 GEMM
score  = sqrt(softplus(logits))      # 或 sigmoid，逐元素
rank   = score + bias                # 可选 per-expert 偏置只影响排名
topk   = argmax_k(rank)              # 选 top-k 专家
weight = score[topk] / sum(score[topk]) * routed_scaling_factor
```

朴素实现要读写 `[T,E]` 的 logits 多次（GEMM 写 global、softmax 读写、topk 读写），**memory-bound**。
Mega Gate 把整条链压进一个 kernel：**GEMM 的累加器直接写 TMEM，让专门的 gate warp 从 TMEM 读回打分**
并原地完成 softplus/sqrt、加 bias、top-k、归一化、映射，最后只把 `[T,K]` 的索引与权重写回 global。
省掉了 `[T,E]` 中间量的一次全量往返，也省掉了多次 kernel launch。

DeepGEMM README 在 2026.09.10 的 release 里把它和 Sparse Indexer、Mega mHC 一起列为新算子
（`README.md:14`，对应 [PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432)）：

> "2026.09.10: Sparse Indexer, Mega Gate, Mega mHC, DeepJIT, MoE and Indexer optimizations and more."

host 侧 API 只有两个输出（`mega_gate.hpp:51-52`）：`topk_idx [T,K']`（int64）与
`topk_weights [T,K']`（fp32），`K' = K + S`（`S = num_shared_experts`，仅当
`use_shared_as_routed` 时非零，`mega_gate.hpp:55`）。

## 2. API 与契约（`csrc/apis/mega_gate.hpp`）

### 2.1 输入/输出

| 张量 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `x` | `[T,H]` | BF16 | token 激活；`H % 256 == 0`（`:80,83`） |
| `weight` | `[E,H]` | BF16 | router 权重；`E % 4 == 0`（`:84-85`） |
| `num_topk` | int | | 路由专家数 K |
| `use_shared_as_routed` | bool | | 把 S 个共享专家也当路由专家一起选 |
| `num_shared_experts` | int | | S，仅 1 或 2（`:103`） |
| `routed_scaling_factor` | float | | 归一化权重再乘的缩放（`:98` 要求有限） |
| `ep_rank` | int | | 用于重复专家的错位分散 |
| `scoring_func` | str | | **只支持 `"sqrtsoftplus"`**（`:99`） |
| `mask` | `[T]` bool | | 非激活 token，输出 -1 / 0 |
| `bias` / `image_bias` | `[E]` fp32 | | 排名偏置（分别对所有 token / 图像 token） |
| `image_token_mask` | `[T]` bool | | 与 `image_bias` 必须同时给（`:116`） |
| `fix_routing_mask` | `[T]` bool | | 这些 token 用 `unmapped_topk_idx` 里给定的专家 |
| `to_physical_map` | `[E+S,dup]` int32 | | logical→physical 专家映射（含重复） |
| `logical_count` | `[E+S]` int32 | | 每个 logical 专家的物理副本数 |
| `unmapped_topk_idx` | `[T,K]` int64（行可 strided） | | 接收 logical top-k；`fix_routing_mask` 时也作为输入 |
| `force_random` | `[T]` bool | | 这些 token 随机选 K 个不同专家 |
| `out` | `([T,K'] int64,[T,K'] fp32)` | | 复用外部缓冲区（可选） |

关键的**语义备注**在 `mega_gate.hpp:54-59` 写得很清楚：

> The kernel ranks experts on **score + bias**, while the emitted weights are the **unbiased scores**
> normalized over the top-k sum and scaled by `routed_scaling_factor`.

即「**排名用带 bias 的分，权重用不带 bias 的分**」。这是 DeepSeek router 的设计：bias 只用来做负载均衡的
排名扰动，不改变实际门控权重。第 6 节会详细讲。

### 2.2 几处硬性契约

- `num_routed_experts <= 512` 且 `% 4 == 0`（`:80,84-85`）；`num_physical_topk <= 32`（`:109`）。
- `num_tokens <= kNumMaxTokens = 2^20`（`:156`，`layout/mega_gate.cuh:19`）。
- `arch_major == 10`（`:158-159`）——**SM100 专属**。
- `to_physical_map` 与 `logical_count` 必须同时给（`:122`）；`image_bias` 与 `image_token_mask` 必须同时给（`:116`）；
  `fix_routing_mask` 要求 `unmapped_topk_idx`（`:137-139`）。
- `num_tokens == 0` 提前返回（`:154-155`）。

### 2.3 score barrier 的宿主侧缓存（`mega_gate.hpp:13-26`）

跨 CTA 的 score 归约需要一块 global 的 `score_barriers`，按 stream 缓存并**在 capture 前 warmup**，
避免 CUDA graph 把一次性的清零动作也 replay（`:20` 的断言）。这是「kernel 跨 CTA 做 barrier 但又不想要
graph replay 副作用」的常见处理。

## 3. 物理坐标与任务分解

### 3.1 kernel 的编译期参数（`impls/sm100_bf16_mega_gate.cuh:25-31`）

```
sm100_bf16_mega_gate_impl<
    SHAPE_K,          // = hidden
    kNumRoutedExperts,
    kNumGateThreads,  // gate warpgroup 线程数
    kNumMmaCtas,      // 1 或 2（CTA pair）
    kNumSplitK,       // K 方向切分
    kNumExpertGroups, // 专家分几组
    kNumTopk,
    kHasPaddedExperts, kHasBias, kHasImageTokenMask, kHasMask,
    kHasFixRoutingMask, kHasForceRandom, kHasPhysicalMap, kHasUnmappedTopkIdx>
```

注意 **`num_tokens` 不在编译期**——kernel 变体只通过 gate warpgroup 数量间接反映 token 量
（`jit_kernels/impls/sm100_bf16_mega_gate.hpp:68`）。这样形状多变的 decode/prefill 复用同一份 binary。

### 3.2 任务分解：一个 token block 由 `num_logical_ctas` 个 CTA 协作

`heuristics/mega_gate.hpp:21` 的注释给出核心模型：

> a token block is computed by `num_logical_ctas = num_expert_groups * num_mma_ctas * num_split_k` CTAs

- **专家分组 `num_expert_groups`**：把 E 个专家按 128 一组切开（`kExpertAlignment=128`，
  `layout:15`）。每个专家组是一个 **UMMA M**：`UMMA_M = kNumExpertsPerGroup = num_aligned_experts /
  num_expert_groups`（`impls:50,53`），单 CTA 时 `UMMA_M==128`，CTA pair 时可为 256（`impls:59-60`）。
- **CTA pair `num_mma_ctas`**：2-CTA 时 `UMMA_M=256`，一条 UMMA 覆盖 256 个专家（`impls:53-58`），
  两个 CTA 在专家维（M）和 token 维（N）上都对半劈。
- **split-K `num_split_k`**：hidden 维切分，`num_split_k * BLOCK_K` 必须整除 hidden（`heuristics:127`）。
  切分产生的部分分在 gate warp 里累加（`impls:304-316`），因此 **sqrt-softplus 不能逐 split 做**，
  必须等部分分加完再做（`impls:284-286`）——这是第 5 节的关键点。

UMMA 的 N 方向是 **token 维**：`UMMA_N = kNumMaxBlockTokens = 256`（`impls:52,54`），
每次进循环按 `get_effective_umma_n`（`impls:124-126`，16 对齐）动态缩小。`BLOCK_K = 64`（bf16，
`layout:10`），`kSwizzleMode = 128`。

### 3.3 启发式选型（`heuristics/mega_gate.hpp:106-187`）

穷举 `(num_expert_groups, num_mma_ctas, num_split_k)` 候选，过滤条件：
- 专家组大小不超过 `2*128`（`:119`）；专家组 >128 时必须 2 CTA（`:123`）；
- `num_split_k ∈ {1,2,4,8}`，且 `hidden % (split*64)==0`（`:126-128`）；
- `num_logical_ctas <= num_sms` 且 `< 64`（`:134`，对齐 `kNumMaxLogicalCtas`）——**一个 token block 的
  所有逻辑 CTA 必须同时驻留**（否则 score barrier 会死锁）。

排序偏好（`compare_mega_gate:89-100`）：更长的 TMEM 转置（`num_token_cols` 大）→ 用更多 SM →
优先 CTA pair → split 少 → 波数少。另有 `is_task_doubling_free`（`:74-87`）避免「翻倍逻辑 CTA 后波数暴涨」
的候选：单 CTA 可翻成 CTA pair，但 single-wave 的 CTA pair 只能靠切 K 翻倍（确定性模式禁止切 K，`:84`）。

Gate warpgroup 数（`:159-166`）：由「TMEM 转置单元数」与「top-k 轮数」两者平衡，上限 7 个
（`kNumMaxGateWarpgroups=7`，`:51`）；多波时直接用满 7 个。每 warpgroup 每轮转置 8 个 TMEM 列、
每轮排 4 个 token（`:49-53`）。

## 4. 数据通路：logits 如何从 MMA 累加器流到 gate warp

这是 Mega Gate 与普通 GEMM 最大的不同：**MMA 的累加器不落 global，而是被 gate warp 从 TMEM 读走**。

1. **MMA 只在 leader CTA 的 warp 1 发**（`impls:168`，`:213-226`），`A=weight`、`B=x`（`impls:222` 注释
   *"the weights are the A operand, so a TMEM lane holds one expert and a column one token"*）。
   累加器放在 `accum_stage_idx * UMMA_N` 处（`:223`），`kNumEpilogueStages=2` 双缓冲。
2. **UMMA 完成时 multicast 通知** `tmem_full_barriers`（`:199-203`）与 `empty_barriers`（`:200`）。
3. **gate warp（warp >= 4）等 `tmem_full`（`impls:279`），从 TMEM 读打分**：
   `sm100_store_tmem_scores_to_global`（`epilogue:39-78`）用
   `ptx::tmem_load_32dp32b<8>`（`epilogue:64`；实现在 `ptx/tcgen05.cuh:191-205`，
   展开成 `SM100_TMEM_LOAD_32dp32b{4,8,...}x`）每次读 8 行，直接写进 global 的 **score scratch**。
4. **split-K 时 score 先落 global、跨 CTA 用 `ScoreBarrier` 汇聚**（`impls:295-303`）：
   每个逻辑 CTA 写自己那一份 split 的部分分，然后 `ScoreBarrier::arrive`；等所有逻辑 CTA 到齐
   （`ScoreBarrier::wait`，`scheduler/mega_gate.cuh:35-38`）后，每个 CTA 才能把各 split 加起来
   （`load_scores`，`impls:304-316`）读到完整 logits。
5. **最后一轮 gate warp `release_tmem`**（`:296-297` 提前 `arrive tmem_empty`），让 MMA 可以复用该 TMEM 列。

score scratch 的布局是 `[num_token_blocks * block_tokens, num_splits, num_experts]`
（`layout/mega_gate.cuh:78-102`，`kTokenStride = num_splits*num_experts`）。

> **为什么这么绕**（TMEM→global→再加回）？因为 split-K 把 K 维切开后，部分分分散在不同 CTA 的 TMEM 里，
> 只有落在同一个 token block 的全局 scratch 上、用跨 CTA barrier 才能汇聚。`kNumSplitK==1` 时
> 则可以省掉这次 global 往返（`kScoreOnStore`，`impls:286`）——把 sqrt-softplus 直接放在 TMEM→global 的
> 写出环节（`epilogue:71-72`），排名时再读回被选中的那个分（`impls:412-415`）。

## 5. 打分函数：为什么是 `sqrt(softplus(x))`，为什么有阈值 20

`epilogue/sm100_mega_gate.cuh:19-23`：

```cpp
CUTLASS_DEVICE float sqrt_softplus(const float& value) {
    const auto softplus_precise = log1pf(expf(value));
    const auto softplus = value > 20.0f ? value : softplus_precise;
    return sqrtf(softplus);
}
```

- softplus `log(1+e^x)` 是 DeepSeek 系 router 在 sigmoid 之外用的平滑激活，恒正、单调、可导。
  开根号 `sqrt` 是为了进一步压小大值、放大区分度（`scoring_func` 只允许 `"sqrtsoftplus"`，`mega_gate.hpp:99`）。
- **阈值 20**：当 `x > 20` 时 `e^x` 已 > 4.8e8，而 `log1p(e^x) ≈ x + e^{-x}` 与 `x` 的相对差 < 2e-9，
  直接用 `x` 既避开 `expf` 的 overflow 风险、又省一次 `exp`。对排名与权重的影响远小于 fp32 精度。
- 向量版（`epilogue:25-37`）先批量算 `exp` 再批量 `log1p`，最后逐元素选分支——为了让编译器向量化发射
  （`log1pf`/`expf` 交错会妨碍）。

**H100 实测**（`harness/exp_mega_gate_routing.out.txt`，4194304 个 `x~U[-60,60]`）：

```
max_abs_err vs double = 2.518e-07, max_rel_err = 1.426e-07
x>20 (走阈值分支, 返回 sqrt(x)) : 1397954 / 4194304 (33.3%)
阈值探针: -60->0.000000  0->0.832555  19.9999->4.472125  20.0000->4.472136
          20.0001->4.472147  40->6.324555  60->7.745967
```

即阈值分支在临界点连续（`sqrt(20)=4.472136`，两侧 `4.472125/4.472147` 平滑过渡），
整段相对误差 ~1.4e-7（纯 fp32 舍入），证明这个分支对数值没有破坏。

## 6. 排名与权重的分离：带 bias 排名、不带 bias 出权重

`epilogue:187-216` 的 `store_topk_token` 是权重的出口：

```cpp
const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;
const auto selected_weight = lane_idx < kNumTopk
    ? selected_unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
```

而排名用的是另一套分（`impls:379-399`）：

```cpp
bias_ptr = ...smem_bias...;                       // 或图像 token 用 image_bias
scores_local[cell+v] = is_valid_expert
    ? unbiased_score + bias_local[v] : -inf;      // 排名 = 无偏分 + bias
```

- **无偏分 `unbiased_score`** = `sqrt_softplus(logit)`，存进 `unbiased_scores_local`（`impls:396`），
  只用于被选中后的权重。于是「专家 e 的物理负载均衡 bias」改变它是否进 top-k，但不改变它的门控大小。
- **`+1e-20f`**：防止全零行（例如强制随机或全 `-inf`）除零。
- **共享专家槽**：`lane_idx >= kNumTopk` 的 lane 填 `selected_expert_idx = lane + num_routed - K`
  （`epilogue:194-195`），权重固定为 `1.0`（`:213`）。测试 `tests/test_mega_gate.py:164-165` 明确断言
  `fused_weights[:, num_topk:] == 1.0`。
- **归一化宽度**：`kSumWidth = K<=8 ? 8 : (K<=16 ? 16 : 32)`（`epilogue:211`），只在覆盖了 top-k 槽的
  warp 子段内做 `shfl` 归约，其余 lane 记 0。

测试里的参考权重（`tests/test_mega_gate.py:137-139`）与此完全一致：

```python
topk_scores = scores[row_mask].gather(1, topk_idx)
return topk_scores / (topk_scores.sum(dim=1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR
```

## 7. warp 级 top-k 选择算法（本文算法的核心）

每个 gate warp 负责一个 token 的「全部专家打分排名」。专家的 lane 布局（`layout:26-29`）：

- 每 lane 持 **4 个连续专家**（`kNumExpertsPerLaneVector=4`）；
- 每 **wave 覆盖 128 个专家**（`kNumExpertsPerWave = 32*4 = 128`，且 `== kExpertAlignment`）；
- `kNumExpertWaves = E / 128`。

选择分三步（`epilogue:107-152`）：

1. **每 lane 内 4 元素排序网络 `sort4_best_first`**（`:88-105`）：5 次 compare-swap 把 lane 的 4 个专家按
   「分降序、同分专家号升序」排好，返回一个 8-bit 的 `permutations`（每 2 bit 记录一个位置的原始槽号）。
2. **3-bit 游标归并**：每个 wave 维护一个 3-bit 游标 `packed_cursors`（0..4，4 表示该 wave 已耗尽）。
   每轮每个 lane 从每个 wave 取「当前游标对应的最佳未消费候选」，warp 内选全局最好的一条
   （`warp_reduce_best`），只有胜出 lane 推进该 wave 的游标（`:147-148`）。重复 K 轮得到 top-k。
3. **破平 = 最小专家号**（`warp_reduce_best`，`:80-86`）：

```cpp
const auto best_score = reduce_max_sync(score);                 // redux.sync.max.f32
const auto tied_expert_idx = score==best_score and expert_idx>=0
    ? (uint32_t)expert_idx : 0xffffffffu;
expert_idx = (int)__reduce_min_sync(0xffffffff, tied_expert_idx); // 平局取最小号
```

**为什么不用 `torch.topk` 式的排序？** 因为这里是 warp 内 32 lane × 4 值的寄存器驻留数据，
排序网络 + 游标归并是**无 shared memory、无 bank conflict** 的纯 warp 内实现，且**完全确定性**
（浮点比较是精确的，破平规则确定），满足「同一输入每次 launch 逐位相同」的要求——
`tests/test_mega_gate.py:185-191` 正是断言重复 10 次输出 `assert_bitwise_equal`。

**H100 对拍**（`src/mega_gate_routing.cu` 逐行移植该算法，参考实现按 `(ranking desc, expert_idx asc)`
稳定排序），`harness/exp_mega_gate_routing.out.txt`：

```
A random no-bias      (waves=1,K=6,E=128,T=4096) : idx 不符 0/24576, 权重 rel 1.242e-07
B random with-bias    (waves=1,K=8,E=128,T=4096) : idx 不符 0/32768, 权重 rel 8.021e-08
C heavy-ties          (waves=1,K=6,E=128,T=4096) : idx 不符 0/24576, 权重 rel 2.980e-08
D 3-waves with-bias   (waves=3,K=6,E=384,T=1024) : idx 不符 0/6144,  权重 rel 6.425e-08
E 4-waves padded(300) (waves=4,K=6,E=512,T=1024) : idx 不符 0/6144,  权重 rel 1.501e-07
```

案例 C 用 `logits ∈ {0,1,2}` 制造**大量并列**，验证破平规则与参考（最小专家号）一致；
案例 E 把 512 个 wave 槽里只填 300 个真实专家、其余 `-inf`，验证 padding 专家永不入选
（对应 `impls:397-398` 的 `-inf` 遮罩）。全部 0 不符、权重误差仅 fp32 舍入级。

## 8. 重复专家（physical map）与 shared-as-routed

### 8.1 logical → physical 映射

目标卡推理时会把热门 logical 专家**复制**成多个 physical 副本（vLLM 风格的额外专家，测试里
`VLLM_NUM_EXTRA_EXPERTS=32`，`tests:19`），以摊薄热点。`store_topk_token`（`epilogue:198-208`）做映射：

```cpp
const auto num_duplicates = routed_logical_count[logical_expert_idx];   // 该 logical 的副本数
const auto duplicate_idx = (args.rank_idx + global_token_idx * 23333u) % num_duplicates;
physical_expert_idx = to_physical_map[logical_expert_idx * num_duplicate_experts + duplicate_idx];
```

- **`23333` 是个大质数**，用「rank + token*质数 mod 副本数」把同一逻辑专家的不同 token
  **确定性地**分散到不同物理副本；不同 rank 起点不同，进一步错开，避免所有 rank 都挤同一副本。
- **H100 实测**（同一 out 文件）：对 `d ∈ {1,2,4,32}`、8 个 rank、65536 token 统计每个副本分到的 token 数，
  `min == max == 65536/d`、**最大/最小 = 1.0000**——完全均衡（`23333 mod 32 = 5`，与 32 互质，
  步进遍历所有副本）。

### 8.2 shared expert 作为 routing

`use_shared_as_routed` 时（`mega_gate.hpp:101-108`），把 S 个共享专家也并进 top-k 一起选，
但它们的槽固定在最后（`epilogue:194-195`）且权重恒为 1（`:213`）。
约束：`num_shared_experts ∈ {1,2}`，`K % S == 0`，`E % (K/S) == 0`（`:103-105`）——
保证共享专家能整除地分配进路由槽。

## 9. 路由控制的四个开关

`impls:326-358` 的 per-token 分支按顺序短路：

1. **mask**（`:326,331-335`）：非激活 token，全部输出 `-1 / 0`。
2. **force_random**（`:327,336-340`）：`store_random_topk_token`（`epilogue:218-253`）用
   **curand Philox** 随机抽 K 个**互不相同**的专家。算法是每 lane 画一个候选、然后「比它小的已抽中的
   候选号整体上移一位」的**有序无放回抽样**（`:239-250`），保证 K 个不同。
3. **fix_routing_mask**（`:328,341-358`）：从 `unmapped_topk_idx` 读入已定专家，只读取其无偏分参与权重，
   不重排。测试 `tests:147` 断言这些行输出与输入一致。
4. **正常排名**（`:359-427`）：第 5~7 节的流程。

`image_bias` 通过 `image_token_mask` 选择 rank 用的 bias（`impls:382-383`），实现「图像 token 用另一套
均衡偏置」。

## 10. 跨 CTA 的 score barrier 与确定性

split-K 与 CTA pair 让一个 token block 的打分分散在多个 CTA。`scheduler/mega_gate.cuh:13-39` 的
`ScoreBarrier` 做汇聚：

- 状态一次原子地编码了「grid 序号 + 已到达 CTA 数」：`encode_state(grid_idx, arrived)`（`:17-19`）。
- `wait_init`（`:25-29`）用「无符号距离」拒绝**上一个 grid 的陈旧标签**，但接受本 grid 的任意到达计数——
  这样即便 barrier 内存被复用、上一轮残留，也不会误判。
- `arrive` 用 `red_async_inc_rel`（`:31-33`），`wait` 用 `ld_acq_gpu` 轮询到
  `encode_state(grid_idx, kNumLogicalCtas)`（`:35-38`）。
- kernel 里由每个 block 的第一个逻辑 CTA **重新 arm**（`:271-276`），其余 CTA `wait_init`。

这份 barrier 与主体 Mega MoE 的 `grid_sync`/`nvlink_barrier`（`comm/barrier.cuh`）思路一致（加减式相位、
免复位），只是作用域是「一个 token block 内的逻辑 CTA」，而非跨 SM/rank。

**确定性**：`use_deterministic_algorithms(True)` 时启发式强制 `num_split_k==1`（`heuristics:110-111,126`）——
不切 K，就没有跨 split 的浮点加序问题，`tests:307-313` 专门验证了确定性模式下的 batch invariance。
rank 排序本身也是确定性的（精确比较 + 固定破平），所以即便不切 K，同一输入也逐位可复现。

## 11. 与主体 Mega MoE 的关系

Mega Gate 的产物正是主体 Mega MoE 的输入：`topk_idx`（physical 专家号）与 `topk_weights`（门控权重）
直接喂给 `sm100_fp8_fp4_mega_moe` 的 dispatch/combine（见 `docs/11`）。两者共享同一套设计语言：

| | Mega Gate | Mega MoE 主体 |
|---|---|---|
| 融合内容 | 打分 GEMM + softplus/sqrt + top-k + 映射 | dispatch + linear1 + SwiGLU + linear2 + combine |
| UMMA 累加器去向 | 写 TMEM，gate warp 读走（不打分写 global） | 写 TMEM，epilogue 做 SwiGLU/量化 |
| 跨 CTA 同步 | `ScoreBarrier`（一个 token block 内） | `grid_sync` / `nvlink_barrier`（跨 SM/rank） |
| 架构 | **SM100 专属** | **SM100 专属** |
| H100 可做 | 路由数值语义移植对拍（本文） | epilogue 数值原语对拍（`docs/10`） |

## 12. Mega Gate 小结

- Mega Gate = **BF16 打分 GEMM + router 全链路**的单 kernel 融合，省掉 `[T,E]` 中间量的 global 往返。
- 三个非平凡设计点：**① 累加器经 TMEM 直接喂给 gate warp**（避免打分落 global）；②
  **带 bias 排名、不带 bias 出权重**（bias 只做负载均衡）；③ **warp 内排序网络 + 3-bit 游标归并**的
  确定性 top-k（破平取最小专家号）。
- H100 上无法跑端到端（`arch_major==10` 硬断言），但把架构无关的路由语义逐行移植后与参考
  完全一致（5 个案例、约 7 万条 top-k 全 0 不符），`sqrt_softplus` 阈值分支、重复专家分散公式
  也都有真实数字佐证。

---

# 第二部分：Mega mHC —— HyperConnection 融合 kernel（P5-2）

> README 把 **Mega mHC** 与 Sparse Indexer、Mega Gate 一起列在 2026.09.10 release
> （`README.md:14`，对应 [PR #432](https://github.com/deepseek-ai/DeepGEMM/pull/432)）。
> 它服务的不是 MoE 本身，而是**网络主干**：DeepSeek 的 HyperConnection（HC）把每一层的残差
> 从单流扩成 `hc_mult` 条并行流，并让每条流按**逐 token 学习的混合系数**重新组合。mHC 就是
> 这个模块的融合算子：**残差多流混合 + mix 系数投影 + Sinkhorn 组合矩阵 + RMSNorm（+ 可选 FP8）**
> 全压进一个 kernel。参考：`csrc/apis/mega_mhc.hpp`、`impls/sm100_mega_mhc.cuh`、
> `layout/mega_mhc.cuh`、`scheduler/mega_mhc.cuh`、`epilogue/sm100_mega_mhc.cuh`、
> `tests/test_mega_mhc.py`。

## 13. 计算图：以 `tests/test_mega_mhc.py` 的参考实现为权威

kernel 的语义在测试里的 `mhc_reference`（`tests/test_mega_mhc.py:123-210`）写得最清楚，逐行读：

**① 残差多流混合**（`:145-148`）：

```
new_residual[t,j,:] = post_mix[t,j] * x[t,:] + Σ_i comb_res_mix[t,j,i] * residual[t,i,:]
new_residual = bf16(new_residual)          # 舍入到 bf16
```

即第 `j` 条新残差流 = 「新激活 `x` 按 `post_mix` 缩放」+「旧 `hc_mult` 条流按 `comb_res_mix[t,j,:]`
加权组合」。`comb_res_mix` 是逐 token 的 `hc x hc` 矩阵。**这里 bf16 舍入是一次共享的数值边界**
（kernel 侧 `epilogue:496-498` 的 `NOTES`）：之后所有消费者（mix 投影、RMSNorm、下一层）都读这个
bf16 值。

**② mix 系数投影 + 归一化 + 仿射**（`:155-170`）：

```
mixes = (new_residual.flatten(1) @ fn.mT) * rsqrt(Σ new_residual²/(hc*H) + hc_norm_eps) * scales + bases
new_prev_mix  = sigmoid(mixes[:, :hc])              + hc_pre_eps
new_post_mix  = sigmoid(mixes[:, hc:2hc]) * hc_post_scale
new_comb_logit= mixes[:, 2hc:].view(hc, hc)
```

- `fn` 是 `[hc*(hc+2), hc*H]` 的**投影矩阵**：把 `hc*H` 维残差投到 `hc*(hc+2)` 个「混合系数」。
  这是本 kernel 里唯一的 GEMM（一个**很窄的输出、很长的 K** 的矩阵乘）。
- 投影结果先乘 `rsqrt(hc_sqr_sum/(hc*H)+eps)`（把残差整体尺度归一化掉），再逐段乘 `scales` 加 `bases`
  （`scales` 长度 3 分别给 pre/post/comb 三块，`:161-166`）。
- pre/post 系数经 sigmoid（pre 加 `hc_pre_eps`、post 乘 `hc_post_scale`）；comb 块是**原始 logits**，
  后面过 Sinkhorn。

**③ Sinkhorn 组合矩阵**（`_sinkhorn_reference`，`:114-120`）：

```
v = softmax(logits, dim=-1) + eps
v = v / (v.sum(dim=-2, keepdim=True) + eps)          # 列归一化
repeat (num_sinkhorn_iters-1) 次:
    v = v / (v.sum(-1) + eps)                         # 行归一化
    v = v / (v.sum(-2) + eps)                         # 列归一化
```

Sinkhorn 交替行/列归一化，把组合矩阵推向**双随机矩阵**（行和、列和都 → 1）。用双随机的 `comb`
做残差混合，等价于「**不带缩放的能量再分配**」——每层混合不放大也不缩小残差总能量，是训练稳定性的
关键。kernel 侧在 `epilogue:63-86` 用 warp 内归约实现（见 §18）。

**④ `norm_input` 与 RMSNorm**（`:149-184`）：

```
norm_input = Σ_j mix_j * new_residual[:, j, :]        # mix = shifted_prev_mix 或 new_prev_mix
norm_input = bf16(norm_input)
y_bf16 = bf16( norm_input * rsqrt(mean(norm_input²)+rmsnorm_eps) * (rmsnorm_weight*rmsnorm_scale) )
```

`y` 是喂给下一层注意力的归一化激活。注意 **RMSNorm 的均方是 `mean` over `H`**（除以 H），
与前面的 `hc_sqr_sum`（对所有 `hc*H` 求和）是**两个不同的统计量**。

### 13.1 normal vs shifted：两个变体的区别

kernel 有 `kIsShifted` 模板开关（`impls:38`）。差别只在**用哪套 pre 系数**以及**是否持久化**：

- **normal mHC**：用刚算出的 `new_prev_mix` 形成 `norm_input`（`:176-177`），**不输出** pre 系数。
- **shifted mHC**：输入 `shifted_prev_mix [T,hc,1]`，用它形成 `norm_input`（`:149-150`），并**输出**
  `new_prev_mix` 供下一层用。API 契约是「全有或全无」：`shifted_prev_mix` 与 `new_prev_mix` 要么都给、
  要么都不给（`mega_mhc.hpp:117-118`）。

这对应两种执行编排：shifted 需要把 `new_residual` 物化到 global 再被 Norm 阶段读回并算
`shifted_prev_mix·residual`；normal 则可以把 Mix→Pre→RMSNorm 融成一条 per-token 流水（§16）。

## 14. API 与契约（`csrc/apis/mega_mhc.hpp`）

host 入口 `mega_mhc(...)`（`:89-115`）的张量表（来自 `:41-88` 的 doc comment）：

| 张量 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `x` | `[T,H]` | BF16 | 本层新激活 |
| `residual` | `[T,hc_mult,H]` | BF16 | `hc_mult` 条旧残差流 |
| `shifted_prev_mix` | `[T,hc_mult,1]` | FP32 | shifted 的 pre 系数（normal 为 None） |
| `post_mix` | `[T,hc_mult,1]` | FP32 | 本层 post 系数 |
| `comb_res_mix` | `[T,hc_mult,hc_mult]` | FP32 | 组合矩阵 |
| `fn` | `[hc_mult*(hc_mult+2),hc_mult*H]` | FP32 | 投影矩阵 |
| `mix_scales` | `[3]` | FP32 | pre/post/comb 分段缩放 |
| `mix_bases` | `[hc_mult*(hc_mult+2)]` | FP32 | 分段偏置 |
| `rmsnorm_weight` | `[H]` | BF16 | RMSNorm 权重 |
| `new_residual` | `[T,hc_mult,H]` | BF16 | 输出：新残差流 |
| `new_prev_mix` | `[T,hc_mult,1]` | FP32 | 输出：shifted 的下一层 pre |
| `new_post_mix` / `new_comb_res_mix` | `[T,hc_mult,1]` / `[T,hc_mult,hc_mult]` | FP32 | 输出系数 |
| `y_bf16` / `y_fp8` | `[T,H]` | BF16 / E4M3 | 输出：归一化激活（至少给一个） |
| `y_gemm_sf` / `y_routed_sf` / `y_shared_sf` | `[T,H/128]` | int32 | FP8 的 UE8M0 SF（三种布局，§20） |

硬契约（`:116-158`）：

- `hc_mult == kNumRoutes == 4`（`:138`，`layout:11`）；`num_hc_outputs = hc*(hc+2) = 24`（`layout:12`）。
- `hidden % (kDefaultNumSplits*BLOCK_K) == 0` → **`hidden` 必须能被 `16*64 = 1024` 整除**（`:141`）。
- `num_tokens <= kNumMaxTokens = 2^20`（`:140`）；`num_sinkhorn_iters >= 1`（`:142`）。
- `arch_major == 10`（`:223`）——**SM100 专属**。
- `y_bf16` 与 `y_fp8` **至少给一个**（`:121`）；给了 `y_fp8` 就必须给 SF，且 `y_gemm_sf` 与
  `y_routed_sf` **互斥**（`:120,122`）。

`get_split_barriers`（`:26-39`）按 **CUDA stream** 缓存跨 CTA 的 split barrier，并在第一次使用前
断言**不在 graph capture 中**（`:33`）——和 Mega Gate 的 score barrier 一样，避免把一次性清零
replay 进 CUDA graph。

## 15. 物理坐标与任务分解

### 15.1 编译期几何（`layout/mega_mhc.cuh`）

```
BLOCK_M = 64, BLOCK_N = kNumHCOutputs = 24, BLOCK_K = 64        (:19-21)
kNumThreads = 6 * 4 * 32 = 768                                  (:31)   # 6 个 warpgroup
```

`hidden` 在 K 方向切成 `kNumSplits` 份，每份再切成 `BLOCK_K` 块；一个 **task = 一个 (m-block,
split)** 对（`impls:28`）。`kDefaultNumSplits=16`、`kNumMaxSplits=64`（`layout:36-37`）。

### 15.2 split 数启发式（`mega_mhc.hpp:15-24`）

```
num_k_blocks = hidden / BLOCK_K
max_num_splits = min(64, max(16, num_sms / num_m_blocks))
num_splits = ceil_div(num_k_blocks, ceil_div(num_k_blocks, max_num_splits))
```

即「**在不超过 `max_num_splits` 的前提下，用最少的 split 数**，让最长的 split task 尽量短」。
确定性模式直接固定 16（`:16-17`）。

## 16. kernel 结构：6 个 warpgroup 的静态角色 + split-K 归约

`sm100_mega_mhc_impl`（`impls:37-701`）是持久化 kernel（`grid = num_sms`，`layout` 里 768 线程）。
它把 6 个 warpgroup 硬绑成静态角色（`impls:78-85`）：

| WG | 角色 | 干什么 |
|---|---|---|
| WG0 | TMA/MMA 生产者 | warp0 载 `residual`+`x`；warp1 载 `fn`；warp2 发 UMMA；warp3 写 `new_residual`+预取系数 |
| WG1-2 | **Post workers** | 算 `new_residual = post*x + comb@residual`（bf16 舍入），写回 smem，累计统计量，把结果存进 TMEM 作为 UMMA 的 A 操作数 |
| WG3 | **workspace-epilogue** | 从 TMEM 读 UMMA 累加器，把 `[BLOCK_M,24]` 部分分写进 global scratch |
| WG4 | **Mix worker** | 归约 split 部分分 → 算 mix 系数（sigmoid + Sinkhorn）|
| WG5 | **Norm worker** | RMSNorm（shifted：读 global `new_residual`；normal：被复用作融合 worker）|

### 16.1 UMMA 是 **TF32**，A 在 TMEM、B 在 SMEM（`impls:233-285`）

```cpp
using umma_t = cute::SM100_MMA_TF32_TS<..., BLOCK_M, BLOCK_N, Major::K, Major::K>;   // :235-238
...
umma_t::fma(tmem_base + a_stage*kNumTmemColumnsPerAStage + route*UMMA_K,             // :273 A@TMEM
            b_desc,                                                                   //  B@SMEM(fn)
            tmem_base + kAccumTmemStartColumn + accum_stage*BLOCK_N,                  // 累加器@TMEM
            k_block_idx>0 or umma_k_idx>0 or route_idx>0, runtime_instr_desc);        // :275 首条清零
```

- 这是 **TS 型**（A 来自 Tensor Memory、B 来自 Shared Memory）的 TF32 UMMA。`A` = Post worker
  写进 TMEM 的 `new_residual`（BF16 舍入后的值，按 TF32 存）；`B` = TMA 载入的 `fn`（FP32）。
- **为什么用 TF32 而不是 BF16？** TF32 有 10 位尾数（BF16 只有 7 位）。`fn` 是 FP32 权重，投影
  系数直接进 sigmoid 和 Sinkhorn 的 logits，多 3 位尾数能显著减小系数误差；而 A 侧激活已在
  `new_residual` 处**强制 bf16 舍入**（`epilogue:496-498` 的共享边界），所以用 TF32 承载它不损失
  额外精度，却能让 GEMM 走 TF32 路径。
- **为什么 A/B 这么摆**？把「宽 K、窄 N=24」的 GEMM 的 A（残差，按 route 分块累积）放 TMEM，
  因为 Post worker 本来就在 TMEM 里写 A；B（`fn`）放 SMEM 由 TMA 流式载入。累加器也放 TMEM，
  由 WG3 用 `SM100_TMEM_LOAD_*` 读回（`impls:603-606`）。

### 16.2 Post worker：残差混合 + 统计 + A 存储（`impls:386-567`）

Post worker 是这套流水的心脏。每 task 它做（`impls:482-530`）：

```cpp
post_value = x * post_coeff;                                     // :488
for i: post_value = fma(residual[i], comb[i,j], post_value);     // :494 组合旧流
post_value_bf16 = bf16_round(post_value);                        // :497 共享数值边界
hc_norm_sqr_sums += post_value²;                                 // :501 Σnew_residual²
if shifted: x1 += post_value * pre_coeff;                        // :505 X1 = Σ pre·new_residual
SM100_TMEM_STORE_16dp256b4x::copy(..., tmem_addr);               // :512-515 写 TMEM 作为 UMMA 的 A
```

统计量（`hc_norm_sqr_sums`）随后经跨 WG barrier 汇聚，写进 `hc_norm_sqr_sum_partials`（`:634-642`），
这正是 mix 系数归一化用的那个 `hc_sqr_sum`。

### 16.3 WG3 workspace-epilogue：split-K 部分分落盘（`impls:569-655`）

每个 task 一份 `[BLOCK_M, 24]` 的 UMMA 结果，从 TMEM 读回后按 `task_idx = m_block*num_splits+split`
写进 global scratch（`:621-624`）。**为什么要落盘**？因为 split 把 K 切开后，同一个 token 的部分分
分散在不同 task（甚至不同 SM）的 TMEM 里，必须落到同一块 global scratch 才能被 Mix 阶段跨 task 相加。

## 17. Split-K scratch 与 Workspace 布局（`layout:73-120`）

```
gemm_partials            [num_m_blocks, kNumSplits, BLOCK_M, 24]
hc_norm_sqr_sum_partials [num_m_blocks, kNumSplits, BLOCK_M]
x1_sqr_sum_partials      [num_m_blocks, kNumSplits, BLOCK_M]
task_idx = m_block_idx * kNumSplits + k_split_idx
```

三块连续排布，`get_num_scratch_bytes`（`:98-101`）= `num_tasks * (24*64 + 2*64) * 4` 字节。
`x1_sqr_sum_partials` 只在 shifted 用（norm_input 的均方）。

## 18. Mix 与 Norm 两条消费者路径

### 18.1 Mix worker：warp 级仿射 + sigmoid + Sinkhorn（`epilogue:16-101`）

`run_mix_task` 一次处理**一个 token**：一个 warp，lane 布局是
`[Pre 4 | Post 4 | Comb 16 | inactive]`（`epilogue:45-49`）：

```cpp
// 归约 split 部分分（hc_sqr_sum 与 24 个 GEMM 输出）
hc_norm_sqr_sum = warp_reduce_sum(Σ_split ...);                               // :28-33
hc_output       = Σ_split gemm_partials[...];                                 // :37-41
hc_rms_scale    = rsqrt(hc_norm_sqr_sum * (1/(hc*H)) + hc_norm_eps);          // :43
// 每 lane 的仿射 + sigmoid
affine = hc_output * hc_rms_scale * scales[idx] + bases[lane];                // :53
pre  = sigmoid(affine) + hc_pre_eps;  post = sigmoid(affine)*hc_post_scale;   // :55-56
```

**Sinkhorn 的 warp 级实现**（`:63-86`）是本文最精巧的一段。comb 的 16 个 logits 按**行主序**放在
lane 8..23（4 行 × 4 列）：

- **行内归约**：`warp_reduce_max<4>` / `warp_reduce_sum<4>`（`math.cuh:203-220`）在**每 4 个相邻 lane**
  内做 `shfl_xor(1,2)`，正好是一行的 4 个元素 → 得到 `softmax` 的行 max/和。
- **列内归约**：`get_col_sum`（`:71-76`）用两次 `shfl_xor`：先 `xor kNumRoutes=4` 合并相邻两行，
  再 `xor 6*kNumRoutes=24` 合并上下半（lane 8-15 ↔ 16-23），两步得到**同一列 4 个元素的和**。
- 流程严格镜像参考实现：`softmax + eps` → 列归一化 → `(iters-1)` 轮 [行、列归一化]。

### 18.2 Normal：Mix+Pre+RMSNorm 融合成 per-token worker（`epilogue:139-281`）

normal 变体的 WG4-5 以及**退场后的生产者 WG**都变成 `run_normal_worker`（`impls:668-682`）。
每个 task 由 `atomicAdd(&smem.next_norm_task_ticket,1)`（`:165-167`）动态认领一个 token，
`token_idx = blockIdx.x + ticket*kNumSMs`（`:168`）。它在一个 warpgroup 内：

1. 算 Pre 系数（`run_mix_task` 的 Pre 部分，`:172-177`），通过 named barrier 广播（`:179`）；
2. 从 global `new_residual` 读回 4 条流，用 `pre_coeff` 加权成 `X1`（`:196-210`），**累加在 FP32、
   再舍入到 BF16**（`:206-209`）；
3. 归约得 RMS scale（`:213-220`）；
4. `normalize_bf16_pair` 写 `y_bf16`（`:244`）；若开 FP8 则算 amax→UE8M0→写 fp8+SF（`:245-259`）。

### 18.3 Shifted：两 token 一 task + 双重预取（`epilogue:291-426`）

shifted 的 Norm 读**物化到 global 的 `new_residual`**，一个 task 处理**两个 token**（
`kNumTokensPerShiftedNormTask=2`，`:284`），并把 hidden 切成若干 partition 以填满空闲 grid
（`:296-300`）。它用 `load_pack` + `next_pack` + `prefetch_pack` 三级寄存器预取隐藏 BF16 回读延迟
（`:315-321,341-344`）。两个 token 的 amax 打包成一个 BF16 pair 共用一个 warp 归约（`:362-365`，
小优化）。

## 19. 跨 CTA 的 SplitBarrier（`scheduler/mega_mhc.cuh`）

Mix 和 Norm 都需要「同一 m-block 的所有 split 都算完」才能开始（因为要跨 split 相加）。这是
`SplitBarrier`（`:13-49`）：

```cpp
encode_state(grid_idx, num_arrived) = (grid_idx+1) * 2 * kNumMaxSplits + num_arrived;   // :20-22
init:       st_rel(encode_state(grid_idx, 0));                                          // :24-26
wait_init:  while (ld_acq_gpu(ptr) - current_grid_base >= kNumSplits);                  // :29-34
arrive:     red_async_inc_rel(ptr);                                                     // :37-41
wait:       while (ld_acq_gpu(ptr) != encode_state(grid_idx, kNumSplits));              // :44-48
```

- 状态**一次原子地编码了「grid 序号 + 已到达数」**（`grid_idx` 每次 launch 自增）。`wait_init` 用
  **无符号距离**拒绝上一 grid 的陈旧标签、接受本 grid 的任意到达数（`:32` 注释）——这样 barrier
  内存被复用、上一轮残留也不会误判。
- `arrive` 用 `red_async_inc_rel`（异步 release 归约，`:39` 注释：SM100 可以把 release 归约异步
  退休，CTA 可以立刻开始下一 task）。
- **两个独立 barrier**：`Norm = SplitBarrier<0>`、`Mix = SplitBarrier<1>`（`:52-53`）——shifted
  的 Norm 与 Mix 是两个不同阶段的汇聚点。
- 首次 `init` 由 Post 的某个 warp 按 grid-stride 做（`impls:171-180`），且必须在
  `cudaGridDependencySynchronize()`（`:170`）之后——**局部初始化可以与前一个 grid 重叠，但持久化的
  split barrier 不行**（`:169` 注释）。

## 20. FP8 输出与三种 SF 布局（`mega_mhc.hpp:194-221`、`epilogue:117-136`）

`y_fp8` 用 **gran_k=32、4 个 UE8M0 SF 打包进一个 int32**（`mega_mhc.hpp:81-82`）。三种 SF 布局由
用途决定：

- **`y_gemm_sf`**：列主序（`stride(0)==1`）+ TMA 对齐（`:204-206`），给后续 GEMM 直接当操作数。
- **`y_routed_sf`**：行主序连续（`:210`），给 routed 路径。
- **`y_shared_sf`**：**Mega MoE 共享专家页布局**（`:212-221`），行号由 `get_shared_sf_row_idx`
  （`epilogue:117-122`）算出：每个 128 行页内**交错 4 个 32 行组**：

```cpp
idx = token % SF_BLOCK_M;
row = token/SF_BLOCK_M * align128(SF_BLOCK_M) + (idx & ~127) + (idx&31)*4 + ((idx>>5)&3);
```

这正是测试 `extra_sf_rows`（`tests:96-104`）的公式，也是 Mega MoE 里 UTCCP 4x32 转置后的 SF 页布局
（见 `docs/02` §3.4）——**mHC 直接把 SF 写成下游 MoE 能直接吃的页布局**。

**`store_fp8_and_pack_sf`（`:124-136`）**：先 `scale_bf16x2_into_fp8x4`（乘幂次 `sf_inv` 精确无损，
再转 4 个 E4M3），再用两次 `__byte_perm` 把 4 个 32-hidden 块的 exponent **打包成一个 int32**
（lane 0/16 落盘）：`__byte_perm(sf_exp, shfl_xor(sf_exp,4), 0x1140)` 再
`__byte_perm(paired, shfl_xor(paired,8), 0x5410)`，得到 `exp0 | exp1<<8 | exp2<<16 | exp3<<24`。

## 21. 确定性与 CUDA Graph

- 确定性模式固定 `num_splits=16`（`mega_mhc.hpp:16-17`），避免不同切分导致浮点加序变化。
- split barrier 在 capture 前必须 warmup（`:32-33`），否则 graph replay 会把一次性清零也录进去。
- 测试 `test_mega_mhc_api_contract`（`tests:241-315`）逐位断言：确定性模式下重复调用 bitwise 相同、
  CUDA Graph 重放 bitwise 相同、多 stream 独立、FP8-only 调用在不同尺寸调用后 bitwise 稳定。

## 22. H100 实测：架构无关语义与位级布局的忠实移植

Mega mHC 主体是 SM100 专属（TF32 TS UMMA / TMEM / tcgen05，`mega_mhc.hpp:223` 硬断言），H100 跑不了。
但它的**数学语义**和几处**位级布局**是架构无关的。实验 `src/mega_mhc_semantics.cu`
（`harness/exp_mega_mhc_semantics.sh` + `.out.txt`，nvcc sm_90）把源码逐行搬到裸 CUDA，并用**独立的
double 参考**（同一文件内实现）对拍：

```
== A 残差混合 + RMSNorm (T=256, H=256, hc=4) ==
   new_residual  rel 2.174e-02 / abs 9.766e-04   hc_sqr_sum rel 5.566e-07
   X1(bf16)      rel 0.000e+00 / abs 0.000e+00   y_bf16(RMSNorm) rel 7.812e-03 / abs 3.125e-02

== B mix 系数 + Sinkhorn (T=4096, iters=10) ==
   new_prev_mix  max_rel_err = 1.731e-07   new_post_mix max_rel_err = 1.890e-07
   new_comb_res_mix max_rel_err = 3.710e-07
   Sinkhorn 输出 |行和-1| max = 2.116e-06, |列和-1| max = 2.101e-06

== C 共享专家 SF 分页 get_shared_sf_row_idx (SF_BLOCK_M=224) ==
   T=1..32768: 与 test extra_sf_rows 不符 0 / 全部

== D FP8 UE8M0 + 4-lane SF 打包 (T=512 packs) ==
   打包 int32 与参考不符: 0 / 512
   E4M3 量化误差 max = 0.5000 ulp
```

**逐项解读**：

- **A**：`new_residual` 的 abs 误差 9.77e-4 与 **bf16 单 ulp**（在 O(1) 数值上 ~2^-9≈2e-3）同量级；
  相对误差 2.17e-2 是被**近零点相消**放大的（`post*x + comb@residual` 可能过零），不是算法问题。
  `hc_sqr_sum` 相对误差 5.6e-7（纯 fp32 累加 vs double）；`X1` 与参考**逐位相同**（rel/abs 全 0）。
- **B**：Pre/Post 系数相对误差 ~2e-7、comb ~4e-7——纯 `__expf`/`fast_rcp` 近似（源码 `epilogue:55,69`）
  带来的 fp32 级偏差。**Sinkhorn 输出行和、列和都稠密到 2.1e-6**，证实它确实收敛到双随机矩阵。
- **C**：`get_shared_sf_row_idx` 与测试 `extra_sf_rows` 在 T∈{1,64,65,200,1025,4096,32768}
  共 **4 万余行 0 不符**，坐实「4 个 32 行组交错进 128 行页」的页布局映射。
- **D**：4 个 exponent 打包成 int32 与「小端 `exp0|exp1<<8|exp2<<16|exp3<<24`」参考 **0/512 不符**；
  FP8 数值与「bf16 × 幂次 SF」的 E4M3 量化误差 **max = 0.5000 ulp**，即**完美 round-to-nearest**
  （0.5 ulp 是理论最优），证明 `get_ue8m0_sf_exp` + `scale_bf16x2_into_fp8x4` 的量化是无偏的。

## 23. Mega mHC 小结

- Mega mHC = **HyperConnection 的残差多流混合 + mix 系数投影 + Sinkhorn 组合矩阵 + RMSNorm
  （+FP8）**的单 kernel 融合。整个算子只有**一个窄输出、长 K 的 GEMM**（投影 `fn`），其余都是
  逐 token 的小矩阵运算——很适合融合。
- 三个非平凡设计点：**① UMMA 走 TF32 的 TS 形态**（A=残差在 TMEM、B=`fn` 在 SMEM），因为系数
  投影需要比 BF16 更高的尾数精度，而激活已在 `new_residual` 处强制 bf16 舍入（共享数值边界）；
  **② Sinkhorn 组合矩阵**用双随机矩阵做「无能量缩放」的残差再分配，kernel 里用 warp 内
  `shfl_xor` 逐行/逐列归约实现；**③ 跨 CTA 的 `SplitBarrier`** 用「grid 序号 + 到达数」一次编码，
  免复位、抗上层 grid 残留。
- H100 上不能端到端跑（`arch_major==10`），但把架构无关的数值语义与位级布局逐行移植后与独立
  double 参考完全一致：残差/RMSNorm 到 bf16 ulp、mix 系数 ~2e-7、Sinkhorn 双随机到 2e-6、
  SF 页布局 0 不符、SF 打包 0 不符、E4M3 量化 0.5 ulp 最优。

---

### 本文件对应的实验与源码

- Mega Gate 实验：`src/mega_gate_routing.cu`、`harness/exp_mega_gate_routing.sh`、
  `harness/exp_mega_gate_routing.out.txt`（H100 真机，nvcc sm_90）。
- Mega mHC 实验：`src/mega_mhc_semantics.cu`、`harness/exp_mega_mhc_semantics.sh`、
  `harness/exp_mega_mhc_semantics.out.txt`（H100 真机，nvcc sm_90）。
- 源码行号均核对自 `~/github/DeepGEMM` 公开版 commit `057ca59`。
