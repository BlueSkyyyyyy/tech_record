# Mega Gate：BF16 打分 GEMM 与 MoE top-k 路由的融合

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的 Mega-kernel 篇（路线图 **P5-1**）。
> 前四篇（`docs/02`~`docs/11`）把主体 Mega MoE kernel 逐段读完了；本篇换一个算子：**Mega Gate**——
> 把「router 打分 GEMM（`x @ W_gate^T`）」与「MoE top-k 路由（softplus→sqrt、加 bias 排名、选 top-k、
> 归一化权重、物理专家映射）」融合进**一个 kernel**。参考实现
> `~/github/DeepGEMM`（公开版 commit `057ca59`）：
> `csrc/apis/mega_gate.hpp`、`deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh`、
> `layout/mega_gate.cuh`、`scheduler/mega_gate.cuh`、`epilogue/sm100_mega_gate.cuh`、
> `csrc/jit_kernels/heuristics/mega_gate.hpp`、`tests/test_mega_gate.py`。
>
> **约束提醒**：Mega Gate 是 **SM100 专属，仅分析**——`mega_gate.hpp:158-159` 硬断言 `arch_major == 10`，
> H100(sm90) 端到端跑不了。但其中**架构无关**的路由数值语义（`sqrt_softplus`、warp 级 top-k 选择、
> 破平规则、权重归一化、重复专家分散公式）可在 H100 上逐行移植并对拍，本文给出真实数字
> （`src/mega_gate_routing.cu` + `harness/exp_mega_gate_routing.{sh,out.txt}`）。

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

## 12. 小结

- Mega Gate = **BF16 打分 GEMM + router 全链路**的单 kernel 融合，省掉 `[T,E]` 中间量的 global 往返。
- 三个非平凡设计点：**① 累加器经 TMEM 直接喂给 gate warp**（避免打分落 global）；②
  **带 bias 排名、不带 bias 出权重**（bias 只做负载均衡）；③ **warp 内排序网络 + 3-bit 游标归并**的
  确定性 top-k（破平取最小专家号）。
- H100 上无法跑端到端（`arch_major==10` 硬断言），但把架构无关的路由语义逐行移植后与参考
  完全一致（5 个案例、约 7 万条 top-k 全 0 不符），`sqrt_softplus` 阈值分支、重复专家分散公式
  也都有真实数字佐证。

> **P5-2 预告**：下一个增量读 **Mega mHC**（`mega_mhc.hpp` / `sm100_mega_mhc.cuh`），
> 讲 HyperConnection 相关的融合；完成后本节补入同一文档。

---

### 本文件对应的实验与源码

- 实验：`src/mega_gate_routing.cu`、`harness/exp_mega_gate_routing.sh`、`harness/exp_mega_gate_routing.out.txt`
  （H100 真机，nvcc sm_90）。
- 源码行号均核对自 `~/github/DeepGEMM` 公开版 commit `057ca59`。
