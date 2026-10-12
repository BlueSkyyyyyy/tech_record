# Mega Gate 的端到端验收测试与路由 oracle（`tests/test_mega_gate.py`）

> 第 13 篇（P5-1）拆过 Mega Gate 的**实现**：BF16 打分 GEMM + UMMA/TMEM 直读 + warp 级确定性
> top-k，并在 H100 上把「架构无关的路由语义」逐行移植对拍（`src/mega_gate_routing.cu`）。
> 第 51 篇（P41-1）拆过 **Mega MoE** 的端到端测试 `tests/test_mega_moe.py`。
> 但 Mega Gate 自己的**验收测试** `tests/test_mega_gate.py`（318 行）一直只被顺带引用过几行。
>
> 本篇把它拆透：它定义了「Mega Gate 的输出到底算什么对」。测试分两半——
> **主体 kernel `sm100_bf16_mega_gate` 是 SM100 专属**（`csrc/apis/mega_gate.hpp:159` 断言
> `arch_major == 10`），H100 跑不了；但它的**路由 oracle**（sqrt-softplus 打分、带 bias 排名、
> top-k 权重归一化、logical→physical 逆映射、top-k 边界 crossing 容忍、mask/fix_routing/force_random
> 语义）**全是纯 torch，架构无关，H100 真机可跑**。这一半就是本篇的实测对象。

---

## 1. 测试全貌：一个融合算子，两个用例函数

`tests/test_mega_gate.py` 只有两个 `@test_filter(lambda: get_arch_major() == 10)` 的入口：

| 用例 | 行号 | 做什么 |
|------|------|--------|
| `test_mega_gate_api_contract` | `:256-313` | 契约与边界：11 组 `GATE_CONFIGS`、`with_bias=False`、极端 bias、routing 控制、全零输入、0 token、`unmapped_topk_idx` 非连续 view、确定性模式 |
| `test_mega_gate` | `:204-225` | 性能/正确性巡礼：45 个 `enumerate_cases()` 形状，逐形状对拍 + `bench_kineto` 计时 |

两者都门控在 `get_arch_major() == 10`，所以在 H100（arch 9）上**整体被跳过**（实测 `get_arch_major()=9`）。
这不是测试不够，而是验证对象 `deep_gemm.bf16_mega_gate` 只在 `csrc/apis/mega_gate.hpp:159` 的
`arch_major == 10` 之后才允许调用——H100 上调它会在 host 断言处直接抛错。

### 1.1 用例矩阵

`enumerate_cases()`（`:27-37`）给出 45 个形状，分两段：

- 第一阶段 `(num_tokens, hidden, 384)`：`num_tokens ∈ {1,3,16,128,512,1024,2048,4096,8192,16384,32768}`、
  `hidden ∈ {4096,5120,7168}`，共 33 个。`num_routed_experts=384` 是 DeepSeek 的专家数。
- 第二阶段手工补的 12 个：覆盖 `num_routed_experts ∈ {128,256,264,384,512}`（含非 128 倍数 `264`
  与上限 `512`）、以及注释点名的 **「4 waves + K loop 不是流水深整数倍」** 用例
  `(32768, 4096, 512)`（`:35-36`，`carries the stage position across blocks`）。

`GATE_CONFIGS`（`:21-24`）是契约用例的 11 组 `(num_routed_experts, num_shared_experts, num_topk)`，
含 `use_shared_as_routed` 合法性组合的枚举（`:262-265`）。常量：`NUM_TOPK=6`、`NUM_SHARED_EXPERTS=1`、
`ROUTED_SCALING_FACTOR=1.5`、`VLLM_NUM_EXTRA_EXPERTS=32`（`:16-19`）。

---

## 2. 路由 oracle：`check_topk_weights` 逐段

融合 kernel 的输出是三张表：`topk_idx [T,K']`、`topk_weights [T,K']`、可选 `unmapped_topk_idx [T,K]`
（logical 索引）。`check_topk_weights`（`:105-173`）是唯一的判据。注意它**不看 `topk_idx` 本身选得对不对**，
而是从 `topk_idx` 反推回 logical 索引、重算一遍参考权重再比：

### 2.1 打分：`sqrt_softplus` vs `softplus(·).sqrt()`

kernel 侧 `sqrt_softplus`（`epilogue/sm100_mega_gate.cuh:19-23`）是：

```
softplus(x) = x > 20 ? x : log1p(exp(x))     # threshold-20 折衷，避免 exp 溢出
score       = sqrt(softplus)
```

测试参考（`:117`）写的是 `scores = F.softplus(logits).sqrt()`。**关键是这两者数值上完全一致**：
`torch.nn.functional.softplus` 的默认 `threshold=20`，恰好采用同一条 `x > 20 ? x : log1p(exp(x))`
分支规则。H100 实测（`exp_mega_gate_test.out.txt` §B）：在 `[-40, 40]` 上 4001 点
`max|Δ| = 0.000e+00`；与 fp64 真值的相对误差 `1.369e-7`（纯 fp32 舍入，`2^-23≈1.19e-7`）；
而 linear 分支丢掉的那个 `log1p(e^{-x})≈e^{-x}` 项，在 `x>20` 上相对误差上界为 `e^{-x}/x = 1.009e-10`
（`x=20` 处），**远小于 fp32 舍入**——所以 threshold-20 折衷不引入可观误差。

### 2.2 排名：`score + bias`，image token 用 image_bias「替换」

`:130-135` 构造排名用的 bias：

```python
ranking_bias = inputs['bias'] if inputs['bias'] is not None else zeros(E)
if inputs['image_token_mask'] is not None:
    ranking_bias = torch.where(image_token_mask[:, None], image_bias, ranking_bias)
ranking = scores + ranking_bias
```

一个容易踩的语义点：image token 的 bias 是 **image_bias 替换 bias**（`where` 二选一），**不是叠加**。
多模态训练里 image token 需要另一套负载均衡偏置，所以是「换」不是「加」。
kernel 侧对应地只缓存一份 `bias_or_image_bias`（layout `caches_logical_count` 把 bias 与 image bias
一起塞进 4 KiB 的 `metadata_cache`，`layout/mega_gate.cuh:45-51`）。

### 2.3 top-k 选择与「crossing 容忍」

kernel 的 top-k 是**确定性**的：排名平局时取最小专家号（`epilogue:83-89` 的
`warp_reduce_best`/`compare_swap_best_first`，`redux.sync.max` + `__reduce_min_sync`）。
测试不逐位复刻排序网络，而是检查一个**等价性质**（`:145-156`）：

```python
ranked_idx = logical_topk_idx[ranked]
sorted_idx = ranked_idx.sort(dim=1).values
assert (sorted_idx[:, 1:] != sorted_idx[:, :-1]).all()          # 选中的 K 个互不相同
selected_ranking = ranking[ranked].gather(1, ranked_idx)
unselected_max   = ranking[ranked].scatter(1, ranked_idx, -inf).max(1).values
crossing         = unselected_max - selected_ranking.min(1).values
tolerance        = 1e-5 + 1e-5 * ranking[ranked].abs().amax(1)
assert (crossing <= tolerance).all()                            # 选中的最小分 ≥ 未选中的最大分（容差内）
```

即「**选中的 K 个排名值最小值 ≥ 未选中排名值最大值**」——这正是「是真正的 top-k」的充要刻画，
且允许 `1e-5 + 1e-5·|ranking|_max` 的数值容差来处理平局。这个写法把「排序网络的实现细节」
与「top-k 的数学定义」解耦：即使 kernel 用了 sort4 网络 + 3-bit 游标归并（第 13 篇），
只要结果满足这个不等式就算对。

### 2.4 权重归一化

参考权重（`:137-139`）：

```python
def reference_weights(row_mask, topk_idx):
    topk_scores = scores[row_mask].gather(1, topk_idx)
    return topk_scores / (topk_scores.sum(1, keepdim=True) + 1e-20) * ROUTED_SCALING_FACTOR
```

与 kernel `store_topk_token`（`epilogue:210-213`）逐字对应：

```cpp
const auto topk_sum = math::warp_reduce_sum<kSumWidth>(selected_unbiased_score) + 1e-20f;
const auto selected_weight = lane_idx < kNumTopk ?
    selected_unbiased_score / topk_sum * args.routed_scaling_factor : 1.0f;
```

**排名用带 bias 的 `ranking`，出权重用不带 bias 的 `scores`**——bias 只做负载均衡、不进权重。
共享专家的槽（`lane_idx ∈ [kNumTopk, K')`）权重恒 `1.0`，测试对应断言 `:164-165`。

### 2.5 判据是 `calc_diff < 1e-11`（不是 `torch.equal`）

与 Mega MoE 端到端测试用 `torch.equal`（第 51 篇）不同，Mega Gate 这里用的是
`calc_diff < 1e-11`（`:141-143`）。原因是参考 logits 来自 `run_reference_gemm`
（`:98-102`）的 **`deep_gemm.cublaslt_gemm_nt`（bf16）**，而 kernel 内部是自己的一条 BF16
打分 GEMM——两条 GEMM 的累加顺序不必逐位相同，产物 logits 就会差最后几位。`calc_diff`
是归一化平方误差（第 40 篇），`1e-11` 这个阈值非常严：对均值 `|w|≈0.25` 的权重，
`calc_diff≈1e-11` 只对应约 `4.5e-6` 的相对误差（实测 §C）。也就是说它**堵死了「实现走错专家/
权重没归一化/给错 bias」这类结构性错误**，只放过 GEMM 末位噪声。

---

## 3. logical → physical 专家映射与它的「隐式逆」

动态专家复制（EPLB / MoonEP / UltraEP，第 53 篇）下，一个 logical 专家有多个 physical 副本。
测试用 `make_physical_map(num_logical_experts)`（`:40-49`）造一份映射：

```python
to_physical_map = full((num_logical, VLLM_NUM_EXTRA_EXPERTS + 1), -1)   # [E, 33]
to_physical_map[:, 0] = arange(num_logical)                            # 第 0 列：直通自身
to_physical_map[:32, 1] = num_logical + arange(32)                     # 第 1 列：前 32 个的副本
logical_count[:32] = 2
```

于是 physical 专家既可能是 `l`（副本 0），也可能是 `E + l`（副本 1，只对前 32 个 logical）。

`check_topk_weights` 在**没有** `unmapped_topk_idx` 时，从 physical 反推 logical（`:121-128`）：

```python
logical_topk_idx = fused_idx[:, :num_topk]
logical_topk_idx = where(logical_topk_idx < num_logical,
                         logical_topk_idx, logical_topk_idx - num_logical)
```

这条逆映射 `p < E ? p : p - E` **只在「副本槽恰好等于 `E + logical`」时成立**——它依赖
`make_physical_map` 把重复槽按 logical 顺序编号这个构造约定。H100 实测（§D）：对每个
`(logical, duplicate)` 槽逆映射都还原出 logical；把某个重复槽改成 `E+5` 后逆映射就还原错
（`inverse=5≠0`）。这是一个**测试内部的隐式耦合**：`make_physical_map` 与逆映射必须联动改。

kernel 侧选哪个副本由 `epilogue:198-207` 决定：

```cpp
const auto num_duplicates = ...logical_count[logical_expert_idx];
const auto duplicate_idx = (args.rank_idx + global_token_idx * 23333u) % num_duplicates;   // :205
physical_expert_idx = args.to_physical_map[logical_expert_idx * args.num_duplicate_experts + duplicate_idx];
```

`(rank + t·23333) % num_duplicates` 把一个 logical 专家的 token 在不同 rank 上分散到不同副本
（实测 §D 在 `num_duplicates=2` 下两块副本各占 `0.50000`）。注意**逆映射与选了哪个副本无关**：
只要副本是 `E + logical`，任一副本都能还原回 logical。

---

## 4. 四个路由控制开关

`check_routing_controls()`（`:228-253`）用一个 512×4096、`E=384`、`use_shared_as_routed=True`、
`with_image_bias=True` 的配置，把四个开关叠在一起：

| 开关 | 语义 | kernel 落点 | 测试断言 |
|------|------|-------------|----------|
| `mask [T]` | `False` 的 token 不路由：`topk_idx=-1`、`weights=0` | epilogue 写零 | `:167-171` |
| `image_bias` | image token 用 image_bias 排名 | `metadata_cache` 里的 bias 替换 | `:130-135` |
| `fix_routing_mask [T]` | 该 token 的 top-k **保持** `unmapped_topk_idx` 里给定的路由 | `store_topk_token` 直接用给定 idx | `:246-247` |
| `force_random [T]` | 该 token 随机选 K 个**互不相同**的专家（负载测试） | `store_random_topk_token`（`epilogue:218-253`） | `:245-253` |

- `fix_routing` **必须**配合 `unmapped_topk_idx`：host 契约 `mega_gate.hpp:137-138` 断言
  `fix_routing_mask.has_value() → unmapped_topk_idx.has_value()`，因为要读的正是那张表。
- `force_random` 里 `fixed = fix_routing_mask & mask & ~force_random`（`:246`）：**随机优先于固定**，
  重叠的 token 走随机。
- `force_random` 的随机是**无放回采样**（`epilogue:230-250`）：lane `i` 从剩余的 `n-i` 里抽，
  再按「比它小且更早抽中的整体右移一位」重编号，保证 K 个互不相同。实测 §E6 验证 idx 互不相同、
  `unmapped==-1`、权重 `>0`。

---

## 5. 逐位判据与确定性

`check_correctness`（`:176-191`）在正常对拍之外，把输出**克隆一份再重复 launch 10 次**：

```python
expected = {name: t.clone() for name, t in outputs.items() if t is not None}
for _ in range(num_repeats):
    run_mega_gate(...)
    for name, tensor in expected.items():
        assert_bitwise_equal(outputs[name], tensor, name)   # :190
```

**逐位**判据在此成立的原因是 kernel 的归约顺序是固定的：split-K 的 cross-CTA 汇聚走
`ScoreBarrier`（第 13 篇，`scheduler/mega_gate.cuh:13-39`），确定性模式下更是**从不 split K**
（`heuristics/mega_gate.hpp:110-111` 注释「deterministic mode is also batch invariant, so K is
never split」、`:126` 的 `num_split_k <= (deterministic ? 1 : kNumMaxSplitK)`）。这也是
`test_mega_gate_api_contract` 在末尾专门用 `use_deterministic_algorithms(True)` 再跑两组的动机
（`:307-313`）。

---

## 6. H100 实测：用「完全正确的代理 gate」验证 oracle 自洽，再用扰动反例验证它能抓错

由于 kernel 是 SM100 专属，本篇换一个角度：**不用 kernel，只验证判据本身**。
`harness/exp_mega_gate_test.py` 直接 `import` 真实的 `tests/test_mega_gate.py` 作为模块，
复用它自带的 `make_inputs`/`make_outputs`/`check_topk_weights`/`make_physical_map`，
再用一个**完全按 oracle 语义写出来的 torch 代理 gate** 去喂 `check_topk_weights`：

- 若代理正确 → `weight_diff = 0.000e+00`（同一份 `scores`、同一个归一化，应逐位为 0）；
- 然后**故意注入各类错误**，验证判据确实会报错。

结果 **47 PASS / 0 FAIL**（完整输出见 `harness/exp_mega_gate_test.out.txt`）：

| 组 | 验证内容 | 结果 |
|----|----------|------|
| A | 常量/枚举：`enumerate_cases`=45、首项 `(1,4096,384)`、含 `(32768,4096,512)`、`GATE_CONFIGS`=11 | 6 PASS |
| B | `sqrt_softplus` 与 `F.softplus(·).sqrt()` `max|Δ|=0`；vs fp64 `1.369e-7`（fp32 舍入）；linear 分支解析误差 `1.009e-10` | 4 PASS |
| C | 正确代理 `weight_diff=0`；扰动反例见下 | 6 PASS |
| D | physical map 形状 `(384,33)`、副本编号、逆映射、反例、副本分散均衡 `0.50000` | 7 PASS |
| E | mask / image_bias / 极端 bias / fix_routing+force_random+mask | 7 PASS |
| F | 全零打分 → `arange(K)`；0 token | 4 PASS |
| G | host 契约与 arch 门控（`arch==10`、`hidden%256`、`E≤512`、`scoring_func`、fix_routing 依赖） | 10 PASS |
| H | 权重公式逐位、反例 | 3 PASS |

### 6.1 扰动反例：判据真的能抓错

| 反例 | 注入 | 结果 |
|------|------|------|
| C3 | 权重整体乘 `(1+1e-3)` | `calc_diff=4.995e-7 > 1e-11` → 断言触发（`:143`） |
| C4 | 权重整体乘 `(1+1e-6)` | `calc_diff=4.543e-13 < 1e-11` → 仍通过（阈值不是无限严） |
| C5 | 把一个选中专家换成第 `K+1` 名、权重自洽 | `crossing > tolerance` → 触发（`:155`） |
| C6 | 两个槽选同一专家，权重自洽 | 唯一性断言触发（`:149`） |
| E2b | 忽略 image_bias（只用 bias）重算选择 | 53.1% 的 token 选择不同 → 证明替换语义是实质的 |
| H3 | 漏掉 `/sum` 归一化 | `calc_diff=0.713` → 明显偏离 |

C5 尤其说明 crossing 判据的价值：它是**结构无关**的，能抓出「排序实现选错了一个专家」
而无需逐位复刻排序网络。

---

## 7. kernel 侧对应速查（SM100 专属，仅分析）

| 测试概念 | kernel 实现 |
|----------|-------------|
| 打分 `sqrt_softplus` | `epilogue/sm100_mega_gate.cuh:19-37`（标量 + 向量版，threshold 20） |
| 打分 GEMM 写 TMEM 再读出 | `epilogue:39-78`（`tmem_load_32dp32b` + `fence_view_async_tmem_load`） |
| 确定性 top-k（平局取小专家号） | `epilogue:80-105`（`warp_reduce_best` + `sort4_best_first`）、`:107-152`（3-bit 游标归并） |
| 权重归一化 + 共享槽 `1.0` | `epilogue:210-215`（`warp_reduce_sum<kSumWidth> + 1e-20f`） |
| logical→physical + 副本分散 | `epilogue:198-207`（`(rank+t·23333)%n_dup`） |
| force_random 无放回采样 | `epilogue:218-253`（`curand_init(rank, t*32+lane, 0)`） |
| split-K 跨 CTA 汇聚 | `scheduler/mega_gate.cuh:13-39`（`ScoreBarrier`） |
| 从不 split K（确定性） | `heuristics/mega_gate.hpp:110-111,126` |

### 7.1 host 契约（`csrc/apis/mega_gate.hpp`）

- 形状/对齐：`hidden % kHiddenAlignment(256) == 0`（`:83`）；`num_routed_experts ≤ 512`（`:80,84-85`）；
  `num_topk ∈ (0, num_routed_experts]`（`:96`）；`num_physical_topk = num_topk + S ≤ 32`（`:108-109`）。
- `use_shared_as_routed`：`num_shared_experts ∈ {1,2}`、`num_topk % S == 0`、
  `num_routed % (num_topk/S) == 0`（`:103-105`）。
- 一致性：`image_bias ⟺ image_token_mask`（`:116`）、`to_physical_map ⟺ logical_count`（`:122`）、
  `fix_routing_mask → unmapped_topk_idx`（`:137-138`）。
- `scoring_func == "sqrtsoftplus"`（`:99`）；`unmapped_topk_idx` 允许行 stride ≠ `num_topk`
  但 slot 连续（`:131-136`，测试用 `(16, num_topk+3)` 的窄 view 验证，`:288-291`）。
- 唯一闸门 `arch_major == 10`（`:159`）——H100 无法调用。

---

## 8. 结论与边界

- **Mega Gate 的「正确」由 `check_topk_weights` 定义**，判据是 `calc_diff < 1e-11`（不是逐位）：
  排名与权重走两套输入——排名带 bias、权重用无 bias 的 `scores` 再归一化——image token 的 bias
  是**替换**不是叠加。
- **top-k 选择用「crossing 不等式」验收**（选中最小分 ≥ 未选中最大分，容差 `1e-5+1e-5·|ranking|_max`），
  与排序网络的实现解耦；加权重的 `1e-11` 阈值则堵死结构性错误、只放过 GEMM 末位噪声。
- **logical→physical 的逆映射 `p<E?p:p-E` 依赖 `make_physical_map` 的编号约定**（副本槽=`E+logical`），
  是测试内部的隐式耦合；kernel 侧选副本的 `(rank+t·23333)%n_dup` 只影响负载、不影响逆映射。
- **H100 实测 47 PASS / 0 FAIL**：一个完全正确的 torch 代理 gate 让判据 `weight_diff=0`，
  六类扰动反例（权重偏 `1e-3`、换专家、重复专家、忽略 image_bias、漏归一化）全部被抓到。
- **边界**：真正的主体 kernel `sm100_bf16_mega_gate` 与端到端巡礼（`bench_kineto` 的 TFLOPS/GB/s）
  是 SM100 专属，本机不可跑，本篇只做源码精读与 oracle 验证。
