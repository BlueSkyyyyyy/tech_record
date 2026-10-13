# DeepEP 的 MoE 路由分数/拓扑生成器 `deep_ep/utils/gate.py`

> 本文对应路线图 **P106-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 116 篇。
>
> 第 65 篇（`docs/65-deepep-refs-and-testing.md`）在总览 DeepEP `deep_ep/utils/` 时把
> `gate.py` 一句话概括为「**流量生成层**」；第 114 篇拆了同目录的 `math.py`（语义工具层）、
> 第 115 篇拆了 `refs.dispatch` 的 FP8 分支。但 **`gate.py` 这 180 行到底在造什么、为什么
> DeepEP 的每一份 EP 测试都要先过它、它保证了什么、又偷偷不保证什么**，一直没被单独读过。
> 本篇把它逐段拆透，并在 H100 上把它的**结构不变式**与**验收判据**逐条跑出来。
>
> **可跑性**：`gate.py` 只 `import torch`（不 import `deep_ep` 包），按文件路径直接 exec
> 真实源码即可，完全绕开 `deep_ep/__init__.py:39` 的 `check_nccl_so()` 运行期阻塞。
> 实验全在 H100 真机跑（`harness/exp_deepep_gate.{sh,py,out.txt}`），**ALL PASS**。
>
> 源码坐标（`~/github/DeepEP`，本地 commit `93eb6eb`、`deep_ep.__version__ == 2.5.0`）：
> - `deep_ep/utils/gate.py:4-29`（`generate_topk_idx`）、`:32-113`（`generate_rank_count`）、
>   `:116-137`（`get_precise_unbalanced_scores`）、`:140-145`（`get_scores_by_factor`）、
>   `:148-164`（`map_unbalanced_ratio_to_factor`）、`:167-180`（random / 入口）
> - 唯一验收规范：`tests/utils/test_gate.py:7-53`
> - 下游调用点：`tests/ep/test_ep.py:13/77`（EP buffer 测试造不均衡流量）

---

## 1. 一段话总览

`gate.py` 是 DeepEP 的 **MoE 路由夹具（fixture）**：它不实现任何算子，而是**制造可控的专家
选择分布**，让 EP 的 dispatch/combine 测试能覆盖「某个 rank 的流量是别人几倍」这种最坏情况——
因为真实 router 的输出不可控、也无法稳定复现。它提供两条互补的生成路线：

- **`precise`（精确）**：先造一张 **rank 级**流量表 `rank_count [T, R]`（每 token 的 top-k
  专家落在各 rank 上的个数），再把这几个位置填上高分；保证 **特殊 rank（rank 0）恰好拿到
  `ratio` 倍流量、其余 rank 近乎完全均衡**（`inequality < 1.02`）。用于**压力测试**。
- **`random`（随机）**：直接给每个专家一个分数（rank 0 的专家分数上界放大 `factor` 倍），
  只保证**比值大致**是 `ratio`，不保证其余 rank 均衡。用于**更接近真实 router 的分布**。

判据（`test_gate.py:38-52`）是 `practical_ratio = count[0] / min(count[1:])` 要落在 `ratio ± 10%`、
且 `inequality = max(count[1:]) / min(count[1:]) < 1.02`——**这个断言只在 `precise` 分支检查**
（`:48` 的 `if precise:`），因为只有 precise 模式承诺均衡。

---

## 2. `generate_topk_idx`：rank_count → top-k 索引的逆映射（`:4-29`）

```python
assert torch.equal(torch.sum(rank_count, dim=1),
                   torch.ones(num_tokens, dtype=torch.int) * num_topk)   # :8
num_experts_per_rank = num_experts // num_ranks                          # :10
base_vals = torch.arange(num_experts).view(1, num_ranks, num_experts_per_rank) \
                .expand(num_tokens, num_ranks, num_experts_per_rank)     # :13
rand_vals = torch.rand(num_tokens, num_ranks, num_experts_per_rank)      # :16
perm_indices = torch.argsort(rand_vals, dim=-1)                          # :17
permuted = torch.gather(base_vals, 2, perm_indices)                      # :18
k_idx  = torch.arange(num_experts_per_rank).view(1,1,num_experts_per_rank) \
             .expand(num_tokens, num_ranks, num_experts_per_rank)        # :21
mask = k_idx < rank_count.unsqueeze(2).expand(...)                       # :22-23
topk_idx = permuted[mask].view(num_tokens, num_topk)                     # :26-27
```

它的输入是一张**每 token 每 rank 选几个专家**的计数表，输出是 `[T, num_topk]` 的**全局专家
索引**。为什么需要它：`get_precise_unbalanced_scores` 有「想要的 rank 级流量」，却要落到
「每个 token 具体选了哪 `num_topk` 个专家」才能造分数。做法是**逐 rank 独立**：

1. 把每个 rank 的专家区间 `[r·E_s, (r+1)·E_s)` 排成 `base_vals`（`:13`），对每个 `(token, rank)`
   用随机排列打乱（`:16-18`）——保证同一 `rank_count` 每次得到**不同**的具体专家（避免测试过拟合）；
2. 用 `mask`（`:22-23`）在每个 rank 区间里**取前 `rank_count[t, r]` 个**（在打乱后的顺序里）；
3. 展平后 `view(T, num_topk)`（`:26-27`）。

两个**可测的结构性质**（H100 Part A，14 例全对）：

- **精确可逆**：把 `topk_idx` 按 `idx // E_s` 落回 `(token, rank)` 计数，**逐位等于**输入的
  `rank_count`（A3）。也就是说 `generate_topk_idx` 与「计数」互为逆。
- **组内有序**：`mask` 按 `(T, rank, within_rank)` row-major 展平，所以同一 token 的索引
  沿列方向**先按 rank 升序**、rank 内才是乱序（A4）——下游若假设「top-k 列按专家号排序」
  会踩坑。
- 顺带的边界：索引恒落在 `[0, E)`（A2）、形状恒为 `(T, num_topk)`（A1）。

---

## 3. `generate_rank_count`：把「比值」精确灌进一张 rank 表（`:32-113`）

这是全文件最核心、也最不直观的一段。目标是造一条 `[T, R]` 的表，满足：
① 每行和恰为 `num_topk`；② 特殊 rank 0 的总量是其余 rank 的 `ratio` 倍；③ 其余 rank 之间
近乎完全均衡。算法分四步。

**第一步，先看真实 top-k 会落在几个不同 rank 上**（`:46-57`）：

```python
random_scores = torch.rand(num_tokens, num_experts)                    # :46
_, topk_indices = torch.topk(random_scores, num_topk, ...)             # :47
topk_indices //= num_experts_per_rank                                  # :48  ← 专家索引折叠成 rank
sorted_topk_indices = torch.sort(topk_indices, dim=1)[0]               # :49
topk_indices_diff_mask = sorted_topk_indices[:,1:] != sorted_topk_indices[:,:-1]  # :50
a = topk_indices_diff_mask.sum(dim=1) + 1                              # :51  ← 不同 rank 数
upper_bound_per_token = int(num_normal_ranks / ratio) + 1              # :54
a = torch.clamp(a, None, upper_bound_per_token)                        # :57
```

`a` 是「一个 token 的 top-k 落在多少个**不同 rank**上」。它被 clamp 到
`upper_bound_per_token = int((R-1)/ratio) + 1`（`:54`）——这是算法自己声明的**每个 token 能
触及的 rank 数上限**。为什么这么取：要造出 `ratio` 倍的 rank 级不均衡，每个 token 能触及的
rank 越少、越容易把流量集中在特殊 rank 上；这个上限正是「其余 rank 少被触及、rank 0 多被触及」
的可行边界（第 7 节）。

**第二步，决定有多少个 token 是「特殊 token」（必须含 rank 0）**（`:60-75`）：

```python
sum_a = torch.sum(a).item()
normal_token_count  = int(sum_a / (num_normal_ranks + ratio))          # :61
special_token_count = sum_a - normal_token_count * num_normal_ranks     # :62
special_token_count = min(special_token_count, int(normal_token_count * ratio) + 1)  # :63
must_mask  = (a == num_ranks)                                          # :66  ← 必然含 rank0 的 token
must_count = int(must_mask.sum().item())
special_token_count = max(must_count, special_token_count)             # :68
```

这里有个**非平凡观察**（H100 Part 探针，写进 `test_gate.py` 的背景）：`must_mask = (a == num_ranks)`
（`:66`）只在「一个 token 的 top-k 已经横跨**全部** rank」时才为真——而这要求
`num_topk >= num_ranks`（否则 `a ≤ num_topk < R`）。实测：`R=8, EPR=8, k=4` 时 `a==R` 出现 0 次；
`k=16` 时出现 890/4096 次。所以对常见的 `k < R` 配置，`must_mask` 恒为空，`special_token_count`
完全由 `:61-63` 的容量守恒式决定。

**第三步，给普通 rank 生成「近乎均匀」的填充序列**（`:78-104`）：

```python
rank_perm = (torch.randperm(num_normal_ranks) + 1).repeat(
                num_tokens * num_topk // num_normal_ranks + 1)          # :78  ← 循环重复同一排列
a_cumsum   = torch.cumsum(torch.cat(([0], a)), dim=0)
row_starts = a_cumsum[:-1]                                              # :81-82
mod_idx    = col_idx % a.view(num_tokens, 1)                            # :98
b_indices  = row_starts.view(num_tokens, 1) + mod_idx                   # :101
result     = rank_perm_with_special_rank[b_indices]                     # :104
```

**这是「其余 rank 均衡」的来源**：每个 token 占了 `a[t]` 个连续槽位，取的是 `rank_perm`
里**连续的一段**；而 `rank_perm` 是「一个随机排列 + 原样重复」。当 `a ≤ num_normal_ranks`
时，连续一段内 rank 两两不同（排列无重复）；更关键的是**所有 token 都在消费同一条重复序列**，
于是普通 rank 被触及时长高度接近——这就是 precise 模式 `inequality < 1.02` 的机制。对特殊 token，
`:85-91` 在它那段的**开头**插入 rank 0，再把普通 rank 往后错开。

**第四步，`scatter_add` 出表**（`:111-112`）。注意一个**反直觉但可测的事实**（H100 Part B）：
`rank_count` 的元素**不是 0/1**。当 `num_topk > num_ranks`（或流量集中）时，一个 token 的
多个 top-k 专家会落在**同一个 rank** 上，于是 `rank_count[t, r]` 可以是 2、甚至 4
（实测 `R=8,EPR=8,k=8` 的 `max_entry=4`、`distinct_ranks` 低到 2）。`generate_topk_idx` 的
round-trip 也允许这种计数（Part A 用了 `[128,8,8,8]` 等含多专家的配置）。

实测结构不变式（H100 Part B，7 组配置）：行和恒 `== k`（B1）、`max_entry ≤ k`（B3）、
特殊 rank 恒为 rank 0 且实测 `ratio=3.996`（B4，目标 4.0）。

---

## 4. 两条生成路线与它们的分数（`:116-180`）

**`get_precise_unbalanced_scores`（`:116-137`）**——把 `rank_count` 变成分数：

```python
rank_count = generate_rank_count(...)                                  # :124
scores = torch.empty((num_tokens, num_experts)); scores.uniform_(to=0.9)         # :127-129
topk_idx = generate_topk_idx(rank_count, ...)                          # :132
topk_scores = torch.empty((num_tokens, num_topk)); topk_scores.uniform_(0.9+1e-6, 1.0)  # :133-134
scores[row_idx, topk_idx] = topk_scores                                # :135-136
```

**背景专家分数 ≤ 0.9、被选中的 top-k 分数 `> 0.9`**——一个干净的**阈值分离**，保证任何
`torch.topk(scores)` 都精确选出 `generate_topk_idx` 指定的那批专家（H100 Part D 实测：
selected ∈ `[0.9000, 1.0000]`、background ∈ `[0.0000, 0.9000]`，D1）。这一步把「rank 表」
翻译成「能被 `torch.topk` 复现的分数张量」，下游 `test_ep.py:77-78` 正是
`topk_weights, topk_idx = torch.topk(scores, num_topk)`。

**`get_random_unbalanced_scores`（`:167-173`）**——不做 rank 表，直接给专家分数：

```python
def get_scores_by_factor(...):
    scores[:, :num_experts_per_rank].uniform_(to=factor)               # :143  rank0 专家分数上界 factor
    scores[:, num_experts_per_rank:].uniform_(to=1)                    # :144  其余 rank 上界 1
def map_unbalanced_ratio_to_factor(...):
    factor_l, factor_r = 1.0, 100.0                                    # :150
    for _ in range(20):                                                # :149  固定 20 轮二分
        factor_mid = (factor_l + factor_r) / 2
        scores = get_scores_by_factor(..., factor_mid)
        rank_idx = topk_idx // num_experts_per_rank
        counts = one_hot(rank_idx, num_ranks).any(dim=1).float().sum(0)  # :158-159
        if counts[0] > counts[1:].mean() * ratio: factor_r = factor_mid  # :160-161
        else:                                     factor_l = factor_mid  # :162-163
    return factor_l                                                    # :164
```

`map_unbalanced_ratio_to_factor` 在**单个标量** `factor`（rank 0 专家分数上界）上做 20 轮
二分，让 `count[0] ≈ ratio · mean(count[1:])`。为什么是单个 factor 而不是逐 rank 调：random
模式的目的是「大致像真实 router」，一个标量够用；而且 `factor` 物理意义直白——rank 0 的专家
在打分时被整体抬高 `factor` 倍。

---

## 5. 为什么 precise 承诺均衡、random 不承诺（H100 Part C）

复刻 `test_gate.py:38-52` 的判据（对每个 rank 各造一次、累加 `R` 次以消 token 级抖动，
`T=2048`，45 组可行配置）实测：

| 模式 | 满足 `ratio±10% ∧ inequality<1.02` | 最差 inequality |
|---|---|---|
| precise | **45 / 45** | 1.0103 @ `(R=64,EPR=8,k=1,ratio=4)` |
| random  | 14 / 45 | **1.1201** @ `(R=64,EPR=1,k=1,ratio=2)`，ratio=2.167 |

**31/45 的 random 配置违反 `inequality<1.02`**。这正是 `test_gate.py:48` 把断言放进
`if precise:` 的原因——不是 random 坏了，而是它**本来就只承诺比值、不承诺均衡**。random 用
`one_hot(...).any()`（只问「rank 是否出现」）来度量流量，天然比 precise 的「精确槽位分配」
抖得多，尤其当 `EPR=1`（每个 rank 只有一个专家、任何随机波动都被放大）时。

（另：`test_gate.py` 只对 `num_ranks in [8,64] and EPR==8` 打印明细、且只对
`num_tokens>1000` 做断言——见 `:42-43/:48`。小 `num_tokens` 时统计量噪声太大，直接跳过。）

---

## 6. `map_unbalanced_ratio_to_factor` 的两个陷阱（H100 Part E）

**单调、可行目标精确命中**（E1/E2）：`ratio` 随 `factor` 单调不减，二分后实测
`ratio(f) ∈ {1.99, 4.08, 2.00, 4.03, 2.02, 4.07}`，全部落在目标 ±10% 内。

**陷阱一：大 `R` 时几乎是「阈值开关」而非连续调节。** 因为度量是 `one_hot(...).any()`，
当 `R` 很大、`ratio` 只由「rank 0 是否出现」决定时，`factor` 微调就会让 `count[0]` 在
「几乎 0」和「远大于目标」之间跳：实测 `R=64,EPR=8,k=8` 在 `factor` 附近有
`ratio(0.9·f)=0.000 → ratio(f)=3.984 → ratio(1.1·f)=7.066`。**二分仍收敛，但解附近不可导**；
若调用方以为「比值随 factor 平滑变化」会误判。

**陷阱二：目标不可行时静默饱和。** `R=16, EPR=1, k=4` 时，rank 0 只有 1 个专家、
`one_hot.any` 至多让 rank 0 每 token 出现一次，而 top-4 里其余 3 个槽由 15 个 normal rank
分摊，于是 `ratio` 的**理论上限 ≈ (R-1)/(k-1) = 5**。目标 `ratio=8` 不可达，二分一路把
`factor_l` 顶到上界：**返回 `factor = 99.9999`、实际 `ratio(f) = 4.959`**（E3），**不报错**。
这是一个真实的**静默失败**：random 路径（`get_unbalanced_scores(..., precise=0)`）直接调用它、
**不做可行性校验**。

---

## 7. 可行性闸门只装在 precise 路径（H100 Part F）

`test_gate.py:20-24` 自己算了一遍可行性：

```python
lower_bound_per_token = max(1, ceil_div(num_topk, num_experts_per_rank))           # :21
upper_bound_per_token = min(min(num_topk, num_ranks), int((num_ranks - 1) / ratio) + 1)  # :22
if lower_bound_per_token > upper_bound_per_token: continue                         # :23-24
```

- **下界** `max(1, ⌈k/E_s⌉)`：一个 token 至少要触及这么多个 rank（top-k 要摊到 `E_s` 个/rank 里）；
- **上界** `min(min(k, R), int((R-1)/ratio)+1)`：与 `generate_rank_count:54` 的
  `upper_bound_per_token` **同源**——即算法内部 clamp `a` 用的那个上限。

对 `test_gate.py` 的完整枚举矩阵（`EPR∈{1,4,8,16} × R∈{2,4,8,16,64,72} × k∈{1,2,4,6,8,9} ×
ratio∈{1,2,4}`）实测：**total=405、tested=379、skipped=26**（≈6.4%）。被跳过的全是
`E_s` 小、`ratio` 大的组合，例如 `R=2,EPR=1,k=2,ratio=2`（`lb=2 > ub=1`）。

**这个闸门只存在于 precise 的测试路径**里；一旦走 random（`precise=0`），就没有这层保护——
第 6 节的 `R=16,EPR=1,k=4,ratio=8` 正是被闸门挡在 precise 之外、却能在 random 里静默产出错误
比值的配置。**实践含义**：若下游直接用 `get_unbalanced_scores(..., precise=False)` 造流量，
必须自己保证 `ratio` 在可行域内，否则会拿到一个「看起来正常、实际远低于目标」的分布。

---

## 8. 随机性：拓扑由 RNG 决定，但结构不变式不依赖它（H100 Part G）

`generate_rank_count` 全程依赖 torch 全局 RNG（`:46` 的 `rand`、`:78`/`:107` 的 `randperm`）。
实测：同 seed（1234）两次得到**逐位相同**的表（`87ea6d91de14`），换 seed（9999）得到不同的表
（`0ab629d674fd`）——G1。但 `row_sum == k` 这类**结构不变式与 seed 无关**（G2）。
因此测试里若要复现特定拓扑必须固定 seed，而只关心「行和/可行性」时不必。

---

## 9. 一页纸小结

- `gate.py` 是 **MoE 路由夹具**：为 EP dispatch/combine 测试制造可控的 rank 级不均衡流量，
  分 **precise**（精确 ratio + 其余 rank 均衡）与 **random**（近似 ratio、不均衡）两条路线。
- `generate_topk_idx`（`:4-29`）是 **rank_count ↔ topk_idx 的精确逆映射**，且列方向**先按 rank
  分组**（H100 实测逐位可逆、组内有序）。
- `generate_rank_count`（`:32-113`）先量「每 token 触及几个不同 rank」`a` 并 clamp 到
  `int((R-1)/ratio)+1`，再用**循环重复的随机排列**把普通 rank 铺得近乎均匀（这是 precise 模式
  `inequality<1.02` 的机制）；`rank_count` 元素**可以是 0..k**（多次选到同一 rank）。
- `get_precise_unbalanced_scores` 用 **0.9 阈值**把「选中/背景」分数分开（`:127-136`）。
- `map_unbalanced_ratio_to_factor`（`:148-164`）在单标量 `factor` 上 20 轮二分：可行目标精确命中，
  但**大 R 时近似阈值开关**、**不可行目标静默饱和到上界**（H100 实测 `ratio=8→4.96`，不报错）。
- **可行性闸门（`:20-24`）只在 precise 测试路径**；random 路径无保护（完整矩阵 26/405 被跳过）。
- 全部 H100 真机验证 **ALL PASS**（`harness/exp_deepep_gate.{sh,py,out.txt}`；源码 sha256
  `gate.py=5372e456…`）。`gate.py` 只依赖 torch，按路径 exec 即可，**不需要** `deep_ep` 包
  （绕开 NCCL 运行期阻塞）。
