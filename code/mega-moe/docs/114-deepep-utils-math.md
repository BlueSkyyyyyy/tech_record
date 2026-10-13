# 114. DeepEP 的 MoE 语义工具层 `deep_ep/utils/math.py`

> 第 65 篇拆了 DeepEP `deep_ep/utils/` 里的 **参考实现层**（`refs.py`）、**流量生成层**
> （`gate.py`）、**计时基座层**（`testing.py`），却只顺带提了一句同目录的 `math.py`
> ——「有个和 DeepGEMM 同名但不同定义的 `calc_diff`」。本篇把这 103 行的 **`math.py`**
> 单独讲透：它虽叫 math，装的其实是 **DeepEP 测试与下游代码共用的一小套 MoE 语义词汇**——
> 判据（`calc_diff`）、参考量化器（`per_token_cast_to_fp8/back`）、**top-k 专家去重 oracle**
> （`inplace_unique`）、**group-limited 路由掩码**（`create_grouped_scores`）以及记账
> （`count_bytes`／`hash_tensor(s)`）。它与第 27 篇（DeepGEMM 参考量化器 `utils/math.py`）、
> 第 40 篇（DeepGEMM 判据 `testing/numeric.py`）正好成**跨库对照**。
>
> **本机可跑**：全部是纯 torch，且 `math.py` 只 `import torch`（不碰 `deep_ep` 包），
> 直接按文件路径 exec 真实源码即可——无需绕过 `check_nccl_so()`。H100 上逐条核验在
> `harness/exp_deepep_utils_math.{py,sh,out.txt}`（**ALL PASS**）。
>
> 源码坐标：`~/github/DeepEP` commit `93eb6eb`（v2.5.0）
> `deep_ep/utils/math.py:1-103`；对照 `~/github/DeepGEMM` commit `057ca59` 的
> `deep_gemm/testing/numeric.py:5-27`、`deep_gemm/utils/math.py:19-49`。

---

## 1. 它在测试栈里的位置

第 65 篇给过 DeepEP 测试栈的「规范层」三文件，`math.py` 是这条线**漏掉的第四个**：

| 文件 | 行数 | 角色 |
|---|---:|---|
| `deep_ep/utils/refs.py` | 243 | dispatch/combine 的**语义规范**（纯 torch） |
| `deep_ep/utils/gate.py` | 180 | 造指定 imbalance 的流量 |
| `deep_ep/utils/testing.py` | 239 | `bench`/`bench_kineto`/`parse_num_bytes` |
| **`deep_ep/utils/math.py`** | **103** | **判据 / 参考量化器 / 去重 oracle / 路由掩码 / 记账** |

`math.py` 的十个导出（行号即真实文件行）：

| 函数 | 行号 | 被谁用（公开仓库） |
|---|---|---|
| `calc_diff` | `:5-9` | `tests/ep/test_ep.py:9`、`tests/bucket/test_all_reduce.py:15` |
| `safe_div` | `:12-19` | `tests/ep/test_ep.py:11,260,273,294` |
| `ceil_div` / `align` | `:22-27` | `gate.py`、多个 test |
| `per_token_cast_to_fp8` | `:31-39` | `test_ep.py:10,97`、`tests/engram/test_engram.py:8,47` |
| `per_token_cast_back` | `:43-56` | `test_ep.py:10` |
| `inplace_unique` | `:59-71` | **公开仓库无调用点** |
| `create_grouped_scores` | `:74-79` | **公开仓库无调用点** |
| `hash_tensor` / `hash_tensors` | `:82-93` | **公开仓库无调用点** |
| `count_bytes` | `:96-103` | `test_ep.py:260,273` |

一个值得记下的事实：**`inplace_unique`、`create_grouped_scores`、`hash_tensor(s)` 在公开仓库里
没有任何调用点**（`rg` 只在 `math.py` 本身命中）。它们是为下游/私有训练栈准备的语义工具，与第 55 篇
拆过的 `common/probe.cuh`（公开版无调用点的休眠设施）同属一类——**不要因为「没人调」就跳过**，
它们恰好把某些 MoE 语义写成了可读的参考。

---

## 2. `calc_diff`：DeepEP 版与 DeepGEMM 版的差别只有「+1」

```python
def calc_diff(x, y):                       # math.py:5-9
    x, y = x.double() + 1, y.double() + 1  # :6   ← 先 +1，再升 fp64
    denominator = (x * x + y * y).sum()    # :7
    sim = 2 * (x * y).sum() / denominator  # :8
    return (1 - sim).item()                # :9
```

DeepGEMM 的版本（`testing/numeric.py:5-12`）几乎逐字相同，**只差一处**：

```python
def calc_diff(x, y):
    x, y = x.double(), y.double()          # 无 +1
    denominator = (x * x + y * y).sum()
    if denominator == 0:                   # ← 显式除零保护
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
```

三点「为什么」：

1. **代数恒等**：`1 - 2⟨x,y⟩/(‖x‖²+‖y‖²) ≡ ‖x−y‖²/(‖x‖²+‖y‖²)`。所以两种写法输出的是**同一个
   归一化平方误差**（第 40 篇已把 DeepGEMM 版等价性坐实）。H100 实测 A1/A1b：余弦式与平方误差式
   在 fp32/bf16 输入上完全一致。
2. **`+1` 的唯一作用是避开 0/0**：全零张量时 DeepEP 的 `x+1=1`，分母 `= 2N>0`，结果恰为 0；
   DeepGEMM 则靠 `if denominator == 0` 兜底。H100 实测 A3：两边都得 `0.0`，**DeepEP 版不会 NaN**。
   这就是它敢省掉 guard 的原因。
3. **代价：破坏尺度不变性**。`+1` 让判据对「整体缩放」敏感——`calc_diff(c·x, c·y)` 随 `c` 变；
   DeepGEMM 的无位移版本对缩放近似不变。H100 实测 A4：

   | 缩放 c | DeepEP | DeepGEMM |
   |---:|---:|---:|
   | 0.1 | 2.11e-4 | 6.87e-3 |
   | 1   | 3.59e-3 | 6.87e-3 |
   | 10  | 6.43e-3 | 6.87e-3 |
   | 100 | 6.83e-3 | 6.87e-3 |

   DeepEP 版 spread `6.6e-3`，DeepGEMM 版 spread `4.6e-9`（纯浮点噪声）。

**读博客时的实用结论**：两个库测试输出里都写「Diff: x.xxe-2」，但**含义不同**——
DeepEP 的是「+1 位移后的归一化平方误差」。二者可用
`calc_diff_dep(x,y) == calc_diff_dgemm((x+1).double, (y+1).double)` 精确互转（实测 A2 成立；
注意 `+1` 要在 double 域做，否则 fp32 的 `x+1` 会再引入一次舍入，实测 A2 首版就因此差到
`8.93e-1 vs 9.87e-1`）。A5 另证 `calc_diff≥0` 恒成立（`2ab≤a²+b²`）。

---

## 3. `per_token_cast_to_fp8/back`：**未打包、未 UE8M0** 的 per-128 参考量化器

```python
@torch.compile(dynamic=True)                          # :30
def per_token_cast_to_fp8(x):                          # :31-39
    assert x.dim() == 2
    m, n = x.shape
    aligned_n = align(n, 128)                          # :34  gran 硬编码 128
    x_padded = F.pad(x, (0, aligned_n - n), value=0)   # :35
    x_padded_view = x_padded.view(m, -1, 128)          # :36
    x_amax = x_padded_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)  # :37
    return (x_padded_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
               .view(m, aligned_n)[:, :n].contiguous(),            # :38 切片回 n
           (x_amax / 448.0).view(m, -1)                            # :39 返回 fp32 scale
```

它与 DeepGEMM 的参考量化器 `deep_gemm/utils/math.py:26-49` **同名，但契约不一样**：

| 维度 | DeepEP `math.py:31-39` | DeepGEMM `utils/math.py:26-49` |
|---|---|---|
| gran_k | 硬编码 **128** | 参数 `gran_k=128` |
| UE8M0 上取整 | **无** | `use_ue8m0` 开关 |
| scale 返回 | fp32 `[m, ceil(n/128)]`，**不打包** | 可选 `use_packed_ue8m0` → int32（每 4 个 1×128 打包） |
| 乘子写法 | `x * (448.0/amax)` | `x * (1.0/sf)`，`sf=amax/448` |
| 编译 | 带 `@torch.compile(dynamic=True)` | 纯 eager |

H100 实测（B 部分）：

- **B2**：**fp32 scale 逐位相同**，但 `448.0/amax` 与 `1.0/(amax/448)` 是两个不同的浮点算式，
  **10/6400（0.156%）** 的 fp8 元素落在相邻码点。也就是说「同一数学、两种写法」在 fp8 舍入边界上
  并非逐位一致——**跨库对拍时别默认它们 bitwise 相同**。
- **B3**：`per_token_cast_back` 的 **int scale 分支**（`:51-53`）用 `view(uint8) << 23` 把
  DeepGEMM 的 packed UE8M0 int32 **重解释**回 fp32 scale。实测「packed-int 解码 == DeepGEMM 的
  UE8M0 fp32 scale」逐位成立，且 `cast_back(int) == cast_back(fp32)`，**两个库的 SF 布局可互喂**。
  注意该分支要求 scale 数为 4 的倍数（DeepGEMM 打包前补 1.0 会把数量对齐到 4），否则
  `view(float)[m, -1, 1]` 与激活的 `[m, ceil(n/128), 128]` 形状对不上——实验首版 `n=100`（1 个
  scale）就撞上了这个广播错误。
- **B1**：round-trip 回 bf16 relerr `2.63e-2`（fp8 per-128 量化层）。
- **B5**：DeepEP 版 scale **不是** 2 的幂（无 UE8M0），与 kernel 侧 packed UE8M0 是有意区分的。

---

## 4. `inplace_unique`：top-k 专家去重 oracle（combine 归约的语义规范）

```python
def inplace_unique(x, num_slots):                       # :59-71
    assert x.dim() == 2
    mask = x < 0                                        # :61
    x_padded = x.masked_fill(mask, num_slots)           # :62  -1 丢到“垃圾桶”槽
    bin_count = zeros((x.size(0), num_slots + 1))       # :63
    bin_count.scatter_add_(1, x_padded, ones_like(x_padded))  # :64  逐专家计数
    bin_count = bin_count[:, :num_slots]                # :65
    sorted_bin_count, sorted_bin_idx = sort(bin_count, desc)  # :66
    sorted_bin_idx.masked_fill_(sorted_bin_count == 0, -1)    # :67  出现 0 次的专家置 -1
    sorted_bin_idx = sort(sorted_bin_idx, desc).values        # :68
    x[:, :].fill_(-1)                                   # :69
    x[:, :min(num_slots, x.size(1))] = sorted_bin_idx[:, :vl] # :70-71
```

它解决的正是 **combine 归约**要面对的问题：一行 `top-k` 专家号里可能有**重复**
（同一 token 的多个 top-k 命中了同一专家，或 DeepEP 的 `-1` padding），归约前必须去重。
第 58 篇拆过 DeepGEMM 侧的对应实现 `combine_utils.cuh` 的 `compute_topk_slots` + `ptx::deduplicate`
（`__match_any_sync` + `__ffs`）；`inplace_unique` 就是它的 **torch 参考 oracle**。

H100 实测（C 部分）：

- **C1**：与独立 Python oracle（`sorted(set(...), reverse=True)` + `-1` 填充）**逐位相等**；
  每行有效专家集合不变（仅去重/排序）；`-1` 全在尾部。
- **C2**（**一个真发现**）：`:66` 那次「按 count 降序」的排序，其实**对最终结果没有任何影响**。
  因为 `:67` 的掩码是**按 bin 下标**做的（`sorted_bin_count==0`），与排序顺序无关；而 `:68`
  又立刻**按专家号的值**重新降序排了一遍。我把它删掉、直接对 `arange(num_slots)` 掩码再按值排序，
  输出与真实实现**逐位相同**。`:66` 是死代码（或纯为可读性）。
- **C3**：DeepEP 输出是**降序**（首列 ≥ 末列），而 DeepGEMM `compute_topk_slots` 用 `__ffs`
  从低位挤到前，是**升序**。两者集合相同、顺序相反——**归约结果不受影响**（累加/去重与顺序无关），
  但这也说明「去重后的槽序」只是实现细节，不是语义。

---

## 5. `create_grouped_scores`：group-limited 路由的专家掩码

```python
def create_grouped_scores(scores, group_idx, num_groups):  # :74-79
    num_tokens, num_experts = scores.shape
    scores = scores.view(num_tokens, num_groups, -1)       # :76  专家按 group 分块
    mask = zeros((num_tokens, num_groups), bool)           # :77
    mask = mask.scatter_(1, group_idx, True).unsqueeze(-1).expand_as(scores)  # :78
    return (scores * mask).view(num_tokens, num_experts)   # :79
```

这是 **DeepSeek-V2/V3 的 group-limited（device-limited）路由**里的**掩码步**：
先把专家按 `num_groups` 分块，`group_idx` 给出每 token 选中的若干 group（通常是按「组内 top-2
之和」选出的 top 组），然后把**未选中 group 的整个专家块清零**，再在剩余专家里做 top-k。
`group_idx` 形状 `[num_tokens, num_selected_groups]`，`scatter_(1, ...)` 把对应的组位置为 True。

H100 实测（D 部分）：独立向量化复刻（`(scores.view(T,G,eg) * sel.unsqueeze(-1)).view(T,E)`）
与真实实现**逐位相等**；每 token 恰好保留「选中组数 × 每组专家数」个非零专家（实测 2 组 × 8 = 16）。

> 与 Mega Gate 的关系：第 05/13 篇的 **Mega Gate** 做的是**扁平 top-k**（UMMA 打分 → 排序网络 →
> top-k，`epilogue:88-152`），**不含 group 约束**；`create_grouped_scores` 展示的是它上游的
> 「先选组、再掩码」这一步。两段合起来才是完整的 DeepSeek 路由。

---

## 6. `safe_div`、`hash_tensor(s)`、`count_bytes`

- **`safe_div(a,b)`（`:12-19`）**：`b==0` 时若 `a==0` 返回 `0`，否则**重抛** `ZeroDivisionError`。
  仅在 `test_ep.py` 里用来算「每 token 字节数」（`:260/273/294`）。实测 E1：`(0,0)→0`、
  `(3,0)` 抛异常、`(6,3)→2`。
- **`hash_tensor(t)`（`:82-83`）**：`t.view(torch.int).sum()`，即把张量按 int32 重解释后求和。
  对**元素顺序不敏感**（和），但**对 dtype/layout 敏感**（重解释的是原始字节）。
  `hash_tensors(*ts)`（`:86-93`）对嵌套 tuple/list 递归，用 **XOR** 合并各张量哈希——
  于是「同一个张量 xor 两次 = 0」，天然可做「集合式」校验。实测 E3 逐条成立。
- **`count_bytes(*ts)`（`:96-103`）**：对嵌套结构递归累加 `numel*element_size`，跳过 `None`。
  **与 DeepGEMM `testing/numeric.py:14-21` 逐字相同**（实测 E2 两边同为 `344`）。
  一个共同前提：两者都**不检查类型**——传进非张量（如字符串）会在 `t.numel()` 处崩
  （实验首版踩到）。即它们假设「非 None 即张量」。

---

## 7. H100 实测汇总

脚本 `harness/exp_deepep_utils_math.{py,sh}`，原始输出 `harness/exp_deepep_utils_math.out.txt`。
**按文件路径 exec 真实源码**，无需绕过 `check_nccl_so`（`math.py` 只 import torch）；打印 sha256：

```
deep_ep/utils/math.py           sha256=ed0dfb4bf9893769…   （与第 65 篇一致）
deep_gemm/testing/numeric.py    sha256=de92b73f28acb3c5…
deep_gemm/utils/math.py         sha256=9cdecd0c0a0a8b3c…
torch 2.12.0a0  cuda 13.2  H100 80GB HBM3
```

| 部分 | 检查 | 结果 |
|---|---|---|
| A. `calc_diff` | 余弦式≡平方误差式；`DeepEP(x,y)==DeepGEMM((x+1),(y+1))`；`!=DeepGEMM(x,y)`；全零两边 `0.0`；缩放 spread `6.6e-3` vs `4.6e-9`；200 组随机 ≥0 | 11/11 PASS |
| B. 参考量化器 | shape/padding 切片；round-trip `2.63e-2`；scale 逐位相同但 fp8 `10/6400` 不同；packed-int 解码≡UE8M0 fp32；int 分支≡fp32 分支；空张量早退；无 UE8M0 | 9/9 PASS |
| C. `inplace_unique` | ≡独立 oracle；集合不变；`-1` 在尾；`:66` 排序死代码；降序（对照 DeepGEMM 升序） | 5/5 PASS |
| D. `create_grouped_scores` | ≡向量化复刻；每 token 保留 16 专家 | 2/2 PASS |
| E. `safe_div`/`hash`/`count_bytes` | `0/0→0`、`3/0` 抛、`6/3→2`；int 视图求和；XOR 递归；`count_bytes`≡DeepGEMM | 6/6 PASS |

**总判定 ALL PASS。**

---

## 8. 边界与结论

- **能跑/能核**：`math.py` 全部十个函数是纯 torch/标量运算，**架构无关**，H100 真机逐条对拍。
- **不能核**：`inplace_unique`/`create_grouped_scores` 在**公开仓库无调用点**，故只能核「参考语义」
  是否自洽，无法核「kernel 是否用同样的顺序/掩码」（去重顺序、group 选择规则由上游/私有栈决定）。
- **跨库结论**：`math.py` 与 DeepGEMM 的三个同名函数构成三组「同名不同契约」，这是本专题反复出现的
  主题（第 65 篇的 `testing.py`、第 40 篇的 `calc_diff`、第 27 篇的量化器）：
  ① `calc_diff` 差一个 `+1`（避 0/0，但破尺度不变）；
  ② `per_token_cast_to_fp8` 不打包、不 UE8M0、gran 硬编码 128；
  ③ `count_bytes` 两库逐字相同。
  读跨库数字时把这三条记牢，能避免把「定义不同」误当成「对不上」。

> 参考链接：DeepEP 仓库 <https://github.com/deepseek-ai/DeepEP>（本机 commit `93eb6eb`）；
> DeepGEMM 仓库 <https://github.com/deepseek-ai/DeepGEMM>（本机 commit `057ca59`）。
> 相关篇目：第 65 篇（`utils/refs.py`/`gate.py`/`testing.py`）、第 27 篇（DeepGEMM 参考量化器）、
> 第 40 篇（DeepGEMM 判据 `testing/numeric.py`）、第 58 篇（combine 归约与 top-k 去重）。
