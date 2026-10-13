# DeepGEMM 的 legacy Triton MoE grouped GEMM：被现代 C++ 取代的「显式索引 gather」

> 前面第 10/11/12/13 篇拆的是 DeepGEMM **现代 C++** 的 MoE grouped GEMM：
> `k_grouped_*`（wgrad，沿收缩维 K 分组）与 `m_grouped_*`（dgrad，沿 M=token 分组）。
> 它们靠 `grouped_layout` + **BLOCK 对齐的连续段**取址，硬性要求 `M%128==0` / `k%128==0`。
> 但 DeepGEMM 里还留着一整套**没被任何一篇讲过**的东西：`deep_gemm/legacy/` 下的 **A100 时代 Triton 兜底 kernel**，
> 官方测试就是 `tests/test_legacy.py`。
>
> 这套 kernel 用的是**逐元素索引张量**（`m_indices` / `m_row_indices` / `k_indices`）实现**任意 gather**，
> 比现代 API 更灵活——它把「分组 GEMM」直接表达成「在 A 或 B 的某个操作数上做行索引」，而不是「按对齐段切」。
> 它是纯 Triton，**架构无关**，H100 能编能跑。本文把这套语义逐条讲清，并在 H100 上给真实数字。
>
> 结论先行：legacy 家族 = **dgrad 两个**（`m_grouped` / `a_fused_m`，后者 A 逐行 gather）+ **wgrad 两个**
> （`a_fused_k` / `b_fused_k`，分别 gather A 或 B）。它与现代 C++ API 是**同一数学、不同的「分组表示」**：
> legacy 用索引张量 + BLOCK 首行定组，现代用 `grouped_layout` + 强制对齐。

参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1）。

---

## 1. 为什么会有这一套，以及为什么本机反而能跑

`deep_gemm/__init__.py:113-120`：

```python
from . import testing
from . import utils
from .utils import *

# Legacy Triton kernels for A100
try:
    from . import legacy
except Exception as e:
    print(f'Failed to load legacy DeepGEMM A100 Triton kernels: {e}')
```

注释写得很直白：**Legacy Triton kernels for A100**。现代 DeepGEMM 的 C++ kernel（SM90 的 WGMMA、
SM100 的 UMMA）只在 Hopper/Blackwell 上有；对 **Ampere（A100, sm80）**，DeepGEMM 提供的是这套
纯 Triton 实现，作为老卡的后备。`deep_gemm/legacy/__init__.py:1-5` 把五个模块 `*` 导出：

```python
# All kernels may be deprecated in the future (or rewrite in TileLang)
from .m_grouped_gemm import *
from .a_fused_m_grouped_gemm import *
from .a_fused_k_grouped_gemm import *
from .b_fused_k_grouped_gemm import *
```

有意思的地方在于：**它是 Triton，所以跟具体架构解耦**。官方测试 `tests/test_legacy.py` 在 H100 上照样
能编能跑（Triton 后端会为 sm90 生成代码），本机实测也确实通过。也就是说，这套「只给 Ampere 用」的
kernel，反而是本机 H100 上可以**真跑真实代码、给真实数字**的一类。本文的实验全部走这条路。

---

## 2. 家族总览：四个 GEMM + 一份调参表

| 文件 | 方向 | 布局 | 分组表示 | gather 谁 |
|---|---|---|---|---|
| `m_grouped_gemm.py` | **dgrad**（M 轴分组，`nt`） | `a=[M,K]`, `b=[G,N,K]`, `d=[M,N]` | `m_indices[M]`（逐行组号） | 不 gather |
| `a_fused_m_grouped_gemm.py` | **dgrad 扩展**（`nt`） | 同上，但 A 是「源池」 | `m_indices` + `m_row_indices[M]` | **A 逐行** gather |
| `a_fused_k_grouped_gemm.py` | **wgrad**（K 轴分组，`tn`） | `a=[K_a,M]`, `b=[K_b,N]`, `d=[G,M,N]` | `k_indices[K_b]` + `k_start/k_end[G]` | **A** 按 k 索引 gather |
| `b_fused_k_grouped_gemm.py` | **wgrad**（K 轴分组，`tn`） | 同上 | `k_indices[K_a]` + `k_start/k_end[G]` | **B** 按 k 索引 gather |

`tn` 的意思是 A/B 都是 **MN-major**（`[K,M]`/`[K,N]`，K 是行），对应 wgrad 的 `D = AᵀB`；
`nt` 是 A 是 `[M,K]`、B 是 `[G,N,K]`（K-major），对应 dgrad 的 `D = A·Bᵀ`。

### 2.1 调参表：A100 的显存预算

`deep_gemm/legacy/tune_options.py:10-28` 定义候选 config，然后用**三条过滤**裁掉：

```python
# NOTES: we only consider A100 shared memory sizes here, as legacy kernels are only used for Ampere
_gemm_configs = list(filter(lambda x: get_config_smem_size(x) <= 166912, _gemm_configs))     # :23
_gemm_configs = list(filter(lambda x: x.kwargs['BLOCK_SIZE_M'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))  # :24
_gemm_configs = list(filter(lambda x: x.kwargs['BLOCK_SIZE_K'] <= get_mk_alignment_for_contiguous_layout(), _gemm_configs))  # :25
```

- 第 23 行：`get_config_smem_size`（`:5-7`）= `(BLOCK_M + BLOCK_N)·BLOCK_K·2·num_stages`，
  硬顶 **166912 B**（A100 的 shared memory 上限，比 Hopper 的 232448 小）；`elem_bytes` 默认 2（BF16），
  注释明说「FP8 kernel 不走 Triton」。
- 第 24/25 行：**BLOCK_M、BLOCK_K 都不能超过 `get_mk_alignment_for_contiguous_layout()`**（本机 H100 上为 128）。
  这条很关键——它是后面「BLOCK 首行定组」能成立的前提。

`m_grouped` 用 `get_m_grouped_gemm_configs()`（`:27`，只过滤 BLOCK_M），`k_grouped` 用
`get_k_grouped_gemm_configs()`（`:28`，只过滤 BLOCK_K）。两个 autotune 装饰器都 `key=[]`（全局调一次），
但 k-grouped 多了 `restore_value=['d_ptr']`：

```python
# a_fused_k_grouped_gemm.py:10 / b_fused_k_grouped_gemm.py:10
@triton.autotune(configs=get_k_grouped_gemm_configs(), key=[], restore_value=['d_ptr'])
```

原因见第 6 节：k-grouped 在 `ACC=True` 时**原地累加进 D**，autotune 试不同 config 时必须把 D 还原，
否则第二次试跑会从「已被上一次累加过的 D」开始。

---

## 3. dgrad 骨架：`m_grouped` 的「逐行组号 + BLOCK 首行定组」

`m_grouped_gemm.py` 的 kernel 是 dgrad 的标准形态：`d[m] = a[m] · b[group(m)]ᵀ`，
其中每组一块专家权重 `b[g]`。分组信息全部编码在 **`m_indices`（长度 M 的 int32，逐行给组号）**。

`m_grouped_gemm.py:31-40`：

```python
m_range = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
n_range = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
n_mask = (n_range < N)[None, :]

# Empty tokens
batch_id = tl.load(m_indices_ptr + pid_m * BLOCK_SIZE_M).to(tl.int64)      # :36
if batch_id < 0:                                                          # :37
    d_ptrs = d_ptr + m_range[:, None].to(tl.int64) * N + n_range[None, :]
    tl.store(d_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=...), mask=n_mask)
    return
```

两个要害：

1. **组号取自 block 的「首行」**（`:36` 的 `pid_m * BLOCK_SIZE_M`）——整块共享一个 `batch_id`。
   这只有在「每个 BLOCK_M 段恰好属于一个组」时才正确，也就是**每组必须按 `BLOCK_SIZE_M` 对齐**
   （这正是 §2.1 第 24 行过滤 `BLOCK_M <= align` 的用途：候选 BLOCK_M 永远 ≤ 对齐粒度 128）。
2. **首行为 `-1` 的整块被写零并提前返回**（`:37-40`）。所以 padding 行不一定要逐行标 `-1`，
   只要保证「每块第一行」是有效组号就行；反过来，若想让某整块输出为零，把它的首行标 `-1` 即可。

主循环 `:43-54` 就是 `a`（按 `m_range` 直读，K-major）乘 `b[batch_id]`（`IS_B_K_MAJOR` 决定 B 的走址：
K-major 时 k 步长 1、n 步长 K；否则相反，`:45-47`）。

**契约**（`:61-72`）里最硬的一条是 `assert a.size(0) % get_mk_alignment_for_contiguous_layout() == 0`
（`:72`）：M 必须能被 128 整除。这与现代 C++ API（第 13 篇）的「M 段对齐 BLOCK_M」是**同一个约束**，
只是现代版之所以需要它，是因为 `grouped_layout[m_block_idx*BLOCK_M]`（`scheduler/gemm.cuh:152`）用
M block 首行代表整块；legacy 版则需要它来保证 BLOCK 首行定组不跨组。

---

## 4. `a_fused_m`：A 操作数的逐行 gather（dgrad 的 scatter 版）

`a_fused_m_grouped_gemm.py` 比 `m_grouped` 多一个 `m_row_indices`——它把「输出行」映回「源激活行」：

```python
# a_fused_m_grouped_gemm.py:44-56
rows = tl.load(m_row_indices_ptr + m_range).to(tl.int64)      # :45  每行一个源行号
for k in range(0, K, BLOCK_SIZE_K):
    k_range = k.to(tl.int64) + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + rows[:, None] * K + k_range[None, :]     # :52  A 按 rows 取行
    b_ptrs = b_ptr + batch_id * K * N + ...                   # :53  B 仍按 batch_id 取组
    a = tl.load(a_ptrs, mask=(rows >= 0)[:, None] & k_mask[None, :], other=0.0)   # :54  rows<0 掩掉
```

于是语义变成：`d[row] = a[m_row_indices[row]] · b[m_indices[block_first(row)]]ᵀ`。
这正好是实现 **dgrad 的 scatter 步**（把每个专家的输出行散布回 token-major 布局）所需要的原语——
现代 C++ API 里没有等价物（那边只能做「M 轴连续段分组」，不能任意 gather）。

注意它仍然保留「BLOCK 首行定组」（`:38-42`），所以 `m_indices` 逐块恒定，只有 `m_row_indices` 逐行自由。
契约 `:74-76` 要求 `d.size(0)==m_indices.numel()==m_row_indices.numel()` 且 `%align==0`。
identity 的 `m_row_indices`（`arange`）时它**逐位等于** `m_grouped`——本机实测 max_abs_diff = 0。

---

## 5. wgrad：`a_fused_k` / `b_fused_k` 的 handle 与「命名之谜」

k-grouped 两个 kernel 走 **`tn`**（A/B 都是 `[K, M]`/`[K, N]`，K 是行），做的是
`d[g] = A[g]ᵀ B[g]` 的 wgrad。分组信息不再是「逐行组号」，而是一个 **handle 三元组**
`(k_indices, k_start, k_end)`：

- `k_start[g] / k_end[g]`：第 g 组在 K 轴上的**起止**（`k_start>=k_end` 表示空组）；
- `k_indices[·]`：**被 gather 那一侧**的行索引。

两者的差别只在主循环里「谁被 `rows` 索引」。`a_fused_k_grouped_gemm.py:51-59`：

```python
for k in range(k_start, k_end, BLOCK_SIZE_K):
    k_range = k + tl.arange(0, BLOCK_SIZE_K)
    rows = tl.load(k_indices_ptr + k_range).to(tl.int64)      # :53
    a_ptrs = a_ptr + m_range[:, None] + rows[None, :] * M     # :54  A 用 rows（gather A）
    b_ptrs = b_ptr + k_range[:, None].to(tl.int64) * N + n_range[None, :]   # :56  B 用 k_range
    a = tl.load(a_ptrs, mask=(rows >= 0)[None, :] & m_mask, other=0)        # :57  rows<0 掩 A
    b = tl.load(b_ptrs, mask=n_mask, other=0)
    acc = tl.dot(a, b, acc)
```

`b_fused_k_grouped_gemm.py:52-58` 恰好对调：A 用 `k_range`，**B 用 `rows`**（gather B）。

所以命名确实是字面意思：**`a_fused` = A 被索引融合（gather A），`b_fused` = B 被索引融合（gather B）**。
当 `k_indices = arange` 时两者退化到同一数学，本机实测 `a_fused_k` 与 `b_fused_k` 的输出**逐位相等**。
handle 的长度约束也对称：`a_fused_k` 要求 `k_indices.size(0)==b.size(0)`（`a:78`，索引的是 A，长度按
B 的 K），`b_fused_k` 要求 `k_indices.size(0)==a.size(0)`（`b:77`）。

契约里同一句话：**非 gather 侧的操作数 K 必须 `%align==0`**（`a_fused_k:80` 的
`b.size(0)%align==0` 与 `b_fused_k:79` 的 `a.size(0)%align==0`）。这就是 §2 表里「非 gather 侧」
那列的来源。

`k_indices` 里的 **`-1` 是一个行掩码**：`a_fused` 掩 A 行（`:57` 的 `(rows>=0)`），`b_fused` 掩 B 行
（`b:58`）。实测把某组前 32 行的 `k_indices` 置 `-1`，输出恰好等于参考里**扣掉这 32 行贡献**的结果。

---

## 6. 空组与原地累加

`k_start >= k_end` 即空组（某些专家这一段没有 token）。两个 k-grouped kernel 的处理一致
（`a_fused_k:43-47` = `b_fused_k:44-48`）：

```python
if k_start >= k_end:
    if not ACC:
        # 写零该组的 D 块
        ...
    return
```

- **`ACC=True`**（wgrad 的常态）：空组直接 return，**D 原样保留**（因为它本来就是累加目标）；
- **`ACC=False`**：空组把对应 `[M,N]` 块**写零**（覆盖掉旧的 `d`）。

`ACC=True` 时主循环结尾是 `acc += tl.load(d_ptrs)`（`a_fused_k:63-64`），也就是 **D 被读回再累加**——
这正是第 15 篇对 wgrad 性能归因里「SM90 `k_grouped_nt` 强制累积、D 被回读使流量翻倍」的同款现象，
只是这里发生在 Triton 里。也正因为 D 是原地累加，autotune 才需要 `restore_value=['d_ptr']`（§2.1）。

另外一个官方测试明确记录的限制（`tests/test_legacy.py:54-56`）：

```python
# Legacy Triton kernels do not mask per-group K tails inside the BLOCK_SIZE_K loop.
if use_psum_layout or k_alignment != 128 or not accumulate or out_dtype != torch.float:
    continue
```

**legacy kernel 不处理「每组 K 尾块」的掩码**——它对 `k in range(k_start, k_end, BLOCK_SIZE_K)` 不裁尾，
所以要求每组的 K 长度是 `BLOCK_SIZE_K` 的整数倍，配套地就要求 `k_alignment==128`、`ACC=True`、
输出 fp32。现代 C++ API 则能处理任意对齐段（含 PSUM/PSUM gap），这是两者能力上的实质差距。

---

## 7. H100 实测

脚本 `harness/exp_legacy_grouped_gemm.py`（driver `harness/exp_legacy_grouped_gemm.sh`，
输出 `harness/exp_legacy_grouped_gemm.out.txt`）在 `kernel_lab`（H100 sm90）上跑，**16/16 PASS**：

```
mk_alignment_for_contiguous_layout = 128

[A] m_grouped（dgrad 骨架）
  PASS  m-grouped vs torch 参考 (calc_diff)            calc_diff=0.000e+00
  PASS  padding 行严格写零                              pad rows=357, nonzeros=0
  PASS  有效行逐位等于参考
  PASS  M%align!=0 被拒 (m_grouped_gemm.py:72)
[B] a_fused_m（A 逐行 gather）
  PASS  identity m_row_indices 逐位等于 m_grouped       max_abs_diff=0.000e+00
  PASS  置换 m_row_indices 复现任意逐行 gather          calc_diff=1.786e-10
  PASS  block 首行 m_indices=-1 => 整块写零
  PASS  m_indices.numel()%align!=0 被拒 (a_fused_m:83)
[C] a_fused_k / b_fused_k（wgrad tn）
  PASS  a_fused_k(identity) 累积 vs AᵀB+C              calc_diff=7.327e-15
  PASS  b_fused_k(identity) == a_fused_k（同一数学）    max_abs=0.000e+00
  PASS  a_fused_k: 索引作用在 A                        calc_diff=7.772e-15
  PASS  b_fused_k: 索引作用在 B                        calc_diff=7.550e-15
  PASS  a_fused_k 的 -1 索引掩掉对应 A 行贡献           calc_diff=1.077e-14
  PASS  非 gather 侧 K%align!=0 被拒 (a_fused_k:82)
[D] 空组
  PASS  ACC=True：空组 D 原样保留                       max_abs=0.000e+00
  PASS  ACC=False：空组 D 写零                          nonzeros=0
```

几点解读：

- **[A] 的 padding 写零**：把 padding 激活清零（与官方 `tests/generators.py:406` 一致）后，
  有效行**逐位**等于参考、padding 行全零。若 padding 激活不清零，则 BLOCK 首行有效的块会把
  padding 行也算出非零——所以「padding 写零」是「padding 激活清零」+「首行定组」共同保证的，
  不是 kernel 逐行掩码出来的。
- **[B] 的 gather**：`calc_diff=1.8e-10` 不是逐位相等（参考用 fp64 累加），但 `max_abs=0.125`
  相对 bf16 的量级（`k=256` 时点积约几十）是舍入底噪。identity 那行则逐位为 0。
- **[C] 的命名坐实**：`a_fused_k` 的索引确实作用在 A、`b_fused_k` 作用在 B（`calc_diff ~1e-14`，
  即 fp32 累加对上 fp64 参考）。
- **[D] 空组**：`ACC=True` 完全不动 D（`max_abs=0`），`ACC=False` 写零（`nonzeros=0`）。

性能（Triton 在 sm90 后端，单次 launch 附近，`bench_kineto` 取 `_impl` kernel）：

| 方向 | 形状 | 耗时 | 吞吐 |
|---|---|---|---|
| m-grouped（dgrad） | M=4096, N=4096, K=4096（G=8） | 962 us | **142.9 TFLOPS** |
| k-grouped（wgrad） | K=16384, M=4096, N=7168（G=4） | 8934 us | **107.7 TFLOPS** |

对照官方 `tests/test_legacy.py` 在 H100 上跑出的更大形状：m-grouped 达 **285–361 TFLOPS**
（M≈3–4 万、N=6144、K=7168），k-grouped 稳定在 **104–109 TFLOPS**（K 很大、wgrad 偏访存）。
这与现代 C++ kernel 在同一台的量级一致（第 12 篇 wgrad ~103/106、dgrad ~449/543 TFLOPS；
SM90 BF16 主 kernel 峰值 801 TFLOPS）——**Triton 兜底版比 C++ 版慢是预期的**，它的价值是覆盖 A100
和老 API，而不是拼峰值。

---

## 8. 与现代 C++ API 的对照

| 维度 | legacy Triton（本文） | 现代 C++（第 12/13 篇） |
|---|---|---|
| 分组表示 | **索引张量**：`m_indices` / `m_row_indices` / `k_indices` | `grouped_layout`（contiguous）/ PSUM / masked |
| 定位一方 | **BLOCK 首行**（`m_indices[pid_m*BLOCK_M]`） | M block 首行（`grouped_layout[m_block_idx*BLOCK_M]`） |
| 对齐要求 | `M%128==0`（m）/ 非 gather 侧 `K%128==0`（k） | 同（128），SM100 为 256 |
| 任意 gather | **支持**（`m_row_indices` / `k_indices`，`-1` 掩码） | 不支持（只能连续段分组） |
| K 尾块 | **不裁尾**（要求 K 长度对齐） | 支持任意对齐段 |
| 布局 | `nt`（dgrad）/ `tn`（wgrad） | `nt`/`nn`/`tn`/`tt` 别名 + SF 多粒度 |
| 精度 | 仅 BF16 | BF16 / FP8 / FP4（block-scale） |
| 目标架构 | Ampere（A100），但 Triton 可移植 | SM90 / SM100 |
| autotune | Triton 候选 + `restore_value=['d_ptr']` | 静态 `get_best_config` 启发式（第 18 篇） |

一句话：**legacy 把「分组」表达为「索引 gather」，现代把「分组」表达为「对齐的连续段」。**
前者灵活（可任意散布、可跨组混合 A 的行），后者高效（TMA/多播/UMMA 都要求段对齐）。
`-1` 索引掩码在 legacy 里是「有效位」，在现代 API 里被 `grouped_layout=-1`（padding 行）
和 masked 布局的 `masked_m` 取代。

---

## 9. 小结

- `deep_gemm/legacy/` 是 DeepGEMM 给 **A100** 的纯 Triton 兜底：**dgrad 两个**（`m_grouped`、
  A 逐行 gather 的 `a_fused_m`）+ **wgrad 两个**（`a_fused_k`/`b_fused_k`，分别 gather A/B），
  官方测试 `tests/test_legacy.py` 覆盖它们。
- 它的分组表示是**逐元素索引张量**（`m_indices`/`m_row_indices`/`k_indices`），比现代 C++
  的 `grouped_layout` + 强制对齐更灵活，代价是不能用 TMA/multicast/UMMA，也**不能裁 K 尾块**。
- 两个贯穿始终的机制：**BLOCK 首行定组**（首行 `-1` 整块写零）与 **对齐要求**
  （`M%128==0` / 非 gather 侧 `K%128==0`）——与现代 C++ 是同一个数的对齐粒度。
- H100 上 16/16 PASS：identity `m_row_indices` 与 `m_grouped` 逐位相等、`a_fused_k`/`b_fused_k`
  逐位相等、空组语义与 `-1` 掩码全部复现；性能 m-grouped ~143 TFLOPS、k-grouped ~108 TFLOPS
  （官方更大形状 m-grouped 285–361），是「可移植兜底」而非「拼峰值」的量级。
