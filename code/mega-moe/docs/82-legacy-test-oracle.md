# DeepGEMM legacy Triton grouped GEMM 的验收测试即正确性规范（`tests/test_legacy.py`）

> 第 29 篇（P29-1）拆了 `deep_gemm/legacy/` 下四个 A100 时代 Triton kernel 的**实现**；
> 第 40/41/43 篇拆了测试栈的三层地基（计时/判对基座、操作数生成器、逐位断言辅助）；
> 第 41/47/50/57/64/66/67/68/69/70/71 篇分别拆了 Mega MoE / Mega Gate / MQA logits / Mega mHC /
> 稠密 GEMM / einsum / SF layout / HC prenorm / lazy init / 局部域等家族的**验收测试即正确性规范**。
> 至此 `tests/` 下**唯一还没被单独拆过的顶层测试文件**只剩 `tests/test_legacy.py`。
>
> 本篇把它当「legacy Triton grouped GEMM 算什么对」的**可执行规格**来拆：它只有 95 行、
> 两个顶层函数、**没有任何 arch 门控**（H100/sm90 上整份真机跑通），却精确定义了 legacy 家族的
> 分组语义（BLOCK 首行定组 / `-1` 行掩码 / 空组）、`nt` vs `nn` 的主序别名薄壳、`expand`
> （`a_fused_m`）的逐行 gather、以及 k-grouped 的唯一合法工作区间（accumulate + FP32 输出 +
> `k_alignment==128` + 非 PSUM）。
>
> **本文的实验在 H100(sm90) 上直接 import 真实测试模块、驱动真实测试体（只把 `bench_kineto`
> 打桩，形状一个没缩）**，另做闸门/别名/expand/判据利齿的独立复核，共 **19 PASS / 0 FAIL**。
> legacy 是纯 Triton，**架构无关**，本机就是它能跑的机器（`deep_gemm/legacy` 的注释即
> 「Legacy Triton kernels for A100」，见 `deep_gemm/__init__.py:115`）。

参考版本：`~/github/DeepGEMM` 公开版 commit `057ca59`（v2.8.1），行号均已核对。实验见
`harness/exp_legacy_test_oracle.{py,sh,out.txt}`。

---

## 1. 这份测试在整条测试栈里的位置

DeepGEMM 的测试栈可以分三层（见第 40/41/43 篇）：**基座**（`deep_gemm/testing/` 的
`bench`/`bench_kineto`/`calc_diff`）、**操作数生成器**（`tests/generators.py`）、**逐位断言辅助**
（`tests/utils.py`）。最上面是各家族的**顶层规格文件**：`test_fp8_fp4.py`/`test_bf16.py`（稠密
GEMM）、`test_einsum.py`、`test_attention.py`、`test_mega_{moe,gate,mhc}.py`、`test_layout.py`、
`test_hyperconnection.py`、`test_lazy_init.py`、`test_locality_domain.py`……以及本篇的
`test_legacy.py`。

`test_legacy.py` 的特点是：

- 它是**唯一测试 legacy（A100 Triton 兜底）家族**的文件——现代 C++ 的 `m_grouped`/`k_grouped`
  测试在 `test_fp8_fp4.py`/`test_bf16.py` 里，legacy 则单独一份。
- 它是**唯一没有任何 arch 门控**的算子测试——没有 `@test_filter`、源里根本不出现
  `get_arch_major`（实测 Part A1）。原因是它测的 Triton kernel 与架构解耦。
- 它**只测 bf16**（两个函数都硬编码 `torch.bfloat16` 与 `use_bf16=True`）——因为 FP8 kernel
  明确「不走 Triton」（`deep_gemm/legacy/tune_options.py:5-6` 注释：
  `# NOTES: FP8 kernels will not use Triton, so by default we assume BF16 kernels`）。

> 与第 64 篇（稠密 GEMM 规格）的 SP1D2D、第 66 篇（einsum 规格）的四分派相比，
> `test_legacy.py` 的**规格面窄但语义独特**：它约束的不是「同一数学的不同精度」，
> 而是「同一数学的不同**分组表示**」（逐元素索引张量 vs 现代 `grouped_layout`）。

---

## 2. 顶层结构（95 行）

导入（`tests/test_legacy.py:1-12`）只有三件事：`torch`、`random`、`deep_gemm`、
从 `deep_gemm.testing` 取 `bench_kineto/calc_diff/count_bytes`、从 `generators` 取
`enumerate_m_grouped_contiguous`/`enumerate_k_grouped_contiguous`（枚举）与
`generate_m_grouped_contiguous`/`generate_k_grouped_contiguous`（造操作数）。

两个顶层函数 `test_m_grouped_gemm_contiguous_tl:14` 与 `test_k_grouped_gemm_contiguous_tl:50`，
`__main__:87-95` 只做 `torch.manual_seed(0)`、`random.seed(0)`、打印 `deep_gemm.__path__`，
然后按序调这两个函数。**没有 `unittest`、没有 `@test_filter`、没有 arch 判断**——
H100 直接整份跑（实测 Part A1/A2）。

---

## 3. m-grouped 测试体逐段精读（`:14-47`）

### 3.1 枚举 32 个配置，但其中 16 个是重复

`for` 循环解包 10 元组（`:16`）：

```python
for _, _, num_groups, expected_m_per_group, n, k, major_a, major_b, _, _ in \
        enumerate_m_grouped_contiguous(torch.bfloat16):
```

`enumerate_m_grouped_contiguous`（`tests/generators.py:194-215`）的轴：
`m_group_list = [(4, 8192), (8, 4096)]`（`:196`）× `n_k_list` 4 个（`:197`）×
`major_a` 锁 `KMajor`、`major_b ∈ {KMajor, MNMajor}`（`:207`，`get_major_ab(False, ...)`）→
`2 × 4 × 2 = 16`。再乘 `get_psum_layout_usage()` 返回的 `(True, False)`（`generators.py:118-119`），
**枚举器给 32 条**。

但测试体**不解包、也不透传第 9 个字段 `use_psum_layout`**（`_` 丢弃），而调用
`generate_m_grouped_contiguous(...)` 时只传 6 个位置参数 + `use_bf16=True`（`:22`）——
`generate_m_grouped_contiguous` 的 `use_psum_layout` 默认 `False`（`generators.py:381`）。
于是 32 条里 `(psum=True, psum=False)` 两两生成**完全相同的操作数**：**16 个唯一配置 + 16 个重复**
（实测 Part B1/B3，打印默认值确为 `False`）。这是「枚举器比测试体信息更全」的一个无害不对称。

| 层次 | 数量 | 来源 |
|---|---|---|
| 枚举配置 | 32 | `2 (m_group) × 4 (n_k) × 2 (major_b) × 2 (psum)` |
| 唯一配置 | 16 | 测试只用了 16 个（psum 轴未透传） |
| 实际发射 kernel | **128** | 每配置内层 `expand(2) × test_alias(2) = 4` |

### 3.2 别名与 expand 的二维矩阵

内层两个布尔循环（`:20-21`）`expand ∈ (False, True)` × `test_alias ∈ (False, True)`：

- `func_name`（`:23`）：非别名恒 `'nt'`；别名用 `major_opt.lower()`——`major_opt` 由
  `major_a`/`major_b` 是否 K-major 拼出（`:17-18`，`'NT'` 或 `'NN'`），故别名分支会去调
  `m_grouped_bf16_gemm_nn_contiguous_tl`（当 `major_b` 是 MN-major）。
- `expand=True` 时把函数名前缀换成 `a_fused_`（`:23`），并额外造一个**恒等** `m_row_indices`
  （`:29`）传给 `a_fused_m` 家族（`:30`）。

### 3.3 `test_alias` 是「主序别名」的薄壳（`:24-27`）

```python
if test_alias:
    assert major_a.is_k_major()
    b = b if major_b.is_k_major() else b.mT
    assert a[0].is_contiguous() and b[0].is_contiguous()
```

三个事实：

1. **A 必须 K-major**（`:25`）——`enumerate_m_grouped_contiguous` 用 `get_major_ab(False, ...)`
   锁死 `major_a=KMajor`（`generators.py:207`），所以这里的断言恒真；它其实在编码
   legacy `nt` 入口的契约。
2. `major_b` 为 MN-major 时把 `b` 换成 `b.mT`（`:26`）。generator 在 `use_bf16` 分支里
   已经把 b 造成 `b if major_b.is_k_major() else b.mT.contiguous().mT`（`generators.py:411`），
   形状仍是 `[G,N,K]` 但 MN-major；测试再 `.mT` 得到 `[G,K,N]`，交给 `nn` 入口，
   而 `nn` 内部再来一次 `.mT`（见下），**两次 `.mT` 抵消**。
3. `a[0]`/`b[0]` 连续断言（`:27`）确认「沿用 K-major 存储」。

`m_grouped_bf16_gemm_nn_contiguous_tl` 本体（`deep_gemm/legacy/m_grouped_gemm.py:82-84`）
只有一行：`m_grouped_bf16_gemm_nt_contiguous_tl(a, b.mT, d, m_indices)`。所以
`nn` = 「把 B 转置后转调 `nt`」，与 `gemm.hpp` 里现代 API 的 `nn` 完全同构（第 36 篇）。

实测 Part E2：同一操作数下 `nn` 与 `nt` 输出 **逐位相等**（`calc_diff=0.000e+00`）——
「别名薄壳」在真实 Triton 代码上也成立。

### 3.4 `expand` 是「A 逐行 gather」的恒等实例（`:28-32`）

```python
if expand:
    m_row_indices = torch.arange(0, m, dtype=torch.int32, device='cuda')
    getattr(deep_gemm.legacy, func_name)(a, b, d, (m_indices, m_row_indices))
else:
    getattr(deep_gemm.legacy, func_name)(a, b, d, m_indices)
```

`a_fused_m_grouped_bf16_gemm_nt_contiguous_tl`（`a_fused_m_grouped_gemm.py:64-87`）比
`m_grouped` 多一个 `m_row_indices`，kernel 里用它去 gather A 的行
（`rows = tl.load(m_row_indices_ptr + m_range)`，`:45`），并按 `(rows >= 0)` 做行掩码
（`:54`）。这里传**恒等** `arange(0, m)`，所以 gather 退化成原样读取——`expand` 分支实际
只在验证「A 带行索引这条路径的恒等实例与 `m_grouped` 等价」。实测 Part E3 逐位相等
（`calc_diff=0.000e+00`）。

> **一个不容易注意的分支差异**：`a_fused_m` 的填充行**不受 `m_indices==-1` 保护**（`a_fused_m`
> impl 没有「首行 `batch_id<0` 就整块写零」的早退，`m_grouped_gemm.py:36-40` 有）。
> 传恒等 `m_row_indices` 时它仍会去 gather 填充行的 A——只是 generator 已把 A 的填充行清零
> （`generators.py:405`，注释「Zero BF16 padding so quantized SFA padding is regular」），
> 结果恰为 0。若把 `m_row_indices` 换成任意非负置换，填充行会被算出**非零**结果（实测 Part E4/E5：
> 639 个填充行写出约 100 万个非零元素）。这正是 `-1` 行掩码的语义边界。

### 3.5 判据与 perf（`:33-46`）

```python
d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(d), d)
diff = calc_diff(d, ref_d)
assert diff < 0.001, f'{m=}, {n=}, {k=}, {major_opt}, {diff:.5f}, alias={test_alias}'
```

- `:33` 先把 `-1` 填充行强制归零再比——把「kernel 是否零填充」这一项从判据里摘出去，
  只比有效行（generator 的 `ref_d` 填充行本就是 0，`generators.py:407`）。
- `:34-35` 判据是 `calc_diff < 0.001`（bf16 阈值，与 `QuantConfig.max_diff` 的 legacy 默认
  一致，`generators.py:69-74`）。
- `:42-46` 是**测试自带的 perf 路径**：`bench_kineto(func, 'm_grouped_bf16_gemm_contiguous_tl_impl')`，
  FLOP 口径 `2·m·n·k`（**不含分组数**，因为每个 token 只乘自己那组的 `[N,K]`），带宽用
  `count_bytes(a,b,d)`。

被 `calc_diff` 判据「放行」的两类错误（实测 Part F）：**单行归零**（`calc_diff=1.5e-05`，
仍过阈值——`calc_diff` 是全局归一化），以及**全局同缩放**（`d*0.999` → `calc_diff=5.0e-07`，
几乎不变）。只有逐位比较抓得住 1-ulp（Part F3）。这说明 `test_legacy.py` 的 `calc_diff<0.001`
是「松判据」，真正把主序/分组语义钉死的是 Part E2/E3 那类逐位等价（那些是本实验补的，
测试体本身依赖 generator 的 `ref_d` 造得足够正确）。

---

## 4. k-grouped 测试体逐段精读（`:50-84`）

### 4.1 13 元组解包（`:52-53`）

```python
for num_groups, m, n, major_a, major_b, _, aligned_ks_cpu, expected_k_per_group, _, \
        k_alignment, use_psum_layout, accumulate, out_dtype in \
        enumerate_k_grouped_contiguous(torch.bfloat16):
```

`enumerate_k_grouped_contiguous`（`generators.py:229-263`）对 bf16 的轴：`sf_layout_list`
在 arch9 只有 `[(128,128)]`（`:231`）、`psum_list` 在 arch9 只有 `(False,)`（`:244`）、
`cd_options` 在 arch9 只有 `[(True, torch.float)]`（`:246`）、外层 7 个形状
（`:251-256`，含 `(8, 768, 2048, 128)` 这条「很多小组、压测 SM90 原地改 tensormap」的
注释）。故 **arch9 = 7 条**（实测 Part B2）。`major_a/major_b` 对 bf16 恒为
`(MNMajor, MNMajor)`（`:236-240`），于是 `major_opt='TN'`，函数名恒走
`*_fused_k_grouped_bf16_gemm_tn_contiguous_tl`。

### 4.2 跳过闸门：arch9 是 no-op，arch10 才拦（`:54-56`）

```python
# Legacy Triton kernels do not mask per-group K tails inside the BLOCK_SIZE_K loop.
if use_psum_layout or k_alignment != 128 or not accumulate or out_dtype != torch.float:
    continue
```

注释即原因：legacy Triton 在 `BLOCK_SIZE_K` 循环内**不裁每组 K 的尾块**——所以要求每组
K 对齐到 `BLOCK_SIZE_K`（128），并要求输出是 fp32 + 累积（`ACC=True`），因为它靠
「先把 D 读回来再加」实现累积（见 §5）。四个闸门因子：

| 因子 | 为什么 legacy 不行 |
|---|---|
| `use_psum_layout` | PSUM 的逻辑 K ≠ 物理 K，尾块会跨组读到奇偶 |
| `k_alignment != 128` | K 尾块不裁，`BLOCK_SIZE_K=128` 必须整除每组 K |
| `not accumulate` | 空组时 `ACC=False` 会把 D 写零；且 legacy 的累积语义靠读回 D |
| `out_dtype != float` | bf16 输出会把「读回 D 再累加」的精度稀释成 bf16 |

实测（Part D）：

- **arch9**：真实枚举 7 条，闸门后保留 **7 条（跳过 0）**——闸门是 no-op。
- **伪造 arch10**（临时把 `generators.get_arch_major` 改成 `10` 再枚举）：126 条，
  闸门后只保留 **7 条（跳过 119）**。拦截原因直方图显示 `psum` / `k_align` / `no_acc` /
  `out_dtype` 四因子都以多种组合出现，只有「四因子全不触发」的 7 条（`k_alignment=128`
  且 accumulate、`out_dtype=float`、非 PSUM）通过。**这正说明这份测试同时兼任 legacy
  kernel 的「可用域」声明**：SM100 上放开的那些配置，legacy 明确声明不支持、直接跳过。

### 4.3 索引张量与物理分组（`:61-70`）

```python
for fused_operand in ('a', 'b'):
    k, a, b, c, d, ref_d, _, _ = generate_k_grouped_contiguous(
        num_groups, m, n, major_a, major_b, aligned_ks_cpu, use_ue8m0=False, use_bf16=True)
    func_name = f"{fused_operand}_fused_k_grouped_bf16_gemm_{major_opt.lower()}_contiguous_tl"
    k_indices = torch.arange(0, k, dtype=torch.int32, device='cuda')
    k_start = torch.empty(len(aligned_ks_cpu), ...); k_end = torch.empty(...)
    for i, group_k in enumerate(aligned_ks_cpu):
        k_start[i] = k_end[i-1] if i > 0 else 0
        k_end[i] = k_start[i] + group_k
    getattr(deep_gemm.legacy, func_name)(a, b, c, (k_indices, k_start, k_end), True)
```

- 传 `aligned_ks_cpu`（**物理/对齐后的**每组 K）而不是逻辑 K，正是 §4.2 的推论：
  在「不裁尾块」的前提下，只能按对齐后的 K 切。
- `k_indices = arange(0, k)` 是**恒等**映射——gather 退化为原样。这与 `m_grouped` 的
  `expand` 分支同构（都用恒等索引）：测试只验证「索引路径的恒等实例」。
- `k_start/k_end` 由 `aligned_ks_cpu` 前缀和构造（`:67-69`），是 kernel 里
  `pid_b` 查的组边界（`a_fused_k_grouped_gemm.py:41-42`）。
- 最后位置参数 `True` 是 `ACC`（`:70`），对应 §4.2 的 `accumulate`。
- `for fused_operand in ('a', 'b')`：分别测「索引 A」的 `a_fused_k`（`a` 按行 gather，
  `a_ptrs = a_ptr + m_range[:,None] + rows[None,:]*M`，`:54`）与「索引 B」的 `b_fused_k`
  （`b_ptrs = b_ptr + rows[:,None]*N + n_range`，`:56`）——两者是**镜像**的 gather 侧。
- 数据摆放：generator 先把 A/B/K 全置零（`generators.py:576-577`），再对每组把逻辑 K 行
  填在**物理块的开头**（`generators.py:585`：`a[start:end] = ...`，其中 `start = end - group_k`
  由 `logical_group_ends` 与 `host_ks` 前缀和给出，`generators.py:567-570`），因此「按对齐块
  迭代 + 恒等索引」时，逻辑 K 之外的 padding 行读到 0 → 结果正确。**这就是「不裁尾块也能对」
  的底层保证**：不是不裁，而是靠把 gap 置零让不裁也安全。

### 4.4 判据与 perf（`:71-83`）

`calc_diff(c, ref_d) < 0.001`，参考 `ref_d[i] = a_groupᵀ @ b_group + c[i]`（生成器
`generators.py:587`）。perf 路径用 `bench_kineto`，FLOP 口径 `2·m·n·k`（`k` 是**总** K，
含所有组），带宽 `count_bytes(a,b,d)`。

---

## 5. 测试体与 legacy kernel 本体的接口对应

| 测试体约束 | kernel 侧落点 | 说明 |
|---|---|---|
| BLOCK 首行定组 | `m_grouped_gemm.py:36` `batch_id = m_indices_ptr[pid_m*BLOCK_SIZE_M]` | 整块用首行组号；`a_fused_m` 同 `:38` |
| `-1` 行 → 整块写零 | `m_grouped_gemm.py:37-40` | `m_grouped`/`a_fused_m` 都有此早退；`a_fused_m` 的填充行只在行索引为 `-1` 时才掩（`:54`） |
| 空组（`k_start>=k_end`） | `a_fused_k_grouped_gemm.py:43-47` / `b_fused_k...:44-48` | `ACC=True` → 保留 D；`ACC=False` → 写零 |
| 累积 `D += AᵀB` | `a_fused_k...:63-64` `acc += tl.load(d_ptrs)` | 输出必 fp32（闸门保证） |
| 主序 `nt`/`nn` | `m_grouped_gemm.py:61-84` | `nn`=`nt`+`B.mT` |
| `restore_value=['d_ptr']` | `a_fused_k...:10` / `b_fused_k...:10` `@triton.autotune(..., restore_value=['d_ptr'])` | 因 kernel **原地累加** D，autotune 重跑前必须恢复 |
| K 尾块不裁 | `a_fused_k...:51` `for k in range(k_start, k_end, BLOCK_SIZE_K)` | 循环上界是 `k_end`（对齐后），无条件裁到逻辑 K |
| `mk_alignment` 约束 | `m_grouped_gemm.py:72`（M%align==0）、`tune_options.py:24-25`（BLOCK≤align） | 来自 `csrc/jit_kernels/heuristics/runtime.hpp:18/53-54` |
| smem 预算 | `tune_options.py:5-23` | A100 上限 `166912`，把 9 个 `Config` 过滤成可用集 |

（这些行号在第 29 篇已逐条核对；本篇只用它们把「测试体的一条断言」映射到「kernel 的一个行为」。）

---

## 6. H100(sm90) 实测

实验 `harness/exp_legacy_test_oracle.{py,sh,out.txt}`：

- **Part A**（真机 introspection）：源文件 95 行、两个 `test_` 函数、`__main__` 直调、
  **无 arch 门控**。
- **Part B**：m-grouped bf16 枚举 32 条（×4 = 128 次发射）、k-grouped bf16 枚举 7 条（×2 = 14 次发射）；
  `generate_m_grouped_contiguous` 的 `use_psum_layout` 默认确为 `False`。
- **Part C**：**直接 import 真实测试模块并调用两个真实测试体**（只把模块内 `bench_kineto`
  换成常量桩，形状一个没缩）——全流程无断言失败，耗时 **16.8 s**；控制台打印的 `Testing ...`
  两段即真实测试体输出。
- **Part D**：arch9 闸门跳过 0；伪造 arch10 枚举 126 条、闸门跳过 119，四因子
  （`psum`/`k_align`/`no_acc`/`out_dtype`）直方图齐全。
- **Part E**（独立 torch oracle 复核）：`nt` vs oracle `calc_diff=2.99e-09`（bf16 累加序差异，
  非逐位）；`nn`==`nt` **逐位**（`0.0`）；`expand` 恒等 == `m_grouped` **逐位**（`0.0`）；
  `expand` 任意置换 vs oracle 有效行 `calc_diff=3.06e-09`；`m_grouped` 填充行全零、
  `expand` 置换下填充行写出非零（语义边界）。
- **Part F**（判据利齿）：`calc_diff` 对 `d*0.999` 仅 `5.0e-07`（全局同缩放几乎不变）、
  单行归零 `1.5e-05`（漏）、1/8 行归零 `6.8e-02`（抓）、1-ulp 只有逐位抓得住；
  真实 `bench_kineto` 采到 m-grouped（G=8, m=33792, n=7168, k=7168）**223.1 TFLOPS / 115 GB/s**、
  k-grouped（G=4, m=4096, n=7168, k=33792）**52.5 TFLOPS / 33 GB/s**。

总判定 **19 PASS / 0 FAIL**。

---

## 7. 小结

- `tests/test_legacy.py` 是整条测试栈里**最后一个被单独拆的顶层规格**，也是唯一无 arch 门控、
  只测 bf16、专测 legacy Triton 家族的文件。
- 它把 legacy 的**分组语义**钉死为：BLOCK 首行定组、`-1` 行掩码（`m_grouped` 写零 /
  `a_fused_m` 由行索引决定）、空组按 `ACC` 取舍、k-grouped 的「不裁尾块 + gap 置零」。
- 它把 `nt`/`nn` 的**主序别名**与 `expand` 的**逐行 gather**表达成两维布尔矩阵，其中
  `nn` 与 `expand` 都只是 `nt`/`m_grouped` 的薄壳（实测逐位相等）。
- 它的 k-grouped 跳过闸门同时是 legacy 的**可用域声明**：arch9 上 no-op（7/7 全跑），
  伪造 arch10 上拦掉 119/126，只留「FP32 + 累积 + `k_alignment==128` + 非 PSUM」这一狭窄区间。
- 判据是松的 `calc_diff<0.001`；真正把语义钉死的是 generator 造的正确 `ref_d` 与
  （本实验补的）逐位等价复核。这类「松判据 + 正确参考」是 legacy 测试的风格，与稠密/einsum
  家族的「从松到严一长串断言」形成对照。

**边界**：legacy 只支持 bf16；FP8/FP4 不走 Triton（`tune_options.py:6`）。A100（sm80）是该家族
的目标硬件，但 Triton 与架构解耦，H100 上仍能真跑（本文全部数字出自 H100）。现代 C++ 的
`m_grouped`/`k_grouped` 见第 10-13 篇；legacy kernel 本体的逐段精读见第 29 篇。
