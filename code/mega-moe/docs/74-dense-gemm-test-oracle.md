# 稠密 GEMM 的验收测试即正确性规范（`tests/test_fp8_fp4.py` / `tests/test_bf16.py`）

> 本专题（Mega MoE 前向 / 反向算子实现解析）的第 74 篇。测试栈此前被拆过三层：第 40 篇拆了
> **计时/判对基座** `deep_gemm/testing/`（`calc_diff` / `bench_kineto`），第 41 篇拆了
> **操作数 / 布局生成器** `tests/generators.py`，第 43 篇拆了**比 relerr 更硬的逐位断言辅助**
> `tests/utils.py`。另外，第 41 / 47 / 50 / 57 篇分别拆过 Mega MoE / Mega Gate / MQA logits /
> Mega mHC 各自**端到端**的验收测试。
>
> 但整条测试栈最基础、最先跑、也是所有这些数字来源的**两个顶层文件**——
> `tests/test_fp8_fp4.py`（406 行）与 `tests/test_bf16.py`（341 行）——一直只被顺带引用过几行。
> 本篇把它们当作**「稠密 GEMM 到底算什么对」的可执行规格**来拆：不是再讲一遍 kernel 实现，
> 而是讲清楚**每个 kernel 出口要被哪些判据卡**，以及这些判据**严格在哪、松在哪**。
>
> 两个文件的**主体 kernel 在 H100(sm90) 上真的会跑**（FP8 走 1D2D、BF16 走 NoSF，见 §2.2），
> 只有 FP4×FP4 / FP8×FP4 / `use_alpha` / `UE8M0` / 随机舍入等分支被 `get_arch_major()==10` 门控掉。
> 本文所有行号均已对照 DeepGEMM commit `057ca59`（Public release 26/09/30）核对过。
> 实测见 `code/mega-moe/harness/exp_dense_gemm_test_oracle.{sh,py,out.txt}`。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 定位：判据链的最顶层

DeepGEMM 的测试栈是一条**从上到下、职责分明**的链：

| 层 | 文件 | 回答的问题 | 本专题 |
|----|------|-----------|--------|
| 顶层规格 | `tests/test_bf16.py` / `test_fp8_fp4.py` | **这个 kernel 出口算什么对、什么分布要测** | 本篇 |
| 断言辅助 | `tests/utils.py` | 「比 relerr 更硬」的逐位性质（direct==累积、随机舍入 oracle、PSUM padding） | 第 43 篇 |
| 操作数生成 | `tests/generators.py` | 操作数**长什么样**、枚举哪些形状/主序/量化配置 | 第 41 篇 |
| 计时/判对基座 | `deep_gemm/testing/` | `calc_diff` 怎么算、`bench_kineto` 怎么计时 | 第 40 篇 |

顶层规格的结构在 `__main__` 里一句话说清：

- `test_fp8_fp4.py:396-406`：`torch.manual_seed(0)` / `random.seed(0)` 后**无条件**跑四个函数
  `test_gemm()`、`test_m_grouped_gemm_contiguous()`、`test_m_grouped_gemm_masked()`、
  `test_k_grouped_gemm_contiguous()`。**没有 arch 门控**——因为 FP8 的 1D2D 路径 SM90 就有。
- `test_bf16.py:326-341`：前三个函数被 `if get_arch_major() >= 9:`（`:333-337`）包住，后三个
  cuBLASLt 函数无条件跑。BF16 主 kernel 需要 SM90 或更高（SM80 只有 cuBLASLt 回退）。

这就是两个文件「同名同构、门控不同」的根源：**BF16 的 GEMM 主体是 sm90+ 才有的，
FP8 的 1D2D 是 sm90 起就有的**——两者在 H100 上都进 `__main__` 主体。

---

## 2. 两个文件的全貌与 arch 门控矩阵

### 2.1 用例矩阵

| 函数 | `test_fp8_fp4.py` | `test_bf16.py` | 枚举器（`generators.py`） |
|------|:--:|:--:|------|
| `test_gemm`（稠密前向+反向） | `:26-111` | `:23-66` | `enumerate_normal`（`:122-173`） |
| `test_m_grouped_gemm_contiguous` | `:114-175` | `:69-108` | `enumerate_m_grouped_contiguous`（`:194-210`） |
| `test_m_grouped_gemm_masked` | `:178-238` | `:111-155` | `enumerate_m_grouped_masked`（`:213-226`） |
| `test_k_grouped_gemm_contiguous` | `:241-393` | `:158-214` | `enumerate_k_grouped_contiguous`（`:229-262`） |
| `test_cublaslt_gemm` | — | `:217-244` | `enumerate_normal` |
| `test_cublaslt_batched_syrk` | — | `:247-280` | `enumerate_batched_syrk_symm`（`:176-191`） |
| `test_cublaslt_batched_symm` | — | `:283-323` | 同上 |

FP8 比 BF16 多出来的「利齿」在 `test_gemm` / `test_m_grouped_*` / `test_k_grouped_*` 里：
量化配置多样性（`QuantConfig`）、FP4 与「无损转 FP8」交叉核对、`SF` 的 K-stride padding。

### 2.2 arch 门控（真实数出 H100 会跑多少）

三个枚举器里的 arch 判定决定了 SM90 与 SM100 的用例分岔：

- `get_kernel_types`（`generators.py:100-104`）：BF16 恒返回 `KernelNoSF`；FP8 **SM90 → `Kernel1D2D`、
  否则 `Kernel1D1D`**。这就是「SM90 上的 FP8 dense 只有 1D2D」的规格来源。
- `QuantConfig.get_list_from_dtype`（`:76-86`）：SM90 只有 legacy `(128,128,False,False)` 一个配置；
  SM100 追加 `(128,32,False,True)`（FP8×FP4）与 `(32,32,True,True)`（FP4×FP4）。
- `get_ue8m0_usage`（`:95-98`）：SM90 恒 `False`（无 UE8M0 打包）、SM100 仅 1D1D 用。
- `enumerate_normal`（`:122-173`）：`m_fwd_list=[1,128,4096]`、`m_bwd_list=[4096]`；BF16 累积
  只在 BF16 或 SM100 才加；SM90 FP8 的 backward 强制回退成 `KMajor + Kernel1D1D`（`:161-163`）。
- `enumerate_k_grouped_contiguous`（`:229-262`）：SM90 只 `sf_layout=(128,128)`、`psum_list=(False,)`、
  `cd_options=[(True, float)]`（**SM90 的 K-grouped 强制累积 + FP32 输出**）。
- `enumerate_m_grouped_*`（`:194-226`）：`ensure_zero_padding` 只在 SM100 + PSUM 才启用。

**H100(sm90) 真机枚举计数**（`harness/exp_dense_gemm_test_oracle.out.txt` Part E）：

| 枚举器 | arch=9（本机） | arch=10（假想） |
|--------|:--:|:--:|
| `enumerate_normal(FP8)` | 42 | 182 |
| `enumerate_normal(FP4)` | — | 42 |
| `enumerate_normal(BF16)` | 84 | — |
| `m_grouped_contiguous(FP8)` | 16 | 120 |
| `m_grouped_contiguous(BF16)` | 32 | — |
| `m_grouped_masked(FP8/BF16)` | 32 / 32 | — |
| `k_grouped_contiguous(FP8)` | 7 | 252 |
| `k_grouped_contiguous(FP4)` | — | 126 |

结论：**H100 上两个文件共 252 个枚举用例会跑**（外加 `test_bf16` 的 cuBLASLt 三函数）；
`use_alpha=True`、`test_sf_k_padding`、随机舍入、FP4、UE8M0 全是 `get_arch_major()==10` 才有。

---

## 3. `tests/test_fp8_fp4.py` 逐函数

### 3.1 `test_gemm`（`:26-111`）：稠密 FP8/FP4 的前向 + 反向

每个 shape 会跑**四件事**，按「松 → 严」排列：

1. **`calc_diff < max_diff`**（`:54-57`）：`d` 对参考 `ref_d`（生成器给的 fp32 参考量化后再算）。
   `test_alias=(False,True)` 两遍——`True` 走 `fp8_fp4_gemm_{nt/nn/tn/tt}` 的**转置薄壳**（`:47-51`，
   非 K-major 的操作数取 `.T` 并要求 `is_contiguous`），验证「转置只改 stride 不改数学」。
   `use_alpha` 随机 `uniform(-1,1)` 只在 SM100（`:28`）。
2. **随机舍入 oracle**（`:60-70`，SM100-only）：`epilogue.BF16StochasticRounding()` 的输出必须
   **逐位等于** `assert_stochastic_bf16_matches_fp32_accumulation` 纯 torch 复刻的四元组哈希（见第 43 篇）。
3. **逐位确定性 20 次**（`:71-82`）：把 `d` 复位成初始值后**重跑 20 次**，要求
   `torch.equal(d, equivalent_d)`。这是「同一输入同一 kernel 配置必须给出完全相同的字节」——
   随机舍入/atomic 归约/split-K 都会在这里现形。
4. **FP4↔FP8 交叉核对**（`:95-100`）：当配置含 FP4 时，把 FP4 操作数用 `convert_to_fp8`（E4M3 可
   **无损**表示所有 E2M1）转成 FP8 再跑一遍，要求两次 BF16 输出 `calc_diff < 1e-14`——
   **UMMA_K 不同（FP4=32、FP8=128）、实现路径不同，但输出应当逐位一致**。
5. **SF 的 K-stride padding**（`:83-94`，SM100-only）：把 SF 的 K 维 pitch 撑大 4（`add_sf_k_padding`，
   用 UE8M0 NaN `-1` 毒化 padding）后重跑，要求与紧凑 pitch **逐位相等**——抓「用了紧凑 pitch 越界读」。

最后是 `bench_kineto`（`:101`）与 cuBLASLt 加速比（`:102-111`）。

### 3.2 `test_m_grouped_gemm_contiguous`（`:114-175`）

M 轴分组的规格差异：**只在 `valid_mask` 上比较**（`:147`、`:154`），因为不合法组的内容无意义。
- `test_alias=True` 时 **A 必须 K-major**（`:135`），B 可转置（`:136`）；这是 m-grouped 的硬约束。
- 20 次逐位（`:144-147`）、FP4↔FP8（`:148-153`）、`calc_diff`（`:154-156`）同前。
- **PSUM 零填充**（`:157-158`）：`use_psum_layout and ensure_zero_padding` 时调
  `assert_psum_zero_padding`，要求每组的「逻辑尾到对齐边界」在 A/D 里**全零**（`:9-17`）。

### 3.3 `test_m_grouped_gemm_masked`（`:178-238`）

与 contiguous 同构，但用 `enumerate_m_grouped_masked`：`num_tests=8` 次随机（`:188`，
`expected_m=int(expected_m_per_group*1.2)`，`:191`），每轮重生成随机 token 分布后判
`valid_mask` 上的 `calc_diff`（`:221-222`）与 20 次逐位（`:211-214`）。

### 3.4 `test_k_grouped_gemm_contiguous`（`:241-393`）：wgrad 的密集验收

这是「MoE wgrad」(`k_grouped`) 的主验收，也是本篇最严的函数：

- **按 arch 选 kernel**（`:244-251`）：`arch==9` 用 `k_grouped_fp8_gemm_nt_contiguous`，
  `arch==10` 用 `k_grouped_fp8_gemm_tn_contiguous`；SM100 另加 FP4 版本。**与第 10/12 篇一致**。
- **每个用例跑三种 K 分布变体**（`:261` → `enumerate_k_grouped_contiguous_test_variants`，`:265-273`）：
  原样、随机 25% 空组（首尾保留非空）、只有中间一组非空。**空组必须 D 全零/NaN 不变**。
- **`recipe=(1,1,gran_k)`**（`:259`）：K 轴分组 per-channel scale 的三元组 recipes。
- **未累积时把 D 填 NaN**（`:269-271`）：direct-output 的 D 初始是 NaN——若 kernel 漏写某个块，
  NaN 会**主动暴露**，而不累积模式下 D 本应被完全覆盖。
- **`host_ks_cpu` 的三种传法**（`:288`）：PSUM 下额外传 `None` 与 `[]`，要求三者结果逐位一致。
- **20 次逐位**（`:289-298`）用了 `assert_bitwise_equal`（含坐标诊断）。
- **direct == FP32 累积**（`:299-306`）：`assert_direct_output_matches_fp32_accumulation`——
  「直接从零输出」必须**逐位等于**「同一 workload 用 C=D 累积」；BF16 输出再 cast 一次。
- **随机舍入**（`:307-311`）：BF16 输出 + `BF16StochasticRounding()` 对 quartet 哈希 oracle。
- **gran_k==32 的打包 UE8M0 逐位**（`:319-340`）：用纯 torch 参考 `pack_ue8m0_to_int` + 按组 padding，
  与 `get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor` 的输出 `torch.equal`——**这正是第 16 篇**。
- **打包 SF 的 K-stride padding**（`:345-352`）：同 §3.1 第 5 点，抓打包布局的紧凑 pitch 越界读。

---

## 4. `tests/test_bf16.py` 逐函数

BF16 的验收与 FP8 共享**骨架**，差异在「没有量化配置、没有 FP4、没有 UE8M0」：

### 4.1 `test_gemm`（`:23-66`）

- 每个 shape 先 `deep_gemm.use_deterministic_algorithms(True)`（`:28`）——**关键**：BF16 稠密 GEMM
  默认会回退到 cuBLASLt（`nvjet_*`，见第 27 篇），只有确定性格局才走 DeepGEMM 的 `bf16_gemm`；
  门控住才能用 `bench_kineto(..., 'bf16_gemm')` 抓到自研 kernel（`:56`）。
- `calc_diff < 1e-5`（`:46`）、20 次逐位在 `test_gemm` 里**没有**（BF16 主循环只对 `alias` 做正确性）；
  逐位确定性只在 `test_fp8_fp4` 与 grouped 里。
- 随机舍入（`:49-54`，SM100-only）；最后是 `bf16_gemm_nt` vs cuBLASLt 的加速比（`:56-66`）。

### 4.2 grouped 三个（`:69-155`）

`test_m_grouped_gemm_contiguous`（`:69-108`）、`test_m_grouped_gemm_masked`（`:111-155`）、
`test_k_grouped_gemm_contiguous`（`:158-214`）与 FP8 版逐条对应，规则更少：
`calc_diff < 1e-5`、PSUM 零填充、`k_grouped` 的 direct==累积 + 随机舍入。
`k_grouped_bf16_gemm_tn_contiguous`（`:174`）在两种 arch 是**同一个**入口（`tn` 对 BF16 都支持）。

### 4.3 cuBLASLt 三个（`:217-323`）：回退路径的独立验收

- `test_cublaslt_gemm`（`:217-244`）：阈值分两档（`:227`）——BF16 累积 `1e-5`，其余 `6e-7`。
  重点在「DeepGEMM 的 `bf16_gemm_nt` 与 cuBLASLt 在同口径下都对参考」——即第 27 篇的回退可接受性。
- `test_cublaslt_batched_syrk` / `_symm`（`:247-323`）：Muon 的 `D=A·Aᵀ` / `D=A·B`，
  阈值恒 `6e-7`（`:252`、`:288`），覆盖 2D↔3D、K-major/MN-major、带 padding 的 strided batch
  （`:254`）。这就是第 45 篇 `batched_syrk/symm` 的验收来源。

---

## 5. 「验收利齿」清单：除了 `calc_diff` 还要卡什么

把两个文件里所有「比 relerr 更硬」的判据汇总如下（阈值实测自 `harness/...out.txt` Part B）：

| 判据 | 出处 | 卡住什么 | 严格度 |
|------|------|----------|--------|
| `calc_diff < QuantConfig.max_diff()` | `generators.py:69-74` | 整体数值接近 | 松（FP8 0.001 / FP8×FP4 0.01 / FP4×FP4 0.02） |
| `calc_diff < 1e-5` | `test_bf16.py:46/91/138/178` | BF16 稠密/分组 | 中 |
| `calc_diff < 6e-7` | `test_bf16.py:227/252/288` | cuBLASLt 参考 / SYRK / SYMM | 中 |
| 20× `torch.equal` / `assert_bitwise_equal` | `test_fp8_fp4.py:79/145/212/289` | 同输入可复现性（无 atomic/split 漂移） | **逐位** |
| `assert_direct_output_matches_fp32_accumulation` | `test_fp8_fp4.py:306`、`test_bf16.py:188` | direct 输出 == 逐位从零累积 | **逐位** |
| `assert_stochastic_bf16_matches_fp32_accumulation` | `test_fp8_fp4.py:65/311`、`test_bf16.py:51/193` | 随机舍入 == quartet 哈希 oracle | **逐位** |
| FP4 == 无损转 FP8（`< 1e-14`） | `test_fp8_fp4.py:99/152/219/285` | 两条量化路径数值等价 | 近逐位 |
| SF K-stride padding 逐位 | `test_fp8_fp4.py:93/351` | 不吃紧凑 pitch / 不越界读 | **逐位** |
| `assert_psum_zero_padding` | `test_fp8_fp4.py:158`、`test_bf16.py:93` | PSUM 逻辑尾到对齐边界全零 | **逐位** |
| 空组 D / NaN 不变 | `test_fp8_fp4.py:269-271` | 不累积模式漏写即暴露 | **逐位** |

一句话：`calc_diff` 只保证「数值上像」，**逐位判据才保证「实现正确」**。

---

## 6. 为什么必须逐位：`calc_diff` 的两处盲区

`calc_diff`（`deep_gemm/testing/numeric.py:5-13`）是**归一化余弦相似度**：

```
calc_diff(x, y) = 1 - 2·<x,y> / (‖x‖² + ‖y‖²)
```

它有两个对你「抓 bug」不利的性质，`harness/...out.txt` Part D 都做了实证：

### 6.1 对全局缩放不变

若 `y = c·x`，则 `calc_diff = 1 - 2c/(1+c²)`，**与 x 本身无关**。实测：

| 缩放 c | `calc_diff` | 相对 x 是否有差异 |
|:--:|:--:|:--:|
| 1.0 | 0 | 无 |
| 1.0001 | 5.0e-9 | 有（但极小） |
| 0.999 | 5.0e-7 | 有 |
| 0.9 | 5.5e-3 | 有 |

所以「整张输出被乘了 1.0001」这种 bug，`calc_diff` 完全无感（5e-9 ≪ 1e-5），
但 `torch.equal` 立刻判否。真实 BF16 GEMM 输出上实测：基准 `calc_diff=1.782e-08`，
把参考整体缩放 0.999/1.0001 后 **`calc_diff` 一字不变**（仍 1.782e-08，因为相似度不变），
而 `torch.equal=False`。

### 6.2 对单元素扰动的极度不敏感

对真实输出某一元素翻 1 个 bf16 ulp（`-18.0 → -18.125`，相对 6.9e-3），
`calc_diff` 只从 1.782e-08 变到 **1.487e-11**（甚至更小），远低于阈值；
而 `assert_bitwise_equal` 精确报出 `coord=(0,0)`、`x_byte=144, y_byte=145`。

**结论**：`calc_diff` 负责「大方向没错」，逐位判据负责「字节级正确」。两个顶层测试文件正是
两条判据**并用**的样板——先用 `calc_diff` 卡住精度，再用逐位 oracle 卡住实现。

---

## 7. H100 实测

`harness/exp_dense_gemm_test_oracle.py` **直接 import 真实测试模块并调用真实测试体**
（`test_fp8_fp4.test_gemm` 等），只把「形状枚举」压到前若干条、「`bench_kineto`」打桩旁路计时
——**测试体本身一行未改**。结果（`...out.txt`，`18 PASS / 0 FAIL`）：

| 分组 | 内容 | 结果 |
|------|------|------|
| A | 两模块可 import、`__main__` 入口序列可见 | PASS |
| B | 阈值表（0.001/0.01/0.02/1e-5/6e-7/1e-14）逐条读出 | PASS |
| C | 真实测试体 **11 个函数**（含 cuBLASLt 三函数、batched syrk/symm 全扫） | **11/11 PASS** |
| D | `calc_diff` 两处盲区（全局缩放不变 + 1-ulp 不敏感），逐位抓到 | PASS |
| E | 枚举计数与 arch 门控矩阵（arch9=252 用例） | PASS |
| F | 真实 `bench_kineto` 吞吐 | PASS |

Part C 里真跑并通过的测试体包括：`test_fp8_fp4.{test_gemm, test_m_grouped_gemm_contiguous,
test_m_grouped_gemm_masked, test_k_grouped_gemm_contiguous}`、
`test_bf16.{test_gemm, test_m_grouped_gemm_contiguous, test_m_grouped_gemm_masked,
test_k_grouped_gemm_contiguous, test_cublaslt_gemm, test_cublaslt_batched_syrk,
test_cublaslt_batched_symm}`。

Part F 用**真实** `bench_kineto` 采的 BF16 `bf16_gemm_nt`（`use_deterministic_algorithms(True)`
强制走自研 kernel，否则默认回退 cuBLASLt 抓不到）：

```
BF16 bf16_gemm_nt (m=4096,n=7168,k=7168):  1790.0 us | 235 TFLOPS
BF16 bf16_gemm_nt (m=4096,n=4096,k=4096):   333.0 us | 413 TFLOPS
```

（注意：Part C 打印里 `1000000 us` 是计时打桩后的占位值，不看；Part C 的验收价值在
**断言全过**，真实吞吐看 Part F。）

---

## 8. 边界与 SM100 专属

以下分支在 H100 上**不会进入**（由 arch 门控跳过），标注为**SM100 专属，仅分析**：

- `use_alpha` 参数（`test_fp8_fp4.py:28`、`test_bf16.py:26`）——只有 `get_arch_major()==10` 才测。
- FP4 相关：`(128,32,False,True)` / `(32,32,True,True)` 配置、FP4 kernel 入口、FP4↔FP8 交叉核对
  （`test_fp8_fp4.py:95-100/148-153/215-220/275-285`）。
- `test_sf_k_padding` / K-grouped 打包 SF 的 padding（`:29/83-94/345-352`）。
- `BF16StochasticRounding` 随机舍入（`test_bf16.py:49`、`test_fp8_fp4.py:60`）。
- `UE8M0` 打包（`get_ue8m0_usage` 在 sm90 恒 False）。
- `ensure_zero_padding` 的 PSUM 零填充（only SM100，`generators.py:203`）。

反过来，H100 上**确实会跑**的：FP8 1D2D 稠密/m-grouped/k-grouped（nt，累积+FP32）、
BF16 NoSF 稠密/m-grouped/k-grouped（tn）、以及 cuBLASLt 的三条回退验收。

---

## 9. 小结

- `tests/test_fp8_fp4.py` 与 `tests/test_bf16.py` 是 DeepGEMM 整条测试栈的**顶层规格**：
  回答「每个稠密/分组 GEMM 出口算什么对、哪些分布/主序/量化要覆盖」。
- 判据**并用** `calc_diff`（数值接近）与**逐位 oracle**（实现正确）：20 次确定性、
  direct==FP32 累积、随机舍入 quartet 哈希、FP4==无损 FP8、SF K-stride padding、PSUM 零填充。
- `calc_diff` 是余弦相似度，**对全局缩放不变、对单元素扰动不敏感**——这是逐位判据不可省的根因。
- H100(sm90) 上 **252 个枚举用例**会真实进入这两个文件的 `__main__`；`18 PASS / 0 FAIL`，
  BF16 峰值实测 235（大 K）/ 413（方阵）TFLOPS。FP4 / `use_alpha` / UE8M0 / 随机舍入为 SM100 专属。
