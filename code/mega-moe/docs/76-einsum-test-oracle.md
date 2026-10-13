# einsum 家族的验收测试即正确性规范（`tests/test_einsum.py`）

> 本专题（Mega MoE 前向 / 反向算子实现解析）的第 76 篇。测试栈此前被拆过三层：第 40 篇拆了
> **计时/判对基座** `deep_gemm/testing/`，第 41 篇拆了**操作数/布局生成器** `tests/generators.py`，
> 第 43 篇拆了**逐位断言辅助** `tests/utils.py`。另外第 41 / 47 / 50 / 57 / 64 篇分别把
> Mega MoE / Mega Gate / MQA logits / Mega mHC / 稠密 GEMM 各自**端到端**的验收测试当
> 「可执行规格」拆透。
>
> 但把整个 **einsum 家族**——`bmk,bnk->mn`（跨 batch 的批归约，第 17 篇）、MLA 双向投影
> `bhr,hdr->bhd` / `bhd,hdr->bhr`（第 18 篇）、权重梯度 `bhd,bhr->hdr`（第 45 篇）以及它们的
> **FP8 block-scale 变体**（第 21 篇）——的出口判据集中定义的 `tests/test_einsum.py`（276 行），
> 一直只被各篇顺带引用过几行。本篇把它当作**「einsum 家族到底算什么对」的可执行规格**来拆。
>
> `test_einsum.py` 里 H100(sm90) 上**真的会跑**的用例：`bmk,bnk->mn`、两个 BF16 MLA 投影、
> BF16 权重梯度、以及 FP8 `bhr,hdr->bhd` 的 **BF16 输出路径**；只有两个 FP8 反向/投影函数被
> `@test_filter(get_arch_major() >= 10)` 门控。本文所有行号均已对照 DeepGEMM commit `057ca59`
> （Public release 26/09/30）核对过。实测见 `code/mega-moe/harness/exp_einsum_test_oracle.{sh,py,out.txt}`。
> 目标卡代号敏感：全文一律用「目标卡」指代。

---

## 1. 定位：einsum 家族的「顶层规格」

DeepGEMM 的 einsum 家族在 `csrc/apis/einsum.hpp` 里按**下标字符串**四分派（见第 17/18/21 篇）：

| einsum 表达式 | 语义 | host 入口 | 本专题 |
|---|---|---|---|
| `bmk,bnk->mn` | `Σ_s A_s B_sᵀ`（batch 并入收缩轴） | `einsum.hpp:22-58` | 第 17 篇 |
| `bhr,hdr->bhd` | MLA 吸收投影（头=group 的 batched GEMM） | `einsum.hpp:60-92` | 第 18 篇 |
| `bhd,hdr->bhr` | 其转置对 | `einsum.hpp:94-126` | 第 18 篇 |
| `bhd,bhr->hdr` | 跨 batch 权重梯度（K=batch） | `einsum.hpp:128-148`（无条件 cuBLASLt） | 第 45 篇 |

`tests/test_einsum.py` 就是这四个语义 + 两个 FP8 变体的**验收测试**——它回答「每个 einsum 出口
要被哪些判据卡、枚举哪些形状、哪些分布必须逐位复现」。它的 `__main__`（`:262-276`）在固定随机种子后
**无条件**顺序调用七个函数：

```
torch.manual_seed(0) / random.seed(0)
test_bmk_bnk_mn()          # :269
test_bhr_hdr_bhd()         # :270
test_bhd_hdr_bhr()         # :271
test_bhd_bhr_hdr()         # :272
test_fp8_bhr_hdr_bhd()     # :274
test_fp8_bhd_hdr_bhr()     # :275  (被 test_filter 跳过)
test_fp8_bhd_bhr_hdr()     # :276  (被 test_filter 跳过)
```

---

## 2. 全貌与 arch 门控矩阵

### 2.1 函数表

| 函数 | 行号 | 表达式 | H100(sm90) |
|---|---|---|---|
| `test_bmk_bnk_mn` | `:21-41` | `bmk,bnk->mn` | 跑 |
| `test_bhr_hdr_bhd` | `:44-72` | `bhr,hdr->bhd`（BF16，双向 cuBLASLt 切换） | 跑 |
| `test_bhd_hdr_bhr` | `:75-104` | `bhd,hdr->bhr`（BF16） | 跑 |
| `test_bhd_bhr_hdr` | `:107-125` | `bhd,bhr->hdr`（BF16，恒定 cuBLASLt） | 跑 |
| `test_fp8_bhr_hdr_bhd` | `:128-191` | FP8 `bhr,hdr->bhd`（bf16 输出 + SM100 FP8 输出对） | **跑**（bf16 路径） |
| `test_fp8_bhd_hdr_bhr` | `:194-233` | FP8 `bhd,hdr->bhr` | 门控跳过 |
| `test_fp8_bhd_bhr_hdr` | `:236-259` | FP8 `bhd,bhr->hdr` | 门控跳过 |

### 2.2 arch 门控（`test_filter`）

两个函数带装饰器：

- `:194` `@test_filter(lambda: get_arch_major() >= 10)` → `test_fp8_bhd_hdr_bhr`
- `:236` `@test_filter(lambda: get_arch_major() >= 10)` → `test_fp8_bhd_bhr_hdr`

`test_filter`（`deep_gemm/testing/utils.py:11-22`，第 30/40 篇）在条件不成立时**只打印一行
`Filtered by ...` 然后返回**，不抛异常。所以在 H100（`get_arch_major()==9`）上，这两个函数
「被调用但不做任何事」——这正是「SM100 专属，仅分析」的规格来源。

注意：`test_fp8_bhr_hdr_bhd` 是**唯一没有装饰器**的 FP8 函数——因为它的 BF16 输出路径走 SM90 的
`1d2d` 骨架（第 21 篇），只有 `output_dtype == torch.float8_e4m3fn` 那一支才需要 SM100。

---

## 3. 逐函数精读

### 3.1 `test_bmk_bnk_mn`（`:21-41`）：批归约

```python
for s in (129, 4096, 8192):                              # :23
    for m, n, k in [(128,384,128),(256,256,256),(384,128,384)]:   # :24
        for dtype in (torch.float, torch.bfloat16):       # :25
            a = randn((s, m, k), bf16); b = randn((s, n, k), bf16)  # :26-27
            d = randn((m, n), dtype)                       # :28
            c = d if dtype == torch.float else None        # :29
            ref_d = (c if dtype == torch.float else 0) + torch.bmm(a.float(), b.float().mT).sum(0)  # :32
            deep_gemm.einsum('bmk,bnk->mn', a, b, d, c=c)  # :33
            assert calc_diff(d, ref_d) < 1e-5              # :34
```

三个要点：

1. **`s=129` 是刻意的非对齐值**（不是 128 的倍数）——抓 split-K/尾块处理。参考实现
   `torch.bmm(a, bᵀ).sum(0)` 把「batch 求和」显式写出来，对应第 17 篇坐实的「等价于把 batch 并进
   收缩轴」；`bmk,bnk->mn` 的批归约可以看作 `(m,s·k) @ (n,s·k)ᵀ`，只是 DeepGEMM 用 `atomicAdd`
   跨 batch 块归约（非确定性，第 17 篇 P8-1）。
2. **fp32 输出必须 `c=d`**（`:29`）：这是 `einsum.hpp:25-26` 对 `bmk_bnk_mn` 的契约——fp32 `D`
   只能累积（`c==d`），不能凭空写。bf16 输出才允许 `c=None`。
3. `bench_kineto(..., 'bmn_bnk_mn_gemm_impl')`（`:36`）——测试用这个（拼写为 `bmn_` 的）kernel 名
   子串过滤计时目标。

> **真机踩到的坑**：fp32 `bmk,bnk->mn` **不传 `c`** 会段错误（`C/D` 契约被违反时不是抛
> `RuntimeError` 而是非法访存）。本实验 Part D3/Part F 一开始就踩到，补上 `c=d` 后正常——
> 见 `harness/exp_einsum_test_oracle.py`。

### 3.2 `test_bhr_hdr_bhd`（`:44-72`）：MLA 吸收投影 + 双向选路

```python
for h, r, d in [(128, 512, 128), (8, 4096, 1024)]:       # :46
    for b in (4, 32, 128, 4096, 8192):                   # :47
        x  = randn((b, h, r), bf16)                      # :48
        fy = randn((h, d, r + 128), bf16)                # :49  权重比 r 宽 128
        y  = fy[:, :, :r]                                # :50  切片成带 padding 的视图
        ref_z = torch.einsum('bhr,hdr->bhd', x, y)       # :51
        for use_cublaslt in (False, True):               # :54
            deep_gemm.use_deterministic_algorithms(not use_cublaslt)  # :55
            z = empty((b,h,d), bf16); einsum(...)        # :56-57
            assert calc_diff(z, ref_z) < 1e-10           # :58
            launch = lambda out, acc: einsum(..., out, c=acc)  # :59
            assert_direct_output_matches_fp32_accumulation(z, launch, ...)  # :60
            if not use_cublaslt and get_arch_major() == 10:  # :61
                einsum(..., epilogue=BF16StochasticRounding())  # :62
                assert_stochastic_bf16_matches_fp32_accumulation(z, launch, ...)  # :63
```

这是整篇最富信息的一段：

- **带 padding 的权重**（`:49-50`）：`fy` 的最后一维是 `r+128`，`y=fy[:,:,:r]` 是一个
  **非连续视图**。测试要求「带 padding 权重」与连续权重结果一致——第 18 篇已实测
  `fy[:,:,:r]` 与连续权重**逐位相等**，这里正是那条实测的规格出处。
- **双向选路**（`:54-65`）：`use_deterministic_algorithms(not use_cublaslt)`——`use_cublaslt=False`
  时打开确定性 → 走 DeepGEMM 自研 `sm90_bf16_gemm`；`True` 时关确定性 → 回退 cuBLASLt `nvjet`
  （三与门，第 27 篇）。`kernel_name = 'nvjet' if use_cublaslt else 'gemm'`（`:64`）正是抓这两条路。
- **`calc_diff < 1e-10`（极紧）**：接近 fp32 相对误差量级；但对 BF16 输出它实际是「逐位相等」
  （本实验 D1 实测 `calc_diff=0.000e+00` 且 `torch.equal=True`）。
- **两个逐位 oracle**：`assert_direct_output_matches_fp32_accumulation`（`:60`，第 43 篇）要求
  「直接输出」**逐位等于**「同一 workload 用零张量当 C/D 累积」；`assert_stochastic_bf16_...`
  （`:63`，仅 `arch==10`）要求随机舍入逐位等于 quartet 哈希 PRNG。

### 3.3 `test_bhd_hdr_bhr`（`:75-104`）

与 3.2 逐条同构（`:89` 阈值 `1e-10`、`:91` direct oracle、`:92-94` SR），只是表达式换成
`bhd,hdr->bhr`。语义上它是 `bhr,hdr->bhd` 的转置对（第 18 篇的 `Batched` 调度用同一个
`GemmType::Batched`，只改 `(m,n,k)` 的映射）。

### 3.4 `test_bhd_bhr_hdr`（`:107-125`）：权重梯度 + C 累积

```python
z_0 = randn((h, d, r), float32) * 10                     # :113  大初始值
ref_z = z_0 + torch.einsum('bhd,bhr->hdr', x.float(), y.float())  # :114
z = z_0.clone()                                          # :116
deep_gemm.einsum('bhd,bhr->hdr', x, y, z, z)             # :117  D=C=z_0，累积
assert calc_diff(z, ref_z) < 1e-10                       # :118
```

- 这是第 45 篇的 `bhd,bhr->hdr`——**无条件 cuBLASLt**（`einsum.hpp:128-148`），`K=batch` 极小。
- **`z_0 * 10` 的用意**：初始值远大于 `A@B` 的贡献，把 C 累积语义放大。若 kernel 写成「覆盖」
  而丢掉 C，误差立刻显形（本实验 D2 实测：正确累积 `calc_diff=1.2e-13`，丢掉 C 变成 `4.7e-2`）。
- 每次都原地 `D=C=z`（同一张量），覆盖 `einsum.hpp` 的「C==D 原地累积」契约（第 46 篇
  `early_return`）。

### 3.5 `test_fp8_bhr_hdr_bhd`（`:128-191`）：FP8 投影的两种输出

`use_deterministic_algorithms(False)`（`:130`）后，只测一组形状 `h=8, r=4096, d=1024`、
`b∈{4,32,128,4096,8192}`。操作数构造：

- `x_fp8 = per_token_cast_to_fp8(x.view(-1, r), use_ue8m0=True)`（`:137`）→ A 是 **per-token 1D
  SF**（1×128 gran），再 reshape 回 `(b,h,r)` 与 `(b,h,r/128)`（`:138`）。
- `y_fp8` 逐头 `per_block_cast_to_fp8(y[i])`（`:141-142`）→ B 是 **per-128×128 块 2D SF**。
  这正是第 21 篇的「A 1D / B 2D」block-scale 组合。

然后**按 arch 分岔输出 dtype**：

```python
output_dtypes = (torch.bfloat16, torch.float8_e4m3fn) if get_arch_major() == 10 else (torch.bfloat16,)  # :144
```

**BF16 路径**（`:165-168`）：直接输出 bf16，`calc_diff < 1e-3`（阈值见 `:181`）。

**FP8 输出对路径**（`:146-164`，仅 SM100）：输出 `(output, sfd)`，其中
`sfd = empty_strided((b, ceil_div(h*d, 32*4)), (1, align(b,4)), int32)`（`:148-149`）——注意
**列主序 stride `(1, align(b,4))`**（TMA 对齐）。关键判据（`:159-164`）：

```python
d_bf16 = empty((b,h,d), bf16); einsum(...)               # 先出 BF16 D
ref_fp8, ref_sfd = per_token_cast_to_fp8(d_bf16.view(b, h*d), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)
assert torch.equal(output, ref_fp8)                      # FP8 数据逐位
assert torch.equal(sfd.contiguous(), ref_sfd)            # 打包 SF 逐位
```

即 **「融合的 FP8 输出」必须逐位等于「先 BF16 D、再做一次 per-token FP8 量化」这条生产管线**——
因为 epilogue 把累加器先舍入成 BF16，之后走同一套 amax/scale/rounding。这是 einsum 家族里
最严的一条判据，也是它**只在 SM100** 的原因（FP8 输出对 = `FP8Quantization` epilogue，第 21 篇）。

### 3.6 `test_fp8_bhd_hdr_bhr`（`:194-233`）/ `test_fp8_bhd_bhr_hdr`（`:236-259`）：SM100 专属

两个都被 `@test_filter(arch>=10)` 门控，**H100 整段跳过**：

- `test_fp8_bhd_hdr_bhr`：per-token A + per-block B，输出 bf16，`calc_diff < 1e-3`（`:213`），
  再跑 `BF16StochasticRounding` 对 quartet 哈希 oracle（`:220-224`）。
- `test_fp8_bhd_bhr_hdr`：**per-channel** cast 两侧（`:246-247`）、`recipe=(1,1,128)`（`:251`）、
  输出 `c=z` 累积，`calc_diff < 1e-3`（`:252`）。只有 `b∈{4096,8192}`。

---

## 4. 验收判据清单（阈值 + 逐位 oracle）

`calc_diff` 阈值（本实验 Part B 从源码字面量读出）：

| 行号 | 阈值 | 对象 |
|---|---|---|
| `:34` | `1e-5` | `bmk,bnk->mn`（FP32/BF16） |
| `:58` | `1e-10` | `bhr,hdr->bhd`（BF16） |
| `:89` | `1e-10` | `bhd,hdr->bhr`（BF16） |
| `:118` | `1e-10` | `bhd,bhr->hdr`（C 累积） |
| `:181` | `2e-3`（FP8 输出）/ `1e-3`（BF16 输出） | FP8 `bhr,hdr->bhd` |
| `:213` | `1e-3` | FP8 `bhd,hdr->bhr`（SM100） |
| `:252` | `1e-3` | FP8 `bhd,bhr->hdr`（SM100） |

比 relerr 更硬的**逐位 oracle**（全部来自 `tests/utils.py`，第 43 篇）：

| oracle | 调用点 | 卡住什么 |
|---|---|---|
| `assert_direct_output_matches_fp32_accumulation` | `:60`、`:91` | direct 输出 == 逐位「从零累积」FP32 基线 |
| `assert_stochastic_bf16_matches_fp32_accumulation` | `:63`、`:94`、`:179`、`:223`（SM100） | 随机舍入 == quartet 哈希 oracle |
| `torch.equal(output, ref_fp8)` / `(sfd, ref_sfd)` | `:163`、`:164`（SM100） | 融合 FP8 输出 == 「BF16 D → per-token 量化」逐位 |

**注意**：BF16 路径用 `1e-10` 这么紧的 `calc_diff`，实测直接就是逐位相等（D1）——einsum 的
BF16 家族在确定性模式下是**完全可复现**的（无 split-K 漂移）；只有 `bmk,bnk->mn` 的 fp32
`atomicAdd` 归约是非确定的（第 17 篇），所以它的阈值放宽到 `1e-5`。

---

## 5. einsum 特有的「利齿与盲区」（负样本控制，H100 实测）

### 5.1 `bhr,hdr->bhd`：紧 `calc_diff` 实际上就是逐位（D1）

真实形状 `b=128, h=128, r=512, d=128`，确定性模式：

```
calc_diff(z, ref) = 0.000e+00     (< 1e-10 -> PASS)
torch.equal(z, ref) = True
```

`calc_diff=0` 说明 BF16 MLA 投影在 sm90 上与 `torch.einsum` **逐位一致**（第 18 篇已记），
`1e-10` 阈值足够严。

### 5.2 `bhd,bhr->hdr`：C 累积语义被大初始值放大（D2）

`b=1024`，`z_0` 为 `randn*10`：

| 情形 | `calc_diff` |
|---|---|
| `D=C=z_0` 正确累积 | `1.203e-13`（< `1e-10`） |
| `D=0` 覆盖（丢 C） | `4.652e-02` |

大初始值让「丢 C」这种 bug 立刻远超阈值——这是测试选 `z_0*10` 而非 `z_0` 的原因。

### 5.3 `bmk,bnk->mn`：批归约语义两条等价（D3）

`s=64, m=128, n=256, k=128`：

| 对照 | `calc_diff` |
|---|---|
| 折叠 batch 的 fp32 GEMM `(m,s·k)@(n,s·k)ᵀ` | `3.9e-14` |
| 逐 batch 部分和 `Σ_s A_s B_sᵀ` | `2.1e-14` |

两条都到 fp64 舍入级（对照张量是 fp32），坐实「batch 并入收缩轴」与「部分和再求和」是同一数学
（第 17 篇 P8-1 的语义结论）。

---

## 6. 确定性开关切换真实 kernel（profiler 实测）

测试用 `deterministic` 开关在「自研 kernel」与「cuBLASLt」之间切换，并靠 kernel 名子串
（`:64`）分别计时。H100 上 profiler 抓到的真实名字：

```
deterministic=True : deep_gemm::sm90_bf16_gemm_impl<(cute::UMMA::Major)0, (cute::UMMA::Major)0, 0u,
                     128u,512u,128u,256u,128u,64u,...,(deep_gemm::GemmType)4,...bfloat16_t>
deterministic=False: nvjet_sm90_tst_128x256_64x4_1x2_h_bz_coopA_TNN
```

顺着模板实参可读出第 18 篇的映射：`(m,n,k)=(128,512,128)`（`h=128` MP 的后两维）、`kNumGroups`
对应 `h`，`GemmType)4 = Batched`，主序 `Major::KK`。`deterministic=False` 走 cuBLASLt 的
`nvjet`（第 27 篇三与门）。

---

## 7. H100 实测证据台账

实验：`harness/exp_einsum_test_oracle.{sh,py}` + `harness/exp_einsum_test_oracle.out.txt`。
**直接 `import test_einsum` 并调用真实测试体**（只把 `bench_kineto` 打桩旁路计时；测试体一行未改），
共 **15 PASS / 0 FAIL**：

| 分组 | 内容 | 结果 |
|---|---|---|
| A | 模块可 import、`__main__` 入口序列、两个 `@test_filter` 门控 | PASS |
| B | 7 处 `calc_diff` 阈值 + 4 类逐位 oracle 调用点 | PASS |
| C | **调用真实测试体** `test_bmk_bnk_mn` / `test_bhr_hdr_bhd` / `test_bhd_hdr_bhr` / `test_bhd_bhr_hdr` / `test_fp8_bhr_hdr_bhd` 全部断言通过；`test_fp8_bhd_hdr_bhr` / `test_fp8_bhd_bhr_hdr` 当场打印 `Filtered` | **7/7 PASS** |
| D | `bhr,hdr->bhd` calc_diff=0 且逐位；`bhd,bhr->hdr` 丢 C → 4.7e-2；`bmk,bnk->mn` 两条等价 ~1e-14 | PASS |
| E | 确定性开关切 kernel（`sm90_bf16_gemm_impl` ⊗ `nvjet_sm90_*`） | PASS |
| F | 真实 `bench_kineto` 吞吐 | PASS |

Part F 真实吞吐（注意 `bhr,hdr->bhd` 的 `h` 是 batch 维，`h=4096,b=8` 时并行度极低）：

```
"bhr,hdr->bhd" (b=8,    h=4096, r=4096, d=1024): 37709.0 us |   7 TFLOPS
"bhr,hdr->bhd" (b=8192, h=128,  r=512,  d=128 ):  1425.0 us |  96 TFLOPS
"bmk,bnk->mn"  (s=8192, m=n=k=256             ):  2211.0 us | 124 TFLOPS
```

---

## 8. 边界与 SM100 专属

H100 上**不会进入**（arch 门控跳过），标注 **SM100 专属，仅分析**：

- `test_fp8_bhd_hdr_bhr`（`:194-233`）与 `test_fp8_bhd_bhr_hdr`（`:236-259`）两个整个函数
  （`@test_filter(arch>=10)`）。
- `test_fp8_bhr_hdr_bhd` 里的 **FP8 输出对**分支（`:146-164`）——`output_dtypes` 在 sm90 只有
  `bfloat16`；以及 `BF16StochasticRounding`（`:170-180`，`cvt.rs` 仅目标卡）。
- 所有 `assert_stochastic_bf16_matches_fp32_accumulation` 调用点（`:63`、`:94`、`:179`、`:223`）。

H100 上**确实会跑**：`bmk,bnk->mn`（fp32 累积 + bf16）、两个 BF16 MLA 投影（自研 + cuBLASLt
双路）、BF16 权重梯度（cuBLASLt）、FP8 `bhr,hdr->bhd` 的 BF16 输出路径（走 SM90 `1d2d` 的
`Batched` 骨架）。

---

## 9. 小结

- `tests/test_einsum.py` 是 DeepGEMM **einsum 家族的顶层规格**：回答四个 `bmk/bnk`、
  `bhr/hdr`、`bhd/hdr`、`bhd/bhr` 语义 + 两个 FP8 变体的出口算什么对。
- 判据从松到严并用：`calc_diff`（`1e-5` / `1e-10` / `1e-3` / `2e-3`）+ **逐位 oracle**
  （direct==累积、随机舍入 quartet 哈希、FP8 输出对==生产管线）。BF16 家族在确定性模式下
  实际逐位相等，`1e-10` 阈值不是「宽松」。
- **einsum 特有的规格点**：带 padding 的权重视图（`:49-50`）、C 累积语义用大初始值放大
  （`:113`）、`bmk,bnk->mn` 的 fp32 强制 `c=d`（否则段错误）、FP8 输出对必须对齐「BF16 D →
  per-token 量化」的逐位关系（仅 SM100）。
- 实测：H100 上 **15 PASS / 0 FAIL**，直接驱动真实测试体；纯 FP8 反向/投影两函数与 FP8 输出对
  为 **SM100 专属，仅分析**。
