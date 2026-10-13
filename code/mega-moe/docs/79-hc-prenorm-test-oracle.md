# HyperConnection prenorm GEMM 的验收测试即正确性规范（`tests/test_hyperconnection.py`，H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P69 增量（第 79 篇）。
> 第 16 篇（P7-2）拆了 `deep_gemm.tf32_hc_prenorm_gemm` 的**实现**——host 契约、SM90 kernel 的
> 「A 搬运时顺路平方累加」融合支点、TF32 是 RNE 的实测判定、split-K 与 roofline。
> 本篇补上它一直缺席的另一半：**「这份算子算什么对」的唯一规范** `tests/test_hyperconnection.py`。
> 前面第 41/47/50/57/64/66/68 篇分别把 Mega MoE / Mega Gate / MQA logits / Mega mHC / 稠密 GEMM /
> einsum / SF 布局的**验收测试**当「可执行规格」拆透；本篇是同一条线的延续，而这份测试还有
> 一个别的测试没有的看点：**它的参考 GEMM 必须依赖 `allow_tf32=True`**——判据本身编码了
> 「kernel 是 TF32」这条事实。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.1.1.post3）。SM90 有实现，
> 故本机 H100 可**真跑整份测试**；SM100 分支（`sm100_tf32_hc_prenorm_gemm.cuh`）为**仅分析**。

## 1. 这份测试是「唯一规范」，且它在 H100 上真的能跑

`tests/test_hyperconnection.py` 只有 57 行，只有一个测试函数 `test_hc_prenorm_gemm`
（`:14-45`），由 `__main__` 无条件调用（`:50-57`）。它的门控极宽：

```
test_hyperconnection.py:14: @test_filter(lambda: get_arch_major() >= 9)
test_hyperconnection.py:15: def test_hc_prenorm_gemm() -> None:
```

`@test_filter` 来自 `deep_gemm/testing/utils.py:11-22`：条件为假时只打印 `Filtered by ...`
并跳过（第 40 篇拆过）。`get_arch_major`（`testing/utils.py:6-8`）返回
`torch.cuda.get_device_capability()[0]`。**`>= 9` 意味着 H100（sm90）与目标卡（sm100）
都能跑**——这与 Mega MoE / Mega Gate / 大部分 MQA logits 测试（`== 10` 门控，H100 直接跳过）
形成对照。本机实测：`get_arch_major() = 9`，测试体 24 个用例全部 assert 通过（§5）。

`__main__` 之前会 `torch.manual_seed(0)` / `random.seed(0)`（`:51-52`），所以每次运行形状与
数据都可复现——这是它作为「规范」而非「基准」的前提。

## 2. 用例矩阵：24 个 deterministic 形状

```
test_hyperconnection.py:21: for m in (13, 137, 4096, 8192):
test_hyperconnection.py:22:     for n, k in [(24, 28672), (24, 7680), (24, 7168)]:
test_hyperconnection.py:23:         for num_splits in [None, 16]:
```

四类形状各有意图：

| 维度 | 取值 | 意图 |
|---|---|---|
| `m` | 13 / 137 | 极小 M：并行度不足，逼出 split-K 的价值（`ceil(m/64)` = 1 / 3 个 M block） |
| `m` | 4096 / 8192 | 大 M：验证稳定吞吐与访存 roofline |
| `n` | 恒 24 | HyperConnection 的**系数通道数**很小（如 `hc×4` 之类），`n` 小是这个算子的定义特征 |
| `k` | 28672 / 7680 / 7168 | 真实模型里的隐藏维/中间维；`k=7168` 与 `k=7680` 只差一个对齐块 |
| `num_splits` | None / 16 | 关/开 split-K，验证两形态输出一致且小 M 加速 |

`n=24` **不是**随便取的：它同时满足 SM90 的 `n <= 32 and n % 8 == 0` 契约
（`sm90_tf32_hc_prenorm_gemm.hpp:30`），又落在 TF32 WGMMA 的 `WGMMA_N=BLOCK_N=align(24,16)=32`
最小对齐上。SM100 把 `n` 上限放宽到 **128**（`sm100_tf32_hc_prenorm_gemm.hpp:29`
`n <= 128 and n % 8 == 0`），测试矩阵却不改——因为规范关心的是**语义**，而非把每个 arch 的
上限都跑满（上限契约由 §6 的负样本单独覆盖）。

## 3. 参考数学与判据：三步，且三步都编码了实现事实

`test_hc_prenorm_gemm` 每个用例体（`:24-38`）：

```python
a = torch.randn((m, k), dtype=torch.bfloat16, device='cuda')          # :24
b = torch.randn((n, k), dtype=torch.float,    device='cuda')          # :25
d = torch.empty((m, n), ...) if num_splits is None else torch.empty((num_splits, m, n), ...)   # :26-27
s = torch.empty((m, ), ...) if num_splits is None else torch.empty((num_splits, m), ...)       # :28-29
deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, num_splits=num_splits)     # :30
final_d = d if num_splits is None else d.sum(0)                       # :31
final_s = s if num_splits is None else s.sum(0)                       # :32
ref_d = a.float() @ b.T                                               # :34
ref_s = a.float().square().sum(-1)                                    # :35
diff = max(calc_diff(final_d, ref_d), calc_diff(final_s, ref_s))      # :37
assert diff < 1e-8, ...                                               # :38
```

三个关键约定：

**3.1 「参考 GEMM 必须开 TF32」——测试开头两句就是判据的一部分**

```
test_hyperconnection.py:17: torch.backends.cuda.matmul.allow_tf32 = True
test_hyperconnection.py:18: torch.backends.cudnn.allow_tf32 = True
```

这不是「为了快」。`ref_d = a.float() @ b.T`（`:34`）是一个 torch GEMM：若 `allow_tf32=False`，
它算的是**真 fp32** matmul；而 kernel 的 D 走 **TF32**（10-bit 尾数，第 16 篇 §3.4）。
两者相差的是 TF32 量化噪声（~2e-4 相对范数），`calc_diff` 会落到 ~2e-8，**越过 1e-8 阈值**。
打开 `allow_tf32=True`，torch 参考也走 TF32，两者才在同一精度层对齐，1e-8 的紧阈值才有意义。

本机实测坐实（`exp_hc_prenorm_test_oracle.out.txt` Part E1）：把参考换回真 fp32 后，
`calc_diff(D, 真fp32) = 2.151e-08`（m=4096,k=7168）与 `2.394e-08`（m=8192,k=28672），
**正好越过 1e-8**。所以这两行 `allow_tf32` 是规范的必需项，而不是可选的加速开关。
反过来，`ref_s = a.float().square().sum(-1)`（`:35`）是**精确**的：A 是 bf16，上转 fp32 无损，
平方和在 fp32 里几乎精确（本机 vs kernel S 的 `calc_diff` 仅 3.4e-14 / 1.2e-12）。

**3.2 `a.float() @ b.T` 揭示 B 的角色**

`ref_d` 用的是 `a.float() @ b.T`（**不是** `torch.einsum('mk,nk->mn')`）。`b` 是 fp32 `(n,k)`，
直接 `.T` 得到 `(k,n)`，与 `(m,k)@(k,n)` 一致。更重要的信息在 A：参考把 **A 显式上转 fp32**
（`a.float()`），而**完全不转 B**——因为 kernel 里 A 是 bf16→fp32（无损）后进 WGMMA，B 的
fp32 低 13 位由硬件按 TF32 处理。参考侧的「只转 A、不动 B、再开 TF32」精确对应了 kernel 的
数据路径（第 16 篇 §3.3-§3.4）。

**3.3 判据是 `calc_diff < 1e-8`，不是逐位**

与第 41/68 篇那些「`torch.equal` 逐位验收」的测试不同，这里用的是松得多的
`calc_diff < 1e-8`（`testing/numeric.py:5-11` 的归一化平方误差 `1 − 2⟨x,y⟩/(‖x‖²+‖y‖²)`）。
**为什么不能逐位**：D 是 TF32 输出，而参考也是 torch 的 TF32 GEMM，两者在**累加顺序**上不同
（kernel 的 split-K、tile 顺序；torch 的实现各异），逐位不可能相等；S 的平方和也随归约树而变。
1e-8 这个阈值正好卡在「TF32 同层对齐后残差」之下、「关掉 TF32 的失配」之上（§3.1），
是一个被精心选过的分离带。

## 4. FLOP / byte 账目口径：S 不进 FLOP

```
test_hyperconnection.py:40: t = bench_kineto(lambda: deep_gemm.tf32_hc_prenorm_gemm(a, b, d, s, ...), 'tf32_hc_prenorm_gemm', ...)
test_hyperconnection.py:41-44: print(... f'{2 * m * n * k / t / 1e12:4.0f} TFLOPS | '
                                        f'{count_bytes(a, b, d, s) / 1e9 / t:4.0f} GB/s')
```

- **FLOP 系数 = `2*m*n*k`**：只数 D 的 MAC（矩阵乘 `m·n·k` 次乘加 = `2mnk` FLOP），
  **`S = Σ A²` 的 `m·k` 次平方累加一个都不算**。这正是「融合」的语义：S 是搭 D 的车白算的，
  若把它也算进 FLOP，就变成「这个 kernel 在算两件事」，掩盖了它天生访存受限的事实。
- **byte 口径 = `count_bytes(a, b, d, s)`**（`testing/numeric.py:14-21`）：递归把
  `(data, sf)` 对与 `None` 处理掉、数 `numel*element_size`。这里 a/b/d/s 都是普通张量，
  故 = A 的 bf16（`2mk`）+ B 的 fp32（`4nk`）+ D 的 fp32（`4mn`）+ S 的 fp32（`4m`）。
  对 `m=8192,n=24,k=28672`：A 独大（469.76 MB / 473.33 MB），规范化 **AI ≈ 23.8 FLOP/B**，
  与第 16 篇量到的 **AI ≈ 17–31** 同量级——**访存受限**，所以「省一遍 A」比「优化 MMA」值钱。

本机实测（Part F）：`2mnk = 11.27 GFLOP`、`count_bytes = 473.33 MB`（A 占 469.76 MB）、
`FLOP/B = 23.82`，与第 16 篇结论一致。

## 5. H100 实测：直接驱动真实测试体

实验 `harness/exp_hc_prenorm_test_oracle.py`（+`.sh` / `.out.txt`）**直接 import 真实测试模块**
（`tests/test_hyperconnection.py`，把它所在目录加进 `sys.path` 以解析 `from generators import ...`），
调用**真实测试体** `test_hc_prenorm_gemm()`，测试体一行未改。**8 PASS / 0 FAIL**。

**Part C —— 正确性 oracle（打桩 `bench_kineto` 只留断言）**：把 `th.bench_kineto` 临时替换为
返回 `1.0` 的桩，跑完 24 个用例，**无一断言被触发**（`no assertion violated`）。这验证的正是
§2-§3 那套参考数学 / 阈值 / 形状矩阵在 SM90 上全部成立。`m=13` 等小形状、`num_splits=16`
分段写 D/S 再 `sum(0)` 的路径都覆盖到了。

**Part D —— 用真实 `bench_kineto` 跑测试自带的 perf 路径**，捕获它打印的 24 行吞吐表。
**注意**：本机 GPU 被仓库内多个自驱任务并行共享（实测 device 0 利用率 100%、load avg ≈ 46），
Part D 的数字抖动很大（同一形状 `k=7168` 与 `k=28672` 的耗时不成比例、split-K 偶发慢数倍）。
因此**本篇不把 Part D 当性能结论**，干净的单 kernel 数字仍以第 16 篇为准（`m=8192,n=24,k=28672`
**166.7 us / 67.8 TFLOPS / 2847.8 GB/s**，即 HBM 峰值 ~85%）。Part D 的价值只在于**证明
「测试自带的 perf 路径确实能跑通、`bench_kineto` 确实抓到 `tf32_hc_prenorm_gemm` 这个 kernel 名」**
（它用 `bench_kineto` 的按名字匹配断言，名字对不上会直接抛错）。

**Part E —— 利齿与盲区**：

- **E1**：如 §3.1，关 `allow_tf32` 使 `calc_diff(D, 真fp32)` = 2.15e-08 / 2.39e-08，**越过 1e-8**。
  这是本篇最硬的一条「为什么」。
- **E2**：host 参数契约真的会拒（异常文本直接带源码行号）：

  ```
  [拒绝] n=64  (>32)     : sm90_tf32_hc_prenorm_gemm.hpp:30  n <= 32 and n % 8 == 0
  [拒绝] n=20  (n%8!=0)  : sm90_tf32_hc_prenorm_gemm.hpp:30  n <= 32 and n % 8 == 0
  [拒绝] k=4104 (k%64!=0): sm90_tf32_hc_prenorm_gemm.hpp:31  k % block_k == 0
  [通过] n=24,k=4096 / n=32,k=7168
  ```

  `n=32` 是 SM90 允许的最大值（`block_n = align(32,16) = 32`），通过；`n=64` 需要 SM100。
- **E3**：split-K 可加性。`num_splits=16` 的 `sum(0)` 与非 split 比：`calc_diff(D)=1.277e-10`、
  `calc_diff(S)=4.219e-14`；16 个 split 的 D Frobenius 范数 `min=6577.9 / max=6748.1 / mean=6653.9`
  （±1.3% 均匀）——因为 `k=7168` 共 `112` 个 K block，`112/16=7` 无 remainder，切得均匀。

## 6. host 契约的完整链条（测试没显式写、但由它间接锁定）

测试只喂合法形状，非法形状的拒绝由 host 保证，链条两层：

- **入口契约**（`csrc/apis/hyperconnection.hpp:10-44`）：
  `get_major_type_ab(a)==Major::K` / `...b...==Major::K`（`:16-17`，A/B 必须 K-major）、
  `check_major_type_cd(d)`（`:18`，D 必须 N-major）、`sqr_sum.is_contiguous()`（`:21`，S 必须连续）；
  dtype 四连（`:37-40`）：`a=kBFloat16`、`b=kFloat`、`d=kFloat`、`sqr_sum=kFloat`；
  split 形状一致性（`:26-35`）：`num_splits` 给出时 `d` 是 3D `(num_splits,m,n)`、`sqr_sum` 是
  `(num_splits,m)`，且 `m/n/k` 三者一致；`m==0` 直接返回（`:43-44`）。
- **实现契约**（`sm90_tf32_hc_prenorm_gemm.hpp:21-31`）：`block_m=block_k=64`、
  `num_math_threads=num_tma_threads=128`、`block_n=align(n,16)`、`n<=block_n`、
  `n<=32 and n%8==0`、`k%block_k==0`。§5-E2 实测这些断言真的拦人。

一个隐含但重要的点：测试**不检查 major**（`a`/`b` 都由 `torch.randn` 生成，天然 K-major 连续），
也不检查 D 的 major。所以 `hyperconnection.hpp:16-18` 的 major 契约在测试里是「被满足」而非
「被验证」——这与第 46 篇「`nt` 是唯一实现、主序由入口层转换」是同一套设计。

## 7. 与第 16 篇的关系、小结

| | 第 16 篇（P7-2） | 本篇（P69-1） |
|---|---|---|
| 对象 | `sm90_tf32_hc_prenorm_gemm.cuh` 实现 | `tests/test_hyperconnection.py` 规范 |
| 回答 | 「怎么算的、为什么这样算」 | 「算什么对、判据如何编码实现事实」 |
| 硬结论 | TF32 对 fp32 是 RNE（30× 优于截断）、融合省一遍 A、访存受限 | 「参考 GEMM 必须开 TF32」是判据一部分（关掉即越过 1e-8）；S 不进 FLOP；split 可加 |

**三个可复用结论**：

1. **判据会编码实现事实**：`allow_tf32=True`（`:17-18`）不是加速，而是「kernel 是 TF32」这句话
   在参考侧的必要镜像。看测试就能反推算子的精度语义——这是「验收测试即正确性规范」的典型例子。
2. **「融合」在账目上表现为「少算一项 FLOP」**：`2*m*n*k` 里没有 S 的份额，因为 S 的平方累加
   藏在 A 的搬运里；配合 `count_bytes` 算出的 AI ≈ 24 FLOP/B，一眼看出这是访存受限算子。
3. **门控 `arch>=9` 使它是少数 H100 能整份跑通的 mega-mHC 家族测试**：正确性 oracle 在 sm90
   上全部成立，SM100 只放宽 `n` 上限（32→128），语义不变——与第 67 篇（Mega mHC 测试 oracle）
   的「门控 `==10`、H100 整体跳过」正好互补。

**本篇实测脚本**：`harness/exp_hc_prenorm_test_oracle.py` / `.sh` / `.out.txt`（H100 真机，
直接 import 真实测试模块、驱动真实测试体，8 PASS / 0 FAIL）。
**SM100 分支**（`sm100_tf32_hc_prenorm_gemm.cuh` / `sm100_tf32_hc_prenorm_gemm.hpp:21-30`）为
**仅分析**——本机 H100 不可运行；`n<=128` 是唯一的契约差异。
