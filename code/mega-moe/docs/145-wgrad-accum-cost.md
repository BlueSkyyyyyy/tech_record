# MoE wgrad 的「强制累积」代价：`k_grouped_nt` 的 fp32 D 回读，到底占多少

> 本文对应路线图 **P135-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 145 篇。
>
> 第 15 篇（P7-1）用 ncu 把**反向**四段 grouped GEMM 归因清楚，其中给 wgrad 列了三条根因，
> 第二条是：**SM90 的 `k_grouped_fp8_gemm_nt_contiguous` 被 API 强制累积**，`D` 被回读，
> 让访存流量几乎翻倍，把算术强度压到 AI~120 ≪ H100 FP8 roofline 拐点 591 → **访存受限**。
> 第 22 篇（P22-1）也顺带量到「稠密 1D2D 比 1D1D 快 ~1.5×」。但**从没有一次对照实验**专门
> 回答两件事：
>
> 1. **这条根因值多少钱**？——把 wgrad 换个**不强制累积**的写法，能快多少？
> 2. 慢下来的流量里，**回读 `C`** 与 **每专家 fp32 输出** 各占多少？
>
> 本篇用一组**同输出规模、同 FLOP** 的对照把这两个问题量化。核心结论会**修正**一点直觉：
>
> **wgrad 在 DeepSeek 量级只有 ~240 TFLOPS 的真凶，不是「K 轴分组」这个 kernel 本身**
> （它甚至比「逐专家发 8 个稠密 GEMM」还快 14%），**而是「每个专家一份 fp32 [M,N] 权重梯度」
> 这个输出规模**——它让整段的 96% 流量压在权重梯度上（读+写），回读 `C` 单独占 48%。
> 把输出换成 bf16、去掉回读，等价写法能跑到 **~537 TFLOPS（2.2×）**；而把各专家**求和**成
> 一个 [M,N]（共享权重假设）能到 **~1240 TFLOPS（5.2×）**——但那是另一个算子，不是 wgrad。
>
> 可跑性：SM90 FP8 kernel（H100 可跑）。SM100 的 wgrad `k_grouped_tn_contiguous`（MN-major）
> 是**仅分析**（第 12 篇）——它同样收 fp32 累积的 `D`，代价结构一致。本机 8 卡共享，本文在
> **空闲的 GPU1** 上单进程测得，报数时注明。
>
> 原始输出：`harness/exp_wgrad_accum_cost.out.txt`
> （`harness/exp_wgrad_accum_cost.{sh,py}`；H100 80GB HBM3, sm90, 驱动 570）。

---

## 1. 合同：SM90 的 wgrad 为什么「必须」回读 D

wgrad 的 host 入口 `k_grouped_fp8_gemm_nt_contiguous`（`csrc/apis/gemm.hpp:373-427`）有两条硬约束：

- 必须是 **1D1D** kernel：`DG_HOST_ASSERT(recipe == std::make_tuple(1, 1, 128))`（`gemm.hpp:384`）；
- **`C` 必须提供**：`DG_HOST_ASSERT(c.has_value() and c.value().is_contiguous())`（`gemm.hpp:403`），
  随后 `sm90_k_grouped_fp8_gemm_1d1d(...)` 把 `c` 与 `d` 一起传下去（`gemm.hpp:422`）。

而 1D1D kernel 的契约在 `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`：

```
DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
```

即 **`D` 必须是 fp32、且必须累积**（`Desc.with_accumulation = c.has_value()`，`hpp:94`）。落到 kernel
里，输出不是普通 TMA store，而是 **`SM90_TMA_REDUCE_ADD_2D`**——把结果「加」到已存在的 `D` 上
（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`）：

```
if (with_accumulation) cute::SM90_TMA_REDUCE_ADD_2D::copy(...);
```

**为什么非累积不可？** 这是**训练语义**决定的，不是 kernel 偏心：

- 每个专家有**自己的**权重，所以 wgrad 要产出 `dW[e] = A_e^T B_e`，**每个专家一份** `[M,N]`
  梯度（`dW1: [E,2I,H]`、`dW2: [E,H,I]`）——输出天然是单专家的 **E 倍**；
- 参数梯度要跨 micro-batch **累积**（gradient accumulation），所以 `D += A^T B` 是刚需，
  且必须 fp32（bf16 累加会在长序列上丢精度）。

对照之下，**dgrad**（`m_grouped_fp8_gemm_nt_contiguous`）不等长在 M 轴、`D` 是 **bf16 直写**
（`gemm.hpp:209` `DG_HOST_ASSERT(d.scalar_type() == torch::kBFloat16)`），**没有累积、没有回读**。
所以「wgrad 比 dgrad 慢」里，有一部分是**结构不对称**，不是实现 bug。

---

## 2. 方法：同输出规模 / 同 FLOP 的三层对照

要量化「强制累积」，必须把**其它变量按住**。本篇在 DeepSeek 量级
（`T=2048, topk=2, E=8` → `sum_k=4096`、每专家 `k=512`；`dW2: M=I=2048,N=H=7168`、
`dW1: M=H=7168,N=2I=4096`）上量四组：

| 记号 | 形态 | 输出规模 | 累积 | 说明 |
|---|---|---|---|---|
| **W** | `k_grouped_nt`（真实 wgrad） | `E×M×N` fp32 | 是（读 C） | 被检验的真身 |
| **W₂** | `E` 个独立稠密 1D1D | `E×M×N` fp32 | 是（读 C） | 只换掉「K 分组」这个 kernel |
| **W₃** | `E` 个独立稠密 1D2D | `E×M×N` **bf16** | 否 | 只换掉「fp32+累积」 |
| **D** | `m_grouped_nt` 单组 | `1×M×N` bf16 | 否 | 把各专家**求和**，是**另一个算子** |

**W vs W₂** 隔离「K 分组 kernel 开销」；**W₂ vs W₃** 隔离「fp32 累积 vs bf16 直写」；
**W vs D** 给一个「不同算子」的上界参照（各专家共享一份权重时）。

先验证 W 与 D 描述的是**同一张量**：`k_grouped` 的 `∑_e dW[e]` 与 `m_grouped` 单组在
**同一份 A、B** 上比，相对误差 **3.68e-2**（`exp_wgrad_accum_cost.out.txt` Part A），正是
FP8 量化噪声——说明两者数学一致，差异只在输出形态。

---

## 3. 结果

### 3.1 wgrad 的真身有多慢（Part B）

| 形状 | `k_grouped`(真实) | `m_grouped` 单组(求和) | `E×`稠密 1D2D bf16 | `E×`稠密 1D1D fp32 |
|---|---:|---:|---:|---:|
| `dW2` M=2048,N=7168 | 500.6 us / **240 TFLOPS** | 97.3 us / 1237 | 223.1 us / 537 | 578.5 us / 208 |
| `dW1` M=7168,N=2I=4096 | 997.0 us / **241 TFLOPS** | 188.0 us / 1279 | 313.8 us / 767 | 1041.9 us / 231 |

### 3.2 三个可分离因子（Part C）

| 因子 | 对照 | 比值 | 判读 |
|---|---|---:|---|
| ① 纯累积（输出 1×、计算受限形状） | 稠密 1D1D fp32/acc vs 1D2D bf16 | **1.37×** | 同形状下回读+fp32 输出只值 1.37× |
| ② K 分组 kernel 本身 | `k_grouped` vs `E×`稠密 1D1D（同 G× 输出、都 fp32/acc） | **0.86×** | **K 分组不但不慢，还快 14%** |
| ③ fp32 累积 vs bf16 直写（G× 输出规模） | `E×`稠密 1D1D vs `E×`稠密 1D2D | **2.58×** | **主因** |
| 合成 | `k_grouped` / `E×`稠密 1D2D | **2.23×** | wgrad vs 同输出规模的「理想无回读 bf16」 |

**因子② 是反直觉的一条**：把 K 切成 8 组、每组只有 4 个 `BLOCK_K`，并没有拖慢 kernel——
`k_grouped` 甚至比手动发 8 个稠密 GEMM 快 14%（省了 7 次发射与调度）。所以「wgrad 慢是因为
K 被分组」这个说法**不成立**。

### 3.3 per-group K 扫描（Part D）

固定 `E=8`、`M=2048,N=7168`，扫每专家 token 数 `k_e`：

| `k_e` | 128 | 256 | 512 | 1024 | 2048 | 4096 |
|---|---:|---:|---:|---:|---:|---:|
| `k_grouped` TFLOPS | 63 | 125 | **240** | 450 | 807 | 842 |
| `m_grouped`(求和) TFLOPS | 850 | 1096 | 1258 | 1318 | 1347 | 1269 |

随着 `k_e` 变大，wgrad 从 63 爬到 842 TFLOPS——因为**每专家 fp32 输出是固定的 `E·M·N`**，
`k_e` 越小，输出相对收缩维越重，越访存受限。DeepSeek 训练里每专家 token 数常落在几百
（`k_e≈512` 即 240 TFLOPS），正落在曲线的陡段。

### 3.4 访存账 + 算术强度（Part E）

| 写法 | 读 MB | 写 MB | 合计 MB | AI | 判读 |
|---|---:|---:|---:|---:|---|
| **W** `k_grouped` 实际 | 508.7 | 469.8 | **978.5** | **122.9** | 每专家一份 fp32 + 读回 C |
| W₂ 理想 wgrad（fp32，无回读） | 38.9 | 469.8 | 508.7 | 236.4 | 去掉 C 回读 |
| W₃ 理想 wgrad（bf16，无回读） | 38.9 | 234.9 | 273.8 | 439.2 | 再去掉 fp32 写 |
| D `m_grouped` 单组（求和） | 38.9 | 29.4 | 68.3 | 1761.0 | 非 wgrad |

H100 FP8 计算 roofline 拐点是 `1979 TFLOPS / 3.35 TB/s ≈ 591 FLOP/B`。**W 的 AI=123 ≪ 591
→ 访存受限**；W₂（去回读）AI=236，**还是**访存受限；只有 W₃（bf16 输出）AI=439 才逼近拐点，
对应实测的 537 TFLOPS。所以：

- **C 回读单独占 wgrad 总流量的 48%**（去掉它，AI 翻倍，但仍是访存受限，理论上限 ~2×）；
- **每专家 fp32 输出（读+写）合计占 96%**——这是更大的一块，且是 wgrad 的**定义**（每专家一份
  参数梯度），不是实现选择。

### 3.5 ncu 硬指标（单次 launch，热 cache）

| kernel（ncu 名） | DRAM 读 | DRAM 写 | 时长 |
|---|---:|---:|---:|
| `sm90_fp8_gemm_1d1d_impl`（W，k_grouped） | 528.3 MB | 449.8 MB | 486.5 us |
| `sm90_fp8_gemm_1d2d_impl`（D，m_grouped 单组） | 41.0 MB | 19.4 MB | 99.5 us |
| `sm90_fp8_gemm_1d1d_impl`（稠密 1D1D，输出 1×） | 99.9 MB | 42.7 MB | 146.2 us |
| `sm90_fp8_gemm_1d2d_impl`（稠密 1D2D，输出 1×） | 47.1 MB | 19.7 MB | 111.8 us |

ncu 读到 W 的 528+450=978 MB，与 Part E 的解析账 **978.5 MB 吻合到 0.1%**。两个 1D1D kernel
（W 与稠密）的 DRAM 流量都远大于同形状 1D2D（978 vs 67 MB），坐实「1D1D = 读回 C + 写 fp32 D」
把流量抬到 1D2D 的 ~14 倍。

---

## 4. 结论：wgrad 该往哪儿优化

把三条根因按「可动性」重排（对第 15 篇的细化）：

1. **不可动：每专家 fp32 参数梯度**。它是 wgrad 的语义本体（`dW: [E,·]`、跨 step 累积），
   输出规模 `E·M·N·4` 且要读+写。这一条决定了 wgrad 大概率的访存下界。
2. **可动但收益有限：C 回读**。占流量 48%，去掉后 AI 翻倍（123→236），**仍 < 591**，
   所以最多 ~2×、且需要改累积协议（如把 K 切分成 fresh 输出再单独归约）。
3. **不是问题：K 轴分组 kernel**。因子② 实测 0.86×——分组是**帮忙**的。

因此，「让 wgrad 更快」的现实路径不是换 kernel，而是：

- **减小输出规模**：走 **bf16（或分块 fp32）参数梯度 + 更高精度只在优化器侧**——W₃ 的 537 TFLOPS
  就是这条路的上界（2.2×）；
- **把累积搬到更便宜的地方**：例如按 K 分 chunk 写 fresh 输出，再用一次带宽友好的归约，
  省掉「原地 2×D」；
- **接受它是访存受限**：把 wgrad 与 dispatch/combine 的通信在 **SM 预算**上分区重叠掉（第 104/144 篇）。

---

## 5. 与 SM100 的关系（仅分析）

SM100 的 wgrad 走 `k_grouped_fp8_gemm_tn_contiguous`（MN-major、`gemm.hpp:320-371`，第 12 篇），
`C/D` 是 3D `[G,M,N]`、**同样收 fp32 累积的 `C`**（`sm100_k_grouped_fp8_gemm_1d1d`）。所以
「每专家 fp32 输出 + 回读」的结构在 SM100 上**一致**；差别只在算力/带宽比：SM100 的 FP8 峰值与
HBM 带宽都更高，但拐点（FLOP/B）量级相近，故 wgrad 在 SM100 上**仍是访存受限**。真正能改变
这一点的只有输出精度/累积协议。此处 SM100 kernel **仅分析**（本机 H100 跑不了）。

---

## 6. 边界

- **能跑/能核**：全部是 SM90 FP8 kernel，H100 真机；正确性 sanity（同数学 relerr 3.68e-2）、
  event 计时、ncu 单次 launch 硬指标、解析访存账四者互相印证（978 MB 对 0.1%）。
- **不能核**：SM100 的 wgrad `tn` 路径只做源码分析；`m_grouped` 单组是**不同算子**（各专家求和），
  只作上界参照，**不等于 wgrad 可达值**。
- **环境**：单进程、空闲 GPU1（`CUDA_VISIBLE_DEVICES=1`，`util=0%`）；8 卡节点共享，若同卡有
  争用数字会更低（沿用本专题报数约定）。
- **不变量**：`E×`稠密 GEMM 的计时按「顺序发 E 个 kernel」量墙体时间，含发射开销——这正是
  `k_grouped` 略快的原因；若要更纯的对照可改用 CUDA Graph 合并发射，但这会削弱「分组省发射」
  这个真实优势，故保留。

---

## 7. 复现

```bash
bash harness/exp_wgrad_accum_cost.sh > harness/exp_wgrad_accum_cost.out.txt
```

脚本先跑 `harness/exp_wgrad_accum_cost.py`（Part A–E），再用 ncu 单次 launch 采
`dram__bytes_{read,write}.sum` / `gpu__time_duration.sum` 核对四个 kernel 的访存账。

### 源码坐标（`~/github/DeepGEMM@057ca59`，行号已核对）

- `csrc/apis/gemm.hpp:373-427`：`k_grouped_fp8_gemm_nt_contiguous` 合同（`:384` recipe、`:403` `c` 必填、
  `:422` 调 `sm90_k_grouped_fp8_gemm_1d1d`）。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84/94`：`c.has_value() and d==float`、`with_accumulation`。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`：`SM90_TMA_REDUCE_ADD_2D`。
- `csrc/apis/gemm.hpp:181-250`：`m_grouped_fp8_fp4_gemm_nt_contiguous`（`:194` `[M,K]@[G,N,K]`、`:209` `d==bf16`）。
- `csrc/apis/gemm.hpp:320-371`：SM100 `k_grouped` tn（仅分析）。

### 实验

- `harness/exp_wgrad_accum_cost.py`（Part A 正确性、B wgrad/dgrad、C 三因子、D K 扫描、E 访存账）。
- `harness/exp_wgrad_accum_cost.sh`（driver + ncu）。
- `harness/exp_wgrad_accum_cost.out.txt`（原始输出）。

### 线上资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- Mega MoE PR #316：https://github.com/deepseek-ai/DeepGEMM/pull/316
- 本专题：`docs/15`（反向 ncu 归因）、`docs/12`（wgrad API）、`docs/22`（SM90 FP8 主 kernel）、
  `docs/104`（反向通信预算）、`docs/144`（前向重叠真机）。
