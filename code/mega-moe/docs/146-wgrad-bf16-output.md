# MoE wgrad 换 bf16 输出：2.2× 上界的落地与精度代价

> 本文对应路线图 **P136-1**，是「DeepSeek Mega MoE 前向 / 反向算子实现解析」的第 146 篇。
>
> 第 145 篇（P135-1）把 wgrad 慢的根因**量化**清楚，结论是：真凶不是「K 轴分组」这个 kernel，
> 而是 **每个专家一份 fp32 `[M,N]` 权重梯度**（读+写占 96% 流量）。它同时给出一条**优化路线**——
> 把参数梯度换成 **bf16 直写**的等价格式，实测能跑到 **537 TFLOPS（2.23×）**。但第 145 篇**只量了性能，
> 没回答代价**：把参数梯度从 fp32 换成 bf16，**精度掉多少**？跨 micro-batch **累积 S 步**之后会不会爆？
>
> 本篇用**真实 SM90 kernel** 把这条 2.2× 上界**落地**，并把精度代价拆开。核心结论有两层：
>
> 1. **精度上基本「免费」**：bf16 输出的误差是**一次末次舍入**，relerr ~1.4e-3，比 wgrad 本来
>    就有的 **FP8 输入量化噪声（3.6e-2）还小 25.8×**，也比 fp32 累积误差（1.2e-4）×12 大——
>    相对于「训练本来就在 FP8 噪声里跑」，bf16 输出不是瓶颈；方向 `cos(bf16, fp32)=0.99932`。
> 2. **但收益要赔在「归约」上**：真实计时里，**GEMM 本身快 2.27×**（500.5→220.8us/步），
>    可把每步 bf16 fresh 结果再 `Σ` 成一份 fp32 梯度要 **838.6us**（`S=8`），端到端 **1.54×**，
>    且**在 `S≈8–16` 封顶 ~1.55×**（`S=1` 甚至 **0.83×**，比 fp32 还慢）。
>    **只有让优化器直接吃 bf16 梯度、不再物化 fp32**，2.2× 才拿得回来。
>
> 可跑性：全部是 SM90 FP8 kernel（H100 可跑）。SM100 的 wgrad `k_grouped_tn_contiguous` 是同族
> 1D1D fp32 累积路径，**仅分析**。本机 8 卡共享，本文在**空闲 GPU1** 单进程测，报数注明。

---

## 1. 合同：SM90 的 wgrad 只有两种输出形态，没有「fp32 直写」

先把可选项钉死。第 12/145 篇已核过，SM90 的 wgrad 入口 `k_grouped_fp8_gemm_nt_contiguous`
（`csrc/apis/gemm.hpp:373-427`）有两条硬约束：`recipe == (1,1,128)`（`:384`）与 **`C` 必填**
（`:403` `DG_HOST_ASSERT(c.has_value() and ...)`）。它唯一的落点是 **1D1D** kernel
`sm90_k_grouped_fp8_gemm_1d1d`，而 1D1D 的 host 契约更死（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84`）：

```
DG_HOST_ASSERT(c.has_value() and d.scalar_type() == torch::kFloat);
```

即 **`D` 必 fp32、且必累积**（`with_accumulation = c.has_value()`，`hpp:94`），落到 kernel 是
**`SM90_TMA_REDUCE_ADD_2D`**（`deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`）。

另一侧，**1D2D** kernel 的 host 契约是**互补**的（`csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`）：

```
DG_HOST_ASSERT(not c.has_value() and d.scalar_type() == torch::kBFloat16);
```

所以 SM90 上**只有两种**输出形态，中间没有第三种：

| 形态 | kernel | `D` 类型 | 累积 | 回读 `C` |
|---|---|---|---|---|
| **W**（真实 wgrad） | 1D1D `k_grouped` | fp32 | 强制 | 是 |
| **W₃**（等价写法） | 1D2D `E×` 稠密 | **bf16** | 否 | 否 |

要「省回读 + bf16 输出」，只能走 1D2D；而 1D2D 的 wgrad 得**逐专家发 `E` 个稠密 GEMM**
（收缩维 K=该专家的 token 数、A/B 转成 `[M,k_e]`/`[N,k_e]` 的 K-major、A 用 1×128 SF、B 用 128×128 SF）。
**没有「fp32 直写、无回读」这个选项**——这是本篇全部结论的合同基础。

---

## 2. 方法：精度三层分解 + S 步累积 + 落地计时

任务：量化「把 wgrad 的输出从 fp32（累积）换成 bf16（直写）」的精度代价，并落地 2.2× 上界。

**基准（可跑）**：DeepSeek 量级 `T=2048, topk=2, E=8` → 每专家 `k_e=512`，`sum_k=4096`；
形状 `dW2: M=I=2048, N=H=7168`。三个部分：

- **Part A 单步精度分解**：同一份未量化激活 `Ag/Bg`，分别构造两条路径的**量化操作数**，在 **fp64**
  里算各自「精确」`dW[e]=A_eᵀB_e`（逐专家）。然后量四项：
  - **FP8 输入量化噪声**：量化参考 vs 未量化 fp64 —— wgrad 本来就有的地板；
  - **REF fp32 累积误差**：真实 `k_grouped` vs 量化 fp64；
  - **仅末次 bf16 舍入**：把 REF 的 fp32 结果 `cast` 到 bf16 再比 —— 输出格式的**纯**代价；
  - **ALT 总误差**：1D2D bf16 结果 vs 其量化 fp64。
  外加方向 `cos(ALT, REF)` 与 `GNS=‖ALT−REF‖/‖REF‖`。
- **Part B 跨步累积**：`S∈{1,2,4,8}`。**REF** = 单 fp32 累加器 `C=D` 跑 S 步；**ALT** = 每步 `E` 个
  1D2D bf16 fresh，再 `Σ_s`（fp32 归约）。两者都 vs 各自 fp64 累积参考。
- **Part C 落地计时**：`S=8` 的真实墙体时间——REF / ALT（GEMM）/ 归约，各段单列。

---

## 3. 结果

### 3.1 单步：bf16 输出误差比 FP8 输入噪声小 25.8×

Part A（`dW2`，fp64 参考，`harness/exp_wgrad_bf16_output.out.txt`）：

| 量 | relerr | 判读 |
|---|---:|---|
| FP8 输入量化噪声（vs 未量化 fp64） | **3.63e-2** | wgrad 本来就有的地板 |
| REF `k_grouped` fp32 累积 vs fp64 | **1.18e-4** | 单累加器几乎无损 |
| 仅末次 bf16 舍入（fp32→bf16） | **1.41e-3** | 2⁻⁹ 量级 |
| ALT 稠密 1D2D bf16 vs fp64 | **1.42e-3** | = 末次舍入（无累积回读） |
| REF vs ALT（含两侧量化差异） | 3.67e-2 | ≈ FP8 噪声 |

方向：`cos(ALT, REF) = 0.9993230`、`GNS = 3.679e-2`。**关键比例**：bf16 输出误差 1.4e-3 比
FP8 输入噪声 3.6e-2 **小 25.8×**，比 fp32 累积误差 1.2e-4 大 ~12×。

一句话：**在 wgrad 已经用 FP8 操作数的前提下，输出换 bf16 引入的误差落在噪声地板以下**——
精度上基本「免费」。（`REF vs ALT` 的 3.7e-2 主要来自「两路径的 SF 粒度不同」：1D2D 的 B 用 128×128
block SF，而真实 wgrad 的 A/B 都是 1×128；这是实验口径，不是 bf16 输出的代价。）

### 3.2 累积 S 步：bf16 误差随 S 增长、但仍远在 FP8 噪声之下

Part B（`dW2`，S 步累积）：

| S | REF fp32 relerr | ALT bf16 relerr | cos(ALT,REF) | ALT/REF |
|---:|---:|---:|---:|---:|
| 1 | 1.18e-4 | 1.42e-3 | 0.9993236 | 12.0 |
| 2 | 1.22e-4 | 1.54e-3 | 0.9993235 | 12.6 |
| 4 | 1.25e-4 | 1.60e-3 | 0.9993229 | 12.9 |
| 8 | 1.26e-4 | 1.63e-3 | 0.9993228 | 13.0 |

**REF 几乎不涨**（1.18e-4→1.26e-4，fp32 单累加器）；**ALT 随 S 缓涨**（1.42e-3→1.63e-3）。
为什么涨幅这么小？因为 ALT 的误差是**每步一次随机符号的 bf16 舍入**，累积时误差按 `√S` 涨，
而和的量值也按 `√S` 涨（各步梯度近似独立），**相对误差近似恒定 ~ε_bf16**。方向 `cos` 稳定在 0.99932
（由 FP8 输入差异主导，bf16 部分淹没其中）。结论：**bf16 输出在梯度累积下依然稳**。

### 3.3 落地计时：GEMM 快 2.27×，端到端只在 1.5× 附近封顶

Part C（`dW2`，墙体时间）。单步：REF `k_grouped` fp32/acc **500.5us（240 TFLOPS）**、
ALT `E×`稠密 1D2D bf16 **220.8us（545 TFLOPS）**、**纯 GEMM 比 2.27×**。再扫累积步数 `S`（ALT 每步写一份
bf16，最后 `Σ_s` 成一份 fp32）：

| S | 归约 us | REF 总 us | ALT 总 us | **端到端比** | 归约占 ALT |
|---:|---:|---:|---:|---:|---:|
| 1 | 379.8 | 500.5 | 600.6 | **0.83×** | 63% |
| 2 | 497.7 | 1001.0 | 939.4 | **1.07×** | 53% |
| 4 | 566.3 | 2002.1 | 1449.7 | **1.38×** | 39% |
| 8 | 838.6 | 4004.2 | 2605.4 | **1.54×** | 32% |
| 16 | 1727.2 | 8008.3 | 5260.8 | **1.52×** | 33% |

**三个反直觉的读法**：

1. **`S=1` 时 bf16 方案反而更慢（0.83×）**——因为要额外**物化一份 fp32 梯度**（`E·M·N·4B = 470MB`），
   而归约必须读回每步的 bf16（235MB）。GEMM 省的 280us 填不平这一次物化。
2. **端到端在 `S≈8–16` 封顶 ~1.55×**，不是 2.2×。归约的每步成本是「读一份 bf16 输出」= `E·M·N·2B = 235MB`；
   取实测归约带宽 ~2.4–2.8 TB/s（HBM 峰值 3.35 的 73–84%，**归约本身已是带宽最优**），折算 **~84–96us/步**。
   渐进上界 = `t_ref / (t_alt + 235MB/BW) ≈ 1.58×`。
3. **归约占 ALT 时间的 32–63%**——「2.2× 的 GEMM 红利」被归约的字节量吃掉大半。

**为什么归约逃不掉？** 真实 wgrad 每步是「读回 `C`(fp32) + 写 `D`(fp32)」；bf16 方案每步只写 `bf16`，
但**要在最后把所有 S 步的 bf16 读回来物化成 fp32**。只要训练框架仍需要一份**独立的 fp32 参数梯度**，
这次归约就躲不开，端到端就被钉在 ~1.55×。**只有让优化器直接吃 bf16 梯度**（不再物化 fp32），
才能回到纯 GEMM 的 2.2×。

---

## 4. 结论：bf16 输出「精度免费、收益有条件」

把本篇与第 145 篇拼起来，wgrad 的优化账是这样：

1. **精度几乎免费**：bf16 输出 relerr 1.4e-3 ≪ FP8 输入噪声 3.6e-2，方向 cos 0.99932，
   累积 S 步也稳（相对误差近似恒定）——对「已在 FP8 精度下训练」的 MoE，这不构成障碍。
2. **收益取决于「要不要物化 fp32 梯度」**：
   - 若训练框架仍需一份 **fp32 参数梯度**（先物化、再交给优化器），就得付归约的账——端到端
     在 `S≈8–16` 封顶 **~1.55×**（`S=1` 因为要额外写一份 fp32 反而 **0.83×**），**不是 2.27×**；
   - 若**优化器直接吃 bf16 梯度**（fp32 master weight 在优化器内维护），省掉这次物化/归约，
     才能拿到第 145 篇的 **2.2×** 上界。
3. **实现落点**：SM90 上只有「`E×` 稠密 1D2D」这条路能产出 bf16（`sm90_fp8_gemm_1d2d.hpp:81`），
   真实 wgrad 的 `k_grouped` 被 API 钉死 fp32 累积（`gemm.hpp:403`）。要吃到收益，要么把
   `k_grouped` 的输出类型放开（新增 bf16-直写变体），要么改训练侧累积协议。**两者都是协议改动，
   不是 kernel 微调**——这与第 145 篇「该动的是输出精度/累积协议」的结论一致。

---

## 5. 与 SM100 的关系（仅分析）

SM100 的 wgrad 走 `k_grouped_fp8_gemm_tn_contiguous`（MN-major，`csrc/apis/gemm.hpp:320-371`），
派发到 `sm100_k_grouped_fp8_gemm_1d1d`（`csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325`）：
签名里 `C` 可选（`gemm.hpp:354` `not c.has_value() or ...`），但**生产 FP8 NT 路径同样传 `C=D` 做累积**，
`C/D` 是 3D `[G,M,N]`、fp32。因此「每专家一份 fp32 + 回读」的结构在 SM100 上**一致**；
换成 bf16 直写的收益/代价结构也一致，差别只在算力/带宽比。SM100 的 1D1D UMMA 直写/`cp.reduce.async.bulk`
等实现细节 **仅分析**（本机 H100 跑不了）。

---

## 6. 边界

- **能跑/能核**：全部 SM90 FP8 kernel，H100 真机；精度参考是**逐专家 fp64**、计时是 event 稳态，
  两者互相独立；`REF vs ALT` 的 3.7e-2 与第 145 篇 Part A 的 3.68e-2 一致（同一 FP8 噪声）。
- **不能核**：SM100 的 wgrad `tn` 路径只做源码分析；`k_grouped` 无 bf16 变体，ALT 是用 `E×` 稠密
  1D2D **模拟**「同一个 wgrad 数学」的等价写法，不是真正的 wgrad API。
- **实验口径**：ALT 的 B 用 128×128 block SF（1D2D 固定），真实 wgrad 的 A/B 均 1×128——故
  `REF vs ALT` 含两路径量化差异；bf16 **纯**输出代价看「仅末次 bf16 舍入」那一行（1.41e-3）。
- **环境**：单进程、空闲 GPU1（`CUDA_VISIBLE_DEVICES=1`，`util=0%`）；8 卡节点共享，沿用本专题报数约定。
- **不变量**：归约用 `torch.sum(dim=0, dtype=fp32)`，实测 ~2.4–2.8 TB/s（HBM 峰值 73–84%），是一个
  **带宽最优**的参照；若换成更差的归约实现，端到端收益只会更低。

---

## 7. 复现

```bash
bash harness/exp_wgrad_bf16_output.sh > harness/exp_wgrad_bf16_output.out.txt
```

脚本跑 `harness/exp_wgrad_bf16_output.py` 的 Part A/B/C。**踩坑记录**：k_grouped 的 host 侧操作数
缓冲是 **group-major** 的 `[E, mn, k_g]`（`quant_kg` 把每组写成 `xf.T.reshape(-1)`），fp64 参考必须
**逐组切片再拼到列上**；当成 `[mn, total_k]` 会让参考 O(1) 崩（本脚本 `deq_kg` 已按前者实现）。

### 源码坐标（`~/github/DeepGEMM@057ca59`，行号已核对）

- `csrc/apis/gemm.hpp:373-427`：SM90 wgrad `k_grouped_fp8_gemm_nt_contiguous`（`:384` recipe、
  `:403` `c` 必填）。
- `csrc/apis/gemm.hpp:320-371`：SM100 wgrad `tn`（`:331` 1D1D、`:354` `c` 可选；仅分析）。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d1d.hpp:84/94`：`c.has_value() and d==float`、`with_accumulation`。
- `csrc/jit_kernels/impls/sm90_fp8_gemm_1d2d.hpp:81`：`not c.has_value() and d==bf16`。
- `deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d1d.cuh:346-349`：`SM90_TMA_REDUCE_ADD_2D`。
- `deep_gemm/utils/math.py:26/52/62`：`per_token_cast_to_fp8` / `per_channel_cast_to_fp8` /
  `per_block_cast_to_fp8`（两条路径的量化入口）。
- `csrc/jit_kernels/impls/sm100_fp8_fp4_gemm_1d1d.hpp:325`：`sm100_k_grouped_fp8_gemm_1d1d`（仅分析）。

### 实验

- `harness/exp_wgrad_bf16_output.py`（Part A 精度分解、B S 步累积、C 落地计时）。
- `harness/exp_wgrad_bf16_output.sh`（driver）。
- `harness/exp_wgrad_bf16_output.out.txt`（原始输出）。

### 线上资料

- DeepGEMM 仓库：https://github.com/deepseek-ai/DeepGEMM
- Mega MoE PR #316：https://github.com/deepseek-ai/DeepGEMM/pull/316
- 本专题：`docs/145`（wgrad 强制累积代价）、`docs/12`（wgrad API）、`docs/15`（反向 ncu 归因）、
  `docs/142`（前向 roofline 对偶）。
