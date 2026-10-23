# FP8 反向设计（P3-1）

> 目标：在 H100（sm90）上实现 FlashAttention **反向的 FP8 版本**，口径尽量对齐
> **TransformerEngine 2.14 fused_attn**（FA 仓库反向没有 FP8）。
> 本文只讲**设计**：量化对象、E4M3/E5M2 分工、rowwise scaling 布局、缩放如何折回、
> 误差来源与容差口径。实现与实测见 `03-fp8-bwd-impl.md` 与 `../ROADMAP.md`。
> 参考：`docs/00-fa-bwd-optimization-catalog.md` §4、`code/te-perf/bench_te.py:297-389`。

## 先记住一个画面

FP8 量化像**把大小不一的货物装进统一规格的箱子**。

- 每个箱子只有固定的格子（尾数），装不下就只能四舍五入；
- 但每行货物可以配一个**自己的换算比例**（rowwise scale）：这一行整体很大，就把比例调大；很小，就调小；
- 到用的时候，再按同一个比例把"箱子里的编号"还原成实际重量。

所以 FP8 不是"把 fp16 粗暴截断"，而是**一行一策的块浮点**：精度损失主要来自"格子不够细"，而不是"范围不够大"。

> **类比的边界**：装箱只损失一次精度；FP8 反向里，`P`、`dP`、`dS` 会**反复被量化、反量化、再量化**。误差会沿着链式法则累积，所以"每一步都只差一点"不等于"最后也只差一点"。这就是为什么 §1 要把误差源单独列出来。

读完这一篇，你应该能回答：

1. 反向有五个矩阵乘，哪些操作数必须量化？为什么累加器必须留在 fp32？
2. E4M3 与 E5M2 的分工依据是什么？为什么 `P` 可以直接用 E4M3 且 scale 固定为 1？
3. rowwise scale 在 MMA 里怎么折回？折错会表现为随机噪声还是系统性偏差？

### 结论速查

| 问题 | 结论 | 详见 |
|---|---|---|
| 量化谁 | Q/K/V/dO/P/dP；`dS`、LSE、`D`、累加器留 fp32 | §1、§2 |
| 格式分工 | Q/K/V/P 用 E4M3；dO/dP 用 E5M2 | §2 |
| 缩放粒度 | rowwise（每行一个 fp32 scale） | §3 |
| 折回位置 | epilogue 乘 `scale_a*scale_b`，或折进下一步量化 | §3.2 |
| 误差量级 | 与 TE 同量级，O(0.24–0.94)（fp16 约 2e-3） | §4 |
| 第一版取舍 | `dS`、`dq/dk/dv` 保留 fp32，先隔离输入量化 | §5 |

---

## 1. 为什么反向能用 FP8，以及误差从哪来

反向有**五个**大矩阵乘（不是四个——`S` 的重算也要算一次）：

| 计算 | A 操作数 | B 操作数 | 说明 |
|---|---|---|---|
| `S = scale·QKᵀ`（recompute） | Q | K | 反向要重算 P，故 QK 也要算 |
| `dP = dO·Vᵀ` | dO | V | 梯度对 P |
| `dV = Pᵀ·dO` | P | dO | 对 V 的梯度 |
| `dQ = scale·dS·K` | dS | K | 对 Q 的梯度 |
| `dK = scale·dSᵀ·Q` | dS | Q | 对 K 的梯度 |

FP8 张量核（`mma.m16n8k32`，sm90 兼容路径；进阶用 `wgmma`）要求**操作数已是 FP8**，累加器保持 **fp32**。所以「反向 FP8」= 在进入这些 MM 前把操作数量化到 FP8，其余统计量（`LSE`、`D`、`dS`、softmax 的 exp）保持 fp32。

**为什么累加器必须 fp32？** 因为一个输出元素是 $K$ 维点积（$K$ 可能是 128 或 512），FP8 乘出来的项要在累加器里相加几百次。若累加也用 FP8/FP16，误差会随项数线性增长；fp32 累加把这一步的误差压到可忽略，**精度损失被限制在"输入量化"这一个环节**。

误差来源按重要性排：

1. **数据量化误差**：`Q/K/V/dO` 从高精度量化到 FP8，动态范围/尾数双重损失。E4M3 只有 3 位尾数，E5M2 只有 2 位尾数但指数范围大。
2. **中间量量化误差**：`P`、`dS` 若也进张量核，需要再量化一次；`dS = P∘(dP−D)` 在 `D` 附近会**灾难性抵消**（小 `P` 处噪声相对大），是 FP8 反向最敏感的一步。
3. **缩放折算**：`scale_a·scale_b` 要乘回 fp32 累加器；rowwise 布局下每行/每列 scale 不同，折错会引入**系统性偏差**（不是随机噪声）。

### 1.1 两种 FP8 格式：一个保精度，一个保范围

| 格式 | 符号 | 指数 | 尾数 | 最大值 | 特点 |
|---|---|---|---|---|---|
| E4M3 | 1 | 4 | 3 | 448 | 尾数多、精度高，但范围小 |
| E5M2 | 1 | 5 | 2 | 57344 | 范围大，但精度更低 |

一句话选型：**值域可控、靠近 $[0,1]$ 的用 E4M3；动态范围大的梯度用 E5M2。**

---

## 2. E4M3 / E5M2 分工（对齐 TE）

TE 的 FP8 fused attention 约定（`te-perf/bench_te.py:308-389`、TE 文档）：

- **前向**：`Q/K/V/S/O` 用 **E4M3**（尾数多、范围够）。
- **反向**：`dO`（及 `dP`、`dQ/dK/dV`）用 **E5M2**（梯度动态范围大，2 位尾数可接受）。
- 输入在计时区外**预先量化**（`Float8Quantizer(rowwise=True)`），kernel 内部只做 rowwise 动态缩放。所以 TE 报的是**纯 FP8 kernel 时间**。

H100 峰值常量（换算用）：FP8 Tensor Core dense **≈1978.8 TFLOPS**，FP16/BF16 **≈989.4**，FP32 CUDA core **≈66.9**，HBM **≈3.35 TB/s**（`code/te-perf/bench_te.py:45-51`）。

### 我们的分工

| 张量 | dtype | 量化粒度 | 归一化上界 |
|---|---|---|---|
| Q | E4M3 | rowwise（每 `(b,s,h)` 行 over D） | 448 |
| K | E4M3 | rowwise | 448 |
| V | E4M3 | rowwise | 448 |
| dO | **E5M2** | rowwise | 57344 |
| P（中间量） | E4M3 | 固定 scale = 1（`P∈[0,1]`） | 448 |
| dP（中间量） | **E5M2** | rowwise | 57344 |
| dS（中间量） | fp32（本版）/ E5M2（进阶） | — | — |
| LSE / D / 累加器 | fp32 | — | — |

> 与 TE 的**差异**（本版 golden 的取舍，见 §5）：TE 还会把 `dS`/`dQKV` 走 E5M2 并量化输出；我们第一版只量化输入与 `P`、`dP`，`dS` 与最终 `dq/dk/dv` 保留 fp32，便于先隔离「输入量化」这一主误差项。

**为什么 `P` 的 scale 固定为 1？** 因为 $P = \mathrm{softmax} \in [0,1]$，上界已知且稳定；E4M3 在 1 附近的绝对分辨率约 $2^{-3}$，更小的值靠次正规数表示、绝对误差反而更小，够用。反过来说，如果给 `P` 也做 rowwise，每行都要多算一次 amax、多存一个 scale，收益却很小——**这是"用结构知识省掉一次归约"的典型例子**。

---

## 3. Rowwise scaling 布局与折算

### 3.1 量化公式（per-row）

对一行 `x[d] (d=0..D-1)`：

```
amax  = max_d |x[d]|
scale = amax / FP8_MAX              # FP8_MAX = 448 (E4M3) 或 57344 (E5M2)
xq[d] = fp8_round( x[d] / scale )   # satfinite 饱和，不是溢出成 inf
x'[d] = float(xq[d]) * scale        # 反量化，送进 MM
```

逐句解释：

- `amax` 取绝对值最大，保证整行都落在 FP8 的表示范围内；
- `scale` 是 **fp32 任意值**（不是 2 的幂）。E4M3/E5M2 都是「指数+尾数」标准格式，不像 `ue8m0` 那样只能表示 2 的幂。**注意**：只有 `ue8m0` 的 2 的幂 scale 才能精确折进 FP8 操作数（见 `agent_skills/kernel-opt.md` 37/56 篇），这里是 fp32 scale，必须显式乘回。
- `satfinite` 表示超范围时饱和到最大有限值，而不是变成 inf——**避免一个异常值污染整行**。

### 3.2 折算（scale 乘回）

FP8 MMA 输出的是 $\sum (xq_a \cdot xq_b)$ 的 fp32 累加。真实值是 $\sum (sa\cdot xq_a)\cdot(sb\cdot xq_b)$，所以要 `acc *= sa*sb`。

rowwise 布局下：

- **A 的 scale 按 A 的行**（`sa[m]`），**B 的 scale 按 B 的行**。在 `m16n8` 的累加器里，`acc[j*4+0/1]` 同一行、`acc[j*4+2/3]` 同一行，同一行 4 个累加器共用同一个 `sa`；`n8` 的两列 `c0/c1` 有各自的 `sb`（`agent_skills` 139 行）。
- 实践中（TE 的做法）把 scale **折进输出写回**：`dQ` 的 scale 折到 `dS` 的量化里，`dK/dV` 的 scale 折到 `dS`/`P` 里，从而 MM 内部只做纯 FP8 乘加。
- **陷阱**：`Q`（E4M3）与 `K`（E4M3）的 scale 不同，`S` 的 scale 是二者乘积；反向的 `dS` 又带着 `P`/`dO`/`V` 的 scale，级联下去必须逐项记账。

打个比方：每行货物有自己的"汇率"，A 行和 B 行乘出来的结果要按**两个汇率的乘积**换回真钱。**漏乘一项，整批结果都会系统性偏大/偏小**——这就是"折错是偏差不是噪声"的含义。

### 3.3 本版（golden，标量）怎么处理

golden 版**不做 MMA**，而是「量化→反量化→fp32 标量乘加」，因此：

- 每个操作数读进来时 `x' = float(xq)*scale`，等价于「FP8 权重 × fp32 scale」；
- 累加天然 fp32，**无需** `sa*sb` 折算（因为在反量化时已经乘回）。
- 这样得到的数值 == 「真实 FP8 张量核 + 正确折算」的结果（模去 fp32 舍入差异），所以 golden 可以用来给后续 MMA 版做数值基准。

**为什么要先写标量 golden？** 因为 FP8 有两个独立的误差源：量化本身、以及 MMA 的布局/折算。如果一上来就写 MMA，数值对不上时你分不清是"量化口径错了"还是"scale 折错位置了"。标量版把两者解耦——**先把量化口径钉死，再上指令**。

---

## 4. 与 TE 的接口/口径对齐

`te_bwd_fp8`（在 `harness/fa_bwd_bench.py` 中实现，参照 `bench_te.py:308-389`）：

```python
qkv_q  = Float8Quantizer(..., fp8_dtype=e4m3, rowwise=True)   # Q/K/V
s_q    = Float8Quantizer(..., fp8_dtype=e4m3, rowwise=True)   # S/P
o_q    = Float8Quantizer(..., fp8_dtype=e4m3, rowwise=True)   # O
do_q   = Float8Quantizer(..., fp8_dtype=e5m2, rowwise=True)   # dO
dp_q   = Float8Quantizer(..., fp8_dtype=e5m2, rowwise=True)   # dP
dqkv_q = Float8Quantizer(..., fp8_dtype=e5m2, rowwise=True)   # dQ/dK/dV
backend = FusedAttnBackend["FP8"]
out, aux = fused_attn_fwd(True, S, S, cu, cu, q8, k8, v8, nominal, backend, None,
                          s_quantizer=s_q, o_quantizer=o_q, ...)
dqkv = fused_attn_bwd(S, S, cu, cu, q8, k8, v8, out, do8, nominal, do8._fp8_dtype,
                      list(aux), backend, s_quantizer=s_q, dp_quantizer=dp_q,
                      dqkv_quantizer=dqkv_q, ...)
```

**已验证**（`probe`，容器内）：TE FP8 反向在 `(1,512,16,128)` 与 `(1,1024,32,128)` 可用，返回 `dq/dk/dv`（bf16）。与 fp32 autograd ref 的 `max_abs`：

| shape | dq | dk | dv |
|---|---|---|---|
| B1 S512 H16 D128 causal | 3.53e-1 | 4.23e-1 | 5.89e-1 |
| B1 S1024 H32 D128 causal | 4.04e-1 | 4.56e-1 | 5.74e-1 |

这就是 **FP8 反向的容差量级（~5e-1，O(1) 的绝对误差）**——比 fp16（~2e-3）松两个数量级。注意 `dq/dk/dv` 的幅值本身是 O(1~10)，所以相对误差约 **几 % ~ 十几 %**。

head_dim 限制：TE 训练口径反向 `qk==v` 最大 256（`results_summary.md`）；本设计固定 D=128。

> **不要用 `max_rel` 做判据**：梯度里有大量接近 0 的元素，相对误差会被放大成天文数字。**看 `max_abs` 与幅值的比值**，或者看分布，不要看单个近零元素。

---

## 5. 实现路线（分阶段）

1. **P3-2（本阶段，golden）**：单文件、标量实现。
   - `quantize_kernel`：输入 fp32 `q/k/v/do`，算 rowwise amax+scale，写出 FP8 `q8/k8/v8`（E4M3）与 `do8`（E5M2）及各自 scale。
   - `preprocess_kernel`：用反量化的 Q/K 重算 `LSE`；用反量化 dO 与 fp32 O 算 `D=rowsum(dO∘O)`。
   - `main_kernel`：反量化载入 Q/K/V/dO，标量累加；`P` 量化到 E4M3（scale=1），`dP` 量化到 E5M2（rowwise）；`dS` 保持 fp32；`dq/dk/dv` fp32。
   - `convert_kernel`：累加缓冲 → 输出。
2. **P3-4（进阶）**：把 `S/dP/dV/dQ/dK` 换成 `mma.m16n8k32`（E5M2×E4M3），rowwise scale 在 epilogue 折回；用 `ldmatrix` + smem padding 消 bank conflict；目标是打满 FP8 峰值的一部分（FR: H100 FP8 1978.8 TFLOPS）。
3. **P3-5**：两文件拆分 + 文档。

**为何先标量 golden**：与 P1/P2 一致（catalog §5：先写功能正确的 CUDA-core 版本作为 golden，再逐层上张量核）。FP8 的误差分析必须先把「量化数值」和「MMA 调度」解耦，否则一旦对不上，无法判断是缩放布局错还是硬件 MMA 布局错。

---

## 6. 待办 / 风险

- `P`/`dP` 量化会让 `dS` 误差进一步放大；需实测确认是否仍在 TE 同量级。
- causal 下 `dS` 的 mask 位置（`jg>qi`）应保持 0，量化不得把 0 变成非零。
- 输出 `dq/dk/dv` 本版是 fp32，TE 返回 bf16；对比时注意这一口径差。
- `dS` 走 E5M2 + `mma` 时，`dS=P∘(dP−D)` 的抵消误差是主要风险点（agent_skills 44 篇「灾难性抵消」类问题）。

### 三个常见误解

1. **"FP8 就是把 fp16 砍掉几位"**：不是。FP8 必须配 per-row scale，否则动态范围根本不够；砍位只是最后一步。
2. **"误差是均匀的"**：不是。`P` 小的位置、`dS` 抵消的位置，相对误差会显著变大；FP8 反向的容差是"量级正确"，不是"每位都对"。
3. **"FP8 一定更快"**：不一定。量化/反量化、scale 折算、输出转换都是额外开销；只有当 main kernel 的算力收益盖过这些开销时，端到端才会变快。

### 自检三问

1. 如果 `dO` 也改用 E4M3（而不是 E5M2），会发生什么？哪些形状最先出问题？
2. 为什么 golden 版反量化后不需要再乘 `sa*sb`，而 MMA 版必须乘？两者的等价性依赖什么假设？
3. `P` 的 scale 固定为 1 省掉了什么？如果某个 head 的 `P` 最大值只有 0.01，这个固定 scale 会浪费多少有效位？
