# MoE 反向的「非 GEMM 段」：SwiGLU 反向 + L1 激活梯度的 FP8 / UE8M0 重量化（H100 实测）

> 本专题第 154 篇，路线图 **P144-1**。前向 single-step 的预算被第 142 篇（GEMM）与第 143 篇
> （L2 写回 + combine 归约）补全；反向的四个 grouped GEMM（d2g/d1g/w2g/w1g）的性能与累积协议被
> 第 15/135~151 篇挖到底；反向的两段通信由第 104 篇补上。但**夹在两个 dgrad GEMM 之间的那段元素级
> 算子**——把 L2 dgrad 出来的 `d(act)` 反解成 gate/up 的激活梯度、再量化成 FP8（供 wgrad 的 B 操作数）
> ——从没被单独量过。它就是前向 L1 epilogue（第 10/142 篇已拆）的**逐元素逆**。
>
> **可跑性**：Mega MoE 融合 kernel 本体（UMMA/tcgen05/UTMEM/SF）是 SM100 专属、仅分析；本段的数学
> 与量化语义架构无关，H100(sm90) 用**真实内核**跑通并留真机数字。
> 实验：`src/moe_backward_activation.cu` + `harness/exp_moe_backward_activation.{sh,py,out.txt}`。
> 全部行号对着 `~/github/DeepGEMM` 真实核对过。

---

## 1. 这段在反向图里的位置

MoE 前向（训练用 expand 版）：

```
dispatch → L1(gate/up) → SwiGLU×topk权重 → FP8 量化 → L2 → combine
```

前向 L1 的 epilogue 把「gate/up 两个投影输出」当场算成 SwiGLU 激活、乘 top-k 路由权重、按 UE8M0
量化成 FP8（`sm100_fp8_fp4_mega_moe.cuh:1108-1219`，第 10 篇 §5、第 142 篇 §3 已拆）。反向拆成
四段 grouped GEMM（第 4/14 篇）：

```
d(y) → combine_bwd(=dispatch) → L2 dgrad(d2g)
     → [SwiGLU 反向 + 量化]  ← 本篇
     → L1 dgrad(d1g) ； 同时 L1 wgrad(w1g) / L2 wgrad(w2g)
     → dispatch_bwd(=combine)
```

`d2g` 输出的是**激活的梯度** `d(act)`（bf16）；但要跑 `d1g`/`wgrad`，需要的是 **gate/up 预激活的
梯度** `d(gate)`、`d(up)`，且必须**量化成 FP8**——因为 `k_grouped` wgrad 的操作数是
`(data, sf)` 的 FP8 对（`csrc/apis/gemm.hpp:373-411`：`a,b` 都是 `std::pair<Tensor,Tensor>`、
recipe `(1,1,128)`）、`m_grouped` dgrad 同样是 FP8 输入。这一「SwiGLU 反向 + 重量化」就是本篇的段。

---

## 2. 数学：前向 L1 epilogue 的逐元素逆

前向（`sm100_fp8_fp4_mega_moe.cuh`）逐元素的数据路径，语义钉死如下：

```cuda
// :1112-1113  先把 gate/up 从 fp32 舍入进 bf16（RNE）
auto bf16_gate = __float22bfloat162_rn(fp32_values[k * 2 + 0]);
auto bf16_up   = __float22bfloat162_rn(fp32_values[k * 2 + 1]);
// :1116-1120  clamp（可选）
// :1122-1135  SwiGLU：silu(gate) = gate / (1 + exp(-gate))，再乘 up 与路由权重 w
auto gate = __bfloat1622float2(bf16_gate);
const auto denom = __fadd2_rn({1.0f, 1.0f}, neg_gate_exp);
gate = __fmul2_rn(gate, {math::fast_rcp(denom.x), math::fast_rcp(denom.y)});  // gate <- silu(gate)
const auto up = __bfloat1622float2(bf16_up);
activation_values[i][k] = __fmul2_rn(__fmul2_rn(gate, up), weights);          // silu(gate)*up*w
```

设 `σ(x)=1/(1+e^{-x})`、`silu(x)=x·σ(x)`。前向输出 `act = silu(g̃)·ũ·w`，其中 `g̃=bf16(gate)`、
`ũ=bf16(up)`。反向（对 `act` 的梯度 `dact` 已知）：

```
d_su = dact · w
d(gate) = d_su · ũ · silu'(g̃) ,   silu'(x) = σ(x)·(1 + x·(1-σ(x)))
d(up)   = d_su · silu(g̃)      = d_su · g̃ · σ(g̃)
```

两个要点：

1. **bf16 舍入是前向刻意做的**，反向对这一步走 **straight-through**（把 `g̃`/`ũ` 当输入、不反传它俩的
   舍入）。第 111 篇（反向数值复刻）已说明这是设计而非 bug；本篇量化它的误差（§4）。
2. **前向的 `fast_rcp`**（`math.cuh` 的 `rcp.approx.ftz`）是 fast-math 路径；反向若也走 fast-math
   就用 `__expf` + `rcp.approx`，本篇内核即如此（与第 142 篇 L1 epilogue 逐行移植的假设一致）。

量化：对 `d(gate)`、`d(up)` 各自按 **32 个连续中间维元素**取 `amax`、算 UE8M0 幂次 scale、cast E4M3——
这套语义与**前向 L1 输出对 `act` 的量化完全一样**（`mega_moe.hpp:200-204` 的 `kGranK = 32`）：
SF 取整与倒数幂次逐位复刻 `common/math.cuh:114-147`（`get_ue8m0_sf_exp` 的进位即「指数上取整」、
`get_ue8m0_sf_inv` 的 `(254 - sf) << 23`）。wgrad 的 B 操作数 SF 配方是 `(1,1,128)`
（`gemm.hpp:384`）——分组粒度不同，但**字节/时间同量级**，本篇按 per-32 量（与前向对齐）。

---

## 3. H100 真实内核

`src/moe_backward_activation.cu` 直接实现 §2：一个 warp 处理 2 个 quant group（每组 32 元素 =
16 lane × 2 元素，`__nv_bfloat162` 向量化），组内 `__shfl_xor_sync` 半 warp 归约 `amax`，写 FP8 与
SF。含两条模板路径：`kQuant=true`（FP8 + UE8M0 SF）与 `kQuant=false`（bf16 直写，隔离
SwiGLU 反向本身的开销）。全部架构无关，H100(sm90) 可跑。

**（A）正确性**：与 host fp64 参考（同 bf16 数据路径、同位运算的 UE8M0 选择）逐组比对，`T=512,I=1024`
下 **SF 逐位不符 0 / 32768、FP8 字节不符 0 / 1048576**。坐实「量化/取整」这条链与 `math.cuh` 完全一致。

**（B）straight-through 误差**：前向 bf16 舍入数据路径 vs 精确 fp64，`max|Δ| = 5.679e-02`（相对
`2.812e-03`）——这是前向刻意的 bf16 round 必然带进反向的、1e-3 级 relative 误差（见 §4 的分层）。

**（C）带宽**（空闲 GPU，热 cache，200 次迭代均值，字节 = 读 3×bf16 + 写 2×fp8 = 8B/元素）：

| T, I | quant | quant GB/s | bf16 | bf16 GB/s |
|---|---|---|---|---|
| 2048, 2048 | 20.12 us | 1668 | 21.14 us | 1984 |
| 8192, 2048 | 68.41 us | 1962 | 67.08 us | 2501 |
| 2048, 7168 | 60.44 us | 1943 | 60.71 us | 2418 |
| 4096, 4096 | 68.38 us | 1963 | 67.13 us | 2499 |
| 512, 512 | 11.58 us | 181 | 11.44 us | 229 |

差距来自量化多出的 E4M3 转换 + 半 warp 归约；`512×512` 被 launch 不占主导。bf16 路径达 ~2.5 TB/s
（HBM 3.35 TB/s 的 ~75%），量化路径 ~1.97 TB/s（~59%）——**是一个真实、接近带宽上限的单遍内核**。

---

## 4. 误差分解：两件事都压在噪声地板上

`harness/exp_moe_backward_activation.py` Part A 用 torch fp64 作独立 oracle，按 §2 的数据路径分层：

| T, I | bf16-round（数据路径） | +FP8 UE8M0 量化（喂 wgrad） |
|---|---|---|
| 2048, 2048 | dg 1.736e-3 / du 1.947e-3 | dg 2.658e-2 / du 2.661e-2，cos 0.999647 |
| 8192, 2048 | 1.736e-3 / 1.943e-3 | 2.655e-2 / 2.654e-2，cos 0.999647 |
| 2048, 7168 | 1.735e-3 / 1.942e-3 | 2.657e-2 / 2.657e-2，cos 0.999647 |

两点结论（与第 146/151 篇的「精度账」同一把尺子）：

1. **bf16 舍入数据路径的误差 ~1.7e-3**，比算子本身的 FP8 **输入**噪声（第 111 篇实测 ~3.6e-2）
   小一个数量级——straight-through 前向 bf16 round 是「精度可接受」的设计。
2. **给 wgrad 的 FP8 重量化再叠 ~2.7e-2**——**恰好落在 FP8 输入噪声地板（~3.6e-2）之内**，cos 仍
   0.99965。即：把 `d(gate)/d(up)` 量化成 FP8 **数值上基本免费**（和第 146 篇「bf16 输出比 FP8
   输入噪声小 25×」是同一现象：输出/中间量的量化只要不显著超过输入噪声，就不额外伤梯度）。

---

## 5. 放回反向 single-step 预算

把本篇新量的激活段，与前面各篇量过的四个 grouped GEMM 拼成一张表（DeepSeek 量级
`T=2048, H=7168, I=2048, E=8, topk=2`，单次 launch）：

| 段 | H100 实测 | 出处 |
|---|---|---|
| d2g（L2 dgrad） | 111.0 us | 第 15/132 篇 |
| **SwiGLU 反向 + FP8 重量化** | **20.0 us**（T=2048,I=2048，量化路径） | 本篇 |
| d1g（L1 dgrad） | 194.3 us | 第 15/132 篇 |
| w2g（L2 wgrad） | 500.6 us | 第 135 篇 |
| w1g（L1 wgrad） | 997.0 us | 第 135 篇 |
| 合计（GEMM + 激活段） | 1822.9 us | — |
| 反向两相位 a2a（T=4096,k=8,EP=8） | ~8000 us（受争用下界） | 第 104 篇 |

**结论**：

- 激活反向/重量化段 **20 us，占「四 GEMM + 激活段」的 1.10%**，相对整步（含通信）< 0.2%。
- 它和前向 L1 epilogue（第 142 篇 24.2 us）**同量级**——前向/反向各有一小段「SwiGLU × 权重 ×
  量化」的元素级工作，都不是瓶颈。
- 反向 step 的钱仍在**四个 grouped GEMM**：wgrad（w1g+w2g=1497.6 us）占 83%（第 132 篇）；通信是
  第二大块（第 104 篇）。**L2 dgrad 的输出要额外过一遍「SwiGLU 反向 + 量化」这个事实，代价是
  ~1% 级别**——它足以说明为什么真实实现要把这段也做进 GEMM 的 epilogue/邻近 kernel 里（而不是
  单独发一个 torch 元素级 op：Part C 实测 torch 朴素实现 199.8 us，是 CUDA 融合核的 **10×**，
  因为它物化 fp32 中间张量）。

> 与前向的对偶：第 142/143 篇把前向 step 拆成「计算 339.4 us + 通信/归约 292 us」，并推出必须
> overlap；本篇给反向补上「四 GEMM 1803 us + 激活段 20 us + 通信」，同样指向「wgrad 才是反向的
> 大头」。

---

## 6. 与其它篇的关系

- 前向 L1 epilogue 的**正向**（SwiGLU + topk 权重 + UE8M0）：第 10 篇 §5、第 142 篇 §3、第 99 篇。
- 前向 step 预算（下半）：第 143 篇（L2 写回 + combine）；前向 overlap：第 144 篇。
- 反向四 GEMM 性能归因：第 15/132 篇；wgrad 累积协议系列：第 135~151 篇。
- 反向通信模型与实测：第 104 篇；反向数值复刻：第 111 篇。
- UE8M0 SF 的位运算语义：第 25/54 篇；`common/math.cuh` 的 SF 选择：第 54 篇。

**边界**：Mega MoE 融合 kernel 本体（UMMA/tcgen05、TMEM、UTCCP、SF 布局）SM100 专属、仅分析；
本篇内核架构无关、H100 真机可跑。测得的是**语义与成本的下界**，不复刻 SM 内 overlap。
