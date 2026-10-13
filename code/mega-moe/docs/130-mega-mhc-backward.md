# Mega mHC 变换的反向：残差多流混合、Sinkhorn 双随机与 RMSNorm 的梯度

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P120-1** 增量（第 130 篇）。
>
> 第 5 篇（P5-2）把 **Mega mHC**（HyperConnection）的**前向**拆透了：6 个 warpgroup 的静态分工、
> TF32 UMMA（A=残差在 TMEM、B=`fn` 在 SMEM）、Sinkhorn 的 warp 内 `shfl_xor` 实现、RMSNorm 融合。
> 第 4 篇（P4-1）处理的是 **MoE router** 的反向、第 127 篇（P117-1）补的是 **Mega Gate** 打分的反向。
> 但 **mHC 这条「残差改写 + 系数生成」通路的反向一直没有**——DeepGEMM 的 Mega mHC **只有前向**
> （主 kernel SM100 专属），而它是 HyperConnection 真正改变残差流的地方：**如果网络要在 mHC 上训练，
> 梯度必须穿过 Sinkhorn 的交替归一化与那个「把 24 个投影输出耦合到同一个平方和」的归一化尺度**。
> 本篇把这条反向按它的**权威参考**（`tests/test_mega_mhc.py` 的 `mhc_reference`）逐行推导出来，
> 并在 H100 上用两条独立路径核验：**手写 VJP vs torch autograd**（float64）与 **device 设备代码 vs
> double 参考**（fp32）。
>
> **可跑性**：mHC 主 kernel 是 SM100 专属、只有前向，端到端跑不了；但它的**计算图数学**
> （残差多流混合、投影、sigmoid/仿射、Sinkhorn 双随机、RMSNorm）与 UMMA/TMA/TMEM **完全无关**，
> 可以逐行搬成可微 torch 与裸 CUDA 在 H100(sm90) 上真跑。**H100 真机**：手写反向 vs autograd
> 最坏 `relerr = 1.01e-15`（6 组形状 × 9~10 个张量梯度）、device(fp32) vs double `~8e-8…3.6e-7`、
> Sinkhorn VJP vs 中心差分 `~1e-11`。源码行号均按 DeepGEMM `057ca59` 核对。

---

## 1. 前向：mHC 到底怎么改写残差（权威参考）

`tests/test_mega_mhc.py:123-210` 的 `mhc_reference` 是语义的 source of truth。逐段（每 token）：

**① 残差多流混合**（`:145-148`）：

```
new_residual[t,j,:] = post_mix[t,j] * x[t,:] + Σ_i comb_res_mix[t,j,i] * residual[t,i,:]
new_residual = bf16(new_residual)          # 舍入到 bf16（共享数值边界）
```

第 `j` 条新残差流 = 「本层新激活 `x` 按 `post_mix_j` 缩放」+「`hc` 条旧残差流按 `comb_res_mix[j,:]`
加权组合」。kernel 侧对应 `impls/sm100_mega_mhc.cuh:486-497`（`__ffma2_rn` 组合旧流、`:497`
`__float22bfloat162_rn` 就是那个共享**数值边界**，注释在 `:496`）。

**② 投影 + 归一化尺度 + 仿射**（`:155-170`）：

```
mixes  = (new_residual.flatten(1) @ fn.mT) * rsqrt(Σ new_residual²/(hc*H) + hc_norm_eps) * scales + bases
pre_j  = sigmoid(mixes[:, :hc])              + hc_pre_eps
post_j = sigmoid(mixes[:, hc:2hc])           * hc_post_scale
logit  = mixes[:, 2hc:].view(hc, hc)
```

`fn` 是 `[hc*(hc+2), hc*H]` 的投影矩阵（本 kernel 唯一的 GEMM，**窄输出、长 K**）。投影先乘
`rsqrt(hc_sqr_sum/(hc*H)+eps)` 把残差尺度归一化掉，再逐段乘 `scales`（长度 3 给 pre/post/comb）、
加 `bases`。kernel 侧：平方和 `impls:501`、尺度 `epilogue/sm100_mega_mhc.cuh:43`、
仿射+sigmoid `epilogue:53-56`。

**③ Sinkhorn 双随机**（`_sinkhorn_reference`，`:114-120`）：

```
v = softmax(logits, dim=-1) + eps
v = v / (v.sum(dim=-2, keepdim=True) + eps)          # 列归一化
repeat (iters-1) 次:
    v = v / (v.sum(-1) + eps)                         # 行归一化
    v = v / (v.sum(-2) + eps)                         # 列归一化
```

把组合矩阵推向**双随机矩阵**（行和、列和都 → 1）：用双随机的 `comb` 做残差混合 = **不带缩放的能量
再分配**，每层混合不放大也不缩小残差总能量。kernel 侧 `epilogue:63-86`（warp 内 `shfl_xor` 实现）。

**④ norm_input 与 RMSNorm**（`:176-184`）：

```
norm_input = Σ_j mix_j * new_residual[:, j, :]        # mix = pre（normal）或 shifted_prev_mix
norm_input = bf16(norm_input)                          # 第二处 bf16 共享边界
y          = bf16( norm_input * rsqrt(mean(norm_input²)+rmsnorm_eps) * (γ * scale) )
```

注意 RMSNorm 的均方是 **`mean` over `H`**，与 ②里对所有 `hc*H` 求和的 `hc_sqr_sum` 是**两个不同的
统计量**——反向里必须分开处理。

### 1.1 normal vs shifted

`kIsShifted` 模板开关只改两处：*用哪套 pre 系数* 与 *是否输出 pre*。
- **normal**：`norm_input` 用**刚算出的** `pre`（`new_prev_mix`），`pre` 不进输出；
  于是 `pre` 只有「经 `norm_input`」一条反向路径。
- **shifted**：输入 `shifted_prev_mix`，`norm_input` 用它；并**输出** `new_prev_mix = pre`。
  于是 `pre` 多了一条「作为输出」的反向路径。

API 契约是「全有或全无」（`csrc/apis/mega_mhc.hpp:78`、`:155-157`）。

---

## 2. 反向：逐段推导

记 `hc = 4`（契约 `mega_mhc.hpp:138`，`layout/mega_mhc.cuh:11`）、`D = hc·H`、
`Nout = hc·(hc+2) = 24`。上游梯度记 `dS`（对 `new_residual`）、`dPost`（对 `new_post_mix`）、
`dM`（对 `new_comb_res_mix`）、`dY`（对 `y`），shifted 另有 `dShift`（对 `new_prev_mix`）。

**⑧ RMSNorm**（`y = n · rg · γ_s`，`γ_s = γ·scale`，`rg = rsqrt(mean(n²)+eps)`）：

```
dn   = dY · rg · γ_s                       # 对 n 的直传
dγ  += Σ_t dY · n · rg · scale             # 权重（逐 H）
dL/drg = Σ_h dY_h · n_h · γ_s
dm   = dL/drg · (−½ rg³)                    # rg 对 m 的导数
dn  += dm · (2/H) · n                       # m = mean(n²) 再回流
```

**⑦ norm_input 的混合**（`n = Σ_j mix_j · s_j`，`s = new_residual`）：

```
dmix_j = Σ_h dn_h · s_{j,h}                 # shift 变体的输入梯度 / normal 下 pre 的上游
ds    += mix_j · dn                         # 对 s 的贡献
ds    += dS                                 # 上游对 new_residual 的贡献
```

**⑥ 系数 → `du`**（`u` 是 24 个仿射后的值）：`pre_j = σ(u_j)+e_pre`、`post_j = σ(u_{hc+j})·c_post`、
`logit_{j,i} = u_{2hc+j·hc+i}`。于是

```
du_j        = dpre_in_j · σ'(u_j)                 # dpre_in = dmix（normal）/ dShift（shifted）
du_{hc+j}  += dPost_j · c_post · σ'(u_{hc+j})
du_{2hc+·} += sinkhorn_backward(dM)
```

**⑤ Sinkhorn 反向**是本文最非平凡的一段。它把 logits 经过一次 softmax 与一串
`v ← v / (reduce(v)+eps)`，反向就是**逐条逆放**：

- **除以和的反向**：前向 `Y = X/(S+eps)`，`S = reduce(X, dim)`。则
  `dX = dY/(S+eps) − broadcast(Σ_dim (dY·X))/(S+eps)²`。
- **softmax 反向**：`dp = p · (dP − (dP·p).sum(-1))`。

把归一化的每一步的输入矩阵与归约维缓存下来，逆序应用即可。H100 上 4×4（`hc=4`）的解析 VJP 与
中心差分一致到 `~1e-11`、与 autograd 一致到 `2e-16`（见 §4）。

**④ 仿射**（`u = u0·g·sc + bases`，`g` 是每 token 标量、`sc` 是分段常量）：

```
du0_k = du_k · g · sc_k
dg   += Σ_k du_k · u0_k · sc_k
dsc_seg += Σ_{k∈seg} du_k · u0_k · g
dbases += du
```

**③ 归一化尺度与平方和**（`g = rsqrt(q/D+eps)`，`q = Σ z²`）：

```
dq = dg · (−½ g³ / D)
dz = dq · 2 z
```

这是最容易漏、也最值得强调的一项：**`g` 把一个 token 的 24 个投影输出全部耦合到同一个 `q` 上**。
若把 `g` 当常量（丢 `dq` 项），残差/输入梯度会系统性错——§4 的负向对照实测 `dx relerr = 6.18e-2`
（而权重类梯度 `dfn``dgamma` 完全不受影响，因为它们不经 `q`）。

**② 投影 GEMM 与 `z`**：`u0_k = z·w_k`（`w_k = fn[k]`）。两条回流：

```
dz  += Σ_k du0_k · w_k       # dgrad：投回残差（z 是 s 展平）
dfn += Σ_t du0_k · z         # wgrad：投影矩阵的梯度
```

**① 残差多流混合**（`s_j = bf16(p_j·x + Σ_i C_{j,i} r_i)`，bf16 按 straight-through）：

```
dA_j = ds_j + dz_j           # dz.reshape 回 [hc,H]：投影路也回流到残差（第二处易漏项）
dx   = Σ_j dA_j · p_j
dp_j = Σ_h dA_{j,h} · x_h
dC_{j,i} = Σ_h dA_{j,h} · r_{i,h}
dr_{i,h} = Σ_j C_{j,i} · dA_{j,h}
```

至此得到对 `x, residual, post_mix, comb_res_mix, fn, scales, bases, rmsnorm_weight`（及 shifted 的
`shifted_prev_mix`）的全部梯度。

### 2.1 三个非平凡点

1. **Sinkhorn 的迭代 VJP**：不是「一次归一化的反」，而是**整条交替归一化序列的逆**；每一步
   `除以和` 都同时改「分子」和「分母」两路，必须两者都算。
2. **`g` 的耦合项 `dq`**：24 个投影输出被同一个 `q` 拉在一起；漏掉它只错残差/输入这类经 `q` 的
   梯度（负向对照坐实）。
3. **两处 bf16 共享边界 + 两条易漏回流**：`new_residual`（`impls:496-497`）与 `norm_input`
   （`test:177`）在连续图上按 straight-through；`dz`（投影路）必须回流到 `s`，否则 `dx/dR/dpost/dcomb`
   全错（本篇开发时真踩到，见 §5）。

---

## 3. 与 Mega Gate 反向的关系

第 127 篇的 Mega Gate 反向处理 `sqrt(softplus)` 打分 + top-k 归一化权重；本篇处理 mHC 的
**Sinkhorn 双随机 + RMSNorm**。两者共享同一类「归一化 Jacobian」结构：

| 归一化 | 前向 | 反向 Jacobian | 出处 |
|---|---|---|---|
| top-k 权重和 | `w_i = s_i/(Σs+eps)` | `ds_i = (dy_i·S − Σ dy_j s_j)/S²` | 第 127 篇（式 (6)） |
| Sinkhorn 行/列 | `v/(Σv+eps)` | `dX = dY/(S+eps) − Σ(dY·X)/(S+eps)²` | 本篇 §2⑤ |
| RMSNorm | `n/√(mean(n²)+eps)` | 标准两路（直传 + `2/H·n·dm`） | 本篇 §2⑧ |

三者都不是「按元素」的，而是**跨槽/跨维的耦合**——这正是这类「归一化密集」算子的反向难点。

---

## 4. H100 实测

> 数字来自 `harness/exp_mega_mhc_backward.sh`（`src/mega_mhc_backward.cu` + 纯 torch 脚本），
> 完整输出见 `harness/exp_mega_mhc_backward.out.txt`。主 kernel 端到端仍需 SM100。

### 4.1 手写 VJP vs torch autograd（float64，`harness/exp_mega_mhc_backward.py`）

**A. Sinkhorn 反向**（4×4，独立核验）：

| iters | dlogits vs autograd relerr | 中心差分 max_abs | 双随机残差（row / col） |
|---|---|---|---|
| 1 | 1.62e-16 | 4.0e-10 | 2.25e-1 / 2.0e-8 |
| 2 | 2.14e-16 | 3.6e-10 | 4.4e-2 / 1.1e-8 |
| 3 | 1.28e-16 | 1.5e-10 | 8.0e-3 / 1.0e-8 |
| 5 | 2.08e-16 | 2.4e-10 | 5.2e-4 / 1.0e-8 |

（行和偏 2.25e-1 出现在 **iters=1**（只做一次列归一化）——此时矩阵还不是双随机的，符合算法语义；
iters 越大越接近双随机。）

**B. 整变换反向**（6 组形状，normal / shifted；每行括注各张量 relerr 的上界）：

```
T=16 H= 8 it=3 normal   max=2.46e-16   （dx..dgamma 全 ≤ 2.5e-16）
T=16 H= 8 it=3 shifted  max=3.49e-16   （含 dshift）
T=32 H=16 it=4 normal   max=2.91e-16
T=32 H=16 it=4 shifted  max=1.01e-15
T= 8 H=32 it=2 normal   max=7.46e-16
T= 8 H=32 it=5 shifted  max=4.01e-16
最坏 1.007e-15
```

**B2. 负向对照**（normal，同 seed）：含 `dq` 项 `dx relerr = 1.09e-16`；**丢掉 `dq` 项**
`dx relerr = 6.18e-2`，而 `dfn/dgamma` 仍为 `2.7e-16/1.4e-16`——坐实「`g` 的耦合只在经 `q` 的
残差/输入梯度上」。

**C. 中心差分独立核验**（normal，抽 7 个标量）：analytic vs fd 的最差 `2.2e-11`（`x[4]`）。

### 4.2 device(fp32) vs double 参考（`src/mega_mhc_backward.cu`）

同一个模板化 `mhc_fwd_bwd<T>` 在 host 用 `double`、在 device 用 `float` 跑同一算法，比对全部梯度：

| H | normal | shifted |
|---|---|---|
| 8 | 7.98e-8 | 8.63e-8 |
| 16 | 7.55e-8 | 1.32e-7 |
| 32 | 9.52e-8 | 3.58e-7 |

**A. 4×4 Sinkhorn 反向解析 vs double 有限差分**：iters=1..5 最差 `8.9e-11`。
**C. double 参考自身的中心差分抽样**：`gamma/bases/fn/comb/post/R/x` 7 项最差 `3.6e-11`——
证明「参考」不是自证。

---

## 5. 开发中的两个真实踩坑（记录以便复核）

1. **漏了 `dz → s` 回流**：第一版把投影路的 `dz` 只用来算 `dfn`，没加回 `dA`，导致
   `dx/dR/dpost/dcomb` 全部错 `~0.1–0.2`、而 `dfn/dgamma` 完全正确。修复即加上
   `ds += dz.reshape(hc,H)`。**这解释了为什么「只对权重、不对输入」的错往往指向特征通路漏项。**
2. **shifted 的 `pre` 有两条上游**：normal 下 `pre` 只经 `norm_input`；shifted 下 `pre` 还是输出
   `new_prev_mix`。第一版漏了「作为输出」的那条，导致 shifted 的 `dfn/dscales/dbases` 错
   （因为 `du[:hc]` 少了一项），而 normal 全对。

两处都靠「分组对比 autograd + 有限差分」定位：**权重类梯度对（`fn`,`scales`,`bases`,`γ`）先全对**，
把错误范围缩小到「残差/输入特征通路」。

---

## 6. 边界与小结

- **仅分析**：Mega mHC 主 kernel（`sm100_mega_mhc.cuh`）是 **SM100 专属、且只有前向**；
  本篇的「反向」是按它的语义推导，UMMA/TMEM/SplitBarrier 不涉及。
- **可跑**：计算图数学与架构无关，H100(sm90) 真机跑真实数值（两条独立路径 + 有限差分）。
- **一句话**：mHC 反向 = **残差混合反向 × 投影 GEMM 反向 × sigmoid/仿射反向 × Sinkhorn 迭代 VJP ×
  RMSNorm 反向**，其中最难的两处是 **Sinkhorn 的交替归一化逆放** 与 **`g` 把 24 路输出耦合到 `q`**
  ——后者是「只改残差尺度」的归一化里最容易被漏掉的一项。

---

## 参考

- DeepGEMM（`057ca59`）：`tests/test_mega_mhc.py:114-210`、`csrc/apis/mega_mhc.hpp:89-158,223`、
  `deep_gemm/include/deep_gemm/layout/mega_mhc.cuh:11-41`、
  `deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh:235-275,486-512`、
  `deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh:28-86,112-113,188-244,323-365`。
- 仓库：https://github.com/deepseek-ai/DeepGEMM
- 本专题：第 5 篇（mHC 前向，`docs/05`）、第 4 篇（MoE router 反向，`docs/04`）、
  第 127 篇（Mega Gate 反向，`docs/127`）、第 16/97 篇（HC prenorm GEMM，`docs/16`、`docs/97`）。
- 实验：`harness/exp_mega_mhc_backward.{py,sh,out.txt}`、`src/mega_mhc_backward.cu`。
