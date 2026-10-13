# MoE 反向的 H100 数值复刻与误差分解

> 第 14 篇（P4-4）把四个 SM90 grouped GEMM kernel 接进了真实 MoE 的反向图，回答了
> **「这些 kernel 接得对不对」**（`vs deq` ~2.6e-4，隔离 kernel + 布局）。第 99 篇（P89-1）把 **前向**
> 用 H100 上真实可跑的算子逐段复刻，量出了 **「端到端误差多少、每一段各贡献多少」**。
> 但**反向**一直缺这块对应物：四项梯度里**误差从哪来、能不能可加地拆开、梯度方向是否保住**，
> 从没被量化过。
>
> 本篇补上——它是第 99 篇前向复刻的**反向对偶**。**Mega MoE 没有反向 mega-kernel**（第 14 篇 §6：
> 反向天然是可分解的独立 grouped GEMM），所以这里的「反向实现解析」= 四个 GEMM + SwiGLU/combine/router
> 反向。所有数字来自 `harness/exp_moe_backward_emul.{py,sh,out.txt}`（H100，DeepGEMM 2.8.1）。
> 主 kernel（SM100 tcgen05/packed UE8M0）为**仅分析**；本机用 SM90 的
> `k_grouped_fp8_gemm_nt_contiguous` / `m_grouped_fp8_gemm_nt_contiguous`。

---

## 1. 反向数据流：权重乘在哪，误差就灌在哪

Mega MoE 前向的一个反直觉设计是**把 top-k 权重提前乘进 L1 输出**（`sm100_fp8_fp4_mega_moe.cuh`
`:1087-1134`：`activation_values = __fmul2_rn(__fmul2_rn(gate, up), weights)`，权重由
`l1_topk_weights_buffer` 每 32 token 读一次、warp 内广播，`:1097` 缓存、`:604` 写入）。
于是 combine 段退化成**纯 fp32 求和**（`:1477-1490` 的 `ptx::accumulate`，`:1512-1515` TMA 回写），
不再乘权重。

这条设计对**反向**有直接后果：

- combine 的反向是 **`dY` 直传**（`y_t = Σ_k z_{t,k}` ⇒ `dz_{t,k} = dY_t`），无权重账本；
- top-k 权重 `w` 出现在 **SwiGLU 反向**里：前向 `a = silu(g)·u·w`，所以
  `dg = dA·w·silu'(g)·u`、`du = dA·w·silu(g)`（`w` 是常量，`tw.detach()`）。

记权重约定 `W1=[E,2I,H]`、`W2=[E,H,I]`（于是 `g=x@W1[:I]ᵀ`、`z=a@W2ᵀ`），则四个大 GEMM 的角色是
（与第 14 篇同表，仅 W 的存储主序不同）：

| 梯度 | 数学式 | 不等长轴 | kernel | A / B 角色 |
|---|---|---|---|---|
| `dW2` | `dYᵀ·a` | K=token | `k_grouped_*_nt_contiguous` | A=dY（激活）、B=a（激活） |
| `dA` | `dY·W2` | M=token | `m_grouped_*_contiguous` | A=dY（激活）、B=W2（权重） |
| `dW1` | `dGᵀ·x` | K=token | `k_grouped_*_nt_contiguous` | A=dG（激活）、B=x（激活） |
| `dX` | `dG·W1` | M=token | `m_grouped_*_contiguous` | A=dG（激活）、B=W1（权重） |

host 入口：wgrad 走 `csrc/apis/gemm.hpp:373`、dgrad 走 `:181`；K 轴对齐硬约束在
`check_k_grouped_args`（`:69` 的 `k % k_alignment == 0`）。**读法**：wgrad 的两个操作数都是
**激活**（token 行），dgrad 的 B 侧是**权重**——这决定了「激活量化」与「权重量化」在反向里作用在
哪些梯度上。

## 2. H100 复刻方案与「分解」的思路

主 kernel 跑不了，但**量化语义架构无关**。第 99 篇前向的复刻用真实 SM90 kernel；本篇为了把误差
**可加地**拆开，做两层：

1. **真实 kernel 锚点（§3）**：四个 GEMM 换成 SM90 kernel，确认「量化 + kernel + 布局」的总误差，
   并复现第 14 篇的 kernel 误差量级。
2. **量化来源分解（§4，架构无关）**：把每个操作数按真实语义量化（激活 per-token 1D、权重 per-128 块
   2D、上游梯度 `dY` per-token），在**去量化后的操作数**上做精确 GEMM（不做 kernel）。这样误差**只剩
   量化**，且可以逐来源开关。参考梯度用 **fp64 手写反向**（`ffn_backward`），并与 `torch.autograd.grad`
   交叉验证。

量化用的正是 DeepGEMM 的测试参考量化器 `deep_gemm/utils/math.py` 的
`per_token_cast_to_fp8` / `per_block_cast_to_fp8`（第 37 篇拆过其语义：`amax/448`、UE8M0 取整、
gran 128）。分解的开关组合：

- `none`：前后向都不量化 → 应只剩 **fp32 GEMM 相消**（相对 fp64 的舍入地板）；
- `fwd_only`：只量化**前向**的 L1/L2 操作数（激活 + 权重）；
- `+bwd_act`：再叠加**反向激活**量化（`dG`、`dY`、`a`、`x`）；
- `+bwd_w`：再叠加**反向权重**量化（`W1`、`W2` 在 dgrad 里）；
- `+dy`：再叠加**上游梯度** `dY` 自身的 per-token 量化；
- `full`：全部叠加。

## 3. 真实 SM90 kernel 反向锚点

配置 `T=64, H=512, I=256, E=8, topk=2`（每个专家 13~20 个 token）：

```
kernel vs fp64 : dX rel=3.4857e-02 cos=0.999384 | dW1 rel=2.4098e-02 cos=0.999662 | dW2 rel=2.4151e-02 cos=0.999663
kernel vs deq  : dX rel=1.4874e-03 | dW1 rel=2.6313e-04 | dW2 rel=2.6334e-04
```

读法：

- **`vs deq` ~2.6e-4（dW）/1.5e-3（dX）**：把喂进 kernel 的 FP8 操作数反量化后在 fp32 重算，只剩
  **kernel + 布局 + bf16 输出舍入**。dW 与第 14 篇的 2.6e-4 逐位同量级。dX 更高（1.5e-3）是因为
  **dgrad 的输出会被两个 top-k 槽 scatter-add、每个槽是 bf16**，bf16 舍入叠加。
- **`vs fp64` 2.4e-2~3.5e-2**：含 FP8 量化。dX（3.5e-2）高于 dW（2.4e-2），与第 14 篇结论一致
  （dgrad 的激活侧量化后还要过 SwiGLU 反向放大）。

**方向完全保住**：三项 `cos ≥ 0.9994`。

## 4. 量化来源分解（三个配置）

`relerr` 相对 fp64 参考，`cos` 为梯度方向余弦：

```
-- T=64 H=512 I=256 E=8 topk=2
   none      dX rel=6.99e-04 cos=1.000000 | dW1 rel=6.27e-04 cos=1.000000 | dW2 rel=5.22e-04 cos=1.000000
   fwd_only  dX rel=7.23e-02 cos=0.996952 | dW1 rel=5.30e-02 cos=0.996988 | dW2 rel=5.52e-02 cos=0.998595
   +bwd_act  dX rel=7.59e-02 cos=0.996703 | dW1 rel=6.51e-02 cos=0.996390 | dW2 rel=5.98e-02 cos=0.998345
   +bwd_w    dX rel=8.25e-02 cos=0.996249 | dW1 rel=6.29e-02 cos=0.996621 | dW2 rel=5.52e-02 cos=0.998595
   +dy       dX rel=7.71e-02 cos=0.996636 | dW1 rel=6.21e-02 cos=0.996671 | dW2 rel=6.06e-02 cos=0.998260
   full      dX rel=8.97e-02 cos=0.995655 | dW1 rel=7.83e-02 cos=0.995691 | dW2 rel=6.46e-02 cos=0.998010
-- T=256 H=1024 I=512 E=8 topk=2   (full: dX 9.87e-02 / dW1 8.42e-02 / dW2 6.36e-02)
-- T=512 H=1024 I=512 E=16 topk=4  (full: dX 9.84e-02 / dW1 8.40e-02 / dW2 6.36e-02)
```

三个非平凡结论：

1. **`none` 不是 0，而是 ~5e-4~7e-4**：这是 **fp32 GEMM 累加 vs fp64 参考**的舍入地板（cos 严格 1.0）。
   分解里一切超过它的部分才是量化贡献。
2. **前向量化是绝对主导**：`fwd_only` 就把 dX 推到 7.2e-2（占 `full` 8.97e-2 的 **80%**），
   反向四项加起来只多 ~1.7e-2。这与第 99 篇前向端到端 ~6.3e-2 同量级——**反向的量化误差一大半是前向
   「带进来的」**（前向把量化后的激活/权重存下来，反向在这些已含噪的值上再算梯度）。
3. **反向里权重 vs 激活的分工逐梯度不同**：
   - `dW2` **对反向权重量化完全免疫**（`+bwd_w` 的 dW2 与 `fwd_only` 逐位相同，5.52e-2）——
     因为 `dW2=dYᵀ·a` 两个操作数都是激活，**根本不含 W2/W1**；
   - 反过来 `dX` 对**反向权重量化最敏感**（`+bwd_w` 把 dX 从 7.23e-2 抬到 8.25e-2）——
     `dX=dG·W1` 的 B 侧正是权重，per-128 块 2D 量化比 per-token 1D 粗；
   - `dW1` 对**反向激活量化**更敏感（`+bwd_act`：5.30e-2→6.51e-2）——`dW1=dGᵀ·x` 两侧都是激活。

**梯度方向**：最坏情况 `full` 下 dX cos=0.9957、dW2 cos=0.9980，**全部 > 0.99**，即 FP8 反向保住
优化方向（与第 14 篇 autograd 对拍的 cos>0.999 一致；此处因为还叠了前向量化所以略低）。

## 5. DeepSeek 量级（T=2048, H=7168, I=2048, E=8, topk=2）

```
   fwd_only  dX rel=1.0651e-01 cos=0.993972 | dW1 rel=8.16e-02 cos=0.993968 | dW2 rel=5.22e-02 cos=0.998644
   full      dX rel=1.1844e-01 cos=0.992684 | dW1 rel=1.0075e-01 cos=0.992699 | dW2 rel=6.25e-02 cos=0.998049
```

量级随规模**单调升**（每专家 token 数从 ~16 涨到 512，per-token/per-block 的相对 amax 估计更差），
但规律不变：**前向主导（~90%）、dW2 最鲁棒、方向 >0.99**。真实 kernel 吞吐见第 14 篇 §5
（wgrad ~100 TFLOPS、dgrad ~450+）。

## 6. 与前向对照：一个「误差账本」

| 阶段 | P89 前向（第 99 篇） | 本篇反向 |
|---|---|---|
| 端到端 vs fp64 | ~6.3e-2 | dX ~9.0e-2 / dW1 ~7.8e-2 / dW2 ~6.5e-2 |
| 主导来源 | 激活与权重**各占一半**（`仅激活` 5.7e-2、`仅权重` 4.6e-2） | **前向整体**带入 ~80–90% |
| kernel + 布局误差 | ~1.5e-3（分层） | ~2.6e-4（dW）/1.5e-3（dX） |
| cos | 0.998 | 0.992–0.998 |

两张账单能对上：前向把「激活量化的 5.7e-2」和「权重量化的 4.6e-2」写进激活/权重，反向在**这些
已含噪的数**上再做四个 GEMM，于是反向误差 ≈ 前向误差（带进来的）+ 反向新引入的（~1.7e-2）。
**实践含义**：想把 MoE 反向的梯度精度提上去，先治**前向**的 block-scale 量化（例如 gran 32 或
更高精度 SF），反向自身的量化是二阶项。

## 7. 小结

- **反向 = 四个独立 grouped GEMM**（第 14 篇），没有 mega-kernel；wgrad 两操作数都是激活、
  dgrad 的 B 侧是权重——这条分工直接决定「权重量化只进 dgrad、不影响 wgrad」。
- **权重提前乘进 L1**（`kernel:1134`）使 combine 反向退化成 `dY` 直传、`w` 只出现在 SwiGLU 反向。
- **误差可加分解**（§4）：前向主导、反向权重只作用在 dgrad、方向全部 >0.99。
- **架构无关**：整个分解只用 DeepGEMM 的参考量化器 + fp64 参考，本机（H100）跑出全部数字；
  SM100 主 kernel 仅分析。

---

### 源码坐标（`~/github/DeepGEMM@057ca59`，行号已核对）

- 权重提前乘进 L1 输出：`deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:1087-1134`
  （`:1097` 缓存权重、`:1112-1113` gate/up round bf16、`:1134` `__fmul2_rn(..., weights)`）。
- combine 纯求和：同文件 `:1477-1490`（`ptx::accumulate`）、`:1512-1515`（TMA 回写）。
- wgrad host 入口：`csrc/apis/gemm.hpp:373`；dgrad host 入口：`:181`；K 轴对齐：`:69`。
- 参考量化器：`deep_gemm/utils/math.py`（`per_token_cast_to_fp8`、`per_block_cast_to_fp8`）。

### 实验

- 驱动：`harness/exp_moe_backward_emul.sh` → `harness/exp_moe_backward_emul.py`；
  原始输出 `harness/exp_moe_backward_emul.out.txt`。
- 约束：SM100 `sm100_fp8_fp4_mega_moe.cuh` 本机不可跑（`mega_moe.hpp:256` 只在 `arch_major==10`
  派发）；故主 kernel 仅分析，反向的数学/量化/布局用 SM90 复刻。

### 线上资料

- DeepGEMM 官方 README 的 MoE backward 工具函数行：`README.md`（`k_grouped_*` / `m_grouped_*`）。
- DeepSeek 官方博客与 GitHub PR `#95`（MoE backward）、`#316`（Mega MoE）。
