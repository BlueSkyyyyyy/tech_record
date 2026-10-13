# Mega mHC 反向的 H100 融合账：三段窄输出长 K GEMM + 三段可融合元素级

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 **P121-1** 增量（第 131 篇）。
>
> 第 130 篇（P120-1）把 Mega mHC 反向的**数学**补上了（残差多流混合、Sinkhorn 迭代 VJP、`g` 的耦合项、
> RMSNorm 与两处 bf16 边界），并证明「手写 VJP ≡ autograd」。本篇换一个工程视角：**把这条反向当成
> 一份待部署的算子流水**——它由几段算子组成？各段「重」在哪？在 H100 上是**算力受限还是带宽受限**？
> 融合能省多少（省 launch？省中间张量？）以及在目标卡（SM100）上它该怎么映射。
>
> **可跑性**：mHC 主 kernel（`sm100_mega_mhc.cuh`）是 SM100 专属、且**只有前向**，本篇的「反向流水」
> 是按它的语义（权威参考 `tests/test_mega_mhc.py:123-210` 的 `mhc_reference`）推导的；但这条流水的
> **每一段都与 UMMA/TMA/TMEM 无关**，可以在 H100(sm90) 上真跑真实设备代码与真实形状。源码行号按
> DeepGEMM `057ca59` 核对。

---

## 1. 为什么把反向拆成「流水」而不是只推导公式

第 130 篇答的是**梯度对不对**（数学）。要做工程决策还差三个问题：

1. **段数**：这条反向能拆成几段可独立并行/融合的算子？哪些能融合、哪些必须是 GEMM？
2. **瓶颈**：每段的 FLOPs、访存字节、算术强度（AI，flop/byte）是多少？相对 H100 的 roofline 拐点，
   它是算力受限还是带宽受限？
3. **融合收益**：朴素「逐 op」实现（torch autograd 那样一段一个 kernel）离理论访存下界有多远？

mHC 这个算子特别值得做这个账，因为它的投影矩阵 `fn` 是**窄输出、长 K**（`Nout = hc(hc+2) = 24`，
`K = hc·H = 4H`）——和 `tf32_hc_prenorm_gemm`（第 16/97 篇）是同一种几何，天生**带宽受限**。

---

## 2. 六段算子流水

记 `hc = 4`（契约 `csrc/apis/mega_mhc.hpp:138`、`layout/mega_mhc.cuh:11`）、`D = hc·H`、
`Nout = hc(hc+2) = 24`。前向（`test_mega_mhc.py:145-184`）：

```
A  = post ⊙ x + comb @ R            # [hc,H]   残差多流混合（每 token）
u0 = A.flatten() @ fnᵀ               # [Nout]   唯一 GEMM（窄输出长 K）
... q/g、sigmoid/仿射、Sinkhorn、norm_input、RMSNorm ...
```

反向（上游 `dS`(对 new_residual)、`dPost`(对 new_post)、`dM`(对 comb)、`dY`(对 y)，shifted 另有 `dShift`）
自然地分成**三段元素级**与**三段 GEMM**：

| 段 | 类型 | 输入 | 输出 | 备注 |
|---|---|---|---|---|
| **K0** fwd_mix | 元素级 | `x,R,post,comb` | `A[T,D]` | 重算 `new_residual` |
| **G0** u0 | GEMM | `A, fn` | `u0[T,24]` | `[T,D]@[D,24]`，dgrad 型 |
| **K1** coeff | 元素级 | `A,u0,dS,dY,dPost,dM,gamma` | `du0[T,24]`, `dAbase[T,D]`, `dgamma/dscales/dbases` | RMSNorm bwd + mix bwd + sigmoid/仿射 bwd + **Sinkhorn VJP** + q/g bwd |
| **G1a** dz | GEMM | `du0, fn` | `dz[T,D]` | `[T,24]@[24,D]`，dgrad 型 |
| **G1b** dfn | GEMM | `du0, A` | `dfn[24,D]` | `[24,T]@[T,D]`，wgrad 型 |
| **K2** mix_bwd | 元素级 | `dAbase,dz,x,R,post,comb` | `dx,dR,dpost,dcomb` | `dA = dAbase + dz` 后四条反向 |

**为什么恰好是这三段 GEMM**：G0 在**正向**里算 `u0`（重算前向的投影，因反向需要 `u0` 算 `du`），
G1a/G1b 是**投影反向**的 dgrad/wgrad。注意 `dz` 的两部分——`dq·2·A`（来自 `g` 的耦合，元素级）已折进
K1 的 `dAbase`，只有 `du0 @ fn`（GEMM）留在 G1a；这是唯一能省的：把可融合的标量部分从 GEMM 里摘出去。

**正确性**：`src/mhc_backward_fused.cu` 把这六段各写成一个真实 CUDA kernel（`k_fwd_mix`/`k_u0`/`k_coeff`/
`k_dz`/`k_dfn`/`k_mix_bwd` + 一个 `k_reduce`），device(fp32) 跑出的全部梯度与 host **单片 double 参考**
（第 130 篇同款 `mhc_fwd_bwd`）逐项对拍：

```
T= 64 H= 8 it=3 normal   worst relerr(fp32 device vs double) = 1.99e-07
T= 64 H= 8 it=3 shifted  worst relerr = 2.69e-07
T=128 H=32 it=4 normal   worst relerr = 3.22e-07
T=128 H=32 it=4 shifted  worst relerr = 2.61e-07
T=256 H=64 it=2 normal   worst relerr = 8.84e-07
T=256 H=64 it=5 shifted  worst relerr = 6.55e-07
```

（`shifted` 变体的 `pre` 多一条「作为输出」的回流路径，已覆盖；`dshift` 正确。）这证明「六段拆分 +
`fn` 全 batch 共享 + 三段 GEMM」与「单片逐 token」**语义一致**（第 130 篇已验证单片 ≡ autograd）。

---

## 3. 逐段账：FLOPs / 字节 / 算术强度

`harness/exp_mhc_backward_fused.py` 在真实形状（`H = 7168`、`hc = 4`、`D = 28672`、`Nout = 24`）下，
按每段实际读写的张量算字节、按 FMA 算 FLOPs：

```
H100: HBM 3.35 TB/s, bf16 989 TFLOPS → roofline 拐点 295 flop/byte

T=16384 H=7168:
  stage                 GFLOP        MB      AI
  K0 fwd_mix            3.758    2114.6     1.8
  G0 u0=A@fnT          22.549     941.7    23.9
  K1 coeff              3.758    3290.8     1.1
  G1a dz=du0@fn        22.549     941.7    23.9
  G1b dfn=du0T@A       22.549     941.7    23.9
  K2 mix_bwd            3.758    3993.5     0.9
  TOTAL                78.920   12223.9     6.5
  三段 GEMM 合计: 67.646 GFLOP, AI≈23.9 flop/byte （≪ 拐点 → 访存受限）
  访存下界（整条反向）≈ 3648.9 us/次；三段 GEMM 计算下界 ≈ 68.4 us/次
```

**结论一：整条 mHC 反向是访存受限，而不是算力受限。**

- **整条反向 AI ≈ 6.5**，远低于 bf16 拐点 295（即使换成 TF32，H100 TF32 拐点 ≈ 67e12/3.35e12 ≈ 20，
  AI=6.5 仍在其下）。
- **三段 GEMM 的 AI ≈ 24**（因为 `Nout = 24`，每读一个 `A`/`du0` 元素只做 ~24 次 FMA），
  同样 ≪ 295。**这就是「窄输出长 K」的本质：FLOPs 随 `Nout` 线性，访存随 `T·D` 线性。**
- 量化地看下界之比：`T=16384` 时整条反向访存下界 **3649 us**，而三段 GEMM 的计算下界仅 **68 us**
  ——**相差 53 倍**。也就是说哪怕 GEMM 一根 tensor core 都不停地满速算，也省不下大头；瓶颈全在搬数据。

这张表和第 16 篇 `tf32_hc_prenorm_gemm` 的结论**同构**：同样是「窄输出长 K 的系数投影 + 逐 token 归一化」，
同样是 bandwdith-bound。mHC 只是把它从「一个 GEMM」扩展成「三段 GEMM + 三段元素级」。

---

## 4. H100 实测

> 数字来自 `harness/exp_mhc_backward_fused.{sh,out.txt}`（`src/mhc_backward_fused.cu` + 纯 torch 脚本）。
> 主 kernel 端到端仍需 SM100；这里跑的是这条反向流水的真实设备代码与真实形状。

### 4.1 元素级三段：真实设备计时（H=64）

`src/mhc_backward_fused.cu` 把 K0/K1/K2 写成「一个 block 处理一个 token、线程 = h」的**合并访存**版本
（元素级段慢在归约与 per-token 串行，见下），在 `T = 65536, H = 64`（`D = 256`）下计时：

```
K0 fwd_mix     155.3 us   1005.8 GB/s
K1 coeff      3816.2 us     56.1 GB/s
K1 -sinkhorn  1396.2 us    153.2 GB/s      ← 计时归因：把 Sinkhorn 换成恒等
K1r reduce     152.7 us    312.5 GB/s      ← H+NOUT+3 个标量的并行归约（非 atomic 尾巴）
K2 mix_bwd    1371.3 us    208.0 GB/s
```

读法（三点）：

1. **K0 达 1006 GB/s ≈ HBM 峰值的 30%**，一个纯元素级融合核能把「读 4 个张量、写 1 个」压进一次
   launch，已经是访存受限的形态。
2. **K1 的「内存部分」是 153 GB/s，但完整版掉到 56 GB/s**——差的那 **2.4 ms 全在 Sinkhorn**。
   原因是本参考实现把 4×4 Sinkhorn 的 softmax + 交替归一化（`epilogue:63-86`）**串行在单线程**里跑，
   其余 63 个线程空等。这是一个**明确的实现坑**而非算法成本：Sinkhorn 是 `O(T·hc²·iters)` 的固定
   per-token 成本，与 `D` 无关，在真实 `H = 7168`（`D = 28672`）下会被 `T·D` 的访存彻底摊薄。
   （`-sinkhorn` 一行就是为此留的计时归因开关。）
3. **atomic 尾巴很贵**：把 `dgamma/dscales/dbases` 的 3.9M 次全局 `atomicAdd`（3 个地址上）换成
   「per-token 部分和 + 并行归约」后，K1 从 9.66 ms 降到 3.82 ms（`K1r reduce` 自己只要 153 us）。
   **教训**：per-token 融合核里对**低基数**输出做原子累加会打爆 L2 原子端口。

### 4.2 三段 GEMM 与朴素 torch 对比（真实形状 H=7168）

```
T=4096 H=7168 D=28672:
  G0  [T,D]@[D,24]         242.4 us   23.3 TFLOPS
  G1a [T,24]@[24,D]        256.9 us   21.9 TFLOPS
  G1b [24,T]@[T,D]         239.1 us   23.6 TFLOPS
  三段 GEMM 合计             738.4 us   22.9 TFLOPS
  元素级 op（torch 逐 op）  15896.5 us
  autograd 整条反向（朴素）  31012.0 us
  整条反向访存下界            913.2 us  （= 3059 MB / 3.35 TB/s）

T=16384 H=7168:
  三段 GEMM 合计            3037.3 us   22.3 TFLOPS
  元素级 op（torch 逐 op）  62271.2 us
  autograd 整条反向（朴素） 119497.4 us
  整条反向访存下界           3648.9 us  （= 12224 MB / 3.35 TB/s）
```

三点：

1. **三段 GEMM 只跑出 ~22 TFLOPS**（`Nout=24` 太窄，cuBLAS 也吃不满），即有效带宽 ≈ 22.3e12/23.9 ≈
   **0.93 TB/s**——这正是「小而窄的 GEMM 是访存受限」的实测形态，和 §3 的 AI≈24 完全吻合。
2. **元素级 op（torch 逐 op）在 `T=16384` 要 62.3 ms**，是访存下界（3649 us）的 **17 倍**，
   也不是三段 GEMM（3.04 ms）的对手——**朴素逐 op 的元素级段才是这条反向的真正大头**（几十个小 kernel，
   每个都来回搬 `[T,D]` 量级的张量）。这与 §4.1 里「K0 单核 1006 GB/s」对照，说明**融合的收益主要在这里**。
3. **整条 autograd（朴素）119.5 ms vs 访存下界 3.65 ms ≈ 33×**——这就是「把 mHC 反向当作待优化算子」
   的空间：不是靠更多 FLOPs，而是靠**把三段元素级压成三段合并访存的融合核 + 让三段窄 GEMM 的权重驻留**。

---

## 5. 该往哪里优化（结构性结论）

结合 §3 的账与 §4 的实测，mHC 反向的优化方向是**确定**的：

1. **别指望 Tensor Core**：三段 GEMM 的 AI≈24，计算下界只占访存下界的不到 2%。Kernel 设计目标应是
   **省访存/省 launch**，不是提算力。
2. **融合三段元素级**：K0/K1/K2 各自读 `[T,D]` 级别张量、中间还有 `A/dAbase/dz` 多次往返。
   把「K1 + K2」融成一个 per-token 核（K1 已生成 `dq·2A` 那部分，K2 只差 `dz`），可省一次
   `dAbase[T,D]` 的写读（每次 ~`2·T·D·2` 字节）。§4.2 里逐 op 的 62 ms 就是这条路的反面教材。
3. **低基数输出别用原子**：`dgamma[H]`、`dscales[3]`、`dbases[24]` 用 per-token 部分和 + 归约，
   实测把 K1 砍掉 ~60%（§4.1）。
4. **Sinkhorn 别串行**：`hc=4` 的软最大/行和列和也是一个 warp 内 `shfl_xor` 能做完的事（前向
   `epilogue:71-76` 正是这么做的），参考实现串行在单线程是纯浪费（§4.1 的 2.4 ms）。

---

## 6. 目标卡（SM100）上的映射：仅分析

DeepGEMM **没有** mHC 反向 kernel，所以这里只做「若要在目标卡上实现，这段流水如何落到 SM100 的既有机制」
的推演（不实测）：

- **G0/G1a/G1b 三段窄 GEMM** 可直接复用 SM100 的 `sm100_bf16_gemm`（第 26/36 篇，`UMMA_K=16`、
  合并 stage）或 `sm100_fp8_fp4_gemm_1d1d`（第 25/35 篇）。注意 `Nout=24` **不是** 8/16 的干净倍数，
  host 契约（如 `k_alignment%block_k`）需要 padding 到 block 边界；TMEM 列预算对 `UMMA_N=24` 很宽裕。
  由于 AI≈24，`set_tc_util` 节流（第 36 篇）在这里有意义——算力本就过剩。
- **三段元素级**可以按前向 `sm100_mega_mhc.cuh` 的思路融进 epilogue：前向的 6 个 warpgroup 里已有
  「normal worker 把 Mix+Pre+RMSNorm 融成 per-token worker」（`epilogue:139-281`）。反向的 K1/K2
  是它的对偶：同一批 `shfl_xor` 归约（Sinkhorn）、同一套 per-token 映射，只是方向相反。**它对 UMMA
  的依赖仅在于 `u0` 来自 G0**——把 G0 的累加器留在 TMEM、epilogue 直接把 `u0` 喂给 K1，就能省掉
  `u0[T,24]` 的一次 global 往返。
- **Sinkhorn VJP 的 4×4 逆放**（第 130 篇 §2⑤）是 `iters` 步固定的标量链，放在单个 warp 里、用
  `shfl_xor` 做行/列和即可——与第 5 篇前向 `epilogue:63-86` 完全同构，只是把归一化顺序反过来。
- **`dq` 耦合项**（24 个输出共享一个 `q` 的平方和）在目标卡上就是一次 warp 级全归约 + 广播，
  和前向的 `hc_sqr_sum`（`impls:501`）对称。

---

## 7. 边界与小结

- **仅分析**：Mega mHC 主 kernel（`sm100_mega_mhc.cuh`）是 **SM100 专属、且只有前向**；本篇的「反向
  流水」是按它的语义推导，UMMA/TMEM/SplitBarrier 不涉及。
- **可跑**：反向流水的数学与形状全部架构无关，H100(sm90) 真机跑**真实设备代码**（六段 kernel）与
  **真实形状**（`H=7168`）；正确性三路对齐（device 六段 vs double、手写 vs autograd、第 130 篇的
  单片 vs autograd）。
- **一句话**：mHC 反向 = **三段窄输出长 K 投影 GEMM（AI≈24）+ 三段可融合元素级（含 Sinkhorn VJP）**；
  整条 **访存受限**（AI≈6.5 ≪ 拐点 295），三段 GEMM 的计算下界只有访存下界的 ~2%。优化的空间不在
  Tensor Core，而在**融合元素级、避免低基数原子、别串行 Sinkhorn、让投影权重驻留**——实测朴素 torch
  autograd 比访存下界慢 **~33×**。

---

## 参考

- DeepGEMM（`057ca59`）：`tests/test_mega_mhc.py:114-120,123-210`、
  `csrc/apis/mega_mhc.hpp:138`、`deep_gemm/include/deep_gemm/layout/mega_mhc.cuh:11`、
  `deep_gemm/include/deep_gemm/impls/sm100_mega_mhc.cuh:486-512`、
  `deep_gemm/include/deep_gemm/epilogue/sm100_mega_mhc.cuh:43,53-56,63-86,71-76,139-281`。
- 仓库：https://github.com/deepseek-ai/DeepGEMM
- 本专题：第 5 篇（mHC 前向，`docs/05`）、第 130 篇（mHC 反向数学，`docs/130`）、
  第 16/97 篇（HC prenorm GEMM，同构的带宽受限窄输出 GEMM，
  `docs/16`/`docs/97`）、第 25/26/35/36 篇（SM100 GEMM 后端与 `set_tc_util`）。
- 实验：`harness/exp_mhc_backward_fused.{py,sh,out.txt}`、`src/mhc_backward_fused.cu`。
