# HyperConnection prenorm GEMM：把「系数投影」与「RMSNorm 平方和」融进一个 TF32 kernel（H100 实测）

> 本文是「DeepSeek Mega MoE 前向 / 反向算子实现解析」专题的 P7 增量（第 16 篇）。
> 第 13 篇讲 **Mega mHC** 时提到：HyperConnection 里 `new_residual` 要同时被两处消费——
> 投影出 **mix 系数**（`new_residual @ fnᵀ`）和提供 **RMSNorm 的分母**（`Σ new_residual²`）。
> `sm100_mega_mhc.cuh` 把这两件事**融进了 mega-kernel 内部**（TF32 TS UMMA，`impls:235-275`）。
> 本篇拆解它的**独立非融合版本**——`deep_gemm.tf32_hc_prenorm_gemm`：它**有 SM90 实现**，
> 所以本机 H100 可以真跑，给出数值、TF32 语义、融合收益与性能的实测数字。
>
> 源码坐标：`~/github/DeepGEMM`，公开版 commit `057ca59`（v2.1.1.post3）。SM100 分支为**仅分析**。

## 1. 它解决什么：一个「小 N、大 K」的融合前置算子

HyperConnection（mHC）里每个 token 的激活（残差多流混合后的 `new_residual ∈ R^{m×k}`）会立刻被两处用到：

1. **mix 投影**：`mixes[m, n] = new_residual[m, :] · fn[n, :]`，其中 `fn` 是系数投影权重，
   输出通道数 `n` 很小（测试里 `n=24`，即「hyperconnection 系数通道数」，如 `hc×4` 之类）；
   随后 `mixes` 走 `sigmoid` 得到 pre/post 与 Sinkhorn 用的 comb（见第 13 篇）。
2. **RMSNorm 分母**：`sqr_sum[m] = Σ_k new_residual[m,k]²`，即 RMSNorm 的「均方」。

两件事**读的是同一份 `new_residual`**。朴素做法是两次遍历：一次 GEMM、一次 reduce。
`tf32_hc_prenorm_gemm` 把它合成一个 kernel，对 A 只读一遍就同时产出 `D`（投影结果，fp32）
和 `S`（平方和，fp32）：

```
D[m, n] = Σ_k A[m,k]·B[n,k]        A: bf16 (m×k)   B: fp32 (n×k)   D: fp32 (m×n)
S[m]    = Σ_k A[m,k]²              n 很小 (≤32 on sm90 / ≤128 on sm100)，k 很大
```

**为什么 B 用 fp32 而不是 bf16**：系数投影的尾数精度直接决定路由/Sinkhorn 系数的噪声，
不能像普通激活那样压到 bf16；于是 A 用 bf16（激活本就是这个精度），B 保 fp32，
MMA 走 **TF32**（10-bit 尾数）。这也是第 13 篇里 mega-mHC 的 mix 投影选 TF32 TS 的同一原因。

## 2. 接口与 host 端契约

host 入口在 `csrc/apis/hyperconnection.hpp`：

- `tf32_hc_prenorm_gemm(a, b, d, sqr_sum, num_splits)`（`:10-55`）。dtype 契约：`a`=bf16、
  `b`=fp32、`d`/`sqr_sum`=fp32（`:37-40`）；A/B 必须 **K-major**、D 为 N-major（`:16-18`）；
  `sqr_sum` 必须连续（`:21`）。
- 形状：`a:(m,k)`、`b:(n,k)`、`sqr_sum:(m,)`；`num_splits` 给出时 D 是 3D `(num_splits,m,n)`、
  `sqr_sum` 是 `(num_splits,m)`（`:26-35`）。空问题直接返回（`:43-44`）。
- **按架构派发**（`:47-54`）：`arch_major==9 → sm90_*`，`==10 → sm100_*`，否则 unreachable。
  这就是本机 H100 能跑、目标卡走 SM100 的分叉点。

SM90 侧参数契约（`csrc/jit_kernels/impls/sm90_tf32_hc_prenorm_gemm.hpp:21-31`）：

```
block_m = 64, block_k = 64
num_math_threads = 128, num_tma_threads = 128   （2 个 warpgroup）
block_n = align(n, 16)
DG_HOST_ASSERT(n <= block_n);
DG_HOST_ASSERT(n <= 32 and n % 8 == 0);   // ← sm90 只支持小 N
DG_HOST_ASSERT(k % block_k == 0);
```

SM100 侧（`sm100_tf32_hc_prenorm_gemm.hpp:21-29`）把 `n` 上限放宽到 **128**，并把线程拆成
`num_mma_threads=128 + num_cast_and_reduce_threads=128`（下面 §4）。

## 3. SM90 kernel 逐段精读（`sm90_tf32_hc_prenorm_gemm.cuh`）

**3.1 grid / 分块（`:107-115`，配套 launch `hpp:106-113`）**

```
kNumKBlocks = ceil_div(K, BLOCK_K)              // 每个 K 块 64
kNumKBlocksPerSplit = kNumKBlocks / kNumSplits
kRemainKBlocks      = kNumKBlocks % kNumSplits
block_idx   = blockIdx.x
m_block_idx = block_idx / kNumSplits            // 同一 M 块的多个 split 连续排布
k_split_idx = block_idx % kNumSplits
k_offset    = (k_split_idx*kNumKBlocksPerSplit + min(k_split_idx, kRemainKBlocks))*BLOCK_K
m_offset    = shape_m * k_split_idx             // sqr_sum 的写出偏移（split 段）
```

即 `grid = num_splits × ceil_div(m, BLOCK_M)`，split-K 的 D/S 各写一段、host/test 再 `sum(0)`。
注意 **split-K 对 D 和 S 同时切**：`m_offset = shape_m * k_split_idx` 让每个 split 的 S 写进
`sqr_sum[k_split_idx*shape_m : ...]`。`cudaGridDependencySynchronize()`（`:120`）表明它支持
**PDL（programmatic dependent launch）**，可被上游 kernel 紧随启动。

**3.2 两个 warpgroup 的分工（`:123-148`）**

- **TMA warp**（`warp_idx == 128/32 == 4`，`:123-146`）：`warpgroup_reg_dealloc<40>` 压到 40 寄存器，
  按 `kNumStages` 环形流水发 `tma::copy` 载入 A（bf16）与 B（fp32），`arrive_and_expect_tx`。
- **Math warps**（`warp_idx < 4`，`:147-285`）：`warpgroup_reg_alloc<256>` 拿满 256 寄存器做 WGMMA。

full/empty barrier 初始化里，**empty 的到达数是 128**（`:99`），因为 4 个 math warp 的 128 个线程
每个 stage 都要 arrive；full 的到达数是 1（TMA 单线程 arrive）。

**3.3 A 的「读 = 转换 = 平方累加」三合一（`:179-203`）——这是融合的关键**

A 不走 smem descriptor 直接喂 WGMMA，而是从 smem 手动 LDS 到寄存器、顺路做两件事：

```c
// :180  行号：warp_idx*16 + lane/4，对应 wgmma 64x8 A 片段布局
// :190-193  一次取 4 个 bf16（上/下两行各 2 个）
a_bf16[0]=upper[elem]; a_bf16[2]=upper[elem+4];
a_bf16[1]=lower[elem]; a_bf16[3]=lower[elem+4];
// :197-200  bf16 -> fp32（无损；也是 TF32 的输入）
a_float2_0 = __bfloat1622float2(...);  a[...] = ...
// :201-202  顺手把 fp32 值的平方累加进 2 个累加器
sqr_sum_acc_0 += x*x + y*y;   sqr_sum_acc_1 += ...;
```

**所以 `S = Σ A²` 完全是「搭便车」的**：平方累加发生在 A 从 smem 搬到寄存器、并转成 fp32 的那一刻，
不需要第二次遍历 A。这就是「prenorm」名字的由来——RMSNorm 的分母在 GEMM 里被顺带算出来了。

**3.4 TF32 WGMMA（`:205-231`）**

A 在寄存器（`a[]`，已 fp32）、B 在 smem（fp32，通过 `make_smem_desc` 直接当 TF32 读）。
WGMMA 选择器 `TF32MMASelector<WGMMA_N=BLOCK_N, true>`（`:157`），内层按「swizzle range 128B / WGMMA_K=8」
两层展开（`:214-226`）。**关键点：B 从 smem 到 WGMMA 没有任何显式转换代码**——B 的 fp32 低 13 位
是被硬件在 MMA 时按 TF32 处理的。到底是**截断**还是**就近舍入**？§5 用实测判定。

**3.5 归约与上屏（`:233-284`）**

- `sqr_sum`：warp 内 `warp_reduce_sum<4>`（`:233-234`）把每个线程的 2 个累加器在 4 线程组内归约，
  每 4 线程的第 0 号线程写回（`:236-242`）。写出位置 `sqr_sum[m_offset + m_idx]`。
- `D` 上屏：把 WGMMA 累加器按 64x8 D 布局经自定义 swizzle（`get_swizzled_bank_group_idx`，`:19-33`）
  写进 smem_cd（`:246-272`），`NamedBarrier::sync(128)` 后由 warp 0 发 `SM90_TMA_STORE_{2,3}D`
  （`:277-284`，2D=无 split / 3D=有 split）。

## 4. SM100 kernel 精读（仅分析，`sm100_tf32_hc_prenorm_gemm.cuh`）

SM100 版换了 MMA 形态，引入**第三个 warpgroup**：

- **UMMA 是 TS 形式**（A 在 TMEM、B 在 SMEM）：`cute::SM100_MMA_TF32_TS<...>`（`:169-173`），
  `UMMA_K = 32/sizeof(float) = 8`，逐 K 发 `umma_t::fma`（`:198-204`）。
- **A 必须先转成 fp32 放进 TMEM**：由 `cast-and-reduce` warpgroup（`:268-342`）完成——
  从 smem 用 `SM90_U32x4_LDSM_N` 取出 bf16（`:296-301`）、`__bfloat1622float2` 转 fp32、
  **同时** `__ffma2_rn(v, v, sum)` 累加平方（`:312-317`），再用 `SM100_TMEM_STORE_16dp256b1x`
  写进 TMEM（`:321-324`）。TMEM 里 A 占 `BLOCK_K * kNumCastStages` 列、D 占 `BLOCK_N` 列（`:75`）。
- **B 是 fp32、`kSwizzleBMode=128`**，MMA warp 用 `make_umma_desc` 逐 stage 推进描述符（`:176-202`）。
- 多出 `full_cast_barriers / empty_cast_barriers` 两级 mbarrier（`:97-116`）：TMA→cast→MMA 三段流水。
- 归约在 cast warpgroup 里完成（`:334-341`），与 SM90 语义一致，只是搬运路径改成了 TMEM。

一句话：**SM90 把 A 转 fp32 放寄存器给 WGMMA；SM100 把 A 转 fp32 放 TMEM 给 UMMA(TS)**，
但「平方和搭便车」的思想完全一致。

## 5. H100 实测（`harness/exp_hc_prenorm_gemm.py` + `.out.txt`）

环境：H100 80GB HBM3（sm90），deep_gemm 2.8.1，torch 参考**关闭 TF32**（真 fp32），保证参考干净。

### 5.1 数值：D 是 TF32、S 是精确平方和

对 A(bf16) 上转无损，D 的误差只来自 B（fp32）被压到 TF32。用两种 TF32 模拟对比
（`D.vs_X` = 归一化 Frobenius 误差 `‖Δ‖/‖ref‖`）：

| m | n | k | splits | D.vs_fp32 | D.vs_trunc | **D.vs_rne** |
|---|---|---|---|---|---|---|
| 4096 | 24 | 7168 | None | 2.079e-4 | 4.982e-4 | **1.665e-5** |
| 4096 | 24 | 7168 | 16 | 2.072e-4 | 5.074e-4 | **1.084e-6** |
| 8192 | 24 | 28672 | None | 2.188e-4 | 4.731e-4 | **6.702e-5** |
| 137 | 24 | 7680 | None | 2.039e-4 | 5.030e-4 | **1.781e-5** |

**判定：TF32 走的是「就近舍入（RNE）」，不是截断。** D 与 RNE 模拟的相对误差比截断小 **~30×**
（`rne/trunc ≈ 0.03`），且与真 fp32 参考的 2e-4 量级正是 10-bit 尾数（2⁻¹¹≈5e-4 的统计平均）应有的水平。
这是一个**硬结论**：尽管 kernel 代码里 B 从 smem 直送 WGMMA、没有任何 `cvt` 指令，
硬件在 TF32 MMA 时对 fp32 操作数做的是**round-to-nearest-even**。

`S` 与 fp64 精确平方和的 `max|Δ| = 5.9e-3`（行和量级 ~7e3，即 rel ≈ 8.3e-7）——因为 bf16 值的平方
在 fp32 中本身可精确表示，误差只来自 128 线程 × 2 累加器的求和顺序。

### 5.2 融合的收益：省掉一遍 A 的读

`m=8192, n=24, k=28672`（A 读一遍 = 469.8 MB）：

| 做法 | 时间 |
|---|---|
| **融合 kernel（D+S）** | **166.7 us** |
| torch 单独 `(a.float()**2).sum(-1)` | 4122.9 us |
| torch 真 fp32 matmul（`a.float() @ b.T`） | 3683.6 us |

融合 kernel 比「单独 matmul + 单独 sqr_sum」快 **~47×**。torch 侧慢是因为它把小算子拆成
multiple elementwise/reduce kernel、且物化了中间 fp32 张量；但即便按**理想的单遍 A 读取**估算，
独立 sqr_sum 也要再读 470 MB ≈ `470/2.85 ≈ 165 us`（按实测 2.85 TB/s），相当于**再花一个完整 kernel 的时间**。
融合把它压成 0（`S` 的平方累加藏在 A 的搬运里）。对训练里每层、每 token 都发生的量，这笔账很可观。

### 5.3 split-K：小 M 的并行度救星 + 可加性

`num_splits=16`：`sum_splits(D)` 与非 split 的 D 归一化误差 1.59e-5、`calc_diff` 1.26e-10；
`sum_splits(S)` 误差 2.88e-7（与单 kernel 的累加误差同量级）。16 个 split 的 D Frobenius 范数
在 6600±3% 内均匀（`k_blocks=112`，每 split 7 块，无 remainder）。

小 M 场景收益最大：`m=137, k=7680` 无 split 只有 **1.5 TFLOPS / 85 GB/s**（只有 3 个 M block、
并行度严重不足）；split-K=16 后变成 **5.4 TFLOPS / 328 GB/s**，时间 33.5 → 9.3 us。

### 5.4 参数契约（host 端断言真的会拒）

```
[拒绝] n=64 (>32)     : sm90_tf32_hc_prenorm_gemm.hpp:30   n <= 32 and n % 8 == 0
[拒绝] k=4104 非 64 倍: sm90_tf32_hc_prenorm_gemm.hpp:31   k % block_k == 0
[通过] n=24, k=4096   : 正常执行
```

### 5.5 性能与 roofline：这是一个**访存受限**算子

| m | n | k | splits | us | TFLOPS | GB/s | AI (F/B) |
|---|---|---|---|---|---|---|---|
| 4096 | 24 | 7168 | None | 33.2 | 42.5 | 1802.6 | 23.6 |
| 4096 | 24 | 7168 | 16 | 35.3 | 39.9 | 1867.1 | 21.4 |
| 8192 | 24 | 28672 | None | 166.2 | 67.8 | **2847.8** | 23.8 |
| 8192 | 24 | 28672 | 16 | 171.2 | 65.9 | 2837.1 | 23.2 |
| 137 | 24 | 7680 | None | 33.5 | 1.5 | 85.2 | 17.7 |
| 137 | 24 | 7680 | 16 | 9.3 | 5.4 | 327.6 | 16.5 |
| 4096 | 32 | 7168 | None | 33.3 | 56.4 | 1804.8 | 31.2 |

算术强度 **AI ≈ 17–31 FLOP/B**，远低于 H100 TF32 的 roofline 拐点（TF32 峰值/带宽 ≈ 数百）。
因此性能由有效带宽决定：大 M 下达到 **2848 GB/s ≈ H100 HBM 峰值（~3.35 TB/s）的 85%**，
此时 TFLOPS 被 `AI × 带宽` 封顶。这也解释了**为什么这个 kernel 要拼命省字节**——它天生就是
带宽受限的，融合 `sqr_sum`（省一遍 A）比优化 MMA 更值钱；而 A 是 bf16（不是 fp32）也正是在省字节。

## 6. 与 Mega mHC 的关系、小结

- `tf32_hc_prenorm_gemm` 是 HyperConnection 里「mix 投影 + RMSNorm 平方和」的**独立算子形态**；
  `sm100_mega_mhc.cuh` 把它的两件事进一步**融进 mega-kernel 的 TF32 TS UMMA**（第 13 篇），
  独立版则用于非融合路径 / 需要单独 D、S 时。
- **三个可复用的结论**：
  1. **融合的支点是 A 的搬运**：`S` 在 A 的 bf16→fp32 转换处顺手累加，零额外带宽。这类「GEMM 顺带
     reduce」的模式在带宽受限的小 N 场景收益最大。
  2. **TF32 MMA 对 fp32 操作数是 RNE（实测 30× 优于截断）**：所以只要 A 是 bf16（无损）、B 保 fp32，
     这个 kernel 的精度损失就完全落在 TF32 的 10-bit 尾数上，可预期、可建模。
  3. **split-K 是「小 N、小 M、大 K」形状的并行度补丁**：D/S 同时分段、host 端求和，代价是 fp32
     累加顺序变化带来的 ~1e-6 级噪声，换来小 batch 下 3–20× 的吞吐。

**本篇实测脚本**：`harness/exp_hc_prenorm_gemm.py` / `.sh` / `.out.txt`（H100 真机）。
**SM100 分支**（`sm100_tf32_hc_prenorm_gemm.cuh`）为**仅分析**——本机 H100 不可运行。
