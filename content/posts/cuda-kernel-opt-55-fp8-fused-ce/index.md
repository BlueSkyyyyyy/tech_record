---
title: "CUDA 算子调优（五十五）：FP8 融合 cross-entropy —— 当 LM head 权重减半，那个卡在 L2 上的 GEMM 会变快吗"
date: 2026-09-22T05:00:00+08:00
draft: false
weight: 55
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "cross-entropy", "LM head", "FP8", "e4m3", "ue8m0", "DeepSeek-V4", "TMA", "wgmma", "logsumexp", "L2", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

[第 54 篇]({{< relref "cuda-kernel-opt-54-fused-cross-entropy" >}})把融合 cross-entropy（LM head）
的账算得很清楚：在 $M=8192$ 这种 compute-bound 形状上，省掉 $[M,V]$ 的 logits 只值 ~2.7 GB 流量，
而 GEMM 自己要 20 ms——**融合 CE 是显存优化，不是吞吐优化**。最后留了一个尾巴：

> 下一个能真正提速的方向：把权重换成 **FP8 wgmma**，权重字节减半、L2 流量减 17%，
> 同时张量核吞吐上限翻倍。

这一篇就把它做出来。有意思的是，它顺带回答了一个真实的模型问题：DeepSeek-V4 整网的线性层都是
FP8（`/ssd/models/DeepSeek-V4-Pro/config.json` 的 `quantization_config`：`e4m3 + ue8m0 + weight_block 128×128`），
**唯独 lm_head 在 checkpoint 里是 bf16**——因为 logits 直接喂 CE，对精度最敏感
（`inference/model.py:712` 的注释：「lm_head in the checkpoint is stored in bf16, while the parameter
here is stored in fp32 for easier computation of logits later」）。所以本篇也是在做一次 what-if：
**如果 lm_head 也走 FP8，能买回多少时间、又要付出多少精度？**

结论先放这：

| 方案（$M=8192,H=7168,V=129280$） | 时间 | 吞吐 | vs bf16 基线 |
|---|---|---|---|
| cuBLAS bf16 GEMM + CE | 24.67 ms | 742.5 TFLOPS | 1.00× |
| 融合 CE，bf16（54 篇复现） | 25.01 ms | 607.1 TFLOPS | 0.99× |
| **融合 CE，FP8** | **12.82 ms** | **1184.6 TFLOPS** | **1.93×** |
| cuBLAS FP8 GEMM + CE（torch `_scaled_mm`+CE） | 13.80 ms | 1268.1 TFLOPS | 1.79× |

FP8 融合 CE 达到 **1184.6 TFLOPS / 59.9% FP8 峰值**，是**同进程 bf16 融合版的 1.95×**、
**cuBLAS FP8 GEMM 单独跑的 94%**（我们还顺手把 CE 做了），比 `cuBLAS FP8 + CE` 快 **1.08×**。
代价：logits 的 per-tensor e4m3 量化误差 **3.7% RMS**，但 **CE loss 只动了 1e-5**。

---

## 一、形状与量化方案

shape 和 54 篇完全一致，取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：
`hidden_size=7168`、`vocab_size=129280`，测 $M=2048/8192/32768$。

复用的量化方案（尽量贴 V4 的语义，同时用一个便宜的 host 折算）：

| 张量 | 格式 | 粒度 | 说明 |
|---|---|---|---|
| 激活 $X[M,K]$ | e4m3 | **per-row**（$1\times K$） | 每行取 absmax，scale 取 **2 的幂**（`pow2_scale`），折回时精确 |
| 权重 $W[V,K]$ | e4m3 | **per-tensor** | $W_q=W/s_w$，`wscale=s_w` 在 epilogue 乘回 |

两点说明：

- **为什么 scale 取 2 的幂**：这样「除以 scale」只是 e4m3 的指数平移，折回时无舍入损失——这正是
  37 篇 [`ue8m0` 折进操作数]({{< relref "cuda-kernel-opt-37-fp8-pb-prescale" >}})的结论。
  一个 per-row 的 2 的幂 scale 折回时只是 fp32 乘一次，精确。
- **per-block 可以免费升级**：V4 的 `128×128` weight block 同样是 2 的幂，可以在 host 端
  `W_folded = q_w * s_block` 折进 e4m3（精确，因为 $q_w\cdot 2^e$ 只是指数平移），
  于是**分组 per-block 的 GEMM 数值等价于一个 per-tensor GEMM**，kernel 一个字都不用改。
  本篇为了 host 简单直接用了「权重 per-tensor + 激活 per-row」；per-block 只改 host 的量化器，
  吞吐数字**原样适用**（见 37 篇实测）。

真实 checkpoint 的权重有长尾 outlier，per-block 会比 per-tensor 稳；但本机只有合成数据
（`fill_fast` 产生 $[-0.025,0.025]$ 的均匀权重），没有 outlier 结构可展示，所以下面的精度数字
**对 per-block 而言是保守的**——这点在「精度」一节再展开。

---

## 二、一个 kernel 同时跑 bf16 和 FP8

54 篇的骨架（TMA + mbarrier + warp specialization + `wgmma` SS SW128 + split-N + tile 级
online logsumexp）几乎原样可用，只需要把 GEMM 的「数据类型」参数化：

```cpp
template <bool FP8, int BM, int BN, int BK, int STAGES, bool FUSE, int GM=0, int GN=0>
__global__ void fce_kernel(...) {
  static_assert(FP8 ? (BK == 128) : (BK == 64), "SW128 内维固定 128 字节");
  ...
  for (int s = 0; s < 4; ++s) {                    // 每个 128B atom 里都是 4 步 ×32B
    uint64_t da = make_desc_sw128(k_addr(smem_u32(a), s), SBO);
    uint64_t db = make_desc_sw128(k_addr(smem_u32(bj), s), SBO);
    if constexpr (FP8) wgmma_m64n128k32_fp8(acc[jn], da, db);
    else               wgmma_m64n128k16    (acc[jn], da, db);
  }
}
```

几个关键点，全部来自前作：

1. **SW128 的 128B atom 对 bf16 和 fp8 逐字节同构**（22/23 篇）：bf16 一行是 64 个元素
   （128 B），fp8 一行是 128 个元素（128 B）；`wgmma` 描述符、`make_desc_sw128`、
   TMA 的 `CU_TENSOR_MAP_SWIZZLE_128B` 完全一样，连 k-step 地址函数都一样
   （`k_addr`：atom 内每步 +32 B、跨 atom +1024 B）。唯一区别是 k 指令：bf16 是 `k16`、fp8 是 `k32`。
2. **TMA 的 tensor map 只差 `rowbytes`**：bf16 是 `K*2=14336`，fp8 是 `K=7168`；
   数据按 `UINT8` 搬字节（驱动枚举没有 fp8，见 23 篇）。box 内维恒 128 B，所以
   bf16 的 `BK=64`、fp8 的 `BK=128`，**每级 stage 的 smem 字节数一样**
   （$(B_M+B_N)\times128$），扫参的 stage 数约束也不变。
3. **epilogue 里的 scale 折算**（fp8 才做）：per-row 激活 scale + per-tensor 权重 scale

```cpp
// fused_ce_fp8.cu —— wgmma_wait0 之后，把真实 logits 还原
if constexpr (FP8) {
  const float s0 = (r0 < M) ? wscale * rowscale[r0] : 1.f;   // 行 r0
  const float s1 = (r1 < M) ? wscale * rowscale[r1] : 1.f;   // 行 r0+8
  for (int jn=0; jn<NSPLIT; ++jn)
    for (int j=0; j<16; ++j) {
      acc[jn][j*4+0] *= s0; acc[jn][j*4+1] *= s0;
      acc[jn][j*4+2] *= s1; acc[jn][j*4+3] *= s1;
    }
}
```

   这次乘法把「同一行的 4 个累加器」乘同一个 scale——因为 wgmma 的 `m16n8` 布局里
   `acc[j*4+0/1]` 同行（`row0`）、`acc[j*4+2/3]` 同行（`row0+8`），每行只有一次乘法，
   **128 个乘加，纯噪声**。折算完成后再走和 54 篇一字不差的 online logsumexp。

4. **superblock swizzle** 沿用 54 篇的 1D 严格双射网格（`sw 4×8` / `sw 8×16`），
   面板 $(G_M B_M+G_N B_N)K\cdot\text{sizeof}$；fp8 下 sizeof 减半，同样的 `G_M,G_N` 面板**小一半**，
   L2 更舒服。

---

## 三、实测

环境：H100 SXM（132 SM，3.35 TB/s，BF16 TC 989 TFLOPS，**FP8 TC 1978 TFLOPS**），
`kernel_lab` 容器。`fused_ce_fp8.cu` 内 `bench_ms(warmup=5, iters=20)`；
bf16/fp8 两条路径在**同一个进程、同一份数据**上跑，避免跨进程降频。

### 3.1 $M=8192$ 主扫参

`plain` = 只做 GEMM 写 bf16 logits（测裸吞吐）；`fused` = 融合 online logsumexp、只写 partial。

| 配置 | plain bf16 | plain fp8 | fused bf16 | **fused fp8** | fp8 峰值% |
|---|---|---|---|---|---|
| `128×128 s4` | 330.9 | 644.4 | 342.3 | 681.6 | 34.5% |
| `128×256 s3` | 350.6 | 687.5 | 365.0 | 733.3 | 37.1% |
| `256×128 s4`（朴素） | 515.7 | 950.1 | 543.0 | 1019.0 | 51.5% |
| `sw 2×8` | 584.9 | 1064.6 | 585.2 | 1099.0 | 55.6% |
| `sw 4×8` | 614.6 | 1136.7 | 607.8 | 1164.0 | 58.8% |
| `sw 4×16` | 613.8 | 1142.7 | 604.4 | 1168.8 | 59.1% |
| **`sw 8×16`** | 615.0 | 1147.8 | 607.1 | **1184.6** | **59.9%** |
| `sw 4×8 s3` | 563.4 | 1066.4 | 578.5 | 1110.9 | 56.2% |
| `128×256 s3 sw` | 558.9 | 1057.4 | 590.3 | 1130.0 | 57.1% |
| `256×256 s2` | 73.6 | 147.9 | 75.8 | 150.2 | 7.6% |

（单位 TFLOPS；`fused bf16` 最好 607.1，和 54 篇的 605.7 一致。）

三个观察：

- **FP8 把峰值翻倍，但绝对效率没有跳变**：bf16 最好 607.1 / 989 = 61.4%，fp8 最好 1184.6 / 1978 = 59.9%。
  两者都卡在同一个地方（下面 ncu 会说是 L2 + 张量核），所以**时间近似对半砍**：
  $25.01\to12.82$ ms，**1.95×**。
- **superblock 依然值钱**：朴素 `256×128`（51.5%）→ `sw 8×16`（59.9%），+16%。
- **`256×256` 依旧崩**（7.6%），和 54 篇一样是寄存器墙（ptxas `C7511`）。

### 3.2 跨 $M$ 的 scaling

| $M$ | cuBLAS bf16+CE | fused bf16 | **fused fp8** | fp8/bf16 | fp8/baseline |
|---|---|---|---|---|---|
| 2048 | 5.74 ms | 6.14 ms | **3.28 ms** | 1.87× | **1.75×** |
| 8192 | 24.67 ms | 25.01 ms | **12.82 ms** | 1.95× | **1.93×** |
| 32768 | 93.12 ms | 101.6 ms | **51.53 ms** | 1.97× | **1.81×** |

FP8 的收益全程稳定在 ~1.9×，$M$ 越大越充分。

### 3.3 对标 cuBLAS / torch

用 `torch._scaled_mm`（底层就是 cuBLAS FP8，per-tensor）做同 shape 的 FP8 GEMM 参考：

| $M=8192$ | GEMM | CE | 端到端 |
|---|---|---|---|
| torch bf16 `x@W.t()` + CE | 20.56 ms / 738.6 TFLOPS | 2.71 ms | 23.27 ms |
| torch **FP8** `_scaled_mm` + CE | 11.97 ms / **1268.1 TFLOPS** | 2.75 ms | 13.80 ms |
| **本篇融合 FP8** | — | — | **12.82 ms** |

- 我们的融合 FP8（12.82 ms）**比 cuBLAS FP8 GEMM 单独跑（11.97 ms）只慢 7%**，
  而这 7% 里还包含了 online logsumexp + combine——**epilogue 是免费的**。
- 对比完整的 `cuBLAS FP8 + CE`（13.80 ms），融合版快 **1.08×**，并省掉 2.12 GB 的 logits 往返。
- 对比 54 篇的 `cuBLAS bf16 + CE`（24.67 ms），快 **1.93×**。

一句话：54 篇的结论「融合 CE 是显存优化不是吞吐优化」在 **bf16** 下成立；
一旦权重是 **FP8**，LM head 的瓶颈从「DRAM + 张量核」变成「L2 + 张量核」，
张量核峰值翻倍直接把时间砍半——**这次融合顺带也赚到了吞吐**。

---

## 四、精度：logits 差 3.7%，loss 差 1e-5

`acc` 模式把 logits 物化回 bf16，逐元素对拍（$M=8192$）：

| 对照 | rel-RMS | max-abs |
|---|---|---|
| 我们的 bf16 kernel vs cuBLAS bf16 | **0**（逐位一致） | 0 |
| 我们的 FP8 vs 我们的 bf16 | **3.735%** | 3.91e-3 |

- 权重 per-tensor e4m3 本身的 rel-RMS 误差是 **2.56%**（对均匀权重、相对 std 0.01443），
  这是 e4m3 的固有属性（3 位尾数 ≈ 6% 相对精度），不是实现 bug。
- **logits 差 3.7%，但 CE loss 只差 1e-5**：$M=8192$ 时 bf16 fusion loss `11.769796` vs
  fp8 fusion `11.769786`（Δ=1.0e-5）；$M=2048/32768$ 时 Δ 都是 1e-6。
  logsumexp 是 129280 个 logit 的加权平均，逐元素的无偏舍入误差在求和中相互抵消，
  所以对 loss 的影响远小于单点误差。

**必须诚实的地方**：上面的权重是合成均匀分布，没有真实训练权重的长尾 outlier。
V4 之所以把 lm_head 留在 bf16，正是因为真实权重里的大 outlier 会让 per-tensor e4m3 明显变差。
如果切到 V4 的 **per-128×128 block**，outlier 被局部 scale 吸收，精度会好很多——
而且按 37 篇的折 scale 结论，**per-block 折进操作数后 GEMM 数值等价于 per-tensor GEMM**，
本篇的吞吐数字（1184.6 TFLOPS）**原样适用**，只需要换 host 端量化器。所以这篇文章给出的
是一个**吞吐上界 + 精度下界**：真实 lm_head 的 FP8 收益只会比这里更大（权重字节一样减半、
per-block 精度更高），代价由 per-block 量化把 outlier 关在 128 宽的笼子里。

---

## 五、ncu：FP8 之后墙在哪里

同进程同 shape（$M=8192$）的 `--set full` 对照：

| 指标 | bf16 `sw 4×8`（fused） | **FP8 `sw 8×16`（fused）** |
|---|---|---|
| ncu Duration | 22.05 ms | **11.35 ms** |
| **L2 Cache Throughput** | **82.9%** | **79.0%** |
| **Compute (SM)**（tensor 主导） | **78.8%** | **73.4%** |
| DRAM Throughput | 24.0%（805 GB/s） | **11.5%（387 GB/s）** |
| L1/TEX | 63.0% | 58.3% |
| Registers / thread | 90 | 90 |
| Achieved Occupancy | 26.4% | 26.3% |
| Block Limit | Reg / Smem = 1 | Reg / Smem = 1 |

机制很清楚：

- **fp8 下 DRAM 不再是问题**（11.5%）：权重字节减半、面板减半，L2 命中更好，
  ncu 口径的 DRAM 字节大约只有 bf16 的 1/4（805→387 GB/s，而时间也减半）。
- **新瓶颈是 L2（79%）与张量核（73%）并驾齐驱**。FP8 让张量核每个周期干两倍活，
  于是「喂数」这条线（L2→SM 的 wgmma 操作数读取）第一次和张量核一样高。
  这也解释了为什么绝对效率停在 60%：**两个 80% 的天花板叠在一起**。
- **occupancy 26%、1 CTA/SM** 不变：`acc[64]` + 197 KB smem 仍然锁死单 CTA。
  这是 Hopper 单 CTA 几何的老问题（24/54 篇），与 fp8 无关。

> 口径提示：ncu 单次 duration（11.35）比 event 稳态（12.82）短 ~12%，
> bf16 也一样（22.05 vs 25.01）。**相对结论用同一口径**（ncu 下 fp8/bf16 = 1.94×）。

---

## 六、负结果与坑

- **`256×256` tile 还是崩**（fp8 7.6%）：`acc[2][64]=128` 寄存器 + wgmma 双累加器
  触发 ptxas `C7511 ... serialized due to insufficient register resources`。54 篇的约束在 fp8 下原样存在。
- **stage 数被 smem 锁死**：$(B_M+B_N)\times128\times S$，`256×128` 下 $S=5$ 就要
  245 KB > 232 KB 上限，`cudaFuncSetAttribute` 失败、kernel 不启动——扫参里那几个「几百万 TFLOPS」
  的假数字就是它。**写扫参时务必检查 `cudaFuncSetAttribute` 的返回值**，否则会得到 $10^7$ TFLOPS 的鬼数据。
- **`float(fp8)` 会返回原始位模式**（32 篇坑的复现）：host 端量化一定要用
  `__nv_fp8_e4m3(x)` 构造 / `(float)q` 取值，别拿 `.__x` 当数值用。
- **最初漏了 per-row 激活 scale**：第一版只折了权重 scale，logits 直接差到 rel-RMS `1.6e4`。
  per-row scale 必须在 epilogue 乘回（或折进存进去的 e4m3 值里），漏了就是数量级错误。

---

## 七、小结

- 把 54 篇的融合 CE 换成 **FP8 wgmma**（`m64n128k32.e4m3`，`BK=128`，TMA SW128），
  骨架、epilogue、superblock swizzle 全部复用，只多了一段 per-row/per-tensor scale 折算。
- 结果：**12.82 ms / 1184.6 TFLOPS（59.9% FP8 峰值）**，
  **1.95× bf16 融合版**、**1.93× cuBLAS bf16+CE**、**1.08× cuBLAS FP8+CE**，
  只比 cuBLAS FP8 GEMM 单独跑慢 7%（且顺手做了 CE）。logits 显存仍是 **66 MB 部分和**。
- **FP8 改变了瓶颈**：DRAM 从 24% 掉到 11.5%，L2（79%）与张量核（73.4%）成为双瓶颈；
  bf16 下「融合不赚吞吐」的判决，在 FP8 下因为张量核峰值翻倍而反转。
- **精度**：logits per-tensor e4m3 rel-RMS 3.7%，CE loss 只动 **1e-5**；
  真实 checkpoint 用 per-128×128 block（37 篇证明可无损折进操作数、吞吐不变）会更稳。
- 下一步候选：把 **per-block 量化器**真正接到 host 上（贴 V4 语义、在真实 outlier 权重上量精度）；
  或者用 **cluster multicast** 攻一攻那 79% 的 L2；又或者把同样的 FP8 融合头用到
  **MoE 的 down/proj**（那里权重也是 FP8、也是「大 N」的形状）。
- 剩下没啃的硬骨头：LM head 已经是 compute/L2 双限，想再往上要么 Blackwell `tcgen05`
  的 TMEM 累加器解锁几何（occupancy 26%），要么 2 CTA/SM 的几何突破。

---

## 复现

```bash
cd code/kernel-opt
# 主实验（bf16 vs fp8，同进程扫参）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -lcublas" \
  scripts/run.sh 55-fp8-fused-ce/fused_ce_fp8.cu 8192 all
# 精度对拍（物化 logits，bf16 vs fp8）
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -lcublas" \
  scripts/run.sh 55-fp8-fused-ce/fused_ce_fp8.cu 8192 acc
# ncu
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda -lcublas" \
  scripts/ncu.sh 55-fp8-fused-ce/fused_ce_fp8.cu --set full --kernel-name regex:fce_kernel -- 8192 prof8
# cuBLAS/torch FP8 参考
python3 55-fp8-fused-ce/torch_fp8_ref.py 8192
```

代码：`code/kernel-opt/55-fp8-fused-ce/fused_ce_fp8.cu`（融合 kernel `fce_kernel`、
FP8 wgmma `wgmma_m64n128k32_fp8`、host 量化 `quant_A_perrow` / `quant_W_tensor`、
`combine_kernel`）；实测 `sweep_M2048/8192/32768.out.txt`、`acc_M8192.out.txt`、
`torch_fp8_ref.out.txt`；ncu `ncu_fp8_sw8x16.out.txt`、`ncu_bf16_sw4x8.out.txt`。
