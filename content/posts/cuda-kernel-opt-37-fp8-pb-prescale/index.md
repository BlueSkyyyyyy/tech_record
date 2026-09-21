---
title: "CUDA 算子调优（三十七）：FP8 per-block GEMM 的第三条路——把 ue8m0 的 2 的幂 scale 折进操作数"
date: 2026-09-21
draft: false
weight: 37
series: ["kernel-opt"]
tags: ["CUDA", "GPU", "算子优化", "DeepSeek", "DeepSeek-V4", "FP8", "e4m3", "ue8m0", "per-block", "GEMM", "wgmma", "TMA", "setmaxnreg", "量化", "H100", "Hopper", "系列"]
categories: ["算子开发"]
---

这是「FP8 GEMM」子系列的第四篇。前情：

- [第 22 篇]({{< relref "cuda-kernel-opt-22-fp8-gemm" >}})：e4m3 的 `mma.sync` / `wgmma`，
  per-tensor 打满到 768 TFLOPS；
- [第 23 篇]({{< relref "cuda-kernel-opt-23-fp8-gemm-tma" >}})：TMA + mbarrier + warp specialization，
  per-tensor **1217 TFLOPS（cuBLAS 的 88%）**；
- [第 24 篇]({{< relref "cuda-kernel-opt-24-fp8-gemm-pb" >}})：per-block 缩放的**两堵墙**（寄存器 + ptxas 序列化），
  同二进制对照下 per-block 只有 **936.5**，比 per-tensor 的 **1209.6** 慢 23%。

这 23% 的差距在 31 / 34 / 36 篇里反复出现（MoE grouped、fused MoE、compressor 投影），是 DeepSeek-V4
`e4m3 + ue8m0 + weight_block 128×128` 这条路上的**公共剩余瓶颈**。24 篇给的出路是「DeepGEMM 式的
1-warpgroup / `setmaxnreg 248`」，第六部分也一直挂着这个 backlog。

这篇把这条路走了一遍，并给出**第三条路**。先给结论：

| 路径 | GEMM | 相对 fold |
|---|---|---|
| per-block fold（每 128-k 块 `wait0` 后折算）| **940 TFLOPS** | 1.00× |
| 折进操作数后跑 per-tensor GEMM | **1208 TFLOPS** | **1.28×** |
| 端到端（含上游 per-1×128 动态量化）| **0.2194 ms** vs 0.2617 ms | **1.19×** |

核心一句话：**DeepSeek-V4 的 `scale_fmt=ue8m0` 是 2 的幂，`A_fp8 × sa` 在 e4m3 里是精确的**——所以
可以把这个 scale 直接折进操作数，把 per-block GEMM 变成 per-tensor GEMM，**折算开销彻底消失**。
而「折」这一步可以融进上游的量化 kernel，等于白送。

---

## 一、问题：per-block 到底慢在哪

真实 shape 取自 `/ssd/models/DeepSeek-V4-Pro/config.json`：

```
hidden = 7168, moe_inter = 3072, 384 routed experts, top-6
量化: fmt=e4m3, scale_fmt=ue8m0, weight_block=[128,128], activation=dynamic 1x128
```

测 `M=4096, N=3072, K=7168`（180.39 GFLOP，AI≈1785，算力受限），FP8 峰值按 1978 TFLOPS。

per-block 的数学是（`kb` = 第几个 128-k 块）：

$$
C[m,n] \;=\; \sum_{kb} \; sa[m,kb] \cdot sb[\,n/128,\,kb] \cdot \bigl(A_{kb} B_{kb}\bigr)[m,n]
$$

`sa` 逐行逐块（1×128，动态激活），`sb` 每个 128×128 权重块一个标量。24 篇的实现是：

```
for kb in 0..K/128-1:
    wait full[kb]                      # TMA 数据到
    issue 4x wgmma.m64n128k32          # 128 个 k 的 4 条 wgmma
    wgmma.wait_group 0                 # ← 排空张量管线
    fin += sa[m,kb]*sb[n,kb] * acc     # ← 64 条 fp32 FMA / 线程
```

即**每 128-k 块都要等 wgmma 全部完成后，用 CUDA core 做 64 条 FMA**。这不是「折算 FLOPs 太多」，
而是**折算时张量核在空转**。

### 算一笔 roofline

单个 CTA（BM=128, BN=128, BK=128）每个 k 块：

- wgmma 工作量：$4 \times 64 \times 128 \times 32 = 1.05\text{M MAC} = 2.10\text{M FLOP}$；
  FP8 张量核 1978 TFLOPS → **2.12 ns**。
- fold 工作量：`64 FMA × 256 线程 = 16384 FMA`；FP32 CUDA core 66.9 TFLOPS → **0.49 ns**。

两者串行时，**fold 占 0.49 / (2.12+0.49) = 19%**——和实测的 23% 差距同量级。理论上只要把 fold 与
wgmma 重叠，就能把这 19% 拿回来。

### ncu 实测

| 指标 | fold（128×128 s4）| per-tensor（256×128 s4）|
|---|---|---|
| tensor pipe 活跃度 | **49.8%** | **67.9%** |
| `sm__throughput` | 49.8% | 67.9% |
| L2 | 65.9% | 57.4% |
| DRAM | 30.4% | 26.9% |
| occupancy | 13.7% | 25.4% |
| stall `barrier` | **1.09** | 0.18 |
| stall `long_scoreboard` | 1.19 | 8.71 |
| 耗时 | 199.5 µs | 149.4 µs |

`barrier` 从 0.18 涨到 1.09，几乎是 fold 版的主 stall 之一——就是每块那个 `wgmma.wait_group 0`。

---

## 二、先试两条「把 fold 藏起来」的路（都失败）

### 路 A：ping-pong 两个累加器，让 fold(kb-1) 与 wgmma(kb) 重叠

思路很直接：准备两个累加器 `acc[0]/acc[1]`，交错使用；`issue(kb)` 后只 `wgmma.wait_group 1`
（只等上一组），于是当前组在飞的同时，CPU core 去折算上一组。

但 ptxas 会主动把 wgmma 串行化，报：

```
(C7514) wgmma.mma_async instructions are serialized due to non wgmma instructions
        reading accumulator registers of a wgmma ...
```

原因是**我们每块用通用 FMA 写 `acc=0` 清零**。DeepGEMM 的解法很巧——它的 `WGMMA::wgmma(desc_a, desc_b, accum, k)`
把循环下标 `k` 当成了 `scale_d`：`k=0` 时 `ScaleOut::Zero`（**由 wgmma 指令自己清零累加器**），
`k>0` 时累加。我们照做（`fp8_gemm_pb2.cu` 里的 `wgmma_m64n128k32(..., scale_d)`，`s==0` 时传 0），
再加上 CUTLASS / DeepGEMM 的 `warpgroup_fence_operand`（空 asm + `"+f"`）：

```cpp
__device__ __forceinline__ void fence_operand(float& r) { asm volatile("" : "+f"(r) :: "memory"); }
// ...
wgmma_m64n128k32(acc[SET][jn], da, db, !(zero_first && s == 0));   // s==0 用指令自身清零
```

**C7514 确实消失了**（对单累加器路径），但 ping-pong 依然慢：

| 配置 | TFLOPS |
|---|---|
| 单累加器 + 每块 wait0（`MODE 0`）| 902–940 |
| ping-pong（`MODE 1`）| **348** |

### 路 B：用 `setmaxnreg` 给 tensor warpgroup 发更多寄存器

ping-pong 需要 `acc[2][64] + fin[64] = 192` 个 fp32/线程，288 线程的静态上限只有 227，ptxas 会 spill
或串行化。DeepGEMM 用 `setmaxnreg`（TMA warp 让出到 40、math warpgroup 吃到 232）绕开静态上限。
我们用内联 PTX 试了：

```cpp
template <int N> __device__ void reg_alloc()   { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N)); }
template <int N> __device__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N)); }
```

结果 ptxas 直接回：

```
(C7507) 'setmaxnreg' ignored to maintain minimum register requirements.
```

**`setmaxnreg` 被忽略了**，ping-pong 仍然 348。结论：在 Hopper 的 288 线程几何下，**双累加器 + fold 的
寄存器需求过不了 ptxas 这一关**；DeepGEMM 能在 `BLOCK_M=128` 上做到，靠的是它 128 线程的 TMA warpgroup +
不同的调度器，不是一条指令的事。24 篇记的「1-warpgroup/248-reg」确实是硬约束，但重置整条流水线的
风险/收益比并不好。

---

## 三、第三条路：把 scale 折进操作数

换一个视角。per-block 缩放里，`sb` 是**权重**的（静态），`sa` 是**激活**的（动态）。两者都是
`ue8m0` 格式——而 `ue8m0` 的含义是「**无符号 8 位指数**」，也就是**纯 2 的幂**。

> 验证：`/ssd/models/DeepSeek-V4-Pro/config.json` → `"scale_fmt": "ue8m0"`。

于是有一个精确的恒等式。设量化是 $q_a = \mathrm{round_{e4m3}}(A / sa)$，则

$$
A_{kb} B_{kb} \cdot sa \cdot sb
\;=\; \underbrace{(q_a \cdot sa)}_{\text{仍是 e4m3}} \cdot \underbrace{(q_b \cdot sb)}_{\text{仍是 e4m3}}
$$

因为 `sa = 2^e` 是 2 的幂，`q_a · 2^e` 只是把 e4m3 的**指数域**平移 `e`，**尾数一位不丢**，在范围内精确可表示
（e4m3 无 inf，溢出用 `__NV_SATFINITE` 饱和）。

```
   per-block（fold）：
     A_fp8  ──┐
              ├─ wgmma ── acc ──(wait0)── fin += sa*sb*acc   ← 折算在 GEMM 里
     B_fp8  ──┘

   折进操作数（prescale）：
     A_fp8 ─×sa→ A' ┐
                    ├─ wgmma （纯 per-tensor，无折算）
     B_fp8 ─×sb→ B' ┘
```

**折 B 是一次性的**（权重静态，加一次就永久免费）；**折 A 是每 forward 一次**。但注意：**折 A 可以融进
上游那个 per-1×128 动态量化 kernel**——量化器本来就要读 fp32 `X`、算出 `sa`、把 `X/sa` 量化成 fp8 写出；
它只要在写回前多乘一次 `sa`，就是把结果写成 `A'`。**寄存器里多一条乘法，零额外访存**。

实现（`fp8_gemm_prescale.cu`，逐字节精确）：

```cpp
__device__ __forceinline__ uint8_t scale_fp8(uint8_t b, float s) {   // s 是 2 的幂
  const float x = __half2float(__nv_cvt_fp8_to_halfraw(b, __NV_E4M3));
  return (uint8_t)__nv_cvt_float_to_fp8(x * s, __NV_SATFINITE, __NV_E4M3);
}
```

（这里特意走 `__nv_cvt_*` 而不是 `float(fp8)`——后者在这个工具链下返回**原始位模式**，是 32 篇踩过的坑。）

于是 GEMM 端直接复用 23 篇的 **per-tensor** 路径（`MODE=3`：累积整个 K、`wait_group<STAGES-2>` 允许 wgmma
跨块流水、无折叠），拿到 per-tensor 的全部性能。

---

## 四、实测

`M=4096, N=3072, K=7168`，`FP8_PEAK=1978`。`prescale_a` 单独计时、`prescale_b` 在计时外（权重静态）。

### GEMM 本体（scale 已折好）

| 实现 | 配置 | ms | TFLOPS | 峰值占比 |
|---|---|---|---|---|
| fold | 128×128 s4 | 0.1918 | **940.6** | 47.6% |
| fold | 128×128 s3 | 0.1976 | 912.7 | 46.1% |
| per-tensor（折后）| 256×128 s4 | 0.1493 | **1207.9** | 61.1% |
| per-tensor（折后）| 128×256 s4 | 0.1519 | 1187.7 | 60.0% |
| per-tensor（折后）| 128×128 s3 | 0.1650 | 1093.0 | 55.3% |
| per-tensor（折后）| 256×256 s3 | 1.2788 | 141.1 | 7.1%（寄存器墙）|

**GEMM 本体 940 → 1208 TFLOPS（1.28×）**，和 per-tensor 的历史最好成绩（24 篇 1209.6）一致。

### 端到端（含一步 per-1×128 动态量化）

Path A = 量化（写 `A_fp8` + `sa`）+ fold GEMM；Path B = 量化（把 scale 折进 `A'`）+ per-tensor GEMM。
两个量化 kernel 的工作量**完全相同**（都读 fp32 `X`、写 fp8），差别只在 GEMM。

| M | Path A（fold）| Path B（prescale）| 加速 |
|---|---|---|---|
| 4096 | 0.2617 ms | **0.2194 ms** | **1.193×** |
| 16384 | 1.0031 ms | **0.8446 ms** | **1.188×** |

正确性：

```
M=4096  : Path B vs Path A 相对误差 = 1.515e-03   (vs 原始 fp32 X 参考 1.62e-02，即量化误差)
M=16384 : Path B vs Path A 相对误差 = 1.336e-03   (vs 原始 fp32 X 参考 1.94e-02)
```

Path B 与 Path A **只差 0.15%**（就是 fp32 累加顺序的差异）——**折进操作数在数值上等价于 per-block fold**，
不是近似。

### M=16384 的稳定性

| 实现 | ms | TFLOPS |
|---|---|---|
| fold 128×128 s4 | 0.7551 | 955.5 |
| per-tensor（折后）256×128 s4 | 0.5947 | **1213.3** |
| prescale_a + per-tensor 256×128 s4 | 0.6775 | 1065.0 |

GEMM 本体 1.27×，端到端 1.19×，与 M=4096 一致。

---

## 五、边界与泛化

**只在 scale 是 2 的幂时精确。** 若 `sa` 是任意 fp32，`q_a · sa` 需要再次舍入到 e4m3，会多引入
~半个 ulp 的误差。DeepSeek-V4 用 `ue8m0` 正是为了「1 字节存 scale」，恰好也送了这个精确性。
对其他模型（如 GLM/Kimi 若用 fp32 scale），折叠是**近似**的，收益仍在但要额外测数值。

**两处工程约束**：

1. `A'` 与「本 GEMM 的 scale」绑定。如果同一份 `A` 要喂给两个 scale 不同的 GEMM，得各存一份 `A'`
   （或退回 fold）。实际 DeepSeek 的 FFN 里 `A` 只喂一个 GEMM，不冲突。
2. 折 A 的收益以「量化器与 GEMM 同属一个算子链」为前提。若要额外跑一遍独立的 `prescale_a`，
   M=4096 时它值 ~23 µs（读 29MB + 写 29MB），端到端仍是 1.11×，但不如融合划算。
   所以**优先把它融进上游量化**。

**能复用到哪里**：31 篇（MoE grouped per-block）、34 篇（fused MoE FP8）、36 篇（compressor 投影 FP8）
都是同一个 `e4m3+ue8m0+128×128` 的 fold 瓶颈——把 A/B 折好之后，这些 kernel 都能直接切到 per-tensor
路径，白拿 ~1.2~1.28×。

---

## 六、小结

- **per-block fold 的那 23% 是「折算时张量核空转」，不是 FLOPs 不够**。ncu 证据：tensor pipe
  **49.8% → 67.9%**、stall `barrier` **1.09 → 0.18**。
- **ping-pong + `setmaxnreg` 两条直路都撞墙**：用 wgmma 自身的 `scale_d=0` 可以消掉 ptxas `C7514`，
  但 288 线程下双累加器的寄存器需求让 ptxas 忽略 `setmaxnreg`（`C7507`），实测 348 TFLOPS。
- **第三条路是把 scale 折进操作数**：`ue8m0` 是 2 的幂 → `A_fp8 × sa` 在 e4m3 里精确 →
  per-block GEMM 退化成 per-tensor GEMM。GEMM 本体 **940 → 1208 TFLOPS（1.28×）**。
- **折 A 融进上游量化器即零成本**：量化器本来就在寄存器里拿着 `sa`，多乘一次即可。端到端
  **1.19×**，且与 fold **数值等价（差 0.15%）**。
- 代价：只对 2 的幂 scale 精确；`A'` 与 scale 绑定；独立 prescale pass 时收益降到 ~1.11×。

代码：`code/kernel-opt/37-fp8-pb-prescale/`（`fp8_gemm_prescale.cu` 是主实验，`fp8_gemm_pb2.cu` 是
`scale_d=0` 清零实验，`fp8_gemm_ws.cu` 是 `setmaxnreg` + ping-pong 负结果）。原始输出：
`prescale_M4096.out.txt`、`prescale_M16384.out.txt`、`overlap_negative.out.txt`、`scale_d_zeroing.out.txt`、
`ncu_fold.out.txt`、`ncu_pt.out.txt`。

复现：

```bash
cd code/kernel-opt
ARCH="" NVCC_FLAGS="-gencode=arch=compute_90a,code=sm_90a -lcuda" \
  scripts/run.sh 37-fp8-pb-prescale/fp8_gemm_prescale.cu 4096 3072 7168 all
```

## 下一篇预告

per-block 的公共瓶颈用「折进操作数」绕过了，但还有两块模型场景算子没做：**Paged KV-cache /
flash-decoding**（主题 25）和 **W4A16 dequant-GEMM**（主题 26）。下一篇优先做 Paged KV-cache
+GQA 解码注意力，对标 FlashInfer / vLLM，把推理侧的变长 KV 访存也纳入性能榜。
